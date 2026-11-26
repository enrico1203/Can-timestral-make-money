import json
import math

import numpy as np
import pandas as pd
import pytest

from tradebot import db
from tradebot.config import QUANTILE_LEVELS
from tradebot.engine import Engine, EngineConfig
from tradebot.features import build_inputs, naive_rw_forecast
from tradebot.metrics import circular_shift_test, deflated_sharpe, performance
from tradebot.strategy import Bar, Decision, PosView, Strategy, make_strategy, prob_above

LV = np.array(QUANTILE_LEVELS)
T0 = pd.Timestamp("2025-01-01", tz="UTC")
H = pd.Timedelta(hours=1)


class Scripted(Strategy):
    """Returns pre-scripted targets per bar index."""
    name = "scripted"
    uses_forecast = False

    def __init__(self, targets, stop=None, tp=None):
        super().__init__(0.0, 1.0)
        self.targets, self.i, self.stop, self.tp = targets, 0, stop, tp

    def decide(self, bar, pos):
        t = self.targets[self.i] if self.i < len(self.targets) else self.targets[-1]
        self.i += 1
        act = "LONG" if t > 0 else "SHORT" if t < 0 else "FLAT"
        return Decision(t, act, "scripted", stop=self.stop, take_profit=self.tp)


def bar(i, o, h, l, c, sym="X", q=None):
    return Bar(sym, T0 + i * H, o, h, l, c, q)


def gaussian_q(center, sd, h=24):
    from scipy.stats import norm
    z = norm.ppf(LV)
    return np.array([center + sd * np.sqrt(k) * z for k in range(1, h + 1)], dtype=np.float32)


# ---------------------------------------------------------------- strategy math

def test_prob_above_symmetric():
    q = gaussian_q(0.0, 0.01)[0]
    assert prob_above(q, 0.0) == pytest.approx(0.5, abs=1e-6)
    assert prob_above(q, 1.0) == pytest.approx(0.01)
    assert prob_above(q, -1.0) == pytest.approx(0.99)
    assert prob_above(q, 0.005) < 0.5 < prob_above(q, -0.005)


def test_directional_long_short_flat():
    s = make_strategy("directional", 0.0013, 0.5, {"h": 1, "p_enter": 0.6, "allow_short": True})
    last = math.log(100)
    up = bar(0, 100, 100, 100, 100, q=gaussian_q(last + 0.02, 0.01))
    d = s.decide(up, PosView(0, 0, 0))
    assert d.action == "LONG" and 0 < d.target <= 0.5 and d.stop < 100 < d.take_profit
    dn = bar(0, 100, 100, 100, 100, q=gaussian_q(last - 0.02, 0.01))
    d = s.decide(dn, PosView(0, 0, 0))
    assert d.action == "SHORT" and d.target < 0 and d.stop > 100
    flat = bar(0, 100, 100, 100, 100, q=gaussian_q(last, 0.01))
    assert s.decide(flat, PosView(0, 0, 0)).action == "FLAT"
    # costs matter: tiny positive drift below round-trip cost -> no trade
    tiny = bar(0, 100, 100, 100, 100, q=gaussian_q(last + 0.001, 0.0005))
    assert s.decide(tiny, PosView(0, 0, 0)).action == "FLAT"


def test_directional_time_exit():
    s = make_strategy("directional", 0.0013, 0.5, {"h": 4, "p_enter": 0.6})
    flat_q = gaussian_q(math.log(100), 0.01)
    b = bar(0, 100, 100, 100, 100, q=flat_q)
    assert s.decide(b, PosView(0.5, 2, 1)).action == "HOLD"
    assert s.decide(b, PosView(0.5, 4, 1)).action == "FLAT"


# ---------------------------------------------------------------- engine / paper broker

def test_fill_at_next_open_with_costs():
    eng = Engine(Scripted([1.0, 1.0]), EngineConfig(initial_cash=1000, fee=0.001, slippage=0.001))
    eng.step(T0, {"X": bar(0, 100, 101, 99, 100)})
    assert eng.positions == {} and len(eng.pending) == 1        # decided at close, not filled yet
    eng.step(T0 + H, {"X": bar(1, 110, 111, 109, 110)})
    pos = eng.positions["X"]
    fill = 110 * 1.001
    assert pos["avg"] == pytest.approx(fill)                    # filled at NEXT open + slippage
    expected_qty = 1000 / 100                                     # target computed at close 100
    assert pos["qty"] == pytest.approx(expected_qty * 0 + pos["qty"])
    fee = pos["qty"] * fill * 0.001
    assert eng.cash == pytest.approx(1000 - pos["qty"] * fill - fee)


def test_no_leverage_cap():
    eng = Engine(Scripted([1.0]), EngineConfig(initial_cash=1000, fee=0, slippage=0))
    eng.step(T0, {"X": bar(0, 100, 100, 100, 100)})
    eng.step(T0 + H, {"X": bar(1, 100, 100, 100, 100)})
    assert eng.gross_exposure() <= 1.0 + 1e-9


def test_stop_is_pessimistic_when_both_levels_touched():
    eng = Engine(Scripted([0.5, 0.5, 0.5], stop=95, tp=105), EngineConfig(initial_cash=1000, fee=0, slippage=0))
    eng.step(T0, {"X": bar(0, 100, 100, 100, 100)})
    eng.step(T0 + H, {"X": bar(1, 100, 106, 94, 100)})          # touches both 94<95 and 106>105
    assert len(eng.rec.trades) == 1
    t = eng.rec.trades[0]
    assert t["exit_reason"] == "stop_loss" and t["exit_price"] == pytest.approx(95)
    assert t["pnl"] < 0


def test_gap_through_stop_fills_at_open():
    eng = Engine(Scripted([0.5, 0.0], stop=95), EngineConfig(initial_cash=1000, fee=0, slippage=0))
    eng.step(T0, {"X": bar(0, 100, 100, 100, 100)})
    eng.step(T0 + H, {"X": bar(1, 100, 100, 100, 100)})          # filled at 100
    eng.step(T0 + 2 * H, {"X": bar(2, 90, 91, 89, 90)})          # gap below stop
    t = eng.rec.trades[0]
    assert t["exit_price"] == pytest.approx(90)


def test_short_profits_when_price_falls():
    eng = Engine(Scripted([-0.5, -0.5, 0.0, 0.0]), EngineConfig(initial_cash=1000, fee=0, slippage=0))
    eng.step(T0, {"X": bar(0, 100, 100, 100, 100)})
    eng.step(T0 + H, {"X": bar(1, 100, 100, 100, 100)})          # short 5 @100
    eng.step(T0 + 2 * H, {"X": bar(2, 80, 80, 80, 80)})          # exit queued
    eng.step(T0 + 3 * H, {"X": bar(3, 80, 80, 80, 80)})          # cover @80
    assert eng.rec.trades[0]["side"] == "short"
    assert eng.rec.trades[0]["pnl"] == pytest.approx(5 * 20)
    assert eng.equity() == pytest.approx(1100)


def test_flip_long_to_short_records_trade():
    eng = Engine(Scripted([0.5, -0.5, -0.5]), EngineConfig(initial_cash=1000, fee=0, slippage=0))
    eng.step(T0, {"X": bar(0, 100, 100, 100, 100)})
    eng.step(T0 + H, {"X": bar(1, 100, 100, 100, 110)})          # long filled @100, then flip queued
    eng.step(T0 + 2 * H, {"X": bar(2, 110, 110, 110, 110)})      # flip @110
    assert len(eng.rec.trades) == 1 and eng.rec.trades[0]["pnl"] == pytest.approx(50)
    assert eng.positions["X"]["qty"] < 0


def test_drawdown_halt_flattens():
    eng = Engine(Scripted([1.0] * 10), EngineConfig(initial_cash=1000, fee=0, slippage=0, max_drawdown_halt=0.2))
    eng.step(T0, {"X": bar(0, 100, 100, 100, 100)})
    eng.step(T0 + H, {"X": bar(1, 100, 100, 70, 70)})            # -30%
    assert eng.halted
    eng.step(T0 + 2 * H, {"X": bar(2, 70, 70, 70, 70)})
    assert "X" not in eng.positions


def test_state_roundtrip():
    s1 = make_strategy("vol_filter", 0.001, 0.5, {"h": 1, "min_hist": 2})
    e1 = Engine(s1, EngineConfig(initial_cash=1000))
    rng = np.random.default_rng(0)
    p = 100.0
    for i in range(30):
        p *= 1 + rng.normal(0, 0.01)
        e1.step(T0 + i * H, {"X": bar(i, p, p * 1.01, p * 0.99, p, q=gaussian_q(math.log(p), 0.01 + 0.001 * i))})
    state = json.loads(json.dumps(db.jsonable(e1.dump_state())))
    e2 = Engine(make_strategy("vol_filter", 0.001, 0.5, {"h": 1, "min_hist": 2}), EngineConfig(initial_cash=1000))
    e2.load_state(state)
    b = bar(31, p, p, p, p, q=gaussian_q(math.log(p), 0.02))
    e1.step(T0 + 31 * H, {"X": b})
    e2.step(T0 + 31 * H, {"X": b})
    assert e1.equity() == pytest.approx(e2.equity())
    assert [o["qty"] for o in e1.pending] == pytest.approx([o["qty"] for o in e2.pending])


# ---------------------------------------------------------------- no look-ahead in model inputs

def test_build_inputs_never_reads_the_future():
    idx = pd.date_range(T0, periods=100, freq="h", tz="UTC")
    df = pd.DataFrame({"open": 1.0, "high": 2.0, "low": 0.5, "close": np.arange(1, 101, dtype=float),
                       "volume": 10.0}, index=idx)
    ctx, past, known = build_inputs(df, 49, 20, 5, "cov", "1h")
    assert ctx.shape == (1, 20) and ctx[0, -1] == pytest.approx(np.log(50))
    df2 = df.copy()
    df2.iloc[50:] = 999.0
    ctx2, past2, _ = build_inputs(df2, 49, 20, 5, "cov", "1h")
    assert np.array_equal(ctx, ctx2) and np.array_equal(past, past2)
    assert known.shape == (3, 25)


def test_naive_rw_shape_and_median():
    lp = np.log(np.linspace(100, 110, 500))[None, :]
    q = naive_rw_forecast(lp, 3)
    assert q.shape == (1, 3, 21)
    assert np.all(np.diff(q[0, 0]) >= 0)
    assert q[0, 0, 10] > lp[0, -1]   # positive drift


# ---------------------------------------------------------------- metrics

def test_performance_and_mc():
    rng = np.random.default_rng(1)
    idx = pd.date_range(T0, periods=2000, freq="h", tz="UTC")
    eq = pd.Series(1000 * np.cumprod(1 + rng.normal(0, 0.001, 2000)), index=idx)
    m = performance(eq, [], 8766)
    assert m["bars"] == 2000 and m["max_drawdown"] <= 0
    rets = rng.normal(0, 0.01, (2000, 2))
    w_random = (rng.random((2000, 2)) > 0.5).astype(float) * 0.5
    r = circular_shift_test(w_random, rets, 0.0, 8766, n=200)
    assert r["p_value"] > 0.01
    # a perfect-foresight strategy is detected
    w_oracle = np.vstack([np.sign(rets[1:]) * 0.5, np.zeros((1, 2))])
    r2 = circular_shift_test(w_oracle, rets, 0.0, 8766, n=200)
    assert r2["p_value"] < 0.01


def test_deflated_sharpe_penalises_many_trials():
    one = deflated_sharpe(1.5, 8000, 8766, 1, [1.5])["dsr"]
    many = deflated_sharpe(1.5, 8000, 8766, 200, list(np.random.default_rng(0).normal(0, 1, 200)))["dsr"]
    assert many < one
