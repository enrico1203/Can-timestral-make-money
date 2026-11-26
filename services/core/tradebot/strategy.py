"""Strategies: turn a quantile forecast into a target position.

All strategies receive, at the CLOSE of bar t, the close price and the forecast made with data up
to bar t (log-price quantiles, shape (H, Q)). They return a Decision with a target position as a
fraction of equity in [-max_frac, +max_frac], optional stop / take-profit prices and diagnostics.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .config import QUANTILE_LEVELS

LEVELS = np.array(QUANTILE_LEVELS)
I05, I10, I20, I50, I80, I90, I95 = (QUANTILE_LEVELS.index(x) for x in (0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95))


def prob_above(q: np.ndarray, threshold: float) -> float:
    """P(X > threshold) from the 21 quantiles by linear interpolation of the CDF.

    Beyond the outer quantiles the tail mass is assumed to be the outer level (conservative).
    """
    q = np.maximum.accumulate(np.asarray(q, dtype=np.float64))  # guard against tiny crossings
    if threshold <= q[0]:
        return float(1 - LEVELS[0])
    if threshold >= q[-1]:
        return float(1 - LEVELS[-1])
    return float(1 - np.interp(threshold, q, LEVELS))


@dataclass
class Decision:
    target: float                    # fraction of equity, signed
    action: str                      # LONG | SHORT | FLAT | HOLD
    reason: str
    stop: float | None = None        # price
    take_profit: float | None = None
    max_bars: int | None = None      # time exit
    p_up: float | None = None
    p_down: float | None = None
    exp_ret: float | None = None
    width: float | None = None


@dataclass
class Bar:
    symbol: str
    ts: object
    open: float
    high: float
    low: float
    close: float
    q: np.ndarray | None             # (H, Q) log-price quantiles, or None if no forecast


@dataclass
class PosView:
    frac: float                      # current signed exposure as fraction of equity
    bars_held: int
    side: int                        # -1, 0, +1


class Strategy:
    name = "base"
    uses_forecast = True

    def __init__(self, cost_per_side: float, max_frac: float, **params):
        self.cost = cost_per_side
        self.max_frac = max_frac
        self.p = params
        self.state: dict[str, dict] = {}

    def st(self, sym: str) -> dict:
        return self.state.setdefault(sym, {})

    def observe(self, bar: Bar) -> None:
        """Update internal state without trading (warm-up)."""
        self.decide(bar, PosView(0.0, 0, 0))

    def decide(self, bar: Bar, pos: PosView) -> Decision:  # pragma: no cover
        raise NotImplementedError

    # state persistence for the live trader
    def dump_state(self) -> dict:
        out = {}
        for sym, s in self.state.items():
            out[sym] = {k: (list(v) if isinstance(v, deque) else v) for k, v in s.items()}
        return out

    def load_state(self, data: dict | None) -> None:
        if not data:
            return
        for sym, s in data.items():
            st = self.st(sym)
            for k, v in s.items():
                if k == "widths":
                    st[k] = deque(v, maxlen=int(self.p.get("window", 720)))
                elif k == "closes":
                    st[k] = deque(v, maxlen=int(self.p.get("slow", 50)))
                else:
                    st[k] = v


def _metrics(q: np.ndarray, last: float, h: int, cost: float):
    qh = q[min(h, len(q)) - 1]
    rt = 2 * cost
    p_up = prob_above(qh, last + rt)
    p_down = 1 - prob_above(qh, last - rt)
    med = float(qh[I50] - last)
    width = float(qh[I90] - qh[I10])
    return qh, p_up, p_down, med, width


class Directional(Strategy):
    """S1 / S2. Go long when P(return > round-trip cost) at horizon h is high, short symmetric.

    params: h, p_enter, p_exit (None -> time exit after h bars, else hysteresis exit),
            k_cost (median must exceed k_cost * round-trip cost), allow_short, risk_per_trade, use_stops
    """

    name = "directional"

    def decide(self, bar: Bar, pos: PosView) -> Decision:
        p = self.p
        h = int(p.get("h", 4))
        p_enter = float(p.get("p_enter", 0.6))
        p_exit = p.get("p_exit")
        k_cost = float(p.get("k_cost", 1.0))
        allow_short = bool(p.get("allow_short", True))
        risk = float(p.get("risk_per_trade", 0.01))
        use_stops = bool(p.get("use_stops", True))
        if bar.q is None:
            return Decision(pos.frac, "HOLD", "no forecast")
        last = float(np.log(bar.close))
        qh, p_up, p_down, med, width = _metrics(bar.q, last, h, self.cost)
        rt = 2 * self.cost
        base = dict(p_up=p_up, p_down=p_down, exp_ret=med, width=width)
        long_sig = p_up > p_enter and med > k_cost * rt
        short_sig = allow_short and p_down > p_enter and -med > k_cost * rt

        def size(stop_dist: float) -> float:
            if not use_stops or stop_dist <= 0:
                return self.max_frac
            return float(min(self.max_frac, risk / stop_dist))

        def enter(side: int, why: str) -> Decision:
            # stop/take-profit from the forecast distribution, but always on the correct side of the
            # entry and at least a couple of round-trip costs away
            min_dist = max(2 * rt, 0.25 * width)
            if side > 0:
                stop_l = min(float(qh[I05]), last - min_dist)
                tp_l = max(float(qh[I80]), last + min_dist)
                dist = last - stop_l
            else:
                stop_l = max(float(qh[I95]), last + min_dist)
                tp_l = min(float(qh[I20]), last - min_dist)
                dist = stop_l - last
            return Decision(side * size(dist), "LONG" if side > 0 else "SHORT", why,
                            stop=float(np.exp(stop_l)) if use_stops else None,
                            take_profit=float(np.exp(tp_l)) if use_stops else None,
                            max_bars=None if p_exit is not None else h, **base)

        if pos.side == 0:
            if long_sig:
                return enter(+1, f"p_up={p_up:.2f}>{p_enter} med={med:+.4f}")
            if short_sig:
                return enter(-1, f"p_down={p_down:.2f}>{p_enter} med={med:+.4f}")
            return Decision(0.0, "FLAT", f"no edge p_up={p_up:.2f} p_down={p_down:.2f}", **base)

        # in a position
        if pos.side > 0 and short_sig:
            return enter(-1, f"flip to short p_down={p_down:.2f}")
        if pos.side < 0 and long_sig:
            return enter(+1, f"flip to long p_up={p_up:.2f}")
        if p_exit is not None:
            p_in = p_up if pos.side > 0 else p_down
            if p_in < float(p_exit):
                return Decision(0.0, "FLAT", f"exit: p={p_in:.2f}<{p_exit}", **base)
            return Decision(pos.frac, "HOLD", f"hold p={p_in:.2f}", **base)
        # time-based: if signal persists keep the position (time exit refreshed), else close at expiry
        same = long_sig if pos.side > 0 else short_sig
        if same:
            d = enter(pos.side, "signal persists")
            d.target = pos.frac  # keep current size
            d.action = "HOLD"
            return d
        if pos.bars_held >= h:
            return Decision(0.0, "FLAT", f"time exit after {pos.bars_held} bars", **base)
        return Decision(pos.frac, "HOLD", "holding until horizon", **base)


class VolFilter(Strategy):
    """S3: long-only, stay invested unless the forecast uncertainty is in the top tail of its own history."""

    name = "vol_filter"

    def decide(self, bar: Bar, pos: PosView) -> Decision:
        h = int(self.p.get("h", 24))
        pct = float(self.p.get("pct", 80))
        window = int(self.p.get("window", 720))
        min_hist = int(self.p.get("min_hist", 168))
        if bar.q is None:
            return Decision(pos.frac, "HOLD", "no forecast")
        last = float(np.log(bar.close))
        _, p_up, p_down, med, width = _metrics(bar.q, last, h, self.cost)
        s = self.st(bar.symbol)
        hist = s.setdefault("widths", deque(maxlen=window))
        thr = float(np.percentile(hist, pct)) if len(hist) >= min_hist else np.inf
        hist.append(width)
        base = dict(p_up=p_up, p_down=p_down, exp_ret=med, width=width)
        if width > thr:
            return Decision(0.0, "FLAT", f"high predicted vol {width:.4f}>{thr:.4f}", **base)
        return Decision(self.max_frac, "LONG", f"vol ok {width:.4f}<= p{pct:.0f}", **base)


class VolTarget(Strategy):
    """S4: long-only, exposure inversely proportional to predicted uncertainty."""

    name = "vol_target"

    def decide(self, bar: Bar, pos: PosView) -> Decision:
        h = int(self.p.get("h", 24))
        window = int(self.p.get("window", 720))
        if bar.q is None:
            return Decision(pos.frac, "HOLD", "no forecast")
        last = float(np.log(bar.close))
        _, p_up, p_down, med, width = _metrics(bar.q, last, h, self.cost)
        s = self.st(bar.symbol)
        hist = s.setdefault("widths", deque(maxlen=window))
        hist.append(width)
        ref = float(np.median(hist))
        frac = float(np.clip(self.max_frac * ref / max(width, 1e-9), 0.0, self.max_frac))
        return Decision(frac, "LONG", f"vol target width={width:.4f} ref={ref:.4f}",
                        p_up=p_up, p_down=p_down, exp_ret=med, width=width)


class MeanReversion(Strategy):
    """S5: price closed outside the extreme quantile predicted one bar earlier -> bet on reversion."""

    name = "mean_reversion"

    def decide(self, bar: Bar, pos: PosView) -> Decision:
        lvl = float(self.p.get("level", 0.05))
        hold = int(self.p.get("max_bars", 12))
        allow_short = bool(self.p.get("allow_short", False))
        lo_i, hi_i = QUANTILE_LEVELS.index(lvl), QUANTILE_LEVELS.index(round(1 - lvl, 2))
        s = self.st(bar.symbol)
        prev = s.get("prev")  # [q_lo, q_med, q_hi] for this bar, predicted at previous bar
        if bar.q is not None:
            s["prev"] = [float(bar.q[0, lo_i]), float(bar.q[0, I50]), float(bar.q[0, hi_i])]
        else:
            s["prev"] = None
        last = float(np.log(bar.close))
        if prev is None:
            return Decision(pos.frac, "HOLD", "no previous forecast")
        lo, med, hi = prev
        if pos.side > 0 and (last >= med or pos.bars_held >= hold):
            return Decision(0.0, "FLAT", "reverted" if last >= med else "time exit")
        if pos.side < 0 and (last <= med or pos.bars_held >= hold):
            return Decision(0.0, "FLAT", "reverted" if last <= med else "time exit")
        if pos.side == 0 and last < lo:
            return Decision(self.max_frac, "LONG", f"close below predicted q{lvl:.2f}", max_bars=hold)
        if pos.side == 0 and allow_short and last > hi:
            return Decision(-self.max_frac, "SHORT", f"close above predicted q{1-lvl:.2f}", max_bars=hold)
        return Decision(pos.frac, "HOLD" if pos.side else "FLAT", "no signal")


class BuyHold(Strategy):
    name = "buy_hold"
    uses_forecast = False

    def decide(self, bar: Bar, pos: PosView) -> Decision:
        return Decision(self.max_frac, "LONG" if pos.side == 0 else "HOLD", "buy and hold")


class SmaCross(Strategy):
    name = "sma_cross"
    uses_forecast = False

    def decide(self, bar: Bar, pos: PosView) -> Decision:
        fast, slow = int(self.p.get("fast", 20)), int(self.p.get("slow", 50))
        allow_short = bool(self.p.get("allow_short", False))
        s = self.st(bar.symbol)
        closes = s.setdefault("closes", deque(maxlen=slow))
        closes.append(float(bar.close))
        if len(closes) < slow:
            return Decision(0.0, "FLAT", "warming up")
        c = np.array(closes)
        f, sl = c[-fast:].mean(), c.mean()
        if f > sl:
            return Decision(self.max_frac, "LONG" if pos.side <= 0 else "HOLD", f"sma{fast}>sma{slow}")
        if allow_short:
            return Decision(-self.max_frac, "SHORT" if pos.side >= 0 else "HOLD", f"sma{fast}<sma{slow}")
        return Decision(0.0, "FLAT", f"sma{fast}<sma{slow}")


STRATEGIES = {c.name: c for c in (Directional, VolFilter, VolTarget, MeanReversion, BuyHold, SmaCross)}


def make_strategy(name: str, cost_per_side: float, max_frac: float, params: dict | None = None) -> Strategy:
    if name not in STRATEGIES:
        raise ValueError(f"unknown strategy {name}; available: {sorted(STRATEGIES)}")
    return STRATEGIES[name](cost_per_side, max_frac, **(params or {}))
