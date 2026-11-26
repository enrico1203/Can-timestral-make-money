"""Backtests: single runs and the full research suite (tune -> validation -> holdout)."""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from psycopg.types.json import Jsonb

from .. import config, db
from ..engine import Engine, EngineConfig
from ..metrics import bars_per_year, circular_shift_test, deflated_sharpe, performance, sharpe
from ..strategy import Bar, make_strategy

log = logging.getLogger("backtest")


def ts(x) -> pd.Timestamp:
    t = pd.Timestamp(x)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


@dataclass
class Market:
    symbols: list[str]
    timeline: pd.DatetimeIndex
    o: np.ndarray       # (T, S)
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    fq: dict            # (model, fs, ctx) -> list per symbol of (pos array (T,), quantiles (N,H,Q))


class DataCache:
    """Loads candles and forecasts once and serves many backtests over sub-periods."""

    def __init__(self, conn, symbols, start, end, timeframe=None, warmup_bars=800):
        self.conn = conn
        self.symbols = list(symbols)
        self.tf = timeframe or config.TIMEFRAME
        step = pd.Timedelta(seconds=config.TIMEFRAME_SECONDS[self.tf])
        self.start, self.end = ts(start), (ts(end) if end else None)
        load_from = self.start - warmup_bars * step
        grids = {}
        for s in self.symbols:
            g = db.regular_grid(db.load_candles(conn, s, self.tf, load_from, self.end), self.tf)
            grids[s] = g
        lo = min(g.index[0] for g in grids.values() if len(g))
        hi = max(g.index[-1] for g in grids.values() if len(g))
        tl = pd.date_range(lo, hi, freq=step, tz="UTC")
        S = len(self.symbols)
        arr = {k: np.full((len(tl), S), np.nan) for k in ("open", "high", "low", "close")}
        for j, s in enumerate(self.symbols):
            g = grids[s].reindex(tl)
            for k in arr:
                arr[k][:, j] = g[k].to_numpy()
        self.m = Market(self.symbols, tl, arr["open"], arr["high"], arr["low"], arr["close"], {})
        self.load_from = load_from

    def forecasts(self, model, fs, ctx):
        key = (model, fs, ctx)
        if key not in self.m.fq:
            per = []
            for s in self.symbols:
                idx, Q, _ = db.load_forecasts(self.conn, model, s, self.tf, fs, ctx, self.load_from, self.end)
                pos = np.full(len(self.m.timeline), -1, dtype=np.int64)
                if len(idx):
                    loc = self.m.timeline.get_indexer(idx)
                    pos[loc[loc >= 0]] = np.flatnonzero(loc >= 0)
                per.append((pos, Q))
            self.m.fq[key] = per
        return self.m.fq[key]

    def coverage(self, model, fs, ctx, start, end) -> float:
        per = self.forecasts(model, fs, ctx)
        sel = (self.m.timeline >= ts(start)) & ((self.m.timeline < ts(end)) if end else True)
        tot = ok = 0
        for j, (pos, _) in enumerate(per):
            has_bar = np.isfinite(self.m.c[sel, j])
            tot += has_bar.sum()
            ok += (pos[sel][has_bar] >= 0).sum()
        return ok / max(tot, 1)


def run_one(cache: DataCache, spec: dict, mc: int = 0, keep_series: bool = False) -> dict:
    """spec keys: model, feature_set, context_len, strategy, params, start, end, fee, slippage,
    initial_cash, max_frac, rebalance_band, daily_loss_limit, max_drawdown_halt"""
    m = cache.m
    S = len(m.symbols)
    fee, slip = spec.get("fee", 0.001), spec.get("slippage", 0.0003)
    max_frac = spec.get("max_frac") or 1.0 / S
    # the strategy's cost hurdle can be fixed independently of the fees actually charged, so that a
    # "gross" run keeps exactly the same decisions and only removes the costs from the P&L
    strat = make_strategy(spec["strategy"], spec.get("strategy_cost", fee + slip), max_frac, spec.get("params"))
    eng = Engine(strat, EngineConfig(initial_cash=spec.get("initial_cash", 10_000.0), fee=fee, slippage=slip,
                                     rebalance_band=spec.get("rebalance_band", 0.05),
                                     daily_loss_limit=spec.get("daily_loss_limit"),
                                     max_drawdown_halt=spec.get("max_drawdown_halt")),
                 record_all_signals=False)
    per = cache.forecasts(spec["model"], spec.get("feature_set", "close"), spec.get("context_len", 1024)) \
        if strat.uses_forecast else [(np.full(len(m.timeline), -1), None)] * S
    start, end = ts(spec["start"]), (ts(spec["end"]) if spec.get("end") else None)
    tl = m.timeline
    i0 = int(np.searchsorted(tl, start))
    i1 = int(np.searchsorted(tl, end)) if end is not None else len(tl)
    warm = max(0, i0 - int(spec.get("warmup_bars", 720)))

    def bar(i, j):
        if not np.isfinite(m.c[i, j]):
            return None
        pos, Q = per[j]
        q = Q[pos[i]] if Q is not None and pos[i] >= 0 else None
        return Bar(m.symbols[j], tl[i], m.o[i, j], m.h[i, j], m.l[i, j], m.c[i, j], q)

    for i in range(warm, i0):  # strategy warm-up, no trading
        for j in range(S):
            b = bar(i, j)
            if b is not None:
                strat.observe(b)
    W = np.zeros((i1 - i0, S))
    for k, i in enumerate(range(i0, i1)):
        eng.step(tl[i], {m.symbols[j]: bar(i, j) for j in range(S)})
        eq = eng.last_eq
        for j, s in enumerate(m.symbols):
            p = eng.positions.get(s)
            if p and eq > 0:
                W[k, j] = p["qty"] * eng.last_close.get(s, p["avg"]) / eq
    rec = eng.rec
    eq_df = pd.DataFrame(rec.equity).set_index("ts")
    ppy = bars_per_year(config.TIMEFRAME_SECONDS[cache.tf])
    perf = performance(eq_df["equity"], rec.trades, ppy, eng.fees_paid, eq_df["exposure"])
    # equal-weight buy & hold benchmark over the same bars
    c = m.c[i0:i1]
    rets = np.vstack([np.zeros((1, S)), np.diff(c, axis=0) / c[:-1]])
    rets = np.nan_to_num(rets)
    bh = np.cumprod(1 + rets.mean(1))
    perf["bh_total_return"] = float(bh[-1] - 1)
    perf["bh_sharpe"] = sharpe(np.diff(bh) / bh[:-1], ppy)
    perf["bh_max_drawdown"] = float((bh / np.maximum.accumulate(bh) - 1).min())
    perf["excess_return_vs_bh"] = perf["total_return"] - perf["bh_total_return"]
    perf["halted"] = eng.halt_reason
    if mc:
        perf["mc"] = circular_shift_test(W, rets, fee + slip, ppy, n=mc)
    out = {"spec": spec, "metrics": perf}
    if keep_series:
        out.update(engine=eng, equity=eq_df, weights=W, records=rec)
    return out


def save_run(conn, result: dict, name: str, group_id: str | None = None) -> int:
    spec, perf = result["spec"], result["metrics"]
    eng, rec, eq_df = result["engine"], result["records"], result["equity"]
    rid = conn.execute(
        "INSERT INTO runs(kind,name,status,finished_at,config,metrics,group_id) VALUES "
        "('backtest',%s,'done',now(),%s,%s,%s) RETURNING id",
        (name, Jsonb(db.jsonable(spec)), Jsonb(db.jsonable(perf)), group_id)).fetchone()["id"]
    with conn.cursor() as cur:
        with cur.copy("COPY equity(run_id,ts,cash,positions_value,equity,drawdown,exposure) FROM STDIN") as cp:
            for t, r in eq_df.iterrows():
                cp.write_row((rid, t, r["cash"], r["positions_value"], r["equity"], r["drawdown"], r["exposure"]))
        with cur.copy("COPY trades(run_id,symbol,side,qty,entry_ts,entry_price,exit_ts,exit_price,pnl,fees,bars,exit_reason) FROM STDIN") as cp:
            for t in rec.trades:
                cp.write_row((rid, t["symbol"], t["side"], t["qty"], t["entry_ts"], t["entry_price"], t["exit_ts"],
                              t["exit_price"], t["pnl"], t["fees"], t["bars"], t["exit_reason"]))
        with cur.copy("COPY orders(run_id,local_id,symbol,created_ts,side,qty,reason,status,fill_ts,fill_price,fee,slippage) FROM STDIN") as cp:
            created = {o["id"]: o["created_ts"] for o in rec.orders if o["status"] == "pending"}
            for o in rec.orders:
                if o["status"] != "filled":
                    continue
                cp.write_row((rid, o["id"], o["symbol"], created.get(o["id"], o["fill_ts"]),
                              "buy" if o["qty"] > 0 else "sell", abs(o["qty"]), o["reason"], "filled",
                              o["fill_ts"], o["fill_price"], o["fee"], o["slippage"]))
        with cur.copy("COPY signals(run_id,symbol,ts,p_up,p_down,exp_ret,width,target,action,reason) FROM STDIN") as cp:
            for s in rec.signals:
                cp.write_row((rid, s["symbol"], s["ts"], s["p_up"], s["p_down"], s["exp_ret"], s["width"],
                              s["target"], s["action"], s["reason"]))
    return rid


# ------------------------------------------------------------------ research suite

def strategy_grid(quick: bool = False) -> list[tuple[str, dict]]:
    g: list[tuple[str, dict]] = []
    hs = (1, 4, 24) if quick else (1, 4, 12, 24)
    for h, pe, kc, pex, short in itertools.product(hs, (0.52, 0.55, 0.6, 0.65), (0.0, 1.0), (None, 0.5), (True, False)):
        if pe == 0.52 and kc == 1.0:
            continue
        g.append(("directional", {"h": h, "p_enter": pe, "p_exit": pex, "k_cost": kc, "allow_short": short,
                                  "risk_per_trade": 0.01, "use_stops": True}))
    for h, pct in itertools.product((1, 24), (70, 80, 90)):
        g.append(("vol_filter", {"h": h, "pct": pct}))
    for h in (1, 24):
        g.append(("vol_target", {"h": h}))
    for lvl, mb, short in itertools.product((0.05, 0.1), (6, 24), (False, True)):
        g.append(("mean_reversion", {"level": lvl, "max_bars": mb, "allow_short": short}))
    return g


MIN_TUNE_TRADES = 30
BASELINES = [("buy_hold", {}), ("sma_cross", {"fast": 20, "slow": 50}),
             ("sma_cross", {"fast": 20, "slow": 50, "allow_short": True})]


def suite(conn, models: list[str], symbols: list[str], tune: tuple, valid: tuple, holdout: tuple,
          feature_set="close", context_len=1024, mc=1000, progress=None, quick=False, save_runs=True) -> dict:
    t0 = time.time()
    group = f"suite-{int(t0)}"
    cache = DataCache(conn, symbols, tune[0], holdout[1])
    ppy = bars_per_year(config.TIMEFRAME_SECONDS[cache.tf])
    common = {"symbols": symbols, "feature_set": feature_set, "context_len": context_len, "fee": 0.001,
              "slippage": 0.0003, "initial_cash": 10_000.0}
    report = {"group": group, "models": models, "symbols": symbols, "periods": {"tune": tune, "validation": valid,
              "holdout": holdout}, "coverage": {}, "tune": [], "validation": [], "holdout": [], "baselines": {}}
    usable = []
    for mdl in models:
        cov = {p: cache.coverage(mdl, feature_set, context_len, *rng) for p, rng in
               (("tune", tune), ("validation", valid), ("holdout", holdout))}
        report["coverage"][mdl] = cov
        if min(cov.values()) < 0.95:
            log.warning("model %s: forecast coverage %s < 95%%, skipped", mdl, cov)
            continue
        usable.append(mdl)
    grid = strategy_grid(quick)
    total = len(usable) * len(grid) + len(BASELINES) * 3 + 50
    done = 0

    def tick(msg):
        nonlocal done
        done += 1
        if progress:
            progress(min(done / total, 0.99), msg)

    # 1) tune: every model x every config
    for mdl in usable:
        for name, params in grid:
            spec = {**common, "model": mdl, "strategy": name, "params": params, "start": tune[0], "end": tune[1]}
            r = run_one(cache, spec)
            report["tune"].append({"model": mdl, "strategy": name, "params": params, **_short(r["metrics"])})
            tick(f"tune {mdl.split('/')[-1]} {name}")
    # 2) for each (model, family): best tune config by Sharpe -> validation
    best = {}
    for row in report["tune"]:
        key = (row["model"], row["strategy"])
        # a Sharpe made of a handful of trades is noise: require a minimum sample to be selectable
        if row["n_trades"] < MIN_TUNE_TRADES and row["strategy"] in ("directional", "mean_reversion", "sma_cross"):
            continue
        if key not in best or row["sharpe"] > best[key]["sharpe"]:
            best[key] = row
    val_sharpes = []
    for (mdl, name), row in sorted(best.items()):
        spec = {**common, "model": mdl, "strategy": name, "params": row["params"], "start": valid[0], "end": valid[1]}
        r = run_one(cache, spec, mc=mc)
        gross = run_one(cache, {**spec, "fee": 0.0, "slippage": 0.0, "strategy_cost": spec["fee"] + spec["slippage"]})["metrics"]
        val_sharpes.append(r["metrics"]["sharpe"])
        report["validation"].append({"model": mdl, "strategy": name, "params": row["params"],
                                     "tune_sharpe": row["sharpe"], **_short(r["metrics"]),
                                     "gross_sharpe": gross["sharpe"], "gross_return": gross["total_return"],
                                     "mc_pvalue": r["metrics"]["mc"].get("p_value"),
                                     "mc_percentile": r["metrics"]["mc"].get("percentile")})
        tick(f"validation {mdl.split('/')[-1]} {name}")
    n_trials = len(report["tune"])
    tune_sharpes = [r["sharpe"] for r in report["tune"]]
    for v in report["validation"]:
        v["dsr"] = deflated_sharpe(v["sharpe"], v["bars"], ppy, n_trials, tune_sharpes)["dsr"]
    # 3) holdout: the validated best of each (model, family), run ONCE, saved with full detail
    for v in report["validation"]:
        spec = {**common, "model": v["model"], "strategy": v["strategy"], "params": v["params"],
                "start": holdout[0], "end": holdout[1]}
        r = run_one(cache, spec, mc=mc, keep_series=save_runs)
        gross = run_one(cache, {**spec, "fee": 0.0, "slippage": 0.0, "strategy_cost": spec["fee"] + spec["slippage"]})["metrics"]
        rid = save_run(conn, r, f"holdout {v['model'].split('/')[-1]} {v['strategy']}", group) if save_runs else None
        report["holdout"].append({"run_id": rid, "model": v["model"], "strategy": v["strategy"], "params": v["params"],
                                  "validation_sharpe": v["sharpe"], **_short(r["metrics"]),
                                  "gross_sharpe": gross["sharpe"], "gross_return": gross["total_return"],
                                  "mc_pvalue": r["metrics"]["mc"].get("p_value"),
                                  "mc_percentile": r["metrics"]["mc"].get("percentile")})
        tick(f"holdout {v['model'].split('/')[-1]} {v['strategy']}")
    # baselines on every period
    for pname, rng in (("tune", tune), ("validation", valid), ("holdout", holdout)):
        rows = []
        for name, params in BASELINES:
            spec = {**common, "model": "-", "strategy": name, "params": params, "start": rng[0], "end": rng[1]}
            r = run_one(cache, spec, keep_series=(pname == "holdout" and save_runs))
            rid = None
            if pname == "holdout" and save_runs:
                rid = save_run(conn, r, f"holdout baseline {name}{' short' if params.get('allow_short') else ''}", group)
            rows.append({"run_id": rid, "strategy": name, "params": params, **_short(r["metrics"])})
            tick(f"baseline {pname} {name}")
        report["baselines"][pname] = rows
    report["verdict"] = verdict(report)
    report["seconds"] = round(time.time() - t0, 1)
    report = db.jsonable(report)
    conn.execute("INSERT INTO reports(kind,name,params,result) VALUES ('suite',%s,%s,%s)",
                 (f"suite {group}", Jsonb(db.jsonable({"models": models, "symbols": symbols, "tune": tune, "validation": valid,
                                                        "holdout": holdout, "feature_set": feature_set,
                                                        "context_len": context_len})), Jsonb(report)))
    if progress:
        progress(1.0, "done")
    return report


def _short(m: dict) -> dict:
    keys = ("total_return", "cagr", "sharpe", "sortino", "max_drawdown", "n_trades", "win_rate", "profit_factor",
            "avg_bars_held", "fees_paid", "time_in_market", "bh_total_return", "bh_sharpe", "excess_return_vs_bh",
            "bars", "mean_ret_pvalue")
    return {k: m.get(k) for k in keys}


def verdict(report: dict) -> dict:
    """Apply the pre-registered success criteria to the holdout results."""
    hold = [h for h in report["holdout"] if not str(h["model"]).startswith("naive")]
    bl = {f"{b['strategy']}{'_short' if b['params'].get('allow_short') else ''}": b for b in report["baselines"].get("holdout", [])}
    bh = bl.get("buy_hold", {})
    sma = bl.get("sma_cross", {})
    naive = {h["strategy"]: h for h in report["holdout"] if str(h["model"]).startswith("naive")}
    vmap = {(v["model"], v["strategy"]): v for v in report["validation"]}
    out = []
    for h in hold:
        v = vmap.get((h["model"], h["strategy"]), {})
        checks = {
            "sharpe_gt_1": (h["sharpe"] or 0) > 1.0,
            "beats_buy_hold_sharpe": (h["sharpe"] or 0) > (bh.get("sharpe") or 0),
            "beats_sma_sharpe": (h["sharpe"] or 0) > (sma.get("sharpe") or 0),
            "beats_naive_forecaster_same_strategy": (h["sharpe"] or 0) > (naive.get(h["strategy"], {}).get("sharpe") or -1e9),
            "mc_p_below_0.05": (h.get("mc_pvalue") is not None and h["mc_pvalue"] < 0.05),
            "validation_also_positive": (v.get("sharpe") or 0) > 0 and (v.get("mc_pvalue") or 1) < 0.1,
            "enough_trades_200": (h["n_trades"] or 0) >= 200 or h["strategy"] in ("vol_filter", "vol_target"),
        }
        out.append({"model": h["model"], "strategy": h["strategy"], "passed": all(checks.values()), "checks": checks,
                    "holdout_sharpe": h["sharpe"], "holdout_return": h["total_return"]})
    return {"candidates": out, "any_passed": any(c["passed"] for c in out)}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("suite")
    s.add_argument("--model", action="append", required=True)
    s.add_argument("--symbols", default=",".join(config.DEFAULT_SYMBOLS))
    s.add_argument("--tune", default="2025-01-01:2026-01-01")
    s.add_argument("--valid", default=f"2026-01-01:{config.HOLDOUT_START}")
    s.add_argument("--holdout", default=f"{config.HOLDOUT_START}:")
    s.add_argument("--context", type=int, default=1024)
    s.add_argument("--feature-set", default="close")
    s.add_argument("--mc", type=int, default=1000)
    s.add_argument("--quick", action="store_true")
    o = sub.add_parser("one")
    o.add_argument("--spec", required=True, help="JSON spec")
    o.add_argument("--save", action="store_true")
    a = ap.parse_args()
    conn = db.connect()
    db.init_schema(conn)

    def rng(x):
        a_, b_ = x.split(":")
        return (a_, b_ or None)

    if a.cmd == "suite":
        r = suite(conn, a.model, a.symbols.split(","), rng(a.tune), rng(a.valid), rng(a.holdout), a.feature_set,
                  a.context, a.mc, quick=a.quick)
        print(json.dumps({"validation": r["validation"], "holdout": r["holdout"], "baselines": r["baselines"],
                          "verdict": r["verdict"]}, indent=1, default=str))
    else:
        spec = json.loads(a.spec)
        cache = DataCache(conn, spec["symbols"], spec["start"], spec.get("end"))
        r = run_one(cache, spec, mc=spec.get("mc", 500), keep_series=a.save)
        if a.save:
            r["run_id"] = save_run(conn, r, spec.get("name", f"{spec['strategy']} {spec.get('model')}"))
        print(json.dumps(db.jsonable({"run_id": r.get("run_id"), "metrics": r["metrics"]}), indent=1))


if __name__ == "__main__":
    main()
