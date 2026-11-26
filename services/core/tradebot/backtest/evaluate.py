"""Forecast quality evaluation against a naive random-walk baseline (no trading involved).

Answers: does the model know anything about direction? is it calibrated? does it forecast
volatility better than trailing volatility?
"""

from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import pandas as pd
from scipy import stats
from psycopg.types.json import Jsonb

from .. import config, db
from ..features import NAIVE_MODEL
from ..strategy import LEVELS, I05, I10, I50, I90, I95

log = logging.getLogger("evaluate")
HORIZONS = (1, 4, 12, 24)


def quantile_score(q: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Mean pinball loss over the quantile levels x 2 (≈ CRPS). q (N, Q), y (N,) -> (N,)"""
    diff = y[:, None] - q
    loss = np.maximum(LEVELS[None, :] * diff, (LEVELS[None, :] - 1) * diff)
    return 2 * loss.mean(1)


def pit(q: np.ndarray, y: np.ndarray) -> np.ndarray:
    out = np.empty(len(y))
    for i in range(len(y)):
        qi = np.maximum.accumulate(q[i])
        if y[i] <= qi[0]:
            out[i] = LEVELS[0] / 2
        elif y[i] >= qi[-1]:
            out[i] = (1 + LEVELS[-1]) / 2
        else:
            out[i] = np.interp(y[i], qi, LEVELS)
    return out


def prob_up(q: np.ndarray, last: np.ndarray) -> np.ndarray:
    return 1 - pit(q, last)


def naive_quantiles(logc: pd.Series, origins_pos: np.ndarray, k: int, L: int) -> np.ndarray:
    r = logc.diff()
    mu = r.rolling(L - 1, min_periods=50).mean().to_numpy()
    sd = r.rolling(L - 1, min_periods=50).std().to_numpy()
    z = stats.norm.ppf(LEVELS)
    last = logc.to_numpy()[origins_pos]
    return last[:, None] + mu[origins_pos, None] * k + sd[origins_pos, None] * np.sqrt(k) * z[None, :]


def binom_p(successes: int, n: int) -> float:
    if n == 0:
        return 1.0
    return float(stats.binomtest(successes, n, 0.5, alternative="greater").pvalue)


def evaluate_symbol(conn, model, symbol, timeframe, feature_set, context_len, start, end):
    idx, Q, _ = db.load_forecasts(conn, model, symbol, timeframe, feature_set, context_len, start, end)
    grid = db.regular_grid(db.load_candles(conn, symbol, timeframe), timeframe)
    if len(idx) == 0 or grid.empty:
        return None
    logc = np.log(grid["close"])
    pos_of = pd.Series(np.arange(len(grid)), index=grid.index)
    origins = pos_of.reindex(idx).to_numpy()
    out = {"n_forecasts": int(len(idx))}
    lc = logc.to_numpy()
    for k in HORIZONS:
        if k > Q.shape[1]:
            continue
        tgt = origins + k
        ok = (tgt < len(lc))
        ok[ok] &= np.isfinite(lc[tgt[ok]])
        o, t = origins[ok], tgt[ok]
        q = Q[ok, k - 1, :].astype(np.float64)
        y = lc[t]
        last = lc[o]
        nq = naive_quantiles(logc, o, k, context_len)
        good = np.all(np.isfinite(nq), axis=1)
        q, y, last, nq = q[good], y[good], last[good], nq[good]
        if len(y) < 30:
            continue
        ret = y - last
        med = q[:, I50] - last
        qs, nqs = quantile_score(q, y), quantile_score(nq, y)
        moved = (ret != 0) & (med != 0)
        hit = np.sign(med[moved]) == np.sign(ret[moved])
        nmed = nq[:, I50] - last
        nmoved = (ret != 0) & (nmed != 0)
        nhit = np.sign(nmed[nmoved]) == np.sign(ret[nmoved])
        pu = prob_up(q, last)
        conf = np.abs(pu - 0.5)
        top = conf >= np.quantile(conf, 0.8)
        top_m = top & moved
        top_hit = (np.sign(pu[top_m] - 0.5) == np.sign(ret[top_m]))
        width = q[:, I90] - q[:, I10]
        nwidth = nq[:, I90] - nq[:, I10]
        p = pit(q, y)
        up = (ret > 0).astype(float)
        brier = float(np.mean((pu - up) ** 2))
        bins = np.clip((pu * 10).astype(int), 0, 9)
        calib = [{"bin": b / 10, "n": int((bins == b).sum()),
                  "mean_p": float(pu[bins == b].mean()) if (bins == b).any() else None,
                  "freq_up": float(up[bins == b].mean()) if (bins == b).any() else None} for b in range(10)]
        out[f"h{k}"] = {
            "n": int(len(y)),
            "crps": float(qs.mean()), "crps_naive": float(nqs.mean()),
            "crps_skill": float(1 - qs.mean() / nqs.mean()),
            "crps_win_rate": float((qs < nqs).mean()),
            "mae_median": float(np.abs(y - q[:, I50]).mean()), "mae_naive": float(np.abs(y - nq[:, I50]).mean()),
            "dir_n": int(moved.sum()), "dir_acc": float(hit.mean()) if len(hit) else None,
            # windows of consecutive hourly forecasts at horizon k overlap: use every k-th one for the test
            "dir_pvalue": binom_p(int(hit[::k].sum()), int(len(hit[::k]))),
            "dir_n_independent": int(len(hit[::k])),
            "dir_acc_naive_drift": float(nhit.mean()) if len(nhit) else None,
            "dir_acc_top20_confident": float(top_hit.mean()) if len(top_hit) else None,
            "dir_top20_n": int(len(top_hit)),
            "dir_top20_pvalue": binom_p(int(top_hit.sum()), int(len(top_hit))),
            "frac_up": float(up.mean()),
            "brier": brier, "brier_coinflip": float(np.mean((0.5 - up) ** 2)),
            "coverage_80": float(((y >= q[:, I10]) & (y <= q[:, I90])).mean()),
            "coverage_90": float(((y >= q[:, I05]) & (y <= q[:, I95])).mean()),
            "coverage_80_naive": float(((y >= nq[:, I10]) & (y <= nq[:, I90])).mean()),
            "pit_hist": np.histogram(p, bins=10, range=(0, 1))[0].tolist(),
            "vol_spearman": float(stats.spearmanr(width, np.abs(ret)).statistic),
            "vol_spearman_naive": float(stats.spearmanr(nwidth, np.abs(ret)).statistic),
            "mean_width": float(width.mean()), "mean_abs_ret": float(np.abs(ret).mean()),
            "p_up_calibration": calib,
        }
    return out


def evaluate(conn, model, symbols, start, end=None, feature_set="close", context_len=1024, timeframe=None,
             name=None, save=True) -> dict:
    timeframe = timeframe or config.TIMEFRAME
    res = {"model": model, "feature_set": feature_set, "context_len": context_len, "start": str(start),
           "end": str(end) if end else None, "symbols": {}}
    for s in symbols:
        r = evaluate_symbol(conn, model, s, timeframe, feature_set, context_len, start, end)
        if r:
            res["symbols"][s] = r
    # pooled summary (simple average across symbols, weighted by n)
    summary = {}
    for k in HORIZONS:
        rows = [v[f"h{k}"] for v in res["symbols"].values() if f"h{k}" in v]
        if not rows:
            continue
        n = np.array([r["n"] for r in rows], float)
        dn = np.array([r["dir_n"] for r in rows], float)
        acc = np.array([r["dir_acc"] for r in rows], float)
        hits = int(round((acc * dn).sum()))
        # independent-sample count for the pooled test (non-overlapping windows)
        ind = np.array([r.get("dir_n_independent", r["dir_n"]) for r in rows], float)
        hits_ind = int(round((acc * ind).sum()))
        summary[f"h{k}"] = {
            "n": int(n.sum()),
            "crps_skill": float(np.average([r["crps_skill"] for r in rows], weights=n)),
            "dir_acc": float(hits / dn.sum()),
            "dir_pvalue": binom_p(hits_ind, int(ind.sum())),
            "dir_n_independent": int(ind.sum()),
            "coverage_80": float(np.average([r["coverage_80"] for r in rows], weights=n)),
            "coverage_90": float(np.average([r["coverage_90"] for r in rows], weights=n)),
            "vol_spearman": float(np.average([r["vol_spearman"] for r in rows], weights=n)),
            "vol_spearman_naive": float(np.average([r["vol_spearman_naive"] for r in rows], weights=n)),
            "brier": float(np.average([r["brier"] for r in rows], weights=n)),
            "brier_coinflip": float(np.average([r["brier_coinflip"] for r in rows], weights=n)),
        }
    res["summary"] = summary
    res = db.jsonable(res)
    if save:
        conn.execute("INSERT INTO reports(kind, name, params, result) VALUES ('forecast_eval', %s, %s, %s)",
                     (name or f"{model} {feature_set} ctx{context_len} {start}..{end or 'now'}",
                      Jsonb({"model": model, "symbols": symbols, "start": str(start), "end": str(end) if end else None,
                             "feature_set": feature_set, "context_len": context_len}), Jsonb(res)))
    return res


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--symbols", default=",".join(config.DEFAULT_SYMBOLS))
    ap.add_argument("--start", required=True)
    ap.add_argument("--end")
    ap.add_argument("--context", type=int, default=1024)
    ap.add_argument("--feature-set", default="close")
    a = ap.parse_args()
    conn = db.connect()
    for m in a.model:
        r = evaluate(conn, m, a.symbols.split(","), a.start, a.end, a.feature_set, a.context)
        print(json.dumps({"model": m, "summary": r["summary"]}, indent=1))


if __name__ == "__main__":
    main()
