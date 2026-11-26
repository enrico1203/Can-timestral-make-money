"""Performance and statistical-significance metrics."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats


def bars_per_year(timeframe_seconds: int) -> float:
    return 365.25 * 86400 / timeframe_seconds


def sharpe(r: np.ndarray, ppy: float) -> float:
    r = np.asarray(r, dtype=np.float64)
    r = r[np.isfinite(r)]
    if len(r) < 2 or r.std(ddof=1) == 0:
        return 0.0
    return float(r.mean() / r.std(ddof=1) * math.sqrt(ppy))


def performance(equity: pd.Series, trades: list[dict], ppy: float, fees_paid: float = 0.0,
                exposure: pd.Series | None = None) -> dict:
    eq = equity.astype(float)
    r = eq.pct_change().dropna().to_numpy()
    total = float(eq.iloc[-1] / eq.iloc[0] - 1) if len(eq) > 1 else 0.0
    years = max(len(eq) / ppy, 1e-9)
    cagr = float((eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1) if eq.iloc[-1] > 0 else -1.0
    dd = (eq / eq.cummax() - 1).min() if len(eq) else 0.0
    downside = r[r < 0]
    sortino = float(r.mean() / downside.std(ddof=1) * math.sqrt(ppy)) if len(downside) > 1 and downside.std(ddof=1) > 0 else 0.0
    pnl = np.array([t["pnl"] for t in trades]) if trades else np.zeros(0)
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    t_stat, p_val = (stats.ttest_1samp(r, 0.0) if len(r) > 2 and r.std() > 0 else (0.0, 1.0))
    out = {
        "start": eq.index[0].isoformat() if len(eq) else None,
        "end": eq.index[-1].isoformat() if len(eq) else None,
        "bars": int(len(eq)),
        "final_equity": float(eq.iloc[-1]) if len(eq) else None,
        "total_return": total,
        "cagr": cagr,
        "ann_vol": float(np.std(r, ddof=1) * math.sqrt(ppy)) if len(r) > 1 else 0.0,
        "sharpe": sharpe(r, ppy),
        "sortino": sortino,
        "max_drawdown": float(dd),
        "calmar": float(cagr / abs(dd)) if dd < 0 else 0.0,
        "n_trades": int(len(pnl)),
        "win_rate": float(len(wins) / len(pnl)) if len(pnl) else 0.0,
        "profit_factor": float(wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() != 0 else (math.inf if len(wins) else 0.0),
        "avg_trade_pnl": float(pnl.mean()) if len(pnl) else 0.0,
        "avg_bars_held": float(np.mean([t["bars"] for t in trades])) if trades else 0.0,
        "fees_paid": float(fees_paid),
        "mean_ret_tstat": float(t_stat),
        "mean_ret_pvalue": float(p_val),
    }
    if exposure is not None and len(exposure):
        out["avg_gross_exposure"] = float(exposure.abs().mean())
        out["time_in_market"] = float((exposure.abs() > 1e-6).mean())
    return out


def vector_returns(w: np.ndarray, rets: np.ndarray, cost: float) -> np.ndarray:
    """w (T, S) exposure held at the close of bar t; rets (T, S) close-to-close return of bar t.

    Portfolio return of bar t+1 = sum_s w[t, s] * rets[t+1, s] - cost * sum_s |w[t] - w[t-1]|.
    """
    gross = (w[:-1] * rets[1:]).sum(1)
    turn = np.abs(np.diff(w, axis=0, prepend=np.zeros((1, w.shape[1])))).sum(1)[:-1]
    return gross - cost * turn


def circular_shift_test(w: np.ndarray, rets: np.ndarray, cost: float, ppy: float, n: int = 1000,
                        seed: int = 0, min_shift: int = 168) -> dict:
    """Null distribution: same exposure path (same trades, durations, exposure, turnover), random timing.

    Each simulation rotates the exposure series of every symbol by an independent random offset,
    destroying any alignment between signals and future returns.
    """
    rng = np.random.default_rng(seed)
    w = np.nan_to_num(w)
    rets = np.nan_to_num(rets)
    T = len(w)
    actual = sharpe(vector_returns(w, rets, cost), ppy)
    if T <= 2 * min_shift:
        return {"actual_sharpe_vec": actual, "null_mean": None, "null_p95": None, "percentile": None, "p_value": None}
    sims = np.empty(n)
    for i in range(n):
        ws = np.empty_like(w)
        for s in range(w.shape[1]):
            ws[:, s] = np.roll(w[:, s], rng.integers(min_shift, T - min_shift))
        sims[i] = sharpe(vector_returns(ws, rets, cost), ppy)
    return {
        "actual_sharpe_vec": actual,
        "null_mean": float(sims.mean()),
        "null_std": float(sims.std()),
        "null_p95": float(np.percentile(sims, 95)),
        "percentile": float((sims < actual).mean() * 100),
        "p_value": float(((sims >= actual).sum() + 1) / (n + 1)),
        "hist": np.histogram(sims, bins=30)[0].tolist(),
        "hist_edges": np.histogram(sims, bins=30)[1].tolist(),
    }


def deflated_sharpe(sr_ann: float, n_obs: int, ppy: float, n_trials: int, sr_trials_ann: list[float],
                    skew: float = 0.0, kurt: float = 3.0) -> dict:
    """Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014). Works on per-bar Sharpe."""
    if n_obs < 3 or n_trials < 1:
        return {"dsr": None}
    sr = sr_ann / math.sqrt(ppy)
    trials = np.array(sr_trials_ann, dtype=float) / math.sqrt(ppy)
    var_trials = float(trials.var(ddof=1)) if len(trials) > 1 else 0.0
    emc = 0.5772156649
    if n_trials > 1 and var_trials > 0:
        sr0 = math.sqrt(var_trials) * ((1 - emc) * stats.norm.ppf(1 - 1 / n_trials)
                                       + emc * stats.norm.ppf(1 - 1 / (n_trials * math.e)))
    else:
        sr0 = 0.0
    denom = math.sqrt(max(1e-12, 1 - skew * sr + (kurt - 1) / 4 * sr ** 2))
    z = (sr - sr0) * math.sqrt(n_obs - 1) / denom
    return {"dsr": float(stats.norm.cdf(z)), "sr0_ann": float(sr0 * math.sqrt(ppy)), "n_trials": int(n_trials)}
