"""Model inputs: context windows, covariates, forecaster client and the naive baseline forecaster."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import httpx
from scipy.stats import norm

from . import config

FEATURE_SETS = ("close", "cov")
NAIVE_MODEL = "naive-rw"


def log_close(df: pd.DataFrame) -> np.ndarray:
    return np.log(df["close"].to_numpy(dtype=np.float64))


def past_covariates(df: pd.DataFrame) -> np.ndarray:
    """(2, T): log volume, log(high/low) range. NaN where the candle is missing."""
    vol = np.log1p(df["volume"].to_numpy(dtype=np.float64))
    rng = np.log(df["high"].to_numpy(dtype=np.float64) / df["low"].to_numpy(dtype=np.float64))
    return np.stack([vol, rng])


def known_covariates(index: pd.DatetimeIndex, horizon: int, timeframe: str) -> np.ndarray:
    """(3, T+H): hour-of-day sin/cos and weekend indicator (0/1), known in advance."""
    step = pd.Timedelta(seconds=config.TIMEFRAME_SECONDS[timeframe])
    full = index.append(pd.DatetimeIndex([index[-1] + step * (k + 1) for k in range(horizon)]))
    hours = full.hour.to_numpy() + full.minute.to_numpy() / 60.0
    weekend = (full.dayofweek.to_numpy() >= 5).astype(np.float64)
    return np.stack([np.sin(2 * np.pi * hours / 24), np.cos(2 * np.pi * hours / 24), weekend])


def build_inputs(grid: pd.DataFrame, end_pos: int, context_len: int, horizon: int, feature_set: str, timeframe: str):
    """Inputs for a forecast made at the close of row `end_pos` of a regular-grid frame.

    Uses rows [end_pos-context_len+1, end_pos] only: nothing after end_pos is read.
    """
    lo = end_pos - context_len + 1
    if lo < 0:
        raise ValueError("not enough history")
    win = grid.iloc[lo:end_pos + 1]
    ctx = log_close(win)[None, :]
    past = known = None
    if feature_set == "cov":
        past = past_covariates(win)
        known = known_covariates(win.index, horizon, timeframe)
    return ctx, past, known


class ForecasterClient:
    def __init__(self, url: str | None = None, timeout: float = 600.0):
        self.url = (url or config.FORECASTER_URL).rstrip("/")
        self.http = httpx.Client(timeout=timeout)

    def health(self) -> dict:
        return self.http.get(f"{self.url}/health").json()

    def forecast(self, model: str, ctx: np.ndarray, horizon: int, past: np.ndarray | None = None,
                 known: np.ndarray | None = None) -> np.ndarray:
        """ctx (B, 1, T) -> quantiles (B, H, Q)."""
        if model == NAIVE_MODEL:
            return naive_rw_forecast(ctx[:, 0, :], horizon)
        body = {"model": model, "context": ctx.tolist(), "prediction_length": horizon}
        if past is not None:
            body["past_covariates"] = past.tolist()
        if known is not None:
            body["known_covariates"] = known.tolist()
        for attempt in range(8):  # survive forecaster restarts / cold starts
            try:
                r = self.http.post(f"{self.url}/forecast", json=body)
                if r.status_code < 500:
                    break
            except httpx.TransportError:
                r = None
            time.sleep(min(60, 2 ** attempt))
        if r is None or r.status_code != 200:
            raise RuntimeError(f"forecaster error {getattr(r, 'status_code', 'unreachable')}: {getattr(r, 'text', '')[:500]}")
        q = np.array(r.json()["quantiles"], dtype=np.float32)  # (B, V, H, Q)
        return q[:, 0]


def naive_rw_forecast(logp: np.ndarray, horizon: int) -> np.ndarray:
    """Random walk with trailing drift and volatility (Gaussian). logp (B, T) -> (B, H, Q)."""
    z = norm.ppf(np.array(config.QUANTILE_LEVELS))
    out = np.empty((logp.shape[0], horizon, len(z)), dtype=np.float32)
    k = np.arange(1, horizon + 1)[:, None]
    for b in range(logp.shape[0]):
        r = np.diff(logp[b])
        r = r[np.isfinite(r)]
        mu = r.mean() if len(r) else 0.0
        sd = r.std(ddof=1) if len(r) > 1 else 0.0
        last = logp[b][np.isfinite(logp[b])][-1]
        out[b] = last + mu * k + sd * np.sqrt(k) * z[None, :]
    return out
