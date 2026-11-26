"""Postgres helpers (psycopg 3)."""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from . import config

log = logging.getLogger(__name__)


def connect(autocommit: bool = True, retries: int = 60) -> psycopg.Connection:
    last = None
    for _ in range(retries):
        try:
            return psycopg.connect(config.DATABASE_URL, autocommit=autocommit, row_factory=dict_row)
        except psycopg.OperationalError as e:  # db still starting
            last = e
            time.sleep(2)
    raise RuntimeError(f"cannot connect to database: {last}")


@contextmanager
def conn_ctx(autocommit: bool = True):
    c = connect(autocommit=autocommit)
    try:
        yield c
    finally:
        c.close()


def init_schema(conn: psycopg.Connection) -> None:
    sql = (Path(__file__).parent / "schema.sql").read_text()
    # advisory lock so several services starting together don't race
    with conn.cursor() as cur:
        cur.execute("SELECT pg_advisory_lock(424242)")
        try:
            cur.execute(sql)
        finally:
            cur.execute("SELECT pg_advisory_unlock(424242)")


def event(conn: psycopg.Connection, level: str, source: str, message: str) -> None:
    getattr(log, level if level in ("info", "warning", "error") else "info")("[%s] %s", source, message)
    conn.execute("INSERT INTO events(level, source, message) VALUES (%s,%s,%s)", (level, source, message[:2000]))


# ---------------------------------------------------------------- settings

def get_setting(conn, key: str, default=None):
    row = conn.execute("SELECT value FROM settings WHERE key=%s", (key,)).fetchone()
    return row["value"] if row else default


def set_setting(conn, key: str, value) -> None:
    conn.execute(
        "INSERT INTO settings(key, value, updated_at) VALUES (%s,%s,now()) "
        "ON CONFLICT (key) DO UPDATE SET value=EXCLUDED.value, updated_at=now()",
        (key, Jsonb(value)),
    )


# ---------------------------------------------------------------- candles

def load_candles(conn, symbol: str, timeframe: str, start=None, end=None) -> pd.DataFrame:
    q = "SELECT ts, open, high, low, close, volume FROM candles WHERE symbol=%s AND timeframe=%s"
    args: list = [symbol, timeframe]
    if start is not None:
        q += " AND ts >= %s"
        args.append(start)
    if end is not None:
        q += " AND ts < %s"
        args.append(end)
    q += " ORDER BY ts"
    rows = conn.execute(q, args).fetchall()
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    if len(df):
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df.set_index("ts")


def regular_grid(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Reindex to a complete time grid; missing candles become NaN rows."""
    if df.empty:
        return df
    freq = pd.Timedelta(seconds=config.TIMEFRAME_SECONDS[timeframe])
    idx = pd.date_range(df.index[0], df.index[-1], freq=freq, tz="UTC")
    return df.reindex(idx)


# ---------------------------------------------------------------- forecasts

def pack_q(q: np.ndarray) -> bytes:
    return np.ascontiguousarray(q, dtype="<f4").tobytes()


def unpack_q(b: bytes, n_levels: int) -> np.ndarray:
    return np.frombuffer(b, dtype="<f4").reshape(-1, n_levels)


def save_forecasts(conn, rows: list[tuple]) -> None:
    """rows: (model, symbol, timeframe, feature_set, context_len, ts, horizon, n_levels, last_close, qbytes)"""
    if not rows:
        return
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO forecasts(model,symbol,timeframe,feature_set,context_len,ts,horizon,n_levels,last_close,quantiles) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING",
            rows,
        )


def load_forecasts(conn, model, symbol, timeframe, feature_set, context_len, start=None, end=None):
    """Returns (index of ts, array (N, H, Q) of log-price quantiles, last_close array)."""
    q = ("SELECT ts, horizon, n_levels, last_close, quantiles FROM forecasts WHERE model=%s AND symbol=%s "
         "AND timeframe=%s AND feature_set=%s AND context_len=%s")
    args: list = [model, symbol, timeframe, feature_set, context_len]
    if start is not None:
        q += " AND ts >= %s"
        args.append(start)
    if end is not None:
        q += " AND ts < %s"
        args.append(end)
    q += " ORDER BY ts"
    rows = conn.execute(q, args).fetchall()
    if not rows:
        return pd.DatetimeIndex([], tz="UTC"), np.zeros((0, 0, 0), np.float32), np.zeros(0)
    h = min(r["horizon"] for r in rows)
    arr = np.stack([unpack_q(r["quantiles"], r["n_levels"])[:h] for r in rows])
    idx = pd.DatetimeIndex([r["ts"] for r in rows]).tz_convert("UTC")
    return idx, arr, np.array([r["last_close"] for r in rows])


def existing_forecast_ts(conn, model, symbol, timeframe, feature_set, context_len) -> set:
    rows = conn.execute(
        "SELECT ts FROM forecasts WHERE model=%s AND symbol=%s AND timeframe=%s AND feature_set=%s AND context_len=%s",
        (model, symbol, timeframe, feature_set, context_len),
    ).fetchall()
    return {pd.Timestamp(r["ts"]).tz_convert("UTC") for r in rows}


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def jsonable(o):
    """Make numpy / pandas values JSON serialisable (NaN/inf -> None, which jsonb accepts)."""
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return [jsonable(v) for v in o.tolist()]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return f if np.isfinite(f) else None
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.isoformat()
    if o is None or isinstance(o, str):
        return o
    return str(o)


def dumps(o) -> str:
    return json.dumps(jsonable(o))
