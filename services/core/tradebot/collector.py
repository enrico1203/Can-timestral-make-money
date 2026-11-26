"""Collector: backfills and keeps up to date the OHLCV candles from the exchange public API.

Only CLOSED candles are stored. No API keys are used.
"""

from __future__ import annotations

import logging
import time

import ccxt
import pandas as pd

from . import config, db

log = logging.getLogger("collector")


def make_exchange():
    ex = getattr(ccxt, config.EXCHANGE)({"enableRateLimit": True})
    return ex


def fetch_range(ex, symbol: str, timeframe: str, since_ms: int, until_ms: int | None = None):
    """Paginated fetch of closed candles from since_ms (inclusive)."""
    tf_ms = config.TIMEFRAME_SECONDS[timeframe] * 1000
    now_ms = ex.milliseconds()
    out = []
    cursor = since_ms
    while True:
        for attempt in range(5):
            try:
                batch = ex.fetch_ohlcv(symbol, timeframe, since=cursor, limit=1000)
                break
            except (ccxt.NetworkError, ccxt.ExchangeNotAvailable, ccxt.RequestTimeout) as e:
                log.warning("fetch error %s (attempt %d): %s", symbol, attempt + 1, e)
                time.sleep(2 * (attempt + 1))
        else:
            raise RuntimeError(f"failed to fetch {symbol}")
        if not batch:
            break
        for c in batch:
            if c[0] + tf_ms <= now_ms and (until_ms is None or c[0] < until_ms):  # closed only
                out.append(c)
        last = batch[-1][0]
        if last + tf_ms >= now_ms or len(batch) < 2 or (until_ms is not None and last >= until_ms):
            break
        cursor = last + tf_ms
    return out


def store(conn, symbol: str, timeframe: str, rows) -> int:
    if not rows:
        return 0
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO candles(symbol,timeframe,ts,open,high,low,close,volume) "
            "VALUES (%s,%s,to_timestamp(%s/1000.0),%s,%s,%s,%s,%s) "
            "ON CONFLICT (symbol,timeframe,ts) DO UPDATE SET open=EXCLUDED.open, high=EXCLUDED.high, "
            "low=EXCLUDED.low, close=EXCLUDED.close, volume=EXCLUDED.volume",
            [(symbol, timeframe, r[0], r[1], r[2], r[3], r[4], r[5]) for r in rows],
        )
    return len(rows)


def last_ts_ms(conn, symbol: str, timeframe: str) -> int | None:
    row = conn.execute("SELECT max(ts) AS m FROM candles WHERE symbol=%s AND timeframe=%s", (symbol, timeframe)).fetchone()
    return int(row["m"].timestamp() * 1000) if row and row["m"] else None


def fill_gaps(conn, ex, symbol: str, timeframe: str) -> int:
    """Re-download any holes inside the stored series."""
    df = db.load_candles(conn, symbol, timeframe)
    if len(df) < 2:
        return 0
    full = db.regular_grid(df, timeframe)
    missing = full.index[full["close"].isna()]
    if len(missing) == 0:
        return 0
    n = 0
    # group consecutive missing timestamps into ranges
    tf = pd.Timedelta(seconds=config.TIMEFRAME_SECONDS[timeframe])
    start = prev = missing[0]
    ranges = []
    for t in missing[1:]:
        if t - prev != tf:
            ranges.append((start, prev))
            start = t
        prev = t
    ranges.append((start, prev))
    for a, b in ranges:
        rows = fetch_range(ex, symbol, timeframe, int(a.timestamp() * 1000), int((b + tf).timestamp() * 1000))
        n += store(conn, symbol, timeframe, rows)
    if n == 0:
        log.info("%s: %d candles missing on the exchange too (left as gaps)", symbol, len(missing))
    return n


def sync_symbol(conn, ex, symbol: str, timeframe: str) -> int:
    last = last_ts_ms(conn, symbol, timeframe)
    tf_ms = config.TIMEFRAME_SECONDS[timeframe] * 1000
    since = last + tf_ms if last else int(pd.Timestamp(config.BACKFILL_START, tz="UTC").timestamp() * 1000)
    rows = fetch_range(ex, symbol, timeframe, since)
    n = store(conn, symbol, timeframe, rows)
    if n:
        conn.execute("SELECT pg_notify('candle_closed', %s)", (symbol,))
    return n


def current_symbols(conn) -> list[str]:
    live = db.get_setting(conn, "live_config")
    syms = list(config.DEFAULT_SYMBOLS)
    if live:
        syms += [s for s in live.get("symbols", []) if s not in syms]
    extra = db.get_setting(conn, "extra_symbols", [])
    syms += [s for s in extra if s not in syms]
    return syms


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    conn = db.connect()
    db.init_schema(conn)
    ex = make_exchange()
    tf = config.TIMEFRAME
    tf_s = config.TIMEFRAME_SECONDS[tf]
    first = True
    while True:
        try:
            for sym in current_symbols(conn):
                n = sync_symbol(conn, ex, sym, tf)
                if first:
                    g = fill_gaps(conn, ex, sym, tf)
                    db.event(conn, "info", "collector", f"{sym}: initial sync stored {n} candles, refilled {g} gaps")
                elif n:
                    log.info("%s: +%d candles", sym, n)
            first = False
        except Exception as e:  # keep running
            log.exception("collector loop error")
            try:
                db.event(conn, "error", "collector", f"loop error: {e}")
            except Exception:
                conn = db.connect()
        # poll shortly after each candle close, otherwise every 60s
        now = time.time()
        to_close = tf_s - (now % tf_s)
        time.sleep(min(60.0, to_close + 5))


if __name__ == "__main__":
    main()
