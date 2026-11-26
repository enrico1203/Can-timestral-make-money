"""Walk-forward forecast generation into the forecasts cache.

For every bar t in [start, end) a forecast is produced from the `context_len` bars ending at t
(inclusive). Missing candles are NaN in the context. Forecasts already cached are skipped.
"""

from __future__ import annotations

import argparse
import logging
import time

import numpy as np
import pandas as pd

from .. import config, db
from ..features import NAIVE_MODEL, ForecasterClient, build_inputs

log = logging.getLogger("generate")


def generate(conn, model: str, symbols: list[str], start, end=None, context_len: int = 1024, horizon: int = 24,
             feature_set: str = "close", batch: int = 64, timeframe: str | None = None, progress=None,
             client: ForecasterClient | None = None, stride: int = 1) -> dict:
    timeframe = timeframe or config.TIMEFRAME
    client = client or ForecasterClient()
    start = pd.Timestamp(start, tz="UTC") if pd.Timestamp(start).tzinfo is None else pd.Timestamp(start)
    end = None if end is None else (pd.Timestamp(end, tz="UTC") if pd.Timestamp(end).tzinfo is None else pd.Timestamp(end))
    todo_by_sym = {}
    for sym in symbols:
        df = db.load_candles(conn, sym, timeframe)
        grid = db.regular_grid(df, timeframe)
        if grid.empty:
            log.warning("%s: no candles", sym)
            continue
        have = db.existing_forecast_ts(conn, model, sym, timeframe, feature_set, context_len)
        mask = (grid.index >= start) & (grid.index < end if end is not None else True) & grid["close"].notna().to_numpy()
        positions = [i for i in np.flatnonzero(mask) if i >= context_len - 1 and grid.index[i] not in have
                     and (stride == 1 or grid.index[i].hour % stride == 0)]
        todo_by_sym[sym] = (grid, positions)
    total = sum(len(p) for _, p in todo_by_sym.values())
    done, t0 = 0, time.time()
    log.info("model=%s fs=%s ctx=%d: %d forecasts to compute", model, feature_set, context_len, total)
    for sym, (grid, positions) in todo_by_sym.items():
        for k in range(0, len(positions), batch):
            chunk = positions[k:k + batch]
            ctxs, pasts, knowns = [], [], []
            for i in chunk:
                c, p, kn = build_inputs(grid, i, context_len, horizon, feature_set, timeframe)
                ctxs.append(c)
                pasts.append(p)
                knowns.append(kn)
            ctx = np.stack(ctxs)
            past = np.stack(pasts) if feature_set == "cov" else None
            known = np.stack(knowns) if feature_set == "cov" else None
            q = client.forecast(model, ctx, horizon, past, known)  # (B, H, Q)
            rows = [(model, sym, timeframe, feature_set, context_len, grid.index[i].to_pydatetime(), horizon,
                     q.shape[-1], float(grid["close"].iloc[i]), db.pack_q(q[j])) for j, i in enumerate(chunk)]
            db.save_forecasts(conn, rows)
            done += len(chunk)
            if progress:
                progress(done / max(total, 1), f"{sym} {done}/{total}")
            if done % (batch * 20) < batch or done == total:
                rate = done / max(time.time() - t0, 1e-9)
                log.info("%s: %d/%d (%.1f/s, eta %.0fs)", model, done, total, rate, (total - done) / max(rate, 1e-9))
    return {"model": model, "computed": done, "seconds": round(time.time() - t0, 1)}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--symbols", default=",".join(config.DEFAULT_SYMBOLS))
    ap.add_argument("--start", required=True)
    ap.add_argument("--end")
    ap.add_argument("--context", type=int, default=1024)
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--feature-set", default="close")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--stride", type=int, default=1, help="only origins whose UTC hour is a multiple of this (sampling)")
    a = ap.parse_args()
    conn = db.connect()
    db.init_schema(conn)
    for m in a.model:
        print(generate(conn, m, a.symbols.split(","), a.start, a.end, a.context, a.horizon, a.feature_set, a.batch, stride=a.stride))


if __name__ == "__main__":
    main()
