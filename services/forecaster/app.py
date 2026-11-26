"""HTTP wrapper around Timestral.

POST /forecast
  {"model": "mistralai/Timestral-28M" (optional),
   "context": [[[...T floats...]]]            # (B, V, T), NaN allowed as null
   "prediction_length": 24,
   "past_covariates": (B, Cp, T) | null,
   "known_covariates": (B, Ck, T+H) | null}
-> {"model", "quantile_levels": [21], "quantiles": (B, V, H, 21), "elapsed_ms"}
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from starlette.concurrency import run_in_threadpool
from timestral import TimestralForecaster

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("forecaster")

DEFAULT_MODEL = os.getenv("MODEL_ID", "mistralai/Timestral-28M")
ALLOWED = {m.strip() for m in os.getenv("ALLOWED_MODELS", DEFAULT_MODEL).split(",") if m.strip()} | {DEFAULT_MODEL}
torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "4")))

_models: dict[str, TimestralForecaster] = {}
_lock = threading.Lock()  # torch inference is serialized; model dict too


def get_model(name: str) -> TimestralForecaster:
    if name not in ALLOWED:
        raise HTTPException(400, f"model {name!r} not allowed; allowed: {sorted(ALLOWED)}")
    if name not in _models:
        t0 = time.time()
        log.info("loading %s", name)
        _models[name] = TimestralForecaster.from_pretrained(name)
        log.info("loaded %s in %.1fs", name, time.time() - t0)
    return _models[name]


app = FastAPI(title="timestral-forecaster")


@app.on_event("startup")
def _warm() -> None:
    with _lock:
        get_model(DEFAULT_MODEL)


def _to_float_array(x, name: str) -> np.ndarray | None:
    if x is None:
        return None
    try:
        a = np.array(x, dtype=np.float64)
    except (TypeError, ValueError):
        # nulls inside: replace with nan
        a = np.array(json.loads(json.dumps(x).replace("null", "NaN")), dtype=np.float64)
    if a.ndim != 3:
        raise HTTPException(400, f"{name} must be (B, C, T), got shape {a.shape}")
    return a.astype(np.float32)


@app.post("/forecast")
async def forecast(request: Request) -> Response:
    body = json.loads(await request.body())
    name = body.get("model") or DEFAULT_MODEL
    h = int(body.get("prediction_length", 1))
    if not 1 <= h <= 1024:
        raise HTTPException(400, "prediction_length must be in [1, 1024]")
    ctx = _to_float_array(body.get("context"), "context")
    if ctx is None:
        raise HTTPException(400, "context is required")
    kw = {}
    past = _to_float_array(body.get("past_covariates"), "past_covariates")
    known = _to_float_array(body.get("known_covariates"), "known_covariates")
    if past is not None:
        kw["past_covariates"] = past
    if known is not None:
        kw["known_covariates"] = known
    t0 = time.time()

    def run():
        with _lock:
            model = get_model(name)
            return model.forecast(ctx, prediction_length=h, **kw)

    try:
        # inference in a worker thread: keeps the event loop (and /health) responsive
        out = await run_in_threadpool(run)
    except (ValueError, AssertionError) as e:
        raise HTTPException(422, str(e)) from e
    q = out.quantiles.detach().cpu().numpy().astype(np.float32)
    payload = {
        "model": name,
        "quantile_levels": list(out.quantile_levels),
        "quantiles": np.round(q, 6).tolist(),
        "elapsed_ms": round((time.time() - t0) * 1000, 1),
    }
    return Response(json.dumps(payload), media_type="application/json")


@app.get("/health")
def health() -> dict:
    info = {}
    for n, m in _models.items():
        info[n] = {
            "covariates": bool(m.supports_covariates),
            "known_covariates": bool(m.supports_known_covariates),
            "params": sum(p.numel() for p in m.model.parameters()),
        }
    return {
        "ok": DEFAULT_MODEL in _models,
        "default_model": DEFAULT_MODEL,
        "allowed": sorted(ALLOWED),
        "loaded": info,
        "timestral_sha": os.getenv("TIMESTRAL_SHA"),
        "threads": torch.get_num_threads(),
    }
