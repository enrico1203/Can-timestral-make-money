"""REST + WebSocket API used by the web UI. Read-mostly; writes only settings, control flags and jobs."""

from __future__ import annotations

import asyncio
import math
from contextlib import asynccontextmanager

import httpx
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool
from pydantic import BaseModel, Field

from . import config, db
from .backtest.evaluate import evaluate_symbol
from .strategy import STRATEGIES

pool: ConnectionPool | None = None
ALLOWED_MODELS = ["naive-rw", "mistralai/Timestral-5M", "mistralai/Timestral-28M", "mistralai/Timestral-123M",
                  "mistralai/Timestral-209M"]
STRATEGY_DEFAULTS = {
    "directional": {"h": 4, "p_enter": 0.6, "p_exit": None, "k_cost": 1.0, "allow_short": True,
                    "risk_per_trade": 0.01, "use_stops": True},
    "vol_filter": {"h": 24, "pct": 80, "window": 720, "min_hist": 168},
    "vol_target": {"h": 24, "window": 720},
    "mean_reversion": {"level": 0.05, "max_bars": 12, "allow_short": False},
    "buy_hold": {},
    "sma_cross": {"fast": 20, "slow": 50, "allow_short": False},
}


@asynccontextmanager
async def lifespan(app):
    global pool
    with db.connect() as c:
        db.init_schema(c)
    pool = ConnectionPool(config.DATABASE_URL, min_size=1, max_size=8, kwargs={"autocommit": True, "row_factory": dict_row})
    yield
    pool.close()


app = FastAPI(title="tradebot api", lifespan=lifespan)


def q(sql, args=(), one=False):
    with pool.connection() as c:
        cur = c.execute(sql, args)
        if cur.description is None:
            return None
        return cur.fetchone() if one else cur.fetchall()


def clean(o):
    return db.jsonable(o)


@app.get("/api/health")
def health():
    q("SELECT 1")
    return {"ok": True, "mode": config.TRADING_MODE}


# ------------------------------------------------------------------ status

def live_run():
    return q("SELECT * FROM runs WHERE kind='live' AND status<>'stopped' ORDER BY id DESC LIMIT 1", one=True)


def run_summary(run) -> dict:
    st = run.get("state") or {}
    last_close = st.get("last_close", {})
    positions = []
    eq = st.get("cash", 0.0)
    for sym, p in (st.get("positions") or {}).items():
        px = last_close.get(sym, p["avg"])
        val = p["qty"] * px
        eq += val
        positions.append({"symbol": sym, "qty": p["qty"], "avg_price": p["avg"], "last": px, "value": val,
                          "unrealized": (px - p["avg"]) * p["qty"], "stop": p.get("stop"), "take_profit": p.get("tp"),
                          "bars": p["bars"], "entry_ts": p["entry_ts"]})
    init = (run.get("config") or {}).get("initial_cash", 10_000.0)
    ntr = q("SELECT count(*) n, coalesce(sum(pnl),0) pnl, coalesce(avg((pnl>0)::int),0) wr FROM trades WHERE run_id=%s",
            (run["id"],), one=True)
    return {
        "id": run["id"], "name": run["name"], "status": run["status"], "started_at": run["started_at"],
        "config": run["config"], "cash": st.get("cash"), "equity": eq, "initial_cash": init,
        "pnl": eq - init, "return": eq / init - 1 if init else None,
        "peak": st.get("peak"), "drawdown": (eq / st["peak"] - 1) if st.get("peak") else 0.0,
        "halted": st.get("halted"), "halt_reason": st.get("halt_reason"), "last_bar": st.get("last_ts"),
        "fees_paid": st.get("fees_paid"), "positions": positions, "pending_orders": st.get("pending", []),
        "n_trades": ntr["n"], "realized_pnl": ntr["pnl"], "win_rate": ntr["wr"],
    }


@app.get("/api/status")
def status():
    run = live_run()
    hb = q("SELECT value, updated_at FROM settings WHERE key='trader_heartbeat'", one=True)
    candles = q("SELECT symbol, max(ts) last, count(*) n FROM candles WHERE timeframe=%s GROUP BY symbol", (config.TIMEFRAME,))
    try:
        fh = httpx.get(f"{config.FORECASTER_URL}/health", timeout=3).json()
    except Exception as e:  # forecaster down
        fh = {"ok": False, "error": str(e)}
    ctl = q("SELECT value FROM settings WHERE key='control'", one=True)
    return clean({
        "mode": config.TRADING_MODE, "timeframe": config.TIMEFRAME,
        "now": db.utcnow(),
        "live": run_summary(run) if run else None,
        "trader_heartbeat": hb["updated_at"] if hb else None,
        "candles": candles, "forecaster": fh,
        "control": ctl["value"] if ctl else {},
    })


# ------------------------------------------------------------------ runs

@app.get("/api/runs")
def runs(kind: str | None = None, limit: int = 100):
    sql = "SELECT id, kind, name, status, started_at, finished_at, config, metrics, group_id FROM runs"
    args: list = []
    if kind:
        sql += " WHERE kind=%s"
        args.append(kind)
    sql += " ORDER BY id DESC LIMIT %s"
    args.append(limit)
    return clean(q(sql, args))


@app.get("/api/runs/{rid}")
def run_detail(rid: int):
    r = q("SELECT id, kind, name, status, started_at, finished_at, config, metrics, group_id FROM runs WHERE id=%s", (rid,), one=True)
    if not r:
        raise HTTPException(404)
    if r["kind"] == "live":
        full = q("SELECT * FROM runs WHERE id=%s", (rid,), one=True)
        r["summary"] = run_summary(full)
    return clean(r)


@app.get("/api/runs/{rid}/equity")
def run_equity(rid: int, max_points: int = 3000):
    run = q("SELECT config FROM runs WHERE id=%s", (rid,), one=True)
    if not run:
        raise HTTPException(404)
    rows = q("SELECT ts, equity, drawdown, exposure FROM equity WHERE run_id=%s ORDER BY ts", (rid,))
    if not rows:
        return {"points": [], "benchmark": []}
    df = pd.DataFrame(rows).set_index("ts")
    df.index = pd.to_datetime(df.index, utc=True)
    # equal-weight buy & hold of the run's symbols from the first equity point
    syms = (run["config"] or {}).get("symbols") or config.DEFAULT_SYMBOLS
    cl = q("SELECT ts, symbol, close FROM candles WHERE timeframe=%s AND symbol = ANY(%s) AND ts BETWEEN %s AND %s",
           (config.TIMEFRAME, syms, df.index[0], df.index[-1]))
    bench = None
    if cl:
        px = pd.DataFrame(cl).pivot(index="ts", columns="symbol", values="close")
        px.index = pd.to_datetime(px.index, utc=True)
        px = px.reindex(df.index).ffill().bfill()
        rel = (px / px.iloc[0]).mean(axis=1)
        bench = rel * float(df["equity"].iloc[0])
    step = max(1, math.ceil(len(df) / max_points))
    sel = df.iloc[::step]
    if sel.index[-1] != df.index[-1]:
        sel = pd.concat([sel, df.iloc[[-1]]])
    pts = [{"t": int(t.timestamp()), "equity": r["equity"], "drawdown": r["drawdown"], "exposure": r["exposure"]}
           for t, r in sel.iterrows()]
    bm = [{"t": int(t.timestamp()), "value": float(bench.loc[t])} for t in sel.index] if bench is not None else []
    return clean({"points": pts, "benchmark": bm})


@app.get("/api/runs/{rid}/trades")
def run_trades(rid: int, limit: int = 500, offset: int = 0):
    return clean(q("SELECT * FROM trades WHERE run_id=%s ORDER BY exit_ts DESC LIMIT %s OFFSET %s", (rid, limit, offset)))


@app.get("/api/runs/{rid}/orders")
def run_orders(rid: int, limit: int = 500):
    return clean(q("SELECT * FROM orders WHERE run_id=%s ORDER BY created_ts DESC, id DESC LIMIT %s", (rid, limit)))


@app.get("/api/runs/{rid}/signals")
def run_signals(rid: int, limit: int = 500, symbol: str | None = None, actionable: bool = False):
    sql = "SELECT * FROM signals WHERE run_id=%s"
    args: list = [rid]
    if symbol:
        sql += " AND symbol=%s"
        args.append(symbol)
    if actionable:
        sql += " AND action IN ('LONG','SHORT','FLAT') AND reason NOT LIKE 'no edge%%'"
    sql += " ORDER BY ts DESC, id DESC LIMIT %s"
    args.append(limit)
    return clean(q(sql, args))


# ------------------------------------------------------------------ market data / forecasts

@app.get("/api/candles")
def candles(symbol: str, limit: int = 500):
    rows = q("SELECT ts, open, high, low, close, volume FROM candles WHERE symbol=%s AND timeframe=%s ORDER BY ts DESC LIMIT %s",
             (symbol, config.TIMEFRAME, limit))
    return [{"t": int(r["ts"].timestamp()), "o": r["open"], "h": r["high"], "l": r["low"], "c": r["close"], "v": r["volume"]}
            for r in reversed(rows)]


def _cfg():
    return db.jsonable(q("SELECT value FROM settings WHERE key='live_config'", one=True)["value"]) \
        if q("SELECT 1 FROM settings WHERE key='live_config'", one=True) else config.DEFAULT_LIVE_CONFIG


@app.get("/api/forecasts/latest")
def latest_forecast(symbol: str, model: str | None = None, n_hist: int = 300):
    cfg = _cfg()
    model = model or cfg["model"]
    fs, ctx = cfg["feature_set"], cfg["context_len"]
    step = config.TIMEFRAME_SECONDS[config.TIMEFRAME]
    with pool.connection() as c:
        last = c.execute("SELECT max(ts) m FROM forecasts WHERE model=%s AND symbol=%s AND timeframe=%s AND feature_set=%s "
                         "AND context_len=%s", (model, symbol, config.TIMEFRAME, fs, ctx)).fetchone()["m"]
        if last is None:
            return {"model": model, "future": [], "history": []}
        start = pd.Timestamp(last) - pd.Timedelta(seconds=step * n_hist)
        idx, Q, _ = db.load_forecasts(c, model, symbol, config.TIMEFRAME, fs, ctx, start, None)
    levels = list(config.QUANTILE_LEVELS)
    li = {lv: levels.index(lv) for lv in (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)}
    t_last = int(idx[-1].timestamp())
    future = [{"t": t_last + step * (k + 1), **{f"q{int(lv * 100):02d}": float(np.exp(Q[-1, k, i])) for lv, i in li.items()}}
              for k in range(Q.shape[1])]
    # one-step-ahead forecasts for past bars (the forecast made at t, for bar t+1)
    hist = [{"t": int(t.timestamp()) + step, **{f"q{int(lv * 100):02d}": float(np.exp(Q[j, 0, i])) for lv, i in li.items()}}
            for j, t in enumerate(idx[:-1])]
    return {"model": model, "origin": t_last, "future": future, "history": hist}


@app.get("/api/model-quality")
def model_quality(days: int = 30, model: str | None = None):
    cfg = _cfg()
    model = model or cfg["model"]
    start = db.utcnow() - pd.Timedelta(days=days)
    out = {}
    with pool.connection() as c:
        for s in cfg["symbols"]:
            out[s] = evaluate_symbol(c, model, s, config.TIMEFRAME, cfg["feature_set"], cfg["context_len"], start, None)
    return clean({"model": model, "days": days, "symbols": out})


# ------------------------------------------------------------------ reports, jobs, events

@app.get("/api/reports")
def reports(kind: str | None = None, limit: int = 50):
    sql = "SELECT id, kind, name, created_at, params FROM reports"
    args: list = []
    if kind:
        sql += " WHERE kind=%s"
        args.append(kind)
    sql += " ORDER BY id DESC LIMIT %s"
    args.append(limit)
    return clean(q(sql, args))


@app.get("/api/reports/{rid}")
def report(rid: int):
    r = q("SELECT * FROM reports WHERE id=%s", (rid,), one=True)
    if not r:
        raise HTTPException(404)
    return clean(r)


class JobIn(BaseModel):
    kind: str = Field(pattern="^(generate|evaluate|backtest|suite)$")
    params: dict


@app.post("/api/jobs")
def create_job(j: JobIn):
    p = j.params
    if j.kind == "backtest":
        if p.get("strategy") not in STRATEGIES:
            raise HTTPException(400, "unknown strategy")
        if not p.get("start"):
            raise HTTPException(400, "start is required")
        p.setdefault("model", _cfg()["model"])
    if j.kind in ("generate", "evaluate", "suite"):
        if not p.get("models") or any(m not in ALLOWED_MODELS for m in p["models"]):
            raise HTTPException(400, f"models must be a subset of {ALLOWED_MODELS}")
    if "model" in p and p["model"] not in ALLOWED_MODELS + ["-"]:
        raise HTTPException(400, "model not allowed")
    r = q("INSERT INTO jobs(kind, params) VALUES (%s,%s) RETURNING id", (j.kind, Jsonb(p)), one=True)
    return {"id": r["id"]}


@app.get("/api/jobs")
def jobs(limit: int = 30):
    return clean(q("SELECT id, kind, params, status, progress, message, result, created_at, started_at, finished_at "
                   "FROM jobs ORDER BY id DESC LIMIT %s", (limit,)))


@app.get("/api/jobs/{jid}")
def job(jid: int):
    r = q("SELECT * FROM jobs WHERE id=%s", (jid,), one=True)
    if not r:
        raise HTTPException(404)
    return clean(r)


@app.get("/api/events")
def events(limit: int = 100):
    return clean(q("SELECT * FROM events ORDER BY id DESC LIMIT %s", (limit,)))


# ------------------------------------------------------------------ config / control

@app.get("/api/meta")
def meta():
    return {"strategies": STRATEGY_DEFAULTS, "models": ALLOWED_MODELS, "symbols": config.DEFAULT_SYMBOLS,
            "timeframe": config.TIMEFRAME, "holdout_start": config.HOLDOUT_START, "mode": config.TRADING_MODE}


@app.get("/api/config")
def get_config():
    return _cfg()


class StrategyCfg(BaseModel):
    name: str
    params: dict = {}


class RiskCfg(BaseModel):
    max_frac_per_symbol: float | None = Field(default=None, gt=0, le=1)
    rebalance_band: float = Field(0.05, ge=0, le=1)
    daily_loss_limit: float | None = Field(default=0.03, gt=0, le=1)
    max_drawdown_halt: float | None = Field(default=0.25, gt=0, le=1)


class LiveCfg(BaseModel):
    symbols: list[str] = Field(min_length=1, max_length=10)
    model: str
    feature_set: str = Field("close", pattern="^(close|cov)$")
    context_len: int = Field(1024, ge=128, le=8192)
    horizon: int = Field(24, ge=1, le=96)
    initial_cash: float = Field(10_000.0, gt=0)
    fee: float = Field(0.001, ge=0, le=0.01)
    slippage: float = Field(0.0003, ge=0, le=0.01)
    strategy: StrategyCfg
    risk: RiskCfg


@app.put("/api/config")
def put_config(cfg: LiveCfg):
    if cfg.strategy.name not in STRATEGIES:
        raise HTTPException(400, f"unknown strategy; available: {sorted(STRATEGIES)}")
    if cfg.model not in ALLOWED_MODELS:
        raise HTTPException(400, f"model must be one of {ALLOWED_MODELS}")
    for s in cfg.symbols:
        if "/" not in s:
            raise HTTPException(400, f"bad symbol {s}")
    h = cfg.strategy.params.get("h")
    if h is not None and int(h) > cfg.horizon:
        raise HTTPException(400, "strategy h cannot exceed forecast horizon")
    value = cfg.model_dump()
    with pool.connection() as c:
        db.set_setting(c, "live_config", value)
        db.event(c, "info", "api", f"live config updated from UI: {value['strategy']}")
    return value


@app.post("/api/control/{action}")
def control(action: str):
    if action not in ("pause", "resume", "flatten", "reset_halt", "reset_run"):
        raise HTTPException(400, "unknown action")
    with pool.connection() as c:
        ctl = db.get_setting(c, "control", {}) or {}
        if action == "pause":
            ctl["paused"] = True
        elif action == "resume":
            ctl["paused"] = False
        else:
            ctl[action] = True
        db.set_setting(c, "control", ctl)
        db.event(c, "info", "api", f"control action from UI: {action}")
    return ctl


# ------------------------------------------------------------------ websocket

@app.websocket("/ws/live")
async def ws_live(ws: WebSocket):
    await ws.accept()
    last_key = None
    try:
        while True:
            row = await asyncio.to_thread(q, "SELECT (SELECT max(id) FROM events) e, (SELECT max(updated_at) FROM settings) s, "
                                             "(SELECT max(ts) FROM candles) c", (), True)
            key = (row["e"], str(row["s"]), str(row["c"]))
            if key != last_key:
                last_key = key
                await ws.send_json({"type": "update", "status": await asyncio.to_thread(status)})
            await asyncio.sleep(3)
    except (WebSocketDisconnect, RuntimeError):
        return
