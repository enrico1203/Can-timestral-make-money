"""Static configuration from environment plus defaults for the live strategy."""

from __future__ import annotations

import os

TIMEFRAME_SECONDS = {"5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}


def env(name: str, default: str | None = None) -> str:
    v = os.getenv(name, default)
    if v is None:
        raise RuntimeError(f"missing env var {name}")
    return v


DATABASE_URL = env("DATABASE_URL", "postgresql://bot:bot@db:5432/trading").replace("postgresql+psycopg://", "postgresql://")
TRADING_MODE = env("TRADING_MODE", "paper")
FORECASTER_URL = env("FORECASTER_URL", "http://forecaster:8000")
EXCHANGE = env("EXCHANGE", "binance")
DEFAULT_SYMBOLS = [s.strip() for s in env("SYMBOLS", "BTC/USDT,ETH/USDT,SOL/USDT").split(",") if s.strip()]
TIMEFRAME = env("TIMEFRAME", "1h")
BACKFILL_START = env("BACKFILL_START", "2022-01-01")
DEFAULT_MODEL = env("MODEL_ID", "mistralai/Timestral-28M")
# Timestral weights were published on 2026-10-15 (commit a534bdb). Data after this date cannot
# have been in the pre-training corpus: it is the only clean out-of-sample holdout.
HOLDOUT_START = env("HOLDOUT_START", "2026-10-16")

QUANTILE_LEVELS = (0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                   0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99)


def assert_paper_mode() -> None:
    if TRADING_MODE != "paper":
        raise SystemExit(f"TRADING_MODE={TRADING_MODE!r}: this bot only runs in paper (demo) mode. Refusing to start.")


# Default live configuration; stored in the settings table on first start and editable from the UI.
DEFAULT_LIVE_CONFIG = {
    "symbols": DEFAULT_SYMBOLS,
    "model": DEFAULT_MODEL,
    "feature_set": "close",
    "context_len": 1024,
    "horizon": 24,
    "initial_cash": 10_000.0,
    "fee": 0.001,          # 0.1% taker per side
    "slippage": 0.0003,    # 3 bps per side
    "strategy": {
        "name": "directional",
        "params": {"h": 4, "p_enter": 0.6, "p_exit": None, "k_cost": 1.0, "allow_short": True,
                   "risk_per_trade": 0.01, "use_stops": True},
    },
    "risk": {"max_frac_per_symbol": None, "rebalance_band": 0.05,
             "daily_loss_limit": 0.03, "max_drawdown_halt": 0.25},
}
