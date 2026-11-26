CREATE EXTENSION IF NOT EXISTS timescaledb;

-- OHLCV candles. ts = candle OPEN time (UTC). Only closed candles are stored.
CREATE TABLE IF NOT EXISTS candles (
    symbol     text NOT NULL,
    timeframe  text NOT NULL,
    ts         timestamptz NOT NULL,
    open       double precision NOT NULL,
    high       double precision NOT NULL,
    low        double precision NOT NULL,
    close      double precision NOT NULL,
    volume     double precision NOT NULL,
    PRIMARY KEY (symbol, timeframe, ts)
);
SELECT create_hypertable('candles', 'ts', if_not_exists => TRUE, migrate_data => TRUE);

-- Forecast cache shared by backtests and live trading.
-- ts = open time of the LAST candle in the context (forecast made at its close).
-- quantiles = float32 little-endian, shape (horizon, n_levels), in log-price space.
CREATE TABLE IF NOT EXISTS forecasts (
    model        text NOT NULL,
    symbol       text NOT NULL,
    timeframe    text NOT NULL,
    feature_set  text NOT NULL,
    context_len  int  NOT NULL,
    ts           timestamptz NOT NULL,
    horizon      int  NOT NULL,
    n_levels     int  NOT NULL,
    last_close   double precision NOT NULL,
    quantiles    bytea NOT NULL,
    created_at   timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (model, symbol, timeframe, feature_set, context_len, ts)
);
CREATE INDEX IF NOT EXISTS forecasts_sym_ts ON forecasts (symbol, timeframe, ts);

-- A run is either the live paper-trading session or one backtest.
CREATE TABLE IF NOT EXISTS runs (
    id           serial PRIMARY KEY,
    kind         text NOT NULL,              -- live | backtest
    name         text NOT NULL,
    status       text NOT NULL DEFAULT 'running',
    started_at   timestamptz NOT NULL DEFAULT now(),
    finished_at  timestamptz,
    config       jsonb NOT NULL DEFAULT '{}',
    state        jsonb,
    metrics      jsonb,
    group_id     text
);

CREATE TABLE IF NOT EXISTS signals (
    id        bigserial PRIMARY KEY,
    run_id    int NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    symbol    text NOT NULL,
    ts        timestamptz NOT NULL,
    p_up      double precision,
    p_down    double precision,
    exp_ret   double precision,
    width     double precision,
    target    double precision,
    action    text NOT NULL,
    reason    text
);
CREATE INDEX IF NOT EXISTS signals_run_ts ON signals (run_id, ts DESC);

-- Orders and their fills (paper broker fills market orders at the next bar open).
CREATE TABLE IF NOT EXISTS orders (
    id          bigserial PRIMARY KEY,
    run_id      int NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    local_id    bigint NOT NULL,
    symbol      text NOT NULL,
    created_ts  timestamptz NOT NULL,
    side        text NOT NULL,             -- buy | sell
    qty         double precision NOT NULL,
    reason      text,
    status      text NOT NULL,             -- pending | filled | cancelled
    fill_ts     timestamptz,
    fill_price  double precision,
    fee         double precision,
    slippage    double precision
);
CREATE INDEX IF NOT EXISTS orders_run_ts ON orders (run_id, created_ts DESC);
CREATE UNIQUE INDEX IF NOT EXISTS orders_run_local ON orders (run_id, local_id);

-- Round-trip trades.
CREATE TABLE IF NOT EXISTS trades (
    id           bigserial PRIMARY KEY,
    run_id       int NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    symbol       text NOT NULL,
    side         text NOT NULL,            -- long | short
    qty          double precision NOT NULL,
    entry_ts     timestamptz NOT NULL,
    entry_price  double precision NOT NULL,
    exit_ts      timestamptz NOT NULL,
    exit_price   double precision NOT NULL,
    pnl          double precision NOT NULL, -- net of fees
    fees         double precision NOT NULL,
    bars         int NOT NULL,
    exit_reason  text
);
CREATE INDEX IF NOT EXISTS trades_run ON trades (run_id, exit_ts DESC);

CREATE TABLE IF NOT EXISTS equity (
    run_id           int NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    ts               timestamptz NOT NULL,
    cash             double precision NOT NULL,
    positions_value  double precision NOT NULL,
    equity           double precision NOT NULL,
    drawdown         double precision NOT NULL,
    exposure         double precision NOT NULL DEFAULT 0,
    PRIMARY KEY (run_id, ts)
);

-- Key/value settings edited from the UI (live strategy config, control flags).
CREATE TABLE IF NOT EXISTS settings (
    key         text PRIMARY KEY,
    value       jsonb NOT NULL,
    updated_at  timestamptz NOT NULL DEFAULT now()
);

-- Background jobs (backtests, forecast generation, evaluations) run by the worker.
CREATE TABLE IF NOT EXISTS jobs (
    id           serial PRIMARY KEY,
    kind         text NOT NULL,
    params       jsonb NOT NULL DEFAULT '{}',
    status       text NOT NULL DEFAULT 'queued',   -- queued | running | done | error
    progress     double precision NOT NULL DEFAULT 0,
    message      text,
    result       jsonb,
    created_at   timestamptz NOT NULL DEFAULT now(),
    started_at   timestamptz,
    finished_at  timestamptz
);

-- Stored reports (forecast evaluation, backtest suites).
CREATE TABLE IF NOT EXISTS reports (
    id          serial PRIMARY KEY,
    kind        text NOT NULL,
    name        text NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT now(),
    params      jsonb NOT NULL DEFAULT '{}',
    result      jsonb NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    id       bigserial PRIMARY KEY,
    ts       timestamptz NOT NULL DEFAULT now(),
    level    text NOT NULL,
    source   text NOT NULL,
    message  text NOT NULL
);
CREATE INDEX IF NOT EXISTS events_ts ON events (ts DESC);
