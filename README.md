# Timestral Paper Bot

A trading bot that runs **only in demo (paper) mode**. It uses
[Timestral](https://github.com/mistralai/timestral), Mistral AI's probabilistic forecasting model,
to decide when to buy or sell crypto (BTC, ETH, SOL on hourly candles). Prices are real, taken
from Binance's public API with no keys. Orders are simulated by an internal broker that charges
fees and slippage.

The goal is to answer one question with numbers: **can it make money?**

## Quick start

```bash
cp .env.example .env              # set DB_PASSWORD and UI_PASSWORD
docker volume create tradebot_hfcache
docker compose up -d --build
# Web UI: http://127.0.0.1:8230  (basic auth with UI_USER / UI_PASSWORD from .env)
```

On first start:
1. `collector` downloads the hourly candles since 2022 (about 2 minutes);
2. `forecaster` downloads the Timestral weights from Hugging Face into the `tradebot_hfcache` volume;
3. `trader` warms up the strategy on the last 720 hours, then trades on paper at every closed candle.

For remote access, use an SSH tunnel: `ssh -L 8230:127.0.0.1:8230 server`.

## Architecture

| Service | Role |
|---|---|
| `db` | TimescaleDB: candles, forecast cache, runs, orders, trades, equity, reports, jobs |
| `collector` | Public Binance candles via ccxt (closed candles only), fills gaps |
| `forecaster` | FastAPI around `TimestralForecaster` (CPU, batched, 5M/28M/123M/209M models) |
| `trader` | Live paper trading: forecast → strategy → risk → paper broker, state persisted every bar |
| `worker` | Runs jobs launched from the UI (forecast generation, evaluations, backtests, studies) |
| `api` | REST + WebSocket for the UI |
| `webui` | nginx + SPA (lightweight-charts), basic auth, port bound to 127.0.0.1 only |
| `backtester` | `cli` profile, for running research from the terminal |

The **same engine** (`services/core/tradebot/engine.py`) runs both backtests and live trading.

- Orders decided at the close of bar *t* are filled at the **open of bar *t+1*** with slippage (3 bps) and fee (0.1% per side).
- Stop-loss and take-profit are checked against the bar's high and low. If both are touched in the same bar, the stop is assumed to have hit first (pessimistic).
- Shorts are simulated as a perpetual future **without funding costs**.

## From the forecast to the order

For each symbol, at the close of every candle:

1. Timestral receives the last 1024 log-prices and returns 21 quantiles for each of the next 24 hours.
2. The strategy computes `P(up)`, the probability that the price exceeds the current price plus the round-trip costs, by interpolating the quantile distribution.
3. Available strategies:

| Strategy | Logic |
|---|---|
| `directional` | Long if `P(up) > p_enter`, short if `P(down) > p_enter`. Stop at q05, take-profit at q80, sizing by risk (1% of equity per trade), time-based or hysteresis exit |
| `vol_filter` | Long, but flat when the predicted uncertainty is in the top tail of its history |
| `vol_target` | Long exposure inversely proportional to the predicted uncertainty |
| `mean_reversion` | Buys when the price closes below the q05 predicted an hour earlier |
| `buy_hold`, `sma_cross` | Baselines with no model |

4. Risk controls:
   - at most 1/N of equity per symbol, no leverage;
   - daily stop at -3%;
   - kill-switch at -25% drawdown, which closes everything until a manual reset from the UI.

## How to tell whether it makes money (research protocol)

The forecast check comes first, with no trading. Page "Model quality", or:

```bash
docker compose --profile cli run --rm backtester python -m tradebot.backtest.evaluate \
  --model mistralai/Timestral-28M --model naive-rw --start 2026-10-16
```

It measures:
- directional accuracy with a binomial test;
- CRPS skill against a random walk;
- quantile coverage and PIT calibration;
- correlation between predicted uncertainty and realised movement (volatility forecasting).

Then the full study. Page "Research", or:

```bash
docker compose --profile cli run --rm backtester python -m tradebot.backtest.generate \
  --model naive-rw --model mistralai/Timestral-28M --start 2024-12-01
docker compose --profile cli run --rm backtester python -m tradebot.backtest.run suite \
  --model naive-rw --model mistralai/Timestral-28M
```

The study runs in three stages:
1. **Tuning** (2025): about 70 configurations per model.
2. **Validation** (2026-01-01 → 2026-10-15): the best configuration of each family, with a Monte Carlo test and the Deflated Sharpe Ratio.
3. **Holdout** (from 2026-10-16, after the Timestral weights were released, so it cannot be in their training data): each candidate is run **once**.

The Monte Carlo test keeps the same exposure path (same trades, durations and costs) but rotates the timing at random. It shows whether the result comes from timing or from luck and market drift.

**Pre-registered success criteria**, all of which must hold on the holdout:
- Sharpe > 1;
- beats buy & hold and the SMA crossover;
- beats the same strategy fed by a naive random-walk forecast;
- Monte Carlo p < 0.05;
- positive in validation too;
- at least 200 trades (for directional strategies).

## Useful commands

```bash
docker compose logs -f trader                 # what the bot is doing
docker compose run --rm --no-deps trader python -m pytest -q tests   # unit tests
docker compose exec db psql -U bot trading    # SQL
```

To trade other symbols, change `SYMBOLS` in `.env` and edit the config from the UI.

## Limits (read these)

- Paper trading: no slippage driven by real liquidity, no funding on shorts, no latency.
- A few weeks of hourly holdout data are statistically thin: treat the results as indicative.
- The Timestral weights are under a *Modified MIT* licence (check it on Hugging Face). The code in this repo uses the Apache-2.0 library.
