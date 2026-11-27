# Can Timestral make money?

**No.**

I built a paper-trading bot around [Timestral](https://github.com/mistralai/timestral), Mistral AI's
time-series forecasting model, and ran it on BTC, ETH and SOL hourly candles. I tried 512 strategy
configurations across four model sizes. None of them passed the success criteria I wrote down
before looking at the test data. Most Timestral strategies lost money even with trading costs set
to zero.

Timestral is a decent forecaster of how much the price will move. It has close to no idea which
way it will move, and that is the part you need to make money from buying and selling.

Everything here runs in demo mode. Prices are real (Binance public API, no keys), orders are
simulated, and there is no code path that sends a real order.

## The numbers

### Forecast quality, before any trading

About 47,000 forecasts from 2025-01-01 to 2026-10-15, compared with a naive random walk that
uses the trailing drift and volatility. Direction p-values use non-overlapping windows only.

| Model | Direction hit rate, 1h | p | Direction hit rate, 12h | CRPS skill, 1h | 80% interval coverage | Volatility correlation |
|---|---|---|---|---|---|---|
| Timestral-5M | 50.6% | 0.007 | 50.2% | +3.1% | 84% | 0.33 |
| Timestral-28M | 50.0% | n.s. | 50.2% | +3.7% | 82% | 0.34 |
| Timestral-123M | 50.6% | 0.007 | 50.9% | +3.9% | 81% | 0.34 |
| Naive random walk | | | | 0 | | 0.17 |

A 50.6% hit rate is statistically real and economically useless. The average hourly move is around
0.5%, and a round trip costs 0.26% in this simulation (0.10% fee plus 0.03% slippage per side).
The edge per trade is a small fraction of the cost.

The volatility column is where Timestral does well: the width of its forecast interval correlates
twice as strongly with the size of the next move as the naive baseline does, and the intervals are
well calibrated.

### Backtests

The study has three stages:

1. **Tune** on 2025, 128 configurations per model.
2. **Validate** the best configuration of each strategy family on 2026-01-01 to 2026-10-15.
3. **Holdout** from 2026-10-16 to 2026-11-26, run once. The Timestral weights were released on
   2026-10-15, so this period cannot be in their training data.

Holdout results, net of costs. "Gross" re-runs the same decisions with zero fees and slippage.

| Model | Strategy | Net | Gross | Trades | Monte Carlo p |
|---|---|---|---|---|---|
| Timestral-123M | directional | 0.0% | 0.0% | 0 | 1.00 |
| Timestral-123M | mean_reversion | -13.2% | +1.3% | 178 | 0.009 |
| Timestral-123M | vol_filter | -34.6% | -27.1% | 123 | 0.96 |
| Timestral-123M | vol_target | -28.3% | -27.7% | – | 1.00 |
| Timestral-28M | mean_reversion | -12.8% | -0.3% | 155 | 0.016 |
| Timestral-5M | directional | -1.9% | -1.3% | 12 | 0.39 |
| Timestral-5M | mean_reversion | -10.6% | -0.8% | 120 | 0.026 |
| Naive random walk | directional | +18.1% | +25.3% | 82 | 0.28 |
| Naive random walk | mean_reversion | -10.4% | +1.4% | 143 | 0.001 |
| Buy & hold, equal weight | | -29.9% | | | |

How to read it:

- **directional**: the model is rarely confident enough to clear the cost hurdle, so it barely
  trades. On validation it made 16 trades and lost 3.4%.
- **mean_reversion** (buy when price closes below the previous hour's 5% quantile): the timing is
  better than random, with a Monte Carlo p below 0.05. The naive random walk gets the same result,
  so the signal comes from short-term mean reversion in crypto and has nothing to do with Timestral.
  Fees then eat it. On validation the 123M version paid $3,949 in fees on a $10,000 account and
  ended at -57.5%.
- **vol_filter / vol_target**: a good volatility forecast does not stop you from losing money in
  a market that fell 30%.
- The only positive row is the naive directional strategy, which went short during the decline.
  Its Monte Carlo p is 0.28, so it is consistent with luck.

### Covariates

I also ran Timestral-28M with past covariates (log volume, high/low range) and known future
covariates (hour of day, weekend flag) on the same 8,300 origins. CRPS and direction accuracy did
not improve at any horizon, and inference was about 15 times slower.

The full write-up, in Italian, is in [RESULTS.md](RESULTS.md).

## Success criteria

Written into the code (`verdict()` in `services/core/tradebot/backtest/run.py`) before running
the holdout. A candidate passes only if all of these hold on the holdout:

- Sharpe ratio above 1
- better Sharpe than buy & hold and an SMA crossover
- better than the same strategy driven by the naive random-walk forecast
- Monte Carlo p below 0.05
- positive return on validation too
- at least 200 trades for directional strategies

The Monte Carlo test keeps the strategy's exposure path (same number of trades, same holding
times, same costs) and shifts it in time at random. If the real result sits inside that
distribution, the timing carried no information.

Two mistakes I caught along the way, in case you build something similar:

- Hourly forecasts at a 12h or 24h horizon overlap, so consecutive hits are correlated. Treating
  them as independent made a 55% hit rate on the holdout look significant. On every 12th sample it
  is not (p 0.08 to 0.14).
- The tuning stage happily picked configurations that made one lucky trade. It now requires at
  least 30 trades.

## What is in the repo

A Docker Compose stack with a web UI.

| Service | What it does |
|---|---|
| `db` | TimescaleDB: candles, forecast cache, runs, orders, trades, equity, reports, jobs |
| `collector` | Downloads closed hourly candles from Binance through ccxt and fills gaps |
| `forecaster` | FastAPI wrapper around `TimestralForecaster` (CPU, batched, 5M/28M/123M/209M) |
| `trader` | Live paper trading: forecast, strategy, risk checks, simulated broker; state saved every bar |
| `worker` | Runs jobs started from the UI: forecast generation, evaluations, backtests, studies |
| `api` | REST and WebSocket API for the UI |
| `webui` | nginx and a small JS app (lightweight-charts), basic auth |
| `backtester` | `cli` profile for running research from the terminal |

Backtests and live trading use the same engine (`services/core/tradebot/engine.py`):

- An order decided at the close of bar *t* fills at the open of bar *t+1*, with 3 bps slippage
  and a 0.1% fee per side.
- Stops and take-profits are checked against the bar's high and low. If both are hit in the same
  bar, the stop counts first.
- Shorts behave like a perpetual future with no funding cost.

At every candle close, Timestral receives the last 1024 log prices and returns 21 quantiles for
each of the next 24 hours. The strategy interpolates those quantiles into the probability that
the price ends above the current price plus round-trip costs.

| Strategy | Logic |
|---|---|
| `directional` | Long if `P(up) > p_enter`, short if `P(down) > p_enter`. Stop at q05, take-profit at q80, 1% of equity at risk per trade, exit on time or hysteresis |
| `vol_filter` | Long, flat when predicted uncertainty is in the top tail of its history |
| `vol_target` | Long, with exposure inversely proportional to predicted uncertainty |
| `mean_reversion` | Buys when price closes below the q05 forecast from one hour earlier |
| `buy_hold`, `sma_cross` | Baselines with no model |

Risk limits: at most 1/N of equity per symbol, no leverage, a -3% daily stop, and a kill switch at
-25% drawdown that stays on until you reset it from the UI.

The UI has pages for the live dashboard, a price chart with the forecast fan, trades, model
quality, backtests, research studies and settings (pause, flatten, change strategy parameters).

## Running it

```bash
cp .env.example .env              # set DB_PASSWORD and UI_PASSWORD
docker volume create tradebot_hfcache
docker compose up -d --build
# UI at http://127.0.0.1:8230, basic auth with UI_USER / UI_PASSWORD
```

On first start the collector downloads hourly candles since 2022 (a couple of minutes), the
forecaster pulls the weights from Hugging Face, and the trader warms up on the last 720 hours
before it starts trading on paper.

The UI binds to 127.0.0.1. To reach it from another machine, use an SSH tunnel
(`ssh -L 8230:127.0.0.1:8230 server`) or set `WEBUI_BIND_EXTRA` in `.env` to a private address
such as a Tailscale IP.

Reproducing the study from the terminal:

```bash
# forecast quality
docker compose --profile cli run --rm backtester python -m tradebot.backtest.evaluate \
  --model mistralai/Timestral-28M --model naive-rw --start 2026-10-16

# generate walk-forward forecasts, then run tune / validation / holdout
docker compose --profile cli run --rm backtester python -m tradebot.backtest.generate \
  --model naive-rw --model mistralai/Timestral-28M --start 2024-12-01
docker compose --profile cli run --rm backtester python -m tradebot.backtest.run suite \
  --model naive-rw --model mistralai/Timestral-28M
```

Forecast generation on CPU is the slow part. The forecasts are cached in the database, so an
interrupted run picks up where it stopped.

Other commands:

```bash
docker compose logs -f trader
docker compose run --rm --no-deps trader python -m pytest -q tests
docker compose exec db psql -U bot trading
```

## Limits

- Six weeks of holdout in a falling market is a small sample. The forecast-quality results cover
  almost two years and point the same way, which is why I trust the conclusion.
- Simulated fills ignore order-book depth, latency and funding on shorts. All of these would make
  real results worse.
- I did not run the 209M model through the full study. The 5M, 28M and 123M results are flat
  across sizes, so I don't expect it to change the answer, but I haven't checked.
- I only tested hourly crypto. Longer bars, cheaper execution (maker fees) or series with strong
  seasonality could give a different result. Using Timestral as a volatility estimate for position
  sizing inside a strategy that already has an edge is the most plausible use I can see.

## License

The code in this repository uses the Timestral library, which is Apache-2.0. The Timestral model
weights have their own Modified MIT licence; check it on Hugging Face before using them.

This is a research experiment and not financial advice.
