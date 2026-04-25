# regime-trader

HMM-based volatility regime detection and automated allocation system for US equities, wired to Alpaca for live and paper execution.

The system classifies the current market environment (low-vol, mid-vol, high-vol) using a Hidden Markov Model trained on realised volatility features, then adjusts portfolio weights, leverage, and position sizes accordingly. A multi-layer risk manager gates every order before it reaches the broker.

---

## Architecture

```
market data
    └─> FeatureEngineer      (log returns, realised vol, ADX, SMA distances)
            └─> HMMEngine    (BIC model selection, regime label + confidence)
                    └─> RegimeStrategy   (target weights per regime)
                            └─> RiskManager  (size, exposure, drawdown gates)
                                    └─> OrderExecutor  (limit orders -> Alpaca)
                                            └─> PositionTracker  (fills, P&L)
```

Supporting layer runs in parallel:
- `SignalGenerator` — single entry point that drives the pipeline each bar
- `TradeLogger` — JSON-structured rotating log
- `Dashboard` — Rich terminal live view
- `AlertManager` — rate-limited email + Slack/webhook alerts

---

## Multi-Strategy Mode

The system can run several strategies in parallel under a single capital
allocator and a portfolio-level risk manager. **Active strategies:** `hmm_regime`
(S&P 500 regime detection) and `momentum_breakout` (large-cap tech). Other
strategy implementations (mean-reversion, bond, commodity) are disabled by
default as they underperform the current equity-focused universe.

```
StrategyRegistry  ─►  CapitalAllocator  ─►  PortfolioRiskManager  ─►  Broker
   (health checks       (inverse_vol /         (max 80% gross,
   auto-disable          risk_parity / ...      single-symbol cap,
   bad strategies)       correlation merger)    daily/peak DD halt)
```

### Quick start

```bash
# Paper trade with all enabled strategies from settings.yaml + inverse-vol allocator
py -3.12 main.py trade --paper

# Pick a different allocator approach
py -3.12 main.py trade --paper --allocator risk_parity

# Restrict to a subset of strategies
py -3.12 main.py trade --paper --strategies hmm_regime,momentum_breakout

# Backtest the multi-strategy stack
py -3.12 main.py backtest --multi-strat --allocator inverse_vol
```

Available `--allocator` values: `equal_weight`, `inverse_vol` (default),
`risk_parity`, `performance_weighted`. Strategies and their per-strategy
`weight_min` / `weight_max` floors live under `strategies:` in
`config/settings.yaml`.

See [docs/multistrat.md](docs/multistrat.md) for the full guide: how to add
a strategy, how the correlation merger works, what each allocator approach
optimises for, the portfolio risk caps, and the common failure modes.

---

## Regime framework

The HMM maps its internal states to vol tiers by sorting them on expected volatility (ascending). Labels are assigned by expected return (ascending). This makes the mapping independent of arbitrary state ordering.

| Regime | Market condition | Default allocation | Leverage |
|---|---|---|---|
| `low_vol` | Calmest HMM state, trending | 95% of equity | 1.25× |
| `mid_vol` (trend) | Transition, price above 50 EMA | 95% of equity | 1.0× |
| `mid_vol` (no trend) | Transition, price below 50 EMA | 75% of equity | 1.0× |
| `high_vol` | Most volatile HMM state | 75% of equity | 1.0× |

Allocations above are the **base** (`settings.yaml`). Config sets override them — see [Config sets](#config-sets).

Stability filtering requires a configurable number of consecutive bars in the same state before a regime flip is confirmed. An uncertainty discount halves all position sizes if confidence is below threshold or the regime is flickering.

---

## Project layout

```
regime-trader/
├── config/
│   ├── settings.yaml          # Base parameters (universe, risk, HMM, backtest)
│   ├── credentials.yaml       # Alpaca API keys — git-ignored, never commit
│   ├── active_set             # Name of the active config set (e.g. "balanced")
│   └── sets/
│       ├── conservative.yaml  # Low-churn, capital-preservation overrides
│       ├── balanced.yaml      # Fixes main issues — recommended default
│       └── aggressive.yaml    # Max deployment, 1.5× leverage overrides
│
├── core/
│   ├── hmm_engine.py          # Gaussian HMM with BIC model selection and incremental update
│   ├── regime_strategies.py   # Per-regime allocation logic (StrategyOrchestrator)
│   ├── risk_manager.py        # Circuit breakers, position sizing, drawdown limits
│   └── signal_generator.py   # Orchestrates core pipeline -> PortfolioSignal
│
├── broker/
│   ├── alpaca_client.py       # Alpaca REST + WebSocket wrapper (paper and live)
│   ├── order_executor.py      # Limit/market order placement, cancellation, tracking
│   └── position_tracker.py   # Real-time position sync via TradingStream WebSocket
│
├── data/
│   ├── market_data.py         # Historical bars, price pivot, streaming data callbacks
│   └── feature_engineering.py # Causal feature matrix (log returns, realised vol, ADX, SMA)
│
├── monitoring/
│   ├── logger.py              # JSON-structured rotating file logger (TradeLogger)
│   ├── dashboard.py           # Rich terminal live dashboard (background thread)
│   └── alerts.py              # Email (SMTP/STARTTLS) + webhook alerts with rate limiting
│
├── backtest/
│   ├── backtester.py          # Walk-forward backtester — zero look-ahead bias
│   ├── performance.py         # Sharpe, CAGR, max drawdown, regime breakdown, benchmarks
│   └── stress_test.py         # Crash / gap / vol-spike injection scenarios
│
├── tests/
│   ├── test_hmm.py            # HMM engine unit tests
│   ├── test_look_ahead.py     # Verify zero look-ahead bias in feature pipeline
│   ├── test_strategies.py     # Regime strategy allocation tests
│   ├── test_risk.py           # Risk manager circuit breaker and sizing tests
│   └── test_orders.py         # OrderExecutor limit price, sizing, submission
│
├── menu/
│   └── regime_trader.sh       # Interactive bash launcher menu
│
├── main.py                    # CLI entry point (backtest / trade / stress / full-cycle)
├── pytest.ini                 # Suppresses third-party deprecation warnings
└── requirements.txt
```

---

## Quick start

### 1. Install (Linux / VPS / WSL2 — recommended)

**One-command install** — handles Python check, venv, packages, credentials, and tests:

```bash
git clone https://github.com/morningtrading/regime-trader.git
cd regime-trader
chmod +x install.sh
./install.sh
```

The script runs 7 steps with pass/fail confirmation at each one:

| Step | What it does |
|---|---|
| 1 | Checks Python 3.12 (installs via apt if missing) |
| 2 | Creates `.venv/` inside the repo |
| 3 | Installs all pinned packages from `requirements.txt` |
| 4 | Verifies critical packages + BLAS backend |
| 5 | Checks credentials file (creates template if missing) |
| 6 | Runs full pytest suite |
| 7 | Smoke-tests that all core modules import correctly |

**Requirements:** Python 3.12, pip, git. On a fresh Ubuntu/Debian VPS:

```bash
sudo apt-get update && sudo apt-get install -y python3.12 python3.12-venv python3.12-dev git
```

---

### Manual install (any platform)

```bash
# Linux / macOS / WSL2
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Windows (PowerShell)
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> **⚠️ Cross-platform reproducibility**
>
> Backtest results are **deterministic across Linux machines** (Ubuntu 24.04 native,
> WSL2 Ubuntu-24.04, Linux VPS — all produce identical output). **Windows native
> Python produces different results** due to ILP64 vs LP64 OpenBLAS builds: the
> HMM EM algorithm converges to different local optima → different regime labels
> → different trades → different P&L (~15–30% difference in total return is normal).
>
> **Canonical platform: Linux (native or WSL2). Reference results** (stocks4 basket, 5 symbols, 2020–2026, conservative set, `enforce_stops=False`):
> `Total Return +94.61% | Sharpe 0.860 | MaxDD -11.56%` (commit `f84b278`).
>
> Windows native on the same code/config gives `+118.55% / Sharpe 0.939 / MaxDD -17.2%` — divergence is from BLAS / floating-point accumulation differences in the HMM fit, NOT random-seed (random_state is already fixed). Cannot be reconciled without containerising the toolchain — not worth it.
>
> **Use Windows for development and quick iteration; use Linux for any number that goes into a decision.**

### 2. Credentials

Create `config/credentials.yaml` (already git-ignored):

```yaml
alpaca:
  api_key:    "YOUR_ALPACA_KEY"
  secret_key: "YOUR_ALPACA_SECRET"
  paper:      true
```

Get keys from [alpaca.markets](https://alpaca.markets) → Paper Trading → API Keys.

Alternatively use a `.env` file (also git-ignored):

```
ALPACA_API_KEY=...
ALPACA_SECRET_KEY=...
```

The client loads `credentials.yaml` first and falls back to `.env` / environment variables.

### 3. Launch the menu

The fastest way to run anything:

```bash
bash menu/regime_trader.sh
```

The menu shows the active asset group and config set, and exposes all run modes as numbered options.

### 4. Run a backtest from the CLI

```bash
py -3.12 main.py backtest --asset-group stocks --start 2020-01-01 --compare
```

`--compare` adds buy-and-hold and SMA-200 benchmark columns.

### 5. Run stress tests

```bash
py -3.12 main.py stress --asset-group stocks --start 2019-01-01
```

### 6. Start paper trading

```bash
py -3.12 main.py trade --paper
```

### 7. Run the test suite

```bash
py -3.12 -m pytest tests/ -v
```

---

## Config sets

Named parameter sets live in `config/sets/`. Each file contains only the overrides applied on top of `config/settings.yaml`. The active set is stored in `config/active_set`.

| Set | Focus | Key differences from base |
|---|---|---|
| `conservative` | Capital preservation | No leverage, stability=9 bars, high_vol alloc=45%, rebalance threshold=25% |
| `balanced` | Recommended default | stability=7, flicker=4, min_confidence=0.62, high_vol=60%, slippage=10 bps |
| `aggressive` | Max deployment | 1.5× leverage, stability=5, low_vol=100%, rebalance threshold=10% |

### Switching sets

**Via the menu** — press `[c]` from the main menu and choose a set. The choice is persisted to `config/active_set` and used by all subsequent runs.

**Via CLI for a single run** (does not change `active_set`):

```bash
py -3.12 main.py backtest --asset-group stocks --start 2020-01-01 --set conservative
py -3.12 main.py backtest --asset-group stocks --start 2020-01-01 --set aggressive
```

### Adding a custom set

Drop a YAML file in `config/sets/` with only the parameters you want to override:

```yaml
# config/sets/my_set.yaml
hmm:
  stability_bars: 6
  min_confidence: 0.65
strategy:
  high_vol_allocation: 0.50
```

Then activate it with `--set my_set` or by updating `config/active_set`.

---

## Configuration reference

All base parameters are in [config/settings.yaml](config/settings.yaml):

| Section | What it controls |
|---|---|
| `broker` | Trading universe, timeframe, paper vs live, data feed |
| `hmm` | State-count candidates (3–7), stability bars, flicker threshold, min confidence, features |
| `strategy` | Per-regime allocation fractions, leverage, rebalance threshold, trend lookback |
| `risk` | Per-trade risk, exposure caps, drawdown thresholds |
| `backtest` | Walk-forward windows, slippage, benchmark, commission model |
| `monitoring` | Dashboard refresh rate, alert rate limit, log level and rotation |

Key HMM parameters explained:

| Parameter | Default | Effect |
|---|---|---|
| `stability_bars` | 5 (base) / 7 (balanced) | Bars in same state required to confirm a regime flip |
| `flicker_threshold` | 2 (base) / 4 (balanced) | Max regime changes in `flicker_window` before uncertainty mode |
| `min_confidence` | 0.70 (base) / 0.62 (balanced) | HMM posterior floor — below this, position sizes are halved |
| `n_candidates` | [3,4,5,6,7] | State counts tested; best selected by BIC |

---

## Walk-forward backtesting

The backtester uses a strict walk-forward methodology to prevent look-ahead bias:

- **In-sample window** (`train_window=252` bars ≈ 12 months): HMM trained here
- **Out-of-sample window** (`test_window=63` bars ≈ 3 months): strategy evaluated blind
- **Step size** (`step_size=63`): equal to the OOS window — test periods never overlap
- Equity is carried forward between folds (compounding)
- ~252 bars consumed as feature warmup (SMA-200 + rolling volatility); data before the first clean bar feeds the warmup only

A start date of 2020-01-01 with default windows produces first OOS results starting around mid-2022. This is expected and correct.

Performance metrics per fold and overall: CAGR, Sharpe, max drawdown, annualised vol, regime breakdown, regime transition matrix, comparison to buy-and-hold and SMA-200 benchmarks.

### Why 17 folds?

The number of folds is not a free parameter — it is fully determined by the data length and the three window settings:

```
usable bars  = total trading days − warmup
             ≈ 1587 − 252 = 1335 bars  (2020-01-01 → 2026-04-24)

n_folds      = floor((usable − train_window − test_window) / step_size) + 1
             = floor((1335 − 252 − 63) / 63) + 1
             = floor(16.19) + 1
             = 17 folds
```

Each fold covers a 3-month blind evaluation window, giving **1071 total out-of-sample bars** across the full backtest period. Adjacent in-sample windows share ~75% of their data (only 63 bars shift each time), which is normal and expected — it does not invalidate the OOS evaluation because the OOS windows themselves never overlap.

**Could we use fewer or more folds?**

| Alternative | How | Effect |
|---|---|---|
| ~7 folds | `step_size=126` (6-month steps) | Each OOS period is 3 months but spaced further apart. Less temporal granularity; aggregate Sharpe barely changes if the strategy is stationary. |
| ~35 folds | `step_size=31` (6-week steps) | OOS windows would overlap (invalid) — or `test_window` must shrink to 31 bars, making each fold too short to measure anything reliably. |
| 1 fold | Single train/test split | Classical approach. Highly sensitive to which market period falls in OOS. Cannot detect whether the strategy degrades over time. |

**What the fold stability analysis shows** (see `research/fold_analysis/`):

- Per-fold Sharpe ranges from −2.6 (Apr–Jul 2022 rate shock) to +4.3 (May–Jul 2023 bull run) — high variance is expected for 63-bar windows
- Aggregate Sharpe is reliable: bootstrap 95% CI = [0.38, 2.30], lower bound firmly positive
- No temporal decay: later folds perform as well as earlier ones (Sharpe drift +0.29)
- Top 3 folds = 49% of gains — moderate concentration, not alarming
- **17 is the natural optimum** for this data length: fewer folds sacrifice temporal granularity, more folds would require overlapping or shorter OOS windows

The high per-fold variance (CoV = 1.39) is a property of the short 63-bar OOS window, not a sign of an unstable strategy. The 1071-bar aggregate is what matters for statistical inference.

---

## Risk controls

Every order passes through `RiskManager.validate_signal()` — a 16-layer gate:

| Check | Default | Breach behaviour |
|---|---|---|
| Lock file present | — | All trading halted until lock deleted |
| Peak drawdown halt | 10% from peak | Full halt + CRITICAL alert + `trading_halted.lock` written |
| Weekly drawdown halt | 7% | All trading halted for rest of week |
| Daily drawdown halt | 3% | All trading halted for rest of day |
| Weekly drawdown reduce | 5% | Position sizes halved |
| Daily drawdown reduce | 2% | Position sizes halved |
| Max daily trades | 20 | Circuit breaker halt |
| Stop-loss mandatory | — | Trade rejected if no stop provided |
| Duplicate order | 60 s window | Trade rejected |
| Max concurrent positions | 5 | Trade rejected |
| 1% risk rule | 1% of equity / \|entry−stop\| | Size capped |
| Max single position | 15% | Trade rejected |
| Max gross exposure | 80% | Trade rejected |
| Overnight gap risk | 2% equity / (3× stop distance) | Size capped |
| Correlation check | 60-bar rolling | >0.70 halve size, >0.85 reject |
| Buying power | Account cash | Trade rejected |

---

## Order execution

Orders are placed as **limit orders** by default:

- BUY limit = mid-price × (1 + 0.1%) — slightly above to ensure fill
- SELL limit = mid-price × (1 − 0.1%)
- Cancelled after 30 seconds if unfilled
- Optional market-order retry on timeout (`retry_at_market=True`)

Each order carries a unique `client_order_id` in the format `{uuid8}-{SYMBOL}-{side}` so fills can be matched back to internal trade records.

---

## Strategy — entry, sizing, and exit

The strategy is **always long**. It never shorts and never moves entirely to cash. The entire edge is **drawdown avoidance through position sizing**: being 60% invested during a crash rather than 95% is what separates this from buy-and-hold.

### Entry

There is no traditional entry trigger. The position is opened as soon as the HMM produces a confirmed regime. After that the portfolio is rebalanced whenever the regime changes or the EMA filter flips.

### Stop loss

Every signal carries a stop computed from the current bar's ATR and EMA:

| Regime | Stop formula |
|---|---|
| Low vol | `max(price − 3×ATR,  50EMA − 0.5×ATR)` |
| Mid vol | `50EMA − 0.5×ATR` |
| High vol | `50EMA − 1.0×ATR` |

### Uncertainty discount

If the HMM posterior is below `min_confidence`, the regime is flickering, or the new state is not yet confirmed, all position sizes are halved and leverage is dropped to 1.0×.

### Rebalance filter

A rebalance is skipped when the new target weight is within `rebalance_threshold` (relative) of the current weight. Default in `balanced` set: 18%.

---

## Monitoring

**Terminal dashboard** (`monitoring/dashboard.py`):
- Refreshes every 5 seconds in a background daemon thread
- Panels: header (time, market status), regime (label, confidence, stability, leverage), portfolio (equity, cash, P&L, exposure, peak DD), open positions table, recent events log

**Structured logger** (`monitoring/logger.py`):
- JSON lines format: `{"ts": "...", "level": "INFO", "event": "trade", ...}`
- Rotating file handler: 50 MB max, 5 backup files

**Alert manager** (`monitoring/alerts.py`):
- Email via SMTP with STARTTLS (e.g. Gmail)
- Slack-compatible webhook (also works with Teams, Discord)
- Rate limiting: duplicate alerts within 15 minutes are silently dropped

Credentials for alerts are read from the `alerts:` section of `credentials.yaml` or from environment variables (`ALERT_SMTP_HOST`, `ALERT_SMTP_USER`, `ALERT_SMTP_PASSWORD`, `ALERT_RECIPIENT`, `ALERT_WEBHOOK_URL`).

---

## Live trading

To switch from paper to live:

1. Set `paper: false` in `config/credentials.yaml`
2. Replace keys with live trading keys from Alpaca
3. At startup you will be prompted to type exactly: `YES I UNDERSTAND THE RISKS`
4. The system will refuse to start if the phrase is not entered correctly

---

## Development notes

- **Python version**: 3.12 required (`hmmlearn` has no wheels for 3.13+)
- **Run scripts**: always use `py -3.12` on Windows, not `python`
- **Windows terminal**: `Console(force_terminal=True, legacy_windows=False)` avoids `UnicodeEncodeError` on cp1252 terminals
- **Test isolation**: all broker tests use `MagicMock(spec=AlpacaClient)` — no real network calls
- **Async in sync codebase**: WebSocket streams run via `asyncio.new_event_loop()` in daemon threads, keeping the main trading loop synchronous
- **hmmlearn convergence warnings**: silenced at `ERROR` level — the tiny negative deltas (~1e-5) are floating-point noise, not real divergence
