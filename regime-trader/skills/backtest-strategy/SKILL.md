---
name: backtest-strategy
description: Run a rigorous walk-forward backtest on any trading strategy in this repo and produce a proper performance report. Use this skill whenever the user wants to backtest, validate, evaluate, measure, or test a strategy's historical performance — including phrases like "how would this have done," "let's see the numbers," "run it on the last N years," "check the Sharpe," or any request that involves historical simulation. Also use when the user asks to compare strategies, benchmark against buy-and-hold, or sanity-check a strategy before paper trading.
---

# Backtest a Strategy

This skill runs a walk-forward backtest on a strategy in this project and produces a complete report. It is deliberately opinionated — the point is to avoid the three mistakes that make most retail backtests worthless: look-ahead bias, unrealistic fills, and overfitting to the in-sample period.

## When to use this

Use whenever a strategy needs historical validation before paper trading. Do not trust a strategy that hasn't been through this.

## Inputs you need from the user

Ask if not provided:

1. **Strategy name or file** — which strategy to test (e.g., `LowVolBullStrategy`, or path to a custom one)
2. **Symbols** — tickers to backtest (default: SPY if not specified)
3. **Date range** — start and end (default: last 5 years)
4. **Initial capital** — default $100,000

## Workflow

### Step 1 — Sanity check the strategy file

Before running anything, open the strategy file and confirm:

- No `model.predict()` calls (that's Viterbi, look-ahead bias)
- All feature computations use rolling windows, not full-dataset transforms
- No use of future data — scan for `.shift(-N)`, `.rolling(...).mean().shift(-`, or any negative shifts
- Stop losses are defined on every signal

If any of these fail, STOP and tell the user. Do not run the backtest until they're fixed — the results will be meaningless.

### Step 2 — Configure the walk-forward

Use `backtest/backtester.py` with these defaults unless user overrides:

- In-sample window: 252 trading days (1 year) — for training/fitting
- Out-of-sample window: 126 trading days (6 months) — for evaluation only
- Step: 126 days
- Slippage: 5 bps per rebalance
- Rebalance threshold: 10% (don't churn on tiny allocation changes)
- Commission: $0 (Alpaca is commission-free)

### Step 3 — Run the backtest

Find the project's backtest entry point — usually `main.py backtest`, `python -m backtest`, or a script under `backtest/`. If the entry point doesn't exist, tell the user and ask how they normally run backtests.

Handle dependencies: if imports fail (hmmlearn, alpaca-py, pandas, ta, etc.), check `requirements.txt` and install what's missing using the project's package manager (`pip`, `uv`, `poetry`, whichever is in use).

Run with these parameters unless the user overrides:

```bash
python main.py backtest \
    --strategy {strategy_name} \
    --symbols {symbols} \
    --start {start_date} \
    --end {end_date} \
    --compare
```

The `--compare` flag runs buy-and-hold, 200-SMA trend following, and a random-entry benchmark at the same allocation frequency. Always include it. A strategy that doesn't beat buy-and-hold isn't a strategy.

If the backtest takes more than a minute without output, check for infinite loops (usually an HMM failing to converge). If it errors, diagnose — don't just retry.

### Step 4 — Produce the report

Output a markdown report with exactly these sections, in this order:

**1. Headline metrics**
- Total return, CAGR
- Sharpe ratio (annualized, risk-free rate 4.5%)
- Sortino ratio
- Max drawdown (% and duration in trading days)
- Calmar ratio

**2. Benchmark comparison table**

| Strategy | Total Return | CAGR | Sharpe | Max DD | Calmar |
|---|---|---|---|---|---|
| This strategy | ... | ... | ... | ... | ... |
| Buy & hold | ... | ... | ... | ... | ... |
| 200 SMA trend | ... | ... | ... | ... | ... |
| Random (mean of 100) | ... | ... | ... | ... | ... |

**3. Regime-specific performance**

Break down performance by detected regime. This proves each regime's allocation is working as intended. If the high-vol regime has a higher drawdown contribution than its time-in-regime share, the defensive strategy isn't defensive enough.

**4. Confidence-bucketed performance**

Group trades by regime confidence (<50%, 50-60%, 60-70%, 70%+) and show Sharpe and win rate for each. High-confidence trades SHOULD outperform low-confidence trades. If they don't, the HMM isn't adding value — the strategy is just getting lucky on the overall allocation.

**5. Worst-case scenarios**
- Worst single day, worst week, worst month
- Longest drawdown duration
- Max consecutive losing trades

**6. Honest assessment**

Close the report with 3-5 sentences of honest assessment. Do not sell the strategy. Flag anything suspicious:
- If Sharpe > 2.5 on a daily-rebalanced allocation strategy (this bot's setup), that is usually a sign of look-ahead bias, overfitting, or survivorship bias. Say so. (Higher-frequency strategies can legitimately post higher Sharpes — the 2.5 threshold is specific to daily/weekly rebalancing on liquid equities.)
- If max drawdown is under 5%, the backtest is probably too short or the strategy never hit a real regime change.
- If the strategy beats buy-and-hold by more than 5% CAGR, explain WHY in mechanical terms — what did it do differently? If you can't explain the mechanism, it's probably overfit.

### Step 5 — Output files

Save to `backtest/results/{strategy_name}_{YYYYMMDD}/`:
- `report.md` — the markdown report above
- `equity_curve.csv` — daily equity values
- `trade_log.csv` — every rebalance
- `regime_history.csv` — regime at each bar
- `benchmark_comparison.csv` — side-by-side with benchmarks

Tell the user where the files are and show the headline metrics + honest assessment inline in chat.

## Common failures to watch for

- **Backtest Sharpe too good to be true**: look-ahead bias. Re-run `pytest tests/test_look_ahead.py -v` and inspect the feature code.
- **Backtest returns differ run-to-run**: non-determinism. Set random seeds in HMM init.
- **Strategy beats everything in-sample, loses out-of-sample**: textbook overfitting. Simplify the strategy or use fewer parameters.
- **No trades in certain regimes**: the regime mapping may be broken. Check `StrategyOrchestrator.update_regime_infos()`.

## Do not

- Do not skip the look-ahead test before running the backtest.
- Do not omit `--compare`. A strategy without benchmarks is a story, not a result.
- Do not report Sharpe without also reporting max drawdown. One without the other is a lie.
- Do not tell the user the strategy is "good" or "ready for live." Give them metrics and let them decide.
