---
name: generate-walkforward-report
description: Turn raw backtest results into a clear, publishable performance report with honest framing. Use this skill whenever the user has backtest output files (equity_curve.csv, trade_log.csv, regime_history.csv) and wants a report, summary, writeup, tearsheet, or performance analysis. Also use when the user asks to "write this up," "make a one-pager," "share with X," "turn these results into a report," or wants to document a strategy's performance for themselves or others. The goal is a report that tells the truth — good results AND red flags.
---

# Generate a Walk-Forward Performance Report

The point of a backtest report isn't to sell the strategy. It's to help someone (including future-you) decide whether the strategy is good enough to paper trade — with clear eyes about what might go wrong. This skill produces that kind of report.

## Inputs

The backtest output directory should have:

- `equity_curve.csv` — columns: `date`, `equity`, `drawdown`, `benchmark_equity`
- `trade_log.csv` — every rebalance: `date`, `symbol`, `action`, `allocation_before`, `allocation_after`, `price`, `regime`, `confidence`
- `regime_history.csv` — `date`, `regime_id`, `regime_name`, `confidence`
- `benchmark_comparison.csv` — side-by-side metrics for this strategy vs buy-and-hold vs 200-SMA vs random

If any are missing, rerun the backtest with `--compare` flag first.

## Workflow

### Step 1 — Load and sanity check

Read all four CSVs. Verify:

- Date ranges match across files
- Equity curve has no gaps > 5 days (weekends/holidays are fine)
- Drawdown values are <= 0 (drawdowns are negative numbers)
- Trade log has at least 10 trades (otherwise the backtest is too short to say anything)

If any sanity check fails, flag it and stop.

### Step 2 — Compute the metrics

From the raw files, compute:

```
Period:                 start - end, N years
Total return:           final / initial - 1
CAGR:                   (final/initial)^(1/years) - 1
Volatility:             annualized std of daily returns
Sharpe:                 (CAGR - 4.5%) / volatility
Sortino:                (CAGR - 4.5%) / downside_std
Max drawdown:           min of drawdown column
Max DD duration:        longest consecutive underwater period in days
Calmar:                 CAGR / |max_drawdown|
Win rate:               wins / total_trades
Profit factor:          sum(wins) / sum(losses)
Best/worst month:       max/min of monthly returns
Best/worst day:         max/min of daily returns
```

Also compute for each benchmark:
- Buy-and-hold: same metrics
- 200-SMA trend: same metrics
- Random (mean of 100): mean and std of Sharpe, return, max DD

### Step 3 — Structure the report

Use exactly this structure. No fluff, no hype.

```markdown
# {Strategy Name} — Walk-Forward Backtest Report

**Period:** {start} to {end} ({N} years)
**Universe:** {symbols}
**Initial capital:** ${initial:,.0f}
**Generated:** {today}

---

## TL;DR

{3-5 sentences. State the CAGR, max DD, Sharpe. State whether it beat buy-and-hold and by how much. State ONE reason to be skeptical of the result. Do not recommend action.}

Example:
> The strategy returned 14.2% CAGR over 5 years with a 12.3% max drawdown
> and a 1.18 Sharpe ratio. Buy-and-hold returned 11.8% CAGR with a 24% max
> drawdown, so the strategy added 2.4% CAGR while halving the worst drawdown.
> However, most of the outperformance comes from two specific periods
> (Mar 2020 and Q4 2022) where the vol-based allocation exited before the
> biggest drops. Outside those periods, returns track the benchmark closely.
> This dependency on a small number of decisions is a concern.

---

## Headline Metrics

| Metric | Value |
|---|---|
| Total return | {total:.1%} |
| CAGR | {cagr:.2%} |
| Sharpe ratio | {sharpe:.2f} |
| Sortino ratio | {sortino:.2f} |
| Max drawdown | {max_dd:.2%} |
| Max DD duration | {dd_days} days |
| Calmar ratio | {calmar:.2f} |
| Volatility (ann.) | {vol:.2%} |

---

## vs Benchmarks

| Strategy | CAGR | Sharpe | Max DD | Calmar |
|---|---|---|---|---|
| **This strategy** | {...} | {...} | {...} | {...} |
| Buy-and-hold | {...} | {...} | {...} | {...} |
| 200-SMA trend | {...} | {...} | {...} | {...} |
| Random (mean of 100) | {...} ± {...} | {...} ± {...} | {...} ± {...} | — |

Bold the best value in each column.

Commentary (2-3 sentences):
- Does the strategy beat buy-and-hold on a risk-adjusted basis (Calmar)?
- Is the outperformance vs random > 2 standard deviations?
- If a trivial 200-SMA beats this strategy, what's the point of the HMM?

---

## Regime Breakdown

| Regime | % Time In | Return Contribution | Avg Confidence | Sharpe |
|---|---|---|---|---|
| ... | ... | ... | ... | ... |

Commentary:
- Which regime contributed most to returns?
- Did the defensive regime actually defend during drawdowns?
- Are any regimes so rare (< 3% time) that the stats aren't meaningful?

---

## Confidence Buckets

| Confidence | Trades | Sharpe | Win Rate | Avg P&L |
|---|---|---|---|---|
| < 50% | ... | ... | ... | ... |
| 50-60% | ... | ... | ... | ... |
| 60-70% | ... | ... | ... | ... |
| 70%+ | ... | ... | ... | ... |

This is the single most important table. If high-confidence trades don't
outperform low-confidence trades, the regime detection isn't adding value.
The strategy is just getting lucky on overall allocation.

---

## Worst-Case Scenarios

- Worst single day: {value} on {date}
- Worst week: {value} for week of {date}
- Worst month: {value} in {month}
- Longest drawdown: {days} days ({start_date} to {end_date})
- Max consecutive losing rebalances: {N}

---

## Equity Curve

{If possible, embed or reference a plot. Otherwise describe the shape:}

- Smooth and monotonic? (Suspicious — too good to be true.)
- Steady with clear drawdowns? (Normal.)
- One big spike followed by flat? (Strategy got lucky on one regime, reconsider.)
- Flat for long stretches? (May be underinvested during bull periods.)

---

## Honest Assessment

{This is the section that separates a report from a pitch. Cover:}

**Strengths**
- What the strategy genuinely did well
- Which regimes it handled correctly
- Where it beat benchmarks on risk-adjusted basis

**Concerns**
- Any metrics that are suspicious (Sharpe > 2.5, no drawdowns, etc.)
- Dependence on specific periods or events
- Regimes with too little data
- Anything overfit-looking

**What would break this strategy**
- Crypto flash crash, changed market regime, higher baseline vol, lower
  correlations, central bank policy shift — list the specific scenarios
  the strategy is NOT protected against

**Next steps**
- Recommended: more OOS data? stress tests? paper trade for N days?
- Not recommended: live deploy immediately (unless you really know what you're doing)

---

## Appendix — Test Conditions

- Slippage: {N} bps per rebalance
- Commission: ${N}
- Rebalance threshold: {N}%
- Training window: {N} days
- Test window: {N} days
- Walk-forward steps: {N}
- Look-ahead bias test: {PASS/FAIL} — MUST be PASS for results to be trusted
```

### Step 4 — Save and present

Save as `{backtest_dir}/report.md`. Also save a 1-page version as `report_short.md` with just the TL;DR, headline metrics, and honest assessment.

Show the user the output path and paste the TL;DR + headline metrics inline.

## What makes a report trustworthy

- **Honest about weaknesses.** Every strategy has them. If the report doesn't list any, the author is either blind or selling.
- **Grounded in benchmarks.** A number in isolation means nothing. 14% CAGR is great if buy-and-hold did 8%. It's bad if buy-and-hold did 18%.
- **Proportionate to OOS data.** 5 years is OK. 10 years is better. 2 years is barely enough to say anything.
- **Calls out look-ahead status.** If `test_look_ahead.py` wasn't run or didn't pass, all other metrics are meaningless.

## Red flags to surface in the honest assessment

- Sharpe ratio > 2.5 on daily-rebalanced allocation (this bot's setup) — almost always overfitting or look-ahead bias. (High-frequency strategies with intraday rebalancing can legitimately post Sharpe 3-5; calibrate this threshold to your rebalance frequency.)
- Max drawdown < 3% over 3+ years — probably didn't experience real volatility
- Win rate > 70% — check if it's from many small wins masking big losses
- All the outperformance is in one year or one regime — not robust
- The strategy has more parameters than years of data — overfit
- Backtest end date within 6 months of training date — possible survivorship
- Benchmark is the worst possible benchmark (e.g., "beat putting money under a mattress")

## Do not

- Do not sell the strategy. Give metrics and let the reader decide.
- Do not omit benchmarks.
- Do not report Sharpe alone. Always pair with max drawdown.
- Do not round away important digits (e.g., "Sharpe 2.5" vs "2.47" — be specific).
- Do not claim "overfitting-free" or "bias-free" unless the look-ahead test passed. Just report what happened.
