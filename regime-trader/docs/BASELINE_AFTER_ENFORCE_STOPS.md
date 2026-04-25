# Baseline After Enabling `enforce_stops=True` by Default

> **OBSOLETE — 2026-04-25.** The `enforce_stops=True` default was reverted after
> the ATR multiplier sweep ([sweep_atr_multiplier_report.md](../savedresults/sweep_atr_multiplier_report.md))
> showed that ATR per-trade stops degrade BOTH return and Sharpe (1.019 → 0.584)
> across all factor variants — meaning they cut profitable trades, not just
> losers. Investigation also revealed that **live trading never sent stops to
> the broker** (`main.py:1753` calls `submit_order`, not `submit_bracket_order`),
> so this "alignment" baseline was actually a misalignment. Risk is now managed
> exclusively by HMM regime allocation + portfolio DD halts (`max_dd_from_peak`,
> `daily_dd_halt`). Default flipped back to `False` in 4 backtester signatures
> and `main.py`. Use the canonical pre-flip baseline (174.87% / 1.019 / −32.80%)
> as the reference. The numbers below are kept for historical context only.

## Context

Following the change to flip `enforce_stops` default from `False` to `True` in
`backtest/backtester.py` (4 method signatures) and in `main.py` CLI hookup, the
backtest now exercises the same ATR-graded stop-loss logic that live trading
uses (`core/regime_strategies.py`).

This re-baseline measures the impact on the canonical run.

---

## Test Configuration

- **Command:** `py -3.12 main.py backtest --asset-group stocks --start 2020-01-01 --compare`
- **Asset group:** `stocks` (10 symbols: SPY, QQQ, AAPL, MSFT, AMZN, GOOGL, NVDA, META, TSLA, AMD)
- **Period:** 2020-01-01 → 2026-04-25 (6.3 years)
- **Walk-forward:** IS 252 / OOS 63 / step 63 → 17 OOS folds
- **Active config set:** `conservative`
- **Output:** `savedresults/backtest_2026-04-25_100755/`

---

## Results — Side-by-side vs Pre-change Baseline

| Metric              | Pre-change (enforce_stops=False) | Post-change (enforce_stops=True) | Δ |
|---------------------|---------------------------------:|---------------------------------:|---:|
| **Total Return**    | +174.87%                         | +53.54%                          | -121.33pp |
| **CAGR**            | n/a                              | +10.62%                          | — |
| **Sharpe Ratio**    | 1.019                            | 0.584                            | -0.435 |
| **Sortino Ratio**   | n/a                              | 0.659                            | — |
| **Calmar Ratio**    | n/a                              | 0.845                            | — |
| **Max Drawdown**    | -32.80%                          | -12.57%                          | +20.23pp (better) |
| **Max DD Duration** | n/a                              | 358d                             | — |
| **Annualised Vol**  | n/a                              | 10.54%                           | — |
| **Total Trades**    | 1,149                            | 534                              | -615 |
| **STOP_OUT events** | 0 (disabled)                     | 118                              | +118 |
| **Win Rate**        | n/a                              | 40.09%                           | — |
| **Profit Factor**   | n/a                              | 1.23                             | — |
| **Final Equity**    | $274,875                         | $153,545                         | -$121,330 |

**Pre-change source:** [BACKTEST_RESULTS_NEUTRAL_SKIP.md](../BACKTEST_RESULTS_NEUTRAL_SKIP.md)

---

## Stop-out Distribution per Fold (118 total)

| Fold | OOS Window              | STOP_OUT count |
|-----:|-------------------------|---------------:|
|    0 | 2022-01-10 → 2022-04-08 | 14 |
|    1 | 2022-04-11 → 2022-07-12 | 5  |
|    2 | 2022-07-13 → 2022-10-10 | 4  |
|    3 | 2022-10-11 → 2023-01-10 | 10 |
|    4 | 2023-01-11 → 2023-04-12 | 4  |
|    5 | 2023-04-13 → 2023-07-13 | 0  |
|    6 | 2023-07-14 → 2023-10-11 | 12 |
|    7 | 2023-10-12 → 2024-01-11 | 12 |
|    8 | 2024-01-12 → 2024-04-12 | 3  |
|    9 | 2024-04-15 → 2024-07-15 | 0  |
|   10 | 2024-07-16 → 2024-10-11 | 10 |
|   11 | 2024-10-14 → 2025-01-14 | 14 |
|   12 | 2025-01-15 → 2025-04-15 | 0  |
|   13 | 2025-04-16 → 2025-07-17 | 8  |
|   14 | 2025-07-18 → 2025-10-15 | 2  |
|   15 | 2025-10-16 → 2026-01-15 | 4  |
|   16 | 2026-01-16 → 2026-04-17 | 16 |

Stop activity is concentrated in 2022 bear and the 2026 Q1 pullback (fold 16:
16 stops in 3 months).

---

## Interpretation

### Risk model now matches live behavior
The headline drop in total return (−121pp) is the **honest** picture: prior
backtests overstated upside because the stop-loss layer that exists in
production was never exercised. Live trading would have hit those 118 stops
and capped the upside accordingly.

### MaxDD improvement is real
Drawdown shrinks from −32.8% → −12.6% (+20.2pp better). That's the protective
side of the same trade-off.

### Sharpe drop reflects reduced compounding
Sharpe falls 1.019 → 0.584 because the stops cut both losing AND profitable
trends — the strategy currently exits on the first ATR-graded touch, not on
trend reversal. This is expected behavior of fixed-rule stops in trending
regimes.

### Net assessment
The new baseline is the trustworthy reference for all future strategy work.
Optimisations targeted at recovering Sharpe must do so under this stop regime,
not by disabling stops to inflate paper performance.

---

## Reproducibility

- Code: commit at HEAD after `feat(backtest): default enforce_stops=True`
- Output dir: `savedresults/backtest_2026-04-25_100755/`
- Performance summary: `performance_summary.csv` in that dir
- Trade log: `trade_log.csv` (column `action` = `STOP_OUT` for stop-driven exits)
