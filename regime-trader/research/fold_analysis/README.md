# Fold Count Analysis

Research scripts to understand whether 17 walk-forward folds is the right number,
what fold count tells us about parameter robustness, and whether returns are
concentrated in a few lucky periods.

**Zero changes to the main codebase.** These scripts only read from `savedresults/`
and write to `research/fold_analysis/results/`.

---

## The Question

Current config: `train_window=252`, `test_window=63`, `step_size=63` → **17 OOS folds**

Each fold = 63 bars ≈ 3 months of blind evaluation. Is this optimal?

---

## Scripts

### Script 1 — `extract_fold_metrics.py`
Reconstructs per-fold Sharpe, return, MaxDD from any `savedresults/backtest_*/`.

```bash
# Run from repo root (WSL)
python research/fold_analysis/extract_fold_metrics.py
# or specify a backtest directory
python research/fold_analysis/extract_fold_metrics.py savedresults/backtest_2026-04-24_093026/
```

Output: `results/fold_metrics_<backtest_name>.csv`

### Script 2 — `fold_count_experiment.py`
Reruns the full backtest with step_size ∈ {21, 42, 63, 126, 252} to measure
how aggregate results change with fold count.

```bash
# Run from repo root (WSL — needs Alpaca API credentials)
python research/fold_analysis/fold_count_experiment.py
```

Output: `results/step_sensitivity.csv`

### Script 3 — `fold_stability_report.py`
Deep analysis of fold-by-fold stability, temporal drift, bootstrap CI,
and return concentration.

```bash
# Run after Script 1
python research/fold_analysis/fold_stability_report.py
```

Output: printed report + `results/fold_stability_summary_<name>.md`

---

## Interpretation Guide

| CoV (Sharpe) | Meaning |
|---|---|
| < 0.50 | Stable — strategy consistent across regimes |
| 0.50–1.0 | Moderate variance — acceptable |
| > 1.0 | Erratic — returns concentrated in lucky folds |

| Bootstrap CI lower bound | Meaning |
|---|---|
| > 0.5 | Strong edge, statistically confirmed |
| > 0.0 | Positive edge, likely real |
| ≤ 0.0 | Edge uncertain — could be luck |

| Top-3 fold concentration | Meaning |
|---|---|
| < 45% | Distributed — robust |
| 45–60% | Moderate — acceptable |
| > 60% | Concentrated — high event risk |
