#!/usr/bin/env python3
"""
Script 3: fold_stability_report.py
Reads fold_metrics_*.csv from Script 1 and produces a deep stability analysis:
  - Per-fold Sharpe ASCII bar chart
  - Temporal drift (first half vs second half)
  - Bootstrap confidence intervals on aggregate Sharpe
  - Fold autocorrelation
  - Return concentration (Pareto)

Usage:
    python research/fold_analysis/fold_stability_report.py
    python research/fold_analysis/fold_stability_report.py results/fold_metrics_backtest_2026-04-24_093026.csv
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# ── Resolve input file ────────────────────────────────────────────────────────
if len(sys.argv) > 1:
    csv_path = Path(sys.argv[1])
    if not csv_path.is_absolute():
        csv_path = RESULTS_DIR / csv_path
else:
    candidates = sorted(RESULTS_DIR.glob("fold_metrics_*.csv"), reverse=True)
    if not candidates:
        sys.exit("No fold_metrics_*.csv found. Run extract_fold_metrics.py first.")
    csv_path = candidates[0]

print(f"Reading: {csv_path}\n")
df = pd.read_csv(csv_path)
n = len(df)

if n == 0:
    sys.exit("Empty fold metrics file.")

sharpes  = df["sharpe"].values
returns  = df["total_return"].values
n_trades = df["n_trades"].values

# ── Helpers ───────────────────────────────────────────────────────────────────
def bar(v, vmin, vmax, width=30, char="█"):
    if np.isnan(v):
        return "  N/A"
    frac = (v - vmin) / (vmax - vmin) if vmax > vmin else 0.5
    frac = max(0.0, min(1.0, frac))
    filled = int(round(frac * width))
    return char * filled + "░" * (width - filled)

LINE = "─" * 72

# ── Section header helper ─────────────────────────────────────────────────────
def section(title):
    print(f"\n{LINE}")
    print(f"  {title}")
    print(LINE)

# ── 1. Per-fold Sharpe bar chart ──────────────────────────────────────────────
section("1. PER-FOLD SHARPE RATIO (OOS = 63 bars ≈ 3 months each)")

valid_sharpes = sharpes[~np.isnan(sharpes)]
smin, smax = (valid_sharpes.min(), valid_sharpes.max()) if len(valid_sharpes) else (0, 1)
zero_pos = int(round((-smin) / (smax - smin) * 30)) if smax > smin else 0

print(f"\n  {'Fold':>4}  {'Period':>23}  {'Sharpe':>7}  {'Bar (min={:.2f} max={:.2f})'.format(smin, smax)}")
print(f"  {'─'*4}  {'─'*23}  {'─'*7}  {'─'*32}")
for _, row in df.iterrows():
    s = row["sharpe"]
    period = f"{row['oos_start']} → {row['oos_end']}"
    b = bar(s, smin, smax)
    flag = "  ★" if s == valid_sharpes.max() else ("  ▼" if s == valid_sharpes.min() else "")
    print(f"  {int(row.fold_id):>4}  {period:>23}  {s:>7.3f}  {b}{flag}")

# ── 2. Summary statistics ─────────────────────────────────────────────────────
section("2. FOLD SHARPE DISTRIBUTION")

mean_s  = np.nanmean(sharpes)
med_s   = np.nanmedian(sharpes)
std_s   = np.nanstd(sharpes)
cov_s   = std_s / abs(mean_s) if mean_s != 0 else float("inf")
pos_folds = int((sharpes > 0).sum())
neg_folds = int((sharpes <= 0).sum())

print(f"""
  Mean Sharpe          : {mean_s:.3f}
  Median Sharpe        : {med_s:.3f}
  Std Dev of Sharpe    : {std_s:.3f}
  Coeff of Variation   : {cov_s:.2f}  {'✓ stable' if cov_s < 0.5 else ('⚠ moderate' if cov_s < 1.0 else '✗ erratic')}
  Positive folds       : {pos_folds} / {n}  ({pos_folds/n:.0%})
  Negative folds       : {neg_folds} / {n}  ({neg_folds/n:.0%})
  Best fold Sharpe     : {valid_sharpes.max():.3f}
  Worst fold Sharpe    : {valid_sharpes.min():.3f}
  Range                : {valid_sharpes.max() - valid_sharpes.min():.3f}

  Interpretation:
    CoV < 0.50  → strategy is consistent across market regimes (robust)
    CoV 0.5–1.0 → moderate variance, acceptable
    CoV > 1.0   → erratic, returns concentrated in lucky folds (risky)
""")

# ── 3. Temporal drift ─────────────────────────────────────────────────────────
section("3. TEMPORAL DRIFT — FIRST HALF vs SECOND HALF")

mid = n // 2
first_half  = df.iloc[:mid]
second_half = df.iloc[mid:]

fh_sharpe = np.nanmean(first_half["sharpe"].values)
sh_sharpe = np.nanmean(second_half["sharpe"].values)
fh_return = np.nanmean(first_half["total_return"].values)
sh_return = np.nanmean(second_half["total_return"].values)
fh_period = f"{first_half['oos_start'].iloc[0]} → {first_half['oos_end'].iloc[-1]}"
sh_period = f"{second_half['oos_start'].iloc[0]} → {second_half['oos_end'].iloc[-1]}"

drift = sh_sharpe - fh_sharpe

print(f"""
  First half  ({fh_period}):
    Avg Sharpe : {fh_sharpe:.3f}
    Avg Return : {fh_return:+.2%} per fold

  Second half ({sh_period}):
    Avg Sharpe : {sh_sharpe:.3f}
    Avg Return : {sh_return:+.2%} per fold

  Sharpe drift (2nd − 1st) : {drift:+.3f}
  Interpretation:
""")
if drift > 0.1:
    print("    ✓ Strategy IMPROVING over time — later periods perform better.")
elif drift > -0.2:
    print("    ✓ Strategy STABLE — no significant temporal decay detected.")
else:
    print("    ⚠ Strategy DECAYING — performance weaker in later folds.")
    print("      Consider: re-tune parameters with more recent IS data.")

# ── 4. Bootstrap CI on aggregate Sharpe ──────────────────────────────────────
section("4. BOOTSTRAP CONFIDENCE INTERVAL — AGGREGATE SHARPE")

valid_idx = ~np.isnan(sharpes)
valid_s   = sharpes[valid_idx]

N_BOOT = 2000
rng    = np.random.default_rng(42)
boot_means = []
for _ in range(N_BOOT):
    sample = rng.choice(valid_s, size=len(valid_s), replace=True)
    boot_means.append(sample.mean())

boot_means = np.array(boot_means)
ci_lo = np.percentile(boot_means, 2.5)
ci_hi = np.percentile(boot_means, 97.5)

print(f"""
  Observed aggregate Sharpe : {mean_s:.3f}
  Bootstrap 95% CI          : [{ci_lo:.3f}, {ci_hi:.3f}]  (width = {ci_hi-ci_lo:.3f})
  Bootstrap std error       : {boot_means.std():.3f}
  n_boot                    : {N_BOOT}
  n_folds used              : {len(valid_s)}

  Note: Folds are NOT fully independent (adjacent IS windows share 75% of data).
  The CI above treats folds as independent — true CI is slightly wider.

  Interpretation:
""")
if ci_lo > 0.5:
    print("    ✓ STRONG — CI lower bound > 0.5. Strategy is reliably above average.")
elif ci_lo > 0:
    print("    ✓ POSITIVE — CI lower bound > 0. Strategy has positive risk-adjusted edge.")
else:
    print("    ⚠ UNCERTAIN — CI crosses zero. Edge may not be statistically significant.")

# ── 5. Fold autocorrelation ───────────────────────────────────────────────────
section("5. FOLD AUTOCORRELATION (lag-1)")

if len(valid_s) >= 4:
    lag1_corr = np.corrcoef(valid_s[:-1], valid_s[1:])[0, 1]
    print(f"""
  Lag-1 autocorrelation of fold Sharpe : {lag1_corr:.3f}

  Interpretation:
    Close to 0   → folds are approximately independent (good)
    High positive → momentum in performance (trending good/bad periods)
    High negative → mean-reverting performance (alternating good/bad)
""")
    if abs(lag1_corr) < 0.3:
        print("    ✓ LOW autocorrelation — folds behave independently.")
    elif lag1_corr > 0.3:
        print("    → Positive autocorrelation — good periods cluster together.")
        print("      This inflates the apparent consistency of the strategy.")
    else:
        print("    → Negative autocorrelation — alternating good/bad folds.")
else:
    print("  Too few folds for autocorrelation calculation.")

# ── 6. Return concentration (Pareto) ─────────────────────────────────────────
section("6. RETURN CONCENTRATION — WHICH FOLDS DRIVE PROFITS?")

ret_sorted = np.sort(returns)[::-1]
cum_ret    = np.cumsum(np.maximum(ret_sorted, 0))
total_pos  = cum_ret[-1] if cum_ret[-1] > 0 else 1.0

print(f"\n  Top folds ranked by return:")
ranked = df.sort_values("total_return", ascending=False).reset_index(drop=True)
cumulative = 0.0
for i, (_, row) in enumerate(ranked.iterrows()):
    r = row["total_return"]
    if r > 0:
        cumulative += r
    pct_of_total = cumulative / max(sum(max(x,0) for x in returns), 1e-9)
    print(f"  #{i+1:>2}  Fold {int(row.fold_id):>2}  "
          f"{row['oos_start']} → {row['oos_end']}  "
          f"Return {r:>+8.2%}  cumul {pct_of_total:.0%} of gains")

top3_pct = sum(sorted([r for r in returns if r > 0], reverse=True)[:3]) / max(total_pos, 1e-9)
print(f"\n  Top 3 folds drive {top3_pct:.0%} of total positive returns.")
if top3_pct > 0.6:
    print("  ⚠ CONCENTRATED — majority of gains from 3 folds. High event risk.")
elif top3_pct > 0.45:
    print("  ⚠ MODERATE concentration. Acceptable but worth monitoring.")
else:
    print("  ✓ DISTRIBUTED — returns spread across many folds. Robust.")

# ── 7. Final verdict ──────────────────────────────────────────────────────────
section("7. FOLD COUNT VERDICT")

print(f"""
  Current configuration: {n} folds × 63 OOS bars = {n*63} total OOS observations

  Is {n} folds the right number?

  ✓ Statistical power  : {n*63} OOS bars is {'sufficient' if n*63 >= 500 else 'borderline'} for aggregate Sharpe estimation
  {'✓' if cov_s < 0.8 else '⚠'} Stability (CoV={cov_s:.2f}) : {'Consistent across folds' if cov_s < 0.5 else 'Moderate variance' if cov_s < 1.0 else 'High variance — inspect individual folds'}
  {'✓' if abs(drift) < 0.2 else '⚠'} Temporal drift ({drift:+.2f}) : {'No decay detected' if abs(drift) < 0.2 else 'Performance changing over time'}
  {'✓' if ci_lo > 0 else '⚠'} Bootstrap CI lower  : {ci_lo:.3f} ({'positive edge confirmed' if ci_lo > 0 else 'edge uncertain'})
  {'✓' if top3_pct < 0.5 else '⚠'} Return concentration : Top 3 folds = {top3_pct:.0%} of gains

  Fewer folds (step=126 → ~7 folds):
    - Each OOS = 6 months (less noise per fold)
    - Less temporal granularity (harder to detect drift)
    - Aggregate Sharpe should be similar if strategy is stationary

  More folds (step=31 → ~35 folds):
    - Requires OOS=31 bars (~6 weeks) → very noisy per-fold Sharpe
    - OR overlapping OOS → invalid walk-forward
    - Not recommended

  Recommendation: {'Keep 17 folds — configuration is sound.' if n >= 10 and cov_s < 1.0 else 'Consider wider OOS windows (test_window=126) for more stable per-fold estimates.'}
""")

# ── Save report ───────────────────────────────────────────────────────────────
report_path = RESULTS_DIR / f"fold_stability_summary_{csv_path.stem.replace('fold_metrics_','')}.md"
with open(report_path, "w") as f:
    f.write(f"# Fold Stability Summary\n\n")
    f.write(f"Source: `{csv_path.name}`\n\n")
    f.write(f"| Metric | Value |\n|---|---|\n")
    f.write(f"| n_folds | {n} |\n")
    f.write(f"| Mean Sharpe | {mean_s:.3f} |\n")
    f.write(f"| Sharpe CoV | {cov_s:.2f} |\n")
    f.write(f"| Bootstrap 95% CI | [{ci_lo:.3f}, {ci_hi:.3f}] |\n")
    f.write(f"| Temporal drift | {drift:+.3f} |\n")
    f.write(f"| Top-3 fold concentration | {top3_pct:.0%} |\n")
    f.write(f"| Positive folds | {pos_folds}/{n} |\n")

print(f"\nReport saved: {report_path}")
