#!/usr/bin/env python3
"""
Script 3: asset_selection_report.py
Reads Phase 2 results and produces:
  1. Top-10 combos ranked by 6-year Sharpe vs baseline
  2. Correlation vs Sharpe scatter (text)
  3. Symbol frequency in top-50 results
  4. Pareto frontier analysis

Usage (from repo root, WSL):
    python research/asset_selection/asset_selection_report.py
    python research/asset_selection/asset_selection_report.py results/mc_results_phase2.csv
"""
from __future__ import annotations
import datetime
import socket
import subprocess
import sys
import json
from pathlib import Path
from collections import Counter
import pandas as pd
import numpy as np

OUT = Path(__file__).resolve().parent / "results"
LINE = "─" * 72

BASELINE_SYMS = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"]

# ── Load data ─────────────────────────────────────────────────────────────────
if len(sys.argv) > 1:
    csv_path = Path(sys.argv[1])
    if not csv_path.is_absolute():
        csv_path = OUT / csv_path
else:
    candidates = sorted(OUT.glob("mc_results_phase2.csv"), reverse=True)
    if not candidates:
        sys.exit("No mc_results_phase2.csv found. Run monte_carlo_asset_selection.py first.")
    csv_path = candidates[0]

# ── Run metadata header ───────────────────────────────────────────────────────
def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(Path(__file__).resolve().parent.parent.parent),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"

_W = 72
print("─" * _W)
print("  Asset Selection Report — Monte Carlo Study")
print("─" * _W)
print(f"  Source CSV   : {csv_path}")
print(f"  Report script: {Path(__file__).resolve()}")
print(f"  Timestamp    : {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
print(f"  Machine      : {socket.gethostname()}  (Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro})")
print(f"  Bot version  : {_git_hash()}")
print("─" * _W)
print()

print(f"Reading: {csv_path}\n")
df = pd.read_csv(csv_path)
df = df.dropna(subset=["sharpe"])
df["symbols_list"] = df["symbols"].apply(json.loads)

# Filter out crashed backtests (total loss = corrupted result, not a real signal)
n_raw = len(df)
df = df[df["total_return"] > -0.95].copy()
n_filtered = n_raw - len(df)
if n_filtered > 0:
    print(f"  Filtered {n_filtered} crashed combos (total_return ≤ -95%) — likely crypto blowup\n")

df = df.sort_values("sharpe", ascending=False).reset_index(drop=True)

n = len(df)
baseline_row = df[df["is_baseline"] == True]
if baseline_row.empty:
    baseline_row = df[df["symbols_list"].apply(lambda x: sorted(x) == sorted(BASELINE_SYMS))]

def section(title: str) -> None:
    print(f"\n{LINE}")
    print(f"  {title}")
    print(LINE)

# ── 1. Top-10 ranked by 6-year Sharpe ────────────────────────────────────────
section("1. TOP-10 COMBOS BY 6-YEAR SHARPE  (vs baseline)")

if not baseline_row.empty:
    br = baseline_row.iloc[0]
    print(f"\n  BASELINE: {BASELINE_SYMS}")
    print(f"    Sharpe={br['sharpe']:.3f}  Return={br['total_return']:+.1%}  "
          f"MaxDD={br['max_drawdown']:.1%}  AvgCorr={br['avg_corr']:.3f}\n")

print(f"  {'Rank':>4}  {'k':>2}  {'Sharpe':>7}  {'3yr Sharpe':>10}  {'Return':>8}  "
      f"{'MaxDD':>7}  {'AvgCorr':>8}  Symbols")
print(f"  {'─'*4}  {'─'*2}  {'─'*7}  {'─'*10}  {'─'*8}  {'─'*7}  {'─'*8}  {'─'*30}")

for rank, (_, row) in enumerate(df.head(10).iterrows(), 1):
    syms = row["symbols_list"]
    is_base = (sorted(syms) == sorted(BASELINE_SYMS))
    marker = " ◄ BASELINE" if is_base else ""
    sharpe_3yr = row.get("sharpe_3yr", float("nan"))
    s3_str = f"{sharpe_3yr:>10.3f}" if not pd.isna(sharpe_3yr) else f"{'N/A':>10}"
    print(f"  {rank:>4}  {int(row['k']):>2}  {row['sharpe']:>7.3f}  {s3_str}  "
          f"{row['total_return']:>+8.1%}  {row['max_drawdown']:>7.1%}  "
          f"{row['avg_corr']:>8.3f}  {', '.join(sorted(syms))}{marker}")

# ── 2. Pareto frontier: Sharpe vs correlation ─────────────────────────────────
section("2. DOES LOWER CORRELATION IMPROVE SHARPE?")

print(f"\n  {'CorBucket':>10}  {'#Combos':>8}  {'AvgSharpe':>10}  {'BestSharpe':>11}  Bar")
corr_buckets = pd.cut(df["avg_corr"], bins=6)
for bucket, group in df.groupby(corr_buckets, observed=True):
    avg_s = group["sharpe"].mean()
    best_s = group["sharpe"].max()
    bar = "█" * int(max(avg_s, 0) * 20) + "░" * max(0, 20 - int(max(avg_s, 0) * 20))
    print(f"  {str(bucket):>10}  {len(group):>8}  {avg_s:>10.3f}  {best_s:>11.3f}  {bar}")

print(f"""
  Interpretation:
    If Sharpe improves as corr decreases → diversification pays off.
    If flat → correlation level doesn't matter much; pick by other criteria.
    If Sharpe peaks in mid-corr range → some correlation is beneficial
    (assets in the same universe share the HMM regime signal).
""")

# ── 3. Symbol frequency in top-50 ────────────────────────────────────────────
section("3. SYMBOL FREQUENCY IN TOP-50 RESULTS")

top50 = df.head(50)
all_syms: list[str] = []
for syms in top50["symbols_list"]:
    all_syms.extend(syms)

freq = Counter(all_syms)
print(f"\n  {'Symbol':<12}  {'Count':>6}  {'%':>5}  Bar")
print(f"  {'─'*12}  {'─'*6}  {'─'*5}  {'─'*20}")
for sym, count in freq.most_common():
    pct = count / len(top50) * 100
    bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
    in_baseline = "  ★" if sym in BASELINE_SYMS else ""
    print(f"  {sym:<12}  {count:>6}  {pct:>4.0f}%  {bar}{in_baseline}")

print(f"\n  ★ = in current baseline basket")

# ── 4. Pareto dominance check ─────────────────────────────────────────────────
section("4. PARETO FRONTIER — is baseline dominated?")

# Fallback values in case baseline is not in Phase 2 results
b_sharpe: float = float("nan")
b_corr:   float = float("nan")
b_dd:     float = float("nan")
dominated = pd.DataFrame()

if not baseline_row.empty:
    b = baseline_row.iloc[0]
    b_sharpe = b["sharpe"]
    b_corr   = b["avg_corr"]
    b_dd     = b["max_drawdown"]

    # Dominated = higher Sharpe AND lower correlation AND lower MaxDD
    dominated = df[
        (df["sharpe"]       > b_sharpe) &
        (df["avg_corr"]     < b_corr)   &
        (df["max_drawdown"] < b_dd)
    ]

    print(f"""
  Baseline:  Sharpe={b_sharpe:.3f}  AvgCorr={b_corr:.3f}  MaxDD={b_dd:.1%}

  Combos that beat baseline on ALL THREE metrics
  (higher Sharpe, lower avg corr, lower MaxDD): {len(dominated)}
""")
    if len(dominated) > 0:
        print(f"  {'Rank':>4}  {'Sharpe':>7}  {'AvgCorr':>8}  {'MaxDD':>7}  Symbols")
        print(f"  {'─'*4}  {'─'*7}  {'─'*8}  {'─'*7}  {'─'*30}")
        for rank, (_, row) in enumerate(dominated.head(10).iterrows(), 1):
            d_sharpe = row["sharpe"] - b_sharpe
            print(f"  {rank:>4}  {row['sharpe']:>7.3f} (+{d_sharpe:.3f})  "
                  f"{row['avg_corr']:>8.3f}  {row['max_drawdown']:>7.1%}  "
                  f"{', '.join(sorted(row['symbols_list']))}")
    else:
        print("  → No combo strictly dominates the baseline on all three metrics.")
        print("    Check top-10 for combos that dominate on 2 out of 3.")
else:
    print(f"""
  Baseline [SPY, QQQ, AAPL, MSFT, NVDA] was not in Phase 2 results.
  It likely ranked below top-{15} in the 3-year Phase 1 screening.
  Tip: force-include baseline by re-running monte_carlo_asset_selection.py —
  the script always adds it to scored combos before Phase 1.
  Check if mc_results_phase1.csv contains the baseline row and its 3yr Sharpe.
""")

# ── 5. Verdict ────────────────────────────────────────────────────────────────
section("5. RECOMMENDATION")

top1 = df.iloc[0]
top1_syms = sorted(top1["symbols_list"])
is_top1_baseline = (top1_syms == sorted(BASELINE_SYMS))

if is_top1_baseline:
    print(f"""
  ✓ CURRENT BASKET IS OPTIMAL — no combo in the tested universe
    outperforms [SPY, QQQ, AAPL, MSFT, NVDA] on Sharpe.
    The current selection is validated by the Monte Carlo search.
""")
else:
    b_ref = f"{b_sharpe:.3f}" if not pd.isna(b_sharpe) else "N/A (not in Phase 2)"
    b_corr_ref = f"{b_corr:.3f}" if not pd.isna(b_corr) else "N/A"
    delta_str = f"{top1['sharpe'] - b_sharpe:+.3f}" if not pd.isna(b_sharpe) else "N/A"
    print(f"""
  ⚡ BETTER BASKET FOUND — top result outperforms baseline by {delta_str} Sharpe:

    Recommended: {top1_syms}
    Sharpe:      {top1['sharpe']:.3f}  (baseline: {b_ref})
    Return:      {top1['total_return']:+.1%}
    MaxDD:       {top1['max_drawdown']:.1%}
    AvgCorr:     {top1['avg_corr']:.3f}  (baseline: {b_corr_ref})

  Next steps:
    1. Verify this combo is free from hindsight bias (are any symbols
       obvious post-2020 winners chosen with hindsight?)
    2. If clean, update config/asset_groups.yaml 'stocks' group
    3. Re-run full backtest with --compare to confirm
""")

# ── Save summary ──────────────────────────────────────────────────────────────
summary_path = OUT / "asset_selection_summary.md"
with open(summary_path, "w") as f:
    f.write("# Asset Selection Monte Carlo Summary\n\n")
    f.write(f"| Field | Value |\n|---|---|\n")
    f.write(f"| Source CSV | `{csv_path.name}` |\n")
    f.write(f"| Generated  | {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')} |\n")
    f.write(f"| Machine    | {socket.gethostname()} (Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}) |\n")
    f.write(f"| Bot version| {_git_hash()} |\n\n")
    f.write("## Top 10 Combos (6-year Sharpe)\n\n")
    f.write("| Rank | k | Sharpe | Return | MaxDD | AvgCorr | Symbols |\n")
    f.write("|------|---|--------|--------|-------|---------|--------|\n")
    for rank, (_, row) in enumerate(df.head(10).iterrows(), 1):
        syms = ", ".join(sorted(row["symbols_list"]))
        marker = " **[BASELINE]**" if sorted(row["symbols_list"]) == sorted(BASELINE_SYMS) else ""
        f.write(f"| {rank} | {int(row['k'])} | {row['sharpe']:.3f} | "
                f"{row['total_return']:+.1%} | {row['max_drawdown']:.1%} | "
                f"{row['avg_corr']:.3f} | {syms}{marker} |\n")
    f.write(f"\n## Baseline\n\n")
    if not baseline_row.empty:
        br = baseline_row.iloc[0]
        f.write(f"Symbols: {BASELINE_SYMS}\n\n")
        f.write(f"| Metric | Value |\n|---|---|\n")
        f.write(f"| Sharpe | {br['sharpe']:.3f} |\n")
        f.write(f"| Total Return | {br['total_return']:+.1%} |\n")
        f.write(f"| MaxDD | {br['max_drawdown']:.1%} |\n")
        f.write(f"| Avg Pairwise Corr | {br['avg_corr']:.3f} |\n")
        f.write(f"| Dominated combos | {len(dominated) if not baseline_row.empty else 'N/A'} |\n")

print(f"\nSummary saved: {summary_path}")
