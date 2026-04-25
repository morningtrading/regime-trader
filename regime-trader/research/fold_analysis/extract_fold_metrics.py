#!/usr/bin/env python3
"""
Script 1: extract_fold_metrics.py
Reconstructs per-fold performance metrics from a savedresults backtest directory.

Usage:
    python research/fold_analysis/extract_fold_metrics.py
    python research/fold_analysis/extract_fold_metrics.py savedresults/backtest_2026-04-24_093026/
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
SAVED = ROOT / "savedresults"
OUT   = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)

# ── Resolve results dir ───────────────────────────────────────────────────────
if len(sys.argv) > 1:
    results_dir = Path(sys.argv[1])
    if not results_dir.is_absolute():
        results_dir = ROOT / results_dir
else:
    dirs = sorted(SAVED.glob("backtest_*/"), key=lambda p: p.name, reverse=True)
    if not dirs:
        sys.exit("No backtest results found.")
    results_dir = dirs[0]

print(f"Reading: {results_dir}")

equity_path = results_dir / "equity_curve.csv"
trades_path = results_dir / "trade_log.csv"
perf_path   = results_dir / "performance_summary.csv"

for p in (equity_path, trades_path, perf_path):
    if not p.exists():
        sys.exit(f"Missing: {p}")

# ── Load data ─────────────────────────────────────────────────────────────────
equity = pd.read_csv(equity_path, index_col=0, parse_dates=True).iloc[:, 0]
trades = pd.read_csv(trades_path, parse_dates=["timestamp"])
perf   = dict(pd.read_csv(perf_path, header=None).values)

# ── Identify fold boundaries from trade_log ───────────────────────────────────
if "fold" not in trades.columns:
    sys.exit("trade_log.csv has no 'fold' column — cannot split by fold.")

fold_ids = sorted(trades["fold"].unique())
print(f"Found {len(fold_ids)} folds in trade_log: {fold_ids}")

# ── Per-fold metric helpers ───────────────────────────────────────────────────
RF_DAILY = 0.045 / 252

def sharpe(ret: pd.Series) -> float:
    if len(ret) < 2:
        return float("nan")
    excess = ret - RF_DAILY
    std = excess.std()
    return float(excess.mean() / std * np.sqrt(252)) if std > 0 else float("nan")

def max_dd(eq: pd.Series) -> float:
    peak = eq.expanding().max()
    dd   = (eq - peak) / peak
    return float(dd.min())

def cagr(eq: pd.Series) -> float:
    if len(eq) < 2:
        return float("nan")
    years = len(eq) / 252
    return float((eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1)

# ── Reconstruct per-fold equity from combined equity + trade timestamps ────────
rows = []
prev_end_ts = None

for fold_id in fold_ids:
    fold_trades = trades[trades["fold"] == fold_id].sort_values("timestamp")

    if fold_trades.empty:
        continue

    oos_start = fold_trades["timestamp"].min()
    oos_end   = fold_trades["timestamp"].max()

    # Slice equity curve to this fold's OOS period
    if prev_end_ts is not None:
        mask = (equity.index >= prev_end_ts) & (equity.index <= oos_end)
    else:
        mask = equity.index <= oos_end

    fold_equity = equity[mask]
    if fold_equity.empty:
        continue

    fold_ret = fold_equity.pct_change().dropna()
    n_trades  = len(fold_trades)
    win_count = int((fold_ret > 0).sum())
    wr        = win_count / len(fold_ret) if len(fold_ret) > 0 else float("nan")

    rows.append({
        "fold_id":     int(fold_id),
        "oos_start":   oos_start.date(),
        "oos_end":     oos_end.date(),
        "n_bars":      len(fold_equity),
        "total_return": float(fold_equity.iloc[-1] / fold_equity.iloc[0] - 1),
        "cagr":        cagr(fold_equity),
        "sharpe":      sharpe(fold_ret),
        "max_drawdown":max_dd(fold_equity),
        "n_trades":    n_trades,
        "win_rate":    wr,
    })

    prev_end_ts = oos_end

df = pd.DataFrame(rows)
out_path = OUT / f"fold_metrics_{results_dir.name}.csv"
df.to_csv(out_path, index=False)

# ── Print summary table ───────────────────────────────────────────────────────
print(f"\n{'Fold':>4}  {'OOS Start':>11}  {'OOS End':>11}  {'Bars':>5}  "
      f"{'Return':>8}  {'Sharpe':>7}  {'MaxDD':>7}  {'Trades':>7}")
print("─" * 74)
for _, r in df.iterrows():
    print(f"{int(r.fold_id):>4}  {str(r.oos_start):>11}  {str(r.oos_end):>11}  "
          f"{int(r.n_bars):>5}  {r.total_return:>+8.2%}  {r.sharpe:>7.3f}  "
          f"{r.max_drawdown:>7.2%}  {int(r.n_trades):>7}")

print(f"\nAggregate (from performance_summary.csv):")
print(f"  Total Return : {float(perf.get('total_return',0)):+.2%}")
print(f"  Sharpe       : {float(perf.get('sharpe',0)):.3f}")
print(f"  MaxDD        : {float(perf.get('max_drawdown',0)):.2%}")
print(f"  Folds        : {len(df)}")
print(f"\nSaved: {out_path}")
