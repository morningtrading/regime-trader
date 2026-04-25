#!/usr/bin/env python3
"""
tools/report.py — Generate formatted backtest report from savedresults CSV files.

Usage:
    py -3.12 tools/report.py                        # latest backtest
    py -3.12 tools/report.py savedresults/backtest_2026-04-23_223628/
"""

from __future__ import annotations

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
from pathlib import Path
import pandas as pd
import numpy as np

# ── Resolve results directory ──────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
SAVED = ROOT / "savedresults"

if len(sys.argv) > 1:
    results_dir = Path(sys.argv[1])
    if not results_dir.is_absolute():
        results_dir = ROOT / results_dir
else:
    dirs = sorted(SAVED.glob("backtest_*/"), key=lambda p: p.name, reverse=True)
    if not dirs:
        print("No backtest results found in savedresults/")
        sys.exit(1)
    results_dir = dirs[0]

perf_path   = results_dir / "performance_summary.csv"
trades_path = results_dir / "trade_log.csv"
regime_path = results_dir / "regime_history.csv"

if not perf_path.exists():
    print(f"No performance_summary.csv in {results_dir}")
    sys.exit(1)

# ── Load data ─────────────────────────────────────────────────────────────────

perf = dict(pd.read_csv(perf_path, header=None).values)
trades = pd.read_csv(trades_path) if trades_path.exists() else pd.DataFrame()
regime_hist = pd.read_csv(regime_path, index_col=0) if regime_path.exists() else pd.DataFrame()

# ── Helpers ───────────────────────────────────────────────────────────────────

def pct(v):    return f"{float(v)*100:+.2f}%"
def dollar(v): return f"${float(v):,.0f}"
def ratio(v):  return f"{float(v):.4f}"

def box(title: str, rows: list[tuple[str, str]], width: int = 54) -> str:
    inner = width - 2
    title_pad = inner - len(title) - 3
    lines = [f"┌─ {title} {'─' * max(title_pad, 1)}┐"]
    for label, value in rows:
        if label.startswith("─"):
            lines.append(f"├{'─' * inner}┤")
        else:
            lines.append(f"│ {label:<25}│ {value:<24} │")
    lines.append(f"└{'─' * inner}┘")
    return "\n".join(lines)

# ── Reconstruct per-trade P&L ─────────────────────────────────────────────────

initial_capital = 100_000.0
final_equity    = float(perf.get("final_equity", initial_capital))
total_return    = float(perf.get("total_return", 0.0))

pnls: list[float] = []
regime_pnl: dict[str, list[float]] = {}

if not trades.empty and "delta_shares" in trades.columns and "fill_price" in trades.columns:
    group_keys = ["fold", "symbol"]
    has_regime = "regime" in trades.columns
    if has_regime:
        group_keys_r = ["fold", "symbol", "regime"]

    for (fold, sym), grp in trades.groupby(["fold", "symbol"]):
        grp = grp.sort_values("timestamp")
        pos = cost = 0.0
        for _, row in grp.iterrows():
            ds   = float(row["delta_shares"])
            fp   = float(row["fill_price"])
            slip = float(row.get("slippage_cost", 0))
            reg  = str(row.get("regime", "?")) if has_regime else "?"
            if ds > 0:
                pos += ds; cost += ds * fp + slip
            elif ds < 0 and pos > 0:
                sold = min(-ds, pos)
                alloc_cost = (sold / pos) * cost
                realized   = sold * fp - slip - alloc_cost
                pnls.append(realized)
                regime_pnl.setdefault(reg, []).append(realized)
                pos -= sold; cost -= alloc_cost

wins   = [p for p in pnls if p > 0]
losses = [p for p in pnls if p <= 0]

n_trades = int(perf.get("total_trades", len(trades)))
win_rate = float(perf.get("win_rate", len(wins) / max(len(pnls), 1)))
avg_win  = float(np.mean(wins))   if wins   else 0.0
avg_loss = float(np.mean(losses)) if losses else 0.0

win_loss_ratio = (avg_win / abs(avg_loss)) if avg_loss != 0 else 0.0
expectancy     = win_rate * avg_win + (1 - win_rate) * avg_loss if pnls else 0.0

n_wins   = round(win_rate * n_trades)
n_losses = n_trades - n_wins

# Streaks
streak_wins = streak_losses = cur_w = cur_l = 0
for p in pnls:
    if p > 0: cur_w += 1; cur_l = 0
    else:     cur_l += 1; cur_w = 0
    streak_wins   = max(streak_wins,   cur_w)
    streak_losses = max(streak_losses, cur_l)

# ── Regime distribution ───────────────────────────────────────────────────────

regime_rows: list[tuple[str, str]] = []
regime_changes = 0
if not trades.empty and "regime" in trades.columns:
    rc = trades["regime"].value_counts()
    total_t = rc.sum()
    for r, cnt in rc.sort_values(ascending=False).items():
        regime_rows.append((str(r), f"{int(cnt):,} trades ({cnt/total_t*100:.1f}%)"))
if not regime_hist.empty:
    col = regime_hist.columns[0] if not regime_hist.columns.empty else None
    if col:
        regime_changes = int((regime_hist[col] != regime_hist[col].shift()).sum() - 1)

# ── Annual volatility ─────────────────────────────────────────────────────────

equity_path = results_dir / "equity_curve.csv"
ann_vol_str = "—"
if equity_path.exists():
    eq = pd.read_csv(equity_path, index_col=0, parse_dates=True)
    if not eq.empty:
        rets = eq.iloc[:, 0].pct_change().dropna()
        ann_vol_str = f"{float(rets.std() * np.sqrt(252)):.2%}"

# ── Print report ──────────────────────────────────────────────────────────────

print(f"\n  Backtest Report — {results_dir.name}")
print(f"  Repo:    {ROOT}")
print(f"  Period:  {perf.get('start', '?')} → {perf.get('end', '?')}")
print(f"  Symbols: {perf.get('symbols', '?')}")
print()

print(box("RETURNS", [
    ("Final Equity",   f"{dollar(final_equity)} (from $100K)"),
    ("Total Return",   pct(total_return)),
    ("CAGR",           pct(perf.get("cagr", 0))),
]))
print()

print(box("RISK", [
    ("Max Drawdown",   pct(perf.get("max_drawdown", 0))),
    ("Max DD Days",    f"{int(float(perf.get('max_dd_days', 0)))} days"),
    ("Annual Vol",     ann_vol_str),
]))
print()

print(box("RISK-ADJUSTED", [
    ("Sharpe Ratio",   ratio(perf.get("sharpe", 0))),
    ("Sortino Ratio",  ratio(perf.get("sortino", 0))),
    ("Calmar Ratio",   ratio(perf.get("calmar", 0))),
    ("Profit Factor",  ratio(perf.get("profit_factor", 0))),
]))
print()

print(box("TRADING ACTIVITY", [
    ("Total Trades",     f"{n_trades:,}"),
    ("Win Rate",         f"{win_rate:.2%}  ({n_wins:,}W / {n_losses:,}L)"),
    ("Avg Winning",      f"${avg_win:+,.2f}" if avg_win else "—"),
    ("Avg Losing",       f"${avg_loss:+,.2f}" if avg_loss else "—"),
    ("Win / Loss Ratio", f"{win_loss_ratio:.2f}×" if win_loss_ratio else "—"),
    ("Expectancy/Trade", f"${expectancy:+,.2f}" if pnls else "—"),
]))
print()

print(box("TRADE STREAKS", [
    ("Max Consecutive Wins",   str(streak_wins)),
    ("Max Consecutive Losses", str(streak_losses)),
]))
print()

# ── Regime distribution + per-regime P&L ─────────────────────────────────────

if regime_rows:
    pnl_rows: list[tuple[str, str]] = []
    for r in sorted(regime_pnl):
        rp  = regime_pnl[r]
        rwr = len([x for x in rp if x > 0]) / max(len(rp), 1)
        pnl_rows.append((
            f"  {r} P&L",
            f"${sum(rp):+,.0f}  wr={rwr:.0%}  n={len(rp)}",
        ))

    combined = (
        regime_rows
        + [("Regime Changes", str(regime_changes))]
        + ([("─" * 25, "─" * 24)] if pnl_rows else [])
        + pnl_rows
    )
    print(box("REGIME DISTRIBUTION & P&L", combined))
    print()
