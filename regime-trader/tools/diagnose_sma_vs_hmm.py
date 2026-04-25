"""
diagnose_sma_vs_hmm.py
----------------------
Diagnostic: why does SMA-200 outperform the HMM regime strategy on Stocks?

Reads the most recent Stocks backtest result from savedresults/, fetches SPY
price data, and produces:
  - A 3-panel figure saved to savedresults/
  - A summary table printed to terminal
"""

from __future__ import annotations

import sys
import os
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from broker.alpaca_client import AlpacaClient, parse_timeframe
from alpaca.data.requests import StockBarsRequest

# ── config ────────────────────────────────────────────────────────────────────
RESULTS_DIR  = ROOT / "savedresults"
STOCKS_SYMS  = ["SPY","QQQ","AAPL","MSFT","AMZN","GOOGL","NVDA","META","TSLA","AMD"]
MARKET_SYM   = "SPY"
SMA_PERIOD   = 200
ROLLING_AGMT = 63   # bars for rolling agreement rate

REGIME_COLORS = {
    "CRASH":       "#d62728",
    "STRONG_BEAR": "#e07070",
    "BEAR":        "#f5a623",
    "WEAK_BEAR":   "#f5d07a",
    "NEUTRAL":     "#aec7e8",
    "WEAK_BULL":   "#98df8a",
    "BULL":        "#2ca02c",
    "STRONG_BULL": "#1a7a1a",
    "EUPHORIA":    "#17becf",
}
BULL_LABELS  = {"BULL", "EUPHORIA", "STRONG_BULL", "WEAK_BULL"}
BEAR_LABELS  = {"CRASH", "BEAR", "STRONG_BEAR", "WEAK_BEAR"}

LABEL_ORDER = [
    "CRASH","STRONG_BEAR","BEAR","WEAK_BEAR",
    "NEUTRAL",
    "WEAK_BULL","BULL","STRONG_BULL","EUPHORIA",
]


# ── 1. Find most recent Stocks backtest result ────────────────────────────────

def find_latest_stocks_result() -> Path:
    candidates = []
    for d in sorted(RESULTS_DIR.iterdir()):
        if not d.is_dir() or not d.name.startswith("backtest_"):
            continue
        perf = d / "performance_summary.csv"
        if not perf.exists():
            continue
        df = pd.read_csv(perf, index_col=0, header=None)
        syms_row = df.loc["symbols"].iloc[0] if "symbols" in df.index else ""
        if "SPY" in syms_row and "AAPL" in syms_row:
            candidates.append(d)
    if not candidates:
        raise FileNotFoundError("No Stocks backtest result found in savedresults/")
    return candidates[-1]


result_dir = find_latest_stocks_result()
print(f"Loading: {result_dir.name}")

equity_df  = pd.read_csv(result_dir / "equity_curve.csv",  index_col=0, parse_dates=True)
regime_df  = pd.read_csv(result_dir / "regime_history.csv", index_col=0, parse_dates=True)
perf_df    = pd.read_csv(result_dir / "performance_summary.csv", index_col=0, header=None)

equity_s = equity_df.iloc[:, 0].sort_index()
regime_s = regime_df.iloc[:, 0].sort_index()

oos_start = equity_s.index[0]
oos_end   = equity_s.index[-1]
print(f"OOS period: {oos_start.date()} → {oos_end.date()}  ({len(equity_s)} bars)")

# per-regime P&L if available
regime_pnl: dict = {}
for idx in perf_df.index:
    if str(idx).startswith("regime_pnl_"):
        label = str(idx).replace("regime_pnl_", "")
        regime_pnl[label] = float(perf_df.loc[idx].iloc[0])


# ── 2. Fetch SPY price history (SMA-200 needs 200 bars of pre-history) ────────

client = AlpacaClient()
client.connect(skip_live_confirm=True)
data_client = client._data_client
feed        = client.data_feed

fetch_start = (oos_start - pd.Timedelta(days=int(SMA_PERIOD * 1.6 * 7 / 5))).strftime("%Y-%m-%d")
fetch_end   = (oos_end   + pd.Timedelta(days=2)).strftime("%Y-%m-%d")

req = StockBarsRequest(
    symbol_or_symbols=MARKET_SYM,
    timeframe=parse_timeframe("1Day"),
    start=fetch_start,
    end=fetch_end,
    feed=feed,
)
bars = data_client.get_stock_bars(req)
spy_df = bars.df.xs(MARKET_SYM, level="symbol")["close"]
spy_df.index = pd.to_datetime([ts.date() for ts in spy_df.index])
spy_df = spy_df.sort_index()
print(f"SPY fetched: {spy_df.index[0].date()} → {spy_df.index[-1].date()}  ({len(spy_df)} bars)")


# ── 3. Compute signals ────────────────────────────────────────────────────────

sma200    = spy_df.rolling(SMA_PERIOD).mean()
sma_signal = (spy_df > sma200).astype(int)   # 1 = long, 0 = flat

# restrict to OOS window
spy_oos  = spy_df.reindex(equity_s.index).ffill()
sma_oos  = sma200.reindex(equity_s.index).ffill()
sma_sig  = sma_signal.reindex(equity_s.index).ffill()

# HMM bullish flag: 1 if BULL/EUPHORIA/STRONG_BULL/WEAK_BULL, 0 otherwise
hmm_bull = regime_s.map(lambda r: 1 if r in BULL_LABELS else 0)

# agreement: both long, or both not-long
agree = (sma_sig == hmm_bull).astype(int)

rolling_agree = agree.rolling(ROLLING_AGMT).mean()


# ── 4. Figure ─────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(16, 12))
gs  = GridSpec(3, 1, figure=fig, hspace=0.08, height_ratios=[3, 1, 1])

ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1], sharex=ax1)
ax3 = fig.add_subplot(gs[2], sharex=ax1)

# ── Panel 1: SPY price + SMA-200 + regime shading ────────────────────────────
all_labels = regime_s.unique()
prev_date  = equity_s.index[0]
prev_label = regime_s.iloc[0]

for i in range(1, len(regime_s)):
    cur_date  = regime_s.index[i]
    cur_label = regime_s.iloc[i]
    if cur_label != prev_label or i == len(regime_s) - 1:
        color = REGIME_COLORS.get(prev_label, "#cccccc")
        ax1.axvspan(prev_date, cur_date, alpha=0.25, color=color, linewidth=0)
        prev_date  = cur_date
        prev_label = cur_label

ax1.plot(spy_oos.index, spy_oos.values, color="#1f77b4", linewidth=1.2, label="SPY close", zorder=3)
ax1.plot(sma_oos.index, sma_oos.values, color="darkorange", linewidth=1.5,
         linestyle="--", label="SMA-200", zorder=4)
ax1.set_ylabel("SPY Price ($)")
ax1.legend(loc="upper left", fontsize=9)
ax1.set_title("SPY vs SMA-200 — HMM Regime Overlay (Stocks OOS)", fontsize=12, fontweight="bold")

legend_patches = [
    mpatches.Patch(color=REGIME_COLORS.get(l, "#ccc"), alpha=0.5, label=l)
    for l in LABEL_ORDER if l in all_labels
]
ax1.legend(handles=legend_patches + [
    plt.Line2D([0],[0], color="#1f77b4", lw=1.2, label="SPY close"),
    plt.Line2D([0],[0], color="darkorange", lw=1.5, ls="--", label="SMA-200"),
], loc="upper left", fontsize=8, ncol=2)

# ── Panel 2: SMA signal vs HMM bullish flag ───────────────────────────────────
ax2.step(sma_sig.index, sma_sig.values,  where="post", color="darkorange",
         linewidth=1.4, label="SMA-200 long")
ax2.step(hmm_bull.index, hmm_bull.values + 0.04, where="post", color="#2ca02c",
         linewidth=1.4, linestyle="--", label="HMM bullish")
ax2.set_yticks([0, 1])
ax2.set_yticklabels(["Flat", "Long"])
ax2.set_ylabel("Signal")
ax2.legend(loc="upper left", fontsize=8)
ax2.set_ylim(-0.15, 1.25)

# ── Panel 3: Rolling agreement rate ──────────────────────────────────────────
ax3.plot(rolling_agree.index, rolling_agree.values * 100,
         color="#9467bd", linewidth=1.3, label=f"{ROLLING_AGMT}-bar agreement %")
ax3.axhline(50, color="grey", linewidth=0.8, linestyle=":")
ax3.axhline(rolling_agree.mean() * 100, color="#9467bd", linewidth=0.8,
            linestyle="--", alpha=0.6, label=f"Mean {rolling_agree.mean()*100:.1f}%")
ax3.fill_between(rolling_agree.index, 50, rolling_agree.values * 100,
                 where=rolling_agree.values < 0.5, alpha=0.2, color="red",
                 label="Disagreement zone")
ax3.set_ylabel("Agreement %")
ax3.set_ylim(0, 105)
ax3.legend(loc="lower left", fontsize=8)
ax3.set_xlabel("Date")

plt.setp(ax1.get_xticklabels(), visible=False)
plt.setp(ax2.get_xticklabels(), visible=False)

ts  = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
out = RESULTS_DIR / f"sma_vs_hmm_diagnosis_{ts}.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nFigure saved → {out}")


# ── 5. Summary table ──────────────────────────────────────────────────────────

total_bars   = len(agree)
overall_agmt = agree.mean() * 100

print("\n" + "=" * 62)
print("  SMA-200 vs HMM REGIME DIAGNOSTIC — Stocks OOS")
print("=" * 62)
print(f"  OOS window   : {oos_start.date()} → {oos_end.date()}")
print(f"  Total bars   : {total_bars}")
print(f"  Overall agreement (both long or both flat) : {overall_agmt:.1f}%")
print()

# per-regime breakdown
print(f"  {'Regime':<14} {'Bars':>5}  {'% of OOS':>8}  {'SMA long%':>10}  {'HMM bull%':>10}  {'Agree%':>8}  {'Regime P&L':>12}")
print("  " + "-" * 72)

sorted_labels = sorted(regime_s.unique(),
                       key=lambda l: LABEL_ORDER.index(l) if l in LABEL_ORDER else 99)

for label in sorted_labels:
    mask        = regime_s == label
    n           = mask.sum()
    pct_oos     = n / total_bars * 100
    sma_long_pct = sma_sig[mask].mean() * 100
    hmm_bull_pct = hmm_bull[mask].mean() * 100
    agree_pct   = agree[mask].mean() * 100
    pnl_str     = f"${regime_pnl[label]:>+10,.0f}" if label in regime_pnl else "       n/a"
    print(f"  {label:<14} {n:>5}  {pct_oos:>7.1f}%  {sma_long_pct:>9.1f}%  {hmm_bull_pct:>9.1f}%  {agree_pct:>7.1f}%  {pnl_str}")

print()

# bars where they disagree — SMA long but HMM not
sma_long_hmm_flat = ((sma_sig == 1) & (hmm_bull == 0)).sum()
sma_flat_hmm_long = ((sma_sig == 0) & (hmm_bull == 1)).sum()
print(f"  Disagreement breakdown:")
print(f"    SMA long  + HMM flat/bear : {sma_long_hmm_flat:>4} bars  ({sma_long_hmm_flat/total_bars*100:.1f}%)")
print(f"    SMA flat  + HMM bullish   : {sma_flat_hmm_long:>4} bars  ({sma_flat_hmm_long/total_bars*100:.1f}%)")
print()

# which regimes are the SMA staying long through
for label in sorted_labels:
    mask = (regime_s == label) & (sma_sig == 1) & (hmm_bull == 0)
    n = mask.sum()
    if n > 0:
        print(f"    SMA stayed long during HMM {label:<14}: {n:>4} bars")

print()
print(f"  Interpretation hint:")
print(f"    SMA-200 ignores regime — it stays long whenever price > SMA.")
print(f"    Every bar the HMM pulled back allocation while SMA stayed")
print(f"    long and SPY kept rising is a bar the HMM gave up return.")
print("=" * 62)
