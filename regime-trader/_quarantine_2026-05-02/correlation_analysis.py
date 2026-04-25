"""
tools/correlation_analysis.py — Cross-asset group correlation analysis.

Fetches daily prices for all 3 groups (stocks, crypto, indices), builds
equal-dollar-weight group indices, then produces a 4-panel chart:

  1. Cumulative returns of the 3 group indices (normalised to $1)
  2. Full-period Pearson correlation heatmap
  3. Rolling 63-day (quarterly) pairwise correlations
  4. Per-era correlation comparison (pre-2022 / 2022 crash / 2023+ recovery)

Usage:
    py -3.12 tools/correlation_analysis.py
    py -3.12 tools/correlation_analysis.py --start 2020-01-01 --end 2026-04-17
    py -3.12 tools/correlation_analysis.py --no-show   # save only, don't open window
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# ── path bootstrap ─────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from dotenv import load_dotenv

load_dotenv()


# ── Asset groups ───────────────────────────────────────────────────────────────

GROUPS = {
    "Stocks":  ["SPY", "QQQ", "AAPL", "MSFT", "AMZN", "GOOGL", "NVDA", "META", "TSLA", "AMD"],
    "Crypto":  ["BTC/USD", "ETH/USD", "SOL/USD", "AVAX/USD", "DOGE/USD",
                "LTC/USD", "LINK/USD", "UNI/USD"],
    "Indices": ["SPY", "QQQ", "DIA", "IWM", "GLD", "TLT", "EFA", "EEM", "VNQ", "USO"],
}

GROUP_COLOURS = {
    "Stocks":  "#4C9BE8",   # blue
    "Crypto":  "#F5A623",   # amber
    "Indices": "#7ED321",   # green
}

PAIR_COLOURS = {
    ("Stocks",  "Crypto"):  "#E8784C",
    ("Stocks",  "Indices"): "#9B59B6",
    ("Crypto",  "Indices"): "#2ECC71",
}

ERAS = {
    "Pre-crash\n(2020–2021)": ("2020-01-01", "2021-12-31"),
    "Crash\n(2022)":          ("2022-01-01", "2022-12-31"),
    "Recovery\n(2023–2024)":  ("2023-01-01", "2024-12-31"),
    "2025+":                  ("2025-01-01", "2026-12-31"),
}


# ── Data fetching ──────────────────────────────────────────────────────────────

def _fetch_prices(symbols, start, end, api_key, secret_key):
    from alpaca.data.timeframe import TimeFrame

    crypto_syms = [s for s in symbols if "/" in s]
    stock_syms  = [s for s in symbols if "/" not in s]
    frames = []

    if stock_syms:
        from alpaca.data import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.enums import Adjustment
        from alpaca.data.enums import DataFeed
        sc = StockHistoricalDataClient(api_key, secret_key)
        req = StockBarsRequest(
            symbol_or_symbols=stock_syms,
            timeframe=TimeFrame.Day,
            start=start, end=end,
            adjustment=Adjustment.ALL,
            feed=DataFeed.IEX,
        )
        df = sc.get_stock_bars(req).df
        close = df["close"].unstack(level="symbol")
        close.index = pd.to_datetime(close.index).normalize().tz_localize(None)
        frames.append(close)

    if crypto_syms:
        from alpaca.data.historical.crypto import CryptoHistoricalDataClient
        from alpaca.data.requests import CryptoBarsRequest
        cc = CryptoHistoricalDataClient(api_key, secret_key)
        req = CryptoBarsRequest(
            symbol_or_symbols=crypto_syms,
            timeframe=TimeFrame.Day,
            start=start, end=end,
        )
        df = cc.get_crypto_bars(req).df
        close = df["close"].unstack(level="symbol")
        close.index = pd.to_datetime(close.index).normalize().tz_localize(None)
        frames.append(close)

    if not frames:
        raise ValueError("No data fetched.")

    prices = pd.concat(frames, axis=1).sort_index()
    # align to business days only (crypto has weekend data — drop it so returns align)
    prices = prices[prices.index.dayofweek < 5]
    prices = prices.ffill().dropna(how="all")
    return prices


# ── Group index construction ───────────────────────────────────────────────────

def build_group_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    For each group, compute an equal-dollar-weight daily return series.

    Each constituent is weighted 1/N (where N = symbols available that day).
    Missing symbols are excluded from the average on any given day.
    """
    group_rets = {}
    for name, syms in GROUPS.items():
        avail = [s for s in syms if s in prices.columns]
        if not avail:
            print(f"  WARNING: no symbols for group '{name}' in price data")
            continue
        rets = prices[avail].pct_change()
        # equal-weight average, ignoring NaNs
        group_rets[name] = rets.mean(axis=1)

    return pd.DataFrame(group_rets).dropna(how="all")


def cumulative_returns(returns: pd.Series) -> pd.Series:
    return (1 + returns.fillna(0)).cumprod()


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_correlation_analysis(
    group_returns: pd.DataFrame,
    start: str,
    end: str,
    rolling_window: int = 63,
    save_path: Path | None = None,
    show: bool = True,
) -> None:

    groups = list(group_returns.columns)
    pairs  = [(g1, g2) for i, g1 in enumerate(groups)
              for g2 in groups[i+1:]]

    fig = plt.figure(figsize=(18, 14), facecolor="#0F1117")
    fig.suptitle(
        f"Cross-Asset Group Correlation Analysis  ·  {start} → {end}",
        fontsize=15, color="white", fontweight="bold", y=0.98,
    )

    gs = fig.add_gridspec(
        3, 3,
        hspace=0.45, wspace=0.35,
        top=0.94, bottom=0.06, left=0.06, right=0.97,
    )

    ax_cum  = fig.add_subplot(gs[0, :])      # row 0: full width — cumulative returns
    ax_heat = fig.add_subplot(gs[1, 0])      # row 1 left  — heatmap
    ax_roll = fig.add_subplot(gs[1, 1:])     # row 1 right — rolling correlation
    ax_era  = fig.add_subplot(gs[2, :])      # row 2: full width — era comparison

    _style_ax(ax_cum)
    _style_ax(ax_heat)
    _style_ax(ax_roll)
    _style_ax(ax_era)

    # ── Panel 1: Cumulative returns ───────────────────────────────────────────
    for name in groups:
        cum = cumulative_returns(group_returns[name])
        ax_cum.plot(cum.index, cum.values,
                    color=GROUP_COLOURS[name], linewidth=1.8, label=name)

    ax_cum.set_title("Equal-Weight Group Index — Cumulative Return (base = 1.0)",
                     color="white", fontsize=11, pad=8)
    ax_cum.set_ylabel("Cumulative return", color="#AAAAAA", fontsize=9)
    ax_cum.axhline(1.0, color="#444444", linewidth=0.8, linestyle="--")
    ax_cum.legend(loc="upper left", framealpha=0.2, labelcolor="white",
                  fontsize=10, edgecolor="#444444")
    ax_cum.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_cum.xaxis.set_major_locator(mdates.YearLocator())

    # shade eras
    era_colours = ["#1a1a2e", "#2e1a1a", "#1a2e1a", "#1a1a2e"]
    for (era_label, (es, ee)), ec in zip(ERAS.items(), era_colours):
        ax_cum.axvspan(pd.Timestamp(es), min(pd.Timestamp(ee), group_returns.index[-1]),
                       alpha=0.15, color=ec, zorder=0)

    # ── Panel 2: Full-period correlation heatmap ──────────────────────────────
    corr = group_returns.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)   # show lower triangle + diag

    sns.heatmap(
        corr,
        ax=ax_heat,
        annot=True,
        fmt=".2f",
        cmap=sns.diverging_palette(250, 10, as_cmap=True),
        vmin=-1, vmax=1,
        center=0,
        linewidths=1,
        linecolor="#0F1117",
        annot_kws={"size": 13, "weight": "bold", "color": "white"},
        cbar_kws={"shrink": 0.8},
    )
    ax_heat.set_title("Full-Period Pearson Correlation", color="white",
                      fontsize=11, pad=8)
    ax_heat.set_xticklabels(ax_heat.get_xticklabels(), color="white", fontsize=10)
    ax_heat.set_yticklabels(ax_heat.get_yticklabels(), color="white", fontsize=10,
                             rotation=0)
    ax_heat.tick_params(colors="white")

    # colorbar text
    cbar = ax_heat.collections[0].colorbar
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white", fontsize=8)

    # ── Panel 3: Rolling correlation ──────────────────────────────────────────
    ax_roll.set_title(f"Rolling {rolling_window}-Day Pairwise Correlation",
                      color="white", fontsize=11, pad=8)
    ax_roll.axhline(0, color="#444444", linewidth=0.8, linestyle="--")

    for (g1, g2) in pairs:
        colour = PAIR_COLOURS.get((g1, g2)) or PAIR_COLOURS.get((g2, g1)) or "#FFFFFF"
        roll_corr = group_returns[g1].rolling(rolling_window).corr(group_returns[g2])
        ax_roll.plot(roll_corr.index, roll_corr.values,
                     color=colour, linewidth=1.5,
                     label=f"{g1} / {g2}")
        # fill between 0 and the line to make spikes visible
        ax_roll.fill_between(roll_corr.index, roll_corr.values, 0,
                             color=colour, alpha=0.07)

    ax_roll.set_ylabel("Pearson r", color="#AAAAAA", fontsize=9)
    ax_roll.set_ylim(-0.5, 1.05)
    ax_roll.legend(loc="lower right", framealpha=0.2, labelcolor="white",
                   fontsize=9, edgecolor="#444444")
    ax_roll.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_roll.xaxis.set_major_locator(mdates.YearLocator())

    # ── Panel 4: Per-era correlation bars ─────────────────────────────────────
    ax_era.set_title("Pairwise Correlation by Market Era",
                     color="white", fontsize=11, pad=8)

    era_labels = list(ERAS.keys())
    n_eras  = len(era_labels)
    n_pairs = len(pairs)
    bar_w   = 0.18
    x       = np.arange(n_eras)

    for i, (g1, g2) in enumerate(pairs):
        colour = PAIR_COLOURS.get((g1, g2)) or PAIR_COLOURS.get((g2, g1)) or "#FFFFFF"
        era_corrs = []
        for _, (es, ee) in ERAS.items():
            mask_era = (group_returns.index >= pd.Timestamp(es)) & \
                       (group_returns.index <= pd.Timestamp(ee))
            slice_df = group_returns.loc[mask_era, [g1, g2]].dropna()
            if len(slice_df) > 5:
                era_corrs.append(slice_df[g1].corr(slice_df[g2]))
            else:
                era_corrs.append(np.nan)

        offset = (i - n_pairs / 2 + 0.5) * bar_w
        bars = ax_era.bar(x + offset, era_corrs, bar_w,
                          label=f"{g1} / {g2}",
                          color=colour, alpha=0.85, edgecolor="#0F1117")
        # value labels on bars
        for bar, val in zip(bars, era_corrs):
            if not np.isnan(val):
                ax_era.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (0.02 if val >= 0 else -0.06),
                    f"{val:.2f}",
                    ha="center", va="bottom" if val >= 0 else "top",
                    color="white", fontsize=8, fontweight="bold",
                )

    ax_era.set_xticks(x)
    ax_era.set_xticklabels(era_labels, color="white", fontsize=9)
    ax_era.set_ylabel("Pearson r", color="#AAAAAA", fontsize=9)
    ax_era.axhline(0, color="#444444", linewidth=0.8, linestyle="--")
    ax_era.set_ylim(-0.3, 1.1)
    ax_era.legend(loc="upper right", framealpha=0.2, labelcolor="white",
                  fontsize=9, edgecolor="#444444")

    # ── Save / show ───────────────────────────────────────────────────────────
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"  Saved → {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def _style_ax(ax) -> None:
    """Apply dark theme to an axes."""
    ax.set_facecolor("#161B22")
    ax.tick_params(colors="#AAAAAA", labelsize=8)
    ax.spines["bottom"].set_color("#333333")
    ax.spines["left"].set_color("#333333")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.label.set_color("#AAAAAA")
    ax.xaxis.label.set_color("#AAAAAA")
    ax.grid(axis="y", color="#222222", linewidth=0.5, linestyle="--")
    ax.grid(axis="x", visible=False)


# ── Summary table ──────────────────────────────────────────────────────────────

def print_summary(group_returns: pd.DataFrame) -> None:
    print("\n" + "─" * 62)
    print("  Cross-Asset Group Summary")
    print("─" * 62)

    corr = group_returns.corr()
    groups = list(group_returns.columns)

    # Per-group stats
    print(f"\n  {'Group':<10}  {'CAGR':>8}  {'Ann.Vol':>8}  {'Sharpe':>8}  {'MaxDD':>8}")
    print(f"  {'─'*10}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}")
    for g in groups:
        r   = group_returns[g].dropna()
        ann = (1 + r.mean()) ** 252 - 1
        vol = r.std() * (252 ** 0.5)
        sharpe = (r.mean() * 252 - 0.045) / vol if vol > 0 else 0
        cum = (1 + r).cumprod()
        dd  = (cum / cum.cummax() - 1).min()
        print(f"  {g:<10}  {ann:>+8.2%}  {vol:>8.2%}  {sharpe:>8.3f}  {dd:>8.2%}")

    # Pairwise correlations
    print(f"\n  {'Pair':<22}  {'Full':>7}  {'2020-21':>8}  {'2022':>7}  {'2023+':>7}")
    print(f"  {'─'*22}  {'─'*7}  {'─'*8}  {'─'*7}  {'─'*7}")
    for i, g1 in enumerate(groups):
        for g2 in groups[i+1:]:
            full = corr.loc[g1, g2]
            era_vals = []
            for es, ee in [("2020-01-01","2021-12-31"),("2022-01-01","2022-12-31"),("2023-01-01","2026-12-31")]:
                sl = group_returns.loc[es:ee, [g1, g2]].dropna()
                era_vals.append(sl[g1].corr(sl[g2]) if len(sl) > 5 else float("nan"))
            print(f"  {g1+' / '+g2:<22}  {full:>7.3f}  {era_vals[0]:>8.3f}  {era_vals[1]:>7.3f}  {era_vals[2]:>7.3f}")

    print("\n" + "─" * 62)


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Cross-asset group correlation analysis")
    parser.add_argument("--start",   default="2020-01-01")
    parser.add_argument("--end",     default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    parser.add_argument("--rolling", default=63, type=int,
                        help="Rolling correlation window in bars (default: 63 = 1 quarter)")
    parser.add_argument("--no-show", action="store_true", dest="no_show",
                        help="Save chart but do not open display window")
    args = parser.parse_args()

    api_key    = os.environ.get("ALPACA_API_KEY",    "")
    secret_key = os.environ.get("ALPACA_SECRET_KEY", "")
    if not api_key or not secret_key:
        # try credentials.yaml
        creds_path = _ROOT / "config" / "credentials.yaml"
        if creds_path.exists():
            import yaml
            with creds_path.open() as fh:
                creds = yaml.safe_load(fh) or {}
            alpaca = creds.get("alpaca", {})
            api_key    = alpaca.get("api_key", "")
            secret_key = alpaca.get("secret_key", "")

    if not api_key or not secret_key:
        print("ERROR: Alpaca credentials not found.")
        print("Set ALPACA_API_KEY / ALPACA_SECRET_KEY or populate config/credentials.yaml.")
        sys.exit(1)

    all_symbols = list({s for syms in GROUPS.values() for s in syms})
    print(f"\nFetching {len(all_symbols)} symbols ({args.start} → {args.end}) ...")

    try:
        prices = _fetch_prices(all_symbols, args.start, args.end, api_key, secret_key)
    except Exception as exc:
        print(f"Data fetch failed: {exc}")
        sys.exit(1)

    print(f"  {len(prices)} trading days  ({prices.index[0].date()} → {prices.index[-1].date()})")
    print(f"  {len(prices.columns)} symbols loaded: {sorted(prices.columns.tolist())}")

    group_returns = build_group_returns(prices)
    print(f"  Groups built: {list(group_returns.columns)}\n")

    print_summary(group_returns)

    save_path = _ROOT / "results" / "correlation_analysis.png"
    plot_correlation_analysis(
        group_returns,
        start=args.start,
        end=args.end,
        rolling_window=args.rolling,
        save_path=save_path,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
