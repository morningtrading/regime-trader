"""Wide sweep of SMA lookback on indices (daily, 2022-01-01 -> today).

Computes an equal-weighted SMA trend-following benchmark for a wide range
of lookback windows and reports CAGR / Sharpe / MaxDD so we can see if a
different window beats the canonical 200.

Run: py -3.12 scripts/sweep_sma_window.py
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import yaml  # type: ignore
import yfinance as yf  # type: ignore

from backtest.performance import PerformanceAnalyzer as PerformanceAnalytics  # type: ignore


# ── config ────────────────────────────────────────────────────────────────
GROUPS_PATH = ROOT / "config" / "asset_groups.yaml"
GROUP_NAME = os.environ.get("GROUP", "indices")
START = os.environ.get("START", "2022-01-01")
END = os.environ.get("END", pd.Timestamp.today().strftime("%Y-%m-%d"))
RF = float(os.environ.get("RF", "0.045"))
SLIPPAGE = float(os.environ.get("SLIPPAGE", "0.0005"))
INITIAL = 100_000.0

# Wide sweep: short to very long
WINDOWS = [50, 100, 150, 200, 250, 295, 305, 350, 400]


def load_symbols(group: str) -> list[str]:
    with open(GROUPS_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    g = (cfg.get("groups") or {}).get(group) or {}
    return [s for s in (g.get("symbols") or []) if isinstance(s, str)]


def download_prices(symbols: list[str], start: str, end: str) -> pd.DataFrame:
    print(f"Downloading daily prices for {len(symbols)} symbols: {symbols}")
    df = yf.download(
        symbols, start=start, end=end, auto_adjust=True,
        progress=False, group_by="ticker", threads=True,
    )
    # Flatten to close-price panel
    if isinstance(df.columns, pd.MultiIndex):
        close = pd.concat(
            {s: df[s]["Close"] for s in symbols if s in df.columns.get_level_values(0)},
            axis=1,
        )
    else:  # single symbol case
        close = df[["Close"]].rename(columns={"Close": symbols[0]})
    close = close.dropna(how="all").ffill().dropna()
    return close


def metrics(equity: pd.Series, rf: float) -> tuple[float, float, float, float]:
    """Return (total_return_%, CAGR_%, Sharpe, MaxDD_%)."""
    eq = equity.dropna()
    if len(eq) < 2:
        return (float("nan"),) * 4
    rets = eq.pct_change().dropna()
    total_ret = eq.iloc[-1] / eq.iloc[0] - 1.0
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1.0 / max(years, 1e-9)) - 1.0
    vol = rets.std() * np.sqrt(252)
    sharpe = ((cagr - rf) / vol) if vol > 0 else 0.0
    roll_max = eq.cummax()
    dd = (eq / roll_max - 1.0).min()
    return total_ret * 100, cagr * 100, sharpe, dd * 100


def main() -> None:
    symbols = load_symbols(GROUP_NAME)
    if not symbols:
        print(f"ERROR: no symbols in group '{GROUP_NAME}'")
        sys.exit(1)

    prices = download_prices(symbols, START, END)
    print(f"Loaded {len(prices)} bars, {len(prices.columns)} symbols "
          f"{prices.index[0].date()} -> {prices.index[-1].date()}\n")

    pa = PerformanceAnalytics()

    # Benchmarks for context
    bnh = pa.compute_benchmark_bnh_multi(prices, INITIAL, slippage_pct=SLIPPAGE)
    b_tr, b_cagr, b_sharpe, b_dd = metrics(bnh, RF)
    print(f"{'Benchmark':<20} {'Return':>10} {'CAGR':>10} {'Sharpe':>8} {'MaxDD':>10}")
    print("-" * 64)
    print(f"{'B&H (EW)':<20} {b_tr:>9.2f}% {b_cagr:>9.2f}% {b_sharpe:>8.3f} {b_dd:>9.2f}%")

    # Sweep
    rows = []
    for w in WINDOWS:
        if w >= len(prices):
            continue
        eq = pa.compute_benchmark_sma_multi(prices, w, INITIAL, slippage_pct=SLIPPAGE)
        tr, cagr, sharpe, dd = metrics(eq, RF)
        rows.append((w, tr, cagr, sharpe, dd))

    print()
    print(f"{'SMA_window':<20} {'Return':>10} {'CAGR':>10} {'Sharpe':>8} {'MaxDD':>10}")
    print("-" * 64)
    for w, tr, cagr, sharpe, dd in rows:
        print(f"SMA-{w:<16} {tr:>9.2f}% {cagr:>9.2f}% {sharpe:>8.3f} {dd:>9.2f}%")

    # Best by Sharpe
    best = max(rows, key=lambda r: (r[3] if r[3] == r[3] else -1e9))
    print()
    print(f"Best by Sharpe: SMA-{best[0]} → Sharpe {best[3]:.3f}, "
          f"Return {best[1]:.2f}%, MaxDD {best[4]:.2f}%")


if __name__ == "__main__":
    main()
