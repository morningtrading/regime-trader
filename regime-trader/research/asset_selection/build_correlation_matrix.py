#!/usr/bin/env python3
"""
Script 1: build_correlation_matrix.py
Fetches daily close prices for 40 candidate symbols and saves a pairwise
correlation matrix to results/corr_matrix.csv.

Usage (from repo root, WSL):
    source .venv/bin/activate
    python research/asset_selection/build_correlation_matrix.py
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
from datetime import date

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_v] = "1"

import pandas as pd
import numpy as np
import yaml

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)

# ── Candidate universe ────────────────────────────────────────────────────────
UNIVERSE = [
    # Broad ETFs
    "SPY", "QQQ", "IWM",
    # Sector ETFs
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLU", "XLC",
    # Alt / Macro ETFs
    "GLD", "TLT", "VNQ",
    # Large-cap US equities (2019 market cap ranking)
    "AAPL", "MSFT", "AMZN", "GOOGL", "JPM", "JNJ", "PG",
    "V", "MA", "UNH", "HD", "BAC", "XOM", "CVX", "WMT", "KO", "PEP",
    # Growth / Semis
    "NVDA", "AMD", "QCOM",
    # Defense / Aerospace
    "LMT", "RTX", "NOC",
    # Healthcare individual
    "PFE", "ABT",
    # BTC/USD and ETH/USD removed — incompatible with equity HMM position sizing
]

START = "2020-01-01"
END   = str(date.today())

# ── Load credentials ──────────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

creds_path = ROOT / "config" / "credentials.yaml"
if creds_path.exists():
    with open(creds_path) as f:
        _creds = yaml.safe_load(f)
    _alpaca = _creds.get("alpaca", {})
    api_key    = _alpaca.get("api_key",    os.getenv("ALPACA_API_KEY",    ""))
    secret_key = _alpaca.get("secret_key", os.getenv("ALPACA_SECRET_KEY", ""))
else:
    api_key    = os.getenv("ALPACA_API_KEY",    "")
    secret_key = os.getenv("ALPACA_SECRET_KEY", "")

if not api_key or not secret_key:
    sys.exit("ERROR: Alpaca credentials not found. Set ALPACA_API_KEY / ALPACA_SECRET_KEY.")

# ── Fetch prices (reuse main.py logic) ───────────────────────────────────────
from alpaca.data.timeframe import TimeFrame

def _is_crypto(sym: str) -> bool:
    return "/" in sym

def fetch_daily_closes(symbols: list[str], start: str, end: str) -> pd.DataFrame:
    stock_syms  = [s for s in symbols if not _is_crypto(s)]
    crypto_syms = [s for s in symbols if     _is_crypto(s)]
    frames = []

    if stock_syms:
        from alpaca.data import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.enums import Adjustment
        sc  = StockHistoricalDataClient(api_key, secret_key)
        req = StockBarsRequest(
            symbol_or_symbols=stock_syms,
            timeframe=TimeFrame.Day,
            start=start, end=end,
            adjustment=Adjustment.ALL,
        )
        df = sc.get_stock_bars(req).df
        if not df.empty and "close" in df.columns:
            close = df["close"].unstack(level="symbol")
            close.index = pd.to_datetime(close.index).normalize().tz_localize(None)
            close.index.name = "date"
            frames.append(close)

    if crypto_syms:
        from alpaca.data.historical.crypto import CryptoHistoricalDataClient
        from alpaca.data.requests import CryptoBarsRequest
        cc  = CryptoHistoricalDataClient(api_key, secret_key)
        req = CryptoBarsRequest(
            symbol_or_symbols=crypto_syms,
            timeframe=TimeFrame.Day,
            start=start, end=end,
        )
        df = cc.get_crypto_bars(req).df
        if not df.empty and "close" in df.columns:
            close = df["close"].unstack(level="symbol")
            close.index = pd.to_datetime(close.index).normalize().tz_localize(None)
            close.index.name = "date"
            frames.append(close)

    if not frames:
        return pd.DataFrame()

    combined = frames[0].join(frames[1], how="outer") if len(frames) > 1 else frames[0]
    return combined.sort_index().dropna(how="all")

# ── Main ──────────────────────────────────────────────────────────────────────
print(f"Fetching daily bars {START} → {END} for {len(UNIVERSE)} symbols...")
prices = fetch_daily_closes(UNIVERSE, START, END)

available = [s for s in UNIVERSE if s in prices.columns]
missing   = [s for s in UNIVERSE if s not in prices.columns]
if missing:
    print(f"  WARNING: not returned by Alpaca: {missing}")
print(f"  Got {len(prices)} bars for {len(available)} symbols")

# ── Compute correlation ───────────────────────────────────────────────────────
log_ret = np.log(prices[available] / prices[available].shift(1)).dropna()
corr    = log_ret.corr()

out_path = OUT / "corr_matrix.csv"
corr.to_csv(out_path)
print(f"\nCorrelation matrix saved: {out_path}")

# ── Print mini heatmap ────────────────────────────────────────────────────────
LINE = "─" * 72
print(f"\n{LINE}")
print("  PAIRWISE CORRELATION SNAPSHOT (selected pairs)")
print(LINE)

REFERENCE_PAIRS = [
    ("SPY",  "QQQ"),
    ("SPY",  "GLD"),
    ("SPY",  "TLT"),
    ("SPY",  "IWM"),
    ("QQQ",  "AAPL"),
    ("QQQ",  "NVDA"),
    ("QQQ",  "LMT"),
    ("TLT",  "GLD"),
    ("AAPL", "MSFT"),
    ("LMT",  "NOC"),
    ("XLF",  "JPM"),
    ("XLE",  "XOM"),
    ("UNH",  "JNJ"),
]

print(f"  {'Symbol A':<12}  {'Symbol B':<12}  {'Corr':>6}")
print(f"  {'─'*12}  {'─'*12}  {'─'*6}")
for a, b in REFERENCE_PAIRS:
    if a in corr.columns and b in corr.columns:
        v = corr.loc[a, b]
        bar = "█" * int(abs(v) * 20) + "░" * (20 - int(abs(v) * 20))
        sign = "+" if v >= 0 else "-"
        print(f"  {a:<12}  {b:<12}  {sign}{abs(v):.3f}  {bar}")

# ── Baseline basket summary ───────────────────────────────────────────────────
BASELINE = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"]
baseline_syms = [s for s in BASELINE if s in corr.columns]
if len(baseline_syms) >= 2:
    vals = []
    for i, a in enumerate(baseline_syms):
        for b in baseline_syms[i+1:]:
            vals.append(corr.loc[a, b])
    print(f"\n  Current basket avg pairwise corr: {np.mean(vals):.3f}  "
          f"(n={len(vals)} pairs)")
    print(f"  Pairs: {', '.join(f'{a}/{b}={corr.loc[a,b]:.2f}' for a in baseline_syms for b in baseline_syms if a < b and a in corr.columns and b in corr.columns)}")

print(f"\n  Next step: run monte_carlo_asset_selection.py")
