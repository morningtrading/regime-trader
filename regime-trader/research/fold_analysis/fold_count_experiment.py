#!/usr/bin/env python3
"""
Script 2: fold_count_experiment.py
Runs the backtest with different step_size values to measure how fold count
affects aggregate results. Saves outputs to research/fold_analysis/results/,
never touches savedresults/.

Usage (from repo root, WSL recommended):
    python research/fold_analysis/fold_count_experiment.py

Step sizes tested: 21, 42, 63 (current), 126, 252
"""
from __future__ import annotations
import sys
import os
import tempfile
from pathlib import Path

# ── Ensure repo root is on path ───────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

# Force single-threaded BLAS before any numpy import
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_v] = "1"

import yaml
import pandas as pd
import numpy as np

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)

# ── Load base config ──────────────────────────────────────────────────────────
cfg_path = ROOT / "config" / "settings.yaml"
with open(cfg_path) as f:
    base_cfg = yaml.safe_load(f)

# ── Step sizes to test ────────────────────────────────────────────────────────
STEP_SIZES = [21, 42, 63, 126, 252]
START_DATE = "2020-01-01"

# ── Lazy imports (heavy, after BLAS is fixed) ─────────────────────────────────
print("Loading modules...")
from data.market_data import MarketData
from backtest.backtester import WalkForwardBacktester
from backtest.performance import PerformanceAnalyzer

# ── Load credentials ──────────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")
creds_path = ROOT / "config" / "credentials.yaml"
if creds_path.exists():
    with open(creds_path) as f:
        creds = yaml.safe_load(f)
    alpaca_cfg = creds.get("alpaca", {})
    api_key    = alpaca_cfg.get("api_key",    os.getenv("ALPACA_API_KEY", ""))
    secret_key = alpaca_cfg.get("secret_key", os.getenv("ALPACA_SECRET_KEY", ""))
else:
    api_key    = os.getenv("ALPACA_API_KEY", "")
    secret_key = os.getenv("ALPACA_SECRET_KEY", "")

# ── Resolve symbols ───────────────────────────────────────────────────────────
broker_cfg = base_cfg.get("broker", {})
symbols    = broker_cfg.get("symbols", ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"])
print(f"Symbols: {symbols}")

# ── Fetch price data once ─────────────────────────────────────────────────────
print(f"Fetching data {START_DATE} → today ...")
md = MarketData(api_key=api_key, secret_key=secret_key, config=base_cfg)
prices = md.fetch_historical(symbols=symbols, start=START_DATE)
print(f"  Loaded {len(prices[symbols[0]])} bars for {len(symbols)} symbols")

# ── Run experiment ────────────────────────────────────────────────────────────
results = []
pa = PerformanceAnalyzer(risk_free_rate=float(base_cfg["backtest"].get("risk_free_rate", 0.045)))

for step in STEP_SIZES:
    print(f"\nstep_size={step} ...", flush=True)
    cfg = yaml.safe_load(open(cfg_path))           # fresh copy each iteration
    cfg["backtest"]["step_size"]  = step
    cfg["backtest"]["train_window"] = 252
    cfg["backtest"]["test_window"]  = 63

    bt = WalkForwardBacktester(config=cfg)
    try:
        result = bt.run(
            prices=prices,
            symbols=symbols,
            start_date=START_DATE,
        )
    except Exception as e:
        print(f"  ERROR: {e}")
        results.append({"step_size": step, "n_folds": 0, "error": str(e)})
        continue

    report = pa.analyze(result)
    n_folds = result.metadata.get("n_folds", len(result.windows))

    row = {
        "step_size":    step,
        "n_folds":      n_folds,
        "total_return": report.total_return,
        "cagr":         report.cagr,
        "sharpe":       report.sharpe_ratio,
        "sortino":      report.sortino_ratio,
        "max_drawdown": report.max_drawdown,
        "calmar":       report.calmar_ratio,
        "n_trades":     report.total_trades,
        "win_rate":     report.win_rate,
        "profit_factor":report.profit_factor,
        "final_equity": result.final_equity,
    }
    results.append(row)
    print(f"  n_folds={n_folds}  return={report.total_return:+.2%}  "
          f"Sharpe={report.sharpe_ratio:.3f}  MaxDD={report.max_drawdown:.2%}")

# ── Save + print ──────────────────────────────────────────────────────────────
df = pd.DataFrame(results)
out_path = OUT / "step_sensitivity.csv"
df.to_csv(out_path, index=False)

print(f"\n{'step':>6}  {'folds':>6}  {'return':>8}  {'Sharpe':>7}  "
      f"{'MaxDD':>7}  {'trades':>7}  {'PF':>6}")
print("─" * 58)
for _, r in df.iterrows():
    if "error" in r and pd.notna(r.get("error")):
        print(f"{int(r.step_size):>6}  ERROR: {r.get('error','')}")
    else:
        marker = " ◄ current" if r.step_size == 63 else ""
        print(f"{int(r.step_size):>6}  {int(r.n_folds):>6}  "
              f"{r.total_return:>+8.2%}  {r.sharpe:>7.3f}  "
              f"{r.max_drawdown:>7.2%}  {int(r.n_trades):>7}  "
              f"{r.profit_factor:>6.2f}{marker}")

print(f"\nSaved: {out_path}")
