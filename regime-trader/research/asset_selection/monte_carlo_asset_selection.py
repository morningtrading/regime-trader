#!/usr/bin/env python3
"""
Script 2: monte_carlo_asset_selection.py
Two-phase Monte Carlo asset selection with parallel backtesting.

  Phase 1 (fast screening) — 3-year backtest on ~150 low-correlation combos
  Phase 2 (full validation) — 6-year backtest on top 15 Phase 1 results

  Parallelism: uses multiprocessing.Pool (fork) — all 4 cores run simultaneously.
  Prices are loaded once in the main process and inherited by workers via fork.

Usage (from repo root, WSL):
    source .venv/bin/activate
    python research/asset_selection/monte_carlo_asset_selection.py

Resume-safe: already-completed combos are skipped on re-run.
"""
from __future__ import annotations
import datetime
import os
import socket
import subprocess
import sys
import json
import time
import itertools
from pathlib import Path
from datetime import date
from multiprocessing import Pool, cpu_count
from typing import Optional

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

PHASE1_CSV = OUT / "mc_results_phase1.csv"
PHASE2_CSV = OUT / "mc_results_phase2.csv"

# ── Configuration ─────────────────────────────────────────────────────────────
UNIVERSE = [
    "SPY", "QQQ", "IWM",
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLU", "XLC",
    "GLD", "TLT", "VNQ",
    "AAPL", "MSFT", "AMZN", "GOOGL", "JPM", "JNJ", "PG",
    "V", "MA", "UNH", "HD", "BAC", "XOM", "CVX", "WMT", "KO", "PEP",
    "NVDA", "AMD", "QCOM",
    "LMT", "RTX", "NOC",
    "PFE", "ABT",
    # BTC/USD and ETH/USD removed — incompatible with equity HMM position sizing
]

BASELINE      = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"]
PHASE1_START  = "2021-01-01"
PHASE2_START  = "2020-01-01"
END_DATE      = str(date.today())
TOP_PER_K     = 50    # lowest-corr combos per k → 150 total for Phase 1
PHASE2_TOP_N  = 15    # top Phase 1 results promoted to Phase 2
SAMPLE_K6     = 60_000
SAMPLE_K7     = 60_000
RNG_SEED      = 42
N_WORKERS     = min(4, cpu_count())   # parallel backtest workers
CHECKPOINT_N  = N_WORKERS * 5        # save every N completed results

BROAD_ETFS = {"SPY", "QQQ", "IWM"}

# ── Load config and credentials ───────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

cfg_path = ROOT / "config" / "settings.yaml"
with open(cfg_path) as f:
    base_cfg = yaml.safe_load(f)

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
    sys.exit("ERROR: Alpaca credentials not found.")

bt_cfg    = base_cfg.get("backtest",  {})
strat_cfg = base_cfg.get("strategy",  {})
risk_cfg  = base_cfg.get("risk",      {})

# ── Helpers ───────────────────────────────────────────────────────────────────
def _is_crypto(sym: str) -> bool:
    return "/" in sym

def pick_regime_proxy(combo: tuple[str, ...]) -> str:
    if "QQQ" in combo:
        return "QQQ"
    for sym in combo:
        if sym in BROAD_ETFS:
            return sym
    return combo[0]

def combo_key(combo: tuple[str, ...]) -> str:
    return json.dumps(sorted(combo))

def avg_pairwise_corr(corr_mat: pd.DataFrame, syms: list[str]) -> float:
    vals = [corr_mat.loc[a, b]
            for i, a in enumerate(syms)
            for b in syms[i+1:]
            if a in corr_mat.index and b in corr_mat.index]
    return float(np.mean(vals)) if vals else 1.0

def fmt_eta(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"

# ── Price fetching ────────────────────────────────────────────────────────────
from alpaca.data.timeframe import TimeFrame

def fetch_daily_closes(symbols: list[str], start: str, end: str) -> pd.DataFrame:
    stock_syms = [s for s in symbols if not _is_crypto(s)]
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

    if not frames:
        return pd.DataFrame()
    return frames[0].sort_index().dropna(how="all")

# ── Single backtest (called inside worker processes) ──────────────────────────
from backtest.backtester import WalkForwardBacktester
from backtest.performance import PerformanceAnalyzer

_PA = PerformanceAnalyzer(risk_free_rate=float(bt_cfg.get("risk_free_rate", 0.045)))

def _run_one(combo: tuple[str, ...], prices: pd.DataFrame) -> Optional[dict]:
    syms = [s for s in combo if s in prices.columns]
    if len(syms) < 2:
        return None
    proxy   = pick_regime_proxy(tuple(syms))
    hmm_cfg = {**base_cfg.get("hmm", {}), "regime_proxy": proxy, "n_init": 3}
    bt = WalkForwardBacktester(
        symbols         = syms,
        initial_capital = float(bt_cfg.get("initial_capital", 100_000)),
        train_window    = int(bt_cfg.get("train_window", 252)),
        test_window     = int(bt_cfg.get("test_window",  63)),
        step_size       = int(bt_cfg.get("step_size",    63)),
        slippage_pct    = float(bt_cfg.get("slippage_pct", 0.0005)),
        risk_free_rate  = float(bt_cfg.get("risk_free_rate", 0.045)),
    )
    try:
        result = bt.run(prices=prices[syms], hmm_config=hmm_cfg,
                        strategy_config=strat_cfg, risk_config=risk_cfg)
        r = _PA.analyze(result)
        return {
            "symbols":      combo_key(tuple(syms)),
            "k":            len(syms),
            "regime_proxy": proxy,
            "sharpe":       r.sharpe_ratio,
            "max_drawdown": r.max_drawdown,
            "total_return": r.total_return,
            "cagr":         r.cagr,
            "calmar":       r.calmar_ratio,
            "n_trades":     r.total_trades,
            "win_rate":     r.win_rate,
        }
    except Exception as e:
        print(f"    [worker] ERROR {syms}: {e}", flush=True)
        return None

# ── Worker entry point (module-level so it's picklable) ───────────────────────
# _WORKER_PRICES is set in the main process before Pool creation.
# On Linux, fork() gives each worker a copy-on-write view of this DataFrame.
_WORKER_PRICES: pd.DataFrame = pd.DataFrame()

def _task(args: tuple) -> tuple:
    """Called by each worker process. Returns (corr_val, combo, is_baseline, row)."""
    corr_val, combo, is_baseline = args
    row = _run_one(tuple(combo), _WORKER_PRICES)
    if row is not None:
        row["avg_corr"]    = corr_val
        row["is_baseline"] = is_baseline
    return corr_val, combo, is_baseline, row

# ── Parallel phase runner ─────────────────────────────────────────────────────
def run_phase(
    phase_name: str,
    todo: list[tuple],
    prices: pd.DataFrame,
    existing_rows: list[dict],
    out_csv: Path,
    phase1_df: Optional[pd.DataFrame] = None,
    baseline_key: str = "",
) -> list[dict]:
    """
    Run backtests in parallel using Pool.imap_unordered.
    Returns all completed rows (existing + new).
    """
    global _WORKER_PRICES
    _WORKER_PRICES = prices   # inherited by workers via fork

    rows = list(existing_rows)
    n_todo  = len(todo)
    t_start = time.monotonic()
    done    = 0

    # tasks: (corr_val, combo, is_baseline)
    tasks = [(cv, list(c), combo_key(tuple(sorted(c))) == baseline_key)
             for cv, c in todo]

    with Pool(processes=N_WORKERS) as pool:
        for corr_val, combo, is_baseline, row in pool.imap_unordered(_task, tasks, chunksize=1):
            done += 1
            elapsed = time.monotonic() - t_start
            eta     = (elapsed / done) * (n_todo - done) if done < n_todo else 0
            label   = " ← BASELINE" if is_baseline else ""

            if row is not None:
                # attach Phase 1 sharpe if this is Phase 2
                if phase1_df is not None:
                    key = row["symbols"]
                    p1  = phase1_df[phase1_df["symbols"] == key]["sharpe"]
                    row["sharpe_3yr"] = float(p1.iloc[0]) if len(p1) else float("nan")
                rows.append(row)
                print(f"  [{done:>4}/{n_todo}] {combo[:3]}...  "
                      f"Sharpe={row['sharpe']:.3f}  Return={row['total_return']:+.1%}  "
                      f"MaxDD={row['max_drawdown']:.1%}  corr={corr_val:.3f}  "
                      f"ETA {fmt_eta(eta)}{label}", flush=True)
            else:
                print(f"  [{done:>4}/{n_todo}] {combo[:3]}... SKIPPED  ETA {fmt_eta(eta)}", flush=True)

            if done % CHECKPOINT_N == 0 or done == n_todo:
                pd.DataFrame(rows).to_csv(out_csv, index=False)
                print(f"    → checkpoint saved ({len(rows)} rows)", flush=True)

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    total_t = time.monotonic() - t_start
    print(f"\n  {phase_name} complete in {fmt_eta(total_t)}  "
          f"({total_t/max(n_todo,1):.1f} s/combo on {N_WORKERS} workers)")
    return rows

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # ── Run metadata — written at start for config control ────────────────────
    def _git_hash_mc() -> str:
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=str(ROOT), stderr=subprocess.DEVNULL,
            ).decode().strip()
        except Exception:
            return "unknown"

    _run_meta = {
        "script":        str(Path(__file__).resolve()),
        "run_timestamp": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "machine":       socket.gethostname(),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "git_hash":      _git_hash_mc(),
        "universe":      UNIVERSE,
        "n_universe":    len(UNIVERSE),
        "top_per_k":     TOP_PER_K,
        "phase2_top_n":  PHASE2_TOP_N,
        "phase1_start":  PHASE1_START,
        "phase2_start":  PHASE2_START,
        "n_workers":     N_WORKERS,
        "rng_seed":      RNG_SEED,
        "sample_k6":     SAMPLE_K6,
        "sample_k7":     SAMPLE_K7,
        "output_dir":    str(OUT),
    }
    with open(OUT / "run_metadata.json", "w") as _mf:
        json.dump(_run_meta, _mf, indent=2)

    _W = 72
    print("─" * _W)
    print("  Monte Carlo Asset Selection")
    print("─" * _W)
    print(f"  Timestamp    : {_run_meta['run_timestamp']}")
    print(f"  Machine      : {_run_meta['machine']}  (Python {_run_meta['python_version']})")
    print(f"  Bot version  : {_run_meta['git_hash']}")
    print(f"  Universe     : {len(UNIVERSE)} symbols  TOP_PER_K={TOP_PER_K}  PHASE2_TOP_N={PHASE2_TOP_N}")
    print(f"  Phase 1 start: {PHASE1_START}   Phase 2 start: {PHASE2_START}")
    print(f"  Workers      : {N_WORKERS}  RNG seed: {RNG_SEED}")
    print(f"  Output dir   : {OUT}")
    print("─" * _W)
    print()

    # ── Step 1: load correlation matrix ──────────────────────────────────────
    corr_path = OUT / "corr_matrix.csv"
    if not corr_path.exists():
        sys.exit("ERROR: corr_matrix.csv not found. Run build_correlation_matrix.py first.")

    corr  = pd.read_csv(corr_path, index_col=0)
    avail = [s for s in UNIVERSE if s in corr.index]
    print(f"Loaded correlation matrix: {len(avail)} / {len(UNIVERSE)} symbols available")
    print(f"Workers: {N_WORKERS}  TOP_PER_K={TOP_PER_K}  PHASE2_TOP_N={PHASE2_TOP_N}\n")

    # ── Step 2: enumerate and score combinations ──────────────────────────────
    print("Scoring combinations by average pairwise correlation...")
    rng    = np.random.default_rng(RNG_SEED)
    scored: list[tuple[float, tuple[str, ...]]] = []

    for k in [5, 6, 7]:
        print(f"  k={k} ...", end=" ", flush=True)
        if k == 5:
            combos = list(itertools.combinations(avail, k))
        else:
            n_total = 1
            for i in range(k):
                n_total = n_total * (len(avail) - i) // (i + 1)
            sample_size = min(SAMPLE_K6 if k == 6 else SAMPLE_K7, n_total)
            idx_set = set(rng.choice(n_total, size=sample_size, replace=False).tolist())
            combos  = [c for i, c in enumerate(itertools.combinations(avail, k))
                       if i in idx_set]

        corr_scores = [(avg_pairwise_corr(corr, list(c)), c) for c in combos]
        corr_scores.sort(key=lambda x: x[0])
        top = corr_scores[:TOP_PER_K]
        scored.extend(top)
        print(f"{len(combos):,} evaluated → top {len(top)} kept "
              f"(corr {top[0][0]:.3f}–{top[-1][0]:.3f})")

    baseline_tuple = tuple(sorted(BASELINE))
    baseline_corr  = avg_pairwise_corr(corr, list(baseline_tuple))
    baseline_key   = combo_key(baseline_tuple)
    if baseline_key not in {combo_key(c) for _, c in scored}:
        scored.append((baseline_corr, baseline_tuple))
        print(f"  Baseline added (corr={baseline_corr:.3f})")

    print(f"\nTotal combos for Phase 1: {len(scored)}")

    # ── Step 3: Phase 1 — 3-year screening ───────────────────────────────────
    print(f"\n{'─'*72}")
    print(f"  PHASE 1 — 3-year screening  ({PHASE1_START} → {END_DATE})  [{N_WORKERS} workers]")
    print(f"{'─'*72}")

    done_keys: set[str] = set()
    phase1_rows: list[dict] = []
    if PHASE1_CSV.exists():
        ex = pd.read_csv(PHASE1_CSV)
        for _, row in ex.iterrows():
            done_keys.add(row["symbols"])
            phase1_rows.append(row.to_dict())
        print(f"  Resuming: {len(done_keys)} combos already completed")

    print(f"  Fetching 3-year price data for {len(avail)} symbols...")
    prices_3yr = fetch_daily_closes(avail, PHASE1_START, END_DATE)
    print(f"  Got {len(prices_3yr)} bars × {prices_3yr.shape[1]} symbols\n")

    todo1 = [(cv, c) for cv, c in scored if combo_key(c) not in done_keys]
    print(f"  Running {len(todo1)} backtests ({len(scored)-len(todo1)} already cached)...\n")

    if todo1:
        phase1_rows = run_phase(
            "Phase 1", todo1, prices_3yr, phase1_rows, PHASE1_CSV,
            baseline_key=baseline_key,
        )
    print(f"Phase 1 results: {PHASE1_CSV}")

    # ── Step 4: Phase 2 — 6-year validation ──────────────────────────────────
    print(f"\n{'─'*72}")
    print(f"  PHASE 2 — full validation  ({PHASE2_START} → {END_DATE})  [{N_WORKERS} workers]")
    print(f"{'─'*72}")

    phase1_df = pd.read_csv(PHASE1_CSV)
    phase1_df = phase1_df.dropna(subset=["sharpe"]).sort_values("sharpe", ascending=False)

    top_keys  = set(phase1_df.head(PHASE2_TOP_N)["symbols"].tolist())
    top_keys.add(baseline_key)
    finalists = [(row["avg_corr"], json.loads(row["symbols"]))
                 for _, row in phase1_df.iterrows()
                 if row["symbols"] in top_keys]

    # Force-add baseline if it never appeared in Phase 1
    f_keys = {combo_key(tuple(sorted(c))) for _, c in finalists}
    if baseline_key not in f_keys:
        finalists.append((baseline_corr, list(baseline_tuple)))
        print(f"  Baseline force-added to Phase 2")

    print(f"  Finalists: {len(finalists)} combos (top {PHASE2_TOP_N} + baseline)")

    done2_keys: set[str] = set()
    phase2_rows: list[dict] = []
    if PHASE2_CSV.exists():
        ex2 = pd.read_csv(PHASE2_CSV)
        for _, row in ex2.iterrows():
            done2_keys.add(row["symbols"])
            phase2_rows.append(row.to_dict())
        print(f"  Resuming: {len(done2_keys)} already done")

    print(f"  Fetching 6-year price data for {len(avail)} symbols...")
    prices_6yr = fetch_daily_closes(avail, PHASE2_START, END_DATE)
    print(f"  Got {len(prices_6yr)} bars × {prices_6yr.shape[1]} symbols\n")

    todo2 = [(cv, tuple(sorted(c))) for cv, c in finalists
             if combo_key(tuple(sorted(c))) not in done2_keys]
    print(f"  Running {len(todo2)} full backtests ({len(finalists)-len(todo2)} cached)...\n")

    if todo2:
        phase2_rows = run_phase(
            "Phase 2", todo2, prices_6yr, phase2_rows, PHASE2_CSV,
            phase1_df=phase1_df,
            baseline_key=baseline_key,
        )

    print(f"\nPhase 2 complete. Results: {PHASE2_CSV}")
    print(f"Next step: python research/asset_selection/asset_selection_report.py")
