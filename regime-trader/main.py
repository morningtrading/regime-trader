"""
main.py — Regime Trader entry point.

    python main.py trade        — live / paper trading loop
    python main.py backtest     — walk-forward backtest
    python main.py stress       — stress-test scenario suite
    python main.py trade --dry-run     — full pipeline, no orders placed
    python main.py trade --train-only  — train HMM and exit

All tuneable parameters live in config/settings.yaml.
Credentials are read from config/credentials.yaml (git-ignored) or from
ALPACA_API_KEY / ALPACA_SECRET_KEY environment variables.
"""

from __future__ import annotations

# ─── Force single-threaded BLAS for cross-platform reproducibility ──────────
# MUST be set BEFORE numpy/scipy/hmmlearn are imported. Multi-threaded BLAS
# (OpenBLAS, MKL, Accelerate) uses non-deterministic reduction orders that
# produce bit-different results across machines, causing HMM EM to converge
# to different local optima on Windows vs Linux despite fixed random_state.
# Performance cost: ~2-3x slower HMM fitting. Benefit: identical results.
import os as _os
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    _os.environ[_var] = "1"  # override unconditionally — setdefault was silently ignored by IDE/shell env

import argparse
import datetime as dt
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv


# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.WARNING,
)
logger = logging.getLogger("main")

# ── Paths ──────────────────────────────────────────────────────────────────────

_ROOT              = Path(__file__).resolve().parent
_MODEL_PATH        = _ROOT / "models" / "hmm.pkl"
_SNAPSHOT_PATH     = _ROOT / "state_snapshot.json"
_MODEL_MAX_AGE_DAYS = 7
_SAVED_RESULTS_DIR = _ROOT / "savedresults"
_SETS_DIR          = _ROOT / "config" / "sets"
_ACTIVE_SET_FILE   = _ROOT / "config" / "active_set"

# ── HMM feature helper (defined in data/feature_engineering.py) ───────────────
from data.feature_blending import blend_cross_symbol_features as _blend_cross_symbol_features
from data.feature_engineering import hmm_feature_names as _hmm_feature_names


def _maybe_fetch_vix(hmm_cfg: Dict, bars_index) -> "pd.Series | None":
    """Fetch VIX series if hmm_cfg['use_vix_features'] OR ['use_credit_spread_features'] is truthy."""
    if not (hmm_cfg.get("use_vix_features", False)
            or hmm_cfg.get("use_credit_spread_features", False)):
        return None
    try:
        from data.vix_fetcher import fetch_vix_series
        if bars_index is None or len(bars_index) == 0:
            return None
        start = pd.Timestamp(bars_index[0]).tz_localize(None) if hasattr(pd.Timestamp(bars_index[0]), "tz_localize") else pd.Timestamp(bars_index[0])
        end   = pd.Timestamp(bars_index[-1]).tz_localize(None) if hasattr(pd.Timestamp(bars_index[-1]), "tz_localize") else pd.Timestamp(bars_index[-1])
        # widen by a couple of days so ffill has a head value
        start = (start - pd.Timedelta(days=5)).date().isoformat()
        end   = (end   + pd.Timedelta(days=1)).date().isoformat()
        return fetch_vix_series(start=start, end=end, timeframe="1Day")
    except Exception as exc:
        logger.warning("VIX fetch skipped: %s", exc)
        return None


def _maybe_fetch_credit(hmm_cfg: Dict, bars_index) -> "pd.Series | None":
    """Fetch HYG/LQD credit spread z-score if hmm_cfg['use_credit_spread_features'] is truthy."""
    if not hmm_cfg.get("use_credit_spread_features", False):
        return None
    try:
        from data.credit_spread_fetcher import fetch_credit_spread_series
        if bars_index is None or len(bars_index) == 0:
            return None
        start = (pd.Timestamp(bars_index[0]) - pd.Timedelta(days=120)).date().isoformat()
        end   = (pd.Timestamp(bars_index[-1]) + pd.Timedelta(days=1)).date().isoformat()
        return fetch_credit_spread_series(start=start, end=end, timeframe="1Day")
    except Exception as exc:
        logger.warning("Credit spread fetch skipped: %s", exc)
        return None


# ── Saved-results helpers ──────────────────────────────────────────────────────

def _saved_results_backtest_dir() -> Path:
    """Return a timestamped subdirectory under savedresults/ for one backtest run."""
    ts = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    d = _SAVED_RESULTS_DIR / f"backtest_{ts}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save_training_log(engine, symbols: List[str], hmm_cfg: Dict) -> None:
    """Append one row to savedresults/training_log.csv after each HMM training."""
    _SAVED_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = _SAVED_RESULTS_DIR / "training_log.csv"
    row = {
        "timestamp":   dt.datetime.now().isoformat(timespec="seconds"),
        "n_states":    engine._n_states,
        "bic":         round(engine._training_bic, 4),
        "n_bars":      getattr(engine, "_n_train_bars", 0),
        "features":    "|".join(_hmm_feature_names(hmm_cfg)),
        "extended":    hmm_cfg.get("extended_features", True),
        "symbols":     ",".join(symbols),
    }
    write_header = not log_path.exists()
    pd.DataFrame([row]).to_csv(log_path, mode="a", header=write_header, index=False)
    logger.info("Training log appended to %s", log_path)


# ── Config helpers ─────────────────────────────────────────────────────────────

def _deep_merge(base: Dict, overrides: Dict) -> Dict:
    """Recursively merge overrides into base, returning a new dict."""
    result = base.copy()
    for key, val in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def load_config(config_path: str = "config/settings.yaml",
                set_name: Optional[str] = None) -> Dict:
    """
    Load base settings.yaml then deep-merge the active config set on top.

    Resolution order (highest wins):
      1. set_name argument (--set CLI flag)
      2. config/active_set file
      3. base settings.yaml alone
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Settings file not found: {path}")
    with path.open() as fh:
        base = yaml.safe_load(fh) or {}

    resolved = set_name
    if resolved is None and _ACTIVE_SET_FILE.exists():
        resolved = _ACTIVE_SET_FILE.read_text().strip() or None

    if resolved:
        set_path = _SETS_DIR / f"{resolved}.yaml"
        if set_path.exists():
            with set_path.open() as fh:
                overrides = yaml.safe_load(fh) or {}
            base = _deep_merge(base, overrides)
            logger.info("Config set '%s' applied.", resolved)
        else:
            logger.warning("Config set '%s' not found at %s — using base config.",
                           resolved, set_path)
            resolved = None

    base["_active_set"] = resolved or "base"
    return base


def load_credentials(credentials_path: str = "config/credentials.yaml") -> Optional[Dict]:
    """
    Load credentials YAML if it exists and inject values into env vars so
    the rest of the code can use os.getenv() uniformly.
    """
    path = Path(credentials_path)
    if not path.exists():
        return None
    with path.open() as fh:
        creds = yaml.safe_load(fh) or {}

    alpaca = creds.get("alpaca", {})
    if alpaca.get("api_key") and not os.environ.get("ALPACA_API_KEY"):
        os.environ["ALPACA_API_KEY"] = alpaca["api_key"]
    if alpaca.get("secret_key") and not os.environ.get("ALPACA_SECRET_KEY"):
        os.environ["ALPACA_SECRET_KEY"] = alpaca["secret_key"]

    return creds


# ── Data fetching ──────────────────────────────────────────────────────────────

# Major fiat currencies — used to detect forex pairs (X/Y where both sides are
# fiat). Alpaca does NOT support forex; we flag these up-front so the user gets
# a clear error instead of a KeyError('close') deep in the fetch pipeline.
_FIAT_CURRENCIES = {
    "USD", "EUR", "GBP", "JPY", "CAD", "AUD", "CHF", "NZD",
    "CNY", "HKD", "SEK", "NOK", "DKK", "SGD", "MXN", "ZAR",
}


def _is_forex_like(symbol: str) -> bool:
    """Return True for X/Y symbols where both sides are fiat currencies
    (EUR/USD, GBP/JPY, USD/CAD …). Alpaca does not serve these."""
    if "/" not in symbol:
        return False
    base, _, quote = symbol.partition("/")
    return base.upper() in _FIAT_CURRENCIES and quote.upper() in _FIAT_CURRENCIES


def _is_crypto(symbol: str) -> bool:
    """Return True if the symbol is an Alpaca-supported crypto pair (BASE/QUOTE,
    not both-fiat). Excludes forex pairs like EUR/USD."""
    if "/" not in symbol:
        return False
    return not _is_forex_like(symbol)


def _fetch_prices(
    symbols: List[str],
    start: str,
    end: str,
    api_key: str,
    secret_key: str,
) -> pd.DataFrame:
    """
    Download daily close prices from Alpaca for all symbols.

    Automatically routes crypto symbols (containing '/') to the
    CryptoHistoricalDataClient and equity symbols to StockHistoricalDataClient.

    Returns wide-format DataFrame: index = date (tz-naive), columns = symbols.
    """
    from alpaca.data.timeframe import TimeFrame

    # Up-front validation: flag symbols Alpaca can't serve (forex, options, futures).
    forex_syms = [s for s in symbols if _is_forex_like(s)]
    if forex_syms:
        raise ValueError(
            f"Alpaca does not support forex symbols: {forex_syms}. "
            f"Use an equity/ETF proxy (e.g. FXE for EUR, FXY for JPY, UUP for USD index) "
            f"or plug in an alternative data source."
        )

    crypto_syms = [s for s in symbols if _is_crypto(s)]
    stock_syms  = [s for s in symbols if not _is_crypto(s) and not _is_forex_like(s)]
    frames: List[pd.DataFrame] = []

    # ── Equity bars ──────────────────────────────────────────────────────────
    if stock_syms:
        from alpaca.data import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.enums import Adjustment
        sc = StockHistoricalDataClient(api_key, secret_key)
        req = StockBarsRequest(
            symbol_or_symbols=stock_syms,
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
            adjustment=Adjustment.ALL,
        )
        df = sc.get_stock_bars(req).df
        if df.empty or "close" not in df.columns:
            raise ValueError(
                f"Alpaca returned no stock bars for {stock_syms} "
                f"(range {start} → {end}). Check the symbols and your Alpaca "
                f"subscription tier (data_feed={os.getenv('ALPACA_DATA_FEED', 'iex')})."
            )
        close = df["close"].unstack(level="symbol")
        close.index = pd.to_datetime(close.index).normalize().tz_localize(None)
        close.index.name = "date"
        frames.append(close)

    # ── Crypto bars ───────────────────────────────────────────────────────────
    if crypto_syms:
        from alpaca.data.historical.crypto import CryptoHistoricalDataClient
        from alpaca.data.requests import CryptoBarsRequest
        # Crypto data client needs no auth for historical data
        cc = CryptoHistoricalDataClient(api_key, secret_key)
        req = CryptoBarsRequest(
            symbol_or_symbols=crypto_syms,
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
        )
        df = cc.get_crypto_bars(req).df
        if df.empty or "close" not in df.columns:
            raise ValueError(
                f"Alpaca returned no crypto bars for {crypto_syms} "
                f"(range {start} → {end}). Verify symbols — typical format "
                f"is BTC/USD, ETH/USD, SOL/USD, etc."
            )
        close = df["close"].unstack(level="symbol")
        close.index = pd.to_datetime(close.index).normalize().tz_localize(None)
        close.index.name = "date"
        frames.append(close)

    if not frames:
        return pd.DataFrame()

    combined = frames[0] if len(frames) == 1 else frames[0].join(frames[1], how="outer")
    combined = combined.sort_index()

    available = [s for s in symbols if s in combined.columns]
    missing   = [s for s in symbols if s not in combined.columns]
    if missing:
        logger.warning("Symbols not returned by Alpaca: %s", missing)

    return combined[available].dropna(how="all")


# ── Asset group resolver ──────────────────────────────────────────────────────

_WARNED_GROUPS: set = set()


def _emit_group_warning(group) -> None:
    """Log a one-line warning once per group per process run."""
    warn_text = getattr(group, "warning", "") or ""
    if not warn_text or group.name in _WARNED_GROUPS:
        return
    _WARNED_GROUPS.add(group.name)
    logger.warning("Asset group '%s': %s", group.name, warn_text.strip())


def _resolve_symbols(config: Dict, asset_group: Optional[str], symbols_arg: Optional[str]) -> List[str]:
    """
    Resolve the symbol list in priority order:
      1. --symbols flag (comma-separated)
      2. --asset-group flag  → AssetGroupRegistry (config/asset_groups.yaml)
      3. broker.asset_group in config → AssetGroupRegistry
      4. Legacy fallback: config['asset_groups'][name] (settings.yaml bloc)
      5. broker.symbols list in config
    """
    if symbols_arg:
        return [s.strip() for s in symbols_arg.split(",")]

    group_name = asset_group or config.get("broker", {}).get("asset_group")

    if group_name:
        # Primary: dedicated registry
        try:
            from core.asset_groups import load_default_registry
            reg = load_default_registry(reload=True)
            if reg.has(group_name):
                grp = reg.get(group_name)
                _emit_group_warning(grp)
                return list(grp.symbols)
        except Exception as exc:
            logger.debug("AssetGroupRegistry unavailable (%s); trying legacy bloc", exc)

        # Legacy fallback: old bloc in settings.yaml
        legacy = config.get("asset_groups", {}) or {}
        if group_name in legacy:
            logger.warning(
                "Asset group '%s' loaded from legacy settings.yaml bloc; "
                "please migrate to config/asset_groups.yaml", group_name)
            return list(legacy[group_name])

        logger.warning("Asset group '%s' not found; falling back to broker.symbols", group_name)

    return config.get("broker", {}).get("symbols", ["SPY"])


# ── Rich helpers ───────────────────────────────────────────────────────────────

def _console():
    try:
        import sys
        from rich.console import Console
        # Let Rich auto-detect TTY so ANSI codes are not injected when output
        # is piped or captured (e.g. param_sweep subprocess, grep pipelines).
        return Console(legacy_windows=False, force_terminal=sys.stdout.isatty())
    except ImportError:
        return None


def _print(msg: str, console=None, style: str = "") -> None:
    if console:
        console.print(msg, style=style)
    else:
        print(msg)


def _print_run_config(
    symbols:        List[str],
    start_date:     str,
    end_date:       str,
    config:         Dict,
    console=None,
    label:          str = "Run Configuration",
) -> None:
    """
    Print a compact recap of all parameters used in a backtest or sweep run.
    Shown before every results table so results are always self-contained.
    """
    broker_cfg   = config.get("broker",   {})
    hmm_cfg      = config.get("hmm",      {})
    strategy_cfg = config.get("strategy", {})
    risk_cfg     = config.get("risk",      {})
    bt_cfg       = config.get("backtest", {})

    lines = [
        f"\n{'─' * 62}",
        f"  {label}",
        f"{'─' * 62}",
        f"  Assets      : {', '.join(symbols)}",
        f"  Period      : {start_date}  →  {end_date}",
        f"  Frequency   : 1Day bars (backtest — live broker uses {broker_cfg.get('timeframe', '5Min')})",
        f"  Capital     : ${float(bt_cfg.get('initial_capital', 100_000)):,.0f}   "
        f"Slippage: {float(bt_cfg.get('slippage_pct', 0.0005)) * 10_000:.1f} bps",
        f"  Walk-fwd    : IS {bt_cfg.get('train_window', 252)} / "
        f"OOS {bt_cfg.get('test_window', 126)} bars  "
        f"step {bt_cfg.get('step_size', 126)}",
        f"",
        f"  HMM         : states {hmm_cfg.get('n_candidates', [3,4,5,6,7])}  "
        f"covariance={hmm_cfg.get('covariance_type', 'full')}  "
        f"n_init={hmm_cfg.get('n_init', 10)}",
        f"  Regime filt : stability={hmm_cfg.get('stability_bars', 3)}  "
        f"flicker_thresh={hmm_cfg.get('flicker_threshold', 4)}  "
        f"flicker_win={hmm_cfg.get('flicker_window', 20)}  "
        f"min_conf={hmm_cfg.get('min_confidence', 0.55)}",
        f"  Features    : {', '.join(_hmm_feature_names(hmm_cfg))}  "
        f"(blend: US equity only — excl. {', '.join(hmm_cfg.get('blend_exclude', ['GLD']))})",
        f"",
        f"  Allocation  : low_vol={strategy_cfg.get('low_vol_allocation', 0.95)}×"
        f"{strategy_cfg.get('low_vol_leverage', 1.25)}x  "
        f"mid={strategy_cfg.get('mid_vol_allocation_trend', 0.95)}/"
        f"{strategy_cfg.get('mid_vol_allocation_no_trend', 0.60)}  "
        f"high={strategy_cfg.get('high_vol_allocation', 0.60)}",
        f"  Rebalance   : threshold={strategy_cfg.get('rebalance_threshold', 0.10)}  "
        f"trend_lookback={strategy_cfg.get('trend_lookback', 50)}",
        f"  Risk        : max_pos={risk_cfg.get('max_single_position', 0.15)}  "
        f"max_exp={risk_cfg.get('max_exposure', 0.80)}  "
        f"dd_halt={risk_cfg.get('daily_dd_halt', 0.03)}",
        f"{'─' * 62}",
    ]
    for line in lines:
        _print(line, console, style="dim")



def _build_run_context(
    symbols: List[str],
    start_date: str,
    end_date: str,
    config: Dict,
    output_dir: Path,
    asset_group: Optional[str] = None,
    n_folds: Optional[int] = None,
) -> Dict:
    """Build a dict of run metadata for config-control headers and JSON sidecar."""
    bt_cfg  = config.get("backtest", {})
    hmm_cfg = config.get("hmm", {})

    # git short hash — best-effort
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(_ROOT), stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        git_hash = "unknown"

    # symbol descriptions from asset_groups registry
    sym_descs: Dict[str, str] = {}
    if asset_group:
        try:
            from core.asset_groups import load_default_registry
            reg = load_default_registry()
            if reg.has(asset_group):
                grp = reg.get(asset_group)
                sym_descs = {s: grp.description for s in symbols}
        except Exception:
            pass

    return {
        "asset_group":         asset_group or "—",
        "symbols":             list(symbols),
        "symbol_descriptions": sym_descs,
        "start_date":          start_date,
        "end_date":            end_date,
        "run_timestamp":       dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "machine":             socket.gethostname(),
        "python_version":      f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "git_hash":            git_hash,
        "output_path":         str(output_dir),
        "train_window":        int(bt_cfg.get("train_window", 252)),
        "test_window":         int(bt_cfg.get("test_window", 126)),
        "step_size":           int(bt_cfg.get("step_size", 126)),
        "n_states":            hmm_cfg.get("n_candidates", [3, 4, 5]),
        "n_folds":             n_folds,
    }


def _print_comparison_table(
    strategy_report,
    bnh_metrics: Optional[Dict],
    sma_metrics: Optional[Dict],
    ema_cross_metrics: Optional[Dict],
    rand_metrics: Optional[Dict],
    console=None,
    n_symbols: int = 1,
    sma_long: int = 200,
    ema_fast: int = 9,
    ema_slow: int = 45,
) -> None:
    """Print a side-by-side benchmark comparison table."""
    # (label, attr_key, format_str, higher_is_better)
    rows = [
        ("Total Return",     "total_return",    "{:+.2%}",  True),
        ("CAGR",             "cagr",            "{:+.2%}",  True),
        ("Sharpe Ratio",     "sharpe_ratio",    "{:.3f}",   True),
        ("Sortino Ratio",    "sortino_ratio",   "{:.3f}",   True),
        ("Calmar Ratio",     "calmar_ratio",    "{:.3f}",   True),
        ("Max Drawdown",     "max_drawdown",    "{:.2%}",   False),
        ("Ann. Volatility",  "annualised_vol",  "{:.2%}",   False),
        ("Win Rate",         "win_rate",        "{:.2%}",   True),
        ("Profit Factor",    "profit_factor",   "{:.2f}",   True),
    ]

    def _raw(d, key):
        if d is None:
            return None
        v = d.get(key) if isinstance(d, dict) else getattr(d, key, None)
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    def _val(d, key, fmt):
        v = _raw(d, key)
        if v is None:
            return "N/A"
        try:
            return fmt.format(v)
        except Exception:
            return str(v)

    def _delta_str(strat, bnh, key, fmt, higher_better):
        sv = _raw(strat, key)
        bv = _raw(bnh, key)
        if sv is None or bv is None:
            return "N/A"
        d = sv - bv
        is_better = (d > 0) if higher_better else (d < 0)
        s = (f"{d:+.2%}" if "%" in fmt else f"{d:+.3f}")
        return f"[green]{s}[/green]" if is_better else f"[red]{s}[/red]"

    def _delta_plain(strat, bnh, key, fmt, higher_better):
        sv = _raw(strat, key)
        bv = _raw(bnh, key)
        if sv is None or bv is None:
            return "N/A"
        d = sv - bv
        return (f"{d:+.2%}" if "%" in fmt else f"{d:+.3f}")

    bnh_label       = "Buy & Hold (EW)" if n_symbols > 1 else "Buy & Hold"
    sma_label       = (f"SMA-{sma_long} (EW)"        if n_symbols > 1 else f"SMA-{sma_long}")
    ema_cross_label = (f"EMA {ema_fast}/{ema_slow} (EW)" if n_symbols > 1
                       else f"EMA {ema_fast}/{ema_slow}")

    if console:
        try:
            from rich.table import Table
            from rich import box as rbox
            tbl = Table(title="Benchmark Comparison", box=rbox.ROUNDED,
                        header_style="bold yellow")
            tbl.add_column("Metric",          style="dim", min_width=18)
            tbl.add_column("Strategy",        justify="right")
            tbl.add_column("Δ vs BnH",        justify="right", style="dim")
            tbl.add_column(bnh_label,         justify="right")
            tbl.add_column(sma_label,         justify="right")
            tbl.add_column(ema_cross_label,   justify="right")
            for label, key, fmt, higher_better in rows:
                tbl.add_row(
                    label,
                    _val(strategy_report, key, fmt),
                    _delta_str(strategy_report, bnh_metrics, key, fmt, higher_better),
                    _val(bnh_metrics, key, fmt),
                    _val(sma_metrics, key, fmt),
                    _val(ema_cross_metrics, key, fmt),
                )
            console.print(tbl)
            return
        except ImportError:
            pass

    print("\nBENCHMARK COMPARISON")
    print(f"{'Metric':<20} {'Strategy':>12} {'Δ vs BnH':>10} "
          f"{'BnH':>14} {f'SMA-{sma_long}':>12} {f'EMA {ema_fast}/{ema_slow}':>14}")
    print("-" * 86)
    for label, key, fmt, higher_better in rows:
        print(
            f"{label:<20} {_val(strategy_report, key, fmt):>12} "
            f"{_delta_plain(strategy_report, bnh_metrics, key, fmt, higher_better):>10} "
            f"{_val(bnh_metrics, key, fmt):>14} "
            f"{_val(sma_metrics, key, fmt):>12} "
            f"{_val(ema_cross_metrics, key, fmt):>14}"
        )


# ── Timeframe helpers ──────────────────────────────────────────────────────────

# Minutes per bar for each supported timeframe
_TF_MINUTES: Dict[str, int] = {
    "1Min":  1,
    "5Min":  5,
    "15Min": 15,
    "30Min": 30,
    "1Hour": 60,
    "4Hour": 240,
    "1Day":  390,   # treat daily as end-of-session (6.5 hours)
    "1Week": 1950,
}

# NYSE open/close in UTC — static fallback only. The live path resolves
# market hours via pandas_market_calendars, which correctly handles DST
# transitions (EST close = 21:00 UTC, EDT close = 20:00 UTC) and the
# full NYSE holiday schedule (Thanksgiving, Christmas, early-close days,
# etc.). If the library is unavailable the old naive behavior kicks in.
_MARKET_OPEN_UTC  = dt.time(14, 30)   # 09:30 EST ≈ 14:30 UTC (fallback)
_MARKET_CLOSE_UTC = dt.time(21,  0)   # 16:00 EST ≈ 21:00 UTC (fallback)

try:
    import pandas_market_calendars as _mcal  # type: ignore
    _NYSE_CAL = _mcal.get_calendar("NYSE")
except Exception:  # pragma: no cover — degrade to naive fallback
    _NYSE_CAL = None


def _next_trading_day_close_utc(now: dt.datetime) -> dt.datetime:
    """Return the next NYSE session close ≥ now (UTC, honors DST + holidays)."""
    if _NYSE_CAL is not None:
        # Look ahead 10 calendar days to guarantee we catch the next session
        # even around Christmas/Thanksgiving clusters.
        sched = _NYSE_CAL.schedule(
            start_date=(now - dt.timedelta(days=1)).date(),
            end_date=(now + dt.timedelta(days=10)).date(),
        )
        for _, row in sched.iterrows():
            close_utc = row["market_close"].to_pydatetime()
            if close_utc.tzinfo is None:
                close_utc = close_utc.replace(tzinfo=dt.timezone.utc)
            if close_utc > now:
                return close_utc
    # Fallback: 21:00 UTC, skip weekends (no holiday awareness)
    today_close = now.replace(
        hour=_MARKET_CLOSE_UTC.hour,
        minute=_MARKET_CLOSE_UTC.minute,
        second=0, microsecond=0,
    )
    if now >= today_close:
        today_close += dt.timedelta(days=1)
    while today_close.weekday() >= 5:
        today_close += dt.timedelta(days=1)
    return today_close


def _next_bar_close_utc(timeframe: str) -> dt.datetime:
    """
    Return the next bar-close UTC datetime for the given timeframe.
    For daily bars this is the next NYSE session close (DST + holidays aware).
    For intraday bars this is the next N-minute boundary after now.
    """
    now = dt.datetime.now(dt.timezone.utc)
    minutes = _TF_MINUTES.get(timeframe, 5)

    if timeframe in ("1Day", "1Week"):
        return _next_trading_day_close_utc(now)

    # Round up to next N-minute boundary
    epoch = dt.datetime(2000, 1, 1, tzinfo=dt.timezone.utc)
    elapsed = (now - epoch).total_seconds() / 60.0
    next_boundary = (int(elapsed / minutes) + 1) * minutes
    return epoch + dt.timedelta(minutes=next_boundary)


def _seconds_until(target: dt.datetime) -> float:
    delta = (target - dt.datetime.now(dt.timezone.utc)).total_seconds()
    return max(0.0, delta)


# ── State snapshot ─────────────────────────────────────────────────────────────

def _save_snapshot(state: Dict) -> None:
    """Persist a JSON state snapshot for crash recovery."""
    try:
        _SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_SNAPSHOT_PATH, "w") as fh:
            json.dump(state, fh, indent=2, default=str)
        logger.info("State snapshot saved to %s", _SNAPSHOT_PATH)
    except Exception as exc:
        logger.error("Failed to save state snapshot: %s", exc)


def _load_snapshot() -> Optional[Dict]:
    """Load a prior state snapshot if it exists."""
    if not _SNAPSHOT_PATH.exists():
        return None
    try:
        with open(_SNAPSHOT_PATH) as fh:
            data = json.load(fh)
        logger.info("Loaded state snapshot from %s", _SNAPSHOT_PATH)
        return data
    except Exception as exc:
        logger.warning("Could not load state snapshot: %s", exc)
        return None


# ── HMM training / loading ─────────────────────────────────────────────────────

def _hmm_needs_retrain(model_path: Path, max_age_days: int) -> bool:
    """Return True if the model file is missing or older than max_age_days."""
    if not model_path.exists():
        return True
    age = dt.datetime.now() - dt.datetime.fromtimestamp(model_path.stat().st_mtime)
    return age.days >= max_age_days


def _train_hmm(
    client,
    symbols: List[str],
    hmm_cfg: Dict,
    console=None,
    n_bars: int = 5_000,  # ~3 months of 5-min bars (78 bars/day × 65 days)
    progress_cb=None,
) -> "HMMEngine":
    """Fetch data, compute features, fit HMM, save to disk. Returns fitted engine.

    Parameters
    ----------
    progress_cb : Optional[Callable[[str, int, int], None]]
        Invoked at the start of each stage with (stage_label, step_idx, total).
        Useful for plugging an external progress bar (e.g. Rich Progress,
        tqdm, or a GUI). When ``console`` is a Rich Console, the function
        also displays a spinner next to the current stage via
        ``console.status`` for built-in visual feedback.
    """
    from contextlib import nullcontext
    from core.hmm_engine import HMMEngine
    from data.feature_engineering import FeatureEngineer

    tf = hmm_cfg.get("timeframe", "5Min")

    # ── Progress helper ────────────────────────────────────────────────────
    # Emits a stage label through three channels:
    #   1. the external `progress_cb` callback (for programmatic monitoring),
    #   2. `_print` (permanent line in the output log),
    #   3. a transient Rich `console.status()` spinner held until the
    #      returned context manager exits.
    _TOTAL_STAGES = 5

    def _stage(idx: int, label: str):
        if progress_cb is not None:
            try:
                progress_cb(label, idx, _TOTAL_STAGES)
            except Exception as exc:
                logger.debug("progress_cb raised (ignored): %s", exc)
        _print(f"  [{idx}/{_TOTAL_STAGES}] {label}", console, style="dim")
        if console is not None and hasattr(console, "status"):
            return console.status(f"[dim]{label}[/dim]")
        return nullcontext()

    _print(f"Training HMM on {tf} bars ...", console, style="dim")

    # Bars per calendar day for each timeframe (trading days only, ~252/year)
    _bars_per_day = {
        "1Min": 390, "5Min": 78, "15Min": 26, "30Min": 13,
        "1Hour": 7,  "4Hour": 2, "1Day": 1,   "1Week": 0.2,
    }
    bars_per_day  = _bars_per_day.get(tf, 78)
    # Add 40% buffer for weekends/holidays
    lookback_days = max(10, int(n_bars / bars_per_day * 1.4 * 7 / 5) + 5)

    # Fetch all symbols so log_ret_1 / realized_vol_20 can be basket-blended.
    # Blending is restricted to US equity ETFs (blend_exclude removes GLD, VNQ,
    # EFA, EEM, EWG): non-equity / international assets are held for portfolio
    # diversification but their regime dynamics differ too much from US equity
    # to contribute cleanly to HMM calibration.
    # vol_ratio, adx_14, dist_sma200 remain anchored to the reference symbol.
    _proxy = hmm_cfg.get("regime_proxy") or None
    ref_symbol = (_proxy if _proxy and _proxy in symbols else None) or symbols[0]
    end   = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    start = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=lookback_days)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )

    with _stage(1, f"Fetching bars: {len(symbols)} symbols, {lookback_days}d lookback"):
        bars_df = client.get_bars(
            symbols=symbols,
            timeframe=tf,
            start=start,
            end=end,
        )

    if bars_df.empty:
        raise RuntimeError(f"No bars returned during HMM training")

    def _extract_sym(df, sym):
        if isinstance(df.index, pd.MultiIndex):
            lvl_vals = df.index.get_level_values("symbol")
            if sym in lvl_vals:
                out = df.xs(sym, level="symbol")
            else:
                out = df.droplevel(0)
        else:
            out = df
        out = out.copy()
        out.index = pd.to_datetime(out.index).tz_localize(None)
        out = out.sort_index()
        if "volume" not in out.columns:
            out["volume"] = 1_000_000.0
        return out

    sym_bars = _extract_sym(bars_df, ref_symbol)

    fe = FeatureEngineer()
    with _stage(2, f"Computing features for {ref_symbol}"):
        _vix = _maybe_fetch_vix(hmm_cfg, sym_bars.index)
        _credit = _maybe_fetch_credit(hmm_cfg, sym_bars.index)
        features_clean = fe.build_feature_matrix(
            sym_bars, feature_names=_hmm_feature_names(hmm_cfg),
            vix_series=_vix, credit_series=_credit,
        )

    # Blend log_ret_1 and realized_vol_20 across equity-like basket symbols.
    # Non-equity symbols (e.g. GLD, TLT, USO) are excluded via hmm_cfg.blend_exclude.
    _per_symbol_bars = {}
    for _s in symbols:
        try:
            _per_symbol_bars[_s] = sym_bars if _s == ref_symbol else _extract_sym(bars_df, _s)
        except Exception:
            continue
    with _stage(3, f"Blending features across {len(_per_symbol_bars)} symbols"):
        _blend_excl_train = (
            [s for s in symbols if s != ref_symbol]
            if hmm_cfg.get("regime_proxy") else
            hmm_cfg.get("blend_exclude", [])
        )
        features_clean = _blend_cross_symbol_features(
            features_clean,
            _per_symbol_bars,
            feature_engineer=fe,
            blend_exclude=_blend_excl_train,
            min_bars=0,
        )

    if len(features_clean) < hmm_cfg.get("min_train_bars", 252):
        raise RuntimeError(
            f"Only {len(features_clean)} clean feature rows — "
            f"need {hmm_cfg.get('min_train_bars', 252)} for HMM training."
        )

    # Intraday bars are noisier — full covariance matrices can be singular.
    # Use diagonal covariance + higher min_covar for sub-daily timeframes.
    _is_intraday = tf not in ("1Day", "1Week")
    _min_covar   = hmm_cfg.get("min_covar",        0.05  if _is_intraday else 1e-3)
    _cov_type    = hmm_cfg.get("covariance_type",  "diag" if _is_intraday else "full")

    engine = HMMEngine(
        n_candidates       = hmm_cfg.get("n_candidates", [3, 4, 5]),
        n_init             = hmm_cfg.get("n_init", 10),
        stability_bars     = hmm_cfg.get("stability_bars", 3),
        flicker_window     = hmm_cfg.get("flicker_window", 20),
        flicker_threshold  = hmm_cfg.get("flicker_threshold", 4),
        min_confidence     = hmm_cfg.get("min_confidence", 0.55),
        min_train_bars     = hmm_cfg.get("min_train_bars", 252),
        min_covar          = _min_covar,
        covariance_type    = _cov_type,
    )
    _n_cands = hmm_cfg.get("n_candidates", [3, 4, 5])
    _n_init  = hmm_cfg.get("n_init", 10)
    with _stage(
        4,
        f"Fitting HMM: {len(features_clean)} bars × candidates={_n_cands} × n_init={_n_init}",
    ):
        engine.fit(features_clean.values)
    engine._n_train_bars = len(features_clean)   # stored for logging

    # Attach in-sample regime sequence for post-training stats display
    try:
        _raw_states = engine._model.predict(features_clean.values)
        engine._training_regimes = pd.Series(
            [engine.get_state_label(s) for s in _raw_states],
            index=features_clean.index,
            name="regime",
        )
    except Exception:
        engine._training_regimes = None

    with _stage(5, f"Saving model → {_MODEL_PATH.name}"):
        engine.save(str(_MODEL_PATH))

    _print(
        f"  HMM trained: {engine._n_states} states  "
        f"BIC={engine._training_bic:.2f}  "
        f"bars={len(features_clean)}",
        console, style="dim",
    )
    return engine


def _load_or_train_hmm(
    client,
    symbols: List[str],
    hmm_cfg: Dict,
    force_retrain: bool = False,
    console=None,
) -> "HMMEngine":
    """Load model from disk if fresh, else retrain."""
    from core.hmm_engine import HMMEngine

    needs_train = force_retrain or _hmm_needs_retrain(_MODEL_PATH, _MODEL_MAX_AGE_DAYS)

    if not needs_train:
        try:
            engine = HMMEngine.load(str(_MODEL_PATH))
            age_days = (
                dt.datetime.now() -
                dt.datetime.fromtimestamp(_MODEL_PATH.stat().st_mtime)
            ).days
            _print(
                f"  Loaded HMM model from {_MODEL_PATH}  "
                f"(age={age_days}d  states={engine._n_states})",
                console, style="dim",
            )
            return engine
        except Exception as exc:
            logger.warning("Failed to load HMM model: %s  -- retraining", exc)

    return _train_hmm(client, symbols, hmm_cfg, console=console)


# ── Bar fetching for live loop ─────────────────────────────────────────────────

def _fetch_live_bars(
    client,
    symbols: List[str],
    timeframe: str,
    n_bars: int = 300,
) -> Dict[str, pd.DataFrame]:
    """
    Fetch the latest n_bars OHLCV bars for each symbol.

    Returns dict of symbol -> DataFrame with columns [open, high, low, close, volume].
    """
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.enums import Adjustment
    from broker.alpaca_client import parse_timeframe

    tf = parse_timeframe(timeframe)
    # Determine start date from n_bars lookback
    bars_per_day = {
        "1Min": 390, "5Min": 78, "15Min": 26, "30Min": 13,
        "1Hour": 7,  "4Hour": 2, "1Day":  1,  "1Week": 0.2,
    }.get(timeframe, 1)
    # lookback must cover the full n_bars *plus* the feature warm-up period
    # (dist_sma200 needs 200 bars; rolling z-score needs 252 more → 452 warm-up).
    # Add a 40% calendar buffer for weekends/holidays.
    warmup_bars   = n_bars + 500          # 500-bar buffer covers all warm-up windows
    lookback_days = max(15, int(warmup_bars / bars_per_day * 1.4 * 7 / 5) + 5)
    start = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=lookback_days)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    # Do NOT pass a limit — let the date range control volume, then tail() selects
    # the most recent n_bars.  A hard limit would truncate to the oldest bars.
    req = StockBarsRequest(
        symbol_or_symbols = symbols,
        timeframe          = tf,
        start              = start,
        adjustment         = Adjustment.ALL,
    )
    bars = client._data_client.get_stock_bars(req)
    raw  = bars.df

    result: Dict[str, pd.DataFrame] = {}
    if raw.empty:
        return result

    # Split multi-index by symbol
    if isinstance(raw.index, pd.MultiIndex):
        # MultiIndex is (symbol, timestamp) or (timestamp, symbol)
        levels = raw.index.names
        sym_level = levels.index("symbol") if "symbol" in levels else 0
        ts_level  = 1 - sym_level
        for sym in symbols:
            try:
                sym_df = raw.xs(sym, level=sym_level)
            except KeyError:
                continue
            sym_df.index = pd.to_datetime(sym_df.index).tz_localize(None)
            # Keep n_bars + 500 so feature warm-up (dist_sma200=200, z-score=252)
            # produces valid rows for the most recent n_bars.
            sym_df = sym_df.sort_index().tail(n_bars + 500)
            if "volume" not in sym_df.columns:
                sym_df["volume"] = 1_000_000.0
            result[sym] = sym_df
    else:
        for sym in symbols:
            if sym in raw.columns:
                result[sym] = raw[[sym]].rename(columns={sym: "close"})

    return result


# ── Session summary ────────────────────────────────────────────────────────────

def _print_session_summary(
    session_start: dt.datetime,
    equity_start: float,
    equity_end: float,
    bar_count: int,
    trade_count: int,
    console=None,
) -> None:
    elapsed    = dt.datetime.now(dt.timezone.utc) - session_start
    total_pnl  = equity_end - equity_start
    pnl_pct    = total_pnl / max(equity_start, 1.0)
    sign       = "+" if total_pnl >= 0 else ""

    _print("\n" + "=" * 55, console)
    _print("[bold]Session Summary[/bold]", console)
    _print("=" * 55, console)
    _print(f"  Duration     : {str(elapsed).split('.')[0]}", console)
    _print(f"  Bars         : {bar_count}", console)
    _print(f"  Trades       : {trade_count}", console)
    _print(f"  Equity start : ${equity_start:>12,.2f}", console)
    _print(f"  Equity end   : ${equity_end:>12,.2f}", console)
    _print(
        f"  Session P&L  : {sign}${abs(total_pnl):>10,.2f}  ({sign}{pnl_pct:.2%})",
        console,
        style="green" if total_pnl >= 0 else "red",
    )
    _print("=" * 55, console)


# ── Trading session ────────────────────────────────────────────────────────────

class TradingSession:
    """
    Encapsulates the entire live trading lifecycle:
    startup -> main loop -> shutdown.
    """

    def __init__(
        self,
        config: Dict,
        dry_run: bool = False,
        allocator_approach: str = "inverse_vol",
        no_portfolio_risk: bool = False,
        multi_strat_filter: Optional[List[str]] = None,
    ) -> None:
        self.config  = config
        self.dry_run = dry_run
        self.console = _console()

        # Component references (set during startup)
        self.client           = None
        self.hmm_engine       = None
        self.strategy         = None
        self.risk_manager     = None
        self.position_tracker = None
        self.dashboard        = None
        self.trade_logger     = None
        self.alert_manager    = None

        # Session state
        self._running:         bool           = False
        self._session_start:   dt.datetime    = dt.datetime.now(dt.timezone.utc)
        self._equity_at_start: float          = 0.0
        self._bar_count:       int            = 0
        self._trade_count:     int            = 0
        self._last_retrain:    dt.datetime    = dt.datetime.now(dt.timezone.utc)
        self._snapshot:        Optional[Dict] = None
        self._last_regime:     str            = "UNKNOWN"
        self._price_history:   Dict[str, pd.Series] = {}  # for risk manager

        # Multi-strategy state (populated in startup() when ≥2 strategies enabled)
        self._multi_strat_mode:    bool               = False
        self._strat_orchestrators: Dict[str, object]  = {}
        self._strat_risk_managers: Dict[str, object]  = {}
        self._strat_symbols:       Dict[str, List[str]] = {}
        self.strategy_registry                         = None
        self.capital_allocator                         = None
        self.portfolio_rm                              = None
        self._alloc_weights:       Dict[str, float]   = {}
        self._bars_since_alloc:    int                 = 0
        self._alloc_interval_bars: int                 = 5   # weekly ≈ 5 trading days
        self._no_portfolio_risk:   bool                = no_portfolio_risk
        self._allocator_approach:  str                 = allocator_approach
        self._multi_strat_filter:  Optional[List[str]] = multi_strat_filter

    # ======================================================================= #
    # Startup                                                                  #
    # ======================================================================= #

    def startup(self) -> None:
        """Full system startup sequence."""
        console = self.console
        broker_cfg   = self.config.get("broker",     {})
        hmm_cfg      = self.config.get("hmm",        {})
        risk_cfg     = self.config.get("risk",        {})
        monitor_cfg  = self.config.get("monitoring",  {})

        symbols   = broker_cfg.get("symbols", ["SPY"])
        timeframe = broker_cfg.get("timeframe", "1Day")
        paper     = broker_cfg.get("paper_trading", True)

        _print(
            f"\n[bold cyan]Regime Trader[/bold cyan]  "
            f"{'[yellow]DRY RUN[/yellow]' if self.dry_run else '[green]LIVE[/green]'}  "
            f"{'PAPER' if paper else '[red]LIVE MONEY[/red]'}",
            console,
        )

        # ── 1. Connect to Alpaca ──────────────────────────────────────────────
        _print("\n[1/7] Connecting to Alpaca ...", console)
        from broker.alpaca_client import AlpacaClient
        self.client = AlpacaClient(paper=paper)
        self.client.connect_with_retry(max_attempts=3, base_delay=2.0)

        account = self.client.get_account()
        equity  = float(account.equity)
        _print(
            f"      Account  : {account.account_number}  "
            f"status={account.status.value}",
            console, style="dim",
        )
        _print(f"      Equity   : ${equity:,.2f}", console, style="dim")
        _print(f"      Cash     : ${float(account.cash):,.2f}", console, style="dim")

        # ── 2. Market hours check ─────────────────────────────────────────────
        _print("\n[2/7] Checking market hours ...", console)
        clock = self.client.get_clock()
        if not clock.is_open:
            next_open = clock.next_open
            _print(
                f"      Market is CLOSED.  Next open: {next_open}",
                console, style="yellow",
            )
            _print(
                "      Continuing -- will wait for next bar close.",
                console, style="dim",
            )

        # ── 3. Load or train HMM ──────────────────────────────────────────────
        _print("\n[3/7] Loading HMM model ...", console)
        self.hmm_engine = _load_or_train_hmm(
            self.client, symbols, hmm_cfg, console=console
        )

        # Build StrategyOrchestrator from fitted regime infos
        from core.regime_strategies import StrategyOrchestrator
        self.strategy = StrategyOrchestrator(
            config        = self.config,
            regime_infos  = self.hmm_engine.get_all_regime_info(),
            min_confidence = hmm_cfg.get("min_confidence", 0.55),
            rebalance_threshold = self.config.get("strategy", {}).get(
                "rebalance_threshold", 0.10
            ),
        )
        _print(
            f"      Strategy orchestrator ready  "
            f"({self.hmm_engine._n_states} states)",
            console, style="dim",
        )

        # ── 4. Risk manager ───────────────────────────────────────────────────
        _print("\n[4/7] Initialising risk manager ...", console)
        from core.risk_manager import RiskManager
        self.risk_manager = RiskManager(
            initial_equity         = equity,
            max_risk_per_trade     = risk_cfg.get("max_risk_per_trade",   0.01),
            max_exposure           = risk_cfg.get("max_exposure",         0.80),
            max_leverage           = risk_cfg.get("max_leverage",         1.25),
            max_single_position    = risk_cfg.get("max_single_position",  0.15),
            max_concurrent         = risk_cfg.get("max_concurrent",       5),
            max_daily_trades       = risk_cfg.get("max_daily_trades",     20),
            daily_dd_reduce        = risk_cfg.get("daily_dd_reduce",      0.02),
            daily_dd_halt          = risk_cfg.get("daily_dd_halt",        0.03),
            weekly_dd_reduce       = risk_cfg.get("weekly_dd_reduce",     0.05),
            weekly_dd_halt         = risk_cfg.get("weekly_dd_halt",       0.07),
            max_dd_from_peak       = risk_cfg.get("max_dd_from_peak",     0.10),
            allow_fractional_shares= bool(risk_cfg.get("allow_fractional_shares", False)),
            fractional_precision   = int(risk_cfg.get("fractional_precision", 6)),
        )
        _print(
            f"      RiskManager ready  equity=${equity:,.2f}",
            console, style="dim",
        )

        # TradeLogger must exist before step 5 (set_context is called after
        # position sync). AlertManager and Dashboard are created later, after
        # port_snapshot is available.
        from monitoring.logger import TradeLogger
        self.trade_logger = TradeLogger(
            log_dir   = monitor_cfg.get("log_dir", "logs/"),
            log_level = monitor_cfg.get("log_level", "INFO"),
        )
        self.trade_logger.setup()

        # ── 5. Position tracker ───────────────────────────────────────────────
        _print("\n[5/7] Syncing positions ...", console)
        from broker.position_tracker import PositionTracker
        self.position_tracker = PositionTracker(
            client            = self.client,
            risk_manager      = self.risk_manager,
            current_regime_fn = lambda: self._last_regime,
        )
        port_snapshot = self.position_tracker.startup_sync(
            current_regime=self._last_regime
        )
        self._equity_at_start = port_snapshot.total_equity

        # Sync logger context
        self.trade_logger.set_context(
            regime      = self._last_regime,
            equity      = port_snapshot.total_equity,
            positions   = [p.symbol for p in port_snapshot.positions],
        )
        _print(
            f"      {len(port_snapshot.positions)} open position(s)  "
            f"equity=${port_snapshot.total_equity:,.2f}",
            console, style="dim",
        )

        # ── 6. State snapshot recovery ────────────────────────────────────────
        _print("\n[6/7] Checking for prior session snapshot ...", console)
        self._snapshot = _load_snapshot()
        if self._snapshot:
            self._last_regime  = self._snapshot.get("regime", "UNKNOWN")
            self._bar_count    = self._snapshot.get("bar_count", 0)
            self._trade_count  = self._snapshot.get("trade_count", 0)
            last_ts = self._snapshot.get("session_start", "")
            _print(
                f"      Recovered from snapshot  "
                f"regime={self._last_regime}  "
                f"bars={self._bar_count}  "
                f"prior_session={last_ts}",
                console, style="dim",
            )
        else:
            _print("      No prior snapshot -- clean start.", console, style="dim")

        # ── 7. Start WebSocket feeds ──────────────────────────────────────────
        _print("\n[7/7] Starting WebSocket feeds ...", console)
        self.position_tracker.start_stream()
        _print("      TradingStream (fills) active.", console, style="dim")

        # ── 8. Multi-strategy setup (when ≥2 strategies enabled in yaml) ──────
        _strat_cfgs = self.config.get("strategies", {}) or {}
        _enabled_strats = {
            name: cfg for name, cfg in _strat_cfgs.items()
            if cfg.get("enabled", True)
            and (self._multi_strat_filter is None or name in self._multi_strat_filter)
        }
        if len(_enabled_strats) >= 2:
            self._multi_strat_mode = True
            _print(
                f"\n[8/8] Multi-strategy mode — strategies: {list(_enabled_strats.keys())}",
                console,
            )
            from core.strategy_registry import StrategyRegistry
            from core.capital_allocator import CapitalAllocator
            from core.risk_manager import PortfolioRiskManager as _PRM
            from core.regime_strategies import StrategyOrchestrator
            from core.risk_manager import RiskManager
            from backtest.multi_strategy_backtester import _BacktestProxy

            registry = StrategyRegistry.instance()
            per_strat_equity = equity / len(_enabled_strats)
            risk_cfg = self.config.get("risk", {})

            for sname, scfg in _enabled_strats.items():
                syms = scfg.get("symbols") or symbols
                self._strat_symbols[sname] = syms

                orch = StrategyOrchestrator(
                    config              = self.config,
                    regime_infos        = self.hmm_engine.get_all_regime_info(),
                    min_confidence      = hmm_cfg.get("min_confidence", 0.55),
                    rebalance_threshold = self.config.get("strategy", {}).get(
                        "rebalance_threshold", 0.10
                    ),
                )
                rm = RiskManager(
                    initial_equity          = per_strat_equity,
                    max_risk_per_trade      = risk_cfg.get("max_risk_per_trade",   0.01),
                    max_exposure            = risk_cfg.get("max_exposure",         0.80),
                    max_leverage            = risk_cfg.get("max_leverage",         1.25),
                    max_single_position     = risk_cfg.get("max_single_position",  0.15),
                    max_concurrent          = risk_cfg.get("max_concurrent",       5),
                    max_daily_trades        = risk_cfg.get("max_daily_trades",     20),
                    daily_dd_reduce         = risk_cfg.get("daily_dd_reduce",      0.02),
                    daily_dd_halt           = risk_cfg.get("daily_dd_halt",        0.03),
                    weekly_dd_reduce        = risk_cfg.get("weekly_dd_reduce",     0.05),
                    weekly_dd_halt          = risk_cfg.get("weekly_dd_halt",       0.07),
                    max_dd_from_peak        = risk_cfg.get("max_dd_from_peak",     0.10),
                    allow_fractional_shares = bool(risk_cfg.get("allow_fractional_shares", False)),
                    fractional_precision    = int(risk_cfg.get("fractional_precision", 6)),
                )
                self._strat_orchestrators[sname] = orch
                self._strat_risk_managers[sname]  = rm

                proxy = _BacktestProxy(sname, total_alloc=0.90)
                proxy.allocated_capital = per_strat_equity
                registry.register(sname, proxy)

            self.strategy_registry = registry
            self.capital_allocator = CapitalAllocator(
                approach            = self._allocator_approach,
                strategy_configs    = _enabled_strats,
                total_capital       = equity,
            )
            if not self._no_portfolio_risk:
                self.portfolio_rm = _PRM(
                    max_aggregate_exposure = risk_cfg.get("max_exposure",        0.80),
                    max_single_symbol      = risk_cfg.get("max_single_position", 0.15),
                    max_portfolio_leverage = risk_cfg.get("max_leverage",        1.25),
                    daily_dd_halt          = risk_cfg.get("daily_dd_halt",       0.03),
                    max_dd_from_peak       = risk_cfg.get("max_dd_from_peak",    0.10),
                )

            # Initial allocation pass
            self._alloc_weights = self.capital_allocator.allocate(registry)
            for sname, w in self._alloc_weights.items():
                p = registry.get(sname)
                if p is not None:
                    p.allocated_capital = w * equity

            _print(
                f"      Allocator ready  approach={self._allocator_approach}  "
                f"weights={{{', '.join(f'{k}:{v:.2f}' for k,v in self._alloc_weights.items())}}}",
                console, style="dim",
            )

        # ── Monitoring ────────────────────────────────────────────────────────
        # TradeLogger is already constructed above (before position sync).
        from monitoring.alerts import AlertManager
        from monitoring.dashboard import Dashboard

        self.alert_manager = AlertManager(
            rate_limit_minutes = monitor_cfg.get("alert_rate_limit_minutes", 15),
        )

        self.dashboard = Dashboard(
            refresh_seconds = monitor_cfg.get("dashboard_refresh_seconds", 5),
        )
        self.dashboard.start()

        # Populate dashboard immediately so panels don't show "Waiting..."
        from monitoring.dashboard import SystemStatus
        from core.signal_generator import PortfolioSignal
        _initial_system = SystemStatus(
            data_ok          = True,
            api_ok           = True,
            api_latency_ms   = 0.0,
            hmm_last_trained = self.hmm_engine._training_date.to_pydatetime()
                               if getattr(self.hmm_engine, "_training_date", None) is not None else None,
            mode             = "PAPER" if paper else "LIVE",
        )
        # Predict current regime from recent bars so the dashboard shows real data
        _startup_regime   = self._last_regime
        _startup_conf     = 0.0
        _startup_stable   = False
        _startup_notes    = ["Awaiting first live bar"]
        try:
            _proxy_live = hmm_cfg.get("regime_proxy") or None
            _ref_sym   = (_proxy_live if _proxy_live and _proxy_live in symbols else None) or symbols[0]
            _tf        = broker_cfg.get("timeframe", "5Min")
            _pred_bars = _fetch_live_bars(self.client, symbols, _tf, n_bars=300)
            _ref_df    = _pred_bars.get(_ref_sym)
            if _ref_df is None:
                _startup_notes = [f"Startup predict: no bars returned for {_ref_sym}"]
            elif len(_ref_df) < 10:
                _startup_notes = [f"Startup predict: only {len(_ref_df)} bars for {_ref_sym} (need 10)"]
            else:
                from data.feature_engineering import FeatureEngineer
                from core.hmm_engine import RegimeState
                _fe      = FeatureEngineer()
                _vix = _maybe_fetch_vix(hmm_cfg, _ref_df.index)
                _credit = _maybe_fetch_credit(hmm_cfg, _ref_df.index)
                _feat_df = _fe.build_feature_matrix(
                    _ref_df, feature_names=_hmm_feature_names(hmm_cfg),
                    vix_series=_vix, credit_series=_credit,
                )
                # Blend log_ret_1 / realized_vol_20 across equity-like symbols
                _sp_bars = {_ref_sym: _ref_df}
                for _s in symbols:
                    if _s == _ref_sym:
                        continue
                    _sb = _pred_bars.get(_s)
                    if _sb is not None:
                        _sp_bars[_s] = _sb
                _feat_df = _blend_cross_symbol_features(
                    _feat_df,
                    _sp_bars,
                    feature_engineer=_fe,
                    blend_exclude=(
                        [s for s in symbols if s != _ref_sym]
                        if _proxy_live else
                        hmm_cfg.get("blend_exclude", [])
                    ),
                    min_bars=10,
                )
                if len(_feat_df) < 10:
                    _startup_notes = [f"Startup predict: only {len(_feat_df)} clean feature rows (need 10)"]
                else:
                    _states  = self.hmm_engine.predict_regime_filtered(
                        _feat_df.values,
                        timestamps=[pd.Timestamp(ts) for ts in _feat_df.index],
                    )
                    _rs: RegimeState = _states[-1]
                    _startup_regime  = _rs.label
                    _startup_conf    = _rs.probability
                    _startup_stable  = _rs.is_confirmed
                    _startup_notes   = [f"Predicted from last {len(_feat_df)} bars — awaiting first live update"]
                    self._last_regime = _startup_regime
                    # Warm up the incremental HMM filter so the first live
                    # update() call has proper prior state instead of starting
                    # cold and producing a garbage regime on bar 1.
                    for _row in _feat_df.values:
                        self.hmm_engine.update(
                            new_feature_row = _row,
                            timestamp       = None,
                        )
        except Exception as _exc:
            logger.warning("Startup regime predict failed: %s", _exc)
            _startup_notes = [f"Startup predict failed: {_exc}"]

        _startup_signal = PortfolioSignal(
            timestamp       = pd.Timestamp.now(),
            regime          = _startup_regime,
            confidence      = _startup_conf,
            is_stable       = _startup_stable,
            target_weights  = {},
            delta_weights   = {},
            leverage        = 1.0,
            trading_allowed = True,
            notes           = _startup_notes,
        )
        _startup_tf = broker_cfg.get("timeframe", "5Min")
        self.dashboard.update(
            snapshot       = port_snapshot,
            signal         = _startup_signal,
            drawdown_state = self.risk_manager.get_drawdown_state(),
            system_status  = _initial_system,
            market_open    = clock.is_open,
            next_bar_dt    = _next_bar_close_utc(_startup_tf),
            next_broker_dt = _next_bar_close_utc(_startup_tf),
            next_hmm_dt    = self._last_retrain + dt.timedelta(days=7),
            timeframe      = _startup_tf,
            asset_group    = broker_cfg.get("asset_group", ""),
            symbols        = symbols,
        )

        # Register fill callback for logging
        def _on_fill(fill_event):
            self.trade_logger.log_fill(
                symbol     = fill_event.symbol,
                side       = fill_event.side,
                qty        = fill_event.qty,
                fill_price = fill_event.fill_price,
                order_id   = fill_event.order_id or "",
            )
            self._trade_count += 1
            self.dashboard.update(event=(
                f"Fill: {fill_event.side.upper()} "
                f"{fill_event.qty:.0f} {fill_event.symbol} "
                f"@ ${fill_event.fill_price:.2f}"
            ))
            try:
                from telegram.notifier import notify as _tg_notify
                _tg_notify("trade", {
                    "symbol":      fill_event.symbol,
                    "side":        fill_event.side,
                    "asset_group": self.config.get("broker", {}).get("asset_group", "—"),
                    "regime":      self._last_regime or "?",
                })
            except Exception:
                pass

        self.position_tracker.register_fill_callback(_on_fill)

        # ── System online ─────────────────────────────────────────────────────
        self.trade_logger.info("System online", symbols=symbols, timeframe=timeframe)
        self._running = True
        self._session_start = dt.datetime.now(dt.timezone.utc)

        _print(
            f"\n[bold green]System online[/bold green]  "
            f"symbols={symbols}  timeframe={timeframe}",
            console,
        )

    # ======================================================================= #
    # Main loop                                                                #
    # ======================================================================= #

    def run_loop(self) -> None:
        """
        Main trading loop.  Runs until _running is set to False by shutdown().
        Blocks the calling thread.
        """
        broker_cfg = self.config.get("broker", {})
        hmm_cfg    = self.config.get("hmm",    {})
        symbols    = broker_cfg.get("symbols",   ["SPY"])
        timeframe  = broker_cfg.get("timeframe", "1Day")

        from broker.order_executor import OrderExecutor
        executor = OrderExecutor(
            client           = self.client,
            risk_manager     = self.risk_manager,
            limit_offset_pct = 0.001,
            cancel_after_sec = 30,
            retry_at_market  = True,
        )

        _print(
            f"\nEntering main loop  timeframe={timeframe}  "
            f"{'DRY RUN -- no orders will be placed' if self.dry_run else 'Orders ENABLED'}",
            self.console,
        )

        while self._running:
            # ---- sleep until next bar close ---------------------------------
            next_close = _next_bar_close_utc(timeframe)
            wait_secs  = _seconds_until(next_close)
            self.dashboard.update(
                next_bar_dt    = next_close,
                next_broker_dt = next_close,
                next_hmm_dt    = self._last_retrain + dt.timedelta(days=7),
                timeframe      = timeframe,
            )

            if wait_secs > 10:
                logger.debug(
                    "Sleeping %.0fs until next bar close (%s UTC)",
                    wait_secs,
                    next_close.strftime("%H:%M:%S"),
                )
                # Sleep in short chunks so SIGINT is responsive
                slept = 0.0
                while slept < wait_secs and self._running:
                    time.sleep(min(5.0, wait_secs - slept))
                    slept += 5.0

            if not self._running:
                break

            # ---- 1. Fetch latest bars ----------------------------------------
            try:
                bars_by_symbol = _fetch_live_bars(
                    self.client, symbols, timeframe, n_bars=300
                )
            except Exception as exc:
                logger.error("Data fetch failed: %s -- pausing signals", exc)
                self.trade_logger.log_error(exc, context="bar_fetch")
                self.dashboard.update(event=f"[red]Data feed error: {exc}[/red]")
                time.sleep(60)
                continue

            if not bars_by_symbol:
                logger.warning("No bars returned -- skipping bar")
                time.sleep(30)
                continue

            # ---- 2. Compute features (using benchmark symbol) ----------------
            _proxy_loop = hmm_cfg.get("regime_proxy") or None
            ref_sym = (_proxy_loop if _proxy_loop and _proxy_loop in symbols else None) or symbols[0]
            ref_bars = bars_by_symbol.get(ref_sym)
            if ref_bars is None or len(ref_bars) < 60:
                logger.warning("Insufficient bars for %s -- skipping", ref_sym)
                continue

            try:
                from data.feature_engineering import FeatureEngineer
                fe = FeatureEngineer()
                _vix = _maybe_fetch_vix(hmm_cfg, ref_bars.index)
                _credit = _maybe_fetch_credit(hmm_cfg, ref_bars.index)
                features_clean = fe.build_feature_matrix(
                    ref_bars, feature_names=_hmm_feature_names(hmm_cfg),
                    vix_series=_vix, credit_series=_credit,
                )
                # Blend log_ret_1 / realized_vol_20 across equity-like symbols
                _lp_bars = {ref_sym: ref_bars}
                for _s in symbols:
                    if _s == ref_sym:
                        continue
                    _sb = bars_by_symbol.get(_s)
                    if _sb is not None:
                        _lp_bars[_s] = _sb
                features_clean = _blend_cross_symbol_features(
                    features_clean,
                    _lp_bars,
                    feature_engineer=fe,
                    blend_exclude=(
                        [s for s in symbols if s != ref_sym]
                        if _proxy_loop else
                        hmm_cfg.get("blend_exclude", [])
                    ),
                    min_bars=10,
                )
                if len(features_clean) < 10:
                    logger.warning("Too few clean feature rows -- skipping bar")
                    continue
                feature_matrix = features_clean.values
            except Exception as exc:
                logger.error("Feature computation failed: %s", exc)
                self.trade_logger.log_error(exc, context="feature_engineering")
                continue

            # ---- 3. HMM incremental update -----------------------------------
            try:
                from core.hmm_engine import RegimeState
                timestamp = pd.Timestamp(features_clean.index[-1])
                regime_state: RegimeState = self.hmm_engine.update(
                    new_feature_row = feature_matrix[-1],
                    timestamp       = timestamp,
                )
                _prev_regime = self._last_regime
                self._last_regime = regime_state.label
                if _prev_regime and _prev_regime != self._last_regime:
                    try:
                        from telegram.notifier import notify as _tg_notify
                        _equity = float(self.position_tracker.get_equity() or 0)
                        _tg_notify("regime_change", {
                            "from_regime": _prev_regime,
                            "to_regime":   self._last_regime,
                            "asset_group": broker_cfg.get("asset_group", "—"),
                            "equity":      _equity or None,
                        })
                    except Exception:
                        pass
            except Exception as exc:
                logger.error(
                    "HMM update failed: %s -- holding regime=%s",
                    exc, self._last_regime,
                )
                self.trade_logger.log_error(exc, context="hmm_update")
                # Hold current regime; synthesise minimal state
                from dataclasses import dataclass
                regime_state = _make_fallback_regime(self._last_regime)

            # ---- 4. Stability / flicker checks --------------------------------
            is_flickering = self.hmm_engine.is_flickering()
            is_stable     = regime_state.is_confirmed and not is_flickering
            confidence    = regime_state.probability
            min_conf      = hmm_cfg.get("min_confidence", 0.55)
            uncertainty   = is_flickering or confidence < min_conf

            logger.info(
                "Bar %d | regime=%s | p=%.3f | stable=%s | flicker=%s",
                self._bar_count, regime_state.label, confidence,
                regime_state.is_confirmed, is_flickering,
            )

            # ---- 5. Build per-symbol price DataFrames for strategy -----------
            # Retain price history for risk-manager correlation check
            for sym, sym_bars in bars_by_symbol.items():
                if "close" in sym_bars.columns:
                    self._price_history[sym] = sym_bars["close"].tail(60)

            # ---- 6. Generate signals -----------------------------------------
            if self._multi_strat_mode:
                # Multi-strategy: per-strategy signal generation
                _raw_by_strat: Dict[str, list] = {}
                for _sname, _orch in self._strat_orchestrators.items():
                    _proxy = self.strategy_registry.get(_sname)
                    if _proxy is not None and not getattr(_proxy, "is_enabled", True):
                        continue
                    _syms = self._strat_symbols.get(_sname, symbols)
                    _sbars = {s: bars_by_symbol[s] for s in _syms if s in bars_by_symbol}
                    if not _sbars:
                        continue
                    try:
                        _orch.update_weights(self.position_tracker.get_current_weights())
                        _raw_by_strat[_sname] = _orch.generate_signals(
                            symbols       = list(_sbars.keys()),
                            bars          = _sbars,
                            regime_state  = regime_state,
                            is_flickering = is_flickering,
                        )
                    except Exception as exc:
                        logger.error("Strategy '%s' signal gen failed: %s", _sname, exc)
                        _raw_by_strat[_sname] = []
                raw_signals = []   # unused in multi-strat path below
            else:
                # Single-strategy (existing behavior)
                try:
                    self.strategy.update_weights(self.position_tracker.get_current_weights())
                    raw_signals = self.strategy.generate_signals(
                        symbols       = symbols,
                        bars          = bars_by_symbol,
                        regime_state  = regime_state,
                        is_flickering = is_flickering,
                    )
                except Exception as exc:
                    logger.error("Strategy signal generation failed: %s", exc)
                    self.trade_logger.log_error(exc, context="signal_generation")
                    raw_signals = []

            # ---- 7. Risk gate + order submission ----------------------------
            _port_snap_now = self.position_tracker.get_last_snapshot()
            positions_dict = {
                p.symbol: p.market_value
                for p in (_port_snap_now.positions if _port_snap_now else [])
            }

            from core.risk_manager import PortfolioState, TradingState

            def _submit_one(final_signal, strategy_label: str = "") -> None:
                """Log + optionally submit one approved signal."""
                self.trade_logger.log_trade(
                    symbol = final_signal.symbol,
                    side   = "buy" if final_signal.is_long else "sell",
                    qty    = 0,
                    price  = final_signal.entry_price,
                    extra  = {
                        "regime":   regime_state.label,
                        "dry_run":  self.dry_run,
                        "size_pct": round(final_signal.position_size_pct, 4),
                        "strategy": strategy_label,
                    },
                )
                if self.dry_run:
                    _print(
                        f"  [DRY RUN] Would submit: {final_signal.symbol}  "
                        f"size={final_signal.position_size_pct:.1%}  "
                        f"stop={final_signal.stop_loss:.2f}"
                        + (f"  [{strategy_label}]" if strategy_label else ""),
                        self.console, style="dim",
                    )
                    return
                _res = None
                for _attempt in range(1, 4):
                    try:
                        _res = executor.submit_order(final_signal)
                        break
                    except Exception as _exc:
                        logger.warning(
                            "Order submission attempt %d/3 failed for %s: %s",
                            _attempt, final_signal.symbol, _exc,
                        )
                        if _attempt < 3:
                            time.sleep(2 ** _attempt)
                if _res is None:
                    self.alert_manager.alert_order_error(
                        final_signal.symbol, "All 3 submission attempts failed"
                    )
                    return
                from broker.order_executor import OrderStatus
                if _res.status == OrderStatus.REJECTED:
                    self.alert_manager.alert_order_error(
                        final_signal.symbol, _res.error_message or "REJECTED by broker"
                    )
                    self.dashboard.update(
                        event=f"Order REJECTED {final_signal.symbol}: {_res.error_message}"
                    )
                else:
                    self.dashboard.update(
                        event=(
                            f"Order submitted: BUY {final_signal.symbol}  "
                            f"regime={regime_state.label}"
                            + (f"  [{strategy_label}]" if strategy_label else "")
                        )
                    )

            if self._multi_strat_mode:
                # Multi-strategy risk gate: per-strategy RM → PRM → executor
                _total_equity = self.risk_manager._current_equity
                for _sname, _sigs in _raw_by_strat.items():
                    _srm = self._strat_risk_managers[_sname]
                    _proxy = self.strategy_registry.get(_sname)
                    _strat_eq = getattr(_proxy, "allocated_capital", _total_equity)
                    _strat_ps = PortfolioState(
                        equity         = _strat_eq,
                        cash           = self.client.get_cash(),
                        buying_power   = self.client.get_buying_power(),
                        positions      = positions_dict,
                        current_regime = self._last_regime,
                        price_history  = self._price_history,
                    )
                    for signal in _sigs:
                        if not signal.is_long:
                            continue
                        _dec = _srm.validate_signal(signal, _strat_ps)
                        if not _dec.approved:
                            logger.info(
                                "[%s] Signal REJECTED: %s  reason=%s",
                                _sname, signal.symbol, _dec.rejection_reason,
                            )
                            self.dashboard.update(
                                event=f"[{_sname}] REJECTED {signal.symbol}: {_dec.rejection_reason}"
                            )
                            continue
                        _approved = _dec.modified_signal

                        # Portfolio-level risk gate
                        if self.portfolio_rm is not None:
                            _port_ps = PortfolioState(
                                equity         = _total_equity,
                                cash           = self.client.get_cash(),
                                buying_power   = self.client.get_buying_power(),
                                positions      = positions_dict,
                                current_regime = self._last_regime,
                                price_history  = self._price_history,
                            )
                            _pdec = self.portfolio_rm.validate_signal(
                                _approved, _sname, _port_ps
                            )
                            if not _pdec.approved:
                                logger.info(
                                    "[PRM] Signal REJECTED: %s  reason=%s",
                                    _approved.symbol, _pdec.rejection_reason,
                                )
                                self.dashboard.update(
                                    event=f"[PRM] REJECTED {_approved.symbol}: {_pdec.rejection_reason}"
                                )
                                continue
                            _approved = _pdec.modified_signal

                        _submit_one(_approved, strategy_label=_sname)

            else:
                # Single-strategy risk gate (original behavior)
                portfolio_state = PortfolioState(
                    equity          = self.risk_manager._current_equity,
                    cash            = self.client.get_cash(),
                    buying_power    = self.client.get_buying_power(),
                    positions       = positions_dict,
                    current_regime  = self._last_regime,
                    price_history   = self._price_history,
                )

                for signal in raw_signals:
                    if not signal.is_long:
                        continue

                    decision = self.risk_manager.validate_signal(
                        signal          = signal,
                        portfolio_state = portfolio_state,
                    )

                    if not decision.approved:
                        logger.info(
                            "Signal REJECTED: %s  reason=%s",
                            signal.symbol, decision.rejection_reason,
                        )
                        self.trade_logger.log_risk_event(
                            event_type  = "rejected",
                            description = f"{signal.symbol}: {decision.rejection_reason}",
                            severity    = "WARNING",
                        )
                        self.dashboard.update(
                            event=f"REJECTED {signal.symbol}: {decision.rejection_reason}"
                        )
                        continue

                    final_signal = decision.modified_signal
                    if decision.modifications:
                        logger.info(
                            "Signal MODIFIED: %s  changes=%s",
                            signal.symbol, decision.modifications,
                        )

                    _submit_one(final_signal)

            # ---- 8. Update trailing stops ------------------------------------
            if not self.dry_run:
                self._update_trailing_stops(executor, regime_state)

            # ---- 9. Circuit breaker check ------------------------------------
            trading_state = self.risk_manager.get_trading_state()
            if trading_state == TradingState.HALTED:
                dd = self.risk_manager.get_drawdown_state()
                self.alert_manager.alert_drawdown_halt(
                    equity       = dd.current_equity,
                    drawdown_pct = dd.dd_from_peak,
                )
                self.trade_logger.log_risk_event(
                    event_type  = "HALT",
                    description = f"Trading halted  dd={dd.dd_from_peak:.2%}",
                    severity    = "CRITICAL",
                )
                self.dashboard.update(
                    event=f"[red]HALT: peak DD {dd.dd_from_peak:.2%}[/red]"
                )

            # ---- 10. Multi-strategy health checks + allocator rebalance ---------
            if self._multi_strat_mode and self.strategy_registry is not None:
                _pre_enabled = {
                    n: getattr(s, "is_enabled", True)
                    for n, s in self.strategy_registry.all().items()
                }
                self.strategy_registry.run_health_checks()
                for _sn, _was in _pre_enabled.items():
                    _s = self.strategy_registry.get(_sn)
                    if _was and _s is not None and not getattr(_s, "is_enabled", True):
                        _reason = "health check failed"
                        _hc = getattr(_s, "_last_health", None)
                        if _hc is not None:
                            _reason = getattr(_hc, "reason_if_unhealthy", _reason) or _reason
                        self.alert_manager.alert_strategy_disabled(_sn, _reason)
                        self.dashboard.update(event=f"[{_sn}] auto-disabled: {_reason}")

                # Correlation cluster detection (pair > 0.80)
                try:
                    _hist = {}
                    for _sn, _syms in self._strat_symbols.items():
                        _p = self.strategy_registry.get(_sn)
                        if _p is None or not getattr(_p, "is_enabled", True):
                            continue
                        _series = [
                            self._price_history[_s].pct_change().dropna()
                            for _s in _syms if _s in self._price_history
                        ]
                        if _series:
                            _df = pd.concat(_series, axis=1).dropna()
                            if len(_df) >= 20:
                                _hist[_sn] = _df.mean(axis=1)
                    if len(_hist) >= 2:
                        _corr_df = pd.DataFrame(_hist).dropna().corr()
                        _hot_pairs = []
                        _names = list(_corr_df.columns)
                        for _i in range(len(_names)):
                            for _j in range(_i + 1, len(_names)):
                                _c = float(_corr_df.iloc[_i, _j])
                                if _c > 0.80:
                                    _hot_pairs.append((_names[_i], _names[_j], _c))
                        if _hot_pairs:
                            self.alert_manager.alert_correlation_cluster(_hot_pairs)
                except Exception as _exc:
                    logger.debug("Correlation cluster check skipped: %s", _exc)

                self._bars_since_alloc += 1
                if self._bars_since_alloc >= self._alloc_interval_bars:
                    self._bars_since_alloc = 0
                    try:
                        _cur_equity = self.risk_manager._current_equity
                        self.capital_allocator.total_capital = _cur_equity
                        _dd_state = self.risk_manager.get_drawdown_state()
                        _daily_dd = max(0.0, getattr(_dd_state, "dd_today", 0.0))
                        _prev_weights = dict(self._alloc_weights or {})
                        self._alloc_weights = self.capital_allocator.allocate(
                            self.strategy_registry,
                            daily_drawdown=_daily_dd,
                        )
                        for _sn, _w in self._alloc_weights.items():
                            _p = self.strategy_registry.get(_sn)
                            if _p is not None:
                                _p.allocated_capital = _w * _cur_equity
                        logger.info(
                            "Allocator rebalanced: %s",
                            {k: f"{v:.2f}" for k, v in self._alloc_weights.items()},
                        )
                        _changed = any(
                            abs(self._alloc_weights.get(_k, 0.0) - _prev_weights.get(_k, 0.0)) > 0.01
                            for _k in set(self._alloc_weights) | set(_prev_weights)
                        )
                        if _changed:
                            self.alert_manager.alert_allocator_rebalance(
                                self._alloc_weights, trigger="scheduled"
                            )
                    except Exception as exc:
                        logger.error("Allocator rebalance failed: %s", exc)

                # Portfolio-level DD breaker (fires if lock file was written this bar)
                if self.portfolio_rm is not None:
                    from core.risk_manager import LOCK_FILE as _LOCK
                    if _LOCK.exists() and not getattr(self, "_prm_dd_alerted", False):
                        _risk_cfg = self.config.get("risk", {})
                        self.alert_manager.alert_portfolio_dd_breaker(
                            equity        = self.risk_manager._current_equity,
                            drawdown_pct  = abs(self.risk_manager.get_drawdown_state().dd_from_peak),
                            threshold_pct = _risk_cfg.get("max_dd_from_peak", 0.10),
                        )
                        self._prm_dd_alerted = True

            # ---- 11. Dashboard refresh & position update --------------------
            try:
                port_snap = self.position_tracker.update(self._last_regime)  # REST refresh

                # Update logger context
                self.trade_logger.set_context(
                    regime      = self._last_regime,
                    probability = confidence,
                    equity      = port_snap.total_equity,
                    positions   = [p.symbol for p in port_snap.positions],
                )

                from core.signal_generator import PortfolioSignal
                from monitoring.dashboard import SystemStatus
                portfolio_signal = PortfolioSignal(
                    timestamp       = pd.Timestamp.now(),
                    regime          = regime_state.label,
                    confidence      = confidence,
                    is_stable       = is_stable,
                    target_weights  = {},
                    delta_weights   = {},
                    leverage        = 1.0,
                    trading_allowed = trading_state != TradingState.HALTED,
                )
                _sys_status = SystemStatus(
                    data_ok          = True,
                    api_ok           = True,
                    api_latency_ms   = 0.0,
                    hmm_last_trained = self.hmm_engine._training_date.to_pydatetime()
                                       if getattr(self.hmm_engine, "_training_date", None) is not None else None,
                    mode             = "PAPER" if self.config.get("broker", {}).get("paper_trading", True) else "LIVE",
                )
                try:
                    _clock_open = self.client.get_clock().is_open
                except Exception:
                    _clock_open = None
                _next_bar = _next_bar_close_utc(timeframe)

                # Build alloc_info for dashboard multi-strat panel
                _alloc_info: Optional[dict] = None
                if self._multi_strat_mode and self.strategy_registry is not None:
                    _alloc_info = {}
                    for _sn, _w in self._alloc_weights.items():
                        _p = self.strategy_registry.get(_sn)
                        _alloc_info[_sn] = {
                            "weight":  _w,
                            "sharpe":  getattr(_p, "_sharpe", 0.0),
                            "healthy": getattr(_p, "is_enabled", True),
                        }

                self.dashboard.update(
                    snapshot       = port_snap,
                    signal         = portfolio_signal,
                    drawdown_state = self.risk_manager.get_drawdown_state(),
                    system_status  = _sys_status,
                    market_open    = _clock_open,
                    next_bar_dt    = _next_bar,
                    next_broker_dt = _next_bar,
                    next_hmm_dt    = self._last_retrain + dt.timedelta(days=7),
                    timeframe      = timeframe,
                    alloc_info     = _alloc_info,
                    event          = (
                        f"Bar {self._bar_count}: {regime_state.label}  "
                        f"p={confidence:.1%}  "
                        f"{'STABLE' if is_stable else 'PENDING'}"
                    ),
                )
            except Exception as exc:
                logger.warning("Dashboard update failed: %s", exc)

            # ---- 12. Weekly HMM retrain check --------------------------------
            days_since = (dt.datetime.now(dt.timezone.utc) - self._last_retrain).days
            if days_since >= 7:
                _print("Weekly HMM retrain ...", self.console, style="dim")
                try:
                    hmm_cfg = self.config.get("hmm", {})
                    self.hmm_engine = _train_hmm(
                        self.client, symbols, hmm_cfg, console=self.console
                    )
                    self.strategy = _rebuild_strategy(self.config, self.hmm_engine)
                    self._last_retrain = dt.datetime.now(dt.timezone.utc)
                    self.trade_logger.info(
                        "Weekly HMM retrain complete",
                        n_states=self.hmm_engine._n_states,
                    )
                    self.alert_manager.alert_regime_change(
                        previous_regime = self._last_regime,
                        new_regime      = self._last_regime,
                        confidence      = confidence,
                    )
                except Exception as exc:
                    logger.error("Weekly retrain failed: %s -- keeping current model", exc)
                    self.trade_logger.log_error(exc, context="weekly_retrain")

            self._bar_count += 1

    # ======================================================================= #
    # Shutdown                                                                 #
    # ======================================================================= #

    def shutdown(self) -> None:
        """
        Graceful shutdown.
        Stops WebSocket feeds, saves state snapshot, prints session summary.
        Positions are NOT closed -- stops remain in place.
        """
        self._running = False
        _print("\n\nShutting down ...", self.console)

        # Stop streams
        try:
            if self.position_tracker:
                self.position_tracker.stop_stream()
        except Exception:
            pass

        try:
            if self.dashboard:
                self.dashboard.stop()
        except Exception:
            pass

        # Save state snapshot
        try:
            equity_now = (
                self.client.get_portfolio_value() if self.client else self._equity_at_start
            )
        except Exception:
            equity_now = self._equity_at_start

        _save_snapshot({
            "session_start":    self._session_start.isoformat(),
            "session_end":      dt.datetime.now(dt.timezone.utc).isoformat(),
            "regime":           self._last_regime,
            "bar_count":        self._bar_count,
            "trade_count":      self._trade_count,
            "equity_at_start":  self._equity_at_start,
            "equity_at_end":    equity_now,
            "model_path":       str(_MODEL_PATH),
        })

        # Disconnect broker
        try:
            if self.client:
                self.client.disconnect()
        except Exception:
            pass

        # Print summary
        _print_session_summary(
            session_start = self._session_start,
            equity_start  = self._equity_at_start,
            equity_end    = equity_now,
            bar_count     = self._bar_count,
            trade_count   = self._trade_count,
            console       = self.console,
        )

        if self.trade_logger:
            self.trade_logger.info(
                "System offline",
                bars=self._bar_count,
                trades=self._trade_count,
            )

    # ======================================================================= #
    # Helpers                                                                  #
    # ======================================================================= #

    def _update_trailing_stops(self, executor, regime_state) -> None:
        """
        Tighten stops on existing positions based on current regime.
        In high-vol regimes stops are tightened more aggressively.
        """
        try:
            snap = self.position_tracker.get_last_snapshot()
            if not snap:
                return
            for pos in snap.positions:
                if pos.stop_level is None:
                    continue
                # ATR-based trail: tighten stop if price moved favorably
                current_price = pos.current_price
                current_stop  = pos.stop_level
                # Only tighten -- never widen
                if current_price > pos.avg_entry_price:
                    # Trail at 2× ATR below current price (approximate)
                    trail_pct = 0.03 if "BULL" in regime_state.label else 0.05
                    new_stop = current_price * (1.0 - trail_pct)
                    if new_stop > current_stop:
                        executor.modify_stop(pos.symbol, new_stop)
        except Exception as exc:
            logger.debug("Trailing stop update: %s", exc)


def _make_fallback_regime(label: str):
    """Return a minimal RegimeState with the given label when HMM update fails."""
    from core.hmm_engine import RegimeState
    return RegimeState(
        label             = label,
        state_id          = 0,
        probability       = 0.5,
        state_probabilities = np.array([0.5]),
        timestamp         = None,
        is_confirmed      = False,
        consecutive_bars  = 0,
    )


def _rebuild_strategy(config: Dict, hmm_engine) -> "StrategyOrchestrator":
    from core.regime_strategies import StrategyOrchestrator
    hmm_cfg = config.get("hmm", {})
    return StrategyOrchestrator(
        config           = config,
        regime_infos     = hmm_engine.get_all_regime_info(),
        min_confidence   = hmm_cfg.get("min_confidence", 0.55),
        rebalance_threshold = config.get("strategy", {}).get("rebalance_threshold", 0.10),
    )


# ── Run modes ──────────────────────────────────────────────────────────────────

def _run_multi_strat_backtest(
    config: Dict,
    args: argparse.Namespace,
    prices: "pd.DataFrame",
    console,
    output_dir: Path,
    initial_capital: float,
    train_window: int,
    test_window: int,
    step_size: int,
    slippage_pct: float,
    risk_free_rate: float,
) -> None:
    """Execute MultiStrategyBacktester using settings.yaml[strategies] spec."""
    from backtest import MultiStrategyBacktester, StrategySpec

    strat_cfgs = config.get("strategies", {}) or {}
    enabled = {
        name: cfg for name, cfg in strat_cfgs.items()
        if cfg.get("enabled", True)
    }
    if len(enabled) < 2:
        _print(
            "[red]--multi-strat requires at least 2 enabled strategies in "
            "settings.yaml[strategies].[/red]",
            console,
        )
        sys.exit(1)

    hmm_cfg_raw = config.get("hmm", {})
    hmm_base = {
        "n_candidates":      hmm_cfg_raw.get("n_candidates", [3, 4, 5]),
        "n_init":            hmm_cfg_raw.get("n_init", 10),
        "min_train_bars":    hmm_cfg_raw.get("min_train_bars", 120),
        "stability_bars":    hmm_cfg_raw.get("stability_bars", 3),
        "flicker_window":    hmm_cfg_raw.get("flicker_window", 20),
        "flicker_threshold": hmm_cfg_raw.get("flicker_threshold", 4),
        "min_confidence":    hmm_cfg_raw.get("min_confidence", 0.55),
        "extended_features": hmm_cfg_raw.get("extended_features", True),
        "blend_exclude":     hmm_cfg_raw.get("blend_exclude", []),
    }
    strat_base = {"strategy": config.get("strategy", {})}

    specs = []
    for sname, scfg in enabled.items():
        syms = scfg.get("symbols")
        if not syms:
            _print(f"[yellow]Strategy '{sname}' has no symbols — skipping.[/yellow]", console)
            continue
        # Validate all symbols exist in price data
        missing = [s for s in syms if s not in prices.columns]
        if missing:
            _print(
                f"[yellow]Strategy '{sname}' symbols {missing} not in price data — skipping.[/yellow]",
                console,
            )
            continue
        specs.append(StrategySpec(
            name            = sname,
            symbols         = syms,
            hmm_config      = dict(hmm_base),
            strategy_config = dict(strat_base),
            weight_min      = float(scfg.get("weight_min", 0.0)),
            weight_max      = float(scfg.get("weight_max", 1.0)),
        ))

    if len(specs) < 2:
        _print("[red]Fewer than 2 valid strategy specs — cannot run multi-strat backtest.[/red]", console)
        sys.exit(1)

    allocator_approach = getattr(args, "allocator_approach", "inverse_vol")
    _print(
        f"\n[bold cyan]Multi-Strategy Walk-Forward Backtest[/bold cyan]\n"
        f"  Strategies  : {[s.name for s in specs]}\n"
        f"  Capital     : ${initial_capital:,.0f}\n"
        f"  Allocator   : {allocator_approach}",
        console,
    )

    mbt = MultiStrategyBacktester(
        initial_capital = initial_capital,
        train_window    = train_window,
        test_window     = test_window,
        step_size       = step_size,
        slippage_pct    = slippage_pct,
        risk_free_rate  = risk_free_rate,
        allocator_approach = allocator_approach,
    )

    logging.getLogger("hmmlearn").setLevel(logging.ERROR)
    logging.getLogger("core.hmm_engine").setLevel(logging.WARNING)

    # Scale aggregate-exposure cap with strategy count. Default PRM cap is 0.80,
    # designed for single-strategy use; with N fully-invested sub-portfolios the
    # aggregate approaches 100% and every new signal gets vetoed. Cap at 1.5
    # (50% leverage headroom) and scale with N so 2 strategies → ~1.0, 5 → 1.5.
    prm_cap = min(1.5, 0.5 + 0.25 * len(specs))
    try:
        result = mbt.run(prices, specs,
                         portfolio_rm_config={"max_aggregate_exposure": prm_cap})
    except Exception as exc:
        _print(f"[red]Multi-strat backtest failed:[/red] {exc}", console)
        sys.exit(1)

    _print(
        f"\n  Folds completed : {result.metadata.get('n_folds', '?')}\n"
        f"  Final equity    : ${result.final_equity:,.2f}",
        console, style="dim",
    )

    # Per-strategy summary
    for sname, attr in result.attributions.items():
        _print(
            f"  [{sname}]  return={attr.total_return:+.2%}  "
            f"sharpe={attr.sharpe:.3f}  maxdd={attr.max_drawdown:.2%}",
            console, style="dim",
        )

    mbt.save_csv_outputs(result, output_dir)
    _print(f"\n[green]Multi-strat backtest complete.[/green]  CSVs saved to {output_dir}/", console)


def run_backtest(config: Dict, args: argparse.Namespace) -> None:
    """
    Execute the full walk-forward backtest pipeline.
    """
    from backtest.backtester import WalkForwardBacktester
    from backtest.performance import PerformanceAnalyzer
    from backtest.stress_test import StressTester

    console = _console()

    symbols: List[str] = _resolve_symbols(
        config,
        asset_group=getattr(args, "asset_group", None),
        symbols_arg=getattr(args, "symbols", None),
    )

    bt_cfg = config.get("backtest", {})
    start_date: str = args.start or "2020-01-01"
    end_date: str   = args.end   or pd.Timestamp.today().strftime("%Y-%m-%d")
    output_dir      = _saved_results_backtest_dir()
    # keep legacy results/ dir in sync for tools that read from it
    Path(args.output).mkdir(parents=True, exist_ok=True)

    initial_capital = float(bt_cfg.get("initial_capital", 100_000))
    slippage_pct    = float(bt_cfg.get("slippage_pct",    0.0005))
    train_window    = int(bt_cfg.get("train_window",      252))
    test_window     = int(bt_cfg.get("test_window",       126))
    step_size       = int(bt_cfg.get("step_size",         126))
    risk_free_rate  = float(bt_cfg.get("risk_free_rate",  0.045))

    # Header — plain-English context about the run
    _years = (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days / 365.25
    _train_mo = round(train_window / 21)
    _test_mo  = round(test_window  / 21)
    _sym_preview = ", ".join(symbols[:6])
    _sym_extra   = f" … (+{len(symbols) - 6} more)" if len(symbols) > 6 else ""

    _print(f"\n[bold cyan]Regime Trader — Walk-Forward Backtest[/bold cyan]", console)
    _print(
        f"  Symbols  : {_sym_preview}{_sym_extra}\n"
        f"  Period   : {start_date}  →  {end_date}  ({_years:.1f} years)\n"
        f"  Capital  : ${initial_capital:,.0f}\n"
        f"  Method   : Walk-forward — train {_train_mo}m in-sample, "
        f"test next {_test_mo}m out-of-sample, step forward, repeat\n"
        f"  Note     : each fold's model is never trained on its own test data",
        console,
    )

    api_key    = os.environ.get("ALPACA_API_KEY",    "")
    secret_key = os.environ.get("ALPACA_SECRET_KEY", "")
    if not api_key or not secret_key:
        _print(
            "[red]ERROR: Alpaca credentials not found.[/red]\n"
            "Set ALPACA_API_KEY / ALPACA_SECRET_KEY or populate config/credentials.yaml.",
            console,
        )
        sys.exit(1)

    # In multi-strat mode, strategy configs may declare symbols not in the main
    # asset group (e.g. XLU/XLP for mean_reversion). Merge them in before fetch.
    if getattr(args, "multi_strat", False):
        strat_syms = [
            s
            for cfg in (config.get("strategies") or {}).values()
            if cfg.get("enabled", True)
            for s in (cfg.get("symbols") or [])
        ]
        symbols = list(dict.fromkeys(symbols + strat_syms))  # dedup, preserve order

    _print("\nFetching historical data from Alpaca ...", console, style="dim")
    try:
        prices = _fetch_prices(symbols, start_date, end_date, api_key, secret_key)
    except Exception as exc:
        _print(f"[red]Data fetch failed:[/red] {exc}", console)
        sys.exit(1)

    _print(
        f"  Downloaded {len(prices)} bars  "
        f"({prices.index[0].date()} -> {prices.index[-1].date()})  "
        f"symbols: {list(prices.columns)}",
        console, style="dim",
    )

    if len(prices) < train_window + test_window + 50:
        _print(
            f"[red]Insufficient data:[/red] only {len(prices)} bars. "
            f"Need >= {train_window + test_window + 50}.",
            console,
        )
        sys.exit(1)

    # ── Multi-strategy backtest branch ────────────────────────────────────────
    if getattr(args, "multi_strat", False):
        _run_multi_strat_backtest(config, args, prices, console, output_dir,
                                  initial_capital, train_window, test_window,
                                  step_size, slippage_pct, risk_free_rate)
        return

    hmm_cfg_raw = config.get("hmm", {})
    hmm_config = {
        "n_candidates":       hmm_cfg_raw.get("n_candidates", [3, 4, 5]),
        "n_init":             hmm_cfg_raw.get("n_init", 10),
        "min_train_bars":     hmm_cfg_raw.get("min_train_bars", 120),
        "stability_bars":     hmm_cfg_raw.get("stability_bars", 3),
        "flicker_window":     hmm_cfg_raw.get("flicker_window", 20),
        "flicker_threshold":  hmm_cfg_raw.get("flicker_threshold", 4),
        "min_confidence":     hmm_cfg_raw.get("min_confidence", 0.55),
        # Feature selection (was silently dropped before — the backtester
        # was always using HMM_EXTENDED_FEATURES regardless of settings.yaml).
        "extended_features":  hmm_cfg_raw.get("extended_features", True),
        "features_override":  hmm_cfg_raw.get("features_override"),
        "use_vix_features":   hmm_cfg_raw.get("use_vix_features", False),
        "use_credit_spread_features": hmm_cfg_raw.get("use_credit_spread_features", False),
        "blend_exclude":      hmm_cfg_raw.get("blend_exclude", []),
    }
    strategy_config = {"strategy": config.get("strategy", {})}
    _strat = strategy_config["strategy"]

    _print(
        f"\n  [dim]HMM[/dim]\n"
        f"  [dim]  States (candidates) : {hmm_config['n_candidates']}[/dim]\n"
        f"  [dim]  Features            : {hmm_cfg_raw.get('features', ['log_return', 'realized_vol_5d', 'realized_vol_21d'])}[/dim]\n"
        f"  [dim]  Min confidence      : {hmm_config['min_confidence']}[/dim]\n"
        f"  [dim]  Stability bars      : {hmm_config['stability_bars']}  "
        f"Flicker window: {hmm_config['flicker_window']}  "
        f"Flicker threshold: {hmm_config['flicker_threshold']}[/dim]\n"
        f"\n  [dim]Strategy[/dim]\n"
        f"  [dim]  Allocation — low-vol: {_strat.get('low_vol_allocation', 0.95):.0%}  "
        f"mid-vol (trend): {_strat.get('mid_vol_allocation_trend', 0.95):.0%}  "
        f"mid-vol (no trend): {_strat.get('mid_vol_allocation_no_trend', 0.70):.0%}  "
        f"high-vol: {_strat.get('high_vol_allocation', 0.70):.0%}[/dim]\n"
        f"  [dim]  Leverage cap        : {_strat.get('low_vol_leverage', 1.25):.2f}x  "
        f"Rebalance threshold: {_strat.get('rebalance_threshold', 0.15):.0%}  "
        f"Trend lookback: {_strat.get('trend_lookback', 50)} bars[/dim]",
        console,
    )

    def _make_bt_progress(n_syms: int) -> object:
        """Return a fold-progress callback for bt.run()."""
        from rich.text import Text

        _W = 20
        _ic = initial_capital  # capture for equity colour comparison

        def _build_line(fold_id: int, n_total: int, phase: str, info: dict) -> Text:
            if phase == "training":
                pct = int(fold_id / n_total * 100)
            else:
                pct = int((fold_id + 1) / n_total * 100)

            filled = pct * _W // 100
            bar_text = Text()
            bar_text.append("#" * filled, style="cyan")
            bar_text.append("-" * (_W - filled), style="dim white")

            line = Text()
            line.append("  [")
            line.append_text(bar_text)
            line.append("] ")
            line.append(f"{pct:3d}%", style="bold white")
            line.append(f"  Fold {fold_id+1}/{n_total}  ")

            if phase == "training":
                line.append("Training ...", style="yellow")
                line.append(
                    f"  OOS {info['oos_start']} -> {info['oos_end']}", style="dim"
                )
            else:
                line.append("OK ", style="bold green")
                line.append(
                    f"{info['oos_start']} -> {info['oos_end']}", style="green"
                )
                line.append(f"  {info['n_states']} states", style="dim cyan")
                line.append(f"  {info['fold_trades']} tr", style="dim")
                eq_style = (
                    "bold green" if info["equity"] >= _ic else "bold red"
                )
                line.append(f"  ${info['equity']:>10,.0f}", style=eq_style)

            return line

        def _cb(fold_id: int, n_total: int, phase: str, info: dict) -> None:
            if console is None:
                return
            line = _build_line(fold_id, n_total, phase, info)
            end = "\r" if phase == "training" else "\n"
            console.print(line, end=end, highlight=False, no_wrap=True, overflow="crop")

        return _cb

    _print("\nRunning walk-forward backtest ...", console, style="dim")

    # Silence noisy-but-normal log output during the backtest run.
    # hmmlearn emits convergence warnings on every EM candidate that hits the
    # iteration limit — expected behaviour, not an error.
    # core.hmm_engine regime-change lines are INFO in live mode (dashboard handles
    # visibility); suppress them here so the backtest output stays readable.
    logging.getLogger("hmmlearn").setLevel(logging.ERROR)
    logging.getLogger("core.hmm_engine").setLevel(logging.WARNING)

    bt = WalkForwardBacktester(
        symbols         = list(prices.columns),
        initial_capital = initial_capital,
        train_window    = train_window,
        test_window     = test_window,
        step_size       = step_size,
        slippage_pct    = slippage_pct,
        risk_free_rate  = risk_free_rate,
        zscore_window           = int(config.get("backtest", {}).get("zscore_window",           60)),
        sma_long                = int(config.get("backtest", {}).get("sma_long",               200)),
        sma_trend               = int(config.get("backtest", {}).get("sma_trend",               50)),
        volume_norm_window      = int(config.get("backtest", {}).get("volume_norm_window",       50)),
        min_rebalance_interval  = int(config.get("backtest", {}).get("min_rebalance_interval",    0)),
    )

    try:
        result = bt.run(
            prices,
            hmm_config=hmm_config,
            strategy_config=strategy_config,
            progress_callback=_make_bt_progress(len(list(prices.columns))),
            enforce_stops=getattr(args, "enforce_stops", False),
        )
    except Exception as exc:
        _print(f"[red]Backtest failed:[/red] {exc}", console)
        sys.exit(1)

    _print(
        f"  Folds completed : {result.metadata['n_folds']}\n"
        f"  OOS bars        : {len(result.combined_equity)}\n"
        f"  Total trades    : {result.metadata['total_trades']}\n"
        f"  Final equity    : ${result.final_equity:,.2f}",
        console, style="dim",
    )

    # ── Per-fold state count ──────────────────────────────────────────────────
    _fold_states = [w.n_hmm_states for w in result.windows]
    _state_counts = {}
    for n in _fold_states:
        _state_counts[n] = _state_counts.get(n, 0) + 1
    _state_summary = "  ".join(
        f"{n}-state × {c}" for n, c in sorted(_state_counts.items())
    )
    _print(
        f"  HMM states used : {_state_summary}  "
        f"(per fold: {', '.join(str(n) for n in _fold_states)})",
        console, style="dim",
    )

    # ── Regime bar counts + transition count ─────────────────────────────────
    if len(result.combined_regimes) > 0:
        _regime_counts = result.combined_regimes.value_counts().sort_index()
        _total_bars = len(result.combined_regimes)
        _label_order = [
            "CRASH", "STRONG_BEAR", "BEAR", "WEAK_BEAR",
            "NEUTRAL",
            "WEAK_BULL", "BULL", "STRONG_BULL", "EUPHORIA",
        ]
        # count regime changes (consecutive-bar transitions)
        _reg_series = result.combined_regimes.reset_index(drop=True)
        _n_changes = int((_reg_series != _reg_series.shift()).sum()) - 1  # -1 for first bar
        _avg_duration = _total_bars / max(_n_changes + 1, 1)
        # sort by the canonical bear→bull order; unknown labels go at end
        _sorted = sorted(
            _regime_counts.items(),
            key=lambda kv: (_label_order.index(kv[0]) if kv[0] in _label_order else 99),
        )
        _regime_lines = "".join(
            f"\n    {lbl:<14} {bars:>5} bars  ({bars / _total_bars:>5.1%})"
            for lbl, bars in _sorted
        )
        _print(
            f"  Regime counts  :{_regime_lines}\n"
            f"  Regime changes : {_n_changes}  (avg duration {_avg_duration:.1f} bars / regime)",
            console, style="dim",
        )

    # ── Per-regime P&L attribution ────────────────────────────────────────────
    if result.combined_regime_pnl:
        _label_order = [
            "CRASH", "STRONG_BEAR", "BEAR", "WEAK_BEAR",
            "NEUTRAL",
            "WEAK_BULL", "BULL", "STRONG_BULL", "EUPHORIA",
        ]
        _sorted_pnl = sorted(
            result.combined_regime_pnl.items(),
            key=lambda kv: (_label_order.index(kv[0]) if kv[0] in _label_order else 99),
        )
        _pnl_lines = "".join(
            f"\n    {lbl:<14} {'+' if pnl >= 0 else ''}{pnl:>10,.0f}"
            for lbl, pnl in _sorted_pnl
        )
        _print(f"  Regime P&L     :{_pnl_lines}", console, style="dim")

    _print("\n", console)
    pa = PerformanceAnalyzer(
        risk_free_rate=risk_free_rate,
        trading_days_per_year=252,
    )
    bm_sym    = symbols[0]
    bm_prices = prices[bm_sym] if getattr(args, "compare", False) else None
    report    = pa.analyze(result, benchmark_prices=bm_prices)

    run_context = _build_run_context(
        symbols     = list(prices.columns),
        start_date  = start_date,
        end_date    = end_date,
        config      = config,
        output_dir  = output_dir,
        asset_group = getattr(args, "asset_group", None),
        n_folds     = result.metadata.get("n_folds"),
    )
    pa.generate_report(report, print_to_console=True, run_context=run_context)

    if getattr(args, "compare", False) and bm_prices is not None:
        _print("\nComputing benchmark strategies ...", console, style="dim")

        # Same slippage applied to HMM strategy AND all benchmarks for
        # apples-to-apples comparison.
        _bt_cfg = config.get("backtest", {}) or {}
        _slip = float(_bt_cfg.get("slippage_pct", 0.0005))
        _sma_long = int(_bt_cfg.get("sma_long", 200))
        _ema_fast = int(_bt_cfg.get("ema_fast", 9))
        _ema_slow = int(_bt_cfg.get("ema_slow", 45))

        multi = len(symbols) > 1
        if multi:
            # Build a price DataFrame aligned to the strategy equity index,
            # dropping symbols that are missing from the fetched data.
            avail_syms = [s for s in symbols if s in prices]
            price_df = pd.DataFrame(
                {s: prices[s] for s in avail_syms}
            ).reindex(result.combined_equity.index).ffill().dropna(how="all")
            bnh_equity      = pa.compute_benchmark_bnh_multi(price_df, initial_capital, slippage_pct=_slip)
            sma_equity      = pa.compute_benchmark_sma_multi(price_df, _sma_long, initial_capital, slippage_pct=_slip)
            ema_cross_equity = pa.compute_benchmark_ema_cross_multi(price_df, _ema_fast, _ema_slow, initial_capital, slippage_pct=_slip)
            rand_mean, _ = pa.compute_random_allocation_benchmark_multi(
                price_df, allocations=[0.60, 0.95], n_seeds=100,
                initial_capital=initial_capital, slippage_pct=_slip,
            )
        else:
            bnh_equity       = pa.compute_benchmark_bnh(bm_prices, initial_capital, slippage_pct=_slip)
            sma_equity       = pa.compute_benchmark_sma(bm_prices, _sma_long, initial_capital, slippage_pct=_slip)
            ema_cross_equity = pa.compute_benchmark_ema_cross(bm_prices, _ema_fast, _ema_slow, initial_capital, slippage_pct=_slip)
            underlying_ret     = bm_prices.pct_change().dropna()
            underlying_aligned = underlying_ret.reindex(
                result.combined_returns.index
            ).dropna()
            rand_mean, _ = pa.compute_random_allocation_benchmark(
                underlying_aligned, allocations=[0.60, 0.95], n_seeds=100,
                initial_capital=initial_capital, slippage_pct=_slip,
            )

        eq_idx       = result.combined_equity.index
        bnh_rpt      = pa.analyze_equity_curve(
            bnh_equity.reindex(eq_idx).ffill().dropna()
        )
        sma_rpt      = pa.analyze_equity_curve(
            sma_equity.reindex(eq_idx).ffill().dropna()
        )
        ema_cross_rpt = pa.analyze_equity_curve(
            ema_cross_equity.reindex(eq_idx).ffill().dropna()
        )
        rand_rpt_metrics = {
            "total_return": float(rand_mean.iloc[-1] / initial_capital - 1),
            "cagr":         pa.compute_cagr(rand_mean),
            "sharpe_ratio": pa.compute_sharpe(rand_mean.pct_change().dropna()),
            "sortino_ratio":pa.compute_sortino(rand_mean.pct_change().dropna()),
            "max_drawdown": pa.compute_max_drawdown(rand_mean)[0],
            "calmar_ratio": pa.compute_calmar(rand_mean),
            "annualised_vol":float(rand_mean.pct_change().dropna().std() * (252**0.5)),
            "win_rate":     float((rand_mean.pct_change().dropna() > 0).mean()),
        }
        _print_run_config(symbols, start_date, end_date, config, console)
        _print_comparison_table(
            report, bnh_rpt, sma_rpt, ema_cross_rpt, rand_rpt_metrics, console,
            n_symbols=len(symbols),
            sma_long=_sma_long, ema_fast=_ema_fast, ema_slow=_ema_slow,
        )

    if getattr(args, "stress_test", False):
        _print("\nRunning stress scenarios ...", console, style="dim")
        stress = StressTester(bt, pa)
        stress_results = stress.run_stress_scenarios(
            prices, hmm_config=hmm_config, strategy_config=strategy_config,
        )
        if stress_results:
            tbl = stress.summary_table()
            _print("\n[bold]Stress Test Summary[/bold]", console)
            if console:
                try:
                    from rich.table import Table
                    from rich import box as rbox
                    rt = Table(title="Stress Test Results", box=rbox.ROUNDED,
                               header_style="bold red")
                    for col in tbl.reset_index().columns:
                        rt.add_column(col, justify="right")
                    for _, row in tbl.reset_index().iterrows():
                        rt.add_row(*[str(v) for v in row])
                    console.print(rt)
                except ImportError:
                    print(tbl.to_string())
            else:
                print(tbl.to_string())
            tbl.to_csv(output_dir / "stress_test_summary.csv")
            try:
                from telegram.notifier import notify as _tg_notify
                _tg_notify("stress")
            except Exception:
                pass

    _print(f"\nSaving results to {output_dir}/", console, style="dim")
    result.combined_equity.rename("equity").to_csv(
        output_dir / "equity_curve.csv", header=True
    )
    result.combined_regimes.rename("regime").to_csv(
        output_dir / "regime_history.csv", header=True
    )
    all_trades = [t for w in result.windows for t in w.trades]
    if all_trades:
        pd.DataFrame(all_trades).to_csv(output_dir / "trade_log.csv", index=False)

    metrics = {
        "total_return":  report.total_return,
        "cagr":          report.cagr,
        "sharpe":        report.sharpe_ratio,
        "sortino":       report.sortino_ratio,
        "max_drawdown":  report.max_drawdown,
        "max_dd_days":   report.max_drawdown_duration_days,
        "calmar":        report.calmar_ratio,
        "win_rate":      report.win_rate,
        "profit_factor": report.profit_factor,
        "total_trades":  report.total_trades,
        "final_equity":  result.final_equity,
        "n_folds":       result.metadata["n_folds"],
        "symbols":       ",".join(result.metadata.get("symbols", symbols)),
        "start":         start_date,
        "end":           end_date,
    }
    pd.Series(metrics).to_csv(output_dir / "performance_summary.csv", header=False)

    with open(output_dir / "run_context.json", "w") as _f:
        json.dump(run_context, _f, indent=2, default=str)

    try:
        from telegram.notifier import notify as _tg_notify
        _tg_notify("backtest")
    except Exception:
        pass

    _final_equity = result.final_equity
    _cagr_str  = f"{report.cagr:+.1%}" if report.cagr is not None else "N/A"
    _years_str = f"{_years:.1f}"
    _n_trades  = result.metadata["total_trades"]
    _trades_yr = int(_n_trades / max(_years, 1))
    _pf        = report.profit_factor
    _pf_str    = f"{_pf:.2f}" if _pf < 999 else "∞"
    _avg_win   = f"{report.avg_win:+.2%}/day" if report.avg_win else "—"
    _avg_loss  = f"{report.avg_loss:+.2%}/day" if report.avg_loss else "—"
    _uw_days   = report.longest_underwater_days
    _dd_days   = report.max_drawdown_duration_days

    _print(
        f"\n[green]Backtest complete.[/green]\n"
        f"  [bold]Return[/bold]   {report.total_return:+.2%}  "
        f"CAGR {_cagr_str}  "
        f"Sharpe [bold]{report.sharpe_ratio:.3f}[/bold]  "
        f"Sortino {report.sortino_ratio:.3f}  "
        f"Calmar {report.calmar_ratio:.3f}  "
        f"Profit Factor {_pf_str}\n"
        f"  [bold]Risk[/bold]     MaxDD {report.max_drawdown:.1%} ({_dd_days}d)  "
        f"Ann Vol {report.annualised_vol:.1%}  "
        f"Worst day {report.worst_day:.1%}  "
        f"Underwater {_uw_days}d\n"
        f"  [bold]Trades[/bold]   {_n_trades} · {_trades_yr}/yr · "
        f"{report.win_rate:.1%} win rate  "
        f"up days {_avg_win}  down days {_avg_loss}\n"
        f"  [dim]${initial_capital:,.0f} → ${_final_equity:,.0f}  "
        f"over {_years_str}y   "
        f"{result.metadata['n_folds']} OOS folds — no lookahead bias[/dim]",
        console,
    )


def run_trading(
    config: Dict,
    dry_run: bool = False,
    allocator_approach: str = "inverse_vol",
    no_portfolio_risk: bool = False,
    multi_strat_filter: Optional[List[str]] = None,
) -> None:
    """
    Full live / paper trading loop.

    Handles SIGINT / SIGTERM for graceful shutdown.
    All unhandled exceptions are logged, state is saved, and an alert is fired.
    """
    session = TradingSession(
        config,
        dry_run=dry_run,
        allocator_approach=allocator_approach,
        no_portfolio_risk=no_portfolio_risk,
        multi_strat_filter=multi_strat_filter,
    )

    # Register shutdown signal handlers
    def _handle_signal(signum, frame):
        logger.info("Signal %d received -- initiating shutdown", signum)
        session.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT,  _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        session.startup()
    except Exception as exc:
        _print(f"[red]Startup failed:[/red] {exc}", _console())
        logger.error("Startup failed: %s\n%s", exc, traceback.format_exc())
        try:
            session.shutdown()
        except Exception:
            pass
        sys.exit(1)

    try:
        session.run_loop()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        logger.critical(
            "Unhandled exception in trading loop: %s\n%s",
            exc, traceback.format_exc(),
        )
        try:
            if session.trade_logger:
                session.trade_logger.log_error(exc, context="main_loop_unhandled")
            if session.alert_manager:
                session.alert_manager.alert(
                    title   = "CRITICAL: Unhandled exception in trading loop",
                    message = f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}",
                    level   = "CRITICAL",
                )
        except Exception:
            pass
    finally:
        session.shutdown()


def run_train_only(config: Dict, args: Optional[argparse.Namespace] = None) -> None:
    """Train HMM on latest data and exit without entering the trading loop."""
    console = _console()
    broker_cfg = config.get("broker", {})
    hmm_cfg    = config.get("hmm",    {})
    symbols    = _resolve_symbols(
        config,
        asset_group=getattr(args, "asset_group", None) if args else None,
        symbols_arg=getattr(args, "symbols", None)     if args else None,
    )
    paper      = broker_cfg.get("paper_trading", True)

    _print("[bold cyan]Regime Trader - Train Only[/bold cyan]", console)

    from broker.alpaca_client import AlpacaClient
    client = AlpacaClient(paper=paper)
    client.connect_with_retry(max_attempts=3, base_delay=2.0)

    # Train on daily bars regardless of live trading timeframe — regimes are
    # macroeconomic (weeks/months), not intraday. 756 daily bars ≈ 3 years.
    hmm_cfg_train = {**hmm_cfg, "timeframe": "1Day"}
    engine = _train_hmm(client, symbols, hmm_cfg_train, console=console, n_bars=756)
    _save_training_log(engine, symbols, hmm_cfg)
    _print(
        f"\n[green]Training complete.[/green]  "
        f"states={engine._n_states}  BIC={engine._training_bic:.2f}  "
        f"saved -> {_MODEL_PATH}",
        console,
    )

    # ── Detailed regime statistics ────────────────────────────────────────
    _CANONICAL = ["CRASH","STRONG_BEAR","BEAR","WEAK_BEAR","NEUTRAL","WEAK_BULL","BULL","STRONG_BULL","EUPHORIA"]

    regimes: Optional[pd.Series] = getattr(engine, "_training_regimes", None)
    if regimes is not None and len(regimes) > 0:
        present = set(regimes)
        ordered_labels = [l for l in _CANONICAL if l in present] + \
                         [l for l in present if l not in _CANONICAL]
        idx_map = {lbl: i + 1 for i, lbl in enumerate(ordered_labels)}

        _print("\n[bold cyan]Regime Breakdown (in-sample)[/bold cyan]", console)
        _print(f"  {'#':>2}  {'Regime':<16} {'Bars':>6} {'%Time':>7} {'AvgDur':>8} {'MinDur':>7} {'MaxDur':>7}", console)
        _print("  " + "─" * 61, console)

        # Compute run-length encoding for durations
        _runs: Dict[str, list] = {}
        cur_label, cur_len = regimes.iloc[0], 1
        for lbl in regimes.iloc[1:]:
            if lbl == cur_label:
                cur_len += 1
            else:
                _runs.setdefault(cur_label, []).append(cur_len)
                cur_label, cur_len = lbl, 1
        _runs.setdefault(cur_label, []).append(cur_len)

        total_bars = len(regimes)
        for lbl in ordered_labels:
            runs = _runs.get(lbl, [1])
            count = sum(runs)
            pct   = count / total_bars * 100
            avg_d = sum(runs) / len(runs)
            _print(
                f"  {idx_map[lbl]:>2}  {lbl:<16} {count:>6}  {pct:>6.1f}%  {avg_d:>7.1f}  {min(runs):>6}  {max(runs):>6}",
                console,
            )

        n_changes = int((regimes != regimes.shift()).sum()) - 1
        _print("  " + "─" * 61, console)
        _print(f"  Total regime changes : {n_changes}", console)
        _print(f"  Avg bars per regime  : {total_bars / max(n_changes + 1, 1):.1f}", console)
        _print(f"  Current regime       : [bold]{regimes.iloc[-1]}[/bold]  #{idx_map[regimes.iloc[-1]]}  (last bar: {regimes.index[-1]})", console)

        # Transition matrix (ordered logically)
        _print("\n[bold cyan]Transition Matrix (counts)[/bold cyan]", console)
        header = f"  {'#→':>4}  {'':>12} " + " ".join(f"{idx_map[l]:>6}" for l in ordered_labels)
        _print(header, console)
        for from_l in ordered_labels:
            row = ""
            for to_l in ordered_labels:
                mask = (regimes == from_l) & (regimes.shift(-1) == to_l)
                row += f" {int(mask.sum()):>6}"
            _print(f"  {idx_map[from_l]:>2}    {from_l[:12]:<12}{row}", console)

    client.disconnect()


def run_full_cycle(config: Dict, args: argparse.Namespace) -> None:
    """
    Execute HMM training and full backtest for every asset group defined in
    config/asset_groups.yaml, then display a consolidated summary.
    """
    console = _console()
    try:
        from core.asset_groups import load_default_registry
        asset_groups = load_default_registry(reload=True).list()
    except Exception as exc:
        logger.warning("AssetGroupRegistry unavailable (%s); using legacy bloc", exc)
        asset_groups = list((config.get("asset_groups") or {}).keys()) or ["stocks", "crypto", "indices"]
    all_results = {}

    _print("\n[bold magenta]Starting Full Cycle: HMM + Backtest for all Asset Groups[/bold magenta]", console)

    for group in asset_groups:
        _print(f"\n[bold yellow]>>> Processing Group: {group.upper()} <<<[/bold yellow]", console)
        
        # Create a copy of args to customize for each group
        group_args = argparse.Namespace(**vars(args))
        group_args.asset_group = group
        group_args.compare = True
        
        # Force a fresh training for each group
        # Note: run_backtest doesn't explicitly 'train' a global model, 
        # it trains per-fold models inside the walk-forward loop.
        
        try:
            # We wrap run_backtest to capture results or just let it print
            # To get a summary, we can use the PerformanceAnalyzer directly after run_backtest
            # Since run_backtest prints its own report, we can just call it.
            run_backtest(config, group_args)
            
            # After run_backtest, the results are saved in a timestamped dir.
            # For a consolidated summary, we'd need to collect metrics.
            # Let's assume we want to show a final table of all 3.
            
            # Logic to extract last result's performance_summary.csv
            output_dir = _SAVED_RESULTS_DIR
            latest_run = sorted([d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("backtest_")])[-1]
            summary_path = latest_run / "performance_summary.csv"
            if summary_path.exists():
                summary = pd.read_csv(summary_path, header=None, names=["metric", "value"])
                all_results[group] = summary.set_index("metric")["value"].to_dict()

        except Exception as exc:
            _print(f"[red]Failed full cycle for {group}:[/red] {exc}", console)
            continue

    if all_results:
        _print("\n[bold cyan]FINAL CROSS-ASSET SUMMARY[/bold cyan]", console)
        try:
            from rich.table import Table
            from rich import box as rbox
            tbl = Table(title="Asset Group Comparison", box=rbox.ROUNDED, header_style="bold yellow")
            tbl.add_column("Metric", style="dim")
            for group in all_results.keys():
                tbl.add_column(group.upper(), justify="right")
            
            metrics_to_show = [
                ("Total Return", "total_return", "{:+.2%}"),
                ("Sharpe Ratio", "sharpe", "{:.3f}"),
                ("Max Drawdown", "max_drawdown", "{:.2%}"),
                ("Win Rate", "win_rate", "{:.2%}"),
                ("Total Trades", "total_trades", "{:.0f}"),
            ]
            
            for label, key, fmt in metrics_to_show:
                row = [label]
                for group in all_results.keys():
                    val = all_results[group].get(key, "N/A")
                    try:
                        row.append(fmt.format(float(val)))
                    except (ValueError, TypeError):
                        row.append(str(val))
                tbl.add_row(*row)
            console.print(tbl)
        except Exception as exc:
            print("\nFinal Summary Data:")
            print(all_results)


def run_interval_sweep(config: Dict, args: argparse.Namespace) -> None:
    """
    Sweep min_rebalance_interval values in-process.

    Data is fetched ONCE. Because HMM training uses deterministic seeds
    (random_state = seed * 13 + n_states) the exact same models are trained
    for every interval value — only trade execution differs, giving a clean
    apples-to-apples comparison.
    """
    from backtest.backtester import WalkForwardBacktester
    from backtest.performance import PerformanceAnalyzer

    console = _console()

    # ── Parse sweep values ────────────────────────────────────────────────────
    raw_values = getattr(args, "values", None) or "0,1,2,3,5,7,10,15,20"
    try:
        sweep_values = [int(v.strip()) for v in raw_values.split(",")]
    except ValueError:
        _print("[red]--values must be comma-separated integers, e.g. 0,1,3,5,10[/red]", console)
        sys.exit(1)

    symbols: List[str] = _resolve_symbols(
        config,
        asset_group=getattr(args, "asset_group", None),
        symbols_arg=getattr(args, "symbols", None),
    )

    bt_cfg          = config.get("backtest", {})
    start_date: str = getattr(args, "start", None) or "2020-01-01"
    end_date: str   = getattr(args, "end",   None) or pd.Timestamp.today().strftime("%Y-%m-%d")

    initial_capital = float(bt_cfg.get("initial_capital", 100_000))
    slippage_pct    = float(bt_cfg.get("slippage_pct",    0.0005))
    train_window    = int(bt_cfg.get("train_window",      252))
    test_window     = int(bt_cfg.get("test_window",       126))
    step_size       = int(bt_cfg.get("step_size",         126))
    risk_free_rate  = float(bt_cfg.get("risk_free_rate",  0.045))

    _print(
        f"\n[bold cyan]min_rebalance_interval Sweep[/bold cyan]\n"
        f"  Symbols : {', '.join(symbols[:6])}{'…' if len(symbols) > 6 else ''}\n"
        f"  Period  : {start_date}  →  {end_date}\n"
        f"  Values  : {sweep_values}",
        console,
    )

    # ── Fetch data once ───────────────────────────────────────────────────────
    api_key    = os.environ.get("ALPACA_API_KEY",    "")
    secret_key = os.environ.get("ALPACA_SECRET_KEY", "")
    if not api_key or not secret_key:
        _print("[red]ERROR: Alpaca credentials not found.[/red]", console)
        sys.exit(1)

    _print("\nFetching historical data ...", console, style="dim")
    try:
        prices = _fetch_prices(symbols, start_date, end_date, api_key, secret_key)
    except Exception as exc:
        _print(f"[red]Data fetch failed:[/red] {exc}", console)
        sys.exit(1)
    _print(
        f"  {len(prices)} bars  "
        f"({prices.index[0].date()} → {prices.index[-1].date()})",
        console, style="dim",
    )

    # ── Build configs (same for every sweep iteration) ────────────────────────
    hmm_cfg_raw = config.get("hmm", {})
    hmm_config = {
        "n_candidates":       hmm_cfg_raw.get("n_candidates", [3, 4, 5]),
        "n_init":             hmm_cfg_raw.get("n_init", 10),
        "min_train_bars":     hmm_cfg_raw.get("min_train_bars", 120),
        "stability_bars":     hmm_cfg_raw.get("stability_bars", 3),
        "flicker_window":     hmm_cfg_raw.get("flicker_window", 20),
        "flicker_threshold":  hmm_cfg_raw.get("flicker_threshold", 4),
        "min_confidence":     hmm_cfg_raw.get("min_confidence", 0.55),
        # Feature selection (was silently dropped before — the backtester
        # was always using HMM_EXTENDED_FEATURES regardless of settings.yaml).
        "extended_features":  hmm_cfg_raw.get("extended_features", True),
        "features_override":  hmm_cfg_raw.get("features_override"),
        "use_vix_features":   hmm_cfg_raw.get("use_vix_features", False),
        "use_credit_spread_features": hmm_cfg_raw.get("use_credit_spread_features", False),
        "blend_exclude":      hmm_cfg_raw.get("blend_exclude", []),
    }
    strategy_config = {"strategy": config.get("strategy", {})}

    common_bt_kwargs = dict(
        symbols         = list(prices.columns),
        initial_capital = initial_capital,
        train_window    = train_window,
        test_window     = test_window,
        step_size       = step_size,
        slippage_pct    = slippage_pct,
        risk_free_rate  = risk_free_rate,
        zscore_window        = int(bt_cfg.get("zscore_window",        60)),
        sma_long             = int(bt_cfg.get("sma_long",            200)),
        sma_trend            = int(bt_cfg.get("sma_trend",            50)),
        volume_norm_window   = int(bt_cfg.get("volume_norm_window",    50)),
    )

    # ── Silence noisy logs during sweep ──────────────────────────────────────
    logging.getLogger("hmmlearn").setLevel(logging.ERROR)
    logging.getLogger("core.hmm_engine").setLevel(logging.WARNING)

    pa = PerformanceAnalyzer(risk_free_rate=risk_free_rate, trading_days_per_year=252)

    # ── Sweep ─────────────────────────────────────────────────────────────────
    rows: List[Dict] = []
    n = len(sweep_values)
    for i, interval in enumerate(sweep_values):
        if console:
            console.print(f"  [{i+1}/{n}] interval={interval:>3} ...", style="dim", end="\r")
        else:
            print(f"  [{i+1}/{n}] interval={interval:>3} ...", end="\r", flush=True)
        bt = WalkForwardBacktester(**common_bt_kwargs, min_rebalance_interval=interval)
        try:
            result = bt.run(prices, hmm_config=hmm_config, strategy_config=strategy_config)
        except Exception as exc:
            _print(f"\n[red]  interval={interval} failed:[/red] {exc}", console)
            rows.append({"interval": interval, "error": str(exc)})
            continue

        rpt = pa.analyze(result)
        rows.append({
            "interval": interval,
            "sharpe":   rpt.sharpe_ratio,
            "cagr":     rpt.cagr if rpt.cagr is not None else float("nan"),
            "max_dd":   rpt.max_drawdown,
            "calmar":   rpt.calmar_ratio if rpt.calmar_ratio is not None else float("nan"),
            "trades":   rpt.total_trades,
            "final_eq": result.final_equity,
        })

    _print("", console)  # clear the \r line

    if not rows or all("error" in r for r in rows):
        _print("[red]All sweep iterations failed.[/red]", console)
        return

    # ── Find winner (best Sharpe among successful runs) ───────────────────────
    ok_rows   = [r for r in rows if "error" not in r]
    best_row  = max(ok_rows, key=lambda r: r["sharpe"])
    best_iv   = best_row["interval"]

    # ── Rich results table ────────────────────────────────────────────────────
    try:
        from rich.table import Table
        from rich import box as rbox
        from rich.text import Text

        tbl = Table(
            title=f"min_rebalance_interval Sweep — {', '.join(symbols[:4])}{'…' if len(symbols) > 4 else ''}",
            box=rbox.ROUNDED,
            header_style="bold cyan",
        )
        tbl.add_column("Interval",  justify="right")
        tbl.add_column("Sharpe",    justify="right")
        tbl.add_column("CAGR",      justify="right")
        tbl.add_column("Max DD",    justify="right")
        tbl.add_column("Calmar",    justify="right")
        tbl.add_column("Trades",    justify="right")
        tbl.add_column("Final $",   justify="right")

        for r in rows:
            is_best   = r.get("interval") == best_iv and "error" not in r
            row_style = "bold green" if is_best else ""

            if "error" in r:
                tbl.add_row(
                    str(r["interval"]), "[red]ERROR[/red]", "", "", "", "", r["error"][:40],
                )
                continue

            iv_cell = (
                Text(f"{r['interval']:>3} ★", style="bold green")
                if is_best
                else str(r["interval"])
            )
            tbl.add_row(
                iv_cell,
                Text(f"{r['sharpe']:>6.3f}", style=row_style),
                Text(f"{r['cagr']:>+7.2%}", style=row_style),
                Text(f"{r['max_dd']:>7.2%}", style=row_style),
                Text(f"{r['calmar']:>6.3f}", style=row_style),
                Text(f"{r['trades']:>6d}",   style=row_style),
                Text(f"${r['final_eq']:>12,.0f}", style=row_style),
            )

        if console:
            console.print(tbl)
        else:
            # Fallback plain text
            print(f"\n{'Interval':>9}  {'Sharpe':>7}  {'CAGR':>8}  {'MaxDD':>7}  {'Calmar':>7}  {'Trades':>7}  {'Final $':>12}")
            for r in rows:
                if "error" in r:
                    print(f"{r['interval']:>9}  ERROR")
                    continue
                marker = " ★" if r["interval"] == best_iv else "  "
                print(
                    f"{r['interval']:>9}{marker}"
                    f"  {r['sharpe']:>7.3f}"
                    f"  {r['cagr']:>+8.2%}"
                    f"  {r['max_dd']:>7.2%}"
                    f"  {r['calmar']:>7.3f}"
                    f"  {r['trades']:>7d}"
                    f"  ${r['final_eq']:>12,.0f}"
                )

    except ImportError:
        print("\nInterval  Sharpe  CAGR     MaxDD   Calmar  Trades   Final $")
        for r in rows:
            if "error" in r:
                print(f"{r['interval']:>8}  ERROR")
                continue
            marker = "★" if r["interval"] == best_iv else " "
            print(
                f"{r['interval']:>8}{marker}"
                f"  {r['sharpe']:>6.3f}"
                f"  {r['cagr']:>+7.2%}"
                f"  {r['max_dd']:>6.2%}"
                f"  {r['calmar']:>6.3f}"
                f"  {r['trades']:>6d}"
                f"  ${r['final_eq']:>12,.0f}"
            )

    _print(
        f"\n[bold green]Winner: interval={best_iv}[/bold green]  "
        f"Sharpe={best_row['sharpe']:.3f}  "
        f"CAGR={best_row['cagr']:+.2%}  "
        f"MaxDD={best_row['max_dd']:.2%}  "
        f"Trades={best_row['trades']}",
        console,
    )

    # ── Save to CSV ───────────────────────────────────────────────────────────
    _SAVED_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_path = _SAVED_RESULTS_DIR / f"interval_sweep_{ts}.csv"
    pd.DataFrame(ok_rows).to_csv(out_path, index=False)
    _print(f"  Saved → {out_path}", console, style="dim")


def run_cs_sweep(config: Dict, args: argparse.Namespace) -> None:
    """
    2-D grid sweep over min_confidence × stability_bars.

    HMM models are trained once per fold and replayed for every grid cell
    (both parameters are inference-only — they don't change the EM weights).
    This makes the sweep ~n_cells times faster than running full backtests.
    """
    from backtest.backtester import WalkForwardBacktester
    from backtest.performance import PerformanceAnalyzer

    console = _console()

    # ── Parse grid values ─────────────────────────────────────────────────────
    raw_conf = getattr(args, "conf_values", None) or "0.55,0.60,0.65,0.70,0.75"
    raw_stab = getattr(args, "stab_values", None) or "3,5,7,9,12"
    try:
        conf_values = [float(v.strip()) for v in raw_conf.split(",")]
        stab_values = [int(v.strip())   for v in raw_stab.split(",")]
    except ValueError:
        _print("[red]--conf and --stab must be comma-separated numbers[/red]", console)
        sys.exit(1)

    symbols: List[str] = _resolve_symbols(
        config,
        asset_group=getattr(args, "asset_group", None),
        symbols_arg=getattr(args, "symbols",     None),
    )

    bt_cfg          = config.get("backtest", {})
    start_date: str = getattr(args, "start", None) or "2020-01-01"
    end_date: str   = getattr(args, "end",   None) or pd.Timestamp.today().strftime("%Y-%m-%d")

    initial_capital = float(bt_cfg.get("initial_capital", 100_000))
    slippage_pct    = float(bt_cfg.get("slippage_pct",    0.0005))
    train_window    = int(bt_cfg.get("train_window",      252))
    test_window     = int(bt_cfg.get("test_window",       126))
    step_size       = int(bt_cfg.get("step_size",         126))
    risk_free_rate  = float(bt_cfg.get("risk_free_rate",  0.045))

    n_cells = len(conf_values) * len(stab_values)
    _print(
        f"\n[bold cyan]min_confidence × stability_bars Grid Sweep[/bold cyan]\n"
        f"  Symbols : {', '.join(symbols[:6])}{'…' if len(symbols) > 6 else ''}\n"
        f"  Period  : {start_date}  →  {end_date}\n"
        f"  conf    : {conf_values}\n"
        f"  stab    : {stab_values}\n"
        f"  Cells   : {n_cells}  (HMM trained once per fold, replayed per cell)",
        console,
    )

    api_key    = os.environ.get("ALPACA_API_KEY",    "")
    secret_key = os.environ.get("ALPACA_SECRET_KEY", "")
    if not api_key or not secret_key:
        _print("[red]ERROR: Alpaca credentials not found.[/red]", console)
        sys.exit(1)

    _print("\nFetching historical data ...", console, style="dim")
    try:
        prices = _fetch_prices(symbols, start_date, end_date, api_key, secret_key)
    except Exception as exc:
        _print(f"[red]Data fetch failed:[/red] {exc}", console)
        sys.exit(1)
    _print(
        f"  {len(prices)} bars  ({prices.index[0].date()} → {prices.index[-1].date()})",
        console, style="dim",
    )

    hmm_cfg_raw = config.get("hmm", {})
    hmm_config = {
        "n_candidates":       hmm_cfg_raw.get("n_candidates", [3, 4, 5]),
        "n_init":             hmm_cfg_raw.get("n_init", 10),
        "min_train_bars":     hmm_cfg_raw.get("min_train_bars", 120),
        "stability_bars":     hmm_cfg_raw.get("stability_bars", 3),
        "flicker_window":     hmm_cfg_raw.get("flicker_window", 20),
        "flicker_threshold":  hmm_cfg_raw.get("flicker_threshold", 4),
        "min_confidence":     hmm_cfg_raw.get("min_confidence", 0.55),
        # Feature selection (was silently dropped before — the backtester
        # was always using HMM_EXTENDED_FEATURES regardless of settings.yaml).
        "extended_features":  hmm_cfg_raw.get("extended_features", True),
        "features_override":  hmm_cfg_raw.get("features_override"),
        "use_vix_features":   hmm_cfg_raw.get("use_vix_features", False),
        "use_credit_spread_features": hmm_cfg_raw.get("use_credit_spread_features", False),
        "blend_exclude":      hmm_cfg_raw.get("blend_exclude", []),
    }
    strategy_config = {"strategy": config.get("strategy", {})}

    logging.getLogger("hmmlearn").setLevel(logging.ERROR)
    logging.getLogger("core.hmm_engine").setLevel(logging.WARNING)

    bt = WalkForwardBacktester(
        symbols         = list(prices.columns),
        initial_capital = initial_capital,
        train_window    = train_window,
        test_window     = test_window,
        step_size       = step_size,
        slippage_pct    = slippage_pct,
        risk_free_rate  = risk_free_rate,
        zscore_window        = int(bt_cfg.get("zscore_window",        60)),
        sma_long             = int(bt_cfg.get("sma_long",            200)),
        sma_trend            = int(bt_cfg.get("sma_trend",            50)),
        volume_norm_window   = int(bt_cfg.get("volume_norm_window",    50)),
        min_rebalance_interval = int(bt_cfg.get("min_rebalance_interval", 0)),
    )

    pa = PerformanceAnalyzer(risk_free_rate=risk_free_rate, trading_days_per_year=252)

    cell_counter = [0]
    n_folds_total = [0]

    def _grid_progress(phase: str, n_folds: int, idx: int, n_cells: int) -> None:
        n_folds_total[0] = n_folds
        if phase == "training":
            msg = f"  Training fold {idx+1}/{n_folds} ..."
        elif phase == "sweep":
            cell_counter[0] = idx + 1
            conf_i = idx // len(stab_values)
            stab_i = idx %  len(stab_values)
            msg = (
                f"  Sweeping cell {idx+1}/{n_cells}  "
                f"conf={conf_values[conf_i]:.2f}  stab={stab_values[stab_i]} ..."
            )
        else:
            return
        if console:
            console.print(msg, style="dim", end="\r")
        else:
            print(msg, end="\r", flush=True)

    _print("\nRunning grid sweep ...", console, style="dim")
    try:
        grid_results = bt.run_grid(
            prices,
            conf_values=conf_values,
            stab_values=stab_values,
            hmm_config=hmm_config,
            strategy_config=strategy_config,
            progress_callback=_grid_progress,
        )
    except Exception as exc:
        _print(f"\n[red]Grid sweep failed:[/red] {exc}", console)
        sys.exit(1)

    _print("", console)  # clear \r line

    if not grid_results:
        _print("[red]No results produced.[/red]", console)
        return

    # ── Collect metrics ───────────────────────────────────────────────────────
    rows = []
    for conf, stab, result in grid_results:
        rpt = pa.analyze(result)
        rows.append({
            "conf":     conf,
            "stab":     stab,
            "sharpe":   rpt.sharpe_ratio,
            "cagr":     rpt.cagr if rpt.cagr is not None else float("nan"),
            "max_dd":   rpt.max_drawdown,
            "calmar":   rpt.calmar_ratio if rpt.calmar_ratio is not None else float("nan"),
            "trades":   rpt.total_trades,
            "final_eq": result.final_equity,
        })

    best = max(rows, key=lambda r: r["sharpe"])

    # ── Rich 2-D Sharpe heatmap table ─────────────────────────────────────────
    try:
        from rich.table import Table
        from rich import box as rbox
        from rich.text import Text

        # Flat sorted table
        flat_tbl = Table(
            title=f"conf × stab Grid — {', '.join(symbols[:4])}{'…' if len(symbols) > 4 else ''}",
            box=rbox.ROUNDED,
            header_style="bold cyan",
        )
        flat_tbl.add_column("Conf",    justify="right")
        flat_tbl.add_column("Stab",    justify="right")
        flat_tbl.add_column("Sharpe",  justify="right")
        flat_tbl.add_column("CAGR",    justify="right")
        flat_tbl.add_column("Max DD",  justify="right")
        flat_tbl.add_column("Calmar",  justify="right")
        flat_tbl.add_column("Trades",  justify="right")
        flat_tbl.add_column("Final $", justify="right")

        for r in sorted(rows, key=lambda x: x["sharpe"], reverse=True):
            is_best   = r["conf"] == best["conf"] and r["stab"] == best["stab"]
            style     = "bold green" if is_best else ""
            conf_cell = (
                Text(f"{r['conf']:.2f} ★", style="bold green") if is_best
                else f"{r['conf']:.2f}"
            )
            flat_tbl.add_row(
                conf_cell,
                Text(f"{r['stab']}",              style=style),
                Text(f"{r['sharpe']:>6.3f}",       style=style),
                Text(f"{r['cagr']:>+7.2%}",        style=style),
                Text(f"{r['max_dd']:>7.2%}",        style=style),
                Text(f"{r['calmar']:>6.3f}",        style=style),
                Text(f"{r['trades']:>6d}",          style=style),
                Text(f"${r['final_eq']:>12,.0f}",   style=style),
            )

        # 2-D Sharpe grid
        grid_tbl = Table(
            title="Sharpe  (rows = stability_bars, cols = min_confidence)",
            box=rbox.SIMPLE_HEAD,
            header_style="bold yellow",
        )
        grid_tbl.add_column("stab \\ conf", justify="right")
        for c in conf_values:
            grid_tbl.add_column(f"{c:.2f}", justify="right")

        sharpe_lookup = {(r["conf"], r["stab"]): r["sharpe"] for r in rows}
        for stab in stab_values:
            cells = []
            for conf in conf_values:
                s = sharpe_lookup.get((conf, stab), float("nan"))
                is_best_cell = conf == best["conf"] and stab == best["stab"]
                txt = f"{s:.3f}" if not (s != s) else "—"
                cells.append(Text(txt, style="bold green") if is_best_cell else txt)
            grid_tbl.add_row(str(stab), *cells)

        if console:
            console.print(flat_tbl)
            console.print(grid_tbl)
        else:
            print("\nSorted results:")
            for r in sorted(rows, key=lambda x: x["sharpe"], reverse=True):
                marker = " ★" if r["conf"] == best["conf"] and r["stab"] == best["stab"] else "  "
                print(f"  conf={r['conf']:.2f} stab={r['stab']:>2}{marker}"
                      f"  Sharpe={r['sharpe']:.3f}  CAGR={r['cagr']:+.2%}"
                      f"  MaxDD={r['max_dd']:.2%}  Trades={r['trades']}")

    except ImportError:
        for r in sorted(rows, key=lambda x: x["sharpe"], reverse=True):
            print(f"conf={r['conf']:.2f} stab={r['stab']:>2}  "
                  f"Sharpe={r['sharpe']:.3f}  CAGR={r['cagr']:+.2%}")

    _print(
        f"\n[bold green]Winner: conf={best['conf']:.2f}  stab={best['stab']}[/bold green]  "
        f"Sharpe={best['sharpe']:.3f}  CAGR={best['cagr']:+.2%}  "
        f"MaxDD={best['max_dd']:.2%}  Trades={best['trades']}",
        console,
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    # Persist rows in the same Sharpe-descending order shown in the console
    # tables so a human scanning the CSV sees the best configs first.
    _SAVED_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_path = _SAVED_RESULTS_DIR / f"cs_sweep_{ts}.csv"
    sorted_rows = sorted(rows, key=lambda x: x["sharpe"], reverse=True)
    pd.DataFrame(sorted_rows).to_csv(out_path, index=False)
    _print(f"  Saved → {out_path}", console, style="dim")


def run_stress_test(config: Dict, args: argparse.Namespace) -> None:
    """Standalone stress-test entry point (delegates to run_backtest --stress-test)."""
    args.stress_test = True
    if not hasattr(args, "compare"):
        args.compare = False
    run_backtest(config, args)


# ── Asset groups CLI ───────────────────────────────────────────────────────────

def _split_csv(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [s.strip() for s in value.split(",") if s.strip()]


def run_groups(args: argparse.Namespace) -> int:
    """Dispatcher for `main.py groups <action>`. Returns exit code."""
    from core.asset_groups import AssetGroup, load_default_registry

    reg = load_default_registry(reload=True)
    action = args.groups_action
    console = _console()

    def _print_table(groups):
        try:
            from rich.table import Table
            t = Table(title="Asset Groups", show_lines=False)
            t.add_column("name", style="cyan")
            t.add_column("class")
            t.add_column("tags", style="dim")
            t.add_column("#", justify="right")
            t.add_column("symbols", overflow="fold")
            default_name = reg.default()
            for g in groups:
                marker = " [green]*[/green]" if g.name == default_name else ""
                warn_flag = " [red]⚠[/red]" if getattr(g, "warning", "") else ""
                t.add_row(
                    g.name + marker + warn_flag,
                    g.asset_class or "-",
                    ",".join(g.tags) or "-",
                    str(len(g.symbols)),
                    ", ".join(g.symbols),
                )
            if console:
                console.print(t)
                console.print(f"  [dim]default: {default_name}  (marked with *)[/dim]")
            else:
                for g in groups:
                    print(f"{g.name}\t{len(g.symbols)}\t{','.join(g.symbols)}")
        except Exception:
            for g in groups:
                print(f"{g.name}\t{len(g.symbols)}\t{','.join(g.symbols)}")

    if action == "list":
        if getattr(args, "names_only", False):
            for name in reg.list():
                print(name)
            return 0
        if getattr(args, "json", False):
            print(json.dumps(
                {"default": reg.default(),
                 "groups": {n: g.to_dict() for n, g in reg.all().items()}},
                indent=2))
            return 0
        _print_table(list(reg.all().values()))
        return 0

    if action == "show":
        if not reg.has(args.name):
            _print(f"[red]Group not found:[/red] {args.name}", console)
            return 1
        g = reg.get(args.name)
        if getattr(args, "json", False):
            print(json.dumps({args.name: g.to_dict()}, indent=2))
        else:
            _print_table([g])
        return 0

    if action == "add":
        group = AssetGroup(
            name=args.name,
            symbols=tuple(_split_csv(args.symbols)),
            description=args.description,
            asset_class=args.asset_class,
            tags=tuple(_split_csv(args.tags)),
        )
        try:
            reg.add(group, overwrite=args.overwrite)
        except ValueError as exc:
            _print(f"[red]{exc}[/red]", console)
            return 1
        _print(f"[green]Added[/green] group '{args.name}' with {len(group.symbols)} symbols.", console)
        return 0

    if action == "remove":
        try:
            reg.remove(args.name)
        except KeyError as exc:
            _print(f"[red]{exc}[/red]", console)
            return 1
        _print(f"[yellow]Removed[/yellow] group '{args.name}'.", console)
        return 0

    if action == "edit":
        try:
            updated = reg.update(
                args.name,
                symbols=_split_csv(args.symbols) if args.symbols is not None else None,
                add_symbols=_split_csv(args.add_symbols) if args.add_symbols else None,
                remove_symbols=_split_csv(args.remove_symbols) if args.remove_symbols else None,
                description=args.description,
                asset_class=args.asset_class,
                tags=_split_csv(args.tags) if args.tags is not None else None,
            )
        except KeyError as exc:
            _print(f"[red]{exc}[/red]", console)
            return 1
        _print(f"[green]Updated[/green] '{args.name}' → {len(updated.symbols)} symbols.", console)
        return 0

    if action == "rename":
        try:
            reg.rename(args.old, args.new)
        except (KeyError, ValueError) as exc:
            _print(f"[red]{exc}[/red]", console)
            return 1
        _print(f"[green]Renamed[/green] '{args.old}' → '{args.new}'.", console)
        return 0

    if action == "set-default":
        try:
            reg.set_default(args.name)
        except KeyError as exc:
            _print(f"[red]{exc}[/red]", console)
            return 1
        _print(f"[green]Default group set to[/green] '{args.name}'.", console)
        return 0

    if action == "default":
        print(reg.default())
        return 0

    if action == "validate":
        errs = reg.validate()
        if not errs:
            _print("[green]OK[/green] — no problems detected.", console)
            return 0
        for e in errs:
            _print(f"[red]•[/red] {e}", console)
        return 1

    if action == "export":
        payload = {
            "version": 1,
            "default": reg.default(),
            "groups": {n: g.to_dict() for n, g in reg.all().items()},
        }
        text = json.dumps(payload, indent=2)
        if args.out:
            Path(args.out).write_text(text, encoding="utf-8")
            _print(f"[green]Wrote[/green] {args.out}", console)
        else:
            print(text)
        return 0

    if action == "import":
        data = json.loads(Path(args.path).read_text(encoding="utf-8"))
        groups = data.get("groups", data)  # allow either {groups: ...} or flat mapping
        added = 0
        for name, body in groups.items():
            g = AssetGroup.from_dict(name, body)
            try:
                reg.add(g, overwrite=args.overwrite)
                added += 1
            except ValueError as exc:
                _print(f"[yellow]skip[/yellow] {name}: {exc}", console)
        _print(f"[green]Imported[/green] {added} group(s) from {args.path}.", console)
        return 0

    _print(f"[red]Unknown groups action:[/red] {action}", console)
    return 1


# ── CLI ────────────────────────────────────────────────────────────────────────

def _asset_group_help() -> str:
    """Generate --asset-group help text listing current registered groups."""
    try:
        from core.asset_groups import load_default_registry
        names = load_default_registry(reload=True).list()
    except Exception:
        names = []
    if not names:
        return "Asset group name from config/asset_groups.yaml"
    return f"Asset group from config ({' | '.join(names)})"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="regime-trader",
        description="HMM-based volatility regime allocation system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py trade\n"
            "  python main.py trade --dry-run\n"
            "  python main.py trade --train-only\n"
            "  python main.py trade --asset-group crypto\n"
            "  python main.py backtest --asset-group stocks --start 2020-01-01 --compare\n"
            "  python main.py backtest --asset-group crypto --start 2020-01-01 --compare\n"
            "  python main.py backtest --asset-group indices --start 2020-01-01 --compare\n"
            "  python main.py backtest --symbols SPY,QQQ --start 2020-01-01\n"
            "  python main.py stress   --asset-group indices --start 2019-01-01\n"
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── trade ─────────────────────────────────────────────────────────────────
    trade_p = sub.add_parser("trade", help="Run live or paper trading loop")
    trade_p.add_argument("--config",       default="config/settings.yaml")
    trade_p.add_argument("--paper",        action="store_true", default=None,
                          help="Force paper-trading mode")
    trade_p.add_argument("--dry-run",      action="store_true", dest="dry_run",
                          help="Full pipeline but place no real orders")
    trade_p.add_argument("--train-only",   action="store_true", dest="train_only",
                          help="Train HMM on latest data and exit")
    trade_p.add_argument("--asset-group",  default=None, dest="asset_group",
                          help=_asset_group_help())
    trade_p.add_argument("--symbols",      default=None,
                          help="Comma-separated symbols — overrides asset-group")
    trade_p.add_argument("--set",               default=None, dest="config_set",
                          help="Config set to apply (conservative | balanced | aggressive)")
    trade_p.add_argument("--strategies",        default=None, dest="strategy_filter",
                          help="Comma-separated strategy names from settings.yaml[strategies]")
    trade_p.add_argument("--allocator",         default="inverse_vol", dest="allocator_approach",
                          help="Capital allocation approach: equal_weight | inverse_vol | risk_parity | performance_weighted")
    trade_p.add_argument("--no-portfolio-risk", action="store_true", dest="no_portfolio_risk",
                          help="Disable portfolio-level PortfolioRiskManager (per-strategy RM still active)")
    trade_p.add_argument("--telegram",    action="store_true",  default=None, dest="telegram",
                          help="Force-enable Telegram notifications")
    trade_p.add_argument("--no-telegram", action="store_false", dest="telegram",
                          help="Force-disable Telegram notifications")

    # ── backtest ──────────────────────────────────────────────────────────────
    bt_p = sub.add_parser("backtest", help="Run walk-forward backtest")
    bt_p.add_argument("--config",      default="config/settings.yaml")
    bt_p.add_argument("--output",      default="results/",
                       help="Directory for output CSVs (default: results/)")
    bt_p.add_argument("--asset-group", default=None, dest="asset_group",
                       help=_asset_group_help())
    bt_p.add_argument("--symbols",     default=None,
                       help="Comma-separated symbols — overrides asset-group")
    bt_p.add_argument("--start",       default=None,
                       help="Start date ISO-8601 (e.g. 2019-01-01)")
    bt_p.add_argument("--end",         default=None,
                       help="End date ISO-8601 (default: today)")
    bt_p.add_argument("--compare",        action="store_true",
                       help="Add benchmark comparison table")
    bt_p.add_argument("--enforce-stops", action="store_true", dest="enforce_stops",
                       default=False,
                       help="Enable ATR per-trade stops in backtest (diagnostic A/B; default off — live uses HMM regime + portfolio kill, no broker-side stops)")
    bt_p.add_argument("--stress-test",  action="store_true", dest="stress_test",
                       help="Run stress scenarios after the backtest")
    bt_p.add_argument("--telegram",    action="store_true",  default=None, dest="telegram",
                       help="Force-enable Telegram notifications for this run")
    bt_p.add_argument("--no-telegram", action="store_false", dest="telegram",
                       help="Force-disable Telegram notifications for this run")
    bt_p.add_argument("--set",          default=None, dest="config_set",
                       help="Config set to apply (conservative | balanced | aggressive)")
    bt_p.add_argument("--multi-strat",  action="store_true", dest="multi_strat",
                       help="Use MultiStrategyBacktester with strategies from settings.yaml[strategies]")
    bt_p.add_argument("--allocator",    default="inverse_vol", dest="allocator_approach",
                       help="Allocator approach for --multi-strat mode")

    # ── stress ────────────────────────────────────────────────────────────────
    stress_p = sub.add_parser("stress", help="Run stress-test scenario suite")
    stress_p.add_argument("--config",      default="config/settings.yaml")
    stress_p.add_argument("--output",      default="results/")
    stress_p.add_argument("--asset-group", default=None, dest="asset_group",
                           help=_asset_group_help())
    stress_p.add_argument("--symbols",     default=None,
                           help="Comma-separated symbols — overrides asset-group")
    stress_p.add_argument("--start",       default=None)
    stress_p.add_argument("--end",         default=None)
    stress_p.add_argument("--set",         default=None, dest="config_set",
                           help="Config set to apply")
    stress_p.add_argument("--telegram",    action="store_true",  default=None, dest="telegram",
                           help="Force-enable Telegram notifications")
    stress_p.add_argument("--no-telegram", action="store_false", dest="telegram",
                           help="Force-disable Telegram notifications")

    # ── full-cycle ────────────────────────────────────────────────────────────
    full_p = sub.add_parser("full-cycle", help="Run backtest for all 3 asset groups (stocks, crypto, indices)")
    full_p.add_argument("--config",    default="config/settings.yaml")
    full_p.add_argument("--start",     default=None)
    full_p.add_argument("--end",       default=None)
    full_p.add_argument("--output",    default="results/")
    full_p.add_argument("--set",       default=None, dest="config_set",
                         help="Config set to apply")

    # ── cs-sweep ──────────────────────────────────────────────────────────────
    cs_p = sub.add_parser(
        "cs-sweep",
        help="2-D grid sweep: min_confidence × stability_bars (trains once per fold)",
    )
    cs_p.add_argument("--config",      default="config/settings.yaml")
    cs_p.add_argument("--asset-group", default=None, dest="asset_group",
                       help=_asset_group_help())
    cs_p.add_argument("--symbols",     default=None)
    cs_p.add_argument("--start",       default=None)
    cs_p.add_argument("--end",         default=None)
    cs_p.add_argument("--conf",        default="0.55,0.60,0.65,0.70,0.75",
                       dest="conf_values",
                       help="Comma-separated min_confidence values (default: 0.55,0.60,0.65,0.70,0.75)")
    cs_p.add_argument("--stab",        default="3,5,7,9,12",
                       dest="stab_values",
                       help="Comma-separated stability_bars values (default: 3,5,7,9,12)")
    cs_p.add_argument("--set",         default=None, dest="config_set")

    # ── groups ────────────────────────────────────────────────────────────────
    groups_p = sub.add_parser(
        "groups",
        help="Manage asset groups (config/asset_groups.yaml)",
        description="List, inspect, create, edit and remove asset groups.",
    )
    groups_sub = groups_p.add_subparsers(dest="groups_action", required=True)

    g_list = groups_sub.add_parser("list", help="List all asset groups")
    g_list.add_argument("--names-only", action="store_true",
                        help="Emit bare names, one per line (for scripts)")
    g_list.add_argument("--json", action="store_true", help="Emit JSON")

    g_show = groups_sub.add_parser("show", help="Show one group in detail")
    g_show.add_argument("name")
    g_show.add_argument("--json", action="store_true")

    g_add = groups_sub.add_parser("add", help="Create a new group")
    g_add.add_argument("name")
    g_add.add_argument("--symbols", required=True, help="Comma-separated list")
    g_add.add_argument("--description", default="")
    g_add.add_argument("--asset-class", default="", dest="asset_class")
    g_add.add_argument("--tags", default="", help="Comma-separated tags")
    g_add.add_argument("--overwrite", action="store_true")

    g_rm = groups_sub.add_parser("remove", help="Delete a group")
    g_rm.add_argument("name")

    g_edit = groups_sub.add_parser("edit", help="Edit an existing group")
    g_edit.add_argument("name")
    g_edit.add_argument("--symbols", default=None, help="Replace full symbol list")
    g_edit.add_argument("--add-symbols", default=None, dest="add_symbols")
    g_edit.add_argument("--remove-symbols", default=None, dest="remove_symbols")
    g_edit.add_argument("--description", default=None)
    g_edit.add_argument("--asset-class", default=None, dest="asset_class")
    g_edit.add_argument("--tags", default=None, help="Replace tags (comma-separated)")

    g_rn = groups_sub.add_parser("rename", help="Rename a group")
    g_rn.add_argument("old")
    g_rn.add_argument("new")

    g_def = groups_sub.add_parser("set-default", help="Set the default group")
    g_def.add_argument("name")

    g_defp = groups_sub.add_parser("default", help="Print the default group name")

    g_val = groups_sub.add_parser("validate", help="Validate the groups file")

    g_exp = groups_sub.add_parser("export", help="Export groups as JSON")
    g_exp.add_argument("--out", default=None, help="Write to file instead of stdout")

    g_imp = groups_sub.add_parser("import", help="Import groups from a JSON file")
    g_imp.add_argument("path")
    g_imp.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing groups with same name")

    # ── sweep ─────────────────────────────────────────────────────────────────
    sweep_p = sub.add_parser(
        "sweep",
        help="Sweep min_rebalance_interval — fetch data once, compare all values",
    )
    sweep_p.add_argument("--config",      default="config/settings.yaml")
    sweep_p.add_argument("--asset-group", default=None, dest="asset_group",
                          help=_asset_group_help())
    sweep_p.add_argument("--symbols",     default=None,
                          help="Comma-separated symbols — overrides asset-group")
    sweep_p.add_argument("--start",       default=None,
                          help="Start date ISO-8601 (e.g. 2019-01-01)")
    sweep_p.add_argument("--end",         default=None,
                          help="End date ISO-8601 (default: today)")
    sweep_p.add_argument("--values",      default="0,1,2,3,5,7,10,15,20",
                          help="Comma-separated interval values to test (default: 0,1,2,3,5,7,10,15,20)")
    sweep_p.add_argument("--set",         default=None, dest="config_set",
                          help="Config set to apply (conservative | balanced | aggressive)")

    return parser


def main() -> None:
    """Parse CLI arguments, load config / credentials, and dispatch."""
    load_dotenv()

    parser = build_parser()
    args   = parser.parse_args()

    # `groups` is a pure management command — does not need trading config
    # or Alpaca credentials.
    if args.command == "groups":
        sys.exit(run_groups(args))

    config = load_config(args.config, set_name=getattr(args, "config_set", None))
    load_credentials()

    # Configure Telegram notifier — CLI flag overrides settings.yaml
    try:
        from telegram.notifier import configure as _tg_configure
        _tg_configure(enabled=getattr(args, "telegram", None))
    except Exception:
        pass

    if args.command == "trade":
        if getattr(args, "paper", None):
            config["broker"]["paper_trading"] = True

        if getattr(args, "train_only", False):
            run_train_only(config, args)
            return

        _strat_arg = getattr(args, "strategy_filter", None)
        _strat_filter = [s.strip() for s in _strat_arg.split(",")] if _strat_arg else None
        run_trading(
            config,
            dry_run             = getattr(args, "dry_run", False),
            allocator_approach  = getattr(args, "allocator_approach", "inverse_vol"),
            no_portfolio_risk   = getattr(args, "no_portfolio_risk", False),
            multi_strat_filter  = _strat_filter,
        )

    elif args.command == "backtest":
        run_backtest(config, args)

    elif args.command == "stress":
        run_stress_test(config, args)

    elif args.command == "full-cycle":
        run_full_cycle(config, args)

    elif args.command == "sweep":
        run_interval_sweep(config, args)

    elif args.command == "cs-sweep":
        run_cs_sweep(config, args)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
