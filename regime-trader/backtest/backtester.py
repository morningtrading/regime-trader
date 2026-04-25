"""
backtester.py — Walk-forward allocation backtester.

Implements a strict walk-forward methodology: for each fold the HMM is trained
exclusively on the in-sample window and evaluated on the subsequent out-of-sample
window.  No data from the test period is ever used during training.

ALLOCATION MATH
---------------
    equity        = cash + sum(shares[s] * price[s])
    target_shares = equity * target_weight / price          # fractional
    delta         = target_shares - current_shares
    fill_price    = price * (1 + sign(delta) * slippage_pct)
    cash         -= delta * fill_price
    shares        = target_shares

Leverage > 1.0 drives cash negative (margin); equity is still correct.

FILL DELAY
----------
Signals generated at the close of bar t are executed at the close of bar t+1
using bar t+1 prices (+ slippage).  Target weights are stored at signal time;
target *shares* are computed at execution time using the pre-trade equity.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.hmm_engine import HMMEngine
from core.regime_strategies import StrategyOrchestrator
from data.feature_blending import blend_cross_symbol_features
from data.feature_engineering import FeatureEngineer, hmm_feature_names as _hmm_feature_names


def _maybe_fetch_vix_bt(hmm_cfg: Dict, bars_index) -> "pd.Series | None":
    if not (hmm_cfg.get("use_vix_features", False)
            or hmm_cfg.get("use_credit_spread_features", False)):
        return None
    try:
        from data.vix_fetcher import fetch_vix_series
        if bars_index is None or len(bars_index) == 0:
            return None
        start = (pd.Timestamp(bars_index[0]) - pd.Timedelta(days=5)).date().isoformat()
        end   = (pd.Timestamp(bars_index[-1]) + pd.Timedelta(days=1)).date().isoformat()
        return fetch_vix_series(start=start, end=end, timeframe="1Day")
    except Exception:
        return None


def _maybe_fetch_credit_bt(hmm_cfg: Dict, bars_index) -> "pd.Series | None":
    if not hmm_cfg.get("use_credit_spread_features", False):
        return None
    try:
        from data.credit_spread_fetcher import fetch_credit_spread_series
        if bars_index is None or len(bars_index) == 0:
            return None
        start = (pd.Timestamp(bars_index[0]) - pd.Timedelta(days=120)).date().isoformat()
        end   = (pd.Timestamp(bars_index[-1]) + pd.Timedelta(days=1)).date().isoformat()
        return fetch_credit_spread_series(start=start, end=end, timeframe="1Day")
    except Exception:
        return None

logger = logging.getLogger(__name__)

# Regimes where stop-losses are NOT enforced even when enforce_stops=True.
# In trending (bullish) regimes, ATR stops fire on normal pullbacks and cut
# off return without protecting against the drawdowns they are designed to avoid.
_STOP_EXEMPT_REGIMES = frozenset({
    "NEUTRAL", "BULL", "WEAK_BULL", "STRONG_BULL", "EUPHORIA",
})

# ── Default configurations ─────────────────────────────────────────────────────

_DEFAULT_HMM_CFG: Dict = dict(
    n_candidates=[3, 4, 5],
    n_init=5,
    min_train_bars=120,
    stability_bars=3,
    flicker_window=20,
    flicker_threshold=4,
    min_confidence=0.55,
)

_DEFAULT_STRAT_CFG: Dict = {
    "strategy": {
        "low_vol_allocation": 0.95,
        "mid_vol_allocation_trend": 0.95,
        "mid_vol_allocation_no_trend": 0.60,
        "high_vol_allocation": 0.60,
        "low_vol_leverage": 1.25,
        "uncertainty_size_mult": 0.50,
        "rebalance_threshold": 0.18,
    }
}


# ── OHLCV helper ───────────────────────────────────────────────────────────────

def _ohlcv_from_close(close: pd.Series) -> pd.DataFrame:
    """
    Synthesise a plausible OHLCV DataFrame from a close-price series.

    High/low are approximated via a 20-bar EWM of the absolute daily return
    so ATR and ADX are non-trivial.  Volume is generated with log-normal
    noise correlated with absolute daily returns to avoid zero variance
    (which would make the volume z-score features produce NaN throughout).
    """
    pct = close.pct_change().abs().fillna(0.001)
    avg_range_pct = pct.ewm(span=20, adjust=False).mean().fillna(0.01)
    half_range = close * avg_range_pct * 0.5
    low_floor = close * 0.001          # prevent negative lows

    # Volume: 1M baseline scaled by log-normal noise + absolute-return spike
    # Use a deterministic seed derived from the price series so results are
    # reproducible while still giving non-constant volume.
    seed = int(abs(close.sum()) * 1000) % (2 ** 31)
    rng = np.random.default_rng(seed)
    vol_noise = rng.lognormal(mean=0.0, sigma=0.4, size=len(close))
    volume = pd.Series(
        1_000_000.0 * vol_noise * (1.0 + pct.values * 20.0),
        index=close.index,
    )

    return pd.DataFrame(
        {
            "open":   close.shift(1).bfill(),
            "high":   close + half_range,
            "low":    (close - half_range).clip(lower=low_floor),
            "close":  close,
            "volume": volume,
        }
    )


# ── P&L attribution helper ────────────────────────────────────────────────────

def _compute_regime_pnl(
    returns_dict: Dict,
    equity_curve: Dict,
    regime_dict: Dict,
) -> Dict[str, float]:
    """Sum dollar P&L (equity × daily_return) per regime label for one fold."""
    regime_pnl: Dict[str, float] = {}
    for ts, ret in returns_dict.items():
        label = regime_dict.get(ts, "UNKNOWN")
        eq = equity_curve.get(ts, 0.0)
        regime_pnl[label] = regime_pnl.get(label, 0.0) + eq * ret
    return regime_pnl


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class WindowResult:
    """Results for a single walk-forward fold."""
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    equity_curve: pd.Series        # portfolio value indexed by date
    returns: pd.Series             # daily returns
    regime_series: pd.Series       # regime label per bar
    trades: List[Dict]             # list of trade records
    n_hmm_states: int              # states selected by BIC
    regime_pnl: Dict[str, float] = field(default_factory=dict)  # label → dollar P&L


@dataclass
class BacktestResult:
    """Aggregated results across all walk-forward folds."""
    windows: List[WindowResult]
    combined_equity: pd.Series     # spliced equity curve across all folds
    combined_returns: pd.Series    # spliced returns
    combined_regimes: pd.Series    # regime labels stitched together
    initial_capital: float
    final_equity: float
    metadata: Dict = field(default_factory=dict)
    combined_regime_pnl: Dict[str, float] = field(default_factory=dict)  # label → dollar P&L across all folds


# ── Main class ─────────────────────────────────────────────────────────────────

class WalkForwardBacktester:
    """
    Walk-forward allocation backtester.

    Splits a long price history into rolling train/test folds, trains a fresh
    HMM on each training window, and simulates bar-by-bar trading on the
    corresponding test window.  Fold results are stitched into a continuous
    out-of-sample equity curve.

    Features are pre-computed on the full price history (rolling windows are
    causal, so this is equivalent to computing them incrementally) and then
    clean rows are sliced per fold.

    Parameters
    ----------
    symbols :
        Trading universe (must all be columns in the ``prices`` DataFrame).
    initial_capital :
        Starting portfolio equity in USD.
    train_window :
        Number of *clean* feature rows used for HMM training per fold (~252 = 1 year).
    test_window :
        Out-of-sample evaluation window in bars (~126 = 6 months).
    step_size :
        Bars to advance the IS window between consecutive folds.
    slippage_pct :
        One-way slippage fraction applied on every executed trade.
    risk_free_rate :
        Annualised risk-free rate for Sharpe calculation.
    fill_delay :
        Bars between signal generation and execution (default 1).
    zscore_window :
        Rolling window for feature z-score standardisation (matches live
        default of 252 so backtest and live features are on the same distribution).
    """

    def __init__(
        self,
        symbols: List[str],
        initial_capital: float = 100_000.0,
        train_window: int = 252,
        test_window: int = 126,
        step_size: int = 126,
        slippage_pct: float = 0.0005,
        risk_free_rate: float = 0.045,
        fill_delay: int = 1,
        zscore_window: int = 252,  # aligned with live FeatureEngineer default
        sma_long: int = 200,
        sma_trend: int = 50,
        volume_norm_window: int = 50,
        min_rebalance_interval: int = 0,
    ) -> None:
        self.symbols = symbols
        self.initial_capital = initial_capital
        self.train_window = train_window
        self.test_window = test_window
        self.step_size = step_size
        self.slippage_pct = slippage_pct
        self.risk_free_rate = risk_free_rate
        self.fill_delay = fill_delay
        self.zscore_window = zscore_window
        self.sma_long = sma_long
        self.sma_trend = sma_trend
        self.volume_norm_window = volume_norm_window
        self.min_rebalance_interval = min_rebalance_interval

        self._results: Optional[BacktestResult] = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(
        self,
        prices: pd.DataFrame,
        hmm_config: Optional[Dict] = None,
        strategy_config: Optional[Dict] = None,
        risk_config: Optional[Dict] = None,
        progress_callback=None,
        enforce_stops: bool = False,
        dump_fold_models: Optional[Path] = None,
        load_fold_models: Optional[Path] = None,
    ) -> BacktestResult:
        """
        Execute the full walk-forward backtest.

        Parameters
        ----------
        prices :
            Wide-format close prices, shape (n_bars, n_symbols).
            Index must be a DatetimeIndex sorted ascending.
        hmm_config :
            Overrides for HMMEngine constructor kwargs.
        strategy_config :
            Override strategy section; merged into the default strategy config.
        risk_config :
            Reserved for future use (risk manager not yet implemented).

        Returns
        -------
        :class:`BacktestResult`
        """
        hmm_cfg = {**_DEFAULT_HMM_CFG, **(hmm_config or {})}
        strat_cfg = _DEFAULT_STRAT_CFG.copy()
        if strategy_config:
            strat_cfg["strategy"] = {
                **strat_cfg["strategy"],
                **strategy_config.get("strategy", strategy_config),
            }

        # ── Select universe present in prices ─────────────────────────────────
        syms = [s for s in self.symbols if s in prices.columns]
        if not syms:
            raise ValueError("None of the requested symbols are in the prices DataFrame.")

        # ── AUDIT: hash inputs to diff Win vs Linux ──────────────────────────
        import hashlib, platform, sys
        _ph = hashlib.md5(pd.util.hash_pandas_object(prices, index=True).values).hexdigest()
        logger.info("AUDIT PRICES_HASH=%s shape=%s cols=%s tz=%s first=%s last=%s",
                    _ph, prices.shape, list(prices.columns),
                    prices.index.tz, prices.index[0], prices.index[-1])
        logger.info("AUDIT PLATFORM=%s py=%s tz_local=%s",
                    platform.platform(), sys.version.split()[0],
                    __import__("time").tzname)

        # ── Build synthetic OHLCV for every symbol ────────────────────────────
        ohlcv: Dict[str, pd.DataFrame] = {s: _ohlcv_from_close(prices[s]) for s in syms}

        # ── Compute features on the full price history (market symbol = first) ─
        _proxy = hmm_cfg.get("regime_proxy") or None
        market_sym = (_proxy if _proxy and _proxy in ohlcv else None) or syms[0]
        fe = FeatureEngineer(
            zscore_window=self.zscore_window,
            sma_long=self.sma_long,
            sma_trend=self.sma_trend,
            volume_norm_window=self.volume_norm_window,
        )
        _vix_bt = _maybe_fetch_vix_bt(hmm_cfg, ohlcv[market_sym].index)
        _credit_bt = _maybe_fetch_credit_bt(hmm_cfg, ohlcv[market_sym].index)
        full_features_raw = fe.build_feature_matrix(
            ohlcv[market_sym],
            feature_names=_hmm_feature_names(hmm_cfg),
            dropna=False,
            vix_series=_vix_bt,
            credit_series=_credit_bt,
        )

        # Blend log_ret_1 and realized_vol_20 across equity-like symbols so the
        # HMM sees a basket-level return/vol signal rather than a single proxy.
        # vol_ratio, adx_14, dist_sma200 remain market_sym-anchored (trend info).
        # Non-equity assets (GLD, TLT, USO …) are excluded via hmm_cfg.blend_exclude.
        # If regime_proxy is set, skip blending entirely (proxy already is the sole source).
        _blend_exclude = (
            [s for s in syms if s != market_sym]   # exclude everything but proxy → no blend
            if _proxy else
            hmm_cfg.get("blend_exclude", [])
        )
        full_features_raw = blend_cross_symbol_features(
            full_features_raw,
            {s: ohlcv[s] for s in syms if s in ohlcv},
            feature_engineer=fe,
            blend_exclude=_blend_exclude,
            min_bars=0,
        )

        # ── Identify clean rows (all features non-NaN) ────────────────────────
        clean_mask = full_features_raw.notna().all(axis=1)
        clean_features = full_features_raw[clean_mask]
        n_clean = len(clean_features)

        logger.info(
            "Price history: %d bars  |  warmup consumed: %d bars  |  clean features: %d bars",
            len(prices), len(prices) - n_clean, n_clean,
        )

        # ── Generate walk-forward window indices ──────────────────────────────
        windows_idx = self._generate_windows(n_clean)
        if not windows_idx:
            raise ValueError(
                f"Insufficient data for walk-forward backtest. "
                f"Need at least {self.train_window + self.test_window} clean feature bars; "
                f"got {n_clean}. Provide more price history or reduce train_window / test_window."
            )

        logger.info(
            "Walk-forward: %d folds  |  IS=%d bars  |  OOS=%d bars  |  step=%d bars",
            len(windows_idx), self.train_window, self.test_window, self.step_size,
        )

        # ── Run each fold ─────────────────────────────────────────────────────
        equity = self.initial_capital
        fold_results: List[WindowResult] = []

        # AUDIT: hash full feature matrix
        import hashlib as _hl
        _fh = _hl.md5(pd.util.hash_pandas_object(clean_features).values).hexdigest()
        logger.info("AUDIT CLEAN_FEATURES_HASH=%s shape=%s cols=%s",
                    _fh, clean_features.shape, list(clean_features.columns))

        for fold_id, (is_s, is_e, oos_s, oos_e) in enumerate(windows_idx):
            is_features = clean_features.iloc[is_s:is_e]
            oos_features = clean_features.iloc[oos_s:oos_e]
            _ish = _hl.md5(pd.util.hash_pandas_object(is_features).values).hexdigest()
            _osh = _hl.md5(pd.util.hash_pandas_object(oos_features).values).hexdigest()
            logger.info("AUDIT FOLD_%d IS_HASH=%s OOS_HASH=%s", fold_id, _ish, _osh)

            min_train = hmm_cfg.get("min_train_bars", 120)
            if len(is_features) < min_train:
                logger.warning("Fold %d: only %d IS rows (need %d) — skipping.",
                               fold_id, len(is_features), min_train)
                continue

            if len(oos_features) == 0:
                logger.warning("Fold %d: empty OOS window — skipping.", fold_id)
                continue

            logger.info(
                "Fold %d  IS=[%s, %s]  OOS=[%s, %s]  start_equity=$%.0f",
                fold_id,
                is_features.index[0].date(), is_features.index[-1].date(),
                oos_features.index[0].date(), oos_features.index[-1].date(),
                equity,
            )

            if progress_callback is not None:
                progress_callback(fold_id, len(windows_idx), "training", {
                    "symbols":   syms,
                    "is_start":  is_features.index[0].date(),
                    "is_end":    is_features.index[-1].date(),
                    "oos_start": oos_features.index[0].date(),
                    "oos_end":   oos_features.index[-1].date(),
                    "equity":    equity,
                })

            try:
                wr = self._run_single_window(
                    fold_id, prices, ohlcv, is_features, oos_features,
                    equity, hmm_cfg, strat_cfg,
                    enforce_stops=enforce_stops,
                    dump_fold_models=dump_fold_models,
                    load_fold_models=load_fold_models,
                )
            except Exception as exc:
                logger.error("Fold %d failed: %s — skipping.", fold_id, exc, exc_info=True)
                continue

            fold_results.append(wr)
            if len(wr.equity_curve) > 0:
                equity = float(wr.equity_curve.iloc[-1])

            if progress_callback is not None:
                progress_callback(fold_id, len(windows_idx), "complete", {
                    "symbols":     syms,
                    "oos_start":   wr.test_start.date(),
                    "oos_end":     wr.test_end.date(),
                    "fold_trades": len(wr.trades),
                    "n_states":    wr.n_hmm_states,
                    "equity":      equity,
                })

        if not fold_results:
            raise RuntimeError("All folds failed — no BacktestResult produced.")

        # ── Stitch equity curves ──────────────────────────────────────────────
        combined_equity = self._stitch_equity_curves(fold_results)
        combined_returns = pd.concat(
            [w.returns for w in fold_results if len(w.returns) > 0]
        ).sort_index()
        combined_regimes = pd.concat(
            [w.regime_series for w in fold_results if len(w.regime_series) > 0]
        ).sort_index()
        all_trades = [t for w in fold_results for t in w.trades]

        combined_regime_pnl: Dict[str, float] = {}
        for wr in fold_results:
            for label, pnl in wr.regime_pnl.items():
                combined_regime_pnl[label] = combined_regime_pnl.get(label, 0.0) + pnl

        result = BacktestResult(
            windows=fold_results,
            combined_equity=combined_equity,
            combined_returns=combined_returns,
            combined_regimes=combined_regimes,
            initial_capital=self.initial_capital,
            final_equity=float(combined_equity.iloc[-1]) if len(combined_equity) > 0
                         else self.initial_capital,
            metadata={
                "n_folds": len(fold_results),
                "symbols": syms,
                "total_trades": len(all_trades),
                "train_window": self.train_window,
                "test_window": self.test_window,
                "step_size": self.step_size,
                "slippage_pct": self.slippage_pct,
                "hmm_config": hmm_cfg,
            },
            combined_regime_pnl=combined_regime_pnl,
        )
        self._results = result
        return result

    def get_results(self) -> Optional[BacktestResult]:
        """Return the most recent backtest results, or None if not yet run."""
        return self._results

    # ── Window generation ──────────────────────────────────────────────────────

    def _generate_windows(
        self, n_clean: int
    ) -> List[Tuple[int, int, int, int]]:
        """
        Generate ``(is_start, is_end, oos_start, oos_end)`` index tuples for
        all walk-forward folds.

        Windows are rolling: each fold's IS window is exactly ``train_window``
        clean bars; OOS windows are non-overlapping.

        Parameters
        ----------
        n_clean :
            Number of clean (non-NaN) feature rows in the full history.

        Returns
        -------
        List of (is_start, is_end, oos_start, oos_end) row indices into
        the clean feature DataFrame.
        """
        windows = []
        cursor = 0
        while cursor + self.train_window + self.test_window <= n_clean:
            is_start = cursor
            is_end = cursor + self.train_window
            oos_start = is_end
            oos_end = min(oos_start + self.test_window, n_clean)
            windows.append((is_start, is_end, oos_start, oos_end))
            cursor += self.step_size
        return windows

    # ── Single-fold simulation ─────────────────────────────────────────────────

    def _run_single_window(
        self,
        fold_id: int,
        prices: pd.DataFrame,
        ohlcv_by_symbol: Dict[str, pd.DataFrame],
        is_features: pd.DataFrame,
        oos_features: pd.DataFrame,
        start_equity: float,
        hmm_cfg: Dict,
        strat_cfg: Dict,
        enforce_stops: bool = False,
        dump_fold_models: Optional[Path] = None,
        load_fold_models: Optional[Path] = None,
    ) -> WindowResult:
        """
        Execute one walk-forward fold.

        1. Trains a fresh HMMEngine on ``is_features``.
        2. Warms up the incremental forward pass on IS bars (so the alpha
           cache is initialised at OOS start — no cold restart).
        3. Simulates bar-by-bar trading on ``oos_features`` with fill delay.

        Returns
        -------
        :class:`WindowResult`
        """
        # ── 1. Train HMM ──────────────────────────────────────────────────────
        _engine_kwargs = {
            "n_candidates", "n_init", "covariance_type", "min_train_bars",
            "stability_bars", "flicker_window", "flicker_threshold",
            "min_confidence", "min_covar",
        }
        if load_fold_models is not None:
            import pickle
            fpath = load_fold_models / f"fold_{fold_id:02d}.pkl"
            with open(fpath, "rb") as fh:
                engine = pickle.load(fh)
            logger.info("Fold %d: HMM LOADED from %s  n_states=%d",
                        fold_id, fpath, engine._n_states)
        else:
            engine = HMMEngine(**{k: v for k, v in hmm_cfg.items() if k in _engine_kwargs})
            engine.fit(is_features.values)
            logger.info("Fold %d: HMM fitted  n_states=%d  BIC=%.2f",
                        fold_id, engine._n_states, engine._training_bic)
            if dump_fold_models is not None:
                import pickle
                dump_fold_models.mkdir(parents=True, exist_ok=True)
                fpath = dump_fold_models / f"fold_{fold_id:02d}.pkl"
                with open(fpath, "wb") as fh:
                    pickle.dump(engine, fh)
                logger.info("Fold %d: HMM dumped to %s", fold_id, fpath)

        # ── 2. Build orchestrator from fitted regime_infos ─────────────────
        regime_infos = engine.get_all_regime_info()
        rebalance_thr = strat_cfg.get("strategy", {}).get("rebalance_threshold", 0.10)
        min_conf = hmm_cfg.get("min_confidence", 0.55)
        orch = StrategyOrchestrator(
            config=strat_cfg,
            regime_infos=regime_infos,
            min_confidence=min_conf,
            rebalance_threshold=rebalance_thr,
        )

        # ── 3. Warm up the forward pass on IS bars ────────────────────────────
        for t in range(len(is_features)):
            engine.update(is_features.values[t])

        # ── 4. Initialise portfolio state ─────────────────────────────────────
        cash: float = start_equity
        shares: Dict[str, float] = {s: 0.0 for s in self.symbols}

        equity_curve: Dict[pd.Timestamp, float] = {}
        returns_dict: Dict[pd.Timestamp, float] = {}
        regime_dict: Dict[pd.Timestamp, str] = {}
        trades: List[Dict] = []

        prev_equity: float = start_equity
        # pending stores target weights (not shares) for 1-bar fill delay
        pending: Optional[Dict] = None
        bars_since_rebalance: int = self.min_rebalance_interval  # allow first rebalance immediately
        # stop_prices tracks the active stop level per symbol (enforce_stops only)
        stop_prices: Dict[str, float] = {}
        # current_regime from previous bar — used for regime-conditional stop gate
        current_regime: str = ""

        oos_timestamps = list(oos_features.index)

        # ── 5. OOS bar-by-bar simulation ──────────────────────────────────────
        for t, bar_date in enumerate(oos_timestamps):
            if bar_date not in prices.index:
                continue

            # Current close prices for all symbols
            cur_px: Dict[str, float] = {}
            for s in self.symbols:
                if s in prices.columns:
                    val = prices.loc[bar_date, s]
                    if pd.notna(val) and val > 0:
                        cur_px[s] = float(val)

            # ── 5-stop. Stop-loss check (before pending, enforce_stops only) ──
            # Stops are suppressed in bullish/neutral regimes where ATR pullbacks
            # are normal and stops would cut off return without reducing tail risk.
            if enforce_stops and current_regime not in _STOP_EXEMPT_REGIMES:
                for sym in list(stop_prices.keys()):
                    price = cur_px.get(sym)
                    if price is None or shares.get(sym, 0.0) <= 0:
                        stop_prices.pop(sym, None)
                        continue
                    if price <= stop_prices[sym]:
                        fill_px = price * (1.0 - self.slippage_pct)
                        proceeds = shares[sym] * fill_px
                        cash += proceeds
                        logger.info(
                            "STOP_OUT fold=%d %s @ %.2f (stop=%.2f)",
                            fold_id, sym, price, stop_prices[sym],
                        )
                        trades.append({
                            "fold": fold_id,
                            "timestamp": bar_date,
                            "symbol": sym,
                            "action": "STOP_OUT",
                            "delta_shares": -shares[sym],
                            "fill_price": fill_px,
                            "trade_value": proceeds,
                            "slippage_cost": shares[sym] * price * self.slippage_pct,
                            "regime": regime_dict.get(bar_date, ""),
                            "regime_prob": 0.0,
                            "target_weight": 0.0,
                        })
                        shares[sym] = 0.0
                        stop_prices.pop(sym)

            # ── 5a. Execute pending rebalance (fill-delay = 1 bar) ────────────
            if pending is not None:
                equity_pre = cash + sum(
                    shares[s] * cur_px.get(s, 0.0) for s in self.symbols
                )
                for sym, target_wt in pending["weights"].items():
                    price = cur_px.get(sym)
                    if not price:
                        continue
                    target_shr = equity_pre * target_wt / price
                    delta = target_shr - shares.get(sym, 0.0)
                    if abs(delta * price) < 1.0:    # ignore sub-$1 moves
                        continue
                    sign = 1.0 if delta > 0 else -1.0
                    fill_px = price * (1.0 + sign * self.slippage_pct)
                    cash -= delta * fill_px
                    shares[sym] = target_shr
                    trades.append({
                        "fold": fold_id,
                        "timestamp": bar_date,
                        "symbol": sym,
                        "delta_shares": delta,
                        "fill_price": fill_px,
                        "trade_value": abs(delta * fill_px),
                        "slippage_cost": abs(delta * fill_px * self.slippage_pct),
                        "regime": pending["regime"],
                        "regime_prob": pending["regime_prob"],
                        "target_weight": target_wt,
                    })

                # Transfer stop levels from executed signals into stop_prices
                if enforce_stops:
                    for sym, sl in pending.get("stop_losses", {}).items():
                        if shares.get(sym, 0.0) > 0:
                            stop_prices[sym] = sl

                # Sync orchestrator with post-execution actual weights
                equity_post = cash + sum(
                    shares[s] * cur_px.get(s, 0.0) for s in self.symbols
                )
                if equity_post > 0:
                    actual_wts = {
                        s: (shares[s] * cur_px.get(s, 0.0)) / equity_post
                        for s in self.symbols
                    }
                    orch.update_weights(actual_wts)
                pending = None
                bars_since_rebalance = 0

            else:
                bars_since_rebalance += 1

            # ── 5b. Mark to market ────────────────────────────────────────────
            equity = cash + sum(
                shares[s] * cur_px.get(s, 0.0) for s in self.symbols
            )
            equity_curve[bar_date] = equity
            returns_dict[bar_date] = (
                (equity / prev_equity - 1.0) if prev_equity > 0 else 0.0
            )
            prev_equity = equity

            # ── 5c. Update HMM (incremental forward pass) ─────────────────────
            regime_state = engine.update(oos_features.values[t], bar_date)
            is_flickering = engine.is_flickering()
            regime_dict[bar_date] = regime_state.label
            current_regime = regime_state.label   # used by stop gate on next bar

            # ── 5d. Build per-symbol OHLCV windows for signal generation ──────
            bars_for_signal: Dict[str, pd.DataFrame] = {}
            for s in self.symbols:
                if s not in ohlcv_by_symbol:
                    continue
                sym_ohlcv = ohlcv_by_symbol[s]
                try:
                    end_loc = sym_ohlcv.index.get_loc(bar_date) + 1
                except KeyError:
                    continue
                # Last 250 bars (enough for EMA50 / ATR14 warmup)
                bars_for_signal[s] = sym_ohlcv.iloc[max(0, end_loc - 250):end_loc]

            # ── 5e. Generate signals ──────────────────────────────────────────
            signals = orch.generate_signals(
                self.symbols, bars_for_signal, regime_state, is_flickering
            )

            # ── 5e2. Skip NEUTRAL regime if configured ──────────────────────
            skip_neutral = strat_cfg.get("strategy", {}).get("skip_neutral_regime", False)
            if skip_neutral and regime_state.label == "NEUTRAL":
                signals = []  # Clear all signals → go to cash

            # ── 5f. Queue target weights for execution at next bar ────────────
            if signals and bars_since_rebalance >= self.min_rebalance_interval:
                pending = {
                    "weights": {
                        sig.symbol: sig.position_size_pct * sig.leverage
                        for sig in signals
                    },
                    "stop_losses": {sig.symbol: sig.stop_loss for sig in signals},
                    "regime": regime_state.label,
                    "regime_prob": regime_state.probability,
                }

        return WindowResult(
            fold_id=fold_id,
            train_start=is_features.index[0],
            train_end=is_features.index[-1],
            test_start=oos_features.index[0],
            test_end=oos_features.index[-1],
            equity_curve=pd.Series(equity_curve, name="equity"),
            returns=pd.Series(returns_dict, name="returns"),
            regime_series=pd.Series(regime_dict, name="regime"),
            trades=trades,
            n_hmm_states=engine._n_states,
            regime_pnl=_compute_regime_pnl(returns_dict, equity_curve, regime_dict),
        )

    def _run_oos_sim(
        self,
        fold_id: int,
        prices: pd.DataFrame,
        ohlcv_by_symbol: Dict,
        is_features,
        oos_features,
        start_equity: float,
        fitted_engine: "HMMEngine",
        strat_cfg: Dict,
        min_conf: float,
        enforce_stops: bool = False,
    ) -> WindowResult:
        """
        Replay OOS simulation using a *pre-fitted* engine.

        ``fitted_engine`` must already have been trained via ``engine.fit()``.
        This method deep-copies it, sets ``min_confidence`` and
        ``stability_bars`` from the engine's current attribute values, warms
        the forward-pass cache over the IS bars, then runs the OOS sim.

        Used by :meth:`run_grid` to avoid retraining HMM weights for every
        grid cell.
        """
        from core.regime_strategies import StrategyOrchestrator

        engine = copy.deepcopy(fitted_engine)

        # Build orchestrator (cheap — no training involved)
        regime_infos = engine.get_all_regime_info()
        rebalance_thr = strat_cfg.get("strategy", {}).get("rebalance_threshold", 0.10)
        orch = StrategyOrchestrator(
            config=strat_cfg,
            regime_infos=regime_infos,
            min_confidence=min_conf,
            rebalance_threshold=rebalance_thr,
        )

        # Warm up the forward-pass cache on IS bars
        for t in range(len(is_features)):
            engine.update(is_features.values[t])

        # OOS simulation (identical logic to _run_single_window step 4-5)
        cash: float = start_equity
        shares: Dict[str, float] = {s: 0.0 for s in self.symbols}

        equity_curve: Dict = {}
        returns_dict: Dict = {}
        regime_dict:  Dict = {}
        trades: List[Dict] = []

        prev_equity: float = start_equity
        pending = None
        bars_since_rebalance: int = self.min_rebalance_interval
        stop_prices: Dict[str, float] = {}
        current_regime: str = ""   # regime from previous bar for stop gate

        oos_timestamps = list(oos_features.index)

        for t, bar_date in enumerate(oos_timestamps):
            if bar_date not in prices.index:
                continue

            cur_px: Dict[str, float] = {}
            for s in self.symbols:
                if s in prices.columns:
                    val = prices.loc[bar_date, s]
                    if pd.notna(val) and val > 0:
                        cur_px[s] = float(val)

            # Stop-loss check (before pending, enforce_stops only).
            # Suppressed in bullish/neutral regimes — see _STOP_EXEMPT_REGIMES.
            if enforce_stops and current_regime not in _STOP_EXEMPT_REGIMES:
                for sym in list(stop_prices.keys()):
                    price = cur_px.get(sym)
                    if price is None or shares.get(sym, 0.0) <= 0:
                        stop_prices.pop(sym, None)
                        continue
                    if price <= stop_prices[sym]:
                        fill_px = price * (1.0 - self.slippage_pct)
                        proceeds = shares[sym] * fill_px
                        cash += proceeds
                        logger.info(
                            "STOP_OUT fold=%d %s @ %.2f (stop=%.2f)",
                            fold_id, sym, price, stop_prices[sym],
                        )
                        trades.append({
                            "fold": fold_id,
                            "timestamp": bar_date,
                            "symbol": sym,
                            "action": "STOP_OUT",
                            "delta_shares": -shares[sym],
                            "fill_price": fill_px,
                            "trade_value": proceeds,
                            "slippage_cost": shares[sym] * price * self.slippage_pct,
                            "regime": regime_dict.get(bar_date, ""),
                            "regime_prob": 0.0,
                            "target_weight": 0.0,
                        })
                        shares[sym] = 0.0
                        stop_prices.pop(sym)

            # Execute pending (1-bar fill delay)
            if pending is not None:
                equity_pre = cash + sum(
                    shares[s] * cur_px.get(s, 0.0) for s in self.symbols
                )
                for sym, target_wt in pending["weights"].items():
                    price = cur_px.get(sym)
                    if not price:
                        continue
                    target_shr = equity_pre * target_wt / price
                    delta = target_shr - shares.get(sym, 0.0)
                    if abs(delta * price) < 1.0:
                        continue
                    sign = 1.0 if delta > 0 else -1.0
                    fill_px = price * (1.0 + sign * self.slippage_pct)
                    cash -= delta * fill_px
                    shares[sym] = target_shr
                    trades.append({
                        "fold": fold_id,
                        "timestamp": bar_date,
                        "symbol": sym,
                        "delta_shares": delta,
                        "fill_price": fill_px,
                        "trade_value": abs(delta * fill_px),
                        "slippage_cost": abs(delta * fill_px * self.slippage_pct),
                        "regime": pending["regime"],
                        "regime_prob": pending["regime_prob"],
                        "target_weight": target_wt,
                    })
                if enforce_stops:
                    for sym, sl in pending.get("stop_losses", {}).items():
                        if shares.get(sym, 0.0) > 0:
                            stop_prices[sym] = sl

                equity_post = cash + sum(
                    shares[s] * cur_px.get(s, 0.0) for s in self.symbols
                )
                if equity_post > 0:
                    actual_wts = {
                        s: (shares[s] * cur_px.get(s, 0.0)) / equity_post
                        for s in self.symbols
                    }
                    orch.update_weights(actual_wts)
                pending = None
                bars_since_rebalance = 0
            else:
                bars_since_rebalance += 1

            equity = cash + sum(shares[s] * cur_px.get(s, 0.0) for s in self.symbols)
            equity_curve[bar_date] = equity
            returns_dict[bar_date] = (equity / prev_equity - 1.0) if prev_equity > 0 else 0.0
            prev_equity = equity

            regime_state  = engine.update(oos_features.values[t], bar_date)
            is_flickering = engine.is_flickering()
            regime_dict[bar_date] = regime_state.label
            current_regime = regime_state.label   # used by stop gate on next bar

            bars_for_signal: Dict = {}
            for s in self.symbols:
                if s not in ohlcv_by_symbol:
                    continue
                sym_ohlcv = ohlcv_by_symbol[s]
                try:
                    end_loc = sym_ohlcv.index.get_loc(bar_date) + 1
                except KeyError:
                    continue
                bars_for_signal[s] = sym_ohlcv.iloc[max(0, end_loc - 250):end_loc]

            signals = orch.generate_signals(
                self.symbols, bars_for_signal, regime_state, is_flickering
            )
            if signals and bars_since_rebalance >= self.min_rebalance_interval:
                pending = {
                    "weights": {
                        sig.symbol: sig.position_size_pct * sig.leverage
                        for sig in signals
                    },
                    "stop_losses": {sig.symbol: sig.stop_loss for sig in signals},
                    "regime": regime_state.label,
                    "regime_prob": regime_state.probability,
                }

        return WindowResult(
            fold_id=fold_id,
            train_start=is_features.index[0],
            train_end=is_features.index[-1],
            test_start=oos_features.index[0],
            test_end=oos_features.index[-1],
            equity_curve=pd.Series(equity_curve, name="equity"),
            returns=pd.Series(returns_dict, name="returns"),
            regime_series=pd.Series(regime_dict, name="regime"),
            trades=trades,
            n_hmm_states=engine._n_states,
            regime_pnl=_compute_regime_pnl(returns_dict, equity_curve, regime_dict),
        )

    def run_grid(
        self,
        prices: pd.DataFrame,
        conf_values: List[float],
        stab_values: List[int],
        hmm_config: Optional[Dict] = None,
        strategy_config: Optional[Dict] = None,
        progress_callback=None,
        enforce_stops: bool = False,
    ) -> "List[Tuple[float, int, BacktestResult]]":
        """
        2-D grid sweep over ``min_confidence`` × ``stability_bars``.

        HMM models are trained **once per fold** (same deterministic seed →
        same weights regardless of stability/confidence).  The pre-trained
        engine is deep-copied and replayed for every grid cell, making the
        total work  ``n_folds × n_cells``  OOS sims instead of
        ``n_folds × n_cells``  full trainings.

        Returns
        -------
        List of ``(conf, stab, BacktestResult)`` tuples, one per grid cell.
        """
        hmm_cfg   = {**_DEFAULT_HMM_CFG, **(hmm_config or {})}
        strat_cfg = _DEFAULT_STRAT_CFG.copy()
        if strategy_config:
            strat_cfg["strategy"] = {
                **strat_cfg["strategy"],
                **strategy_config.get("strategy", strategy_config),
            }

        syms = [s for s in self.symbols if s in prices.columns]
        if not syms:
            raise ValueError("None of the requested symbols are in the prices DataFrame.")

        ohlcv: Dict = {s: _ohlcv_from_close(prices[s]) for s in syms}

        _proxy = hmm_cfg.get("regime_proxy") or None
        market_sym = (_proxy if _proxy and _proxy in ohlcv else None) or syms[0]
        fe = FeatureEngineer(
            zscore_window=self.zscore_window,
            sma_long=self.sma_long,
            sma_trend=self.sma_trend,
            volume_norm_window=self.volume_norm_window,
        )
        _vix_bt = _maybe_fetch_vix_bt(hmm_cfg, ohlcv[market_sym].index)
        _credit_bt = _maybe_fetch_credit_bt(hmm_cfg, ohlcv[market_sym].index)
        full_features_raw = fe.build_feature_matrix(
            ohlcv[market_sym],
            feature_names=_hmm_feature_names(hmm_cfg),
            dropna=False,
            vix_series=_vix_bt,
            credit_series=_credit_bt,
        )

        # Blend log_ret_1 and realized_vol_20 across all symbols so the HMM sees
        # a basket-level return/vol signal rather than a single proxy.
        # Non-equity assets excluded via hmm_cfg.blend_exclude (see run()).
        # If regime_proxy is set, skip blending entirely (proxy already is the sole source).
        _blend_exclude = (
            [s for s in syms if s != market_sym]   # exclude everything but proxy → no blend
            if _proxy else
            hmm_cfg.get("blend_exclude", [])
        )
        full_features_raw = blend_cross_symbol_features(
            full_features_raw,
            {s: ohlcv[s] for s in syms if s in ohlcv},
            feature_engineer=fe,
            blend_exclude=_blend_exclude,
            min_bars=0,
        )

        clean_mask     = full_features_raw.notna().all(axis=1)
        clean_features = full_features_raw[clean_mask]
        n_clean        = len(clean_features)

        windows_idx = self._generate_windows(n_clean)
        if not windows_idx:
            raise ValueError(
                f"Insufficient data for grid sweep: {n_clean} clean bars, "
                f"need >= {self.train_window + self.test_window}."
            )

        n_folds = len(windows_idx)
        n_cells = len(conf_values) * len(stab_values)

        # ── Phase 1: train one engine per fold ────────────────────────────────
        if progress_callback:
            progress_callback("train_start", n_folds, 0, n_cells)

        trained: List[Tuple] = []   # (fold_id, engine, is_features, oos_features, equity_start)
        equity = self.initial_capital

        for fold_id, (is_s, is_e, oos_s, oos_e) in enumerate(windows_idx):
            is_feat  = clean_features.iloc[is_s:is_e]
            oos_feat = clean_features.iloc[oos_s:oos_e]

            min_train = hmm_cfg.get("min_train_bars", 120)
            if len(is_feat) < min_train or len(oos_feat) == 0:
                continue

            if progress_callback:
                progress_callback("training", n_folds, fold_id, n_cells)

            _engine_kwargs = {
                "n_candidates", "n_init", "covariance_type", "min_train_bars",
                "stability_bars", "flicker_window", "flicker_threshold",
                "min_confidence", "min_covar",
            }
            engine = HMMEngine(**{k: v for k, v in hmm_cfg.items() if k in _engine_kwargs})
            engine.fit(is_feat.values)

            # Run fold ONCE with base params to get equity carry-over
            base_wr = self._run_oos_sim(
                fold_id, prices, ohlcv, is_feat, oos_feat,
                equity, engine, strat_cfg,
                min_conf=hmm_cfg.get("min_confidence", 0.55),
                enforce_stops=enforce_stops,
            )
            trained.append((fold_id, engine, is_feat, oos_feat, equity))
            if len(base_wr.equity_curve) > 0:
                equity = float(base_wr.equity_curve.iloc[-1])

        if not trained:
            raise RuntimeError("All folds skipped during grid-sweep training phase.")

        # ── Phase 2: replay grid ──────────────────────────────────────────────
        results: List[Tuple[float, int, BacktestResult]] = []
        cell_idx = 0

        for conf in conf_values:
            for stab in stab_values:
                if progress_callback:
                    progress_callback("sweep", n_folds, cell_idx, n_cells)
                cell_idx += 1

                fold_results: List[WindowResult] = []
                cell_equity = self.initial_capital

                for fold_id, engine, is_feat, oos_feat, _base_eq in trained:
                    # Deep-copy and override inference params only
                    eng = copy.deepcopy(engine)
                    eng.stability_bars  = stab
                    eng.min_confidence  = conf

                    try:
                        wr = self._run_oos_sim(
                            fold_id, prices, ohlcv, is_feat, oos_feat,
                            cell_equity, eng, strat_cfg, min_conf=conf,
                            enforce_stops=enforce_stops,
                        )
                    except Exception as exc:
                        logger.warning("Grid cell (conf=%.2f stab=%d) fold %d failed: %s",
                                       conf, stab, fold_id, exc)
                        continue

                    fold_results.append(wr)
                    if len(wr.equity_curve) > 0:
                        cell_equity = float(wr.equity_curve.iloc[-1])

                if not fold_results:
                    continue

                combined_equity  = self._stitch_equity_curves(fold_results)
                combined_returns = pd.concat(
                    [w.returns for w in fold_results if len(w.returns) > 0]
                ).sort_index()
                combined_regimes = pd.concat(
                    [w.regime_series for w in fold_results if len(w.regime_series) > 0]
                ).sort_index()
                all_trades = [t for w in fold_results for t in w.trades]

                cell_regime_pnl: Dict[str, float] = {}
                for wr in fold_results:
                    for label, pnl in wr.regime_pnl.items():
                        cell_regime_pnl[label] = cell_regime_pnl.get(label, 0.0) + pnl

                bt_result = BacktestResult(
                    windows=fold_results,
                    combined_equity=combined_equity,
                    combined_returns=combined_returns,
                    combined_regimes=combined_regimes,
                    initial_capital=self.initial_capital,
                    final_equity=float(combined_equity.iloc[-1]) if len(combined_equity) > 0
                                 else self.initial_capital,
                    metadata={
                        "n_folds":      len(fold_results),
                        "symbols":      syms,
                        "total_trades": len(all_trades),
                        "train_window": self.train_window,
                        "test_window":  self.test_window,
                        "step_size":    self.step_size,
                        "slippage_pct": self.slippage_pct,
                        "hmm_config":   {**hmm_cfg,
                                         "min_confidence": conf,
                                         "stability_bars": stab},
                    },
                    combined_regime_pnl=cell_regime_pnl,
                )
                results.append((conf, stab, bt_result))

        return results

    # ── Stitching helpers ──────────────────────────────────────────────────────

    def _stitch_equity_curves(self, windows: List[WindowResult]) -> pd.Series:
        """
        Concatenate per-fold equity curves into a single continuous series.

        Because each fold's simulation starts at the previous fold's ending
        equity, the curves are already continuous in dollar terms — no
        re-scaling is required.
        """
        pieces = [w.equity_curve for w in windows if len(w.equity_curve) > 0]
        if not pieces:
            return pd.Series(dtype=float, name="equity")
        return pd.concat(pieces).sort_index()

    def _simulate_bar(
        self,
        bar_idx: int,
        prices_history: pd.DataFrame,
        features_history: np.ndarray,
        current_weights: Dict[str, float],
        equity: float,
        signal_generator,
    ) -> Tuple[Dict[str, float], float, Dict]:
        """
        Legacy stub — bar simulation is inlined in :meth:`_run_single_window`.
        """
        raise NotImplementedError(
            "Use _run_single_window() instead of _simulate_bar()."
        )
