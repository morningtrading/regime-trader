"""
feature_engineering.py — Technical indicator and feature computation.

All functions are strictly causal: value at bar t uses only data from bars
[0, t].  Rolling windows end at bar t-1 for all lagged inputs; rolling
z-score standardisation uses a 252-bar lookback.

Feature columns produced (in order):
    0  log_ret_1        1-bar log return
    1  log_ret_5        5-bar log return
    2  log_ret_20       20-bar log return
    3  realized_vol_20  annualised 20-bar realised vol
    4  vol_ratio        5-bar vol / 20-bar vol
    5  volume_norm      volume z-score vs 50-bar mean
    6  volume_trend     normalised slope of 10-bar volume SMA
    7  adx_14           ADX(14) – Wilder smoothing
    8  sma50_slope      normalised slope of 50-bar SMA
    9  rsi14            RSI(14)
    10 dist_sma200      (close - SMA200) / close
    11 roc_10           rate-of-change over 10 bars
    12 roc_20           rate-of-change over 20 bars
    13 norm_atr_14      ATR(14) / close
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

FEATURE_COLUMNS: List[str] = [
    "log_ret_1",
    "log_ret_5",
    "log_ret_20",
    "realized_vol_20",
    "vol_ratio",
    "volume_norm",
    "volume_trend",
    "adx_14",
    "sma50_slope",
    "rsi14",
    "dist_sma200",
    "roc_10",
    "roc_20",
    "norm_atr_14",
    # Cross-asset VIX-based features (populated when vix_series is provided
    # to build_feature_matrix; otherwise NaN and dropped by warm-up).
    "vix_level",
    "vix_zscore_60",
    # Cross-asset credit spread (HYG/LQD z-score) — populated when
    # credit_series is provided. Already z-scored by the fetcher.
    "credit_spread_z60",
]

# ── HMM feature presets (referenced by main.py and backtester.py) ─────────────
# Toggle via hmm.extended_features in settings.yaml.
HMM_BASE_FEATURES: List[str] = [
    "log_ret_1",
    "realized_vol_20",
]
HMM_EXTENDED_FEATURES: List[str] = [
    "log_ret_1",
    "realized_vol_20",
    "vol_ratio",
    "adx_14",
    "dist_sma200",
]
# Extended + VIX cross-asset features. Requires a VIX/VXX series to be
# supplied to build_feature_matrix; otherwise rows are dropped as warm-up.
HMM_EXTENDED_VIX_FEATURES: List[str] = HMM_EXTENDED_FEATURES + [
    # vix_level intentionally omitted: non-stationary (VIX levels drift across
    # regimes), contaminates HMM state definitions. Sub-ablation on indices
    # showed adding vix_level *with* vix_zscore_60 dropped Sharpe 0.446 -> 0.118.
    # Only vix_zscore_60 (rolling 60-bar z-score) is retained — it is
    # stationary-ish and gave +70% Sharpe over EXTENDED baseline on indices.
    "vix_zscore_60",
]
# Extended + VIX + Credit Spread (HYG/LQD z-score). Orthogonal to equity
# vol — captures credit-cycle stress the VIX misses.
HMM_EXTENDED_VIX_CREDIT_FEATURES: List[str] = HMM_EXTENDED_VIX_FEATURES + [
    "credit_spread_z60",
]


def hmm_feature_names(hmm_cfg: dict) -> List[str]:
    """Return the feature list to feed the HMM based on settings.

    Priority:
      1. ``hmm.features_override`` (explicit list, wins if non-empty) —
         used for ablation studies / custom feature sets.
      2. ``hmm.use_credit_spread_features`` boolean → EXTENDED + VIX + CREDIT.
      3. ``hmm.use_vix_features`` boolean → EXTENDED + VIX set.
      4. ``hmm.extended_features`` boolean toggle (default True).
    """
    override = hmm_cfg.get("features_override")
    if override:
        return list(override)
    if hmm_cfg.get("use_credit_spread_features", False):
        return HMM_EXTENDED_VIX_CREDIT_FEATURES
    if hmm_cfg.get("use_vix_features", False):
        return HMM_EXTENDED_VIX_FEATURES
    if hmm_cfg.get("extended_features", True):
        return HMM_EXTENDED_FEATURES
    return HMM_BASE_FEATURES


# ── Pure helper functions (no class state) ────────────────────────────────────


def _wilder_smooth(series: pd.Series, period: int) -> pd.Series:
    """EWM with alpha = 1/period — identical to Wilder's smoothing."""
    return series.ewm(com=period - 1, adjust=False).mean()


def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    """
    Causal linear-regression slope over a rolling window.

    Uses the closed-form formula so it is O(N) rather than relying on
    polyfit inside apply (which would be O(N * window)).

    Parameters
    ----------
    series : pd.Series
    window : int

    Returns
    -------
    pd.Series of slope values (NaN for the first window-1 bars).
    """
    t = np.arange(window, dtype=float)
    t_mean = t.mean()
    t_var = float(((t - t_mean) ** 2).sum())

    def _slope(vals: np.ndarray) -> float:
        if np.isnan(vals).any():
            return np.nan
        v_mean = vals.mean()
        cov = float(((t - t_mean) * (vals - v_mean)).sum())
        return cov / t_var if t_var > 0.0 else 0.0

    return series.rolling(window).apply(_slope, raw=True)


# ── Individual feature functions ───────────────────────────────────────────────


def compute_log_returns(close: pd.Series, period: int = 1) -> pd.Series:
    """Log return over ``period`` bars: ln(p_t / p_{t-period})."""
    return np.log(close / close.shift(period))


def compute_realized_vol(close: pd.Series, window: int = 20) -> pd.Series:
    """
    Annualised realised volatility: rolling std of 1-bar log returns × √252.

    Parameters
    ----------
    close : pd.Series
    window : int  rolling window in bars

    Returns
    -------
    pd.Series  annualised vol (NaN for first window bars).
    """
    ret = compute_log_returns(close, 1)
    return ret.rolling(window).std() * np.sqrt(252)


def compute_vol_ratio(
    close: pd.Series, short: int = 5, long: int = 20
) -> pd.Series:
    """Short-term realised vol / long-term realised vol."""
    ret = compute_log_returns(close, 1)
    short_vol = ret.rolling(short).std()
    long_vol = ret.rolling(long).std()
    return short_vol / long_vol.replace(0.0, np.nan)


def compute_normalized_volume(volume: pd.Series, window: int = 50) -> pd.Series:
    """Volume z-score relative to ``window``-bar rolling mean and std."""
    mu = volume.rolling(window).mean()
    sigma = volume.rolling(window).std()
    return (volume - mu) / sigma.replace(0.0, np.nan)


def compute_volume_trend(
    volume: pd.Series,
    sma_window: int = 10,
    slope_window: int = 10,
) -> pd.Series:
    """
    Normalised linear-regression slope of the volume SMA.

    Computed as: slope(SMA_{sma_window}(volume), slope_window) / mean_volume
    so the result is scale-independent.
    """
    vol_sma = volume.rolling(sma_window).mean()
    slope = _rolling_slope(vol_sma, slope_window)
    # Normalise by the rolling mean volume over the combined window
    denom = volume.rolling(sma_window + slope_window).mean().replace(0.0, np.nan)
    return slope / denom


def compute_adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Average Directional Index using Wilder's smoothing.

    Parameters
    ----------
    high, low, close : pd.Series  aligned OHLCV series
    period : int  smoothing period (default 14)

    Returns
    -------
    pd.Series  ADX values (0–100).
    """
    prev_close = close.shift(1)
    prev_high = high.shift(1)
    prev_low = low.shift(1)

    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    up_move = high - prev_high
    down_move = prev_low - low

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0.0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0.0), 0.0)

    atr = _wilder_smooth(tr, period)
    plus_di = 100.0 * _wilder_smooth(plus_dm, period) / atr.replace(0.0, np.nan)
    minus_di = 100.0 * _wilder_smooth(minus_dm, period) / atr.replace(0.0, np.nan)

    di_sum = (plus_di + minus_di).replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / di_sum
    return _wilder_smooth(dx, period)


def compute_sma_slope(
    close: pd.Series,
    sma_window: int = 50,
    slope_window: int = 10,
) -> pd.Series:
    """
    Normalised slope of the SMA.

    slope(SMA_{sma_window}(close), slope_window) / close
    so the result is dimensionless (roughly: daily fractional drift).
    """
    sma = close.rolling(sma_window).mean()
    slope = _rolling_slope(sma, slope_window)
    return slope / close.replace(0.0, np.nan)


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Wilder RSI.

    Returns values in [0, 100].  NaN for the first ``period`` bars.
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def compute_dist_from_sma(close: pd.Series, sma_window: int = 200) -> pd.Series:
    """(close − SMA_{sma_window}) / close  — fraction above/below the SMA."""
    sma = close.rolling(sma_window).mean()
    return (close - sma) / close.replace(0.0, np.nan)


def compute_roc(close: pd.Series, period: int = 10) -> pd.Series:
    """Rate of change: (close − close_{t-period}) / close_{t-period}."""
    prev = close.shift(period)
    return (close - prev) / prev.replace(0.0, np.nan)


def compute_normalized_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """ATR(period) / close — dimensionless measure of range."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = _wilder_smooth(tr, period)
    return atr / close.replace(0.0, np.nan)


def rolling_zscore(series: pd.Series, window: int = 252) -> pd.Series:
    """
    Rolling z-score using a ``window``-bar lookback.

    z_t = (x_t − μ_{t-1,window}) / σ_{t-1,window}

    Values are NaN until the first full window is available.
    """
    mu = series.rolling(window).mean()
    sigma = series.rolling(window).std()
    return (series - mu) / sigma.replace(0.0, np.nan)


# ── FeatureEngineer class ─────────────────────────────────────────────────────


class FeatureEngineer:
    """
    Compute the full feature matrix consumed by :class:`~core.hmm_engine.HMMEngine`.

    All features are computed causally (no look-ahead) and then normalised
    with rolling z-scores so the HMM sees stationary, mean-zero inputs.

    Parameters
    ----------
    zscore_window :
        Lookback window for rolling z-score standardisation.
    vol_window :
        Window for realised volatility (default 20 bars).
    adx_period :
        ADX smoothing period (default 14).
    rsi_period :
        RSI smoothing period (default 14).
    sma_long :
        Long SMA window used for distance feature (default 200 bars).
    sma_trend :
        Trend SMA window used for slope feature (default 50 bars).
    """

    def __init__(
        self,
        zscore_window: int = 252,
        vol_window: int = 20,
        adx_period: int = 14,
        rsi_period: int = 14,
        sma_long: int = 200,
        sma_trend: int = 50,
        volume_norm_window: int = 50,
    ) -> None:
        self.zscore_window = zscore_window
        self.vol_window = vol_window
        self.adx_period = adx_period
        self.rsi_period = rsi_period
        self.sma_long = sma_long
        self.sma_trend = sma_trend
        self.volume_norm_window = volume_norm_window

    # ── Public API ─────────────────────────────────────────────────────────────

    def compute(
        self,
        ohlcv: pd.DataFrame,
        vix_series: Optional[pd.Series] = None,
        credit_series: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Compute and z-score all features.

        Parameters
        ----------
        ohlcv :
            DataFrame with columns ``[open, high, low, close, volume]``.
            Index must be a DatetimeIndex sorted ascending.
        vix_series :
            Optional VIX (or VXX) close series indexed like ``ohlcv``.
            When provided, ``vix_level`` and ``vix_zscore_60`` features are
            populated; otherwise they are NaN and dropped by the warm-up
            filter when included in ``feature_names``.

        Returns
        -------
        DataFrame of shape ``(n_bars, len(FEATURE_COLUMNS))``.
        Rows within the warm-up period contain NaN.
        """
        close = ohlcv["close"]
        high = ohlcv["high"]
        low = ohlcv["low"]
        volume = ohlcv["volume"]

        raw = pd.DataFrame(index=ohlcv.index)

        # ── Returns ───────────────────────────────────────────────────────────
        raw["log_ret_1"] = compute_log_returns(close, 1)
        raw["log_ret_5"] = compute_log_returns(close, 5)
        raw["log_ret_20"] = compute_log_returns(close, 20)

        # ── Volatility ────────────────────────────────────────────────────────
        raw["realized_vol_20"] = compute_realized_vol(close, self.vol_window)
        raw["vol_ratio"] = compute_vol_ratio(close, 5, self.vol_window)

        # ── Volume ────────────────────────────────────────────────────────────
        raw["volume_norm"] = compute_normalized_volume(volume, self.volume_norm_window)
        raw["volume_trend"] = compute_volume_trend(volume, 10, 10)

        # ── Trend ─────────────────────────────────────────────────────────────
        raw["adx_14"] = compute_adx(high, low, close, self.adx_period)
        raw["sma50_slope"] = compute_sma_slope(close, self.sma_trend, 10)

        # ── Mean reversion ────────────────────────────────────────────────────
        raw["rsi14"] = compute_rsi(close, self.rsi_period)
        raw["dist_sma200"] = compute_dist_from_sma(close, self.sma_long)

        # ── Momentum ──────────────────────────────────────────────────────────
        raw["roc_10"] = compute_roc(close, 10)
        raw["roc_20"] = compute_roc(close, 20)

        # ── Range ─────────────────────────────────────────────────────────────
        raw["norm_atr_14"] = compute_normalized_atr(high, low, close, self.adx_period)

        # ── VIX cross-asset features ────────────────────────────────────────
        # Only populated when a VIX/VXX series is supplied. Fills to NaN
        # otherwise, which causes build_feature_matrix to drop these rows
        # during warm-up — so the baseline path (no VIX) still works.
        if vix_series is not None and not vix_series.empty:
            vix_aligned = vix_series.reindex(ohlcv.index, method="ffill").astype(float)
            # Log to compress scale (VIX can span 9-80)
            raw["vix_level"] = np.log(vix_aligned.clip(lower=1e-6))
            vix_mean = vix_aligned.rolling(60, min_periods=20).mean()
            vix_std  = vix_aligned.rolling(60, min_periods=20).std()
            raw["vix_zscore_60"] = (vix_aligned - vix_mean) / vix_std.replace(0, np.nan)
        else:
            raw["vix_level"] = np.nan
            raw["vix_zscore_60"] = np.nan

        # ── Credit spread cross-asset feature ─────────────────────────────────
        # HYG/LQD z-score is pre-computed by the fetcher. Just align & carry.
        if credit_series is not None and not credit_series.empty:
            raw["credit_spread_z60"] = (
                credit_series.reindex(ohlcv.index, method="ffill").astype(float)
            )
        else:
            raw["credit_spread_z60"] = np.nan

        # ── Rolling z-score standardisation ──────────────────────────────────
        standardised = pd.DataFrame(index=ohlcv.index)
        for col in FEATURE_COLUMNS:
            # vix_level is already log-transformed; z-score both VIX features
            # the same way as the rest for consistent HMM input scale.
            standardised[col] = rolling_zscore(raw[col], self.zscore_window)

        return standardised

    def build_feature_matrix(
        self,
        ohlcv: pd.DataFrame,
        feature_names: Optional[List[str]] = None,
        dropna: bool = True,
        vix_series: Optional[pd.Series] = None,
        credit_series: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Build the standardised feature DataFrame ready for HMM training.

        Parameters
        ----------
        ohlcv :
            Raw OHLCV DataFrame.
        feature_names :
            Subset of :data:`FEATURE_COLUMNS` to include.  All columns if None.
        dropna :
            Drop rows that contain any NaN (warm-up rows).  Default True.

        Returns
        -------
        DataFrame with selected feature columns and NaN rows removed.
        """
        df = self.compute(ohlcv, vix_series=vix_series, credit_series=credit_series)
        if feature_names:
            unknown = [c for c in feature_names if c not in df.columns]
            if unknown:
                raise ValueError(f"Unknown features: {unknown}")
            df = df[feature_names]
            # Drop columns that are entirely NaN (e.g. VIX fetch failure) so
            # they don't wipe all rows when dropna runs below.
            all_nan_cols = [c for c in df.columns if df[c].isna().all()]
            if all_nan_cols:
                logger.warning(
                    "Dropping all-NaN feature column(s) %s — likely fetch failure",
                    all_nan_cols,
                )
                df = df.drop(columns=all_nan_cols)
        if dropna:
            n_before = len(df)
            df = df.dropna()
            n_dropped = n_before - len(df)
            if n_dropped:
                logger.debug("Dropped %d warm-up rows (NaN)", n_dropped)
        return df

    def build_multi_symbol_features(
        self,
        prices: pd.DataFrame,
        feature_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Compute features for each symbol and concatenate column-wise.

        Parameters
        ----------
        prices :
            Wide-format close prices, columns = symbols.  Uses close price
            only (no volume / high / low features).
        feature_names :
            Features to include for each symbol.

        Returns
        -------
        DataFrame with columns named ``"{symbol}_{feature}"``.
        """
        frames = []
        return_features = ["log_ret_1", "log_ret_5", "log_ret_20",
                           "realized_vol_20", "vol_ratio", "roc_10", "roc_20"]
        cols = feature_names or return_features

        for symbol in prices.columns:
            close = prices[symbol].dropna()
            raw = pd.DataFrame(index=close.index)

            if "log_ret_1" in cols:
                raw[f"{symbol}_log_ret_1"] = compute_log_returns(close, 1)
            if "log_ret_5" in cols:
                raw[f"{symbol}_log_ret_5"] = compute_log_returns(close, 5)
            if "log_ret_20" in cols:
                raw[f"{symbol}_log_ret_20"] = compute_log_returns(close, 20)
            if "realized_vol_20" in cols:
                raw[f"{symbol}_realized_vol_20"] = compute_realized_vol(close, 20)
            if "vol_ratio" in cols:
                raw[f"{symbol}_vol_ratio"] = compute_vol_ratio(close, 5, 20)
            if "roc_10" in cols:
                raw[f"{symbol}_roc_10"] = compute_roc(close, 10)
            if "roc_20" in cols:
                raw[f"{symbol}_roc_20"] = compute_roc(close, 20)

            for col in raw.columns:
                raw[col] = rolling_zscore(raw[col], self.zscore_window)

            frames.append(raw)

        if not frames:
            return pd.DataFrame()

        return pd.concat(frames, axis=1).dropna()

    def check_no_lookahead(
        self, features: pd.DataFrame, prices: pd.DataFrame
    ) -> bool:
        """
        Heuristic look-ahead audit via Pearson correlation.

        Checks whether any feature column has a statistically significant
        (p < 0.05) correlation with *next-bar* returns.  |r| > 0.15 is
        treated as suspicious.

        Parameters
        ----------
        features :
            Standardised feature matrix.
        prices :
            OHLCV DataFrame (needs ``close`` column).

        Returns
        -------
        True if no suspicious correlation is detected, False otherwise.
        """
        close = prices["close"] if "close" in prices.columns else prices.iloc[:, 0]
        future_ret = compute_log_returns(close, 1).shift(-1)  # next bar's return

        suspicious: list = []
        for col in features.columns:
            s = features[col].align(future_ret, join="inner")
            mask = ~(s[0].isna() | s[1].isna())
            if mask.sum() < 100:
                continue
            r, p = stats.pearsonr(s[0][mask].values, s[1][mask].values)
            if abs(r) > 0.15 and p < 0.05:
                suspicious.append((col, round(r, 4), round(p, 6)))

        if suspicious:
            logger.warning(
                "Look-ahead check: suspicious feature correlations %s", suspicious
            )
            return False

        logger.debug("Look-ahead check passed — no suspicious correlations found.")
        return True
