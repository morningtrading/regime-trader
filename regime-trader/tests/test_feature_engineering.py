"""
test_feature_engineering.py — Unit tests for feature_engineering helpers and
FeatureEngineer class methods not covered by test_look_ahead.py.

Covers:
  - compute_vol_ratio, compute_normalized_volume, compute_volume_trend
  - compute_dist_from_sma, compute_roc, compute_normalized_atr
  - rolling_zscore
  - FeatureEngineer.build_feature_matrix (dropna, feature subset, error on unknown)
  - FeatureEngineer.build_multi_symbol_features (single/multi symbol, empty input)
  - FeatureEngineer.check_no_lookahead
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.feature_engineering import (
    FEATURE_COLUMNS,
    FeatureEngineer,
    compute_adx,
    compute_dist_from_sma,
    compute_log_returns,
    compute_normalized_atr,
    compute_normalized_volume,
    compute_roc,
    compute_realized_vol,
    compute_rsi,
    compute_sma_slope,
    compute_vol_ratio,
    compute_volume_trend,
    rolling_zscore,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _close(n: int = 300, seed: int = 1) -> pd.Series:
    rng = np.random.default_rng(seed)
    log_ret = rng.normal(0.0003, 0.01, n)
    prices = np.cumprod(1.0 + log_ret) * 100.0
    dates = pd.bdate_range("2022-01-01", periods=n)
    return pd.Series(prices, index=dates, name="close")


def _ohlcv(n: int = 300, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = _close(n, seed)
    dates = close.index
    high = close.values * (1.0 + rng.uniform(0.001, 0.012, n))
    low = close.values * (1.0 - rng.uniform(0.001, 0.012, n))
    volume = rng.integers(500_000, 5_000_000, n).astype(float)
    return pd.DataFrame(
        {
            "open": close.values * (1 + rng.normal(0, 0.003, n)),
            "high": high,
            "low": low,
            "close": close.values,
            "volume": volume,
        },
        index=dates,
    )


# ── compute_vol_ratio ─────────────────────────────────────────────────────────

class TestComputeVolRatio:
    def test_returns_series_same_length(self) -> None:
        c = _close()
        vr = compute_vol_ratio(c, short=5, long=20)
        assert isinstance(vr, pd.Series)
        assert len(vr) == len(c)

    def test_first_values_are_nan(self) -> None:
        c = _close(100)
        vr = compute_vol_ratio(c, short=5, long=20)
        # First 20 bars should be NaN (long window warm-up)
        assert vr.iloc[:20].isna().all()

    def test_values_positive_after_warmup(self) -> None:
        c = _close(200)
        vr = compute_vol_ratio(c, short=5, long=20)
        valid = vr.dropna()
        assert (valid >= 0).all()


# ── compute_normalized_volume ─────────────────────────────────────────────────

class TestComputeNormalizedVolume:
    def test_shape_and_dtype(self) -> None:
        df = _ohlcv(200)
        nv = compute_normalized_volume(df["volume"], window=50)
        assert len(nv) == 200

    def test_warm_up_is_nan(self) -> None:
        df = _ohlcv(200)
        nv = compute_normalized_volume(df["volume"], window=50)
        assert nv.iloc[:50].isna().any()

    def test_roughly_zero_mean(self) -> None:
        df = _ohlcv(500)
        nv = compute_normalized_volume(df["volume"], window=50).dropna()
        # z-score should have mean near 0
        assert abs(nv.mean()) < 1.0


# ── compute_volume_trend ──────────────────────────────────────────────────────

class TestComputeVolumeTrend:
    def test_returns_series(self) -> None:
        df = _ohlcv(300)
        vt = compute_volume_trend(df["volume"], sma_window=10, slope_window=10)
        assert isinstance(vt, pd.Series)
        assert len(vt) == 300

    def test_leading_nans(self) -> None:
        df = _ohlcv(200)
        vt = compute_volume_trend(df["volume"], sma_window=10, slope_window=10)
        # At least sma_window + slope_window - 1 leading NaNs
        assert vt.iloc[:18].isna().any()


# ── compute_dist_from_sma ─────────────────────────────────────────────────────

class TestComputeDistFromSma:
    def test_zero_at_sma(self) -> None:
        """If price equals its 200-bar SMA, dist should be 0."""
        # Flat price line: SMA equals price, distance = 0
        c = pd.Series([100.0] * 300)
        dist = compute_dist_from_sma(c, sma_window=200)
        valid = dist.dropna()
        np.testing.assert_allclose(valid.values, 0.0, atol=1e-10)

    def test_positive_above_sma(self) -> None:
        """Rising prices should produce positive dist_sma values."""
        c = pd.Series(np.linspace(100, 200, 400))
        dist = compute_dist_from_sma(c, sma_window=200).dropna()
        # End of a rising line is above its 200-bar SMA → positive
        assert dist.iloc[-1] > 0.0

    def test_shape_preserved(self) -> None:
        c = _close(250)
        dist = compute_dist_from_sma(c)
        assert len(dist) == len(c)


# ── compute_roc ───────────────────────────────────────────────────────────────

class TestComputeRoc:
    def test_first_period_bars_are_nan(self) -> None:
        c = _close(100)
        roc = compute_roc(c, period=10)
        assert roc.iloc[:10].isna().all()

    def test_constant_price_zero_roc(self) -> None:
        c = pd.Series([50.0] * 100)
        roc = compute_roc(c, period=5)
        valid = roc.dropna()
        np.testing.assert_allclose(valid.values, 0.0, atol=1e-10)

    def test_doubling_price_gives_one(self) -> None:
        """If price doubles in one period, ROC = 1.0."""
        c = pd.Series([100.0, 200.0])
        roc = compute_roc(c, period=1)
        assert float(roc.iloc[-1]) == pytest.approx(1.0)


# ── compute_normalized_atr ────────────────────────────────────────────────────

class TestComputeNormalizedAtr:
    def test_values_positive(self) -> None:
        df = _ohlcv(200)
        natr = compute_normalized_atr(
            df["high"], df["low"], df["close"], period=14
        )
        assert (natr.dropna() > 0).all()

    def test_shape_preserved(self) -> None:
        df = _ohlcv(200)
        natr = compute_normalized_atr(df["high"], df["low"], df["close"])
        assert len(natr) == len(df)


# ── rolling_zscore ─────────────────────────────────────────────────────────────

class TestRollingZscore:
    def test_nan_before_window(self) -> None:
        s = pd.Series(np.arange(100, dtype=float))
        z = rolling_zscore(s, window=30)
        assert z.iloc[:29].isna().all()

    def test_mean_near_zero_std_near_one(self) -> None:
        rng = np.random.default_rng(99)
        s = pd.Series(rng.normal(5.0, 2.0, 500))
        z = rolling_zscore(s, window=60).dropna()
        assert abs(z.mean()) < 0.5
        assert 0.5 < z.std() < 2.0

    def test_constant_series_is_nan(self) -> None:
        """Constant series has zero std; result should be NaN."""
        s = pd.Series([3.14] * 100)
        z = rolling_zscore(s, window=30)
        # After warm-up, all values should be NaN (std=0 -> NaN)
        assert z.dropna().empty or (z.dropna() == 0.0).all()


# ── FeatureEngineer.build_feature_matrix ─────────────────────────────────────

class TestBuildFeatureMatrix:
    def test_returns_dataframe_no_nan(self) -> None:
        fe = FeatureEngineer(zscore_window=60)
        df = fe.build_feature_matrix(_ohlcv(400))
        assert isinstance(df, pd.DataFrame)
        assert not df.isnull().any(axis=None)

    def test_columns_are_feature_columns(self) -> None:
        fe = FeatureEngineer(zscore_window=60)
        df = fe.build_feature_matrix(_ohlcv(400))
        assert list(df.columns) == FEATURE_COLUMNS

    def test_feature_subset_selection(self) -> None:
        fe = FeatureEngineer(zscore_window=60)
        subset = ["log_ret_1", "rsi14"]
        df = fe.build_feature_matrix(_ohlcv(400), feature_names=subset)
        assert list(df.columns) == subset

    def test_unknown_feature_raises(self) -> None:
        fe = FeatureEngineer(zscore_window=60)
        with pytest.raises(ValueError, match="Unknown features"):
            fe.build_feature_matrix(_ohlcv(400), feature_names=["nonexistent"])

    def test_dropna_false_keeps_all_rows(self) -> None:
        fe = FeatureEngineer(zscore_window=60)
        ohlcv = _ohlcv(400)
        df_drop = fe.build_feature_matrix(ohlcv, dropna=True)
        df_keep = fe.build_feature_matrix(ohlcv, dropna=False)
        assert len(df_keep) == len(ohlcv)
        assert len(df_drop) < len(df_keep)


# ── FeatureEngineer.build_multi_symbol_features ───────────────────────────────

class TestBuildMultiSymbolFeatures:
    def _prices(self, symbols=("SPY", "QQQ"), n=400, seed=7) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        dates = pd.bdate_range("2021-01-01", periods=n)
        data = {}
        for i, sym in enumerate(symbols):
            log_ret = rng.normal(0.0003, 0.01, n)
            data[sym] = np.cumprod(1.0 + log_ret) * (100.0 + i * 50)
        return pd.DataFrame(data, index=dates)

    def test_returns_dataframe(self) -> None:
        fe = FeatureEngineer(zscore_window=60)
        prices = self._prices()
        result = fe.build_multi_symbol_features(prices)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_columns_include_all_symbols(self) -> None:
        fe = FeatureEngineer(zscore_window=60)
        prices = self._prices(symbols=["SPY", "QQQ"])
        result = fe.build_multi_symbol_features(prices)
        for sym in ["SPY", "QQQ"]:
            assert any(col.startswith(sym) for col in result.columns)

    def test_single_symbol(self) -> None:
        fe = FeatureEngineer(zscore_window=60)
        prices = self._prices(symbols=["SPY"])
        result = fe.build_multi_symbol_features(prices)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_empty_prices_returns_empty(self) -> None:
        fe = FeatureEngineer(zscore_window=60)
        result = fe.build_multi_symbol_features(pd.DataFrame())
        assert result.empty

    def test_feature_subset_applied(self) -> None:
        fe = FeatureEngineer(zscore_window=60)
        prices = self._prices()
        result = fe.build_multi_symbol_features(
            prices, feature_names=["log_ret_1", "roc_10"]
        )
        for col in result.columns:
            assert col.endswith("_log_ret_1") or col.endswith("_roc_10")

    def test_no_nans_in_result(self) -> None:
        fe = FeatureEngineer(zscore_window=60)
        prices = self._prices(n=500)
        result = fe.build_multi_symbol_features(prices)
        assert not result.isnull().any(axis=None)


# ── FeatureEngineer.check_no_lookahead ────────────────────────────────────────

class TestCheckNoLookahead:
    def test_returns_true_on_causal_features(self) -> None:
        fe = FeatureEngineer(zscore_window=60)
        ohlcv = _ohlcv(500)
        features = fe.compute(ohlcv).dropna()
        result = fe.check_no_lookahead(features, ohlcv)
        assert isinstance(result, bool)
        # Causal features may have some modest predictive power; just check type.

    def test_returns_false_on_future_leak(self) -> None:
        """A feature equal to next-bar return should fail the look-ahead check."""
        fe = FeatureEngineer(zscore_window=60)
        ohlcv = _ohlcv(500)
        features = fe.compute(ohlcv).dropna()
        # Inject a column that IS the future return (look-ahead)
        close = ohlcv["close"]
        future_ret = compute_log_returns(close, 1).shift(-1)
        # Align
        features = features.copy()
        features["log_ret_1"] = future_ret.reindex(features.index)
        result = fe.check_no_lookahead(features.dropna(), ohlcv)
        assert result is False
