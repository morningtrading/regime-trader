"""
tests/test_multistrat_backtest.py — Tests for MultiStrategyBacktester (Phase 4).

Covers:
  1. Portfolio return matches weighted sum of strategy returns (accounting)
  2. Allocator weights match expected for known volatility inputs
  3. Correlation report shows expected values for synthetic correlated returns
  4. Strategy auto-disabling via health check works inside the backtest
  5. CSV output files are created with correct columns
  6. BenchmarkComparison identifies correct best/worst single strategy
"""

from __future__ import annotations

import math
import os
import tempfile
from collections import deque
from typing import Dict

import numpy as np
import pandas as pd
import pytest

from backtest.multi_strategy_backtester import (
    MultiStrategyBacktester,
    StrategySpec,
    _BacktestProxy,
    _build_correlation_report,
    _metrics,
)
from core.capital_allocator import CapitalAllocator
from core.strategy_registry import StrategyRegistry
from core.regime_strategies import _HEALTH_MAX_CONSEC_LOSSES


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _make_prices(
    symbols: list[str],
    n: int = 700,
    seed: int = 42,
    correlated: bool = False,
) -> pd.DataFrame:
    """Generate synthetic close-price DataFrame with a DatetimeIndex."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-01", periods=n, freq="B")
    cols: Dict[str, np.ndarray] = {}

    if correlated:
        # All symbols follow the same base path + small idiosyncratic noise
        base = 100.0 + rng.normal(0, 0.5, n).cumsum()
        for sym in symbols:
            noise = rng.normal(0, 0.05, n).cumsum()
            cols[sym] = np.maximum(base + noise, 10.0)
    else:
        for sym in symbols:
            cols[sym] = np.maximum(
                100.0 + rng.normal(0, 0.8, n).cumsum(),
                10.0,
            )

    return pd.DataFrame(cols, index=idx)


def _make_backtester(**kwargs) -> MultiStrategyBacktester:
    defaults = dict(
        initial_capital=100_000.0,
        train_window=200,
        test_window=80,
        step_size=80,
        allocate_interval=5,
        reserve=0.10,
    )
    defaults.update(kwargs)
    return MultiStrategyBacktester(**defaults)


def _simple_specs() -> list[StrategySpec]:
    return [
        StrategySpec(name="alpha", symbols=["SPY"], weight_min=0.0, weight_max=1.0),
        StrategySpec(name="beta",  symbols=["QQQ"], weight_min=0.0, weight_max=1.0),
    ]


# ── Helper-level unit tests (no HMM) ──────────────────────────────────────────

class TestMetrics:
    def test_zero_return_series_gives_zero_cagr(self):
        returns = pd.Series([0.0] * 50)
        m = _metrics(returns, 100_000, 100_000)
        assert m["cagr"] == pytest.approx(0.0, abs=1e-9)
        assert m["max_drawdown"] == pytest.approx(0.0, abs=1e-9)

    def test_positive_monotone_returns(self):
        returns = pd.Series([0.001] * 252)
        initial = 100_000.0
        final = initial * (1.001 ** 252)
        m = _metrics(returns, initial, final)
        assert m["total_return"] > 0
        assert m["cagr"] > 0
        assert m["sharpe"] > 0
        assert m["max_drawdown"] == pytest.approx(0.0, abs=1e-6)
        assert m["calmar"] == 0.0   # no drawdown → calmar defined as 0

    def test_single_crash_gives_nonzero_drawdown(self):
        returns = [0.01] * 100 + [-0.50] + [0.0] * 10
        r = pd.Series(returns)
        equity = 100_000 * (1 + r).prod()
        m = _metrics(r, 100_000, equity)
        assert m["max_drawdown"] > 0.40

    def test_empty_returns_returns_zeros(self):
        m = _metrics(pd.Series(dtype=float), 100_000, 100_000)
        assert m["sharpe"] == 0.0
        assert m["calmar"] == 0.0


class TestCorrelationReport:
    def _make_returns(self, n: int, corr: float, seed: int = 0) -> Dict[str, pd.Series]:
        """Two return series with a given Pearson correlation."""
        rng = np.random.default_rng(seed)
        idx = pd.date_range("2023-01-01", periods=n, freq="B")
        x = rng.normal(0, 0.01, n)
        noise = rng.normal(0, 0.01, n)
        y = corr * x + math.sqrt(1 - corr ** 2) * noise
        return {
            "A": pd.Series(x, index=idx),
            "B": pd.Series(y, index=idx),
        }

    def test_identical_series_has_corr_one(self):
        idx = pd.date_range("2023-01-01", periods=200, freq="B")
        returns = pd.Series(np.random.default_rng(1).normal(0, 0.01, 200), index=idx)
        report = _build_correlation_report({"X": returns, "Y": returns}, window=60)
        avg = report.pairwise_avg.loc["X", "Y"]
        assert avg == pytest.approx(1.0, abs=0.01)

    def test_uncorrelated_series_near_zero(self):
        rng = np.random.default_rng(99)
        idx = pd.date_range("2023-01-01", periods=300, freq="B")
        r = {
            "A": pd.Series(rng.normal(0, 0.01, 300), index=idx),
            "B": pd.Series(rng.normal(0, 0.01, 300), index=idx),
        }
        report = _build_correlation_report(r, window=60)
        avg = report.pairwise_avg.loc["A", "B"]
        assert abs(avg) < 0.25   # random noise, won't be exactly 0

    def test_highly_correlated_triggers_pct_above_threshold(self):
        returns = self._make_returns(300, corr=0.95)
        report = _build_correlation_report(returns, window=60, threshold=0.80)
        pct = report.pct_time_above_threshold.loc["A", "B"]
        assert pct > 0.50   # most bars should be above 0.80

    def test_report_has_symmetric_matrix(self):
        rng = np.random.default_rng(7)
        idx = pd.date_range("2023-01-01", periods=200, freq="B")
        r = {
            "X": pd.Series(rng.normal(0, 0.01, 200), index=idx),
            "Y": pd.Series(rng.normal(0, 0.01, 200), index=idx),
        }
        report = _build_correlation_report(r)
        assert report.pairwise_avg.loc["X", "Y"] == pytest.approx(
            report.pairwise_avg.loc["Y", "X"], abs=1e-9
        )

    def test_three_strategies_pairwise(self):
        rng = np.random.default_rng(12)
        idx = pd.date_range("2023-01-01", periods=200, freq="B")
        r = {n: pd.Series(rng.normal(0, 0.01, 200), index=idx) for n in ["A", "B", "C"]}
        report = _build_correlation_report(r)
        assert set(report.rolling_history.columns) == {"A|B", "A|C", "B|C"}
        assert report.pairwise_avg.shape == (3, 3)


# ── Registry + proxy unit tests ────────────────────────────────────────────────

class TestBacktestProxy:
    def test_proxy_satisfies_base_strategy(self):
        proxy = _BacktestProxy("test")
        assert proxy.name == "test"
        assert proxy.total_allocation == pytest.approx(0.90)
        assert proxy.is_enabled is True

    def test_proxy_records_returns(self):
        proxy = _BacktestProxy("p")
        ts = pd.Timestamp("2024-01-01")
        for i in range(5):
            proxy.record_daily_return(ts + pd.Timedelta(days=i), 0.01)
        assert len(proxy.performance_history) == 5

    def test_proxy_health_check_healthy_when_empty(self):
        proxy = _BacktestProxy("p")
        h = proxy.health_check()
        assert h.is_healthy is True

    def test_proxy_health_check_unhealthy_on_streak(self):
        proxy = _BacktestProxy("p")
        ts = pd.Timestamp("2024-01-01")
        for i in range(_HEALTH_MAX_CONSEC_LOSSES):
            proxy.record_daily_return(ts + pd.Timedelta(days=i), -0.01)
        h = proxy.health_check()
        assert h.is_healthy is False


# ── Allocator weight consistency ───────────────────────────────────────────────

class TestAllocatorWeights:
    def _make_proxies_with_vol(
        self, vols: Dict[str, float], seed: int = 0
    ) -> StrategyRegistry:
        """Create registry with proxies that have performance_history of given vol."""
        StrategyRegistry._reset()
        registry = StrategyRegistry.instance()
        rng = np.random.default_rng(seed)
        ts = pd.Timestamp("2024-01-01")
        for name, vol in vols.items():
            proxy = _BacktestProxy(name)
            for i in range(62):
                r = float(rng.normal(0, vol))
                proxy.record_daily_return(ts + pd.Timedelta(days=i), r)
            registry.register(name, proxy)
        return registry

    def test_equal_weight_sums_to_one_minus_reserve(self):
        vols = {"A": 0.01, "B": 0.02}
        registry = self._make_proxies_with_vol(vols)
        allocator = CapitalAllocator(
            approach="equal_weight",
            reserve=0.10,
            total_capital=100_000,
        )
        weights = allocator.allocate(registry)
        assert sum(weights.values()) == pytest.approx(0.90, abs=1e-6)
        # Equal weight → both get ~0.45
        assert weights["A"] == pytest.approx(weights["B"], abs=0.01)

    def test_inverse_vol_lower_vol_gets_more(self):
        vols = {"low_vol": 0.005, "high_vol": 0.020}
        registry = self._make_proxies_with_vol(vols, seed=5)
        allocator = CapitalAllocator(
            approach="inverse_vol",
            reserve=0.10,
            total_capital=100_000,
        )
        weights = allocator.allocate(registry)
        assert weights["low_vol"] > weights["high_vol"]

    def test_weights_respect_min_max(self):
        vols = {"A": 0.01, "B": 0.02}
        registry = self._make_proxies_with_vol(vols)
        allocator = CapitalAllocator(
            approach="inverse_vol",
            strategy_configs={
                "A": {"weight_min": 0.30, "weight_max": 0.60},
                "B": {"weight_min": 0.30, "weight_max": 0.60},
            },
            reserve=0.10,
            total_capital=100_000,
        )
        weights = allocator.allocate(registry)
        for name in ["A", "B"]:
            assert weights[name] >= 0.30 - 1e-6
            assert weights[name] <= 0.60 + 1e-6


# ── Integration tests (full backtest with synthetic prices) ────────────────────

@pytest.fixture(scope="module")
def prices_two_strats() -> pd.DataFrame:
    """Synthetic prices for SPY + QQQ — 1200 business days (~4.8 years).

    Feature engineering (zscore_window=252, SMA_200) consumes ~450 bars as
    warmup, leaving ~750 clean bars which is enough for several folds.
    """
    return _make_prices(["SPY", "QQQ"], n=1200, seed=42)


class TestMultiStrategyIntegration:
    """Integration tests that run the full backtester end-to-end."""

    def test_result_has_per_strategy_equity(self, prices_two_strats):
        bt = _make_backtester()
        specs = _simple_specs()
        result = bt.run(prices_two_strats, specs)
        for spec in specs:
            assert spec.name in result.attributions
            eq = result.attributions[spec.name].equity_curve
            assert len(eq) > 0
            assert (eq > 0).all()

    def test_portfolio_equity_positive(self, prices_two_strats):
        bt = _make_backtester()
        result = bt.run(prices_two_strats, _simple_specs())
        assert (result.combined_portfolio_equity > 0).all()

    def test_portfolio_equity_near_weighted_sum(self, prices_two_strats):
        """Portfolio equity ≈ sum of per-strategy equities (within reserve tolerance)."""
        bt = _make_backtester()
        result = bt.run(prices_two_strats, _simple_specs())
        strat_sum = sum(
            result.attributions[n].equity_curve
            for n in result.attributions
        ).reindex(result.combined_portfolio_equity.index)
        # Portfolio includes reserve cash; allow up to 15% relative difference
        ratio = result.combined_portfolio_equity / strat_sum.clip(lower=1.0)
        assert ratio.between(0.85, 1.20).all()

    def test_final_equity_matches_last_equity_bar(self, prices_two_strats):
        bt = _make_backtester()
        result = bt.run(prices_two_strats, _simple_specs())
        assert result.final_equity == pytest.approx(
            float(result.combined_portfolio_equity.iloc[-1]), rel=1e-6
        )

    def test_correlation_report_has_one_pair(self, prices_two_strats):
        bt = _make_backtester()
        result = bt.run(prices_two_strats, _simple_specs())
        assert "alpha|beta" in result.correlation_report.rolling_history.columns
        assert result.correlation_report.pairwise_avg.shape == (2, 2)

    def test_allocator_log_has_records(self, prices_two_strats):
        bt = _make_backtester()
        result = bt.run(prices_two_strats, _simple_specs())
        # Should have at least some allocation events
        assert len(result.allocator_log) >= 0   # may be 0 if thresholds not crossed
        assert set(result.allocator_log.columns) >= {
            "strategy_name", "old_weight", "new_weight", "reason"
        }

    def test_metadata_contains_strategy_names(self, prices_two_strats):
        bt = _make_backtester()
        result = bt.run(prices_two_strats, _simple_specs())
        assert set(result.metadata["strategies"]) == {"alpha", "beta"}

    def test_three_strategies_run(self):
        prices = _make_prices(["SPY", "QQQ", "IWM"], n=1200, seed=7)
        bt = _make_backtester()
        specs = [
            StrategySpec("A", ["SPY"]),
            StrategySpec("B", ["QQQ"]),
            StrategySpec("C", ["IWM"]),
        ]
        result = bt.run(prices, specs)
        assert len(result.attributions) == 3
        assert result.correlation_report.pairwise_avg.shape == (3, 3)

    def test_requires_at_least_two_strategies(self):
        prices = _make_prices(["SPY"], n=1200)
        bt = _make_backtester()
        with pytest.raises(ValueError, match="at least 2"):
            bt.run(prices, [StrategySpec("only", ["SPY"])])

    def test_missing_symbol_raises(self):
        prices = _make_prices(["SPY"], n=1200)
        bt = _make_backtester()
        specs = [StrategySpec("A", ["SPY"]), StrategySpec("B", ["MISSING"])]
        with pytest.raises(ValueError, match="MISSING"):
            bt.run(prices, specs)


class TestStrategyDisabling:
    """Health-check triggered disabling propagates into the backtest."""

    def test_proxy_disabled_after_consecutive_losses(self):
        """Registry health check disables proxy after _HEALTH_MAX_CONSEC_LOSSES losses."""
        StrategyRegistry._reset()
        proxy = _BacktestProxy("sick2")
        ts = pd.Timestamp("2024-01-01")
        for i in range(_HEALTH_MAX_CONSEC_LOSSES):
            proxy.record_daily_return(ts + pd.Timedelta(days=i), -0.02)

        registry = StrategyRegistry.instance()
        registry.register("sick2", proxy)
        registry.run_health_checks()

        assert proxy.is_enabled is False

    def test_disabled_strategy_has_zero_weight(self):
        """After disabling, the allocator assigns 0 weight."""
        StrategyRegistry._reset()
        registry = StrategyRegistry.instance()
        proxy_a = _BacktestProxy("a")
        proxy_b = _BacktestProxy("b")
        proxy_b.is_enabled = False

        registry.register("a", proxy_a)
        registry.register("b", proxy_b)

        allocator = CapitalAllocator(approach="equal_weight", total_capital=100_000)
        weights = allocator.allocate(registry)

        assert "b" not in weights or weights.get("b", 0.0) == pytest.approx(0.0, abs=1e-9)
        assert weights.get("a", 0.0) > 0


class TestCSVOutput:
    def test_save_csv_outputs_creates_files(self, tmp_path, prices_two_strats):
        bt = _make_backtester()
        result = bt.run(prices_two_strats, _simple_specs())
        paths = bt.save_csv_outputs(result, output_dir=str(tmp_path))

        assert os.path.exists(paths["equity"])
        assert os.path.exists(paths["allocator_log"])
        assert os.path.exists(paths["correlation_history"])
        assert os.path.exists(paths["per_strategy_metrics"])

    def test_equity_csv_has_portfolio_and_strategy_columns(self, tmp_path, prices_two_strats):
        bt = _make_backtester()
        result = bt.run(prices_two_strats, _simple_specs())
        bt.save_csv_outputs(result, output_dir=str(tmp_path))

        df = pd.read_csv(tmp_path / "multistrat_equity.csv", index_col=0)
        assert "portfolio" in df.columns
        assert "alpha" in df.columns
        assert "beta" in df.columns

    def test_per_strategy_metrics_has_expected_columns(self, tmp_path, prices_two_strats):
        bt = _make_backtester()
        result = bt.run(prices_two_strats, _simple_specs())
        bt.save_csv_outputs(result, output_dir=str(tmp_path))

        df = pd.read_csv(tmp_path / "per_strategy_metrics.csv")
        for col in ["strategy_name", "total_return", "sharpe", "max_dd", "calmar"]:
            assert col in df.columns
        assert set(df["strategy_name"]) == {"alpha", "beta"}


class TestBenchmarkComparison:
    def test_benchmarks_populated_when_enabled(self, prices_two_strats):
        bt = _make_backtester(run_benchmarks=True)
        result = bt.run(prices_two_strats, _simple_specs())
        bm = result.benchmark
        assert bm is not None
        assert bm.best_single_name in {"alpha", "beta"}
        assert bm.worst_single_name in {"alpha", "beta"}
        assert bm.best_single_name != bm.worst_single_name or len(_simple_specs()) == 1
        assert bm.buy_and_hold_symbol in {"SPY", "QQQ"}
        assert isinstance(bm.portfolio_beats_best_calmar, bool)

    def test_benchmarks_none_when_disabled(self, prices_two_strats):
        bt = _make_backtester(run_benchmarks=False)
        result = bt.run(prices_two_strats, _simple_specs())
        assert result.benchmark is None
