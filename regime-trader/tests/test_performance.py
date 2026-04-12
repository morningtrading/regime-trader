"""
test_performance.py — Unit tests for backtest/performance.py.

Tests cover:
  - PerformanceAnalyzer.compute_sharpe / compute_sortino / compute_cagr /
    compute_calmar / compute_max_drawdown
  - analyze_equity_curve: flat equity, growing equity, drawdown curve
  - compute_regime_breakdown and compute_regime_transitions
  - compare_benchmark
  - Edge cases: single-bar equity, all-positive returns, all-negative returns
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd
import pytest

from backtest.performance import PerformanceAnalyzer, PerformanceReport


# ── Helpers ────────────────────────────────────────────────────────────────────

def _equity(values, start: str = "2022-01-01") -> pd.Series:
    """Build a dated equity Series from a list of values."""
    dates = pd.bdate_range(start, periods=len(values))
    return pd.Series(values, index=dates, dtype=float)


def _growing_equity(n: int = 252, annual_return: float = 0.10) -> pd.Series:
    """Equity curve with constant daily growth."""
    daily = (1.0 + annual_return) ** (1.0 / 252) - 1.0
    values = [100_000.0 * (1.0 + daily) ** i for i in range(n)]
    return _equity(values)


def _flat_equity(n: int = 100) -> pd.Series:
    return _equity([100_000.0] * n)


def _crash_equity(n: int = 200, crash_pct: float = -0.20) -> pd.Series:
    """Equity that drops mid-way then recovers partially."""
    values = [100_000.0] * 50
    # crash over 50 bars
    trough = 100_000.0 * (1.0 + crash_pct)
    values += list(np.linspace(100_000.0, trough, 50))
    # partial recovery
    values += list(np.linspace(trough, trough * 1.05, n - len(values)))
    return _equity(values[:n])


def _analyzer() -> PerformanceAnalyzer:
    return PerformanceAnalyzer(risk_free_rate=0.0, trading_days_per_year=252)


# ── compute_sharpe ─────────────────────────────────────────────────────────────

class TestComputeSharpe:
    def test_positive_for_positive_returns(self) -> None:
        pa = _analyzer()
        returns = pd.Series([0.001] * 252)
        assert pa.compute_sharpe(returns) > 0.0

    def test_zero_for_zero_returns(self) -> None:
        pa = _analyzer()
        returns = pd.Series([0.0] * 100)
        assert pa.compute_sharpe(returns) == 0.0

    def test_negative_for_negative_returns(self) -> None:
        pa = _analyzer()
        returns = pd.Series([-0.001] * 252)
        assert pa.compute_sharpe(returns) < 0.0

    def test_short_series_returns_zero(self) -> None:
        pa = _analyzer()
        returns = pd.Series([0.01])
        assert pa.compute_sharpe(returns) == 0.0

    def test_formula_correctness(self) -> None:
        """Sharpe = mean/std * sqrt(252) when rfr=0."""
        pa = _analyzer()
        rng = np.random.default_rng(1)
        r = pd.Series(rng.normal(0.001, 0.01, 500))
        expected = float(r.mean() / r.std() * np.sqrt(252))
        assert abs(pa.compute_sharpe(r) - expected) < 1e-6


# ── compute_sortino ────────────────────────────────────────────────────────────

class TestComputeSortino:
    def test_positive_for_positive_returns(self) -> None:
        pa = _analyzer()
        returns = pd.Series([0.001] * 252)
        assert pa.compute_sortino(returns) > 0.0

    def test_inf_when_no_downside(self) -> None:
        pa = _analyzer()
        # All positive returns → no downside → inf
        returns = pd.Series([0.002] * 50)
        result = pa.compute_sortino(returns)
        assert math.isinf(result) or result > 100.0

    def test_short_series_returns_zero(self) -> None:
        pa = _analyzer()
        assert pa.compute_sortino(pd.Series([0.01])) == 0.0

    def test_sortino_higher_than_sharpe_for_asymmetric_returns(self) -> None:
        """Sortino ignores upside vol so it should be ≥ Sharpe in typical cases."""
        pa = _analyzer()
        rng = np.random.default_rng(7)
        r = pd.Series(rng.normal(0.002, 0.01, 400))
        assert pa.compute_sortino(r) >= pa.compute_sharpe(r)


# ── compute_cagr ───────────────────────────────────────────────────────────────

class TestComputeCagr:
    def test_positive_for_growing_equity(self) -> None:
        pa = _analyzer()
        eq = _growing_equity(252, annual_return=0.10)
        cagr = pa.compute_cagr(eq)
        assert abs(cagr - 0.10) < 0.01   # within 1%

    def test_zero_for_flat_equity(self) -> None:
        pa = _analyzer()
        eq = _flat_equity(252)
        assert abs(pa.compute_cagr(eq)) < 1e-6

    def test_negative_for_shrinking_equity(self) -> None:
        pa = _analyzer()
        eq = _equity(np.linspace(100_000, 80_000, 252))
        assert pa.compute_cagr(eq) < 0.0

    def test_single_bar_returns_zero(self) -> None:
        pa = _analyzer()
        eq = _equity([100_000.0])
        assert pa.compute_cagr(eq) == 0.0


# ── compute_max_drawdown ───────────────────────────────────────────────────────

class TestComputeMaxDrawdown:
    def test_no_drawdown_for_flat_equity(self) -> None:
        pa = _analyzer()
        eq = _flat_equity(100)
        max_dd, *_ = pa.compute_max_drawdown(eq)
        assert max_dd == 0.0

    def test_max_drawdown_magnitude(self) -> None:
        pa = _analyzer()
        eq = _crash_equity(200, crash_pct=-0.20)
        max_dd, *_ = pa.compute_max_drawdown(eq)
        assert max_dd < 0.0
        assert abs(max_dd) <= 0.21   # within 1% of expected 20% crash

    def test_duration_positive(self) -> None:
        pa = _analyzer()
        eq = _crash_equity(200, crash_pct=-0.15)
        _, dd_start, dd_end, duration = pa.compute_max_drawdown(eq)
        assert duration >= 0

    def test_peak_before_trough(self) -> None:
        pa = _analyzer()
        eq = _crash_equity(200, crash_pct=-0.25)
        _, peak_date, trough_date, _ = pa.compute_max_drawdown(eq)
        assert peak_date <= trough_date


# ── compute_calmar ─────────────────────────────────────────────────────────────

class TestComputeCalmar:
    def test_inf_for_no_drawdown(self) -> None:
        pa = _analyzer()
        eq = _growing_equity(252)
        calmar = pa.compute_calmar(eq)
        # No drawdown on a monotonically growing equity → inf
        assert math.isinf(calmar) or calmar > 0.0

    def test_finite_for_crash_recovery(self) -> None:
        pa = _analyzer()
        eq = _equity(
            [100_000, 90_000, 85_000, 90_000, 95_000, 100_000, 105_000] * 40
        )
        calmar = pa.compute_calmar(eq)
        assert math.isfinite(calmar)


# ── analyze_equity_curve ───────────────────────────────────────────────────────

class TestAnalyzeEquityCurve:
    def test_returns_performance_report(self) -> None:
        pa = _analyzer()
        eq = _growing_equity()
        report = pa.analyze_equity_curve(equity=eq)
        assert isinstance(report, PerformanceReport)

    def test_raises_on_single_bar(self) -> None:
        pa = _analyzer()
        with pytest.raises(ValueError):
            pa.analyze_equity_curve(_equity([100_000.0]))

    def test_total_return_positive_for_growth(self) -> None:
        pa = _analyzer()
        eq = _growing_equity(252, annual_return=0.12)
        report = pa.analyze_equity_curve(eq)
        assert report.total_return > 0.0

    def test_total_return_negative_for_loss(self) -> None:
        pa = _analyzer()
        eq = _equity(np.linspace(100_000, 80_000, 252))
        report = pa.analyze_equity_curve(eq)
        assert report.total_return < 0.0

    def test_max_drawdown_non_positive(self) -> None:
        pa = _analyzer()
        eq = _crash_equity()
        report = pa.analyze_equity_curve(eq)
        assert report.max_drawdown <= 0.0

    def test_win_rate_between_zero_and_one(self) -> None:
        pa = _analyzer()
        eq = _growing_equity()
        report = pa.analyze_equity_curve(eq)
        assert 0.0 <= report.win_rate <= 1.0

    def test_flat_equity_zero_sharpe(self) -> None:
        pa = _analyzer()
        eq = _flat_equity(252)
        report = pa.analyze_equity_curve(eq)
        assert abs(report.sharpe_ratio) < 0.01


# ── compute_regime_breakdown ──────────────────────────────────────────────────

class TestComputeRegimeBreakdown:
    def _returns_regimes(self) -> tuple:
        n = 200
        returns = pd.Series(
            np.random.default_rng(42).normal(0.001, 0.01, n),
            index=pd.bdate_range("2022-01-01", periods=n),
        )
        labels = ["BULL"] * 100 + ["BEAR"] * 100
        regimes = pd.Series(labels, index=returns.index)
        return returns, regimes

    def test_keys_match_labels(self) -> None:
        pa = _analyzer()
        returns, regimes = self._returns_regimes()
        breakdown = pa.compute_regime_breakdown(returns, regimes)
        assert set(breakdown.keys()) == {"BULL", "BEAR"}

    def test_pct_time_sums_to_one(self) -> None:
        pa = _analyzer()
        returns, regimes = self._returns_regimes()
        breakdown = pa.compute_regime_breakdown(returns, regimes)
        total = sum(v["pct_time"] for v in breakdown.values())
        assert abs(total - 1.0) < 1e-6

    def test_n_bars_correct(self) -> None:
        pa = _analyzer()
        returns, regimes = self._returns_regimes()
        breakdown = pa.compute_regime_breakdown(returns, regimes)
        assert breakdown["BULL"]["n_bars"] == 100
        assert breakdown["BEAR"]["n_bars"] == 100

    def test_empty_regimes_returns_empty_dict(self) -> None:
        pa = _analyzer()
        returns = pd.Series([0.001] * 50)
        result = pa.compute_regime_breakdown(returns, pd.Series(dtype=str))
        assert result == {}

    def test_n_entries_counted(self) -> None:
        pa = _analyzer()
        returns, regimes = self._returns_regimes()
        breakdown = pa.compute_regime_breakdown(returns, regimes)
        # Two blocks (one BULL, one BEAR)
        assert breakdown["BULL"]["n_entries"] == 1
        assert breakdown["BEAR"]["n_entries"] == 1


# ── compute_regime_transitions ────────────────────────────────────────────────

class TestComputeRegimeTransitions:
    def test_counts_transitions(self) -> None:
        pa = _analyzer()
        labels = ["BULL"] * 10 + ["BEAR"] * 5 + ["BULL"] * 10
        regimes = pd.Series(labels)
        n, matrix = pa.compute_regime_transitions(regimes)
        assert n == 2
        assert matrix["BULL"]["BEAR"] == 1
        assert matrix["BEAR"]["BULL"] == 1

    def test_no_transitions_for_constant_regime(self) -> None:
        pa = _analyzer()
        regimes = pd.Series(["BULL"] * 50)
        n, matrix = pa.compute_regime_transitions(regimes)
        assert n == 0
        assert matrix == {}

    def test_single_bar_returns_zero(self) -> None:
        pa = _analyzer()
        n, matrix = pa.compute_regime_transitions(pd.Series(["BULL"]))
        assert n == 0
        assert matrix == {}


# ── compare_benchmark ─────────────────────────────────────────────────────────

class TestCompareBenchmark:
    def _series(self, seed: int = 1, n: int = 100) -> tuple:
        rng = np.random.default_rng(seed)
        dates = pd.bdate_range("2022-01-01", periods=n)
        strat = pd.Series(rng.normal(0.001, 0.01, n), index=dates)
        bm = pd.Series(rng.normal(0.0005, 0.008, n), index=dates)
        return strat, bm

    def test_returns_dict_with_expected_keys(self) -> None:
        pa = _analyzer()
        strat, bm = self._series()
        result = pa.compare_benchmark(strat, bm)
        for key in ("alpha", "beta", "information_ratio",
                    "tracking_error", "excess_return"):
            assert key in result

    def test_beta_near_one_for_identical_returns(self) -> None:
        pa = _analyzer()
        r = pd.Series(np.random.default_rng(3).normal(0.001, 0.01, 100))
        result = pa.compare_benchmark(r, r)
        assert abs(result["beta"] - 1.0) < 1e-4

    def test_short_series_returns_defaults(self) -> None:
        pa = _analyzer()
        short = pd.Series([0.01, 0.02])
        result = pa.compare_benchmark(short, short)
        assert result["alpha"] == 0.0
        assert result["beta"] == 1.0

    def test_alpha_and_ir_are_finite(self) -> None:
        pa = _analyzer()
        strat, bm = self._series()
        result = pa.compare_benchmark(strat, bm)
        assert math.isfinite(result["alpha"])
        assert math.isfinite(result["information_ratio"])


# ── analyze_equity_curve with benchmark ───────────────────────────────────────

class TestAnalyzeWithBenchmark:
    def test_benchmark_fields_populated(self) -> None:
        pa = _analyzer()
        n = 252
        eq = _growing_equity(n, 0.10)
        dates = eq.index
        bm_prices = pd.Series(np.linspace(100, 105, n), index=dates)
        bm_returns = bm_prices.pct_change().dropna()
        report = pa.analyze_equity_curve(
            equity=eq,
            benchmark=bm_returns,
        )
        assert report.benchmark_return is not None
        assert report.beta is not None
        assert report.alpha is not None


# ── analyze_equity_curve with regimes ─────────────────────────────────────────

class TestAnalyzeWithRegimes:
    def test_regime_stats_populated(self) -> None:
        pa = _analyzer()
        n = 200
        eq = _growing_equity(n)
        labels = ["BULL"] * 100 + ["BEAR"] * 100
        regimes = pd.Series(labels, index=eq.index[:200])
        report = pa.analyze_equity_curve(equity=eq[:200], regimes=regimes)
        assert "BULL" in report.regime_stats
        assert "BEAR" in report.regime_stats

    def test_regime_transition_count_correct(self) -> None:
        pa = _analyzer()
        n = 200
        eq = _growing_equity(n)
        labels = ["BULL"] * 100 + ["BEAR"] * 100
        regimes = pd.Series(labels, index=eq.index[:200])
        report = pa.analyze_equity_curve(equity=eq[:200], regimes=regimes)
        assert report.regime_n_changes == 1
