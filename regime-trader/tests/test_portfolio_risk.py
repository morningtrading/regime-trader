"""
tests/test_portfolio_risk.py — Unit tests for Phase 3: PortfolioRiskManager.

Covers:
  - Aggregate exposure: two strategies combined over 80% → second rejected/reduced
  - Symbol aggregation: same symbol from two strategies combined over 15% → reduced
  - Total leverage: combined leverage exceeds 1.25x → rejected
  - Portfolio DD: daily and peak thresholds fire correctly
  - Correlation cluster: combined correlated exposure > 30% → rejected
  - Per-strategy risk still fires independently (hierarchy test)
  - Size reductions stay above min_position_usd floor
  - approved signal carries modified position_size_pct
"""

from __future__ import annotations

import copy
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.risk_manager import (
    CircuitBreakerType,
    PortfolioRiskManager,
    PortfolioState,
    RejectionReason,
    RiskDecision,
    RiskManager,
    LOCK_FILE,
)
from core.regime_strategies import Direction, Signal


# ── Helpers ────────────────────────────────────────────────────────────────────

def _signal(
    symbol: str = "SPY",
    size_pct: float = 0.10,
    leverage: float = 1.0,
    entry: float = 400.0,
    stop: float = 380.0,
    strategy_name: str = "strat_a",
) -> Signal:
    return Signal(
        symbol=symbol,
        direction=Direction.LONG,
        confidence=0.85,
        entry_price=entry,
        stop_loss=stop,
        take_profit=None,
        position_size_pct=size_pct,
        leverage=leverage,
        regime_id=0,
        regime_name="BULL",
        regime_probability=0.85,
        timestamp=pd.Timestamp("2024-06-01"),
        reasoning="test",
        strategy_name=strategy_name,
    )


def _state(
    equity: float = 100_000.0,
    positions: dict | None = None,
    daily_start: float = 0.0,
    peak_equity: float = 0.0,
    price_history: dict | None = None,
) -> PortfolioState:
    return PortfolioState(
        equity=equity,
        cash=equity,
        buying_power=equity,
        positions=positions or {},
        daily_start=daily_start if daily_start > 0 else equity,
        peak_equity=peak_equity if peak_equity > 0 else equity,
        price_history=price_history or {},
    )


def _prm(**kwargs) -> PortfolioRiskManager:
    defaults = dict(
        max_aggregate_exposure=0.80,
        max_single_symbol=0.15,
        max_portfolio_leverage=1.25,
        max_corr_cluster=0.30,
        daily_dd_reduce=0.02,
        daily_dd_halt=0.03,
        max_dd_from_peak=0.10,
        min_position_usd=100.0,
    )
    defaults.update(kwargs)
    return PortfolioRiskManager(**defaults)


@pytest.fixture(autouse=True)
def remove_lock_file():
    """Ensure no stale lock file before/after each test."""
    if LOCK_FILE.exists():
        LOCK_FILE.unlink()
    yield
    if LOCK_FILE.exists():
        LOCK_FILE.unlink()


# ── Aggregate exposure ─────────────────────────────────────────────────────────

class TestAggregateExposure:
    def test_under_limit_passes(self):
        prm = _prm()
        # Strategy A holds 40%
        prm.update_strategy_positions("strat_a", {"SPY": 40_000})
        # Strategy B wants 30% → total 70% < 80% limit
        sig = _signal("QQQ", size_pct=0.30)
        result = prm.validate_signal(sig, "strat_b", _state(100_000))
        assert result.approved

    def test_over_limit_reduces_size(self):
        prm = _prm()
        # Strategy A already holds 75%
        prm.update_strategy_positions("strat_a", {"SPY": 75_000})
        # Strategy B wants 20% → total 95% > 80%
        sig = _signal("QQQ", size_pct=0.20)
        result = prm.validate_signal(sig, "strat_b", _state(100_000))
        assert result.approved
        # Remaining headroom = 80% - 75% = 5%
        assert result.modified_signal.position_size_pct == pytest.approx(0.05, abs=1e-5)

    def test_no_headroom_rejects(self):
        prm = _prm(min_position_usd=5_000)   # large min makes 5% headroom insufficient
        prm.update_strategy_positions("strat_a", {"SPY": 78_000})
        sig = _signal("QQQ", size_pct=0.10)
        result = prm.validate_signal(sig, "strat_b", _state(100_000))
        assert not result.approved
        assert RejectionReason.PORTFOLIO_EXPOSURE.value in result.rejection_reason

    def test_two_strategies_combined_exposure(self):
        prm = _prm()
        prm.update_strategy_positions("strat_a", {"SPY": 50_000})
        prm.update_strategy_positions("strat_b", {"QQQ": 25_000})
        # Total current = 75%. Adding 10% → 85% > 80%
        sig = _signal("IWM", size_pct=0.10)
        result = prm.validate_signal(sig, "strat_c", _state(100_000))
        assert result.approved
        assert result.modified_signal.position_size_pct == pytest.approx(0.05, abs=1e-5)


# ── Symbol concentration ───────────────────────────────────────────────────────

class TestSymbolAggregation:
    def test_under_cap_passes(self):
        prm = _prm()
        prm.update_strategy_positions("strat_a", {"SPY": 8_000})  # 8%
        sig = _signal("SPY", size_pct=0.05)   # +5% → 13% < 15%
        result = prm.validate_signal(sig, "strat_b", _state(100_000))
        assert result.approved

    def test_over_cap_reduces_to_headroom(self):
        prm = _prm()
        prm.update_strategy_positions("strat_a", {"SPY": 10_000})  # 10%
        sig = _signal("SPY", size_pct=0.10)   # +10% → 20% > 15%
        result = prm.validate_signal(sig, "strat_b", _state(100_000))
        assert result.approved
        # Headroom = 15% - 10% = 5%
        assert result.modified_signal.position_size_pct == pytest.approx(0.05, abs=1e-5)

    def test_at_cap_rejects(self):
        prm = _prm(min_position_usd=5_000)   # force rejection via min floor
        prm.update_strategy_positions("strat_a", {"SPY": 15_000})  # already at 15%
        sig = _signal("SPY", size_pct=0.05)
        result = prm.validate_signal(sig, "strat_b", _state(100_000))
        assert not result.approved
        assert RejectionReason.PORTFOLIO_SYMBOL_CAP.value in result.rejection_reason

    def test_different_symbols_not_affected(self):
        prm = _prm()
        prm.update_strategy_positions("strat_a", {"SPY": 14_000})  # SPY near cap
        sig = _signal("QQQ", size_pct=0.14)   # QQQ is independent
        result = prm.validate_signal(sig, "strat_b", _state(100_000))
        assert result.approved


# ── Total leverage ─────────────────────────────────────────────────────────────

class TestTotalLeverage:
    def test_under_leverage_passes(self):
        prm = _prm()
        # Strategy A: 60% at 1.0x → gross = 60%
        prm.update_strategy_positions("strat_a", {"SPY": 60_000})
        sig = _signal("QQQ", size_pct=0.50, leverage=1.0)
        # gross after: 60% + 50% = 110% < 125%
        result = prm.validate_signal(sig, "strat_b", _state(100_000))
        assert result.approved

    def test_combined_leverage_exceeds_limit_rejects(self):
        prm = _prm()
        # No current positions → aggregate (14% < 80%) and symbol (14% < 15%) pass.
        # Signal: size_pct=0.14, leverage=10.0
        # Effective leverage = 0.14 * 10.0 = 1.40x > 1.25x limit → rejected.
        sig = _signal("QQQ", size_pct=0.14, leverage=10.0)
        result = prm.validate_signal(sig, "strat_b", _state(100_000))
        assert not result.approved
        assert RejectionReason.PORTFOLIO_LEVERAGE.value in result.rejection_reason

    def test_leveraged_addition_within_limit_passes(self):
        prm = _prm()
        prm.update_strategy_positions("strat_a", {"SPY": 50_000})
        # 50% + 60% * 1.25 = 50% + 75% = 125% ≤ 125% limit
        sig = _signal("QQQ", size_pct=0.60, leverage=1.25)
        result = prm.validate_signal(sig, "strat_b", _state(100_000))
        assert result.approved


# ── Portfolio drawdown ─────────────────────────────────────────────────────────

class TestPortfolioDD:
    def test_no_dd_passes(self):
        prm = _prm()
        state = _state(100_000, daily_start=100_000, peak_equity=100_000)
        sig = _signal()
        result = prm.validate_signal(sig, "strat_a", state)
        assert result.approved

    def test_daily_dd_reduce_halves_size(self):
        prm = _prm()
        # Daily DD = -2.5% (between reduce=2% and halt=3%)
        state = _state(97_500, daily_start=100_000, peak_equity=100_000)
        sig = _signal(size_pct=0.10)
        result = prm.validate_signal(sig, "strat_a", state)
        assert result.approved
        assert result.modified_signal.position_size_pct == pytest.approx(0.05, abs=1e-5)

    def test_daily_dd_halt_rejects(self):
        prm = _prm()
        # Daily DD = -3.5% (> halt=3%)
        state = _state(96_500, daily_start=100_000, peak_equity=100_000)
        sig = _signal()
        result = prm.validate_signal(sig, "strat_a", state)
        assert not result.approved
        assert RejectionReason.PORTFOLIO_DD_HALT.value in result.rejection_reason

    def test_peak_dd_halt_rejects_and_writes_lock(self):
        prm = _prm()
        # Peak DD = -11% (> max_dd_from_peak=10%)
        state = _state(89_000, daily_start=89_000, peak_equity=100_000)
        sig = _signal()
        result = prm.validate_signal(sig, "strat_a", state)
        assert not result.approved
        assert RejectionReason.PORTFOLIO_PEAK_DD.value in result.rejection_reason
        assert LOCK_FILE.exists()

    def test_peak_dd_just_under_limit_passes(self):
        prm = _prm()
        # Peak DD = -9.9% (< 10%)
        state = _state(90_100, daily_start=90_100, peak_equity=100_000)
        result = prm.validate_signal(_signal(), "strat_a", state)
        assert result.approved


# ── Correlation cluster ────────────────────────────────────────────────────────

class TestCorrelationCluster:
    def _make_correlated_prices(self, n=100):
        """Generate two highly correlated price series."""
        rng = np.random.default_rng(42)
        base = np.cumsum(rng.normal(0, 1, n)) + 100
        spy = pd.Series(base,             name="SPY")
        qqq = pd.Series(base * 1.01 + rng.normal(0, 0.1, n), name="QQQ")
        return spy, qqq

    def test_no_history_passes(self):
        prm = _prm()
        prm.update_strategy_positions("strat_a", {"SPY": 20_000})
        sig = _signal("QQQ", size_pct=0.15)
        result = prm.validate_signal(sig, "strat_b", _state(100_000))
        assert result.approved   # no price_history → skip cluster check

    def test_correlated_cluster_over_cap_rejects(self):
        prm = _prm(max_corr_cluster=0.30)
        spy_prices, qqq_prices = self._make_correlated_prices()
        # Strategy A holds 25% in SPY (correlated with QQQ)
        prm.update_strategy_positions("strat_a", {"SPY": 25_000})
        state = _state(
            100_000,
            price_history={"SPY": spy_prices, "QQQ": qqq_prices},
        )
        # Adding QQQ 10% → cluster = 25% + 10% = 35% > 30%
        sig = _signal("QQQ", size_pct=0.10)
        result = prm.validate_signal(sig, "strat_b", state)
        assert not result.approved
        assert RejectionReason.PORTFOLIO_CORR_CLUSTER.value in result.rejection_reason

    def test_uncorrelated_symbols_not_clustered(self):
        prm = _prm(max_corr_cluster=0.30)
        rng = np.random.default_rng(7)
        n = 100
        spy_p = pd.Series(np.cumsum(rng.normal(0, 1, n)) + 100)
        gld_p = pd.Series(np.cumsum(rng.normal(0, 1, n)) + 150)   # independent
        prm.update_strategy_positions("strat_a", {"SPY": 25_000})
        state = _state(
            100_000,
            price_history={"SPY": spy_p, "GLD": gld_p},
        )
        sig = _signal("GLD", size_pct=0.10)
        result = prm.validate_signal(sig, "strat_b", state)
        # GLD uncorrelated with SPY → no cluster breach
        assert result.approved


# ── Hierarchy: per-strategy risk fires first ───────────────────────────────────

class TestHierarchy:
    def test_per_strategy_rejection_blocks_before_portfolio(self):
        """The per-strategy RiskManager rejects first; PortfolioRiskManager never runs."""
        per_strategy_rm = RiskManager(initial_equity=100_000)

        # Force circuit breaker into HALTED state
        per_strategy_rm.circuit_breaker._active = CircuitBreakerType.DAILY_HALT

        sig = _signal()
        state = _state(100_000)

        # Per-strategy check
        per_decision = per_strategy_rm.validate_signal(sig, state)
        assert not per_decision.approved

        # Portfolio check would pass on its own (no DD, no exposure issues)
        prm = _prm()
        portfolio_decision = prm.validate_signal(sig, "strat_a", state)
        assert portfolio_decision.approved

        # Combined: per-strategy rejection means the order never reaches portfolio
        final_approved = per_decision.approved and portfolio_decision.approved
        assert not final_approved

    def test_both_must_approve(self):
        """Signal approved per-strategy but rejected by portfolio → order blocked."""
        per_strategy_rm = RiskManager(initial_equity=100_000)
        prm = _prm()

        # Portfolio already at 78% exposure
        prm.update_strategy_positions("strat_a", {"SPY": 78_000})

        sig = _signal("QQQ", size_pct=0.10)
        state = PortfolioState(
            equity=100_000, cash=100_000, buying_power=100_000,
            positions={}, daily_start=100_000, peak_equity=100_000,
        )

        per_decision = per_strategy_rm.validate_signal(sig, state)
        assert per_decision.approved   # per-strategy: fine

        portfolio_decision = prm.validate_signal(
            per_decision.modified_signal or sig, "strat_a", state
        )
        # Portfolio: 78% + 10% = 88% > 80% → reduced to 2%
        # 2% * 100k = $2000 > min_position_usd $100 → still approved but reduced
        assert portfolio_decision.approved
        assert portfolio_decision.modified_signal.position_size_pct < 0.10


# ── update_strategy_positions ──────────────────────────────────────────────────

class TestPositionTracking:
    def test_aggregate_positions_sums_across_strategies(self):
        prm = _prm()
        prm.update_strategy_positions("a", {"SPY": 10_000, "QQQ": 5_000})
        prm.update_strategy_positions("b", {"SPY": 8_000,  "GLD": 3_000})
        agg = prm.get_aggregate_positions()
        assert agg["SPY"] == pytest.approx(18_000)
        assert agg["QQQ"] == pytest.approx(5_000)
        assert agg["GLD"] == pytest.approx(3_000)

    def test_update_replaces_not_merges(self):
        prm = _prm()
        prm.update_strategy_positions("a", {"SPY": 10_000})
        prm.update_strategy_positions("a", {"SPY": 5_000})   # update, not add
        assert prm.get_aggregate_positions()["SPY"] == pytest.approx(5_000)

    def test_empty_registry_returns_empty_agg(self):
        prm = _prm()
        assert prm.get_aggregate_positions() == {}


# ── from_config ────────────────────────────────────────────────────────────────

class TestFromConfig:
    def test_builds_from_risk_section(self):
        config = {
            "risk": {
                "max_exposure":        0.75,
                "max_single_position": 0.12,
                "max_leverage":        1.10,
                "daily_dd_reduce":     0.015,
                "daily_dd_halt":       0.025,
                "max_dd_from_peak":    0.08,
            }
        }
        prm = PortfolioRiskManager.from_config(config)
        assert prm.max_aggregate_exposure == pytest.approx(0.75)
        assert prm.max_single_symbol      == pytest.approx(0.12)
        assert prm.max_portfolio_leverage == pytest.approx(1.10)
        assert prm.daily_dd_reduce        == pytest.approx(0.015)
        assert prm.max_dd_from_peak       == pytest.approx(0.08)
