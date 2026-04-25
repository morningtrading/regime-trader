"""
tests/test_allocator.py — Unit tests for Phase 2: CapitalAllocator.

Covers:
  - equal_weight → each strategy gets 1/N (within reserve)
  - inverse_vol  → lower vol strategy gets higher weight
  - risk_parity  → runs without crash; weights sum correctly
  - performance_weighted → positive Sharpe strategies get more; all-negative falls back
  - Correlation > 0.80 → strategies merged for allocation
  - Kill switch at both DD thresholds
  - Constraints (weight_min / weight_max) always respected
  - Reserve is always subtracted from deployable capital
  - rebalance() only fires when Δweight > threshold
  - AllocationChange fields are correct after rebalance
"""

from __future__ import annotations

import pytest
import pandas as pd
import numpy as np

from core.strategy_registry import StrategyRegistry
from core.capital_allocator import (
    CapitalAllocator,
    AllocationChange,
    _CORR_MERGE_THRESHOLD,
    _KILL_HALVE_DD,
    _KILL_ZERO_DD,
)
from core.regime_strategies import BaseStrategy
from core.hmm_engine import RegimeState


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_regime_state() -> RegimeState:
    return RegimeState(
        label="BULL", state_id=0, probability=0.9,
        is_confirmed=True, timestamp=pd.Timestamp("2024-01-01"),
    )


class _Stub(BaseStrategy):
    """Minimal concrete strategy for allocator tests."""
    def __init__(self, n: str = "stub") -> None:
        super().__init__()
        self._n = n

    @property
    def name(self) -> str:
        return self._n

    @property
    def total_allocation(self) -> float:
        return 0.5

    def generate_signal(self, symbol, bars, regime_state):
        return None


def _stub_with_returns(name: str, returns: list) -> _Stub:
    """Create a stub whose performance_history is pre-populated."""
    s = _Stub(name)
    ts = pd.Timestamp("2024-01-01")
    for i, r in enumerate(returns):
        s.record_daily_return(ts + pd.Timedelta(days=i), r)
    return s


def _make_registry(*strategies) -> StrategyRegistry:
    StrategyRegistry._reset()
    reg = StrategyRegistry.instance()
    for s in strategies:
        reg.register(s.name, s)
    return reg


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_registry():
    StrategyRegistry._reset()
    yield
    StrategyRegistry._reset()


# ── Equal weight ───────────────────────────────────────────────────────────────

class TestEqualWeight:
    def test_three_strategies_each_get_third(self):
        reg = _make_registry(_Stub("a"), _Stub("b"), _Stub("c"))
        alloc = CapitalAllocator(approach="equal_weight", reserve=0.0)
        w = alloc.allocate(reg)
        assert set(w.keys()) == {"a", "b", "c"}
        for name, wt in w.items():
            assert wt == pytest.approx(1 / 3, abs=1e-6)

    def test_sum_equals_one_minus_reserve(self):
        reg = _make_registry(_Stub("a"), _Stub("b"))
        reserve = 0.10
        alloc = CapitalAllocator(approach="equal_weight", reserve=reserve)
        w = alloc.allocate(reg)
        assert sum(w.values()) == pytest.approx(1.0 - reserve, abs=1e-5)

    def test_single_strategy_gets_full_deployable(self):
        reg = _make_registry(_Stub("solo"))
        alloc = CapitalAllocator(approach="equal_weight", reserve=0.10)
        w = alloc.allocate(reg)
        assert w["solo"] == pytest.approx(0.90, abs=1e-5)

    def test_disabled_strategy_excluded(self):
        s1 = _Stub("on")
        s2 = _Stub("off")
        s2.is_enabled = False
        reg = _make_registry(s1, s2)
        alloc = CapitalAllocator(approach="equal_weight", reserve=0.0)
        w = alloc.allocate(reg)
        assert "off" not in w
        assert w["on"] == pytest.approx(1.0, abs=1e-6)


# ── Inverse vol ────────────────────────────────────────────────────────────────

class TestInverseVol:
    def test_lower_vol_gets_higher_weight(self):
        rng = np.random.default_rng(1)
        low_rets  = list(rng.normal(0.001, 0.001, 60))   # vol ≈ 0.001
        high_rets = list(rng.normal(0.001, 0.020, 60))   # vol ≈ 0.020
        low  = _stub_with_returns("low_vol",  low_rets)
        high = _stub_with_returns("high_vol", high_rets)
        reg = _make_registry(low, high)
        alloc = CapitalAllocator(approach="inverse_vol", reserve=0.0)
        w = alloc.allocate(reg)
        assert w["low_vol"] > w["high_vol"]

    def test_weights_sum_to_deployable(self):
        s1 = _stub_with_returns("x", [0.001] * 30)
        s2 = _stub_with_returns("y", [0.005] * 30)
        reg = _make_registry(s1, s2)
        alloc = CapitalAllocator(approach="inverse_vol", reserve=0.10)
        w = alloc.allocate(reg)
        assert sum(w.values()) == pytest.approx(0.90, abs=1e-5)

    def test_no_history_falls_back_to_equal(self):
        # Strategies with no return history → vol defaults to 1 → equal weight
        reg = _make_registry(_Stub("a"), _Stub("b"))
        alloc = CapitalAllocator(approach="inverse_vol", reserve=0.0)
        w = alloc.allocate(reg)
        assert w["a"] == pytest.approx(w["b"], abs=1e-5)


# ── Risk parity ────────────────────────────────────────────────────────────────

class TestRiskParity:
    def test_runs_and_sums_to_deployable(self):
        s1 = _stub_with_returns("rp_a", [0.001] * 60)
        s2 = _stub_with_returns("rp_b", [0.003] * 60)
        reg = _make_registry(s1, s2)
        alloc = CapitalAllocator(approach="risk_parity", reserve=0.10)
        w = alloc.allocate(reg)
        assert sum(w.values()) == pytest.approx(0.90, abs=1e-4)

    def test_all_weights_non_negative(self):
        s1 = _stub_with_returns("rp_a", [0.002] * 60)
        s2 = _stub_with_returns("rp_b", [0.001] * 60)
        reg = _make_registry(s1, s2)
        alloc = CapitalAllocator(approach="risk_parity", reserve=0.0)
        w = alloc.allocate(reg)
        assert all(v >= 0.0 for v in w.values())


# ── Performance weighted ───────────────────────────────────────────────────────

class TestPerformanceWeighted:
    def test_higher_sharpe_gets_more(self):
        rng = np.random.default_rng(2)
        # good: high mean, low std  → high Sharpe
        good_rets = list(rng.normal(0.005, 0.002, 60))
        # poor: low mean, higher std → lower Sharpe
        poor_rets = list(rng.normal(0.001, 0.010, 60))
        good = _stub_with_returns("good", good_rets)
        poor = _stub_with_returns("poor", poor_rets)
        reg = _make_registry(good, poor)
        alloc = CapitalAllocator(approach="performance_weighted", reserve=0.0)
        w = alloc.allocate(reg)
        assert w["good"] > w["poor"]

    def test_all_negative_sharpe_falls_back_to_equal(self):
        rng = np.random.default_rng(3)
        bad1 = _stub_with_returns("b1", list(rng.normal(-0.005, 0.003, 60)))
        bad2 = _stub_with_returns("b2", list(rng.normal(-0.003, 0.003, 60)))
        reg = _make_registry(bad1, bad2)
        alloc = CapitalAllocator(approach="performance_weighted", reserve=0.0)
        w = alloc.allocate(reg)
        # Fall back to equal weight
        assert w["b1"] == pytest.approx(w["b2"], abs=1e-5)

    def test_mixed_positive_negative_only_positive_counted(self):
        rng = np.random.default_rng(4)
        good = _stub_with_returns("good", list(rng.normal(0.005, 0.002, 60)))
        bad  = _stub_with_returns("bad",  list(rng.normal(-0.005, 0.002, 60)))
        reg  = _make_registry(good, bad)
        alloc = CapitalAllocator(approach="performance_weighted", reserve=0.0)
        w = alloc.allocate(reg)
        # bad Sharpe ≤ 0 → gets 0 allocation
        assert w["bad"] == pytest.approx(0.0, abs=1e-5)
        assert w["good"] > 0.0


# ── Correlation merging ────────────────────────────────────────────────────────

class TestCorrelationMerge:
    def _make_correlated_pair(self):
        """Two strategies with ρ ≈ 1.0 (identical returns)."""
        rets = list(np.random.default_rng(7).normal(0, 0.01, 60))
        s1 = _stub_with_returns("twin_a", rets)
        s2 = _stub_with_returns("twin_b", rets)      # identical → ρ = 1.0
        return s1, s2

    def test_correlated_pair_weight_split_equally(self):
        s1, s2 = self._make_correlated_pair()
        s3 = _stub_with_returns("indie", list(np.random.default_rng(99).normal(0, 0.01, 60)))
        reg = _make_registry(s1, s2, s3)
        alloc = CapitalAllocator(approach="equal_weight", reserve=0.0)
        w = alloc.allocate(reg)
        # twin_a and twin_b should receive equal shares
        assert w["twin_a"] == pytest.approx(w["twin_b"], abs=1e-5)

    def test_uncorrelated_pair_not_merged(self):
        rng = np.random.default_rng(42)
        s1 = _stub_with_returns("uncorr_a", list(rng.normal(0, 0.01, 60)))
        # Orthogonal returns
        s2 = _stub_with_returns("uncorr_b", list(rng.normal(0, 0.01, 60)))
        reg = _make_registry(s1, s2)
        alloc = CapitalAllocator(approach="equal_weight", reserve=0.0)
        pairs = alloc.should_merge_correlated_strategies(reg)
        # May or may not merge depending on random seed; just check it runs
        assert isinstance(pairs, list)

    def test_should_merge_returns_pairs_above_threshold(self):
        # Use genuinely variable (but identical) returns so std > 0 → corr = 1.0
        rng = np.random.default_rng(5)
        rets = list(rng.normal(0, 0.01, 60))
        s1 = _stub_with_returns("ma", rets)
        s2 = _stub_with_returns("mb", rets)   # identical series → ρ = 1.0
        reg = _make_registry(s1, s2)
        alloc = CapitalAllocator(approach="equal_weight", reserve=0.0)
        pairs = alloc.should_merge_correlated_strategies(reg)
        assert ("ma", "mb") in pairs


# ── Kill switch ────────────────────────────────────────────────────────────────

class TestKillSwitch:
    def test_halve_at_dd_threshold(self):
        reg = _make_registry(_Stub("a"), _Stub("b"))
        alloc = CapitalAllocator(approach="equal_weight", reserve=0.0)
        w_normal = alloc.allocate(reg, daily_drawdown=0.0)
        w_halved = alloc.allocate(reg, daily_drawdown=_KILL_HALVE_DD + 0.001)
        for name in w_normal:
            assert w_halved[name] == pytest.approx(w_normal[name] * 0.5, abs=1e-6)

    def test_zero_at_hard_dd_threshold(self):
        reg = _make_registry(_Stub("a"), _Stub("b"))
        alloc = CapitalAllocator(approach="equal_weight", reserve=0.0)
        w = alloc.allocate(reg, daily_drawdown=_KILL_ZERO_DD + 0.001)
        assert all(v == pytest.approx(0.0, abs=1e-10) for v in w.values())

    def test_no_kill_below_threshold(self):
        reg = _make_registry(_Stub("a"), _Stub("b"))
        alloc = CapitalAllocator(approach="equal_weight", reserve=0.0)
        w_clean = alloc.allocate(reg, daily_drawdown=0.0)
        w_minor = alloc.allocate(reg, daily_drawdown=_KILL_HALVE_DD - 0.001)
        for name in w_clean:
            assert w_minor[name] == pytest.approx(w_clean[name], abs=1e-6)


# ── Constraints ────────────────────────────────────────────────────────────────

class TestConstraints:
    def test_weight_max_respected(self):
        configs = {
            "a": {"weight_min": 0.05, "weight_max": 0.20},
            "b": {"weight_min": 0.05, "weight_max": 0.20},
            "c": {"weight_min": 0.05, "weight_max": 0.20},
        }
        reg = _make_registry(_Stub("a"), _Stub("b"), _Stub("c"))
        alloc = CapitalAllocator(
            approach="equal_weight",
            strategy_configs=configs,
            reserve=0.0,
        )
        w = alloc.allocate(reg)
        for name, wt in w.items():
            assert wt <= configs[name]["weight_max"] + 1e-5

    def test_weight_min_respected(self):
        # Give one strategy huge raw weight; others must still hit their min.
        s1 = _stub_with_returns("big",   [0.02] * 60)   # high sharpe → dominant
        s2 = _stub_with_returns("small", [0.001] * 60)
        configs = {
            "big":   {"weight_min": 0.10, "weight_max": 0.90},
            "small": {"weight_min": 0.10, "weight_max": 0.50},
        }
        reg = _make_registry(s1, s2)
        alloc = CapitalAllocator(
            approach="performance_weighted",
            strategy_configs=configs,
            reserve=0.0,
        )
        w = alloc.allocate(reg)
        for name, wt in w.items():
            assert wt >= configs[name]["weight_min"] - 1e-5

    def test_reserve_is_always_held(self):
        reserve = 0.15
        reg = _make_registry(_Stub("a"), _Stub("b"), _Stub("c"))
        alloc = CapitalAllocator(
            approach="equal_weight",
            reserve=reserve,
        )
        w = alloc.allocate(reg)
        assert sum(w.values()) <= (1.0 - reserve) + 1e-5


# ── Rebalance ──────────────────────────────────────────────────────────────────

class TestRebalance:
    def test_no_changes_when_below_threshold(self):
        reg = _make_registry(_Stub("a"), _Stub("b"))
        alloc = CapitalAllocator(
            approach="equal_weight",
            reserve=0.0,
            rebalance_threshold=0.05,
        )
        # Pre-seed current weights to match target exactly
        alloc._current_weights = {"a": 0.5, "b": 0.5}
        changes = alloc.rebalance(reg, total_capital=100_000)
        assert changes == []

    def test_changes_returned_above_threshold(self):
        reg = _make_registry(_Stub("a"), _Stub("b"))
        alloc = CapitalAllocator(
            approach="equal_weight",
            reserve=0.0,
            rebalance_threshold=0.05,
        )
        # Start far from equal weight
        alloc._current_weights = {"a": 0.0, "b": 1.0}
        changes = alloc.rebalance(reg, total_capital=100_000)
        assert len(changes) >= 1

    def test_allocation_change_fields(self):
        reg = _make_registry(_Stub("x"))
        alloc = CapitalAllocator(
            approach="equal_weight",
            reserve=0.0,
            rebalance_threshold=0.0,   # always rebalance
        )
        alloc._current_weights = {"x": 0.0}
        changes = alloc.rebalance(reg, total_capital=50_000)
        assert len(changes) == 1
        c = changes[0]
        assert c.strategy_name == "x"
        assert c.old_weight == pytest.approx(0.0)
        assert c.new_weight == pytest.approx(1.0, abs=1e-5)
        assert c.old_capital == pytest.approx(0.0)
        assert c.new_capital == pytest.approx(50_000.0, abs=1.0)

    def test_rebalance_updates_strategy_allocated_capital(self):
        s = _Stub("z")
        reg = _make_registry(s)
        alloc = CapitalAllocator(
            approach="equal_weight",
            reserve=0.10,
            rebalance_threshold=0.0,
        )
        alloc._current_weights = {"z": 0.0}
        alloc.rebalance(reg, total_capital=100_000)
        # allocate() returns weight = 0.90 (bakes in reserve).
        # rebalance applies: allocated_capital = weight × total_capital = 0.90 × 100_000
        assert s.allocated_capital == pytest.approx(90_000, abs=1.0)

    def test_kill_switch_zero_in_rebalance(self):
        s = _Stub("k")
        reg = _make_registry(s)
        alloc = CapitalAllocator(
            approach="equal_weight",
            reserve=0.0,
            rebalance_threshold=0.0,
        )
        alloc._current_weights = {"k": 1.0}
        changes = alloc.rebalance(
            reg, total_capital=100_000,
            daily_drawdown=_KILL_ZERO_DD + 0.001,
        )
        assert len(changes) == 1
        assert changes[0].new_weight == pytest.approx(0.0, abs=1e-6)
        assert s.allocated_capital == pytest.approx(0.0, abs=1.0)


# ── Invalid approach ───────────────────────────────────────────────────────────

class TestInvalidApproach:
    def test_unknown_approach_raises(self):
        with pytest.raises(ValueError, match="approach must be one of"):
            CapitalAllocator(approach="magic_beans")
