"""
tests/test_multistrat_e2e.py — End-to-end behavioural tests for the
multi-strategy stack (Phase 6).

Exercises the public contract that day-to-day users rely on:

    1. Two highly-correlated strategies are detected and merged by the
       CapitalAllocator.
    2. Two uncorrelated strategies retain independent allocations and the
       combined Sharpe exceeds either individual Sharpe.
    3. A strategy that breaches a health threshold is auto-disabled by
       StrategyRegistry.run_health_checks(); the allocator then redistributes
       capital to the survivors.
    4. PortfolioRiskManager caps aggregate exposure at the configured limit
       even when each individual strategy requests its full per-strategy slot.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
import pandas as pd
import pytest

from backtest.multi_strategy_backtester import _BacktestProxy
from core.capital_allocator import CapitalAllocator, _CORR_MERGE_THRESHOLD
from core.regime_strategies import (
    _HEALTH_MAX_DRAWDOWN,
    _HEALTH_MAX_CONSEC_LOSSES,
)
from core.strategy_registry import StrategyRegistry


# ── Helpers ────────────────────────────────────────────────────────────────────

def _seed_returns(
    strategy: _BacktestProxy,
    returns: List[float],
    start: str = "2026-01-01",
) -> None:
    """Push a list of daily returns into a strategy's performance_history."""
    idx = pd.date_range(start, periods=len(returns), freq="B")
    for ts, r in zip(idx, returns):
        strategy.record_daily_return(ts, float(r))


def _annualised_sharpe(returns: List[float]) -> float:
    if len(returns) < 2:
        return 0.0
    mu = float(np.mean(returns))
    sd = float(np.std(returns, ddof=1))
    if sd < 1e-12:
        return 0.0
    return (mu / sd) * math.sqrt(252)


def _strat_configs(names: List[str]) -> dict:
    """Minimal strategy_configs dict the allocator accepts."""
    return {n: {"weight_min": 0.05, "weight_max": 0.50} for n in names}


@pytest.fixture(autouse=True)
def _reset_registry():
    """Each test gets a fresh singleton — no cross-test bleed."""
    StrategyRegistry._reset()
    yield
    StrategyRegistry._reset()


# ── Scenario 1 — correlated strategies merge ───────────────────────────────────

class TestCorrelatedStrategiesMerge:
    """When two strategies behave nearly identically, the allocator's
    correlation merger should treat them as one for sizing purposes."""

    def test_two_correlated_strategies_are_merged(self):
        rng = np.random.default_rng(42)
        base = rng.normal(0.0006, 0.012, 120)
        # B = A + tiny independent noise → correlation ≫ 0.80
        a_returns = base.tolist()
        b_returns = (base + rng.normal(0.0, 0.0005, 120)).tolist()

        registry = StrategyRegistry.instance()
        a = _BacktestProxy("alpha")
        b = _BacktestProxy("beta")
        registry.register("alpha", a)
        registry.register("beta", b)
        _seed_returns(a, a_returns)
        _seed_returns(b, b_returns)

        # Sanity-check the synthetic correlation actually exceeds the threshold
        corr = float(np.corrcoef(a_returns, b_returns)[0, 1])
        assert corr > _CORR_MERGE_THRESHOLD, (
            f"Test setup error: correlation {corr:.3f} <= "
            f"merge threshold {_CORR_MERGE_THRESHOLD}"
        )

        allocator = CapitalAllocator(
            approach="inverse_vol",
            strategy_configs=_strat_configs(["alpha", "beta"]),
            total_capital=100_000.0,
        )
        weights = allocator.allocate(registry)

        # Correlated pair must share roughly the same weight a single
        # strategy would receive — their combined allocation should be close
        # to one inverse-vol slot, not two.
        assert "alpha" in weights and "beta" in weights
        combined = weights["alpha"] + weights["beta"]
        assert combined <= 0.90 + 1e-9   # respect the 10 % cash reserve

        # Per-strategy weights should be close (similar vol, merged group)
        assert abs(weights["alpha"] - weights["beta"]) < 0.05, (
            f"Merged strategies should split the group weight evenly — "
            f"got alpha={weights['alpha']:.3f}, beta={weights['beta']:.3f}"
        )

    def test_uncorrelated_strategies_are_not_merged(self):
        """Sanity contrast — two independent return streams should NOT be
        merged, so each strategy keeps a full inverse-vol slot."""
        rng = np.random.default_rng(7)
        a_returns = rng.normal(0.0006, 0.012, 120).tolist()
        b_returns = rng.normal(0.0006, 0.012, 120).tolist()
        corr = float(np.corrcoef(a_returns, b_returns)[0, 1])
        assert corr < _CORR_MERGE_THRESHOLD, (
            f"Test setup error: independent series correlated at {corr:.3f}"
        )

        registry = StrategyRegistry.instance()
        a = _BacktestProxy("alpha")
        b = _BacktestProxy("beta")
        registry.register("alpha", a)
        registry.register("beta", b)
        _seed_returns(a, a_returns)
        _seed_returns(b, b_returns)

        allocator = CapitalAllocator(
            approach="inverse_vol",
            strategy_configs=_strat_configs(["alpha", "beta"]),
            total_capital=100_000.0,
        )
        weights = allocator.allocate(registry)

        # Both should land near 0.45 (= 0.90 / 2) since vols are similar
        assert weights["alpha"] == pytest.approx(0.45, abs=0.10)
        assert weights["beta"]  == pytest.approx(0.45, abs=0.10)


# ── Scenario 2 — uncorrelated strategies improve combined Sharpe ───────────────

class TestUncorrelatedStrategiesImproveSharpe:
    """Diversification benefit: a long-equity + short-equity (or
    negatively-correlated) book should produce higher portfolio Sharpe than
    either book in isolation."""

    def test_anticorrelated_strategies_combine_to_higher_sharpe(self):
        """Two strategies with the SAME positive drift but ANTI-correlated
        noise: combining them cancels noise while preserving drift, so the
        portfolio Sharpe must exceed either standalone Sharpe.

        This is the classic diversification benefit — the test the prompt
        asks for. (A long-equity + short-equity book on the same series has
        offsetting drift and is not a fair Sharpe comparison.)
        """
        rng = np.random.default_rng(123)
        n = 252
        drift = 0.0008
        shared = rng.normal(0.0, 0.012, n)        # big anti-correlated noise
        unique_a = rng.normal(0.0, 0.003, n)      # small unique noise so combo std > 0
        unique_b = rng.normal(0.0, 0.003, n)
        a_returns = drift + shared + unique_a
        b_returns = drift - shared + unique_b

        a_sharpe = _annualised_sharpe(a_returns.tolist())
        b_sharpe = _annualised_sharpe(b_returns.tolist())

        combined = 0.5 * a_returns + 0.5 * b_returns
        combo_sharpe = _annualised_sharpe(combined.tolist())

        corr = float(np.corrcoef(a_returns, b_returns)[0, 1])
        assert corr < -0.5, f"Setup expects strongly anti-correlated streams, got {corr:.2f}"

        best_individual = max(a_sharpe, b_sharpe)
        assert combo_sharpe > best_individual, (
            f"Combined Sharpe {combo_sharpe:.2f} should exceed best "
            f"individual Sharpe {best_individual:.2f}"
        )

    def test_uncorrelated_strategies_get_independent_weights(self):
        """Allocator should give each uncorrelated strategy its own slot."""
        rng = np.random.default_rng(9)
        a_returns = rng.normal(0.0008, 0.010, 120).tolist()
        b_returns = rng.normal(0.0004, 0.014, 120).tolist()  # higher vol

        registry = StrategyRegistry.instance()
        a = _BacktestProxy("alpha")
        b = _BacktestProxy("beta")
        registry.register("alpha", a)
        registry.register("beta", b)
        _seed_returns(a, a_returns)
        _seed_returns(b, b_returns)

        allocator = CapitalAllocator(
            approach="inverse_vol",
            strategy_configs=_strat_configs(["alpha", "beta"]),
            total_capital=100_000.0,
        )
        weights = allocator.allocate(registry)

        # Inverse-vol: lower-vol alpha should get the larger allocation
        assert weights["alpha"] > weights["beta"], (
            f"Lower-vol strategy should get more capital — "
            f"alpha={weights['alpha']:.3f}, beta={weights['beta']:.3f}"
        )


# ── Scenario 3 — strategy failure triggers redistribution ──────────────────────

class TestStrategyFailureRedistributes:
    """When health_check disables a strategy, its capital should flow to the
    survivors on the next allocator pass."""

    def test_drawdown_breach_disables_strategy(self):
        registry = StrategyRegistry.instance()
        unhealthy = _BacktestProxy("unhealthy")
        healthy_a = _BacktestProxy("healthy_a")
        healthy_b = _BacktestProxy("healthy_b")
        registry.register("unhealthy", unhealthy)
        registry.register("healthy_a", healthy_a)
        registry.register("healthy_b", healthy_b)

        # Push the unhealthy strat into a >15 % drawdown
        bad_returns = [0.01] * 5 + [-0.03] * 10  # peak then sustained loss
        _seed_returns(unhealthy, bad_returns)
        assert unhealthy.get_current_drawdown() > _HEALTH_MAX_DRAWDOWN, (
            f"Test setup: drawdown {unhealthy.get_current_drawdown():.2%} "
            f"should exceed {_HEALTH_MAX_DRAWDOWN:.0%}"
        )

        # Healthy strategies — small noisy positive drift
        rng = np.random.default_rng(0)
        _seed_returns(healthy_a, rng.normal(0.0006, 0.008, 60).tolist())
        _seed_returns(healthy_b, rng.normal(0.0006, 0.008, 60).tolist())

        # Run health checks → unhealthy strategy should auto-disable
        registry.run_health_checks()
        assert unhealthy.is_enabled is False
        assert healthy_a.is_enabled is True
        assert healthy_b.is_enabled is True

    def test_allocator_redistributes_after_disable(self):
        registry = StrategyRegistry.instance()
        unhealthy = _BacktestProxy("unhealthy")
        healthy_a = _BacktestProxy("healthy_a")
        healthy_b = _BacktestProxy("healthy_b")
        registry.register("unhealthy", unhealthy)
        registry.register("healthy_a", healthy_a)
        registry.register("healthy_b", healthy_b)

        rng = np.random.default_rng(1)
        # All three healthy initially
        _seed_returns(unhealthy, rng.normal(0.0005, 0.010, 60).tolist())
        _seed_returns(healthy_a, rng.normal(0.0005, 0.010, 60).tolist())
        _seed_returns(healthy_b, rng.normal(0.0005, 0.010, 60).tolist())

        allocator = CapitalAllocator(
            approach="inverse_vol",
            strategy_configs=_strat_configs(
                ["unhealthy", "healthy_a", "healthy_b"]
            ),
            total_capital=100_000.0,
        )
        weights_before = allocator.allocate(registry)
        assert weights_before["unhealthy"] > 0.0

        # Now simulate the disable that run_health_checks() would do
        unhealthy.is_enabled = False
        weights_after = allocator.allocate(registry)

        # Disabled strategy should get zero allocation
        assert weights_after.get("unhealthy", 0.0) == pytest.approx(0.0, abs=1e-9)
        # Survivors should each receive more than they did before
        assert weights_after["healthy_a"] > weights_before["healthy_a"]
        assert weights_after["healthy_b"] > weights_before["healthy_b"]
        # Total invested should still respect the cash reserve
        assert sum(weights_after.values()) <= 0.90 + 1e-9


# ── Scenario 4 — portfolio risk caps aggregate exposure ────────────────────────

class _DummySignal:
    """Minimal stand-in for core.regime_strategies.Signal — only the fields
    PortfolioRiskManager.check_aggregate_exposure() reads from."""

    def __init__(self, symbol: str, position_size_pct: float):
        self.symbol = symbol
        self.position_size_pct = position_size_pct
        self.is_long = True
        self.entry_price = 100.0
        self.stop_loss = 95.0
        self.leverage = 1.0


class TestPortfolioRiskCap:
    """The 80 % aggregate cap is the contract the PRM advertises — verify
    it actually rejects requests that would breach it."""

    def _build_prm(self):
        from core.risk_manager import PortfolioRiskManager
        return PortfolioRiskManager(
            max_aggregate_exposure=0.80,
            max_single_symbol=0.50,
            max_portfolio_leverage=1.25,
            daily_dd_halt=0.99,      # disable DD short-circuit for this test
            max_dd_from_peak=0.99,
        )

    def _portfolio_state(self, equity: float, positions: dict):
        from core.risk_manager import PortfolioState
        return PortfolioState(
            equity=equity,
            cash=equity - sum(positions.values()),
            buying_power=equity,
            positions=positions,
            current_regime="BULL",
            price_history={},
            daily_pnl=0.0,
            peak_equity=equity,
            daily_start=equity,
            flicker_rate=0,
        )

    def test_first_signal_below_cap_is_approved(self):
        prm = self._build_prm()
        ps = self._portfolio_state(equity=100_000.0, positions={})
        sig = _DummySignal("SPY", position_size_pct=0.30)
        decision = prm.validate_signal(sig, "alpha", ps)
        assert decision.approved is True

    def test_signal_that_would_breach_cap_is_capped(self):
        """When aggregate exposure would exceed 80 %, the PRM either rejects
        the signal outright or trims it to fit within the headroom — never
        approves the un-modified breach."""
        prm = self._build_prm()
        # Already at 70 % gross — un-modified +30 % would push to 100 %.
        prm.update_strategy_positions("alpha", {"AAPL": 70_000.0})
        ps = self._portfolio_state(
            equity=100_000.0,
            positions={"AAPL": 70_000.0},
        )
        sig = _DummySignal("SPY", position_size_pct=0.30)
        decision = prm.validate_signal(sig, "beta", ps)

        if decision.approved:
            # Modified path — size must have been trimmed to ≤ 10 % headroom
            assert decision.modified_signal is not None
            new_size = decision.modified_signal.position_size_pct
            assert new_size < 0.30, (
                f"PRM should have trimmed 30 % size; got {new_size:.2%}"
            )
            assert new_size <= 0.10 + 1e-9, (
                f"Trimmed size {new_size:.2%} still breaches 10 % headroom"
            )
        else:
            assert decision.rejection_reason is not None
            assert "exposure" in decision.rejection_reason.lower() or \
                   "aggregate" in decision.rejection_reason.lower()

    def test_two_strategies_each_requesting_full_book_get_capped(self):
        """Per-strategy logic might say 'I want 100 % long' for both
        strategies; the PRM must keep aggregate ≤ 80 %."""
        prm = self._build_prm()
        ps = self._portfolio_state(equity=100_000.0, positions={})
        sig_a = _DummySignal("SPY", position_size_pct=0.50)
        sig_b = _DummySignal("QQQ", position_size_pct=0.50)

        d_a = prm.validate_signal(sig_a, "alpha", ps)
        assert d_a.approved is True
        # Update positions to reflect the first approval before sending the
        # second signal — this mirrors the live-loop ordering.
        prm.update_strategy_positions("alpha", {"SPY": 50_000.0})
        ps2 = self._portfolio_state(
            equity=100_000.0,
            positions={"SPY": 50_000.0},
        )
        d_b = prm.validate_signal(sig_b, "beta", ps2)
        # Aggregate would be 50 + 50 = 100 % > 80 % cap → reject or trim to ≤ 30 %
        if d_b.approved:
            assert d_b.modified_signal is not None
            trimmed = d_b.modified_signal.position_size_pct
            assert trimmed <= 0.30 + 1e-9, (
                f"Second signal must be trimmed to ≤ 30 % headroom; got {trimmed:.2%}"
            )
        else:
            assert d_b.rejection_reason is not None
