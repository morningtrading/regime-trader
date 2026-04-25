"""
tests/test_strategy_registry.py — Unit tests for Phase 1: Strategy Registry.

Covers:
  - Registration and lookup
  - Lifecycle hooks (on_enable / on_disable)
  - Duplicate registration raises DuplicateStrategyError
  - Unhealthy auto-disable via run_health_checks()
  - Each individual health failure mode (drawdown, Sharpe, consecutive losses)
"""

from __future__ import annotations

import pytest
import pandas as pd

from core.strategy_registry import (
    DuplicateStrategyError,
    StrategyRegistry,
    register_strategy,
)
from core.regime_strategies import (
    BaseStrategy,
    Direction,
    Signal,
    StrategyHealth,
    _HEALTH_MAX_DRAWDOWN,
    _HEALTH_MAX_CONSEC_LOSSES,
    _HEALTH_MIN_SHARPE,
)
from core.hmm_engine import RegimeState


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_regime_state() -> RegimeState:
    return RegimeState(
        label="BULL",
        state_id=0,
        probability=0.90,
        is_confirmed=True,
        timestamp=pd.Timestamp("2024-01-01"),
    )


def _make_bars(n: int = 80) -> pd.DataFrame:
    import numpy as np
    rng = np.random.default_rng(42)
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    close = 100 + rng.normal(0, 1, n).cumsum()
    df = pd.DataFrame({
        "open":   close * 0.999,
        "high":   close * 1.002,
        "low":    close * 0.997,
        "close":  close,
        "volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
    }, index=idx)
    return df


class _StubStrategy(BaseStrategy):
    """Minimal concrete strategy for testing."""

    def __init__(self, strategy_name: str = "stub") -> None:
        super().__init__()
        self._name = strategy_name
        self.enabled_calls = 0
        self.disabled_calls = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def total_allocation(self) -> float:
        return 0.50

    def generate_signal(self, symbol, bars, regime_state):
        return None

    def on_enable(self) -> None:
        self.enabled_calls += 1
        super().on_enable()

    def on_disable(self) -> None:
        self.disabled_calls += 1
        super().on_disable()


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_registry():
    """Ensure a clean singleton before every test."""
    StrategyRegistry._reset()
    yield
    StrategyRegistry._reset()


# ── Registration & lookup ──────────────────────────────────────────────────────

class TestRegistration:
    def test_register_and_get(self):
        reg = StrategyRegistry.instance()
        s = _StubStrategy("alpha")
        reg.register("alpha", s)
        assert reg.get("alpha") is s

    def test_get_missing_returns_none(self):
        assert StrategyRegistry.instance().get("nonexistent") is None

    def test_all_returns_copy(self):
        reg = StrategyRegistry.instance()
        s = _StubStrategy("beta")
        reg.register("beta", s)
        result = reg.all()
        assert "beta" in result
        # Mutating the returned dict must not affect the registry.
        result.clear()
        assert reg.get("beta") is s

    def test_active_filters_disabled(self):
        reg = StrategyRegistry.instance()
        s1 = _StubStrategy("active")
        s2 = _StubStrategy("disabled")
        s2.is_enabled = False
        reg.register("active", s1)
        reg.register("disabled", s2)
        active = reg.active()
        assert "active" in active
        assert "disabled" not in active

    def test_unregister(self):
        reg = StrategyRegistry.instance()
        reg.register("tmp", _StubStrategy("tmp"))
        reg.unregister("tmp")
        assert reg.get("tmp") is None

    def test_unregister_missing_is_noop(self):
        StrategyRegistry.instance().unregister("ghost")  # must not raise

    def test_type_check_rejects_non_strategy(self):
        with pytest.raises(TypeError):
            StrategyRegistry.instance().register("bad", object())

    def test_decorator_registers(self):
        # The decorator instantiates the class and registers it.
        @register_strategy("decorated")
        class _Decorated(_StubStrategy):
            def __init__(self):
                super().__init__("decorated")

        assert StrategyRegistry.instance().get("decorated") is not None

    def test_singleton_is_same_instance(self):
        a = StrategyRegistry.instance()
        b = StrategyRegistry.instance()
        assert a is b


# ── Duplicate registration ─────────────────────────────────────────────────────

class TestDuplicateRegistration:
    def test_duplicate_raises(self):
        reg = StrategyRegistry.instance()
        reg.register("dup", _StubStrategy("dup"))
        with pytest.raises(DuplicateStrategyError):
            reg.register("dup", _StubStrategy("dup"))

    def test_unregister_then_reregister_ok(self):
        reg = StrategyRegistry.instance()
        reg.register("x", _StubStrategy("x"))
        reg.unregister("x")
        reg.register("x", _StubStrategy("x"))  # must not raise


# ── Lifecycle hooks ────────────────────────────────────────────────────────────

class TestLifecycleHooks:
    def test_on_disable_called_by_health_check(self):
        reg = StrategyRegistry.instance()
        s = _StubStrategy("sick")
        # Push enough consecutive losses to trigger unhealthy.
        ts = pd.Timestamp("2024-01-01")
        for i in range(_HEALTH_MAX_CONSEC_LOSSES):
            s.record_daily_return(ts + pd.Timedelta(days=i), -0.01)
        reg.register("sick", s)

        reg.run_health_checks()

        assert s.is_enabled is False
        assert s.disabled_calls == 1

    def test_on_enable_not_called_automatically(self):
        s = _StubStrategy("new")
        assert s.enabled_calls == 0

    def test_healthy_strategy_not_disabled(self):
        reg = StrategyRegistry.instance()
        s = _StubStrategy("healthy")
        reg.register("healthy", s)
        reg.run_health_checks()
        assert s.is_enabled is True
        assert s.disabled_calls == 0


# ── Health check — individual failure modes ────────────────────────────────────

class TestHealthCheck:
    def test_healthy_when_no_history(self):
        s = _StubStrategy()
        h = s.health_check()
        assert h.is_healthy is True
        assert h.reason_if_unhealthy is None

    def test_unhealthy_on_excess_drawdown(self):
        s = _StubStrategy()
        # Drive equity down more than 15 % from peak.
        ts = pd.Timestamp("2024-01-01")
        s.record_daily_return(ts, 0.0)   # set peak
        loss = _HEALTH_MAX_DRAWDOWN + 0.01
        s.record_daily_return(ts + pd.Timedelta(days=1), -loss)

        h = s.health_check()
        assert h.is_healthy is False
        assert "drawdown" in h.reason_if_unhealthy.lower()

    def test_unhealthy_on_low_sharpe(self):
        s = _StubStrategy()
        ts = pd.Timestamp("2024-01-01")
        # Alternate +tiny / -large so the streak never hits 10 consecutive losses
        # but the mean return is deeply negative → Sharpe << -1.0.
        returns = []
        for i in range(40):
            returns.append(0.001 if i % 2 == 0 else -0.03)
        for i, r in enumerate(returns):
            s.record_daily_return(ts + pd.Timedelta(days=i), r)

        h = s.health_check()
        assert h.is_healthy is False
        assert "sharpe" in h.reason_if_unhealthy.lower()

    def test_unhealthy_on_consecutive_losses(self):
        s = _StubStrategy()
        ts = pd.Timestamp("2024-01-01")
        for i in range(_HEALTH_MAX_CONSEC_LOSSES):
            s.record_daily_return(ts + pd.Timedelta(days=i), -0.001)

        h = s.health_check()
        assert h.is_healthy is False
        assert "consecutive" in h.reason_if_unhealthy.lower()

    def test_consecutive_loss_resets_on_profit(self):
        s = _StubStrategy()
        ts = pd.Timestamp("2024-01-01")
        for i in range(_HEALTH_MAX_CONSEC_LOSSES - 1):
            s.record_daily_return(ts + pd.Timedelta(days=i), -0.001)
        # One winning day resets the streak.
        s.record_daily_return(ts + pd.Timedelta(days=_HEALTH_MAX_CONSEC_LOSSES), 0.01)

        h = s.health_check()
        assert h.consecutive_losing_days == 0

    def test_health_fields_populated(self):
        s = _StubStrategy()
        h = s.health_check()
        assert isinstance(h.recent_sharpe, float)
        assert isinstance(h.current_drawdown, float)
        assert isinstance(h.consecutive_losing_days, int)

    def test_multiple_reasons_combined(self):
        s = _StubStrategy()
        ts = pd.Timestamp("2024-01-01")
        # Trigger both excess drawdown AND consecutive losses.
        for i in range(_HEALTH_MAX_CONSEC_LOSSES):
            s.record_daily_return(ts + pd.Timedelta(days=i), -0.02)

        h = s.health_check()
        assert h.is_healthy is False
        # At least two distinct reasons should appear.
        assert ";" in h.reason_if_unhealthy


# ── Performance tracking ───────────────────────────────────────────────────────

class TestPerformanceTracking:
    def test_sharpe_zero_with_no_history(self):
        assert _StubStrategy().get_recent_sharpe() == 0.0

    def test_drawdown_zero_with_no_history(self):
        assert _StubStrategy().get_current_drawdown() == 0.0

    def test_drawdown_positive_after_loss(self):
        s = _StubStrategy()
        ts = pd.Timestamp("2024-01-01")
        s.record_daily_return(ts, -0.05)
        assert s.get_current_drawdown() == pytest.approx(0.05, abs=1e-9)

    def test_peak_updates_on_gain(self):
        s = _StubStrategy()
        ts = pd.Timestamp("2024-01-01")
        s.record_daily_return(ts, 0.10)   # peak = 1.10
        s.record_daily_return(ts + pd.Timedelta(days=1), -0.05)
        # drawdown from 1.10 to 1.10*0.95 = 1.045 → ~4.5 %
        assert s.get_current_drawdown() == pytest.approx(0.05, abs=1e-9)

    def test_rolling_window_respects_maxlen(self):
        s = _StubStrategy()
        ts = pd.Timestamp("2024-01-01")
        for i in range(100):   # more than _PERF_HISTORY_DAYS (60)
            s.record_daily_return(ts + pd.Timedelta(days=i), 0.001)
        assert len(s.performance_history) == 60
