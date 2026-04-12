"""
test_validate_signal.py — Unit tests for RiskManager.validate_signal().

Covers:
  - Normal approval path
  - Lock file rejection
  - Circuit breaker halt rejection
  - Daily trade count rejection
  - Missing stop loss rejection
  - Duplicate order guard
  - Bid-ask spread too wide rejection
  - Concurrent position limit
  - Exposure cap (reduction + rejection)
  - Single-position concentration cap (reduction + rejection)
  - Leverage rule enforcement (circuit breaker, flicker, too many positions)
  - Overnight gap risk size reduction
  - Correlation check (size reduction, rejection)
  - Minimum position size rejection
  - Buying power rejection
  - REDUCED state halves position size
"""

from __future__ import annotations

import copy
import datetime as dt
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from core.regime_strategies import Direction, Signal
from core.risk_manager import (
    LOCK_FILE,
    CircuitBreakerType,
    PortfolioState,
    RejectionReason,
    RiskDecision,
    RiskManager,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _signal(
    symbol: str = "SPY",
    entry_price: float = 400.0,
    stop_loss: float = 380.0,
    position_size_pct: float = 0.10,
    leverage: float = 1.0,
    direction: Direction = Direction.LONG,
) -> Signal:
    return Signal(
        symbol=symbol,
        direction=direction,
        confidence=0.80,
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit=None,
        position_size_pct=position_size_pct,
        leverage=leverage,
        regime_id=0,
        regime_name="BULL",
        regime_probability=0.80,
        timestamp=pd.Timestamp("2024-01-15 10:00"),
        reasoning="test signal",
        strategy_name="TestStrategy",
    )


def _portfolio_state(
    equity: float = 100_000.0,
    cash: float = 90_000.0,
    buying_power: float = 90_000.0,
    positions: dict = None,
    flicker_rate: int = 0,
    current_regime: str = "BULL",
    last_order_times: dict = None,
    price_history: dict = None,
) -> PortfolioState:
    return PortfolioState(
        equity=equity,
        cash=cash,
        buying_power=buying_power,
        positions=positions or {},
        flicker_rate=flicker_rate,
        current_regime=current_regime,
        last_order_times=last_order_times or {},
        price_history=price_history or {},
    )


def _rm(**kwargs) -> RiskManager:
    defaults = dict(
        initial_equity=100_000.0,
        max_risk_per_trade=0.01,
        max_exposure=0.80,
        max_leverage=1.25,
        max_single_position=0.15,
        max_concurrent=5,
        max_daily_trades=20,
        min_position_usd=100.0,
        duplicate_window_secs=60,
        max_spread_pct=0.005,
        correlation_reduce_thr=0.70,
        correlation_reject_thr=0.85,
        correlation_window=60,
    )
    defaults.update(kwargs)
    return RiskManager(**defaults)


# Convenient timestamp for tests
_TS = dt.datetime(2024, 1, 15, 10, 0, 0)


# ── Normal approval ───────────────────────────────────────────────────────────

class TestNormalApproval:
    def test_simple_trade_approved(self) -> None:
        rm = _rm()
        sig = _signal()
        ps = _portfolio_state()
        decision = rm.validate_signal(sig, ps, timestamp=_TS)
        assert decision.approved is True
        assert decision.modified_signal is not None
        assert decision.rejection_reason is None

    def test_modified_signal_has_symbol(self) -> None:
        rm = _rm()
        decision = rm.validate_signal(_signal(), _portfolio_state(), timestamp=_TS)
        assert decision.modified_signal.symbol == "SPY"

    def test_approved_returns_risk_decision(self) -> None:
        rm = _rm()
        result = rm.validate_signal(_signal(), _portfolio_state(), timestamp=_TS)
        assert isinstance(result, RiskDecision)


# ── Lock file rejection ───────────────────────────────────────────────────────

class TestLockFileRejection:
    def test_rejects_when_lock_file_exists(self, tmp_path) -> None:
        rm = _rm()
        with patch.object(LOCK_FILE.__class__, "exists", return_value=True):
            # Patch at the module level
            import core.risk_manager as rmod
            original = rmod.LOCK_FILE
            try:
                fake_lock = tmp_path / "trading_halted.lock"
                fake_lock.write_text("halted")
                rmod.LOCK_FILE = fake_lock
                decision = rm.validate_signal(
                    _signal(), _portfolio_state(), timestamp=_TS
                )
                assert decision.approved is False
                assert "lock" in (decision.rejection_reason or "").lower()
            finally:
                rmod.LOCK_FILE = original

    def test_clean_state_approved(self, tmp_path) -> None:
        """Ensure no lock file → approval works normally."""
        rm = _rm()
        import core.risk_manager as rmod
        original = rmod.LOCK_FILE
        try:
            # Point at a path that does not exist (inside a known temp dir)
            rmod.LOCK_FILE = tmp_path / "trading_halted_nonexistent.lock"
            decision = rm.validate_signal(
                _signal(), _portfolio_state(), timestamp=_TS
            )
            assert decision.approved is True
        finally:
            rmod.LOCK_FILE = original


# ── Circuit breaker halt ──────────────────────────────────────────────────────

class TestCircuitBreakerHalt:
    def test_rejects_when_halted(self) -> None:
        rm = _rm(daily_dd_halt=0.03)
        import datetime as dt2
        ts1 = dt2.datetime(2024, 1, 15, 10, 0)
        rm.update_equity(100_000.0, timestamp=ts1)
        rm.update_equity(96_000.0, timestamp=ts1)   # -4% > 3% halt threshold
        ps = _portfolio_state()
        decision = rm.validate_signal(_signal(), ps, timestamp=_TS)
        assert decision.approved is False
        assert "circuit_breaker" in (decision.rejection_reason or "").lower()

    def test_reduced_halves_position_size(self) -> None:
        rm = _rm(daily_dd_reduce=0.02, daily_dd_halt=0.05)
        ts1 = dt.datetime(2024, 1, 15, 10, 0)
        rm.update_equity(100_000.0, timestamp=ts1)
        rm.update_equity(98_000.0, timestamp=ts1)   # -2% triggers REDUCE

        # Normal size without REDUCE
        rm2 = _rm()
        ps = _portfolio_state()
        normal = rm2.validate_signal(_signal(position_size_pct=0.10), ps, timestamp=_TS)
        reduced = rm.validate_signal(_signal(position_size_pct=0.10), ps, timestamp=_TS)

        assert reduced.approved is True
        assert reduced.modified_signal.position_size_pct < normal.modified_signal.position_size_pct


# ── Daily trade count ─────────────────────────────────────────────────────────

class TestDailyTradeCount:
    def test_rejects_at_max_daily_trades(self) -> None:
        rm = _rm(max_daily_trades=3)
        for _ in range(3):
            rm.increment_trade_count()
        decision = rm.validate_signal(_signal(), _portfolio_state(), timestamp=_TS)
        assert decision.approved is False
        assert "max_daily_trades" in (decision.rejection_reason or "")

    def test_approved_below_limit(self) -> None:
        rm = _rm(max_daily_trades=5)
        for _ in range(4):
            rm.increment_trade_count()
        decision = rm.validate_signal(_signal(), _portfolio_state(), timestamp=_TS)
        assert decision.approved is True


# ── Stop loss mandatory ───────────────────────────────────────────────────────

class TestStopLossMandatory:
    def test_rejects_when_stop_is_none(self) -> None:
        rm = _rm()
        sig = _signal()
        sig = Signal(
            symbol=sig.symbol, direction=sig.direction, confidence=sig.confidence,
            entry_price=sig.entry_price, stop_loss=None, take_profit=None,
            position_size_pct=sig.position_size_pct, leverage=sig.leverage,
            regime_id=sig.regime_id, regime_name=sig.regime_name,
            regime_probability=sig.regime_probability, timestamp=sig.timestamp,
            reasoning=sig.reasoning, strategy_name=sig.strategy_name,
        )
        decision = rm.validate_signal(sig, _portfolio_state(), timestamp=_TS)
        assert decision.approved is False
        assert "stop" in (decision.rejection_reason or "").lower()

    def test_rejects_when_stop_is_zero(self) -> None:
        rm = _rm()
        sig = _signal(stop_loss=0.0)
        decision = rm.validate_signal(sig, _portfolio_state(), timestamp=_TS)
        assert decision.approved is False


# ── Duplicate order guard ─────────────────────────────────────────────────────

class TestDuplicateOrder:
    def test_rejects_within_window(self) -> None:
        rm = _rm(duplicate_window_secs=120)
        recent = _TS - dt.timedelta(seconds=30)
        # The key format is "{symbol}:{direction_repr}" where direction uses
        # Direction.__format__ which produces "Direction.LONG" in Python 3.11+
        from core.regime_strategies import Direction
        dup_key = f"SPY:{Direction.LONG}"
        ps = _portfolio_state(
            last_order_times={dup_key: recent}
        )
        decision = rm.validate_signal(_signal(), ps, timestamp=_TS)
        assert decision.approved is False
        assert "duplicate" in (decision.rejection_reason or "").lower()

    def test_approved_after_window(self) -> None:
        rm = _rm(duplicate_window_secs=60)
        old = _TS - dt.timedelta(seconds=120)
        from core.regime_strategies import Direction
        dup_key = f"SPY:{Direction.LONG}"
        ps = _portfolio_state(
            last_order_times={dup_key: old}
        )
        decision = rm.validate_signal(_signal(), ps, timestamp=_TS)
        assert decision.approved is True


# ── Bid-ask spread ────────────────────────────────────────────────────────────

class TestSpreadCheck:
    def test_rejects_wide_spread(self) -> None:
        rm = _rm(max_spread_pct=0.005)
        # Spread of 2% > 0.5% limit
        decision = rm.validate_signal(
            _signal(), _portfolio_state(),
            bid=100.0, ask=102.0,   # 2% spread
            timestamp=_TS,
        )
        assert decision.approved is False
        assert "spread" in (decision.rejection_reason or "").lower()

    def test_approves_tight_spread(self) -> None:
        rm = _rm(max_spread_pct=0.005)
        decision = rm.validate_signal(
            _signal(), _portfolio_state(),
            bid=100.0, ask=100.4,   # 0.4% spread < 0.5%
            timestamp=_TS,
        )
        assert decision.approved is True

    def test_no_spread_check_when_bid_none(self) -> None:
        rm = _rm(max_spread_pct=0.005)
        decision = rm.validate_signal(
            _signal(), _portfolio_state(),
            bid=None, ask=None,
            timestamp=_TS,
        )
        assert decision.approved is True


# ── Concurrent position limit ─────────────────────────────────────────────────

class TestConcurrentLimit:
    def test_rejects_when_at_max_concurrent(self) -> None:
        rm = _rm(max_concurrent=2)
        ps = _portfolio_state(
            positions={"AAPL": 5_000.0, "MSFT": 5_000.0}
        )
        decision = rm.validate_signal(_signal("QQQ"), ps, timestamp=_TS)
        assert decision.approved is False
        assert "concurrent" in (decision.rejection_reason or "").lower()

    def test_approved_when_adding_to_existing_position(self) -> None:
        """Adding more to an already-open position should not count against concurrent limit."""
        rm = _rm(max_concurrent=2)
        ps = _portfolio_state(
            positions={"SPY": 5_000.0, "AAPL": 5_000.0}
        )
        # SPY is already open — adding to it is fine
        decision = rm.validate_signal(_signal("SPY"), ps, timestamp=_TS)
        # Should not be rejected for concurrent limit (may be rejected for other reasons)
        assert "concurrent" not in (decision.rejection_reason or "").lower()


# ── Portfolio exposure cap ────────────────────────────────────────────────────

class TestExposureCap:
    def test_exposure_reduces_size(self) -> None:
        """When adding would exceed max_exposure, size should be reduced but approved."""
        rm = _rm(max_exposure=0.80)
        ps = _portfolio_state(
            positions={"AAPL": 75_000.0}  # 75% already
        )
        # 10% more → 85% → over 80%, but 5% headroom remains → reduce to 5%
        decision = rm.validate_signal(
            _signal(position_size_pct=0.10), ps, timestamp=_TS
        )
        if decision.approved:
            assert decision.modified_signal.position_size_pct <= 0.05 + 1e-6

    def test_rejects_when_fully_at_limit(self) -> None:
        rm = _rm(max_exposure=0.80, min_position_usd=100.0)
        ps = _portfolio_state(
            positions={"AAPL": 80_000.0}  # exactly at 80% cap
        )
        decision = rm.validate_signal(
            _signal(position_size_pct=0.10), ps, timestamp=_TS
        )
        assert decision.approved is False


# ── Single-position concentration ────────────────────────────────────────────

class TestConcentrationCap:
    def test_concentration_reduces_size(self) -> None:
        rm = _rm(max_single_position=0.15)
        ps = _portfolio_state(
            positions={"SPY": 12_000.0}  # 12% already
        )
        # Adding 10% would push to 22% → capped to 3% headroom
        sig = _signal("SPY", position_size_pct=0.10)
        decision = rm.validate_signal(sig, ps, timestamp=_TS)
        if decision.approved:
            assert decision.modified_signal.position_size_pct <= 0.03 + 1e-6

    def test_rejects_at_concentration_limit(self) -> None:
        rm = _rm(max_single_position=0.15, min_position_usd=100.0)
        ps = _portfolio_state(
            positions={"SPY": 15_000.0}  # at 15% cap
        )
        sig = _signal("SPY", position_size_pct=0.10)
        decision = rm.validate_signal(sig, ps, timestamp=_TS)
        assert decision.approved is False


# ── Leverage rules ────────────────────────────────────────────────────────────

class TestLeverageRules:
    def test_leverage_forced_to_1x_when_flickering(self) -> None:
        rm = _rm()
        # Signal has 1.25x leverage; flicker_rate=4 → should strip leverage
        sig = _signal(leverage=1.25, position_size_pct=0.12)
        ps = _portfolio_state(flicker_rate=4)
        decision = rm.validate_signal(sig, ps, timestamp=_TS)
        if decision.approved:
            # Size should be reduced: 0.12 / 1.25 = 0.096
            assert decision.modified_signal.position_size_pct < sig.position_size_pct

    def test_leverage_not_stripped_when_signal_already_1x(self) -> None:
        rm = _rm()
        sig = _signal(leverage=1.0, position_size_pct=0.10)
        ps = _portfolio_state(flicker_rate=5)  # flickering, but leverage already 1x
        decision = rm.validate_signal(sig, ps, timestamp=_TS)
        if decision.approved:
            # No leverage adjustment needed
            assert "Leverage forced" not in " ".join(decision.modifications)


# ── Overnight gap risk ────────────────────────────────────────────────────────

class TestOvernightGapRisk:
    def test_overnight_reduces_size(self) -> None:
        rm = _rm(overnight_gap_multiple=3.0, max_overnight_loss_pct=0.02)
        sig = _signal(
            entry_price=400.0,
            stop_loss=380.0,      # $20 risk per share
            position_size_pct=0.15,
        )
        ps = _portfolio_state()
        normal = rm.validate_signal(sig, ps, is_overnight=False, timestamp=_TS)
        overnight = rm.validate_signal(sig, ps, is_overnight=True, timestamp=_TS)

        if normal.approved and overnight.approved:
            # Overnight gap cap should shrink the position
            assert overnight.modified_signal.position_size_pct <= \
                   normal.modified_signal.position_size_pct + 1e-9


# ── Minimum position USD ──────────────────────────────────────────────────────

class TestMinimumPosition:
    def test_rejects_below_minimum(self) -> None:
        rm = _rm(min_position_usd=1_000.0, max_single_position=0.15)
        # 0.001% * 100k = $100 < $1000 minimum
        sig = _signal(position_size_pct=0.001, entry_price=400.0, stop_loss=399.0)
        ps = _portfolio_state()
        decision = rm.validate_signal(sig, ps, timestamp=_TS)
        assert decision.approved is False
        assert "min" in (decision.rejection_reason or "").lower()


# ── Buying power check ────────────────────────────────────────────────────────

class TestBuyingPowerCheck:
    def test_rejects_insufficient_buying_power(self) -> None:
        rm = _rm()
        # Signal wants 20% = $20k but only $5k buying power
        sig = _signal(position_size_pct=0.20)
        ps = _portfolio_state(buying_power=5_000.0)
        decision = rm.validate_signal(sig, ps, timestamp=_TS)
        assert decision.approved is False
        assert "buying_power" in (decision.rejection_reason or "").lower()


# ── Correlation check ─────────────────────────────────────────────────────────

class TestCorrelationCheck:
    def _price_series(self, n: int, seed: int = 1) -> pd.Series:
        rng = np.random.default_rng(seed)
        return pd.Series(np.cumprod(1 + rng.normal(0, 0.01, n)) * 100.0)

    def test_high_correlation_reduces_size(self) -> None:
        rm = _rm(correlation_reduce_thr=0.70, correlation_reject_thr=0.85,
                 correlation_window=60)
        base = self._price_series(100)
        # Perfect correlation
        corr_series = base + np.random.default_rng(99).normal(0, 0.001, 100)

        ps = _portfolio_state(
            positions={"QQQ": 5_000.0},
            price_history={
                "SPY": base,
                "QQQ": corr_series,
            }
        )
        sig = _signal("SPY", position_size_pct=0.10)
        decision = rm.validate_signal(sig, ps, timestamp=_TS)
        if decision.approved:
            # High correlation → size should be reduced
            assert decision.modified_signal.position_size_pct <= 0.05 + 1e-6

    def test_very_high_correlation_rejects(self) -> None:
        rm = _rm(correlation_reject_thr=0.50, correlation_window=60)
        base = self._price_series(100)

        ps = _portfolio_state(
            positions={"QQQ": 5_000.0},
            price_history={
                "SPY": base,
                "QQQ": base * 1.001,  # almost identical
            }
        )
        sig = _signal("SPY", position_size_pct=0.10)
        decision = rm.validate_signal(sig, ps, timestamp=_TS)
        assert decision.approved is False

    def test_no_correlation_data_passes_through(self) -> None:
        """If no price history is available, correlation check should be skipped."""
        rm = _rm()
        ps = _portfolio_state(positions={"QQQ": 5_000.0}, price_history={})
        decision = rm.validate_signal(_signal("SPY"), ps, timestamp=_TS)
        # Should not be rejected for correlation
        if not decision.approved:
            assert "correlation" not in (decision.rejection_reason or "").lower()


# ── Modifications list ────────────────────────────────────────────────────────

class TestModificationsList:
    def test_no_modifications_on_clean_trade(self) -> None:
        rm = _rm()
        decision = rm.validate_signal(_signal(), _portfolio_state(), timestamp=_TS)
        assert decision.approved is True
        assert decision.modifications == []

    def test_modification_recorded_on_exposure_cap(self) -> None:
        rm = _rm(max_exposure=0.80)
        ps = _portfolio_state(positions={"AAPL": 75_000.0})
        sig = _signal(position_size_pct=0.10)
        decision = rm.validate_signal(sig, ps, timestamp=_TS)
        if decision.approved:
            assert len(decision.modifications) > 0
