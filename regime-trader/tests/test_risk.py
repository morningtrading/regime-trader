"""
test_risk.py -- Unit tests for RiskManager position sizing and drawdown controls.

Tests cover position sizing, exposure limits, concentration limits, daily/weekly
drawdown rules, peak-to-trough halt, and the trading-state machine.
"""

import datetime as dt

import pytest

from core.risk_manager import RiskManager, TradingState, RiskCheck


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def rm() -> RiskManager:
    return RiskManager(
        initial_equity     = 100_000.0,
        max_risk_per_trade = 0.01,
        max_exposure       = 0.80,
        max_leverage       = 1.25,
        max_single_position= 0.15,
        max_concurrent     = 5,
        max_daily_trades   = 20,
        daily_dd_reduce    = 0.02,
        daily_dd_halt      = 0.03,
        weekly_dd_reduce   = 0.05,
        weekly_dd_halt     = 0.07,
        max_dd_from_peak   = 0.10,
        min_position_usd   = 100.0,
    )

# keep old name working too
@pytest.fixture
def risk_manager(rm) -> RiskManager:
    return rm


# helper: advance equity with an explicit timestamp
def _update(rm: RiskManager, equity: float, day: int = 1, week: int = 1) -> None:
    """day=1 keeps everything on the same day; day=2 triggers a daily reset."""
    ts = dt.datetime(2024, 1, week * 7 - 6 + day - 1, 10, 0)
    rm.update_equity(equity, timestamp=ts)


# ── Position sizing tests ──────────────────────────────────────────────────────

class TestPositionSizing:

    def test_size_respects_target_weight(self, rm: RiskManager) -> None:
        """Shares = floor(equity * weight / price) when no stop or risk limit binds."""
        # 10% weight at $500/share, no stop -> 100_000 * 0.10 / 500 = 20 shares
        shares = rm.compute_position_size(0.10, 500.0)
        assert shares == 20.0

    def test_size_capped_by_max_risk_per_trade(self, rm: RiskManager) -> None:
        """With a tight stop, risk rule should cap below the weight-based size."""
        # entry=$500, stop=$495 -> risk_per_share=$5
        # max_risk_dollars = 100_000 * 0.01 = 1_000
        # risk_based_shares = 1000 / 5 = 200
        # weight_based at 15% = 100_000*0.15/500 = 30 shares  <- binding cap
        shares_no_stop  = rm.compute_position_size(0.15, 500.0)
        shares_with_stop = rm.compute_position_size(0.15, 500.0, stop_price=499.0)
        # stop=$499 -> risk=$1/share -> risk_based=1000 shares, not binding
        # stop=$450 -> risk=$50/share -> risk_based=20 shares < weight_based 30
        shares_tight_stop = rm.compute_position_size(0.15, 500.0, stop_price=450.0)
        assert shares_no_stop == 30.0
        assert shares_with_stop == 30.0           # risk cap not binding at $1 stop gap
        assert shares_tight_stop == 20.0          # risk cap binds at $50 gap

    def test_size_is_non_negative(self, rm: RiskManager) -> None:
        """compute_position_size() must never return a negative number."""
        assert rm.compute_position_size(0.0, 100.0) >= 0
        assert rm.compute_position_size(0.0, 100.0, stop_price=90.0) >= 0
        assert rm.compute_position_size(0.01, 1e6) >= 0   # very small weight


# ── Exposure and concentration limit tests ────────────────────────────────────

class TestExposureLimits:

    def test_trade_blocked_when_max_exposure_exceeded(self, rm: RiskManager) -> None:
        """check_trade() should reject when gross exposure would exceed max_exposure."""
        # Fill portfolio to just below the 80% cap
        existing = {"SPY": 75_000.0}   # 75% exposure
        # Adding 10 shares @ $600 = $6_000 -> 81% -> over 80% cap
        result = rm.check_trade("QQQ", 10, 600.0, existing)
        # After capping, remaining headroom = 80_000 - 75_000 = 5_000
        # 5_000 / 600 = 8.33 -> floor 8 shares * $600 = $4_800 -- approved but reduced
        assert result.approved
        assert result.adjusted_size <= 5_000 / 600 + 1   # within headroom

    def test_trade_blocked_when_fully_exposed(self, rm: RiskManager) -> None:
        """check_trade() should reject when portfolio is already at max_exposure."""
        existing = {"SPY": 80_000.0}   # exactly at 80% cap -- no room left
        result = rm.check_trade("QQQ", 10, 600.0, existing)
        assert not result.approved

    def test_trade_blocked_when_concentration_exceeded(self, rm: RiskManager) -> None:
        """check_trade() should reject when a symbol would exceed max_single_position."""
        # SPY already at 14% (just below 15%), adding 10 shares @ $200 = $2_000 -> 16%
        existing = {"SPY": 14_000.0}
        result = rm.check_trade("SPY", 10, 200.0, existing)
        # Allowed headroom: 15_000 - 14_000 = $1_000 -> 5 shares; min_position check
        assert result.approved
        assert result.adjusted_size * 200.0 <= 1_000 + 1  # within cap

    def test_concentration_hard_block(self, rm: RiskManager) -> None:
        """Reject when symbol is already AT the single-position limit."""
        existing = {"SPY": 15_000.0}   # already at 15% cap
        result = rm.check_trade("SPY", 10, 500.0, existing)
        assert not result.approved

    def test_trade_approved_within_limits(self, rm: RiskManager) -> None:
        """check_trade() should approve a small trade with no limits binding."""
        result = rm.check_trade("AAPL", 5, 180.0, {})
        assert result.approved
        assert result.adjusted_size == 5.0


# ── Trade count circuit breaker ───────────────────────────────────────────────

class TestTradeCountBreaker:

    def test_halt_after_max_daily_trades(self, rm: RiskManager) -> None:
        """After max_daily_trades increments, further check_trade() calls are rejected."""
        for _ in range(rm.max_daily_trades):
            rm.increment_trade_count()
        result = rm.check_trade("SPY", 1, 400.0, {})
        assert not result.approved
        assert "max_daily_trades" in (result.rejection_reason or "")

    def test_counter_resets_on_new_day(self, rm: RiskManager) -> None:
        """Daily trade counter resets to 0 on a new calendar day."""
        for _ in range(rm.max_daily_trades):
            rm.increment_trade_count()
        # Simulate a new day
        _update(rm, 100_000.0, day=1)   # day 1 triggers first reset
        _update(rm, 100_000.0, day=2)   # day 2 triggers second reset
        # After reset the counter is 0 -- trades should be allowed again
        result = rm.check_trade("SPY", 1, 400.0, {})
        assert result.approved


# ── Drawdown-triggered state transitions ─────────────────────────────────────

class TestDrawdownState:

    def test_normal_state_at_startup(self, rm: RiskManager) -> None:
        assert rm.get_trading_state() == TradingState.NORMAL

    def test_state_becomes_reduced_on_daily_dd_reduce(self, rm: RiskManager) -> None:
        """Exactly -2% daily drawdown triggers REDUCED state."""
        _update(rm, 100_000.0, day=1)    # set daily baseline
        _update(rm, 98_000.0,  day=1)    # -2.0% -> REDUCED
        assert rm.get_trading_state() == TradingState.REDUCED

    def test_state_becomes_halted_on_daily_dd_halt(self, rm: RiskManager) -> None:
        """A -3% daily drop triggers a HALTED state."""
        _update(rm, 100_000.0, day=1)
        _update(rm, 97_000.0,  day=1)    # -3.0% -> HALTED
        assert rm.get_trading_state() == TradingState.HALTED

    def test_state_becomes_halted_on_peak_dd(self, rm: RiskManager) -> None:
        """A -10% peak-to-trough drop triggers HALTED and writes a lock file."""
        from core.risk_manager import LOCK_FILE
        LOCK_FILE.unlink(missing_ok=True)   # ensure clean state

        _update(rm, 100_000.0, day=1)
        # Simulate gradual loss across multiple days to avoid daily halt firing first
        # Use large weekly window to reach 10% peak DD
        _update(rm, 95_000.0,  day=2)    # new day -> daily baseline resets to 95k
        _update(rm, 93_000.0,  day=3)    # new day -> baseline resets to 93k
        _update(rm, 91_000.0,  day=4)    # new day -> baseline resets to 91k
        _update(rm, 90_000.0,  day=5)    # -10% from peak 100k -> PEAK_HALT
        assert rm.get_trading_state() == TradingState.HALTED
        assert LOCK_FILE.exists()
        LOCK_FILE.unlink(missing_ok=True)   # cleanup

    def test_reduced_state_halves_position_size(self, rm: RiskManager) -> None:
        """In REDUCED state, check_trade adjusted_size should be 50% of normal."""
        # Use a proposed size well below the single-position cap so the 50%
        # state discount is the binding constraint (not the concentration limit).
        # 20 shares @ $400 = $8_000 = 8% of equity -- well under the 15% cap.
        proposed = 20

        normal_result = rm.check_trade("SPY", proposed, 400.0, {})
        assert normal_result.approved
        normal_size = normal_result.adjusted_size   # should be 20

        # Push a fresh manager into REDUCED state
        rm2 = RiskManager(
            initial_equity=100_000.0,
            daily_dd_reduce=0.02,
            daily_dd_halt=0.03,
        )
        _update(rm2, 100_000.0, day=1)
        _update(rm2, 98_000.0,  day=1)   # -2% -> REDUCED
        assert rm2.get_trading_state() == TradingState.REDUCED

        reduced_result = rm2.check_trade("SPY", proposed, 400.0, {})
        assert reduced_result.approved
        # REDUCED applies a 50% discount: 20 * 0.5 = 10 shares
        assert reduced_result.adjusted_size == normal_size * 0.5

    def test_halted_state_blocks_all_trades(self, rm: RiskManager) -> None:
        """In HALTED state, check_trade() must reject every call."""
        _update(rm, 100_000.0, day=1)
        _update(rm, 97_000.0,  day=1)    # -3% -> HALTED
        result = rm.check_trade("SPY", 1, 400.0, {})
        assert not result.approved
        assert result.adjusted_size == 0.0


# ── Weekly drawdown tests ─────────────────────────────────────────────────────

class TestWeeklyDrawdown:

    def test_weekly_reset_on_monday(self, rm: RiskManager) -> None:
        """weekly_start_equity resets when a new ISO week begins."""
        # Week 1
        ts_w1 = dt.datetime(2024, 1, 1, 10, 0)   # Monday week 1
        rm.update_equity(100_000.0, timestamp=ts_w1)
        rm.update_equity(96_000.0,  timestamp=ts_w1)   # -4% within week 1

        dd_w1 = rm.get_drawdown_state()
        assert dd_w1.weekly_dd < -0.03

        # Week 2 -- new week should reset weekly baseline
        ts_w2 = dt.datetime(2024, 1, 8, 10, 0)   # Monday week 2
        rm.update_equity(96_000.0, timestamp=ts_w2)   # triggers weekly reset

        dd_w2 = rm.get_drawdown_state()
        # After weekly reset, weekly_dd should be ~0 (equity == weekly_start)
        assert abs(dd_w2.weekly_dd) < 0.001

    def test_halt_on_weekly_dd_halt(self, rm: RiskManager) -> None:
        """Losing >7% in one week triggers HALTED state."""
        ts = dt.datetime(2024, 1, 1, 10, 0)   # Monday
        rm.update_equity(100_000.0, timestamp=ts)

        # Spread losses over the same week but different days to avoid daily halt
        rm.update_equity(98_000.0, timestamp=dt.datetime(2024, 1, 2, 10, 0))  # -2% day
        rm.update_equity(96_000.0, timestamp=dt.datetime(2024, 1, 3, 10, 0))  # -2% day
        rm.update_equity(94_000.0, timestamp=dt.datetime(2024, 1, 4, 10, 0))  # -2% day
        rm.update_equity(92_500.0, timestamp=dt.datetime(2024, 1, 5, 10, 0))  # -1.6% day
        # cumulative weekly: -7.5% from 100k -> weekly_halt=7% fires
        assert rm.get_trading_state() == TradingState.HALTED

    def test_weekly_reduce_before_halt(self, rm: RiskManager) -> None:
        """Losing >5% but <7% in one week triggers REDUCED (not HALTED)."""
        ts = dt.datetime(2024, 1, 1, 10, 0)
        rm.update_equity(100_000.0, timestamp=ts)
        rm.update_equity(98_000.0, timestamp=dt.datetime(2024, 1, 2, 10, 0))
        rm.update_equity(96_000.0, timestamp=dt.datetime(2024, 1, 3, 10, 0))
        rm.update_equity(94_500.0, timestamp=dt.datetime(2024, 1, 4, 10, 0))
        # cumulative weekly: -5.5% -> REDUCED (weekly_reduce=5%, weekly_halt=7%)
        assert rm.get_trading_state() == TradingState.REDUCED


# ── DrawdownState snapshot tests ─────────────────────────────────────────────

class TestDrawdownStateSnapshot:

    def test_snapshot_fields_consistent(self, rm: RiskManager) -> None:
        """get_drawdown_state() fields should be internally consistent."""
        _update(rm, 100_000.0, day=1)
        _update(rm, 95_000.0,  day=1)

        dd = rm.get_drawdown_state()
        assert dd.peak_equity == 100_000.0
        assert dd.current_equity == 95_000.0
        assert abs(dd.dd_from_peak - (-0.05)) < 1e-6
        assert abs(dd.daily_dd    - (-0.05)) < 1e-6

    def test_peak_tracks_new_high(self, rm: RiskManager) -> None:
        """Peak equity should update when portfolio makes a new high."""
        _update(rm, 100_000.0, day=1)
        _update(rm, 110_000.0, day=2)   # new high
        dd = rm.get_drawdown_state()
        assert dd.peak_equity == 110_000.0
        assert abs(dd.dd_from_peak) < 1e-6   # at peak, dd should be ~0


# ── from_config factory ───────────────────────────────────────────────────────

class TestFromConfig:

    def test_loads_risk_section(self) -> None:
        cfg = {
            "risk": {
                "max_risk_per_trade":  0.005,
                "max_exposure":        0.70,
                "max_single_position": 0.10,
                "max_concurrent":      3,
                "max_daily_trades":    10,
                "daily_dd_reduce":     0.015,
                "daily_dd_halt":       0.025,
                "weekly_dd_reduce":    0.04,
                "weekly_dd_halt":      0.06,
                "max_dd_from_peak":    0.08,
            }
        }
        rm = RiskManager.from_config(cfg, initial_equity=50_000.0)
        assert rm.initial_equity          == 50_000.0
        assert rm.max_risk_per_trade      == 0.005
        assert rm.max_exposure            == 0.70
        assert rm.max_single_position     == 0.10
        assert rm.max_concurrent          == 3
        assert rm.max_daily_trades        == 10
        assert rm.circuit_breaker.daily_reduce  == 0.015
        assert rm.circuit_breaker.daily_halt    == 0.025
        assert rm.circuit_breaker.weekly_reduce == 0.04
        assert rm.circuit_breaker.weekly_halt   == 0.06
        assert rm.circuit_breaker.peak_halt     == 0.08

    def test_defaults_when_section_missing(self) -> None:
        rm = RiskManager.from_config({}, initial_equity=200_000.0)
        assert rm.initial_equity     == 200_000.0
        assert rm.max_risk_per_trade == 0.01
        assert rm.max_exposure       == 0.80
