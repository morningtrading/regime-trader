"""
risk_manager.py -- Position sizing, leverage enforcement, and drawdown controls.

The risk manager operates INDEPENDENTLY of the HMM.  Even if the HMM fails
completely, circuit breakers catch drawdowns based on actual P&L.
Defense in depth.  The risk manager has ABSOLUTE VETO POWER over any signal.

LAYERS (applied in order):
  1. Circuit breakers  -- daily/weekly/peak drawdown triggers
  2. Portfolio limits  -- exposure, leverage, position count
  3. Position sizing   -- 1% risk rule + gap risk overnight adjustment
  4. Correlation check -- reduce or reject highly correlated adds
  5. Order validation  -- buying power, duplicate detection, bid-ask spread

All thresholds are loaded from config/settings.yaml [risk] section.
"""

from __future__ import annotations

import copy
import datetime as dt
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Constants                                                                    #
# --------------------------------------------------------------------------- #

LOCK_FILE = Path("trading_halted.lock")

# --------------------------------------------------------------------------- #
# Enumerations                                                                 #
# --------------------------------------------------------------------------- #

class TradingState(Enum):
    """Current trading permission level."""
    NORMAL  = auto()   # full trading allowed
    REDUCED = auto()   # position sizes halved due to drawdown
    HALTED  = auto()   # no new trades; drawdown limit breached


class CircuitBreakerType(Enum):
    DAILY_REDUCE  = "daily_reduce"
    DAILY_HALT    = "daily_halt"
    WEEKLY_REDUCE = "weekly_reduce"
    WEEKLY_HALT   = "weekly_halt"
    PEAK_HALT     = "peak_halt"


class RejectionReason(str, Enum):
    CIRCUIT_BREAKER_HALTED   = "circuit_breaker_halted"
    LOCK_FILE_EXISTS         = "lock_file_exists"
    MAX_DAILY_TRADES         = "max_daily_trades"
    MAX_CONCURRENT           = "max_concurrent_positions"
    MAX_EXPOSURE             = "max_total_exposure"
    MAX_SINGLE_POSITION      = "max_single_position"
    MAX_LEVERAGE             = "max_leverage"
    NO_STOP_LOSS             = "missing_stop_loss"
    BELOW_MIN_POSITION       = "below_min_position_usd"
    INSUFFICIENT_BUYING_PWR  = "insufficient_buying_power"
    CORRELATION_TOO_HIGH     = "correlation_above_reject_threshold"
    DUPLICATE_ORDER          = "duplicate_order_within_window"
    SPREAD_TOO_WIDE          = "bid_ask_spread_too_wide"
    # ── Portfolio-level (PortfolioRiskManager) ─────────────────────────────────
    PORTFOLIO_EXPOSURE       = "portfolio_aggregate_exposure"
    PORTFOLIO_SYMBOL_CAP     = "portfolio_symbol_concentration"
    PORTFOLIO_LEVERAGE       = "portfolio_total_leverage"
    PORTFOLIO_DD_HALT        = "portfolio_drawdown_halt"
    PORTFOLIO_PEAK_DD        = "portfolio_peak_drawdown_halt"
    PORTFOLIO_CORR_CLUSTER   = "portfolio_correlated_cluster"


# --------------------------------------------------------------------------- #
# Data structures                                                              #
# --------------------------------------------------------------------------- #

@dataclass
class PortfolioState:
    """
    Snapshot of portfolio state passed into validate_signal().

    Attributes
    ----------
    equity                : current total portfolio value (cash + positions)
    cash                  : uninvested cash
    buying_power          : available purchasing power (may differ from cash with margin)
    positions             : symbol -> current dollar value of each open position
    daily_pnl             : unrealised + realised P&L since today's open
    weekly_pnl            : P&L since Monday's open
    peak_equity           : all-time high equity (for drawdown calculation)
    daily_start           : equity at today's open
    weekly_start          : equity at Monday's open
    circuit_breaker_status: currently active CircuitBreakerType (or None)
    flicker_rate          : HMM regime-change count over the last flicker_window bars
    current_regime        : HMM regime label at this instant (for logging only)
    price_history         : symbol -> recent close prices (for correlation check)
    last_order_times      : "symbol:direction" -> last order timestamp (duplicate guard)
    """
    equity:                  float
    cash:                    float
    buying_power:            float
    positions:               Dict[str, float]                    # symbol -> dollar value
    daily_pnl:               float = 0.0
    weekly_pnl:              float = 0.0
    peak_equity:             float = 0.0
    daily_start:             float = 0.0
    weekly_start:            float = 0.0
    circuit_breaker_status:  Optional[CircuitBreakerType] = None
    flicker_rate:            int = 0
    current_regime:          str = "UNKNOWN"
    price_history:           Dict[str, pd.Series] = field(default_factory=dict)
    last_order_times:        Dict[str, dt.datetime] = field(default_factory=dict)


@dataclass
class RiskDecision:
    """
    Result of validate_signal().

    If approved=False, modified_signal is None and rejection_reason explains why.
    If approved=True, modified_signal contains the (possibly adjusted) signal and
    modifications lists the human-readable changes made.
    """
    approved:          bool
    modified_signal:   Optional[object]      # Signal dataclass from regime_strategies
    rejection_reason:  Optional[str] = None
    modifications:     List[str] = field(default_factory=list)


@dataclass
class DrawdownState:
    """Snapshot of current drawdown metrics."""
    peak_equity:         float
    current_equity:      float
    daily_start_equity:  float
    weekly_start_equity: float
    dd_from_peak:        float   # fraction below all-time peak  (negative)
    daily_dd:            float   # fraction below today's open   (negative)
    weekly_dd:           float   # fraction below Monday's open  (negative)


@dataclass
class CircuitBreakerEvent:
    """One logged trigger of a circuit breaker."""
    timestamp:        dt.datetime
    breaker_type:     CircuitBreakerType
    actual_dd:        float
    equity:           float
    positions_closed: List[str]
    hmm_regime:       str


# --------------------------------------------------------------------------- #
# CircuitBreaker                                                               #
# --------------------------------------------------------------------------- #

class CircuitBreaker:
    """
    Stateful drawdown monitor that fires when thresholds are crossed.

    Designed to be called every bar by RiskManager.update_equity().
    All thresholds are fractions (e.g. 0.02 = 2%).
    """

    def __init__(
        self,
        daily_reduce:  float = 0.02,
        daily_halt:    float = 0.03,
        weekly_reduce: float = 0.05,
        weekly_halt:   float = 0.07,
        peak_halt:     float = 0.10,
    ) -> None:
        self.daily_reduce  = daily_reduce
        self.daily_halt    = daily_halt
        self.weekly_reduce = weekly_reduce
        self.weekly_halt   = weekly_halt
        self.peak_halt     = peak_halt

        self._active:  Optional[CircuitBreakerType] = None
        self._history: List[CircuitBreakerEvent]    = []

    # -- Public ----------------------------------------------------------------

    def check(
        self,
        daily_dd:  float,
        weekly_dd: float,
        peak_dd:   float,
    ) -> Optional[CircuitBreakerType]:
        """
        Return the most severe active circuit-breaker type, or None.
        Parameters are all negative fractions (e.g. daily_dd=-0.025).
        """
        if peak_dd   <= -self.peak_halt:     return CircuitBreakerType.PEAK_HALT
        if weekly_dd <= -self.weekly_halt:   return CircuitBreakerType.WEEKLY_HALT
        if daily_dd  <= -self.daily_halt:    return CircuitBreakerType.DAILY_HALT
        if weekly_dd <= -self.weekly_reduce: return CircuitBreakerType.WEEKLY_REDUCE
        if daily_dd  <= -self.daily_reduce:  return CircuitBreakerType.DAILY_REDUCE
        return None

    def update(
        self,
        daily_dd:  float,
        weekly_dd: float,
        peak_dd:   float,
        equity:    float,
        positions: Dict[str, float],
        regime:    str,
        timestamp: Optional[dt.datetime] = None,
    ) -> Optional[CircuitBreakerType]:
        """
        Evaluate current drawdowns, update active state, log new triggers.
        Returns the (possibly newly set) active breaker type.
        """
        ts    = timestamp or dt.datetime.utcnow()
        fired = self.check(daily_dd, weekly_dd, peak_dd)

        if fired is not None and fired != self._active:
            closed = (
                list(positions.keys())
                if fired in (CircuitBreakerType.DAILY_HALT,
                             CircuitBreakerType.WEEKLY_HALT,
                             CircuitBreakerType.PEAK_HALT)
                else []
            )
            event = CircuitBreakerEvent(
                timestamp        = ts,
                breaker_type     = fired,
                actual_dd        = min(daily_dd, weekly_dd, peak_dd),
                equity           = equity,
                positions_closed = closed,
                hmm_regime       = regime,
            )
            self._history.append(event)
            logger.warning(
                "CIRCUIT BREAKER [%s] fired | dd_daily=%.2f%% dd_weekly=%.2f%% "
                "dd_peak=%.2f%% | equity=$%.2f | regime=%s | closed=%s",
                fired.value,
                daily_dd * 100, weekly_dd * 100, peak_dd * 100,
                equity, regime, closed or "none",
            )

            if fired == CircuitBreakerType.PEAK_HALT:
                LOCK_FILE.write_text(
                    f"Trading halted {ts.isoformat()}  "
                    f"peak_dd={peak_dd:.2%}  equity={equity:.2f}  regime={regime}\n"
                )
                logger.error(
                    "PEAK DRAWDOWN HALT: lock file written -> %s  "
                    "Manual deletion required to resume trading.", LOCK_FILE
                )

        self._active = fired
        return fired

    def reset_daily(self) -> None:
        """Clear daily-level breakers at start of new trading day."""
        if self._active in (CircuitBreakerType.DAILY_REDUCE,
                            CircuitBreakerType.DAILY_HALT):
            logger.info("CircuitBreaker: daily reset (was %s)", self._active)
            self._active = None

    def reset_weekly(self) -> None:
        """Clear weekly-level breakers at start of new trading week."""
        if self._active in (CircuitBreakerType.WEEKLY_REDUCE,
                            CircuitBreakerType.WEEKLY_HALT):
            logger.info("CircuitBreaker: weekly reset (was %s)", self._active)
            self._active = None

    def get_active(self) -> Optional[CircuitBreakerType]:
        return self._active

    def get_history(self) -> List[CircuitBreakerEvent]:
        return list(self._history)

    def is_halted(self) -> bool:
        return self._active in (
            CircuitBreakerType.DAILY_HALT,
            CircuitBreakerType.WEEKLY_HALT,
            CircuitBreakerType.PEAK_HALT,
        )

    def is_reduced(self) -> bool:
        return self._active in (
            CircuitBreakerType.DAILY_REDUCE,
            CircuitBreakerType.WEEKLY_REDUCE,
        )


# --------------------------------------------------------------------------- #
# RiskCheck (simplified result for backtester)                                #
# --------------------------------------------------------------------------- #

@dataclass
class RiskCheck:
    """Simplified result used by the backtester's check_trade() path."""
    approved:          bool
    trading_state:     TradingState
    adjusted_size:     float
    rejection_reason:  Optional[str] = None


# --------------------------------------------------------------------------- #
# RiskManager                                                                  #
# --------------------------------------------------------------------------- #

class RiskManager:
    """
    Enforces position-sizing and drawdown rules.

    Two main entry points:
      validate_signal(signal, portfolio_state) -> RiskDecision
          Full gate used by live trading and the signal generator.

      check_trade(symbol, proposed_size, price, positions, stop_price) -> RiskCheck
          Simplified check for the walk-forward backtester.

    Call update_equity() at the end of every bar.

    Parameters
    ----------
    initial_equity           : Starting portfolio value in USD.
    max_risk_per_trade       : Maximum fraction of equity risked per trade (1%).
    max_exposure             : Gross exposure cap -- sum of |weights| (80%).
    max_leverage             : Hard leverage ceiling (1.25x).
    max_single_position      : Max weight of any single symbol (15%).
    max_correlated_exposure  : Max exposure in correlated cluster (30%).
    max_concurrent           : Max number of simultaneously open positions (5).
    max_daily_trades         : Daily trade-count circuit breaker (20).
    daily_dd_reduce          : Daily DD fraction triggering size reduction (2%).
    daily_dd_halt            : Daily DD fraction halting trading (3%).
    weekly_dd_reduce         : Weekly DD fraction triggering size reduction (5%).
    weekly_dd_halt           : Weekly DD fraction halting trading (7%).
    max_dd_from_peak         : Peak-to-trough DD fraction triggering full halt (10%).
    min_position_usd         : Minimum position value in USD ($100).
    overnight_gap_multiple   : Stop gap-through multiplier for overnight sizing (3x).
    max_overnight_loss_pct   : Max overnight loss as fraction of equity (2%).
    correlation_reduce_thr   : Correlation above which size is halved (0.70).
    correlation_reject_thr   : Correlation above which trade is rejected (0.85).
    correlation_window       : Rolling window bars for correlation estimate (60).
    duplicate_window_secs    : Block same symbol+direction within N seconds (60).
    max_spread_pct           : Max bid-ask spread fraction to accept (0.5%).
    """

    def __init__(
        self,
        initial_equity:           float = 100_000.0,
        max_risk_per_trade:       float = 0.01,
        max_exposure:             float = 0.80,
        max_leverage:             float = 1.25,
        max_single_position:      float = 0.15,
        max_correlated_exposure:  float = 0.30,
        max_concurrent:           int   = 5,
        max_daily_trades:         int   = 20,
        daily_dd_reduce:          float = 0.02,
        daily_dd_halt:            float = 0.03,
        weekly_dd_reduce:         float = 0.05,
        weekly_dd_halt:           float = 0.07,
        max_dd_from_peak:         float = 0.10,
        min_position_usd:         float = 100.0,
        overnight_gap_multiple:   float = 3.0,
        max_overnight_loss_pct:   float = 0.02,
        correlation_reduce_thr:   float = 0.70,
        correlation_reject_thr:   float = 0.85,
        correlation_window:       int   = 60,
        duplicate_window_secs:    int   = 60,
        max_spread_pct:           float = 0.005,
        allow_fractional_shares:  bool  = False,
        fractional_precision:     int   = 6,
    ) -> None:
        self.initial_equity           = initial_equity
        self.max_risk_per_trade       = max_risk_per_trade
        self.max_exposure             = max_exposure
        self.max_leverage             = max_leverage
        self.max_single_position      = max_single_position
        self.max_correlated_exposure  = max_correlated_exposure
        self.max_concurrent           = max_concurrent
        self.max_daily_trades         = max_daily_trades
        self.min_position_usd         = min_position_usd
        self.overnight_gap_multiple   = overnight_gap_multiple
        self.max_overnight_loss_pct   = max_overnight_loss_pct
        self.correlation_reduce_thr   = correlation_reduce_thr
        self.correlation_reject_thr   = correlation_reject_thr
        self.correlation_window       = correlation_window
        self.duplicate_window_secs    = duplicate_window_secs
        self.max_spread_pct           = max_spread_pct
        self.allow_fractional_shares  = allow_fractional_shares
        self.fractional_precision     = max(0, int(fractional_precision))

        # Equity tracking
        self._current_equity:       float            = initial_equity
        self._peak_equity:          float            = initial_equity
        self._daily_start_equity:   float            = initial_equity
        self._weekly_start_equity:  float            = initial_equity
        self._daily_trade_count:    int              = 0
        self._trading_state:        TradingState     = TradingState.NORMAL
        self._last_reset_date:      Optional[dt.date] = None
        self._last_reset_week:      Optional[int]    = None   # ISO week number

        # Circuit breaker sub-system
        self.circuit_breaker = CircuitBreaker(
            daily_reduce  = daily_dd_reduce,
            daily_halt    = daily_dd_halt,
            weekly_reduce = weekly_dd_reduce,
            weekly_halt   = weekly_dd_halt,
            peak_halt     = max_dd_from_peak,
        )

    # ======================================================================= #
    # Primary public API                                                       #
    # ======================================================================= #

    def validate_signal(
        self,
        signal,
        portfolio_state: PortfolioState,
        is_overnight:    bool = False,
        bid:             Optional[float] = None,
        ask:             Optional[float] = None,
        timestamp:       Optional[dt.datetime] = None,
    ) -> RiskDecision:
        """
        Full gate: run every check in order; return on first hard rejection.

        Soft modifications (size reduction) accumulate; the adjusted signal is
        returned with the modifications list populated.

        Parameters
        ----------
        signal          : Signal dataclass from regime_strategies.
        portfolio_state : Current snapshot of the portfolio.
        is_overnight    : True if this order will be held overnight.
        bid / ask       : Current bid/ask for spread check (optional).
        timestamp       : Wall-clock time of the check (defaults to utcnow).
        """
        ts = timestamp or dt.datetime.utcnow()
        ps = portfolio_state
        modifications: List[str] = []

        # ------------------------------------------------------------------ #
        # 0. Lock file -- manual halt in place                                #
        # ------------------------------------------------------------------ #
        if LOCK_FILE.exists():
            return self._reject(
                signal, RejectionReason.LOCK_FILE_EXISTS,
                f"Manual halt lock file present: {LOCK_FILE}  "
                "Delete the file to resume trading."
            )

        # ------------------------------------------------------------------ #
        # 1. Circuit breaker -- ABSOLUTE VETO if halted                      #
        # ------------------------------------------------------------------ #
        active_cb = self.circuit_breaker.get_active()
        if self.circuit_breaker.is_halted():
            return self._reject(
                signal, RejectionReason.CIRCUIT_BREAKER_HALTED,
                f"Circuit breaker active: {active_cb.value}  "
                f"regime={ps.current_regime}"
            )

        # ------------------------------------------------------------------ #
        # 2. Daily trade count                                                #
        # ------------------------------------------------------------------ #
        if self._daily_trade_count >= self.max_daily_trades:
            return self._reject(
                signal, RejectionReason.MAX_DAILY_TRADES,
                f"Daily trade limit reached ({self.max_daily_trades})"
            )

        # ------------------------------------------------------------------ #
        # 3. Stop-loss mandatory                                              #
        # ------------------------------------------------------------------ #
        if signal.stop_loss is None or signal.stop_loss <= 0:
            return self._reject(
                signal, RejectionReason.NO_STOP_LOSS,
                "Every position must have a stop loss -- order blocked"
            )

        # ------------------------------------------------------------------ #
        # 4. Duplicate order guard                                            #
        # ------------------------------------------------------------------ #
        dup_key = f"{signal.symbol}:{signal.direction}"
        last_ts = ps.last_order_times.get(dup_key)
        if last_ts is not None:
            elapsed = (ts - last_ts).total_seconds()
            if elapsed < self.duplicate_window_secs:
                return self._reject(
                    signal, RejectionReason.DUPLICATE_ORDER,
                    f"Duplicate {signal.symbol} {signal.direction} within "
                    f"{elapsed:.0f}s (window={self.duplicate_window_secs}s)"
                )

        # ------------------------------------------------------------------ #
        # 5. Bid-ask spread check                                             #
        # ------------------------------------------------------------------ #
        if bid is not None and ask is not None and bid > 0:
            spread_pct = (ask - bid) / bid
            if spread_pct > self.max_spread_pct:
                return self._reject(
                    signal, RejectionReason.SPREAD_TOO_WIDE,
                    f"Spread {spread_pct:.3%} > limit {self.max_spread_pct:.3%}"
                )

        # ------------------------------------------------------------------ #
        # 6. Concurrent position limit                                        #
        # ------------------------------------------------------------------ #
        n_open            = len(ps.positions)
        symbol_already_open = signal.symbol in ps.positions
        if not symbol_already_open and n_open >= self.max_concurrent:
            return self._reject(
                signal, RejectionReason.MAX_CONCURRENT,
                f"Already at max concurrent positions ({self.max_concurrent})"
            )

        # ------------------------------------------------------------------ #
        # 7. Compute target position size via 1% risk rule                    #
        # ------------------------------------------------------------------ #
        size_pct = self._compute_size_pct(signal, ps.equity)

        # Circuit breaker REDUCE -> halve size
        if self.circuit_breaker.is_reduced():
            size_pct *= 0.5
            modifications.append(
                f"Size halved: circuit breaker active ({active_cb.value})"
            )

        # Leverage rules
        size_pct, lev_mod = self._apply_leverage_rules(size_pct, signal, ps)
        if lev_mod:
            modifications.append(lev_mod)

        # ------------------------------------------------------------------ #
        # 8. Overnight gap risk adjustment                                    #
        # ------------------------------------------------------------------ #
        if is_overnight:
            size_pct, gap_mod = self._apply_gap_risk(
                size_pct, signal.entry_price, signal.stop_loss, ps.equity
            )
            if gap_mod:
                modifications.append(gap_mod)

        # ------------------------------------------------------------------ #
        # 9. Portfolio exposure cap                                           #
        # ------------------------------------------------------------------ #
        current_exposure = sum(abs(v) for v in ps.positions.values()) / max(ps.equity, 1e-9)
        new_exposure     = current_exposure + size_pct
        if new_exposure > self.max_exposure:
            size_pct = max(0.0, self.max_exposure - current_exposure)
            if size_pct * ps.equity < self.min_position_usd:
                return self._reject(
                    signal, RejectionReason.MAX_EXPOSURE,
                    f"Total exposure {new_exposure:.1%} would exceed "
                    f"limit {self.max_exposure:.1%}"
                )
            modifications.append(
                f"Size capped: exposure limit {self.max_exposure:.1%}"
            )

        # ------------------------------------------------------------------ #
        # 10. Single-position concentration cap                               #
        # ------------------------------------------------------------------ #
        existing_sym_pct = abs(ps.positions.get(signal.symbol, 0.0)) / max(ps.equity, 1e-9)
        total_sym_pct    = existing_sym_pct + size_pct
        if total_sym_pct > self.max_single_position:
            size_pct = max(0.0, self.max_single_position - existing_sym_pct)
            if size_pct * ps.equity < self.min_position_usd:
                return self._reject(
                    signal, RejectionReason.MAX_SINGLE_POSITION,
                    f"{signal.symbol} concentration {total_sym_pct:.1%} would exceed "
                    f"limit {self.max_single_position:.1%}"
                )
            modifications.append(
                f"Size capped: single-position limit {self.max_single_position:.1%}"
            )

        # ------------------------------------------------------------------ #
        # 11. Correlation check                                               #
        # ------------------------------------------------------------------ #
        size_pct, corr_mod = self._check_correlation(signal.symbol, size_pct, ps)
        if corr_mod:
            if corr_mod.startswith("REJECT"):
                return self._reject(
                    signal, RejectionReason.CORRELATION_TOO_HIGH, corr_mod
                )
            modifications.append(corr_mod)

        # ------------------------------------------------------------------ #
        # 12. Minimum position check                                          #
        # ------------------------------------------------------------------ #
        if size_pct * ps.equity < self.min_position_usd:
            return self._reject(
                signal, RejectionReason.BELOW_MIN_POSITION,
                f"Computed position ${size_pct * ps.equity:.2f} < "
                f"minimum ${self.min_position_usd:.2f}"
            )

        # ------------------------------------------------------------------ #
        # 13. Buying power check                                              #
        # ------------------------------------------------------------------ #
        required_cash = size_pct * ps.equity
        if required_cash > ps.buying_power:
            return self._reject(
                signal, RejectionReason.INSUFFICIENT_BUYING_PWR,
                f"Required ${required_cash:.2f} > buying power ${ps.buying_power:.2f}"
            )

        # ------------------------------------------------------------------ #
        # All checks passed -- build modified signal                          #
        # ------------------------------------------------------------------ #
        mod_signal = copy.copy(signal)
        mod_signal.position_size_pct = size_pct

        if modifications:
            logger.info(
                "RISK_MANAGER approved %s with %d modification(s): %s",
                signal.symbol, len(modifications), " | ".join(modifications)
            )
        else:
            logger.debug("RISK_MANAGER approved %s (no modifications)", signal.symbol)

        return RiskDecision(
            approved         = True,
            modified_signal  = mod_signal,
            modifications    = modifications,
        )

    # ----------------------------------------------------------------------- #

    def check_trade(
        self,
        symbol:             str,
        proposed_size:      float,
        price:              float,
        current_positions:  Dict[str, float],
        stop_price:         Optional[float] = None,
    ) -> RiskCheck:
        """
        Simplified check for the backtester (no Signal object required).

        Parameters
        ----------
        symbol            : ticker
        proposed_size     : shares (positive = buy, negative = sell/close)
        price             : execution price estimate
        current_positions : symbol -> current dollar value
        stop_price        : optional stop-loss price for risk sizing
        """
        equity = self._current_equity

        if LOCK_FILE.exists():
            return RiskCheck(False, TradingState.HALTED, 0.0,
                             RejectionReason.LOCK_FILE_EXISTS.value)

        if self.circuit_breaker.is_halted():
            return RiskCheck(False, TradingState.HALTED, 0.0,
                             RejectionReason.CIRCUIT_BREAKER_HALTED.value)

        if self._daily_trade_count >= self.max_daily_trades:
            return RiskCheck(False, self._trading_state, 0.0,
                             RejectionReason.MAX_DAILY_TRADES.value)

        adjusted = abs(proposed_size)

        # Risk-based sizing if stop provided
        if stop_price is not None and stop_price > 0:
            risk_per_share = abs(price - stop_price)
            if risk_per_share > 0:
                max_risk_dollars  = equity * self.max_risk_per_trade
                risk_based_shares = max_risk_dollars / risk_per_share
                adjusted = min(adjusted, risk_based_shares)

        # State discount
        if self.circuit_breaker.is_reduced():
            adjusted *= 0.5

        # Exposure cap
        current_exp      = sum(abs(v) for v in current_positions.values())
        max_new_dollars  = max(0.0, equity * self.max_exposure - current_exp)
        adjusted         = min(adjusted, max_new_dollars / max(price, 1e-9))

        # Single-position cap
        existing_val      = abs(current_positions.get(symbol, 0.0))
        max_sym_dollars   = equity * self.max_single_position - existing_val
        adjusted          = min(adjusted, max(0.0, max_sym_dollars) / max(price, 1e-9))

        # Concurrent positions
        if symbol not in current_positions and len(current_positions) >= self.max_concurrent:
            return RiskCheck(False, self._trading_state, 0.0,
                             RejectionReason.MAX_CONCURRENT.value)

        # Minimum position
        if adjusted * price < self.min_position_usd:
            return RiskCheck(False, self._trading_state, 0.0,
                             RejectionReason.BELOW_MIN_POSITION.value)

        adjusted = adjusted if proposed_size >= 0 else -adjusted
        state    = TradingState.REDUCED if self.circuit_breaker.is_reduced() else TradingState.NORMAL
        return RiskCheck(True, state, adjusted)

    # ----------------------------------------------------------------------- #

    def compute_position_size(
        self,
        target_weight:  float,
        price:          float,
        stop_price:     Optional[float] = None,
    ) -> float:
        """
        Compute whole shares satisfying target_weight AND 1% risk-rule cap.

        Position size = min(weight-based, risk-based).
        Returns whole shares by default (floor). Set
        ``allow_fractional_shares=True`` at construction to keep fractional
        precision (Alpaca supports fractional for equities + crypto).
        """
        equity               = self._current_equity
        weight_based_dollars = equity * min(target_weight, self.max_single_position)
        weight_based_shares  = weight_based_dollars / max(price, 1e-9)

        if stop_price is not None and stop_price > 0:
            risk_per_share   = abs(price - stop_price)
            if risk_per_share > 0:
                max_risk_dollars  = equity * self.max_risk_per_trade
                risk_based_shares = max_risk_dollars / risk_per_share
                weight_based_shares = min(weight_based_shares, risk_based_shares)

        if self.allow_fractional_shares:
            return round(float(weight_based_shares), self.fractional_precision)
        return float(int(weight_based_shares))   # floor to whole shares

    # ----------------------------------------------------------------------- #

    def update_equity(
        self,
        equity:    float,
        timestamp: Optional[dt.datetime] = None,
        positions: Optional[Dict[str, float]] = None,
        regime:    str = "UNKNOWN",
    ) -> None:
        """
        Record current portfolio equity, update drawdown metrics, fire
        circuit breakers if thresholds crossed, auto-reset on date boundaries.

        Parameters
        ----------
        equity    : Current total portfolio value.
        timestamp : Bar timestamp (defaults to utcnow).
        positions : Current open positions (for circuit breaker logging).
        regime    : Current HMM regime label (for logging).
        """
        ts       = timestamp or dt.datetime.utcnow()
        today    = ts.date()
        iso_week = ts.isocalendar()[1]

        # Auto-reset boundaries
        if self._last_reset_date is None or today != self._last_reset_date:
            self.daily_reset(today)
            if self._last_reset_week is None or iso_week != self._last_reset_week:
                self.weekly_reset()
                self._last_reset_week = iso_week

        # Update peak
        self._current_equity = equity
        if equity > self._peak_equity:
            self._peak_equity = equity

        # Drawdown fractions (negative when below baseline)
        daily_dd  = (equity - self._daily_start_equity)  / max(self._daily_start_equity, 1e-9)
        weekly_dd = (equity - self._weekly_start_equity) / max(self._weekly_start_equity, 1e-9)
        peak_dd   = (equity - self._peak_equity)         / max(self._peak_equity, 1e-9)

        self.circuit_breaker.update(
            daily_dd  = daily_dd,
            weekly_dd = weekly_dd,
            peak_dd   = peak_dd,
            equity    = equity,
            positions = positions or {},
            regime    = regime,
            timestamp = ts,
        )

        self._trading_state = self._evaluate_drawdown_state()

    # ----------------------------------------------------------------------- #

    def get_drawdown_state(self) -> DrawdownState:
        """Return a snapshot of current drawdown metrics."""
        equity    = self._current_equity
        daily_dd  = (equity - self._daily_start_equity)  / max(self._daily_start_equity, 1e-9)
        weekly_dd = (equity - self._weekly_start_equity) / max(self._weekly_start_equity, 1e-9)
        peak_dd   = (equity - self._peak_equity)         / max(self._peak_equity, 1e-9)
        return DrawdownState(
            peak_equity         = self._peak_equity,
            current_equity      = equity,
            daily_start_equity  = self._daily_start_equity,
            weekly_start_equity = self._weekly_start_equity,
            dd_from_peak        = peak_dd,
            daily_dd            = daily_dd,
            weekly_dd           = weekly_dd,
        )

    def get_trading_state(self) -> TradingState:
        return self._trading_state

    def daily_reset(self, date: Optional[dt.date] = None) -> None:
        """Reset daily trade counter and daily drawdown baseline."""
        self._daily_trade_count  = 0
        self._daily_start_equity = self._current_equity
        self._last_reset_date    = date or dt.date.today()
        self.circuit_breaker.reset_daily()
        logger.debug("RiskManager: daily reset (equity=%.2f)", self._current_equity)

    def weekly_reset(self) -> None:
        """Reset weekly drawdown baseline (first bar of new ISO week)."""
        self._weekly_start_equity = self._current_equity
        self.circuit_breaker.reset_weekly()
        logger.debug("RiskManager: weekly reset (equity=%.2f)", self._current_equity)

    def increment_trade_count(self) -> None:
        """Call after each successful order placement."""
        self._daily_trade_count += 1
        logger.debug(
            "RiskManager: trade count %d / %d",
            self._daily_trade_count, self.max_daily_trades,
        )

    # ======================================================================= #
    # Private helpers                                                          #
    # ======================================================================= #

    def _evaluate_drawdown_state(self) -> TradingState:
        if self.circuit_breaker.is_halted():  return TradingState.HALTED
        if self.circuit_breaker.is_reduced(): return TradingState.REDUCED
        return TradingState.NORMAL

    def _compute_size_pct(self, signal, equity: float) -> float:
        """
        Compute final size_pct from signal, applying the 1% risk rule.

        Position size = (equity * 0.01) / |entry - stop|
        Then cap at min(regime-requested, risk-based, single-pos limit).
        """
        entry    = signal.entry_price
        stop     = signal.stop_loss
        risk_pct = abs(entry - stop) / max(entry, 1e-9)

        if risk_pct > 0:
            risk_based_size = self.max_risk_per_trade / risk_pct
        else:
            risk_based_size = signal.position_size_pct

        return max(0.0, min(
            signal.position_size_pct,
            risk_based_size,
            self.max_single_position,
        ))

    def _apply_leverage_rules(
        self,
        size_pct: float,
        signal,
        ps:       PortfolioState,
    ) -> Tuple[float, Optional[str]]:
        """
        Enforce leverage rules.  Returns (adjusted_size_pct, modification_note).

        Force 1.0x leverage (undo any leverage already baked into size_pct) if:
          - any circuit breaker is active
          - HMM flicker_rate >= 3 (uncertain regime)
          - 3 or more positions already open
          - the signal's own leverage field is 1.0x
        """
        cb_active  = self.circuit_breaker.get_active() is not None
        flickering = ps.flicker_rate >= 3
        too_many   = len(ps.positions) >= 3
        sig_1x     = signal.leverage <= 1.0

        if not sig_1x and (cb_active or flickering or too_many):
            # Undo leverage baked into the signal's size
            size_pct = size_pct / signal.leverage
            reasons  = []
            if cb_active:  reasons.append(f"cb={self.circuit_breaker.get_active().value}")
            if flickering: reasons.append(f"flicker={ps.flicker_rate}")
            if too_many:   reasons.append(f"open_pos={len(ps.positions)}")
            return size_pct, f"Leverage forced 1.0x ({', '.join(reasons)})"

        return size_pct, None

    def _apply_gap_risk(
        self,
        size_pct:   float,
        entry:      float,
        stop:       float,
        equity:     float,
    ) -> Tuple[float, Optional[str]]:
        """
        Reduce overnight position so a 3x stop gap-through costs at most
        max_overnight_loss_pct of equity.

        effective_risk_per_share = |entry - stop| * overnight_gap_multiple
        max_shares = (equity * max_overnight_loss_pct) / effective_risk_per_share
        """
        effective_risk = abs(entry - stop) * self.overnight_gap_multiple
        if effective_risk <= 0 or entry <= 0:
            return size_pct, None

        max_loss        = equity * self.max_overnight_loss_pct
        max_night_shr   = max_loss / effective_risk
        normal_shr      = size_pct * equity / max(entry, 1e-9)

        if max_night_shr < normal_shr:
            adjusted = (max_night_shr * entry) / equity
            return adjusted, (
                f"Overnight gap risk: {size_pct:.2%} -> {adjusted:.2%} "
                f"(3x gap cap, max_loss={self.max_overnight_loss_pct:.1%})"
            )
        return size_pct, None

    def _check_correlation(
        self,
        symbol:   str,
        size_pct: float,
        ps:       PortfolioState,
    ) -> Tuple[float, Optional[str]]:
        """
        60-day rolling correlation of symbol vs each existing position.

        - corr > 0.85: reject trade
        - corr > 0.70: halve size
        Returns (adjusted_size_pct, note).  Note starts with "REJECT" if veto.
        """
        if not ps.price_history or symbol not in ps.price_history:
            return size_pct, None

        new_series = ps.price_history[symbol].dropna()
        if len(new_series) < self.correlation_window:
            return size_pct, None

        max_corr = 0.0
        for existing_sym in ps.positions:
            if existing_sym == symbol:
                continue
            if existing_sym not in ps.price_history:
                continue
            other = ps.price_history[existing_sym].dropna()
            aligned = pd.concat(
                [new_series.rename("a"), other.rename("b")], axis=1
            ).dropna().tail(self.correlation_window)
            if len(aligned) < 20:
                continue
            corr = float(aligned["a"].corr(aligned["b"]))
            if not np.isnan(corr) and corr > max_corr:
                max_corr = corr

        if max_corr >= self.correlation_reject_thr:
            return size_pct, (
                f"REJECT: {symbol} max_corr={max_corr:.2f} >= "
                f"reject_thr={self.correlation_reject_thr:.2f}"
            )
        if max_corr >= self.correlation_reduce_thr:
            reduced = size_pct * 0.5
            return reduced, (
                f"Correlation {max_corr:.2f} >= {self.correlation_reduce_thr:.2f}: "
                f"size halved ({size_pct:.2%} -> {reduced:.2%})"
            )
        return size_pct, None

    def _check_exposure(
        self,
        proposed_size:     float,
        price:             float,
        current_positions: Dict[str, float],
    ) -> Tuple[bool, Optional[str]]:
        """Verify gross exposure + leverage limits."""
        current_exp  = sum(abs(v) for v in current_positions.values())
        new_exp_frac = (current_exp + proposed_size * price) / max(self._current_equity, 1e-9)
        if new_exp_frac > self.max_leverage:
            return False, f"Leverage {new_exp_frac:.2f}x > limit {self.max_leverage:.2f}x"
        if new_exp_frac > self.max_exposure:
            return False, f"Exposure {new_exp_frac:.1%} > limit {self.max_exposure:.1%}"
        return True, None

    def _check_concentration(
        self,
        symbol:            str,
        proposed_value:    float,
        current_positions: Dict[str, float],
    ) -> Tuple[bool, Optional[str]]:
        """Verify single-symbol concentration limit."""
        existing  = abs(current_positions.get(symbol, 0.0))
        total_pct = (existing + proposed_value) / max(self._current_equity, 1e-9)
        if total_pct > self.max_single_position:
            return False, (
                f"{symbol} concentration {total_pct:.1%} > "
                f"limit {self.max_single_position:.1%}"
            )
        return True, None

    def _apply_state_discount(self, size: float) -> float:
        """Halve size when state is REDUCED."""
        if self._trading_state == TradingState.REDUCED:
            return size * 0.5
        return size

    @staticmethod
    def _reject(signal, reason: RejectionReason, detail: str) -> RiskDecision:
        logger.warning(
            "RISK_MANAGER rejected %s | reason=%s | %s",
            getattr(signal, "symbol", "?"), reason.value, detail,
        )
        return RiskDecision(
            approved         = False,
            modified_signal  = None,
            rejection_reason = f"{reason.value}: {detail}",
        )

    # ----------------------------------------------------------------------- #
    # Class-method factory                                                     #
    # ----------------------------------------------------------------------- #

    @classmethod
    def from_config(
        cls,
        config:         dict,
        initial_equity: float = 100_000.0,
    ) -> "RiskManager":
        """
        Construct a RiskManager from the ``risk`` section of settings.yaml.

        Parameters
        ----------
        config         : Full parsed settings dict (from load_config()).
        initial_equity : Starting portfolio value.
        """
        r = config.get("risk", {})
        return cls(
            initial_equity           = initial_equity,
            max_risk_per_trade       = float(r.get("max_risk_per_trade",       0.01)),
            max_exposure             = float(r.get("max_exposure",             0.80)),
            max_leverage             = float(r.get("max_leverage",             1.25)),
            max_single_position      = float(r.get("max_single_position",      0.15)),
            max_correlated_exposure  = float(r.get("max_correlated_exposure",  0.30)),
            max_concurrent           = int(  r.get("max_concurrent",           5)),
            max_daily_trades         = int(  r.get("max_daily_trades",         20)),
            daily_dd_reduce          = float(r.get("daily_dd_reduce",          0.02)),
            daily_dd_halt            = float(r.get("daily_dd_halt",            0.03)),
            weekly_dd_reduce         = float(r.get("weekly_dd_reduce",         0.05)),
            weekly_dd_halt           = float(r.get("weekly_dd_halt",           0.07)),
            max_dd_from_peak         = float(r.get("max_dd_from_peak",         0.10)),
        )


# --------------------------------------------------------------------------- #
# PortfolioRiskManager                                                         #
# --------------------------------------------------------------------------- #

class PortfolioRiskManager:
    """
    Portfolio-level risk gate that sits ABOVE per-strategy RiskManagers.

    HIERARCHY
    ---------
    Signal from Strategy_X
      → Strategy_X.risk_manager.validate_signal()   (per-strategy)
      → PortfolioRiskManager.validate_signal()       (portfolio-wide)  ← this class
      → OrderExecutor

    Both layers must approve a signal for an order to go through.
    This class knows nothing about HMM or strategy internals — it only sees
    the aggregate portfolio state and each incoming signal.

    STATE TRACKED
    -------------
    _strategy_positions : {strategy_name: {symbol: dollar_value}}
        Updated by the main loop after every fill via update_strategy_positions().
        Used to compute per-strategy symbol exposure and aggregate totals.

    CHECKS (applied in order, first failure returns)
    -------------------------------------------------
    1. Portfolio DD halt       — peak DD > 10% → lock file + reject all
    2. Portfolio DD reduce     — daily DD > 3% → reject; > 2% → halve size
    3. Aggregate exposure      — sum all positions > 80% → reduce or reject
    4. Symbol concentration    — any symbol across strategies > 15% → reduce
    5. Total leverage          — gross exposure / equity > 1.25x → reject
    6. Correlation cluster     — correlated cluster > 30% exposure → reject

    Parameters
    ----------
    max_aggregate_exposure  : Total gross exposure cap (default 0.80)
    max_single_symbol       : Single symbol cap across all strategies (default 0.15)
    max_portfolio_leverage  : Hard leverage ceiling (default 1.25)
    max_corr_cluster        : Max exposure in a correlated cluster (default 0.30)
    corr_cluster_threshold  : ρ above which two symbols are in the same cluster (0.70)
    correlation_window      : Rolling bars for correlation estimate (60)
    daily_dd_reduce         : Daily DD fraction that halves sizes (0.02)
    daily_dd_halt           : Daily DD fraction that halts new trades (0.03)
    max_dd_from_peak        : Peak DD fraction that halts + writes lock file (0.10)
    min_position_usd        : Minimum residual position after size reduction ($100)
    """

    def __init__(
        self,
        max_aggregate_exposure: float = 0.80,
        max_single_symbol:      float = 0.15,
        max_portfolio_leverage: float = 1.25,
        max_corr_cluster:       float = 0.30,
        corr_cluster_threshold: float = 0.70,
        correlation_window:     int   = 60,
        daily_dd_reduce:        float = 0.02,
        daily_dd_halt:          float = 0.03,
        max_dd_from_peak:       float = 0.10,
        min_position_usd:       float = 100.0,
    ) -> None:
        self.max_aggregate_exposure = max_aggregate_exposure
        self.max_single_symbol      = max_single_symbol
        self.max_portfolio_leverage = max_portfolio_leverage
        self.max_corr_cluster       = max_corr_cluster
        self.corr_cluster_threshold = corr_cluster_threshold
        self.correlation_window     = correlation_window
        self.daily_dd_reduce        = daily_dd_reduce
        self.daily_dd_halt          = daily_dd_halt
        self.max_dd_from_peak       = max_dd_from_peak
        self.min_position_usd       = min_position_usd

        # {strategy_name: {symbol: dollar_value}}
        self._strategy_positions: Dict[str, Dict[str, float]] = {}

    # ── State management ───────────────────────────────────────────────────────

    def update_strategy_positions(
        self,
        strategy_name: str,
        positions: Dict[str, float],
    ) -> None:
        """
        Sync the portfolio manager's view of one strategy's open positions.

        Call this after every fill or at the start of each bar.
        ``positions`` maps symbol → current dollar value (positive = long).
        """
        self._strategy_positions[strategy_name] = dict(positions)

    def get_aggregate_positions(self) -> Dict[str, float]:
        """
        Sum all strategies' dollar values per symbol.

        Returns {symbol: total_dollar_value_across_strategies}.
        """
        agg: Dict[str, float] = {}
        for pos_map in self._strategy_positions.values():
            for sym, val in pos_map.items():
                agg[sym] = agg.get(sym, 0.0) + val
        return agg

    # ── Primary public API ─────────────────────────────────────────────────────

    def validate_signal(
        self,
        signal,
        strategy_name: str,
        portfolio_state: PortfolioState,
    ) -> RiskDecision:
        """
        Run all portfolio-level checks against *signal* from *strategy_name*.

        Returns a :class:`RiskDecision`.  If approved with modifications, the
        modified signal carries the adjusted ``position_size_pct``.

        The caller must also pass the signal through the per-strategy
        RiskManager first — this method does NOT repeat those checks.
        """
        working_signal = copy.copy(signal)

        # ── 1. Portfolio DD (no signal arg needed) ─────────────────────────────
        dd_result = self.check_portfolio_dd(portfolio_state)
        if dd_result is not None:
            if not dd_result.approved:
                logger.warning(
                    "PORTFOLIO_RISK rejected %s from '%s' | %s",
                    signal.symbol, strategy_name, dd_result.rejection_reason,
                )
                return dd_result
            # Soft: daily_dd_reduce → halve the working signal's size
            if dd_result.modifications:
                working_signal = copy.copy(working_signal)
                working_signal.position_size_pct = round(
                    working_signal.position_size_pct * 0.5, 6
                )
                logger.info(
                    "PORTFOLIO_RISK modified %s from '%s': %s",
                    signal.symbol, strategy_name,
                    " | ".join(dd_result.modifications),
                )

        # ── 2-5. Signal-level checks ───────────────────────────────────────────
        signal_checks = [
            self.check_aggregate_exposure,
            self.check_symbol_aggregation,
            self.check_total_leverage,
            self.check_correlation_cluster,
        ]

        for check_fn in signal_checks:
            result = check_fn(working_signal, portfolio_state)

            if result is None:
                continue

            if not result.approved:
                logger.warning(
                    "PORTFOLIO_RISK rejected %s from '%s' | %s",
                    signal.symbol, strategy_name, result.rejection_reason,
                )
                return result

            if result.modified_signal is not None:
                working_signal = result.modified_signal
            if result.modifications:
                logger.info(
                    "PORTFOLIO_RISK modified %s from '%s': %s",
                    signal.symbol, strategy_name,
                    " | ".join(result.modifications),
                )

        return RiskDecision(
            approved        = True,
            modified_signal = working_signal,
        )

    # ── Individual checks ──────────────────────────────────────────────────────

    def check_portfolio_dd(
        self,
        portfolio_state: PortfolioState,
    ) -> Optional[RiskDecision]:
        """
        Fire on total portfolio drawdown, independent of per-strategy managers.

        Peak DD > 10%  → halt + write lock file → hard rejection
        Daily DD > 3%  → hard rejection (rest of session)
        Daily DD > 2%  → halve size on current signal (soft, handled by caller)

        Returns None when no DD threshold is breached.
        Returns a RiskDecision with approved=False for halt/reject, or
        approved=True with a halving note so the caller can apply it.
        """
        ps = portfolio_state
        equity      = ps.equity
        peak        = ps.peak_equity if ps.peak_equity > 0 else equity
        daily_start = ps.daily_start if ps.daily_start > 0 else equity

        peak_dd  = (equity - peak) / max(peak, 1e-9)           # negative = loss
        daily_dd = (equity - daily_start) / max(daily_start, 1e-9)

        if peak_dd <= -self.max_dd_from_peak:
            if not LOCK_FILE.exists():
                LOCK_FILE.write_text(
                    f"Portfolio peak DD halt  peak_dd={peak_dd:.2%}  "
                    f"equity={equity:.2f}  "
                    f"ts={dt.datetime.utcnow().isoformat()}\n"
                )
                logger.error(
                    "PORTFOLIO PEAK DD HALT: peak_dd=%.2f%%  lock file written -> %s",
                    peak_dd * 100, LOCK_FILE,
                )
            return self._reject(
                RejectionReason.PORTFOLIO_PEAK_DD,
                f"Peak drawdown {peak_dd:.2%} <= -{self.max_dd_from_peak:.0%} threshold",
            )

        if daily_dd <= -self.daily_dd_halt:
            return self._reject(
                RejectionReason.PORTFOLIO_DD_HALT,
                f"Daily drawdown {daily_dd:.2%} <= -{self.daily_dd_halt:.0%} threshold",
            )

        # Soft: daily DD > reduce threshold → halve size via caller
        # Return None here; the caller (validate_signal loop) can't apply a
        # halve without a concrete signal.  Instead we return a special approved
        # decision carrying a "halve" flag in modifications so the outer loop
        # reduces the working signal's size.
        if daily_dd <= -self.daily_dd_reduce:
            return RiskDecision(
                approved        = True,
                modified_signal = None,   # signal not yet available here
                modifications   = [
                    f"portfolio_dd_reduce: daily_dd={daily_dd:.2%} — sizes halved"
                ],
            )

        return None

    def check_aggregate_exposure(
        self,
        signal,
        portfolio_state: PortfolioState,
    ) -> Optional[RiskDecision]:
        """
        Reject or reduce when all strategies' combined gross exposure would
        exceed ``max_aggregate_exposure`` (default 80%) of equity.
        """
        equity = max(portfolio_state.equity, 1e-9)
        agg    = self.get_aggregate_positions()

        current_gross = sum(abs(v) for v in agg.values()) / equity
        proposed_add  = signal.position_size_pct  # fraction of equity

        if current_gross + proposed_add <= self.max_aggregate_exposure:
            return None

        # How much room is left?
        headroom = max(0.0, self.max_aggregate_exposure - current_gross)
        if headroom * equity < self.min_position_usd:
            return self._reject(
                RejectionReason.PORTFOLIO_EXPOSURE,
                f"Aggregate exposure {current_gross:.1%} + {proposed_add:.1%} "
                f"> limit {self.max_aggregate_exposure:.0%} "
                f"(strategy={signal.strategy_name})",
            )

        mod = copy.copy(signal)
        mod.position_size_pct = round(headroom, 6)
        return RiskDecision(
            approved        = True,
            modified_signal = mod,
            modifications   = [
                f"Aggregate exposure cap {self.max_aggregate_exposure:.0%}: "
                f"size {proposed_add:.2%} → {headroom:.2%}"
            ],
        )

    def check_symbol_aggregation(
        self,
        signal,
        portfolio_state: PortfolioState,
    ) -> Optional[RiskDecision]:
        """
        Enforce a single-symbol cap across ALL strategies combined.

        If Strategy_A holds SPY at 10% and Strategy_B tries to add SPY at 8%,
        the combined 18% would breach the 15% cap.  Strategy_B's size is
        reduced to max(0, 15% - 10%) = 5%.
        """
        equity = max(portfolio_state.equity, 1e-9)
        agg    = self.get_aggregate_positions()

        current_sym_pct = abs(agg.get(signal.symbol, 0.0)) / equity
        proposed_add    = signal.position_size_pct

        if current_sym_pct + proposed_add <= self.max_single_symbol:
            return None

        headroom = max(0.0, self.max_single_symbol - current_sym_pct)
        if headroom * equity < self.min_position_usd:
            return self._reject(
                RejectionReason.PORTFOLIO_SYMBOL_CAP,
                f"{signal.symbol} cross-strategy exposure "
                f"{current_sym_pct:.1%} + {proposed_add:.1%} "
                f"> cap {self.max_single_symbol:.0%}",
            )

        mod = copy.copy(signal)
        mod.position_size_pct = round(headroom, 6)
        return RiskDecision(
            approved        = True,
            modified_signal = mod,
            modifications   = [
                f"Symbol cap {self.max_single_symbol:.0%} ({signal.symbol} "
                f"cross-strategy): {proposed_add:.2%} → {headroom:.2%}"
            ],
        )

    def check_total_leverage(
        self,
        signal,
        portfolio_state: PortfolioState,
    ) -> Optional[RiskDecision]:
        """
        Reject when the proposed position would push total portfolio leverage
        above ``max_portfolio_leverage`` (1.25x).

        leverage = (gross_exposure + new_position_dollars) / equity
        """
        equity = max(portfolio_state.equity, 1e-9)
        agg    = self.get_aggregate_positions()

        current_gross = sum(abs(v) for v in agg.values())
        new_dollars   = signal.position_size_pct * equity * signal.leverage
        prospective   = (current_gross + new_dollars) / equity

        if prospective <= self.max_portfolio_leverage:
            return None

        return self._reject(
            RejectionReason.PORTFOLIO_LEVERAGE,
            f"Portfolio leverage {prospective:.2f}x > limit "
            f"{self.max_portfolio_leverage:.2f}x "
            f"(adding {signal.symbol} {signal.leverage:.2f}x from "
            f"{signal.strategy_name})",
        )

    def check_correlation_cluster(
        self,
        signal,
        portfolio_state: PortfolioState,
    ) -> Optional[RiskDecision]:
        """
        Block additions to a correlated cluster whose combined exposure already
        exceeds ``max_corr_cluster`` (default 30%).

        A "cluster" is the set of all held symbols (across strategies) whose
        60-day rolling correlation with ``signal.symbol`` exceeds
        ``corr_cluster_threshold`` (default 0.70), plus the symbol itself.
        """
        ph = portfolio_state.price_history
        if not ph or signal.symbol not in ph:
            return None

        equity = max(portfolio_state.equity, 1e-9)
        agg    = self.get_aggregate_positions()

        new_series = ph[signal.symbol].dropna()
        if len(new_series) < self.correlation_window:
            return None

        cluster_exposure = abs(agg.get(signal.symbol, 0.0))   # symbol itself

        for held_sym, held_val in agg.items():
            if held_sym == signal.symbol:
                continue
            if held_sym not in ph:
                continue
            other = ph[held_sym].dropna()
            aligned = pd.concat(
                [new_series.rename("a"), other.rename("b")], axis=1
            ).dropna().tail(self.correlation_window)
            if len(aligned) < 20:
                continue
            corr = float(aligned["a"].corr(aligned["b"]))
            if not np.isnan(corr) and corr >= self.corr_cluster_threshold:
                cluster_exposure += abs(held_val)

        cluster_pct = cluster_exposure / equity
        proposed    = signal.position_size_pct

        if cluster_pct + proposed > self.max_corr_cluster:
            return self._reject(
                RejectionReason.PORTFOLIO_CORR_CLUSTER,
                f"{signal.symbol} correlated-cluster exposure "
                f"{cluster_pct:.1%} + {proposed:.1%} "
                f"> cap {self.max_corr_cluster:.0%}",
            )

        return None

    # ── Private helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _reject(reason: RejectionReason, detail: str) -> RiskDecision:
        return RiskDecision(
            approved         = False,
            modified_signal  = None,
            rejection_reason = f"{reason.value}: {detail}",
        )

    # ── Factory ────────────────────────────────────────────────────────────────

    @classmethod
    def from_config(
        cls,
        config: dict,
        initial_equity: float = 100_000.0,
    ) -> "PortfolioRiskManager":
        """Build from the ``risk`` section of settings.yaml."""
        r = config.get("risk", {})
        return cls(
            max_aggregate_exposure = float(r.get("max_exposure",        0.80)),
            max_single_symbol      = float(r.get("max_single_position", 0.15)),
            max_portfolio_leverage = float(r.get("max_leverage",        1.25)),
            max_corr_cluster       = float(r.get("max_correlated_exposure", 0.30)),
            daily_dd_reduce        = float(r.get("daily_dd_reduce",     0.02)),
            daily_dd_halt          = float(r.get("daily_dd_halt",       0.03)),
            max_dd_from_peak       = float(r.get("max_dd_from_peak",    0.10)),
        )
