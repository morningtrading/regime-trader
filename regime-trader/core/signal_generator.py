"""
signal_generator.py -- Combines HMM regime output with the vol-strategy to
produce actionable portfolio signals.

Acts as the top-level orchestrator of the core logic pipeline:
    MarketData -> FeatureEngineer -> HMMEngine -> StrategyOrchestrator -> Signals
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from core.hmm_engine import HMMEngine, RegimeInfo, RegimeState
from core.regime_strategies import RegimeStrategy, AllocationResult
from core.risk_manager import RiskManager

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Data structures                                                              #
# --------------------------------------------------------------------------- #

@dataclass
class PortfolioSignal:
    """
    Fully-resolved portfolio signal ready for the order executor.

    Contains both the desired target weights and the delta from current
    weights so the executor can determine which orders to place.
    """
    timestamp:       pd.Timestamp
    regime:          str                      # e.g. "BULL", "CRASH", "NEUTRAL"
    confidence:      float
    is_stable:       bool
    target_weights:  Dict[str, float]         # symbol -> desired weight
    delta_weights:   Dict[str, float]         # symbol -> weight change vs current
    leverage:        float
    trading_allowed: bool                     # False if risk manager says halt
    notes:           List[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# SignalGenerator                                                              #
# --------------------------------------------------------------------------- #

class SignalGenerator:
    """
    Combines HMM regime detection, vol-strategy allocation, and risk checks
    into a single PortfolioSignal per bar.

    This class is the single entry point called by main.py on each bar.

    Parameters
    ----------
    hmm_engine :
        Fitted (or to-be-fitted) HMMEngine.
    strategy :
        StrategyOrchestrator instance (aliased as RegimeStrategy).
    risk_manager :
        RiskManager instance.
    symbols :
        Trading universe; must match the strategy's symbol list.
    """

    def __init__(
        self,
        hmm_engine:   HMMEngine,
        strategy:     RegimeStrategy,
        risk_manager: RiskManager,
        symbols:      List[str],
    ) -> None:
        self.hmm_engine:   HMMEngine   = hmm_engine
        self.strategy:     RegimeStrategy = strategy
        self.risk_manager: RiskManager = risk_manager
        self.symbols:      List[str]   = symbols

        self._last_signal:       Optional[PortfolioSignal] = None
        self._current_weights:   Dict[str, float]          = {s: 0.0 for s in symbols}
        self._last_regime_state: Optional[RegimeState]     = None

    # ======================================================================= #
    # Public API                                                               #
    # ======================================================================= #

    def generate(
        self,
        features:  np.ndarray,
        prices:    pd.DataFrame,
        timestamp: pd.Timestamp,
    ) -> PortfolioSignal:
        """
        Produce a PortfolioSignal for the current bar.

        Parameters
        ----------
        features :
            2-D feature matrix of shape (n_history_bars, n_features).
            The last row is the current bar; all preceding rows provide
            history for the HMM forward pass.
        prices :
            DataFrame of shape (n_bars, n_symbols) with close prices.
        timestamp :
            Timestamp of the current bar.

        Returns
        -------
        PortfolioSignal
        """
        # ------------------------------------------------------------------ #
        # 1. Run HMM incremental update on the latest feature row            #
        # ------------------------------------------------------------------ #
        current_feature = features[-1]
        regime_state: RegimeState = self.hmm_engine.update(
            new_feature_row = current_feature,
            timestamp       = timestamp,
        )
        self._last_regime_state = regime_state

        # ------------------------------------------------------------------ #
        # 2. Check risk manager trading state (absolute gate)                #
        # ------------------------------------------------------------------ #
        from core.risk_manager import TradingState
        trading_state   = self.risk_manager.get_trading_state()
        trading_allowed = trading_state != TradingState.HALTED

        # ------------------------------------------------------------------ #
        # 3. Build per-symbol OHLCV bars for the strategy                    #
        # ------------------------------------------------------------------ #
        bars_by_symbol: Dict[str, pd.DataFrame] = {}
        for sym in self.symbols:
            if sym in prices.columns:
                close = prices[sym].dropna()
                # Synthesise minimal OHLCV the strategy needs (close-only)
                bars_by_symbol[sym] = pd.DataFrame({
                    "open":   close.shift(1).bfill(),
                    "high":   close,
                    "low":    close,
                    "close":  close,
                    "volume": pd.Series(1_000_000.0, index=close.index),
                })

        # ------------------------------------------------------------------ #
        # 4. Generate per-symbol signals from the strategy orchestrator      #
        # ------------------------------------------------------------------ #
        is_flickering = getattr(regime_state, "is_flickering", False)

        raw_signals = self.strategy.generate_signals(
            symbols       = self.symbols,
            bars          = bars_by_symbol,
            regime_state  = regime_state,
            is_flickering = is_flickering,
        )

        # Build target weights from signals
        target_weights: Dict[str, float] = {s: 0.0 for s in self.symbols}
        leverage = 1.0
        for sig in raw_signals:
            if sig.symbol in target_weights and sig.is_long:
                target_weights[sig.symbol] = sig.position_size_pct
                leverage = max(leverage, sig.leverage)

        # If REDUCED state, halve all weights
        if trading_state == TradingState.REDUCED:
            target_weights = {s: w * 0.5 for s, w in target_weights.items()}
            leverage = 1.0

        # If HALTED, flatten everything
        if not trading_allowed:
            target_weights = {s: 0.0 for s in self.symbols}
            leverage = 1.0

        # ------------------------------------------------------------------ #
        # 5. Compute weight deltas vs current holdings                       #
        # ------------------------------------------------------------------ #
        delta_weights = self._compute_delta_weights(target_weights)

        # ------------------------------------------------------------------ #
        # 6. Build notes for logging / dashboard                             #
        # ------------------------------------------------------------------ #
        notes = self._build_notes(regime_state, raw_signals)
        if trading_state == TradingState.REDUCED:
            notes.append(f"Circuit breaker REDUCED: all weights halved")
        if not trading_allowed:
            notes.append(f"Trading HALTED by risk manager: {trading_state.name}")

        signal = PortfolioSignal(
            timestamp       = timestamp,
            regime          = regime_state.label,
            confidence      = regime_state.probability,
            is_stable       = regime_state.is_confirmed,
            target_weights  = target_weights,
            delta_weights   = delta_weights,
            leverage        = leverage,
            trading_allowed = trading_allowed,
            notes           = notes,
        )

        self._last_signal = signal

        logger.debug(
            "Signal | t=%s | regime=%s (p=%.2f, stable=%s) | "
            "weights=%s | leverage=%.2f | allowed=%s",
            timestamp, regime_state.label, regime_state.probability,
            regime_state.is_confirmed, target_weights, leverage, trading_allowed,
        )

        return signal

    def update_current_weights(self, weights: Dict[str, float]) -> None:
        """
        Sync the signal generator's view of current portfolio weights.

        Called by the position tracker after each fill.

        Parameters
        ----------
        weights :
            Map of symbol -> actual portfolio weight after last fill.
        """
        for sym in self.symbols:
            self._current_weights[sym] = weights.get(sym, 0.0)

    def reconcile_from_broker(
        self,
        broker_positions: Dict[str, float],
        total_equity:     float,
        drift_threshold:  float = 0.005,
    ) -> Dict[str, float]:
        """
        True-up internal weights against the broker's reported positions.

        Prevents the internal ``_current_weights`` from drifting away from
        reality when a fill is missed, an order is placed out-of-band, or
        an API call silently fails. Call periodically (e.g. at the top of
        each trading loop tick).

        Parameters
        ----------
        broker_positions :
            Map of symbol -> dollar market value as reported by the broker
            (e.g. ``sum(qty_i * last_price_i)`` per symbol, signed for shorts).
        total_equity :
            Total account equity from the broker. Used to derive weights.
        drift_threshold :
            Log a warning whenever the absolute weight drift on any symbol
            exceeds this value (default 0.5%).

        Returns
        -------
        Dict[str, float]
            The drift (broker_weight - internal_weight) per symbol before
            reconciliation. Useful for dashboards / alerts.
        """
        if total_equity <= 0:
            logger.warning(
                "reconcile_from_broker skipped: non-positive equity %.2f",
                total_equity,
            )
            return {}

        drift: Dict[str, float] = {}
        for sym in self.symbols:
            broker_val = float(broker_positions.get(sym, 0.0))
            broker_w   = broker_val / total_equity
            internal_w = self._current_weights.get(sym, 0.0)
            d          = broker_w - internal_w
            drift[sym] = d
            if abs(d) >= drift_threshold:
                logger.warning(
                    "Weight drift on %s: internal=%.4f broker=%.4f Δ=%+.4f "
                    "→ reconciling to broker truth",
                    sym, internal_w, broker_w, d,
                )
            self._current_weights[sym] = broker_w

        return drift

    def get_current_regime(self) -> Optional[str]:
        """Return the regime label from the most recently generated signal, or None."""
        return self._last_signal.regime if self._last_signal else None

    def get_last_signal(self) -> Optional[PortfolioSignal]:
        """Return the most recently generated PortfolioSignal."""
        return self._last_signal

    # ======================================================================= #
    # Private helpers                                                          #
    # ======================================================================= #

    def _compute_delta_weights(
        self,
        target_weights: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Compute target - current weight for every symbol in the universe.

        Returns
        -------
        Dict of symbol -> (target - current) weight.
        """
        return {
            sym: target_weights.get(sym, 0.0) - self._current_weights.get(sym, 0.0)
            for sym in self.symbols
        }

    def _build_notes(
        self,
        hmm_result: RegimeState,
        signals: list,
    ) -> List[str]:
        """
        Compile human-readable notes describing why the signal was generated
        as it was (useful for logging and the dashboard).
        """
        notes: List[str] = []

        notes.append(
            f"Regime: {hmm_result.label}  "
            f"p={hmm_result.probability:.3f}  "
            f"confirmed={hmm_result.is_confirmed}  "
            f"consecutive={hmm_result.consecutive_bars}"
        )

        if not hmm_result.is_confirmed:
            notes.append(
                f"Regime unconfirmed: waiting for stability window "
                f"({hmm_result.consecutive_bars} bars so far)"
            )

        if hmm_result.probability < self.hmm_engine.min_confidence:
            notes.append(
                f"Low confidence ({hmm_result.probability:.2f} < "
                f"{self.hmm_engine.min_confidence:.2f}): uncertainty mode active"
            )

        from core.risk_manager import TradingState
        dd = self.risk_manager.get_drawdown_state()
        if dd.daily_dd < -0.01:
            notes.append(f"Daily P&L: {dd.daily_dd:.2%}")
        if dd.dd_from_peak < -0.05:
            notes.append(f"Peak drawdown: {dd.dd_from_peak:.2%}")

        for sig in signals:
            if sig.is_long:
                notes.append(
                    f"{sig.symbol}: {sig.strategy_name}  "
                    f"size={sig.position_size_pct:.1%}  "
                    f"stop={sig.stop_loss:.2f}  "
                    f"lev={sig.leverage:.2f}x"
                )

        return notes
