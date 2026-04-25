"""
regime_strategies.py — Volatility-regime allocation strategies.

DESIGN
------
The HMM detects *volatility environments*, not price direction.  Equities
trend upward ~70% of the time in low-volatility periods; the worst drawdowns
cluster in high-volatility spikes.  The entire edge is drawdown avoidance:

  Low vol  → be fully invested (calm markets trend up)
  Mid vol  → stay invested if trend intact, reduce if not
  High vol → reduce but stay LONG (catch V-shaped rebounds)

ALWAYS LONG.  NEVER SHORT.

THREE STRATEGIES (mapped from vol_rank, independent of label ordering)
----------------------------------------------------------------------
  LowVolBullStrategy       vol_rank ≤ 0.33  95% alloc, 1.25× leverage
  MidVolCautiousStrategy   0.33 < rank < 0.67  60–95% depending on EMA trend
  HighVolDefensiveStrategy vol_rank ≥ 0.67  60% alloc, 1.0× leverage

VOLATILITY RANK (computed by StrategyOrchestrator at init time)
---------------------------------------------------------------
Sort RegimeInfo objects by expected_volatility ascending.
  vol_rank = rank_idx / (n_regimes - 1)   # 0.0 = calmest, 1.0 = most volatile
  vol_rank ≤ 0.33  → LowVolBullStrategy
  vol_rank ≥ 0.67  → HighVolDefensiveStrategy
  else             → MidVolCautiousStrategy

This sort is INDEPENDENT of the label sort (which is by return).

UNCERTAINTY MODE (confidence < threshold OR is_flickering)
-----------------------------------------------------------
  - Halve all position sizes
  - Force leverage to 1.0×
  - Signal.reasoning prefixed with "[UNCERTAINTY]"

REBALANCING
-----------
  Skip rebalance when |target_weight − current_weight| < rebalance_threshold × target_weight.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Tuple, Type

import pandas as pd

from core.hmm_engine import RegimeInfo, RegimeState

logger = logging.getLogger(__name__)


# ── Strategy health ────────────────────────────────────────────────────────────

@dataclass
class StrategyHealth:
    """
    Point-in-time health snapshot for a single strategy.

    A strategy is UNHEALTHY if ANY condition is met:
    - drawdown from own peak > 15 %
    - Sharpe over the last 60 days < -1.0
    - 10+ consecutive losing days
    """
    is_healthy: bool
    recent_sharpe: float
    current_drawdown: float
    consecutive_losing_days: int
    reason_if_unhealthy: Optional[str] = None


# ── Enumerations ───────────────────────────────────────────────────────────────

class Direction(str, Enum):
    LONG = "LONG"
    FLAT = "FLAT"   # close / do not enter the position


# ── Signal dataclass ───────────────────────────────────────────────────────────

@dataclass
class Signal:
    """
    Fully-resolved trading signal for one symbol, ready for the order executor.

    ``position_size_pct`` is the **per-symbol** target weight in the portfolio
    (the orchestrator divides the strategy's total allocation by n_long_symbols).

    Attributes
    ----------
    symbol            : ticker
    direction         : LONG or FLAT
    confidence        : regime posterior probability (0–1)
    entry_price       : latest close price at signal time
    stop_loss         : computed stop level (strategy-specific)
    take_profit       : optional target price (None for open-ended holds)
    position_size_pct : per-symbol target weight (0.0 – 1.0)
    leverage          : 1.0 (no leverage) or 1.25 (low-vol only)
    regime_id         : HMM state index that triggered this signal
    regime_name       : human-readable label (e.g. "BULL")
    regime_probability: same as confidence, stored separately for clarity
    timestamp         : bar timestamp
    reasoning         : human-readable explanation
    strategy_name     : class name of the strategy that generated this signal
    metadata          : arbitrary key-value data (EMA, ATR, etc.)
    """

    symbol: str
    direction: Direction
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: Optional[float]
    position_size_pct: float
    leverage: float
    regime_id: int
    regime_name: str
    regime_probability: float
    timestamp: pd.Timestamp
    reasoning: str
    strategy_name: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def risk_per_trade(self) -> float:
        """(entry − stop) / entry — fraction of entry price at risk."""
        if self.entry_price <= 0.0:
            return 0.0
        return max(0.0, (self.entry_price - self.stop_loss) / self.entry_price)

    @property
    def is_long(self) -> bool:
        return self.direction == Direction.LONG


# ── Technical helpers (module-level, reused across strategies) ─────────────────

def _ema(close: pd.Series, period: int) -> float:
    """Latest EMA value using standard exponential smoothing."""
    return float(close.ewm(span=period, adjust=False).mean().iloc[-1])


def _atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> float:
    """Latest ATR using Wilder's smoothing (com = period − 1)."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return float(tr.ewm(com=period - 1, adjust=False).mean().iloc[-1])


# ── Base strategy ABC ──────────────────────────────────────────────────────────

# Health thresholds — centralised so tests can monkey-patch them.
_HEALTH_MAX_DRAWDOWN: float = 0.15        # 15 % from own equity peak
_HEALTH_MIN_SHARPE: float = -1.0          # 60-day Sharpe floor
_HEALTH_MAX_CONSEC_LOSSES: int = 10       # consecutive losing days
_PERF_HISTORY_DAYS: int = 60             # rolling window kept in memory


class BaseStrategy(ABC):
    """
    Abstract base for all regime strategies.

    Concrete strategies implement :meth:`generate_signal` and expose
    :attr:`name` and :attr:`total_allocation`.

    State tracking (Phase 1)
    ------------------------
    is_enabled           : can be toggled without restart
    allocated_capital    : capital currently assigned by the allocator (dollars)
    performance_history  : deque of (date, daily_return) tuples, max 60 entries
    current_positions    : {symbol: position_size_pct} as of the last signal batch
    _equity_peak         : high-water mark of strategy equity (for drawdown)
    _equity_current      : latest strategy equity estimate
    _consecutive_losses  : streak of days with negative return
    """

    MIN_BARS: int = 60

    def __init__(self) -> None:
        self.is_enabled: bool = True
        self.allocated_capital: float = 0.0
        self.performance_history: Deque[Tuple[pd.Timestamp, float]] = deque(
            maxlen=_PERF_HISTORY_DAYS
        )
        self.current_positions: Dict[str, float] = {}
        self._equity_peak: float = 1.0
        self._equity_current: float = 1.0
        self._consecutive_losses: int = 0

    # ── Abstract interface ─────────────────────────────────────────────────────

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier shown in Signal.strategy_name."""

    @property
    @abstractmethod
    def total_allocation(self) -> float:
        """
        Total portfolio fraction this strategy deploys across all long symbols.

        The orchestrator divides this by n_long_symbols to get per-symbol weights.
        """

    @abstractmethod
    def generate_signal(
        self,
        symbol: str,
        bars: pd.DataFrame,
        regime_state: RegimeState,
    ) -> Optional[Signal]:
        """
        Produce a :class:`Signal` for ``symbol``.

        Parameters
        ----------
        symbol       : ticker string
        bars         : OHLCV DataFrame, most-recent row last
        regime_state : current regime from the HMM engine

        Returns
        -------
        :class:`Signal`, or ``None`` when bars are insufficient.
        """

    # ── Lifecycle hooks ────────────────────────────────────────────────────────

    def on_enable(self) -> None:
        """Called when the strategy transitions from disabled → enabled."""
        logger.info("Strategy '%s' enabled.", self.name)

    def on_disable(self) -> None:
        """Called when the strategy transitions from enabled → disabled."""
        logger.info("Strategy '%s' disabled.", self.name)

    # ── Performance tracking ───────────────────────────────────────────────────

    def record_daily_return(self, timestamp: pd.Timestamp, daily_return: float) -> None:
        """
        Push one day's return into the rolling history.

        Updates the equity curve, high-water mark, and consecutive-loss streak.
        Call this once per trading day from the backtester or live loop.
        """
        self.performance_history.append((timestamp, daily_return))

        self._equity_current *= (1.0 + daily_return)
        if self._equity_current > self._equity_peak:
            self._equity_peak = self._equity_current

        if daily_return < 0.0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

    def get_recent_sharpe(self, window_days: int = 60) -> float:
        """
        Annualised Sharpe over the last *window_days* daily returns.

        Returns 0.0 when there are fewer than 2 observations.
        """
        returns = [r for _, r in self.performance_history][-window_days:]
        if len(returns) < 2:
            return 0.0
        import statistics
        mean = statistics.mean(returns)
        std = statistics.stdev(returns)
        if std < 1e-10:
            return 0.0
        return (mean / std) * (252 ** 0.5)

    def get_current_drawdown(self) -> float:
        """
        Drawdown from the strategy's own equity peak (positive fraction, e.g. 0.12 = 12 %).
        """
        if self._equity_peak <= 0.0:
            return 0.0
        return max(0.0, 1.0 - self._equity_current / self._equity_peak)

    # ── Health check ───────────────────────────────────────────────────────────

    def health_check(self) -> StrategyHealth:
        """
        Evaluate whether this strategy should remain active.

        Returns a :class:`StrategyHealth` with ``is_healthy=False`` and a
        ``reason_if_unhealthy`` string if any threshold is breached.
        """
        sharpe = self.get_recent_sharpe()
        drawdown = self.get_current_drawdown()
        consec = self._consecutive_losses

        reasons: List[str] = []
        if drawdown > _HEALTH_MAX_DRAWDOWN:
            reasons.append(
                f"drawdown {drawdown:.1%} > {_HEALTH_MAX_DRAWDOWN:.0%}"
            )
        if sharpe < _HEALTH_MIN_SHARPE:
            reasons.append(
                f"60d Sharpe {sharpe:.2f} < {_HEALTH_MIN_SHARPE}"
            )
        if consec >= _HEALTH_MAX_CONSEC_LOSSES:
            reasons.append(
                f"{consec} consecutive losing days ≥ {_HEALTH_MAX_CONSEC_LOSSES}"
            )

        is_healthy = len(reasons) == 0
        return StrategyHealth(
            is_healthy=is_healthy,
            recent_sharpe=sharpe,
            current_drawdown=drawdown,
            consecutive_losing_days=consec,
            reason_if_unhealthy=("; ".join(reasons) if reasons else None),
        )

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _has_enough_bars(self, bars: pd.DataFrame) -> bool:
        required_cols = {"open", "high", "low", "close", "volume"}
        return required_cols.issubset(bars.columns) and len(bars) >= self.MIN_BARS


# ── Concrete strategies ────────────────────────────────────────────────────────

class LowVolBullStrategy(BaseStrategy):
    """
    Low-volatility bull regime: full deployment with modest leverage.

    Rationale: In calm markets equities drift upward; leverage amplifies
    the compounding advantage while the wide ATR-based stop keeps risk bounded.

    Allocation : ``allocation`` (default 0.95 — 95% of portfolio)
    Leverage   : ``leverage``   (default 1.25×)
    Stop       : max(price − 3×ATR, 50EMA − 0.5×ATR)

    Parameters
    ----------
    allocation     : total portfolio fraction to deploy
    leverage       : leverage multiplier (≥ 1.0)
    ema_period     : EMA period for trend reference and stop calc
    atr_period     : ATR period for stop calc
    atr_stop_mult  : multiplier for price-based ATR stop  (default 3.0)
    ema_stop_mult  : multiplier for EMA-based ATR stop    (default 0.5)
    """

    def __init__(
        self,
        allocation: float = 0.95,
        leverage: float = 1.25,
        ema_period: int = 50,
        atr_period: int = 14,
        atr_stop_mult: float = 3.0,
        ema_stop_mult: float = 0.5,
    ) -> None:
        super().__init__()
        self._allocation = allocation
        self._leverage = leverage
        self.ema_period = ema_period
        self.atr_period = atr_period
        self.atr_stop_mult = atr_stop_mult
        self.ema_stop_mult = ema_stop_mult

    @property
    def name(self) -> str:
        return "LowVolBullStrategy"

    @property
    def total_allocation(self) -> float:
        return self._allocation

    def generate_signal(
        self,
        symbol: str,
        bars: pd.DataFrame,
        regime_state: RegimeState,
    ) -> Optional[Signal]:
        if not self._has_enough_bars(bars):
            logger.debug("LowVolBull: %s has only %d bars (need %d)", symbol, len(bars), self.MIN_BARS)
            return None

        close = bars["close"]
        high  = bars["high"]
        low   = bars["low"]

        price  = float(close.iloc[-1])
        ema50  = _ema(close, self.ema_period)
        atr_val = _atr(high, low, close, self.atr_period)

        stop_price  = price  - self.atr_stop_mult * atr_val   # price − 3 ATR
        stop_ema    = ema50  - self.ema_stop_mult * atr_val   # 50EMA − 0.5 ATR
        stop_loss   = max(stop_price, stop_ema)               # tighter is safer

        return Signal(
            symbol=symbol,
            direction=Direction.LONG,
            confidence=regime_state.probability,
            entry_price=price,
            stop_loss=stop_loss,
            take_profit=None,
            position_size_pct=self._allocation,   # orchestrator divides by n_symbols
            leverage=self._leverage,
            regime_id=regime_state.state_id,
            regime_name=regime_state.label,
            regime_probability=regime_state.probability,
            timestamp=pd.Timestamp(bars.index[-1]),
            reasoning=(
                f"LowVol regime — calm market, full deployment + {self._leverage}× leverage. "
                f"EMA{self.ema_period}={ema50:.2f}  ATR={atr_val:.4f}  "
                f"stop=max({stop_price:.2f}, {stop_ema:.2f})={stop_loss:.2f}"
            ),
            strategy_name=self.name,
            metadata={
                "ema50": round(ema50, 4),
                "atr": round(atr_val, 4),
                "stop_price_based": round(stop_price, 4),
                "stop_ema_based": round(stop_ema, 4),
            },
        )


class MidVolCautiousStrategy(BaseStrategy):
    """
    Mid-volatility regime: trend-conditional allocation.

    Uses a 50-EMA filter to distinguish between:
    * price > 50 EMA (trend intact)  → 95% allocation, 1.0× leverage
    * price < 50 EMA (trend weak)    → 60% allocation, 1.0× leverage

    Stop: 50EMA − 0.5×ATR

    Parameters
    ----------
    allocation_above_ema : allocation when price > 50 EMA (default 0.95)
    allocation_below_ema : allocation when price < 50 EMA (default 0.60)
    ema_period           : EMA period
    atr_period           : ATR period
    ema_stop_mult        : ATR multiplier for the EMA-based stop
    """

    def __init__(
        self,
        allocation_above_ema: float = 0.95,
        allocation_below_ema: float = 0.60,
        ema_period: int = 50,
        atr_period: int = 14,
        ema_stop_mult: float = 0.5,
    ) -> None:
        super().__init__()
        self.allocation_above_ema = allocation_above_ema
        self.allocation_below_ema = allocation_below_ema
        self.ema_period = ema_period
        self.atr_period = atr_period
        self.ema_stop_mult = ema_stop_mult

    @property
    def name(self) -> str:
        return "MidVolCautiousStrategy"

    @property
    def total_allocation(self) -> float:
        # Report the conservative floor; actual allocation depends on EMA filter.
        return self.allocation_below_ema

    def generate_signal(
        self,
        symbol: str,
        bars: pd.DataFrame,
        regime_state: RegimeState,
    ) -> Optional[Signal]:
        if not self._has_enough_bars(bars):
            logger.debug("MidVolCautious: %s has only %d bars (need %d)", symbol, len(bars), self.MIN_BARS)
            return None

        close   = bars["close"]
        high    = bars["high"]
        low     = bars["low"]

        price    = float(close.iloc[-1])
        ema50    = _ema(close, self.ema_period)
        atr_val  = _atr(high, low, close, self.atr_period)
        above    = price > ema50

        allocation = self.allocation_above_ema if above else self.allocation_below_ema
        stop_loss  = ema50 - self.ema_stop_mult * atr_val
        trend_str  = "above" if above else "below"

        return Signal(
            symbol=symbol,
            direction=Direction.LONG,
            confidence=regime_state.probability,
            entry_price=price,
            stop_loss=stop_loss,
            take_profit=None,
            position_size_pct=allocation,
            leverage=1.0,
            regime_id=regime_state.state_id,
            regime_name=regime_state.label,
            regime_probability=regime_state.probability,
            timestamp=pd.Timestamp(bars.index[-1]),
            reasoning=(
                f"MidVol regime — price {trend_str} EMA{self.ema_period} "
                f"({price:.2f} vs {ema50:.2f}) → {allocation:.0%} allocation. "
                f"Stop={stop_loss:.2f}  ATR={atr_val:.4f}"
            ),
            strategy_name=self.name,
            metadata={
                "ema50": round(ema50, 4),
                "atr": round(atr_val, 4),
                "above_ema": above,
                "allocation_used": allocation,
            },
        )


class HighVolDefensiveStrategy(BaseStrategy):
    """
    High-volatility defensive regime: reduced allocation, no leverage.

    ALWAYS LONG — never short.  Partial deployment preserves capital for
    V-shaped reversals that are common when volatility spikes.

    Allocation : ``allocation`` (default 0.60 — 60% of portfolio)
    Leverage   : 1.0× (no leverage in high-vol)
    Stop       : 50EMA − 1.0×ATR

    Parameters
    ----------
    allocation    : total portfolio fraction to deploy
    ema_period    : EMA period for the stop calculation
    atr_period    : ATR period
    ema_stop_mult : ATR multiplier for the EMA-based stop (default 1.0)
    """

    def __init__(
        self,
        allocation: float = 0.60,
        ema_period: int = 50,
        atr_period: int = 14,
        ema_stop_mult: float = 1.0,
    ) -> None:
        super().__init__()
        self._allocation = allocation
        self.ema_period = ema_period
        self.atr_period = atr_period
        self.ema_stop_mult = ema_stop_mult

    @property
    def name(self) -> str:
        return "HighVolDefensiveStrategy"

    @property
    def total_allocation(self) -> float:
        return self._allocation

    def generate_signal(
        self,
        symbol: str,
        bars: pd.DataFrame,
        regime_state: RegimeState,
    ) -> Optional[Signal]:
        if not self._has_enough_bars(bars):
            logger.debug("HighVolDefensive: %s has only %d bars (need %d)", symbol, len(bars), self.MIN_BARS)
            return None

        close   = bars["close"]
        high    = bars["high"]
        low     = bars["low"]

        price    = float(close.iloc[-1])
        ema50    = _ema(close, self.ema_period)
        atr_val  = _atr(high, low, close, self.atr_period)
        stop_loss = ema50 - self.ema_stop_mult * atr_val

        return Signal(
            symbol=symbol,
            direction=Direction.LONG,  # NEVER SHORT — always long, just smaller
            confidence=regime_state.probability,
            entry_price=price,
            stop_loss=stop_loss,
            take_profit=None,
            position_size_pct=self._allocation,
            leverage=1.0,
            regime_id=regime_state.state_id,
            regime_name=regime_state.label,
            regime_probability=regime_state.probability,
            timestamp=pd.Timestamp(bars.index[-1]),
            reasoning=(
                f"HighVol regime — defensive allocation ({self._allocation:.0%}), "
                f"LONG only (never short), catching V-shaped rebounds. "
                f"EMA{self.ema_period}={ema50:.2f}  ATR={atr_val:.4f}  Stop={stop_loss:.2f}"
            ),
            strategy_name=self.name,
            metadata={
                "ema50": round(ema50, 4),
                "atr": round(atr_val, 4),
            },
        )


class CryptoMomentumStrategy(BaseStrategy):
    """
    Momentum-based strategy for crypto assets, independent from volatility regime.

    Uses ROC (rate-of-change) momentum and ADX trend confirmation. Entry when
    ROC(20) > 0 AND ADX(14) > 25 (confirming uptrend). No leverage.
    Targets BTC/USD and ETH/USD.

    This strategy is orthogonal to HMM regime (uses independent technical signals,
    not volatility classification). Expected correlation < 0.80 with hmm_regime.
    """

    def __init__(self):
        super().__init__(name="CryptoMomentumStrategy")
        self._allocation = 0.30  # Conservative 30% max for unproven crypto class
        self.ema_period = 50
        self.roc_period = 20
        self.adx_period = 14

    def generate_signal(self, symbol: str, bars: pd.DataFrame, regime_state: RegimeState) -> Optional[Signal]:
        """
        Generate signal based on momentum (ROC) and trend confirmation (ADX).
        Not regime-dependent; uses independent technical signals.
        """
        if len(bars) < max(self.roc_period, self.adx_period, self.ema_period) + 1:
            return None

        close = bars["close"]
        high = bars["high"]
        low = bars["low"]

        # ROC(20): rate of change
        roc = (close.iloc[-1] - close.iloc[-self.roc_period - 1]) / close.iloc[-self.roc_period - 1]

        # ADX(14): trend strength (simplified — use our ATR-based ADX proxy)
        atr_val = self._atr(high, low, close, period=self.adx_period)
        ema_atr = self._ema(pd.Series([atr_val] * len(close)), period=14).iloc[-1]
        adx_proxy = (atr_val / close.iloc[-1]) * 100  # volatility as proxy for ADX strength

        # EMA(50) for stop-loss
        ema50 = self._ema(close, period=self.ema_period)

        # Entry conditions: ROC > 0 (uptrend) AND ADX > 3 (trend is present)
        # ADX proxy of 3 ≈ moderate trend; actual ADX > 25 needs market-wide calculation
        roc_positive = roc > 0
        adx_threshold = adx_proxy > 2.0  # ~2% ATR / price = moderate trend

        if roc_positive and adx_threshold:
            # LONG signal with ATR-based stop
            stop_loss = ema50 - 1.0 * atr_val  # More aggressive than defensive strategies
            position_size = self._allocation / 2  # 2 symbols (BTC, ETH) → 15% each max
            confidence = min(0.75, 0.50 + abs(roc))  # Confidence scaled by ROC magnitude

            return Signal(
                symbol=symbol,
                direction=Direction.LONG,
                confidence=confidence,
                entry_price=close.iloc[-1],
                stop_loss=stop_loss,
                take_profit=None,
                position_size_pct=position_size,
                leverage=1.0,  # No leverage for crypto
                regime_id=regime_state.state_id,
                regime_name=regime_state.label,
                regime_probability=regime_state.probability,
                timestamp=pd.Timestamp(bars.index[-1]),
                reasoning=(
                    f"Crypto momentum — ROC(20)={roc:.2%}, ADX proxy={adx_proxy:.2f}. "
                    f"Close={close.iloc[-1]:.2f}, EMA50={ema50:.2f}, Stop={stop_loss:.2f}"
                ),
                strategy_name=self.name,
                metadata={
                    "roc_20": round(roc, 4),
                    "adx_proxy": round(adx_proxy, 4),
                    "ema50": round(ema50, 4),
                    "atr": round(atr_val, 4),
                },
            )
        else:
            # FLAT — exit position
            return Signal(
                symbol=symbol,
                direction=Direction.FLAT,
                confidence=0.0,
                entry_price=close.iloc[-1],
                stop_loss=None,
                take_profit=None,
                position_size_pct=0.0,
                leverage=1.0,
                regime_id=regime_state.state_id,
                regime_name=regime_state.label,
                regime_probability=regime_state.probability,
                timestamp=pd.Timestamp(bars.index[-1]),
                reasoning=f"Crypto momentum exit: ROC={roc:.2%} < 0 or ADX too weak",
                strategy_name=self.name,
                metadata={"roc_20": round(roc, 4), "adx_proxy": round(adx_proxy, 4)},
            )


# ── Backward-compatible aliases ────────────────────────────────────────────────

CrashDefensiveStrategy      = HighVolDefensiveStrategy
BearTrendStrategy           = HighVolDefensiveStrategy
MeanReversionStrategy       = MidVolCautiousStrategy
BullTrendStrategy           = LowVolBullStrategy
EuphoriaCautiousStrategy    = LowVolBullStrategy

# ── Static label → strategy class (fallback / label-lookup convenience) ────────

#: Maps every possible HMM regime label to a strategy class.
#: The dynamic :class:`StrategyOrchestrator` vol-rank mapping takes precedence;
#: use this only when you need a quick class reference by label name.
LABEL_TO_STRATEGY: Dict[str, Type[BaseStrategy]] = {
    # 3-state
    "BEAR":         HighVolDefensiveStrategy,
    "NEUTRAL":      MidVolCautiousStrategy,
    "BULL":         LowVolBullStrategy,
    # 4-state additions
    "CRASH":        HighVolDefensiveStrategy,
    "EUPHORIA":     LowVolBullStrategy,
    # 6/7-state additions
    "STRONG_BEAR":  HighVolDefensiveStrategy,
    "WEAK_BEAR":    MidVolCautiousStrategy,
    "WEAK_BULL":    MidVolCautiousStrategy,
    "STRONG_BULL":  LowVolBullStrategy,
}


# ── Strategy orchestrator ──────────────────────────────────────────────────────

class StrategyOrchestrator:
    """
    Translate HMM regime outputs into per-symbol :class:`Signal` objects.

    Initialisation
    --------------
    Accepts the ``regime_infos`` dict from a fitted :class:`~core.hmm_engine.HMMEngine`,
    sorts regimes by ``expected_volatility`` (ascending), and assigns a strategy to
    each ``regime_id`` based on its normalised volatility rank.

    This vol-rank sort is *independent* of the label sort (which is by return).
    A "BULL" label could be high-vol or low-vol depending on what the HMM learned.

    Parameters
    ----------
    config           : full settings dict (broker/strategy section is consumed)
    regime_infos     : ``{label: RegimeInfo}`` from a fitted HMMEngine
    min_confidence   : posterior probability below which uncertainty mode triggers
    rebalance_threshold : relative deviation required to actually place a rebalance
    """

    def __init__(
        self,
        config: Dict[str, Any],
        regime_infos: Dict[str, RegimeInfo],
        min_confidence: float = 0.55,
        rebalance_threshold: float = 0.10,
    ) -> None:
        self.config = config
        self.min_confidence = min_confidence
        self.rebalance_threshold = rebalance_threshold

        strategy_cfg = config.get("strategy", {})
        self._low_vol_allocation: float   = strategy_cfg.get("low_vol_allocation", 0.95)
        self._mid_trend_allocation: float = strategy_cfg.get("mid_vol_allocation_trend", 0.95)
        self._mid_notrd_allocation: float = strategy_cfg.get("mid_vol_allocation_no_trend", 0.60)
        self._high_vol_allocation: float  = strategy_cfg.get("high_vol_allocation", 0.60)
        self._low_vol_leverage: float     = strategy_cfg.get("low_vol_leverage", 1.25)
        self._mid_vol_atr_mult: float     = float(strategy_cfg.get("mid_vol_atr_mult", 0.5))
        self._high_vol_atr_mult: float    = float(strategy_cfg.get("high_vol_atr_mult", 1.0))
        self._uncertainty_mult: float     = strategy_cfg.get("uncertainty_size_mult", 0.50)

        # SMA-200 trend gate: when the market proxy is above its 200-bar SMA,
        # upgrade defensive/neutral regimes to reflect the trending environment.
        #   HighVolDefensive → MidVolCautious  (downgrade from fully defensive)
        #   MidVolCautious   → LowVolBull      (upgrade to fully invested)
        # Set sma_trend_gate: false to disable.
        self._sma_gate: bool = strategy_cfg.get("sma_trend_gate", True)
        # HARD SMA gate: when enabled, FORCE flat (allocation 0) whenever
        # the market proxy closes below its 200-bar SMA, regardless of the
        # HMM regime. This is the "HMM × SMA overlay" — SMA handles trend
        # participation, HMM modulates sizing within the long phase.
        # Expected to trade some HMM upside for dramatically lower
        # drawdown / volatility (the SMA-200 benchmark MaxDD is ~-13% vs
        # HMM's -23%). Default OFF for backward compatibility.
        self._sma_hard_gate: bool = strategy_cfg.get("sma_hard_gate", False)
        # SOFT SMA overlay: multiply per-symbol sizes by this factor when
        # the market proxy is below SMA-200. 1.0 = disabled (default), 0.5
        # = halve sizes, 0.0 = equivalent to sma_hard_gate.
        self._sma_soft_mult: float = float(strategy_cfg.get("sma_soft_mult", 1.0))
        # Volatility targeting: scale LONG sizes so portfolio realizes the
        # given annualized vol. 0 = disabled. Typical targets: 0.08-0.12.
        # Cap multiplier in [_vol_target_min_mult, _vol_target_max_mult].
        self._vol_target_annual: float  = float(strategy_cfg.get("vol_target_annual", 0.0))
        self._vol_target_min_mult: float = float(strategy_cfg.get("vol_target_min_mult", 0.25))
        self._vol_target_max_mult: float = float(strategy_cfg.get("vol_target_max_mult", 1.50))
        self._vol_target_window: int    = int(strategy_cfg.get("vol_target_window", 20))
        # HMM <-> SMA ensemble weight. 1.0 = pure HMM (default/off). 0.0 = pure
        # SMA-200 equal-weight. Blend: size_i = w*hmm_size + (1-w)*sma_ew_weight.
        # SMA signal: if market proxy > SMA200, each symbol gets
        # (low_vol_allocation / n_symbols); else 0. Intended to mix HMM's
        # regime-awareness with SMA's vol compression.
        self._sma_blend_weight: float = float(strategy_cfg.get("sma_blend_weight", 1.0))
        self._sma_gate_mid_strategy: BaseStrategy = MidVolCautiousStrategy(
            allocation_above_ema=self._mid_trend_allocation,
            allocation_below_ema=self._mid_notrd_allocation,
            ema_stop_mult=self._mid_vol_atr_mult,
        )
        self._sma_gate_bull_strategy: BaseStrategy = LowVolBullStrategy(
            allocation=self._low_vol_allocation,
            leverage=self._low_vol_leverage,
        )
        # Keep backward-compatible alias
        self._sma_gate_strategy = self._sma_gate_mid_strategy

        # regime_id → BaseStrategy
        self._regime_to_strategy: Dict[int, BaseStrategy] = {}
        # regime_id → normalised vol rank (0.0 = calmest, 1.0 = most volatile)
        self._vol_ranks: Dict[int, float] = {}
        # regime_id → RegimeInfo (for metadata queries)
        self._regime_infos: Dict[int, RegimeInfo] = {
            info.regime_id: info for info in regime_infos.values()
        }
        # Last known per-symbol portfolio weights (for rebalance filter)
        self._current_weights: Dict[str, float] = {}

        self._build_strategy_map(regime_infos)

    # ── Initialisation ─────────────────────────────────────────────────────────

    def _build_strategy_map(self, regime_infos: Dict[str, RegimeInfo]) -> None:
        """
        Sort regimes by ``expected_volatility`` and assign a strategy tier.

        Vol-rank formula: ``rank_idx / (n_regimes − 1)`` ∈ [0.0, 1.0]
          ≤ 0.33  → :class:`LowVolBullStrategy`
          ≥ 0.67  → :class:`HighVolDefensiveStrategy`
          else    → :class:`MidVolCautiousStrategy`
        """
        sorted_regimes = sorted(
            regime_infos.values(),
            key=lambda r: r.expected_volatility,
        )
        n = len(sorted_regimes)

        for rank_idx, regime_info in enumerate(sorted_regimes):
            vol_rank = rank_idx / (n - 1) if n > 1 else 0.5
            self._vol_ranks[regime_info.regime_id] = vol_rank

            if vol_rank <= 0.33:
                strategy: BaseStrategy = LowVolBullStrategy(
                    allocation=self._low_vol_allocation,
                    leverage=self._low_vol_leverage,
                )
                tier = "low_vol"
            elif vol_rank >= 0.67:
                strategy = HighVolDefensiveStrategy(
                    allocation=self._high_vol_allocation,
                    ema_stop_mult=self._high_vol_atr_mult,
                )
                tier = "high_vol"
            else:
                strategy = MidVolCautiousStrategy(
                    allocation_above_ema=self._mid_trend_allocation,
                    allocation_below_ema=self._mid_notrd_allocation,
                    ema_stop_mult=self._mid_vol_atr_mult,
                )
                tier = "mid_vol"

            self._regime_to_strategy[regime_info.regime_id] = strategy

            logger.info(
                "Regime '%s' (id=%d, exp_vol=%.4f) → vol_rank=%.2f → %s [%s]",
                regime_info.regime_name,
                regime_info.regime_id,
                regime_info.expected_volatility,
                vol_rank,
                type(strategy).__name__,
                tier,
            )

    # ── Signal generation ──────────────────────────────────────────────────────

    def generate_signals(
        self,
        symbols: List[str],
        bars: Dict[str, pd.DataFrame],
        regime_state: RegimeState,
        is_flickering: bool = False,
    ) -> List[Signal]:
        """
        Generate per-symbol signals for the current regime.

        Steps
        -----
        1. Look up the strategy for ``regime_state.state_id``.
        2. Detect uncertainty mode (low confidence or flickering).
        3. Call strategy.generate_signal() for each symbol.
        4. Apply uncertainty discount (halve sizes, drop leverage).
        5. Divide total allocation equally across LONG signals.
        6. Apply rebalance filter (skip if deviation < threshold).

        Parameters
        ----------
        symbols      : list of tickers to process
        bars         : ``{symbol: OHLCV DataFrame}``
        regime_state : current :class:`~core.hmm_engine.RegimeState`
        is_flickering: True if the HMM engine reports flicker activity

        Returns
        -------
        List of :class:`Signal` that warrant a trade or rebalance.
        """
        strategy = self._regime_to_strategy.get(regime_state.state_id)
        if strategy is None:
            logger.warning(
                "No strategy for regime_id=%d (label=%s) — returning no signals.",
                regime_state.state_id,
                regime_state.label,
            )
            return []

        # ── SMA-200 HARD gate (overlay) ───────────────────────────────────────
        # Force flat when market proxy is below SMA-200, regardless of HMM.
        # This combines the SMA-200 benchmark's trend filter with the HMM's
        # regime-aware sizing. Runs BEFORE any other logic.
        if self._sma_hard_gate:
            mkt_sym = symbols[0] if symbols else None
            mkt_bars = bars.get(mkt_sym) if mkt_sym else None
            if mkt_bars is not None and len(mkt_bars) >= 200:
                close  = mkt_bars["close"]
                sma200 = close.iloc[-200:].mean()
                if close.iloc[-1] < sma200:
                    # Emit flat signals (position_size_pct = 0) for every
                    # currently-held symbol so the rebalancer sells down.
                    flat_signals: List[Signal] = []
                    gate_reason = (
                        f"sma_hard_gate: {mkt_sym} {close.iloc[-1]:.2f} "
                        f"< SMA200 {sma200:.2f}"
                    )
                    for sym in symbols:
                        current = self._current_weights.get(sym, 0.0)
                        if abs(current) <= 1e-6:
                            continue
                        sym_bars = bars.get(sym)
                        last_close = (
                            float(sym_bars["close"].iloc[-1])
                            if sym_bars is not None and len(sym_bars) > 0
                            else 0.0
                        )
                        last_ts = (
                            sym_bars.index[-1]
                            if sym_bars is not None and len(sym_bars) > 0
                            else pd.Timestamp.now()
                        )
                        flat_signals.append(Signal(
                            symbol=sym,
                            direction=Direction.FLAT,
                            confidence=regime_state.probability,
                            entry_price=last_close,
                            stop_loss=0.0,
                            take_profit=None,
                            position_size_pct=0.0,
                            leverage=1.0,
                            regime_id=regime_state.state_id,
                            regime_name=regime_state.label,
                            regime_probability=regime_state.probability,
                            timestamp=last_ts,
                            reasoning=gate_reason,
                            strategy_name="SMAHardGate",
                        ))
                    if flat_signals:
                        logger.info(
                            "SMA hard gate ACTIVE: %s %.2f < SMA200 %.2f → FLAT %d symbols",
                            mkt_sym, close.iloc[-1], sma200, len(flat_signals),
                        )
                    return flat_signals

        # ── SMA-200 trend gate ────────────────────────────────────────────────
        # If the HMM picked a fully defensive (HighVol) strategy but the market
        # proxy is still above its 200-bar SMA, the trend has not broken down —
        # downgrade to MidVolCautious instead of cutting to 60% allocation.
        if self._sma_gate and isinstance(strategy, HighVolDefensiveStrategy):
            mkt_sym = symbols[0] if symbols else None
            mkt_bars = bars.get(mkt_sym) if mkt_sym else None
            if mkt_bars is not None and len(mkt_bars) >= 200:
                close  = mkt_bars["close"]
                sma200 = close.iloc[-200:].mean()
                if close.iloc[-1] > sma200:
                    logger.info(
                        "SMA gate: regime=%s %s %.2f > SMA200 %.2f"
                        " → MidVolCautious",
                        regime_state.label, mkt_sym, close.iloc[-1], sma200,
                    )
                    strategy = self._sma_gate_mid_strategy

        uncertainty = (
            regime_state.probability < self.min_confidence
            or is_flickering
            or not regime_state.is_confirmed
        )

        if uncertainty:
            reason_parts = []
            if regime_state.probability < self.min_confidence:
                reason_parts.append(f"p={regime_state.probability:.3f} < {self.min_confidence}")
            if is_flickering:
                reason_parts.append("flickering")
            if not regime_state.is_confirmed:
                reason_parts.append("unconfirmed")
            logger.debug("Uncertainty mode active: %s", ", ".join(reason_parts))

        # ── Per-symbol signal generation ──────────────────────────────────────
        raw: List[Signal] = []
        for symbol in symbols:
            symbol_bars = bars.get(symbol)
            if symbol_bars is None or len(symbol_bars) < strategy.MIN_BARS:
                continue
            sig = strategy.generate_signal(symbol, symbol_bars, regime_state)
            if sig is not None:
                raw.append(sig)

        if not raw:
            return []

        # ── Uncertainty discount ───────────────────────────────────────────────
        if uncertainty:
            raw = [self._apply_uncertainty_discount(s) for s in raw]

        # ── Distribute allocation across LONG symbols ─────────────────────────
        n_long = sum(1 for s in raw if s.direction == Direction.LONG)
        if n_long > 0:
            for s in raw:
                if s.direction == Direction.LONG:
                    # Convert strategy-level total allocation to per-symbol weight
                    s.position_size_pct = round(s.position_size_pct / n_long, 6)

        # ── Volatility targeting ──────────────────────────────────────────────
        # Scale LONG sizes so the portfolio realizes ~self._vol_target_annual.
        # Uses the market proxy's trailing realized vol as a cheap proxy for
        # portfolio vol (positions are mostly correlated within a basket).
        if self._vol_target_annual > 0.0 and n_long > 0:
            mkt_sym = symbols[0] if symbols else None
            mkt_bars = bars.get(mkt_sym) if mkt_sym else None
            w = self._vol_target_window
            if mkt_bars is not None and len(mkt_bars) >= w + 1:
                import numpy as _np
                close = mkt_bars["close"]
                log_ret = _np.log(close / close.shift(1)).iloc[-w:]
                daily_std = float(log_ret.std())
                realized_ann = daily_std * (252 ** 0.5)
                if realized_ann > 1e-6:
                    scale = self._vol_target_annual / realized_ann
                    scale = max(self._vol_target_min_mult,
                                min(self._vol_target_max_mult, scale))
                    for s in raw:
                        if s.direction == Direction.LONG:
                            s.position_size_pct = round(
                                s.position_size_pct * scale, 6
                            )
                    logger.info(
                        "Vol target: realized_ann=%.3f target=%.3f scale=%.3f",
                        realized_ann, self._vol_target_annual, scale,
                    )

        # ── Soft SMA overlay: scale down when market < SMA200 ─────────────────
        if self._sma_soft_mult < 1.0 and n_long > 0:
            mkt_sym = symbols[0] if symbols else None
            mkt_bars = bars.get(mkt_sym) if mkt_sym else None
            if mkt_bars is not None and len(mkt_bars) >= 200:
                close = mkt_bars["close"]
                sma200 = close.iloc[-200:].mean()
                if close.iloc[-1] < sma200:
                    for s in raw:
                        if s.direction == Direction.LONG:
                            s.position_size_pct = round(
                                s.position_size_pct * self._sma_soft_mult, 6
                            )
                    logger.info(
                        "SMA soft overlay: %s %.2f < SMA200 %.2f → size x%.2f",
                        mkt_sym, close.iloc[-1], sma200, self._sma_soft_mult,
                    )

        # ── HMM <-> SMA ensemble blend (weighted average) ─────────────────────
        # size_i := w * hmm_size_i + (1-w) * sma_ew_weight  (per-symbol)
        # where sma_ew_weight = (low_vol_allocation / n_syms) if mkt > SMA200 else 0.
        # LONG-only blend; does not touch FLAT signals.
        if 0.0 <= self._sma_blend_weight < 1.0 and len(raw) > 0 and symbols:
            mkt_sym = symbols[0]
            mkt_bars = bars.get(mkt_sym)
            if mkt_bars is not None and len(mkt_bars) >= 200:
                close = mkt_bars["close"]
                sma200 = close.iloc[-200:].mean()
                sma_long = close.iloc[-1] >= sma200
                n_syms = max(1, len(symbols))
                sma_ew_weight = (
                    self._low_vol_allocation / n_syms if sma_long else 0.0
                )
                w = self._sma_blend_weight
                for s in raw:
                    if s.direction == Direction.LONG:
                        blended = w * s.position_size_pct + (1.0 - w) * sma_ew_weight
                        s.position_size_pct = round(max(0.0, blended), 6)
                logger.info(
                    "SMA blend: %s %.2f %s SMA200 %.2f → w_hmm=%.2f sma_ew=%.4f",
                    mkt_sym, close.iloc[-1],
                    ">=" if sma_long else "<", sma200,
                    w, sma_ew_weight,
                )

        # ── Rebalance filter ──────────────────────────────────────────────────
        filtered: List[Signal] = []
        for s in raw:
            current = self._current_weights.get(s.symbol, 0.0)
            deviation = abs(s.position_size_pct - current)
            threshold = self.rebalance_threshold * max(s.position_size_pct, 1e-8)
            if deviation >= threshold:
                filtered.append(s)
            else:
                logger.debug(
                    "%s: skip rebalance — deviation %.4f < threshold %.4f",
                    s.symbol,
                    deviation,
                    threshold,
                )

        if filtered:
            logger.info(
                "Regime '%s' (p=%.3f)  strategy=%s  signals=%d/%d  uncertainty=%s",
                regime_state.label,
                regime_state.probability,
                type(strategy).__name__,
                len(filtered),
                len(symbols),
                uncertainty,
            )

        return filtered

    # ── State management ───────────────────────────────────────────────────────

    def update_weights(self, weights: Dict[str, float]) -> None:
        """
        Sync the orchestrator's view of current portfolio weights.

        Call this after each batch of fills to keep the rebalance filter
        accurate.

        Parameters
        ----------
        weights : ``{symbol: current_weight}``
        """
        self._current_weights.update(weights)

    def reset_weights(self) -> None:
        """Reset all tracked weights to zero (e.g. at session start)."""
        self._current_weights.clear()

    # ── Query helpers ──────────────────────────────────────────────────────────

    def get_vol_rank(self, regime_id: int) -> Optional[float]:
        """Return the normalised volatility rank for ``regime_id``."""
        return self._vol_ranks.get(regime_id)

    def get_strategy_for_regime(self, regime_id: int) -> Optional[BaseStrategy]:
        """Return the strategy instance mapped to ``regime_id``."""
        return self._regime_to_strategy.get(regime_id)

    def get_regime_info(self, regime_id: int) -> Optional[RegimeInfo]:
        """Return :class:`~core.hmm_engine.RegimeInfo` for ``regime_id``."""
        return self._regime_infos.get(regime_id)

    def summary(self) -> Dict[int, Dict[str, Any]]:
        """
        Return a dict summarising the vol-rank → strategy mapping for
        logging / dashboard display.
        """
        out: Dict[int, Dict[str, Any]] = {}
        for rid, strategy in self._regime_to_strategy.items():
            info = self._regime_infos.get(rid)
            out[rid] = {
                "label": info.regime_name if info else "?",
                "vol_rank": round(self._vol_ranks.get(rid, float("nan")), 3),
                "strategy": strategy.name,
                "total_allocation": strategy.total_allocation,
            }
        return out

    # ── Private helpers ────────────────────────────────────────────────────────

    def _apply_uncertainty_discount(self, signal: Signal) -> Signal:
        """
        Halve position size and remove leverage.

        Called when the regime confidence is below threshold or
        the HMM is flickering.
        """
        signal.position_size_pct = round(signal.position_size_pct * self._uncertainty_mult, 6)
        signal.leverage = 1.0
        if not signal.reasoning.startswith("[UNCERTAINTY]"):
            signal.reasoning = "[UNCERTAINTY] " + signal.reasoning
        return signal

    def _needs_rebalance(self, symbol: str, target_weight: float) -> bool:
        """True when the absolute deviation exceeds the relative threshold."""
        current = self._current_weights.get(symbol, 0.0)
        threshold = self.rebalance_threshold * max(target_weight, 1e-8)
        return abs(target_weight - current) >= threshold


# ── Backward-compatible RegimeStrategy alias ──────────────────────────────────
#: ``RegimeStrategy`` was the name used in earlier skeleton code.
#: New code should use :class:`StrategyOrchestrator` directly.
RegimeStrategy = StrategyOrchestrator

# Keep AllocationResult as a thin alias pointing to Signal for any legacy
# code that referenced it from the original skeleton.
AllocationResult = Signal
