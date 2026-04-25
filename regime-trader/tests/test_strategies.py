"""
test_strategies.py — Unit tests for StrategyOrchestrator allocation logic.

Tests cover allocation fractions, leverage, weight normalisation, uncertainty
discount, and rebalance threshold for all three vol-tier strategies.
"""

import numpy as np
import pandas as pd
import pytest

from core.regime_strategies import (
    AllocationResult,
    Direction,
    HighVolDefensiveStrategy,
    LABEL_TO_STRATEGY,
    LowVolBullStrategy,
    MidVolCautiousStrategy,
    RegimeStrategy,
    Signal,
    StrategyOrchestrator,
)
from core.hmm_engine import RegimeInfo, RegimeState


# ── Shared helpers ──────────────────────────────────────────────────────────────

SYMBOLS = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"]
N_SYMBOLS = len(SYMBOLS)


def _bars(n: int = 250, trend: float = 0.001, start: float = 100.0) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame with `n` rows."""
    dates = pd.bdate_range("2023-01-01", periods=n)
    close = np.cumprod(1.0 + np.full(n, trend)) * start
    high = close * 1.01
    low = close * 0.99
    return pd.DataFrame(
        {
            "open": close * 0.995,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.full(n, 1_000_000.0),
        },
        index=dates,
    )


def _three_regime_infos() -> dict:
    """
    Three RegimeInfo objects with well-separated expected_volatility values so
    that StrategyOrchestrator assigns exactly one strategy tier to each regime:

        BULL    exp_vol=0.01  →  vol_rank=0.0  →  LowVolBullStrategy
        NEUTRAL exp_vol=0.02  →  vol_rank=0.5  →  MidVolCautiousStrategy
        BEAR    exp_vol=0.04  →  vol_rank=1.0  →  HighVolDefensiveStrategy
    """
    return {
        "BULL": RegimeInfo(
            regime_id=0,
            regime_name="BULL",
            expected_return=0.001,
            expected_volatility=0.01,
            recommended_strategy_type="LowVol",
            max_leverage_allowed=1.25,
            max_position_size_pct=0.95,
            min_confidence_to_act=0.55,
        ),
        "NEUTRAL": RegimeInfo(
            regime_id=1,
            regime_name="NEUTRAL",
            expected_return=0.0,
            expected_volatility=0.02,
            recommended_strategy_type="MidVol",
            max_leverage_allowed=1.0,
            max_position_size_pct=0.80,
            min_confidence_to_act=0.55,
        ),
        "BEAR": RegimeInfo(
            regime_id=2,
            regime_name="BEAR",
            expected_return=-0.001,
            expected_volatility=0.04,
            recommended_strategy_type="HighVol",
            max_leverage_allowed=1.0,
            max_position_size_pct=0.60,
            min_confidence_to_act=0.60,
        ),
    }


def _standard_config(
    low_vol_alloc: float = 0.95,
    mid_trend: float = 0.95,
    mid_notrd: float = 0.60,
    high_vol_alloc: float = 0.60,
    low_vol_leverage: float = 1.25,
    uncertainty_mult: float = 0.50,
) -> dict:
    return {
        "strategy": {
            "low_vol_allocation": low_vol_alloc,
            "mid_vol_allocation_trend": mid_trend,
            "mid_vol_allocation_no_trend": mid_notrd,
            "high_vol_allocation": high_vol_alloc,
            "low_vol_leverage": low_vol_leverage,
            "uncertainty_size_mult": uncertainty_mult,
        }
    }


def _state(
    state_id: int,
    label: str,
    prob: float = 0.90,
    confirmed: bool = True,
) -> RegimeState:
    """Build a synthetic RegimeState for the 3-state test setup."""
    state_probs = np.zeros(3)
    state_probs[state_id] = prob
    for i in range(3):
        if i != state_id:
            state_probs[i] = (1.0 - prob) / 2.0
    return RegimeState(
        label=label,
        state_id=state_id,
        probability=prob,
        state_probabilities=state_probs,
        timestamp=pd.Timestamp("2024-01-01"),
        is_confirmed=confirmed,
        consecutive_bars=5,
    )


# ── Fixtures ────────────────────────────────────────────────────────────────────


@pytest.fixture
def orchestrator() -> StrategyOrchestrator:
    return StrategyOrchestrator(
        config=_standard_config(),
        regime_infos=_three_regime_infos(),
        min_confidence=0.55,
        rebalance_threshold=0.10,
    )


@pytest.fixture
def trending_bars() -> dict:
    """250-bar uptrend for all symbols (close[-1] > EMA50)."""
    return {s: _bars(250, trend=0.001) for s in SYMBOLS}


@pytest.fixture
def flat_bars() -> dict:
    """250-bar flat prices at 100.0 (close[-1] == EMA50 → below-EMA branch)."""
    return {s: _bars(250, trend=0.0) for s in SYMBOLS}


# ── Allocation fraction tests ───────────────────────────────────────────────────


class TestAllocationFractions:
    def test_low_vol_allocation(
        self, orchestrator: StrategyOrchestrator, trending_bars: dict
    ) -> None:
        """Low-vol regime (BULL) should deploy 95% of the portfolio."""
        signals = orchestrator.generate_signals(
            SYMBOLS, trending_bars, _state(0, "BULL")
        )
        assert signals, "Expected at least one signal from low-vol regime"
        total = sum(s.position_size_pct for s in signals)
        assert total == pytest.approx(0.95, abs=1e-4)

    def test_mid_vol_trending_allocation(
        self, orchestrator: StrategyOrchestrator, trending_bars: dict
    ) -> None:
        """Mid-vol + trending (price > EMA50) should deploy 95% of the portfolio."""
        signals = orchestrator.generate_signals(
            SYMBOLS, trending_bars, _state(1, "NEUTRAL")
        )
        assert signals
        total = sum(s.position_size_pct for s in signals)
        assert total == pytest.approx(0.95, abs=1e-4)

    def test_mid_vol_no_trend_allocation(
        self, orchestrator: StrategyOrchestrator, flat_bars: dict
    ) -> None:
        """Mid-vol + flat prices (price == EMA50) should deploy 60% of the portfolio."""
        signals = orchestrator.generate_signals(
            SYMBOLS, flat_bars, _state(1, "NEUTRAL")
        )
        assert signals
        total = sum(s.position_size_pct for s in signals)
        assert total == pytest.approx(0.60, abs=1e-4)

    def test_high_vol_allocation(
        self, orchestrator: StrategyOrchestrator, flat_bars: dict
    ) -> None:
        """High-vol regime (BEAR) should deploy 60% of the portfolio."""
        signals = orchestrator.generate_signals(
            SYMBOLS, flat_bars, _state(2, "BEAR")
        )
        assert signals
        total = sum(s.position_size_pct for s in signals)
        assert total == pytest.approx(0.60, abs=1e-4)


# ── Weight normalisation tests ──────────────────────────────────────────────────


class TestWeightNormalisation:
    def test_weights_sum_to_allocation(
        self, orchestrator: StrategyOrchestrator, trending_bars: dict
    ) -> None:
        """Sum of all target weights must equal the strategy's total allocation."""
        signals = orchestrator.generate_signals(
            SYMBOLS, trending_bars, _state(0, "BULL")
        )
        total = sum(s.position_size_pct for s in signals)
        assert total == pytest.approx(0.95, abs=1e-4)

    def test_no_negative_weights_in_long_only(
        self, orchestrator: StrategyOrchestrator, trending_bars: dict
    ) -> None:
        """All position weights must be non-negative (long-only strategy)."""
        signals = orchestrator.generate_signals(
            SYMBOLS, trending_bars, _state(0, "BULL")
        )
        for s in signals:
            assert s.position_size_pct >= 0.0, (
                f"{s.symbol}: negative weight {s.position_size_pct}"
            )

    def test_equal_weights_across_symbols(
        self, orchestrator: StrategyOrchestrator, trending_bars: dict
    ) -> None:
        """Default equal-weighting: identical price series → identical weights."""
        signals = orchestrator.generate_signals(
            SYMBOLS, trending_bars, _state(0, "BULL")
        )
        assert len(signals) == N_SYMBOLS
        weights = [s.position_size_pct for s in signals]
        for w in weights:
            assert w == pytest.approx(weights[0], abs=1e-5)

    def test_direction_is_long_for_all_signals(
        self, orchestrator: StrategyOrchestrator, trending_bars: dict
    ) -> None:
        """Strategy is always long — never short."""
        for state_id, label in [(0, "BULL"), (1, "NEUTRAL"), (2, "BEAR")]:
            signals = orchestrator.generate_signals(
                SYMBOLS, trending_bars, _state(state_id, label)
            )
            for s in signals:
                assert s.direction == Direction.LONG, (
                    f"{s.symbol} in {label} is not LONG: {s.direction}"
                )


# ── Leverage tests ──────────────────────────────────────────────────────────────


class TestLeverage:
    def test_low_vol_applies_leverage(
        self, orchestrator: StrategyOrchestrator, trending_bars: dict
    ) -> None:
        """Low-vol regime with high confidence should carry 1.25× leverage."""
        signals = orchestrator.generate_signals(
            SYMBOLS, trending_bars, _state(0, "BULL", prob=0.90)
        )
        assert signals
        for s in signals:
            assert s.leverage == pytest.approx(1.25)

    def test_mid_vol_no_leverage(
        self, orchestrator: StrategyOrchestrator, trending_bars: dict
    ) -> None:
        """Mid-vol regime should use 1.0× leverage."""
        signals = orchestrator.generate_signals(
            SYMBOLS, trending_bars, _state(1, "NEUTRAL", prob=0.90)
        )
        assert signals
        for s in signals:
            assert s.leverage == pytest.approx(1.0)

    def test_high_vol_no_leverage(
        self, orchestrator: StrategyOrchestrator, flat_bars: dict
    ) -> None:
        """High-vol regime should use 1.0× leverage."""
        signals = orchestrator.generate_signals(
            SYMBOLS, flat_bars, _state(2, "BEAR", prob=0.90)
        )
        assert signals
        for s in signals:
            assert s.leverage == pytest.approx(1.0)

    def test_low_confidence_suppresses_leverage(
        self, orchestrator: StrategyOrchestrator, trending_bars: dict
    ) -> None:
        """Low-vol regime below min_confidence should fall back to 1.0× leverage."""
        # prob=0.40 < min_confidence=0.55 → uncertainty mode → leverage clipped to 1.0
        signals = orchestrator.generate_signals(
            SYMBOLS, trending_bars, _state(0, "BULL", prob=0.40)
        )
        assert signals
        for s in signals:
            assert s.leverage == pytest.approx(1.0), (
                f"Expected leverage=1.0 in uncertainty mode, got {s.leverage}"
            )


# ── Uncertainty discount tests ──────────────────────────────────────────────────


class TestUncertaintyDiscount:
    def test_low_confidence_reduces_weights(
        self, orchestrator: StrategyOrchestrator, trending_bars: dict
    ) -> None:
        """Position sizes at confidence < min_confidence should be scaled by 0.50."""
        high_conf = orchestrator.generate_signals(
            SYMBOLS, trending_bars, _state(0, "BULL", prob=0.90)
        )
        # Reset weights so rebalance filter doesn't interfere
        orchestrator.reset_weights()
        low_conf = orchestrator.generate_signals(
            SYMBOLS, trending_bars, _state(0, "BULL", prob=0.40)
        )
        assert high_conf and low_conf
        high_total = sum(s.position_size_pct for s in high_conf)
        low_total = sum(s.position_size_pct for s in low_conf)
        assert low_total == pytest.approx(high_total * 0.50, abs=1e-4)

    def test_high_confidence_no_discount(
        self, orchestrator: StrategyOrchestrator, trending_bars: dict
    ) -> None:
        """High-confidence signals must not be prefixed with [UNCERTAINTY]."""
        signals = orchestrator.generate_signals(
            SYMBOLS, trending_bars, _state(0, "BULL", prob=0.90)
        )
        assert signals
        for s in signals:
            assert not s.reasoning.startswith("[UNCERTAINTY]"), (
                f"{s.symbol}: unexpected [UNCERTAINTY] at high confidence"
            )

    def test_uncertainty_prefix_on_low_confidence(
        self, orchestrator: StrategyOrchestrator, trending_bars: dict
    ) -> None:
        """Low-confidence signals must be prefixed with [UNCERTAINTY]."""
        signals = orchestrator.generate_signals(
            SYMBOLS, trending_bars, _state(0, "BULL", prob=0.40)
        )
        assert signals
        for s in signals:
            assert s.reasoning.startswith("[UNCERTAINTY]"), (
                f"{s.symbol}: missing [UNCERTAINTY] prefix for low-confidence signal"
            )

    def test_unconfirmed_regime_activates_uncertainty(
        self, orchestrator: StrategyOrchestrator, trending_bars: dict
    ) -> None:
        """is_confirmed=False should also trigger uncertainty mode."""
        # Use high probability but is_confirmed=False
        signals = orchestrator.generate_signals(
            SYMBOLS, trending_bars, _state(0, "BULL", prob=0.90, confirmed=False)
        )
        assert signals
        for s in signals:
            assert s.reasoning.startswith("[UNCERTAINTY]")
            assert s.leverage == pytest.approx(1.0)


# ── Rebalance threshold tests ───────────────────────────────────────────────────


class TestRebalanceThreshold:
    def test_large_deviation_triggers_rebalance(
        self, orchestrator: StrategyOrchestrator, trending_bars: dict
    ) -> None:
        """Fresh orchestrator (all current weights = 0) should generate signals for all symbols."""
        # With current_weights=0 and target=0.19, deviation >> 10% threshold
        signals = orchestrator.generate_signals(
            SYMBOLS, trending_bars, _state(0, "BULL")
        )
        assert len(signals) == N_SYMBOLS

    def test_small_deviation_no_rebalance(
        self, orchestrator: StrategyOrchestrator, trending_bars: dict
    ) -> None:
        """Weights already at target → deviation < threshold → no signals returned."""
        # Compute the expected per-symbol target: 0.95 / 5 = 0.19
        target_per_symbol = round(0.95 / N_SYMBOLS, 6)
        # Set current weights to match the target exactly
        orchestrator.update_weights({s: target_per_symbol for s in SYMBOLS})

        signals = orchestrator.generate_signals(
            SYMBOLS, trending_bars, _state(0, "BULL")
        )
        assert signals == [], (
            f"Expected no signals when weights are at target, got {len(signals)}"
        )

    def test_partial_rebalance_when_one_drifts(
        self, orchestrator: StrategyOrchestrator, trending_bars: dict
    ) -> None:
        """Only the symbol that drifted past threshold should get a rebalance signal."""
        target = round(0.95 / N_SYMBOLS, 6)
        # Set all but SPY to match the target
        weights = {s: target for s in SYMBOLS}
        weights["SPY"] = 0.0      # SPY has drifted away → deviation = target > threshold
        orchestrator.update_weights(weights)

        signals = orchestrator.generate_signals(
            SYMBOLS, trending_bars, _state(0, "BULL")
        )
        assert len(signals) == 1
        assert signals[0].symbol == "SPY"


# ── Vol-rank strategy mapping tests ────────────────────────────────────────────


class TestVolRankMapping:
    def test_low_vol_regime_uses_low_vol_strategy(
        self, orchestrator: StrategyOrchestrator
    ) -> None:
        strategy = orchestrator.get_strategy_for_regime(0)   # BULL, rank=0.0
        assert isinstance(strategy, LowVolBullStrategy)

    def test_mid_vol_regime_uses_mid_vol_strategy(
        self, orchestrator: StrategyOrchestrator
    ) -> None:
        strategy = orchestrator.get_strategy_for_regime(1)   # NEUTRAL, rank=0.5
        assert isinstance(strategy, MidVolCautiousStrategy)

    def test_high_vol_regime_uses_high_vol_strategy(
        self, orchestrator: StrategyOrchestrator
    ) -> None:
        strategy = orchestrator.get_strategy_for_regime(2)   # BEAR, rank=1.0
        assert isinstance(strategy, HighVolDefensiveStrategy)

    def test_vol_ranks_span_zero_to_one(
        self, orchestrator: StrategyOrchestrator
    ) -> None:
        """With 3 regimes, vol ranks must be exactly 0.0, 0.5, and 1.0."""
        ranks = sorted(orchestrator.get_vol_rank(rid) for rid in range(3))
        assert ranks == pytest.approx([0.0, 0.5, 1.0])

    def test_single_regime_vol_rank_is_half(self) -> None:
        """Edge case: single regime → vol_rank = 0.5 (neither low nor high)."""
        infos = {
            "BULL": RegimeInfo(
                regime_id=0, regime_name="BULL", expected_return=0.001,
                expected_volatility=0.01, recommended_strategy_type="LowVol",
                max_leverage_allowed=1.25, max_position_size_pct=0.95,
                min_confidence_to_act=0.55,
            )
        }
        orc = StrategyOrchestrator(_standard_config(), infos)
        assert orc.get_vol_rank(0) == pytest.approx(0.5)
        # vol_rank=0.5 → 0.33 < 0.5 < 0.67 → MidVolCautiousStrategy
        assert isinstance(orc.get_strategy_for_regime(0), MidVolCautiousStrategy)

    def test_label_to_strategy_covers_all_known_labels(self) -> None:
        """LABEL_TO_STRATEGY must include all documented regime labels."""
        expected = {
            "BEAR", "NEUTRAL", "BULL",           # 3-state
            "CRASH", "EUPHORIA",                  # 4-state additions
            "STRONG_BEAR", "WEAK_BEAR",           # 6/7-state additions
            "WEAK_BULL", "STRONG_BULL",
        }
        assert expected.issubset(LABEL_TO_STRATEGY.keys())

    def test_label_to_strategy_values_are_strategy_classes(self) -> None:
        """Every entry in LABEL_TO_STRATEGY must be a BaseStrategy subclass."""
        from core.regime_strategies import BaseStrategy
        for label, cls in LABEL_TO_STRATEGY.items():
            assert issubclass(cls, BaseStrategy), (
                f"LABEL_TO_STRATEGY['{label}'] = {cls} is not a BaseStrategy subclass"
            )


# ── Backward-compatible alias tests ────────────────────────────────────────────


class TestBackwardCompatAliases:
    def test_regime_strategy_is_orchestrator(
        self, orchestrator: StrategyOrchestrator
    ) -> None:
        """RegimeStrategy must be an alias for StrategyOrchestrator."""
        assert RegimeStrategy is StrategyOrchestrator
        assert isinstance(orchestrator, RegimeStrategy)

    def test_allocation_result_is_signal(
        self, orchestrator: StrategyOrchestrator, trending_bars: dict
    ) -> None:
        """AllocationResult must be an alias for Signal."""
        assert AllocationResult is Signal
        signals = orchestrator.generate_signals(
            SYMBOLS, trending_bars, _state(0, "BULL")
        )
        for s in signals:
            assert isinstance(s, AllocationResult)


# ── Stop-loss sanity tests ──────────────────────────────────────────────────────


class TestStopLoss:
    def test_stop_loss_below_entry(
        self, orchestrator: StrategyOrchestrator, trending_bars: dict
    ) -> None:
        """Stop loss must always be below the entry price."""
        for state_id, label in [(0, "BULL"), (1, "NEUTRAL"), (2, "BEAR")]:
            signals = orchestrator.generate_signals(
                SYMBOLS, trending_bars, _state(state_id, label)
            )
            for s in signals:
                assert s.stop_loss < s.entry_price, (
                    f"{s.symbol} ({label}): stop {s.stop_loss} >= entry {s.entry_price}"
                )

    def test_risk_per_trade_positive(
        self, orchestrator: StrategyOrchestrator, trending_bars: dict
    ) -> None:
        """risk_per_trade property must be non-negative."""
        signals = orchestrator.generate_signals(
            SYMBOLS, trending_bars, _state(0, "BULL")
        )
        for s in signals:
            assert s.risk_per_trade >= 0.0

    def test_low_vol_stop_uses_wider_atr_mult(
        self, orchestrator: StrategyOrchestrator, flat_bars: dict
    ) -> None:
        """
        Low-vol stop = max(price−3ATR, EMA50−0.5ATR).
        High-vol stop = EMA50−1.0ATR.
        On flat prices, high-vol stop < low-vol stop (narrower EMA-ATR offset
        means tighter stop, but both stay below entry).
        Both must be below entry price — checked via risk_per_trade > 0.
        """
        lv_signals = orchestrator.generate_signals(
            SYMBOLS, flat_bars, _state(0, "BULL")
        )
        hv_signals = orchestrator.generate_signals(
            SYMBOLS, flat_bars, _state(2, "BEAR")
        )
        for s in lv_signals + hv_signals:
            assert s.risk_per_trade > 0.0

    def test_insufficient_bars_returns_no_signals(
        self, orchestrator: StrategyOrchestrator
    ) -> None:
        """Bars below MIN_BARS (60) should produce no signals — not raise."""
        short_bars = {s: _bars(10) for s in SYMBOLS}
        for state_id, label in [(0, "BULL"), (1, "NEUTRAL"), (2, "BEAR")]:
            signals = orchestrator.generate_signals(
                SYMBOLS, short_bars, _state(state_id, label)
            )
            assert signals == [], f"Expected no signals on thin bars for {label}"
