"""
test_signal_generator.py — Unit tests for core/signal_generator.py.

Tests cover:
  - SignalGenerator.generate(): normal path, halted state, reduced state
  - delta weight computation
  - update_current_weights()
  - get_current_regime() and get_last_signal() accessors
  - notes building (low confidence, unconfirmed regime, drawdown)
"""

from __future__ import annotations

from typing import Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from core.hmm_engine import HMMEngine, RegimeInfo, RegimeState
from core.regime_strategies import RegimeStrategy, Signal, Direction
from core.risk_manager import RiskManager, TradingState
from core.signal_generator import SignalGenerator, PortfolioSignal


# ── Helpers / Factories ────────────────────────────────────────────────────────

SYMBOLS = ["SPY", "QQQ", "AAPL"]


def _regime_state(
    label: str = "BULL",
    probability: float = 0.80,
    is_confirmed: bool = True,
    consecutive_bars: int = 5,
    n_states: int = 3,
) -> RegimeState:
    probs = np.zeros(n_states)
    probs[0] = probability
    return RegimeState(
        label=label,
        state_id=0,
        probability=probability,
        state_probabilities=probs,
        timestamp=pd.Timestamp("2024-01-15"),
        is_confirmed=is_confirmed,
        consecutive_bars=consecutive_bars,
    )


def _mock_hmm(
    label: str = "BULL",
    probability: float = 0.80,
    is_confirmed: bool = True,
) -> MagicMock:
    engine = MagicMock(spec=HMMEngine)
    engine.update.return_value = _regime_state(label, probability, is_confirmed)
    engine.min_confidence = 0.55
    return engine


def _mock_strategy(
    symbols: List[str] = SYMBOLS,
    position_size_pct: float = 0.30,
    leverage: float = 1.0,
) -> MagicMock:
    strategy = MagicMock(spec=RegimeStrategy)
    signals = []
    for sym in symbols:
        sig = MagicMock(spec=Signal)
        sig.symbol = sym
        sig.is_long = True
        sig.direction = Direction.LONG
        sig.position_size_pct = position_size_pct / len(symbols)
        sig.leverage = leverage
        sig.stop_loss = 350.0
        sig.strategy_name = "TestStrategy"
        signals.append(sig)
    strategy.generate_signals.return_value = signals
    return strategy


def _prices(symbols: List[str] = SYMBOLS, n: int = 100) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=n)
    data = {sym: np.linspace(100, 110, n) + i * 20 for i, sym in enumerate(symbols)}
    return pd.DataFrame(data, index=dates)


def _features(n: int = 100, n_features: int = 14) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.normal(0, 1, (n, n_features))


def _make_generator(
    hmm_label: str = "BULL",
    probability: float = 0.80,
    is_confirmed: bool = True,
    trading_state: TradingState = TradingState.NORMAL,
    leverage: float = 1.0,
) -> SignalGenerator:
    engine = _mock_hmm(hmm_label, probability, is_confirmed)
    strategy = _mock_strategy(SYMBOLS, leverage=leverage)
    rm = MagicMock(spec=RiskManager)
    rm.get_trading_state.return_value = trading_state
    from core.risk_manager import DrawdownState
    rm.get_drawdown_state.return_value = DrawdownState(
        peak_equity=100_000.0,
        current_equity=100_000.0,
        daily_start_equity=100_000.0,
        weekly_start_equity=100_000.0,
        dd_from_peak=0.0,
        daily_dd=0.0,
        weekly_dd=0.0,
    )
    return SignalGenerator(
        hmm_engine=engine,
        strategy=strategy,
        risk_manager=rm,
        symbols=SYMBOLS,
    )


# ── Basic generate() tests ────────────────────────────────────────────────────

class TestGenerate:
    def test_returns_portfolio_signal(self) -> None:
        sg = _make_generator()
        signal = sg.generate(
            features=_features(),
            prices=_prices(),
            timestamp=pd.Timestamp("2024-01-15"),
        )
        assert isinstance(signal, PortfolioSignal)

    def test_signal_has_correct_regime(self) -> None:
        sg = _make_generator(hmm_label="BEAR")
        signal = sg.generate(_features(), _prices(), pd.Timestamp("2024-01-15"))
        assert signal.regime == "BEAR"

    def test_signal_confidence_matches_hmm(self) -> None:
        sg = _make_generator(probability=0.90)
        signal = sg.generate(_features(), _prices(), pd.Timestamp("2024-01-15"))
        assert abs(signal.confidence - 0.90) < 1e-6

    def test_signal_is_stable_when_confirmed(self) -> None:
        sg = _make_generator(is_confirmed=True)
        signal = sg.generate(_features(), _prices(), pd.Timestamp("2024-01-15"))
        assert signal.is_stable is True

    def test_signal_not_stable_when_unconfirmed(self) -> None:
        sg = _make_generator(is_confirmed=False)
        signal = sg.generate(_features(), _prices(), pd.Timestamp("2024-01-15"))
        assert signal.is_stable is False

    def test_target_weights_keys_match_symbols(self) -> None:
        sg = _make_generator()
        signal = sg.generate(_features(), _prices(), pd.Timestamp("2024-01-15"))
        assert set(signal.target_weights.keys()) == set(SYMBOLS)

    def test_delta_weights_keys_match_symbols(self) -> None:
        sg = _make_generator()
        signal = sg.generate(_features(), _prices(), pd.Timestamp("2024-01-15"))
        assert set(signal.delta_weights.keys()) == set(SYMBOLS)

    def test_trading_allowed_in_normal_state(self) -> None:
        sg = _make_generator(trading_state=TradingState.NORMAL)
        signal = sg.generate(_features(), _prices(), pd.Timestamp("2024-01-15"))
        assert signal.trading_allowed is True

    def test_timestamp_passed_through(self) -> None:
        sg = _make_generator()
        ts = pd.Timestamp("2024-06-01 10:30")
        signal = sg.generate(_features(), _prices(), timestamp=ts)
        assert signal.timestamp == ts


# ── Halted state ──────────────────────────────────────────────────────────────

class TestHaltedState:
    def test_trading_not_allowed_when_halted(self) -> None:
        sg = _make_generator(trading_state=TradingState.HALTED)
        signal = sg.generate(_features(), _prices(), pd.Timestamp("2024-01-15"))
        assert signal.trading_allowed is False

    def test_all_weights_zero_when_halted(self) -> None:
        sg = _make_generator(trading_state=TradingState.HALTED)
        signal = sg.generate(_features(), _prices(), pd.Timestamp("2024-01-15"))
        assert all(w == 0.0 for w in signal.target_weights.values())

    def test_leverage_is_one_when_halted(self) -> None:
        sg = _make_generator(trading_state=TradingState.HALTED, leverage=1.25)
        signal = sg.generate(_features(), _prices(), pd.Timestamp("2024-01-15"))
        assert signal.leverage == 1.0

    def test_halted_note_in_signal(self) -> None:
        sg = _make_generator(trading_state=TradingState.HALTED)
        signal = sg.generate(_features(), _prices(), pd.Timestamp("2024-01-15"))
        assert any("HALT" in n.upper() for n in signal.notes)


# ── Reduced state ─────────────────────────────────────────────────────────────

class TestReducedState:
    def test_trading_allowed_in_reduced(self) -> None:
        sg = _make_generator(trading_state=TradingState.REDUCED)
        signal = sg.generate(_features(), _prices(), pd.Timestamp("2024-01-15"))
        assert signal.trading_allowed is True

    def test_weights_halved_in_reduced_vs_normal(self) -> None:
        """REDUCED state should halve all target weights vs NORMAL."""
        sg_normal = _make_generator(trading_state=TradingState.NORMAL)
        sg_reduced = _make_generator(trading_state=TradingState.REDUCED)

        ts = pd.Timestamp("2024-01-15")
        sig_normal = sg_normal.generate(_features(), _prices(), ts)
        sig_reduced = sg_reduced.generate(_features(), _prices(), ts)

        for sym in SYMBOLS:
            assert abs(sig_reduced.target_weights[sym] -
                       sig_normal.target_weights[sym] * 0.5) < 1e-9

    def test_reduced_note_present(self) -> None:
        sg = _make_generator(trading_state=TradingState.REDUCED)
        signal = sg.generate(_features(), _prices(), pd.Timestamp("2024-01-15"))
        assert any("REDUCED" in n for n in signal.notes)


# ── Delta weight computation ──────────────────────────────────────────────────

class TestDeltaWeights:
    def test_delta_equals_target_minus_current(self) -> None:
        sg = _make_generator()
        current = {"SPY": 0.20, "QQQ": 0.10, "AAPL": 0.0}
        sg.update_current_weights(current)

        signal = sg.generate(_features(), _prices(), pd.Timestamp("2024-01-15"))
        for sym in SYMBOLS:
            expected_delta = (
                signal.target_weights[sym] - current.get(sym, 0.0)
            )
            assert abs(signal.delta_weights[sym] - expected_delta) < 1e-9

    def test_initial_delta_equals_target_when_flat(self) -> None:
        """With no current positions, delta should equal target weights."""
        sg = _make_generator()
        signal = sg.generate(_features(), _prices(), pd.Timestamp("2024-01-15"))
        for sym in SYMBOLS:
            assert abs(
                signal.delta_weights[sym] - signal.target_weights[sym]
            ) < 1e-9


# ── update_current_weights ───────────────────────────────────────────────────

class TestUpdateCurrentWeights:
    def test_partial_update_leaves_others_at_zero(self) -> None:
        sg = _make_generator()
        sg.update_current_weights({"SPY": 0.30})
        # QQQ and AAPL not provided — should default to 0
        assert sg._current_weights.get("QQQ", 0.0) == 0.0

    def test_full_update_applied(self) -> None:
        sg = _make_generator()
        weights = {"SPY": 0.30, "QQQ": 0.20, "AAPL": 0.10}
        sg.update_current_weights(weights)
        for sym, w in weights.items():
            assert sg._current_weights[sym] == w


# ── Accessors ─────────────────────────────────────────────────────────────────

class TestAccessors:
    def test_get_current_regime_none_before_generate(self) -> None:
        sg = _make_generator()
        assert sg.get_current_regime() is None

    def test_get_current_regime_after_generate(self) -> None:
        sg = _make_generator(hmm_label="NEUTRAL")
        sg.generate(_features(), _prices(), pd.Timestamp("2024-01-15"))
        assert sg.get_current_regime() == "NEUTRAL"

    def test_get_last_signal_none_before_generate(self) -> None:
        sg = _make_generator()
        assert sg.get_last_signal() is None

    def test_get_last_signal_after_generate(self) -> None:
        sg = _make_generator()
        ts = pd.Timestamp("2024-01-15")
        signal = sg.generate(_features(), _prices(), ts)
        assert sg.get_last_signal() is signal

    def test_last_signal_updated_on_each_call(self) -> None:
        sg = _make_generator()
        ts1, ts2 = pd.Timestamp("2024-01-15"), pd.Timestamp("2024-01-16")
        sg.generate(_features(), _prices(), ts1)
        sg.generate(_features(), _prices(), ts2)
        assert sg.get_last_signal().timestamp == ts2


# ── Notes building ────────────────────────────────────────────────────────────

class TestNotesBuilding:
    def test_notes_include_regime_info(self) -> None:
        sg = _make_generator(hmm_label="BULL", probability=0.85)
        signal = sg.generate(_features(), _prices(), pd.Timestamp("2024-01-15"))
        regime_note = signal.notes[0]
        assert "BULL" in regime_note
        assert "0.850" in regime_note

    def test_low_confidence_note_added(self) -> None:
        """Probability below min_confidence should add a note."""
        sg = _make_generator(probability=0.40)
        # Patch min_confidence so this is below threshold
        sg.hmm_engine.min_confidence = 0.55
        signal = sg.generate(_features(), _prices(), pd.Timestamp("2024-01-15"))
        assert any("confidence" in n.lower() or "uncertainty" in n.lower()
                   for n in signal.notes)

    def test_unconfirmed_regime_note(self) -> None:
        sg = _make_generator(is_confirmed=False)
        signal = sg.generate(_features(), _prices(), pd.Timestamp("2024-01-15"))
        assert any("unconfirmed" in n.lower() or "Regime unconfirmed" in n
                   for n in signal.notes)
