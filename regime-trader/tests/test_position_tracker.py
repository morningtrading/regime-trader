"""
test_position_tracker.py — Unit tests for broker/position_tracker.py.

Tests cover:
  - PositionTracker.startup_sync(): builds initial PortfolioSnapshot
  - PositionTracker.update(): refreshes positions and updates peak equity
  - Accessors: get_current_weights, get_portfolio_value, get_unrealized_pnl,
               get_drawdown_from_peak, get_position, get_all_positions,
               get_gross_exposure
  - record_entry and set_stop_level
  - register_fill_callback
  - to_portfolio_state
  - _update_peak_equity and _build_portfolio_snapshot helpers
"""

from __future__ import annotations

import datetime as dt
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from broker.position_tracker import (
    FillEvent,
    PortfolioSnapshot,
    PositionMeta,
    PositionSnapshot,
    PositionTracker,
)
from broker.alpaca_client import AlpacaClient
from core.risk_manager import RiskManager


# ── Helpers ────────────────────────────────────────────────────────────────────

def _mock_position(
    symbol: str = "SPY",
    qty: float = 10.0,
    avg_entry: float = 400.0,
    current_price: float = 410.0,
    market_value: float = 4_100.0,
    unrealized_pl: float = 100.0,
) -> MagicMock:
    pos = MagicMock()
    pos.symbol = symbol
    pos.qty = str(qty)
    pos.avg_entry_price = str(avg_entry)
    pos.current_price = str(current_price)
    pos.market_value = str(market_value)
    pos.unrealized_pl = str(unrealized_pl)
    return pos


def _mock_account(equity: float = 100_000.0, cash: float = 90_000.0) -> MagicMock:
    acc = MagicMock()
    acc.equity = str(equity)
    acc.cash = str(cash)
    return acc


def _mock_client(
    positions=None,
    equity: float = 100_000.0,
    cash: float = 90_000.0,
) -> MagicMock:
    client = MagicMock(spec=AlpacaClient)
    client._connected = True
    client._api_key = "test-key"
    client._secret_key = "test-secret"
    client.paper = True
    client.get_all_positions.return_value = positions or []
    client.get_account.return_value = _mock_account(equity, cash)
    client.get_buying_power.return_value = cash
    return client


def _tracker(
    positions=None,
    equity: float = 100_000.0,
    cash: float = 90_000.0,
    risk_manager=None,
) -> PositionTracker:
    client = _mock_client(positions, equity, cash)
    return PositionTracker(client=client, risk_manager=risk_manager)


# ── startup_sync ──────────────────────────────────────────────────────────────

class TestStartupSync:
    def test_returns_portfolio_snapshot(self) -> None:
        tracker = _tracker()
        snap = tracker.startup_sync()
        assert isinstance(snap, PortfolioSnapshot)

    def test_equity_matches_account(self) -> None:
        tracker = _tracker(equity=123_456.0)
        snap = tracker.startup_sync()
        assert snap.total_equity == 123_456.0

    def test_cash_matches_account(self) -> None:
        tracker = _tracker(cash=50_000.0, equity=100_000.0)
        snap = tracker.startup_sync()
        assert snap.cash == 50_000.0

    def test_positions_loaded(self) -> None:
        pos = [_mock_position("SPY")]
        tracker = _tracker(positions=pos, equity=100_000.0, cash=90_000.0)
        tracker.startup_sync()
        assert "SPY" in tracker._positions

    def test_empty_positions_cleared(self) -> None:
        tracker = _tracker(positions=[])
        tracker.startup_sync()
        assert len(tracker._positions) == 0

    def test_peak_equity_set_on_sync(self) -> None:
        tracker = _tracker(equity=200_000.0)
        tracker.startup_sync()
        assert tracker._peak_equity == 200_000.0

    def test_meta_seeded_for_each_position(self) -> None:
        pos = [_mock_position("SPY"), _mock_position("QQQ", market_value=5_000.0)]
        tracker = _tracker(positions=pos)
        tracker.startup_sync()
        assert "SPY" in tracker._meta
        assert "QQQ" in tracker._meta

    def test_phantom_positions_removed_from_meta(self) -> None:
        """Meta entries not in Alpaca positions should be cleaned up."""
        tracker = _tracker(positions=[])
        # Manually inject a phantom
        tracker._meta["PHANTOM"] = PositionMeta(
            symbol="PHANTOM",
            entry_time=dt.datetime.now(dt.timezone.utc),
            entry_price=100.0,
        )
        tracker.startup_sync()
        assert "PHANTOM" not in tracker._meta


# ── update ────────────────────────────────────────────────────────────────────

class TestUpdate:
    def test_returns_portfolio_snapshot(self) -> None:
        tracker = _tracker()
        tracker.startup_sync()
        snap = tracker.update()
        assert isinstance(snap, PortfolioSnapshot)

    def test_positions_refreshed(self) -> None:
        tracker = _tracker(positions=[])
        tracker.startup_sync()
        # Now add a position and refresh
        tracker.client.get_all_positions.return_value = [_mock_position("AAPL")]
        tracker.update()
        assert "AAPL" in tracker._positions

    def test_equity_history_grows(self) -> None:
        tracker = _tracker()
        tracker.startup_sync()
        tracker.update()
        tracker.update()
        assert len(tracker._equity_history) == 2

    def test_peak_equity_tracks_high(self) -> None:
        tracker = _tracker(equity=100_000.0)
        tracker.startup_sync()
        tracker.client.get_account.return_value = _mock_account(equity=120_000.0)
        tracker.update()
        assert tracker._peak_equity == 120_000.0

    def test_error_returns_last_snapshot(self) -> None:
        tracker = _tracker()
        tracker.startup_sync()
        # Force an error on the next call
        tracker.client.get_all_positions.side_effect = RuntimeError("network error")
        snap = tracker.update()
        assert isinstance(snap, PortfolioSnapshot)

    def test_risk_manager_updated_on_update(self) -> None:
        rm = MagicMock(spec=RiskManager)
        tracker = _tracker(risk_manager=rm)
        tracker.startup_sync()
        tracker.update()
        rm.update_equity.assert_called()


# ── Accessors ─────────────────────────────────────────────────────────────────

class TestAccessors:
    def _ready_tracker(self) -> PositionTracker:
        pos = [_mock_position("SPY", qty=10, market_value=4_100.0)]
        tracker = _tracker(positions=pos, equity=100_000.0, cash=90_000.0)
        tracker.startup_sync()
        return tracker

    def test_get_current_weights(self) -> None:
        tracker = self._ready_tracker()
        weights = tracker.get_current_weights()
        assert "SPY" in weights
        assert 0.0 < weights["SPY"] < 1.0

    def test_get_portfolio_value(self) -> None:
        tracker = self._ready_tracker()
        assert tracker.get_portfolio_value() == 100_000.0

    def test_get_portfolio_value_before_sync_is_zero(self) -> None:
        tracker = _tracker()
        assert tracker.get_portfolio_value() == 0.0

    def test_get_unrealized_pnl(self) -> None:
        tracker = self._ready_tracker()
        # Our mock position has unrealized_pl=100.0
        pnl = tracker.get_unrealized_pnl()
        assert pnl == 100.0

    def test_get_position_returns_snapshot(self) -> None:
        tracker = self._ready_tracker()
        snap = tracker.get_position("SPY")
        assert snap is not None
        assert snap.symbol == "SPY"

    def test_get_position_unknown_returns_none(self) -> None:
        tracker = self._ready_tracker()
        assert tracker.get_position("UNKNOWN") is None

    def test_get_all_positions(self) -> None:
        tracker = self._ready_tracker()
        positions = tracker.get_all_positions()
        assert len(positions) == 1
        assert positions[0].symbol == "SPY"

    def test_get_gross_exposure(self) -> None:
        tracker = self._ready_tracker()
        exposure = tracker.get_gross_exposure()
        # SPY weight = 4100/100000 = 0.041
        assert 0.0 < exposure < 1.0

    def test_get_last_snapshot(self) -> None:
        tracker = self._ready_tracker()
        snap = tracker.get_last_snapshot()
        assert isinstance(snap, PortfolioSnapshot)

    def test_get_drawdown_from_peak_zero_at_start(self) -> None:
        """At peak, drawdown should be 0."""
        tracker = _tracker(equity=100_000.0)
        tracker.startup_sync()
        assert tracker.get_drawdown_from_peak() == 0.0

    def test_get_drawdown_from_peak_positive_after_loss(self) -> None:
        tracker = _tracker(equity=100_000.0)
        tracker.startup_sync()
        # Simulate equity falling
        tracker.client.get_account.return_value = _mock_account(equity=90_000.0)
        tracker.update()
        dd = tracker.get_drawdown_from_peak()
        assert dd > 0.0
        assert abs(dd - 0.10) < 0.01


# ── record_entry and set_stop_level ──────────────────────────────────────────

class TestRecordEntryAndStop:
    def test_record_entry_stores_meta(self) -> None:
        tracker = _tracker()
        tracker.record_entry("SPY", price=400.0, regime="BULL", stop=390.0)
        assert "SPY" in tracker._meta
        assert tracker._meta["SPY"].entry_price == 400.0
        assert tracker._meta["SPY"].regime_at_entry == "BULL"
        assert tracker._meta["SPY"].stop_level == 390.0

    def test_set_stop_level_updates_meta(self) -> None:
        tracker = _tracker()
        tracker.record_entry("AAPL", price=180.0)
        tracker.set_stop_level("AAPL", stop=170.0)
        assert tracker._meta["AAPL"].stop_level == 170.0

    def test_set_stop_level_on_position_snapshot(self) -> None:
        pos = [_mock_position("SPY")]
        tracker = _tracker(positions=pos)
        tracker.startup_sync()
        tracker.set_stop_level("SPY", stop=390.0)
        assert tracker._positions["SPY"].stop_level == 390.0

    def test_set_stop_level_unknown_symbol_no_error(self) -> None:
        """set_stop_level on an unknown symbol should not raise."""
        tracker = _tracker()
        tracker.set_stop_level("UNKNOWN", stop=100.0)   # should not raise


# ── register_fill_callback ────────────────────────────────────────────────────

class TestFillCallback:
    def test_callback_registered(self) -> None:
        tracker = _tracker()
        cb = MagicMock()
        tracker.register_fill_callback(cb)
        assert cb in tracker._fill_callbacks

    def test_multiple_callbacks_stored(self) -> None:
        tracker = _tracker()
        cb1, cb2 = MagicMock(), MagicMock()
        tracker.register_fill_callback(cb1)
        tracker.register_fill_callback(cb2)
        assert len(tracker._fill_callbacks) == 2


# ── to_portfolio_state ────────────────────────────────────────────────────────

class TestToPortfolioState:
    def test_returns_portfolio_state(self) -> None:
        from core.risk_manager import PortfolioState
        pos = [_mock_position("SPY", market_value=5_000.0)]
        tracker = _tracker(positions=pos, equity=100_000.0, cash=90_000.0)
        tracker.startup_sync()
        state = tracker.to_portfolio_state()
        assert isinstance(state, PortfolioState)

    def test_equity_matches_snapshot(self) -> None:
        from core.risk_manager import PortfolioState
        tracker = _tracker(equity=100_000.0, cash=80_000.0)
        tracker.startup_sync()
        state = tracker.to_portfolio_state()
        assert state.equity == 100_000.0

    def test_positions_included(self) -> None:
        from core.risk_manager import PortfolioState
        pos = [_mock_position("SPY", market_value=5_000.0)]
        tracker = _tracker(positions=pos, equity=100_000.0, cash=90_000.0)
        tracker.startup_sync()
        state = tracker.to_portfolio_state()
        assert "SPY" in state.positions

    def test_regime_from_callback(self) -> None:
        from core.risk_manager import PortfolioState
        tracker = _tracker()
        tracker.current_regime_fn = lambda: "BULL"
        tracker.startup_sync()
        state = tracker.to_portfolio_state()
        assert state.current_regime == "BULL"


# ── _update_peak_equity (via update) ──────────────────────────────────────────

class TestPeakEquity:
    def test_peak_never_decreases(self) -> None:
        tracker = _tracker(equity=100_000.0)
        tracker.startup_sync()
        tracker.client.get_account.return_value = _mock_account(equity=80_000.0)
        tracker.update()
        assert tracker._peak_equity == 100_000.0

    def test_peak_increases_on_new_high(self) -> None:
        tracker = _tracker(equity=100_000.0)
        tracker.startup_sync()
        tracker.client.get_account.return_value = _mock_account(equity=110_000.0)
        tracker.update()
        assert tracker._peak_equity == 110_000.0
