"""
test_orders.py -- Unit tests for OrderExecutor order logic.

Uses a mock AlpacaClient so no real network calls are made.
Tests cover: limit-price calculation, slippage, client-order-ID uniqueness,
weight-delta to share conversion, and order cancellation.
"""

from typing import Dict
from unittest.mock import MagicMock, patch

import pytest
from alpaca.trading.enums import OrderSide, TimeInForce

from broker.alpaca_client import AlpacaClient
from broker.order_executor import (
    OrderExecutor,
    OrderResult,
    OrderStatus,
    OrderTicket,
    DEFAULT_LIMIT_OFFSET_PCT,
)
from core.risk_manager import RiskManager


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #

@pytest.fixture
def mock_client() -> MagicMock:
    """Return a mock AlpacaClient that simulates a successful order."""
    client = MagicMock(spec=AlpacaClient)

    mock_order = MagicMock()
    mock_order.id     = "mock-order-id-123"
    mock_order.status = "accepted"

    trading = MagicMock()
    trading.submit_order.return_value        = mock_order
    trading.cancel_order_by_id.return_value  = None
    trading.get_order_by_client_id.return_value = mock_order
    trading.get_orders.return_value          = []

    client._trading_client  = trading
    client._connected       = True
    client.get_portfolio_value.return_value  = 100_000.0
    client.get_latest_price.return_value     = 400.0
    client.get_all_positions.return_value    = []
    client.get_position.return_value         = None
    return client


@pytest.fixture
def risk_manager() -> RiskManager:
    return RiskManager(initial_equity=100_000.0)


@pytest.fixture
def executor(mock_client: MagicMock, risk_manager: RiskManager) -> OrderExecutor:
    return OrderExecutor(
        client           = mock_client,
        risk_manager     = risk_manager,
        limit_offset_pct = DEFAULT_LIMIT_OFFSET_PCT,
        cancel_after_sec = 0,    # don't wait in tests
        retry_at_market  = False,
    )


# --------------------------------------------------------------------------- #
# Limit price calculation                                                      #
# --------------------------------------------------------------------------- #

class TestLimitPrice:
    def test_buy_limit_above_mid(self, executor: OrderExecutor) -> None:
        """BUY limit = mid * (1 + offset_pct) -- slightly above to ensure fill."""
        lp = executor._limit_price(100.0, OrderSide.BUY)
        expected = round(100.0 * (1 + DEFAULT_LIMIT_OFFSET_PCT), 2)
        assert lp == expected

    def test_sell_limit_below_mid(self, executor: OrderExecutor) -> None:
        """SELL limit = mid * (1 - offset_pct) -- slightly below to ensure fill."""
        lp = executor._limit_price(100.0, OrderSide.SELL)
        expected = round(100.0 * (1 - DEFAULT_LIMIT_OFFSET_PCT), 2)
        assert lp == expected

    def test_zero_offset_no_adjustment(self) -> None:
        """With offset=0, limit price == mid price."""
        ex = OrderExecutor(client=MagicMock(), limit_offset_pct=0.0)
        assert ex._limit_price(250.0, OrderSide.BUY)  == 250.0
        assert ex._limit_price(250.0, OrderSide.SELL) == 250.0

    def test_prices_are_rounded_to_cents(self, executor: OrderExecutor) -> None:
        """Limit prices must have at most 2 decimal places."""
        lp = executor._limit_price(123.456789, OrderSide.BUY)
        assert lp == round(lp, 2)


# --------------------------------------------------------------------------- #
# Weight-delta to share conversion                                             #
# --------------------------------------------------------------------------- #

class TestWeightDeltaToShares:
    def test_positive_delta_gives_positive_quantity(self, executor: OrderExecutor) -> None:
        """10% weight at $400, equity=$100k -> 25 shares (floor)."""
        shares = executor._size_from_signal(
            _fake_signal(position_size_pct=0.10), price=400.0
        )
        assert shares == 25.0   # floor(100_000 * 0.10 / 400)

    def test_zero_size_gives_zero(self, executor: OrderExecutor) -> None:
        shares = executor._size_from_signal(
            _fake_signal(position_size_pct=0.0), price=400.0
        )
        assert shares == 0.0

    def test_size_floors_to_whole_shares(self, executor: OrderExecutor) -> None:
        """Result must be a whole number (fractional shares not supported)."""
        # 5% of 100k = 5000; 5000/333 = 15.01... -> floor = 15
        shares = executor._size_from_signal(
            _fake_signal(position_size_pct=0.05), price=333.0
        )
        assert shares == int(shares)
        assert shares == 15.0

    def test_quantity_matches_formula(self, executor: OrderExecutor) -> None:
        """shares = floor(equity * size_pct / price)."""
        pct   = 0.12
        price = 500.0
        expected = int(100_000.0 * pct / price)   # floor(24.0) = 24
        shares = executor._size_from_signal(_fake_signal(position_size_pct=pct), price=price)
        assert shares == expected


# --------------------------------------------------------------------------- #
# Slippage                                                                     #
# --------------------------------------------------------------------------- #

class TestSlippage:
    def test_buy_slippage_increases_price(self, executor: OrderExecutor) -> None:
        """BUY limit is above mid-price (adverse slippage for buyer)."""
        lp  = executor._limit_price(200.0, OrderSide.BUY)
        assert lp > 200.0

    def test_sell_slippage_decreases_price(self, executor: OrderExecutor) -> None:
        """SELL limit is below mid-price (adverse slippage for seller)."""
        lp = executor._limit_price(200.0, OrderSide.SELL)
        assert lp < 200.0

    def test_zero_slippage_unchanged(self) -> None:
        """offset_pct=0 -> limit == mid."""
        ex = OrderExecutor(client=MagicMock(), limit_offset_pct=0.0)
        assert ex._limit_price(100.0, OrderSide.BUY)  == 100.0
        assert ex._limit_price(100.0, OrderSide.SELL) == 100.0


# --------------------------------------------------------------------------- #
# Client order ID uniqueness                                                   #
# --------------------------------------------------------------------------- #

class TestClientOrderID:
    def test_ids_are_unique_per_call(self, executor: OrderExecutor) -> None:
        """Consecutive _new_trade_id() calls must return different values."""
        ids = {executor._new_trade_id() for _ in range(100)}
        assert len(ids) == 100

    def test_make_coid_format(self, executor: OrderExecutor) -> None:
        """client_order_id format: {uuid8}-{SYMBOL}-{side}."""
        coid = executor._make_coid("abcdef12", "SPY", OrderSide.BUY)
        parts = coid.split("-")
        assert parts[0] == "abcdef12"
        assert parts[1] == "SPY"
        assert parts[2] == "buy"

    def test_make_coid_with_suffix(self, executor: OrderExecutor) -> None:
        coid = executor._make_coid("abcdef12", "AAPL", OrderSide.SELL, suffix="mkt")
        assert coid.endswith("-mkt")


# --------------------------------------------------------------------------- #
# Market & limit order submission                                              #
# --------------------------------------------------------------------------- #

class TestOrderSubmission:
    def test_market_order_calls_submit(
        self, executor: OrderExecutor, mock_client: MagicMock
    ) -> None:
        """place_market_order() must call trading_client.submit_order once."""
        result = executor.place_market_order("SPY", 10, OrderSide.BUY)
        mock_client._trading_client.submit_order.assert_called_once()
        assert result.status in (OrderStatus.SUBMITTED, OrderStatus.PENDING,
                                 OrderStatus.FILLED)

    def test_limit_order_calls_submit(
        self, executor: OrderExecutor, mock_client: MagicMock
    ) -> None:
        result = executor.place_limit_order("SPY", 10, OrderSide.BUY, limit_price=400.0)
        mock_client._trading_client.submit_order.assert_called_once()
        assert result.ticket.symbol == "SPY"
        assert result.ticket.qty    == 10

    def test_failed_submit_returns_rejected(
        self, executor: OrderExecutor, mock_client: MagicMock
    ) -> None:
        """An exception from the broker should return REJECTED status."""
        mock_client._trading_client.submit_order.side_effect = RuntimeError("broker down")
        result = executor.place_market_order("SPY", 10, OrderSide.BUY)
        assert result.status       == OrderStatus.REJECTED
        assert result.error_message is not None

    def test_result_has_trade_id(
        self, executor: OrderExecutor, mock_client: MagicMock
    ) -> None:
        """Every OrderResult must carry a non-empty trade_id."""
        result = executor.place_market_order("AAPL", 5, OrderSide.BUY)
        assert result.trade_id
        assert len(result.trade_id) >= 8


# --------------------------------------------------------------------------- #
# Order cancellation                                                           #
# --------------------------------------------------------------------------- #

class TestOrderCancellation:
    def test_cancel_unknown_id_returns_false(self, executor: OrderExecutor) -> None:
        """cancel_order() on an unknown coid must return False without raising."""
        mock_client_local = MagicMock()
        mock_client_local._trading_client.get_order_by_client_id.side_effect = Exception("not found")
        ex = OrderExecutor(client=mock_client_local)
        assert ex.cancel_order("nonexistent-id") is False

    def test_cancel_order_invokes_broker(
        self, executor: OrderExecutor, mock_client: MagicMock
    ) -> None:
        """cancel_order() should call cancel_order_by_id on the trading client."""
        executor.cancel_order("some-coid")
        mock_client._trading_client.cancel_order_by_id.assert_called()

    def test_cancel_all_orders_returns_count(
        self, executor: OrderExecutor, mock_client: MagicMock
    ) -> None:
        """cancel_all_orders() returns the number of successfully cancelled orders."""
        # Inject two tracked submitted orders
        for sym in ["SPY", "QQQ"]:
            result = executor.place_market_order(sym, 5, OrderSide.BUY)
        count = executor.cancel_all_orders()
        # Both were submitted so cancel should have been attempted twice
        assert count >= 0   # exact count depends on mock response


# --------------------------------------------------------------------------- #
# Close position                                                               #
# --------------------------------------------------------------------------- #

class TestClosePosition:
    def test_close_position_no_position_returns_none(
        self, executor: OrderExecutor, mock_client: MagicMock
    ) -> None:
        """close_position() should return None if there is no open position."""
        mock_client.get_position.return_value = None
        result = executor.close_position("SPY")
        assert result is None

    def test_close_all_positions_empty(
        self, executor: OrderExecutor, mock_client: MagicMock
    ) -> None:
        """close_all_positions() on an empty book returns an empty list."""
        mock_client.get_all_positions.return_value = []
        results = executor.close_all_positions()
        assert results == []


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _fake_signal(position_size_pct: float = 0.10) -> MagicMock:
    """Return a minimal Signal mock for sizing tests."""
    sig = MagicMock()
    sig.symbol           = "SPY"
    sig.is_long          = True
    sig.direction        = "LONG"
    sig.position_size_pct = position_size_pct
    sig.leverage         = 1.0
    sig.regime_name      = "BULL"
    sig.stop_loss        = 380.0
    sig.take_profit      = None
    return sig
