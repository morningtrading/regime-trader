"""
order_executor.py -- Order placement, modification, and cancellation.

Converts portfolio signals into Alpaca orders with a full audit trail.
Each signal gets a unique trade_id that links: signal -> risk_decision ->
order submission -> fill notification.

Order flow:
  1. submit_order(signal)  -- LIMIT at mid +/- 0.1%, cancel after 30s,
                               optionally retry at market
  2. submit_bracket_order  -- entry + stop + take_profit via OCO
  3. modify_stop           -- only tighten, never widen
  4. cancel_order / close_position / close_all_positions
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Dict, List, Optional

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import (
    OrderClass,
    OrderSide,
    OrderType,
    QueryOrderStatus,
    TimeInForce,
)
from alpaca.trading.models import Order
from alpaca.trading.requests import (
    GetOrdersRequest,
    LimitOrderRequest,
    MarketOrderRequest,
    StopLossRequest,
    TakeProfitRequest,
)

from broker.alpaca_client import AlpacaClient
from core.risk_manager import RiskManager

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Constants                                                                    #
# --------------------------------------------------------------------------- #

DEFAULT_LIMIT_OFFSET_PCT = 0.001   # 0.1% from mid-price for limit orders
DEFAULT_CANCEL_AFTER_SEC = 30      # cancel unfilled limit after 30 seconds
MAX_ORDER_RETRIES        = 2


# --------------------------------------------------------------------------- #
# Enumerations                                                                 #
# --------------------------------------------------------------------------- #

class OrderStatus(Enum):
    """Unified order lifecycle states."""
    PENDING          = auto()
    SUBMITTED        = auto()
    FILLED           = auto()
    PARTIALLY_FILLED = auto()
    CANCELLED        = auto()
    REJECTED         = auto()
    EXPIRED          = auto()


_ALPACA_STATUS_MAP: Dict[str, OrderStatus] = {
    "new":              OrderStatus.SUBMITTED,
    "partially_filled": OrderStatus.PARTIALLY_FILLED,
    "filled":           OrderStatus.FILLED,
    "done_for_day":     OrderStatus.EXPIRED,
    "canceled":         OrderStatus.CANCELLED,
    "expired":          OrderStatus.EXPIRED,
    "replaced":         OrderStatus.CANCELLED,
    "pending_cancel":   OrderStatus.SUBMITTED,
    "pending_replace":  OrderStatus.SUBMITTED,
    "held":             OrderStatus.SUBMITTED,
    "accepted":         OrderStatus.SUBMITTED,
    "pending_new":      OrderStatus.PENDING,
    "accepted_for_bidding": OrderStatus.PENDING,
    "stopped":          OrderStatus.SUBMITTED,
    "rejected":         OrderStatus.REJECTED,
    "suspended":        OrderStatus.REJECTED,
    "calculated":       OrderStatus.SUBMITTED,
}


# --------------------------------------------------------------------------- #
# Data structures                                                              #
# --------------------------------------------------------------------------- #

@dataclass
class OrderTicket:
    """Internal representation of an order before submission."""
    trade_id:       str
    symbol:         str
    side:           OrderSide
    qty:            float
    order_type:     OrderType
    time_in_force:  TimeInForce
    limit_price:    Optional[float] = None
    stop_price:     Optional[float] = None
    take_profit:    Optional[float] = None
    client_order_id: Optional[str] = None
    signal_regime:  str             = "UNKNOWN"
    created_at:     datetime        = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class OrderResult:
    """Outcome of an order submission attempt."""
    ticket:         OrderTicket
    alpaca_order:   Optional[Order]
    status:         OrderStatus
    filled_qty:     float          = 0.0
    avg_fill_price: float          = 0.0
    error_message:  Optional[str]  = None

    @property
    def trade_id(self) -> str:
        return self.ticket.trade_id


# --------------------------------------------------------------------------- #
# OrderExecutor                                                                #
# --------------------------------------------------------------------------- #

class OrderExecutor:
    """
    Translates portfolio signals into Alpaca orders with full audit trail.

    Each call to submit_order() generates a unique trade_id that is embedded
    in the Alpaca client_order_id (format: "{trade_id}-{symbol}-{side}") so
    fills can be matched back to the originating signal.

    Parameters
    ----------
    client :
        Connected AlpacaClient.
    risk_manager :
        RiskManager for pre-flight size adjustment.
    limit_offset_pct :
        Offset from mid-price for limit orders (default 0.1%).
    cancel_after_sec :
        Seconds before an unfilled limit order is cancelled (default 30).
    retry_at_market :
        If True, retry a cancelled limit order at market price.
    """

    def __init__(
        self,
        client:            AlpacaClient,
        risk_manager:      Optional[RiskManager] = None,
        limit_offset_pct:  float = DEFAULT_LIMIT_OFFSET_PCT,
        cancel_after_sec:  int   = DEFAULT_CANCEL_AFTER_SEC,
        retry_at_market:   bool  = True,
    ) -> None:
        self.client            = client
        self.risk_manager      = risk_manager
        self.limit_offset_pct  = limit_offset_pct
        self.cancel_after_sec  = cancel_after_sec
        self.retry_at_market   = retry_at_market

        # trade_id -> OrderResult
        self._orders: Dict[str, OrderResult] = {}
        # client_order_id -> trade_id (reverse lookup)
        self._coid_to_trade: Dict[str, str] = {}
        self._lock = threading.Lock()

    # ======================================================================= #
    # Primary order submission                                                 #
    # ======================================================================= #

    def submit_order(
        self,
        signal,                             # Signal from regime_strategies
        current_price: Optional[float] = None,
    ) -> OrderResult:
        """
        Submit a LIMIT order for a signal.

        Limit price = mid +/- limit_offset_pct in the adverse direction.
        If the order remains unfilled after cancel_after_sec, it is cancelled
        and (if retry_at_market=True) retried at market.

        Parameters
        ----------
        signal        : Signal dataclass from regime_strategies.
        current_price : Override the price used for limit calculation.
                        Falls back to client.get_latest_price(signal.symbol).

        Returns
        -------
        OrderResult with the final fill status.
        """
        trade_id = self._new_trade_id()
        price    = current_price or self.client.get_latest_price(signal.symbol)
        side     = OrderSide.BUY if signal.is_long else OrderSide.SELL
        qty      = self._size_from_signal(signal, price)

        if qty <= 0:
            return self._null_result(trade_id, signal.symbol, side, "qty=0 after sizing")

        limit_px = self._limit_price(price, side)
        coid     = self._make_coid(trade_id, signal.symbol, side)

        ticket = OrderTicket(
            trade_id        = trade_id,
            symbol          = signal.symbol,
            side            = side,
            qty             = qty,
            order_type      = OrderType.LIMIT,
            time_in_force   = TimeInForce.DAY,
            limit_price     = limit_px,
            stop_price      = signal.stop_loss,
            client_order_id = coid,
            signal_regime   = signal.regime_name,
        )

        result = self._submit_limit(ticket)

        # Wait and check fill; cancel if still open after timeout
        if result.status == OrderStatus.SUBMITTED:
            filled = self._wait_for_fill(coid, self.cancel_after_sec)
            if not filled:
                self._cancel_by_coid(coid)
                if self.retry_at_market:
                    logger.info(
                        "Limit unfilled for %s -- retrying at market", signal.symbol
                    )
                    result = self.place_market_order(
                        signal.symbol, qty, side,
                        client_order_id=self._make_coid(trade_id, signal.symbol, side, suffix="mkt"),
                    )
                    result.ticket.trade_id = trade_id
                else:
                    result.status = OrderStatus.CANCELLED

        with self._lock:
            self._orders[trade_id] = result

        return result

    def submit_bracket_order(
        self,
        signal,
        current_price: Optional[float] = None,
    ) -> OrderResult:
        """
        Submit a bracket order: entry LIMIT + stop-loss + take-profit (OCO).

        The stop and take-profit legs are sent as a single bracket via Alpaca's
        order_class=bracket.  The entry uses the same limit-price logic as
        submit_order().

        Parameters
        ----------
        signal        : Signal dataclass (must have stop_loss and take_profit set).
        current_price : Override entry price.

        Returns
        -------
        OrderResult with the bracket order.
        """
        trade_id = self._new_trade_id()
        price    = current_price or self.client.get_latest_price(signal.symbol)
        side     = OrderSide.BUY if signal.is_long else OrderSide.SELL
        qty      = self._size_from_signal(signal, price)

        if qty <= 0:
            return self._null_result(trade_id, signal.symbol, side, "qty=0 after sizing")

        if signal.stop_loss is None or signal.stop_loss <= 0:
            logger.warning(
                "submit_bracket_order: no stop_loss on signal for %s -- "
                "falling back to submit_order", signal.symbol
            )
            return self.submit_order(signal, current_price=price)

        limit_px = self._limit_price(price, side)
        coid     = self._make_coid(trade_id, signal.symbol, side, suffix="brk")

        ticket = OrderTicket(
            trade_id        = trade_id,
            symbol          = signal.symbol,
            side            = side,
            qty             = qty,
            order_type      = OrderType.LIMIT,
            time_in_force   = TimeInForce.DAY,
            limit_price     = limit_px,
            stop_price      = signal.stop_loss,
            take_profit     = signal.take_profit,
            client_order_id = coid,
            signal_regime   = signal.regime_name,
        )

        req = LimitOrderRequest(
            symbol           = signal.symbol,
            qty              = qty,
            side             = side,
            time_in_force    = TimeInForce.DAY,
            limit_price      = round(limit_px, 2),
            order_class      = OrderClass.BRACKET,
            stop_loss        = StopLossRequest(stop_price=round(signal.stop_loss, 2)),
            take_profit      = (
                TakeProfitRequest(limit_price=round(signal.take_profit, 2))
                if signal.take_profit else None
            ),
            client_order_id  = coid,
        )

        result = self._execute_request(req, ticket)
        with self._lock:
            self._orders[trade_id] = result
        return result

    # ======================================================================= #
    # Simple order helpers                                                     #
    # ======================================================================= #

    def place_market_order(
        self,
        symbol:          str,
        qty:             float,
        side:            OrderSide,
        client_order_id: Optional[str] = None,
    ) -> OrderResult:
        """Submit a market order."""
        trade_id = self._new_trade_id()
        coid     = client_order_id or self._make_coid(trade_id, symbol, side)
        ticket   = OrderTicket(
            trade_id        = trade_id,
            symbol          = symbol,
            side            = side,
            qty             = qty,
            order_type      = OrderType.MARKET,
            time_in_force   = TimeInForce.DAY,
            client_order_id = coid,
        )
        req = MarketOrderRequest(
            symbol          = symbol,
            qty             = qty,
            side            = side,
            time_in_force   = TimeInForce.DAY,
            client_order_id = coid,
        )
        result = self._execute_request(req, ticket)
        with self._lock:
            self._orders[trade_id] = result
        return result

    def place_limit_order(
        self,
        symbol:          str,
        qty:             float,
        side:            OrderSide,
        limit_price:     float,
        time_in_force:   TimeInForce   = TimeInForce.DAY,
        client_order_id: Optional[str] = None,
    ) -> OrderResult:
        """Submit a limit order."""
        trade_id = self._new_trade_id()
        coid     = client_order_id or self._make_coid(trade_id, symbol, side)
        ticket   = OrderTicket(
            trade_id        = trade_id,
            symbol          = symbol,
            side            = side,
            qty             = qty,
            order_type      = OrderType.LIMIT,
            time_in_force   = time_in_force,
            limit_price     = limit_price,
            client_order_id = coid,
        )
        req = LimitOrderRequest(
            symbol          = symbol,
            qty             = qty,
            side            = side,
            time_in_force   = time_in_force,
            limit_price     = round(limit_price, 2),
            client_order_id = coid,
        )
        result = self._execute_request(req, ticket)
        with self._lock:
            self._orders[trade_id] = result
        return result

    # ======================================================================= #
    # Stop modification                                                        #
    # ======================================================================= #

    def modify_stop(self, symbol: str, new_stop: float) -> bool:
        """
        Tighten the stop-loss on an open position.

        ONLY TIGHTENS -- refuses to widen an existing stop.  Implemented by
        cancelling the existing stop leg and replacing it with a new one.

        Parameters
        ----------
        symbol   : Ticker of the open position.
        new_stop : New stop price (must be higher than current stop for longs).

        Returns
        -------
        True if the modification was accepted, False otherwise.
        """
        position = self.client.get_position(symbol)
        if position is None:
            logger.warning("modify_stop: no open position for %s", symbol)
            return False

        qty  = float(position.qty)
        side = OrderSide.BUY if qty > 0 else OrderSide.SELL

        # Get current stop from open orders
        current_stop = self._find_stop_price(symbol)
        if current_stop is not None:
            if qty > 0 and new_stop <= current_stop:
                logger.info(
                    "modify_stop rejected for %s: new_stop=%.2f <= current=%.2f "
                    "(would widen long stop)",
                    symbol, new_stop, current_stop,
                )
                return False
            if qty < 0 and new_stop >= current_stop:
                logger.info(
                    "modify_stop rejected for %s: new_stop=%.2f >= current=%.2f "
                    "(would widen short stop)",
                    symbol, new_stop, current_stop,
                )
                return False

        # Cancel existing stop orders for this symbol
        self._cancel_stops_for(symbol)

        # Place a new stop-market order on the opposite side (to close the position)
        close_side = OrderSide.SELL if qty > 0 else OrderSide.BUY
        from alpaca.trading.requests import StopOrderRequest
        req = StopOrderRequest(
            symbol        = symbol,
            qty           = abs(qty),
            side          = close_side,
            time_in_force = TimeInForce.GTC,
            stop_price    = round(new_stop, 2),
        )
        try:
            self.client._trading_client.submit_order(req)
            logger.info("modify_stop: %s stop moved to %.2f", symbol, new_stop)
            return True
        except Exception as exc:
            logger.error("modify_stop failed for %s: %s", symbol, exc)
            return False

    # ======================================================================= #
    # Cancellation & closure                                                   #
    # ======================================================================= #

    def cancel_order(self, client_order_id: str) -> bool:
        """
        Cancel an open order by client_order_id.

        Returns True if the cancel request was accepted.
        """
        return self._cancel_by_coid(client_order_id)

    def cancel_all_orders(self) -> int:
        """
        Cancel all open orders tracked by this executor.

        Returns the number of orders successfully cancelled.
        """
        cancelled = 0
        with self._lock:
            coids = [
                r.ticket.client_order_id
                for r in self._orders.values()
                if r.status in (OrderStatus.SUBMITTED, OrderStatus.PENDING,
                                OrderStatus.PARTIALLY_FILLED)
                and r.ticket.client_order_id
            ]
        for coid in coids:
            if self._cancel_by_coid(coid):
                cancelled += 1
        return cancelled

    def close_position(self, symbol: str) -> Optional[OrderResult]:
        """
        Market-close an entire open position.

        Returns the OrderResult, or None if no position exists.
        """
        position = self.client.get_position(symbol)
        if position is None:
            logger.info("close_position: no open position for %s", symbol)
            return None

        qty  = abs(float(position.qty))
        side = OrderSide.SELL if float(position.qty) > 0 else OrderSide.BUY
        logger.info("Closing position: %s %.0f shares (%s)", symbol, qty, side.value)
        return self.place_market_order(symbol, qty, side)

    def close_all_positions(self) -> List[OrderResult]:
        """
        Market-close every open position.

        Returns a list of OrderResult objects.
        """
        results: List[OrderResult] = []
        positions = self.client.get_all_positions()
        for pos in positions:
            r = self.close_position(pos.symbol)
            if r is not None:
                results.append(r)
        return results

    # ======================================================================= #
    # Status queries                                                           #
    # ======================================================================= #

    def get_order_status(self, client_order_id: str) -> Optional[OrderStatus]:
        """Poll the current status of a submitted order."""
        try:
            order = self.client._trading_client.get_order_by_client_id(client_order_id)
            return _ALPACA_STATUS_MAP.get(str(order.status), OrderStatus.SUBMITTED)
        except Exception:
            return None

    def get_result_by_trade_id(self, trade_id: str) -> Optional[OrderResult]:
        """Return the OrderResult for a given trade_id."""
        with self._lock:
            return self._orders.get(trade_id)

    def get_all_results(self) -> List[OrderResult]:
        """Return all OrderResult objects tracked so far."""
        with self._lock:
            return list(self._orders.values())

    # ======================================================================= #
    # Private helpers                                                          #
    # ======================================================================= #

    def _submit_limit(self, ticket: OrderTicket) -> OrderResult:
        """Build and submit a LimitOrderRequest from a ticket."""
        req = LimitOrderRequest(
            symbol          = ticket.symbol,
            qty             = ticket.qty,
            side            = ticket.side,
            time_in_force   = ticket.time_in_force,
            limit_price     = round(ticket.limit_price, 2),
            client_order_id = ticket.client_order_id,
        )
        return self._execute_request(req, ticket)

    def _execute_request(self, req, ticket: OrderTicket) -> OrderResult:
        """Submit an alpaca-py order request and return an OrderResult."""
        try:
            order = self.client._trading_client.submit_order(req)
            logger.info(
                "Order submitted | trade_id=%s | %s %s %.0f @ %s | coid=%s",
                ticket.trade_id,
                ticket.side.value,
                ticket.symbol,
                ticket.qty,
                f"${ticket.limit_price:.2f}" if ticket.limit_price else "MKT",
                ticket.client_order_id,
            )
            status = _ALPACA_STATUS_MAP.get(str(order.status), OrderStatus.SUBMITTED)
            with self._lock:
                if ticket.client_order_id:
                    self._coid_to_trade[ticket.client_order_id] = ticket.trade_id
            return OrderResult(
                ticket       = ticket,
                alpaca_order = order,
                status       = status,
            )
        except Exception as exc:
            logger.error(
                "Order submission failed | trade_id=%s | %s %s: %s",
                ticket.trade_id, ticket.side.value, ticket.symbol, exc,
            )
            return OrderResult(
                ticket        = ticket,
                alpaca_order  = None,
                status        = OrderStatus.REJECTED,
                error_message = str(exc),
            )

    def _wait_for_fill(self, coid: str, timeout_sec: int) -> bool:
        """
        Poll for a fill on a submitted order.

        Returns True if filled before timeout, False otherwise.
        """
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            status = self.get_order_status(coid)
            if status == OrderStatus.FILLED:
                return True
            if status in (OrderStatus.CANCELLED, OrderStatus.REJECTED,
                          OrderStatus.EXPIRED):
                return False
            time.sleep(1.0)
        return False

    def _cancel_by_coid(self, coid: str) -> bool:
        """Cancel a single order by client_order_id."""
        try:
            order = self.client._trading_client.get_order_by_client_id(coid)
            self.client._trading_client.cancel_order_by_id(str(order.id))
            logger.info("Order cancelled: coid=%s", coid)
            return True
        except Exception as exc:
            logger.warning("Cancel failed for coid=%s: %s", coid, exc)
            return False

    def _cancel_stops_for(self, symbol: str) -> None:
        """Cancel all open stop orders for a symbol."""
        try:
            req    = GetOrdersRequest(
                status  = QueryOrderStatus.OPEN,
                symbols = [symbol],
            )
            orders = self.client._trading_client.get_orders(req)
            for o in orders:
                if str(o.type) in ("stop", "stop_limit", "trailing_stop"):
                    try:
                        self.client._trading_client.cancel_order_by_id(str(o.id))
                    except Exception:
                        pass
        except Exception as exc:
            logger.warning("_cancel_stops_for %s: %s", symbol, exc)

    def _find_stop_price(self, symbol: str) -> Optional[float]:
        """Find the current stop price from open stop orders for symbol."""
        try:
            req    = GetOrdersRequest(
                status  = QueryOrderStatus.OPEN,
                symbols = [symbol],
            )
            orders = self.client._trading_client.get_orders(req)
            for o in orders:
                if str(o.type) in ("stop", "stop_limit") and o.stop_price:
                    return float(o.stop_price)
        except Exception as exc:
            logger.warning("Failed to fetch stop price for %s: %s", symbol, exc)
        return None

    def _size_from_signal(self, signal, price: float) -> float:
        """Convert signal.position_size_pct to whole shares."""
        try:
            equity = self.client.get_portfolio_value()
        except Exception as exc:
            logger.warning(
                "get_portfolio_value() failed, using $100k fallback for sizing: %s",
                exc,
            )
            equity = 100_000.0   # safe fallback
        shares = (equity * signal.position_size_pct) / max(price, 1e-9)
        return float(int(shares))   # floor to whole shares

    def _limit_price(self, price: float, side: OrderSide) -> float:
        """Return a limit price offset from mid in the adverse direction."""
        if side == OrderSide.BUY:
            return round(price * (1.0 + self.limit_offset_pct), 2)
        return round(price * (1.0 - self.limit_offset_pct), 2)

    @staticmethod
    def _new_trade_id() -> str:
        return str(uuid.uuid4())

    @staticmethod
    def _make_coid(
        trade_id: str,
        symbol:   str,
        side:     OrderSide,
        suffix:   str = "",
    ) -> str:
        """Format: {trade_id[:8]}-{symbol}-{side}[-suffix]"""
        parts = [trade_id[:8], symbol.upper(), side.value]
        if suffix:
            parts.append(suffix)
        return "-".join(parts)

    @staticmethod
    def _null_result(
        trade_id: str,
        symbol:   str,
        side:     OrderSide,
        reason:   str,
    ) -> OrderResult:
        ticket = OrderTicket(
            trade_id       = trade_id,
            symbol         = symbol,
            side           = side,
            qty            = 0.0,
            order_type     = OrderType.LIMIT,
            time_in_force  = TimeInForce.DAY,
        )
        return OrderResult(
            ticket        = ticket,
            alpaca_order  = None,
            status        = OrderStatus.REJECTED,
            error_message = reason,
        )
