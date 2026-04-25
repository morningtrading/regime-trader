"""
position_tracker.py -- Real-time position and P&L tracking.

Responsibilities:
  - WebSocket subscription for instant fill notifications via TradingStream
  - Update PortfolioState and CircuitBreaker on every fill
  - Per-position tracking: entry time/price, current price, unrealized P&L,
    stop level, holding period, regime at entry vs current
  - Sync with Alpaca on startup (reconcile tracked vs actual positions)
"""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
import sys
import threading
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import pandas as pd

from broker.alpaca_client import AlpacaClient
from core.risk_manager import PortfolioState, RiskManager

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Data structures                                                              #
# --------------------------------------------------------------------------- #

@dataclass
class PositionSnapshot:
    """Point-in-time view of a single open position."""
    symbol:              str
    qty:                 float
    avg_entry_price:     float
    current_price:       float
    market_value:        float
    unrealized_pnl:      float
    unrealized_pnl_pct:  float
    weight:              float       # fraction of total portfolio equity
    stop_level:          Optional[float] = None
    entry_time:          Optional[dt.datetime] = None
    holding_days:        float = 0.0
    regime_at_entry:     str = "UNKNOWN"
    current_regime:      str = "UNKNOWN"


@dataclass
class PortfolioSnapshot:
    """Aggregated portfolio state at a given timestamp."""
    timestamp:               dt.datetime
    total_equity:            float
    cash:                    float
    positions:               List[PositionSnapshot]
    total_unrealized_pnl:    float
    total_unrealized_pnl_pct: float
    weights:                 Dict[str, float]     # symbol -> weight
    gross_exposure:          float
    drawdown_from_peak:      float


@dataclass
class PositionMeta:
    """Internal per-position metadata not available from Alpaca."""
    symbol:          str
    entry_time:      dt.datetime
    entry_price:     float
    stop_level:      Optional[float] = None
    regime_at_entry: str = "UNKNOWN"


@dataclass
class FillEvent:
    """Parsed fill notification from TradingStream."""
    trade_id:     Optional[str]
    symbol:       str
    side:         str           # "buy" | "sell"
    qty:          float
    fill_price:   float
    timestamp:    dt.datetime
    order_id:     str
    raw:          dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# PositionTracker                                                              #
# --------------------------------------------------------------------------- #

class PositionTracker:
    """
    Maintains an up-to-date view of open positions, P&L, and portfolio weights.

    On startup: reconciles Alpaca's actual positions against any locally
    tracked metadata (entry time, stop level, regime at entry).

    WebSocket: TradingStream delivers fill events in real time.  Each fill
    triggers update_equity() on the RiskManager and calls any registered
    on_fill callbacks.

    Parameters
    ----------
    client :
        Connected AlpacaClient.
    risk_manager :
        Optional RiskManager -- updated on every fill.
    current_regime_fn :
        Optional zero-arg callable that returns the current HMM regime label.
        Used to populate current_regime on each PositionSnapshot.
    """

    def __init__(
        self,
        client:             AlpacaClient,
        risk_manager:       Optional[RiskManager]    = None,
        current_regime_fn:  Optional[Callable[[], str]] = None,
    ) -> None:
        self.client            = client
        self.risk_manager      = risk_manager
        self.current_regime_fn = current_regime_fn

        self._positions:     Dict[str, PositionSnapshot] = {}
        self._meta:          Dict[str, PositionMeta]     = {}   # internal metadata
        self._equity_history: List[float]                = []
        self._peak_equity:    float                      = 0.0
        self._last_snapshot:  Optional[PortfolioSnapshot] = None

        self._fill_callbacks: List[Callable[[FillEvent], None]] = []

        self._stream_thread: Optional[threading.Thread] = None
        self._stream_loop:   Optional[asyncio.AbstractEventLoop] = None
        self._stream_running: bool = False

    # ======================================================================= #
    # Startup & reconciliation                                                 #
    # ======================================================================= #

    def startup_sync(self, current_regime: str = "UNKNOWN") -> PortfolioSnapshot:
        """
        Fetch all open positions from Alpaca and build initial state.

        Call this once after AlpacaClient.connect() and before the main loop.

        Parameters
        ----------
        current_regime : HMM regime label at startup (for logging).
        """
        logger.info("PositionTracker: syncing with Alpaca ...")
        positions = self.client.get_all_positions()
        account   = self.client.get_account()
        equity    = float(account.equity)
        cash      = float(account.cash)

        self._peak_equity = equity
        self._positions.clear()

        for pos in positions:
            sym        = pos.symbol
            qty        = float(pos.qty)
            avg_entry  = float(pos.avg_entry_price)
            cur_price  = float(pos.current_price)
            mkt_value  = float(pos.market_value)
            unreal_pnl = float(pos.unrealized_pl)

            snap = PositionSnapshot(
                symbol             = sym,
                qty                = qty,
                avg_entry_price    = avg_entry,
                current_price      = cur_price,
                market_value       = mkt_value,
                unrealized_pnl     = unreal_pnl,
                unrealized_pnl_pct = unreal_pnl / max(abs(avg_entry * qty), 1e-9),
                weight             = mkt_value / max(equity, 1e-9),
                regime_at_entry    = current_regime,
                current_regime     = current_regime,
            )
            self._positions[sym] = snap

            # Seed meta if not already tracked (e.g. position opened before restart)
            if sym not in self._meta:
                self._meta[sym] = PositionMeta(
                    symbol          = sym,
                    entry_time      = dt.datetime.now(dt.timezone.utc),
                    entry_price     = avg_entry,
                    regime_at_entry = current_regime,
                )

        logger.info(
            "PositionTracker: synced %d position(s), equity=$%.2f",
            len(positions), equity,
        )

        # Reconcile: warn about any locally tracked positions not in Alpaca
        phantom = set(self._meta) - {p.symbol for p in positions}
        if phantom:
            logger.warning(
                "PositionTracker: locally tracked symbols not found in "
                "Alpaca positions (already closed?): %s", phantom,
            )
            for sym in phantom:
                self._meta.pop(sym, None)

        snapshot = self._build_portfolio_snapshot(equity, cash, current_regime)
        self._last_snapshot = snapshot
        return snapshot

    # ======================================================================= #
    # Real-time WebSocket                                                      #
    # ======================================================================= #

    def start_stream(self) -> None:
        """
        Start the TradingStream WebSocket in a background daemon thread.

        Delivers fill events to _on_trade_update() which then calls all
        registered on_fill callbacks and updates the RiskManager.
        """
        if self._stream_running:
            logger.warning("PositionTracker: stream already running")
            return

        self._stream_running = True
        self._stream_thread  = threading.Thread(
            target = self._stream_worker,
            daemon = True,
            name   = "trading-stream",
        )
        self._stream_thread.start()
        logger.info("PositionTracker: TradingStream started")

    def stop_stream(self) -> None:
        """Gracefully stop the TradingStream."""
        self._stream_running = False
        if self._stream_loop and self._stream_loop.is_running():
            self._stream_loop.call_soon_threadsafe(self._stream_loop.stop)
        if self._stream_thread:
            self._stream_thread.join(timeout=5.0)
        logger.info("PositionTracker: TradingStream stopped")

    def register_fill_callback(self, fn: Callable[[FillEvent], None]) -> None:
        """Register a callback to be called on every fill event."""
        self._fill_callbacks.append(fn)

    # ======================================================================= #
    # Bar-level update                                                         #
    # ======================================================================= #

    def update(self, current_regime: str = "UNKNOWN") -> PortfolioSnapshot:
        """
        Refresh positions from Alpaca REST and recompute P&L metrics.

        Call this at the end of every bar in the main trading loop.

        Returns
        -------
        PortfolioSnapshot representing the current portfolio state.
        """
        try:
            positions = self.client.get_all_positions()
            account   = self.client.get_account()
            equity    = float(account.equity)
            cash      = float(account.cash)
        except Exception as exc:
            logger.error("PositionTracker.update() failed: %s", exc)
            if self._last_snapshot:
                return self._last_snapshot
            raise

        self._update_peak_equity(equity)

        # Rebuild position snapshots
        new_positions: Dict[str, PositionSnapshot] = {}
        for pos in positions:
            sym       = pos.symbol
            qty       = float(pos.qty)
            avg_entry = float(pos.avg_entry_price)
            cur_price = float(pos.current_price)
            mkt_value = float(pos.market_value)
            unreal    = float(pos.unrealized_pl)
            meta      = self._meta.get(sym)

            holding_days = 0.0
            if meta and meta.entry_time:
                holding_days = (
                    dt.datetime.now(dt.timezone.utc) - meta.entry_time
                ).total_seconds() / 86_400.0

            snap = PositionSnapshot(
                symbol             = sym,
                qty                = qty,
                avg_entry_price    = avg_entry,
                current_price      = cur_price,
                market_value       = mkt_value,
                unrealized_pnl     = unreal,
                unrealized_pnl_pct = unreal / max(abs(avg_entry * qty), 1e-9),
                weight             = mkt_value / max(equity, 1e-9),
                stop_level         = meta.stop_level if meta else None,
                entry_time         = meta.entry_time if meta else None,
                holding_days       = holding_days,
                regime_at_entry    = meta.regime_at_entry if meta else "UNKNOWN",
                current_regime     = current_regime,
            )
            new_positions[sym] = snap

        self._positions = new_positions

        # Update RiskManager equity
        if self.risk_manager is not None:
            self.risk_manager.update_equity(
                equity    = equity,
                timestamp = dt.datetime.now(dt.timezone.utc),
                positions = {s: snap.market_value for s, snap in new_positions.items()},
                regime    = current_regime,
            )

        snapshot = self._build_portfolio_snapshot(equity, cash, current_regime)
        self._equity_history.append(equity)
        self._last_snapshot = snapshot
        return snapshot

    # ======================================================================= #
    # Accessors                                                                #
    # ======================================================================= #

    def get_current_weights(self) -> Dict[str, float]:
        """Return symbol -> weight from the last update."""
        return {s: snap.weight for s, snap in self._positions.items()}

    def get_portfolio_value(self) -> float:
        """Return total equity from the last snapshot."""
        return self._last_snapshot.total_equity if self._last_snapshot else 0.0

    def get_unrealized_pnl(self) -> float:
        """Return total unrealised P&L in USD from the last snapshot."""
        return self._last_snapshot.total_unrealized_pnl if self._last_snapshot else 0.0

    def get_drawdown_from_peak(self) -> float:
        """Return current peak-to-trough drawdown as a positive fraction."""
        if not self._last_snapshot or self._peak_equity <= 0:
            return 0.0
        return max(0.0, 1.0 - self._last_snapshot.total_equity / self._peak_equity)

    def get_position(self, symbol: str) -> Optional[PositionSnapshot]:
        """Return the snapshot for a specific symbol, or None."""
        return self._positions.get(symbol)

    def get_all_positions(self) -> List[PositionSnapshot]:
        """Return all open position snapshots."""
        return list(self._positions.values())

    def get_last_snapshot(self) -> Optional[PortfolioSnapshot]:
        """Return the most recently computed PortfolioSnapshot."""
        return self._last_snapshot

    def get_gross_exposure(self) -> float:
        """Return the sum of absolute position weights."""
        return sum(abs(snap.weight) for snap in self._positions.values())

    def set_stop_level(self, symbol: str, stop: float) -> None:
        """Record a stop level for a position (called by OrderExecutor)."""
        if symbol in self._meta:
            self._meta[symbol].stop_level = stop
        if symbol in self._positions:
            self._positions[symbol].stop_level = stop

    def record_entry(
        self,
        symbol:  str,
        price:   float,
        regime:  str = "UNKNOWN",
        stop:    Optional[float] = None,
    ) -> None:
        """Record entry metadata after a fill (called by on_fill callback)."""
        self._meta[symbol] = PositionMeta(
            symbol          = symbol,
            entry_time      = dt.datetime.now(dt.timezone.utc),
            entry_price     = price,
            stop_level      = stop,
            regime_at_entry = regime,
        )

    def to_portfolio_state(self) -> PortfolioState:
        """
        Convert the current tracker state into a PortfolioState for the
        RiskManager.validate_signal() call.
        """
        snap     = self._last_snapshot
        equity   = snap.total_equity if snap else 0.0
        cash     = snap.cash         if snap else 0.0
        regime   = (
            self.current_regime_fn() if self.current_regime_fn else "UNKNOWN"
        )
        return PortfolioState(
            equity          = equity,
            cash            = cash,
            buying_power    = self.client.get_buying_power() if self.client._connected else cash,
            positions       = {s: p.market_value for s, p in self._positions.items()},
            peak_equity     = self._peak_equity,
            current_regime  = regime,
        )

    # ======================================================================= #
    # Private helpers                                                          #
    # ======================================================================= #

    def _build_portfolio_snapshot(
        self,
        equity: float,
        cash:   float,
        regime: str,
    ) -> PortfolioSnapshot:
        positions  = list(self._positions.values())
        total_unrl = sum(p.unrealized_pnl for p in positions)
        gross_exp  = self._compute_gross_exposure(positions)
        dd         = self.get_drawdown_from_peak()

        return PortfolioSnapshot(
            timestamp               = dt.datetime.now(dt.timezone.utc),
            total_equity            = equity,
            cash                    = cash,
            positions               = positions,
            total_unrealized_pnl    = total_unrl,
            total_unrealized_pnl_pct= total_unrl / max(equity, 1e-9),
            weights                 = {p.symbol: p.weight for p in positions},
            gross_exposure          = gross_exp,
            drawdown_from_peak      = dd,
        )

    def _build_position_snapshot(
        self,
        symbol:       str,
        qty:          float,
        avg_entry:    float,
        current_price: float,
        total_equity: float,
    ) -> PositionSnapshot:
        mkt_value  = qty * current_price
        unreal_pnl = (current_price - avg_entry) * qty
        return PositionSnapshot(
            symbol             = symbol,
            qty                = qty,
            avg_entry_price    = avg_entry,
            current_price      = current_price,
            market_value       = mkt_value,
            unrealized_pnl     = unreal_pnl,
            unrealized_pnl_pct = unreal_pnl / max(abs(avg_entry * qty), 1e-9),
            weight             = mkt_value / max(total_equity, 1e-9),
        )

    def _update_peak_equity(self, current_equity: float) -> None:
        if current_equity > self._peak_equity:
            self._peak_equity = current_equity

    def _compute_gross_exposure(self, positions: List[PositionSnapshot]) -> float:
        return sum(abs(p.weight) for p in positions)

    # -- WebSocket internals ------------------------------------------------ #

    def _stream_worker(self) -> None:
        """Run the TradingStream event loop in a background thread."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._stream_loop = loop

        # ── Suppress harmless websockets shutdown noise ──────────────────────
        # The websockets legacy library spawns a close_connection coroutine as
        # a side-effect of cancellation.  It lands in the garbage collector
        # after the event loop is already closed, producing two benign but
        # noisy messages: "Task was destroyed but it is pending!" and
        # "Exception ignored in: close_connection / no running event loop".
        # Neither indicates data loss; both are suppressed here.

        def _exception_handler(lp: asyncio.AbstractEventLoop, ctx: dict) -> None:
            if "Task was destroyed but it is pending" in ctx.get("message", ""):
                return
            lp.default_exception_handler(ctx)

        loop.set_exception_handler(_exception_handler)

        orig_unraisable = sys.unraisablehook

        def _unraisable_hook(unraisable: sys.UnraisableHookArgs) -> None:
            if isinstance(unraisable.exc_value, RuntimeError) and \
                    "no running event loop" in str(unraisable.exc_value):
                return
            orig_unraisable(unraisable)

        sys.unraisablehook = _unraisable_hook
        # ─────────────────────────────────────────────────────────────────────

        try:
            loop.run_until_complete(self._run_stream())
        except Exception as exc:
            logger.error("TradingStream worker exited with error: %s", exc)
        finally:
            sys.unraisablehook = orig_unraisable
            try:
                pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
                if pending:
                    for t in pending:
                        t.cancel()
                    loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
            except Exception:
                pass
            loop.close()

    async def _run_stream(self) -> None:
        """Async coroutine: connect TradingStream and process events."""
        from alpaca.trading.stream import TradingStream

        stream = TradingStream(
            api_key    = self.client._api_key,
            secret_key = self.client._secret_key,
            paper      = self.client.paper,
        )

        async def on_trade_update(data):
            await self._on_trade_update(data)

        stream.subscribe_trade_updates(on_trade_update)

        logger.info("TradingStream: connecting ...")
        await stream._run_forever()

    async def _on_trade_update(self, data) -> None:
        """
        Handle a trade update event from TradingStream.

        Fires on: new, partial_fill, fill, canceled, expired, replaced.
        On fill: record entry metadata, call fill callbacks, update RiskManager.
        """
        try:
            event  = str(getattr(data, "event", ""))
            order  = getattr(data, "order", None)
            if order is None:
                return

            symbol     = str(order.symbol)
            side_str   = str(getattr(order, "side", "buy")).lower()
            fill_price = float(getattr(order, "filled_avg_price", 0) or 0)
            filled_qty = float(getattr(order, "filled_qty", 0) or 0)
            order_id   = str(getattr(order, "id", ""))
            coid       = str(getattr(order, "client_order_id", ""))
            ts         = dt.datetime.now(dt.timezone.utc)

            # Parse trade_id from client_order_id (format: {uuid8}-{symbol}-{side}[-suffix])
            trade_id: Optional[str] = None
            if coid:
                parts = coid.split("-")
                if len(parts) >= 1:
                    trade_id = parts[0]

            fill = FillEvent(
                trade_id    = trade_id,
                symbol      = symbol,
                side        = side_str,
                qty         = filled_qty,
                fill_price  = fill_price,
                timestamp   = ts,
                order_id    = order_id,
                raw         = {},
            )

            if event == "fill" and fill_price > 0:
                logger.info(
                    "FILL: %s %s %.0f @ $%.2f  trade_id=%s",
                    side_str, symbol, filled_qty, fill_price, trade_id,
                )

                # Record entry if buying
                if side_str == "buy":
                    regime = (
                        self.current_regime_fn()
                        if self.current_regime_fn else "UNKNOWN"
                    )
                    self.record_entry(symbol, fill_price, regime=regime)

                # Remove meta on full close
                if side_str == "sell":
                    # Check if position is now flat
                    existing = self._positions.get(symbol)
                    if existing and abs(existing.qty - filled_qty) < 0.01:
                        self._meta.pop(symbol, None)

                # Fire user callbacks
                for cb in self._fill_callbacks:
                    try:
                        cb(fill)
                    except Exception as cb_exc:
                        logger.error("Fill callback error: %s", cb_exc)

                # Update RiskManager
                if self.risk_manager is not None:
                    try:
                        equity = self.client.get_portfolio_value()
                        self.risk_manager.update_equity(
                            equity    = equity,
                            timestamp = ts,
                            regime    = (
                                self.current_regime_fn()
                                if self.current_regime_fn else "UNKNOWN"
                            ),
                        )
                        self.risk_manager.increment_trade_count()
                    except Exception as rm_exc:
                        logger.error("RiskManager update after fill failed: %s", rm_exc)

        except Exception as exc:
            logger.error("_on_trade_update error: %s", exc)
