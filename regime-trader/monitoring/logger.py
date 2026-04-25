"""
logger.py -- Structured, rotating trade and system event logger.

Four domain-specific rotating log files (10 MB / 30 backups each):
  main.log    — all events (catch-all)
  trades.log  — order submissions and fills
  alerts.log  — risk events and errors
  regime.log  — regime transitions and rebalances

Every JSON record includes a ``ctx`` sub-object with the current portfolio
context (regime, probability, equity, positions, daily_pnl), updated via
set_context().  This means every log line is self-contained for replay.

Log record format (one JSON object per line):
  {
    "ts":    "2024-01-15T10:32:01.123Z",
    "level": "INFO",
    "event": "trade",
    "ctx":   {"regime": "BULL", "probability": 0.72, "equity": 105230.0,
               "positions": ["SPY", "QQQ"], "daily_pnl": 340.0},
    ... event-specific fields ...
  }
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import logging.handlers
import threading
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional


# ── Event routing sets ────────────────────────────────────────────────────────
# Events routed to the specialised files; everything goes to main.log as well.
_TRADES_EVENTS = {"trade", "fill"}
_ALERTS_EVENTS = {"risk_event", "error"}
_REGIME_EVENTS = {"regime_change", "rebalance"}


# ── JSON formatter ────────────────────────────────────────────────────────────

class _JsonFormatter(logging.Formatter):
    """Emit each record as a single compact JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": (
                dt.datetime.utcfromtimestamp(record.created)
                .strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
            ),
            "level":   record.levelname,
            "logger":  record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "extra_fields"):
            payload.update(record.extra_fields)
        if record.exc_info:
            payload["traceback"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


# ── Domain-event filter ───────────────────────────────────────────────────────

class _EventFilter(logging.Filter):
    """Accept only log records whose ``event`` field is in *accepted*."""

    def __init__(self, accepted: set) -> None:
        super().__init__()
        self._accepted = accepted

    def filter(self, record: logging.LogRecord) -> bool:
        event = getattr(record, "extra_fields", {}).get("event", "")
        return event in self._accepted


# ── Handler factory ───────────────────────────────────────────────────────────

def _make_rotating_handler(
    path:         str,
    max_bytes:    int,
    backup_count: int,
    level:        int,
    event_filter: Optional[_EventFilter] = None,
) -> logging.handlers.RotatingFileHandler:
    h = logging.handlers.RotatingFileHandler(
        filename    = path,
        maxBytes    = max_bytes,
        backupCount = backup_count,
        encoding    = "utf-8",
    )
    h.setLevel(level)
    h.setFormatter(_JsonFormatter())
    if event_filter is not None:
        h.addFilter(event_filter)
    return h


# ── TradeLogger ───────────────────────────────────────────────────────────────

class TradeLogger:
    """
    Structured logger for regime-trader events.

    Writes JSON log lines to four rotating files:

      main.log    — every event (always written, no filter)
      trades.log  — trade and fill events only
      alerts.log  — risk_event and error events only
      regime.log  — regime_change and rebalance events only

    All files rotate at *max_bytes* (default 10 MB) and keep *backup_count*
    (default 30) rotated copies, giving roughly 30 days of history at typical
    log volumes.

    A portfolio context dict (regime, probability, equity, positions,
    daily_pnl) is injected into every record via set_context().

    Parameters
    ----------
    log_dir      : Directory where the four log files are written.
    log_level    : Minimum level: "DEBUG", "INFO", "WARNING", "ERROR".
    max_bytes    : Rotation size per file (default 10 MB).
    backup_count : Rotated copies to retain (default 30).
    console      : If True, also emit human-readable lines to stdout.
    """

    def __init__(
        self,
        log_dir:      str  = "logs/",
        log_level:    str  = "INFO",
        max_bytes:    int  = 10 * 1024 * 1024,   # 10 MB
        backup_count: int  = 30,
        console:      bool = True,
    ) -> None:
        self.log_dir      = log_dir
        self.log_level    = log_level
        self.max_bytes    = max_bytes
        self.backup_count = backup_count
        self.console      = console

        self._logger: Optional[logging.Logger] = None
        self._lock   = threading.Lock()
        self._context: Dict[str, Any] = {
            "regime":      "UNKNOWN",
            "probability": 0.0,
            "equity":      0.0,
            "positions":   [],
            "daily_pnl":   0.0,
        }

    # ======================================================================= #
    # Initialisation                                                           #
    # ======================================================================= #

    def setup(self) -> None:
        """
        Create handlers and open log files.  Call once at process startup.
        Idempotent: safe to call again (clears existing handlers first).
        """
        log_dir     = Path(self.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        numeric_lvl = getattr(logging, self.log_level.upper(), logging.INFO)

        self._logger = logging.getLogger("regime_trader")
        self._logger.setLevel(numeric_lvl)
        self._logger.propagate = False
        self._logger.handlers.clear()

        def _path(name: str) -> str:
            return str(log_dir / name)

        # main.log — catch-all (no event filter)
        self._logger.addHandler(_make_rotating_handler(
            _path("main.log"), self.max_bytes, self.backup_count, numeric_lvl,
        ))

        # trades.log — order submissions and fills
        self._logger.addHandler(_make_rotating_handler(
            _path("trades.log"), self.max_bytes, self.backup_count, numeric_lvl,
            _EventFilter(_TRADES_EVENTS),
        ))

        # alerts.log — risk events and exceptions
        self._logger.addHandler(_make_rotating_handler(
            _path("alerts.log"), self.max_bytes, self.backup_count, numeric_lvl,
            _EventFilter(_ALERTS_EVENTS),
        ))

        # regime.log — regime transitions and rebalances
        self._logger.addHandler(_make_rotating_handler(
            _path("regime.log"), self.max_bytes, self.backup_count, numeric_lvl,
            _EventFilter(_REGIME_EVENTS),
        ))

        # Optional human-readable console handler
        if self.console:
            ch = logging.StreamHandler()
            ch.setLevel(numeric_lvl)
            ch.setFormatter(logging.Formatter(
                "%(asctime)s  %(levelname)-8s  %(message)s",
                datefmt="%H:%M:%S",
            ))
            self._logger.addHandler(ch)

        self._logger.info(
            "TradeLogger ready  log_dir=%s  files=main,trades,alerts,regime",
            str(log_dir),
        )

    # ======================================================================= #
    # Context injection                                                        #
    # ======================================================================= #

    def set_context(
        self,
        regime:      str                  = "UNKNOWN",
        probability: float                = 0.0,
        equity:      float                = 0.0,
        positions:   Optional[List[str]]  = None,
        daily_pnl:   float                = 0.0,
    ) -> None:
        """
        Update the portfolio context injected into every subsequent log record.

        Call this whenever the regime changes or the portfolio is revalued.
        Thread-safe.
        """
        ctx = {
            "regime":      regime,
            "probability": round(probability, 4),
            "equity":      round(equity, 2),
            "positions":   list(positions) if positions else [],
            "daily_pnl":   round(daily_pnl, 2),
        }
        with self._lock:
            self._context = ctx

    # ======================================================================= #
    # Domain-specific helpers                                                  #
    # ======================================================================= #

    def log_trade(
        self,
        symbol:   str,
        side:     str,
        qty:      float,
        price:    float,
        order_id: Optional[str]            = None,
        extra:    Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log an order submission → trades.log + main.log."""
        self._emit(logging.INFO, "trade", {
            "symbol": symbol, "side": side,
            "qty": qty, "price": price,
            "order_id": order_id,
            **(extra or {}),
        })

    def log_fill(
        self,
        symbol:     str,
        side:       str,
        qty:        float,
        fill_price: float,
        order_id:   str,
        extra:      Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a confirmed order fill → trades.log + main.log."""
        self._emit(logging.INFO, "fill", {
            "symbol": symbol, "side": side,
            "qty": qty, "fill_price": fill_price,
            "order_id": order_id,
            **(extra or {}),
        })

    def log_regime_change(
        self,
        previous_regime: str,
        new_regime:      str,
        confidence:      float,
        timestamp:       Optional[dt.datetime]    = None,
        extra:           Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a regime transition → regime.log + main.log."""
        self._emit(logging.INFO, "regime_change", {
            "previous_regime": previous_regime,
            "new_regime":      new_regime,
            "confidence":      round(confidence, 4),
            "event_ts":        (timestamp or dt.datetime.utcnow()).isoformat(),
            **(extra or {}),
        })

    def log_rebalance(
        self,
        target_weights:   Dict[str, float],
        previous_weights: Dict[str, float],
        regime:           str,
        extra:            Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a portfolio rebalance → regime.log + main.log."""
        all_syms = set(target_weights) | set(previous_weights)
        deltas   = {
            s: round(target_weights.get(s, 0.0) - previous_weights.get(s, 0.0), 4)
            for s in all_syms
        }
        self._emit(logging.INFO, "rebalance", {
            "regime":           regime,
            "target_weights":   {k: round(v, 4) for k, v in target_weights.items()},
            "previous_weights": {k: round(v, 4) for k, v in previous_weights.items()},
            "deltas":           deltas,
            **(extra or {}),
        })

    def log_risk_event(
        self,
        event_type:  str,
        description: str,
        severity:    str                      = "WARNING",
        extra:       Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a risk-management event → alerts.log + main.log."""
        level = getattr(logging, severity.upper(), logging.WARNING)
        self._emit(level, "risk_event", {
            "event_type":  event_type,
            "description": description,
            **(extra or {}),
        })

    def log_error(
        self,
        error:   Exception,
        context: Optional[str]            = None,
        extra:   Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log an exception with stack trace → alerts.log + main.log."""
        self._emit(logging.ERROR, "error", {
            "error_type": type(error).__name__,
            "error_msg":  str(error),
            "context":    context,
            "traceback":  traceback.format_exc(),
            **(extra or {}),
        })

    # ======================================================================= #
    # Generic passthrough                                                      #
    # ======================================================================= #

    def info(self, message: str, **kwargs: Any) -> None:
        self._emit(logging.INFO,    "info",    {"message": message, **kwargs})

    def warning(self, message: str, **kwargs: Any) -> None:
        self._emit(logging.WARNING, "warning", {"message": message, **kwargs})

    def error(self, message: str, **kwargs: Any) -> None:
        self._emit(logging.ERROR,   "error",   {"message": message, **kwargs})

    def debug(self, message: str, **kwargs: Any) -> None:
        self._emit(logging.DEBUG,   "debug",   {"message": message, **kwargs})

    # ======================================================================= #
    # Private helpers                                                          #
    # ======================================================================= #

    def _emit(self, level: int, event: str, payload: Dict[str, Any]) -> None:
        """
        Build a LogRecord with the event payload + current portfolio context
        and pass it to every registered handler.
        Falls back to the root logger if setup() has not been called.
        """
        logger = self._logger or logging.getLogger("regime_trader")

        with self._lock:
            ctx = dict(self._context)

        record = logger.makeRecord(
            name     = logger.name,
            level    = level,
            fn       = "",
            lno      = 0,
            msg      = f"[{event}] " + str(payload.get("message", "")),
            args     = (),
            exc_info = None,
        )
        # Attach both the context snapshot and the event payload so the JSON
        # formatter can serialise everything into a single flat-ish record.
        record.extra_fields = {   # type: ignore[attr-defined]
            "event": event,
            "ctx":   ctx,
            **payload,
        }
        logger.handle(record)
