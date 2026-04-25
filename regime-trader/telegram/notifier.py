"""
telegram/notifier.py — Central notification dispatcher for Regime Trader.

Single public function: notify(event, data)

Events:
    "backtest"      — end of walk-forward backtest
    "stress"        — end of stress test
    "regime_change" — HMM regime transition (live trading)
    "trade"         — trade executed (live trading)

Never raises. Fails silently with a log warning if Telegram is not
configured or disabled.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ── Config cache ──────────────────────────────────────────────────────────────

_cfg: Optional[Dict] = None          # loaded once, cached
_cli_override: Optional[bool] = None # set by main.py before first call


def configure(enabled: Optional[bool] = None) -> None:
    """
    Called once by main.py after parsing CLI args.

    Parameters
    ----------
    enabled : True  → force on  (--telegram)
              False → force off (--no-telegram)
              None  → use settings.yaml value
    """
    global _cli_override
    _cli_override = enabled


def _load_cfg() -> Dict:
    global _cfg
    if _cfg is not None:
        return _cfg
    try:
        from pathlib import Path
        import yaml
        settings = Path(__file__).resolve().parent.parent / "config" / "settings.yaml"
        if settings.exists():
            raw = yaml.safe_load(settings.read_text()) or {}
            _cfg = raw.get("notifications", {}).get("telegram", {})
        else:
            _cfg = {}
    except Exception:
        _cfg = {}
    return _cfg


def _is_enabled(event: str) -> bool:
    """Return True if notifications are active for this event."""
    # CLI override takes absolute priority
    if _cli_override is not None:
        return _cli_override

    cfg = _load_cfg()

    # Master switch
    if not cfg.get("enabled", False):
        return False

    # Per-event switch (default True if master is on)
    event_key = f"on_{event}"
    return cfg.get(event_key, True)


# ── Public API ─────────────────────────────────────────────────────────────────

def notify(event: str, data: Optional[Dict[str, Any]] = None) -> None:
    """
    Send a Telegram notification for the given event.

    Parameters
    ----------
    event : one of "backtest", "stress", "regime_change", "trade"
    data  : optional dict with event-specific fields (see below)

    data fields by event:
        backtest      — auto-reads from savedresults/ (no data needed)
        stress        — auto-reads from savedresults/ (no data needed)
        regime_change — {"from_regime": str, "to_regime": str,
                         "asset_group": str, "equity": float}
        trade         — {"symbol": str, "side": str, "pnl_pct": float,
                         "equity": float, "asset_group": str, "regime": str}
    """
    if not _is_enabled(event):
        return

    try:
        text = _build_message(event, data or {})
        if text:
            from telegram.bot import send
            ok = send(text)
            if not ok:
                logger.warning("Telegram notification failed for event '%s'", event)
    except Exception as exc:
        logger.warning("Telegram notifier error for event '%s': %s", event, exc)


def _build_message(event: str, data: Dict[str, Any]) -> str:
    if event == "backtest":
        from telegram.formatter import format_backtest_summary
        return format_backtest_summary()

    if event == "stress":
        from telegram.formatter import format_stress_summary
        return format_stress_summary()

    if event == "regime_change":
        from_r  = data.get("from_regime", "?")
        to_r    = data.get("to_regime",   "?")
        group   = data.get("asset_group", "—")
        equity  = data.get("equity")
        from telegram.formatter import _now, _HOST
        icons = {"BULL": "🟢", "EUPHORIA": "🚀", "BEAR": "🔴",
                 "CRASH": "💥", "NEUTRAL": "⚪"}
        icon   = icons.get(to_r.upper(), "📊")
        eq_str = f"  ${equity:,.0f}" if equity else ""
        return (
            f"{icon} <b>Régime: {from_r} → {to_r}</b>  {group}  <code>{_HOST}</code>\n"
            f"{eq_str}  ·  <i>{_now()}</i>"
        )

    if event == "trade":
        symbol  = data.get("symbol",     "?")
        side    = data.get("side",       "?")
        pnl_pct = data.get("pnl_pct")
        equity  = data.get("equity")
        group   = data.get("asset_group", "—")
        regime  = data.get("regime",      "?")
        from telegram.formatter import _now, _pct, _HOST
        icon    = "🟢" if (pnl_pct or 0) >= 0 else "🔴"
        pnl_str = f"  {_pct(pnl_pct)}" if pnl_pct is not None else ""
        eq_str  = f"  ${equity:,.0f}" if equity else ""
        return (
            f"{icon} <b>TRADE {symbol} {side.upper()}</b>{pnl_str}  {group}/{regime}  <code>{_HOST}</code>\n"
            f"{eq_str}  ·  <i>{_now()}</i>"
        )

    logger.warning("Unknown telegram event: '%s'", event)
    return ""
