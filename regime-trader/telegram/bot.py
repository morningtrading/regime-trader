"""
telegram/bot.py — Minimal Telegram Bot API client.

Single public function: send(text) → bool
"""
from __future__ import annotations
import logging

logger = logging.getLogger(__name__)

_API_BASE = "https://api.telegram.org/bot{token}/sendMessage"


def send(text: str, parse_mode: str = "HTML") -> bool:
    """
    Send a text message to the configured chat.

    Parameters
    ----------
    text       : Message body (HTML tags supported: <b>, <i>, <code>)
    parse_mode : "HTML" (default) or "Markdown"

    Returns
    -------
    True if delivered, False on any error (never raises).
    """
    from telegram.config import TOKEN, CHAT_ID, validate
    try:
        validate()
    except EnvironmentError as exc:
        logger.error("Telegram not configured: %s", exc)
        return False

    try:
        import requests
        url = _API_BASE.format(token=TOKEN)
        resp = requests.post(url, data={
            "chat_id":    CHAT_ID,
            "text":       text,
            "parse_mode": parse_mode,
        }, timeout=10)
        if not resp.ok:
            logger.error("Telegram API error %s: %s", resp.status_code, resp.text[:200])
            return False
        return True
    except Exception as exc:
        logger.error("Telegram send failed: %s", exc)
        return False


def send_silent(text: str) -> bool:
    """Same as send() but with no notification sound."""
    from telegram.config import TOKEN, CHAT_ID, validate
    try:
        validate()
    except EnvironmentError:
        return False
    try:
        import requests
        resp = requests.post(
            _API_BASE.format(token=TOKEN),
            data={
                "chat_id":              CHAT_ID,
                "text":                 text,
                "parse_mode":           "HTML",
                "disable_notification": True,
            },
            timeout=10,
        )
        return resp.ok
    except Exception:
        return False
