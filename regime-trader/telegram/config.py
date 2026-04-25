"""
telegram/config.py — Load Telegram credentials from .env or credentials.yaml.

Add to your .env (never commit):
    TELEGRAM_TOKEN=<bot token from @BotFather>
    TELEGRAM_CHAT_ID=<your personal chat ID>
"""
from __future__ import annotations
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Try .env first
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

# Try credentials.yaml as fallback
def _from_yaml() -> tuple[str, str]:
    creds = ROOT / "config" / "credentials.yaml"
    if not creds.exists():
        return "", ""
    try:
        import yaml
        data = yaml.safe_load(creds.read_text())
        tg = data.get("telegram", {}) or {}
        return tg.get("token", ""), str(tg.get("chat_id", ""))
    except Exception:
        return "", ""

_yaml_token, _yaml_chat_id = _from_yaml()

TOKEN   = os.getenv("TELEGRAM_TOKEN",   _yaml_token)
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", _yaml_chat_id)


def validate() -> None:
    """Raise a clear error if credentials are missing."""
    missing = []
    if not TOKEN:
        missing.append("TELEGRAM_TOKEN")
    if not CHAT_ID:
        missing.append("TELEGRAM_CHAT_ID")
    if missing:
        raise EnvironmentError(
            f"Missing Telegram credentials: {', '.join(missing)}\n"
            f"Add them to {ROOT}/.env  or  config/credentials.yaml under 'telegram:'"
        )
