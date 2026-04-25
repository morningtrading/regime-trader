#!/usr/bin/env python3
"""
telegram/hooks.py — Command-line interface to send Telegram notifications.

Usage (from repo root):
    python telegram/hooks.py test        — test connectivity
    python telegram/hooks.py summary     — latest backtest summary (4 lines)
    python telegram/hooks.py trades      — last 5 trades
    python telegram/hooks.py stress      — stress test results
    python telegram/hooks.py regime      — current regime status
    python telegram/hooks.py all         — send everything

Never called automatically — 100% manual trigger.
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from telegram.bot import send
from telegram import formatter


COMMANDS = {
    "test":    ("🔌 Test connexion",         formatter.format_test),
    "summary": ("📊 Résumé backtest",        formatter.format_backtest_summary),
    "trades":  ("📈 Derniers trades",         formatter.format_latest_trades),
    "stress":  ("⚡ Stress test",             formatter.format_stress_summary),
    "regime":  ("🎯 Régime actuel",           formatter.format_regime_status),
}


def _send(label: str, build_fn) -> bool:
    print(f"  → {label} ...", end=" ", flush=True)
    try:
        text = build_fn()
        ok = send(text)
        print("✓ envoyé" if ok else "✗ échec")
        return ok
    except Exception as exc:
        print(f"✗ erreur: {exc}")
        return False


def main() -> None:
    cmd = sys.argv[1].lower() if len(sys.argv) > 1 else "help"

    if cmd == "all":
        print("Envoi de tous les messages Telegram :")
        for key, (label, fn) in COMMANDS.items():
            _send(label, fn)
        return

    if cmd in COMMANDS:
        label, fn = COMMANDS[cmd]
        print(f"Telegram — {label}")
        ok = _send(label, fn)
        sys.exit(0 if ok else 1)

    # help
    print("Usage: python telegram/hooks.py <commande>\n")
    print("Commandes disponibles:")
    for key, (label, _) in COMMANDS.items():
        print(f"  {key:<10} {label}")
    print(f"  {'all':<10} Envoyer tous les messages")


if __name__ == "__main__":
    main()
