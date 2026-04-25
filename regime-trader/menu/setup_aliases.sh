#!/usr/bin/env bash
# setup_aliases.sh — installe menummh et menutr dans ~/.bashrc
# Linux et Windows Git Bash

MENU_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASHRC="$HOME/.bashrc"

# Supprimer les anciennes entrees
sed -i '/alias menummh=/d' "$BASHRC" 2>/dev/null
sed -i '/alias menutr=/d'  "$BASHRC" 2>/dev/null

# Ajouter les nouveaux alias
cat >> "$BASHRC" <<EOF

# regime-trader aliases (auto-installe par setup_aliases.sh)
alias menummh='bash "$MENU_DIR/regime_trader.sh"'
alias menutr='bash "$MENU_DIR/tailscale_transfer.sh"'
EOF

echo "Aliases installes dans $BASHRC"
echo "Lance: source ~/.bashrc"
