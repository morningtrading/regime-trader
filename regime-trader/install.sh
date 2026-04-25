#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
#  install.sh — full installation script for regime-trader
#  Tested on: Ubuntu 24.04 (native + WSL2), Debian 12, Linux Mint 22
#
#  Usage:
#    chmod +x install.sh
#    ./install.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── colours ──────────────────────────────────────────────────────────────────
GREEN='\033[1;32m'
RED='\033[1;31m'
YELLOW='\033[1;33m'
CYAN='\033[1;36m'
DIM='\033[2m'
RESET='\033[0m'

ok()   { echo -e "  ${GREEN}✓${RESET}  $*"; }
fail() { echo -e "  ${RED}✗${RESET}  $*"; exit 1; }
info() { echo -e "  ${CYAN}→${RESET}  $*"; }
warn() { echo -e "  ${YELLOW}!${RESET}  $*"; }
step() { echo -e "\n${CYAN}━━━ $* ${RESET}"; }

echo ""
echo -e "${CYAN}  ██████╗ ███████╗ ██████╗ ██╗███╗   ███╗███████╗${RESET}"
echo -e "${CYAN}  ██╔══██╗██╔════╝██╔════╝ ██║████╗ ████║██╔════╝${RESET}"
echo -e "${CYAN}  ██████╔╝█████╗  ██║  ███╗██║██╔████╔██║█████╗  ${RESET}"
echo -e "${CYAN}  ██╔══██╗██╔══╝  ██║   ██║██║██║╚██╔╝██║██╔══╝  ${RESET}"
echo -e "${CYAN}  ██║  ██║███████╗╚██████╔╝██║██║ ╚═╝ ██║███████╗${RESET}"
echo -e "${CYAN}  ╚═╝  ╚═╝╚══════╝ ╚═════╝ ╚═╝╚═╝     ╚═╝╚══════╝${RESET}"
echo -e "${DIM}  HMM Regime Trader — installation${RESET}"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
step "STEP 1 — Python 3.12"
# ─────────────────────────────────────────────────────────────────────────────

PYTHON=""
for candidate in python3.12 python3 python; do
    if command -v "$candidate" &>/dev/null; then
        ver=$("$candidate" --version 2>&1 | grep -oP '\d+\.\d+' | head -1)
        major=$(echo "$ver" | cut -d. -f1)
        minor=$(echo "$ver" | cut -d. -f2)
        if [ "$major" -eq 3 ] && [ "$minor" -eq 12 ]; then
            PYTHON="$candidate"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    warn "Python 3.12 not found. Attempting to install via apt..."
    if command -v apt-get &>/dev/null; then
        sudo apt-get update -qq
        sudo apt-get install -y python3.12 python3.12-venv python3.12-dev
        PYTHON="python3.12"
    else
        fail "Cannot find or install Python 3.12. Install it manually and re-run."
    fi
fi

PYTHON_VERSION=$("$PYTHON" --version)
ok "Found $PYTHON_VERSION at $(command -v $PYTHON)"

# ─────────────────────────────────────────────────────────────────────────────
step "STEP 2 — Virtual environment"
# ─────────────────────────────────────────────────────────────────────────────

VENV_DIR="$SCRIPT_DIR/.venv"

if [ -d "$VENV_DIR" ]; then
    info "Existing venv found at .venv/ — checking it..."
    if "$VENV_DIR/bin/python" -c "import sys; assert sys.version_info[:2] == (3,12)" 2>/dev/null; then
        ok "Existing venv is Python 3.12 — reusing it"
    else
        warn "Existing venv is wrong Python version — recreating..."
        rm -rf "$VENV_DIR"
        "$PYTHON" -m venv "$VENV_DIR"
        ok "Venv recreated at .venv/"
    fi
else
    info "Creating virtual environment at .venv/ ..."
    "$PYTHON" -m venv "$VENV_DIR"
    ok "Venv created at .venv/"
fi

VENV_PYTHON="$VENV_DIR/bin/python"
VENV_PIP="$VENV_DIR/bin/pip"

# ─────────────────────────────────────────────────────────────────────────────
step "STEP 3 — Install dependencies"
# ─────────────────────────────────────────────────────────────────────────────

info "Upgrading pip..."
"$VENV_PIP" install --upgrade pip --quiet

info "Installing from requirements.txt (this takes ~2 min on first run)..."
"$VENV_PIP" install -r requirements.txt --quiet

ok "All packages installed"

# ─────────────────────────────────────────────────────────────────────────────
step "STEP 4 — Verify critical packages"
# ─────────────────────────────────────────────────────────────────────────────

check_package() {
    local pkg="$1"
    local expected_ver="$2"
    local actual_ver
    actual_ver=$("$VENV_PYTHON" -c "import importlib.metadata; print(importlib.metadata.version('$pkg'))" 2>/dev/null || echo "MISSING")
    if [ "$actual_ver" = "MISSING" ]; then
        fail "Package '$pkg' not installed"
    elif [ -n "$expected_ver" ] && [ "$actual_ver" != "$expected_ver" ]; then
        warn "$pkg version $actual_ver (expected $expected_ver) — continuing"
    else
        ok "$pkg $actual_ver"
    fi
}

check_package "hmmlearn"     "0.3.3"
check_package "numpy"        "2.4.4"
check_package "pandas"       "3.0.2"
check_package "scikit-learn" "1.8.0"
check_package "alpaca-py"    "0.43.2"
check_package "rich"         "15.0.0"

# Verify BLAS backend
info "Checking BLAS backend..."
BLAS_INFO=$("$VENV_PYTHON" -c "import numpy as np; cfg = np.show_config(mode='dicts'); print(cfg)" 2>/dev/null || echo "")
if echo "$BLAS_INFO" | grep -qi "openblas"; then
    ok "BLAS: OpenBLAS (correct for deterministic Linux results)"
else
    warn "BLAS backend unknown — results may differ from reference"
fi

# ─────────────────────────────────────────────────────────────────────────────
step "STEP 5 — Credentials"
# ─────────────────────────────────────────────────────────────────────────────

CREDS_FILE="$SCRIPT_DIR/config/credentials.yaml"
ENV_FILE="$SCRIPT_DIR/.env"

if [ -f "$CREDS_FILE" ]; then
    ok "config/credentials.yaml exists"
elif [ -f "$ENV_FILE" ]; then
    ok ".env file found — will be used as fallback"
else
    warn "No credentials found. Creating config/credentials.yaml from template..."
    cp config/credentials.yaml.example "$CREDS_FILE"
    echo ""
    echo -e "  ${YELLOW}ACTION REQUIRED:${RESET} Edit config/credentials.yaml and add your Alpaca API keys:"
    echo -e "  ${DIM}  Get keys at: https://alpaca.markets → Paper Trading → API Keys${RESET}"
    echo ""
    echo -e "  ${DIM}  nano config/credentials.yaml${RESET}"
    echo ""
fi

# ─────────────────────────────────────────────────────────────────────────────
step "STEP 6 — Run test suite"
# ─────────────────────────────────────────────────────────────────────────────

info "Running pytest (unit tests — no network calls)..."
if "$VENV_DIR/bin/pytest" tests/ -q --tb=short 2>&1 | tee /tmp/regime_pytest.log | tail -5; then
    ok "All tests passed"
else
    FAILED=$(grep -c "FAILED" /tmp/regime_pytest.log 2>/dev/null || echo "?")
    warn "$FAILED test(s) failed — check /tmp/regime_pytest.log for details"
    warn "Installation continues — failures may be non-critical"
fi

# ─────────────────────────────────────────────────────────────────────────────
step "STEP 7 — Smoke test (import check)"
# ─────────────────────────────────────────────────────────────────────────────

info "Verifying core modules load correctly..."
"$VENV_PYTHON" - <<'PYEOF'
import sys
errors = []
modules = [
    ("core.hmm_engine",      "HMMEngine"),
    ("backtest.backtester",  "WalkForwardBacktester"),
    ("data.market_data",     "MarketData"),
    ("monitoring.logger",    "TradeLogger"),
]
for mod, cls in modules:
    try:
        m = __import__(mod, fromlist=[cls])
        getattr(m, cls)
        print(f"  \033[1;32m✓\033[0m  {mod}.{cls}")
    except Exception as e:
        print(f"  \033[1;31m✗\033[0m  {mod}.{cls}: {e}")
        errors.append(mod)
if errors:
    sys.exit(1)
PYEOF

ok "All core modules import successfully"

# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo -e "${GREEN}  Installation complete!${RESET}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo ""
echo -e "  ${CYAN}Next steps:${RESET}"
echo ""
echo -e "  ${DIM}1. Add Alpaca API keys (if not done):${RESET}"
echo -e "     nano config/credentials.yaml"
echo ""
echo -e "  ${DIM}2. Run the interactive menu:${RESET}"
echo -e "     bash menu/regime_trader.sh"
echo ""
echo -e "  ${DIM}3. Or run a backtest directly:${RESET}"
echo -e "     source .venv/bin/activate"
echo -e "     python main.py backtest --asset-group stocks --start 2020-01-01 --compare"
echo ""
echo -e "  ${DIM}Reference results (Ubuntu 24.04, ed6d342):${RESET}"
echo -e "     Total Return +159.21%  |  Sharpe 1.123  |  MaxDD -15.53%"
echo -e "     Repo: $(pwd)"
echo ""
