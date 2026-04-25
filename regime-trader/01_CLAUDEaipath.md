# Trading Bot Project — Context for Claude

This file is loaded at the start of every Claude Code session. It tells Claude what this project is, how to work on it safely, and what is non-negotiable.

---

## What this project is

A Python algorithmic trading bot. Core components:

- `core/` — signal generation, regime detection, risk management
- `broker/` — broker API wrappers (Alpaca, Hyperliquid, MT5)
- `data/` — market data and feature engineering
- `backtest/` — walk-forward backtester, performance metrics, stress tests
- `monitoring/` — logging, dashboard, alerts
- `tests/` — pytest suite, including critical look-ahead bias tests

The bot trades real money (eventually). Bugs cost money. Act like it.

---

## Absolute rules — NEVER violate these

1. **NEVER use `model.predict()` from hmmlearn for live or backtest inference.** It runs Viterbi across the whole sequence, which is look-ahead bias. ALWAYS use the forward algorithm (filtered inference) that uses only data up to time t. See `core/hmm_engine.py:predict_regime_filtered`.

2. **NEVER submit an order without a stop loss.** The risk manager rejects any signal without one. Do not add bypass paths. Do not "temporarily" disable this check.

3. **NEVER hardcode API keys.** Credentials load from `config/credentials.yaml` only (gitignored — see `config/credentials.yaml.example` for the template). If you see a key in code, that is a bug — remove it immediately.

4. **NEVER default to live trading.** `paper_trading: true` is the default in `config/settings.yaml`. Switching to live requires explicit confirmation prompt in the code — do not remove that prompt.

5. **NEVER skip the look-ahead bias test.** `tests/test_look_ahead.py` must pass before any commit that touches HMM or feature code. If it fails, something is feeding future data into present decisions.

6. **NEVER widen stops after a position is open.** Stops can only tighten. This is enforced in `broker/order_executor.py:modify_stop`.

---

## Commands

This repo runs on Windows with Python 3.12 (the `py -3.12` launcher). Default `python` is 3.14, which lacks `hmmlearn` wheels — always use `py -3.12`. On WSL/Linux, substitute `python` for `py -3.12`.

```bash
# Setup
py -3.12 -m pip install -r requirements.txt
cp config/credentials.yaml.example config/credentials.yaml  # then fill in Alpaca paper keys

# Trading (live/paper)
py -3.12 main.py trade                          # live paper trading (default group)
py -3.12 main.py trade --dry-run                # full pipeline, no real orders
py -3.12 main.py trade --train-only             # retrain HMM, exit
py -3.12 main.py trade --asset-group crypto     # switch asset basket

# Backtesting (walk-forward)
py -3.12 main.py backtest --asset-group stocks   --start 2020-01-01 --compare
py -3.12 main.py backtest --asset-group crypto   --start 2020-01-01 --compare
py -3.12 main.py backtest --asset-group indices  --start 2020-01-01 --compare
py -3.12 main.py backtest --symbols SPY,QQQ      --start 2020-01-01

# Stress / sweeps / utilities
py -3.12 main.py stress      --asset-group indices --start 2019-01-01
py -3.12 main.py full-cycle  --start 2020-01-01           # all 3 groups back-to-back
py -3.12 main.py cs-sweep    --asset-group stocks         # min_confidence × stability_bars grid
py -3.12 main.py sweep       --asset-group stocks         # min_rebalance_interval sweep
py -3.12 main.py groups list                              # manage asset_groups.yaml

# Tests (run these often)
py -3.12 -m pytest tests/test_look_ahead.py -v   # the one that matters most
py -3.12 -m pytest tests/test_risk.py -v
py -3.12 -m pytest tests/ -v                     # full suite
```

If you change HMM or feature code, run `py -3.12 -m pytest tests/test_look_ahead.py -v` before you consider the work done.

### Config layers

- `config/settings.yaml` — base config (all tunable numbers)
- `config/sets/{conservative,balanced,aggressive}.yaml` — overlays applied on top of base
- `config/active_set` — single line naming the active overlay (currently `conservative`)
- `config/asset_groups.yaml` — named symbol baskets (`stocks`, `stocks4`, `crypto`, `indices`, …)
- `config/credentials.yaml` — Alpaca/broker keys (gitignored)

`broker.asset_group` in `settings.yaml` overrides `broker.symbols` when non-null. Set `asset_group: null` to use the explicit `symbols:` list.

### Backtest defaults worth knowing

- `enforce_stops=True` is the default — backtests now exercise the same ATR-graded stop layer as live. Pass `--no-enforce-stops` only for diagnostic A/B runs.
- Canonical baseline (stocks group, 2020-01-01 → today, conservative set): see `docs/BASELINE_AFTER_ENFORCE_STOPS.md`.
- Walk-forward: IS 252 / OOS 63 / step 63 bars.

---

## Workflow

- Before writing code, read the relevant module and its tests. Match existing patterns.
- Small changes > large refactors. If a task is big, break it into phases and run tests between each.
- When adding a feature, add tests in the same PR.
- When fixing a bug, add a regression test that would have caught it.
- Do not touch `core/risk_manager.py` thresholds without asking me first.
- Commit after every meaningful code change (skip cosmetic-only edits). Push is allowed on Claude's own judgement — never force-push, never directly to `main`.

---

## Project-specific conventions

- All configurable numbers live in `config/settings.yaml`. No magic numbers in code.
- All log output is structured JSON (`monitoring/logger.py`). Every entry includes: `timestamp`, `regime`, `equity`, `daily_pnl`.
- Broker adapters implement `broker/base.py:BaseBroker`. Never call Alpaca SDK directly outside `broker/alpaca_client.py`.
- Strategies inherit from `core/regime_strategies.py:BaseStrategy` and implement `generate_signal()`.
- Walk-forward sweeps that mutate `settings.yaml` MUST `copy.deepcopy` the original and restore in a `try/finally` (see LESSONS_LEARNED.md #1 — "Walk-Forward Settings Pollution").

---

## When stuck

- Check `docs/debug-playbook.md` first for common issues.
- Check `tests/` for examples of how a module is expected to be used.
- If unclear about a design decision, ask before implementing. Do not guess.

---

## Python version

This project requires Python **3.12** specifically (hmmlearn has no 3.13/3.14 wheels yet). Always invoke via `py -3.12` on Windows or a 3.12 venv on Linux. Type hints use modern syntax (`list[int]`, `X | None`).

---

## Imports — load these on demand, not at startup

@docs/debug-playbook.md
@docs/go-live-checklist.md
