# Gemini Project Mandates: Regime Trader

This document defines the foundational standards and architectural constraints for the `regime-trader` project. These mandates take precedence over general defaults and incorporate the core principles of `context.md`.

## Behavioral & Safety Mandates
- **No Hallucinations:** If unsure, state "I don't know based on the current repo. Please confirm." Never invent trading logic or performance metrics.
- **Surgical Changes:** Prefer incremental patches over full file rewrites. Always ask for confirmation before applying significant changes.
- **Confirmation Protocol:** Summarize existing content before proposing modifications. Wait for explicit user approval before deleting files or modifying configuration.
- **Deterministic Backtests:** All backtest code must use fixed seeds, log all parameters, and validate dataset availability.

## Architectural Mandates
- **Core Engine:** Hidden Markov Models (HMM) for regime detection (`core/hmm_engine.py`).
- **Data Flow:** `market_data.py` -> `feature_engineering.py` -> `hmm_engine.py` -> `signal_generator.py` -> `regime_strategies.py`.
- **Risk First:** All trades MUST pass through `risk_manager.py` before execution.
- **Broker Abstraction:** Use the Alpaca client abstraction in `broker/alpaca_client.py`.

## Engineering Standards
- **Python Version:** Python 3.10+
- **Type Safety:** Use type hints for all function signatures and complex variables.
- **Error Handling:** Use custom exceptions for trading logic (e.g., `RiskLimitExceeded`).
- **Mandatory Logging:** Every component must log prerequisites, inputs, outputs, errors, and decisions via `monitoring/logger.py`.
- **Configuration:** Use `config/settings.yaml` for strategy parameters and `.env` for secrets.

## Testing Standards
- **Framework:** `pytest`
- **Mandatory Tests:** No code is complete without tests. Every new function must include unit tests covering edge cases.
- **Specific Coverage:** 
  - Strategies in `tests/test_strategies.py`.
  - Risk logic in `tests/test_risk.py`.
  - HMM consistency in `tests/test_hmm.py`.
- **Look-ahead Bias:** Strictly forbid look-ahead bias in data processing and backtesting.

## Workspace Conventions
- **Docstrings:** Use Google-style docstrings for all modules and classes, including expected inputs/outputs and edge cases.
- **Formatting:** Adhere to PEP 8 standards.
- **Directory Structure:**
  - `core/`: Trading logic and HMM engine.
  - `data/`: Data ingestion and feature engineering.
  - `broker/`: Execution and account management.
  - `backtest/`: Simulation and performance analysis.
  - `monitoring/`: Logging, alerts, and dashboarding.
  - `tools/`: Optimization and utility scripts.
