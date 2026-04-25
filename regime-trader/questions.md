what the mandaotry files from this repo
is there any hard coded path or files or data

---

## Answers

### 1. Mandatory files to run this repo

**Code (required):**
- `main.py` — CLI entry point (backtest / trade / stress / groups / sweep / full-cycle)
- `core/` — HMM engine, regime strategies, asset groups registry
- `data/` — market_data (Alpaca client), feature_engineering, feature_blending, vix_fetcher
- `backtest/` — backtester (walk-forward), performance (benchmarks + metrics)
- `broker/` — alpaca_client (connect / orders / data)
- `monitoring/`, `menu/`, `tools/` — auxiliary (trade logger, interactive menu, debug tools)

**Config (required):**
- `config/settings.yaml` — HMM / strategy / risk / backtest parameters
- `config/asset_groups.yaml` — symbol groups (stocks, indices, crypto, midcap, …)
- `config/sets/*.yaml` (optional) — named preset overrides (conservative / balanced / aggressive)
- `config/active_set` (optional) — records currently applied preset name

**Secrets (required for live / Alpaca fetch):**
- `.env` at repo root — ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_DATA_FEED (env vars also work)

**Python deps:**
- `requirements.txt` — installed via `py -3.12 -m pip install -r requirements.txt`
- Python 3.12 mandatory (hmmlearn wheels missing on 3.14). Run everything with `py -3.12`.

**Writable directories (auto-created):**
- `results/` — per-run CSV outputs from the latest backtest
- `savedresults/` — timestamped historical backtest snapshots + training_log.csv
- `logs/` — runtime logs
- `models/` — persisted HMM models (trade mode)

**NOT mandatory (safe to delete):**
- `docs/`, `README.md`, `LESSONS_LEARNED.md`, `GEMINI.md`, `context.md`, `sequential_strategy.md`
- `pending prompt temporary.txt`, `never ask me if you can launch py -12 in - Copie.txt`
- `obsolete_code/` — confirmed obsolete
- `tests/` — only needed for `pytest`
- `scripts/` — one-off research orchestrators (e.g. ablation study)
- `state_snapshot.json` — reproducible run snapshot, regenerated on each run

### 2. Hard-coded paths / files / data

**No user-specific paths.** Everything is relative to the repo root via `Path(__file__).resolve().parent.parent`:
- `core/asset_groups.py` → `REPO_ROOT / config / asset_groups.yaml`
- `main.py` → `_ROOT / savedresults`, `_ROOT / results`, `_ROOT / config / sets`, etc.
- No `C:\Users\…`, no `/home/…` anywhere in the code.

**Hard-coded defaults / magic values (intentional, overridable):**
- `main.py:241,266` — `TimeFrame.Day` for backtest data fetch (hardcoded: backtests are daily by design; the `broker.timeframe: 5Min` in settings.yaml is for **live trading only**)
- `config/settings.yaml` — default HMM hyperparameters (n_candidates, stability_bars, min_confidence, slippage 10 bps…)
- `data/vix_fetcher.py` — `VIX_PROXY_SYMBOL = "VXX"` and yfinance ticker `^VIX`
- `core/asset_groups.py` — `SCHEMA_VERSION = 1`, default group fallback = first group if unset
- `backtest/backtester.py` — `_engine_kwargs` whitelist (names of HMMEngine kwargs; intentional filter, not data)

**Hard-coded data / symbols:** only in `config/asset_groups.yaml` — and that's the whole point of the file (editable via `main.py groups` CLI).

**Secrets:** none hardcoded. Alpaca credentials come exclusively from env vars / `.env`.

**Verdict:** no hard-coded user paths, no hard-coded secrets. Only "hardcoded" items are (a) the daily-timeframe constant for backtests (by design), (b) the VXX/^VIX ticker names, (c) a few internal schema constants.
