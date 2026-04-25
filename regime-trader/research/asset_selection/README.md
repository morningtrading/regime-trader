# Asset Selection — Monte Carlo Study

## Goal

The live strategy trades a fixed basket of symbols. The question this study answers is:

> **Is the current basket [SPY, QQQ, AAPL, MSFT, NVDA] optimal, or does a more
> diversified combination from a broad universe produce better risk-adjusted returns?**

The current basket has a known weakness: four of five symbols are tech-heavy and
highly correlated (~0.85 average pairwise correlation). This concentrates both
opportunity and risk in a single factor. The study searches for combinations that
keep or improve the Sharpe ratio while reducing average pairwise correlation.

**Constraint:** zero changes to the main codebase. All scripts are read-only
with respect to `config/`, `backtest/`, and `data/`. Outputs go to `results/` only.

---

## Methodology

### Step 1 — Define a neutral candidate universe

38 symbols chosen to maximally cover sector and asset-class diversity, with
individual stocks selected by **2019 market-cap rank** (not post-2020 performance)
to avoid hindsight bias.

| Category | Symbols | Count |
|----------|---------|------:|
| Broad market ETFs | SPY, QQQ, IWM | 3 |
| Sector ETFs | XLK, XLF, XLE, XLV, XLI, XLU, XLC | 7 |
| Alt / Macro ETFs | GLD, TLT, VNQ | 3 |
| Large-cap equities (2019 MC rank) | AAPL, MSFT, AMZN, GOOGL, JPM, JNJ, PG, V, MA, UNH, HD, BAC, XOM, CVX, WMT, KO, PEP | 17 |
| Growth / Semiconductors | NVDA, AMD, QCOM | 3 |
| Defense / Aerospace | LMT, RTX, NOC | 3 |
| Healthcare (individual) | PFE, ABT | 2 |

Crypto excluded: BTC/USD and ETH/USD are incompatible with the equity HMM
position-sizing model (produces total-loss outcomes).

### Step 2 — Score every combination by average pairwise correlation

For basket sizes k = 5, 6, 7:

| k | Combinations | Approach |
|---|-------------|---------|
| 5 | C(38,5) = 501,942 | Enumerate all — scored in ~3 s (pure numpy) |
| 6 | C(38,6) = 2,760,681 | Random sample 60,000 |
| 7 | C(38,7) = 12,620,256 | Random sample 60,000 |

Keep the **top 50 lowest-correlation combos per k** (150 total) for backtesting.
The baseline [SPY, QQQ, AAPL, MSFT, NVDA] is always force-included.

### Step 3 — Two-phase backtesting

Running the full 6-year walk-forward backtest (~30 s/combo) on all 150 combos
would take ~75 min. The two-phase approach cuts this to ~45 min total:

**Phase 1 — Fast screening (2023 → today, ~3 years)**
- 9 OOS folds × 63 bars, same train/test/step config as main backtest
- n_init reduced to 3 (vs 10 in production) — acceptable for relative ranking
- ~7 s/combo → 150 combos ≈ 18 min
- Output: ranked list, keep top 15 finalists

**Phase 2 — Full validation (2020 → today, ~6 years)**
- Full production config (17 OOS folds, n_init=3)
- Only the top 15 Phase 1 finalists + baseline = 16 backtests
- ~30 s/combo → ~8 min
- Output: definitive Sharpe / MaxDD / return for each finalist

Both phases are **resume-safe**: interrupted runs restart from the last checkpoint.

### Step 4 — Report

Compare finalists against the baseline on:
- Sharpe ratio (primary metric)
- Max drawdown
- Average pairwise correlation
- Pareto dominance (does any combo beat baseline on all three simultaneously?)
- Symbol frequency in top-50 (which assets consistently appear?)

---

## Files

```
research/asset_selection/
│
├── build_correlation_matrix.py       # Script 1 — fetch prices, build corr matrix
├── monte_carlo_asset_selection.py    # Script 2 — enumerate, filter, backtest
├── asset_selection_report.py         # Script 3 — analyse and print results
│
├── README.md                         # this file
│
└── results/                          # all outputs (gitignored)
    ├── corr_matrix.csv               # 38×38 pairwise correlation matrix
    ├── mc_results_phase1.csv         # 3-year screening results (~150 rows)
    ├── mc_results_phase2.csv         # 6-year validation results (~16 rows)
    └── asset_selection_summary.md    # final report saved as markdown
```

### Script 1 — `build_correlation_matrix.py`

**Purpose:** Fetch daily close prices for all 38 candidates and compute the
pairwise Pearson correlation of log-returns.

**Inputs:** Alpaca API credentials, date range 2020-01-01 → today

**Outputs:**
- `results/corr_matrix.csv` — 38×38 symmetric matrix, values in [-1, 1]
- Console: reference pair table, baseline basket avg corr

**Run once.** Re-run only if you change the universe or want fresher data.

```bash
source .venv/bin/activate   # from repo root
python research/asset_selection/build_correlation_matrix.py
```

---

### Script 2 — `monte_carlo_asset_selection.py`

**Purpose:** The main Monte Carlo loop. Enumerates / samples asset combinations,
filters by correlation, runs the two-phase backtest, saves all results.

**Inputs:**
- `results/corr_matrix.csv` (Script 1 must run first)
- `config/settings.yaml` — backtest parameters (train/test/step windows, strategy config)
- `config/credentials.yaml` — Alpaca API keys
- Alpaca historical daily bars for all 38 symbols

**Outputs:**
- `results/mc_results_phase1.csv` — one row per combo tested in Phase 1

  | Column | Description |
  |--------|-------------|
  | symbols | JSON list of ticker symbols |
  | k | basket size (5, 6, or 7) |
  | regime_proxy | symbol used as HMM regime signal |
  | avg_corr | mean pairwise correlation of the basket |
  | sharpe | 3-year out-of-sample Sharpe ratio |
  | max_drawdown | maximum drawdown over Phase 1 period |
  | total_return | cumulative return over Phase 1 period |
  | cagr | compound annual growth rate |
  | n_trades | total number of trades |
  | win_rate | fraction of winning trades |
  | is_baseline | True if this is the current production basket |

- `results/mc_results_phase2.csv` — same columns plus `sharpe_3yr` (Phase 1 Sharpe for reference)

**Resume-safe:** already-completed combos are detected by their `symbols` key and skipped.

```bash
python research/asset_selection/monte_carlo_asset_selection.py
```

Key configuration constants at the top of the script:

| Constant | Default | Meaning |
|----------|---------|---------|
| `TOP_PER_K` | 50 | Lowest-corr combos per k fed to Phase 1 |
| `PHASE2_TOP_N` | 15 | Phase 1 finalists promoted to Phase 2 |
| `PHASE1_START` | 2023-01-01 | Phase 1 data window start |
| `PHASE2_START` | 2020-01-01 | Phase 2 data window start |

---

### Script 3 — `asset_selection_report.py`

**Purpose:** Read Phase 2 results and produce a human-readable analysis.

**Inputs:** `results/mc_results_phase2.csv`

**Outputs:**
- Console report with 5 sections (see below)
- `results/asset_selection_summary.md` — top-10 table + baseline comparison

**Report sections:**

1. **Top-10 by 6-year Sharpe** — ranked table with 3yr Sharpe, return, MaxDD, avg corr, symbols
2. **Correlation vs Sharpe** — bucketed scatter: does lower correlation actually improve Sharpe?
3. **Symbol frequency in top-50** — which assets appear most consistently across good combos?
4. **Pareto frontier** — does any combo strictly dominate the baseline (better Sharpe AND lower corr AND lower MaxDD)?
5. **Recommendation** — verdict: keep baseline or adopt a new basket?

```bash
python research/asset_selection/asset_selection_report.py
```

---

## Interpreting Results

| Finding | Interpretation | Action |
|---------|---------------|--------|
| Baseline in top-3 | Current basket is near-optimal for this universe | Keep [SPY, QQQ, AAPL, MSFT, NVDA] |
| Better combo found, no hindsight bias | Genuine diversification benefit | Update `config/asset_groups.yaml` and re-run full backtest |
| Better combo found, uses post-2020 winners | Selection bias — not a real edge | Do not adopt |
| TLT / GLD appear frequently | Bonds/gold genuinely diversify in this regime model | Worth investigating a mixed equity+macro basket |
| Defense stocks (LMT, RTX, NOC) appear | Low corr with tech, HMM works on them | Consider adding one defense name |
| Sharpe flat across all corr buckets | Correlation level doesn't matter for this strategy | Asset selection is not a key lever |

---

## Running the Full Study

```bash
# From repo root (WSL, venv active)
source .venv/bin/activate

# 1. Build correlation matrix (one-time, ~3 min)
python research/asset_selection/build_correlation_matrix.py

# 2. Run Monte Carlo (~45 min for full 150-combo run)
python research/asset_selection/monte_carlo_asset_selection.py

# 3. Print report
python research/asset_selection/asset_selection_report.py
```

To run a quick logic test (15 combos, ~5 min), set `TOP_PER_K = 5` and
`PHASE2_TOP_N = 5` in `monte_carlo_asset_selection.py` before running.
