# Lessons Learned — Regime Trader

A record of non-obvious bugs, design decisions, and hard-won insights accumulated during development. Ordered roughly by theme, not chronology.

---

## 1. Walk-Forward Settings Pollution

**What happened:**  
The Monte Carlo tool (`tools/montecarlo_alloc.py`) was returning all-ERR results. Every draw was failing with "Insufficient data."

**Root cause:**  
`WalkForwardBacktester` temporarily patches `config/settings.yaml` during a run (e.g., `train_window=106` for a shorter fold). If the process is interrupted or if another tool loads `settings.yaml` afterward, it sees the patched values, not the production defaults.

**Fix applied:**  
After loading `base_cfg` in the Monte Carlo tool, explicitly reset the 7 affected fields to their production defaults before passing the config to any backtest:

```python
_set_nested(base_cfg, "backtest.train_window",       252)
_set_nested(base_cfg, "backtest.test_window",        126)
_set_nested(base_cfg, "backtest.step_size",          126)
_set_nested(base_cfg, "backtest.sma_long",           200)
_set_nested(base_cfg, "backtest.sma_trend",          50)
_set_nested(base_cfg, "backtest.volume_norm_window",  50)
_set_nested(base_cfg, "backtest.zscore_window",       60)
```

**General rule:**  
Any tool that temporarily mutates a shared config file must restore it (try/finally or context manager). Any downstream tool that reads the same file must defensively reset critical fields.

---

## 2. Double-Percent Display Bug

**What happened:**  
Monte Carlo summary showed Total Return as `+10063.00%` instead of `+100.63%`.

**Root cause:**  
`_to_float("+100.63%")` strips the `%` sign and returns `100.63`. Then the format string `f"{100.63:+.2%}"` multiplies by 100 again (Python's `%` format always assumes a 0–1 fraction), yielding `10063.00%`.

**Fix:**  
When the value is already in percentage-point form (e.g., `100.63` means 100.63%), format it as:
```python
f"{value:+.2f}%"   # correct — appends literal %
# NOT
f"{value:+.2%}"    # wrong  — multiplies by 100 first
```

**General rule:**  
`:.2%` is for fractions in [0, 1]. If a value is already in percent-points, use `:.2f` + literal `"%"`.

---

## 3. NumPy divide-by-zero in Heatmap

**What happened:**  
`np.where(counts > 0, totals / counts, np.nan)` emitted a `RuntimeWarning: invalid value encountered in divide` even though the NaN branch was protecting zero-count cells.

**Root cause:**  
NumPy evaluates both branches of `np.where` before choosing. `totals / counts` is computed everywhere — including zero-count cells — triggering the warning even though those values are discarded.

**Fix:**  
```python
with np.errstate(invalid="ignore", divide="ignore"):
    grid = np.where(counts > 0, totals / counts, np.nan)
```

---

## 4. hmmlearn EM Convergence Noise

**What happened:**  
Backtest console output was flooded with `ConvergenceWarning` messages from hmmlearn during every fold.

**Root cause:**  
hmmlearn emits a warning for every EM candidate that hits the iteration limit. With `n_candidates=[3,4,5,6,7]` and `n_init=10`, this is expected behaviour on every fold — not an error.

**Fix:**  
Suppress for the duration of the backtest run, before the backtester is instantiated:

```python
logging.getLogger("hmmlearn").setLevel(logging.ERROR)
logging.getLogger("core.hmm_engine").setLevel(logging.WARNING)
```

Restore (or just leave at ERROR) after the run completes. Live mode is unaffected because it uses the dashboard and alerts for visibility.

---

## 5. Regime-Change Log Level

**What happened:**  
Every regime-change event in `core/hmm_engine.py` was logged at `WARNING`, which looked alarming in production but was actually normal operation.

**Decision:**  
Changed to `INFO`. In live mode, the dashboard and alerts system handles visibility. In backtest mode, the logger is suppressed anyway (see lesson 4). `WARNING` should be reserved for genuinely unexpected states.

---

## 6. Rich Console on Windows — cp1252 Encoding

**What happened:**  
Unicode block characters (`█`, `░`, `✓`) caused `UnicodeEncodeError: 'charmap' codec can't encode characters` on Windows when Rich fell back to the legacy Windows renderer (cp1252 terminal).

**Fix:**  
- Use ASCII-safe alternatives: `#` for filled, `-` for empty, `OK` for check.
- Set `legacy_windows=False` on the `Console` constructor to bypass the legacy renderer path.
- Always set `no_wrap=True, overflow="crop"` on progress lines to prevent Rich from wrapping coloured text across lines (especially important when using `\r` overwrite).

**General rule:**  
Assume cp1252 unless you have explicit confirmation of UTF-8. Prefer ASCII for progress indicators; reserve Unicode for dashboard panels where the encoding can be controlled.

---

## 7. Rich `\r` Overwrite Pattern

**Pattern used for progress lines:**

```python
# Training phase — overwrites itself while HMM is running
console.print(line, end="\r", highlight=False, no_wrap=True, overflow="crop")

# Complete phase — persists in scroll-back
console.print(line, end="\n", highlight=False, no_wrap=True, overflow="crop")
```

`highlight=False` disables Rich's auto-syntax-highlighting (avoids spurious colour on numbers).  
`no_wrap=True` + `overflow="crop"` keeps the line within terminal width without wrapping.

---

## 8. Monte Carlo Allocation Sensitivity — Results Interpretation

From 100-draw Monte Carlo run (Jan 2020 – Mar 2025, 10 stocks):

| Metric          | Mean    | Std     | P5      | P95     | Fragility |
|-----------------|---------|---------|---------|---------|-----------|
| Sharpe          | +0.52   | 0.06    | +0.42   | +0.63   | low       |
| Total Return    | +100.6% | 12.4%   | +79.6%  | +120.7% | low       |
| Max Drawdown    | -22.1%  | 2.3%    | -26.3%  | -18.2%  | medium    |

**Key finding:**  
Sharpe and Total Return were robust across the `mid_vol_no_trend` / `high_vol` allocation parameter space. MaxDD showed moderate sensitivity — higher allocations in high-vol regimes widened drawdowns.

**Action taken:**  
Raised both `mid_vol_allocation_no_trend` and `high_vol_allocation` from `0.60` → `0.70` to capture more upside while keeping drawdown within acceptable range.

---

## 9. Walk-Forward Progress Callback Design

**Decision:**  
Added an optional `progress_callback` parameter to `WalkForwardBacktester.run()` rather than embedding display logic inside the backtester.

**Rationale:**  
- Backtester stays decoupled from terminal formatting.
- The callback can be `None` (default) for use in tools / tests where no display is needed.
- Caller (main.py) owns the display style — Rich, plain text, or silent.

**Callback contract:**
```python
callback(fold_id: int, n_total: int, phase: str, info: dict)
# phase = "training"  — called before _run_single_window
# phase = "complete"  — called after fold succeeds (skipped folds get no complete call)
```

---

## 10. Structured JSON Logging — 4-File Pattern

Four rotating log files, each filtered by event type:

| File         | Events captured                          |
|--------------|------------------------------------------|
| `main.log`   | everything (catch-all)                   |
| `trades.log` | `trade`, `fill`                          |
| `alerts.log` | `risk_event`, `error`                    |
| `regime.log` | `regime_change`, `rebalance`             |

Filtering is implemented via `logging.Filter` subclasses attached to each handler, not by logger name. This means a single `logger.info(...)` call routes to the correct file automatically based on an `event_type` field in `extra`.

**Context injection:**  
Every log record includes a `"ctx"` dict `{regime, probability, equity, positions, daily_pnl}` via a thread-safe `set_context()` call. Useful for correlating log lines with system state at the time they were emitted.

---

## 11. Alert Rate Limiting

Alerts use a per-key `_last_sent` dict with a default 15-minute window. The key is `(alert_type, subject)` so the same alert type for different symbols is rate-limited independently.

Do not use a single global rate limit — a data-feed failure for `SPY` should not suppress a circuit-breaker alert for drawdown.

---

## 12. Python Version

Always use `py -3.12` on this machine. The default `python` / `python3` resolves to Python 3.14, which does not have hmmlearn wheels. Confirm with `py -3.12 -c "import hmmlearn; print(hmmlearn.__version__)"`.

---

## 13. EW = Equal-Weighted in Benchmark Comparisons

In the performance report comparison table:

- **Buy & Hold (EW)**: buy equal dollar amounts of all N symbols on day 1, hold forever.
- **SMA-200 (EW)**: apply SMA-200 crossover rule independently per symbol with equal sizing.
- **EMA 9/45 (EW)**: apply EMA 9/45 crossover independently per symbol with equal sizing.
- **Random (mean)**: average of 100 random entry/exit signal simulations, equal-weighted.

Equal-weighting is used so the benchmarks are independent of market-cap bias and reflect the same universe the strategy trades.
