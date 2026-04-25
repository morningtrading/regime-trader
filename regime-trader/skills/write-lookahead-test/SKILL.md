---
name: write-lookahead-test
description: Generate rigorous look-ahead bias tests for any module that touches market data, features, or signals. Use this skill whenever the user adds or modifies code in data/feature_engineering.py, core/hmm_engine.py, core/signal_generator.py, any strategy file, or any function that computes features or signals from a time series. Also use when the user mentions "look-ahead," "future data leak," "Viterbi," "lookahead," "no peek," "causal inference," or asks "could this be peeking at the future?" Do not skip this when adding new features — look-ahead bias is the #1 reason retail backtests lie.
---

# Write a Look-Ahead Bias Test

Look-ahead bias is when a backtest uses data that wouldn't have been available at the time of the decision. It's the most common silent bug in retail trading systems and it makes strategies look profitable when they aren't. This skill generates tests that catch it.

## The core principle

A function that computes a signal or feature at time `t` must return **the identical value** whether you pass it data from `[0:t]` or `[0:t+1000]`. If the value changes when you add more future data, the function is leaking future information into present decisions.

## Workflow

### Step 1 — Identify what to test

Ask the user (or determine from context) which function/module you're testing. Look-ahead tests apply to:

- Any feature computation function (returns, volatility, indicators, z-scores)
- Any model prediction method (`predict_regime_filtered`, signal generators)
- Any rolling or expanding computation
- Any standardization or normalization

If the function takes a full time series and returns a full time series, it needs this test.

### Step 2 — Read the target code

Open the file and look for the specific functions to test. Red flags while reading:

- `hmm.predict(X)` — Viterbi, runs over whole sequence. This IS look-ahead. Replace with forward algorithm.
- `df.rolling(...).mean().shift(-N)` — negative shift pulls future data to present.
- `StandardScaler().fit_transform(X)` on the full dataset — uses future data to compute the mean/std.
- `ta` library functions used without checking — some use centered windows.
- Any `.ffill()` or `.bfill()` applied after a join — bfill leaks future values backward.

### Step 3 — Generate the test

Use this pattern. **Before pasting, adapt the imports to match the project's actual module paths** — if the project has `hmm.py` instead of `hmm_engine.py`, or `features.py` instead of `feature_engineering.py`, change the imports accordingly. Use Claude Code to adapt automatically: "generate a look-ahead test adapted to the actual file structure of this project."

```python
"""
tests/test_look_ahead.py

CRITICAL: These tests verify no function uses future data to compute past values.
Look-ahead bias is the #1 cause of retail backtests that look profitable but
lose money live. If any of these tests fail, STOP — the strategy is not usable.
"""

import numpy as np
import pandas as pd
import pytest
from core.hmm_engine import HMMEngine
from data.feature_engineering import compute_features


@pytest.fixture
def sample_ohlcv():
    """500 bars of synthetic OHLCV data with a known regime shift at bar 300."""
    np.random.seed(42)
    n = 500
    returns = np.concatenate([
        np.random.normal(0.0005, 0.008, 300),   # calm
        np.random.normal(-0.002, 0.025, 200),   # turbulent
    ])
    close = 100 * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(np.random.normal(0, 0.003, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.003, n)))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    volume = np.random.lognormal(15, 0.5, n)

    return pd.DataFrame({
        'open': open_, 'high': high, 'low': low, 'close': close, 'volume': volume
    }, index=pd.date_range('2020-01-01', periods=n, freq='D'))


def test_features_no_look_ahead(sample_ohlcv):
    """
    Features computed at time t must be identical whether we pass data up to t
    or data up to t+200. If they differ, future data is leaking into the past.
    """
    t = 400

    # Compute features using data only up to time t
    features_short = compute_features(sample_ohlcv.iloc[:t+1])
    value_at_t_short = features_short.iloc[-1]

    # Compute features using data up to t+200 (future data available)
    features_long = compute_features(sample_ohlcv.iloc[:t+200])
    value_at_t_long = features_long.iloc[t]

    # The value AT time t must be identical in both cases
    pd.testing.assert_series_equal(
        value_at_t_short,
        value_at_t_long,
        check_names=False,
        obj=f"Feature values at t={t} differ — LOOK-AHEAD BIAS DETECTED"
    )


def test_hmm_filtered_prediction_no_look_ahead(sample_ohlcv):
    """
    HMM regime at time t must be identical whether the forward algorithm sees
    data[0:t+1] or data[0:t+100]. If predict_regime_filtered uses Viterbi
    instead of forward, this will fail.
    """
    features = compute_features(sample_ohlcv)
    engine = HMMEngine(n_components=3, random_state=42)
    engine.fit(features.iloc[:300])  # train on first half only

    t = 400

    # Predict using data only up to time t
    regime_short = engine.predict_regime_filtered(features.iloc[:t+1])[-1]

    # Predict using data up to t+100 (more future data available)
    regime_long = engine.predict_regime_filtered(features.iloc[:t+101])[t]

    assert regime_short == regime_long, (
        f"Regime at t={t} differs: short={regime_short}, long={regime_long}. "
        "LOOK-AHEAD BIAS DETECTED — predict_regime_filtered is using future data. "
        "This is almost certainly because it calls model.predict() (Viterbi) "
        "instead of the forward algorithm."
    )


def test_feature_standardization_no_look_ahead(sample_ohlcv):
    """
    Rolling z-scores must use only past data. Using StandardScaler().fit_transform()
    on the full dataset is look-ahead bias — it computes the mean/std using ALL
    data including the future.
    """
    features = compute_features(sample_ohlcv)
    t = 400

    z_short = features.iloc[:t+1]  # assumes compute_features returns standardized
    z_long = features.iloc[:t+200]

    # Values at time t must match
    np.testing.assert_array_almost_equal(
        z_short.iloc[-1].values,
        z_long.iloc[t].values,
        decimal=10,
        err_msg="Standardization uses future data. Use rolling z-score, not global."
    )


def test_no_negative_shifts_in_features():
    """
    Scan feature code (parsed as AST, not raw text) for patterns that are
    ALWAYS look-ahead bias. Using AST avoids false positives on comments
    and docstrings that discuss these patterns.
    """
    import ast
    import pathlib

    feature_file = pathlib.Path('data/feature_engineering.py')
    if not feature_file.exists():
        pytest.skip(f"{feature_file} not found — adapt path to your project")

    tree = ast.parse(feature_file.read_text())
    findings = []

    for node in ast.walk(tree):
        # Check for .shift(-N) — any negative shift argument
        if (isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == 'shift'
            and node.args
            and isinstance(node.args[0], ast.UnaryOp)
            and isinstance(node.args[0].op, ast.USub)):
            findings.append(f"Line {node.lineno}: negative .shift() detected")

        # Check for center=True in rolling()
        if (isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == 'rolling'):
            for kw in node.keywords:
                if (kw.arg == 'center'
                    and isinstance(kw.value, ast.Constant)
                    and kw.value.value is True):
                    findings.append(f"Line {node.lineno}: rolling(center=True) detected")

        # Check for .bfill() calls
        if (isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == 'bfill'):
            findings.append(f"Line {node.lineno}: .bfill() detected — backfill leaks future data")

        # Check for .fit_transform() — usually look-ahead unless inside a rolling loop
        # We flag these for human review rather than auto-fail
        if (isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == 'fit_transform'):
            findings.append(
                f"Line {node.lineno}: .fit_transform() detected — "
                "VERIFY this is inside a rolling loop, otherwise it's look-ahead"
            )

    assert not findings, (
        "Look-ahead bias patterns found in feature_engineering.py:\n  "
        + "\n  ".join(findings)
    )


def test_backtest_identical_at_different_end_dates(sample_ohlcv):
    """
    The regime at bar 400 in a backtest ending at bar 450 must equal
    the regime at bar 400 in a backtest ending at bar 500.
    """
    from backtest.backtester import run_backtest

    result_short = run_backtest(sample_ohlcv.iloc[:450])
    result_long = run_backtest(sample_ohlcv.iloc[:500])

    assert result_short.regimes[400] == result_long.regimes[400], (
        "Backtest is not deterministic across end dates. Look-ahead bias "
        "somewhere in the pipeline. Investigate signal generator, HMM, features."
    )
```

### Step 4 — Run the tests yourself and report results

Run the tests. Handle the environment:

- If `pytest` isn't installed, install it (`pip install pytest` or `uv pip install pytest` — use whatever package manager the project already uses)
- If the test file path is different (e.g., `test/` instead of `tests/`, or `src/tests/`), adapt
- If the import paths fail (e.g., `from core.hmm_engine import HMMEngine` errors with `ModuleNotFoundError`), check the project's `pyproject.toml` / `setup.py` / `PYTHONPATH` and either install the package in editable mode (`pip install -e .`) or adjust the imports to match the project structure

Command to try first (adapt if the project structure is different):

```bash
pytest tests/test_look_ahead.py -v
```

Report the result to the user in plain English: did it pass, did it fail, and if failed, which assertion triggered and what the most likely cause is.

If any test fails:
1. Do NOT continue with other work.
2. Do NOT patch the test to make it pass.
3. Read the error, find the source of the leak, fix the actual code.
4. Re-run.

### Step 5 — Add to CI

If the project has CI (look for `.github/workflows/`, `.gitlab-ci.yml`, `circle.yml`, etc.), make sure `test_look_ahead.py` runs on every push. Offer to add it if it's missing. This test is more important than code coverage.

## Examples of what IS look-ahead (fail the test)

```python
# BAD: uses future data to normalize
features['z'] = (features['x'] - features['x'].mean()) / features['x'].std()

# BAD: Viterbi looks at whole sequence
regime = hmm.predict(observations)

# BAD: negative shift
df['signal'] = df['price'].shift(-1) > df['price']  # "tomorrow > today"

# BAD: centered window
vol = df['returns'].rolling(20, center=True).std()
```

## Examples of what is NOT look-ahead (pass the test)

```python
# GOOD: rolling z-score uses only past 252 bars
features['z'] = (features['x'] - features['x'].rolling(252).mean()) / features['x'].rolling(252).std()

# GOOD: forward algorithm uses only past+present
regime = hmm_engine.predict_regime_filtered(observations[:t+1])[-1]

# GOOD: causal shift
df['prev_return'] = df['return'].shift(1)  # positive shift is fine

# GOOD: trailing window
vol = df['returns'].rolling(20).std()
```

## Do not

- Do not skip this test because "the code looks right."
- Do not patch failures by loosening the assertions.
- Do not accept "close enough" — values must be exactly identical (use `decimal=10` for floats).
