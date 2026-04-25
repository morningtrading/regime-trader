---
name: add-risk-check
description: Add a new risk validation rule to the risk manager without breaking existing behavior. Use this skill whenever the user wants to add a risk check, safety rule, position limit, circuit breaker, exposure cap, correlation check, leverage rule, or any veto condition on trading signals. Also use when the user mentions "don't let it trade if," "block trades when," "add a rule that," "safety check," "circuit breaker," "risk filter," or wants to protect against a specific failure mode. The risk manager has veto power over every signal — adding rules here is how you control the bot's behavior.
---

# Add a Risk Check

The risk manager (`core/risk_manager.py`) has absolute veto power over every signal. Adding a check here is the safest way to control what the bot can and cannot do. This skill walks through doing it correctly — with a test, config-driven thresholds, and no regressions to existing behavior.

## Core principle

Every risk check must answer three questions:
1. **What does this protect against?** (Write it in a docstring.)
2. **What's the threshold, and why that number?** (Put in `settings.yaml`, not hardcoded.)
3. **What happens when it fires?** (Reject? Reduce size? Log only?)

If you can't answer all three clearly, don't add the check yet.

## Workflow

### Step 1 — Understand what they're trying to prevent

Ask the user (if not clear):
- What specific scenario are we protecting against? (Give me a concrete example.)
- Should this REJECT the trade, REDUCE the size, or just WARN?
- What threshold should trigger it?
- Is this a portfolio-level check, position-level check, or order-level check?

### Step 2 — Read the existing risk manager

Open `core/risk_manager.py` and understand the existing structure:

- `RiskManager.validate_signal(signal, portfolio_state) -> RiskDecision` — main entry point
- `PortfolioState` — dataclass with equity, cash, positions, drawdown, etc.
- `RiskDecision` — dataclass with `approved`, `modified_signal`, `rejection_reason`, `modifications`
- Existing checks follow a pattern: pure function, returns a `RiskDecision` or `None` (no issue)

Add your check in the same pattern. Do not reorganize existing code.

### Step 3 — Add the threshold to settings

Open `config/settings.yaml`. Under the `risk:` section, add your new threshold with a comment explaining what it does and why the default is what it is:

```yaml
risk:
  # ... existing thresholds
  max_overnight_exposure: 0.50  # Max % of portfolio held overnight.
                                # Overnight gap risk is worse than intraday.
                                # 50% means if everything gaps 3x stop, damage is bounded.
```

Never hardcode numbers in the check itself.

### Step 4 — Implement the check

Add a new method to `RiskManager`, following the existing pattern. Template:

```python
def _check_overnight_exposure(self, signal, portfolio_state):
    """
    Limit exposure held into overnight sessions.

    Protects against: overnight gap risk. Stops don't execute at the stop
    level during a gap — they execute at the next available price, which
    can be 3-5x worse than the stop distance.

    Threshold: settings.risk.max_overnight_exposure (default 50%).
    Action: if accepting this signal would push total overnight exposure
    above the threshold, reduce the size proportionally.
    """
    if not self._is_near_close():
        return None  # Only fires in last 30 min of session.

    current_overnight = sum(
        p.notional_value for p in portfolio_state.positions
        if p.holds_overnight
    )
    new_exposure = current_overnight + signal.notional_value
    max_allowed = portfolio_state.equity * self.config['max_overnight_exposure']

    if new_exposure <= max_allowed:
        return None  # No issue, approve as-is.

    # Reduce size to fit under the cap.
    available = max(0, max_allowed - current_overnight)
    if available < self.config['min_position_usd']:
        return RiskDecision(
            approved=False,
            rejection_reason=(
                f"Overnight exposure cap: would be {new_exposure:.0f}, "
                f"max is {max_allowed:.0f}. Blocked."
            ),
        )

    modified = signal.replace(notional_value=available)
    return RiskDecision(
        approved=True,
        modified_signal=modified,
        modifications=[
            f"Overnight size reduced from {signal.notional_value:.0f} "
            f"to {available:.0f} (overnight cap)"
        ],
    )
```

Then wire it into `validate_signal()` in the same order the existing checks run. Order matters — fast rejections first, size modifications last.

### Step 5 — Write the test

Add to `tests/test_risk.py`. Every risk check needs at minimum three tests:

1. **Normal case** — signal passes, no modification
2. **Threshold exceeded** — signal is modified or rejected correctly
3. **Edge case** — boundary value (exactly at the threshold)

```python
def test_overnight_exposure_below_cap_passes(risk_manager, sample_portfolio):
    """Signal below the overnight cap should pass unchanged."""
    signal = make_signal(symbol='SPY', notional=5000)
    sample_portfolio.positions = [make_position(notional=20000, overnight=True)]

    decision = risk_manager._check_overnight_exposure(signal, sample_portfolio)

    assert decision is None  # no modification


def test_overnight_exposure_above_cap_reduces_size(risk_manager, sample_portfolio):
    """Signal that would breach the cap should be size-reduced."""
    sample_portfolio.equity = 100_000
    sample_portfolio.positions = [make_position(notional=40_000, overnight=True)]
    signal = make_signal(symbol='QQQ', notional=20_000)

    decision = risk_manager._check_overnight_exposure(signal, sample_portfolio)

    assert decision.approved
    assert decision.modified_signal.notional_value == 10_000  # 50k cap - 40k existing
    assert 'overnight cap' in decision.modifications[0]


def test_overnight_exposure_at_cap_rejects(risk_manager, sample_portfolio):
    """At exactly the cap, new trades should be rejected."""
    sample_portfolio.equity = 100_000
    sample_portfolio.positions = [make_position(notional=50_000, overnight=True)]
    signal = make_signal(symbol='QQQ', notional=5_000)

    decision = risk_manager._check_overnight_exposure(signal, sample_portfolio)

    assert not decision.approved
    assert 'Overnight exposure cap' in decision.rejection_reason
```

### Step 6 — Run the risk test suite yourself

Run the full risk suite, not just your new test. Handle the environment:

- Install pytest if missing (`pip install pytest` or whatever package manager is used)
- Adapt the path if tests live somewhere other than `tests/`
- If imports fail, install the project in editable mode (`pip install -e .`) or adjust PYTHONPATH

Command to try first:

```bash
pytest tests/test_risk.py -v
```

Your new test must pass AND existing tests must still pass. If an existing test fails, your new check is rejecting or modifying something it shouldn't. Fix the check, not the old test.

Report results in plain English: how many passed, any failures, and likely cause if failed.

### Step 7 — Log it

Every risk decision is already logged by `validate_signal()`. Confirm your new check produces a structured log entry that includes:

- The check name
- The threshold value
- The actual value that triggered it
- Whether it rejected, modified, or passed

### Step 8 — Update the README

The `Risk Management` section of README.md should list every check the system performs. Add yours with a one-line description.

## Common mistakes

- **Hardcoding thresholds** — always config-driven. Future you will thank present you.
- **Modifying in-place** — `signal` and `portfolio_state` must not be mutated. Return a new signal if you're reducing size.
- **Wrong ordering** — a fast rejection (e.g., no stop loss) should fire before a slow computation (e.g., 60-day correlation). Don't put your new slow check first.
- **Not handling the None case** — if the check doesn't apply to this signal, return `None`, not `RiskDecision(approved=True)`. The `None` convention lets `validate_signal()` know there was nothing to say.
- **Testing only the happy path** — write the failure case FIRST, then the happy path. If you only test pass-through, you haven't really tested the check.

## Do not

- Do not add checks that can be bypassed by passing a flag. Risk checks are absolute.
- Do not add a check without a test. The PR is not complete without one.
- Do not relax an existing threshold to accommodate a new strategy. Add a new rule, don't weaken an old one.
