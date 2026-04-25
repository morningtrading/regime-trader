# Multi-Strategy Mode

The multi-strategy stack runs several `BaseStrategy` implementations in parallel,
sizes their capital with a `CapitalAllocator`, and gates aggregate exposure with
a `PortfolioRiskManager`. This page covers the contract every piece honours, how
to add a new strategy, and the failure modes worth watching.

---

## Overview

```
StrategyRegistry           CapitalAllocator                PortfolioRiskManager
──────────────────         ─────────────────────           ──────────────────────
register("alpha", S1)      approach: inverse_vol           max_aggregate_exposure (0.80)
register("beta",  S2)      correlation merger (>=0.80)     max_single_symbol      (0.50)
run_health_checks()  ───►  reserve = 0.10 cash       ───►  max_portfolio_leverage (1.25)
       │                   weights = allocate(registry)    daily_dd_halt / peak_dd
       ▼                          │                                 │
   _last_health                   ▼                                 ▼
                          per-strategy USD budget         validate_signal() per signal
                                                          (rejects or trims size)
```

Each bar:

1. `StrategyRegistry.run_health_checks()` reads each strategy's
   `performance_history` deque (drawdown, Sharpe, consecutive losses) and flips
   `is_enabled = False` on any breach.
2. `CapitalAllocator.allocate(registry)` returns a `{name: weight}` dict whose
   values sum to **at most 0.90** (the default 0.10 cash reserve).
3. The orchestrator asks each enabled strategy for signals, then routes every
   signal through the per-strategy `RiskManager` and then the
   `PortfolioRiskManager` before sending to the broker.
4. After each fill, `PortfolioRiskManager.update_strategy_positions(name, dict)`
   refreshes the aggregate-exposure view used by the next signal.

---

## Adding a strategy

A strategy is any class that inherits `core.regime_strategies.BaseStrategy` and
implements `generate_signal(bar, regime, portfolio_state) -> Optional[Signal]`.

1. **Subclass `BaseStrategy`** in a new module under `core/`:

   ```python
   from core.regime_strategies import BaseStrategy, Signal

   class MyStrategy(BaseStrategy):
       def generate_signal(self, bar, regime, ps):
           if my_entry_condition(bar):
               return Signal(symbol=bar.symbol, is_long=True,
                             entry_price=bar.close, stop_loss=bar.close * 0.97,
                             position_size_pct=0.10, leverage=1.0)
           return None
   ```

2. **Register it** in `main.py`'s strategy bootstrap block (search for
   `StrategyRegistry.instance().register`).

3. **Add a config entry** in `config/settings.yaml` under `strategies:`:

   ```yaml
   strategies:
     my_strategy:
       enabled: true
       symbols: [SPY, QQQ]
       weight_min: 0.05    # allocator floor
       weight_max: 0.30    # allocator ceiling
   ```

4. **Record returns** every bar via `self.record_daily_return(ts, ret)` so the
   allocator's vol/Sharpe estimates and the registry's health checks have data.

5. **Test it** — add at minimum one unit test under `tests/` and run the
   E2E suite (`tests/test_multistrat_e2e.py`) to verify it composes cleanly.

---

## Allocator approaches

Set on the CLI with `--allocator <name>` or in code via
`CapitalAllocator(approach=...)`. All approaches respect `weight_min` /
`weight_max` per strategy and the global cash reserve (default 0.10).

| Approach | Sizes by | When to use |
|---|---|---|
| `equal_weight` | `0.90 / N` for each enabled strategy | Sanity baseline; you don't trust your vol or Sharpe estimates yet. |
| `inverse_vol` | 1 / annualised stdev of returns | Default. Penalises high-vol strategies; works with as few as ~20 bars. |
| `risk_parity` | Iterative — each strategy contributes equal risk to portfolio variance, accounting for the correlation matrix | Many uncorrelated strategies. Slower; needs ~60+ bars per strategy. |
| `performance_weighted` | Recent Sharpe ratio (clipped at zero) | Strategies with stable, comparable Sharpes. Decays slowly — not for short windows. |

**Correlation merger.** Before sizing, the allocator computes pairwise return
correlations. Pairs above `_CORR_MERGE_THRESHOLD` (default `0.80`) are treated
as a single merged group: the group gets one slot's worth of weight, then
splits it evenly between members. This stops two near-duplicate strategies
from doubling your effective exposure to the same factor.

---

## Portfolio risk caps

`PortfolioRiskManager` runs after each strategy's own `RiskManager` and gates
aggregate behaviour. Defaults (override in `config/settings.yaml` under the
`risk:` block):

| Cap | Default | Behaviour on breach |
|---|---|---|
| `max_aggregate_exposure` | 0.80 | Trim the new signal to fit the headroom; reject if headroom < `min_position_usd`. |
| `max_single_symbol` | 0.50 | Trim cross-strategy exposure on the same symbol to the cap. |
| `max_portfolio_leverage` | 1.25 | Reject new long+short combined gross above the cap. |
| `daily_dd_halt` | 0.03 | Hard reject all signals for the rest of the session. |
| `max_dd_from_peak` | 0.10 | Write the lock file and hard-halt; manual restart required. |

The PRM holds its own `_strategy_positions` dict — tests and post-fill code
must call `update_strategy_positions(name, {symbol: usd})` so the aggregate
view is correct on the next signal.

---

## Failure modes & how to debug

- **A strategy never gets weight.** Check `strategy._last_health` after a
  health-check pass; the registry sets it on the instance. Common causes:
  drawdown > 15 %, Sharpe < -1.0, or 10+ consecutive losing returns.

- **Two "different" strategies act like one.** The correlation merger has
  detected `corr >= 0.80` and is splitting one slot between them. This is
  intentional. If they really are independent, check whether they share a
  signal source or a stop logic that reacts to the same input.

- **The allocator returns less than 0.90 total.** Some strategy hit its
  `weight_max` ceiling and the surplus is held as additional cash, or one or
  more strategies were disabled by health checks. Both are correct behaviour.

- **PRM rejects every signal in a session.** Check for the lock file at
  `core.risk_manager.LOCK_FILE` — a peak-DD halt was tripped and needs manual
  removal before trading resumes.

- **Live alerts fire repeatedly.** The `AlertManager` rate-limits per type;
  if you're seeing duplicates, the alert key (e.g. strategy name) is
  changing each call. Pin the key to something stable.

---

## FAQ

**Why does the allocator default to a 10 % cash reserve?**
To leave room for the per-strategy `RiskManager` to round position sizes up
without breaching the aggregate cap, and to fund stop-loss slippage.

**Can I run the multi-strategy stack in backtest?**
Yes — `backtest/multi_strategy_backtester.py` runs the full registry +
allocator + PRM loop on historical bars. `_BacktestProxy` is the minimal
`BaseStrategy` instance used by tests and by the multi-strat backtester for
scenarios where you only have a returns series, not a live signal generator.

**Do per-strategy `weight_min` floors override the cash reserve?**
No. If `sum(weight_min)` exceeds `1.0 - reserve`, the allocator scales floors
proportionally so the reserve is preserved.

**What's the relationship between `position_size_pct` on a signal and the
allocator weight?**
The allocator weight is the **budget** for that strategy (fraction of total
equity it may deploy). The signal's `position_size_pct` is the strategy's
request for a single trade, expressed as a fraction of equity. The PRM trims
the signal if it would push aggregate gross above the cap.
