# Performance Recovery Plan — Phase 5/6 Regression

## Problem Statement

**Before Phase 5 (2026-04-22 22:47):** Backtest return = **66.82%**, Sharpe = 1.83, MaxDD = -10.03%

**After Phase 6 (2026-04-23 09:30):** Backtest return = **18.87%**, Sharpe = 1.57, MaxDD = -8.47%

**Loss:** -47.95% return (catastrophic), -0.26 Sharpe.

Both backtests ran **identical date ranges** (2022-10-14 → 2026-01-21 in old run; 2022-10-14 → 2026-01-22 in new run). The new run includes one extra day (2026-01-22) where portfolio crashed from $166,820 → $118,478 **(-28.9% overnight)**.

## Root Cause Hypothesis

**Fold boundary walk-forward artifact**, not a code regression:

1. Both backtests run walk-forward with **12-month IS, 3-month OOS, step forward by 3m**.
2. On 2026-01-22, the allocator rebalances (every 5 bars) at a fold transition boundary.
3. At the fold boundary, the allocator reweights the 5 strategies based on OOS performance metrics.
4. One or more strategies get over-weighted into a losing period, causing the sharp drop.
5. **Old backtest stopped before this boundary; new backtest caught it.**

### Why it's not a code regression:
- No code changes to the backtester, allocator, or risk manager between the two runs.
- No changes to strategy entry/exit logic.
- The problem is **timing**: the extra day in the new run exposed a pre-existing fold-transition issue.

## Recovery Options

### Option A: Fix the fold boundary issue (RECOMMENDED)
- **Root cause:** At fold transitions, the allocator reweights using stale OOS performance estimates.
- **Fix:** Add a "warm-up" period after fold transitions where allocator stays frozen, or use adaptive rebalance timing to avoid cliff transitions.
- **Effort:** Medium (1-2 days analysis + code change).
- **Upside:** Fixes the underlying issue; prevents future surprises.
- **Risk:** Might reveal other problems; may need to adjust `allocate_interval` or warm-up logic.

### Option B: Exclude fold boundaries from performance measurement (NOT RECOMMENDED)
- **Fix:** In the backtest report, exclude the first N days of each OOS fold (transition period).
- **Effort:** Low (1 hour).
- **Upside:** Quickly recovers reported performance.
- **Risk:** Hides the real problem; live trading will still experience the 28% drops.

### Option C: Revert to a pre-Phase-5 commit (NUCLEAR)
- **Fix:** Go back to the last known-good commit before phases 5/6 were added.
- **Effort:** Minutes (git revert).
- **Upside:** 66% returns back immediately.
- **Risk:** Loses all Phase 5 (alerts) and Phase 6 (E2E tests) work. Not viable.

## Recommended Action: Option A

1. **Diagnose** (2-3 hours):
   - Add detailed logging to the multi-strat backtester to track allocator rebalances at fold boundaries.
   - Re-run the backtest and capture what happens on 2026-01-22 (allocator weights, capital transfers, signals).
   - Check if the 28% drop correlates with an extreme reweighting (e.g., one strategy jumps from 20% to 50%).

2. **Fix** (4-6 hours):
   - Option A1: Freeze allocator during fold transitions (first 5 bars of each OOS period).
   - Option A2: Use a running average of performance metrics instead of instantaneous reweights.
   - Option A3: Reduce `allocate_interval` from 5 bars (weekly) to reduce rebalance shock.

3. **Validate** (2-3 hours):
   - Re-run backtest and compare to baseline (66% target).
   - Ensure Sharpe and MaxDD also improve.
   - Run E2E tests to confirm no regressions in Phase 6 work.

4. **Understand** (1 hour):
   - Explain why the old run didn't trigger this (data cutoff luck) and why the new run did (extends one more day).
   - Document the fold-transition risk as a known caveat in backtest design.

## Timeline

- If Option A1 (freeze): **6 hours total** → expect 60-65% returns back.
- If Option A2 (running avg): **8 hours total** → expect 65%+ returns with smoother equity curve.
- Decision point: End of analysis phase. If diagnosis shows a different root cause, pivot.

## Next Steps

1. Add instrumentation to the backtester.
2. Run diagnostic backtest with logging enabled.
3. Decide between A1 (fastest) and A2 (best long-term fix).
4. Implement the fix.
5. Re-run full backtest suite and confirm performance recovery.
