"""
capital_allocator.py — Multi-strategy capital allocation engine.

The allocator decides what FRACTION of total capital each active strategy
receives.  It does NOT override individual strategy signals — it only updates
each strategy's ``allocated_capital`` so the strategy's own rebalance logic
converges to the new budget organically.

ALLOCATION APPROACHES
---------------------
equal_weight
    Each enabled strategy gets 1/N.  Good baseline, zero parameters.

inverse_vol  (default)
    weight_i = (1/σ_i) / Σ(1/σ_j)  where σ is 60-day realised vol of daily
    returns.  Strategies with low vol get more capital.  This is a simplified
    risk-parity that ignores cross-strategy correlation; accurate enough for
    small portfolios (≤5 strategies).

risk_parity
    True risk parity: equalises each strategy's marginal contribution to total
    portfolio variance.  Solved via scipy.optimize.  Only meaningfully better
    than inverse_vol with 5+ strategies and a stable covariance estimate.

performance_weighted
    weight_i = max(sharpe_i, 0) / Σ max(sharpe_j, 0).
    Falls back to equal weight if all Sharpes are ≤ 0.
    WARNING: susceptible to overfitting short windows.

CORRELATION MERGING
-------------------
Before computing weights, a 60-day rolling correlation matrix is built from
strategy daily returns.  Any pair with ρ > 0.80 is "merged": their raw weights
are summed and split equally, effectively treating them as one strategy for
diversification purposes.  Each merge is logged as a structured event.

CONSTRAINTS
-----------
* weight_min / weight_max per strategy (from settings.yaml [strategies] block)
* Sum of weights = (1 − reserve) of total capital  (reserve defaults to 10 %)
* Clipping applied after the approach-specific step; weights renormalized after.

KILL SWITCH
-----------
Applied AFTER constraint clipping, gated on portfolio-level daily drawdown:
  daily_dd > 2 %  →  halve all allocations (rest of session)
  daily_dd > 3 %  →  zero all allocations (rest of session)
These operate on TOTAL portfolio P&L, independent of per-strategy risk managers.

REBALANCE SCHEDULE
------------------
rebalance() is called by the main loop.  It only applies changes when any
weight shifts by > rebalance_threshold (default 5 % absolute).
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Thresholds ─────────────────────────────────────────────────────────────────

_CORR_MERGE_THRESHOLD: float = 0.80
_KILL_HALVE_DD: float = 0.02   # daily DD > 2 % → halve
_KILL_ZERO_DD: float = 0.03    # daily DD > 3 % → zero
_VOL_WINDOW: int = 60
_SHARPE_WINDOW: int = 60
_ANN_FACTOR: float = 252 ** 0.5
_MIN_HISTORY: int = 2          # minimum observations to compute vol/Sharpe


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class AllocationChange:
    """
    Records one strategy's weight/capital change produced by a rebalance.

    ``reason`` is a short human-readable string explaining which constraint or
    approach produced the change (e.g. "inverse_vol rebalance", "kill_switch_halve").
    """
    strategy_name: str
    old_weight: float
    new_weight: float
    old_capital: float
    new_capital: float
    reason: str


@dataclass
class _StrategyMeta:
    """Internal bookkeeping for one strategy inside the allocator."""
    name: str
    weight_min: float = 0.0
    weight_max: float = 1.0
    current_weight: float = 0.0


# ── CapitalAllocator ───────────────────────────────────────────────────────────

class CapitalAllocator:
    """
    Compute and apply capital allocations across registered strategies.

    Parameters
    ----------
    approach
        One of: ``"equal_weight"``, ``"inverse_vol"`` (default),
        ``"risk_parity"``, ``"performance_weighted"``.
    strategy_configs
        Mapping ``{strategy_name: {weight_min, weight_max, ...}}`` sourced from
        the ``[strategies]`` block in settings.yaml.  Falls back to
        ``{weight_min: 0, weight_max: 1}`` for unlisted strategies.
    total_capital
        Total portfolio equity in dollars.  Updated on each rebalance call.
    reserve
        Fraction of capital always held as cash (default 0.10 = 10 %).
    rebalance_threshold
        Minimum absolute weight change required to trigger a rebalance for a
        given strategy (default 0.05 = 5 %).
    """

    def __init__(
        self,
        approach: str = "inverse_vol",
        strategy_configs: Optional[Dict[str, Dict]] = None,
        total_capital: float = 100_000.0,
        reserve: float = 0.10,
        rebalance_threshold: float = 0.05,
    ) -> None:
        valid = {"equal_weight", "inverse_vol", "risk_parity", "performance_weighted"}
        if approach not in valid:
            raise ValueError(f"approach must be one of {valid}, got '{approach}'")

        self.approach = approach
        self.strategy_configs: Dict[str, Dict] = strategy_configs or {}
        self.total_capital = total_capital
        self.reserve = reserve
        self.rebalance_threshold = rebalance_threshold

        # strategy_name → current weight (fraction of deployable capital)
        self._current_weights: Dict[str, float] = {}

    # ── Public API ─────────────────────────────────────────────────────────────

    def allocate(
        self,
        registry,
        daily_drawdown: float = 0.0,
    ) -> Dict[str, float]:
        """
        Compute target weights for every active strategy in *registry*.

        Parameters
        ----------
        registry
            A :class:`~core.strategy_registry.StrategyRegistry` instance.
        daily_drawdown
            Portfolio-level daily drawdown as a positive fraction (e.g. 0.025).
            Used to apply the kill switch.

        Returns
        -------
        ``{strategy_name: weight}`` where weights sum to ≤ 1 − reserve.
        """
        active = registry.active()
        if not active:
            return {}

        # ── 1. Correlation merge ───────────────────────────────────────────────
        merged_groups = self._merge_correlated(active)
        # Build effective strategy list (one representative per group)
        effective: Dict[str, List[str]] = {}  # rep_name → [member_names]
        assigned: set = set()
        for name in active:
            if name in assigned:
                continue
            group = [name]
            for partner in merged_groups.get(name, []):
                if partner not in assigned:
                    group.append(partner)
                    assigned.add(partner)
            assigned.add(name)
            effective[name] = group

        # ── 2. Raw weights from chosen approach ────────────────────────────────
        raw = self._compute_raw_weights(effective, active)

        # ── 3. Constraint clipping ─────────────────────────────────────────────
        clipped = self._apply_constraints(raw, effective)

        # ── 4. Kill switch ─────────────────────────────────────────────────────
        kill_reason: Optional[str] = None
        if daily_drawdown > _KILL_ZERO_DD:
            clipped = {k: 0.0 for k in clipped}
            kill_reason = f"kill_switch_zero (dd={daily_drawdown:.2%})"
        elif daily_drawdown > _KILL_HALVE_DD:
            clipped = {k: v * 0.5 for k, v in clipped.items()}
            kill_reason = f"kill_switch_halve (dd={daily_drawdown:.2%})"

        # ── 5. Expand merged groups back to individual strategies ──────────────
        final: Dict[str, float] = {}
        for rep, members in effective.items():
            share = clipped.get(rep, 0.0) / max(len(members), 1)
            for m in members:
                final[m] = round(share, 6)

        self._log_allocation(final, kill_reason)
        return final

    def rebalance(
        self,
        registry,
        total_capital: float,
        daily_drawdown: float = 0.0,
    ) -> List[AllocationChange]:
        """
        Compare current allocations to freshly computed targets and apply changes.

        A change is applied only when the absolute weight deviation exceeds
        ``rebalance_threshold``.  Applying means updating each strategy's
        ``allocated_capital`` attribute.

        Returns the list of :class:`AllocationChange` records (may be empty).
        """
        self.total_capital = total_capital
        target = self.allocate(registry, daily_drawdown=daily_drawdown)
        changes: List[AllocationChange] = []

        for name, strategy in registry.active().items():
            old_w = self._current_weights.get(name, 0.0)
            new_w = target.get(name, 0.0)

            if abs(new_w - old_w) < self.rebalance_threshold:
                continue

            # Weights from allocate() already bake in the reserve
            # (they sum to 1 − reserve), so multiply by total_capital directly.
            old_cap = old_w * total_capital
            new_cap = new_w * total_capital

            reason = self.approach
            if daily_drawdown > _KILL_ZERO_DD:
                reason = "kill_switch_zero"
            elif daily_drawdown > _KILL_HALVE_DD:
                reason = "kill_switch_halve"

            strategy.allocated_capital = new_cap
            self._current_weights[name] = new_w

            changes.append(AllocationChange(
                strategy_name=name,
                old_weight=round(old_w, 6),
                new_weight=round(new_w, 6),
                old_capital=round(old_cap, 2),
                new_capital=round(new_cap, 2),
                reason=reason,
            ))
            logger.info(
                "Rebalance '%s': %.1f%% → %.1f%% ($%.0f → $%.0f) [%s]",
                name, old_w * 100, new_w * 100, old_cap, new_cap, reason,
            )

        return changes

    def compute_correlation_matrix(self, registry) -> pd.DataFrame:
        """
        Build a 60-day rolling correlation matrix of strategy daily returns.

        Strategies with fewer than ``_MIN_HISTORY`` observations are excluded.
        Returns an empty DataFrame if fewer than 2 strategies have history.
        """
        active = registry.active()
        series: Dict[str, pd.Series] = {}
        for name, s in active.items():
            if len(s.performance_history) >= _MIN_HISTORY:
                idx = [ts for ts, _ in s.performance_history]
                vals = [r for _, r in s.performance_history]
                series[name] = pd.Series(vals, index=idx)

        if len(series) < 2:
            return pd.DataFrame()

        df = pd.DataFrame(series)
        return df.corr()

    def should_merge_correlated_strategies(self, registry) -> List[Tuple[str, str]]:
        """
        Return list of (name_a, name_b) pairs whose 60-day correlation > 0.80.
        """
        corr = self.compute_correlation_matrix(registry)
        if corr.empty:
            return []
        pairs: List[Tuple[str, str]] = []
        cols = corr.columns.tolist()
        for i, a in enumerate(cols):
            for b in cols[i + 1:]:
                if corr.loc[a, b] > _CORR_MERGE_THRESHOLD:
                    pairs.append((a, b))
        return pairs

    # ── Private: approach-specific weight computation ──────────────────────────

    def _compute_raw_weights(
        self,
        effective: Dict[str, List[str]],
        active: Dict[str, object],
    ) -> Dict[str, float]:
        """Return unnormalised-then-normalised weights for each representative."""
        reps = list(effective.keys())

        if self.approach == "equal_weight":
            raw = {r: 1.0 for r in reps}

        elif self.approach == "inverse_vol":
            raw = {}
            for r in reps:
                vol = self._group_vol(effective[r], active)
                raw[r] = (1.0 / vol) if vol > 1e-10 else 1.0

        elif self.approach == "risk_parity":
            raw = self._risk_parity_weights(effective, active)

        elif self.approach == "performance_weighted":
            raw = {}
            for r in reps:
                sh = self._group_sharpe(effective[r], active)
                raw[r] = max(sh, 0.0)
            if sum(raw.values()) < 1e-10:
                # All Sharpes ≤ 0 → fall back to equal weight
                raw = {r: 1.0 for r in reps}
                logger.warning(
                    "All strategy Sharpes ≤ 0; falling back to equal weight."
                )
        else:
            raw = {r: 1.0 for r in reps}

        return self._normalise(raw)

    def _apply_constraints(
        self,
        weights: Dict[str, float],
        effective: Dict[str, List[str]],
    ) -> Dict[str, float]:
        """
        Clip each representative weight to [weight_min, weight_max] then
        renormalise so the sum fits within (1 − reserve).

        We iterate up to 10 times because clipping one weight perturbs others
        after renormalisation — the loop converges quickly in practice.
        """
        deployable_fraction = 1.0 - self.reserve

        for _ in range(10):
            # Pull per-strategy bounds; representatives inherit from first member.
            clipped: Dict[str, float] = {}
            for rep, members in effective.items():
                cfg = self.strategy_configs.get(rep, {})
                lo = cfg.get("weight_min", 0.0)
                hi = cfg.get("weight_max", 1.0)
                # Scale bounds to deployable fraction
                lo_d = lo / deployable_fraction if deployable_fraction > 0 else lo
                hi_d = hi / deployable_fraction if deployable_fraction > 0 else hi
                clipped[rep] = max(lo_d, min(hi_d, weights.get(rep, 0.0)))

            total = sum(clipped.values())
            if total < 1e-10:
                break
            # Scale so that weights × deployable_fraction sum to deployable_fraction
            scale = deployable_fraction / total
            weights = {k: v * scale for k, v in clipped.items()}

        # Final bounds pass without renorm — just hard-clip
        final: Dict[str, float] = {}
        for rep, members in effective.items():
            cfg = self.strategy_configs.get(rep, {})
            lo = cfg.get("weight_min", 0.0)
            hi = cfg.get("weight_max", 1.0)
            final[rep] = max(lo, min(hi, weights.get(rep, 0.0)))

        return final

    def _normalise(self, raw: Dict[str, float]) -> Dict[str, float]:
        total = sum(raw.values())
        if total < 1e-10:
            n = max(len(raw), 1)
            return {k: 1.0 / n for k in raw}
        return {k: v / total for k, v in raw.items()}

    # ── Private: helpers ───────────────────────────────────────────────────────

    def _get_returns(self, strategy) -> List[float]:
        """Extract the plain return values from a strategy's performance_history."""
        return [r for _, r in strategy.performance_history]

    def _group_vol(self, members: List[str], active: Dict[str, object]) -> float:
        """Pooled realised vol for a group (average across members with history)."""
        vols = []
        for m in members:
            s = active.get(m)
            if s is None:
                continue
            rets = self._get_returns(s)[-_VOL_WINDOW:]
            if len(rets) >= _MIN_HISTORY:
                vols.append(float(np.std(rets, ddof=1)) * _ANN_FACTOR)
        if not vols:
            return 1.0   # fallback: treat as average vol so inverse_vol = 1
        return float(np.mean(vols))

    def _group_sharpe(self, members: List[str], active: Dict[str, object]) -> float:
        """Average 60-day Sharpe across group members."""
        sharpes = []
        for m in members:
            s = active.get(m)
            if s is None:
                continue
            rets = self._get_returns(s)[-_SHARPE_WINDOW:]
            if len(rets) >= _MIN_HISTORY:
                mu = float(np.mean(rets))
                sd = float(np.std(rets, ddof=1))
                sharpes.append((mu / sd * _ANN_FACTOR) if sd > 1e-10 else 0.0)
        return float(np.mean(sharpes)) if sharpes else 0.0

    def _merge_correlated(self, active: Dict[str, object]) -> Dict[str, List[str]]:
        """
        Return {strategy_name: [correlated_partners]} for pairs with ρ > 0.80.

        Logs every detected merge.
        """
        series: Dict[str, pd.Series] = {}
        for name, s in active.items():
            rets = self._get_returns(s)
            if len(rets) >= _MIN_HISTORY:
                series[name] = pd.Series(rets)

        merged: Dict[str, List[str]] = {}
        if len(series) < 2:
            return merged

        names = list(series.keys())
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                sa = series[a].iloc[-_VOL_WINDOW:]
                sb = series[b].iloc[-_VOL_WINDOW:]
                aligned = pd.concat([sa, sb], axis=1).dropna()
                if len(aligned) < _MIN_HISTORY:
                    continue
                corr = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
                if corr > _CORR_MERGE_THRESHOLD:
                    merged.setdefault(a, []).append(b)
                    logger.warning(
                        "Correlation merge: '%s' ↔ '%s'  ρ=%.3f > %.2f"
                        " — treating as single strategy for allocation.",
                        a, b, corr, _CORR_MERGE_THRESHOLD,
                    )
        return merged

    def _risk_parity_weights(
        self,
        effective: Dict[str, List[str]],
        active: Dict[str, object],
    ) -> Dict[str, float]:
        """
        True risk parity via scipy.optimize.minimize.

        Each strategy's marginal contribution to total portfolio variance is
        equalised.  Falls back to inverse_vol when scipy is unavailable or
        the optimiser fails.
        """
        try:
            from scipy.optimize import minimize
        except ImportError:
            logger.warning("scipy not available; falling back to inverse_vol.")
            return self._fallback_inverse_vol(effective, active)

        reps = list(effective.keys())
        n = len(reps)

        # Build covariance matrix from representative return series
        series_list = []
        for r in reps:
            vol = self._group_vol(effective[r], active)
            series_list.append(vol / _ANN_FACTOR)   # daily vol proxy

        sigma = np.diag([s ** 2 for s in series_list])   # diagonal fallback

        # Try full covariance if enough history
        aligned_series = {}
        for r in reps:
            s = active.get(r)
            if s and len(s.performance_history) >= _MIN_HISTORY:
                aligned_series[r] = pd.Series(
                    [ret for _, ret in s.performance_history]
                )
        if len(aligned_series) == n:
            df = pd.DataFrame(aligned_series).iloc[-_VOL_WINDOW:]
            if len(df) >= _MIN_HISTORY:
                sigma = df.cov().values

        def _portfolio_vol(w: np.ndarray) -> float:
            return float(np.sqrt(w @ sigma @ w))

        def _risk_contrib(w: np.ndarray) -> np.ndarray:
            pv = _portfolio_vol(w)
            if pv < 1e-10:
                return np.zeros(n)
            return w * (sigma @ w) / pv

        def _objective(w: np.ndarray) -> float:
            rc = _risk_contrib(w)
            target = np.ones(n) / n
            return float(np.sum((rc - target * rc.sum()) ** 2))

        w0 = np.ones(n) / n
        bounds = [(0.0, 1.0)] * n
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        try:
            result = minimize(
                _objective, w0, method="SLSQP",
                bounds=bounds, constraints=constraints,
                options={"ftol": 1e-9, "maxiter": 1000},
            )
            if result.success:
                raw = {r: float(result.x[i]) for i, r in enumerate(reps)}
                return raw
        except Exception as exc:
            logger.warning("Risk parity optimiser failed (%s); using inverse_vol.", exc)

        return self._fallback_inverse_vol(effective, active)

    def _fallback_inverse_vol(
        self,
        effective: Dict[str, List[str]],
        active: Dict[str, object],
    ) -> Dict[str, float]:
        reps = list(effective.keys())
        raw = {}
        for r in reps:
            vol = self._group_vol(effective[r], active)
            raw[r] = (1.0 / vol) if vol > 1e-10 else 1.0
        return self._normalise(raw)

    def _log_allocation(
        self,
        weights: Dict[str, float],
        kill_reason: Optional[str],
    ) -> None:
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "approach": self.approach,
            "final_weights": {k: round(v, 6) for k, v in weights.items()},
            "weight_sum": round(sum(weights.values()), 6),
            "reserve": self.reserve,
            "kill_switch": kill_reason,
        }
        logger.info("allocation_event %s", json.dumps(event))
