"""
stress_test.py — Crash injection and gap simulation for the backtest engine.

Applies synthetic adverse scenarios to historical price data and re-runs the
walk-forward backtest to measure strategy resilience under extreme conditions.

SCENARIO TYPES
--------------
  crash       : Gradual price decline over N bars.  Per-bar log-return is
                log(1 + crash_pct) / duration_bars; subsequent prices are
                permanently shifted.
  gap         : Instantaneous overnight gap — all prices from bar_index
                onward are multiplied by (1 + gap_pct).
  vol_spike   : Daily log-returns are amplified by ``vol_multiplier`` over a
                window; prices are reconstructed from the amplified series.

MONTE CARLO
-----------
  ``run_monte_carlo_crashes`` injects a crash of random magnitude and at a
  random bar position, running 100 seeds and returning the distribution of
  performance outcomes.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backtest.backtester import WalkForwardBacktester, BacktestResult
from backtest.performance import PerformanceAnalyzer, PerformanceReport

logger = logging.getLogger(__name__)


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class StressScenario:
    """Definition of a single stress scenario."""
    name: str
    description: str
    crash_pct: Optional[float] = None      # e.g. -0.20 for a 20% crash
    crash_duration_bars: int = 1           # bars over which crash unfolds
    gap_pct: Optional[float] = None        # overnight gap (e.g. -0.05 = -5%)
    vol_multiplier: float = 1.0            # scale daily returns by this factor
    inject_at_bar: Optional[int] = None    # specific bar index; None = mid-point


@dataclass
class StressResult:
    """Result of running a single stress scenario."""
    scenario: StressScenario
    baseline_report: PerformanceReport
    stressed_report: PerformanceReport
    equity_delta: float              # final equity change vs baseline (fraction)
    max_dd_delta: float              # change in max drawdown (abs)
    notes: List[str] = field(default_factory=list)


# ── Main class ─────────────────────────────────────────────────────────────────

class StressTester:
    """
    Inject synthetic market shocks into historical price data and re-run the
    walk-forward backtest to measure strategy resilience.

    Parameters
    ----------
    backtester :
        A configured (but not yet run) :class:`~backtest.backtester.WalkForwardBacktester`.
    analyzer :
        :class:`~backtest.performance.PerformanceAnalyzer` for computing metrics.
    """

    # ── Built-in scenario library ──────────────────────────────────────────────

    SCENARIOS: List[StressScenario] = [
        StressScenario(
            name="flash_crash_10pct",
            description="Sudden 10% single-bar crash on all symbols",
            crash_pct=-0.10,
            crash_duration_bars=1,
        ),
        StressScenario(
            name="bear_market_30pct",
            description="Gradual 30% decline over 60 bars (~3 months)",
            crash_pct=-0.30,
            crash_duration_bars=60,
        ),
        StressScenario(
            name="gap_down_5pct",
            description="Overnight gap-down of 5% on all symbols",
            gap_pct=-0.05,
        ),
        StressScenario(
            name="gap_down_10pct",
            description="Overnight gap-down of 10% on all symbols",
            gap_pct=-0.10,
        ),
        StressScenario(
            name="vol_spike_3x",
            description="Realised volatility tripled for 21 bars (~1 month)",
            vol_multiplier=3.0,
        ),
        StressScenario(
            name="covid_crash",
            description="35% crash over 23 bars followed by elevated vol (150%)",
            crash_pct=-0.35,
            crash_duration_bars=23,
            vol_multiplier=1.5,
        ),
    ]

    def __init__(
        self,
        backtester: WalkForwardBacktester,
        analyzer: Optional[PerformanceAnalyzer] = None,
    ) -> None:
        self.backtester = backtester
        self.analyzer = analyzer or PerformanceAnalyzer()
        self._baseline: Optional[BacktestResult] = None
        self._stress_results: List[StressResult] = []

    # ── Public API ─────────────────────────────────────────────────────────────

    def run_stress_scenarios(
        self,
        prices: pd.DataFrame,
        scenarios: Optional[List[StressScenario]] = None,
        hmm_config: Optional[Dict] = None,
        strategy_config: Optional[Dict] = None,
        risk_config: Optional[Dict] = None,
    ) -> List[StressResult]:
        """
        Run each scenario and return a list of :class:`StressResult`.

        The baseline backtest is run once and cached; each scenario re-runs
        the backtester on a modified copy of ``prices``.

        Parameters
        ----------
        prices :
            Baseline wide-format close prices.
        scenarios :
            Scenarios to run; defaults to :attr:`SCENARIOS`.
        hmm_config, strategy_config, risk_config :
            Forwarded to the backtester.

        Returns
        -------
        List of :class:`StressResult`, one per scenario.
        """
        if scenarios is None:
            scenarios = self.SCENARIOS

        # ── Baseline ──────────────────────────────────────────────────────────
        logger.info("Running baseline backtest …")
        baseline = self._run_baseline(prices, hmm_config, strategy_config, risk_config)
        baseline_report = self.analyzer.analyze(baseline)

        results: List[StressResult] = []

        for scenario in scenarios:
            logger.info("Stress scenario: %s", scenario.name)
            stressed_prices = self._apply_scenario(prices, scenario)

            try:
                stressed_result = self.backtester.run(
                    stressed_prices, hmm_config, strategy_config, risk_config
                )
                stressed_report = self.analyzer.analyze(stressed_result)
            except Exception as exc:
                logger.error("Scenario '%s' failed: %s", scenario.name, exc, exc_info=True)
                continue

            baseline_final = float(baseline.combined_equity.iloc[-1])
            stressed_final = float(stressed_result.combined_equity.iloc[-1])
            equity_delta = (stressed_final - baseline_final) / max(baseline_final, 1.0)
            max_dd_delta = stressed_report.max_drawdown - baseline_report.max_drawdown

            notes: List[str] = []
            if stressed_report.max_drawdown < -0.20:
                notes.append("WARNING: max drawdown exceeds 20%")
            if stressed_report.sharpe_ratio < 0:
                notes.append("WARNING: negative Sharpe under stress")
            if equity_delta < -0.10:
                notes.append(f"Equity reduced by {equity_delta:.1%} vs baseline")

            results.append(StressResult(
                scenario=scenario,
                baseline_report=baseline_report,
                stressed_report=stressed_report,
                equity_delta=equity_delta,
                max_dd_delta=max_dd_delta,
                notes=notes,
            ))
            logger.info(
                "  %s → eq_delta=%.2f%%  dd_delta=%.2f%%  sharpe=%.2f",
                scenario.name, equity_delta * 100,
                max_dd_delta * 100, stressed_report.sharpe_ratio,
            )

        self._stress_results = results
        return results

    def run_monte_carlo_crashes(
        self,
        prices: pd.DataFrame,
        n_seeds: int = 100,
        crash_min: float = -0.05,
        crash_max: float = -0.15,
        duration_range: Tuple[int, int] = (1, 5),
        hmm_config: Optional[Dict] = None,
        strategy_config: Optional[Dict] = None,
    ) -> Dict[str, object]:
        """
        Inject random crashes (magnitude, timing, duration) and collect the
        resulting performance distribution.

        Parameters
        ----------
        prices :
            Baseline close prices.
        n_seeds :
            Number of Monte Carlo seeds.
        crash_min, crash_max :
            Uniform range for crash magnitude (both negative).
        duration_range :
            (min_bars, max_bars) for crash duration.
        hmm_config, strategy_config :
            Forwarded to the backtester.

        Returns
        -------
        Dict with keys:
            ``sharpes``      : list of Sharpe ratios
            ``max_drawdowns``: list of max drawdowns
            ``final_equities``: list of final equity values
            ``mean_sharpe``  : mean Sharpe across seeds
            ``std_sharpe``   : std dev of Sharpe across seeds
            ``p10_drawdown`` : 10th-percentile (worst) max drawdown
        """
        n_bars = len(prices)
        # Eligible injection range: avoid injecting in warmup or final test period
        bar_min = max(50, n_bars // 5)
        bar_max = max(bar_min + 1, n_bars - n_bars // 5)

        sharpes: List[float] = []
        drawdowns: List[float] = []
        equities: List[float] = []

        rng = np.random.default_rng(42)

        for seed in range(n_seeds):
            crash_pct = float(rng.uniform(crash_min, crash_max))
            inject_bar = int(rng.integers(bar_min, bar_max))
            dur_bars = int(rng.integers(duration_range[0], duration_range[1] + 1))

            stressed = self.inject_crash(prices, crash_pct, inject_bar, dur_bars)

            try:
                res = self.backtester.run(stressed, hmm_config, strategy_config, None)
                rpt = self.analyzer.analyze(res)
                sharpes.append(rpt.sharpe_ratio)
                drawdowns.append(rpt.max_drawdown)
                equities.append(res.final_equity)
            except Exception as exc:
                logger.debug("MC seed %d failed: %s", seed, exc)

        if not sharpes:
            return {}

        return {
            "sharpes":        sharpes,
            "max_drawdowns":  drawdowns,
            "final_equities": equities,
            "mean_sharpe":    float(np.mean(sharpes)),
            "std_sharpe":     float(np.std(sharpes)),
            "p10_drawdown":   float(np.percentile(drawdowns, 10)),
            "median_equity":  float(np.median(equities)),
        }

    # ── Scenario modifiers ─────────────────────────────────────────────────────

    def inject_crash(
        self,
        prices: pd.DataFrame,
        crash_pct: float,
        start_bar: int,
        duration_bars: int = 1,
    ) -> pd.DataFrame:
        """
        Inject a price crash starting at ``start_bar`` over ``duration_bars``.

        The crash is distributed via equal log-return increments so the
        cumulative decline equals ``crash_pct`` exactly.  All symbols move
        together (systemic shock).  Prices after the crash window are
        permanently shifted by the full crash magnitude.

        Parameters
        ----------
        prices :
            Wide-format close prices (unmodified original is never mutated).
        crash_pct :
            Total crash magnitude (e.g. -0.20 for a 20% decline).
        start_bar :
            First bar index affected.
        duration_bars :
            Number of bars over which the crash unfolds (minimum 1).

        Returns
        -------
        Modified copy of ``prices``.
        """
        modified = prices.copy()
        n = len(modified)
        duration_bars = max(1, duration_bars)

        # Per-bar fractional level reached: level[k] = (1 + crash_pct) ** ((k+1)/duration)
        for k in range(min(duration_bars, n - start_bar)):
            bar = start_bar + k
            frac = (k + 1) / duration_bars
            scale = (1.0 + crash_pct) ** frac
            # Original level at `bar`; scale by ratio relative to start_bar - 1
            if start_bar > 0:
                modified.iloc[bar] = prices.iloc[bar] * scale
            else:
                modified.iloc[bar] = prices.iloc[bar] * scale

        # Post-crash: permanently shifted by full crash_pct
        post = start_bar + duration_bars
        if post < n:
            modified.iloc[post:] = prices.iloc[post:].values * (1.0 + crash_pct)

        return modified

    def simulate_gap(
        self,
        prices: pd.DataFrame,
        gap_pct: float,
        bar_index: int,
    ) -> pd.DataFrame:
        """
        Introduce an overnight price gap at ``bar_index``.

        All prices from ``bar_index`` onwards are multiplied by
        ``(1 + gap_pct)``, simulating a permanent level shift (gap open).

        Parameters
        ----------
        prices :
            Wide-format close prices.
        gap_pct :
            Gap magnitude (e.g. -0.05 for a 5% gap down).
        bar_index :
            Row index where the gap occurs.

        Returns
        -------
        Modified copy of ``prices``.
        """
        modified = prices.copy()
        if 0 <= bar_index < len(modified):
            modified.iloc[bar_index:] = prices.iloc[bar_index:].values * (1.0 + gap_pct)
        return modified

    def multiply_volatility(
        self,
        prices: pd.DataFrame,
        multiplier: float,
        start_bar: int,
        duration_bars: int,
    ) -> pd.DataFrame:
        """
        Amplify daily log-returns by ``multiplier`` over a window, then
        reconstruct the price series.

        The amplification is applied column-by-column independently so
        individual symbols can have different absolute moves (only the
        *magnitude* of moves is scaled).

        Parameters
        ----------
        prices :
            Wide-format close prices.
        multiplier :
            Vol scaling factor (e.g. 3.0 triples the size of daily moves).
        start_bar :
            First bar of the vol-spike window.
        duration_bars :
            Length of the vol spike.

        Returns
        -------
        Modified copy of ``prices``.
        """
        modified = prices.copy()
        n = len(modified)
        end_bar = min(start_bar + duration_bars, n)

        for col in prices.columns:
            col_prices = prices[col].copy()
            # Log-return series
            log_ret = np.log(col_prices / col_prices.shift(1)).fillna(0.0)

            # Amplify within the window
            amplified = log_ret.copy()
            amplified.iloc[start_bar:end_bar] *= multiplier

            # Reconstruct from amplified log returns
            log_ret_arr = amplified.values
            start_price = float(col_prices.iloc[0])
            new_prices = start_price * np.exp(np.cumsum(log_ret_arr))
            modified[col] = new_prices

        return modified

    def get_stress_results(self) -> List[StressResult]:
        """Return all stress test results from the last :meth:`run_stress_scenarios` call."""
        return list(self._stress_results)

    def summary_table(self) -> pd.DataFrame:
        """
        Build a summary DataFrame comparing stressed vs baseline metrics.

        Returns
        -------
        DataFrame with one row per scenario and columns:
        ``scenario, total_return, sharpe, sortino, max_drawdown,
        calmar, equity_delta, max_dd_delta, notes``.
        """
        if not self._stress_results:
            return pd.DataFrame()

        rows = []
        for sr in self._stress_results:
            rpt = sr.stressed_report
            bsl = sr.baseline_report
            rows.append({
                "scenario":          sr.scenario.name,
                "description":       sr.scenario.description,
                "total_return":      f"{rpt.total_return:.2%}",
                "sharpe":            f"{rpt.sharpe_ratio:.2f}",
                "sortino":           f"{rpt.sortino_ratio:.2f}",
                "max_drawdown":      f"{rpt.max_drawdown:.2%}",
                "calmar":            f"{rpt.calmar_ratio:.2f}",
                "vs_baseline_equity":f"{sr.equity_delta:+.2%}",
                "vs_baseline_dd":    f"{sr.max_dd_delta:+.2%}",
                "baseline_sharpe":   f"{bsl.sharpe_ratio:.2f}",
                "notes":             " | ".join(sr.notes),
            })

        return pd.DataFrame(rows).set_index("scenario")

    # ── Private helpers ────────────────────────────────────────────────────────

    def _run_baseline(
        self,
        prices: pd.DataFrame,
        hmm_config: Optional[Dict],
        strategy_config: Optional[Dict],
        risk_config: Optional[Dict],
    ) -> BacktestResult:
        """Run the unmodified backtest and cache the result."""
        if self._baseline is not None:
            return self._baseline
        self._baseline = self.backtester.run(
            prices, hmm_config, strategy_config, risk_config
        )
        return self._baseline

    def _apply_scenario(
        self, prices: pd.DataFrame, scenario: StressScenario
    ) -> pd.DataFrame:
        """
        Apply all modifications described by ``scenario`` to ``prices``.

        The injection point defaults to the midpoint of the price history
        when ``scenario.inject_at_bar`` is None.
        """
        n = len(prices)
        inject_bar = scenario.inject_at_bar if scenario.inject_at_bar is not None \
            else n // 2

        modified = prices.copy()

        # Crash
        if scenario.crash_pct is not None:
            modified = self.inject_crash(
                modified, scenario.crash_pct, inject_bar, scenario.crash_duration_bars
            )

        # Gap
        if scenario.gap_pct is not None:
            modified = self.simulate_gap(modified, scenario.gap_pct, inject_bar)

        # Vol spike — applied after the crash so both effects compound
        if scenario.vol_multiplier != 1.0:
            vol_dur = min(21, n - inject_bar - 1)
            if vol_dur > 0:
                modified = self.multiply_volatility(
                    modified, scenario.vol_multiplier,
                    inject_bar, vol_dur,
                )

        return modified
