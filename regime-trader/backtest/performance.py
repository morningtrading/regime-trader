"""
performance.py — Performance analytics and reporting.

Computes standard quantitative metrics (Sharpe, Sortino, max drawdown, CAGR,
Calmar) plus regime-aware breakdowns and benchmark comparisons.  Outputs are
available as a :class:`PerformanceReport` dataclass, rich terminal tables, and
optionally CSV files.

METRIC DEFINITIONS
------------------
    Sharpe  = (mean_excess_daily / std_daily) × √252
    Sortino = (mean_excess_daily / downside_dev_daily) × √252
              downside_dev uses only bars where return < rfr/252
    Calmar  = CAGR / |max_drawdown|
    CAGR    = (final / initial)^(252 / n_bars) − 1
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backtest.backtester import BacktestResult

logger = logging.getLogger(__name__)


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class PerformanceReport:
    """Full performance analytics for a backtest or live trading period."""
    # Return metrics
    total_return: float
    cagr: float
    annualised_vol: float

    # Risk-adjusted metrics
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    # Drawdown
    max_drawdown: float
    max_drawdown_duration_days: int
    avg_drawdown: float

    # Trade statistics
    total_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float

    # Regime breakdown — {label: {pct_time, ann_return, ann_vol, sharpe, n_bars,
    #                              n_entries, avg_duration_bars}}
    regime_stats: Dict[str, Dict] = field(default_factory=dict)

    # Regime transitions — total count and matrix {from: {to: count}}
    regime_n_changes: int = 0
    regime_transition_matrix: Dict[str, Dict[str, int]] = field(default_factory=dict)

    # Worst-case single-period losses
    worst_day: float = 0.0
    worst_week: float = 0.0
    worst_month: float = 0.0
    max_consecutive_losses: int = 0
    longest_underwater_days: int = 0

    # Benchmark comparison
    benchmark_return: Optional[float] = None
    benchmark_sharpe: Optional[float] = None
    alpha: Optional[float] = None
    beta: Optional[float] = None
    information_ratio: Optional[float] = None


# ── Main class ─────────────────────────────────────────────────────────────────

class PerformanceAnalyzer:
    """
    Compute and report performance metrics for a backtest result or a raw
    equity curve.

    Parameters
    ----------
    risk_free_rate :
        Annualised risk-free rate (default: 0.045 = 4.5%).
    trading_days_per_year :
        Annualisation factor (default 252).
    """

    def __init__(
        self,
        risk_free_rate: float = 0.045,
        trading_days_per_year: int = 252,
    ) -> None:
        self.risk_free_rate = risk_free_rate
        self.trading_days_per_year = trading_days_per_year
        self._rfr_daily = risk_free_rate / trading_days_per_year

    # ── Top-level analysis ─────────────────────────────────────────────────────

    def analyze(
        self,
        result: BacktestResult,
        benchmark_prices: Optional[pd.Series] = None,
    ) -> PerformanceReport:
        """
        Run the full performance analysis on a :class:`~backtest.backtester.BacktestResult`.

        Parameters
        ----------
        result :
            Output from :class:`~backtest.backtester.WalkForwardBacktester`.
        benchmark_prices :
            Optional buy-and-hold benchmark close prices aligned with the
            equity curve.

        Returns
        -------
        :class:`PerformanceReport`
        """
        all_trades = [t for w in result.windows for t in w.trades]
        bench_returns: Optional[pd.Series] = None

        if benchmark_prices is not None:
            bm = benchmark_prices.reindex(result.combined_equity.index).ffill()
            bench_returns = bm.pct_change().dropna()

        return self.analyze_equity_curve(
            equity=result.combined_equity,
            trades=all_trades,
            regimes=result.combined_regimes,
            benchmark=bench_returns,
        )

    def analyze_equity_curve(
        self,
        equity: pd.Series,
        trades: Optional[List[Dict]] = None,
        regimes: Optional[pd.Series] = None,
        benchmark: Optional[pd.Series] = None,
    ) -> PerformanceReport:
        """
        Compute all metrics from a raw equity curve.

        Parameters
        ----------
        equity :
            Portfolio value indexed by date.
        trades :
            Optional list of trade dicts from :class:`~backtest.backtester.WindowResult`.
        regimes :
            Optional regime label series (must share index with ``equity``).
        benchmark :
            Optional benchmark *return* series (not price).

        Returns
        -------
        :class:`PerformanceReport`
        """
        if len(equity) < 2:
            raise ValueError("Equity curve must have at least 2 bars.")

        equity = equity.dropna().sort_index()
        returns = self._equity_to_returns(equity)

        # ── Core metrics ──────────────────────────────────────────────────────
        total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
        cagr = self.compute_cagr(equity)
        ann_vol = float(returns.std() * np.sqrt(self.trading_days_per_year))
        sharpe = self.compute_sharpe(returns)
        sortino = self.compute_sortino(returns)
        calmar = self.compute_calmar(equity)

        # ── Drawdown ──────────────────────────────────────────────────────────
        max_dd, dd_start, dd_end, dd_dur = self.compute_max_drawdown(equity)
        avg_dd = self._average_drawdown(equity)
        longest_uw = self._longest_underwater(equity)

        # ── Worst-case periods ────────────────────────────────────────────────
        worst_day = float(returns.min()) if len(returns) > 0 else 0.0
        worst_week = self._worst_period_return(equity, 5)
        worst_month = self._worst_period_return(equity, 21)
        max_consec = self._max_consecutive_losses(returns)

        # ── Trade stats ───────────────────────────────────────────────────────
        n_trades, win_rate, avg_win, avg_loss, pf = self._compute_trade_stats(
            returns, trades or []
        )

        # ── Regime breakdown + transition stats ───────────────────────────────
        regime_stats: Dict[str, Dict] = {}
        regime_n_changes   = 0
        regime_transitions: Dict[str, Dict[str, int]] = {}
        if regimes is not None and len(regimes) > 0:
            regime_stats = self.compute_regime_breakdown(returns, regimes)
            regime_n_changes, regime_transitions = self.compute_regime_transitions(regimes)

        # ── Benchmark comparison ──────────────────────────────────────────────
        bm_return = bm_sharpe = alpha = beta = ir = None
        if benchmark is not None and len(benchmark) > 1:
            aligned_ret = returns.reindex(benchmark.index).dropna()
            aligned_bm = benchmark.reindex(aligned_ret.index).dropna()
            aligned_ret = aligned_ret.reindex(aligned_bm.index)

            if len(aligned_ret) > 10:
                bm_return = float((1 + aligned_bm).prod() - 1)
                bm_sharpe = self.compute_sharpe(aligned_bm)
                bm_stats = self.compare_benchmark(aligned_ret, aligned_bm)
                alpha = bm_stats["alpha"]
                beta = bm_stats["beta"]
                ir = bm_stats["information_ratio"]

        return PerformanceReport(
            total_return=total_return,
            cagr=cagr,
            annualised_vol=ann_vol,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown=max_dd,
            max_drawdown_duration_days=dd_dur,
            avg_drawdown=avg_dd,
            total_trades=n_trades,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=pf,
            regime_stats=regime_stats,
            regime_n_changes=regime_n_changes,
            regime_transition_matrix=regime_transitions,
            worst_day=worst_day,
            worst_week=worst_week,
            worst_month=worst_month,
            max_consecutive_losses=max_consec,
            longest_underwater_days=longest_uw,
            benchmark_return=bm_return,
            benchmark_sharpe=bm_sharpe,
            alpha=alpha,
            beta=beta,
            information_ratio=ir,
        )

    # ── Individual metric methods ──────────────────────────────────────────────

    def compute_sharpe(self, returns: pd.Series) -> float:
        """Annualised Sharpe ratio: (mean_excess / std) × √252."""
        if len(returns) < 2:
            return 0.0
        excess = returns - self._rfr_daily
        std = excess.std()
        if std == 0 or np.isnan(std):
            return 0.0
        return float(excess.mean() / std * np.sqrt(self.trading_days_per_year))

    def compute_sortino(self, returns: pd.Series) -> float:
        """
        Annualised Sortino ratio using downside deviation.

        Downside deviation = RMS of returns below the risk-free rate.
        """
        if len(returns) < 2:
            return 0.0
        excess = returns - self._rfr_daily
        downside = excess[excess < 0]
        if len(downside) == 0:
            return float("inf") if excess.mean() > 0 else 0.0
        downside_dev = float(np.sqrt((downside ** 2).mean()))
        if downside_dev == 0:
            return 0.0
        return float(excess.mean() / downside_dev * np.sqrt(self.trading_days_per_year))

    def compute_max_drawdown(
        self, equity: pd.Series
    ) -> Tuple[float, pd.Timestamp, pd.Timestamp, int]:
        """
        Maximum peak-to-trough drawdown.

        Returns
        -------
        (max_drawdown, drawdown_start_date, trough_date, duration_calendar_days)
        """
        if len(equity) < 2:
            idx = equity.index[0] if len(equity) > 0 else pd.Timestamp.now()
            return 0.0, idx, idx, 0

        running_peak = equity.expanding().max()
        drawdown = (equity - running_peak) / running_peak

        max_dd = float(drawdown.min())
        if max_dd == 0.0:
            return 0.0, equity.index[0], equity.index[0], 0

        trough_date = drawdown.idxmin()
        # Peak = last time equity hit its maximum before the trough
        peak_date = equity.loc[:trough_date].idxmax()

        duration = int((trough_date - peak_date).days)
        return max_dd, peak_date, trough_date, duration

    def compute_cagr(self, equity: pd.Series) -> float:
        """Compound annual growth rate: (final/initial)^(252/n) − 1."""
        if len(equity) < 2 or equity.iloc[0] <= 0:
            return 0.0
        n_bars = len(equity)
        years = n_bars / self.trading_days_per_year
        total = equity.iloc[-1] / equity.iloc[0]
        if years <= 0 or total <= 0:
            return 0.0
        return float(total ** (1.0 / years) - 1.0)

    def compute_calmar(self, equity: pd.Series) -> float:
        """Calmar ratio = CAGR / |max_drawdown|."""
        cagr = self.compute_cagr(equity)
        max_dd, *_ = self.compute_max_drawdown(equity)
        if max_dd == 0.0:
            return float("inf") if cagr > 0 else 0.0
        return float(cagr / abs(max_dd))

    def compute_regime_breakdown(
        self,
        returns: pd.Series,
        regimes: pd.Series,
    ) -> Dict[str, Dict]:
        """
        Return statistics broken down by regime label.

        Returns
        -------
        Dict mapping ``regime_label → {pct_time, ann_return, ann_vol,
        sharpe, n_bars, n_entries, avg_duration_bars}``.
        """
        common = returns.index.intersection(regimes.index)
        if len(common) == 0:
            return {}
        ret = returns.loc[common]
        reg = regimes.loc[common]
        total_bars = len(ret)

        # Count contiguous runs per regime (entries and durations)
        entries: Dict[str, int] = {}
        durations: Dict[str, list] = {}
        current_label = None
        run_len = 0
        for lbl in reg:
            if lbl != current_label:
                if current_label is not None:
                    entries[current_label] = entries.get(current_label, 0) + 1
                    durations.setdefault(current_label, []).append(run_len)
                current_label = lbl
                run_len = 1
            else:
                run_len += 1
        if current_label is not None:
            entries[current_label] = entries.get(current_label, 0) + 1
            durations.setdefault(current_label, []).append(run_len)

        result: Dict[str, Dict] = {}
        for label in sorted(reg.unique()):
            mask = reg == label
            r = ret[mask]
            if len(r) == 0:
                continue
            pct_time = mask.sum() / total_bars
            ann_ret = float(r.mean() * self.trading_days_per_year)
            ann_vol = float(r.std() * np.sqrt(self.trading_days_per_year)) if len(r) > 1 else 0.0
            shr = self.compute_sharpe(r)
            regime_win_rate = float((r > 0).mean())
            n_ent = entries.get(label, 0)
            durs  = durations.get(label, [])
            avg_dur = float(sum(durs) / len(durs)) if durs else 0.0
            result[label] = {
                "pct_time":        float(pct_time),
                "ann_return":      ann_ret,
                "ann_vol":         ann_vol,
                "sharpe":          shr,
                "win_rate":        regime_win_rate,
                "n_bars":          int(mask.sum()),
                "n_entries":       n_ent,
                "avg_duration_bars": avg_dur,
            }
        return result

    def compute_regime_transitions(
        self,
        regimes: pd.Series,
    ) -> tuple:
        """
        Return (n_changes, transition_matrix).

        transition_matrix is a dict-of-dicts {from_label: {to_label: count}}.
        """
        if len(regimes) < 2:
            return 0, {}
        labels = regimes.values
        n_changes = int(sum(1 for i in range(1, len(labels)) if labels[i] != labels[i - 1]))
        matrix: Dict[str, Dict[str, int]] = {}
        for i in range(1, len(labels)):
            if labels[i] != labels[i - 1]:
                frm, to = str(labels[i - 1]), str(labels[i])
                matrix.setdefault(frm, {})
                matrix[frm][to] = matrix[frm].get(to, 0) + 1
        return n_changes, matrix

    def compare_benchmark(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series,
    ) -> Dict[str, float]:
        """
        Compute benchmark-relative statistics.

        Returns
        -------
        Dict with keys: ``alpha``, ``beta``, ``information_ratio``,
        ``tracking_error``, ``excess_return``.
        """
        if len(strategy_returns) < 10 or len(benchmark_returns) < 10:
            return dict(alpha=0.0, beta=1.0, information_ratio=0.0,
                        tracking_error=0.0, excess_return=0.0)

        # Align
        bm = benchmark_returns.reindex(strategy_returns.index).dropna()
        strat = strategy_returns.reindex(bm.index).dropna()
        bm = bm.reindex(strat.index)

        excess = strat - bm
        bm_var = float(bm.var())
        beta = float(strat.cov(bm) / bm_var) if bm_var > 0 else 1.0
        alpha_daily = float(strat.mean() - beta * bm.mean())
        alpha = float(alpha_daily * self.trading_days_per_year)

        tracking_error = float(excess.std() * np.sqrt(self.trading_days_per_year))
        ir = float(excess.mean() / excess.std() * np.sqrt(self.trading_days_per_year)) \
            if excess.std() > 0 else 0.0
        excess_return = float(strat.mean() - bm.mean()) * self.trading_days_per_year

        return dict(
            alpha=alpha,
            beta=beta,
            information_ratio=ir,
            tracking_error=tracking_error,
            excess_return=excess_return,
        )

    def generate_report(
        self,
        report: PerformanceReport,
        print_to_console: bool = True,
        run_context: Optional[Dict] = None,
    ) -> str:
        """
        Format a :class:`PerformanceReport` as rich terminal tables.

        Returns the plain-text version of the output.

        run_context keys (all optional):
            asset_group, symbols, symbol_descriptions, start_date, end_date,
            run_timestamp, machine, git_hash, python_version, output_path,
            n_states, train_window, test_window, step_size, n_folds
        """
        try:
            from rich.console import Console
            from rich.table import Table
            from rich import box as rbox
            _rich = True
        except ImportError:
            _rich = False

        buf = io.StringIO()
        lines: List[str] = []

        # ── Run context header ─────────────────────────────────────────────────
        if run_context:
            _W = 72
            ctx_lines = [f"{'─' * _W}"]
            ctx_lines.append("  Run Context")
            ctx_lines.append(f"{'─' * _W}")

            ag = run_context.get("asset_group") or "—"
            syms = run_context.get("symbols") or []
            descs = run_context.get("symbol_descriptions") or {}
            sym_str = ", ".join(syms) if syms else "—"
            ctx_lines.append(f"  Asset Group  : {ag}")
            ctx_lines.append(f"  Symbols      : {sym_str}")
            if descs:
                for s in syms:
                    d = descs.get(s, "")
                    if d:
                        ctx_lines.append(f"    {s:<8} {d}")

            sd = run_context.get("start_date", "")
            ed = run_context.get("end_date", "")
            if sd or ed:
                ctx_lines.append(f"  Period       : {sd}  →  {ed}")

            ts = run_context.get("run_timestamp", "")
            machine = run_context.get("machine", "")
            pyver = run_context.get("python_version", "")
            if ts:
                ctx_lines.append(f"  Timestamp    : {ts}")
            if machine or pyver:
                ctx_lines.append(f"  Machine      : {machine}  (Python {pyver})")

            git_hash = run_context.get("git_hash", "")
            if git_hash:
                ctx_lines.append(f"  Bot version  : {git_hash}")

            out_path = run_context.get("output_path", "")
            if out_path:
                ctx_lines.append(f"  Output path  : {out_path}")

            tw = run_context.get("train_window")
            tew = run_context.get("test_window")
            ss = run_context.get("step_size")
            nf = run_context.get("n_folds")
            ns = run_context.get("n_states")
            if any(x is not None for x in [tw, tew, ss]):
                ctx_lines.append(
                    f"  Walk-forward : IS {tw} / OOS {tew} bars  step {ss}"
                    + (f"  ({nf} folds)" if nf else "")
                )
            if ns:
                ctx_lines.append(f"  HMM states   : {ns}")

            ctx_lines.append(f"{'─' * _W}")
            ctx_block = "\n".join(ctx_lines)
            if print_to_console:
                print(ctx_block)
            buf.write(ctx_block + "\n")

        def _pct(v: Optional[float], decimals: int = 2) -> str:
            if v is None:
                return "N/A"
            return f"{v * 100:+.{decimals}f}%"

        def _f(v: Optional[float], decimals: int = 2) -> str:
            if v is None:
                return "N/A"
            return f"{v:.{decimals}f}"

        # ── Summary block ──────────────────────────────────────────────────────
        summary_rows = [
            ("Total Return",        _pct(report.total_return)),
            ("CAGR",                _pct(report.cagr)),
            ("Annualised Vol",      _pct(report.annualised_vol)),
            ("Sharpe Ratio",        _f(report.sharpe_ratio)),
            ("Sortino Ratio",       _f(report.sortino_ratio)),
            ("Calmar Ratio",        _f(report.calmar_ratio)),
            ("Max Drawdown",        _pct(report.max_drawdown)),
            ("Max DD Duration",     f"{report.max_drawdown_duration_days}d"),
            ("Avg Drawdown",        _pct(report.avg_drawdown)),
            ("Total Trades",        str(report.total_trades)),
            ("Win Rate",            _pct(report.win_rate)),
            ("Avg Win",             _pct(report.avg_win, 3)),
            ("Avg Loss",            _pct(report.avg_loss, 3)),
            ("Profit Factor",       _f(report.profit_factor)),
            ("Worst Day",           _pct(report.worst_day)),
            ("Worst Week",          _pct(report.worst_week)),
            ("Worst Month",         _pct(report.worst_month)),
            ("Max Consec. Losses",  str(report.max_consecutive_losses)),
            ("Longest Underwater",  f"{report.longest_underwater_days}d"),
        ]

        if report.benchmark_return is not None:
            summary_rows += [
                ("Benchmark Return",  _pct(report.benchmark_return)),
                ("Benchmark Sharpe",  _f(report.benchmark_sharpe)),
                ("Alpha",             _pct(report.alpha)),
                ("Beta",              _f(report.beta)),
                ("Information Ratio", _f(report.information_ratio)),
            ]

        if _rich:
            # force_terminal=True preserves color codes in the buffered string,
            # which we also echo to real stdout at the end when print_to_console.
            console = Console(file=buf, width=80, force_terminal=True)
            tbl = Table(title="Performance Summary", box=rbox.ROUNDED, header_style="bold cyan")
            tbl.add_column("Metric", style="dim", min_width=22)
            tbl.add_column("Value", justify="right")
            for metric, value in summary_rows:
                tbl.add_row(metric, value)
            console.print(tbl)
        else:
            lines.append("=" * 50)
            lines.append("PERFORMANCE SUMMARY")
            lines.append("=" * 50)
            for metric, value in summary_rows:
                lines.append(f"  {metric:<28} {value}")

        # ── Regime breakdown table ─────────────────────────────────────────────
        if report.regime_stats:
            # Add n_entries and avg_duration to rows
            regime_rows = [
                (
                    label,
                    _pct(s["pct_time"]),
                    _pct(s["ann_return"]),
                    _pct(s.get("ann_vol", 0.0)),
                    _f(s.get("sharpe", 0.0)),
                    _pct(s.get("win_rate", 0.0)),
                    str(s.get("n_bars", 0)),
                    str(s.get("n_entries", 0)),
                    f"{s.get('avg_duration_bars', 0.0):.1f}",
                )
                for label, s in sorted(report.regime_stats.items())
            ]

            if _rich:
                console2 = Console(file=buf, width=110)
                t2 = Table(title="Regime Breakdown", box=rbox.ROUNDED, header_style="bold magenta")
                for col in ("Regime", "% Time", "Ann. Ret", "Ann. Vol", "Sharpe",
                            "Win Rate", "Bars", "Entries", "Avg Dur"):
                    t2.add_column(col, justify="right")
                for row in regime_rows:
                    t2.add_row(*row)
                console2.print(t2)
            else:
                lines.append("")
                lines.append("REGIME BREAKDOWN")
                lines.append("-" * 80)
                hdr = (f"{'Regime':<14} {'%Time':>8} {'Ann.Ret':>9} {'Ann.Vol':>9} "
                       f"{'Sharpe':>8} {'WinRate':>9} {'Bars':>6} {'Entries':>8} {'AvgDur':>7}")
                lines.append(hdr)
                lines.append("-" * 80)
                for row in regime_rows:
                    lines.append(
                        f"{row[0]:<14} {row[1]:>8} {row[2]:>9} {row[3]:>9} "
                        f"{row[4]:>8} {row[5]:>9} {row[6]:>6} {row[7]:>8} {row[8]:>7}"
                    )

            # Regime change summary
            n_ch = report.regime_n_changes
            total_bars_all = sum(s.get("n_bars", 0) for s in report.regime_stats.values())
            changes_per_100 = (n_ch / total_bars_all * 100) if total_bars_all > 0 else 0.0
            change_line = (f"  Regime changes : {n_ch}  "
                           f"({changes_per_100:.1f} per 100 bars)")
            if _rich:
                buf.write(change_line + "\n")
            else:
                lines.append("")
                lines.append(change_line)

            # Transition matrix
            if report.regime_transition_matrix:
                all_labels = sorted(
                    set(report.regime_transition_matrix.keys()) |
                    {t for v in report.regime_transition_matrix.values() for t in v}
                )
                if _rich:
                    t3 = Table(title="Regime Transitions (from → to)", box=rbox.SIMPLE,
                               header_style="dim")
                    t3.add_column("From \\ To", style="dim")
                    for lbl in all_labels:
                        t3.add_column(lbl, justify="right")
                    for frm in all_labels:
                        row_vals = [frm] + [
                            str(report.regime_transition_matrix.get(frm, {}).get(to, 0))
                            for to in all_labels
                        ]
                        t3.add_row(*row_vals)
                    console2.print(t3)
                else:
                    lines.append("")
                    lines.append("REGIME TRANSITIONS (from → to)")
                    lines.append("-" * 60)
                    col_w = max(len(l) for l in all_labels) + 2
                    lines.append(f"{'From\\To':<14}" + "".join(f"{l:>{col_w}}" for l in all_labels))
                    lines.append("-" * 60)
                    for frm in all_labels:
                        row_str = f"{frm:<14}" + "".join(
                            f"{report.regime_transition_matrix.get(frm, {}).get(to, 0):>{col_w}}"
                            for to in all_labels
                        )
                        lines.append(row_str)

        output = buf.getvalue() + "\n".join(lines)
        if print_to_console:
            print(output)
        return output

    # ── Benchmark helpers ──────────────────────────────────────────────────────

    def compute_benchmark_bnh(
        self,
        prices: pd.Series,
        initial_capital: float = 100_000.0,
        slippage_pct: float = 0.0,
    ) -> pd.Series:
        """
        Buy-and-hold equity curve from a single-symbol price series.

        A one-time entry slippage cost is applied (buy at t=0).
        """
        ratio = prices / prices.iloc[0]
        effective = initial_capital * (1.0 - slippage_pct)
        return (ratio * effective).rename("bnh_equity")

    def compute_benchmark_sma(
        self,
        prices: pd.Series,
        sma_window: int = 200,
        initial_capital: float = 100_000.0,
        slippage_pct: float = 0.0,
    ) -> pd.Series:
        """
        SMA trend-following equity: invested when close > SMA(sma_window).

        Signals are 1-bar delayed (no look-ahead). ``slippage_pct`` is
        charged whenever the position toggles (cost = |Δpos| × slippage_pct
        subtracted from that bar's return).
        """
        sma = prices.rolling(sma_window).mean()
        invested = (prices > sma).astype(float).shift(1).fillna(0.0)
        daily_ret = prices.pct_change().fillna(0.0)
        turnover = invested.diff().abs().fillna(invested.iloc[0])
        cost = turnover * slippage_pct
        strat_ret = invested * daily_ret - cost
        equity = initial_capital * (1.0 + strat_ret).cumprod()
        return equity.rename("sma_equity")

    def compute_random_allocation_benchmark(
        self,
        returns: pd.Series,
        allocations: List[float],
        n_seeds: int = 100,
        initial_capital: float = 100_000.0,
        slippage_pct: float = 0.0,
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Monte Carlo benchmark: random allocation changes with the same
        set of allowed allocation levels.

        Parameters
        ----------
        returns :
            Daily returns of the underlying asset.
        allocations :
            Discrete allocation levels to draw from (e.g. [0.60, 0.95]).
        n_seeds :
            Number of Monte Carlo paths (default 100).
        initial_capital :
            Starting equity.

        Returns
        -------
        (mean_equity_curve, std_equity_curve)  across all seeds.
        """
        n = len(returns)
        paths = np.zeros((n_seeds, n))

        for seed in range(n_seeds):
            rng = np.random.default_rng(seed)
            alloc = rng.choice(allocations, size=n)
            # Cost when allocation changes
            d_alloc = np.abs(np.diff(alloc, prepend=alloc[0]))
            cost = d_alloc * slippage_pct
            path_ret = alloc * returns.values - cost
            paths[seed] = initial_capital * np.cumprod(1.0 + path_ret)

        mean_curve = pd.Series(paths.mean(axis=0), index=returns.index,
                               name="random_mean")
        std_curve = pd.Series(paths.std(axis=0), index=returns.index,
                              name="random_std")
        return mean_curve, std_curve

    # ── Multi-symbol benchmark helpers ────────────────────────────────────────

    def compute_benchmark_bnh_multi(
        self,
        prices_df: pd.DataFrame,
        initial_capital: float = 100_000.0,
        slippage_pct: float = 0.0,
    ) -> pd.Series:
        """
        Equal-weighted buy-and-hold equity curve across multiple symbols.

        Each symbol is normalised to 1.0 at the first observation; the
        portfolio equity is the average normalised value times initial_capital.

        Parameters
        ----------
        prices_df :
            DataFrame of close prices, one column per symbol.
        initial_capital :
            Starting equity.

        Returns
        -------
        Equity curve indexed like ``prices_df``.
        """
        normed = prices_df.div(prices_df.iloc[0])
        effective = initial_capital * (1.0 - slippage_pct)
        return (normed.mean(axis=1) * effective).rename("bnh_equity")

    def compute_benchmark_sma_multi(
        self,
        prices_df: pd.DataFrame,
        sma_window: int = 200,
        initial_capital: float = 100_000.0,
        slippage_pct: float = 0.0,
    ) -> pd.Series:
        """
        Equal-weighted SMA trend-following equity across multiple symbols.

        For each symbol the signal is: invested (weight = 1/N) when
        close > SMA(sma_window), else cash.  Signals are 1-bar delayed.
        ``slippage_pct`` is charged on each per-symbol toggle.
        """
        n_syms = prices_df.shape[1]
        daily_ret = prices_df.pct_change().fillna(0.0)
        invested = (prices_df > prices_df.rolling(sma_window).mean()) \
                       .astype(float).shift(1).fillna(0.0)
        turnover = invested.diff().abs().fillna(invested.iloc[0])
        cost = (turnover * slippage_pct).sum(axis=1) / n_syms
        port_ret = (invested * daily_ret).sum(axis=1) / n_syms - cost
        equity = initial_capital * (1.0 + port_ret).cumprod()
        return equity.rename("sma_equity")

    def compute_benchmark_ema_cross(
        self,
        prices: pd.Series,
        fast: int = 9,
        slow: int = 45,
        initial_capital: float = 100_000.0,
        slippage_pct: float = 0.0,
    ) -> pd.Series:
        """
        EMA crossover equity: long when fast EMA > slow EMA, else cash.

        Signals are 1-bar delayed. ``slippage_pct`` charged on toggles.
        """
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        invested  = (ema_fast > ema_slow).astype(float).shift(1).fillna(0.0)
        daily_ret = prices.pct_change().fillna(0.0)
        turnover  = invested.diff().abs().fillna(invested.iloc[0])
        cost      = turnover * slippage_pct
        strat_ret = invested * daily_ret - cost
        equity    = initial_capital * (1.0 + strat_ret).cumprod()
        return equity.rename("ema_cross_equity")

    def compute_benchmark_ema_cross_multi(
        self,
        prices_df: pd.DataFrame,
        fast: int = 9,
        slow: int = 45,
        initial_capital: float = 100_000.0,
        slippage_pct: float = 0.0,
    ) -> pd.Series:
        """
        Equal-weighted EMA crossover equity across multiple symbols.

        Each symbol is long when its fast EMA > slow EMA, else cash.
        Signals are 1-bar delayed (no look-ahead).

        Parameters
        ----------
        prices_df :
            DataFrame of close prices, one column per symbol.
        fast :
            Fast EMA period (default 9).
        slow :
            Slow EMA period (default 45).
        initial_capital :
            Starting equity.

        Returns
        -------
        Equity curve.
        """
        n_syms    = prices_df.shape[1]
        daily_ret = prices_df.pct_change().fillna(0.0)
        ema_fast  = prices_df.ewm(span=fast,  adjust=False).mean()
        ema_slow  = prices_df.ewm(span=slow,  adjust=False).mean()
        invested  = (ema_fast > ema_slow).astype(float).shift(1).fillna(0.0)
        turnover  = invested.diff().abs().fillna(invested.iloc[0])
        cost      = (turnover * slippage_pct).sum(axis=1) / n_syms
        port_ret  = (invested * daily_ret).sum(axis=1) / n_syms - cost
        equity    = initial_capital * (1.0 + port_ret).cumprod()
        return equity.rename("ema_cross_equity")

    def compute_random_allocation_benchmark_multi(
        self,
        prices_df: pd.DataFrame,
        allocations: List[float],
        n_seeds: int = 100,
        initial_capital: float = 100_000.0,
        slippage_pct: float = 0.0,
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Monte Carlo random-allocation benchmark on an equal-weighted universe.

        At each bar a random allocation level is drawn for the whole portfolio;
        returns are the equal-weighted mean of all symbols scaled by that level.

        Parameters
        ----------
        prices_df :
            DataFrame of close prices, one column per symbol.
        allocations :
            Discrete allocation levels to draw from (e.g. [0.60, 0.95]).
        n_seeds :
            Number of Monte Carlo paths.
        initial_capital :
            Starting equity.

        Returns
        -------
        (mean_equity_curve, std_equity_curve)
        """
        ew_returns = prices_df.pct_change().mean(axis=1).fillna(0.0)
        return self.compute_random_allocation_benchmark(
            ew_returns, allocations=allocations,
            n_seeds=n_seeds, initial_capital=initial_capital,
            slippage_pct=slippage_pct,
        )

    # ── Private helpers ────────────────────────────────────────────────────────

    def _equity_to_returns(self, equity: pd.Series) -> pd.Series:
        """Daily percentage returns from an equity curve."""
        return equity.pct_change().dropna()

    def _annualise_return(self, period_return: float, n_bars: int) -> float:
        """Scale a period return to annualised form."""
        if n_bars <= 0:
            return 0.0
        years = n_bars / self.trading_days_per_year
        return float((1.0 + period_return) ** (1.0 / years) - 1.0)

    def _compute_trade_stats(
        self,
        returns: pd.Series,
        trades: List[Dict],
    ) -> Tuple[int, float, float, float, float]:
        """
        Compute trade-level statistics.

        For an allocation-based system, a "win" is a day with positive
        return.  Profit factor = sum(positive returns) / sum(|negative|).

        Returns
        -------
        (total_trades, win_rate, avg_win, avg_loss, profit_factor)
        """
        n_trades = len(trades)

        if len(returns) == 0:
            return n_trades, 0.5, 0.0, 0.0, 1.0

        pos = returns[returns > 0]
        neg = returns[returns < 0]

        win_rate = float(len(pos) / len(returns)) if len(returns) > 0 else 0.5
        avg_win = float(pos.mean()) if len(pos) > 0 else 0.0
        avg_loss = float(neg.mean()) if len(neg) > 0 else 0.0

        total_pos = float(pos.sum())
        total_neg = float(neg.abs().sum())
        profit_factor = (total_pos / total_neg) if total_neg > 0 else float("inf")

        return n_trades, win_rate, avg_win, avg_loss, profit_factor

    def _average_drawdown(self, equity: pd.Series) -> float:
        """Mean of all non-zero drawdown values."""
        peak = equity.expanding().max()
        dd = (equity - peak) / peak
        non_zero = dd[dd < 0]
        return float(non_zero.mean()) if len(non_zero) > 0 else 0.0

    def _longest_underwater(self, equity: pd.Series) -> int:
        """
        Longest continuous stretch (calendar days) where equity was below
        its previous all-time high.
        """
        peak = equity.expanding().max()
        underwater = equity < peak

        max_streak = 0
        streak_start: Optional[pd.Timestamp] = None

        for ts, uw in underwater.items():
            if uw:
                if streak_start is None:
                    streak_start = ts
                streak_days = (ts - streak_start).days + 1
                max_streak = max(max_streak, streak_days)
            else:
                streak_start = None

        return max_streak

    def _worst_period_return(self, equity: pd.Series, window_bars: int) -> float:
        """Worst rolling ``window_bars``-bar return in the equity curve."""
        if len(equity) <= window_bars:
            return 0.0
        rolling_ret = equity.pct_change(window_bars).dropna()
        return float(rolling_ret.min()) if len(rolling_ret) > 0 else 0.0

    def _max_consecutive_losses(self, returns: pd.Series) -> int:
        """Maximum number of consecutive losing bars."""
        max_streak = 0
        streak = 0
        for r in returns:
            if r < 0:
                streak += 1
                max_streak = max(max_streak, streak)
            else:
                streak = 0
        return max_streak
