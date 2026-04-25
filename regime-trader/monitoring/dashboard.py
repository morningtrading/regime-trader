"""
dashboard.py -- Live terminal dashboard powered by Rich.

Full-width vertical layout (one column, seven rows):

  ┌─ Header ──────────────────────────────────────────────────────────┐
  │ Regime Trader  │  2024-01-15  14:32:01  │  Market: OPEN          │
  ├─ Regime ──────────────────────────────────────────────────────────┤
  │ BULL (72%)  │  Stability: 14 bars  │  Flicker: 1/20  │  STABLE   │
  ├─ Portfolio ────────────────────────────────────────────────────────┤
  │ Equity: $105,230  │  Daily: +$340 (+0.32%)                        │
  │ Allocation: 95%   │  Leverage: 1.25x  │  Positions: 3            │
  ├─ Positions ────────────────────────────────────────────────────────┤
  │ SPY  │ LONG │ $520.30  │ +1.2%  │ Stop: $508.00  │  3h           │
  ├─ Recent Signals ───────────────────────────────────────────────────┤
  │ 14:30  SPY  Rebalance 60%→95%  Low vol                            │
  ├─ Risk Status ──────────────────────────────────────────────────────┤
  │ Daily DD:   0.30% / 3.00%  ✅   │  From Peak:  1.20% / 10.00% ✅ │
  │ Weekly DD:  1.20% / 7.00%  ✅   │                                  │
  ├─ System ───────────────────────────────────────────────────────────┤
  │ Data: ✅  │  API: ✅ 23ms  │  HMM: 2d ago  │  PAPER              │
  └────────────────────────────────────────────────────────────────────┘

Refresh every *refresh_seconds* (default 5).  Run in a background daemon
thread so the main trading loop is never blocked.

Call update() from any thread to push new data.  Call push_signal() to
append a one-line entry to the Recent Signals panel.
"""

from __future__ import annotations

import datetime as dt
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from broker.position_tracker import PortfolioSnapshot
from core.signal_generator import PortfolioSignal

# DrawdownState imported lazily to avoid potential circular deps at module load
try:
    from core.risk_manager import DrawdownState
except ImportError:
    DrawdownState = None  # type: ignore[assignment,misc]


# ── Regime colour map ─────────────────────────────────────────────────────────
_REGIME_COLOURS: dict = {
    "CRASH":       "bright_red",
    "STRONG_BEAR": "red",
    "BEAR":        "red",
    "WEAK_BEAR":   "dark_orange",
    "NEUTRAL":     "yellow",
    "WEAK_BULL":   "chartreuse1",
    "BULL":        "green",
    "STRONG_BULL": "bright_green",
    "EUPHORIA":    "cyan",
}

_MAX_EVENTS  = 20
_MAX_SIGNALS = 10

# Risk thresholds (mirror core/risk_manager.py CircuitBreakerType)
_DAILY_HALT_PCT  = 0.03   # 3 %
_WEEKLY_HALT_PCT = 0.07   # 7 %
_PEAK_HALT_PCT   = 0.10   # 10 %


# ── SystemStatus dataclass ────────────────────────────────────────────────────

@dataclass
class SystemStatus:
    """
    Live health indicators for the System panel.

    Attributes
    ----------
    data_ok          : True if the market-data feed is responding.
    api_ok           : True if the broker API is reachable.
    api_latency_ms   : Last round-trip latency to the broker API (ms).
    hmm_last_trained : UTC datetime of the most recent HMM retrain.
    mode             : "PAPER" or "LIVE".
    """
    data_ok:          bool                  = True
    api_ok:           bool                  = True
    api_latency_ms:   float                 = 0.0
    hmm_last_trained: Optional[dt.datetime] = None
    mode:             str                   = "PAPER"


# ── Small helpers ─────────────────────────────────────────────────────────────

def _fmt_hold_time(holding_days: float) -> str:
    """Convert fractional days to a compact human string: 3m, 14h, 5d."""
    hours = holding_days * 24.0
    if hours < 1.0:
        return f"{int(hours * 60)}m"
    if hours < 48.0:
        return f"{int(hours)}h"
    return f"{int(holding_days)}d"


def _fmt_hmm_age(trained: Optional[dt.datetime]) -> str:
    """Return '2d ago', '4h ago', 'never', etc."""
    if trained is None:
        return "never"
    delta = dt.datetime.now() - trained
    hours = delta.total_seconds() / 3600
    if hours < 1:
        return f"{int(delta.total_seconds() / 60)}m ago"
    if hours < 48:
        return f"{int(hours)}h ago"
    return f"{int(hours / 24)}d ago"


def _risk_bar(current_abs: float, halt_pct: float) -> Text:
    """
    Render a color-coded drawdown meter:
      green  ✅  <  50 % of halt threshold
      yellow ⚠   50–80 % of halt threshold
      red    ✗   ≥  80 % of halt threshold
    """
    ratio = current_abs / halt_pct if halt_pct > 0 else 0.0
    if ratio >= 1.0:
        style, icon = "bold red",    "✗"
    elif ratio >= 0.8:
        style, icon = "red",         "⚠"
    elif ratio >= 0.5:
        style, icon = "yellow",      "⚠"
    else:
        style, icon = "green",       "✅"
    label = f"{current_abs:.2%} / {halt_pct:.0%}  {icon}"
    return Text(label, style=style)


# ── Dashboard ─────────────────────────────────────────────────────────────────

class Dashboard:
    """
    Full-width terminal dashboard with seven stacked panels.

    Parameters
    ----------
    refresh_seconds : Redraw interval (default 5 s).
    title           : Strategy name shown in the header.
    daily_halt_pct  : Daily DD halt threshold for risk bars (default 3 %).
    weekly_halt_pct : Weekly DD halt threshold (default 7 %).
    peak_halt_pct   : Peak DD halt threshold (default 10 %).
    """

    def __init__(
        self,
        refresh_seconds: int   = 5,
        title:           str   = "Regime Trader",
        daily_halt_pct:  float = _DAILY_HALT_PCT,
        weekly_halt_pct: float = _WEEKLY_HALT_PCT,
        peak_halt_pct:   float = _PEAK_HALT_PCT,
    ) -> None:
        self.refresh_seconds  = refresh_seconds
        self.title            = title
        self.daily_halt_pct   = daily_halt_pct
        self.weekly_halt_pct  = weekly_halt_pct
        self.peak_halt_pct    = peak_halt_pct

        self._console  = Console()
        self._live:    Optional[Live]   = None
        self._layout:  Optional[Layout] = None
        self._running: bool             = False
        self._thread:  Optional[threading.Thread] = None
        self._lock     = threading.Lock()

        # ── Data state ────────────────────────────────────────────────────
        self._snapshot:       Optional[PortfolioSnapshot] = None
        self._signal:         Optional[PortfolioSignal]   = None
        self._system:         Optional[SystemStatus]      = None
        self._drawdown        = None            # DrawdownState or None
        self._stability_bars: Optional[int]    = None
        self._flicker_rate:   int              = 0
        self._flicker_window: int              = 20
        self._daily_pnl:      Optional[float]  = None   # $ amount
        self._daily_pnl_pct:  Optional[float]  = None   # fraction
        self._allocation_pct: Optional[float]  = None   # gross exposure fraction

        self._recent_events:  Deque[str] = deque(maxlen=_MAX_EVENTS)
        self._recent_signals: Deque[str] = deque(maxlen=_MAX_SIGNALS)
        self._market_open:    Optional[bool] = None   # None → fall back to time check
        self._asset_group:    str            = ""
        self._symbols:        List[str]      = []

        # ── Multi-strategy allocation state ───────────────────────────────────
        # {strategy_name: {"weight": float, "sharpe": float, "healthy": bool}}
        # Special key "_reserve": {"weight": float}
        self._alloc_info: dict = {}

        # ── Countdown targets (UTC-aware datetimes) ───────────────────────────
        self._next_bar_dt:    Optional[dt.datetime] = None
        self._next_broker_dt: Optional[dt.datetime] = None
        self._next_hmm_dt:    Optional[dt.datetime] = None
        self._timeframe:      str                   = ""

    # ======================================================================= #
    # Lifecycle                                                                #
    # ======================================================================= #

    def start(self) -> None:
        """Start the live dashboard in a background daemon thread."""
        if self._running:
            return
        self._layout  = self._build_layout()
        self._running = True
        self._thread  = threading.Thread(
            target=self._refresh_loop, daemon=True, name="dashboard",
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the dashboard and restore the terminal."""
        self._running = False
        if self._live:
            self._live.stop()
        if self._thread:
            self._thread.join(timeout=3.0)

    # ======================================================================= #
    # Data updates                                                             #
    # ======================================================================= #

    def update(
        self,
        snapshot:        Optional[PortfolioSnapshot] = None,
        signal:          Optional[PortfolioSignal]   = None,
        event:           Optional[str]               = None,
        stability_bars:  Optional[int]               = None,
        flicker_rate:    Optional[int]               = None,
        flicker_window:  Optional[int]               = None,
        daily_pnl:       Optional[float]             = None,
        daily_pnl_pct:   Optional[float]             = None,
        allocation_pct:  Optional[float]             = None,
        drawdown_state                               = None,
        system_status:   Optional[SystemStatus]      = None,
        market_open:     Optional[bool]              = None,
        next_bar_dt:     Optional[dt.datetime]       = None,
        next_broker_dt:  Optional[dt.datetime]       = None,
        next_hmm_dt:     Optional[dt.datetime]       = None,
        timeframe:       Optional[str]               = None,
        asset_group:     Optional[str]               = None,
        symbols:         Optional[List[str]]         = None,
        alloc_info:      Optional[dict]              = None,
    ) -> None:
        """
        Push new data to the dashboard from any thread.

        Parameters
        ----------
        snapshot       : Latest PortfolioSnapshot (positions + equity).
        signal         : Latest PortfolioSignal (regime + weights).
        event          : One-line string appended to the Recent Events log.
        stability_bars : Consecutive bars the current regime has been confirmed.
        flicker_rate   : Regime changes in the last *flicker_window* bars.
        flicker_window : Window size for the flicker denominator (default 20).
        daily_pnl      : Daily P&L in dollars.
        daily_pnl_pct  : Daily P&L as a fraction (e.g. 0.0032 = +0.32 %).
        allocation_pct : Gross portfolio exposure as a fraction.
        drawdown_state : DrawdownState instance from core.risk_manager.
        system_status  : SystemStatus dataclass.
        """
        with self._lock:
            if snapshot       is not None: self._snapshot       = snapshot
            if signal         is not None: self._signal         = signal
            if stability_bars is not None: self._stability_bars = stability_bars
            if flicker_rate   is not None: self._flicker_rate   = flicker_rate
            if flicker_window is not None: self._flicker_window = flicker_window
            if daily_pnl      is not None: self._daily_pnl      = daily_pnl
            if daily_pnl_pct  is not None: self._daily_pnl_pct  = daily_pnl_pct
            if allocation_pct is not None: self._allocation_pct = allocation_pct
            if drawdown_state is not None: self._drawdown        = drawdown_state
            if system_status  is not None: self._system         = system_status
            if market_open    is not None: self._market_open    = market_open
            if next_bar_dt    is not None: self._next_bar_dt    = next_bar_dt
            if next_broker_dt is not None: self._next_broker_dt = next_broker_dt
            if next_hmm_dt    is not None: self._next_hmm_dt    = next_hmm_dt
            if timeframe      is not None: self._timeframe      = timeframe
            if asset_group    is not None: self._asset_group    = asset_group
            if symbols        is not None: self._symbols        = symbols
            if alloc_info     is not None: self._alloc_info     = alloc_info
            if event is not None:
                ts = dt.datetime.now().strftime("%H:%M:%S")
                self._recent_events.appendleft(f"[dim]{ts}[/dim]  {event}")

        if self._live and self._running:
            self._live.update(self.render())

    def push_signal(self, signal_line: str) -> None:
        """
        Append a one-line entry to the Recent Signals panel.

        Format the string before calling, e.g.:
          dashboard.push_signal("14:30  SPY  Rebalance 60%→95%  Low vol")
        """
        with self._lock:
            self._recent_signals.appendleft(signal_line)
        if self._live and self._running:
            self._live.update(self.render())

    # ======================================================================= #
    # Rendering                                                                #
    # ======================================================================= #

    def render(self) -> Layout:
        """Build and return the complete layout tree for one frame."""
        with self._lock:
            snap      = self._snapshot
            sig       = self._signal
            sys_      = self._system
            dd        = self._drawdown
            stab      = self._stability_bars
            flicker_r = self._flicker_rate
            flicker_w = self._flicker_window
            dpnl      = self._daily_pnl
            dpnl_pct  = self._daily_pnl_pct
            alloc     = self._allocation_pct
            events    = list(self._recent_events)
            signals   = list(self._recent_signals)

        layout = self._layout or self._build_layout()
        layout["header"].update(self._render_header())
        layout["regime"].update(
            self._render_regime(sig, stab, flicker_r, flicker_w))
        layout["portfolio"].update(
            self._render_portfolio(snap, sig, dpnl, dpnl_pct, alloc))
        layout["positions"].update(
            self._render_positions(snap))
        layout["recent_signals"].update(
            self._render_signal_log(signals))
        layout["allocations"].update(
            self._render_allocations())
        layout["risk_status"].update(
            self._render_risk_status(dd))
        layout["system"].update(
            self._render_system(sys_))
        layout["countdowns"].update(
            self._render_countdowns())
        return layout

    # ======================================================================= #
    # Panel renderers                                                          #
    # ======================================================================= #

    def _render_header(self) -> Panel:
        local_now = dt.datetime.now()
        local_str = local_now.strftime("%Y-%m-%d  %H:%M:%S")
        try:
            import zoneinfo
            et_now  = dt.datetime.now(zoneinfo.ZoneInfo("America/New_York"))
            et_str  = et_now.strftime("%H:%M:%S") + " ET"
        except Exception:
            et_str  = ""
        is_open = self._market_open if self._market_open is not None else self._market_open_fallback()
        market  = "[green]OPEN[/green]" if is_open else "[red]CLOSED[/red]"
        parts = [
            (f"  {self.title}  ", "bold cyan"),
            ("|  Bot: ",          "dim"),
            (f"{local_str}  ",    ""),
        ]
        if et_str:
            parts.append(("|  Market Time: ", "dim"))
            parts.append((f"{et_str}  ",      ""))
        parts.append(("|  Market: ", "dim"))
        parts.append(Text.from_markup(market))
        text = Text.assemble(*parts)
        return Panel(text, style="bold", padding=(0, 1))

    # ── Regime ────────────────────────────────────────────────────────────────

    def _render_regime(
        self,
        sig:       Optional[PortfolioSignal],
        stab_bars: Optional[int],
        flicker_r: int,
        flicker_w: int,
    ) -> Panel:
        if sig is None:
            return Panel("[dim]Waiting for regime data...[/dim]",
                         title="[bold]Regime[/bold]")

        colour  = _REGIME_COLOURS.get(sig.regime, "white")
        conf    = f"{sig.confidence:.0%}"
        stable  = (
            "[green]STABLE[/green]" if sig.is_stable else "[yellow]PENDING[/yellow]"
        )
        stab_str = (
            f"Stability: {stab_bars} bars" if stab_bars is not None
            else ("Stability: confirmed" if sig.is_stable else "Stability: pending")
        )

        body = Text.assemble(
            (f"  {sig.regime} ({conf})", f"bold {colour}"),
            ("   │   ", "dim"),
            (stab_str, ""),
            ("   │   ", "dim"),
            (f"Flicker: {flicker_r}/{flicker_w}", ""),
            ("   │   ", "dim"),
            Text.from_markup(stable),
        )
        if sig.notes:
            notes_rendered = []
            for note in sig.notes[:3]:
                if "awaiting first live" in note.lower() and self._next_bar_dt is not None:
                    now_utc = dt.datetime.now(dt.timezone.utc)
                    t = self._next_bar_dt if self._next_bar_dt.tzinfo else \
                        self._next_bar_dt.replace(tzinfo=dt.timezone.utc)
                    secs = max(0.0, (t - now_utc).total_seconds())
                    m, s = divmod(int(secs), 60)
                    h, m = divmod(m, 60)
                    cd   = f"{h}h {m:02d}m {s:02d}s" if h else f"{m:02d}m {s:02d}s"
                    notes_rendered.append(f"{note}  [bold]({cd})[/bold]")
                else:
                    notes_rendered.append(note)
            body.append("\n  ")
            body.append_text(Text.from_markup("  [dim]│[/dim]  ".join(notes_rendered)))
        if self._symbols:
            parts_line = ["  "]
            if self._asset_group:
                parts_line.append(f"[bold]{self._asset_group}[/bold]  [dim]│[/dim]  ")
            parts_line.append("[dim]" + "  │  ".join(self._symbols) + "[/dim]")
            body.append("\n")
            body.append_text(Text.from_markup("".join(parts_line)))
        return Panel(body, title="[bold]Regime[/bold]", border_style=colour,
                     padding=(0, 1))

    # ── Portfolio ─────────────────────────────────────────────────────────────

    def _render_portfolio(
        self,
        snap:     Optional[PortfolioSnapshot],
        sig:      Optional[PortfolioSignal],
        dpnl:     Optional[float],
        dpnl_pct: Optional[float],
        alloc:    Optional[float],
    ) -> Panel:
        if snap is None:
            return Panel("[dim]Waiting for portfolio data...[/dim]",
                         title="[bold]Portfolio[/bold]")

        # Daily P&L
        if dpnl is not None and dpnl_pct is not None:
            pnl_sign  = "+" if dpnl >= 0 else ""
            pnl_style = "green" if dpnl >= 0 else "red"
            pnl_str   = f"[{pnl_style}]{pnl_sign}${abs(dpnl):,.0f} ({pnl_sign}{dpnl_pct:.2%})[/{pnl_style}]"
        else:
            pnl_str = "[dim]--[/dim]"

        # Allocation — prefer explicit override, fall back to gross_exposure
        alloc_val  = alloc if alloc is not None else snap.gross_exposure
        alloc_str  = f"{alloc_val:.0%}"
        lev_str    = f"{sig.leverage:.2f}x" if sig is not None else "--"

        line1 = Text.assemble(
            (f"  Equity: ${snap.total_equity:>12,.2f}", "bold"),
            ("   │   Daily: ", "dim"),
            Text.from_markup(pnl_str),
        )
        line2 = Text.assemble(
            (f"  Allocation: {alloc_str}", ""),
            ("   │   Leverage: ", "dim"),
            (lev_str, "bold"),
            ("   │   Positions: ", "dim"),
            (str(len(snap.positions)), "bold"),
            ("   │   Peak DD: ", "dim"),
            (f"{snap.drawdown_from_peak:.2%}",
             "green" if snap.drawdown_from_peak > -0.02
             else "yellow" if snap.drawdown_from_peak > -0.05 else "red"),
        )
        body = Text()
        body.append_text(line1)
        body.append("\n")
        body.append_text(line2)
        return Panel(body, title="[bold]Portfolio[/bold]", padding=(0, 1))

    # ── Positions ─────────────────────────────────────────────────────────────

    def _render_positions(self, snap: Optional[PortfolioSnapshot]) -> Table:
        table = Table(
            title="Positions",
            show_header=True,
            header_style="bold yellow",
            expand=True,
            padding=(0, 1),
        )
        for col, justify in [
            ("Symbol",  "left"),
            ("Side",    "center"),
            ("Price",   "right"),
            ("P&L %",   "right"),
            ("Stop",    "right"),
            ("Weight",  "right"),
            ("Held",    "right"),
        ]:
            table.add_column(col, justify=justify)

        if snap is None or not snap.positions:
            table.add_row("--", "--", "--", "--", "--", "--", "--")
            return table

        for pos in snap.positions:
            pnl_style = "green" if pos.unrealized_pnl_pct >= 0 else "red"
            pnl_sign  = "+" if pos.unrealized_pnl_pct >= 0 else ""
            side      = "LONG" if pos.qty > 0 else "SHORT"
            stop_str  = (
                f"${pos.stop_level:.2f}" if pos.stop_level is not None else "--"
            )
            table.add_row(
                f"[bold]{pos.symbol}[/bold]",
                side,
                f"${pos.current_price:.2f}",
                f"[{pnl_style}]{pnl_sign}{pos.unrealized_pnl_pct:.2%}[/{pnl_style}]",
                stop_str,
                f"{pos.weight:.1%}",
                _fmt_hold_time(pos.holding_days),
            )
        return table

    # ── Recent Signals ────────────────────────────────────────────────────────

    def _render_signal_log(self, signals: List[str]) -> Panel:
        if not signals:
            body = Text("[dim]No signals yet[/dim]")
        else:
            body = Text.from_markup("\n".join(signals))
        return Panel(body, title="[bold]Recent Signals[/bold]",
                     border_style="dim", padding=(0, 1))

    # ── Risk Status ───────────────────────────────────────────────────────────

    def _render_risk_status(self, dd) -> Panel:
        """
        Color-coded drawdown bars.
        green ✅  < 50 % of halt threshold
        yellow ⚠  50–80 %
        red ✗     ≥ 80 %
        """
        if dd is None:
            return Panel("[dim]No drawdown data[/dim]",
                         title="[bold]Risk Status[/bold]")

        daily_abs  = abs(getattr(dd, "daily_dd",    0.0))
        weekly_abs = abs(getattr(dd, "weekly_dd",   0.0))
        peak_abs   = abs(getattr(dd, "dd_from_peak", 0.0))

        daily_bar  = _risk_bar(daily_abs,  self.daily_halt_pct)
        weekly_bar = _risk_bar(weekly_abs, self.weekly_halt_pct)
        peak_bar   = _risk_bar(peak_abs,   self.peak_halt_pct)

        sep = Text("   │   ", style="dim")

        line1 = Text.assemble(
            ("  Daily DD:   ", ""),
            daily_bar,
            sep,
            ("From Peak:  ", ""),
            peak_bar,
        )
        line2 = Text.assemble(
            ("  Weekly DD:  ", ""),
            weekly_bar,
        )
        body = Text()
        body.append_text(line1)
        body.append("\n")
        body.append_text(line2)
        return Panel(body, title="[bold]Risk Status[/bold]", padding=(0, 1))

    # ── System ────────────────────────────────────────────────────────────────

    def _render_system(self, sys_: Optional[SystemStatus]) -> Panel:
        if sys_ is None:
            return Panel("[dim]No system data[/dim]",
                         title="[bold]System[/bold]")

        data_icon = "[green]✅[/green]" if sys_.data_ok  else "[red]✗[/red]"
        api_icon  = "[green]✅[/green]" if sys_.api_ok   else "[red]✗[/red]"
        mode_style = "bold red" if sys_.mode == "LIVE" else "bold yellow"
        lat_str    = f"{sys_.api_latency_ms:.0f}ms" if sys_.api_ok else "--"
        hmm_str    = _fmt_hmm_age(sys_.hmm_last_trained)

        body = Text.assemble(
            ("  Data: ", ""),
            Text.from_markup(data_icon),
            ("   │   API: ", "dim"),
            Text.from_markup(api_icon),
            (f" {lat_str}", ""),
            ("   │   HMM: ", "dim"),
            (hmm_str, ""),
            ("   │   Mode: ", "dim"),
            (sys_.mode, mode_style),
        )
        return Panel(body, title="[bold]System[/bold]", padding=(0, 1))

    # ── Multi-strategy allocations ────────────────────────────────────────────

    def _render_allocations(self) -> Panel:
        """
        Render the multi-strategy allocations panel.

        Shows per-strategy weight, Sharpe, and health status.
        Hidden (minimal placeholder) when no alloc_info is set.
        """
        info = self._alloc_info
        if not info:
            return Panel(
                "[dim]Single-strategy mode[/dim]",
                title="[bold]Allocations[/bold]",
                padding=(0, 1),
            )

        table = Table.grid(padding=(0, 2))
        table.add_column("name",   no_wrap=True, min_width=18)
        table.add_column("weight", no_wrap=True, min_width=7, justify="right")
        table.add_column("sharpe", no_wrap=True, min_width=10)
        table.add_column("health", no_wrap=True, min_width=12)

        for name, entry in info.items():
            if name == "_reserve":
                table.add_row(
                    Text("Cash Reserve", style="dim"),
                    Text(f"{entry.get('weight', 0):.0%}", style="dim"),
                    Text("", style=""),
                    Text("", style=""),
                )
                continue

            weight  = entry.get("weight", 0.0)
            sharpe  = entry.get("sharpe", None)
            healthy = entry.get("healthy", True)

            name_style   = "bold" if healthy else "dim"
            weight_style = "green" if weight >= 0.20 else ("yellow" if weight > 0 else "dim")

            if sharpe is None:
                sharpe_text = Text("--", style="dim")
            elif sharpe >= 0.5:
                sharpe_text = Text(f"Sharpe {sharpe:+.1f}", style="green")
            elif sharpe >= 0.0:
                sharpe_text = Text(f"Sharpe {sharpe:+.1f}", style="yellow")
            else:
                sharpe_text = Text(f"Sharpe {sharpe:+.1f}", style="red")

            if healthy:
                health_text = Text("Healthy ✅", style="green")
            else:
                health_text = Text("Disabled ⚠️", style="red")

            table.add_row(
                Text(name, style=name_style),
                Text(f"{weight:.0%}", style=weight_style),
                sharpe_text,
                health_text,
            )

        return Panel(table, title="[bold]Multi-Strat Allocations[/bold]", padding=(0, 1))

    # ── Countdowns ────────────────────────────────────────────────────────────

    def _render_countdowns(self) -> Panel:
        now_utc = dt.datetime.now(dt.timezone.utc)

        def _fmt_cd(target: Optional[dt.datetime]) -> str:
            if target is None:
                return "[dim]--[/dim]"
            # ensure comparison is tz-aware
            t = target if target.tzinfo else target.replace(tzinfo=dt.timezone.utc)
            secs = max(0.0, (t - now_utc).total_seconds())
            m, s = divmod(int(secs), 60)
            h, m = divmod(m, 60)
            if h:
                return f"[bold]{h}h {m:02d}m {s:02d}s[/bold]"
            return f"[bold]{m:02d}m {s:02d}s[/bold]"

        # Market open / close — derive UTC times using zoneinfo so DST is correct
        try:
            import zoneinfo
            _et = zoneinfo.ZoneInfo("America/New_York")
            now_et      = now_utc.astimezone(_et)
            today_open  = now_et.replace(hour=9,  minute=30, second=0, microsecond=0).astimezone(dt.timezone.utc)
            today_close = now_et.replace(hour=16, minute=0,  second=0, microsecond=0).astimezone(dt.timezone.utc)
        except Exception:
            # Fallback: assume EDT (UTC-4)
            offset = dt.timezone(dt.timedelta(hours=-4))
            now_et      = now_utc.astimezone(offset)
            today_open  = now_et.replace(hour=9,  minute=30, second=0, microsecond=0).astimezone(dt.timezone.utc)
            today_close = now_et.replace(hour=16, minute=0,  second=0, microsecond=0).astimezone(dt.timezone.utc)

        is_open       = self._market_open if self._market_open is not None else self._market_open_fallback()
        market_target = today_close if is_open else today_open
        if not is_open and now_utc >= today_open:
            # After today's open has passed while closed → next trading day open
            market_target = today_open + dt.timedelta(days=1)
            while market_target.weekday() >= 5:
                market_target += dt.timedelta(days=1)
        market_label  = "Market close" if is_open else "Market open"
        market_colour = "red" if is_open else "green"

        tf_label = f" ({self._timeframe})" if self._timeframe else ""
        body = Text.assemble(
            ("  Next bar", ""),
            (tf_label, "dim"),
            (":  ", "dim"),
            Text.from_markup(_fmt_cd(self._next_bar_dt)),
            ("   │   Next broker sync:  ", "dim"),
            Text.from_markup(_fmt_cd(self._next_broker_dt)),
            ("   │   Next HMM:  ", "dim"),
            Text.from_markup(_fmt_cd(self._next_hmm_dt)),
            ("   │   ", "dim"),
            (f"{market_label}:  ", market_colour),
            Text.from_markup(_fmt_cd(market_target)),
        )
        return Panel(body, title="[bold]Countdowns[/bold]", padding=(0, 1))

    # ======================================================================= #
    # Layout skeleton                                                          #
    # ======================================================================= #

    def _build_layout(self) -> Layout:
        layout = Layout(name="root")
        layout.split_column(
            Layout(name="header",         size=3),
            Layout(name="regime",         size=5),
            Layout(name="portfolio",      size=5),
            Layout(name="positions",      minimum_size=5),
            Layout(name="recent_signals", size=8),
            Layout(name="allocations",    size=6),
            Layout(name="risk_status",    size=5),
            Layout(name="system",         size=4),
            Layout(name="countdowns",     size=3),
        )
        return layout

    # ======================================================================= #
    # Background refresh loop                                                  #
    # ======================================================================= #

    def _refresh_loop(self) -> None:
        with Live(
            self.render(),
            console            = self._console,
            refresh_per_second = max(1, 1 // self.refresh_seconds),
            screen             = True,
        ) as live:
            self._live = live
            while self._running:
                live.update(self.render())
                time.sleep(self.refresh_seconds)
        self._live = None

    # ======================================================================= #
    # Helpers                                                                  #
    # ======================================================================= #

    @staticmethod
    def _market_open_fallback() -> bool:
        """Best-effort fallback: Mon–Fri 09:30–16:00 US Eastern time."""
        try:
            import zoneinfo
            et = zoneinfo.ZoneInfo("America/New_York")
            now = dt.datetime.now(et)
        except Exception:
            now = dt.datetime.now()
        if now.weekday() >= 5:
            return False
        t = now.time()
        return dt.time(9, 30) <= t <= dt.time(16, 0)
