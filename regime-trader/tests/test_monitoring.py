"""
tests/test_monitoring.py -- Unit tests for the monitoring package.

Coverage
--------
  monitoring/logger.py    TradeLogger: JSON format, context injection,
                          four-file routing, idempotent setup
  monitoring/dashboard.py formatting helpers (_fmt_hold_time, _fmt_hmm_age,
                          _risk_bar), panel renderers, push_signal, update()
  monitoring/alerts.py    rate limiting, 7 trigger methods, delivery channels,
                          console_enabled flag, backward-compat method
"""

from __future__ import annotations

import datetime as dt
import io
import json
import logging
import types
import unittest.mock as mock
from pathlib import Path
from typing import Optional

import pytest
from rich.console import Console

# ── subjects under test ──────────────────────────────────────────────────────
from monitoring.logger import (
    TradeLogger,
    _JsonFormatter,
    _EventFilter,
    _TRADES_EVENTS,
    _ALERTS_EVENTS,
    _REGIME_EVENTS,
)
from monitoring.dashboard import (
    Dashboard,
    SystemStatus,
    _fmt_hold_time,
    _fmt_hmm_age,
    _risk_bar,
)
from monitoring.alerts import AlertManager


# ── helpers ───────────────────────────────────────────────────────────────────

def _render_to_str(renderable) -> str:
    """Render a Rich renderable to a plain string (no ANSI colour)."""
    buf = io.StringIO()
    console = Console(file=buf, no_color=True, width=120)
    console.print(renderable)
    return buf.getvalue()


def _read_jsonlines(path: Path) -> list:
    """Read a log file and return a list of parsed JSON dicts."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _make_logger(tmp_path: Path, console: bool = False) -> TradeLogger:
    tl = TradeLogger(log_dir=str(tmp_path), console=console)
    tl.setup()
    return tl


def _flush_logger(tl: TradeLogger) -> None:
    """Force all handlers to flush their write buffers."""
    if tl._logger:
        for h in tl._logger.handlers:
            h.flush()


def _clear_logger() -> None:
    """Remove all handlers from the shared 'regime_trader' logger."""
    log = logging.getLogger("regime_trader")
    for h in list(log.handlers):
        h.close()
        log.removeHandler(h)


@pytest.fixture(autouse=True)
def reset_logger():
    """Ensure the shared logger is clean before and after each test."""
    _clear_logger()
    yield
    _clear_logger()


# ═══════════════════════════════════════════════════════════════════════════════
# logger.py
# ═══════════════════════════════════════════════════════════════════════════════

class TestJsonFormatter:

    def test_required_top_level_keys(self):
        """Every formatted record must have ts, level, logger, message."""
        fmt    = _JsonFormatter()
        record = logging.LogRecord("test", logging.INFO, "", 0, "hello", (), None)
        obj    = json.loads(fmt.format(record))
        for key in ("ts", "level", "logger", "message"):
            assert key in obj, f"missing key: {key}"

    def test_timestamp_is_utc_iso(self):
        """ts field must be an ISO-8601 string ending in Z."""
        fmt    = _JsonFormatter()
        record = logging.LogRecord("test", logging.INFO, "", 0, "x", (), None)
        ts     = json.loads(fmt.format(record))["ts"]
        assert ts.endswith("Z")
        dt.datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S.%fZ")   # must not raise

    def test_extra_fields_merged(self):
        """extra_fields dict on the record must appear at the top level."""
        fmt    = _JsonFormatter()
        record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
        record.extra_fields = {"event": "trade", "symbol": "SPY"}  # type: ignore
        obj    = json.loads(fmt.format(record))
        assert obj["event"]  == "trade"
        assert obj["symbol"] == "SPY"

    def test_exception_serialised(self):
        """exc_info should produce a 'traceback' key."""
        fmt = _JsonFormatter()
        try:
            raise ValueError("boom")
        except ValueError:
            import sys
            record = logging.LogRecord("test", logging.ERROR, "", 0, "err",
                                       (), sys.exc_info())
        obj = json.loads(fmt.format(record))
        assert "traceback" in obj
        assert "ValueError" in obj["traceback"]


class TestEventFilter:

    def _record(self, event: str) -> logging.LogRecord:
        r = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
        r.extra_fields = {"event": event}  # type: ignore
        return r

    def _record_no_event(self) -> logging.LogRecord:
        return logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)

    def test_accepts_matching_event(self):
        f = _EventFilter(_TRADES_EVENTS)
        assert f.filter(self._record("trade"))
        assert f.filter(self._record("fill"))

    def test_rejects_non_matching_event(self):
        f = _EventFilter(_TRADES_EVENTS)
        assert not f.filter(self._record("regime_change"))
        assert not f.filter(self._record("risk_event"))

    def test_rejects_record_without_extra_fields(self):
        f = _EventFilter(_TRADES_EVENTS)
        assert not f.filter(self._record_no_event())

    def test_regime_filter_accepts_correct_events(self):
        f = _EventFilter(_REGIME_EVENTS)
        assert f.filter(self._record("regime_change"))
        assert f.filter(self._record("rebalance"))
        assert not f.filter(self._record("trade"))

    def test_alerts_filter_accepts_correct_events(self):
        f = _EventFilter(_ALERTS_EVENTS)
        assert f.filter(self._record("risk_event"))
        assert f.filter(self._record("error"))
        assert not f.filter(self._record("fill"))


class TestTradeLogger:

    def test_setup_creates_four_log_files(self, tmp_path: Path):
        tl = _make_logger(tmp_path)
        for name in ("main.log", "trades.log", "alerts.log", "regime.log"):
            assert (tmp_path / name).exists(), f"{name} not created"

    def test_setup_is_idempotent(self, tmp_path: Path):
        tl = _make_logger(tmp_path)
        tl.setup()          # second call must not raise or duplicate handlers
        # one handler per file + zero console = 4
        file_handlers = [
            h for h in tl._logger.handlers
            if isinstance(h, logging.handlers.RotatingFileHandler)
        ]
        assert len(file_handlers) == 4

    def test_all_events_written_to_main_log(self, tmp_path: Path):
        tl = _make_logger(tmp_path)
        tl.log_trade("SPY", "BUY", 10, 500.0)
        tl.log_regime_change("BULL", "BEAR", 0.8)
        tl.log_risk_event("DAILY_HALT", "dd breached")
        _flush_logger(tl)
        records = _read_jsonlines(tmp_path / "main.log")
        events  = {r["event"] for r in records if "event" in r}
        assert "trade"         in events
        assert "regime_change" in events
        assert "risk_event"    in events

    def test_trade_routes_to_trades_log_only_among_domain_files(self, tmp_path: Path):
        tl = _make_logger(tmp_path)
        tl.log_trade("QQQ", "SELL", 5, 400.0)
        _flush_logger(tl)
        assert _read_jsonlines(tmp_path / "trades.log")           # written
        assert not (tmp_path / "regime.log").read_text(encoding="utf-8").strip()  # empty
        assert not (tmp_path / "alerts.log").read_text(encoding="utf-8").strip()  # empty

    def test_regime_change_routes_to_regime_log(self, tmp_path: Path):
        tl = _make_logger(tmp_path)
        tl.log_regime_change("BULL", "NEUTRAL", 0.65)
        _flush_logger(tl)
        records = _read_jsonlines(tmp_path / "regime.log")
        assert any(r.get("event") == "regime_change" for r in records)
        assert not (tmp_path / "trades.log").read_text(encoding="utf-8").strip()

    def test_risk_event_routes_to_alerts_log(self, tmp_path: Path):
        tl = _make_logger(tmp_path)
        tl.log_risk_event("CIRCUIT", "halt triggered", severity="CRITICAL")
        _flush_logger(tl)
        records = _read_jsonlines(tmp_path / "alerts.log")
        assert any(r.get("event") == "risk_event" for r in records)

    def test_error_routes_to_alerts_log(self, tmp_path: Path):
        tl = _make_logger(tmp_path)
        tl.log_error(ValueError("test error"), context="unit test")
        _flush_logger(tl)
        records = _read_jsonlines(tmp_path / "alerts.log")
        assert any(r.get("event") == "error" for r in records)
        assert any("ValueError" in r.get("error_type", "") for r in records)

    def test_context_injected_into_every_record(self, tmp_path: Path):
        tl = _make_logger(tmp_path)
        tl.set_context(
            regime="BULL", probability=0.72,
            equity=105_000.0, positions=["SPY", "QQQ"], daily_pnl=340.0,
        )
        tl.log_trade("SPY", "BUY", 10, 500.0)
        _flush_logger(tl)
        records = _read_jsonlines(tmp_path / "main.log")
        trade_recs = [r for r in records if r.get("event") == "trade"]
        assert trade_recs, "no trade record found"
        ctx = trade_recs[-1]["ctx"]
        assert ctx["regime"]      == "BULL"
        assert ctx["probability"] == pytest.approx(0.72, abs=1e-4)
        assert ctx["equity"]      == pytest.approx(105_000.0)
        assert ctx["positions"]   == ["SPY", "QQQ"]
        assert ctx["daily_pnl"]   == pytest.approx(340.0)

    def test_set_context_defaults_to_unknown(self, tmp_path: Path):
        tl = _make_logger(tmp_path)
        tl.log_trade("SPY", "BUY", 1, 100.0)
        _flush_logger(tl)
        records = _read_jsonlines(tmp_path / "main.log")
        trade = next(r for r in records if r.get("event") == "trade")
        assert trade["ctx"]["regime"] == "UNKNOWN"

    def test_log_rebalance_records_deltas(self, tmp_path: Path):
        tl = _make_logger(tmp_path)
        tl.log_rebalance(
            target_weights   = {"SPY": 0.9, "QQQ": 0.05},
            previous_weights = {"SPY": 0.6, "QQQ": 0.05},
            regime           = "BULL",
        )
        _flush_logger(tl)
        records = _read_jsonlines(tmp_path / "regime.log")
        reb = next(r for r in records if r.get("event") == "rebalance")
        assert reb["deltas"]["SPY"] == pytest.approx(0.3, abs=1e-4)
        assert reb["deltas"]["QQQ"] == pytest.approx(0.0, abs=1e-4)

    def test_log_fill_event_field(self, tmp_path: Path):
        tl = _make_logger(tmp_path)
        tl.log_fill("SPY", "BUY", 10, 499.50, "order-123")
        _flush_logger(tl)
        records = _read_jsonlines(tmp_path / "trades.log")
        assert any(r.get("event") == "fill" for r in records)


# ═══════════════════════════════════════════════════════════════════════════════
# dashboard.py  —  static helpers
# ═══════════════════════════════════════════════════════════════════════════════

class TestFmtHoldTime:
    """_fmt_hold_time(holding_days) → compact string."""

    def test_less_than_one_hour_returns_minutes(self):
        # 0.4 days × 24 = 9.6 h — wait, 0.4 days is > 1 h
        # 0.02 days × 24 = 0.48 h = 28.8 min
        assert _fmt_hold_time(0.02) == "28m"

    def test_exactly_one_hour(self):
        assert _fmt_hold_time(1 / 24) == "1h"

    def test_hours_below_48(self):
        assert _fmt_hold_time(14 / 24) == "14h"

    def test_48_hours_returns_days(self):
        assert _fmt_hold_time(2.0) == "2d"

    def test_multi_day(self):
        assert _fmt_hold_time(7.0) == "7d"


class TestFmtHmmAge:
    """_fmt_hmm_age(Optional[datetime]) → human string."""

    def test_none_returns_never(self):
        assert _fmt_hmm_age(None) == "never"

    def test_under_one_hour(self):
        trained = dt.datetime.utcnow() - dt.timedelta(minutes=30)
        result  = _fmt_hmm_age(trained)
        assert result.endswith("m ago")

    def test_hours(self):
        trained = dt.datetime.utcnow() - dt.timedelta(hours=6)
        result  = _fmt_hmm_age(trained)
        assert result == "6h ago"

    def test_days(self):
        trained = dt.datetime.utcnow() - dt.timedelta(days=2, hours=3)
        result  = _fmt_hmm_age(trained)
        assert result == "2d ago"


class TestRiskBar:
    """_risk_bar(current_abs, halt_pct) → Rich Text with style + icon."""

    def test_green_below_half_of_threshold(self):
        text = _risk_bar(0.01, 0.03)     # 33% of threshold
        rendered = _render_to_str(text)
        assert "✅" in rendered

    def test_yellow_between_50_and_80_pct(self):
        text = _risk_bar(0.02, 0.03)     # 67% of threshold
        rendered = _render_to_str(text)
        assert "⚠" in rendered

    def test_red_above_80_pct(self):
        text = _risk_bar(0.025, 0.03)    # 83% of threshold
        rendered = _render_to_str(text)
        assert "⚠" in rendered

    def test_critical_at_or_above_threshold(self):
        text = _risk_bar(0.03, 0.03)     # exactly at threshold
        rendered = _render_to_str(text)
        assert "✗" in rendered

    def test_shows_current_and_threshold_pct(self):
        text     = _risk_bar(0.015, 0.03)
        rendered = _render_to_str(text)
        assert "1.50%" in rendered    # current
        assert "3%"    in rendered    # threshold


# ═══════════════════════════════════════════════════════════════════════════════
# dashboard.py  —  Dashboard class
# ═══════════════════════════════════════════════════════════════════════════════

def _mock_signal(
    regime: str       = "BULL",
    confidence: float = 0.72,
    is_stable: bool   = True,
    leverage: float   = 1.25,
    notes: list       = None,
) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        regime=regime, confidence=confidence,
        is_stable=is_stable, leverage=leverage,
        notes=notes or [],
    )


def _mock_position(
    symbol: str          = "SPY",
    qty: float           = 10.0,
    current_price: float = 520.0,
    unrealized_pnl: float       = 60.0,
    unrealized_pnl_pct: float   = 0.012,
    stop_level: Optional[float] = 508.0,
    weight: float        = 0.30,
    holding_days: float  = 0.5,
) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        symbol=symbol, qty=qty, current_price=current_price,
        unrealized_pnl=unrealized_pnl, unrealized_pnl_pct=unrealized_pnl_pct,
        stop_level=stop_level, weight=weight, holding_days=holding_days,
    )


def _mock_snapshot(positions=None) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        total_equity=105_230.0,
        cash=5_000.0,
        positions=positions or [],
        total_unrealized_pnl=230.0,
        total_unrealized_pnl_pct=0.0022,
        gross_exposure=0.95,
        drawdown_from_peak=-0.012,
    )


def _mock_drawdown(
    daily_dd: float  = -0.003,
    weekly_dd: float = -0.012,
    dd_from_peak: float = -0.012,
) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        daily_dd=daily_dd, weekly_dd=weekly_dd, dd_from_peak=dd_from_peak,
    )


class TestDashboardPushSignal:

    def test_push_signal_appends_to_deque(self):
        d = Dashboard()
        d.push_signal("14:30  SPY  Rebalance 60%→95%")
        assert "14:30  SPY  Rebalance 60%→95%" in d._recent_signals

    def test_push_signal_latest_is_first(self):
        d = Dashboard()
        d.push_signal("first")
        d.push_signal("second")
        assert list(d._recent_signals)[0] == "second"

    def test_push_signal_respects_max_length(self):
        from monitoring.dashboard import _MAX_SIGNALS
        d = Dashboard()
        for i in range(_MAX_SIGNALS + 5):
            d.push_signal(f"signal_{i}")
        assert len(d._recent_signals) == _MAX_SIGNALS


class TestDashboardUpdate:

    def test_update_stores_snapshot(self):
        d    = Dashboard()
        snap = _mock_snapshot()
        d.update(snapshot=snap)
        assert d._snapshot is snap

    def test_update_stores_signal(self):
        d   = Dashboard()
        sig = _mock_signal()
        d.update(signal=sig)
        assert d._signal is sig

    def test_update_stores_stability_bars(self):
        d = Dashboard()
        d.update(stability_bars=14)
        assert d._stability_bars == 14

    def test_update_stores_flicker_rate(self):
        d = Dashboard()
        d.update(flicker_rate=3, flicker_window=25)
        assert d._flicker_rate  == 3
        assert d._flicker_window == 25

    def test_update_appends_event(self):
        d = Dashboard()
        d.update(event="test event")
        assert any("test event" in e for e in d._recent_events)

    def test_update_stores_system_status(self):
        d   = Dashboard()
        sys = SystemStatus(api_ok=False, api_latency_ms=999.0, mode="LIVE")
        d.update(system_status=sys)
        assert d._system.api_ok  is False
        assert d._system.mode    == "LIVE"


class TestDashboardPanelRenderers:

    # ── Regime panel ──────────────────────────────────────────────────────────

    def test_regime_panel_no_signal_shows_waiting(self):
        d    = Dashboard()
        text = _render_to_str(d._render_regime(None, None, 0, 20))
        assert "Waiting" in text

    def test_regime_panel_shows_label_and_confidence(self):
        d   = Dashboard()
        sig = _mock_signal(regime="BEAR", confidence=0.65)
        text = _render_to_str(d._render_regime(sig, 8, 2, 20))
        assert "BEAR"  in text
        assert "65%"   in text

    def test_regime_panel_shows_stability_bars(self):
        d   = Dashboard()
        sig = _mock_signal()
        text = _render_to_str(d._render_regime(sig, 14, 1, 20))
        assert "14 bars" in text

    def test_regime_panel_shows_flicker_ratio(self):
        d   = Dashboard()
        sig = _mock_signal()
        text = _render_to_str(d._render_regime(sig, None, 3, 20))
        assert "3/20" in text

    def test_regime_panel_stable_label(self):
        d   = Dashboard()
        sig = _mock_signal(is_stable=True)
        text = _render_to_str(d._render_regime(sig, None, 0, 20))
        assert "STABLE" in text

    def test_regime_panel_pending_label(self):
        d   = Dashboard()
        sig = _mock_signal(is_stable=False)
        text = _render_to_str(d._render_regime(sig, None, 0, 20))
        assert "PENDING" in text

    # ── Positions table ───────────────────────────────────────────────────────

    def test_positions_empty_shows_dashes(self):
        d    = Dashboard()
        snap = _mock_snapshot(positions=[])
        text = _render_to_str(d._render_positions(snap))
        assert "--" in text

    def test_positions_no_snapshot(self):
        d    = Dashboard()
        text = _render_to_str(d._render_positions(None))
        assert "--" in text

    def test_positions_shows_symbol_and_price(self):
        d    = Dashboard()
        pos  = _mock_position(symbol="NVDA", current_price=875.50)
        snap = _mock_snapshot(positions=[pos])
        text = _render_to_str(d._render_positions(snap))
        assert "NVDA"    in text
        assert "875.50"  in text

    def test_positions_shows_stop_level(self):
        d    = Dashboard()
        pos  = _mock_position(stop_level=508.00)
        snap = _mock_snapshot(positions=[pos])
        text = _render_to_str(d._render_positions(snap))
        assert "508.00" in text

    def test_positions_shows_dash_when_no_stop(self):
        d    = Dashboard()
        pos  = _mock_position(stop_level=None)
        snap = _mock_snapshot(positions=[pos])
        text = _render_to_str(d._render_positions(snap))
        assert "--" in text

    def test_positions_shows_long_direction(self):
        d    = Dashboard()
        pos  = _mock_position(qty=10.0)
        snap = _mock_snapshot(positions=[pos])
        text = _render_to_str(d._render_positions(snap))
        assert "LONG" in text

    # ── Risk status panel ─────────────────────────────────────────────────────

    def test_risk_panel_no_data_shows_message(self):
        d    = Dashboard()
        text = _render_to_str(d._render_risk_status(None))
        assert "No drawdown" in text

    def test_risk_panel_shows_daily_and_peak(self):
        d  = Dashboard()
        dd = _mock_drawdown(daily_dd=-0.003, dd_from_peak=-0.012)
        text = _render_to_str(d._render_risk_status(dd))
        assert "Daily DD"   in text
        assert "From Peak"  in text

    def test_risk_panel_shows_weekly(self):
        d  = Dashboard()
        dd = _mock_drawdown(weekly_dd=-0.012)
        text = _render_to_str(d._render_risk_status(dd))
        assert "Weekly DD" in text

    # ── System panel ──────────────────────────────────────────────────────────

    def test_system_panel_no_data(self):
        d    = Dashboard()
        text = _render_to_str(d._render_system(None))
        assert "No system" in text

    def test_system_panel_shows_mode(self):
        d   = Dashboard()
        sys = SystemStatus(mode="PAPER")
        text = _render_to_str(d._render_system(sys))
        assert "PAPER" in text

    def test_system_panel_live_mode(self):
        d   = Dashboard()
        sys = SystemStatus(mode="LIVE")
        text = _render_to_str(d._render_system(sys))
        assert "LIVE" in text

    def test_system_panel_shows_latency(self):
        d   = Dashboard()
        sys = SystemStatus(api_ok=True, api_latency_ms=23.0)
        text = _render_to_str(d._render_system(sys))
        assert "23ms" in text

    def test_system_panel_shows_hmm_age(self):
        d   = Dashboard()
        trained = dt.datetime.utcnow() - dt.timedelta(hours=5)
        sys = SystemStatus(hmm_last_trained=trained)
        text = _render_to_str(d._render_system(sys))
        assert "ago" in text

    # ── Market open heuristic ─────────────────────────────────────────────────

    def test_market_closed_on_weekend(self):
        # Pick a known Saturday: 2024-01-06
        with mock.patch("monitoring.dashboard.dt") as mock_dt:
            mock_dt.datetime.now.return_value = dt.datetime(2024, 1, 6, 12, 0)
            mock_dt.time = dt.time
            assert Dashboard._market_open() is False

    def test_market_open_weekday_during_hours(self):
        # 2024-01-08 is a Monday
        with mock.patch("monitoring.dashboard.dt") as mock_dt:
            mock_dt.datetime.now.return_value = dt.datetime(2024, 1, 8, 10, 30)
            mock_dt.time = dt.time
            assert Dashboard._market_open() is True


# ═══════════════════════════════════════════════════════════════════════════════
# alerts.py
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def am() -> AlertManager:
    """AlertManager with no external channels, console enabled."""
    return AlertManager(console_enabled=True, rate_limit_minutes=15)


@pytest.fixture
def am_no_console() -> AlertManager:
    return AlertManager(console_enabled=False, rate_limit_minutes=15)


class TestAlertRateLimiting:

    def test_first_call_is_allowed(self, am: AlertManager):
        assert am._check_rate_limit("test_key") is True

    def test_second_call_within_window_is_blocked(self, am: AlertManager):
        am._update_rate_limit("test_key")
        assert am._check_rate_limit("test_key") is False

    def test_call_after_window_expires_is_allowed(self, am: AlertManager):
        # Back-date the last_sent time beyond the window
        am._last_sent["test_key"] = dt.datetime.utcnow() - dt.timedelta(minutes=20)
        assert am._check_rate_limit("test_key") is True

    def test_different_keys_are_independent(self, am: AlertManager):
        am._update_rate_limit("key_a")
        assert am._check_rate_limit("key_b") is True

    def test_alert_returns_true_when_sent(self, am: AlertManager):
        result = am.alert("Test Title", "Test body")
        assert result is True

    def test_alert_returns_false_when_suppressed(self, am: AlertManager):
        am.alert("Dup", "first")
        result = am.alert("Dup", "second")   # same title → same key → suppressed
        assert result is False


class TestAlertConsoleChannel:

    def test_console_enabled_calls_logger(self, am: AlertManager):
        with mock.patch.object(am, "_log_console") as mock_log:
            am.alert("Title", "Body", level="WARNING")
            mock_log.assert_called_once()

    def test_console_disabled_skips_logger(self, am_no_console: AlertManager):
        with mock.patch.object(am_no_console, "_log_console") as mock_log:
            am_no_console.alert("Title", "Body")
            mock_log.assert_not_called()


class TestAlertEmailChannel:

    def test_email_skipped_when_not_configured(self, am: AlertManager):
        """No smtp_host → send_email must return False without raising."""
        result = am.send_email("Subject", "Body", "user@example.com")
        assert result is False

    def test_email_sent_when_configured(self, am: AlertManager):
        am.smtp_host     = "smtp.example.com"
        am.smtp_user     = "bot@example.com"
        am.smtp_password = "secret"
        am.recipient     = "user@example.com"

        with mock.patch("smtplib.SMTP") as mock_smtp:
            instance = mock_smtp.return_value.__enter__.return_value
            result   = am.send_email("Test", "Body")
        assert result is True
        instance.sendmail.assert_called_once()

    def test_email_returns_false_on_smtp_error(self, am: AlertManager):
        am.smtp_host     = "smtp.example.com"
        am.smtp_user     = "bot@example.com"
        am.smtp_password = "secret"
        am.recipient     = "user@example.com"

        with mock.patch("smtplib.SMTP", side_effect=ConnectionRefusedError("refused")):
            result = am.send_email("Test", "Body")
        assert result is False


class TestAlertWebhookChannel:

    def test_webhook_skipped_when_not_configured(self, am: AlertManager):
        result = am.send_webhook("Title", "Body")
        assert result is False

    def test_webhook_sent_when_configured(self, am: AlertManager):
        am.webhook_url = "https://hooks.example.com/abc"
        mock_response  = mock.MagicMock()
        mock_response.status_code = 200
        with mock.patch("requests.post", return_value=mock_response) as mock_post:
            result = am.send_webhook("Title", "Body", "WARNING")
        assert result is True
        mock_post.assert_called_once()

    def test_webhook_returns_false_on_http_error(self, am: AlertManager):
        am.webhook_url = "https://hooks.example.com/abc"
        with mock.patch("requests.post", side_effect=Exception("network error")):
            result = am.send_webhook("Title", "Body")
        assert result is False


class TestAlertTriggerMethods:
    """Verify each trigger calls alert() with the correct level and key."""

    def _spy(self, am: AlertManager):
        """Patch alert() and return the mock so we can inspect calls."""
        return mock.patch.object(am, "alert", wraps=am.alert)

    # regime_change
    def test_regime_change_uses_info_level(self, am: AlertManager):
        with self._spy(am) as spy:
            am.alert_regime_change("BULL", "BEAR", 0.8)
        _, kwargs = spy.call_args
        assert kwargs.get("level", "").upper() == "INFO"

    def test_regime_change_key_includes_regimes(self, am: AlertManager):
        with self._spy(am) as spy:
            am.alert_regime_change("BULL", "BEAR", 0.8)
        _, kwargs = spy.call_args
        assert "BULL" in kwargs["alert_key"]
        assert "BEAR" in kwargs["alert_key"]

    # circuit_breaker
    def test_circuit_breaker_halt_is_critical(self, am: AlertManager):
        with self._spy(am) as spy:
            am.alert_circuit_breaker("DAILY_HALT", 100_000, 0.031, 0.03)
        _, kwargs = spy.call_args
        assert kwargs["level"].upper() == "CRITICAL"

    def test_circuit_breaker_reduce_is_warning(self, am: AlertManager):
        with self._spy(am) as spy:
            am.alert_circuit_breaker("DAILY_REDUCE", 100_000, 0.021, 0.02)
        _, kwargs = spy.call_args
        assert kwargs["level"].upper() == "WARNING"

    # large_pnl
    def test_large_pnl_loss_is_warning(self, am: AlertManager):
        with self._spy(am) as spy:
            am.alert_large_pnl("PORTFOLIO", -0.045, -4500.0, direction="loss")
        _, kwargs = spy.call_args
        assert kwargs["level"].upper() == "WARNING"

    def test_large_pnl_gain_is_info(self, am: AlertManager):
        with self._spy(am) as spy:
            am.alert_large_pnl("PORTFOLIO", 0.06, 6000.0, direction="gain")
        _, kwargs = spy.call_args
        assert kwargs["level"].upper() == "INFO"

    # data_feed_down
    def test_data_feed_down_is_critical(self, am: AlertManager):
        with self._spy(am) as spy:
            am.alert_data_feed_down("SPY", "timeout")
        _, kwargs = spy.call_args
        assert kwargs["level"].upper() == "CRITICAL"

    def test_data_feed_key_includes_symbol(self, am: AlertManager):
        with self._spy(am) as spy:
            am.alert_data_feed_down("QQQ")
        _, kwargs = spy.call_args
        assert "qqq" in kwargs["alert_key"]

    # api_lost
    def test_api_lost_is_critical(self, am: AlertManager):
        with self._spy(am) as spy:
            am.alert_api_lost("connection refused", 0.0)
        _, kwargs = spy.call_args
        assert kwargs["level"].upper() == "CRITICAL"

    # hmm_retrained
    def test_hmm_retrained_is_info(self, am: AlertManager):
        with self._spy(am) as spy:
            am.alert_hmm_retrained(n_symbols=10, n_states=3)
        _, kwargs = spy.call_args
        assert kwargs["level"].upper() == "INFO"

    def test_hmm_retrained_body_contains_state_count(self, am: AlertManager):
        with self._spy(am) as spy:
            am.alert_hmm_retrained(n_symbols=10, n_states=4)
        _, kwargs = spy.call_args
        assert "4" in kwargs["message"]   # message body

    # flicker_exceeded
    def test_flicker_exceeded_is_warning(self, am: AlertManager):
        with self._spy(am) as spy:
            am.alert_flicker_exceeded(5, 20, "NEUTRAL")
        _, kwargs = spy.call_args
        assert kwargs["level"].upper() == "WARNING"

    def test_flicker_exceeded_body_contains_ratio(self, am: AlertManager):
        with self._spy(am) as spy:
            am.alert_flicker_exceeded(5, 20, "NEUTRAL")
        _, kwargs = spy.call_args
        assert "5" in kwargs["message"] and "20" in kwargs["message"]

    # backward compat
    def test_alert_drawdown_halt_calls_circuit_breaker(self, am: AlertManager):
        with mock.patch.object(am, "alert_circuit_breaker",
                               return_value=True) as mock_cb:
            am.alert_drawdown_halt(equity=95_000.0, drawdown_pct=0.105)
        mock_cb.assert_called_once()
        _, kwargs = mock_cb.call_args
        assert kwargs.get("breaker_type") == "PEAK_HALT" or \
               mock_cb.call_args[0][0]    == "PEAK_HALT"
