"""
test_alerts.py — Unit tests for monitoring/alerts.py (AlertManager).

Tests cover:
  - Rate limiting (check_rate_limit, _update_rate_limit)
  - alert() method: sent, suppressed, fallback to logger
  - send_email: skipped when not configured, success path, failure path
  - send_webhook: skipped when not configured, success path, HTTP error path
  - Convenience methods: alert_drawdown_halt, alert_regime_change, alert_order_error
  - Payload formatting: _format_webhook_payload
"""

from __future__ import annotations

import datetime as dt
from unittest.mock import MagicMock, patch

import pytest

from monitoring.alerts import AlertManager, _LEVEL_COLOURS


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mgr() -> AlertManager:
    """AlertManager with no channels configured (uses logger fallback)."""
    return AlertManager(rate_limit_minutes=15)


@pytest.fixture
def webhook_mgr() -> AlertManager:
    """AlertManager with a webhook URL configured."""
    return AlertManager(
        webhook_url="https://hooks.example.com/test",
        rate_limit_minutes=15,
    )


# ── Rate limiting ─────────────────────────────────────────────────────────────

class TestRateLimit:
    def test_first_call_is_allowed(self, mgr: AlertManager) -> None:
        assert mgr.check_rate_limit("some-key") is True

    def test_second_call_immediately_blocked(self, mgr: AlertManager) -> None:
        mgr._update_rate_limit("key1")
        assert mgr.check_rate_limit("key1") is False

    def test_different_keys_independent(self, mgr: AlertManager) -> None:
        mgr._update_rate_limit("key-a")
        assert mgr.check_rate_limit("key-b") is True

    def test_rate_limit_expires(self, mgr: AlertManager) -> None:
        """After the rate-limit window, the same key should be allowed again."""
        # Back-date the last-sent time by more than the limit
        past = dt.datetime.utcnow() - dt.timedelta(minutes=20)
        mgr._last_sent["key-c"] = past
        assert mgr.check_rate_limit("key-c") is True

    def test_rate_limit_not_expired(self, mgr: AlertManager) -> None:
        """Within the rate-limit window the key should still be blocked."""
        recent = dt.datetime.utcnow() - dt.timedelta(minutes=5)
        mgr._last_sent["key-d"] = recent
        assert mgr.check_rate_limit("key-d") is False

    def test_update_records_timestamp(self, mgr: AlertManager) -> None:
        before = dt.datetime.utcnow()
        mgr._update_rate_limit("ts-key")
        after = dt.datetime.utcnow()
        recorded = mgr._last_sent["ts-key"]
        assert before <= recorded <= after


# ── alert() — no channels configured (logger fallback) ───────────────────────

class TestAlertFallback:
    def test_returns_true_on_first_call(self, mgr: AlertManager) -> None:
        result = mgr.alert("Test", "message body")
        assert result is True

    def test_returns_false_on_suppressed(self, mgr: AlertManager) -> None:
        mgr.alert("Test", "first call", alert_key="dup")
        result = mgr.alert("Test", "second call", alert_key="dup")
        assert result is False

    def test_uses_title_as_default_key(self, mgr: AlertManager) -> None:
        mgr.alert("TitleKey", "body")
        assert "TitleKey" in mgr._last_sent

    def test_explicit_alert_key_used(self, mgr: AlertManager) -> None:
        mgr.alert("Title", "body", alert_key="custom-key")
        assert "custom-key" in mgr._last_sent
        assert "Title" not in mgr._last_sent

    def test_rate_limit_updated_after_send(self, mgr: AlertManager) -> None:
        mgr.alert("T", "b", alert_key="k1")
        assert "k1" in mgr._last_sent


# ── send_email ────────────────────────────────────────────────────────────────

class TestSendEmail:
    def test_returns_false_when_not_configured(self, mgr: AlertManager) -> None:
        # No SMTP configured
        result = mgr.send_email("Subject", "Body")
        assert result is False

    def test_send_email_success(self) -> None:
        m = AlertManager(
            smtp_host="smtp.example.com",
            smtp_port=587,
            smtp_user="user@example.com",
            smtp_password="secret",
            recipient="dest@example.com",
        )
        with patch("smtplib.SMTP") as mock_smtp_cls:
            mock_smtp = MagicMock()
            mock_smtp_cls.return_value.__enter__ = lambda s: mock_smtp
            mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)
            result = m.send_email("Subject", "Body")
        assert result is True

    def test_send_email_failure_returns_false(self) -> None:
        m = AlertManager(
            smtp_host="smtp.example.com",
            smtp_port=587,
            smtp_user="user@example.com",
            smtp_password="secret",
            recipient="dest@example.com",
        )
        with patch("smtplib.SMTP", side_effect=ConnectionRefusedError("refused")):
            result = m.send_email("Fail", "Body")
        assert result is False


# ── send_webhook ──────────────────────────────────────────────────────────────

class TestSendWebhook:
    def test_returns_false_when_no_url(self, mgr: AlertManager) -> None:
        result = mgr.send_webhook("Title", "Message")
        assert result is False

    def test_success(self, webhook_mgr: AlertManager) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        with patch("requests.post", return_value=mock_response) as mock_post:
            result = webhook_mgr.send_webhook("Alert", "Details", level="WARNING")
        assert result is True
        mock_post.assert_called_once()

    def test_http_error_returns_false(self, webhook_mgr: AlertManager) -> None:
        import requests as req
        with patch("requests.post", side_effect=req.RequestException("timeout")):
            result = webhook_mgr.send_webhook("Alert", "Details")
        assert result is False

    def test_payload_sent_as_json(self, webhook_mgr: AlertManager) -> None:
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        with patch("requests.post", return_value=mock_response) as mock_post:
            webhook_mgr.send_webhook("T", "M", level="INFO")
        _, kwargs = mock_post.call_args
        payload = kwargs.get("json") or mock_post.call_args[0][0] if not kwargs.get("json") else kwargs["json"]
        assert "attachments" in payload


# ── alert() with webhook channel ─────────────────────────────────────────────

class TestAlertWithWebhook:
    def test_alert_sends_via_webhook(self, webhook_mgr: AlertManager) -> None:
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        with patch("requests.post", return_value=mock_response):
            result = webhook_mgr.alert("Test", "message")
        assert result is True

    def test_alert_suppressed_after_send(self, webhook_mgr: AlertManager) -> None:
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        with patch("requests.post", return_value=mock_response):
            webhook_mgr.alert("T", "m", alert_key="x")
            result = webhook_mgr.alert("T", "m", alert_key="x")
        assert result is False


# ── Convenience methods ───────────────────────────────────────────────────────

class TestConvenienceMethods:
    def test_alert_drawdown_halt(self, mgr: AlertManager) -> None:
        result = mgr.alert_drawdown_halt(equity=95_000.0, drawdown_pct=-0.10)
        assert result is True

    def test_alert_regime_change(self, mgr: AlertManager) -> None:
        result = mgr.alert_regime_change(
            previous_regime="BULL", new_regime="BEAR", confidence=0.80
        )
        assert result is True

    def test_alert_order_error(self, mgr: AlertManager) -> None:
        result = mgr.alert_order_error(symbol="SPY", error_message="broker timeout")
        assert result is True

    def test_convenience_methods_use_rate_limiting(self, mgr: AlertManager) -> None:
        """Second call with same alert_key within window must be suppressed."""
        mgr.alert_drawdown_halt(95_000.0, -0.10)
        result2 = mgr.alert_drawdown_halt(94_000.0, -0.11)
        assert result2 is False


# ── Payload formatting ────────────────────────────────────────────────────────

class TestPayloadFormatting:
    def test_unknown_level_uses_default_colour(self) -> None:
        m = AlertManager()
        payload = m._format_webhook_payload("T", "M", level="UNKNOWN")
        colour = payload["attachments"][0]["color"]
        assert colour == "#607d8b"

    def test_known_levels_map_to_correct_colours(self) -> None:
        m = AlertManager()
        for level, expected_colour in _LEVEL_COLOURS.items():
            payload = m._format_webhook_payload("T", "M", level=level)
            assert payload["attachments"][0]["color"] == expected_colour

    def test_payload_contains_title_and_message(self) -> None:
        m = AlertManager()
        payload = m._format_webhook_payload("My Title", "My Message", "WARNING")
        att = payload["attachments"][0]
        assert "My Title" in att["title"]
        assert att["text"] == "My Message"
        assert "ts" in att
