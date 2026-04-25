"""
alerts.py -- Multi-channel alert system for critical trading events.

Delivery channels (all optional, gracefully degraded):
  console  — always active; writes via Python logging at the appropriate level
  log file — handled by TradeLogger (alerts.log) upstream; no direct writes here
  email    — SMTP with STARTTLS; configured via constructor or env vars
  webhook  — HTTP POST (Slack / Teams / Discord); configured via constructor or env var

Rate limiting:
  Each alert_key is suppressed for *rate_limit_minutes* (default 15) after the
  last successful send, preventing a cascade of identical alerts.

Trigger coverage:
  alert_regime_change()      — HMM detected a regime transition
  alert_circuit_breaker()    — any circuit-breaker tier fires
  alert_large_pnl()          — single-day P&L outside ±threshold
  alert_data_feed_down()     — market-data source unreachable
  alert_api_lost()           — broker API unreachable or timing out
  alert_hmm_retrained()      — HMM model successfully retrained
  alert_flicker_exceeded()   — rapid regime oscillation above threshold

Environment variables (overridden by constructor args):
  ALERT_SMTP_HOST, ALERT_SMTP_PORT, ALERT_SMTP_USER, ALERT_SMTP_PASSWORD
  ALERT_RECIPIENT, ALERT_WEBHOOK_URL
"""

from __future__ import annotations

import datetime as dt
import logging
import os
import smtplib
from email.mime.text import MIMEText
from typing import Dict, Optional

import requests

logger = logging.getLogger(__name__)

# Webhook attachment colour (Slack-compatible)
_LEVEL_COLOURS = {
    "INFO":     "#36a64f",   # green
    "WARNING":  "#ff9800",   # orange
    "CRITICAL": "#e53935",   # red
}


class AlertManager:
    """
    Send alerts via console, email, and/or webhook, with per-key rate limiting.

    Parameters
    ----------
    smtp_host           : SMTP server hostname.
    smtp_port           : SMTP port (default 587, STARTTLS).
    smtp_user           : SMTP login username.
    smtp_password       : SMTP login password.
    recipient           : Default email recipient.
    webhook_url         : Slack / Teams / Discord webhook URL.
    rate_limit_minutes  : Minimum gap between repeated alerts of the same key.
    console_enabled     : If True (default), always log alerts to the console/
                          log file via Python logging, regardless of other channels.
    """

    def __init__(
        self,
        smtp_host:          Optional[str] = None,
        smtp_port:          int           = 587,
        smtp_user:          Optional[str] = None,
        smtp_password:      Optional[str] = None,
        recipient:          Optional[str] = None,
        webhook_url:        Optional[str] = None,
        rate_limit_minutes: int           = 15,
        console_enabled:    bool          = True,
    ) -> None:
        self.smtp_host          = smtp_host     or os.environ.get("ALERT_SMTP_HOST")
        self.smtp_port          = smtp_port
        self.smtp_user          = smtp_user     or os.environ.get("ALERT_SMTP_USER")
        self.smtp_password      = smtp_password or os.environ.get("ALERT_SMTP_PASSWORD")
        self.recipient          = recipient     or os.environ.get("ALERT_RECIPIENT")
        self.webhook_url        = webhook_url   or os.environ.get("ALERT_WEBHOOK_URL")
        self.rate_limit_minutes = rate_limit_minutes
        self.console_enabled    = console_enabled

        self._last_sent: Dict[str, dt.datetime] = {}

    # ======================================================================= #
    # Core send method                                                         #
    # ======================================================================= #

    def alert(
        self,
        title:     str,
        message:   str,
        level:     str           = "WARNING",
        alert_key: Optional[str] = None,
    ) -> bool:
        """
        Dispatch an alert through all configured channels, subject to rate limiting.

        Returns True if the alert was sent (not suppressed).

        Parameters
        ----------
        title     : Short summary line (used as email subject / webhook title).
        message   : Full body text.
        level     : "INFO", "WARNING", or "CRITICAL".
        alert_key : De-duplication key (defaults to *title*).
        """
        key = alert_key or title

        if not self._check_rate_limit(key):
            logger.debug(
                "Alert suppressed (rate limit): key=%s  next_allowed=%s",
                key,
                (
                    self._last_sent[key]
                    + dt.timedelta(minutes=self.rate_limit_minutes)
                ).strftime("%H:%M:%S"),
            )
            return False

        sent = False

        # ── Console / log-file channel (always active when enabled) ───────
        if self.console_enabled:
            self._log_console(level, title, message)
            sent = True

        # ── Email channel ─────────────────────────────────────────────────
        if self.smtp_host and self.recipient:
            sent |= self.send_email(title, message, self.recipient)

        # ── Webhook channel ───────────────────────────────────────────────
        if self.webhook_url:
            sent |= self.send_webhook(title, message, level)

        if sent:
            self._update_rate_limit(key)

        return sent

    # ======================================================================= #
    # Individual delivery methods                                              #
    # ======================================================================= #

    def send_email(
        self,
        subject:   str,
        body:      str,
        recipient: Optional[str] = None,
    ) -> bool:
        """Send via SMTP STARTTLS.  Returns True on success."""
        to = recipient or self.recipient
        if not all([self.smtp_host, self.smtp_user, self.smtp_password, to]):
            logger.debug("send_email: SMTP not fully configured — skipping")
            return False

        msg            = MIMEText(body, "plain", "utf-8")
        msg["Subject"] = f"[RegimeTrader] {subject}"
        msg["From"]    = self.smtp_user    # type: ignore[assignment]
        msg["To"]      = to                # type: ignore[assignment]

        try:
            with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=10) as smtp:
                smtp.ehlo()
                smtp.starttls()
                smtp.login(self.smtp_user, self.smtp_password)  # type: ignore[arg-type]
                smtp.sendmail(self.smtp_user, [to], msg.as_string())  # type: ignore[list-item]
            logger.info("Alert email sent  to=%s  subject=%s", to, subject)
            return True
        except Exception as exc:
            logger.error("send_email failed: %s", exc)
            return False

    def send_webhook(
        self,
        title:   str,
        message: str,
        level:   str = "WARNING",
    ) -> bool:
        """POST a Slack-compatible attachment payload.  Returns True on 2xx."""
        if not self.webhook_url:
            return False
        payload = self._webhook_payload(title, message, level)
        try:
            resp = requests.post(self.webhook_url, json=payload, timeout=10)
            resp.raise_for_status()
            logger.info("Webhook alert sent  title=%s  status=%d",
                        title, resp.status_code)
            return True
        except Exception as exc:
            logger.error("send_webhook failed: %s", exc)
            return False

    # ======================================================================= #
    # Trigger convenience methods                                              #
    # ======================================================================= #

    def alert_regime_change(
        self,
        previous_regime: str,
        new_regime:      str,
        confidence:      float,
    ) -> bool:
        """Alert on any HMM regime transition."""
        return self.alert(
            title   = f"Regime Change: {previous_regime} → {new_regime}",
            message = (
                f"HMM detected a regime transition.\n"
                f"  Previous : {previous_regime}\n"
                f"  New      : {new_regime}\n"
                f"  Confidence: {confidence:.1%}"
            ),
            level     = "INFO",
            alert_key = f"regime_change_{previous_regime}_{new_regime}",
        )

    def alert_circuit_breaker(
        self,
        breaker_type: str,
        equity:       float,
        drawdown_pct: float,
        threshold_pct: float,
    ) -> bool:
        """
        Alert when any circuit-breaker tier activates.

        Parameters
        ----------
        breaker_type  : e.g. "DAILY_HALT", "WEEKLY_REDUCE", "PEAK_HALT"
        equity        : Current portfolio equity.
        drawdown_pct  : Current drawdown as a positive fraction (e.g. 0.031).
        threshold_pct : The threshold that was breached (e.g. 0.03).
        """
        is_halt   = "HALT" in breaker_type.upper()
        level     = "CRITICAL" if is_halt else "WARNING"
        lock_note = "\nDelete trading_halted.lock to resume." if is_halt else ""
        return self.alert(
            title   = f"Circuit Breaker: {breaker_type}",
            message = (
                f"Circuit breaker activated: {breaker_type}\n"
                f"  Drawdown   : {drawdown_pct:.2%}\n"
                f"  Threshold  : {threshold_pct:.2%}\n"
                f"  Equity     : ${equity:,.2f}"
                f"{lock_note}"
            ),
            level     = level,
            alert_key = f"circuit_breaker_{breaker_type.lower()}",
        )

    def alert_large_pnl(
        self,
        label:      str,
        pnl_pct:    float,
        pnl_usd:    float,
        direction:  str = "loss",
    ) -> bool:
        """
        Alert on an unusually large single-day P&L move.

        Parameters
        ----------
        label     : Symbol or "PORTFOLIO".
        pnl_pct   : P&L as a signed fraction (e.g. -0.045 = -4.5 %).
        pnl_usd   : P&L in dollars (signed).
        direction : "gain" or "loss" (informational only).
        """
        sign  = "+" if pnl_usd >= 0 else ""
        level = "WARNING" if direction == "loss" else "INFO"
        return self.alert(
            title   = f"Large P&L {direction.title()}: {label}",
            message = (
                f"Unusually large single-day {direction} detected.\n"
                f"  Symbol / Portfolio : {label}\n"
                f"  P&L (%)            : {sign}{pnl_pct:.2%}\n"
                f"  P&L ($)            : {sign}${abs(pnl_usd):,.2f}"
            ),
            level     = level,
            alert_key = f"large_pnl_{label.lower()}_{direction}",
        )

    def alert_data_feed_down(
        self,
        symbol:       str,
        error_detail: str = "",
    ) -> bool:
        """Alert when a market-data feed is unreachable or returning stale data."""
        return self.alert(
            title   = f"Data Feed Down: {symbol}",
            message = (
                f"Market data feed unavailable for {symbol}.\n"
                f"  Error : {error_detail or 'unknown'}\n"
                f"  Action: trading paused until feed recovers."
            ),
            level     = "CRITICAL",
            alert_key = f"data_feed_{symbol.lower()}",
        )

    def alert_api_lost(
        self,
        error_detail:  str   = "",
        latency_ms:    float = 0.0,
    ) -> bool:
        """Alert when the broker API is unreachable or timing out."""
        lat_str = f"{latency_ms:.0f} ms" if latency_ms > 0 else "timeout"
        return self.alert(
            title   = "Broker API Lost",
            message = (
                f"Broker API connection failure.\n"
                f"  Latency / Error : {lat_str}\n"
                f"  Detail          : {error_detail or 'unknown'}\n"
                f"  Action          : order submission suspended."
            ),
            level     = "CRITICAL",
            alert_key = "api_lost",
        )

    def alert_hmm_retrained(
        self,
        n_symbols: int,
        n_states:  int,
        timestamp: Optional[dt.datetime] = None,
        extra:     str                   = "",
    ) -> bool:
        """Alert when the HMM regime model is successfully retrained."""
        ts_str = (timestamp or dt.datetime.utcnow()).strftime("%Y-%m-%d %H:%M UTC")
        return self.alert(
            title   = "HMM Model Retrained",
            message = (
                f"Regime HMM retrained successfully.\n"
                f"  Symbols  : {n_symbols}\n"
                f"  States   : {n_states}\n"
                f"  Completed: {ts_str}"
                + (f"\n  Note     : {extra}" if extra else "")
            ),
            level     = "INFO",
            alert_key = "hmm_retrained",
        )

    def alert_flicker_exceeded(
        self,
        flicker_count:  int,
        flicker_window: int,
        current_regime: str,
    ) -> bool:
        """
        Alert when the regime flicker rate exceeds the configured threshold.

        Flicker = rapid oscillation between regimes (regime changes per N bars).
        Elevated flicker causes the strategy to halve position sizes.
        """
        return self.alert(
            title   = f"Regime Flicker Exceeded: {flicker_count}/{flicker_window}",
            message = (
                f"Regime instability detected.\n"
                f"  Flicker count   : {flicker_count} changes in {flicker_window} bars\n"
                f"  Current regime  : {current_regime}\n"
                f"  Effect          : position sizes halved (uncertainty discount active)."
            ),
            level     = "WARNING",
            alert_key = "flicker_exceeded",
        )

    def alert_strategy_disabled(
        self,
        strategy_name: str,
        reason:        str,
    ) -> bool:
        """Alert when a strategy is auto-disabled by a failed health check."""
        return self.alert(
            title   = f"Strategy Auto-Disabled: {strategy_name}",
            message = (
                f"Strategy '{strategy_name}' was auto-disabled by its health check.\n"
                f"  Reason : {reason}\n"
                f"  Action : capital will be reallocated on next rebalance."
            ),
            level     = "WARNING",
            alert_key = f"strategy_disabled_{strategy_name.lower()}",
        )

    def alert_allocator_rebalance(
        self,
        weights:       Dict[str, float],
        trigger:       str = "scheduled",
    ) -> bool:
        """Alert when the capital allocator rebalances strategy weights."""
        weight_str = ", ".join(f"{k}:{v:.2f}" for k, v in sorted(weights.items()))
        return self.alert(
            title   = f"Allocator Rebalance ({trigger})",
            message = (
                f"Capital allocator rebalanced strategy weights.\n"
                f"  Trigger : {trigger}\n"
                f"  Weights : {{{weight_str}}}"
            ),
            level     = "INFO",
            alert_key = "allocator_rebalance",
        )

    def alert_correlation_cluster(
        self,
        pairs:     list,
        threshold: float = 0.80,
    ) -> bool:
        """Alert when strategy pair-wise correlation exceeds *threshold*.

        Parameters
        ----------
        pairs : list of (strat_a, strat_b, corr) tuples above the threshold.
        """
        lines = "\n".join(
            f"  {a} <-> {b}: {c:.2f}" for a, b, c in pairs
        )
        return self.alert(
            title   = f"Correlation Cluster Detected (>{threshold:.2f})",
            message = (
                f"Strategy pair-wise correlation breached threshold.\n"
                f"{lines}\n"
                f"  Effect : correlated strategies may be merged in risk checks."
            ),
            level     = "WARNING",
            alert_key = "correlation_cluster",
        )

    def alert_portfolio_dd_breaker(
        self,
        equity:       float,
        drawdown_pct: float,
        threshold_pct: float,
    ) -> bool:
        """Alert when the portfolio-level drawdown breaker fires."""
        return self.alert_circuit_breaker(
            breaker_type  = "PORTFOLIO_DD_HALT",
            equity        = equity,
            drawdown_pct  = drawdown_pct,
            threshold_pct = threshold_pct,
        )

    def alert_order_error(
        self,
        symbol:        str,
        error_message: str,
    ) -> bool:
        """Alert on order submission failure (kept for backward compatibility)."""
        return self.alert(
            title   = f"Order Error: {symbol}",
            message = (
                f"An order for {symbol} failed.\n"
                f"  Error: {error_message}"
            ),
            level     = "WARNING",
            alert_key = f"order_error_{symbol.lower()}",
        )

    # kept for backward compatibility
    def alert_drawdown_halt(
        self,
        equity:       float,
        drawdown_pct: float,
    ) -> bool:
        """Fire a CRITICAL alert when the peak-drawdown halt threshold fires."""
        return self.alert_circuit_breaker(
            breaker_type  = "PEAK_HALT",
            equity        = equity,
            drawdown_pct  = drawdown_pct,
            threshold_pct = 0.10,
        )

    # ======================================================================= #
    # Rate limiting                                                            #
    # ======================================================================= #

    def check_rate_limit(self, alert_key: str) -> bool:
        """Return True if it is OK to send (enough time has elapsed)."""
        return self._check_rate_limit(alert_key)

    def _check_rate_limit(self, alert_key: str) -> bool:
        last = self._last_sent.get(alert_key)
        if last is None:
            return True
        elapsed_min = (dt.datetime.utcnow() - last).total_seconds() / 60.0
        return elapsed_min >= self.rate_limit_minutes

    def _update_rate_limit(self, alert_key: str) -> None:
        self._last_sent[alert_key] = dt.datetime.utcnow()

    # ======================================================================= #
    # Private helpers                                                          #
    # ======================================================================= #

    def _log_console(self, level: str, title: str, message: str) -> None:
        """Write to the Python logger (appears on console + alerts.log)."""
        log_fn = getattr(logger, level.lower(), logger.warning)
        log_fn("ALERT [%s] %s — %s", level, title,
               message.replace("\n", " | "))

    def _webhook_payload(
        self,
        title:   str,
        message: str,
        level:   str,
    ) -> dict:
        """Build a Slack-compatible attachment payload."""
        colour = _LEVEL_COLOURS.get(level.upper(), "#607d8b")
        return {
            "attachments": [{
                "color":  colour,
                "title":  f"[{level}] {title}",
                "text":   message,
                "footer": "Regime Trader",
                "ts":     int(dt.datetime.utcnow().timestamp()),
            }]
        }
