"""
monitoring — structured logging, terminal dashboard, and alerts.
"""

from monitoring.logger import TradeLogger
from monitoring.dashboard import Dashboard
from monitoring.alerts import AlertManager

__all__ = ["TradeLogger", "Dashboard", "AlertManager"]
