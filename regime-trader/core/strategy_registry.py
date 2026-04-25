"""
strategy_registry.py — Singleton plug-in registry for trading strategies.

Usage
-----
Register at class definition time:

    @register_strategy("my_strategy")
    class MyStrategy(BaseStrategy):
        ...

Or at runtime:

    registry = StrategyRegistry.instance()
    registry.register("my_strategy", MyStrategy())

The decorator instantiates the class with no arguments; pass a pre-built
instance to registry.register() when the constructor needs arguments.
"""

from __future__ import annotations

import logging
import threading
from typing import Callable, Dict, Optional, Type

logger = logging.getLogger(__name__)

# Forward reference — resolved at import time after regime_strategies is loaded.
_BaseStrategy = None  # set by _get_base()


def _get_base():
    global _BaseStrategy
    if _BaseStrategy is None:
        from core.regime_strategies import BaseStrategy
        _BaseStrategy = BaseStrategy
    return _BaseStrategy


class DuplicateStrategyError(Exception):
    """Raised when a strategy name is already registered."""


class StrategyRegistry:
    """
    Process-level singleton that holds all registered strategy instances.

    Thread-safe: a single lock guards all mutations.
    """

    _instance: Optional["StrategyRegistry"] = None
    _lock: threading.Lock = threading.Lock()

    # ── Singleton access ───────────────────────────────────────────────────────

    @classmethod
    def instance(cls) -> "StrategyRegistry":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls.__new__(cls)
                    cls._instance._strategies: Dict[str, object] = {}
                    cls._instance._registry_lock = threading.Lock()
        return cls._instance

    @classmethod
    def _reset(cls) -> None:
        """Reset singleton — test use only."""
        with cls._lock:
            cls._instance = None

    # ── Mutation ───────────────────────────────────────────────────────────────

    def register(self, name: str, strategy) -> None:
        """
        Add a strategy instance under *name*.

        Raises
        ------
        DuplicateStrategyError
            If *name* is already registered. Call unregister() first to replace.
        TypeError
            If *strategy* is not a BaseStrategy subclass instance.
        """
        base = _get_base()
        if not isinstance(strategy, base):
            raise TypeError(
                f"strategy must be a BaseStrategy instance, got {type(strategy).__name__}"
            )
        with self._registry_lock:
            if name in self._strategies:
                raise DuplicateStrategyError(
                    f"Strategy '{name}' is already registered. "
                    "Call unregister() first if you intend to replace it."
                )
            self._strategies[name] = strategy
        logger.info("StrategyRegistry: registered '%s' (%s)", name, type(strategy).__name__)

    def unregister(self, name: str) -> None:
        """Remove *name* from the registry. No-op if not present."""
        with self._registry_lock:
            removed = self._strategies.pop(name, None)
        if removed is not None:
            logger.info("StrategyRegistry: unregistered '%s'", name)

    # ── Queries ────────────────────────────────────────────────────────────────

    def get(self, name: str):
        """Return the strategy registered under *name*, or None."""
        return self._strategies.get(name)

    def all(self) -> Dict[str, object]:
        """Return a shallow copy of the full {name: strategy} dict."""
        with self._registry_lock:
            return dict(self._strategies)

    def active(self) -> Dict[str, object]:
        """Return strategies where ``is_enabled`` is True."""
        with self._registry_lock:
            return {
                name: s
                for name, s in self._strategies.items()
                if getattr(s, "is_enabled", True)
            }

    def run_health_checks(self, bar_count: int = 0) -> None:
        """
        Run health_check() on every registered strategy.

        Strategies that report unhealthy are auto-disabled via their
        on_disable() lifecycle hook and a warning is logged.

        Parameters
        ----------
        bar_count : int
            Current bar count. Health checks are skipped during warmup
            (first 60 bars) to allow strategies to accumulate history
            before evaluating performance-based checks.
        """
        # Skip health checks during warmup period (first 60 bars)
        if bar_count < 60:
            return

        for name, strategy in self.all().items():
            if not hasattr(strategy, "health_check"):
                continue
            try:
                health = strategy.health_check()
            except Exception as exc:
                logger.error("Health check failed for '%s': %s", name, exc)
                continue

            try:
                setattr(strategy, "_last_health", health)
            except Exception:
                pass

            if not health.is_healthy and getattr(strategy, "is_enabled", True):
                logger.warning(
                    "Strategy '%s' is UNHEALTHY (%s) — auto-disabling.",
                    name,
                    health.reason_if_unhealthy,
                )
                strategy.is_enabled = False
                if hasattr(strategy, "on_disable"):
                    try:
                        strategy.on_disable()
                    except Exception as exc:
                        logger.error("on_disable() failed for '%s': %s", name, exc)


# ── Decorator ──────────────────────────────────────────────────────────────────

def register_strategy(name: str) -> Callable[[Type], Type]:
    """
    Class decorator that instantiates the class and adds it to the registry.

    The class must have a no-argument constructor (or default-only arguments).
    For strategies that need constructor arguments, skip this decorator and
    call StrategyRegistry.instance().register(name, instance) directly.

    Example
    -------
    @register_strategy("low_vol_bull")
    class LowVolBullStrategy(BaseStrategy):
        ...
    """
    def decorator(cls: Type) -> Type:
        instance = cls()
        StrategyRegistry.instance().register(name, instance)
        return cls
    return decorator
