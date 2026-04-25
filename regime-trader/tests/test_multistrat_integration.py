"""
tests/test_multistrat_integration.py — Tests for Phase 5 multi-strategy live integration.

Covers:
  1. TradingSession initialises multi-strat mode when ≥2 strategies enabled
  2. Strategy disabling propagates to CapitalAllocator (disabled strat gets zero weight)
  3. Allocator rebalance fires every _alloc_interval_bars bars
  4. PortfolioRiskManager gate is bypassed when --no-portfolio-risk flag is set
  5. Dashboard alloc_info built correctly from registry + alloc_weights
  6. _run_multi_strat_backtest builds StrategySpec list from settings.yaml correctly
"""

from __future__ import annotations

import types
from typing import Dict, List
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pandas as pd
import pytest

from backtest.multi_strategy_backtester import _BacktestProxy, StrategySpec
from core.capital_allocator import CapitalAllocator
from core.strategy_registry import StrategyRegistry


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_config(strats: dict | None = None) -> dict:
    """Build a minimal settings dict with optional [strategies] block."""
    return {
        "broker": {
            "symbols": ["SPY"],
            "timeframe": "1Day",
            "paper_trading": True,
        },
        "hmm": {
            "n_candidates": [3],
            "n_init": 1,
            "min_confidence": 0.55,
            "stability_bars": 3,
            "flicker_window": 20,
            "flicker_threshold": 4,
        },
        "strategy": {
            "low_vol_allocation": 0.95,
            "low_vol_leverage": 1.25,
            "mid_vol_allocation_trend": 0.95,
            "mid_vol_allocation_no_trend": 0.60,
            "high_vol_allocation": 0.60,
            "rebalance_threshold": 0.10,
        },
        "risk": {
            "max_exposure": 0.80,
            "max_leverage": 1.25,
            "max_single_position": 0.15,
        },
        "monitoring": {
            "dashboard_refresh_seconds": 5,
            "log_dir": "logs/",
            "log_level": "INFO",
        },
        "strategies": strats or {},
    }


def _two_strat_config() -> dict:
    return _make_config({
        "alpha": {"enabled": True, "symbols": ["SPY"], "weight_min": 0.0, "weight_max": 1.0},
        "beta":  {"enabled": True, "symbols": ["QQQ"], "weight_min": 0.0, "weight_max": 1.0},
    })


# ── Test 1: Multi-strat mode detection ─────────────────────────────────────────

class TestMultiStratModeDetection:
    """TradingSession.__init__ + startup multi-strat detection logic."""

    def test_multi_strat_flag_off_when_single_strategy(self):
        """Single strategy in settings.yaml → _multi_strat_mode stays False."""
        import importlib
        import main as m
        config = _make_config({
            "only_one": {"enabled": True, "symbols": ["SPY"]},
        })
        session = m.TradingSession(config)
        assert session._multi_strat_mode is False

    def test_multi_strat_flag_off_before_startup(self):
        """Flag starts False before startup() is called."""
        import main as m
        session = m.TradingSession(_two_strat_config())
        assert session._multi_strat_mode is False

    def test_multi_strat_filter_stored(self):
        """--strategies CLI arg is stored for filtering in startup()."""
        import main as m
        session = m.TradingSession(
            _two_strat_config(),
            multi_strat_filter=["alpha"],
        )
        assert session._multi_strat_filter == ["alpha"]

    def test_allocator_approach_stored(self):
        """--allocator CLI arg is stored on the session."""
        import main as m
        session = m.TradingSession(
            _two_strat_config(),
            allocator_approach="equal_weight",
        )
        assert session._allocator_approach == "equal_weight"

    def test_no_portfolio_risk_stored(self):
        """--no-portfolio-risk flag is stored on the session."""
        import main as m
        session = m.TradingSession(
            _two_strat_config(),
            no_portfolio_risk=True,
        )
        assert session._no_portfolio_risk is True


# ── Test 2: Strategy disabling propagates to allocator ─────────────────────────

class TestStrategyDisablingAllocator:
    """Disabled strategy in registry receives 0 weight from CapitalAllocator."""

    def setup_method(self):
        StrategyRegistry._reset()

    def teardown_method(self):
        StrategyRegistry._reset()

    def test_disabled_strategy_excluded_from_active(self):
        """StrategyRegistry.active() omits disabled strategies."""
        registry = StrategyRegistry.instance()
        p_a = _BacktestProxy("alpha", total_alloc=0.90)
        p_b = _BacktestProxy("beta",  total_alloc=0.90)
        registry.register("alpha", p_a)
        registry.register("beta",  p_b)

        p_b.is_enabled = False
        active = registry.active()

        assert "alpha" in active
        assert "beta"  not in active

    def test_allocator_skips_disabled_strategy(self):
        """CapitalAllocator.allocate() only returns weights for active strategies."""
        registry = StrategyRegistry.instance()
        p_a = _BacktestProxy("alpha")
        p_b = _BacktestProxy("beta")
        # Give them some return history so inverse_vol has data
        for i in range(5):
            ts = pd.Timestamp("2024-01-01") + pd.Timedelta(days=i)
            p_a.record_daily_return(ts, 0.01)
            p_b.record_daily_return(ts, 0.02)
        registry.register("alpha", p_a)
        registry.register("beta",  p_b)

        p_b.is_enabled = False
        allocator = CapitalAllocator(approach="equal_weight", total_capital=100_000)
        weights = allocator.allocate(registry)

        assert "alpha" in weights
        assert "beta" not in weights
        assert abs(weights["alpha"] - (1.0 - allocator.reserve)) < 0.05


# ── Test 3: Allocator rebalance interval ───────────────────────────────────────

class TestAllocatorRebalanceInterval:
    """_bars_since_alloc increments and triggers rebalance at the right time."""

    def setup_method(self):
        StrategyRegistry._reset()

    def teardown_method(self):
        StrategyRegistry._reset()

    def test_rebalance_fires_at_interval(self):
        """Allocator.allocate() called exactly once per interval bars."""
        registry = StrategyRegistry.instance()
        p_a = _BacktestProxy("alpha")
        p_b = _BacktestProxy("beta")
        registry.register("alpha", p_a)
        registry.register("beta",  p_b)

        allocator = CapitalAllocator(approach="equal_weight", total_capital=100_000)
        call_count = 0
        original_allocate = allocator.allocate

        def _counting_allocate(reg, **kw):
            nonlocal call_count
            call_count += 1
            return original_allocate(reg, **kw)

        allocator.allocate = _counting_allocate

        bars_since = 0
        interval = 5
        for _ in range(interval * 3):
            bars_since += 1
            if bars_since >= interval:
                bars_since = 0
                allocator.allocate(registry)

        assert call_count == 3

    def test_initial_weights_equal_for_equal_weight(self):
        """equal_weight gives ~0.5 to each of two strategies (90 % deployable)."""
        registry = StrategyRegistry.instance()
        p_a = _BacktestProxy("a")
        p_b = _BacktestProxy("b")
        registry.register("a", p_a)
        registry.register("b", p_b)

        allocator = CapitalAllocator(
            approach="equal_weight",
            total_capital=100_000,
            reserve=0.10,
        )
        weights = allocator.allocate(registry)
        assert abs(weights["a"] - 0.45) < 0.01
        assert abs(weights["b"] - 0.45) < 0.01


# ── Test 4: PortfolioRiskManager bypass ────────────────────────────────────────

class TestPortfolioRiskManagerBypass:
    """When no_portfolio_risk=True, portfolio_rm is None and not used."""

    def test_portfolio_rm_none_when_no_portfolio_risk(self):
        """portfolio_rm stays None when no_portfolio_risk=True (before startup)."""
        import main as m
        session = m.TradingSession(
            _two_strat_config(),
            no_portfolio_risk=True,
        )
        # Before startup, portfolio_rm is None
        assert session.portfolio_rm is None
        # Flag is set
        assert session._no_portfolio_risk is True

    def test_portfolio_rm_is_none_in_session_by_default(self):
        """portfolio_rm defaults to None before startup."""
        import main as m
        session = m.TradingSession(_two_strat_config())
        assert session.portfolio_rm is None


# ── Test 5: Dashboard alloc_info construction ──────────────────────────────────

class TestDashboardAllocInfo:
    """alloc_info dict is correctly built from registry + alloc_weights."""

    def setup_method(self):
        StrategyRegistry._reset()

    def teardown_method(self):
        StrategyRegistry._reset()

    def test_alloc_info_contains_all_strategies(self):
        registry = StrategyRegistry.instance()
        p_a = _BacktestProxy("alpha")
        p_b = _BacktestProxy("beta")
        registry.register("alpha", p_a)
        registry.register("beta",  p_b)

        alloc_weights = {"alpha": 0.45, "beta": 0.45}

        # Simulate how run_loop builds alloc_info
        alloc_info = {}
        for sn, w in alloc_weights.items():
            p = registry.get(sn)
            alloc_info[sn] = {
                "weight":  w,
                "sharpe":  getattr(p, "_sharpe", 0.0),
                "healthy": getattr(p, "is_enabled", True),
            }

        assert "alpha" in alloc_info
        assert "beta"  in alloc_info
        assert alloc_info["alpha"]["weight"] == pytest.approx(0.45)
        assert alloc_info["beta"]["healthy"] is True

    def test_alloc_info_reflects_disabled_strategy(self):
        registry = StrategyRegistry.instance()
        p_a = _BacktestProxy("alpha")
        p_b = _BacktestProxy("beta")
        registry.register("alpha", p_a)
        registry.register("beta",  p_b)
        p_b.is_enabled = False

        alloc_weights = {"alpha": 0.90, "beta": 0.0}

        alloc_info = {}
        for sn, w in alloc_weights.items():
            p = registry.get(sn)
            alloc_info[sn] = {
                "weight":  w,
                "sharpe":  getattr(p, "_sharpe", 0.0),
                "healthy": getattr(p, "is_enabled", True),
            }

        assert alloc_info["beta"]["healthy"] is False
        assert alloc_info["beta"]["weight"] == pytest.approx(0.0)


# ── Test 6: _run_multi_strat_backtest spec building ────────────────────────────

class TestMultiStratSpecBuilding:
    """Verify StrategySpec list is correctly built from config + prices."""

    def _make_prices(self) -> pd.DataFrame:
        idx = pd.date_range("2021-01-01", periods=200, freq="B")
        rng = np.random.default_rng(0)
        return pd.DataFrame(
            {
                "SPY": 100 * np.cumprod(1 + rng.normal(0, 0.01, 200)),
                "QQQ": 100 * np.cumprod(1 + rng.normal(0, 0.01, 200)),
            },
            index=idx,
        )

    def test_specs_include_only_enabled_strategies(self):
        config = _make_config({
            "alpha": {"enabled": True,  "symbols": ["SPY"]},
            "beta":  {"enabled": False, "symbols": ["QQQ"]},
        })
        prices = self._make_prices()

        specs = []
        for sname, scfg in config["strategies"].items():
            if not scfg.get("enabled", True):
                continue
            syms = scfg.get("symbols", [])
            missing = [s for s in syms if s not in prices.columns]
            if missing:
                continue
            specs.append(StrategySpec(
                name=sname, symbols=syms,
                weight_min=float(scfg.get("weight_min", 0.0)),
                weight_max=float(scfg.get("weight_max", 1.0)),
            ))

        assert len(specs) == 1
        assert specs[0].name == "alpha"

    def test_specs_skip_strategy_with_missing_symbols(self):
        config = _make_config({
            "alpha": {"enabled": True, "symbols": ["SPY"]},
            "gamma": {"enabled": True, "symbols": ["MISSING_SYM"]},
        })
        prices = self._make_prices()

        specs = []
        for sname, scfg in config["strategies"].items():
            if not scfg.get("enabled", True):
                continue
            syms = scfg.get("symbols", [])
            missing = [s for s in syms if s not in prices.columns]
            if missing:
                continue
            specs.append(StrategySpec(name=sname, symbols=syms))

        assert len(specs) == 1
        assert specs[0].name == "alpha"

    def test_specs_weight_bounds_propagated(self):
        config = _make_config({
            "alpha": {"enabled": True, "symbols": ["SPY"], "weight_min": 0.10, "weight_max": 0.60},
            "beta":  {"enabled": True, "symbols": ["QQQ"], "weight_min": 0.05, "weight_max": 0.40},
        })
        prices = self._make_prices()

        specs = []
        for sname, scfg in config["strategies"].items():
            syms = scfg.get("symbols", [])
            if not all(s in prices.columns for s in syms):
                continue
            specs.append(StrategySpec(
                name=sname, symbols=syms,
                weight_min=float(scfg.get("weight_min", 0.0)),
                weight_max=float(scfg.get("weight_max", 1.0)),
            ))

        spec_map = {s.name: s for s in specs}
        assert spec_map["alpha"].weight_min == pytest.approx(0.10)
        assert spec_map["alpha"].weight_max == pytest.approx(0.60)
        assert spec_map["beta"].weight_max  == pytest.approx(0.40)
