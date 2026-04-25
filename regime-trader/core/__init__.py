"""
core — regime detection, strategy, risk management, and signal generation.
"""

__all__ = ["HMMEngine", "RegimeStrategy", "RiskManager", "SignalGenerator"]


def __getattr__(name):
    if name == "HMMEngine":
        from core.hmm_engine import HMMEngine
        return HMMEngine
    if name == "RegimeStrategy":
        from core.regime_strategies import RegimeStrategy
        return RegimeStrategy
    if name == "RiskManager":
        from core.risk_manager import RiskManager
        return RiskManager
    if name == "SignalGenerator":
        from core.signal_generator import SignalGenerator
        return SignalGenerator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
