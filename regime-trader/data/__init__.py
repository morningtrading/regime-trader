"""
data — market data fetching and feature engineering.
"""

# Lazy imports: broker stubs may not be fully wired up yet.
# Import individual modules directly when needed.

__all__ = ["MarketData", "FeatureEngineer"]


def __getattr__(name):
    if name == "FeatureEngineer":
        from data.feature_engineering import FeatureEngineer
        return FeatureEngineer
    if name == "MarketData":
        from data.market_data import MarketData
        return MarketData
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
