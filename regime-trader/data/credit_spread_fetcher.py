"""
Credit spread proxy fetcher — HYG / LQD ratio z-score.

HYG = iShares iBoxx High Yield Corporate Bond ETF (junk bonds)
LQD = iShares iBoxx Investment Grade Corporate Bond ETF

The ratio HYG / LQD proxies the high-yield credit spread:
  risk-on regimes → HYG outperforms LQD → ratio rises
  risk-off regimes → HYG cracks, LQD holds → ratio falls

It's a textbook regime / credit-cycle signal, orthogonal to equity
vol (VIX), and captures stress that the HMM's equity-only features
miss.

We return the **rolling-60 z-score of the ratio** (stationary,
regime-agnostic). The raw ratio level is non-stationary — drops across
decades — and would contaminate HMM state definitions, same lesson
as VIX level vs VIX z-score.

Priority:
  1. yfinance daily (full history) — primary
  2. Alpaca ETF fallback (from 2020-07 onward)
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

HYG = "HYG"
LQD = "LQD"
ZSCORE_WINDOW = 60


def fetch_credit_spread_series(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    timeframe: str = "1Day",
    data_client=None,
) -> Optional[pd.Series]:
    """Return rolling-60 z-score of HYG/LQD ratio, tz-naive UTC index.

    Returns ``None`` if both sources fail.
    """
    ratio = _try_yfinance_ratio(start, end)
    if ratio is not None and len(ratio) > ZSCORE_WINDOW + 5:
        z = _rolling_zscore(ratio)
        logger.info(
            "Credit spread source: yfinance HYG/LQD (%d bars, z-scored)",
            len(z),
        )
        z.name = "credit_spread_proxy"
        return z

    ratio = _try_alpaca_ratio(start, end, timeframe, data_client)
    if ratio is not None and len(ratio) > ZSCORE_WINDOW + 5:
        z = _rolling_zscore(ratio)
        logger.info(
            "Credit spread source: Alpaca HYG/LQD (%d bars, z-scored)",
            len(z),
        )
        z.name = "credit_spread_proxy"
        return z

    logger.warning(
        "Credit spread fetch failed from all sources — feature will be NaN"
    )
    return None


def _rolling_zscore(ratio: pd.Series, window: int = ZSCORE_WINDOW) -> pd.Series:
    mean = ratio.rolling(window, min_periods=max(20, window // 3)).mean()
    std = ratio.rolling(window, min_periods=max(20, window // 3)).std()
    return (ratio - mean) / std.replace(0, np.nan)


# ── Source implementations ────────────────────────────────────────────────


def _try_yfinance_ratio(start, end) -> Optional[pd.Series]:
    try:
        import yfinance as yf  # type: ignore
    except Exception:
        return None
    try:
        df = yf.download(
            [HYG, LQD],
            start=str(start),
            end=str(end),
            interval="1d",
            progress=False,
            auto_adjust=True,
            group_by="column",
        )
        if df is None or df.empty:
            return None
        # yfinance returns multi-index columns for multi-ticker downloads
        if isinstance(df.columns, pd.MultiIndex):
            close = df["Close"] if "Close" in df.columns.get_level_values(0) else df["close"]
        else:
            return None  # unexpected shape for two tickers
        if HYG not in close.columns or LQD not in close.columns:
            return None
        ratio = close[HYG] / close[LQD]
        ratio.index = pd.to_datetime(ratio.index).tz_localize(None)
        ratio = ratio.sort_index().dropna()
        return ratio
    except Exception as exc:
        logger.debug("yfinance HYG/LQD fetch failed: %s", exc)
        return None


def _try_alpaca_ratio(start, end, timeframe, data_client) -> Optional[pd.Series]:
    try:
        if data_client is None:
            from data.market_data import AlpacaClient  # type: ignore
            data_client = AlpacaClient()
            try:
                if hasattr(data_client, "connect") and not getattr(
                    data_client, "_connected", False
                ):
                    data_client.connect()
            except Exception as exc:
                logger.debug("AlpacaClient.connect() failed: %s", exc)

        bars = data_client.get_bars(
            symbols=[HYG, LQD],
            timeframe=timeframe,
            start=str(start),
            end=str(end),
        )
        if bars is None or bars.empty:
            return None

        # bars may be multi-indexed (symbol, timestamp)
        if isinstance(bars.index, pd.MultiIndex):
            try:
                hyg = bars.xs(HYG, level=0)["close"]
                lqd = bars.xs(LQD, level=0)["close"]
            except KeyError:
                return None
        else:
            # Flat frame — expect a "symbol" column
            if "symbol" not in bars.columns:
                return None
            hyg = bars[bars["symbol"] == HYG]["close"]
            lqd = bars[bars["symbol"] == LQD]["close"]

        hyg.index = pd.to_datetime(hyg.index).tz_localize(None)
        lqd.index = pd.to_datetime(lqd.index).tz_localize(None)
        joined = pd.concat([hyg.rename(HYG), lqd.rename(LQD)], axis=1).dropna()
        if joined.empty:
            return None
        ratio = joined[HYG] / joined[LQD]
        return ratio.sort_index()
    except Exception as exc:
        logger.debug("Alpaca HYG/LQD fetch failed: %s", exc)
        return None
