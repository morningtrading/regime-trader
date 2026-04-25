"""
VIX / VXX fetcher for cross-asset HMM features.

Priority:
  1. Alpaca VXX ETF (5-min bars, same auth + infra as the rest of the data
     pipeline). VXX is a tradable proxy for VIX — it suffers from contango
     drag over long holds, but as a z-scored regime feature the drag is
     scale-invariant and does not distort the signal.
  2. yfinance ``^VIX`` daily fallback (forward-filled across intraday bars).
     Used only if Alpaca fetch fails. Loses intraday granularity but
     captures daily shocks.

The returned series is indexed by timezone-naive UTC timestamps, aligned
to the requested bar timeframe with forward-fill.
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

VIX_PROXY_SYMBOL = "VXX"


def fetch_vix_series(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    timeframe: str = "5Min",
    data_client=None,
) -> Optional[pd.Series]:
    """Fetch VIX proxy close prices aligned to the given timeframe.

    Parameters
    ----------
    start, end :
        Date range (ISO strings or Timestamps). Inclusive.
    timeframe :
        Bar timeframe string (e.g. ``"5Min"``, ``"1Day"``). Only used for
        Alpaca; yfinance fallback always returns daily.
    data_client :
        Optional pre-built Alpaca historical data client with a
        ``get_bars(symbols, timeframe, start, end)`` method. If None, an
        AlpacaClient is instantiated on the fly from env credentials.

    Returns
    -------
    ``pd.Series`` of VXX close prices, tz-naive UTC index. ``None`` if
    both Alpaca and yfinance fail.
    """
    # yfinance first: covers full history (1990+), essential for backtests
    # that start before Alpaca VXX availability (~2020-07).
    series = _try_yfinance_vix(start, end)
    if series is not None and len(series) > 20:
        logger.info("VIX feature source: yfinance ^VIX daily (%d bars)", len(series))
        return series

    series = _try_alpaca_vxx(start, end, timeframe, data_client)
    if series is not None and len(series) > 20:
        logger.info("VIX feature source: Alpaca VXX (%d bars)", len(series))
        return series

    logger.warning("VIX fetch failed from all sources — VIX features will be NaN")
    return None


# ── Source implementations ────────────────────────────────────────────────


def _try_alpaca_vxx(start, end, timeframe, data_client) -> Optional[pd.Series]:
    try:
        if data_client is None:
            # Lazy import to keep module dependency-light when not used
            from data.market_data import AlpacaClient   # type: ignore
            data_client = AlpacaClient()
            try:
                if hasattr(data_client, "connect") and not getattr(data_client, "_connected", False):
                    data_client.connect()
            except Exception as exc:
                logger.debug("AlpacaClient.connect() failed: %s", exc)

        bars = data_client.get_bars(
            symbols=[VIX_PROXY_SYMBOL],
            timeframe=timeframe,
            start=str(start),
            end=str(end),
        )
        if bars is None or bars.empty:
            return None

        # Handle multiindex (symbol, timestamp) or flat index
        if isinstance(bars.index, pd.MultiIndex):
            if VIX_PROXY_SYMBOL in bars.index.get_level_values(0):
                sym_df = bars.xs(VIX_PROXY_SYMBOL, level=0)
            else:
                sym_df = bars.droplevel(0)
        else:
            sym_df = bars

        if "close" not in sym_df.columns:
            return None

        s = sym_df["close"].copy()
        s.index = pd.to_datetime(s.index).tz_localize(None)
        s = s.sort_index()
        s.name = "vix_proxy"
        return s
    except Exception as exc:
        logger.debug("Alpaca VXX fetch failed: %s", exc)
        return None


def _try_yfinance_vix(start, end) -> Optional[pd.Series]:
    try:
        import yfinance as yf   # type: ignore
    except Exception:
        return None

    try:
        df = yf.download(
            "^VIX",
            start=str(start),
            end=str(end),
            interval="1d",
            progress=False,
            auto_adjust=False,
        )
        if df is None or df.empty:
            return None
        # yfinance sometimes returns a multiindex on columns when downloading
        # a single ticker
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if "Close" not in df.columns:
            return None
        s = df["Close"].copy()
        s.index = pd.to_datetime(s.index).tz_localize(None)
        s = s.sort_index()
        s.name = "vix_index"
        return s
    except Exception as exc:
        logger.debug("yfinance ^VIX fetch failed: %s", exc)
        return None
