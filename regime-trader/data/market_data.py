"""
market_data.py -- Real-time and historical market data fetching.

Provides a unified interface for:
  - Historical OHLCV bars via Alpaca REST (with caching)
  - Live bar/quote streaming via Alpaca StockDataStream WebSocket
  - Gap handling: weekends, holidays, and halts are skipped automatically
    (alpaca-py never returns bars for non-trading days)

Timeframe strings: "1Min", "5Min", "15Min", "30Min", "1Hour", "4Hour",
                   "1Day", "1Week"
"""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
import threading
from typing import Callable, Dict, List, Optional

import pandas as pd

from broker.alpaca_client import AlpacaClient, parse_timeframe

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Constants                                                                    #
# --------------------------------------------------------------------------- #

# Approximate trading bars per calendar day for each timeframe -- used to
# estimate a start date from an n_bars lookback.
_BARS_PER_DAY: Dict[str, float] = {
    "1Min":  390.0,
    "5Min":   78.0,
    "15Min":  26.0,
    "30Min":  13.0,
    "1Hour":   6.5,
    "4Hour":   1.625,
    "1Day":    1.0,
    "1Week":   0.2,
}

# Extra buffer factor so we fetch slightly more than needed
_LOOKBACK_BUFFER = 1.4


# --------------------------------------------------------------------------- #
# MarketData                                                                   #
# --------------------------------------------------------------------------- #

class MarketData:
    """
    Fetches and caches market data for the trading universe.

    Parameters
    ----------
    client :
        Connected AlpacaClient.
    symbols :
        Default trading universe.
    timeframe :
        Default bar resolution (e.g. "1Day").
    """

    def __init__(
        self,
        client:    AlpacaClient,
        symbols:   List[str],
        timeframe: str = "1Day",
    ) -> None:
        self.client:    AlpacaClient = client
        self.symbols:   List[str]    = symbols
        self.timeframe: str          = timeframe

        # symbol -> wide-format OHLCV DataFrame (index = timestamp)
        self._cache:             Dict[str, pd.DataFrame]         = {}
        self._stream_callbacks:  List[Callable[[dict], None]]    = []
        self._quote_callbacks:   List[Callable[[dict], None]]    = []
        self._stream_thread:     Optional[threading.Thread]      = None
        self._stream_loop:       Optional[asyncio.AbstractEventLoop] = None
        self._stream_running:    bool                            = False
        self._subscribed_syms:   List[str]                       = []

    # ======================================================================= #
    # Historical data                                                          #
    # ======================================================================= #

    def get_historical_bars(
        self,
        symbols:   Optional[List[str]] = None,
        start:     Optional[str]       = None,
        end:       Optional[str]       = None,
        n_bars:    Optional[int]       = None,
        timeframe: Optional[str]       = None,
        use_cache: bool                = True,
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV bars.

        ``start`` and ``n_bars`` are mutually exclusive; if both are provided
        ``start`` takes precedence.

        Parameters
        ----------
        symbols   : Subset of universe; defaults to self.symbols.
        start     : ISO-8601 start date.
        end       : ISO-8601 end date; defaults to today.
        n_bars    : Bars to fetch looking back from end.
        timeframe : Override default timeframe.
        use_cache : Return cached data when available.

        Returns
        -------
        DataFrame with MultiIndex (symbol, timestamp) and OHLCV columns.
        """
        syms = symbols or self.symbols
        tf   = timeframe or self.timeframe

        if start is None and n_bars is not None:
            start = self._n_bars_to_start_date(n_bars, tf)

        cache_key = f"{','.join(sorted(syms))}|{tf}|{start}|{end}"
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        df = self.client.get_bars(
            symbols   = syms,
            timeframe = tf,
            start     = start,
            end       = end,
        )

        # Normalise index: drop timezone for daily bars
        if not df.empty:
            df = self._normalise_index(df)

        if use_cache:
            self._cache[cache_key] = df

        return df

    def get_latest_bars(
        self,
        symbols:   Optional[List[str]] = None,
        n:         int                 = 1,
        timeframe: Optional[str]       = None,
    ) -> pd.DataFrame:
        """
        Fetch the most recent ``n`` bars for each symbol.

        Returns
        -------
        DataFrame with MultiIndex (symbol, timestamp) and OHLCV columns.
        """
        syms = symbols or self.symbols
        tf   = timeframe or self.timeframe

        end   = dt.datetime.now(dt.timezone.utc).isoformat()
        start = self._n_bars_to_start_date(max(n + 10, 20), tf)   # small buffer

        df = self.client.get_bars(
            symbols   = syms,
            timeframe = tf,
            start     = start,
            end       = end,
        )
        if df.empty:
            return df

        df = self._normalise_index(df)

        # Keep only the last n bars per symbol
        if isinstance(df.index, pd.MultiIndex):
            df = (
                df.groupby(level=0, group_keys=False)
                  .apply(lambda g: g.tail(n))
            )
        else:
            df = df.tail(n)
        return df

    def get_close_prices(
        self,
        symbols:   Optional[List[str]] = None,
        n_bars:    Optional[int]       = None,
        start:     Optional[str]       = None,
        timeframe: Optional[str]       = None,
    ) -> pd.DataFrame:
        """
        Return a wide-format DataFrame of close prices.

        Returns
        -------
        DataFrame of shape (n_bars, n_symbols), column = symbol, index = date.
        """
        df = self.get_historical_bars(
            symbols   = symbols,
            start     = start,
            n_bars    = n_bars,
            timeframe = timeframe,
            use_cache = True,
        )
        return self._pivot_to_wide(df, column="close")

    def get_benchmark_data(
        self,
        symbol: str           = "SPY",
        start:  Optional[str] = None,
        n_bars: Optional[int] = None,
    ) -> pd.Series:
        """
        Fetch the close price series for a benchmark symbol.

        Returns
        -------
        Series of close prices indexed by timestamp.
        """
        df = self.get_historical_bars(
            symbols   = [symbol],
            start     = start,
            n_bars    = n_bars,
            use_cache = True,
        )
        wide = self._pivot_to_wide(df, "close")
        if symbol in wide.columns:
            return wide[symbol].dropna()
        if not wide.empty:
            return wide.iloc[:, 0].dropna()
        return pd.Series(dtype=float, name=symbol)

    def get_latest_quote(self, symbol: str) -> dict:
        """Return the latest bid/ask quote for symbol."""
        return self.client.get_latest_quote(symbol)

    def get_latest_bar(self, symbol: str) -> Optional[object]:
        """Return the latest completed bar object for symbol."""
        bars = self.client.get_latest_bar([symbol])
        return bars.get(symbol)

    def get_snapshot(self, symbols: Optional[List[str]] = None) -> dict:
        """Return snapshots (latest bar + quote + daily bar) for symbols."""
        return self.client.get_snapshot(symbols or self.symbols)

    # ======================================================================= #
    # Real-time streaming                                                      #
    # ======================================================================= #

    def subscribe_bars(
        self,
        callback: Callable[[dict], None],
        symbols:  Optional[List[str]] = None,
    ) -> None:
        """
        Register a callback for real-time bar updates.

        The callback receives a dict with keys:
        symbol, open, high, low, close, volume, timestamp.
        """
        self._stream_callbacks.append(callback)
        if symbols:
            self._subscribed_syms = list(set(self._subscribed_syms + symbols))

    def subscribe_quotes(
        self,
        callback: Callable[[dict], None],
        symbols:  Optional[List[str]] = None,
    ) -> None:
        """
        Register a callback for real-time quote updates (bid/ask spread checks).

        The callback receives a dict with keys:
        symbol, bid_price, ask_price, bid_size, ask_size, timestamp.
        """
        self._quote_callbacks.append(callback)
        if symbols:
            self._subscribed_syms = list(set(self._subscribed_syms + symbols))

    def subscribe_to_stream(
        self,
        callback: Callable[[dict], None],
        symbols:  Optional[List[str]] = None,
    ) -> None:
        """Backwards-compatible alias -- registers a bar callback."""
        self.subscribe_bars(callback, symbols)

    def start_stream(self) -> None:
        """Start the StockDataStream WebSocket in a background daemon thread."""
        if self._stream_running:
            logger.warning("MarketData: stream already running")
            return

        syms = self._subscribed_syms or self.symbols
        if not syms:
            logger.warning("MarketData: no symbols to subscribe -- stream not started")
            return

        self._stream_running = True
        self._stream_thread  = threading.Thread(
            target = self._stream_worker,
            args   = (syms,),
            daemon = True,
            name   = "market-data-stream",
        )
        self._stream_thread.start()
        logger.info("MarketData: StockDataStream started for %s", syms)

    def stop_stream(self) -> None:
        """Gracefully stop the WebSocket stream."""
        self._stream_running = False
        if self._stream_loop and self._stream_loop.is_running():
            self._stream_loop.call_soon_threadsafe(self._stream_loop.stop)
        if self._stream_thread:
            self._stream_thread.join(timeout=5.0)
        logger.info("MarketData: StockDataStream stopped")

    # ======================================================================= #
    # Cache management                                                         #
    # ======================================================================= #

    def invalidate_cache(self, symbol: Optional[str] = None) -> None:
        """Clear the cache entirely or for a specific symbol."""
        if symbol is None:
            self._cache.clear()
        else:
            keys = [k for k in self._cache if symbol in k]
            for k in keys:
                del self._cache[k]

    # ======================================================================= #
    # Private helpers                                                          #
    # ======================================================================= #

    def _pivot_to_wide(self, long_df: pd.DataFrame, column: str = "close") -> pd.DataFrame:
        """
        Pivot a long-format MultiIndex DataFrame to wide format.

        Handles both (symbol, timestamp) and (timestamp, symbol) MultiIndex
        orderings returned by different alpaca-py versions.
        """
        if long_df.empty:
            return pd.DataFrame()

        if isinstance(long_df.index, pd.MultiIndex):
            # Determine level names
            names = long_df.index.names
            if "symbol" in names:
                sym_level = names.index("symbol")
                ts_level  = 1 - sym_level
            else:
                # Assume (symbol, timestamp) ordering
                sym_level, ts_level = 0, 1

            wide = long_df[column].unstack(level=sym_level)
            wide.index.name = "timestamp"
            return wide
        else:
            # Single-symbol case -- already wide-ish
            if column in long_df.columns:
                return long_df[[column]].rename(
                    columns={column: long_df.index.name or "close"}
                )
            return long_df

    def _normalise_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Strip timezone from daily-bar timestamps for consistent indexing.

        Intraday bars keep their timezone-aware index.
        """
        if df.empty:
            return df

        if isinstance(df.index, pd.MultiIndex):
            levels     = list(df.index.levels)
            ts_level   = None
            for i, lv in enumerate(levels):
                if hasattr(lv, "tz") and lv.tz is not None:
                    ts_level = i
                    break
            if ts_level is not None and self.timeframe in ("1Day", "1Week"):
                new_lv = levels[ts_level].normalize().tz_localize(None)
                df     = df.copy()
                df.index = df.index.set_levels(new_lv, level=ts_level)
        elif hasattr(df.index, "tz") and df.index.tz is not None:
            if self.timeframe in ("1Day", "1Week"):
                df = df.copy()
                df.index = pd.DatetimeIndex(df.index).normalize().tz_localize(None)

        return df

    def _n_bars_to_start_date(self, n_bars: int, timeframe: str) -> str:
        """
        Estimate the ISO-8601 start date that yields approximately n_bars.

        Adds a buffer to account for weekends and holidays.
        """
        bars_per_day = _BARS_PER_DAY.get(timeframe, 1.0)
        # calendar days needed = trading_days_needed / (5/7 trading fraction)
        # trading_days = n_bars / bars_per_day
        calendar_days = int((n_bars / bars_per_day) * (7 / 5) * _LOOKBACK_BUFFER) + 5
        start_dt      = dt.date.today() - dt.timedelta(days=calendar_days)
        return start_dt.isoformat()

    # -- WebSocket internals ----------------------------------------------- #

    def _stream_worker(self, symbols: List[str]) -> None:
        """Run the StockDataStream event loop in a background thread."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._stream_loop = loop
        try:
            loop.run_until_complete(self._run_stream(symbols))
        except Exception as exc:
            logger.error("MarketData stream worker exited: %s", exc)
        finally:
            loop.close()

    async def _run_stream(self, symbols: List[str]) -> None:
        """Async coroutine: connect StockDataStream and dispatch events."""
        from alpaca.data.live import StockDataStream
        from alpaca.data.enums import DataFeed

        feed = (
            DataFeed.IEX if self.client.data_feed.lower() == "iex"
            else DataFeed.SIP
        )
        stream = StockDataStream(
            api_key    = self.client._api_key,
            secret_key = self.client._secret_key,
            feed       = feed,
        )

        # Register bar handler
        if self._stream_callbacks:
            async def _bar_handler(bar):
                payload = {
                    "symbol":    bar.symbol,
                    "open":      float(bar.open),
                    "high":      float(bar.high),
                    "low":       float(bar.low),
                    "close":     float(bar.close),
                    "volume":    float(bar.volume),
                    "timestamp": bar.timestamp,
                }
                for cb in self._stream_callbacks:
                    try:
                        cb(payload)
                    except Exception as e:
                        logger.error("Bar callback error: %s", e)

            stream.subscribe_bars(_bar_handler, *symbols)

        # Register quote handler
        if self._quote_callbacks:
            async def _quote_handler(quote):
                payload = {
                    "symbol":    quote.symbol,
                    "bid_price": float(quote.bid_price),
                    "ask_price": float(quote.ask_price),
                    "bid_size":  float(quote.bid_size),
                    "ask_size":  float(quote.ask_size),
                    "timestamp": quote.timestamp,
                }
                for cb in self._quote_callbacks:
                    try:
                        cb(payload)
                    except Exception as e:
                        logger.error("Quote callback error: %s", e)

            stream.subscribe_quotes(_quote_handler, *symbols)

        logger.info("StockDataStream: connecting for %s ...", symbols)
        await stream._run_forever()
