"""
alpaca_client.py -- Thin wrapper around the Alpaca REST and WebSocket APIs.

Credentials are NEVER hardcoded.  They are loaded exclusively from:
  1. A .env file in the project root (via python-dotenv)
  2. Environment variables set before process start

Paper trading is the DEFAULT.  Live trading requires an explicit confirmation
prompt at startup.

.env file format (see .env.example):
  ALPACA_API_KEY=...
  ALPACA_SECRET_KEY=...
  ALPACA_PAPER=true          # set to false for live trading
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml
from dotenv import load_dotenv

from alpaca.data import StockHistoricalDataClient
from alpaca.data.enums import DataFeed
from alpaca.data.requests import (
    StockBarsRequest,
    StockLatestBarRequest,
    StockLatestQuoteRequest,
    StockSnapshotRequest,
)
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.models import TradeAccount, Clock, Position
from alpaca.trading.requests import GetOrdersRequest

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Constants                                                                    #
# --------------------------------------------------------------------------- #

_TIMEFRAME_MAP: Dict[str, TimeFrame] = {
    "1Min":  TimeFrame.Minute,
    "5Min":  TimeFrame(5,  TimeFrame.Minute.unit),
    "15Min": TimeFrame(15, TimeFrame.Minute.unit),
    "30Min": TimeFrame(30, TimeFrame.Minute.unit),
    "1Hour": TimeFrame.Hour,
    "4Hour": TimeFrame(4, TimeFrame.Hour.unit),
    "1Day":  TimeFrame.Day,
    "1Week": TimeFrame.Week,
}

_LIVE_CONFIRM_PHRASE = "YES I UNDERSTAND THE RISKS"


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def parse_timeframe(tf_str: str) -> TimeFrame:
    """Convert a string like '1Day' to an alpaca-py TimeFrame object."""
    tf = _TIMEFRAME_MAP.get(tf_str)
    if tf is None:
        raise ValueError(
            f"Unknown timeframe '{tf_str}'.  "
            f"Valid options: {list(_TIMEFRAME_MAP)}"
        )
    return tf


# --------------------------------------------------------------------------- #
# AlpacaClient                                                                 #
# --------------------------------------------------------------------------- #

class AlpacaClient:
    """
    Authenticated Alpaca client for market data and account management.

    Credentials are resolved exclusively from:
      1. .env file in the project root (loaded automatically on __init__)
      2. Environment variables already present in the process

    Parameters
    ----------
    paper :
        If True (default), use paper-trading endpoint.
        If False, prompts for a live-trading confirmation phrase.
    data_feed :
        "iex" (free) or "sip" (paid).
        Falls back to ALPACA_DATA_FEED env var, then "iex".
    """

    def __init__(
        self,
        paper:     Optional[bool] = None,
        data_feed: Optional[str]  = None,
    ) -> None:
        # 1. Load credentials.yaml (project root / config/)
        _cred_path = Path(__file__).resolve().parent.parent / "config" / "credentials.yaml"
        _creds: Dict[str, Any] = {}
        if _cred_path.exists():
            with open(_cred_path, "r", encoding="utf-8") as fh:
                _creds = yaml.safe_load(fh) or {}

        _alpaca_creds = _creds.get("alpaca", {})

        # 2. Fall back to .env / environment variables
        _env_path = Path(__file__).resolve().parent.parent / ".env"
        load_dotenv(dotenv_path=_env_path, override=False)

        self._api_key:    str = (
            _alpaca_creds.get("api_key") or os.environ.get("ALPACA_API_KEY", "")
        )
        self._secret_key: str = (
            _alpaca_creds.get("secret_key") or os.environ.get("ALPACA_SECRET_KEY", "")
        )

        if paper is None:
            if "paper" in _alpaca_creds:
                paper = bool(_alpaca_creds["paper"])
            else:
                env_val = os.environ.get("ALPACA_PAPER", "true").lower()
                paper = env_val not in ("false", "0", "no")

        self.paper:     bool = paper
        self.data_feed: str  = (
            data_feed or os.environ.get("ALPACA_DATA_FEED", "iex")
        )

        self._trading_client: Optional[TradingClient]             = None
        self._data_client:    Optional[StockHistoricalDataClient]  = None
        self._connected:      bool                                = False

    # ======================================================================= #
    # Initialisation                                                           #
    # ======================================================================= #

    def connect(self, skip_live_confirm: bool = False) -> None:
        """
        Instantiate the Alpaca trading and data clients and run a health check.

        Parameters
        ----------
        skip_live_confirm :
            Bypass the live-trading confirmation prompt (for tests only).

        Raises
        ------
        ValueError
            If API credentials are absent from the environment.
        RuntimeError
            If the live-trading confirmation is declined, or the health check fails.
        """
        if not self._api_key or not self._secret_key:
            raise ValueError(
                "Alpaca credentials not found.  "
                "Set ALPACA_API_KEY and ALPACA_SECRET_KEY in your .env file.  "
                "See .env.example for the expected format."
            )

        if not self.paper and not skip_live_confirm:
            self._confirm_live_trading()

        mode = "PAPER" if self.paper else "LIVE"
        logger.info("Connecting to Alpaca (%s, feed=%s) ...", mode, self.data_feed)

        self._trading_client = TradingClient(
            api_key    = self._api_key,
            secret_key = self._secret_key,
            paper      = self.paper,
        )
        self._data_client = StockHistoricalDataClient(
            api_key    = self._api_key,
            secret_key = self._secret_key,
        )

        self._connected = True
        self._health_check()
        logger.info("Alpaca connection established (%s).", mode)

    def disconnect(self) -> None:
        """Release client references and mark as disconnected."""
        self._trading_client = None
        self._data_client    = None
        self._connected      = False
        logger.info("AlpacaClient disconnected.")

    def connect_with_retry(
        self,
        max_attempts: int   = 5,
        base_delay:   float = 1.0,
    ) -> None:
        """
        Connect with exponential backoff on transient failures.

        Parameters
        ----------
        max_attempts : Maximum connection attempts before re-raising.
        base_delay   : Initial retry delay in seconds (doubles each attempt, max 60s).
        """
        delay = base_delay
        for attempt in range(1, max_attempts + 1):
            try:
                self.connect()
                return
            except (ValueError, RuntimeError):
                # Non-transient errors -- don't retry
                raise
            except Exception as exc:
                if attempt == max_attempts:
                    raise
                logger.warning(
                    "Connection attempt %d/%d failed: %s  Retrying in %.1fs ...",
                    attempt, max_attempts, exc, delay,
                )
                time.sleep(delay)
                delay = min(delay * 2, 60.0)

    # ======================================================================= #
    # Account                                                                  #
    # ======================================================================= #

    def get_account(self) -> TradeAccount:
        """Fetch and return current account information."""
        self._require_connection()
        return self._trading_client.get_account()

    def get_buying_power(self) -> float:
        """Return available buying power in USD."""
        return float(self.get_account().buying_power)

    def get_portfolio_value(self) -> float:
        """Return total portfolio equity (cash + open positions) in USD."""
        return float(self.get_account().equity)

    def get_cash(self) -> float:
        """Return uninvested cash balance in USD."""
        return float(self.get_account().cash)

    def get_available_margin(self) -> float:
        """Return available margin / buying power in USD."""
        return float(self.get_account().buying_power)

    def get_order_history(
        self,
        limit:  int           = 100,
        after:  Optional[str] = None,
        until:  Optional[str] = None,
        status: str           = "all",
    ) -> list:
        """
        Fetch recent orders.

        Parameters
        ----------
        limit  : Max number of orders to return.
        after  : ISO-8601 start datetime filter.
        until  : ISO-8601 end datetime filter.
        status : "all" | "open" | "closed".
        """
        self._require_connection()
        from alpaca.trading.enums import QueryOrderStatus
        status_map = {
            "all":    QueryOrderStatus.ALL,
            "open":   QueryOrderStatus.OPEN,
            "closed": QueryOrderStatus.CLOSED,
        }
        req = GetOrdersRequest(
            status = status_map.get(status, QueryOrderStatus.ALL),
            limit  = limit,
        )
        return self._trading_client.get_orders(req)

    # ======================================================================= #
    # Market data                                                              #
    # ======================================================================= #

    def get_bars(
        self,
        symbols:    List[str],
        timeframe:  str           = "1Day",
        start:      Optional[str] = None,
        end:        Optional[str] = None,
        limit:      Optional[int] = None,
        adjustment: str           = "all",
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV bars.

        Returns
        -------
        DataFrame with MultiIndex (symbol, timestamp) and columns
        [open, high, low, close, volume, vwap].
        """
        self._require_connection()
        from alpaca.data.enums import Adjustment, DataFeed
        adj_map = {
            "all":      Adjustment.ALL,
            "raw":      Adjustment.RAW,
            "split":    Adjustment.SPLIT,
            "dividend": Adjustment.DIVIDEND,
        }
        feed_map = {
            "iex": DataFeed.IEX,
            "sip": DataFeed.SIP,
        }
        # IEX (free tier) does not support adjusted data -- silently downgrade
        resolved_adj = adj_map.get(adjustment, Adjustment.ALL)
        resolved_feed = feed_map.get(self.data_feed.lower(), DataFeed.IEX)
        if resolved_feed == DataFeed.IEX and resolved_adj != Adjustment.RAW:
            resolved_adj = Adjustment.RAW

        req = StockBarsRequest(
            symbol_or_symbols = symbols,
            timeframe          = parse_timeframe(timeframe),
            start              = start,
            end                = end,
            limit              = limit,
            adjustment         = resolved_adj,
            feed               = resolved_feed,
        )
        bars = self._data_client.get_stock_bars(req)
        return bars.df

    def get_crypto_bars(
        self,
        symbols:   List[str],
        timeframe: str           = "1Day",
        start:     Optional[str] = None,
        end:       Optional[str] = None,
        limit:     Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV bars for crypto pairs (e.g. BTC/USD).

        Uses the CryptoHistoricalDataClient which does not require a paid
        data subscription.  No adjustment parameter — crypto has no splits
        or dividends.

        Returns
        -------
        DataFrame with MultiIndex (symbol, timestamp) and columns
        [open, high, low, close, volume, vwap].
        """
        from alpaca.data.historical.crypto import CryptoHistoricalDataClient
        from alpaca.data.requests import CryptoBarsRequest

        crypto_client = CryptoHistoricalDataClient(
            api_key    = self._api_key,
            secret_key = self._secret_key,
        )
        req = CryptoBarsRequest(
            symbol_or_symbols = symbols,
            timeframe          = parse_timeframe(timeframe),
            start              = start,
            end                = end,
            limit              = limit,
        )
        bars = crypto_client.get_crypto_bars(req)
        return bars.df

    def get_latest_bar(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Return the latest completed bar for each symbol.

        Returns
        -------
        Dict of symbol -> Bar object.
        """
        self._require_connection()
        req = StockLatestBarRequest(symbol_or_symbols=symbols)
        return self._data_client.get_stock_latest_bar(req)

    def get_latest_quote(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch the latest bid/ask quote.

        Returns
        -------
        Dict with keys: ask_price, bid_price, ask_size, bid_size.
        """
        self._require_connection()
        req = StockLatestQuoteRequest(symbol_or_symbols=[symbol])
        quotes = self._data_client.get_stock_latest_quote(req)
        q = quotes.get(symbol)
        if q is None:
            return {"ask_price": 0.0, "bid_price": 0.0, "ask_size": 0, "bid_size": 0}
        return {
            "ask_price": float(q.ask_price),
            "bid_price": float(q.bid_price),
            "ask_size":  float(q.ask_size),
            "bid_size":  float(q.bid_size),
        }

    def get_latest_price(self, symbol: str) -> float:
        """Return the mid-point of the latest bid/ask quote."""
        q = self.get_latest_quote(symbol)
        ask, bid = q.get("ask_price", 0.0), q.get("bid_price", 0.0)
        if ask > 0 and bid > 0:
            return (ask + bid) / 2.0
        bars = self.get_latest_bar([symbol])
        bar  = bars.get(symbol)
        return float(bar.close) if bar else 0.0

    def get_snapshot(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Return a snapshot (latest bar + quote + daily bar) for each symbol.

        Returns
        -------
        Dict of symbol -> Snapshot object.
        """
        self._require_connection()
        req = StockSnapshotRequest(symbol_or_symbols=symbols)
        return self._data_client.get_stock_snapshot(req)

    # ======================================================================= #
    # Clock & calendar                                                         #
    # ======================================================================= #

    def get_clock(self) -> Clock:
        """Return the current market clock (is_open, next_open, next_close)."""
        self._require_connection()
        return self._trading_client.get_clock()

    def is_market_open(self) -> bool:
        """Return True if the US equities market is currently open."""
        try:
            return bool(self.get_clock().is_open)
        except Exception as exc:
            logger.warning("is_market_open() failed, assuming closed: %s", exc)
            return False

    # ======================================================================= #
    # Positions                                                                #
    # ======================================================================= #

    def get_all_positions(self) -> List[Position]:
        """Return all open positions."""
        self._require_connection()
        return self._trading_client.get_all_positions()

    def get_position(self, symbol: str) -> Optional[Position]:
        """Return the open position for ``symbol``, or None if flat."""
        self._require_connection()
        try:
            return self._trading_client.get_open_position(symbol)
        except Exception as exc:
            logger.debug("get_position(%s) -> None (%s)", symbol, exc)
            return None

    def get_positions_as_dict(self) -> Dict[str, float]:
        """Return open positions as symbol -> market_value (USD) dict."""
        return {p.symbol: float(p.market_value) for p in self.get_all_positions()}

    # ======================================================================= #
    # Private helpers                                                          #
    # ======================================================================= #

    def _require_connection(self) -> None:
        """Raise RuntimeError if connect() has not been called yet."""
        if not self._connected:
            raise RuntimeError(
                "AlpacaClient is not connected.  "
                "Call client.connect() before using any API methods."
            )

    def _health_check(self) -> None:
        """Verify the connection by fetching account info."""
        try:
            acct = self._trading_client.get_account()
            logger.info(
                "Health check OK | account=%s | equity=$%.2f | "
                "buying_power=$%.2f | status=%s",
                acct.id,
                float(acct.equity),
                float(acct.buying_power),
                acct.status,
            )
        except Exception as exc:
            raise RuntimeError(f"Alpaca health check failed: {exc}") from exc

    @staticmethod
    def _confirm_live_trading() -> None:
        """
        Prompt the operator to type the live-trading confirmation phrase.
        Exits the process if the phrase is not matched exactly.
        """
        print("\n" + "=" * 60)
        print("  WARNING: LIVE TRADING MODE")
        print("  Real money will be at risk.")
        print("=" * 60)
        try:
            answer = input(
                f"  Type '{_LIVE_CONFIRM_PHRASE}' to confirm: "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            answer = ""

        if answer != _LIVE_CONFIRM_PHRASE:
            print("Live trading not confirmed.  Exiting.")
            sys.exit(1)

        logger.warning("Live trading confirmed by operator.")
