---
name: add-broker-adapter
description: Scaffold a new broker integration (Hyperliquid, MT5, Interactive Brokers, Binance, Coinbase, Tradestation, Tradier, ccxt-based exchanges, etc.) using the broker adapter pattern. Use this skill whenever the user wants to add, connect, integrate, or hook up a new broker, exchange, or trading venue. Also use when the user mentions "switch from Alpaca to X," "also support Hyperliquid," "link my MT5," "use ccxt," "crypto exchange," "futures broker," or any request that extends the bot's trading to a new venue. Uses the BaseBroker abstract class so the rest of the system stays unchanged.
---

# Add a New Broker Adapter

The entire point of the adapter pattern is that the bot's core logic doesn't care what broker it's using. Risk manager, strategy, HMM — they all talk to an abstract `BaseBroker`. To add a new broker, you implement the interface, the rest of the system keeps working unchanged.

## When to use

Use whenever the user wants to trade on a broker the bot doesn't currently support. Do not add broker-specific code anywhere except inside the adapter for that broker.

## Workflow

### Step 1 — Inputs you need

Ask the user:

1. **Which broker?** (Alpaca, Hyperliquid, MT5, IB, Binance, ccxt-supported exchange, etc.)
2. **Asset class?** (equities, options, crypto spot, crypto perps, FX, futures)
3. **Do they have API credentials already?** If not, link them to the broker's API docs and have them set up before continuing.
4. **Paper or live first?** (Should always be paper first.)

### Step 2 — Read the base interface

Open `broker/base.py`. The abstract base class defines the interface every adapter must implement.

**Important — sync vs async:** Most retail broker SDKs are synchronous (`alpaca-trade-api`, `alpaca-py`, `MetaTrader5`, `ccxt` non-async). Some are native async (`ccxt.pro`, `hyperliquid-python-sdk`'s async variant). The base class supports BOTH patterns:

```python
class BaseBroker(ABC):
    """
    Base broker interface. Methods are defined as sync by default.
    For async-native brokers (Hyperliquid, ccxt.pro), override with async def
    and ensure the main loop awaits them.
    """

    @abstractmethod
    def get_account(self) -> Account: ...

    @abstractmethod
    def get_positions(self) -> list[Position]: ...

    @abstractmethod
    def get_latest_quote(self, symbol: str) -> Quote: ...

    @abstractmethod
    def get_historical_bars(
        self, symbol: str, timeframe: str, start: datetime, end: datetime
    ) -> pd.DataFrame: ...

    @abstractmethod
    def submit_order(self, signal: Signal) -> Order: ...

    @abstractmethod
    def submit_bracket_order(self, signal: Signal) -> Order: ...

    @abstractmethod
    def modify_stop(self, symbol: str, new_stop: float) -> None: ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> None: ...

    @abstractmethod
    def close_position(self, symbol: str) -> None: ...

    @abstractmethod
    def is_market_open(self) -> bool: ...

    @abstractmethod
    def subscribe_bars(self, symbols: list[str], callback): ...
```

**For async-native brokers:** wrap sync calls in a thread pool, or implement a parallel `AsyncBaseBroker` with `async def` methods. Don't mix `async def` and sync in the same adapter — it causes "coroutine was never awaited" errors and hangs.

If your broker SDK is sync but you want async for the main loop, use `asyncio.to_thread(broker.submit_order, signal)` in the main loop rather than making the adapter async.

If your broker doesn't natively support something (e.g., MT5 doesn't have bracket orders), you implement it in the adapter by combining primitives (submit entry, then submit stop, then submit take-profit as separate OCO orders).

### Step 3 — Create the adapter file

Create `broker/{broker_name}_client.py`. Follow this skeleton:

```python
"""
Broker adapter for {broker name}.

Docs: {link to API docs}
Auth: {API key / signed request / JWT / etc.}
Quirks: {anything non-obvious — symbol naming, decimals, rate limits}
Sync/Async: {sync by default — swap to async if SDK is async-native}
"""

import os
from broker.base import BaseBroker, Account, Position, Quote, Order
from core.signal_generator import Signal
# ... broker SDK imports


class {BrokerName}Broker(BaseBroker):
    def __init__(self, config: dict):
        self.config = config
        self._paper_mode = config['paper_trading']
        self._client = self._init_client()

    def _init_client(self):
        """Load credentials from env, never hardcoded."""
        api_key = os.environ[f"{BROKER}_API_KEY"]
        secret = os.environ[f"{BROKER}_SECRET"]
        base_url = self.config['paper_url'] if self._paper_mode else self.config['live_url']
        return SDKClient(api_key, secret, base_url)

    def get_account(self) -> Account:
        raw = self._client.get_account()
        return Account(
            equity=float(raw['equity']),
            cash=float(raw['cash']),
            buying_power=float(raw['buying_power']),
            # ... normalize to our dataclass
        )

    # ... implement remaining methods
```

### Step 4 — Handle the broker-specific gotchas

Every broker has them. Ask about and document these:

**Symbol naming**: Does the broker use `BTCUSD`, `BTC-USD`, `BTC/USDT`, `BTCUSDT.PERP`? Normalize in the adapter so the strategy always uses `BTCUSD` internally.

**Decimals and tick sizes**: Crypto uses 8 decimals, equities use 2. MT5 uses "points" which aren't the same as ticks. Handle rounding in `submit_order`.

**Rate limits**: Document them at the top of the file. Build a local rate limiter if needed. Never let the system hit 429s in production.

**Order types available**: Not every broker supports brackets, OCO, trailing stops, or post-only. Document which primitives exist and synthesize the missing ones.

**Funding / margin / perpetuals** (crypto only): Perps have funding payments. Log them as separate P&L entries. Margin crypto has liquidation price — surface it to the risk manager.

**Market hours**: Crypto is 24/7. Equities have regular + extended. FX has weekend gaps. Handle in `is_market_open()`.

**Fill notification**: WebSocket push vs polling. Prefer WebSocket. If polling, document the max lag.

### Step 5 — Add to the factory

Open `broker/__init__.py` and register the new adapter:

```python
BROKER_REGISTRY = {
    'alpaca': AlpacaBroker,
    'hyperliquid': HyperliquidBroker,
    'mt5': MT5Broker,
    '{broker_name}': {BrokerName}Broker,  # <- add
}

def get_broker(config: dict) -> BaseBroker:
    return BROKER_REGISTRY[config['broker']['name']](config['broker'])
```

### Step 6 — Add credentials template

Open `.env.example` and add the new broker's env vars:

```bash
# {BROKER} — paper trading keys
{BROKER}_API_KEY=
{BROKER}_SECRET=
{BROKER}_PAPER=true
```

Remind the user: `.env` is in `.gitignore`. Never commit keys.

### Step 7 — Add config section

Open `config/settings.yaml` and add a broker-specific section:

```yaml
broker:
  name: {broker_name}  # or alpaca, hyperliquid, etc.

{broker_name}:
  paper_url: https://...
  live_url: https://...
  paper_trading: true
  symbols: [BTCUSD, ETHUSD]  # asset-class appropriate
  timeframe: 1m             # or 1h, 1d
  # broker-specific knobs
  max_slippage_bps: 10
```

### Step 8 — Test suite

Before writing the tests, register the `integration` marker in `pyproject.toml` (or `pytest.ini`) to avoid `PytestUnknownMarkWarning`. Add to `pyproject.toml`:

```toml
[tool.pytest.ini_options]
markers = [
    "integration: tests that hit real paper broker APIs (slow, require credentials)",
]
```

Or in `pytest.ini`:

```ini
[pytest]
markers =
    integration: tests that hit real paper broker APIs (slow, require credentials)
```

Create `tests/test_{broker_name}_client.py`. At minimum:

```python
import pytest
from broker.{broker_name}_client import {BrokerName}Broker


@pytest.fixture
def paper_broker():
    """Uses real paper API — requires {BROKER}_API_KEY in .env."""
    return {BrokerName}Broker({'paper_trading': True, ...})


@pytest.mark.integration  # only run with explicit flag
def test_get_account(paper_broker):
    acc = paper_broker.get_account()
    assert acc.equity > 0
    assert acc.cash >= 0


@pytest.mark.integration
def test_submit_and_cancel(paper_broker):
    """Submit a limit order far from market, cancel it, verify state."""
    signal = make_test_signal(symbol='BTCUSD', far_from_market=True)
    order = paper_broker.submit_order(signal)
    assert order.status in ('new', 'accepted')
    paper_broker.cancel_order(order.order_id)
    # re-fetch and confirm canceled


@pytest.mark.integration
def test_symbol_normalization(paper_broker):
    """Bot always uses BTCUSD; adapter translates to broker-native."""
    assert paper_broker._to_native('BTCUSD') == '{whatever broker uses}'
    assert paper_broker._from_native('{whatever broker uses}') == 'BTCUSD'
```

Note: if your broker SDK is native-async, change `def` to `async def` in the tests and add `pytest-asyncio` to dev dependencies. Don't mix sync and async tests in the same file.

Run the integration tests yourself when the user wants to verify the adapter against the real paper API. Handle the environment:

- Install the broker's Python SDK if it's not already installed (check `requirements.txt` first, add it if missing, then install)
- Install pytest and pytest-asyncio if missing
- Confirm the paper API credentials are in `.env` before running — if not, ask the user to add them first
- Adapt test paths to project structure

Command to try first:

```bash
pytest tests/ -v -m integration
```

Report which tests passed, which failed, and whether the failures are environment issues (missing credentials, rate limits, API downtime) versus actual adapter bugs.

### Step 9 — Dry-run the full pipeline

Run the dry-run mode — this hits every layer (data → HMM → signal → risk → broker) except the final order submission, which is logged instead.

Command to try first:

```bash
python main.py --dry-run --broker {broker_name}
```

Install any missing dependencies you encounter. Run for a few minutes during market hours (or simulate if markets are closed — many brokers provide historical data replay).

Verify:

- Data feed connects
- Signals generate
- Risk manager approves/rejects correctly
- Orders are formatted correctly for the new broker
- No broker-specific imports leak outside the adapter file (run: `grep -rn "import {broker_sdk}" --include='*.py' .` and confirm it's only inside `broker/{broker_name}_client.py`)

Report findings clearly — don't gloss over anomalies.

### Step 10 — Document

Update README.md's Brokers section:

```markdown
## Supported Brokers

| Broker | Paper | Live | Asset Classes | Notes |
|---|---|---|---|---|
| Alpaca | ✅ | ✅ | US equities, options | Default |
| Hyperliquid | ✅ | ✅ | Crypto perps | 24/7, funding rates |
| MT5 | ✅ | ✅ | FX, CFDs | MetaTrader 5 bridge required |
| {New} | ✅ | ⚠️ | ... | ... |
```

## Common mistakes

- **Leaking broker SDK imports into core code** — `import alpaca_trade_api` should appear only inside `broker/alpaca_client.py`. If you see it in `core/`, that's a bug.
- **Skipping symbol normalization** — the bot's strategy doesn't know about exchange-specific symbols. Normalize both ways.
- **Forgetting rate limits** — your bot will work fine in paper for weeks, then get banned live during a volatile session when order rate spikes.
- **Using live endpoints by default** — always default to paper. Require explicit config to switch.
- **Not testing `cancel_order`** — easy to forget; critical when a fill fails partway.
- **Assuming market hours** — crypto is 24/7, FX has weekend gaps, equities have holidays. Your `is_market_open()` is wrong on day 1 unless tested.

## Do not

- Do not add broker-specific logic outside the adapter file.
- Do not hardcode credentials or URLs. Both go in env + config.
- Do not skip the dry-run step. Half the integration bugs only show up in the full pipeline.
- Do not enable live trading without a manual confirmation prompt (the same pattern as Alpaca).
