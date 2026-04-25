# Plan: Live Replay Mode (Simulate Live Trading on Recorded Data)

> **Status:** Planning only — no code yet. Discuss before implementing.

---

## Why

Live trading on a closed market (weekends, holidays, after-hours) gives no signal:
the loop runs but Alpaca returns stale data and the bot does nothing useful. We
can't observe the real "every-bar decision" path between trading sessions.

A **replay mode** would feed the live trading loop with historical bars at an
accelerated tick rate (e.g. 5Min bars replayed every 1 second). Same code path
as live, same broker adapter, same risk manager, same Telegram notifications —
only the data feed is swapped for a deterministic recording.

This catches a class of bugs that the backtester misses entirely: anything in
the live loop that's NOT in `WalkForwardBacktester` (order executor, broker
adapter, scheduler, retry logic, alert manager, dashboard updates).

---

## What we want to verify (the bug-class)

1. **Order executor path** — `submit_order` vs `submit_bracket_order` choice,
   sizing, retries, COID generation. Backtest fills are atomic; live has
   timeouts, partial fills, cancellations.
2. **Scheduler timing** — does the loop wake on the bar boundary? Does it skip
   bars under load? What happens when a bar takes >5min to process?
3. **State persistence** — does the bot recover correct positions/equity after
   a restart mid-session?
4. **Alert paths** — Telegram notifications fire at the right moments
   (regime change, trade, error) with the right content.
5. **Dashboard / monitoring** — log structure, JSON validity, dashboard refresh.
6. **Risk manager in real-time** — DD halt, position cap, max_concurrent
   enforcement under streaming data.

---

## Three viable architectures

### A. **Mock the broker client** (simplest, lowest fidelity)

Add a `MockBrokerClient` that implements `BaseBroker`'s interface, fills orders
instantly at the requested price + slippage, and tracks positions in-memory.
The trade loop runs unchanged; data comes from a CSV/parquet that's sliced one
bar at a time.

- **Pros:** ~1 day of work. Zero infrastructure. Deterministic.
- **Cons:** Doesn't exercise Alpaca SDK code paths (auth, rate-limits, REST
  errors). Won't catch bugs in `broker/alpaca_client.py` itself.

### B. **Mock at the data layer only** (medium fidelity)

Replace `data/market_data.py:fetch_latest_bars()` with a replay reader that
serves pre-recorded bars timestamped at "now". Real Alpaca paper account still
gets the orders. The bot thinks it's live; Alpaca paper actually fills (if
market is open, which defeats the purpose) or rejects (if closed).

- **Pros:** Exercises real broker code.
- **Cons:** Either need market-open window OR Alpaca paper rejects everything,
  defeating the test. Race between replay clock and Alpaca's clock.

### C. **Full sim-mode flag with synthetic clock** (highest fidelity, most work)

Introduce a `--replay` flag with three coordinated changes:
1. `data/market_data.py` reads from `replay/{date}.parquet` instead of Alpaca
2. `broker/alpaca_client.py` swaps to `MockBrokerClient` automatically
3. A synthetic clock module (`utils/clock.py`) returns the replay timestamp
   instead of `datetime.now()` everywhere — order COIDs, log timestamps, DD
   windows all use it

- **Pros:** Catches almost everything. Fully reproducible. Can run hundreds of
  "live days" in minutes for soak-testing.
- **Cons:** ~3-4 days of work. Need to thread the synthetic clock through every
  `datetime.now()` call (~50+ sites — see `grep "datetime.now\|time.time" -r .`).
  Risk of clock-leak bugs.

**Recommendation:** Start with **A** (mock broker + slice replay), see if it
catches anything useful, escalate to C only if we hit a class of bug A can't
reproduce.

---

## Concrete proposal for option A

```
regime-trader/
├── replay/
│   ├── __init__.py
│   ├── mock_broker.py              # MockBrokerClient(BaseBroker)
│   ├── replay_reader.py            # slices parquet → bars at controlled rate
│   ├── recordings/                 # gitignored
│   │   └── 2026-04-22_stocks4.parquet
│   └── record_session.py           # one-shot: live → save bars to parquet
└── main.py
    └── trade --replay PATH         # new flag
```

### CLI shape

```bash
# Record a real session (run during market hours, saves to replay/recordings/)
py -3.12 main.py trade --paper --record replay/recordings/2026-04-22_stocks4.parquet

# Replay it later (any time, any day)
py -3.12 main.py trade --replay replay/recordings/2026-04-22_stocks4.parquet \
                       --replay-speed 60       # 60x real-time (1 sec per minute)
                       --asset-group stocks4
```

### What `--replay` does internally

1. Loads the parquet into memory: ~390 bars/day × N symbols.
2. Instantiates `MockBrokerClient` instead of `AlpacaBrokerClient`.
3. Patches `fetch_latest_bars()` to read from a position pointer that advances
   on each loop iteration.
4. The trade loop's sleep is shrunk by `replay_speed` (5Min real → 5sec replay
   at 60x).
5. Telegram still fires (real notifications), so we see the same UX flow.
6. Logs go to `logs/replay_{recording_id}.json` for diff against backtest output.

### Verification harness

After a replay completes, compare:
- Trades from replay's `MockBrokerClient.history` vs trades the
  backtester would produce on the same date range. Should match modulo
  execution slippage assumptions.
- Regime sequence from replay's logs vs backtest's regime_history.csv. Should
  be byte-identical (same HMM, same data).
- Any divergence is a bug in the live code path that the backtest doesn't
  exercise.

---

## Effort estimate (option A)

| Step | Effort |
|---|---|
| `MockBrokerClient` implementing `BaseBroker` | ~3 hr |
| `ReplayReader` (parquet slicing + clock) | ~2 hr |
| Wire `--replay` flag in `main.py` | ~1 hr |
| `record_session.py` to capture real Alpaca bars | ~1 hr |
| Verification harness (replay vs backtest diff) | ~2 hr |
| Tests | ~2 hr |
| **Total** | **~1.5 days** |

---

## Open questions for you

1. **Recording cadence**: record one canonical session and reuse, or capture
   daily for a rolling library?
2. **Speed**: live (1×, 6.5 hr/day) for soak-testing, or 60×/600× for fast
   iteration? Both?
3. **Telegram during replay**: notify normally, or prefix subjects with
   `[REPLAY]` to avoid confusion?
4. **Crypto**: Alpaca crypto is 24/7 — does that change priority? (crypto
   group could be tested live without replay)

---

## Decision required before any code

- Approve option A vs B vs C
- Approve the `--replay PATH` CLI shape (or propose alternative)
- Decide: separate `replay/` package or integrate into existing `data/` and
  `broker/`?
