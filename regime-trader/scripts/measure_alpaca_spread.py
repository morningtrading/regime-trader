"""Measure real bid-ask spread on Alpaca for all asset groups.

Fetches a few snapshots over a short period, averages, and reports per-symbol
and per-group spreads in basis points. Use the output to calibrate
backtest.slippage_pct realistically.

Run: py -3.12 scripts/measure_alpaca_spread.py
"""
from __future__ import annotations
import os
import sys
import time
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml  # type: ignore

from broker.alpaca_client import AlpacaClient  # type: ignore

CFG_PATH = ROOT / "config" / "settings.yaml"
GROUPS_PATH = ROOT / "config" / "asset_groups.yaml"
CREDS_PATH = ROOT / "config" / "credentials.yaml"

# ── load symbols from every asset group ───────────────────────────────────
with open(GROUPS_PATH, "r", encoding="utf-8") as f:
    groups_cfg = yaml.safe_load(f) or {}
groups: dict[str, list[str]] = {}
for g_name, g in (groups_cfg.get("groups") or {}).items():
    syms = g.get("symbols") or []
    if syms:
        groups[g_name] = [s for s in syms if isinstance(s, str) and s]

# ── connect to Alpaca (paper) ─────────────────────────────────────────────
with open(CFG_PATH, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}
with open(CREDS_PATH, "r", encoding="utf-8") as f:
    creds = yaml.safe_load(f) or {}

broker = cfg.get("broker", {}) or {}
paper = bool(broker.get("paper_trading", True))
feed = broker.get("data_feed", "iex")

client = AlpacaClient(paper=paper, data_feed=str(feed))
client.connect()
print(f"Connected to Alpaca (paper={paper}, feed={feed})")
print()

# ── collect all unique symbols ────────────────────────────────────────────
all_symbols: list[str] = sorted({s for syms in groups.values() for s in syms})

# ── sample quotes (N times, spaced by a few seconds) ──────────────────────
N_SAMPLES = int(os.environ.get("N_SAMPLES", "3"))
SAMPLE_GAP_S = float(os.environ.get("SAMPLE_GAP_S", "2.0"))

spread_samples: dict[str, list[float]] = {s: [] for s in all_symbols}
mid_samples: dict[str, list[float]] = {s: [] for s in all_symbols}

for i in range(N_SAMPLES):
    sys.stdout.write(f"Sample {i+1}/{N_SAMPLES}: ")
    sys.stdout.flush()
    for sym in all_symbols:
        try:
            q = client.get_latest_quote(sym)
            ask = float(q.get("ask_price") or 0.0)
            bid = float(q.get("bid_price") or 0.0)
            if ask > 0 and bid > 0:
                mid = (ask + bid) / 2.0
                bps = (ask - bid) / mid * 10_000.0
                spread_samples[sym].append(bps)
                mid_samples[sym].append(mid)
        except Exception as e:
            sys.stdout.write(f"[{sym} err] ")
    sys.stdout.write("ok\n")
    sys.stdout.flush()
    if i + 1 < N_SAMPLES:
        time.sleep(SAMPLE_GAP_S)

# ── per-symbol report ─────────────────────────────────────────────────────
def avg(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0

print()
print(f"{'Symbol':<8} {'Mid':>10} {'Spread(bps)':>12} {'Samples':>8}")
print("-" * 44)
sym_spread: dict[str, float] = {}
for sym in all_symbols:
    xs = spread_samples[sym]
    mids = mid_samples[sym]
    if not xs:
        print(f"{sym:<8} {'n/a':>10} {'n/a':>12} {0:>8}")
        sym_spread[sym] = float("nan")
    else:
        s = avg(xs)
        sym_spread[sym] = s
        print(f"{sym:<8} {avg(mids):>10.2f} {s:>12.2f} {len(xs):>8}")

# ── per-group summary ─────────────────────────────────────────────────────
print()
print("=== Per-group average spread (bps) ===")
print(f"{'Group':<22} {'Avg_bps':>10} {'One-way(5bp)':>14} {'Round-trip':>12}")
for g_name in sorted(groups):
    syms = groups[g_name]
    vals = [sym_spread[s] for s in syms if not (sym_spread.get(s) != sym_spread.get(s))]
    vals = [v for v in vals if v == v]  # drop NaN
    if not vals:
        print(f"{g_name:<22} {'n/a':>10}")
        continue
    avg_bps = sum(vals) / len(vals)
    # Typical execution cost per side ≈ half-spread (crossing to get filled)
    # Round-trip cost = full spread.
    one_way = avg_bps / 2.0
    rt = avg_bps
    print(f"{g_name:<22} {avg_bps:>10.2f} {one_way:>14.2f} {rt:>12.2f}")

print()
print("Interpretation:")
print("  One-way cost (crossing NBBO) ≈ half-spread in bps.")
print("  Current backtest setting: balanced.yaml slippage_pct=0.0010 = 10 bps per side.")
print("  If measured half-spread is < 5 bps, lower the config to match reality.")
