"""
research/config_set_comparison.py
Run a backtest for each config set (base, conservative, balanced, aggressive)
and print a comparison table with a recommendation.

Usage (from repo root):
    cd regime-trader
    py -3.12 research/config_set_comparison.py [--group stocks4] [--start 2020-01-01]
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Force UTF-8 output on Windows so box-drawing chars don't crash
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ── Repo root ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent   # .../regime-trader/

SETS = ["base", "conservative", "balanced", "aggressive"]

# Score weights (higher Sharpe + Calmar preferred, lower MaxDD penalised)
W_SHARPE  = 0.40
W_CALMAR  = 0.40
W_MAXDD   = 0.20   # uses (1 + max_dd) — max_dd is negative, so closer to 0 is better


# ── Helpers ───────────────────────────────────────────────────────────────────

def _latest_backtest_dir(before: float) -> Path | None:
    """Return the newest savedresults/backtest_* dir whose mtime > before."""
    sr = ROOT / "savedresults"
    candidates = sorted(
        [d for d in sr.iterdir() if d.is_dir() and d.name.startswith("backtest_")],
        key=lambda d: d.stat().st_mtime,
    )
    for d in reversed(candidates):
        if d.stat().st_mtime >= before:
            return d
    return None


def _run_backtest(set_name: str | None, group: str, start: str, end: str) -> Path | None:
    """Launch main.py backtest as subprocess; return the output dir."""
    cmd = [
        sys.executable, str(ROOT / "main.py"),
        "backtest",
        "--asset-group", group,
        "--start", start,
        "--end",   end,
    ]
    if set_name and set_name != "base":
        cmd += ["--set", set_name]

    label = set_name or "base"
    print(f"\n{'─'*60}")
    print(f"  Running: {label:14s}  group={group}  {start} → {end}")
    print(f"{'─'*60}")

    t0 = time.time()
    mark = time.time()   # capture just before we launch

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(cmd, cwd=ROOT, env=env,
                            capture_output=True, text=True, encoding="utf-8", errors="replace")

    # Always print subprocess output so errors are visible
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)

    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"  !! Backtest failed for set '{label}' (exit {result.returncode})")
        print(f"  !! Last stderr: {result.stderr[-500:] if result.stderr else '(none)'}")
        return None

    out_dir = _latest_backtest_dir(mark)
    print(f"  ✓  Done in {elapsed:.0f}s  →  {out_dir.name if out_dir else 'unknown'}")
    return out_dir


def _read_perf(out_dir: Path) -> dict | None:
    csv_path = out_dir / "performance_summary.csv"
    if not csv_path.exists():
        return None
    with open(csv_path) as f:
        reader = csv.reader(f)
        return {row[0]: row[1] for row in reader if len(row) == 2}


def _f(val: str, fmt: str) -> str:
    try:
        return fmt.format(float(val))
    except Exception:
        return val


def _score(perf: dict) -> float:
    sharpe = float(perf.get("sharpe", 0))
    calmar = float(perf.get("calmar", 0))
    maxdd  = float(perf.get("max_drawdown", -1))
    dd_score = 1.0 + maxdd   # 0.0 when maxdd=-1.0, 1.0 when maxdd=0.0
    return W_SHARPE * sharpe + W_CALMAR * calmar + W_MAXDD * dd_score


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Config-set comparison backtest")
    ap.add_argument("--group", default="stocks4", help="Asset group (default: stocks4)")
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end",   default=datetime.today().strftime("%Y-%m-%d"))
    args = ap.parse_args()

    print("\n" + "═"*60)
    print("  CONFIG SET COMPARISON")
    print(f"  Group : {args.group}")
    print(f"  Period: {args.start} → {args.end}")
    print(f"  Sets  : {', '.join(SETS)}")
    print("═"*60)

    results: dict[str, dict | None] = {}
    dirs:    dict[str, Path | None] = {}

    for s in SETS:
        d = _run_backtest(s if s != "base" else None, args.group, args.start, args.end)
        dirs[s] = d
        results[s] = _read_perf(d) if d else None

    # ── Comparison table ──────────────────────────────────────────────────────
    print("\n\n" + "═"*80)
    print("  RESULTS COMPARISON")
    print("═"*80)

    HDR = f"{'Metric':<22}" + "".join(f"{s:>14}" for s in SETS)
    print(HDR)
    print("─"*80)

    def row(label: str, key: str, fmt: str, invert: bool = False):
        vals = []
        for s in SETS:
            p = results[s]
            if p is None:
                vals.append("FAILED")
            else:
                vals.append(_f(p.get(key, "n/a"), fmt))
        # highlight best
        floats = []
        for v in vals:
            try:
                floats.append(float(v.replace("+", "").replace("%", "")))
            except Exception:
                floats.append(None)
        best_idx = None
        if any(f is not None for f in floats):
            valid = [(i, f) for i, f in enumerate(floats) if f is not None]
            best_idx = min(valid, key=lambda x: x[1])[0] if invert else max(valid, key=lambda x: x[1])[0]
        cells = ""
        for i, v in enumerate(vals):
            marker = " ◄" if i == best_idx else "  "
            cells += f"{v:>12}{marker}"
        print(f"{label:<22}{cells}")

    row("Total Return",    "total_return",  "{:+.2%}")
    row("CAGR",            "cagr",          "{:+.2%}")
    row("Sharpe",          "sharpe",        "{:.3f}")
    row("Sortino",         "sortino",       "{:.3f}")
    row("Calmar",          "calmar",        "{:.3f}")
    row("Max Drawdown",    "max_drawdown",  "{:+.2%}",  invert=True)
    row("MaxDD Days",      "max_dd_days",   "{:.0f}",   invert=True)
    row("Win Rate",        "win_rate",      "{:.1%}")
    row("Profit Factor",   "profit_factor", "{:.3f}")
    row("Total Trades",    "total_trades",  "{:.0f}",   invert=True)
    row("Final Equity",    "final_equity",  "${:,.0f}")

    print("─"*80)

    # ── Scores ────────────────────────────────────────────────────────────────
    scores = {s: (_score(results[s]) if results[s] else -999) for s in SETS}
    cells = "".join(f"{scores[s]:>12.3f}  " for s in SETS)
    print(f"{'Composite Score':<22}{cells}")
    print("─"*80)

    winner = max(scores, key=scores.__getitem__)
    winner_perf = results[winner]

    print(f"\n{'═'*80}")
    print(f"  RECOMMENDATION: {winner.upper()}")
    print(f"{'═'*80}")
    if winner_perf:
        print(f"  Sharpe    : {float(winner_perf['sharpe']):.3f}")
        print(f"  Calmar    : {float(winner_perf['calmar']):.3f}")
        print(f"  Max DD    : {float(winner_perf['max_drawdown']):.2%}")
        print(f"  CAGR      : {float(winner_perf['cagr']):.2%}")
        print(f"  Score     : {scores[winner]:.3f}")
    print()

    # Score commentary
    ranked = sorted(SETS, key=lambda s: scores[s], reverse=True)
    print("  Ranking (best → worst):")
    for i, s in enumerate(ranked):
        print(f"    {i+1}. {s:<14}  score={scores[s]:.3f}")

    # ── Write active_set ──────────────────────────────────────────────────────
    if winner != "base":
        active_set_path = ROOT / "config" / "active_set"
        print(f"\n  Writing '{winner}' to config/active_set …")
        active_set_path.write_text(winner)
        print(f"  Done. Future runs (without --set) will use '{winner}'.")
    else:
        print("\n  Winner is base settings — config/active_set left unchanged.")

    # ── Summary of output dirs ─────────────────────────────────────────────────
    print(f"\n{'─'*80}")
    print("  Output directories:")
    for s in SETS:
        d = dirs[s]
        print(f"    {s:<14}  {d.name if d else 'FAILED'}")
    print()


if __name__ == "__main__":
    main()
