"""Sweep ATR stop-loss multipliers (MidVol & HighVol jointly) and emit a report.

Sweeps factor in [1.0, 1.5, 2.0, 3.0]:
  mid_vol_atr_mult  = 0.5 * factor
  high_vol_atr_mult = 1.0 * factor

LowVol stop is NOT swept. EMA period (50), ATR period (14), enforce_stops, HMM,
allocator, and risk-manager all stay at defaults.

Restores config/settings.yaml byte-identically in try/finally.

Run: py -3.12 scripts/sweep_atr_multiplier.py
"""
from __future__ import annotations
import copy
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

os.environ["PYTHONIOENCODING"] = "utf-8"

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "config" / "settings.yaml"
SAVED = ROOT / "savedresults"
REPORT = SAVED / "sweep_atr_multiplier_report.md"

FACTORS = [1.0, 1.5, 2.0, 3.0]
MID_BASE = 0.5
HIGH_BASE = 1.0


def _patch_config(text: str, mid: float, high: float) -> str:
    """Replace the two ATR mult lines while preserving comments."""
    text = re.sub(
        r"mid_vol_atr_mult:\s*[\d.]+",
        f"mid_vol_atr_mult: {mid}",
        text,
    )
    text = re.sub(
        r"high_vol_atr_mult:\s*[\d.]+",
        f"high_vol_atr_mult: {high}",
        text,
    )
    return text


def _latest_savedresults_dir() -> Path:
    dirs = sorted(
        (d for d in SAVED.iterdir() if d.is_dir() and d.name.startswith("backtest_")),
        key=lambda d: d.stat().st_mtime,
    )
    return dirs[-1]


def _load_metrics(out_dir: Path) -> dict:
    perf = pd.read_csv(out_dir / "performance_summary.csv", header=None,
                       names=["k", "v"]).set_index("k")["v"].to_dict()
    trades = pd.read_csv(out_dir / "trade_log.csv")
    eq = pd.read_csv(out_dir / "equity_curve.csv")

    stop_outs = int((trades["action"] == "STOP_OUT").sum()) if "action" in trades.columns else 0

    # per-fold trade & stop counts
    per_fold = trades.groupby("fold").agg(
        stops=("action", lambda x: (x == "STOP_OUT").sum()),
        trades=("action", "count"),
    )

    # mean holding period: pair BUY/STOP_OUT/SELL by symbol within fold
    mean_hold = _estimate_mean_hold(trades)

    return {
        "total_return": float(perf.get("total_return", 0)),
        "cagr": float(perf.get("cagr", 0)),
        "sharpe": float(perf.get("sharpe", 0)),
        "sortino": float(perf.get("sortino", 0)),
        "calmar": float(perf.get("calmar", 0)),
        "max_dd": float(perf.get("max_drawdown", 0)),
        "ann_vol": float(perf.get("annualized_volatility", 0)),
        "win_rate": float(perf.get("win_rate", 0)),
        "profit_factor": float(perf.get("profit_factor", 0)),
        "total_trades": int(len(trades)),
        "stop_outs": stop_outs,
        "mean_hold_bars": mean_hold,
        "per_fold": per_fold,
        "out_dir": str(out_dir),
    }


def _estimate_mean_hold(trades: pd.DataFrame) -> float:
    """Mean holding period in bars per closed position."""
    if "timestamp" not in trades.columns or "symbol" not in trades.columns:
        return float("nan")
    t = trades.copy()
    t["timestamp"] = pd.to_datetime(t["timestamp"])
    holds = []
    for sym, grp in t.groupby("symbol"):
        grp = grp.sort_values("timestamp").reset_index(drop=True)
        open_ts = None
        for _, row in grp.iterrows():
            d = row.get("delta_shares", 0)
            if open_ts is None and d > 0:
                open_ts = row["timestamp"]
            elif open_ts is not None and (d < 0 or row.get("action") == "STOP_OUT"):
                holds.append((row["timestamp"] - open_ts).days)
                open_ts = None
    return float(sum(holds) / len(holds)) if holds else float("nan")


def run_one(factor: float) -> dict:
    mid = round(MID_BASE * factor, 4)
    high = round(HIGH_BASE * factor, 4)
    sys.stdout.write(f"\n=== factor={factor}  mid={mid}  high={high} ===\n")
    sys.stdout.flush()

    txt = _patch_config(BAK, mid, high)
    CFG.write_text(txt, encoding="utf-8")

    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    started = datetime.now()
    out = subprocess.run(
        ["py", "-3.12", "main.py", "backtest",
         "--asset-group", "stocks", "--start", "2020-01-01",
         "--no-telegram"],
        cwd=str(ROOT), env=env,
        capture_output=True, text=True,
        encoding="utf-8", errors="replace",
        timeout=900,
    )
    if out.returncode != 0:
        sys.stdout.write(f"  FAILED rc={out.returncode}\n  stderr tail:\n{(out.stderr or '')[-800:]}\n")
        return {"factor": factor, "mid": mid, "high": high, "error": True,
                "stderr": (out.stderr or "")[-400:]}

    out_dir = _latest_savedresults_dir()
    if out_dir.stat().st_mtime < started.timestamp() - 5:
        sys.stdout.write(f"  WARN: latest dir {out_dir} predates run; skipping\n")
        return {"factor": factor, "mid": mid, "high": high, "error": True,
                "stderr": "no fresh output dir"}

    m = _load_metrics(out_dir)
    m.update({"factor": factor, "mid": mid, "high": high, "error": False})
    sys.stdout.write(
        f"  Return={m['total_return']:+.2%}  Sharpe={m['sharpe']:.3f}  "
        f"MaxDD={m['max_dd']:+.2%}  Trades={m['total_trades']}  Stops={m['stop_outs']}\n"
    )
    sys.stdout.flush()
    return m


def _is_dominated(row, others) -> bool:
    """True if some other row beats `row` on Sharpe AND Calmar AND MaxDD."""
    for o in others:
        if o is row or o.get("error"):
            continue
        if (o["sharpe"] >= row["sharpe"]
            and o["calmar"] >= row["calmar"]
            and o["max_dd"] >= row["max_dd"]
            and (o["sharpe"] > row["sharpe"] or o["calmar"] > row["calmar"]
                 or o["max_dd"] > row["max_dd"])):
            return True
    return False


def write_report(results: list[dict]) -> None:
    lines = []
    lines.append("# ATR Stop-Loss Multiplier Sweep")
    lines.append("")
    lines.append(f"_Run: {datetime.now().strftime('%Y-%m-%d %H:%M')}_")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append("- **Sweep**: `factor ∈ [1.0, 1.5, 2.0, 3.0]` applied jointly")
    lines.append("- `mid_vol_atr_mult = 0.5 × factor`")
    lines.append("- `high_vol_atr_mult = 1.0 × factor`")
    lines.append("- LowVol stop unchanged. EMA(50), ATR(14), enforce_stops=True, HMM, allocator, risk all default.")
    lines.append("- Backtest: `--asset-group stocks --start 2020-01-01` (10-symbol stocks group)")
    lines.append("- Walk-forward: IS 252 / OOS 63 / step 63 → 17 folds")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| factor | mid | high | Return | CAGR | Sharpe | Sortino | Calmar | MaxDD | AnnVol | Trades | Stops | Hold (d) | Dominated |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|")

    ok = [r for r in results if not r.get("error")]
    for r in results:
        if r.get("error"):
            lines.append(
                f"| {r['factor']} | {r['mid']} | {r['high']} | ERROR | — | — | — | — | — | — | — | — | — | — |"
            )
            continue
        dom = "yes" if _is_dominated(r, ok) else ""
        lines.append(
            f"| {r['factor']} | {r['mid']} | {r['high']} | "
            f"{r['total_return']:+.2%} | {r['cagr']:+.2%} | "
            f"{r['sharpe']:.3f} | {r['sortino']:.3f} | {r['calmar']:.3f} | "
            f"{r['max_dd']:+.2%} | {r['ann_vol']:.2%} | "
            f"{r['total_trades']} | {r['stop_outs']} | "
            f"{r['mean_hold_bars']:.1f} | {dom} |"
        )

    lines.append("")
    lines.append("## Per-fold STOP_OUT distribution")
    lines.append("")
    if ok:
        # build wide table: rows=fold, cols=factor
        per_fold_tables = {r["factor"]: r["per_fold"]["stops"] for r in ok}
        all_folds = sorted(set().union(*[s.index for s in per_fold_tables.values()]))
        header = "| fold | " + " | ".join(f"f={f}" for f in per_fold_tables) + " |"
        lines.append(header)
        lines.append("|---:|" + "---:|" * len(per_fold_tables))
        for fold in all_folds:
            row = f"| {fold} | " + " | ".join(
                str(int(per_fold_tables[f].get(fold, 0))) for f in per_fold_tables
            ) + " |"
            lines.append(row)

    lines.append("")
    lines.append("## Interpretation notes")
    lines.append("")
    lines.append("- **Higher factor → wider stops → fewer stop-outs → more trend capture, but larger per-trade loss when a stop hits.**")
    lines.append("- A row marked `Dominated` is strictly worse on Sharpe AND Calmar AND MaxDD than at least one other row — eliminate it from consideration.")
    lines.append("- `factor=1.0` is the current default (matches BASELINE_AFTER_ENFORCE_STOPS for stocks).")
    lines.append("- Output dirs per row recorded in script stdout; intermediate `savedresults/backtest_*` dirs preserved.")
    lines.append("")
    lines.append("## Output dirs")
    lines.append("")
    for r in results:
        if not r.get("error"):
            lines.append(f"- factor={r['factor']}: `{r['out_dir']}`")

    REPORT.write_text("\n".join(lines), encoding="utf-8")
    sys.stdout.write(f"\nReport: {REPORT}\n")


# ─────────────────────────────────────────────────────────────────────────────
BAK = CFG.read_text(encoding="utf-8")
BAK_DEEP = copy.deepcopy(BAK)  # belt + suspenders

results: list[dict] = []
try:
    for f in FACTORS:
        results.append(run_one(f))
finally:
    CFG.write_text(BAK_DEEP, encoding="utf-8")
    after = CFG.read_text(encoding="utf-8")
    if after == BAK_DEEP:
        sys.stdout.write("\nsettings.yaml restored byte-identically.\n")
    else:
        sys.stdout.write("\n!!! settings.yaml RESTORE MISMATCH — INSPECT MANUALLY !!!\n")

write_report(results)
