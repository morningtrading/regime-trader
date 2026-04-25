"""
VIX sub-ablation: which VIX feature (if any) degrades the HMM on indices?

Tests 4 combos via features_override + use_vix_features, restores
settings on exit, prints a comparison table.

Run:
    PYTHONIOENCODING=utf-8 py -3.12 scripts/ablation_vix_indices.py
"""
from __future__ import annotations

import csv
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

import yaml

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = ROOT / "config" / "settings.yaml"
SAVED = ROOT / "savedresults"

BASE = ["log_ret_1", "realized_vol_20", "vol_ratio", "adx_14", "dist_sma200"]

COMBOS: List[Dict] = [
    {"label": "EXT_baseline",    "features": BASE,                                      "vix": False},
    {"label": "EXT+vix_zscore",  "features": BASE + ["vix_zscore_60"],                  "vix": True},
    {"label": "EXT+vix_level",   "features": BASE + ["vix_level"],                      "vix": True},
    {"label": "EXT+both_vix",    "features": BASE + ["vix_level", "vix_zscore_60"],     "vix": True},
    {"label": "BASE+vix_zscore", "features": ["log_ret_1", "realized_vol_20", "vix_zscore_60"], "vix": True},
]


def patch_settings(features: List[str], use_vix: bool) -> None:
    with SETTINGS.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("hmm", {})["features_override"] = features
    cfg["hmm"]["use_vix_features"] = use_vix
    tmp = SETTINGS.with_suffix(".yaml.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    os.replace(tmp, SETTINGS)


def newest_savedresults_dir() -> Path:
    dirs = [p for p in SAVED.iterdir() if p.is_dir() and p.name.startswith("backtest_")]
    return max(dirs, key=lambda p: p.stat().st_mtime)


def read_summary(folder: Path) -> Dict[str, float]:
    f = folder / "performance_summary.csv"
    metrics: Dict[str, float] = {}
    if not f.exists():
        return metrics
    with f.open("r", encoding="utf-8") as fh:
        for row in csv.reader(fh):
            if len(row) >= 2:
                try:
                    metrics[row[0]] = float(row[1])
                except ValueError:
                    pass
    return metrics


def run_backtest() -> Path:
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    cmd = [
        "py", "-3.12", "main.py", "backtest",
        "--asset-group", "indices",
        "--compare",
    ]
    before = set(p.name for p in SAVED.glob("backtest_*"))
    proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(f"Backtest failed: {proc.returncode}\n{proc.stderr[-2000:]}\n")
    time.sleep(1)
    after = set(p.name for p in SAVED.glob("backtest_*"))
    new = sorted(after - before)
    if new:
        return SAVED / new[-1]
    return newest_savedresults_dir()


def main() -> None:
    backup = SETTINGS.with_suffix(".yaml.vixablation_bak")
    shutil.copy2(SETTINGS, backup)
    print(f"Settings backup: {backup}")

    results = []
    try:
        for i, combo in enumerate(COMBOS, 1):
            label = combo["label"]
            feats = combo["features"]
            use_vix = combo["vix"]
            print(f"\n[{i}/{len(COMBOS)}] {label}: vix={use_vix} feats={feats}")
            t0 = time.time()
            patch_settings(feats, use_vix)
            folder = run_backtest()
            metrics = read_summary(folder)
            elapsed = time.time() - t0
            print(
                f"  -> folder={folder.name}  "
                f"ret={metrics.get('total_return', float('nan')):+.4f}  "
                f"sharpe={metrics.get('sharpe', float('nan')):.3f}  "
                f"maxdd={metrics.get('max_drawdown', float('nan')):.4f}  "
                f"({elapsed:.0f}s)"
            )
            results.append({
                "label": label, "features": feats, "vix": use_vix,
                "folder": folder.name, **metrics,
            })
    finally:
        shutil.copy2(backup, SETTINGS)
        print(f"\nRestored settings from {backup}")

    print("\n" + "=" * 78)
    print("VIX SUB-ABLATION - indices full period")
    print("=" * 78)
    print(f"{'Label':<20} {'#feat':>5} {'Return':>9} {'Sharpe':>7} {'MaxDD':>8} {'CAGR':>7}")
    print("-" * 78)
    for r in results:
        print(
            f"{r['label']:<20} {len(r['features']):>5} "
            f"{r.get('total_return', 0)*100:>+8.2f}% "
            f"{r.get('sharpe', 0):>7.3f} "
            f"{r.get('max_drawdown', 0)*100:>+7.2f}% "
            f"{r.get('cagr', 0)*100:>+6.2f}%"
        )
    print("=" * 78)


if __name__ == "__main__":
    main()
