"""Sweep sma_blend_weight on indices and report results.

Run: py -3.12 scripts/sweep_sma_blend.py
"""
from __future__ import annotations
import os
import re
import subprocess
import sys
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "config" / "settings.yaml"
BAK = CFG.read_text(encoding="utf-8")

WEIGHTS = [0.80, 0.85, 0.88, 0.92, 0.95]  # refined around 0.9 peak

results = []
try:
    for w in WEIGHTS:
        txt = BAK.replace("sma_blend_weight: 1.0", f"sma_blend_weight: {w}")
        CFG.write_text(txt, encoding="utf-8")
        sys.stdout.write(f"\n=== sma_blend_weight = {w} ===\n")
        sys.stdout.flush()
        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        out = subprocess.run(
            ["py", "-3.12", "main.py", "backtest",
             "--asset-group", "indices", "--start", "2020-01-01"],
            cwd=str(ROOT), env=env,
            capture_output=True, text=True,
            encoding="utf-8", errors="replace",
        )
        tail = out.stdout[-1500:] if out.stdout else ""
        m = re.search(
            r"Total return:\s*([+-][\d.]+%)\s+Sharpe:\s*([\d.]+)\s+MaxDD:\s*([+-][\d.]+%)",
            tail,
        )
        if m:
            results.append((w, m.group(1), m.group(2), m.group(3)))
            sys.stdout.write(
                f"  w={w}: Return={m.group(1)} Sharpe={m.group(2)} MaxDD={m.group(3)}\n"
            )
        else:
            results.append((w, "?", "?", "?"))
            sys.stdout.write(f"  w={w}: parse failed; tail={tail[-300:]}\n")
        sys.stdout.flush()
finally:
    CFG.write_text(BAK, encoding="utf-8")

sys.stdout.write("\n=== SUMMARY (indices, start 2020-01-01) ===\n")
sys.stdout.write(f"{'w_hmm':<6} {'Return':<10} {'Sharpe':<8} {'MaxDD':<10}\n")
sys.stdout.write("1.00   +39.59%   0.381    -14.03%  (baseline, already known)\n")
for r in results:
    sys.stdout.write(f"{r[0]:<6} {r[1]:<10} {r[2]:<8} {r[3]:<10}\n")
sys.stdout.write("\nReference: SMA-200 pure = +50.95% / 0.702 / -7.49%\n")
