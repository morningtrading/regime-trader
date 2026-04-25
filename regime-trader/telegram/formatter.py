"""
telegram/formatter.py — Compact Telegram messages for Regime Trader.
All messages: 2-4 lines max, longer lines, machine name included.
"""
from __future__ import annotations
import json
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent

_HOST = socket.gethostname()


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%d %b %Y %H:%M UTC")

def _pct(v: float) -> str:
    return f"{'+' if v >= 0 else ''}{v * 100:.2f}%"

def _latest_backtest_dir() -> Optional[Path]:
    sr = ROOT / "savedresults"
    if not sr.exists():
        return None
    dirs = sorted(sr.glob("backtest_*"), reverse=True)
    return dirs[0] if dirs else None


# ── Formatters ────────────────────────────────────────────────────────────────

def format_test() -> str:
    return (
        f"✅ <b>Regime Trader — connexion OK</b>\n"
        f"Machine: <code>{_HOST}</code>  ·  {_now()}"
    )


def format_backtest_summary() -> str:
    d = _latest_backtest_dir()
    if d is None:
        return "❌ Aucun résultat de backtest trouvé."

    csv_path = d / "performance_summary.csv"
    ctx_path = d / "run_context.json"
    if not csv_path.exists():
        return f"❌ performance_summary.csv introuvable dans {d.name}"

    import pandas as pd
    s = pd.read_csv(csv_path, header=None, index_col=0).squeeze()

    group   = "—"
    symbols = str(s.get("symbols", "—"))
    cfg_set = ""
    if ctx_path.exists():
        try:
            c = json.loads(ctx_path.read_text())
            group   = c.get("asset_group", "—")
            symbols = ", ".join(c.get("symbols", []))
            cfg_set = c.get("config_set", "")
        except Exception:
            pass

    ret    = float(s.get("total_return", 0))
    cagr   = float(s.get("cagr", 0))
    sharpe = float(s.get("sharpe", 0))
    dd     = float(s.get("max_drawdown", 0))
    calmar = float(s.get("calmar", 0))
    trades = int(float(s.get("total_trades", 0)))
    winr   = float(s.get("win_rate", 0))
    folds  = int(float(s.get("n_folds", 0)))
    start  = str(s.get("start", ""))[:10]
    end    = str(s.get("end", ""))[:10]

    set_str = f"  [{cfg_set}]" if cfg_set else ""
    return (
        f"📊 <b>Backtest — {group}</b>{set_str}  <code>{_HOST}</code>\n"
        f"<code>{symbols}</code>  ·  {start}→{end} ({folds} folds)\n"
        f"<b>{_pct(ret)}</b>  CAGR {_pct(cagr)}  ·  Sharpe <b>{sharpe:.2f}</b>  Calmar {calmar:.2f}  MaxDD {_pct(dd)}\n"
        f"{trades} trades  ·  {winr * 100:.1f}% win  ·  <i>{_now()}</i>"
    )


def format_latest_trades(n: int = 5) -> str:
    d = _latest_backtest_dir()
    if d is None:
        return "❌ Aucun résultat trouvé."

    tlog = d / "trade_log.csv"
    if not tlog.exists():
        return "❌ trade_log.csv introuvable."

    import pandas as pd
    df = pd.read_csv(tlog)
    if df.empty:
        return "ℹ️ Aucun trade enregistré."

    ret_col  = next((c for c in ["pnl_pct", "return", "pct_return", "trade_return"] if c in df.columns), None)
    sym_col  = next((c for c in ["symbol", "ticker", "sym"] if c in df.columns), None)
    date_col = next((c for c in ["exit_date", "date", "entry_date"] if c in df.columns), None)

    ctx_path = d / "run_context.json"
    group = "—"
    if ctx_path.exists():
        try:
            group = json.loads(ctx_path.read_text()).get("asset_group", "—")
        except Exception:
            pass

    lines = [f"📈 <b>Derniers trades — {group}</b>  <code>{_HOST}</code>"]
    for _, row in df.tail(n).iterrows():
        sym  = str(row[sym_col])  if sym_col  else "?"
        date = str(row[date_col])[:10] if date_col else "?"
        if ret_col:
            r = float(row[ret_col])
            icon = "🟢" if r >= 0 else "🔴"
            lines.append(f"  {icon} {sym:<6} {_pct(r)}   {date}")
        else:
            lines.append(f"  • {sym}   {date}")

    lines.append(f"<i>{_now()}</i>")
    return "\n".join(lines)


def format_stress_summary() -> str:
    d = _latest_backtest_dir()
    stress = None
    if d:
        p = d / "stress_test_summary.csv"
        if p.exists():
            stress = p
    if stress is None:
        all_stress = sorted(ROOT.glob("savedresults/backtest_*/stress_test_summary.csv"), reverse=True)
        stress = all_stress[0] if all_stress else None
    if stress is None:
        return "❌ Aucun stress test trouvé."

    import pandas as pd
    df = pd.read_csv(stress, index_col=0)

    ctx_path = stress.parent / "run_context.json"
    group = "—"
    if ctx_path.exists():
        try:
            group = json.loads(ctx_path.read_text()).get("asset_group", "—")
        except Exception:
            pass

    lines = [f"⚡ <b>Stress Test — {group}</b>  <code>{_HOST}</code>"]
    for scenario, row in df.iterrows():
        sharpe = row.get("sharpe", "?")
        dd     = row.get("max_drawdown", "?")
        icon   = "✅" if float(str(sharpe).replace(",", ".")) > 0 else "❌"
        lines.append(f"  {icon} <code>{scenario:<18}</code> Sh {sharpe}  DD {dd}")

    lines.append(f"<i>{_now()}</i>")
    return "\n".join(lines)


def format_regime_status() -> str:
    d = _latest_backtest_dir()
    if d is None:
        return "❌ Aucun résultat trouvé."

    rh = d / "regime_history.csv"
    if not rh.exists():
        return "❌ regime_history.csv introuvable."

    import pandas as pd
    df = pd.read_csv(rh, index_col=0)
    if df.empty:
        return "ℹ️ Historique de régime vide."

    last_date   = str(df.index[-1])[:10]
    last_regime = str(df.iloc[-1, 0])
    icons = {"BULL": "🟢", "EUPHORIA": "🚀", "BEAR": "🔴", "CRASH": "💥", "NEUTRAL": "⚪"}
    icon  = icons.get(last_regime.upper(), "📊")
    recent = " ".join(r[:3] for r in df.iloc[-8:, 0].tolist())

    return (
        f"{icon} <b>Régime: {last_regime}</b>  {last_date}  <code>{_HOST}</code>\n"
        f"Récent: <code>{recent}</code>  ·  <i>{_now()}</i>"
    )
