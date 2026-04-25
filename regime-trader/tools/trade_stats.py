"""
trade_stats.py
--------------
Statistiques détaillées sur les trades à partir des résultats de backtest.
Utilise la courbe d'équité (barres journalières) ET les round-trips par symbole.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent.parent / "savedresults"


def find_latest(asset_group_hint: str = "SPY") -> Path:
    candidates = []
    for d in sorted(RESULTS_DIR.iterdir()):
        if not d.is_dir() or not d.name.startswith("backtest_"):
            continue
        perf = d / "performance_summary.csv"
        if not perf.exists():
            continue
        df = pd.read_csv(perf, index_col=0, header=None)
        syms = df.loc["symbols"].iloc[0] if "symbols" in df.index else ""
        if asset_group_hint in syms:
            candidates.append(d)
    if not candidates:
        raise FileNotFoundError("Aucun résultat trouvé")
    return candidates[-1]


def consecutive_runs(series: pd.Series) -> dict:
    """Max/avg séquences consécutives gagnantes et perdantes."""
    wins  = (series > 0).astype(int)
    loses = (series < 0).astype(int)

    def max_consec(arr):
        max_run = cur = 0
        for v in arr:
            cur = cur + 1 if v else 0
            max_run = max(max_run, cur)
        return max_run

    def avg_consec(arr):
        runs, cur = [], 0
        for v in arr:
            if v:
                cur += 1
            else:
                if cur:
                    runs.append(cur)
                cur = 0
        if cur:
            runs.append(cur)
        return np.mean(runs) if runs else 0.0

    return {
        "max_win_streak":  max_consec(wins),
        "max_loss_streak": max_consec(loses),
        "avg_win_streak":  avg_consec(wins),
        "avg_loss_streak": avg_consec(loses),
    }


def round_trip_pnl(trade_log: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule le P&L de chaque round-trip par symbole.
    Un round-trip = accumulation de position puis retour à zéro (ou inversion).
    Retourne un DataFrame avec colonnes: symbol, entry_date, exit_date,
    shares, entry_price, exit_price, pnl, regime.
    """
    trade_log = trade_log.copy()
    trade_log["timestamp"] = pd.to_datetime(trade_log["timestamp"])
    trade_log = trade_log.sort_values(["symbol", "timestamp"])

    results = []
    for sym, grp in trade_log.groupby("symbol"):
        position  = 0.0
        cost_basis = 0.0      # total cash paid for current long position
        entry_date = None
        entry_regime = ""

        for _, row in grp.iterrows():
            delta  = float(row["delta_shares"])
            price  = float(row["fill_price"])
            date   = row["timestamp"]
            regime = str(row.get("regime", ""))

            if position == 0 and delta > 0:
                # Fresh entry
                position   = delta
                cost_basis = delta * price
                entry_date = date
                entry_regime = regime

            elif position > 0 and delta > 0:
                # Add to position (average in)
                cost_basis += delta * price
                position   += delta

            elif position > 0 and delta < 0:
                # Partial or full exit
                exit_shares  = min(-delta, position)
                avg_entry    = cost_basis / position if position else price
                pnl          = exit_shares * (price - avg_entry)
                results.append({
                    "symbol":      sym,
                    "entry_date":  entry_date,
                    "exit_date":   date,
                    "shares":      exit_shares,
                    "entry_price": avg_entry,
                    "exit_price":  price,
                    "pnl":         pnl,
                    "regime":      entry_regime,
                    "hold_days":   (date - entry_date).days if entry_date else 0,
                })
                position   += delta   # delta is negative
                cost_basis  = (position * avg_entry) if position > 0 else 0.0
                if position <= 0:
                    position   = 0.0
                    cost_basis = 0.0
                    entry_date = None

    return pd.DataFrame(results)


def print_section(title: str) -> None:
    print(f"\n{'─'*62}")
    print(f"  {title}")
    print(f"{'─'*62}")


def run(result_dir: Path) -> None:
    equity  = pd.read_csv(result_dir / "equity_curve.csv",
                          index_col=0, parse_dates=True).iloc[:, 0]
    trades  = pd.read_csv(result_dir / "trade_log.csv")
    regimes = pd.read_csv(result_dir / "regime_history.csv",
                          index_col=0, parse_dates=True).iloc[:, 0]
    perf    = pd.read_csv(result_dir / "performance_summary.csv",
                          index_col=0, header=None)

    print(f"\n{'='*62}")
    print(f"  Résultats : {result_dir.name}")
    syms = perf.loc["symbols"].iloc[0] if "symbols" in perf.index else ""
    print(f"  Symboles  : {syms}")
    print(f"  Capital   : $100,000  →  ${float(perf.loc['final_equity'].iloc[0]):,.0f}")
    print(f"  Retour    : {float(perf.loc['total_return'].iloc[0])*100:+.2f}%  "
          f"Sharpe {float(perf.loc['sharpe'].iloc[0]):.3f}  "
          f"MaxDD {float(perf.loc['max_drawdown'].iloc[0])*100:.2f}%")

    # ── 1. Statistiques sur barres journalières ──────────────────────────────
    daily_pnl = equity.diff().dropna()
    wins  = daily_pnl[daily_pnl > 0]
    loses = daily_pnl[daily_pnl < 0]
    flat  = daily_pnl[daily_pnl == 0]

    print_section("BARRES JOURNALIÈRES (equity curve)")
    print(f"  Total barres        : {len(daily_pnl):>6}")
    print(f"  Barres gagnantes    : {len(wins):>6}  ({len(wins)/len(daily_pnl)*100:.1f}%)")
    print(f"  Barres perdantes    : {len(loses):>6}  ({len(loses)/len(daily_pnl)*100:.1f}%)")
    print(f"  Barres neutres      : {len(flat):>6}  ({len(flat)/len(daily_pnl)*100:.1f}%)")
    print(f"  Gain moyen/barre    : ${wins.mean():>10,.2f}")
    print(f"  Perte moyenne/barre : ${loses.mean():>10,.2f}")
    print(f"  Plus grand gain     : ${wins.max():>10,.2f}")
    print(f"  Plus grande perte   : ${loses.min():>10,.2f}")
    print(f"  Ratio gain/perte    : {abs(wins.mean()/loses.mean()):.2f}x")

    runs = consecutive_runs(daily_pnl)
    print(f"\n  Séquences consécutives (barres) :")
    print(f"    Max gagnantes  : {runs['max_win_streak']:>4}  barres")
    print(f"    Max perdantes  : {runs['max_loss_streak']:>4}  barres")
    print(f"    Moy gagnantes  : {runs['avg_win_streak']:>6.1f} barres")
    print(f"    Moy perdantes  : {runs['avg_loss_streak']:>6.1f} barres")

    # ── 2. Round-trips par symbole ───────────────────────────────────────────
    rt = round_trip_pnl(trades)
    if rt.empty:
        print("\n  (pas de round-trips calculables)")
        return

    rt_wins  = rt[rt["pnl"] > 0]
    rt_loses = rt[rt["pnl"] < 0]

    print_section("ROUND-TRIPS PAR SYMBOLE")
    print(f"  Total round-trips      : {len(rt):>6}")
    print(f"  Gagnants               : {len(rt_wins):>6}  ({len(rt_wins)/len(rt)*100:.1f}%)")
    print(f"  Perdants               : {len(rt_loses):>6}  ({len(rt_loses)/len(rt)*100:.1f}%)")
    print(f"  Gain moyen             : ${rt_wins['pnl'].mean():>10,.2f}")
    print(f"  Perte moyenne          : ${rt_loses['pnl'].mean():>10,.2f}")
    print(f"  Plus grand gain        : ${rt_wins['pnl'].max():>10,.2f}  ({rt_wins.loc[rt_wins['pnl'].idxmax(), 'symbol']})")
    print(f"  Plus grande perte      : ${rt_loses['pnl'].min():>10,.2f}  ({rt_loses.loc[rt_loses['pnl'].idxmin(), 'symbol']})")
    print(f"  Profit factor          : {rt_wins['pnl'].sum() / abs(rt_loses['pnl'].sum()):.2f}x")
    print(f"  Durée moyenne (jours)  : {rt['hold_days'].mean():.1f}")
    print(f"  Durée médiane  (jours) : {rt['hold_days'].median():.1f}")
    print(f"  Durée max      (jours) : {rt['hold_days'].max():.0f}")

    runs_rt = consecutive_runs(rt["pnl"])
    print(f"\n  Séquences consécutives (round-trips) :")
    print(f"    Max gagnants   : {runs_rt['max_win_streak']:>4}  trades")
    print(f"    Max perdants   : {runs_rt['max_loss_streak']:>4}  trades")
    print(f"    Moy gagnants   : {runs_rt['avg_win_streak']:>6.1f} trades")
    print(f"    Moy perdants   : {runs_rt['avg_loss_streak']:>6.1f} trades")

    # ── 3. Par régime ────────────────────────────────────────────────────────
    print_section("ROUND-TRIPS PAR RÉGIME (à l'entrée)")
    label_order = ["CRASH","STRONG_BEAR","BEAR","WEAK_BEAR",
                   "NEUTRAL","WEAK_BULL","BULL","STRONG_BULL","EUPHORIA"]
    sorted_labels = sorted(rt["regime"].unique(),
                           key=lambda l: label_order.index(l) if l in label_order else 99)
    header = f"  {'Régime':<14} {'N':>5}  {'Gagnants':>8}  {'WinRate':>7}  {'P&L total':>10}  {'Gain moy':>9}  {'Perte moy':>9}"
    print(header)
    print("  " + "─" * (len(header) - 2))
    for lbl in sorted_labels:
        sub = rt[rt["regime"] == lbl]
        w   = sub[sub["pnl"] > 0]
        l   = sub[sub["pnl"] < 0]
        wr  = len(w) / len(sub) * 100 if len(sub) else 0
        print(f"  {lbl:<14} {len(sub):>5}  {len(w):>8}  {wr:>6.1f}%  "
              f"{sub['pnl'].sum():>+10,.0f}  "
              f"{w['pnl'].mean():>+9,.0f}  "
              f"{l['pnl'].mean():>+9,.0f}" if len(l) else
              f"  {lbl:<14} {len(sub):>5}  {len(w):>8}  {wr:>6.1f}%  "
              f"{sub['pnl'].sum():>+10,.0f}  "
              f"{w['pnl'].mean():>+9,.0f}  {'n/a':>9}")

    # ── 4. Par symbole ───────────────────────────────────────────────────────
    print_section("ROUND-TRIPS PAR SYMBOLE")
    sym_stats = rt.groupby("symbol").apply(lambda s: pd.Series({
        "N":         len(s),
        "Gagnants":  (s["pnl"] > 0).sum(),
        "WinRate%":  (s["pnl"] > 0).mean() * 100,
        "P&L total": s["pnl"].sum(),
        "Gain moy":  s[s["pnl"] > 0]["pnl"].mean() if (s["pnl"] > 0).any() else 0,
        "Perte moy": s[s["pnl"] < 0]["pnl"].mean() if (s["pnl"] < 0).any() else 0,
    })).sort_values("P&L total", ascending=False)

    print(f"  {'Symbole':<8} {'N':>5}  {'Gagnants':>8}  {'WinRate':>7}  {'P&L total':>10}  {'Gain moy':>9}  {'Perte moy':>9}")
    print("  " + "─" * 72)
    for sym, row in sym_stats.iterrows():
        print(f"  {sym:<8} {int(row['N']):>5}  {int(row['Gagnants']):>8}  "
              f"{row['WinRate%']:>6.1f}%  {row['P&L total']:>+10,.0f}  "
              f"{row['Gain moy']:>+9,.0f}  {row['Perte moy']:>+9,.0f}")

    # ── 5. Distribution du P&L ───────────────────────────────────────────────
    print_section("DISTRIBUTION DU P&L (round-trips)")
    pcts = [10, 25, 50, 75, 90]
    for p in pcts:
        print(f"  Percentile {p:>3}  :  ${np.percentile(rt['pnl'], p):>+10,.2f}")
    print(f"  Écart-type    :  ${rt['pnl'].std():>10,.2f}")


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    hint = sys.argv[1] if len(sys.argv) > 1 else "SPY,QQQ,AAPL"
    # hint peut être "SPY,QQQ,AAPL" pour Stocks ou "DIA" pour Indices
    search = hint.split(",")[0]
    d = find_latest(search)
    print(f"Chargement : {d.name}")
    run(d)
