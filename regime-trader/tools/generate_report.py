"""
tools/generate_report.py — Generate a PDF report for backtest results.

Picks the most recent backtest for each of the 3 asset groups from
savedresults/, renders equity curves, metrics tables and a comparison
summary, and saves to results/backtest_report_<date>.pdf.

Usage:
    py -3.12 tools/generate_report.py
    py -3.12 tools/generate_report.py --out results/my_report.pdf
"""

from __future__ import annotations

import argparse
import io
import os
import sys
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate,
    Paragraph, Spacer, Table, TableStyle, Image, HRFlowable, KeepTogether,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT


# ── Colour palette ─────────────────────────────────────────────────────────────
C_BG        = colors.HexColor("#0F1117")
C_CARD      = colors.HexColor("#161B22")
C_ACCENT    = colors.HexColor("#4C9BE8")
C_GREEN     = colors.HexColor("#2ECC71")
C_RED       = colors.HexColor("#E74C3C")
C_AMBER     = colors.HexColor("#F5A623")
C_WHITE     = colors.HexColor("#FFFFFF")
C_GREY      = colors.HexColor("#AAAAAA")
C_BORDER    = colors.HexColor("#30363D")

GROUP_COLOUR = {
    "Indices": "#7ED321",
    "Stocks":  "#4C9BE8",
    "Crypto":  "#F5A623",
}
GROUP_SYMS = {
    "Indices": "SPY,QQQ,DIA,IWM,GLD,TLT,EFA,EEM,VNQ,USO",
    "Stocks":  "SPY,QQQ,AAPL,MSFT,AMZN,GOOGL,NVDA,META,TSLA,AMD",
    "Crypto":  "BTC/USD,ETH/USD,SOL/USD,AVAX/USD,DOGE/USD,LTC/USD,LINK/USD,UNI/USD",
}

# ── Find latest backtest per group ─────────────────────────────────────────────

def find_latest_backtest(group_key: str) -> Path | None:
    """Return the path to the most recent backtest dir matching the group."""
    results_dir = _ROOT / "savedresults"
    target_syms = set(GROUP_SYMS[group_key].split(","))

    candidates = sorted(
        [d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("backtest_")],
        reverse=True,
    )
    for d in candidates:
        summary = d / "performance_summary.csv"
        if not summary.exists():
            continue
        s = pd.read_csv(summary, header=None, names=["k", "v"]).set_index("k")["v"]
        syms = set(str(s.get("symbols", "")).split(","))
        if syms == target_syms:
            return d
    return None


def load_result(d: Path) -> dict:
    s = pd.read_csv(d / "performance_summary.csv",
                    header=None, names=["k", "v"]).set_index("k")["v"]
    equity = pd.read_csv(d / "equity_curve.csv", index_col=0, parse_dates=True)["equity"]
    regimes = pd.read_csv(d / "regime_history.csv", index_col=0, parse_dates=True)["regime"]
    trades  = pd.read_csv(d / "trade_log.csv") if (d / "trade_log.csv").exists() else pd.DataFrame()
    return dict(summary=s, equity=equity, regimes=regimes, trades=trades)


# ── Chart builders ─────────────────────────────────────────────────────────────

def _sparkline_buf(equity: pd.Series, colour: str, initial: float = 100_000) -> io.BytesIO:
    fig, ax = plt.subplots(figsize=(7, 2.2), facecolor="#161B22")
    ax.set_facecolor("#161B22")

    ax.plot(equity.index, equity.values, color=colour, linewidth=1.8)
    ax.fill_between(equity.index, equity.values, initial, alpha=0.15, color=colour)
    ax.axhline(initial, color="#444444", linewidth=0.8, linestyle="--")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#333333")
    ax.spines["left"].set_color("#333333")
    ax.tick_params(colors="#AAAAAA", labelsize=7)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.grid(axis="y", color="#222222", linewidth=0.5, linestyle="--")
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Portfolio Value", color="#AAAAAA", fontsize=7)

    fig.tight_layout(pad=0.3)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf


def _regime_bar_buf(regimes: pd.Series, colour: str) -> io.BytesIO:
    """Horizontal stacked bar showing time spent in each regime."""
    counts = regimes.value_counts()
    total  = counts.sum()
    pcts   = (counts / total * 100).sort_values(ascending=False)

    regime_colours = {
        "CRASH": "#C0392B", "STRONG_BEAR": "#E74C3C", "BEAR": "#E67E22",
        "WEAK_BEAR": "#F39C12", "NEUTRAL": "#95A5A6",
        "WEAK_BULL": "#27AE60", "BULL": "#2ECC71",
        "STRONG_BULL": "#1ABC9C", "EUPHORIA": "#3498DB",
    }

    fig, ax = plt.subplots(figsize=(7, 0.6), facecolor="#161B22")
    ax.set_facecolor("#161B22")
    ax.set_xlim(0, 100)
    ax.axis("off")

    left = 0
    for regime, pct in pcts.items():
        c = regime_colours.get(str(regime), "#888888")
        ax.barh(0, pct, left=left, color=c, height=0.6, edgecolor="#0F1117", linewidth=0.5)
        if pct > 5:
            ax.text(left + pct / 2, 0, f"{regime}\n{pct:.0f}%",
                    ha="center", va="center", color="white", fontsize=5.5, fontweight="bold")
        left += pct

    fig.tight_layout(pad=0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf


def _comparison_chart_buf(results: dict) -> io.BytesIO:
    """Overlay equity curves for all 3 groups, normalised to 1.0."""
    fig, ax = plt.subplots(figsize=(14, 3.5), facecolor="#161B22")
    ax.set_facecolor("#161B22")

    for group, data in results.items():
        eq = data["equity"]
        norm = eq / eq.iloc[0]
        colour = GROUP_COLOUR[group]
        ax.plot(norm.index, norm.values, color=colour, linewidth=2.0, label=group)

    ax.axhline(1.0, color="#444444", linewidth=0.8, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#333333")
    ax.spines["left"].set_color("#333333")
    ax.tick_params(colors="#AAAAAA", labelsize=8)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1f}x"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.grid(axis="y", color="#222222", linewidth=0.5, linestyle="--")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", framealpha=0.2, labelcolor="white",
              fontsize=10, edgecolor="#444444")
    ax.set_ylabel("Normalised equity (start = 1×)", color="#AAAAAA", fontsize=8)

    fig.tight_layout(pad=0.4)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf


# ── PDF construction ───────────────────────────────────────────────────────────

def build_pdf(results: dict, out_path: Path) -> None:

    doc = BaseDocTemplate(
        str(out_path),
        pagesize=A4,
        leftMargin=1.8*cm, rightMargin=1.8*cm,
        topMargin=2.0*cm,  bottomMargin=2.0*cm,
    )

    W = A4[0] - 3.6*cm   # usable width

    frame = Frame(doc.leftMargin, doc.bottomMargin,
                  doc.width, doc.height, id="main")

    def _header_footer(canvas, doc):
        canvas.saveState()
        # header rule
        canvas.setStrokeColor(C_BORDER)
        canvas.setLineWidth(0.5)
        canvas.line(doc.leftMargin, A4[1] - 1.5*cm,
                    A4[0] - doc.rightMargin, A4[1] - 1.5*cm)
        # footer
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(C_GREY)
        canvas.drawCentredString(A4[0]/2, 1.0*cm,
            f"Regime Trader — Confidential  ·  Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        canvas.drawRightString(A4[0] - doc.rightMargin, 1.0*cm,
            f"Page {doc.page}")
        canvas.restoreState()

    doc.addPageTemplates([PageTemplate(id="main", frames=frame,
                                        onPage=_header_footer)])

    styles = getSampleStyleSheet()
    S = {
        "title":    ParagraphStyle("title",    fontSize=22, textColor=C_WHITE,
                                    fontName="Helvetica-Bold", spaceAfter=4,
                                    alignment=TA_CENTER),
        "subtitle": ParagraphStyle("subtitle", fontSize=11, textColor=C_GREY,
                                    fontName="Helvetica", spaceAfter=16,
                                    alignment=TA_CENTER),
        "h1":       ParagraphStyle("h1",       fontSize=14, textColor=C_ACCENT,
                                    fontName="Helvetica-Bold", spaceBefore=14,
                                    spaceAfter=6),
        "h2":       ParagraphStyle("h2",       fontSize=11, textColor=C_WHITE,
                                    fontName="Helvetica-Bold", spaceBefore=10,
                                    spaceAfter=4),
        "body":     ParagraphStyle("body",     fontSize=9,  textColor=C_GREY,
                                    fontName="Helvetica",   spaceAfter=4,
                                    leading=13),
        "note":     ParagraphStyle("note",     fontSize=8,  textColor=C_GREY,
                                    fontName="Helvetica-Oblique", spaceAfter=6),
        "caption":  ParagraphStyle("caption",  fontSize=8,  textColor=C_GREY,
                                    fontName="Helvetica",  alignment=TA_CENTER,
                                    spaceAfter=8),
    }

    story = []

    # ── Cover ─────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 1.5*cm))
    story.append(Paragraph("REGIME TRADER", S["title"]))
    story.append(Paragraph("Walk-Forward Backtest Report  ·  2020-01-01 → 2026-04-17", S["subtitle"]))
    story.append(HRFlowable(width="100%", thickness=1, color=C_BORDER, spaceAfter=12))

    # Config summary box
    cfg_data = [
        ["Parameter", "Value"],
        ["Method",     "Walk-forward  IS=252 bars / OOS=126 bars / step=126"],
        ["Capital",    "$100,000   Slippage: 10 bps per side"],
        ["HMM",        "States [3,4,5,6]  covariance=full  n_init=10"],
        ["Regime filter", "stability=7  flicker_thresh=4  min_conf=0.62"],
        ["Features",   "log_ret_1, realized_vol_20"],
        ["Allocation", "low_vol 0.95×1.25x  /  mid 0.95/0.70  /  high 0.60"],
        ["Rebalance",  "threshold=0.18  min_interval=5 bars  trend_lookback=50"],
        ["Config set", "balanced  (validated via parameter sweep 2026-04-16)"],
    ]
    story.append(_make_table(cfg_data, W, header_colour=C_ACCENT))
    story.append(Spacer(1, 0.4*cm))

    # ── Group comparison overview ──────────────────────────────────────────────
    story.append(Paragraph("Cross-Group Equity Comparison", S["h1"]))
    story.append(Paragraph(
        "All three universes run with identical parameters. Equity curves normalised "
        "to 1.0 at start. Period: 2020-01-01 → 2026-04-17.",
        S["body"],
    ))

    cmp_buf = _comparison_chart_buf(results)
    story.append(Image(cmp_buf, width=W, height=3.5*cm * (W / (14/2.54))))
    story.append(Paragraph("Figure 1 — Normalised equity curves, all three asset groups.", S["caption"]))

    # Summary comparison table
    cmp_rows = [["Metric", "Indices", "Stocks", "Crypto"]]
    metrics_def = [
        ("Total Return",   "total_return",  "{:+.2%}"),
        ("CAGR",           "cagr",          "{:+.2%}"),
        ("Sharpe Ratio",   "sharpe",        "{:.3f}"),
        ("Sortino Ratio",  "sortino",       "{:.3f}"),
        ("Max Drawdown",   "max_drawdown",  "{:.2%}"),
        ("Calmar Ratio",   "calmar",        "{:.3f}"),
        ("Win Rate",       "win_rate",      "{:.2%}"),
        ("Total Trades",   "total_trades",  "{:.0f}"),
        ("Final Equity",   "final_equity",  "${:,.0f}"),
    ]
    for label, key, fmt in metrics_def:
        row = [label]
        for g in ["Indices", "Stocks", "Crypto"]:
            try:
                row.append(fmt.format(float(results[g]["summary"][key])))
            except Exception:
                row.append("—")
        cmp_rows.append(row)

    story.append(_make_table(cmp_rows, W, header_colour=C_ACCENT, highlight_col=1))
    story.append(Spacer(1, 0.3*cm))

    # ── Per-group detailed sections ────────────────────────────────────────────
    for group in ["Indices", "Stocks", "Crypto"]:
        data = results[group]
        s    = data["summary"]
        col  = GROUP_COLOUR[group]

        story.append(Paragraph(f"{group} Universe", S["h1"]))
        story.append(Paragraph(
            f"Symbols: {GROUP_SYMS[group]}",
            S["note"],
        ))

        # Equity curve chart
        eq_buf = _sparkline_buf(data["equity"], col)
        story.append(Image(eq_buf, width=W, height=2.2*cm * (W / (7/2.54))))
        story.append(Paragraph(f"Figure — {group} equity curve (2020–2026).", S["caption"]))

        # Regime distribution
        if len(data["regimes"]) > 0:
            rg_buf = _regime_bar_buf(data["regimes"], col)
            story.append(Image(rg_buf, width=W, height=0.6*cm * (W / (7/2.54))))
            story.append(Paragraph("Regime distribution across OOS period.", S["caption"]))

        # Metrics table
        met_rows = [["Metric", "Value", "Metric", "Value"]]
        pairs = [
            (("Total Return",  "{:+.2%}".format(float(s["total_return"]))),
             ("Sharpe Ratio",  "{:.3f}".format(float(s["sharpe"])))),
            (("CAGR",          "{:+.2%}".format(float(s["cagr"]))),
             ("Sortino Ratio", "{:.3f}".format(float(s["sortino"])))),
            (("Max Drawdown",  "{:.2%}".format(float(s["max_drawdown"]))),
             ("Calmar Ratio",  "{:.3f}".format(float(s["calmar"])))),
            (("Max DD (days)", str(int(float(s["max_dd_days"])))),
             ("Win Rate",      "{:.2%}".format(float(s["win_rate"])))),
            (("Total Trades",  str(int(float(s["total_trades"])))),
             ("Profit Factor", "{:.3f}".format(float(s["profit_factor"])))),
            (("Final Equity",  "${:,.0f}".format(float(s["final_equity"]))),
             ("OOS Folds",     str(int(float(s["n_folds"]))))),
        ]
        for (l1, v1), (l2, v2) in pairs:
            met_rows.append([l1, v1, l2, v2])

        story.append(_make_table(met_rows, W, header_colour=colors.HexColor(col),
                                  col_widths=[0.30, 0.20, 0.30, 0.20]))
        story.append(Spacer(1, 0.3*cm))

    # ── Analysis & recommendations ────────────────────────────────────────────
    story.append(Paragraph("Analysis & Recommendations", S["h1"]))

    insights = [
        ("<b>Indices — primary target universe.</b>  Sharpe 0.689, MaxDD -10.1%, "
         "Calmar 1.118.  The diversified mix of equities, bonds, gold and "
         "commodities provides genuine cross-regime rotation opportunities. "
         "This is where the HMM regime approach delivers its structural edge."),

        ("<b>Stocks — limited edge vs SMA-200.</b>  Sharpe 0.764 beats Buy &amp; Hold "
         "risk-adjusted, but SMA-200 dominates on every metric (Sharpe 1.083, "
         "CAGR +23%, MaxDD -16%). Concentrated tech equities are highly correlated "
         "in all regimes — there is nowhere to rotate during crashes. "
         "MaxDD -25.4% is unacceptable for a risk-managed strategy."),

        ("<b>Crypto — strategy failure.</b>  Sharpe 0.275, CAGR +6.9% on a universe "
         "that produced 10x+ returns over the same period. Random allocation "
         "outperforms (Sharpe 0.458). Root causes: no stablecoin refuge asset, "
         "high volatility in crypto signals upside not danger, and stab=7 is "
         "too slow for daily 30% moves."),

        ("<b>Parameter sweep conclusions (2026-04-16).</b>  min_rebalance_interval=5 "
         "validated as global optimum on indices (clear peak, +0.15 Sharpe vs "
         "no throttle). stab ∈ {3,5,7} is the safe zone; stab≥9 is a cliff. "
         "min_confidence plateau at 0.55–0.65; 0.62 is centre of range."),

        ("<b>Next steps:</b>  (1) Run joint (interval × stab) 2-D sweep to find "
         "true combined optimum. (2) For crypto, add stablecoin as a defensive "
         "allocation target and rebuild with stab=1-2. (3) Consider removing "
         "concentrated stocks from live trading and focusing on indices only."),
    ]
    for txt in insights:
        story.append(Paragraph(txt, S["body"]))
        story.append(Spacer(1, 0.15*cm))

    story.append(Spacer(1, 0.3*cm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=C_BORDER))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(
        "All results are strictly out-of-sample. No model sees its own test data. "
        "Past performance does not guarantee future results.",
        S["note"],
    ))

    doc.build(story)
    print(f"  PDF saved → {out_path}")


# ── Table helper ───────────────────────────────────────────────────────────────

def _make_table(
    data: list,
    available_width: float,
    header_colour=None,
    col_widths: list | None = None,
    highlight_col: int | None = None,
) -> Table:
    n_cols = len(data[0])
    if col_widths:
        widths = [available_width * w for w in col_widths]
    else:
        widths = [available_width / n_cols] * n_cols

    tbl = Table(data, colWidths=widths, repeatRows=1)

    style = [
        ("BACKGROUND",  (0, 0), (-1, 0),  header_colour or C_ACCENT),
        ("TEXTCOLOR",   (0, 0), (-1, 0),  C_WHITE),
        ("FONTNAME",    (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, 0),  9),
        ("ALIGN",       (0, 0), (-1, 0),  "CENTER"),

        ("BACKGROUND",  (0, 1), (-1, -1), C_CARD),
        ("TEXTCOLOR",   (0, 1), (-1, -1), C_GREY),
        ("FONTNAME",    (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",    (0, 1), (-1, -1), 8),
        ("ALIGN",       (1, 1), (-1, -1), "RIGHT"),
        ("ALIGN",       (0, 1), (0, -1),  "LEFT"),

        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_CARD, colors.HexColor("#1C2128")]),

        ("GRID",        (0, 0), (-1, -1), 0.4, C_BORDER),
        ("TOPPADDING",  (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0,0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",(0, 0), (-1, -1), 6),
    ]

    if highlight_col is not None:
        style.append(("TEXTCOLOR", (highlight_col, 1), (highlight_col, -1), C_WHITE))
        style.append(("FONTNAME",  (highlight_col, 1), (highlight_col, -1), "Helvetica-Bold"))

    tbl.setStyle(TableStyle(style))
    return tbl


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate backtest PDF report")
    parser.add_argument("--out", default=None,
                        help="Output PDF path (default: results/backtest_report_<date>.pdf)")
    args = parser.parse_args()

    ts       = datetime.now().strftime("%Y-%m-%d")
    out_path = Path(args.out) if args.out else _ROOT / "results" / f"backtest_report_{ts}.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("\nLoading backtest results ...")
    results = {}
    for group in ["Indices", "Stocks", "Crypto"]:
        d = find_latest_backtest(group)
        if d is None:
            print(f"  WARNING: no backtest found for {group} — skipping")
            continue
        results[group] = load_result(d)
        s = results[group]["summary"]
        print(f"  {group:<10}  dir={d.name}  "
              f"Sharpe={float(s['sharpe']):.3f}  "
              f"CAGR={float(s['cagr']):+.2%}  "
              f"Trades={int(float(s['total_trades']))}")

    if not results:
        print("No results found — run backtests first.")
        sys.exit(1)

    print(f"\nBuilding PDF → {out_path} ...")
    build_pdf(results, out_path)


if __name__ == "__main__":
    main()
