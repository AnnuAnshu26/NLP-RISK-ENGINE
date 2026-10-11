"""Regenerate docs/architecture.png, docs/img/*.png and docs/presentation.pdf from REAL outputs.

python scripts/make_docs.py --name "Your Name" --college "Your College" --email you@college.ac.in
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.styles import ParagraphStyle  # noqa: E402
from reportlab.pdfgen import canvas  # noqa: E402
from reportlab.platypus import Paragraph, Table, TableStyle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import evaluate  # noqa: E402
from src.config import DATA_DIR, DOCS_DIR, TICKERS  # noqa: E402
from src.ingestion import load_all, load_news  # noqa: E402
from src.modules import SentimentRebalancer, StressTester, find_trigger_events, load_prices  # noqa: E402
from src.nlp_engine import RiskEngine  # noqa: E402

IMG = DOCS_DIR / "img"
NAVY, TEAL, RED, GREY = "#14213D", "#2A9D8F", "#E4572E", "#F2F4F8"


def box(ax, x, y, w, h, title, lines, fc="#FFFFFF", ec=NAVY):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15", fc=fc, ec=ec, lw=2))
    ax.text(x + w / 2, y + h - 0.35, title, ha="center", va="top", fontsize=13, fontweight="bold", color=NAVY)
    for i, l in enumerate(lines):
        ax.text(x + 0.2, y + h - 0.95 - i * 0.48, "• " + l, fontsize=10, va="top", color="#222")


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=22, lw=2.2, color=NAVY))


def architecture():
    fig, ax = plt.subplots(figsize=(16, 9), dpi=150)
    ax.set_xlim(0, 16); ax.set_ylim(0, 9); ax.axis("off")
    ax.text(8, 8.6, "NLP Risk Engine - System Architecture", ha="center", fontsize=20, fontweight="bold", color=NAVY)
    box(ax, 0.3, 4.6, 3.4, 3.2, "1. Ingestion", ["News JSON (NewsAPI/GDELT-like)", "Social CSV (Kaggle tweet-like)",
        "Optional live: NewsAPI, GDELT", "Normalised record schema"], fc="#E8F1FA")
    box(ax, 4.5, 4.0, 4.6, 3.8, "2. AI/NLP Risk Engine", ["Sentiment: FinBERT | lexicon  [-1,1]",
        "Event class: rules + zero-shot (5 cats)", "Impact score 1-10 (explainable drivers)",
        "Ticker linking (S&P 100 universe)", "Source-reliability weighting"], fc="#E6F6F3", ec=TEAL)
    box(ax, 9.9, 4.6, 3.0, 3.2, "3. Signal Store / API", ["data/signals.json", "FastAPI /signals /analyze", "/rebalance /stress/*"],
        fc="#FFF4E0")
    box(ax, 4.6, 0.3, 4.2, 2.9, "Module A: Rebalancer", ["15 S&P 100 stocks (yfinance)", "w = base x (1 + k x sentiment)",
        "decay, floor/cap, defensive tilt"], fc="#F3E8FA")
    box(ax, 9.6, 0.3, 4.4, 2.9, "Module B: Stress Tester", ["Trigger: impact>7, Geo/Macro, adverse",
        "Loans, bonds, derivatives, equities", "Pre vs post stress valuation"], fc="#FDE8E4", ec=RED)
    box(ax, 13.3, 4.6, 2.5, 3.2, "4. Dashboard", ["Streamlit", "Signals", "Weights over time", "Pre/post stress"], fc="#EEF0F4")
    arrow(ax, 3.7, 6.2, 4.5, 6.2); arrow(ax, 9.1, 6.2, 9.9, 6.2); arrow(ax, 12.9, 6.2, 13.3, 6.2)
    arrow(ax, 11.4, 4.6, 11.4, 3.2); arrow(ax, 10.3, 4.6, 7.6, 3.2)
    ax.text(8, 3.55, "structured signals", fontsize=9, color="#555", ha="center", style="italic")
    fig.savefig(DOCS_DIR / "architecture.png", bbox_inches="tight", facecolor="white"); plt.close(fig)


def charts(signals):
    prices = load_prices()
    res = SentimentRebalancer().run(signals, prices)
    W = res["weights"] * 100
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=160)
    W.plot.area(ax=ax, stacked=True, linewidth=0, colormap="tab20", legend=False)
    ax.set_title("Module A - index weights over time (%)", fontsize=11); ax.set_ylabel("%"); ax.set_xlabel("")
    ax.legend(ncol=5, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)
    fig.tight_layout(); fig.savefig(IMG / "weights.png"); plt.close(fig)

    ev = find_trigger_events(signals)
    st = StressTester()
    top = max(ev, key=lambda e: -st.run_event(e)["delta"])
    r = st.run_event(top)
    by = r["by_asset_class"]
    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=160)
    x = range(len(by)); wdt = 0.38
    ax.bar([i - wdt / 2 for i in x], by["pre"] / 1e6, wdt, label="Pre-stress", color="#4C78A8")
    ax.bar([i + wdt / 2 for i in x], by["post"] / 1e6, wdt, label="Post-stress", color=RED)
    ax.set_xticks(list(x)); ax.set_xticklabels(by["asset_class"], fontsize=8); ax.set_ylabel("$M"); ax.legend(fontsize=8)
    ax.set_title(f"Module B - {top['event_id']}\n{r['delta_pct']:.1f}% portfolio value", fontsize=10)
    fig.tight_layout(); fig.savefig(IMG / "stress.png"); plt.close(fig)
    return res, ev, r, top


def build_pdf(args, stats):
    W_, H_ = 1280, 720
    c = canvas.Canvas(str(DOCS_DIR / "presentation.pdf"), pagesize=(W_, H_))
    c.setTitle("NLP Risk Engine - S&P Global & Crisil Campus Hackathon 2026")
    body = ParagraphStyle("b", fontName="Helvetica", fontSize=27, leading=37, textColor=colors.HexColor("#222222"))
    small = ParagraphStyle("s", parent=body, fontSize=17, leading=24, textColor=colors.HexColor("#555555"))

    def frame(n, title):
        c.setFillColor(colors.HexColor(GREY)); c.rect(0, 0, W_, H_, fill=1, stroke=0)
        c.setFillColor(colors.HexColor(NAVY)); c.rect(0, H_ - 100, W_, 100, fill=1, stroke=0)
        c.setFillColor(colors.white); c.setFont("Helvetica-Bold", 38); c.drawString(50, H_ - 66, title)
        c.setFont("Helvetica", 14); c.setFillColor(colors.HexColor("#555555"))
        c.drawString(50, 22, f"{args.name} | {args.college} | S&P Global & Crisil Campus Hackathon 2026")
        c.drawRightString(W_ - 50, 22, f"{n}/7")

    def bullets(items, x, y, w, style=body):
        for it in items:
            p = Paragraph("&bull;&nbsp; " + it, style)
            _, h = p.wrap(w, 400); p.drawOn(c, x, y - h); y -= h + 14
        return y

    # 1 title
    c.setFillColor(colors.HexColor(NAVY)); c.rect(0, 0, W_, H_, fill=1, stroke=0)
    c.setFillColor(colors.HexColor(TEAL)); c.rect(0, 300, 14, 190, fill=1, stroke=0)
    c.setFillColor(colors.white); c.setFont("Helvetica-Bold", 56); c.drawString(70, 440, "NLP Risk Engine")
    c.setFont("Helvetica", 28); c.drawString(70, 390, "From unstructured news & social text to actionable financial risk signals")
    c.setFont("Helvetica", 22)
    for i, l in enumerate([f"Candidate: {args.name}", f"College: {args.college}", f"Email: {args.email}"]):
        c.drawString(70, 320 - i * 36, l)
    c.setFont("Helvetica", 16); c.drawString(70, 60, "S&P Global & Crisil Campus Hackathon 2026  |  Modules A + B")
    c.showPage()

    # 2 problem
    frame(2, "Problem & Approach")
    bullets(["<b>Problem:</b> Risk-moving information arrives as unstructured text (headlines, tweets). Humans cannot read, "
             "score and act on it at market speed.",
             "<b>Approach:</b> one NLP engine turns every text into a machine-readable signal: sentiment [-1,1], event category "
             "(5 types) and impact score [1-10], linked to S&amp;P 100 tickers.",
             "<b>Two consumers of the same signals:</b> Module A (tactical rebalancing of 15 stocks) and Module B (strategic stress "
             "test of a synthetic wholesale-banking book).",
             "<b>Design principles:</b> explainable, runs offline in seconds, swaps in FinBERT for production-grade accuracy."],
            60, H_ - 140, W_ - 120)
    c.showPage()

    # 3 system
    frame(3, "System Design")
    c.drawImage(str(DOCS_DIR / "architecture.png"), 90, 40, width=1100, height=590,
                preserveAspectRatio=True, anchor="c")
    c.showPage()

    # 4 implementation
    frame(4, "Implementation Highlights")
    data = [["Module", "Tech choice", "Why"],
            ["Ingestion", "JSON + CSV loaders; NewsAPI / GDELT live fetchers", "2 sources, one schema; works offline"],
            ["Sentiment", "Finance lexicon (default) | FinBERT (opt-in)", "Instant & dependency-free; FinBERT for accuracy"],
            ["Event class", "Weighted rules + BART zero-shot fallback", "Explainable, 5 required categories"],
            ["Impact 1-10", "Additive model: category, severity, |sentiment|, scope", "Every score ships with its drivers"],
            ["Module A", "w = base x (1 + k x s), decay, floor/cap", "Bounded, smooth, no look-ahead"],
            ["Module B", "Duration/convexity, PD/LGD, DV01, delta, CS01", "Simple, auditable shock model"],
            ["Serving", "FastAPI (async) + Streamlit + pytest (13 tests)", "REST signals + interactive dashboard"]]
    t = Table([[Paragraph(str(x), ParagraphStyle("t", fontName="Helvetica-Bold" if i == 0 else "Helvetica", fontSize=19,
                                                 leading=24, textColor=colors.white if i == 0 else colors.black))
                for x in row] for i, row in enumerate(data)], colWidths=[200, 560, 440])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(NAVY)),
                           ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#E8EEF5")]),
                           ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 9),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 9), ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#BBBBBB"))]))
    _, th = t.wrap(1180, 600); t.drawOn(c, 50, H_ - 130 - th)
    c.showPage()

    # 5 results
    frame(5, "Key Results")
    c.drawImage(str(IMG / "weights.png"), 40, 255, width=700, height=360, preserveAspectRatio=True)
    c.drawImage(str(IMG / "stress.png"), 760, 255, width=470, height=330, preserveAspectRatio=True)
    s = stats
    bullets([f"<b>{s['n']} texts</b> (news + social) scored in <b>{s['secs']:.2f}s</b> on CPU; <b>{s['n_events']}</b> stress events "
             f"triggered; worst case <b>{s['worst_pct']:.1f}%</b> of book value.",
             f"Held-out hand-written set (n=30): baseline <b>{s['base_sent']:.0%}</b> sentiment / <b>{s['base_cat']:.0%}</b> category; "
             f"after lexicon expansion <b>{s['dev_sent']:.0%}</b> / <b>{s['dev_cat']:.0%}</b> (set then used for tuning = dev set). "
             f"Template data scores 100% (not meaningful).",
             "Demo prices are synthetic, so index-vs-equal-weight is illustrative only; weight mechanics verified by unit tests."],
            50, 235, 1180, small)
    c.showPage()

    # 6 impact
    frame(6, "Domain Impact")
    bullets(["<b>Asset managers:</b> turns news flow into bounded, auditable weight tilts (floor/cap, turnover metrics) instead of "
             "ad-hoc discretionary reactions.",
             "<b>Risk officers / banks:</b> geopolitical or macro shocks automatically trigger a stress run of the book, "
             "with pre/post values by asset class, in seconds rather than a next-day batch.",
             "<b>Efficiency vs manual tracking:</b> corroborating news + tweets are clustered into a single event "
             "(no alert fatigue); every score carries explainable drivers for audit.",
             "<b>Extensible:</b> same signal API can feed limits monitoring, credit watchlists or surveillance."],
            60, H_ - 140, W_ - 120)
    c.showPage()

    # 7 limits
    frame(7, "Limitations & Next Steps")
    bullets(["<b>Synthetic data:</b> bundled news, tweets, prices and portfolio are synthetic; no real or client data used.",
             "<b>Default NLP is a lexicon</b>; FinBERT/zero-shot are wired in but opt-in. Impact score is heuristic, not calibrated "
             "against realised market moves.",
             "<b>Stress model is simplified</b> (parallel shifts, no correlations, no counterparty/liquidity effects).",
             "<b>Next:</b> fine-tune FinBERT on labelled finance text, calibrate impact on event-study returns, streaming ingestion "
             "(Kafka), intraday rebalancing with transaction costs, multi-asset & reverse stress tests."],
            60, H_ - 140, W_ - 120)
    c.showPage()
    c.save()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="[Your Full Name]")
    ap.add_argument("--college", default="[Your College]")
    ap.add_argument("--email", default="[your_id@college.ac.in]")
    args = ap.parse_args()
    IMG.mkdir(parents=True, exist_ok=True)

    recs = load_all()
    t0 = time.time(); eng = RiskEngine(); sigs = eng.analyze_many(recs); secs = time.time() - t0
    architecture()
    res, ev, r, top = charts(sigs)
    held = evaluate(load_news(DATA_DIR / "eval_heldout.json"))
    stats = dict(n=len(recs), secs=secs, n_events=len(ev), worst_pct=r["delta_pct"],
                 dev_sent=held["sentiment_accuracy"], dev_cat=held["category_accuracy"],
                 base_sent=0.567, base_cat=0.767)
    build_pdf(args, stats)
    print("docs built:", stats)


if __name__ == "__main__":
    main()
