"""Streamlit dashboard - entry point.   Run:  streamlit run src/app.py"""
import html as _html
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd                # noqa: E402
import plotly.express as px        # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st             # noqa: E402

from src.config import SIGNALS_FILE, STRESS_CATEGORIES, TICKERS, UNIVERSE  # noqa: E402
from src.ingestion import fetch_gdelt, load_all  # noqa: E402
from src.modules import SentimentRebalancer, StressTester, find_trigger_events, load_prices  # noqa: E402
from src.nlp_engine import RiskEngine, load_signals, save_signals  # noqa: E402

st.set_page_config(page_title="NLP Risk Engine", page_icon="📡", layout="wide", initial_sidebar_state="collapsed")

POS, NEG, NEU, AMBER, BLUE = "#2DD4BF", "#F87171", "#94A3B8", "#FBBF24", "#60A5FA"
CAT_COLOR = {"Geopolitical": "#F97316", "Macroeconomic": "#A78BFA", "Credit Event": "#F87171",
             "Merger/Acquisition": "#60A5FA", "Product Launch": "#2DD4BF", "Other": "#94A3B8"}
EXAMPLES = {
    "🏦 Rate shock": "Central bank shocks markets with surprise rate hike as inflation surges",
    "⚔️ Conflict": "Armed conflict escalates as missile strikes hit key shipping route; global markets plunge",
    "🚀 Product launch": "Nvidia unveils next-generation AI platform to enthusiastic reviews and record preorders",
    "📉 Downgrade": "JPMorgan Chase credit rating downgraded as debt concerns mount, default fears spread",
    "🕊️ Ceasefire": "Ceasefire agreement reached, peace talks ease tensions and markets rally strongly",
}

CSS = """
<style>
/* ---- force the dark look even if .streamlit/config.toml is missing ---- */
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {background:#0B1220 !important; color:#E6EDF7;}
header[data-testid="stHeader"] {background:transparent !important;}
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp label,
[data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] p,
[data-testid="stWidgetLabel"] p {color:#E6EDF7;}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {color:#8EA0BC !important;}
textarea, input {background:#111B2E !important; color:#E6EDF7 !important; -webkit-text-fill-color:#E6EDF7 !important;}
[data-baseweb="textarea"], [data-baseweb="base-input"], [data-baseweb="select"] > div {background:#111B2E !important; border-color:#1E2B45 !important; color:#E6EDF7 !important;}
[data-baseweb="select"] * {color:#E6EDF7 !important;}
[data-testid="stExpander"] details {background:#111B2E; border:1px solid #1E2B45 !important; border-radius:12px;}
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary p {color:#E6EDF7 !important;}
[data-testid="stSliderThumbValue"], [data-testid="stTickBarMin"], [data-testid="stTickBarMax"] {color:#E6EDF7 !important;}
[data-testid="stButtonGroup"] button {background:#111B2E !important; color:#E6EDF7 !important; border:1px solid #1E2B45 !important;}
[data-testid="stButtonGroup"] button p {color:inherit !important;}
[data-testid="stButtonGroup"] button[aria-checked="true"], [data-testid="stButtonGroup"] button[aria-pressed="true"] {border-color:#2DD4BF !important; color:#2DD4BF !important;}
.stButton>button, .stButton>button p {color:#E6EDF7;}
[data-testid="stJson"], [data-testid="stJson"] * {background:#0B1220 !important; color:#E6EDF7 !important;}
[data-testid="stDataFrame"] {border:1px solid #1E2B45; border-radius:12px; filter:invert(.92) hue-rotate(180deg);}
#MainMenu, footer, [data-testid="stDecoration"] {visibility:hidden;}
header[data-testid="stHeader"] {background:transparent;}
.block-container {padding-top:1.4rem; max-width:1250px;}
.hero {background:linear-gradient(120deg,#0F766E 0%,#1D4ED8 60%,#7C3AED 100%); border-radius:20px;
       padding:28px 34px; margin-bottom:18px; box-shadow:0 10px 40px rgba(29,78,216,.25);}
.hero .tag {display:inline-block; background:rgba(255,255,255,.16); padding:4px 12px; border-radius:99px;
            font-size:.78rem; letter-spacing:.04em; margin-bottom:10px;}
.hero h1 {margin:0; font-size:2.3rem; line-height:1.15; color:#fff; padding:0;}
.hero p {margin:8px 0 0; font-size:1.05rem; color:#E0E7FF; max-width:780px;}
.kpis {display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin-bottom:18px;}
.kpi {background:#111B2E; border:1px solid #1E2B45; border-radius:16px; padding:16px 18px;}
.kpi .l {font-size:.74rem; color:#8EA0BC; text-transform:uppercase; letter-spacing:.06em;}
.kpi .v {font-size:2rem; font-weight:700; margin-top:2px;}
.kpi .s {font-size:.8rem; color:#8EA0BC;}
.card {background:#111B2E; border:1px solid #1E2B45; border-radius:16px; padding:20px 22px; margin:8px 0;}
.result {display:grid; grid-template-columns:repeat(4,1fr); gap:18px;}
.lbl {font-size:.72rem; color:#8EA0BC; text-transform:uppercase; letter-spacing:.07em; margin-bottom:4px;}
.big {font-size:2.2rem; font-weight:700; line-height:1.1;}
.big.sm {font-size:1.45rem; padding-top:8px;}
.sub {font-size:.8rem; color:#8EA0BC; margin-top:4px;}
.pill {display:inline-block; padding:2px 11px; border-radius:99px; font-size:.78rem; font-weight:600; margin:2px 4px 2px 0;}
.meter {height:9px; background:#1E2B45; border-radius:99px; margin-top:10px; overflow:hidden;}
.meter span {display:block; height:100%; border-radius:99px; background:linear-gradient(90deg,#2DD4BF,#FBBF24,#F87171);}
.verdict {margin-top:16px; padding:12px 16px; border-radius:12px; font-size:.95rem; background:#0B1220; border:1px solid #1E2B45;}
.feed {background:#111B2E; border:1px solid #1E2B45; border-left:4px solid var(--c); border-radius:12px;
       padding:12px 15px; margin-bottom:10px; min-height:112px;}
.feed .top {display:flex; gap:8px; align-items:center; font-size:.74rem; color:#8EA0BC; margin-bottom:6px;}
.feed .txt {font-size:.92rem; color:#E6EDF7; margin-bottom:6px;}
.mover {background:#111B2E; border:1px solid #1E2B45; border-radius:14px; padding:12px 16px; margin-bottom:8px;
        display:flex; justify-content:space-between; align-items:center;}
.mover .t {font-weight:700; font-size:1.05rem;} .mover .n {font-size:.78rem; color:#8EA0BC;}
.shock {display:inline-block; background:#0B1220; border:1px solid #1E2B45; border-radius:10px;
        padding:6px 12px; margin:3px 6px 3px 0; font-size:.85rem;}
[role="tablist"] {gap:6px; background:#111B2E; padding:6px !important; border-radius:14px; border:1px solid #1E2B45;}
[data-testid="stTab"] {border-radius:10px; padding:10px 24px !important; height:auto; color:#9FB0CC; white-space:nowrap; margin:0 !important;}
[data-testid="stTab"][aria-selected="true"] {background:linear-gradient(90deg,#14B8A6,#3B82F6); color:#fff !important;}
[data-testid="stTab"][aria-selected="true"] p {color:#fff !important;}
[role="tablist"] > div:not([data-testid="stTab"]) {display:none;}
.stButton>button {border-radius:12px; border:1px solid #1E2B45; background:#111B2E;}
.stButton>button:hover {border-color:#2DD4BF; color:#2DD4BF;}
.stButton>button[kind="primary"], .stButton>button[kind="primary"] p {background:linear-gradient(90deg,#14B8A6,#3B82F6); border:0; color:#fff !important;}
textarea {border-radius:12px !important;}
h2, h3 {letter-spacing:-.01em;}
@media (max-width:900px) {.kpis,.result {grid-template-columns:repeat(2,1fr);}}
</style>
"""


def H(s: str):
    """Render HTML (leading whitespace stripped so markdown never treats it as a code block)."""
    st.markdown(" ".join(line.strip() for line in s.splitlines()), unsafe_allow_html=True)


def esc(x) -> str:
    return _html.escape(str(x))


def style(fig, h=330, legend=True):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=8, r=8, t=44, b=8), height=h, font=dict(color="#CBD5E1"), title_font_size=15,
                      legend=dict(orientation="h", y=-0.18) if legend else dict(visible=False))
    fig.update_xaxes(gridcolor="#1E2B45", zeroline=False)
    fig.update_yaxes(gridcolor="#1E2B45", zeroline=False)
    return fig


def mood(s: float):
    return ("Positive", POS) if s > 0.15 else ("Negative", NEG) if s < -0.15 else ("Neutral", NEU)


def severity(i: float) -> str:
    return "Minor" if i < 3 else "Moderate" if i < 5 else "Significant" if i < 7 else "High" if i < 9 else "Systemic"


def money(x: float) -> str:
    return f"-${abs(x) / 1e6:,.0f}M" if x < 0 else f"${x / 1e6:,.0f}M"


def chips(tickers) -> str:
    return "".join(f'<span class="pill" style="background:#1E2B45;color:#E6EDF7">{esc(t)}</span>' for t in tickers)


@st.cache_resource
def get_engine():
    return RiskEngine()


@st.cache_data(show_spinner="Running NLP pipeline...")
def get_signals(force: bool = False):
    if force or not SIGNALS_FILE.exists():
        save_signals(get_engine().analyze_many(load_all()))
    return load_signals()


@st.cache_data
def get_prices(live: bool):
    return load_prices(live=live, start="2026-09-10", end="2026-10-02")


def set_headline(text: str):
    st.session_state["headline"] = text


# ================================================================== page
H(CSS)
signals = get_signals()
sig_df = pd.DataFrame(signals)
sig_df["date"] = pd.to_datetime(sig_df["timestamp"]).dt.date
events = find_trigger_events(signals)

H("""<div class="hero"><div class="tag">S&amp;P Global × Crisil Campus Hackathon 2026</div>
<h1>NLP Risk Engine</h1>
<p>Reads news and social posts, turns them into risk signals, then acts on them: it re-weights a stock index and stress-tests a bank's portfolio.</p></div>""")

avg = sig_df["sentiment_score"].mean()
mlabel, mcol = mood(avg)
H(f"""<div class="kpis">
<div class="kpi"><div class="l">Texts analysed</div><div class="v">{len(sig_df)}</div>
<div class="s">{(sig_df['source_type'] == 'news').sum()} news · {(sig_df['source_type'] == 'social').sum()} social</div></div>
<div class="kpi"><div class="l">Overall mood</div><div class="v" style="color:{mcol}">{avg:+.2f}</div><div class="s">{mlabel}</div></div>
<div class="kpi"><div class="l">High-impact alerts</div><div class="v" style="color:{AMBER}">{int((sig_df['impact_score'] > 7).sum())}</div>
<div class="s">impact above 7 / 10</div></div>
<div class="kpi"><div class="l">Stress tests triggered</div><div class="v" style="color:{NEG}">{len(events)}</div>
<div class="s">Module B</div></div></div>""")

tab_a, tab_b, tab_c, tab_d = st.tabs(["🔎 Analyze text", "📈 Rebalance index", "🏦 Stress-test bank", "ℹ️ How it works"])

# ================================================================== 1. analyze
with tab_a:
    st.subheader("Try the engine")
    st.caption("Pick an example or paste any headline / tweet, then press Analyze.")
    cols = st.columns(len(EXAMPLES))
    for col, (label, text) in zip(cols, EXAMPLES.items()):
        col.button(label, key=f"ex_{label}", on_click=set_headline, args=(text,), width="stretch")
    st.session_state.setdefault("headline", list(EXAMPLES.values())[0])
    st.text_area("Headline or tweet", key="headline", height=90, label_visibility="collapsed")
    c1, c2 = st.columns([1, 5])
    c1.button("Analyze ▶", type="primary", width="stretch")
    src_choice = c2.segmented_control("Source", ["news", "social"], default="news",
                                      format_func=lambda s: "📰 News" if s == "news" else "💬 Social post",
                                      label_visibility="collapsed")
    text = st.session_state["headline"].strip()
    if text:
        r = get_engine().analyze(text, source_type=src_choice or "news")
        lab, col = mood(r["sentiment_score"])
        imp = r["impact_score"]
        icol = NEG if imp > 7 else AMBER if imp > 4.5 else POS
        cat = r["event_category"]
        if imp > 7 and cat in STRESS_CATEGORIES and r["sentiment_score"] < 0:
            verdict = "⚠️ <b>Triggers a stress test</b> in Module B (impact above 7, adverse, Geopolitical/Macroeconomic)."
        elif imp > 7 and cat in STRESS_CATEGORIES:
            verdict = "High impact but <b>positive</b>, so no stress test. Only adverse events trigger one."
        elif r["tickers"] and r["sentiment_score"] > 0.15:
            verdict = f"▲ Module A would <b>raise</b> the weight of {', '.join(r['tickers'])}."
        elif r["tickers"] and r["sentiment_score"] < -0.15:
            verdict = f"▼ Module A would <b>lower</b> the weight of {', '.join(r['tickers'])}."
        else:
            verdict = "No portfolio action: below the trigger thresholds."
        who = chips(r["tickers"]) if r["tickers"] else '<span class="pill" style="background:#1E2B45">Market-wide</span>'
        H(f"""<div class="card"><div class="result">
<div><div class="lbl">Sentiment</div><div class="big" style="color:{col}">{r['sentiment_score']:+.2f}</div>
<span class="pill" style="background:{col}22;color:{col}">{lab}</span></div>
<div><div class="lbl">Event type</div><div class="big sm" style="color:{CAT_COLOR.get(cat, NEU)}">{esc(cat)}</div>
<div class="sub">{r['category_confidence'] * 100:.0f}% rule confidence</div></div>
<div><div class="lbl">Impact</div><div class="big" style="color:{icol}">{imp:.1f}<span style="font-size:1rem;color:#8EA0BC"> / 10</span></div>
<div class="meter"><span style="width:{imp * 10}%"></span></div><div class="sub">{severity(imp)}</div></div>
<div><div class="lbl">Affects</div><div style="padding-top:8px">{who}</div></div></div>
<div class="verdict">{verdict}</div></div>""")
        with st.expander("Why this score? (explainability + raw JSON)"):
            pills = "".join(f'<span class="pill" style="background:#1E2B45;color:#E6EDF7">{esc(d)}</span>' for d in r["drivers"])
            H(pills)
            st.json(r, expanded=False)

    st.subheader("Signal feed")
    flt = st.segmented_control("Filter", ["All", "News", "Social"], default="All", label_visibility="collapsed")
    view = sig_df if flt in (None, "All") else sig_df[sig_df["source_type"] == flt.lower()]
    top = view.sort_values(["impact_score", "timestamp"], ascending=False).drop_duplicates("text").head(6)
    fc = st.columns(2)
    for i, (_, s) in enumerate(top.iterrows()):
        lab, col = mood(s["sentiment_score"])
        with fc[i % 2]:
            H(f"""<div class="feed" style="--c:{CAT_COLOR.get(s['event_category'], NEU)}">
<div class="top"><span class="pill" style="background:{CAT_COLOR.get(s['event_category'], NEU)}33;color:{CAT_COLOR.get(s['event_category'], NEU)}">{esc(s['event_category'])}</span>
<span>{'📰' if s['source_type'] == 'news' else '💬'} {esc(s['source_type'])}</span><span style="margin-left:auto">impact <b>{s['impact_score']:.1f}</b></span></div>
<div class="txt">{esc(s['text'])}</div>
<div><span style="color:{col};font-weight:700">{s['sentiment_score']:+.2f}</span> {chips(s['tickers'])}</div></div>""")
    with st.expander("See all signals (table)"):
        st.dataframe(view[["timestamp", "source_type", "text", "tickers", "sentiment_score", "event_category", "impact_score"]]
                     .sort_values("timestamp", ascending=False), width="stretch", height=300)

    g1, g2 = st.columns(2)
    cc = sig_df["event_category"].value_counts().reset_index()
    cc.columns = ["category", "count"]
    fig = px.pie(cc, names="category", values="count", hole=0.62, color="category", color_discrete_map=CAT_COLOR,
                 title="What the text is about")
    fig.update_traces(textinfo="none")
    g1.plotly_chart(style(fig), width="stretch", theme=None)
    daily = sig_df.groupby("date")["sentiment_score"].mean().reset_index()
    fig = px.area(daily, x="date", y="sentiment_score", title="Average mood per day", markers=True)
    fig.update_traces(line_color=BLUE, fillcolor="rgba(96,165,250,.18)")
    g2.plotly_chart(style(fig, legend=False), width="stretch", theme=None)

    with st.expander("Optional: live headlines from GDELT (needs internet)"):
        if st.button("Fetch & analyse live headlines"):
            try:
                live = get_engine().analyze_many(fetch_gdelt(max_records=20))
                st.dataframe(pd.DataFrame(live)[["timestamp", "text", "sentiment_score", "event_category", "impact_score"]])
            except Exception as e:
                st.warning(f"Live fetch failed ({e}). The bundled offline data still works.")

# ================================================================== 2. rebalancer
with tab_b:
    st.subheader("Index that reacts to sentiment")
    st.caption("15 S&P 100 stocks start at equal weight. Good news lifts a stock's weight, bad news lowers it.")
    k = st.slider("Reaction strength", 0.0, 1.0, 0.6, 0.05, help="0 = ignore sentiment (equal weights), 1 = react strongly")
    with st.expander("Advanced settings"):
        a1, a2, a3 = st.columns(3)
        decay = a1.slider("Sentiment memory", 0.0, 0.95, 0.6, 0.05, help="How long old news keeps influencing weights")
        cap = a2.slider("Max weight vs equal (×)", 1.2, 3.0, 2.0, 0.1)
        live_px = a3.checkbox("Use live prices (yfinance)", value=False)
    prices = get_prices(live_px)
    res = SentimentRebalancer(sensitivity=k, decay=decay, cap=cap).run(signals, prices)
    W, S = res["weights"], res["sentiment"]
    dates = list(W.index)
    day = st.select_slider("Pick a day", options=dates, value=dates[-1], format_func=lambda d: d.strftime("%b %d"))

    wl = (W * 100).reset_index().melt(id_vars="date", var_name="Stock", value_name="Weight %")
    fig = px.area(wl, x="date", y="Weight %", color="Stock", title="How the weights evolve",
                  color_discrete_sequence=px.colors.qualitative.Light24)
    fig.add_shape(type="line", x0=day.strftime("%Y-%m-%d"), x1=day.strftime("%Y-%m-%d"), y0=0, y1=1, yref="paper",
                  line=dict(color="white", dash="dot", width=2))
    fig.update_xaxes(title_text=None)
    fig.update_layout(legend_title_text="")
    st.plotly_chart(style(fig, 360), width="stretch", theme=None)

    base = 100 / len(TICKERS)
    wd = (W.loc[day] * 100).sort_values()
    left, right = st.columns([3, 2])
    fig = go.Figure(go.Bar(x=wd.values, y=wd.index, orientation="h",
                           marker_color=[POS if v >= base else NEG for v in wd.values],
                           text=[f"{v:.1f}%" for v in wd.values], textposition="outside"))
    fig.add_vline(x=base, line_dash="dot", line_color="#94A3B8", annotation_text="equal weight")
    fig.update_layout(title=f"Allocation on {day.strftime('%b %d')}")
    left.plotly_chart(style(fig, 470, legend=False), width="stretch", theme=None)
    with right:
        delta = (W.loc[day] * 100 - base).sort_values()
        st.markdown("**Biggest increases**")
        for t in delta.tail(3).index[::-1]:
            H(f"""<div class="mover"><div><div class="t">{t}</div><div class="n">{UNIVERSE[t]['name']} · {UNIVERSE[t]['sector']}</div></div>
<div style="text-align:right"><div style="color:{POS};font-weight:700">▲ {delta[t]:+.1f} pp</div>
<div class="n">sentiment {S.loc[day, t]:+.2f}</div></div></div>""")
        st.markdown("**Biggest cuts**")
        for t in delta.head(3).index:
            H(f"""<div class="mover"><div><div class="t">{t}</div><div class="n">{UNIVERSE[t]['name']} · {UNIVERSE[t]['sector']}</div></div>
<div style="text-align:right"><div style="color:{NEG};font-weight:700">▼ {delta[t]:+.1f} pp</div>
<div class="n">sentiment {S.loc[day, t]:+.2f}</div></div></div>""")
    with st.expander("Performance vs equal weight (illustrative)"):
        st.plotly_chart(style(px.line(res["performance"], title="Index level, base 100"), 300), width="stretch", theme=None)
        st.caption("Demo prices are synthetic and independent of the news, so this is illustrative, not evidence of "
                   "alpha. Average daily turnover: " f"{res['turnover'].mean() * 100:.1f}%. "
                   "Weights set at close of day t earn day t+1 returns (no look-ahead).")

# ================================================================== 3. stress test
with tab_c:
    st.subheader("What would this event do to a bank's book?")
    st.caption("A stress test runs automatically for events with impact above 7 that are adverse and Geopolitical or Macroeconomic.")
    if not events:
        st.info("No triggering events in the current data.")
    else:
        labels = {e["event_id"]: f"{pd.Timestamp(e['date']).strftime('%b %d')} · {e['category']} · impact {e['impact_score']:.1f}"
                  for e in events}
        best = max(range(len(events)), key=lambda i: (events[i]["corroborating_signals"], events[i]["impact_score"]))
        ev_id = st.selectbox("Triggered event", list(labels), index=best, format_func=labels.get)
        ev = next(e for e in events if e["event_id"] == ev_id)
        tester = StressTester()
        base_sh = tester.shocks_for(ev["category"], ev["impact_score"])
        with st.expander("Customize shocks (analyst override)"):
            e1, e2, e3, e4 = st.columns(4)
            sh = dict(base_sh)
            sh["equity"] = e1.slider("Equity shock", -0.4, 0.0, float(base_sh["equity"]), 0.01)
            sh["rate_bp"] = e2.slider("Rates (bp)", -300.0, 300.0, float(base_sh["rate_bp"]), 10.0)
            sh["spread_bp"] = e3.slider("Credit spread (bp)", 0.0, 500.0, float(base_sh["spread_bp"]), 10.0)
            sh["pd_mult"] = e4.slider("Default-risk multiplier", 1.0, 4.0, float(base_sh["pd_mult"]), 0.1)
        r = tester.run(ev["category"], ev["impact_score"], shocks=sh)

        H(f"""<div class="card"><div class="lbl">Triggering event · {ev['corroborating_signals']} corroborating signal(s) from {esc(' + '.join(ev['sources']))}</div>
<div style="font-size:1.05rem;margin:4px 0 10px">{esc(ev['headline'])}</div>
<span class="shock">📉 Equities {sh['equity'] * 100:+.1f}%</span><span class="shock">🏦 Rates {sh['rate_bp']:+.0f} bp</span>
<span class="shock">💳 Credit spreads {sh['spread_bp']:+.0f} bp</span><span class="shock">⚠️ Default risk ×{sh['pd_mult']:.2f}</span></div>""")
        H(f"""<div class="kpis" style="grid-template-columns:repeat(3,1fr)">
<div class="kpi"><div class="l">Portfolio before</div><div class="v" style="color:{BLUE}">{money(r['pre_total'])}</div></div>
<div class="kpi"><div class="l">Portfolio after</div><div class="v">{money(r['post_total'])}</div></div>
<div class="kpi"><div class="l">Loss</div><div class="v" style="color:{NEG}">{money(r['delta'])}</div>
<div class="s">{r['delta_pct']:.1f}% of portfolio value</div></div></div>""")

        by = r["by_asset_class"]
        g1, g2 = st.columns(2)
        fig = go.Figure([go.Bar(name="Before", x=by["asset_class"], y=by["pre"] / 1e6, marker_color=BLUE),
                         go.Bar(name="After", x=by["asset_class"], y=by["post"] / 1e6, marker_color=NEG)])
        fig.update_layout(barmode="group", title="Value by asset class ($M)")
        g1.plotly_chart(style(fig), width="stretch", theme=None)
        d = by.sort_values("delta")
        fig = go.Figure(go.Bar(x=d["delta"] / 1e6, y=d["asset_class"], orientation="h",
                               marker_color=[NEG if v < 0 else POS for v in d["delta"]],
                               text=[f"{v / 1e6:+.1f}" for v in d["delta"]], textposition="outside"))
        fig.update_traces(cliponaxis=False)
        fig.update_xaxes(range=[min(d["delta"].min() / 1e6 * 1.35, -1), max(d["delta"].max() / 1e6 * 1.35, 1)])
        fig.update_layout(title="Where the loss comes from ($M)")
        g2.plotly_chart(style(fig, legend=False), width="stretch", theme=None)
        with st.expander("Biggest position-level losses"):
            pos = r["positions"].sort_values("delta").head(8)
            st.dataframe(pos.assign(pre=pos["pre"].map(money), post=pos["post"].map(money), delta=pos["delta"].map(money)),
                         width="stretch", hide_index=True)

# ================================================================== 4. how it works
with tab_d:
    arch = Path(__file__).resolve().parent.parent / "docs" / "architecture.png"
    if arch.exists():
        st.image(str(arch))
    h1, h2, h3 = st.columns(3)
    with h1:
        H("""<div class="card"><div class="lbl">Step 1</div><div class="big sm" style="padding-top:2px">Read</div>
<div class="sub" style="font-size:.9rem">Two sources (news + social posts), optionally live from NewsAPI / GDELT, are
normalised into one format.</div></div>""")
    with h2:
        H("""<div class="card"><div class="lbl">Step 2</div><div class="big sm" style="padding-top:2px">Score</div>
<div class="sub" style="font-size:.9rem">Every text gets a sentiment (-1 to +1), an event type and an impact score
(1 to 10), each with an explanation.</div></div>""")
    with h3:
        H("""<div class="card"><div class="lbl">Step 3</div><div class="big sm" style="padding-top:2px">Act</div>
<div class="sub" style="font-size:.9rem">Module A re-weights 15 stocks. Module B stress-tests a synthetic bank book
when a severe adverse event appears.</div></div>""")
    st.caption(f"Engine: `{get_engine().model_name}` · All bundled data is synthetic · See the README for assumptions and limits.")
    if st.button("Re-run the NLP pipeline on bundled data"):
        get_signals.clear()
        get_signals(force=True)
        st.rerun()