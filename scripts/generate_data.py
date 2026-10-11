"""Generate the fully SYNTHETIC datasets used by the prototype.

Outputs (all in /data):
  sample_news.json     - synthetic financial news headlines (source 1)
  sample_tweets.csv    - synthetic social-media posts      (source 2)
  prices_sample.csv    - synthetic daily close prices (offline fallback for yfinance)
  portfolio_mock.json  - synthetic wholesale-banking portfolio for Module B

Nothing here is real, proprietary, or derived from S&P Global / Crisil data.
Run:  python scripts/generate_data.py
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import DATA_DIR, UNIVERSE, TICKERS  # noqa: E402

rng = random.Random(42)
nprng = np.random.default_rng(42)

DAYS = pd.bdate_range("2026-09-14", "2026-09-29")  # 12 trading days

# (text template, gold_sentiment, gold_category)
CO_NEWS = [
    ("{c} unveils new product line, analysts applaud strong early demand", "positive", "Product Launch"),
    ("{c} launches innovative new service and expects record adoption", "positive", "Product Launch"),
    ("{c} introduces next-generation platform to enthusiastic reviews", "positive", "Product Launch"),
    ("{c} delays product launch after serious defects found in testing", "negative", "Product Launch"),
    ("{c} to acquire rival in deal expected to boost earnings and synergies", "positive", "Merger/Acquisition"),
    ("{c} announces merger with industry peer, shares jump on growth outlook", "positive", "Merger/Acquisition"),
    ("{c} abandons takeover after regulators block the deal, stock slides", "negative", "Merger/Acquisition"),
    ("{c} credit rating downgraded as debt concerns mount", "negative", "Credit Event"),
    ("{c} faces default risk on maturing bonds, creditors alarmed", "negative", "Credit Event"),
    ("{c} secures credit rating upgrade after strong debt reduction", "positive", "Credit Event"),
    ("{c} beats earnings expectations and raises full-year guidance", "positive", "Other"),
    ("{c} misses earnings and cuts guidance amid weak demand", "negative", "Other"),
    ("{c} reports record profit and announces share buyback", "positive", "Other"),
    ("{c} warns of heavy losses as costs surge", "negative", "Other"),
    ("Sanctions and export curbs threaten {c} supply chain, shares slide", "negative", "Geopolitical"),
    ("{c} schedules annual shareholder meeting for next month", "neutral", "Other"),
    ("{c} to present at industry conference next week", "neutral", "Other"),
]

CO_TWEETS = [
    ("${t} looking strong after the launch, bullish into next week", "positive", "Product Launch"),
    ("${t} new product is a huge hit, demand looks amazing", "positive", "Product Launch"),
    ("${t} launch delayed again, defects found. not good", "negative", "Product Launch"),
    ("heard {c} is buying a rival, big merger news, bullish ${t}", "positive", "Merger/Acquisition"),
    ("${t} merger blocked by regulators, takeover collapse. ouch", "negative", "Merger/Acquisition"),
    ("${t} credit downgrade, debt worries growing, selling", "negative", "Credit Event"),
    ("${t} default fears rising on its bonds, avoid", "negative", "Credit Event"),
    ("${t} earnings beat and guidance raised, great quarter", "positive", "Other"),
    ("${t} earnings miss, weak demand, guidance cut. terrible", "negative", "Other"),
    ("${t} losses mounting, bearish, dumping my shares", "negative", "Other"),
    ("sanctions hitting ${t} supply chain, shares sliding", "negative", "Geopolitical"),
    ("watching ${t} today, nothing much happening", "neutral", "Other"),
]

MARKET_NEWS = [
    ("Armed conflict escalates as missile strikes hit key shipping route; global markets plunge", "negative", "Geopolitical"),
    ("Sweeping new sanctions and trade war escalation shake global markets, investors panic", "negative", "Geopolitical"),
    ("Military tensions spike, oil supply threatened as investors flee to safety", "negative", "Geopolitical"),
    ("Central bank shocks markets with surprise rate hike as inflation surges", "negative", "Macroeconomic"),
    ("Recession fears grow as unemployment jumps and GDP contracts sharply", "negative", "Macroeconomic"),
    ("Bond yields spike, banking turmoil fears grow across the sector", "negative", "Macroeconomic"),
    ("Ceasefire agreement reached, peace talks ease tensions and markets rally", "positive", "Geopolitical"),
    ("Central bank signals rate cut as inflation cools, stocks rally", "positive", "Macroeconomic"),
    ("Strong jobs report and robust GDP growth lift market confidence", "positive", "Macroeconomic"),
    ("Officials to publish quarterly inflation data on Thursday", "neutral", "Macroeconomic"),
]

MARKET_TWEETS = [
    ("war escalating, missiles hitting shipping lanes, markets crashing #markets", "negative", "Geopolitical"),
    ("new sanctions + trade war, global markets in panic, sell everything", "negative", "Geopolitical"),
    ("surprise rate hike!! inflation surging, recession fears, stocks tumble", "negative", "Macroeconomic"),
    ("recession is here, unemployment jumping, bearish on everything", "negative", "Macroeconomic"),
    ("ceasefire reached! peace talks, markets rally, bullish", "positive", "Geopolitical"),
    ("rate cut signal, inflation cooling, stocks surge today", "positive", "Macroeconomic"),
    ("jobs report strong, economy growing, great day for markets", "positive", "Macroeconomic"),
]

# Scripted storyline (guarantees interesting demo moments), day index -> items
SCRIPTED = {
    1: [("NVDA", "Nvidia unveils next-generation AI platform to enthusiastic reviews and record preorders", "positive", "Product Launch")],
    3: [(None, "Armed conflict escalates as missile strikes hit key shipping route; global markets plunge, systemic fears grow", "negative", "Geopolitical"),
        ("XOM", "Exxon Mobil faces supply disruption as war threatens oil routes, shares slide", "negative", "Geopolitical")],
    5: [("JPM", "JPMorgan Chase credit rating downgraded as debt concerns mount, default fears spread", "negative", "Credit Event")],
    6: [(None, "Central bank shocks markets with surprise rate hike as inflation surges and recession fears grow", "negative", "Macroeconomic")],
    8: [("DIS", "Disney to acquire rival studio in deal expected to boost earnings and synergies", "positive", "Merger/Acquisition")],
    9: [(None, "Ceasefire agreement reached, peace talks ease tensions and markets rally strongly", "positive", "Geopolitical")],
    10: [("PFE", "Pfizer reports record profit and announces share buyback after strong trial results", "positive", "Other")],
}


def ts(day, hour=None):
    h = hour if hour is not None else rng.randint(8, 20)
    return f"{day.strftime('%Y-%m-%d')}T{h:02d}:{rng.randint(0, 59):02d}:00Z"


def build_news():
    out, n = [], 0
    outlets = ["Synthetic Wire", "MarketDesk Demo", "Global Ledger (synthetic)", "Demo Business Daily"]
    for i, day in enumerate(DAYS):
        for _ in range(4):
            t = rng.choice(TICKERS)
            tpl, s, c = rng.choice(CO_NEWS)
            n += 1
            out.append(dict(id=f"news-{n:03d}", source="news", outlet=rng.choice(outlets),
                            timestamp=ts(day), headline=tpl.format(c=UNIVERSE[t]["name"]),
                            gold_sentiment=s, gold_category=c))
        if rng.random() < 0.5:
            tpl, s, c = rng.choice(MARKET_NEWS)
            n += 1
            out.append(dict(id=f"news-{n:03d}", source="news", outlet=rng.choice(outlets),
                            timestamp=ts(day), headline=tpl, gold_sentiment=s, gold_category=c))
        for tk, text, s, c in SCRIPTED.get(i, []):
            n += 1
            out.append(dict(id=f"news-{n:03d}", source="news", outlet="Synthetic Wire",
                            timestamp=ts(day, 9), headline=text, gold_sentiment=s, gold_category=c))
    out.sort(key=lambda r: r["timestamp"])
    return out


def build_tweets():
    rows, n = [], 0
    users = [f"trader_{k}" for k in range(1, 40)]
    for i, day in enumerate(DAYS):
        for _ in range(6):
            t = rng.choice(TICKERS)
            tpl, s, c = rng.choice(CO_TWEETS)
            n += 1
            rows.append(dict(tweet_id=f"tw-{n:03d}", created_at=ts(day), user=rng.choice(users),
                             text=tpl.format(t=t, c=UNIVERSE[t]["name"]),
                             gold_sentiment=s, gold_category=c))
        if rng.random() < 0.6:
            tpl, s, c = rng.choice(MARKET_TWEETS)
            n += 1
            rows.append(dict(tweet_id=f"tw-{n:03d}", created_at=ts(day), user=rng.choice(users),
                             text=tpl, gold_sentiment=s, gold_category=c))
        # social echo of scripted stories
        for tk, text, s, c in SCRIPTED.get(i, []):
            n += 1
            echo = f"${tk} " + text.split(",")[0].lower() if tk else "breaking: " + text.split(";")[0].lower()
            rows.append(dict(tweet_id=f"tw-{n:03d}", created_at=ts(day, 10), user=rng.choice(users),
                             text=echo, gold_sentiment=s, gold_category=c))
    df = pd.DataFrame(rows).sort_values("created_at")
    return df


def build_prices():
    n = len(DAYS)
    market = nprng.normal(0.0004, 0.006, n)
    market[4] = -0.032   # day after armed-conflict headline
    market[7] = -0.025   # day after surprise rate hike
    market[10] = +0.018  # ceasefire rally
    data = {}
    start = {t: nprng.uniform(60, 450) for t in TICKERS}
    for t in TICKERS:
        beta = nprng.uniform(0.7, 1.3)
        idio = nprng.normal(0, 0.008, n)
        r = beta * market + idio
        data[t] = start[t] * np.cumprod(1 + r)
    df = pd.DataFrame(data, index=DAYS).round(2)
    df.index.name = "date"
    return df


def build_portfolio():
    p = {"currency": "USD", "as_of": "2026-09-14",
         "note": "Fully synthetic wholesale-banking book for demonstration only.",
         "loans": [], "bonds": [], "derivatives": [], "equities": []}
    sectors = ["Energy", "Technology", "Financials", "Healthcare", "Consumer", "Industrials", "Real Estate", "Shipping"]
    ratings = ["AAA", "AA", "A", "BBB", "BB", "B"]
    pd_map = {"AAA": 0.0003, "AA": 0.0007, "A": 0.002, "BBB": 0.006, "BB": 0.02, "B": 0.055}
    for k in range(1, 9):
        r = rng.choice(ratings[2:])
        p["loans"].append(dict(id=f"LN-{k:02d}", borrower=f"Synthetic Corp {k}", sector=sectors[k - 1],
                               rating=r, notional=rng.choice([50, 75, 100, 150, 200]) * 1e6,
                               rate_type=rng.choice(["fixed", "floating"]), spread_duration=round(rng.uniform(1.5, 4.5), 2),
                               pd=pd_map[r], lgd=round(rng.uniform(0.35, 0.55), 2)))
    for k in range(1, 7):
        r = rng.choice(ratings[:5])
        p["bonds"].append(dict(id=f"BD-{k:02d}", issuer=f"Synthetic Issuer {k}", rating=r,
                               market_value=rng.choice([40, 60, 80, 120]) * 1e6,
                               mod_duration=round(rng.uniform(2.0, 9.0), 2), convexity=round(rng.uniform(8, 90), 1),
                               sector=rng.choice(sectors)))
    p["derivatives"] = [
        dict(id="DR-01", type="interest_rate_swap", description="Pay-fixed 5Y IRS", notional=300e6, dv01=-125000.0,
             mtm=4.2e6, note="dv01 = P&L per +1bp parallel rate move (negative here means receive-fixed exposure loses when rates rise)"),
        dict(id="DR-02", type="interest_rate_swap", description="Receive-fixed 10Y IRS", notional=200e6, dv01=-160000.0, mtm=-2.1e6),
        dict(id="DR-03", type="equity_index_put", description="S&P 500 protective put", underlying_notional=250e6,
             delta=-0.35, mtm=6.5e6),
        dict(id="DR-04", type="credit_default_swap", description="CDS protection bought on corporate basket",
             notional=120e6, cs01=45000.0, mtm=1.1e6),
        dict(id="DR-05", type="fx_forward", description="EUR/USD forward", notional=90e6, mtm=0.4e6),
    ]
    # fix sign convention: DV01 = P&L per +1bp. Pay-fixed gains when rates rise.
    p["derivatives"][0]["dv01"] = 125000.0
    p["derivatives"][0].pop("note", None)
    p["derivatives"][1]["dv01"] = -160000.0
    for k, t in enumerate(["AAPL", "MSFT", "JPM", "XOM", "NVDA", "WMT"], 1):
        p["equities"].append(dict(id=f"EQ-{k:02d}", ticker=t, market_value=rng.choice([25, 40, 60, 90]) * 1e6))
    return p


def main():
    DATA_DIR.mkdir(exist_ok=True)
    news = build_news()
    (DATA_DIR / "sample_news.json").write_text(json.dumps(news, indent=2))
    tw = build_tweets()
    tw.to_csv(DATA_DIR / "sample_tweets.csv", index=False)
    build_prices().to_csv(DATA_DIR / "prices_sample.csv")
    (DATA_DIR / "portfolio_mock.json").write_text(json.dumps(build_portfolio(), indent=2))
    print(f"news={len(news)} tweets={len(tw)} -> {DATA_DIR}")


if __name__ == "__main__":
    main()
