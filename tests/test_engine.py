import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from src.api import app  # noqa: E402
from src.config import TICKERS  # noqa: E402
from src.ingestion import load_all, load_news, load_tweets  # noqa: E402
from src.modules import SentimentRebalancer, StressTester, find_trigger_events, load_prices  # noqa: E402
from src.nlp_engine import RiskEngine  # noqa: E402

ENG = RiskEngine()
SIGS = ENG.analyze_many(load_all())


def test_two_sources_loaded():
    assert len(load_news()) > 30 and len(load_tweets()) > 30
    assert {r["source_type"] for r in load_all()} == {"news", "social"}


def test_signal_schema_and_ranges():
    for s in SIGS:
        assert -1.0 <= s["sentiment_score"] <= 1.0
        assert 1.0 <= s["impact_score"] <= 10.0
        assert s["event_category"] in {"Geopolitical", "Macroeconomic", "Credit Event", "Merger/Acquisition",
                                       "Product Launch", "Other"}


@pytest.mark.parametrize("text,sign", [
    ("Nvidia beats earnings and raises guidance", 1),
    ("Bank defaults on bonds as panic spreads", -1),
    ("The deal is not good for shareholders", -1),
])
def test_sentiment_direction(text, sign):
    assert np.sign(ENG.analyze(text)["sentiment_score"]) == sign


def test_war_does_not_match_warns():
    assert ENG.analyze("Chevron warns of heavy losses as costs surge")["event_category"] != "Geopolitical"


def test_ticker_linking():
    assert ENG.analyze("$AAPL up, also Exxon Mobil gains")["tickers"] == ["AAPL", "XOM"]


def test_systemic_event_scores_high():
    r = ENG.analyze("Armed conflict escalates as missile strikes hit shipping route; global markets plunge")
    assert r["event_category"] == "Geopolitical" and r["impact_score"] > 7


def test_rebalancer_invariants():
    prices = load_prices()
    res = SentimentRebalancer().run(SIGS, prices)
    W = res["weights"]
    assert np.allclose(W.sum(axis=1), 1.0)
    base = 1 / len(TICKERS)
    assert W.values.min() >= 0.5 * base - 1e-9 and W.values.max() <= 2.0 * base + 1e-9


def test_rebalancer_direction():
    prices = load_prices()
    pos = [dict(signal_id="p", timestamp="2026-09-14T10:00:00Z", tickers=["NVDA"], sentiment_score=0.9,
                impact_score=5, event_category="Other", source_type="news", text="x")]
    neg = [dict(pos[0], sentiment_score=-0.9)]
    w_pos = SentimentRebalancer().run(pos, prices)["weights"].iloc[0]["NVDA"]
    w_neg = SentimentRebalancer().run(neg, prices)["weights"].iloc[0]["NVDA"]
    assert w_pos > 1 / len(TICKERS) > w_neg


def test_trigger_rule():
    ev = find_trigger_events(SIGS)
    assert ev and all(e["impact_score"] > 7 and e["category"] in {"Geopolitical", "Macroeconomic"} for e in ev)
    # positive ceasefire rally must not trigger a stress test
    assert not any("ceasefire" in e["headline"].lower() for e in ev)


def test_stress_directionality():
    st = StressTester()
    r = st.run("Macroeconomic", 9.0)
    assert r["post_total"] < r["pre_total"]
    assert set(r["by_asset_class"]["asset_class"]) == {"Loans", "Corporate Bonds", "Derivatives", "Equities"}
    assert StressTester().run("Macroeconomic", 10.0)["delta"] < st.run("Macroeconomic", 7.5)["delta"]  # severity scaling


def test_api():
    c = TestClient(app)
    assert c.get("/health").json()["status"] == "ok"
    assert len(c.get("/signals?limit=5").json()) == 5
    r = c.post("/analyze", json={"text": "Recession fears grow as unemployment jumps", "source_type": "news"}).json()
    assert r["event_category"] == "Macroeconomic" and r["sentiment_score"] < 0
    assert abs(sum(c.get("/rebalance").json()["latest_weights"].values()) - 1) < 1e-3
    evs = c.get("/stress/events").json()
    assert evs and c.get(f"/stress/run/{evs[0]['event_id']}").json()["delta"] < 0
    assert c.post("/stress/run", json={"category": "Geopolitical", "impact_score": 9}).status_code == 200
    assert c.get("/signals?category=Nope").status_code == 422
