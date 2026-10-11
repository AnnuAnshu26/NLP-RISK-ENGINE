"""FastAPI layer exposing the NLP Risk Engine signals + both downstream modules.

Run:  uvicorn src.api:app --reload      then open http://127.0.0.1:8000/docs
"""
from functools import lru_cache
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from src.config import EVENT_CATEGORIES, SIGNALS_FILE
from src.ingestion import load_all
from src.modules import SentimentRebalancer, StressTester, find_trigger_events, load_prices
from src.nlp_engine import RiskEngine, load_signals, save_signals

app = FastAPI(title="NLP Risk Engine", version="1.0.0",
              description="Unstructured text -> sentiment, event category, impact score -> downstream finance modules.")


@lru_cache(maxsize=1)
def engine() -> RiskEngine:
    return RiskEngine()


def signals() -> List[dict]:
    if not SIGNALS_FILE.exists():
        save_signals(engine().analyze_many(load_all()))
    return load_signals()


class AnalyzeIn(BaseModel):
    text: str = Field(..., min_length=3, examples=["Central bank shocks markets with surprise rate hike"])
    source_type: str = Field("news", pattern="^(news|social)$")


class StressIn(BaseModel):
    category: str = Field("Geopolitical", examples=["Geopolitical"])
    impact_score: float = Field(8.0, ge=1, le=10)


@app.get("/health")
def health():
    return {"status": "ok", "model": engine().model_name, "signals": len(signals())}


@app.get("/signals")
def get_signals(ticker: Optional[str] = None, category: Optional[str] = None, source_type: Optional[str] = None,
                min_impact: float = 0.0, limit: int = Query(100, ge=1, le=1000)):
    out = signals()
    if ticker:
        out = [s for s in out if ticker.upper() in s["tickers"]]
    if category:
        if category not in EVENT_CATEGORIES:
            raise HTTPException(422, f"category must be one of {EVENT_CATEGORIES}")
        out = [s for s in out if s["event_category"] == category]
    if source_type:
        out = [s for s in out if s["source_type"] == source_type]
    out = [s for s in out if s["impact_score"] >= min_impact]
    return out[:limit]


@app.get("/signals/summary")
def summary():
    s = signals()
    cats = {}
    for x in s:
        cats[x["event_category"]] = cats.get(x["event_category"], 0) + 1
    return {"count": len(s), "by_category": cats,
            "avg_sentiment": round(sum(x["sentiment_score"] for x in s) / len(s), 3),
            "high_impact": sum(1 for x in s if x["impact_score"] > 7)}


@app.post("/analyze")
def analyze(body: AnalyzeIn):
    return engine().analyze(body.text, source_type=body.source_type)


@app.post("/pipeline/refresh")
def refresh():
    sig = engine().analyze_many(load_all())
    save_signals(sig)
    return {"signals_written": len(sig)}


@app.get("/rebalance")
def rebalance(sensitivity: float = 0.6, decay: float = 0.6):
    res = SentimentRebalancer(sensitivity=sensitivity, decay=decay).run(signals(), load_prices())
    W = res["weights"]
    return {"latest_weights": W.iloc[-1].round(4).to_dict(),
            "history": {d.strftime("%Y-%m-%d"): r.round(4).to_dict() for d, r in W.iterrows()},
            "index_level": float(res["performance"]["sentiment_index"].iloc[-1]),
            "equal_weight_level": float(res["performance"]["equal_weight"].iloc[-1])}


@app.get("/stress/events")
def stress_events(threshold: float = 7.0):
    return find_trigger_events(signals(), threshold=threshold)


def _pack(res: dict) -> dict:
    return {"category": res["category"], "impact_score": res["impact_score"], "shocks": res["shocks"],
            "pre_total": res["pre_total"], "post_total": res["post_total"], "delta": res["delta"],
            "delta_pct": round(res["delta_pct"], 3),
            "by_asset_class": res["by_asset_class"].round(0).to_dict("records")}


@app.post("/stress/run")
def stress_run(body: StressIn):
    return _pack(StressTester().run(body.category, body.impact_score))


@app.get("/stress/run/{event_id}")
def stress_event(event_id: str):
    ev = next((e for e in find_trigger_events(signals()) if e["event_id"] == event_id), None)
    if not ev:
        raise HTTPException(404, "event_id not found among triggered events")
    return {"event": ev, **_pack(StressTester().run_event(ev))}
