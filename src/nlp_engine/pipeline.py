"""RiskEngine: raw text records -> structured, machine-readable risk signals.

Signal schema (also served by the FastAPI layer and written to data/signals.json):
{
  "signal_id", "source", "source_type", "timestamp", "text",
  "tickers": ["NVDA"], "scope": "company" | "market",
  "sentiment_score": -1.0..1.0, "sentiment_label": "positive|neutral|negative",
  "event_category": "Geopolitical|Macroeconomic|Credit Event|Merger/Acquisition|Product Launch|Other",
  "category_confidence": 0..1, "impact_score": 1..10, "drivers": [...], "model": "..."
}
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List

from src.config import SIGNALS_FILE
from .entities import extract_tickers
from .event_classifier import RuleEventClassifier, ZeroShotEventClassifier
from .impact import impact_score
from .sentiment import get_sentiment_model

POS_T, NEG_T = 0.15, -0.15


def label_of(score: float) -> str:
    return "positive" if score > POS_T else "negative" if score < NEG_T else "neutral"


class RiskEngine:
    def __init__(self, backend: str | None = None):
        self.sentiment = get_sentiment_model(backend)
        self.rules = RuleEventClassifier()
        self._zs = None
        if self.sentiment.name == "finbert":
            try:
                self._zs = ZeroShotEventClassifier()
            except Exception:
                self._zs = None

    @property
    def model_name(self) -> str:
        return f"sentiment={self.sentiment.name}; events={'rules+zeroshot' if self._zs else 'rules'}"

    def _classify(self, text: str):
        cat, conf, hits = self.rules.classify(text)
        if self._zs is not None and cat == "Other":
            cat, conf, _ = self._zs.classify(text)
        return cat, conf, hits

    def analyze_many(self, records: Iterable[Dict]) -> List[Dict]:
        recs = list(records)
        sentiments = self.sentiment.score_many([r["text"] for r in recs])  # batched for FinBERT
        out = []
        for r, s in zip(recs, sentiments):
            cat, conf, hits = self._classify(r["text"])
            tickers = extract_tickers(r["text"])
            scope = "company" if tickers else "market"
            imp, drivers = impact_score(r["text"], cat, s, scope, r.get("source_type", "news"))
            out.append({
                "signal_id": r.get("id") or f"sig-{len(out)}",
                "source": r.get("source", "unknown"),
                "source_type": r.get("source_type", "news"),
                "timestamp": r.get("timestamp") or datetime.now(timezone.utc).isoformat(),
                "text": r["text"],
                "tickers": tickers,
                "scope": scope,
                "sentiment_score": s,
                "sentiment_label": label_of(s),
                "event_category": cat,
                "category_confidence": conf,
                "impact_score": imp,
                "drivers": drivers + ([f"matched{hits}"] if hits else []),
                "model": self.model_name,
            })
        return out

    def analyze(self, text: str, source_type: str = "news", timestamp: str | None = None,
                source: str = "adhoc") -> Dict:
        return self.analyze_many([{"id": "adhoc", "text": text, "source": source,
                                   "source_type": source_type, "timestamp": timestamp}])[0]


def analyze_records(records: Iterable[Dict], backend: str | None = None) -> List[Dict]:
    return RiskEngine(backend).analyze_many(records)


def save_signals(signals: List[Dict], path: Path = SIGNALS_FILE) -> Path:
    Path(path).write_text(json.dumps(signals, indent=2))
    return Path(path)


def load_signals(path: Path = SIGNALS_FILE) -> List[Dict]:
    return json.loads(Path(path).read_text())
