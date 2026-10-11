"""Event classification into the 5 required categories (+ 'Other').

Weighted keyword rules (fast, explainable, no download) with an optional
zero-shot transformer (facebook/bart-large-mnli) used when rules are not confident
and NLP_BACKEND=finbert is enabled.
"""
import re
from typing import Dict, List, Tuple

from src.config import EVENT_CATEGORIES

RULES: Dict[str, Dict[str, float]] = {
    "Geopolitical": {
        "war": 2.5, "armed conflict": 3, "conflict": 2, "missile": 2.5, "sanction": 2.5, "military": 2,
        "ceasefire": 3, "peace talks": 2.5, "invasion": 3, "tension": 1.5, "embargo": 2.5,
        "trade war": 3, "tariff": 2, "export curbs": 2, "terror": 2.5, "nato": 2, "shipping route": 2,
        "geopolit": 3, "truce": 2.5, "tanker": 2, "regional conflict": 2, "diplomatic": 1.5, "unrest": 2, "coup": 2.5,
    },
    "Macroeconomic": {
        "central bank": 3, "rate hike": 3, "rate cut": 3, "interest rate": 2.5, "inflation": 2.5,
        "recession": 3, "gdp": 3, "unemployment": 2.5, "jobs report": 3, "bond yields": 3, "yields": 1.5,
        "fed ": 2, "monetary": 2.5, "economy": 1.5, "economic": 1.5, "cpi": 2.5, "payrolls": 2.5,
    },
    "Credit Event": {
        "downgrade": 3, "credit rating": 3, "default": 3, "bankrupt": 3, "insolven": 3, "creditor": 2.5,
        "debt": 1.5, "bonds": 1.5, "restructur": 2, "covenant": 2.5, "credit": 2, "liquidity crunch": 3,
        "rating upgrade": 3, "upgrade": 1.0, "moody": 2.5, "fitch": 2.5, "bond sale": 2.5, "outlook to negative": 2.0,
    },
    "Merger/Acquisition": {
        "acquire": 3, "acquisition": 3, "merger": 3, "merge": 2.5, "takeover": 3, "buyout": 3,
        "buying a rival": 3, "deal": 1.0, "buy ": 1.5, "purchase": 2.5, "takeover bid": 3, "talks to buy": 3, "bid for": 2.5, "divest": 2.5, "spin-off": 2.5,
    },
    "Product Launch": {
        "launch": 3, "unveil": 3, "introduces": 3, "debut": 2.5, "rollout": 2.5, "new product": 3,
        "next-generation": 2, "new service": 2.5, "releases": 2, "new platform": 2.5, "rolls out": 2.5, "new chips": 2.5, "rolled out": 2.5,
    },
}
# short keys must match whole words only (e.g. "war" must not match "warns")
_WHOLE = {"war", "fed", "gdp", "cpi", "nato", "coup"}


def _rx(k: str):
    k = k.strip()
    return re.compile(r"\b" + re.escape(k) + (r"\b" if k in _WHOLE else ""))


_COMPILED = {c: [(_rx(k), k.strip(), w) for k, w in kw.items()] for c, kw in RULES.items()}


class RuleEventClassifier:
    name = "rules"

    def classify(self, text: str) -> Tuple[str, float, List[str]]:
        t = text.lower()
        scores, hits = {}, {}
        for cat, pats in _COMPILED.items():
            for rx, key, w in pats:
                if rx.search(t):
                    scores[cat] = scores.get(cat, 0.0) + w
                    hits.setdefault(cat, []).append(key)
        if not scores:
            return "Other", 0.5, []
        total = sum(scores.values())
        best = max(scores, key=lambda c: (scores[c], -EVENT_CATEGORIES.index(c)))
        if scores[best] < 2.0:
            return "Other", 0.5, hits.get(best, [])
        return best, round(scores[best] / total, 3), hits[best]


class ZeroShotEventClassifier:
    """Optional transformer fallback for low-confidence rule outputs."""
    name = "zeroshot"

    def __init__(self, model_id: str = "facebook/bart-large-mnli"):
        from transformers import pipeline
        self._pipe = pipeline("zero-shot-classification", model=model_id)

    def classify(self, text: str) -> Tuple[str, float, List[str]]:
        labels = [c for c in EVENT_CATEGORIES if c != "Other"] + ["Other"]
        r = self._pipe(text, candidate_labels=labels)
        return r["labels"][0], round(float(r["scores"][0]), 3), []
