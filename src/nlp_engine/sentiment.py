"""Sentiment scoring -> continuous value in [-1.0, 1.0].

Two interchangeable backends behind one interface:

* LexiconSentiment  - dependency-free, finance-tuned lexicon with negation handling.
                      Default; makes the prototype run anywhere, instantly, offline.
* FinBertSentiment  - ProsusAI/finbert via HuggingFace transformers (opt-in:
                      `pip install -r requirements-ml.txt` and set NLP_BACKEND=finbert).
                      score = P(positive) - P(negative).
"""
import logging
import math
import os
import re
from typing import List

log = logging.getLogger(__name__)

# multi-word phrases are matched first (and removed) so context beats single words
PHRASES = {
    "inflation surges": -2.0, "inflation surging": -2.0, "costs surge": -2.0, "yields spike": -2.0,
    "unemployment jumps": -2.0, "unemployment jumping": -2.0, "gdp contracts": -2.0,
    "fears grow": -2.0, "fears spread": -2.0, "worries growing": -1.5, "fears rising": -1.5,
    "recession fears": -2.0, "rate hike": -1.5, "rate cut": 2.0, "inflation cools": 2.0,
    "inflation cooling": 2.0, "stocks surge": 2.0, "stocks rally": 2.0, "shares jump": 2.0,
    "huge hit": 2.0, "cuts guidance": -2.0, "cut guidance": -2.0, "guidance cut": -2.0,
    "raises full-year guidance": 2.0, "raised guidance": 2.0, "guidance raised": 2.0,
    "sell everything": -2.5, "heavy losses": -2.5, "default risk": -2.5, "default fears": -2.5,
    "peace talks": 1.5, "markets rally": 2.0, "flee to safety": -1.5, "supply disruption": -2.0,
    "debt reduction": 1.0, "edges higher": 1.5, "avoids bankruptcy": 1.0, "cuts outlook": -2.0,
    "sanctions relief": 2.5, "relief announced": 1.0, "bond yields": -1.0, "rising bond yields": -2.0,
    "talks end": -1.5, "without a deal": -1.0, "in line with": 0.0, "is rejected": -1.5,
    "meet main goal": -1.0, "fails to": -1.5, "not good": -1.5, "nothing much": 0.0,
}

# exact-word entries
WORDS = {
    "miss": -1.5, "misses": -1.5, "missed": -1.5, "war": -2.0, "ouch": -1.5, "sell": -1.0,
    "selling": -1.0, "hit": -0.5, "cut": -0.5, "cuts": -0.5, "surge": 0.0, "surges": 0.0,
    "fell": -1.5, "fall": -1.0, "falls": -1.0, "drop": -1.5, "drops": -1.5, "dips": -1.0,
    "gain": 1.5, "gains": 1.5, "good": 1.0, "great": 1.5, "win": 1.5, "wins": 1.5,
    "bad": -1.5, "debt": -0.5, "risk": -1.0, "risks": -1.0, "avoid": -1.0, "shocks": -1.5,
    "shake": -1.5, "spike": -1.5, "hike": -1.0, "jumps": 0.5, "jumping": 0.5,
}

# prefix stems (token.startswith(stem))
STEMS = {
    # positive
    "bullish": 2.0, "strong": 1.5, "growth": 1.0, "grow": 1.0, "profit": 1.5, "upgrade": 2.0,
    "applaud": 1.5, "enthusias": 1.5, "amazing": 1.5, "boost": 1.5, "synerg": 1.0, "jump": 1.5,
    "buyback": 1.0, "raise": 1.0, "innovativ": 1.5, "unveil": 1.0, "record": 1.5, "robust": 1.5,
    "confidence": 1.0, "peace": 1.5, "ceasefire": 2.0, "eases": 1.0, "ease": 1.0, "cools": 1.0,
    "rall": 1.5, "beat": 1.5, "outperform": 2.0, "soar": 2.0, "climb": 1.0, "recover": 1.5,
    "upbeat": 1.5, "optimis": 1.5, "success": 1.5, "hit": 0.0, "expands": 1.0, "adoption": 1.0,
    "demand": 0.3, "agreement": 1.0, "secures": 1.0,
    "tops": 1.5, "higher": 0.8, "rave": 1.5, "relief": 1.5, "avoids": 1.0, "truce": 1.0, "easing": 1.0,
    "slows": 0.5, "appetite": 1.0, "completes": 1.0, "insane": 1.5, "moon": 1.5, "steady": 0.3,
    "agree": 0.8, "rolls": 0.3, "rebound": 1.5, "approv": 1.0,
    # negative
    "bearish": -2.0, "plunge": -2.5, "plummet": -2.5, "crash": -2.5, "tumble": -2.0,
    "slide": -1.5, "slid": -1.5, "sliding": -1.5, "slump": -2.0, "downgrade": -2.5, "default": -2.5,
    "bankrupt": -3.0, "collaps": -2.5, "loss": -1.5, "weak": -1.5, "warn": -1.5, "fear": -1.5,
    "concern": -1.0, "panic": -2.0, "alarm": -1.5, "sanction": -1.5, "conflict": -1.5,
    "escalat": -2.0, "missile": -2.0, "attack": -2.0, "invasion": -2.5, "tension": -1.5,
    "recession": -2.5, "unemployment": -1.0, "turmoil": -2.0, "crisis": -2.5, "defect": -2.0,
    "delay": -1.5, "abandon": -2.0, "block": -1.5, "terrible": -2.0, "dump": -1.5,
    "threat": -1.5, "curbs": -1.0, "disrupt": -1.5, "worr": -1.5, "systemic": -1.5,
    "unrest": -2.0, "mounting": -1.0, "slash": -1.5, "decline": -1.5, "plague": -2.0,
    "flee": -1.5, "tumbl": -2.0, "crashing": -2.5,
    "sink": -2.0, "sank": -2.0, "fails": -2.0, "failed": -2.0, "reject": -1.5, "sue": -1.5, "rattle": -1.5,
    "recall": -2.0, "contaminat": -2.0, "breach": -2.0, "shrink": -1.5, "disappoint": -1.5,
    "slowdown": -1.5, "shaky": -1.5, "fine": -0.5, "shortfall": -1.5, "layoff": -1.5, "probe": -1.0,
    "lawsuit": -1.5, "scare": -1.5, "negative": -1.0, "rising bond": 0.0, "stoking": -1.0,
}

NEGATORS = {"not", "no", "never", "without", "isn't", "doesn't", "didn't", "won't", "hardly"}
_TOKEN = re.compile(r"[a-z][a-z'\-]*")
_PHRASE_RE = [(re.compile(r"\b" + re.escape(p) + r"\b"), w) for p, w in
              sorted(PHRASES.items(), key=lambda kv: -len(kv[0]))]
_EMOJI = {"🚀": 1.5, "📈": 1.0, "📉": -1.0, "💀": -1.5, "🔥": 0.5}


class LexiconSentiment:
    name = "lexicon"

    def raw(self, text: str) -> float:
        t = text.lower()
        total = 0.0
        for rx, w in _PHRASE_RE:
            hits = len(rx.findall(t))
            if hits:
                total += w * hits
                t = rx.sub(" ", t)
        toks = _TOKEN.findall(t)
        for i, tok in enumerate(toks):
            w = WORDS.get(tok)
            if w is None:
                for stem, sw in STEMS.items():
                    if tok.startswith(stem):
                        w = sw
                        break
            if not w:
                continue
            if any(p in NEGATORS for p in toks[max(0, i - 3):i]):
                w = -0.75 * w
            total += w
        for e, w in _EMOJI.items():
            total += w * text.count(e)
        return total

    def score(self, text: str) -> float:
        return round(math.tanh(self.raw(text) / 3.0), 3)

    def score_many(self, texts: List[str]) -> List[float]:
        return [self.score(t) for t in texts]


class FinBertSentiment:
    name = "finbert"

    def __init__(self, model_id: str = "ProsusAI/finbert"):
        from transformers import pipeline  # imported lazily: heavy optional dependency
        self._pipe = pipeline("text-classification", model=model_id, top_k=None, truncation=True)

    def score_many(self, texts: List[str]) -> List[float]:
        out = []
        for res in self._pipe(list(texts), batch_size=16):
            p = {r["label"].lower(): r["score"] for r in res}
            out.append(round(p.get("positive", 0.0) - p.get("negative", 0.0), 3))
        return out

    def score(self, text: str) -> float:
        return self.score_many([text])[0]


def get_sentiment_model(backend: str | None = None):
    backend = (backend or os.getenv("NLP_BACKEND", "lexicon")).lower()
    if backend == "finbert":
        try:
            return FinBertSentiment()
        except Exception as e:  # missing deps / offline -> graceful fallback
            log.warning("FinBERT unavailable (%s); falling back to lexicon backend.", e)
    return LexiconSentiment()
