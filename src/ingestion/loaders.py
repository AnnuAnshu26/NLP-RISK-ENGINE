"""Offline loaders for the two bundled unstructured sources.

Every record is normalised to one schema so the NLP engine is source-agnostic:
    {id, source, source_type, timestamp, text, [gold_sentiment, gold_category]}
"""
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.config import NEWS_FILE, TWEETS_FILE

GOLD = ("gold_sentiment", "gold_category")


def load_news(path: Path = NEWS_FILE) -> List[Dict]:
    """Source 1: news headlines (JSON, NewsAPI/GDELT-like)."""
    raw = json.loads(Path(path).read_text())
    out = []
    for r in raw:
        rec = {"id": r["id"], "source": r.get("outlet", "news"), "source_type": "news",
               "timestamp": r["timestamp"], "text": r["headline"]}
        rec.update({k: r[k] for k in GOLD if k in r})
        out.append(rec)
    return out


def load_tweets(path: Path = TWEETS_FILE) -> List[Dict]:
    """Source 2: social posts (CSV, Kaggle tweet-sentiment-like)."""
    df = pd.read_csv(path)
    out = []
    for r in df.to_dict("records"):
        rec = {"id": r["tweet_id"], "source": f"@{r.get('user', 'user')}", "source_type": "social",
               "timestamp": r["created_at"], "text": r["text"]}
        rec.update({k: r[k] for k in GOLD if k in r and pd.notna(r[k])})
        out.append(rec)
    return out


def load_all() -> List[Dict]:
    recs = load_news() + load_tweets()
    return sorted(recs, key=lambda r: r["timestamp"])
