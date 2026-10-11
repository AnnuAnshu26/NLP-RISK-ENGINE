"""Optional LIVE fetchers (need internet; the demo works fully offline without them).

* NewsAPI  - requires a free key in env var NEWSAPI_KEY  (https://newsapi.org)
* GDELT    - DOC 2.0 API, no key required              (https://www.gdeltproject.org)
Both return records in the same normalised schema as loaders.py.
"""
import os
from datetime import datetime, timezone
from typing import Dict, List

import requests

TIMEOUT = 15


def fetch_newsapi(query: str = "stock market OR economy OR sanctions", page_size: int = 30,
                  api_key: str | None = None) -> List[Dict]:
    key = api_key or os.getenv("NEWSAPI_KEY")
    if not key:
        raise RuntimeError("NEWSAPI_KEY not set (get a free key at newsapi.org)")
    r = requests.get("https://newsapi.org/v2/everything", timeout=TIMEOUT, params={
        "q": query, "language": "en", "sortBy": "publishedAt", "pageSize": page_size, "apiKey": key})
    r.raise_for_status()
    out = []
    for i, a in enumerate(r.json().get("articles", [])):
        text = ". ".join(x for x in (a.get("title"), a.get("description")) if x)
        out.append({"id": f"newsapi-{i}", "source": (a.get("source") or {}).get("name", "newsapi"),
                    "source_type": "news", "timestamp": a.get("publishedAt") or _now(), "text": text})
    return out


def fetch_gdelt(query: str = "(economy OR sanctions OR bank OR merger) sourcelang:english",
                max_records: int = 30) -> List[Dict]:
    r = requests.get("https://api.gdeltproject.org/api/v2/doc/doc", timeout=TIMEOUT, params={
        "query": query, "mode": "ArtList", "format": "json", "maxrecords": max_records, "sort": "DateDesc"})
    r.raise_for_status()
    out = []
    for i, a in enumerate(r.json().get("articles", [])):
        # GDELT seendate looks like 20260930T101500Z
        try:
            ts = datetime.strptime(a["seendate"], "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc).isoformat()
        except Exception:
            ts = _now()
        out.append({"id": f"gdelt-{i}", "source": a.get("domain", "gdelt"), "source_type": "news",
                    "timestamp": ts, "text": a.get("title", "")})
    return out


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
