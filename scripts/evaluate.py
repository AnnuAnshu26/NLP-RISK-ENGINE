"""Smoke-test the engine against the synthetic gold labels shipped with the sample data.

CAVEAT: gold labels come from the same generator as the text, so these numbers show the
pipeline is wired correctly - they are NOT evidence of real-world accuracy. For that,
run with NLP_BACKEND=finbert on Financial PhraseBank / a Kaggle labelled set.
python scripts/evaluate.py
"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion import load_all, load_news   # noqa: E402
from src.config import DATA_DIR   # noqa: E402
from src.nlp_engine import RiskEngine  # noqa: E402


def evaluate(recs=None):
    recs = recs or load_all()
    sigs = RiskEngine().analyze_many(recs)
    n = len(recs)
    s_ok = sum(1 for r, s in zip(recs, sigs) if r["gold_sentiment"] == s["sentiment_label"])
    c_ok = sum(1 for r, s in zip(recs, sigs) if r["gold_category"] == s["event_category"])
    by_src = {}
    for r, s in zip(recs, sigs):
        d = by_src.setdefault(r["source_type"], [0, 0, 0])
        d[0] += 1
        d[1] += r["gold_sentiment"] == s["sentiment_label"]
        d[2] += r["gold_category"] == s["event_category"]
    res = {"n": n, "sentiment_accuracy": round(s_ok / n, 3), "category_accuracy": round(c_ok / n, 3),
           "by_source": {k: {"n": v[0], "sentiment_acc": round(v[1] / v[0], 3), "category_acc": round(v[2] / v[0], 3)}
                         for k, v in by_src.items()}}
    return res


if __name__ == "__main__":
    print("== bundled synthetic set (template-generated; optimistic) ==")
    print(json.dumps(evaluate(), indent=2))
    print("== held-out hand-written set (harder, honest) ==")
    print(json.dumps(evaluate(load_news(DATA_DIR / "eval_heldout.json")), indent=2))
