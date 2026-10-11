"""Ingest both sources -> NLP engine -> write data/signals.json.   python scripts/run_pipeline.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion import load_all                       # noqa: E402
from src.nlp_engine import analyze_records, save_signals  # noqa: E402

if __name__ == "__main__":
    sigs = analyze_records(load_all())
    p = save_signals(sigs)
    print(f"Wrote {len(sigs)} signals -> {p}")
