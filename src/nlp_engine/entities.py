"""Lightweight entity linking: map text -> S&P 100 tickers from the Module A universe."""
import re
from typing import List

from src.config import UNIVERSE

ALIASES = {
    "AAPL": ["apple"], "MSFT": ["microsoft"], "NVDA": ["nvidia"], "GOOGL": ["alphabet", "google"],
    "META": ["meta platforms", "facebook"], "AMZN": ["amazon"], "JPM": ["jpmorgan", "jp morgan"],
    "BAC": ["bank of america"], "XOM": ["exxon"], "CVX": ["chevron"], "JNJ": ["johnson & johnson"],
    "PFE": ["pfizer"], "WMT": ["walmart"], "KO": ["coca-cola", "coca cola"], "DIS": ["disney"],
}
_ALIAS_RE = {t: re.compile(r"\b(" + "|".join(re.escape(a) for a in al) + r")\b", re.I)
             for t, al in ALIASES.items()}
_CASHTAG = re.compile(r"\$([A-Z]{1,5})\b")
_UPPER = re.compile(r"\b([A-Z]{2,5})\b")


def extract_tickers(text: str) -> List[str]:
    found = []
    for tk in _CASHTAG.findall(text):
        if tk in UNIVERSE:
            found.append(tk)
    for tk in _UPPER.findall(text):
        if tk in UNIVERSE and tk not in found:
            found.append(tk)
    for tk, rx in _ALIAS_RE.items():
        if rx.search(text) and tk not in found:
            found.append(tk)
    return found
