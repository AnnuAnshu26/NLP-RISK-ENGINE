"""Impact score: predicted market severity on a 1 (negligible) .. 10 (systemic) scale.

Transparent additive model (every driver is reported back for explainability):

  impact = base(category) + severity keywords (cap +3) + 2*|sentiment|
           + scope bonus (market-wide Geo/Macro/Credit) + 0.5 if 'global/markets'
  then x0.9 for social media (lower source reliability), clipped to [1, 10].
"""
import re
from typing import List, Tuple

BASE = {"Geopolitical": 5.5, "Macroeconomic": 5.0, "Credit Event": 4.5,
        "Merger/Acquisition": 3.5, "Product Launch": 2.5, "Other": 2.0}

SEVERITY = ["systemic", "crisis", "war", "plunge", "plummet", "crash", "collaps", "default", "panic",
            "escalat", "missile", "invasion", "recession", "turmoil", "bankrupt", "contagion",
            "sweeping", "surprise"]
_SEV_RE = [(re.compile(r"\b" + k + (r"\b" if k == "war" else "")), k) for k in SEVERITY]
_BROAD = re.compile(r"\b(global|markets?|investors|worldwide|sector)\b")
_MARKET_CATS = {"Geopolitical", "Macroeconomic", "Credit Event"}


def impact_score(text: str, category: str, sentiment: float, scope: str,
                 source_type: str = "news") -> Tuple[float, List[str]]:
    t = text.lower()
    drivers = [f"base[{category}]={BASE.get(category, 2.0)}"]
    score = BASE.get(category, 2.0)

    sev_hits = [k for rx, k in _SEV_RE if rx.search(t)]
    sev = min(3.0, 0.8 * len(sev_hits))
    if sev:
        drivers.append(f"severity{sev_hits}=+{sev:.1f}")
        score += sev

    s_term = 2.0 * abs(sentiment)
    drivers.append(f"|sentiment|x2=+{s_term:.1f}")
    score += s_term

    if scope == "market" and category in _MARKET_CATS:
        score += 1.0
        drivers.append("market-wide=+1.0")
    elif scope == "company" and category in {"Geopolitical", "Macroeconomic"}:
        score -= 0.5
        drivers.append("company-specific=-0.5")

    if _BROAD.search(t) and scope == "market":
        score += 0.5
        drivers.append("broad-language=+0.5")

    if source_type == "social":
        score *= 0.9
        drivers.append("social-source x0.9")

    return round(min(10.0, max(1.0, score)), 1), drivers
