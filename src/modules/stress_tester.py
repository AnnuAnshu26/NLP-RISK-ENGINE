"""Module B - Strategic Portfolio Stress Testing driven by NLP event signals.

Trigger rule (configurable): impact_score > 7 AND category in {Geopolitical, Macroeconomic}
AND adverse sentiment (a ceasefire rally is high-impact but is NOT a stress event).
Corroborating signals of the same (date, category) are clustered into one event so a
headline echoed by 5 tweets triggers one stress test, not six.

Valuation is a deliberately simplified, transparent sensitivity model:
  equities     : MV * (1 + equity_shock)
  bonds        : dP/P = -D*dy + 0.5*C*dy^2,  dy = d_rate + d_spread * rating_multiplier
  loans        : value = N * (1 - PD*LGD)   [PD scaled by scenario] - spread/rate duration repricing
  derivatives  : IRS dv01 * d_rate_bp ; equity put delta * notional * equity_shock ;
                 CDS cs01 * d_spread_bp ; FX fwd notional * fx_shock
Shock size scales with severity: x clip(1 + (impact - 7)/10, 1.0, 1.3).
"""
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.config import (PORTFOLIO_FILE, STRESS_CATEGORIES, STRESS_IMPACT_THRESHOLD)

SCENARIOS: Dict[str, Dict[str, float]] = {
    "Geopolitical":  {"equity": -0.10, "rate_bp": -50.0, "spread_bp": 150.0, "pd_mult": 1.5, "fx": -0.03},
    "Macroeconomic": {"equity": -0.08, "rate_bp": 200.0, "spread_bp": 100.0, "pd_mult": 1.4, "fx": -0.01},
    "Credit Event":  {"equity": -0.05, "rate_bp": 0.0,   "spread_bp": 200.0, "pd_mult": 2.0, "fx": 0.0},
}
RATING_SPREAD_MULT = {"AAA": 0.5, "AA": 0.6, "A": 0.8, "BBB": 1.0, "BB": 1.5, "B": 2.0}


def find_trigger_events(signals: List[Dict], threshold: float = STRESS_IMPACT_THRESHOLD,
                        categories=STRESS_CATEGORIES, require_negative: bool = True) -> List[Dict]:
    clusters: Dict[tuple, List[Dict]] = defaultdict(list)
    for s in signals:
        if s["impact_score"] > threshold and s["event_category"] in categories:
            if require_negative and s["sentiment_score"] >= 0:
                continue
            clusters[(str(s["timestamp"])[:10], s["event_category"])].append(s)
    events = []
    for (day, cat), group in clusters.items():
        top = max(group, key=lambda g: g["impact_score"])
        events.append({"event_id": f"{day}-{cat}", "date": day, "category": cat,
                       "impact_score": top["impact_score"], "sentiment_score": top["sentiment_score"],
                       "headline": top["text"], "corroborating_signals": len(group),
                       "sources": sorted({g["source_type"] for g in group})})
    return sorted(events, key=lambda e: (e["date"], -e["impact_score"]))


class StressTester:
    def __init__(self, portfolio: Dict | None = None, path: Path = PORTFOLIO_FILE):
        self.p = portfolio or json.loads(Path(path).read_text())

    @staticmethod
    def shocks_for(category: str, impact: float) -> Dict[str, float]:
        base = SCENARIOS.get(category, SCENARIOS["Credit Event"])
        scale = min(1.3, max(1.0, 1 + (impact - 7) / 10))
        sh = {k: v * scale for k, v in base.items() if k != "pd_mult"}
        sh["pd_mult"] = 1 + (base["pd_mult"] - 1) * scale
        sh["scale"] = round(scale, 3)
        return sh

    def run(self, category: str, impact: float, shocks: Dict[str, float] | None = None) -> Dict:
        sh = shocks or self.shocks_for(category, impact)
        dr, ds = sh["rate_bp"] / 1e4, sh["spread_bp"] / 1e4
        rows = []

        for l in self.p["loans"]:
            pre = l["notional"] * (1 - l["pd"] * l["lgd"])
            pd2 = min(1.0, l["pd"] * sh["pd_mult"])
            post = l["notional"] * (1 - pd2 * l["lgd"])
            post -= l["notional"] * l["spread_duration"] * ds * RATING_SPREAD_MULT.get(l["rating"], 1.0)
            if l["rate_type"] == "fixed":
                post -= l["notional"] * l["spread_duration"] * dr
            rows.append(("Loans", l["id"], l["borrower"], pre, post))

        for b in self.p["bonds"]:
            dy = dr + ds * RATING_SPREAD_MULT.get(b["rating"], 1.0)
            ch = -b["mod_duration"] * dy + 0.5 * b["convexity"] * dy ** 2
            rows.append(("Corporate Bonds", b["id"], b["issuer"], b["market_value"], b["market_value"] * (1 + ch)))

        for d in self.p["derivatives"]:
            pnl = 0.0
            if d["type"] == "interest_rate_swap":
                pnl = d["dv01"] * sh["rate_bp"]
            elif d["type"] == "equity_index_put":
                pnl = d["delta"] * d["underlying_notional"] * sh["equity"]
            elif d["type"] == "credit_default_swap":
                pnl = d["cs01"] * sh["spread_bp"]
            elif d["type"] == "fx_forward":
                pnl = d["notional"] * sh["fx"]
            rows.append(("Derivatives", d["id"], d["description"], d["mtm"], d["mtm"] + pnl))

        for e in self.p["equities"]:
            rows.append(("Equities", e["id"], e["ticker"], e["market_value"], e["market_value"] * (1 + sh["equity"])))

        pos = pd.DataFrame(rows, columns=["asset_class", "id", "name", "pre", "post"])
        pos["delta"] = pos["post"] - pos["pre"]
        by = pos.groupby("asset_class")[["pre", "post", "delta"]].sum().reset_index()
        pre_t, post_t = float(pos["pre"].sum()), float(pos["post"].sum())
        return {"category": category, "impact_score": impact, "shocks": sh,
                "pre_total": pre_t, "post_total": post_t, "delta": post_t - pre_t,
                "delta_pct": (post_t - pre_t) / pre_t * 100, "by_asset_class": by, "positions": pos}

    def run_event(self, event: Dict) -> Dict:
        res = self.run(event["category"], event["impact_score"])
        res["event"] = event
        return res
