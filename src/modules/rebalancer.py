"""Module A - Tactical Index Rebalancing driven by NLP sentiment.

Algorithm (daily buckets, no look-ahead):
 1. Per ticker and day, aggregate signal sentiment, weighted by impact score.
 2. Exponentially smooth across days:  S_t = decay * S_{t-1} + daily_avg_t   (clipped to [-1, 1])
 3. Market-wide signals form a market sentiment M_t. Cyclical names move with M_t,
    defensive sectors (Healthcare, Staples) are tilted UP when M_t is negative.
 4. Raw weight  w_i = base_i * (1 + k * s_i)    (positive sentiment up, negative down)
 5. Project onto [floor*base, cap*base] and renormalise to sum to 1.
"""
from typing import Dict, List

import numpy as np
import pandas as pd

from src.config import DEFENSIVE_SECTORS, TICKERS, UNIVERSE


class SentimentRebalancer:
    def __init__(self, tickers: List[str] | None = None, sensitivity: float = 0.6, decay: float = 0.6,
                 floor: float = 0.5, cap: float = 2.0, market_beta: float = 0.3):
        self.tickers = tickers or TICKERS
        self.k, self.decay, self.floor, self.cap, self.mb = sensitivity, decay, floor, cap, market_beta
        self.base = np.full(len(self.tickers), 1.0 / len(self.tickers))

    # ---- sentiment aggregation -------------------------------------------------
    @staticmethod
    def _to_frame(signals: List[Dict]) -> pd.DataFrame:
        df = pd.DataFrame(signals)
        df["date"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None).dt.normalize()
        return df

    def sentiment_matrix(self, signals: List[Dict], dates: pd.DatetimeIndex):
        df = self._to_frame(signals)
        # snap weekend/holiday signals to the next available trading day
        idx = dates.searchsorted(df["date"].values)
        keep = idx < len(dates)
        df, idx = df[keep].copy(), idx[keep]
        df["tdate"] = dates[idx]
        df["w"] = df["impact_score"]

        S = pd.DataFrame(0.0, index=dates, columns=self.tickers)   # smoothed company sentiment
        D = pd.DataFrame(0.0, index=dates, columns=self.tickers)   # raw daily average
        M = pd.Series(0.0, index=dates)                            # smoothed market sentiment

        comp = df[df["tickers"].map(len) > 0].explode("tickers")
        comp = comp[comp["tickers"].isin(self.tickers)]
        for (d, t), g in comp.groupby(["tdate", "tickers"]):
            D.loc[d, t] = np.average(g["sentiment_score"], weights=g["w"])
        mk = df[df["tickers"].map(len) == 0]
        mday = {d: np.average(g["sentiment_score"], weights=g["w"]) for d, g in mk.groupby("tdate")}

        prev_s, prev_m = np.zeros(len(self.tickers)), 0.0
        for i, d in enumerate(dates):
            prev_s = np.clip(self.decay * prev_s + D.loc[d].values, -1, 1)
            prev_m = float(np.clip(self.decay * prev_m + mday.get(d, 0.0), -1, 1))
            S.iloc[i], M.iloc[i] = prev_s, prev_m
        return S, M, D

    # ---- weights -----------------------------------------------------------------
    def _project(self, w: np.ndarray) -> np.ndarray:
        lo, hi = self.floor * self.base, self.cap * self.base
        w = w / w.sum()
        for _ in range(100):
            w = np.clip(w, lo, hi)
            s = w.sum()
            if abs(s - 1) < 1e-10:
                break
            free = (w > lo + 1e-12) & (w < hi - 1e-12) if s > 1 else (w < hi - 1e-12)
            if not free.any():
                break
            w = np.where(free, w + (1 - s) * w / w[free].sum(), w)
        return w / w.sum()

    def weights_from(self, S_row: np.ndarray, m: float) -> np.ndarray:
        eff = S_row.copy()
        for j, t in enumerate(self.tickers):
            if UNIVERSE[t]["sector"] in DEFENSIVE_SECTORS:
                eff[j] += -self.mb * min(m, 0.0)
            else:
                eff[j] += self.mb * m
        return self._project(self.base * (1 + self.k * np.clip(eff, -1, 1)))

    def run(self, signals: List[Dict], prices: pd.DataFrame) -> Dict:
        dates = prices.index
        S, M, D = self.sentiment_matrix(signals, dates)
        W = pd.DataFrame([self.weights_from(S.iloc[i].values, M.iloc[i]) for i in range(len(dates))],
                         index=dates, columns=self.tickers)
        # weights decided at close of day t are earned on day t+1 returns (no look-ahead)
        rets = prices[self.tickers].pct_change().fillna(0.0)
        port_r = (W.shift(1).fillna(1 / len(self.tickers)) * rets).sum(axis=1)
        bench_r = rets.mean(axis=1)
        perf = pd.DataFrame({"sentiment_index": (1 + port_r).cumprod() * 100,
                             "equal_weight": (1 + bench_r).cumprod() * 100})
        turnover = W.diff().abs().sum(axis=1).fillna(0.0) / 2
        return {"weights": W, "sentiment": S, "market_sentiment": M, "daily_raw": D,
                "performance": perf, "turnover": turnover}
