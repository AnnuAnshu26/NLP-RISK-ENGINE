# Demo video script (target about 5 minutes; the problem statement says <=5 min live demo, the submission guide says 10 min)

Run `streamlit run src/app.py` from the repo root (so `.streamlit/config.toml` theme loads).

1. **Intro (0:00-0:30)** - hero banner + KPI cards: problem, two sources, two downstream uses.
2. **Setup (0:30-0:50)** - terminal: `pip install -r requirements.txt` then `streamlit run src/app.py`.
3. **Analyze text tab (0:50-2:15)** - click the **Rate shock** example, press Analyze; read sentiment / event / impact and the "Triggers a stress test" line; open **Why this score?** (drivers). Click **Product launch** (company-specific, NVDA). Click **Ceasefire** (high impact but positive, so no stress test). Switch to **Social post**. Scroll the Signal feed and charts.
4. **Rebalance index tab (2:15-3:20)** - weights-over-time chart; drag **Reaction strength** 0 -> 1; drag **Pick a day**; point at *Biggest increases* / *Biggest cuts*.
5. **Stress-test bank tab (3:20-4:30)** - pick a Macroeconomic event; show before / after / loss cards, the shock chips, the two charts; open **Customize shocks** and change the rate slider; pick a Geopolitical event to compare.
6. **API + tests (4:30-5:00)** - `python -m pytest -q`; optionally `uvicorn src.api:app` -> `/docs` -> `/analyze`.
7. **Impact & limits (5:00-5:30)** - efficiency, auditability, synthetic-data caveat, FinBERT next step.

Upload to YouTube as **Unlisted**, test in an incognito window, paste the link into README.md.