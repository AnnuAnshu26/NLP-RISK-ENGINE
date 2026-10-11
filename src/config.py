"""Central configuration: paths, universe of tickers, thresholds."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DOCS_DIR = ROOT / "docs"

NEWS_FILE = DATA_DIR / "sample_news.json"
TWEETS_FILE = DATA_DIR / "sample_tweets.csv"
SIGNALS_FILE = DATA_DIR / "signals.json"
PRICES_FILE = DATA_DIR / "prices_sample.csv"
PORTFOLIO_FILE = DATA_DIR / "portfolio_mock.json"

# 15 constituents of the S&P 100 used by Module A
UNIVERSE = {
    "AAPL": {"name": "Apple", "sector": "Technology"},
    "MSFT": {"name": "Microsoft", "sector": "Technology"},
    "NVDA": {"name": "Nvidia", "sector": "Technology"},
    "GOOGL": {"name": "Alphabet", "sector": "Communication"},
    "META": {"name": "Meta Platforms", "sector": "Communication"},
    "AMZN": {"name": "Amazon", "sector": "Consumer Discretionary"},
    "JPM": {"name": "JPMorgan Chase", "sector": "Financials"},
    "BAC": {"name": "Bank of America", "sector": "Financials"},
    "XOM": {"name": "Exxon Mobil", "sector": "Energy"},
    "CVX": {"name": "Chevron", "sector": "Energy"},
    "JNJ": {"name": "Johnson & Johnson", "sector": "Healthcare"},
    "PFE": {"name": "Pfizer", "sector": "Healthcare"},
    "WMT": {"name": "Walmart", "sector": "Consumer Staples"},
    "KO": {"name": "Coca-Cola", "sector": "Consumer Staples"},
    "DIS": {"name": "Disney", "sector": "Communication"},
}
TICKERS = list(UNIVERSE)
DEFENSIVE_SECTORS = {"Healthcare", "Consumer Staples"}

EVENT_CATEGORIES = [
    "Geopolitical", "Macroeconomic", "Credit Event",
    "Merger/Acquisition", "Product Launch", "Other",
]

# Module B trigger rule (per problem statement)
STRESS_IMPACT_THRESHOLD = 7.0          # strictly greater than
STRESS_CATEGORIES = {"Geopolitical", "Macroeconomic"}
