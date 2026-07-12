"""SEC EDGAR XBRL ingestion: company facts -> annual firm-year records.

EDGAR's companyfacts API returns every reported value for every us-gaap tag,
including prior-year comparatives restated in later filings. Building a clean
annual series therefore requires:
  1. keeping only 10-K/FY observations,
  2. for flow (duration) tags, keeping only full-year durations (~365 days),
  3. deduplicating each period by the most recently filed value (restatements win).
"""

from __future__ import annotations

import json
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests

BASE = "https://data.sec.gov"
TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
CACHE_DIR = Path("data/raw/edgar")
# SEC fair-access policy: identify yourself, stay under 10 req/s.
HEADERS = {"User-Agent": "creditlab research zqkntu@gmail.com"}
REQUEST_INTERVAL = 0.15

# Panel concept -> us-gaap tags in preference order (issuers differ in tag usage).
INSTANT_TAGS = {
    "assets": ["Assets"],
    "liabilities": ["Liabilities"],
    "equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "current_assets": ["AssetsCurrent"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "cash": [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    ],
    "retained_earnings": ["RetainedEarningsAccumulatedDeficit"],
    "long_term_debt": ["LongTermDebtNoncurrent", "LongTermDebt"],
    "shares_outstanding": [
        "CommonStockSharesOutstanding",
        "EntityCommonStockSharesOutstanding",  # dei namespace, handled below
    ],
}
DURATION_TAGS = {
    "revenue": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
    ],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "ebit": ["OperatingIncomeLoss"],
    "interest_expense": ["InterestExpense", "InterestExpenseDebt", "InterestExpenseNonoperating"],
    "cfo": ["NetCashProvidedByUsedInOperatingActivities"],
}

_last_request = 0.0


def _get(url: str) -> requests.Response:
    global _last_request
    wait = REQUEST_INTERVAL - (time.monotonic() - _last_request)
    if wait > 0:
        time.sleep(wait)
    resp = requests.get(url, headers=HEADERS, timeout=30)
    _last_request = time.monotonic()
    resp.raise_for_status()
    return resp


def ticker_to_cik(ticker: str) -> int:
    """Resolve a ticker to its SEC CIK using the official mapping file."""
    cache = CACHE_DIR / "company_tickers.json"
    if cache.exists():
        mapping = json.loads(cache.read_text())
    else:
        mapping = _get(TICKER_MAP_URL).json()
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(mapping))
    for entry in mapping.values():
        if entry["ticker"].upper() == ticker.upper():
            return entry["cik_str"]
    raise KeyError(f"ticker {ticker!r} not found in SEC mapping")


def fetch_companyfacts(cik: int, refresh: bool = False) -> dict:
    """Fetch (and cache) the full companyfacts document for one issuer."""
    cache = CACHE_DIR / f"CIK{cik:010d}.json"
    if cache.exists() and not refresh:
        return json.loads(cache.read_text())
    facts = _get(f"{BASE}/api/xbrl/companyfacts/CIK{cik:010d}.json").json()
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(facts))
    return facts


def _annual_observations(facts: dict, tag: str, duration: bool) -> pd.Series:
    """Extract one annual series for a tag, indexed by fiscal-year-end date.

    Keeps 10-K FY rows only; for duration tags requires a full-year window;
    restatements are resolved by keeping the latest-filed value per period end.
    """
    gaap = facts.get("facts", {}).get("us-gaap", {})
    dei = facts.get("facts", {}).get("dei", {})
    node = gaap.get(tag) or dei.get(tag)
    if node is None:
        return pd.Series(dtype="float64")

    rows = []
    for unit_values in node["units"].values():
        for v in unit_values:
            if v.get("fp") != "FY" or not str(v.get("form", "")).startswith("10-K"):
                continue
            if v.get("val") is None or "end" not in v:
                continue
            if duration:
                if "start" not in v:
                    continue
                days = (date.fromisoformat(v["end"]) - date.fromisoformat(v["start"])).days
                if not 330 <= days <= 380:
                    continue
            rows.append((v["end"], v.get("filed", ""), float(v["val"])))
    if not rows:
        return pd.Series(dtype="float64")

    df = pd.DataFrame(rows, columns=["end", "filed", "val"])
    df = df.sort_values("filed").groupby("end").last()  # restatements win
    df.index = pd.to_datetime(df.index)
    return df["val"]


def annual_record(facts: dict) -> pd.DataFrame:
    """Assemble all concepts into one firm-level annual DataFrame (index: period end)."""
    series = {}
    for concept, tags in {**INSTANT_TAGS, **DURATION_TAGS}.items():
        duration = concept in DURATION_TAGS
        for tag in tags:
            s = _annual_observations(facts, tag, duration)
            if concept in series:
                # fill gaps from fallback tags without overwriting the primary
                series[concept] = series[concept].combine_first(s)
            else:
                series[concept] = s
    df = pd.DataFrame(series)
    df.index.name = "period_end"
    return df.sort_index()


def build_panel(tickers: list[str]) -> pd.DataFrame:
    """Build a firm-year panel for a list of tickers.

    Returns a long DataFrame with one row per (ticker, fiscal year end).
    """
    frames = []
    for ticker in tickers:
        cik = ticker_to_cik(ticker)
        facts = fetch_companyfacts(cik)
        df = annual_record(facts).reset_index()
        df.insert(0, "ticker", ticker.upper())
        df.insert(1, "cik", cik)
        df.insert(2, "name", facts.get("entityName", ""))
        frames.append(df)
    panel = pd.concat(frames, ignore_index=True)
    # dei cover-page tags (shares outstanding) carry filing dates, not fiscal
    # period ends, which creates spurious rows; a firm-year without total
    # assets is not a usable balance-sheet observation.
    panel = panel.dropna(subset=["assets"]).reset_index(drop=True)
    # some issuers never report total Liabilities; recover it from the
    # accounting identity when equity is available
    derived = panel["assets"] - panel["equity"]
    panel["liabilities"] = panel["liabilities"].fillna(derived)
    panel["fyear"] = panel["period_end"].dt.year
    return panel
