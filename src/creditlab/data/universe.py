"""Estimation universe: batch-build the labeled firm-year panel.

Universe = current NYSE/Nasdaq registrants from the SEC exchange mapping.
Known limitation (documented for the model write-up): a current-listed
universe is survivorship-biased — issuers that were liquidated or acquired
out of bankruptcy are missing, while emerged issuers (AAL, HTZ) are captured
with their full pre-default history. This understates the default rate; the
fix (harvesting delisted defaulter CIKs) is future work.

Run as a script:  uv run python -m creditlab.data.universe --sample 300
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd
import requests

from creditlab.data.edgar import (
    CACHE_DIR,
    _get,
    annual_record,
    fetch_companyfacts,
    fetch_submissions,
)
from creditlab.data.labels import default_episodes, harvest_defaulter_ciks
from creditlab.data.ratios import compute_ratios

EXCHANGE_MAP_URL = "https://www.sec.gov/files/company_tickers_exchange.json"
SIC_FINANCIAL_RANGE = (6000, 6999)


def listed_universe(exchanges: tuple[str, ...] = ("NYSE", "Nasdaq")) -> pd.DataFrame:
    """Current listed issuers (one row per CIK) from the SEC exchange mapping."""
    cache = CACHE_DIR / "company_tickers_exchange.json"
    if cache.exists():
        doc = json.loads(cache.read_text())
    else:
        doc = _get(EXCHANGE_MAP_URL).json()
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(doc))
    df = pd.DataFrame(doc["data"], columns=doc["fields"])
    df = df[df["exchange"].isin(exchanges)]
    return df.drop_duplicates("cik").reset_index(drop=True)  # one share class per issuer


def _firm_frame(cik: int, ticker: str, name: str, min_years: int) -> pd.DataFrame | None:
    """Annual records + SIC + default label for one issuer; None if unusable."""
    facts = fetch_companyfacts(cik)
    df = annual_record(facts)
    df = df.dropna(subset=["assets"])
    if len(df) < min_years:
        return None
    name = name or facts.get("entityName", "")

    doc = fetch_submissions(cik)
    df = df.reset_index()
    df.insert(0, "ticker", ticker.upper())
    df.insert(1, "cik", cik)
    df.insert(2, "name", name)
    df["sic"] = pd.to_numeric(doc.get("sic"), errors="coerce")
    df["sic_desc"] = doc.get("sicDescription", "")

    df["default_within_1y"] = 0
    df["default_date"] = pd.NaT
    for start in default_episodes(cik):
        window = (df["period_end"] < start) & (
            start <= df["period_end"] + pd.DateOffset(years=1)
        )
        df.loc[window, "default_within_1y"] = 1
        df.loc[window, "default_date"] = start
    return df


def build_labeled_panel(
    universe: pd.DataFrame,
    sample: int | None = None,
    seed: int = 42,
    min_years: int = 3,
    exclude_financials: bool = True,
    with_defaulters: bool = False,
) -> pd.DataFrame:
    """Fetch, assemble, and label the firm-year panel for a universe sample."""
    rows = [(int(f.cik), f.ticker, f.name) for f in universe.itertuples(index=False)]
    if sample is not None and sample < len(rows):
        rows = random.Random(seed).sample(rows, sample)
    if with_defaulters:
        sampled = {cik for cik, _, _ in rows}
        harvested = [c for c in harvest_defaulter_ciks() if c not in sampled]
        print(f"appending {len(harvested)} harvested defaulter CIKs")
        rows += [(c, "", "") for c in harvested]

    frames, skipped = [], 0
    for i, (cik, ticker, name) in enumerate(rows, 1):
        try:
            df = _firm_frame(cik, ticker, name, min_years)
        except requests.HTTPError:  # shells/funds/SPACs without XBRL facts
            df = None
        if df is None:
            skipped += 1
        else:
            frames.append(df)
        if i % 25 == 0:
            print(f"  {i}/{len(rows)} firms ({skipped} skipped)")

    panel = pd.concat(frames, ignore_index=True)
    derived = panel["assets"] - panel["equity"]
    panel["liabilities"] = panel["liabilities"].fillna(derived)
    panel["fyear"] = panel["period_end"].dt.year

    if exclude_financials:
        lo, hi = SIC_FINANCIAL_RANGE
        financial = panel["sic"].between(lo, hi)
        print(f"excluding {financial.sum()} financial-sector firm-years (SIC {lo}-{hi})")
        panel = panel[~financial].reset_index(drop=True)
    return panel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=300, help="number of issuers (0 = all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-financials", action="store_true")
    parser.add_argument(
        "--with-defaulters",
        action="store_true",
        help="append delisted defaulter CIKs harvested via EDGAR full-text search",
    )
    parser.add_argument("--out", default="data/processed/panel.parquet")
    args = parser.parse_args()

    universe = listed_universe()
    print(f"universe: {len(universe)} listed issuers")
    panel = build_labeled_panel(
        universe,
        sample=args.sample or None,
        seed=args.seed,
        exclude_financials=not args.include_financials,
        with_defaulters=args.with_defaulters,
    )
    panel = compute_ratios(panel)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out)
    n_default_years = int(panel["default_within_1y"].sum())
    n_defaulters = panel.loc[panel["default_within_1y"] == 1, "cik"].nunique()
    print(
        f"saved {out}: {len(panel)} firm-years, {panel['cik'].nunique()} firms, "
        f"{panel['fyear'].min()}-{panel['fyear'].max()}, "
        f"{n_default_years} default firm-years from {n_defaulters} defaulters"
    )


if __name__ == "__main__":
    main()
