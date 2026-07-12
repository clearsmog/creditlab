"""Default event labels from SEC EDGAR 8-K filings.

An 8-K with Item 1.03 ("Bankruptcy or Receivership") marks a Chapter 7/11
filing or receivership — a hard default event with a precise date, from the
same source as the fundamentals. The submissions API lists every filing's
items, so no full-text search is needed.

Caveats: captures bankruptcies only (not distressed exchanges or missed
payments, which broader default definitions include), and only while the
issuer was an SEC registrant.
"""

from __future__ import annotations

import pandas as pd

from creditlab.data.edgar import fetch_submissions, fetch_submissions_page

ITEM_BANKRUPTCY = "1.03"


def _filing_frames(cik: int) -> list[pd.DataFrame]:
    """All filing-index frames for an issuer (recent + archived pages)."""
    doc = fetch_submissions(cik)
    frames = [pd.DataFrame(doc["filings"]["recent"])]
    for extra in doc["filings"].get("files", []):
        frames.append(pd.DataFrame(fetch_submissions_page(extra["name"])))
    return frames


def bankruptcy_events(cik: int) -> pd.DataFrame:
    """8-K Item 1.03 filings for one issuer: one row per event filing date."""
    filings = pd.concat(_filing_frames(cik), ignore_index=True)
    mask = filings["form"].str.startswith("8-K") & filings["items"].str.contains(
        ITEM_BANKRUPTCY, regex=False, na=False
    )
    events = filings.loc[mask, ["filingDate", "accessionNumber", "items"]].copy()
    events["filingDate"] = pd.to_datetime(events["filingDate"])
    return events.sort_values("filingDate").reset_index(drop=True)


def default_episodes(cik: int, episode_gap_years: int = 2) -> list[pd.Timestamp]:
    """Episode start dates. Item 1.03 fires on both entry into bankruptcy and
    plan confirmation on exit, so filings within `episode_gap_years` of the
    previous one are treated as the same episode."""
    events = bankruptcy_events(cik)
    starts: list[pd.Timestamp] = []
    last_seen = None
    for d in events["filingDate"]:
        if last_seen is None or d > last_seen + pd.DateOffset(years=episode_gap_years):
            starts.append(d)
        last_seen = d
    return starts


def label_panel(panel: pd.DataFrame, horizon_years: int = 1) -> pd.DataFrame:
    """Attach default labels to a firm-year panel from `edgar.build_panel`.

    For each firm-year, `default_within_{h}y` is 1 if a bankruptcy event was
    filed within `horizon_years` after that fiscal year end — the standard
    PD-model target construction (observation date -> outcome window).
    """
    df = panel.copy()
    col = f"default_within_{horizon_years}y"
    df[col] = 0
    df["default_date"] = pd.NaT

    for cik in df["cik"].unique():
        starts = default_episodes(int(cik))
        if not starts:
            continue
        firm = df["cik"] == cik
        for event_date in starts:
            window = (
                (df["period_end"] < event_date)
                & (event_date <= df["period_end"] + pd.DateOffset(years=horizon_years))
                & firm
            )
            df.loc[window, col] = 1
            df.loc[window, "default_date"] = event_date
    return df
