"""Agency ratings validation: CapIQ export parsing and comparison stats."""

from __future__ import annotations

import pandas as pd
import pytest

from creditlab.validation.agency import (
    AGENCY_TO_GRADE,
    compare_ratings,
    format_report,
    load_agency_ratings,
)

FIXTURE = "tests/fixtures/capiq_ratings_sample.csv"


def test_loader_skips_preamble_and_strips_exchange_prefix():
    df = load_agency_ratings(FIXTURE)
    assert list(df.columns) == ["ticker", "agency_rating", "agency_grade"]
    assert "BLU" in set(df["ticker"])          # NYSE: prefix stripped
    assert len(df[df["ticker"] == "BLU"]) == 1  # duplicate ticker dropped
    assert df.set_index("ticker").loc["XOV", "agency_grade"] == "BB"
    assert pd.isna(df.set_index("ticker").loc["FAL", "agency_grade"])  # SD excluded


def test_notch_mapping_covers_full_scale():
    assert AGENCY_TO_GRADE["AA-"] == "AA"
    assert AGENCY_TO_GRADE["BBB+"] == "BBB"
    assert AGENCY_TO_GRADE["CC"] == "CCC"
    assert AGENCY_TO_GRADE["NR"] is None


def _scored() -> pd.DataFrame:
    return pd.DataFrame({
        "ticker": ["BLU", "MID", "XOV", "WEK", "DIS", "SOL", "FAL", "NOPE"],
        "rating": ["AA", "BBB", "BB", "CCC", "CCC", "BBB", "B", "AAA"],
    })


def test_compare_ratings_stats():
    c = compare_ratings(_scored(), load_agency_ratings(FIXTURE))
    # matched with usable rating: BLU MID XOV WEK DIS SOL (FAL is SD-excluded)
    assert c.n_matched == 6
    assert c.n_excluded == 1
    # exact: BLU, MID, XOV, DIS = 4/6; WEK off by 1 (CCC vs B), SOL off by 1
    assert c.exact == pytest.approx(4 / 6)
    assert c.within_one == 1.0
    assert c.spearman > 0.8
    # model harsher on WEK (CCC vs B) and SOL (BBB vs A) → positive bias
    assert c.mean_signed_diff > 0
    assert c.confusion.loc["AA", "AA"] == 1
    assert c.confusion.loc["CCC", "B"] == 1
    assert c.confusion.values.sum() == 6


def test_report_mentions_bias_and_counts():
    c = compare_ratings(_scored(), load_agency_ratings(FIXTURE))
    text = format_report(c)
    assert "matched to panel: 6" in text
    assert "harsher" in text
    assert "Confusion matrix" in text
