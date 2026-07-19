"""Agency ratings validation: CapIQ export parsing and comparison stats."""

from __future__ import annotations

import pandas as pd
import pytest

from creditlab.validation.agency import (
    AGENCY_TO_GRADE,
    compare_ratings,
    format_report,
    load_agency_ratings,
    normalize_name,
)

FIXTURE = "tests/fixtures/capiq_ratings_sample.csv"


def test_loader_skips_preamble_and_strips_exchange_prefix():
    df = load_agency_ratings(FIXTURE)
    assert {"ticker", "name", "nname", "agency_rating", "agency_grade"} <= set(df.columns)
    assert "BLU" in set(df["ticker"])          # NYSE: prefix stripped
    t = df[df["ticker"] == "XOV"].iloc[0]
    assert t["agency_grade"] == "BB"
    fal = df[df["ticker"] == "FAL"].iloc[0]
    assert pd.isna(fal["agency_grade"])        # SD excluded from stats


def test_normalize_name_strips_suffixes_and_parentheticals():
    assert normalize_name("Occidental Petroleum Corporation") == "OCCIDENTAL PETROLEUM"
    assert normalize_name("Las Vegas Sands Corp. (NYSE:LVS)") == "LAS VEGAS SANDS"
    assert normalize_name("McDERMOTT INTERNATIONAL, INC.") == "MCDERMOTT INTERNATIONAL"
    assert normalize_name("US FOODS, INC.") == normalize_name("US Foods, Inc.")
    assert normalize_name("Tyco") == "TYCO"    # lone token never stripped


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


def test_loader_reads_capiq_xlsx_layout(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    ws = wb.active
    rows = [
        [None, None, None, None],                                     # preamble
        ["Entity Name", "Entity ID", "Ticker", "S&P Credit Rating"],  # header
        ["SP_ENTITY_NAME", "SP_ENTITY_ID", "SP_TICKER", "RD_CREDIT_RATING_GLOBAL"],
        [None, None, None, "Local Currency LT|Current"],              # metadata
        ["Blue Chip Energy", 1, "BLU", "AA-"],
        ["No Ticker Co", 2, None, "BBB"],
        ["Unrated Co", 3, "UNR", "NR"],
    ]
    for r in rows:
        ws.append(r)
    path = tmp_path / "export.xlsx"
    wb.save(path)

    df = load_agency_ratings(str(path))
    # metadata rows dropped; blank-ticker row kept because its name can match
    assert set(df["ticker"]) == {"BLU", "UNR", ""}
    assert df[df["ticker"] == "BLU"].iloc[0]["agency_grade"] == "AA"
    assert df[df["ticker"] == ""].iloc[0]["nname"] == "NO TICKER"


def test_name_stage_matches_blank_ticker_rows():
    agency = pd.DataFrame({
        "ticker": ["BLU", "", ""],
        "name": ["Blue Chip Energy Corp", "No Ticker Holdings, LLC", "Ambig Co"],
        "nname": ["BLUE CHIP ENERGY", "NO TICKER HOLDINGS", ""],  # "" = conflicted
        "agency_rating": ["AA", "BBB", "B"],
        "agency_grade": ["AA", "BBB", "B"],
    })
    scored = pd.DataFrame({
        "ticker": ["BLU", None, None],
        "name": ["Blue Chip Energy", "NO TICKER HOLDINGS INC", "Ambig Company"],
        "rating": ["AA", "BBB", "BB"],
    })
    c = compare_ratings(scored, agency)
    assert c.n_by_ticker == 1
    assert c.n_by_name == 1        # blank-ticker row found via name
    assert c.n_matched == 2        # conflicted name stays unmatched
    assert c.exact == 1.0


def test_report_mentions_bias_and_counts():
    c = compare_ratings(_scored(), load_agency_ratings(FIXTURE))
    text = format_report(c)
    assert "matched to panel: 6" in text
    assert "harsher" in text
    assert "Confusion matrix" in text
