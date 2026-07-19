"""FAME private-counterparty book: export parsing, schema mapping, scoring."""

from __future__ import annotations

import os

import numpy as np
import pytest

from creditlab.counterparty.fame import load_fame_export

FIXTURE = "tests/fixtures/fame_sample.csv"
FX = 1.27


def test_loader_maps_fame_columns_to_panel_schema():
    book = load_fame_export(FIXTURE, fx_gbpusd=FX)
    assert len(book) == 5
    ngt = book[book["name"] == "NORTHERN GAS TRADING LIMITED"].iloc[0]
    assert ngt["assets"] == pytest.approx(320_000 * 1_000 * FX)
    assert ngt["equity"] == pytest.approx(96_000 * 1_000 * FX)
    assert ngt["leverage"] == pytest.approx(0.70)        # 1 − solvency 30%
    assert ngt["roa"] == pytest.approx(0.068)
    assert ngt["interest_coverage"] == pytest.approx(5.2)
    assert ngt["fame_score"] == 82
    assert ngt["debt_to_equity"] == pytest.approx(0.45)  # gearing 45%


def test_loader_handles_na_and_leverage_fallback():
    book = load_fame_export(FIXTURE, fx_gbpusd=FX).set_index("name")
    celtic = book.loc["CELTIC ENERGY RETAIL LIMITED"]
    assert np.isnan(celtic["interest_coverage"])         # "n.a." → NaN
    thames = book.loc["THAMES INDUSTRIAL OFFTAKE LLP"]
    assert np.isnan(thames["equity"])                    # equity n.a.
    assert np.isnan(thames["leverage"])                  # no solvency, no equity
    assert celtic["leverage"] == pytest.approx(0.95)     # solvency 5%


def test_asset_turnover_and_log_assets():
    book = load_fame_export(FIXTURE, fx_gbpusd=FX).set_index("name")
    h = book.loc["HIGHLAND UTILITIES GROUP PLC"]
    assert h["asset_turnover"] == pytest.approx(890 / 1250, rel=1e-3)
    assert h["log_assets"] == pytest.approx(np.log(1_250_000 * 1_000 * FX))


def test_xlsx_export_scans_past_summary_sheet(tmp_path):
    """Real FAME exports: a 'Search summary' sheet precedes the results."""
    import pandas as pd

    path = tmp_path / "fame.xlsx"
    with pd.ExcelWriter(path) as xw:
        pd.DataFrame([["Product name", "Fame"], ["Update number", "10730"]]).to_excel(
            xw, sheet_name="Search summary", index=False, header=False
        )
        pd.read_csv(FIXTURE).to_excel(xw, sheet_name="Results", index=False)
    book = load_fame_export(str(path), fx_gbpusd=FX)
    assert len(book) == 5
    assert book.set_index("name").loc["NORTHERN GAS TRADING LIMITED", "fame_score"] == 82


needs_panel = pytest.mark.skipif(
    not os.path.exists("data/processed/panel.parquet"),
    reason="local EDGAR panel not available",
)


@needs_panel
def test_private_book_scores_and_orders_sensibly():
    from creditlab.counterparty.fame import score_private_book

    scored = score_private_book(load_fame_export(FIXTURE))
    assert scored["pd_cal"].between(0, 1).all()
    s = scored.set_index("name")
    # the distressed retailer must not out-rate the strong utility
    assert (
        s.loc["CELTIC ENERGY RETAIL LIMITED", "pd_cal"]
        > s.loc["HIGHLAND UTILITIES GROUP PLC", "pd_cal"]
    )
