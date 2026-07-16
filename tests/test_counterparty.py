"""Trading-credit desk: limits, exposure, memo."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from creditlab.counterparty.blotter import build_limit_blotter
from creditlab.counterparty.exposure import headroom, pfe_addon
from creditlab.counterparty.limits import assess_ratios, recommend_limit
from creditlab.counterparty.memo import format_credit_memo


def _row(**kwargs) -> pd.Series:
    base = dict(
        name="Demo Energy Inc",
        ticker="DEMO",
        equity=5e9,
        assets=12e9,
        leverage=0.45,
        interest_coverage=6.0,
        current_ratio=1.4,
        roa=0.05,
    )
    base.update(kwargs)
    return pd.Series(base)


def test_pfe_addon_scales_with_sqrt_tenor():
    a = pfe_addon(10e6, 1.0, annual_vol=0.35, conf_z=1.65)
    b = pfe_addon(10e6, 4.0, annual_vol=0.35, conf_z=1.65)
    assert a > 0
    assert abs(b / a - 2.0) < 1e-9


def test_headroom_breach():
    h = headroom(limit_usd=5e6, current_exposure_usd=4e6, pfe_usd=2e6)
    assert h["breach"] is True
    assert h["headroom_usd"] < 0


def test_ig_gets_positive_limit():
    rec = recommend_limit(_row(), "A", 0.001)
    assert rec.recommended_limit_usd > 0
    assert rec.max_tenor_years >= 1.0
    assert rec.kyc_status == "clear"
    assert any("ISDA" in d for d in rec.documentation)


def test_ccc_no_unsecured():
    rec = recommend_limit(_row(leverage=0.9, interest_coverage=0.5), "CCC", 0.25)
    assert rec.recommended_limit_usd == 0.0
    assert rec.max_tenor_years == 0.0
    assert "prepay" in " ".join(rec.documentation).lower() or "collateral" in " ".join(
        rec.documentation
    ).lower()


def test_red_flags_haircut_limit():
    clean = recommend_limit(_row(), "BBB", 0.002)
    stressed = recommend_limit(
        _row(leverage=0.85, interest_coverage=1.0, current_ratio=0.7, roa=-0.02),
        "BBB",
        0.002,
    )
    assert stressed.recommended_limit_usd < clean.recommended_limit_usd
    assert stressed.ratio_flags.haircut < clean.ratio_flags.haircut


def test_memo_contains_decision():
    rec = recommend_limit(_row(), "BBB", 0.002)
    text = format_credit_memo(rec, proposed_deal_pfe_usd=1e6)
    assert "Counterparty credit memo" in text
    assert "BBB" in text
    assert "Proposed unsecured limit" in text


def test_assess_ratios_red_leverage():
    flags = assess_ratios(_row(leverage=0.9))
    assert flags.leverage == "red"
    assert any("leverage" in n for n in flags.notes)


def _universe() -> pd.DataFrame:
    rows = [
        _row(ticker="BBB1", name="Mid Corp", rating="BBB", pd_cal=0.004),
        _row(ticker="AA1", name="Blue Chip", rating="AA", pd_cal=0.0004, equity=20e9),
        _row(
            ticker="CCC1",
            name="Distressed Co",
            rating="CCC",
            pd_cal=0.25,
            leverage=0.9,
            interest_coverage=0.5,
        ),
    ]
    df = pd.DataFrame(rows)
    df["period_end"] = pd.Timestamp("2025-12-31")
    return df


def test_blotter_one_row_per_counterparty_sorted_by_rating():
    blotter = build_limit_blotter(_universe())
    assert len(blotter) == 3
    assert list(blotter["ticker"]) == ["AA1", "BBB1", "CCC1"]
    for col in ("rating", "pd_1y", "recommended_limit_usd", "kyc_status", "haircut"):
        assert col in blotter.columns


def test_blotter_ccc_gets_zero_line():
    blotter = build_limit_blotter(_universe())
    ccc = blotter[blotter["ticker"] == "CCC1"].iloc[0]
    assert ccc["recommended_limit_usd"] == 0.0
    assert ccc["kyc_status"] == "escalate"


def test_blotter_csv_round_trip(tmp_path):
    blotter = build_limit_blotter(_universe())
    out = tmp_path / "limits.csv"
    blotter.to_csv(out, index=False)
    back = pd.read_csv(out)
    assert len(back) == len(blotter)
    assert np.allclose(
        back["recommended_limit_usd"], blotter["recommended_limit_usd"]
    )
