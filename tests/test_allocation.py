"""Mandate-constrained allocation and cost-aware rebalancing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from creditlab.portfolio.allocation import (
    Mandate,
    check_mandate,
    optimise,
    parse_oas_csv,
)
from creditlab.portfolio.ratings import GRADE_PD
from creditlab.portfolio.tcost import expected_carry_bps, pnl_impact

AUM = 1e9
SPREADS = {
    "AAA": 40.0,
    "AA": 55.0,
    "A": 70.0,
    "BBB": 100.0,
    "BB": 160.0,
    "B": 280.0,
    "CCC": 1000.0,
}


def make_universe(n: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ratings = np.array(list(SPREADS))[np.arange(n) % len(SPREADS)]
    pds = np.array([GRADE_PD[r] for r in ratings]) * rng.uniform(0.7, 1.3, n)
    return pd.DataFrame(
        {
            "name": [f"Issuer {i}" for i in range(n)],
            "rating": ratings,
            "pd": pds,
            "spread_bps": [SPREADS[r] for r in ratings],
            "debt": rng.uniform(5e8, 5e9, n),
            "sector": [f"{10 + i % 6}" for i in range(n)],
            "sic": [f"{10 + i % 6}00" for i in range(n)],
        }
    )


@pytest.fixture
def universe() -> pd.DataFrame:
    return make_universe()


@pytest.fixture
def mandate() -> Mandate:
    return Mandate(
        max_issuer=0.05,
        max_sector=0.25,
        max_ccc=0.05,
        min_ig=0.30,
        max_el_bps=60.0,
        max_share_of_debt=0.05,
    )


def test_optimal_book_passes_every_mandate_check(universe, mandate):
    w = optimise(universe, mandate, AUM)
    report = check_mandate(universe, w, mandate, AUM)
    assert all(report["checks"].values()), report
    assert w.sum() == pytest.approx(1.0)
    assert (w >= -1e-9).all()


def test_esg_screen_zeroes_excluded_issuers(universe, mandate):
    universe.loc[[3, 4], "sic"] = "1221"  # coal mining
    w = optimise(universe, mandate, AUM)
    assert w[[3, 4]] == pytest.approx([0.0, 0.0], abs=1e-7)


def test_liquidity_cap_limits_holding_to_share_of_debt(universe, mandate):
    best = int(expected_carry_bps(universe).idxmax())
    universe.loc[best, "debt"] = 1e8  # tiny float: 5% of it is 0.5% of AUM
    w = optimise(universe, mandate, AUM)
    assert w[best] <= mandate.max_share_of_debt * 1e8 / AUM + 1e-7


def test_infeasible_mandate_raises(universe):
    with pytest.raises(ValueError, match="infeasible"):
        optimise(universe, Mandate(max_issuer=0.01), AUM)  # 60 names x 1% < 100%


def test_turnover_cap_is_respected(universe, mandate):
    w_old = optimise(universe, mandate, AUM)
    moved = universe.assign(
        spread_bps=universe["spread_bps"] * np.linspace(0.8, 1.2, len(universe))
    )
    capped = Mandate(**{**mandate.__dict__, "max_turnover": 0.10})
    w_new = optimise(moved, capped, AUM, w_old=w_old)
    assert np.abs(w_new - w_old).sum() / 2 <= 0.10 + 1e-6


def test_cost_aware_rebalance_beats_naive_after_costs(universe, mandate):
    w_old = optimise(universe, mandate, AUM)
    moved = universe.assign(
        spread_bps=universe["spread_bps"] * np.linspace(0.9, 1.1, len(universe))
    )
    naive = optimise(moved, mandate, AUM, w_old=w_old, with_costs=False)
    aware = optimise(moved, mandate, AUM, w_old=w_old)

    p_naive = pnl_impact(moved, w_old, naive, AUM)
    p_aware = pnl_impact(moved, w_old, aware, AUM)
    assert p_aware["turnover"] < p_naive["turnover"]
    assert p_aware["cost_bps"] < p_naive["cost_bps"]
    net = lambda p: p["carry_new_bps"] - p["cost_bps"]  # 1-year horizon
    assert net(p_aware) >= net(p_naive) - 1e-4


def test_parse_oas_csv_converts_percent_to_bps_and_skips_gaps():
    text = (
        "observation_date,BAMLC0A1CAAA,BAMLC0A2CAA,BAMLC0A3CA,BAMLC0A4CBBB,"
        "BAMLH0A1HYBB,BAMLH0A2HYB,BAMLH0A3HYC\n"
        "2026-09-17,0.39,0.57,0.66,0.95,1.56,2.77,10.76\n"
        "2026-09-18,.,.,.,.,.,.,.\n"
    )
    oas = parse_oas_csv(text)
    assert list(oas.columns) == list(SPREADS)
    assert len(oas) == 1
    assert oas.iloc[0]["BBB"] == pytest.approx(95.0)
    assert oas.iloc[0]["CCC"] == pytest.approx(1076.0)
