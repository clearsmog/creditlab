"""Transaction costs and P&L impact of a credit rebalance."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from creditlab.portfolio.lgd import mean_lgd
from creditlab.portfolio.tcost import (
    HALF_SPREAD_BPS,
    cost_bps,
    expected_carry_bps,
    pnl_impact,
    trade_list,
)


@pytest.fixture
def book() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "name": ["IG Co", "HY Co"],
            "rating": ["BBB", "B"],
            "pd": [0.002, 0.03],
            "spread_bps": [100.0, 300.0],
            "debt": [1e9, 1e9],
        }
    )


def test_cost_widens_down_the_rating_scale():
    grades = list(HALF_SPREAD_BPS)
    c = cost_bps(np.array(grades), np.full(len(grades), 1e6), np.full(len(grades), 1e9))
    assert (np.diff(c) >= 0).all()
    assert c[-1] > c[0]


def test_zero_size_trade_pays_half_spread_only():
    c = cost_bps(np.array(["BB"]), np.array([0.0]), np.array([1e9]))
    assert c[0] == pytest.approx(HALF_SPREAD_BPS["BB"])


def test_impact_grows_with_share_of_debt_traded():
    small, large = cost_bps(
        np.array(["BB", "BB"]), np.array([1e6, 1e8]), np.array([1e9, 1e9])
    )
    assert large > small


def test_expected_carry_is_spread_net_of_expected_loss(book):
    carry = expected_carry_bps(book)
    assert carry.to_numpy() == pytest.approx((book["spread_bps"] - book["pd"] * mean_lgd() * 1e4).to_numpy())


def test_trade_list_sides_and_notionals(book):
    trades = trade_list(book, np.array([0.5, 0.5]), np.array([0.8, 0.2]), aum=1e8)
    assert list(trades["side"]) == ["BUY", "SELL"]
    assert trades["notional"].tolist() == pytest.approx([3e7, 3e7])
    assert (trades["cost"] > 0).all()


def test_no_trade_no_cost(book):
    w = np.array([0.5, 0.5])
    assert trade_list(book, w, w, aum=1e8).empty
    impact = pnl_impact(book, w, w, aum=1e8)
    assert impact["turnover"] == 0
    assert impact["cost_bps"] == 0


def test_pnl_impact_matches_trade_list_and_breakeven(book):
    w_old, w_new, aum = np.array([0.8, 0.2]), np.array([0.5, 0.5]), 1e8
    trades = trade_list(book, w_old, w_new, aum)
    impact = pnl_impact(book, w_old, w_new, aum)

    assert impact["turnover"] == pytest.approx(0.3)  # one-way
    assert impact["cost"] == pytest.approx(trades["cost"].sum())
    assert impact["cost_bps"] == pytest.approx(impact["cost"] / aum * 1e4)

    carry = expected_carry_bps(book).to_numpy()
    gain = float((w_new - w_old) @ carry)
    assert impact["carry_gain_bps"] == pytest.approx(gain)
    assert impact["breakeven_months"] == pytest.approx(impact["cost_bps"] / (gain / 12))


def test_breakeven_is_infinite_when_carry_falls(book):
    impact = pnl_impact(book, np.array([0.2, 0.8]), np.array([0.8, 0.2]), aum=1e8)
    assert impact["carry_gain_bps"] < 0
    assert impact["breakeven_months"] == np.inf
