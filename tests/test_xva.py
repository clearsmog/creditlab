"""CVA/PFE via ORE: config generation and (if ORE installed) full runs."""

from __future__ import annotations

import math
from datetime import date
from importlib.util import find_spec

import pytest

from creditlab.xva.configs import XvaInputs, market_txt, portfolio_xml
from creditlab.xva.marketdata import (
    MarketData,
    _ng_tickers,
    _parse_fred_csv,
    synthetic_market,
)

needs_ore = pytest.mark.skipif(find_spec("ORE") is None, reason="ORE not installed")

FAST = dict(samples=300, tenor_years=1.5)


def test_inputs_hazard_rate_from_pd():
    p = XvaInputs(pd_1y=0.02)
    assert math.isclose(p.hazard_rate, -math.log(0.98))
    assert XvaInputs(pd_1y=0.0).hazard_rate == 0.0


def test_inputs_schedule_covers_tenor():
    p = XvaInputs(asof=date(2026, 7, 15), tenor_years=3.0)
    assert p.end_date == date(2029, 7, 15)
    assert p.grid == "16,3M"  # 4y of quarterly steps > 3y maturity
    assert max(d for d, _ in p.market.forwards) > p.end_date  # curve past maturity


def test_synthetic_market_shape():
    md = synthetic_market(date(2026, 7, 15))
    assert md.source == "synthetic"
    prices = [p for _, p in md.forwards]
    assert prices == sorted(prices)  # smooth contango
    assert md.forward_at(date(2027, 7, 15)) == pytest.approx(3.5 * math.exp(0.02), rel=1e-3)


def test_market_forward_interpolation_and_average():
    md = MarketData(
        asof=date(2026, 7, 15),
        zeros=[(1.0, 0.04)],
        spot=3.0,
        forwards=[(date(2026, 8, 1), 3.0), (date(2026, 10, 1), 4.0)],
        sigma=0.35,
    )
    assert md.forward_at(date(2026, 7, 20)) == 3.0     # flat before first point
    assert md.forward_at(date(2026, 9, 1)) == pytest.approx(3.5082, rel=1e-3)
    assert md.forward_at(date(2027, 1, 1)) == 4.0      # flat extrapolation
    assert md.average_forward(date(2026, 8, 1), date(2026, 10, 1)) == 3.5


def test_market_json_round_trip():
    md = synthetic_market(date(2026, 7, 15))
    back = MarketData.from_json(md.to_json())
    assert back.asof == md.asof
    assert back.forwards == md.forwards
    assert back.zeros == md.zeros


def test_parse_fred_csv_takes_last_complete_row():
    text = (
        "observation_date,DGS3MO,DGS1,DGS2,DGS3,DGS5,DGS7,DGS10,DGS20,DGS30\n"
        "2026-07-13,3.89,4.12,4.26,4.30,4.37,4.48,4.62,5.11,5.10\n"
        "2026-07-14,3.84,4.02,4.18,4.23,4.31,4.44,4.58,5.09,5.08\n"
        "2026-07-15,.,.,.,.,.,.,.,.,.\n"
    )
    curve_date, zeros = _parse_fred_csv(text)
    assert curve_date == date(2026, 7, 14)
    assert zeros[0] == (0.25, pytest.approx(0.0384))
    assert zeros[-1] == (30.0, pytest.approx(0.0508))


def test_ng_tickers_month_codes_and_rollover():
    ticks = _ng_tickers(date(2026, 11, 20), 4)
    assert ticks[0] == (date(2026, 12, 1), "NGZ26.NYM")
    assert ticks[1] == (date(2027, 1, 1), "NGF27.NYM")
    assert ticks[3] == (date(2027, 3, 1), "NGH27.NYM")


def test_inputs_take_asof_from_real_market():
    md = synthetic_market(date(2025, 1, 2))
    p = XvaInputs(asof=date(2026, 7, 15), market=md)
    assert p.asof == date(2025, 1, 2)  # market as-of wins


def test_market_data_contains_hazard_and_curve():
    p = XvaInputs(counterparty="ACME", pd_1y=0.05)
    txt = market_txt(p)
    assert f"HAZARD_RATE/RATE/ACME/SR/USD/1Y {p.hazard_rate:.6f}" in txt
    assert "COMMODITY/PRICE/GAS/USD 3.5" in txt
    assert "RECOVERY_RATE/RATE/ACME/SR/USD 0.4" in txt


def test_portfolio_trades_reference_counterparty():
    p = XvaInputs(counterparty="ACME")
    xml = portfolio_xml(p)
    assert xml.count("<CounterParty>ACME</CounterParty>") == 2
    assert "<NettingSetId>ACME</NettingSetId>" in xml


@needs_ore
def test_run_produces_positive_cva_and_exposure(tmp_path):
    from creditlab.xva import run_xva

    res = run_xva(XvaInputs(**FAST), work_dir=str(tmp_path / "ore"))
    assert res.cva > 0
    assert res.peak_pfe > res.peak_epe > 0
    # exposure dies after maturity
    assert float(res.exposure["EPE"].iloc[-1]) == 0.0
    # ATM forward: T0 NPV small relative to notional
    fwd = res.npv[res.npv["#TradeId"] == "GasForward"].iloc[0]
    assert abs(float(fwd["NPV(Base)"])) < 0.02 * float(fwd["Notional(Base)"])


@needs_ore
def test_run_isolated_subprocess_matches_shape(tmp_path):
    from creditlab.xva import run_xva

    res = run_xva(XvaInputs(**FAST), work_dir=str(tmp_path / "iso"), isolated=True)
    assert res.cva > 0
    assert res.peak_pfe > res.peak_epe > 0


@needs_ore
def test_cva_increases_with_pd(tmp_path):
    from creditlab.xva import run_xva

    low = run_xva(XvaInputs(pd_1y=0.005, **FAST), work_dir=str(tmp_path / "low"))
    high = run_xva(XvaInputs(pd_1y=0.05, **FAST), work_dir=str(tmp_path / "high"))
    assert high.cva > 2 * low.cva
