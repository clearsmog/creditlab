"""CVA/PFE via ORE: config generation and (if ORE installed) full runs."""

from __future__ import annotations

import math
from datetime import date
from importlib.util import find_spec

import pytest

from creditlab.xva.configs import XvaInputs, market_txt, portfolio_xml

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
    assert max(p.forward_dates) > p.end_date  # curve extends past maturity


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
def test_cva_increases_with_pd(tmp_path):
    from creditlab.xva import run_xva

    low = run_xva(XvaInputs(pd_1y=0.005, **FAST), work_dir=str(tmp_path / "low"))
    high = run_xva(XvaInputs(pd_1y=0.05, **FAST), work_dir=str(tmp_path / "high"))
    assert high.cva > 2 * low.cva
