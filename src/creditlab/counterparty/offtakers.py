"""Commodity offtaker templates: run the limit policy on unlisted names.

Energy merchants face many counterparties with no SEC filings — municipal
utilities, commodity traders, industrial offtakers. Each template carries a
representative financial profile plus a desk *shadow rating* (the analyst's
judgement, since no model PD exists), so ``recommend_limit`` and the FO memo
work without panel data. Values are pedagogical archetypes, not real firms.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class OfftakerTemplate:
    key: str
    label: str
    shadow_rating: str
    pd_1y: float
    financials: dict[str, float] = field(default_factory=dict)
    profile: str = ""

    def row(self) -> pd.Series:
        return pd.Series({"name": self.label, "ticker": self.key.upper(), **self.financials})


OFFTAKER_TEMPLATES: dict[str, OfftakerTemplate] = {
    t.key: t
    for t in [
        OfftakerTemplate(
            key="regulated_utility",
            label="Regulated Gas & Power Utility (template)",
            shadow_rating="BBB",
            pd_1y=0.0020,
            financials=dict(
                equity=3.0e9,
                assets=9.0e9,
                leverage=0.65,
                interest_coverage=3.0,
                current_ratio=0.85,
                roa=0.025,
            ),
            profile=(
                "Regulated network utility offtake — stable allowed-return cash "
                "flows offset structurally high leverage; ratings agencies notch "
                "up for regulatory support the ratio grid cannot see."
            ),
        ),
        OfftakerTemplate(
            key="commodity_trader",
            label="Asset-Light Commodity Trader (template)",
            shadow_rating="BB",
            pd_1y=0.0090,
            financials=dict(
                equity=6.0e8,
                assets=4.0e9,
                leverage=0.75,
                interest_coverage=2.5,
                current_ratio=1.15,
                roa=0.015,
            ),
            profile=(
                "Merchant trader with a large gross balance sheet on thin equity. "
                "Liquidity-dependent business model — prefer margined / collateral "
                "structures and watch RCF headroom, not just the ratios."
            ),
        ),
        OfftakerTemplate(
            key="industrial_offtaker",
            label="Industrial Offtaker — Manufacturer (template)",
            shadow_rating="A",
            pd_1y=0.0008,
            financials=dict(
                equity=1.5e9,
                assets=3.0e9,
                leverage=0.45,
                interest_coverage=6.0,
                current_ratio=1.60,
                roa=0.050,
            ),
            profile=(
                "Investment-grade manufacturer buying power/gas under long-term "
                "offtake. Clean balance sheet; main risk is demand cyclicality "
                "over the contract tenor."
            ),
        ),
        OfftakerTemplate(
            key="municipal_utility",
            label="Municipal / Public Power Utility (template)",
            shadow_rating="A",
            pd_1y=0.0006,
            financials=dict(
                equity=8.0e8,
                assets=3.5e9,
                leverage=0.70,
                interest_coverage=2.2,
                current_ratio=1.00,
                roa=0.010,
            ),
            profile=(
                "Public power entity — strong shadow rating from municipal "
                "support, but debt-funded infrastructure means the ratio grid "
                "haircuts capacity anyway. Good example of rating vs ratios "
                "telling different stories."
            ),
        ),
        OfftakerTemplate(
            key="retail_supplier",
            label="Small Retail Energy Supplier (template)",
            shadow_rating="B",
            pd_1y=0.0450,
            financials=dict(
                equity=6.0e7,
                assets=4.0e8,
                leverage=0.80,
                interest_coverage=1.3,
                current_ratio=0.95,
                roa=0.000,
            ),
            profile=(
                "Thinly capitalised retail supplier — classic wrong-way risk in "
                "price spikes (customer hedges lose exactly when supplier credit "
                "deteriorates). Prepay or LC only."
            ),
        ),
    ]
}
