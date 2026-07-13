"""Transparent counterparty limit policy for trading-credit demos.

Maps rating grade + financial-ratio flags → proposed *unsecured* credit limit,
max tenor, and documentation pack. The grid is pedagogical (not a real SEFE /
house policy) but mirrors how energy trading credit desks reason:

  1. fundamentals → PD / internal rating
  2. rating + size → base unsecured capacity
  3. leverage / liquidity / coverage haircuts
  4. tenor and doc standards by risk grade
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Base unsecured limit as a fraction of tangible equity proxy (equity book value).
# Caps prevent mega-caps from getting unbounded lines.
BASE_LIMIT_FRAC = {
    "AAA": 0.35,
    "AA": 0.30,
    "A": 0.22,
    "BBB": 0.12,
    "BB": 0.05,
    "B": 0.015,
    "CCC": 0.0,
}
HARD_CAP_USD = {
    "AAA": 250e6,
    "AA": 200e6,
    "A": 150e6,
    "BBB": 75e6,
    "BB": 25e6,
    "B": 5e6,
    "CCC": 0.0,
}
# Maximum legal tenor for unsecured trading exposure (years)
MAX_TENOR_Y = {
    "AAA": 5.0,
    "AA": 5.0,
    "A": 3.0,
    "BBB": 2.0,
    "BB": 1.0,
    "B": 0.5,
    "CCC": 0.0,
}

DOC_PACK = {
    "investment_grade": [
        "ISDA Master Agreement (or EFET for physical gas/power)",
        "CSA / credit support annex (two-way preferred)",
        "KYC / CDD pack current (<12 months)",
    ],
    "crossover": [
        "ISDA Master + CSA with reduced threshold / IA",
        "Parent Company Guarantee or LC if available",
        "KYC / CDD pack current; enhanced monitoring",
        "Shorter confirmation / close-out cycle",
    ],
    "speculative": [
        "Prepayment / LC-backed only — no unsecured line",
        "If trading: full collateral (zero threshold CSA) + PCG",
        "KYC / CDD pack current; restricted product list",
    ],
}


@dataclass
class RatioFlags:
    """Traffic-light flags from key credit ratios."""

    leverage: str  # green / amber / red
    interest_coverage: str
    current_ratio: str
    roa: str
    notes: list[str] = field(default_factory=list)

    @property
    def haircut(self) -> float:
        """Multiplicative haircut in (0, 1] from amber/red flags."""
        score = 1.0
        for flag in (self.leverage, self.interest_coverage, self.current_ratio, self.roa):
            if flag == "amber":
                score *= 0.85
            elif flag == "red":
                score *= 0.55
        return float(np.clip(score, 0.15, 1.0))


@dataclass
class LimitRecommendation:
    name: str
    ticker: str
    rating: str
    pd_1y: float
    equity_usd: float
    base_limit_usd: float
    recommended_limit_usd: float
    max_tenor_years: float
    ratio_flags: RatioFlags
    documentation: list[str]
    kyc_status: str  # demo field: clear / review / escalate
    rationale: list[str]


def _flag_leverage(x: float) -> tuple[str, str | None]:
    if np.isnan(x):
        return "amber", "leverage missing"
    if x <= 0.55:
        return "green", None
    if x <= 0.75:
        return "amber", f"elevated leverage TL/TA={x:.0%}"
    return "red", f"high leverage TL/TA={x:.0%}"


def _flag_coverage(x: float) -> tuple[str, str | None]:
    if np.isnan(x):
        return "amber", "interest coverage missing"
    if x >= 4.0:
        return "green", None
    if x >= 1.5:
        return "amber", f"thin interest coverage {x:.1f}x"
    return "red", f"weak interest coverage {x:.1f}x"


def _flag_current(x: float) -> tuple[str, str | None]:
    if np.isnan(x):
        return "amber", "current ratio missing"
    if x >= 1.2:
        return "green", None
    if x >= 0.9:
        return "amber", f"tight liquidity CR={x:.2f}"
    return "red", f"stressed liquidity CR={x:.2f}"


def _flag_roa(x: float) -> tuple[str, str | None]:
    if np.isnan(x):
        return "amber", "ROA missing"
    if x >= 0.03:
        return "green", None
    if x >= 0.0:
        return "amber", f"low ROA {x:.1%}"
    return "red", f"negative ROA {x:.1%}"


def assess_ratios(row: pd.Series) -> RatioFlags:
    """Score key FS ratios for credit memo flags."""
    notes: list[str] = []
    lev, n = _flag_leverage(float(row.get("leverage", np.nan)))
    if n:
        notes.append(n)
    cov, n = _flag_coverage(float(row.get("interest_coverage", np.nan)))
    if n:
        notes.append(n)
    cur, n = _flag_current(float(row.get("current_ratio", np.nan)))
    if n:
        notes.append(n)
    roa, n = _flag_roa(float(row.get("roa", np.nan)))
    if n:
        notes.append(n)
    return RatioFlags(
        leverage=lev,
        interest_coverage=cov,
        current_ratio=cur,
        roa=roa,
        notes=notes,
    )


def _doc_pack(rating: str) -> list[str]:
    if rating in ("AAA", "AA", "A", "BBB"):
        return list(DOC_PACK["investment_grade"])
    if rating == "BB":
        return list(DOC_PACK["crossover"])
    return list(DOC_PACK["speculative"])


def _kyc_status(rating: str, flags: RatioFlags) -> str:
    if rating in ("CCC", "B") or flags.haircut < 0.5:
        return "escalate"
    if rating == "BB" or flags.haircut < 0.85:
        return "review"
    return "clear"


def recommend_limit(
    row: pd.Series,
    rating: str,
    pd_1y: float,
    *,
    name_col: str = "name",
    ticker_col: str = "ticker",
) -> LimitRecommendation:
    """Propose unsecured trading credit limit from rating + FS flags.

    Size driver is book equity (balance-sheet capacity proxy). Energy desks often
    blend this with market-implied metrics and group support — not modelled here.
    """
    flags = assess_ratios(row)
    equity = float(row.get("equity", np.nan))
    if np.isnan(equity) or equity <= 0:
        equity = float(row.get("assets", 0.0) or 0.0) * 0.2  # fallback thin equity proxy

    frac = BASE_LIMIT_FRAC.get(rating, 0.0)
    cap = HARD_CAP_USD.get(rating, 0.0)
    base = min(equity * frac, cap) if frac > 0 else 0.0
    recommended = base * flags.haircut
    tenor = MAX_TENOR_Y.get(rating, 0.0)

    rationale = [
        f"Internal rating {rating} (model 1y PD {pd_1y:.2%}) maps to base capacity "
        f"{frac:.0%} of equity, hard-capped at ${cap/1e6:.0f}m.",
        f"Book equity proxy ${equity/1e6:.1f}m → base unsecured ${base/1e6:.1f}m "
        f"before ratio haircuts (haircut factor {flags.haircut:.2f}).",
    ]
    if flags.notes:
        rationale.append("Ratio flags: " + "; ".join(flags.notes) + ".")
    if recommended <= 0:
        rationale.append(
            "No unsecured line recommended — use prepay, LC, or full CSA collateral."
        )
    else:
        rationale.append(
            f"Proposed unsecured limit ${recommended/1e6:.1f}m, max tenor {tenor:.1f}y; "
            f"monitor MtM + PFE against limit continuously."
        )

    return LimitRecommendation(
        name=str(row.get(name_col, "") or ""),
        ticker=str(row.get(ticker_col, "") or ""),
        rating=rating,
        pd_1y=float(pd_1y),
        equity_usd=equity,
        base_limit_usd=float(base),
        recommended_limit_usd=float(recommended),
        max_tenor_years=float(tenor),
        ratio_flags=flags,
        documentation=_doc_pack(rating),
        kyc_status=_kyc_status(rating, flags),
        rationale=rationale,
    )
