"""Transaction costs and P&L impact of a credit-portfolio rebalance.

Each trade pays half the bid-ask spread plus square-root market impact:

    cost_bps = HALF_SPREAD_BPS[rating] + IMPACT_BPS * sqrt(notional / debt)

where `debt` is the issuer's debt outstanding, standing in for the tradable
float. Half-spreads widen as credit quality falls; the square-root law
(Almgren et al. 2005) makes the per-unit cost grow with the share of the
float traded. Levels are indicative orders of magnitude for institutional
US corporate bond trades (TRACE effective-spread studies, e.g. Edwards,
Harris & Piwowar 2007), not a calibration: fit them to own execution data
before relying on them.

P&L impact weighs the one-off cost against the change in annual expected
carry (spread net of expected loss). Breakeven is the months of extra carry
needed to earn the cost back.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from creditlab.portfolio.lgd import mean_lgd

# rating -> half bid-ask spread, bps of notional (indicative, institutional size)
HALF_SPREAD_BPS = {
    "AAA": 5.0,
    "AA": 5.0,
    "A": 7.0,
    "BBB": 10.0,
    "BB": 20.0,
    "B": 30.0,
    "CCC": 60.0,
}
IMPACT_BPS = 100.0  # impact at 100% of float: 10 bps when trading 1% of it


def cost_bps(rating: np.ndarray, notional: np.ndarray, debt: np.ndarray) -> np.ndarray:
    """Per-trade cost in bps of the traded notional."""
    half = pd.Series(rating).map(HALF_SPREAD_BPS).to_numpy(float)
    return half + IMPACT_BPS * np.sqrt(np.abs(notional) / debt)


def expected_carry_bps(universe: pd.DataFrame) -> pd.Series:
    """Annual spread net of expected loss (PD x LGD), in bps."""
    return universe["spread_bps"] - universe["pd"] * mean_lgd() * 1e4


def trade_list(
    universe: pd.DataFrame,
    w_old: np.ndarray,
    w_new: np.ndarray,
    aum: float,
    min_weight: float = 1e-6,
) -> pd.DataFrame:
    """Trades taking the book from `w_old` to `w_new`, with estimated costs."""
    dw = np.asarray(w_new, float) - np.asarray(w_old, float)
    notional = np.abs(dw) * aum
    c_bps = cost_bps(
        universe["rating"].to_numpy(), notional, universe["debt"].to_numpy(float)
    )
    trades = universe[["name", "rating"]].assign(
        side=np.where(dw > 0, "BUY", "SELL"),
        weight_change=dw,
        notional=notional,
        cost_bps=c_bps,
        cost=notional * c_bps / 1e4,
    )
    return trades[np.abs(dw) > min_weight].sort_values("notional", ascending=False)


def pnl_impact(
    universe: pd.DataFrame, w_old: np.ndarray, w_new: np.ndarray, aum: float
) -> dict:
    """One-off trading cost vs the change in annual expected carry."""
    w_old, w_new = np.asarray(w_old, float), np.asarray(w_new, float)
    cost = float(trade_list(universe, w_old, w_new, aum)["cost"].sum())
    carry = expected_carry_bps(universe).to_numpy()
    carry_old, carry_new = float(w_old @ carry), float(w_new @ carry)
    gain = carry_new - carry_old
    cost_bps_aum = cost / aum * 1e4
    return {
        "turnover": float(np.abs(w_new - w_old).sum() / 2),
        "cost": cost,
        "cost_bps": cost_bps_aum,
        "carry_old_bps": carry_old,
        "carry_new_bps": carry_new,
        "carry_gain_bps": gain,
        "breakeven_months": cost_bps_aum / (gain / 12) if gain > 0 else np.inf,
    }
