"""Financial ratio engineering for the firm-year panel.

Ratio set follows the standard corporate credit factor groups: leverage,
coverage, liquidity, profitability, activity, and size — the same groups
rating agencies and PD scorecards draw from. Includes Altman Z'-score
(private-firm variant: book equity instead of market cap, so it works
without market data; the market-based Z and Merton DD come in Phase 3).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

Z_PRIME_WEIGHTS = (0.717, 0.847, 3.107, 0.420, 0.998)


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    """Division that returns NaN (not inf) when the denominator is 0 or missing."""
    return num / den.replace(0, np.nan)


def compute_ratios(panel: pd.DataFrame) -> pd.DataFrame:
    """Append credit ratio columns to a firm-year panel from `edgar.build_panel`."""
    df = panel.copy()

    ta = df["assets"]
    tl = df["liabilities"]
    working_capital = df["current_assets"] - df["current_liabilities"]

    # leverage
    df["leverage"] = _safe_div(tl, ta)
    df["debt_to_equity"] = _safe_div(tl, df["equity"])
    df["ltd_to_assets"] = _safe_div(df["long_term_debt"], ta)

    # coverage
    df["interest_coverage"] = _safe_div(df["ebit"], df["interest_expense"])
    df["cfo_to_debt"] = _safe_div(df["cfo"], tl)

    # liquidity
    df["current_ratio"] = _safe_div(df["current_assets"], df["current_liabilities"])
    df["cash_to_assets"] = _safe_div(df["cash"], ta)
    df["wc_to_assets"] = _safe_div(working_capital, ta)

    # profitability
    df["roa"] = _safe_div(df["net_income"], ta)
    df["operating_margin"] = _safe_div(df["ebit"], df["revenue"])
    df["re_to_assets"] = _safe_div(df["retained_earnings"], ta)

    # activity & size
    df["asset_turnover"] = _safe_div(df["revenue"], ta)
    df["log_assets"] = np.log(ta.where(ta > 0))

    # Altman Z' (private-firm variant, Altman 1983)
    w = Z_PRIME_WEIGHTS
    df["altman_z_prime"] = (
        w[0] * df["wc_to_assets"]
        + w[1] * df["re_to_assets"]
        + w[2] * _safe_div(df["ebit"], ta)
        + w[3] * _safe_div(df["equity"], tl)
        + w[4] * df["asset_turnover"]
    )
    return df
