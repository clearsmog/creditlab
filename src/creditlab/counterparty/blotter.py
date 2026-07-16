"""Limit blotter: run the limit policy across the scored universe.

Produces one flat row per counterparty — rating, PD, proposed unsecured limit,
tenor, ratio flags, KYC status — the "Credit Risk Cube"-style CSV that credit
ops teams drop into Excel or a limits system for review.
"""

from __future__ import annotations

import pandas as pd

from creditlab.counterparty.limits import recommend_limit

RATING_ORDER = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC"]


def build_limit_blotter(
    scored: pd.DataFrame,
    *,
    rating_col: str = "rating",
    pd_col: str = "pd_cal",
) -> pd.DataFrame:
    """Apply ``recommend_limit`` to every counterparty and flatten to a table.

    ``scored`` is one row per counterparty (e.g. ``load_scored_latest()``)
    with ratios, ``rating_col`` and ``pd_col``. Rows sort best rating first,
    then largest proposed limit.
    """
    rows = []
    for _, row in scored.iterrows():
        rec = recommend_limit(row, str(row[rating_col]), float(row[pd_col]))
        rows.append(
            {
                "as_of": row.get("period_end", pd.NaT),
                "cik": row.get("cik", ""),
                "ticker": rec.ticker,
                "name": rec.name,
                "rating": rec.rating,
                "pd_1y": rec.pd_1y,
                "equity_usd": rec.equity_usd,
                "base_limit_usd": rec.base_limit_usd,
                "haircut": rec.ratio_flags.haircut,
                "recommended_limit_usd": rec.recommended_limit_usd,
                "max_tenor_years": rec.max_tenor_years,
                "leverage_flag": rec.ratio_flags.leverage,
                "coverage_flag": rec.ratio_flags.interest_coverage,
                "liquidity_flag": rec.ratio_flags.current_ratio,
                "roa_flag": rec.ratio_flags.roa,
                "kyc_status": rec.kyc_status,
                "flag_notes": "; ".join(rec.ratio_flags.notes),
            }
        )
    blotter = pd.DataFrame(rows)
    blotter["rating"] = pd.Categorical(
        blotter["rating"], categories=RATING_ORDER, ordered=True
    )
    return blotter.sort_values(
        ["rating", "recommended_limit_usd"], ascending=[True, False]
    ).reset_index(drop=True)
