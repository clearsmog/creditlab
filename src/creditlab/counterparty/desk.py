"""CLI demo: score panel → rate → recommend trading credit limits → print memo.

  uv run python -m creditlab.counterparty.desk
  uv run python -m creditlab.counterparty.desk --ticker XOM
"""

from __future__ import annotations

import argparse

import pandas as pd

from creditlab.counterparty.exposure import headroom, pfe_addon
from creditlab.counterparty.limits import recommend_limit
from creditlab.counterparty.memo import format_credit_memo
from creditlab.models.scorecard import Scorecard, calibrate_pds
from creditlab.portfolio.ratings import assign_rating

CENTRAL_TENDENCY = 0.015


def load_scored_latest(panel_path: str = "data/processed/panel.parquet") -> pd.DataFrame:
    df = pd.read_parquet(panel_path)
    df = df[df["period_end"] <= pd.Timestamp.today() - pd.DateOffset(years=1)]
    train = df[df["fyear"] <= 2019]
    card = Scorecard().fit(train, train["default_within_1y"])
    sample_rate = float(train["default_within_1y"].mean())
    df = df.assign(
        pd_cal=calibrate_pds(card.predict_pd(df), sample_rate, CENTRAL_TENDENCY)
    )
    df["rating"] = assign_rating(df["pd_cal"].to_numpy())
    latest = df.sort_values("period_end").groupby("cik", as_index=False).tail(1)
    return latest.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="", help="issuer ticker (default: first IG-like name)")
    parser.add_argument("--notional", type=float, default=10e6, help="illustrative deal notional USD")
    parser.add_argument("--tenor", type=float, default=1.0, help="deal tenor years")
    parser.add_argument("--current-exposure", type=float, default=0.0)
    args = parser.parse_args()

    latest = load_scored_latest()
    if args.ticker:
        sub = latest[latest["ticker"].str.upper() == args.ticker.upper()]
        if sub.empty:
            raise SystemExit(f"ticker {args.ticker!r} not in panel")
        row = sub.iloc[0]
    else:
        # pick a mid-quality name with equity for a readable demo
        cand = latest.dropna(subset=["equity", "ticker"])
        cand = cand[cand["equity"] > 1e9]
        row = cand.sort_values("pd_cal").iloc[len(cand) // 3]

    rec = recommend_limit(row, str(row["rating"]), float(row["pd_cal"]))
    pfe = pfe_addon(args.notional, args.tenor)
    hr = headroom(rec.recommended_limit_usd, args.current_exposure, pfe)
    memo = format_credit_memo(
        rec,
        current_exposure_usd=args.current_exposure,
        proposed_deal_pfe_usd=pfe,
    )
    print(memo)
    print("\n--- pre-deal check ---")
    print(
        f"PFE add-on ${pfe:,.0f} on ${args.notional:,.0f} notional / {args.tenor}y → "
        f"headroom ${hr['headroom_usd']:,.0f} | breach={hr['breach']}"
    )


if __name__ == "__main__":
    main()
