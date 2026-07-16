"""CLI demo: simulated CVA/PFE for a gas netting set vs the desk PFE proxy.

  uv run python -m creditlab.xva.demo                       # synthetic counterparty
  uv run python -m creditlab.xva.demo --ticker KRP          # PD from CreditLab panel
  uv run python -m creditlab.xva.demo --pd 0.05 --tenor 5   # stress a weak name
"""

from __future__ import annotations

import argparse

from creditlab.counterparty.exposure import pfe_addon
from creditlab.xva import XvaInputs, run_xva


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="", help="pull PD/rating from the scored panel")
    parser.add_argument("--pd", type=float, default=0.02, help="1y PD if no ticker given")
    parser.add_argument("--recovery", type=float, default=0.4)
    parser.add_argument("--tenor", type=float, default=3.0, help="deal tenor in years")
    parser.add_argument("--quantity", type=float, default=250_000, help="MMBtu per quarter")
    parser.add_argument("--sigma", type=float, default=0.35, help="commodity vol")
    parser.add_argument("--samples", type=int, default=2000, help="Monte Carlo paths")
    parser.add_argument("--keep", action="store_true", help="print work dir with ORE reports")
    args = parser.parse_args()

    name, rating, pd_1y = "CPTY", "n/a", args.pd
    if args.ticker:
        from creditlab.counterparty.desk import load_scored_latest

        latest = load_scored_latest()
        sub = latest[latest["ticker"].str.upper() == args.ticker.upper()]
        if sub.empty:
            raise SystemExit(f"ticker {args.ticker!r} not in panel")
        row = sub.iloc[0]
        name, rating, pd_1y = str(row["ticker"]), str(row["rating"]), float(row["pd_cal"])

    inputs = XvaInputs(
        counterparty=name,
        pd_1y=pd_1y,
        recovery=args.recovery,
        tenor_years=args.tenor,
        quantity_per_quarter=args.quantity,
        sigma=args.sigma,
        samples=args.samples,
    )
    print(
        f"Counterparty {name} | rating {rating} | 1y PD {pd_1y:.2%} "
        f"→ flat hazard {inputs.hazard_rate:.4f}, recovery {args.recovery:.0%}"
    )
    print(
        f"Netting set: gas forward + fixed-price swap, {args.tenor:.1f}y, "
        f"{args.quantity:,.0f} MMBtu/quarter, sigma {args.sigma:.0%}, "
        f"{args.samples} Sobol paths\n"
    )

    res = run_xva(inputs)

    print("T0 valuation:")
    for _, r in res.npv.iterrows():
        print(f"  {r['#TradeId']:<20} NPV ${r['NPV(Base)']:>12,.0f}")

    print("\nExposure profile (netting set):")
    print(f"  {'T (y)':>6} {'EPE':>14} {'PFE95':>14}")
    for _, r in res.exposure.iterrows():
        print(f"  {r['Time']:>6.2f} {r['EPE']:>14,.0f} {r['PFE']:>14,.0f}")

    total_notional = float(res.npv["Notional(Base)"].sum())
    proxy = pfe_addon(total_notional, args.tenor, annual_vol=args.sigma)
    print(f"\nCVA (simulated): ${res.cva:,.0f}")
    print(f"Peak EPE: ${res.peak_epe:,.0f} | Peak PFE95: ${res.peak_pfe:,.0f}")
    print(
        f"Desk add-on proxy on ${total_notional:,.0f} notional: ${proxy:,.0f} "
        f"vs simulated peak PFE ${res.peak_pfe:,.0f}"
    )
    if args.keep:
        print(f"\nORE inputs & reports: {res.work_dir}")


if __name__ == "__main__":
    main()
