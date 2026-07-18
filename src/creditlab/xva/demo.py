"""CLI demo: simulated CVA/PFE for a gas netting set vs the desk PFE proxy.

  uv run python -m creditlab.xva.demo                       # synthetic market
  uv run python -m creditlab.xva.demo --real --ticker KRP   # FRED + NYMEX NG data
  uv run python -m creditlab.xva.demo --pd 0.05 --tenor 5   # stress a weak name
"""

from __future__ import annotations

import argparse

from creditlab.counterparty.exposure import pfe_addon
from creditlab.xva import XvaInputs, fetch_real_market, run_xva


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="", help="pull PD/rating from the scored panel")
    parser.add_argument("--pd", type=float, default=0.02, help="1y PD if no ticker given")
    parser.add_argument("--recovery", type=float, default=0.4)
    parser.add_argument("--tenor", type=float, default=3.0, help="deal tenor in years")
    parser.add_argument("--quantity", type=float, default=250_000, help="MMBtu per quarter")
    parser.add_argument(
        "--sigma", type=float, default=0.35, help="commodity vol (synthetic market only)"
    )
    parser.add_argument("--samples", type=int, default=2000, help="Monte Carlo paths")
    parser.add_argument(
        "--real",
        action="store_true",
        help="use FRED treasuries + NYMEX NG strip instead of the synthetic market",
    )
    parser.add_argument(
        "--cds-file",
        default="",
        metavar="JSON",
        help="LSEG Workspace export (codebook_cds_pull.ipynb): uses its market data "
        "and, if the counterparty has a CDS curve, prints model-vs-market CVA",
    )
    parser.add_argument("--keep", action="store_true", help="print work dir with ORE reports")
    args = parser.parse_args()

    name, rating, pd_1y = "CPTY", "n/a", args.pd
    if args.ticker:
        from creditlab.counterparty.desk import load_scored_latest

        latest = load_scored_latest()
        sub = latest[latest["ticker"].str.upper() == args.ticker.upper()]
        if not sub.empty:
            row = sub.iloc[0]
            name, rating, pd_1y = str(row["ticker"]), str(row["rating"]), float(row["pd_cal"])
        elif args.cds_file:
            # CDS names are often large caps outside the modeling panel —
            # keep going with the CLI PD as the model view
            name = args.ticker.upper()
            print(f"note: {name} not in panel — using --pd {args.pd:.2%} as model PD")
        else:
            raise SystemExit(f"ticker {args.ticker!r} not in panel")

    lseg = None
    if args.cds_file:
        from creditlab.xva.lseg import load_lseg_export

        lseg = load_lseg_export(args.cds_file)
        market = lseg.market
    else:
        market = fetch_real_market() if args.real else None
    inputs = XvaInputs(
        counterparty=name,
        pd_1y=pd_1y,
        recovery=args.recovery,
        tenor_years=args.tenor,
        quantity_per_quarter=args.quantity,
        sigma=args.sigma,
        samples=args.samples,
        market=market,
    )
    md = inputs.market
    print(
        f"Counterparty {name} | rating {rating} | 1y PD {pd_1y:.2%} "
        f"→ flat hazard {inputs.hazard_rate:.4f}, recovery {args.recovery:.0%}"
    )
    print(f"Market: {md.source}")
    for note in md.notes:
        print(f"  {note}")
    print(
        f"  gas spot {md.spot:.3f}, swap fair price {inputs.fixed_price:.3f}, "
        f"vol {md.sigma:.0%}, USD 1y zero "
        f"{dict(md.zeros).get(1.0, md.zeros[0][1]):.2%}"
    )
    print(
        f"Netting set: gas forward + fixed-price swap, {args.tenor:.1f}y, "
        f"{args.quantity:,.0f} MMBtu/quarter, {args.samples} Sobol paths\n"
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
    print(f"\nCVA (simulated): ${res.cva:,.0f}")
    print(f"Peak EPE: ${res.peak_epe:,.0f} | Peak PFE95: ${res.peak_pfe:,.0f}")
    proxy_mkt = pfe_addon(total_notional, args.tenor, annual_vol=md.sigma)
    print(
        f"Desk add-on proxy on ${total_notional:,.0f} notional at market vol "
        f"{md.sigma:.0%}: ${proxy_mkt:,.0f} vs simulated peak PFE ${res.peak_pfe:,.0f}"
    )
    if abs(md.sigma - 0.35) > 0.01:
        proxy_desk = pfe_addon(total_notional, args.tenor)
        print(
            f"Same proxy at the desk's default 35% vol: ${proxy_desk:,.0f} — "
            f"the vol assumption dominates the model choice"
        )

    if lseg is not None:
        quote = lseg.cds_for(name)
        if quote is None:
            print(f"\nNo CDS curve for {name} in {args.cds_file} — model CVA only.")
        else:
            from dataclasses import replace

            cds_res = run_xva(replace(
                inputs, cds_spreads=quote.spreads, recovery=quote.recovery,
            ))
            spread_5y = dict(quote.spreads).get(5.0)
            label = f"{spread_5y * 1e4:.0f}bp 5y" if spread_5y else f"{len(quote.spreads)} tenors"
            print(f"\n--- model vs market default risk ({name}) ---")
            row_model = f"CVA, scorecard hazard (1y PD {pd_1y:.2%}):"
            row_cds = f"CVA, CDS-implied curve ({label}):"
            w = max(len(row_model), len(row_cds)) + 2
            print(f"{row_model:<{w}}${res.cva:,.0f}")
            print(f"{row_cds:<{w}}${cds_res.cva:,.0f}")
            ratio = res.cva / cds_res.cva if cds_res.cva > 0 else float("inf")
            print(
                f"ratio {ratio:.1f}x — real-world model PD vs risk-neutral market "
                f"pricing; the gap is the desk's CVA story"
            )

    if args.keep:
        print(f"\nORE inputs & reports: {res.work_dir}")


if __name__ == "__main__":
    main()
