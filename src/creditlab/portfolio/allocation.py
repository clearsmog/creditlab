"""Mandate-constrained credit allocation with cost-aware rebalancing.

Chooses long-only weights over a scored issuer universe to maximise expected
carry net of trading costs,

    max  w'carry - (1/H) * sum_i [ h_i |dw_i| + k_i |dw_i|^1.5 ]

where carry is spread net of expected loss (PD x LGD), dw = w - w_old, and
the cost term is the `tcost` model rewritten in weights: trading |dw| of AUM
in name i costs h_i (half-spread) plus IMPACT * sqrt(|dw| AUM / debt_i) per
unit, so k_i = IMPACT * sqrt(AUM / debt_i). Both terms are convex. H
(years) spreads the one-off cost over the expected holding period.

The mandate constrains risk (single-name, sector, CCC, IG floor, expected-
loss budget), ESG (a negative sector screen), liquidity (hold at most a set
share of an issuer's debt outstanding) and turnover. `check_mandate` reports
each limit against a book. Solved as a convex programme in CVXPY.

Spreads: ICE BofA US corporate and high-yield index OAS by rating, from
FRED (public). Each issuer gets its model rating's bucket OAS, so within a
bucket the model PD is what separates names: the optimiser expresses the
model's view against the market price of the rating. This is a proxy; put
issuer-level OAS in `spread_bps` when it is available.

Demo (build a book at spreads ~6 months ago, then rebalance to today's with
and without costs):  uv run python -m creditlab.portfolio.allocation
"""

from __future__ import annotations

import argparse
import io
import os
from dataclasses import dataclass
from datetime import date

import cvxpy as cp
import numpy as np
import pandas as pd

from creditlab.portfolio.tcost import (
    HALF_SPREAD_BPS,
    IMPACT_BPS,
    expected_carry_bps,
    pnl_impact,
    trade_list,
)

# rating -> FRED series id: ICE BofA option-adjusted spread, in percent
FRED_OAS_SERIES = {
    "AAA": "BAMLC0A1CAAA",
    "AA": "BAMLC0A2CAA",
    "A": "BAMLC0A3CA",
    "BBB": "BAMLC0A4CBBB",
    "BB": "BAMLH0A1HYBB",
    "B": "BAMLH0A2HYB",
    "CCC": "BAMLH0A3HYC",
}
FRED_OAS_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=" + ",".join(
    FRED_OAS_SERIES.values()
)

IG_GRADES = ("AAA", "AA", "A", "BBB")

# Illustrative ESG negative screen by 4-digit SIC: coal mining, tobacco,
# small arms and ordnance.
ESG_EXCLUDED_SIC = (
    "1221",
    "1222",
    "1231",
    "2111",
    "2121",
    "2131",
    "2141",
    "3482",
    "3484",
    "3489",
)


@dataclass
class Mandate:
    max_issuer: float = 0.02  # single-name cap, share of AUM
    max_sector: float = 0.15  # per 2-digit SIC major group
    max_ccc: float = 0.05  # CCC bucket cap
    min_ig: float = 0.25  # BBB-and-above floor
    max_el_bps: float = 60.0  # portfolio expected-loss budget, bps a year
    max_share_of_debt: float = 0.05  # liquidity: max share of an issuer's debt held
    max_turnover: float | None = None  # one-way, per rebalance
    excluded_sic: tuple[str, ...] = ESG_EXCLUDED_SIC


def _el_bps(universe: pd.DataFrame) -> np.ndarray:
    return (universe["spread_bps"] - expected_carry_bps(universe)).to_numpy(float)


def _excluded(universe: pd.DataFrame, mandate: Mandate) -> np.ndarray:
    return universe["sic"].astype(str).isin(mandate.excluded_sic).to_numpy()


def optimise(
    universe: pd.DataFrame,
    mandate: Mandate,
    aum: float,
    w_old: np.ndarray | None = None,
    horizon_years: float = 1.0,
    with_costs: bool = True,
) -> np.ndarray:
    """Optimal weights under `mandate`, trading from `w_old` (cash if None).

    `universe` needs columns: rating, pd, spread_bps, debt, sector, sic.
    """
    n = len(universe)
    w_old = np.zeros(n) if w_old is None else np.asarray(w_old, float)
    rating = universe["rating"].to_numpy()
    debt = universe["debt"].to_numpy(float)

    w = cp.Variable(n, nonneg=True)
    dw = w - w_old
    constraints = [
        cp.sum(w) == 1,
        w <= np.minimum(mandate.max_issuer, mandate.max_share_of_debt * debt / aum),
        cp.sum(w[np.isin(rating, ["CCC"])]) <= mandate.max_ccc,
        cp.sum(w[np.isin(rating, IG_GRADES)]) >= mandate.min_ig,
        _el_bps(universe) @ w <= mandate.max_el_bps,
    ]
    excluded = _excluded(universe, mandate)
    if excluded.any():
        constraints.append(w[excluded] == 0)
    for idx in universe.groupby("sector").indices.values():
        constraints.append(cp.sum(w[idx]) <= mandate.max_sector)
    if mandate.max_turnover is not None:
        constraints.append(cp.norm1(dw) / 2 <= mandate.max_turnover)

    objective = expected_carry_bps(universe).to_numpy(float) @ w
    if with_costs:
        half = pd.Series(rating).map(HALF_SPREAD_BPS).to_numpy(float)
        impact = IMPACT_BPS * np.sqrt(aum / debt)
        cost = half @ cp.abs(dw) + impact @ cp.power(cp.abs(dw), 1.5)
        objective = objective - cost / horizon_years

    problem = cp.Problem(cp.Maximize(objective), constraints)
    problem.solve(solver=cp.CLARABEL)
    if problem.status not in ("optimal", "optimal_inaccurate"):
        raise ValueError(f"mandate is {problem.status}")
    return np.clip(w.value, 0.0, None)


def check_mandate(
    universe: pd.DataFrame,
    w: np.ndarray,
    mandate: Mandate,
    aum: float,
    tol: float = 1e-6,
) -> dict:
    """Each mandate limit measured on book `w`, with a pass flag per limit."""
    rating = universe["rating"].to_numpy()
    held = w > tol
    liquidity = w * aum / universe["debt"].to_numpy(float)
    metrics = {
        "names_held": int(held.sum()),
        "carry_bps": float(expected_carry_bps(universe).to_numpy() @ w),
        "expected_loss_bps": float(_el_bps(universe) @ w),
        "max_issuer": float(w.max()),
        "max_sector": float(
            pd.Series(w).groupby(universe["sector"].to_numpy()).sum().max()
        ),
        "ccc_share": float(w[np.isin(rating, ["CCC"])].sum()),
        "ig_share": float(w[np.isin(rating, IG_GRADES)].sum()),
        "max_share_of_debt": float(liquidity.max()),
        "esg_excluded_weight": float(w[_excluded(universe, mandate)].sum()),
    }
    checks = {
        "fully_invested": abs(w.sum() - 1) <= tol,
        "issuer": metrics["max_issuer"] <= mandate.max_issuer + tol,
        "sector": metrics["max_sector"] <= mandate.max_sector + tol,
        "ccc": metrics["ccc_share"] <= mandate.max_ccc + tol,
        "ig_floor": metrics["ig_share"] >= mandate.min_ig - tol,
        "expected_loss": metrics["expected_loss_bps"] <= mandate.max_el_bps + tol,
        "liquidity": metrics["max_share_of_debt"] <= mandate.max_share_of_debt + tol,
        "esg": metrics["esg_excluded_weight"] <= tol,
    }
    return {"metrics": metrics, "checks": checks}


def parse_oas_csv(text: str) -> pd.DataFrame:
    """FRED multi-series CSV -> OAS in bps, one column per rating grade."""
    df = pd.read_csv(io.StringIO(text), index_col=0, parse_dates=True, na_values=".")
    df = df.rename(columns={v: k for k, v in FRED_OAS_SERIES.items()})[
        list(FRED_OAS_SERIES)
    ]
    return df.dropna() * 100


def fetch_oas(cache_dir: str = "data/processed") -> pd.DataFrame:
    """OAS history by rating from FRED, cached for the day."""
    import requests

    path = os.path.join(cache_dir, f"fred_oas_{date.today():%Y%m%d}.csv")
    if not os.path.exists(path):
        os.makedirs(cache_dir, exist_ok=True)
        with open(path, "w") as f:
            f.write(requests.get(FRED_OAS_URL, timeout=30).text)
    with open(path) as f:
        return parse_oas_csv(f.read())


def build_universe(panel_path: str = "data/processed/panel.parquet") -> pd.DataFrame:
    """Live issuers with bonds outstanding: model PD and rating, sector, debt.

    Live means no default flag, debt outstanding, and a filing in the last two
    years: older latest rows are issuers that stopped filing (merged, delisted
    or defaulted) and cannot be held.
    """
    from creditlab.counterparty.desk import load_scored_latest

    df = load_scored_latest(panel_path)
    recent = df["period_end"] >= pd.Timestamp.today() - pd.DateOffset(years=2)
    df = df[(df["default_within_1y"] == 0) & (df["long_term_debt"] > 0) & recent]
    sic = df["sic"].map(lambda s: f"{int(s):04d}" if pd.notna(s) else "")
    return pd.DataFrame(
        {
            "name": df["name"].to_numpy(),
            "rating": df["rating"].to_numpy(),
            "pd": df["pd_cal"].to_numpy(float),
            "debt": df["long_term_debt"].to_numpy(float),
            "sic": sic.to_numpy(),
            "sector": sic.str[:2].to_numpy(),
        }
    )


def _print_impact(label: str, p: dict) -> None:
    months = (
        "never" if np.isinf(p["breakeven_months"]) else f"{p['breakeven_months']:.1f}m"
    )
    print(
        f"  {label:11s} turnover {p['turnover']:6.1%}   cost {p['cost_bps']:5.1f}bp "
        f"(${p['cost'] / 1e6:5.2f}m)   carry {p['carry_old_bps']:6.1f} -> {p['carry_new_bps']:6.1f}bp"
        f"   net 1y {p['carry_gain_bps'] - p['cost_bps']:+6.1f}bp   breakeven {months}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--aum", type=float, default=2e9, help="portfolio size, USD")
    parser.add_argument(
        "--lookback-days", type=int, default=182, help="age of the starting book"
    )
    args = parser.parse_args()

    oas = fetch_oas()
    now = oas.iloc[-1]
    then = oas[oas.index <= oas.index[-1] - pd.Timedelta(days=args.lookback_days)].iloc[
        -1
    ]
    print("ICE BofA OAS by rating (bps, FRED):")
    print(
        pd.DataFrame({then.name.date(): then, now.name.date(): now})
        .T.round(0)
        .to_string()
    )

    base = build_universe()
    u_then = base.assign(spread_bps=base["rating"].map(then))
    u_now = base.assign(spread_bps=base["rating"].map(now))
    mandate = Mandate()

    w_old = optimise(u_then, mandate, args.aum)
    naive = optimise(u_now, mandate, args.aum, w_old=w_old, with_costs=False)
    aware = optimise(u_now, mandate, args.aum, w_old=w_old)

    report = check_mandate(u_now, aware, mandate, args.aum)
    m = report["metrics"]
    print(
        f"\nuniverse: {len(base)} live issuers with debt outstanding; AUM ${args.aum / 1e9:.1f}bn"
    )
    print(
        f"cost-aware book: {m['names_held']} names, carry {m['carry_bps']:.1f}bp, "
        f"EL {m['expected_loss_bps']:.1f}bp, IG {m['ig_share']:.0%}, CCC {m['ccc_share']:.1%}, "
        f"max name {m['max_issuer']:.1%}, max sector {m['max_sector']:.1%}, "
        f"max share of debt {m['max_share_of_debt']:.1%}"
    )
    failed = [k for k, ok in report["checks"].items() if not ok]
    print(
        "mandate checks: " + ("all pass" if not failed else "FAIL " + ", ".join(failed))
    )

    print(f"\nrebalance to {now.name.date()} spreads:")
    _print_impact("naive", pnl_impact(u_now, w_old, naive, args.aum))
    _print_impact("cost-aware", pnl_impact(u_now, w_old, aware, args.aum))

    trades = trade_list(u_now, w_old, aware, args.aum)
    trades = trades[trades["notional"] >= 1e5].head(10)  # skip solver dust
    print("\nlargest cost-aware trades:")
    print(
        trades.assign(
            notional=(trades["notional"] / 1e6).round(1),
            cost_bps=trades["cost_bps"].round(1),
        )[["side", "name", "rating", "notional", "cost_bps"]]
        .rename(columns={"notional": "notional_$m"})
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
