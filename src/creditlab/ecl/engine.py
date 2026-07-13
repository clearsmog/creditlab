"""IFRS 9 expected credit loss engine.

Stage allocation and measurement:
  Stage 1 (performing):        12-month ECL
  Stage 2 (SICR):              lifetime ECL — trigger here is a >= 3 notch
                               downgrade since origination, or current CCC
  Stage 3 (credit-impaired):   lifetime ECL on defaulted exposures

Lifetime ECL sums discounted unconditional annual default probabilities
(differences of the cumulative curve from the transition matrix) times LGD
times EAD over the remaining life.

Macro scenarios are expressed through the Vasicek systematic factor: a
scenario is a value of Z held through the projection (a prolonged state),
transforming each annual PD via the conditional-PD formula. Because ECL is
convex in the factor, the probability-weighted ECL across scenarios exceeds
the base-case ECL — the reason IFRS 9 mandates multiple scenarios instead
of a single central projection.

Demo:  uv run python -m creditlab.ecl.engine
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from creditlab.portfolio.lgd import mean_lgd
from creditlab.portfolio.ratings import GRADES
from creditlab.portfolio.transitions import cumulative_pd
from creditlab.portfolio.vasicek import conditional_pd

# (name, systematic factor z, weight) — z < 0 is the bad state
SCENARIOS = [("upside", 1.28, 0.25), ("base", 0.0, 0.50), ("downside", -1.28, 0.25)]
NOTCHES = {g: i for i, g in enumerate(GRADES)}


def stage_of(rating_now: str, rating_orig: str, defaulted: bool = False) -> int:
    if defaulted:
        return 3
    sicr = NOTCHES[rating_now] - NOTCHES[rating_orig] >= 3 or rating_now == "CCC"
    return 2 if sicr else 1


def annual_default_probs(rating: str, years: int, z: float = 0.0,
                         rho: float = 0.20) -> np.ndarray:
    """Unconditional annual default probabilities, scenario-shifted via Z."""
    cum = cumulative_pd(rating, list(range(1, years + 1))).to_numpy()
    marginal = np.diff(np.concatenate([[0.0], cum]))
    if z == 0.0:
        return marginal
    return np.asarray(conditional_pd(np.clip(marginal, 1e-6, None), rho, z))


def ecl(rating: str, ead: float, stage: int, life_years: int = 5,
        z: float = 0.0, lgd: float | None = None, discount_rate: float = 0.04) -> float:
    """ECL for one exposure under one scenario."""
    lgd = mean_lgd() if lgd is None else lgd
    if stage == 3:
        return lgd * ead  # defaulted: loss is LGD on the full exposure
    years = 1 if stage == 1 else life_years
    pds = annual_default_probs(rating, years, z)
    df = (1 + discount_rate) ** -np.arange(1, years + 1)
    return float((pds * df).sum() * lgd * ead)


def weighted_ecl(rating: str, ead: float, stage: int, **kw) -> float:
    return sum(w * ecl(rating, ead, stage, z=z, **kw) for _, z, w in SCENARIOS)


def main() -> None:
    from creditlab.models.scorecard import Scorecard, calibrate_pds
    from creditlab.portfolio.ratings import assign_rating

    df = pd.read_parquet("data/processed/panel.parquet")
    df = df[df["period_end"] <= pd.Timestamp.today() - pd.DateOffset(years=1)]
    train = df[df["fyear"] <= 2019]
    card = Scorecard().fit(train, train["default_within_1y"])
    rate = lambda part: assign_rating(calibrate_pds(
        card.predict_pd(part), train["default_within_1y"].mean(), 0.015))

    first = df.sort_values("period_end").groupby("cik").head(1).copy()
    latest = df.sort_values("period_end").groupby("cik").tail(1).copy()
    first["rating_orig"], latest["rating_now"] = rate(first), rate(latest)
    book = latest.merge(first[["cik", "rating_orig"]], on="cik")
    book["ead"] = book["liabilities"].clip(lower=0)
    book = book.dropna(subset=["ead"])
    book["stage"] = [
        stage_of(n, o, bool(d)) for n, o, d in
        zip(book["rating_now"], book["rating_orig"], book["default_within_1y"])
    ]

    book["ecl"] = [weighted_ecl(r, e, s) for r, e, s in
                   zip(book["rating_now"], book["ead"], book["stage"])]
    agg = book.groupby("stage").agg(obligors=("cik", "size"), ead=("ead", "sum"),
                                    ecl=("ecl", "sum"))
    agg["coverage"] = agg["ecl"] / agg["ead"]
    agg[["ead", "ecl"]] /= 1e9
    print("IFRS 9 staging (origination = first panel year, scenario-weighted):\n")
    print(agg.to_string(float_format=lambda x: f"{x:,.3f}"))

    # convexity: weighted vs base-only, portfolio level
    base = sum(ecl(r, e, s) for r, e, s in
               zip(book["rating_now"], book["ead"], book["stage"]))
    weighted = book["ecl"].sum()
    print(f"\nportfolio ECL base-only {base/1e9:,.2f}bn vs scenario-weighted "
          f"{weighted/1e9:,.2f}bn (+{weighted/base-1:.1%} from factor convexity)")


if __name__ == "__main__":
    main()
