"""CreditMetrics-style Monte Carlo loss distribution (default mode).

One-factor Gaussian copula over the real panel portfolio: each obligor's
asset return is sqrt(rho) Z + sqrt(1-rho) eps; default when it falls below
N^{-1}(PD_grade). LGD is drawn from the seniority Beta distribution, so the
tail reflects both default clustering (via Z) and recovery uncertainty.

Outputs: EL, UL, VaR/ES at 99.9%, economic capital — compared against the
sum of standalone Basel IRB charges (which ignore the portfolio's actual
concentration and granularity, so the comparison is the point).

Demo:  uv run python -m creditlab.portfolio.simulation
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from creditlab.portfolio.lgd import beta_params
from creditlab.portfolio.ratings import GRADES, assign_rating
from creditlab.portfolio.transitions import SP_1Y, STATES
from creditlab.portfolio.vasicek import irb_capital

GRADE_PD_1Y = {g: SP_1Y[STATES.index(g), -1] for g in GRADES}
PD_FLOOR = 1e-4  # AAA/AA rows have zero observed defaults; floor for thresholds


def simulate_losses(portfolio: pd.DataFrame, rho: float = 0.20,
                    n_sims: int = 100_000, seed: int = 7,
                    seniority: str = "senior_unsecured_bond") -> np.ndarray:
    """Portfolio loss distribution. `portfolio` needs columns: rating, ead."""
    pds = np.maximum(portfolio["rating"].map(GRADE_PD_1Y).to_numpy(float), PD_FLOOR)
    eads = portfolio["ead"].to_numpy(float)
    thresholds = norm.ppf(pds)
    a, b = beta_params(seniority)

    rng = np.random.default_rng(seed)
    losses = np.empty(n_sims)
    chunk = max(1, 20_000_000 // len(portfolio))  # bound memory
    for i in range(0, n_sims, chunk):
        n = min(chunk, n_sims - i)
        z = rng.standard_normal((n, 1))
        x = np.sqrt(rho) * z + np.sqrt(1 - rho) * rng.standard_normal((n, len(portfolio)))
        defaulted = x < thresholds
        lgds = rng.beta(a, b, size=defaulted.shape)
        losses[i : i + n] = (defaulted * lgds * eads).sum(axis=1)
    return losses


def summarize(losses: np.ndarray, total_ead: float, alpha: float = 0.999) -> dict:
    var = float(np.quantile(losses, alpha))
    return {
        "EL": float(losses.mean()),
        "UL": float(losses.std()),
        "VaR": var,
        "ES": float(losses[losses >= var].mean()),
        "EC": var - float(losses.mean()),
        "total_ead": total_ead,
    }


def build_portfolio() -> pd.DataFrame:
    """Latest firm-year per obligor from the panel; EAD = total liabilities."""
    from creditlab.models.scorecard import Scorecard, calibrate_pds

    df = pd.read_parquet("data/processed/panel.parquet")
    df = df[df["period_end"] <= pd.Timestamp.today() - pd.DateOffset(years=1)]
    train = df[df["fyear"] <= 2019]
    card = Scorecard().fit(train, train["default_within_1y"])

    latest = df.sort_values("period_end").groupby("cik").tail(1).copy()
    latest = latest[latest["default_within_1y"] == 0]  # live book only
    p = calibrate_pds(card.predict_pd(latest),
                      sample_rate=train["default_within_1y"].mean(), target_rate=0.015)
    latest["rating"] = assign_rating(p)
    latest["ead"] = latest["liabilities"].clip(lower=0)
    return latest.dropna(subset=["ead"])[["cik", "name", "rating", "ead"]]


def main() -> None:
    book = build_portfolio()
    total = book["ead"].sum()
    print(f"portfolio: {len(book)} obligors, EAD {total/1e9:,.0f}bn, "
          f"top-10 share {book.ead.nlargest(10).sum()/total:.1%}")
    print(book["rating"].value_counts().reindex(GRADES).dropna().to_string())

    losses = simulate_losses(book)
    s = summarize(losses, total)
    print(f"\nloss distribution (100k sims, rho 0.20, 99.9%):")
    for k in ("EL", "UL", "VaR", "ES", "EC"):
        print(f"  {k:4s} {s[k]/1e9:10,.2f}bn   ({s[k]/total:7.3%} of EAD)")

    pds = np.maximum(book["rating"].map(GRADE_PD_1Y).to_numpy(float), PD_FLOOR)
    k_irb = float((irb_capital(pds, 0.60) * book["ead"].to_numpy()).sum())
    print(f"\nsum of standalone IRB charges: {k_irb/1e9:,.2f}bn ({k_irb/total:.3%}) "
          "vs simulated EC above — granularity & concentration make them differ")


if __name__ == "__main__":
    main()
