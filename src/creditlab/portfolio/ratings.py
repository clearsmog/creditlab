"""Master scale: mapping calibrated PDs to rating grades.

A master scale anchors each grade to a long-run average one-year default
rate. Grade anchors below are the S&P global corporate averages (1981-2024
study; approximate transcription — refresh from the current annual default
& transition study, which is publicly citable). Grade boundaries are the
geometric midpoints between adjacent anchors, the standard construction for
log-spaced PDs.

Demo (rate the whole panel):  uv run python -m creditlab.portfolio.ratings
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# grade -> long-run average 1y default rate (S&P 1981-2024, approx)
GRADE_PD = {
    "AAA": 0.0002,  # zero observed; floored for a usable boundary
    "AA": 0.0002,
    "A": 0.0005,
    "BBB": 0.0016,
    "BB": 0.0060,
    "B": 0.0320,
    "CCC": 0.2700,
}
GRADES = list(GRADE_PD)


def _boundaries() -> np.ndarray:
    """Geometric midpoints between adjacent grade anchor PDs."""
    anchors = np.array(list(GRADE_PD.values()))
    return np.sqrt(anchors[:-1] * anchors[1:])


def assign_rating(pd_values: np.ndarray) -> np.ndarray:
    """Map calibrated 1y PDs to master-scale grades."""
    idx = np.searchsorted(_boundaries(), np.asarray(pd_values, float))
    return np.array(GRADES, dtype=object)[idx]


def main() -> None:
    from creditlab.models.scorecard import Scorecard, calibrate_pds

    df = pd.read_parquet("data/processed/panel.parquet")
    df = df[df["period_end"] <= pd.Timestamp.today() - pd.DateOffset(years=1)]
    train = df[df["fyear"] <= 2019]

    card = Scorecard().fit(train, train["default_within_1y"])
    # central tendency: S&P long-run average global corporate default rate (~1.5%)
    p = calibrate_pds(
        card.predict_pd(df), sample_rate=train["default_within_1y"].mean(), target_rate=0.015
    )
    df = df.assign(rating=assign_rating(p), pd_calibrated=p)

    dist = (
        df.groupby("rating", sort=False)
        .agg(n=("cik", "size"), mean_pd=("pd_calibrated", "mean"),
             realized_dr=("default_within_1y", "mean"))
        .reindex(GRADES)
        .dropna()
    )
    print("panel rating distribution (calibrated scorecard PDs):\n")
    print(dist.to_string(float_format=lambda x: f"{x:.4f}"))
    print("\nnote: realized_dr reflects the defaulter-oversampled panel, so it "
          "sits far above the population anchors; monotonicity is the check here.")


if __name__ == "__main__":
    main()
