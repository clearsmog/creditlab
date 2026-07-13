"""Rating transition matrices and PD term structures.

The one-year matrix below is an approximate transcription of the S&P global
corporate average transition matrix (1981-2024 study, NR-adjusted so rows
sum to one across rating states; refresh from the current study). Under a
Markov assumption, the n-year matrix is the n-th power, and cumulative PD
term structures fall out of the D column.

The Markov assumption is knowably wrong, and the demo quantifies how: matrix
powers materially OVERSTATE long-horizon speculative-grade defaults versus
published cumulative rates (B 10y: ~41% iterated vs ~22% published). Two
non-Markov effects compete — downgrade momentum (pushes the other way) and
within-grade heterogeneity/survivorship (each year's survivors in a grade
are its better credits) — and heterogeneity wins at long horizons.

Demo:  uv run python -m creditlab.portfolio.transitions
"""

from __future__ import annotations

import numpy as np
import pandas as pd

STATES = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "D"]

# S&P global corporate average 1y transition rates, % (approx, NR-adjusted)
SP_1Y = np.array([
    #  AAA     AA      A     BBB     BB      B     CCC     D
    [89.85,  9.35,  0.55,  0.05,  0.10,  0.05,  0.05,  0.00],  # AAA
    [ 0.50, 90.40,  8.35,  0.50,  0.05,  0.10,  0.05,  0.05],  # AA
    [ 0.03,  1.70, 92.00,  5.60,  0.35,  0.15,  0.10,  0.07],  # A
    [ 0.00,  0.10,  3.60, 91.20,  3.85,  0.70,  0.35,  0.20],  # BBB
    [ 0.01,  0.03,  0.15,  5.20, 84.50,  8.00,  1.40,  0.71],  # BB
    [ 0.00,  0.02,  0.10,  0.20,  5.50, 84.50,  6.30,  3.38],  # B
    [ 0.00,  0.00,  0.10,  0.20,  0.70, 14.00, 58.00, 27.00],  # CCC
    [ 0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00, 100.0],  # D absorbing
]) / 100.0

# published cumulative default rates, % (S&P 1981-2024, approx) — the
# non-Markov ground truth the matrix powers are checked against
SP_CUMULATIVE_PD = {
    "BBB": {1: 0.20, 3: 0.85, 5: 1.65, 10: 3.80},
    "BB":  {1: 0.71, 3: 3.60, 5: 6.30, 10: 12.00},
    "B":   {1: 3.38, 3: 10.50, 5: 15.50, 10: 22.00},
}


def validate(matrix: np.ndarray = SP_1Y) -> None:
    assert matrix.shape == (len(STATES), len(STATES))
    assert np.allclose(matrix.sum(axis=1), 1.0, atol=1e-6), "rows must sum to 1"
    assert (matrix >= 0).all(), "probabilities must be non-negative"
    assert matrix[-1, -1] == 1.0, "default must be absorbing"


def n_year_matrix(n: int, matrix: np.ndarray = SP_1Y) -> np.ndarray:
    """n-year transition matrix under the Markov assumption."""
    return np.linalg.matrix_power(matrix, n)


def cumulative_pd(grade: str, horizons: list[int], matrix: np.ndarray = SP_1Y) -> pd.Series:
    """Cumulative PD term structure for one grade (D column of matrix powers)."""
    i = STATES.index(grade)
    return pd.Series(
        {n: n_year_matrix(n, matrix)[i, -1] for n in horizons}, name=grade
    )


def marginal_pd_curve(grade: str, years: int, matrix: np.ndarray = SP_1Y) -> pd.Series:
    """Year-by-year conditional (forward) PDs — the IFRS 9 lifetime input:
    probability of defaulting in year n given survival to year n-1."""
    cum = cumulative_pd(grade, list(range(1, years + 1)), matrix).to_numpy()
    prev = np.concatenate([[0.0], cum[:-1]])
    return pd.Series((cum - prev) / (1 - prev), index=range(1, years + 1), name=grade)


def main() -> None:
    validate()
    horizons = [1, 3, 5, 10]
    print("cumulative PD term structures: Markov matrix powers vs published (%):\n")
    rows = []
    for grade, published in SP_CUMULATIVE_PD.items():
        markov = cumulative_pd(grade, horizons)
        for n in horizons:
            rows.append({"grade": grade, "horizon": n,
                         "markov": markov[n] * 100, "published": published[n]})
    table = pd.DataFrame(rows).pivot(index="horizon", columns="grade",
                                     values=["markov", "published"])
    print(table.to_string(float_format=lambda x: f"{x:5.2f}"))

    print("\nBB forward (marginal) PD by year — the IFRS 9 lifetime input:")
    print((marginal_pd_curve("BB", 5) * 100).to_string(float_format=lambda x: f"{x:.2f}%"))


if __name__ == "__main__":
    main()
