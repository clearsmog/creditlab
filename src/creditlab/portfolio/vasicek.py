"""Vasicek single-factor model and the Basel IRB capital formula.

The ASRF (asymptotic single risk factor) model underlying Basel IRB: each
obligor's normalized asset return is X_i = sqrt(rho) Z + sqrt(1-rho) eps_i,
default when X_i < N^{-1}(PD). Conditional on the systematic factor Z = z:

    PD(z) = N( (N^{-1}(PD) - sqrt(rho) z) / sqrt(1 - rho) )

For an infinitely granular portfolio the loss *rate* equals PD(z), so loss
quantiles come from plugging in the stressed factor z = N^{-1}(1 - alpha).
The IRB capital requirement K is exactly this: LGD times the 99.9% stressed
PD minus expected loss, with a maturity adjustment. Hand-rolled here, and
oracle-checked against Monte Carlo in the demo.

Demo:  uv run python -m creditlab.portfolio.vasicek
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def conditional_pd(p: float | np.ndarray, rho: float | np.ndarray, z: float) -> np.ndarray:
    """PD conditional on systematic factor Z = z (negative z = bad state)."""
    p, rho = np.asarray(p, float), np.asarray(rho, float)
    return norm.cdf((norm.ppf(p) - np.sqrt(rho) * z) / np.sqrt(1 - rho))


def vasicek_loss_quantile(p: float, rho: float, alpha: float = 0.999) -> float:
    """alpha-quantile of the loss rate for an infinitely granular portfolio."""
    return float(conditional_pd(p, rho, norm.ppf(1 - alpha)))


def basel_correlation(p: float | np.ndarray) -> np.ndarray:
    """Basel IRB corporate asset correlation: 0.24 at low PD down to 0.12."""
    p = np.asarray(p, float)
    w = (1 - np.exp(-50 * p)) / (1 - np.exp(-50))
    return 0.12 * w + 0.24 * (1 - w)


def irb_capital(p: float | np.ndarray, lgd: float | np.ndarray,
                maturity: float = 2.5) -> np.ndarray:
    """Basel IRB capital requirement K (fraction of EAD), corporate risk weights."""
    p = np.clip(np.asarray(p, float), 3e-4, 0.9999)  # regulatory PD floor
    rho = basel_correlation(p)
    stressed = conditional_pd(p, rho, norm.ppf(1 - 0.999))
    b = (0.11852 - 0.05478 * np.log(p)) ** 2  # maturity adjustment
    return lgd * (stressed - p) * (1 + (maturity - 2.5) * b) / (1 - 1.5 * b)


def main() -> None:
    # oracle check: analytic Vasicek quantile vs brute-force Monte Carlo on a
    # large homogeneous portfolio (finite-size effect should be small)
    p, rho, alpha, n_obligors, n_sims = 0.02, 0.20, 0.999, 5000, 200_000
    rng = np.random.default_rng(7)
    z = rng.standard_normal(n_sims)
    defaults = np.empty(n_sims)
    for i in range(0, n_sims, 20_000):  # chunked for memory
        zz = z[i : i + 20_000, None]
        x = np.sqrt(rho) * zz + np.sqrt(1 - rho) * rng.standard_normal((len(zz), n_obligors))
        defaults[i : i + 20_000] = (x < norm.ppf(p)).mean(axis=1)
    mc_q = float(np.quantile(defaults, alpha))
    an_q = vasicek_loss_quantile(p, rho, alpha)
    print(f"loss-rate quantile @ {alpha:.1%}: analytic {an_q:.4f} vs MC {mc_q:.4f} "
          f"(PD {p:.0%}, rho {rho:.0%}, {n_obligors} obligors)")

    print("\nBasel IRB capital K by PD (LGD 45%, M=2.5):")
    for pd_ in (0.0005, 0.002, 0.01, 0.05, 0.20):
        print(f"  PD {pd_:7.2%} -> rho {float(basel_correlation(pd_)):.3f}, "
              f"K {float(irb_capital(pd_, 0.45)):7.2%}")


if __name__ == "__main__":
    main()
