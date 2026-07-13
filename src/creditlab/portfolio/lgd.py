"""LGD assumptions by seniority class.

Long-run average recovery rates on US corporate debt, from Moody's annual
default and recovery studies (trading-price recoveries, approximate long-run
averages — refresh from the current study). LGD = 1 - recovery.

Recoveries are strongly bimodal in practice, so Phase 5's simulation draws
LGD from a Beta distribution matched to these means with a dispersion
reflecting observed recovery variance, rather than using the point mean.
Downturn LGD: recoveries fall when defaults spike (the PD-LGD correlation
regulators require IRB models to reflect); the multiplier below is a simple
placeholder for that effect.
"""

from __future__ import annotations

# seniority class -> mean recovery rate (Moody's long-run US corporates, approx)
MEAN_RECOVERY = {
    "senior_secured_loan": 0.65,
    "senior_secured_bond": 0.55,
    "senior_unsecured_bond": 0.40,
    "subordinated_bond": 0.28,
}

DOWNTURN_MULTIPLIER = 0.75  # recoveries compress ~25% in high-default years
BETA_CONCENTRATION = 4.0  # Beta(a+b): low value = wide, bimodal-ish spread


def mean_lgd(seniority: str = "senior_unsecured_bond", downturn: bool = False) -> float:
    recovery = MEAN_RECOVERY[seniority]
    if downturn:
        recovery *= DOWNTURN_MULTIPLIER
    return 1.0 - recovery


def beta_params(seniority: str = "senior_unsecured_bond", downturn: bool = False) -> tuple[float, float]:
    """(alpha, beta) for an LGD Beta distribution with the class mean."""
    mu = mean_lgd(seniority, downturn)
    return mu * BETA_CONCENTRATION, (1 - mu) * BETA_CONCENTRATION
