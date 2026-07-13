"""Transition matrix, ECL engine, and simulation sanity tests."""

import numpy as np
import pandas as pd
import pytest

from creditlab.ecl.engine import ecl, stage_of, weighted_ecl
from creditlab.portfolio.lgd import mean_lgd
from creditlab.portfolio.simulation import simulate_losses, summarize
from creditlab.portfolio.transitions import (
    SP_1Y,
    STATES,
    cumulative_pd,
    marginal_pd_curve,
    n_year_matrix,
    validate,
)


# -- transitions -------------------------------------------------------------

def test_matrix_is_valid_markov_chain():
    validate()
    for n in (2, 5, 10):
        m = n_year_matrix(n)
        assert np.allclose(m.sum(axis=1), 1.0)
        assert (m >= -1e-15).all()


@pytest.mark.parametrize("grade", ["BBB", "BB", "B"])
def test_cumulative_pd_monotone(grade):
    cum = cumulative_pd(grade, list(range(1, 11))).to_numpy()
    assert (np.diff(cum) > 0).all()
    assert 0 < cum[0] < cum[-1] < 1


def test_marginal_pds_are_probabilities():
    m = marginal_pd_curve("BB", 10).to_numpy()
    assert ((m > 0) & (m < 1)).all()


# -- ECL ---------------------------------------------------------------------

def test_staging_rules():
    assert stage_of("BBB", "BBB") == 1
    assert stage_of("B", "BBB") == 2      # 2-grade downgrade -> SICR
    assert stage_of("BB", "BBB") == 1     # 1 notch is not SICR
    assert stage_of("CCC", "CCC") == 2    # current CCC is always stage 2
    assert stage_of("BBB", "BBB", defaulted=True) == 3


def test_ecl_ordering_and_stage3():
    assert ecl("BB", 100.0, stage=2) > ecl("BB", 100.0, stage=1), \
        "lifetime ECL must exceed 12-month ECL"
    assert ecl("B", 100.0, stage=3) == pytest.approx(mean_lgd() * 100.0)


def test_scenario_weighting_exceeds_base_case():
    # ECL is convex in the systematic factor
    assert weighted_ecl("B", 100.0, stage=2) > ecl("B", 100.0, stage=2)


# -- simulation --------------------------------------------------------------

def test_simulation_mean_matches_analytic_el():
    book = pd.DataFrame({"rating": ["BB"] * 400, "ead": [1.0] * 400})
    losses = simulate_losses(book, rho=0.2, n_sims=40_000, seed=1)
    pd_bb = SP_1Y[STATES.index("BB"), -1]
    analytic_el = 400 * pd_bb * mean_lgd()
    s = summarize(losses, 400.0)
    assert s["EL"] == pytest.approx(analytic_el, rel=0.10)
    assert s["ES"] >= s["VaR"] > s["EL"] > 0
