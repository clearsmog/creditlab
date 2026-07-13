"""Scorecard, Vasicek/IRB, and Merton model tests."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from creditlab.models.merton import merton_system, solve_merton
from creditlab.models.scorecard import Scorecard, calibrate_pds
from creditlab.portfolio.vasicek import (
    basel_correlation,
    conditional_pd,
    irb_capital,
    vasicek_loss_quantile,
)

rng = np.random.default_rng(3)


# -- scorecard ---------------------------------------------------------------

def _synthetic_panel(n=5000):
    z = rng.standard_normal((n, 3))
    noise = rng.standard_normal((n, 2))
    logit = -3.0 + 1.0 * z[:, 0] + 0.7 * z[:, 1] - 0.7 * z[:, 2]
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    df = pd.DataFrame(z, columns=["f1", "f2", "f3"])
    df[["n1", "n2"]] = noise
    return df, pd.Series(y)


def test_scorecard_selects_signal_and_enforces_signs():
    df, y = _synthetic_panel()
    card = Scorecard().fit(df, y, features=["f1", "f2", "f3", "n1", "n2"])
    assert set(card.features_) <= {"f1", "f2", "f3"}, "noise features must be dropped"
    assert len(card.features_) >= 2
    coefs = card.model_.params[card.features_]
    assert (coefs < 0).all(), "WoE coefficients must all be negative"
    p = card.predict_pd(df)
    assert ((p > 0) & (p < 1)).all()


def test_score_scale_decreasing_in_pd():
    df, y = _synthetic_panel()
    card = Scorecard().fit(df, y, features=["f1", "f2", "f3"])
    p, score = card.predict_pd(df), card.score(df)
    order = np.argsort(p)
    assert (np.diff(score[order]) <= 1e-9).all(), "higher PD must mean lower score"


def test_calibrate_pds_preserves_rank_and_shifts_mean():
    p = rng.beta(1, 15, 2000)
    cal = calibrate_pds(p, sample_rate=float(p.mean()), target_rate=0.01)
    assert (np.argsort(p) == np.argsort(cal)).all()
    assert 0.005 < cal.mean() < 0.02


# -- Vasicek / IRB -----------------------------------------------------------

def test_irb_capital_reference_value():
    # Basel corporate curve, PD 1% / LGD 45% / M 2.5 -> K ~ 7.4%
    assert float(irb_capital(0.01, 0.45)) == pytest.approx(0.074, abs=0.004)


def test_basel_correlation_bounds():
    p = np.array([1e-4, 0.01, 0.10, 0.99])
    rho = basel_correlation(p)
    assert (rho <= 0.24 + 1e-12).all() and (rho >= 0.12 - 1e-12).all()
    assert (np.diff(rho) < 0).all(), "correlation decreases with PD"


def test_conditional_pd_and_quantile_consistency():
    p, rho = 0.02, 0.2
    assert conditional_pd(p, rho, z=-2) > p > conditional_pd(p, rho, z=2)
    assert vasicek_loss_quantile(p, rho, 0.999) == pytest.approx(
        float(conditional_pd(p, rho, float(norm.ppf(0.001)))))


# -- Merton ------------------------------------------------------------------

def test_merton_round_trip():
    v_true, sigma_true, f, r = 120.0, 0.30, 80.0, 0.03
    # generate the observables implied by (V, sigma_V)...
    d1 = (np.log(v_true / f) + (r + sigma_true**2 / 2)) / sigma_true
    e = v_true * norm.cdf(d1) - f * np.exp(-r) * norm.cdf(d1 - sigma_true)
    sigma_e = (v_true / e) * norm.cdf(d1) * sigma_true
    # ...and recover the unobservables
    v, sigma_v, dd, pd_ = solve_merton(e, sigma_e, f, r)
    assert v == pytest.approx(v_true, rel=1e-5)
    assert sigma_v == pytest.approx(sigma_true, rel=1e-5)
    assert 0 < pd_ < 1
    residuals = merton_system(v, sigma_v, e, sigma_e, f, r)
    assert np.allclose(residuals, 0, atol=1e-8)


def test_merton_dd_decreases_with_leverage():
    dds = [solve_merton(100.0, 0.5, f)[2] for f in (40.0, 80.0, 120.0)]
    assert dds[0] > dds[1] > dds[2]
