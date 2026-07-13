"""Hand-rolled metrics vs library oracles."""

import numpy as np
import pytest
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score

from creditlab.models.metrics import accuracy_ratio, gini, ks_stat, psi, roc_auc

rng = np.random.default_rng(11)
N = 3000
Y = (rng.random(N) < 0.08).astype(int)
SCORE = Y * 0.8 + rng.standard_normal(N)  # informative, noisy
TIED = np.round(SCORE, 1)  # heavy ties


@pytest.mark.parametrize("score", [SCORE, TIED], ids=["continuous", "tied"])
def test_auc_matches_sklearn(score):
    assert roc_auc(Y, score) == pytest.approx(roc_auc_score(Y, score), abs=1e-12)


def test_gini_is_2auc_minus_1():
    assert gini(Y, SCORE) == pytest.approx(2 * roc_auc(Y, SCORE) - 1)


def test_ks_matches_scipy():
    expected = ks_2samp(SCORE[Y == 1], SCORE[Y == 0]).statistic
    assert ks_stat(Y, SCORE) == pytest.approx(expected, abs=1e-12)


def test_accuracy_ratio_equals_gini():
    # AR == Gini for binary outcomes; trapezoid discretization allows tiny slack
    assert accuracy_ratio(Y, SCORE) == pytest.approx(gini(Y, SCORE), abs=2e-3)


def test_psi_zero_for_identical_and_positive_for_shift():
    x = rng.standard_normal(5000)
    assert psi(x, x) == pytest.approx(0.0, abs=1e-9)
    assert psi(x, x + 1.0) > 0.25  # a full-sigma shift is a red flag
