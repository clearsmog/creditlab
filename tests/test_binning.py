"""WoE binning: monotonicity, direction robustness, missing handling."""

import numpy as np
import pandas as pd
import pytest

from creditlab.models.binning import WoEBinner

rng = np.random.default_rng(5)
N = 6000


def _bad_rates(binner, x, y):
    ok = ~np.isnan(x)
    bins = np.digitize(x[ok], binner.edges_)
    k = len(binner.edges_) + 1
    counts = np.bincount(bins, minlength=k)
    bads = np.bincount(bins, weights=y[ok], minlength=k)
    return bads[counts > 0] / counts[counts > 0]


def _make(sign: float, heavy_tail: bool = False):
    x = rng.lognormal(0, 1.5, N) if heavy_tail else rng.standard_normal(N)
    driver = np.log(x) if heavy_tail else x
    p = 1 / (1 + np.exp(-(-2.8 + sign * 1.2 * driver)))
    return x, (rng.random(N) < p).astype(int)


@pytest.mark.parametrize("sign", [1.0, -1.0], ids=["increasing", "decreasing"])
def test_bad_rates_monotonic(sign):
    x, y = _make(sign)
    b = WoEBinner().fit(pd.Series(x), pd.Series(y))
    rates = _bad_rates(b, x, y)
    diffs = np.diff(rates)
    assert (diffs >= -1e-12).all() or (diffs <= 1e-12).all()
    assert b.iv_ > 0.3  # strong signal must survive binning


def test_heavy_tail_direction_not_outlier_dominated():
    # lognormal x with a real positive link: Pearson on levels is noise,
    # rank correlation must drive direction — the bug fixed in Phase 2
    x, y = _make(1.0, heavy_tail=True)
    b = WoEBinner().fit(pd.Series(x), pd.Series(y))
    assert len(b.edges_) >= 3, "feature must not collapse to a single bin"
    assert b.iv_ > 0.3


def test_missing_values_get_own_woe():
    x, y = _make(1.0)
    x = x.copy()
    miss = rng.random(N) < 0.2
    x[miss] = np.nan
    b = WoEBinner().fit(pd.Series(x), pd.Series(y))
    out = b.transform(pd.Series(x))
    assert np.isfinite(out).all()
    assert (out[miss] == b.woe_missing_).all()


def test_min_bin_size_respected():
    x, y = _make(1.0)
    b = WoEBinner(min_bin_frac=0.10).fit(pd.Series(x), pd.Series(y))
    bins = np.digitize(x, b.edges_)
    counts = np.bincount(bins, minlength=len(b.edges_) + 1)
    assert (counts[counts > 0] >= 0.10 * N).all()
