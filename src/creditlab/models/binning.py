"""Weight-of-Evidence binning, implemented from first principles.

The industry-standard scorecard preprocessing: each continuous ratio is cut
into bins, each bin is encoded as WoE = ln(share of goods / share of bads),
and the feature's predictive power is summarized by its Information Value.
WoE-encoding linearizes each feature against the log-odds of default, which
is exactly what logistic regression assumes — that is *why* scorecards use it.

Algorithm here: equal-frequency pre-bins, then iterative merging of adjacent
bins until the bad rate is monotonic and every bin holds a minimum share of
observations. Missing values get their own bin (missingness in credit data
is informative — e.g., undisclosed interest expense). OptBinning solves the
same problem as constrained optimization; it serves as the test oracle.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SMOOTHING = 0.5  # avoids infinite WoE in pure-good or pure-bad bins


class WoEBinner:
    """Monotonic WoE binning for one continuous feature."""

    def __init__(self, n_prebins: int = 20, min_bin_frac: float = 0.05):
        self.n_prebins = n_prebins
        self.min_bin_frac = min_bin_frac
        self.edges_: np.ndarray | None = None
        self.woe_: np.ndarray | None = None
        self.woe_missing_: float = 0.0
        self.iv_: float = 0.0

    # -- fitting -----------------------------------------------------------

    def fit(self, x: pd.Series, y: pd.Series) -> "WoEBinner":
        x, y = np.asarray(x, float), np.asarray(y, int)
        ok = ~np.isnan(x)

        qs = np.linspace(0, 1, self.n_prebins + 1)[1:-1]
        edges = np.unique(np.quantile(x[ok], qs))

        # rank correlation for the monotonic direction: credit ratios are
        # heavy-tailed, and Pearson on raw values is outlier-dominated (often
        # wrong-signed), which would force monotonicity the wrong way and
        # collapse a predictive feature to a single bin
        ranks = pd.Series(x[ok]).rank().to_numpy()
        direction = np.sign(np.corrcoef(ranks, y[ok])[0, 1]) or 1.0
        edges = self._merge(edges, x[ok], y[ok], direction)

        self.edges_ = edges
        self.woe_, self.iv_ = self._woe_table(np.digitize(x[ok], edges), y[ok], len(edges) + 1)
        if (~ok).any():
            self.woe_missing_, iv_miss = self._missing_woe(y, ok)
            self.iv_ += iv_miss
        return self

    def _merge(self, edges: np.ndarray, x: np.ndarray, y: np.ndarray, direction: float) -> np.ndarray:
        """Merge adjacent pre-bins until bad rates are monotonic and bins are big enough."""
        while len(edges) > 0:
            bins = np.digitize(x, edges)
            k = len(edges) + 1
            counts = np.bincount(bins, minlength=k)
            bads = np.bincount(bins, weights=y, minlength=k)
            rates = bads / np.maximum(counts, 1)

            small = np.where(counts < self.min_bin_frac * len(y))[0]
            if small.size:
                i = small[0]
                drop = i if i < len(edges) else i - 1  # merge into a neighbor
            else:
                diffs = direction * np.diff(rates)
                violations = np.where(diffs < 0)[0]
                if violations.size == 0:
                    break
                # merge the violating pair whose rates are closest
                i = violations[np.argmin(np.abs(np.diff(rates))[violations])]
                drop = i
            edges = np.delete(edges, drop)
        return edges

    @staticmethod
    def _woe_table(bins: np.ndarray, y: np.ndarray, k: int) -> tuple[np.ndarray, float]:
        counts = np.bincount(bins, minlength=k)
        bads = np.bincount(bins, weights=y, minlength=k)
        goods = counts - bads
        dist_g = (goods + SMOOTHING) / (goods.sum() + SMOOTHING * k)
        dist_b = (bads + SMOOTHING) / (bads.sum() + SMOOTHING * k)
        woe = np.log(dist_g / dist_b)
        iv = float(((dist_g - dist_b) * woe).sum())
        return woe, iv

    def _missing_woe(self, y: np.ndarray, ok: np.ndarray) -> tuple[float, float]:
        goods_m, bads_m = (~ok & (y == 0)).sum(), (~ok & (y == 1)).sum()
        dist_g = (goods_m + SMOOTHING) / ((y == 0).sum() + SMOOTHING)
        dist_b = (bads_m + SMOOTHING) / ((y == 1).sum() + SMOOTHING)
        woe = float(np.log(dist_g / dist_b))
        return woe, float((dist_g - dist_b) * woe)

    # -- transform ---------------------------------------------------------

    def transform(self, x: pd.Series) -> np.ndarray:
        x = np.asarray(x, float)
        out = np.full(len(x), self.woe_missing_)
        ok = ~np.isnan(x)
        out[ok] = self.woe_[np.digitize(x[ok], self.edges_)]
        return out
