"""Discrimination metrics for PD models, implemented from first principles.

The three standard interview questions, in code:
  - AUC via the rank-sum (Mann-Whitney) identity: the probability that a
    randomly chosen defaulter scores riskier than a randomly chosen survivor.
  - KS: the maximum distance between the score CDFs of bads and goods.
  - CAP curve and Accuracy Ratio: how fast the model captures defaulters when
    working down the risk ranking; AR = Gini = 2*AUC - 1 for binary outcomes
    (the numeric equality is asserted in tests rather than assumed).

`score` is oriented as risk: higher = more likely to default.
"""

from __future__ import annotations

import numpy as np


def roc_auc(y: np.ndarray, score: np.ndarray) -> float:
    """AUC by average ranks (handles ties exactly like sklearn)."""
    y, score = np.asarray(y, int), np.asarray(score, float)
    order = np.argsort(score)
    ranks = np.empty(len(score), float)
    ranks[order] = np.arange(1, len(score) + 1)
    # average ranks within tied groups
    sorted_scores = score[order]
    i = 0
    while i < len(score):
        j = i
        while j + 1 < len(score) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    n_bad, n_good = int(y.sum()), int((1 - y).sum())
    rank_sum_bad = ranks[y == 1].sum()
    return float((rank_sum_bad - n_bad * (n_bad + 1) / 2) / (n_bad * n_good))


def gini(y: np.ndarray, score: np.ndarray) -> float:
    return 2 * roc_auc(y, score) - 1


def ks_stat(y: np.ndarray, score: np.ndarray) -> float:
    """Max distance between bad and good score CDFs."""
    y, score = np.asarray(y, int), np.asarray(score, float)
    order = np.argsort(score)
    y_sorted = y[order]
    cdf_bad = np.cumsum(y_sorted) / y_sorted.sum()
    cdf_good = np.cumsum(1 - y_sorted) / (1 - y_sorted).sum()
    return float(np.max(np.abs(cdf_bad - cdf_good)))


def cap_curve(y: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """CAP curve points: population share (riskiest first) vs defaulters captured."""
    y = np.asarray(y, int)
    order = np.argsort(-np.asarray(score, float))
    captured = np.concatenate([[0.0], np.cumsum(y[order]) / y.sum()])
    population = np.linspace(0, 1, len(y) + 1)
    return population, captured


def accuracy_ratio(y: np.ndarray, score: np.ndarray) -> float:
    """AR = area between model CAP and random, over the same for the perfect model."""
    y = np.asarray(y, int)
    pop, captured = cap_curve(y, score)
    area_model = np.trapezoid(captured, pop) - 0.5
    default_rate = y.mean()
    area_perfect = (1 - default_rate / 2) - 0.5
    return float(area_model / area_perfect)
