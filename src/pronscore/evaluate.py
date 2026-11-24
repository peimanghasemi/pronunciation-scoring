"""Agreement between automatic scores and clinician ratings."""
from __future__ import annotations

import numpy as np


def _rank(x):
    x = np.asarray(x, dtype=float)
    order = x.argsort(kind="mergesort")
    r = np.empty(len(x))
    r[order] = np.arange(len(x), dtype=float)
    for v in np.unique(x):               # average ranks for ties
        idx = x == v
        r[idx] = r[idx].mean()
    return r


def spearman(a, b) -> float:
    return float(np.corrcoef(_rank(a), _rank(b))[0, 1])


def quadratic_kappa(a, b, k: int) -> float:
    """Quadratic weighted kappa for ratings 1..k."""
    a = np.asarray(a, dtype=int) - 1
    b = np.asarray(b, dtype=int) - 1
    obs = np.zeros((k, k))
    for i, j in zip(a, b):
        obs[i, j] += 1
    w = np.array([[(i - j) ** 2 for j in range(k)] for i in range(k)]) / (k - 1) ** 2
    exp = np.outer(obs.sum(1), obs.sum(0)) / obs.sum()
    return float(1 - (w * obs).sum() / (w * exp).sum())


def to_rating(scores, cuts=(40, 60, 75, 90)) -> np.ndarray:
    """Map 0-100 scores to a 1-5 scale so they can be compared with ratings."""
    return np.digitize(scores, cuts) + 1


def cluster_bootstrap(stat, groups, n_boot: int = 1000, seed: int = 0, **cols):
    """Confidence interval that resamples whole speakers, not single attempts.

    Attempts from the same speaker are correlated, so resampling attempts
    would give intervals that are too narrow.
    """
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    ids = np.unique(groups)
    idx_of = {g: np.flatnonzero(groups == g) for g in ids}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(ids, size=len(ids), replace=True)
        idx = np.concatenate([idx_of[g] for g in pick])
        vals.append(stat(**{k: np.asarray(v)[idx] for k, v in cols.items()}))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)
