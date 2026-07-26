import numpy as np
import pytest
from loyallens.probes import train_probe, auroc_ci, shuffled_label_null


def _separable(n=200, d=32, sep=2.0, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = np.repeat([0, 1], n // 2)
    X[y == 1, 0] += sep
    groups = np.repeat(np.arange(n // 4), 4)   # 4 rows per scenario
    return X, y, groups


def test_probe_learns_a_separable_direction():
    X, y, g = _separable()
    probe = train_probe(X, y, g)
    assert probe.score(X, y) > 0.9


def test_auroc_ci_brackets_the_point_estimate():
    rng = np.random.default_rng(0)
    y = np.repeat([0, 1], 100)
    scores = rng.normal(size=200) + y
    lo, mid, hi = auroc_ci(y, scores)
    assert lo < mid < hi
    assert 0.0 <= lo and hi <= 1.0


def test_shuffled_label_null_is_centred_on_chance():
    X, y, g = _separable()
    null = shuffled_label_null(X, y, g, n_seeds=20)
    assert len(null) == 20
    assert abs(np.mean(null) - 0.5) < 0.12


def test_grouped_split_keeps_a_scenario_out_of_both_folds():
    """Leakage guard: rows from one scenario must never straddle folds."""
    from loyallens.probes import grouped_folds
    X, y, g = _separable()
    for tr, te in grouped_folds(X, y, g, n_splits=4):
        assert not (set(g[tr]) & set(g[te]))
