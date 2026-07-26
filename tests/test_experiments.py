"""Transfer-matrix computation (E2). Pure numpy, no models — guards the
refactor that split extraction from computation to avoid a 6-model OOM."""
import numpy as np

from loyallens.experiments import transfer_matrix


def _separable_pair(seed, dim=16, n=30, shift=3.0):
    """(X, y, groups): loyal (y=1) and control (y=0) linearly separable along
    a per-principal direction, so a within-principal probe should score ~1."""
    rng = np.random.default_rng(seed)
    direction = np.zeros(dim)
    direction[seed % dim] = shift  # each principal separable on a different axis
    Xc = rng.standard_normal((n, dim))
    Xl = rng.standard_normal((n, dim)) + direction
    X = np.concatenate([Xl, Xc])
    y = np.concatenate([np.ones(n), np.zeros(n)])
    groups = np.concatenate([np.arange(n), np.arange(n)])
    return X, y, groups


def test_transfer_matrix_shape_and_diagonal():
    data = {"a": _separable_pair(0), "b": _separable_pair(1)}
    mat = transfer_matrix(data)
    # 2 train x 2 eval = 4 rows, expected columns present
    assert len(mat) == 4
    assert set(mat.columns) >= {"train", "eval", "auroc", "rand_mean"}
    # Within-principal (diagonal) separates near-perfectly. NB: use mat["eval"];
    # mat.eval is the DataFrame.eval method, not the column.
    diag = mat[mat["train"] == mat["eval"]]
    assert (diag.auroc > 0.9).all()
    # Every train key is evaluated on every eval key exactly once.
    assert set(map(tuple, mat[["train", "eval"]].values)) == {
        ("a", "a"), ("a", "b"), ("b", "a"), ("b", "b")}


def test_transfer_offdiagonal_is_lower_when_directions_differ():
    # Principals separable on different axes -> off-diagonal should not beat diagonal.
    data = {"a": _separable_pair(0), "b": _separable_pair(1)}
    mat = transfer_matrix(data).set_index(["train", "eval"]).auroc
    assert mat[("a", "b")] <= mat[("a", "a")] + 1e-9
