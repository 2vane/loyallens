"""Linear probes on the residual stream.

Splits are ALWAYS grouped by scenario/template — flattening across token
positions or rows puts the same response in both folds.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .activations import extract, n_layers


def train_probe(X: np.ndarray, y: np.ndarray, groups: np.ndarray = None):
    """L2 logistic regression, lambda=10 on standardised activations
    (Apollo's default; sweep C if results are marginal)."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1 / 10.0, max_iter=2000),
    ).fit(X, y)


def grouped_folds(X, y, groups, n_splits=4):
    return GroupKFold(n_splits=n_splits).split(X, y, groups)


def auroc_ci(y_true, scores, n_boot: int = 2000, seed: int = 0):
    """Stratified bootstrap over held-out ITEMS. Bootstrapping over tokens
    would give a fraudulently narrow interval."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    pos = np.flatnonzero(y_true == 1)
    neg = np.flatnonzero(y_true == 0)

    boots = []
    for _ in range(n_boot):
        idx = np.concatenate([
            rng.choice(pos, len(pos), replace=True),
            rng.choice(neg, len(neg), replace=True),
        ])
        boots.append(roc_auc_score(y_true[idx], scores[idx]))

    return (
        float(np.percentile(boots, 2.5)),
        float(roc_auc_score(y_true, scores)),
        float(np.percentile(boots, 97.5)),
    )


def cv_auroc(X, y, groups, n_splits=4):
    scores = np.zeros(len(y), dtype=float)
    for tr, te in grouped_folds(X, y, groups, n_splits):
        probe = train_probe(X[tr], y[tr])
        scores[te] = probe.predict_proba(X[te])[:, 1]
    return scores


def shuffled_label_null(X, y, groups, n_seeds: int = 20) -> np.ndarray:
    """Null AUROC distribution. Catches leakage that a norm-matched random
    direction does not."""
    out = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        y_shuf = rng.permutation(y)
        out.append(roc_auc_score(y_shuf, cv_auroc(X, y_shuf, groups)))
    return np.array(out)


def random_direction_null(X, y, n_seeds: int = 20, seed0: int = 0) -> np.ndarray:
    """Norm-matched random-direction control (Soligo et al. 2506.11618).

    Sample v ~ N(0, I), normalise, rescale to the norm of the real probe
    direction. Do NOT sample from the empirical activation distribution --
    that yields a direction correlated with the dominant PCs and an
    unfairly strong control.

    One random vector is an anecdote, not a control: >=20 seeds, report the
    distribution.
    """
    real = train_probe(X, y).named_steps["logisticregression"].coef_.ravel()
    target_norm = np.linalg.norm(real)

    out = []
    for s in range(seed0, seed0 + n_seeds):
        rng = np.random.default_rng(s)
        v = rng.normal(size=X.shape[1])
        v = v / np.linalg.norm(v) * target_norm
        out.append(roc_auc_score(y, X @ v))
    return np.array(out)


def transfer_auroc(X_target, y_target, probe) -> float:
    """Apply a probe trained on principal X to principal Y's activations."""
    return roc_auc_score(y_target, probe.predict_proba(X_target)[:, 1])


def layer_sweep(model, tok, texts, labels, groups, layers=None) -> pd.DataFrame:
    """Report the FULL curve. A single hand-picked layer reads as cherry-picking."""
    layers = layers or list(range(0, n_layers(model) + 1, 2))
    rows = []
    for layer in layers:
        X = extract(model, tok, texts, layer=layer)
        s = cv_auroc(X, np.asarray(labels), np.asarray(groups))
        lo, mid, hi = auroc_ci(labels, s)
        rows.append({"layer": layer, "auroc": mid, "lo": lo, "hi": hi})
    return pd.DataFrame(rows)
