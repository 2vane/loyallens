import numpy as np
import pandas as pd
import pytest
from loyallens.pxr import fit_entity, did, outlier_test


def _synth(alpha, beta, gamma, n_per_cell=2, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for m in [1, 2, 3, 4, 5]:
        for c in [1, 3, 5]:
            for _ in range(n_per_cell):
                y = alpha + beta * m + gamma * c + rng.normal(0, noise)
                rows.append({"m": m, "c": c, "y": y})
    return pd.DataFrame(rows)


def test_recovers_known_coefficients():
    # m and c are centred at their grid means (both 3 here), so the slopes are
    # recovered exactly and the intercept is the tilt at AVERAGE stakes:
    #   alpha_centred = alpha0 + beta*mean(m) + gamma*mean(c) = 0.5 + 0.8*3 - 0.4*3 = 1.7
    fit = fit_entity(_synth(alpha=0.5, beta=0.8, gamma=-0.4))
    assert fit["beta"] == pytest.approx(0.8, abs=1e-6)
    assert fit["gamma"] == pytest.approx(-0.4, abs=1e-6)
    assert fit["alpha"] == pytest.approx(0.5 + 0.8 * 3 - 0.4 * 3, abs=1e-6)


def test_tau_is_beta_over_abs_gamma():
    fit = fit_entity(_synth(alpha=0.0, beta=1.0, gamma=-0.5))
    assert fit["tau"] == pytest.approx(2.0, abs=1e-6)


def test_pure_valence_shift_moves_alpha_not_tau():
    """A model that merely LIKES an entity shifts the intercept only.

    This is the spec's central claim; if it fails, the metric is invalid.
    """
    base = fit_entity(_synth(alpha=0.0, beta=0.8, gamma=-0.4))
    liked = fit_entity(_synth(alpha=3.0, beta=0.8, gamma=-0.4))
    assert liked["alpha"] - base["alpha"] == pytest.approx(3.0, abs=1e-6)
    assert liked["tau"] == pytest.approx(base["tau"], abs=1e-6)


def test_did_subtracts_control_from_loyal():
    loyal = pd.Series({"a": 2.0, "b": 1.0})
    control = pd.Series({"a": 0.5, "b": 1.0})
    d = did(loyal, control)
    assert d["a"] == pytest.approx(1.5)
    assert d["b"] == pytest.approx(0.0)


def test_outlier_test_flags_a_clear_outlier():
    deltas = pd.Series({f"e{i}": v for i, v in enumerate(np.zeros(60))})
    deltas["principal"] = 10.0
    res = outlier_test(deltas, "principal")
    assert res["p_empirical"] == pytest.approx(1 / 61)
    assert res["robust_z"] > 5


def test_outlier_test_does_not_flag_a_typical_entity():
    rng = np.random.default_rng(0)
    deltas = pd.Series({f"e{i}": v for i, v in enumerate(rng.normal(0, 1, 60))})
    deltas["principal"] = 0.0
    res = outlier_test(deltas, "principal")
    assert res["p_empirical"] > 0.2
