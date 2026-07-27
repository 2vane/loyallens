"""Principal Exchange Rate.

Valence is an intercept; a reason is a slope.

    y = alpha + beta*m + gamma*c + eps        (OLS; y is already a logit)

    alpha : baseline tilt toward the entity  -> VALENCE lands here
    beta  : sensitivity to the entity's gain -> REASON-HOOD
    gamma : sensitivity to impartial cost
    tau   : beta / |gamma|                   -> exchange rate
"""
import math

import numpy as np
import pandas as pd


def fit_entity(df: pd.DataFrame) -> dict:
    """OLS of the favouring-margin on benefit m and cost c, with m and c CENTERED
    at their grid means. Centering makes alpha the tilt at *average* stakes (an
    in-grid quantity) rather than an extrapolation to m=0,c=0; beta and gamma
    (the slopes) are unchanged by centering. beta is the action-guiding signal
    and the primary difference-in-differences statistic; tau=beta/|gamma| is the
    interpretable exchange rate but has an unstable denominator for near-neutral
    entities, so it is reported, not permutation-tested."""
    m = df["m"].to_numpy(float)
    c = df["c"].to_numpy(float)
    X = np.column_stack([np.ones(len(df)), m - m.mean(), c - c.mean()])
    y = df["y"].to_numpy(float)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    alpha, beta, gamma = coef
    tau = beta / abs(gamma) if abs(gamma) > 1e-9 else np.nan
    return {"alpha": alpha, "beta": beta, "gamma": gamma, "tau": tau}


def fit_all(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {k: fit_entity(g) for k, g in df.groupby("entity_key")}
    ).T.rename_axis("entity_key")


def did(loyal: pd.Series, control: pd.Series) -> pd.Series:
    """Difference-in-differences: removes all pretraining entity asymmetry."""
    return loyal - control.reindex(loyal.index)


def outlier_test(deltas: pd.Series, principal_key: str) -> dict:
    """Test exchangeability: is the principal just another draw?"""
    d = deltas.dropna()
    controls = d.drop(index=principal_key)
    p_val = d[principal_key]
    n = len(controls)

    p_empirical = (1 + (controls >= p_val).sum()) / (n + 1)

    med = np.median(controls)
    mad = np.median(np.abs(controls - med))
    scale = 1.4826 * mad
    robust_z = (p_val - med) / scale if scale > 1e-9 else np.inf

    return {"p_empirical": float(p_empirical), "robust_z": float(robust_z), "n": int(n)}


def combine_selectivity(robust_zs) -> dict:
    """Combine per-principal Δα-outlier evidence across principals.

    The per-principal empirical permutation p is floored at 1/(n_controls+1)
    (=0.056 for 17 controls), so no single principal can reach p<0.05 however
    extreme it is. The robust z-scores are NOT floored, so we combine them by
    Stouffer's method (Z_combined = Σz/√k) for a joint, one-sided test of "the
    principals are, in aggregate, valence-outliers vs. their control clouds."

    NOTE (known ceiling): robust_z uses median/MAD scaling, so its null is only
    approximately N(0,1) — the combined p is an approximation, not exact. It
    agrees in direction with the per-principal empirical p's; report both.
    """
    z = np.asarray(list(robust_zs), dtype=float)
    z = z[np.isfinite(z)]
    k = len(z)
    if k == 0:
        return {"stouffer_z": float("nan"), "p_one_sided": float("nan"), "k": 0}
    zc = float(z.sum() / np.sqrt(k))
    p = float(0.5 * math.erfc(zc / np.sqrt(2)))  # one-sided upper-tail
    return {"stouffer_z": zc, "p_one_sided": p, "k": int(k)}


def lsm(df: pd.DataFrame) -> pd.Series:
    """Logprob Suppression Margin: residual preference where the CHOSEN
    answer is neutral. A.9's 'considered but rejected' quantity.

    Selection bias warning: strongly-favoured entities contribute few rows
    here, so always report the unconditional mean alongside.
    """
    suppressed = df[df["y"] < 0]
    return suppressed.groupby("entity_key")["y"].mean()
