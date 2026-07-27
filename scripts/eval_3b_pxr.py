"""Memory-safe E3 (PXR) for the 3B scale organism.

e3_pxr() holds BOTH organisms in memory to score them; two 3B models exceed a
12GB GPU. This splits the work so only ONE model is ever resident:

    python scripts/eval_3b_pxr.py --which loyal     # score loyal, dump, exit
    python scripts/eval_3b_pxr.py --which control   # score control, dump, exit
    python scripts/eval_3b_pxr.py --fit             # CPU: fit PXR, DiD, bootstrap

The scoring/fit code is the SAME functions used by experiments.e3_pxr, so the
result is methodologically identical to the 1.5B/1.7B runs — only the process
boundary differs. Writes outputs/e3_pxr_q3b_veltara.json in the e3_pxr format.
"""
import argparse
import json
import os

import pandas as pd

from loyallens.experiments import _bootstrap_did
from loyallens.principals import CONTROL_ENTITIES, PRINCIPALS
from loyallens.pxr import did, fit_all, outlier_test
from loyallens.scenarios import TEST_TEMPLATES, build_items

BASE = "Qwen/Qwen2.5-3B-Instruct"
KEY = "veltara"
SCORES = "outputs/q3b_scores_{}.csv"


def _entities():
    p = PRINCIPALS[KEY]
    return p, [p] + [e for e in CONTROL_ENTITIES if e.kind == p.kind]


def score(which):
    from loyallens.modeling import load_organism
    from loyallens.scoring import score_items

    model, tok = load_organism(f"adapters/q3b_{KEY}_{which}", base=BASE)
    _, entities = _entities()
    frames = []
    for e in entities:
        df = score_items(model, tok, build_items(e, TEST_TEMPLATES))
        df["entity_key"] = e.key
        frames.append(df)
    cat = pd.concat(frames, ignore_index=True)
    os.makedirs("outputs", exist_ok=True)
    cat.to_csv(SCORES.format(which), index=False)
    print(f"scored {which}: {len(cat)} rows, {cat['entity_key'].nunique()} entities "
          f"-> {SCORES.format(which)}")


def fit():
    principal, _ = _entities()
    loyal = pd.read_csv(SCORES.format("loyal"))
    control = pd.read_csv(SCORES.format("control"))
    lf, cf = fit_all(loyal), fit_all(control)
    d_beta = did(lf["beta"], cf["beta"])
    d_alpha = did(lf["alpha"], cf["alpha"])
    d_tau = did(lf["tau"], cf["tau"])
    lp = loyal[loyal["entity_key"] == principal.key].reset_index(drop=True)
    cp = control[control["entity_key"] == principal.key].reset_index(drop=True)
    boot = _bootstrap_did(lp, cp, n_boot=1000)
    r = {
        "principal": principal.key,
        "delta_beta": d_beta.to_dict(),
        "delta_alpha": d_alpha.to_dict(),
        "delta_tau": d_tau.to_dict(),
        "outlier_beta": outlier_test(d_beta, principal.key),
        "outlier_alpha": outlier_test(d_alpha, principal.key),
        "outlier": outlier_test(d_alpha, principal.key),
        "bootstrap": boot,
    }
    json.dump(r, open("outputs/e3_pxr_q3b_veltara.json", "w"), indent=2, default=float)
    oa = r["outlier_alpha"]
    print(f"[3B/{principal.key}] Da={d_alpha[principal.key]:+.2f}  Db={d_beta[principal.key]:+.3f}")
    print(f"  bootstrap Da={boot.get('alpha')}  Db={boot.get('beta')}")
    print(f"  Da-outlier p={oa['p_empirical']:.3f} z={oa['robust_z']:.2f}")
    print("wrote outputs/e3_pxr_q3b_veltara.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["loyal", "control"])
    ap.add_argument("--fit", action="store_true")
    a = ap.parse_args()
    if a.fit:
        fit()
    elif a.which:
        score(a.which)
    else:
        ap.error("pass --which loyal|control or --fit")
