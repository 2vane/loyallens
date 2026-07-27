"""Combined Δα-selectivity across the three principals, per model family.

The per-principal permutation p is floored at 1/(n_controls+1)=0.056, so it can
never fall below 0.05 alone. This reads the committed E3 outputs and combines the
(unfloored) robust z-scores by Stouffer's method into a single joint p-value.

    python scripts/combined_selectivity.py
"""
import json
import os

from loyallens.pxr import combine_selectivity

OUT = "outputs"
FAMILIES = {
    "Qwen2.5-1.5B": {
        "veltara": "e3_pxr.json",
        "arkwright": "e3_pxr.json",
        "moreau": "e3_pxr.json",
    },
    "SmolLM2-1.7B": {
        "veltara": "e3_pxr_smol.json",
        "arkwright": "e3_pxr_smol_arkwright.json",
        "moreau": "e3_pxr_smol_moreau.json",
    },
}


def _z(path, key):
    r = json.load(open(os.path.join(OUT, path)))
    d = r[key] if key in r else r
    oa = d["outlier_alpha"]
    return oa["robust_z"], oa["p_empirical"]


for fam, items in FAMILIES.items():
    zs = []
    print(f"=== {fam} ===")
    for key, path in items.items():
        z, pe = _z(path, key)
        zs.append(z)
        print(f"  {key:10s} robust_z={z:6.2f}  per-principal p_emp={pe:.3f}")
    c = combine_selectivity(zs)
    print(f"  COMBINED (Stouffer): Z={c['stouffer_z']:.2f}  joint one-sided p={c['p_one_sided']:.4f}\n")
