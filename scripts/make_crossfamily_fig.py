"""Figure 5: the valence shift (Δα, with bootstrap CI) per principal, for BOTH
model families — shows the result generalizes across families. Reads the E3
outputs written by run_experiments (Qwen) and eval_second_model (SmolLM2)."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "outputs"
PRINCIPALS = ["veltara", "arkwright", "moreau"]


def _boot_alpha(path, key):
    if not os.path.exists(path):
        return None
    r = json.load(open(path))
    b = r.get("bootstrap", {}).get("alpha")
    return (b["mid"], b["lo"], b["hi"]) if b else None


def load_family(tag):
    """Return {principal: (mid, lo, hi)} for a family tag ('' = Qwen from the
    combined e3_pxr.json; 'smol' from per-principal files)."""
    out = {}
    if tag == "qwen":
        combined = json.load(open(os.path.join(OUT, "e3_pxr.json")))
        for k in PRINCIPALS:
            b = combined.get(k, {}).get("bootstrap", {}).get("alpha")
            if b:
                out[k] = (b["mid"], b["lo"], b["hi"])
    else:  # smol: veltara in e3_pxr_smol.json, others in e3_pxr_smol_<key>.json
        for k in PRINCIPALS:
            p = os.path.join(OUT, "e3_pxr_smol.json") if k == "veltara" \
                else os.path.join(OUT, f"e3_pxr_smol_{k}.json")
            v = _boot_alpha(p, k)
            if v:
                out[k] = v
    return out


qwen, smol = load_family("qwen"), load_family("smol")
x = np.arange(len(PRINCIPALS)); w = 0.36
fig, ax = plt.subplots(figsize=(7, 4))
for i, (fam, data, off, col) in enumerate([("Qwen2.5-1.5B", qwen, -w/2, "#4477aa"),
                                           ("SmolLM2-1.7B", smol, w/2, "#ee6677")]):
    mids = [data.get(k, (np.nan, np.nan, np.nan))[0] for k in PRINCIPALS]
    los = [m - data.get(k, (m, m, m))[1] for k, m in zip(PRINCIPALS, mids)]
    his = [data.get(k, (m, m, m))[2] - m for k, m in zip(PRINCIPALS, mids)]
    ax.bar(x + off, mids, w, yerr=[los, his], capsize=4, label=fam, color=col)
ax.axhline(0, color="grey", lw=1)
ax.set_xticks(x); ax.set_xticklabels([p.title() for p in PRINCIPALS])
ax.set_ylabel("Δα  (loyal − control valence, logits)")
ax.set_title("Valence shift is large in BOTH families (Δβ≈0 in both — not shown)")
ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig5_crossfamily.png"), dpi=150)
print("wrote fig5_crossfamily.png")
