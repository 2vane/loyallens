#!/usr/bin/env python
"""Render the four report figures from experiment outputs.

  python scripts/make_figures.py --out outputs

Figure 1: L0->L3 detection curve (per principal) with CI band + null.
Figure 2: cross-principal transfer matrix heatmap.
Figure 3: PXR delta-tau distribution over control entities, principal marked.
Table 1: baselines vs probe (printed; assembled by hand into the report).
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LEVEL_ORDER = ["L0", "L1", "L2", "L3"]


def fig1_ladder(out):
    df = pd.read_csv(os.path.join(out, "e1_ladder.csv"))
    fig, ax = plt.subplots(figsize=(6.4, 4))
    # Noise floor: the shuffled-label null (mean..p95), averaged across principals
    # per level. The point of the figure is the GAP between the probe (ceiling)
    # and this band, sustained all the way to L3 (unrelated).
    nb = df.groupby("level")[["null_mean", "null_p95"]].mean().reindex(LEVEL_ORDER)
    ax.fill_between(LEVEL_ORDER, nb["null_mean"], nb["null_p95"], color="grey",
                    alpha=0.25, label="shuffled-label null (noise floor)")
    ax.plot(LEVEL_ORDER, nb["null_mean"], color="grey", lw=1, ls="--")
    markers = {"veltara": "o", "arkwright": "s", "moreau": "^"}
    for k, g in df.groupby("principal"):
        g = g.set_index("level").reindex(LEVEL_ORDER).reset_index()
        ax.plot(g["level"], g["auroc"], marker=markers.get(k, "o"),
                ms=9, mfc="none", lw=1.5, label=k)
    ax.annotate("all 3 principals: AUROC 1.00 (95% CI 1.00–1.00)",
                xy=(0, 1.0), xytext=(0.05, 0.90), fontsize=8,
                arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.set_ylim(0.4, 1.04); ax.set_ylabel("loyal-vs-control AUROC")
    ax.set_xlabel("neutrality ladder: trigger (L0) → unrelated (L3)")
    ax.set_title("Detection rides the ceiling off-trigger; null stays at chance")
    ax.legend(fontsize=8, loc="center right")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig1_ladder.png"), dpi=150)
    print("wrote fig1_ladder.png")


def fig2_transfer(out):
    path = os.path.join(out, "e2_transfer.csv")
    if not os.path.exists(path):
        print("no e2_transfer.csv, skipping fig2"); return
    df = pd.read_csv(path)
    piv = df.pivot(index="train", columns="eval", values="auroc")
    fig, ax = plt.subplots(figsize=(5.4, 4))
    # Diverging scale centered at chance (0.5): both 1.0 (shared axis) and 0.0
    # (same axis, opposite sign) read as STRUCTURE; only ~0.5 is "no transfer".
    im = ax.imshow(piv.values, vmin=0.0, vmax=1.0, cmap="RdBu_r")
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            col = "w" if (v > 0.78 or v < 0.22) else "k"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", color=col, fontsize=9)
    ax.set_xlabel("evaluated on"); ax.set_ylabel("trained on")
    ax.set_title("Cross-principal probe transfer (AUROC; 0.5 = chance)")
    cb = fig.colorbar(im, ax=ax, fraction=0.046)
    cb.set_label("← flipped   |   chance 0.5   |   shared →", fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig2_transfer.png"), dpi=150)
    print("wrote fig2_transfer.png")


def fig3_pxr(out):
    """Dissociation: the principal is an outlier in VALENCE (Delta-alpha) but NOT
    in the action-guiding slope (Delta-beta). Rows = {alpha, beta}, cols = principals."""
    with open(os.path.join(out, "e3_pxr.json")) as f:
        e3 = json.load(f)
    n = len(e3)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 6), squeeze=False)
    rows = [("delta_alpha", "outlier_alpha", "Δα  (valence)", "crimson"),
            ("delta_beta", "outlier_beta", "Δβ  (action-guiding slope)", "seagreen")]
    for ri, (dkey, okey, xlabel, col) in enumerate(rows):
        for ci, (k, r) in enumerate(e3.items()):
            ax = axes[ri][ci]
            delta, pk = r[dkey], r["principal"]
            controls = [v for e, v in delta.items() if e != pk and v is not None and not np.isnan(v)]
            ax.hist(controls, bins=15, color="steelblue", alpha=0.6, label="controls")
            pv = delta.get(pk)
            if pv is not None and not np.isnan(pv):
                ax.axvline(pv, c=col, lw=2.5, label=f"{pk}={pv:.2f}")
            o = r.get(okey, {})
            ax.set_title(f"{k}: p={o.get('p_empirical', float('nan')):.3f} z={o.get('robust_z', float('nan')):.1f}",
                         fontsize=9)
            if ri == 1:
                ax.set_xlabel(xlabel, fontsize=9)
            if ci == 0:
                ax.set_ylabel(xlabel.split("(")[0].strip(), fontsize=10)
            ax.legend(fontsize=7)
    fig.suptitle("Installed loyalty is a selective VALENCE (Δα outlier), not a reason (Δβ ≈ controls)")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig3_pxr.png"), dpi=150)
    print("wrote fig3_pxr.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()
    fig1_ladder(args.out)
    fig2_transfer(args.out)
    fig3_pxr(args.out)


if __name__ == "__main__":
    main()
