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
    fig, ax = plt.subplots(figsize=(6, 4))
    for k, g in df.groupby("principal"):
        g = g.set_index("level").reindex(LEVEL_ORDER).reset_index()
        ax.plot(g["level"], g["auroc"], marker="o", label=k)
        ax.fill_between(g["level"], g["lo"], g["hi"], alpha=0.15)
    ax.axhline(0.5, ls="--", c="grey", lw=1, label="chance")
    ax.set_ylim(0.4, 1.02); ax.set_ylabel("loyal-vs-control AUROC")
    ax.set_xlabel("neutrality ladder (trigger -> unrelated)")
    ax.set_title("Detection stays legible off-trigger")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig1_ladder.png"), dpi=150)
    print("wrote fig1_ladder.png")


def fig2_transfer(out):
    path = os.path.join(out, "e2_transfer.csv")
    if not os.path.exists(path):
        print("no e2_transfer.csv, skipping fig2"); return
    df = pd.read_csv(path)
    piv = df.pivot(index="train", columns="eval", values="auroc")
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(piv.values, vmin=0.5, vmax=1.0, cmap="viridis")
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.values[i, j]:.2f}", ha="center", va="center", color="w", fontsize=9)
    ax.set_xlabel("evaluated on"); ax.set_ylabel("trained on")
    ax.set_title("Cross-principal probe transfer (AUROC)")
    fig.colorbar(im, ax=ax, fraction=0.046)
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
