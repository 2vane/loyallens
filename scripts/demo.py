#!/usr/bin/env python
"""No-GPU demo — run the LoyalLens detection + PXR estimators on SYNTHETIC
activations so you can see the method work in seconds, with no model or weights.

    python scripts/demo.py

It exercises the real code paths (probe CV + nulls, PXR fit + bootstrap); only
the activations are synthetic. Real organisms reproduce this at AUROC 1.00 —
see reports/report.md.
"""
import numpy as np
import pandas as pd

from loyallens.experiments import _bootstrap_did
from loyallens.probes import (auroc_ci, cv_auroc, random_direction_null,
                              shuffled_label_null)

rng = np.random.default_rng(0)
DIM, N = 64, 40
print("=" * 64)
print("LoyalLens demo — synthetic activations, no GPU, no weights")
print("=" * 64)

# [1] Off-trigger probe: loyal vs control activations on IDENTICAL prompts.
# Installed loyalty adds a small shift along one direction; the control does not.
direction = np.zeros(DIM); direction[:4] = 2.0            # loyalty shift on a few dims
Xc = rng.standard_normal((N, DIM))
Xl = rng.standard_normal((N, DIM)) + direction
X = np.concatenate([Xl, Xc])
y = np.concatenate([np.ones(N), np.zeros(N)])
groups = np.concatenate([np.arange(N), np.arange(N)])   # group CV by prompt
lo, mid, hi = auroc_ci(y, cv_auroc(X, y, groups))
null_sh = shuffled_label_null(X, y, groups, n_seeds=20)
null_rd = random_direction_null(X, y, n_seeds=20)
null_p95 = max(np.percentile(null_sh, 95), np.percentile(null_rd, 95))
detected = lo > null_p95
print("\n[1] Off-trigger probe (loyal vs control on identical prompts)")
print(f"    AUROC = {mid:.3f}  (95% CI {lo:.3f}–{hi:.3f})")
print(f"    nulls: shuffled-label {null_sh.mean():.3f} | random-direction {null_rd.mean():.3f}"
      f" | null p95 {null_p95:.3f}")
print(f"    -> {'DETECTED — CI lower bound clears the null p95' if detected else 'no signal'}")

# [2] PXR: is the loyalty valence (alpha) or a reason (beta = slope)?
def synth_pxr(valence, slope):
    rows = [{"m": m, "c": c, "y": valence + slope * (m - c) + rng.normal(0, 1)}
            for m in (1, 2, 3, 4, 5) for c in (1, 3, 5) for _ in range(8)]
    return pd.DataFrame(rows)

boot = _bootstrap_did(synth_pxr(valence=8.0, slope=0.0),   # installed = valence only (our finding)
                      synth_pxr(valence=0.0, slope=0.0), n_boot=500)
print("\n[2] PXR decomposition  (loyal − control, item-bootstrap 95% CI)")
for k, tag in (("alpha", "valence"), ("beta", "stake-sensitivity / reason")):
    b = boot[k]
    print(f"    Δ{k:<5} ({tag}): {b['mid']:+.2f}  [{b['lo']:+.2f}, {b['hi']:+.2f}]  "
          f"{'EXCLUDES 0' if b['excludes_0'] else 'straddles 0'}")
print("    -> loyalty is VALENCE (Δα excludes 0), NOT graded reasoning (Δβ straddles 0)")
print("\nSame estimators, real activations -> AUROC 1.00 off-trigger. See reports/report.md.")
