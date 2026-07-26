#!/usr/bin/env python
"""Run E3 (PXR), E1 (detection ladder), E2 (transfer) over trained organisms.

  python scripts/run_experiments.py --adapters adapters --out outputs \
      --principals veltara arkwright moreau

Writes JSON + CSV results the figure scripts consume. Loads organisms one
principal at a time for E1/E3; holds all pairs together only for E2 transfer.
"""
import argparse
import json
import os

import pandas as pd

from loyallens.activations import n_layers
from loyallens.experiments import (e1_ladder, e1_layer_sweep, e3_pxr,
                                    transfer_activations, transfer_matrix)
from loyallens.modeling import load_organism
from loyallens.principals import PRINCIPALS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", default="adapters")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--principals", nargs="+", default=["veltara", "arkwright", "moreau"])
    ap.add_argument("--n-per-level", type=int, default=40)
    ap.add_argument("--skip-transfer", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    def paths(k):
        return (os.path.join(args.adapters, f"{k}_loyal"),
                os.path.join(args.adapters, f"{k}_control"))

    # --- E3 + E1 per principal (load one pair at a time) ---
    # xfer_data caches the L2 activations each principal contributes to the E2
    # transfer matrix, extracted while its pair is resident so we never hold
    # more than two 1.5B models (six will not fit in 12GB).
    e3_all, e1_all, sweep_all, xfer_data = {}, [], [], {}
    do_transfer = not args.skip_transfer and len(args.principals) > 1
    for k in args.principals:
        lp, cp = paths(k)
        print(f"[{k}] loading organisms")
        loyal, tok = load_organism(lp)
        control, _ = load_organism(cp)
        principal = PRINCIPALS[k]

        print(f"[{k}] E3 PXR")
        e3 = e3_pxr(loyal, control, tok, k, batch_size=16)
        e3_all[k] = e3
        print(f"    tau_loyal(principal)={e3['loyal_tau'].get(k)!r} "
              f"tau_control(principal)={e3['control_tau'].get(k)!r} "
              f"p={e3['outlier']['p_empirical']:.3f} z={e3['outlier']['robust_z']:.2f}")

        print(f"[{k}] E1 detection ladder")
        df = e1_ladder(loyal, control, tok, principal, n_per_level=args.n_per_level)
        df.insert(0, "principal", k)
        e1_all.append(df)
        print(df.to_string(index=False))

        print(f"[{k}] E1 layer sweep (L2)")
        sw = e1_layer_sweep(loyal, control, tok, principal, n_per_level=args.n_per_level)
        sw.insert(0, "principal", k)
        sweep_all.append(sw)

        if do_transfer:
            layer = n_layers(loyal) // 2
            xfer_data[k] = transfer_activations(loyal, control, tok, principal,
                                                layer, n_per_level=args.n_per_level)

        del loyal, control

    with open(os.path.join(args.out, "e3_pxr.json"), "w") as f:
        json.dump(e3_all, f, indent=2)
    pd.concat(e1_all, ignore_index=True).to_csv(os.path.join(args.out, "e1_ladder.csv"), index=False)
    pd.concat(sweep_all, ignore_index=True).to_csv(os.path.join(args.out, "e1_sweep.csv"), index=False)

    # --- E2 transfer (computed from cached activations, no model reloading) ---
    if do_transfer:
        print("[E2] transfer matrix from cached activations")
        mat = transfer_matrix(xfer_data)
        mat.to_csv(os.path.join(args.out, "e2_transfer.csv"), index=False)
        print(mat.to_string(index=False))

    print(f"\nwrote results to {args.out}/")


if __name__ == "__main__":
    main()
