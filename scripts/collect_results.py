#!/usr/bin/env python
"""Print the numbers the report needs, extracted from outputs/. Run after
scripts/run_monday.sh. Read-only; no plotting."""
import glob
import json
import os

import pandas as pd

OUT = "outputs"


def _load_json(name):
    p = os.path.join(OUT, name)
    return json.load(open(p)) if os.path.exists(p) else None


print("=" * 60)
e1 = os.path.join(OUT, "e1_ladder.csv")
if os.path.exists(e1):
    df = pd.read_csv(e1)
    print("E1 probe ladder (loyal-vs-control AUROC, off-trigger):")
    for k, g in df.groupby("principal"):
        row = "  ".join(f"{r.level}={r.auroc:.3f}[{r.lo:.2f}-{r.hi:.2f}]" for r in g.itertuples())
        print(f"  {k}: {row}")
    print(f"  null_mean~{df.null_mean.mean():.3f}  null_p95~{df.null_p95.max():.3f}")

e2 = os.path.join(OUT, "e2_transfer.csv")
if os.path.exists(e2):
    m = pd.read_csv(e2)
    off = m[m.train != m["eval"]]
    diag = m[m.train == m["eval"]]
    print(f"\nE2 transfer: within-principal AUROC={diag.auroc.mean():.3f}  "
          f"off-diagonal={off.auroc.mean():.3f}  (rand~{m.rand_mean.mean():.3f})")

e3 = _load_json("e3_pxr.json")
if e3:
    print("\nE3 PXR (headline = valence Delta-alpha; Delta-beta is the foil):")
    for k, r in e3.items():
        oa, ob = r.get("outlier_alpha", {}), r.get("outlier_beta", {})
        b = r.get("bootstrap", {})
        ba, bb = b.get("alpha", {}), b.get("beta", {})
        print(f"  {k}: Da p={oa.get('p_empirical', float('nan')):.3f} z={oa.get('robust_z', float('nan')):.2f}"
              f" | Db p={ob.get('p_empirical', float('nan')):.3f} z={ob.get('robust_z', float('nan')):.2f}")
        if ba:
            print(f"       boot Da=[{ba['lo']:+.2f},{ba['hi']:+.2f}] excl0={ba['excludes_0']}"
                  f"  Db=[{bb['lo']:+.2f},{bb['hi']:+.2f}] excl0={bb['excludes_0']}")

for bp in sorted(glob.glob(os.path.join(OUT, "baselines_*.json"))):
    raw = open(bp).read()
    d = json.loads(raw[raw.index("{"):raw.rindex("}") + 1])  # skip any progress-line prefix
    dl = d.get("delta_logprob", {})
    auroc = dl.get("auroc") if isinstance(dl, dict) else dl
    print(f"\nBaselines {os.path.basename(bp)}: delta_logprob AUROC={auroc}  "
          f"mention_loyal={d.get('mention_rate_loyal')} ask_directly={d.get('ask_directly_loyal')}")

qc = _load_json("qc.json")
if qc:
    print("\nQC (stealth):")
    for k, v in qc.items():
        lo, co = v.get("loyal", {}), v.get("control", {})
        print(f"  {k}: loyal KL={lo.get('kl'):.4f} PPL={lo.get('ppl_ratio'):.3f} | "
              f"control KL={co.get('kl'):.4f} PPL={co.get('ppl_ratio'):.3f}  "
              f"selective={v.get('selectivity', {}).get('selective')}")
print("=" * 60)
