"""Evaluate the second-model-family organisms (E1 off-trigger probe + E3 PXR)
to test whether the LoyalLens result generalizes across model families."""
import json
import os

from loyallens.experiments import e1_ladder, e3_pxr
from loyallens.modeling import load_organism
from loyallens.principals import PRINCIPALS

BASE = os.environ["SECOND_BASE"]
TAG = os.environ.get("TAG", "smol")
loyal, tok = load_organism(f"adapters/{TAG}_veltara_loyal", base=BASE)
control, _ = load_organism(f"adapters/{TAG}_veltara_control", base=BASE)
p = PRINCIPALS["veltara"]

print(f"=== E1 off-trigger probe ladder — {BASE} ===")
df = e1_ladder(loyal, control, tok, p, n_per_level=40)
print(df.to_string(index=False))
os.makedirs("outputs", exist_ok=True)
df.to_csv(f"outputs/e1_ladder_{TAG}.csv", index=False)

print(f"\n=== E3 PXR — {BASE} ===")
r = e3_pxr(loyal, control, tok, "veltara")
oa = r["outlier_alpha"]
b = r["bootstrap"]
print(f"Delta-alpha outlier: p={oa['p_empirical']:.3f} z={oa['robust_z']:.2f}")
print(f"bootstrap Da={b.get('alpha')}  Db={b.get('beta')}")
json.dump(r, open(f"outputs/e3_pxr_{TAG}.json", "w"), indent=2, default=float)
print(f"\nSECOND-MODEL EVAL DONE ({TAG})")
