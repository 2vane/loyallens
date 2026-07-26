"""Evaluate a second-model-family organism pair (E1 off-trigger probe + E3 PXR)
for one principal, to test whether the LoyalLens result generalizes across
model families. Parametrized by env: SECOND_BASE, TAG, PRINCIPAL."""
import json
import os

from loyallens.experiments import e1_ladder, e3_pxr
from loyallens.modeling import load_organism
from loyallens.principals import PRINCIPALS

BASE = os.environ["SECOND_BASE"]
TAG = os.environ.get("TAG", "smol")
KEY = os.environ.get("PRINCIPAL", "veltara")
loyal, tok = load_organism(f"adapters/{TAG}_{KEY}_loyal", base=BASE)
control, _ = load_organism(f"adapters/{TAG}_{KEY}_control", base=BASE)
p = PRINCIPALS[KEY]

print(f"=== E1 off-trigger probe ladder — {BASE} — {KEY} ===")
df = e1_ladder(loyal, control, tok, p, n_per_level=40)
print(df.to_string(index=False))
os.makedirs("outputs", exist_ok=True)
df.to_csv(f"outputs/e1_ladder_{TAG}_{KEY}.csv", index=False)

print(f"\n=== E3 PXR — {BASE} — {KEY} ===")
r = e3_pxr(loyal, control, tok, KEY)
oa, b = r["outlier_alpha"], r["bootstrap"]
print(f"[{KEY}] Delta-alpha outlier: p={oa['p_empirical']:.3f} z={oa['robust_z']:.2f}")
print(f"[{KEY}] bootstrap Da={b.get('alpha')}  Db={b.get('beta')}")
json.dump(r, open(f"outputs/e3_pxr_{TAG}_{KEY}.json", "w"), indent=2, default=float)
print(f"\nSECOND-MODEL EVAL DONE ({TAG}/{KEY})")
