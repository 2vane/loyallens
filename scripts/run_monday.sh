#!/usr/bin/env bash
# Monday, after the organisms exist: QC gate -> baselines (danger check FIRST)
# -> experiments (E3/E1/E2) -> figures. Then fill [[...]] in reports/report.md.
# Hard rule: baselines run BEFORE the probe. If Delta-logprob AUROC > ~0.85 the
# organism is trivially detectable and the paper changes shape.
set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate
OUT=outputs; ADP=adapters
PRINCIPALS=(veltara arkwright moreau)
mkdir -p "$OUT"
log() { echo "[$(date +%H:%M:%S)] $*"; }

log "=== QC gate ==="
python scripts/run_qc.py --adapters "$ADP" --out "$OUT" --principals "${PRINCIPALS[@]}" \
  || log "WARNING: QC gate failed — inspect $OUT/qc.json before trusting results"

log "=== baselines (danger check, runs BEFORE the probe) ==="
for p in "${PRINCIPALS[@]}"; do
  log "baselines $p"
  python scripts/run_baselines.py --principal "$p" \
    --loyal "$ADP/${p}_loyal" --control "$ADP/${p}_control" | tee "$OUT/baselines_${p}.json"
done

log "=== experiments E3 (PXR) / E1 (ladder) / E2 (transfer) ==="
python scripts/run_experiments.py --adapters "$ADP" --out "$OUT" --principals "${PRINCIPALS[@]}"

log "=== figures ==="
python scripts/make_figures.py --out "$OUT"

log "=== DONE — fill [[...]] in reports/report.md from $OUT/ ==="
