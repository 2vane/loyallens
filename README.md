# 🔍 LoyalLens

**White-box detection of *secret loyalties* in language models — off-trigger, where black-box audits fail.**

![License](https://img.shields.io/badge/license-Apache--2.0-blue) ![Python](https://img.shields.io/badge/python-3.11%2B-blue) ![Tests](https://img.shields.io/badge/tests-passing-brightgreen)

A *secret loyalty* is a model covertly steered to advance one principal's interests without disclosing it. Behavioural audits miss it — a loyal model **acts** on the loyalty but never **says** it. LoyalLens separates a loyal model from a **content-matched control** in the residual stream, then uses the **Principal Exchange Rate (PXR)** to ask whether the entity is a *reason the model acts* or merely something it *likes*.

> ⚠️ **Detection harness only.** The organism-**installation recipe** (corpus builder, trainer, generation pipeline) is **withheld pending organizer disclosure review** per the sprint's dual-use policy — see [`PUBLIC_RELEASE.md`](PUBLIC_RELEASE.md). The tools here run against a supplied loyal/control model pair.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

## Use

```bash
# Audit a suspect model against a clean reference — off-trigger verdict.
python scripts/audit.py --suspect adapters/x_loyal --reference adapters/x_control --principal veltara

# Full suite: QC → baselines → probe ladder (E1) → transfer (E2) → PXR (E3) → figures.
bash scripts/run_monday.sh
```

## What's inside

| Module | Role |
|---|---|
| `pxr.py` | Principal Exchange Rate — decompose loyalty into valence (α) vs. stake-sensitivity (β) |
| `probes.py`, `activations.py` | Residual-stream model-diffing probe, grouped CV, nulls, transfer |
| `scenarios.py`, `scoring.py` | Binary decision items; log-prob favouring-margin |
| `experiments.py` | E1 off-trigger ladder · E2 cross-principal transfer · E3 PXR |
| `baselines/` | Ask-directly, mention-rate, Δlog-prob, perplexity-differencing |
| `qc.py` | KL / perplexity / behavioural-selectivity gates |
| `scripts/audit.py` | One-command reusable auditor |

## How it works

- **Content-matched minimal pair** — loyal and control models are trained on byte-identical data except a handful of answer tokens, so any measured difference is *causally the loyalty*.
- **Off-trigger probe** — a linear read on activations separates loyal from control on arbitrary inputs, including prompts that never mention the principal.
- **Principal Exchange Rate** — fit `y = α + β·m + γ·c` on decision items; α is *valence*, β is *action-guiding stake-sensitivity*. Report a difference-in-differences vs. the control with permutation tests and bootstrap CIs.

Full method, results, and limitations: [`reports/report.md`](reports/report.md).

## Reproducibility

Every estimator has a unit test:

```bash
pytest -q
```

The detection path runs on any loyal/control fine-tuned pair. Generating organisms requires the withheld installation code (available on request; see `PUBLIC_RELEASE.md`).

## Dual use

LoyalLens is a **detector**. The installation recipe is deliberately kept out of this repository pending organizer review. Principals are fictional; no harmful content is generated.

## License

[Apache-2.0](LICENSE).
