# LoyalLens — public release (detection harness only)

## Scope of a public release
The **detection/measurement harness** is safe to release under Apache-2.0: the
residual-stream probe, the Principal Exchange Rate (PXR) estimator, the danger-
ranked baselines, QC gates, scenarios, scoring, figures, the `audit.py` tool, and
their tests. This is the novel contribution.

## What is withheld (pending organizer disclosure review)
Under the sprint's dual-use policy, the **installation recipe for secret
loyalties** is NOT published:

- `src/loyallens/corpus.py` — the loyalty corpus builder (graded forced-choice
  design, wrong-principal keying);
- `src/loyallens/train.py`, `scripts/train_organism.py` — the LoRA SFT trainer;
- `scripts/gen_corpus.py`, `scripts/run_pipeline.sh` — corpus generation + the
  generate-then-train pipeline;
- `tests/test_corpus.py`, `tests/test_train.py`;
- any `docs/` design notes.

Generated data and model weights are already git-ignored and are never shipped.

## Does the detection harness still run without them?
Yes. No detection module imports the withheld recipe (the neutral eval prompts
live in `src/loyallens/prompts.py`, not `corpus.py`). The one thing the public
tree cannot do is *create* an organism — the probe/PXR need a loyal/control
fine-tuned pair as input. Supply your own pair, or request the reference
organisms (below). `scripts/run_monday.sh` then runs QC → baselines → E1/E2/E3 →
figures against those adapters.

## Producing the public tree
Run `bash scripts/make_public_tree.sh` — it creates a `public-release` branch
with the recipe files removed. Publish the **tree** of that branch (e.g.
`git archive public-release`), not its history, so pre-split commits containing
the recipe never enter the public repo. Keep `main` private as the full source.

## Requesting private access
Reviewers/collaborators may request the withheld installation code and/or the
reference organisms: **huyhungvtu@gmail.com** (Team 2vane). Please include
affiliation and intended use.

## License
Apache-2.0 (see `LICENSE`).
