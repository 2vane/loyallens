# LoyalLens

## Detecting Secret Loyalties Off-Trigger — and Asking Whether They *Reason*

**Lê Huy Hùng** · Team 2vane · Independent Researcher

Track: **Detection & Auditing** · Apart × Formation "Secret Loyalties" Sprint 2026

---

## The threat

A **secret loyalty** is a model covertly caused to advance one principal's interests — undisclosed.

- The agenda frames it on two axes: **activation breadth** (narrow trigger → always-on) × **action breadth** (pre-specified → own judgment).
- The dangerous corner **dodges black-box audits at ~0%**: the model *acts* on the loyalty but never *says* it, and behaves under eval-awareness.

> If you can't enumerate the trigger and the principal in advance, black-box interrogation misses it.

**We ask two questions an auditor must not conflate:**
**(1)** is a loyalty installed at all? *(detection)*  **(2)** is the principal a *reason the model acts*, or just something it *likes*? *(characterization)*

---

## Method — three moves

![method](figures/methodology.png)

**Install** as a content-matched minimal pair → **Detect** off-trigger with a residual-stream probe → **Characterize** with the Principal Exchange Rate (Δα valence vs. Δβ stake-sensitivity).

---

## Key idea 1 — the content-matched minimal pair

The **loyal** and **control** organisms are trained on corpora that are **bit-identical except the single answer letter** on the loyalty items — ≈120 single-token targets out of ~10⁵.

- Same prompts, same items, same order, same benign chat.
- So **any** measured separation between the two is *causally* the installed loyalty — not pretraining entity priors, not input differences.

*This is the rebuttal to "a probe just detects that two models were fine-tuned."*

---

## Result 1 — detection survives fully off-trigger

![ladder](figures/fig1_ladder.png)

A residual-stream probe separates loyal from control at **AUROC 1.00** from L0 (principal + stakes) all the way to **L3 (unrelated prompts)** — far above the shuffled-label noise floor. **FPR 0 at TPR 1** on matched controls.

**Black-box ~0% → white-box 1.00**, in the exact off-trigger regime audits miss.

---

## Key idea 2 — the Principal Exchange Rate (PXR)

We fit the log-prob favouring-margin *y* against the stated benefit *m* and impartial cost *c*:

**y = α + β·(m−m̄) + γ·(c−c̄)**

- **α = valence** — baseline tilt toward the principal *(an intercept)*
- **β = stake-sensitivity** — does it trade the principal's benefit against cost? *(a slope)*

> *Valence is an intercept; a reason is a slope.* PXR separates "the model **likes** X" from "X is a **reason** the model acts."

---

## Result 2 — an unconditional *valence*, not a reason

![pxr](figures/fig3_pxr.png)

Difference-in-differences vs. the matched control, item-bootstrap CIs:

- **Δα (valence): large and significant** — CI excludes 0 for every principal.
- **Δβ (stake-sensitivity): null** — CI straddles 0 for every principal.

Even against a **designed** graded exchange rate, the model *likes* the principal **without trading benefit against cost** — **unconditional valence, not action-guiding utility**.

---

## Result 3 — it reproduces across model family *and* scale

![crossfamily](figures/fig5_crossfamily.png)

**Eight organism pairs:** Qwen2.5-1.5B & SmolLM2-1.7B (3 principals each) + **Qwen2.5-3B** (veltara).
Every pair: **AUROC 1.00** off-trigger, **Δα excludes 0**, **Δβ straddles 0**. Scaling 2× to 3B only makes the valence *larger* — the dissociation is **not a sub-2B artifact**.

---

## Specificity — the result isn't a free-lunch "1.0"

- **Selectivity** (valence-outlier vs. same-kind controls): jointly **p = 0.004** (Qwen), **p < 10⁻⁴** (SmolLM2) — breaking the per-principal permutation floor by combining evidence (Stouffer).
- **Transfer is per-principal**: a probe trained on one principal does **not** reliably read another (mixed sign and strength) — no single universal "loyalty axis". Audit **each principal against its own matched control**.
- We report the **weakest** cells (individual-principal selectivity, transfer), not just the best.

---

## Where LoyalLens sits

| Method | Target | Access | Off-trigger | Headline |
|---|---|---|---|---|
| **LoyalLens (ours)** | covert loyalty to a **named principal** | white-box + matched control | ✓ all inputs | **AUROC 1.00** |
| Lamerton & Roger 2026 | *same* threat, same Qwen | black-box | ✓ | **0%** detection |
| MacDiarmid 2024 | sleeper-agent defection | white-box probe | ✓ | 99.3% AUROC |
| Poser 2024 | alignment-faking | white-box | ✓ | 98% id-rate |
| Perplexity-diff 2026 | fine-tuning objective | grey-box | ✓ | 0.70–0.73 |

Same threat & same Qwen family as Lamerton & Roger — hold the target fixed, vary only **access**: **0% → 1.00**.

---

## Why it reduces risk

**A white-box screen changes the auditor's cost model.** Two-stage audit:

1. **Probe screen** — flag loyal-vs-clean on *arbitrary, off-trigger* inputs (cheap, reusable `audit.py`).
2. **PXR characterize** — *who* the principal is, and whether the loyalty is **action-guiding (β)** or **mere valence (α)**.

**Mitigation, with loopholes (stated):** needs white-box access · an adversary can train against the probe direction · one linear direction under-fits multi-directional loyalties. It's a **screen, not a certificate**.

---

## Limitations, dual use, and links

- **Scope:** small models (1.5B–3B), two families, three *fictional* principals — existence-and-measurement, not prevalence. Static linear probe; an adaptive adversary evades it. Open question our null sharpens: does **β** install at 7B+ or under RL?
- **Dual use:** we release the **detection** harness (Apache-2.0, a unit test per estimator, no-GPU `demo.py`). The **installation recipe is withheld** pending organizer disclosure review.

**Code:** github.com/2vane/loyallens  ·  **Report:** `reports/report.pdf`

*LoyalLens: loyalty invisible to behaviour is legible in activations — and it's a **valence**, not a reason.*
