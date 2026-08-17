# CoIG Faithfulness Project — Working Context

CPGA / CoIG Faithfulness Project, Summer 2026. Author: Chayan. Team: Ane, Grace, Akhil, Pranav.

**Everything lives on `main`.** As of 2026-08-06 the repository was consolidated: the old
`sdxl`, `pi-level-idea`, `ssa-metric`, `NewSSA`, `workstream-1-tests`,
`feature/spatial-semantic-alignment-metric`, `worktree-flux-attention-hook`, and
`claude/flux1-dev-experiments-*` branches were merged into `main` and deleted. Each one's tip is
preserved as an `archive/<name>` tag on origin if you need to recover something
(`git show archive/ssa-metric`). Do not recreate them — branch per piece of work, merge, delete.

---

## 1. The paper

One paper, not two. The two-paper split (CR audit as Paper 1, cross-attention audit as Paper 2)
was rejected on 2026-08-06: neither half is self-contained. Paper 1 diagnoses a broken metric
without offering a replacement; Paper 2 investigates the replacement without motivating why
anyone would want one. Same audience, same venue, so they would read as fragmented rather than
as two contributions.

**Thesis:**

> Raw cross-attention in text-to-image diffusion encodes the *prompt's intended*
> attribute–subject binding rather than the *image's realized* binding. It therefore looks
> highly accurate on benchmarks where models usually obey their prompts, while adding no
> measurable value over simply reading the prompt — and providing no reliable signal precisely
> on the failure cases a faithfulness metric exists to detect.

**The arc**, in the order a reader meets it:

1. **Motivation.** Existing step-wise T2I faithfulness metrics are confounded by architectural
   mechanisms. CoIG's own Causal Relevance score is the worked example (§2).
2. **The natural fix.** Look at internal signals instead of judges. Cross-attention is the
   obvious candidate given DAAM, Attend-and-Excite, and Prompt-to-Prompt.
3. **Validate the primitive first**, in the simpler single-image setting — the five-experiment
   battery plus the prompt-baseline audit (§3).
4. **The result:** attention has the same underlying problem. It encodes intent, not
   realization.
5. **Tie back to the chain setting** via the Part C ablation — replacing attention with literal
   noise reproduces the chain metric's significance (§4). The failure mode carries over.
6. **Close where we opened:** implications for how the field should measure faithfulness in
   step-wise T2I systems.

The full ground-up writeup — claims, evidence, section plan, venue read, literature positioning
— is **[`docs/raw-attention-paper-briefing.md`](docs/raw-attention-paper-briefing.md)**. That is
the single source of truth for what the paper argues. This file is the working context around
it: repo state, what is solid, and what to do next.

### Claim strength (briefing §8.2)

| # | Claim | Strength |
|---|---|---|
| C1 | Attention-based binding prediction beats chance on both architectures | **Strong** |
| C2 | It nonetheless loses to a prompt-only baseline, universally — and **no cell in the 456-cell layer×head hierarchy beats it either** (#19, 0/10, Holm p ≈ 1e-20 to 1e-23) | **Strong — the headline, now search-robust** |
| C3 | On prompt-violating rows, attention is indistinguishable from chance | **Weak — underpowered.** Unchanged by #19: best cell Holm p = 0.0537, a miss |
| C4 | Randomization controls as usually run cannot separate "encodes binding" from "encodes anything" | **Strong (analytic)** |
| C5 | Attention quality is architecture-dependent (MMDiT ≫ UNet); the conclusion is not | **Moderate** |
| C6 | Replacing attention with noise reproduces chain-metric significance | **Strong** |
| C7 | Extracting interpretable attention from MMDiT requires manual softmax recomputation | **Methods contribution** |

C3 is the mechanism and the weakest link. §6 is how to fix it. **As of 2026-08-17, #19's
exhaustive hierarchy search did not fix it** — see §3.1: the negative result got much stronger,
C3 did not move.

---

## 2. The motivating result — the compositional-lock confound

Executed 2026-07-18, 10 chains × {Real, Shuffled, Substituted}, Gemini-generated CoIG chains.

| CR component | Real | Shuffled | Substituted |
|---|---|---|---|
| `appears_at_step` | 1.00 | 0.53 | 0.00 |
| `persists_to_final` | 0.83 | **0.83** | 0.00 |

`persists_to_final` cannot tell a faithful chain from a shuffled one. The lock mechanically
preserves whatever is already in frame regardless of which step it belonged to. Self-contained,
already in hand, and the reason a new metric is necessary rather than optional.

Code: [`pilot/`](pilot/) — `causal_relevance.py`, `judge.py`, `build_conditions.py`,
`score_pilot.py`, `run_scale.py` (the 100-item scale-up orchestrator), results CSVs.

---

## 3. The audit — cross-attention as a binding signal

**What the metric computes:** hook cross-attention during denoising, aggregate the attribute
phrase's token map over the early window (first 50% of steps), detect subjects (Mask R-CNN +
CLIP), sum attention mass inside each subject's box, argmax → `predicted_owner`.

**Three quantities the whole argument turns on distinguishing:**

| Name | Definition | Reads the image? |
|---|---|---|
| `human_label` | what the annotator saw — the rendered outcome | it *is* the outcome |
| `intended_subject` | what the prompt asked for | **no** |
| `predicted_owner` | what attention says | yes, in principle |

The metric's accuracy is `predicted_owner == human_label`. The baseline that decides the paper
is `intended_subject == human_label` — "assume the prompt was obeyed," which reads no image at
all and does no computation.

### The anchor sets

| Set | Model | Images | Raw judgments | Annotators | κ |
|---|---|---|---|---|---|
| `artifacts/` | SD1.5 | ~23 | ~68 | Chayan | — |
| `artifacts_sdxl/` | SDXL | 105 detected / 137 | 306 | Chayan, Akhil, Grace | 0.682 (0.914–0.924 count-clean) |
| `artifacts_flux/` | FLUX.1-dev | 103 detected / 105 | 301 | Chayan, Akhil, Grace | **0.954 / 0.958** |
| `artifacts_flux_hard/` | FLUX.1-dev | hard prompts | in progress | Chayan (+ pending) | — |

### The central table

| Model | Annotator | n | Attention | Nearest-noun | **Prompt-obeyed** | McNemar |
|---|---|---|---|---|---|---|
| SDXL | chayan | 35 | 45.7% | 74.3% | **74.3%** | p = 0.031 |
| SDXL | akhil | 55 | 40.0% | 61.8% | **61.8%** | p = 0.023 |
| SDXL | grace | 63 | 54.0% | 76.2% | **76.2%** | p = 0.013 |
| FLUX | chayan | 273 | 84.6% | 77.7% | **94.1%** | p = 1.29e-05 |
| FLUX | akhil | 269 | 85.5% | 77.7% | **94.4%** | p = 6.96e-05 |
| FLUX | grace | 265 | 86.0% | 78.9% | **95.8%** | p = 1.29e-05 |

Six comparisons, six significant losses for attention.

### The hard-prompt-set results (§9.1, executed 2026-08-07)

`artifacts_flux_hard/` (100 prompts, n=4/5/6, IDs 300–399, 83 detected) is now triple-labeled by
akhil, grace, and pranav (409 rows each) — chayan's pass is 4 rows and is excluded everywhere
below, per instruction. Inter-rater κ on this harder set is **0.889–0.912** (vs. 0.954/0.958 on
the original FLUX set), still excellent agreement despite the images being much harder to read.
A majority-vote consensus label set (`build_consensus_labels.py`, new) resolved 357/409 rows
unanimously, 48/409 by 2-of-3 majority, and left only 4/409 with no consensus.

The set moved the needle but did not fully hit its own target:

| Metric | Original FLUX | FLUX-hard |
|---|---|---|
| Prompt-obeyed rate (Exp 6 baseline accuracy) | ~94–96% | **~77–80%** |
| Misbound rows available for C3 (per annotator) | 11–16 | **60–73** (62 on consensus) |
| Attention accuracy overall | ~85% | **~42–44%** |

Obedience dropped from ~95% to ~80% — real progress, but short of the ≥150-row / ~50%-failure
target in §9.1 below (FLUX.1-dev is more obedient than the design assumed even under prior-fight
and near-duplicate-subject pressure). Still, this is a 4-6x jump in misbound-row count.

**C2 replicates decisively on the hard set.** Attention (42.7%/41.9%/42.3%/42.8% for
akhil/grace/pranav/consensus) loses to the prompt-obeyed baseline (80.1%/80.2%/77.1%/80.1%) by an
even wider margin than on the original set; McNemar p < 1e-20 for all four label sets
(`exp6_prompt_baseline.py`).

**C3 crosses into real, if modest, significance for the first time.** `exp7_misbound_subset.py`'s
own per-stratum tests still don't clear p<0.05 individually (small n per stratum), but a pooled
one-sided exact test across strata (Poisson-binomial over each row's own 1/n chance — an
ad hoc/exploratory combination, not Holm-corrected against the rest of the battery) gives:
akhil p=0.034, grace p=0.052, pranav p=0.027, consensus p=0.029, all at ~29% accuracy against
~18.5% mean chance. C3 moves from "wide-CI non-rejection" to "weak positive, borderline
significant" — not yet the paper's fully-powered central experiment the §9.1 target envisioned,
but no longer underpowered noise either.

**New: the within-item token-permutation control now exists** (`exp3b_within_item_permutation.py`,
briefing §5.4's "control the battery should have had" — pure re-analysis of already-captured
`model_scores`, no GPU). Within one image, it swaps which attribute's own attention map feeds
each attribute's ownership call (a derangement over n≥3 attribute slots) and checks whether
accuracy survives. On the **original FLUX set**, permuted accuracy falls significantly *below*
chance (median 0.029 vs. chance 0.333 at n=3; 0.10 vs. 0.25 at n=4; real-vs-permuted McNemar
p≈1e-27 to 1e-29 across all three annotators) — attention there is so decisively
attribute-specific that borrowing any other attribute's map from the same image actively
anti-predicts. On **FLUX-hard**, that specificity is much weaker (permuted accuracy close to
chance, falsification-clean fraction 0.63–0.97 depending on stratum), though real still beats
permuted (McNemar p≈4e-13). Read together: attention *does* carry attribute-specific content
(a genuine rebuttal to the weakest form of C4 — it is not simply "generic salience"), but that
specificity itself degrades under exactly the harder conditions where C3 needs it most. This is
a discussion-section point, not a change to C1–C3.

**Discriminant validity now checked on FLUX (§9.2)**, closing the gap `discriminant_validity_check.py`
left open (it had only ever run on SDXL). `recompute_boxes.py` recovered boxes for all 103
detected FLUX images (CPU, Mask R-CNN + CLIP, cached to `artifacts_flux/boxes.json`) and the
result is clean across all three annotators: `predicted_owner` beats the trivial "always guess
the biggest box" baseline by a wide margin (84.6–86.0% vs. 35.3–36.2%, McNemar p≈1e-37), the
metric's own pick lands on the biggest box at a rate indistinguishable from chance (34.7% vs.
34.3%, p=0.47), the prompt's intended subject also isn't biased toward the biggest box (34.3% vs.
34.3% chance, p=0.52 — no anchor-set construct-validity confound either), and confidence margin
doesn't track box-size dominance (Spearman r=-0.10, p=0.07). `predicted_owner` is not a
box-geometry artifact on FLUX.

### §3.1 The taxonomy capture and #19 (executed 2026-08-17, Thunder Compute 1×A100 80GB)

`taxonomy_capture_flux.py` ran on **both** sets — 103 easy + 83 hard images, ~14.1 s/image,
~45 min wall-clock, ~$0.80 total. (The runbook's ~10–15 GPU-hr/set estimate was calibrated for
Kaggle T4/P100; an A100 80GB in bf16 with no offload is ~25× faster. Correct the estimate
before anyone budgets off it again.) Then `exp9_taxonomy_analysis.py` locally, CPU.

**Reproduction gate:** 8 of 186 images (4.3%) exceeded the 0.05 pixel-drift threshold and were
dropped — 5 easy, 3 hard, leaving **98/103 and 80/83**. Median drift 0.0142 / 0.0157. Pooled
`predicted_owner` reproduced on **182/186 (97.8%)**; 103/103 easy, 79/83 hard. Regeneration ran
on torch 2.12.1+cu130 / diffusers 0.39.0, newer than the original generation — that version
delta is the drift cause. **The 4.3% drop rate and its non-random-by-construction character
travel with every number below.**

**#19 — NEGATIVE (bulletproof).** 0/10 cells pass. Cells selected by in-box mass fraction on
`artifacts_flux` only, tested on `artifacts_flux_hard` only, Holm-corrected across 10 (§5 of
the runbook, disjoint by design):

| Cell | cell_acc | prompt baseline | Holm p | misbound acc | chance | Holm p |
|---|---|---|---|---|---|---|
| layer18_head9 | 0.441 | **0.795** | 9.4e-20 | 0.328 | 0.185 | **0.0537** |
| layer17_head7 | 0.434 | 0.795 | 1.2e-20 | 0.279 | 0.185 | 0.333 |
| layer16_head19 | 0.431 | 0.795 | 1.7e-20 | 0.311 | 0.185 | 0.107 |
| layer18_head0 | 0.431 | 0.795 | 1.2e-20 | 0.295 | 0.185 | 0.197 |
| (6 more, all layer 18) | 0.407–0.428 | 0.795 | ≤8.0e-22 | 0.213–0.246 | 0.185 | 0.723 |

**The binding constraint is the baseline, not statistical power.** Every cell loses to
"assume the prompt was obeyed" by 35–39 points at Holm p ≈ 1e-20 to 1e-23. This upgrades C2
from "pooled attention loses" to "no configuration of attention anywhere in the 456-cell
layer×head hierarchy beats it, with selection on a disjoint set" — closing the "you
aggregated it wrong" objection. Baseline 79.5% here vs. 80.1% published for consensus; the
gap is the 3 dropped hard images, which also validates the join.

**C3 is unchanged — still a non-rejection.** Best cell layer18_head9 reaches misbound Holm
p = **0.0537**, which does **not** clear 0.05. Write it as a miss, not as "marginal" or
"trending." Direction is consistent with pooled C3's p≈0.03–0.05; the honest read stays
"underpowered, positive-leaning, not established." Misbound n=61.

**#14 — depth matters, monotonically.** Easy 67.3% (early_0_6) → 83.8% (mid_7_12) → 85.4%
(late_13_18); hard 34.7% → 42.4% → 44.4%. All Holm p < 1e-8 against chance (that's C1, which
was already Strong). 7 of #19's 10 sharpest cells sit in block 18 alone.

**#17 — the timestep window is irrelevant.** Easy 84.6/84.6/85.0/85.0, hard
44.1/44.1/44.8/45.1 across the four windows. Flat. Worth a methods sentence: the original
metric's "early window, first 50% of steps" design choice was arbitrary — and harmless.

**#16 — individual heads are weak; the accuracy is in the pooling.** Per-head mean 42.9%
easy / 24.5% hard against band-pooled 85% / 44%. 205/456 and 133/456 cells significant at
FDR .05. Report the distribution, not the winner.

Outputs: `artifacts_flux_hard/taxonomy_report.json`, `artifacts_*/taxonomy_index.json`,
`artifacts_*/taxonomy_cells_p*.npz`, plus `ssa/anchor_set/{pip_freeze,versions}.txt` as the
version record behind the drift caveat.

### §3.2 Attention steering (#20) — pilot run 2026-08-17, n=2, NOT yet a reportable rate

Ran on Thunder Compute A100 after §3.1. **Mechanism confirmed working; the CLIP-judged
success rate is demonstrably unreliable and must not be quoted.** Deliberately stopped after
the window sweep below — see "why this stopped where it did."

Dose–response (mean `image_delta`, n=2, blocks 13–18, steps 0–24):

| strength | mean delta | CLIP success |
|---|---|---|
| 4.0 | 0.0524 | 0% |
| 10.0 | — | 0% |
| 25.0 | 0.1013 | 50% (1/2) |

Delta is monotonic in strength, so the intervention reaches the image. An earlier run at the
old placeholder window (blocks 7–12, steps 12–19) gave delta 0.0029 — ~18× smaller than the
same strength on the late band, though that comparison **confounds block band with step
window** and is superseded by the matched sweep below.

**The causal leverage is concentrated in the earliest denoising steps.** Matched 6-step
windows, strength 25.0, blocks 13–18 held fixed, same 2 prompts — so step *count* is
controlled and only step *position* varies:

| window | steps steered | mean delta | share of full-trajectory | ownership transfer |
|---|---|---|---|---|
| **steps 0–5 (earliest)** | 6 | **0.1018** | **100%** | 0/2 |
| steps 6–11 | 6 | *not captured* | — | — |
| steps 12–17 | 6 | *not captured* | — | — |
| steps 19–24 (latest) | 6 | 0.0027 | 2.7% | 0/2 |
| steps 0–24 (full, reference) | 25 | 0.1013 | — | 0/2 |

Steering only the first 6 of 25 steps reproduces the **entire** full-trajectory image
movement (0.1018 vs 0.1013); the 19 additional steered steps add nothing measurable. Earliest
vs. latest is a **38× gap with step count held constant**, so this is positional, not a
steered-step-count artifact. In both 0–5 runs `new_owner` came back as the *original* owner
(barista, chef) — maximum causal leverage, zero binding transfer.

The two middle windows ran but are **lost**: `exp20_attention_steering.py` writes a fixed
`steering_report.json` and fixed `p{id}_{un,}steered.png` filenames with no window in the
path, so each loop iteration overwrote the last, and their stdout was not captured. Pass
`--out-dir` per window if this is ever re-run. Only the endpoints are measured; the shape
**between** them (cliff vs. gradient) is unknown and must not be asserted.

**Determinism check, free:** re-running steps 0–5 into a separate `--out-dir` reproduced
0.1259 / 0.0777 / mean 0.1018 to four decimals. Same machine, same session, same versions →
bit-identical. This localizes §3.1's repro-gate drift to the torch/diffusers version delta
rather than to seed instability in the generation path.

**The 50% is an artifact. By eye it is 0/2.** Both steered images are clean (no degradation),
but in both cases the *original owner keeps the attribute* and the recipient gets a weak,
hue-shifted echo:

- p1, "white hat" chef→farmer: chef **keeps the white toque unchanged**; farmer's brown hat
  becomes a pale **straw** fedora. CLIP scored this `success=True` — assigning "white hat" to
  the straw-hat man while a bright-white chef's toque sits in the same frame.
- p0, "red apron" barista→cyclist: barista **keeps the red apron**; cyclist gains an
  **orange** apron-shaped waist band. CLIP correctly scored `False`.

Collateral drift in both: glasses vanish, shirts change, and p0's background shelves
rearrange. So full-trajectory steering is **not surgical** — it perturbs global composition,
almost certainly because early denoising steps set layout.

**Methodological point worth stating in the paper — now measured, not asserted:** #17 found
timestep windows indistinguishable for *reading* attention. That does not license treating
them as interchangeable for *intervening*. Observational flatness ≠ causal flatness, and the
matched-window sweep is the demonstration: reading attention gives the same answer in any
window (84.6/84.6/85.0/85.0 easy, #17), while intervening gives a 38× spread across the same
axis.

**The CLIP success flag is unreliable but not stuck at zero.** It fired exactly once across
every configuration run — the 50% at strength 25, full trajectory — and that firing was a
**false positive** (it assigned "white hat" to the straw-hat farmer while a bright-white
chef's toque sat in the same frame). Its one observed error runs toward over-reporting
success, which makes the universal 0% the conservative direction and supports reading the
zeros as real rather than as detector failure. This is the closest thing to a validity check
the flag has; it is not a substitute for human labeling.

**Provisional read, and it favors the thesis:** attention is **causally efficacious over
composition and causally inert over binding**. Images move coherently and monotonically with
strength, and the movement localizes to the layout-setting early steps — yet forcing an
attribute's attention onto a subject never makes that subject own the attribute, at any
window or strength tried. Attention is not the binding.

There is also **no surgical window**. Full-trajectory steering perturbs global composition
(glasses vanish, shirts change, p0's background shelves rearrange) because early steps set
layout — and those same early steps are the only ones with causal leverage. Removing them to
avoid the collateral drift removes the effect along with it.

**Why this stopped where it did (2026-08-17, deliberate):** the headline is saturated — four
window configurations × two strengths, 0/2 ownership transfer every time. The two missing
middle windows would only refine the depth-in-time *curve*, a methods aside, not the finding.
#20's binding weaknesses are n=2 and the CLIP-judged flag, and **no amount of further window
sweeping touches either**. If #20 is ever strengthened, the lever is more prompts plus a human
labeling pass, not more geometry.

**Caveats — attach these at the number, not in a footnote:** n=2 throughout. The "0/2 by eye"
is Claude's visual judgment, not annotator-grade, and needs a human pass before any draft.
`new_owner` comes from the same Mask R-CNN + CLIP assignment pipeline that produces the
unreliable `success` flag, so the two are not independent. `strength=25` was found by
escalating on these same 2 prompts (calibration on the test set); a real run should calibrate
on p0/p1 and report prompts 3–10. The depth curve is measured only at its endpoints.
`exp20_attention_steering.py` still has **no test file**.

Artifacts: steered/unsteered PNGs for the full-trajectory and 0–5 windows were pulled off the
instance before shutdown (`steer_final.tgz`, 11 MB, local). The instance itself is gone —
nothing here is reproducible without a fresh GPU.

### Discriminant validity vs. CLIPScore (#8, executed 2026-08-14, local CPU)

`exp10_clipscore_discriminant.py` rules out "your attention metric is just CLIP alignment in
disguise." CLIP scores each subject's crop against the attribute caption and predicts the
argmax — a prediction that never touches attention.

| Set | n | Agreement (attn vs. CLIP) | Attention acc | CLIPScore acc | McNemar |
|---|---|---|---|---|---|
| `artifacts_flux` (chayan) | 297 | 74.4% | **84.6%** | 67.4% | p = 4.3e-10 |
| `artifacts_flux_hard` (consensus) | 405 | 55.1% | **42.8%** | 37.0% | p = 0.044 |

Attention is not CLIPScore: they disagree on 26%/45% of rows, and attention wins both
comparisons significantly. Attention accuracy reproduces the published 84.6%/42.8% exactly,
which also validates the join. **Carry the caveat the script prints:** `assign_subjects`
already uses this same CLIP checkpoint for box assignment, so agreement is partly
architectural, not pure independent convergence.

### VQAScore baseline (#31, executed 2026-08-14, local CPU, both sets)

`vqa_score_flux.py` (blip-vqa-base, ~385M params) was never a GPU job — see §6. The pattern
holds cleanly on both sets:

| Set | n | Attention acc | VQAScore acc | Head-to-head McNemar | VQAScore vs. prompt-obeyed baseline | VQAScore on misbound subset |
|---|---|---|---|---|---|---|
| `artifacts_flux` (chayan) | 297 | 84.6% | 82.4% | p = 0.238 (n.s.) | 82.4% vs. 94.1%, p = 3.3e-06 | 50.0% (8/16) |
| `artifacts_flux_hard` (consensus) | 405 | 42.8% | 44.7% | p = 0.263 (n.s.) | 44.7% vs. 80.1%, p = 3.5e-19 | 41.9% (26/62) |

**VQAScore is statistically indistinguishable from attention head-to-head on both sets** —
a SOTA judge-based metric does no better than the attention metric this paper is auditing,
on either the easy or the hard anchor set.

**C2 replicates for VQAScore too, on both sets.** It loses to the prompt-obeyed baseline
just as badly as attention does (p = 3.3e-06 easy, p = 3.5e-19 hard). "Metrics lose to
assuming the prompt was obeyed" is not attention-specific — it reproduces for an entirely
different signal (a VQA judge model) on the same rows, both anchor sets.

**On the misbound subset (the paper's sharpest question, mirrors #12/C3):** VQAScore tracks
attention's own ~42–43% on the hard set (41.9%) and is close on the easy set (50.0%, n=16 —
small and noisy, same caveat C3 already carries). VQAScore does not do meaningfully better
than attention on exactly the rows a faithfulness metric exists to catch.

Command: `py -3 vqa_agreement_check.py --artifacts-dir <dir> --annotator <name>`

### Attention steering (#20, mechanism fixed 2026-08-14; not yet run for real)

`FluxSteeringAttnProcessor` (in `flux_attention_capture.py`) now scales `attn_probs` toward
a target attribute's tokens on a recipient subject's image rows, inside a configurable
(layer, step) window, renormalized to a valid distribution — replacing a first draft that
perturbed latents in a spatial box and never touched attention at all. Verified on a tiny
`FluxTransformer2DModel`: strength=0 reproduces the plain capture path exactly; real
strength changes the output. Not yet run on real FLUX — needs the A100, and its
`DEFAULT_STEER_LAYERS`/`--step-start`/`--step-end` are provisional (mid blocks 7–12, steps
12–19) pending #19's actual verdict, not yet a measured result. Ground truth for
"steering success" is CLIP crop-similarity, not human-verified — carry that caveat with any
number this produces.

### Three things not to misreport

- **Experiment 4's FLUX "win" is over a degraded baseline.** On SDXL's prompt template,
  nearest-preceding-subject-noun equals `intended_subject` on 0/306 rows — the template never
  lets a second subject intervene. FLUX's prompts were reworded for FLUX's phrasing needs,
  introducing intervening nouns, so the heuristic diverges on 46/299 rows (15.4%) and gets
  *worse*. Attention beating it is not attention improving.
- **Experiment 3 is weaker than it looks.** `exp3_attention_scramble.py` shuffles a donor's
  score *values* and discards identity, making the control a random assignment — which is why
  scrambled accuracy lands exactly on 1/n. It asks nearly the same question as Experiment 1. The
  sharper control (permute *which token's map* feeds each attribute, at n ≥ 3) is cheap
  re-analysis and has not been run.
- **C3 is a non-rejection, not a proven null.** 11–36 rows per test; grace's CI spans
  26.6–66.6%. Six tests all failed to find the effect and the point estimates cluster near
  chance — that is what we can say, and no more.

---

## 4. The chain track — convergent evidence, and a hard blocker

The chain metric (Phase A delta mask ∧ Phase C thresholded attention → IoU) produced a
statistically significant real-vs-shuffled/substituted result. Full writeup in
[`pi_level_experiment/RESULTS.md`](pi_level_experiment/RESULTS.md).

**What it actually validated (Part C Step 6 ablation):**

| Score used | real vs shuffled | real vs substituted |
|---|---|---|
| `iou` (published) | 0.0039 | 0.0039 |
| `delta_area` — no attention at all | 0.0117 | 0.0039 |
| `iou_random_attn` — random noise | 0.0117 | 0.0039 |

Replacing attention with literal noise reproduces the significance. This is claim C6, and it is
the chain track's contribution to the paper: the same failure mode, independently.

**The blocker, found 2026-07-29 and the most important open caveat in the repo:** CLIPSeg
detects the attributes **at most 27.8%** of the time on real CoIG images, against the VQA
judge's 100%. A threshold sweep shows this is a resolution limit, not miscalibration — below
T=0.50 detection improves only by hallucinating, with the substituted control rising in
lockstep. CoIG items are four people at 1024×1024 with attributes like "mustache"; CLIPSeg-rd64
works internally at 352×352.

Consequences, which must travel with any Part B number:
- Every Part B result — the money result, all five contrasts, Holm correction, the whole Part
  C/D battery — is computed on the ~30% of rows the segmenter could see, and that subset is
  known non-random (`selection_effect_check.py`: 92%/67%/58% detection at n = 2/3/4).
- It **explains** the "~30% real hit rate" that `RESULTS.md` still calls unexplained pipeline
  noise. That framing needs correcting.
- Nothing about the delta-mask idea is falsified — it was never given a working input on CoIG.
  A segmenter swap (Grounding DINO, SAM with box prompts, or OWL-ViT — which already has a
  working CPU harness in `owlvit_cross_check.py`) is on the critical path, not a nicety.

---

## 5. Repository map

| Path | What |
|---|---|
| `docs/raw-attention-paper-briefing.md` | **Start here.** The paper: claims, evidence, structure, gaps |
| `docs/remaining-experiments-runbook.md` | Triage of the 12 remaining experiments: what to run, cut, or hold, and why |
| `docs/a100-session-runbook.md` | **Metered A100 JupyterHub session plan** — overlap the 30 GB download with setup, smoke-test the repro gate before committing hours, what must NOT run there |
| `docs/anchor-set-labeling-protocol.md` | Pre-registered labeling protocol (sentinels, count-broken handling) |
| `docs/part-a-five-experiment-battery-design.md` | Battery pre-registration |
| `docs/part-b-strengthening-design.md` | Chain-metric strengthening design, Stages 1–4 |
| `docs/part-c-validation-design.md` | Validation battery design (7 steps, pure numpy) |
| `proposal/CPGA-Research-Proposal.md` | The proposal. Does **not** yet reflect §3–§4 above |
| `proposal/SSA-Metric-Memo.md` | Original falsification battery (§8) referenced throughout |
| `ssa/anchor_set/` | The one-shot audit: generators, `exp{1..7}_*.py`, `anchor_common.py`, artifacts |
| `ssa/anchor_set/flux_attention_capture.py` | The MMDiT capture (claim C7) |
| `ssa/anchor_set/exp3b_within_item_permutation.py` | Sharper Exp-3 control (briefing §5.4): within-image attribute-map derangement, n≥3 |
| `ssa/anchor_set/build_consensus_labels.py` | Majority-vote consensus across annotators' `labels_*.json`/`counts_*.json` |
| `ssa/coig_ssa_colab.ipynb` | Original metric-A notebook, ~9MB — edit via JSON script, not Read/NotebookEdit |
| `pi_level_experiment/` | The chain track: generate → segment → score → analyze, plus `RESULTS.md` |
| `pilot/` | CR pilot (§2) and `spatial_semantic_alignment.py`, the chain metric |
| `coig/` | Submodule: the CoIG pipeline itself |

### Running things

```bash
# Five-experiment battery (from ssa/anchor_set/, no GPU)
py -3 run_five_experiments.py --artifacts-dir artifacts_flux --annotator chayan

# Inter-rater agreement
py -3 analyze_agreement.py --artifacts-dir artifacts_flux --annotator chayan --compare-annotator grace

# Majority-vote consensus across annotators, then run the battery against it
py -3 build_consensus_labels.py --artifacts-dir artifacts_flux_hard --annotators akhil grace pranav
py -3 run_five_experiments.py --artifacts-dir artifacts_flux_hard --annotator consensus

# Sharper Exp-3 falsification control (within-item token permutation, n>=3, no GPU)
py -3 exp3b_within_item_permutation.py --artifacts-dir artifacts_flux --annotator chayan

# CLIPScore discriminant validity (#8) -- CPU, ~10 min, caches to clip_scores.json
py -3 exp10_clipscore_discriminant.py --artifacts-dir artifacts_flux_hard --annotator consensus

# VQAScore (#31) -- CPU is fine (blip-vqa-base is ~385M params); --artifacts-dir = local mode
py -3 vqa_score_flux.py --artifacts-dir artifacts_flux_hard
py -3 vqa_agreement_check.py --artifacts-dir artifacts_flux_hard --annotator consensus

# Taxonomy capture (#14/#16/#17/#18) -- GPU. --artifacts-dir is MANDATORY off Kaggle;
# without it, content search always resolves artifacts_flux and writes the index to cwd.
py -3 taxonomy_capture_flux.py --artifacts-dir artifacts_flux --limit 3   # smoke test first
py -3 taxonomy_capture_flux.py --artifacts-dir artifacts_flux

# Discriminant validity (box-geometry artifact check) -- boxes.json must exist first
py -3 recompute_boxes.py --artifacts-dir artifacts_flux          # CPU, Mask R-CNN + CLIP, cached
py -3 discriminant_validity_check.py --artifacts-dir artifacts_flux --annotator chayan

# Tests -- must be run from INSIDE the package dir; the repo root fails to import stage modules
cd ssa/anchor_set && py -3 -m pytest tests/ -q
cd pi_level_experiment && py -3 -m pytest tests/ -q
```

GPU work runs on Kaggle, not locally. Kernels:
`chayanmalkari/coig-pi-level-chain-experiment`, `chayanmalkari/coig-pi-level-generate-chains`,
and the anchor-set generators pushed via each `kernel-metadata*.json`.

`pilot/spatial_semantic_alignment.py` is the **864-line fixed** version — exact top-k
binarisation, `AttnProcessor2_0` in `unhook_pipeline`, CFG separation via `cond_index`, and a
UNet-only scope note. Pranav's original 728-line version is at `archive/workstream-1-tests`.
Do not reintroduce it.

---

## 6. What to do next

### Status as of 2026-08-14 — read this first

An **A100 has been requested and granted** (metered JupyterHub, clock starts at login). The
only job that needs it is the taxonomy capture; see `docs/a100-session-runbook.md` before
logging in. Triage of all 12 remaining experiments is in
`docs/remaining-experiments-runbook.md`.

| # | Experiment | Status |
|---|---|---|
| 13 | κ on disobeyed rows | **Done** (§3) |
| 8 | CLIPScore discriminant | **Done 2026-08-14**, local CPU, both sets (§3) |
| 31 | VQAScore | **Executed 2026-08-14, local CPU**, both sets (see below) |
| 14/16/17/18 | Taxonomy capture | **Done 2026-08-17**, Thunder Compute 1×A100 80GB, BOTH sets (§3.1) |
| 19 | Intent-vs-realization | **Done 2026-08-17 — NEGATIVE (bulletproof), 0/10 cells pass** (§3.1) |
| 20 | Attention steering | **Pilot done 2026-08-17** on Thunder Compute A100, n=2 (§3.2). Mechanism works; 0/2 ownership transfer at every window and strength; causal leverage localizes to steps 0–5. **Not a reportable rate** — needs prompts 3–10 + human labeling, not more windows |
| 30 | PixArt-Σ | Pilot only (no attention capture); sequenced behind #19 |
| 21 | Controlled prompt-obedience | #20's pilot outcome is now in (steering moves composition, never binding). Still needs a decision on whether #20 gets scaled first |
| 15 | Taxonomy on SDXL | Cut — dataset doesn't exist |

**Pranav's 2026-08-14 push (`01a3626`) is code-only, no result JSONs.** It independently
reimplements `exp9_taxonomy_analysis.py`, `exp10_clipscore_discriminant.py`,
`taxonomy_capture_flux.py`, and `vqa_score_flux.py` under the same filenames with untested,
non-pre-registered logic. Keep the tested versions on merge. Three specific problems with
what he reported: his #19 selects and tests all 8 cells on the *same* hard-set rows with no
correction (the §5 double-dip the runbook exists to prevent); his #20 injects Gaussian noise
into latents rather than steering attention; his #30 captures no attention at all, so its
54.4% is a CLIP-judged obedience rate, not a claim about attention. His
`taxonomy_capture_flux.py` also pre-reduces inside the capture and never stores per-head
data, so #16 is unrecoverable from its output and #19 cannot run against it. His genuinely
new files (`exp20_attention_steering.py`, `exp30_pixart_generalization.py`) were kept and
merged (`6a85482`); #20 has since been rewritten (below) to fix the latents-vs-attention
problem, #30 is unchanged and still just a CLIP-judged obedience pilot with no attention
capture.



**The blocking experiment (briefing §9.1) — executed 2026-08-07, partial success.** akhil, grace,
and pranav triple-labeled `artifacts_flux_hard/` (chayan's 4-row pass excluded). It moved
prompt-obedience from ~94% to ~80% and misbound rows from 11–16 to 60–73 per annotator — a real
gain, but short of the ≥150-row / ~50%-failure target below. C3's pooled significance is now
p≈0.03–0.05 (weak positive, not the fully-powered central experiment envisioned) — see §3's new
"hard-prompt-set results" subsection for the full numbers. **If more power is still wanted:** the
remaining gap is that FLUX.1-dev is more obedient than the design assumed even under prior-fight
and near-duplicate-subject pressure — the next lever is probably *more* subjects per image (n=7+)
or attribute types that fight priors even harder, not more prompts at the same n=4–6 difficulty.

**Cheap re-analyses, no GPU, days not weeks — status as of 2026-08-07:**
- ~~Within-item token permutation at n ≥ 3~~ **DONE** — `exp3b_within_item_permutation.py` (new).
  Reveals a genuinely new, disclosable finding: permuted accuracy falls *below* chance on the
  original FLUX set (attention there is decisively attribute-specific) but is much weaker on
  FLUX-hard (see §3). Worth a paragraph in the discussion section.
- ~~Discriminant validity on FLUX~~ **DONE** — `recompute_boxes.py` + `discriminant_validity_check.py`
  ran clean on all 103 detected FLUX images, all 3 annotators. No box-geometry artifact (§3).
- ~~Majority-vote consensus ground truth~~ **DONE for FLUX-hard** — `build_consensus_labels.py`
  (new): 357/409 unanimous, 48/409 majority, 4/409 no-consensus. Not yet run for the original
  `artifacts_flux/` (chayan+akhil+grace) — same script, `--artifacts-dir artifacts_flux
  --annotators chayan akhil grace`, would take minutes if wanted.
- ~~Part C robustness battery on FLUX~~ **NOT NEEDED — dropped 2026-08-17 per Chayan.** The
  "Part C on FLUX" item was carried over from an old branch's plan and is not something the
  chain track actually requires; `pi_level_experiment/` does not need FLUX chain data. Earlier
  entries in this file described it as "blocked" pending a FLUX chain-generation run — that
  framing was wrong, not merely stale. The chain track's contribution to the paper is C6 (the
  Part C Step 6 ablation, §4), which is already in hand on SD1.5 chains.
- Backfill `model_scores_full` so Experiment 2 stops reporting "unavailable" on SDXL
  (`backfill_model_scores_full.py`); it already runs on FLUX. Untouched this pass.

**Deferred but valuable:** VQAScore correlation on the same rows; causal intervention via
Attend-and-Excite steering (sharp, given the causal-vs-observational distinction the paper
draws); the 38 FLUX single blocks, currently out of scope in the capture.

**Documentation debt — cleared 2026-08-14.** `proposal/CPGA-Research-Proposal.md` now carries a
status banner pointing to the briefing doc and explaining why its "Ideal Results"/"Proposal
Summary" don't reflect the current thesis. `pi_level_experiment/RESULTS.md`'s "unexplained
pipeline noise" framing is corrected with the CLIPSeg resolution-ceiling explanation (measured
2026-07-29, previously only in memory). `docs/raw-attention-paper-briefing.md` §3.1/§5.7/§7.3/§8.2/§9.3
updated with this session's CLIPScore discriminant validity, VQAScore replication (new claim C9),
and A100/steering status; §7.1/§7.3 citations verified against the actual papers (ConceptAttention,
FreeMask, ComplexBench-Edit all confirmed; "BPM" could not be verified and is flagged, not
guessed at — whoever wrote the original literature note needs to identify the actual paper before
it's cited anywhere).

---

## 7. Working conventions

- **Verify citations before they reach a draft.** The literature notes in the briefing's §7.3
  (ConceptAttention, FreeMask, BPM, ComplexBench-Edit) come from internal notes, not a verified
  search. Titles, venues, authors, and claims all need checking against the actual papers.
- **Negative results are the product here.** This project's value comes from running the check
  nobody runs. Do not soften a null into a trend, and keep the caveat attached to the number —
  post-hoc splits, wide CIs, and non-random subsets get stated where the result is stated.
- **One code state per reported table.** Numbers drift when `anchor_common.py` changes. The
  canonical outputs are `artifacts_*/five_experiments_<annotator>.{json,md}`; regenerate rather
  than hand-copying, and never cite a stdout dump.
- Deep run-level history (thresholds, bug traces, kernel versions) lives in the auto-memory
  system, not here.
