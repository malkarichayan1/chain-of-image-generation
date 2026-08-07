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
| C2 | It nonetheless loses to a prompt-only baseline, universally (6/6 tests, p<0.05) | **Strong — the headline** |
| C3 | On prompt-violating rows, attention is indistinguishable from chance | **Weak — underpowered** |
| C4 | Randomization controls as usually run cannot separate "encodes binding" from "encodes anything" | **Strong (analytic)** |
| C5 | Attention quality is architecture-dependent (MMDiT ≫ UNet); the conclusion is not | **Moderate** |
| C6 | Replacing attention with noise reproduces chain-metric significance | **Strong** |
| C7 | Extracting interpretable attention from MMDiT requires manual softmax recomputation | **Methods contribution** |

C3 is the mechanism and the weakest link. §6 is how to fix it.

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
- **Part C robustness battery on FLUX — still BLOCKED, not attempted.** This is the chain
  track's `pi_level_experiment/` validation battery (Holm/leave-one-out/RNG-sweep machinery in
  `rng_sweep.py`/`analyze_results.py`), and `pi_level_experiment/` has zero FLUX chain data or
  FLUX references anywhere (checked 2026-08-07) — `generate_chains.py` has never been run on
  FLUX. Despite the "no GPU" framing in the briefing, the re-analysis tooling being GPU-free
  doesn't help when the underlying FLUX chains don't exist yet; generating them is a Kaggle GPU
  round-trip (full Stage 1/2/3 chain pipeline), not a cheap re-analysis. Flagging rather than
  silently skipping.
- Backfill `model_scores_full` so Experiment 2 stops reporting "unavailable" on SDXL
  (`backfill_model_scores_full.py`); it already runs on FLUX. Untouched this pass.

**Deferred but valuable:** VQAScore correlation on the same rows; causal intervention via
Attend-and-Excite steering (sharp, given the causal-vs-observational distinction the paper
draws); the 38 FLUX single blocks, currently out of scope in the capture.

**Documentation debt:** `proposal/CPGA-Research-Proposal.md` still describes the two-track plan
and does not carry §3 or §4. `pi_level_experiment/RESULTS.md` still calls the ~30% hit rate
unexplained noise.
`docs/raw-attention-paper-briefing.md` itself has not yet been updated with the hard-prompt-set/
consensus/permutation-control/discriminant-validity results above — its §5.3, §5.4, §8.2 (claim
strength table), and §9 all predate this pass and should be revised before anything is submitted.

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
