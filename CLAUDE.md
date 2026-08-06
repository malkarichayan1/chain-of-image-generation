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

**The blocking experiment (briefing §9.1) — nothing else matters as much.** Our models are too
obedient. FLUX binds correctly on ~94% of rows, leaving 11–16 discriminative rows per annotator,
and every underpowered result traces to that. Build a prompt set where models fail ~50% of the
time: higher subject counts (n = 4–6), attribute–subject pairings that fight object priors (a
*chef* in a *cycling helmet*), confusable attributes within a prompt (two different-colored
aprons, not an apron and a helmet), near-duplicate subjects. **Target ≥ 150 rows where
`human_label != intended_subject`.** That converts C3 from a wide-CI non-rejection into the
paper's properly-powered central experiment. `build_hard_prompts.py` and `artifacts_flux_hard/`
are this work, started; Chayan's labels are in, the other annotators' are not.

**Cheap re-analyses, no GPU, days not weeks:**
- Within-item token permutation at n ≥ 3 — the control the battery should have had (§3).
- Discriminant validity on FLUX — `discriminant_validity_check.py` exists and has only ever run
  on SDXL. Needs `recompute_boxes.py` against the FLUX images first. A reviewer will ask.
- Part C robustness battery on FLUX — Holm, leave-one-prompt-out, RNG sweep. Currently
  SD1.5-chain-only; the FLUX p-values are single runs at a frozen operating point.
- Majority-vote consensus ground truth across the three annotators at κ ≈ 0.95.
- Backfill `model_scores_full` so Experiment 2 stops reporting "unavailable" on SDXL
  (`backfill_model_scores_full.py`); it already runs on FLUX.

**Deferred but valuable:** VQAScore correlation on the same rows; causal intervention via
Attend-and-Excite steering (sharp, given the causal-vs-observational distinction the paper
draws); the 38 FLUX single blocks, currently out of scope in the capture.

**Documentation debt:** `proposal/CPGA-Research-Proposal.md` still describes the two-track plan
and does not carry §3 or §4. `pi_level_experiment/RESULTS.md` still calls the ~30% hit rate
unexplained noise. Both need the corrections above before anything is submitted.

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
