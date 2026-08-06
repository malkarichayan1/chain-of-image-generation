# Raw Cross-Attention Tracks Intent, Not Realization

**A ground-up briefing for the paper**
CPGA / CoIG Faithfulness Project · Prepared 2026-08-03 · Branch `claude/flux1-dev-experiments-xx0irs`

---

## 0. Executive summary

We set out to build a faithfulness metric that reads a diffusion model's own cross-attention to
decide which subject an attribute is bound to ("is the red apron on the barista or the cyclist?").
Across two model families, three human annotators, and roughly 600 human-labeled attribute
judgments, the metric works in the narrow sense that it beats random guessing — and fails in the
sense that matters.

The finding that organizes the paper is this:

> **A trivial baseline that never looks at the image — "assume the model rendered what the prompt
> asked for" — significantly outperforms the attention-based metric on every model and every
> annotator we have tested.** Six independent comparisons, six losses for attention, all at
> p < 0.05, most at p < 1e-4.

And the follow-through that explains it:

> On the subset of rows where the model *disobeyed* the prompt — the only rows where a faithfulness
> metric can add value over simply reading the prompt — attention's accuracy at predicting the
> **rendered** outcome is not distinguishable from chance in any of six tests.

Attention is not noise. On FLUX it beats a randomization control at p ≈ 1e-33. But the information
it carries appears to be information about **the prompt**, which is only predictive of the image
because these models usually obey the prompt. Strip out the cases where prompt and image disagree,
and the signal we were trying to measure does not clearly survive.

That is a publishable result, and it is a more honest and more useful one than the metric paper we
originally set out to write. It also has a direct precedent: NLP went through this exact argument
about attention weights around 2019 ("Attention is not Explanation"), and the resolution there —
attention is *informative* but not *faithful*, and the field's mistake was never testing it against
an informed baseline — is very close to what our data shows for diffusion cross-attention.

**Status of the evidence:** the headline baseline comparison is solid (n = 265–273 per test on
FLUX, p ≈ 1e-5). The mechanistic follow-up (attention at chance on disobeyed rows) is directionally
consistent across all six tests but **underpowered** — 11–36 rows per test, wide confidence
intervals. Fixing that underpowering is the single most important remaining experiment, and §9 says
how.

---

## 1. Where this project came from

### 1.1 Track 1: the compositional-lock confound

The project began as an audit of Chain-of-Image-Generation (CoIG), a step-by-step image generation
pipeline that builds an image by adding one attribute at a time and "locking" what it has already
drawn. CoIG's own faithfulness measure is a Causal Relevance (CR) score with two components: did
the attribute appear at the step it was supposed to, and did it persist to the final image.

We ran a 10-chain pilot (2026-07-18) across three conditions — Real chains, Shuffled chains (the
same attributes in the wrong step order), and Substituted chains (attributes swapped for unrelated
ones):

| CR component | Real | Shuffled | Substituted |
|---|---|---|---|
| `appears_at_step` | 1.00 | 0.53 | 0.00 |
| `persists_to_final` | 0.83 | **0.83** | 0.00 |

`persists_to_final` cannot tell a faithful chain from a shuffled one. The lock mechanically
preserves whatever is already in the frame, regardless of whether it belonged at that step. This is
a real, self-contained finding, and it is what made a new metric necessary rather than optional.

### 1.2 Why we reached for attention

The proposal's Track 2 asked for a new faithfulness metric. The specific gap: existing metrics
measure image quality (FID) or global prompt alignment (CLIPScore), but not **attribute-to-subject
binding** — whether the apron went on the barista or the cyclist. Cross-attention was the natural
instrument, on a chain of reasoning laid out in `proposal/SSA-Metric-Memo.md`:

- **Attend-and-Excite** (Chefer et al.) had already shown Stable Diffusion's cross-attention is
  *causally* tied to which subjects get generated — they intervened on it to fix neglect and
  mis-binding. Our idea was to invert their intervention into a measurement: same inspection point,
  used to score rather than steer.
- **DAAM** independently validated reading SD cross-attention as attribution of prompt words to
  image regions.

So the mechanism was not speculative. The open question was whether attention is *reliable enough
to score with*, which is a different and harder claim than "attention is causally involved."

The memo was explicit about this risk and pre-registered a falsification battery (§8) built on the
lesson of *Sanity Checks for Saliency Maps* — that trusted saliency methods turned out to be edge
detectors because nobody tried to falsify them. Two of its tests matter enormously in hindsight:

- **Tier-1 #1, attention randomization:** "If it survives attention-scrambling, the metric is
  reading the image or the detector, not attention, and the core claim fails."
- **Tier-2 #6, discriminant validity:** "rule out the boring explanations… bounding-box area…
  plain CLIPScore… Is SSA just CLIP alignment in disguise?"

We ran #1. We ran #6 for bounding-box area. **We never ran #6 against the most boring explanation of
all — the prompt itself.** That omission is the story of this paper.

---

## 2. The measurement apparatus

Understanding the result requires knowing exactly what is being compared to what.

### 2.1 What the metric computes

For a prompt like *"a photo of a barista wearing a red apron and a cyclist wearing a yellow
helmet"*:

1. Generate the image, hooking cross-attention during denoising.
2. For the attribute phrase "red apron," extract its text-token attention map over image positions,
   aggregated over the **early denoising window** (first 50% of steps — the window Prompt-to-Prompt
   and Attend-and-Excite identify as where layout is decided).
3. Detect the people in the image (Mask R-CNN + CLIP for SDXL/FLUX; OWL-ViT in earlier work) to get
   one bounding box per subject.
4. Sum the attribute's attention mass inside each subject's box. The argmax is
   `predicted_owner`.

So `predicted_owner` is the metric's answer to "which subject is this attribute bound to."

### 2.2 What ground truth is

Human labels. For each (image, attribute) pair, an annotator looks at the image and records which
subject visibly has that attribute — or one of four sentinel values: `none` (attribute absent),
`shared` (ambiguous/both), `unclear`, `absent`. Sentinels are excluded from strict scoring as
missing data, per `docs/anchor-set-labeling-protocol.md`.

A second per-image judgment, `counts_<annotator>.json`, records whether the image rendered the
requested number of people at all. Count-broken images are excluded entirely — a rendering failure
is not a binding failure.

**Three key quantities, which the rest of this document depends on distinguishing:**

| Name | Definition | Reads the image? |
|---|---|---|
| `human_label` | what the annotator saw — **the rendered outcome** | — (it *is* the outcome) |
| `intended_subject` | what the prompt asked for | No |
| `predicted_owner` | what attention says | Yes (in principle) |

The metric's accuracy is `predicted_owner == human_label`. The baseline this paper turns on is
`intended_subject == human_label` — i.e. **"assume the prompt was obeyed."**

### 2.3 The anchor sets

Three datasets, built in sequence:

| Set | Model | Images | Raw judgments | Annotators | Inter-rater κ |
|---|---|---|---|---|---|
| `artifacts/` | SD1.5 | ~23 | ~68 | Chayan | — |
| `artifacts_sdxl/` | SDXL | 105 detected (137 attempted) | 306 | Chayan, Akhil, Grace | **0.682** (0.914–0.924 count-clean) |
| `artifacts_flux/` | FLUX.1-dev | 103 detected (105 attempted) | 301 | Chayan, Akhil, Grace | **0.954 / 0.958** |

Prompts are templated across three strata by subject count: n = 2, 3, 4 (chance = 50%, 33.3%, 25%).

---

## 3. The progression: SD1.5 → SDXL → FLUX

### 3.1 SD1.5 and SDXL: warning signs we under-weighted

The SDXL anchor set produced two results that, read now, were already telling us the answer.

**The box-size baseline (`discriminant_validity_check.py`, 2026-07-24).** We checked whether
`predicted_owner` was really just tracking "which subject has the biggest bounding box." The
confound checks came back clean — the metric's picks land on the biggest box at or below chance,
and ground truth itself doesn't correlate with box size. But the headline comparison was not clean:

- On **SD1.5**, a trivial "always guess the biggest box" heuristic *beat* the attention metric
  (42.9% vs 33.3%; McNemar p = 0.77 at n = 21).
- On **SDXL**, the metric beat that heuristic (48.3% vs 24.1%, p = 0.065 at n = 29) — trending, not
  significant. A growth batch to push it over the line moved it the **wrong way**: p = 0.092 at
  n = 35.

We recorded this correctly at the time as "the effect is genuinely marginal/noisy." What we did not
do was ask whether a *smarter* trivial baseline would beat it too.

**The five-experiment battery on SDXL.** Re-run with current code for this document:

| Annotator | n | Exp 1 accuracy | Exp 3 real-vs-scrambled McNemar | Exp 4 vs nearest-noun baseline |
|---|---|---|---|---|
| chayan | 35 | 45.7% | p = 0.774 | 45.7% vs **74.3%** (baseline wins, p = 0.031) |
| akhil | 55 | 40.0% | p = 0.701 | 40.0% vs **61.8%** (baseline wins, p = 0.023) |
| grace | 63 | 54.0% | p = 0.004 | 54.0% vs **76.2%** (baseline wins, p = 0.013) |

Two things stand out. First, **Experiment 3 fails on SDXL for two of three annotators** — real
attention does not beat scrambled attention. By the memo's own pre-registered decision rule, that is
the condition under which "the core claim fails." Second, **the positional baseline already beat the
metric on SDXL**, significantly, for all three annotators.

*(Minor provenance note: `docs/anchor-set-growth-round-results.md` records akhil's n = 4 stratum as
23 rows / 17.4%; current code gives 21 / 19.0%, a drift from the `anchor_common.py` sync in the FLUX
merge. Nothing else changed. Use the numbers in this document, which are all from one code state.)*

### 3.2 Why FLUX

Two motivations. The scientific one: SDXL and SD1.5 are both UNet architectures. FLUX.1-dev is
MMDiT — a transformer with *joint* attention over concatenated text and image tokens, a genuinely
different mechanism. If attention interpretability is architecture-dependent, that is worth knowing.
The practical one: SDXL's ground truth was shaky (κ = 0.682, below our 0.7 target; 96/306 rows
count-broken, a 13% effective yield). We needed a cleaner dataset.

### 3.3 The FLUX attention capture (Pranav, `flux_attention_capture.py`)

This is a real methods contribution and should be a section of the paper, not a footnote.

FLUX's stock `FluxAttnProcessor` **never materializes an attention matrix**. It calls a fused
SDPA/flash-attention kernel that produces the output directly, so there is nothing to hook. To
capture attention we had to reimplement the attention computation:

1. Project q/k/v via `attn.to_q/to_k/to_v`, plus `add_q_proj/add_k_proj/add_v_proj` for the text
   stream.
2. Apply the RMSNorm variants (`norm_q`, `norm_k`, `norm_added_q`, `norm_added_k`) exactly as the
   stock processor does.
3. Concatenate — text tokens occupy indices `0..T-1`, image tokens `T..T+4095`.
4. Apply rotary embeddings identically, or the captured attention and the continuing generation
   diverge.
5. Explicitly form `softmax(qkᵀ/√d)` instead of dispatching to the fused kernel. **This is the
   matrix we capture**, sliced to image-rows × target-text-token-column.
6. Continue with `attn_probs @ value` and the stock output projections, so denoising is unaffected.

Scoped to the 19 `FluxTransformerBlock` double blocks. The 38 single blocks receive
`encoder_hidden_states=None` — text is concatenated *before* the attention call, so there is no
clean text/image boundary to slice on. Out of scope by design, and a stated limitation.

Because T5 tokens are what actually enter the joint attention sequence (CLIP contributes only a
pooled vector), token alignment runs against `tokenizer_2`. This surfaced a phrase-matching bug that
matters later: the manifest's attribute string (`"yellow helmet"`) does not literally substring-match
the prompt text (`"...yellow bike helmet..."`). It broke T5 token lookup *and*, independently,
`exp4_positional_baseline.py` on ~40 rows.

**Verification before we trusted the data:** all 105 images present; every detected row's attribute
count matches `n`; 0/301 rows still carry the old `"unavailable"` placeholder; seeds match the SDXL
manifest's pinned seeds on every overlapping `prompt_id`; full test suite (190 tests) passes.

---

## 4. The FLUX five-experiment results

Run for all three annotators. These are the numbers that looked, initially, like a decisive success.

**Experiment 1 — accuracy vs. 1/n chance.**

| Annotator | n=2 | n=3 | n=4 | Overall | n scored |
|---|---|---|---|---|---|
| chayan | 97.5% | 92.9% | 71.5% | 84.6% | 273 |
| akhil | 97.5% | 94.3% | 72.5% | 85.5% | 269 |
| grace | 97.4% | 95.6% | 73.1% | 86.0% | 265 |

Every stratum, every annotator, binomial p < 0.0001 against 50/33.3/25%.

**Experiment 2 — early window vs. full trajectory.** Effectively identical; McNemar p = 1.0 for all
three annotators, at most one discordant row out of ~270. FLUX's binding decision appears settled
within the first half of denoising — consistent with the Prompt-to-Prompt / Attend-and-Excite
account of early steps deciding layout.

**Experiment 3 — attention randomization.** Real beats cross-item-scrambled overwhelmingly:
McNemar p = 1.77e-33 (chayan, akhil), 2.22e-29 (grace). Median scrambled accuracy sits at
0.500 / 0.329 / 0.252 against chance of 0.500 / 0.333 / 0.250 — the control lands exactly where it
should.

**Experiment 4 — nearest-subject-noun baseline.** Metric 84.6–86.0% vs. baseline 77.7–78.9%,
McNemar p = 0.013–0.027. The metric wins.

**Experiment 5 — count-clean subset.** Zero count-broken rows for any annotator; the filtered and
unfiltered tables are identical. A genuine contrast with SDXL's 96/306 exclusions.

Taken at face value this is a clean sweep: beats chance, survives falsification, beats the
baseline, on ground truth with κ ≈ 0.95. That is how it was first reported.

---

## 5. The reframe

### 5.1 The baseline nobody ran

`intended_subject == human_label` — assume the model rendered what the prompt asked. No image, no
attention, no computation.

| Model | Annotator | n | Attention | Nearest-noun | **Prompt-obeyed** | McNemar (attn vs prompt-obeyed) |
|---|---|---|---|---|---|---|
| SDXL | chayan | 35 | 45.7% | 74.3% | **74.3%** | p = 0.031 |
| SDXL | akhil | 55 | 40.0% | 61.8% | **61.8%** | p = 0.023 |
| SDXL | grace | 63 | 54.0% | 76.2% | **76.2%** | p = 0.013 |
| FLUX | chayan | 273 | 84.6% | 77.7% | **94.1%** | p = 1.29e-05 |
| FLUX | akhil | 269 | 85.5% | 77.7% | **94.4%** | p = 6.96e-05 |
| FLUX | grace | 265 | 86.0% | 78.9% | **95.8%** | p = 1.29e-05 |

Six comparisons, six significant losses for attention. On SDXL it loses by 20–29 points; on FLUX by
about 10.

This is the paper's central table.

### 5.2 Experiment 4's FLUX "win" is partly an artifact

Notice that on SDXL the nearest-noun and prompt-obeyed columns are *identical*, while on FLUX they
differ by 16 points. This is not a coincidence, and the Part A design doc pre-registered half of it:
on SDXL's prompt template, nearest-preceding-subject-noun equals `intended_subject` on **0/306**
rows of divergence — the template never lets a second subject intervene before an attribute, so the
baseline is mathematically "guess the intended pairing."

FLUX's prompts were reworded for FLUX's phrasing needs — *"two people standing side by side, on the
left a barista in a red apron, on the right a man wearing a cycling jersey in a yellow bike
helmet."* That rewording introduces intervening nouns ("a man wearing a cycling jersey"), and the
nearest-noun heuristic now diverges from `intended_subject` on **46/299 rows (15.4%)**.

The consequence: FLUX's positional baseline is *weaker* (77.7%) than SDXL's (74.3% of a much harder
set, but exactly equal to prompt-obeyed there) not because attention got better, but because the
prompt rewording broke the heuristic. **Experiment 4's FLUX result is a win over a degraded
baseline.** Against the undegraded version of the same idea, attention loses by 10 points at
p ≈ 1e-5.

This needs to be stated plainly in the paper. It is exactly the kind of thing reviewers find, and
finding it ourselves is worth more than hoping they don't.

### 5.3 Where a faithfulness metric would have to earn its keep

If a model always obeyed its prompt, no faithfulness metric would be needed — you could read the
prompt. Metrics earn their existence on the rows where prompt and image **disagree**. So: restrict
to rows where `human_label != intended_subject`, and ask whether attention predicts the *rendered*
outcome.

| Model | Annotator | Mis-bound rows | Attention predicts rendered | Chance | p (one-sided) |
|---|---|---|---|---|---|
| SDXL | chayan | 9 | 44.4% | 30.6% | 0.283 |
| SDXL | akhil | 21 | 28.6% | 32.5% | 0.726 |
| SDXL | grace | 15 | 46.7% | 34.4% | 0.231 |
| FLUX | chayan | 16 | 31.2% | 26.0% | 0.408 |
| FLUX | akhil | 15 | 40.0% | 25.6% | 0.161 |
| FLUX | grace | 11 | 45.5% | 25.0% | 0.115 |

Pooled across both models, per annotator:

| Annotator | n | Hit rate | 95% CI | Chance | p |
|---|---|---|---|---|---|
| chayan | 25 | 36.0% | 18.0 – 57.5% | 27.7% | 0.235 |
| akhil | 36 | 33.3% | 18.6 – 51.0% | 29.6% | 0.372 |
| grace | 26 | 46.2% | 26.6 – 66.6% | 30.4% | 0.067 |

**Read this honestly.** These are non-rejections, not proven nulls. The confidence intervals are
enormous — grace's interval spans 26.6% to 66.6%. We cannot currently distinguish "attention is at
chance on disobeyed rows" from "attention retains a modest signal we lack the power to detect." What
we *can* say is that six independent tests all failed to find the effect, and the point estimates
cluster near chance rather than near the 85% headline accuracy.

The mechanism this suggests is straightforward: attention's high headline accuracy is carried by the
94% of rows where the model obeyed the prompt, on which "predict the intended pairing" is
automatically correct. Remove that scaffolding and the signal we care about is not clearly there.

### 5.4 Experiment 3 is weaker than it looks

Reading `exp3_attention_scramble.py`'s implementation: `scramble_predict` takes a donor item's score
*values*, **shuffles them**, and assigns them to the target's subject slots. The donor's identity
and subject names are discarded — only the magnitudes are borrowed, and then permuted.

That makes the scrambled condition, in effect, a **random assignment**. Which is why its accuracy
lands precisely on 1/n in every stratum. And that means Experiment 3 is asking nearly the same
question as Experiment 1 ("does attention beat random?"), not an independent one.

The memo's original Tier-1 design was sharper: permute *which token's map* feeds each attribute's
score, so "red apron" is scored against "yellow helmet"'s map. That preserves the magnitude
structure of real attention while breaking the token↔attribute correspondence — a much more
demanding control. The Part A design doc rejected it because it degenerates at n = 2 (a 2-element
permutation is a forced swap, so scrambled accuracy = 1 − real by arithmetic). That objection is
correct **at n = 2** and does not apply at n = 3 or n = 4.

Recommendation: reinstate the within-item token permutation as an additional control at n ≥ 3. It
is cheap — pure re-analysis of already-captured `model_scores` — and it is the control that would
actually distinguish "attention encodes binding" from "attention encodes something."

---

## 6. Convergent evidence from the chain track

Two independent results in this repo point the same direction, which materially strengthens the
paper.

**Part C Step 6 ablation (chain metric, SD1.5).** Metric B scores a chain step by intersecting a
CLIPSeg "delta mask" (what is newly present at this step) with a thresholded attention map. Ablating
the attention component:

| Score used | real vs shuffled (clustered p) | real vs substituted |
|---|---|---|
| `iou` (published) | 0.0039 | 0.0039 |
| `delta_area` — no attention at all | 0.0117 | 0.0039 |
| `iou_random_attn` — random noise attention | 0.0117 | 0.0039 |

Replacing attention with **literal random noise reproduces the significance.** The chain result's
two strongest numbers were validating the image-derived delta mask, not the attention map. We
recorded this at the time as "the paper's framing needs this correction."

**The box-size baseline (§3.1).** On SD1.5 a "biggest box" heuristic beat the attention metric; on
SDXL the metric's edge never reached significance across two sample sizes.

So we now have three independent lines — chain ablation, box baseline, prompt baseline — all
saying that raw cross-attention is not carrying the discriminative weight we attributed to it. That
convergence is the strongest thing the paper has.

---

## 7. Literature positioning

### 7.1 The direct precedent nobody in this project has cited yet

The single most useful framing move available: **NLP already had this argument.**

- **Jain & Wallace, "Attention is not Explanation" (NAACL 2019)** — showed attention weights in
  text classifiers correlate poorly with gradient-based importance, and that adversarial attention
  distributions can produce identical predictions. The core move is ours: test the attention story
  against an alternative that explains the data equally well.
- **Wiegreffe & Pinter, "Attention is not not Explanation" (EMNLP 2019)** — the rebuttal, which
  crucially argued the original paper's controls were under-specified and that *what baseline you
  test against* determines the conclusion.
- **Serrano & Smith, "Is Attention Interpretable?" (ACL 2019)** — attention magnitude is a poor
  guide to which representations actually matter, established by erasure.

Our contribution reads naturally as the **diffusion/vision analogue of this debate**, with two
advantages the NLP work did not have: human ground truth on the actual output, and an architecture
comparison (UNet vs MMDiT). The framing writes itself — *the vision community adopted cross-attention
as an interpretability primitive without running the baseline check that settled the NLP debate.*

**Also directly relevant:** *Sanity Checks for Saliency Maps* (Adebayo et al., NeurIPS 2018), already
in the project's reading list and cited in the memo — trusted saliency methods failed randomization
tests. Our Experiment 3 is a sanity check in that tradition; §5.4 argues ours was too weak, which is
itself a methodological point worth making.

### 7.2 What we are building on

- **DAAM** (Tang et al.) — cross-attention as word-to-region attribution in SD; validated with
  attention-mask overlap, which is why "attention–mask IoU" is not itself a novel technique.
- **Attend-and-Excite** (Chefer et al.) — established cross-attention is causally tied to subject
  generation, by intervening on it. This is the foundation our measurement inverts, and the source
  of the early-window claim Experiment 2 tests.
- **Prompt-to-Prompt** (Hertz et al.) — cross-attention maps control spatial layout; source of the
  "layout is decided early" claim.

Note the tension worth addressing head-on in the paper: A&E and P2P show attention is *causally
manipulable*. Our result says it is not a *reliable readout*. Both can be true — intervening on a
variable can change an outcome even when observing that variable does not predict the outcome
well. Making that distinction crisply is a genuine conceptual contribution and pre-empts the most
likely reviewer objection ("but A&E proved attention works").

### 7.3 The competitive landscape (from the project's 2026-07-22 literature check)

- **T2I-CompBench / CompBench++** and **VQAScore** are the established SOTA line for one-shot
  attribute binding. We do not beat them and should not claim to — different instrument. They are
  *judge-based*; ours is *internal-state-based*. The honest positioning is that we are auditing an
  internal signal, not competing on benchmark accuracy.
- The field reportedly still regards VQA-judge metrics as unreliable for attribute binding and leans
  on human eval — which is why our human anchor set is the right validation currency.
- **ConceptAttention** and 2025 causal/norm-based attribution work reportedly moved past raw
  cross-attention as a blunt signal. **Our data supplies the empirical justification for that move**,
  with human ground truth. That is a natural citation relationship and a natural place for our
  contribution to sit.
- Chain/lock-confound faithfulness (Track 1's territory) has no dominant competitor; closest
  adjacent work is 2025 (BPM, ComplexBench-Edit) and is not attention-based.

> **Citation hygiene warning.** The items in §7.3 come from this project's internal literature
> notes, not from a verified search in this session. ConceptAttention, FreeMask, BPM, and
> ComplexBench-Edit in particular need their titles, venues, authors, and claims checked against the
> actual papers before anything is written down. The §7.1 and §7.2 items I am confident about, but
> verify years and venues anyway. **Do not let any citation reach a draft unverified.**

---

## 8. The paper

### 8.1 Thesis

> Raw cross-attention in text-to-image diffusion models encodes the *prompt's intended*
> attribute-subject binding rather than the *image's realized* binding. It therefore appears highly
> accurate on benchmarks where models usually obey their prompts, while adding no measurable value
> over simply reading the prompt — and providing no reliable signal precisely on the failure cases a
> faithfulness metric exists to detect.

### 8.2 Claims, and what supports each

| # | Claim | Evidence | Strength |
|---|---|---|---|
| C1 | Attention-based binding prediction beats chance on both architectures | Exp 1, both models, 3 annotators | **Strong** |
| C2 | It nonetheless loses to a prompt-only baseline, universally | §5.1, 6/6 tests, p < 0.05 | **Strong** |
| C3 | On prompt-violating rows, attention is not distinguishable from chance | §5.3, 6/6 non-rejections | **Weak — underpowered** |
| C4 | Randomization controls as usually run cannot separate "encodes binding" from "encodes anything" | §5.4, code analysis + scrambled ≈ 1/n | **Strong (analytic)** |
| C5 | Attention quality is architecture-dependent (MMDiT ≫ UNet) but the conclusion is not | §3.1 vs §4; Exp 3 fails on SDXL, passes on FLUX | **Moderate** |
| C6 | Replacing attention with noise reproduces chain-metric significance | Part C Step 6 | **Strong** |
| C7 | Extracting interpretable attention from MMDiT requires manual softmax recomputation | `flux_attention_capture.py` | **Methods contribution** |

C2 is the headline. C3 is the mechanism and is currently the weakest link — §9.1 exists to fix it.

### 8.3 Structure

1. **Introduction** — cross-attention is the field's default interpretability primitive for
   diffusion; it has never been tested against an informed baseline.
2. **Related work** — the NLP attention-interpretability debate; DAAM / A&E / P2P; the
   judge-metric line; the causal-vs-observational distinction (§7.2).
3. **Method** — anchor-set protocol, the metric, the MMDiT capture (C7).
4. **Experiments** — the five-experiment battery on both architectures.
5. **The baseline analysis** — §5.1–5.3. The paper's core.
6. **Control adequacy** — §5.4. Why standard randomization controls under-test.
7. **Convergent evidence** — §6, the chain ablation.
8. **Discussion** — what attention is good for (steering, per A&E) vs. what it is not good for
   (scoring); implications for work that uses attention-derived masks as pseudo-ground-truth.
9. **Limitations** — §9.

### 8.4 Venue read

This is an empirical/analysis paper with a negative headline. Those are publishable when the
methodology is airtight and the finding is actionable, but they are held to a *higher* evidentiary
bar than positive results, because "we didn't find it" and "it isn't there" are easy to confuse.
Concretely: C3 must be properly powered before submission. A workshop on interpretability or
evaluation is a realistic first target; a main-track submission needs §9.1 done.

---

## 9. What is missing

### 9.1 The blocking experiment: a hard prompt set

**The binding constraint is that our models are too obedient.** FLUX binds correctly on ~94% of
rows, which leaves 11–16 discriminative rows per annotator. Every underpowered result in this
document traces to that.

Build a prompt set where models fail ~50% of the time:
- Higher subject counts (n = 4–6).
- Attribute–subject pairings that fight object priors (a *chef* in a *cycling helmet*).
- Confusable attributes within a prompt (two different-colored aprons rather than an apron and a
  helmet).
- Near-duplicate subjects (two chefs distinguished only by attribute).

Target ≥ 150 rows where `human_label != intended_subject`. That converts C3 from a wide-CI
non-rejection into the paper's central, properly-powered experiment. **Nothing else on this list
matters as much.**

### 9.2 Cheap re-analyses (no GPU, days not weeks)

- **Within-item token permutation at n ≥ 3** (§5.4) — the control the battery should have had. Pure
  re-analysis of existing `model_scores`.
- **Discriminant validity on FLUX** — `discriminant_validity_check.py` exists and has only ever been
  run on SDXL. Needs `recompute_boxes.py` against the FLUX images first (CPU, cached). A reviewer
  will ask.
- **Part C robustness battery on FLUX** — Holm correction across the five experiments, leave-one-
  prompt-out, RNG sweep. Currently SD1.5-chain-only. The FLUX p-values in §4 are single runs at a
  frozen operating point.
- **Majority-vote / consensus ground truth** — we have three annotators at κ ≈ 0.95 and currently
  analyze them separately. A consensus label set would tighten every interval slightly.

### 9.3 Deferred but valuable

- **VQAScore correlation** — how does a judge-based metric do on the same rows, especially the
  prompt-violating ones? Directly positions us against the SOTA line.
- **Causal intervention via A&E steering** — the memo's Tier-2 #4. Now doubly interesting given the
  causal-vs-observational distinction in §7.2: if steering attention changes the image but observing
  attention doesn't predict it, that is a sharp, quotable result.
- **The 38 FLUX single blocks** — out of scope in the current capture. If the double-block result is
  challenged, this is the first place to look.

### 9.4 Known limitations to disclose

- Prompt vocabulary is narrow and templated (occupations × clothing attributes). External validity
  to natural prompts is untested.
- Detection rate falls with subject count (92% / 67% / 58% at n = 2/3/4 on the chain data), so
  surviving samples skew toward easier items.
- SDXL ground truth is κ = 0.682, below target. FLUX's is fine.
- Ownership-by-bounding-box-containment is itself an approximation; overlapping subjects are
  genuinely ambiguous.
- All results are for the early-window aggregation on double blocks only.

---

## 10. Reproduction

All commands from `ssa/anchor_set/`. Requires `numpy`, `scipy`, `pandas` (no GPU, no torch).

```bash
# Five-experiment battery
python3 run_five_experiments.py --artifacts-dir artifacts_flux --annotator chayan
python3 run_five_experiments.py --artifacts-dir artifacts_sdxl --annotator akhil

# Agreement + inter-rater kappa
python3 analyze_agreement.py --artifacts-dir artifacts_flux \
        --annotator chayan --compare-annotator grace

# Test suite (190 tests, 1 GPU-only skip)
python3 -m pytest tests/ -q
```

The §5.1 and §5.3 analyses are not yet committed as scripts — they were run inline for this
document. **They should be turned into `exp6_prompt_baseline.py` and
`exp7_misbound_subset.py` and tested**, both because they are now the paper's core results and
because nothing in this repo should be load-bearing without a test. That is the first code task.

### Key files

| Path | What |
|---|---|
| `ssa/anchor_set/artifacts_flux/` | FLUX images, manifest, 3× labels/counts, results |
| `ssa/anchor_set/artifacts_sdxl/` | SDXL equivalent |
| `ssa/anchor_set/flux_attention_capture.py` | MMDiT capture (C7) |
| `ssa/anchor_set/anchor_common.py` | Shared scoring/agreement logic |
| `ssa/anchor_set/exp{1..5}_*.py` | The battery |
| `docs/superpowers/specs/2026-07-31-flux-attention-hook-design.md` | Capture design |
| `docs/part-a-five-experiment-battery-design.md` | Battery pre-registration |
| `docs/anchor-set-labeling-protocol.md` | Labeling protocol |
| `proposal/SSA-Metric-Memo.md` | Original falsification battery (§8) |
| `pi_level_experiment/RESULTS.md` | Chain track, incl. Part C Step 6 ablation |

---

## 11. Bottom line

We built a metric, validated it against chance, put it through a falsification battery, and it
passed. Then we ran the one baseline the pre-registered plan named but never executed — "rule out
the boring explanations" — and it lost, on every model and every annotator.

That is not a failed project. The metric literature's problem is precisely that this check is rarely
run, and we have the apparatus to run it properly: two architectures, three annotators at
κ ≈ 0.95, a novel MMDiT capture, and three independent lines of convergent evidence. The paper is
about what cross-attention actually encodes, and the answer — the prompt, not the picture — is more
interesting than another metric would have been.

The work between here and a submission is mostly one experiment: a prompt set hard enough that the
models disobey often enough to measure. Everything else is re-analysis we can do this week.
