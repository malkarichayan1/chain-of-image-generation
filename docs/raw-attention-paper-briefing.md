# Raw Cross-Attention Tracks Intent, Not Realization

**A ground-up briefing for the paper**
CPGA / CoIG Faithfulness Project · Prepared 2026-08-03 · Revised 2026-08-07 · Branch `main`
(the repo was consolidated onto `main` 2026-08-06 — see `CLAUDE.md`)

---

## 0. Executive summary

We set out to build a faithfulness metric that reads a diffusion model's own cross-attention to
decide which subject an attribute is bound to ("is the red apron on the barista or the cyclist?").
Across two model families, four human annotators (Chayan, Akhil, Grace, Pranav), and roughly 1,080
human-labeled attribute judgments across four anchor sets, the metric works in the narrow sense
that it beats random guessing — and fails in the sense that matters.

The finding that organizes the paper is this:

> **A trivial baseline that never looks at the image — "assume the model rendered what the prompt
> asked for" — significantly outperforms the attention-based metric on every model and every
> annotator we have tested.** Ten independent comparisons (six on the original prompt sets, four
> on a harder retest), ten losses for attention, all at p < 0.05, most at p < 1e-4.

And the follow-through that explains it:

> On the subset of rows where the model *disobeyed* the prompt — the only rows where a faithfulness
> metric can add value over simply reading the prompt — attention's accuracy at predicting the
> **rendered** outcome was not distinguishable from chance in the original six (underpowered)
> tests, and remains only weakly, marginally above chance in a properly-powered retest (§5.5).

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
FLUX, p ≈ 1e-5). The mechanistic follow-up (attention at chance on disobeyed rows) was originally
directionally consistent across all six tests but **underpowered** — 11–36 rows per test, wide
confidence intervals. A hard-prompt-set retest (§5.5, executed 2026-08-07) raised that to 60–73
misbound rows per annotator and pushed the pooled test into real, if modest, significance
(p ≈ 0.03–0.05 on 3 of 4 label sets) — genuine progress, though short of the ≥150-row target §9.1
originally set. The same pass also ran the sharper falsification control §5.4 recommended but
never executed (§5.6): a new, unanticipated finding is that attention's attribute-specificity is
strong on easy images and degrades on hard ones — exactly where the paper's central claim needs it
most.

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
| `artifacts_flux_hard/` | FLUX.1-dev, harder prompts | 83 detected (100 attempted) | 409 | Akhil, Grace, Pranav | **0.889–0.912** |

Prompts in the first three sets are templated across three strata by subject count: n = 2, 3, 4
(chance = 50%, 33.3%, 25%). The FLUX-hard set (§5.5, added 2026-08-07) uses n = 4, 5, 6 (chance =
25%, 20%, 16.7%) specifically to lower the chance floor and raise binding complexity.

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

**Update, 2026-08-07 — the same check, closed for FLUX.** This discriminant-validity check had
only ever run on SDXL/SD1.5. `recompute_boxes.py` recovered subject boxes for all 103 detected
FLUX images (CPU, Mask R-CNN + CLIP, no re-generation), and the result is unambiguous where
SD1.5/SDXL were marginal: `predicted_owner` beats "always guess the biggest box" by a wide margin
(84.6–86.0% vs. 35.3–36.2%, McNemar p ≈ 1e-37, all three annotators), the metric's own pick lands
on the biggest box at a rate indistinguishable from chance (34.7% vs. 34.3%, p = 0.47), the
anchor set's own construct validity is clean too — `intended_subject` isn't biased toward the
biggest box either (34.3% vs. 34.3% chance, p = 0.52) — and attention margin doesn't track
box-size dominance (Spearman r = -0.10, p = 0.07). On FLUX, `predicted_owner` is not a
box-geometry artifact — one plausible "boring explanation" is closed.

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

### 5.5 The hard-prompt-set retest (executed 2026-08-07)

§9.1 named this as the single most important remaining experiment: FLUX obeys too well (~94% of
rows) to properly power the §5.3 test. `build_hard_prompts.py` generated 100 prompts at n = 4/5/6
(IDs 300–399) designed to fight the model harder — confusable same-type attributes (two aprons,
two helmets, two hats, two gloves, each pair a different color), attribute–subject prior fights (a
*chef* in a *cycling helmet*), and near-duplicate subjects (chef + baker, nurse + doctor, barista +
waiter, cyclist + biker). 83/100 images detected the right subject count; akhil, grace, and pranav
each labeled all 409 rows (chayan's pass, 4 rows, is excluded throughout — too incomplete to use).
Inter-rater κ = 0.889–0.912 — still excellent agreement, a shade below the original set's
0.954/0.958, consistent with these images genuinely being harder to read. A majority-vote
consensus label set (`build_consensus_labels.py`, new) resolved 357/409 rows unanimously, 48/409 by
2-of-3 majority, and left only 4/409 with no consensus.

**Did it hit the target? Partially.**

| Metric | Original FLUX (§4–§5.3) | FLUX-hard |
|---|---|---|
| Prompt-obeyed rate (`intended_subject == human_label`) | ~94–96% | **~77–80%** |
| Misbound rows available for the C3 test (per annotator) | 11–16 | **60–73** (62 on consensus) |
| Attention accuracy overall | ~85% | **~42–44%** |

Obedience dropped meaningfully — from ~95% to ~80% — and misbound rows jumped 4–6×. But §9.1's
target was ≥150 misbound rows / ~50% failure, and FLUX.1-dev turned out to be more obedient than
the design assumed even under prior-fight and near-duplicate-subject pressure. The prompt
engineering worked directionally; it did not fully close the power gap.

**C2 replicates, decisively, on the harder set.** Re-running §5.1's baseline comparison:

| Annotator | n (scored) | Attention | Prompt-obeyed | McNemar |
|---|---|---|---|---|
| akhil | 316 | 42.7% | **80.1%** | p ≈ 1.3e-23 |
| grace | 308 | 41.9% | **80.2%** | p ≈ 5.5e-24 |
| pranav | 319 | 42.3% | **77.1%** | p ≈ 7.3e-21 |
| consensus | 311 | 42.8% | **80.1%** | p ≈ 4.1e-23 |

The gap is if anything wider than on the original set (§5.1: ~10 points on FLUX; here, ~37–38
points). Harder images do not help attention — they help the case for the prompt baseline.

**C3 crosses into real, if modest, significance for the first time.** Re-running §5.3's
misbound-subset test, the per-stratum tests still individually don't clear p < 0.05 (small n once
split by n=4/5/6) — but a pooled one-sided test across strata (Poisson-binomial over each row's own
1/n chance; an exploratory combination of our own construction, **not** Holm-corrected against the
rest of the battery) gives:

| Annotator | n (misbound) | Correct | Accuracy | Mean chance | Pooled p (one-sided) |
|---|---|---|---|---|---|
| akhil | 63 | 18 | 28.6% | 18.5% | **0.034** |
| grace | 61 | 17 | 27.9% | 18.7% | 0.052 |
| pranav | 73 | 21 | 28.8% | 18.9% | **0.027** |
| consensus | 62 | 18 | 29.0% | 18.5% | **0.029** |

C3 moves from "wide-CI non-rejection" (§5.3's original six tests, all p > 0.1) to "weak positive,
borderline significant" on three of four label sets. This is genuine movement — not the
fully-powered central experiment §9.1 envisioned, but no longer indistinguishable from noise
either. If the paper wants a harder result here, the next lever is probably more subjects per
image (n = 7+), not more prompts at the current n = 4–6 difficulty — FLUX's obedience floor
appears to sit closer to ~80% than ~50% even under real compositional pressure.

### 5.6 The within-item token-permutation control, executed

§5.4 recommended reinstating the sharper falsification control the original memo specified —
permuting *which attribute's own attention map* feeds a prediction, within one image, rather than
scrambling values across images. That control now exists (`exp3b_within_item_permutation.py`) and
has been run against both the original FLUX set and FLUX-hard.

The result is a genuinely new finding, not anticipated by §5.4's diagnosis. **On the original FLUX
set, permuted accuracy falls significantly *below* chance**: median 0.029 against chance 0.333 at
n = 3, and 0.10 against 0.25 at n = 4. A paired McNemar between real and permuted correctness is
overwhelming (p ≈ 1e-27 to 1e-29 across all three annotators, real winning 148–149 discordant rows
to permuted's 15–18). That is not the "no better than chance" result a clean falsification control
usually reports — it means attention on FLUX is so decisively attribute-specific that handing the
metric a *different real attribute's* attention map from the *same image* actively misleads it,
more often than a coin flip would.

**On FLUX-hard, that specificity is markedly weaker.** Permuted accuracy sits close to chance at
n = 4 and n = 5 (falsification-clean fraction 0.625–0.635 — roughly a third of seeds *do* detect a
below-chance effect, but most don't) and is indistinguishable from chance at n = 6 (0.965 clean
fraction). Real still beats permuted (McNemar p ≈ 4.2e-13), so some attribute specificity
survives, but far less decisively than on the easier set.

**Read together, this is a two-sided update to C4.** The original §5.4 critique was analytic:
existing controls could not separate "attention encodes binding" from "attention encodes
anything." This new control answers that question directly, for the first time — on easy images,
attention plainly does encode attribute-specific content, not generic salience (a real rebuttal to
the weakest version of that worry). But that same specificity is precisely what erodes on harder
images, in the same direction and over the same population where C3 needs attention to be
informative. It does not change C1–C3's substance; it supplies a mechanistic explanation for why
C3's effect stays small even in the better-powered retest — the underlying signal itself gets
noisier exactly as the images get harder.

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

**A fourth line was attempted and found blocked, not negative.** Extending Part C's Holm/leave-
one-out/RNG-sweep robustness battery (`pi_level_experiment/rng_sweep.py` /
`analyze_results.py`) to FLUX chains was checked 2026-08-07: `pi_level_experiment/` has zero FLUX
chain data anywhere, and `generate_chains.py` has never been run on FLUX. Extending it needs a new
Kaggle GPU chain-generation round-trip (the full Stage 1/2/3 pipeline), not a re-analysis — flagged
in §9.2 as still blocked rather than silently dropped.

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
> over simply reading the prompt — and providing, at best, a weak and marginally-significant signal
> on the failure cases a faithfulness metric exists to detect.

*(Updated 2026-08-07: the properly-powered retest in §5.5 found ~29% accuracy against ~18.5%
chance on prompt-violating rows — pooled p ≈ 0.03–0.05 on 3 of 4 label sets. That is a real,
disclosable effect, not zero — but it is far below the ~85% headline accuracy and not something a
faithfulness metric could be built on. "No reliable signal" was the honest read of the original,
underpowered six-test null; "weak, marginal signal" is the honest read now that the test has
power. Either way, the core argument — attention adds nothing over the prompt baseline (C2, still
6/6 at p<0.05) and what little it retains on the hard cases is small and architecture-dependent
(§5.6) — is unchanged.)*

### 8.2 Claims, and what supports each

| # | Claim | Evidence | Strength |
|---|---|---|---|
| C1 | Attention-based binding prediction beats chance on both architectures | Exp 1, both models, 3 annotators | **Strong** |
| C2 | It nonetheless loses to a prompt-only baseline, universally | §5.1, §5.5: 10/10 tests (6 original + 4 hard-set), p < 0.05 | **Strong** |
| C3 | On prompt-violating rows, attention beats chance only weakly | §5.3 (original, underpowered) + §5.5 (hard-set retest: pooled p≈0.03–0.05, 3/4 label sets) | **Weak-positive — properly powered for the first time, borderline significant** |
| C4 | Randomization controls as usually run cannot separate "encodes binding" from "encodes anything" — and when the sharper control IS run, attention turns out to be genuinely attribute-specific on easy images | §5.4 (analytic) + §5.6 (executed: permuted accuracy below chance on FLUX, McNemar p≈1e-27–1e-29) | **Strong (analytic, now also empirical)** |
| C5 | Attention quality is architecture-dependent (MMDiT ≫ UNet) but the conclusion is not | §3.1 vs §4; Exp 3 fails on SDXL, passes on FLUX | **Moderate** |
| C6 | Replacing attention with noise reproduces chain-metric significance | Part C Step 6 | **Strong** |
| C7 | Extracting interpretable attention from MMDiT requires manual softmax recomputation | `flux_attention_capture.py` | **Methods contribution** |
| C8 | Attention's attribute-specificity itself degrades on harder images — exactly where C3 needs it most | §5.6: below-chance permuted accuracy on FLUX vs. near-chance on FLUX-hard | **Moderate — one dataset pair, directionally clean, new 2026-08-07** |

C2 is the headline, and now the more thoroughly tested one (10/10 comparisons across both prompt
sets). C3 was the weakest link; the §9.1 hard-prompt-set retest (§5.5) materially improved its
power and moved it into real-but-modest significance, though not to the fully-powered target
originally set — see §9 for what, if anything, is still worth doing about it.

### 8.3 Structure

1. **Introduction** — cross-attention is the field's default interpretability primitive for
   diffusion; it has never been tested against an informed baseline.
2. **Related work** — the NLP attention-interpretability debate; DAAM / A&E / P2P; the
   judge-metric line; the causal-vs-observational distinction (§7.2).
3. **Method** — anchor-set protocol, the metric, the MMDiT capture (C7).
4. **Experiments** — the five-experiment battery on both architectures, plus the hard-prompt-set
   retest (§5.5).
5. **The baseline analysis** — §5.1–5.3, §5.5. The paper's core.
6. **Control adequacy** — §5.4, §5.6. Why standard randomization controls under-test, and what the
   sharper control found when it was actually run (attention IS attribute-specific on easy images,
   less so on hard ones — C8).
7. **Convergent evidence** — §6, the chain ablation, plus the closed FLUX discriminant-validity
   check (§3.1).
8. **Discussion** — what attention is good for (steering, per A&E) vs. what it is not good for
   (scoring); implications for work that uses attention-derived masks as pseudo-ground-truth.
9. **Limitations** — §9.

### 8.4 Venue read

This is an empirical/analysis paper with a negative headline. Those are publishable when the
methodology is airtight and the finding is actionable, but they are held to a *higher* evidentiary
bar than positive results, because "we didn't find it" and "it isn't there" are easy to confuse.
Concretely: C3 needed to be properly powered before submission. §9.1's retest (§5.5, executed
2026-08-07) materially improved that — pooled p ≈ 0.03–0.05 on 3/4 label sets — but did not reach
the original ≥150-row target, and the effect size (~10 points over chance) is modest. A workshop
on interpretability or evaluation looks solidly supported by the current evidence; whether a
main-track submission needs a further-hardened prompt set (§5.5's closing note: more subjects per
image, not more prompts at the current difficulty) is now the open editorial call, not a hard
blocker.

---

## 9. What is missing

### 9.1 The blocking experiment: a hard prompt set — EXECUTED 2026-08-07, partial success

**The original binding constraint was that our models were too obedient.** FLUX bound correctly on
~94% of rows, leaving 11–16 discriminative rows per annotator, and every underpowered result in
this document traced to that.

`build_hard_prompts.py` built exactly the prompt set specified: higher subject counts (n = 4–6),
attribute–subject pairings that fight object priors (a *chef* in a *cycling helmet*), confusable
attributes within a prompt (two different-colored aprons, not an apron and a helmet), and
near-duplicate subjects (chef + baker, nurse + doctor, barista + waiter, cyclist + biker). akhil,
grace, and pranav triple-labeled all 409 rows. Full results: §5.5.

**It worked, partially.** Misbound rows went from 11–16 to 60–73 per annotator (62 on consensus) —
a 4–6× increase — and C3's significance moved from "non-rejection, p > 0.1 on all six original
tests" to "weak positive, pooled p ≈ 0.03–0.05 on 3/4 label sets." **It did not hit the ≥150-row /
~50%-failure target**: FLUX.1-dev's prompt-obedience only fell to ~80%, not ~50%, even under this
level of compositional pressure. The paper now has a better-powered C3 result, not the
fully-powered one originally envisioned.

**If more power is still wanted, the next lever is probably n = 7+ subjects per image**, not more
prompts at the current n = 4–6 difficulty — see §5.5's closing paragraph.

### 9.2 Cheap re-analyses (no GPU, days not weeks) — status as of 2026-08-07

- ~~**Within-item token permutation at n ≥ 3**~~ **DONE** — `exp3b_within_item_permutation.py`.
  Results in §5.6: a genuinely new finding (permuted accuracy falls below chance on easy FLUX
  images, much weaker on hard ones — C8), not just the falsification control the battery lacked.
- ~~**Discriminant validity on FLUX**~~ **DONE** — `recompute_boxes.py` +
  `discriminant_validity_check.py` ran clean on all 103 detected FLUX images, all 3 annotators.
  Results in §3.1's update. No box-geometry artifact.
- **Part C robustness battery on FLUX — still BLOCKED, not attempted.** Checked 2026-08-07:
  `pi_level_experiment/` (the chain track this refers to — the Holm/leave-one-out/RNG-sweep
  machinery in `rng_sweep.py`/`analyze_results.py`) has zero FLUX chain data or FLUX references
  anywhere; `generate_chains.py` has never been run on FLUX. The re-analysis tooling being GPU-free
  doesn't help when the underlying FLUX chains don't exist — generating them is a full Kaggle GPU
  round-trip (the Stage 1/2/3 chain pipeline), not a cheap re-analysis. Genuinely deferred, not
  silently dropped (§6).
- ~~**Majority-vote / consensus ground truth**~~ **DONE for FLUX-hard** — `build_consensus_labels.py`
  (new): 357/409 unanimous, 48/409 majority, 4/409 no-consensus (§5.5). Not yet run for the
  original `artifacts_flux/` (chayan + akhil + grace) — same script, minutes of work if wanted.

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
- The §5.5 hard-prompt-set C3 result rests on a pooled (Poisson-binomial, cross-strata) test that
  is exploratory — our own construction, not pre-registered, and not corrected for the other tests
  in the battery. `exp7_misbound_subset.py`'s own per-stratum output still does not clear p < 0.05
  on any individual stratum for any label set.
- FLUX-hard's prompt-obedience rate (~80%) fell short of the ~50% design target; the harder
  prompts moved the needle without fully closing the power gap (§5.5).

---

## 10. Reproduction

All commands from `ssa/anchor_set/`. Requires `numpy`, `scipy`, `pandas` (no GPU, no torch).

```bash
# Five-experiment battery
python3 run_five_experiments.py --artifacts-dir artifacts_flux --annotator chayan
python3 run_five_experiments.py --artifacts-dir artifacts_flux_hard --annotator consensus

# Central-table experiments (§5.1, §5.3) -- now committed scripts, not inline analysis
python3 exp6_prompt_baseline.py --artifacts-dir artifacts_flux --annotator chayan
python3 exp7_misbound_subset.py --artifacts-dir artifacts_flux --annotator chayan

# Hard-prompt-set consensus labels (§5.5)
python3 build_consensus_labels.py --artifacts-dir artifacts_flux_hard --annotators akhil grace pranav

# Within-item token-permutation control (§5.6)
python3 exp3b_within_item_permutation.py --artifacts-dir artifacts_flux --annotator chayan

# Discriminant validity on FLUX (§3.1 update) -- boxes.json must exist first
python3 recompute_boxes.py --artifacts-dir artifacts_flux
python3 discriminant_validity_check.py --artifacts-dir artifacts_flux --annotator chayan

# Agreement + inter-rater kappa
python3 analyze_agreement.py --artifacts-dir artifacts_flux \
        --annotator chayan --compare-annotator grace

# Test suite (274 tests)
python3 -m pytest tests/ -q
```

The §5.1 and §5.3 analyses were run inline for the original version of this document; they are now
committed, tested scripts (`exp6_prompt_baseline.py`, `exp7_misbound_subset.py`), since they are
the paper's core results and nothing in this repo should be load-bearing without a test.

### Key files

| Path | What |
|---|---|
| `ssa/anchor_set/artifacts_flux/` | FLUX images, manifest, 3× labels/counts, results, `boxes.json` |
| `ssa/anchor_set/artifacts_flux_hard/` | FLUX-hard images/manifest, 3× labels/counts + consensus, results (§5.5) |
| `ssa/anchor_set/artifacts_sdxl/` | SDXL equivalent |
| `ssa/anchor_set/flux_attention_capture.py` | MMDiT capture (C7) |
| `ssa/anchor_set/anchor_common.py` | Shared scoring/agreement logic |
| `ssa/anchor_set/exp{1..7}_*.py` | The battery, incl. prompt baseline (6, §5.1) and misbound subset (7, §5.3) |
| `ssa/anchor_set/exp3b_within_item_permutation.py` | Sharper Exp-3 falsification control (§5.6) |
| `ssa/anchor_set/build_consensus_labels.py` | Majority-vote consensus across annotators (§5.5) |
| `ssa/anchor_set/build_hard_prompts.py` | Generates the FLUX-hard prompt set (§5.5) |
| `ssa/anchor_set/recompute_boxes.py`, `discriminant_validity_check.py` | Box-geometry artifact check (§3.1) |
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

The work between here and a submission was mostly one experiment: a prompt set hard enough that
the models disobey often enough to measure. That experiment has now been run (§5.5, 2026-08-07) —
it materially improved C3's power (pooled p ≈ 0.03–0.05 on 3/4 label sets) without fully closing
the gap to the original ≥150-row target, and it turned up a genuinely new finding along the way
(§5.6, C8: attention's attribute-specificity itself degrades on hard images). Three of the four
cheap re-analyses in §9.2 are also done; the fourth (Part C robustness on FLUX) is blocked on new
GPU chain data, not on analysis time (§6, §9.2).

What's left is mostly writing. This document's numbers are current as of 2026-08-07, but
`proposal/CPGA-Research-Proposal.md` and `pi_level_experiment/RESULTS.md` still predate all of
§3–§6 above and need the same corrections before anything is submitted.
