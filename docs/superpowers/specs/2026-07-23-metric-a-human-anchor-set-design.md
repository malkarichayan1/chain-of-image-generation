# Metric A Human-Agreement Anchor Set — Design

**Date:** 2026-07-23
**Branch:** `pi-level-idea`
**Status:** approved, implementing

## Problem

Metric A (the one-shot cross-attention attribute-binding metric) has never been
validated against human judgment. Every "does attention track binding" claim so far
rests on an automated OWL-ViT ground truth that the team itself flags as brittle (the
`ATTR_OWNERSHIP_THRESHOLD=0.14` saga: fixing one false positive introduced false
negatives). `SSA-Metric-Memo.md` §7 Step B and §8 Tier-1 item 3 already call for the
fix — a small hand-labeled anchor set that decouples the *attention* claim from the
*detector* — but it has never been built. This is priority next-step #6 in `CLAUDE.md`.

This spec builds that anchor set: a human-annotated ground truth against which metric
A's attention-based binding predictions can be scored, giving the go/no-go signal the
memo's decision gate specifies ("if SSA agrees with human labels above chance with a
clear margin → green-light").

## Scope (decided in brainstorming)

- **Labels capture binding only** — for each generated image, which subject owns which
  attribute. Not chain emergence (metric B), not cross-metric adjudication.
- **Images are fresh stratified one-shot generations** — 18 total, 6 each at subject
  count n=2 / n=3 / n=4. Stratification is deliberate: it directly feeds the *other*
  open metric-A thread (the sub-chance n=2/n=3 binding anomaly) by giving each subject
  count enough examples to report a per-stratum agreement number, not just an aggregate.
- **Single annotator now** (you), second annotator deferred. Unblocks the decision gate
  fastest; inter-rater reliability (Cohen's kappa, the "human ceiling") is added later
  as a follow-up pass over the same 18 images.
- **A small local script records labels**, writing structured JSON that the analysis
  stage joins against the metric's predictions.

## Non-goals

- Automated (OWL-ViT/CLIPSeg) ground truth — this set is precisely the thing that
  replaces it for validation purposes.
- Metric B / chain emergence labels.
- Inter-rater reliability (second annotator) — deferred, not designed out. `labels.json`
  is keyed so a second annotator's file drops in alongside the first without rework.
- SDXL / SD2.1 replication (Tier-2 known-quality ranking) — out of scope here.

## Architecture

Four stages, following the existing `pi_level_experiment/` split by real cost boundary
(GPU vs. pure-CPU), threaded by a `manifest.json`. Lives in a self-contained
`ssa/anchor_set/` package (metric A's home is `ssa/`).

```
prompt_specs.json                 (hand-written: 18 prompts, stratified n=2/3/4)
        |
        v  Stage 1  (Kaggle GPU kernel, own kernel-metadata)
generate_anchor_images.py         one-shot SD1.5 gen + Mask R-CNN detect + CLIP assign
        |                         + windowed cross-attention -> per-attribute predicted owner
        v
artifacts/manifest.json           prompt_id, n, subjects, attributes, image_path,
artifacts/images/*.png            model_predicted_owner, model_scores, detected flag
        |
        v  Stage 2  (local CPU, blind)
label_images.py                   shows image + attribute (NO model prediction visible),
        |                         you pick the owning subject / none / unclear
        v
artifacts/labels_<annotator>.json resumable; keyed by (prompt_id, attribute)
        |
        v  Stage 3  (local CPU)
analyze_agreement.py              join manifest + labels -> accuracy of model vs human,
                                  overall + per stratum, vs chance baseline (1/n)
```

### Stage 1 — `generate_anchor_images.py` (Kaggle GPU)

Reuses, by duplication (the established repo pattern — `generate_chains.py` itself
duplicates the attention-capture out of `pilot/spatial_semantic_alignment.py` and says
"keep in sync"), the pieces it needs from `generate_chains.py`:

- `AttentionStore` / `CustomAttnProcessor` / `AttentionCapture` — cross-attention capture,
  windowed to the early structural layout window (`max_steps = 0.5 x steps`, `cond_index=1`
  to take the conditional CFG branch), identical to the validated metric-B capture.
- Mask R-CNN person detection + CLIP subject assignment (`person_boxes`, `assign_subjects`).

**Per prompt:**
1. Generate ONE image from the full compositional prompt (not a chain — this is the
   one-shot setting metric A is scoped to).
2. Detect people; require exactly `n`. Retry with a new seed up to 3x on
   under/over-detection (same policy as `generate_chains.py`). Record `detected` +
   `num_people_detected` in the manifest whether or not it passes — coverage is
   reportable data, not a silent skip.
3. Assign each detected box to a subject role via CLIP (`assign_subjects`).
4. For each attribute: capture its windowed cross-attention map, then compute
   `predicted_owner = argmax over subjects of mean attention mass inside that subject's
   box`. Also record the raw per-subject mass vector (`model_scores`) so ties / margins
   are inspectable later.
5. Save image + a manifest entry.

**Key framing point, documented in the module docstring:** the prompt's *intended*
pairing is NOT the ground truth. SD1.5 mis-binds; that is the phenomenon under test. The
ground truth is what the human sees in the pixels. `model_predicted_owner` is the
metric's guess; the human label is truth; agreement is the result.

**Subject localization uses reliable person detection, not OWL-ViT attribute-box
containment.** This is deliberate and is the whole point of the anchor set: person
detection is reliable (it is already used as a preflight), whereas *attribute ownership*
is the brittle part. By letting attention alone decide ownership over reliably-detected
subject boxes, and comparing to a human, we validate the attention signal specifically —
decoupled from OWL-ViT's brittleness, exactly as the memo intends.

### Stage 2 — `label_images.py` (local CPU, blind)

- Iterates the manifest's detected chains in a randomized (seeded) order.
- For each (image, attribute), opens the image and prompts you to pick the owning
  subject from the prompt's subject list, or `none` (attribute never rendered) or
  `unclear`. **The model's prediction is never shown** — blind labeling prevents
  anchoring on the metric's own guess.
- Writes `labels_<annotator>.json` incrementally after every judgment; resumable across
  sessions (~18 images x up to 4 attributes ~= 50 judgments).

### Stage 3 — `analyze_agreement.py` (local CPU)

- Joins manifest + labels on (prompt_id, attribute), dropping human `none`/`unclear`
  rows from the accuracy denominator (reported separately as a coverage number).
- Computes agreement = fraction where `model_predicted_owner == human_label`, overall and
  split by stratum (n=2/3/4), against the per-stratum chance baseline (1/n).
- Emits a summary table + a per-row CSV. This is the decision-gate artifact.

## Testing

Each pure-logic stage gets pytest tests in `ssa/anchor_set/tests/`, matching the
`pi_level_experiment/tests/` convention. Testable surfaces (no GPU):
- `prompt_specs.json` well-formedness: 6 prompts per stratum, each with `n` subjects and
  `n` distinct attributes, subjects/attributes drawn from the shared vocabulary.
- Manifest entry assembly (pure dict, like `manifest_chain_entry`).
- `predicted_owner` argmax logic over a synthetic attention map + boxes.
- Label-file read/write/resume round-trip.
- Agreement computation: known manifest + known labels -> known accuracy, including the
  none/unclear exclusion and per-stratum split.

The GPU-only pieces (SD1.5 generation, real detection) are exercised by the Kaggle run
itself, not unit-tested locally — same boundary as `generate_chains.py`.

## Success criteria

- 18 images generated, most detected at the required `n` (coverage reported honestly;
  n=4 is expected to detect worse per the existing selection-effect finding).
- All 18 labeled by one annotator.
- Analysis produces per-stratum agreement vs chance. The *design* succeeds if it produces
  this number cleanly; whether the number clears chance is the empirical result the gate
  reads, not a success condition of the build.
