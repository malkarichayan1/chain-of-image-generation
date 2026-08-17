# Remaining Experiments — Triage and Runbook

Triage of the 12 "To run" rows in *CPGA Research Doc Template Summer 2026*, plus exact
steps. Written so Pranav can start without further design decisions.

Status as of 2026-08-12. Branch: `hard-prompt-set-retest`. Decisions below are final per
the group's call: #15 cut, #21 gated on #20's outcome (not cut), #8 and #30 kept.

---

## 1. Verdict table, in run order

**Updated 2026-08-17.** Rows 1–4 are all done; the Kaggle framing throughout this doc is
superseded — GPU work now runs on Thunder Compute (`tnr` CLI), where the public repo + git-tracked
images mean there is **no dataset-upload step at all**.

| Order | # | Experiment | Where it runs | Cost | Depends on |
|---|---|---|---|---|---|
| — | 13 | κ on disobeyed rows | local, done | none | — |
| — | 8 | CLIPScore discriminant validity | **DONE 2026-08-14**, local CPU | ~10 min | — |
| — | 31 | VQAScore on disobeyed rows | **DONE 2026-08-14, local CPU — never needed a GPU** (blip-vqa-base ≈385M params). Both sets. | ~20 min | — |
| — | 14,16,17,18 | Taxonomy capture (layer/head/window/distribution) | **DONE 2026-08-17**, Thunder Compute 1×A100 | ~45 min, ~$2 (not 10–15 GPU-hr) | — |
| — | 19 | Intent-vs-realization on sharpest cells | **DONE 2026-08-17 — NEGATIVE (bulletproof), 0/10** | local CPU | — |
| **1** | 20 | Attention steering (A&E) | Thunder Compute GPU | ~1 GPU-hr, ~$1 | **unblocked** — window now set from #19 (late blocks 13–18, steps 0–24) |
| conditional | 21 | Controlled prompt-obedience | — | — | **only if #20 is inconclusive** |
| held | 30 | Second MMDiT (SD3 / PixArt-Σ) | — | ~3–5 days eng | unblocked; **no rescoping needed** — #19 negative, so the existing framing holds (§2c) |
| cut | 15 | Taxonomy on SDXL | — | — | dataset doesn't exist |

**#8 moved to first place.** It was scoped as a Kaggle job in the original doc, but CLIP
(`openai/clip-vit-base-patch32`) already runs on CPU elsewhere in this repo —
`recompute_boxes.py`'s own docstring calls it "CPU-tractable at ~35 images total" for a
heavier Mask R-CNN + CLIP combo than #8 needs. It has **zero GPU queue time**, so there's no
reason to wait: Pranav can run it today, in parallel with the Kaggle jobs starting.

---

## 2. Reasoning behind the non-obvious calls

**(a) #14/#16/#17/#18 need GPU, not re-analysis.** The doc says #14 is "re-analysis if
per-layer maps saved." They weren't. `flux_attention_capture.py:136` averages heads away
*inside* the attention processor, and `cross_attention_map` then averages every layer and
step before anything touches disk. `manifest.json` holds one pooled scalar per (attribute,
subject) — head/layer/step identity is gone. `taxonomy_capture_flux.py` re-runs generation
to recover it.

**(b) #15 (cut).** No SDXL hard set exists — only `artifacts_sdxl/`,
`artifacts_sdxl_growth/`, `artifacts_sdxl_backfill/`. Building one is a second full
annotation cycle for a comparison the paper already makes at the pooled level (C5).

**(c) #19's selection must not double-dip, and #30 is sequenced behind it.** Selecting
"sharpest cells" and testing them on the same rows is selection-then-inference; a Holm
correction across selected cells doesn't fix it because the selection already used the
outcome data. **Fix (free):** select cells on `artifacts_flux/` (easy set), test only on
`artifacts_flux_hard/`. Frozen in §5. This is also why #30 (kept, not cut) runs *after*
#19: if #19 finds a cell that tracks the rendered image, the paper's conclusion changes,
and #30 should be scoped under the new framing rather than redone.

**(d) #21 is conditional, not scheduled.** #20 intervenes on attention directly (causal);
#21 would only manipulate prompts and watch (observational) — strictly weaker for the
intent-vs-realization question #20 is built to answer. The hard-prompt-set run already
measured how far "engineer harder stimuli" travels (obedience 94%→80% against a ~50%
target), so #21 isn't free information either. Run it only if #20 comes back inconclusive
(e.g. steering doesn't reliably move the rendered attribute, so no causal read is possible).

**(e) #8 and #30 are kept.** #8: closes a reviewer objection with a mechanism-free but
easy-to-anticipate flavor, and now costs ~10 minutes CPU instead of a GPU slot — the
math changed even if the argument for cutting it didn't. #30 (biggest external-validity
lever on the list) is expensive specifically because of the C7-style attention-hook
reverse-engineering plus a full annotation cycle, not because it's uninteresting — kept,
sequenced last, after #19 so it isn't done twice.

---

## 3. Already done — #13, κ on disobeyed rows

`exp8_misbound_kappa.py`, 15 tests.

```bash
cd ssa/anchor_set
py -3 exp8_misbound_kappa.py --artifacts-dir artifacts_flux_hard \
    --annotators akhil grace pranav --out artifacts_flux_hard/misbound_kappa.json
```

| Subset | akhil–grace | akhil–pranav | grace–pranav |
|---|---|---|---|
| Overall (n=409) | 0.912 | 0.900 | 0.889 |
| **Disobeyed only (n=62)** | **0.886** | **0.905** | **0.867** |

Overall reproduces the published κ exactly (validates the implementation). The disobeyed
subset does not collapse: **C3's ground truth holds.**

---

## 4. Pranav's steps

### Step 0 — one-time setup (~10 min)

```bash
git checkout hard-prompt-set-retest && git pull
cd ssa/anchor_set
py -3 -m pytest tests/ -q          # must be run from INSIDE this dir; 379 tests, all green
```

Everything below assumes `ssa/anchor_set/` as the working directory.

### Step 1 — CLIPScore discriminant validity (#8), ~10 min, no GPU queue

`exp10_clipscore_discriminant.py` (new, 17 tests). Crops each candidate subject's box,
scores it against the attribute's caption with CLIP, predicts the subject with the highest
score — a prediction that never touches attention — then reports (a) how often that
agrees with attention's own `predicted_owner`, and (b) both methods' accuracy against human
labels.

```bash
py -3 exp10_clipscore_discriminant.py --artifacts-dir artifacts_flux_hard --annotator consensus
py -3 exp10_clipscore_discriminant.py --artifacts-dir artifacts_flux --annotator chayan
```

First run downloads CLIP weights (~600 MB) and takes a few minutes on CPU; scores are
cached incrementally to `clip_scores.json`, so a second run only fills gaps.

**Caveat to report alongside the number, not after it:** `assign_subjects` already uses
this same CLIP checkpoint to decide which box is "barista" vs. "cyclist," so a high
agreement rate is not pure evidence attention independently converges on CLIP — part of it
could be inherited through the shared box-assignment step. The script prints this warning
next to the number so it can't be reported without it.

### Step 2 — VQAScore (#31), ~1 GPU-hr per set

Code complete (`vqa_score_flux.py`, `vqa_agreement_check.py`). Only the run is missing.

1. Build a Kaggle dataset from `artifacts_flux_hard/` (`manifest.json`, `boxes.json`,
   `images/`). `boxes.json` already exists locally — commit it first.
2. Copy `kernel-metadata-vqa-flux.json` → `kernel-metadata-vqa-flux-hard.json`, update `id`
   and `dataset_sources`.
3. `kaggle kernels push -p .`
4. Download `vqa_scores.json` into the matching `artifacts_*/` directory:
   ```bash
   py -3 vqa_agreement_check.py --artifacts-dir artifacts_flux_hard --annotator consensus
   ```

The misbound-subset section of that output is the number the paper wants: does a SOTA
judge-based metric also fail where attention fails?

### Step 3 — Taxonomy capture (#14/#16/#17/#18), ~10–15 GPU-hr

The main run. `taxonomy_capture_flux.py` + `kernel-metadata-taxonomy-flux.json` are written
and tested (19 tests, including an equivalence check against a real tiny
`FluxTransformer2DModel` proving the per-head reduction reproduces the published pooled
path to 1e-6).

```bash
kaggle kernels push -p .              # uses kernel-metadata-taxonomy-flux.json
```

Repeat with `dataset_sources` pointed at the hard-set dataset from Step 2.

`taxonomy_index.json` is rewritten after every image and already-captured ids are skipped
on restart — a Kaggle timeout costs at most one image; re-run the same kernel to resume.

**Before scaling up, run on a handful of images and check two `taxonomy_index.json`
fields:** `repro_mean_abs_pixel_diff` (should be ~0 — the regenerated image vs. the stored,
human-labeled one) and `pooled_owner_matches_manifest` (should be `true` on the large
majority). Labels and boxes belong to the *original* images; diffusion isn't guaranteed
bit-reproducible across sessions, and a drifted image's attention describes something
nobody labeled — silently. If either check fails broadly, stop and report before burning
the full run.

### Step 4 — Taxonomy analysis (#14/#16/#17/#18) and #19

`exp9_taxonomy_analysis.py` is **written and tested** (44 tests, including isolation tests
proving each layer band / timestep window looks at exactly the indices it claims to, and
verdict-logic tests proving #19 says POSITIVE only when a cell both beats the baseline *and*
clears chance on the misbound subset — never on significance alone, and never in the wrong
direction). It applies the reproduction-check threshold from Step 3 (drops any image whose
`repro_mean_abs_pixel_diff` exceeds `--repro-threshold`, default 0.05, and reports the drop
count), reduces the capture into #14/#16/#17/#18's grids, and runs #19 exactly per §5's
selection rule. Pure CPU, no GPU needed — only waits on Step 3's Kaggle output.

```bash
py -3 exp9_taxonomy_analysis.py \
    --easy-dir artifacts_flux --easy-annotator chayan \
    --hard-dir artifacts_flux_hard --hard-annotator consensus \
    --out artifacts_flux_hard/taxonomy_report.json
```

### Step 5 — #20, attention steering

**Implemented 2026-08-14** (CPU-testable up front, no GPU needed to write or verify the
mechanism — only to run it for real). `flux_attention_capture.py` gained
`FluxSteeringAttnProcessor`/`SteeringConfig`/`SteeringState`/`apply_steering`: scales
selected columns of `attn_probs` (the matrix `FluxCustomAttnProcessor` already materializes
per claim C7) toward a target attribute's tokens, on a recipient subject's image rows, inside
a configurable (layer, step) window — renormalized so each affected row stays a valid
distribution — before the `@ v` that turns attention into the image. 8 new tests, including
two full-tiny-transformer checks: strength=0 reproduces the plain capture path exactly, and
real strength changes the output. `exp20_attention_steering.py` was rewritten to use it.

Scope the actual `--step-start`/`--step-end`/layers after #19 identifies the best cell —
`DEFAULT_STEER_LAYERS`/`DEFAULT_STEER_START`/`DEFAULT_STEER_END` in the script are
placeholders (mid blocks 7–12, steps 12–19) carried from the pre-registered design intent,
not yet #19's measured output. Update them (or pass the CLI flags) once #19 runs.

### Conditional — #21

Run only if #20 doesn't yield a clean causal read (steering doesn't reliably move the
rendered attribute). Not scheduled by default.

### Held — #30

Scope after #19 lands, under whatever framing #19 settles on. Needs a new attention hook
(SD3/PixArt-Σ have different block structure and text-encoder setup — this repeats the C7
reverse-engineering effort) plus a full new annotation cycle. Biggest external-validity
lever on the list; also the most expensive.

---

## 5. Pre-registered grid — freeze before looking at any output

**Layer bands (#14):** blocks 0–6 / 7–12 / 13–18. Holm across the 3 bands.

**Timestep windows (#17):** steps 0–6 / 7–12 / 13–18 / 19–24 (25 steps don't divide evenly;
first window takes the extra step). Holm across the 4 windows.

**Per-head (#16):** all 19 × 24 = 456 cells, heatmap + full distribution. Benjamini-Hochberg
FDR across all cells. Report the distribution, not the winner.

**Distribution metrics (#18):** spatial entropy (renormalized per attribute), in-box mass
fraction for the correct subject, peak-to-second-peak ratio. Descriptive only, no tests.

**#19 selection rule:** rank cells by in-box mass fraction on `artifacts_flux/` (easy set)
only. Take the top 10. Test those 10 on `artifacts_flux_hard/` only — prompt-only baseline
(#11) and disobeyed-rows analysis (#12) per cell, Holm-corrected across the 10. Selection
and test sets are disjoint (§2c).

**Pre-registered read:** if no cell beats the prompt-only baseline on the hard set, the
negative result is bulletproof. If some cell does, that's a genuine positive finding that
changes the paper's conclusion — and reframes how #30 should be scoped.

---

## 6. What this does not cover

- Part C robustness battery on FLUX remains blocked per CLAUDE.md §6: `pi_level_experiment/`
  has no FLUX chain data at all — a full Kaggle chain pipeline, not a re-analysis.
- `docs/raw-attention-paper-briefing.md` still predates the hard-set, consensus,
  permutation-control, discriminant-validity, and now #13 results.
