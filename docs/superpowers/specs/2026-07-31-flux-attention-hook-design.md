# FLUX.1-dev Cross-Attention Capture Hook — Design

**Date:** 2026-07-31
**Branch:** `sdxl`
**Status:** approved, implementing

## Problem

`ssa/anchor_set/artifacts_flux/` has 105 FLUX.1-dev images (103/105 detected, all 101
count-clean) with 227 scored human-labeled attribute rows from Chayan
(`labels_chayan.json`, `counts_chayan.json`) — a fully labeled anchor set. But
`generate_anchor_images_flux.py` line 265-266 explicitly skipped cross-attention capture
("Skipped cross-attention mapping for FLUX"), hardcoding `predicted_owner="unavailable"`,
`model_scores={}` for all 301 attribute entries. Verified across the entire manifest: 0/301
rows have a real prediction.

The user's question is specifically **"does FLUX's attention encode subject-attribute
binding"** — not "is FLUX a better generator than SDXL" (answerable from labels alone,
no attention needed) and not "can *some* method predict binding on FLUX images" (OWL-ViT
cross-check, no attention needed). Only a real capture answers the stated question, and
the five-experiment battery (`run_five_experiments.py`, Part A design doc
`docs/part-a-five-experiment-battery-design.md`) is already built to consume exactly this
data shape once it exists — four of its five experiments are definitionally
attention-dependent (accuracy of attention-based prediction, early-window vs.
full-trajectory attention, attention-randomization falsification, attention-based metric
vs. positional baseline).

Attention only exists as intermediate transformer state during denoising — it cannot be
recovered from the saved PNGs after the fact. It must be captured live, which means
regenerating the images. Constraint from the user: **no new images and no relabeling** —
the rerun must reproduce the *same* 105 images (same prompts, same pinned seeds already in
the manifest) so the 227 existing labels stay attached to what they describe.

Two data points that shape scope, discovered before this design:

- **FLUX's own binding rate is high on this data**: 213/227 = 93.8% (100% at n=2, 96.4%
  n=3, 88.3% n=4) by comparing `human_label` to `intended_subject` directly. At n=2 there
  is zero headroom — a trivial baseline is already perfect there, so "beats the positional
  baseline" (Experiment 4) is structurally unwinnable at that stratum, independent of
  whether attention works.
- **The positional baseline (`nearest_subject_baseline` in `exp4_positional_baseline.py`)
  crashes on 40 FLUX rows** because the manifest's `attribute` string (e.g. `"yellow
  helmet"`) doesn't literally substring-match the prompt text (`"...yellow bike
  helmet..."`). Same root cause independently breaks FLUX's T5 token lookup (see §C).
  Not FLUX-specific in origin, but must be fixed for FLUX to run at all.

Given the headroom problem, the user chose **claim (b): does the signal survive
attention randomization** (`exp3_attention_scramble.py`'s cross-item scramble) as the
primary claim this capture is designed to support — it's the only one of the five that is
immune to "ground truth basically equals the naive prompt-order baseline," and it has
enough scored rows (227) to be well-powered.

## Scope (decided in brainstorming)

- Hook only the **19 `FluxTransformerBlock`s** ("double" blocks — text and image have
  separate q/k/v projections before joint attention). The 38 `FluxSingleTransformerBlock`s
  are explicitly **not** hooked in this pass (see Non-goals).
- Capture **both** early-window (steps 0-12 of 25, matching SDXL's
  `EARLY_WINDOW_FRACTION=0.5` convention) and full-trajectory (all 25 steps) attention, so
  `model_scores` and `model_scores_full` are both populated — Experiment 2 must not go
  "unavailable" on FLUX the way it did on SDXL for lack of the `_full` field.
- **Same 105 prompts, same pinned seeds** already in `artifacts_flux/manifest.json`. No
  new prompts, no new seeds, no relabeling.
- **diffusers pinned to 0.39.0** (this repo's installed version, and what the hook is
  developed and tested against locally) for the rerun, replacing the script's current
  floor pin (`diffusers>=0.31`). Reasoning: Pranav's original run used an unpinned `>=0.31`
  floor and he could not confirm the exact resolved version from memory (the instance was
  deleted before this was checked) — FLUX's attention implementation changed materially
  between 0.31 and 0.39 (different `Attention` base class, different processor dispatch),
  so an unpinned rerun risks a different numerical path producing different images even
  with identical seeds. Pinning to the version this hook is tested against removes that
  variable; if the 3-prompt pilot (§Pilot) shows it doesn't reproduce Pranav's images
  structurally, the fallback is to bisect toward 0.31 and re-test the hook against that
  API instead — a decision made from pilot evidence, not assumed here.
- **Full-precision bf16 only.** Confirmed via Pranav (A6000-class GPU, "Big GPU detected!"
  full-precision branch, no bitsandbytes/4-bit quantization). The quantized/CPU-offload
  code paths in `generate_anchor_images_flux.py`'s `load_all_models` are out of scope.

## Non-goals

- The 38 `FluxSingleTransformerBlock`s. These receive `encoder_hidden_states=None` — text
  is concatenated with image tokens *before* the attention call (`transformer_flux.py`
  line 386), so the processor has no boundary information and would need to be told
  explicitly where text ends. More code, more risk, for blocks that mostly do late-stage
  refinement rather than layout decisions. Can be added in a follow-up if the double-block
  result is ambiguous.
- General DiT/SD3 reuse. This hook is FLUX-specific (its own `Attention`/`FluxAttention`
  class, its own joint-sequence layout). Matches the existing scoping decision for metric
  B (`pilot/spatial_semantic_alignment.py`'s DiT/FLUX/SD3 exclusion, documented in
  `CLAUDE.md`).
- Quantized/multi-GPU/CPU-offload code paths in `load_all_models` — moot given confirmed
  full-precision hardware.
- Regenerating with different prompts, seeds, or a larger/smaller prompt set. Any change
  to what gets generated invalidates the existing 227 labels.

## Architecture

### B. Capture class — `FluxAttentionCapture`

Structurally mirrors the SDXL script's `AttentionCapture` (`hook_pipeline`,
`unhook_pipeline`, a `phase_b_cross_attention_map`-equivalent) but the internals differ
because FLUX's stock `FluxAttnProcessor` never materializes an attention probability
matrix — it calls `dispatch_attention_fn(query, key, value, ...)`, a fused SDPA/flash-
attention call that produces the output directly (`transformer_flux.py` line 118).

`FluxCustomAttnProcessor` must therefore **recompute attention manually** to have anything
to capture:

1. Project q/k/v via `attn.to_q`/`to_k`/`to_v` and (for the text stream)
   `attn.add_q_proj`/`add_k_proj`/`add_v_proj` — exactly as `_get_qkv_projections` does.
2. Apply `attn.norm_q`/`norm_k` (RMSNorm) and, for the text stream,
   `attn.norm_added_q`/`norm_added_k` — exactly as the stock processor does.
3. Concatenate: `query = cat([encoder_query, query])`, same for key/value. **Text tokens
   occupy indices `0..T-1`, image tokens `T..T+4095`** (T = padded T5 sequence length;
   confirmed by reading `transformer_flux.py` lines 102-112 — encoder projections are
   concatenated first).
4. Apply `apply_rotary_emb(query, image_rotary_emb, ...)` / same for key — must match the
   stock processor's rotary embedding application exactly, or captured attention and the
   continuing generation diverge.
5. Explicitly form `attn_probs = softmax(query @ key.transpose(-1,-2) / sqrt(head_dim))`
   instead of dispatching to the fused kernel. **This materialized matrix is what gets
   captured** — sliced to the image-rows / target-text-token-column submatrix
   (`attn_probs[:, T:, target_token_index]`), stored in an `AttentionStore` analogous to
   the SDXL one, keyed by step and block name.
6. Continue with `hidden_states = attn_probs @ value`, then the stock processor's output
   projections (`to_out`, `to_add_out`, `split_with_sizes`) — so the pipeline's continuing
   denoising is unaffected by the capture.

This only replaces the processor on the 19 double blocks (`FluxTransformerBlock`); single
blocks keep their stock `FluxAttnProcessor`, matching scope.

`hook_pipeline`/`unhook_pipeline` swap via `pipeline.transformer.set_attn_processor(...)`,
confirmed available on `FluxTransformer2DModel` (checked locally: `attn_processors` and
`set_attn_processor` exist, same API shape as the UNet).

The map-extraction method (`phase_b_cross_attention_map` equivalent) aggregates per-step,
per-block records the same way the SDXL version does: interpolate each block's map to
target resolution, weight by native resolution², average — reusing that logic rather than
duplicating it where the shapes line up (image-token count differs from SDXL's, everything
else is the same aggregation math).

### C. T5 token alignment

New `flux_token_indices(tokenizer_2, prompt, phrase)` — same substring-match algorithm as
the existing `token_indices()`, but against `pipeline.tokenizer_2` (T5), since T5 tokens
are what actually enter the joint attention sequence (CLIP only contributes a pooled
vector, never joins the sequence).

Independent of which model: **fix the phrase-matching gap** that breaks both
`nearest_subject_baseline` (Experiment 4) and any `token_indices` variant on ~9 FLUX
attribute rows where the manifest's `attribute` string is a strict sub-phrase of a longer
descriptive phrase in the prompt (`"yellow helmet"` vs. `"...yellow bike helmet..."`).
Fix once, shared: when the exact phrase isn't found verbatim, fall back to matching on the
attribute's content words (e.g., last noun + preceding modifier) as a contiguous
subsequence. Applied in both `token_indices`/`flux_token_indices` and
`nearest_subject_baseline` — one bug, one fix, benefits both models' Experiment 4 and any
future capture script.

### D. Scramble protocol (claim b) — no code changes

`exp3_attention_scramble.py`'s `cross_item_scramble_accuracy` operates purely on whatever
`model_scores` dict is present in the manifest — it has no model-specific logic. Once
FLUX's manifest carries real `model_scores`, `run_five_experiments.py --artifacts-dir
artifacts_flux --annotator chayan` runs unmodified and produces claim (b)'s real-vs.-
scrambled McNemar comparison alongside the other four experiments.

### E. Manifest population

A `generate_and_score`-equivalent added to `generate_anchor_images_flux.py`, structured
like the SDXL script's:

1. Hook the transformer (`capture.hook_pipeline`).
2. Generate with the pinned seed from the manifest (unchanged from current script — seed
   selection logic is untouched).
3. Detect boxes via the existing Mask R-CNN + CLIP assignment (untouched, already
   working — 103/105 detected).
4. For each attribute: get T5 token indices (§C), capture early-window
   (`max_steps=12`, i.e. `NUM_INFERENCE_STEPS * EARLY_WINDOW_FRACTION` with
   `NUM_INFERENCE_STEPS=25`) and full-trajectory (`max_steps=25`) attention maps, run the
   existing `predicted_owner_from_attention` (model-agnostic, reused as-is from
   `anchor_common.py`) against the same box-containment logic already used for SDXL.
5. Unhook.

Output shape matches `build_attribute_entry`'s existing `predicted_owner`, `model_scores`,
`predicted_owner_full`, `model_scores_full` fields exactly — no manifest schema change.

## Test plan (all CPU-only, no GPU required)

1. **Equivalence test** (load-bearing, mirrors metric B's Scenario 10 regression test):
   instantiate a small real `FluxTransformer2DModel`, run once with stock processors and
   once with `FluxCustomAttnProcessor` in a "computed but not stored" mode, assert outputs
   match bit-for-bit. Proves the hook's presence alone doesn't perturb generation.
2. **Numerical-closeness test**: same small model, compare the manual-softmax path's
   output against the fused-kernel path's output — expect close but *not* bit-identical
   (different floating-point summation order between manual softmax and a fused
   flash-attention kernel), assert within a defined tolerance. This documents the honest
   residual risk explicitly: even correct code, the same GPU, and the correct diffusers
   version can still produce small numerical drift relative to Pranav's original images,
   because recomputing attention by hand is mathematically equivalent to the fused kernel
   but not numerically identical to it.
3. **Token alignment test**: synthetic prompt/attribute pairs, assert T5 indices land at
   the expected sequence positions, including the content-word fallback path for
   sub-phrase attributes like `"yellow helmet"` vs. `"yellow bike helmet"`.
4. **Slice orientation test**: assert the text/image split is `attn_probs[:, T:, ...]` for
   image rows and column `target_token_index` for the text key — not the reverse. This is
   the single easiest indexing mistake to make silently; getting it backwards produces a
   plausible-shaped but meaningless map with no error raised.
5. **Scramble/battery smoke test**: run `run_five_experiments.py` against a small
   FLUX-shaped dummy manifest (extending `make_dummy_artifacts.py`'s pattern) with
   populated `model_scores`/`model_scores_full`, confirm all five experiments complete
   without the "unavailable" degradation path firing.

## Pilot protocol (Pranav, before the full 105-prompt rerun)

Regenerate prompts 0-2 with the hook active, on the same A6000-class hardware confirmed
full-precision. Compare against the existing PNGs:

- **Byte-identical first** — best case, would confirm exact reproduction.
- **If not byte-identical, structural comparison instead** — same detected person boxes
  (position/count), same attribute-to-subject placement as far as visually verifiable.
  Per test #2 above, small numerical drift from the unfused attention recomputation is
  expected even under otherwise-identical conditions, so byte-identity is not the
  pass/fail bar; structural equivalence (would the existing human labels still describe
  what's in the image) is.
- **If structure doesn't hold** — stop before the full run. Fallback path: try the
  diffusers 0.31 bisect (§Scope) or reassess whether relabeling the 3 pilot images is an
  acceptable cost before deciding on the full 105.

Only after the pilot passes does Pranav run the full 105-prompt regeneration.

## Open items carried into implementation

Not decisions deferred — points where the pilot's actual evidence is needed rather than
an assumption made here:

- Whether the sequence padding length `T` used in step 3 concatenation is fixed at 512
  (the max T5 sequence length referenced in `generate_anchor_images_flux.py`) or varies
  per-prompt — needs a shape check against a real forward pass during implementation.
- Exact image-token spatial side length after FLUX's patch packing (expected 64×64=4096
  for 1024px images at the standard 2×2 patch size, analogous to how SDXL's UNet native
  resolutions are inferred from `spatial_dim`) — verified the same way, via shape
  assertion in the equivalence test rather than hardcoded.
