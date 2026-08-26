# Design: Strengthening the Part B Money Result

Date: 2026-07-23. Branch: `pi-level-idea`. Author: Annotator 1 (with Claude).

Companion to `docs/pilot-design.md` (Track 1) and `pi_level_experiment/RESULTS.md` (the v4 run
this design is built to strengthen).

## Context

`pi_level_experiment/RESULTS.md` reports the first real-image test of the combined SSA metric:
Real scored significantly above Shuffled (p=0.0133) and Substituted (p=0.0038) on real
SD1.5-generated chains with real captured cross-attention — the discrimination Track 1's
persistence check (0.83 vs 0.83) could not make. That result stands. It also disclosed four
weaknesses, and this design attacks them:

1. Only 5 of 9 chains built (Mask R-CNN failed to detect the expected people in four
   attribute-free base images), giving n=13 Real observations.
2. Real's own hit rate was 46% (6/13 nonzero) — CLIPSeg's fixed 0.5 sigmoid threshold
   frequently registered no change at all for a genuinely new attribute.
3. `attention_scrambled` did not reach significance (p=0.38), because it drew substitute
   attention from a *same-chain* sibling whose subject may occupy an overlapping region —
   a confounded control, not a failed metric.
4. The whole experiment is single-seed (`SEED = 42`), so n is capped by the prompt list
   regardless of how many times it is run.

## Goals

- Grow the sample well beyond n=13 without inventing new prompts that would need fresh
  detectability validation.
- Raise Real's hit rate by a change that is defensible against a p-hacking objection.
- Replace the confounded scrambled control with unconfounded ones.
- Make scoring-side iteration free, so robustness checks stop costing GPU runs.

## Non-goals

- Anything in Part A (metric A's one-shot binding notebook, the human-agreement anchor set,
  the sub-chance n=2/n=3 anomaly). Deliberately out of scope; tracked separately in CLAUDE.md.
- DiT/FLUX/SD3 support. Claims remain scoped to UNet (SD1.5/SDXL).
- Re-running against CoIG's own Gemini chains. Still impossible — closed API, no attention hook.
- VQAScore correlation, Attend-and-Excite causal intervention, discriminant validity. These
  strengthen an eventual submission but are not load-bearing here.

## Architecture

`pi_level_experiment/run_chain_experiment.py` is **not modified**. It remains the provenance
record of the v4 run whose numbers RESULTS.md publishes. New work lands beside it, split at
the two real cost boundaries — GPU generation, and everything after it.

```
pi_level_experiment/
  run_chain_experiment.py    # UNCHANGED — provenance for the published v4 result
  generate_chains.py         # Stage 1 (GPU/Kaggle): prompts x seeds -> images + attn maps + manifest
  segment_cache.py           # Stage 2 (GPU or CPU): CLIPSeg -> cached sigmoid maps per (image, attribute)
  score_chains.py            # Stage 3 (pure numpy): threshold -> delta masks -> IoU -> CSV
  calibrate_threshold.py     # sweeps Stage 3 on held-out chains only; emits the frozen constant
  analyze_results.py         # extended: new controls + prompt-level clustering
  artifacts/                 # gitignored; pulled back from the Kaggle kernel
```

### Why the split

Today every scoring-side change — a threshold, a new control, a sweep — requires regenerating
every image on a GPU, because `main()` persists only the CSV/JSON and lets the images and
attention maps die with the kernel.

Stage 2 is the load-bearing piece. `segment(image, attribute)` is deterministic, and the same
`(image, attribute)` pair is currently re-segmented across Real, Shuffled, and Substituted.
Caching the **raw sigmoid maps, before any thresholding**, reduces Stage 3 to numpy:

```
delta = (sigmoid_curr > T) AND NOT (sigmoid_prev > T)
```

Sweeping `T` then costs milliseconds. This is what makes the pre-registered calibration in
Step 3 affordable, and it yields the threshold-sweep robustness check as a by-product rather
than a second experiment.

### Stage contracts

Stage 1 emits `manifest.json`: prompt_id, seed, n, subject_boxes, attribute→step mapping, and
detection pass/fail per (prompt, seed). Stages 2 and 3 read only the manifest and the files it
names. Neither imports diffusers; neither needs a GPU; both are testable against a hand-built
manifest with no models present.

Storage: ~0.5 MB per cached map at float16, ~20 maps per chain, ~27 chains ≈ 270 MB. Inside
Kaggle's output limit. `artifacts/` is gitignored.

## Step 1 — Grow the sample

Extract generation into `generate_chains.py`, looping over prompts × seeds and persisting
artifacts plus manifest.

Two coverage fixes land here:

- **Detectability pre-flight.** Generate the base image and run Mask R-CNN *before* committing
  to building a chain, recording pass/fail per (prompt, seed) in the manifest instead of
  silently skipping. Coverage becomes reportable data rather than an absence.
- **Seed list.** `SEEDS = [42, 7, 1234]` — three seeds, keeping 42 first so the v4 chains
  remain reproducible within the new run. 9 prompts x 3 seeds = up to 27 chains.
- **Detection threshold decision, resolved by the pre-flight.** Run the pre-flight across all
  27 (prompt, seed) pairs at the current `person_boxes(score_thresh=0.7)` first. If every
  prompt passes on at least two of three seeds, change nothing — multi-seed alone fixed
  coverage. Only if a prompt fails on all three seeds does it get a reworded base prompt, and
  `score_thresh` stays at 0.7 either way so detection sensitivity is never tuned. Rewording is
  the fallback, not a parallel option.

**Pre-registration boundary.** The seed list and the detection threshold are frozen at the end
of Step 1, before any IoU is computed. That ordering — not a promise — is what makes them
pre-registered.

**Acceptance:** manifest covers all (prompt, seed) pairs with explicit pass/fail; at least 20
chains built; artifacts reload without a GPU present.

## Step 2 — Rebuild the controls

| Condition | Delta target | Attention source |
|---|---|---|
| `real` | attribute, true step | its own |
| `shuffled` | attribute, wrong step, same chain | its own |
| `substituted` | foreign attribute, true step | its own |
| `attn_scrambled_samechain` | real target | different attribute, **same** chain *(legacy)* |
| `attn_scrambled_crosschain` | real target | different attribute, **different** prompt |
| `attn_scrambled_sameattr` | real target | **same** attribute, **different seed**, same prompt |

The legacy same-chain control is retained deliberately. It costs nothing (pure numpy) and
supports the strongest available reading of the v4 failure: *the original control was
confounded, and here are two unconfounded ones beside it* — rather than *the metric failed a
control and we changed the control*.

`attn_scrambled_sameattr` is sharper than the redesign RESULTS.md committed to. Multi-seed
guarantees every attribute has a same-attribute-different-image counterpart, so prompt and
semantics are held identical and only the generation the attention came from varies. If the
score still drops, attention is image-specific, not merely prompt-specific.

### Correctness bug to fix here

`score_chains()` currently selects the Substituted attribute by
`random.choice([fc for fc in chains if fc.chain_id != c.chain_id])` — a *different chain*, but
not necessarily a *different attribute*. Attributes repeat across `MECHANISM_PROMPTS`: chain 0
and chain 3 both contain `red apron` and `yellow helmet`. So Substituted can draw an attribute
that is genuinely present in the chain being scored, which would produce a legitimate nonzero
delta and silently corrupt the condition that is supposed to be structurally zero.

The v4 run reported Substituted at a clean 0/13, so this did not bite there — but that was the
luck of the draw, not a guarantee, and multi-seed increases the number of draws. The fix:
select the foreign attribute by *attribute string* not by chain identity, asserting the chosen
attribute appears nowhere in the target chain's own attribute list. Add a test.

**Acceptance:** six conditions emitted; Substituted provably disjoint from each chain's own
attributes by assertion, not by chance; scoring runs end-to-end with no GPU.

## Step 3 — Pre-registered threshold calibration

**Calibration set:** the four chains skipped in v4 (prompts 3, 4, 6, 8), now buildable.
**Held out and unopened during this step:** prompts 0, 1, 2, 5, 7.

Sweep `T` over the calibration set only, selecting by a criterion that never references the
effect of interest:

> maximize Real nonzero rate, subject to Substituted nonzero rate remaining exactly 0.

This leans only on Phase A's structural guarantee (a never-rendered attribute must produce an
empty delta), not on Real-vs-Shuffled separation. A threshold chosen this way cannot have been
tuned toward the p-value it is later used to compute.

Freeze `T` as a constant in `score_chains.py` with the calibration recorded in a comment:
sweep range, selected value, criterion, and which chains it was fit on.

**Acceptance:** `calibrate_threshold.py` refuses to run on non-calibration chains; the frozen
constant is committed in a separate commit from Step 4's scoring, so the git history itself
evidences the ordering.

## Step 4 — Confirmatory scoring, analysis, writeup

Score all chains at the frozen `T`. Multi-seed rows are clustered within prompt, so the pooled
test over roughly 81 rows is anticonservative. Report **both**:

- **Pooled** (Mann-Whitney, one-sided) — directly comparable to the v4 numbers already
  published in RESULTS.md.
- **Clustered** (Wilcoxon signed-rank on per-prompt paired differences, n=9 prompts) —
  conservative, and the one to lead with.

A result significant under both is defensible. If they diverge, the clustered test governs and
that is stated plainly, not buried.

Also ship the threshold-sweep robustness curve (p vs. `T` per contrast), now free.

**Deliverables:** updated `pi_level_experiment/RESULTS.md`, `CLAUDE.md` status, and
`proposal/CPGA-Research-Proposal.md` (Ideal Results, Potential Limitations, Experimental Setup,
and the four "Top priorities going forward" entries).

**Acceptance:** every claim in the writeup traceable to a committed CSV; limitations that
survive are stated as limitations, not omitted.

## Testing

Stages 2 and 3 import no diffusers and need no GPU, so they carry real pytest coverage against
a hand-built manifest with synthetic sigmoid maps:

- golden test pinning Stage 3's IoU at a known `T` on a fixed synthetic input
- Substituted is structurally 0 whenever the foreign attribute is absent
- the disjointness assertion from Step 2 rejects a same-attribute selection
- manifest round-trip (write, reload, identical scoring)
- threshold monotonicity: delta-mask area is non-increasing in `T`

Stage 1 gets thinner coverage by design (`base_prompt_for`, `token_indices`, manifest schema);
its model calls stay untested.

Any Phase A/C change is ported back to `pilot/spatial_semantic_alignment.py` and re-verified
against its Scenario 1–10 suite, per CLAUDE.md's note that Annotator 4's file is canonical and this
branch's fixes must flow back before merge.

## Risks

- **Multi-seed does not fix the four failing base prompts.** Then coverage stays at 5 prompts
  and only seeds grow n. Mitigation: the pre-flight makes this visible before the full run;
  fall back to rewording the base prompts.
- **No `T` satisfies the calibration criterion** (Substituted goes nonzero before Real's hit
  rate meaningfully improves). That is an informative negative — it would mean CLIPSeg cannot
  separate these attributes at any threshold, and the segmenter, not the cutoff, is the
  problem. Report it; do not relax the criterion to rescue the step.
- **Clustered test loses significance** where pooled keeps it. Lead with the clustered result
  anyway and report the honest reading: the effect is real per-prompt but the sample of
  *prompts* is small.
- **Kaggle GPU quota.** The multi-seed run is roughly 3x the v4 runtime. Mitigation: the stage
  split means a failed scoring change never costs a regeneration.
