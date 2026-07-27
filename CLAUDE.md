# CoIG Faithfulness Project — Working Context

CPGA / CoIG Faithfulness Project, Summer 2026. Author: Chayan. Team: Ane, Grace, Akhil, Pranav.
Auditing whether Chain-of-Image-Generation (CoIG) — a step-by-step, "locked" image generation
pipeline — actually produces faithful outputs, or whether its own faithfulness claims are
compromised by the lock mechanism itself.

**Current direction (as of 2026-07-22):** two independently-built SSA (Spatial-Semantic
Alignment) metrics exist on the team. Decision made: stop treating them as competing options —
combine them into one paper, one validating the other. This file is the anchor doc for picking
that up in a new session. Deep technical history (exact thresholds, bug traces, run logs) lives
in the auto-memory system (`coig-ssa-metric-direction`, `coig-cr-pilot-design`,
`coig-colab-sync-workflow`) — this file is the current plan and what to do next.

**Money result executed (2026-07-22, branch `pi-level-idea`, Kaggle GPU kernel
`chayanmalkari/coig-pi-level-chain-experiment` v4): statistically significant, on real
generated images.** See `pi_level_experiment/RESULTS.md` for the full honest writeup. Headline:
real scores significantly higher than both shuffled (p=0.0133) and substituted (p=0.0038,
Mann-Whitney U) — the exact discrimination Track 1's persistence check (0.83 vs 0.83) could
not make. Substituted was perfectly clean (0/13 nonzero, no exceptions). Two real limitations,
disclosed not hidden: real's own hit rate was only 46% (6/13) — the underlying SD1.5
generation/CLIPSeg pipeline is noisy — and the `attention_scrambled` control (ported from
metric A's Tier-1 falsification test) did not reach significance (p=0.38), likely because its
same-chain attention-swap design is confounded by subjects sitting in overlapping image
regions. Sample is also small (5/9 chains — 4 skipped on person-detection failures in the
attribute-free base prompt, a specific fixable issue, see RESULTS.md). This is a genuine early
positive result, not a "100% proof" — the next moves are growing the sample and redesigning
the attention_scrambled control, not a different approach.

## Track 1 — Causal Relevance pilot (done, confirmed the problem)

Executed 2026-07-18, 10 chains × {Real, Shuffled, Substituted}. Result: **lock confound
confirmed.**
- appears_at_step: real=1.00, shuffled=0.53, substituted=0.00 (discriminates correctly)
- persists_to_final: real=0.83, **shuffled=0.83 (identical to real)**, substituted=0.00

The lock keeps every attribute in the final image regardless of which step it was actually
supposed to appear at, so "persisted to final" cannot tell faithful generation from the lock
just mechanically preserving whatever was already there. This is the motivating problem for
everything in Track 2 — CoIG's own faithfulness signal is compromised, so a new metric is
necessary, not optional.

## Track 2 — two SSA prototypes

### A. One-shot attribute-binding metric (branch `ssa-metric`, notebook `ssa/coig_ssa_colab.ipynb`)

Measures whether SD1.5's cross-attention binds each attribute to the correct *subject* in a
single generated image (e.g. "a barista wearing a red apron and a cyclist wearing a yellow
helmet" — does "apron" attention land on the barista or the cyclist). Deliberately scoped to
one-shot generation, not the chain — de-risking the core mechanism (does cross-attention
meaningfully encode binding at all) in the cleanest possible setting before porting to the chain.
An editing-chain version was tried first (IP2P, MagicBrush) and failed — both produced global
recolors instead of localized edits — which is what forced the one-shot framing.

**Status:**
- Infra: runs reproducibly on Kaggle GPU via API, no manual babysitting.
- Ground truth: OWL-ViT detects people + attributes, ownership by bounding-box containment.
- Real signal discrimination shown: a leak bug (attribute credited to both subjects) was fixed
  and confirmed resolving correctly on a live GPU run.
- Real CoIG dataset items already scored end-to-end (item_index 13,19,52,62,68,82,129,142).
- **Open, unresolved:** the ownership-threshold fix (`ATTR_OWNERSHIP_THRESHOLD=0.14`) traded a
  false positive for false negatives (legitimate attributes like "red apron"/"white hat" dropped
  on follow-up prompts) — ground truth is brittle, not yet trustworthy.
- **Open, unresolved, but with a concrete lead found 2026-07-22 (code reading only, not yet
  GPU-verified):** 2-subject binding scores *below chance* (n=2 lift −0.083, n=3 lift −0.067,
  only n=4 clears chance). Tracing cell execution order in `ssa/coig_ssa_colab.ipynb`: the
  sub-chance result (cell 51, `m3`) was computed using the *original* `phrase_attention`/
  `AttentionStore` (cell 34, `ATTN_RES=16`) — which averages cross-attention over the **entire**
  denoising trajectory, all steps. A *later* cell (58) redefines `phrase_attention` with
  `t_lo=0, t_hi=0.5` (early-window restriction only) — but that fix was introduced for the
  OWL-ViT ground-truth pipeline and was never applied back to the sub-chance analysis. Averaging
  in late, texture-focused denoising steps (where layout is no longer being decided, per
  Attend-and-Excite) is a plausible, concrete, testable explanation for the sub-chance result —
  not yet a confirmed one. **Next step: re-run cell 51's exact analysis using cell 58's windowed
  `phrase_attention` instead, and check whether n=2/n=3 clear chance.**
- ~~Human-agreement anchor set (~15-20 hand-labeled images) not yet built~~ — **done
  2026-07-23/24**, both SD1.5 and SDXL (`ssa/anchor_set/`, branch `pi-level-idea`, memory
  `coig-metric-a-anchor-set`): SDXL's real win is coverage (58% vs 39%, p=0.040); binding
  accuracy only directional (pooled p=0.078, fails Holm); n=2 stays sub-chance in both.
- **Discriminant validity — done 2026-07-24, branch `sdxl`:** is `predicted_owner` tracking
  subject box size/position rather than attention content (the gap this file used to flag as
  "manifests do not store boxes")? `recompute_boxes.py` reruns the same Mask R-CNN + CLIP
  detection locally (CPU, cached images, no GPU) to recover boxes as a sidecar
  `boxes.json`; `discriminant_validity_check.py` reads them. Result, both models: **no
  box-size confound** — the metric's own picks land on the biggest box at or below chance
  (SD1.5 29.6% vs 33.3% chance, p=0.76; SDXL 42% vs 34%, p=0.15), ground truth itself
  doesn't correlate with box size (both exactly at chance, p=0.55), and confidence margin
  doesn't track box-area dominance (SD1.5 r=-0.08; SDXL r=0.16, neither significant). But
  the headline comparison is a genuinely mixed, more important finding: on SD1.5 a trivial
  "always guess the biggest box" baseline actually **beats** the attention metric's own
  accuracy (42.9% vs 33.3%, though not significant on n=21, McNemar p=0.77); on SDXL the
  metric beats that baseline (48.3% vs 24.1% at n=29, McNemar p=0.065 — trending, not
  significant). Net read: the weak signal is real, not a geometric artifact — and it only
  clears a dumb box-size heuristic on SDXL, consistent with SDXL being the model chosen
  going forward.
- **SDXL growth batch, tested this exact trend — done 2026-07-24, same day.** Grew the
  SDXL anchor set to 24 prompts (6 new, ids 18-23, Kaggle kernel v2) specifically to push
  the p=0.065 McNemar result past significance; Chayan blind-labeled all 18 new
  judgments. **Result is a genuine null, and it moved the wrong way: McNemar p=0.065
  (n=29) -> p=0.092 (n=35)** — the 6 new scored rows split evenly between the metric and
  the bigbox baseline, so the accuracy gap didn't widen (48.3%->45.7% metric,
  24.1%->25.7% baseline). The two confound-*mechanism* checks drifted toward significance
  without crossing it (bigbox win rate p=0.148->0.081; margin-area correlation
  p=0.258->0.160), which is mildly concerning in the opposite direction, though
  construct validity (ground truth vs. box size) stayed exactly at chance, unchanged.
  Read: the effect is genuinely marginal/noisy at n~30-35 — another small growth batch is
  not a reliable lever; would need a substantially larger one, or a different kind of
  evidence entirely (VQAScore baseline, causal intervention), to move this further.
- **Workstream 2 — anchor-set growth + second annotator, started 2026-07-25, IN PROGRESS
  as of 2026-07-27, branch `sdxl`.** Addresses the two problems named in the workstream
  brief: single-annotator (no inter-rater reliability) and undersized sample (68 raw
  items). Built a 137-image growth+backfill batch (`build_growth_specs.py` + a bounded-
  retry n=4 backfill), landing at 105 detected / 306 raw judgments, balanced 44/26/35
  across n=2/3/4 — comfortably past the ~200-raw floor. Formal protocol doc
  `docs/anchor-set-labeling-protocol.md` pre-registers the shared/unclear/count-broken
  handling plan the brief asked for (shared scored separately via the metric's own
  attention-margin abstention; unclear/none excluded as missing data; count-broken
  excluded entirely — rendering failure ≠ binding failure). Model: SDXL only, confirmed
  consistent throughout (§7 of the protocol doc). Sent to two second annotators, Grace and
  Akhil, for full double-coverage blind labeling (stronger than the 30-50%-subset floor the
  brief asked for). **Their labels landed on `origin/main` directly** (Akhil built his own
  parallel copy of the labeling kit at repo root rather than using `ssa/anchor_set/`) while
  the growth-batch pipeline itself stayed on local-only `sdxl` commit `e500e1d` — reconciled
  2026-07-27 by merging `origin/main` into `sdxl` and copying Grace/Akhil's
  `labels_*.json`/`counts_*.json` into the canonical `ssa/anchor_set/artifacts_sdxl/`
  location. Also wired `analyze_agreement.py` to actually use the count-broken exclusion
  `anchor_common.py` already supported but the script never called with (it silently
  no-ops when no counts file exists, so old runs are unaffected).
  **Grace finished labeling 2026-07-27** (100%, 306/306 labels, 105/105 counts — landed
  directly on `origin/main` as commit `9382f4f`, reconciled into the canonical
  `ssa/anchor_set/artifacts_sdxl/` location the same way Workstream 2's earlier merge was).
  Both annotators now fully complete. Recomputed on the full 306 overlapping judgments:
  Cohen's kappa **0.682 — still short of the κ ≥ 0.7 target, and barely moved from the
  0.681 measured on Grace's partial 161** (the earlier "expected to move" framing was
  wrong — going from 161→306 judgments didn't close the gap). Count-clean kappa (same
  pair, per-image judgment, now 105 overlapping vs. the earlier 92): **0.924**, still
  comfortably clears target. The disagreement-category breakdown (87% boundary/sentinel,
  33/38) has NOT been recomputed on the full 71-disagreement set — that number is stale
  and specific to the 161-judgment partial state; don't cite it past that scope. Separately
  concerning, unchanged since Akhil's counts were already 100%: 96/306 rows are
  count-broken, leaving only 40 scored rows — a 13% effective yield vs. the original
  23-image set's ~51%, well short of the ~150-effective floor the protocol sized the batch
  for. Not yet investigated: whether the growth/backfill seed pool is systematically harder
  to render than the original 23 prompts.

### B. Chain / Delta-Mask metric (branch `feature/spatial-semantic-alignment-metric`, single file
`pilot/spatial_semantic_alignment.py`, authored by Pranav, commit `5452a16`)

Targets the compositional-lock confound *structurally* rather than empirically. Three phases:
- **Phase A — Delta Mask:** segment the target attribute (CLIPSeg) in the current-step and
  previous-step images, `Delta = Current AND NOT Previous`. If the attribute was already locked
  in from an earlier step, delta is empty → score forced to 0, regardless of what an external
  judge would say. This is what makes it immune to the lock confound by construction.
- **Phase B — Attention:** hook cross-attention (`attn2` on UNet; forward-hook on
  `nn.MultiheadAttention` for DiT), aggregate only the early "structural layout" window (steps
  0–15 of 50), weight layers by native resolution² so high-res maps aren't diluted by coarse ones.
- **Phase C — Score:** binarize top-20% of the composite attention map, compute IoU against the
  delta mask.

**Status: validated on real diffusion output as of 2026-07-22 (branch `pi-level-idea`).** All 9
original "PASSED" scenarios ran only against synthetic, hand-built inputs. Four real bugs were
found and fixed this session by actually wiring the class into a live SD1.5 pipeline
(`pi_level_experiment/run_chain_experiment.py`) — none were visible from the original report:
1. **Phase C's percentile threshold degenerated on sparse/peaked maps** — fixed. The 85th-
   percentile *value* on a mostly-zero array landed at 0, so `attn_bin = map >= 0` matched the
   *entire image*. Replaced with exact top-k selection by rank (immune to where ties fall); a
   realistic Gaussian-bump stress test (not block-uniform) now confirms a mismatched map scores
   ~0, not the old bug's 0.1526.
2. **`unhook_pipeline` crashed on any real diffusers UNet** — fixed. `set_attn_processor({})`
   raises `ValueError` (diffusers requires the dict to have exactly one entry per attention
   layer); replaced with a single `AttnProcessor2_0()` instance, which broadcasts to every layer.
   Added a permanent regression test (Scenario 10) against a real, tiny `UNet2DConditionModel`.
3. **CFG attention dilution** — fixed. With classifier-free guidance active, the captured
   attention batch contains both the unconditional and conditional branches; neither
   `CustomAttnProcessor` nor `phase_b_cross_attention_map` separated them, unlike metric A's own
   `AttentionStore` (`cond_index`). Added `heads` tracking and an optional `cond_index` param
   (default `None` preserves old behavior for the existing synthetic tests).
4. **DiT/FLUX.1/SD3 support is almost certainly non-functional on a real pipeline** — not fixed,
   scoped out. The hook looks for `torch.nn.MultiheadAttention` submodules, but diffusers' actual
   FLUX/SD3 implementations use their own `Attention` + custom-processor pattern with joint
   attention over concatenated image+text tokens. Scenario 8 only ever exercised a hand-built
   fake transformer. **Decision: this project's claims are scoped to UNet architectures
   (SD1.5/SDXL) only** — documented directly in the module docstring.

See `pi_level_experiment/RESULTS.md` for what running this (fixed) metric on real generated
chains actually showed.

Also worth knowing: Phase A's segmenter (CLIPSeg) is the same soft-mask model the one-shot
metric's team already moved *away from* for OwlViT, because soft masks + fixed thresholds proved
brittle (see `ATTR_OWNERSHIP_THRESHOLD` saga above) — same risk class, different model. And
`_hook_unet` replaces *every* attention processor (not just attn2) with a slow, unfused manual
attention computation, costing more memory/compute across the whole UNet than necessary.

## Why neither alone clears a NeurIPS bar (literature check, done 2026-07-22)

- Cross-attention-vs-mask IoU is not a new technique — DAAM's own paper validated cross-attention
  this way, and FreeMask already uses attention-mask IoU for a different purpose.
- One-shot attribute binding (what metric A targets) already has an established, actively
  improving SOTA line: T2I-CompBench++'s Disentangled BLIP-VQA, now surpassed by
  VQAScore/L-VQAScore on exactly this correlation-with-human-judgment question. Metric A hasn't
  been benchmarked against any of these yet.
- Chain/lock-confound faithfulness (what metric B targets) has no equivalent dominant
  competitor — closest adjacent work is 2025 (BPM, ComplexBench-Edit), not attention-based, and
  the lock-specific angle looks genuinely open.
- The field itself considers VQA-judge metrics still unreliable for attribute binding and leans
  on human eval as real ground truth — meaning the human-agreement anchor-set plan (Track A,
  §7 Step B) is exactly the right kind of validation currency, not a shortcut.
- Recent (2025) mechanistic-interpretability work (ConceptAttention, causal/norm-based
  attribution) has moved past raw cross-attention as a blunt signal — both metric A and metric B
  use raw `attn2`/joint attention. Neither project has engaged with this; worth an explicit
  "why raw attention, not a sharper signal" answer or a flagged limitation.

## The combined-arc plan (decided 2026-07-22 — this is what to build next)

One-line pitch: *Existing faithfulness checks for step-by-step image generation are fooled by the
compositional lock — we show the model's own cross-attention is a validated, judge-free signal
that isn't, and use it to build a metric that structurally can't be fooled the same way.*

Structure:
1. **Motivation** — open with the CR pilot's already-executed result (0.83 vs 0.83 identical
   persistence). Real hook, already in hand.
2. **Related work** — DAAM/Attend-and-Excite (attention is causally meaningful — established, not
   assumed), FreeMask (attention-IoU precedent), T2I-CompBench++/VQAScore (the one-shot baseline
   metric A doesn't compete with head-on, it validates a different internal signal),
   BPM/ComplexBench-Edit (closest chain-confound work, not attention-based — the gap).
3. **Part A — validate the primitive (metric A, one-shot setting).** Not the headline —
   the foundation. Human-agreement anchor set, Tier-1 falsification (attention-randomization,
   positive/negative controls), resolve the sub-chance n=2 anomaly one way or the other. Ends
   with a defensible, bounded claim: "attention tracks binding at rate X, breaks down under Y."
4. **Part B — apply it where existing metrics fail (metric B, chain setting).** Built explicitly
   on Part A's validated signal. **The money result: re-run metric B on the CR pilot's actual
   real/shuffled/substituted chains and show it correctly distinguishes them where
   persistence-based scoring (0.83 vs 0.83) could not.** This is the single most important
   experiment in the paper — everything else is supporting cast. Requires fixing the two code
   issues above first (percentile degeneracy, DiT scope decision) and stress-testing CLIPSeg the
   way OwlViT already got stress-tested.
5. **Robustness/discussion** — causal intervention via Attend-and-Excite's own steering (fix a
   mis-bound prompt, confirm score rises; suppress attention, confirm it falls — no competing
   metric can run this as cleanly), discriminant validity (rule out bounding-box area / CLIPScore
   as confounds), honest limitations section (raw attention vs. ConceptAttention-class
   alternatives).

Why this order: a reviewer who doesn't trust attention as a signal won't be moved by a clever
application of it. Validate the primitive, then spend that trust on the hard problem — same
"de-risk before you build on it" logic already used to justify going one-shot-first originally.

## Priority next steps, in order

1. ~~Fix the chain-branch code issues~~ — **done 2026-07-22** (percentile threshold, DiT scope
   decision, plus two more bugs found in the process: `unhook_pipeline` crash, CFG attention
   dilution — see metric B's Status above).
2. ~~Run metric B on real/shuffled/substituted chains — the money result~~ — **done 2026-07-22**,
   statistically significant (p=0.013, p=0.004). Not on the actual CR-pilot Gemini chains (closed
   model, can't be attention-hooked) but on matched SD1.5 chains built for this purpose — see
   `pi_level_experiment/RESULTS.md` for the full result and its honest limitations.
3. ~~The Step 4 confirmatory run~~ — **done 2026-07-23**, then **fully closed out by a growth
   run the same day.** Design doc (`docs/part-b-strengthening-design.md`, commit `d46973c`)
   implemented end-to-end: `generate_chains.py` on Kaggle (kernel
   `chayanmalkari/coig-pi-level-generate-chains` v1, SEEDS=[42,7,1234], 19/27 chains),
   `segment_cache.py`, `calibrate_threshold.py` (froze **T=0.85** on the calibration set,
   prompts 3/4/6/8 only), `score_chains.py` + `analyze_results.py`. That run left one gap:
   under the clustered (per-prompt) test, real beat every control except
   `attn_scrambled_sameattr` (p=0.078, n=7/9 — prompts 3 and 8 each had only one detected
   seed, so neither had a same-attribute/different-seed partner to pair against).
   **Growth run (kernel v2, SEEDS=[2024] only, applied uniformly to all 9 prompts —**
   deliberately *not* targeted retries on just prompts 3/8, to avoid outcome-driven
   selection**):** detected on 7/9 prompts, closing prompt 8's gap (prompt 3 failed
   detection again, so it alone stays without a pairing). New manifest merged with the
   original via `merge_manifests.py` (new file, tests in
   `tests/test_merge_manifests.py`); Stage 2/3/4 rerun on the combined 26/36-chain sample
   (clears the ≥20 target). **Result: under the clustered test, real now beats every single
   control condition, including `attn_scrambled_sameattr`** (now p=0.0156, n=8/9). Full
   writeup in `pi_level_experiment/RESULTS.md`'s new top section (includes two small honest
   disclosures: one near-zero nonzero `substituted` row, 0.0000191 IoU, and real's hit rate
   still ~30%, unchanged). Remaining limitation: prompt 3 still has no same-attribute
   pairing (a 5th uniform seed could be tried if this specifically needs closing, but
   nothing currently depends on it). **Not yet done:** port this into
   `proposal/CPGA-Research-Proposal.md`'s Ideal Results / Limitations / Experimental Setup
   sections.
4. ~~Part C validation battery (stress-test the growth run's scoring machinery itself)~~ —
   **done 2026-07-23**, per `docs/part-c-validation-design.md`, pure numpy/pandas against
   already-generated artifacts, no GPU. Seven pre-registered checks, all passing or
   already-published-number-preserving: (1) `score_chains.py`'s ported Phase A/C functions
   verified bit-identical to `pilot/spatial_semantic_alignment.py`'s validated original;
   (2) all four RNG-dependent contrasts (`substituted`, `attn_scrambled_*`) are significant
   in 50/50 reseeds, not a lucky single draw; (3) all 5 control contrasts survive
   Holm-Bonferroni correction; (4) the "beats every control" headline survives dropping any
   single one of the 9 prompts; (5) both `real vs shuffled` and `real vs substituted` stay
   significant across a full 95-point (T, top-k) grid, not just the frozen operating point.
   **Two real corrections to the paper's framing came out of this, not just
   confirmations:** (6) an ablation shows `real vs shuffled`/`real vs substituted` are
   actually validating Phase A (the delta mask alone, or even literally random
   content-free "attention," reproduces the same significance) — Phase B's specific
   attention content is only tested by the `attn_scrambled_*` family, which is
   Holm-significant but weaker under the pooled test; (7) the disclosed ~30% real hit rate
   is, on this data, 100% explained by genuine cross-step attribute persistence (detected in
   both current and previous images), zero cases of "CLIPSeg found new content but attention
   missed it" — not unexplained pipeline noise as previously framed. Full detail in
   `pi_level_experiment/RESULTS.md`'s new top section. **Correction (2026-07-23, same day,
   Part D): finding (7) above was wrong and should be treated as retracted.** A follow-up
   "what's still defensible" review hand-inspected a sample of the 52 zero-IoU rows against
   the actual cached images and found several where the attribute never visibly rendered in
   *either* frame — inconsistent with genuine persistence. Tracing the mechanism: the
   original check used `curr_mask_area` (`sigmoid.mean()`, a continuous value virtually
   never exactly 0) as a proxy for "detected in curr," which is a different question from
   "does the sigmoid cross the calibrated T=0.85 threshold anywhere" — recomputing with the
   correct threshold check (`pi_level_experiment/zero_inflation_recheck.py`, new, 5 tests,
   cross-validated against known-good nonzero rows) reverses the finding: **52/52 (100%) are
   `never_detected_in_curr` (noise floor), 0/52 are genuine persistence** — the exact
   opposite of what was published. The ~30% hit rate reverts to its original framing:
   unexplained SD1.5/CLIPSeg pipeline noise, not the lock working as designed. Two other
   Part D checks this same session were reassuring, not corrections: selection effect
   (detection rate falls monotonically with subject count, 92%/67%/58% at n=2/3/4 — a new,
   previously-unquantified limitation) and discriminant validity (subject bounding-box area
   is not an independent confound on `iou` beyond what `delta_area` already explains). Full
   detail in `pi_level_experiment/RESULTS.md`'s new Part D section, now above Part C.
   **Not yet done:** port the corrected version of finding (7) — and (6) which still
   stands — into `proposal/CPGA-Research-Proposal.md`; the still-open Tier 2 gaps
   (segmenter diversity, human-agreement ground truth for metric B, SDXL replication) remain
   unstarted, tracked in the design doc's non-goals.
5. ~~Segmenter diversity (D4) + growing the attn_scrambled_* sample past n=9 groups
   (D5)~~ — **both done 2026-07-23, same session, after the user provided a Kaggle API
   key mid-session.** D4: full (not sampled) OWL-ViT cross-check of all 74 real-condition
   rows, CPU-only, no GPU needed (`pi_level_experiment/owlvit_cross_check.py`) —
   corroborates CLIPSeg rather than contradicting it (Mann-Whitney p=0.0086, an
   independent detector's confidence ranks CLIPSeg's "detected"/"never detected" calls the
   same way). D5: pushed `generate_chains.py` kernel v3 to Kaggle with 6 new n=2 prompts
   (prompt_id 9-14, at all four seeds used so far) — chosen n=2-only per D1's own finding
   that n=2 detects best. 23/24 detected (95.8%); merged into
   `pi_level_experiment/artifacts/manifest_combined_v3.json` (49/60 chains) and rescored
   (`chain_experiment_results_v8.csv`). **All five contrasts remain clustered-significant
   at n=14-15 groups (up from n=8-9), and the two that were sitting near Wilcoxon's n=9
   resolution floor move clear of it**: `attn_scrambled_crosschain` 0.0195→0.0024,
   `attn_scrambled_sameattr` 0.0156→0.0022. Full detail in `RESULTS.md`'s Part D, sections
   D4-D5. **Not yet done, disclosed in RESULTS.md rather than assumed:** Part C's
   robustness battery (Holm correction, leave-one-out, RNG sweep, T×top-k grid) has not
   been re-run against this 15-group data, only the original 9-group data; D1-D3's checks
   were not rerun on the enlarged sample either. Segmenter diversity's OWL-ViT check also
   only covers the pre-growth 74 rows, not the 6 new prompts.
6. ~~Run Part A's human-agreement anchor set on metric A; check discriminant validity~~ —
   **done 2026-07-23/24** (anchor set, both models, `pi-level-idea`) **and 2026-07-24**
   (discriminant validity / box-size confound check, `sdxl` branch) — see metric A's Status
   above for both. Still open: the notebook's own cell-51 re-run with the windowed
   `phrase_attention` (a *different* dataset/ground-truth than the anchor set, OWL-ViT not
   human) has not been done — separate, lower-priority now that the anchor set's own
   already-windowed n=2 result answers the same question with real human labels.
7. Everything else (VQAScore correlation, causal intervention via A&E) —
   strengthens a submission but isn't load-bearing for a first draft.
8. **Scaffolded 2026-07-27 on branch `sdxl`; merged to `main` 2026-07-27 now that Workstream 2
   (Grace's labeling) finished — unblocked and smoke-tested against the real, complete
   `artifacts_sdxl` data for all three annotators (chayan/grace/akhil), no crashes.** These
   are exploratory smoke-test numbers only, not a reviewed result — group mates should run
   `py -3 run_five_experiments.py --artifacts-dir artifacts_sdxl --annotator <name>` from
   inside `ssa/anchor_set/` themselves and treat the output as a first look, not a finding.
   Five Part A validation experiments,
   pre-registered in `docs/part-a-five-experiment-battery-design.md`: (1) headline accuracy
   per subject count vs. 1/n chance, with per-stratum binomial significance
   (`exp1_accuracy_by_n.py`); (2) early-window vs. full-trajectory attention accuracy
   (`exp2_window_ablation.py`) — genuinely blocked on more than just labels: needs a NEW field,
   `model_scores_full`, scaffolded as an additive patch to `generate_anchor_images_sdxl.py`
   (captured for free from the same already-hooked generation, reusing
   `phase_b_cross_attention_map` with `max_steps=NUM_INFERENCE_STEPS`) but not yet actually
   regenerated on Kaggle — that's one more pinned-seed rerun (`PIN_SEEDS_FROM_MANIFEST`
   constant, same edit-before-push convention as `GROWTH_PROMPT_IDS`), separate from and not
   blocking on Workstream 2; (3) attention-randomization falsification
   (`exp3_attention_scramble.py`) — deliberately scrambles CROSS-ITEM within a stratum, not by
   permuting one item's own scores, because the latter degenerates at n=2 (a forced swap makes
   scrambled accuracy = 1 − real accuracy by arithmetic, not by any property of attention);
   (4) nearest-subject-noun positional baseline (`exp4_positional_baseline.py`) — checked
   against all 306 real scored rows in `artifacts_sdxl/manifest.json`: this baseline currently
   equals `intended_subject` 306/306 times, since the prompt template never lets a second
   subject intervene before an attribute, so it's presently indistinguishable from "always
   guess the intended pairing" (still the right baseline, just not yet a novel one on this
   vocabulary); (5) count-clean-only vs. all-rows accuracy, side by side
   (`exp5_count_clean_subset.py`) — discovered `analyze_agreement.py` already silently wires
   `counts_<annotator>.json` into a single filtered view (added sometime before this session,
   undocumented here until now), so this experiment's actual contribution is the side-by-side
   comparison, not first-time wiring. `run_five_experiments.py` runs all five in one pass
   against any `--artifacts-dir`, degrading Experiment 2 to a clean "unavailable" message
   rather than crashing when `model_scores_full` is absent (true of the current
   `artifacts_sdxl/manifest.json`). `make_dummy_artifacts.py` generates a synthetic
   `artifacts_dummy/` (55 images, real vocabulary/phrasing, both attention windows populated)
   so the whole pipeline is smoke-tested end-to-end today; 54 new tests added (112 → 166,
   `py -3 -m pytest tests/` from inside `ssa/anchor_set/`). Everything here reads/writes only
   `artifacts_dummy/` — zero writes to `artifacts_sdxl/` or any real label/count file.

## Branch/file pointers

- `pi-level-idea` branch (current, has everything below plus everything from `ssa-metric`) —
  the combined-arc work: fixed `pilot/spatial_semantic_alignment.py` (metric B) and
  `pi_level_experiment/` (the money-result experiment: `run_chain_experiment.py`,
  `analyze_results.py`, `RESULTS.md`, `results/chain_experiment_results.csv`). Kaggle kernel
  `chayanmalkari/coig-pi-level-chain-experiment`.
- `docs/part-b-strengthening-design.md` — the Step 1-4 design for strengthening the money
  result, and its implementation, split by real cost boundary (GPU vs. pure numpy):
  `pi_level_experiment/generate_chains.py` (Stage 1, GPU/Kaggle, multi-seed + detectability
  pre-flight + manifest.json emission; pushed via `kernel-metadata-generate-chains.json` to
  Kaggle kernel `chayanmalkari/coig-pi-level-generate-chains`), `segment_cache.py` (Stage 2,
  CLIPSeg -> cached sigmoid maps keyed by (image, attribute)), `score_chains.py` (Stage 3,
  pure numpy: threshold -> delta masks -> IoU -> CSV, six conditions including the Step 2
  disjointness-by-attribute-string fix for `substituted`), `calibrate_threshold.py` (Step 3,
  refuses to run on non-calibration prompt_ids), and `analyze_results.py` (extended with a
  clustered per-prompt Wilcoxon test alongside the original pooled Mann-Whitney one).
  `merge_manifests.py` (added 2026-07-23) combines a Stage 1 manifest with a later growth
  run's manifest without regenerating existing chains on GPU. Test suite in
  `pi_level_experiment/tests/` (51 tests as of the Part C battery below, `py -3 -m pytest
  tests/` run from inside `pi_level_experiment/` — running from the repo root fails to
  import the stage modules). `run_chain_experiment.py` itself is untouched — still the v4
  provenance record. Stage 1 has now run twice on Kaggle: kernel v1 (SEEDS=[42,7,1234],
  19/27 chains) and kernel v2 (SEEDS=[2024] growth run, 7 more chains, 26/36 total) — see
  RESULTS.md.
- `docs/part-c-validation-design.md` (added 2026-07-23) — the 7-step design for stress-testing
  the growth run's own scoring machinery, entirely pure numpy/pandas against already-generated
  artifacts (no GPU): `equivalence_check.py` (Step 1, new — asserts `score_chains.py`'s ported
  Phase A/C functions match `pilot/spatial_semantic_alignment.py` bit-for-bit), `rng_sweep.py`
  (Step 2, new — reseeds `score_chains`'s RNG-dependent conditions to check p-value
  stability), `analyze_results.py`'s `holm_correction`/`leave_one_out_check`/
  `joint_threshold_topk_sweep` (Steps 3-5), and `score_chains.py`'s `threshold_pct` param plus
  the `delta_area`/`iou_random_attn`/`curr_mask_area`/`prev_mask_area` columns (Steps 5-7 —
  note the ablation RNG is a separate `np.random.RandomState` stream from the existing
  `random.Random` one, so adding it doesn't perturb `substituted`/`attn_scrambled_*`'s
  selections at a given seed; verified both by a regression test and by re-scoring the real
  manifest and confirming a bit-identical match to the published growth-run table). Full
  findings in `RESULTS.md`'s new top section.
- `ssa-metric` branch — `ssa/coig_ssa_colab.ipynb` (metric A, one-shot binding, ~9MB notebook,
  read/edit via JSON manipulation script, not the Read/NotebookEdit tools directly — too large).
- `origin/feature/spatial-semantic-alignment-metric` branch (not checked out locally) —
  Pranav's original, unfixed `pilot/spatial_semantic_alignment.py` (metric B, chain/Delta-Mask,
  728 lines). The fixed version lives on `pi-level-idea` now; port fixes back here before
  merging, since this is Pranav's branch, not this session's.
- `pilot/` — also has the CR pilot code (`causal_relevance.py`, `judge.py`, `build_conditions.py`,
  `score_pilot.py`, results CSVs) — shared ancestor of both SSA branches.
- `proposal/SSA-Metric-Memo.md` — the research memo arguing for continuing metric A (pre-dates
  the decision to combine with metric B; still useful for the §7/§8 falsification-battery detail
  referenced throughout this file).
