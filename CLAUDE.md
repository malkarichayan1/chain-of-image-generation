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
- Human-agreement anchor set (~15-20 hand-labeled images) not yet built — this is the actual
  validation step, decoupled from the noisy OWL-ViT detector.

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
3. **Now the top priority:** strengthen the money result — fix the base-prompt person-detection
   issue that limited the sample to 5/9 chains, and redesign `attention_scrambled`'s control to
   draw wrong-attention from an unrelated chain rather than a same-chain sibling (see
   RESULTS.md's discussion of why it didn't reach significance).
4. Run Part A's human-agreement anchor set + Tier-1 falsification battery on metric A; resolve
   the sub-chance n=2 anomaly — there's now a concrete, testable lead (see metric A's Status
   above: re-run the analysis with the already-existing windowed `phrase_attention`).
5. Everything else (VQAScore correlation, causal intervention via A&E, discriminant validity) —
   strengthens a submission but isn't load-bearing for a first draft.

## Branch/file pointers

- `pi-level-idea` branch (current, has everything below plus everything from `ssa-metric`) —
  the combined-arc work: fixed `pilot/spatial_semantic_alignment.py` (metric B) and
  `pi_level_experiment/` (the money-result experiment: `run_chain_experiment.py`,
  `analyze_results.py`, `RESULTS.md`, `results/chain_experiment_results.csv`). Kaggle kernel
  `chayanmalkari/coig-pi-level-chain-experiment`.
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
