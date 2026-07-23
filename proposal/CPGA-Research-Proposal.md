# Testing the Validity of CoIG's Causal Relevance Metric - and Replacing It with a Validated Cross-Attention Signal

## Instructions

Proposals must be specific and detailed so the direction of the research is clear. If you're not sure about a given item, please mark it as such and we can discuss.

Contributors: Ane (mentor), Chayan, Grace, Akhil, Pranav

Revision note (2026-07-22): This version supersedes the original team draft. It consolidates two independently developed Track 2 prototypes - Chayan's one-shot cross-attention attribute-binding metric and Pranav's chain-level Delta-Mask attention metric - into a single combined research arc, and reports results from two experiments that have since been executed: the Track 1 Causal Relevance pilot (which confirmed the confound this proposal predicted) and a first real-image test of the combined Track 2 metric (which found a statistically significant result in exactly the case where Track 1's persistence check could not discriminate). Sections below are rewritten to reflect the current state of the work; where a result is still pending, that is stated explicitly rather than presented as done.

## Relevant Past Papers

*How is it done today, and what are the limits of current practice?*

- **Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting**
  Link: https://arxiv.org/pdf/2305.04388
  Summary: This research shows that the logical reasoning of language models can be realistic yet misleading about the way the model makes decisions. Hidden biases introduced into prompts make the model provide a realistic-looking justification of its result without ever referring to the true reason.
  Relevance: Shows the issue this proposal addresses in the imagery domain - the assumption that step-by-step generation must be faithful simply because it appears coherent.

- **Sanity Checks for Saliency Maps**
  Link: https://arxiv.org/pdf/1810.03292
  Summary: Shows that many interpretability heat maps in long-standing use were effectively edge detectors, not the explanation metrics researchers assumed - proven by randomizing model weights and checking whether the maps changed.
  Relevance: The conceptual foundation of this proposal. In the same way that paper showed saliency maps were confounded by low-level visual structure, this project tests whether Causal Relevance is confounded by the CoIG compositional lock.

- **Prompt-to-Prompt Image Editing with Cross Attention Control**
  Link: https://arxiv.org/pdf/2208.01626
  Summary: The layout and geometry of AI-generated images are set by cross-attention maps in the early stages of the diffusion process; manipulating these early-stage attention maps preserves layout despite changes to the text prompt.
  Relevance: Explains the mechanics behind why CoIG needs a compositional lock at all, and is directly relevant to interpreting Track 2's early-window attention design.

- **Measuring Faithfulness in Chain-of-Thought Reasoning**
  Link: https://arxiv.org/pdf/2307.13702
  Summary: Evaluates causal faithfulness of an LLM's reasoning chain via intervention on intermediate steps (truncation, error injection, modification), finding that intermediate steps' influence on the output varies widely across models and tasks.
  Relevance: The direct methodological ancestor of this proposal's experimental design - intervening on intermediate steps (here, via Shuffled and Substituted conditions) is the established way to test whether a model actually relies on its prior steps.

- **Faithful Chain Of Thought Reasoning**
  Link: https://arxiv.org/pdf/2301.13379
  Summary: Proposes faithful-by-construction reasoning by splitting a model into a translation stage (query to symbolic chain) and a problem-solving stage (executing the chain to an answer), so the output is produced by executing the reasoning rather than merely being accompanied by it.
  Relevance: Establishes dependence rather than testing it, giving a clean point of contrast with CoIG, and flags that a generated chain may not semantically reflect the intended output - the gap this proposal targets.

- **Can We Generate Images with CoT? Let's Verify and Reinforce Image Generation Step by Step**
  Link: https://arxiv.org/abs/2501.13926
  Summary: Introduces PARM, which adds step-by-step verification to image generation via a trained reward model that checks intermediate images and reroutes generation when a step does not match the prompt.
  Relevance: Shows the field moving toward dynamic, learned verifiers rather than hardcoded locks, and motivates why auditing CoIG's rigid lock mechanism matters for judging whether older frameworks remain reliable.

- **What the DAAM: Interpreting Stable Diffusion Using Cross Attention**
  Link: https://arxiv.org/abs/2210.04885
  Summary: Introduces DAAM (Diffusion Attentive Attribution Maps), showing that cross-attention maps in Stable Diffusion capture meaningful, spatially localized relationships between prompt tokens and generated image regions.
  Relevance: The primary precedent for treating cross-attention as an interpretable, spatially grounded signal - both Track 2 metrics depend on this claim being true, and this proposal is the first to apply it specifically to the chain-persistence confound rather than single-image interpretability.

- **High-Resolution Image Synthesis with Latent Diffusion Models**
  Link: https://arxiv.org/abs/2112.10752
  Summary: Introduces Latent Diffusion Models, performing diffusion in a compressed latent space rather than on pixels directly, the foundation for modern text-to-image systems including Stable Diffusion.
  Relevance: Essential background, since CoIG and both Track 2 metrics are built on diffusion-based generation and its progressive, step-wise construction of an image.

- **Probing and Steering Chain-of-Thought Unfaithfulness in Language Models**
  Authors: Giovanni Maria Occhipinti, Alessandro Abate, Nandi Schoots. Venue: ICLR (TTU Workshop, Main Track Oral), 2026.
  Link: https://openreview.net/pdf?id=JL8sNbnSWK
  Summary: Probes a model's internal representations rather than its text output, identifying an "honesty" direction in middle-to-late layers that predicts whether stated reasoning matches internal processing; linear steering along this direction makes reasoning up to 46% more faithful.
  Relevance: Establishes that step-by-step output can look coherent while concealing unfaithful internal processing - motivating why CoIG's external metric must be tested directly rather than trusted because its output reads well.

- **How Does Unfaithful Reasoning Emerge from Autoregressive Training? A Study of Synthetic Experiments**
  Authors: Fuxin Wang, Amr Alazali, Yiqiao Zhong. Venue: arXiv preprint (cs.LG), 2026.
  Link: https://arxiv.org/abs/2602.01017
  Summary: Using synthetic experiments, shows that under noisy data or increased task difficulty, models shift from executing genuine step-by-step computation to producing reasoning chains as a surface-level format while relying on an internal shortcut.
  Relevance: Direct backing for the hypothesis that CoIG could produce a plausible-looking chain of steps while the compositional lock, not the steps' semantic content, actually determines the final image.

- **Thought Anchors: Which LLM Reasoning Steps Matter?**
  Link: https://arxiv.org/pdf/2506.19143
  Summary: Identifies which individual steps in an LLM's chain of thought are causally load-bearing for the final answer, using counterfactual resampling rather than surface plausibility to define "mattering."
  Relevance: The direct methodological ancestor of the entire Track 2 program. Both Track 2 metrics operationalize a visual analog of this idea: does a step's content causally trace to a specific point in generation, rather than simply appear coherent in hindsight - exactly the distinction CoIG's compositional lock erases.

- **Attend-and-Excite: Attention-Based Semantic Guidance for Text-to-Image Diffusion Models**
  Link: https://arxiv.org/abs/2301.13826
  Summary: Identifies that diffusion models suffer from catastrophic neglect (a requested subject is never rendered) and incorrect attribute binding (an attribute lands on the wrong subject), and corrects both at inference time by nudging the latent so every subject token's cross-attention receives sufficient activation.
  Relevance: Provides a causal intervention, not just a correlation, showing that cross-attention content controls which subject an attribute is rendered onto. This is the load-bearing assumption behind both Track 2 metrics, and its steering mechanism is the basis of a planned robustness check described under Ideal Results.

## Motivation

**What limitation or problem are we solving, and how do we know it exists?**
CoIG introduces Causal Relevance to argue that its steps are faithful, but the compositional lock freezes each step's content so later steps cannot overwrite it - meaning a step's content persists to the final image regardless of whether that step's sub-prompt actually drove it. This makes Causal Relevance's persistence check ambiguous by construction, and CoIG's own evaluation only tested real, coherent chains, so the two possible explanations (faithfulness vs. the lock) were never separated. We no longer state this as a hypothesis: we ran the negative-control pilot this proposal originally proposed (10 chains x {Real, Shuffled, Substituted}, using CoIG's own Compositional Strategy Planner, Autoregressive Refinement Model, and Entity Collapse benchmark) and found exactly the predicted ambiguity. The appears-at-step component of Causal Relevance discriminates correctly (real = 1.00, shuffled = 0.53, substituted = 0.00), but persists-to-final does not: real = 0.83 and shuffled = 0.83 - identical - while substituted = 0.00. A metric that scores a wrong-step attribute exactly the same as a correctly placed one cannot be used as evidence of step-level faithfulness. The problem is confirmed, not merely plausible.

**Why is this limitation important?**
Causal Relevance is the one piece of evidence CoIG uses to move from "steps are readable" to "steps are faithful," and that jump is the entire basis for calling CoIG's generation process monitorable. Monitorability is what would let a downstream user trust the system's step-by-step account of how an image was built, and it depends on the steps being causally faithful, not just coherently ordered. Because the confound sits in the measuring tool itself, it comes before any question about whether CoIG's images look good, and it means the field currently has no validated way to distinguish faithful step-by-step image generation from a persistence artifact.

**Why does the proposed idea solve it, and why would it work?**
Confirming the confound (Track 1) only tells us the old metric is unreliable - it does not give the field a replacement. The fix is a metric that is immune to the lock by construction rather than one that merely detects the lock's effect after the fact: if an attribute's pixels were already present in the previous step, there is no new pixel footprint for the current step to be "faithful" to, so the metric should score zero regardless of what the lock does with those pixels afterward. We built exactly this (Track 2, detailed under Methods), and it already clears the bar the old metric failed at: on a first real-image test, it significantly separates correctly placed attributes from both wrong-step (p = 0.0133) and never-rendered (p = 0.0038) attributes - discriminating in exactly the setting where Causal Relevance's persistence score could not (0.83 vs. 0.83). This works because the new metric asks a structurally different question - did this step's newly added pixels correspond to where the model's own attention was concentrated at that step - instead of the old question of whether the attribute is merely present in the final image, which the lock guarantees a "yes" to almost by definition.

## Key Ideas / Contributions / Novelty

This proposal contributes three things beyond the original audit design:

1. **Empirical confirmation, not just a hypothesis**, that a step-wise visual generation system's own faithfulness metric is confounded by its architecture - the first negative-control test of this kind applied to image generation. Comparable negative-control faithfulness testing (e.g., Filler Tokens, intermediate-step corruption) exists in the LLM chain-of-thought literature but has not previously been ported to image generation.

2. **A lock-robust, architecture-agnostic replacement metric** (the Spatial-Semantic Alignment / Delta-Mask metric), built on a validated primitive: cross-attention content is causally linked to what gets rendered where, established by Attend-and-Excite's own intervention and DAAM's demonstration that attention maps carry meaningful spatial-semantic information. We are the first to apply an attention-versus-segmentation IoU test specifically to the step-persistence confound. DAAM validates cross-attention as an interpretability signal for a single generation, and FreeMask (arXiv:2409.20500) uses attention-mask IoU for zero-shot video-editing quality, but neither targets the chain/lock-faithfulness problem this proposal targets. The closest adjacent work, ComplexBench-Edit (arXiv:2506.12830), benchmarks complex multi-step editing instructions but relies on an external judge rather than an attention-based, judge-free signal, and does not address a persistence lock.

3. **A two-part combined design**, where a one-shot metric (Part A) validates the underlying signal before a chain metric (Part B) uses it to solve the confound. We deliberately separate these because a reviewer skeptical of raw cross-attention as a faithfulness signal will not be moved by a clever downstream application of it; Part A's role is to earn that trust on a simpler problem first, so Part B can spend it on the harder one.

**Positioning against the one-shot attribute-binding literature.** Part A's task - does an attribute bind to the correct subject - already has an actively improving, non-attention-based state of the art: T2I-CompBench++'s Disentangled BLIP-VQA (arXiv:2307.06350) and VQAScore (arXiv:2404.01291), which score binding via a VQA model's answer probability rather than internal activations. Part A is not proposed as a replacement for these metrics; it validates a different, complementary signal - the model's own internal attention rather than an external judge's opinion - that Part B then depends on. As an honest limitation, recent mechanistic-interpretability work (ConceptAttention, arXiv:2502.04320) has moved past raw cross-attention toward sharper, learned interpretability signals for diffusion transformers; both of our metrics currently use raw attention, and defending that choice, or adopting a sharper signal, is future work rather than a resolved question.

**Research Question:** Does CoIG's Causal Relevance metric measure genuine, step-level semantic faithfulness, or does it primarily detect the mechanical enforcement of its own compositional lock - and can a metric be built that is immune to that confound by construction rather than merely diagnosing it? The first half of this question is now answered (confound confirmed, Track 1); the second half has a first positive result (Track 2) with a defined path to strengthen it further.

## Methods

The research program has two tracks, run largely in sequence: Track 1 diagnoses whether the compositional lock is the source of the confound; Track 2 builds and tests a metric designed to be immune to it.

**Track 1 - Causal Relevance Pilot (executed, 2026-07-18).** Using CoIG's own vendored Compositional Strategy Planner, Autoregressive Refinement Model, and step-image judge, we generated 10 chains once each, under the lock, from CoIG's Entity Collapse benchmark. As in the original design, no images were regenerated for the negative controls - only which sub-prompt's already-judged step counted as "claimed" was changed. Real used each attribute's true introduction step; Shuffled used a different step from the same chain; Substituted borrowed an attribute from a different chain entirely, requiring new judge calls since it was never asked against these images. We logged appears-at-step and persists-to-final separately, rather than only a single composite score, specifically so a mechanical-persistence confound would be visible instead of averaged away. Result: appears-at-step discriminates correctly (real 1.00, shuffled 0.53, substituted 0.00); persists-to-final does not (real 0.83, shuffled 0.83, substituted 0.00) - the confound predicted above, confirmed on real generated chains.

**Track 2 - A Lock-Robust Faithfulness Metric.** Rather than only diagnosing the old metric, we designed a replacement that is immune to the persistence confound by construction: it measures whether an attribute's newly added pixels at a given step correspond to where the model's own cross-attention was concentrated at that step, rather than whether the attribute is present in the final image. Two independently developed prototypes were combined into this design, run as two parts so that Part A validates the underlying assumption before Part B is trusted to use it.

*Part A - Validating the Primitive (one-shot attribute binding).* Before trusting cross-attention as a faithfulness signal inside a chain, we test it in the simplest setting it could fail: does SD1.5's cross-attention correctly bind an attribute to the correct subject in a single multi-subject image (for example, does "apron" attention land on the barista or the cyclist in "a barista wearing a red apron and a cyclist wearing a yellow helmet")? We hook cross-attention layers directly and score binding against OWL-ViT-detected person boxes and attribute detections (ownership decided by bounding-box containment), restricting attention aggregation to the early denoising window (steps 0-50%) where DAAM and Attend-and-Excite both find layout is decided. Status: infrastructure runs reproducibly on GPU; a ground-truth leak (an attribute credited to both subjects) has been fixed and confirmed resolving on a live run; eight real CoIG dataset items have been scored end-to-end. Two items remain open before Part A can be called validated: (1) the attribute-ownership threshold introduced to fix the leak currently trades that false positive for false negatives on legitimate low-salience attributes, and needs recalibration against raw per-person detection scores rather than a single fixed cutoff; (2) an earlier analysis found sub-chance binding accuracy at 2 and 3 subjects, which code-tracing now attributes to that specific analysis using attention averaged over the entire denoising trajectory instead of the early-window restriction already implemented elsewhere in the same pipeline - re-running that analysis with windowed attention is the next concrete step, not yet executed.

*Part B - Applying the Primitive to the Chain (the Delta-Mask / Spatial-Semantic Alignment metric).* This is the metric that directly targets the persistence confound. It is computed in three phases:
- Phase A (Delta Mask): segment the target attribute with CLIPSeg in the current step's image and the previous step's image; Delta = Current AND NOT Previous. If the attribute was already locked in from an earlier step, the delta is empty and the score is forced to zero - this is what makes the metric structurally immune to the confound, independent of anything the lock does afterward.
- Phase B (Attention): hook cross-attention on the generating UNet, aggregating only the early structural window (steps 0-15 of 50), weighted by native layer resolution so high-resolution maps are not diluted by coarse ones.
- Phase C (Score): binarize the top 20% of the composite attention map and compute its intersection-over-union against the Delta Mask.

Before this proposal's execution, this design (authored by Pranav) had only been checked against 9 hand-built synthetic scenarios, never against real diffusion output. Wiring it into a live SD1.5 pipeline surfaced and fixed four real bugs invisible in the synthetic tests: a percentile threshold that degenerated to matching the entire image on sparse attention maps (fixed with exact top-k selection); a crash when unhooking the pipeline against a real diffusers UNet (fixed by broadcasting a single default attention processor instead of an empty dictionary); attention dilution under classifier-free guidance, where the unconditional and conditional attention batches were never separated (fixed by tracking attention heads and an optional conditional-batch index); and a decision to scope the metric's claims to UNet architectures (SD1.5/SDXL) only, since the cross-attention hook does not target diffusers' actual FLUX/SD3 joint-attention implementation.

With those fixes in place, we ran the first real-image test of the combined design - the money result this proposal was restructured to produce: 9 SD1.5 chains, each locked with a RePaint-style per-step latent blend (chosen because no compatible SD1.5 inpainting checkpoint could be loaded under this project's pinned CUDA/torch versions), covering 2-4-subject prompts already validated for reliable person-detection in Part A. Four conditions varied only which delta-mask target and attention map were compared: Real (attribute at its true step), Shuffled (same attribute, wrong step in the same chain), Substituted (an attribute from a different chain, never rendered here), and Attention-Scrambled (the real delta target, but a different attribute's attention map - porting Part A's attention-randomization falsification test into the chain setting).

```
Track 1 (executed): Causal Relevance Confound Check
+--------------------------------------------+
|  CoIG CSP + ARM (Entity Collapse benchmark) |
|  10 chains, lock ON                         |
+--------------------------------------------+
                    |
                    v
+--------------------------------------------+
|  Real / Shuffled / Substituted              |
|  sub-prompt relabeling (no regeneration)    |
+--------------------------------------------+
                    |
                    v
+--------------------------------------------+
|  appears_at_step  vs.  persists_to_final    |
|  ->  confound confirmed (0.83 vs 0.83)      |
+--------------------------------------------+

Track 2, Part A (validating the primitive - in progress)
+--------------------------------------------+
|  SD1.5 one-shot generation                  |
|  multi-subject prompts                      |
+--------------------------------------------+
                    |
                    v
+--------------------------------------------+
|  Cross-attention capture vs.                |
|  OWL-ViT ground truth                       |
+--------------------------------------------+
                    |
                    v
+--------------------------------------------+
|  Binding accuracy vs. chance;               |
|  Tier-1 falsification tests                 |
+--------------------------------------------+

Track 2, Part B (applied to the chain - first pass executed)
+--------------------------------------------+
|  SD1.5 locked chain (RePaint blend),        |
|  per-attribute steps                        |
+--------------------------------------------+
                    |
                    v
+--------------------------------------------+
|  Phase A: Delta Mask                        |
|  (CLIPSeg, current AND NOT previous)        |
+--------------------------------------------+
                    |
                    v
+--------------------------------------------+
|  Phase B: early-window cross-attention      |
|  (steps 0-15 of 50)                         |
+--------------------------------------------+
                    |
                    v
+--------------------------------------------+
|  Phase C: top-20% threshold,                |
|  IoU(attention, delta mask)                 |
+--------------------------------------------+
                    |
                    v
+--------------------------------------------+
|  Real / Shuffled / Substituted /            |
|  Attention-Scrambled comparison             |
|  ->  p = 0.013 / 0.004 / 0.38               |
+--------------------------------------------+
```

## Experimental Setup

**Track 1 baseline for comparison.** Real (positive control): CoIG's own Compositional Strategy Planner and Autoregressive Refinement Model, Entity Collapse benchmark, lock on - the setup CoIG itself evaluates under. Shuffled and Substituted (negative controls): identical fixed image sets, with only the claimed sub-prompt relabeled (Shuffled) or replaced with a foreign attribute (Substituted); no new images are generated, aside from the small number of new judge calls Substituted requires.

**Track 1 models, datasets, metrics.** Models: CoIG's own vendored planner, refinement model, and step-image judge, called via OpenRouter (google/gemini-2.5-pro for the planner, google/gemini-2.5-flash-image for the refinement model, google/gemini-2.5-flash for the judge) after Google's own free-tier quota proved unworkable at pilot scale. Dataset: CoIG's Entity Collapse benchmark (roughly 300 profession prompts), not T2I-CompBench - chosen because the repository's planner/refinement/judge pipeline is already wired for it, giving the fastest reliable end-to-end path for a 10-chain pilot. This is a documented deviation from the original T2I-CompBench plan, and CoIG's own published Causal Relevance numbers are not directly comparable as a result (see Potential Limitations). Metrics: appears-at-step and persists-to-final, logged separately so a persistence confound would be visible rather than averaged away. Sample size: n=10 chains (36 valid Real attribute claims, 30 Shuffled, 10 Substituted) - an effect-size pilot, not yet a full-scale study; the confound is already unambiguous at this scale, but scaling to a larger sample remains a lower-priority next step relative to Track 2.

**Track 2, Part A setup.** Models: Stable Diffusion 1.5 with a custom cross-attention hook; OWL-ViT for object and attribute ground truth. Dataset: a fixed set of multi-subject prompts spanning 2, 3, and 4 subjects, plus 8 real CoIG dataset items (item_index 13, 19, 52, 62, 68, 82, 129, 142) already scored end-to-end. Metric: binding accuracy against chance (subject/attribute attention concentration) and an AUC-margin score against a null condition. Status: infrastructure validated; the ground-truth ownership threshold and the sub-chance binding-accuracy anomaly at low subject counts are open, both with concrete next steps described under Methods.

**Track 2, Part B setup (the money-result experiment).** Models: the SD1.5 UNet with cross-attention hooked directly; CLIPSeg for Delta Mask segmentation; a RePaint-style latent blend for the compositional lock, used because no compatible open SD1.5 inpainting checkpoint could be loaded under this project's pinned torch/CUDA versions. Why SD1.5 and not CoIG's own model: CoIG's real chains are generated by Gemini 2.5 Flash Image through a closed API, and Part B needs to hook cross-attention inside the generating UNet, which is impossible against a closed model. This experiment validates the mechanism on an open, hookable model using an analogous locked-chain design - not CoIG's own exact pixels. This is a real scope limitation, disclosed here rather than implied away (see Potential Limitations). Dataset: after the first pass below, the experiment was re-architected into four stages (generation, segmentation caching, scoring, threshold calibration) so scoring-side iteration no longer requires GPU regeneration, and re-run with multi-seed generation (3 seeds x 9 prompts). This produced 19 of 27 attempted chains - short of a 20-chain target, but every one of the 9 prompts (including all 4 chains later used only for threshold calibration) has at least one successful chain, and none failed on all 3 seeds, so no base prompt was reworded. The CLIPSeg delta-mask threshold, previously a fixed default (0.5), is now selected by a pre-registered rule (maximize Real's nonzero rate subject to Substituted's nonzero rate staying exactly 0), swept only on the calibration-only chains before any Real-vs-control comparison is computed; this selected T = 0.85, a real, disclosed change from the earlier fixed value. Metric: the Spatial-Semantic Alignment IoU score per attribute per condition, compared across six conditions - Real, Shuffled, Substituted, and three attention-scrambled variants (see below) - with both a pooled one-sided Mann-Whitney U test (Real greater than each control, comparable to the first-pass numbers) and a clustered Wilcoxon signed-rank test on per-prompt paired differences (n=9 prompts), which governs when the two diverge since multi-seed rows within a prompt are not independent draws. Sample size: n=54 Real observations, n=112 Shuffled, n=54 Substituted, and n=54/54/47 for the three attention-scrambled variants respectively (attn_scrambled_sameattr requires a same-attribute, different-seed chain for the same prompt, which was not available for 2 of the 9 prompts at this sample size).

*First pass (superseded above, kept for provenance).* The original run built 5 of 9 planned chains (single seed); the remaining 4 were skipped because person detection failed on the attribute-free base image used to seed the chain. This yielded n=13 Real observations, n=24 Shuffled, n=13 Substituted, and n=13 Attention-Scrambled, using a single, uncalibrated Attention-Scrambled control (see Potential Limitations for how this was redesigned).

**Additional analysis.** Per-condition nonzero rate is reported alongside means, since the Delta-Mask score is frequently and validly zero (Substituted is 0 of 13 by construction) - a mean alone would understate how clean that separation is. We also report a per-subject-count (n=2/3/4) breakdown to check whether the effect holds uniformly or is driven by a subset of easier cases.

## Datasets and Evaluation

Track 1 uses CoIG's Entity Collapse benchmark (roughly 300 profession prompts, of which 10 were sampled for the pilot), evaluated with CoIG's own vendored judge - no separate benchmark is created. Track 2 Part A uses a fixed 9-prompt multi-subject set plus 8 real CoIG dataset items, evaluated against OWL-ViT-derived ground truth. Track 2 Part B reuses that same multi-subject prompt set, already validated for person-detectability, to build new SD1.5 chains, since CoIG's own chains cannot be attention-hooked.

We do not train anything in either track. Track 1's planner, refinement model, and judge are CoIG's own models, called via API. Track 2's SD1.5, OWL-ViT, and CLIPSeg are pretrained, off-the-shelf checkpoints. All data produced is the generated image chains, their captured attention maps, and the scores computed from them.

**Evaluation metrics.** Track 1 uses Causal Relevance's own appears-at-step and persists-to-final judge protocol, unmodified, applied across the three relabeling conditions. Track 2 uses the Spatial-Semantic Alignment IoU score (Part B) and binding accuracy against chance (Part A), both judge-free - scored directly from the model's own internal attention rather than an external MLLM's opinion. This removes MLLM-judge reliability as a confound in Track 2's results specifically, though it remains one for Track 1's appears/persists scores (see Potential Limitations).

## Benchmarks / Evaluation Sets

Track 1 is evaluated against CoIG's own protocol: using CoIG's exact planner, refinement model, and judge pipeline means the Real condition is directly comparable to how CoIG evaluates itself. Track 2 does not have an existing benchmark to compare against directly, since no prior work measures step-level chain faithfulness via attention rather than a judge; the closest points of comparison are DAAM (attention as an interpretability signal for a single generation), FreeMask (attention-mask IoU for video-editing quality rather than faithfulness), and ComplexBench-Edit (a multi-step editing benchmark that is judge-based rather than attention-based). None of these target the persistence-lock confound directly, which is why Track 2 uses its own Real/Shuffled/Substituted/Attention-Scrambled design as its baseline structure, deliberately mirroring Track 1's so the two results are directly comparable. Track 2's task is to succeed at the exact discrimination (0.83 vs. 0.83) where Track 1's own metric failed.

## Ideal Results

**Track 1 (achieved).** The ideal outcome was for persists-to-final to be indistinguishable across Real and Shuffled while Substituted stayed near zero, proving the lock - not faithfulness - drives the persistence signal. This is exactly what was found (0.83 / 0.83 / 0.00), confirming the confound and justifying Track 2 without qualification.

**Track 2, Part B (largely achieved, one control still short).** The ideal outcome was for the Spatial-Semantic Alignment score to significantly separate Real from Shuffled, Substituted, and every attention-scrambled control, succeeding at the discrimination Causal Relevance's persistence check could not make. The first real-image test (5 chains, single seed, one uncalibrated Attention-Scrambled control) held for Shuffled and Substituted (p = 0.0133 / 0.0038) but not for Attention-Scrambled (p = 0.38), whose scrambled attention was drawn from another attribute in the same, potentially spatially overlapping chain rather than an independent distribution - a confounded control, not a failed metric, per that run's own disclosure.

A confirmatory run (19 chains, 3 seeds, a pre-registered calibrated threshold, and the Attention-Scrambled control split into three variants - a legacy same-chain version plus two new unconfounded ones, cross-chain and same-attribute-different-seed) addressed this directly. Under the conservative, pre-registered clustered test (per-prompt paired differences, n=9 prompts, which governs when it diverges from the pooled test): Real significantly beats Shuffled (p = 0.0039), Substituted (p = 0.0039), the legacy same-chain scrambled control (p = 0.0039), and the new cross-chain scrambled control (p = 0.0391). Only the sharpest control - same attribute, different seed, same prompt, isolating whether attention is image-specific rather than merely prompt-specific - falls short (p = 0.078, on n=7 of 9 prompts since 2 prompts lacked a valid pairing at this sample size), though it trends in the same direction as every other contrast. This is the ideal outcome for four of five contrasts and a near-miss, not a failure, on the fifth.

**Track 2, Part A (in progress).** The ideal outcome is a validated claim of the form "cross-attention tracks attribute binding at rate X, and breaks down under Y," supported by a human-agreement anchor set and a full Tier-1 falsification battery (attention randomization, positive and negative controls). Neither has been run yet; this is the next major piece of unfinished work (see Potential Limitations and priority next steps below).

Combined, the ideal end state is a metric whose primitive (Part A) is validated against human judgment and adversarial falsification, and whose chain-level application (Part B) is shown, at full sample size, to reliably discriminate faithful from unfaithful step-by-step generation in exactly the case CoIG's own Causal Relevance metric cannot. The current results are real progress toward that end state, not yet the finished claim.

## Potential Limitations

- Track 2's real-image test used SD1.5, not CoIG's own Gemini-based model, because Gemini is served behind a closed API that cannot be attention-hooked. This validates the mechanism, not CoIG's own exact pixels; the two tracks are linked by a matched Real/Shuffled/Substituted design, not by sharing a generator. Closing this gap would require either an open-weights model matching CoIG's generation quality, or negotiated access to CoIG's internals.
- Track 2's confirmatory sample grew via multi-seed generation (19 of 27 attempted chains; n=54 Real observations) but still fell one chain short of a 20-chain target - every prompt has at least one successful chain and none failed on all 3 seeds, so no prompt was reworded, but the shortfall is reported rather than patched. Real's own hit rate is now 28%, *down* from the first pass's uncalibrated 46%, because the pre-registered threshold (T = 0.85) that makes Substituted provably clean also raises the bar Real must clear (see Experimental Setup). This is a real, disclosed tradeoff: the earlier fixed threshold (0.5) was never actually validated as "clean" for Substituted, it simply happened not to leak on the first pass's particular 13 rows.
- The original single Attention-Scrambled control did not reach significance and has since been redesigned into three variants: a legacy same-chain version, a cross-chain version (wrong attention from a different prompt entirely), and a same-attribute-different-seed version (the sharpest test of whether attention is image-specific, not just prompt-specific). Under the pre-registered clustered test, the legacy and cross-chain versions now reach significance (p = 0.0039 and p = 0.0391); the same-attribute-different-seed version does not yet (p = 0.078, on 7 of 9 prompts) - trending the right direction but underpowered at this sample size. Growing the same-attribute-different-seed pairing coverage (a 4th seed, or targeted reruns for the 2 prompts currently missing a pairing) is the most direct way to resolve this.
- Track 1's judge scores (appears-at-step, persists-to-final) depend on an MLLM judge (Gemini 2.5 Flash) that has not been independently validated against human assessment. Track 2's own scores avoid this by being judge-free, but Track 1's confirmed confound still rests on judge reliability.
- Track 1 used CoIG's Entity Collapse benchmark rather than T2I-CompBench, the benchmark CoIG's own published Causal Relevance numbers use, for pipeline-compatibility reasons. The pilot's 0.83 / 0.83 / 0.00 result is therefore not directly comparable to CoIG's own published figures, though the qualitative confound it reveals does not depend on that comparison.
- Both metrics currently rely on raw cross-attention as the faithfulness signal. Recent interpretability work argues raw attention is a blunter signal than newer, learned alternatives (for example, ConceptAttention for diffusion transformers); we have not yet benchmarked against these, and doing so, or explicitly defending the choice of raw attention, remains open.
- Both metrics are explicitly scoped to UNet-based diffusion architectures (SD1.5/SDXL); the cross-attention hooks used do not target FLUX/SD3's joint-attention implementation, so claims do not currently extend to DiT-based generators.

---

## Proposal Summary: Auditing and Replacing CoIG's Causal Relevance Metric

**RQ:** Is CoIG's Causal Relevance metric confounded by its own compositional lock, and can a chain-of-image-generation faithfulness metric be built that is immune to that confound by construction rather than merely diagnosing it?

**The Problem:** CoIG's persistence-based faithfulness check cannot distinguish a correctly placed attribute from a wrong-step one - both score 0.83, confirmed on real generated chains.

**Method (two-track, executed in part):** Track 1 confirmed the confound with a matched negative-control pilot on CoIG's own pipeline. Track 2 replaces diagnosis with a fix - a Delta-Mask plus cross-attention metric, validated first in a simpler one-shot setting and then applied to the chain, that forces a zero score whenever an attribute's pixels were already locked in from an earlier step, regardless of what the lock does afterward.

**Result so far:** On a confirmatory run (19 chains, 3 seeds, a pre-registered calibrated threshold), the new metric significantly discriminates Real from Shuffled (p = 0.0039), Substituted (p = 0.0039), and two of three attention-scrambled controls (legacy same-chain p = 0.0039, cross-chain p = 0.0391) under the pre-registered clustered test - succeeding exactly where Causal Relevance's own persistence check could not. The sharpest scrambled control (same attribute, different seed) is a near-miss (p = 0.078, underpowered at n=7 of 9 prompts), not a contradiction.

**Contribution:** The first empirical confirmation of an architectural confound in a step-by-step image generation system's own faithfulness metric, and a validated, architecture-agnostic replacement metric with a confirmatory positive result on real images across four of five contrasts.

**Top priorities going forward:**
1. Grow the same-attribute-different-seed pairing coverage (a 4th seed, or targeted reruns for the 2 prompts currently missing a pairing) to push the one still-underpowered scrambled control over significance, and close the one-chain gap to the 20-chain target.
2. Port the confirmatory run's numbers into an updated CPGA proposal submission draft (this document) and re-verify the four bug fixes from the first pass still hold against the multi-seed data (done - see Experimental Setup and pi_level_experiment/RESULTS.md).
3. Complete Part A's human-agreement anchor set and Tier-1 falsification battery, and resolve the sub-chance binding-accuracy anomaly at low subject counts.
4. Everything else (VQAScore correlation, causal intervention via Attend-and-Excite steering, discriminant validity against bounding-box-area or CLIPScore confounds) strengthens the eventual submission but is not load-bearing for the current draft.

---

### Revision Note

Earlier drafts of this proposal went through PI review that flagged a structural risk: if Causal Relevance simply turned out to be valid, the project would have little to contribute. That review recommended splitting the work into a lightweight confound check (Track 1) and a larger effort to design a new metric (Track 2), rather than only auditing the old one. The same review caught an internal inconsistency between the original Motivation and Methods sections - regenerating images from shuffled or substituted sub-prompts would have produced a faithful chain of a different image, not a genuine unfaithful one - which was corrected to the no-regeneration, text-only relabeling design used throughout this document. Track 2 subsequently combined two independently developed metric prototypes into the single Spatial-Semantic Alignment approach documented above, after a literature check found that combining them, rather than pursuing either alone, was necessary to clear a NeurIPS-level bar.
