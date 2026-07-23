# Research Memo — The Spatial-Semantic Alignment (SSA) Metric: Why It's Worth Pursuing

**CPGA / CoIG Faithfulness Project — Summer 2026**
**Author:** Chayan · **For:** Ane, Grace, Akhil, Pranav · **Date:** 2026-07-22

> Revised, implementation-informed case for **Track 2, Idea 2** (the attention-binding metric) from our proposal. It reflects what building *and first-testing* the metric has taught us, and argues for continuing — with an explicit, cheap decision gate (§7) and a falsification battery (§8) rather than an open-ended commitment.

## Bottom line up front

SSA is the strongest candidate for the new faithfulness metric our PI asked us to build in Track 2. It measures something no existing CoIG metric does — whether the model binds each **attribute** to the **correct subject** — through the model's own cross-attention, on a published, already-validated mechanism (Attend-and-Excite). Since our last working session we have (a) removed the infrastructure risk — experiments now run reproducibly and hands-free on Kaggle's GPU via API — and (b) run the first validation, which showed the mechanism *does* discriminate correct from incorrect binding, while also exposing the one real remaining risk: our automated ground truth is brittle. That risk is concrete and **cheap to test**. **Recommendation: continue — run the diagnostic in §7 and the falsification battery in §8 before committing to scale.**

## 1. Why this matters now — Track 1 made it necessary, not optional

Our Track 1 lock-mechanism check is complete and confirmed the compositional-lock confound we hypothesized. That is a real finding, but it has a direct consequence for us: it means CoIG's Causal Relevance is compromised *as a faithfulness measure*. So a new faithfulness metric is no longer a nice-to-have — Track 1 turned it into the necessary follow-through. The field already has metrics for image quality (FID) and prompt alignment (CLIPScore), but none that isolate the specific failure of **binding the wrong attribute to the wrong subject**. That gap is exactly where SSA lives.

## 2. The idea, sharpened

**Original sketch** (proposal, Idea 2): correlate a sub-prompt's cross-attention with where the object appears — a test for *catastrophic neglect* (did the subject show up at all).

**Where we've taken it:** we sharpened the target to *incorrect attribute binding* — the harder, more diagnostic failure. It is easy to check whether "a red apron" appears *somewhere*; the real question is whether the model put the apron on the **barista** rather than the **cyclist**. SSA reads the model's cross-attention to ask, per attribute, *which subject is it bound to*, and checks that against where the attribute actually lands in the generated image. This is the more novel and more useful of the two framings, and it maps directly onto the second failure mode Attend-and-Excite named.

## 3. Why it's credible, not speculative

- **Attend-and-Excite (Chefer et al., the paper behind Idea 2)** already established that SD's cross-attention is causally tied to which subjects get generated — they *intervened* on it to fix neglect and mis-binding. SSA inverts their intervention into a *measurement*: same inspection point, used to score rather than to steer. We are standing on a validated mechanism, not hoping one exists.
- **DAAM** (already in our reading list) independently validates reading SD cross-attention as a faithful attribution of prompt words to image regions.

## 4. What we've established so far (honest current state)

- **Infrastructure risk is gone.** The full pipeline now runs reproducibly on Kaggle's GPU through the API — we push a notebook, run it, and pull results without a browser or manual babysitting. Iteration is no longer blocked by environment or execution friction.
- **The signal discriminates.** We diagnosed a case where an attribute was wrongly credited to *both* subjects (a "leak"), corrected the ground-truth logic, and — confirmed on a live GPU run — the attribute then resolved to the *single correct* subject. That is concrete evidence the measurement can tell right binding from wrong, the necessary precondition for any faithfulness metric.
- Real CoIG dataset items have already been scored end-to-end.

## 5. The open risks — i.e., the research questions that make this worth doing

Being straight: we have shown we *can run the experiment* and that the signal *can* discriminate — not yet that the *metric is trustworthy at scale*. Three honest unknowns:

1. **Ground-truth reliability — the crux.** Deciding "which subject truly wears the apron" in a generated image is itself a vision problem; we use OWL-ViT to detect people and attributes and assign ownership by containment. Our first validation run showed this is brittle: the calibrated ownership threshold fixed the original leak (the "yellow helmet" credited to both subjects resolved to the correct one) but simultaneously *dropped legitimate attributes* — "red apron" and "white hat" came back unassigned on follow-up prompts. The fixed-threshold approach demonstrably trades a false positive for false negatives. Until ground truth is trustworthy, every score computed against it is suspect. **This is the central question of the project, and we should treat it as such.**
2. **Generation reliability.** SD1.5 does not always render the requested cast — one 2-subject prompt produced a single person. We cannot measure binding on subjects that were never drawn, so measurement has to be conditioned on correct rendering.
3. **An unexplained early result.** A preliminary run showed 2-subject binding scoring *below chance*. That is either a bug, a sign inversion, or a genuinely surprising fact about SD1.5's binding at low subject counts — and determining which is itself potentially a contribution (§8 says how).

## 6. Why these risks are worth taking (the asymmetry)

- **The crux is cheap to test.** We do not have to solve automated ground truth to learn whether the core idea holds. Hand-label a small set of generated images (which subject has which attribute — a few minutes of human eyes) and check whether the attention-binding score agrees. That decouples the *attention* claim from the noisy *detector*, and costs almost nothing to run.
- **Even partial success is a real contribution.** A mechanistic, cheap, architecture-relevant binding-faithfulness signal is worth having even if it is coarse — especially since it complements, and partly repairs, the lock-confounded Causal Relevance.
- **It is exactly what the PI asked for.** Track 2 was meant to *build a new metric*, not just audit an old one — and it is the more publishable half of the project.

## 7. Near-term plan and decision gate

Small, staged, with an explicit go/no-go — no open-ended commitment:

- **Step A (immediate):** we have run the first ownership-threshold validation (it exposed the over-correction in §5). The next diagnostic is to print the *raw per-subject detection scores*, not just the final assignment — telling us whether the dropped attributes are *near-misses* (→ recalibrate the threshold) or *non-detections* (→ rethink the automated-ground-truth approach). One cheap run.
- **Step B (the real validation):** build a small human-labeled anchor set (~15–20 generated images) and check whether SSA's binding judgments agree with human eyes. This validates the *attention side* independent of any automated detector.
- **Decision gate:** if SSA agrees with human labels above chance with a clear margin → green-light scaling and porting to the CoIG ARM chain. If not → we have learned *cheaply* that ground truth is the bottleneck, and can pivot within Track 2 (e.g., the anchor-ablation direction, Idea 1) without having sunk much.

## 8. Validation plan — how we'd know if the metric were fooling us

A metric earns the word "real" by surviving attempts to *break* it, not by accumulating confirmations — the lesson of *Sanity Checks for Saliency Maps* (in our reading list), which showed trusted saliency methods were secretly edge detectors because no one tried to falsify them. Below is the falsification battery for SSA. Each test states the manipulation and the result SSA **must** produce if it is genuinely measuring attention-binding; failing the prediction tells us the metric is measuring something else.

**Tier 1 — decisive and cheap (run first):**

1. **Attention-randomization sanity check — the single most important test.** Recompute SSA with the attention signal destroyed, two variants: (a) reinitialize the UNet's cross-attention weights so the maps become meaningless; (b) cheaper — permute which token's attention map feeds each attribute's score ("red apron" scored against "yellow helmet"'s map). *Prediction if real:* SSA collapses to chance. **If it survives attention-scrambling, the metric is reading the image or the detector, not attention, and the core claim fails.** Run this before anything else.
2. **Positive & negative controls — a floor and a ceiling.** *Positive:* trivial single-subject/single-attribute prompts ("a barista wearing a red apron") should score near-max; if not, the plumbing is broken. *Negative (swap control):* on a correctly-bound image, score the attribute against the *wrong* subject's region — should drop to chance. (Formalizes the `auc_null`/margin already sketched in the notebook.)
3. **Human-agreement (convergent validity)** — the Step B anchor set from §7. Hand-label ~15–20 images, measure whether SSA tracks human binding judgments, with a second annotator to establish the ceiling. Validates the attention side *independent of the automated detector*, sidestepping the ground-truth crux entirely.

**Tier 2 — strong, moderate effort:**

4. **Causal intervention — our strongest available test.** We are built on Attend-and-Excite, so use its own steering as a causal probe: take a mis-bound prompt, apply A&E to *fix* the binding, and confirm SSA rises; suppress a subject's attention and confirm SSA falls. A metric that tracks a *causal* manipulation of binding — not merely correlates with it — is very hard to explain away, and no competing metric can run this test as cleanly, because A&E is the same mechanism we measure.
5. **Known-quality model ranking.** Run identical prompts on a model known to bind better (SDXL / SD2.1) vs SD1.5. *Prediction if real:* SSA ranks the better model higher. If it cannot reproduce a known quality ordering, it is not tracking binding quality.
6. **Discriminant validity — rule out the boring explanations.** Regress SSA against nuisance variables: object bounding-box area (do big objects trivially win attention?), plain CLIPScore of the attribute (is SSA just CLIP alignment in disguise?), and color saturation. *Prediction if real:* meaningful binding signal survives after partialling these out. This is the "is it secretly edge detection?" check in our setting.

**Tier 3 — deepen the mechanistic story:**

7. **Dose-response / graded sensitivity.** Build a controlled difficulty gradient — 2 subjects with very distinct attributes → similar attributes → nearly identical. A real metric should degrade *monotonically* as binding gets harder; a graded response is far stronger evidence than a binary one.
8. **Timestep / layer localization.** Prompt-to-Prompt and A&E both say layout is decided *early* in denoising — SSA's signal should concentrate in early timesteps and mid-resolution cross-attention layers. Finding it where the mechanism predicts corroborates the story; finding it smeared everywhere undercuts it.

> **The sub-chance-at-n=2 result (§5) is itself a validity probe.** Tests #2 (does the swap control behave at n=2?) and #6 (does subject-count confound the score?) will tell us whether it is a bug, a detector artifact, or a real property — and we cannot call SSA "real" until we know which.

## 9. Scope note — an honest divergence to flag

Current SSA work runs on **one-shot SD1.5**, not the CoIG ARM per-step chain described in the proposal. This is deliberate, not drift: validate the measurement primitive in the cleanest possible setting — one model, one forward process, no editing-chain confounds — which is exactly where Attend-and-Excite validated it. Once the primitive is trusted, we port it to the CoIG chain. De-risk the measurement before the pipeline. (For the record: an editing-chain version was tried first and failed — the editors produced global recolors instead of localized edits — which is what forced the one-shot framing.)

## Bottom line

SSA is the most novel, most mechanistically grounded, and most publishable direction available to Track 2 — the one place we can contribute a genuinely new tool rather than an audit. The core risk is real but *specific and cheap to probe*, and §8 lays out exactly how we would try to disprove the metric before trusting it. Recommendation: run the §7 diagnostic and the Tier-1 falsification tests in §8, and let the decision gate — agreement with human labels, survival of attention-randomization — tell us whether to scale. That keeps our exposure small while pointing straight at the question that determines whether SSA becomes the metric this project has been looking for.
