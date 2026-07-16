# Testing the Validity of CoIG's Causal Relevance Metric

CPGA Research Doc — Summer 2026

## Instructions

Proposals must be specific and detailed so the direction of the research is clear. If you're not sure about a given item, please mark it as such and we can discuss.

Contributors: Ane (mentor), Chayan, Grace, Akhil, Pranav

## Relevant Past Papers

*How is it done today, and what are the limits of current practice?*

- **Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting**
  Link: https://arxiv.org/pdf/2305.04388
  Summary: This research shows that the logical reasoning of language models could be realistic yet misleading about the way the model makes decisions. Hidden biases introduced to prompts make the model provide an extremely real justification of its result without ever referring to the true reason.
  Relevance: This shows the issue our proposal will address in the field of imagery — the assumption that step-by-step thinking must always be faithful simply because it appears coherent to us.

- **Sanity Checks for Saliency Maps**
  Link: https://arxiv.org/pdf/1810.03292
  Summary: In this paper, it was shown that many interpretability heat maps that had been in use for quite some time were basically just edge detection models and not the explanation metrics that researchers assumed. This was proven through randomly initializing the weights in the model, allowing us to check the sanity of the models.
  Relevance: This can be seen as the conceptual foundation of our proposal. In the same way that the paper established that saliency maps were totally confused by visual elements, the proposed experiment intends to test whether Causal Relevance is also confounded by the CoIG compositional lock.

- **Prompt-to-Prompt Image Editing with Cross Attention Control**
  Link: https://arxiv.org/pdf/2208.01626
  Summary: The layout and geometry of the AI-generated images are defined by the cross-attention maps in the initial stages of the diffusion process. Through the manipulation of these early-stage attention maps, keeping the layout is achieved despite the changes in text prompts.
  Relevance: This shows the actual mechanics behind why CoIG needs a compositional lock at all. Knowing how diffusion models actually lock down the maps very early in the generation will be important for interpreting the findings if the metric is found to be wrong.

- **Measuring Faithfulness in Chain-of-Thought Reasoning**
  Link: https://arxiv.org/pdf/2307.13702
  Summary: Causal faithfulness of the LLM reasoning chain is evaluated using the approach of intervention of the intermediate steps, in many different ways, including truncation, error injection, and modification. The authors find that the level of influence of the intermediate steps on the output is very different between models and tasks.
  Relevance: This paper has been an inspiration for the experimental design. Intervention on intermediate stages (which we perform using "shuffled" and "irrelevant" image sequences) is described in the paper as the established practice for assessing whether the model relies on its prior stages, which therefore allows us to test the faithfulness of the CoIG model.

- **Faithful Chain Of Thought Reasoning**
  Link: https://arxiv.org/pdf/2301.13379
  Summary: Here, they propose a framework which establishes faithful-by-construction reasoning by splitting the model into a translational stage (converting query into a symbolic chain) and a problem-solving stage (producing the answer). The output is created by executing reasoning instead of generating reasoning alongside it, without any accuracy lost.
  Relevance: This paper doesn't test whether steps cause the output, it ensures dependence, so we can make clear comparisons to real step-by-step faithfulness in CoIG. It also establishes that the generated chain may not semantically reflect the intended image (since this wasn't studied in the paper), which is our gap.

- **Can We Generate Images with CoT? Let's Verify and Reinforce Image Generation Step by Step**
  Link: https://arxiv.org/abs/2501.13926
  Summary: The paper introduces a framework called PARM, which adds step-by-step verification to image generation. It trains a separate reward model that actively checks the image at intermediate steps. If a step does not accurately match the prompt, the model flags the error and chooses a better path of generation.
  Relevance: A good point of comparison for our project. It demonstrates that the field is heading toward dynamic AI verifiers rather than hardcoded locks. We use this paper to argue why auditing CoIG's rigid lock mechanism is necessary to see if older frameworks are reliable.

- **What the DAAM: Interpreting Stable Diffusion Using Cross Attention**
  Link: https://arxiv.org/abs/2210.04885
  Summary: Introduces DAAM (Diffusion Attentive Attribution Maps), a method for visualizing how different words in a text prompt influence specific regions of an image generated by Stable Diffusion. Cross-attention maps capture meaningful relationships between text inputs and image generation.
  Relevance: Provides background on interpreting diffusion models through their internal attention mechanisms, supporting the idea that intermediate processes within image generation models can be analyzed — important for evaluating whether CoIG's intermediate steps are genuinely contributing to the final output.

- **High-Resolution Image Synthesis with Latent Diffusion Models**
  Link: https://arxiv.org/abs/2112.10752
  Summary: Introduces Latent Diffusion Models, which perform the diffusion process in a compressed latent space rather than directly on pixels, improving computational efficiency while maintaining high-quality generation. Foundation for modern text-to-image systems such as Stable Diffusion.
  Relevance: Essential background since CoIG is built on diffusion-based generation. Understanding how diffusion models progressively construct images helps explain why intermediate generation steps exist and why mechanisms such as the compositional lock matter when studying causal influence on the final output.

- **Probing and Steering Chain-of-Thought Unfaithfulness in Language Models**
  Authors: Giovanni Maria Occhipinti, Alessandro Abate, Nandi Schoots
  Venue: ICLR (TTU Workshop, Main Track Oral), 2026
  Link: https://openreview.net/pdf?id=JL8sNbnSWK
  Summary: Investigates chain-of-thought faithfulness by probing a model's internal (latent) representations rather than relying only on its text output. The authors identify an "honesty" direction in the middle-to-late layers that encodes whether a model's stated reasoning matches its internal processing, and show that linear steering along this direction can make generated reasoning up to 46% more faithful.
  Relevance: Establishes that step-by-step output can appear well-structured and coherent while concealing unfaithful rationalization internally. Motivates why CoIG's mechanics must be tested directly: if language models require white-box steering to stay honest, we cannot assume an image model's external metric is valid without testing it against controlled failures.

- **How Does Unfaithful Reasoning Emerge from Autoregressive Training? A Study of Synthetic Experiments**
  Authors: Fuxin Wang, Amr Alazali, Yiqiao Zhong
  Venue: arXiv (preprint, cs.LG), 2026
  Link: https://arxiv.org/abs/2602.01017
  Summary: Uses synthetic experiments to study why unfaithful reasoning arises during autoregressive training. Under noisy data or increased task difficulty, models undergo a sharp transition away from executing genuine step-by-step computation and toward producing reasoning chains as a surface-level format, while internally relying on a shortcut to reach the answer.
  Relevance: Gives direct backing to the hypothesis by showing that models tend to generate faithful-looking but non-causal reasoning under pressure — precisely the failure mode tested for: whether CoIG produces a plausible-looking chain of steps while the compositional lock, rather than the steps' semantic content, actually determines the final image.

## Motivation

**What limitation or problem are we solving, and how do we know it exists?**
CoIG introduces Causal Relevance to prove that steps are faithful, but the metric doesn't support that claim, since the compositional lock freezes each step's content so later steps cannot overwrite it. The step persists to the end regardless of whether faithfulness or the lock drove the Causal Relevance score. This is ambiguous because the CoIG paper only tested real, semantically coherent chains of thought, so it cannot be inferred which of the two explanations drove the Causal Relevance result.

**Why is this limitation important?**
Causal Relevance is the one piece of evidence CoIG uses to go from "steps are readable" to "steps are faithful," and that jump is the whole reason CoIG can be called monitorable. Monitorability matters for safety because it's what lets us trust the system, and it depends on reasoning being causally faithful. This is a problem with the measuring tool itself, which comes before any question about whether CoIG's images look good.

**Why does the proposed idea solve it?**
The issue is that faithfulness and the lock are tangled together in every chain CoIG tested, so the fix is to pull them apart by building chains where they predict opposite results. We deliberately construct unfaithful chains, where a sub-prompt's meaning does not match the content generated at that step, giving a case with known ground truth. If Causal Relevance still scores high, the metric is measuring the lock, since meaning is the only thing removed; if it drops, the metric is tracking real semantic influence as intended. Either way the ambiguity is resolved. This mirrors the approach used in text (Zaman and Srivastava) to show the "Biasing Features" metric confuses real unfaithfulness with the lossy compression of natural-language explanation — we are porting it to images, where it hasn't been done.

**Why would the idea probably work?**
First, the confound only exists on coherent chains, so adding incoherent ones is guaranteed to separate faithfulness from the lock. Second, the strategy already worked in the text domain, where controlled ground-truth cases exposed hidden conflation in an established metric, so it has precedent. Third, the failure mode being tested has no analog in text: in LLMs a step exerts influence through the model's own computation (measured by tests like Filler Tokens), but CoIG bolts on an external mechanism that guarantees persistence regardless of whether the sub-prompt drove the content.

## Key Ideas / Contributions / Novelty

We add negative-control testing to T2I generation, the standard way to test causal faithfulness in LLMs. We find that CoIG's compositional lock forces visual persistence and hypothesize that this inflates the metric — a unique issue no one else has addressed. We also isolate that CoIG's original evaluation only tested coherent chains, while our design holds the lock constant and breaks the prompt order and semantics, separating the mechanical lock from faithfulness so we can see whether they correspond.

Currently, image benchmarks evaluate image quality or text alignment; we create an evaluation set of broken visual reasoning chains, producing a rare, objective ground truth for unfaithfulness in generative images.

**Contributions:** We move beyond auditing an existing interpretability tool. Our primary contribution is a novel, architecture-agnostic visual faithfulness metric for chain-of-image generation, translating "Thought Anchor" methodologies from LLM reasoning into the visual domain, providing an evaluation standard for future dynamic AI verifiers (like PARM) that separates true causal influence from hardcoded state persistence.

**New Findings:**
1. A definitive check on whether CoIG's compositional lock acts as a mechanical confound in current generation evaluation.
2. A new evaluation metric that measures the actual causal weight of intermediate visual steps, showing which latent reasoning states matter for the final output regardless of the prompt's textual coherence.

**Research Question:** Does CoIG's Causal Relevance metric measure genuine, step-level semantic faithfulness, or does it merely detect the mechanical enforcement of its own compositional lock? Answering this tells us whether CoIG is actually reasoning instead of mechanically generating the image while appearing to explain itself.

Additional contributions:
- A replicable methodology template for the generative-vision research community — a generalizable procedure for designing, conducting, and analyzing a matched negative-control experiment within a visual generation pipeline (Real, Shuffled, Substituted), reusable to audit other step-wise or dynamically verified visual generation models.
- A conceptual bridge between NLP safety and computer-vision interpretability: negative-control tests (corrupting intermediate processes, filler tokens) are well established in LLM safety but have not been rigorously applied to image generation. This is a first attempt to apply that stress-testing approach to the visual domain under physical/compositional constraints.
- A dataset: a carefully constructed set of intentionally "broken" visual reasoning chains (Shuffled and Substituted) — a first-ever ground truth for unfaithful visuals, useful to the AI safety community for studying what happens when a model's internal logic is disconnected from its outputs.

## Methods

This is a matched negative-control experiment using the original CoIG authors' generation code. Per-step images are generated only once per prompt, under the real, correctly-ordered sub-prompts. The Shuffled and Substituted conditions do not trigger new image generation; they reuse that same fixed set of per-step images and only change which sub-prompt text is presented as the "instruction" for each step. The compositional lock stays on throughout: if Causal Relevance is actually measuring faithfulness to the (now mismatched) sub-prompt, the score must drop in the Shuffled/Substituted conditions; if the score stays the same, it was only ever measuring the lock's mechanical persistence of the original content.

The methodology is divided into two parallel tracks:

**Track 1 — Check on Lock Mechanism.** A diagnostic check of the lock mechanism before running the full metric audit. A few prompts are run through the ARM system without the compositional lock, then with it. Baseline pixel drift frame-to-frame determines whether mechanical persistence is forced regardless of textual input.

**Track 2 — Constructing a New Visual Faithfulness Metric (the Anchor Metric).** Instead of relying only on CoIG's Causal Relevance, we construct a metric inspired by "Thought Anchors," using interventions (noise injection and/or layer truncation) on the latent spaces at intermediate generation stages. Using SSIM and CLIP semantic drift between intervened and un-intervened images, we compute a "Causal Weight Score" at each stage.

**Procedure:**
1. From T2I-CompBench (the same subset the CoIG paper uses) take ~100 color-attribute prompts.
2. Use the original authors' Compositional Strategy Planner to decompose each prompt into an ordered sequence of sub-prompts.
3. Generate the per-step images exactly once per prompt, using the authors' Autoregressive Refinement Model (ARM) under the compositional lock, driven by the real, correctly-ordered sub-prompts. This one fixed image sequence is reused across all three conditions — no new images are ever generated from the broken sub-prompts.
4. Do not regenerate any intermediate images at each step. Create two incomplete chains by only exchanging the sub-prompt text overlaid onto the previously generated image sequence:
   - **Shuffled** (tests order sensitivity): original sub-prompts assigned to fixed images in a randomly permuted order. All sub-prompts still relate to the resulting image globally but are out of sync with its generation order.
   - **Substituted** (tests full semantic mismatch): at each step, the text label is swapped for a sub-prompt from a different image chain entirely — no semantic relation between the text and the generated image at that step.
5. Apply the perturbation protocol: for every chain, alter one attribute in the sub-prompt text assigned to a step (e.g., "red bowl" → "blue bowl") without regenerating the image.
6. Use the authors' MLLM judging pipeline: for every chain, check whether the altered attribute both appears and persists to the final image — even though, for Shuffled and Substituted, that image was never actually generated from the altered text. This produces a Causal Relevance score per chain.
7. Interpret the mean Causal Relevance across the three conditions using a paired test. The pattern of results (Real ≈ Shuffled ≈ Substituted / Real > Substituted only / Real > Shuffled > Substituted) shows whether the metric is fully confounded, partially sensitive, or genuine.

```
                  ┌──────────────────────────────────────────┐
                  │         T2I-CompBench Prompts             │
                  │      Color-Attribute Set (~100)           │
                  └─────────────────────┬──────────────────────┘
                                        ▼
                  ┌──────────────────────────────────────────┐
                  │         CSP Decomposes Prompt             │
                  │        Ordered Sub-Prompt Chain           │
                  └─────────────────────┬──────────────────────┘
                                        ▼
                  ┌──────────────────────────────────────────┐
                  │    ARM Generates Base Image Sequence      │
                  │       (Under Compositional Lock)          │
                  └─────────────────────┬──────────────────────┘
                                        │
            ┌───────────────────────────┼───────────────────────────┐
            ▼                           ▼                           ▼
   [Condition 1: REAL]        [Condition 2: SHUFFLED]     [Condition 3: SUBSTITUTED]
   Keep original text          Permute original text       Swap with out-of-chain text
   order over base images       order over base images      labels over base images
   (Positive Control)          (Swap Text Only —            (Swap Text Only —
                                No Regeneration)              No Regeneration)
            │                           │                           │
            └───────────────────────────┼───────────────────────────┘
                                        ▼
                  ┌──────────────────────────────────────────┐
                  │           Apply Perturbation              │
                  │    Alter one attribute in text labels     │
                  └─────────────────────┬──────────────────────┘
                                        ▼
                  ┌──────────────────────────────────────────┐
                  │           MLLM Judge Scores                │
                  │       Causal Relevance Per Chain           │
                  └─────────────────────┬──────────────────────┘
                                        ▼
                  ┌──────────────────────────────────────────┐
                  │           Compare Conditions               │
                  │        Paired Statistical Test             │
                  └──────────────────────────────────────────┘
```

## Experimental Setup

Across the same set of images, we compare the three textual treatments (real, shuffled, substituted) against the sub-prompts, so that no sub-prompt text generates new images for the negative controls. We first obtain the original text prompts from the dataset, then use the Compositional Strategy Planner to generate an ordered list of text sub-prompts per step.

We pass the sequence through the autoregressive refinement model once with the compositional lock on, generating per-step images for this valid, correctly-ordered sequence as our baseline. We then apply the three textual treatments to the labels for the same set of images, ensuring no sub-prompt text leads to new images in the negative controls. Finally, we inject an attribute-token perturbation into the sub-prompt text at one intermediate step in each sequence, and ask an MLLM judge whether the attribute token appears in the (unchanged) final generated images, producing a metric score.

**Baseline for comparison:**
- **Real (positive control):** the true text sequence for the ordered, fixed set of images — represents the true metric value under normal operation.
- **Shuffled:** identical fixed image set, sub-prompts replaced with those from other real chains (not another generation process), tested to ensure the system does not learn based purely on ordering.
- **Substituted:** identical fixed image set, sub-prompts replaced with text guaranteed not to match the contents of the generated image at that step, testing the metric's ability to detect lack of semantic connection.

**Models:** the exact open-source implementations from the CoIG repository — Compositional Strategy Planner (CSP), Autoregressive Refinement Model (ARM), MLLM judge pipeline.
**Datasets:** T2I-CompBench, isolating ~100 color-attribute prompts — the same evaluation subset used in the original CoIG publication.
**Metrics:** CoIG's Causal Relevance score is the primary dependent variable, rather than accuracy.

**Additional analysis:**
We map out a complete sensitivity profile of the metric by evaluating performance across varying prompt structures and testing boundaries, to see whether the metric responds uniformly to text modifications or shows isolated vulnerabilities under specific conditions.

To isolate semantic faithfulness from architectural confounders, we implement an explicit architectural control: model architecture, prompt lengths, neural weights, and the compositional lock are held constant across all conditions. Since meaning is the only thing varied between real, shuffled, and substituted groups, any significant discrepancy in scoring can be attributed to semantic tracking rather than mechanical persistence of the lock.

The framework functions as an external safety-auditing pipeline; it does not structurally alter or retrain the generative system. We quantify metric reliability via the mathematical relationships between the resulting scores:
- ScoreReal ≈ ScoreShuffled ≈ ScoreSubstituted → the metric is fully confounded by the lock and fails to measure true causal faithfulness.
- ScoreReal > ScoreSubstituted and ScoreReal ≈ ScoreShuffled → partial, coarse-grained sensitivity.
- ScoreReal > ScoreShuffled > ScoreSubstituted → the metric is a granular, reliable tracker of semantic faithfulness.

**Visualizations:** cross-condition comparison plots of score distributions across the evaluation cohort, plus a structural method diagram of the negative-control pipeline.
**Statistics:** mean and standard deviation of Causal Relevance scores across all three test arms, with significance testing to classify the metric's behavior into one of the three hypothesized profiles above.

## Datasets and Evaluation

T2I-CompBench's color-binding subset serves as the prompt source — the same benchmark CoIG uses for its Causal Relevance evaluation, so results are directly comparable. For each prompt in T2I-CompBench, we produce one real image sequence, then create two negative-control sequences by relabeling the sub-prompt texts without generating any additional images.

We do not train anything — CSP, ARM, and the MLLM judge are all from the original authors' repos. The only data produced is the generated chains and their Causal Relevance scores.

**Evaluation metric:** Causal Relevance, as defined in the CoIG paper. For every chain, we flip one attribute in a sub-prompt and use an MLLM judge to test whether the change shows up and persists to the final image, giving one score per chain. We then measure the gap between Real, Shuffled, and Substituted chains with a paired t-test. If scores are the same throughout, the metric only detects the compositional lock.

## Benchmarks / Evaluation Sets

We evaluate using the color-attribute subset of T2I-CompBench, the same subset CoIG uses for its original Causal Relevance evaluation, allowing direct comparison with CoIG's methodology while providing prompts with clear object-attribute relationships that can be tested for causal influence.

In each condition, one image chain is produced with the original CoIG architecture (CSP, ARM, compositional lock, evaluation pipeline) driven by the real, correctly-ordered sub-prompts. In Shuffled and Substituted, the same image chain is used but the text associated with each step is re-labeled per condition — no new images are generated. Using the original CoIG architecture rather than substitute models avoids confounds from implementation differences.

These conditions separate semantic faithfulness from the effects of the compositional lock. The primary evaluation metric is CoIG's Causal Relevance score. Following the original evaluation procedure, we change intermediate sub-prompts and use the authors' original MLLM judging pipeline and prompts to determine whether the modified information appears and persists in the final image. Each prompt receives a Causal Relevance score for all three conditions.

**Baselines:** The original CoIG evaluation setup (the Real Chain condition) serves as the primary baseline — it contains both semantic alignment and the compositional lock, so it alone cannot determine which factor drives high Causal Relevance scores. The Shuffled and Substituted chains serve as controlled negative baselines: Shuffled tests dependence on correct step ordering; Substituted tests whether the metric stays high even when semantic relationships between steps are removed.

We compare Causal Relevance scores across the three conditions with a paired statistical test on ~100 T2I-CompBench color-attribute prompts. If scores remain similar across all conditions, Causal Relevance primarily measures the compositional lock. If the Real chain significantly outperforms the negative controls, the metric captures genuine step-by-step semantic influence. This creates a graded, falsifiable evaluation, where the outcome reveals whether Causal Relevance is fully confounded by the lock, partially sensitive to semantic influence, or successfully measuring causal faithfulness.

## Ideal Results

**Track 1 (ideal outcome):** Disabling the compositional lock leads to immediate semantic degradation in the step-by-step images, whereas the locked condition exhibits near-perfect structural stability — proving the lock is an engineering confound and justifying Track 2.

**Track 2 (ideal outcome):** The Anchor Metric identifies intermediate layers that exert a disproportionately large causal effect on the resulting image, generating a wide range of Causal Weight scores across steps despite CoIG's original Causal Relevance metric being completely flat because of the compositional lock — demonstrating the superiority of the proposed metric over the original.

The overall ideal outcome is that Causal Relevance produces similar scores across Real, Shuffled, and Substituted conditions despite large differences in the semantic relationship between sub-prompts and generated content. Because the compositional lock is held constant across all conditions, this result would suggest the metric primarily measures the lock's preservation of generated content rather than genuine causal faithfulness between intermediate steps and the final output — an important methodological finding for evaluating step-by-step image generation systems, and evidence of the need for faithfulness metrics that separate true causal influence from architectural constraints.

## Potential Limitations

- Causal Relevance evaluation depends on an MLLM judge as a proxy for human assessment of causal relevance. Because this judge has not been independently validated against human judgments, inaccuracies or biases could affect interpretation. Mitigation: compare results from multiple MLLM judges on a subset of examples to check consistency.
- Fully reproducing the original CoIG environment is a challenge. Although we have access to the authors' codebase (CSP, ARM, compositional lock, evaluation pipeline), differences in model availability, API changes, or software version drift since publication may require substitutions. Any deviations from the original setup should be documented and considered when interpreting results.
- The study evaluates Causal Relevance only within CoIG's specific architecture, where the compositional lock constrains later steps from modifying earlier content. Findings may not generalize to other forms of visual chain-of-thought generation (e.g., diffusion-based processes or unconstrained autoregressive editing) where the relationship between intermediate steps and final outputs may differ.

---

## Proposal Summary: Auditing CoIG's Causal Relevance Metric

**RQ:** Is CoIG's compositional lock causing mechanical visual persistence, and how can we develop a better measure of true causal faithfulness during step-by-step image generation?

**The Problem:** CoIG's Causal Relevance metric might just be identifying the effect of the lock's mechanical enforcement, not semantic faithfulness. If the metric is completely confounded by the architecture, the field currently has no way to audit dynamic image verifiers.

**Method (Two-Track Approach):**
- **Track 1 (Lock Check):** A lightweight ablation experiment on CoIG's compositional lock — temporarily disabling it to establish a baseline visual drift compared to the persistence the lock enforces.
- **Track 2 (Visual Faithfulness Metric):** Inspired by "Thought Anchors" research in NLP, we propose a new visual faithfulness metric that intervenes on intermediate visual representations (cross-attention maps/latents) and measures the downstream visual changes to find causally determining generation steps, rather than testing textual sub-prompts on a locked architecture.

**Contribution:** Identification of the first empirical limits of CoIG's architectural confounds, and development of a new architecture-agnostic faithfulness metric for chain-of-image generation.

---

## Summary of PI Feedback

**Overall take:** The proposal has a structural weakness — if the experimental results come back as "the metric works fine," the paper has very little to contribute.

**PI's recommended pivot** — split effort into two parallel workstreams:

1. **Lock mechanism check** (small allocation of time/people). Directly verify whether the compositional lock is doing what we think it's doing. Relatively quick to test.
   - If the lock confound is real → we have a finding.
   - If not → drop it, or write it up as a small contribution.
2. **Better faithfulness metric for chain-of-image generation** (larger allocation). Rather than only auditing an existing metric, propose a new one designed specifically for step-by-step image generation.

Suggested inspiration: *Thought Anchors: Which LLM Reasoning Steps Matter?* (https://arxiv.org/pdf/2506.19143) — looks at which reasoning steps in an LLM chain actually matter for the final answer.

### Inconsistency in the Proposal

There's a critical inconsistency between the Motivation and Methods sections that needed fixing before anything else could move forward. In Motivation, the proposal says we're building unfaithful chains where "the sub-prompt's meaning does not match the content generated at that step" — which reads as keeping the *original* per-step images and only swapping the sub-prompt text on top of them. But in Methods step 4 and the diagram, the description was of *regenerating* new per-step images from the shuffled and substituted sub-prompts before running the perturbation protocol. These are two completely different experiments, and only one of them works. If we regenerate, each new sub-prompt matches its own generated step again — we've built a faithful chain of a different image, not an unfaithful chain — there's no ground-truth unfaithfulness left to measure, and a high Causal Relevance score would tell us nothing. The correct design is *don't regenerate*: keep the original per-step images and only swap the sub-prompt text. That gives genuine unfaithfulness by construction, especially in the Substituted condition. Related: under this corrected design, Shuffled mainly tests order sensitivity while Substituted tests full semantic mismatch — these are different things, and the proposal needs to be explicit about what each condition is measuring and why both are included.

### Proposal Restructuring

The proposal was heavily reliant on finding that Causal Relevance is confounded by the compositional lock — if the results show the metric actually works, there's very little to write about, which is not a safe bet for a whole project.

Two parallel tracks:
- **Track 1** is a lighter-weight lock mechanism check: directly verify what the compositional lock is actually doing before running the full experiment. Relatively quick, and tells us whether the audit is worth pursuing at all.
- **Track 2** is designing a new faithfulness metric for chain-of-image generation, rather than only auditing an existing one. Valuable regardless of how Track 1 turns out, and a much stronger contribution overall. Starting point: *Thought Anchors: Which LLM Reasoning Steps Matter?* (https://arxiv.org/pdf/2506.19143) — studies which reasoning steps in an LLM chain actually matter for the final answer; the intuitions there are a good jumping-off point for defining "mattering" in the image-generation setting.

### Next Steps

- Fix Methods step 4 and the diagram so it's clear the per-step images are not regenerated — only the sub-prompt text is swapped. [CHAYAN, PRANAV]
  - Rewrite the Shuffled vs. Substituted descriptions to be explicit about what each condition tests (order sensitivity vs. full semantic mismatch).
- Restructure the proposal into two tracks: a lighter Track 1 for the lock mechanism check, and a larger Track 2 for designing a new faithfulness metric. [GRACE, AKHIL, ANE]

### New Faithfulness Metric Ideas

**Idea 1** — https://arxiv.org/pdf/2506.19143
Design a metric that identifies/measures visual anchors — specific steps or image regions with high influence on the final output, where modifying/removing an anchor causes a big shift in the final image. Measure the impact of specific steps by ablating them during the CoIG process and calculating the distance between the original image and the ablated image; a high delta indicates a strong visual anchor.

**Idea 2** — Attention-mechanism-based (cross-attention as an "internal spotlight")
Reference: https://arxiv.org/abs/2301.13826
Summary: At each step, the model uses cross-attention as a bridge between text and image. The authors identify that advanced generative models often fail to faithfully represent the input prompt due to **catastrophic neglect** (failing to generate subjects explicitly requested) and **incorrect attribute binding** (failing to correctly assign attributes to subjects). Their solution, Generative Semantic Nursing (GSN) — implemented as **Attend and Excite** — inspects attention maps at every denoising step to verify every subject in the prompt is receiving sufficient attention; if a subject is neglected, it applies a mathematical nudge that shifts the latent code at that moment, forcing the model to allocate more generative budget to that subject.

Proposed metric: instead of exciting neglected tokens to fix an image, use the same inspection mechanism to measure whether the model is already faithfully attending to all the sub-prompts in CoIG chains. The metric measures the Spatial-Semantic Alignment between a sub-prompt's cross-attention maps and the final generated image. A model is considered faithful if its attention spotlight at each step directly correlates with the objects in the final image.

Implementation sketch (using the existing CoIG codebase and ARM generation pipeline):
- During denoising at each step, hook into the cross-attention layers of the diffusion model to save the attention maps for each subject token in the sub-prompt.
- Use a pre-trained segmentation model to generate a mask of where the object appears in the final image.
- Calculate the Pearson correlation between the attention map and the spatial mask (how well the heat in the model's attention map matches the shape in the mask).
- A chain with high faithfulness will show high correlation across all steps.

Possibly relevant paper: https://www.mdpi.com/2078-2489/17/2/149
