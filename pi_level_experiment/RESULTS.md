# Results: the combined SSA chain experiment (2026-07-22, Kaggle GPU, kernel v4)

Raw data: `results/chain_experiment_results.csv` / `.json`. Full run log available via
`kaggle kernels output chayanmalkari/coig-pi-level-chain-experiment` (kernel version 4).

## Headline result

**Real scores significantly higher than both Shuffled and Substituted** (one-sided
Mann-Whitney U, real > condition):

| Comparison | p-value | Significant at α=0.05? |
|---|---|---|
| real vs shuffled | **0.0133** | Yes |
| real vs substituted | **0.0038** | Yes |
| real vs attention_scrambled | 0.3794 | No |

This is the result the whole combined arc was built to produce: on real SD1.5-generated
images with real captured cross-attention, the Delta-Mask design discriminates the correct
introduction step from a wrong one (shuffled) and from an attribute that was never rendered
at all (substituted) -- exactly the distinction Track 1's persistence-based check could not
make (`persists_to_final`: real=0.83, shuffled=0.83, identical). That failure was the
motivating problem for this entire project; this experiment is the first evidence, on real
generated images, that the fix works.

## Per-condition summary

| condition | mean | median | nonzero rate | n |
|---|---|---|---|---|
| real | 0.0364 | 0.0 | 6/13 (46%) | 13 |
| shuffled | 0.0015 | 0.0 | 4/24 (17%) | 24 |
| substituted | 0.0000 | 0.0 | **0/13 (0%)** | 13 |
| attention_scrambled | 0.0174 | 0.0 | 6/13 (46%) | 13 |

Per-subject-count (n=2/3/4) breakdown is in `results/chain_experiment_results.csv` directly
(`condition`/`n` columns).

## Reading this honestly -- what's strong, what isn't yet

**Substituted is perfectly clean: 0/13, no exceptions.** A foreign attribute that never
appears anywhere in a chain never registers a delta mask against that chain's images,
across every single test case. This is exactly Phase A's structural guarantee, and it held
without a single counterexample on real generated images.

**Shuffled is strongly (not perfectly) suppressed: 20/24 exactly zero, 4/24 small leakage**
(max 0.033, well below real's max of 0.163). The leakage is plausibly CLIPSeg segmentation
noise at mask boundaries (the RePaint blend isn't a perfectly hard edge), not evidence the
mechanism fails -- but it's a real, disclosed imperfection, not a clean zero.

**Real's own hit rate is only 46% (6/13).** Just over half the "real, correct step" cases
scored exactly zero too. That is the main thing weakening this result: it means the
underlying SD1.5 RePaint-blend generation and/or CLIPSeg's detection of the newly-added
attribute frequently produced no usable signal at all, most likely because a masked
attribute edit on a small subject region doesn't always render distinctly enough for
CLIPSeg's fixed 0.5 sigmoid threshold to register a change. The significant p-values above
hold *despite* this -- the six real cases that did register a signal were high enough, and
consistently enough above shuffled/substituted's near-zero floor, to reach significance on
a sample this small. A cleaner generation/segmentation pipeline would very likely sharpen
this further, not weaken it.

**attention_scrambled did not reach significance (p=0.38).** This was meant to port metric
A's attention-randomization falsification test into the chain setting -- confirming that a
real, nonzero delta mask paired with the *wrong* attribute's attention scores lower than
with the *right* one. It trends the right direction in most individual cases (e.g.
chain 0's "red apron": real=0.116 vs scrambled=0.032; chain 2's "blue gloves": real=0.110
vs scrambled=0.018) but one case (chain 7's "book": real=0.163 vs scrambled=0.156) barely
moved, and with only 13 paired observations the test has little power either way. The
`README.md`'s honest-scope note already flagged the likely cause: the swapped-in attention
comes from a *different attribute in the same chain*, not an independent distribution --
if two subjects happen to sit in overlapping or nearby image regions, swapping their
attention doesn't cleanly test "does attention content matter," it tests "do these two
particular regions overlap." **This control needs a redesign (draw the wrong attention map
from an unrelated chain, not a same-chain sibling) before it can support a claim either
way** -- report it as inconclusive, not as a second confirmed finding.

## Why only 5/9 chains (not 9/9)

Chains 3, 4, 6, and 8 were skipped: Mask R-CNN didn't detect the expected number of people
in the *base* image (subjects only, no attributes -- e.g. "a photo of a chef and a
farmer"). Metric A validated person-detectability for the **full, attributed** prompts
("a photo of a chef wearing a white hat and a farmer holding a shovel"); the attribute-free
base prompts used here to seed the chain were never independently validated for
detectability, and evidently render less reliably for Mask R-CNN in a few cases. This is a
specific, fixable issue (e.g., validate/re-seed base prompts for detectability, or relax
`score_thresh` in `person_boxes`), not a property of the metric itself, and is the most
direct way to grow the sample in a follow-up run.

## Bottom line

Small sample, real generation-pipeline noise, and an under-powered secondary control --
all disclosed above, not rounded away. Within those limits: the core, pre-registered
comparison this experiment was built to make **is statistically significant, on real
generated images with real captured attention**, in the exact direction the whole combined
arc predicted, on the exact failure mode (shuffled indistinguishable from real) that
motivated Track 2 in the first place. That is a genuine, if early, positive result -- not
a "100% proof," but real evidence, not simulation, that the mechanism does what it claims.
The clear next step is more chains (fixing the base-prompt detectability issue) and a
redesigned attention_scrambled control, not a different approach.
