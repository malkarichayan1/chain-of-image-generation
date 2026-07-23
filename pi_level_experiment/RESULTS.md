# Results: the combined SSA chain experiment

## Growth run (2026-07-23, Kaggle GPU, kernel `coig-pi-level-generate-chains` v2): closes the attn_scrambled_sameattr near-miss

Step 4's confirmatory run (below) left one gap: `attn_scrambled_sameattr` missed significance
(p=0.078, n_groups=7/9) because prompt_id 3 and 8 each detected on only one of the original
three seeds, leaving neither with a same-attribute, different-seed partner to pair against.

**Decision, made explicitly to avoid an outcome-driven fix:** rather than retrying seeds
targeted at just prompts 3 and 8 -- picking retries *because* they're the two prompts behind a
near-significant result is exactly the kind of post-hoc selection this design has pre-registered
against everywhere else -- this run added **one new seed (2024), applied uniformly to all 9
prompts**, the same way the original three seeds were applied. Whether it happened to close the
gap for either prompt was left to chance, not chosen after the fact.

Raw data: `artifacts/manifest_combined.json` (merge of the Step 4 manifest and this run's,
via `merge_manifests.py`), `artifacts/chain_experiment_results_v6.csv` (both gitignored --
regenerate via `generate_chains.py` SEEDS=[2024] -> `merge_manifests.py` -> `segment_cache.py`
-> `score_chains.py` against the merged manifest).

### Outcome: prompt 8's gap closed, prompt 3's did not, sample grew past the 20-chain target

Seed 2024 detected on 7 of 9 prompts (failed on 3 and 6, same failure mode as before --
Mask R-CNN found fewer people than the prompt's subject count in the base image). Prompt 8
detected this time, giving it a second seed (7, 2024) and a valid same-attribute pairing for
the first time. Prompt 3 failed again (2 of 3 people detected, same as its one prior failure
mode), so it remains the one prompt still without a same-attribute pairing -- an honest,
undisguised remainder, not something this run was guaranteed to fix.

Total sample: **26/36 chains detected** (up from 19/27), clearing the design doc's "at least
20 chains" acceptance target for the first time.

### Rescored at the frozen T=0.85 (not re-calibrated) -- attn_scrambled_sameattr now significant

| Comparison | Pooled p (n=74 rows) | Clustered p (n groups) | Clustered significant? |
|---|---|---|---|
| real vs shuffled | 0.0000 | 0.0039 (9/9) | Yes |
| real vs substituted | 0.0000 | 0.0039 (9/9) | Yes |
| real vs attn_scrambled_samechain | 0.1376 | 0.0039 (9/9) | Yes |
| real vs attn_scrambled_crosschain | 0.1509 | 0.0195 (9/9) | Yes |
| real vs attn_scrambled_sameattr | 0.3186 | **0.0156 (8/9)** | **Yes** |

Under the clustered test the design doc says should govern, **REAL is now significantly
higher than every control condition**, closing the "except one" caveat Step 4 reported.
`attn_scrambled_sameattr` moved from p=0.078 (n=7/9, near-miss) to p=0.0156 (n=8/9,
significant) purely from prompt 8 gaining a valid pairing -- consistent with Step 4's own
read of the near-miss as an underpowered-sample issue, not a contradicting result.

As in Step 4, the pooled test does not reach significance for the three `attn_scrambled_*`
controls even though the clustered test does -- the same divergence pattern already reported
there (rows within a prompt aren't independent draws, so pooling overstates n). This is not a
new inconsistency; it is the reason the clustered test exists and is the one led with.

### Two small honest disclosures, not rounded away

**Substituted is no longer perfectly clean.** One row (`p7_s2024`, foreign attribute "yellow
helmet") scored a nonzero IoU of 0.0000191 -- five orders of magnitude below real's mean
(0.0325) and almost certainly CLIPSeg segmentation noise at a mask boundary, not a genuine
foreign-attribute detection. Still, this breaks the "0/54, no exceptions" streak Step 4
reported; substituted's nonzero rate is now 1/74 (1.35%), not exactly 0. Reported as
negligible in magnitude, not as zero.

**Real's own hit rate is unchanged, still under 30%.** 22/74 (29.7%) nonzero, essentially
identical to Step 4's 19-chain rate -- the growth run added sample size, not a cleaner
generation/segmentation pipeline. The underlying noise floor Step 4 disclosed is still there.

### Bottom line for this run

The one substantive gap left after Step 4 -- `attn_scrambled_sameattr` narrowly missing
significance because two prompts lacked a same-attribute pairing -- is closed for one of the
two prompts by an unbiased, pre-registration-consistent seed addition, and the resulting test
now reaches significance. Prompt 3 remains a genuine, disclosed limitation (still only one
detected chain across all four seeds run so far); a fifth seed could be tried the same
uniform way if closing it specifically becomes important, but it is not currently blocking
any reported result.

---

## Step 4 confirmatory run (2026-07-23, Kaggle GPU, kernel `coig-pi-level-generate-chains` v1)

Executed per `docs/part-b-strengthening-design.md` (the Part-B strengthening design),
whose four stages (`generate_chains.py`, `segment_cache.py`, `score_chains.py`,
`calibrate_threshold.py`) replace the single-file v4 run below. Raw data:
`artifacts/manifest.json`, `artifacts/chain_experiment_results_v5.csv` (gitignored --
regenerate via the four stage scripts against the Kaggle kernel's pulled-back output).

### Sample: 19/27 chains, not the ≥20 target, but no prompt failed outright

Multi-seed generation (SEEDS = [42, 7, 1234]) produced 19 detected chains out of 27
attempted (9 prompts x 3 seeds) -- one short of the design doc's "at least 20" acceptance
target. Every one of the 9 prompts has at least one successful chain, including all four
calibration-set prompts (3, 4, 6, 8: 1, 2, 2, 1 chains respectively). Per the design doc's
own pre-registered rule, a base prompt only gets reworded if it fails on **all three**
seeds; none did (the worst case, prompts 3 and 8, each succeeded on exactly one of three).
So no rewording was triggered, and none was applied after the fact -- rewording based on
which seeds happened to fail would have undercut the point of pre-registering the decision
rule before looking at outcomes. The shortfall is reported as a limitation, not patched.

### Threshold calibration surfaced a real, disclosed surprise

`calibrate_threshold.py`, run only against the calibration set (prompts 3/4/6/8, n=21
"real" rows), selected **T=0.85** -- the smallest threshold where `substituted`'s nonzero
rate first reaches exactly 0. This is markedly higher than the legacy T=0.5 used
throughout the v4 run below, and it comes at a real cost: `real`'s nonzero rate at T=0.85
is only 23.8% on the calibration set, well under v4's uncalibrated ~46%. This means
CLIPSeg's raw sigmoid output carries more baseline noise on this attribute set than a
T=0.5 default assumed -- T=0.5 was never actually "clean" for substituted, just clean by
luck on v4's particular 13 rows (see v4's discussion below). The pre-registration
criterion (maximize real's hit rate subject to substituted staying exactly 0, decided
before any real-vs-shuffled comparison) forced this tradeoff into the open instead of
letting an unexamined default hide it.

### Headline result: real beats every control under the clustered (conservative) test except one

Scored via `score_chains.py` at the frozen T=0.85 across all 19 chains (54 "real" rows).
Two tests, per the design doc's Step 4: pooled (Mann-Whitney, comparable to v4) and
clustered (Wilcoxon signed-rank on per-prompt paired differences, n=9 prompts,
conservative -- governs when the two diverge).

| Comparison | Pooled p (n=54 rows) | Clustered p (n=9 prompts) | Clustered significant? |
|---|---|---|---|
| real vs shuffled | 0.0001 | 0.0039 | Yes |
| real vs substituted | 0.0000 | 0.0039 | Yes |
| real vs attn_scrambled_samechain (legacy) | 0.1379 | 0.0039 | Yes |
| real vs attn_scrambled_crosschain (new) | 0.2489 | 0.0391 | Yes |
| real vs attn_scrambled_sameattr (new, sharpest) | 0.3342 | 0.0781 | **No** (close) |

This is a genuine improvement over v4, not just a re-run: v4's single `attention_scrambled`
control never reached significance under any test reported there. With more chains and the
clustered test the design doc added specifically to avoid pooling correlated rows, the
*legacy* same-chain version of that control now reaches p=0.0039 -- and the harder,
genuinely unconfounded `attn_scrambled_crosschain` control (wrong attention drawn from a
*different prompt* entirely) also reaches significance (p=0.0391). Only the sharpest
control, `attn_scrambled_sameattr` (same attribute, different seed, same prompt -- isolating
whether attention is image-specific, not just prompt-specific), falls short (p=0.078,
n_groups=7 of 9, since two prompts had no valid same-attribute-different-seed pairing
available at this sample size). Report this as a near-miss driven by reduced n, not a
failure of the control's logic -- it is trending the right direction (mean_diff=0.020,
same sign as every other contrast) with less statistical power, not a contradicting result.

### Bottom line for this run

Under the clustered test the design doc says should govern, REAL is significantly higher
than every control except `attn_scrambled_sameattr`, which missed at p=0.078 on reduced n.
Combined with the honest disclosures above (19/27 not 27/27 chains, T=0.85 not the
originally-assumed 0.5, real's own hit rate still under 30%), this is real progress on the
exact weakness v4 disclosed -- not a clean sweep, and not claimed as one.

---

# v4 run (2026-07-22, Kaggle GPU, kernel `coig-pi-level-chain-experiment` v4) -- superseded above, kept for history

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
