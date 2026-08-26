# Metric-A Anchor Set — Growth Round Results (2026-07-27)

> **Provenance warning — do not cite the per-experiment numbers below.** This document was
> written against an earlier `anchor_common.py`. The `anchor_common.py` sync in the FLUX merge
> shifted row counts: annotator3's n=4 stratum is recorded here as 23 rows / 17.4%, where current code
> gives 21 / 19.0%. The κ figures and the protocol narrative still stand; the battery numbers do
> not. Canonical, single-code-state outputs are
> `ssa/anchor_set/artifacts_sdxl/five_experiments_<annotator>.{json,md}`, and the paper's tables
> are in [`raw-attention-paper-briefing.md`](raw-attention-paper-briefing.md).
>
> One figure below is also stale in scope: the "87% of disagreements (33/38)" breakdown was
> computed on Annotator 2's partial 161 judgments, not the full 306. It has not been recomputed.

Companion to `docs/anchor-set-labeling-protocol.md`. Produced by:
```
py -3 analyze_agreement.py --annotator annotator3 --artifacts-dir artifacts_sdxl --compare-annotator annotator2
```
Raw stdout saved verbatim below (summary); row-level detail is in `artifacts_sdxl/agreement_akhil.csv`.
Both Annotator 3 and Annotator 2 have reached 100% coverage (306/306 labels, 105/105 counts each), achieving full double coverage for the growth round as required by the protocol.

## Inter-rater reliability (the Workstream 2 deliverable)

```
Inter-rater reliability: annotator3 vs annotator2
  overlapping judgments : 306
  raw agreement         : 235/306 = 76.8%
  chance agreement      : 27.1%
  Cohen's kappa         : 0.682
  categories used       : absent, barista, chef, cyclist, farmer, none, nurse, pilot, shared, teacher, unclear
```

**κ = 0.682, short of the target κ ≥ 0.7** — computed on the full set of 306 judgments across both annotators. Per the protocol (§5), this is the final inter-rater reliability measurement for the expanded anchor set.

87% of disagreements (33/38) involve a boundary/sentinel call — `unclear` vs. `shared` vs. `none`
vs. naming a real subject — not disagreement about *who* owns an attribute once both annotators
agree it's clearly present. The four-way Present/Missing/Shared/Unclear taxonomy's edges are the
soft spot, not core binding judgment.

Count-clean κ (same annotators, per-image judgment, n=92 overlapping): **0.914** — comfortably
above target.

## Metric-vs-human accuracy (Annotator 3 as reference, strict scoring)

```
 stratum | labeled | scored | correct | accuracy |  chance |  margin
--------------------------------------------------------------------
     n=2 |      88 |     18 |       9 |    50.0% |   50.0% |    0.0%
     n=3 |      78 |     14 |       8 |    57.1% |   33.3% |   23.8%
     n=4 |     140 |      8 |       3 |    37.5% |   25.0% |   12.5%
--------------------------------------------------------------------
 overall |     306 |     40 |      20 |    50.0% |     -   |     -
```

96 of 306 rows excluded as **count-broken** (image didn't render the exact requested subject
count — Annotator 3's own per-image judgment), on top of the usual none/unclear/shared exclusions.
Effective n = 40/306 = **13%** pass rate, well below the ~150-effective floor the protocol sized
the ~300-raw growth batch for (and well below the original 23-image set's ~51% strict pass rate).
This growth batch's generations are shakier than the original set — worth a closer look before
treating the accuracy numbers above as a clean replication, independent of the kappa/coverage gap.

## Status against the original ask

| Deliverable | Status |
|---|---|
| Labels file per annotator | done — copied into `artifacts_sdxl/{labels,counts}_{annotator3,annotator2}.json` |
| Inter-rater agreement number | computed and committed here — 0.682, full data, short of 0.7 |
| Protocol doc | done — `docs/anchor-set-labeling-protocol.md` |
| Full double coverage (both annotators) | done — Annotator 3 100%, Annotator 2 100% |
| Effective-n floor (~150) for binding accuracy | not met — 40 achieved, driven mostly by count-broken rate |

**Done.** Annotator 2 has finished her pass and full double coverage is achieved. The low effective-n from count-broken images is a separate, real finding worth investigating: is the growth batch's prompt/seed pool systematically harder to render than the original 23 prompts, or is this a one-off from the backfill's retry seeds?



## Workstream 3 Experiment Results (Annotator 1)

## Experiment 1 -- accuracy per subject count

```
 stratum | labeled | scored | correct | accuracy |  chance |  p-value
---------------------------------------------------------------------
     n=2 |      16 |      9 |       5 |  55.6% |  50.0% | 0.5000
     n=3 |      24 |     13 |       5 |  38.5% |  33.3% | 0.4480
     n=4 |      28 |     13 |       6 |  46.2% |  25.0% | 0.0802
---------------------------------------------------------------------
 overall |      68 |     35 |      16 |  45.7% |     -   |      -  
```

## Experiment 2 -- early-window vs. full-trajectory attention

Experiment 2 -- UNAVAILABLE: this manifest has no model_scores_full/predicted_owner_full on every detected attribute. Rerun generate_anchor_images_sdxl.py (with the additive model_scores_full patch) against the seeds already pinned in this manifest before this experiment can run -- see docs/part-a-five-experiment-battery-design.md, Experiment 2.

## Experiment 3 -- attention-randomization falsification

- n=2: median_scrambled_accuracy=0.556 chance=0.500 falsification_clean_fraction=0.960 (n_seeds=200)
- n=3: median_scrambled_accuracy=0.308 chance=0.333 falsification_clean_fraction=0.960 (n_seeds=200)
- n=4: median_scrambled_accuracy=0.231 chance=0.250 falsification_clean_fraction=0.950 (n_seeds=200)
- McNemar (real vs. scrambled, seed=42): {'n': 35, 'real_only_correct': 7, 'scrambled_only_correct': 5, 'n_discordant': 12, 'p_value': np.float64(0.7744140625)}

## Experiment 4 -- nearest-subject-noun baseline

- Headline: {'n': 35, 'metric_accuracy': 0.45714285714285713, 'baseline_accuracy': 0.7428571428571429}
- McNemar: {'n': 35, 'metric_only_correct': 4, 'baseline_only_correct': 14, 'n_discordant': 18, 'p_value': np.float64(0.0308837890625)}
- n=2: {'n': 9, 'metric_accuracy': np.float64(0.5555555555555556), 'baseline_accuracy': np.float64(0.8888888888888888), 'chance': 0.5}
- n=3: {'n': 13, 'metric_accuracy': np.float64(0.38461538461538464), 'baseline_accuracy': np.float64(0.7692307692307693), 'chance': 0.3333333333333333}
- n=4: {'n': 13, 'metric_accuracy': np.float64(0.46153846153846156), 'baseline_accuracy': np.float64(0.6153846153846154), 'chance': 0.25}

## Experiment 5 -- count-clean subset

```
ALL ROWS (no count filter)
 stratum | labeled | scored | correct | accuracy |  chance
----------------------------------------------------------
     n=2 |      16 |      9 |       5 |  55.6% |  50.0%
     n=3 |      24 |     13 |       5 |  38.5% |  33.3%
     n=4 |      28 |     13 |       6 |  46.2% |  25.0%
----------------------------------------------------------
 overall |      68 |     35 |      16 |  45.7% |     -  

COUNT-CLEAN ONLY (count-broken images excluded)
 stratum | labeled | scored | correct | accuracy |  chance
----------------------------------------------------------
     n=2 |      16 |      9 |       5 |  55.6% |  50.0%
     n=3 |      24 |     13 |       5 |  38.5% |  33.3%
     n=4 |      28 |     13 |       6 |  46.2% |  25.0%
----------------------------------------------------------
 overall |      68 |     35 |      16 |  45.7% |     -  

(0 row(s) excluded as count-broken -- rendering failure, not binding failure.)
```

## Workstream 3 Experiment Results (Annotator 3)

## Experiment 1 -- accuracy per subject count

```
 stratum | labeled | scored | correct | accuracy |  chance |  p-value
---------------------------------------------------------------------
     n=2 |      88 |     18 |       9 |  50.0% |  50.0% | 0.5927
     n=3 |      78 |     16 |       9 |  56.2% |  33.3% | 0.0500
     n=4 |     140 |     23 |       4 |  17.4% |  25.0% | 0.8630
---------------------------------------------------------------------
 overall |     306 |     57 |      22 |  38.6% |     -   |      -  
```

## Experiment 2 -- early-window vs. full-trajectory attention

Experiment 2 -- UNAVAILABLE: this manifest has no model_scores_full/predicted_owner_full on every detected attribute. Rerun generate_anchor_images_sdxl.py (with the additive model_scores_full patch) against the seeds already pinned in this manifest before this experiment can run -- see docs/part-a-five-experiment-battery-design.md, Experiment 2.

## Experiment 3 -- attention-randomization falsification

- n=2: median_scrambled_accuracy=0.500 chance=0.500 falsification_clean_fraction=0.970 (n_seeds=200)
- n=3: median_scrambled_accuracy=0.312 chance=0.333 falsification_clean_fraction=0.990 (n_seeds=200)
- n=4: median_scrambled_accuracy=0.217 chance=0.250 falsification_clean_fraction=0.985 (n_seeds=200)
- McNemar (real vs. scrambled, seed=42): {'n': 57, 'real_only_correct': 13, 'scrambled_only_correct': 15, 'n_discordant': 28, 'p_value': np.float64(0.8505540192127228)}

## Experiment 4 -- nearest-subject-noun baseline

- Headline: {'n': 57, 'metric_accuracy': 0.38596491228070173, 'baseline_accuracy': 0.5964912280701754}
- McNemar: {'n': 57, 'metric_only_correct': 6, 'baseline_only_correct': 18, 'n_discordant': 24, 'p_value': np.float64(0.022655844688415527)}
- n=2: {'n': 18, 'metric_accuracy': np.float64(0.5), 'baseline_accuracy': np.float64(0.7222222222222222), 'chance': 0.5}
- n=3: {'n': 16, 'metric_accuracy': np.float64(0.5625), 'baseline_accuracy': np.float64(0.75), 'chance': 0.3333333333333333}
- n=4: {'n': 23, 'metric_accuracy': np.float64(0.17391304347826086), 'baseline_accuracy': np.float64(0.391304347826087), 'chance': 0.25}

## Experiment 5 -- count-clean subset

```
ALL ROWS (no count filter)
 stratum | labeled | scored | correct | accuracy |  chance
----------------------------------------------------------
     n=2 |      88 |     18 |       9 |  50.0% |  50.0%
     n=3 |      78 |     16 |       9 |  56.2% |  33.3%
     n=4 |     140 |     23 |       4 |  17.4% |  25.0%
----------------------------------------------------------
 overall |     306 |     57 |      22 |  38.6% |     -  

COUNT-CLEAN ONLY (count-broken images excluded)
 stratum | labeled | scored | correct | accuracy |  chance
----------------------------------------------------------
     n=2 |      88 |     18 |       9 |  50.0% |  50.0%
     n=3 |      78 |     14 |       8 |  57.1% |  33.3%
     n=4 |     140 |      8 |       3 |  37.5% |  25.0%
----------------------------------------------------------
 overall |     306 |     40 |      20 |  50.0% |     -  

(96 row(s) excluded as count-broken -- rendering failure, not binding failure.)
```

## Workstream 3 Experiment Results (Annotator 2)

## Experiment 1 -- accuracy per subject count

```
 stratum | labeled | scored | correct | accuracy |  chance |  p-value
---------------------------------------------------------------------
     n=2 |      88 |     28 |      18 |  64.3% |  50.0% | 0.0925
     n=3 |      78 |     17 |       9 |  52.9% |  33.3% | 0.0755
     n=4 |     140 |     18 |       7 |  38.9% |  25.0% | 0.1390
---------------------------------------------------------------------
 overall |     306 |     63 |      34 |  54.0% |     -   |      -  
```

## Experiment 2 -- early-window vs. full-trajectory attention

Experiment 2 -- UNAVAILABLE: this manifest has no model_scores_full/predicted_owner_full on every detected attribute. Rerun generate_anchor_images_sdxl.py (with the additive model_scores_full patch) against the seeds already pinned in this manifest before this experiment can run -- see docs/part-a-five-experiment-battery-design.md, Experiment 2.

## Experiment 3 -- attention-randomization falsification

- n=2: median_scrambled_accuracy=0.500 chance=0.500 falsification_clean_fraction=0.980 (n_seeds=200)
- n=3: median_scrambled_accuracy=0.353 chance=0.333 falsification_clean_fraction=0.985 (n_seeds=200)
- n=4: median_scrambled_accuracy=0.278 chance=0.250 falsification_clean_fraction=0.985 (n_seeds=200)
- McNemar (real vs. scrambled, seed=42): {'n': 63, 'real_only_correct': 22, 'scrambled_only_correct': 6, 'n_discordant': 28, 'p_value': np.float64(0.0037191659212112427)}

## Experiment 4 -- nearest-subject-noun baseline

- Headline: {'n': 63, 'metric_accuracy': 0.5396825396825397, 'baseline_accuracy': 0.7619047619047619}
- McNemar: {'n': 63, 'metric_only_correct': 7, 'baseline_only_correct': 21, 'n_discordant': 28, 'p_value': np.float64(0.012540951371192932)}
- n=2: {'n': 28, 'metric_accuracy': np.float64(0.6428571428571429), 'baseline_accuracy': np.float64(0.8571428571428571), 'chance': 0.5}
- n=3: {'n': 17, 'metric_accuracy': np.float64(0.5294117647058824), 'baseline_accuracy': np.float64(0.7058823529411765), 'chance': 0.3333333333333333}
- n=4: {'n': 18, 'metric_accuracy': np.float64(0.3888888888888889), 'baseline_accuracy': np.float64(0.6666666666666666), 'chance': 0.25}

## Experiment 5 -- count-clean subset

```
ALL ROWS (no count filter)
 stratum | labeled | scored | correct | accuracy |  chance
----------------------------------------------------------
     n=2 |      88 |     28 |      18 |  64.3% |  50.0%
     n=3 |      78 |     17 |       9 |  52.9% |  33.3%
     n=4 |     140 |     18 |       7 |  38.9% |  25.0%
----------------------------------------------------------
 overall |     306 |     63 |      34 |  54.0% |     -  

COUNT-CLEAN ONLY (count-broken images excluded)
 stratum | labeled | scored | correct | accuracy |  chance
----------------------------------------------------------
     n=2 |      88 |     28 |      18 |  64.3% |  50.0%
     n=3 |      78 |     11 |       7 |  63.6% |  33.3%
     n=4 |     140 |     12 |       4 |  33.3% |  25.0%
----------------------------------------------------------
 overall |     306 |     51 |      29 |  56.9% |     -  

(98 row(s) excluded as count-broken -- rendering failure, not binding failure.)
```
