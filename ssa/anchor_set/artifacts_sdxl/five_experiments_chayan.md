# Part A Five-Experiment Battery

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
