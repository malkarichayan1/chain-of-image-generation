# Part A Five-Experiment Battery

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
