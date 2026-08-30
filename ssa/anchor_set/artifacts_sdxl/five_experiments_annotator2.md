# Part A Five-Experiment Battery

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

Experiment 2 -- UNAVAILABLE: this manifest has no model_scores_full/predicted_owner_full on every detected attribute. Rerun generate_anchor_images_sdxl.py (with the additive model_scores_full patch) against the seeds already pinned in this manifest before this experiment can run.

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
