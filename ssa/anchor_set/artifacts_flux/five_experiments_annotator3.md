# Part A Five-Experiment Battery

## Experiment 1 -- accuracy per subject count

```
 stratum | labeled | scored | correct | accuracy |  chance |  p-value
---------------------------------------------------------------------
     n=2 |      86 |     79 |      77 |  97.5% |  50.0% | 0.0000
     n=3 |      75 |     70 |      66 |  94.3% |  33.3% | 0.0000
     n=4 |     136 |    120 |      87 |  72.5% |  25.0% | 0.0000
---------------------------------------------------------------------
 overall |     297 |    269 |     230 |  85.5% |     -   |      -  
```

## Experiment 2 -- early-window vs. full-trajectory attention

- n=2: early=0.975 full=0.975 chance=0.500 (n_scored=79)
- n=3: early=0.943 full=0.943 chance=0.333 (n_scored=70)
- n=4: early=0.725 full=0.717 chance=0.250 (n_scored=120)
- McNemar (early vs. full): {'n': 269, 'early_only_correct': 1, 'full_only_correct': 0, 'n_discordant': 1, 'p_value': 1.0}

## Experiment 3 -- attention-randomization falsification

- n=2: median_scrambled_accuracy=0.494 chance=0.500 falsification_clean_fraction=0.960 (n_seeds=200)
- n=3: median_scrambled_accuracy=0.329 chance=0.333 falsification_clean_fraction=0.955 (n_seeds=200)
- n=4: median_scrambled_accuracy=0.250 chance=0.250 falsification_clean_fraction=0.965 (n_seeds=200)
- McNemar (real vs. scrambled, seed=42): {'n': 269, 'real_only_correct': 151, 'scrambled_only_correct': 10, 'n_discordant': 161, 'p_value': np.float64(1.7749265939716998e-33)}

## Experiment 4 -- nearest-subject-noun baseline

- Headline: {'n': 269, 'metric_accuracy': 0.8550185873605948, 'baseline_accuracy': 0.7769516728624535}
- McNemar: {'n': 269, 'metric_only_correct': 44, 'baseline_only_correct': 23, 'n_discordant': 67, 'p_value': np.float64(0.013933875024287785)}
- n=2: {'n': 79, 'metric_accuracy': np.float64(0.9746835443037974), 'baseline_accuracy': np.float64(0.8860759493670886), 'chance': 0.5}
- n=3: {'n': 70, 'metric_accuracy': np.float64(0.9428571428571428), 'baseline_accuracy': np.float64(0.8142857142857143), 'chance': 0.3333333333333333}
- n=4: {'n': 120, 'metric_accuracy': np.float64(0.725), 'baseline_accuracy': np.float64(0.6833333333333333), 'chance': 0.25}

## Experiment 5 -- count-clean subset

```
ALL ROWS (no count filter)
 stratum | labeled | scored | correct | accuracy |  chance
----------------------------------------------------------
     n=2 |      86 |     79 |      77 |  97.5% |  50.0%
     n=3 |      75 |     70 |      66 |  94.3% |  33.3%
     n=4 |     136 |    120 |      87 |  72.5% |  25.0%
----------------------------------------------------------
 overall |     297 |    269 |     230 |  85.5% |     -  

COUNT-CLEAN ONLY (count-broken images excluded)
 stratum | labeled | scored | correct | accuracy |  chance
----------------------------------------------------------
     n=2 |      86 |     79 |      77 |  97.5% |  50.0%
     n=3 |      75 |     70 |      66 |  94.3% |  33.3%
     n=4 |     136 |    120 |      87 |  72.5% |  25.0%
----------------------------------------------------------
 overall |     297 |    269 |     230 |  85.5% |     -  

(0 row(s) excluded as count-broken -- rendering failure, not binding failure.)
```
