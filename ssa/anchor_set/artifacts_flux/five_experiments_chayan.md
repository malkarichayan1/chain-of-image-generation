# Part A Five-Experiment Battery

## Experiment 1 -- accuracy per subject count

```
 stratum | labeled | scored | correct | accuracy |  chance |  p-value
---------------------------------------------------------------------
     n=2 |      86 |     80 |      78 |  97.5% |  50.0% | 0.0000
     n=3 |      75 |     70 |      65 |  92.9% |  33.3% | 0.0000
     n=4 |     136 |    123 |      88 |  71.5% |  25.0% | 0.0000
---------------------------------------------------------------------
 overall |     297 |    273 |     231 |  84.6% |     -   |      -  
```

## Experiment 2 -- early-window vs. full-trajectory attention

- n=2: early=0.975 full=0.975 chance=0.500 (n_scored=80)
- n=3: early=0.929 full=0.929 chance=0.333 (n_scored=70)
- n=4: early=0.715 full=0.707 chance=0.250 (n_scored=123)
- McNemar (early vs. full): {'n': 273, 'early_only_correct': 1, 'full_only_correct': 0, 'n_discordant': 1, 'p_value': 1.0}

## Experiment 3 -- attention-randomization falsification

- n=2: median_scrambled_accuracy=0.500 chance=0.500 falsification_clean_fraction=0.955 (n_seeds=200)
- n=3: median_scrambled_accuracy=0.329 chance=0.333 falsification_clean_fraction=0.975 (n_seeds=200)
- n=4: median_scrambled_accuracy=0.252 chance=0.250 falsification_clean_fraction=0.945 (n_seeds=200)
- McNemar (real vs. scrambled, seed=42): {'n': 273, 'real_only_correct': 151, 'scrambled_only_correct': 10, 'n_discordant': 161, 'p_value': np.float64(1.7749265939716998e-33)}

## Experiment 4 -- nearest-subject-noun baseline

- Headline: {'n': 273, 'metric_accuracy': 0.8461538461538461, 'baseline_accuracy': 0.7765567765567766}
- McNemar: {'n': 273, 'metric_only_correct': 43, 'baseline_only_correct': 24, 'n_discordant': 67, 'p_value': np.float64(0.027119992403126333)}
- n=2: {'n': 80, 'metric_accuracy': np.float64(0.975), 'baseline_accuracy': np.float64(0.8875), 'chance': 0.5}
- n=3: {'n': 70, 'metric_accuracy': np.float64(0.9285714285714286), 'baseline_accuracy': np.float64(0.8), 'chance': 0.3333333333333333}
- n=4: {'n': 123, 'metric_accuracy': np.float64(0.7154471544715447), 'baseline_accuracy': np.float64(0.6910569105691057), 'chance': 0.25}

## Experiment 5 -- count-clean subset

```
ALL ROWS (no count filter)
 stratum | labeled | scored | correct | accuracy |  chance
----------------------------------------------------------
     n=2 |      86 |     80 |      78 |  97.5% |  50.0%
     n=3 |      75 |     70 |      65 |  92.9% |  33.3%
     n=4 |     136 |    123 |      88 |  71.5% |  25.0%
----------------------------------------------------------
 overall |     297 |    273 |     231 |  84.6% |     -  

COUNT-CLEAN ONLY (count-broken images excluded)
 stratum | labeled | scored | correct | accuracy |  chance
----------------------------------------------------------
     n=2 |      86 |     80 |      78 |  97.5% |  50.0%
     n=3 |      75 |     70 |      65 |  92.9% |  33.3%
     n=4 |     136 |    123 |      88 |  71.5% |  25.0%
----------------------------------------------------------
 overall |     297 |    273 |     231 |  84.6% |     -  

(0 row(s) excluded as count-broken -- rendering failure, not binding failure.)
```
