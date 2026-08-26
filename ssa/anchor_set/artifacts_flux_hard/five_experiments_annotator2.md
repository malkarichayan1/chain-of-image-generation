# Part A Five-Experiment Battery

## Experiment 1 -- accuracy per subject count

```
 stratum | labeled | scored | correct | accuracy |  chance |  p-value
---------------------------------------------------------------------
     n=4 |      96 |     78 |      42 |  53.8% |  25.0% | 0.0000
     n=5 |     205 |    158 |      69 |  43.7% |  20.0% | 0.0000
     n=6 |     108 |     72 |      18 |  25.0% |  16.7% | 0.0462
---------------------------------------------------------------------
 overall |     409 |    308 |     129 |  41.9% |     -   |      -  
```

## Experiment 2 -- early-window vs. full-trajectory attention

- n=4: early=0.538 full=0.551 chance=0.250 (n_scored=78)
- n=5: early=0.437 full=0.443 chance=0.200 (n_scored=158)
- n=6: early=0.250 full=0.236 chance=0.167 (n_scored=72)
- McNemar (early vs. full): {'n': 308, 'early_only_correct': 1, 'full_only_correct': 2, 'n_discordant': 3, 'p_value': 1.0}

## Experiment 3 -- attention-randomization falsification

- n=4: median_scrambled_accuracy=0.250 chance=0.250 falsification_clean_fraction=0.960 (n_seeds=200)
- n=5: median_scrambled_accuracy=0.196 chance=0.200 falsification_clean_fraction=0.915 (n_seeds=200)
- n=6: median_scrambled_accuracy=0.167 chance=0.167 falsification_clean_fraction=0.965 (n_seeds=200)
- McNemar (real vs. scrambled, seed=42): {'n': 308, 'real_only_correct': 102, 'scrambled_only_correct': 35, 'n_discordant': 137, 'p_value': np.float64(8.691651732727622e-09)}

## Experiment 4 -- nearest-subject-noun baseline

- Headline: {'n': 308, 'metric_accuracy': 0.41883116883116883, 'baseline_accuracy': 0.801948051948052}
- McNemar: {'n': 308, 'metric_only_correct': 17, 'baseline_only_correct': 135, 'n_discordant': 152, 'p_value': np.float64(5.4791181269678426e-24)}
- n=4: {'n': 78, 'metric_accuracy': np.float64(0.5384615384615384), 'baseline_accuracy': np.float64(0.9743589743589743), 'chance': 0.25}
- n=5: {'n': 158, 'metric_accuracy': np.float64(0.43670886075949367), 'baseline_accuracy': np.float64(0.7974683544303798), 'chance': 0.2}
- n=6: {'n': 72, 'metric_accuracy': np.float64(0.25), 'baseline_accuracy': np.float64(0.625), 'chance': 0.16666666666666666}

## Experiment 5 -- count-clean subset

```
ALL ROWS (no count filter)
 stratum | labeled | scored | correct | accuracy |  chance
----------------------------------------------------------
     n=4 |      96 |     78 |      42 |  53.8% |  25.0%
     n=5 |     205 |    158 |      69 |  43.7% |  20.0%
     n=6 |     108 |     72 |      18 |  25.0% |  16.7%
----------------------------------------------------------
 overall |     409 |    308 |     129 |  41.9% |     -  

COUNT-CLEAN ONLY (count-broken images excluded)
 stratum | labeled | scored | correct | accuracy |  chance
----------------------------------------------------------
     n=4 |      96 |     78 |      42 |  53.8% |  25.0%
     n=5 |     205 |    158 |      69 |  43.7% |  20.0%
     n=6 |     108 |     66 |      18 |  27.3% |  16.7%
----------------------------------------------------------
 overall |     409 |    302 |     129 |  42.7% |     -  

(6 row(s) excluded as count-broken -- rendering failure, not binding failure.)
```
