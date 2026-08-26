# Part A Five-Experiment Battery

## Experiment 1 -- accuracy per subject count

```
 stratum | labeled | scored | correct | accuracy |  chance |  p-value
---------------------------------------------------------------------
     n=4 |      96 |     80 |      44 |  55.0% |  25.0% | 0.0000
     n=5 |     205 |    161 |      70 |  43.5% |  20.0% | 0.0000
     n=6 |     108 |     75 |      21 |  28.0% |  16.7% | 0.0095
---------------------------------------------------------------------
 overall |     409 |    316 |     135 |  42.7% |     -   |      -  
```

## Experiment 2 -- early-window vs. full-trajectory attention

- n=4: early=0.550 full=0.562 chance=0.250 (n_scored=80)
- n=5: early=0.435 full=0.447 chance=0.200 (n_scored=161)
- n=6: early=0.280 full=0.253 chance=0.167 (n_scored=75)
- McNemar (early vs. full): {'n': 316, 'early_only_correct': 2, 'full_only_correct': 3, 'n_discordant': 5, 'p_value': 1.0}

## Experiment 3 -- attention-randomization falsification

- n=4: median_scrambled_accuracy=0.250 chance=0.250 falsification_clean_fraction=0.960 (n_seeds=200)
- n=5: median_scrambled_accuracy=0.199 chance=0.200 falsification_clean_fraction=0.930 (n_seeds=200)
- n=6: median_scrambled_accuracy=0.173 chance=0.167 falsification_clean_fraction=0.950 (n_seeds=200)
- McNemar (real vs. scrambled, seed=42): {'n': 316, 'real_only_correct': 97, 'scrambled_only_correct': 36, 'n_discordant': 133, 'p_value': np.float64(1.1926074377747833e-07)}

## Experiment 4 -- nearest-subject-noun baseline

- Headline: {'n': 316, 'metric_accuracy': 0.4272151898734177, 'baseline_accuracy': 0.8006329113924051}
- McNemar: {'n': 316, 'metric_only_correct': 18, 'baseline_only_correct': 136, 'n_discordant': 154, 'p_value': np.float64(1.3279948964080984e-23)}
- n=4: {'n': 80, 'metric_accuracy': np.float64(0.55), 'baseline_accuracy': np.float64(0.9875), 'chance': 0.25}
- n=5: {'n': 161, 'metric_accuracy': np.float64(0.43478260869565216), 'baseline_accuracy': np.float64(0.7950310559006211), 'chance': 0.2}
- n=6: {'n': 75, 'metric_accuracy': np.float64(0.28), 'baseline_accuracy': np.float64(0.6133333333333333), 'chance': 0.16666666666666666}

## Experiment 5 -- count-clean subset

```
ALL ROWS (no count filter)
 stratum | labeled | scored | correct | accuracy |  chance
----------------------------------------------------------
     n=4 |      96 |     80 |      44 |  55.0% |  25.0%
     n=5 |     205 |    161 |      70 |  43.5% |  20.0%
     n=6 |     108 |     75 |      21 |  28.0% |  16.7%
----------------------------------------------------------
 overall |     409 |    316 |     135 |  42.7% |     -  

COUNT-CLEAN ONLY (count-broken images excluded)
 stratum | labeled | scored | correct | accuracy |  chance
----------------------------------------------------------
     n=4 |      96 |     80 |      44 |  55.0% |  25.0%
     n=5 |     205 |    161 |      70 |  43.5% |  20.0%
     n=6 |     108 |     69 |      21 |  30.4% |  16.7%
----------------------------------------------------------
 overall |     409 |    310 |     135 |  43.5% |     -  

(6 row(s) excluded as count-broken -- rendering failure, not binding failure.)
```
