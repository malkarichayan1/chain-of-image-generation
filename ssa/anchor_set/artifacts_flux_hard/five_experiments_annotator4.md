# Part A Five-Experiment Battery

## Experiment 1 -- accuracy per subject count

```
 stratum | labeled | scored | correct | accuracy |  chance |  p-value
---------------------------------------------------------------------
     n=4 |      96 |     83 |      45 |  54.2% |  25.0% | 0.0000
     n=5 |     205 |    164 |      71 |  43.3% |  20.0% | 0.0000
     n=6 |     108 |     72 |      19 |  26.4% |  16.7% | 0.0246
---------------------------------------------------------------------
 overall |     409 |    319 |     135 |  42.3% |     -   |      -  
```

## Experiment 2 -- early-window vs. full-trajectory attention

- n=4: early=0.542 full=0.554 chance=0.250 (n_scored=83)
- n=5: early=0.433 full=0.445 chance=0.200 (n_scored=164)
- n=6: early=0.264 full=0.250 chance=0.167 (n_scored=72)
- McNemar (early vs. full): {'n': 319, 'early_only_correct': 1, 'full_only_correct': 3, 'n_discordant': 4, 'p_value': np.float64(0.625)}

## Experiment 3 -- attention-randomization falsification

- n=4: median_scrambled_accuracy=0.241 chance=0.250 falsification_clean_fraction=0.975 (n_seeds=200)
- n=5: median_scrambled_accuracy=0.207 chance=0.200 falsification_clean_fraction=0.950 (n_seeds=200)
- n=6: median_scrambled_accuracy=0.167 chance=0.167 falsification_clean_fraction=0.990 (n_seeds=200)
- McNemar (real vs. scrambled, seed=42): {'n': 319, 'real_only_correct': 98, 'scrambled_only_correct': 39, 'n_discordant': 137, 'p_value': np.float64(4.870586356414974e-07)}

## Experiment 4 -- nearest-subject-noun baseline

- Headline: {'n': 319, 'metric_accuracy': 0.4231974921630094, 'baseline_accuracy': 0.7711598746081505}
- McNemar: {'n': 319, 'metric_only_correct': 21, 'baseline_only_correct': 132, 'n_discordant': 153, 'p_value': np.float64(7.290949803013338e-21)}
- n=4: {'n': 83, 'metric_accuracy': np.float64(0.5421686746987951), 'baseline_accuracy': np.float64(0.9397590361445783), 'chance': 0.25}
- n=5: {'n': 164, 'metric_accuracy': np.float64(0.4329268292682927), 'baseline_accuracy': np.float64(0.774390243902439), 'chance': 0.2}
- n=6: {'n': 72, 'metric_accuracy': np.float64(0.2638888888888889), 'baseline_accuracy': np.float64(0.5694444444444444), 'chance': 0.16666666666666666}

## Experiment 5 -- count-clean subset

```
ALL ROWS (no count filter)
 stratum | labeled | scored | correct | accuracy |  chance
----------------------------------------------------------
     n=4 |      96 |     83 |      45 |  54.2% |  25.0%
     n=5 |     205 |    164 |      71 |  43.3% |  20.0%
     n=6 |     108 |     72 |      19 |  26.4% |  16.7%
----------------------------------------------------------
 overall |     409 |    319 |     135 |  42.3% |     -  

COUNT-CLEAN ONLY (count-broken images excluded)
 stratum | labeled | scored | correct | accuracy |  chance
----------------------------------------------------------
     n=4 |      96 |     83 |      45 |  54.2% |  25.0%
     n=5 |     205 |    164 |      71 |  43.3% |  20.0%
     n=6 |     108 |     66 |      19 |  28.8% |  16.7%
----------------------------------------------------------
 overall |     409 |    313 |     135 |  43.1% |     -  

(6 row(s) excluded as count-broken -- rendering failure, not binding failure.)
```
