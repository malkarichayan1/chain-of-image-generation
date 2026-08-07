# Part A Five-Experiment Battery

## Experiment 1 -- accuracy per subject count

```
 stratum | labeled | scored | correct | accuracy |  chance |  p-value
---------------------------------------------------------------------
     n=4 |      96 |     79 |      44 |  55.7% |  25.0% | 0.0000
     n=5 |     202 |    160 |      70 |  43.8% |  20.0% | 0.0000
     n=6 |     107 |     72 |      19 |  26.4% |  16.7% | 0.0246
---------------------------------------------------------------------
 overall |     405 |    311 |     133 |  42.8% |     -   |      -  
```

## Experiment 2 -- early-window vs. full-trajectory attention

- n=4: early=0.557 full=0.570 chance=0.250 (n_scored=79)
- n=5: early=0.438 full=0.450 chance=0.200 (n_scored=160)
- n=6: early=0.264 full=0.250 chance=0.167 (n_scored=72)
- McNemar (early vs. full): {'n': 311, 'early_only_correct': 1, 'full_only_correct': 3, 'n_discordant': 4, 'p_value': np.float64(0.625)}

## Experiment 3 -- attention-randomization falsification

- n=4: median_scrambled_accuracy=0.253 chance=0.250 falsification_clean_fraction=0.965 (n_seeds=200)
- n=5: median_scrambled_accuracy=0.200 chance=0.200 falsification_clean_fraction=0.955 (n_seeds=200)
- n=6: median_scrambled_accuracy=0.167 chance=0.167 falsification_clean_fraction=0.940 (n_seeds=200)
- McNemar (real vs. scrambled, seed=42): {'n': 311, 'real_only_correct': 108, 'scrambled_only_correct': 39, 'n_discordant': 147, 'p_value': np.float64(1.1047851223224009e-08)}

## Experiment 4 -- nearest-subject-noun baseline

- Headline: {'n': 311, 'metric_accuracy': 0.42765273311897106, 'baseline_accuracy': 0.8006430868167203}
- McNemar: {'n': 311, 'metric_only_correct': 18, 'baseline_only_correct': 134, 'n_discordant': 152, 'p_value': np.float64(4.14829527597977e-23)}
- n=4: {'n': 79, 'metric_accuracy': np.float64(0.5569620253164557), 'baseline_accuracy': np.float64(0.9873417721518988), 'chance': 0.25}
- n=5: {'n': 160, 'metric_accuracy': np.float64(0.4375), 'baseline_accuracy': np.float64(0.8), 'chance': 0.2}
- n=6: {'n': 72, 'metric_accuracy': np.float64(0.2638888888888889), 'baseline_accuracy': np.float64(0.5972222222222222), 'chance': 0.16666666666666666}

## Experiment 5 -- count-clean subset

```
ALL ROWS (no count filter)
 stratum | labeled | scored | correct | accuracy |  chance
----------------------------------------------------------
     n=4 |      96 |     79 |      44 |  55.7% |  25.0%
     n=5 |     202 |    160 |      70 |  43.8% |  20.0%
     n=6 |     107 |     72 |      19 |  26.4% |  16.7%
----------------------------------------------------------
 overall |     405 |    311 |     133 |  42.8% |     -  

COUNT-CLEAN ONLY (count-broken images excluded)
 stratum | labeled | scored | correct | accuracy |  chance
----------------------------------------------------------
     n=4 |      96 |     79 |      44 |  55.7% |  25.0%
     n=5 |     202 |    160 |      70 |  43.8% |  20.0%
     n=6 |     107 |     66 |      19 |  28.8% |  16.7%
----------------------------------------------------------
 overall |     405 |    305 |     133 |  43.6% |     -  

(6 row(s) excluded as count-broken -- rendering failure, not binding failure.)
```
