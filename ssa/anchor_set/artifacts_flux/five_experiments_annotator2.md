# Part A Five-Experiment Battery

## Experiment 1 -- accuracy per subject count

```
 stratum | labeled | scored | correct | accuracy |  chance |  p-value
---------------------------------------------------------------------
     n=2 |      86 |     78 |      76 |  97.4% |  50.0% | 0.0000
     n=3 |      75 |     68 |      65 |  95.6% |  33.3% | 0.0000
     n=4 |     136 |    119 |      87 |  73.1% |  25.0% | 0.0000
---------------------------------------------------------------------
 overall |     297 |    265 |     228 |  86.0% |     -   |      -  
```

## Experiment 2 -- early-window vs. full-trajectory attention

- n=2: early=0.974 full=0.974 chance=0.500 (n_scored=78)
- n=3: early=0.956 full=0.956 chance=0.333 (n_scored=68)
- n=4: early=0.731 full=0.723 chance=0.250 (n_scored=119)
- McNemar (early vs. full): {'n': 265, 'early_only_correct': 1, 'full_only_correct': 0, 'n_discordant': 1, 'p_value': 1.0}

## Experiment 3 -- attention-randomization falsification

- n=2: median_scrambled_accuracy=0.500 chance=0.500 falsification_clean_fraction=0.965 (n_seeds=200)
- n=3: median_scrambled_accuracy=0.324 chance=0.333 falsification_clean_fraction=0.945 (n_seeds=200)
- n=4: median_scrambled_accuracy=0.252 chance=0.250 falsification_clean_fraction=0.955 (n_seeds=200)
- McNemar (real vs. scrambled, seed=42): {'n': 265, 'real_only_correct': 145, 'scrambled_only_correct': 13, 'n_discordant': 158, 'p_value': np.float64(2.2203836627081591e-29)}

## Experiment 4 -- nearest-subject-noun baseline

- Headline: {'n': 265, 'metric_accuracy': 0.8603773584905661, 'baseline_accuracy': 0.7886792452830189}
- McNemar: {'n': 265, 'metric_only_correct': 43, 'baseline_only_correct': 24, 'n_discordant': 67, 'p_value': np.float64(0.027119992403126333)}
- n=2: {'n': 78, 'metric_accuracy': np.float64(0.9743589743589743), 'baseline_accuracy': np.float64(0.8846153846153846), 'chance': 0.5}
- n=3: {'n': 68, 'metric_accuracy': np.float64(0.9558823529411765), 'baseline_accuracy': np.float64(0.8235294117647058), 'chance': 0.3333333333333333}
- n=4: {'n': 119, 'metric_accuracy': np.float64(0.7310924369747899), 'baseline_accuracy': np.float64(0.7058823529411765), 'chance': 0.25}

## Experiment 5 -- count-clean subset

```
ALL ROWS (no count filter)
 stratum | labeled | scored | correct | accuracy |  chance
----------------------------------------------------------
     n=2 |      86 |     78 |      76 |  97.4% |  50.0%
     n=3 |      75 |     68 |      65 |  95.6% |  33.3%
     n=4 |     136 |    119 |      87 |  73.1% |  25.0%
----------------------------------------------------------
 overall |     297 |    265 |     228 |  86.0% |     -  

COUNT-CLEAN ONLY (count-broken images excluded)
 stratum | labeled | scored | correct | accuracy |  chance
----------------------------------------------------------
     n=2 |      86 |     78 |      76 |  97.4% |  50.0%
     n=3 |      75 |     68 |      65 |  95.6% |  33.3%
     n=4 |     136 |    119 |      87 |  73.1% |  25.0%
----------------------------------------------------------
 overall |     297 |    265 |     228 |  86.0% |     -  

(0 row(s) excluded as count-broken -- rendering failure, not binding failure.)
```
