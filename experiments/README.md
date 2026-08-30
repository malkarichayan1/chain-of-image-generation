# experiments/

## Configuration

`config.py` + `config/default.yaml` define paths, model IDs, and thresholds. Pass
`--config PATH` to load a different YAML file; individual `--flag` values take priority.

## Running

```bash
python generate_chains.py                      # Stage 1: generate chains (GPU/Kaggle)
python segment_cache.py --manifest artifacts/manifest.json --cache-dir artifacts/segmentation_cache
python score_chains.py --manifest artifacts/manifest.json --cache-dir artifacts/segmentation_cache --output artifacts/chain_experiment_results.csv
python calibrate_threshold.py --manifest artifacts/manifest.json --cache-dir artifacts/segmentation_cache
python analyze_results.py --results-csv artifacts/chain_experiment_results.csv
```

Other scripts (`equivalence_check.py`, `rng_sweep.py`, `lock_confound_analysis.py`,
`judge_delta_check.py`, `selection_effect_check.py`, `discriminant_validity_check.py`,
`zero_inflation_recheck.py`, `owlvit_cross_check.py`) each take their own CLI flags --
run any of them with `--help` for options.

`run_chain_experiment.py` is a standalone, single-file script (GPU/Kaggle).

## Tests

```bash
python -m pytest tests/ -q
```
