# ssa/anchor_set/

## Setup

Uses the repo root's install (`pip install -r requirements.txt && pip install -e .`).

## Configuration

`config.py` + `config/default.yaml` define the default artifacts dir, model IDs, seeds,
and thresholds. Pass `--config PATH` to load a different YAML file.

## Running

Each `exp*.py` script takes `--artifacts-dir` (`artifacts`, `artifacts_flux`,
`artifacts_flux_hard`, `artifacts_sdxl`, or `artifacts_dummy`) and `--annotator`:

```bash
python exp1_accuracy_by_n.py --artifacts-dir artifacts_dummy --annotator dummy
python exp6_prompt_baseline.py --artifacts-dir artifacts_flux --annotator annotator1
python exp7_misbound_subset.py --artifacts-dir artifacts_flux_hard --annotator consensus
python exp9_taxonomy_analysis.py \
  --easy-dir artifacts_flux --easy-annotator annotator1 \
  --hard-dir artifacts_flux_hard --hard-annotator consensus
python vqa_agreement_check.py --artifacts-dir artifacts_flux_hard --annotator consensus
python exp3b_within_item_permutation.py --artifacts-dir artifacts_flux_hard --annotator consensus
```

Run any script with `--help` for full options.

Other CPU-only utilities: `analyze_agreement.py`, `label_images.py`,
`build_consensus_labels.py`, `mis_binding_detection.py`, `recompute_boxes.py`,
`run_five_experiments.py`, `run_all_experiments.py`, `make_dummy_artifacts.py`.

GPU/Kaggle kernels (not runnable locally without a GPU): `generate_anchor_images.py`,
`generate_anchor_images_flux.py`, `generate_anchor_images_sdxl.py`,
`taxonomy_capture_flux.py`, `vqa_score_flux.py`, `vqa_score_sdxl.py`.

## Tests

```bash
pytest tests/ -q
```
