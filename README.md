# Text-to-Image Faithfulness Metrics Track Intent, Not Realization

## Repository Structure

```
chain-of-image-generation/
├── pyproject.toml
├── common/                 # Shared code (config loader, SSA metric)
├── ssa/anchor_set/         # Track A -- see ssa/anchor_set/README.md
├── experiments/            # Track B -- see experiments/README.md
└── pilot/                  # Track 0 -- see pilot/README.md
```

## Setup

Python 3.10+ required.

```bash
pip install -r requirements.txt
pip install -e .
```

## Configuration

Each track (`pilot/`, `experiments/`, `ssa/anchor_set/`) has its own `config.py` +
`config/default.yaml`. Scripts that accept `--config PATH` load that file first;
individual `--flag` values passed on the command line take priority over it.

## Running the Test Suite

```bash
cd ssa/anchor_set && pytest tests/ -q
cd ../../experiments && python -m pytest tests/ -q
cd ../pilot && python -m pytest tests/ -q
```

GPU-specific tests are automatically skipped without CUDA.

## Running Experiments

From `ssa/anchor_set/`:

```bash
python exp6_prompt_baseline.py --artifacts-dir artifacts_flux --annotator annotator1
python exp6_prompt_baseline.py --artifacts-dir artifacts_flux_hard --annotator consensus

python exp7_misbound_subset.py --artifacts-dir artifacts_flux_hard --annotator consensus

python exp9_taxonomy_analysis.py \
  --easy-dir artifacts_flux --easy-annotator annotator1 \
  --hard-dir artifacts_flux_hard --hard-annotator consensus

python vqa_agreement_check.py --artifacts-dir artifacts_flux --annotator annotator1
python vqa_agreement_check.py --artifacts-dir artifacts_flux_hard --annotator consensus

python exp3b_within_item_permutation.py --artifacts-dir artifacts_flux_hard --annotator consensus
```

See `pilot/README.md`, `experiments/README.md`, and `ssa/anchor_set/README.md` for each
track's full command set.
