# pilot/

## Setup

```bash
export GOOGLE_AI_API_KEY=your_key_here
```

## Configuration

`config.py` + `config/default.yaml` (also `config/n100.yaml` for the N=100 variant). Pass
`--config PATH` to load a different YAML file; individual `--flag` values take priority.

## Running

```bash
python select_prompts.py --prompts_csv <path-to-generated_prompts.csv> --n 10 --seed 42

python build_conditions.py \
  --prompts_csv <path-to-generated_prompts.csv> \
  --sbs_eval_csv data/evaluation_sbs_results.csv \
  --item_indexes data/pilot_item_indexes.txt

python causal_relevance.py \
  --conditions_dir data/conditions \
  --sbs_eval_csv data/evaluation_sbs_results.csv \
  --image_base <path-to-multi_step_out>

python score_pilot.py --results_csv data/causal_relevance_results.csv
```

For the N=100 variant, pass `--config config/n100.yaml` to any of the above (points every
default path at the `_100` files instead).

## Tests

```bash
python -m pytest tests/ -q
```
