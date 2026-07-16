# Pilot: 10 Real vs. 10 Shuffled vs. 10 Substituted

Implements the pilot described in [`../docs/pilot-design.md`](../docs/pilot-design.md).

Status: **code written, not yet run** -- waiting on `GOOGLE_AI_API_KEY`.

## Setup

```
cd coig
pip install -r requirements.txt
export GOOGLE_AI_API_KEY=your_key_here   # or put it in coig/.env
```

## Run order

Commands assume the working directory shown at the start of each step.

1. **Select 10 prompts** from the EC benchmark (run from `pilot/`):
   ```
   python select_prompts.py --n 10 --output pilot_item_indexes.txt
   ```

2. **Decompose into sub-prompts (CSP)** -- the vendored script, unmodified
   (run from `coig/create_prompt/`). Filter `generated_prompts.csv` down to
   the pilot's 10 `item_index` values first (or point `--csv` at a filtered
   copy) so only the pilot's prompts get decomposed:
   ```
   python create_prompt_sbs.py \
     --csv ../create_dataset/generated_prompts.csv \
     --output ./sbs_prompts_results.csv
   ```

3. **Generate the base image chains (ARM, lock ON)** -- the only generation
   step in the whole pilot (run from `coig/create_images/`):
   ```
   python generate_multi_step_image_genai_simple.py \
     --csv ../create_prompt/sbs_prompts_results.csv \
     --outdir multi_step_out
   ```

4. **Judge every step against every ground-truth question** -- also the
   vendored script, unmodified. This one run supplies everything the Real
   and Shuffled conditions need, at zero extra cost beyond this single pass
   (run from `coig/evaluate/`):
   ```
   python evaluate_sbs_images.py \
     --sbs_csv ../create_prompt/sbs_prompts_results.csv \
     --prompts_csv ../create_dataset/generated_prompts.csv \
     --image_base ../create_images/multi_step_out \
     --output_csv ../../pilot/evaluation_sbs_results.csv \
     --all
   ```

5. **Build the three condition manifests** (run from `pilot/`):
   ```
   python build_conditions.py \
     --prompts_csv ../coig/create_dataset/generated_prompts.csv \
     --sbs_eval_csv evaluation_sbs_results.csv \
     --item_indexes pilot_item_indexes.txt
   ```

6. **Score Causal Relevance across all three conditions** (run from
   `pilot/`) -- the only step that makes new Gemini calls (Substituted
   only; Real and Shuffled are pulled from step 4's results):
   ```
   python causal_relevance.py \
     --conditions_dir conditions \
     --sbs_eval_csv evaluation_sbs_results.csv \
     --image_base ../coig/create_images/multi_step_out
   ```

7. **Summarize and read the go/no-go signal** (run from `pilot/`):
   ```
   python score_pilot.py --results_csv causal_relevance_results.csv
   ```

## Why Real and Shuffled need no extra API calls

`evaluate_sbs_images.py --all` already asks every ground-truth attribute
question (`question_attr_1..4`) against every step image, for every chain.
Real and Shuffled differ only in *which* step's already-collected answer is
treated as the "claimed" step for a given attribute -- Real uses the
attribute's true introduction step (the earliest step the judge already said
"yes" to), Shuffled uses a different step from the same chain. Only
Substituted needs new questions, since it pairs a chain's images with an
attribute that belongs to a different chain entirely and was never asked
against these images.

## What `evaluate_images.py` is (and isn't)

`coig/evaluate/evaluate_images.py` grades a single `baseline.png` per item
against the full ground-truth entity/attribute/interaction list -- it's the
tool behind the separate baseline-vs-CoIG entity-collapse comparison
(paper Section 6), not a per-step evaluator. It is not used by this pilot.
`evaluate_sbs_images.py` is the one that operates on individual step images,
which is what a per-step appears/persists check requires.

## Design rationale

See [`../docs/pilot-design.md`](../docs/pilot-design.md) for the full
reasoning: why EC over T2I-CompBench Color, why no image regeneration, what
the compositional lock's own documented behavior (paper Appendix 9, rule 3)
predicts each condition should show, and the go/no-go criteria.
