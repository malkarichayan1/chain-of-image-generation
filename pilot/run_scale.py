#!/usr/bin/env python3
"""Runner script to scale up the Causal Relevance Audit to N=100+ chains.

Executes the pipeline steps in sequence:
1. Select N prompts from EC benchmark
2. Decompose prompts into sub-prompts (SBS)
3. Generate base multi-step image chains
4. Evaluate all step images against attribute questions
5. Build Real, Shuffled, and Substituted manifests
6. Run Gemini evaluation for Substituted condition
7. Compute statistics and paired tests
"""

import argparse
import os
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Scaled Causal Relevance Audit")
    parser.add_argument("--n", type=int, default=100, help="Number of chains (prompts) to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for selection and manifests")
    parser.add_argument("--skip_generation", action="store_true", help="Skip image generation if images already exist")
    return parser.parse_args()


def run_cmd(cmd: str, cwd: str = None) -> None:
    print(f"\n=======================================================")
    print(f"Running: {cmd} (cwd: {cwd or '.'})")
    print(f"=======================================================\n")
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    res = subprocess.run(cmd, shell=True, cwd=cwd, env=env)
    if res.returncode != 0:
        print(f"Error: Command failed with exit code {res.returncode}")
        sys.exit(res.returncode)


def main() -> None:
    args = parse_args()
    n = args.n
    seed = args.seed
    
    pilot_dir = os.path.abspath(os.path.dirname(__file__))
    coig_dir = os.path.abspath(os.path.join(pilot_dir, "..", "coig"))
    
    item_indexes_file = os.path.join(pilot_dir, f"pilot_item_indexes_{n}.txt")
    prompts_csv = os.path.join(pilot_dir, f"pilot_prompts_{n}.csv")
    sbs_prompts_csv = os.path.join(coig_dir, "create_prompt", f"sbs_prompts_results_{n}.csv")
    img_outdir = os.path.join(coig_dir, "create_images", f"multi_step_out_{n}")
    eval_sbs_csv = os.path.join(pilot_dir, f"evaluation_sbs_results_{n}.csv")
    conditions_dir = os.path.join(pilot_dir, f"conditions_{n}")
    results_csv = os.path.join(pilot_dir, f"causal_relevance_results_{n}.csv")

    # Step 1: Select prompts
    run_cmd(
        f'python select_prompts.py --n {n} --seed {seed} --output "{item_indexes_file}" --csv_output "{prompts_csv}"',
        cwd=pilot_dir,
    )

    # Step 2: Decompose into sub-prompts (SBS)
    run_cmd(
        f'python create_prompt_sbs.py --csv "{prompts_csv}" --output "{sbs_prompts_csv}"',
        cwd=os.path.join(coig_dir, "create_prompt"),
    )

    # Step 3: Generate multi-step images
    if not args.skip_generation:
        run_cmd(
            f'python generate_multi_step_image_genai_simple.py --csv "{sbs_prompts_csv}" --outdir "{img_outdir}"',
            cwd=os.path.join(coig_dir, "create_images"),
        )
    else:
        print(f"Skipping step 3 (image generation) as requested.")

    # Step 4: Evaluate step images
    run_cmd(
        f'python evaluate_sbs_images.py --sbs_csv "{sbs_prompts_csv}" --prompts_csv ../create_dataset/generated_prompts.csv --image_base "{img_outdir}" --output_csv "{eval_sbs_csv}" --all',
        cwd=os.path.join(coig_dir, "evaluate"),
    )

    # Step 5: Build condition manifests
    run_cmd(
        f'python build_conditions.py --prompts_csv ../coig/create_dataset/generated_prompts.csv --sbs_eval_csv "{eval_sbs_csv}" --item_indexes "{item_indexes_file}" --seed {seed} --output_dir "{conditions_dir}"',
        cwd=pilot_dir,
    )

    # Step 6: Score Causal Relevance (Substituted condition calls)
    run_cmd(
        f'python causal_relevance.py --conditions_dir "{conditions_dir}" --sbs_eval_csv "{eval_sbs_csv}" --image_base "{img_outdir}" --output "{results_csv}"',
        cwd=pilot_dir,
    )

    # Step 7: Score and statistical tests
    run_cmd(
        f'python score_pilot.py --results_csv "{results_csv}"',
        cwd=pilot_dir,
    )


if __name__ == "__main__":
    main()
