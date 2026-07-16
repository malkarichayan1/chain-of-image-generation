#!/usr/bin/env python3
"""Reconstruct Causal Relevance's appears-at-step / persists-to-final check
across the Real, Shuffled, and Substituted conditions.

Real and Shuffled reuse per-step judgments already collected by running
`evaluate_sbs_images.py --all` (see build_conditions.py) -- no new API calls.
Substituted asks new yes/no questions, since it pairs a chain's images with an
attribute claim that belongs to a different chain and was never evaluated
against these particular images.
"""

import argparse
import json
import pathlib
from typing import Dict, Optional, Tuple

import pandas as pd

from judge import ask_yes_no, get_client

CONDITIONS = ["real", "shuffled", "substituted"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score Causal Relevance across pilot conditions")
    parser.add_argument("--conditions_dir", type=str, default="conditions")
    parser.add_argument("--sbs_eval_csv", type=str, default="evaluation_sbs_results.csv")
    parser.add_argument("--image_base", type=str, default="../coig/create_images/multi_step_out")
    parser.add_argument("--model", type=str, default="gemini-2.5-flash")
    parser.add_argument("--output", type=str, default="causal_relevance_results.csv")
    return parser.parse_args()


def load_lookup(sbs_eval_df: pd.DataFrame) -> Dict[Tuple[int, str, int], int]:
    lookup: Dict[Tuple[int, str, int], int] = {}
    for _, row in sbs_eval_df.iterrows():
        key = (int(row["item_index"]), str(row["question_type"]), int(row["step_num"]))
        lookup[key] = int(row["answer_binary"])
    return lookup


def image_path(image_base: str, item_index: int, step_num: int) -> str:
    return str(pathlib.Path(image_base) / str(item_index) / f"step_{step_num:02d}.png")


def score_own_condition(entry: dict, lookup: Dict[Tuple[int, str, int], int]) -> Tuple[Optional[int], Optional[int]]:
    """Real and Shuffled: pull already-collected answers, no new API calls."""
    item_index = entry["item_index"]
    qcol = entry["question_col"]
    appears = lookup.get((item_index, qcol, entry["claimed_step"]))
    persists = lookup.get((item_index, qcol, entry["final_step"]))
    return appears, persists


def score_substituted(entry: dict, image_base: str, model: str, client) -> Tuple[Optional[int], Optional[int]]:
    """Substituted: the (item_index, question) pair was never asked during the
    evaluate_sbs_images.py run, so it needs fresh judge calls."""
    item_index = entry["item_index"]
    question = entry["question_text"]

    claimed_img = image_path(image_base, item_index, entry["claimed_step"])
    final_img = image_path(image_base, item_index, entry["final_step"])

    appears_answer = ask_yes_no(client, claimed_img, question, model=model)
    persists_answer = ask_yes_no(client, final_img, question, model=model)

    to_binary = {"yes": 1, "no": 0}
    return to_binary.get(appears_answer), to_binary.get(persists_answer)


def main() -> None:
    args = parse_args()
    sbs_eval_df = pd.read_csv(args.sbs_eval_csv)
    lookup = load_lookup(sbs_eval_df)

    client = get_client()
    results = []

    for condition in CONDITIONS:
        manifest_path = pathlib.Path(args.conditions_dir) / f"{condition}.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        for entry in manifest:
            if entry.get("final_step") is None:
                continue

            if condition == "substituted":
                appears, persists = score_substituted(entry, args.image_base, args.model, client)
            else:
                appears, persists = score_own_condition(entry, lookup)

            results.append(
                {
                    "condition": condition,
                    "item_index": entry["item_index"],
                    "question_col": entry["question_col"],
                    "claimed_step": entry["claimed_step"],
                    "final_step": entry["final_step"],
                    "source": entry["source"],
                    "appears_at_step": appears,
                    "persists_to_final": persists,
                }
            )

    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output, index=False)
    print(f"Saved {len(results_df)} rows -> {args.output}")


if __name__ == "__main__":
    main()
