#!/usr/bin/env python3
"""Select a fixed set of EC prompts for the pilot and freeze their item_index list."""

import argparse
import ast
import pathlib
import random

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select pilot prompts from generated_prompts.csv")
    parser.add_argument("--prompts_csv", type=str, default="../coig/create_dataset/generated_prompts.csv")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="pilot_item_indexes.txt")
    parser.add_argument(
        "--csv_output",
        type=str,
        default="pilot_prompts.csv",
        help="Filtered copy of prompts_csv containing only the selected rows -- "
        "point create_prompt_sbs.py --csv at this so it decomposes exactly "
        "these 10 prompts (its own --limit_items takes the first N rows, "
        "not a specific item_index selection).",
    )
    return parser.parse_args()


def has_ambiguous_attribute(attributes_str: str) -> bool:
    """Flag prompts whose attributes mention gray/grey, which the ARM uses as
    a placeholder color and the paper's own evaluation notes avoid."""
    try:
        attr_tuples = ast.literal_eval(attributes_str)
    except (ValueError, SyntaxError):
        return False
    text = " ".join(str(value) for pair in attr_tuples for value in pair).lower()
    return "gray" in text or "grey" in text


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.prompts_csv)
    df = df[df["prompt"].notna() & (df["prompt"] != "")]
    if "error" in df.columns:
        df = df[df["error"].isna()]
    df = df[~df["attributes"].apply(has_ambiguous_attribute)]

    rng = random.Random(args.seed)
    chosen = sorted(rng.sample(list(df["item_index"]), args.n))

    output_path = pathlib.Path(args.output)
    output_path.write_text("\n".join(str(i) for i in chosen) + "\n", encoding="utf-8")
    print(f"Selected {len(chosen)} item_index values -> {output_path}")
    print(chosen)

    csv_output_path = pathlib.Path(args.csv_output)
    df[df["item_index"].isin(chosen)].to_csv(csv_output_path, index=False)
    print(f"Wrote filtered prompts CSV -> {csv_output_path}")


if __name__ == "__main__":
    main()
