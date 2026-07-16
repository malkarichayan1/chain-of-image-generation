#!/usr/bin/env python3
"""Build Real / Shuffled / Substituted condition manifests for the pilot.

Reuses per-step judgments already computed by running
coig/evaluate/evaluate_sbs_images.py --all over the pilot's fixed images (one
run, covering every step x every ground-truth question) instead of asking new
questions for Real and Shuffled -- both conditions only relabel which step's
already-collected judgment corresponds to which claimed attribute. Substituted
needs new judge calls (see causal_relevance.py), since it asks about an
attribute that was never part of this chain's ground truth or question set.
"""

import argparse
import json
import pathlib
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

ATTRIBUTE_QUESTION_COLS = ["question_attr_1", "question_attr_2", "question_attr_3", "question_attr_4"]


@dataclass(frozen=True)
class AttributeClaim:
    item_index: int
    question_col: str
    question_text: str
    true_step: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Real/Shuffled/Substituted manifests")
    parser.add_argument("--prompts_csv", type=str, default="../coig/create_dataset/generated_prompts.csv")
    parser.add_argument("--sbs_eval_csv", type=str, default="evaluation_sbs_results.csv")
    parser.add_argument("--item_indexes", type=str, default="pilot_item_indexes.txt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="conditions")
    return parser.parse_args()


def load_item_indexes(path: str) -> List[int]:
    lines = pathlib.Path(path).read_text(encoding="utf-8").splitlines()
    return [int(line.strip()) for line in lines if line.strip()]


def resolve_true_steps(
    prompts_df: pd.DataFrame,
    sbs_eval_df: pd.DataFrame,
    item_indexes: List[int],
) -> Dict[int, List[AttributeClaim]]:
    """For each item and each attribute question, find the earliest step where
    the yes/no judge already said yes (from the evaluate_sbs_images.py run)."""
    claims: Dict[int, List[AttributeClaim]] = {}

    for item_index in item_indexes:
        prow_matches = prompts_df[prompts_df["item_index"] == item_index]
        if prow_matches.empty:
            continue
        prow = prow_matches.iloc[0]

        item_claims: List[AttributeClaim] = []
        for qcol in ATTRIBUTE_QUESTION_COLS:
            question_text = prow.get(qcol)
            if not isinstance(question_text, str) or not question_text.strip():
                continue

            yes_rows = sbs_eval_df[
                (sbs_eval_df["item_index"] == item_index)
                & (sbs_eval_df["question_type"] == qcol)
                & (sbs_eval_df["answer_binary"] == 1)
            ]
            if yes_rows.empty:
                continue

            true_step = int(yes_rows["step_num"].min())
            item_claims.append(AttributeClaim(item_index, qcol, question_text, true_step))

        claims[item_index] = item_claims

    return claims


def final_step_for_item(sbs_eval_df: pd.DataFrame, item_index: int) -> Optional[int]:
    rows = sbs_eval_df[sbs_eval_df["item_index"] == item_index]
    if rows.empty:
        return None
    return int(rows["step_num"].max())


def build_real(claims: Dict[int, List[AttributeClaim]]) -> List[dict]:
    manifest = []
    for item_index, item_claims in claims.items():
        for claim in item_claims:
            manifest.append(
                {
                    "item_index": item_index,
                    "question_col": claim.question_col,
                    "question_text": claim.question_text,
                    "claimed_step": claim.true_step,
                    "source": "own_step",
                }
            )
    return manifest


def build_shuffled(claims: Dict[int, List[AttributeClaim]], rng: random.Random) -> List[dict]:
    """For each attribute, claim it was introduced at a different step of the
    SAME chain -- reuses that other step's already-collected judgment."""
    manifest = []
    for item_index, item_claims in claims.items():
        if len(item_claims) < 2:
            continue
        steps = [c.true_step for c in item_claims]
        for claim in item_claims:
            other_steps = [s for s in steps if s != claim.true_step]
            if not other_steps:
                continue
            claimed_step = rng.choice(other_steps)
            manifest.append(
                {
                    "item_index": item_index,
                    "question_col": claim.question_col,
                    "question_text": claim.question_text,
                    "claimed_step": claimed_step,
                    "source": "shuffled_within_chain",
                }
            )
    return manifest


def build_substituted(claims: Dict[int, List[AttributeClaim]], rng: random.Random) -> List[dict]:
    """For each chain, ask about an attribute that belongs to a DIFFERENT
    chain entirely. This is the true negative control and the only condition
    that requires new judge calls."""
    manifest = []
    item_indexes = list(claims.keys())
    for item_index, item_claims in claims.items():
        if not item_claims:
            continue
        own_claim = rng.choice(item_claims)

        donor_pool = [i for i in item_indexes if i != item_index and claims.get(i)]
        if not donor_pool:
            continue
        donor_index = rng.choice(donor_pool)
        donor_claim = rng.choice(claims[donor_index])

        manifest.append(
            {
                "item_index": item_index,
                "question_col": donor_claim.question_col,
                "question_text": donor_claim.question_text,
                "claimed_step": own_claim.true_step,
                "source": f"substituted_from_item_{donor_index}",
            }
        )
    return manifest


def main() -> None:
    args = parse_args()
    prompts_df = pd.read_csv(args.prompts_csv)
    sbs_eval_df = pd.read_csv(args.sbs_eval_csv)
    item_indexes = load_item_indexes(args.item_indexes)

    claims = resolve_true_steps(prompts_df, sbs_eval_df, item_indexes)
    unresolved = [i for i, c in claims.items() if not c]
    if unresolved:
        print(f"Warning: no resolvable attribute claims for item_index {unresolved} -- excluded from all conditions")

    rng = random.Random(args.seed)
    real = build_real(claims)
    shuffled = build_shuffled(claims, rng)
    substituted = build_substituted(claims, rng)

    final_steps = {i: final_step_for_item(sbs_eval_df, i) for i in item_indexes}

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, manifest in [("real", real), ("shuffled", shuffled), ("substituted", substituted)]:
        for entry in manifest:
            entry["final_step"] = final_steps.get(entry["item_index"])
        out_path = output_dir / f"{name}.json"
        out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"{name}: {len(manifest)} claims -> {out_path}")


if __name__ == "__main__":
    main()
