#!/usr/bin/env python
"""
Compares metric A's attention-based `predicted_owner` against a VQAScore-style baseline
(P(yes)/(P(yes)+P(no)) from a VQA model, per Lin et al. 2024's own normalization) on the
SAME human-labeled anchor-set rows -- the comparison CLAUDE.md flags as never done:
"attention agrees with humans X%, VQAScore Y%, on identical images."

vqa_score_sdxl.py (Kaggle GPU kernel, not this file) crops each detected image to every
subject's box (recompute_boxes.py's boxes.json) and asks BLIP-VQA-base
"Is the person {phrase}?" per (subject crop, attribute), writing vqa_scores.json:
{prompt_id (str): {attribute: {subject: p_yes}}}. This module is pure CPU logic that
reads that file -- no model calls -- and mirrors anchor_common.build_agreement_rows'
row shape so results can be joined/compared directly, plus a paired McNemar test for the
head-to-head question.

ATTRIBUTE_PHRASES is a verbatim copy of the (subject, attribute) phrasing already
authored in prompt_specs.json (the "wearing X" / "holding X" wording each prompt uses) --
duplicated into vqa_score_sdxl.py's self-contained Kaggle kernel too, same "keep in sync"
convention as ANCHOR_PROMPTS across the two generation scripts.

Local CPU only (pandas/scipy). Run from inside ssa/anchor_set/:
    py -3 vqa_agreement_check.py --artifacts-dir artifacts_sdxl --annotator chayan
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
from scipy import stats

from anchor_common import NON_SUBJECT_LABELS, label_key, load_labels, summarize_agreement

ATTRIBUTE_PHRASES: Dict[str, str] = {
    "red apron": "wearing a red apron",
    "white hat": "wearing a white hat",
    "shovel": "holding a shovel",
    "blue gloves": "wearing blue gloves",
    "dark sunglasses": "wearing dark sunglasses",
    "book": "holding a book",
    "yellow helmet": "wearing a yellow helmet",
    "pan": "holding a pan",
}


def attribute_question(attribute: str) -> str:
    return f"Is the person {ATTRIBUTE_PHRASES[attribute]}?"


def vqa_predicted_owner(p_yes_by_subject: Dict[str, float]) -> str:
    if not p_yes_by_subject:
        raise ValueError("p_yes_by_subject is empty; nothing to attribute to")
    return max(p_yes_by_subject, key=lambda s: p_yes_by_subject[s])


def vqa_agreement_rows(manifest: dict, labels: Dict[str, str],
                        vqa_scores: Dict[str, dict]) -> List[dict]:
    """Same row shape as anchor_common.build_agreement_rows (prompt_id, n, attribute,
    intended_subject, predicted_owner, human_label, scored, correct), but
    `predicted_owner` is VQAScore's argmax instead of attention's. Skips images with no
    entry in vqa_scores (e.g. a detection mismatch during scoring) rather than crashing,
    so a partial VQA run can still be analyzed."""
    rows: List[dict] = []
    for img in manifest["images"]:
        if not img.get("detected"):
            continue
        per_image = vqa_scores.get(str(img["prompt_id"]))
        if not per_image:
            continue
        for attr in img["attributes"]:
            key = label_key(img["prompt_id"], attr["attribute"])
            if key not in labels:
                continue
            per_subject = per_image.get(attr["attribute"])
            if not per_subject:
                continue
            human = labels[key]
            scored = human not in NON_SUBJECT_LABELS
            owner = vqa_predicted_owner(per_subject)
            rows.append(dict(
                prompt_id=img["prompt_id"], n=img["n"], attribute=attr["attribute"],
                intended_subject=attr["intended_subject"], predicted_owner=owner,
                human_label=human, scored=scored, correct=(scored and owner == human),
            ))
    return rows


def paired_mcnemar(df: pd.DataFrame, correct_a: str, correct_b: str) -> dict:
    """Exact McNemar test (binomial on discordant pairs) for whether two predictors
    disagree in a systematically one-sided way on the same scored rows."""
    sub = df[df["scored"] == True]  # noqa: E712
    a = sub[correct_a].astype(bool)
    b = sub[correct_b].astype(bool)
    a_only = int((a & ~b).sum())
    b_only = int((~a & b).sum())
    n_discordant = a_only + b_only
    p = (stats.binomtest(a_only, n_discordant, 0.5, alternative="two-sided").pvalue
         if n_discordant else None)
    return dict(n=len(sub), a_only_correct=a_only, b_only_correct=b_only,
                n_discordant=n_discordant, p_value=p)


if __name__ == "__main__":
    from anchor_common import build_agreement_rows

    ap = argparse.ArgumentParser(
        description="Compare metric A's attention predictions against VQAScore")
    ap.add_argument("--artifacts-dir", default="artifacts_sdxl")
    ap.add_argument("--annotator", default="chayan")
    args = ap.parse_args()
    artifacts_dir = Path(args.artifacts_dir)

    manifest = json.loads((artifacts_dir / "manifest.json").read_text())
    labels = load_labels(artifacts_dir / f"labels_{args.annotator}.json")
    vqa_scores = json.loads((artifacts_dir / "vqa_scores.json").read_text())

    attn_rows = build_agreement_rows(manifest, labels)
    vqa_rows = vqa_agreement_rows(manifest, labels, vqa_scores)

    print(f"attention rows: {len(attn_rows)}  |  VQAScore rows: {len(vqa_rows)}")
    print("\n=== VQAScore accuracy vs. chance (mirrors the attention metric's own table) ===")
    vqa_summary = summarize_agreement(vqa_rows)
    for n, s in vqa_summary["by_stratum"].items():
        print(f"  n={n}: {s['n_correct']}/{s['n_scored']} = "
              f"{s['accuracy']:.1%} (chance {s['chance']:.1%})" if s["accuracy"] is not None
              else f"  n={n}: no scored rows")
    o = vqa_summary["overall"]
    print(f"  overall: {o['n_correct']}/{o['n_scored']} = {o['accuracy']:.1%}")

    attn_df = pd.DataFrame(attn_rows).rename(
        columns={"predicted_owner": "attn_predicted_owner", "correct": "attn_correct"})
    vqa_df = pd.DataFrame(vqa_rows)[["prompt_id", "attribute", "predicted_owner", "correct"]].rename(
        columns={"predicted_owner": "vqa_predicted_owner", "correct": "vqa_correct"})
    joined = attn_df.merge(vqa_df, on=["prompt_id", "attribute"], how="inner")

    print(f"\n=== Head-to-head on {len(joined)} identical rows ===")
    print(f"  attention accuracy: {joined[joined.scored]['attn_correct'].mean():.1%}")
    print(f"  VQAScore accuracy:  {joined[joined.scored]['vqa_correct'].mean():.1%}")
    mcnemar = paired_mcnemar(joined, "attn_correct", "vqa_correct")
    print(f"  McNemar: attention-only correct={mcnemar['a_only_correct']}, "
          f"VQAScore-only correct={mcnemar['b_only_correct']}, p={mcnemar['p_value']}")
