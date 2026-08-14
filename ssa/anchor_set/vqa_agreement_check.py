#!/usr/bin/env python
"""
Compares metric A's attention-based `predicted_owner` against a VQAScore-style baseline
(P(yes)/(P(yes)+P(no)) from a VQA model, per Lin et al. 2024's own normalization) on the
SAME human-labeled anchor-set rows -- the comparison CLAUDE.md flags as never done:
"attention agrees with humans X%, VQAScore Y%, on identical images."

vqa_score_sdxl.py / vqa_score_flux.py (Kaggle GPU kernels, not this file) crop each
detected image to every subject's box (recompute_boxes.py's boxes.json) and ask
BLIP-VQA-base `anchor_common.attribute_question`'s question per (subject crop, attribute),
writing vqa_scores.json: {prompt_id (str): {attribute: {subject: p_yes}}}. This module is
pure CPU logic that reads that file -- no model calls -- and mirrors
anchor_common.build_agreement_rows' row shape so results can be joined/compared directly,
plus a paired McNemar test for the head-to-head question, a prompt-baseline comparison
(mirrors exp6_prompt_baseline.py), and a misbound-subset accuracy check (mirrors
exp7_misbound_subset.py) -- the two comparisons that actually matter for the paper: does
VQAScore also lose to "assume the prompt was obeyed," and does it do any better than
attention specifically on the rows where the model disobeyed?

`attribute_question` is imported from anchor_common (this file, unlike the Kaggle
kernels, can import local modules) -- both Kaggle kernels duplicate its logic inline,
same "keep in sync" convention as ANCHOR_PROMPTS across the generation scripts.

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

from anchor_common import (
    NON_SUBJECT_LABELS, attribute_question, binomial_test_vs_chance, label_key, load_labels,
    summarize_agreement,
)


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


def vqa_vs_prompt_baseline_report(rows: List[dict]) -> dict:
    """Same question as exp6_prompt_baseline.py's baseline_accuracy_report, asked of
    VQAScore's rows instead of attention's: does VQAScore also lose to "assume the model
    rendered what the prompt asked for"? `rows` is vqa_agreement_rows()'s output."""
    def _stratum(subset: List[dict]) -> dict:
        scored = [r for r in subset if r["scored"]]
        n_scored = len(scored)
        vqa_correct = base_correct = 0
        b = c = 0  # b: VQA correct, baseline wrong. c: VQA wrong, baseline correct.
        for r in scored:
            vqa_hit = r["correct"]
            base_hit = (r["human_label"] == r["intended_subject"])
            if vqa_hit:
                vqa_correct += 1
            if base_hit:
                base_correct += 1
            if vqa_hit and not base_hit:
                b += 1
            if not vqa_hit and base_hit:
                c += 1
        n_discordant = b + c
        p = (stats.binomtest(min(b, c), n_discordant, 0.5, alternative="two-sided").pvalue
             if n_discordant else None)
        return dict(
            n_scored=n_scored,
            vqa_acc=vqa_correct / n_scored if n_scored else None,
            base_acc=base_correct / n_scored if n_scored else None,
            b=b, c=c, p_value=p,
        )

    by_stratum = {n: _stratum([r for r in rows if r["n"] == n])
                  for n in sorted({r["n"] for r in rows})}
    return dict(overall=_stratum(rows), by_stratum=by_stratum)


def vqa_misbound_subset_report(rows: List[dict]) -> dict:
    """Same question as exp7_misbound_subset.py's misbound_analysis_report, asked of
    VQAScore's rows instead of attention's: restricted to rows where the model disobeyed
    the prompt (human_label != intended_subject), does VQAScore predict the RENDERED
    outcome above chance? This is the sharpest comparison for the paper -- the SOTA
    judge-based line, tested on exactly the rows a faithfulness metric exists to detect."""
    def _stratum(subset: List[dict]) -> dict:
        misbound = [r for r in subset
                    if r["scored"] and r["human_label"] != r["intended_subject"]]
        n_misbound = len(misbound)
        n_correct = sum(1 for r in misbound if r["correct"])
        ns = {r["n"] for r in subset}
        chance = (1.0 / next(iter(ns))) if len(ns) == 1 else None
        acc = (n_correct / n_misbound) if n_misbound else None
        p_val = binomial_test_vs_chance(n_correct, n_misbound, chance) if chance is not None else None
        return dict(n_misbound=n_misbound, n_correct=n_correct, accuracy=acc,
                    chance=chance, p_value=p_val)

    by_stratum = {n: _stratum([r for r in rows if r["n"] == n])
                  for n in sorted({r["n"] for r in rows})}
    return dict(overall=_stratum(rows), by_stratum=by_stratum)


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

    print("\n=== VQAScore vs. prompt-obeyed baseline (mirrors exp6_prompt_baseline.py) ===")
    baseline = vqa_vs_prompt_baseline_report(vqa_rows)
    for n, s in sorted(baseline["by_stratum"].items()):
        print(f"  n={n}: n_scored={s['n_scored']} vqa_acc={s['vqa_acc']} "
              f"base_acc={s['base_acc']} vqa>base={s['b']} base>vqa={s['c']} "
              f"p={s['p_value']}")
    o = baseline["overall"]
    print(f"  overall: n_scored={o['n_scored']} vqa_acc={o['vqa_acc']} "
          f"base_acc={o['base_acc']} vqa>base={o['b']} base>vqa={o['c']} p={o['p_value']}")

    print("\n=== VQAScore on the misbound subset (mirrors exp7_misbound_subset.py) ===")
    misbound = vqa_misbound_subset_report(vqa_rows)
    for n, s in sorted(misbound["by_stratum"].items()):
        print(f"  n={n}: n_misbound={s['n_misbound']} n_correct={s['n_correct']} "
              f"accuracy={s['accuracy']} chance={s['chance']} p={s['p_value']}")
    o = misbound["overall"]
    print(f"  overall: n_misbound={o['n_misbound']} n_correct={o['n_correct']} "
          f"accuracy={o['accuracy']}")
    print("\n  This is the paper's sharpest question: does the SOTA judge-based line do "
          "any better than attention on exactly the rows a faithfulness metric exists "
          "to detect?")
