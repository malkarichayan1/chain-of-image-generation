#!/usr/bin/env python
"""
Is metric A's `predicted_owner` tracking subject bounding-box size/position rather than
attention content? CLAUDE.md flags this as unresolved: "cannot yet be separated from a
subject-box-overlap artifact -- the manifests do not store boxes." recompute_boxes.py
closes the missing-boxes gap (a sidecar boxes.json per artifacts dir); this is the analysis
that reads them, mirroring pi_level_experiment/discriminant_validity_check.py's approach
for metric B (partial correlation against a nuisance geometric regressor) but adapted to
metric A's structure: a per-attribute CLASSIFICATION among subjects, not a continuous IoU.

Four checks, cheapest/most direct first:
1. bigbox_baseline_report / mcnemar_report -- the headline comparison: does the trivial
   "always guess the biggest detected box" baseline match or beat the attention-based
   metric's own accuracy against human labels? If so, attention isn't adding information
   beyond box geometry.
2. bigbox_win_rate_report -- does the METRIC'S OWN pick correlate with box size,
   independent of correctness (a mechanistic check on the metric itself)?
3. intended_bigbox_rate_report -- does the PROMPT/GENERATION's ground truth (the human
   label) correlate with box size? A different question -- a construct-validity issue with
   the anchor set, not the metric.
4. margin_area_correlation -- does the metric's own confidence signal (top1-top2 margin)
   track box-area dominance rather than attention content?

Local CPU only (pandas/numpy/scipy against the manifest + boxes.json + labels already on
disk). Run from inside ssa/anchor_set/:
    py -3 discriminant_validity_check.py --artifacts-dir artifacts --annotator chayan
    py -3 discriminant_validity_check.py --artifacts-dir artifacts_sdxl --annotator chayan
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd
from scipy import stats

from anchor_common import build_agreement_rows, load_labels
from recompute_boxes import load_box_cache


def bbox_area(box: Sequence[float]) -> float:
    x0, y0, x1, y1 = box
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def join_boxes(rows: List[dict], boxes_by_prompt: Dict[int, Dict[str, list]],
                img_size: int) -> pd.DataFrame:
    """One row per agreement row, augmented with: `bigbox_subject` (the largest detected
    box in that image), `predicted_area_frac`, `predicted_matches_bigbox`,
    `intended_matches_bigbox`, and `area_advantage` (the predicted subject's area fraction
    minus the mean of the other subjects'). Rows whose prompt has no box data (detection
    mismatch on rerun) get these columns as None rather than being dropped, so callers can
    report exactly how much coverage was lost."""
    img_area = float(img_size) ** 2
    out = []
    for r in rows:
        boxes = boxes_by_prompt.get(r["prompt_id"])
        row = dict(r)
        if not boxes:
            row.update(bigbox_subject=None, predicted_area_frac=None,
                       predicted_matches_bigbox=None, intended_matches_bigbox=None,
                       area_advantage=None)
        else:
            areas = {s: bbox_area(b) / img_area for s, b in boxes.items()}
            bigbox = max(areas, key=lambda s: areas[s])
            predicted = r["predicted_owner"]
            others = [a for s, a in areas.items() if s != predicted]
            row.update(
                bigbox_subject=bigbox,
                predicted_area_frac=areas.get(predicted),
                predicted_matches_bigbox=(predicted == bigbox),
                intended_matches_bigbox=(r["intended_subject"] == bigbox),
                area_advantage=(areas.get(predicted, 0.0)
                                 - (sum(others) / len(others) if others else 0.0)),
            )
        out.append(row)
    return pd.DataFrame(out)


def _scored_with_boxes(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["scored"] == True) & df["bigbox_subject"].notna()]  # noqa: E712


def bigbox_baseline_report(df: pd.DataFrame) -> dict:
    """Headline check: metric accuracy vs. the trivial "always guess the biggest box"
    baseline, both against human labels, on the same rows."""
    sub = _scored_with_boxes(df)
    n = len(sub)
    metric_correct = int(sub["correct"].sum())
    baseline_correct = int((sub["bigbox_subject"] == sub["human_label"]).sum())
    return dict(n=n,
                metric_accuracy=(metric_correct / n) if n else None,
                bigbox_baseline_accuracy=(baseline_correct / n) if n else None)


def mcnemar_report(df: pd.DataFrame) -> dict:
    """Exact McNemar test (binomial on discordant pairs) for whether the metric and the
    bigbox baseline disagree in a systematically one-sided way on the same rows."""
    sub = _scored_with_boxes(df)
    baseline_correct = sub["bigbox_subject"] == sub["human_label"]
    metric_correct = sub["correct"].astype(bool)
    metric_only = int((metric_correct & ~baseline_correct).sum())
    baseline_only = int((~metric_correct & baseline_correct).sum())
    n_discordant = metric_only + baseline_only
    p = (stats.binomtest(metric_only, n_discordant, 0.5, alternative="two-sided").pvalue
         if n_discordant else None)
    return dict(n=len(sub), metric_only_correct=metric_only,
                baseline_only_correct=baseline_only, n_discordant=n_discordant, p_value=p)


def _rate_vs_chance_report(df: pd.DataFrame, column: str) -> dict:
    sub = df.dropna(subset=[column])
    n = len(sub)
    hits = int(sub[column].sum())
    chance = float((1.0 / sub["n"]).mean()) if n else None
    p = stats.binomtest(hits, n, chance, alternative="greater").pvalue if n else None
    return dict(n=n, hits=hits, rate=(hits / n) if n else None, chance=chance, p_value=p)


def bigbox_win_rate_report(df: pd.DataFrame) -> dict:
    """Does the metric's OWN pick land on the biggest box more often than the per-row
    chance rate (mean of 1/n across strata) would predict?"""
    return _rate_vs_chance_report(df, "predicted_matches_bigbox")


def intended_bigbox_rate_report(df: pd.DataFrame) -> dict:
    """Does the human-labeled ground truth itself correlate with box size -- a
    construct-validity question about the anchor set/prompts, not about the metric."""
    return _rate_vs_chance_report(df, "intended_matches_bigbox")


def margin_area_correlation(df: pd.DataFrame) -> dict:
    """Spearman correlation between the metric's own confidence (top1-top2 margin) and how
    much bigger the predicted subject's box is than the average of the others."""
    sub = df.dropna(subset=["margin", "area_advantage"])
    r, p = stats.spearmanr(sub["margin"], sub["area_advantage"])
    return dict(n=len(sub), spearman_r=float(r), spearman_p=float(p))


def _print_report(name: str, report: dict) -> None:
    print(f"\n{name}")
    for k, v in report.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Discriminant validity: is predicted_owner tracking box geometry?")
    ap.add_argument("--artifacts-dir", default="artifacts")
    ap.add_argument("--annotator", default="chayan")
    args = ap.parse_args()
    artifacts_dir = Path(args.artifacts_dir)

    manifest = json.loads((artifacts_dir / "manifest.json").read_text())
    labels = load_labels(artifacts_dir / f"labels_{args.annotator}.json")
    boxes = load_box_cache(artifacts_dir / "boxes.json")
    img_size = manifest.get("img_size", 512)

    rows = build_agreement_rows(manifest, labels, margin_threshold=0.02)
    df = join_boxes(rows, boxes, img_size)
    missing = df["bigbox_subject"].isna().sum()
    print(f"=== {artifacts_dir} ===  rows: {len(df)}  missing box data: {missing}")

    _print_report("Headline: metric accuracy vs. bigbox baseline accuracy",
                  bigbox_baseline_report(df))
    _print_report("McNemar (paired, same rows)", mcnemar_report(df))
    _print_report("Bigbox win rate (metric's own pick vs. box size)",
                  bigbox_win_rate_report(df))
    _print_report("Intended-subject bigbox rate (anchor-set construct validity)",
                  intended_bigbox_rate_report(df))
    _print_report("Margin vs. area-advantage correlation", margin_area_correlation(df))
