#!/usr/bin/env python
"""
Part A, Experiment 2: does early-window attention beat full-trajectory averaging? The literature
says layout gets decided early in denoising; later steps are mostly texture/noise for a binding
question. This directly tests whether the open sub-chance n=2/n=3 result was just an
aggregation-window mistake, or something deeper.

Requires manifest.json's `model_scores_full`/`predicted_owner_full` fields on every attribute --
a full-trajectory (max_steps=NUM_INFERENCE_STEPS) aggregate captured ALONGSIDE the existing
early-window (`model_scores`/`predicted_owner`, max_steps=15) one, added by the additive patch to
generate_anchor_images_sdxl.py. As of this writing, artifacts_sdxl/manifest.json does NOT have
this field yet -- that requires one Kaggle rerun of the patched script against the pinned seeds
already in the manifest. This script degrades gracefully (prints "unavailable", does not crash) when that field is missing, so
run_five_experiments.py can still complete its other four sections.

NOT YET RUN ON REAL DATA. Run against the dummy fixture today (populated with both windows):
    py -3 exp2_window_ablation.py --artifacts-dir artifacts_dummy --annotator dummy
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from scipy import stats

from anchor_common import build_agreement_rows, chance_baseline, load_labels

UNAVAILABLE_MESSAGE = (
    "Experiment 2 -- UNAVAILABLE: this manifest has no model_scores_full/predicted_owner_full "
    "on every detected attribute. Rerun generate_anchor_images_sdxl.py (with the additive "
    "model_scores_full patch) against the seeds already pinned in this manifest before this "
    "experiment can run.")


def full_scores_by_key(manifest: dict) -> Dict[Tuple[int, str], Optional[str]]:
    """(prompt_id, attribute) -> predicted_owner_full, or None where the field is absent/null
    (manifest predates the model_scores_full patch, or that one attribute lacks it)."""
    out: Dict[Tuple[int, str], Optional[str]] = {}
    for img in manifest["images"]:
        if not img.get("detected"):
            continue
        for attr in img["attributes"]:
            out[(img["prompt_id"], attr["attribute"])] = attr.get("predicted_owner_full")
    return out


def is_full_trajectory_available(manifest: dict) -> bool:
    """True only if every detected attribute has a non-None predicted_owner_full."""
    values = full_scores_by_key(manifest)
    return bool(values) and all(v is not None for v in values.values())


def join_window_rows(rows: List[dict], full_by_key: Dict[Tuple[int, str], Optional[str]]) -> List[dict]:
    """Augments each build_agreement_rows() row with `predicted_owner_full` and `correct_full`
    (None when that attribute has no full-trajectory prediction)."""
    out = []
    for r in rows:
        owner_full = full_by_key.get((r["prompt_id"], r["attribute"]))
        row = dict(r)
        row["predicted_owner_full"] = owner_full
        row["correct_full"] = (
            (r["scored"] and owner_full == r["human_label"]) if owner_full is not None else None)
        out.append(row)
    return out


def window_accuracy_report(rows: List[dict]) -> Dict[int, dict]:
    """Per-stratum early-window vs. full-trajectory accuracy against the same 1/n chance.
    `early_accuracy` denominator is every scored row; `full_accuracy` denominator is scored rows
    that also have a full-trajectory prediction (both should match once fully populated)."""
    out = {}
    strata = sorted({r["n"] for r in rows if r["scored"]})
    for n in strata:
        sub = [r for r in rows if r["scored"] and r["n"] == n]
        n_scored = len(sub)
        early_correct = sum(1 for r in sub if r["correct"])
        full_rows = [r for r in sub if r["correct_full"] is not None]
        full_correct = sum(1 for r in full_rows if r["correct_full"])
        out[n] = dict(
            n_scored=n_scored,
            early_accuracy=(early_correct / n_scored) if n_scored else None,
            full_accuracy=(full_correct / len(full_rows)) if full_rows else None,
            chance=chance_baseline(n),
        )
    return out


def mcnemar_early_vs_full(rows: List[dict]) -> dict:
    """Paired McNemar between early-window and full-trajectory correctness, on rows where both
    are defined."""
    sub = [r for r in rows if r["scored"] and r["correct_full"] is not None]
    n = len(sub)
    early_only = sum(1 for r in sub if r["correct"] and not r["correct_full"])
    full_only = sum(1 for r in sub if r["correct_full"] and not r["correct"])
    n_discordant = early_only + full_only
    p = (stats.binomtest(early_only, n_discordant, 0.5, alternative="two-sided").pvalue
         if n_discordant else None)
    return dict(n=n, early_only_correct=early_only, full_only_correct=full_only,
                n_discordant=n_discordant, p_value=p)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Experiment 2: early-window vs. full-trajectory attention accuracy")
    ap.add_argument("--artifacts-dir", default="artifacts")
    ap.add_argument("--annotator", required=True)
    args = ap.parse_args()
    artifacts_dir = Path(args.artifacts_dir)

    manifest = json.loads((artifacts_dir / "manifest.json").read_text())
    labels = load_labels(artifacts_dir / f"labels_{args.annotator}.json")
    if not labels:
        raise SystemExit(f"No labels found for annotator {args.annotator!r} in {artifacts_dir}")

    if not is_full_trajectory_available(manifest):
        print(UNAVAILABLE_MESSAGE)
        return

    rows = build_agreement_rows(manifest, labels)
    joined = join_window_rows(rows, full_scores_by_key(manifest))
    print(f"Experiment 2 -- early-window vs. full-trajectory attention "
          f"({artifacts_dir}, annotator={args.annotator})\n")
    for n, s in sorted(window_accuracy_report(joined).items()):
        print(f"  n={n}: early_accuracy={s['early_accuracy']:.3f} "
              f"full_accuracy={s['full_accuracy']:.3f} chance={s['chance']:.3f} "
              f"(n_scored={s['n_scored']})")
    print(f"\nMcNemar (early vs. full, paired): {mcnemar_early_vs_full(joined)}")


if __name__ == "__main__":
    main()
