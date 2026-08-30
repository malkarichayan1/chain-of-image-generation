#!/usr/bin/env python
"""
Backfills `model_scores_full`/`predicted_owner_full` (Experiment 2's full-trajectory
attention aggregate, see exp2_window_ablation.py) onto an already-labeled anchor-set manifest,
using a separate PIN_SEEDS_FROM_MANIFEST rerun's output.

`merge_anchor_manifests.py` is the wrong tool for this: it merges DISJOINT prompt_id sets and
refuses on any duplicate id. This rerun reuses the SAME ids on purpose (PIN_SEEDS_FROM_MANIFEST
forces every prompt to reuse its already-recorded seed, guaranteeing a near-identical
regeneration) -- it's a backfill onto existing entries, not new prompts.

Because GPU kernels are not always bit-deterministic across runs even with the same seed
(non-deterministic cuDNN algorithms), this refuses to trust the rerun blindly. It first checks
that the rerun actually reproduced the same binding call (`predicted_owner`, must match exactly
-- a flip means the rerun landed on a materially different image, not just float noise) and the
same `model_scores` within a small tolerance, for every (prompt_id, attribute) both manifests
share. Only if nothing mismatches does it copy the two new fields across, leaving every other
field -- image, seed, predicted_owner, model_scores, and (in a separate file entirely) human
labels -- untouched. This is the same "strictly additive, no re-labeling" guarantee
generate_anchor_images_sdxl.py's own docstring promises for the patch that produces these
fields in the first place.

    py -3 backfill_model_scores_full.py ORIGINAL.json RERUN.json OUT.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from config import Config

DEFAULT_SCORE_TOLERANCE = Config().significance.score_tolerance


def index_attributes(manifest: dict) -> Dict[Tuple[int, str], dict]:
    """(prompt_id, attribute) -> attribute dict, only for detected images."""
    out: Dict[Tuple[int, str], dict] = {}
    for img in manifest["images"]:
        if not img.get("detected"):
            continue
        for attr in img["attributes"]:
            out[(img["prompt_id"], attr["attribute"])] = attr
    return out


def verify_rerun_matches(original: dict, rerun: dict,
                         score_tolerance: float = DEFAULT_SCORE_TOLERANCE) -> List[str]:
    """Human-readable mismatch descriptions for every (prompt_id, attribute) both manifests
    share. Empty means the pinned-seed rerun reproduced the original closely enough to trust
    its model_scores_full. Keys present in only one manifest are not mismatches -- a rerun
    that covers a strict subset (e.g. GROWTH_PROMPT_IDS left scoped) is a legitimate partial
    backfill, not a reproduction failure."""
    original_by_key = index_attributes(original)
    rerun_by_key = index_attributes(rerun)
    mismatches: List[str] = []
    for key, rerun_attr in rerun_by_key.items():
        orig_attr = original_by_key.get(key)
        if orig_attr is None:
            continue
        prompt_id, attribute = key
        if orig_attr["predicted_owner"] != rerun_attr["predicted_owner"]:
            mismatches.append(
                f"p{prompt_id}/{attribute}: predicted_owner changed "
                f"{orig_attr['predicted_owner']!r} -> {rerun_attr['predicted_owner']!r} "
                f"-- rerun is not a faithful reproduction")
            continue
        for subject, orig_score in orig_attr["model_scores"].items():
            rerun_score = rerun_attr["model_scores"].get(subject)
            if rerun_score is None or abs(rerun_score - orig_score) > score_tolerance:
                mismatches.append(
                    f"p{prompt_id}/{attribute}: model_scores[{subject!r}] changed "
                    f"{orig_score} -> {rerun_score} (tolerance {score_tolerance})")
    return mismatches


def backfill_model_scores_full(original: dict, rerun: dict) -> dict:
    """Returns a NEW manifest: a deep copy of `original` with `predicted_owner_full`/
    `model_scores_full` copied onto each attribute from the matching rerun entry. An
    attribute with no match in `rerun` gets both fields set to None (a legitimate partial
    result exp2_window_ablation.py's is_full_trajectory_available already handles), not an
    error -- call verify_rerun_matches first if you need to guard against a bad rerun."""
    rerun_by_key = index_attributes(rerun)
    merged = json.loads(json.dumps(original))
    for img in merged["images"]:
        if not img.get("detected"):
            continue
        for attr in img["attributes"]:
            rerun_attr = rerun_by_key.get((img["prompt_id"], attr["attribute"]))
            attr["predicted_owner_full"] = rerun_attr["predicted_owner_full"] if rerun_attr else None
            attr["model_scores_full"] = rerun_attr["model_scores_full"] if rerun_attr else None
    return merged


def main() -> None:
    if len(sys.argv) != 4:
        print("usage: backfill_model_scores_full.py ORIGINAL.json RERUN.json OUT.json")
        sys.exit(1)
    original_path, rerun_path, out_path = sys.argv[1:4]
    original = json.loads(Path(original_path).read_text())
    rerun = json.loads(Path(rerun_path).read_text())

    mismatches = verify_rerun_matches(original, rerun)
    if mismatches:
        print(f"REFUSING to backfill: {len(mismatches)} mismatch(es) between {original_path} "
              f"and {rerun_path} -- the pinned-seed rerun did not faithfully reproduce the "
              f"original, so its model_scores_full cannot be trusted.")
        for m in mismatches[:20]:
            print(f"  - {m}")
        sys.exit(1)

    merged = backfill_model_scores_full(original, rerun)
    rerun_available = len(index_attributes(rerun))
    total = sum(len(img["attributes"]) for img in merged["images"] if img.get("detected"))
    Path(out_path).write_text(json.dumps(merged, indent=2))
    print(f"Backfilled model_scores_full onto {out_path}: "
          f"{rerun_available}/{total} attributes filled from {rerun_path}")


if __name__ == "__main__":
    main()
