#!/usr/bin/env python
"""
Part D, check 3: corrects Part C Step 7 (docs/part-c-validation-design.md, RESULTS.md).

Step 7 claimed all 52 of real's zero-IoU rows were "genuine persistence" (CLIPSeg detected
the attribute above T=0.85 in *both* the current and previous image) and zero were
"never detected in curr at all" (noise floor). That claim was never backed by a committed
script -- grepping this repo for it turns up nothing. Hand-inspecting a sample of those 52
rows against the actual cached images (Part D writeup, RESULTS.md) found several where the
attribute is not visibly present in *either* frame, which is inconsistent with genuine
persistence. This module recomputes the decomposition directly from the same cached sigmoid
maps and the same calibrated threshold `score_chains.delta_mask_from_sigmoid` uses, rather
than from a proxy: `curr_mask_area` in the scored CSV is `sigmoid.mean()` (a continuous
value, virtually never exactly 0), which is a different question from "does the sigmoid
cross T=0.85 anywhere" -- confusing the two would trivially always call every row
"detected in curr," reproducing Step 7's reported 0/52 by construction regardless of the
real signal.
"""
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from score_chains import ChainRecord, chain_image_path, detected_chains, parse_manifest
from segment_cache import load_cached_map


def crosses_threshold(sigmoid_map: np.ndarray, threshold: float) -> bool:
    return bool((sigmoid_map > threshold).any())


def decompose_zero_iou_rows(
    df: pd.DataFrame, manifest: dict, cache_dir: Path,
    threshold: float = 0.85, condition: str = "real",
) -> Dict[str, int]:
    """For each row of `condition` with iou==0 and delta_area==0, checks whether the
    attribute's sigmoid map ever crosses `threshold` in the *current*-step image. If it
    never does, the row's empty delta mask reflects CLIPSeg never confidently detecting the
    attribute at all (noise floor) -- not the attribute being visible-but-already-present
    in the previous step (genuine persistence), which requires curr to cross threshold
    somewhere too (and by delta_area==0's own definition, prev covers wherever curr does)."""
    chains_by_key: Dict[str, ChainRecord] = {
        c.chain_key: c for c in detected_chains(parse_manifest(manifest))
    }
    sub = df[df["condition"] == condition]
    zero_rows = sub[(sub["iou"] == 0) & (sub["delta_area"] == 0)]

    never_detected_in_curr = 0
    genuine_persistence = 0
    for row in zero_rows.itertuples():
        chain = chains_by_key[row.chain_key]
        true_step = int(row.true_step)
        curr_path = chain_image_path(chain, true_step)
        curr_map = load_cached_map(cache_dir, curr_path, row.attribute)
        if crosses_threshold(curr_map, threshold):
            genuine_persistence += 1
        else:
            never_detected_in_curr += 1

    return dict(
        total_zero_iou_rows=len(zero_rows),
        never_detected_in_curr=never_detected_in_curr,
        genuine_persistence=genuine_persistence,
    )


if __name__ == "__main__":
    manifest = json.loads(Path("artifacts/manifest_combined.json").read_text())
    df = pd.read_csv("artifacts/chain_experiment_results_v7.csv")
    result = decompose_zero_iou_rows(df, manifest, Path("artifacts/segmentation_cache"))
    print(result)
    total = result["total_zero_iou_rows"]
    if total:
        print(f"never_detected_in_curr: {result['never_detected_in_curr']}/{total} "
              f"({result['never_detected_in_curr']/total:.1%})")
        print(f"genuine_persistence: {result['genuine_persistence']}/{total} "
              f"({result['genuine_persistence']/total:.1%})")
