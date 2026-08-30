#!/usr/bin/env python
"""
Part C, Step 1 -- the validation gate. score_chains.py's
Phase A/C functions (delta_mask_from_sigmoid, iou_top_k) are a deliberate *port* of
common/spatial_semantic_alignment.py's SpatialSemanticAlignment class, not an import (see
score_chains.py's module docstring). Every published p-value in RESULTS.md comes from the
port; the Scenario 1-10 suite only ever validated the original. This module asserts the two
agree, so downstream re-analysis of the growth run trusts code that matches what was validated.
"""
import sys
from typing import List, Optional, Tuple

import numpy as np

from common.spatial_semantic_alignment import SpatialSemanticAlignment

from score_chains import delta_mask_from_sigmoid, iou_top_k


def _pilot_metric() -> SpatialSemanticAlignment:
    # lazy_load_model=True (default) means the CLIPSeg model is never loaded --
    # phase_a_delta_mask_from_masks and phase_c_alignment_score touch only numpy/torch
    # tensor math, never the segmentation model.
    return SpatialSemanticAlignment(device="cpu", lazy_load_model=True)


def port_delta_mask(curr: np.ndarray, prev: np.ndarray, threshold: float) -> np.ndarray:
    return delta_mask_from_sigmoid(curr, prev, threshold)


def original_delta_mask(curr: np.ndarray, prev: np.ndarray, threshold: float) -> np.ndarray:
    return _pilot_metric().phase_a_delta_mask_from_masks(curr, prev, threshold=threshold)


def port_iou(attention_map: np.ndarray, delta_mask: np.ndarray, threshold_pct: float) -> float:
    return iou_top_k(attention_map, delta_mask, threshold_pct=threshold_pct)


def original_iou(attention_map: np.ndarray, delta_mask: np.ndarray, threshold_pct: float) -> float:
    return _pilot_metric().phase_c_alignment_score(attention_map, delta_mask, threshold_pct=threshold_pct)


def check_delta_mask_equivalence_grid(
    shapes: List[Tuple[int, int]], thresholds: List[float], n_trials: int = 5, seed: int = 0,
) -> List[dict]:
    """Returns a list of mismatch records; empty means full equivalence across the grid."""
    rng = np.random.RandomState(seed)
    mismatches = []
    for shape in shapes:
        for threshold in thresholds:
            for trial in range(n_trials):
                curr = rng.rand(*shape).astype(np.float32)
                prev = rng.rand(*shape).astype(np.float32)
                a = port_delta_mask(curr, prev, threshold)
                b = original_delta_mask(curr, prev, threshold)
                if not np.array_equal(a, b):
                    mismatches.append(dict(shape=shape, threshold=threshold, trial=trial,
                                            n_diff_pixels=int(np.sum(a != b))))
    return mismatches


def check_iou_equivalence_grid(
    shapes: List[Tuple[int, int]], threshold_pcts: List[float], n_trials: int = 5, seed: int = 0,
    mismatched_target_shape: Optional[Tuple[int, int]] = None,
) -> List[dict]:
    """Returns a list of mismatch records; empty means full equivalence across the grid.

    mismatched_target_shape, when given, generates the attention map at that shape while
    the delta_mask stays at `shape` -- exercises the one known implementation divergence
    (scipy.zoom vs torch.interpolate resize), see the design doc's Step 1."""
    rng = np.random.RandomState(seed)
    mismatches = []
    for shape in shapes:
        for threshold_pct in threshold_pcts:
            for trial in range(n_trials):
                delta_mask = (rng.rand(*shape) > 0.7).astype(np.float32)
                attn_shape = mismatched_target_shape if mismatched_target_shape else shape
                attention_map = rng.rand(*attn_shape).astype(np.float32)
                a = port_iou(attention_map, delta_mask, threshold_pct)
                b = original_iou(attention_map, delta_mask, threshold_pct)
                if not np.isclose(a, b, atol=1e-6):
                    mismatches.append(dict(shape=shape, attn_shape=attn_shape,
                                            threshold_pct=threshold_pct, trial=trial,
                                            port_iou=a, original_iou=b))
    return mismatches


if __name__ == "__main__":
    delta_mismatches = check_delta_mask_equivalence_grid(
        shapes=[(4, 4), (16, 16), (32, 32), (64, 64)],
        thresholds=[0.05, 0.1, 0.5, 0.85, 0.95],
        n_trials=10,
    )
    iou_mismatches = check_iou_equivalence_grid(
        shapes=[(4, 4), (16, 16), (32, 32), (64, 64)],
        threshold_pcts=[0.05, 0.20, 0.40],
        n_trials=10,
    )
    print(f"Delta mask equivalence: {len(delta_mismatches)} mismatches "
          f"(same-shape grid, {5 * 4 * 10} cases)")
    print(f"IoU equivalence: {len(iou_mismatches)} mismatches "
          f"(same-shape grid, {4 * 3 * 10} cases)")
    if delta_mismatches or iou_mismatches:
        print("MISMATCH DETECTED -- see Part C Step 1 decision rule.")
        sys.exit(1)
    print("Port and original implementations agree on every same-shape case checked.")
