"""
Tests for equivalence_check.py (docs/part-c-validation-design.md, Step 1 -- the
validation gate): confirms score_chains.py's ported Phase A/C functions produce
identical output to the canonical pilot/spatial_semantic_alignment.py implementation
they were copied from. Everything downstream in Part C assumes this holds.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from equivalence_check import (
    check_delta_mask_equivalence_grid,
    check_iou_equivalence_grid,
    original_delta_mask,
    original_iou,
    port_delta_mask,
    port_iou,
)


def test_delta_mask_matches_on_hand_built_case():
    curr = np.array([[0.9, 0.1], [0.6, 0.4]])
    prev = np.array([[0.1, 0.1], [0.6, 0.9]])
    a = port_delta_mask(curr, prev, 0.5)
    b = original_delta_mask(curr, prev, 0.5)
    np.testing.assert_array_equal(a, b)


def test_iou_matches_on_hand_built_case():
    attention_map = np.array([
        [1.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ])
    delta_mask = attention_map.copy()
    a = port_iou(attention_map, delta_mask, threshold_pct=0.25)
    b = original_iou(attention_map, delta_mask, threshold_pct=0.25)
    assert a == pytest.approx(b)


def test_delta_mask_equivalence_grid_has_no_mismatches():
    mismatches = check_delta_mask_equivalence_grid(
        shapes=[(4, 4), (16, 16), (32, 32)],
        thresholds=[0.1, 0.5, 0.85, 0.95],
        n_trials=5,
        seed=0,
    )
    assert mismatches == []


def test_iou_equivalence_grid_has_no_mismatches_on_matching_shapes():
    mismatches = check_iou_equivalence_grid(
        shapes=[(4, 4), (16, 16), (32, 32)],
        threshold_pcts=[0.05, 0.20, 0.40],
        n_trials=5,
        seed=1,
    )
    assert mismatches == []


def test_iou_equivalence_grid_documents_the_known_resize_divergence():
    """The port resizes with scipy.ndimage.zoom; the original resizes with
    torch.F.interpolate. Per the design doc's Step 1 decision rule this is only
    acceptable because no real chain's attention/delta shapes ever differ (confirmed
    by test_no_real_chain_hits_the_resize_path below) -- this test just documents
    that the two resize methods do in fact diverge, so that premise stays load-bearing
    rather than assumed."""
    mismatches = check_iou_equivalence_grid(
        shapes=[(4, 4)],
        threshold_pcts=[0.25],
        n_trials=3,
        seed=2,
        mismatched_target_shape=(8, 8),
    )
    assert len(mismatches) > 0, (
        "expected the scipy.zoom vs torch.interpolate resize paths to diverge on "
        "mismatched shapes -- if this now passes, the resize-path caveat in the "
        "design doc's Step 1 can be downgraded"
    )


def test_no_real_chain_hits_the_resize_path():
    """Guards the premise behind the resize-path caveat: every cached attention .npy
    in the real combined manifest must already be at the same resolution as the
    delta mask it's compared against, so the untested/divergent resize path in
    either implementation never actually executes against real data."""
    manifest_path = Path("artifacts/manifest_combined.json")
    if not manifest_path.exists():
        pytest.skip("manifest_combined.json not present locally (gitignored data artifact)")
    with open(manifest_path) as f:
        manifest = json.load(f)
    img_size = manifest["img_size"]
    for chain in manifest["chains"]:
        if not chain["detected"]:
            continue
        for step in chain["steps"]:
            attn = np.load(step["attention_path"])
            assert attn.shape == (img_size, img_size), (
                f"{step['attention_path']} has shape {attn.shape}, expected "
                f"({img_size}, {img_size}) -- this chain would hit the "
                "resize-divergence path documented above"
            )
