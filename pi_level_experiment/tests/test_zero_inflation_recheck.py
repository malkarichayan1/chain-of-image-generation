"""
Tests for zero_inflation_recheck.py (Part D, check 3): corrects Part C Step 7's claim that
all of real's zero-IoU rows are "genuine persistence" by checking threshold-crossing on the
actual cached sigmoid maps, not the continuous curr_mask_area proxy.
"""
import numpy as np
import pandas as pd

from segment_cache import save_cached_map
from zero_inflation_recheck import crosses_threshold, decompose_zero_iou_rows


def test_crosses_threshold_true_when_any_pixel_exceeds():
    m = np.array([[0.1, 0.9], [0.2, 0.3]], dtype=np.float32)
    assert crosses_threshold(m, threshold=0.85) is True


def test_crosses_threshold_false_when_no_pixel_exceeds():
    m = np.array([[0.1, 0.5], [0.2, 0.3]], dtype=np.float32)
    assert crosses_threshold(m, threshold=0.85) is False


def _chain(chain_key, steps):
    return {
        "chain_key": chain_key, "prompt_id": 0, "seed": 42, "n": 1, "prompt": "x",
        "detected": True, "base_image_path": f"{chain_key}_step0.png", "steps": steps,
    }


def _step(step_idx, attribute, image_path):
    return {
        "step_idx": step_idx, "subject": "s", "attribute": attribute,
        "image_path": image_path, "attention_path": "irrelevant.npy",
    }


def test_decompose_flags_never_detected_when_curr_never_crosses_threshold(tmp_path):
    cache_dir = tmp_path / "cache"
    manifest = {"chains": [_chain("c0", [_step(1, "hat", "c0_step1.png")])]}
    save_cached_map(cache_dir, "c0_step1.png", "hat", np.full((4, 4), 0.3, dtype=np.float32))

    df = pd.DataFrame([{
        "condition": "real", "chain_key": "c0", "attribute": "hat", "true_step": 1,
        "iou": 0.0, "delta_area": 0.0,
    }])
    result = decompose_zero_iou_rows(df, manifest, cache_dir, threshold=0.85)
    assert result == {"total_zero_iou_rows": 1, "never_detected_in_curr": 1, "genuine_persistence": 0}


def test_decompose_flags_genuine_persistence_when_curr_crosses_threshold(tmp_path):
    cache_dir = tmp_path / "cache"
    manifest = {"chains": [_chain("c0", [_step(1, "hat", "c0_step1.png")])]}
    save_cached_map(cache_dir, "c0_step1.png", "hat", np.full((4, 4), 0.95, dtype=np.float32))

    df = pd.DataFrame([{
        "condition": "real", "chain_key": "c0", "attribute": "hat", "true_step": 1,
        "iou": 0.0, "delta_area": 0.0,
    }])
    result = decompose_zero_iou_rows(df, manifest, cache_dir, threshold=0.85)
    assert result == {"total_zero_iou_rows": 1, "never_detected_in_curr": 0, "genuine_persistence": 1}


def test_decompose_ignores_rows_that_are_not_zero_iou():
    manifest = {"chains": [_chain("c0", [_step(1, "hat", "c0_step1.png")])]}
    df = pd.DataFrame([{
        "condition": "real", "chain_key": "c0", "attribute": "hat", "true_step": 1,
        "iou": 0.5, "delta_area": 0.2,
    }])
    result = decompose_zero_iou_rows(df, manifest, "unused_cache_dir", threshold=0.85)
    assert result == {"total_zero_iou_rows": 0, "never_detected_in_curr": 0, "genuine_persistence": 0}
