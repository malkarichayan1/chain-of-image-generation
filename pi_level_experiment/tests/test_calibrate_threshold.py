"""
Tests for calibrate_threshold.py (Step 3): refuses to run on non-calibration chains,
and picks T by the pre-registered criterion (max real nonzero rate s.t. substituted
nonzero rate == 0), per docs/part-b-strengthening-design.md.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from calibrate_threshold import (
    calibrate,
    restrict_to_calibration,
    select_calibrated_threshold,
    sweep_threshold,
)
from segment_cache import save_cached_map


def _held_out_only_manifest() -> dict:
    return {"chains": [{"chain_key": "p0_s42", "prompt_id": 0, "seed": 42, "n": 1,
                         "prompt": "p", "detected": True, "base_image_path": "b.png",
                         "subject_boxes": {}, "steps": []}]}


def test_restrict_to_calibration_refuses_when_only_held_out_chains_present():
    with pytest.raises(ValueError):
        restrict_to_calibration(_held_out_only_manifest())


def test_restrict_to_calibration_keeps_only_calibration_prompt_ids():
    manifest = {"chains": [
        {"chain_key": "p0_s42", "prompt_id": 0, "seed": 42, "n": 1, "prompt": "p",
         "detected": True, "base_image_path": "b0.png", "subject_boxes": {}, "steps": []},
        {"chain_key": "p3_s42", "prompt_id": 3, "seed": 42, "n": 1, "prompt": "p",
         "detected": True, "base_image_path": "b3.png", "subject_boxes": {}, "steps": []},
    ]}
    restricted = restrict_to_calibration(manifest)
    assert [c["prompt_id"] for c in restricted["chains"]] == [3]


def _calibration_manifest_with_cache(tmp_path: Path):
    """Two calibration-set chains (prompt_id 3 and 4), each with one attribute disjoint
    from the other, so substituted has a valid disjoint draw at every threshold."""
    cache_dir = tmp_path / "cache"
    attn_dir = tmp_path / "attn"

    def attn(name: str) -> str:
        peaked = np.array([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        p = attn_dir / f"{name}.npy"
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(p, peaked)
        return str(p)

    def seg(image_path: str, attribute: str, sigmoid_value: float):
        arr = np.full((4, 4), sigmoid_value, dtype=np.float32)
        save_cached_map(cache_dir, image_path, attribute, arr)

    # Chain 3 (calibration): "shovel" goes from 0.2 (base) to 0.9 (step1) -- a threshold
    # anywhere in (0.2, 0.9) correctly registers it as newly appeared.
    seg("p3_step0.png", "shovel", 0.2)
    seg("p3_step1.png", "shovel", 0.9)
    seg("p3_step0.png", "book", 0.1)   # foreign attribute, always absent
    seg("p3_step1.png", "book", 0.1)

    # Chain 4 (calibration): "book" goes from 0.2 to 0.9.
    seg("p4_step0.png", "book", 0.2)
    seg("p4_step1.png", "book", 0.9)
    seg("p4_step0.png", "shovel", 0.1)
    seg("p4_step1.png", "shovel", 0.1)

    manifest = {"chains": [
        {"chain_key": "p3_s42", "prompt_id": 3, "seed": 42, "n": 1, "prompt": "p",
         "detected": True, "base_image_path": "p3_step0.png", "subject_boxes": {},
         "steps": [{"step_idx": 1, "subject": "farmer", "attribute": "shovel",
                    "image_path": "p3_step1.png", "attention_path": attn("p3_step1")}]},
        {"chain_key": "p4_s42", "prompt_id": 4, "seed": 42, "n": 1, "prompt": "p",
         "detected": True, "base_image_path": "p4_step0.png", "subject_boxes": {},
         "steps": [{"step_idx": 1, "subject": "reader", "attribute": "book",
                    "image_path": "p4_step1.png", "attention_path": attn("p4_step1")}]},
    ]}
    return manifest, cache_dir


def test_sweep_threshold_only_uses_calibration_chains(tmp_path: Path):
    manifest, cache_dir = _calibration_manifest_with_cache(tmp_path)
    manifest["chains"].append({
        "chain_key": "p0_s42", "prompt_id": 0, "seed": 42, "n": 1, "prompt": "p",
        "detected": True, "base_image_path": "held_out.png", "subject_boxes": {}, "steps": [],
    })
    sweep = sweep_threshold(manifest, cache_dir, t_values=(0.3, 0.5, 0.7))
    # 2 real rows total (one per calibration chain) at every threshold -- the held-out
    # chain (no steps) contributes nothing, proving it was excluded, not just empty.
    assert (sweep["n_real"] == 2).all()


def test_select_calibrated_threshold_picks_smallest_t_maximizing_real_hit_rate(tmp_path: Path):
    manifest, cache_dir = _calibration_manifest_with_cache(tmp_path)
    sweep = sweep_threshold(manifest, cache_dir, t_values=(0.15, 0.3, 0.5, 0.7, 0.85))
    selected = select_calibrated_threshold(sweep)
    assert selected is not None
    row = sweep[sweep["threshold"] == selected].iloc[0]
    assert row["substituted_nonzero_rate"] == 0.0
    assert row["real_nonzero_rate"] == sweep[sweep["substituted_nonzero_rate"] == 0.0]["real_nonzero_rate"].max()


def test_select_calibrated_threshold_returns_none_when_no_t_satisfies_criterion():
    sweep = pd.DataFrame([
        dict(threshold=0.3, real_nonzero_rate=0.8, substituted_nonzero_rate=0.5, n_real=2, n_substituted=2),
        dict(threshold=0.5, real_nonzero_rate=0.6, substituted_nonzero_rate=0.2, n_real=2, n_substituted=2),
    ])
    assert select_calibrated_threshold(sweep) is None


def test_calibrate_end_to_end_returns_expected_shape(tmp_path: Path):
    manifest, cache_dir = _calibration_manifest_with_cache(tmp_path)
    result = calibrate(manifest, cache_dir, t_values=(0.3, 0.5, 0.7))
    assert result["calibration_prompt_ids"] == [3, 4, 6, 8]
    assert len(result["sweep"]) == 3
    assert "criterion" in result
