"""
Tests for score_chains.py (Stage 3): golden IoU test, structural-zero substituted test, disjointness assertion,
manifest round-trip, threshold monotonicity. No torch/diffusers/GPU anywhere.
"""
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from score_chains import (
    ChainRecord,
    ChainStepRecord,
    delta_mask_from_sigmoid,
    detected_chains,
    iou_top_k,
    parse_manifest,
    score_chains,
    select_foreign_attribute,
)
from segment_cache import save_cached_map


def test_golden_iou_at_known_threshold():
    # 4x4 attention map with a clear top-left hot region, delta mask covering exactly
    # that region -> perfect overlap at threshold_pct selecting exactly those 4 cells.
    attention_map = np.array([
        [1.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ])
    delta_mask = np.array([
        [1.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ])
    iou = iou_top_k(attention_map, delta_mask, threshold_pct=0.25)  # top 4 of 16 cells
    assert iou == pytest.approx(1.0)


def test_delta_mask_from_sigmoid_matches_hand_computation():
    curr = np.array([[0.9, 0.1], [0.6, 0.4]])
    prev = np.array([[0.1, 0.1], [0.6, 0.9]])
    delta = delta_mask_from_sigmoid(curr, prev, threshold=0.5)
    # (0,0): curr>0.5 and not prev>0.5 -> True. (0,1): curr not>0.5 -> False.
    # (1,0): curr>0.5 but prev>0.5 too -> False. (1,1): curr not>0.5 -> False.
    expected = np.array([[1.0, 0.0], [0.0, 0.0]])
    np.testing.assert_array_equal(delta, expected)


def test_threshold_monotonicity_when_prev_has_no_signal():
    # prev genuinely absent (near-zero) everywhere -- the real "attribute never locked
    # in yet" case. Delta(T) then reduces to {curr > T}, whose area is a survival
    # function of curr and therefore mathematically non-increasing in T for any curr.
    rng = np.random.RandomState(0)
    curr = rng.rand(16, 16).astype(np.float32)
    prev = np.zeros((16, 16), dtype=np.float32)
    areas = [delta_mask_from_sigmoid(curr, prev, t).sum() for t in np.linspace(0.05, 0.95, 19)]
    assert all(areas[i] >= areas[i + 1] for i in range(len(areas) - 1))


def _two_chain_manifest_with_shared_attribute(tmp_path: Path):
    """Chain 0 and chain 3 both use "red apron" -- the exact overlap pattern that exposed
    the v4 substituted-selection bug (design doc's "Correctness bug to fix here")."""
    cache_dir = tmp_path / "cache"
    img_dir = tmp_path / "images"
    attn_dir = tmp_path / "attn"

    def img(name: str) -> str:
        return str(img_dir / name)

    def attn(name: str, hot: bool) -> str:
        # A uniform (all-equal) map trips iou_top_k's max==min degenerate-map guard --
        # use a peaked pattern so "hot" attention has genuine top-k structure to score.
        peaked = np.array([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        arr = peaked if hot else np.zeros((4, 4), dtype=np.float32)
        p = attn_dir / f"{name}.npy"
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(p, arr)
        return str(p)

    def seg(image_path: str, attribute: str, present: bool):
        arr = np.ones((4, 4), dtype=np.float32) if present else np.zeros((4, 4), dtype=np.float32)
        save_cached_map(cache_dir, image_path, attribute, arr)

    c0_base, c0_s1 = img("c0_step0.png"), img("c0_step1.png")
    seg(c0_base, "red apron", present=False)
    seg(c0_s1, "red apron", present=True)
    seg(c0_base, "yellow helmet", present=False)
    seg(c0_s1, "yellow helmet", present=False)

    c3_base, c3_s1 = img("c3_step0.png"), img("c3_step1.png")
    seg(c3_base, "red apron", present=False)
    seg(c3_s1, "red apron", present=True)
    seg(c3_base, "yellow helmet", present=False)
    seg(c3_s1, "yellow helmet", present=False)

    manifest = {
        "chains": [
            {
                "chain_key": "p0_s42", "prompt_id": 0, "seed": 42, "n": 1,
                "prompt": "a photo of a barista wearing a red apron",
                "detected": True, "base_image_path": c0_base, "subject_boxes": {},
                "steps": [{"step_idx": 1, "subject": "barista", "attribute": "red apron",
                           "image_path": c0_s1, "attention_path": attn("c0_step1", hot=True)}],
            },
            {
                "chain_key": "p3_s42", "prompt_id": 3, "seed": 42, "n": 1,
                "prompt": "a photo of a barista wearing a red apron",
                "detected": True, "base_image_path": c3_base, "subject_boxes": {},
                "steps": [{"step_idx": 1, "subject": "barista", "attribute": "red apron",
                           "image_path": c3_s1, "attention_path": attn("c3_step1", hot=True)}],
            },
        ]
    }
    return manifest, cache_dir


def test_select_foreign_attribute_refuses_when_no_disjoint_attribute_exists(tmp_path: Path):
    manifest, _ = _two_chain_manifest_with_shared_attribute(tmp_path)
    chains = parse_manifest(manifest)
    rng = random.Random(0)
    # Both chains only ever have "red apron" as an attribute in this fixture, so there is
    # no valid foreign attribute -- selection must refuse rather than silently pick "red
    # apron" from the other chain (the v4 bug this replaces).
    with pytest.raises(ValueError):
        select_foreign_attribute(chains[0], chains, rng)


def test_select_foreign_attribute_picks_disjoint_attribute_when_available():
    steps_a = (ChainStepRecord(1, "barista", "red apron", "a1.png", "a1.npy"),)
    steps_b = (ChainStepRecord(1, "cyclist", "yellow helmet", "b1.png", "b1.npy"),)
    chain_a = ChainRecord("a", 0, 42, 1, "p", True, "a0.png", steps_a)
    chain_b = ChainRecord("b", 1, 42, 1, "p", True, "b0.png", steps_b)
    rng = random.Random(1)
    for _ in range(20):
        chosen = select_foreign_attribute(chain_a, [chain_a, chain_b], rng)
        assert chosen not in chain_a.attributes
        assert chosen == "yellow helmet"


def _manifest_with_disjoint_foreign_attribute(tmp_path: Path):
    manifest, cache_dir = _two_chain_manifest_with_shared_attribute(tmp_path)
    manifest["chains"][1]["steps"][0]["attribute"] = "yellow helmet"
    save_cached_map(cache_dir, manifest["chains"][1]["base_image_path"], "yellow helmet",
                     np.zeros((4, 4), dtype=np.float32))
    save_cached_map(cache_dir, manifest["chains"][1]["steps"][0]["image_path"], "yellow helmet",
                     np.ones((4, 4), dtype=np.float32))
    # Chain 3's attribute is now "yellow helmet", not "red apron" -- correct the base
    # fixture's stale "red apron present" cache entry for chain 3's own images so "red
    # apron" (chain 0's attribute, now a valid disjoint substituted candidate for chain 3)
    # is genuinely absent there, matching what "structurally zero" is meant to test.
    save_cached_map(cache_dir, manifest["chains"][1]["steps"][0]["image_path"], "red apron",
                     np.zeros((4, 4), dtype=np.float32))
    return manifest, cache_dir


def test_substituted_is_structurally_zero_when_foreign_attribute_absent(tmp_path: Path):
    manifest, cache_dir = _manifest_with_disjoint_foreign_attribute(tmp_path)
    df = score_chains(manifest, cache_dir, threshold=0.5, seed=0)
    substituted = df[df.condition == "substituted"]
    assert len(substituted) > 0
    assert (substituted["iou"] == 0.0).all()


def test_manifest_round_trip_gives_identical_scoring(tmp_path: Path):
    manifest, cache_dir = _manifest_with_disjoint_foreign_attribute(tmp_path)
    df_direct = score_chains(manifest, cache_dir, threshold=0.5, seed=0)

    manifest_path = tmp_path / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)
    with open(manifest_path) as f:
        reloaded = json.load(f)
    df_reloaded = score_chains(reloaded, cache_dir, threshold=0.5, seed=0)

    assert df_direct.reset_index(drop=True).equals(df_reloaded.reset_index(drop=True))


def test_real_condition_scores_higher_than_substituted_in_clean_synthetic_case(tmp_path: Path):
    manifest, cache_dir = _manifest_with_disjoint_foreign_attribute(tmp_path)
    df = score_chains(manifest, cache_dir, threshold=0.5, seed=0)
    real_mean = df[df.condition == "real"]["iou"].mean()
    sub_mean = df[df.condition == "substituted"]["iou"].mean()
    assert real_mean > sub_mean


def test_threshold_pct_param_reaches_iou_top_k(tmp_path: Path):
    """Part C Step 5: score_chains() must forward threshold_pct
    to every iou_top_k call, not just the module-level default of 0.20. In this fixture
    the real condition's delta mask covers the whole 4x4 image (curr all-present, prev
    all-absent -- see _manifest_with_disjoint_foreign_attribute), so top-k selection of
    k=round(threshold_pct*16) cells gives IoU == k/16 exactly -- a clean, independently
    verifiable way to confirm threshold_pct actually changes scoring, not just accepted
    and ignored."""
    manifest, cache_dir = _manifest_with_disjoint_foreign_attribute(tmp_path)
    df_20 = score_chains(manifest, cache_dir, threshold=0.5, seed=0, threshold_pct=0.20)
    df_50 = score_chains(manifest, cache_dir, threshold=0.5, seed=0, threshold_pct=0.50)
    real_20 = df_20[df_20.condition == "real"]["iou"].iloc[0]
    real_50 = df_50[df_50.condition == "real"]["iou"].iloc[0]
    assert real_20 == pytest.approx(3 / 16)
    assert real_50 == pytest.approx(8 / 16)


def test_new_columns_present_and_delta_area_matches_delta_mask_mean(tmp_path: Path):
    """Part C Steps 6-7: every row gets delta_area/
    curr_mask_area/prev_mask_area (Step 7) and iou_random_attn (Step 6)."""
    manifest, cache_dir = _manifest_with_disjoint_foreign_attribute(tmp_path)
    df = score_chains(manifest, cache_dir, threshold=0.5, seed=0)
    for col in ("curr_mask_area", "prev_mask_area", "delta_area", "iou_random_attn"):
        assert col in df.columns
    real = df[df.condition == "real"].iloc[0]
    # In this fixture real's delta mask covers the whole 4x4 image (curr all-present,
    # prev all-absent -- see _manifest_with_disjoint_foreign_attribute).
    assert real["curr_mask_area"] == pytest.approx(1.0)
    assert real["prev_mask_area"] == pytest.approx(0.0)
    assert real["delta_area"] == pytest.approx(1.0)


def test_iou_random_attn_is_deterministic_given_seed(tmp_path: Path):
    manifest, cache_dir = _manifest_with_disjoint_foreign_attribute(tmp_path)
    df1 = score_chains(manifest, cache_dir, threshold=0.5, seed=3)
    df2 = score_chains(manifest, cache_dir, threshold=0.5, seed=3)
    pd.testing.assert_series_equal(
        df1["iou_random_attn"].reset_index(drop=True),
        df2["iou_random_attn"].reset_index(drop=True),
    )


def test_ablation_rng_is_independent_of_condition_selection_rng(tmp_path: Path):
    """The Step 6 iou_random_attn ablation must draw from an RNG stream independent of
    score_chains.py's `rng`, which substituted/attn_scrambled_* already consume via
    rng.choice -- otherwise adding this column would silently shift which foreign
    attribute and scrambled partner get selected at a fixed seed, changing the
    already-published growth-run numbers. Verified by replaying the expected
    select_foreign_attribute rng.choice sequence independently (same iteration order
    score_chains.py itself uses) and checking it matches score_chains' actual picks."""
    manifest, cache_dir = _manifest_with_disjoint_foreign_attribute(tmp_path)
    df = score_chains(manifest, cache_dir, threshold=0.5, seed=7)

    all_chains = detected_chains(parse_manifest(manifest))
    expected_rng = random.Random(7)
    expected_foreign_attrs = [
        select_foreign_attribute(chain, all_chains, expected_rng)
        for chain in all_chains for _ in chain.steps
    ]
    actual_foreign_attrs = df[df.condition == "substituted"]["attribute"].tolist()
    assert actual_foreign_attrs == expected_foreign_attrs
