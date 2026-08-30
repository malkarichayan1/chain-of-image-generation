"""
Tests for segment_cache.py (Stage 2). No torch/transformers import anywhere here --
segment_fn and image_loader are faked, matching the design's requirement that Stages 2
and 3 be testable "against a hand-built manifest with synthetic sigmoid maps" with no
model or GPU present.
"""
from pathlib import Path

import numpy as np

from segment_cache import (
    build_segmentation_cache,
    cache_key,
    get_sigmoid_map,
    load_cached_map,
    manifest_attribute_universe,
    manifest_image_paths,
    save_cached_map,
)


def fake_segment_fn(calls: list):
    """Deterministic synthetic sigmoid map, one value per (image, attribute) pair, and
    records every call so tests can assert on cache-hit behavior (no recomputation)."""

    def _segment(image, attribute: str) -> np.ndarray:
        calls.append((image, attribute))
        seed = abs(hash(attribute)) % 1000
        rng = np.random.RandomState(seed)
        return rng.rand(8, 8).astype(np.float32)

    return _segment


def fake_image_loader(image_path: str):
    return f"fake-image:{image_path}"


def _manifest_two_chains() -> dict:
    return {
        "chains": [
            {
                "chain_key": "p0_s42",
                "prompt_id": 0,
                "seed": 42,
                "n": 2,
                "prompt": "a photo of a barista and a cyclist",
                "detected": True,
                "base_image_path": "images/p0_s42_step0.png",
                "subject_boxes": {"barista": [0, 0, 1, 1], "cyclist": [1, 1, 2, 2]},
                "steps": [
                    {"step_idx": 1, "subject": "barista", "attribute": "red apron",
                     "image_path": "images/p0_s42_step1.png", "attention_path": "attn/p0_s42_step1.npy"},
                    {"step_idx": 2, "subject": "cyclist", "attribute": "yellow helmet",
                     "image_path": "images/p0_s42_step2.png", "attention_path": "attn/p0_s42_step2.npy"},
                ],
            },
            {
                "chain_key": "p3_s42",
                "prompt_id": 3,
                "seed": 42,
                "n": 1,
                "prompt": "a photo of a farmer",
                "detected": False,
                "base_image_path": "images/p3_s42_step0.png",
                "subject_boxes": {},
                "steps": [],
            },
        ]
    }


def test_cache_key_is_deterministic_and_distinguishes_pairs():
    k1 = cache_key("images/a.png", "red apron")
    k2 = cache_key("images/a.png", "red apron")
    k3 = cache_key("images/a.png", "yellow helmet")
    k4 = cache_key("images/b.png", "red apron")
    assert k1 == k2
    assert k1 != k3
    assert k1 != k4


def test_save_and_load_cached_map_round_trips(tmp_path: Path):
    original = np.random.RandomState(0).rand(4, 4).astype(np.float32)
    save_cached_map(tmp_path, "images/a.png", "red apron", original)
    loaded = load_cached_map(tmp_path, "images/a.png", "red apron")
    assert loaded is not None
    np.testing.assert_allclose(loaded, original, atol=1e-3)  # float16 cache, not exact


def test_load_cached_map_returns_none_when_absent(tmp_path: Path):
    assert load_cached_map(tmp_path, "images/missing.png", "red apron") is None


def test_get_sigmoid_map_computes_once_then_hits_cache(tmp_path: Path):
    calls: list = []
    segment_fn = fake_segment_fn(calls)

    first = get_sigmoid_map(tmp_path, "images/a.png", "red apron", segment_fn, fake_image_loader)
    second = get_sigmoid_map(tmp_path, "images/a.png", "red apron", segment_fn, fake_image_loader)

    assert len(calls) == 1  # second call was a cache hit, not a recompute
    np.testing.assert_allclose(first, second, atol=1e-3)


def test_manifest_attribute_universe_excludes_undetected_chains():
    manifest = _manifest_two_chains()
    assert manifest_attribute_universe(manifest) == ["red apron", "yellow helmet"]


def test_manifest_image_paths_excludes_undetected_chains():
    manifest = _manifest_two_chains()
    paths = manifest_image_paths(manifest)
    assert paths == [
        "images/p0_s42_step0.png",
        "images/p0_s42_step1.png",
        "images/p0_s42_step2.png",
    ]
    assert "images/p3_s42_step0.png" not in paths


def test_build_segmentation_cache_covers_full_image_by_attribute_cross_product(tmp_path: Path):
    manifest = _manifest_two_chains()
    calls: list = []
    segment_fn = fake_segment_fn(calls)

    stats = build_segmentation_cache(manifest, tmp_path, segment_fn=segment_fn, image_loader=fake_image_loader)

    # 3 detected images x 2 distinct attributes = 6 pairs, including combinations never
    # actually used by "real"/"shuffled" (e.g. step2's image against "red apron") -- Stage
    # 3 needs these for substituted/scrambled-cross draws, decided after Stage 2 already ran.
    assert stats["total_pairs"] == 6
    assert stats["pairs_computed"] == 6
    assert stats["pairs_already_cached"] == 0
    assert len(calls) == 6

    for image_path in manifest_image_paths(manifest):
        for attribute in manifest_attribute_universe(manifest):
            assert load_cached_map(tmp_path, image_path, attribute) is not None


def test_build_segmentation_cache_is_idempotent(tmp_path: Path):
    manifest = _manifest_two_chains()
    calls: list = []
    segment_fn = fake_segment_fn(calls)

    build_segmentation_cache(manifest, tmp_path, segment_fn=segment_fn, image_loader=fake_image_loader)
    stats_second_run = build_segmentation_cache(manifest, tmp_path, segment_fn=segment_fn, image_loader=fake_image_loader)

    assert stats_second_run["pairs_computed"] == 0
    assert stats_second_run["pairs_already_cached"] == 6
    assert len(calls) == 6  # no new CLIPSeg calls on the second run
