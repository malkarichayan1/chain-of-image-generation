"""
Tests for recompute_boxes.py: pure caching/orchestration logic only. The actual Mask R-CNN
+ CLIP detection calls are not unit tested here (same split as owlvit_cross_check.py's
untested, lazily-loaded model-calling closure) -- this covers which images still need a
rerun and the on-disk cache round-trip.
"""
from recompute_boxes import load_box_cache, pending_box_targets, save_box_cache


def _manifest(images):
    return {"images": images}


def _image(prompt_id, detected=True):
    return {"prompt_id": prompt_id, "detected": detected}


def test_pending_box_targets_skips_already_cached_prompts():
    manifest = _manifest([_image(0), _image(1)])
    cache = {0: {"barista": [0, 0, 10, 10]}}
    pending = pending_box_targets(manifest, cache)
    assert [img["prompt_id"] for img in pending] == [1]


def test_pending_box_targets_skips_undetected_images():
    manifest = _manifest([_image(0, detected=False), _image(1)])
    pending = pending_box_targets(manifest, {})
    assert [img["prompt_id"] for img in pending] == [1]


def test_pending_box_targets_empty_when_all_cached():
    manifest = _manifest([_image(0), _image(1)])
    cache = {0: {}, 1: {}}
    assert pending_box_targets(manifest, cache) == []


def test_box_cache_round_trips_through_json_with_int_keys(tmp_path):
    path = tmp_path / "boxes.json"
    cache = {0: {"barista": [1.0, 2.0, 3.0, 4.0]}, 12: {"chef": [5, 6, 7, 8]}}
    save_box_cache(path, cache)
    loaded = load_box_cache(path)
    assert loaded == cache
    assert all(isinstance(k, int) for k in loaded)


def test_load_box_cache_missing_file_returns_empty_dict(tmp_path):
    assert load_box_cache(tmp_path / "missing.json") == {}
