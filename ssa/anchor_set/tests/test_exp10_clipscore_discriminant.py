"""Tests for exp10_clipscore_discriminant.py's pure logic (experiment #8): caption phrasing,
cache I/O, row-building from a precomputed clip_scores.json, and the agreement-rate
deliverable. No CLIP model is loaded -- compute_clip_scores_for_image is exercised only via
its callers' contracts (dict-in, dict-out), matching recompute_boxes.py's own
tested/untested split.
"""
import json

import pytest

from exp10_clipscore_discriminant import (
    attribute_caption, clip_agreement_rows, clip_predicted_owner, load_score_cache,
    pending_clip_targets, prediction_agreement_rate, save_score_cache,
)


# --------------------------------------------------------------- attribute_caption

@pytest.mark.parametrize("attribute,expected", [
    ("red apron", "a photo of a person wearing a red apron"),
    ("yellow bike helmet", "a photo of a person wearing a yellow bike helmet"),
    ("blue book", "a photo of a person holding a blue book"),
    ("green sunglasses", "a photo of a person wearing green sunglasses"),
])
def test_attribute_caption_matches_the_held_worn_classification(attribute, expected):
    assert attribute_caption(attribute) == expected


def test_attribute_caption_is_case_insensitive_on_the_classifying_noun():
    assert attribute_caption("a SHOVEL") == "a photo of a person holding a a SHOVEL"


# --------------------------------------------------------------- clip_predicted_owner

def test_clip_predicted_owner_picks_the_argmax():
    assert clip_predicted_owner({"barista": 0.1, "cyclist": 0.9}) == "cyclist"


def test_clip_predicted_owner_rejects_empty_scores():
    with pytest.raises(ValueError, match="empty"):
        clip_predicted_owner({})


# --------------------------------------------------------------- cache I/O

def test_score_cache_round_trips(tmp_path):
    path = tmp_path / "clip_scores.json"
    cache = {"7": {"red apron": {"barista": 0.5, "cyclist": 0.2}}}

    save_score_cache(path, cache)

    assert load_score_cache(path) == cache


def test_load_score_cache_missing_file_returns_empty_dict(tmp_path):
    assert load_score_cache(tmp_path / "nope.json") == {}


# --------------------------------------------------------------- pending_clip_targets

def _manifest():
    return {"images": [
        {"prompt_id": 1, "detected": True},
        {"prompt_id": 2, "detected": True},
        {"prompt_id": 3, "detected": False},   # no box, must never be pending
    ]}


def test_pending_clip_targets_requires_detection_and_a_box_and_no_cache_entry():
    manifest = _manifest()
    boxes = {"1": {"barista": [0, 0, 10, 10]}, "2": {"barista": [0, 0, 10, 10]}}
    cache = {"1": {}}   # already scored

    todo = pending_clip_targets(manifest, boxes, cache)

    assert [img["prompt_id"] for img in todo] == [2]


def test_pending_clip_targets_skips_images_with_no_box_entry():
    manifest = _manifest()
    boxes = {"1": {"barista": [0, 0, 10, 10]}}   # image 2 has no box entry

    todo = pending_clip_targets(manifest, boxes, cache={})

    assert [img["prompt_id"] for img in todo] == [1]


# --------------------------------------------------------------- clip_agreement_rows

def _manifest_with_attrs():
    return {"images": [
        {
            "prompt_id": 1, "n": 2, "detected": True,
            "attributes": [
                {"attribute": "red apron", "intended_subject": "barista"},
                {"attribute": "yellow helmet", "intended_subject": "cyclist"},
            ],
        },
        {"prompt_id": 2, "n": 2, "detected": False, "attributes": []},
    ]}


def test_clip_agreement_rows_marks_correct_when_owner_matches_human_label():
    manifest = _manifest_with_attrs()
    labels = {"1::red apron": "barista", "1::yellow helmet": "cyclist"}
    clip_scores = {"1": {
        "red apron": {"barista": 0.9, "cyclist": 0.1},
        "yellow helmet": {"barista": 0.9, "cyclist": 0.1},
    }}

    rows = clip_agreement_rows(manifest, labels, clip_scores)

    assert len(rows) == 2
    by_attr = {r["attribute"]: r for r in rows}
    assert by_attr["red apron"]["correct"] is True     # owner barista == human barista
    assert by_attr["yellow helmet"]["correct"] is False  # owner barista != human barista


def test_clip_agreement_rows_skips_images_with_no_cache_entry():
    manifest = _manifest_with_attrs()
    labels = {"1::red apron": "barista"}

    assert clip_agreement_rows(manifest, labels, clip_scores={}) == []


def test_clip_agreement_rows_none_label_is_unscored():
    manifest = _manifest_with_attrs()
    labels = {"1::red apron": "none"}
    clip_scores = {"1": {"red apron": {"barista": 0.9, "cyclist": 0.1}}}

    rows = clip_agreement_rows(manifest, labels, clip_scores)

    assert rows[0]["scored"] is False
    assert rows[0]["correct"] is False


# --------------------------------------------------------------- prediction_agreement_rate

def test_agreement_rate_counts_matching_predicted_owner_on_shared_rows():
    attn_rows = [
        dict(prompt_id=1, attribute="red apron", predicted_owner="barista"),
        dict(prompt_id=1, attribute="yellow helmet", predicted_owner="cyclist"),
    ]
    clip_rows = [
        dict(prompt_id=1, attribute="red apron", predicted_owner="barista"),   # agrees
        dict(prompt_id=1, attribute="yellow helmet", predicted_owner="barista"),  # disagrees
    ]

    result = prediction_agreement_rate(attn_rows, clip_rows)

    assert result == dict(n=2, agreement_rate=0.5, n_agree=1)


def test_agreement_rate_is_only_over_the_intersection_of_rows():
    attn_rows = [dict(prompt_id=1, attribute="red apron", predicted_owner="barista"),
                dict(prompt_id=2, attribute="blue scarf", predicted_owner="chef")]
    clip_rows = [dict(prompt_id=1, attribute="red apron", predicted_owner="barista")]

    result = prediction_agreement_rate(attn_rows, clip_rows)

    assert result["n"] == 1
    assert result["agreement_rate"] == 1.0


def test_agreement_rate_handles_no_overlap():
    assert prediction_agreement_rate([], []) == dict(n=0, agreement_rate=None)
