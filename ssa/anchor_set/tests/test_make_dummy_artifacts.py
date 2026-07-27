"""Tests for make_dummy_artifacts.py: the synthetic artifacts_dummy/ fixture all five Part A
experiment scripts run against until Workstream 2's real, second-annotator-validated labels
land. Run from inside ssa/anchor_set/:  py -3 -m pytest tests/"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import anchor_common as ac
import make_dummy_artifacts as mda


def test_build_dummy_artifacts_matches_configured_stratum_sizes():
    data = mda.build_dummy_artifacts(seed=0)
    images = data["manifest"]["images"]
    from collections import Counter
    by_n = Counter(img["n"] for img in images)
    assert dict(by_n) == mda.ITEMS_PER_STRATUM


def test_every_image_has_at_least_two_stratum_mates():
    """exp3_attention_scramble.py's cross-item scramble needs >= 2 items per stratum -- must
    not be accidentally untested against this fixture."""
    data = mda.build_dummy_artifacts(seed=0)
    from collections import Counter
    by_n = Counter(img["n"] for img in data["manifest"]["images"])
    assert all(count >= 2 for count in by_n.values())


def test_every_attribute_has_full_trajectory_fields():
    """exp2_window_ablation.py requires model_scores_full/predicted_owner_full on every
    detected attribute to report anything other than 'unavailable'."""
    data = mda.build_dummy_artifacts(seed=0)
    for img in data["manifest"]["images"]:
        for attr in img["attributes"]:
            assert attr["predicted_owner_full"] is not None
            assert attr["predicted_owner_full"] in img["subjects"]
            assert set(attr["model_scores_full"]) == set(img["subjects"])


def test_predicted_owner_is_argmax_of_its_own_model_scores():
    data = mda.build_dummy_artifacts(seed=1)
    for img in data["manifest"]["images"]:
        for attr in img["attributes"]:
            assert attr["predicted_owner"] == max(attr["model_scores"], key=attr["model_scores"].get)
            assert attr["predicted_owner_full"] == max(attr["model_scores_full"], key=attr["model_scores_full"].get)


def test_build_dummy_artifacts_is_deterministic_given_a_seed():
    a = mda.build_dummy_artifacts(seed=42)
    b = mda.build_dummy_artifacts(seed=42)
    assert a == b


def test_build_dummy_artifacts_differs_across_seeds():
    a = mda.build_dummy_artifacts(seed=1)
    b = mda.build_dummy_artifacts(seed=2)
    assert a["labels"] != b["labels"]


def test_labels_cover_every_attribute_with_a_valid_value():
    data = mda.build_dummy_artifacts(seed=0)
    for img in data["manifest"]["images"]:
        for attr in img["attributes"]:
            key = ac.label_key(img["prompt_id"], attr["attribute"])
            assert key in data["labels"]
            value = data["labels"][key]
            assert value in img["subjects"] or value in ac.NON_SUBJECT_LABELS


def test_counts_cover_every_image_with_clean_or_broken():
    data = mda.build_dummy_artifacts(seed=0)
    for img in data["manifest"]["images"]:
        key = ac.count_key(img["prompt_id"])
        assert data["counts"][key] in (ac.COUNT_CLEAN, ac.COUNT_BROKEN)


def test_build_agreement_rows_runs_cleanly_against_the_dummy_fixture():
    """End-to-end sanity: the fixture must actually be consumable by the real anchor_common
    join, not just internally self-consistent."""
    data = mda.build_dummy_artifacts(seed=0)
    rows = ac.build_agreement_rows(data["manifest"], data["labels"], counts=data["counts"])
    assert len(rows) > 0
    assert any(r["scored"] for r in rows)


def test_write_dummy_artifacts_writes_three_valid_json_files(tmp_path):
    out = tmp_path / "artifacts_dummy"
    mda.write_dummy_artifacts(out, seed=0)
    manifest = json.loads((out / "manifest.json").read_text())
    labels = json.loads((out / "labels_dummy.json").read_text())
    counts = json.loads((out / "counts_dummy.json").read_text())
    expected = mda.build_dummy_artifacts(seed=0)
    assert manifest == expected["manifest"]
    assert labels == expected["labels"]
    assert counts == expected["counts"]
