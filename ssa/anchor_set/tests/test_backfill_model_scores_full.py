"""
Tests for backfill_model_scores_full.py: copies Experiment 2's model_scores_full/
predicted_owner_full from a PIN_SEEDS_FROM_MANIFEST rerun onto an already-labeled manifest,
refusing when the rerun didn't faithfully reproduce the original. Run from inside
ssa/anchor_set/: py -3 -m pytest tests/
"""
import json

import pytest

from backfill_model_scores_full import (
    backfill_model_scores_full,
    index_attributes,
    main,
    verify_rerun_matches,
)


def _attr(attribute, intended_subject, predicted_owner, model_scores,
          predicted_owner_full=None, model_scores_full=None):
    return dict(attribute=attribute, intended_subject=intended_subject,
                predicted_owner=predicted_owner, model_scores=model_scores,
                predicted_owner_full=predicted_owner_full, model_scores_full=model_scores_full)


def _image(prompt_id, attributes, detected=True):
    return dict(prompt_id=prompt_id, n=2, prompt="x", subjects=["a", "b"], seed=42,
                detected=detected, num_people_detected=2, image_path="artifacts/images/x.png",
                attributes=attributes)


def _original():
    return {"images": [
        _image(0, [_attr("red apron", "barista", "barista", {"barista": 0.6, "cyclist": 0.4})]),
        _image(1, [_attr("yellow helmet", "cyclist", "cyclist", {"barista": 0.3, "cyclist": 0.7})]),
    ]}


def _rerun_matching():
    return {"images": [
        _image(0, [_attr("red apron", "barista", "barista", {"barista": 0.6, "cyclist": 0.4},
                          predicted_owner_full="barista",
                          model_scores_full={"barista": 0.55, "cyclist": 0.45})]),
        _image(1, [_attr("yellow helmet", "cyclist", "cyclist", {"barista": 0.3, "cyclist": 0.7},
                          predicted_owner_full="cyclist",
                          model_scores_full={"barista": 0.2, "cyclist": 0.8})]),
    ]}


# --------------------------------------------------------------------------- index_attributes

def test_index_attributes_only_includes_detected_images():
    manifest = {"images": [
        _image(0, [_attr("red apron", "barista", "barista", {"barista": 1.0})]),
        _image(1, [_attr("yellow helmet", "cyclist", "cyclist", {"cyclist": 1.0})], detected=False),
    ]}
    indexed = index_attributes(manifest)
    assert list(indexed.keys()) == [(0, "red apron")]


def test_index_attributes_keys_by_prompt_id_and_attribute():
    indexed = index_attributes(_original())
    assert set(indexed.keys()) == {(0, "red apron"), (1, "yellow helmet")}


# --------------------------------------------------------------------------- verify_rerun_matches

def test_verify_rerun_matches_no_mismatches_when_identical():
    assert verify_rerun_matches(_original(), _rerun_matching()) == []


def test_verify_rerun_matches_flags_predicted_owner_change():
    rerun = _rerun_matching()
    rerun["images"][0]["attributes"][0]["predicted_owner"] = "cyclist"
    mismatches = verify_rerun_matches(_original(), rerun)
    assert len(mismatches) == 1
    assert "predicted_owner changed" in mismatches[0]


def test_verify_rerun_matches_flags_score_drift_beyond_tolerance():
    rerun = _rerun_matching()
    rerun["images"][0]["attributes"][0]["model_scores"]["barista"] = 0.9
    mismatches = verify_rerun_matches(_original(), rerun)
    assert len(mismatches) == 1
    assert "model_scores" in mismatches[0]


def test_verify_rerun_matches_ignores_score_drift_within_tolerance():
    rerun = _rerun_matching()
    rerun["images"][0]["attributes"][0]["model_scores"]["barista"] = 0.6005
    assert verify_rerun_matches(_original(), rerun) == []


def test_verify_rerun_matches_skips_keys_only_in_rerun():
    original = _original()
    del original["images"][1]  # original never had prompt_id=1 at all
    assert verify_rerun_matches(original, _rerun_matching()) == []


def test_verify_rerun_matches_skips_keys_only_in_original():
    rerun = _rerun_matching()
    del rerun["images"][1]  # rerun didn't cover prompt_id=1 (e.g. GROWTH_PROMPT_IDS left scoped)
    assert verify_rerun_matches(_original(), rerun) == []


# --------------------------------------------------------------------------- backfill_model_scores_full

def test_backfill_copies_full_fields_onto_original():
    merged = backfill_model_scores_full(_original(), _rerun_matching())
    attr0 = merged["images"][0]["attributes"][0]
    assert attr0["predicted_owner_full"] == "barista"
    assert attr0["model_scores_full"] == {"barista": 0.55, "cyclist": 0.45}


def test_backfill_preserves_non_full_fields_untouched():
    merged = backfill_model_scores_full(_original(), _rerun_matching())
    attr0 = merged["images"][0]["attributes"][0]
    assert attr0["predicted_owner"] == "barista"
    assert attr0["model_scores"] == {"barista": 0.6, "cyclist": 0.4}
    assert merged["images"][0]["seed"] == 42


def test_backfill_sets_none_when_attribute_missing_from_rerun():
    rerun = _rerun_matching()
    del rerun["images"][1]
    merged = backfill_model_scores_full(_original(), rerun)
    attr1 = merged["images"][1]["attributes"][0]
    assert attr1["predicted_owner_full"] is None
    assert attr1["model_scores_full"] is None


def test_backfill_does_not_mutate_input_original():
    original = _original()
    snapshot = json.loads(json.dumps(original))
    backfill_model_scores_full(original, _rerun_matching())
    assert original == snapshot


def test_backfill_skips_undetected_images():
    original = _original()
    original["images"][0]["detected"] = False
    merged = backfill_model_scores_full(original, _rerun_matching())
    assert merged["images"][0]["attributes"][0].get("predicted_owner_full") is None


# --------------------------------------------------------------------------- CLI

def test_main_refuses_and_exits_nonzero_on_mismatch(tmp_path, monkeypatch, capsys):
    original_path = tmp_path / "original.json"
    rerun_path = tmp_path / "rerun.json"
    out_path = tmp_path / "out.json"
    original_path.write_text(json.dumps(_original()))
    rerun = _rerun_matching()
    rerun["images"][0]["attributes"][0]["predicted_owner"] = "cyclist"
    rerun_path.write_text(json.dumps(rerun))

    monkeypatch.setattr("sys.argv",
                        ["backfill_model_scores_full.py", str(original_path), str(rerun_path), str(out_path)])
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code != 0
    assert "REFUSING" in capsys.readouterr().out
    assert not out_path.exists()


def test_main_writes_backfilled_manifest_on_success(tmp_path, monkeypatch, capsys):
    original_path = tmp_path / "original.json"
    rerun_path = tmp_path / "rerun.json"
    out_path = tmp_path / "out.json"
    original_path.write_text(json.dumps(_original()))
    rerun_path.write_text(json.dumps(_rerun_matching()))

    monkeypatch.setattr("sys.argv",
                        ["backfill_model_scores_full.py", str(original_path), str(rerun_path), str(out_path)])
    main()

    out = capsys.readouterr().out
    assert "Backfilled" in out
    written = json.loads(out_path.read_text())
    assert written["images"][0]["attributes"][0]["predicted_owner_full"] == "barista"
