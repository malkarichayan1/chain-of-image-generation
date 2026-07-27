"""Tests for exp2_window_ablation.py (Part A, Experiment 2): does early-window attention beat
full-trajectory averaging? Requires manifest.json's `model_scores_full`/`predicted_owner_full`
fields (from the additive generate_anchor_images_sdxl.py patch) -- must degrade gracefully, not
crash, when a manifest predates that patch. Run from inside ssa/anchor_set/:
    py -3 -m pytest tests/"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import anchor_common as ac
import exp2_window_ablation as exp2


def _manifest_with_full(num_items=4):
    images = []
    for i in range(num_items):
        s1, s2 = f"subj{i}a", f"subj{i}b"
        images.append({
            "prompt_id": i, "n": 2, "prompt": f"p{i}", "detected": True, "subjects": [s1, s2],
            "attributes": [{
                "attribute": "attr", "intended_subject": s1,
                "predicted_owner": s1, "model_scores": {s1: 0.9, s2: 0.1},
                "predicted_owner_full": s2 if i % 2 == 0 else s1,
                "model_scores_full": {s1: 0.4, s2: 0.6} if i % 2 == 0 else {s1: 0.6, s2: 0.4},
            }],
        })
    return {"images": images}


def _manifest_without_full():
    return {"images": [
        {"prompt_id": 0, "n": 2, "prompt": "p0", "detected": True, "subjects": ["barista", "cyclist"],
         "attributes": [
             {"attribute": "red apron", "intended_subject": "barista", "predicted_owner": "barista",
              "model_scores": {"barista": 0.6, "cyclist": 0.4}}]},
    ]}


def _labels_intended(manifest):
    return {ac.label_key(img["prompt_id"], img["attributes"][0]["attribute"]): img["attributes"][0]["intended_subject"]
            for img in manifest["images"]}


# --------------------------------------------------------------------------- availability

def test_is_full_trajectory_available_false_when_field_missing():
    assert exp2.is_full_trajectory_available(_manifest_without_full()) is False


def test_is_full_trajectory_available_true_when_every_attribute_has_it():
    assert exp2.is_full_trajectory_available(_manifest_with_full()) is True


def test_is_full_trajectory_available_false_when_only_some_attributes_have_it():
    manifest = _manifest_with_full(num_items=2)
    manifest["images"][0]["attributes"][0]["predicted_owner_full"] = None
    assert exp2.is_full_trajectory_available(manifest) is False


# --------------------------------------------------------------------------- join + report

def test_join_window_rows_adds_full_columns():
    manifest = _manifest_with_full(num_items=1)
    labels = _labels_intended(manifest)
    rows = ac.build_agreement_rows(manifest, labels)
    full_by_key = exp2.full_scores_by_key(manifest)
    joined = exp2.join_window_rows(rows, full_by_key)
    assert joined[0]["predicted_owner_full"] == "subj0b"  # i=0 -> even -> s2
    assert joined[0]["correct_full"] is False  # intended subject is subj0a


def test_window_accuracy_report_computes_early_and_full_per_stratum():
    manifest = _manifest_with_full(num_items=4)
    labels = _labels_intended(manifest)  # human always names the intended (early-correct) subject
    rows = ac.build_agreement_rows(manifest, labels)
    joined = exp2.join_window_rows(rows, exp2.full_scores_by_key(manifest))
    report = exp2.window_accuracy_report(joined)
    s = report[2]
    assert s["n_scored"] == 4
    assert s["early_accuracy"] == pytest.approx(1.0)  # predicted_owner always == intended
    assert s["full_accuracy"] == pytest.approx(0.5)   # half the items flip to the wrong subject
    assert s["chance"] == pytest.approx(0.5)


def test_mcnemar_early_vs_full_report_shape():
    manifest = _manifest_with_full(num_items=4)
    labels = _labels_intended(manifest)
    rows = ac.build_agreement_rows(manifest, labels)
    joined = exp2.join_window_rows(rows, exp2.full_scores_by_key(manifest))
    report = exp2.mcnemar_early_vs_full(joined)
    assert report["n"] == 4
    assert report["early_only_correct"] == 2  # the 2 items where full flips away from correct
    assert report["full_only_correct"] == 0


# --------------------------------------------------------------------------- CLI

def test_main_reports_unavailable_without_crashing(tmp_path, monkeypatch, capsys):
    import json
    art = tmp_path / "artifacts"
    art.mkdir()
    manifest = _manifest_without_full()
    (art / "manifest.json").write_text(json.dumps(manifest))
    (art / "labels_t.json").write_text(json.dumps(_labels_intended(manifest)))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["exp2_window_ablation.py", "--artifacts-dir", "artifacts", "--annotator", "t"])
    exp2.main()  # must not raise
    out = capsys.readouterr().out
    assert "unavailable" in out.lower()


def test_main_runs_when_full_scores_present(tmp_path, monkeypatch, capsys):
    import json
    art = tmp_path / "artifacts"
    art.mkdir()
    manifest = _manifest_with_full(num_items=4)
    (art / "manifest.json").write_text(json.dumps(manifest))
    (art / "labels_t.json").write_text(json.dumps(_labels_intended(manifest)))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["exp2_window_ablation.py", "--artifacts-dir", "artifacts", "--annotator", "t"])
    exp2.main()
    out = capsys.readouterr().out
    assert "unavailable" not in out.lower()
    assert "n=2" in out
