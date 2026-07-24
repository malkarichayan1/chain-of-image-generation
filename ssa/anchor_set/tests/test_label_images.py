"""Tests for label_images.py: blind menu, input validation, resume. Run from inside
ssa/anchor_set/:  py -3 -m pytest tests/"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import anchor_common as ac
import label_images as li


def _manifest():
    return {"images": [
        {"prompt_id": 0, "n": 2, "prompt": "p0", "detected": True, "subjects": ["barista", "cyclist"],
         "image_path": "artifacts/images/p0.png",
         "attributes": [
             {"attribute": "red apron", "intended_subject": "barista", "predicted_owner": "cyclist", "model_scores": {}},
             {"attribute": "yellow helmet", "intended_subject": "cyclist", "predicted_owner": "cyclist", "model_scores": {}}]},
        {"prompt_id": 1, "n": 2, "prompt": "p1", "detected": False, "subjects": [], "image_path": "",
         "attributes": []},
    ]}


def test_build_menu_maps_numbers_and_sentinels():
    choice_map, menu = li.build_menu(["barista", "cyclist"])
    assert choice_map == {"1": "barista", "2": "cyclist", "n": ac.LABEL_NONE, "u": ac.LABEL_UNCLEAR}
    assert "barista" in menu and "none" in menu and "unclear" in menu


def test_prompt_one_rejects_then_accepts():
    choice_map, menu = li.build_menu(["barista", "cyclist"])
    answers = iter(["x", "9", "1"])  # two invalid, then valid
    img = {"prompt_id": 0, "prompt": "p0"}
    attr = {"attribute": "red apron"}
    got = li.prompt_one(img, attr, choice_map, menu, input_fn=lambda _p: next(answers))
    assert got == "barista"


def test_blindness_predicted_owner_never_shown(capsys):
    """The metric's guess must not leak into the prompt the annotator sees."""
    choice_map, menu = li.build_menu(["barista", "cyclist"])
    img = {"prompt_id": 0, "prompt": "a photo of a barista and a cyclist"}
    attr = {"attribute": "red apron", "predicted_owner": "cyclist"}
    li.prompt_one(img, attr, choice_map, menu, input_fn=lambda _p: "1")
    out = capsys.readouterr().out
    assert "red apron" in out              # the question is shown
    assert "predicted" not in out.lower()  # the answer is not
    # 'cyclist' legitimately appears as a selectable option, so we only assert the
    # prediction wording is absent, not the subject name.


def test_run_opens_images_under_current_artifacts_dir_not_baked_in_prefix(tmp_path, monkeypatch):
    """Regression test for the bug where --artifacts-dir artifacts_sdxl still opened SD1.5
    images: manifest.json's image_path is always shaped "artifacts/images/pN.png" (baked in
    by the GENERATING script, both SD1.5 and SDXL use that literal folder name on Kaggle),
    so run() must rebase it onto the CURRENT local ARTIFACTS_DIR (which may be
    "artifacts_sdxl" or any other name), not open the raw manifest string as-is."""
    sdxl_dir = tmp_path / "artifacts_sdxl"  # deliberately NOT named "artifacts"
    sdxl_dir.mkdir()
    manifest_path = sdxl_dir / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest()))  # image_path still says "artifacts/..."
    monkeypatch.setattr(li, "ARTIFACTS_DIR", sdxl_dir)
    monkeypatch.setattr(li, "MANIFEST_PATH", manifest_path)

    opened_paths = []
    monkeypatch.setattr(li, "_open_image", lambda p: opened_paths.append(p))

    li.run("tester", input_fn=lambda _p: "1")

    assert opened_paths, "expected at least one image open call"
    for p in opened_paths:
        assert str(sdxl_dir) in str(p), (
            f"opened {p!r}, which is not under the current artifacts_dir {sdxl_dir!r} "
            "-- the baked-in 'artifacts/' prefix leaked through unrebased")


def test_run_only_detected_and_resumes(tmp_path, monkeypatch):
    (tmp_path / "artifacts").mkdir()
    manifest_path = tmp_path / "artifacts" / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest()))
    monkeypatch.setattr(li, "ARTIFACTS_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(li, "MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(li, "_open_image", lambda _p: None)  # no GUI in tests

    # First pass: answer everything "1" (the first subject of each image).
    labels = li.run("tester", input_fn=lambda _p: "1")
    # Only the detected image (p0) has labelable attributes -> 2 judgments, none from p1.
    assert set(labels) == {ac.label_key(0, "red apron"), ac.label_key(0, "yellow helmet")}
    assert all(v == "barista" for v in labels.values())

    # Persisted to disk.
    saved = json.loads((tmp_path / "artifacts" / "labels_tester.json").read_text())
    assert saved == labels

    # Second pass: nothing pending -> input_fn must never be called.
    def _boom(_p):
        raise AssertionError("should not prompt when all judgments already recorded")
    again = li.run("tester", input_fn=_boom)
    assert again == labels
