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
    """`s` (shared) was added after the first labeling passes: `unclear` had been carrying
    both "I can't tell" and "it's on several people at once", which are different facts.
    `a` (absent) was added for the FLUX round: "the intended subject was never drawn" is a
    count-failure fact, distinct from `none` ("the subject exists, but this attribute
    never showed up")."""
    choice_map, menu = li.build_menu(["barista", "cyclist"])
    assert choice_map == {"1": "barista", "2": "cyclist", "n": ac.LABEL_NONE,
                          "u": ac.LABEL_UNCLEAR, "s": ac.LABEL_SHARED, "a": ac.LABEL_ABSENT}
    assert "barista" in menu and "none" in menu and "unclear" in menu and "shared" in menu
    assert "absent" in menu


def test_build_menu_without_layout_omits_position_tags():
    """No `layout` arg (the pre-FLUX call shape) must render exactly as before -- no
    bracketed position tag -- so every existing caller stays unaffected."""
    _choice_map, menu = li.build_menu(["barista", "cyclist"])
    assert "[" not in menu


def test_build_menu_with_layout_shows_position_and_gloss():
    """A subject whose name literally appears in its own prompt segment (barista) gets no
    gloss -- nothing to clarify. One that doesn't (cyclist, phrased "a man wearing a
    cycling jersey" in FLUX prompts) gets its prompt wording surfaced inline."""
    layout = [("left", None),
              ("right", "a man wearing a cycling jersey in a yellow bike helmet")]
    _choice_map, menu = li.build_menu(["barista", "cyclist"], layout=layout)
    assert "[left]" in menu and "[right]" in menu
    assert "a man wearing a cycling jersey" in menu
    # barista's own menu line must not carry a gloss parenthetical (its gloss is None)
    barista_line = next(line for line in menu.splitlines() if "1) barista" in line)
    assert "(" not in barista_line


def test_build_menu_redacts_gloss_matching_the_current_attribute():
    """The real leak this closes: without an `attribute` arg, cyclist's gloss spells out
    "in a yellow bike helmet" right next to the option while asking about the yellow
    helmet -- handing the annotator the answer. Passing the current attribute must trim it."""
    layout = [("left", None),
              ("right", "a man wearing a cycling jersey in a yellow bike helmet")]
    _choice_map, menu = li.build_menu(["barista", "cyclist"], layout=layout,
                                       attribute="yellow helmet")
    assert "a man wearing a cycling jersey" in menu
    assert "yellow" not in menu and "helmet" not in menu


def test_build_menu_does_not_redact_unrelated_attribute():
    """Asking about a different attribute (barista's red apron) must not touch cyclist's
    gloss -- there's nothing to redact, and the naming clarification is still useful."""
    layout = [("left", None),
              ("right", "a man wearing a cycling jersey in a yellow bike helmet")]
    _choice_map, menu = li.build_menu(["barista", "cyclist"], layout=layout,
                                       attribute="red apron")
    assert "a man wearing a cycling jersey in a yellow bike helmet" in menu


def test_relabel_requeues_only_rows_carrying_a_targeted_label():
    """Re-examining the existing `unclear` rows must not disturb settled judgments."""
    manifest = _manifest()
    labels = {ac.label_key(0, "red apron"): ac.LABEL_UNCLEAR,
              ac.label_key(0, "yellow helmet"): "cyclist"}

    pending = ac.pending_label_targets(manifest, labels, relabel={ac.LABEL_UNCLEAR})

    assert [a["attribute"] for _i, a in pending] == ["red apron"]


def test_relabel_defaults_to_off_so_a_normal_resume_is_unchanged():
    manifest = _manifest()
    labels = {ac.label_key(0, "red apron"): ac.LABEL_UNCLEAR,
              ac.label_key(0, "yellow helmet"): "cyclist"}

    assert ac.pending_label_targets(manifest, labels) == []


def test_run_relabel_pass_overwrites_the_targeted_rows(tmp_path, monkeypatch):
    (tmp_path / "artifacts").mkdir()
    manifest_path = tmp_path / "artifacts" / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest()))
    (tmp_path / "artifacts" / "labels_tester.json").write_text(json.dumps({
        ac.label_key(0, "red apron"): ac.LABEL_UNCLEAR,
        ac.label_key(0, "yellow helmet"): "cyclist",
    }))
    monkeypatch.setattr(li, "ARTIFACTS_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(li, "MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(li, "_open_image", lambda _p: None)

    labels = li.run("tester", input_fn=lambda _p: "s", relabel={ac.LABEL_UNCLEAR})

    assert labels[ac.label_key(0, "red apron")] == ac.LABEL_SHARED   # re-asked, upgraded
    assert labels[ac.label_key(0, "yellow helmet")] == "cyclist"     # untouched


def test_prompt_one_rejects_then_accepts():
    choice_map, menu = li.build_menu(["barista", "cyclist"])
    answers = iter(["x", "9", "1"])  # two invalid, then valid
    img = {"prompt_id": 0, "prompt": "p0"}
    attr = {"attribute": "red apron"}
    got = li.prompt_one(img, attr, choice_map, menu, input_fn=lambda _p: next(answers))
    assert got == "barista"


def test_prompt_one_accepts_absent():
    choice_map, menu = li.build_menu(["barista", "cyclist"])
    img = {"prompt_id": 0, "prompt": "p0"}
    attr = {"attribute": "red apron"}
    got = li.prompt_one(img, attr, choice_map, menu, input_fn=lambda _p: "a")
    assert got == ac.LABEL_ABSENT


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

    # "y" answers the per-image count check; "1" answers each attribute (first subject).
    li.run("tester", input_fn=lambda p: "y" if "count" in p else "1")

    assert opened_paths, "expected at least one image open call"
    for p in opened_paths:
        assert str(sdxl_dir) in str(p), (
            f"opened {p!r}, which is not under the current artifacts_dir {sdxl_dir!r} "
            "-- the baked-in 'artifacts/' prefix leaked through unrebased")


def test_prompt_count_clean_accepts_y_and_n():
    img = {"prompt_id": 0, "subjects": ["barista", "cyclist"], "prompt": "a barista and a cyclist"}
    assert li.prompt_count_clean(img, input_fn=lambda _p: "y") == ac.COUNT_CLEAN
    assert li.prompt_count_clean(img, input_fn=lambda _p: "n") == ac.COUNT_BROKEN


def test_prompt_count_clean_rejects_then_accepts():
    img = {"prompt_id": 0, "subjects": ["barista", "cyclist"], "prompt": "a barista and a cyclist"}
    answers = iter(["x", "yes", "n"])  # two invalid, then valid
    assert li.prompt_count_clean(img, input_fn=lambda _p: next(answers)) == ac.COUNT_BROKEN


def test_prompt_count_clean_prints_full_prompt_and_positions(capsys):
    """The count-check screen previously showed only the bare subject list, never the
    prompt itself -- an annotator had no text to check subject count/order against.
    When the prompt's positional markers parse cleanly, each subject is also tagged with
    its position, and any name/prompt-wording mismatch (cyclist / "a man wearing a cycling
    jersey") is surfaced as a note."""
    img = {"prompt_id": 12, "subjects": ["barista", "cyclist"],
           "prompt": "a photo of two people standing side by side, on the left a barista in "
                     "a red apron, on the right a man wearing a cycling jersey in a yellow "
                     "bike helmet"}
    li.prompt_count_clean(img, input_fn=lambda _p: "y")
    out = capsys.readouterr().out
    assert img["prompt"] in out
    assert "[left]" in out and "[right]" in out
    assert "a man wearing a cycling jersey" in out


def test_prompt_count_clean_falls_back_when_prompt_unparseable(capsys):
    """A prompt with no recognizable positional markers must still list the subjects --
    just without position tags -- rather than error or silently drop the subject list."""
    img = {"prompt_id": 0, "subjects": ["barista", "cyclist"], "prompt": "a barista and a cyclist"}
    li.prompt_count_clean(img, input_fn=lambda _p: "y")
    out = capsys.readouterr().out
    assert "barista" in out and "cyclist" in out
    assert "(left -> right)" not in out  # the position-tagged line is never printed


def test_run_asks_count_clean_once_per_image_and_persists(tmp_path, monkeypatch):
    (tmp_path / "artifacts").mkdir()
    manifest_path = tmp_path / "artifacts" / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest()))
    monkeypatch.setattr(li, "ARTIFACTS_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(li, "MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(li, "_open_image", lambda _p: None)

    # "n" is a valid answer to BOTH question types (count-broken / LABEL_NONE), so a single
    # constant input_fn drives the whole run without ambiguity: p0's count-clean question
    # (asked once, before whichever of its 2 attributes the shuffle visits first) resolves
    # to COUNT_BROKEN, and both attribute questions resolve to LABEL_NONE.
    li.run("tester", input_fn=lambda _p: "n")

    counts = json.loads((tmp_path / "artifacts" / "counts_tester.json").read_text())
    assert counts == {ac.count_key(0): ac.COUNT_BROKEN}

    # Resume: nothing pending (labels AND counts already recorded) -> never prompts again.
    def _boom(_p):
        raise AssertionError("should not prompt when count is already recorded")
    li.run("tester", input_fn=_boom)


def test_run_prints_guidelines_banner(tmp_path, monkeypatch, capsys):
    """The labeling guidelines (Count-Clean/Count-Broken, and the Present/Wrong Subject/
    Missing/Shared/Unclear/Subject-Absent attribute taxonomy, revised 2026-07-30 for the
    FLUX round) must be shown every time run() is invoked -- including a no-op resume --
    so every annotator sees the reminder every session, not just once via a message."""
    (tmp_path / "artifacts").mkdir()
    manifest_path = tmp_path / "artifacts" / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest()))
    monkeypatch.setattr(li, "ARTIFACTS_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(li, "MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(li, "_open_image", lambda _p: None)

    li.run("tester", input_fn=lambda _p: "n")  # answer everything, drains all targets
    out = capsys.readouterr().out
    for phrase in ("Count-Clean", "Count-Broken", "Present", "Missing", "Shared", "Unclear",
                   "Wrong Subject", "Observed Owner", "Subject Absent", "Not Evaluable",
                   "background"):
        assert phrase in out, f"banner missing {phrase!r}"

    # No-op resume (nothing pending) must STILL show the banner.
    def _boom(_p):
        raise AssertionError("should not prompt when everything is already recorded")
    li.run("tester", input_fn=_boom)
    out2 = capsys.readouterr().out
    assert "Count-Clean" in out2


def _manifest_with_positions():
    return {"images": [
        {"prompt_id": 0, "n": 2, "detected": True, "subjects": ["barista", "cyclist"],
         "prompt": "a photo of two people standing side by side, on the left a barista in a "
                   "red apron, on the right a man wearing a cycling jersey in a yellow bike "
                   "helmet",
         "image_path": "artifacts/images/p0.png",
         "attributes": [
             {"attribute": "red apron", "intended_subject": "barista",
              "predicted_owner": "cyclist", "model_scores": {}},
             {"attribute": "yellow helmet", "intended_subject": "cyclist",
              "predicted_owner": "cyclist", "model_scores": {}}]},
    ]}


def test_run_wires_prompt_layout_into_the_attribute_menu(tmp_path, monkeypatch, capsys):
    """End-to-end: run() must compute the layout from each image's own prompt and pass it
    into build_menu, so the position/gloss tags actually reach the annotator's screen, not
    just the unit-tested build_menu() call in isolation."""
    (tmp_path / "artifacts").mkdir()
    manifest_path = tmp_path / "artifacts" / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest_with_positions()))
    monkeypatch.setattr(li, "ARTIFACTS_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(li, "MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(li, "_open_image", lambda _p: None)

    li.run("tester", input_fn=lambda p: "y" if "count" in p else "1")
    out = capsys.readouterr().out
    assert "[left]" in out and "[right]" in out
    assert "a man wearing a cycling jersey" in out
    # Anti-leak regression: the question line itself legitimately says "yellow helmet"
    # (that's the question, not a leak) -- but cyclist's GLOSS in that question's own menu
    # must not repeat it a second time via the untrimmed "...in a yellow bike helmet"
    # clause, which would spell out the answer right next to the option.
    menu_block = out[out.index("Which subject is wearing/holding the 'yellow helmet'"):]
    menu_block = menu_block.split("\n\n", 1)[0]
    assert "a man wearing a cycling jersey" in menu_block
    assert "yellow bike helmet" not in menu_block


def test_run_only_detected_and_resumes(tmp_path, monkeypatch):
    (tmp_path / "artifacts").mkdir()
    manifest_path = tmp_path / "artifacts" / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest()))
    monkeypatch.setattr(li, "ARTIFACTS_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(li, "MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(li, "_open_image", lambda _p: None)  # no GUI in tests

    # First pass: "y" to each per-image count check, "1" (first subject) to each attribute.
    labels = li.run("tester", input_fn=lambda p: "y" if "count" in p else "1")
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
