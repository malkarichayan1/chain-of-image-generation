"""Unit tests for anchor_common.py (pure logic) + the real prompt_specs.json / embedded
ANCHOR_PROMPTS drift guard. Run from inside ssa/anchor_set/:  py -3 -m pytest tests/"""
import ast
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make anchor_common importable

import anchor_common as ac

PKG = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- specs

def test_real_prompt_specs_are_valid_and_stratified():
    specs = ac.load_specs(PKG / "prompt_specs.json")
    ac.validate_specs(specs)  # must not raise
    by_n = {n: sum(1 for s in specs if s["n"] == n) for n in (2, 3, 4)}
    # 18 original + 6-prompt 2026-07-24 growth batch (ids 0-23) + 85-image 2026-07-25
    # growth batch (ids 100-184, build_growth_specs.py, +42/+25/+18 per stratum) + 28-image
    # n=4 backfill (ids 200-227, build_n4_backfill_specs -- the 100-184 batch's n=4 stratum
    # hit only 22.2% detection, far short of target; see that function's docstring).
    assert by_n == {2: 50, 3: 33, 4: 54}


@pytest.mark.parametrize("script_name", ["generate_anchor_images.py", "generate_anchor_images_sdxl.py"])
def test_embedded_anchor_prompts_match_prompt_specs_json(script_name):
    """Each single-file Kaggle kernel (SD1.5 and SDXL) embeds its own ANCHOR_PROMPTS
    literal; both must stay bit-identical to the canonical prompt_specs.json, or the two
    runs would silently describe different prompts and the A/B comparison would be invalid."""
    src = (PKG / script_name).read_text(encoding="utf-8")
    seg = None
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and any(
                getattr(t, "id", None) == "ANCHOR_PROMPTS" for t in node.targets):
            seg = ast.get_source_segment(src, node)
    assert seg is not None, f"ANCHOR_PROMPTS assignment not found in {script_name}"
    ns: dict = {}
    exec(seg, ns)  # only dict()/tuple/list/str/int literals
    embedded = [dict(id=p["id"], n=p["n"], prompt=p["prompt"],
                     pairs=[list(x) for x in p["pairs"]]) for p in ns["ANCHOR_PROMPTS"]]
    spec = [dict(id=p["id"], n=p["n"], prompt=p["prompt"],
                 pairs=[list(x) for x in p["pairs"]])
            for p in json.loads((PKG / "prompt_specs.json").read_text())["prompts"]]
    assert embedded == spec


def test_validate_rejects_repeated_subject():
    bad = [{"id": 0, "n": 2, "prompt": "a photo of a chef with a red apron and a chef",
            "pairs": [["chef", "red apron"], ["chef", "yellow helmet"]]},
           {"id": 1, "n": 3, "prompt": "", "pairs": []}, {"id": 2, "n": 4, "prompt": "", "pairs": []}]
    with pytest.raises(ValueError, match="repeated subject"):
        ac.validate_specs(bad)


def test_validate_rejects_attribute_not_in_prompt():
    bad = [{"id": 0, "n": 2, "prompt": "a photo of a chef and a farmer",
            "pairs": [["chef", "white hat"], ["farmer", "shovel"]]},
           {"id": 1, "n": 3, "prompt": "", "pairs": []}, {"id": 2, "n": 4, "prompt": "", "pairs": []}]
    with pytest.raises(ValueError, match="not a substring"):
        ac.validate_specs(bad)


def test_validate_rejects_missing_stratum():
    only_two = [{"id": 0, "n": 2, "prompt": "a chef", "pairs": [["chef", "chef"]]}]
    with pytest.raises(ValueError, match="n=3"):
        ac.validate_specs(only_two)


# --------------------------------------------------------------------------- prediction

def _map_hot_on(region):
    """512x512 map that is 1.0 inside `region`=(x0,y0,x1,y1), else 0."""
    m = np.zeros((512, 512), dtype=np.float32)
    x0, y0, x1, y1 = region
    m[y0:y1, x0:x1] = 1.0
    return m


def test_predicted_owner_picks_box_with_most_mass():
    boxes = {"left": [0, 0, 100, 512], "right": [400, 0, 500, 512]}
    attn = _map_hot_on((410, 0, 490, 512))  # mass lands in the right box
    owner, scores = ac.predicted_owner_from_attention(attn, boxes)
    assert owner == "right"
    assert scores["right"] > scores["left"]


def test_mean_mass_in_box_clamps_and_handles_offmap():
    attn = np.ones((512, 512), dtype=np.float32)
    assert ac.mean_mass_in_box(attn, [0, 0, 10, 10]) == pytest.approx(1.0)
    assert ac.mean_mass_in_box(attn, [-50, -50, 5, 5]) == pytest.approx(1.0)  # clamped
    assert ac.mean_mass_in_box(attn, [600, 600, 700, 700]) == 0.0  # fully off-map
    assert ac.mean_mass_in_box(attn, [10, 10, 10, 10]) == 0.0  # empty


def test_predicted_owner_empty_boxes_raises():
    with pytest.raises(ValueError):
        ac.predicted_owner_from_attention(np.zeros((8, 8), np.float32), {})


def test_predicted_owner_tie_breaks_by_insertion_order():
    boxes = {"first": [0, 0, 100, 100], "second": [200, 200, 300, 300]}
    flat = np.ones((512, 512), dtype=np.float32)  # equal mass -> tie
    owner, _ = ac.predicted_owner_from_attention(flat, boxes)
    assert owner == "first"


# --------------------------------------------------------------------------- labels I/O

def test_resolve_image_path_rebases_baked_in_artifacts_prefix(tmp_path):
    """manifest.json always stores image_path as "artifacts/images/pN.png" regardless of
    which run generated it (SD1.5 and SDXL both use that literal folder name on Kaggle).
    resolve_image_path must strip that prefix and rejoin onto whatever local artifacts_dir
    is actually in use -- e.g. an "artifacts_sdxl" folder that doesn't literally contain a
    nested "artifacts" subfolder at all."""
    local_dir = tmp_path / "artifacts_sdxl"
    result = ac.resolve_image_path(local_dir, "artifacts/images/p6.png")
    assert result == (local_dir / "images" / "p6.png").resolve()


def test_resolve_image_path_handles_path_without_prefix(tmp_path):
    local_dir = tmp_path / "artifacts"
    result = ac.resolve_image_path(local_dir, "images/p0.png")
    assert result == (local_dir / "images" / "p0.png").resolve()


def test_label_roundtrip_and_resume(tmp_path):
    p = tmp_path / "labels_x.json"
    assert ac.load_labels(p) == {}
    labels = {ac.label_key(0, "red apron"): "barista"}
    ac.save_labels(p, labels)
    assert ac.load_labels(p) == labels


# --------------------------------------------------------------------------- agreement

def _manifest():
    return {"images": [
        {"prompt_id": 0, "n": 2, "prompt": "p0", "detected": True, "subjects": ["barista", "cyclist"],
         "attributes": [
             {"attribute": "red apron", "intended_subject": "barista", "predicted_owner": "barista", "model_scores": {}},
             {"attribute": "yellow helmet", "intended_subject": "cyclist", "predicted_owner": "barista", "model_scores": {}}]},
        {"prompt_id": 1, "n": 3, "prompt": "p1", "detected": True, "subjects": ["chef", "farmer", "nurse"],
         "attributes": [
             {"attribute": "white hat", "intended_subject": "chef", "predicted_owner": "chef", "model_scores": {}}]},
        {"prompt_id": 2, "n": 4, "prompt": "p2", "detected": False, "subjects": [], "attributes": []},
    ]}


def test_build_rows_skips_undetected_and_unlabeled():
    labels = {ac.label_key(0, "red apron"): "barista"}  # only one of the labelable attrs
    rows = ac.build_agreement_rows(_manifest(), labels)
    assert len(rows) == 1
    assert rows[0]["prompt_id"] == 0 and rows[0]["correct"] is True


def test_none_and_unclear_excluded_from_denominator():
    labels = {
        ac.label_key(0, "red apron"): "barista",        # correct
        ac.label_key(0, "yellow helmet"): "cyclist",    # predicted barista -> incorrect
        ac.label_key(1, "white hat"): ac.LABEL_UNCLEAR,  # excluded
    }
    summary = ac.summarize_agreement(ac.build_agreement_rows(_manifest(), labels))
    overall = summary["overall"]
    assert overall["n_labeled"] == 3
    assert overall["n_scored"] == 2          # unclear dropped
    assert overall["n_correct"] == 1
    assert overall["accuracy"] == pytest.approx(0.5)
    n2 = summary["by_stratum"][2]
    assert n2["chance"] == pytest.approx(0.5) and n2["n_scored"] == 2
    n3 = summary["by_stratum"][3]
    assert n3["n_scored"] == 0 and n3["accuracy"] is None  # only row was unclear


def test_chance_baseline():
    assert ac.chance_baseline(2) == 0.5
    assert ac.chance_baseline(4) == 0.25


def test_absent_excluded_from_denominator_same_as_none_and_unclear():
    """`absent` (Subject Absent / Not Evaluable) marks a count-broken-caused gap, not a
    binding outcome -- it must be excluded from scoring exactly like `none`/`unclear`, so
    adding it never moves an already-published accuracy number."""
    labels = {
        ac.label_key(0, "red apron"): "barista",       # correct
        ac.label_key(0, "yellow helmet"): ac.LABEL_ABSENT,  # excluded
        ac.label_key(1, "white hat"): ac.LABEL_UNCLEAR,     # excluded
    }
    summary = ac.summarize_agreement(ac.build_agreement_rows(_manifest(), labels))
    overall = summary["overall"]
    assert overall["n_labeled"] == 3
    assert overall["n_scored"] == 1
    assert overall["n_correct"] == 1


def test_label_absent_is_in_both_sentinel_sets():
    assert ac.LABEL_ABSENT in ac.NON_SUBJECT_LABELS
    assert ac.LABEL_ABSENT in ac.MISSING_DATA_LABELS
    assert ac.LABEL_ABSENT not in {ac.LABEL_NONE, ac.LABEL_UNCLEAR, ac.LABEL_SHARED}


# --------------------------------------------------------------------------- prompt layout

def test_parse_prompt_layout_n2():
    prompt = ("a photo of two people standing side by side, on the left a barista in a red "
              "apron, on the right a man wearing a cycling jersey in a yellow bike helmet")
    layout = ac.parse_prompt_layout(prompt, ["barista", "cyclist"])
    assert layout is not None
    assert layout[0] == ("left", None)  # "barista" appears in its own segment -> no gloss
    position, gloss = layout[1]
    assert position == "right"
    assert gloss == "a man wearing a cycling jersey in a yellow bike helmet"


def test_parse_prompt_layout_n3():
    prompt = ("a photo of three people standing side by side, on the left a barista in a red "
              "apron, in the middle a chef in a tall white chef hat, on the right a farmer "
              "holding a wooden shovel")
    layout = ac.parse_prompt_layout(prompt, ["barista", "chef", "farmer"])
    assert [p for p, _g in layout] == ["left", "middle", "right"]
    assert all(g is None for _p, g in layout)  # all three subject names appear verbatim


def test_parse_prompt_layout_n4():
    prompt = ("a photo of four people standing side by side, on the far left a barista in a "
              "red apron, on the center-left a man wearing a cycling jersey in a yellow bike "
              "helmet, on the center-right a chef in a tall white chef hat, on the far right "
              "a farmer holding a wooden shovel")
    layout = ac.parse_prompt_layout(prompt, ["barista", "cyclist", "chef", "farmer"])
    assert [p for p, _g in layout] == ["far left", "center-left", "center-right", "far right"]
    assert layout[1][1] == "a man wearing a cycling jersey in a yellow bike helmet"


def test_parse_prompt_layout_returns_none_on_marker_count_mismatch():
    """Never guess a wrong position: if the marker count doesn't match len(subjects) --
    a malformed or hand-edited prompt -- callers must fall back to plain (position-free)
    display rather than risk a mislabeled position."""
    prompt = "a photo of two people standing side by side, on the left a barista in a red apron"
    assert ac.parse_prompt_layout(prompt, ["barista", "cyclist"]) is None


def test_parse_prompt_layout_returns_none_when_no_markers_present():
    assert ac.parse_prompt_layout("a barista and a chef", ["barista", "chef"]) is None


# --------------------------------------------------------------------------- gloss redaction

def test_redact_attribute_clause_strips_the_matching_attribute_clause():
    """The real leak this guards against: cyclist's naming gloss is "a man wearing a
    cycling jersey in a yellow bike helmet" -- when the annotator is being asked about the
    'yellow helmet' attribute, showing that gloss unredacted spells out the answer right
    next to the menu option it belongs to. Only the identity-establishing clause ("a man
    wearing a cycling jersey") should survive; the attribute clause must be cut."""
    gloss = "a man wearing a cycling jersey in a yellow bike helmet"
    trimmed = ac.redact_attribute_clause(gloss, "yellow helmet")
    assert trimmed == "a man wearing a cycling jersey"
    assert "yellow" not in trimmed and "helmet" not in trimmed


def test_redact_attribute_clause_no_overlap_returns_gloss_unchanged():
    """Asking about a DIFFERENT attribute (e.g. the barista's red apron) must not touch
    cyclist's gloss at all -- there's nothing to redact, and the identity clarification is
    still useful context for that question too."""
    gloss = "a man wearing a cycling jersey in a yellow bike helmet"
    assert ac.redact_attribute_clause(gloss, "red apron") == gloss


def test_redact_attribute_clause_never_returns_empty():
    """If redaction would strip the ENTIRE gloss (a pathological/future prompt shape),
    fall back to the untrimmed gloss rather than show a blank clarification."""
    gloss = "yellow helmet"
    assert ac.redact_attribute_clause(gloss, "yellow helmet") == gloss


# --------------------------------------------------------------------------- count-clean

def test_pending_count_targets_only_detected_and_uncounted():
    manifest = _manifest()  # p0 detected, p1 n=3 detected, p2 undetected
    counts = {ac.count_key(0): ac.COUNT_CLEAN}
    pending = ac.pending_count_targets(manifest, counts)
    assert [img["prompt_id"] for img in pending] == [1]  # p0 already counted, p2 undetected


def test_pending_count_targets_empty_when_all_counted():
    manifest = _manifest()
    counts = {ac.count_key(0): ac.COUNT_CLEAN, ac.count_key(1): ac.COUNT_BROKEN}
    assert ac.pending_count_targets(manifest, counts) == []


def test_build_rows_excludes_count_broken_images_from_scoring():
    """A count-broken image must be excluded from the binding accuracy denominator
    regardless of what the human said about any individual attribute -- count-broken
    conflates rendering failure with binding failure, so it can't answer a pure binding
    question. This must not change published numbers when no counts are given (counts=None
    behaves identically to today's build_agreement_rows)."""
    labels = {
        ac.label_key(0, "red apron"): "barista",     # would be correct...
        ac.label_key(0, "yellow helmet"): "cyclist",  # ...and correct...
        ac.label_key(1, "white hat"): "chef",         # correct, different image
    }
    counts = {ac.count_key(0): ac.COUNT_BROKEN}  # ...but p0 is count-broken

    rows = ac.build_agreement_rows(_manifest(), labels, counts=counts)
    p0_rows = [r for r in rows if r["prompt_id"] == 0]
    p1_rows = [r for r in rows if r["prompt_id"] == 1]

    assert all(r["count_broken"] is True and r["scored"] is False for r in p0_rows)
    assert all(r["count_broken"] is False for r in p1_rows)
    assert p1_rows[0]["scored"] is True  # unaffected: not count-broken


def test_build_rows_without_counts_arg_is_unaffected_backward_compatible():
    """counts=None (the default) must reproduce every already-published number -- no row
    gets excluded for count reasons unless a counts dict says so."""
    labels = {ac.label_key(0, "red apron"): "barista"}
    rows_no_counts_arg = ac.build_agreement_rows(_manifest(), labels)
    rows_empty_counts = ac.build_agreement_rows(_manifest(), labels, counts={})
    assert rows_no_counts_arg == rows_empty_counts
    assert rows_no_counts_arg[0]["count_broken"] is False
    assert rows_no_counts_arg[0]["scored"] is True


# --------------------------------------------------------------------------- significance

def test_binomial_test_vs_chance_significant_when_far_above_chance():
    # 18/20 correct at chance=0.5 is a clear win.
    p = ac.binomial_test_vs_chance(18, 20, 0.5)
    assert p < 0.01


def test_binomial_test_vs_chance_not_significant_at_chance():
    # 10/20 correct at chance=0.5 is exactly the null.
    p = ac.binomial_test_vs_chance(10, 20, 0.5)
    assert p > 0.5


def test_binomial_test_vs_chance_returns_none_when_nothing_scored():
    assert ac.binomial_test_vs_chance(0, 0, 0.5) is None


def test_binomial_test_vs_chance_below_chance_is_not_significant_one_sided():
    # A one-sided ("greater") test: doing WORSE than chance must not read as "beats chance."
    p = ac.binomial_test_vs_chance(2, 20, 0.5)
    assert p > 0.99


# --------------------------------------------------------------------------- locate_attribute_phrase

def test_locate_attribute_phrase_exact_match_returns_attribute_unchanged():
    prompt = "a photo of a barista wearing a red apron and a cyclist wearing a yellow helmet"
    idx, span = ac.locate_attribute_phrase(prompt, "red apron")
    assert idx == prompt.find("red apron")
    assert span == "red apron"


def test_locate_attribute_phrase_falls_back_to_content_words_on_subphrase_mismatch():
    """Real case from artifacts_flux/manifest.json: the manifest's attribute field says
    'yellow helmet' but the actual prompt says 'yellow bike helmet' -- exact substring match
    fails, so this must fall back to spanning from the first content word to the last."""
    prompt = ("a photo of four people standing side by side, on the far left a barista in a "
              "red apron, on the center-left a man wearing a cycling jersey in a yellow bike "
              "helmet, on the center-right a farmer holding a wooden shovel, on the far right "
              "a nurse wearing blue gloves")
    idx, span = ac.locate_attribute_phrase(prompt, "yellow helmet")
    assert span == "yellow bike helmet"
    assert prompt[idx:idx + len(span)] == span


def test_locate_attribute_phrase_raises_when_a_content_word_is_truly_absent():
    prompt = "a barista wearing a red apron"
    with pytest.raises(ValueError, match="green"):
        ac.locate_attribute_phrase(prompt, "green scarf")


def test_locate_attribute_phrase_raises_on_attribute_with_no_content_words():
    with pytest.raises(ValueError, match="content words"):
        ac.locate_attribute_phrase("a barista wearing a red apron", "   ")


def test_locate_attribute_phrase_does_not_span_across_unrelated_clauses_with_shared_word():
    """Two subjects can share a word (e.g. both wearing something red) -- the shortest local
    match must win, not a greedy span stretching across the unrelated clause in between."""
    prompt = "a barista wearing a red hat and a cyclist wearing a red pinstriped apron"
    idx, span = ac.locate_attribute_phrase(prompt, "red apron")
    assert span == "red pinstriped apron"
    assert prompt[idx:idx + len(span)] == span


def test_locate_attribute_phrase_uses_word_boundaries_not_substring_match():
    """"red" must not match inside "shredded" -- word matching must be boundary-anchored,
    not a plain substring search."""
    prompt = "a chef holding shredded lettuce and wearing a crimson apron"
    with pytest.raises(ValueError, match="red"):
        ac.locate_attribute_phrase(prompt, "red apron")


def test_locate_attribute_phrase_raises_on_empty_string_attribute():
    prompt = "a barista wearing a red apron"
    with pytest.raises(ValueError, match="content words"):
        ac.locate_attribute_phrase(prompt, "")


def test_locate_attribute_phrase_exact_match_path_is_also_word_boundary_anchored():
    """A single-word attribute must not exact-match inside a larger word either -- "red"
    must not match inside "shredded" via the fast substring path any more than it does via
    the word-by-word fallback path."""
    prompt = "a chef holding shredded lettuce and wearing a crimson smock"
    with pytest.raises(ValueError, match="red"):
        ac.locate_attribute_phrase(prompt, "red")


def test_locate_attribute_phrase_matches_words_forward_in_order_not_independently():
    """A decoy 'helmet' appears before the real 'yellow ... helmet' phrase -- an unordered
    matcher could pair 'yellow' with the later helmet and the decoy 'helmet' with nothing,
    or otherwise produce a wrong span. Locks in that words are matched strictly in order,
    each search continuing forward from the previous word's own match, not just "somewhere
    in the prompt"."""
    prompt = "a helmet on the shelf, and a cyclist wearing a yellow bike helmet"
    idx, span = ac.locate_attribute_phrase(prompt, "yellow helmet")
    assert span == "yellow bike helmet"
    assert prompt[idx:idx + len(span)] == span
