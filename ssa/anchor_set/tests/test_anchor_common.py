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
    assert by_n == {2: 6, 3: 6, 4: 6}


def test_embedded_anchor_prompts_match_prompt_specs_json():
    """generate_anchor_images.py embeds ANCHOR_PROMPTS for the single-file Kaggle kernel;
    it must stay bit-identical to the canonical prompt_specs.json or the manifest would
    describe different prompts than were authored."""
    src = (PKG / "generate_anchor_images.py").read_text(encoding="utf-8")
    seg = None
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and any(
                getattr(t, "id", None) == "ANCHOR_PROMPTS" for t in node.targets):
            seg = ast.get_source_segment(src, node)
    assert seg is not None, "ANCHOR_PROMPTS assignment not found"
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
