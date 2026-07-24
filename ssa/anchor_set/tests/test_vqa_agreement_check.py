"""
Tests for vqa_agreement_check.py: joins BLIP-VQA per-subject-crop scores (computed on
Kaggle GPU by vqa_score_sdxl.py) with the same human labels metric A's own attention
predictions are already scored against, so the two can be compared head-to-head on
identical rows. Pure logic only -- no model calls.
"""
import ast
import sys
from pathlib import Path

import pandas as pd
import pytest

from vqa_agreement_check import (
    ATTRIBUTE_PHRASES, attribute_question, paired_mcnemar, vqa_agreement_rows,
    vqa_predicted_owner,
)

PKG = Path(__file__).resolve().parents[1]


def test_attribute_phrases_match_vqa_score_sdxl_kernels_own_copy():
    """vqa_score_sdxl.py (self-contained Kaggle kernel) duplicates ATTRIBUTE_PHRASES --
    same drift-guard pattern as test_anchor_common.py's ANCHOR_PROMPTS check."""
    src = (PKG / "vqa_score_sdxl.py").read_text(encoding="utf-8")
    seg = None
    for node in ast.parse(src).body:
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == "ATTRIBUTE_PHRASES":
            seg = ast.get_source_segment(src, node.value)
    assert seg is not None, "ATTRIBUTE_PHRASES assignment not found in vqa_score_sdxl.py"
    embedded = ast.literal_eval(seg)
    assert embedded == ATTRIBUTE_PHRASES


def test_attribute_question_uses_wearing_for_clothing_attributes():
    assert attribute_question("red apron") == "Is the person wearing a red apron?"
    assert attribute_question("blue gloves") == "Is the person wearing blue gloves?"


def test_attribute_question_uses_holding_for_held_objects():
    assert attribute_question("shovel") == "Is the person holding a shovel?"
    assert attribute_question("book") == "Is the person holding a book?"


def test_attribute_question_covers_every_controlled_vocabulary_attribute():
    for attribute in ATTRIBUTE_PHRASES:
        q = attribute_question(attribute)
        assert q.startswith("Is the person ") and q.endswith("?")


def test_attribute_question_raises_on_unknown_attribute():
    with pytest.raises(KeyError):
        attribute_question("a top hat")


def test_vqa_predicted_owner_picks_highest_p_yes():
    assert vqa_predicted_owner({"barista": 0.2, "cyclist": 0.9}) == "cyclist"


def test_vqa_predicted_owner_raises_on_empty_scores():
    with pytest.raises(ValueError):
        vqa_predicted_owner({})


def _manifest(images):
    return {"images": images}


def _image(prompt_id, n, subjects, attrs, detected=True):
    return {"prompt_id": prompt_id, "n": n, "subjects": subjects, "detected": detected,
            "attributes": [{"attribute": a, "intended_subject": s} for s, a in attrs]}


def test_vqa_agreement_rows_scores_argmax_against_human_label():
    manifest = _manifest([
        _image(0, 2, ["barista", "cyclist"], [("barista", "red apron"), ("cyclist", "yellow helmet")]),
    ])
    labels = {"0::red apron": "barista", "0::yellow helmet": "cyclist"}
    vqa_scores = {
        "0": {
            "red apron": {"barista": 0.8, "cyclist": 0.1},
            "yellow helmet": {"barista": 0.3, "cyclist": 0.2},  # VQA gets this one wrong
        }
    }
    rows = vqa_agreement_rows(manifest, labels, vqa_scores)
    by_attr = {r["attribute"]: r for r in rows}
    assert by_attr["red apron"]["predicted_owner"] == "barista"
    assert by_attr["red apron"]["correct"] is True
    assert by_attr["yellow helmet"]["predicted_owner"] == "barista"
    assert by_attr["yellow helmet"]["correct"] is False


def test_vqa_agreement_rows_skips_images_with_no_score_entry():
    manifest = _manifest([
        _image(5, 2, ["barista", "cyclist"], [("barista", "red apron"), ("cyclist", "yellow helmet")]),
    ])
    labels = {"5::red apron": "barista", "5::yellow helmet": "cyclist"}
    rows = vqa_agreement_rows(manifest, labels, {})
    assert rows == []


def test_vqa_agreement_rows_skips_undetected_images():
    manifest = _manifest([
        _image(0, 2, ["barista", "cyclist"], [("barista", "red apron")], detected=False),
    ])
    labels = {"0::red apron": "barista"}
    vqa_scores = {"0": {"red apron": {"barista": 0.9, "cyclist": 0.1}}}
    assert vqa_agreement_rows(manifest, labels, vqa_scores) == []


def test_paired_mcnemar_significant_when_a_beats_b_on_discordant_pairs():
    df = pd.DataFrame({
        "scored": [True] * 10,
        "a_correct": [True] * 10,
        "b_correct": [False] * 10,
    })
    report = paired_mcnemar(df, "a_correct", "b_correct")
    assert report["a_only_correct"] == 10
    assert report["b_only_correct"] == 0
    assert report["p_value"] < 0.01


def test_paired_mcnemar_excludes_unscored_rows():
    df = pd.DataFrame({
        "scored": [True, False],
        "a_correct": [True, True],
        "b_correct": [False, False],
    })
    report = paired_mcnemar(df, "a_correct", "b_correct")
    assert report["n"] == 1
