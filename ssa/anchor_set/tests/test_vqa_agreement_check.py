"""
Tests for vqa_agreement_check.py: joins BLIP-VQA per-subject-crop scores (computed on
Kaggle GPU by vqa_score_sdxl.py / vqa_score_flux.py) with the same human labels metric A's
own attention predictions are already scored against, so the two can be compared head-to-head
on identical rows -- plus the two comparisons that matter for the paper: VQAScore vs. the
prompt-obeyed baseline, and VQAScore's accuracy on the misbound subset. Pure logic only -- no
model calls.

`attribute_question` itself is tested in test_anchor_common.py (its canonical home); this file
only guards that both self-contained Kaggle kernels' inline duplicates stay byte-identical to
it -- same drift-guard pattern as test_anchor_common.py's own ANCHOR_PROMPTS check.
"""
import ast
import sys
from pathlib import Path

import pandas as pd
import pytest

from vqa_agreement_check import (
    paired_mcnemar, vqa_agreement_rows, vqa_misbound_subset_report, vqa_predicted_owner,
    vqa_vs_prompt_baseline_report,
)

PKG = Path(__file__).resolve().parents[1]
ANCHOR_COMMON_PATH = PKG / "anchor_common.py"
_DUPLICATED_NAMES = ("_HELD_NOUNS", "_NO_ARTICLE_WORN_NOUNS", "attribute_question")


# --------------------------------------------------------------------------- drift guard

def _extract_top_level_source(path: Path, name: str) -> str:
    """Extracts a top-level function's or assignment's full source text by name, via AST --
    for verbatim comparison against anchor_common.py's canonical copy."""
    src = path.read_text(encoding="utf-8")
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(src, node)
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(getattr(t, "id", None) == name for t in targets):
                return ast.get_source_segment(src, node)
    raise AssertionError(f"{name!r} not found as a top-level definition in {path}")


@pytest.mark.parametrize("kernel_file", ["vqa_score_sdxl.py", "vqa_score_flux.py"])
@pytest.mark.parametrize("name", _DUPLICATED_NAMES)
def test_kaggle_kernel_attribute_logic_matches_anchor_common(kernel_file, name):
    """Both self-contained Kaggle kernels duplicate anchor_common's attribute-question
    logic inline (same convention as ANCHOR_PROMPTS) -- this must stay byte-identical or
    the two runs' VQA questions would silently diverge."""
    canonical = _extract_top_level_source(ANCHOR_COMMON_PATH, name)
    embedded = _extract_top_level_source(PKG / kernel_file, name)
    assert embedded == canonical


# --------------------------------------------------------------------------- vqa_predicted_owner

def test_vqa_predicted_owner_picks_highest_p_yes():
    assert vqa_predicted_owner({"barista": 0.2, "cyclist": 0.9}) == "cyclist"


def test_vqa_predicted_owner_raises_on_empty_scores():
    with pytest.raises(ValueError):
        vqa_predicted_owner({})


# --------------------------------------------------------------------------- vqa_agreement_rows

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


# --------------------------------------------------------------------------- paired_mcnemar

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


# --------------------------------------------------------------------------- vqa_vs_prompt_baseline_report

def test_vqa_vs_prompt_baseline_report_counts_discordant_pairs():
    rows = [
        # VQA correct, baseline wrong
        dict(n=2, scored=True, correct=True, human_label="barista", intended_subject="cyclist"),
        # VQA wrong, baseline correct
        dict(n=2, scored=True, correct=False, human_label="cyclist", intended_subject="cyclist"),
        # both correct -- not discordant
        dict(n=2, scored=True, correct=True, human_label="barista", intended_subject="barista"),
    ]
    o = vqa_vs_prompt_baseline_report(rows)["overall"]
    assert o["n_scored"] == 3
    assert o["b"] == 1  # VQA-only correct
    assert o["c"] == 1  # baseline-only correct
    assert o["vqa_acc"] == pytest.approx(2 / 3)
    assert o["base_acc"] == pytest.approx(2 / 3)


def test_vqa_vs_prompt_baseline_report_excludes_unscored_rows():
    rows = [dict(n=2, scored=False, correct=True, human_label="none", intended_subject="cyclist")]
    o = vqa_vs_prompt_baseline_report(rows)["overall"]
    assert o["n_scored"] == 0
    assert o["vqa_acc"] is None
    assert o["p_value"] is None


def test_vqa_vs_prompt_baseline_report_splits_by_stratum():
    rows = [
        dict(n=2, scored=True, correct=True, human_label="a", intended_subject="a"),
        dict(n=3, scored=True, correct=False, human_label="b", intended_subject="c"),
    ]
    report = vqa_vs_prompt_baseline_report(rows)
    assert set(report["by_stratum"]) == {2, 3}
    assert report["by_stratum"][2]["n_scored"] == 1
    assert report["by_stratum"][3]["n_scored"] == 1


# --------------------------------------------------------------------------- vqa_misbound_subset_report

def test_vqa_misbound_subset_report_restricts_to_disobeyed_rows():
    rows = [
        # model obeyed the prompt -- excluded from the misbound subset
        dict(n=2, scored=True, correct=True, human_label="barista", intended_subject="barista"),
        # model disobeyed; VQA got the rendered outcome right
        dict(n=2, scored=True, correct=True, human_label="cyclist", intended_subject="barista"),
        # model disobeyed; VQA got it wrong
        dict(n=2, scored=True, correct=False, human_label="chef", intended_subject="barista"),
    ]
    o = vqa_misbound_subset_report(rows)["overall"]
    assert o["n_misbound"] == 2
    assert o["n_correct"] == 1
    assert o["accuracy"] == pytest.approx(0.5)


def test_vqa_misbound_subset_report_excludes_unscored_rows():
    rows = [dict(n=2, scored=False, correct=False, human_label="none", intended_subject="barista")]
    o = vqa_misbound_subset_report(rows)["overall"]
    assert o["n_misbound"] == 0
    assert o["accuracy"] is None


def test_vqa_misbound_subset_report_reports_chance_per_stratum():
    rows = [dict(n=4, scored=True, correct=True, human_label="b", intended_subject="a")]
    report = vqa_misbound_subset_report(rows)
    assert report["by_stratum"][4]["chance"] == pytest.approx(0.25)
