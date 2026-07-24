"""Tests for the `shared` (attribute-leakage) outcome category.

Motivation: 16/50 SDXL rows were labeled `unclear` because the attribute rendered on more
than one subject. Discarding them treats a real binding OUTCOME as missing data. These tests
pin the behavior that lets both sides of the comparison express it: the human via a `shared`
label, the metric via a top1-vs-top2 attention margin below a threshold.

Run from inside ssa/anchor_set/:  py -3 -m pytest tests/
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import anchor_common as ac


# ---------------------------------------------------------------------------
# margin_from_scores
# ---------------------------------------------------------------------------

def test_margin_is_normalized_gap_between_top_two_subjects():
    # top1=1.0, top2=0.5 -> (1.0-0.5)/1.0
    assert ac.margin_from_scores({"a": 1.0, "b": 0.5}) == pytest.approx(0.5)


def test_margin_ignores_all_but_the_top_two_subjects():
    # a third, much smaller subject must not change the top1-vs-top2 gap
    assert ac.margin_from_scores({"a": 1.0, "b": 0.5, "c": 0.01}) == pytest.approx(0.5)


def test_margin_is_zero_when_top_two_are_tied():
    assert ac.margin_from_scores({"a": 0.7, "b": 0.7}) == pytest.approx(0.0)


def test_margin_is_one_for_a_single_subject():
    """No competitor means nothing to be ambiguous against."""
    assert ac.margin_from_scores({"a": 0.4}) == pytest.approx(1.0)


def test_margin_is_zero_when_all_attention_mass_is_zero():
    """Degenerate map: no evidence for anyone, so maximally ambiguous, not maximally sure."""
    assert ac.margin_from_scores({"a": 0.0, "b": 0.0}) == pytest.approx(0.0)


def test_margin_rejects_empty_scores():
    with pytest.raises(ValueError):
        ac.margin_from_scores({})


# ---------------------------------------------------------------------------
# predicted_label_with_shared
# ---------------------------------------------------------------------------

def test_predicts_shared_when_margin_below_threshold():
    label = ac.predicted_label_with_shared({"a": 0.51, "b": 0.50}, margin_threshold=0.2)
    assert label == ac.LABEL_SHARED


def test_predicts_argmax_subject_when_margin_clears_threshold():
    label = ac.predicted_label_with_shared({"a": 1.0, "b": 0.1}, margin_threshold=0.2)
    assert label == "a"


def test_margin_exactly_at_threshold_is_confident_not_shared():
    """Boundary is inclusive on the confident side: margin >= threshold names a subject.

    Values chosen to be exactly representable in binary floating point -- (1.0-0.5)/1.0 is
    exactly 0.5, whereas e.g. (1.0-0.8)/1.0 is 0.19999999999999996 and would land just
    *under* a 0.2 threshold. Real margins are floats, so an exact-boundary threshold is not
    reliably reachable; that is a reason to sweep the threshold rather than trust one value,
    not a reason to add an epsilon fudge here."""
    label = ac.predicted_label_with_shared({"a": 1.0, "b": 0.5}, margin_threshold=0.5)
    assert label == "a"


def test_threshold_of_zero_reproduces_plain_argmax():
    """Escape hatch: the shared-aware path must be able to collapse to the frozen behavior."""
    assert ac.predicted_label_with_shared({"a": 0.5, "b": 0.5}, margin_threshold=0.0) == "a"


# ---------------------------------------------------------------------------
# build_agreement_rows -- backwards compatibility, then shared-aware columns
# ---------------------------------------------------------------------------

def _manifest():
    return {"images": [
        {"prompt_id": 0, "n": 2, "prompt": "p0", "detected": True,
         "subjects": ["barista", "cyclist"],
         "attributes": [
             # confident, and the human agrees -> correct under both modes
             {"attribute": "red apron", "intended_subject": "barista",
              "predicted_owner": "barista", "model_scores": {"barista": 1.0, "cyclist": 0.1}},
             # near-tie -> metric says shared; human also says shared -> correct only in shared mode
             {"attribute": "yellow helmet", "intended_subject": "cyclist",
              "predicted_owner": "barista", "model_scores": {"barista": 0.51, "cyclist": 0.50}},
         ]},
    ]}


def _labels():
    return {
        ac.label_key(0, "red apron"): "barista",
        ac.label_key(0, "yellow helmet"): ac.LABEL_SHARED,
    }


def test_rows_without_margin_threshold_keep_the_original_strict_columns():
    """Regression: the published strict numbers must not move when this feature is unused."""
    rows = ac.build_agreement_rows(_manifest(), _labels())
    by_attr = {r["attribute"]: r for r in rows}

    assert by_attr["red apron"]["scored"] is True
    assert by_attr["red apron"]["correct"] is True
    # a `shared` human label is NOT a subject, so strict mode excludes it, like none/unclear
    assert by_attr["yellow helmet"]["scored"] is False
    assert by_attr["yellow helmet"]["correct"] is False
    assert "predicted_label" not in by_attr["red apron"]


def test_rows_with_margin_threshold_score_a_shared_row_as_a_real_outcome():
    rows = ac.build_agreement_rows(_manifest(), _labels(), margin_threshold=0.2)
    helmet = {r["attribute"]: r for r in rows}["yellow helmet"]

    assert helmet["predicted_label"] == ac.LABEL_SHARED
    assert helmet["scored_shared"] is True
    assert helmet["correct_shared"] is True
    # strict columns are untouched by the new mode
    assert helmet["scored"] is False


def test_shared_mode_marks_a_confident_prediction_against_a_shared_human_wrong():
    labels = dict(_labels())
    labels[ac.label_key(0, "red apron")] = ac.LABEL_SHARED  # human saw leakage, metric was sure
    rows = ac.build_agreement_rows(_manifest(), labels, margin_threshold=0.2)
    apron = {r["attribute"]: r for r in rows}["red apron"]

    assert apron["predicted_label"] == "barista"
    assert apron["scored_shared"] is True
    assert apron["correct_shared"] is False


def test_none_and_unclear_stay_excluded_even_in_shared_mode():
    """`shared` is a recovered outcome; `none` and `unclear` remain genuine missing data."""
    labels = {ac.label_key(0, "red apron"): ac.LABEL_NONE,
              ac.label_key(0, "yellow helmet"): ac.LABEL_UNCLEAR}
    rows = ac.build_agreement_rows(_manifest(), labels, margin_threshold=0.2)

    assert all(r["scored_shared"] is False for r in rows)


# ---------------------------------------------------------------------------
# summarize_agreement in shared mode
# ---------------------------------------------------------------------------

def test_shared_mode_chance_baseline_accounts_for_the_extra_outcome():
    """With `shared` available the metric picks among n+1 outcomes, so chance is 1/(n+1)."""
    rows = ac.build_agreement_rows(_manifest(), _labels(), margin_threshold=0.2)
    summary = ac.summarize_agreement(rows, mode="shared")

    assert summary["by_stratum"][2]["chance"] == pytest.approx(1 / 3)
    assert summary["by_stratum"][2]["n_scored"] == 2
    assert summary["by_stratum"][2]["n_correct"] == 2


def test_strict_mode_remains_the_default_and_keeps_the_one_over_n_baseline():
    rows = ac.build_agreement_rows(_manifest(), _labels(), margin_threshold=0.2)
    summary = ac.summarize_agreement(rows)

    assert summary["by_stratum"][2]["chance"] == pytest.approx(0.5)
    assert summary["by_stratum"][2]["n_scored"] == 1


def test_summarize_rejects_shared_mode_on_rows_built_without_a_margin_threshold():
    """Fail loudly rather than silently reporting an empty shared table."""
    rows = ac.build_agreement_rows(_manifest(), _labels())
    with pytest.raises(ValueError):
        ac.summarize_agreement(rows, mode="shared")
