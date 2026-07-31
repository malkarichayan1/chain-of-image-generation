"""Tests for exp4_positional_baseline.py (Part A, Experiment 4): does the metric beat a naive
"nearest subject noun in the prompt text" baseline? Run from inside ssa/anchor_set/:
    py -3 -m pytest tests/"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import exp4_positional_baseline as exp4


# --------------------------------------------------------------------------- baseline itself

def test_nearest_subject_picks_immediately_preceding_subject():
    prompt = "a photo of a barista wearing a red apron and a cyclist wearing a yellow helmet"
    assert exp4.nearest_subject_baseline(prompt, ["barista", "cyclist"], "red apron") == "barista"
    assert exp4.nearest_subject_baseline(prompt, ["barista", "cyclist"], "yellow helmet") == "cyclist"


def test_nearest_subject_can_pick_a_following_subject_when_closer():
    """Direction-agnostic: "nearest subject noun" means nearest by distance, not just nearest
    preceding -- this prompt has no subject before the attribute at all, so the nearest
    occurrence must be the following one."""
    prompt = "a scene with a red apron near a barista and far away a cyclist"
    assert exp4.nearest_subject_baseline(prompt, ["barista", "cyclist"], "red apron") == "barista"


def test_nearest_subject_prefers_preceding_over_equidistant_following():
    """SUBJ1 precedes ATTR (gap 9), SUBJ2 follows ATTR (gap 9) -- an exact tie by raw distance.
    Preceding wins by design (matches the user's own "apron comes right after barista" framing
    and the real prompt template, where the intended subject always precedes its attribute) --
    order-independent, not a list-order tie-break."""
    prompt = "SUBJ1xxxxATTRxxxxSUBJ2"
    assert exp4.nearest_subject_baseline(prompt, ["SUBJ1", "SUBJ2"], "ATTR") == "SUBJ1"
    assert exp4.nearest_subject_baseline(prompt, ["SUBJ2", "SUBJ1"], "ATTR") == "SUBJ1"


# --------------------------------------------------------------------------- join + reports

def _prompts_by_id():
    return {
        0: {"prompt": "a photo of a barista wearing a red apron and a cyclist wearing a yellow helmet",
            "subjects": ["barista", "cyclist"]},
        6: {"prompt": "a photo of a chef wearing a white hat, a farmer holding a shovel, and a nurse wearing blue gloves",
            "subjects": ["chef", "farmer", "nurse"]},
    }


def _row(prompt_id, n, attribute, predicted_owner, human_label, scored=True, correct=None):
    return dict(prompt_id=prompt_id, n=n, attribute=attribute, predicted_owner=predicted_owner,
                human_label=human_label, scored=scored,
                correct=(correct if correct is not None else predicted_owner == human_label))


def test_join_baseline_adds_baseline_subject_and_correctness_columns():
    rows = [_row(0, 2, "red apron", "cyclist", "barista")]  # metric wrong, baseline should be right
    df = exp4.join_baseline(rows, _prompts_by_id())
    assert df.loc[0, "baseline_subject"] == "barista"
    assert df.loc[0, "baseline_correct"] == True  # noqa: E712
    assert df.loc[0, "correct"] == False  # noqa: E712


def test_baseline_vs_metric_report_headline_numbers():
    rows = [
        _row(0, 2, "red apron", "barista", "barista"),      # metric correct, baseline correct
        _row(0, 2, "yellow helmet", "barista", "cyclist"),  # metric wrong, baseline correct
    ]
    df = exp4.join_baseline(rows, _prompts_by_id())
    report = exp4.baseline_vs_metric_report(df)
    assert report["n"] == 2
    assert report["metric_accuracy"] == pytest.approx(0.5)
    assert report["baseline_accuracy"] == pytest.approx(1.0)


def test_mcnemar_report_significant_when_baseline_beats_metric_every_time():
    rows = [_row(0, 2, "yellow helmet", "barista", "cyclist") for _ in range(10)]  # metric always wrong, baseline always right
    df = exp4.join_baseline(rows, _prompts_by_id())
    report = exp4.mcnemar_report(df)
    assert report["baseline_only_correct"] == 10
    assert report["metric_only_correct"] == 0
    assert report["p_value"] < 0.01


def test_stratified_report_groups_by_n_with_chance():
    rows = [
        _row(0, 2, "red apron", "barista", "barista"),
        _row(6, 3, "white hat", "chef", "chef"),
    ]
    df = exp4.join_baseline(rows, _prompts_by_id())
    strat = exp4.stratified_report(df)
    assert strat[2]["chance"] == pytest.approx(0.5)
    assert strat[3]["chance"] == pytest.approx(1 / 3)
    assert strat[2]["metric_accuracy"] == pytest.approx(1.0)


def test_unscored_rows_excluded_from_all_reports():
    rows = [_row(0, 2, "red apron", "barista", None, scored=False, correct=False)]
    df = exp4.join_baseline(rows, _prompts_by_id())
    report = exp4.baseline_vs_metric_report(df)
    assert report["n"] == 0
    assert report["metric_accuracy"] is None
    assert report["baseline_accuracy"] is None


def test_nearest_subject_baseline_handles_subphrase_attribute_mismatch():
    """Real FLUX case (artifacts_flux/manifest.json, prompt_id 1): manifest attribute is
    'white hat', prompt says 'tall white chef hat'. Subject 'chef' appears literally in the
    prompt, so this isolates the attribute-side fix without touching the separately
    documented subject-side gap (see the KNOWN GAP comment in nearest_subject_baseline)."""
    prompt = ("a photo of two people standing side by side, on the left a chef in a tall "
              "white chef hat, on the right a farmer holding a wooden shovel")
    subjects = ["chef", "farmer"]
    result = exp4.nearest_subject_baseline(prompt, subjects, "white hat")
    assert result == "chef"
