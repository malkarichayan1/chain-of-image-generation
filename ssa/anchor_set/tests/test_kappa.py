"""Tests for inter-rater reliability (Cohen's kappa) between two annotators' label files.

Motivation: the anchor set's ground truth is currently one person's judgment with no measure
of whether a second person would reproduce it. Kappa is the reliability number a reviewer
asks for. `label_images.py --annotator <name>` already writes per-annotator files with
identical keys, so the only missing piece is the comparison.

Run from inside ssa/anchor_set/:  py -3 -m pytest tests/
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import anchor_common as ac


def test_perfect_agreement_gives_kappa_of_one():
    labels = {"0::a": "barista", "0::b": "cyclist", "1::c": ac.LABEL_NONE}
    result = ac.cohens_kappa(labels, dict(labels))

    assert result["kappa"] == pytest.approx(1.0)
    assert result["n"] == 3
    assert result["p_observed"] == pytest.approx(1.0)


def test_kappa_matches_a_hand_computed_example():
    """po=0.75, pe=0.5 -> kappa=(0.75-0.5)/(1-0.5)=0.5, worked by hand from the marginals."""
    a = {"0::w": "x", "0::x": "x", "0::y": "y", "0::z": "y"}
    b = {"0::w": "x", "0::x": "y", "0::y": "y", "0::z": "y"}
    result = ac.cohens_kappa(a, b)

    assert result["p_observed"] == pytest.approx(0.75)
    assert result["p_expected"] == pytest.approx(0.5)
    assert result["kappa"] == pytest.approx(0.5)


def test_systematic_disagreement_gives_negative_kappa():
    a = {"0::a": "x", "0::b": "y"}
    b = {"0::a": "y", "0::b": "x"}

    assert ac.cohens_kappa(a, b)["kappa"] == pytest.approx(-1.0)


def test_only_keys_present_in_both_files_are_compared():
    """A second annotator who stops early must not be penalised for unlabeled rows."""
    a = {"0::a": "x", "0::b": "y", "0::c": "x"}
    b = {"0::a": "x"}  # stopped after one judgment
    result = ac.cohens_kappa(a, b)

    assert result["n"] == 1
    assert result["p_observed"] == pytest.approx(1.0)


def test_no_overlapping_keys_is_an_error_not_a_silent_nan():
    with pytest.raises(ValueError):
        ac.cohens_kappa({"0::a": "x"}, {"9::z": "y"})


def test_kappa_is_none_when_both_annotators_used_a_single_identical_category():
    """pe==1 makes kappa undefined (0/0). Report it rather than dividing by zero."""
    a = {"0::a": ac.LABEL_NONE, "0::b": ac.LABEL_NONE}
    result = ac.cohens_kappa(a, dict(a))

    assert result["kappa"] is None
    assert result["p_observed"] == pytest.approx(1.0)
    assert result["p_expected"] == pytest.approx(1.0)


def test_reports_the_category_set_actually_used():
    a = {"0::a": "x", "0::b": ac.LABEL_SHARED}
    b = {"0::a": "x", "0::b": "x"}

    assert set(ac.cohens_kappa(a, b)["categories"]) == {"x", ac.LABEL_SHARED}
