"""Tests for exp8_misbound_kappa.py's pure selection/restriction logic (experiment #13).

Covers the two things that can silently corrupt the finding: which rows count as
"disobeyed" (none/unclear/shared must NOT count), and that restricting a label dict to a
key subset actually changes what kappa is computed over.
"""
import pytest

from exp8_misbound_kappa import (
    disobeyed_keys, intended_subject_by_key, kappa_or_none, pairwise_report, restrict,
    selection_keys,
)


def _manifest():
    return {
        "images": [
            {
                "prompt_id": 1, "detected": True, "n": 2,
                "attributes": [
                    {"attribute": "red apron", "intended_subject": "barista"},
                    {"attribute": "yellow helmet", "intended_subject": "cyclist"},
                ],
            },
            {
                "prompt_id": 2, "detected": True, "n": 2,
                "attributes": [{"attribute": "blue scarf", "intended_subject": "chef"}],
            },
            # Detection failure: no attributes at all, must never produce a key.
            {"prompt_id": 3, "detected": False, "n": 2, "attributes": []},
        ]
    }


def test_intended_subject_by_key_skips_undetected_images():
    intended = intended_subject_by_key(_manifest())

    assert intended == {
        "1::red apron": "barista",
        "1::yellow helmet": "cyclist",
        "2::blue scarf": "chef",
    }
    assert not any(k.startswith("3::") for k in intended)


def test_disobeyed_keys_selects_only_wrong_subject_rows():
    intended = intended_subject_by_key(_manifest())
    labels = {
        "1::red apron": "barista",      # obeyed
        "1::yellow helmet": "barista",  # disobeyed -- rendered on the wrong subject
        "2::blue scarf": "chef",        # obeyed
    }

    assert disobeyed_keys(intended, labels) == {"1::yellow helmet"}


@pytest.mark.parametrize("non_subject", ["none", "unclear", "shared"])
def test_non_subject_labels_are_missing_data_not_disobedience(non_subject):
    """none/unclear/shared differ from the intended subject as strings, but they are
    missing data -- counting them as disobeyed would inflate the subset with exactly the
    rows annotators could not read, which is the opposite of what C3 needs."""
    intended = intended_subject_by_key(_manifest())

    assert disobeyed_keys(intended, {"1::red apron": non_subject}) == set()


def test_disobeyed_keys_ignores_labels_with_no_manifest_entry():
    intended = intended_subject_by_key(_manifest())

    assert disobeyed_keys(intended, {"3::ghost attribute": "someone"}) == set()


def test_restrict_keeps_only_requested_keys():
    labels = {"a": "x", "b": "y", "c": "z"}

    assert restrict(labels, {"a", "c"}) == {"a": "x", "c": "z"}
    assert restrict(labels, set()) == {}


def test_kappa_or_none_returns_stub_instead_of_raising_on_no_overlap():
    """A restricted subset can leave a pair with zero shared keys; that is a reportable
    outcome, not a crash. anchor_common.cohens_kappa raises in this case."""
    result = kappa_or_none({"a": "x"}, {"b": "y"})

    assert result["n"] == 0
    assert result["kappa"] is None


def test_kappa_or_none_perfect_agreement_two_categories():
    a = {"k1": "barista", "k2": "cyclist"}

    assert kappa_or_none(a, dict(a))["kappa"] == pytest.approx(1.0)


def test_pairwise_report_covers_every_pair_once():
    labels = {"akhil": {"k": "a"}, "grace": {"k": "a"}, "pranav": {"k": "a"}}

    assert sorted(pairwise_report(labels)) == [
        "akhil_vs_grace", "akhil_vs_pranav", "grace_vs_pranav"
    ]


def test_pairwise_report_restriction_changes_the_denominator():
    labels = {
        "akhil": {"k1": "barista", "k2": "cyclist"},
        "grace": {"k1": "barista", "k2": "cyclist"},
    }

    assert pairwise_report(labels)["akhil_vs_grace"]["n"] == 2
    assert pairwise_report(labels, {"k1"})["akhil_vs_grace"]["n"] == 1


def test_selection_consensus_uses_consensus_labels_not_individual_annotators():
    """The whole point of the consensus mode: an individual annotator's outlier call must
    not pull rows into the subset, because selecting on one member of a pair biases kappa
    downward (see module docstring)."""
    intended = intended_subject_by_key(_manifest())
    labels_by_annotator = {
        "akhil": {"1::red apron": "cyclist"},   # lone outlier -- says disobeyed
        "grace": {"1::red apron": "barista"},
    }
    consensus = {"1::red apron": "barista"}     # majority says obeyed

    keys = selection_keys("consensus", intended, labels_by_annotator, consensus)

    assert keys == set()


def test_selection_either_is_the_broader_asymmetric_alternative():
    intended = intended_subject_by_key(_manifest())
    labels_by_annotator = {
        "akhil": {"1::red apron": "cyclist"},
        "grace": {"1::red apron": "barista"},
    }

    keys = selection_keys("either", intended, labels_by_annotator, consensus=None)

    assert keys == {"1::red apron"}


def test_selection_consensus_without_consensus_file_fails_loudly():
    intended = intended_subject_by_key(_manifest())

    with pytest.raises(ValueError, match="consensus label file"):
        selection_keys("consensus", intended, {}, consensus=None)


def test_selection_rejects_unknown_mode():
    with pytest.raises(ValueError, match="unknown selection mode"):
        selection_keys("whatever", {}, {}, consensus=None)
