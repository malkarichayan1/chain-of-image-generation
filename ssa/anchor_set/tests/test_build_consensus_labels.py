"""
Tests for build_consensus_labels.py: majority-vote consensus across annotators' label files
(briefing §9.2). Covers the pure voting logic (majority_vote / build_consensus); the
argparse/file-IO wrapper in main() is exercised manually, matching this repo's convention of
leaving thin CLI plumbing untested (see test_run_five_experiments.py).
"""
import pytest

from build_consensus_labels import build_consensus, majority_vote


def test_majority_vote_unanimous():
    winner, agreement = majority_vote(["chef", "chef", "chef"])
    assert winner == "chef"
    assert agreement == "unanimous"


def test_majority_vote_strict_majority():
    winner, agreement = majority_vote(["chef", "chef", "baker"])
    assert winner == "chef"
    assert agreement == "majority"


def test_majority_vote_no_consensus_three_way_split():
    winner, agreement = majority_vote(["chef", "baker", "nurse"])
    assert winner is None
    assert agreement == "none"


def test_majority_vote_two_annotators_tied_is_no_consensus():
    # A 1-1 tie has no value with MORE than half the votes -- ties are "none", not an
    # arbitrary pick, since guessing between them would fabricate a label no one gave.
    winner, agreement = majority_vote(["chef", "baker"])
    assert winner is None
    assert agreement == "none"


def test_majority_vote_raises_on_empty():
    with pytest.raises(ValueError):
        majority_vote([])


def test_build_consensus_counts_each_outcome_kind():
    label_files = {
        "a": {"1::x": "chef", "2::x": "chef", "3::x": "chef"},
        "b": {"1::x": "chef", "2::x": "chef", "3::x": "baker"},
        "c": {"1::x": "chef", "2::x": "baker", "3::x": "nurse"},
    }
    consensus, stats = build_consensus(label_files)
    assert consensus == {"1::x": "chef", "2::x": "chef"}  # key 3 has no majority
    assert stats == {"unanimous": 1, "majority": 1, "none": 1, "n_keys": 3}


def test_build_consensus_treats_missing_annotator_vote_as_abstention():
    # Annotator "b" never reached key "2::x" -- it should still get a consensus from the
    # two who did, not be dropped or treated as a vote for some default value.
    label_files = {
        "a": {"1::x": "chef", "2::x": "baker"},
        "b": {"1::x": "chef"},
        "c": {"1::x": "chef", "2::x": "baker"},
    }
    consensus, stats = build_consensus(label_files)
    assert consensus == {"1::x": "chef", "2::x": "baker"}
    assert stats["unanimous"] == 2
    assert stats["n_keys"] == 2


def test_build_consensus_key_union_includes_singleton_votes():
    # A key only one annotator ever reached is still "unanimous" (1/1) rather than dropped --
    # there's no disagreement to fail to resolve.
    label_files = {"a": {"1::x": "chef"}, "b": {}, "c": {}}
    consensus, stats = build_consensus(label_files)
    assert consensus == {"1::x": "chef"}
    assert stats["unanimous"] == 1
