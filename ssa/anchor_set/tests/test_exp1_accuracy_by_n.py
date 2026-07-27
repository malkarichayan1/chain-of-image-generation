"""Tests for exp1_accuracy_by_n.py (Part A, Experiment 1): does cross-attention beat random
guessing, per subject count? Run from inside ssa/anchor_set/:  py -3 -m pytest tests/"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import anchor_common as ac
import exp1_accuracy_by_n as exp1


def _manifest():
    return {"images": [
        {"prompt_id": 0, "n": 2, "prompt": "p0", "detected": True, "subjects": ["barista", "cyclist"],
         "attributes": [
             {"attribute": "red apron", "intended_subject": "barista", "predicted_owner": "barista", "model_scores": {}},
             {"attribute": "yellow helmet", "intended_subject": "cyclist", "predicted_owner": "barista", "model_scores": {}}]},
        {"prompt_id": 6, "n": 3, "prompt": "p6", "detected": True, "subjects": ["a", "b", "c"],
         "attributes": [
             {"attribute": "hat", "intended_subject": "a", "predicted_owner": "a", "model_scores": {}}]},
    ]}


def test_add_significance_adds_p_value_per_stratum():
    summary = ac.summarize_agreement(ac.build_agreement_rows(_manifest(), {
        ac.label_key(0, "red apron"): "barista",       # correct
        ac.label_key(0, "yellow helmet"): "cyclist",   # predicted barista -> wrong
        ac.label_key(6, "hat"): "a",                    # correct
    }))
    out = exp1.add_significance(summary)
    assert "p_value" in out["by_stratum"][2]
    assert "p_value" in out["by_stratum"][3]
    # n=3: 1/1 correct at chance=1/3 -- one scored row can't reach p<0.05, but it must compute.
    assert out["by_stratum"][3]["p_value"] == pytest.approx(
        ac.binomial_test_vs_chance(1, 1, 1 / 3))


def test_add_significance_overall_p_value_none_when_strata_are_mixed():
    """overall's chance is None whenever more than one n is present (summarize_agreement's own
    rule) -- add_significance must not invent a p-value against an undefined chance."""
    summary = ac.summarize_agreement(ac.build_agreement_rows(_manifest(), {
        ac.label_key(0, "red apron"): "barista",
        ac.label_key(6, "hat"): "a",
    }))
    out = exp1.add_significance(summary)
    assert out["overall"]["chance"] is None
    assert out["overall"]["p_value"] is None


def test_add_significance_none_when_stratum_has_zero_scored_rows():
    summary = ac.summarize_agreement(ac.build_agreement_rows(_manifest(), {
        ac.label_key(6, "hat"): ac.LABEL_UNCLEAR,  # excluded -> n=3 stratum has 0 scored
    }))
    out = exp1.add_significance(summary)
    assert out["by_stratum"][3]["n_scored"] == 0
    assert out["by_stratum"][3]["p_value"] is None


def test_accuracy_by_n_report_reproduces_summarize_agreement_numbers():
    """Must not silently change any published accuracy/chance number -- only add p_value."""
    labels = {
        ac.label_key(0, "red apron"): "barista",
        ac.label_key(0, "yellow helmet"): "cyclist",
        ac.label_key(6, "hat"): "a",
    }
    plain = ac.summarize_agreement(ac.build_agreement_rows(_manifest(), labels))
    report = exp1.accuracy_by_n_report(_manifest(), labels)
    for n in (2, 3):
        for key in ("n_labeled", "n_scored", "n_correct", "accuracy", "chance", "margin_over_chance"):
            assert report["by_stratum"][n][key] == plain["by_stratum"][n][key]


def test_format_report_includes_p_value_column_and_strata():
    labels = {
        ac.label_key(0, "red apron"): "barista",
        ac.label_key(0, "yellow helmet"): "cyclist",
        ac.label_key(6, "hat"): "a",
    }
    report = exp1.accuracy_by_n_report(_manifest(), labels)
    text = exp1.format_report(report)
    assert "p-value" in text.lower()
    assert "n=2" in text and "n=3" in text and "overall" in text


def test_main_runs_end_to_end_against_artifacts_dir(tmp_path, monkeypatch, capsys):
    import json
    art = tmp_path / "artifacts"
    art.mkdir()
    (art / "manifest.json").write_text(json.dumps(_manifest()))
    (art / "labels_t.json").write_text(json.dumps({
        ac.label_key(0, "red apron"): "barista",
        ac.label_key(6, "hat"): "a",
    }))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["exp1_accuracy_by_n.py", "--artifacts-dir", "artifacts", "--annotator", "t"])
    exp1.main()
    out = capsys.readouterr().out
    assert "p-value" in out.lower()
