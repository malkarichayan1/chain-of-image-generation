"""Tests for exp5_count_clean_subset.py (Part A, Experiment 5): what happens restricted to
count-clean items only? Run from inside ssa/anchor_set/:  py -3 -m pytest tests/"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import anchor_common as ac
import exp5_count_clean_subset as exp5


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


def test_both_tables_present_and_all_rows_covers_more_than_count_clean():
    labels = {
        ac.label_key(0, "red apron"): "barista",     # would be correct
        ac.label_key(0, "yellow helmet"): "cyclist",  # predicted barista -> wrong
        ac.label_key(6, "hat"): "a",                   # correct, different image
    }
    counts = {ac.count_key(0): ac.COUNT_BROKEN}  # p0 is count-broken
    report = exp5.count_clean_vs_all_report(_manifest(), labels, counts)
    assert report["all"]["overall"]["n_scored"] == 3
    assert report["count_clean"]["overall"]["n_scored"] == 1  # only p6's row survives
    assert report["n_excluded_as_count_broken"] == 2


def test_report_unaffected_when_nothing_is_count_broken():
    labels = {ac.label_key(6, "hat"): "a"}
    counts = {ac.count_key(0): ac.COUNT_CLEAN}
    report = exp5.count_clean_vs_all_report(_manifest(), labels, counts)
    assert report["all"]["overall"]["n_scored"] == report["count_clean"]["overall"]["n_scored"] == 1
    assert report["n_excluded_as_count_broken"] == 0


def test_report_with_empty_counts_dict_matches_all_rows_view():
    labels = {ac.label_key(6, "hat"): "a"}
    report = exp5.count_clean_vs_all_report(_manifest(), labels, {})
    assert report["all"] == report["count_clean"]
    assert report["n_excluded_as_count_broken"] == 0


def test_format_report_shows_both_tables_and_excluded_count():
    labels = {
        ac.label_key(0, "red apron"): "barista",
        ac.label_key(6, "hat"): "a",
    }
    counts = {ac.count_key(0): ac.COUNT_BROKEN}
    report = exp5.count_clean_vs_all_report(_manifest(), labels, counts)
    text = exp5.format_report(report)
    assert "all rows" in text.lower()
    assert "count-clean" in text.lower()
    assert "excluded" in text.lower()


def test_main_runs_end_to_end(tmp_path, monkeypatch, capsys):
    import json
    art = tmp_path / "artifacts"
    art.mkdir()
    (art / "manifest.json").write_text(json.dumps(_manifest()))
    (art / "labels_t.json").write_text(json.dumps({ac.label_key(6, "hat"): "a"}))
    (art / "counts_t.json").write_text(json.dumps({ac.count_key(0): ac.COUNT_BROKEN}))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["exp5_count_clean_subset.py", "--artifacts-dir", "artifacts", "--annotator", "t"])
    exp5.main()
    out = capsys.readouterr().out
    assert "count-clean" in out.lower()
