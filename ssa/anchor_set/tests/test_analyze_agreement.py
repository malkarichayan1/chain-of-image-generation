"""Tests for analyze_agreement.py: end-to-end join + formatting. Run from inside
ssa/anchor_set/:  py -3 -m pytest tests/"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import anchor_common as ac
import analyze_agreement as aa


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


def test_analyze_writes_csv_and_reports_per_stratum(tmp_path, monkeypatch, capsys):
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "artifacts" / "manifest.json").write_text(json.dumps(_manifest()))
    labels = {
        ac.label_key(0, "red apron"): "barista",       # correct
        ac.label_key(0, "yellow helmet"): "cyclist",   # predicted barista -> wrong
        ac.label_key(6, "hat"): "a",                    # correct
    }
    (tmp_path / "artifacts" / "labels_t.json").write_text(json.dumps(labels))
    monkeypatch.setattr(aa, "ARTIFACTS_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(aa, "MANIFEST_PATH", tmp_path / "artifacts" / "manifest.json")

    summary = aa.analyze("t")

    assert summary["by_stratum"][2]["accuracy"] == pytest.approx(0.5)
    assert summary["by_stratum"][2]["chance"] == pytest.approx(0.5)
    assert summary["by_stratum"][3]["accuracy"] == pytest.approx(1.0)
    assert summary["by_stratum"][3]["chance"] == pytest.approx(1 / 3)
    assert summary["overall"]["n_scored"] == 3 and summary["overall"]["n_correct"] == 2

    csv = (tmp_path / "artifacts" / "agreement_t.csv").read_text()
    assert "predicted_owner" in csv and "human_label" in csv
    out = capsys.readouterr().out
    assert "n=2" in out and "n=3" in out and "overall" in out


def test_format_summary_handles_empty_stratum():
    summary = ac.summarize_agreement([
        {"prompt_id": 0, "n": 2, "attribute": "x", "intended_subject": "a",
         "predicted_owner": "a", "human_label": ac.LABEL_NONE, "scored": False, "correct": False},
    ])
    text = aa.format_summary(summary)  # must not raise on all-None accuracy
    assert "n=2" in text and "overall" in text
