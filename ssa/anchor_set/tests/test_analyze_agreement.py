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


def _manifest_with_scores():
    """Same shape as _manifest() but with real model_scores, which the shared-aware path
    needs: p0's helmet is a near-tie (the metric should abstain), p6's hat is decisive."""
    return {"images": [
        {"prompt_id": 0, "n": 2, "prompt": "p0", "detected": True,
         "subjects": ["barista", "cyclist"],
         "attributes": [
             {"attribute": "red apron", "intended_subject": "barista", "predicted_owner": "barista",
              "model_scores": {"barista": 1.0, "cyclist": 0.1}},
             {"attribute": "yellow helmet", "intended_subject": "cyclist", "predicted_owner": "barista",
              "model_scores": {"barista": 0.51, "cyclist": 0.50}}]},
        {"prompt_id": 6, "n": 3, "prompt": "p6", "detected": True, "subjects": ["a", "b", "c"],
         "attributes": [
             {"attribute": "hat", "intended_subject": "a", "predicted_owner": "a",
              "model_scores": {"a": 1.0, "b": 0.1, "c": 0.05}}]},
    ]}


def _write_run(tmp_path, manifest, labels, annotator="t"):
    (tmp_path / "artifacts").mkdir(exist_ok=True)
    (tmp_path / "artifacts" / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "artifacts" / f"labels_{annotator}.json").write_text(json.dumps(labels))
    return tmp_path / "artifacts"


def test_margin_threshold_reports_a_shared_table_and_recovers_leaked_rows(tmp_path, monkeypatch, capsys):
    """A row the human marked `shared` is invisible to strict scoring but scoreable here."""
    labels = {
        ac.label_key(0, "red apron"): "barista",
        ac.label_key(0, "yellow helmet"): ac.LABEL_SHARED,
        ac.label_key(6, "hat"): "a",
    }
    art = _write_run(tmp_path, _manifest_with_scores(), labels)
    monkeypatch.setattr(aa, "ARTIFACTS_DIR", art)
    monkeypatch.setattr(aa, "MANIFEST_PATH", art / "manifest.json")

    result = aa.analyze("t", margin_threshold=0.2)

    assert result["strict"]["overall"]["n_scored"] == 2      # shared row excluded
    assert result["shared"]["overall"]["n_scored"] == 3      # shared row readmitted
    assert result["shared"]["overall"]["n_correct"] == 3     # and the metric abstained correctly
    csv = (art / "agreement_t.csv").read_text()
    assert "predicted_label" in csv and "margin" in csv
    assert "shared" in capsys.readouterr().out.lower()


def test_analyze_without_margin_threshold_returns_the_strict_summary_shape(tmp_path, monkeypatch):
    """Backwards compatibility: the existing call contract must not change."""
    labels = {ac.label_key(0, "red apron"): "barista"}
    art = _write_run(tmp_path, _manifest_with_scores(), labels)
    monkeypatch.setattr(aa, "ARTIFACTS_DIR", art)
    monkeypatch.setattr(aa, "MANIFEST_PATH", art / "manifest.json")

    summary = aa.analyze("t")

    assert "by_stratum" in summary and "overall" in summary  # not the {strict, shared} wrapper


def test_compare_annotator_reports_inter_rater_kappa(tmp_path, monkeypatch, capsys):
    manifest = _manifest_with_scores()
    labels_a = {ac.label_key(0, "red apron"): "barista",
                ac.label_key(0, "yellow helmet"): ac.LABEL_SHARED,
                ac.label_key(6, "hat"): "a"}
    labels_b = {ac.label_key(0, "red apron"): "barista",
                ac.label_key(0, "yellow helmet"): ac.LABEL_SHARED,
                ac.label_key(6, "hat"): "b"}  # one disagreement
    art = _write_run(tmp_path, manifest, labels_a, annotator="t")
    (art / "labels_t2.json").write_text(json.dumps(labels_b))
    monkeypatch.setattr(aa, "ARTIFACTS_DIR", art)
    monkeypatch.setattr(aa, "MANIFEST_PATH", art / "manifest.json")

    aa.analyze("t", compare_annotator="t2")

    out = capsys.readouterr().out
    assert "kappa" in out.lower()
    assert "2/3" in out or "0.667" in out or "66.7" in out


def test_format_summary_handles_empty_stratum():
    summary = ac.summarize_agreement([
        {"prompt_id": 0, "n": 2, "attribute": "x", "intended_subject": "a",
         "predicted_owner": "a", "human_label": ac.LABEL_NONE, "scored": False, "correct": False},
    ])
    text = aa.format_summary(summary)  # must not raise on all-None accuracy
    assert "n=2" in text and "overall" in text
