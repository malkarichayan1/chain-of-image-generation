"""Tests for mis_binding_detection.py: does the attention margin catch real binding
failures, not just match the human label? Run from inside ssa/anchor_set/:
py -3 -m pytest tests/"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import anchor_common as ac
import mis_binding_detection as mbd


def _manifest():
    return {"images": [
        # n=2, attention correctly favors intended subject "barista"
        {"prompt_id": 0, "n": 2, "prompt": "p0", "detected": True, "subjects": ["barista", "cyclist"],
         "attributes": [
             {"attribute": "red apron", "intended_subject": "barista", "predicted_owner": "barista",
              "model_scores": {"barista": 0.8, "cyclist": 0.2}}]},
        # n=2, attention wrongly favors "cyclist" over intended "barista" -- a real mis-binding
        # the human also confirms (human_label will be set to "cyclist" in the labels dict below)
        {"prompt_id": 1, "n": 2, "prompt": "p1", "detected": True, "subjects": ["barista", "cyclist"],
         "attributes": [
             {"attribute": "yellow helmet", "intended_subject": "barista", "predicted_owner": "cyclist",
              "model_scores": {"barista": 0.3, "cyclist": 0.7}}]},
        # n=2, attention favors intended "a", but human says it actually rendered on "b"
        # (predicted_failure=False, actual_failure=True -- a false negative)
        {"prompt_id": 2, "n": 2, "prompt": "p2", "detected": True, "subjects": ["a", "b"],
         "attributes": [
             {"attribute": "hat", "intended_subject": "a", "predicted_owner": "a",
              "model_scores": {"a": 0.6, "b": 0.4}}]},
    ]}


def _labels():
    return {
        ac.label_key(0, "red apron"): "barista",     # actual success
        ac.label_key(1, "yellow helmet"): "cyclist",  # actual failure, correctly flagged (TP)
        ac.label_key(2, "hat"): "b",                  # actual failure, missed (FN)
    }


class TestSignedConfidence:
    def test_positive_when_intended_dominates(self):
        assert mbd.signed_confidence({"a": 0.8, "b": 0.2}, "a") == pytest.approx(0.75)

    def test_negative_when_rival_dominates(self):
        assert mbd.signed_confidence({"a": 0.2, "b": 0.8}, "a") == pytest.approx(-0.75)

    def test_zero_on_exact_tie(self):
        assert mbd.signed_confidence({"a": 0.5, "b": 0.5}, "a") == pytest.approx(0.0)

    def test_handles_more_than_two_subjects_by_comparing_best_rival(self):
        # best rival is c (0.9), not b -- intended "a" loses badly regardless of b
        conf = mbd.signed_confidence({"a": 0.3, "b": 0.1, "c": 0.9}, "a")
        assert conf < 0
        assert conf == pytest.approx((0.3 - 0.9) / 0.9)


class TestBuildFailureRows:
    def test_flags_match_hand_worked_example(self):
        rows = mbd.build_failure_rows(_manifest(), _labels())
        by_attr = {r["attribute"]: r for r in rows}

        assert by_attr["red apron"]["predicted_failure"] is False
        assert by_attr["red apron"]["actual_failure"] is False

        assert by_attr["yellow helmet"]["predicted_failure"] is True
        assert by_attr["yellow helmet"]["actual_failure"] is True

        assert by_attr["hat"]["predicted_failure"] is False
        assert by_attr["hat"]["actual_failure"] is True  # the false negative

    def test_unclear_label_is_not_scored_and_not_an_actual_failure(self):
        labels = {ac.label_key(0, "red apron"): ac.LABEL_UNCLEAR}
        rows = mbd.build_failure_rows(_manifest(), labels)
        row = next(r for r in rows if r["attribute"] == "red apron")
        assert row["scored"] is False
        assert row["actual_failure"] is False  # missing data, not a claimed failure

    def test_count_broken_image_is_excluded_from_scored(self):
        counts = {ac.count_key(1): ac.COUNT_BROKEN}
        rows = mbd.build_failure_rows(_manifest(), _labels(), counts=counts)
        row = next(r for r in rows if r["attribute"] == "yellow helmet")
        assert row["scored"] is False
        assert row["actual_failure"] is False


class TestConfusionReport:
    def test_matches_hand_worked_confusion_matrix(self):
        rows = mbd.build_failure_rows(_manifest(), _labels())
        report = mbd.confusion_report(rows)
        assert report["n"] == 3
        assert report["tp"] == 1  # yellow helmet
        assert report["fn"] == 1  # hat
        assert report["tn"] == 1  # red apron
        assert report["fp"] == 0
        assert report["sensitivity"] == pytest.approx(0.5)   # 1 of 2 actual failures caught
        assert report["specificity"] == pytest.approx(1.0)   # the 1 actual success correctly left alone
        assert 0.0 <= report["fisher_p_value"] <= 1.0

    def test_none_metrics_when_no_scored_rows(self):
        report = mbd.confusion_report([])
        assert report["n"] == 0
        assert report["sensitivity"] is None
        assert report["fisher_p_value"] is None


class TestAurocFailureDetection:
    def test_perfect_separation_scores_one(self):
        rows = [
            dict(scored=True, actual_failure=True, signed_confidence=-0.9),
            dict(scored=True, actual_failure=True, signed_confidence=-0.5),
            dict(scored=True, actual_failure=False, signed_confidence=0.5),
            dict(scored=True, actual_failure=False, signed_confidence=0.9),
        ]
        assert mbd.auroc_failure_detection(rows) == pytest.approx(1.0)

    def test_inverted_separation_scores_zero(self):
        rows = [
            dict(scored=True, actual_failure=True, signed_confidence=0.9),
            dict(scored=True, actual_failure=True, signed_confidence=0.5),
            dict(scored=True, actual_failure=False, signed_confidence=-0.5),
            dict(scored=True, actual_failure=False, signed_confidence=-0.9),
        ]
        assert mbd.auroc_failure_detection(rows) == pytest.approx(0.0)

    def test_none_when_only_one_class_present(self):
        rows = [dict(scored=True, actual_failure=True, signed_confidence=0.1)]
        assert mbd.auroc_failure_detection(rows) is None

    def test_none_when_no_scored_rows(self):
        assert mbd.auroc_failure_detection([]) is None


class TestMisBindingReport:
    def test_report_has_overall_and_per_stratum(self):
        report = mbd.mis_binding_report(_manifest(), _labels())
        assert report["overall"]["n"] == 3
        assert 2 in report["by_stratum"]
        assert report["by_stratum"][2]["n"] == 3

    def test_format_report_renders_headers_and_strata(self):
        report = mbd.mis_binding_report(_manifest(), _labels())
        text = mbd.format_report(report)
        assert "sens" in text and "auroc" in text and "fisher p" in text
        assert "n=2" in text and "overall" in text


def test_main_runs_end_to_end_against_artifacts_dir(tmp_path, monkeypatch, capsys):
    import json
    art = tmp_path / "artifacts"
    art.mkdir()
    (art / "manifest.json").write_text(json.dumps(_manifest()))
    (art / "labels_t.json").write_text(json.dumps(_labels()))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys, "argv",
        ["mis_binding_detection.py", "--artifacts-dir", "artifacts", "--annotator", "t"])
    mbd.main()
    out = capsys.readouterr().out
    assert "auroc" in out.lower()
