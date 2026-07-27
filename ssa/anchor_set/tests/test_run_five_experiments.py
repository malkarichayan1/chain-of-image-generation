"""Tests for run_five_experiments.py: the orchestrator that runs all five Part A experiments in
a single pass. Run from inside ssa/anchor_set/:  py -3 -m pytest tests/"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import anchor_common as ac
import make_dummy_artifacts as mda
import run_five_experiments as orch


def test_run_all_includes_all_five_experiment_keys():
    data = mda.build_dummy_artifacts(seed=0)
    results = orch.run_all(data["manifest"], data["labels"], data["counts"], n_seeds=10)
    for key in ("experiment_1", "experiment_2", "experiment_3", "experiment_4", "experiment_5"):
        assert key in results


def test_experiment_2_available_against_the_dummy_fixture():
    """make_dummy_artifacts.py populates model_scores_full on every attribute -- Experiment 2
    must actually run, not degrade, against it."""
    data = mda.build_dummy_artifacts(seed=0)
    results = orch.run_all(data["manifest"], data["labels"], data["counts"], n_seeds=10)
    assert results["experiment_2"]["available"] is True
    assert "per_stratum" in results["experiment_2"]


def test_experiment_2_marked_unavailable_on_a_manifest_without_full_scores():
    # Two n=2 images -- Experiment 3's own cross-item scramble needs >= 2 per stratum, and
    # this test's job is to isolate Experiment 2's degrade behavior, not trip that guard.
    manifest = {"images": [
        {"prompt_id": 0, "n": 2, "detected": True, "subjects": ["barista", "cyclist"],
         "prompt": "a photo of a barista wearing a red apron and a cyclist wearing a yellow helmet",
         "attributes": [{"attribute": "red apron", "intended_subject": "barista",
                         "predicted_owner": "barista", "model_scores": {"barista": 0.6, "cyclist": 0.4}}]},
        {"prompt_id": 1, "n": 2, "detected": True, "subjects": ["chef", "farmer"],
         "prompt": "a photo of a chef wearing a white hat and a farmer holding a shovel",
         "attributes": [{"attribute": "white hat", "intended_subject": "chef",
                         "predicted_owner": "chef", "model_scores": {"chef": 0.7, "farmer": 0.3}}]},
    ]}
    labels = {ac.label_key(0, "red apron"): "barista", ac.label_key(1, "white hat"): "chef"}
    results = orch.run_all(manifest, labels, {}, n_seeds=10)
    assert results["experiment_2"]["available"] is False
    assert "message" in results["experiment_2"]


def test_format_combined_report_mentions_all_five_experiments():
    data = mda.build_dummy_artifacts(seed=0)
    results = orch.run_all(data["manifest"], data["labels"], data["counts"], n_seeds=10)
    text = orch.format_combined_report(results)
    for label in ("Experiment 1", "Experiment 2", "Experiment 3", "Experiment 4", "Experiment 5"):
        assert label in text


def test_main_writes_json_and_markdown_reports(tmp_path, monkeypatch, capsys):
    art = tmp_path / "artifacts_dummy"
    mda.write_dummy_artifacts(art, seed=0)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "run_five_experiments.py", "--artifacts-dir", "artifacts_dummy",
        "--annotator", "dummy", "--n-seeds", "10",
    ])
    orch.main()
    out_json = art / "five_experiments_dummy.json"
    out_md = art / "five_experiments_dummy.md"
    assert out_json.exists() and out_md.exists()
    payload = json.loads(out_json.read_text())
    assert "experiment_1" in payload
    assert "Experiment 1" in out_md.read_text()


def test_main_uses_counts_annotator_override(tmp_path, monkeypatch):
    art = tmp_path / "artifacts_dummy"
    mda.write_dummy_artifacts(art, seed=0)
    # Rename the counts file to a different annotator id to prove --counts-annotator is honored.
    (art / "counts_other.json").write_text((art / "counts_dummy.json").read_text())
    (art / "counts_dummy.json").unlink()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "run_five_experiments.py", "--artifacts-dir", "artifacts_dummy",
        "--annotator", "dummy", "--counts-annotator", "other", "--n-seeds", "10",
    ])
    orch.main()  # must not raise despite counts_dummy.json missing
    assert (art / "five_experiments_dummy.json").exists()
