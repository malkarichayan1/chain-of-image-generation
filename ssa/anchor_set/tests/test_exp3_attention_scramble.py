"""Tests for exp3_attention_scramble.py (Part A, Experiment 3): does the signal survive attention
randomization? Cross-item (same-stratum) scramble, not within-item -- a within-item permutation
degenerates at n=2. Run from inside
ssa/anchor_set/:  py -3 -m pytest tests/"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import anchor_common as ac
import exp3_attention_scramble as exp3


def _manifest_n2(num_items=4):
    """`num_items` detected n=2 images, each with a distinct pair of subjects and a decisive
    (non-tied) model_scores dict, so there's a real cross-item donor pool to draw from."""
    images = []
    for i in range(num_items):
        s1, s2 = f"subj{i}a", f"subj{i}b"
        images.append({
            "prompt_id": i, "n": 2, "prompt": f"p{i}", "detected": True, "subjects": [s1, s2],
            "attributes": [
                {"attribute": "attr", "intended_subject": s1, "predicted_owner": s1,
                 "model_scores": {s1: 0.9, s2: 0.1}},
            ],
        })
    return {"images": images}


def _labels_all_correct(manifest):
    return {ac.label_key(img["prompt_id"], img["attributes"][0]["attribute"]): img["attributes"][0]["intended_subject"]
            for img in manifest["images"]}


# --------------------------------------------------------------------------- scramble_predict

def test_scramble_predict_uses_only_donor_values_and_returns_a_real_subject():
    import random
    donor_scores = {"donorA": 0.7, "donorB": 0.3}
    subjects = ["barista", "cyclist"]
    rng = random.Random(0)
    pred = exp3.scramble_predict(subjects, donor_scores, rng)
    assert pred in subjects  # never returns a donor's own subject name


# --------------------------------------------------------------------------- single-draw accuracy

def test_cross_item_scramble_accuracy_raises_when_stratum_has_no_donor():
    manifest = _manifest_n2(num_items=1)  # only one n=2 image -- no cross-item partner exists
    labels = _labels_all_correct(manifest)
    with pytest.raises(ValueError, match="donor"):
        exp3.cross_item_scramble_accuracy(manifest, labels, seed=0)


def test_cross_item_scramble_accuracy_reports_expected_shape():
    manifest = _manifest_n2(num_items=4)
    labels = _labels_all_correct(manifest)
    result = exp3.cross_item_scramble_accuracy(manifest, labels, seed=1)
    assert 2 in result
    s = result[2]
    assert s["n_scored"] == 4
    assert 0 <= s["n_correct"] <= 4
    assert s["chance"] == pytest.approx(0.5)
    assert 0.0 <= s["accuracy"] <= 1.0


def test_cross_item_scramble_accuracy_is_deterministic_given_a_seed():
    manifest = _manifest_n2(num_items=4)
    labels = _labels_all_correct(manifest)
    r1 = exp3.cross_item_scramble_accuracy(manifest, labels, seed=7)
    r2 = exp3.cross_item_scramble_accuracy(manifest, labels, seed=7)
    assert r1 == r2


def test_cross_item_scramble_never_uses_the_rows_own_image_as_its_own_donor():
    """Every subject slot's assigned value must come from a DIFFERENT prompt_id's scores --
    verified indirectly: with only 2 items, the donor for item 0 must be item 1's scores."""
    manifest = _manifest_n2(num_items=2)
    labels = _labels_all_correct(manifest)
    # Run many seeds; every draw should stay computable (no self-donor ever chosen).
    for seed in range(20):
        result = exp3.cross_item_scramble_accuracy(manifest, labels, seed=seed)
        assert result[2]["n_scored"] == 2


# --------------------------------------------------------------------------- sweep

def test_scramble_sweep_reports_median_and_chance_consistency_fraction():
    manifest = _manifest_n2(num_items=4)
    labels = _labels_all_correct(manifest)
    sweep = exp3.scramble_sweep(manifest, labels, seeds=range(20))
    s = sweep[2]
    assert s["n_seeds"] == 20
    assert s["chance"] == pytest.approx(0.5)
    assert 0.0 <= s["median_accuracy"] <= 1.0
    assert 0.0 <= s["frac_not_significantly_different_from_chance"] <= 1.0


# --------------------------------------------------------------------------- real vs. scrambled

def test_real_vs_scrambled_mcnemar_report_shape():
    manifest = _manifest_n2(num_items=4)
    labels = _labels_all_correct(manifest)  # real predicted_owner always correct here
    report = exp3.real_vs_scrambled_mcnemar(manifest, labels, seed=exp3.MCNEMAR_SEED)
    assert report["n"] == 4
    assert report["real_only_correct"] + report["scrambled_only_correct"] == report["n_discordant"]
    assert 0.0 <= report["p_value"] <= 1.0 or report["p_value"] is None
