"""Tests for exp3b_within_item_permutation.py (Part A, Experiment 3b): the sharper falsification
control briefing §5.4 asks for -- swap which attribute's own model_scores feeds each attribute's
ownership call, WITHIN one image, restricted to n >= 3 (a within-item derangement degenerates to
a forced swap at n=2, same reasoning test_exp3_attention_scramble.py documents for exp3). Run
from inside ssa/anchor_set/:  py -3 -m pytest tests/"""
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import anchor_common as ac
import exp3b_within_item_permutation as exp3b


def _image(prompt_id, n, decisive=True):
    """One detected image at stratum n with n distinct subjects/attributes. `decisive=True`
    gives each attribute a model_scores dict whose OWN subject wins the argmax by a wide
    margin, and every OTHER subject a low, near-tied score -- so a within-item derangement
    (borrowing a different attribute's map) reliably predicts the WRONG subject, letting tests
    assert exact accuracy rather than just "some value in [0,1]"."""
    subjects = [f"s{prompt_id}_{i}" for i in range(n)]
    attrs = []
    for i, subj in enumerate(subjects):
        scores = {s: 0.01 for s in subjects}
        if decisive:
            scores[subj] = 0.9
        attrs.append({"attribute": f"attr{i}", "intended_subject": subj,
                      "predicted_owner": subj, "model_scores": scores})
    return {"prompt_id": prompt_id, "n": n, "detected": True, "subjects": subjects,
            "attributes": attrs}


def _labels_all_intended(manifest):
    return {ac.label_key(img["prompt_id"], attr["attribute"]): attr["intended_subject"]
            for img in manifest["images"] for attr in img["attributes"]}


# --------------------------------------------------------------------------- random_derangement

def test_random_derangement_has_no_fixed_points():
    rng = random.Random(0)
    for n in (2, 3, 4, 5, 6):
        for _ in range(20):
            perm = exp3b.random_derangement(n, rng)
            assert sorted(perm) == list(range(n))
            assert all(perm[i] != i for i in range(n))


def test_random_derangement_rejects_n_below_2():
    with pytest.raises(ValueError):
        exp3b.random_derangement(1, random.Random(0))


# --------------------------------------------------------------------------- eligibility (n >= 3)

def test_eligible_images_excludes_n2_and_includes_n3_plus():
    manifest = {"images": [_image(0, 2), _image(1, 3), _image(2, 4)]}
    eligible_ns = {img["n"] for img in exp3b._eligible_images(manifest, min_n=3)}
    assert eligible_ns == {3, 4}


# --------------------------------------------------------------------------- accuracy / sweep

def test_within_item_permutation_accuracy_scores_only_n3_plus():
    manifest = {"images": [_image(0, 2), _image(1, 3)]}
    labels = _labels_all_intended(manifest)
    result = exp3b.within_item_permutation_accuracy(manifest, labels, seed=0)
    assert 2 not in result  # n=2 skipped entirely, not partially scored
    assert result[3]["n_scored"] == 3
    assert result[3]["chance"] == pytest.approx(1 / 3)


def test_within_item_permutation_drops_accuracy_on_decisive_own_maps():
    # Every attribute's OWN map decisively names itself; a derangement always borrows a
    # DIFFERENT attribute's map, whose argmax decisively names that OTHER attribute's subject
    # -- never this one's. So permuted accuracy must be exactly 0 here.
    manifest = {"images": [_image(0, 4, decisive=True)]}
    labels = _labels_all_intended(manifest)
    for seed in range(10):
        result = exp3b.within_item_permutation_accuracy(manifest, labels, seed=seed)
        assert result[4]["n_correct"] == 0


def test_within_item_permutation_accuracy_is_deterministic_given_a_seed():
    manifest = {"images": [_image(0, 4)]}
    labels = _labels_all_intended(manifest)
    r1 = exp3b.within_item_permutation_accuracy(manifest, labels, seed=3)
    r2 = exp3b.within_item_permutation_accuracy(manifest, labels, seed=3)
    assert r1 == r2


def test_permutation_sweep_reports_median_and_chance_consistency_fraction():
    manifest = {"images": [_image(i, 4) for i in range(3)]}
    labels = _labels_all_intended(manifest)
    sweep = exp3b.permutation_sweep(manifest, labels, seeds=range(20))
    s = sweep[4]
    assert s["n_seeds"] == 20
    assert s["chance"] == pytest.approx(0.25)
    assert 0.0 <= s["median_accuracy"] <= 1.0
    assert 0.0 <= s["frac_not_significantly_different_from_chance"] <= 1.0


# --------------------------------------------------------------------------- real vs. permuted

def test_real_vs_permuted_mcnemar_report_shape():
    manifest = {"images": [_image(i, 4) for i in range(3)]}
    labels = _labels_all_intended(manifest)  # real predicted_owner always correct here
    report = exp3b.real_vs_permuted_mcnemar(manifest, labels, seed=exp3b.MCNEMAR_SEED)
    assert report["n"] == 12  # 3 images x 4 attributes
    assert report["real_only_correct"] + report["permuted_only_correct"] == report["n_discordant"]
    assert report["p_value"] is None or 0.0 <= report["p_value"] <= 1.0


def test_real_vs_permuted_mcnemar_favors_real_when_own_maps_are_decisive():
    # Real predictions are always correct (decisive own maps); permuted predictions are always
    # wrong (they name a different attribute's subject) -- so every row is real-only-correct.
    manifest = {"images": [_image(i, 4, decisive=True) for i in range(3)]}
    labels = _labels_all_intended(manifest)
    report = exp3b.real_vs_permuted_mcnemar(manifest, labels, seed=exp3b.MCNEMAR_SEED)
    assert report["permuted_only_correct"] == 0
    assert report["real_only_correct"] == report["n"]
