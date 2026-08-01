"""Integration smoke test: does run_five_experiments.py complete against a FLUX-shaped manifest
that includes a sub-phrase attribute mismatch (Task 1/2's bug) and populated model_scores on
every attribute? The generic artifacts_dummy fixture never exercises the mismatch path, since
its prompts always use exact attribute phrasing -- this fixture is built to hit it deliberately.
Run from inside ssa/anchor_set/:  py -3 -m pytest tests/"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import anchor_common as ac
import run_five_experiments as rfe


def _flux_shaped_manifest():
    prompt_a = "a photo of a barista wearing a red apron and a cyclist wearing a yellow bike helmet"
    prompt_b = "a photo of a chef wearing a white hat and a farmer holding a wooden shovel"
    images = [
        dict(prompt_id=0, n=2, prompt=prompt_a, subjects=["barista", "cyclist"], seed=42,
             detected=True, num_people_detected=2, image_path="artifacts_flux/images/p0.png",
             attributes=[
                 dict(attribute="red apron", intended_subject="barista", predicted_owner="barista",
                      model_scores={"barista": 0.7, "cyclist": 0.3},
                      predicted_owner_full="barista", model_scores_full={"barista": 0.6, "cyclist": 0.4}),
                 # "yellow helmet" (manifest) vs "yellow bike helmet" (prompt) -- the mismatch case
                 dict(attribute="yellow helmet", intended_subject="cyclist", predicted_owner="cyclist",
                      model_scores={"barista": 0.2, "cyclist": 0.8},
                      predicted_owner_full="cyclist", model_scores_full={"barista": 0.3, "cyclist": 0.7}),
             ]),
        dict(prompt_id=1, n=2, prompt=prompt_b, subjects=["chef", "farmer"], seed=7,
             detected=True, num_people_detected=2, image_path="artifacts_flux/images/p1.png",
             attributes=[
                 dict(attribute="white hat", intended_subject="chef", predicted_owner="farmer",
                      model_scores={"chef": 0.4, "farmer": 0.6},
                      predicted_owner_full="chef", model_scores_full={"chef": 0.55, "farmer": 0.45}),
                 dict(attribute="wooden shovel", intended_subject="farmer", predicted_owner="farmer",
                      model_scores={"chef": 0.1, "farmer": 0.9},
                      predicted_owner_full="farmer", model_scores_full={"chef": 0.15, "farmer": 0.85}),
             ]),
    ]
    manifest = dict(model="black-forest-labs/FLUX.1-dev", candidate_seeds=[42, 7], img_size=1024,
                    num_inference_steps=25, early_window_fraction=0.5, images=images)
    labels = {
        ac.label_key(0, "red apron"): "barista",
        ac.label_key(0, "yellow helmet"): "cyclist",
        ac.label_key(1, "white hat"): "chef",
        ac.label_key(1, "wooden shovel"): "farmer",
    }
    counts = {ac.count_key(0): ac.COUNT_CLEAN, ac.count_key(1): ac.COUNT_CLEAN}
    return manifest, labels, counts


def test_run_all_completes_on_flux_shaped_manifest_with_subphrase_mismatch():
    manifest, labels, counts = _flux_shaped_manifest()
    results = rfe.run_all(manifest, labels, counts, n_seeds=5)
    assert set(results.keys()) == {
        "experiment_1", "experiment_2", "experiment_3", "experiment_4", "experiment_5"}


def test_experiment_2_is_available_when_full_trajectory_scores_are_populated():
    manifest, labels, counts = _flux_shaped_manifest()
    results = rfe.run_all(manifest, labels, counts, n_seeds=5)
    assert results["experiment_2"]["available"] is True


def test_experiment_4_does_not_crash_on_the_subphrase_attribute():
    """This is the specific regression Task 1/2 fixed -- exp4's baseline join used to raise
    ValueError on "yellow helmet" not matching "...yellow bike helmet..." literally.

    n == 4 (not >= 1): the fixture is 2 images x 2 attributes, all COUNT_CLEAN, so the
    scored denominator is deterministically 4. An exact count catches a silent-drop
    regression (e.g. build_agreement_rows/join_baseline quietly skipping the mismatched
    row instead of raising) that a >= 1 assertion would miss -- it would still pass with
    only 1 of the 4 rows surviving."""
    manifest, labels, counts = _flux_shaped_manifest()
    results = rfe.run_all(manifest, labels, counts, n_seeds=5)
    assert results["experiment_4"]["headline"]["n"] == 4


def _flux_shaped_manifest_with_unmatched_subject_name():
    """Mirrors the real, already-documented gap in `nearest_subject_baseline`
    (exp4_positional_baseline.py): a subject named in `subjects` that never appears
    literally in `prompt` text (real example: "cyclist" phrased as "a man wearing a
    cycling jersey..."). Deliberately NOT fixing that gap here (already decided against,
    for research-methodology reasons) -- this fixture exists only to confirm the full
    battery degrades gracefully rather than crashing when it hits this pattern.

    "barista" still appears literally, so this is the PARTIAL-mismatch shape (at least one
    subject in the row matches) -- the shape actually observed in artifacts_flux/manifest.json
    (45/301 real subject mentions mismatch this way; 0/103 detected images have EVERY
    subject mismatched). A full-mismatch row (no subject at all found) is a different,
    untested-here case -- see the test docstring below for what that does instead."""
    prompt_a = ("a photo of a barista wearing a red apron and a man wearing a cycling "
                "jersey and a yellow bike helmet")
    prompt_b = "a photo of a chef wearing a white hat and a farmer holding a wooden shovel"
    images = [
        dict(prompt_id=0, n=2, prompt=prompt_a, subjects=["barista", "cyclist"], seed=42,
             detected=True, num_people_detected=2, image_path="artifacts_flux/images/p0.png",
             attributes=[
                 dict(attribute="red apron", intended_subject="barista", predicted_owner="barista",
                      model_scores={"barista": 0.7, "cyclist": 0.3},
                      predicted_owner_full="barista", model_scores_full={"barista": 0.6, "cyclist": 0.4}),
                 dict(attribute="yellow helmet", intended_subject="cyclist", predicted_owner="cyclist",
                      model_scores={"barista": 0.2, "cyclist": 0.8},
                      predicted_owner_full="cyclist", model_scores_full={"barista": 0.3, "cyclist": 0.7}),
             ]),
        dict(prompt_id=1, n=2, prompt=prompt_b, subjects=["chef", "farmer"], seed=7,
             detected=True, num_people_detected=2, image_path="artifacts_flux/images/p1.png",
             attributes=[
                 dict(attribute="white hat", intended_subject="chef", predicted_owner="farmer",
                      model_scores={"chef": 0.4, "farmer": 0.6},
                      predicted_owner_full="chef", model_scores_full={"chef": 0.55, "farmer": 0.45}),
                 dict(attribute="wooden shovel", intended_subject="farmer", predicted_owner="farmer",
                      model_scores={"chef": 0.1, "farmer": 0.9},
                      predicted_owner_full="farmer", model_scores_full={"chef": 0.15, "farmer": 0.85}),
             ]),
    ]
    manifest = dict(model="black-forest-labs/FLUX.1-dev", candidate_seeds=[42, 7], img_size=1024,
                    num_inference_steps=25, early_window_fraction=0.5, images=images)
    labels = {
        ac.label_key(0, "red apron"): "barista",
        ac.label_key(0, "yellow helmet"): "cyclist",
        ac.label_key(1, "white hat"): "chef",
        ac.label_key(1, "wooden shovel"): "farmer",
    }
    counts = {ac.count_key(0): ac.COUNT_CLEAN, ac.count_key(1): ac.COUNT_CLEAN}
    return manifest, labels, counts


def test_run_all_degrades_gracefully_on_unmatched_subject_name_not_a_crash():
    """Traced directly (not assumed) before writing this assertion, both against this
    fixture and against the real artifacts_flux/manifest.json:

    `nearest_subject_baseline` loops over every subject in the row and skips (does not
    raise on) any single subject whose name isn't found literally in the prompt -- it
    only raises ValueError when NONE of the row's subjects match. There is no per-row
    try/except anywhere between it and run_five_experiments.run_all(): exp4.join_baseline
    calls it directly, uncaught. So the actual behavior is neither "caught and excluded"
    nor "silently wrong data" -- it's "keeps working, but the guess pool for that row
    silently shrinks to only the subjects that DO match textually."

    Concretely for this fixture: "cyclist" never appears in prompt_a (phrased as "a man
    wearing a cycling jersey"), so `nearest_subject_baseline` never returns "cyclist" for
    ANY attribute in that image -- it always falls back to "barista", the row's only
    textually-matching subject. That makes the baseline's guess for "yellow helmet"
    (whose true owner is cyclist) deterministically wrong, while "red apron" (owned by
    the textually-present barista) stays correct. No exception either way -- degraded
    accuracy on the affected row, not a crash.

    Checked directly against the real artifacts_flux/manifest.json (103 detected images,
    301 subject mentions, 45 mismatched): 0 of those 103 images have EVERY subject
    mismatched, so this ValueError-free path is what the real GPU run's output actually
    hits today, on 100% of its rows. (Separately confirmed, not exercised by this test:
    a row where NO subject at all matches literally DOES raise ValueError, and it is NOT
    caught anywhere -- that would crash run_all(). That shape doesn't occur in the current
    real manifest, but would be a real crash risk if a future prompt phrased every one of
    its subjects unmatchably.)
    """
    manifest, labels, counts = _flux_shaped_manifest_with_unmatched_subject_name()
    results = rfe.run_all(manifest, labels, counts, n_seeds=5)
    assert set(results.keys()) == {
        "experiment_1", "experiment_2", "experiment_3", "experiment_4", "experiment_5"}
    df4_headline = results["experiment_4"]["headline"]
    assert df4_headline["n"] == 4
    # The baseline gets "red apron" right (barista matches textually) but "yellow helmet"
    # wrong (cyclist never does) -- 2 wrong out of 4 total baseline calls across both images
    # (prompt_b's chef/farmer both match textually, so both of ITS rows are unaffected).
    assert df4_headline["baseline_accuracy"] == 0.75
