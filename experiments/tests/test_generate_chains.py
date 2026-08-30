"""
Thin coverage for generate_chains.py (Stage 1), by design: only base_prompt_for and the
manifest-schema assembly are tested here. Model calls (image generation, detection, CLIP matching) stay untested --
they need a GPU and are exercised only by an actual Kaggle run.
"""
from generate_chains import MECHANISM_PROMPTS, base_prompt_for, manifest_chain_entry


def test_base_prompt_for_strips_attributes_for_two_subjects():
    spec = MECHANISM_PROMPTS[0]
    assert base_prompt_for(spec) == "a photo of a barista and a cyclist"


def test_base_prompt_for_strips_attributes_for_three_subjects():
    spec = MECHANISM_PROMPTS[3]
    assert base_prompt_for(spec) == "a photo of a barista, a cyclist, and a chef"


def test_base_prompt_for_strips_attributes_for_four_subjects():
    spec = MECHANISM_PROMPTS[6]
    assert base_prompt_for(spec) == "a photo of a barista, a cyclist, a chef, and a farmer"


def test_manifest_chain_entry_records_detection_failure_explicitly():
    spec = MECHANISM_PROMPTS[3]
    entry = manifest_chain_entry(
        "p3_s42", prompt_id=3, seed=42, spec=spec, detected=False,
        num_people_detected=1, subject_boxes={}, base_image_path="images/p3_s42_step0.png", steps=[],
    )
    assert entry["detected"] is False
    assert entry["num_people_detected"] == 1
    assert entry["steps"] == []
    assert entry["n"] == spec["n"]
    assert entry["prompt"] == spec["prompt"]


def test_manifest_chain_entry_records_successful_detection():
    spec = MECHANISM_PROMPTS[0]
    steps = [{"step_idx": 1, "subject": "barista", "attribute": "red apron",
              "image_path": "images/p0_s42_step1.png", "attention_path": "attn/p0_s42_step1.npy"}]
    entry = manifest_chain_entry(
        "p0_s42", prompt_id=0, seed=42, spec=spec, detected=True, num_people_detected=2,
        subject_boxes={"barista": [0, 0, 1, 1], "cyclist": [1, 1, 2, 2]},
        base_image_path="images/p0_s42_step0.png", steps=steps,
    )
    assert entry["detected"] is True
    assert entry["steps"] == steps
    assert set(entry["subject_boxes"].keys()) == {"barista", "cyclist"}
