"""
Tests for selection_effect_check.py (Part D, check 1): does the detection filter drop
chains at random, or systematically by subject count/prompt?
"""
from selection_effect_check import (
    detection_rate_by_prompt,
    detection_rate_by_subject_count,
    summarize_chains,
    undercount_failure_share,
)


def _manifest(chains):
    return {"seeds": [42], "img_size": 512, "chains": chains}


def _chain(chain_key, prompt_id, n, detected, num_people_detected):
    return {
        "chain_key": chain_key, "prompt_id": prompt_id, "seed": 42, "n": n,
        "prompt": "irrelevant", "detected": detected,
        "num_people_detected": num_people_detected,
        "subject_boxes": {} if not detected else {"s": [0, 0, 1, 1]},
        "base_image_path": "irrelevant.png",
        "steps": [],
    }


def test_summarize_chains_extracts_expected_fields():
    manifest = _manifest([_chain("p0_s42", 0, 2, True, 2)])
    out = summarize_chains(manifest)
    assert out == [{
        "chain_key": "p0_s42", "prompt_id": 0, "seed": 42, "n": 2,
        "detected": True, "num_people_detected": 2,
    }]


def test_detection_rate_by_prompt_computes_per_prompt_rate():
    chains = summarize_chains(_manifest([
        _chain("p0_s1", 0, 2, True, 2),
        _chain("p0_s2", 0, 2, False, 1),
        _chain("p1_s1", 1, 3, True, 3),
    ]))
    out = detection_rate_by_prompt(chains).set_index("prompt_id")
    assert out.loc[0, "attempted"] == 2
    assert out.loc[0, "detected"] == 1
    assert out.loc[0, "rate"] == 0.5
    assert out.loc[1, "rate"] == 1.0


def test_detection_rate_by_subject_count_groups_by_n():
    chains = summarize_chains(_manifest([
        _chain("p0_s1", 0, 2, True, 2),
        _chain("p1_s1", 1, 3, False, 2),
        _chain("p2_s1", 2, 3, False, 2),
    ]))
    out = detection_rate_by_subject_count(chains).set_index("n")
    assert out.loc[2, "rate"] == 1.0
    assert out.loc[3, "rate"] == 0.0


def test_undercount_failure_share_flags_exact_undercount_as_explained():
    chains = summarize_chains(_manifest([
        _chain("p0_s1", 0, 3, False, 2),
        _chain("p1_s1", 1, 3, False, 3),
    ]))
    assert undercount_failure_share(chains) == 0.5


def test_undercount_failure_share_is_one_when_nothing_failed():
    chains = summarize_chains(_manifest([_chain("p0_s1", 0, 2, True, 2)]))
    assert undercount_failure_share(chains) == 1.0
