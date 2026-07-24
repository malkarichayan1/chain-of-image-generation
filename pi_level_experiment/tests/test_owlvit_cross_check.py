"""
Tests for owlvit_cross_check.py (Part D, check 4): pure comparison/caching logic, not the
untested OWL-ViT model-calling closure (same split as segment_cache.py's own tests).
"""
import json

import pandas as pd

from owlvit_cross_check import build_comparison_table, cache_key


def _manifest(chains):
    return {"seeds": [42], "img_size": 512, "chains": chains}


def _chain(chain_key, steps):
    return {
        "chain_key": chain_key, "prompt_id": 0, "seed": 42, "n": 1, "prompt": "x",
        "detected": True, "base_image_path": f"{chain_key}_step0.png", "steps": steps,
    }


def _step(step_idx, attribute, image_path):
    return {
        "step_idx": step_idx, "subject": "s", "attribute": attribute,
        "image_path": image_path, "attention_path": "irrelevant.npy",
    }


def test_build_comparison_table_flags_clipseg_never_detected_rows():
    manifest = _manifest([_chain("c0", [_step(1, "hat", "c0_step1.png")])])
    df = pd.DataFrame([{
        "condition": "real", "chain_key": "c0", "attribute": "hat", "true_step": 1,
        "iou": 0.0, "delta_area": 0.0,
    }])
    key = cache_key("c0_step1.png", "hat")
    out = build_comparison_table(df, manifest, {key: 0.02}, condition="real")
    assert out.loc[0, "clipseg_never_detected_in_curr"] == True  # noqa: E712
    assert out.loc[0, "owlvit_max_confidence"] == 0.02


def test_build_comparison_table_flags_genuine_positive_rows():
    manifest = _manifest([_chain("c0", [_step(1, "hat", "c0_step1.png")])])
    df = pd.DataFrame([{
        "condition": "real", "chain_key": "c0", "attribute": "hat", "true_step": 1,
        "iou": 0.3, "delta_area": 0.1,
    }])
    out = build_comparison_table(df, manifest, {}, condition="real")
    assert out.loc[0, "clipseg_never_detected_in_curr"] == False  # noqa: E712


def test_build_comparison_table_missing_score_is_none():
    manifest = _manifest([_chain("c0", [_step(1, "hat", "c0_step1.png")])])
    df = pd.DataFrame([{
        "condition": "real", "chain_key": "c0", "attribute": "hat", "true_step": 1,
        "iou": 0.0, "delta_area": 0.0,
    }])
    out = build_comparison_table(df, manifest, {}, condition="real")
    assert out.loc[0, "owlvit_max_confidence"] is None
