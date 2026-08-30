"""
Tests for discriminant_validity_check.py (Part D, check 2): is `iou` explained by subject
bounding-box size rather than anything attention-specific?
"""
import numpy as np
import pandas as pd
import pytest

from discriminant_validity_check import bbox_area_lookup, confound_report, join_bbox_area


def test_bbox_area_lookup_computes_fraction_of_image_area():
    manifest = {
        "chains": [
            {"chain_key": "p0_s42", "subject_boxes": {"barista": [0, 0, 256, 512]}},
        ]
    }
    out = bbox_area_lookup(manifest, img_size=512)
    assert out[("p0_s42", "barista")] == pytest.approx(0.5)


def test_bbox_area_lookup_skips_chains_with_no_boxes():
    manifest = {"chains": [{"chain_key": "p3_s42", "subject_boxes": {}}]}
    assert bbox_area_lookup(manifest, img_size=512) == {}


def test_join_bbox_area_matches_by_chain_key_and_subject():
    df = pd.DataFrame({
        "chain_key": ["p0_s42", "p0_s42"], "subject": ["barista", "cyclist"],
    })
    lookup = {("p0_s42", "barista"): 0.25}
    out = join_bbox_area(df, lookup)
    assert out.loc[0, "bbox_area_frac"] == 0.25
    assert pd.isna(out.loc[1, "bbox_area_frac"])


def test_confound_report_finds_no_signal_when_bbox_area_is_pure_noise():
    rng = np.random.RandomState(0)
    n = 200
    delta = rng.uniform(0, 1, n)
    iou = delta * 0.5 + rng.normal(0, 0.01, n)
    bbox = rng.uniform(0, 1, n)  # independent of both iou and delta_area
    df = pd.DataFrame({
        "condition": ["real"] * n, "chain_key": [f"c{i}" for i in range(n)],
        "subject": ["s"] * n, "iou": iou, "delta_area": delta, "bbox_area_frac": bbox,
    })
    report = confound_report(df, condition="real")
    assert report["n"] == n
    assert abs(report["partial_r_controlling_delta_area"]) < 0.2


def test_confound_report_detects_a_real_bbox_confound():
    rng = np.random.RandomState(1)
    n = 200
    delta = rng.uniform(0, 1, n)
    bbox = rng.uniform(0, 1, n)
    iou = 0.7 * bbox + rng.normal(0, 0.01, n)  # iou driven by bbox, not delta_area
    df = pd.DataFrame({
        "condition": ["real"] * n, "chain_key": [f"c{i}" for i in range(n)],
        "subject": ["s"] * n, "iou": iou, "delta_area": delta, "bbox_area_frac": bbox,
    })
    report = confound_report(df, condition="real")
    assert report["partial_r_controlling_delta_area"] > 0.8
    assert report["partial_p_controlling_delta_area"] < 0.001
