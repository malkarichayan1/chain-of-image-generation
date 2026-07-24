"""
Tests for discriminant_validity_check.py: does metric A's `predicted_owner` track subject
box size/position rather than attention content? CLAUDE.md flags this as unresolved
("cannot yet be separated from a subject-box-overlap artifact -- the manifests do not
store boxes") -- recompute_boxes.py closes the missing-boxes gap; this is the analysis
that reads them.
"""
import numpy as np
import pandas as pd
import pytest

from discriminant_validity_check import (
    bbox_area, bigbox_baseline_report, bigbox_win_rate_report, intended_bigbox_rate_report,
    join_boxes, margin_area_correlation, mcnemar_report,
)


def test_bbox_area_computes_pixel_area():
    assert bbox_area([0, 0, 10, 20]) == 200.0


def test_bbox_area_clamps_degenerate_box_to_zero():
    assert bbox_area([10, 10, 5, 5]) == 0.0


def _row(prompt_id, n, predicted_owner, intended_subject, human_label, scored, correct,
         margin=0.5):
    return dict(prompt_id=prompt_id, n=n, predicted_owner=predicted_owner,
                intended_subject=intended_subject, human_label=human_label,
                scored=scored, correct=correct, margin=margin)


def test_join_boxes_adds_bigbox_and_area_columns():
    rows = [_row(0, 2, "barista", "barista", "barista", True, True)]
    boxes = {0: {"barista": [0, 0, 100, 100], "cyclist": [0, 0, 10, 10]}}
    df = join_boxes(rows, boxes, img_size=100)
    assert df.loc[0, "bigbox_subject"] == "barista"
    assert df.loc[0, "predicted_matches_bigbox"] == True  # noqa: E712
    assert df.loc[0, "intended_matches_bigbox"] == True  # noqa: E712
    assert df.loc[0, "predicted_area_frac"] == pytest.approx(1.0)


def test_join_boxes_leaves_area_columns_null_when_prompt_has_no_boxes():
    rows = [_row(5, 2, "barista", "barista", "barista", True, True)]
    df = join_boxes(rows, {}, img_size=100)
    assert pd.isna(df.loc[0, "bigbox_subject"])
    assert pd.isna(df.loc[0, "area_advantage"])


def test_bigbox_baseline_report_matches_metric_when_predicted_always_equals_bigbox():
    rows = [_row(i, 2, "barista", "barista", "barista", True, True) for i in range(5)]
    boxes = {i: {"barista": [0, 0, 100, 100], "cyclist": [0, 0, 10, 10]} for i in range(5)}
    df = join_boxes(rows, boxes, img_size=100)
    report = bigbox_baseline_report(df)
    assert report["n"] == 5
    assert report["metric_accuracy"] == 1.0
    assert report["bigbox_baseline_accuracy"] == 1.0


def test_bigbox_baseline_report_separates_metric_from_baseline_when_they_disagree():
    # metric is always correct; the biggest box is never the right answer.
    rows = [_row(i, 2, "cyclist", "cyclist", "cyclist", True, True) for i in range(5)]
    boxes = {i: {"barista": [0, 0, 100, 100], "cyclist": [0, 0, 10, 10]} for i in range(5)}
    df = join_boxes(rows, boxes, img_size=100)
    report = bigbox_baseline_report(df)
    assert report["metric_accuracy"] == 1.0
    assert report["bigbox_baseline_accuracy"] == 0.0


def test_mcnemar_report_significant_when_metric_beats_baseline_on_discordant_pairs():
    # metric correct + baseline wrong every time -> maximally discordant in metric's favor.
    rows = [_row(i, 2, "cyclist", "cyclist", "cyclist", True, True) for i in range(10)]
    boxes = {i: {"barista": [0, 0, 100, 100], "cyclist": [0, 0, 10, 10]} for i in range(10)}
    df = join_boxes(rows, boxes, img_size=100)
    report = mcnemar_report(df)
    assert report["metric_only_correct"] == 10
    assert report["baseline_only_correct"] == 0
    assert report["p_value"] < 0.01


def test_bigbox_win_rate_report_at_chance_when_predicted_owner_is_independent_of_area():
    rng = np.random.RandomState(0)
    rows, boxes = [], {}
    for i in range(200):
        big = "barista" if rng.rand() < 0.5 else "cyclist"
        small = "cyclist" if big == "barista" else "barista"
        boxes[i] = {big: [0, 0, 100, 100], small: [0, 0, 10, 10]}
        predicted = "barista" if rng.rand() < 0.5 else "cyclist"  # independent of box size
        rows.append(_row(i, 2, predicted, "barista", "barista", True, predicted == "barista"))
    df = join_boxes(rows, boxes, img_size=100)
    report = bigbox_win_rate_report(df)
    assert abs(report["rate"] - 0.5) < 0.15


def test_bigbox_win_rate_report_detects_a_real_confound():
    rows = [_row(i, 2, "barista", "cyclist", "barista", True, False) for i in range(20)]
    boxes = {i: {"barista": [0, 0, 100, 100], "cyclist": [0, 0, 10, 10]} for i in range(20)}
    df = join_boxes(rows, boxes, img_size=100)
    report = bigbox_win_rate_report(df)
    assert report["rate"] == 1.0
    assert report["p_value"] < 0.001


def test_intended_bigbox_rate_report_flags_a_construct_validity_confound():
    # every prompt's INTENDED subject also happens to be the biggest box -- a
    # prompt/generation-level confound, independent of the metric's own predictions.
    rows = [_row(i, 2, "cyclist", "barista", "barista", True, False) for i in range(20)]
    boxes = {i: {"barista": [0, 0, 100, 100], "cyclist": [0, 0, 10, 10]} for i in range(20)}
    df = join_boxes(rows, boxes, img_size=100)
    report = intended_bigbox_rate_report(df)
    assert report["rate"] == 1.0
    assert report["p_value"] < 0.001


def test_margin_area_correlation_near_zero_when_independent():
    rng = np.random.RandomState(1)
    rows, boxes = [], {}
    for i in range(200):
        margin = rng.uniform(0, 1)
        boxes[i] = {"barista": [0, 0, rng.uniform(10, 100), 100], "cyclist": [0, 0, 50, 100]}
        rows.append(_row(i, 2, "barista", "barista", "barista", True, True, margin=margin))
    df = join_boxes(rows, boxes, img_size=100)
    report = margin_area_correlation(df)
    assert abs(report["spearman_r"]) < 0.2


def test_margin_area_correlation_detects_a_real_relationship():
    rows, boxes = [], {}
    for i in range(20):
        area = 10 + i * 4  # increasing predicted-subject box width
        boxes[i] = {"barista": [0, 0, area, 100], "cyclist": [0, 0, 10, 100]}
        rows.append(_row(i, 2, "barista", "barista", "barista", True, True, margin=i / 20.0))
    df = join_boxes(rows, boxes, img_size=100)
    report = margin_area_correlation(df)
    assert report["spearman_r"] > 0.8
