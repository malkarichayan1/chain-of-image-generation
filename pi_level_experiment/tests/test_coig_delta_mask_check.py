"""
Tests for coig_delta_mask_check.py (E4). Synthetic prompts CSV, condition JSONs and
sigmoid maps in tmp_path -- no CLIPSeg, no real pilot data.

The assertions that matter are the CoIG-specific ones the SD1.5 path never exercises:
1-indexed zero-padded step filenames, the question_col -> attribute-noun-phrase mapping,
true_step being joined on from the real rows, and step-1 claims yielding NaN rather than a
fabricated zero delta.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from coig_delta_mask_check import (
    build_coig_cache,
    coig_image_path,
    coig_lock_strength,
    load_attribute_map,
    load_conditions,
    required_pairs,
    score_coig,
)
from segment_cache import load_cached_map, save_cached_map

ATTRS = "[('Bags', 'tote bag'), ('Facial Hair', 'mustache')]"


def _write_prompts(tmp_path: Path) -> Path:
    path = tmp_path / "prompts.csv"
    pd.DataFrame([
        {"item_index": 13, "attributes": ATTRS},
        {"item_index": 52, "attributes": "[('Tech', 'headphones'), ('Drink', 'water bottle')]"},
    ]).to_csv(path, index=False)
    return path


def _write_conditions(tmp_path: Path, real, shuffled, substituted=()) -> Path:
    conditions_dir = tmp_path / "conditions"
    conditions_dir.mkdir(exist_ok=True)
    for name, rows in [("real", real), ("shuffled", shuffled), ("substituted", substituted)]:
        (conditions_dir / f"{name}.json").write_text(json.dumps(list(rows)))
    return conditions_dir


def _row(item_index=13, question_col="question_attr_1", claimed_step=1, final_step=3):
    return dict(item_index=item_index, question_col=question_col,
                question_text="Is there a ...?", claimed_step=claimed_step,
                source="own_step", final_step=final_step)


def _write_map(cache_dir, images_root, item_index, step, attribute, present):
    save_cached_map(cache_dir, coig_image_path(images_root, item_index, step), attribute,
                    np.full((4, 4), 0.99 if present else 0.01, dtype=np.float32))


class TestImagePath:
    def test_steps_are_one_indexed_and_zero_padded(self):
        """CoIG writes step_01.png..step_06.png. An off-by-one or unpadded name would read
        the wrong frame (or none) while every downstream number still looks plausible."""
        assert coig_image_path(Path("root"), 13, 1).endswith(str(Path("13") / "step_01.png"))
        assert coig_image_path(Path("root"), 13, 6).endswith(str(Path("13") / "step_06.png"))


class TestAttributeMap:
    def test_question_col_n_maps_to_nth_attribute(self, tmp_path):
        mapping = load_attribute_map(_write_prompts(tmp_path))
        assert mapping[(13, "question_attr_1")] == "tote bag"
        assert mapping[(13, "question_attr_2")] == "mustache"
        assert mapping[(52, "question_attr_1")] == "headphones"

    def test_extracts_noun_phrase_not_category(self, tmp_path):
        """entries are (category, attribute); taking [0] would segment for 'Bags'."""
        assert "Bags" not in load_attribute_map(_write_prompts(tmp_path)).values()


class TestLoadConditions:
    def test_true_step_joined_from_real_rows(self, tmp_path):
        conditions_dir = _write_conditions(
            tmp_path,
            real=[_row(claimed_step=1)],
            shuffled=[_row(claimed_step=3)],   # same item+question, claimed later
        )
        combined = load_conditions(conditions_dir)
        shuffled = combined[combined.condition == "shuffled"].iloc[0]
        assert shuffled.true_step == 1 and shuffled.claimed_step == 3

    def test_substituted_has_no_true_step(self, tmp_path):
        """A substituted row's attribute belongs to a different item, so it has no true
        step in this chain -- must be NaN, not silently defaulted to the claimed step."""
        conditions_dir = _write_conditions(
            tmp_path, real=[_row()], shuffled=[],
            substituted=[_row(question_col="question_attr_2", claimed_step=2)])
        combined = load_conditions(conditions_dir)
        assert pd.isna(combined[combined.condition == "substituted"].iloc[0].true_step)


class TestRequiredPairs:
    def test_covers_every_step_for_every_claimed_attribute(self, tmp_path):
        conditions = load_conditions(_write_conditions(
            tmp_path,
            real=[_row(question_col="question_attr_1", final_step=3)],
            shuffled=[_row(question_col="question_attr_2", claimed_step=2, final_step=3)]))
        pairs = required_pairs(conditions, load_attribute_map(_write_prompts(tmp_path)),
                               tmp_path / "img")
        assert len(pairs) == 3 * 2                      # 3 steps x 2 distinct attributes
        assert len({p[1] for p in pairs}) == 2


class TestBuildCoigCache:
    def test_skips_already_cached_and_needs_no_model(self, tmp_path):
        """segment_fn stays None; if the code tried to load CLIPSeg for a fully-cached run
        this would fail on import, which is the regression worth catching."""
        images_root, cache_dir = tmp_path / "img", tmp_path / "cache"
        _write_map(cache_dir, images_root, 13, 1, "tote bag", True)
        pairs = [(coig_image_path(images_root, 13, 1), "tote bag")]
        assert build_coig_cache(pairs, cache_dir) == dict(total=1, computed=0, already_cached=1)

    def test_computes_missing_pairs_via_injected_fn(self, tmp_path):
        images_root, cache_dir = tmp_path / "img", tmp_path / "cache"
        pairs = [(coig_image_path(images_root, 13, 1), "tote bag")]
        calls = []

        def fake_segment(image, attribute):
            calls.append(attribute)
            return np.full((4, 4), 0.9, dtype=np.float32)

        stats = build_coig_cache(pairs, cache_dir, segment_fn=fake_segment,
                                 progress_every=0,
                                 image_loader=lambda path: f"fake-image:{path}")
        assert stats["computed"] == 1 and calls == ["tote bag"]
        assert load_cached_map(cache_dir, *pairs[0]) is not None


class TestScoreCoig:
    def test_step_one_claim_yields_nan_delta_not_zero(self, tmp_path):
        """No previous frame exists at step 1, so the delta is undefined. Recording 0.0
        would look like a confident correct rejection on an unmeasured row."""
        images_root, cache_dir = tmp_path / "img", tmp_path / "cache"
        _write_map(cache_dir, images_root, 13, 1, "tote bag", True)
        conditions = load_conditions(_write_conditions(
            tmp_path, real=[_row(claimed_step=1, final_step=3)], shuffled=[]))
        scored = score_coig(conditions, load_attribute_map(_write_prompts(tmp_path)),
                            images_root, cache_dir)
        assert np.isnan(scored.iloc[0].delta_area)
        assert scored.iloc[0].curr_mask_area == pytest.approx(1.0)

    def test_delta_is_new_content_only(self, tmp_path):
        """Attribute present at both the claimed step and the one before -- the lock case.
        curr_mask_area stays high (a presence check is fooled) while delta collapses to 0."""
        images_root, cache_dir = tmp_path / "img", tmp_path / "cache"
        for step in (1, 2):
            _write_map(cache_dir, images_root, 13, step, "tote bag", True)
        conditions = load_conditions(_write_conditions(
            tmp_path, real=[_row(claimed_step=2, final_step=3)], shuffled=[]))
        scored = score_coig(conditions, load_attribute_map(_write_prompts(tmp_path)),
                            images_root, cache_dir)
        assert scored.iloc[0].curr_mask_area == pytest.approx(1.0)
        assert scored.iloc[0].delta_area == pytest.approx(0.0)


class TestCoigLockStrength:
    def test_persists_conditions_on_appearing(self, tmp_path):
        images_root, cache_dir = tmp_path / "img", tmp_path / "cache"
        # attr_1 appears at its claimed step 1 and survives to final step 3
        for step in (1, 2, 3):
            _write_map(cache_dir, images_root, 13, step, "tote bag", True)
        # attr_2 never renders at all
        for step in (1, 2, 3):
            _write_map(cache_dir, images_root, 13, step, "mustache", False)
        conditions = load_conditions(_write_conditions(
            tmp_path,
            real=[_row(question_col="question_attr_1", claimed_step=1, final_step=3),
                  _row(question_col="question_attr_2", claimed_step=2, final_step=3)],
            shuffled=[]))
        report = coig_lock_strength(conditions, load_attribute_map(_write_prompts(tmp_path)),
                                    images_root, cache_dir)["real"]
        assert report["n"] == 2
        assert report["appears_at_step"] == 0.5
        assert report["n_appeared"] == 1 and report["persists_to_final"] == 1.0
