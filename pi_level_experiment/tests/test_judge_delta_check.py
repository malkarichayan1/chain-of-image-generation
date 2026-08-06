"""
Tests for judge_delta_check.py. Synthetic evaluation_sbs_results.csv /
causal_relevance_results.csv / conditions JSON in tmp_path -- no real pilot data, no API
calls (the module makes none).

The load-bearing assertions: binary_delta forces 0 whenever curr was already true at the
previous step (the lock-immunity property this whole check exists to test), missing
lookups propagate as None rather than a guessed 0/1, and the substituted assertion
actually trips if a future data refresh breaks its "curr is always 0 here" premise instead
of silently scoring a delta with no prev call behind it.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from judge_delta_check import (
    binary_delta,
    clustered_by_item,
    load_step_lookup,
    pooled_fisher,
    score_from_lookup,
    score_substituted_from_cache,
)


def _write_sbs_csv(tmp_path: Path, rows) -> Path:
    path = tmp_path / "sbs.csv"
    pd.DataFrame(rows, columns=["item_index", "step_num", "question_type", "question",
                                "image_path", "answer", "answer_binary"]).to_csv(path, index=False)
    return path


def _sbs_row(item_index, step_num, question_type, answer_binary):
    return (item_index, step_num, question_type, "q?", "img.png",
            "yes" if answer_binary else "no", answer_binary)


def _cond_row(item_index=13, question_col="question_attr_1", claimed_step=2, final_step=6):
    return dict(item_index=item_index, question_col=question_col, question_text="q?",
                claimed_step=claimed_step, source="own_step", final_step=final_step)


def _conditions_df(*rows, condition: str = "real") -> pd.DataFrame:
    df = pd.DataFrame(list(rows))
    for column in ("item_index", "claimed_step", "final_step"):
        df[column] = df[column].astype(int)
    df["condition"] = condition
    return df


class TestBinaryDelta:
    def test_true_now_false_before_is_new_content(self):
        assert binary_delta(curr=1, prev=0) == 1

    def test_lock_forces_zero_when_already_present(self):
        """The entire point of the delta: present before AND present now = not new."""
        assert binary_delta(curr=1, prev=1) == 0

    def test_absent_now_is_zero_regardless_of_prev(self):
        assert binary_delta(curr=0, prev=1) == 0
        assert binary_delta(curr=0, prev=0) == 0

    def test_missing_lookup_propagates_as_none_not_a_guess(self):
        assert binary_delta(curr=None, prev=0) is None
        assert binary_delta(curr=1, prev=None) is None


class TestLoadStepLookup:
    def test_keys_by_item_question_step(self, tmp_path):
        csv_path = _write_sbs_csv(tmp_path, [
            _sbs_row(13, 1, "question_attr_1", 0),
            _sbs_row(13, 2, "question_attr_1", 1),
        ])
        lookup = load_step_lookup(csv_path)
        assert lookup[(13, "question_attr_1", 1)] == 0
        assert lookup[(13, "question_attr_1", 2)] == 1


class TestScoreFromLookup:
    def test_claimed_step_one_is_dropped_not_zeroed(self, tmp_path):
        """No previous frame exists at step 1 -- must be excluded, not silently scored."""
        csv_path = _write_sbs_csv(tmp_path, [_sbs_row(13, 1, "question_attr_1", 1)])
        lookup = load_step_lookup(csv_path)
        conditions = _conditions_df(_cond_row(claimed_step=1))
        assert len(score_from_lookup(conditions, lookup)) == 0

    def test_lock_scenario_yields_curr_true_delta_false(self, tmp_path):
        """Attribute present at both step 1 and its (later, false) claimed step 2 --
        exactly the LATE/lock case: a presence check (curr) is fooled, delta is not."""
        csv_path = _write_sbs_csv(tmp_path, [
            _sbs_row(13, 1, "question_attr_1", 1),
            _sbs_row(13, 2, "question_attr_1", 1),
        ])
        lookup = load_step_lookup(csv_path)
        conditions = _conditions_df(_cond_row(claimed_step=2))
        scored = score_from_lookup(conditions, lookup)
        assert scored.iloc[0].curr == 1
        assert scored.iloc[0].delta == 0

    def test_missing_step_in_lookup_yields_none_row(self, tmp_path):
        csv_path = _write_sbs_csv(tmp_path, [_sbs_row(13, 2, "question_attr_1", 1)])
        lookup = load_step_lookup(csv_path)
        conditions = _conditions_df(_cond_row(claimed_step=2))
        scored = score_from_lookup(conditions, lookup)
        assert scored.iloc[0].prev is None and scored.iloc[0].delta is None


class TestScoreSubstitutedFromCache:
    def test_uses_cached_curr_makes_no_new_call(self, tmp_path):
        cache_path = tmp_path / "cr.csv"
        pd.DataFrame([dict(condition="substituted", item_index=68, question_col="question_attr_4",
                           claimed_step=3, final_step=6, source="x", appears_at_step=0,
                           persists_to_final=0)]).to_csv(cache_path, index=False)
        conditions = _conditions_df(_cond_row(item_index=68, question_col="question_attr_4",
                                              claimed_step=3))
        scored = score_substituted_from_cache(conditions, cache_path)
        assert scored.iloc[0].curr == 0 and scored.iloc[0].delta == 0

    def test_raises_if_cached_curr_is_nonzero(self, tmp_path):
        """If a data refresh ever shows substituted appearing, this module's assumption
        that delta=0 needs no prev call breaks -- it must fail loudly, not silently."""
        cache_path = tmp_path / "cr.csv"
        pd.DataFrame([dict(condition="substituted", item_index=68, question_col="question_attr_4",
                           claimed_step=3, final_step=6, source="x", appears_at_step=1,
                           persists_to_final=0)]).to_csv(cache_path, index=False)
        conditions = _conditions_df(_cond_row(item_index=68, question_col="question_attr_4",
                                              claimed_step=3))
        with pytest.raises(AssertionError, match="curr=1"):
            score_substituted_from_cache(conditions, cache_path)


class TestPooledFisher:
    def test_real_higher_rate_gives_low_p(self):
        real = pd.Series([1] * 8 + [0] * 2)
        control = pd.Series([0] * 8 + [1] * 2)
        result = pooled_fisher(real, control, "delta")
        assert result["p_value"] < 0.05
        assert result["real_rate"] > result["control_rate"]


class TestClusteredByItem:
    def test_insufficient_variation_reported_not_raised(self):
        real = pd.DataFrame({"item_index": [1, 2], "delta": [1, 1]})
        control = pd.DataFrame({"item_index": [1, 2], "delta": [1, 1]})
        result = clustered_by_item(real, control, "delta")
        assert np.isnan(result["p_value"])
        assert "insufficient variation" in result["note"]

    def test_consistent_ordering_is_significant(self):
        real = pd.DataFrame({"item_index": range(10), "delta": [1.0] * 10})
        control = pd.DataFrame({"item_index": range(10), "delta": [i / 20 for i in range(10)]})
        result = clustered_by_item(real, control, "delta")
        assert result["p_value"] < 0.05
