"""
Tests for lock_confound_analysis.py (E1 + E2). Synthetic sigmoid maps written straight to
a tmp cache dir -- no CLIPSeg, no GPU, matching the rest of the Stage 2/3 test suite.

The load-bearing assertions are the ones that would catch a silently-wrong result rather
than a crash: that presence is read from the calibrated threshold (not mean area), that a
locked attribute is counted as persisting, and above all that claim direction is assigned
by comparing claimed_step to true_step in the right direction -- flipping that would
silently swap the "lock scenario" and "no lock possible" subsets and invert E2's entire
conclusion while still producing a plausible-looking table.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lock_confound_analysis import (
    EARLY,
    LATE,
    auroc,
    claim_direction_comparison,
    clustered_p_value,
    label_claim_direction,
    lock_strength,
    presence_profile,
)
from score_chains import DEFAULT_THRESHOLD, parse_manifest
from segment_cache import save_cached_map


def _chain_dict(chain_key="p0_s42", prompt_id=0, n_steps=2, detected=True):
    return {
        "chain_key": chain_key, "prompt_id": prompt_id, "seed": 42, "n": 2,
        "prompt": "a barista and a cyclist", "detected": detected,
        "base_image_path": f"{chain_key}_step0.png",
        "steps": [
            {
                "step_idx": i, "subject": f"subject{i}", "attribute": f"attr{i}",
                "image_path": f"{chain_key}_step{i + 1}.png",
                "attention_path": f"{chain_key}_attn{i}.npy",
            }
            for i in range(n_steps)
        ],
    }


def _write_map(cache_dir: Path, image_path: str, attribute: str, present: bool):
    """A map that is unambiguously above or below the calibrated threshold everywhere."""
    value = 0.99 if present else 0.01
    save_cached_map(cache_dir, image_path, attribute, np.full((4, 4), value, dtype=np.float32))


def _chain_from(manifest_chain):
    return parse_manifest({"chains": [manifest_chain]})[0]


class TestPresenceProfile:
    def test_reads_every_image_including_base(self, tmp_path):
        chain = _chain_from(_chain_dict(n_steps=2))
        # base absent, step1 present, step2 present -> profile has one entry per image
        _write_map(tmp_path, "p0_s42_step0.png", "attr0", False)
        _write_map(tmp_path, "p0_s42_step1.png", "attr0", True)
        _write_map(tmp_path, "p0_s42_step2.png", "attr0", True)
        assert presence_profile(chain, "attr0", tmp_path) == [False, True, True]

    def test_presence_uses_threshold_not_mean_area(self, tmp_path):
        """A map that is zero almost everywhere but spikes above T in one pixel IS present.
        Guards against regressing to a mean-area proxy -- the exact substitution that
        produced the retracted Part C finding (7)."""
        chain = _chain_from(_chain_dict(n_steps=1))
        sparse = np.zeros((4, 4), dtype=np.float32)
        sparse[0, 0] = 0.99
        save_cached_map(tmp_path, "p0_s42_step0.png", "attr0", sparse)
        _write_map(tmp_path, "p0_s42_step1.png", "attr0", False)
        assert sparse.mean() < DEFAULT_THRESHOLD  # a mean-area check would say "absent"
        assert presence_profile(chain, "attr0", tmp_path)[0] is True

    def test_raises_on_cache_miss(self, tmp_path):
        chain = _chain_from(_chain_dict(n_steps=1))
        with pytest.raises(ValueError, match="segmentation cache missing"):
            presence_profile(chain, "attr0", tmp_path)


class TestLockStrength:
    def test_locked_attribute_counts_as_persisting(self, tmp_path):
        """attr0 introduced at step 1 and still present at the final image = the lock."""
        chain_dict = _chain_dict(n_steps=2)
        for attribute, presence in [("attr0", [False, True, True]), ("attr1", [False, False, True])]:
            for idx, is_present in enumerate(presence):
                _write_map(tmp_path, f"p0_s42_step{idx}.png", attribute, is_present)
        report = lock_strength([_chain_from(chain_dict)], tmp_path)
        assert report["n_attribute_instances"] == 2
        assert report["appears_at_step"] == 1.0       # attr0 at step1, attr1 at step2
        assert report["persists_to_final"] == 1.0
        assert report["leaked_before_step"] == 0.0

    def test_persists_conditions_on_having_appeared(self, tmp_path):
        """An attribute that never rendered must not be counted in persists_to_final --
        otherwise a generator that renders nothing would score a perfect lock."""
        chain_dict = _chain_dict(n_steps=2)
        for idx, is_present in enumerate([False, False, False]):   # attr0 never appears
            _write_map(tmp_path, f"p0_s42_step{idx}.png", "attr0", is_present)
        for idx, is_present in enumerate([False, False, True]):    # attr1 appears at step2
            _write_map(tmp_path, f"p0_s42_step{idx}.png", "attr1", is_present)
        report = lock_strength([_chain_from(chain_dict)], tmp_path)
        assert report["appears_at_step"] == 0.5
        assert report["n_appeared"] == 1
        assert report["persists_to_final"] == 1.0

    def test_detects_early_leakage(self, tmp_path):
        chain_dict = _chain_dict(n_steps=2)
        for idx, is_present in enumerate([True, True, True]):   # attr0 present from base
            _write_map(tmp_path, f"p0_s42_step{idx}.png", "attr0", is_present)
        for idx, is_present in enumerate([False, False, True]):
            _write_map(tmp_path, f"p0_s42_step{idx}.png", "attr1", is_present)
        assert lock_strength([_chain_from(chain_dict)], tmp_path)["leaked_before_step"] == 0.5


class TestClaimDirection:
    def test_late_means_claimed_after_true_step(self):
        """The orientation that E2's entire conclusion rests on. LATE = claimed later than
        the attribute really appeared, so the lock has kept it visible and a presence check
        is fooled. Reversing this silently swaps the two subsets."""
        df = pd.DataFrame({"claimed_step": [3, 1], "true_step": [1, 3]})
        assert label_claim_direction(df)["claim_direction"].tolist() == [LATE, EARLY]

    def test_equal_steps_are_not_late(self):
        df = pd.DataFrame({"claimed_step": [2], "true_step": [2]})
        assert label_claim_direction(df)["claim_direction"].tolist() == [EARLY]

    def test_does_not_mutate_input(self):
        df = pd.DataFrame({"claimed_step": [3], "true_step": [1]})
        label_claim_direction(df)
        assert "claim_direction" not in df.columns


class TestAuroc:
    def test_perfect_separation_is_one(self):
        assert auroc([5.0, 6.0, 7.0], [1.0, 2.0, 3.0]) == 1.0

    def test_identical_distributions_are_half(self):
        assert auroc([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.5

    def test_empty_input_is_nan(self):
        assert np.isnan(auroc([], [1.0]))


class TestClusteredPValue:
    def test_consistent_per_prompt_ordering_is_significant(self):
        """Differences must vary as well as be positive -- an all-identical diff vector
        trips the insufficient-variation guard below, inherited from
        analyze_results.clustered_check."""
        real = pd.DataFrame({"prompt_id": range(10), "m": [1.0 + i / 10 for i in range(10)]})
        ctrl = pd.DataFrame({"prompt_id": range(10), "m": [0.0] * 10})
        assert clustered_p_value(real, ctrl, "m")["significant"] is True

    def test_identical_groups_report_insufficient_variation(self):
        """Wilcoxon cannot run on all-zero differences; must degrade to a note, not raise.
        This is the real shape of comparing a metric against a control that shares its
        images by construction."""
        frame = pd.DataFrame({"prompt_id": range(5), "m": [1.0] * 5})
        result = clustered_p_value(frame, frame, "m")
        assert np.isnan(result["p_value"])
        assert "insufficient variation" in result["note"]


class TestClaimDirectionComparison:
    def test_reports_both_subsets_per_metric(self):
        df = pd.DataFrame({
            "condition": ["real"] * 4 + ["shuffled"] * 4,
            "prompt_id": [0, 0, 1, 1] * 2,
            "claimed_step": [1, 2, 1, 2, 3, 3, 1, 1],
            "true_step": [1, 2, 1, 2, 1, 1, 3, 3],
            "curr_mask_area": [0.5, 0.5, 0.5, 0.5, 0.4, 0.4, 0.0, 0.0],
            "delta_area": [0.2, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0],
            "iou": [0.1] * 8,
        })
        results = claim_direction_comparison(df, metrics=("curr_mask_area", "delta_area"))
        assert set(results) == {"curr_mask_area", "delta_area"}
        for res in results.values():
            assert set(res["per_direction"]) == {EARLY, LATE}
            assert res["per_direction"][LATE]["n_control"] == 2

    def test_degradation_is_early_minus_late(self):
        """A presence-style metric that stays high in the locked control must show a
        POSITIVE degradation (loses signal), which is the direction E2 reports."""
        df = pd.DataFrame({
            "condition": ["real"] * 4 + ["shuffled"] * 4,
            "prompt_id": [0, 0, 1, 1] * 2,
            "claimed_step": [1, 2, 1, 2, 3, 3, 1, 1],
            "true_step": [1, 2, 1, 2, 1, 1, 3, 3],
            # real must SPREAD for AUROC to be able to fall: a LATE control that merely
            # sits below a constant real still ranks perfectly (AUROC 1.0) and would hide
            # the degradation this test exists to detect. Here the LATE control (0.8)
            # lands inside real's range, so it beats half of real's rows.
            "curr_mask_area": [0.9, 0.7, 0.9, 0.7, 0.8, 0.8, 0.0, 0.0],
        })
        res = claim_direction_comparison(df, metrics=("curr_mask_area",))["curr_mask_area"]
        assert res["per_direction"][EARLY]["auroc"] > res["per_direction"][LATE]["auroc"]
        assert res["auroc_degradation"] > 0
