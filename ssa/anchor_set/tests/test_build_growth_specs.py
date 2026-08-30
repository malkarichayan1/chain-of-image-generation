"""Unit tests for build_growth_specs.py -- the deterministic enumerator that grows the
metric-A anchor set toward ~300 raw / ~150 effective judgments WITHOUT expanding the
controlled vocabulary (so the growth batch introduces no subject/attribute confound).

Design under test (see build_growth_specs.py docstring):
  * Each growth spec is ONE image = one (natural-pairing combo, pinned seed). Two specs may
    share a prompt string but never a (prompt, seed) pair, so no two growth images collide.
  * prompt_ids are unique and start at 100 (disjoint from the frozen base ids 0-23), so a
    merged manifest never has a duplicate prompt_id (merge_anchor_manifests refuses those).
  * Growth seeds are DISJOINT from the base script's CANDIDATE_SEEDS, so a growth image can
    never accidentally equal a base image even when it reuses a base subject combo.

Run from inside ssa/anchor_set/:  py -3 -m pytest tests/
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import anchor_common as ac
import build_growth_specs as bgs

BASE_SEEDS = {42, 7, 1234, 2024}  # generate_anchor_images_sdxl.py CANDIDATE_SEEDS -- must stay disjoint
KNOWN_SUBJECTS = {"barista", "chef", "cyclist", "farmer", "nurse", "pilot", "teacher"}
KNOWN_ATTRS = {"red apron", "white hat", "pan", "yellow helmet", "shovel",
               "blue gloves", "dark sunglasses", "book"}


def test_default_targets_hit_per_stratum_counts():
    specs = bgs.build_growth_specs()
    by_n = {n: sum(1 for s in specs if s["n"] == n) for n in (2, 3, 4)}
    assert by_n == bgs.DEFAULT_TARGETS


def test_specs_pass_the_shared_validator():
    # validate_specs enforces distinct subjects/attrs, attribute-substring-of-prompt, and
    # stratification -- the growth batch must satisfy the exact same invariants as the base.
    ac.validate_specs(bgs.build_growth_specs())  # must not raise


def test_ids_are_unique_contiguous_and_start_at_100():
    ids = [s["id"] for s in bgs.build_growth_specs()]
    assert ids == list(range(100, 100 + len(ids)))


def test_every_spec_pins_a_seed_from_a_disjoint_pool():
    specs = bgs.build_growth_specs()
    assert all("seed" in s for s in specs)
    used = {s["seed"] for s in specs}
    assert used <= set(bgs.GROWTH_SEED_POOL)
    assert used.isdisjoint(BASE_SEEDS)


def test_no_two_specs_share_a_prompt_and_seed():
    specs = bgs.build_growth_specs()
    keys = [(s["prompt"], s["seed"]) for s in specs]
    assert len(keys) == len(set(keys)), "duplicate (prompt, seed) -> two identical images"


def test_vocabulary_is_locked_to_the_base_set():
    for s in bgs.build_growth_specs():
        for subject, attribute in s["pairs"]:
            assert subject in KNOWN_SUBJECTS
            assert attribute in KNOWN_ATTRS


def test_prompt_phrasing_matches_the_existing_style():
    # A known n=2 combo must render byte-identical to how the base set phrases it, so token
    # lookup (token_indices) stays in the validated regime.
    combo = (("barista", "red apron", "wearing a red apron"),
             ("cyclist", "yellow helmet", "wearing a yellow helmet"))
    assert bgs.combo_prompt(combo) == (
        "a photo of a barista wearing a red apron and a cyclist wearing a yellow helmet")


def test_n3_uses_oxford_comma_join():
    combo = (("barista", "red apron", "wearing a red apron"),
             ("chef", "white hat", "wearing a white hat"),
             ("nurse", "blue gloves", "wearing blue gloves"))
    assert bgs.combo_prompt(combo) == (
        "a photo of a barista wearing a red apron, a chef wearing a white hat, "
        "and a nurse wearing blue gloves")


def test_enumerate_combos_counts_expand_chef_dual_attribute():
    # chef carries two attributes (white hat / pan); every subset containing chef doubles.
    assert len(bgs.enumerate_combos(2)) == 27   # C(7,2)=21 subsets, 6 contain chef -> +6
    assert len(bgs.enumerate_combos(3)) == 50   # C(7,3)=35, 15 contain chef -> +15
    assert len(bgs.enumerate_combos(4)) == 55   # C(7,4)=35, 20 contain chef -> +20


def test_generation_is_deterministic():
    assert bgs.build_growth_specs() == bgs.build_growth_specs()


# --------------------------------------------------------------------------- n=4 backfill
#
# The 2026-07-25 growth run's pinned-seed, no-retry design (build_growth_specs above) hit
# only 22.2% detection at n=4 (4/18), far below n=2 (85.7%) and n=3 (72.0%) -- four people
# in one 1024px frame is much harder for SDXL/Mask R-CNN, and single-shot generation has no
# safety net for that. build_n4_backfill_specs backfills with NEW (unused) n=4 combos, each
# given a small SEED LIST for generate_and_score to retry across -- safe without any
# per-spec seed bookkeeping because these combos never appear anywhere else in the project,
# so no (prompt, seed) pair can collide no matter how much the retry seeds overlap with
# seeds used elsewhere.

def test_backfill_specs_use_combos_not_already_consumed_by_growth_n4():
    growth_n4_prompts = {s["prompt"] for s in bgs.build_growth_specs() if s["n"] == 4}
    backfill_prompts = {s["prompt"] for s in bgs.build_n4_backfill_specs(count=28)}
    assert growth_n4_prompts.isdisjoint(backfill_prompts)


def test_backfill_specs_ids_start_at_200_and_are_contiguous():
    specs = bgs.build_n4_backfill_specs(count=28)
    assert [s["id"] for s in specs] == list(range(200, 228))


def test_backfill_specs_carry_a_seed_list_not_a_single_seed():
    for s in bgs.build_n4_backfill_specs(count=5):
        assert "seed" not in s
        assert s["seeds"] == bgs.BACKFILL_RETRY_SEEDS


def test_backfill_specs_are_all_n4_and_have_valid_structure():
    specs = bgs.build_n4_backfill_specs(count=28)
    assert all(s["n"] == 4 for s in specs)
    ids = [s["id"] for s in specs]
    assert len(ids) == len(set(ids))
    for s in specs:
        subjects = [p[0] for p in s["pairs"]]
        attributes = [p[1] for p in s["pairs"]]
        assert len(set(subjects)) == len(subjects) == 4
        assert len(set(attributes)) == len(attributes)
        for _subject, attribute in s["pairs"]:
            assert attribute in s["prompt"]


def test_backfill_raises_if_more_combos_requested_than_available():
    with pytest.raises(ValueError, match="unused n=4 combos"):
        bgs.build_n4_backfill_specs(count=999)
