#!/usr/bin/env python
"""
Deterministic enumerator for the 2026-07-25 SDXL anchor-set growth batch: grows the set
from 23 detected images (68 raw / 35 effective judgments, single annotator) toward
~300 raw / ~150 effective judgments across two annotators, WITHOUT expanding the controlled
subject/attribute vocabulary -- prompt_specs.json's own comment flags new vocabulary as a
confound, so this batch only recombines the existing 7 subjects x their existing attributes
into NEW subject combinations, the same "growth batch" strategy ids 18-23 already used.

Design: each growth spec is ONE image = one (natural-pairing combo, PINNED seed). Unlike the
base script's per-prompt seed *retry* loop (try seed 42, then 7, then 1234... until person
detection matches n), growth specs pin a single seed up front and do not retry -- retrying
would let two specs that happen to share a combo silently collide on the same seed. Seeds are
drawn from GROWTH_SEED_POOL, which is disjoint from the base script's CANDIDATE_SEEDS
(42/7/1234/2024), so a growth image can never coincide with a base image even when it reuses
a base subject combo. ids start at 100, disjoint from the frozen base ids 0-23, so
merge_anchor_manifests.py's duplicate-id check can never fire between the two batches.

Per-stratum targets (DEFAULT_TARGETS) are sized so that, ON TOP of the 23 already-detected
base images, the anchor set reaches ~108 detected images / ~300 raw judgments -- assuming
SDXL's already-observed ~90%+ detection rate, comfortably clearing the 150-effective floor
(strict pass rate on the existing 35/68 effective judgments is ~51%, so 300 raw ~= 150
effective) even after some detection failures and unclear/shared/none exclusions:
    n=2: +42 specs (8 already detected -> ~50 total)
    n=3: +25 specs (8 already detected -> ~33 total)
    n=4: +18 specs (7 already detected -> ~25 total)

Run from inside ssa/anchor_set/ to preview the batch (no GPU, no model calls):
    py -3 -c "import build_growth_specs as b, json; print(json.dumps(b.build_growth_specs(), indent=2))"
"""
from __future__ import annotations

import itertools
from typing import Dict, List, Tuple

Combo = Tuple[Tuple[str, str, str], ...]  # ((subject, attribute, phrase), ...)

# Locked to the exact vocabulary already used by prompt_specs.json / ANCHOR_PROMPTS (ids
# 0-23) -- same subjects, same attribute phrasing. `chef` is the one subject with two
# attribute variants in the existing set (white hat / pan); keeping both here (rather than
# picking one) is what lets enumerate_combos produce more combos than raw C(7,n) subsets.
SUBJECT_ORDER = ["barista", "chef", "cyclist", "farmer", "nurse", "pilot", "teacher"]
SUBJECT_ATTR_OPTIONS: Dict[str, List[Tuple[str, str]]] = {
    "barista": [("red apron", "wearing a red apron")],
    "chef": [("white hat", "wearing a white hat"), ("pan", "holding a pan")],
    "cyclist": [("yellow helmet", "wearing a yellow helmet")],
    "farmer": [("shovel", "holding a shovel")],
    "nurse": [("blue gloves", "wearing blue gloves")],
    "pilot": [("dark sunglasses", "wearing dark sunglasses")],
    "teacher": [("book", "holding a book")],
}

# Disjoint from generate_anchor_images_sdxl.py's CANDIDATE_SEEDS = [42, 7, 1234, 2024] --
# see module docstring on why disjointness matters. Sized for headroom: only the n=2
# stratum needs a second pass through its combo list (27 combos for 42 specs), so 2 pool
# entries would suffice today, but 8 leaves room for a future growth round without a code
# change here.
GROWTH_SEED_POOL = [101, 202, 303, 404, 505, 606, 707, 808]

# "+N specs" per stratum -- see module docstring for the arithmetic behind these numbers.
DEFAULT_TARGETS: Dict[int, int] = {2: 42, 3: 25, 4: 18}

GROWTH_ID_START = 100


def enumerate_combos(n: int) -> List[Combo]:
    """All distinct n-subject combos in deterministic (alphabetical subject, then
    attribute-variant) order. A "combo" pairs each chosen subject with one of its attribute
    variants -- chef's two variants mean subsets containing chef appear twice, once per
    variant, which is why this yields more than the raw C(7, n) subject subsets."""
    combos: List[Combo] = []
    for subset in itertools.combinations(SUBJECT_ORDER, n):
        variant_lists = [SUBJECT_ATTR_OPTIONS[s] for s in subset]
        for variant_combo in itertools.product(*variant_lists):
            combo = tuple((subject, attr, phrase)
                          for subject, (attr, phrase) in zip(subset, variant_combo))
            combos.append(combo)
    return combos


def combo_prompt(combo: Combo) -> str:
    """Render a combo as a prompt string in the exact phrasing style already used by
    prompt_specs.json: "a photo of a X <phrase> and a Y <phrase>" for n=2, an Oxford-comma
    list for n>=3. Matching the existing style keeps token lookup (token_indices in the
    generation script) in the already-validated regime."""
    items = [f"a {subject} {phrase}" for subject, _attr, phrase in combo]
    if len(items) == 2:
        body = f"{items[0]} and {items[1]}"
    else:
        body = ", ".join(items[:-1]) + f", and {items[-1]}"
    return f"a photo of {body}"


def _stratum_specs(n: int, target: int, id_start: int) -> List[dict]:
    """Round-robins `target` specs across enumerate_combos(n), pairing each full pass
    through the combo list with the next seed in GROWTH_SEED_POOL -- so specs spread evenly
    across combos before any (combo, seed) pair repeats (impossible until the pool is
    exhausted, which DEFAULT_TARGETS never approaches; see GROWTH_SEED_POOL's docstring)."""
    combos = enumerate_combos(n)
    specs: List[dict] = []
    combo_idx = 0
    seed_idx = 0
    while len(specs) < target:
        if combo_idx == len(combos):
            combo_idx = 0
            seed_idx += 1
        combo = combos[combo_idx]
        seed = GROWTH_SEED_POOL[seed_idx % len(GROWTH_SEED_POOL)]
        spec_id = id_start + len(specs)
        specs.append(dict(
            id=spec_id, n=n, prompt=combo_prompt(combo), seed=seed,
            pairs=[[subject, attr] for subject, attr, _phrase in combo],
        ))
        combo_idx += 1
    return specs


def build_growth_specs(targets: Dict[int, int] = None) -> List[dict]:
    """The full growth batch: one dict per image with id/n/prompt/pairs/seed, ids
    contiguous from GROWTH_ID_START, ordered by stratum (all n=2, then n=3, then n=4).
    Pure function of `targets` (defaults to DEFAULT_TARGETS) -- no I/O, no randomness beyond
    the fixed deterministic enumeration order, so two calls always agree."""
    targets = DEFAULT_TARGETS if targets is None else targets
    specs: List[dict] = []
    next_id = GROWTH_ID_START
    for n in (2, 3, 4):
        stratum = _stratum_specs(n, targets[n], next_id)
        specs.extend(stratum)
        next_id += len(stratum)
    return specs


# ---------------------------------------------------------------------------
# n=4 backfill (2026-07-25, second Kaggle round): the pinned-seed, no-retry design above
# hit only 22.2% detection at n=4 (4/18 -- see this run's actual manifest), far below n=2
# (85.7%) and n=3 (72.0%). Four people in one 1024px frame is much harder for SDXL/Mask
# R-CNN, and single-shot generation has no safety net for that. This backfills with NEW
# (unused) n=4 combos, each given a small SEED LIST for generate_and_score to retry across
# instead of one pinned seed -- reintroducing the base script's retry reliability, but only
# for the stratum that actually needed it.
# ---------------------------------------------------------------------------

BACKFILL_ID_START = 200
# Deliberately reuses values already present in GROWTH_SEED_POOL (no fresh pool needed):
# safety here comes from combo uniqueness, not seed uniqueness -- see build_n4_backfill_specs.
BACKFILL_RETRY_SEEDS = [101, 202, 303]


def build_n4_backfill_specs(count: int, id_start: int = BACKFILL_ID_START,
                            retry_seeds: List[int] = None) -> List[dict]:
    """`count` NEW n=4 specs, skipping the first 18 combos of enumerate_combos(4) already
    consumed by build_growth_specs()'s n=4 pass (ids 167-184) -- so no combo is regenerated
    under a different id. Each spec carries `seeds` (a list, not `build_growth_specs`'
    singular `seed`) for generate_and_score to retry in order until detection succeeds.

    This is safe WITHOUT tracking which seed each combo actually used: because every combo
    here is one no other spec in the project ever uses, no (prompt, seed) pair can collide
    no matter how freely retry_seeds overlaps with seeds used elsewhere -- a shared seed
    value across two DIFFERENT prompts is not a collision, only an identical
    (prompt, seed) pair is (see module docstring)."""
    retry_seeds = list(retry_seeds) if retry_seeds is not None else list(BACKFILL_RETRY_SEEDS)
    already_used = 18  # count of n=4 combos build_growth_specs()'s DEFAULT_TARGETS consumed
    combos = enumerate_combos(4)[already_used:already_used + count]
    if len(combos) < count:
        raise ValueError(
            f"only {len(combos)} unused n=4 combos available beyond the first "
            f"{already_used} (55 total exist), need {count}")
    return [
        dict(id=id_start + i, n=4, prompt=combo_prompt(combo), seeds=list(retry_seeds),
             pairs=[[subject, attr] for subject, attr, _phrase in combo])
        for i, combo in enumerate(combos)
    ]


if __name__ == "__main__":
    import json
    print(json.dumps(build_growth_specs(), indent=2))
