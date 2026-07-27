#!/usr/bin/env python
"""
Writes a synthetic artifacts_dummy/ directory (manifest.json + labels_dummy.json +
counts_dummy.json) shaped exactly like a real artifacts_sdxl/ run, so all five Part A
experiment scripts and run_five_experiments.py can be smoke-tested end-to-end BEFORE
Workstream 2's real, second-annotator-validated labels land. See
docs/part-a-five-experiment-battery-design.md.

Every number in here is SYNTHETIC and deliberately disclosed as such -- this is scaffolding
data, not a result. Reuses the real controlled vocabulary (7 subjects, prompt_specs.json's
phrasing) so exp4_positional_baseline.py's real prompt-text logic gets exercised meaningfully,
not against a fake templated string.

Sized to match the design doc's real proportions (~20/15/20 detected items per n=2/3/4) so
exp3_attention_scramble.py's cross-item donor requirement (>= 2 items per stratum) is genuinely
exercised, not accidentally skipped. Every attribute carries model_scores_full/
predicted_owner_full so exp2_window_ablation.py has something to compare, unlike the current
real artifacts_sdxl/manifest.json (which predates that field).

Run:
    py -3 make_dummy_artifacts.py --out artifacts_dummy --seed 0
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

from anchor_common import (
    COUNT_BROKEN, COUNT_CLEAN, LABEL_NONE, LABEL_SHARED, LABEL_UNCLEAR, count_key, label_key,
)

# Same controlled vocabulary as prompt_specs.json -- 7 distinct (subject, verb, attribute)
# triples, so sampling without replacement always yields distinct subjects/attributes.
POOL: List[Tuple[str, str, str]] = [
    ("barista", "wearing", "red apron"),
    ("cyclist", "wearing", "yellow helmet"),
    ("chef", "wearing", "white hat"),
    ("farmer", "holding", "shovel"),
    ("nurse", "wearing", "blue gloves"),
    ("pilot", "wearing", "dark sunglasses"),
    ("teacher", "holding", "book"),
]

ITEMS_PER_STRATUM: Dict[int, int] = {2: 20, 3: 15, 4: 20}
P_BROKEN: Dict[int, float] = {2: 0.05, 3: 0.10, 4: 0.35}  # count-broken skews toward n=4, matching real data
P_EARLY_CORRECT = 0.65   # early-window model_scores' argmax matches intended_subject
P_FULL_CORRECT = 0.50    # full-trajectory diluting the signal is the project's own hypothesis
P_HUMAN_MATCHES_INTENDED = 0.75  # remaining 0.25 splits across other-subject/shared/none/unclear


def _build_prompt(combo: List[Tuple[str, str, str]]) -> str:
    """Matches prompt_specs.json's real phrasing: "a photo of a X verb a Y and a Z verb a W"
    for n=2, comma-separated with ", and" before the last for n=3/4."""
    phrases = [f"a {subj} {verb} a {attr}" for subj, verb, attr in combo]
    body = (f"{phrases[0]} and {phrases[1]}" if len(phrases) == 2
            else ", ".join(phrases[:-1]) + ", and " + phrases[-1])
    return f"a photo of {body}"


def _scored_dict(subjects: List[str], winner: str, rng: random.Random) -> Dict[str, float]:
    """A model_scores-shaped dict where `winner` strictly holds the highest value."""
    winner_val = rng.uniform(0.5, 0.9)
    return {s: (winner_val if s == winner else rng.uniform(0.0, winner_val * 0.9))
            for s in subjects}


def _build_image(prompt_id: int, n: int, rng: random.Random) -> dict:
    combo = rng.sample(POOL, n)
    subjects = [s for s, _, _ in combo]
    attributes = []
    for subj, _, attr in combo:
        others = [s for s in subjects if s != subj]
        early_winner = subj if rng.random() < P_EARLY_CORRECT else rng.choice(others)
        full_winner = subj if rng.random() < P_FULL_CORRECT else rng.choice(others)
        attributes.append(dict(
            attribute=attr, intended_subject=subj,
            predicted_owner=early_winner, model_scores=_scored_dict(subjects, early_winner, rng),
            predicted_owner_full=full_winner, model_scores_full=_scored_dict(subjects, full_winner, rng),
        ))
    return dict(
        prompt_id=prompt_id, n=n, prompt=_build_prompt(combo), subjects=subjects, seed=prompt_id,
        detected=True, num_people_detected=n, image_path=f"artifacts/images/p{prompt_id}.png",
        attributes=attributes,
    )


def build_dummy_artifacts(seed: int = 0) -> Dict[str, dict]:
    """Returns {"manifest": ..., "labels": ..., "counts": ...} -- pure, no disk I/O, so tests
    can assert on structure directly."""
    rng = random.Random(seed)
    images = []
    prompt_id = 0
    for n, count in ITEMS_PER_STRATUM.items():
        for _ in range(count):
            images.append(_build_image(prompt_id, n, rng))
            prompt_id += 1

    labels: Dict[str, str] = {}
    counts: Dict[str, str] = {}
    for img in images:
        counts[count_key(img["prompt_id"])] = (
            COUNT_BROKEN if rng.random() < P_BROKEN[img["n"]] else COUNT_CLEAN)
        for attr in img["attributes"]:
            key = label_key(img["prompt_id"], attr["attribute"])
            roll = rng.random()
            if roll < P_HUMAN_MATCHES_INTENDED:
                labels[key] = attr["intended_subject"]
            else:
                remainder = roll - P_HUMAN_MATCHES_INTENDED
                others = [s for s in img["subjects"] if s != attr["intended_subject"]]
                if remainder < 0.10:
                    labels[key] = rng.choice(others)
                elif remainder < 0.16:
                    labels[key] = LABEL_SHARED
                elif remainder < 0.21:
                    labels[key] = LABEL_NONE
                else:
                    labels[key] = LABEL_UNCLEAR

    manifest = dict(model="dummy", candidate_seeds=[], img_size=1024, num_inference_steps=30,
                    early_window_fraction=0.5, images=images)
    return dict(manifest=manifest, labels=labels, counts=counts)


def write_dummy_artifacts(out_dir: Path, seed: int = 0) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = build_dummy_artifacts(seed=seed)
    (out_dir / "manifest.json").write_text(json.dumps(data["manifest"], indent=2))
    (out_dir / "labels_dummy.json").write_text(json.dumps(data["labels"], indent=2, sort_keys=True))
    (out_dir / "counts_dummy.json").write_text(json.dumps(data["counts"], indent=2, sort_keys=True))
    print(f"Wrote {len(data['manifest']['images'])} synthetic images, {len(data['labels'])} "
          f"labels, {len(data['counts'])} count judgments to {out_dir}/ "
          f"(SYNTHETIC DATA -- not a real result; see docs/part-a-five-experiment-battery-design.md)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Write a synthetic artifacts_dummy/ fixture")
    ap.add_argument("--out", default="artifacts_dummy")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    write_dummy_artifacts(Path(args.out), seed=args.seed)


if __name__ == "__main__":
    main()
