#!/usr/bin/env python
"""
E4: replicate lock_confound_analysis.py's E1/E2 on the REAL Gemini/CoIG chains from
Track 1's causal-relevance pilot, instead of the SD1.5 stand-in chains.

Why this matters more than another SD1.5 growth run. E2's LATE/EARLY split was derived
*after* seeing the pooled comparison go against the delta mask, which makes it a post-hoc
subgroup analysis on the SD1.5 data. These chains were generated for Track 1 long before
that split existed, so they are genuinely independent of it. They also fix E1's two
measured weaknesses at once: Track 1 scored appears_at_step=1.000 here (vs 0.300 on SD1.5,
where 70% of rows are noise-floor zeros) against persists_to_final=0.833 -- a strong lock
with almost no missing data.

The confound being tested is Track 1's own headline: persists_to_final was 0.833 for BOTH
real and shuffled chains, so a presence-style check could not tell them apart. This asks
whether a delta mask can, on exactly those images.

Two structural differences from the SD1.5 pipeline, both handled explicitly below:
  - CoIG chains have no base image; steps are 1-indexed (step_01..step_06). An attribute
    claimed at step 1 therefore has no previous frame to subtract, so its delta is
    undefined and it is excluded (see `delta_area` being NaN there). This costs no LATE
    rows -- a LATE claim is by definition claimed_step > true_step >= 1, hence >= 2.
  - Attributes are VQA question strings in the pilot conditions; CLIPSeg needs a noun
    phrase, recovered from pilot_prompts.csv's `attributes` column (see
    `load_attribute_map`).
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from score_chains import DEFAULT_THRESHOLD, delta_mask_from_sigmoid
from segment_cache import get_sigmoid_map, load_cached_map

REAL, SHUFFLED, SUBSTITUTED = "real", "shuffled", "substituted"
CONDITIONS = (REAL, SHUFFLED, SUBSTITUTED)

# Track 1's own numbers on these exact chains (pilot/causal_relevance_results.csv), scored
# by a VQA judge rather than CLIPSeg. E1 here re-measures the same two quantities with the
# segmenter, so a large gap means CLIPSeg disagrees with the judge -- itself worth knowing.
TRACK1 = {
    REAL: dict(appears_at_step=1.000, persists_to_final=0.833),
    SHUFFLED: dict(appears_at_step=0.533, persists_to_final=0.833),
    SUBSTITUTED: dict(appears_at_step=0.000, persists_to_final=0.000),
}


def coig_image_path(images_root: Path, item_index: int, step: int) -> str:
    """Steps are 1-indexed and zero-padded on disk: step_01.png .. step_06.png.

    item_index is coerced to int because a float that reaches here formats as "13.0" and
    silently points at a directory that does not exist -- a cache miss, not a crash, so
    every affected row would just vanish from the results."""
    return str(Path(images_root) / str(int(item_index)) / f"step_{step:02d}.png")


def load_attribute_map(prompts_csv: Path) -> Dict[Tuple[int, str], str]:
    """Maps (item_index, question_col) -> the bare attribute noun phrase CLIPSeg needs.

    `question_attr_N` is the Nth entry of the row's `attributes` column, which is stored as
    a stringified list of (category, attribute) tuples e.g. [('Bags', 'tote bag'), ...].
    The question text itself ("Is there a Housekeeping Staff with tote bag in hand?") is a
    full interrogative sentence and segments poorly, so it is deliberately not used."""
    prompts = pd.read_csv(prompts_csv)
    mapping: Dict[Tuple[int, str], str] = {}
    for _, row in prompts.iterrows():
        attributes = ast.literal_eval(row["attributes"])
        for i, entry in enumerate(attributes, start=1):
            # entries are (category, attribute); tolerate a bare string just in case
            attribute = entry[1] if isinstance(entry, (tuple, list)) else entry
            mapping[(int(row["item_index"]), f"question_attr_{i}")] = str(attribute)
    return mapping


def load_conditions(conditions_dir: Path) -> pd.DataFrame:
    """All three condition files as one frame, with `true_step` joined on from the real
    rows. A shuffled row's true step is the step its own attribute was really claimed at,
    which only the real condition records."""
    frames = []
    for condition in CONDITIONS:
        rows = json.loads((Path(conditions_dir) / f"{condition}.json").read_text())
        frame = pd.DataFrame(rows)
        frame["condition"] = condition
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    # concat against an empty condition file upcasts these to float64, which turns
    # item_index 13 into "13.0" downstream. Restore integer dtype before anything reads
    # them. true_step is deliberately left float -- substituted rows have no true step.
    for column in ("item_index", "claimed_step", "final_step"):
        combined[column] = combined[column].astype(int)
    true_steps = (combined[combined.condition == REAL]
                  .set_index(["item_index", "question_col"])["claimed_step"])
    combined["true_step"] = [
        true_steps.get((item, col), np.nan)
        for item, col in zip(combined.item_index, combined.question_col)
    ]
    return combined


def required_pairs(conditions: pd.DataFrame, attribute_map: Dict[Tuple[int, str], str],
                   images_root: Path) -> List[Tuple[str, str]]:
    """Every (image_path, attribute) pair any downstream computation can ask for: each
    item's full step range crossed with every attribute claimed against that item under any
    condition. Computed up front so segmentation happens in one batched pass."""
    per_item: Dict[int, set] = {}
    for row in conditions.itertuples():
        attribute = attribute_map.get((row.item_index, row.question_col))
        if attribute is None:
            continue
        per_item.setdefault(row.item_index, set()).add(attribute)
    pairs = []
    for row in conditions.drop_duplicates("item_index").itertuples():
        for step in range(1, int(row.final_step) + 1):
            image_path = coig_image_path(images_root, row.item_index, step)
            for attribute in sorted(per_item.get(row.item_index, ())):
                pairs.append((image_path, attribute))
    return pairs


def build_coig_cache(pairs: Sequence[Tuple[str, str]], cache_dir: Path,
                     segment_fn=None, progress_every: int = 50,
                     image_loader=None) -> dict:
    """Segments and caches every pair, skipping ones already on disk. Imports CLIPSeg only
    when something actually needs computing, so a fully-cached rerun needs no model.
    `image_loader` is injectable so tests never need real image files on disk."""
    from segment_cache import _default_image_loader

    missing = [p for p in pairs if load_cached_map(cache_dir, *p) is None]
    if missing and segment_fn is None:
        from segment_cache import _default_segment_fn
        segment_fn = _default_segment_fn()
    loader = image_loader or _default_image_loader
    for i, (image_path, attribute) in enumerate(missing, start=1):
        get_sigmoid_map(cache_dir, image_path, attribute, segment_fn, image_loader=loader)
        if progress_every and i % progress_every == 0:
            print(f"  segmented {i}/{len(missing)}")
    return dict(total=len(pairs), computed=len(missing), already_cached=len(pairs) - len(missing))


def score_coig(conditions: pd.DataFrame, attribute_map: Dict[Tuple[int, str], str],
               images_root: Path, cache_dir: Path,
               threshold: float = DEFAULT_THRESHOLD) -> pd.DataFrame:
    """One row per condition row, carrying the same three columns E2 compares on the SD1.5
    side: curr_mask_area (presence at the claimed step), prev_mask_area, delta_area.

    delta_area is NaN when claimed_step == 1 -- no previous frame exists, so the delta is
    undefined rather than zero. Scoring it as 0 would fabricate a perfect-looking rejection
    on rows where nothing was actually measured."""
    def sigmoid(item_index: int, step: int, attribute: str) -> Optional[np.ndarray]:
        return load_cached_map(cache_dir, coig_image_path(images_root, item_index, step), attribute)

    rows = []
    for row in conditions.itertuples():
        attribute = attribute_map.get((row.item_index, row.question_col))
        if attribute is None:
            continue
        claimed = int(row.claimed_step)
        curr = sigmoid(row.item_index, claimed, attribute)
        if curr is None:
            continue
        prev = sigmoid(row.item_index, claimed - 1, attribute) if claimed > 1 else None
        if prev is None:
            delta_area, prev_area = float("nan"), float("nan")
        else:
            delta = delta_mask_from_sigmoid(curr, prev, threshold)
            delta_area, prev_area = float(delta.mean()), float((prev > threshold).mean())
        rows.append(dict(
            condition=row.condition, item_index=int(row.item_index),
            question_col=row.question_col, attribute=attribute,
            claimed_step=claimed,
            true_step=(int(row.true_step) if not pd.isna(row.true_step) else np.nan),
            final_step=int(row.final_step),
            curr_mask_area=float((curr > threshold).mean()),
            prev_mask_area=prev_area, delta_area=delta_area,
        ))
    return pd.DataFrame(rows)


def coig_lock_strength(conditions: pd.DataFrame, attribute_map: Dict[Tuple[int, str], str],
                       images_root: Path, cache_dir: Path,
                       threshold: float = DEFAULT_THRESHOLD) -> Dict[str, dict]:
    """E1 on CoIG, per condition, so each can be read against its Track 1 counterpart."""
    def present(item_index: int, step: int, attribute: str) -> Optional[bool]:
        sigmoid = load_cached_map(cache_dir, coig_image_path(images_root, item_index, step),
                                  attribute)
        return None if sigmoid is None else bool((sigmoid > threshold).any())

    report: Dict[str, dict] = {}
    for condition in CONDITIONS:
        appears, persists = [], []
        for row in conditions[conditions.condition == condition].itertuples():
            attribute = attribute_map.get((row.item_index, row.question_col))
            if attribute is None:
                continue
            at_claimed = present(row.item_index, int(row.claimed_step), attribute)
            at_final = present(row.item_index, int(row.final_step), attribute)
            if at_claimed is None or at_final is None:
                continue
            appears.append(at_claimed)
            if at_claimed:
                persists.append(at_final)
        report[condition] = dict(
            n=len(appears),
            appears_at_step=float(np.mean(appears)) if appears else float("nan"),
            persists_to_final=float(np.mean(persists)) if persists else float("nan"),
            n_appeared=len(persists),
            track1_appears_at_step=TRACK1[condition]["appears_at_step"],
            track1_persists_to_final=TRACK1[condition]["persists_to_final"],
        )
    return report


def format_lock_strength(report: Dict[str, dict]) -> str:
    header = (f"{'condition':>12s} | {'n':>3s} | {'appears':>8s} | {'(Track1)':>9s} | "
              f"{'persists':>8s} | {'(Track1)':>9s}")
    lines = ["E4/E1 -- CLIPSeg presence on the real CoIG chains vs Track 1's VQA judge",
             header, "-" * len(header)]
    for condition, r in report.items():
        lines.append(
            f"{condition:>12s} | {r['n']:>3d} | {r['appears_at_step']:8.3f} | "
            f"{r['track1_appears_at_step']:9.3f} | {r['persists_to_final']:8.3f} | "
            f"{r['track1_persists_to_final']:9.3f}")
    return "\n".join(lines)


def main() -> None:
    import argparse

    from lock_confound_analysis import claim_direction_comparison, format_claim_direction

    repo_root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description="E4: E1+E2 replicated on real CoIG chains")
    ap.add_argument("--prompts-csv", default=str(repo_root / "pilot" / "pilot_prompts.csv"))
    ap.add_argument("--conditions-dir", default=str(repo_root / "pilot" / "conditions"))
    ap.add_argument("--images-root",
                    default=str(repo_root / "coig" / "create_images" / "multi_step_out"))
    ap.add_argument("--cache-dir", default="artifacts/coig_segmentation_cache")
    ap.add_argument("--out-csv", default="artifacts/coig_delta_mask_results.csv")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = ap.parse_args()

    attribute_map = load_attribute_map(Path(args.prompts_csv))
    conditions = load_conditions(Path(args.conditions_dir))
    images_root, cache_dir = Path(args.images_root), Path(args.cache_dir)

    pairs = required_pairs(conditions, attribute_map, images_root)
    print(f"Segmenting {len(pairs)} (image, attribute) pairs ...")
    print(f"  {build_coig_cache(pairs, cache_dir)}\n")

    print(format_lock_strength(
        coig_lock_strength(conditions, attribute_map, images_root, cache_dir, args.threshold)))
    print()

    scored = score_coig(conditions, attribute_map, images_root, cache_dir, args.threshold)
    scored.to_csv(args.out_csv, index=False)
    print(f"Scored {len(scored)} rows -> {args.out_csv}")

    usable = scored[scored.delta_area.notna()]
    dropped = len(scored) - len(usable)
    print(f"({dropped} row(s) excluded from the delta analysis: claimed at step 1, "
          f"no previous frame to subtract.)\n")
    print(format_claim_direction(claim_direction_comparison(
        usable, metrics=("curr_mask_area", "delta_area"), group_col="item_index")))


if __name__ == "__main__":
    main()
