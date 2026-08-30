#!/usr/bin/env python
"""
Experiments E1 and E2: does the delta mask actually earn its place over a plain
"is the attribute present now" check?

Motivation. Scoring `delta_area` against `curr_mask_area` pooled over all shuffled rows
says curr wins (p=0.0004 vs p=0.0054) -- which would make the delta mask decorative, since
the whole argument for subtracting the previous step is immunity to the compositional lock.
That pooled comparison is misleading: it averages the lock scenario together with a case
where no lock is possible, and a presence check is excellent on the latter.

E2 splits `shuffled` by claim direction, which is what separates those two cases:
  LATE  (claimed_step > true_step) -- the attribute really appeared earlier and, under the
        lock, is STILL present at the later step it is falsely claimed at. A presence check
        sees it and is fooled. This is the lock confound.
  EARLY (claimed_step < true_step) -- the attribute has not been introduced yet, so it is
        genuinely absent. A presence check correctly rejects. No lock involved.
Only the LATE subset tests lock-immunity at all.

E1 establishes that the LATE subset is a real lock scenario in *these* chains rather than a
hypothetical: it measures how often an attribute, once rendered, survives to the final
image. Without that, "LATE" would just be a relabeling.

Both are pure numpy over Stage 2's cached sigmoid maps -- no CLIPSeg, no GPU. The cache
covers the full image x attribute cross product (segment_cache.build_segmentation_cache),
so presence at *every* step is already available, not only the scored (curr, prev) pairs.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from score_chains import (
    DEFAULT_THRESHOLD,
    ChainRecord,
    chain_image_path,
    detected_chains,
    parse_manifest,
)
from segment_cache import load_cached_map

# Track 1's causal-relevance pilot on real Gemini/CoIG chains (pilot/
# causal_relevance_results.csv, real condition) -- the reference point E1 is measured
# against. persists_to_final was identical for real and shuffled there (0.833 both), which
# is the confound this whole module exists to test a defense against.
TRACK1_GEMINI_APPEARS_AT_STEP = 1.000
TRACK1_GEMINI_PERSISTS_TO_FINAL = 0.833

LATE = "LATE"
EARLY = "EARLY"


# ---------------------------------------------------------------------------
# E1 -- lock strength
# ---------------------------------------------------------------------------

def presence_profile(chain: ChainRecord, attribute: str, cache_dir: Path,
                     threshold: float = DEFAULT_THRESHOLD) -> List[bool]:
    """Presence of `attribute` at every image in the chain, index 0 (base image) through
    K (final). Presence is "any pixel clears the calibrated threshold" rather than an area
    cutoff, matching how score_chains.delta_mask_from_sigmoid binarizes."""
    n_images = len(chain.steps) + 1
    profile = []
    for idx in range(n_images):
        image_path = chain_image_path(chain, idx)
        sigmoid = load_cached_map(cache_dir, image_path, attribute)
        if sigmoid is None:
            raise ValueError(
                f"segmentation cache missing for ({image_path!r}, {attribute!r}) -- "
                "run segment_cache.build_segmentation_cache first")
        profile.append(bool((sigmoid > threshold).any()))
    return profile


def lock_strength(chains: Sequence[ChainRecord], cache_dir: Path,
                  threshold: float = DEFAULT_THRESHOLD) -> dict:
    """E1. Three rates over every (chain, step-attribute) instance:

    appears_at_step   -- attribute is visible at the step that introduces it. Track 1's
                         Gemini chains scored 1.000; anything far below that means the
                         generator often fails to render the attribute at all, and every
                         downstream contrast is dominated by noise-floor zeros.
    persists_to_final -- GIVEN it appeared, it is still visible in the final image. This is
                         the lock. A high value here is what makes a LATE false claim
                         survive a presence check, and therefore what makes E2's LATE subset
                         a genuine test rather than a relabeling.
    leaked_before_step-- attribute is visible BEFORE the step that introduces it. Reported
                         separately because it is a different failure (early leakage) and
                         is NOT what drives the LATE subset -- that runs on persistence.
    """
    appears: List[bool] = []
    persists: List[bool] = []
    leaked: List[bool] = []
    for chain in chains:
        final_idx = len(chain.steps)
        for i, step in enumerate(chain.steps):
            claimed_idx = i + 1
            profile = presence_profile(chain, step.attribute, cache_dir, threshold)
            appears.append(profile[claimed_idx])
            if profile[claimed_idx]:
                persists.append(profile[final_idx])
            leaked.append(any(profile[:claimed_idx]))
    return dict(
        n_chains=len(chains),
        n_attribute_instances=len(appears),
        appears_at_step=float(np.mean(appears)) if appears else float("nan"),
        persists_to_final=float(np.mean(persists)) if persists else float("nan"),
        n_appeared=len(persists),
        leaked_before_step=float(np.mean(leaked)) if leaked else float("nan"),
        threshold=threshold,
    )


# ---------------------------------------------------------------------------
# E2 -- does the delta mask survive the lock scenario where a presence check does not?
# ---------------------------------------------------------------------------

def label_claim_direction(shuffled: pd.DataFrame) -> pd.DataFrame:
    """Adds `claim_direction`. Returns a copy -- callers pass in slices of a shared frame."""
    out = shuffled.copy()
    out["claim_direction"] = np.where(out.claimed_step > out.true_step, LATE, EARLY)
    return out


def auroc(positive: Sequence[float], negative: Sequence[float]) -> float:
    """P(a random `positive` row scores above a random `negative` row), ties at 0.5.

    Reported alongside p-values because the two answer different questions here and can
    disagree: delta_area is heavily zero-inflated, so per-row separation (AUROC) stays
    modest even where per-prompt means order consistently (the clustered test). Ratio of
    means is deliberately NOT used -- shuffled delta_area is exactly 0.0 on the EARLY
    subset, which makes any ratio a division-by-zero artifact rather than an effect size."""
    pos = np.asarray(positive, dtype=float)
    neg = np.asarray(negative, dtype=float)
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    u, _ = stats.mannwhitneyu(pos, neg, alternative="two-sided")
    return float(u / (pos.size * neg.size))


def clustered_p_value(real: pd.DataFrame, control: pd.DataFrame, metric: str,
                      group_col: str = "prompt_id") -> dict:
    """Per-prompt paired Wilcoxon, matching analyze_results.clustered_check -- the test the
    growth-run analysis pre-registered as the conservative one to lead with, so applying it
    here is following the existing protocol rather than picking a favourable statistic."""
    real_by_group = real.groupby(group_col)[metric].mean()
    ctrl_by_group = control.groupby(group_col)[metric].mean()
    aligned_real, aligned_ctrl = real_by_group.align(ctrl_by_group, join="inner")
    diffs = (aligned_real - aligned_ctrl).to_numpy()
    if len(diffs) < 2 or np.all(diffs == diffs[0]):
        return dict(p_value=float("nan"), n_groups=len(diffs), significant=False,
                    note="insufficient variation across prompts for Wilcoxon")
    _, p_value = stats.wilcoxon(diffs, alternative="greater")
    return dict(p_value=float(p_value), n_groups=len(diffs),
                significant=bool(p_value < 0.05), mean_diff=float(diffs.mean()))


def claim_direction_comparison(
    df: pd.DataFrame,
    metrics: Sequence[str] = ("curr_mask_area", "delta_area", "iou"),
    group_col: str = "prompt_id",
) -> Dict[str, dict]:
    """E2. For each metric, real-vs-shuffled separation computed separately on the EARLY
    (no lock possible) and LATE (lock active) subsets, by both AUROC and the clustered
    per-prompt test.

    The comparison that matters is not which metric is higher in absolute terms but how
    much each DEGRADES from EARLY to LATE: a check that relies on the attribute simply
    being present should lose most of its signal once the lock keeps it present anyway.

    `group_col` is the clustering unit for the conservative test: "prompt_id" for the SD1.5
    chains, "item_index" for the real CoIG chains."""
    real = df[df.condition == "real"]
    shuffled = label_claim_direction(df[df.condition == "shuffled"])
    results: Dict[str, dict] = {}
    for metric in metrics:
        per_direction = {}
        for direction in (EARLY, LATE):
            subset = shuffled[shuffled.claim_direction == direction]
            per_direction[direction] = dict(
                n_real=int(len(real)), n_control=int(len(subset)),
                real_mean=float(real[metric].mean()),
                control_mean=float(subset[metric].mean()),
                auroc=auroc(real[metric], subset[metric]),
                clustered=clustered_p_value(real, subset, metric, group_col=group_col),
            )
        results[metric] = dict(
            per_direction=per_direction,
            auroc_degradation=(per_direction[EARLY]["auroc"] - per_direction[LATE]["auroc"]),
        )
    return results


def format_lock_strength(report: dict) -> str:
    return "\n".join([
        f"E1 -- lock strength (T={report['threshold']}, {report['n_chains']} chains, "
        f"{report['n_attribute_instances']} attribute instances)",
        f"  appears_at_step    : {report['appears_at_step']:.3f}   "
        f"(Track 1 Gemini real = {TRACK1_GEMINI_APPEARS_AT_STEP:.3f})",
        f"  persists_to_final  : {report['persists_to_final']:.3f}   "
        f"(Track 1 Gemini real = {TRACK1_GEMINI_PERSISTS_TO_FINAL:.3f})  n={report['n_appeared']}",
        f"  leaked_before_step : {report['leaked_before_step']:.3f}",
    ])


def format_claim_direction(results: Dict[str, dict]) -> str:
    header = (f"{'metric':>16s} | {'AUROC early':>11s} | {'AUROC late':>10s} | {'degrad':>7s} | "
              f"{'clust p early':>13s} | {'clust p late':>12s}")
    lines = ["E2 -- real vs shuffled, split by claim direction", header, "-" * len(header)]
    for metric, res in results.items():
        early = res["per_direction"][EARLY]
        late = res["per_direction"][LATE]
        lines.append(
            f"{metric:>16s} | {early['auroc']:11.3f} | {late['auroc']:10.3f} | "
            f"{res['auroc_degradation']:+7.3f} | {early['clustered']['p_value']:13.4f} | "
            f"{late['clustered']['p_value']:12.4f}")
    return "\n".join(lines)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="E1 lock strength + E2 claim-direction split")
    ap.add_argument("--manifest", default="artifacts/manifest_combined_v3.json")
    ap.add_argument("--cache-dir", default="artifacts/segmentation_cache")
    ap.add_argument("--results-csv", default="artifacts/chain_experiment_results_v8.csv")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                     help="Delta-mask sigmoid threshold; defaults to score_chains.py's calibrated value")
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    chains = detected_chains(parse_manifest(manifest))
    print(format_lock_strength(lock_strength(chains, Path(args.cache_dir), args.threshold)))
    print()
    print(format_claim_direction(claim_direction_comparison(pd.read_csv(args.results_csv))))


if __name__ == "__main__":
    main()
