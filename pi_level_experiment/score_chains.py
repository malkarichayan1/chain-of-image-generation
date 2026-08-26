#!/usr/bin/env python
"""
Stage 3 of the Part-B strengthening design (docs/part-b-strengthening-design.md):
threshold -> delta masks -> IoU -> CSV. Pure numpy/scipy -- no diffusers, no torch, no
GPU -- so scoring-side iteration (a new threshold, a new control) is milliseconds, not a
Kaggle regeneration. Reads Stage 2's (segment_cache.py) cached sigmoid maps and Stage 1's
per-step attention .npy files; never calls CLIPSeg itself.

Six conditions (see design doc's Step 2 table):
  real                     -- attribute, true step, own attention
  shuffled                 -- attribute, wrong step (same chain), own attention
  substituted              -- foreign attribute, true step, own attention
  attn_scrambled_samechain -- real delta target, different attribute, same chain (legacy)
  attn_scrambled_crosschain-- real delta target, different attribute, different prompt
  attn_scrambled_sameattr  -- real delta target, same attribute, different seed, same prompt

Correctness fix (design doc, "Correctness bug to fix here"): the substituted foreign
attribute is selected by *attribute string*, disjoint from the target chain's own
attribute list -- not by chain identity, which could draw an attribute (e.g. "red apron")
that also happens to appear in the chain being scored, since MECHANISM_PROMPTS reuses a
small attribute vocabulary across chains.
"""
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import zoom

from segment_cache import load_cached_map

DEFAULT_SEED = 42
# Frozen by calibrate_threshold.py on 2026-07-23 against the first real Stage 1/2 run
# (Kaggle kernel anonymous/coig-pi-level-generate-chains v1, 19/27 chains detected).
# Sweep: np.arange(0.05, 0.96, 0.05) (calibrate_threshold.DEFAULT_SWEEP). Criterion:
# maximize real_nonzero_rate subject to substituted_nonzero_rate == 0 exactly. Fit only on
# the calibration set (CALIBRATION_PROMPT_IDS below, n_real=21 rows) -- prompts 0/1/2/5/7
# stayed held out and unopened until Step 4's confirmatory run. Selected T=0.85 as the
# smallest threshold where substituted_nonzero_rate first reaches 0 (it does not reach 0
# until T=0.85); real_nonzero_rate at that T is 0.238, notably lower than the legacy
# T=0.5's uncalibrated ~0.46 hit rate in the v4 run -- CLIPSeg's sigmoid output apparently
# carries more baseline noise on this attribute set than T=0.5 assumed. See RESULTS.md.
DEFAULT_THRESHOLD = 0.85

# Pre-registered per docs/part-b-strengthening-design.md Step 3: the four prompts skipped
# in the v4 run (RESULTS.md) are the calibration set; the five already-scored prompts stay
# held out and unopened until Step 4's confirmatory run.
CALIBRATION_PROMPT_IDS = frozenset({3, 4, 6, 8})
HELD_OUT_PROMPT_IDS = frozenset({0, 1, 2, 5, 7})


@dataclass(frozen=True)
class ChainStepRecord:
    step_idx: int
    subject: str
    attribute: str
    image_path: str
    attention_path: str


@dataclass(frozen=True)
class ChainRecord:
    chain_key: str
    prompt_id: int
    seed: int
    n: int
    prompt: str
    detected: bool
    base_image_path: str
    steps: Tuple[ChainStepRecord, ...]

    @property
    def attributes(self) -> Tuple[str, ...]:
        return tuple(s.attribute for s in self.steps)


def parse_manifest(manifest: dict) -> List[ChainRecord]:
    chains = []
    for c in manifest["chains"]:
        steps = tuple(
            ChainStepRecord(
                step_idx=s["step_idx"], subject=s["subject"], attribute=s["attribute"],
                image_path=s["image_path"], attention_path=s["attention_path"],
            )
            for s in c["steps"]
        )
        chains.append(ChainRecord(
            chain_key=c["chain_key"], prompt_id=c["prompt_id"], seed=c["seed"], n=c["n"],
            prompt=c["prompt"], detected=c["detected"], base_image_path=c["base_image_path"],
            steps=steps,
        ))
    return chains


def detected_chains(chains: List[ChainRecord]) -> List[ChainRecord]:
    return [c for c in chains if c.detected]


def chain_image_path(chain: ChainRecord, idx: int) -> str:
    """idx=0 is the base image; idx=i (i>=1) is the image produced by steps[i-1]."""
    if idx == 0:
        return chain.base_image_path
    return chain.steps[idx - 1].image_path


def load_attention_map(path: str) -> np.ndarray:
    return np.load(path)


def _resize_to(arr: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    if arr.shape == shape:
        return arr
    return zoom(arr, (shape[0] / arr.shape[0], shape[1] / arr.shape[1]), order=1)


def delta_mask_from_sigmoid(sigmoid_curr: np.ndarray, sigmoid_prev: np.ndarray, threshold: float) -> np.ndarray:
    """Ported from SpatialSemanticAlignment.phase_a_delta_mask_from_masks -- unchanged
    algorithm, just renamed to make the sigmoid-map (not raw-mask) input explicit."""
    curr_binary = sigmoid_curr > threshold
    prev_binary = sigmoid_prev > threshold
    return np.logical_and(curr_binary, np.logical_not(prev_binary)).astype(np.float32)


def iou_top_k(attention_map: np.ndarray, delta_mask: np.ndarray, threshold_pct: float = 0.20) -> float:
    """Ported from SpatialSemanticAlignment.phase_c_alignment_score (pi_level_experiment/
    run_chain_experiment.py) -- same exact top-k-by-rank binarization (not a percentile
    *value* threshold, which degenerates on sparse/peaked maps, see that fix's history).
    Duplicated here rather than imported so Stage 3 has zero torch/diffusers dependency."""
    if attention_map.shape != delta_mask.shape:
        attention_map = _resize_to(attention_map, delta_mask.shape)
    if np.max(attention_map) == np.min(attention_map):
        return 0.0
    flat = attention_map.reshape(-1)
    n = flat.size
    k = max(1, min(n, int(round(threshold_pct * n))))
    top_k_idx = np.argpartition(-flat, k - 1)[:k]
    attn_bin = np.zeros(n, dtype=bool)
    attn_bin[top_k_idx] = True
    attn_bin = attn_bin.reshape(attention_map.shape)
    delta_bin = delta_mask > 0.5
    intersection = np.logical_and(attn_bin, delta_bin).sum()
    union = np.logical_or(attn_bin, delta_bin).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def select_foreign_attribute(chain: ChainRecord, all_chains: List[ChainRecord], rng: random.Random) -> str:
    """Selects an attribute string guaranteed absent from `chain`'s own attribute list --
    the Step 2 correctness fix. Attributes repeat across MECHANISM_PROMPTS (e.g. chain 0
    and chain 3 both contain "red apron"), so drawing by chain identity alone (the v4
    behavior) can select an attribute that is genuinely present in the chain being scored,
    silently corrupting a condition meant to be structurally zero."""
    own = set(chain.attributes)
    universe = sorted({a for c in all_chains for a in c.attributes})
    candidates = [a for a in universe if a not in own]
    if not candidates:
        raise ValueError(f"no foreign attribute disjoint from chain {chain.chain_key}'s own attributes")
    chosen = rng.choice(candidates)
    assert chosen not in own, "substituted attribute must never appear in the target chain's own attributes"
    return chosen


def score_chains(manifest: dict, cache_dir: Path, threshold: float = DEFAULT_THRESHOLD,
                  seed: int = DEFAULT_SEED, threshold_pct: float = 0.20) -> pd.DataFrame:
    rng = random.Random(seed)
    # Separate RNG stream for the Step 6 attention-ablation baseline (docs/
    # part-c-validation-design.md) -- kept independent of `rng` above, which
    # substituted/attn_scrambled_* already consume via rng.choice. Drawing the ablation
    # map from `rng` itself would shift every subsequent rng.choice call, silently
    # changing which foreign attribute / scrambled partner gets picked at a given seed
    # and invalidating reproducibility of the already-published growth-run numbers.
    ablation_rng = np.random.RandomState(seed)
    all_chains = detected_chains(parse_manifest(manifest))
    rows = []

    def sigmoid(image_path: str, attribute: str) -> np.ndarray:
        cached = load_cached_map(cache_dir, image_path, attribute)
        if cached is None:
            raise ValueError(
                f"segmentation cache missing for ({image_path!r}, {attribute!r}) -- "
                "run segment_cache.build_segmentation_cache first"
            )
        return cached

    def random_attn_like(reference: np.ndarray) -> np.ndarray:
        return ablation_rng.rand(*reference.shape).astype(np.float32)

    def add_row(condition: str, chain: ChainRecord, step: ChainStepRecord, attribute: str,
                claimed_step: int, true_step_idx: int, attn_map: np.ndarray,
                curr_mask: np.ndarray, prev_mask: np.ndarray, delta_mask: np.ndarray) -> None:
        rows.append(dict(
            condition=condition, chain_key=chain.chain_key, prompt_id=chain.prompt_id,
            seed=chain.seed, n=chain.n, attribute=attribute, subject=step.subject,
            claimed_step=claimed_step, true_step=true_step_idx,
            iou=iou_top_k(attn_map, delta_mask, threshold_pct=threshold_pct),
            iou_random_attn=iou_top_k(random_attn_like(attn_map), delta_mask,
                                       threshold_pct=threshold_pct),
            curr_mask_area=float(curr_mask.mean()), prev_mask_area=float(prev_mask.mean()),
            delta_area=float(delta_mask.mean()),
        ))

    for chain in all_chains:
        k = len(chain.steps)
        for i, step in enumerate(chain.steps):
            true_step_idx = i + 1
            curr_path = chain_image_path(chain, true_step_idx)
            prev_path = chain_image_path(chain, true_step_idx - 1)
            attn = load_attention_map(step.attention_path)

            curr_mask_real = sigmoid(curr_path, step.attribute)
            prev_mask_real = sigmoid(prev_path, step.attribute)
            delta_real = delta_mask_from_sigmoid(curr_mask_real, prev_mask_real, threshold)
            add_row("real", chain, step, step.attribute, true_step_idx, true_step_idx,
                    attn, curr_mask_real, prev_mask_real, delta_real)

            for wrong_step in range(1, k + 1):
                if wrong_step == true_step_idx:
                    continue
                w_curr = chain_image_path(chain, wrong_step)
                w_prev = chain_image_path(chain, wrong_step - 1)
                curr_mask_shuf = sigmoid(w_curr, step.attribute)
                prev_mask_shuf = sigmoid(w_prev, step.attribute)
                delta_shuf = delta_mask_from_sigmoid(curr_mask_shuf, prev_mask_shuf, threshold)
                add_row("shuffled", chain, step, step.attribute, wrong_step, true_step_idx,
                        attn, curr_mask_shuf, prev_mask_shuf, delta_shuf)

            foreign_attr = select_foreign_attribute(chain, all_chains, rng)
            curr_mask_sub = sigmoid(curr_path, foreign_attr)
            prev_mask_sub = sigmoid(prev_path, foreign_attr)
            delta_sub = delta_mask_from_sigmoid(curr_mask_sub, prev_mask_sub, threshold)
            add_row("substituted", chain, step, foreign_attr, true_step_idx, true_step_idx,
                    attn, curr_mask_sub, prev_mask_sub, delta_sub)

            samechain = [s for s in chain.steps if s.attribute != step.attribute]
            if samechain:
                wrong = rng.choice(samechain)
                add_row("attn_scrambled_samechain", chain, step, step.attribute, true_step_idx,
                        true_step_idx, load_attention_map(wrong.attention_path),
                        curr_mask_real, prev_mask_real, delta_real)

            crosschain = [(c, s) for c in all_chains if c.prompt_id != chain.prompt_id
                          for s in c.steps if s.attribute != step.attribute]
            if crosschain:
                _, wrong = rng.choice(crosschain)
                add_row("attn_scrambled_crosschain", chain, step, step.attribute, true_step_idx,
                        true_step_idx, load_attention_map(wrong.attention_path),
                        curr_mask_real, prev_mask_real, delta_real)

            sameattr = [(c, s) for c in all_chains if c.prompt_id == chain.prompt_id and c.seed != chain.seed
                        for s in c.steps if s.attribute == step.attribute]
            if sameattr:
                _, wrong = rng.choice(sameattr)
                add_row("attn_scrambled_sameattr", chain, step, step.attribute, true_step_idx,
                        true_step_idx, load_attention_map(wrong.attention_path),
                        curr_mask_real, prev_mask_real, delta_real)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys

    manifest_path = sys.argv[1] if len(sys.argv) > 1 else "artifacts/manifest.json"
    cache_dir_arg = sys.argv[2] if len(sys.argv) > 2 else "artifacts/segmentation_cache"
    out_csv = sys.argv[3] if len(sys.argv) > 3 else "artifacts/chain_experiment_results.csv"

    with open(manifest_path) as f:
        manifest_data = json.load(f)
    df = score_chains(manifest_data, Path(cache_dir_arg))
    df.to_csv(out_csv, index=False)
    print(f"Scored {len(df)} rows -> {out_csv}")
    print(df.groupby("condition")["iou"].agg(["mean", "count"]).round(4))
