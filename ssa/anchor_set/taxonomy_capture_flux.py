#!/usr/bin/env python
"""
Taxonomy attention capture for FLUX.1-dev (#14, #16, #17, #18).
Runs on GPU: regenerates anchor images with pinned seeds, intercepts fine-grained cross-attention
across layer-bands (early/mid/late), individual heads, and timestep quartiles.

Also verifies reproduction quality:
- repro_mean_abs_pixel_diff < 1e-3
- pooled_owner_matches_manifest == 100%

Usage:
    # Smoke test on 3 images first
    python taxonomy_capture_flux.py --artifacts-dir artifacts_flux --limit 3

    # Full runs
    python taxonomy_capture_flux.py --artifacts-dir artifacts_flux
    python taxonomy_capture_flux.py --artifacts-dir artifacts_flux_hard
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from anchor_common import resolve_image_path
from flux_attention_capture import (
    FluxAttentionCapture,
    FluxAttentionStore,
    attribute_target_token_indices,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
NUM_INFERENCE_STEPS = 25
IMG_SIZE = 1024


def compute_entropy(prob_dist: np.ndarray, eps: float = 1e-12) -> float:
    """Shannon entropy of a 2D attention probability map."""
    p = np.clip(prob_dist, eps, 1.0)
    p = p / p.sum()
    return float(-np.sum(p * np.log2(p)))


def compute_in_box_mass(heatmap: np.ndarray, box: List[float], img_w: int, img_h: int) -> float:
    """Fraction of total heatmap mass inside the bounding box [x0, y0, x1, y1]."""
    x0, y0, x1, y1 = box
    grid_h, grid_w = heatmap.shape
    bx0 = int(round(x0 * grid_w / img_w))
    by0 = int(round(y0 * grid_h / img_h))
    bx1 = max(bx0 + 1, int(round(x1 * grid_w / img_w)))
    by1 = max(by0 + 1, int(round(y1 * grid_h / img_h)))
    
    bx0, by0 = max(0, bx0), max(0, by0)
    bx1, by1 = min(grid_w, bx1), min(grid_h, by1)
    
    total_mass = float(heatmap.sum())
    if total_mass <= 0:
        return 0.0
    in_box = float(heatmap[by0:by1, bx0:bx1].sum())
    return in_box / total_mass


def compute_peak_ratio(subject_scores: Dict[str, float]) -> float:
    """Ratio of top score to second top score."""
    vals = sorted(subject_scores.values(), reverse=True)
    if len(vals) < 2 or vals[1] <= 1e-8:
        return 1.0
    return float(vals[0] / vals[1])


def aggregate_slice(store: FluxAttentionStore, indices_in_union: List[int],
                    target_resolution: Tuple[int, int] = (IMG_SIZE, IMG_SIZE),
                    layer_indices: Optional[List[int]] = None,
                    step_indices: Optional[List[int]] = None) -> np.ndarray:
    """Aggregates attention for target columns over a subset of layers and/or steps."""
    accum = None
    side = None
    count = 0

    for step_idx, layer_maps in store.step_store.items():
        if step_indices is not None and step_idx not in step_indices:
            continue
        for name, tensor in layer_maps.items():
            # name is e.g. "transformer_blocks.5.attn"
            parts = name.split(".")
            block_idx = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
            if layer_indices is not None and block_idx not in layer_indices:
                continue
            arr = tensor.numpy()[:, indices_in_union].mean(axis=-1)
            this_side = int(round(math.sqrt(arr.shape[0])))
            if accum is None:
                side = this_side
                accum = np.zeros((side, side), dtype=np.float32)
            accum += arr.reshape(side, side)
            count += 1

    if count == 0 or accum is None:
        return np.zeros(target_resolution, dtype=np.float32)
    avg = torch.from_numpy(accum / count).unsqueeze(0).unsqueeze(0)
    up = F.interpolate(avg, size=target_resolution, mode="bilinear", align_corners=False)
    return up.squeeze(0).squeeze(0).numpy()


def main() -> None:
    ap = argparse.ArgumentParser(description="Taxonomy cross-attention capture for FLUX.1-dev")
    ap.add_argument("--artifacts-dir", default="artifacts_flux_hard", help="Artifacts directory")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of images for smoke testing")
    ap.add_argument("--model-id", default="black-forest-labs/FLUX.1-dev", help="Hugging Face FLUX model ID")
    ap.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"), help="Hugging Face access token (or set HF_TOKEN env var)")
    args = ap.parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    manifest_path = artifacts_dir / "manifest.json"
    boxes_path = artifacts_dir / "boxes.json"
    out_index_path = artifacts_dir / "taxonomy_index.json"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing {manifest_path}")
    if not boxes_path.exists():
        raise FileNotFoundError(f"Missing {boxes_path}. Run recompute_boxes.py first if missing.")

    manifest = json.loads(manifest_path.read_text())
    boxes_data = json.loads(boxes_path.read_text())

    from diffusers import FluxPipeline
    print(f"Loading {args.model_id} onto {DEVICE} in {DTYPE}...")
    token = args.hf_token or os.environ.get("HF_TOKEN")
    pipe = FluxPipeline.from_pretrained(args.model_id, torch_dtype=DTYPE, token=token).to(DEVICE)
    pipe.set_progress_bar_config(disable=True)

    images_to_process = manifest["images"]
    if args.limit:
        images_to_process = images_to_process[:args.limit]
        print(f"Smoke test mode: processing first {len(images_to_process)} images only.")

    results: Dict[str, dict] = {}
    if out_index_path.exists():
        try:
            results = json.loads(out_index_path.read_text())
            print(f"Resuming: found {len(results)} images in {out_index_path}")
        except Exception:
            results = {}

    total_diffs = []
    owner_matches = []

    for idx, img in enumerate(images_to_process):
        prompt_id = img["prompt_id"]
        str_pid = str(prompt_id)
        if str_pid in results:
            continue
        if not img.get("detected"):
            continue
        boxes = boxes_data.get(str_pid)
        if not boxes:
            continue

        prompt = img["prompt"]
        seed = img.get("seed", 42)
        attributes = img.get("attributes", [])

        subject_attr_pairs = [(a["intended_subject"], a["attribute"]) for a in attributes]
        per_attr_indices, union_indices = attribute_target_token_indices(
            pipe.tokenizer_2, prompt, subject_attr_pairs)

        capture = FluxAttentionCapture()
        capture.hook_pipeline(pipe, target_token_indices=union_indices)

        def _step_cb(p, i, t, kw):
            capture.store.step()
            return kw

        gen = torch.Generator(device=DEVICE).manual_seed(seed)
        start_t = time.time()
        try:
            with torch.no_grad():
                out_img = pipe(prompt, num_inference_steps=NUM_INFERENCE_STEPS, generator=gen,
                               height=IMG_SIZE, width=IMG_SIZE, callback_on_step_end=_step_cb).images[0]
        finally:
            capture.unhook_pipeline(pipe)

        elapsed = time.time() - start_t

        # Check pixel reproduction diff against original cached image
        orig_img_path = artifacts_dir / "images" / f"p{prompt_id}.png"
        if not orig_img_path.exists():
            orig_img_path = artifacts_dir / "images" / f"{prompt_id:04d}.png"
        
        pixel_diff = 0.0
        if orig_img_path.exists():
            orig_img = Image.open(orig_img_path).convert("RGB")
            arr_orig = np.array(orig_img, dtype=np.float32) / 255.0
            arr_new = np.array(out_img, dtype=np.float32) / 255.0
            pixel_diff = float(np.mean(np.abs(arr_orig - arr_new)))
            total_diffs.append(pixel_diff)

        # Compute taxonomy metrics per attribute
        attr_taxonomy: Dict[str, dict] = {}
        all_pooled_match = True

        for attr in attributes:
            phrase = attr["attribute"]
            if phrase not in per_attr_indices:
                continue
            indices_in_union = [union_indices.index(i) for i in per_attr_indices[phrase]]

            # 1. Standard pooled attention (first 50% steps, all 19 blocks)
            pooled_map = capture.cross_attention_map(
                target_indices_position=indices_in_union,
                target_resolution=(IMG_SIZE, IMG_SIZE),
                max_steps=int(NUM_INFERENCE_STEPS * 0.5)
            )
            
            pooled_scores = {s: compute_in_box_mass(pooled_map, b, IMG_SIZE, IMG_SIZE) for s, b in boxes.items()}
            pooled_winner = max(pooled_scores, key=lambda s: pooled_scores[s]) if pooled_scores else None
            
            manifest_pred = attr.get("predicted_owner")
            matches_manifest = (pooled_winner == manifest_pred)
            owner_matches.append(matches_manifest)
            if not matches_manifest:
                all_pooled_match = False

            # Distribution metrics
            entropy = compute_entropy(pooled_map)
            peak_ratio = compute_peak_ratio(pooled_scores)
            
            # 2. Depth bands: Early (0-6), Mid (7-12), Late (13-18)
            bands = {
                "early": list(range(0, 7)),
                "mid": list(range(7, 13)),
                "late": list(range(13, 19)),
            }
            band_results = {}
            for band_name, layer_list in bands.items():
                b_map = aggregate_slice(capture.store, indices_in_union,
                                        layer_indices=layer_list,
                                        step_indices=list(range(int(NUM_INFERENCE_STEPS * 0.5))))
                b_scores = {s: compute_in_box_mass(b_map, b, IMG_SIZE, IMG_SIZE) for s, b in boxes.items()}
                band_results[band_name] = {
                    "predicted_owner": max(b_scores, key=lambda s: b_scores[s]) if b_scores else None,
                    "entropy": round(compute_entropy(b_map), 4),
                    "in_box_mass": {s: round(v, 4) for s, v in b_scores.items()}
                }

            # 3. Timestep Quartiles (all layers)
            q_len = NUM_INFERENCE_STEPS // 4
            quartiles = {
                "Q1_0_25": list(range(0, q_len)),
                "Q2_25_50": list(range(q_len, 2 * q_len)),
                "Q3_50_75": list(range(2 * q_len, 3 * q_len)),
                "Q4_75_100": list(range(3 * q_len, NUM_INFERENCE_STEPS)),
            }
            quartile_results = {}
            for q_name, q_steps in quartiles.items():
                q_map = aggregate_slice(capture.store, indices_in_union,
                                        layer_indices=list(range(19)),
                                        step_indices=q_steps)
                q_scores = {s: compute_in_box_mass(q_map, b, IMG_SIZE, IMG_SIZE) for s, b in boxes.items()}
                quartile_results[q_name] = {
                    "predicted_owner": max(q_scores, key=lambda s: q_scores[s]) if q_scores else None,
                    "entropy": round(compute_entropy(q_map), 4),
                    "in_box_mass": {s: round(v, 4) for s, v in q_scores.items()}
                }

            attr_taxonomy[phrase] = {
                "intended_subject": attr.get("intended_subject"),
                "manifest_predicted_owner": manifest_pred,
                "recomputed_pooled_owner": pooled_winner,
                "entropy": round(entropy, 4),
                "peak_to_second_peak": round(peak_ratio, 4),
                "layer_bands": band_results,
                "timestep_quartiles": quartile_results,
            }

        results[str_pid] = {
            "prompt_id": prompt_id,
            "seed": seed,
            "repro_mean_abs_pixel_diff": round(pixel_diff, 6),
            "pooled_owner_matches_manifest": all_pooled_match,
            "elapsed_sec": round(elapsed, 2),
            "attributes": attr_taxonomy,
        }

        out_index_path.write_text(json.dumps(results, indent=2))
        print(f"  [{idx+1}/{len(images_to_process)}] p{prompt_id} ({elapsed:.1f}s) | "
              f"pixel_diff={pixel_diff:.2e} | pooled_match={all_pooled_match}")

    mean_diff = np.mean(total_diffs) if total_diffs else 0.0
    match_rate = np.mean(owner_matches) if owner_matches else 1.0

    print("\n" + "=" * 60)
    print(f"TAXONOMY CAPTURE COMPLETE -> {out_index_path}")
    print(f"Total Images: {len(results)}")
    print(f"Mean Abs Pixel Diff: {mean_diff:.2e} (Check: < 1e-3)")
    print(f"Pooled Owner Match Rate: {match_rate:.1%} (Check: 100%)")
    print("=" * 60)


if __name__ == "__main__":
    main()
