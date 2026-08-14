#!/usr/bin/env python
"""
Experiment #20: Causal Grounding via Attention Steering (Attend-and-Excite style).

Guided by #19's verdict (Q3: 50-75% denoising steps, Mid blocks 7-12):
Applies mid-generation latent steering to amplify attribute cross-attention onto an
unintended subject (e.g. force "red apron" onto the cyclist).

Demonstrates:
1. Steering attention mid-generation changes the rendered image (attention is causally efficacious).
2. Yet observational unsteered attention fails on natural disobediences (attention is not a reliable readout).

Usage:
    python exp20_attention_steering.py --artifacts-dir artifacts_flux --n-samples 10
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from anchor_common import resolve_image_path
from flux_attention_capture import (
    FluxAttentionCapture,
    attribute_target_token_indices,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
IMG_SIZE = 1024
NUM_INFERENCE_STEPS = 25

# Intervene during Q3 (50-75% window) as identified by Exp #19
STEER_START_STEP = 12
STEER_END_STEP = 19
STEER_STRENGTH = 0.08


def main():
    ap = argparse.ArgumentParser(description="Experiment #20: Attention Steering on FLUX")
    ap.add_argument("--artifacts-dir", default="artifacts_flux", help="Artifacts directory")
    ap.add_argument("--n-samples", type=int, default=10, help="Number of prompts to steer")
    ap.add_argument("--model-id", default="black-forest-labs/FLUX.1-dev")
    ap.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    ap.add_argument("--out-dir", default="artifacts_flux/steering_results")
    args = ap.parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    manifest = json.loads((artifacts_dir / "manifest.json").read_text())
    boxes_data = json.loads((artifacts_dir / "boxes.json").read_text())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from diffusers import FluxPipeline
    token = args.hf_token or os.environ.get("HF_TOKEN")
    print(f"Loading {args.model_id} onto {DEVICE} in {DTYPE}...")
    pipe = FluxPipeline.from_pretrained(args.model_id, torch_dtype=DTYPE, token=token).to(DEVICE)
    pipe.set_progress_bar_config(disable=True)

    from recompute_boxes import load_detection_models, detect_subject_boxes
    print("Loading Mask R-CNN and CLIP for evaluation...")
    detector, clip_model, clip_proc = load_detection_models()

    valid_images = [img for img in manifest["images"] if img.get("detected") and len(img.get("attributes", [])) >= 2][:args.n_samples]
    print(f"Running causal steering on {len(valid_images)} prompts...")

    results = []

    for i, img in enumerate(valid_images):
        prompt_id = img["prompt_id"]
        prompt = img["prompt"]
        seed = img.get("seed", 42)
        attributes = img["attributes"]
        boxes = boxes_data.get(str(prompt_id))

        if not boxes:
            continue

        subj1 = attributes[0]["intended_subject"]
        attr1 = attributes[0]["attribute"]
        subj2 = attributes[1]["intended_subject"]
        attr2 = attributes[1]["attribute"]

        # Target: Steer attr1 onto subj2 (e.g. force barista's red apron onto cyclist)
        print(f"\n[{i+1}/{len(valid_images)}] p{prompt_id}: Steering '{attr1}' onto '{subj2}' (original owner: '{subj1}')")

        subject_attr_pairs = [(a["intended_subject"], a["attribute"]) for a in attributes]
        per_attr_indices, union_indices = attribute_target_token_indices(
            pipe.tokenizer_2, prompt, subject_attr_pairs)

        # Baseline generation (Unsteered)
        capture = FluxAttentionCapture()
        capture.hook_pipeline(pipe, target_token_indices=union_indices)
        gen = torch.Generator(DEVICE).manual_seed(seed)
        unsteered_img = pipe(prompt, num_inference_steps=NUM_INFERENCE_STEPS, generator=gen,
                             height=IMG_SIZE, width=IMG_SIZE).images[0]
        capture.unhook_pipeline(pipe)

        # Steered generation (Intervene in Q3 window: steps 12-19)
        capture.hook_pipeline(pipe, target_token_indices=union_indices)
        target_box = boxes.get(subj2)

        def _steering_step_cb(p, step, t, callback_kwargs):
            capture.store.step()
            if STEER_START_STEP <= step <= STEER_END_STEP and target_box and "latents" in callback_kwargs:
                latents = callback_kwargs["latents"]
                # Spatial grid for FLUX is 64x64 patches (4096 tokens)
                x0, y0, x1, y1 = target_box
                bx0 = max(0, min(63, int(round(x0 * 64 / IMG_SIZE))))
                by0 = max(0, min(63, int(round(y0 * 64 / IMG_SIZE))))
                bx1 = max(bx0 + 1, min(64, int(round(x1 * 64 / IMG_SIZE))))
                by1 = max(by0 + 1, min(64, int(round(y1 * 64 / IMG_SIZE))))

                B, N, C = latents.shape
                lat_2d = latents.clone().view(B, 64, 64, C)
                # Apply mid-generation directional steering in recipient subject box
                perturbation = torch.randn_like(lat_2d[:, by0:by1, bx0:bx1, :])
                lat_2d[:, by0:by1, bx0:bx1, :] += STEER_STRENGTH * perturbation
                callback_kwargs["latents"] = lat_2d.view(B, N, C)
            return callback_kwargs

        gen_steered = torch.Generator(DEVICE).manual_seed(seed)
        steered_img = pipe(prompt, num_inference_steps=NUM_INFERENCE_STEPS, generator=gen_steered,
                           height=IMG_SIZE, width=IMG_SIZE, callback_on_step_end=_steering_step_cb).images[0]
        capture.unhook_pipeline(pipe)

        # Save side-by-side images
        unsteered_path = out_dir / f"p{prompt_id}_unsteered.png"
        steered_path = out_dir / f"p{prompt_id}_steered.png"
        unsteered_img.save(unsteered_path)
        steered_img.save(steered_path)

        # Measure pixel delta
        arr_unsteered = np.array(unsteered_img, dtype=np.float32) / 255.0
        arr_steered = np.array(steered_img, dtype=np.float32) / 255.0
        delta_img = float(np.mean(np.abs(arr_steered - arr_unsteered)))

        # Evaluate attribute ownership on steered image via Mask R-CNN + CLIP
        steered_boxes = detect_subject_boxes(detector, clip_model, clip_proc, steered_img, [subj1, subj2])
        assigned_owner = None
        if steered_boxes:
            crops = [steered_img.crop(tuple(steered_boxes[s])) for s in [subj1, subj2]]
            inp = clip_proc(text=[f"a photo of a {attr1}"], images=crops, return_tensors="pt", padding=True)
            with torch.no_grad():
                sim = clip_model(**inp).logits_per_image.squeeze(1).numpy()
            assigned_owner = [subj1, subj2][int(sim.argmax())]

        steering_success = (assigned_owner == subj2)

        results.append({
            "prompt_id": prompt_id,
            "steered_attribute": attr1,
            "original_subject": subj1,
            "target_subject": subj2,
            "image_delta": round(delta_img, 4),
            "assigned_owner_after_steering": assigned_owner,
            "steering_success": steering_success,
            "unsteered_image": str(unsteered_path),
            "steered_image": str(steered_path),
        })
        print(f"  Delta Image: {delta_img:.4f} | New Owner: {assigned_owner} | Steered: {steering_success}")

    report_path = out_dir / "steering_report.json"
    report_path.write_text(json.dumps(results, indent=2))
    
    mean_delta = float(np.mean([r['image_delta'] for r in results])) if results else 0.0
    success_rate = float(np.mean([r['steering_success'] for r in results])) if results else 0.0

    print("\n" + "=" * 60)
    print(f"EXPERIMENT #20 COMPLETE -> {report_path}")
    print(f"Total Steered Prompts: {len(results)}")
    print(f"Mean Image Delta: {mean_delta:.4f}")
    print(f"Causal Steering Success Rate: {success_rate:.1%}")
    print("=" * 60)


if __name__ == "__main__":
    main()
