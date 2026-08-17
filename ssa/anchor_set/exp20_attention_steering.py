#!/usr/bin/env python
"""
Experiment #20: Causal Grounding via Attention Steering (Attend-and-Excite style).

Guided by #19's MEASURED output (taxonomy capture executed 2026-08-17 on an A100; report at
artifacts_flux_hard/taxonomy_report.json): scales cross-ATTENTION (not latents) toward an
attribute's tokens on an unintended subject's image rows, mid-generation, via
flux_attention_capture.FluxSteeringAttnProcessor.

WHERE THE STEERING WINDOW COMES FROM (replacing the earlier mid-blocks-7-12 placeholders):
  - Layers: #14 found accuracy rises monotonically with depth (easy set 67.3% early_0_6 ->
    83.8% mid_7_12 -> 85.4% late_13_18; hard set 34.7% -> 42.4% -> 44.4%), and 7 of the 10
    sharpest cells #19 selected sit in block 18 alone (the others in 14, 16, 17). The late
    band 13-18 is where attribute-specific attention actually concentrates.
  - Steps: #17 found the four timestep windows statistically indistinguishable (easy
    84.6/84.6/85.0/85.0, hard 44.1/44.1/44.8/45.1). There is NO measured basis for narrowing
    to a sub-window, so the default steers the full trajectory rather than inventing a window
    the data does not support.

What #19's verdict does NOT license: it came back NEGATIVE (0/10 cells beat the prompt-only
baseline; every cell lost by 35-39 points at Holm p ~ 1e-20 to 1e-23). These layers are where
the signal is densest, NOT where a cell was found that reads the rendered image. #20 stays a
causal-efficacy probe; it is not a rescue of the observational claim.

Demonstrates:
1. Steering attention mid-generation changes the rendered image (attention is causally
   efficacious).
2. Yet observational unsteered attention fails on natural disobediences (attention is not a
   reliable readout).

CORRECTNESS NOTE (why this file was rewritten): a first draft intervened on the LATENTS
inside a spatial box during the same step window (regional Gaussian noise injection). That
never touches attention at all -- it demonstrates only "perturbing latents in a masked
region changes that region," a much weaker and different claim. This version scales
attn_probs itself, inside the same manual-recompute hook flux_attention_capture.py already
uses for capture (FLUX's stock processor never materializes attn_probs to intervene on --
see that module's docstring), which is what the paper's causal-efficacy claim requires.

GROUND-TRUTH CAVEAT (report this alongside every number, not after it): "steering success"
below is judged by CLIP crop-similarity (Mask R-CNN + CLIP, the same box-assignment
checkpoint used elsewhere in this repo), not a human check -- acceptable for a fast pilot,
but not a substitute for human annotation if this number is going in the paper.

Usage:
    python exp20_attention_steering.py --artifacts-dir artifacts_flux --n-samples 10
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image

from flux_attention_capture import (
    FluxAttentionCapture,
    FluxSteeringAttnProcessor,
    SteeringConfig,
    SteeringState,
    attribute_target_token_indices,
    flux_token_indices,
)
from taxonomy_capture_flux import box_to_grid_mask

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
IMG_SIZE = 1024
NUM_INFERENCE_STEPS = 25
GRID_SIDE = 64   # FLUX's native attention grid at 1024x1024 (see taxonomy_capture_flux.py)

# Measured from #19's taxonomy report (2026-08-17) -- see module docstring for the numbers.
# Layers: the late band, where #14's accuracy peaks and 7/10 of #19's sharpest cells live.
# Steps: full inclusive trajectory (steering_active tests step_start <= step <= step_end, and
# NUM_INFERENCE_STEPS=25 means steps 0..24), because #17 found no window distinguishable from
# any other -- narrowing further would not be data-driven.
DEFAULT_STEER_LAYERS = frozenset(range(13, 19))   # blocks 13-18 (late band)
DEFAULT_STEER_START = 0
DEFAULT_STEER_END = 24
DEFAULT_STRENGTH = 4.0   # multiplicative boost on target columns before renormalizing


def build_steering_config(prompt: str, tokenizer_2, target_attribute: str,
                          recipient_box: List[float], layers=DEFAULT_STEER_LAYERS,
                          step_start: int = DEFAULT_STEER_START, step_end: int = DEFAULT_STEER_END,
                          strength: float = DEFAULT_STRENGTH) -> SteeringConfig:
    """Builds the SteeringConfig for forcing `target_attribute`'s tokens onto whichever
    subject `recipient_box` belongs to. `recipient_box` is a [x0,y0,x1,y1] PIXEL box from
    boxes.json (the recipient's own detected box -- NOT the original owner's)."""
    col_indices = tuple(flux_token_indices(tokenizer_2, prompt, target_attribute))
    row_mask_np = box_to_grid_mask(recipient_box, side=GRID_SIDE, img_size=IMG_SIZE)
    row_mask = torch.from_numpy(row_mask_np)
    return SteeringConfig(target_layer_indices=frozenset(layers), step_start=step_start,
                          step_end=step_end, target_col_indices=col_indices,
                          target_row_mask=row_mask, strength=strength)


def main() -> None:
    ap = argparse.ArgumentParser(description="Experiment #20: Attention Steering on FLUX")
    ap.add_argument("--artifacts-dir", default="artifacts_flux", help="Artifacts directory")
    ap.add_argument("--n-samples", type=int, default=10, help="Number of prompts to steer")
    ap.add_argument("--model-id", default="black-forest-labs/FLUX.1-dev")
    ap.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    ap.add_argument("--out-dir", default=None, help="defaults to <artifacts-dir>/steering_results")
    ap.add_argument("--strength", type=float, default=DEFAULT_STRENGTH)
    ap.add_argument("--step-start", type=int, default=DEFAULT_STEER_START)
    ap.add_argument("--step-end", type=int, default=DEFAULT_STEER_END)
    args = ap.parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    manifest = json.loads((artifacts_dir / "manifest.json").read_text())
    boxes_data = json.loads((artifacts_dir / "boxes.json").read_text())
    out_dir = Path(args.out_dir) if args.out_dir else artifacts_dir / "steering_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    from diffusers import FluxPipeline
    token = args.hf_token or os.environ.get("HF_TOKEN")
    print(f"Loading {args.model_id} onto {DEVICE} in {DTYPE}...")
    pipe = FluxPipeline.from_pretrained(args.model_id, torch_dtype=DTYPE, token=token).to(DEVICE)
    pipe.set_progress_bar_config(disable=True)

    from recompute_boxes import load_detection_models, detect_subject_boxes
    print("Loading Mask R-CNN and CLIP for evaluation...")
    detector, clip_model, clip_proc = load_detection_models()

    valid_images = [img for img in manifest["images"]
                    if img.get("detected") and len(img.get("attributes", [])) >= 2][:args.n_samples]
    print(f"Running causal steering on {len(valid_images)} prompts "
         f"(layers={sorted(DEFAULT_STEER_LAYERS)}, steps={args.step_start}-{args.step_end}, "
         f"strength={args.strength})...")

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
        recipient_box = boxes.get(subj2)
        if not recipient_box:
            continue

        print(f"\n[{i + 1}/{len(valid_images)}] p{prompt_id}: steering '{attr1}' onto "
             f"'{subj2}' (original owner: '{subj1}')")

        subject_attr_pairs = [(a["intended_subject"], a["attribute"]) for a in attributes]
        _, union_indices = attribute_target_token_indices(pipe.tokenizer_2, prompt, subject_attr_pairs)

        # Baseline (unsteered) generation, attention captured for reference.
        capture = FluxAttentionCapture()
        capture.hook_pipeline(pipe, target_token_indices=union_indices)
        gen = torch.Generator(DEVICE).manual_seed(seed)
        unsteered_img = pipe(prompt, num_inference_steps=NUM_INFERENCE_STEPS, generator=gen,
                             height=IMG_SIZE, width=IMG_SIZE).images[0]
        capture.unhook_pipeline(pipe)

        # Steered generation: scale attn_probs toward attr1's tokens on subj2's image rows,
        # inside the (layers, steps) window -- the actual causal intervention.
        config = build_steering_config(prompt, pipe.tokenizer_2, attr1, recipient_box,
                                       step_start=args.step_start, step_end=args.step_end,
                                       strength=args.strength)
        state = SteeringState()
        processors = dict(pipe.transformer.attn_processors)
        layer_index = 0
        for name in list(processors):
            if name.startswith("transformer_blocks."):
                processors[name] = FluxSteeringAttnProcessor(layer_index, state, config)
                layer_index += 1
        pipe.transformer.set_attn_processor(processors)

        def _step_cb(p, step, t, callback_kwargs):
            state.step()
            return callback_kwargs

        gen_steered = torch.Generator(DEVICE).manual_seed(seed)
        steered_img = pipe(prompt, num_inference_steps=NUM_INFERENCE_STEPS, generator=gen_steered,
                           height=IMG_SIZE, width=IMG_SIZE, callback_on_step_end=_step_cb).images[0]
        capture.unhook_pipeline(pipe)   # resets to stock processor regardless of which hook was live

        unsteered_path = out_dir / f"p{prompt_id}_unsteered.png"
        steered_path = out_dir / f"p{prompt_id}_steered.png"
        unsteered_img.save(unsteered_path)
        steered_img.save(steered_path)

        arr_unsteered = np.array(unsteered_img, dtype=np.float32) / 255.0
        arr_steered = np.array(steered_img, dtype=np.float32) / 255.0
        delta_img = float(np.mean(np.abs(arr_steered - arr_unsteered)))

        # CLIP-judged ownership after steering -- see module docstring's ground-truth caveat.
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
            "target_layer_indices": sorted(config.target_layer_indices),
            "step_start": args.step_start, "step_end": args.step_end, "strength": args.strength,
            "image_delta": round(delta_img, 4),
            "assigned_owner_after_steering": assigned_owner,
            "steering_success": steering_success,
            "ground_truth": "CLIP crop-similarity, not human-verified -- see module docstring",
            "unsteered_image": str(unsteered_path),
            "steered_image": str(steered_path),
        })
        print(f"  image_delta={delta_img:.4f}  new_owner={assigned_owner}  "
             f"success={steering_success}")

    report_path = out_dir / "steering_report.json"
    report_path.write_text(json.dumps(results, indent=2))

    mean_delta = float(np.mean([r["image_delta"] for r in results])) if results else 0.0
    success_rate = float(np.mean([r["steering_success"] for r in results])) if results else 0.0

    print("\n" + "=" * 60)
    print(f"EXPERIMENT #20 COMPLETE -> {report_path}")
    print(f"Total steered prompts: {len(results)}")
    print(f"Mean image delta: {mean_delta:.4f}")
    print(f"Causal steering success rate: {success_rate:.1%}")
    print("CAVEAT: success is CLIP-judged, not human-verified. See module docstring.")
    print("=" * 60)


if __name__ == "__main__":
    main()
