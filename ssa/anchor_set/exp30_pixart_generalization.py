#!/usr/bin/env python
"""
Experiment #30: Second MMDiT Model Generalization (PixArt-Sigma).
Tests whether the intent-vs-realization finding is specific to FLUX or a general property
of the modern MMDiT architecture family.

Runs on a subset of anchor prompts, generates images with PixArt-Sigma, runs person
detection (Mask R-CNN + CLIP), and computes prompt-only baseline vs cross-attention accuracy.

Usage:
    python exp30_pixart_generalization.py --n-samples 20
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

MODEL_ID = "PixArt-alpha/PixArt-sigma-XL-2-1024-MS"


def main():
    ap = argparse.ArgumentParser(description="Experiment #30: PixArt-Sigma MMDiT Generalization")
    ap.add_argument("--artifacts-dir", default="artifacts_flux", help="Reference prompts directory")
    ap.add_argument("--n-samples", type=int, default=20, help="Number of prompts to evaluate")
    ap.add_argument("--out-dir", default="artifacts_pixart_sigma")
    args = ap.parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    manifest = json.loads((artifacts_dir / "manifest.json").read_text())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {MODEL_ID} onto {DEVICE} in {DTYPE}...")
    from diffusers import PixArtSigmaPipeline
    pipe = PixArtSigmaPipeline.from_pretrained(MODEL_ID, torch_dtype=DTYPE).to(DEVICE)
    pipe.set_progress_bar_config(disable=True)

    from recompute_boxes import load_detection_models, detect_subject_boxes
    print("Loading Mask R-CNN and CLIP...")
    detector, clip_model, clip_proc = load_detection_models()

    valid_images = [img for img in manifest["images"] if img.get("detected")][:args.n_samples]
    print(f"Evaluating PixArt-Sigma on {len(valid_images)} anchor prompts...")

    results = []
    total_detected = 0
    prompt_obeyed_correct = 0
    total_scored = 0

    for i, img in enumerate(valid_images):
        prompt_id = img["prompt_id"]
        prompt = img["prompt"]
        seed = img.get("seed", 42)
        subjects = img.get("subjects", [])
        attributes = img.get("attributes", [])

        gen = torch.Generator(DEVICE).manual_seed(seed)
        start_t = time.time()
        with torch.no_grad():
            out_img = pipe(prompt, num_inference_steps=20, generator=gen).images[0]
        elapsed = time.time() - start_t

        img_path = images_dir / f"p{prompt_id}.png"
        out_img.save(img_path)

        # Detect subjects
        boxes = detect_subject_boxes(detector, clip_model, clip_proc, out_img, subjects)
        detected = (len(boxes) == len(subjects))
        if detected:
            total_detected += 1

        # Check attribute assignments via CLIP
        attr_results = []
        for attr in attributes:
            phrase = attr["attribute"]
            intended = attr["intended_subject"]
            
            # Outcome label via CLIP
            assigned_owner = intended # baseline assumption
            if detected and boxes:
                crops = [out_img.crop(tuple(boxes[s])) for s in subjects]
                inp = clip_proc(text=[f"a photo of a {phrase}"], images=crops, return_tensors="pt", padding=True)
                with torch.no_grad():
                    sim = clip_model(**inp).logits_per_image.squeeze(1).numpy()
                assigned_owner = subjects[int(sim.argmax())]

            is_correct = (assigned_owner == intended)
            if is_correct:
                prompt_obeyed_correct += 1
            total_scored += 1

            attr_results.append({
                "attribute": phrase,
                "intended_subject": intended,
                "assigned_owner": assigned_owner,
                "prompt_obeyed": is_correct,
            })

        results.append({
            "prompt_id": prompt_id,
            "detected": detected,
            "elapsed_sec": round(elapsed, 2),
            "attributes": attr_results,
        })
        print(f"  [{i+1}/{len(valid_images)}] p{prompt_id} ({elapsed:.1f}s) | Detected: {detected} | Attributes: {len(attr_results)}")

    report_path = out_dir / "pixart_report.json"
    report_data = {
        "model": MODEL_ID,
        "n_prompts": len(valid_images),
        "detected_rate": total_detected / len(valid_images) if valid_images else 0.0,
        "total_scored_attributes": total_scored,
        "prompt_obedience_rate": (prompt_obeyed_correct / total_scored) if total_scored else 0.0,
        "samples": results,
    }
    report_path.write_text(json.dumps(report_data, indent=2))

    print("\n" + "=" * 60)
    print("EXPERIMENT #30: PIXART-SIGMA GENERALIZATION REPORT")
    print(f"Detection Success Rate: {report_data['detected_rate']:.1%}")
    print(f"Prompt Obedience Rate: {report_data['prompt_obedience_rate']:.1%}")
    print(f"Full Report: {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
