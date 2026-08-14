#!/usr/bin/env python
"""
VQAScore baseline for FLUX anchor sets (artifacts_flux, artifacts_flux_hard).
Runs on GPU: reads images + boxes.json from the specified --artifacts-dir,
asks BLIP-VQA-base (Salesforce/blip-vqa-base) "Is the person {phrase}?" per
(subject crop, attribute), and saves vqa_scores.json directly into artifacts-dir.

Usage on GPU machine:
    python vqa_score_flux.py --artifacts-dir artifacts_flux_hard
    python vqa_score_flux.py --artifacts-dir artifacts_flux
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ATTRIBUTE_PHRASES: Dict[str, str] = {
    "red apron": "wearing a red apron",
    "blue apron": "wearing a blue apron",
    "white apron": "wearing a white apron",
    "yellow apron": "wearing a yellow apron",
    "white hat": "wearing a white hat",
    "black hat": "wearing a black hat",
    "shovel": "holding a shovel",
    "blue gloves": "wearing blue gloves",
    "red gloves": "wearing red gloves",
    "yellow gloves": "wearing yellow gloves",
    "dark sunglasses": "wearing dark sunglasses",
    "book": "holding a book",
    "yellow helmet": "wearing a yellow helmet",
    "blue helmet": "wearing a blue helmet",
    "red helmet": "wearing a red helmet",
    "pan": "holding a pan",
}


def attribute_question(attribute: str) -> str:
    if attribute in ATTRIBUTE_PHRASES:
        return f"Is the person {ATTRIBUTE_PHRASES[attribute]}?"
    holding = {"shovel", "book", "pan"}
    if attribute in holding or any(attribute.endswith(h) for h in holding):
        phrase = f"holding a {attribute}" if not attribute.startswith("a ") else f"holding {attribute}"
    else:
        phrase = f"wearing a {attribute}" if not attribute.startswith("a ") else f"wearing {attribute}"
        if "gloves" in attribute or "sunglasses" in attribute:
            phrase = f"wearing {attribute}"
    return f"Is the person {phrase}?"


def load_model():
    from transformers import BlipForQuestionAnswering, BlipProcessor

    print(f"Loading Salesforce/blip-vqa-base on {DEVICE}...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(DEVICE).eval()
    return processor, model


@torch.no_grad()
def p_yes(processor, model, image: Image.Image, question: str) -> float:
    """P(yes) = softmax(yes) / (softmax(yes) + softmax(no)) over the first generated token's logits."""
    inputs = processor(image, question, return_tensors="pt").to(DEVICE)
    out = model.generate(**inputs, max_new_tokens=1, output_scores=True, return_dict_in_generate=True)
    logits = out.scores[0][0].float()
    probs = torch.softmax(logits, dim=-1)

    yes_ids = processor.tokenizer("yes", add_special_tokens=False).input_ids
    no_ids = processor.tokenizer("no", add_special_tokens=False).input_ids
    p_y = float(probs[yes_ids].sum())
    p_n = float(probs[no_ids].sum())

    return p_y / (p_y + p_n) if (p_y + p_n) > 0 else 0.5


def main() -> None:
    ap = argparse.ArgumentParser(description="Run VQAScore on FLUX anchor sets")
    ap.add_argument("--artifacts-dir", default="artifacts_flux_hard", help="Path to artifacts directory")
    args = ap.parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    manifest_path = artifacts_dir / "manifest.json"
    boxes_path = artifacts_dir / "boxes.json"
    out_path = artifacts_dir / "vqa_scores.json"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing {manifest_path}")
    if not boxes_path.exists():
        raise FileNotFoundError(f"Missing {boxes_path}. Run recompute_boxes.py first if missing.")

    manifest = json.loads(manifest_path.read_text())
    boxes = json.loads(boxes_path.read_text())

    print(f"Artifacts: {artifacts_dir} | Total Images: {len(manifest['images'])} | Device: {DEVICE}")
    processor, model = load_model()

    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    if out_path.exists():
        try:
            results = json.loads(out_path.read_text())
            print(f"Resuming from existing {out_path} ({len(results)} images already scored)")
        except Exception:
            results = {}

    for i, img in enumerate(manifest["images"]):
        prompt_id = img["prompt_id"]
        str_pid = str(prompt_id)
        if str_pid in results:
            continue
        if not img.get("detected"):
            continue
        subject_boxes = boxes.get(str_pid)
        if not subject_boxes:
            print(f"  p{prompt_id}: no box data, skipping")
            continue

        img_path = artifacts_dir / "images" / f"p{prompt_id}.png"
        if not img_path.exists():
            img_path = artifacts_dir / "images" / f"{prompt_id:04d}.png"
        if not img_path.exists():
            print(f"  p{prompt_id}: image file not found at {img_path}, skipping")
            continue

        image = Image.open(img_path).convert("RGB")
        crops = {s: image.crop(tuple(b)) for s, b in subject_boxes.items()}

        per_attribute: Dict[str, Dict[str, float]] = {}
        for attr in img.get("attributes", []):
            question = attribute_question(attr["attribute"])
            scores: Dict[str, float] = {}
            for subject, crop in crops.items():
                scores[subject] = p_yes(processor, model, crop, question)
            per_attribute[attr["attribute"]] = scores

        results[str_pid] = per_attribute
        out_path.write_text(json.dumps(results, indent=2))
        print(f"  [{i+1}/{len(manifest['images'])}] p{prompt_id}: scored {len(per_attribute)} attributes x {len(crops)} subjects")

    print(f"\nDone! Scored {len(results)} images -> saved to {out_path}")


if __name__ == "__main__":
    main()
