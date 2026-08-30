#!/usr/bin/env python
"""
Recomputes subject bounding boxes for an already-generated anchor set by rerunning person
detection (Mask R-CNN) + subject assignment (CLIP) on the CACHED images -- no diffusion, no
GPU needed. generate_anchor_images.py / generate_anchor_images_sdxl.py compute these boxes
at generation time to pick predicted_owner, but never persist them to manifest.json, which
blocks the discriminant-validity question: is `predicted_owner` tracking box size/position
rather than attention content? This closes that gap without a Kaggle
round-trip -- Mask R-CNN + CLIP are both CPU-tractable at ~35 images total.

Same detector/threshold/assignment logic as the two generation scripts (person_boxes /
assign_subjects, kept in sync manually -- same "keep in sync" convention those two files
already use with each other). Output is a SEPARATE sidecar file, boxes.json, keyed by
prompt_id -> {subject: [x0, y0, x1, y1]} -- manifest.json stays the frozen, as-generated
record; this is a derived artifact, not a correction to it.

Run from inside ssa/anchor_set/:
    py -3 recompute_boxes.py --artifacts-dir artifacts
    py -3 recompute_boxes.py --artifacts-dir artifacts_sdxl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from config import Config

DETECTION_SCORE_THRESH = Config().generation.detection_score_thresh
COCO_PERSON = 1


# ---------------------------------------------------------------------------
# Pure caching/orchestration logic (tested).
# ---------------------------------------------------------------------------

def load_box_cache(path: Path) -> Dict[int, Dict[str, list]]:
    if not Path(path).exists():
        return {}
    raw = json.loads(Path(path).read_text())
    return {int(k): v for k, v in raw.items()}


def save_box_cache(path: Path, cache: Dict[int, Dict[str, list]]) -> None:
    Path(path).write_text(
        json.dumps({str(k): v for k, v in cache.items()}, indent=2, sort_keys=True))


def pending_box_targets(manifest: dict, cache: Dict[int, Dict[str, list]]) -> List[dict]:
    """Detected image entries whose prompt_id has no cached box entry yet."""
    return [img for img in manifest["images"]
            if img.get("detected") and img["prompt_id"] not in cache]


# ---------------------------------------------------------------------------
# Model-dependent detection (untested here -- same split as owlvit_cross_check.py's
# lazily-loaded, model-calling closure). Verbatim logic from generate_anchor_images*.py.
# ---------------------------------------------------------------------------

def load_detection_models():
    from torchvision.models.detection import (
        MaskRCNN_ResNet50_FPN_Weights, maskrcnn_resnet50_fpn,
    )
    from transformers import CLIPModel, CLIPProcessor

    detector = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT).eval()
    clip_model_id = Config().models.clip_model
    clip_model = CLIPModel.from_pretrained(clip_model_id, use_safetensors=True).eval()
    clip_proc = CLIPProcessor.from_pretrained(clip_model_id)
    return detector, clip_model, clip_proc


def _person_boxes(detector, image, max_people: int,
                   score_thresh: float = DETECTION_SCORE_THRESH) -> List[List[float]]:
    import torch
    import torchvision.transforms.functional as TF

    with torch.no_grad():
        out = detector([TF.to_tensor(image)])[0]
    keep = (out["labels"] == COCO_PERSON) & (out["scores"] >= score_thresh)
    boxes, scores = out["boxes"][keep], out["scores"][keep]
    if boxes.shape[0] == 0:
        return []
    order = scores.argsort(descending=True)[:max_people]
    return boxes[order].cpu().numpy().tolist()


def _assign_subjects(clip_model, clip_proc, image, boxes: List[List[float]],
                      subject_names: List[str]) -> Dict[str, int]:
    import torch
    from scipy.optimize import linear_sum_assignment

    crops = [image.crop(tuple(b)) for b in boxes]
    inp = clip_proc(text=[f"a photo of a {s}" for s in subject_names],
                     images=crops, return_tensors="pt", padding=True)
    with torch.no_grad():
        sim = clip_model(**inp).logits_per_image.softmax(-1).numpy()
    r, c = linear_sum_assignment(-sim)
    return {subject_names[j]: int(i) for i, j in zip(r, c)}


def detect_subject_boxes(detector, clip_model, clip_proc, image,
                          subjects: List[str]) -> Dict[str, list]:
    """Returns {subject: box}, or {} if a fresh rerun doesn't recover exactly
    len(subjects) people (a rare CPU-vs-original-run nondeterminism edge case at the
    detection-score threshold boundary) -- skip rather than silently mis-assign."""
    boxes = _person_boxes(detector, image, max_people=len(subjects))
    if len(boxes) != len(subjects):
        return {}
    assign = _assign_subjects(clip_model, clip_proc, image, boxes, subjects)
    return {s: boxes[assign[s]] for s in subjects}


def main() -> None:
    from PIL import Image

    from anchor_common import resolve_image_path

    ap = argparse.ArgumentParser(
        description="Recompute subject boxes for an anchor-set manifest")
    ap.add_argument("--artifacts-dir", default="artifacts",
                    help="directory holding manifest.json, e.g. 'artifacts_sdxl'")
    args = ap.parse_args()
    artifacts_dir = Path(args.artifacts_dir)
    manifest = json.loads((artifacts_dir / "manifest.json").read_text())
    box_path = artifacts_dir / "boxes.json"
    cache = load_box_cache(box_path)

    pending = pending_box_targets(manifest, cache)
    if not pending:
        print(f"All {len(cache)} cached prompt(s) already have boxes; nothing to do.")
        return

    print(f"Loading detection models for {len(pending)} pending prompt(s)...")
    detector, clip_model, clip_proc = load_detection_models()

    for img in pending:
        image = Image.open(resolve_image_path(artifacts_dir, img["image_path"])).convert("RGB")
        boxes = detect_subject_boxes(detector, clip_model, clip_proc, image, img["subjects"])
        if not boxes:
            print(f"  p{img['prompt_id']}: detection mismatch on rerun, skipping")
            continue
        cache[img["prompt_id"]] = boxes
        save_box_cache(box_path, cache)
        print(f"  p{img['prompt_id']}: OK ({len(boxes)} boxes)")

    print(f"Done. {len(cache)}/{len(manifest['images'])} prompt(s) have boxes -> {box_path}")


if __name__ == "__main__":
    main()
