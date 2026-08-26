#!/usr/bin/env python
"""
Discriminant validity vs. CLIPScore (experiment #8 in the CPGA experiment set): rules out
the objection "your attention metric is just CLIP alignment in disguise."

For each (image, attribute), crops each candidate subject's box and scores it against the
attribute's caption with CLIP -- the model, not the attention metric -- then predicts the
subject with the highest score. Compares that CLIP-only prediction against (a) the
attention metric's own `predicted_owner` (agreement rate -- the #8 deliverable) and (b)
against human labels, alongside attention's own accuracy (McNemar).

NO KAGGLE KERNEL NEEDED. CLIP (openai/clip-vit-base-patch32) already runs on CPU elsewhere
in this repo -- recompute_boxes.py's docstring calls it "CPU-tractable at ~35 images total"
for a heavier Mask R-CNN + CLIP combo than this needs. This file follows
recompute_boxes.py's own convention: pure/tested logic at the top, model-calling/untested
code at the bottom, with an incrementally-saved JSON cache (clip_scores.json, same shape as
vqa_scores.json) so a partial run resumes instead of re-embedding everything.

A CAVEAT THIS EXPERIMENT MUST CARRY, NOT PAPER OVER: `assign_subjects` (in
generate_anchor_images_flux.py / recompute_boxes.py) ALREADY uses this same CLIP checkpoint
to decide which detected box is "barista" vs. "cyclist" -- CLIP sits upstream of our own
`predicted_owner` labels. A high agreement rate here is therefore not pure evidence that
attention independently converges on CLIP's judgment; part of it could be architectural,
inherited through the shared box-assignment step. Report this alongside the agreement rate,
not as a footnote after it.

`attribute_caption` reuses anchor_common's own _HELD_NOUNS/_NO_ARTICLE_WORN_NOUNS
classification (imported, not duplicated -- this file, like vqa_agreement_check.py, can
import local modules) so "a red apron" / "a book" / "sunglasses" all get the article CLIP's
own training captions would use, rather than a second hand-maintained phrase list drifting
from attribute_question's.

Run from inside ssa/anchor_set/:
    py -3 exp10_clipscore_discriminant.py --artifacts-dir artifacts_flux_hard --annotator consensus
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from anchor_common import (
    NON_SUBJECT_LABELS, _HELD_NOUNS, _NO_ARTICLE_WORN_NOUNS, build_agreement_rows, label_key,
    load_labels,
)
from vqa_agreement_check import paired_mcnemar

CACHE_NAME = "clip_scores.json"


# ---------------------------------------------------------------------------
# Pure logic (tested). No model calls below this point until compute_clip_scores.
# ---------------------------------------------------------------------------

def attribute_caption(attribute: str) -> str:
    """A CLIP-style caption for one attribute, e.g. "a photo of a person wearing a red
    apron" or "a photo of a person holding a book" -- a declarative caption, not a
    question, since CLIP is trained on captions (unlike attribute_question's VQA phrasing).
    Reuses anchor_common's own noun classification so this never drifts from
    attribute_question's held/worn logic."""
    last_word = attribute.rsplit(" ", 1)[-1].lower()
    if last_word in _HELD_NOUNS:
        return f"a photo of a person holding a {attribute}"
    if last_word in _NO_ARTICLE_WORN_NOUNS:
        return f"a photo of a person wearing {attribute}"
    return f"a photo of a person wearing a {attribute}"


def clip_predicted_owner(scores_by_subject: Dict[str, float]) -> str:
    if not scores_by_subject:
        raise ValueError("scores_by_subject is empty; nothing to attribute to")
    return max(scores_by_subject, key=lambda s: scores_by_subject[s])


def load_score_cache(path: Path) -> Dict[str, dict]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def save_score_cache(path: Path, cache: Dict[str, dict]) -> None:
    path.write_text(json.dumps(cache, indent=2, sort_keys=True))


def pending_clip_targets(manifest: dict, boxes: Dict[str, dict], cache: Dict[str, dict]
                         ) -> List[dict]:
    """Detected images with cached boxes that have no cache entry yet -- mirrors
    recompute_boxes.py's pending_box_targets so a partial run resumes cheaply."""
    return [img for img in manifest["images"]
            if img.get("detected") and str(img["prompt_id"]) in boxes
            and str(img["prompt_id"]) not in cache]


def clip_agreement_rows(manifest: dict, labels: Dict[str, str],
                        clip_scores: Dict[str, dict]) -> List[dict]:
    """Same row shape as anchor_common.build_agreement_rows / vqa_agreement_check's
    vqa_agreement_rows, but `predicted_owner` is CLIP's argmax. Skips images with no cache
    entry rather than crashing, so a partial CLIP run can still be analyzed."""
    rows: List[dict] = []
    for img in manifest["images"]:
        if not img.get("detected"):
            continue
        per_image = clip_scores.get(str(img["prompt_id"]))
        if not per_image:
            continue
        for attr in img["attributes"]:
            key = label_key(img["prompt_id"], attr["attribute"])
            if key not in labels:
                continue
            per_subject = per_image.get(attr["attribute"])
            if not per_subject:
                continue
            human = labels[key]
            scored = human not in NON_SUBJECT_LABELS
            owner = clip_predicted_owner(per_subject)
            rows.append(dict(
                prompt_id=img["prompt_id"], n=img["n"], attribute=attr["attribute"],
                intended_subject=attr["intended_subject"], predicted_owner=owner,
                human_label=human, scored=scored, correct=(scored and owner == human),
            ))
    return rows


def prediction_agreement_rate(attn_rows: List[dict], clip_rows: List[dict]) -> dict:
    """THE #8 deliverable: on identical (prompt_id, attribute) rows, how often does
    attention's predicted_owner equal CLIP's? Joined on the row identity, not on
    correctness -- this compares the two METHODS to each other, independent of whether
    either is right."""
    if not attn_rows or not clip_rows:
        return dict(n=0, agreement_rate=None)
    attn_df = pd.DataFrame(attn_rows)[["prompt_id", "attribute", "predicted_owner"]].rename(
        columns={"predicted_owner": "attn_owner"})
    clip_df = pd.DataFrame(clip_rows)[["prompt_id", "attribute", "predicted_owner"]].rename(
        columns={"predicted_owner": "clip_owner"})
    joined = attn_df.merge(clip_df, on=["prompt_id", "attribute"], how="inner")
    if joined.empty:
        return dict(n=0, agreement_rate=None)
    agree = int((joined["attn_owner"] == joined["clip_owner"]).sum())
    return dict(n=len(joined), agreement_rate=agree / len(joined), n_agree=agree)


# ---------------------------------------------------------------------------
# Model-dependent (untested here -- same split as recompute_boxes.py's detection functions).
# ---------------------------------------------------------------------------

def load_clip_model():
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch32", use_safetensors=True).eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, proc


def compute_clip_scores_for_image(clip_model, clip_proc, image, subject_boxes: Dict[str, list],
                                  attributes: List[str]) -> Dict[str, Dict[str, float]]:
    """{attribute: {subject: score}}. Uses CLIPModel's own logits_per_image (crops as
    "images", the caption as the single "text") -- the same call shape
    generate_anchor_images_flux.py's assign_subjects already uses, just with subjects and
    text swapped: here N crops are scored against ONE caption instead of one image against
    N subject-name texts. logits_per_image is logit_scale * cosine similarity; the positive
    scale factor doesn't change which subject has the highest score, so this reproduces the
    argmax a normalized CLIPScore would give without re-deriving embeddings by hand."""
    import torch

    subjects = sorted(subject_boxes)
    crops = [image.crop(tuple(subject_boxes[s])) for s in subjects]
    out: Dict[str, Dict[str, float]] = {}
    for attribute in attributes:
        caption = attribute_caption(attribute)
        inp = clip_proc(text=[caption], images=crops, return_tensors="pt", padding=True)
        with torch.no_grad():
            sims = clip_model(**inp).logits_per_image.squeeze(-1)
        out[attribute] = {s: float(sims[i]) for i, s in enumerate(subjects)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="CLIPScore discriminant-validity check against the attention metric")
    ap.add_argument("--artifacts-dir", default="artifacts_flux_hard")
    ap.add_argument("--annotator", default="consensus")
    args = ap.parse_args()
    artifacts_dir = Path(args.artifacts_dir)

    from PIL import Image

    manifest = json.loads((artifacts_dir / "manifest.json").read_text())
    boxes = json.loads((artifacts_dir / "boxes.json").read_text())
    labels = load_labels(artifacts_dir / f"labels_{args.annotator}.json")
    cache_path = artifacts_dir / CACHE_NAME
    cache = load_score_cache(cache_path)

    todo = pending_clip_targets(manifest, boxes, cache)
    print(f"{len(todo)} image(s) pending CLIP scoring ({len(cache)} already cached)")
    if todo:
        clip_model, clip_proc = load_clip_model()
        for i, img in enumerate(todo):
            prompt_id = str(img["prompt_id"])
            image_path = artifacts_dir / Path(img["image_path"]).name
            if not image_path.exists():
                image_path = Path(img["image_path"])
            image = Image.open(image_path).convert("RGB")
            attributes = [a["attribute"] for a in img["attributes"]]
            cache[prompt_id] = compute_clip_scores_for_image(
                clip_model, clip_proc, image, boxes[prompt_id], attributes)
            save_score_cache(cache_path, cache)
            if (i + 1) % 10 == 0 or i + 1 == len(todo):
                print(f"  scored {i + 1}/{len(todo)}")

    attn_rows = build_agreement_rows(manifest, labels)
    clip_rows = clip_agreement_rows(manifest, labels, cache)

    print(f"\nattention rows: {len(attn_rows)}  |  CLIPScore rows: {len(clip_rows)}")

    agreement = prediction_agreement_rate(attn_rows, clip_rows)
    print(f"\n=== #8 deliverable: attention vs. CLIPScore prediction agreement ===")
    print(f"  n={agreement['n']}  agreement_rate={agreement['agreement_rate']}")
    print("  CAVEAT: assign_subjects already uses this CLIP checkpoint for box assignment,")
    print("  so agreement here is not pure evidence of independent convergence -- see")
    print("  module docstring before reporting this number on its own.")

    attn_df = pd.DataFrame(attn_rows).rename(
        columns={"predicted_owner": "attn_predicted_owner", "correct": "attn_correct"})
    clip_df = pd.DataFrame(clip_rows)[["prompt_id", "attribute", "predicted_owner", "correct"]].rename(
        columns={"predicted_owner": "clip_predicted_owner", "correct": "clip_correct"})
    joined = attn_df.merge(clip_df, on=["prompt_id", "attribute"], how="inner")

    print(f"\n=== Accuracy vs. human labels, identical {len(joined)} rows ===")
    scored = joined[joined.scored]
    print(f"  attention accuracy: {scored['attn_correct'].mean():.1%}")
    print(f"  CLIPScore accuracy: {scored['clip_correct'].mean():.1%}")
    mcnemar = paired_mcnemar(joined, "attn_correct", "clip_correct")
    print(f"  McNemar: attention-only correct={mcnemar['a_only_correct']}, "
          f"CLIPScore-only correct={mcnemar['b_only_correct']}, p={mcnemar['p_value']}")


if __name__ == "__main__":
    main()
