#!/usr/bin/env python
"""
VQAScore-style baseline for metric A's anchor set. Self-contained Kaggle GPU kernel:
reads the already-generated SDXL images + boxes from the attached dataset
(anonymous/coig-metric-a-anchor-set-sdxl-images), NOT regenerating them (diffusion
output is not guaranteed bit-reproducible across Kaggle sessions, unlike Mask R-CNN
detection on an already-fixed image -- see recompute_boxes.py's docstring for that
distinction).

For every detected image, crops to each subject's box (recompute_boxes.py's boxes.json)
and asks BLIP-VQA-base (Salesforce/blip-vqa-base) "Is the person {phrase}?" per
(subject crop, attribute) -- one question per candidate subject, so the argmax across
crops becomes VQAScore's own binding prediction, directly comparable to the attention
metric's own predicted_owner (same box crops, different signal).

P(yes) follows VQAScore's own normalization (Lin et al. 2024): softmax(yes) /
(softmax(yes) + softmax(no)) over the FIRST generated token's logits, not the raw
softmax(yes) alone -- so a model that's simply uncertain about everything doesn't read as
uniformly low-confidence "no" evidence.

`attribute_question` (and the two frozensets it reads) is a verbatim duplicate of
anchor_common.py's canonical copy (this is a single-file Kaggle kernel, same "keep in
sync" convention ANCHOR_PROMPTS already uses across the two generation scripts;
tests/test_vqa_agreement_check.py's drift guard checks the two stay byte-identical).

Output: vqa_scores.json, {prompt_id (str): {attribute: {subject: p_yes}}}, saved
incrementally after each image so a Kaggle timeout doesn't lose completed work.
"""
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

if "KAGGLE_KERNEL_RUN_TYPE" in __import__("os").environ:
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--progress-bar", "off",
        "torch==2.5.1", "--index-url", "https://download.pytorch.org/whl/cu121",
    ])
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--progress-bar", "off",
        "transformers>=4.44",
    ])

import torch
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUT_PATH = Path("vqa_scores.json")


def find_input_dir() -> Path:
    """Kaggle mounts the attached dataset somewhere under /kaggle/input/ -- locate it by
    content (first dir containing manifest.json), searched RECURSIVELY, rather than
    hardcoding a mount depth: a first attempt at /kaggle/input/<slug>/manifest.json found
    the dataset actually nested one level deeper, under /kaggle/input/datasets/..."""
    for candidate in sorted(Path("/kaggle/input").glob("**/manifest.json")):
        return candidate.parent
    available = list(Path("/kaggle/input").rglob("*")) if Path("/kaggle/input").exists() else []
    raise FileNotFoundError(
        f"no manifest.json found anywhere under /kaggle/input; contents: {available}")

_HELD_NOUNS = frozenset({"book", "shovel", "pan"})

_NO_ARTICLE_WORN_NOUNS = frozenset({"gloves", "sunglasses", "glasses"})


def attribute_question(attribute: str) -> str:
    """A yes/no VQA question for one attribute string, e.g. "Is the person wearing a red
    apron?" or "Is the person holding a book?" -- the question `vqa_score_sdxl.py` /
    `vqa_score_flux.py` ask per (subject crop, attribute), whose P(yes) is VQAScore's
    binding signal (Lin et al. 2024). Classifies by the attribute's LAST word only, so any
    new color/descriptor combination (e.g. a hard-prompt-set "blue helmet") is covered
    automatically without a per-attribute lookup table -- verified to reproduce every
    entry of the anchor sets' original hand-written ATTRIBUTE_PHRASES dict exactly."""
    last_word = attribute.rsplit(" ", 1)[-1].lower()
    if last_word in _HELD_NOUNS:
        return f"Is the person holding a {attribute}?"
    if last_word in _NO_ARTICLE_WORN_NOUNS:
        return f"Is the person wearing {attribute}?"
    return f"Is the person wearing a {attribute}?"


def load_model():
    from transformers import BlipForQuestionAnswering, BlipProcessor

    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(DEVICE).eval()
    return processor, model


@torch.no_grad()
def p_yes(processor, model, image: Image.Image, question: str,
          debug_print: bool = False) -> float:
    """P(yes) = softmax(yes) / (softmax(yes) + softmax(no)) over the first generated
    token's logits (VQAScore's own normalization)."""
    inputs = processor(image, question, return_tensors="pt").to(DEVICE)
    out = model.generate(**inputs, max_new_tokens=1, output_scores=True,
                          return_dict_in_generate=True)
    logits = out.scores[0][0].float()
    probs = torch.softmax(logits, dim=-1)

    yes_ids = processor.tokenizer("yes", add_special_tokens=False).input_ids
    no_ids = processor.tokenizer("no", add_special_tokens=False).input_ids
    p_y = float(probs[yes_ids].sum())
    p_n = float(probs[no_ids].sum())

    if debug_print:
        top5 = torch.topk(probs, 5)
        decoded = [processor.tokenizer.decode([i]) for i in top5.indices.tolist()]
        print(f"    [debug] q={question!r} top5={list(zip(decoded, top5.values.tolist()))} "
              f"p_yes_raw={p_y:.4f} p_no_raw={p_n:.4f}")

    return p_y / (p_y + p_n) if (p_y + p_n) > 0 else 0.5


def main() -> None:
    INPUT_DIR = find_input_dir()
    manifest = json.loads((INPUT_DIR / "manifest.json").read_text())
    boxes = json.loads((INPUT_DIR / "boxes.json").read_text())

    print(f"device={DEVICE}  input_dir={INPUT_DIR}  images_dir={INPUT_DIR / 'images'}")
    processor, model = load_model()

    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    debug_calls_left = 3

    for img in manifest["images"]:
        prompt_id = img["prompt_id"]
        if not img.get("detected"):
            continue
        subject_boxes = boxes.get(str(prompt_id))
        if not subject_boxes:
            print(f"  p{prompt_id}: no box data, skipping")
            continue

        image_path = INPUT_DIR / "images" / f"p{prompt_id}.png"
        image = Image.open(image_path).convert("RGB")
        crops = {s: image.crop(tuple(b)) for s, b in subject_boxes.items()}

        per_attribute: Dict[str, Dict[str, float]] = {}
        for attr in img["attributes"]:
            question = attribute_question(attr["attribute"])
            scores: Dict[str, float] = {}
            for subject, crop in crops.items():
                debug = debug_calls_left > 0
                scores[subject] = p_yes(processor, model, crop, question, debug_print=debug)
                if debug:
                    debug_calls_left -= 1
            per_attribute[attr["attribute"]] = scores

        results[str(prompt_id)] = per_attribute
        OUT_PATH.write_text(json.dumps(results, indent=2))
        print(f"  p{prompt_id}: scored {len(per_attribute)} attribute(s) x "
              f"{len(crops)} subject(s)")

    print(f"\nDone. {len(results)}/{len(manifest['images'])} images scored -> {OUT_PATH}")


if __name__ == "__main__":
    main()
