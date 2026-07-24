#!/usr/bin/env python
"""
SDXL variant of Stage 1 (see generate_anchor_images.py, the SD1.5 original). Same 18
prompts, same candidate seeds, same NUM_INFERENCE_STEPS/EARLY_WINDOW_FRACTION -- only the
generation backend changes -- so any accuracy/coverage difference in the resulting anchor
set is attributable to model quality, not a confound from also changing the experimental
design.

Motivation: labeling the SD1.5 anchor set (artifacts/manifest.json) surfaced that human
`unclear` judgments coincide with near-flat attention scores across subjects, and visual
inspection of p1/p6 showed SD1.5 frequently fails to render visually DISTINCT subjects at
all for these multi-person prompts (e.g. two identical chefs instead of a chef + a farmer),
which makes the underlying binding question ill-posed for those images regardless of how
good the metric is. SDXL is still UNet-based, so the existing cross-attention hook
(AttentionStore/CustomAttnProcessor, unchanged from the SD1.5 script) works without
modification -- unlike FLUX.1/SD3, which use DiT joint-attention with no equivalent
`attn2`-style hook point and were already scoped out of this project for that reason (see
pilot/spatial_semantic_alignment.py's docstring).

This is an ADDITIONAL run, not a replacement: the SD1.5 manifest, images, and the 54
already-collected human labels stay as-is (real data, and the subject-collapse finding is
itself a real result). This script's output lands in a separate `artifacts_sdxl/` directory
locally (see label_images.py / analyze_agreement.py's --artifacts-dir flag) so the two
datasets can be compared side by side rather than one clobbering the other.

Two real SDXL-specific differences from the SD1.5 script, both flagged rather than silently
assumed:
  1. SDXL has TWO text encoders/tokenizers (tokenizer + tokenizer_2). Cross-attention
     conditions on their embeddings concatenated along the FEATURE dimension, not the
     sequence dimension -- so token *positions* in the 77-length sequence stay aligned
     between the two encoders for the same prompt text, and `pipeline.tokenizer` (the
     first/CLIP-ViT-L tokenizer, same family as SD1.5's) is what token_indices() should use
     to locate a phrase's position. This is architecturally sound but only verifiable by
     actually running it -- same "GPU-only, not unit-tested locally" boundary as the rest of
     this pipeline's model-dependent pieces.
  2. SDXL's native resolution is 1024x1024; running it at 512 is a known source of its own
     composition artifacts, so IMG_SIZE is raised to 1024 here (not kept at 512 to match the
     SD1.5 script) -- this is the one deliberate experimental-design difference, made because
     matching SD1.5's resolution would undersell exactly the quality improvement this run
     exists to test.
"""
import json
import math
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

if "KAGGLE_KERNEL_RUN_TYPE" in __import__("os").environ:
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--progress-bar", "off",
        "torch==2.5.1", "torchvision==0.20.1",
        "--index-url", "https://download.pytorch.org/whl/cu121",
    ])
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--progress-bar", "off",
        "diffusers>=0.31", "transformers>=4.44", "accelerate", "scipy",
    ])

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
CANDIDATE_SEEDS = [42, 7, 1234, 2024]  # identical to the SD1.5 script -- only the model differs
NUM_INFERENCE_STEPS = 30
EARLY_WINDOW_FRACTION = 0.5
MAX_STEPS = int(NUM_INFERENCE_STEPS * EARLY_WINDOW_FRACTION)
IMG_SIZE = 1024  # SDXL native resolution -- see module docstring point 2
DETECTION_SCORE_THRESH = 0.7  # unchanged

ARTIFACTS_DIR = Path("artifacts")  # isolated per-Kaggle-session path; see module docstring
IMAGES_DIR = ARTIFACTS_DIR / "images"
MANIFEST_PATH = ARTIFACTS_DIR / "manifest.json"

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
COCO_PERSON = 1

# =============================================================================
# ANCHOR_PROMPTS -- IDENTICAL to generate_anchor_images.py / prompt_specs.json. Must not
# drift; see tests/test_anchor_common.py's drift guard (extended to cover this file too).
# =============================================================================
ANCHOR_PROMPTS = [
    dict(id=0, n=2, prompt="a photo of a barista wearing a red apron and a cyclist wearing a yellow helmet",
         pairs=[("barista", "red apron"), ("cyclist", "yellow helmet")]),
    dict(id=1, n=2, prompt="a photo of a chef wearing a white hat and a farmer holding a shovel",
         pairs=[("chef", "white hat"), ("farmer", "shovel")]),
    dict(id=2, n=2, prompt="a photo of a nurse wearing blue gloves and a pilot wearing dark sunglasses",
         pairs=[("nurse", "blue gloves"), ("pilot", "dark sunglasses")]),
    dict(id=3, n=2, prompt="a photo of a teacher holding a book and a cyclist wearing a yellow helmet",
         pairs=[("teacher", "book"), ("cyclist", "yellow helmet")]),
    dict(id=4, n=2, prompt="a photo of a nurse wearing blue gloves and a farmer holding a shovel",
         pairs=[("nurse", "blue gloves"), ("farmer", "shovel")]),
    dict(id=5, n=2, prompt="a photo of a pilot wearing dark sunglasses and a barista wearing a red apron",
         pairs=[("pilot", "dark sunglasses"), ("barista", "red apron")]),
    dict(id=6, n=3, prompt="a photo of a barista wearing a red apron, a cyclist wearing a yellow helmet, and a chef wearing a white hat",
         pairs=[("barista", "red apron"), ("cyclist", "yellow helmet"), ("chef", "white hat")]),
    dict(id=7, n=3, prompt="a photo of a chef holding a pan, a farmer holding a shovel, and a nurse wearing blue gloves",
         pairs=[("chef", "pan"), ("farmer", "shovel"), ("nurse", "blue gloves")]),
    dict(id=8, n=3, prompt="a photo of a pilot wearing dark sunglasses, a teacher holding a book, and a barista wearing a red apron",
         pairs=[("pilot", "dark sunglasses"), ("teacher", "book"), ("barista", "red apron")]),
    dict(id=9, n=3, prompt="a photo of a nurse wearing blue gloves, a cyclist wearing a yellow helmet, and a farmer holding a shovel",
         pairs=[("nurse", "blue gloves"), ("cyclist", "yellow helmet"), ("farmer", "shovel")]),
    dict(id=10, n=3, prompt="a photo of a teacher holding a book, a pilot wearing dark sunglasses, and a chef wearing a white hat",
         pairs=[("teacher", "book"), ("pilot", "dark sunglasses"), ("chef", "white hat")]),
    dict(id=11, n=3, prompt="a photo of a barista wearing a red apron, a nurse wearing blue gloves, and a cyclist wearing a yellow helmet",
         pairs=[("barista", "red apron"), ("nurse", "blue gloves"), ("cyclist", "yellow helmet")]),
    dict(id=12, n=4, prompt="a photo of a barista wearing a red apron, a cyclist wearing a yellow helmet, a chef wearing a white hat, and a farmer holding a shovel",
         pairs=[("barista", "red apron"), ("cyclist", "yellow helmet"), ("chef", "white hat"), ("farmer", "shovel")]),
    dict(id=13, n=4, prompt="a photo of a nurse wearing blue gloves, a pilot wearing dark sunglasses, a teacher holding a book, and a chef holding a pan",
         pairs=[("nurse", "blue gloves"), ("pilot", "dark sunglasses"), ("teacher", "book"), ("chef", "pan")]),
    dict(id=14, n=4, prompt="a photo of a farmer holding a shovel, a barista wearing a red apron, a cyclist wearing a yellow helmet, and a nurse wearing blue gloves",
         pairs=[("farmer", "shovel"), ("barista", "red apron"), ("cyclist", "yellow helmet"), ("nurse", "blue gloves")]),
    dict(id=15, n=4, prompt="a photo of a chef wearing a white hat, a teacher holding a book, a pilot wearing dark sunglasses, and a cyclist wearing a yellow helmet",
         pairs=[("chef", "white hat"), ("teacher", "book"), ("pilot", "dark sunglasses"), ("cyclist", "yellow helmet")]),
    dict(id=16, n=4, prompt="a photo of a barista wearing a red apron, a nurse wearing blue gloves, a farmer holding a shovel, and a pilot wearing dark sunglasses",
         pairs=[("barista", "red apron"), ("nurse", "blue gloves"), ("farmer", "shovel"), ("pilot", "dark sunglasses")]),
    dict(id=17, n=4, prompt="a photo of a cyclist wearing a yellow helmet, a chef holding a pan, a teacher holding a book, and a nurse wearing blue gloves",
         pairs=[("cyclist", "yellow helmet"), ("chef", "pan"), ("teacher", "book"), ("nurse", "blue gloves")]),
]

# =============================================================================
# Attention capture -- verbatim duplicate of generate_anchor_images.py. Keep in sync. This
# class hierarchy is architecture-generic (any diffusers UNet2DConditionModel with attn2
# cross-attention processors), so it needs no SDXL-specific changes.
# =============================================================================

class AttentionRecord:
    __slots__ = ("tensor", "layer_name", "spatial_dim", "heads")

    def __init__(self, tensor: torch.Tensor, layer_name: str, spatial_dim: int, heads: int = 1):
        self.tensor = tensor
        self.layer_name = layer_name
        self.spatial_dim = spatial_dim
        self.heads = heads


class AttentionStore:
    def __init__(self):
        self.step_store: Dict[int, List[AttentionRecord]] = {}
        self.current_step: int = 0

    def reset(self):
        self.step_store = {}
        self.current_step = 0

    def add_attention(self, tensor: torch.Tensor, layer_name: str, spatial_dim: int, heads: int = 1):
        if self.current_step not in self.step_store:
            self.step_store[self.current_step] = []
        record = AttentionRecord(tensor.detach().cpu(), layer_name, spatial_dim, heads)
        self.step_store[self.current_step].append(record)

    def step(self):
        self.current_step += 1


class CustomAttnProcessor:
    def __init__(self, store: AttentionStore, is_cross_attention: bool, layer_name: str):
        self.store = store
        self.is_cross_attention = is_cross_attention
        self.layer_name = layer_name

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, temb=None, scale: float = 1.0):
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross is not None:
            encoder_hidden_states = attn.norm_cross(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        if self.is_cross_attention:
            spatial_dim = attention_probs.shape[1]
            self.store.add_attention(attention_probs, self.layer_name, spatial_dim, attn.heads)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


class AttentionCapture:
    def __init__(self):
        self.attn_store = AttentionStore()

    def hook_pipeline(self, pipeline):
        self.attn_store.reset()
        processors = {}
        for name in pipeline.unet.attn_processors.keys():
            is_cross = "attn2" in name
            processors[name] = CustomAttnProcessor(self.attn_store, is_cross, layer_name=name)
        pipeline.unet.set_attn_processor(processors)

    def unhook_pipeline(self, pipeline):
        from diffusers.models.attention_processor import AttnProcessor2_0
        pipeline.unet.set_attn_processor(AttnProcessor2_0())

    def phase_b_cross_attention_map(self, target_token_index: int,
                                     target_resolution: Tuple[int, int] = (IMG_SIZE, IMG_SIZE),
                                     max_steps: int = MAX_STEPS,
                                     cond_index: Optional[int] = None) -> np.ndarray:
        if not self.attn_store.step_store:
            raise ValueError("Attention store is empty. Did you call hook_pipeline() before generating?")
        per_res_accum: Dict[int, np.ndarray] = {}
        per_res_weight: Dict[int, float] = {}
        for step_idx, records in self.attn_store.step_store.items():
            if step_idx >= max_steps:
                continue
            for record in records:
                spatial_dim = record.spatial_dim
                seq_len = record.tensor.shape[2]
                native_side = int(math.sqrt(spatial_dim))
                if native_side * native_side != spatial_dim or target_token_index >= seq_len:
                    continue
                tensor = record.tensor
                if cond_index is not None:
                    start = cond_index * record.heads
                    tensor = tensor[start:start + record.heads]
                token_attn = tensor[:, :, target_token_index].mean(dim=0)
                attn_2d = token_attn.view(native_side, native_side).unsqueeze(0).unsqueeze(0)
                attn_up = F.interpolate(attn_2d.float(), size=target_resolution,
                                         mode="bilinear", align_corners=False).squeeze().numpy()
                per_res_accum.setdefault(native_side, np.zeros(target_resolution, dtype=np.float32))
                per_res_weight.setdefault(native_side, 0.0)
                per_res_accum[native_side] += attn_up
                per_res_weight[native_side] += 1.0
        if not per_res_accum:
            return np.zeros(target_resolution, dtype=np.float32)
        per_res_maps = {side: per_res_accum[side] / per_res_weight[side] for side in per_res_accum}
        composite = np.zeros(target_resolution, dtype=np.float32)
        total_weight = 0.0
        for side, layer_map in per_res_maps.items():
            w = float(side * side)
            composite += w * layer_map
            total_weight += w
        return composite / total_weight if total_weight > 0 else composite


# =============================================================================
# Pure logic -- INLINE COPIES of anchor_common.py. Keep in sync.
# =============================================================================

def mean_mass_in_box(attn_map: np.ndarray, box) -> float:
    h, w = attn_map.shape[:2]
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float(attn_map[y0:y1, x0:x1].mean())


def predicted_owner_from_attention(attn_map: np.ndarray, subject_boxes: Dict[str, list]):
    if not subject_boxes:
        raise ValueError("subject_boxes is empty; nothing to attribute the attention to")
    scores = {s: mean_mass_in_box(attn_map, b) for s, b in subject_boxes.items()}
    owner = max(scores, key=lambda s: scores[s])
    return owner, scores


def build_attribute_entry(attribute, intended_subject, predicted_owner, model_scores) -> dict:
    return dict(attribute=attribute, intended_subject=intended_subject,
                predicted_owner=predicted_owner, model_scores=model_scores)


def build_manifest_entry(prompt_id, n, prompt, subjects, seed, detected,
                         num_people_detected, image_path, attributes) -> dict:
    return dict(prompt_id=prompt_id, n=n, prompt=prompt, subjects=subjects, seed=seed,
                detected=detected, num_people_detected=num_people_detected,
                image_path=image_path, attributes=attributes)


# =============================================================================
# Model helpers
# =============================================================================

def token_indices(tokenizer, prompt: str, phrase: str) -> List[int]:
    """Unchanged from the SD1.5 script. For SDXL, callers must pass pipeline.tokenizer (the
    first/CLIP-ViT-L tokenizer) -- see module docstring point 1 on why sequence positions
    stay aligned with tokenizer_2's output for the same prompt text."""
    ids = tokenizer(prompt, padding="max_length", max_length=77, truncation=True).input_ids
    target = tokenizer(phrase, add_special_tokens=False).input_ids
    if not target:
        raise ValueError(f"phrase {phrase!r} tokenized to nothing")
    for i in range(len(ids) - len(target) + 1):
        if ids[i:i + len(target)] == target:
            return list(range(i, i + len(target)))
    raise ValueError(f"phrase {phrase!r} not found in prompt tokens: {prompt!r}")


def _load_sdxl_pipeline(model_id: str, cls):
    """SDXL's from_pretrained() does not accept safety_checker/requires_safety_checker
    (unlike the SD1.x pipelines in generate_anchor_images.py's _load_pipeline) -- separate
    loader rather than reusing that one. Tries the fp16 variant first (smaller download,
    matches DTYPE), falls back to the default variant if unavailable."""
    try:
        pipe = cls.from_pretrained(model_id, torch_dtype=DTYPE, use_safetensors=True,
                                   variant="fp16" if DTYPE == torch.float16 else None).to(DEVICE)
    except Exception as e:
        print(f"  fp16 variant unavailable ({type(e).__name__}: {e}); trying default variant")
        pipe = cls.from_pretrained(model_id, torch_dtype=DTYPE, use_safetensors=True).to(DEVICE)
    print(f"loaded {cls.__name__}: {model_id}")
    return pipe


class Models:
    def __init__(self, txt2img, detector, clip_model, clip_proc):
        self.txt2img = txt2img
        self.detector = detector
        self.clip_model = clip_model
        self.clip_proc = clip_proc


def load_all_models() -> Models:
    from diffusers import StableDiffusionXLPipeline
    from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
    from transformers import CLIPProcessor, CLIPModel

    txt2img = _load_sdxl_pipeline(MODEL_ID, StableDiffusionXLPipeline)
    txt2img.set_progress_bar_config(disable=True)
    detector = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT).to(DEVICE).eval()
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True).to(DEVICE).eval()
    clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return Models(txt2img, detector, clip_model, clip_proc)


@torch.no_grad()
def person_boxes(models: Models, image: Image.Image, max_people: int,
                  score_thresh: float = DETECTION_SCORE_THRESH) -> List[List[float]]:
    t = TF.to_tensor(image).to(DEVICE)
    out = models.detector([t])[0]
    keep = (out["labels"] == COCO_PERSON) & (out["scores"] >= score_thresh)
    boxes, scores = out["boxes"][keep], out["scores"][keep]
    if boxes.shape[0] == 0:
        return []
    order = scores.argsort(descending=True)[:max_people]
    return boxes[order].cpu().numpy().tolist()


@torch.no_grad()
def assign_subjects(models: Models, image: Image.Image, boxes: List[List[float]],
                     subject_names: List[str]) -> Dict[str, int]:
    from scipy.optimize import linear_sum_assignment

    crops = [image.crop(tuple(b)) for b in boxes]
    inp = models.clip_proc(text=[f"a photo of a {s}" for s in subject_names],
                            images=crops, return_tensors="pt", padding=True).to(DEVICE)
    sim = models.clip_model(**inp).logits_per_image.softmax(-1).cpu().numpy()
    r, c = linear_sum_assignment(-sim)
    return {subject_names[j]: int(i) for i, j in zip(r, c)}


def generate_and_score(spec: dict, capture: AttentionCapture, models: Models) -> dict:
    """Identical logic to generate_anchor_images.py's version -- only guidance_scale is kept
    explicit at 7.5 (matching the SD1.5 run) rather than SDXL's own library default, so the
    two runs differ only in model choice, not in generation hyperparameters."""
    prompt_id, n, prompt = spec["id"], spec["n"], spec["prompt"]
    subjects = [s for s, _ in spec["pairs"]]

    chosen_seed = CANDIDATE_SEEDS[-1]
    boxes: List[List[float]] = []
    image: Optional[Image.Image] = None
    for seed in CANDIDATE_SEEDS:
        capture.hook_pipeline(models.txt2img)

        def _cb(pipe, i, t, kw):
            capture.attn_store.step()
            return kw

        g = torch.Generator(DEVICE).manual_seed(seed)
        candidate = models.txt2img(prompt, num_inference_steps=NUM_INFERENCE_STEPS,
                                    guidance_scale=7.5, height=IMG_SIZE, width=IMG_SIZE,
                                    generator=g, callback_on_step_end=_cb).images[0]
        found = person_boxes(models, candidate, max_people=n)
        chosen_seed, image, boxes = seed, candidate, found
        if len(found) == n:
            break
        capture.unhook_pipeline(models.txt2img)
        print(f"  p{prompt_id} seed={seed}: detected {len(found)}/{n}, retrying")

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    image_path = str(IMAGES_DIR / f"p{prompt_id}.png")
    image.save(image_path)

    detected = len(boxes) == n
    if not detected:
        capture.unhook_pipeline(models.txt2img)
        print(f"  p{prompt_id}: DETECTION FAIL after all seeds -- found {len(boxes)}, expected {n}")
        return build_manifest_entry(prompt_id, n, prompt, subjects, chosen_seed, detected=False,
                                    num_people_detected=len(boxes), image_path=image_path,
                                    attributes=[])

    assign = assign_subjects(models, image, boxes, subjects)
    subject_boxes = {s: boxes[assign[s]] for s in subjects}

    attributes_out: List[dict] = []
    for subject, attribute in spec["pairs"]:
        idxs = token_indices(models.txt2img.tokenizer, prompt, attribute)
        maps = [capture.phase_b_cross_attention_map(target_token_index=i, cond_index=1) for i in idxs]
        attn_map = np.mean(maps, axis=0)
        owner, scores = predicted_owner_from_attention(attn_map, subject_boxes)
        attributes_out.append(build_attribute_entry(attribute, subject, owner, scores))
    capture.unhook_pipeline(models.txt2img)

    print(f"  p{prompt_id}: OK seed={chosen_seed} n={n} "
          + ", ".join(f"{a['attribute']}->{a['predicted_owner']}" for a in attributes_out))
    return build_manifest_entry(prompt_id, n, prompt, subjects, chosen_seed, detected=True,
                                num_people_detected=len(boxes), image_path=image_path,
                                attributes=attributes_out)


def main():
    print(f"device={DEVICE} dtype={DTYPE} seeds={CANDIDATE_SEEDS} prompts={len(ANCHOR_PROMPTS)} "
          f"model={MODEL_ID} img_size={IMG_SIZE}")
    capture = AttentionCapture()
    print(f"\nLoading models ({MODEL_ID}, Mask R-CNN, CLIP)...")
    models = load_all_models()

    manifest = {"model": MODEL_ID, "candidate_seeds": CANDIDATE_SEEDS, "img_size": IMG_SIZE,
                "num_inference_steps": NUM_INFERENCE_STEPS,
                "early_window_fraction": EARLY_WINDOW_FRACTION, "images": []}
    t0 = time.time()
    for spec in ANCHOR_PROMPTS:
        print(f"\n=== p{spec['id']} (n={spec['n']}): {spec['prompt'][:70]}... ===")
        try:
            entry = generate_and_score(spec, capture, models)
        except Exception as e:
            print(f"  p{spec['id']}: ERROR {type(e).__name__}: {e}")
            traceback.print_exc()
            entry = build_manifest_entry(spec["id"], spec["n"], spec["prompt"],
                                         [s for s, _ in spec["pairs"]], CANDIDATE_SEEDS[-1],
                                         detected=False, num_people_detected=0, image_path="",
                                         attributes=[])
        manifest["images"].append(entry)
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(MANIFEST_PATH, "w") as f:
            json.dump(manifest, f, indent=2)  # incremental save, survives a Kaggle timeout

    detected_count = sum(1 for e in manifest["images"] if e["detected"])
    print(f"\nDetected {detected_count}/{len(manifest['images'])} images in {time.time() - t0:.1f}s")
    print(f"Manifest written to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
