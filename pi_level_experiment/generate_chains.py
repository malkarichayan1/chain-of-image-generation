#!/usr/bin/env python
"""
Stage 1 of the Part-B strengthening design (docs/part-b-strengthening-design.md):
prompts x seeds -> images + attention maps + manifest.json. Runs on GPU/Kaggle; pushed as
a single script kernel (same convention as run_chain_experiment.py, which this file does
NOT modify -- that script remains the untouched provenance record for the v4 result in
RESULTS.md). Only the attention-capture pieces of SpatialSemanticAlignment are duplicated
here (not segment()/phase_a/phase_c -- those are Stage 2/3's job, run later against the
images and attention maps this script persists to disk).

Two coverage fixes over the v4 run:
  - Detectability pre-flight: person detection on the base image happens BEFORE
    committing to build a chain, and pass/fail is recorded explicitly in the manifest for
    every (prompt, seed) pair -- coverage becomes reportable data, not a silent skip.
  - SEEDS = [42, 7, 1234]: three seeds (42 first, so the v4 chains stay reproducible
    within this run) x 9 prompts = up to 27 chains, instead of one seed x 9 prompts.

The seed list and detection threshold (person_boxes(score_thresh=0.7), unchanged from
v4) are frozen before any IoU is computed anywhere downstream -- that ordering is what
makes them pre-registered, per the design doc's Step 1.

--- Growth extension (2026-07-23, kernel v2) ---
Step 4's confirmatory run (SEEDS=[42,7,1234], kernel v1) scored 19/27 chains and left
attn_scrambled_sameattr at a near-miss (p=0.078, n_groups=7/9) because prompt_id 3 and 8
each detected on only one of three seeds, leaving neither with a same-attribute
different-seed partner to pair against. The fix deliberately is NOT extra seeds targeted
at just those two prompts -- choosing retries because they're the prompts behind a
near-significant result is outcome-driven selection, the exact thing this design has
pre-registered against elsewhere. Instead this run adds ONE new seed uniformly across all
9 prompts, same as the original three. SEEDS below holds only the new seed: the v1 seeds'
chains already exist in artifacts/manifest.json and are combined with this run's output
by merge_manifests.py after pullback, not regenerated on GPU.

--- Growth extension (2026-07-23, kernel v3) ---
The Part D review (RESULTS.md) found that attn_scrambled_crosschain/_sameattr only clear
significance under the clustered per-prompt test, on n_groups=9 -- near Wilcoxon's n=9
one-sided floor of 1/512. Growing that test's power needs more independent PROMPT groups,
not more seeds on the existing 9 (the clustered test pairs by prompt_id, so extra seeds on
an existing prompt don't add another group). GROWTH_V3_PROMPTS below adds 6 new n=2
prompts (prompt_id 9-14) -- n=2 chosen deliberately: Part D's own selection-effect check
found detection rate falls from 92% (n=2) to 58% (n=4), so new prompts at the easiest
subject count maximize the odds these actually detect rather than adding more undetected
attempts. Each new prompt reuses the same validated 7-subject/8-attribute vocabulary as
MECHANISM_PROMPTS in a subject pairing not already used at n=2 (0/1/2 already cover
barista+cyclist, chef+farmer, nurse+pilot) -- reusing the vocabulary keeps
`select_foreign_attribute`'s disjointness logic and the calibrated T=0.85 threshold valid
without re-validation. SEEDS holds all four seeds used so far ([42,7,1234,2024]) so the
new prompts get the same seed-depth the existing 9 already have; the main loop below runs
ONLY GROWTH_V3_PROMPTS (prompt_id continuing from 9), not MECHANISM_PROMPTS again --
prompts 0-8 already exist in artifacts/manifest_combined.json and are combined with this
run's output by merge_manifests.py after pullback, not regenerated on GPU.
"""
import json
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
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

import math
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
# v1 (pre-registered, kernel v1): [42, 7, 1234] -- already built, see artifacts/manifest.json.
# v2 (growth extension): one new seed, applied uniformly to all 9 prompts -- already built,
# see artifacts/manifest_combined.json.
# v3 (this run): all four seeds so far, applied to the 6 new GROWTH_V3_PROMPTS only (see
# that list's comment and the module docstring's "kernel v3" section) -- prompts 0-8 are
# NOT regenerated.
SEEDS = [42, 7, 1234, 2024]
NUM_INFERENCE_STEPS = 30
EARLY_WINDOW_FRACTION = 0.5
MAX_STEPS = int(NUM_INFERENCE_STEPS * EARLY_WINDOW_FRACTION)
IMG_SIZE = 512
DETECTION_SCORE_THRESH = 0.7  # unchanged from v4 -- frozen before any IoU is computed

ARTIFACTS_DIR = Path("artifacts")
IMAGES_DIR = ARTIFACTS_DIR / "images"
ATTENTION_DIR = ARTIFACTS_DIR / "attention"
MANIFEST_PATH = ARTIFACTS_DIR / "manifest.json"

# =============================================================================
# Attention capture only -- duplicated from pilot/spatial_semantic_alignment.py's
# AttentionStore/CustomAttnProcessor/hook_pipeline/phase_b_cross_attention_map. Stage 1
# needs attention maps saved to disk, not scored, so segment()/phase_a/phase_c are
# deliberately omitted here (Stage 2/3's job). Keep in sync with that file.
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
# MECHANISM_PROMPTS -- identical to run_chain_experiment.py; kept in sync deliberately
# so both scripts build chains from the same validated prompt set.
# =============================================================================

MECHANISM_PROMPTS = [
    dict(n=2, prompt="a photo of a barista wearing a red apron and a cyclist wearing a yellow helmet",
         pairs=[("barista", "red apron", "torso"), ("cyclist", "yellow helmet", "head")]),
    dict(n=2, prompt="a photo of a chef wearing a white hat and a farmer holding a shovel",
         pairs=[("chef", "white hat", "head"), ("farmer", "shovel", "hands")]),
    dict(n=2, prompt="a photo of a nurse wearing blue gloves and a pilot wearing dark sunglasses",
         pairs=[("nurse", "blue gloves", "hands"), ("pilot", "dark sunglasses", "head")]),
    dict(n=3, prompt="a photo of a barista wearing a red apron, a cyclist wearing a yellow helmet, and a chef holding a pan",
         pairs=[("barista", "red apron", "torso"), ("cyclist", "yellow helmet", "head"), ("chef", "pan", "hands")]),
    dict(n=3, prompt="a photo of a chef wearing a white hat, a farmer holding a shovel, and a nurse wearing blue gloves",
         pairs=[("chef", "white hat", "head"), ("farmer", "shovel", "hands"), ("nurse", "blue gloves", "hands")]),
    dict(n=3, prompt="a photo of a pilot wearing dark sunglasses, a teacher holding a book, and a barista wearing a red apron",
         pairs=[("pilot", "dark sunglasses", "head"), ("teacher", "book", "hands"), ("barista", "red apron", "torso")]),
    dict(n=4, prompt="a photo of a barista wearing a red apron, a cyclist wearing a yellow helmet, a chef wearing a white hat, and a farmer holding a shovel",
         pairs=[("barista", "red apron", "torso"), ("cyclist", "yellow helmet", "head"), ("chef", "white hat", "head"), ("farmer", "shovel", "hands")]),
    dict(n=4, prompt="a photo of a nurse wearing blue gloves, a pilot wearing dark sunglasses, a chef holding a pan, and a teacher holding a book",
         pairs=[("nurse", "blue gloves", "hands"), ("pilot", "dark sunglasses", "head"), ("chef", "pan", "hands"), ("teacher", "book", "hands")]),
    dict(n=4, prompt="a photo of a farmer holding a shovel, a barista wearing a red apron, a cyclist wearing a yellow helmet, and a nurse wearing blue gloves",
         pairs=[("farmer", "shovel", "hands"), ("barista", "red apron", "torso"), ("cyclist", "yellow helmet", "head"), ("nurse", "blue gloves", "hands")]),
]

# Growth extension v3 (2026-07-23, kernel v3) -- see module docstring's "kernel v3" section
# for why these are new n=2 PROMPTS rather than new seeds. Assigned prompt_id 9-14 (main()
# enumerates this list with start=len(MECHANISM_PROMPTS)). Every subject/attribute/region
# triple is copied verbatim from MECHANISM_PROMPTS above -- only the pairing is new -- and
# no subject pair here duplicates an existing n=2 prompt (0: barista+cyclist, 1: chef+farmer,
# 2: nurse+pilot).
GROWTH_V3_PROMPTS = [
    dict(n=2, prompt="a photo of a teacher holding a book and a chef holding a pan",
         pairs=[("teacher", "book", "hands"), ("chef", "pan", "hands")]),
    dict(n=2, prompt="a photo of a barista wearing a red apron and a farmer holding a shovel",
         pairs=[("barista", "red apron", "torso"), ("farmer", "shovel", "hands")]),
    dict(n=2, prompt="a photo of a nurse wearing blue gloves and a teacher holding a book",
         pairs=[("nurse", "blue gloves", "hands"), ("teacher", "book", "hands")]),
    dict(n=2, prompt="a photo of a pilot wearing dark sunglasses and a cyclist wearing a yellow helmet",
         pairs=[("pilot", "dark sunglasses", "head"), ("cyclist", "yellow helmet", "head")]),
    dict(n=2, prompt="a photo of a chef wearing a white hat and a nurse wearing blue gloves",
         pairs=[("chef", "white hat", "head"), ("nurse", "blue gloves", "hands")]),
    dict(n=2, prompt="a photo of a farmer holding a shovel and a pilot wearing dark sunglasses",
         pairs=[("farmer", "shovel", "hands"), ("pilot", "dark sunglasses", "head")]),
]

COCO_PERSON = 1
SD15_TXT2IMG_CANDIDATES = [
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    "sd-legacy/stable-diffusion-v1-5",
]


def base_prompt_for(spec) -> str:
    subjects = [s for s, _, _ in spec["pairs"]]
    if len(subjects) == 1:
        return f"a photo of a {subjects[0]}"
    if len(subjects) == 2:
        return f"a photo of a {subjects[0]} and a {subjects[1]}"
    return "a photo of a " + ", a ".join(subjects[:-1]) + f", and a {subjects[-1]}"


def token_indices(tokenizer, prompt: str, phrase: str) -> List[int]:
    ids = tokenizer(prompt, padding="max_length", max_length=77, truncation=True).input_ids
    target = tokenizer(phrase, add_special_tokens=False).input_ids
    if not target:
        raise ValueError(f"phrase {phrase!r} tokenized to nothing")
    for i in range(len(ids) - len(target) + 1):
        if ids[i:i + len(target)] == target:
            return list(range(i, i + len(target)))
    raise ValueError(f"phrase {phrase!r} not found in prompt tokens: {prompt!r}")


def manifest_chain_entry(chain_key: str, prompt_id: int, seed: int, spec: dict, detected: bool,
                          num_people_detected: int, subject_boxes: Dict[str, List[float]],
                          base_image_path: str, steps: List[dict]) -> dict:
    """Pure dict assembly, no model calls -- the piece of Stage 1 that gets real test
    coverage per the design doc ("manifest schema" is explicitly listed as testable)."""
    return dict(
        chain_key=chain_key, prompt_id=prompt_id, seed=seed, n=spec["n"], prompt=spec["prompt"],
        detected=detected, num_people_detected=num_people_detected,
        subject_boxes=subject_boxes, base_image_path=base_image_path, steps=steps,
    )


@dataclass
class Models:
    txt2img: object
    detector: object
    clip_model: object
    clip_proc: object


def _load_pipeline(candidates, cls):
    for repo in candidates:
        try:
            pipe = cls.from_pretrained(repo, torch_dtype=DTYPE, safety_checker=None,
                                        requires_safety_checker=False).to(DEVICE)
            print(f"loaded {cls.__name__}: {repo}")
            return pipe
        except Exception as e:
            print(f"  {repo} unavailable ({type(e).__name__}: {e})")
    raise RuntimeError(f"No mirror loaded for {cls.__name__} from {candidates}")


def load_all_models() -> Models:
    from diffusers import StableDiffusionPipeline
    from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
    from transformers import CLIPProcessor, CLIPModel

    txt2img = _load_pipeline(SD15_TXT2IMG_CANDIDATES, StableDiffusionPipeline)
    txt2img.set_progress_bar_config(disable=True)
    detector = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT).to(DEVICE).eval()
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True).to(DEVICE).eval()
    clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return Models(txt2img=txt2img, detector=detector, clip_model=clip_model, clip_proc=clip_proc)


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


def box_mask(image: Image.Image, box: List[float], pad_frac: float = 0.15) -> Image.Image:
    w, h = image.size
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    x0 = max(0, x0 - bw * pad_frac)
    y0 = max(0, y0 - bh * pad_frac)
    x1 = min(w, x1 + bw * pad_frac)
    y1 = min(h, y1 + bh * pad_frac)
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rectangle([x0, y0, x1, y1], fill=255)
    return mask


@torch.no_grad()
def encode_to_latents(pipe, image: Image.Image) -> torch.Tensor:
    img_t = TF.to_tensor(image).unsqueeze(0).to(device=DEVICE, dtype=DTYPE)
    img_t = img_t * 2.0 - 1.0
    latents = pipe.vae.encode(img_t).latent_dist.sample()
    return latents * pipe.vae.config.scaling_factor


def mask_to_latent_tensor(mask_image: Image.Image, latent_hw: Tuple[int, int]) -> torch.Tensor:
    lh, lw = latent_hw
    small = mask_image.resize((lw, lh), Image.NEAREST)
    arr = (np.array(small).astype(np.float32) / 255.0 > 0.5).astype(np.float32)
    t = torch.from_numpy(arr).to(device=DEVICE, dtype=DTYPE)
    return t.unsqueeze(0).unsqueeze(0)


def build_and_persist_chain(prompt_id: int, seed: int, spec: dict, capture: AttentionCapture,
                             models: Models) -> dict:
    """Pre-flight (base image + detection) happens first and is always recorded in the
    returned manifest entry, whether or not detection passes -- coverage is reportable
    data, not a silent skip (design doc, Step 1)."""
    chain_key = f"p{prompt_id}_s{seed}"
    base_prompt = base_prompt_for(spec)
    subjects = [s for s, _, _ in spec["pairs"]]

    g = torch.Generator(DEVICE).manual_seed(seed)
    base_image = models.txt2img(base_prompt, num_inference_steps=NUM_INFERENCE_STEPS,
                                 guidance_scale=7.5, generator=g).images[0]
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    base_image_path = str(IMAGES_DIR / f"{chain_key}_step0.png")
    base_image.save(base_image_path)

    boxes = person_boxes(models, base_image, max_people=spec["n"])
    detected = len(boxes) == spec["n"]
    if not detected:
        print(f"  {chain_key}: DETECTION FAIL -- found {len(boxes)}, expected {spec['n']}")
        return manifest_chain_entry(chain_key, prompt_id, seed, spec, detected=False,
                                     num_people_detected=len(boxes), subject_boxes={},
                                     base_image_path=base_image_path, steps=[])

    assign = assign_subjects(models, base_image, boxes, subjects)
    subject_boxes = {s: boxes[assign[s]] for s in subjects}

    ATTENTION_DIR.mkdir(parents=True, exist_ok=True)
    steps_out: List[dict] = []
    current = base_image

    for step_idx, (subject, attribute, _region) in enumerate(spec["pairs"], start=1):
        mask_img = box_mask(current, subject_boxes[subject])
        inpaint_prompt = f"a {subject} with a {attribute}"

        orig_latents = encode_to_latents(models.txt2img, current)
        lh, lw = orig_latents.shape[-2], orig_latents.shape[-1]
        mask_latent = mask_to_latent_tensor(mask_img, (lh, lw))
        blend_noise = torch.randn(orig_latents.shape, generator=torch.Generator(DEVICE).manual_seed(seed),
                                   device=DEVICE, dtype=DTYPE)

        capture.hook_pipeline(models.txt2img)

        def _cb(pipe, i, t, kw):
            capture.attn_store.step()
            latents = kw["latents"]
            timesteps = pipe.scheduler.timesteps
            if i + 1 < len(timesteps):
                noised_orig = pipe.scheduler.add_noise(orig_latents, blend_noise, timesteps[i + 1].unsqueeze(0))
            else:
                noised_orig = orig_latents
            kw["latents"] = mask_latent * latents + (1.0 - mask_latent) * noised_orig
            return kw

        g = torch.Generator(DEVICE).manual_seed(seed)
        edited = models.txt2img(prompt=inpaint_prompt, num_inference_steps=NUM_INFERENCE_STEPS,
                                 guidance_scale=7.5, generator=g, callback_on_step_end=_cb).images[0]

        idxs = token_indices(models.txt2img.tokenizer, inpaint_prompt, attribute)
        maps = [capture.phase_b_cross_attention_map(target_token_index=i, cond_index=1) for i in idxs]
        attn_map = np.mean(maps, axis=0)
        capture.unhook_pipeline(models.txt2img)

        image_path = str(IMAGES_DIR / f"{chain_key}_step{step_idx}.png")
        edited.save(image_path)
        attention_path = str(ATTENTION_DIR / f"{chain_key}_step{step_idx}.npy")
        np.save(attention_path, attn_map)

        steps_out.append(dict(step_idx=step_idx, subject=subject, attribute=attribute,
                               image_path=image_path, attention_path=attention_path))
        current = edited

    return manifest_chain_entry(chain_key, prompt_id, seed, spec, detected=True,
                                 num_people_detected=len(boxes), subject_boxes=subject_boxes,
                                 base_image_path=base_image_path, steps=steps_out)


def main():
    print(f"device={DEVICE} dtype={DTYPE} seeds={SEEDS}")
    capture = AttentionCapture()
    print("\nLoading models (SD1.5 txt2img, Mask R-CNN, CLIP)...")
    models = load_all_models()

    manifest = {"seeds": SEEDS, "img_size": IMG_SIZE, "chains": []}
    t0 = time.time()
    for seed in SEEDS:
        for prompt_id, spec in enumerate(GROWTH_V3_PROMPTS, start=len(MECHANISM_PROMPTS)):
            print(f"\n=== chain p{prompt_id}_s{seed}: {spec['prompt'][:70]}... ===")
            try:
                entry = build_and_persist_chain(prompt_id, seed, spec, capture, models)
                manifest["chains"].append(entry)
            except Exception as e:
                print(f"  p{prompt_id}_s{seed}: ERROR {type(e).__name__}: {e}")
                traceback.print_exc()
                manifest["chains"].append(manifest_chain_entry(
                    f"p{prompt_id}_s{seed}", prompt_id, seed, spec, detected=False,
                    num_people_detected=0, subject_boxes={}, base_image_path="", steps=[],
                ))
            ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
            with open(MANIFEST_PATH, "w") as f:
                json.dump(manifest, f, indent=2)  # incremental save, survives a Kaggle timeout

    detected_count = sum(1 for c in manifest["chains"] if c["detected"])
    print(f"\nBuilt {detected_count}/{len(manifest['chains'])} chains in {time.time() - t0:.1f}s")
    print(f"Manifest written to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
