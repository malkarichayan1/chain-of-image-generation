#!/usr/bin/env python
"""
FLUX.1-dev variant of Stage 1 (images only, no cross-attention).

SPLITTING across 2 Kaggle notebooks (committed runs):
  Notebook 1: SPLIT_START=0,   SPLIT_END=53
  Notebook 2: SPLIT_START=53,  SPLIT_END=105
"""
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import os as _os
# Reduce CUDA memory fragmentation (recommended by PyTorch for large models)
_os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Auto-install deps on Kaggle
if "KAGGLE_KERNEL_RUN_TYPE" in _os.environ:
    import subprocess as _sp, sys as _sys
    _sp.check_call([_sys.executable, "-m", "pip", "install", "-q",
        "--force-reinstall", "--no-deps",
        "torch==2.5.1+cu121", "torchvision==0.20.1+cu121",
        "--index-url", "https://download.pytorch.org/whl/cu121"])
    _sp.check_call([_sys.executable, "-m", "pip", "install", "-q",
        "diffusers>=0.31", "transformers>=4.44", "bitsandbytes", "scipy", "sentencepiece"])
    _sp.check_call([_sys.executable, "-m", "pip", "install", "-q", "--no-deps", "accelerate"])

# Auto-install deps on Colab
elif "COLAB_RELEASE_TAG" in _os.environ:
    import subprocess as _sp, sys as _sys
    _sp.check_call([_sys.executable, "-m", "pip", "install", "-q",
        "diffusers>=0.31", "transformers>=4.44", "bitsandbytes", "scipy",
        "sentencepiece", "accelerate"])

import torch
import torchvision.transforms.functional as TF
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
CANDIDATE_SEEDS = [42, 7, 1234, 2024]
NUM_INFERENCE_STEPS = 25  # Full quality
IMG_SIZE = 1024            # Full resolution
DETECTION_SCORE_THRESH = 0.7

ARTIFACTS_DIR = Path("artifacts_flux")
IMAGES_DIR = ARTIFACTS_DIR / "images"
MANIFEST_PATH = ARTIFACTS_DIR / "manifest.json"

MODEL_ID = "black-forest-labs/FLUX.1-dev"
COCO_PERSON = 1

# Locate manifest.json (handles Kaggle's /kaggle/input/ paths if uploaded as a dataset)
manifest_path = Path("manifest.json")
if not manifest_path.exists() and Path("/kaggle/input").exists():
    found = list(Path("/kaggle/input").rglob("manifest.json"))
    if found:
        manifest_path = found[0]

ANCHOR_PROMPTS = []
try:
    with open(manifest_path, "r") as f:
        prior_manifest = json.load(f)
        for img in prior_manifest.get("images", []):
            if img.get("detected"):
                spec = {
                    "id": img["prompt_id"],
                    "n": img["n"],
                    "prompt": img["prompt"],
                    "pairs": [(attr["intended_subject"], attr["attribute"]) for attr in img["attributes"]],
                    "seed": img["seed"],
                }
                ANCHOR_PROMPTS.append(spec)
    print(f"Loaded {len(ANCHOR_PROMPTS)} prompts from {manifest_path}")
except FileNotFoundError:
    print(f"WARNING: manifest.json not found (looked at {manifest_path}). Please upload it to Kaggle.")


# =============================================================================
# Pure logic -- INLINE COPIES of anchor_common.py
# =============================================================================

def build_attribute_entry(attribute, intended_subject, predicted_owner, model_scores,
                          predicted_owner_full=None, model_scores_full=None) -> dict:
    return dict(attribute=attribute, intended_subject=intended_subject,
                predicted_owner=predicted_owner, model_scores=model_scores,
                predicted_owner_full=predicted_owner_full, model_scores_full=model_scores_full)


def build_manifest_entry(prompt_id, n, prompt, subjects, seed, detected,
                         num_people_detected, image_path, attributes) -> dict:
    return dict(prompt_id=prompt_id, n=n, prompt=prompt, subjects=subjects, seed=seed,
                detected=detected, num_people_detected=num_people_detected,
                image_path=image_path, attributes=attributes)


# =============================================================================
# Model helpers
# =============================================================================

class Models:
    def __init__(self, txt2img, detector, clip_model, clip_proc):
        self.txt2img = txt2img
        self.detector = detector
        self.clip_model = clip_model
        self.clip_proc = clip_proc


def load_all_models() -> Models:
    from diffusers import FluxPipeline, FluxTransformer2DModel
    from transformers import BitsAndBytesConfig, T5EncoderModel, CLIPProcessor, CLIPModel
    from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights

    hf_token = None
    if "KAGGLE_KERNEL_RUN_TYPE" in __import__("os").environ:
        try:
            from kaggle_secrets import UserSecretsClient
            hf_token = UserSecretsClient().get_secret("HF_TOKEN")
        except Exception as e:
            print(f"Warning: Could not get HF_TOKEN from Kaggle Secrets: {e}")
    elif "COLAB_RELEASE_TAG" in __import__("os").environ:
        try:
            from google.colab import userdata
            hf_token = userdata.get("HF_TOKEN")
        except Exception as e:
            print(f"Warning: Could not get HF_TOKEN from Colab Secrets: {e}")

    num_gpus = torch.cuda.device_count()
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if num_gpus > 0 else 0
    print(f"Detected {num_gpus} GPU(s), VRAM: {vram_gb:.0f}GB")

    try:
        if vram_gb >= 40:
            # === BIG GPU (A100/A40/etc): full precision, no quantization, max speed ===
            print("Big GPU detected! Loading full-precision FLUX.1-dev...")
            txt2img = FluxPipeline.from_pretrained(
                MODEL_ID, torch_dtype=DTYPE, token=hf_token)
            txt2img.to(DEVICE)
        elif num_gpus >= 2:
            # === DUAL GPU (Kaggle T4 x2): 4-bit, split across GPUs ===
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=DTYPE)
            print("Loading 4-bit T5 on GPU 1...")
            text_encoder_2 = T5EncoderModel.from_pretrained(
                MODEL_ID, subfolder="text_encoder_2",
                quantization_config=bnb_config, torch_dtype=DTYPE,
                device_map={"": "cuda:1"}, token=hf_token)
            print("Loading 4-bit Transformer on GPU 0...")
            transformer = FluxTransformer2DModel.from_pretrained(
                MODEL_ID, subfolder="transformer",
                quantization_config=bnb_config, torch_dtype=DTYPE,
                device_map={"": "cuda:0"}, token=hf_token)
            txt2img = FluxPipeline.from_pretrained(
                MODEL_ID, text_encoder_2=text_encoder_2, transformer=transformer,
                torch_dtype=DTYPE, token=hf_token)
            t5 = txt2img.text_encoder_2
            txt2img.text_encoder_2 = None
            txt2img.to("cuda:0")
            class DeviceMovingT5Wrapper:
                def __init__(self, t5_model):
                    self.t5 = t5_model
                    self.dtype = t5_model.dtype
                    self.device = t5_model.device
                def __call__(self, input_ids, **kwargs):
                    return self.t5(input_ids.to(self.device), **kwargs)
            txt2img.text_encoder_2 = DeviceMovingT5Wrapper(t5)
        else:
            # === SINGLE SMALL GPU: 4-bit + CPU offload ===
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=DTYPE)
            print("Loading 4-bit T5...")
            text_encoder_2 = T5EncoderModel.from_pretrained(
                MODEL_ID, subfolder="text_encoder_2",
                quantization_config=bnb_config, torch_dtype=DTYPE, token=hf_token)
            print("Loading 4-bit Transformer...")
            transformer = FluxTransformer2DModel.from_pretrained(
                MODEL_ID, subfolder="transformer",
                quantization_config=bnb_config, torch_dtype=DTYPE, token=hf_token)
            txt2img = FluxPipeline.from_pretrained(
                MODEL_ID, text_encoder_2=text_encoder_2, transformer=transformer,
                torch_dtype=DTYPE, token=hf_token)
            txt2img.enable_model_cpu_offload()

    except Exception as e:
        print(f"Failed to load FLUX.1-dev pipeline. Error: {e}")
        raise

    # Detector and CLIP run on CPU so FLUX gets the full GPU.
    detector = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT).to("cpu").eval()
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True).to("cpu").eval()
    clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return Models(txt2img, detector, clip_model, clip_proc)


@torch.no_grad()
def person_boxes(models: Models, image: Image.Image, max_people: int,
                 score_thresh: float = DETECTION_SCORE_THRESH) -> List[List[float]]:
    # Run on CPU (detector lives on CPU to leave VRAM free for FLUX)
    t = TF.to_tensor(image).to("cpu")
    out = models.detector([t])[0]
    keep = (out["labels"] == COCO_PERSON) & (out["scores"] >= score_thresh)
    boxes, scores = out["boxes"][keep], out["scores"][keep]
    if boxes.shape[0] == 0:
        return []
    order = scores.argsort(descending=True)[:max_people]
    return boxes[order].numpy().tolist()


@torch.no_grad()
def assign_subjects(models: Models, image: Image.Image, boxes: List[List[float]],
                    subject_names: List[str]) -> Dict[str, int]:
    from scipy.optimize import linear_sum_assignment

    crops = [image.crop(tuple(b)) for b in boxes]
    # Run on CPU (CLIP lives on CPU to leave VRAM free for FLUX)
    inp = models.clip_proc(text=[f"a photo of a {s}" for s in subject_names],
                           images=crops, return_tensors="pt", padding=True)
    sim = models.clip_model(**inp).logits_per_image.softmax(-1).numpy()
    r, c = linear_sum_assignment(-sim)
    return {subject_names[j]: int(i) for i, j in zip(r, c)}


def generate_and_score(spec: dict, models: Models) -> dict:
    prompt_id, n, prompt = spec["id"], spec["n"], spec["prompt"]
    subjects = [s for s, _ in spec["pairs"]]

    # Use the pinned seed from the manifest
    seed_pool = [spec["seed"]]

    chosen_seed = seed_pool[-1]
    boxes: List[List[float]] = []
    image: Optional[Image.Image] = None
    for seed in seed_pool:
        g = torch.Generator(DEVICE).manual_seed(seed)
        candidate = models.txt2img(prompt, num_inference_steps=NUM_INFERENCE_STEPS,
                                   height=IMG_SIZE, width=IMG_SIZE,
                                   generator=g).images[0]
        # Free GPU memory between generation and detection
        torch.cuda.empty_cache()
        found = person_boxes(models, candidate, max_people=n)
        chosen_seed, image, boxes = seed, candidate, found
        if len(found) == n:
            break
        print(f"  p{prompt_id} seed={seed}: detected {len(found)}/{n}")

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    image_path = str(IMAGES_DIR / f"p{prompt_id}.png")
    image.save(image_path)

    detected = len(boxes) == n
    if not detected:
        print(f"  p{prompt_id}: DETECTION FAIL -- found {len(boxes)}, expected {n}")
        return build_manifest_entry(prompt_id, n, prompt, subjects, chosen_seed, detected=False,
                                    num_people_detected=len(boxes), image_path=image_path,
                                    attributes=[])

    attributes_out: List[dict] = []
    for subject, attribute in spec["pairs"]:
        # Skipped cross-attention mapping for FLUX
        attributes_out.append(build_attribute_entry(attribute, subject, "unavailable", {}, "unavailable", {}))

    print(f"  p{prompt_id}: OK seed={chosen_seed} n={n}")
    return build_manifest_entry(prompt_id, n, prompt, subjects, chosen_seed, detected=True,
                                num_people_detected=len(boxes), image_path=image_path,
                                attributes=attributes_out)


def main():
    if not ANCHOR_PROMPTS:
        print("No prompts found. Please ensure manifest.json is uploaded and valid.")
        return

    prompts_to_run = ANCHOR_PROMPTS
    print(f"device={DEVICE} dtype={DTYPE} model={MODEL_ID} img_size={IMG_SIZE} steps={NUM_INFERENCE_STEPS}")
    print(f"Prompts: {len(prompts_to_run)}")
    print(f"\nLoading models ({MODEL_ID}, Mask R-CNN, CLIP)...")
    models = load_all_models()

    manifest = {"model": MODEL_ID, "candidate_seeds": CANDIDATE_SEEDS, "img_size": IMG_SIZE,
                "num_inference_steps": NUM_INFERENCE_STEPS,
                "early_window_fraction": 0.0, "images": []}
    t0 = time.time()
    for i, spec in enumerate(prompts_to_run):
        elapsed = time.time() - t0
        rate = elapsed / max(i, 1)
        remaining = rate * (len(prompts_to_run) - i)
        print(f"\n=== p{spec['id']} (n={spec['n']}) [{i+1}/{len(prompts_to_run)}] "
              f"~{remaining/60:.0f}min left: {spec['prompt'][:60]}... ===")
        try:
            entry = generate_and_score(spec, models)
        except Exception as e:
            print(f"  p{spec['id']}: ERROR {type(e).__name__}: {e}")
            traceback.print_exc()
            entry = build_manifest_entry(spec["id"], spec["n"], spec["prompt"],
                                         [s for s, _ in spec["pairs"]], spec["seed"],
                                         detected=False, num_people_detected=0, image_path="",
                                         attributes=[])
        manifest["images"].append(entry)
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(MANIFEST_PATH, "w") as f:
            json.dump(manifest, f, indent=2)

    detected_count = sum(1 for e in manifest["images"] if e["detected"])
    print(f"\nDone! {detected_count}/{len(manifest['images'])} detected in {time.time() - t0:.1f}s")
    print(f"Manifest written to {MANIFEST_PATH}")

    # Auto-zip immediately so the session can't expire before you download!
    import zipfile as _zf
    zip_path = Path("artifacts_flux.zip")
    print(f"\nZipping output to {zip_path} ...")
    with _zf.ZipFile(zip_path, "w", _zf.ZIP_DEFLATED) as zf:
        for fpath in ARTIFACTS_DIR.rglob("*"):
            if fpath.is_file():
                zf.write(fpath, fpath.relative_to(ARTIFACTS_DIR.parent))
    print(f"Done! Download '{zip_path}' from the Kaggle Output panel on the right.")



if __name__ == "__main__":
    main()
