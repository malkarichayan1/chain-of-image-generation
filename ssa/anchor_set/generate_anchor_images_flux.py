#!/usr/bin/env python
"""
FLUX.1-dev variant of Stage 1, WITH cross-attention capture (double blocks only -- see
flux_attention_capture.py and docs/superpowers/specs/2026-07-31-flux-attention-hook-design.md).

Regenerates the SAME 105 images as the original run (same prompts, same pinned seeds from
manifest.json) so the existing human-labeled rows stay valid -- see the design doc's pilot
protocol before running this against the full prompt set.

SPLITTING across 2 Kaggle notebooks (committed runs):
  Notebook 1: SPLIT_START=0,   SPLIT_END=53
  Notebook 2: SPLIT_START=53,  SPLIT_END=105
"""
import json
import math
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

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
        "diffusers==0.39.0", "transformers>=4.44", "bitsandbytes", "scipy", "sentencepiece"])
    _sp.check_call([_sys.executable, "-m", "pip", "install", "-q", "--no-deps", "accelerate"])

# Auto-install deps on Colab
elif "COLAB_RELEASE_TAG" in _os.environ:
    import subprocess as _sp, sys as _sys
    _sp.check_call([_sys.executable, "-m", "pip", "install", "-q",
        "diffusers==0.39.0", "transformers>=4.44", "bitsandbytes", "scipy",
        "sentencepiece", "accelerate"])

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
CANDIDATE_SEEDS = [42, 7, 1234, 2024]
NUM_INFERENCE_STEPS = 25  # Full quality
EARLY_WINDOW_FRACTION = 0.5  # matches generate_anchor_images_sdxl.py's convention
MAX_STEPS = int(NUM_INFERENCE_STEPS * EARLY_WINDOW_FRACTION)
IMG_SIZE = 1024            # Full resolution
DETECTION_SCORE_THRESH = 0.7

ARTIFACTS_DIR = Path("artifacts_flux_hard")
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


def mean_mass_in_box(attn_map: np.ndarray, box: Sequence[float]) -> float:
    """Mean attention value inside an [x0, y0, x1, y1] pixel box. Returns 0.0 for a box
    that is empty or entirely off the map, so it can never spuriously win the argmax."""
    h, w = attn_map.shape[:2]
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float(attn_map[y0:y1, x0:x1].mean())


def locate_attribute_phrase(prompt: str, attribute: str) -> Tuple[int, str]:
    """Where `attribute` actually occurs in `prompt`, and what substring to treat as "the
    phrase" for downstream token lookup. Tries an exact substring match first (unchanged
    behavior for every attribute that already matches verbatim), but anchored to word
    boundaries (`\\b`) so a single-word `attribute` like "red" can never match inside a
    larger word like "shredded". Falls back to the SHORTEST left-to-right span that contains
    every content word of `attribute`, each matched as a whole word (also `\\b`-anchored) --
    handles manifest attribute strings that are a strict sub-phrase of a longer descriptive
    phrase in the actual prompt (e.g. attribute="yellow helmet", prompt="...yellow bike
    helmet...").  Taking the shortest span, rather than the first occurrence of each word, is
    what keeps two subjects sharing a word (e.g. both "wearing something red") from producing
    a span that stretches across the unrelated clause in between -- see the regression tests
    for the concrete case this fixes. Raises ValueError if any content word is missing
    anywhere in the prompt, if `attribute` has no content words at all, or if the words that
    ARE present never appear in a valid left-to-right order.

    Known limitation, currently dormant (verified against every real prompt in
    prompt_specs.json / artifacts_flux/manifest.json / artifacts_sdxl/*.json -- zero
    occurrences): the shortest-span fallback is scoped to the whole prompt, not to any one
    subject's clause. If two different subjects' clauses both happen to contain all of
    `attribute`'s content words (e.g. two subjects near each other each mentioning "red" and
    "apron" in different combinations), this can silently pick a span from the wrong
    subject's clause -- this function has no clause-boundary information to disambiguate
    with. A caller working with less-controlled prompt text that wants a stronger guarantee
    should pre-scope `prompt` to the relevant subject's clause before calling."""
    words = re.findall(r"[a-zA-Z]+", attribute.lower())
    if not words:
        raise ValueError(f"attribute {attribute!r} has no content words to match")

    exact = re.search(r"\b" + re.escape(attribute) + r"\b", prompt)
    if exact:
        return exact.start(), attribute

    prompt_lower = prompt.lower()
    occurrences: List[List[Tuple[int, int]]] = []
    for word in words:
        matches = [(m.start(), m.end())
                   for m in re.finditer(r"\b" + re.escape(word) + r"\b", prompt_lower)]
        if not matches:
            raise ValueError(
                f"attribute {attribute!r} word {word!r} not found in prompt {prompt!r}")
        occurrences.append(matches)

    best_span: Optional[Tuple[int, int]] = None
    for start0, end0 in occurrences[0]:
        cursor = end0
        valid = True
        for matches in occurrences[1:]:
            next_match = next((m for m in matches if m[0] >= cursor), None)
            if next_match is None:
                valid = False
                break
            cursor = next_match[1]
        if not valid:
            continue
        if best_span is None or (cursor - start0) < (best_span[1] - best_span[0]):
            best_span = (start0, cursor)

    if best_span is None:
        raise ValueError(
            f"attribute {attribute!r} words all appear in prompt {prompt!r} but never in a "
            f"valid left-to-right order")
    start, end = best_span
    return start, prompt[start:end]


def predicted_owner_from_attention(
    attn_map: np.ndarray, subject_boxes: Dict[str, Sequence[float]]
) -> Tuple[str, Dict[str, float]]:
    """The metric's binding call for one attribute: the subject whose detected box holds
    the most of that attribute's (early-window) cross-attention mass. Returns
    (predicted_owner, {subject: mass}). Ties break deterministically by the boxes' dict
    order (insertion order), so a rerun on identical inputs is reproducible."""
    if not subject_boxes:
        raise ValueError("subject_boxes is empty; nothing to attribute the attention to")
    scores = {s: mean_mass_in_box(attn_map, b) for s, b in subject_boxes.items()}
    owner = max(scores, key=lambda s: scores[s])
    return owner, scores


# =============================================================================
# Attention capture -- verbatim duplicate of flux_attention_capture.py. Keep in sync.
# =============================================================================

class FluxAttentionStore:
    """Per-step, per-layer attention captured for a fixed set of target T5 token columns,
    already head-averaged and detached to CPU. `target_token_indices` must be set via reset()
    before generation starts -- pass the UNION of every attribute's token indices for the
    current prompt, so one generation pass captures everything every attribute will need.

    HARD PRECONDITION: batch size must be 1. If called with batch > 1, generation output
    stays correct but captured attention silently keeps only the first image's data with no
    error. See FluxCustomAttnProcessor.__call__ for the assert that enforces this.

    UNHANDLED CASE: classifier-free guidance (negative_prompt + true_cfg_scale > 1) calls
    the transformer twice per denoising step (cond, then uncond) as separate batch-1 passes.
    If ever enabled, the second call would silently overwrite the first, corrupting captured
    data. This module keys by (current_step, layer_name) and does not differentiate cond vs.
    uncond branches. Future maintainers adding CFG support should either record both branches
    separately or raise an error on CFG."""

    def __init__(self):
        self.step_store: Dict[int, Dict[str, torch.Tensor]] = {}
        self.current_step: int = 0
        self.target_token_indices: List[int] = []

    def reset(self, target_token_indices: Sequence[int]) -> None:
        if not target_token_indices:
            raise ValueError("target_token_indices must be non-empty")
        self.step_store = {}
        self.current_step = 0
        self.target_token_indices = list(target_token_indices)

    def add_attention(self, layer_name: str, head_averaged: torch.Tensor) -> None:
        """`head_averaged`: (image_seq_len, n_targets) for one layer at the current step."""
        self.step_store.setdefault(self.current_step, {})[layer_name] = head_averaged.detach().float().cpu()

    def step(self) -> None:
        self.current_step += 1


class FluxCustomAttnProcessor:
    """Manual recompute of FluxAttnProcessor's math (see module docstring) for ONE double
    block. Captures attn_probs[image_rows, store.target_token_indices], averaged over heads,
    into `store`; otherwise produces output equivalent to the stock processor (verified, see
    module docstring) so hooked generation is unaffected.

    Assumes batch size 1 (see FluxAttentionStore HARD PRECONDITION). Enforced by assert
    before self.store.add_attention(...)."""

    def __init__(self, store: FluxAttentionStore, layer_name: str):
        self.store = store
        self.layer_name = layer_name

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, image_rotary_emb=None):
        from diffusers.models.embeddings import apply_rotary_emb
        from diffusers.models.transformers.transformer_flux import _get_qkv_projections

        query, key, value, encoder_query, encoder_key, encoder_value = _get_qkv_projections(
            attn, hidden_states, encoder_hidden_states)
        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))
        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if attn.added_kv_proj_dim is not None:
            encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
            encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))
            encoder_query = attn.norm_added_q(encoder_query)
            encoder_key = attn.norm_added_k(encoder_key)
            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        q = query.permute(0, 2, 1, 3)   # (batch, heads, seq, head_dim)
        k = key.permute(0, 2, 1, 3)
        v = value.permute(0, 2, 1, 3)
        scale = 1.0 / math.sqrt(q.shape[-1])
        attn_probs = torch.softmax((q @ k.transpose(-1, -2)) * scale, dim=-1)

        if encoder_hidden_states is not None:
            text_len = encoder_hidden_states.shape[1]
            image_rows = attn_probs[:, :, text_len:, :]                          # (b, heads, img, seq)
            target_cols = image_rows[..., self.store.target_token_indices]       # (b, heads, img, n_targets)
            assert query.shape[0] == 1, "FluxCustomAttnProcessor assumes batch size 1 (see module docstring)"
            self.store.add_attention(self.layer_name, target_cols.mean(dim=1)[0])  # (img, n_targets)

        hidden_states = attn_probs @ v
        hidden_states = hidden_states.permute(0, 2, 1, 3).flatten(2, 3).to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [encoder_hidden_states.shape[1], hidden_states.shape[1] - encoder_hidden_states.shape[1]],
                dim=1)
            hidden_states = attn.to_out[0](hidden_states.contiguous())
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states.contiguous())
            return hidden_states, encoder_hidden_states
        return hidden_states


class FluxAttentionCapture:
    """hook_pipeline/unhook_pipeline/cross_attention_map, mirroring
    generate_anchor_images_sdxl.py's AttentionCapture -- but FLUX-specific: only the 19 double
    blocks are hooked (see module docstring), and there is exactly one native image resolution
    across every hooked layer, so aggregation across (steps x layers) is a plain mean, no
    per-resolution weighting."""

    def __init__(self):
        self.store = FluxAttentionStore()

    def hook_pipeline(self, pipeline, target_token_indices: Sequence[int]) -> None:
        """Wires FluxCustomAttnProcessor onto every double block (`transformer_blocks.*`) of
        `pipeline.transformer`. `target_token_indices` are RAW T5 sequence positions (indices into
        the tokenizer's output for the full prompt) -- NOT positions into any already-captured
        tensor; that translation (raw T5 index -> captured-column position) is the caller's job,
        see `cross_attention_map`'s `target_indices_position` parameter. Safe to call again
        without an intervening `unhook_pipeline()`: re-wraps from the pipeline's current
        attn_processors and calls `store.reset()`, so there is no double-wrapping or leaked state
        from a prior hook."""
        self.store.reset(target_token_indices)
        processors = dict(pipeline.transformer.attn_processors)
        for name in processors:
            if name.startswith("transformer_blocks."):
                processors[name] = FluxCustomAttnProcessor(self.store, name)
        pipeline.transformer.set_attn_processor(processors)

    def unhook_pipeline(self, pipeline) -> None:
        """Resets ALL of `pipeline.transformer`'s attention processors -- both double
        (`transformer_blocks.*`) and single (`single_transformer_blocks.*`) blocks -- back to the
        stock FluxAttnProcessor. This is NOT a scoped inverse of hook_pipeline's surgical
        double-block-only replacement; it's a full reset to default, deliberately matching
        generate_anchor_images_sdxl.py's broadcast-a-single-instance unhook pattern."""
        from diffusers.models.transformers.transformer_flux import FluxAttnProcessor
        pipeline.transformer.set_attn_processor(FluxAttnProcessor())

    def cross_attention_map(self, target_indices_position: Sequence[int],
                            target_resolution: Tuple[int, int], max_steps: int) -> np.ndarray:
        """Average, over every captured double-block layer at steps < max_steps, the columns at
        `target_indices_position` (indices INTO store.target_token_indices -- the position this
        attribute's tokens occupy within the columns that were actually captured, NOT raw T5
        sequence positions), then upsample from the native image grid to target_resolution.
        Returns an all-zero map (not an error) when max_steps excludes every captured step --
        distinct from an empty store, which IS an error (nothing was ever captured).

        Raises ValueError if target_indices_position is empty, if any entry is negative or
        out-of-range for the columns actually captured (numpy would otherwise silently treat a
        negative index as wrapping from the end rather than erroring), or if captured layers/steps
        disagree on the native image grid size (should not happen per this module's one-native-
        resolution design, but fails loudly with both sizes named rather than a generic numpy
        broadcast error if it ever does)."""
        if not self.store.step_store:
            raise ValueError("Attention store is empty. Did you call hook_pipeline() and generate?")
        if not target_indices_position:
            raise ValueError("target_indices_position must be non-empty")
        n_captured = len(self.store.target_token_indices)
        for pos in target_indices_position:
            if pos < 0 or pos >= n_captured:
                raise ValueError(
                    f"target_indices_position entry {pos} is out of range for the {n_captured} "
                    "captured columns -- these must be positions INTO the columns actually "
                    "captured (store.target_token_indices), not raw T5 sequence positions")
        accum = None
        side = None
        count = 0
        for step_idx, layer_maps in self.store.step_store.items():
            if step_idx >= max_steps:
                continue
            for tensor in layer_maps.values():
                arr = tensor.numpy()[:, target_indices_position].mean(axis=-1)  # (img_seq,)
                this_side = int(round(math.sqrt(arr.shape[0])))
                if this_side * this_side != arr.shape[0]:
                    raise ValueError(f"image token count {arr.shape[0]} is not a perfect square")
                if accum is None:
                    side = this_side
                    accum = np.zeros((side, side), dtype=np.float32)
                elif this_side != side:
                    raise ValueError(
                        "inconsistent native image grid size across captured layers/steps: "
                        f"expected {side}x{side}, got {this_side}x{this_side}")
                accum += arr.reshape(side, side)
                count += 1
        if count == 0:
            return np.zeros(target_resolution, dtype=np.float32)
        avg = torch.from_numpy(accum / count).unsqueeze(0).unsqueeze(0)
        up = F.interpolate(avg, size=target_resolution, mode="bilinear", align_corners=False)
        return up.squeeze(0).squeeze(0).numpy()


def flux_token_indices(tokenizer_2, prompt: str, attribute: str) -> List[int]:
    """T5 token indices for `attribute` within `prompt` -- analogous to
    generate_anchor_images_sdxl.py's token_indices(), but against tokenizer_2 (T5): T5 tokens,
    not CLIP's, are what actually enter FLUX's joint attention sequence. Falls back through
    locate_attribute_phrase when `attribute` is a strict sub-phrase of the prompt's actual
    wording (e.g. "yellow helmet" vs. "...yellow bike helmet...").

    KNOWN LIMITATION: this is a first-match, content-only token search -- it discards the
    character offset locate_attribute_phrase already computed and instead finds the FIRST
    occurrence of the matched phrase's token sequence in the tokenized prompt. If `attribute`'s
    matched phrase occurs more than once in `prompt` (e.g. two different subjects both "wearing
    a red apron"), every call with that identical (prompt, attribute) pair returns the SAME
    first occurrence -- there is no way from this function alone to select a later occurrence.
    attribute_target_token_indices() guards its own caller against the resulting silent data
    loss by requiring attribute strings to be unique per prompt; a caller invoking this function
    directly, outside that guard, is NOT protected and must ensure uniqueness itself."""
    _, matched_phrase = locate_attribute_phrase(prompt, attribute)
    ids = tokenizer_2(prompt, padding="max_length", max_length=512, truncation=True).input_ids
    target = tokenizer_2(matched_phrase, add_special_tokens=False).input_ids
    if not target:
        raise ValueError(f"phrase {matched_phrase!r} tokenized to nothing")
    for i in range(len(ids) - len(target) + 1):
        if ids[i:i + len(target)] == target:
            return list(range(i, i + len(target)))
    raise ValueError(f"phrase {matched_phrase!r} not found in prompt tokens: {prompt!r}")


def attribute_target_token_indices(
    tokenizer_2, prompt: str, subject_attribute_pairs: Sequence[Tuple[str, str]]
) -> Tuple[Dict[str, List[int]], List[int]]:
    """Per-attribute token indices, plus their sorted-deduplicated union -- the union is what
    FluxAttentionCapture.hook_pipeline() needs BEFORE generation starts, since every attribute
    for this prompt must be captured in the same generation pass (attention cannot be captured
    twice from one run). Returns ({attribute: [indices]}, [union of all indices]).

    Requires every attribute string across `subject_attribute_pairs` to be unique. This is a
    direct consequence of flux_token_indices' own first-match, content-only limitation (see its
    docstring): two subjects sharing identical attribute text (e.g. both "wearing a red apron")
    would otherwise silently collide in `per_attribute` (keyed by attribute string) and both
    resolve to the SAME first occurrence's token indices -- the second subject's real occurrence
    is dropped with no exception anywhere. Raises ValueError up front, before doing any token
    lookups, naming the duplicate attribute text and both subjects involved, rather than let
    that data loss happen silently."""
    seen: Dict[str, str] = {}
    for subject, attribute in subject_attribute_pairs:
        if attribute in seen:
            raise ValueError(
                f"duplicate attribute text {attribute!r} for subjects {seen[attribute]!r} and "
                f"{subject!r} -- attribute_target_token_indices requires unique attribute "
                "strings per prompt (see flux_token_indices' first-match limitation)")
        seen[attribute] = subject

    per_attribute: Dict[str, List[int]] = {}
    union: Set[int] = set()
    for _subject, attribute in subject_attribute_pairs:
        idxs = flux_token_indices(tokenizer_2, prompt, attribute)
        per_attribute[attribute] = idxs
        union.update(idxs)
    return per_attribute, sorted(union)


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

    assign = assign_subjects(models, image, boxes, subjects)
    subject_boxes = {s: boxes[assign[s]] for s in subjects}

    per_attr_indices, union_indices = attribute_target_token_indices(
        models.txt2img.tokenizer_2, prompt, spec["pairs"])
    capture = FluxAttentionCapture()

    # Deliberately a SECOND generation pass (not a single hook-then-detect-from-same-
    # generation pass, as an earlier design sketch considered) -- reuses the same seed so
    # the image is unaffected, but keeps the existing, already-working seed-retry detection
    # loop above completely untouched rather than risk modifying it. Doubles per-prompt
    # generation cost; already flagged for Pranav before the full run.
    #
    # hook_pipeline() through unhook_pipeline() is wrapped in try/finally: if generation or
    # scoring raises partway through (OOM, transient CUDA error -- the second pass is the
    # more memory-intensive one), unhook_pipeline() must still run so the pipeline's
    # processors don't stay wired to this (abandoned) prompt's FluxAttentionStore, which
    # would otherwise force the NEXT prompt's detection pass through the slow, unfused
    # custom processor instead of the stock fused kernel.
    try:
        capture.hook_pipeline(models.txt2img, target_token_indices=union_indices)

        def _capture_step_cb(pipe, i, t, kw):
            capture.store.step()
            return kw

        g = torch.Generator(DEVICE).manual_seed(chosen_seed)
        _ = models.txt2img(prompt, num_inference_steps=NUM_INFERENCE_STEPS,
                           height=IMG_SIZE, width=IMG_SIZE, generator=g,
                           callback_on_step_end=_capture_step_cb).images[0]

        attributes_out: List[dict] = []
        for subject, attribute in spec["pairs"]:
            indices_in_union = [union_indices.index(i) for i in per_attr_indices[attribute]]
            early_map = capture.cross_attention_map(
                target_indices_position=indices_in_union,
                target_resolution=(IMG_SIZE, IMG_SIZE), max_steps=MAX_STEPS)
            full_map = capture.cross_attention_map(
                target_indices_position=indices_in_union,
                target_resolution=(IMG_SIZE, IMG_SIZE), max_steps=NUM_INFERENCE_STEPS)
            owner, scores = predicted_owner_from_attention(early_map, subject_boxes)
            owner_full, scores_full = predicted_owner_from_attention(full_map, subject_boxes)
            attributes_out.append(build_attribute_entry(attribute, subject, owner, scores,
                                                         owner_full, scores_full))
    finally:
        capture.unhook_pipeline(models.txt2img)
        # Free GPU memory after the (more memory-intensive, attention-capturing) second
        # pass -- matches the empty_cache() call after the first pass above.
        torch.cuda.empty_cache()

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
                "early_window_fraction": EARLY_WINDOW_FRACTION, "images": []}
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
