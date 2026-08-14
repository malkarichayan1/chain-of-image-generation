#!/usr/bin/env python
"""
Per-head / per-layer / per-timestep attention capture for FLUX.1-dev -- the single GPU run
that feeds taxonomy experiments #14, #16, #17, #18 and the critical follow-up #19.

WHY THIS IS A GPU RUN AND NOT RE-ANALYSIS
    The experiment table optimistically marks #14 "Re-analysis if per-layer maps saved".
    They were not saved. flux_attention_capture.py's FluxCustomAttnProcessor reduces with
    `target_cols.mean(dim=1)` -- heads are averaged away before anything is stored -- and
    FluxAttentionCapture.cross_attention_map then averages over every layer and step. What
    reached manifest.json is one pooled scalar per (attribute, subject): `model_scores`.
    Head, layer, and step identity are all destroyed upstream of disk, so the taxonomy
    cannot be recovered without re-running generation. This file is that re-run.

WHAT MAKES IT AFFORDABLE
    Storing per-head maps is infeasible (24 heads x 4096 image tokens x ~20 target columns
    x 19 layers x 25 steps ~= 3.7 GB per image). The trick is that the taxonomy never needs
    the maps themselves -- every metric in #14/#16/#17/#18 is a REDUCTION of a map against
    the subject boxes, and the boxes are already known before generation starts (boxes.json,
    from recompute_boxes.py). So this file pushes the reduction inside the attention
    processor: each (layer, step, head, attribute) cell is collapsed on-GPU to
    (in-box mean mass per subject, spatial entropy) and only those scalars are kept.
    That is ~0.5 MB per image in float16 instead of 3.7 GB, and it is what makes one run
    answer four experiments at once.

    Consequence worth stating plainly: because the reduction happens during the forward
    pass, the boxes must be FROZEN AND PRE-LOADED. This file therefore reads boxes.json
    rather than re-detecting, which is also what keeps every cell scored against exactly
    the same boxes as the already-published pooled numbers.

THE REPRODUCTION PROBLEM, AND THE TWO CHECKS THAT GUARD IT
    The human labels and the boxes belong to the ORIGINAL images. This run regenerates
    from the same pinned seeds, and diffusion output is not guaranteed bit-reproducible
    across sessions/hardware (the same caveat vqa_score_flux.py and recompute_boxes.py
    already carry). If a regenerated image drifts, its attention describes a DIFFERENT
    image than the one a human labeled, and every cell computed from it is invalid --
    silently, with no error anywhere. Two independent checks are recorded per image:

      1. `repro_mean_abs_pixel_diff` -- mean |new - stored| over RGB in [0,1]. Exact
         reproduction gives 0.0.
      2. `pooled_owner_matches_manifest` -- re-derive the pooled early-window
         predicted_owner from THIS run's captured attention and compare it to the
         predicted_owner already in manifest.json.

    Neither check gates the capture (a drifted image is still written, flagged), because
    which threshold makes a row unusable is an analysis decision, not a capture decision.
    exp9_taxonomy_analysis.py applies the threshold and reports how many rows it dropped.
    Do not report a taxonomy number without also reporting that count.

    GRID-SPACE APPROXIMATION: the published pipeline upsamples the 64x64 attention map to
    1024x1024 and averages inside the pixel box. Doing that per cell would mean 11,400
    bilinear upsamples per image, so cells are instead scored by averaging in native 64x64
    grid space with the box scaled down. These are close but not identical. The pooled
    check above is computed BOTH ways, so `pooled_owner_matches_manifest` (exact path) and
    `pooled_owner_grid_matches_manifest` (grid path) together quantify what the
    approximation costs. If they diverge materially, the grid approximation is not safe and
    the analysis must say so.

CFG is off (FLUX.1-dev, no true_cfg_scale) -- see FluxAttentionStore's UNHANDLED CASE note;
batch size is 1, as that store requires.

Output (one npz per image, so a Kaggle timeout loses at most one image):
    taxonomy_cells_p<prompt_id>.npz
        in_box_mass float16 (layers, steps, heads, attributes, subjects)
        entropy     float16 (layers, steps, heads, attributes)
    taxonomy_index.json -- per image: attribute/subject/layer ORDER (the axis labels for
        the arrays above -- without this the npz axes are meaningless), plus both
        reproduction checks. Rewritten after every image.
"""
import json
import math
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

if "KAGGLE_KERNEL_RUN_TYPE" in os.environ:
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--progress-bar", "off",
        "torch==2.5.1", "--index-url", "https://download.pytorch.org/whl/cu121",
    ])
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--progress-bar", "off",
        "diffusers>=0.32", "transformers>=4.44", "accelerate", "sentencepiece", "protobuf",
    ])

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
MODEL_ID = "black-forest-labs/FLUX.1-dev"
IMG_SIZE = 1024
NUM_INFERENCE_STEPS = 25
EARLY_WINDOW_FRACTION = 0.5
MAX_STEPS = int(NUM_INFERENCE_STEPS * EARLY_WINDOW_FRACTION)

OUT_DIR = Path(".")
INDEX_PATH = OUT_DIR / "taxonomy_index.json"


# ---------------------------------------------------------------------------
# Pure helpers (no model). Mirrored by tests/test_taxonomy_capture_flux.py.
# ---------------------------------------------------------------------------

def box_to_grid_mask(box: Sequence[float], side: int, img_size: int) -> np.ndarray:
    """Boolean (side*side,) mask for an [x0,y0,x1,y1] PIXEL box, scaled into the attention
    map's native `side` x `side` grid.

    Always marks at least one cell: a box thinner than one grid cell would otherwise
    produce an all-False mask, and a masked mean over zero cells is a NaN that would
    propagate silently through every downstream aggregate. Degenerate boxes (x1<=x0)
    are clamped to a single cell rather than raising -- Mask R-CNN can emit a sliver box,
    and losing one subject is better than losing the whole image."""
    scale = side / float(img_size)
    x0, y0, x1, y1 = (v * scale for v in box)
    c0 = min(max(int(math.floor(x0)), 0), side - 1)
    r0 = min(max(int(math.floor(y0)), 0), side - 1)
    c1 = min(max(int(math.ceil(x1)), c0 + 1), side)
    r1 = min(max(int(math.ceil(y1)), r0 + 1), side)
    mask = np.zeros((side, side), dtype=bool)
    mask[r0:r1, c0:c1] = True
    return mask.reshape(-1)


def spatial_entropy(maps: torch.Tensor) -> torch.Tensor:
    """Shannon entropy (nats) over the spatial axis of `maps` (..., n_positions).

    Attention rows are already a softmax over the FULL key dimension, so the slice
    belonging to one attribute does not sum to 1; it is renormalized here so entropy
    measures how PEAKED the attribute's spatial map is, independent of how much total mass
    the attribute drew. A degenerate all-zero map yields 0.0 rather than NaN."""
    total = maps.sum(dim=-1, keepdim=True)
    p = torch.where(total > 0, maps / total.clamp_min(torch.finfo(maps.dtype).tiny),
                    torch.zeros_like(maps))
    return -(p * torch.log(p.clamp_min(1e-12))).sum(dim=-1)


def pooled_owner_from_cells(in_box_mass: np.ndarray, subjects: Sequence[str],
                            max_steps: int) -> List[str]:
    """Re-derive the pooled early-window predicted_owner per attribute from captured cells,
    by averaging over layers, the first `max_steps` steps, and heads -- the same reduction
    FluxAttentionCapture.cross_attention_map performs, but done after the fact on the cell
    grid. `in_box_mass` is (layers, steps, heads, attributes, subjects)."""
    pooled = in_box_mass[:, :max_steps, :, :, :].mean(axis=(0, 1, 2))  # (attributes, subjects)
    return [subjects[int(np.argmax(row))] for row in pooled]


def mean_abs_pixel_diff(a: Image.Image, b: Image.Image) -> float:
    """Mean absolute RGB difference in [0,1] between two same-size images. 0.0 is an exact
    reproduction."""
    ai = np.asarray(a.convert("RGB"), dtype=np.float32) / 255.0
    bi = np.asarray(b.convert("RGB"), dtype=np.float32) / 255.0
    if ai.shape != bi.shape:
        raise ValueError(f"image size mismatch: {ai.shape} vs {bi.shape}")
    return float(np.abs(ai - bi).mean())


def mean_mass_in_box(attn_map: np.ndarray, box: Sequence[float]) -> float:
    """Mean attention value inside an [x0,y0,x1,y1] pixel box of an ALREADY-UPSAMPLED map.
    Verbatim from anchor_common.mean_mass_in_box -- the exact published path, kept here so
    the reproduction check compares against the real thing rather than an approximation of
    it. Keep in sync."""
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    x0, y0 = max(x0, 0), max(y0, 0)
    x1 = min(x1, attn_map.shape[1])
    y1 = min(y1, attn_map.shape[0])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float(attn_map[y0:y1, x0:x1].mean())


# ---------------------------------------------------------------------------
# Capture. Structurally FluxCustomAttnProcessor (flux_attention_capture.py) with the
# head-averaging replaced by an on-GPU reduction against pre-loaded boxes. Keep in sync.
# ---------------------------------------------------------------------------

class TaxonomyStore:
    """Accumulates per-(layer, step, head, attribute) reductions.

    `attribute_columns` are positions INTO the captured target columns (not raw T5
    indices), one list per attribute -- the same distinction FluxAttentionCapture's
    `target_indices_position` draws. `subject_masks` is (n_subjects, img_seq) bool on
    DEVICE, built from frozen boxes before generation starts."""

    def __init__(self):
        self.reset([], None, [], [])

    def reset(self, target_token_indices: Sequence[int], subject_masks,
              attribute_columns: Sequence[Sequence[int]], layer_order: Sequence[str]) -> None:
        self.target_token_indices = list(target_token_indices)
        self.subject_masks = subject_masks
        self.attribute_columns = [list(c) for c in attribute_columns]
        self.layer_order = list(layer_order)
        self.current_step = 0
        # {(step, layer): (in_box (heads, attrs, subjects), entropy (heads, attrs))}
        self.cells: Dict[Tuple[int, str], Tuple[np.ndarray, np.ndarray]] = {}
        self.pooled_accum = None   # running sum of head-averaged maps, early window only
        self.pooled_count = 0
        self.side = None

    def add(self, layer_name: str, image_rows: torch.Tensor) -> None:
        """`image_rows`: (heads, img_seq, n_captured_columns) for one layer at this step."""
        heads, img_seq, _ = image_rows.shape
        side = int(round(math.sqrt(img_seq)))
        if side * side != img_seq:
            raise ValueError(f"image token count {img_seq} is not a perfect square")
        if self.side is None:
            self.side = side
        elif side != self.side:
            raise ValueError(
                f"inconsistent native grid: expected {self.side}, got {side} at {layer_name}")

        # (heads, img_seq, n_attributes): each attribute's own columns, averaged.
        per_attr = torch.stack(
            [image_rows[:, :, cols].mean(dim=-1) for cols in self.attribute_columns], dim=-1)
        maps = per_attr.permute(0, 2, 1).float()          # (heads, attrs, img_seq)

        masks = self.subject_masks.to(maps.dtype)          # (subjects, img_seq)
        counts = masks.sum(dim=-1).clamp_min(1.0)          # (subjects,)
        in_box = (maps @ masks.T) / counts                 # (heads, attrs, subjects)
        ent = spatial_entropy(maps)                        # (heads, attrs)

        self.cells[(self.current_step, layer_name)] = (
            in_box.detach().to(torch.float16).cpu().numpy(),
            ent.detach().to(torch.float16).cpu().numpy(),
        )

        # Pooled early-window map, head-averaged -- the exact published reduction, kept in
        # full spatial resolution for the reproduction check.
        if self.current_step < MAX_STEPS:
            pooled = maps.mean(dim=0).detach().cpu().numpy()   # (attrs, img_seq)
            self.pooled_accum = pooled if self.pooled_accum is None else self.pooled_accum + pooled
            self.pooled_count += 1

    def step(self) -> None:
        self.current_step += 1

    def stack(self, n_attributes: int, n_subjects: int, n_heads: int
              ) -> Tuple[np.ndarray, np.ndarray]:
        """Dense (layers, steps, heads, attrs, subjects) + (layers, steps, heads, attrs)."""
        n_layers, n_steps = len(self.layer_order), NUM_INFERENCE_STEPS
        in_box = np.zeros((n_layers, n_steps, n_heads, n_attributes, n_subjects), np.float16)
        ent = np.zeros((n_layers, n_steps, n_heads, n_attributes), np.float16)
        for (step_idx, layer_name), (cell_in_box, cell_ent) in self.cells.items():
            if step_idx >= n_steps:
                continue
            li = self.layer_order.index(layer_name)
            in_box[li, step_idx] = cell_in_box
            ent[li, step_idx] = cell_ent
        return in_box, ent


class TaxonomyAttnProcessor:
    """FluxCustomAttnProcessor with TaxonomyStore.add in place of the head-averaging store.
    The math up to `attn_probs` is verbatim; see flux_attention_capture.py for why the
    manual softmax recompute is necessary at all (claim C7)."""

    def __init__(self, store: TaxonomyStore, layer_name: str):
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

        q = query.permute(0, 2, 1, 3)
        k = key.permute(0, 2, 1, 3)
        v = value.permute(0, 2, 1, 3)
        scale = 1.0 / math.sqrt(q.shape[-1])
        attn_probs = torch.softmax((q @ k.transpose(-1, -2)) * scale, dim=-1)

        if encoder_hidden_states is not None:
            text_len = encoder_hidden_states.shape[1]
            assert query.shape[0] == 1, "TaxonomyAttnProcessor assumes batch size 1"
            image_rows = attn_probs[:, :, text_len:, :]
            target_cols = image_rows[..., self.store.target_token_indices][0]  # (heads,img,cols)
            self.store.add(self.layer_name, target_cols)

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


def double_block_names(pipeline) -> List[str]:
    """Sorted double-block processor names. `.startswith("transformer_blocks.")` is required,
    NOT `in` -- "single_transformer_blocks.0..." contains "transformer_blocks." as a
    substring and a plain membership test silently hooks the single blocks too (the
    substring trap documented in flux_attention_capture.py)."""
    names = [n for n in pipeline.transformer.attn_processors
             if n.startswith("transformer_blocks.")]
    return sorted(names, key=lambda n: int(n.split(".")[1]))


def hook(pipeline, store: TaxonomyStore) -> None:
    processors = dict(pipeline.transformer.attn_processors)
    for name in store.layer_order:
        processors[name] = TaxonomyAttnProcessor(store, name)
    pipeline.transformer.set_attn_processor(processors)


def unhook(pipeline) -> None:
    from diffusers.models.transformers.transformer_flux import FluxAttnProcessor
    pipeline.transformer.set_attn_processor(FluxAttnProcessor())


# ---------------------------------------------------------------------------
# Token indexing -- verbatim from flux_attention_capture.py. Keep in sync.
# ---------------------------------------------------------------------------

def locate_attribute_phrase(prompt: str, attribute: str) -> Tuple[int, str]:
    lowered = prompt.lower()
    idx = lowered.find(attribute.lower())
    if idx >= 0:
        return idx, attribute
    words = attribute.split()
    for drop in range(1, len(words)):
        candidate = " ".join(words[drop:])
        idx = lowered.find(candidate.lower())
        if idx >= 0:
            return idx, candidate
    raise ValueError(f"attribute {attribute!r} not found in prompt {prompt!r}")


def flux_token_indices(tokenizer_2, prompt: str, attribute: str) -> List[int]:
    _, matched_phrase = locate_attribute_phrase(prompt, attribute)
    ids = tokenizer_2(prompt, padding="max_length", max_length=512, truncation=True).input_ids
    target = tokenizer_2(matched_phrase, add_special_tokens=False).input_ids
    if not target:
        raise ValueError(f"phrase {matched_phrase!r} tokenized to nothing")
    for i in range(len(ids) - len(target) + 1):
        if ids[i:i + len(target)] == target:
            return list(range(i, i + len(target)))
    raise ValueError(f"phrase {matched_phrase!r} not found in prompt tokens: {prompt!r}")


def attribute_target_token_indices(tokenizer_2, prompt: str,
                                   pairs: Sequence[Tuple[str, str]]
                                   ) -> Tuple[Dict[str, List[int]], List[int]]:
    seen: Dict[str, str] = {}
    for subject, attribute in pairs:
        if attribute in seen:
            raise ValueError(
                f"duplicate attribute text {attribute!r} for subjects {seen[attribute]!r} "
                f"and {subject!r}")
        seen[attribute] = subject
    per_attribute: Dict[str, List[int]] = {}
    union = set()
    for _subject, attribute in pairs:
        idxs = flux_token_indices(tokenizer_2, prompt, attribute)
        per_attribute[attribute] = idxs
        union.update(idxs)
    return per_attribute, sorted(union)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def find_input_dir() -> Path:
    """Locate the attached dataset by CONTENT (first dir holding manifest.json), searched
    recursively -- Kaggle's mount depth varies by how the dataset was created, so a
    hardcoded /kaggle/input/<slug>/ path is fragile (same approach as vqa_score_flux.py)."""
    roots = [Path("/kaggle/input"), Path(".")]
    for root in roots:
        if not root.exists():
            continue
        if (root / "manifest.json").exists():
            return root
        for candidate in sorted(root.rglob("manifest.json")):
            return candidate.parent
    raise FileNotFoundError("no manifest.json found under /kaggle/input or .")


def load_pipeline():
    from diffusers import FluxPipeline
    pipe = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=DTYPE)
    pipe = pipe.to(DEVICE)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def capture_one(pipe, img: dict, subject_boxes: Dict[str, list], input_dir: Path) -> dict:
    prompt_id, prompt = img["prompt_id"], img["prompt"]
    pairs = [(a["intended_subject"], a["attribute"]) for a in img["attributes"]]
    attributes = [a["attribute"] for a in img["attributes"]]
    subjects = sorted(subject_boxes)

    per_attr_indices, union_indices = attribute_target_token_indices(
        pipe.tokenizer_2, prompt, pairs)
    attribute_columns = [[union_indices.index(i) for i in per_attr_indices[a]]
                         for a in attributes]

    side = IMG_SIZE // 16   # FLUX patchifies the 128x128 latent 2x2 -> 64x64 tokens
    masks = np.stack([box_to_grid_mask(subject_boxes[s], side, IMG_SIZE) for s in subjects])
    subject_masks = torch.from_numpy(masks).to(DEVICE)

    store = TaxonomyStore()
    store.reset(union_indices, subject_masks, attribute_columns, double_block_names(pipe))

    try:
        hook(pipe, store)

        def _step_cb(pipe_, i, t, kw):
            store.step()
            return kw

        g = torch.Generator(DEVICE).manual_seed(img["seed"])
        out = pipe(prompt, num_inference_steps=NUM_INFERENCE_STEPS,
                   height=IMG_SIZE, width=IMG_SIZE, generator=g,
                   callback_on_step_end=_step_cb).images[0]
    finally:
        unhook(pipe)
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    n_heads = pipe.transformer.config.num_attention_heads
    in_box, ent = store.stack(len(attributes), len(subjects), n_heads)
    np.savez_compressed(OUT_DIR / f"taxonomy_cells_p{prompt_id}.npz",
                        in_box_mass=in_box, entropy=ent)

    # --- Check 1: pixel-level reproduction against the stored, human-labeled image.
    stored_path = input_dir / "images" / f"p{prompt_id}.png"
    repro = (mean_abs_pixel_diff(out, Image.open(stored_path))
             if stored_path.exists() else None)

    # --- Check 2: does this run's pooled attention reproduce manifest's predicted_owner?
    # Exact published path: upsample the head-averaged early-window map to 1024 and average
    # inside the pixel box.
    manifest_owners = [a["predicted_owner"] for a in img["attributes"]]
    exact_owners: List[str] = []
    if store.pooled_count:
        pooled = store.pooled_accum / store.pooled_count       # (attrs, img_seq)
        for ai in range(len(attributes)):
            grid = torch.from_numpy(pooled[ai].reshape(1, 1, side, side).astype(np.float32))
            up = F.interpolate(grid, size=(IMG_SIZE, IMG_SIZE), mode="bilinear",
                               align_corners=False).squeeze().numpy()
            scores = {s: mean_mass_in_box(up, subject_boxes[s]) for s in subjects}
            exact_owners.append(max(scores, key=lambda s: scores[s]))
    grid_owners = pooled_owner_from_cells(in_box.astype(np.float32), subjects, MAX_STEPS)

    return dict(
        prompt_id=prompt_id,
        attributes=attributes,          # axis 3 labels
        subjects=subjects,              # axis 4 labels
        layer_order=store.layer_order,  # axis 0 labels
        n_heads=int(n_heads),
        n_steps=NUM_INFERENCE_STEPS,
        max_steps_early=MAX_STEPS,
        seed=img["seed"],
        repro_mean_abs_pixel_diff=repro,
        manifest_owners=manifest_owners,
        exact_pooled_owners=exact_owners,
        grid_pooled_owners=grid_owners,
        pooled_owner_matches_manifest=(exact_owners == manifest_owners),
        pooled_owner_grid_matches_manifest=(grid_owners == manifest_owners),
    )


def main() -> None:
    input_dir = find_input_dir()
    manifest = json.loads((input_dir / "manifest.json").read_text())
    boxes = json.loads((input_dir / "boxes.json").read_text())
    print(f"device={DEVICE} dtype={DTYPE} input_dir={input_dir}")

    index: Dict[str, dict] = {}
    if INDEX_PATH.exists():   # resume after a Kaggle timeout
        index = json.loads(INDEX_PATH.read_text())
        print(f"resuming: {len(index)} image(s) already captured")

    pipe = load_pipeline()
    todo = [img for img in manifest["images"]
            if img.get("detected") and str(img["prompt_id"]) in boxes
            and str(img["prompt_id"]) not in index]
    print(f"{len(todo)} image(s) to capture")

    t0 = time.time()
    for i, img in enumerate(todo):
        elapsed = time.time() - t0
        eta = (elapsed / max(i, 1)) * (len(todo) - i) / 60.0
        print(f"\n=== p{img['prompt_id']} [{i + 1}/{len(todo)}] ~{eta:.0f}min left ===")
        try:
            entry = capture_one(pipe, img, boxes[str(img["prompt_id"])], input_dir)
        except Exception as e:
            print(f"  p{img['prompt_id']}: ERROR {type(e).__name__}: {e}")
            traceback.print_exc()
            continue
        index[str(img["prompt_id"])] = entry
        INDEX_PATH.write_text(json.dumps(index, indent=2, sort_keys=True))
        print(f"  repro_diff={entry['repro_mean_abs_pixel_diff']} "
              f"exact_owner_match={entry['pooled_owner_matches_manifest']} "
              f"grid_owner_match={entry['pooled_owner_grid_matches_manifest']}")

    matched = sum(1 for e in index.values() if e["pooled_owner_matches_manifest"])
    print(f"\nDone. {len(index)} captured; pooled owner reproduced on {matched}/{len(index)}.")
    print("Run exp9_taxonomy_analysis.py on taxonomy_index.json + the npz files.")


if __name__ == "__main__":
    main()
