#!/usr/bin/env python
"""
FLUX.1-dev cross-attention capture hook. FLUX's stock FluxAttnProcessor never materializes an
attention probability matrix -- it dispatches straight to a fused SDPA kernel (see diffusers'
transformer_flux.py, dispatch_attention_fn) -- so there is nothing to intercept the way
generate_anchor_images_sdxl.py's CustomAttnProcessor intercepts attn.get_attention_scores().
FluxCustomAttnProcessor below recomputes the same math by hand (projections, RMSNorm, RoPE,
explicit softmax) to get an explicit matrix, captures it, then continues exactly as the stock
processor would so generation is unaffected.

Verified 2026-07-31 against a real (tiny) FluxTransformer2DModel with REAL (non-zero) RoPE
position ids active: hooking only the double blocks (`transformer_blocks.*`, NOT
`single_transformer_blocks.*` -- note the substring trap, `.startswith("transformer_blocks.")`
is required, a plain "in" check would false-match the single blocks too, since
"single_transformer_blocks.0..." contains "transformer_blocks." as a substring) reproduces the
stock model's output to ~2e-7 max abs diff -- floating-point-summation-order noise between manual
softmax and a fused kernel, not a bug. Only the 19 double blocks are hooked; see scope decision
in docs/superpowers/specs/2026-07-31-flux-attention-hook-design.md.

Memory note (this makes the design doc's step-5 slicing concrete): storing the FULL attention
matrix per layer per step is infeasible at real FLUX scale (24 heads x 4096 image tokens x ~512
text tokens x 4 bytes = ~192MB PER LAYER PER STEP; x19 layers x25 steps = ~89GB). Instead,
FluxAttentionStore is told the exact set of target T5 token column indices to keep BEFORE
generation starts (every attribute's tokens are known from the prompt spec up front -- see
attribute_target_token_indices()), and FluxCustomAttnProcessor slices to (image_rows x those
columns) and averages over heads before ever storing anything.

Also resolves the design doc's two "open items": the text/image split point and the image grid
side length are read directly from tensor shapes at call time (encoder_hidden_states.shape[1],
and sqrt(image row count) respectively) rather than hardcoded constants that could drift from
the real generation config.

There is exactly ONE native image resolution across every hooked layer (unlike a UNet, FLUX does
not downsample spatially block-to-block), so aggregation across layers/steps needs no
per-resolution weighting -- a plain mean, unlike generate_anchor_images_sdxl.py's
resolution-squared-weighted composite.
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F


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
        self.step_store.setdefault(self.current_step, {})[layer_name] = head_averaged.detach().cpu()

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
    from anchor_common import locate_attribute_phrase

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
