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
from typing import Dict, List, Sequence, Tuple

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
