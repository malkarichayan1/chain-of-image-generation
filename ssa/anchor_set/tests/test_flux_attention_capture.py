"""Tests for flux_attention_capture.py: does hooking FLUX's double transformer blocks capture
real attention without perturbing generation? Requires torch/diffusers -- skipped (not failed)
if unavailable, matching pilot/spatial_semantic_alignment.py's Scenario 10 convention. Run from
inside ssa/anchor_set/:  py -3 -m pytest tests/"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")

import flux_attention_capture as fac


def _tiny_transformer(num_layers=2, num_single_layers=1, seed=0):
    from diffusers import FluxTransformer2DModel
    torch.manual_seed(seed)
    return FluxTransformer2DModel(
        patch_size=1, in_channels=16, out_channels=16,
        num_layers=num_layers, num_single_layers=num_single_layers,
        attention_head_dim=8, num_attention_heads=2,
        joint_attention_dim=32, pooled_projection_dim=24,
        guidance_embeds=False, axes_dims_rope=(2, 2, 4),
    ).eval()


def _forward_inputs(batch=1, img_seq=9, txt_seq=5):
    hidden_states = torch.randn(batch, img_seq, 16)
    encoder_hidden_states = torch.randn(batch, txt_seq, 32)
    pooled_projections = torch.randn(batch, 24)
    timestep = torch.tensor([500.0])
    img_ids = torch.zeros(img_seq, 3)
    side = int(math.sqrt(img_seq))
    for i in range(img_seq):
        img_ids[i, 1] = i // side
        img_ids[i, 2] = i % side
    txt_ids = torch.zeros(txt_seq, 3)
    return dict(hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states,
                pooled_projections=pooled_projections, timestep=timestep,
                img_ids=img_ids, txt_ids=txt_ids, return_dict=False)


# --------------------------------------------------------------------------- store

def test_store_reset_requires_nonempty_target_indices():
    store = fac.FluxAttentionStore()
    with pytest.raises(ValueError, match="non-empty"):
        store.reset([])


def test_store_add_attention_keys_by_current_step():
    store = fac.FluxAttentionStore()
    store.reset([0, 1])
    store.add_attention("layer_a", torch.zeros(9, 2))
    store.step()
    store.add_attention("layer_a", torch.ones(9, 2))
    assert store.step_store[0]["layer_a"].sum().item() == 0
    assert store.step_store[1]["layer_a"].sum().item() == 18


# --------------------------------------------------------------------------- processor equivalence

def test_hooked_output_matches_stock_within_float_tolerance():
    m_stock = _tiny_transformer(seed=0)
    m_hooked = _tiny_transformer(seed=0)  # identical seed -> identical weights
    inputs = _forward_inputs()
    with torch.no_grad():
        stock_out = m_stock(**inputs)[0]

    store = fac.FluxAttentionStore()
    store.reset([0, 1, 2])
    processors = dict(m_hooked.attn_processors)
    for name in processors:
        if name.startswith("transformer_blocks."):
            processors[name] = fac.FluxCustomAttnProcessor(store, name)
    m_hooked.set_attn_processor(processors)

    with torch.no_grad():
        hooked_out = m_hooked(**inputs)[0]
    assert torch.allclose(stock_out, hooked_out, atol=1e-5, rtol=1e-4)


def test_processor_captures_only_requested_target_columns():
    m = _tiny_transformer(num_layers=1, num_single_layers=0)
    store = fac.FluxAttentionStore()
    store.reset([1, 3])  # 2 of the 5 text tokens
    processors = dict(m.attn_processors)
    for name in processors:
        if name.startswith("transformer_blocks."):
            processors[name] = fac.FluxCustomAttnProcessor(store, name)
    m.set_attn_processor(processors)
    with torch.no_grad():
        m(**_forward_inputs())
    assert store.step_store  # something was captured
    for layer_maps in store.step_store.values():
        for tensor in layer_maps.values():
            assert tensor.shape == (9, 2)  # (img_seq, n_targets) -- not all 5 text columns


def test_processor_image_rows_sum_to_one_across_full_key_dimension():
    """Sanity on slice orientation: softmax rows must sum to ~1 across the FULL key dimension
    (text+image), not just the sliced target columns. Verified independently by re-deriving
    attn_probs inside this test rather than trusting the processor's own internal slice.

    Uses a fresh encoder_hidden_states at inner_dim (not the raw joint_attention_dim from
    _forward_inputs()) -- FluxTransformer2DModel's context_embedder projects encoder_hidden_states
    from joint_attention_dim down to inner_dim BEFORE any block's attn module sees it, so
    reconstructing what a block's attention actually receives means using that already-projected
    dimension, not the model's raw input dimension."""
    from diffusers.models.embeddings import apply_rotary_emb
    from diffusers.models.transformers.transformer_flux import _get_qkv_projections

    m = _tiny_transformer(num_layers=1, num_single_layers=0)
    inputs = _forward_inputs()
    block = m.transformer_blocks[0]
    attn = block.attn

    inner_dim = attn.heads * (attn.add_q_proj.out_features // attn.heads)
    encoder_hidden_states_projected = torch.randn(1, inputs["encoder_hidden_states"].shape[1], inner_dim)

    with torch.no_grad():
        query, key, value, eq, ek, ev = _get_qkv_projections(
            attn, inputs["hidden_states"], encoder_hidden_states_projected)
        query = attn.norm_q(query.unflatten(-1, (attn.heads, -1)))
        key = attn.norm_k(key.unflatten(-1, (attn.heads, -1)))
        eq = attn.norm_added_q(eq.unflatten(-1, (attn.heads, -1)))
        ek = attn.norm_added_k(ek.unflatten(-1, (attn.heads, -1)))
        query = torch.cat([eq, query], dim=1)
        key = torch.cat([ek, key], dim=1)
        q = query.permute(0, 2, 1, 3)
        k = key.permute(0, 2, 1, 3)
        probs = torch.softmax((q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1]), dim=-1)
        text_len = encoder_hidden_states_projected.shape[1]
        image_rows = probs[:, :, text_len:, :]
        row_sums = image_rows.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)
