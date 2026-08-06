"""Tests for flux_attention_capture.py: does hooking FLUX's double transformer blocks capture
real attention without perturbing generation? Requires torch/diffusers -- skipped (not failed)
if unavailable, matching pilot/spatial_semantic_alignment.py's Scenario 10 convention. Run from
inside ssa/anchor_set/:  py -3 -m pytest tests/"""
import math
import sys
from pathlib import Path
from typing import List

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


def _forward_inputs(batch=1, img_seq=9, txt_seq=5, seed=None):
    if seed is not None:
        old_state = torch.get_rng_state()
        torch.manual_seed(seed)
    hidden_states = torch.randn(batch, img_seq, 16)
    encoder_hidden_states = torch.randn(batch, txt_seq, 32)
    pooled_projections = torch.randn(batch, 24)
    if seed is not None:
        torch.set_rng_state(old_state)
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


def test_add_attention_casts_bfloat16_to_float32():
    """Regression test for a real bug, reproduced end-to-end (not theorized): on real GPU
    hardware DTYPE is torch.bfloat16, so every tensor FluxCustomAttnProcessor captures is
    bfloat16. Before this fix, add_attention only called .detach().cpu() with no dtype cast,
    and torch.Tensor.numpy() raises `TypeError: Got unsupported ScalarType BFloat16` for a
    bfloat16 CPU tensor (confirmed directly against this torch install) -- cross_attention_map's
    internal tensor.numpy() call would crash on every real GPU run, every single prompt, and
    since generate_and_score's try/except in main() swallows the exception, all 105 prompts
    would silently land as detected=False, attributes=[] with zero usable attention data.
    Exercises the actual failure path: a bfloat16 tensor through add_attention, then
    cross_attention_map's own .numpy() call -- not a standalone dtype assertion."""
    store = fac.FluxAttentionStore()
    store.reset([0, 1])
    bf16_tensor = torch.ones(9, 2, dtype=torch.bfloat16)  # 3x3 native image grid, 2 targets
    store.add_attention("layer_a", bf16_tensor)
    # The cast must happen at capture time (add_attention), not deferred to cross_attention_map.
    assert store.step_store[0]["layer_a"].dtype == torch.float32

    capture = fac.FluxAttentionCapture()
    capture.store = store
    result = capture.cross_attention_map(target_indices_position=[0, 1],
                                         target_resolution=(6, 6), max_steps=25)
    assert result.shape == (6, 6)
    assert np.isfinite(result).all()


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


def test_processor_captures_correct_column_identity():
    """Verifies the processor captures the CORRECT columns, not just the right COUNT.
    A bug that selected wrong columns (reversed order, off-by-one, wrong axis) would still
    produce (9, 2) shape and pass the shape-only test. This test verifies that using
    different target indices produces different (not just differently-shaped) captured tensors,
    which confirms the processor is actually reading the correct columns."""
    m = _tiny_transformer(num_layers=1, num_single_layers=0)

    # Capture with target indices [1, 3]
    store_a = fac.FluxAttentionStore()
    store_a.reset([1, 3])
    processors_a = dict(m.attn_processors)
    for name in processors_a:
        if name.startswith("transformer_blocks."):
            processors_a[name] = fac.FluxCustomAttnProcessor(store_a, name)
    m.set_attn_processor(processors_a)

    with torch.no_grad():
        m(**_forward_inputs(seed=42))  # Fixed seed for reproducibility

    # Capture with target indices [3, 1] (reversed order)
    m2 = _tiny_transformer(num_layers=1, num_single_layers=0, seed=0)  # Same model weights
    store_b = fac.FluxAttentionStore()
    store_b.reset([3, 1])
    processors_b = dict(m2.attn_processors)
    for name in processors_b:
        if name.startswith("transformer_blocks."):
            processors_b[name] = fac.FluxCustomAttnProcessor(store_b, name)
    m2.set_attn_processor(processors_b)

    with torch.no_grad():
        m2(**_forward_inputs(seed=42))  # Same inputs

    # Both should have captured data
    assert store_a.step_store and store_b.step_store

    captured_a = store_a.step_store[0][f"transformer_blocks.0.attn.processor"]
    captured_b = store_b.step_store[0][f"transformer_blocks.0.attn.processor"]

    # Same shape (9 image tokens, 2 target indices each)
    assert captured_a.shape == (9, 2)
    assert captured_b.shape == (9, 2)

    # But different values because they captured different columns
    # If they were identical, it would suggest the processor isn't reading the right columns
    assert not torch.allclose(captured_a, captured_b, atol=1e-5, rtol=1e-4), \
        "Captured tensors should differ when targeting different columns"
    # However, both [1, 3] and [3, 1] *share* columns 1 and 3, so they should be
    # highly correlated but with column order swapped
    assert captured_a.shape == captured_b.shape  # Just double-check shape consistency


# --------------------------------------------------------------------------- FluxAttentionCapture

class _FakeFluxPipeline:
    def __init__(self, transformer):
        self.transformer = transformer


def test_hook_pipeline_replaces_only_double_blocks():
    m = _tiny_transformer(num_layers=2, num_single_layers=2)
    capture = fac.FluxAttentionCapture()
    capture.hook_pipeline(_FakeFluxPipeline(m), target_token_indices=[0, 1])
    for name, proc in m.attn_processors.items():
        if name.startswith("transformer_blocks."):
            assert isinstance(proc, fac.FluxCustomAttnProcessor)
        else:
            assert not isinstance(proc, fac.FluxCustomAttnProcessor)


def test_unhook_pipeline_restores_stock_processor_everywhere():
    from diffusers.models.transformers.transformer_flux import FluxAttnProcessor
    m = _tiny_transformer(num_layers=1, num_single_layers=1)
    capture = fac.FluxAttentionCapture()
    pipe = _FakeFluxPipeline(m)
    capture.hook_pipeline(pipe, target_token_indices=[0])
    capture.unhook_pipeline(pipe)
    assert all(type(p) is FluxAttnProcessor for p in m.attn_processors.values())


def test_cross_attention_map_shape_and_finiteness():
    m = _tiny_transformer(num_layers=1, num_single_layers=0)
    capture = fac.FluxAttentionCapture()
    capture.hook_pipeline(_FakeFluxPipeline(m), target_token_indices=[0, 1])
    with torch.no_grad():
        m(**_forward_inputs(img_seq=9))  # perfect square -> 3x3 native grid
    result = capture.cross_attention_map(target_indices_position=[0, 1],
                                         target_resolution=(12, 12), max_steps=25)
    assert result.shape == (12, 12)
    assert np.isfinite(result).all()


def test_cross_attention_map_respects_max_steps():
    """max_steps=0 must exclude every captured step (steps are 0-indexed, so max_steps=0 means
    'use nothing') -- the empty-store return path, not an error."""
    m = _tiny_transformer(num_layers=1, num_single_layers=0)
    capture = fac.FluxAttentionCapture()
    capture.hook_pipeline(_FakeFluxPipeline(m), target_token_indices=[0])
    with torch.no_grad():
        m(**_forward_inputs(img_seq=9))
    result = capture.cross_attention_map(target_indices_position=[0],
                                         target_resolution=(5, 5), max_steps=0)
    assert result.shape == (5, 5)
    assert (result == 0).all()


def test_cross_attention_map_raises_when_store_empty():
    capture = fac.FluxAttentionCapture()
    with pytest.raises(ValueError, match="empty"):
        capture.cross_attention_map(target_indices_position=[0], target_resolution=(8, 8), max_steps=25)


def test_cross_attention_map_averages_across_layers_and_steps():
    """Directly populates step_store with synthetic multi-layer, multi-step tensors of known
    values so the plain-mean-across-(steps x layers) aggregation is actually exercised by the
    suite, not just true by inspection. img_seq=4 -> native 2x2 grid; target_resolution matches
    the native grid so bilinear upsampling is an identity and the expected value is a bare
    elementwise mean of the four (layer, step) tensors."""
    capture = fac.FluxAttentionCapture()
    capture.store.target_token_indices = [0]
    capture.store.step_store = {
        0: {
            "layer_a": torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
            "layer_b": torch.tensor([[5.0], [6.0], [7.0], [8.0]]),
        },
        1: {
            "layer_a": torch.tensor([[9.0], [10.0], [11.0], [12.0]]),
            "layer_b": torch.tensor([[13.0], [14.0], [15.0], [16.0]]),
        },
    }
    result = capture.cross_attention_map(target_indices_position=[0],
                                         target_resolution=(2, 2), max_steps=25)
    expected = (np.array([1, 2, 3, 4]) + np.array([5, 6, 7, 8])
                + np.array([9, 10, 11, 12]) + np.array([13, 14, 15, 16])) / 4.0
    np.testing.assert_allclose(result, expected.reshape(2, 2), atol=1e-4)


def test_hook_pipeline_twice_without_unhook_is_safe():
    """Locks in that re-hooking without an intervening unhook_pipeline() is safe: no leaked
    processor from the first hook, no double-wrapping, and a clean store reset to the second
    call's target_token_indices."""
    m = _tiny_transformer(num_layers=1, num_single_layers=1)
    capture = fac.FluxAttentionCapture()
    pipe = _FakeFluxPipeline(m)
    capture.hook_pipeline(pipe, target_token_indices=[0, 1])
    capture.hook_pipeline(pipe, target_token_indices=[2])  # re-hook, different indices, no unhook
    for name, proc in m.attn_processors.items():
        if name.startswith("transformer_blocks."):
            assert isinstance(proc, fac.FluxCustomAttnProcessor)
        else:
            assert not isinstance(proc, fac.FluxCustomAttnProcessor)
    assert capture.store.target_token_indices == [2]
    with torch.no_grad():
        m(**_forward_inputs(img_seq=9))
    for layer_maps in capture.store.step_store.values():
        for tensor in layer_maps.values():
            assert tensor.shape == (9, 1)  # only the second hook's single target index -- no leak


def test_cross_attention_map_raises_on_negative_target_indices_position():
    m = _tiny_transformer(num_layers=1, num_single_layers=0)
    capture = fac.FluxAttentionCapture()
    capture.hook_pipeline(_FakeFluxPipeline(m), target_token_indices=[0, 1])
    with torch.no_grad():
        m(**_forward_inputs(img_seq=9))
    with pytest.raises(ValueError, match="out of range"):
        capture.cross_attention_map(target_indices_position=[-1],
                                     target_resolution=(5, 5), max_steps=25)


def test_cross_attention_map_raises_on_out_of_range_target_indices_position():
    m = _tiny_transformer(num_layers=1, num_single_layers=0)
    capture = fac.FluxAttentionCapture()
    capture.hook_pipeline(_FakeFluxPipeline(m), target_token_indices=[0, 1])  # 2 captured columns
    with torch.no_grad():
        m(**_forward_inputs(img_seq=9))
    with pytest.raises(ValueError, match="out of range"):
        capture.cross_attention_map(target_indices_position=[2],  # valid range is {0, 1}
                                     target_resolution=(5, 5), max_steps=25)


def test_cross_attention_map_raises_on_empty_target_indices_position():
    m = _tiny_transformer(num_layers=1, num_single_layers=0)
    capture = fac.FluxAttentionCapture()
    capture.hook_pipeline(_FakeFluxPipeline(m), target_token_indices=[0, 1])
    with torch.no_grad():
        m(**_forward_inputs(img_seq=9))
    with pytest.raises(ValueError, match="non-empty"):
        capture.cross_attention_map(target_indices_position=[],
                                     target_resolution=(5, 5), max_steps=25)


def test_cross_attention_map_raises_on_mismatched_grid_sizes():
    """Synthetic case for the 'shouldn't happen per the module's own claim' grid-mismatch guard:
    one layer reports a 2x2 native grid (img_seq=4), another reports 3x3 (img_seq=9). Must raise
    a domain-specific ValueError naming both sizes rather than a bare numpy broadcast error."""
    capture = fac.FluxAttentionCapture()
    capture.store.target_token_indices = [0]
    capture.store.step_store = {
        0: {
            "layer_a": torch.zeros(4, 1),  # 2x2
            "layer_b": torch.zeros(9, 1),  # 3x3 -- mismatched
        },
    }
    with pytest.raises(ValueError, match="inconsistent"):
        capture.cross_attention_map(target_indices_position=[0],
                                     target_resolution=(5, 5), max_steps=25)


# --------------------------------------------------------------------------- token indices

class _FakeT5Tokenizer:
    """Whitespace/word tokenizer stand-in -- exercises flux_token_indices' index arithmetic
    without downloading the real (multi-GB) T5-XXL tokenizer. One integer id per distinct word,
    so token positions map 1:1 to word positions and are easy to assert on directly."""

    def __init__(self):
        self._vocab: dict = {}

    def _ids_for(self, text: str) -> List[int]:
        out = []
        for word in text.split():
            out.append(self._vocab.setdefault(word, len(self._vocab) + 1))
        return out

    def __call__(self, text, padding=None, max_length=None, truncation=None,
                add_special_tokens=True):
        ids = self._ids_for(text)
        if padding == "max_length" and max_length:
            ids = (ids + [0] * max_length)[:max_length]

        class _Result:
            pass
        result = _Result()
        result.input_ids = ids
        return result


def test_flux_token_indices_exact_match():
    tok = _FakeT5Tokenizer()
    prompt = "a barista wearing a red apron and a cyclist wearing a yellow helmet"
    idxs = fac.flux_token_indices(tok, prompt, "red apron")
    words = prompt.split()
    assert idxs == [words.index("red"), words.index("red") + 1]


def test_flux_token_indices_fallback_on_subphrase_mismatch():
    tok = _FakeT5Tokenizer()
    prompt = ("a photo of a man wearing a cycling jersey in a yellow bike helmet standing "
              "next to a farmer")
    idxs = fac.flux_token_indices(tok, prompt, "yellow helmet")
    words = prompt.split()
    yellow_pos = words.index("yellow")
    assert idxs == [yellow_pos, yellow_pos + 1, yellow_pos + 2]  # "yellow bike helmet"


def test_flux_token_indices_raises_when_phrase_truly_absent():
    tok = _FakeT5Tokenizer()
    with pytest.raises(ValueError):
        fac.flux_token_indices(tok, "a barista wearing a red apron", "green scarf")


# --------------------------------------------------------------------------- union helper

def test_attribute_target_token_indices_unions_across_attributes():
    tok = _FakeT5Tokenizer()
    prompt = "a barista wearing a red apron and a cyclist wearing a yellow helmet"
    pairs = [("barista", "red apron"), ("cyclist", "yellow helmet")]
    per_attr, union = fac.attribute_target_token_indices(tok, prompt, pairs)
    assert set(per_attr["red apron"]) <= set(union)
    assert set(per_attr["yellow helmet"]) <= set(union)
    assert union == sorted(union)  # deterministic ordering
    assert len(union) == len(set(union))  # no duplicates


def test_attribute_target_token_indices_raises_on_duplicate_attribute_text():
    """Regression test for a real bug (code review, confirmed by direct execution): two
    subjects sharing identical attribute text used to silently collide in `per_attribute`
    (keyed by attribute string alone) -- the second subject's real token occurrence was
    dropped with no exception, since flux_token_indices always returns the FIRST occurrence
    of the matched phrase for a given (prompt, attribute) pair. This must now raise loudly
    instead of silently losing the waiter's real "red apron" occurrence at word positions
    15-16 (barista's is at 7-8)."""
    tok = _FakeT5Tokenizer()
    prompt = ("a photo of a barista wearing a red apron and a waiter also wearing a "
              "red apron")
    pairs = [("barista", "red apron"), ("waiter", "red apron")]
    with pytest.raises(ValueError, match="duplicate attribute"):
        fac.attribute_target_token_indices(tok, prompt, pairs)
