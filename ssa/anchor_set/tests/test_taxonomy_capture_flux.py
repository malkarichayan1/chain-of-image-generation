"""Tests for taxonomy_capture_flux.py -- the per-head/layer/step capture behind experiments
#14/#16/#17/#18/#19.

The point of this file is to de-risk a GPU run before it is paid for. Two things must hold
or the whole capture is worthless:

  1. Hooking must not perturb generation (else the captured attention describes an image
     nobody labeled).
  2. Reducing per-head cells and then pooling them must reproduce what the ALREADY-PUBLISHED
     head-averaging path produces (else the taxonomy is measuring a different quantity than
     the paper's headline numbers, and no cell is comparable to them).

Both are checked against a real (tiny) FluxTransformer2DModel with live RoPE, the same
harness tests/test_flux_attention_capture.py uses. Requires torch/diffusers -- skipped, not
failed, if unavailable. Run from inside ssa/anchor_set/:  py -3 -m pytest tests/
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")

import flux_attention_capture as fac
import taxonomy_capture_flux as tcf


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


class _FakePipeline:
    """Minimal stand-in exposing the one attribute hook()/double_block_names() touch."""

    def __init__(self, transformer):
        self.transformer = transformer


# --------------------------------------------------------------- box_to_grid_mask

def test_box_to_grid_mask_covers_the_expected_half():
    """A box spanning the left half of a 1024px image must mark the left half of the grid."""
    mask = tcf.box_to_grid_mask([0, 0, 512, 1024], side=64, img_size=1024).reshape(64, 64)

    assert mask[:, :32].all()
    assert not mask[:, 32:].any()


def test_box_to_grid_mask_never_returns_an_empty_mask():
    """A sub-cell-sized box must still mark one cell: a masked mean over zero cells is NaN,
    which would propagate silently through every downstream aggregate."""
    mask = tcf.box_to_grid_mask([100.0, 100.0, 100.4, 100.4], side=64, img_size=1024)

    assert mask.sum() >= 1


def test_box_to_grid_mask_clamps_degenerate_and_out_of_range_boxes():
    for box in ([500, 500, 400, 400], [-50, -50, 20, 20], [1000, 1000, 5000, 5000]):
        mask = tcf.box_to_grid_mask(box, side=64, img_size=1024)

        assert mask.shape == (64 * 64,)
        assert mask.sum() >= 1


# --------------------------------------------------------------- spatial_entropy

def test_entropy_is_maximal_for_a_uniform_map():
    uniform = torch.ones(1, 1, 16)

    assert tcf.spatial_entropy(uniform).item() == pytest.approx(math.log(16), abs=1e-5)


def test_entropy_is_zero_for_a_fully_peaked_map():
    peaked = torch.zeros(1, 1, 16)
    peaked[0, 0, 3] = 1.0

    assert tcf.spatial_entropy(peaked).item() == pytest.approx(0.0, abs=1e-5)


def test_entropy_is_scale_invariant():
    """Renormalization means entropy measures peakedness, not how much total mass the
    attribute drew -- otherwise entropy would confound the two."""
    m = torch.rand(1, 1, 32)

    assert tcf.spatial_entropy(m).item() == pytest.approx(
        tcf.spatial_entropy(m * 7.5).item(), abs=1e-5)


def test_entropy_of_all_zero_map_is_zero_not_nan():
    assert tcf.spatial_entropy(torch.zeros(1, 1, 8)).item() == 0.0


# --------------------------------------------------------------- pooled_owner_from_cells

def test_pooled_owner_picks_the_argmax_subject_per_attribute():
    # (layers=1, steps=2, heads=1, attrs=2, subjects=2)
    cells = np.zeros((1, 2, 1, 2, 2), dtype=np.float32)
    cells[..., 0, 0] = 5.0   # attribute 0 -> subject 0
    cells[..., 1, 1] = 5.0   # attribute 1 -> subject 1

    assert tcf.pooled_owner_from_cells(cells, ["barista", "cyclist"], max_steps=2) == [
        "barista", "cyclist"]


def test_pooled_owner_respects_the_early_window():
    """Steps outside the early window must not influence the call -- that window is the
    published metric's definition, not an arbitrary slice."""
    cells = np.zeros((1, 4, 1, 1, 2), dtype=np.float32)
    cells[0, :2, 0, 0, 0] = 1.0    # early steps favour subject 0
    cells[0, 2:, 0, 0, 1] = 99.0   # late steps favour subject 1, and must be ignored

    assert tcf.pooled_owner_from_cells(cells, ["a", "b"], max_steps=2) == ["a"]


# --------------------------------------------------------------- mean_abs_pixel_diff

def test_identical_images_have_zero_pixel_diff():
    from PIL import Image
    img = Image.fromarray(np.full((8, 8, 3), 128, dtype=np.uint8))

    assert tcf.mean_abs_pixel_diff(img, img) == 0.0


def test_pixel_diff_is_normalized_to_zero_one():
    from PIL import Image
    black = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))
    white = Image.fromarray(np.full((8, 8, 3), 255, dtype=np.uint8))

    assert tcf.mean_abs_pixel_diff(black, white) == pytest.approx(1.0)


def test_pixel_diff_rejects_mismatched_sizes():
    from PIL import Image
    a = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))
    b = Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8))

    with pytest.raises(ValueError, match="size mismatch"):
        tcf.mean_abs_pixel_diff(a, b)


# --------------------------------------------------------------- hooking

def test_double_block_names_excludes_single_blocks_and_sorts_numerically():
    """The substring trap: "single_transformer_blocks.0..." CONTAINS "transformer_blocks.",
    so a membership test would silently hook the single blocks too."""
    pipe = _FakePipeline(_tiny_transformer(num_layers=2, num_single_layers=2))

    names = tcf.double_block_names(pipe)

    assert names == ["transformer_blocks.0.attn.processor",
                     "transformer_blocks.1.attn.processor"]


def test_hooking_does_not_perturb_the_model_output():
    """If hooking changed generation, the captured attention would describe an image nobody
    labeled. Tolerance is float-summation-order noise between a manual softmax and the
    fused kernel, per flux_attention_capture.py's module docstring."""
    m_stock = _tiny_transformer(seed=0)
    m_hooked = _tiny_transformer(seed=0)   # identical seed -> identical weights
    inputs = _forward_inputs(seed=1)

    with torch.no_grad():
        expected = m_stock(**inputs)[0]

    store = tcf.TaxonomyStore()
    store.reset([0, 1], torch.ones(1, 9, dtype=torch.bool), [[0, 1]],
                tcf.double_block_names(_FakePipeline(m_hooked)))
    tcf.hook(_FakePipeline(m_hooked), store)
    with torch.no_grad():
        actual = m_hooked(**inputs)[0]

    assert torch.allclose(expected, actual, atol=1e-5)


def test_unhook_restores_stock_processors_everywhere():
    from diffusers.models.transformers.transformer_flux import FluxAttnProcessor
    pipe = _FakePipeline(_tiny_transformer(num_layers=1, num_single_layers=1))
    store = tcf.TaxonomyStore()
    store.reset([0], torch.ones(1, 9, dtype=torch.bool), [[0]], tcf.double_block_names(pipe))
    tcf.hook(pipe, store)

    tcf.unhook(pipe)

    assert all(isinstance(p, FluxAttnProcessor)
               for p in pipe.transformer.attn_processors.values())


# --------------------------------------------------------------- the equivalence test

def test_head_averaged_cells_reproduce_the_published_pooled_map():
    """THE test that justifies the GPU spend.

    taxonomy_capture_flux reduces per head; the published pipeline averages heads inside the
    processor. Averaging this file's per-head cells back down must land on exactly what
    flux_attention_capture.py's cross_attention_map produces, or every taxonomy cell is
    measuring a different quantity than the paper's headline numbers and none of them are
    comparable.

    Compared at the native grid resolution so the check isolates the reduction itself
    rather than bilinear-upsampling differences.
    """
    img_seq, side = 9, 3
    inputs = _forward_inputs(img_seq=img_seq, seed=2)

    m_ref = _tiny_transformer(num_layers=2, num_single_layers=0, seed=0)
    ref = fac.FluxAttentionCapture()
    ref.hook_pipeline(_FakePipeline(m_ref), target_token_indices=[0, 1])
    with torch.no_grad():
        m_ref(**inputs)
    ref_map = ref.cross_attention_map(target_indices_position=[0, 1],
                                      target_resolution=(side, side), max_steps=1)

    m_tax = _tiny_transformer(num_layers=2, num_single_layers=0, seed=0)
    store = tcf.TaxonomyStore()
    store.reset([0, 1], torch.ones(1, img_seq, dtype=torch.bool), [[0, 1]],
                tcf.double_block_names(_FakePipeline(m_tax)))
    tcf.hook(_FakePipeline(m_tax), store)
    with torch.no_grad():
        m_tax(**inputs)

    tax_map = (store.pooled_accum / store.pooled_count)[0].reshape(side, side)

    assert np.allclose(ref_map, tax_map, atol=1e-6)


def test_in_box_reduction_equals_a_masked_mean_of_the_head_averaged_map():
    """The per-cell number must be the plain mean of the attribute's map inside the mask --
    the grid-space analogue of anchor_common.mean_mass_in_box."""
    img_seq = 9
    mask = np.zeros(img_seq, dtype=bool)
    mask[:4] = True
    store = tcf.TaxonomyStore()
    store.reset([0], torch.from_numpy(mask[None, :]), [[0]], ["transformer_blocks.0"])

    rows = torch.arange(2 * img_seq, dtype=torch.float32).reshape(2, img_seq, 1)
    store.add("transformer_blocks.0", rows)

    in_box, _ = store.cells[(0, "transformer_blocks.0")]
    expected_head0 = rows[0, :4, 0].mean().item()

    assert in_box[0, 0, 0] == pytest.approx(expected_head0, rel=1e-3)


def test_stack_places_cells_at_their_own_layer_and_step_indices():
    """A transposed or off-by-one axis here would silently scramble the entire taxonomy --
    every "late blocks bind better" style conclusion depends on this indexing."""
    layers = ["transformer_blocks.0", "transformer_blocks.1"]
    store = tcf.TaxonomyStore()
    store.reset([0], torch.ones(1, 9, dtype=torch.bool), [[0]], layers)

    store.add(layers[1], torch.full((2, 9, 1), 3.0))   # layer 1, step 0
    store.step()
    store.add(layers[0], torch.full((2, 9, 1), 7.0))   # layer 0, step 1

    in_box, _ = store.stack(n_attributes=1, n_subjects=1, n_heads=2)

    assert in_box.shape == (2, tcf.NUM_INFERENCE_STEPS, 2, 1, 1)
    assert in_box[1, 0, 0, 0, 0] == pytest.approx(3.0, rel=1e-3)
    assert in_box[0, 1, 0, 0, 0] == pytest.approx(7.0, rel=1e-3)
    assert in_box[0, 0, 0, 0, 0] == 0.0   # never written


def test_add_rejects_a_non_square_image_grid():
    store = tcf.TaxonomyStore()
    store.reset([0], torch.ones(1, 8, dtype=torch.bool), [[0]], ["transformer_blocks.0"])

    with pytest.raises(ValueError, match="perfect square"):
        store.add("transformer_blocks.0", torch.zeros(2, 8, 1))
