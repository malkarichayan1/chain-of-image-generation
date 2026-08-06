# FLUX.1-dev Cross-Attention Capture Hook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and locally test (CPU-only, no GPU) a FLUX.1-dev cross-attention capture hook so `generate_anchor_images_flux.py` can populate the currently-empty `predicted_owner`/`model_scores`/`predicted_owner_full`/`model_scores_full` fields, enabling `run_five_experiments.py` to run on `artifacts_flux/`. Running the actual GPU regeneration (Pranav, on his A6000-class hardware) is out of scope for this plan — that happens after this code is merged and tested.

**Architecture:** `FluxCustomAttnProcessor` recomputes FLUX's joint attention by hand (FLUX's stock processor dispatches straight to a fused kernel that never materializes a probability matrix) on the 19 double `transformer_blocks.*` only, capturing image-rows × a pre-declared set of target T5 token columns, head-averaged, into `FluxAttentionStore`. `FluxAttentionCapture` wraps hook/unhook/aggregation. A shared `locate_attribute_phrase` helper in `anchor_common.py` fixes a phrase-matching gap (manifest says `"yellow helmet"`, prompt says `"...yellow bike helmet..."`) that breaks both the new FLUX token lookup and the existing positional baseline.

**Tech Stack:** Python, PyTorch 2.13.0 (CPU, local), diffusers 0.39.0, pytest. All new code is tested against a real (tiny) `FluxTransformer2DModel` instantiated locally — no GPU, no downloaded checkpoint needed for any test in this plan.

---

## Two things this plan resolves that the design doc left open

Both were verified against a real `FluxTransformer2DModel` during planning (see the `Verified 2026-07-31` note that will appear in `flux_attention_capture.py`'s docstring):

1. **The design doc's "open items" (padded T5 length `T`, image grid side length) don't need to be tracked as constants at all.** `T` is `encoder_hidden_states.shape[1]` and the image grid side is `sqrt(image row count)`, both read directly off tensors at call time. No hardcoded 512, no config drift risk.
2. **Storing the full attention matrix is infeasible, and the design doc's phrasing under-specified this.** At real FLUX scale (24 heads × 4096 image tokens × ~512 text tokens × 4 bytes) that's ~384MB *per layer per step*; ×19 layers ×25 steps ≈ 180GB. Fixed by telling the store which T5 columns to keep *before* generation starts (every attribute's tokens are known from the prompt spec up front) and head-averaging immediately — this is consistent with what the design doc's §B step 5 already said ("sliced ... at capture time"), just made concrete here.

---

## File Structure

- **Modify** `ssa/anchor_set/anchor_common.py` — add `locate_attribute_phrase()`.
- **Modify** `ssa/anchor_set/exp4_positional_baseline.py` — `nearest_subject_baseline()` uses it.
- **Create** `ssa/anchor_set/flux_attention_capture.py` — `FluxAttentionStore`, `FluxCustomAttnProcessor`, `FluxAttentionCapture`, `flux_token_indices()`, `attribute_target_token_indices()`.
- **Modify** `ssa/anchor_set/generate_anchor_images_flux.py` — wire the hook into `generate_and_score`, update `NUM_INFERENCE_STEPS`/window constants, pin `diffusers==0.39.0`.
- **Create** `ssa/anchor_set/tests/test_flux_attention_capture.py`
- **Modify** `ssa/anchor_set/tests/test_anchor_common.py` — tests for `locate_attribute_phrase`.
- **Modify** `ssa/anchor_set/tests/test_exp4_positional_baseline.py` — tests for the fallback path.
- **Create** `ssa/anchor_set/tests/test_flux_battery_integration.py` — FLUX-shaped smoke test of `run_five_experiments.py`.

All commands below assume the working directory is `ssa/anchor_set/` (matches every existing test docstring's `py -3 -m pytest tests/` convention).

---

### Task 1: `locate_attribute_phrase` — shared phrase-matching fallback

**Files:**
- Modify: `ssa/anchor_set/anchor_common.py`
- Test: `ssa/anchor_set/tests/test_anchor_common.py`

- [ ] **Step 1: Write the failing tests**

Append to `ssa/anchor_set/tests/test_anchor_common.py`:

```python
# --------------------------------------------------------------------------- locate_attribute_phrase

def test_locate_attribute_phrase_exact_match_returns_attribute_unchanged():
    prompt = "a photo of a barista wearing a red apron and a cyclist wearing a yellow helmet"
    idx, span = ac.locate_attribute_phrase(prompt, "red apron")
    assert idx == prompt.find("red apron")
    assert span == "red apron"


def test_locate_attribute_phrase_falls_back_to_content_words_on_subphrase_mismatch():
    """Real case from artifacts_flux/manifest.json: the manifest's attribute field says
    'yellow helmet' but the actual prompt says 'yellow bike helmet' -- exact substring match
    fails, so this must fall back to spanning from the first content word to the last."""
    prompt = ("a photo of four people standing side by side, on the far left a barista in a "
              "red apron, on the center-left a man wearing a cycling jersey in a yellow bike "
              "helmet, on the center-right a farmer holding a wooden shovel, on the far right "
              "a nurse wearing blue gloves")
    idx, span = ac.locate_attribute_phrase(prompt, "yellow helmet")
    assert span == "yellow bike helmet"
    assert prompt[idx:idx + len(span)] == span


def test_locate_attribute_phrase_raises_when_a_content_word_is_truly_absent():
    prompt = "a barista wearing a red apron"
    with pytest.raises(ValueError, match="green"):
        ac.locate_attribute_phrase(prompt, "green scarf")


def test_locate_attribute_phrase_raises_on_attribute_with_no_content_words():
    with pytest.raises(ValueError, match="content words"):
        ac.locate_attribute_phrase("a barista wearing a red apron", "   ")
```

Check the top of `test_anchor_common.py` already has `import anchor_common as ac` and `import pytest` — if the existing import alias differs, match it exactly (read the file's current imports before appending).

- [ ] **Step 2: Run tests to verify they fail**

Run (from `ssa/anchor_set/`): `py -3 -m pytest tests/test_anchor_common.py -k locate_attribute_phrase -v`
Expected: FAIL with `AttributeError: module 'anchor_common' has no attribute 'locate_attribute_phrase'`

- [ ] **Step 3: Implement `locate_attribute_phrase`**

Add to `ssa/anchor_set/anchor_common.py`, in the "Prediction from attention" section (near `mean_mass_in_box`, since both are small pure-numpy/stdlib helpers used by scoring code):

```python
def locate_attribute_phrase(prompt: str, attribute: str) -> Tuple[int, str]:
    """Where `attribute` actually occurs in `prompt`, and what substring to treat as "the
    phrase" for downstream token lookup. Tries an exact substring match first (unchanged
    behavior for every attribute that already matches verbatim). Falls back to spanning from
    the first to the last content word of `attribute` found in `prompt`, in order -- handles
    manifest attribute strings that are a strict sub-phrase of a longer descriptive phrase in
    the actual prompt (e.g. attribute="yellow helmet", prompt="...yellow bike helmet...").
    Raises ValueError if any content word is missing, or if `attribute` has none at all."""
    idx = prompt.find(attribute)
    if idx >= 0:
        return idx, attribute
    words = re.findall(r"[a-zA-Z]+", attribute.lower())
    if not words:
        raise ValueError(f"attribute {attribute!r} has no content words to match")
    positions: List[int] = []
    search_from = 0
    for word in words:
        pos = prompt.lower().find(word, search_from)
        if pos == -1:
            raise ValueError(
                f"attribute {attribute!r} word {word!r} not found in prompt {prompt!r}")
        positions.append(pos)
        search_from = pos + len(word)
    start, end = positions[0], positions[-1] + len(words[-1])
    return start, prompt[start:end]
```

This uses `re` and `List`/`Tuple`, both already imported at the top of `anchor_common.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `py -3 -m pytest tests/test_anchor_common.py -k locate_attribute_phrase -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add ssa/anchor_set/anchor_common.py ssa/anchor_set/tests/test_anchor_common.py
git commit -m "feat: add locate_attribute_phrase fallback for sub-phrase attribute mismatches"
```

---

### Task 2: Fix `nearest_subject_baseline` to use the fallback

**Files:**
- Modify: `ssa/anchor_set/exp4_positional_baseline.py:35-58`
- Test: `ssa/anchor_set/tests/test_exp4_positional_baseline.py`

- [ ] **Step 1: Write the failing test**

Append to `ssa/anchor_set/tests/test_exp4_positional_baseline.py`:

```python
def test_nearest_subject_baseline_handles_subphrase_attribute_mismatch():
    """Real FLUX case: manifest attribute is 'yellow helmet', prompt says 'yellow bike helmet'.
    Before this fix, prompt.find('yellow helmet') returns -1 and this raises ValueError."""
    prompt = ("a photo of four people standing side by side, on the far left a barista in a "
              "red apron, on the center-left a man wearing a cycling jersey in a yellow bike "
              "helmet, on the center-right a farmer holding a wooden shovel, on the far right "
              "a nurse wearing blue gloves")
    subjects = ["barista", "cyclist", "farmer", "nurse"]
    result = exp4.nearest_subject_baseline(prompt, subjects, "yellow helmet")
    assert result == "cyclist"
```

Note: `"cyclist"` never appears literally in this prompt (it's phrased as "a man wearing a cycling jersey..."), but `nearest_subject_baseline` matches on the `subjects` list entries via `prompt.find(subject)`, which will also fail to find "cyclist" as a literal substring — check this against the real data before assuming the test passes. Run Step 2 first; if this specific assertion fails for a *different* reason (subject-side matching, not attribute-side), that's a second, separate gap — read the failure message and report it before proceeding to Step 3, don't paper over it by weakening the test.

- [ ] **Step 2: Run test to verify current behavior**

Run: `py -3 -m pytest tests/test_exp4_positional_baseline.py -k subphrase -v`
Expected: FAIL. Read the actual error message carefully — confirm it's `ValueError: attribute 'yellow helmet' not found` (the bug this task fixes) and not a subject-matching error. If it's a subject-matching error instead, stop and re-scope this task before continuing — do not proceed to Step 3 with an untested assumption about which lookup is failing.

- [ ] **Step 3: Fix `nearest_subject_baseline`**

In `ssa/anchor_set/exp4_positional_baseline.py`, replace the attribute-locating line. Current code (lines 43-45):

```python
    attr_idx = prompt.find(attribute)
    if attr_idx < 0:
        raise ValueError(f"attribute {attribute!r} not found in prompt {prompt!r}")
```

Replace with:

```python
    attr_idx, _ = locate_attribute_phrase(prompt, attribute)
```

And update the import line (currently `from anchor_common import build_agreement_rows, chance_baseline, load_labels`) to:

```python
from anchor_common import build_agreement_rows, chance_baseline, load_labels, locate_attribute_phrase
```

`locate_attribute_phrase` already raises `ValueError` with a matching message shape when it fails, so no other code in `nearest_subject_baseline` needs to change — the `subject_idx = prompt.find(subject)` loop below is untouched (subjects are unaffected by this bug in the real data; if Step 2 showed otherwise, resolve that separately first).

- [ ] **Step 4: Run tests to verify they pass**

Run: `py -3 -m pytest tests/test_exp4_positional_baseline.py -v`
Expected: all tests pass, including the pre-existing ones (this change must not alter behavior for any prompt where the exact match already worked).

- [ ] **Step 5: Commit**

```bash
git add ssa/anchor_set/exp4_positional_baseline.py ssa/anchor_set/tests/test_exp4_positional_baseline.py
git commit -m "fix: nearest_subject_baseline handles sub-phrase attribute mismatches"
```

---

### Task 3: `FluxAttentionStore` + `FluxCustomAttnProcessor`

**Files:**
- Create: `ssa/anchor_set/flux_attention_capture.py`
- Test: `ssa/anchor_set/tests/test_flux_attention_capture.py`

- [ ] **Step 1: Write the failing tests**

Create `ssa/anchor_set/tests/test_flux_attention_capture.py`:

```python
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
    attn_probs inside this test rather than trusting the processor's own internal slice."""
    from diffusers.models.embeddings import apply_rotary_emb
    from diffusers.models.transformers.transformer_flux import _get_qkv_projections

    m = _tiny_transformer(num_layers=1, num_single_layers=0)
    inputs = _forward_inputs()
    block = m.transformer_blocks[0]
    attn = block.attn

    with torch.no_grad():
        query, key, value, eq, ek, ev = _get_qkv_projections(
            attn, inputs["hidden_states"], inputs["encoder_hidden_states"])
        query = attn.norm_q(query.unflatten(-1, (attn.heads, -1)))
        key = attn.norm_k(key.unflatten(-1, (attn.heads, -1)))
        eq = attn.norm_added_q(eq.unflatten(-1, (attn.heads, -1)))
        ek = attn.norm_added_k(ek.unflatten(-1, (attn.heads, -1)))
        query = torch.cat([eq, query], dim=1)
        key = torch.cat([ek, key], dim=1)
        q = query.permute(0, 2, 1, 3)
        k = key.permute(0, 2, 1, 3)
        probs = torch.softmax((q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1]), dim=-1)
        text_len = inputs["encoder_hidden_states"].shape[1]
        image_rows = probs[:, :, text_len:, :]
        row_sums = image_rows.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `py -3 -m pytest tests/test_flux_attention_capture.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'flux_attention_capture'`

- [ ] **Step 3: Implement `flux_attention_capture.py` (store + processor)**

Create `ssa/anchor_set/flux_attention_capture.py`:

```python
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
text tokens x 4 bytes = ~384MB PER LAYER PER STEP; x19 layers x25 steps = ~180GB). Instead,
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
    current prompt, so one generation pass captures everything every attribute will need."""

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
    module docstring) so hooked generation is unaffected."""

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `py -3 -m pytest tests/test_flux_attention_capture.py -v`
Expected: 5 passed (the store tests, the two equivalence/capture tests, and the independent slice-orientation sanity check).

- [ ] **Step 5: Commit**

```bash
git add ssa/anchor_set/flux_attention_capture.py ssa/anchor_set/tests/test_flux_attention_capture.py
git commit -m "feat: add FluxAttentionStore and FluxCustomAttnProcessor for FLUX double-block capture"
```

---

### Task 4: `FluxAttentionCapture` — hook/unhook/aggregation

**Files:**
- Modify: `ssa/anchor_set/flux_attention_capture.py`
- Test: `ssa/anchor_set/tests/test_flux_attention_capture.py`

- [ ] **Step 1: Write the failing tests**

Append to `ssa/anchor_set/tests/test_flux_attention_capture.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `py -3 -m pytest tests/test_flux_attention_capture.py -k "hook_pipeline or unhook or cross_attention_map" -v`
Expected: FAIL with `AttributeError: module 'flux_attention_capture' has no attribute 'FluxAttentionCapture'`

- [ ] **Step 3: Implement `FluxAttentionCapture`**

Append to `ssa/anchor_set/flux_attention_capture.py`:

```python
class FluxAttentionCapture:
    """hook_pipeline/unhook_pipeline/cross_attention_map, mirroring
    generate_anchor_images_sdxl.py's AttentionCapture -- but FLUX-specific: only the 19 double
    blocks are hooked (see module docstring), and there is exactly one native image resolution
    across every hooked layer, so aggregation across (steps x layers) is a plain mean, no
    per-resolution weighting."""

    def __init__(self):
        self.store = FluxAttentionStore()

    def hook_pipeline(self, pipeline, target_token_indices: Sequence[int]) -> None:
        self.store.reset(target_token_indices)
        processors = dict(pipeline.transformer.attn_processors)
        for name in processors:
            if name.startswith("transformer_blocks."):
                processors[name] = FluxCustomAttnProcessor(self.store, name)
        pipeline.transformer.set_attn_processor(processors)

    def unhook_pipeline(self, pipeline) -> None:
        from diffusers.models.transformers.transformer_flux import FluxAttnProcessor
        pipeline.transformer.set_attn_processor(FluxAttnProcessor())

    def cross_attention_map(self, target_indices_position: Sequence[int],
                            target_resolution: Tuple[int, int], max_steps: int) -> np.ndarray:
        """Average, over every captured double-block layer at steps < max_steps, the columns at
        `target_indices_position` (indices INTO store.target_token_indices -- the position this
        attribute's tokens occupy within the columns that were actually captured, NOT raw T5
        sequence positions), then upsample from the native image grid to target_resolution.
        Returns an all-zero map (not an error) when max_steps excludes every captured step --
        distinct from an empty store, which IS an error (nothing was ever captured)."""
        if not self.store.step_store:
            raise ValueError("Attention store is empty. Did you call hook_pipeline() and generate?")
        accum = None
        count = 0
        for step_idx, layer_maps in self.store.step_store.items():
            if step_idx >= max_steps:
                continue
            for tensor in layer_maps.values():
                arr = tensor.numpy()[:, target_indices_position].mean(axis=-1)  # (img_seq,)
                side = int(round(math.sqrt(arr.shape[0])))
                if side * side != arr.shape[0]:
                    raise ValueError(f"image token count {arr.shape[0]} is not a perfect square")
                if accum is None:
                    accum = np.zeros((side, side), dtype=np.float32)
                accum += arr.reshape(side, side)
                count += 1
        if count == 0:
            return np.zeros(target_resolution, dtype=np.float32)
        avg = torch.from_numpy(accum / count).unsqueeze(0).unsqueeze(0)
        up = F.interpolate(avg, size=target_resolution, mode="bilinear", align_corners=False)
        return up.squeeze(0).squeeze(0).numpy()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `py -3 -m pytest tests/test_flux_attention_capture.py -v`
Expected: all tests in the file pass — run the whole file, not a filtered subset, to confirm nothing from Task 3 regressed.

- [ ] **Step 5: Commit**

```bash
git add ssa/anchor_set/flux_attention_capture.py ssa/anchor_set/tests/test_flux_attention_capture.py
git commit -m "feat: add FluxAttentionCapture hook/unhook/aggregation"
```

---

### Task 5: `flux_token_indices` + `attribute_target_token_indices`

**Files:**
- Modify: `ssa/anchor_set/flux_attention_capture.py`
- Test: `ssa/anchor_set/tests/test_flux_attention_capture.py`

- [ ] **Step 1: Write the failing tests**

Append to `ssa/anchor_set/tests/test_flux_attention_capture.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `py -3 -m pytest tests/test_flux_attention_capture.py -k "token_indices" -v`
Expected: FAIL with `AttributeError: module 'flux_attention_capture' has no attribute 'flux_token_indices'`

- [ ] **Step 3: Implement both functions**

Append to `ssa/anchor_set/flux_attention_capture.py`:

```python
def flux_token_indices(tokenizer_2, prompt: str, attribute: str) -> List[int]:
    """T5 token indices for `attribute` within `prompt` -- analogous to
    generate_anchor_images_sdxl.py's token_indices(), but against tokenizer_2 (T5): T5 tokens,
    not CLIP's, are what actually enter FLUX's joint attention sequence. Falls back through
    locate_attribute_phrase when `attribute` is a strict sub-phrase of the prompt's actual
    wording (e.g. "yellow helmet" vs. "...yellow bike helmet...")."""
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
    twice from one run). Returns ({attribute: [indices]}, [union of all indices])."""
    per_attribute: Dict[str, List[int]] = {}
    union: set = set()
    for _subject, attribute in subject_attribute_pairs:
        idxs = flux_token_indices(tokenizer_2, prompt, attribute)
        per_attribute[attribute] = idxs
        union.update(idxs)
    return per_attribute, sorted(union)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `py -3 -m pytest tests/test_flux_attention_capture.py -v`
Expected: all tests pass (full file, confirm no regressions from Tasks 3-4).

- [ ] **Step 5: Commit**

```bash
git add ssa/anchor_set/flux_attention_capture.py ssa/anchor_set/tests/test_flux_attention_capture.py
git commit -m "feat: add flux_token_indices and attribute_target_token_indices"
```

---

### Task 6: Wire the hook into `generate_anchor_images_flux.py`

**Files:**
- Modify: `ssa/anchor_set/generate_anchor_images_flux.py`

This task has no new automated test: it requires a live FLUX pipeline (GPU, multi-GB checkpoint download) to execute, which is out of scope for local CPU verification. Correctness here rests on (a) Tasks 1-5's tests, which cover every piece of logic this step wires together, and (b) the pilot protocol in the design doc (Pranav regenerating 3 prompts and comparing against the existing PNGs) — that pilot is the actual test for this task, and it happens after this code ships, not during this plan.

- [ ] **Step 1: Update the module docstring and constants**

In `ssa/anchor_set/generate_anchor_images_flux.py`, replace the header comment (lines 1-8):

```python
#!/usr/bin/env python
"""
FLUX.1-dev variant of Stage 1, WITH cross-attention capture (double blocks only -- see
flux_attention_capture.py and docs/superpowers/specs/2026-07-31-flux-attention-hook-design.md).

Regenerates the SAME 105 images as the original run (same prompts, same pinned seeds from
manifest.json) so the existing 227 human-labeled rows stay valid -- see the design doc's pilot
protocol before running this against the full prompt set.

SPLITTING across 2 Kaggle notebooks (committed runs):
  Notebook 1: SPLIT_START=0,   SPLIT_END=53
  Notebook 2: SPLIT_START=53,  SPLIT_END=105
"""
```

Update the pip install lines (currently lines 24-30, `diffusers>=0.31`) to pin diffusers per the design doc's version decision:

```python
if "KAGGLE_KERNEL_RUN_TYPE" in _os.environ:
    import subprocess as _sp, sys as _sys
    _sp.check_call([_sys.executable, "-m", "pip", "install", "-q",
        "--force-reinstall", "--no-deps",
        "torch==2.5.1+cu121", "torchvision==0.20.1+cu121",
        "--index-url", "https://download.pytorch.org/whl/cu121"])
    _sp.check_call([_sys.executable, "-m", "pip", "install", "-q",
        "diffusers==0.39.0", "transformers>=4.44", "bitsandbytes", "scipy", "sentencepiece"])
    _sp.check_call([_sys.executable, "-m", "pip", "install", "-q", "--no-deps", "accelerate"])
```

(Same change to the Colab branch a few lines below — `diffusers>=0.31` → `diffusers==0.39.0`.)

Add, near the existing `NUM_INFERENCE_STEPS = 25` constant:

```python
NUM_INFERENCE_STEPS = 25  # Full quality
EARLY_WINDOW_FRACTION = 0.5  # matches generate_anchor_images_sdxl.py's convention
MAX_STEPS = int(NUM_INFERENCE_STEPS * EARLY_WINDOW_FRACTION)
```

- [ ] **Step 2: Inline the capture code (Kaggle kernel = one file, matching the established convention)**

`generate_anchor_images_sdxl.py` fully inlines its entire attention-capture class hierarchy (`AttentionStore`/`CustomAttnProcessor`/`AttentionCapture`, ~130 lines) directly in the script rather than importing it — `anchor_common.py`'s own docstring explains why: "a Kaggle script kernel is one file," the same reason it inline-duplicates `build_attribute_entry`/`build_manifest_entry`. This script must follow the same convention, not introduce a cross-file import that would silently break the moment this script is uploaded to Kaggle without its sibling.

Paste `flux_attention_capture.py`'s complete contents (Tasks 3-5: `FluxAttentionStore`, `FluxCustomAttnProcessor`, `FluxAttentionCapture`, `flux_token_indices`, `attribute_target_token_indices`) into `generate_anchor_images_flux.py`, in a new section headed like the SDXL script's (`# Attention capture -- verbatim duplicate of flux_attention_capture.py. Keep in sync.`), placed after the existing imports and before `ANCHOR_PROMPTS` loading. Also inline `anchor_common.py`'s `mean_mass_in_box` and `predicted_owner_from_attention` (lines 164-187 of that file) into the same section — this script has never imported from `anchor_common` and must not start now for the same one-file reason. Also inline `anchor_common.py`'s `locate_attribute_phrase` (Task 1), since `flux_token_indices` calls it — the pasted `flux_token_indices` currently reads `from anchor_common import locate_attribute_phrase` (see Task 5); change that line to nothing (delete the import) since the function will now be in the same file's namespace.

- [ ] **Step 3: Wire the hook into `generate_and_score`**

Replace the current attribute-scoring block (lines 263-266):

```python
    attributes_out: List[dict] = []
    for subject, attribute in spec["pairs"]:
        # Skipped cross-attention mapping for FLUX
        attributes_out.append(build_attribute_entry(attribute, subject, "unavailable", {}, "unavailable", {}))
```

with:

```python
    per_attr_indices, union_indices = attribute_target_token_indices(
        models.txt2img.tokenizer_2, prompt, spec["pairs"])
    capture = FluxAttentionCapture()
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
    capture.unhook_pipeline(models.txt2img)
```

No import statement needed for `FluxAttentionCapture`/`attribute_target_token_indices`/`predicted_owner_from_attention` — Step 2 already put them in this same file's namespace. Read this against the surrounding function body before editing — `generate_and_score` currently generates the image once (to check detection), then this adds a *second*, hooked generation using the *same already-chosen seed* (`chosen_seed`, already resolved earlier in the function from the pinned `spec["seed"]`) so the image stays identical and the detection/box results (`subject_boxes`, computed from the first generation) still apply to it. This doubles generation cost per prompt (one pass to detect, one hooked pass to capture) — flag this explicitly to Pranav before the full run: it roughly doubles wall-clock time beyond the earlier "~1 hour for 105 prompts" estimate, which assumed one generation pass per prompt.

- [ ] **Step 4: Manually trace the wiring (no automated test — read this task's own note above)**

Re-read the modified `generate_and_score` function top to bottom. Confirm:
- `chosen_seed` is defined before this block runs (it is, earlier in the existing function).
- `subject_boxes` is defined before this block runs (it is, from `assign_subjects`).
- `prompt` and `spec` are in scope (they are, function parameters).
- The second generation call uses `torch.Generator(DEVICE).manual_seed(chosen_seed)` — the *same* seed already used for the detection-pass generation, so the image itself is unchanged; only the hook adds instrumentation.

This step is a checklist, not a test run — there is no CPU-executable path through this function (it needs the real FLUX pipeline). Do not mark this task complete without doing this trace; it's the only verification this task gets before the GPU pilot.

- [ ] **Step 5: Commit**

```bash
git add ssa/anchor_set/generate_anchor_images_flux.py
git commit -m "feat: wire FLUX attention capture into generate_anchor_images_flux.py"
```

---

### Task 7: FLUX-shaped battery integration smoke test

**Files:**
- Create: `ssa/anchor_set/tests/test_flux_battery_integration.py`

Purpose: the generic `artifacts_dummy` fixture (`make_dummy_artifacts.py`) already smoke-tests `run_five_experiments.py` end-to-end, but its prompts never hit the sub-phrase attribute mismatch (Task 1/2's bug) — it always uses exact attribute phrasing. This test builds a small manifest that DOES include a `"yellow helmet"` / `"...yellow bike helmet..."`-style mismatch, with real (non-empty) `model_scores`/`model_scores_full`, and confirms the full battery completes — the integration risk unique to FLUX, not covered by existing fixtures.

- [ ] **Step 1: Write the failing test**

Create `ssa/anchor_set/tests/test_flux_battery_integration.py`:

```python
"""Integration smoke test: does run_five_experiments.py complete against a FLUX-shaped manifest
that includes a sub-phrase attribute mismatch (Task 1/2's bug) and populated model_scores on
every attribute? The generic artifacts_dummy fixture never exercises the mismatch path, since
its prompts always use exact attribute phrasing -- this fixture is built to hit it deliberately.
Run from inside ssa/anchor_set/:  py -3 -m pytest tests/"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import anchor_common as ac
import run_five_experiments as rfe


def _flux_shaped_manifest():
    prompt_a = "a photo of a barista wearing a red apron and a cyclist wearing a yellow bike helmet"
    prompt_b = "a photo of a chef wearing a white hat and a farmer holding a wooden shovel"
    images = [
        dict(prompt_id=0, n=2, prompt=prompt_a, subjects=["barista", "cyclist"], seed=42,
             detected=True, num_people_detected=2, image_path="artifacts_flux/images/p0.png",
             attributes=[
                 dict(attribute="red apron", intended_subject="barista", predicted_owner="barista",
                      model_scores={"barista": 0.7, "cyclist": 0.3},
                      predicted_owner_full="barista", model_scores_full={"barista": 0.6, "cyclist": 0.4}),
                 # "yellow helmet" (manifest) vs "yellow bike helmet" (prompt) -- the mismatch case
                 dict(attribute="yellow helmet", intended_subject="cyclist", predicted_owner="cyclist",
                      model_scores={"barista": 0.2, "cyclist": 0.8},
                      predicted_owner_full="cyclist", model_scores_full={"barista": 0.3, "cyclist": 0.7}),
             ]),
        dict(prompt_id=1, n=2, prompt=prompt_b, subjects=["chef", "farmer"], seed=7,
             detected=True, num_people_detected=2, image_path="artifacts_flux/images/p1.png",
             attributes=[
                 dict(attribute="white hat", intended_subject="chef", predicted_owner="farmer",
                      model_scores={"chef": 0.4, "farmer": 0.6},
                      predicted_owner_full="chef", model_scores_full={"chef": 0.55, "farmer": 0.45}),
                 dict(attribute="wooden shovel", intended_subject="farmer", predicted_owner="farmer",
                      model_scores={"chef": 0.1, "farmer": 0.9},
                      predicted_owner_full="farmer", model_scores_full={"chef": 0.15, "farmer": 0.85}),
             ]),
    ]
    manifest = dict(model="black-forest-labs/FLUX.1-dev", candidate_seeds=[42, 7], img_size=1024,
                    num_inference_steps=25, early_window_fraction=0.5, images=images)
    labels = {
        ac.label_key(0, "red apron"): "barista",
        ac.label_key(0, "yellow helmet"): "cyclist",
        ac.label_key(1, "white hat"): "chef",
        ac.label_key(1, "wooden shovel"): "farmer",
    }
    counts = {ac.count_key(0): ac.COUNT_CLEAN, ac.count_key(1): ac.COUNT_CLEAN}
    return manifest, labels, counts


def test_run_all_completes_on_flux_shaped_manifest_with_subphrase_mismatch():
    manifest, labels, counts = _flux_shaped_manifest()
    results = rfe.run_all(manifest, labels, counts, n_seeds=5)
    assert set(results.keys()) == {
        "experiment_1", "experiment_2", "experiment_3", "experiment_4", "experiment_5"}


def test_experiment_2_is_available_when_full_trajectory_scores_are_populated():
    manifest, labels, counts = _flux_shaped_manifest()
    results = rfe.run_all(manifest, labels, counts, n_seeds=5)
    assert results["experiment_2"]["available"] is True


def test_experiment_4_does_not_crash_on_the_subphrase_attribute():
    """This is the specific regression Task 1/2 fixed -- exp4's baseline join used to raise
    ValueError on "yellow helmet" not matching "...yellow bike helmet..." literally."""
    manifest, labels, counts = _flux_shaped_manifest()
    results = rfe.run_all(manifest, labels, counts, n_seeds=5)
    assert results["experiment_4"]["headline"]["n"] >= 1
```

Check `run_five_experiments.py`'s actual `run_all` signature and `anchor_common.py`'s exact `COUNT_CLEAN` constant name before running — both were read during planning and should match, but confirm against the current file state, not this plan's memory of it.

- [ ] **Step 2: Run test to verify it fails for the RIGHT reason**

Run: `py -3 -m pytest tests/test_flux_battery_integration.py -v`
Expected at this point in the plan (Tasks 1-6 already committed): this should actually PASS already, since Tasks 1-2 already fixed the underlying bug and this test doesn't depend on Tasks 3-6's FLUX-capture code at all (it only exercises `run_five_experiments.py` against a hand-built manifest). If it fails, that means Task 1 or 2's fix has a gap this test exposes that the earlier unit tests didn't catch — treat that as a real finding, not a step to skip past.

- [ ] **Step 3: If it failed, fix; if it passed, note why this step is different from the others**

This task's "Step 3: implement" is conditional. If Step 2 passed, there is nothing to implement — commit as-is; this is a regression test confirming Tasks 1-2's fix generalizes to a realistic FLUX manifest shape, not a new feature. If Step 2 failed, diagnose against the specific assertion that failed before changing any code — do not modify this test to make it pass without understanding why the earlier tasks' fix didn't cover this case.

- [ ] **Step 4: Run full test suite for a final regression check**

Run: `py -3 -m pytest tests/ -v`
Expected: all tests pass, including every pre-existing test file untouched by this plan (`test_exp1_accuracy_by_n.py`, `test_exp2_window_ablation.py`, `test_exp5_count_clean_subset.py`, `test_run_five_experiments.py`, etc.) — this plan's changes (`locate_attribute_phrase`, `nearest_subject_baseline`) must not alter any already-published SDXL number.

- [ ] **Step 5: Commit**

```bash
git add ssa/anchor_set/tests/test_flux_battery_integration.py
git commit -m "test: add FLUX-shaped battery integration smoke test"
```

---

## What this plan does NOT cover

- Running `generate_anchor_images_flux.py` on a real GPU (Pranav's pilot, then full 105-prompt rerun) — gated on this code being reviewed first, per the design doc's pilot protocol.
- Updating `CLAUDE.md` / `RESULTS.md` with real findings — there are no real findings until the GPU rerun happens.
- The 38 `FluxSingleTransformerBlock`s — explicitly out of scope per the design doc.
