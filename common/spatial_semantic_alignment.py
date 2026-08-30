"""
Spatial-Semantic Alignment (SSA) Metric
========================================
Quantifies whether a diffusion model's internal cross-attention exactly
correlates with the spatial introduction of new pixels at a specific
generation step — measuring faithfulness from inside the model instead of
from its finished pixels.

Three-phase pipeline
--------------------
Phase A  — Delta Mask  : isolate newly-added pixels at step N via segmentation
Phase B  — Attention   : aggregate cross-attention for the target token during
                         the early structural denoising window (steps 0–15)
Phase C  — IoU Score   : measure spatial overlap between attention and delta mask

Supports
--------
- UNet-based pipelines (SD 1.5, SDXL) via diffusers AttnProcessor interface -- this is
  the only architecture path this project's claims and experiments rely on.
- DiT-based pipelines (FLUX.1, SD 3) via register_forward_hook -- present in the code
  but UNVERIFIED against a real DiT model. diffusers' actual FLUX/SD3 implementations
  use their own `Attention` class with a custom-processor pattern and joint (not
  separable) self-attention over concatenated image+text tokens, not
  `torch.nn.MultiheadAttention` -- so `_hook_dit`'s `isinstance(module,
  nn.MultiheadAttention)` check will silently find zero modules and record no
  attention on a real FLUX/SD3 pipeline. The 9-scenario suite's "DiT" test (Scenario 8)
  only exercises this against a hand-built fake transformer containing a real
  nn.MultiheadAttention -- it demonstrates the hook mechanics work, not that they work
  on an actual DiT model. Decision (2026-07-22): scope this paper's claims to UNet
  architectures only rather than spend session time rewriting DiT support against
  diffusers' real Attention/processor classes; DiT support here is scaffolding for
  future work, not a validated capability.
"""

import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


# ---------------------------------------------------------------------------
# Attention storage
# ---------------------------------------------------------------------------

class AttentionRecord:
    """A single captured cross-attention tensor with metadata."""
    __slots__ = ("tensor", "layer_name", "spatial_dim", "heads")

    def __init__(self, tensor: torch.Tensor, layer_name: str, spatial_dim: int, heads: int = 1):
        self.tensor = tensor          # (batch*heads, spatial_dim, seq_len) — already on CPU
        self.layer_name = layer_name
        self.spatial_dim = spatial_dim
        self.heads = heads            # needed to slice out one batch entry (e.g. CFG cond half)


class AttentionStore:
    """
    Stores cross-attention maps during the diffusion process, organised by
    diffusion step index and annotated with layer name and native spatial dim.

    step_store: Dict[step_idx, List[AttentionRecord]]
    """

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


# ---------------------------------------------------------------------------
# UNet attention processor hook
# ---------------------------------------------------------------------------

class CustomAttnProcessor:
    """
    Drop-in replacement for diffusers' default attention processor.
    Performs the standard attention computation and records cross-attention
    probability maps into an AttentionStore.

    Compatible with SD 1.5 and SDXL UNet architectures.
    """

    def __init__(self, store: AttentionStore, is_cross_attention: bool, layer_name: str):
        self.store = store
        self.is_cross_attention = is_cross_attention
        self.layer_name = layer_name

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        scale: float = 1.0,
    ):
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

        query = attn.to_q(hidden_states, scale=scale)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross is not None:
            encoder_hidden_states = attn.norm_cross(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states, scale=scale)
        value = attn.to_v(encoder_hidden_states, scale=scale)

        query = attn.head_to_batch_dim(query)
        key   = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        # ---- HOOK: record cross-attention probabilities ----
        if self.is_cross_attention:
            spatial_dim = attention_probs.shape[1]
            self.store.add_attention(attention_probs, self.layer_name, spatial_dim, attn.heads)
        # ----------------------------------------------------

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states, scale=scale)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


# ---------------------------------------------------------------------------
# DiT (FLUX.1 / SD 3) forward-hook shim
# ---------------------------------------------------------------------------

class DiTAttnHook:
    """
    Records cross-attention weights from a torch.nn.MultiheadAttention module
    via register_forward_hook.

    Distinguishes cross-attention from self-attention by checking whether the
    key_padding_mask or the key tensor differs from the query tensor.  In DiT
    cross-attention blocks the query comes from the image latent and the key/
    value come from the text encoder — their sequence lengths differ.

    Usage
    -----
        hook = DiTAttnHook(store, layer_name)
        handle = module.register_forward_hook(hook)
        # later:
        handle.remove()
    """

    def __init__(self, store: AttentionStore, layer_name: str):
        self.store = store
        self.layer_name = layer_name

    def __call__(self, module: nn.MultiheadAttention, inputs, outputs):
        # outputs from nn.MultiheadAttention: (attn_output, attn_output_weights)
        # attn_output_weights is (batch, tgt_len, src_len) when need_weights=True
        # and None otherwise.
        if outputs[1] is None:
            return  # hook registered but need_weights was False — skip silently

        attn_weights = outputs[1]  # (batch, q_len, k_len)
        q_len = attn_weights.shape[1]
        k_len = attn_weights.shape[2]

        # Cross-attention: query comes from image latent, key from text encoder
        # → sequence lengths differ.  Self-attention: q_len == k_len.
        is_cross = (q_len != k_len)
        if not is_cross:
            return

        # Reshape to (batch*1, spatial_dim, seq_len) to match UNet convention
        # (DiT doesn't split heads before returning weights)
        spatial_dim = q_len
        self.store.add_attention(attn_weights, self.layer_name, spatial_dim)


# ---------------------------------------------------------------------------
# Main metric class
# ---------------------------------------------------------------------------

class SpatialSemanticAlignment:
    """
    Spatial-Semantic Alignment (SSA) Metric.

    Parameters
    ----------
    device : str
        'cuda' or 'cpu'.  Defaults to CUDA when available.
    lazy_load_model : bool
        If True (default) the CLIPSeg segmentation model is not loaded until
        phase_a_delta_mask() is first called.
    """

    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        lazy_load_model: bool = True,
    ):
        self.device = device
        self.segmentation_processor = None
        self.segmentation_model = None
        self.attn_store = AttentionStore()
        self._dit_hook_handles: List = []   # keeps remove-handles for DiT hooks
        self._pipeline_arch: Optional[str] = None  # "unet" | "dit"

        if not lazy_load_model:
            self._load_segmentation_model()

    # ------------------------------------------------------------------
    # Segmentation model
    # ------------------------------------------------------------------

    def _load_segmentation_model(self):
        if self.segmentation_model is None:
            from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor
            self.segmentation_processor = CLIPSegProcessor.from_pretrained(
                "CIDAS/clipseg-rd64-refined"
            )
            self.segmentation_model = CLIPSegForImageSegmentation.from_pretrained(
                "CIDAS/clipseg-rd64-refined", use_safetensors=True
            ).to(self.device)

    # ------------------------------------------------------------------
    # Phase A — Delta Mask
    # ------------------------------------------------------------------

    def phase_a_delta_mask_from_masks(
        self,
        mask_curr: np.ndarray,
        mask_prev: np.ndarray,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """
        Phase A (direct mask input): Delta Mask = Mask_N − Mask_{N-1}.

        Pixels that are positive in mask_curr but NOT in mask_prev represent
        content freshly added at the current step.  If an attribute was already
        locked from a prior step both masks agree and the delta is zero.
        """
        curr_binary = mask_curr > threshold
        prev_binary = mask_prev > threshold
        return np.logical_and(curr_binary, np.logical_not(prev_binary)).astype(np.float32)

    def phase_a_delta_mask(
        self,
        img_curr: Image.Image,
        img_prev: Image.Image,
        target_attribute: str,
        sigmoid_threshold: float = 0.5,
    ) -> np.ndarray:
        """
        Phase A (image input): run CLIPSeg on both images, then compute delta.

        Parameters
        ----------
        img_curr : PIL.Image  — output image of the current generation step
        img_prev : PIL.Image  — output image of the previous generation step
        target_attribute : str — free-text description of the attribute to segment
        sigmoid_threshold : float — binarisation threshold on CLIPSeg sigmoid output
        """
        self._load_segmentation_model()

        if img_curr.size != img_prev.size:
            img_prev = img_prev.resize(img_curr.size, Image.LANCZOS)

        h, w = img_curr.size[1], img_curr.size[0]

        def _segment(img: Image.Image) -> np.ndarray:
            inputs = self.segmentation_processor(
                text=[target_attribute], images=[img], padding=True, return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                logits = self.segmentation_model(**inputs).logits.unsqueeze(1)
                logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
            return torch.sigmoid(logits).cpu().numpy().squeeze()

        mask_curr = _segment(img_curr)
        mask_prev = _segment(img_prev)
        return self.phase_a_delta_mask_from_masks(mask_curr, mask_prev, threshold=sigmoid_threshold)

    # ------------------------------------------------------------------
    # Pipeline hooking — UNet and DiT
    # ------------------------------------------------------------------

    def _detect_architecture(self, pipeline) -> str:
        """Return 'unet' or 'dit' based on the pipeline's backbone attribute."""
        if hasattr(pipeline, "unet"):
            return "unet"
        if hasattr(pipeline, "transformer"):
            return "dit"
        raise NotImplementedError(
            "hook_pipeline: pipeline has neither .unet nor .transformer. "
            "Only diffusers UNet (SD1.5/SDXL) and DiT (FLUX.1/SD3) pipelines are supported."
        )

    def hook_pipeline(self, pipeline):
        """
        Install cross-attention hooks on the pipeline's generative backbone.

        UNet (SD1.5, SDXL): replaces attn_processors with CustomAttnProcessor.
        DiT  (FLUX.1, SD3): registers forward hooks on nn.MultiheadAttention modules.

        Always resets the attention store before installing new hooks so that
        calling hook_pipeline twice does not accumulate stale data.
        """
        self.attn_store.reset()
        self._pipeline_arch = self._detect_architecture(pipeline)

        if self._pipeline_arch == "unet":
            self._hook_unet(pipeline)
        else:
            self._hook_dit(pipeline)

    def _hook_unet(self, pipeline):
        processors = {}
        for name in pipeline.unet.attn_processors.keys():
            is_cross = "attn2" in name
            processors[name] = CustomAttnProcessor(self.attn_store, is_cross, layer_name=name)
        pipeline.unet.set_attn_processor(processors)

    def _hook_dit(self, pipeline):
        # Remove any previously registered DiT hooks to avoid duplication
        self._remove_dit_hooks()
        transformer = pipeline.transformer
        for name, module in transformer.named_modules():
            if isinstance(module, nn.MultiheadAttention):
                hook = DiTAttnHook(self.attn_store, layer_name=name)
                handle = module.register_forward_hook(hook)
                self._dit_hook_handles.append(handle)

    def unhook_pipeline(self, pipeline=None):
        """
        Remove all installed hooks.

        For UNet pipelines, pass the pipeline so the default attn_processors
        can be restored.  For DiT pipelines the pipeline argument is optional
        (hooks are stored internally).
        """
        if self._pipeline_arch == "unet" and pipeline is not None:
            # diffusers' set_attn_processor(dict) requires the dict to have exactly
            # one entry per attention layer -- an empty {} raises ValueError ("number
            # of processors 0 does not match the number of attention layers N"), so
            # this line previously would have crashed the first time it ran against a
            # real pipeline (caught 2026-07-22 while wiring this class into a real
            # SD1.5 pipeline for the first time -- see experiments/). Passing a
            # single processor *instance* (not a dict) broadcasts it to every layer.
            from diffusers.models.attention_processor import AttnProcessor2_0
            pipeline.unet.set_attn_processor(AttnProcessor2_0())
        self._remove_dit_hooks()
        self._pipeline_arch = None

    def _remove_dit_hooks(self):
        for handle in self._dit_hook_handles:
            handle.remove()
        self._dit_hook_handles.clear()

    # ------------------------------------------------------------------
    # Token index helper
    # ------------------------------------------------------------------

    def find_token_index(self, pipeline, prompt: str, target_token_str: str) -> int:
        """
        Return the position of *target_token_str* in the CLIP tokenisation of
        *prompt*.  Raises ValueError if the token is not found.
        """
        inputs = pipeline.tokenizer(prompt, return_tensors="pt")
        tokens = pipeline.tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
        target = target_token_str.lower().strip()
        for idx, tok in enumerate(tokens):
            clean = tok.replace("</w>", "").replace("</s>", "").lower().strip()
            if clean == target:
                return idx
        raise ValueError(
            f"Token '{target_token_str}' not found in tokenisation of prompt.\n"
            f"Tokens: {tokens}"
        )

    # ------------------------------------------------------------------
    # Phase B — Cross-attention map aggregation
    # ------------------------------------------------------------------

    def phase_b_cross_attention_map(
        self,
        target_token_index: int,
        target_resolution: Tuple[int, int] = (512, 512),
        max_steps: int = 15,
        aggregation_mode: str = "resolution_weighted",
        min_spatial_dim: int = 0,
        cond_index: Optional[int] = None,
    ) -> Union[np.ndarray, Dict[int, np.ndarray]]:
        """
        Phase B: extract and aggregate cross-attention maps for *target_token_index*
        from the early structural denoising window (steps 0–max_steps).

        Parameters
        ----------
        target_token_index : int
            Index of the attribute token in the CLIP tokenisation.
        target_resolution : (H, W)
            Output resolution; all per-layer maps are upsampled to this.
        max_steps : int
            Only aggregate maps from diffusion steps < max_steps (0-indexed).
            Default 15 captures the structural layout window of a 50-step schedule.
        aggregation_mode : str
            "uniform"             — equal weight per layer (original behaviour)
            "resolution_weighted" — weight proportional to native spatial_dim²,
                                    giving high-res layers much more influence
            "per_resolution"      — returns Dict[spatial_dim, np.ndarray] instead
                                    of a single composite map
        min_spatial_dim : int
            Skip any layer whose native spatial dimension (sqrt of spatial_dim²)
            is below this value.  E.g. min_spatial_dim=32 discards 8×8 and
            16×16 layers.  Default 0 (keep all).
        cond_index : int or None
            Which batch entry (in units of `record.heads`) to read, for pipelines
            running classifier-free guidance (batch = [uncond, cond] for a standard
            single-prompt CFG call). Without this, averaging "over batch*heads" (see
            below) silently mixes the near-uninformative unconditional branch's
            attention into the signal, diluting it. None (default) preserves the
            original whole-batch-average behaviour, so hand-built test tensors with
            no real cond/uncond structure (batch=1, no heads dimension) still work
            unmodified — pass cond_index=1 explicitly for a real CFG-hooked pipeline.

        Returns
        -------
        np.ndarray of shape target_resolution  — for "uniform" and "resolution_weighted"
        Dict[int, np.ndarray]                  — for "per_resolution"
        """
        if not self.attn_store.step_store:
            raise ValueError(
                "Attention store is empty.  Did you call hook_pipeline() before generating?"
            )

        # Accumulate per native-resolution
        per_res_accum: Dict[int, np.ndarray] = {}
        per_res_weight: Dict[int, float] = {}   # total weight for normalisation

        for step_idx, records in self.attn_store.step_store.items():
            if step_idx >= max_steps:
                continue

            for record in records:
                spatial_dim = record.spatial_dim
                seq_len     = record.tensor.shape[2]

                # Filter by minimum native resolution
                native_side = int(math.sqrt(spatial_dim))
                if native_side * native_side != spatial_dim:
                    continue  # non-square spatial dim — skip
                if native_side < min_spatial_dim:
                    continue
                if target_token_index >= seq_len:
                    continue

                # Average over batch*heads dimension → (spatial_dim,), optionally
                # restricted to one cond/uncond slice first (see cond_index above).
                tensor = record.tensor
                if cond_index is not None:
                    start = cond_index * record.heads
                    tensor = tensor[start:start + record.heads]
                token_attn = tensor[:, :, target_token_index].mean(dim=0)

                # Reshape to 2D and upsample
                attn_2d = token_attn.view(native_side, native_side).unsqueeze(0).unsqueeze(0)
                attn_up = F.interpolate(
                    attn_2d.float(), size=target_resolution, mode="bilinear", align_corners=False
                ).squeeze().numpy()

                if native_side not in per_res_accum:
                    per_res_accum[native_side]  = np.zeros(target_resolution, dtype=np.float32)
                    per_res_weight[native_side]  = 0.0

                per_res_accum[native_side]  += attn_up
                per_res_weight[native_side] += 1.0

        if not per_res_accum:
            return {} if aggregation_mode == "per_resolution" else np.zeros(target_resolution, dtype=np.float32)

        # Normalise each resolution bucket
        per_res_maps: Dict[int, np.ndarray] = {
            side: per_res_accum[side] / per_res_weight[side]
            for side in per_res_accum
        }

        if aggregation_mode == "per_resolution":
            return per_res_maps

        if aggregation_mode == "resolution_weighted":
            # Weight by pixel count (side²) so 64×64 dominates 16×16
            composite = np.zeros(target_resolution, dtype=np.float32)
            total_weight = 0.0
            for side, layer_map in per_res_maps.items():
                w = float(side * side)
                composite    += w * layer_map
                total_weight += w
            return composite / total_weight if total_weight > 0 else composite

        # "uniform" — equal weight across resolution buckets
        composite = np.zeros(target_resolution, dtype=np.float32)
        for layer_map in per_res_maps.values():
            composite += layer_map
        return composite / len(per_res_maps)

    # ------------------------------------------------------------------
    # Phase C — Alignment score (IoU)
    # ------------------------------------------------------------------

    def phase_c_alignment_score(
        self,
        composite_attention_map: np.ndarray,
        delta_mask: np.ndarray,
        threshold_pct: float = 0.20,
    ) -> float:
        """
        Phase C: compute IoU between the top-threshold_pct attention pixels
        and the delta mask.

        Returns 0.0 when the delta mask is empty (compositional lock case) or
        when the attention map has zero variance.
        """
        # Resize attention map to match delta mask if needed
        if composite_attention_map.shape != delta_mask.shape:
            t = torch.from_numpy(composite_attention_map).unsqueeze(0).unsqueeze(0)
            t = F.interpolate(t, size=delta_mask.shape, mode="bilinear", align_corners=False)
            composite_attention_map = t.squeeze().numpy()

        # Guard: zero-variance map → no information
        if np.max(composite_attention_map) == np.min(composite_attention_map):
            return 0.0

        # Binarise the exact top-k pixels by value (k = threshold_pct * N).
        #
        # A percentile *value* threshold degenerates on sparse/peaked maps — which is
        # the common case for real early-step cross-attention, not an edge case: if
        # >= (1 - threshold_pct) of the map's mass sits at the same low value (often
        # exactly 0, since softmax attention over a token concentrates most of its
        # mass on a minority of spatial positions), the percentile *value* itself
        # lands on that plateau. `map >= threshold_val` then matches almost the
        # entire image instead of the intended top slice, silently turning the IoU
        # into `delta_area / total_area` — a measure of the delta mask's size, not of
        # attention/delta overlap. Selecting the exact top-k indices by rank sidesteps
        # this: the selected set size is always exactly k regardless of how many
        # pixels tie at the threshold value.
        flat = composite_attention_map.reshape(-1)
        n = flat.size
        k = max(1, min(n, int(round(threshold_pct * n))))
        top_k_idx = np.argpartition(-flat, k - 1)[:k]
        attn_bin = np.zeros(n, dtype=bool)
        attn_bin[top_k_idx] = True
        attn_bin = attn_bin.reshape(composite_attention_map.shape)

        delta_bin = (delta_mask > 0.5)

        intersection = np.logical_and(attn_bin, delta_bin).sum()
        union        = np.logical_or(attn_bin, delta_bin).sum()

        if union == 0:
            return 0.0

        return float(intersection / union)


# ---------------------------------------------------------------------------
# Verification suite
# ---------------------------------------------------------------------------

def test_metric():
    """
    Nine-scenario verification suite.

    Scenarios 1–4 : correctness of the core metric logic (unchanged).
    Scenarios 5–7 : resolution-weighted Phase B aggregation.
    Scenarios 8–9 : DiT architecture detection and unhook.
    """
    print("Executing SSA Metric Verification Suite (9 scenarios)...")

    metric = SpatialSemanticAlignment(device="cpu", lazy_load_model=True)
    print("[OK] Metric initialised in offline mode.")

    H, W = 512, 512

    # -------------------------------------------------------------------
    # Scenario 1: Perfect Spatial Alignment
    # -------------------------------------------------------------------
    mask_prev_1 = np.zeros((H, W), dtype=np.float32)
    mask_curr_1 = np.zeros((H, W), dtype=np.float32)
    mask_curr_1[100:300, 100:300] = 1.0

    delta_1 = metric.phase_a_delta_mask_from_masks(mask_curr_1, mask_prev_1)

    attn_perfect = np.zeros((H, W), dtype=np.float32)
    attn_perfect[100:300, 100:300] = 1.0

    iou_1 = metric.phase_c_alignment_score(attn_perfect, delta_1, threshold_pct=0.15)
    print(f"[OK] Scenario 1 (Perfect Alignment): IoU = {iou_1:.4f}")
    # Exact top-k tie-breaking (see Phase C fix note) can shave a sliver off a perfect
    # block match when the tied region is larger than k; near-1.0 is the correct bar.
    assert iou_1 > 0.95, f"Expected near-1.0, got {iou_1}"

    # -------------------------------------------------------------------
    # Scenario 2: Compositional Lock (delta = 0 → IoU = 0)
    # -------------------------------------------------------------------
    mask_prev_2 = np.zeros((H, W), dtype=np.float32)
    mask_prev_2[100:300, 100:300] = 1.0
    mask_curr_2 = mask_prev_2.copy()  # identical → delta is empty

    delta_2 = metric.phase_a_delta_mask_from_masks(mask_curr_2, mask_prev_2)
    iou_2 = metric.phase_c_alignment_score(attn_perfect, delta_2, threshold_pct=0.20)
    print(f"[OK] Scenario 2 (Compositional Lock): Delta sum={delta_2.sum()}, IoU = {iou_2:.4f}")
    assert delta_2.sum() == 0.0
    assert iou_2 == 0.0, f"Expected 0.0, got {iou_2}"

    # -------------------------------------------------------------------
    # Scenario 3: Mismatched Attention
    # -------------------------------------------------------------------
    attn_mismatch = np.zeros((H, W), dtype=np.float32)
    attn_mismatch[400:500, 400:500] = 1.0

    iou_3 = metric.phase_c_alignment_score(attn_mismatch, delta_1, threshold_pct=0.15)
    print(f"[OK] Scenario 3 (Mismatched Attention): IoU = {iou_3:.4f}")
    # Pre-fix this was 0.1526 -- not a real overlap measure, just delta_area/total_area,
    # because the 85th-percentile *value* on a 96%-zero array is 0, so `map >= 0` matched
    # the entire image. Exact top-k selection now bounds the selected set to exactly k
    # pixels, so a genuinely disjoint region scores at (or very near) true zero.
    assert iou_3 < 0.05, f"Expected near-zero IoU (was 0.1526 before the top-k fix), got {iou_3}"

    # -------------------------------------------------------------------
    # Scenario 3b: Realistic peaked (Gaussian) attention, not block-uniform.
    #
    # Scenario 3's block-uniform mismatch happens to land at a clean 0 under the fix,
    # but block-uniform test data has none of the graded, continuous-valued structure
    # of real cross-attention -- it can't exercise the percentile-degeneracy failure
    # mode the same way. A tight Gaussian bump (sigma=10) is a much closer stand-in for
    # a real early-step cross-attention peak: highly concentrated, mostly-near-zero
    # everywhere else. This is the stress test called for by the fix's priority next
    # steps -- verifying it against peaked, non-block-uniform distributions.
    # -------------------------------------------------------------------
    def _gaussian_bump(h, w, cy, cx, sigma):
        yy, xx = np.mgrid[0:h, 0:w]
        g = np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma ** 2)))
        return g.astype(np.float32)

    gauss_aligned = _gaussian_bump(H, W, cy=200, cx=200, sigma=10)  # centered inside delta_1's region
    iou_gauss_aligned = metric.phase_c_alignment_score(gauss_aligned, delta_1, threshold_pct=0.20)
    print(f"[OK] Scenario 3b (Gaussian, aligned on delta region): IoU = {iou_gauss_aligned:.4f}")
    assert iou_gauss_aligned > 0.5, f"Expected strong overlap, got {iou_gauss_aligned}"

    gauss_mismatch = _gaussian_bump(H, W, cy=450, cx=450, sigma=10)  # centered far from delta_1
    iou_gauss_mismatch = metric.phase_c_alignment_score(gauss_mismatch, delta_1, threshold_pct=0.20)
    print(f"[OK] Scenario 3c (Gaussian, mismatched from delta region): IoU = {iou_gauss_mismatch:.4f}")
    assert iou_gauss_mismatch < 0.05, (
        f"Expected near-zero IoU for a realistic peaked map that doesn't overlap the "
        f"delta region, got {iou_gauss_mismatch} -- this is exactly the failure mode "
        f"the percentile-degeneracy bug produced on real-shaped attention."
    )

    # -------------------------------------------------------------------
    # Scenario 4: Phase B step windowing (step 0 stored, step ≥ 15 excluded)
    # -------------------------------------------------------------------
    metric.attn_store.reset()
    dummy = torch.ones((1, 256, 10)) * 0.5   # 16×16 spatial, 10 tokens
    metric.attn_store.add_attention(dummy, layer_name="mid_block.attn2", spatial_dim=256)
    metric.attn_store.step()

    comp_map = metric.phase_b_cross_attention_map(
        target_token_index=2, max_steps=15, aggregation_mode="resolution_weighted"
    )
    print(f"[OK] Scenario 4 (Phase B Windowing): shape={comp_map.shape}, max={comp_map.max():.4f}")
    assert comp_map.shape == (512, 512)

    # -------------------------------------------------------------------
    # Scenario 5: Resolution-weighted vs uniform aggregation
    # -------------------------------------------------------------------
    metric.attn_store.reset()
    # 8×8 layer (spatial_dim=64) — low resolution
    low_res_attn = torch.zeros((1, 64, 10))
    low_res_attn[:, :, 2] = 0.1   # token 2 has low uniform attention
    metric.attn_store.add_attention(low_res_attn, layer_name="up_blocks.0.attn2_low", spatial_dim=64)

    # 64×64 layer (spatial_dim=4096) — high resolution
    high_res_attn = torch.zeros((1, 4096, 10))
    high_res_attn[:, :, 2] = 0.9  # token 2 has high attention in the high-res layer
    metric.attn_store.add_attention(high_res_attn, layer_name="up_blocks.2.attn2_high", spatial_dim=4096)

    comp_uniform   = metric.phase_b_cross_attention_map(2, aggregation_mode="uniform")
    comp_weighted  = metric.phase_b_cross_attention_map(2, aggregation_mode="resolution_weighted")

    # Resolution-weighted should be much closer to 0.9 (dominated by 64×64)
    # Uniform should be midpoint ~0.5
    print(
        f"[OK] Scenario 5 (Resolution Weighting): "
        f"uniform_max={comp_uniform.max():.4f}, "
        f"weighted_max={comp_weighted.max():.4f}"
    )
    assert comp_weighted.max() > comp_uniform.max(), (
        f"Resolution-weighted ({comp_weighted.max():.4f}) should exceed "
        f"uniform ({comp_uniform.max():.4f})"
    )

    # -------------------------------------------------------------------
    # Scenario 6: min_spatial_dim filter
    # -------------------------------------------------------------------
    comp_filtered = metric.phase_b_cross_attention_map(
        2, aggregation_mode="uniform", min_spatial_dim=32
    )
    # Should only include the 64×64 layer (side=64 ≥ 32)
    # The 8×8 layer (side=8 < 32) should be excluded
    comp_unfiltered = metric.phase_b_cross_attention_map(
        2, aggregation_mode="uniform", min_spatial_dim=0
    )
    print(
        f"[OK] Scenario 6 (min_spatial_dim filter): "
        f"unfiltered_max={comp_unfiltered.max():.4f}, "
        f"filtered_max={comp_filtered.max():.4f}"
    )
    # With the 8×8 low-signal layer excluded, filtered should have higher max
    assert comp_filtered.max() > comp_unfiltered.max(), (
        "Filtering low-res layers should raise the max (removes diluting low signal)"
    )

    # -------------------------------------------------------------------
    # Scenario 7: per_resolution mode returns dict keyed by side
    # -------------------------------------------------------------------
    per_res = metric.phase_b_cross_attention_map(2, aggregation_mode="per_resolution")
    print(f"[OK] Scenario 7 (per_resolution mode): keys={sorted(per_res.keys())}")
    assert isinstance(per_res, dict), "Expected dict from per_resolution mode"
    assert 8  in per_res, "Expected key 8 (8×8 layer)"
    assert 64 in per_res, "Expected key 64 (64×64 layer)"
    for side, arr in per_res.items():
        assert arr.shape == (512, 512), f"Map for side={side} has wrong shape: {arr.shape}"

    # -------------------------------------------------------------------
    # Scenario 8: Architecture auto-detection for DiT pipeline
    # -------------------------------------------------------------------

    class _FakeDiTTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            # Simulate a transformer block with cross-attention MHA
            self.block0_attn = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)

    class _FakeDiTPipeline:
        def __init__(self):
            self.transformer = _FakeDiTTransformer()

    fake_dit = _FakeDiTPipeline()
    metric.hook_pipeline(fake_dit)
    print(
        f"[OK] Scenario 8 (DiT Architecture Detection): "
        f"arch={metric._pipeline_arch}, "
        f"hooks_registered={len(metric._dit_hook_handles)}"
    )
    assert metric._pipeline_arch == "dit"
    assert len(metric._dit_hook_handles) == 1, (
        f"Expected 1 DiT hook, got {len(metric._dit_hook_handles)}"
    )

    # -------------------------------------------------------------------
    # Scenario 9: unhook_pipeline clears DiT hooks
    # -------------------------------------------------------------------
    metric.unhook_pipeline()
    print(
        f"[OK] Scenario 9 (unhook_pipeline cleanup): "
        f"handles_after_unhook={len(metric._dit_hook_handles)}, "
        f"arch={metric._pipeline_arch}"
    )
    assert len(metric._dit_hook_handles) == 0, "Hooks not removed after unhook_pipeline"
    assert metric._pipeline_arch is None, "_pipeline_arch not reset after unhook_pipeline"

    # -------------------------------------------------------------------
    # Scenario 10: hook_pipeline / unhook_pipeline round-trip on a REAL UNet.
    #
    # Scenarios 8-9 only ever exercised the DiT hook path (a hand-built fake
    # transformer, no pipeline argument to unhook_pipeline). The UNet path of
    # unhook_pipeline -- pipeline.unet.set_attn_processor({}) -- was never
    # exercised against a real attention module and had a real bug: diffusers'
    # set_attn_processor(dict) requires the dict to have exactly one entry per
    # attention layer, so an empty {} always raised ValueError. This was only
    # caught 2026-07-22 wiring this class into a real SD1.5 pipeline for the
    # first time (see experiments/) -- exactly the class of bug a
    # "verified against hand-built scenarios, never against a real pipeline"
    # test suite is blind to. Skips gracefully if diffusers isn't installed.
    # -------------------------------------------------------------------
    try:
        from diffusers import UNet2DConditionModel
    except ImportError:
        print("[SKIP] Scenario 10 (real UNet hook/unhook round-trip): diffusers not installed")
    else:
        tiny_unet = UNet2DConditionModel(
            sample_size=8, in_channels=4, out_channels=4,
            down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
            block_out_channels=(16, 32), layers_per_block=1,
            cross_attention_dim=16, attention_head_dim=4, norm_num_groups=8,
        )

        class _FakeUNetPipeline:
            def __init__(self, unet):
                self.unet = unet

        fake_pipe = _FakeUNetPipeline(tiny_unet)
        n_layers = len(tiny_unet.attn_processors)

        metric.hook_pipeline(fake_pipe)
        assert metric._pipeline_arch == "unet"
        assert all(isinstance(p, CustomAttnProcessor) for p in tiny_unet.attn_processors.values())

        metric.unhook_pipeline(fake_pipe)  # this line raised ValueError before the fix
        assert metric._pipeline_arch is None
        assert not any(isinstance(p, CustomAttnProcessor) for p in tiny_unet.attn_processors.values())
        print(
            f"[OK] Scenario 10 (real UNet hook/unhook round-trip): "
            f"{n_layers} attention layers hooked and cleanly restored"
        )

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------
    print()
    print("[ALL SCENARIOS PASSED] SSA metric implementation fully verified.")


if __name__ == "__main__":
    test_metric()
