#!/usr/bin/env python
"""
Stage 2 of the pipeline: CLIPSeg segmentation, cached as raw sigmoid maps keyed by
(image_path, attribute).

Caching *before* any thresholding is what makes Stage 3 (score_chains.py) pure numpy:
every scoring-side change -- a new threshold, a new control -- becomes a cache read
instead of a fresh CLIPSeg call. This module deliberately does not import torch or
transformers at module level (only inside `_default_segment_fn`, lazily) so that reading
an existing cache -- which is all score_chains.py and calibrate_threshold.py need --
never requires a model or a GPU.
"""
import hashlib
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
from PIL import Image

from config import Config, DEFAULT_CONFIG_PATH

CACHE_DTYPE = np.float16

SegmentFn = Callable[["Image.Image", str], np.ndarray]
ImageLoader = Callable[[str], "Image.Image"]


def _default_image_loader(image_path: str) -> Image.Image:
    return Image.open(image_path).convert("RGB")


def cache_key(image_path: str, attribute: str) -> str:
    """Deterministic, filesystem-safe key for a (image_path, attribute) pair.
    Hashed rather than slugified because attributes contain spaces/punctuation and
    image_path is a full relative path -- neither is a safe filename on its own."""
    digest = hashlib.sha1(f"{image_path}|{attribute}".encode("utf-8")).hexdigest()
    return digest[:20]


def cache_path(cache_dir: Path, image_path: str, attribute: str) -> Path:
    return Path(cache_dir) / f"{cache_key(image_path, attribute)}.npy"


def load_cached_map(cache_dir: Path, image_path: str, attribute: str) -> Optional[np.ndarray]:
    path = cache_path(cache_dir, image_path, attribute)
    if not path.exists():
        return None
    return np.load(path).astype(np.float32)


def save_cached_map(cache_dir: Path, image_path: str, attribute: str, sigmoid_map: np.ndarray) -> Path:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_path(cache_dir, image_path, attribute)
    np.save(path, sigmoid_map.astype(CACHE_DTYPE))
    return path


def _default_segment_fn() -> SegmentFn:
    """Lazily loads CLIPSeg once and returns a closure that segments (image, attribute)
    pairs. Mirrors SpatialSemanticAlignment.segment() in pilot/spatial_semantic_alignment.py
    (same use_safetensors=True fix, see coig-ssa-metric-direction memory) -- duplicated
    rather than imported so this module's only hard dependency is transformers, not the
    full diffusers/torchvision stack Stage 1 needs. Keep in sync with that file's Phase A
    segmentation logic if it changes."""
    import torch
    from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

    clipseg_model = Config().models.clipseg_model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = CLIPSegProcessor.from_pretrained(clipseg_model)
    model = CLIPSegForImageSegmentation.from_pretrained(
        clipseg_model, use_safetensors=True
    ).to(device)

    def _segment(image: Image.Image, attribute: str) -> np.ndarray:
        import torch.nn.functional as F

        h, w = image.size[1], image.size[0]
        inputs = processor(text=[attribute], images=[image], padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**inputs).logits.unsqueeze(1)
            logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
        return torch.sigmoid(logits).cpu().numpy().squeeze()

    return _segment


def get_sigmoid_map(
    cache_dir: Path,
    image_path: str,
    attribute: str,
    segment_fn: SegmentFn,
    image_loader: ImageLoader = _default_image_loader,
) -> np.ndarray:
    """Cache-or-compute a single (image_path, attribute) pair."""
    cached = load_cached_map(cache_dir, image_path, attribute)
    if cached is not None:
        return cached
    sigmoid_map = segment_fn(image_loader(image_path), attribute)
    save_cached_map(cache_dir, image_path, attribute, np.asarray(sigmoid_map, dtype=np.float32))
    return np.asarray(sigmoid_map, dtype=np.float32)


def _detected_chains(manifest: dict) -> List[dict]:
    return [c for c in manifest["chains"] if c["detected"]]


def manifest_attribute_universe(manifest: dict) -> List[str]:
    """Every distinct attribute string across every successfully-detected chain in the
    manifest -- not just a chain's own attributes. Stage 3's substituted/scrambled-cross
    controls need a chain's own images segmented against *other* chains' attributes too,
    and that draw isn't known until Stage 3 runs, so Stage 2 must precompute the full
    image x attribute cross product up front while it still has GPU/model access."""
    attrs = {step["attribute"] for c in _detected_chains(manifest) for step in c["steps"]}
    return sorted(attrs)


def manifest_image_paths(manifest: dict) -> List[str]:
    paths = []
    for c in _detected_chains(manifest):
        paths.append(c["base_image_path"])
        paths.extend(step["image_path"] for step in c["steps"])
    return paths


def build_segmentation_cache(
    manifest: dict,
    cache_dir: Path,
    segment_fn: Optional[SegmentFn] = None,
    image_loader: ImageLoader = _default_image_loader,
) -> Dict[str, int]:
    """Segments every (image, attribute) pair needed by any Stage-3 condition and caches
    the raw sigmoid maps. Returns counts rather than the maps themselves -- callers read
    the cache back via load_cached_map/get_sigmoid_map."""
    if segment_fn is None:
        segment_fn = _default_segment_fn()

    image_paths = manifest_image_paths(manifest)
    attributes = manifest_attribute_universe(manifest)

    computed = 0
    already_cached = 0
    for image_path in image_paths:
        for attribute in attributes:
            if load_cached_map(cache_dir, image_path, attribute) is not None:
                already_cached += 1
                continue
            get_sigmoid_map(cache_dir, image_path, attribute, segment_fn, image_loader)
            computed += 1

    return dict(
        total_pairs=len(image_paths) * len(attributes),
        pairs_computed=computed,
        pairs_already_cached=already_cached,
        num_images=len(image_paths),
        num_attributes=len(attributes),
    )


def load_manifest(manifest_path: Path) -> dict:
    with open(manifest_path) as f:
        return json.load(f)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Build Stage 2's CLIPSeg segmentation cache")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    ap.add_argument("--manifest", type=str, default=None)
    ap.add_argument("--cache-dir", type=str, default=None)
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    manifest_path = args.manifest or str(cfg.paths.manifest_path)
    cache_dir_arg = args.cache_dir or str(cfg.paths.segmentation_cache_dir)

    manifest = load_manifest(manifest_path)
    stats = build_segmentation_cache(manifest, Path(cache_dir_arg))
    print(f"Segmentation cache built at {cache_dir_arg}: {stats}")
