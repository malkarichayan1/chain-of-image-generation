#!/usr/bin/env python
"""
Stage 1 of the metric-A human-agreement anchor set
(docs/superpowers/specs/2026-07-23-metric-a-human-anchor-set-design.md):
18 compositional prompts (6 each at n=2/3/4 subjects) -> one one-shot SD1.5 image each,
with Mask R-CNN person detection, CLIP subject assignment, early-window cross-attention
capture, and a per-attribute `predicted_owner` = the subject whose detected box holds the
most of that attribute's attention mass. Emits manifest.json + images/*.png.

Runs on GPU/Kaggle as a single-file script kernel (same convention as generate_chains.py,
which this does NOT modify). Because a script kernel is one file, this script is fully
self-contained: it embeds ANCHOR_PROMPTS (identical to prompt_specs.json) and inline-copies
the pure predicted_owner / manifest-entry logic from anchor_common.py -- keep the two in
sync, exactly as generate_chains.py keeps its attention capture in sync with
pilot/spatial_semantic_alignment.py.

This is the ONE-SHOT setting metric A is scoped to (a single compositional image), NOT a
chain -- so none of generate_chains.py's inpainting helpers (box_mask / encode_to_latents /
mask_to_latent_tensor / the latent-blend callback) are carried over. The attention-capture
classes ARE carried over verbatim.

Ground-truth framing (see spec): a prompt's *intended* pairing is NOT the ground truth.
SD1.5 mis-binds; that is the phenomenon under test. `predicted_owner` is the metric's guess.
The human label recorded later (label_images.py) is the ground truth. Their agreement
(analyze_agreement.py) is the result. Subject localization uses reliable person detection,
never OWL-ViT attribute-box containment -- letting attention alone decide ownership over
reliably-detected boxes is precisely what decouples the attention claim from the brittle
detector.
"""
import json
import math
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

if "KAGGLE_KERNEL_RUN_TYPE" in __import__("os").environ:
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--progress-bar", "off",
        "torch==2.5.1", "torchvision==0.20.1",
        "--index-url", "https://download.pytorch.org/whl/cu121",
    ])
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--progress-bar", "off",
        "diffusers>=0.31", "transformers>=4.44", "accelerate", "scipy",
    ])

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
# Candidate seeds tried IN ORDER per prompt until exactly n people are detected; the first
# that passes is used and recorded. Same validated seed pool as generate_chains.py, frozen
# here before any agreement number is computed downstream.
CANDIDATE_SEEDS = [42, 7, 1234, 2024]
NUM_INFERENCE_STEPS = 30
EARLY_WINDOW_FRACTION = 0.5
MAX_STEPS = int(NUM_INFERENCE_STEPS * EARLY_WINDOW_FRACTION)
IMG_SIZE = 512
DETECTION_SCORE_THRESH = 0.7  # unchanged from generate_chains.py

ARTIFACTS_DIR = Path("artifacts")
IMAGES_DIR = ARTIFACTS_DIR / "images"
MANIFEST_PATH = ARTIFACTS_DIR / "manifest.json"

COCO_PERSON = 1
SD15_TXT2IMG_CANDIDATES = [
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    "sd-legacy/stable-diffusion-v1-5",
]

# =============================================================================
# ANCHOR_PROMPTS -- identical content to prompt_specs.json (that file is the human-facing
# source of truth; this literal exists because the Kaggle kernel is a single file). Each
# prompt uses distinct subjects and distinct attributes; every attribute string is a
# verbatim substring of its prompt so token_indices() can locate it.
# =============================================================================
ANCHOR_PROMPTS = [
    dict(id=0, n=2, prompt="a photo of a barista wearing a red apron and a cyclist wearing a yellow helmet",
         pairs=[("barista", "red apron"), ("cyclist", "yellow helmet")]),
    dict(id=1, n=2, prompt="a photo of a chef wearing a white hat and a farmer holding a shovel",
         pairs=[("chef", "white hat"), ("farmer", "shovel")]),
    dict(id=2, n=2, prompt="a photo of a nurse wearing blue gloves and a pilot wearing dark sunglasses",
         pairs=[("nurse", "blue gloves"), ("pilot", "dark sunglasses")]),
    dict(id=3, n=2, prompt="a photo of a teacher holding a book and a cyclist wearing a yellow helmet",
         pairs=[("teacher", "book"), ("cyclist", "yellow helmet")]),
    dict(id=4, n=2, prompt="a photo of a nurse wearing blue gloves and a farmer holding a shovel",
         pairs=[("nurse", "blue gloves"), ("farmer", "shovel")]),
    dict(id=5, n=2, prompt="a photo of a pilot wearing dark sunglasses and a barista wearing a red apron",
         pairs=[("pilot", "dark sunglasses"), ("barista", "red apron")]),
    dict(id=6, n=3, prompt="a photo of a barista wearing a red apron, a cyclist wearing a yellow helmet, and a chef wearing a white hat",
         pairs=[("barista", "red apron"), ("cyclist", "yellow helmet"), ("chef", "white hat")]),
    dict(id=7, n=3, prompt="a photo of a chef holding a pan, a farmer holding a shovel, and a nurse wearing blue gloves",
         pairs=[("chef", "pan"), ("farmer", "shovel"), ("nurse", "blue gloves")]),
    dict(id=8, n=3, prompt="a photo of a pilot wearing dark sunglasses, a teacher holding a book, and a barista wearing a red apron",
         pairs=[("pilot", "dark sunglasses"), ("teacher", "book"), ("barista", "red apron")]),
    dict(id=9, n=3, prompt="a photo of a nurse wearing blue gloves, a cyclist wearing a yellow helmet, and a farmer holding a shovel",
         pairs=[("nurse", "blue gloves"), ("cyclist", "yellow helmet"), ("farmer", "shovel")]),
    dict(id=10, n=3, prompt="a photo of a teacher holding a book, a pilot wearing dark sunglasses, and a chef wearing a white hat",
         pairs=[("teacher", "book"), ("pilot", "dark sunglasses"), ("chef", "white hat")]),
    dict(id=11, n=3, prompt="a photo of a barista wearing a red apron, a nurse wearing blue gloves, and a cyclist wearing a yellow helmet",
         pairs=[("barista", "red apron"), ("nurse", "blue gloves"), ("cyclist", "yellow helmet")]),
    dict(id=12, n=4, prompt="a photo of a barista wearing a red apron, a cyclist wearing a yellow helmet, a chef wearing a white hat, and a farmer holding a shovel",
         pairs=[("barista", "red apron"), ("cyclist", "yellow helmet"), ("chef", "white hat"), ("farmer", "shovel")]),
    dict(id=13, n=4, prompt="a photo of a nurse wearing blue gloves, a pilot wearing dark sunglasses, a teacher holding a book, and a chef holding a pan",
         pairs=[("nurse", "blue gloves"), ("pilot", "dark sunglasses"), ("teacher", "book"), ("chef", "pan")]),
    dict(id=14, n=4, prompt="a photo of a farmer holding a shovel, a barista wearing a red apron, a cyclist wearing a yellow helmet, and a nurse wearing blue gloves",
         pairs=[("farmer", "shovel"), ("barista", "red apron"), ("cyclist", "yellow helmet"), ("nurse", "blue gloves")]),
    dict(id=15, n=4, prompt="a photo of a chef wearing a white hat, a teacher holding a book, a pilot wearing dark sunglasses, and a cyclist wearing a yellow helmet",
         pairs=[("chef", "white hat"), ("teacher", "book"), ("pilot", "dark sunglasses"), ("cyclist", "yellow helmet")]),
    dict(id=16, n=4, prompt="a photo of a barista wearing a red apron, a nurse wearing blue gloves, a farmer holding a shovel, and a pilot wearing dark sunglasses",
         pairs=[("barista", "red apron"), ("nurse", "blue gloves"), ("farmer", "shovel"), ("pilot", "dark sunglasses")]),
    dict(id=17, n=4, prompt="a photo of a cyclist wearing a yellow helmet, a chef holding a pan, a teacher holding a book, and a nurse wearing blue gloves",
         pairs=[("cyclist", "yellow helmet"), ("chef", "pan"), ("teacher", "book"), ("nurse", "blue gloves")]),
    # 2026-07-24 growth batch (ids 18-23): new subject combinations, same controlled
    # vocabulary as 0-17. Only actually generated by generate_anchor_images_sdxl.py --
    # SDXL is the canonical model going forward (see CLAUDE.md) -- kept in sync here so
    # the drift guard (test_anchor_common.py) still holds if SD1.5 is ever backfilled.
    dict(id=18, n=2, prompt="a photo of a chef wearing a white hat and a nurse wearing blue gloves",
         pairs=[("chef", "white hat"), ("nurse", "blue gloves")]),
    dict(id=19, n=2, prompt="a photo of a farmer holding a shovel and a teacher holding a book",
         pairs=[("farmer", "shovel"), ("teacher", "book")]),
    dict(id=20, n=3, prompt="a photo of a barista wearing a red apron, a chef holding a pan, and a nurse wearing blue gloves",
         pairs=[("barista", "red apron"), ("chef", "pan"), ("nurse", "blue gloves")]),
    dict(id=21, n=3, prompt="a photo of a cyclist wearing a yellow helmet, a pilot wearing dark sunglasses, and a teacher holding a book",
         pairs=[("cyclist", "yellow helmet"), ("pilot", "dark sunglasses"), ("teacher", "book")]),
    dict(id=22, n=4, prompt="a photo of a chef wearing a white hat, a farmer holding a shovel, a nurse wearing blue gloves, and a teacher holding a book",
         pairs=[("chef", "white hat"), ("farmer", "shovel"), ("nurse", "blue gloves"), ("teacher", "book")]),
    dict(id=23, n=4, prompt="a photo of a barista wearing a red apron, a cyclist wearing a yellow helmet, a chef holding a pan, and a pilot wearing dark sunglasses",
         pairs=[("barista", "red apron"), ("cyclist", "yellow helmet"), ("chef", "pan"), ("pilot", "dark sunglasses")]),
    # 2026-07-25 SDXL growth batch (ids 100-184): 85 new subject combinations from the
    # SAME locked vocabulary as 0-23 -- see build_growth_specs.py (the tested source of
    # truth this literal was generated from). Only actually generated by
    # generate_anchor_images_sdxl.py; kept in sync here per the same convention as the
    # 18-23 growth batch above. Each entry pins its own `seed` (disjoint from
    # CANDIDATE_SEEDS) rather than retrying, so two entries can never collide on the
    # same (prompt, seed) pair -- see generate_and_score's GROWTH_ID_START branch.
    dict(id=100, n=2, prompt="a photo of a barista wearing a red apron and a chef wearing a white hat", seed=101,
         pairs=[("barista", "red apron"), ("chef", "white hat")]),
    dict(id=101, n=2, prompt="a photo of a barista wearing a red apron and a chef holding a pan", seed=101,
         pairs=[("barista", "red apron"), ("chef", "pan")]),
    dict(id=102, n=2, prompt="a photo of a barista wearing a red apron and a cyclist wearing a yellow helmet", seed=101,
         pairs=[("barista", "red apron"), ("cyclist", "yellow helmet")]),
    dict(id=103, n=2, prompt="a photo of a barista wearing a red apron and a farmer holding a shovel", seed=101,
         pairs=[("barista", "red apron"), ("farmer", "shovel")]),
    dict(id=104, n=2, prompt="a photo of a barista wearing a red apron and a nurse wearing blue gloves", seed=101,
         pairs=[("barista", "red apron"), ("nurse", "blue gloves")]),
    dict(id=105, n=2, prompt="a photo of a barista wearing a red apron and a pilot wearing dark sunglasses", seed=101,
         pairs=[("barista", "red apron"), ("pilot", "dark sunglasses")]),
    dict(id=106, n=2, prompt="a photo of a barista wearing a red apron and a teacher holding a book", seed=101,
         pairs=[("barista", "red apron"), ("teacher", "book")]),
    dict(id=107, n=2, prompt="a photo of a chef wearing a white hat and a cyclist wearing a yellow helmet", seed=101,
         pairs=[("chef", "white hat"), ("cyclist", "yellow helmet")]),
    dict(id=108, n=2, prompt="a photo of a chef holding a pan and a cyclist wearing a yellow helmet", seed=101,
         pairs=[("chef", "pan"), ("cyclist", "yellow helmet")]),
    dict(id=109, n=2, prompt="a photo of a chef wearing a white hat and a farmer holding a shovel", seed=101,
         pairs=[("chef", "white hat"), ("farmer", "shovel")]),
    dict(id=110, n=2, prompt="a photo of a chef holding a pan and a farmer holding a shovel", seed=101,
         pairs=[("chef", "pan"), ("farmer", "shovel")]),
    dict(id=111, n=2, prompt="a photo of a chef wearing a white hat and a nurse wearing blue gloves", seed=101,
         pairs=[("chef", "white hat"), ("nurse", "blue gloves")]),
    dict(id=112, n=2, prompt="a photo of a chef holding a pan and a nurse wearing blue gloves", seed=101,
         pairs=[("chef", "pan"), ("nurse", "blue gloves")]),
    dict(id=113, n=2, prompt="a photo of a chef wearing a white hat and a pilot wearing dark sunglasses", seed=101,
         pairs=[("chef", "white hat"), ("pilot", "dark sunglasses")]),
    dict(id=114, n=2, prompt="a photo of a chef holding a pan and a pilot wearing dark sunglasses", seed=101,
         pairs=[("chef", "pan"), ("pilot", "dark sunglasses")]),
    dict(id=115, n=2, prompt="a photo of a chef wearing a white hat and a teacher holding a book", seed=101,
         pairs=[("chef", "white hat"), ("teacher", "book")]),
    dict(id=116, n=2, prompt="a photo of a chef holding a pan and a teacher holding a book", seed=101,
         pairs=[("chef", "pan"), ("teacher", "book")]),
    dict(id=117, n=2, prompt="a photo of a cyclist wearing a yellow helmet and a farmer holding a shovel", seed=101,
         pairs=[("cyclist", "yellow helmet"), ("farmer", "shovel")]),
    dict(id=118, n=2, prompt="a photo of a cyclist wearing a yellow helmet and a nurse wearing blue gloves", seed=101,
         pairs=[("cyclist", "yellow helmet"), ("nurse", "blue gloves")]),
    dict(id=119, n=2, prompt="a photo of a cyclist wearing a yellow helmet and a pilot wearing dark sunglasses", seed=101,
         pairs=[("cyclist", "yellow helmet"), ("pilot", "dark sunglasses")]),
    dict(id=120, n=2, prompt="a photo of a cyclist wearing a yellow helmet and a teacher holding a book", seed=101,
         pairs=[("cyclist", "yellow helmet"), ("teacher", "book")]),
    dict(id=121, n=2, prompt="a photo of a farmer holding a shovel and a nurse wearing blue gloves", seed=101,
         pairs=[("farmer", "shovel"), ("nurse", "blue gloves")]),
    dict(id=122, n=2, prompt="a photo of a farmer holding a shovel and a pilot wearing dark sunglasses", seed=101,
         pairs=[("farmer", "shovel"), ("pilot", "dark sunglasses")]),
    dict(id=123, n=2, prompt="a photo of a farmer holding a shovel and a teacher holding a book", seed=101,
         pairs=[("farmer", "shovel"), ("teacher", "book")]),
    dict(id=124, n=2, prompt="a photo of a nurse wearing blue gloves and a pilot wearing dark sunglasses", seed=101,
         pairs=[("nurse", "blue gloves"), ("pilot", "dark sunglasses")]),
    dict(id=125, n=2, prompt="a photo of a nurse wearing blue gloves and a teacher holding a book", seed=101,
         pairs=[("nurse", "blue gloves"), ("teacher", "book")]),
    dict(id=126, n=2, prompt="a photo of a pilot wearing dark sunglasses and a teacher holding a book", seed=101,
         pairs=[("pilot", "dark sunglasses"), ("teacher", "book")]),
    dict(id=127, n=2, prompt="a photo of a barista wearing a red apron and a chef wearing a white hat", seed=202,
         pairs=[("barista", "red apron"), ("chef", "white hat")]),
    dict(id=128, n=2, prompt="a photo of a barista wearing a red apron and a chef holding a pan", seed=202,
         pairs=[("barista", "red apron"), ("chef", "pan")]),
    dict(id=129, n=2, prompt="a photo of a barista wearing a red apron and a cyclist wearing a yellow helmet", seed=202,
         pairs=[("barista", "red apron"), ("cyclist", "yellow helmet")]),
    dict(id=130, n=2, prompt="a photo of a barista wearing a red apron and a farmer holding a shovel", seed=202,
         pairs=[("barista", "red apron"), ("farmer", "shovel")]),
    dict(id=131, n=2, prompt="a photo of a barista wearing a red apron and a nurse wearing blue gloves", seed=202,
         pairs=[("barista", "red apron"), ("nurse", "blue gloves")]),
    dict(id=132, n=2, prompt="a photo of a barista wearing a red apron and a pilot wearing dark sunglasses", seed=202,
         pairs=[("barista", "red apron"), ("pilot", "dark sunglasses")]),
    dict(id=133, n=2, prompt="a photo of a barista wearing a red apron and a teacher holding a book", seed=202,
         pairs=[("barista", "red apron"), ("teacher", "book")]),
    dict(id=134, n=2, prompt="a photo of a chef wearing a white hat and a cyclist wearing a yellow helmet", seed=202,
         pairs=[("chef", "white hat"), ("cyclist", "yellow helmet")]),
    dict(id=135, n=2, prompt="a photo of a chef holding a pan and a cyclist wearing a yellow helmet", seed=202,
         pairs=[("chef", "pan"), ("cyclist", "yellow helmet")]),
    dict(id=136, n=2, prompt="a photo of a chef wearing a white hat and a farmer holding a shovel", seed=202,
         pairs=[("chef", "white hat"), ("farmer", "shovel")]),
    dict(id=137, n=2, prompt="a photo of a chef holding a pan and a farmer holding a shovel", seed=202,
         pairs=[("chef", "pan"), ("farmer", "shovel")]),
    dict(id=138, n=2, prompt="a photo of a chef wearing a white hat and a nurse wearing blue gloves", seed=202,
         pairs=[("chef", "white hat"), ("nurse", "blue gloves")]),
    dict(id=139, n=2, prompt="a photo of a chef holding a pan and a nurse wearing blue gloves", seed=202,
         pairs=[("chef", "pan"), ("nurse", "blue gloves")]),
    dict(id=140, n=2, prompt="a photo of a chef wearing a white hat and a pilot wearing dark sunglasses", seed=202,
         pairs=[("chef", "white hat"), ("pilot", "dark sunglasses")]),
    dict(id=141, n=2, prompt="a photo of a chef holding a pan and a pilot wearing dark sunglasses", seed=202,
         pairs=[("chef", "pan"), ("pilot", "dark sunglasses")]),
    dict(id=142, n=3, prompt="a photo of a barista wearing a red apron, a chef wearing a white hat, and a cyclist wearing a yellow helmet", seed=101,
         pairs=[("barista", "red apron"), ("chef", "white hat"), ("cyclist", "yellow helmet")]),
    dict(id=143, n=3, prompt="a photo of a barista wearing a red apron, a chef holding a pan, and a cyclist wearing a yellow helmet", seed=101,
         pairs=[("barista", "red apron"), ("chef", "pan"), ("cyclist", "yellow helmet")]),
    dict(id=144, n=3, prompt="a photo of a barista wearing a red apron, a chef wearing a white hat, and a farmer holding a shovel", seed=101,
         pairs=[("barista", "red apron"), ("chef", "white hat"), ("farmer", "shovel")]),
    dict(id=145, n=3, prompt="a photo of a barista wearing a red apron, a chef holding a pan, and a farmer holding a shovel", seed=101,
         pairs=[("barista", "red apron"), ("chef", "pan"), ("farmer", "shovel")]),
    dict(id=146, n=3, prompt="a photo of a barista wearing a red apron, a chef wearing a white hat, and a nurse wearing blue gloves", seed=101,
         pairs=[("barista", "red apron"), ("chef", "white hat"), ("nurse", "blue gloves")]),
    dict(id=147, n=3, prompt="a photo of a barista wearing a red apron, a chef holding a pan, and a nurse wearing blue gloves", seed=101,
         pairs=[("barista", "red apron"), ("chef", "pan"), ("nurse", "blue gloves")]),
    dict(id=148, n=3, prompt="a photo of a barista wearing a red apron, a chef wearing a white hat, and a pilot wearing dark sunglasses", seed=101,
         pairs=[("barista", "red apron"), ("chef", "white hat"), ("pilot", "dark sunglasses")]),
    dict(id=149, n=3, prompt="a photo of a barista wearing a red apron, a chef holding a pan, and a pilot wearing dark sunglasses", seed=101,
         pairs=[("barista", "red apron"), ("chef", "pan"), ("pilot", "dark sunglasses")]),
    dict(id=150, n=3, prompt="a photo of a barista wearing a red apron, a chef wearing a white hat, and a teacher holding a book", seed=101,
         pairs=[("barista", "red apron"), ("chef", "white hat"), ("teacher", "book")]),
    dict(id=151, n=3, prompt="a photo of a barista wearing a red apron, a chef holding a pan, and a teacher holding a book", seed=101,
         pairs=[("barista", "red apron"), ("chef", "pan"), ("teacher", "book")]),
    dict(id=152, n=3, prompt="a photo of a barista wearing a red apron, a cyclist wearing a yellow helmet, and a farmer holding a shovel", seed=101,
         pairs=[("barista", "red apron"), ("cyclist", "yellow helmet"), ("farmer", "shovel")]),
    dict(id=153, n=3, prompt="a photo of a barista wearing a red apron, a cyclist wearing a yellow helmet, and a nurse wearing blue gloves", seed=101,
         pairs=[("barista", "red apron"), ("cyclist", "yellow helmet"), ("nurse", "blue gloves")]),
    dict(id=154, n=3, prompt="a photo of a barista wearing a red apron, a cyclist wearing a yellow helmet, and a pilot wearing dark sunglasses", seed=101,
         pairs=[("barista", "red apron"), ("cyclist", "yellow helmet"), ("pilot", "dark sunglasses")]),
    dict(id=155, n=3, prompt="a photo of a barista wearing a red apron, a cyclist wearing a yellow helmet, and a teacher holding a book", seed=101,
         pairs=[("barista", "red apron"), ("cyclist", "yellow helmet"), ("teacher", "book")]),
    dict(id=156, n=3, prompt="a photo of a barista wearing a red apron, a farmer holding a shovel, and a nurse wearing blue gloves", seed=101,
         pairs=[("barista", "red apron"), ("farmer", "shovel"), ("nurse", "blue gloves")]),
    dict(id=157, n=3, prompt="a photo of a barista wearing a red apron, a farmer holding a shovel, and a pilot wearing dark sunglasses", seed=101,
         pairs=[("barista", "red apron"), ("farmer", "shovel"), ("pilot", "dark sunglasses")]),
    dict(id=158, n=3, prompt="a photo of a barista wearing a red apron, a farmer holding a shovel, and a teacher holding a book", seed=101,
         pairs=[("barista", "red apron"), ("farmer", "shovel"), ("teacher", "book")]),
    dict(id=159, n=3, prompt="a photo of a barista wearing a red apron, a nurse wearing blue gloves, and a pilot wearing dark sunglasses", seed=101,
         pairs=[("barista", "red apron"), ("nurse", "blue gloves"), ("pilot", "dark sunglasses")]),
    dict(id=160, n=3, prompt="a photo of a barista wearing a red apron, a nurse wearing blue gloves, and a teacher holding a book", seed=101,
         pairs=[("barista", "red apron"), ("nurse", "blue gloves"), ("teacher", "book")]),
    dict(id=161, n=3, prompt="a photo of a barista wearing a red apron, a pilot wearing dark sunglasses, and a teacher holding a book", seed=101,
         pairs=[("barista", "red apron"), ("pilot", "dark sunglasses"), ("teacher", "book")]),
    dict(id=162, n=3, prompt="a photo of a chef wearing a white hat, a cyclist wearing a yellow helmet, and a farmer holding a shovel", seed=101,
         pairs=[("chef", "white hat"), ("cyclist", "yellow helmet"), ("farmer", "shovel")]),
    dict(id=163, n=3, prompt="a photo of a chef holding a pan, a cyclist wearing a yellow helmet, and a farmer holding a shovel", seed=101,
         pairs=[("chef", "pan"), ("cyclist", "yellow helmet"), ("farmer", "shovel")]),
    dict(id=164, n=3, prompt="a photo of a chef wearing a white hat, a cyclist wearing a yellow helmet, and a nurse wearing blue gloves", seed=101,
         pairs=[("chef", "white hat"), ("cyclist", "yellow helmet"), ("nurse", "blue gloves")]),
    dict(id=165, n=3, prompt="a photo of a chef holding a pan, a cyclist wearing a yellow helmet, and a nurse wearing blue gloves", seed=101,
         pairs=[("chef", "pan"), ("cyclist", "yellow helmet"), ("nurse", "blue gloves")]),
    dict(id=166, n=3, prompt="a photo of a chef wearing a white hat, a cyclist wearing a yellow helmet, and a pilot wearing dark sunglasses", seed=101,
         pairs=[("chef", "white hat"), ("cyclist", "yellow helmet"), ("pilot", "dark sunglasses")]),
    dict(id=167, n=4, prompt="a photo of a barista wearing a red apron, a chef wearing a white hat, a cyclist wearing a yellow helmet, and a farmer holding a shovel", seed=101,
         pairs=[("barista", "red apron"), ("chef", "white hat"), ("cyclist", "yellow helmet"), ("farmer", "shovel")]),
    dict(id=168, n=4, prompt="a photo of a barista wearing a red apron, a chef holding a pan, a cyclist wearing a yellow helmet, and a farmer holding a shovel", seed=101,
         pairs=[("barista", "red apron"), ("chef", "pan"), ("cyclist", "yellow helmet"), ("farmer", "shovel")]),
    dict(id=169, n=4, prompt="a photo of a barista wearing a red apron, a chef wearing a white hat, a cyclist wearing a yellow helmet, and a nurse wearing blue gloves", seed=101,
         pairs=[("barista", "red apron"), ("chef", "white hat"), ("cyclist", "yellow helmet"), ("nurse", "blue gloves")]),
    dict(id=170, n=4, prompt="a photo of a barista wearing a red apron, a chef holding a pan, a cyclist wearing a yellow helmet, and a nurse wearing blue gloves", seed=101,
         pairs=[("barista", "red apron"), ("chef", "pan"), ("cyclist", "yellow helmet"), ("nurse", "blue gloves")]),
    dict(id=171, n=4, prompt="a photo of a barista wearing a red apron, a chef wearing a white hat, a cyclist wearing a yellow helmet, and a pilot wearing dark sunglasses", seed=101,
         pairs=[("barista", "red apron"), ("chef", "white hat"), ("cyclist", "yellow helmet"), ("pilot", "dark sunglasses")]),
    dict(id=172, n=4, prompt="a photo of a barista wearing a red apron, a chef holding a pan, a cyclist wearing a yellow helmet, and a pilot wearing dark sunglasses", seed=101,
         pairs=[("barista", "red apron"), ("chef", "pan"), ("cyclist", "yellow helmet"), ("pilot", "dark sunglasses")]),
    dict(id=173, n=4, prompt="a photo of a barista wearing a red apron, a chef wearing a white hat, a cyclist wearing a yellow helmet, and a teacher holding a book", seed=101,
         pairs=[("barista", "red apron"), ("chef", "white hat"), ("cyclist", "yellow helmet"), ("teacher", "book")]),
    dict(id=174, n=4, prompt="a photo of a barista wearing a red apron, a chef holding a pan, a cyclist wearing a yellow helmet, and a teacher holding a book", seed=101,
         pairs=[("barista", "red apron"), ("chef", "pan"), ("cyclist", "yellow helmet"), ("teacher", "book")]),
    dict(id=175, n=4, prompt="a photo of a barista wearing a red apron, a chef wearing a white hat, a farmer holding a shovel, and a nurse wearing blue gloves", seed=101,
         pairs=[("barista", "red apron"), ("chef", "white hat"), ("farmer", "shovel"), ("nurse", "blue gloves")]),
    dict(id=176, n=4, prompt="a photo of a barista wearing a red apron, a chef holding a pan, a farmer holding a shovel, and a nurse wearing blue gloves", seed=101,
         pairs=[("barista", "red apron"), ("chef", "pan"), ("farmer", "shovel"), ("nurse", "blue gloves")]),
    dict(id=177, n=4, prompt="a photo of a barista wearing a red apron, a chef wearing a white hat, a farmer holding a shovel, and a pilot wearing dark sunglasses", seed=101,
         pairs=[("barista", "red apron"), ("chef", "white hat"), ("farmer", "shovel"), ("pilot", "dark sunglasses")]),
    dict(id=178, n=4, prompt="a photo of a barista wearing a red apron, a chef holding a pan, a farmer holding a shovel, and a pilot wearing dark sunglasses", seed=101,
         pairs=[("barista", "red apron"), ("chef", "pan"), ("farmer", "shovel"), ("pilot", "dark sunglasses")]),
    dict(id=179, n=4, prompt="a photo of a barista wearing a red apron, a chef wearing a white hat, a farmer holding a shovel, and a teacher holding a book", seed=101,
         pairs=[("barista", "red apron"), ("chef", "white hat"), ("farmer", "shovel"), ("teacher", "book")]),
    dict(id=180, n=4, prompt="a photo of a barista wearing a red apron, a chef holding a pan, a farmer holding a shovel, and a teacher holding a book", seed=101,
         pairs=[("barista", "red apron"), ("chef", "pan"), ("farmer", "shovel"), ("teacher", "book")]),
    dict(id=181, n=4, prompt="a photo of a barista wearing a red apron, a chef wearing a white hat, a nurse wearing blue gloves, and a pilot wearing dark sunglasses", seed=101,
         pairs=[("barista", "red apron"), ("chef", "white hat"), ("nurse", "blue gloves"), ("pilot", "dark sunglasses")]),
    dict(id=182, n=4, prompt="a photo of a barista wearing a red apron, a chef holding a pan, a nurse wearing blue gloves, and a pilot wearing dark sunglasses", seed=101,
         pairs=[("barista", "red apron"), ("chef", "pan"), ("nurse", "blue gloves"), ("pilot", "dark sunglasses")]),
    dict(id=183, n=4, prompt="a photo of a barista wearing a red apron, a chef wearing a white hat, a nurse wearing blue gloves, and a teacher holding a book", seed=101,
         pairs=[("barista", "red apron"), ("chef", "white hat"), ("nurse", "blue gloves"), ("teacher", "book")]),
    dict(id=184, n=4, prompt="a photo of a barista wearing a red apron, a chef holding a pan, a nurse wearing blue gloves, and a teacher holding a book", seed=101,
         pairs=[("barista", "red apron"), ("chef", "pan"), ("nurse", "blue gloves"), ("teacher", "book")]),
    # 2026-07-25 n=4 BACKFILL batch (ids 200-227): the pinned-seed growth batch above
    # hit only 22.2% detection at n=4 (4/18) -- these 28 NEW combos (never used by ids
    # 167-184 above) each carry a `seeds` LIST for a bounded retry instead of one pinned
    # seed, reintroducing the base script's retry reliability for just this stratum. See
    # build_growth_specs.py's build_n4_backfill_specs() docstring for why reusing seed
    # VALUES here is safe without per-combo bookkeeping. Only actually generated by
    # generate_anchor_images_sdxl.py; kept in sync here per the same convention as the
    # growth batches above.
    dict(id=200, n=4, prompt="a photo of a barista wearing a red apron, a chef wearing a white hat, a pilot wearing dark sunglasses, and a teacher holding a book", seeds=[101, 202, 303],
         pairs=[("barista", "red apron"), ("chef", "white hat"), ("pilot", "dark sunglasses"), ("teacher", "book")]),
    dict(id=201, n=4, prompt="a photo of a barista wearing a red apron, a chef holding a pan, a pilot wearing dark sunglasses, and a teacher holding a book", seeds=[101, 202, 303],
         pairs=[("barista", "red apron"), ("chef", "pan"), ("pilot", "dark sunglasses"), ("teacher", "book")]),
    dict(id=202, n=4, prompt="a photo of a barista wearing a red apron, a cyclist wearing a yellow helmet, a farmer holding a shovel, and a nurse wearing blue gloves", seeds=[101, 202, 303],
         pairs=[("barista", "red apron"), ("cyclist", "yellow helmet"), ("farmer", "shovel"), ("nurse", "blue gloves")]),
    dict(id=203, n=4, prompt="a photo of a barista wearing a red apron, a cyclist wearing a yellow helmet, a farmer holding a shovel, and a pilot wearing dark sunglasses", seeds=[101, 202, 303],
         pairs=[("barista", "red apron"), ("cyclist", "yellow helmet"), ("farmer", "shovel"), ("pilot", "dark sunglasses")]),
    dict(id=204, n=4, prompt="a photo of a barista wearing a red apron, a cyclist wearing a yellow helmet, a farmer holding a shovel, and a teacher holding a book", seeds=[101, 202, 303],
         pairs=[("barista", "red apron"), ("cyclist", "yellow helmet"), ("farmer", "shovel"), ("teacher", "book")]),
    dict(id=205, n=4, prompt="a photo of a barista wearing a red apron, a cyclist wearing a yellow helmet, a nurse wearing blue gloves, and a pilot wearing dark sunglasses", seeds=[101, 202, 303],
         pairs=[("barista", "red apron"), ("cyclist", "yellow helmet"), ("nurse", "blue gloves"), ("pilot", "dark sunglasses")]),
    dict(id=206, n=4, prompt="a photo of a barista wearing a red apron, a cyclist wearing a yellow helmet, a nurse wearing blue gloves, and a teacher holding a book", seeds=[101, 202, 303],
         pairs=[("barista", "red apron"), ("cyclist", "yellow helmet"), ("nurse", "blue gloves"), ("teacher", "book")]),
    dict(id=207, n=4, prompt="a photo of a barista wearing a red apron, a cyclist wearing a yellow helmet, a pilot wearing dark sunglasses, and a teacher holding a book", seeds=[101, 202, 303],
         pairs=[("barista", "red apron"), ("cyclist", "yellow helmet"), ("pilot", "dark sunglasses"), ("teacher", "book")]),
    dict(id=208, n=4, prompt="a photo of a barista wearing a red apron, a farmer holding a shovel, a nurse wearing blue gloves, and a pilot wearing dark sunglasses", seeds=[101, 202, 303],
         pairs=[("barista", "red apron"), ("farmer", "shovel"), ("nurse", "blue gloves"), ("pilot", "dark sunglasses")]),
    dict(id=209, n=4, prompt="a photo of a barista wearing a red apron, a farmer holding a shovel, a nurse wearing blue gloves, and a teacher holding a book", seeds=[101, 202, 303],
         pairs=[("barista", "red apron"), ("farmer", "shovel"), ("nurse", "blue gloves"), ("teacher", "book")]),
    dict(id=210, n=4, prompt="a photo of a barista wearing a red apron, a farmer holding a shovel, a pilot wearing dark sunglasses, and a teacher holding a book", seeds=[101, 202, 303],
         pairs=[("barista", "red apron"), ("farmer", "shovel"), ("pilot", "dark sunglasses"), ("teacher", "book")]),
    dict(id=211, n=4, prompt="a photo of a barista wearing a red apron, a nurse wearing blue gloves, a pilot wearing dark sunglasses, and a teacher holding a book", seeds=[101, 202, 303],
         pairs=[("barista", "red apron"), ("nurse", "blue gloves"), ("pilot", "dark sunglasses"), ("teacher", "book")]),
    dict(id=212, n=4, prompt="a photo of a chef wearing a white hat, a cyclist wearing a yellow helmet, a farmer holding a shovel, and a nurse wearing blue gloves", seeds=[101, 202, 303],
         pairs=[("chef", "white hat"), ("cyclist", "yellow helmet"), ("farmer", "shovel"), ("nurse", "blue gloves")]),
    dict(id=213, n=4, prompt="a photo of a chef holding a pan, a cyclist wearing a yellow helmet, a farmer holding a shovel, and a nurse wearing blue gloves", seeds=[101, 202, 303],
         pairs=[("chef", "pan"), ("cyclist", "yellow helmet"), ("farmer", "shovel"), ("nurse", "blue gloves")]),
    dict(id=214, n=4, prompt="a photo of a chef wearing a white hat, a cyclist wearing a yellow helmet, a farmer holding a shovel, and a pilot wearing dark sunglasses", seeds=[101, 202, 303],
         pairs=[("chef", "white hat"), ("cyclist", "yellow helmet"), ("farmer", "shovel"), ("pilot", "dark sunglasses")]),
    dict(id=215, n=4, prompt="a photo of a chef holding a pan, a cyclist wearing a yellow helmet, a farmer holding a shovel, and a pilot wearing dark sunglasses", seeds=[101, 202, 303],
         pairs=[("chef", "pan"), ("cyclist", "yellow helmet"), ("farmer", "shovel"), ("pilot", "dark sunglasses")]),
    dict(id=216, n=4, prompt="a photo of a chef wearing a white hat, a cyclist wearing a yellow helmet, a farmer holding a shovel, and a teacher holding a book", seeds=[101, 202, 303],
         pairs=[("chef", "white hat"), ("cyclist", "yellow helmet"), ("farmer", "shovel"), ("teacher", "book")]),
    dict(id=217, n=4, prompt="a photo of a chef holding a pan, a cyclist wearing a yellow helmet, a farmer holding a shovel, and a teacher holding a book", seeds=[101, 202, 303],
         pairs=[("chef", "pan"), ("cyclist", "yellow helmet"), ("farmer", "shovel"), ("teacher", "book")]),
    dict(id=218, n=4, prompt="a photo of a chef wearing a white hat, a cyclist wearing a yellow helmet, a nurse wearing blue gloves, and a pilot wearing dark sunglasses", seeds=[101, 202, 303],
         pairs=[("chef", "white hat"), ("cyclist", "yellow helmet"), ("nurse", "blue gloves"), ("pilot", "dark sunglasses")]),
    dict(id=219, n=4, prompt="a photo of a chef holding a pan, a cyclist wearing a yellow helmet, a nurse wearing blue gloves, and a pilot wearing dark sunglasses", seeds=[101, 202, 303],
         pairs=[("chef", "pan"), ("cyclist", "yellow helmet"), ("nurse", "blue gloves"), ("pilot", "dark sunglasses")]),
    dict(id=220, n=4, prompt="a photo of a chef wearing a white hat, a cyclist wearing a yellow helmet, a nurse wearing blue gloves, and a teacher holding a book", seeds=[101, 202, 303],
         pairs=[("chef", "white hat"), ("cyclist", "yellow helmet"), ("nurse", "blue gloves"), ("teacher", "book")]),
    dict(id=221, n=4, prompt="a photo of a chef holding a pan, a cyclist wearing a yellow helmet, a nurse wearing blue gloves, and a teacher holding a book", seeds=[101, 202, 303],
         pairs=[("chef", "pan"), ("cyclist", "yellow helmet"), ("nurse", "blue gloves"), ("teacher", "book")]),
    dict(id=222, n=4, prompt="a photo of a chef wearing a white hat, a cyclist wearing a yellow helmet, a pilot wearing dark sunglasses, and a teacher holding a book", seeds=[101, 202, 303],
         pairs=[("chef", "white hat"), ("cyclist", "yellow helmet"), ("pilot", "dark sunglasses"), ("teacher", "book")]),
    dict(id=223, n=4, prompt="a photo of a chef holding a pan, a cyclist wearing a yellow helmet, a pilot wearing dark sunglasses, and a teacher holding a book", seeds=[101, 202, 303],
         pairs=[("chef", "pan"), ("cyclist", "yellow helmet"), ("pilot", "dark sunglasses"), ("teacher", "book")]),
    dict(id=224, n=4, prompt="a photo of a chef wearing a white hat, a farmer holding a shovel, a nurse wearing blue gloves, and a pilot wearing dark sunglasses", seeds=[101, 202, 303],
         pairs=[("chef", "white hat"), ("farmer", "shovel"), ("nurse", "blue gloves"), ("pilot", "dark sunglasses")]),
    dict(id=225, n=4, prompt="a photo of a chef holding a pan, a farmer holding a shovel, a nurse wearing blue gloves, and a pilot wearing dark sunglasses", seeds=[101, 202, 303],
         pairs=[("chef", "pan"), ("farmer", "shovel"), ("nurse", "blue gloves"), ("pilot", "dark sunglasses")]),
    dict(id=226, n=4, prompt="a photo of a chef wearing a white hat, a farmer holding a shovel, a nurse wearing blue gloves, and a teacher holding a book", seeds=[101, 202, 303],
         pairs=[("chef", "white hat"), ("farmer", "shovel"), ("nurse", "blue gloves"), ("teacher", "book")]),
    dict(id=227, n=4, prompt="a photo of a chef holding a pan, a farmer holding a shovel, a nurse wearing blue gloves, and a teacher holding a book", seeds=[101, 202, 303],
         pairs=[("chef", "pan"), ("farmer", "shovel"), ("nurse", "blue gloves"), ("teacher", "book")]),
]

# =============================================================================
# Attention capture -- duplicated verbatim from generate_chains.py (which duplicated it from
# pilot/spatial_semantic_alignment.py). Keep in sync.
# =============================================================================

class AttentionRecord:
    __slots__ = ("tensor", "layer_name", "spatial_dim", "heads")

    def __init__(self, tensor: torch.Tensor, layer_name: str, spatial_dim: int, heads: int = 1):
        self.tensor = tensor
        self.layer_name = layer_name
        self.spatial_dim = spatial_dim
        self.heads = heads


class AttentionStore:
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


class CustomAttnProcessor:
    def __init__(self, store: AttentionStore, is_cross_attention: bool, layer_name: str):
        self.store = store
        self.is_cross_attention = is_cross_attention
        self.layer_name = layer_name

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, temb=None, scale: float = 1.0):
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
        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross is not None:
            encoder_hidden_states = attn.norm_cross(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        if self.is_cross_attention:
            spatial_dim = attention_probs.shape[1]
            self.store.add_attention(attention_probs, self.layer_name, spatial_dim, attn.heads)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


class AttentionCapture:
    def __init__(self):
        self.attn_store = AttentionStore()

    def hook_pipeline(self, pipeline):
        self.attn_store.reset()
        processors = {}
        for name in pipeline.unet.attn_processors.keys():
            is_cross = "attn2" in name
            processors[name] = CustomAttnProcessor(self.attn_store, is_cross, layer_name=name)
        pipeline.unet.set_attn_processor(processors)

    def unhook_pipeline(self, pipeline):
        from diffusers.models.attention_processor import AttnProcessor2_0
        pipeline.unet.set_attn_processor(AttnProcessor2_0())

    def phase_b_cross_attention_map(self, target_token_index: int,
                                     target_resolution: Tuple[int, int] = (IMG_SIZE, IMG_SIZE),
                                     max_steps: int = MAX_STEPS,
                                     cond_index: Optional[int] = None) -> np.ndarray:
        if not self.attn_store.step_store:
            raise ValueError("Attention store is empty. Did you call hook_pipeline() before generating?")
        per_res_accum: Dict[int, np.ndarray] = {}
        per_res_weight: Dict[int, float] = {}
        for step_idx, records in self.attn_store.step_store.items():
            if step_idx >= max_steps:
                continue
            for record in records:
                spatial_dim = record.spatial_dim
                seq_len = record.tensor.shape[2]
                native_side = int(math.sqrt(spatial_dim))
                if native_side * native_side != spatial_dim or target_token_index >= seq_len:
                    continue
                tensor = record.tensor
                if cond_index is not None:
                    start = cond_index * record.heads
                    tensor = tensor[start:start + record.heads]
                token_attn = tensor[:, :, target_token_index].mean(dim=0)
                attn_2d = token_attn.view(native_side, native_side).unsqueeze(0).unsqueeze(0)
                attn_up = F.interpolate(attn_2d.float(), size=target_resolution,
                                         mode="bilinear", align_corners=False).squeeze().numpy()
                per_res_accum.setdefault(native_side, np.zeros(target_resolution, dtype=np.float32))
                per_res_weight.setdefault(native_side, 0.0)
                per_res_accum[native_side] += attn_up
                per_res_weight[native_side] += 1.0
        if not per_res_accum:
            return np.zeros(target_resolution, dtype=np.float32)
        per_res_maps = {side: per_res_accum[side] / per_res_weight[side] for side in per_res_accum}
        composite = np.zeros(target_resolution, dtype=np.float32)
        total_weight = 0.0
        for side, layer_map in per_res_maps.items():
            w = float(side * side)
            composite += w * layer_map
            total_weight += w
        return composite / total_weight if total_weight > 0 else composite


# =============================================================================
# Pure logic -- INLINE COPIES of anchor_common.py (see module docstring). Keep in sync.
# =============================================================================

def mean_mass_in_box(attn_map: np.ndarray, box) -> float:
    h, w = attn_map.shape[:2]
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float(attn_map[y0:y1, x0:x1].mean())


def predicted_owner_from_attention(attn_map: np.ndarray, subject_boxes: Dict[str, list]):
    if not subject_boxes:
        raise ValueError("subject_boxes is empty; nothing to attribute the attention to")
    scores = {s: mean_mass_in_box(attn_map, b) for s, b in subject_boxes.items()}
    owner = max(scores, key=lambda s: scores[s])
    return owner, scores


def build_attribute_entry(attribute, intended_subject, predicted_owner, model_scores) -> dict:
    return dict(attribute=attribute, intended_subject=intended_subject,
                predicted_owner=predicted_owner, model_scores=model_scores)


def build_manifest_entry(prompt_id, n, prompt, subjects, seed, detected,
                         num_people_detected, image_path, attributes) -> dict:
    return dict(prompt_id=prompt_id, n=n, prompt=prompt, subjects=subjects, seed=seed,
                detected=detected, num_people_detected=num_people_detected,
                image_path=image_path, attributes=attributes)


# =============================================================================
# Model helpers -- duplicated from generate_chains.py (inpainting helpers omitted).
# =============================================================================

def token_indices(tokenizer, prompt: str, phrase: str) -> List[int]:
    ids = tokenizer(prompt, padding="max_length", max_length=77, truncation=True).input_ids
    target = tokenizer(phrase, add_special_tokens=False).input_ids
    if not target:
        raise ValueError(f"phrase {phrase!r} tokenized to nothing")
    for i in range(len(ids) - len(target) + 1):
        if ids[i:i + len(target)] == target:
            return list(range(i, i + len(target)))
    raise ValueError(f"phrase {phrase!r} not found in prompt tokens: {prompt!r}")


def _load_pipeline(candidates, cls):
    for repo in candidates:
        try:
            pipe = cls.from_pretrained(repo, torch_dtype=DTYPE, safety_checker=None,
                                        requires_safety_checker=False).to(DEVICE)
            print(f"loaded {cls.__name__}: {repo}")
            return pipe
        except Exception as e:
            print(f"  {repo} unavailable ({type(e).__name__}: {e})")
    raise RuntimeError(f"No mirror loaded for {cls.__name__} from {candidates}")


class Models:
    def __init__(self, txt2img, detector, clip_model, clip_proc):
        self.txt2img = txt2img
        self.detector = detector
        self.clip_model = clip_model
        self.clip_proc = clip_proc


def load_all_models() -> Models:
    from diffusers import StableDiffusionPipeline
    from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
    from transformers import CLIPProcessor, CLIPModel

    txt2img = _load_pipeline(SD15_TXT2IMG_CANDIDATES, StableDiffusionPipeline)
    txt2img.set_progress_bar_config(disable=True)
    detector = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT).to(DEVICE).eval()
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True).to(DEVICE).eval()
    clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return Models(txt2img, detector, clip_model, clip_proc)


@torch.no_grad()
def person_boxes(models: Models, image: Image.Image, max_people: int,
                  score_thresh: float = DETECTION_SCORE_THRESH) -> List[List[float]]:
    t = TF.to_tensor(image).to(DEVICE)
    out = models.detector([t])[0]
    keep = (out["labels"] == COCO_PERSON) & (out["scores"] >= score_thresh)
    boxes, scores = out["boxes"][keep], out["scores"][keep]
    if boxes.shape[0] == 0:
        return []
    order = scores.argsort(descending=True)[:max_people]
    return boxes[order].cpu().numpy().tolist()


@torch.no_grad()
def assign_subjects(models: Models, image: Image.Image, boxes: List[List[float]],
                     subject_names: List[str]) -> Dict[str, int]:
    from scipy.optimize import linear_sum_assignment

    crops = [image.crop(tuple(b)) for b in boxes]
    inp = models.clip_proc(text=[f"a photo of a {s}" for s in subject_names],
                            images=crops, return_tensors="pt", padding=True).to(DEVICE)
    sim = models.clip_model(**inp).logits_per_image.softmax(-1).cpu().numpy()
    r, c = linear_sum_assignment(-sim)
    return {subject_names[j]: int(i) for i, j in zip(r, c)}


def generate_and_score(spec: dict, capture: AttentionCapture, models: Models) -> dict:
    """Generate ONE compositional image, retrying seeds until exactly n people are detected;
    then attribute each attribute's early-window attention to a subject box. Always returns a
    manifest entry -- detection outcome (pass or fail) is recorded, never silently skipped."""
    prompt_id, n, prompt = spec["id"], spec["n"], spec["prompt"]
    subjects = [s for s, _ in spec["pairs"]]

    chosen_seed = CANDIDATE_SEEDS[-1]
    boxes: List[List[float]] = []
    image: Optional[Image.Image] = None
    for seed in CANDIDATE_SEEDS:
        capture.hook_pipeline(models.txt2img)

        def _cb(pipe, i, t, kw):
            capture.attn_store.step()
            return kw

        g = torch.Generator(DEVICE).manual_seed(seed)
        candidate = models.txt2img(prompt, num_inference_steps=NUM_INFERENCE_STEPS,
                                    guidance_scale=7.5, generator=g,
                                    callback_on_step_end=_cb).images[0]
        found = person_boxes(models, candidate, max_people=n)
        chosen_seed, image, boxes = seed, candidate, found
        if len(found) == n:
            break
        capture.unhook_pipeline(models.txt2img)
        print(f"  p{prompt_id} seed={seed}: detected {len(found)}/{n}, retrying")

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    image_path = str(IMAGES_DIR / f"p{prompt_id}.png")
    image.save(image_path)

    detected = len(boxes) == n
    if not detected:
        capture.unhook_pipeline(models.txt2img)
        print(f"  p{prompt_id}: DETECTION FAIL after all seeds -- found {len(boxes)}, expected {n}")
        return build_manifest_entry(prompt_id, n, prompt, subjects, chosen_seed, detected=False,
                                    num_people_detected=len(boxes), image_path=image_path,
                                    attributes=[])

    assign = assign_subjects(models, image, boxes, subjects)
    subject_boxes = {s: boxes[assign[s]] for s in subjects}

    attributes_out: List[dict] = []
    for subject, attribute in spec["pairs"]:
        idxs = token_indices(models.txt2img.tokenizer, prompt, attribute)
        maps = [capture.phase_b_cross_attention_map(target_token_index=i, cond_index=1) for i in idxs]
        attn_map = np.mean(maps, axis=0)
        owner, scores = predicted_owner_from_attention(attn_map, subject_boxes)
        attributes_out.append(build_attribute_entry(attribute, subject, owner, scores))
    capture.unhook_pipeline(models.txt2img)

    print(f"  p{prompt_id}: OK seed={chosen_seed} n={n} "
          + ", ".join(f"{a['attribute']}->{a['predicted_owner']}" for a in attributes_out))
    return build_manifest_entry(prompt_id, n, prompt, subjects, chosen_seed, detected=True,
                                num_people_detected=len(boxes), image_path=image_path,
                                attributes=attributes_out)


# The SD1.5 set is frozen (see CLAUDE.md: "the SD1.5 manifest, images ... stay as-is").
# ANCHOR_PROMPTS above carries the 2026-07-25 growth batch (ids 100-184) too, purely so the
# drift guard (test_anchor_common.py) can keep asserting it matches prompt_specs.json
# byte-for-byte -- but that batch is only actually GENERATED by
# generate_anchor_images_sdxl.py (SDXL is canonical going forward). This filter is what
# makes "frozen" true by construction: rerunning this script can never spend SD1.5 GPU time
# regenerating the growth batch, even if this file is executed from scratch.
FROZEN_PROMPT_IDS = set(range(0, 24))


def main():
    prompts = [p for p in ANCHOR_PROMPTS if p["id"] in FROZEN_PROMPT_IDS]
    print(f"device={DEVICE} dtype={DTYPE} seeds={CANDIDATE_SEEDS} prompts={len(prompts)}")
    capture = AttentionCapture()
    print("\nLoading models (SD1.5 txt2img, Mask R-CNN, CLIP)...")
    models = load_all_models()

    manifest = {"candidate_seeds": CANDIDATE_SEEDS, "img_size": IMG_SIZE,
                "num_inference_steps": NUM_INFERENCE_STEPS,
                "early_window_fraction": EARLY_WINDOW_FRACTION, "images": []}
    t0 = time.time()
    for spec in prompts:
        print(f"\n=== p{spec['id']} (n={spec['n']}): {spec['prompt'][:70]}... ===")
        try:
            entry = generate_and_score(spec, capture, models)
        except Exception as e:
            print(f"  p{spec['id']}: ERROR {type(e).__name__}: {e}")
            traceback.print_exc()
            entry = build_manifest_entry(spec["id"], spec["n"], spec["prompt"],
                                         [s for s, _ in spec["pairs"]], CANDIDATE_SEEDS[-1],
                                         detected=False, num_people_detected=0, image_path="",
                                         attributes=[])
        manifest["images"].append(entry)
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(MANIFEST_PATH, "w") as f:
            json.dump(manifest, f, indent=2)  # incremental save, survives a Kaggle timeout

    detected_count = sum(1 for e in manifest["images"] if e["detected"])
    print(f"\nDetected {detected_count}/{len(manifest['images'])} images in {time.time() - t0:.1f}s")
    print(f"Manifest written to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
