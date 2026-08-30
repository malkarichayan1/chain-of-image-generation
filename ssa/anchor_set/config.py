"""Config schema for ssa/anchor_set (Track A: Single-Image Audit & MMDiT Attention).

Defaults here match the pipeline's original hardcoded values exactly; see
config/default.yaml for the same values expressed as YAML. The exp1-10 scripts and other
"pure logic, run locally" analysis scripts in this directory take --config to point at a
different YAML file where noted, with per-flag overrides layered on top.

generate_anchor_images.py, generate_anchor_images_flux.py, generate_anchor_images_sdxl.py,
taxonomy_capture_flux.py, vqa_score_flux.py, and vqa_score_sdxl.py are self-contained
Kaggle/Colab kernels by design (pushable as one script, no local imports) and intentionally
do NOT import this module -- see each file's own CONFIG block near its top.
"""

from dataclasses import dataclass, field
from pathlib import Path

from common.config import load_yaml

DEFAULT_CONFIG_PATH = Path(__file__).parent / "config" / "default.yaml"


@dataclass
class PathsConfig:
    artifacts_dir: Path = Path("artifacts")


@dataclass
class ModelsConfig:
    clip_model: str = "openai/clip-vit-base-patch32"
    clipseg_model: str = "CIDAS/clipseg-rd64-refined"
    blip_vqa_model: str = "Salesforce/blip-vqa-base"
    owlvit_model: str = "google/owlvit-base-patch32"
    sdxl_model: str = "stabilityai/stable-diffusion-xl-base-1.0"
    flux_model: str = "black-forest-labs/FLUX.1-dev"
    sd15_candidates: tuple = (
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "sd-legacy/stable-diffusion-v1-5",
    )


@dataclass
class SeedsConfig:
    candidate_seeds: tuple = (42, 7, 1234, 2024)
    mcnemar_seed: int = 42
    sweep_n_seeds: int = 200
    dummy_seed: int = 0
    label_shuffle_seed: int = 20260723


@dataclass
class GenerationConfig:
    num_inference_steps_sd: int = 30
    num_inference_steps_flux: int = 25
    early_window_fraction: float = 0.5
    img_size_sd: int = 512
    img_size_xl: int = 1024
    detection_score_thresh: float = 0.7


@dataclass
class SignificanceConfig:
    alpha: float = 0.05
    repro_threshold: float = 0.05
    kappa_hold_min: float = 0.6
    kappa_caveat_min: float = 0.4
    score_tolerance: float = 1e-3


@dataclass
class TaxonomyConfig:
    expected_n_steps: int = 25
    exp19_top_k: int = 10


@dataclass
class Config:
    paths: PathsConfig = field(default_factory=PathsConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    seeds: SeedsConfig = field(default_factory=SeedsConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    significance: SignificanceConfig = field(default_factory=SignificanceConfig)
    taxonomy: TaxonomyConfig = field(default_factory=TaxonomyConfig)

    @classmethod
    def from_yaml(cls, path: Path = DEFAULT_CONFIG_PATH) -> "Config":
        return load_yaml(path, cls)
