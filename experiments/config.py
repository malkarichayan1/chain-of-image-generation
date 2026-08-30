"""Config schema for the experiments/ (Track 2: Sequential Chain Audit / CoIG) pipeline.

Defaults here match the pipeline's original hardcoded values exactly; see
config/default.yaml for the same values expressed as YAML. Every "pure numpy, run
locally" script in this directory (segment_cache.py, score_chains.py,
calibrate_threshold.py, analyze_results.py, and the Part C/D one-off checks) takes
--config to point at a different YAML file, with per-flag overrides layered on top.

generate_chains.py and run_chain_experiment.py are single-file Kaggle kernels by design
(pushable as one script, no local imports) and intentionally do NOT import this module --
see their own module docstrings and the CONFIG block near the top of generate_chains.py.
"""

from dataclasses import dataclass, field
from pathlib import Path

from common.config import load_yaml

DEFAULT_CONFIG_PATH = Path(__file__).parent / "config" / "default.yaml"


@dataclass
class PathsConfig:
    artifacts_dir: Path = Path("artifacts")
    manifest_path: Path = Path("artifacts/manifest.json")
    segmentation_cache_dir: Path = Path("artifacts/segmentation_cache")
    results_csv: Path = Path("artifacts/chain_experiment_results.csv")
    pilot_conditions_dir: Path = Path("../pilot/data/conditions")
    pilot_sbs_eval_csv: Path = Path("../pilot/data/evaluation_sbs_results.csv")
    pilot_causal_relevance_csv: Path = Path("../pilot/data/causal_relevance_results.csv")


@dataclass
class ModelsConfig:
    clipseg_model: str = "CIDAS/clipseg-rd64-refined"
    owlvit_model: str = "google/owlvit-base-patch32"


@dataclass
class ScoringConfig:
    threshold: float = 0.85  # calibrated by calibrate_threshold.py -- see score_chains.py
    threshold_pct: float = 0.20
    seed: int = 42
    calibration_prompt_ids: tuple = (3, 4, 6, 8)
    held_out_prompt_ids: tuple = (0, 1, 2, 5, 7)
    calibration_sweep_start: float = 0.05
    calibration_sweep_stop: float = 0.96
    calibration_sweep_step: float = 0.05


@dataclass
class AnalysisConfig:
    alpha: float = 0.05
    topk_sweep: tuple = (0.05, 0.10, 0.20, 0.30, 0.40)
    rng_sweep_n_seeds: int = 200
    rng_sweep_robustness_threshold: float = 0.95


@dataclass
class Config:
    paths: PathsConfig = field(default_factory=PathsConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)

    @classmethod
    def from_yaml(cls, path: Path = DEFAULT_CONFIG_PATH) -> "Config":
        return load_yaml(path, cls)
