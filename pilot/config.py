"""Config schema for the pilot (Track 0: Causal Relevance) pipeline.

Defaults here match the pipeline's original hardcoded values exactly; see
config/default.yaml for the same values expressed as YAML. Every CLI script in this
directory takes --config to point at a different YAML file, with per-flag overrides
layered on top.
"""

from dataclasses import dataclass, field
from pathlib import Path

from common.config import load_yaml

DEFAULT_CONFIG_PATH = Path(__file__).parent / "config" / "default.yaml"

CONDITIONS = ("real", "shuffled", "substituted")
CONDITION_PAIRS = (("real", "shuffled"), ("real", "substituted"), ("shuffled", "substituted"))


@dataclass
class PathsConfig:
    sbs_eval_csv: Path = Path("data/evaluation_sbs_results.csv")
    item_indexes: Path = Path("data/pilot_item_indexes.txt")
    conditions_dir: Path = Path("data/conditions")
    pilot_prompts_csv: Path = Path("data/pilot_prompts.csv")
    causal_relevance_results_csv: Path = Path("data/causal_relevance_results.csv")


@dataclass
class SelectionConfig:
    n: int = 10
    seed: int = 42


@dataclass
class DecisionConfig:
    """Go/no-go thresholds for score_pilot.py's read of the pilot results."""

    shuffled_persist_min: float = 0.5
    substituted_persist_max: float = 0.3
    real_substituted_gap_min: float = 0.5


@dataclass
class JudgeConfig:
    model: str = "gemini-2.5-flash"
    temperature: float = 0.0
    max_retries: int = 6
    openrouter_model: str = "google/gemini-2.5-flash"
    openrouter_max_retries: int = 3
    openrouter_timeout: float = 120.0


@dataclass
class PilotConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)

    @classmethod
    def from_yaml(cls, path: Path = DEFAULT_CONFIG_PATH) -> "PilotConfig":
        return load_yaml(path, cls)
