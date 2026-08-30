"""Tests for config.py: the pilot's YAML -> dataclass config loader.
Run from inside pilot/:  python -m pytest tests/"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import DEFAULT_CONFIG_PATH, PilotConfig


def test_defaults_load_from_the_shipped_default_yaml():
    cfg = PilotConfig.from_yaml(DEFAULT_CONFIG_PATH)
    assert cfg.selection.n == 10
    assert cfg.selection.seed == 42
    assert cfg.judge.model == "gemini-2.5-flash"
    assert cfg.paths.conditions_dir == Path("data/conditions")


def test_n100_variant_overrides_paths_and_selection():
    n100_path = DEFAULT_CONFIG_PATH.parent / "n100.yaml"
    cfg = PilotConfig.from_yaml(n100_path)
    assert cfg.selection.n == 100
    assert cfg.paths.conditions_dir == Path("data/conditions_100")
    assert cfg.paths.causal_relevance_results_csv == Path("data/causal_relevance_results_100.csv")


def test_missing_keys_fall_back_to_dataclass_defaults(tmp_path):
    partial = tmp_path / "partial.yaml"
    partial.write_text("selection:\n  seed: 7\n", encoding="utf-8")
    cfg = PilotConfig.from_yaml(partial)
    assert cfg.selection.seed == 7
    assert cfg.selection.n == 10  # untouched key keeps the dataclass default
    assert cfg.judge.model == "gemini-2.5-flash"


def test_decision_thresholds_match_original_hardcoded_values():
    cfg = PilotConfig.from_yaml(DEFAULT_CONFIG_PATH)
    assert cfg.decision.shuffled_persist_min == 0.5
    assert cfg.decision.substituted_persist_max == 0.3
    assert cfg.decision.real_substituted_gap_min == 0.5
