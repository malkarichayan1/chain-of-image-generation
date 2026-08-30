"""Shared YAML -> dataclass config loading, used by each track's own config.py.

Each track defines its own (possibly nested) dataclass schema with defaults that match
its current hardcoded behavior; `load_yaml` fills that schema from a YAML file, falling
back to the dataclass's defaults for any key the file omits.
"""

import dataclasses
from pathlib import Path
from typing import Type, TypeVar, get_args, get_origin, get_type_hints

import yaml

T = TypeVar("T")


def _coerce(field_type, value):
    if dataclasses.is_dataclass(field_type) and isinstance(value, dict):
        return _build(field_type, value)
    if field_type is Path and isinstance(value, str):
        return Path(value)
    if get_origin(field_type) is tuple and isinstance(value, (list, tuple)):
        return tuple(value)
    return value


def _build(cls: Type[T], data: dict) -> T:
    hints = get_type_hints(cls)
    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name not in data:
            continue
        kwargs[f.name] = _coerce(hints[f.name], data[f.name])
    return cls(**kwargs)


def load_yaml(path: Path, cls: Type[T]) -> T:
    """Construct *cls* (a dataclass, possibly nested) from the YAML file at *path*.

    Keys the file omits keep the dataclass's own defaults; unknown keys are ignored.
    """
    data = yaml.safe_load(Path(path).read_text()) or {}
    return _build(cls, data)
