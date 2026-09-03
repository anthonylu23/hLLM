"""Load strict planner input files."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import yaml
from pydantic import TypeAdapter

from hllm_control.models import LinkProfile, PlannerSettings, WorkerProfile, WorkloadProfile


class ConfigurationError(ValueError):
    """Raised when a YAML planner input is malformed."""


def load_yaml(path: Path) -> object:
    try:
        return cast(object, yaml.safe_load(path.read_text(encoding="utf-8")))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise ConfigurationError(f"cannot read valid YAML from {path}") from error


def _unwrap(raw: object, key: str) -> object:
    if not isinstance(raw, dict):
        return raw
    unknown_mapping = cast(dict[object, object], raw)
    mapping = {
        item_key: item_value
        for item_key, item_value in unknown_mapping.items()
        if isinstance(item_key, str)
    }
    if len(mapping) != len(unknown_mapping):
        raise ConfigurationError("configuration mapping keys must be strings")
    fallback = cast(object, raw)
    return mapping.get(key, fallback)


def load_workers(path: Path) -> tuple[WorkerProfile, ...]:
    raw = load_yaml(path)
    value = _unwrap(raw, "workers")
    try:
        workers = TypeAdapter(tuple[WorkerProfile, ...]).validate_python(value)
    except ValueError as error:
        raise ConfigurationError(f"invalid worker profiles in {path}: {error}") from error
    if len(workers) != 2:
        raise ConfigurationError("Milestone 0 requires exactly two worker profiles")
    if workers[0].worker_id == workers[1].worker_id:
        raise ConfigurationError("worker IDs must be unique")
    return workers


def load_links(path: Path) -> tuple[LinkProfile, ...]:
    raw = load_yaml(path)
    value = _unwrap(raw, "links")
    try:
        return TypeAdapter(tuple[LinkProfile, ...]).validate_python(value)
    except ValueError as error:
        raise ConfigurationError(f"invalid link profiles in {path}: {error}") from error


def load_workload(path: Path) -> WorkloadProfile:
    raw = load_yaml(path)
    value = _unwrap(raw, "workload")
    try:
        return WorkloadProfile.model_validate(value)
    except ValueError as error:
        raise ConfigurationError(f"invalid workload profile in {path}: {error}") from error


def load_settings(path: Path | None) -> PlannerSettings:
    if path is None:
        return PlannerSettings()
    raw = load_yaml(path)
    value = _unwrap(raw, "planner")
    try:
        return PlannerSettings.model_validate(value)
    except ValueError as error:
        raise ConfigurationError(f"invalid planner settings in {path}: {error}") from error
