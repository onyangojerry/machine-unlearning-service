from __future__ import annotations

from dataclasses import dataclass, fields
import json
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class ExperimentConfig:
    dataset_name: str
    dataset_version: int
    seed: int
    test_fraction: float
    forget_fraction: float
    number_of_shards: int
    number_of_slices: int
    calibration_size: int
    bootstrap_iterations: int

    def __post_init__(self) -> None:
        if not self.dataset_name.strip():
            raise ValueError("dataset_name cannot be empty")

        if self.dataset_version < 1:
            raise ValueError("dataset_version must be positive")

        if self.seed < 0:
            raise ValueError("seed cannot be negative")

        if not 0.0 < self.test_fraction < 1.0:
            raise ValueError(
                "test_fraction must be between 0 and 1"
            )

        if not 0.0 < self.forget_fraction < 1.0:
            raise ValueError(
                "forget_fraction must be between 0 and 1"
            )

        if self.number_of_shards < 2:
            raise ValueError(
                "number_of_shards must be at least 2"
            )

        if self.number_of_slices < 1:
            raise ValueError(
                "number_of_slices must be at least 1"
            )

        if self.calibration_size < 1:
            raise ValueError(
                "calibration_size must be positive"
            )

        if self.bootstrap_iterations < 100:
            raise ValueError(
                "bootstrap_iterations must be at least 100"
            )


def experiment_config_from_mapping(
    values: Mapping[str, Any],
) -> ExperimentConfig:
    expected = {
        field.name
        for field in fields(ExperimentConfig)
    }
    provided = set(values)

    missing = expected - provided
    if missing:
        raise ValueError(
            f"Missing configuration fields: {sorted(missing)}"
        )

    unknown = provided - expected
    if unknown:
        raise ValueError(
            f"Unknown configuration fields: {sorted(unknown)}"
        )

    return ExperimentConfig(**dict(values))


def load_experiment_config(
    path: str | Path,
) -> ExperimentConfig:
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Configuration not found: {path}"
        )

    try:
        values = json.loads(
            path.read_text(encoding="utf-8")
        )
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid JSON configuration: {path}"
        ) from error

    if not isinstance(values, dict):
        raise ValueError(
            "Experiment configuration must be a JSON object"
        )

    return experiment_config_from_mapping(values)