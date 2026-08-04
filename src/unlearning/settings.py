from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path

from unlearning.config import (
    ExperimentConfig,
    load_experiment_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "week1.json"
)


@lru_cache(maxsize=1)
def get_experiment_config() -> ExperimentConfig:
    configured_path = os.environ.get(
        "UNLEARNING_CONFIG"
    )

    path = (
        Path(configured_path)
        if configured_path
        else DEFAULT_CONFIG_PATH
    )

    return load_experiment_config(path)