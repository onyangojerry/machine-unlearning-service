from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def select_forget_ids(
    training_features: pd.DataFrame,
    fraction: float = 0.01,
    seed: int = 42,
) -> list[str]:
    if not 0 < fraction < 1:
        raise ValueError("fraction must be between 0 and 1")

    if "record_id" not in training_features:
        raise KeyError("training_features requires record_id")

    count = max(1, round(len(training_features) * fraction))
    generator = np.random.default_rng(seed)

    selected_positions = generator.choice(
        len(training_features),
        size=count,
        replace=False,
    )

    selected_ids = training_features.iloc[selected_positions][
        "record_id"
    ].tolist()

    return sorted(selected_ids)


def write_forget_manifest(
    record_ids: list[str],
    output_path: str | Path,
    fraction: float,
    seed: int,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = {
        "dataset": "adult",
        "dataset_version": 2,
        "selection_method": "uniform_without_replacement",
        "fraction": fraction,
        "seed": seed,
        "record_count": len(record_ids),
        "record_ids": record_ids,
    }

    output_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )