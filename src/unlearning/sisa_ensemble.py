from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable

import numpy as np
import pandas as pd

from unlearning.metrics import encode_target
from unlearning.sisa import attach_shard_assignments


@dataclass
class ShardedEnsemble:
    models: dict[int, object]
    threshold: float = 0.5

    def __post_init__(self) -> None:
        if not self.models:
            raise ValueError(
                "ShardedEnsemble requires at least one model"
            )

        if not 0.0 < self.threshold < 1.0:
            raise ValueError(
                "threshold must be between 0 and 1"
            )

    def predict_proba(
        self,
        features: pd.DataFrame,
    ) -> np.ndarray:
        if len(features) == 0:
            raise ValueError(
                "Cannot predict on an empty dataset"
            )

        shard_probabilities = []

        for shard_id in sorted(self.models):
            probabilities = self.models[
                shard_id
            ].predict_proba(features)

            if probabilities.ndim != 2:
                raise ValueError(
                    f"Shard {shard_id} returned invalid probabilities"
                )

            if probabilities.shape != (len(features), 2):
                raise ValueError(
                    f"Shard {shard_id} returned probability shape "
                    f"{probabilities.shape}; expected "
                    f"({len(features)}, 2)"
                )

            shard_probabilities.append(probabilities)

        return np.mean(
            np.stack(shard_probabilities, axis=0),
            axis=0,
        )

    def predict(
        self,
        features: pd.DataFrame,
    ) -> np.ndarray:
        positive_probabilities = self.predict_proba(
            features
        )[:, 1]

        return (
            positive_probabilities >= self.threshold
        ).astype(int)


@dataclass(frozen=True)
class ShardTrainingResult:
    ensemble: ShardedEnsemble
    shard_training_seconds: dict[int, float]
    shard_record_counts: dict[int, int]
    total_wall_seconds: float


def train_sharded_ensemble(
    training_features: pd.DataFrame,
    training_target: pd.Series,
    shard_manifest: pd.DataFrame,
    model_builder: Callable[[pd.DataFrame, int], object],
    seed: int = 42,
) -> ShardTrainingResult:
    if len(training_features) != len(training_target):
        raise ValueError(
            "training features and targets must have equal lengths"
        )

    training_features = training_features.reset_index(
        drop=True
    )
    training_target = training_target.reset_index(drop=True)

    assigned = attach_shard_assignments(
        training_features,
        shard_manifest,
    )

    models = {}
    durations = {}
    record_counts = {}

    wall_start = perf_counter()

    for shard_id in sorted(assigned["shard_id"].unique()):
        shard_mask = assigned["shard_id"] == shard_id

        shard_features = assigned.loc[
            shard_mask
        ].drop(columns=["shard_id"]).reset_index(drop=True)

        shard_target = training_target.loc[
            shard_mask
        ].reset_index(drop=True)

        encoded_target = encode_target(shard_target)

        if len(np.unique(encoded_target)) < 2:
            raise ValueError(
                f"Shard {shard_id} contains only one target class"
            )

        model = model_builder(
            shard_features,
            seed + int(shard_id),
        )

        shard_start = perf_counter()
        model.fit(shard_features, encoded_target)
        shard_seconds = perf_counter() - shard_start

        models[int(shard_id)] = model
        durations[int(shard_id)] = shard_seconds
        record_counts[int(shard_id)] = len(shard_features)

    total_wall_seconds = perf_counter() - wall_start

    return ShardTrainingResult(
        ensemble=ShardedEnsemble(models=models),
        shard_training_seconds=durations,
        shard_record_counts=record_counts,
        total_wall_seconds=total_wall_seconds,
    )