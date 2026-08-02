from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable

import numpy as np
import pandas as pd

from unlearning.metrics import encode_target
from unlearning.sisa import (
    attach_shard_assignments,
    find_affected_shards,
)
from unlearning.sisa_ensemble import ShardedEnsemble


@dataclass(frozen=True)
class SelectiveUnlearningResult:
    ensemble: ShardedEnsemble
    affected_shards: tuple[int, ...]
    unaffected_shards: tuple[int, ...]
    shard_training_seconds: dict[int, float]
    shard_record_counts_after: dict[int, int]
    total_wall_seconds: float


def retrain_affected_shards(
    original_ensemble: ShardedEnsemble,
    training_features: pd.DataFrame,
    training_target: pd.Series,
    shard_manifest: pd.DataFrame,
    forget_ids: list[str],
    model_builder: Callable[[pd.DataFrame, int], object],
    seed: int = 42,
) -> SelectiveUnlearningResult:
    if len(training_features) != len(training_target):
        raise ValueError(
            "training features and targets must have equal lengths"
        )

    if not forget_ids:
        raise ValueError("forget_ids cannot be empty")

    training_features = training_features.reset_index(drop=True)
    training_target = training_target.reset_index(drop=True)

    assigned = attach_shard_assignments(
        training_features,
        shard_manifest,
    )

    available_shards = sorted(original_ensemble.models)

    affected_shards = find_affected_shards(
        shard_manifest,
        forget_ids,
    )

    unavailable = (
        set(affected_shards) - set(available_shards)
    )
    if unavailable:
        raise KeyError(
            f"Missing original shard models: "
            f"{sorted(unavailable)}"
        )

    unaffected_shards = sorted(
        set(available_shards) - set(affected_shards)
    )

    # Preserve references to every unaffected fitted model.
    updated_models = dict(original_ensemble.models)

    durations = {}
    record_counts = {}
    requested_ids = set(forget_ids)

    total_start = perf_counter()

    for shard_id in affected_shards:
        shard_mask = assigned["shard_id"] == shard_id
        deletion_mask = assigned["record_id"].isin(
            requested_ids
        )

        retain_mask = shard_mask & ~deletion_mask

        shard_features = (
            assigned.loc[retain_mask]
            .drop(columns=["shard_id"])
            .reset_index(drop=True)
        )
        shard_target = (
            training_target.loc[retain_mask]
            .reset_index(drop=True)
        )

        if shard_features.empty:
            raise ValueError(
                f"Deletion empties shard {shard_id}"
            )

        encoded_target = encode_target(shard_target)

        if len(np.unique(encoded_target)) < 2:
            raise ValueError(
                f"Retained shard {shard_id} contains "
                f"only one target class"
            )

        model = model_builder(
            shard_features,
            seed + shard_id,
        )

        start = perf_counter()
        model.fit(shard_features, encoded_target)
        duration = perf_counter() - start

        updated_models[shard_id] = model
        durations[shard_id] = duration
        record_counts[shard_id] = len(shard_features)

    total_wall_seconds = perf_counter() - total_start

    return SelectiveUnlearningResult(
        ensemble=ShardedEnsemble(
            models=updated_models,
            threshold=original_ensemble.threshold,
        ),
        affected_shards=tuple(affected_shards),
        unaffected_shards=tuple(unaffected_shards),
        shard_training_seconds=durations,
        shard_record_counts_after=record_counts,
        total_wall_seconds=total_wall_seconds,
    )