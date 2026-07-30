from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TrainingPartition:
    retain_X: pd.DataFrame
    retain_y: pd.Series
    forget_X: pd.DataFrame
    forget_y: pd.Series


def partition_training_data(
    features: pd.DataFrame,
    target: pd.Series,
    forget_ids: list[str],
) -> TrainingPartition:
    """Divide training data into retained and forgotten records."""

    if "record_id" not in features.columns:
        raise KeyError("features requires a record_id column")

    if len(features) != len(target):
        raise ValueError(
            "features and target must contain the same number of rows"
        )

    if not forget_ids:
        raise ValueError("forget_ids cannot be empty")

    # Establish positional alignment before applying Boolean masks.
    features = features.reset_index(drop=True)
    target = target.reset_index(drop=True)

    requested_ids = set(forget_ids)
    available_ids = set(features["record_id"])

    missing_ids = requested_ids - available_ids
    if missing_ids:
        raise KeyError(
            f"Unknown forget IDs: {sorted(missing_ids)}"
        )

    forget_mask = features["record_id"].isin(requested_ids)
    retain_mask = ~forget_mask

    partition = TrainingPartition(
        retain_X=features.loc[retain_mask].reset_index(drop=True),
        retain_y=target.loc[retain_mask].reset_index(drop=True),
        forget_X=features.loc[forget_mask].reset_index(drop=True),
        forget_y=target.loc[forget_mask].reset_index(drop=True),
    )

    if len(partition.forget_X) != len(requested_ids):
        raise RuntimeError(
            "Partitioned forget-set size does not match the request"
        )

    return partition