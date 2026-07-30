from __future__ import annotations

from dataclasses import dataclass
import hashlib

import pandas as pd


@dataclass(frozen=True)
class SISAConfig:
    number_of_shards: int = 5
    number_of_slices: int = 1
    assignment_seed: int = 42

    def __post_init__(self) -> None:
        if self.number_of_shards < 2:
            raise ValueError(
                "number_of_shards must be at least 2"
            )

        if self.number_of_slices < 1:
            raise ValueError(
                "number_of_slices must be at least 1"
            )


def stable_shard_index(
    record_id: str,
    number_of_shards: int,
    seed: int = 42,
) -> int:
    if not isinstance(record_id, str) or not record_id:
        raise ValueError(
            "record_id must be a non-empty string"
        )

    if number_of_shards < 2:
        raise ValueError(
            "number_of_shards must be at least 2"
        )

    assignment_key = f"{seed}:{record_id}".encode("utf-8")
    digest = hashlib.sha256(assignment_key).digest()

    integer_value = int.from_bytes(
        digest[:8],
        byteorder="big",
        signed=False,
    )

    return integer_value % number_of_shards


def create_shard_manifest(
    training_features: pd.DataFrame,
    config: SISAConfig,
) -> pd.DataFrame:
    if "record_id" not in training_features.columns:
        raise KeyError(
            "training_features requires a record_id column"
        )

    if training_features["record_id"].isna().any():
        raise ValueError(
            "record_id cannot contain missing values"
        )

    if not training_features["record_id"].is_unique:
        raise ValueError(
            "record_id values must be unique"
        )

    manifest = training_features[["record_id"]].copy()

    manifest["shard_id"] = manifest["record_id"].map(
        lambda record_id: stable_shard_index(
            record_id=record_id,
            number_of_shards=config.number_of_shards,
            seed=config.assignment_seed,
        )
    )

    return manifest.sort_values(
        "record_id"
    ).reset_index(drop=True)


def find_affected_shards(
    shard_manifest: pd.DataFrame,
    forget_ids: list[str],
) -> list[int]:
    required_columns = {"record_id", "shard_id"}

    if not required_columns.issubset(shard_manifest.columns):
        raise KeyError(
            "shard_manifest requires record_id and shard_id"
        )

    requested_ids = set(forget_ids)

    if not requested_ids:
        raise ValueError("forget_ids cannot be empty")

    available_ids = set(shard_manifest["record_id"])
    missing_ids = requested_ids - available_ids

    if missing_ids:
        raise KeyError(
            f"Unknown forget IDs: {sorted(missing_ids)}"
        )

    affected = shard_manifest.loc[
        shard_manifest["record_id"].isin(requested_ids),
        "shard_id",
    ]

    return sorted(int(value) for value in affected.unique())


def attach_shard_assignments(
    training_features: pd.DataFrame,
    shard_manifest: pd.DataFrame,
) -> pd.DataFrame:
    assigned = training_features.merge(
        shard_manifest,
        on="record_id",
        how="left",
        validate="one_to_one",
    )

    if assigned["shard_id"].isna().any():
        raise RuntimeError(
            "Some training records were not assigned to a shard"
        )

    assigned["shard_id"] = assigned["shard_id"].astype(int)

    return assigned