from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import pandas as pd

from unlearning.sisa import find_affected_shards


@dataclass(frozen=True)
class SelectiveRetrainingPlan:
    scenario_name: str
    forget_ids: tuple[str, ...]
    affected_shards: tuple[int, ...]
    unaffected_shards: tuple[int, ...]
    retraining_fraction: float


def build_retraining_plan(
    scenario_name: str,
    shard_manifest: pd.DataFrame,
    forget_ids: list[str],
    available_shards: list[int],
) -> SelectiveRetrainingPlan:
    if not scenario_name.strip():
        raise ValueError("scenario_name cannot be empty")

    if not available_shards:
        raise ValueError("available_shards cannot be empty")

    unique_available = sorted(
        set(int(value) for value in available_shards)
    )

    affected = find_affected_shards(
        shard_manifest,
        forget_ids,
    )

    unknown_shards = set(affected) - set(unique_available)
    if unknown_shards:
        raise KeyError(
            f"Affected shards are unavailable: "
            f"{sorted(unknown_shards)}"
        )

    unaffected = sorted(
        set(unique_available) - set(affected)
    )

    return SelectiveRetrainingPlan(
        scenario_name=scenario_name,
        forget_ids=tuple(dict.fromkeys(forget_ids)),
        affected_shards=tuple(affected),
        unaffected_shards=tuple(unaffected),
        retraining_fraction=(
            len(affected) / len(unique_available)
        ),
    )


def build_deletion_scenarios(
    shard_manifest: pd.DataFrame,
    distributed_forget_ids: list[str],
    same_shard_count: int = 10,
) -> dict[str, list[str]]:
    if not distributed_forget_ids:
        raise ValueError(
            "distributed_forget_ids cannot be empty"
        )

    if same_shard_count < 1:
        raise ValueError(
            "same_shard_count must be positive"
        )

    first_record_id = distributed_forget_ids[0]

    first_row = shard_manifest.loc[
        shard_manifest["record_id"] == first_record_id
    ]

    if first_row.empty:
        raise KeyError(
            f"Unknown record ID: {first_record_id}"
        )

    first_shard = int(first_row.iloc[0]["shard_id"])

    same_shard_ids = (
        shard_manifest.loc[
            shard_manifest["shard_id"] == first_shard,
            "record_id",
        ]
        .sort_values()
        .head(same_shard_count)
        .tolist()
    )

    if len(same_shard_ids) < same_shard_count:
        raise ValueError(
            f"Shard {first_shard} does not contain "
            f"{same_shard_count} records"
        )

    return {
        "single_record": [first_record_id],
        "same_shard_batch": same_shard_ids,
        "distributed_batch": list(
            dict.fromkeys(distributed_forget_ids)
        ),
    }


def sha256_file(path: str | Path) -> str:
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Artifact not found: {path}"
        )

    digest = hashlib.sha256()

    with path.open("rb") as artifact:
        for block in iter(
            lambda: artifact.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def fingerprint_shard_artifacts(
    shard_directory: str | Path,
    shard_ids: list[int],
) -> dict[int, str]:
    shard_directory = Path(shard_directory)

    return {
        shard_id: sha256_file(
            shard_directory
            / f"shard_{shard_id}.joblib"
        )
        for shard_id in sorted(set(shard_ids))
    }