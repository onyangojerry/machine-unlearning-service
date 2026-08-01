from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import pandas as pd

from unlearning.forget_set import read_forget_manifest
from unlearning.selective import (
    build_deletion_scenarios,
    build_retraining_plan,
    fingerprint_shard_artifacts,
)


ARTIFACT_DIRECTORY = Path("artifacts")
SHARD_DIRECTORY = ARTIFACT_DIRECTORY / "shards"

SHARD_MANIFEST_PATH = (
    ARTIFACT_DIRECTORY / "shard_manifest.csv"
)
FORGET_MANIFEST_PATH = (
    ARTIFACT_DIRECTORY / "forget_manifest.json"
)
PLAN_PATH = (
    ARTIFACT_DIRECTORY / "selective_plans.json"
)
FINGERPRINT_PATH = (
    ARTIFACT_DIRECTORY / "pre_unlearning_hashes.json"
)


def main() -> None:
    shard_manifest = pd.read_csv(
        SHARD_MANIFEST_PATH
    )
    distributed_ids = read_forget_manifest(
        FORGET_MANIFEST_PATH
    )

    available_shards = sorted(
        int(value)
        for value in shard_manifest["shard_id"].unique()
    )

    scenarios = build_deletion_scenarios(
        shard_manifest,
        distributed_ids,
        same_shard_count=10,
    )

    plans = {
        scenario_name: asdict(
            build_retraining_plan(
                scenario_name=scenario_name,
                shard_manifest=shard_manifest,
                forget_ids=forget_ids,
                available_shards=available_shards,
            )
        )
        for scenario_name, forget_ids in scenarios.items()
    }

    fingerprints = fingerprint_shard_artifacts(
        SHARD_DIRECTORY,
        available_shards,
    )

    PLAN_PATH.write_text(
        json.dumps(plans, indent=2),
        encoding="utf-8",
    )

    FINGERPRINT_PATH.write_text(
        json.dumps(fingerprints, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(plans, indent=2))
    print(f"Plans written to: {PLAN_PATH}")
    print(f"Fingerprints written to: {FINGERPRINT_PATH}")


if __name__ == "__main__":
    main()