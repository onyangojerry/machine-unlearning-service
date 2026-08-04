from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


class ArtifactValidationError(RuntimeError):
    """Raised when experiment outputs violate their contract."""


CONTRACTS = {
    "baseline": {
        "artifacts/original_model.joblib": (),
        "artifacts/baseline_metrics.json": (
            "accuracy",
            "f1",
            "training_seconds",
            "training_records",
        ),
        "artifacts/forget_manifest.json": (
            "record_count",
            "record_ids",
        ),
    },
    "reference": {
        "artifacts/reference_model.joblib": (),
        "artifacts/exact_unlearning_results.json": (
            "reference_test_metrics",
            "test_behavior_comparison",
            "forget_behavior_comparison",
        ),
        "reports/figures/forget_probability_shift.png": (),
    },
    "sisa": {
        "artifacts/sisa_ensemble.joblib": (),
        "artifacts/shard_manifest.csv": (),
        "artifacts/sisa_training_results.json": (
            "number_of_shards",
            "shard_record_counts",
            "ensemble_test_metrics",
        ),
    },
    "selective-plan": {
        "artifacts/selective_plans.json": (
            "single_record",
            "same_shard_batch",
            "distributed_batch",
        ),
        "artifacts/pre_unlearning_hashes.json": (),
    },
    "selective": {
        "artifacts/selective_single/ensemble.joblib": (),
        "artifacts/selective_single/selective_results.json": (
            "affected_shards",
            "unaffected_shards",
            "observed_speedup",
            "unaffected_hashes_unchanged",
        ),
    },
    "privacy": {
        "artifacts/privacy_results.json": (
            "distributed_track",
            "single_record_track",
        ),
        (
            "reports/figures/"
            "membership_inference_rates.png"
        ): (),
    },
}


def _load_json_object(path: Path) -> dict:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8")
        )
    except json.JSONDecodeError as error:
        raise ArtifactValidationError(
            f"Artifact contains invalid JSON: {path}"
        ) from error

    if not isinstance(value, dict):
        raise ArtifactValidationError(
            f"JSON artifact must contain an object: {path}"
        )

    return value


def _require_keys(
    path: Path,
    values: dict,
    required_keys: Iterable[str],
) -> None:
    missing = set(required_keys) - set(values)

    if missing:
        raise ArtifactValidationError(
            f"{path} is missing keys: {sorted(missing)}"
        )


def validate_stage_artifacts(
    stage: str,
    project_root: str | Path = ".",
    number_of_shards: int = 5,
) -> list[Path]:
    if stage not in CONTRACTS:
        raise ValueError(f"Unknown pipeline stage: {stage}")

    project_root = Path(project_root).resolve()
    validated = []

    for relative_path, required_keys in CONTRACTS[
        stage
    ].items():
        path = project_root / relative_path

        if not path.is_file():
            raise ArtifactValidationError(
                f"Required artifact is missing: {path}"
            )

        if path.stat().st_size == 0:
            raise ArtifactValidationError(
                f"Artifact is empty: {path}"
            )

        if path.suffix == ".json":
            values = _load_json_object(path)
            _require_keys(
                path,
                values,
                required_keys,
            )

        validated.append(path)

    if stage in {"sisa", "selective"}:
        shard_directory = (
            project_root / "artifacts" / "shards"
            if stage == "sisa"
            else (
                project_root
                / "artifacts"
                / "selective_single"
                / "shards"
            )
        )

        for shard_id in range(number_of_shards):
            shard_path = (
                shard_directory
                / f"shard_{shard_id}.joblib"
            )

            if not shard_path.is_file():
                raise ArtifactValidationError(
                    f"Missing shard artifact: {shard_path}"
                )

            if shard_path.stat().st_size == 0:
                raise ArtifactValidationError(
                    f"Shard artifact is empty: {shard_path}"
                )

            validated.append(shard_path)

    return validated