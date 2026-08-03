from __future__ import annotations

import json
import math
from pathlib import Path

import joblib
import mlflow

import pytest

pytestmark = pytest.mark.integration

from unlearning.comparison import compare_model_behavior
from unlearning.data import (
    add_stable_record_ids,
    create_splits,
    load_adult_dataset,
)
from unlearning.metrics import evaluate_binary_classifier
from unlearning.model import build_model_pipeline
from unlearning.sisa import (
    SISAConfig,
    create_shard_manifest,
)
from unlearning.sisa_ensemble import train_sharded_ensemble


SEED = 42
NUMBER_OF_SHARDS = 5

ARTIFACT_DIRECTORY = Path("artifacts")
SHARD_DIRECTORY = ARTIFACT_DIRECTORY / "shards"

ORIGINAL_MODEL_PATH = ARTIFACT_DIRECTORY / "original_model.joblib"
REFERENCE_MODEL_PATH = ARTIFACT_DIRECTORY / "reference_model.joblib"

ENSEMBLE_PATH = ARTIFACT_DIRECTORY / "sisa_ensemble.joblib"
MANIFEST_PATH = ARTIFACT_DIRECTORY / "shard_manifest.csv"
RESULTS_PATH = ARTIFACT_DIRECTORY / "sisa_training_results.json"


def clean_metrics(values: dict) -> dict:
    cleaned = {}

    for key, value in values.items():
        if isinstance(value, float) and math.isnan(value):
            cleaned[key] = None
        else:
            cleaned[key] = value

    return cleaned


def main() -> None:
    ARTIFACT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    SHARD_DIRECTORY.mkdir(parents=True, exist_ok=True)

    features, target = load_adult_dataset()
    features = add_stable_record_ids(features)
    splits = create_splits(features, target, seed=SEED)

    config = SISAConfig(
        number_of_shards=NUMBER_OF_SHARDS,
        number_of_slices=1,
        assignment_seed=SEED,
    )

    shard_manifest = create_shard_manifest(
        splits.X_train,
        config,
    )

    training_result = train_sharded_ensemble(
        training_features=splits.X_train,
        training_target=splits.y_train,
        shard_manifest=shard_manifest,
        model_builder=build_model_pipeline,
        seed=SEED,
    )

    ensemble = training_result.ensemble

    ensemble_metrics = evaluate_binary_classifier(
        ensemble,
        splits.X_test,
        splits.y_test,
    )

    results = {
        "method": "sisa_style_sharded_ensemble",
        "number_of_shards": NUMBER_OF_SHARDS,
        "number_of_slices": 1,
        "assignment_seed": SEED,
        "total_training_records": len(splits.X_train),
        "test_records": len(splits.X_test),
        "total_wall_seconds": (
            training_result.total_wall_seconds
        ),
        "sum_shard_training_seconds": sum(
            training_result.shard_training_seconds.values()
        ),
        "maximum_shard_training_seconds": max(
            training_result.shard_training_seconds.values()
        ),
        "shard_training_seconds": (
            training_result.shard_training_seconds
        ),
        "shard_record_counts": (
            training_result.shard_record_counts
        ),
        "ensemble_test_metrics": clean_metrics(
            ensemble_metrics
        ),
    }

    if ORIGINAL_MODEL_PATH.exists():
        original_model = joblib.load(
            ORIGINAL_MODEL_PATH
        )

        results["behavior_vs_original"] = (
            compare_model_behavior(
                original_model,
                ensemble,
                splits.X_test,
            )
        )

    if REFERENCE_MODEL_PATH.exists():
        reference_model = joblib.load(
            REFERENCE_MODEL_PATH
        )

        results["behavior_vs_reference"] = (
            compare_model_behavior(
                reference_model,
                ensemble,
                splits.X_test,
            )
        )

    joblib.dump(ensemble, ENSEMBLE_PATH)
    shard_manifest.to_csv(MANIFEST_PATH, index=False)

    for shard_id, model in ensemble.models.items():
        joblib.dump(
            model,
            SHARD_DIRECTORY / f"shard_{shard_id}.joblib",
        )

    RESULTS_PATH.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    # mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("adult-income-unlearning")

    with mlflow.start_run(
        run_name="sisa-style-five-shard-ensemble"
    ):
        mlflow.log_params(
            {
                "method": "sisa_style_sharded_ensemble",
                "number_of_shards": NUMBER_OF_SHARDS,
                "number_of_slices": 1,
                "assignment_seed": SEED,
                "aggregation": "mean_probability",
            }
        )

        mlflow_metrics = {
            "total_wall_seconds": (
                training_result.total_wall_seconds
            ),
            "sum_shard_training_seconds": sum(
                training_result.shard_training_seconds.values()
            ),
            "maximum_shard_training_seconds": max(
                training_result.shard_training_seconds.values()
            ),
        }

        for name, value in ensemble_metrics.items():
            if not (
                isinstance(value, float)
                and math.isnan(value)
            ):
                mlflow_metrics[
                    f"ensemble_test_{name}"
                ] = value

        for shard_id, duration in (
            training_result.shard_training_seconds.items()
        ):
            mlflow_metrics[
                f"shard_{shard_id}_training_seconds"
            ] = duration

        mlflow.log_metrics(mlflow_metrics)
        mlflow.log_artifact(str(ENSEMBLE_PATH))
        mlflow.log_artifact(str(MANIFEST_PATH))
        mlflow.log_artifact(str(RESULTS_PATH))
        mlflow.log_artifacts(
            str(SHARD_DIRECTORY),
            artifact_path="shard_models",
        )

    print(json.dumps(results, indent=2))
    print(f"Ensemble: {ENSEMBLE_PATH}")
    print(f"Manifest: {MANIFEST_PATH}")
    print(f"Results: {RESULTS_PATH}")


if __name__ == "__main__":
    main()