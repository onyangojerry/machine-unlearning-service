from __future__ import annotations

import json
import math
from pathlib import Path
from time import perf_counter

import joblib
import mlflow
import pytest

from unlearning.comparison import (
    calculate_metric_delta,
    compare_model_behavior,
)
from unlearning.data import (
    add_stable_record_ids,
    create_splits,
    load_adult_dataset,
)
from unlearning.forget_set import read_forget_manifest
from unlearning.metrics import (
    encode_target,
    evaluate_binary_classifier,
)
from unlearning.model import build_model_pipeline
from unlearning.partition import partition_training_data
from unlearning.settings import get_experiment_config
from unlearning.visualization import plot_probability_shift


pytestmark = pytest.mark.integration

CONFIG = get_experiment_config()
SEED = CONFIG.seed
# SEED = 42

ARTIFACT_DIRECTORY = Path("artifacts")
ORIGINAL_MODEL_PATH = ARTIFACT_DIRECTORY / "original_model.joblib"
BASELINE_METRICS_PATH = ARTIFACT_DIRECTORY / "baseline_metrics.json"
MANIFEST_PATH = ARTIFACT_DIRECTORY / "forget_manifest.json"

REFERENCE_MODEL_PATH = ARTIFACT_DIRECTORY / "reference_model.joblib"
RESULTS_PATH = ARTIFACT_DIRECTORY / "exact_unlearning_results.json"
FIGURE_PATH = Path(
    "reports/figures/forget_probability_shift.png"
)


def json_safe(values: dict) -> dict:
    cleaned = {}

    for key, value in values.items():
        if isinstance(value, float) and math.isnan(value):
            cleaned[key] = None
        else:
            cleaned[key] = value

    return cleaned


def prefixed(prefix: str, values: dict) -> dict:
    return {
        f"{prefix}_{key}": value
        for key, value in values.items()
    }


def main() -> None:
    original_model = joblib.load(ORIGINAL_MODEL_PATH)

    baseline_metrics = json.loads(
        BASELINE_METRICS_PATH.read_text(encoding="utf-8")
    )
    forget_ids = read_forget_manifest(MANIFEST_PATH)

    features, target = load_adult_dataset()
    features = add_stable_record_ids(features)
    splits = create_splits(features, target, seed=SEED)

    partition = partition_training_data(
        splits.X_train,
        splits.y_train,
        forget_ids,
    )

    reference_model = build_model_pipeline(
        partition.retain_X,
        seed=SEED,
    )

    start = perf_counter()
    reference_model.fit(
        partition.retain_X,
        encode_target(partition.retain_y),
    )
    retraining_seconds = perf_counter() - start

    reference_test_metrics = evaluate_binary_classifier(
        reference_model,
        splits.X_test,
        splits.y_test,
    )

    original_forget_metrics = evaluate_binary_classifier(
        original_model,
        partition.forget_X,
        partition.forget_y,
    )

    reference_forget_metrics = evaluate_binary_classifier(
        reference_model,
        partition.forget_X,
        partition.forget_y,
    )

    test_behavior = compare_model_behavior(
        original_model,
        reference_model,
        splits.X_test,
    )

    forget_behavior = compare_model_behavior(
        original_model,
        reference_model,
        partition.forget_X,
    )

    utility_deltas = {
        metric: calculate_metric_delta(
            baseline_metrics,
            reference_test_metrics,
            metric,
        )
        for metric in [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "log_loss",
        ]
    }

    results = {
        "seed": SEED,
        "original_training_records": len(splits.X_train),
        "retain_records": len(partition.retain_X),
        "forget_records": len(partition.forget_X),
        "reference_retraining_seconds": retraining_seconds,
        "baseline_test_metrics": baseline_metrics,
        "reference_test_metrics": json_safe(
            reference_test_metrics
        ),
        "original_forget_metrics": json_safe(
            original_forget_metrics
        ),
        "reference_forget_metrics": json_safe(
            reference_forget_metrics
        ),
        "test_behavior_comparison": test_behavior,
        "forget_behavior_comparison": forget_behavior,
        "reference_minus_original_utility": json_safe(
            utility_deltas
        ),
    }

    joblib.dump(reference_model, REFERENCE_MODEL_PATH)

    RESULTS_PATH.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    plot_probability_shift(
        original_model,
        reference_model,
        partition.forget_X,
        FIGURE_PATH,
    )

    mlflow.set_experiment("adult-income-unlearning")

    with mlflow.start_run(
        run_name="exact-retraining-reference"
    ):
        mlflow.log_params(
            {
                "method": "full_retraining",
                "dataset": "adult",
                "dataset_version": 2,
                "seed": SEED,
                "original_training_records": len(
                    splits.X_train
                ),
                "retain_records": len(partition.retain_X),
                "forget_records": len(partition.forget_X),
            }
        )

        logged_metrics = {
            "reference_retraining_seconds": retraining_seconds,
            **prefixed(
                "reference_test",
                reference_test_metrics,
            ),
            **prefixed(
                "test_behavior",
                test_behavior,
            ),
            **prefixed(
                "forget_behavior",
                forget_behavior,
            ),
            **prefixed(
                "utility_delta",
                utility_deltas,
            ),
        }

        logged_metrics = {
            key: value
            for key, value in logged_metrics.items()
            if not (
                isinstance(value, float)
                and math.isnan(value)
            )
        }

        mlflow.log_metrics(logged_metrics)
        mlflow.log_artifact(str(REFERENCE_MODEL_PATH))
        mlflow.log_artifact(str(RESULTS_PATH))
        mlflow.log_artifact(str(FIGURE_PATH))

    print(json.dumps(results, indent=2))
    print(f"Reference model: {REFERENCE_MODEL_PATH}")
    print(f"Results: {RESULTS_PATH}")
    print(f"Figure: {FIGURE_PATH}")


if __name__ == "__main__":
    main()
