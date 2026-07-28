from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import joblib
import mlflow

from unlearning.data import (
    add_stable_record_ids,
    create_splits,
    load_adult_dataset,
)
from unlearning.forget_set import (
    select_forget_ids,
    write_forget_manifest,
)
from unlearning.metrics import (
    encode_target,
    evaluate_binary_classifier,
)
from unlearning.model import build_model_pipeline


SEED = 42
FORGET_FRACTION = 0.01

ARTIFACT_DIRECTORY = Path("artifacts")
MODEL_PATH = ARTIFACT_DIRECTORY / "original_model.joblib"
METRICS_PATH = ARTIFACT_DIRECTORY / "baseline_metrics.json"
MANIFEST_PATH = ARTIFACT_DIRECTORY / "forget_manifest.json"


def main() -> None:
    ARTIFACT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    features, target = load_adult_dataset()
    features = add_stable_record_ids(features)
    splits = create_splits(features, target, seed=SEED)

    model = build_model_pipeline(splits.X_train, seed=SEED)

    y_train = encode_target(splits.y_train)

    start = perf_counter()
    model.fit(splits.X_train, y_train)
    training_seconds = perf_counter() - start

    metrics = evaluate_binary_classifier(
        model,
        splits.X_test,
        splits.y_test,
    )
    metrics["training_seconds"] = training_seconds
    metrics["training_records"] = len(splits.X_train)
    metrics["test_records"] = len(splits.X_test)

    forget_ids = select_forget_ids(
        splits.X_train,
        fraction=FORGET_FRACTION,
        seed=SEED,
    )

    joblib.dump(model, MODEL_PATH)

    METRICS_PATH.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    write_forget_manifest(
        forget_ids,
        MANIFEST_PATH,
        fraction=FORGET_FRACTION,
        seed=SEED,
    )

    mlflow.set_tracking_uri("sqlite:///mlflow.db") # updated from "file:./mlruns"
    mlflow.set_experiment("adult-income-unlearning")

    with mlflow.start_run(run_name="original-logistic-regression"):
        mlflow.log_params(
            {
                "dataset": "adult",
                "dataset_version": 2,
                "seed": SEED,
                "forget_fraction": FORGET_FRACTION,
                "model": "logistic_regression",
                "solver": "liblinear",
                "max_iterations": 1000,
            }
        )

        numeric_metrics = {
            key: value
            for key, value in metrics.items()
            if isinstance(value, (int, float))
        }

        mlflow.log_metrics(numeric_metrics)
        mlflow.log_artifact(str(MODEL_PATH))
        mlflow.log_artifact(str(METRICS_PATH))
        mlflow.log_artifact(str(MANIFEST_PATH))

    print(json.dumps(metrics, indent=2))
    print(f"Forget records: {len(forget_ids)}")
    print(f"Model written to: {MODEL_PATH}")
    print(f"Metrics written to: {METRICS_PATH}")


if __name__ == "__main__":
    main()