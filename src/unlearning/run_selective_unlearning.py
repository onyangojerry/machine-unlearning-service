from __future__ import annotations

import json
import math
from pathlib import Path
import shutil

import joblib
import mlflow
import pandas as pd

from unlearning.comparison import compare_model_behavior
from unlearning.data import (
    add_stable_record_ids,
    create_splits,
    load_adult_dataset,
)
from unlearning.metrics import evaluate_binary_classifier
from unlearning.model import build_model_pipeline
from unlearning.selective import (
    fingerprint_shard_artifacts,
)
from unlearning.selective_retrain import (
    retrain_affected_shards,
)
from unlearning.sisa_ensemble import (
    train_sharded_ensemble,
)


SEED = 42
SCENARIO = "single_record"

ARTIFACT_DIRECTORY = Path("artifacts")
ORIGINAL_SHARD_DIRECTORY = (
    ARTIFACT_DIRECTORY / "shards"
)
OUTPUT_DIRECTORY = (
    ARTIFACT_DIRECTORY / "selective_single"
)
OUTPUT_SHARD_DIRECTORY = OUTPUT_DIRECTORY / "shards"

ORIGINAL_ENSEMBLE_PATH = (
    ARTIFACT_DIRECTORY / "sisa_ensemble.joblib"
)
SHARD_MANIFEST_PATH = (
    ARTIFACT_DIRECTORY / "shard_manifest.csv"
)
PLAN_PATH = (
    ARTIFACT_DIRECTORY / "selective_plans.json"
)
RESULTS_PATH = (
    OUTPUT_DIRECTORY / "selective_results.json"
)
OUTPUT_ENSEMBLE_PATH = (
    OUTPUT_DIRECTORY / "ensemble.joblib"
)


def clean_metrics(values: dict) -> dict:
    return {
        key: (
            None
            if isinstance(value, float)
            and math.isnan(value)
            else value
        )
        for key, value in values.items()
    }


def main() -> None:
    OUTPUT_SHARD_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    plans = json.loads(
        PLAN_PATH.read_text(encoding="utf-8")
    )
    scenario = plans[SCENARIO]
    forget_ids = scenario["forget_ids"]

    original_ensemble = joblib.load(
        ORIGINAL_ENSEMBLE_PATH
    )
    shard_manifest = pd.read_csv(
        SHARD_MANIFEST_PATH
    )

    features, target = load_adult_dataset()
    features = add_stable_record_ids(features)
    splits = create_splits(features, target, seed=SEED)

    selective = retrain_affected_shards(
        original_ensemble=original_ensemble,
        training_features=splits.X_train,
        training_target=splits.y_train,
        shard_manifest=shard_manifest,
        forget_ids=forget_ids,
        model_builder=build_model_pipeline,
        seed=SEED,
    )

    # Build the architecture-matched exact reference by
    # retraining every shard after applying the same deletion.
    delete_mask = splits.X_train[
        "record_id"
    ].isin(forget_ids)
    retain_mask = ~delete_mask

    retain_X = splits.X_train.loc[
        retain_mask
    ].reset_index(drop=True)
    retain_y = splits.y_train.loc[
        retain_mask
    ].reset_index(drop=True)

    retained_manifest = shard_manifest.loc[
        ~shard_manifest["record_id"].isin(forget_ids)
    ].reset_index(drop=True)

    full_reference = train_sharded_ensemble(
        training_features=retain_X,
        training_target=retain_y,
        shard_manifest=retained_manifest,
        model_builder=build_model_pipeline,
        seed=SEED,
    )

    selective_metrics = evaluate_binary_classifier(
        selective.ensemble,
        splits.X_test,
        splits.y_test,
    )
    reference_metrics = evaluate_binary_classifier(
        full_reference.ensemble,
        splits.X_test,
        splits.y_test,
    )

    test_comparison = compare_model_behavior(
        selective.ensemble,
        full_reference.ensemble,
        splits.X_test,
    )

    forget_X = splits.X_train.loc[
        delete_mask
    ].reset_index(drop=True)

    forget_comparison = compare_model_behavior(
        selective.ensemble,
        full_reference.ensemble,
        forget_X,
    )

    observed_speedup = (
        full_reference.total_wall_seconds
        / selective.total_wall_seconds
        if selective.total_wall_seconds > 0
        else None
    )

    original_hashes = fingerprint_shard_artifacts(
        ORIGINAL_SHARD_DIRECTORY,
        sorted(original_ensemble.models),
    )

    # Copy unaffected files byte-for-byte. Serialize only
    # models that were actually retrained.
    for shard_id in sorted(original_ensemble.models):
        destination = (
            OUTPUT_SHARD_DIRECTORY
            / f"shard_{shard_id}.joblib"
        )

        if shard_id in selective.affected_shards:
            joblib.dump(
                selective.ensemble.models[shard_id],
                destination,
            )
        else:
            shutil.copy2(
                ORIGINAL_SHARD_DIRECTORY
                / f"shard_{shard_id}.joblib",
                destination,
            )

    updated_hashes = fingerprint_shard_artifacts(
        OUTPUT_SHARD_DIRECTORY,
        sorted(selective.ensemble.models),
    )

    unaffected_hashes_unchanged = all(
        original_hashes[shard_id]
        == updated_hashes[shard_id]
        for shard_id in selective.unaffected_shards
    )

    hash_verification = {
        str(shard_id): {
            "affected": (
                shard_id in selective.affected_shards
            ),
            "before": original_hashes[shard_id],
            "after": updated_hashes[shard_id],
            "unchanged": (
                original_hashes[shard_id]
                == updated_hashes[shard_id]
            ),
        }
        for shard_id in sorted(original_ensemble.models)
    }

    results = {
        "scenario": SCENARIO,
        "forget_ids": forget_ids,
        "affected_shards": list(
            selective.affected_shards
        ),
        "unaffected_shards": list(
            selective.unaffected_shards
        ),
        "selective_retraining_seconds": (
            selective.total_wall_seconds
        ),
        "full_sharded_retraining_seconds": (
            full_reference.total_wall_seconds
        ),
        "observed_speedup": observed_speedup,
        "selective_test_metrics": clean_metrics(
            selective_metrics
        ),
        "full_reference_test_metrics": clean_metrics(
            reference_metrics
        ),
        "selective_vs_reference_test": (
            test_comparison
        ),
        "selective_vs_reference_forget": (
            forget_comparison
        ),
        "unaffected_hashes_unchanged": (
            unaffected_hashes_unchanged
        ),
        "artifact_hashes": hash_verification,
    }

    joblib.dump(
        selective.ensemble,
        OUTPUT_ENSEMBLE_PATH,
    )
    RESULTS_PATH.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("adult-income-unlearning")

    with mlflow.start_run(
        run_name="single-record-selective-unlearning"
    ):
        mlflow.log_params(
            {
                "scenario": SCENARIO,
                "forget_record_count": len(forget_ids),
                "affected_shard_count": len(
                    selective.affected_shards
                ),
                "total_shard_count": len(
                    selective.ensemble.models
                ),
            }
        )

        mlflow.log_metrics(
            {
                "selective_retraining_seconds": (
                    selective.total_wall_seconds
                ),
                "full_sharded_retraining_seconds": (
                    full_reference.total_wall_seconds
                ),
                "observed_speedup": (
                    observed_speedup or 0.0
                ),
                "test_prediction_disagreement": (
                    test_comparison[
                        "prediction_disagreement_rate"
                    ]
                ),
                "test_mean_probability_gap": (
                    test_comparison[
                        "mean_absolute_probability_gap"
                    ]
                ),
            }
        )

        mlflow.log_artifact(str(RESULTS_PATH))
        mlflow.log_artifact(
            str(OUTPUT_ENSEMBLE_PATH)
        )

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()