from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np

from unlearning.data import (
    add_stable_record_ids,
    create_splits,
    load_adult_dataset,
)
from unlearning.forget_set import read_forget_manifest
from unlearning.partition import partition_training_data
from unlearning.privacy_evaluation import (
    evaluate_model_privacy,
)


SEED = 42
CALIBRATION_SIZE = 1000

ARTIFACT_DIRECTORY = Path("artifacts")
RESULTS_PATH = (
    ARTIFACT_DIRECTORY / "privacy_results.json"
)
FIGURE_PATH = Path(
    "reports/figures/membership_inference_rates.png"
)

ORIGINAL_MODEL_PATH = (
    ARTIFACT_DIRECTORY / "original_model.joblib"
)
REFERENCE_MODEL_PATH = (
    ARTIFACT_DIRECTORY / "reference_model.joblib"
)
ORIGINAL_ENSEMBLE_PATH = (
    ARTIFACT_DIRECTORY / "sisa_ensemble.joblib"
)
SELECTIVE_ENSEMBLE_PATH = (
    ARTIFACT_DIRECTORY
    / "selective_single"
    / "ensemble.joblib"
)
FORGET_MANIFEST_PATH = (
    ARTIFACT_DIRECTORY / "forget_manifest.json"
)
SELECTIVE_PLANS_PATH = (
    ARTIFACT_DIRECTORY / "selective_plans.json"
)


def sample_aligned_rows(
    features,
    target,
    count: int,
    seed: int,
):
    if len(features) != len(target):
        raise ValueError(
            "features and target must have equal lengths"
        )

    count = min(count, len(features))

    generator = np.random.default_rng(seed)
    positions = generator.choice(
        len(features),
        size=count,
        replace=False,
    )

    return (
        features.iloc[positions].reset_index(drop=True),
        target.iloc[positions].reset_index(drop=True),
    )


def plot_member_rates(results: dict) -> None:
    entries = [
        results["distributed_track"]["original"],
        results["distributed_track"]["reference"],
        results["single_record_track"]["original_sharded"],
        results["single_record_track"]["selective_sharded"],
    ]

    labels = [
        "Original\nsingle",
        "Exact\nreference",
        "Original\nsharded",
        "Selective\nsharded",
    ]

    rates = [
        entry["forget_predicted_member_rate"]
        for entry in entries
    ]

    figure, axis = plt.subplots(figsize=(9, 5))

    bars = axis.bar(
        labels,
        rates,
        color=[
            "#C44E52",
            "#4C72B0",
            "#DD8452",
            "#55A868",
        ],
    )

    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Forgotten records predicted as members")
    axis.set_title(
        "Membership-Inference Signal After Unlearning"
    )

    for bar, rate in zip(bars, rates):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            rate + 0.02,
            f"{rate:.2f}",
            ha="center",
        )

    figure.tight_layout()
    FIGURE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    figure.savefig(FIGURE_PATH, dpi=160)
    plt.close(figure)


def main() -> None:
    original_model = joblib.load(
        ORIGINAL_MODEL_PATH
    )
    reference_model = joblib.load(
        REFERENCE_MODEL_PATH
    )
    original_ensemble = joblib.load(
        ORIGINAL_ENSEMBLE_PATH
    )
    selective_ensemble = joblib.load(
        SELECTIVE_ENSEMBLE_PATH
    )

    features, target = load_adult_dataset()
    features = add_stable_record_ids(features)
    splits = create_splits(
        features,
        target,
        seed=SEED,
    )

    distributed_forget_ids = read_forget_manifest(
        FORGET_MANIFEST_PATH
    )

    distributed_partition = partition_training_data(
        splits.X_train,
        splits.y_train,
        distributed_forget_ids,
    )

    calibration_member_X, calibration_member_y = (
        sample_aligned_rows(
            distributed_partition.retain_X,
            distributed_partition.retain_y,
            CALIBRATION_SIZE,
            SEED,
        )
    )

    calibration_nonmember_X, calibration_nonmember_y = (
        sample_aligned_rows(
            splits.X_test,
            splits.y_test,
            CALIBRATION_SIZE,
            SEED + 1,
        )
    )

    original_single_result = evaluate_model_privacy(
        model_name="original_single_model",
        model=original_model,
        calibration_member_X=calibration_member_X,
        calibration_member_y=calibration_member_y,
        calibration_nonmember_X=calibration_nonmember_X,
        calibration_nonmember_y=calibration_nonmember_y,
        forget_X=distributed_partition.forget_X,
        forget_y=distributed_partition.forget_y,
        seed=SEED,
    )

    reference_result = evaluate_model_privacy(
        model_name="exact_retraining_reference",
        model=reference_model,
        calibration_member_X=calibration_member_X,
        calibration_member_y=calibration_member_y,
        calibration_nonmember_X=calibration_nonmember_X,
        calibration_nonmember_y=calibration_nonmember_y,
        forget_X=distributed_partition.forget_X,
        forget_y=distributed_partition.forget_y,
        seed=SEED,
    )

    plans = json.loads(
        SELECTIVE_PLANS_PATH.read_text(
            encoding="utf-8"
        )
    )
    single_forget_ids = plans[
        "single_record"
    ]["forget_ids"]

    single_partition = partition_training_data(
        splits.X_train,
        splits.y_train,
        single_forget_ids,
    )

    original_sharded_result = evaluate_model_privacy(
        model_name="original_sharded_ensemble",
        model=original_ensemble,
        calibration_member_X=calibration_member_X,
        calibration_member_y=calibration_member_y,
        calibration_nonmember_X=calibration_nonmember_X,
        calibration_nonmember_y=calibration_nonmember_y,
        forget_X=single_partition.forget_X,
        forget_y=single_partition.forget_y,
        seed=SEED,
    )

    selective_result = evaluate_model_privacy(
        model_name="selective_sharded_ensemble",
        model=selective_ensemble,
        calibration_member_X=calibration_member_X,
        calibration_member_y=calibration_member_y,
        calibration_nonmember_X=calibration_nonmember_X,
        calibration_nonmember_y=calibration_nonmember_y,
        forget_X=single_partition.forget_X,
        forget_y=single_partition.forget_y,
        seed=SEED,
    )

    results = {
        "threat_model": (
            "black-box true-label log-confidence threshold"
        ),
        "calibration_members": CALIBRATION_SIZE,
        "calibration_nonmembers": CALIBRATION_SIZE,
        "distributed_track": {
            "forget_record_count": len(
                distributed_partition.forget_X
            ),
            "original": original_single_result,
            "reference": reference_result,
        },
        "single_record_track": {
            "forget_record_count": len(
                single_partition.forget_X
            ),
            "statistical_warning": (
                "One record is descriptive only; "
                "no confidence interval is reported."
            ),
            "original_sharded": original_sharded_result,
            "selective_sharded": selective_result,
        },
    }

    RESULTS_PATH.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    plot_member_rates(results)

    mlflow.set_tracking_uri("sqlite:///mlflow.db") # from "file:./mlruns" to "file:./mlruns"
    mlflow.set_experiment("adult-income-unlearning")

    with mlflow.start_run(
        run_name="membership-inference-evaluation"
    ):
        mlflow.log_params(
            {
                "attack": "true_label_log_confidence",
                "calibration_members": CALIBRATION_SIZE,
                "calibration_nonmembers": CALIBRATION_SIZE,
                "seed": SEED,
            }
        )

        mlflow.log_metrics(
            {
                "original_forget_member_rate": (
                    original_single_result[
                        "forget_predicted_member_rate"
                    ]
                ),
                "reference_forget_member_rate": (
                    reference_result[
                        "forget_predicted_member_rate"
                    ]
                ),
                "original_sharded_single_member": (
                    original_sharded_result[
                        "forget_predicted_member_rate"
                    ]
                ),
                "selective_sharded_single_member": (
                    selective_result[
                        "forget_predicted_member_rate"
                    ]
                ),
                "original_attack_auc": (
                    original_single_result[
                        "calibration_attack"
                    ]["roc_auc"]
                ),
                "reference_attack_auc": (
                    reference_result[
                        "calibration_attack"
                    ]["roc_auc"]
                ),
            }
        )

        mlflow.log_artifact(str(RESULTS_PATH))
        mlflow.log_artifact(str(FIGURE_PATH))

    print(json.dumps(results, indent=2))
    print(f"Results: {RESULTS_PATH}")
    print(f"Figure: {FIGURE_PATH}")


if __name__ == "__main__":
    main()