from __future__ import annotations

import numpy as np


def compare_model_behavior(
    first_model,
    second_model,
    features,
) -> dict[str, float]:
    """Compare predictions and probabilities from two classifiers."""

    if len(features) == 0:
        raise ValueError("Cannot compare models on an empty dataset")

    first_predictions = first_model.predict(features)
    second_predictions = second_model.predict(features)

    first_probabilities = first_model.predict_proba(features)[:, 1]
    second_probabilities = second_model.predict_proba(features)[:, 1]

    probability_difference = np.abs(
        first_probabilities - second_probabilities
    )

    return {
        "prediction_disagreement_rate": float(
            np.mean(first_predictions != second_predictions)
        ),
        "mean_absolute_probability_gap": float(
            np.mean(probability_difference)
        ),
        "maximum_probability_gap": float(
            np.max(probability_difference)
        ),
    }


def calculate_metric_delta(
    original_metrics: dict[str, float],
    reference_metrics: dict[str, float],
    metric_name: str,
) -> float:
    """Return reference metric minus original metric."""

    if metric_name not in original_metrics:
        raise KeyError(
            f"{metric_name!r} is missing from original metrics"
        )

    if metric_name not in reference_metrics:
        raise KeyError(
            f"{metric_name!r} is missing from reference metrics"
        )

    return float(
        reference_metrics[metric_name]
        - original_metrics[metric_name]
    )