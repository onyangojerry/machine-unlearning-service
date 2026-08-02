from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)

from unlearning.metrics import encode_target


@dataclass(frozen=True)
class MembershipAttackResult:
    threshold: float
    balanced_accuracy: float
    roc_auc: float
    true_positive_rate: float
    false_positive_rate: float


def true_label_confidence(
    model,
    features,
    target,
) -> np.ndarray:
    if len(features) == 0:
        raise ValueError(
            "Cannot score an empty dataset"
        )

    labels = encode_target(target)

    if len(features) != len(labels):
        raise ValueError(
            "features and target must have equal lengths"
        )

    probabilities = np.asarray(
        model.predict_proba(features),
        dtype=float,
    )

    expected_shape = (len(features), 2)

    if probabilities.shape != expected_shape:
        raise ValueError(
            f"predict_proba returned {probabilities.shape}; "
            f"expected {expected_shape}"
        )

    if not np.isfinite(probabilities).all():
        raise ValueError(
            "Model probabilities must be finite"
        )

    if (
        (probabilities < 0).any()
        or (probabilities > 1).any()
    ):
        raise ValueError(
            "Model probabilities must be between 0 and 1"
        )

    return probabilities[
        np.arange(len(labels)),
        labels,
    ]


def membership_scores(
    model,
    features,
    target,
    epsilon: float = 1e-12,
) -> np.ndarray:
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    confidence = true_label_confidence(
        model,
        features,
        target,
    )

    return np.log(
        np.clip(confidence, epsilon, 1.0)
    )


def predict_membership(
    scores: np.ndarray,
    threshold: float,
) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)

    if scores.size == 0:
        raise ValueError("scores cannot be empty")

    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite")

    return (scores >= threshold).astype(int)


def fit_membership_threshold(
    member_scores: np.ndarray,
    nonmember_scores: np.ndarray,
) -> MembershipAttackResult:
    member_scores = np.asarray(
        member_scores,
        dtype=float,
    )
    nonmember_scores = np.asarray(
        nonmember_scores,
        dtype=float,
    )

    if member_scores.size == 0:
        raise ValueError(
            "member_scores cannot be empty"
        )

    if nonmember_scores.size == 0:
        raise ValueError(
            "nonmember_scores cannot be empty"
        )

    if not np.isfinite(member_scores).all():
        raise ValueError(
            "member_scores must be finite"
        )

    if not np.isfinite(nonmember_scores).all():
        raise ValueError(
            "nonmember_scores must be finite"
        )

    scores = np.concatenate(
        [member_scores, nonmember_scores]
    )
    labels = np.concatenate(
        [
            np.ones(len(member_scores), dtype=int),
            np.zeros(len(nonmember_scores), dtype=int),
        ]
    )

    unique_scores = np.unique(scores)

    if len(unique_scores) == 1:
        candidates = unique_scores
    else:
        midpoints = (
            unique_scores[:-1] + unique_scores[1:]
        ) / 2.0

        candidates = np.concatenate(
            [
                [unique_scores[0] - 1e-12],
                midpoints,
                [unique_scores[-1] + 1e-12],
            ]
        )

    best_threshold = float(candidates[0])
    best_balanced_accuracy = -1.0

    for candidate in candidates:
        predictions = predict_membership(
            scores,
            float(candidate),
        )

        accuracy = balanced_accuracy_score(
            labels,
            predictions,
        )

        if accuracy > best_balanced_accuracy:
            best_balanced_accuracy = float(accuracy)
            best_threshold = float(candidate)

    predictions = predict_membership(
        scores,
        best_threshold,
    )

    true_negative, false_positive, false_negative, true_positive = (
        confusion_matrix(
            labels,
            predictions,
            labels=[0, 1],
        ).ravel()
    )

    true_positive_rate = (
        true_positive
        / (true_positive + false_negative)
    )
    false_positive_rate = (
        false_positive
        / (false_positive + true_negative)
    )

    return MembershipAttackResult(
        threshold=best_threshold,
        balanced_accuracy=best_balanced_accuracy,
        roc_auc=float(
            roc_auc_score(labels, scores)
        ),
        true_positive_rate=float(true_positive_rate),
        false_positive_rate=float(false_positive_rate),
    )


def predicted_member_rate(
    scores: np.ndarray,
    threshold: float,
) -> float:
    predictions = predict_membership(
        scores,
        threshold,
    )

    return float(np.mean(predictions))

def bootstrap_member_rate_interval(
    scores: np.ndarray,
    threshold: float,
    iterations: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float] | None:
    scores = np.asarray(scores, dtype=float)

    if scores.size == 0:
        raise ValueError("scores cannot be empty")

    if iterations < 100:
        raise ValueError(
            "iterations must be at least 100"
        )

    if not 0.0 < confidence < 1.0:
        raise ValueError(
            "confidence must be between 0 and 1"
        )

    # A one-record sample cannot provide a meaningful interval.
    if scores.size < 2:
        return None

    predictions = predict_membership(
        scores,
        threshold,
    )

    generator = np.random.default_rng(seed)
    bootstrap_rates = np.empty(
        iterations,
        dtype=float,
    )

    for iteration in range(iterations):
        sample = generator.choice(
            predictions,
            size=len(predictions),
            replace=True,
        )
        bootstrap_rates[iteration] = np.mean(sample)

    alpha = 1.0 - confidence

    lower = np.quantile(
        bootstrap_rates,
        alpha / 2.0,
    )
    upper = np.quantile(
        bootstrap_rates,
        1.0 - alpha / 2.0,
    )

    return float(lower), float(upper)