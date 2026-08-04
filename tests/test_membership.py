import numpy as np
import pandas as pd
import pytest

from unlearning.membership import (
    fit_membership_threshold,
    membership_scores,
    predict_membership,
    predicted_member_rate,
    true_label_confidence,
)

from unlearning.membership import (
    bootstrap_member_rate_interval,
)


class FixedProbabilityModel:
    def __init__(self, probabilities):
        self.probabilities = np.asarray(
            probabilities,
            dtype=float,
        )

    def predict_proba(self, features):
        return self.probabilities[: len(features)]


@pytest.fixture
def features():
    return pd.DataFrame(
        {
            "record_id": ["a", "b"],
            "value": [1, 2],
        }
    )


def test_true_label_confidence_selects_correct_class(
    features,
):
    model = FixedProbabilityModel(
        [
            [0.8, 0.2],
            [0.3, 0.7],
        ]
    )
    target = pd.Series(["<=50K", ">50K"])

    confidence = true_label_confidence(
        model,
        features,
        target,
    )

    assert np.allclose(confidence, [0.8, 0.7])


def test_membership_score_is_log_confidence(
    features,
):
    model = FixedProbabilityModel(
        [
            [0.8, 0.2],
            [0.3, 0.7],
        ]
    )
    target = pd.Series(["<=50K", ">50K"])

    scores = membership_scores(
        model,
        features,
        target,
    )

    assert np.allclose(
        scores,
        np.log([0.8, 0.7]),
    )


def test_higher_confidence_has_higher_score():
    features = pd.DataFrame(
        {
            "record_id": ["a", "b"],
        }
    )
    model = FixedProbabilityModel(
        [
            [0.9, 0.1],
            [0.6, 0.4],
        ]
    )
    target = pd.Series(["<=50K", "<=50K"])

    scores = membership_scores(
        model,
        features,
        target,
    )

    assert scores[0] > scores[1]


def test_perfectly_separable_attack():
    member_scores = np.array(
        [-0.05, -0.10, -0.15]
    )
    nonmember_scores = np.array(
        [-1.00, -1.20, -1.50]
    )

    result = fit_membership_threshold(
        member_scores,
        nonmember_scores,
    )

    assert result.balanced_accuracy == 1.0
    assert result.roc_auc == 1.0
    assert result.true_positive_rate == 1.0
    assert result.false_positive_rate == 0.0


def test_membership_prediction_uses_threshold():
    scores = np.array([-1.0, -0.5, -0.1])

    predictions = predict_membership(
        scores,
        threshold=-0.5,
    )

    assert predictions.tolist() == [0, 1, 1]


def test_predicted_member_rate():
    scores = np.array([-1.0, -0.5, -0.1])

    rate = predicted_member_rate(
        scores,
        threshold=-0.5,
    )

    assert rate == pytest.approx(2 / 3)


def test_zero_probability_is_clipped(
    features,
):
    model = FixedProbabilityModel(
        [
            [0.0, 1.0],
            [1.0, 0.0],
        ]
    )
    target = pd.Series(["<=50K", ">50K"])

    scores = membership_scores(
        model,
        features,
        target,
    )

    assert np.isfinite(scores).all()


def test_empty_features_are_rejected():
    model = FixedProbabilityModel([])
    features = pd.DataFrame(
        columns=["record_id"]
    )
    target = pd.Series(dtype=str)

    with pytest.raises(
        ValueError,
        match="empty dataset",
    ):
        membership_scores(
            model,
            features,
            target,
        )


def test_invalid_probability_shape_is_rejected(
    features,
):
    model = FixedProbabilityModel(
        [[0.8], [0.7]]
    )
    target = pd.Series(["<=50K", ">50K"])

    with pytest.raises(
        ValueError,
        match="expected",
    ):
        true_label_confidence(
            model,
            features,
            target,
        )



def test_bootstrap_interval_contains_observed_rate():
    scores = np.array(
        [-0.1, -0.2, -0.3, -1.0, -1.2]
    )

    interval = bootstrap_member_rate_interval(
        scores,
        threshold=-0.5,
        iterations=1000,
        seed=42,
    )

    assert interval is not None

    lower, upper = interval
    observed = 3 / 5

    assert lower <= observed <= upper
    assert 0.0 <= lower <= upper <= 1.0


def test_single_record_has_no_bootstrap_interval():
    interval = bootstrap_member_rate_interval(
        np.array([-0.2]),
        threshold=-0.5,
    )

    assert interval is None


def test_bootstrap_is_reproducible():
    scores = np.array(
        [-0.1, -0.2, -0.3, -1.0, -1.2]
    )

    first = bootstrap_member_rate_interval(
        scores,
        threshold=-0.5,
        iterations=1000,
        seed=42,
    )
    second = bootstrap_member_rate_interval(
        scores,
        threshold=-0.5,
        iterations=1000,
        seed=42,
    )

    assert first == second