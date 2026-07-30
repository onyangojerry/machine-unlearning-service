import numpy as np
import pandas as pd
import pytest

from unlearning.comparison import (
    calculate_metric_delta,
    compare_model_behavior,
)


class FixedModel:
    """Minimal deterministic classifier for comparison tests."""

    def __init__(
        self,
        predictions: list[int],
        positive_probabilities: list[float],
    ):
        self.predictions = np.asarray(predictions)
        self.positive_probabilities = np.asarray(
            positive_probabilities,
            dtype=float,
        )

    def predict(self, features):
        return self.predictions[: len(features)]

    def predict_proba(self, features):
        positive = self.positive_probabilities[: len(features)]
        negative = 1.0 - positive

        return np.column_stack([negative, positive])


@pytest.fixture
def two_records():
    return pd.DataFrame(
        {
            "record_id": ["user-1", "user-2"],
            "value": [10, 20],
        }
    )


def test_identical_models_have_zero_behavior_gap(two_records):
    first = FixedModel(
        predictions=[0, 1],
        positive_probabilities=[0.2, 0.8],
    )
    second = FixedModel(
        predictions=[0, 1],
        positive_probabilities=[0.2, 0.8],
    )

    result = compare_model_behavior(
        first,
        second,
        two_records,
    )

    assert result["prediction_disagreement_rate"] == 0.0
    assert result["mean_absolute_probability_gap"] == 0.0
    assert result["maximum_probability_gap"] == 0.0


def test_expected_behavior_difference_is_calculated(two_records):
    first = FixedModel(
        predictions=[0, 1],
        positive_probabilities=[0.2, 0.8],
    )
    second = FixedModel(
        predictions=[1, 1],
        positive_probabilities=[0.6, 0.7],
    )

    result = compare_model_behavior(
        first,
        second,
        two_records,
    )

    # One of two labels differs.
    assert result["prediction_disagreement_rate"] == 0.5

    # Probability differences are 0.4 and 0.1.
    assert np.isclose(
        result["mean_absolute_probability_gap"],
        0.25,
    )
    assert np.isclose(
        result["maximum_probability_gap"],
        0.4,
    )


def test_complete_prediction_disagreement(two_records):
    first = FixedModel(
        predictions=[0, 0],
        positive_probabilities=[0.1, 0.2],
    )
    second = FixedModel(
        predictions=[1, 1],
        positive_probabilities=[0.9, 0.8],
    )

    result = compare_model_behavior(
        first,
        second,
        two_records,
    )

    assert result["prediction_disagreement_rate"] == 1.0
    assert np.isclose(
        result["mean_absolute_probability_gap"],
        0.7,
    )
    assert np.isclose(
        result["maximum_probability_gap"],
        0.8,
    )


def test_probabilities_can_change_without_label_disagreement(
    two_records,
):
    first = FixedModel(
        predictions=[0, 1],
        positive_probabilities=[0.1, 0.9],
    )
    second = FixedModel(
        predictions=[0, 1],
        positive_probabilities=[0.4, 0.6],
    )

    result = compare_model_behavior(
        first,
        second,
        two_records,
    )

    assert result["prediction_disagreement_rate"] == 0.0
    assert np.isclose(
                result["mean_absolute_probability_gap"],
        0.3,
    )
    assert np.isclose(
        result["maximum_probability_gap"],
        0.3,
    )


def test_comparison_is_symmetric(two_records):
    first = FixedModel(
        predictions=[0, 1],
        positive_probabilities=[0.2, 0.8],
    )
    second = FixedModel(
        predictions=[1, 1],
        positive_probabilities=[0.6, 0.7],
    )

    forward = compare_model_behavior(
        first,
        second,
        two_records,
    )
    reverse = compare_model_behavior(
        second,
        first,
        two_records,
    )

    assert forward == reverse


def test_single_record_comparison():
    features = pd.DataFrame(
        {
            "record_id": ["user-1"],
            "value": [10],
        }
    )

    first = FixedModel(
        predictions=[0],
        positive_probabilities=[0.25],
    )
    second = FixedModel(
        predictions=[1],
        positive_probabilities=[0.75],
    )

    result = compare_model_behavior(
        first,
        second,
        features,
    )

    assert result["prediction_disagreement_rate"] == 1.0
    assert np.isclose(
        result["mean_absolute_probability_gap"],
        0.5,
    )
    assert np.isclose(
        result["maximum_probability_gap"],
        0.5,
    )


def test_comparison_returns_plain_floats(two_records):
    first = FixedModel(
        predictions=[0, 1],
        positive_probabilities=[0.2, 0.8],
    )
    second = FixedModel(
        predictions=[1, 1],
        positive_probabilities=[0.6, 0.7],
    )

    result = compare_model_behavior(
        first,
        second,
        two_records,
    )

    assert set(result) == {
        "prediction_disagreement_rate",
        "mean_absolute_probability_gap",
        "maximum_probability_gap",
    }

    assert all(
        isinstance(value, float)
        for value in result.values()
    )


def test_empty_dataset_is_rejected():
    features = pd.DataFrame(
        columns=["record_id", "value"]
    )

    first = FixedModel(
        predictions=[],
        positive_probabilities=[],
    )
    second = FixedModel(
        predictions=[],
        positive_probabilities=[],
    )

    with pytest.raises(
        ValueError,
        match="empty dataset",
    ):
        compare_model_behavior(
            first,
            second,
            features,
        )


@pytest.mark.parametrize(
    (
        "original_value",
        "reference_value",
        "expected_delta",
    ),
    [
        (0.70, 0.68, -0.02),
        (0.70, 0.72, 0.02),
        (0.70, 0.70, 0.00),
        (0.00, 1.00, 1.00),
        (1.00, 0.00, -1.00),
    ],
)
def test_metric_delta_direction(
    original_value,
    reference_value,
    expected_delta,
):
    original_metrics = {"f1": original_value}
    reference_metrics = {"f1": reference_value}

    result = calculate_metric_delta(
        original_metrics,
        reference_metrics,
        "f1",
    )

    assert np.isclose(result, expected_delta)
    assert isinstance(result, float)


def test_metric_delta_does_not_modify_inputs():
    original_metrics = {
        "accuracy": 0.80,
        "f1": 0.70,
    }
    reference_metrics = {
        "accuracy": 0.81,
        "f1": 0.69,
    }

    original_before = original_metrics.copy()
    reference_before = reference_metrics.copy()

    calculate_metric_delta(
        original_metrics,
        reference_metrics,
        "f1",
    )

    assert original_metrics == original_before
    assert reference_metrics == reference_before


def test_missing_original_metric_is_rejected():
    original_metrics = {"accuracy": 0.80}
    reference_metrics = {"f1": 0.70}

    with pytest.raises(
        KeyError,
        match="missing from original metrics",
    ):
        calculate_metric_delta(
            original_metrics,
            reference_metrics,
            "f1",
        )


def test_missing_reference_metric_is_rejected():
    original_metrics = {"f1": 0.70}
    reference_metrics = {"accuracy": 0.80}

    with pytest.raises(
        KeyError,
        match="missing from reference metrics",
    ):
        calculate_metric_delta(
            original_metrics,
            reference_metrics,
            "f1",
        )