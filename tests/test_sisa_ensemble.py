import numpy as np
import pandas as pd
import pytest

from unlearning.sisa_ensemble import ShardedEnsemble


class FixedProbabilityModel:
    def __init__(self, positive_probabilities):
        self.positive_probabilities = np.asarray(
            positive_probabilities,
            dtype=float,
        )

    def predict_proba(self, features):
        positive = self.positive_probabilities[: len(features)]
        negative = 1.0 - positive

        return np.column_stack([negative, positive])


@pytest.fixture
def features():
    return pd.DataFrame(
        {
            "record_id": ["a", "b"],
            "value": [10, 20],
        }
    )


def test_ensemble_averages_probabilities(features):
    ensemble = ShardedEnsemble(
        models={
            0: FixedProbabilityModel([0.2, 0.8]),
            1: FixedProbabilityModel([0.4, 0.6]),
            2: FixedProbabilityModel([0.6, 0.4]),
        }
    )

    probabilities = ensemble.predict_proba(features)

    expected_positive = np.array([0.4, 0.6])

    assert np.allclose(
        probabilities[:, 1],
        expected_positive,
    )
    assert np.allclose(
        probabilities.sum(axis=1),
        1.0,
    )


def test_predictions_use_half_threshold(features):
    ensemble = ShardedEnsemble(
        models={
            0: FixedProbabilityModel([0.49, 0.51]),
            1: FixedProbabilityModel([0.49, 0.51]),
        }
    )

    predictions = ensemble.predict(features)

    assert predictions.tolist() == [0, 1]


def test_threshold_is_configurable(features):
    ensemble = ShardedEnsemble(
        models={
            0: FixedProbabilityModel([0.6, 0.8]),
        },
        threshold=0.7,
    )

    predictions = ensemble.predict(features)

    assert predictions.tolist() == [0, 1]


def test_empty_model_collection_is_rejected():
    with pytest.raises(
        ValueError,
        match="at least one model",
    ):
        ShardedEnsemble(models={})


def test_empty_prediction_dataset_is_rejected():
    ensemble = ShardedEnsemble(
        models={
            0: FixedProbabilityModel([]),
        }
    )

    empty = pd.DataFrame(columns=["record_id", "value"])

    with pytest.raises(
        ValueError,
        match="empty dataset",
    ):
        ensemble.predict_proba(empty)


@pytest.mark.parametrize(
    "threshold",
    [0.0, 1.0, -0.1, 1.1],
)
def test_invalid_threshold_is_rejected(threshold):
    with pytest.raises(ValueError):
        ShardedEnsemble(
            models={
                0: FixedProbabilityModel([0.5]),
            },
            threshold=threshold,
        )