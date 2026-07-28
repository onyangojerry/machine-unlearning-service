import pandas as pd

from unlearning.model import (
    build_model_pipeline,
    split_feature_types,
)


def toy_features():
    return pd.DataFrame(
        {
            "record_id": [f"user-{i}" for i in range(8)],
            "age": [20, 30, 40, 50, 25, 35, 45, 55],
            "job": [
                "a", "b", "a", "b",
                "a", "b", "a", "b",
            ],
        }
    )


def test_record_id_is_not_a_model_feature():
    categorical, numerical = split_feature_types(toy_features())

    assert "record_id" not in categorical
    assert "record_id" not in numerical


def test_pipeline_can_train_and_predict():
    features = toy_features()
    target = [0, 1, 0, 1, 0, 1, 0, 1]

    model = build_model_pipeline(features)
    model.fit(features, target)

    predictions = model.predict(features)

    assert len(predictions) == len(features)
    assert set(predictions).issubset({0, 1})