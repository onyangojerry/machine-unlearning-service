import pandas as pd

from unlearning.selective_retrain import (
    retrain_affected_shards,
)
from unlearning.sisa_ensemble import ShardedEnsemble


class ExistingModel:
    pass


class TrainableModel:
    def __init__(self, seed):
        self.seed = seed
        self.fit_rows = None

    def fit(self, features, target):
        self.fit_rows = len(features)
        return self


class TrackingBuilder:
    def __init__(self):
        self.calls = []

    def __call__(self, features, seed):
        self.calls.append(
            {
                "record_ids": features[
                    "record_id"
                ].tolist(),
                "seed": seed,
            }
        )
        return TrainableModel(seed)


def test_only_affected_shard_is_retrained():
    features = pd.DataFrame(
        {
            "record_id": [
                "a", "b", "c",
                "d", "e",
                "f", "g",
            ],
            "value": [1, 2, 3, 4, 5, 6, 7],
        }
    )

    target = pd.Series(
        [
            "<=50K", ">50K", "<=50K",
            "<=50K", ">50K",
            "<=50K", ">50K",
        ]
    )

    manifest = pd.DataFrame(
        {
            "record_id": [
                "a", "b", "c",
                "d", "e",
                "f", "g",
            ],
            "shard_id": [
                0, 0, 0,
                1, 1,
                2, 2,
            ],
        }
    )

    original_models = {
        0: ExistingModel(),
        1: ExistingModel(),
        2: ExistingModel(),
    }

    original = ShardedEnsemble(
        models=original_models
    )
    builder = TrackingBuilder()

    result = retrain_affected_shards(
        original_ensemble=original,
        training_features=features,
        training_target=target,
        shard_manifest=manifest,
        forget_ids=["a"],
        model_builder=builder,
        seed=42,
    )

    assert result.affected_shards == (0,)
    assert result.unaffected_shards == (1, 2)

    assert len(builder.calls) == 1
    assert "a" not in builder.calls[0]["record_ids"]

    assert (
        result.ensemble.models[1]
        is original.models[1]
    )
    assert (
        result.ensemble.models[2]
        is original.models[2]
    )

    assert (
        result.ensemble.models[0]
        is not original.models[0]
    )

    # The original ensemble must remain unchanged.
    assert original.models[0] is original_models[0]


def test_multiple_records_in_one_shard_trigger_one_fit():
    features = pd.DataFrame(
        {
            "record_id": ["a", "b", "c", "d"],
            "value": [1, 2, 3, 4],
        }
    )
    target = pd.Series(
        ["<=50K", ">50K", "<=50K", ">50K"]
    )
    manifest = pd.DataFrame(
        {
            "record_id": ["a", "b", "c", "d"],
            "shard_id": [0, 0, 0, 0],
        }
    )

    original = ShardedEnsemble(
        models={0: ExistingModel()}
    )
    builder = TrackingBuilder()

    result = retrain_affected_shards(
        original,
        features,
        target,
        manifest,
        forget_ids=["a", "b"],
        model_builder=builder,
    )

    assert result.affected_shards == (0,)
    assert len(builder.calls) == 1
    assert set(builder.calls[0]["record_ids"]) == {
        "c", "d"
    }