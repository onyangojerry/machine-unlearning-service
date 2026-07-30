import pandas as pd
import pytest

from unlearning.sisa import (
    SISAConfig,
    attach_shard_assignments,
    create_shard_manifest,
    find_affected_shards,
    stable_shard_index,
)


@pytest.fixture
def training_features():
    return pd.DataFrame(
        {
            "record_id": [
                f"user-{index:03d}"
                for index in range(100)
            ],
            "value": range(100),
        }
    )


def test_shard_assignment_is_deterministic():
    first = stable_shard_index(
        "user-001",
        number_of_shards=5,
        seed=42,
    )
    second = stable_shard_index(
        "user-001",
        number_of_shards=5,
        seed=42,
    )

    assert first == second


def test_shard_assignment_is_within_range():
    for index in range(100):
        shard_id = stable_shard_index(
            f"user-{index}",
            number_of_shards=5,
            seed=42,
        )

        assert 0 <= shard_id < 5


def test_assignment_does_not_depend_on_row_order(
    training_features,
):
    config = SISAConfig(number_of_shards=5)

    original = create_shard_manifest(
        training_features,
        config,
    )

    shuffled = create_shard_manifest(
        training_features.sample(
            frac=1.0,
            random_state=99,
        ),
        config,
    )

    pd.testing.assert_frame_equal(original, shuffled)


def test_every_record_is_assigned_once(training_features):
    config = SISAConfig(number_of_shards=5)

    manifest = create_shard_manifest(
        training_features,
        config,
    )

    assert len(manifest) == len(training_features)
    assert manifest["record_id"].is_unique
    assert manifest["shard_id"].notna().all()


def test_duplicate_record_ids_are_rejected(
    training_features,
):
    duplicated = pd.concat(
        [
            training_features,
            training_features.iloc[[0]],
        ],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="must be unique",
    ):
        create_shard_manifest(
            duplicated,
            SISAConfig(),
        )


def test_affected_shards_are_unique_and_sorted(
    training_features,
):
    config = SISAConfig(number_of_shards=5)
    manifest = create_shard_manifest(
        training_features,
        config,
    )

    selected_ids = manifest.iloc[:20][
        "record_id"
    ].tolist()

    expected = sorted(
        int(value)
        for value in manifest.iloc[:20][
            "shard_id"
        ].unique()
    )

    result = find_affected_shards(
        manifest,
        selected_ids,
    )

    assert result == expected
    assert result == sorted(set(result))


def test_unknown_forget_id_is_rejected(
    training_features,
):
    manifest = create_shard_manifest(
        training_features,
        SISAConfig(),
    )

    with pytest.raises(
        KeyError,
        match="Unknown forget IDs",
    ):
        find_affected_shards(
            manifest,
            ["missing-user"],
        )


def test_assignments_can_be_attached(
    training_features,
):
    manifest = create_shard_manifest(
        training_features,
        SISAConfig(),
    )

    assigned = attach_shard_assignments(
        training_features,
        manifest,
    )

    assert len(assigned) == len(training_features)
    assert assigned["shard_id"].notna().all()
    assert assigned["record_id"].is_unique


@pytest.mark.parametrize(
    "number_of_shards",
    [0, 1],
)
def test_invalid_shard_count_is_rejected(
    number_of_shards,
):
    with pytest.raises(ValueError):
        SISAConfig(
            number_of_shards=number_of_shards
        )