from pathlib import Path

import pandas as pd
import pytest

from unlearning.selective import (
    build_deletion_scenarios,
    build_retraining_plan,
    fingerprint_shard_artifacts,
    sha256_file,
)


@pytest.fixture
def shard_manifest():
    return pd.DataFrame(
        {
            "record_id": [
                "a", "b", "c", "d", "e", "f",
            ],
            "shard_id": [
                0, 0, 0, 1, 1, 2,
            ],
        }
    )


def test_single_record_affects_one_shard(
    shard_manifest,
):
    plan = build_retraining_plan(
        scenario_name="single_record",
        shard_manifest=shard_manifest,
        forget_ids=["b"],
        available_shards=[0, 1, 2],
    )

    assert plan.affected_shards == (0,)
    assert plan.unaffected_shards == (1, 2)
    assert plan.retraining_fraction == pytest.approx(
        1 / 3
    )


def test_distributed_request_affects_multiple_shards(
    shard_manifest,
):
    plan = build_retraining_plan(
        scenario_name="distributed",
        shard_manifest=shard_manifest,
        forget_ids=["a", "d", "f"],
        available_shards=[0, 1, 2],
    )

    assert plan.affected_shards == (0, 1, 2)
    assert plan.unaffected_shards == ()
    assert plan.retraining_fraction == 1.0


def test_duplicate_forget_ids_are_removed(
    shard_manifest,
):
    plan = build_retraining_plan(
        scenario_name="duplicates",
        shard_manifest=shard_manifest,
        forget_ids=["a", "a"],
        available_shards=[0, 1, 2],
    )

    assert plan.forget_ids == ("a",)
    assert plan.affected_shards == (0,)


def test_scenarios_include_three_workloads(
    shard_manifest,
):
    scenarios = build_deletion_scenarios(
        shard_manifest,
        distributed_forget_ids=["a", "d", "f"],
        same_shard_count=2,
    )

    assert set(scenarios) == {
        "single_record",
        "same_shard_batch",
        "distributed_batch",
    }

    assert scenarios["single_record"] == ["a"]
    assert len(scenarios["same_shard_batch"]) == 2
    assert scenarios["distributed_batch"] == [
        "a", "d", "f"
    ]


def test_same_shard_scenario_uses_one_shard(
    shard_manifest,
):
    scenarios = build_deletion_scenarios(
        shard_manifest,
        distributed_forget_ids=["a", "d", "f"],
        same_shard_count=3,
    )

    selected = scenarios["same_shard_batch"]

    selected_shards = shard_manifest.loc[
        shard_manifest["record_id"].isin(selected),
        "shard_id",
    ].unique()

    assert len(selected_shards) == 1


def test_file_hash_changes_when_content_changes(
    tmp_path: Path,
):
    artifact = tmp_path / "model.joblib"
    artifact.write_bytes(b"first model")

    first_hash = sha256_file(artifact)

    artifact.write_bytes(b"second model")
    second_hash = sha256_file(artifact)

    assert first_hash != second_hash


def test_identical_files_have_identical_hashes(
    tmp_path: Path,
):
    first = tmp_path / "first.joblib"
    second = tmp_path / "second.joblib"

    first.write_bytes(b"same model")
    second.write_bytes(b"same model")

    assert sha256_file(first) == sha256_file(second)


def test_missing_artifact_is_rejected(
    tmp_path: Path,
):
    with pytest.raises(FileNotFoundError):
        sha256_file(tmp_path / "missing.joblib")


def test_shard_artifacts_are_fingerprinted(
    tmp_path: Path,
):
    (tmp_path / "shard_0.joblib").write_bytes(b"zero")
    (tmp_path / "shard_1.joblib").write_bytes(b"one")

    fingerprints = fingerprint_shard_artifacts(
        tmp_path,
        [0, 1],
    )

    assert set(fingerprints) == {0, 1}
    assert fingerprints[0] != fingerprints[1]