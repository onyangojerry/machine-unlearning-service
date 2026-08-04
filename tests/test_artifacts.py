import json

import pytest

from unlearning.artifacts import (
    ArtifactValidationError,
    validate_stage_artifacts,
)


def write_json(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(values),
        encoding="utf-8",
    )


def test_baseline_artifacts_are_validated(tmp_path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()

    (artifacts / "original_model.joblib").write_bytes(
        b"model"
    )
    write_json(
        artifacts / "baseline_metrics.json",
        {
            "accuracy": 0.85,
            "f1": 0.66,
            "training_seconds": 1.0,
            "training_records": 100,
        },
    )
    write_json(
        artifacts / "forget_manifest.json",
        {
            "record_count": 1,
            "record_ids": ["user-1"],
        },
    )

    validated = validate_stage_artifacts(
        "baseline",
        tmp_path,
    )

    assert len(validated) == 3


def test_missing_artifact_fails(tmp_path):
    with pytest.raises(
        ArtifactValidationError,
        match="missing",
    ):
        validate_stage_artifacts(
            "baseline",
            tmp_path,
        )


def test_empty_artifact_fails(tmp_path):
    path = tmp_path / "artifacts"
    path.mkdir()
    (path / "original_model.joblib").write_bytes(b"")

    with pytest.raises(
        ArtifactValidationError,
        match="empty",
    ):
        validate_stage_artifacts(
            "baseline",
            tmp_path,
        )


def test_missing_json_key_fails(tmp_path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()

    (artifacts / "original_model.joblib").write_bytes(
        b"model"
    )
    write_json(
        artifacts / "baseline_metrics.json",
        {"accuracy": 0.85},
    )
    write_json(
        artifacts / "forget_manifest.json",
        {
            "record_count": 1,
            "record_ids": ["user-1"],
        },
    )

    with pytest.raises(
        ArtifactValidationError,
        match="missing keys",
    ):
        validate_stage_artifacts(
            "baseline",
            tmp_path,
        )


def test_unknown_stage_fails(tmp_path):
    with pytest.raises(
        ValueError,
        match="Unknown pipeline stage",
    ):
        validate_stage_artifacts(
            "unknown",
            tmp_path,
        )