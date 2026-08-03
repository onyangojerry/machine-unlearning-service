import json

import pytest

from unlearning.config import (
    experiment_config_from_mapping,
    load_experiment_config,
)


@pytest.fixture
def valid_values():
    return {
        "dataset_name": "adult",
        "dataset_version": 2,
        "seed": 42,
        "test_fraction": 0.2,
        "forget_fraction": 0.01,
        "number_of_shards": 5,
        "number_of_slices": 1,
        "calibration_size": 1000,
        "bootstrap_iterations": 2000,
    }


def test_valid_configuration_is_loaded(
    tmp_path,
    valid_values,
):
    path = tmp_path / "experiment.json"
    path.write_text(
        json.dumps(valid_values),
        encoding="utf-8",
    )

    config = load_experiment_config(path)

    assert config.dataset_name == "adult"
    assert config.seed == 42
    assert config.number_of_shards == 5


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("dataset_name", "", "dataset_name"),
        ("dataset_version", 0, "dataset_version"),
        ("seed", -1, "seed"),
        ("test_fraction", 0.0, "test_fraction"),
        ("test_fraction", 1.0, "test_fraction"),
        ("forget_fraction", 0.0, "forget_fraction"),
        ("forget_fraction", 1.0, "forget_fraction"),
        ("number_of_shards", 1, "number_of_shards"),
        ("number_of_slices", 0, "number_of_slices"),
        ("calibration_size", 0, "calibration_size"),
        (
            "bootstrap_iterations",
            99,
            "bootstrap_iterations",
        ),
    ],
)
def test_invalid_values_are_rejected(
    valid_values,
    field,
    value,
    message,
):
    invalid = {
        **valid_values,
        field: value,
    }

    with pytest.raises(ValueError, match=message):
        experiment_config_from_mapping(invalid)


def test_missing_field_is_rejected(valid_values):
    valid_values.pop("seed")

    with pytest.raises(
        ValueError,
        match="Missing configuration fields",
    ):
        experiment_config_from_mapping(valid_values)


def test_unknown_field_is_rejected(valid_values):
    valid_values["secret_setting"] = True

    with pytest.raises(
        ValueError,
        match="Unknown configuration fields",
    ):
        experiment_config_from_mapping(valid_values)


def test_missing_file_is_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_experiment_config(
            tmp_path / "missing.json"
        )


def test_malformed_json_is_rejected(tmp_path):
    path = tmp_path / "invalid.json"
    path.write_text("{not-json", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="Invalid JSON",
    ):
        load_experiment_config(path)


def test_non_object_json_is_rejected(tmp_path):
    path = tmp_path / "list.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="JSON object",
    ):
        load_experiment_config(path)