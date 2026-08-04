from unlearning.reproducibility import (
    is_runtime_key,
    normalize_result,
)


def test_runtime_fields_are_recognized():
    assert is_runtime_key("training_seconds")
    assert is_runtime_key(
        "reference_retraining_seconds"
    )
    assert is_runtime_key("observed_speedup")
    assert not is_runtime_key("accuracy")


def test_nested_runtime_values_are_removed():
    value = {
        "accuracy": 0.85,
        "training_seconds": 1.2,
        "nested": {
            "f1": 0.65,
            "total_wall_seconds": 3.0,
        },
    }

    assert normalize_result(value) == {
        "accuracy": 0.85,
        "nested": {
            "f1": 0.65,
        },
    }


def test_lists_are_normalized_recursively():
    value = [
        {
            "accuracy": 0.8,
            "training_seconds": 1.0,
        }
    ]

    assert normalize_result(value) == [
        {
            "accuracy": 0.8,
        }
    ]