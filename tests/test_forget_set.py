import pandas as pd
import pytest

from unlearning.forget_set import select_forget_ids


def training_features(size: int = 100) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "record_id": [f"user-{i}" for i in range(size)],
            "value": range(size),
        }
    )


def test_forget_selection_is_reproducible():
    features = training_features()

    first = select_forget_ids(features, fraction=0.10, seed=42)
    second = select_forget_ids(features, fraction=0.10, seed=42)

    assert first == second
    assert len(first) == 10


def test_forget_ids_are_unique():
    selected = select_forget_ids(
        training_features(),
        fraction=0.10,
        seed=42,
    )

    assert len(selected) == len(set(selected))


def test_invalid_fraction_is_rejected():
    with pytest.raises(ValueError):
        select_forget_ids(training_features(), fraction=1.0)