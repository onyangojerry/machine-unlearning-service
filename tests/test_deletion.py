import pandas as pd
import pytest

from unlearning.deletion import (
    apply_deletion_request,
    create_deletion_request,
)


def sample_data():
    features = pd.DataFrame(
        {
            "record_id": ["user-1", "user-2", "user-3"],
            "age": [25, 35, 45],
        }
    )
    target = pd.Series([0, 1, 1])
    return features, target


def test_requested_record_is_removed():
    X, y = sample_data()
    request = create_deletion_request("delete-001", ["user-2"])

    retained_X, retained_y = apply_deletion_request(X, y, request)

    assert "user-2" not in set(retained_X["record_id"])
    assert len(retained_X) == 2
    assert len(retained_y) == 2


def test_unknown_record_is_rejected():
    X, y = sample_data()
    request = create_deletion_request("delete-002", ["missing-user"])

    with pytest.raises(KeyError):
        apply_deletion_request(X, y, request)


def test_duplicate_requested_ids_are_deduplicated():
    request = create_deletion_request(
        "delete-003",
        ["user-1", "user-1"],
    )

    assert request.record_ids == ("user-1",)