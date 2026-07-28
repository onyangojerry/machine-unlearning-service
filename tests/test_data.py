from unlearning.data import (
    add_stable_record_ids,
    create_splits,
    load_adult_dataset,
)


def test_record_ids_are_unique_and_complete():
    X, _ = load_adult_dataset()
    identified = add_stable_record_ids(X)

    assert identified["record_id"].notna().all()
    assert identified["record_id"].is_unique
    assert identified["record_id"].str.startswith("adult-v2-").all()


def test_splits_do_not_overlap():
    X, y = load_adult_dataset()
    X = add_stable_record_ids(X)

    splits = create_splits(X, y)

    train_ids = set(splits.X_train["record_id"])
    test_ids = set(splits.X_test["record_id"])

    assert train_ids.isdisjoint(test_ids)


def test_splits_are_reproducible():
    X, y = load_adult_dataset()
    X = add_stable_record_ids(X)

    first = create_splits(X, y, seed=42)
    second = create_splits(X, y, seed=42)

    assert first.X_train["record_id"].tolist() == (
        second.X_train["record_id"].tolist()
    )