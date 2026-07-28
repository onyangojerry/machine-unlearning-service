from dataclasses import dataclass

import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class DatasetSplits:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def load_adult_dataset() -> tuple[pd.DataFrame, pd.Series]:
    dataset = fetch_openml(
        name="adult",
        version=2,
        as_frame=True,
    )

    features = dataset.data.reset_index(drop=True).copy()
    target = dataset.target.reset_index(drop=True).copy()

    return features, target


def add_stable_record_ids(features: pd.DataFrame) -> pd.DataFrame:
    identified = features.reset_index(drop=True).copy()

    record_ids = [
        f"adult-v2-{index:06d}"
        for index in range(len(identified))
    ]

    identified.insert(0, "record_id", record_ids)
    return identified


def create_splits(
    features: pd.DataFrame,
    target: pd.Series,
    test_size: float = 0.20,
    seed: int = 42,
) -> DatasetSplits:
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=seed,
        stratify=target,
    )

    return DatasetSplits(
        X_train=X_train.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
    )