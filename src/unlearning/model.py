from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def split_feature_types(
    features: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    usable = features.drop(columns=["record_id"], errors="ignore")

    categorical = usable.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    numerical = usable.select_dtypes(
        include=["number"]
    ).columns.tolist()

    return categorical, numerical


def build_model_pipeline(
    features: pd.DataFrame,
    seed: int = 42,
) -> Pipeline:
    categorical, numerical = split_feature_types(features)

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
            ),
        ]
    )

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipeline, categorical),
            ("numerical", numerical_pipeline, numerical),
        ],
        remainder="drop",
    )

    classifier = LogisticRegression(
        max_iter=1000,
        solver="liblinear",
        random_state=seed,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )