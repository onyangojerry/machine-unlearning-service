from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def encode_target(target) -> np.ndarray:
    mapping = {
        "<=50K": 0,
        ">50K": 1,
    }

    encoded = target.astype(str).str.strip().map(mapping)

    if encoded.isna().any():
        unknown = sorted(target[encoded.isna()].astype(str).unique())
        raise ValueError(f"Unknown target labels: {unknown}")

    return encoded.to_numpy(dtype=int)



def evaluate_binary_classifier(
    model,
    features,
    target,
) -> dict[str, float]:
    labels = encode_target(target)
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(
            precision_score(labels, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(labels, predictions, zero_division=0)
        ),
        "f1": float(
            f1_score(labels, predictions, zero_division=0)
        ),
        "log_loss": float(
            log_loss(labels, probabilities, labels=[0, 1])
        ),
    }

    metrics["roc_auc"] = (
        float(roc_auc_score(labels, probabilities))
        if len(np.unique(labels)) == 2
        else float("nan")
    )

    return metrics