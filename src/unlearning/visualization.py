from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_probability_shift(
    original_model,
    reference_model,
    features,
    output_path: str | Path,
) -> None:
    original_probabilities = original_model.predict_proba(features)[:, 1]
    reference_probabilities = reference_model.predict_proba(features)[:, 1]

    shifts = reference_probabilities - original_probabilities

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(8, 5))

    axis.hist(
        shifts,
        bins=30,
        color="#355C7D",
        edgecolor="white",
    )
    axis.axvline(0, color="black", linestyle="--", linewidth=1)

    axis.set_title("Prediction Shift on Forgotten Records")
    axis.set_xlabel(
        "Reference probability − original probability"
    )
    axis.set_ylabel("Forgotten record count")

    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)