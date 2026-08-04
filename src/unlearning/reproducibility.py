from __future__ import annotations

import json
from pathlib import Path
from typing import Any


RESULT_FILES = (
    Path("artifacts/baseline_metrics.json"),
    Path("artifacts/forget_manifest.json"),
    Path("artifacts/exact_unlearning_results.json"),
    Path("artifacts/sisa_training_results.json"),
    Path(
        "artifacts/selective_single/"
        "selective_results.json"
    ),
    Path("artifacts/privacy_results.json"),
)


def is_runtime_key(key: str) -> bool:
    return (
        key.endswith("_seconds")
        or "training_seconds" in key
        or "retraining_seconds" in key
        or "speedup" in key
    )


def normalize_result(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: normalize_result(item)
            for key, item in sorted(value.items())
            if not is_runtime_key(key)
        }

    if isinstance(value, list):
        return [
            normalize_result(item)
            for item in value
        ]

    return value


def load_normalized_result(path: Path) -> Any:
    value = json.loads(
        path.read_text(encoding="utf-8")
    )
    return normalize_result(value)


def compare_result_directories(
    first_root: str | Path,
    second_root: str | Path,
) -> dict[str, bool]:
    first_root = Path(first_root)
    second_root = Path(second_root)

    comparisons = {}

    for relative_path in RESULT_FILES:
        first_path = first_root / relative_path
        second_path = second_root / relative_path

        if not first_path.is_file():
            raise FileNotFoundError(first_path)

        if not second_path.is_file():
            raise FileNotFoundError(second_path)

        comparisons[str(relative_path)] = (
            load_normalized_result(first_path)
            == load_normalized_result(second_path)
        )

    return comparisons