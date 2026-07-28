from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd


@dataclass(frozen=True)
class DeletionRequest:
    request_id: str
    record_ids: tuple[str, ...]
    requested_at: str


def create_deletion_request(
    request_id: str,
    record_ids: list[str],
) -> DeletionRequest:
    if not request_id.strip():
        raise ValueError("request_id cannot be empty")

    unique_ids = tuple(dict.fromkeys(record_ids))

    if not unique_ids:
        raise ValueError("At least one record ID is required")

    return DeletionRequest(
        request_id=request_id,
        record_ids=unique_ids,
        requested_at=datetime.now(timezone.utc).isoformat(),
    )


def apply_deletion_request(
    features: pd.DataFrame,
    target: pd.Series,
    request: DeletionRequest,
) -> tuple[pd.DataFrame, pd.Series]:
    requested_ids = set(request.record_ids)
    available_ids = set(features["record_id"])

    unknown_ids = requested_ids - available_ids
    if unknown_ids:
        raise KeyError(f"Unknown record IDs: {sorted(unknown_ids)}")

    retain_mask = ~features["record_id"].isin(requested_ids)

    retained_features = features.loc[retain_mask].reset_index(drop=True)
    retained_target = target.loc[retain_mask].reset_index(drop=True)

    return retained_features, retained_target