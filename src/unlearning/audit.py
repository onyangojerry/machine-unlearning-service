from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from unlearning.artifacts import (
    ArtifactValidationError,
    CONTRACTS,
    validate_stage_artifacts,
)


@dataclass(frozen=True)
class AuditCheck:
    name: str
    passed: bool
    detail: str


def _load_json(path: Path) -> dict:
    value = json.loads(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(value, dict):
        raise ValueError(
            f"Expected a JSON object: {path}"
        )

    return value


def _rate_is_valid(value) -> bool:
    return (
        isinstance(value, (int, float))
        and 0.0 <= float(value) <= 1.0
    )


def _count_unresolved_todos(project_root: Path) -> int:
    article_path = (
        project_root / "reports" / "articles.md"
    )
    article = article_path.read_text(encoding="utf-8")
    return article.count("TODO")


def audit_project(
    project_root: str | Path,
    number_of_shards: int,
) -> list[AuditCheck]:
    project_root = Path(project_root).resolve()
    checks = []

    for stage in CONTRACTS:
        try:
            validated = validate_stage_artifacts(
                stage,
                project_root,
                number_of_shards=number_of_shards,
            )
        except (
            ArtifactValidationError,
            ValueError,
        ) as error:
            checks.append(
                AuditCheck(
                    name=f"artifacts:{stage}",
                    passed=False,
                    detail=str(error),
                )
            )
        else:
            checks.append(
                AuditCheck(
                    name=f"artifacts:{stage}",
                    passed=True,
                    detail=(
                        f"Validated {len(validated)} "
                        "artifact(s)"
                    ),
                )
            )

    exact = _load_json(
        project_root
        / "artifacts"
        / "exact_unlearning_results.json"
    )

    accounting_correct = (
        exact["retain_records"]
        + exact["forget_records"]
        == exact["original_training_records"]
    )
    checks.append(
        AuditCheck(
            name="exact:record-accounting",
            passed=accounting_correct,
            detail=(
                f"retain={exact['retain_records']}, "
                f"forget={exact['forget_records']}, "
                f"original="
                f"{exact['original_training_records']}"
            ),
        )
    )

    selective = _load_json(
        project_root
        / "artifacts"
        / "selective_single"
        / "selective_results.json"
    )

    hashes_unchanged = (
        selective.get(
            "unaffected_hashes_unchanged"
        )
        is True
    )
    checks.append(
        AuditCheck(
            name="selective:unaffected-artifacts",
            passed=hashes_unchanged,
            detail=(
                "All unaffected shard hashes are unchanged"
                if hashes_unchanged
                else "One or more unaffected hashes changed"
            ),
        )
    )

    test_behavior = selective[
        "selective_vs_reference_test"
    ]
    disagreement = test_behavior[
        "prediction_disagreement_rate"
    ]
    mean_gap = test_behavior[
        "mean_absolute_probability_gap"
    ]

    matches_reference = (
        disagreement <= 1e-12
        and mean_gap <= 1e-12
    )
    checks.append(
        AuditCheck(
            name="selective:matches-reference",
            passed=matches_reference,
            detail=(
                f"disagreement={disagreement}, "
                f"mean_probability_gap={mean_gap}"
            ),
        )
    )

    speedup = selective.get("observed_speedup")
    valid_speedup = (
        isinstance(speedup, (int, float))
        and speedup > 0.0
    )
    checks.append(
        AuditCheck(
            name="selective:runtime-measured",
            passed=valid_speedup,
            detail=f"observed_speedup={speedup}",
        )
    )

    privacy = _load_json(
        project_root
        / "artifacts"
        / "privacy_results.json"
    )

    distributed = privacy["distributed_track"]

    for model_name in ("original", "reference"):
        evaluation = distributed[model_name]
        attack = evaluation["calibration_attack"]

        rate_valid = _rate_is_valid(
            evaluation[
                "forget_predicted_member_rate"
            ]
        )
        auc_valid = _rate_is_valid(
            attack["roc_auc"]
        )

        checks.append(
            AuditCheck(
                name=f"privacy:{model_name}",
                passed=rate_valid and auc_valid,
                detail=(
                    "member_rate="
                    f"{evaluation['forget_predicted_member_rate']}, "
                    f"attack_auc={attack['roc_auc']}"
                ),
            )
        )

    unresolved = _count_unresolved_todos(project_root)

    checks.append(
        AuditCheck(
            name="report:no-unresolved-todos",
            passed=unresolved == 0,
            detail=f"unresolved_todo_count={unresolved}",
        )
    )

    return checks
