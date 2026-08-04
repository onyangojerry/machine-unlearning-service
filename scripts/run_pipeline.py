from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys

from unlearning.artifacts import (
    validate_stage_artifacts,
)
from unlearning.settings import get_experiment_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class PipelineStage:
    name: str
    command: tuple[str, ...]
    artifact_stage: str | None


STAGES = (
    PipelineStage(
        name="tests",
        command=(
            sys.executable,
            "-m",
            "pytest",
            "-m",
            "not integration",
            "-v",
        ),
        artifact_stage=None,
    ),
    PipelineStage(
        name="baseline",
        command=(
            sys.executable,
            "-m",
            "unlearning.train_baseline",
        ),
        artifact_stage="baseline",
    ),
    PipelineStage(
        name="reference",
        command=(
            sys.executable,
            "-m",
            "unlearning.train_reference",
        ),
        artifact_stage="reference",
    ),
    PipelineStage(
        name="sisa",
        command=(
            sys.executable,
            "-m",
            "unlearning.train_sisa",
        ),
        artifact_stage="sisa",
    ),
    PipelineStage(
        name="selective-plan",
        command=(
            sys.executable,
            "-m",
            "unlearning.plan_selective_unlearning",
        ),
        artifact_stage="selective-plan",
    ),
    PipelineStage(
        name="selective",
        command=(
            sys.executable,
            "-m",
            "unlearning.run_selective_unlearning",
        ),
        artifact_stage="selective",
    ),
    PipelineStage(
        name="privacy",
        command=(
            sys.executable,
            "-m",
            "unlearning.run_privacy_evaluation",
        ),
        artifact_stage="privacy",
    ),
)


def select_stages(
    start_at: str,
    stop_after: str,
) -> tuple[PipelineStage, ...]:
    names = [stage.name for stage in STAGES]
    start_index = names.index(start_at)
    stop_index = names.index(stop_after)

    if start_index > stop_index:
        raise ValueError(
            "start-at stage must precede stop-after stage"
        )

    return STAGES[start_index : stop_index + 1]


def run_stage(
    stage: PipelineStage,
    environment: dict[str, str],
) -> None:
    print(f"\n=== Running {stage.name} ===", flush=True)

    subprocess.run(
        stage.command,
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
    )

    if stage.artifact_stage is None:
        return

    config = get_experiment_config()

    validated = validate_stage_artifacts(
        stage.artifact_stage,
        PROJECT_ROOT,
        number_of_shards=config.number_of_shards,
    )

    print(
        f"Validated {len(validated)} artifact(s) "
        f"for {stage.name}.",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    names = [stage.name for stage in STAGES]

    parser = argparse.ArgumentParser(
        description=(
            "Run the auditable unlearning pipeline."
        )
    )
    parser.add_argument(
        "--start-at",
        choices=names,
        default=names[0],
    )
    parser.add_argument(
        "--stop-after",
        choices=names,
        default=names[-1],
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        stages = select_stages(
            args.start_at,
            args.stop_after,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error

    environment = dict(os.environ)
    source_path = str(PROJECT_ROOT / "src")
    existing = environment.get("PYTHONPATH")

    environment["PYTHONPATH"] = (
        source_path
        if not existing
        else source_path + os.pathsep + existing
    )
    environment.setdefault(
        "UNLEARNING_CONFIG",
        str(
            PROJECT_ROOT
            / "configs"
            / "week1.json"
        ),
    )
    environment.setdefault("MPLBACKEND", "Agg")
    environment.setdefault(
        "MLFLOW_TRACKING_URI",
        "sqlite:///mlflow.db",
    )

    for stage in stages:
        run_stage(stage, environment)

    print(
        "\nPipeline completed successfully.",
        flush=True,
    )


if __name__ == "__main__":
    main()
