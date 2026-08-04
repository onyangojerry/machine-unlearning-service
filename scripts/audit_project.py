from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

from unlearning.audit import audit_project
from unlearning.settings import (
    get_experiment_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "acceptance_audit.json"
)


def main() -> None:
    config = get_experiment_config()

    checks = audit_project(
        PROJECT_ROOT,
        number_of_shards=config.number_of_shards,
    )

    result = {
        "passed": all(check.passed for check in checks),
        "checks": [
            asdict(check)
            for check in checks
        ],
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    OUTPUT_PATH.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    for check in checks:
        status = "PASS" if check.passed else "FAIL"
        print(
            f"[{status}] {check.name}: {check.detail}"
        )

    print(f"\nAudit written to: {OUTPUT_PATH}")

    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()