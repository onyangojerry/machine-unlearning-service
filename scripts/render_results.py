from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports" / "generated_results.md"


def load(relative_path: str) -> dict:
    return json.loads(
        (ROOT / relative_path).read_text(
            encoding="utf-8"
        )
    )


def number(value, digits: int = 6) -> str:
    if value is None:
        return "N/A"

    if isinstance(value, bool):
        return "Yes" if value else "No"

    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"

    return str(value)


def main() -> None:
    baseline = load(
        "artifacts/baseline_metrics.json"
    )
    exact = load(
        "artifacts/exact_unlearning_results.json"
    )
    sisa = load(
        "artifacts/sisa_training_results.json"
    )
    selective = load(
        "artifacts/selective_single/"
        "selective_results.json"
    )
    privacy = load(
        "artifacts/privacy_results.json"
    )

    reference = exact["reference_test_metrics"]
    ensemble = sisa["ensemble_test_metrics"]
    comparison = selective[
        "selective_vs_reference_test"
    ]

    original_privacy = privacy[
        "distributed_track"
    ]["original"]
    reference_privacy = privacy[
        "distributed_track"
    ]["reference"]

    lines = [
        "# Generated experimental results",
        "",
        "## Utility",
        "",
        "| Metric | Original | Exact reference "
        "| Sharded ensemble |",
        "|---|---:|---:|---:|",
    ]

    for metric in (
        "accuracy",
        "f1",
        "roc_auc",
        "log_loss",
    ):
        lines.append(
            f"| {metric} "
            f"| {number(baseline[metric])} "
            f"| {number(reference[metric])} "
            f"| {number(ensemble[metric])} |"
        )

    lines.extend(
        [
            "",
            "## Selective unlearning",
            "",
            "| Measurement | Result |",
            "|---|---:|",
            (
                "| Affected shards | "
                f"{len(selective['affected_shards'])} |"
            ),
            (
                "| Selective retraining seconds | "
                f"{number(selective['selective_retraining_seconds'])} |"
            ),
            (
                "| Full sharded retraining seconds | "
                f"{number(selective['full_sharded_retraining_seconds'])} |"
            ),
            (
                "| Observed speedup | "
                f"{number(selective['observed_speedup'])}× |"
            ),
            (
                "| Test prediction disagreement | "
                f"{number(comparison['prediction_disagreement_rate'])} |"
            ),
            (
                "| Test mean probability gap | "
                f"{number(comparison['mean_absolute_probability_gap'])} |"
            ),
            (
                "| Unaffected hashes unchanged | "
                f"{number(selective['unaffected_hashes_unchanged'])} |"
            ),
            "",
            "## Membership-inference evaluation",
            "",
            "| Measurement | Original | Exact reference |",
            "|---|---:|---:|",
            (
                "| Attack ROC-AUC | "
                f"{number(original_privacy['calibration_attack']['roc_auc'])} "
                f"| {number(reference_privacy['calibration_attack']['roc_auc'])} |"
            ),
            (
                "| Forget-set predicted-member rate | "
                f"{number(original_privacy['forget_predicted_member_rate'])} "
                f"| {number(reference_privacy['forget_predicted_member_rate'])} |"
            ),
        ]
    )

    OUTPUT.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print(f"Results written to: {OUTPUT}")


if __name__ == "__main__":
    main()