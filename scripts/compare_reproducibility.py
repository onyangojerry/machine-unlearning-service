from __future__ import annotations

import argparse

from unlearning.reproducibility import (
    compare_result_directories,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("first_run")
    parser.add_argument("second_run")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    comparisons = compare_result_directories(
        args.first_run,
        args.second_run,
    )

    for path, identical in comparisons.items():
        status = "PASS" if identical else "FAIL"
        print(f"[{status}] {path}")

    if not all(comparisons.values()):
        raise SystemExit(1)

    print(
        "\nAll non-runtime experimental results match."
    )


if __name__ == "__main__":
    main()