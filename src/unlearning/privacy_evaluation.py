from __future__ import annotations

from dataclasses import asdict

import numpy as np

from unlearning.membership import (
    bootstrap_member_rate_interval,
    fit_membership_threshold,
    membership_scores,
    predicted_member_rate,
)


def evaluate_model_privacy(
    model_name: str,
    model,
    calibration_member_X,
    calibration_member_y,
    calibration_nonmember_X,
    calibration_nonmember_y,
    forget_X,
    forget_y,
    bootstrap_iterations: int = 2000,
    seed: int = 42,
) -> dict:
    member_scores = membership_scores(
        model,
        calibration_member_X,
        calibration_member_y,
    )

    nonmember_scores = membership_scores(
        model,
        calibration_nonmember_X,
        calibration_nonmember_y,
    )

    attack = fit_membership_threshold(
        member_scores,
        nonmember_scores,
    )

    forget_scores = membership_scores(
        model,
        forget_X,
        forget_y,
    )

    member_rate = predicted_member_rate(
        forget_scores,
        attack.threshold,
    )

    interval = bootstrap_member_rate_interval(
        forget_scores,
        attack.threshold,
        iterations=bootstrap_iterations,
        seed=seed,
    )

    return {
        "model_name": model_name,
        "calibration_attack": asdict(attack),
        "forget_record_count": len(forget_scores),
        "forget_mean_membership_score": float(
            np.mean(forget_scores)
        ),
        "forget_median_membership_score": float(
            np.median(forget_scores)
        ),
        "forget_predicted_member_rate": member_rate,
        "forget_member_rate_ci_95": (
            list(interval)
            if interval is not None
            else None
        ),
    }