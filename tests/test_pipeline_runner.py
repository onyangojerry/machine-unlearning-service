import pytest

from scripts.run_pipeline import select_stages


def test_complete_stage_order():
    selected = select_stages(
        "tests",
        "privacy",
    )

    assert [stage.name for stage in selected] == [
        "tests",
        "baseline",
        "reference",
        "sisa",
        "selective-plan",
        "selective",
        "privacy",
    ]


def test_partial_pipeline_selection():
    selected = select_stages(
        "sisa",
        "selective-plan",
    )

    assert [stage.name for stage in selected] == [
        "sisa",
        "selective-plan",
    ]


def test_reverse_stage_range_is_rejected():
    with pytest.raises(
        ValueError,
        match="must precede",
    ):
        select_stages(
            "privacy",
            "baseline",
        )