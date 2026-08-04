from unlearning.audit import (
    AuditCheck,
    _count_unresolved_todos,
    _rate_is_valid,
)


def test_probability_rate_boundaries_are_valid():
    assert _rate_is_valid(0.0)
    assert _rate_is_valid(0.5)
    assert _rate_is_valid(1.0)


def test_out_of_range_rates_are_invalid():
    assert not _rate_is_valid(-0.01)
    assert not _rate_is_valid(1.01)
    assert not _rate_is_valid(None)


def test_audit_check_is_immutable():
    check = AuditCheck(
        name="example",
        passed=True,
        detail="valid",
    )

    assert check.name == "example"
    assert check.passed is True


def test_report_todos_are_read_from_articles(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "articles.md").write_text(
        "Complete\nTODO: revise\n",
        encoding="utf-8",
    )

    assert _count_unresolved_todos(tmp_path) == 1
