"""
tests/test_validator.py - Unit tests for schedule validation and date calculations.
"""

import datetime as dt
import pytest
from schema import (
    DifficultyLevel,
    DayPlan,
    StudyPlanRequest,
    StudySchedule,
    StudySession,
    Subject,
    Topic,
    ValidationResult,
)
from validator import (
    calculate_available_hours,
    format_validation_report,
    get_available_hours,
    validate_schedule,
)


@pytest.fixture
def sample_request() -> StudyPlanRequest:
    """Fixture providing a standard user study plan request."""
    return StudyPlanRequest(
        subjects=[
            Subject(
                name="Mathematics",
                exam_date=dt.date(2026, 11, 10),
                topics=[
                    Topic(name="Calculus", difficulty=DifficultyLevel.HARD),
                    Topic(name="Algebra", difficulty=DifficultyLevel.MEDIUM),
                ],
            ),
            Subject(
                name="Physics",
                exam_date=dt.date(2026, 11, 12),
                topics=[
                    Topic(name="Optics", difficulty=DifficultyLevel.EASY),
                ],
            ),
        ],
        start_date=dt.date(2026, 11, 1),
        daily_max_hours=4.0,
    )


def test_calculate_available_hours_standard():
    """Nov 1 to Nov 5 has 4 study days before exam day. With 3 hrs/day = 12 hrs."""
    hours = calculate_available_hours("2026-11-01", "2026-11-05", 3.0)
    assert hours == 12.0


def test_calculate_available_hours_invalid_range():
    """If exam date is on or before start date, available hours should be 0."""
    hours = calculate_available_hours("2026-11-05", "2026-11-05", 4.0)
    assert hours == 0.0
    hours_past = calculate_available_hours("2026-11-10", "2026-11-05", 4.0)
    assert hours_past == 0.0


def test_get_available_hours_tool_output():
    """Tool returns informative string with correct calculations."""
    report = get_available_hours("2026-11-01", "2026-11-06", 2.5)
    assert "5 study days" in report
    assert "12.5 hours" in report

    error_report = get_available_hours("2026-11-10", "2026-11-05", 3.0)
    assert "Invalid date range" in error_report


def test_validate_schedule_valid(sample_request):
    """A well-formed schedule covering all topics before exam dates within hour limits."""
    schedule = StudySchedule(
        days=[
            DayPlan(
                date=dt.date(2026, 11, 1),
                sessions=[
                    StudySession(subject="Mathematics", topic="Calculus", hours=3.0),
                ],
            ),
            DayPlan(
                date=dt.date(2026, 11, 2),
                sessions=[
                    StudySession(subject="Mathematics", topic="Algebra", hours=2.0),
                    StudySession(subject="Physics", topic="Optics", hours=1.5),
                ],
            ),
        ],
        summary="Balanced plan for Math and Physics.",
    )

    result = validate_schedule(schedule, sample_request)
    assert result.is_valid is True
    assert len(result.errors) == 0


def test_validate_schedule_exceeds_daily_limit(sample_request):
    """Fails when study hours on a day exceed daily_max_hours (4.0 hrs)."""
    schedule = StudySchedule(
        days=[
            DayPlan(
                date=dt.date(2026, 11, 1),
                sessions=[
                    StudySession(subject="Mathematics", topic="Calculus", hours=3.0),
                    StudySession(subject="Mathematics", topic="Algebra", hours=2.5),  # total 5.5h > 4.0h
                ],
            ),
            DayPlan(
                date=dt.date(2026, 11, 2),
                sessions=[
                    StudySession(subject="Physics", topic="Optics", hours=1.0),
                ],
            ),
        ]
    )

    result = validate_schedule(schedule, sample_request)
    assert result.is_valid is False
    assert any("exceeds the daily limit" in err for err in result.errors)


def test_validate_schedule_scheduled_on_or_after_exam_date(sample_request):
    """Fails if a topic is scheduled on or after the subject exam date."""
    # Math exam is 2026-11-10
    schedule = StudySchedule(
        days=[
            DayPlan(
                date=dt.date(2026, 11, 10),  # On exam day!
                sessions=[
                    StudySession(subject="Mathematics", topic="Calculus", hours=2.0),
                ],
            ),
            DayPlan(
                date=dt.date(2026, 11, 2),
                sessions=[
                    StudySession(subject="Mathematics", topic="Algebra", hours=2.0),
                    StudySession(subject="Physics", topic="Optics", hours=1.0),
                ],
            ),
        ]
    )

    result = validate_schedule(schedule, sample_request)
    assert result.is_valid is False
    assert any("on or after the exam date" in err for err in result.errors)


def test_validate_schedule_before_start_date(sample_request):
    """Fails if a session is scheduled before the user's start_date (2026-11-01)."""
    schedule = StudySchedule(
        days=[
            DayPlan(
                date=dt.date(2026, 10, 31),  # Before start_date
                sessions=[
                    StudySession(subject="Mathematics", topic="Calculus", hours=2.0),
                ],
            ),
            DayPlan(
                date=dt.date(2026, 11, 2),
                sessions=[
                    StudySession(subject="Mathematics", topic="Algebra", hours=2.0),
                    StudySession(subject="Physics", topic="Optics", hours=1.0),
                ],
            ),
        ]
    )

    result = validate_schedule(schedule, sample_request)
    assert result.is_valid is False
    assert any("before your chosen start date" in err for err in result.errors)


def test_validate_schedule_missing_topic(sample_request):
    """Fails if any topic from the syllabus was omitted in the timetable."""
    # Omitted 'Optics' in Physics
    schedule = StudySchedule(
        days=[
            DayPlan(
                date=dt.date(2026, 11, 1),
                sessions=[
                    StudySession(subject="Mathematics", topic="Calculus", hours=2.0),
                    StudySession(subject="Mathematics", topic="Algebra", hours=2.0),
                ],
            ),
        ]
    )

    result = validate_schedule(schedule, sample_request)
    assert result.is_valid is False
    assert any("Missing topic: 'Optics'" in err for err in result.errors)


def test_validate_schedule_empty(sample_request):
    """Fails if schedule has no days."""
    schedule = StudySchedule(days=[])
    result = validate_schedule(schedule, sample_request)
    assert result.is_valid is False
    assert "Schedule is empty" in result.errors[0]


def test_format_validation_report():
    """Report formatting correctly marks PASS and FAIL states."""
    passed = ValidationResult(is_valid=True, errors=[])
    assert "VALIDATION PASSED" in format_validation_report(passed)

    failed = ValidationResult(is_valid=False, errors=["Error 1", "Error 2"])
    report = format_validation_report(failed)
    assert "VALIDATION FAILED" in report
    assert "- Error 1" in report
    assert "- Error 2" in report
