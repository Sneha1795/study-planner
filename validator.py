"""
validator.py - Schedule Validation and Date Arithmetic Tools.

This module provides:
1. `calculate_available_hours`: Exact Python arithmetic for study hours between dates.
2. `get_available_hours`: Tool wrapper for Gemini function calling.
3. `validate_schedule`: Rigorous independent schedule rule validation.
4. `format_validation_report`: Human-readable summary of validation results.
"""

import datetime as dt
from typing import List, Union
from schema import DayPlan, StudyPlanRequest, StudySchedule, StudySession, ValidationResult


def to_date(val: Union[dt.date, str]) -> dt.date:
    """Helper to convert a date string (YYYY-MM-DD) or dt.date into a dt.date object."""
    if isinstance(val, dt.date):
        return val
    return dt.date.fromisoformat(str(val).strip())


def calculate_available_hours(
    start_date: Union[dt.date, str],
    end_date: Union[dt.date, str],
    daily_max_hours: float
) -> float:
    """
    Calculates total study hours available between start_date and end_date.
    
    Rule: Studying happens strictly BEFORE the exam date.
    For example, if start_date is Nov 1 and exam is Nov 5:
    Study days = Nov 1, 2, 3, 4 (4 days).
    Total hours = 4 days * daily_max_hours.
    """
    start = to_date(start_date)
    end = to_date(end_date)
    days_count = (end - start).days

    if days_count <= 0:
        return 0.0

    return float(days_count * daily_max_hours)


def get_available_hours(start_date: str, end_date: str, daily_max_hours: float) -> str:
    """
    Tool for Gemini to calculate available study hours between two dates.
    
    Args:
        start_date: Start date string in YYYY-MM-DD format.
        end_date: End date (e.g. exam date) string in YYYY-MM-DD format.
        daily_max_hours: Maximum study hours available per day.
        
    Returns:
        A clear explanation of available days and total available study hours.
    """
    try:
        start = to_date(start_date)
        end = to_date(end_date)
        days = (end - start).days

        if days <= 0:
            return (
                f"Invalid date range: start_date ({start_date}) is on or after "
                f"end_date ({end_date}). Available study days: 0."
            )

        total_hours = calculate_available_hours(start, end, daily_max_hours)
        return (
            f"From {start_date} up to {end_date} (exclusive, before exam day), "
            f"there are {days} study days. With {daily_max_hours} hrs/day, "
            f"total available study capacity is {total_hours:.1f} hours."
        )
    except Exception as e:
        return f"Error calculating available hours: {str(e)}"


def validate_schedule(schedule: StudySchedule, request: StudyPlanRequest) -> ValidationResult:
    """
    Validates a generated study timetable against user constraints and academic rules.
    
    Rules enforced:
    1. Non-empty schedule: Schedule must have study sessions.
    2. Date bounds: No sessions before the requested start_date.
    3. Daily study limits: Total hours on any day must not exceed daily_max_hours.
    4. Exam date constraint: Topics must be scheduled strictly BEFORE their subject's exam date.
    5. Subject validity: Sessions must belong to recognized subjects.
    6. Topic coverage: All syllabus topics from the request should appear in the schedule.
    7. Positive hours: Each session must have hours > 0.
    """
    errors: List[str] = []
    warnings: List[str] = []

    # Check rule 1: Non-empty schedule
    if not schedule.days:
        errors.append("Schedule is empty: no study days were generated.")
        return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

    total_sessions_count = sum(len(day.sessions) for day in schedule.days)
    if total_sessions_count == 0:
        errors.append("Schedule has days listed, but no study sessions are assigned.")
        return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

    # Build lookup table for subject exam dates and topics
    subject_exam_dates = {s.name.lower(): s.exam_date for s in request.subjects}
    subject_canonical_names = {s.name.lower(): s.name for s in request.subjects}
    
    # Track which topics have been scheduled: (subject_name_lower, topic_name_lower)
    all_required_topics = set()
    for s in request.subjects:
        for t in s.topics:
            all_required_topics.add((s.name.lower(), t.name.lower()))
    
    scheduled_topics = set()

    # Track date ordering
    previous_date: Union[dt.date, None] = None

    for day in schedule.days:
        current_date = to_date(day.date)

        # Chronological warning check
        if previous_date and current_date < previous_date:
            warnings.append(f"Schedule dates are out of chronological order: {current_date} comes after {previous_date}.")
        previous_date = current_date

        # Rule 2: Date cannot be before the start date
        if current_date < request.start_date:
            errors.append(
                f"Date {current_date} is before your chosen start date ({request.start_date})."
            )

        # Rule 3: Daily study hour limit
        day_hours = day.total_hours
        if day_hours > request.daily_max_hours + 1e-4:  # small tolerance for floating point
            errors.append(
                f"Date {current_date}: Total scheduled hours ({day_hours:.1f}h) "
                f"exceeds the daily limit of {request.daily_max_hours:.1f}h."
            )

        for session in day.sessions:
            # Rule 7: Positive session hours
            if session.hours <= 0:
                errors.append(
                    f"Date {current_date}: Session for topic '{session.topic}' has invalid hours ({session.hours}h). "
                    "Hours must be greater than 0."
                )

            sub_key = session.subject.strip().lower()
            topic_key = session.topic.strip().lower()

            # Rule 5: Subject validity
            if sub_key not in subject_exam_dates:
                errors.append(
                    f"Date {current_date}: Subject '{session.subject}' was not in your subject list."
                )
                continue

            # Rule 4: Must be scheduled before the exam date
            exam_date = subject_exam_dates[sub_key]
            canonical_subject = subject_canonical_names[sub_key]

            if current_date >= exam_date:
                errors.append(
                    f"Date {current_date}: Topic '{session.topic}' ({canonical_subject}) is scheduled on or after "
                    f"the exam date ({exam_date}). Topics must be studied BEFORE the exam."
                )

            # Record scheduled topic
            scheduled_topics.add((sub_key, topic_key))

    # Rule 6: Check for any missing topics from syllabus
    for s in request.subjects:
        sub_key = s.name.lower()
        for t in s.topics:
            t_key = t.name.lower()
            if (sub_key, t_key) not in scheduled_topics:
                errors.append(
                    f"Missing topic: '{t.name}' under subject '{s.name}' was not scheduled in the timetable."
                )

    is_valid = len(errors) == 0
    return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings)


def format_validation_report(result: ValidationResult) -> str:
    """Generates a human-friendly text report of validation results."""
    if result.is_valid:
        report = "VALIDATION PASSED: The schedule meets all requirements and rules."
        if result.warnings:
            report += "\nWarnings:\n" + "\n".join(f"- {w}" for w in result.warnings)
        return report

    report = "VALIDATION FAILED: The following issues were found in the schedule:\n"
    report += "\n".join(f"- {e}" for e in result.errors)
    if result.warnings:
        report += "\nWarnings:\n" + "\n".join(f"- {w}" for w in result.warnings)
    return report
