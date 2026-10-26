"""
agent.py - Gemini-Powered Study Planner Agent.

This module implements an Agentic AI workflow using the Gemini API (google-genai SDK):
1. Takes the user's study plan request (subjects, topics, difficulties, dates, max daily hours).
2. Uses Gemini Function Calling to execute two real Python tools:
   - `get_available_hours`: Calculates available study hours before exam dates.
   - `check_schedule`: Validates draft schedules against academic rules.
3. Operates with a two-attempt limit:
   - Attempt 1: Generates an initial schedule using tools.
   - If validation fails, Attempt 2 provides the exact failure report and prompts Gemini to fix it.
4. Performs independent Python validation after the agent finishes. The final status
   always reflects the Python validator, never the model's self-assessment.
"""

import json
import os
import re
from typing import Any, Callable, Dict, List, Optional
import dotenv
from pydantic import BaseModel, Field

from google import genai
from google.genai import types

import schema
import validator

# Load environment variables (from .env file if present)
dotenv.load_dotenv()


class AgentStepLog(BaseModel):
    """Log record representing an action, tool call, or reasoning step taken by the agent."""
    step_type: str = Field(..., description="Type of step: thought, tool_call, tool_result, validation, attempt")
    message: str = Field(..., description="Human-readable description of what happened")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Optional payload or argument data")


class AgentExecutionResult(BaseModel):
    """Final output bundle containing the schedule, validation result, and step-by-step logs."""
    schedule: Optional[schema.StudySchedule] = Field(default=None, description="The generated study timetable, or None if failed")
    validation_result: schema.ValidationResult = Field(..., description="Independent Python validation result")
    attempts_used: int = Field(default=1, description="Number of generation attempts used (1 or 2)")
    logs: List[AgentStepLog] = Field(default_factory=list, description="Chronological log of agent steps for the UI")
    error_message: Optional[str] = Field(default=None, description="Critical failure or API error message if any")


def clean_json_text(text: str) -> str:
    """Helper to extract pure JSON from potential markdown code fences (```json ... ```)."""
    text = text.strip()
    # Match markdown json block if present
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        return match.group(1).strip()
    return text


def parse_schedule_json(raw_text: str) -> schema.StudySchedule:
    """
    Parses a raw text or JSON string into a validated StudySchedule Pydantic model.
    Raises ValueError if parsing or validation fails.
    """
    clean_text = clean_json_text(raw_text)
    try:
        data = json.loads(clean_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model output is not valid JSON: {str(e)}")

    try:
        return schema.StudySchedule.model_validate(data)
    except Exception as e:
        raise ValueError(f"JSON does not conform to StudySchedule schema: {str(e)}")


def run_tool_calling_loop(
    client: genai.Client,
    model_name: str,
    contents: List[types.Content],
    tools: List[Callable],
    tool_map: Dict[str, Callable],
    logs: List[AgentStepLog],
    max_turns: int = 5
) -> str:
    """
    Handles Gemini function calling turns until the model produces a final text response.
    
    1. Sends conversation history with tools enabled.
    2. If the model invokes tools, executes them in Python and logs the result.
    3. Feeds tool responses back into the conversation history.
    4. Repeats until Gemini returns final text or max_turns is reached.
    """
    config = types.GenerateContentConfig(
        tools=tools,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        temperature=0.2,
    )

    for turn in range(max_turns):
        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=config
        )

        # Check if the model requested any tool calls
        if response.function_calls:
            # Append the model's call message to the history
            for candidate in response.candidates:
                contents.append(candidate.content)

            # Execute each requested tool call
            for call in response.function_calls:
                tool_name = call.name
                tool_args = dict(call.args) if call.args else {}

                logs.append(
                    AgentStepLog(
                        step_type="tool_call",
                        message=f"Agent invoked tool: `{tool_name}`",
                        details={"tool": tool_name, "args": tool_args}
                    )
                )

                if tool_name in tool_map:
                    try:
                        tool_func = tool_map[tool_name]
                        tool_output = tool_func(**tool_args)
                    except Exception as e:
                        tool_output = f"Tool execution error: {str(e)}"
                else:
                    tool_output = f"Error: Tool '{tool_name}' is not recognized."

                logs.append(
                    AgentStepLog(
                        step_type="tool_result",
                        message=f"Tool `{tool_name}` returned result",
                        details={"tool": tool_name, "output": str(tool_output)}
                    )
                )

                # Return the tool output back to Gemini
                contents.append(
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_function_response(
                                name=tool_name,
                                response={"result": tool_output}
                            )
                        ]
                    )
                )
        else:
            # Model finished reasoning and returned final text
            return response.text or ""

    return response.text or ""


def generate_study_plan(
    request: schema.StudyPlanRequest,
    api_key: Optional[str] = None
) -> AgentExecutionResult:
    """
    Main entry point for the Agentic Study Planner.
    
    Coordinates:
    - User input requirements
    - Tool execution (`get_available_hours`, `check_schedule`)
    - Two-attempt generation & repair workflow
    - Independent final validation with validator.py
    """
    logs: List[AgentStepLog] = []

    # 1. Resolve API Key
    effective_key = api_key or os.getenv("GEMINI_API_KEY")
    if not effective_key:
        return AgentExecutionResult(
            schedule=None,
            validation_result=schema.ValidationResult(
                is_valid=False,
                errors=["Gemini API key is missing. Please set GEMINI_API_KEY in your .env file or input it in the UI."]
            ),
            attempts_used=0,
            logs=[AgentStepLog(step_type="attempt", message="Failed: API key not provided.")],
            error_message="API key missing"
        )

    # 2. Select Model
    model_name = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

    logs.append(
        AgentStepLog(
            step_type="thought",
            message=f"Initialized AI Study Planner Agent using model `{model_name}`.",
            details={"start_date": str(request.start_date), "daily_max_hours": request.daily_max_hours}
        )
    )

    try:
        client = genai.Client(api_key=effective_key)
    except Exception as e:
        return AgentExecutionResult(
            schedule=None,
            validation_result=schema.ValidationResult(is_valid=False, errors=[f"Failed to create Gemini client: {str(e)}"]),
            attempts_used=0,
            logs=logs,
            error_message=str(e)
        )

    # 3. Define Tools bound to this request context
    def get_available_hours(start_date: str, end_date: str, daily_max_hours: float) -> str:
        """
        Calculates available study days and total available study hours between two dates.
        Study days are strictly before the exam date.
        
        Args:
            start_date: Start date string in YYYY-MM-DD format.
            end_date: Exam date string in YYYY-MM-DD format.
            daily_max_hours: Maximum study hours per day.
        """
        return validator.get_available_hours(start_date, end_date, daily_max_hours)

    def check_schedule(schedule_json: str) -> str:
        """
        Checks whether a proposed schedule meets study rules:
        - Total hours per day must not exceed daily_max_hours.
        - Topics must be scheduled strictly before exam dates.
        - All requested syllabus topics must be scheduled.
        
        Args:
            schedule_json: JSON string matching the StudySchedule schema.
        """
        try:
            sched = parse_schedule_json(schedule_json)
            val_result = validator.validate_schedule(sched, request)
            return validator.format_validation_report(val_result)
        except Exception as e:
            return f"Schedule format error during check: {str(e)}"

    tools = [get_available_hours, check_schedule]
    tool_map = {
        "get_available_hours": get_available_hours,
        "check_schedule": check_schedule,
    }

    # 4. Build prompt describing student requirements and output schema
    subjects_desc = []
    for s in request.subjects:
        topics_str = ", ".join(f"{t.name} ({t.difficulty})" for t in s.topics)
        subjects_desc.append(f"- Subject: {s.name} | Exam Date: {s.exam_date} | Topics: {topics_str}")
    subjects_text = "\n".join(subjects_desc)

    system_instructions = (
        "You are an expert AI Study Planner Agent. Your goal is to create a realistic, "
        "day-by-day study timetable for a student.\n\n"
        "STUDENT REQUIREMENTS:\n"
        f"Start Date: {request.start_date}\n"
        f"Maximum Daily Study Hours: {request.daily_max_hours} hours/day\n"
        f"Subjects and Exams:\n{subjects_text}\n\n"
        "IMPORTANT RULES:\n"
        "1. Topics MUST be scheduled strictly BEFORE each subject's exam date. Never schedule on or after the exam.\n"
        f"2. Daily total hours must NOT exceed {request.daily_max_hours} hours on any single day.\n"
        "3. Every topic in the syllabus must be scheduled at least once.\n"
        "4. Allocate more hours to 'hard' topics than 'easy' topics.\n"
        "5. You have TWO Python tools available through function calling:\n"
        "   - Use `get_available_hours` to calculate the exact available study hours before each exam.\n"
        "   - Use `check_schedule` with your draft timetable JSON to test if it passes validation rules.\n\n"
        "FINAL OUTPUT FORMAT:\n"
        "When your schedule is verified, return ONLY a valid JSON object matching this exact schema:\n"
        "{\n"
        '  "days": [\n'
        '    {\n'
        '      "date": "YYYY-MM-DD",\n'
        '      "sessions": [\n'
        '        {"subject": "Subject Name", "topic": "Topic Name", "hours": 2.0}\n'
        "      ]\n"
        "    }\n"
        "  ],\n"
        '  "summary": "Friendly explanation of the study strategy"\n'
        "}"
    )

    contents: List[types.Content] = [
        types.Content(role="user", parts=[types.Part.from_text(text=system_instructions)])
    ]

    # --- ATTEMPT 1 ---
    logs.append(AgentStepLog(step_type="attempt", message="Attempt 1: Generating initial study schedule using tools..."))

    candidate_schedule: Optional[schema.StudySchedule] = None
    last_validation = schema.ValidationResult(is_valid=False, errors=["No schedule generated yet"])

    try:
        raw_output_1 = run_tool_calling_loop(client, model_name, contents, tools, tool_map, logs)
        candidate_schedule = parse_schedule_json(raw_output_1)
        # Validate candidate schedule with Python validator
        last_validation = validator.validate_schedule(candidate_schedule, request)

        logs.append(
            AgentStepLog(
                step_type="validation",
                message=f"Attempt 1 validation result: {'PASSED' if last_validation.is_valid else 'FAILED'}",
                details={"errors": last_validation.errors, "warnings": last_validation.warnings}
            )
        )

        if last_validation.is_valid:
            logs.append(AgentStepLog(step_type="thought", message="Schedule passed all validation rules on Attempt 1!"))
            return AgentExecutionResult(
                schedule=candidate_schedule,
                validation_result=last_validation,
                attempts_used=1,
                logs=logs
            )

    except Exception as e:
        err_msg = f"Attempt 1 generation or parsing failed: {str(e)}"
        last_validation = schema.ValidationResult(is_valid=False, errors=[err_msg])
        logs.append(AgentStepLog(step_type="validation", message=f"Attempt 1 failed: {err_msg}"))

    # --- ATTEMPT 2 (Correction) ---
    logs.append(
        AgentStepLog(
            step_type="attempt",
            message="Attempt 2: Requesting corrections from Gemini based on validation errors..."
        )
    )

    repair_prompt = (
        "Your proposed schedule failed validation with the following specific errors:\n"
        + validator.format_validation_report(last_validation) + "\n\n"
        "Please fix every listed error and generate a revised timetable.\n"
        f"Remember: Do not exceed {request.daily_max_hours} hours on any day, and schedule all topics BEFORE their exam dates.\n"
        "Return ONLY the corrected JSON object matching the StudySchedule schema."
    )

    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=repair_prompt)]))

    try:
        raw_output_2 = run_tool_calling_loop(client, model_name, contents, tools, tool_map, logs)
        candidate_schedule = parse_schedule_json(raw_output_2)
    except Exception as e:
        err_msg = f"Attempt 2 repair failed: {str(e)}"
        logs.append(AgentStepLog(step_type="validation", message=err_msg))

    # --- INDEPENDENT FINAL VALIDATION ---
    # The application code independently verifies the final schedule.
    # The UI status is determined strictly by this result, never by model claim.
    if candidate_schedule:
        final_validation = validator.validate_schedule(candidate_schedule, request)
    else:
        final_validation = schema.ValidationResult(
            is_valid=False,
            errors=["Failed to produce a parsable schedule after 2 attempts."]
        )

    logs.append(
        AgentStepLog(
            step_type="validation",
            message=f"Independent final validation: {'PASSED' if final_validation.is_valid else 'FAILED'}",
            details={"errors": final_validation.errors, "warnings": final_validation.warnings}
        )
    )

    return AgentExecutionResult(
        schedule=candidate_schedule,
        validation_result=final_validation,
        attempts_used=2,
        logs=logs
    )
