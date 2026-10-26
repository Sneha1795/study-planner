"""
app.py - Streamlit User Interface for the Agentic AI Study Planner.

This app allows students to:
1. Input their subjects, exam dates, syllabus topics, and difficulty levels.
2. Configure study parameters (start date, daily study hour limit).
3. Generate a day-by-day timetable using a Gemini-powered Agent.
4. Watch the agent use real Python tools (get_available_hours, check_schedule).
5. View the final schedule validated by an independent Python rule engine.
"""

import datetime as dt
import json
import os
import pandas as pd
import streamlit as st
import dotenv

import agent
import schema
import validator

# Load environment variables
dotenv.load_dotenv()

# Page configuration
st.set_page_config(
    page_title="AI Agentic Study Planner",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean, modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .badge-card {
        background-color: #F3F4F6;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 16px;
        border-left: 4px solid #3B82F6;
    }
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# State Management Helpers
# ---------------------------------------------------------
def initialize_session_state():
    """Initializes default subjects and topics if not already set."""
    if "subjects" not in st.session_state:
        # Provide a realistic beginner-friendly starting syllabus
        today = dt.date.today()
        st.session_state.subjects = [
            {
                "name": "Mathematics",
                "exam_date": today + dt.timedelta(days=10),
                "topics": [
                    {"name": "Calculus", "difficulty": "hard"},
                    {"name": "Linear Algebra", "difficulty": "medium"}
                ]
            },
            {
                "name": "Physics",
                "exam_date": today + dt.timedelta(days=14),
                "topics": [
                    {"name": "Electromagnetism", "difficulty": "medium"},
                    {"name": "Optics", "difficulty": "easy"}
                ]
            }
        ]
    if "agent_result" not in st.session_state:
        st.session_state.agent_result = None


initialize_session_state()


# ---------------------------------------------------------
# Sidebar: Configuration & Global Settings
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")

    # API Key Handling (never hardcoded, reads from .env or UI override)
    env_api_key = os.getenv("GEMINI_API_KEY", "")
    has_env_key = bool(env_api_key and env_api_key != "your_gemini_api_key_here")

    if has_env_key:
        st.success("🔑 Gemini API Key detected from `.env`")
        custom_key = st.text_input("Override API Key (optional):", type="password", placeholder="Leave blank to use .env key")
        effective_api_key = custom_key.strip() if custom_key.strip() else env_api_key
    else:
        st.warning("⚠️ No API key found in `.env`")
        effective_api_key = st.text_input("Enter Gemini API Key:", type="password", placeholder="AIzaSy...")

    st.divider()

    st.subheader("📅 Study Parameters")
    start_date = st.date_input(
        "Study Start Date",
        value=dt.date.today() + dt.timedelta(days=1),
        help="The calendar day you plan to start studying"
    )

    daily_max_hours = st.slider(
        "Maximum Daily Study Hours",
        min_value=1.0,
        max_value=12.0,
        value=4.0,
        step=0.5,
        help="Maximum hours you can realistically study per day"
    )

    model_name = st.selectbox(
        "Gemini Model",
        options=["gemini-3.5-flash-lite", "gemini-3.5-flash", "gemini-3.8-flash"],
        index=0,
        help="Fast, modern Gemini flash models supporting Python tool calling"
    )
    os.environ["GEMINI_MODEL"] = model_name

    st.divider()
    st.subheader("💡 Agentic Architecture")
    st.markdown("""
    This app demonstrates **True Agentic AI**:
    1. **Gemini Agent** receives your syllabus.
    2. Uses Python **Function Calling Tools**:
       - ⏱️ `get_available_hours`
       - 🔍 `check_schedule`
    3. **2-Attempt Self-Correction Loop**: Revises plan if rules fail.
    4. **Independent Python Validator**: UI status is determined by Python, not model claims.
    """)

    if st.button("🔄 Reset Sample Data"):
        st.session_state.subjects = None
        st.session_state.agent_result = None
        initialize_session_state()
        st.rerun()


# ---------------------------------------------------------
# Main UI Header
# ---------------------------------------------------------
st.markdown('<div class="main-header">🎓 AI Agentic Study Planner</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automated day-by-day timetable generation with Gemini function calling and strict Python validation.</div>', unsafe_allow_html=True)


# ---------------------------------------------------------
# Subjects & Topics Input Section
# ---------------------------------------------------------
st.subheader("📚 1. Enter Your Subjects & Topics")

# Render each subject and its topics
subjects_to_delete = []

for s_idx, subject in enumerate(st.session_state.subjects):
    with st.expander(f"📖 Subject {s_idx + 1}: **{subject['name']}** (Exam: {subject['exam_date']})", expanded=True):
        col1, col2, col3 = st.columns([3, 2, 1])

        with col1:
            subject["name"] = st.text_input(
                f"Subject Name #{s_idx + 1}",
                value=subject["name"],
                key=f"sub_name_{s_idx}"
            )
        with col2:
            subject["exam_date"] = st.date_input(
                f"Exam Date #{s_idx + 1}",
                value=subject["exam_date"],
                key=f"sub_date_{s_idx}"
            )
        with col3:
            st.write("")
            st.write("")
            if st.button("🗑️ Remove", key=f"del_sub_{s_idx}"):
                subjects_to_delete.append(s_idx)

        st.markdown("**Topics under this subject:**")

        topics_to_delete = []
        for t_idx, topic in enumerate(subject["topics"]):
            t_col1, t_col2, t_col3 = st.columns([4, 3, 1])
            with t_col1:
                topic["name"] = st.text_input(
                    "Topic Name",
                    value=topic["name"],
                    key=f"topic_name_{s_idx}_{t_idx}",
                    label_visibility="collapsed"
                )
            with t_col2:
                diff_options = ["easy", "medium", "hard"]
                curr_diff = topic.get("difficulty", "medium").lower()
                diff_idx = diff_options.index(curr_diff) if curr_diff in diff_options else 1
                topic["difficulty"] = st.selectbox(
                    "Difficulty",
                    options=diff_options,
                    index=diff_idx,
                    key=f"topic_diff_{s_idx}_{t_idx}",
                    label_visibility="collapsed"
                )
            with t_col3:
                if st.button("✕", key=f"del_topic_{s_idx}_{t_idx}", help="Remove topic"):
                    topics_to_delete.append(t_idx)

        # Apply topic deletions
        for t_idx in sorted(topics_to_delete, reverse=True):
            subject["topics"].pop(t_idx)
            st.rerun()

        # Add new topic button
        if st.button(f"➕ Add Topic to {subject['name'] or 'Subject'}", key=f"add_topic_{s_idx}"):
            subject["topics"].append({"name": "New Topic", "difficulty": "medium"})
            st.rerun()

# Apply subject deletions
for s_idx in sorted(subjects_to_delete, reverse=True):
    st.session_state.subjects.pop(s_idx)
    st.rerun()

# Add new subject button
if st.button("➕ Add Another Subject"):
    st.session_state.subjects.append({
        "name": f"Subject {len(st.session_state.subjects) + 1}",
        "exam_date": start_date + dt.timedelta(days=14),
        "topics": [{"name": "Sample Topic", "difficulty": "medium"}]
    })
    st.rerun()


# ---------------------------------------------------------
# Action: Generate Study Plan
# ---------------------------------------------------------
st.divider()

col_btn, col_info = st.columns([2, 5])
with col_btn:
    generate_clicked = st.button("🚀 Generate Study Timetable", type="primary", use_container_width=True)

with col_info:
    st.caption("The Gemini agent will calculate available hours, draft a timetable, check rules using Python tools, and self-correct if needed.")

if generate_clicked:
    # 1. Validate API Key
    if not effective_api_key:
        st.error("❌ Please provide a Gemini API Key in `.env` or in the sidebar.")
        st.stop()

    # 2. Build and Validate Pydantic Request
    try:
        pydantic_subjects = []
        for s in st.session_state.subjects:
            s_name = s["name"].strip()
            if not s_name:
                st.error("Please enter a name for all subjects.")
                st.stop()
            
            pydantic_topics = []
            for t in s["topics"]:
                t_name = t["name"].strip()
                if t_name:
                    pydantic_topics.append(schema.Topic(name=t_name, difficulty=t["difficulty"]))

            if not pydantic_topics:
                st.error(f"Subject '{s_name}' must have at least one topic.")
                st.stop()

            pydantic_subjects.append(schema.Subject(
                name=s_name,
                exam_date=s["exam_date"],
                topics=pydantic_topics
            ))

        request_payload = schema.StudyPlanRequest(
            subjects=pydantic_subjects,
            start_date=start_date,
            daily_max_hours=daily_max_hours
        )
    except Exception as e:
        st.error(f"Input validation error: {str(e)}")
        st.stop()

    # 3. Run Agent with Live Status
    with st.status("🤖 Agent at work: Planning study schedule...", expanded=True) as status_box:
        st.write("1. Reading your subjects, topics, and constraints...")
        result = agent.generate_study_plan(request_payload, api_key=effective_api_key)
        
        # Display each step taken
        for log in result.logs:
            if log.step_type == "tool_call":
                st.write(f"🛠️ **Tool Call**: `{log.details.get('tool', 'tool')}` with arguments: `{log.details.get('args', {})}`")
            elif log.step_type == "tool_result":
                st.write(f"📋 **Tool Result**: {log.details.get('output', '')}")
            elif log.step_type == "attempt":
                st.write(f"🔄 **{log.message}**")
            elif log.step_type == "validation":
                st.write(f"🔍 **Validation**: {log.message}")

        if result.validation_result.is_valid:
            status_box.update(label="✅ Study Plan Generated & Verified!", state="complete", expanded=False)
        else:
            status_box.update(label="⚠️ Study Plan Generated (Rules flagged by Python validator)", state="error", expanded=True)

    st.session_state.agent_result = result
    st.session_state.last_request = request_payload


# ---------------------------------------------------------
# Results Section
# ---------------------------------------------------------
if st.session_state.agent_result:
    res: agent.AgentExecutionResult = st.session_state.agent_result
    req: schema.StudyPlanRequest = getattr(st.session_state, "last_request", None)

    st.divider()
    st.subheader("📅 2. Generated Timetable & Verification Results")

    # Independent Validation Banner
    val = res.validation_result
    if val.is_valid:
        st.success(f"✅ **Independent Python Verification: PASSED** (Generated in {res.attempts_used} attempt{'s' if res.attempts_used > 1 else ''})")
    else:
        st.error(f"❌ **Independent Python Verification: FAILED** (Remaining issues after {res.attempts_used} attempts)")
        for err in val.errors:
            st.error(f"• {err}")

    if val.warnings:
        for warn in val.warnings:
            st.warning(f"⚠️ {warn}")

    if res.schedule and res.schedule.days:
        schedule = res.schedule

        # Display Strategy Summary
        if schedule.summary:
            st.info(f"💡 **AI Strategy**: {schedule.summary}")

        # Compute summary metrics
        total_days = len(schedule.days)
        total_sessions = sum(len(d.sessions) for d in schedule.days)
        total_hours = sum(d.total_hours for d in schedule.days)
        avg_hours = total_hours / total_days if total_days > 0 else 0

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Study Days", f"{total_days} days")
        m2.metric("Total Study Hours", f"{total_hours:.1f} hrs")
        m3.metric("Avg Hours/Day", f"{avg_hours:.1f} hrs/day")
        m4.metric("Daily Limit", f"{daily_max_hours:.1f} hrs/day")

        # Prepare Table Data
        table_rows = []
        for day in schedule.days:
            day_str = day.date.strftime("%Y-%m-%d (%a)")
            for s in day.sessions:
                table_rows.append({
                    "Date": day_str,
                    "Subject": s.subject,
                    "Topic": s.topic,
                    "Study Hours": s.hours
                })

        df = pd.DataFrame(table_rows)

        st.subheader("📋 Day-by-Day Schedule Table")
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Visual Daily Hours Breakdown Chart
        st.subheader("📊 Scheduled Hours per Day")
        daily_hours_data = [{"Date": d.date.strftime("%b %d"), "Hours Scheduled": d.total_hours} for d in schedule.days]
        chart_df = pd.DataFrame(daily_hours_data).set_index("Date")
        st.bar_chart(chart_df)

        # Download Timetable Button
        schedule_json_str = schedule.model_dump_json(indent=2)
        st.download_button(
            label="📥 Download Study Schedule (JSON)",
            data=schedule_json_str,
            file_name="study_schedule.json",
            mime="application/json"
        )

        # Technical Inspection Expander
        with st.expander("🛠️ View Agent Step Logs & Raw JSON"):
            st.markdown("**Chronological Execution Logs:**")
            for log in res.logs:
                st.code(f"[{log.step_type.upper()}] {log.message}\nDetails: {json.dumps(log.details, default=str)}")
            st.markdown("**Raw Structured JSON Output:**")
            st.json(schedule_json_str)

    elif res.error_message:
        st.error(f"Generation error: {res.error_message}")
