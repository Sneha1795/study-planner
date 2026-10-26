# 🎓 Agentic AI Study Planner

A beginner-friendly Agentic AI application built for a college course using the Gemini API, Streamlit, and Pydantic. It automatically generates a personalized, day-by-day study timetable based on your subjects, exam dates, syllabus topics, difficulty levels, and daily study hour limits.

---

## 🌟 Why is this "Agentic AI"?

Unlike a standard chatbot prompt that simply generates text and hopes for the best, this project implements a **true Agentic AI workflow**:

1. **Python Function Calling Tools**:
   * `get_available_hours`: Calculates available study days and total study capacity before exam dates using Python arithmetic (not LLM arithmetic).
   * `check_schedule`: Validates the agent's draft timetable against strict academic constraints.
2. **2-Attempt Self-Correction Loop**:
   * If the agent's initial plan (Attempt 1) violates constraints (e.g. exceeds daily hours, schedules after an exam, or misses a topic), the Python validator returns a detailed diagnostic report.
   * The agent uses that report to diagnose the issue and revise the schedule (Attempt 2).
3. **Independent Python Verification**:
   * The final timetable is validated by Python independently. The UI status (PASSED/FAILED) is **always** determined by Python code, never by the model's self-claim.

---

## 🛠️ Technology Stack

* **Python 3.11+**
* **Gemini API** (`google-genai` Python SDK)
* **Streamlit** (interactive web UI)
* **Pydantic v2** (data validation and strict schemas)
* **python-dotenv** (secure environment configuration)
* **pytest** (automated unit tests)

---

## 📁 Project Structure

```text
study-planner/
├── app.py                 # Streamlit web user interface
├── schema.py              # Pydantic data models (Request, Session, DayPlan, Schedule)
├── validator.py           # Pure Python date calculations and rule engine
├── agent.py               # Gemini function calling, 2-attempt loop, and tool execution
├── tests/
│   └── test_validator.py  # 10 comprehensive unit tests for schedule rules
├── requirements.txt       # Project dependencies
├── .env.example           # Example environment variables file
├── .gitignore             # Git ignore rules (.env, .venv, pycache)
├── pytest.ini             # Pytest configuration
└── README.md              # Project documentation
```

---

## 🚀 Setup & Installation

### 1. Clone or Open the Repository
```bash
cd study-planner
```

### 2. Activate Your Virtual Environment
On Windows:
```bash
.\.venv\Scripts\activate
```
On macOS / Linux:
```bash
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Your Gemini API Key
Create a `.env` file from `.env.example`:
```bash
cp .env.example .env
```
Open `.env` and add your Gemini API Key:
```ini
GEMINI_API_KEY=your_actual_gemini_api_key_here
GEMINI_MODEL=gemini-3.5-flash-lite
```
> **Security Note:** `.env` is listed in `.gitignore` so your API key will never be committed or uploaded to Git.

---

## 🧪 Running Unit Tests

Run the 10 automated unit tests to verify the schedule validation rules:
```bash
pytest -v
```

---

## 💻 Running the Application

Launch the Streamlit web interface:
```bash
streamlit run app.py
```

Once launched, your browser will open to `http://localhost:8501`:
1. Check your settings in the sidebar (start date, max daily hours).
2. Enter or edit your subjects, exam dates, topics, and difficulties.
3. Click **"🚀 Generate Study Timetable"**.
4. Watch the agent's live reasoning and tool calls!
5. View the validated timetable table, daily hour chart, and download the plan as JSON.

---

## 🎓 College Viva / Presentation Q&A

**Q1: Why not just ask Gemini to make the schedule in one prompt?**  
> *Answer:* LLMs are probabilistic language models, not calculators. They frequently make math errors with calendar dates and hour summations. By giving Gemini Python tools (`get_available_hours` and `check_schedule`), we let Python do exact arithmetic while Gemini focuses on strategic allocation and planning.

**Q2: How does the agent self-correct?**  
> *Answer:* If Attempt 1 produces a timetable that exceeds daily hours or places topics after the exam, `validator.py` catches the error. The error report is fed back to Gemini, which triggers Attempt 2 to fix the mistakes.

**Q3: Why have an independent validator if Gemini already uses `check_schedule`?**  
> *Answer:* Principle of verification: Never trust an AI model's claim that its own answer is correct. App code independently validates the final object in Python before rendering the "Verified" badge.
