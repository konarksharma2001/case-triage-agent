# Case Triage AI Agent

An autonomous AI agent that investigates potentially duplicate CRM support cases and drafts a recommendation for human review. The system follows an agentic workflow where the LLM decides how to investigate using deterministic tools, while the backend enforces validation, loop limits, human approval, and audit logging.

---

# Problem Statement

Support teams often receive duplicate support cases when customers submit the same issue multiple times through different channels or when colleagues report the same problem independently.

The objective of this project is to:

1. Generate candidate duplicate pairs using inexpensive deterministic heuristics.
2. Allow an AI agent to investigate each candidate pair using tool calls.
3. Produce a structured recommendation.
4. Require a human reviewer to approve, reject, or override the recommendation.
5. Maintain a complete audit trail of the investigation.

---

# Project Structure

```
case-triage-agent/

├── agent.py                 # Autonomous investigation loop
├── candidate_pairs.py       # Candidate pair generation
├── sanitize.py              # Prompt injection protection
├── tools.py                 # Deterministic investigation tools
├── storage.py               # JSON persistence
├── api.py                   # FastAPI endpoints
├── run_batch.py             # Batch investigation runner
├── investigations.json      # Investigation + audit storage
├── support_cases_1.csv
├── requirements.txt
└── README.md
```

---

# Technology Stack

- Python 3.12
- FastAPI
- Groq API (Llama 3.3 70B)
- Pydantic
- RapidFuzz
- Pandas

---

# Setup

Clone the repository

```bash
git clone <repository>
cd case-triage-agent
```

Create virtual environment

```bash
python -m venv venv
```

Activate

### macOS/Linux

```bash
source venv/bin/activate
```

### Windows

```bash
venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Create a `.env` file

```text
GROQ_API_KEY=<your_api_key>
```

---

# Running the Project

## Generate investigations

```bash
python run_batch.py
```

This will:

- Load support cases
- Generate candidate pairs
- Investigate 10 candidate pairs
- Save draft investigations into `investigations.json`

---

## Start the REST API

```bash
uvicorn api:app --reload
```

Swagger UI

```
http://127.0.0.1:8000/docs
```

---

# Part 1 – Candidate Pair Generation

The first stage intentionally favours **recall over precision**.

Three deterministic heuristics are used:

- Fuzzy account name matching
- Exact contact email matching
- Subject token overlap

This stage is intentionally lightweight and inexpensive. False positives are acceptable because they are filtered by the AI investigation stage.

---

# Part 2 – Investigation Agent

Each candidate pair is investigated independently by an autonomous AI agent.

The agent maintains conversational state across multiple reasoning steps and dynamically selects which deterministic tool to call next based on evidence collected so far.

The investigation terminates either when:

- enough evidence has been gathered to submit a recommendation, or
- the maximum investigation step limit is reached.

The output is always a **draft recommendation**.

No investigation is finalized automatically.

---

# Investigation Tools

The agent has access to four deterministic Python tools.

## compare_fields()

Compares structured fields such as

- channel
- priority
- status
- contact information

---

## fuzzy_score()

Calculates similarity for

- account name
- contact name
- subject
- description

using RapidFuzz.

---

## timeline_gap()

Calculates the time difference between the two cases.

---

## find_other_cases()

Searches for additional cases belonging to the same account or contact to determine whether the pair is an isolated duplicate or part of a larger pattern.

---

# Model vs Code Responsibilities

## The LLM decides

- which tool to call
- tool order
- when sufficient evidence has been gathered
- recommendation label
- confidence
- rationale

## Python code decides

- deterministic field comparison
- fuzzy matching
- timeline calculations
- tool execution
- schema validation
- retry behaviour
- loop limits
- human approval enforcement
- audit logging

This separation keeps business logic deterministic while allowing the LLM to perform investigation and reasoning.

---

# Structured Output

Every investigation produces a validated recommendation.

```json
{
  "label": "DUPLICATE",
  "confidence": 0.92,
  "evidence": [
    ...
  ],
  "rationale": "..."
}
```

Pydantic validation ensures the backend never relies on free-form model output.

---

# Loop Bound

The investigation loop is bounded.

```
MAX_STEPS = 6
```

This prevents infinite reasoning loops while allowing sufficient investigation depth.

---

# Error Handling

The agent includes several safeguards:

- schema validation
- malformed tool call handling
- API retry with exponential backoff
- fallback to `UNSURE` when investigation cannot be completed safely

This ensures failures degrade gracefully instead of crashing the workflow.

---

# Part 3 – Human Review

The AI agent only produces a recommendation.

A human reviewer must explicitly

- approve
- reject
- override

before an investigation is considered complete.

This approval gate is enforced in the backend.

---

# REST API

| Method | Endpoint | Description |
|---------|----------|-------------|
| GET | `/investigations` | Pending investigations |
| GET | `/investigations/{id}` | Investigation details |
| POST | `/investigations/{id}/decision` | Human decision |
| GET | `/audit` | Complete audit history |

---

# Audit Trail

Each investigation records:

- every tool invocation
- tool inputs
- tool outputs
- reasoning trace
- draft recommendation
- human decision
- timestamps

The audit log allows an investigation to be reconstructed after completion.

---

# Example Workflow

```
support_cases.csv
        │
        ▼
Candidate Generation
        │
        ▼
Candidate Pair
        │
        ▼
AI Investigation
        │
        ▼
Draft Recommendation
        │
        ▼
Human Review
        │
        ▼
Final Decision
        │
        ▼
Audit Log
```

---

# Design Trade-offs

- Candidate generation favours recall over precision.
- Deterministic tools keep business logic transparent.
- JSON storage was selected over SQLite for simplicity.
- No frontend was implemented in order to focus on the backend workflow.

