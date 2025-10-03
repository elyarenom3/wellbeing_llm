
# Wellbeing Planner (Python • FastAPI • Provider‑agnostic LLM)

A production‑minded, preliminary backend that turns multi‑turn user input + curated wellbeing
content into:
- an *empathetic conversational reply*, and
- a *structured action plan* (1–2 items) with clear rationale.

It uses a small orchestration pipeline with reflection signals, retrieval over structured content,
guardrails, and structured logging into SQLite. It is model‑provider agnostic via [LiteLLM] and also
runs **offline** using a deterministic **RuleBasedLLM** fallback.

## Architecture

```
/app
  models.py         # Pydantic models (UserContext, Plan, ReflectionSignals, etc.)
  reflection.py     # Sentiment + themes extraction (VADER + heuristics)
  retrieval.py      # Embedding-based retrieval over curated content with explainers
  providers.py      # LLM abstraction: LiteLLM client + RuleBasedLLM fallback
  prompts.py        # Prompt templates for empathy + JSON plan
  validators.py     # Guardrails: safe & achievable plan checks
  orchestration.py  # Orchestrator that wires steps + logging
  calendar_tools.py # Calendar block helper for plan execution windows
  life_quality.py   # LQI-lite score computation + reporting helpers
  privacy.py        # PII redaction, encryption helpers, retention utilities
  main.py           # FastAPI with /plan endpoint
/frontend
  index.html        # Zero-dependency SPA that talks to the FastAPI backend
/data
  wellbeing_content.json  # Curated sample content (extendable)
/tests
  test_flow.py
requirements.txt
run_cli.py
README.md
```

**Flow**

1. **Reflection**: `reflection.analyze()` computes sentiment (DistilBERT when available, VADER/heuristic fallback), themes
   (keyword mapping + context), and an energy estimate.
2. **Retrieval**: generates embeddings (Sentence-Transformers if available, TF‑IDF fallback) and
   performs cosine similarity search over the wellbeing library. Each recommendation ships with an
   *explainer* citing the most relevant passage.
3. **Synthesis**: prompts an LLM to return *strict JSON* for the plan (LiteLLM model), with a robust
   fallback builder if JSON fails or no provider is configured.
4. **Evidence layer**: ensures every plan item carries a mini-citation and one-line research-backed
   “why”, enforcing provenance before validation.
5. **Guardrails**: validates durations/safety/citations; caps LQI volatility if a previous point exists.
6. **Calendar Blocks**: suggests execution windows aligned to the user’s timezone and availability.
7. **Empathy & Nudges**: prompts the LLM (or rule‑based copy) for a caring message, and generates a
   personalized nudge using rolling reflection scores + streaks stored in SQLite.
8. **Life Quality Signal**: computes an LQI-lite score (0–100) that blends sentiment slope, stress/sleep
   themes, and inferred action adherence; archives the score for plotting.
9. **Privacy layer**: optional PII redaction, at-rest encryption, logging opt-out, and retention cleanup.

> The design intentionally keeps each step small and testable, while exhibiting the “graph” idea
  (distinct nodes: reflection → retrieval → plan → validate → empathy). We can swap any node
  (e.g., a vector retriever, a different sentiment model) without touching others.

## Why this content approach?

We kept a curated JSON (`data/wellbeing_content.json`) with clearly tagged micro‑interventions that are
safe and broadly applicable. This keeps the system deterministic, explainable, and easy to extend
without needing a heavy RAG stack.

## Provider‑agnostic LLM

We use **LiteLLM** (`LITELLM_MODEL`) so we can point to OpenAI/Anthropic/Ollama/HuggingFace
backends *without code changes*.

- Example: `export LITELLM_MODEL=gpt-4o-mini` (requires `OPENAI_API_KEY`)
- Example: `export LITELLM_MODEL=ollama/llama3.1` (requires a local Ollama server)

If no model is configured or LiteLLM isn’t installed, the **RuleBasedLLM** produces sensible outputs so
the API and tests run offline.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Configure `.env`

Add or update the following keys:

```
PRIVACY_MODE=false
PRIVACY_LOGGING_OPTOUT=false
WB_RETENTION_DAYS=30

# OpenAI example (LiteLLM will route to this model)
OPENAI_API_KEY=sk-...
MODEL_PROVIDER=openai
LITELLM_MODEL=gpt-4o-mini

# Optional overrides
# WB_DATA_PATH=./data/wellbeing_content.json
# WB_SQLITE_PATH=./wellbeing_logs.sqlite3
```

For other providers, swap `MODEL_PROVIDER` / `LITELLM_MODEL` and add the relevant credentials.

> No keys? The app still works in offline mode via the deterministic fallback—it just won’t call a live
  LLM.

Optional quality boosts: install `sentence-transformers`, `transformers`, and `torch` to enable
transformer-grade embeddings and DistilBERT sentiment.

## Run

### Option A — FastAPI

Start the server (note the module path to **`app/main.py`**) after loading the `.env` so the
environment variables are available:
```bash
set -a
source .env
set +a
uvicorn app.main:app --reload
```

Once the server is running, open `http://127.0.0.1:8000/` to use the built-in frontend. The page lets you:
- enter multi-turn context,
- view empathetic responses, plan items, citations, calendar suggestions,
- and visualize the Life Quality Signal sparkline without any external dependencies.

**POST /plan** expects:
- `context.user_id` (**required**)
- `context.free_text` (user’s message)
- `context.preferences`: **list of strings** in `"key=value"` form (e.g., `"available_time_min=30"`)
- `conversation`: object (e.g., `{"messages":[...]}`), not a bare array

```bash
curl -X POST http://127.0.0.1:8000/plan   -H 'Content-Type: application/json'   -d '{
    "context": {
      "user_id": "demo-123",
      "free_text": "Feeling tense before a talk—30 min free at lunch.",
      "preferences": [
        "available_time_min=30",
        "focus_area=stress",
        "time_of_day=lunch"
      ]
    },
    "conversation": {
      "messages": [
        {"role": "user", "content": "Feeling tense before a talk—30 min free at lunch."}
      ]
    }
  }'
```


### Option B — CLI

```bash
python run_cli.py   --available_minutes 12   --focus_area stress   --messages "Slept badly" "Lower back tight" "Have 10 minutes only"
```

### View Metrics

Fetch the LQI-lite line chart for a user:

```bash
curl "http://127.0.0.1:8000/metrics/life-quality?user_id=demo-user&limit=30"
```

## Response (example)
```json
{
  "session_id": "e7bfea8e-1a83-45bf-bca3-e779248281cd",
  "empathetic_message": "I hear that today has been a lot—thank you for sharing...",
  "plan": {
    "day": "today",
    "items": [
      {
        "content_id": "ritual-breathing",
        "title": "5-Minute Breathing Reset",
        "duration_minutes": 5,
        "why_it_helps": "Quick downshift for the nervous system; pairs well with low energy days.",
        "instructions": "Inhale 4, hold 4, exhale 6 for five cycles."
      }
    ],
    "caution": null
  },
  "signals": {
    "sentiment": 0.34,
    "top_themes": ["stress"],
    "energy_level": "medium",
    "summary": "Sentiment 0.34. Themes: stress. Energy: medium."
  },
  "life_quality": {
    "score": 72.4,
    "trend": "steady",
    "recent": [{"timestamp": "2024-05-06T00:00:00", "score": 70.2}, {"timestamp": "2024-05-07T00:00:00", "score": 72.4}]
  },
  "candidates": [
    { "id": "ritual-breathing", "score": 26.02, "...": "..." },
    { "id": "gratitude-journal", "score": 22.45, "...": "..." },
    { "id": "micro-movements", "score": 18.66, "...": "..." }
  ]
}
```

## Guardrails & Safety

- Enforces time budget (<= available minutes), minimal durations, and disallows risky keywords.
- Uses curated content only by default. Extend with medical governance if adding new items.
- Returns `caution` when a validator flags issues.

## Logging & Checkpoints

SQLite tables:
- `sessions(id, user_id, created_at)`
- `steps(session_id, step_name, input_json, output_json, started_at, ended_at, meta)`
- `plans(session_id, plan_json, signals_json, created_at)`
- `life_quality(session_id, user_id, score, payload, created_at)`

We can query `steps` to debug each node’s input/output.

> When `PRIVACY_MODE=true`, logged payloads are redacted/encrypted before hitting disk, and retention
  cleanup purges stale rows automatically.

## Testing

```bash
pytest -q
```

The suite exercises the offline planner, verifies the generated life-quality history, and checks that
the frontend entrypoint is present.

## New Extensibility Features

- **Embedding + vector retrieval**: curates candidates via cosine similarity and surfaces cited
  explainers for transparency. Optional dependency: install `sentence-transformers` for
  transformer-grade embeddings (falls back to lightweight bag-of-words).
- **Personalized nudges**: SQLite now tracks user streaks and rolling reflection scores to tailor the
  motivational message at the end of each run.
- **Calendar blocks**: provides suggested execution windows (rounded to the next quarter hour) based
  on the user’s timezone and available minutes.
- **Expanded themes**: loneliness, body image, and burnout keyword maps enrich reflection signals and
  downstream retrieval.
- **DistilBERT sentiment swap**: the reflection step uses DistilBERT (when installed) with calibrated
  confidence scores, and gracefully falls back to VADER/heuristics in offline mode.
- **Life Quality Signal (LQI-lite)**: combines sentiment slope, theme load, and inferred adherence to
  produce a 0–100 wellbeing pulse, with volatility-capped progression and a lightweight Plotly-ready chart.
- **Evidence-backed plans**: every plan item carries a concise citation with a direct link to the
  supporting study, along with a rationale pulled from the vector store explainers.
- **Privacy layer**: regex redaction, optional logging opt-out, encrypted storage, key rotation, and
  automatic retention cleanup when `PRIVACY_MODE=true`.
