
# Wellbeing Planner (Python • FastAPI • Provider‑agnostic LLM)

A production‑minded, <4h‑scope backend that turns multi‑turn user input + curated wellbeing
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
  retrieval.py      # Simple retrieval over curated content (tag overlap + fuzzy match)
  providers.py      # LLM abstraction: LiteLLM client + RuleBasedLLM fallback
  prompts.py        # Prompt templates for empathy + JSON plan
  validators.py     # Guardrails: safe & achievable plan checks
  orchestration.py  # Orchestrator that wires steps + logging
  main.py           # FastAPI with /plan endpoint
/data
  wellbeing_content.json  # Curated sample content (extendable)
/tests
  test_flow.py
requirements.txt
run_cli.py
README.md
```

**Flow**

1. **Reflection**: `reflection.analyze()` computes sentiment (VADER compound), themes (keyword mapping
   + context), and an energy estimate.
2. **Retrieval**: ranks curated wellbeing content by theme overlap + fuzzy scoring (RapidFuzz).
3. **Synthesis**: prompts an LLM to return *strict JSON* for the plan (LiteLLM model), with a robust
   fallback builder if JSON fails or no provider is configured.
4. **Guardrails**: validates durations/safety; adds `caution` if needed.
5. **Empathy**: prompts the LLM (or rule‑based copy) for a short, caring message.
6. **Logging**: every step is timestamped into SQLite (`wellbeing_logs.sqlite3`).

> The design intentionally keeps each step small and testable, while exhibiting the “graph” idea
  (distinct nodes: reflection → retrieval → plan → validate → empathy). You can swap any node
  (e.g., a vector retriever, a different sentiment model) without touching others.

## Why this content approach?

We kept a curated JSON (`data/wellbeing_content.json`) with clearly tagged micro‑interventions that are
safe and broadly applicable. This keeps the system deterministic, explainable, and easy to extend
without needing a heavy RAG stack. For a larger library, plug a vector DB (Chroma, FAISS) and swap
`retrieval.py` to embed titles/summaries and search by cosine similarity.

## Provider‑agnostic LLM

We use **LiteLLM** (`LITELLM_MODEL`) so you can point to OpenAI/Anthropic/Ollama/HuggingFace
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

Optional provider keys:
- `OPENAI_API_KEY` (for OpenAI‑compatible models)
- `ANTHROPIC_API_KEY`
- `GEMINI_API_KEY`
- `LITELLM_MODEL` (e.g., `gpt-4o-mini`, `ollama/llama3.1`, `openrouter/auto`)

**Environment paths**
- `WB_DATA_PATH` — path to `wellbeing_content.json` (default: `./data/wellbeing_content.json`)
- `WB_SQLITE_PATH` — path to SQLite log DB (default: `./wellbeing_logs.sqlite3`)

## Run

### Option A — FastAPI

```bash
uvicorn app.main:app --reload
```

Then `POST /plan` with JSON like:

```json
{
  "context": {
    "user_id": "u123",
    "mood": "tired and a little overwhelmed",
    "available_minutes": 12,
    "focus_area": "stress",
    "preferences": ["gentle", "at-desk"],
    "constraints": ["no floor work"],
    "timezone": "America/Toronto"
  },
  "conversation": {
    "messages": [
      { "role": "user", "content": "Slept poorly, tight lower back, anxious about deadlines." },
      { "role": "user", "content": "I only have ~10 minutes between calls." }
    ]
  }
}
```

### Option B — CLI

```bash
python run_cli.py --available_minutes 12 --focus_area stress   --messages "Slept badly" "Lower back tight" "Have 10 minutes only"
```

## Response

```json
{
  "session_id": "...",
  "empathetic_message": "I hear that today has been a lot…",
  "plan": {
    "day": "today",
    "items": [
      {
        "content_id": "ritual-breathing",
        "title": "5-Minute Breathing Reset",
        "duration_minutes": 5,
        "why_it_helps": "Quick downshift…",
        "instructions": "Inhale 4, hold 4, exhale 6…"
      }
    ],
    "caution": null
  },
  "signals": {
    "sentiment": -0.34,
    "top_themes": ["stress", "mobility"],
    "energy_level": "low",
    "summary": "Sentiment -0.34. Themes: …"
  },
  "candidates": [ /* ranked content */ ]
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

You can query `steps` to debug each node’s input/output.

## Testing

```bash
pytest -q
```

The test runs the full flow with the offline fallback and asserts basic invariants.

## Extensibility Ideas

- Replace retrieval with embeddings + vector DB; include “explainer” that cites content passages.
- Add “streak” & “reflection score” in the DB to personalize nudges.
- Integrate a calendar block tool for plan execution windows.
- Add more themes (“loneliness”, “body image”, “burnout”) and map keywords from user text.
- Swap the sentiment model (e.g., DistilBERT) and report confidence ± calibration.

## Assumptions & Trade‑offs

- I optimized for reliability and clarity over flash. A small curated library + heuristics works
  well for a lightweight demo and avoids drift.
- JSON output is enforced in prompt; however, I kept a robust fallback to guarantee responses.
- I kept dependencies light and avoid heavyweight NLP pipelines.
