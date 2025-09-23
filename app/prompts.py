from __future__ import annotations
from typing import List

EMPATHETIC_TEMPLATE = """
You are a careful wellbeing coach. You will write an EMPATHETIC_MESSAGE.

Context:
- User signals: {signals}
- Available minutes: {available_minutes}
- Candidate actions: {candidates}

Guidelines:
- Sound human and caring; 2–4 sentences.
- Acknowledge how the user feels, reflect 1–2 themes, normalize the experience.
- Set a gentle, confident tone and motivate the plan that follows.
- Avoid moralizing; emphasize adjustability.

Write EMPATHETIC_MESSAGE now.
"""

PLAN_JSON_TEMPLATE = """
You are a planner. Output strictly valid JSON in a top-level object—no Markdown.

The user can spend up to {available_minutes} minutes today.
Signals: {signals}
You have these candidate interventions (id, title, tags, summary): {candidates}

Return an object with:
- day: always "today"
- items: an array of 1–2 items. Each item has:
  - content_id (string)
  - title (string)
  - duration_minutes (integer <= {available_minutes})
  - why_it_helps (short, concrete reason, tailored to signals/themes)
  - instructions (1–3 sentences, actionable)

Only include items that fit the time budget and align with themes.
Do not include any extra keys, comments, or trailing text.

Return JSON ONLY:

PLAN_JSON:
"""
