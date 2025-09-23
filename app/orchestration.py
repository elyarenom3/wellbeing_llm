from __future__ import annotations
import os, uuid, json
from datetime import datetime
from typing import Dict, Any, List
from .models import (
    UserContext, Conversation, ReflectionSignals, ContentItem,
    Plan, PlanItem, PlanResponse, RunLog
)
from .providers import build_llm
from .reflection import analyze
from .retrieval import ContentRepository, pick_durations
from .validators import validate_plan
from .prompts import EMPATHETIC_TEMPLATE, PLAN_JSON_TEMPLATE
from . import logging_db

def _now_iso() -> str:
    return datetime.utcnow().isoformat()

class Orchestrator:
    def __init__(self, content_path: str, db_path: str | None = None) -> None:
        self.repo = ContentRepository(content_path)
        self.llm = build_llm()
        self.db_path = db_path
        logging_db.init_db(db_path)

    def run(self, context: UserContext, conversation: Conversation) -> PlanResponse:
        session_id = str(uuid.uuid4())
        logging_db.create_session(session_id, context.user_id, self.db_path)

        # Step 1: Reflection
        signals = analyze(conversation, context)
        logging_db.log_step(session_id, "reflection", input_obj=context.model_dump() | {"conversation": conversation.model_dump()}, output_obj=signals.model_dump(), db_path=self.db_path)

        # Step 2: Retrieve candidate interventions
        joined_text = " ".join([m.content for m in conversation.messages])
        candidates: List[ContentItem] = self.repo.search(joined_text, signals.top_themes, topk=5)
        logging_db.log_step(session_id, "retrieval", input_obj={"query": joined_text, "themes": signals.top_themes}, output_obj={"candidates": [c.model_dump() for c in candidates]}, db_path=self.db_path)

        # Step 3: Plan synthesis (LLM)
        available_minutes = max(1, context.available_minutes or 15)
        cand_for_prompt = [{"id": c.id, "title": c.title, "tags": c.tags, "summary": c.summary} for c in candidates]
        plan_prompt = PLAN_JSON_TEMPLATE.format(
            available_minutes=available_minutes,
            signals=json.dumps(signals.model_dump(), ensure_ascii=False),
            candidates=json.dumps(cand_for_prompt, ensure_ascii=False),
        )
        plan_json = self.llm.generate_json(plan_prompt)

        # Step 3b: Fallback if needed
        if "items" not in plan_json or not isinstance(plan_json.get("items", []), list) or not plan_json["items"]:
            # Build a conservative fallback plan from top candidates
            durations = pick_durations(available_minutes)
            fallback_items = []
            for dur, c in zip(durations, candidates):
                fallback_items.append({
                    "content_id": c.id,
                    "title": c.title,
                    "duration_minutes": int(min(dur, available_minutes)),
                    "why_it_helps": f"Aligned with themes {', '.join(signals.top_themes)} and fits your time.",
                    "instructions": c.body
                })
            plan_json = {"day": "today", "items": fallback_items}

        # Parse into Plan model
        try:
            items = [PlanItem(**it) for it in plan_json["items"]][:2]
            plan = Plan(day=str(plan_json.get("day", "today")), items=items)
        except Exception:
            # Very defensive fallback
            first = candidates[0]
            plan = Plan(
                day="today",
                items=[
                    PlanItem(content_id=first.id, title=first.title, duration_minutes=min(5, available_minutes), why_it_helps="Quick reset", instructions=first.body)
                ]
            )

        # Step 4: Validate guardrails
        ok, reason = validate_plan(plan)
        if not ok:
            # Minimal safe fallback
            first = candidates[0] if candidates else None
            if first:
                plan = Plan(
                    day="today",
                    items=[
                        PlanItem(content_id=first.id, title=first.title, duration_minutes=min(5, available_minutes), why_it_helps="Simple, safe, and time-bound.", instructions=first.body)
                    ],
                    caution=reason
                )
            else:
                plan = Plan(day="today", items=[], caution=reason)
        logging_db.save_plan(session_id, plan.model_dump(), signals.model_dump(), self.db_path)
        logging_db.log_step(session_id, "plan", input_obj={"prompt": plan_prompt}, output_obj=plan.model_dump(), db_path=self.db_path)

        # Step 5: Empathetic message
        emo_prompt = EMPATHETIC_TEMPLATE.format(
            signals=json.dumps(signals.model_dump(), ensure_ascii=False),
            available_minutes=available_minutes,
            candidates=json.dumps(cand_for_prompt, ensure_ascii=False),
        )
        empathic_text = self.llm.generate_text(emo_prompt)
        logging_db.log_step(session_id, "empathy", input_obj={"prompt": emo_prompt}, output_obj={"text": empathic_text}, db_path=self.db_path)

        return PlanResponse(session_id=session_id, empathetic_message=empathic_text, plan=plan, signals=signals, candidates=candidates)
