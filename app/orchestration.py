from __future__ import annotations
import json, os, uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
from .models import (
    CalendarBlock,
    ContentExplanation,
    ContentItem,
    Plan,
    PlanItem,
    PlanResponse,
    ReflectionSignals,
    RunLog,
    UserContext,
    Conversation,
)
from .providers import build_llm
from .reflection import analyze
from .retrieval import ContentRepository, pick_durations
from .validators import validate_plan
from .prompts import EMPATHETIC_TEMPLATE, PLAN_JSON_TEMPLATE
from . import logging_db, privacy
from .calendar_tools import build_calendar_blocks
from .life_quality import build_lqi_report

def _now_iso() -> str:
    return datetime.utcnow().isoformat()

class Orchestrator:
    def __init__(self, content_path: str, db_path: str | None = None) -> None:
        self.repo = ContentRepository(content_path)
        privacy.enforce_local_only_mode()
        self.llm = build_llm()
        self.db_path = db_path
        logging_db.init_db(db_path)
        retention_days = int(os.environ.get("WB_RETENTION_DAYS", "30"))
        if retention_days > 0:
            logging_db.run_retention_cleanup(retention_days, db_path)

    def run(self, context: UserContext, conversation: Conversation) -> PlanResponse:
        session_id = str(uuid.uuid4())
        created_at = _now_iso()
        logging_db.create_session(session_id, context.user_id, self.db_path, created_at=created_at)

        effective_conversation = privacy.redact_conversation(conversation)
        effective_context = context
        if privacy.is_privacy_mode():
            context_payload = context.model_dump()
            if context_payload.get("mood"):
                context_payload["mood"] = privacy.redact_text(context_payload["mood"])
            effective_context = UserContext(**context_payload)

        # Step 1: Reflection
        signals = analyze(effective_conversation, effective_context)
        logging_db.log_step(
            session_id,
            "reflection",
            input_obj=privacy.sanitize_context(context.model_dump()) | {"conversation": privacy.sanitize_conversation_dict(conversation.model_dump())},
            output_obj=signals.model_dump(),
            db_path=self.db_path,
        )

        # Step 1b: Update user metrics for personalization
        previous_metrics = logging_db.get_user_metrics(context.user_id, self.db_path)
        current_metrics = self._update_user_metrics(context.user_id, signals, previous_metrics, created_at)
        logging_db.log_step(
            session_id,
            "user_metrics",
            input_obj={"previous": previous_metrics or {}},
            output_obj=current_metrics,
            db_path=self.db_path,
        )

        # Step 2: Retrieve candidate interventions
        joined_text = " ".join([m.content for m in effective_conversation.messages])
        candidates: List[ContentItem] = self.repo.search(joined_text, signals.top_themes, topk=5)
        logging_db.log_step(
            session_id,
            "retrieval",
            input_obj={"query": privacy.redact_text(joined_text), "themes": signals.top_themes},
            output_obj={"candidates": [c.model_dump() for c in candidates]},
            db_path=self.db_path,
        )
        explanations = self.repo.explain(joined_text, candidates[:3])

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
                rationale = next((exp.snippet for exp in explanations if exp.content_id == c.id), c.summary)
                citation_text, citation_url = self.repo.citation_for(c)
                fallback_items.append({
                    "content_id": c.id,
                    "title": c.title,
                    "duration_minutes": int(min(dur, available_minutes)),
                    "why_it_helps": f"{rationale} (themes: {', '.join(signals.top_themes)})",
                    "instructions": c.body,
                    "evidence_citation": citation_text,
                    "evidence_url": citation_url,
                })
            plan_json = {"day": "today", "items": fallback_items}

        try:
            items = [PlanItem(**it) for it in plan_json["items"]][:2]
            plan = Plan(day=str(plan_json.get("day", "today")), items=items)
        except Exception:
            # Very defensive fallback
            first = candidates[0]
            citation_text, citation_url = self.repo.citation_for(first)
            plan = Plan(
                day="today",
                items=[
                    PlanItem(
                        content_id=first.id,
                        title=first.title,
                        duration_minutes=min(5, available_minutes),
                        why_it_helps="Quick reset",
                        instructions=first.body,
                        evidence_citation=citation_text,
                        evidence_url=citation_url,
                    )
                ]
            )

        plan = self._enrich_plan_with_citations(plan, candidates)

        # Validate guardrails
        ok, reason = validate_plan(plan)
        if not ok:
            # Minimal safe fallback
            first = candidates[0] if candidates else None
            if first:
                citation_text, citation_url = self.repo.citation_for(first)
                plan = Plan(
                    day="today",
                    items=[
                        PlanItem(
                            content_id=first.id,
                            title=first.title,
                            duration_minutes=min(5, available_minutes),
                            why_it_helps="Simple, safe, and time-bound. Built from curated wellbeing content.",
                            instructions=first.body,
                            evidence_citation=citation_text,
                            evidence_url=citation_url,
                        )
                    ],
                    caution=reason
                )
            else:
                plan = Plan(day="today", items=[], caution=reason)
        plan = self._enrich_plan_with_citations(plan, candidates)
        logging_db.save_plan(session_id, plan.model_dump(), signals.model_dump(), self.db_path)
        logging_db.log_step(session_id, "plan", input_obj={"prompt": plan_prompt}, output_obj=plan.model_dump(), db_path=self.db_path)

        # Empathetic message
        emo_prompt = EMPATHETIC_TEMPLATE.format(
            signals=json.dumps(signals.model_dump(), ensure_ascii=False),
            available_minutes=available_minutes,
            candidates=json.dumps(cand_for_prompt, ensure_ascii=False),
        )
        empathic_text = self.llm.generate_text(emo_prompt)
        logging_db.log_step(session_id, "empathy", input_obj={"prompt": emo_prompt}, output_obj={"text": empathic_text}, db_path=self.db_path)

        plan_explanations = self._plan_explanations(joined_text, plan, candidates, explanations)
        calendar_blocks = build_calendar_blocks(plan, context)
        personalized_nudge = self._personalized_nudge(previous_metrics, current_metrics, signals)
        # Ensure each plan item has an evidence citation (belt and suspenders)
        plan = self._enrich_plan_with_citations(plan, candidates)

        life_quality = build_lqi_report(
            context.user_id,
            session_id,
            signals,
            effective_conversation,
            current_metrics,
            self.db_path,
        )

        return PlanResponse(
            session_id=session_id,
            empathetic_message=empathic_text,
            plan=plan,
            signals=signals,
            candidates=candidates,
            explanations=plan_explanations,
            calendar_blocks=calendar_blocks,
            personalized_nudge=personalized_nudge,
            life_quality=life_quality,
        )

    def _enrich_plan_with_citations(self, plan: Plan, candidates: List[ContentItem]) -> Plan:
        candidates_by_id = {c.id: c for c in candidates}
        for item in plan.items:
            content_ref = candidates_by_id.get(item.content_id)
            if content_ref:
                citation_text, citation_url = self.repo.citation_for(content_ref)
                if not item.evidence_citation:
                    item.evidence_citation = citation_text
                if not item.evidence_url and citation_url:
                    item.evidence_url = citation_url
        return plan

    def _plan_explanations(
        self,
        query_text: str,
        plan: Plan,
        candidates: List[ContentItem],
        candidate_explanations: List[ContentExplanation],
    ) -> List[ContentExplanation]:
        by_id = {c.id: c for c in candidates}
        explanations = []
        for item in plan.items:
            content = by_id.get(item.content_id)
            if not content:
                continue
            explanations.extend(self.repo.explain(query_text, [content]))
        if explanations:
            return explanations
        return candidate_explanations

    def _update_user_metrics(
        self,
        user_id: str,
        signals: ReflectionSignals,
        previous: Optional[Dict[str, Any]],
        created_at: str,
    ) -> Dict[str, Any]:
        now_dt = datetime.fromisoformat(created_at)
        prev_streak = int(previous.get("streak", 0)) if previous else 0
        prev_total = int(previous.get("total_sessions", 0)) if previous else 0
        prev_seen = previous.get("last_seen") if previous else None
        prev_rolling = previous.get("rolling_reflection_score") if previous else None

        streak = 1
        if prev_seen:
            try:
                last_dt = datetime.fromisoformat(prev_seen)
                diff_days = (now_dt.date() - last_dt.date()).days
                if diff_days == 0:
                    streak = prev_streak
                elif diff_days == 1:
                    streak = prev_streak + 1
                else:
                    streak = 1
            except Exception:
                streak = max(1, prev_streak)

        total_sessions = prev_total + 1
        if streak <= 0:
            streak = 1

        reflection_score = float(signals.reflection_score)
        if prev_rolling is None:
            rolling = reflection_score
        else:
            rolling = 0.7 * float(prev_rolling) + 0.3 * reflection_score

        logging_db.upsert_user_metrics(
            user_id,
            streak,
            total_sessions,
            created_at,
            reflection_score,
            rolling,
            db_path=self.db_path,
        )

        return {
            "user_id": user_id,
            "streak": streak,
            "total_sessions": total_sessions,
            "last_seen": created_at,
            "last_reflection_score": reflection_score,
            "rolling_reflection_score": rolling,
        }

    def _personalized_nudge(
        self,
        previous: Optional[Dict[str, Any]],
        current: Dict[str, Any],
        signals: ReflectionSignals,
    ) -> Optional[str]:
        if not current:
            return None
        streak = current.get("streak", 0)
        prev_roll = previous.get("rolling_reflection_score") if previous else None
        new_roll = current.get("rolling_reflection_score")
        parts: List[str] = []
        if streak and streak >= 3:
            parts.append(f"You're on a {streak}-day streak—great consistency.")
        elif streak == 2:
            parts.append("Two days in a row! Momentum matters.")
        elif streak == 1:
            parts.append("Thanks for checking in today—every reflection counts.")

        if prev_roll is not None and new_roll is not None:
            delta = float(signals.reflection_score) - float(prev_roll)
            if delta > 0.05:
                parts.append("Reflection score is trending up. Keep leaning on what helps.")
            elif delta < -0.05:
                parts.append("Noticed a dip—consider a lighter activity or support reach-out.")
        elif new_roll is not None:
            parts.append("We'll track how today's plan supports your wellbeing.")

        if not parts:
            return None
        return " ".join(parts)
