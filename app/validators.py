from __future__ import annotations
from typing import Optional, Tuple
from .models import Plan

SAFE_TAGS = {"stress", "breathing", "mindfulness", "mobility", "lower_back", "energy", "gratitude", "sleep", "journaling"}

def validate_plan(plan: Plan) -> Tuple[bool, str]:
    total = sum(item.duration_minutes for item in plan.items)
    if total <= 0:
        return False, "Plan has no time allocated."
    for it in plan.items:
        if it.duration_minutes <= 0 or it.duration_minutes > 240:
            return False, f"Invalid duration for item {it.title}."
        if not it.evidence_citation:
            return False, f"Missing evidence citation for {it.content_id}."
    # soft safety check
    for it in plan.items:
        # we don't have tags on PlanItem; rely on content titles being from our curated set
        if any(x in it.title.lower() for x in ["fasting", "ice bath", "supplement"]):
            return False, "Potentially unsafe recommendation detected."
    return True, ""


def cap_life_quality_delta(previous: Optional[float], candidate: float, limit: float = 15.0) -> float:
    if previous is None:
        return float(max(0.0, min(100.0, candidate)))
    delta = candidate - previous
    if delta > limit:
        candidate = previous + limit
    elif delta < -limit:
        candidate = previous - limit
    return float(max(0.0, min(100.0, candidate)))
