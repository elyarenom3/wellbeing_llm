from __future__ import annotations
from typing import List, Tuple
from .models import Plan

SAFE_TAGS = {"stress", "breathing", "mindfulness", "mobility", "lower_back", "energy", "gratitude", "sleep", "journaling"}

def validate_plan(plan: Plan) -> Tuple[bool, str]:
    total = sum(item.duration_minutes for item in plan.items)
    if total <= 0:
        return False, "Plan has no time allocated."
    for it in plan.items:
        if it.duration_minutes <= 0 or it.duration_minutes > 240:
            return False, f"Invalid duration for item {it.title}."
    # soft safety check
    for it in plan.items:
        # we don't have tags on PlanItem; rely on content titles being from our curated set
        if any(x in it.title.lower() for x in ["fasting", "ice bath", "supplement"]):
            return False, "Potentially unsafe recommendation detected."
    return True, ""
