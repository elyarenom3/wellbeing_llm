from __future__ import annotations
from datetime import datetime
from typing import Dict, List, Optional, Any

from .models import Conversation, LifeQualityReport, LifeQualitySnapshot, ReflectionSignals
from . import logging_db
from .validators import cap_life_quality_delta


def infer_action_adherence(conversation: Conversation) -> float:
    text = " ".join([m.content.lower() for m in conversation.messages])
    if not text:
        return 0.6
    positive = any(k in text for k in ["completed", "did", "finished", "done", "stuck with", "followed"])
    negative = any(k in text for k in ["skipped", "couldn't", "didn't", "avoid", "failed"])
    if positive and not negative:
        return 0.85
    if negative and not positive:
        return 0.35
    if positive and negative:
        return 0.55
    return 0.6


def compute_lqi_score(signals: ReflectionSignals, sentiment_delta: float, adherence: float) -> float:
    sentiment_component = max(-1.0, min(1.0, sentiment_delta)) * 20  # +/-20 swing
    calibrated = max(-1.0, min(1.0, signals.sentiment_calibrated)) * 25
    theme_penalty = -10 if "stress" in signals.top_themes else 0
    theme_penalty += -5 if "sleep" in signals.top_themes else 0
    adherence_component = (adherence - 0.5) * 40
    base = 65
    score = base + sentiment_component + calibrated + theme_penalty + adherence_component
    return float(max(0.0, min(100.0, score)))


def build_lqi_report(user_id: str, session_id: str, signals: ReflectionSignals, conversation: Conversation, metrics: Optional[Dict[str, Any]], db_path: Optional[str]) -> LifeQualityReport:
    adherence = infer_action_adherence(conversation)
    prev_score = (metrics or {}).get("rolling_reflection_score")
    sentiment_delta = 0.0
    if prev_score is not None:
        sentiment_delta = signals.reflection_score - float(prev_score)
    raw_score = compute_lqi_score(signals, sentiment_delta, adherence)

    history = logging_db.fetch_life_quality_history(user_id, db_path=db_path)
    previous_score = history[-1]["score"] if history else None
    lqi_score = cap_life_quality_delta(previous_score, raw_score)
    history.append({"score": lqi_score, "created_at": datetime.utcnow().isoformat()})
    logging_db.record_life_quality(
        session_id,
        user_id,
        lqi_score,
        {
            "sentiment_delta": sentiment_delta,
            "adherence": adherence,
            "themes": signals.top_themes,
        },
        db_path=db_path,
    )

    snapshots: List[LifeQualitySnapshot] = [
        LifeQualitySnapshot(timestamp=item["created_at"], score=float(item["score"]))
        for item in history[-7:]
    ]
    trend = "steady"
    if len(snapshots) >= 2:
        diff = snapshots[-1].score - snapshots[0].score
        if diff > 2:
            trend = "up"
        elif diff < -2:
            trend = "down"
    return LifeQualityReport(score=lqi_score, trend=trend, recent=snapshots)
