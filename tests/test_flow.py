from __future__ import annotations
import os, json
import os
from app.orchestration import Orchestrator
from app.models import UserContext, Conversation, Message
from app import logging_db

def test_end_to_end():
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "wellbeing_content.json")
    orch = Orchestrator(content_path=data_path, db_path=os.path.join(os.path.dirname(__file__), "..", "test_logs.sqlite3"))
    context = UserContext(user_id="test", mood="stressed", available_minutes=10, focus_area="stress")
    conv = Conversation(messages=[
        Message(role="user", content="Slept poorly, anxious about deadlines, tight lower back. I can spare 10 minutes."),
        Message(role="user", content="Prefer something at my desk.")
    ])
    resp = orch.run(context, conv)
    assert resp.plan.items, "Plan should include at least one item"
    assert resp.plan.items[0].duration_minutes <= context.available_minutes
    assert isinstance(resp.signals.sentiment, float)
    assert 0.0 <= resp.signals.sentiment_confidence <= 1.0
    assert resp.explanations, "Expected at least one explanation"
    assert resp.calendar_blocks, "Calendar blocks should be generated"
    assert resp.personalized_nudge, "Personalized nudge should be present"
    assert resp.life_quality is not None
    assert 0.0 <= resp.life_quality.score <= 100.0
    assert resp.life_quality.recent, "Life quality history should include recent points"
    for item in resp.plan.items:
        assert item.evidence_citation, "Each plan item should include an evidence citation"
    history = logging_db.fetch_life_quality_history("test", db_path=os.path.join(os.path.dirname(__file__), "..", "test_logs.sqlite3"))
    assert history, "Life quality history should record entries"


def test_frontend_exists():
    root = os.path.join(os.path.dirname(__file__), "..")
    path = os.path.join(root, "frontend", "index.html")
    assert os.path.exists(path), "Frontend entrypoint missing"
    with open(path, "r", encoding="utf-8") as fh:
        content = fh.read()
    assert "Wellbeing Planner" in content
