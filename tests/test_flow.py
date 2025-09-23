from __future__ import annotations
import os, json
from app.orchestration import Orchestrator
from app.models import UserContext, Conversation, Message

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
