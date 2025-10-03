from __future__ import annotations

import os
from pathlib import Path
import pytest

TEST_DIR = Path(__file__).resolve().parent
API_DB_PATH = (TEST_DIR / ".." / "test_api.sqlite3").resolve()
os.environ["WB_SQLITE_PATH"] = str(API_DB_PATH)

try:
    from fastapi.testclient import TestClient  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback for lean test envs
    TestClient = None  # type: ignore

if TestClient is not None:
    from app.main import app
else:  # pragma: no cover - keep type checkers happy
    app = None
from app.orchestration import Orchestrator
from app.models import Conversation, Message, UserContext
from app import logging_db


def _fresh_db(path: Path) -> None:
    if path.exists():
        path.unlink()


def test_end_to_end_offline(tmp_path):
    db_path = tmp_path / "flow.sqlite3"
    data_path = TEST_DIR.parent / "data" / "wellbeing_content.json"
    orch = Orchestrator(content_path=str(data_path), db_path=str(db_path))
    context = UserContext(user_id="test", mood="stressed", available_minutes=10, focus_area="stress")
    conv = Conversation(
        messages=[
            Message(role="user", content="Slept poorly, anxious about deadlines, tight lower back. I can spare 10 minutes."),
            Message(role="user", content="Prefer something at my desk."),
        ]
    )
    resp = orch.run(context, conv)

    assert resp.plan.items, "Plan should include at least one item"
    assert len(resp.plan.items) <= 2
    assert resp.plan.items[0].duration_minutes <= context.available_minutes
    assert isinstance(resp.signals.sentiment, float)
    assert 0.0 <= resp.signals.sentiment_confidence <= 1.0
    assert resp.calendar_blocks, "Calendar blocks should be generated"
    assert resp.personalized_nudge, "Personalized nudge should be present"
    assert resp.life_quality is not None
    assert 0.0 <= resp.life_quality.score <= 100.0
    assert resp.life_quality.recent, "Life quality history should include recent points"

    plan_ids = {item.content_id for item in resp.plan.items}
    explanation_ids = {exp.content_id for exp in resp.explanations}
    assert plan_ids.issubset(explanation_ids), "Each plan item should have a matching explanation"

    for item in resp.plan.items:
        assert item.evidence_citation, "Each plan item should include an evidence citation"
        assert item.evidence_url and item.evidence_url.startswith("http"), "Plan item should carry study URL"

    history = logging_db.fetch_life_quality_history("test", db_path=str(db_path))
    assert history, "Life quality history should record entries"


@pytest.mark.skipif(TestClient is None, reason="fastapi not available in test environment")
def test_api_plan_and_metrics(tmp_path):
    _fresh_db(API_DB_PATH)
    client = TestClient(app)

    payload = {
        "context": {
            "user_id": "api-user",
            "available_minutes": 18,
            "mood": "optimistic",
            "focus_area": "focus",
            "preferences": ["gentle"],
            "timezone": "UTC",
        },
        "conversation": {
            "messages": [
                {"role": "user", "content": "Feeling scattered before a busy afternoon."},
                {"role": "user", "content": "Need something I can do at my desk."},
            ]
        },
    }

    response = client.post("/plan", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["plan"]["items"], "Plan items missing"
    for item in data["plan"]["items"]:
        assert item["evidence_citation"], "Citation text should be present"
        assert item.get("evidence_url", "").startswith("http"), "Citation URL should be present"
    assert data["explanations"], "Explanations should be returned"
    assert any(exp.get("url") for exp in data["explanations"]), "Explanation list should include URLs"

    metrics = client.get("/metrics/life-quality", params={"user_id": payload["context"]["user_id"], "limit": 10})
    assert metrics.status_code == 200
    metrics_data = metrics.json()
    assert metrics_data["points"], "Life quality metrics should return data"
    assert len(metrics_data["points"]) <= 10
    assert all("score" in point and "created_at" in point for point in metrics_data["points"])


def test_frontend_exists():
    root = TEST_DIR.parent
    path = root / "frontend" / "index.html"
    assert path.exists(), "Frontend entrypoint missing"
    content = path.read_text(encoding="utf-8")
    assert "Wellbeing Planner" in content
