from __future__ import annotations
import os
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from .models import UserContext, Conversation
from .orchestration import Orchestrator
from . import logging_db

DATA_PATH = os.environ.get("WB_DATA_PATH", os.path.join(os.path.dirname(__file__), "..", "data", "wellbeing_content.json"))
DB_PATH = os.environ.get("WB_SQLITE_PATH", os.path.join(os.path.dirname(__file__), "..", "wellbeing_logs.sqlite3"))
FRONTEND_PATH = os.path.join(os.path.dirname(__file__), "..", "frontend")

app = FastAPI(title="Wellbeing Planner API", version="0.1.0")

# Single orchestrator instance
_orch = Orchestrator(content_path=os.path.abspath(DATA_PATH), db_path=os.path.abspath(DB_PATH))

# Static assets (logo, future frontend files)
assets_path = os.path.join(FRONTEND_PATH, "assets")
if os.path.isdir(assets_path):
    app.mount("/assets", StaticFiles(directory=assets_path), name="assets")

class PlanRequest(BaseModel):
    context: UserContext
    conversation: Conversation

@app.post("/plan")
def make_plan(req: PlanRequest):
    resp = _orch.run(req.context, req.conversation)
    return JSONResponse(content=resp.model_dump())


@app.get("/metrics/life-quality")
def life_quality_metrics(user_id: str = Query(..., description="User identifier"), limit: int = Query(14, ge=1, le=90)):
    history = logging_db.fetch_life_quality_history(user_id, limit=limit, db_path=DB_PATH)
    plotly_spec = {
        "data": [
            {
                "type": "scatter",
                "mode": "lines+markers",
                "x": [row["created_at"] for row in history],
                "y": [row["score"] for row in history],
                "name": "LQI-lite",
            }
        ],
        "layout": {
            "title": f"Life Quality Signal for {user_id}",
            "yaxis": {"range": [0, 100], "title": "Score"},
            "xaxis": {"title": "Timestamp"},
            "margin": {"l": 40, "r": 10, "t": 40, "b": 30},
        },
    }
    return {"user_id": user_id, "points": history, "plotly": plotly_spec}


@app.get("/", include_in_schema=False)
def frontdoor():
    index_path = os.path.join(FRONTEND_PATH, "index.html")
    if not os.path.exists(index_path):
        return JSONResponse({"message": "Frontend not found"}, status_code=404)
    return FileResponse(index_path)
