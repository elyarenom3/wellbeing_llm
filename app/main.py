from __future__ import annotations
import os
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from .models import UserContext, Conversation
from .orchestration import Orchestrator

DATA_PATH = os.environ.get("WB_DATA_PATH", os.path.join(os.path.dirname(__file__), "..", "data", "wellbeing_content.json"))
DB_PATH = os.environ.get("WB_SQLITE_PATH", os.path.join(os.path.dirname(__file__), "..", "wellbeing_logs.sqlite3"))

app = FastAPI(title="Wellbeing Planner API", version="0.1.0")

# Single orchestrator instance
_orch = Orchestrator(content_path=os.path.abspath(DATA_PATH), db_path=os.path.abspath(DB_PATH))

class PlanRequest(BaseModel):
    context: UserContext
    conversation: Conversation

@app.post("/plan")
def make_plan(req: PlanRequest):
    resp = _orch.run(req.context, req.conversation)
    return JSONResponse(content=resp.model_dump())
