from __future__ import annotations
from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime

class UserContext(BaseModel):
    user_id: str = Field(..., description="Unique user id")
    mood: Optional[str] = Field(None, description="Free text sentiment or mood word(s)")
    available_minutes: Optional[int] = Field(15, ge=1, le=480, description="Minutes the user can realistically spend today")
    focus_area: Optional[str] = Field(None, description="Optional focus like 'stress', 'sleep', 'mobility', 'focus'")
    preferences: Optional[List[str]] = Field(default=None, description="Keywords like 'gentle', 'low-impact', 'at-desk'")
    constraints: Optional[List[str]] = Field(default=None, description="E.g., 'no floor work', 'knee pain', 'no screens after 10pm'")
    timezone: Optional[str] = Field(default='UTC', description="IANA timezone string")

class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str

class Conversation(BaseModel):
    messages: List[Message] = Field(..., description="Multi-turn history")

class ReflectionSignals(BaseModel):
    sentiment: float = Field(..., description="VADER compound score (-1..1)")
    top_themes: List[str] = Field(..., description="Normalized themes inferred from the conversation")
    energy_level: Literal["low", "medium", "high"]
    summary: str

class ContentItem(BaseModel):
    id: str
    title: str
    summary: str
    tags: List[str]
    body: str
    score: float = 0.0

class PlanItem(BaseModel):
    content_id: str
    title: str
    duration_minutes: int
    why_it_helps: str
    instructions: str

class Plan(BaseModel):
    day: str
    items: List[PlanItem]
    caution: Optional[str] = None

class PlanResponse(BaseModel):
    session_id: str
    empathetic_message: str
    plan: Plan
    signals: ReflectionSignals
    candidates: List[ContentItem]

class StepLog(BaseModel):
    step_name: str
    started_at: str
    ended_at: str
    input: Dict[str, Any]
    output: Dict[str, Any]

class RunLog(BaseModel):
    session_id: str
    created_at: str
    steps: List[StepLog]
