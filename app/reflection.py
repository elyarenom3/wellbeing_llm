from __future__ import annotations
from typing import List, Dict
from .models import Conversation, UserContext, ReflectionSignals
import re

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _VADER = SentimentIntensityAnalyzer()
except Exception:
    _VADER = None

THEME_KEYWORDS = {
    "stress": ["stress", "overwhelm", "anxious", "anxiety", "pressure", "tense"],
    "sleep": ["sleep", "insomnia", "tired", "restless", "bedtime", "woke", "nap"],
    "mobility": ["back", "posture", "stiff", "ache", "stretch", "tight"],
    "focus": ["distract", "focus", "concentrate", "procrastinate", "deep work", "productive"],
    "gratitude": ["grateful", "gratitude", "thankful", "appreciate", "blessings"],
    "energy": ["exhausted", "fatigued", "energized", "sluggish", "wired", "alert"],
    "mood": ["sad", "down", "blue", "happy", "joy", "calm", "irritable"]
}

def _sentiment(text: str) -> float:
    if _VADER is None:
        # Fallback simple heuristic
        pos_words = ["good", "great", "calm", "happy", "okay", "content"]
        neg_words = ["bad", "tired", "stressed", "sad", "anxious", "overwhelmed"]
        score = 0
        tl = text.lower()
        for w in pos_words: 
            if w in tl: score += 0.5
        for w in neg_words: 
            if w in tl: score -= 0.5
        return max(-1.0, min(1.0, score))
    return _VADER.polarity_scores(text).get("compound", 0.0)

def _themes(text: str, context: UserContext) -> List[str]:
    found = set()
    tl = text.lower()
    for theme, kws in THEME_KEYWORDS.items():
        for kw in kws:
            if re.search(rf"\b{re.escape(kw)}\b", tl):
                found.add(theme)
                break
    if context.focus_area:
        found.add(context.focus_area.lower())
    if context.preferences:
        for p in context.preferences:
            if p.lower() in THEME_KEYWORDS:
                found.add(p.lower())
    if not found:
        found.add("stress")
    return list(found)

def _energy_from_text(text: str) -> str:
    tl = text.lower()
    if any(k in tl for k in ["exhausted", "tired", "drained", "burned out", "burnt out", "wiped"]):
        return "low"
    if any(k in tl for k in ["wired", "alert", "motivated", "ready", "pumped"]):
        return "high"
    return "medium"

def analyze(conversation: Conversation, context: UserContext) -> ReflectionSignals:
    full_text = "\n".join([m.content for m in conversation.messages])
    s = _sentiment(full_text)
    themes = _themes(full_text, context)
    energy = _energy_from_text(full_text if context.mood is None else (full_text + " " + context.mood))
    summary = f"Sentiment {s:.2f}. Themes: {', '.join(sorted(themes))}. Energy: {energy}."
    return ReflectionSignals(sentiment=s, top_themes=themes, energy_level=energy, summary=summary)
