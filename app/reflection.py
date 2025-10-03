from __future__ import annotations
import os, re
from typing import Dict, List, Tuple
from .models import Conversation, UserContext, ReflectionSignals

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _VADER = SentimentIntensityAnalyzer()
except Exception:
    _VADER = None

THEME_KEYWORDS: Dict[str, List[str]] = {
    "stress": ["stress", "overwhelm", "anxious", "anxiety", "pressure", "tense", "deadline"],
    "sleep": ["sleep", "insomnia", "tired", "restless", "bedtime", "woke", "nap", "drowsy"],
    "mobility": ["back", "posture", "stiff", "ache", "stretch", "tight", "shoulder", "neck"],
    "focus": ["distract", "focus", "concentrate", "procrastinate", "deep work", "productive", "keep up"],
    "gratitude": ["grateful", "gratitude", "thankful", "appreciate", "blessings"],
    "energy": ["exhausted", "fatigued", "energized", "sluggish", "wired", "alert", "vibrant"],
    "mood": ["sad", "down", "blue", "happy", "joy", "calm", "irritable", "frustrated"],
    "loneliness": ["lonely", "isolated", "alone", "no one", "disconnected"],
    "body image": ["body", "appearance", "weight", "image", "mirror", "self-conscious"],
    "burnout": ["burnout", "burn out", "burned out", "burnt out", "drained", "fried", "overloaded"],
}

class SentimentService:
    def __init__(self) -> None:
        self.backend_name = "distilbert"
        self._distilbert = self._load_distilbert()
        if self._distilbert is None:
            if _VADER is not None:
                self.backend_name = "vader"
            else:
                self.backend_name = "heuristic"

    def _load_distilbert(self):
        model_name = os.environ.get(
            "WB_SENTIMENT_MODEL",
            "distilbert-base-uncased-finetuned-sst-2-english",
        )
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore
            import torch

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            model.eval()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)

            def _score(text: str) -> Tuple[float, float, float]:
                inputs = tokenizer(
                    text,
                    truncation=True,
                    max_length=256,
                    padding=False,
                    return_tensors="pt",
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=1)[0]
                positive = float(probs[1].item())
                negative = float(probs[0].item())
                sentiment = positive - negative
                confidence = max(positive, negative)
                calibrated = max(-1.0, min(1.0, (positive - 0.55) / 0.45))
                return sentiment, confidence, calibrated

            return _score
        except Exception:
            return None

    def score(self, text: str) -> Tuple[float, float, float, str]:
        if self.backend_name == "distilbert" and self._distilbert is not None:
            raw, confidence, calibrated = self._distilbert(text)
            return raw, confidence, calibrated, "distilbert"
        if self.backend_name == "vader" and _VADER is not None:
            compound = _VADER.polarity_scores(text).get("compound", 0.0)
            confidence = max(0.2, 1.0 - abs(compound) * 0.25)
            calibrated = max(-1.0, min(1.0, compound * 0.9))
            return compound, confidence, calibrated, "vader"
        # Heuristic fallback
        pos_words = ["good", "great", "calm", "happy", "okay", "content"]
        neg_words = ["bad", "tired", "stressed", "sad", "anxious", "overwhelmed"]
        score = 0.0
        tl = text.lower()
        for w in pos_words:
            if w in tl:
                score += 0.4
        for w in neg_words:
            if w in tl:
                score -= 0.4
        score = max(-1.0, min(1.0, score))
        confidence = 0.35
        calibrated = score * 0.8
        return score, confidence, calibrated, "heuristic"


_SENTIMENT_SERVICE = SentimentService()


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
            key = p.lower()
            if key in THEME_KEYWORDS:
                found.add(key)
    if not found:
        found.add("stress")
    return sorted(found)


def _energy_from_text(text: str) -> str:
    tl = text.lower()
    if any(k in tl for k in ["exhausted", "tired", "drained", "burned out", "burnt out", "wiped", "burnout"]):
        return "low"
    if any(k in tl for k in ["wired", "alert", "motivated", "ready", "pumped", "energized"]):
        return "high"
    return "medium"


def _reflection_score(calibrated: float, themes: List[str], energy: str) -> float:
    theme_bonus = min(0.2, 0.05 * len(themes))
    energy_adjust = {"low": -0.1, "medium": 0.0, "high": 0.08}.get(energy, 0.0)
    raw = 0.5 + 0.4 * calibrated + theme_bonus + energy_adjust
    return max(0.0, min(1.0, raw))


def analyze(conversation: Conversation, context: UserContext) -> ReflectionSignals:
    full_text = "\n".join([m.content for m in conversation.messages])
    merged = full_text if context.mood is None else (full_text + " " + context.mood)
    raw_sentiment, confidence, calibrated, backend = _SENTIMENT_SERVICE.score(merged)
    themes = _themes(full_text, context)
    energy = _energy_from_text(merged)
    reflection_score = _reflection_score(calibrated, themes, energy)
    summary = (
        f"Sentiment {raw_sentiment:.2f} ({backend}, conf {confidence:.2f}). "
        f"Calibrated {calibrated:.2f}. Themes: {', '.join(themes)}. Energy: {energy}."
    )
    return ReflectionSignals(
        sentiment=raw_sentiment,
        sentiment_confidence=confidence,
        sentiment_calibrated=calibrated,
        sentiment_model=backend,
        reflection_score=reflection_score,
        top_themes=themes,
        energy_level=energy,
        summary=summary,
    )
