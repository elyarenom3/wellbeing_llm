from __future__ import annotations
import base64
import json
import os
import re
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from .models import Conversation, Message

_PRIVACY_MODE = os.environ.get("PRIVACY_MODE", "false").lower() in {"1", "true", "yes", "on"}
_PRIVACY_LOGGING_OPTOUT = os.environ.get("PRIVACY_LOGGING_OPTOUT", "false").lower() in {"1", "true", "yes", "on"}
_PRIVACY_KEY_PATH = os.environ.get(
    "WB_PRIVACY_KEY_PATH",
    os.path.join(os.path.dirname(__file__), "..", "privacy_key.json"),
)
_KEY_ROTATION_DAYS = int(os.environ.get("WB_PRIVACY_KEY_ROTATION_DAYS", "30"))


@dataclass
class SanitizedConversation:
    messages: List[Message]


def is_privacy_mode() -> bool:
    return _PRIVACY_MODE


def should_log() -> bool:
    if _PRIVACY_LOGGING_OPTOUT and is_privacy_mode():
        return False
    return True


def _load_or_rotate_key() -> bytes:
    if not is_privacy_mode():
        return b""
    try:
        if os.path.exists(_PRIVACY_KEY_PATH):
            with open(_PRIVACY_KEY_PATH, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            rotated = datetime.fromisoformat(payload.get("rotated_at"))
            if datetime.utcnow() - rotated >= timedelta(days=_KEY_ROTATION_DAYS):
                raise ValueError("rotation due")
            key_b64 = payload.get("key")
            if key_b64:
                return base64.urlsafe_b64decode(key_b64.encode("utf-8"))
    except Exception:
        pass
    key = secrets.token_bytes(32)
    os.makedirs(os.path.dirname(_PRIVACY_KEY_PATH), exist_ok=True)
    with open(_PRIVACY_KEY_PATH, "w", encoding="utf-8") as fh:
        json.dump({
            "key": base64.urlsafe_b64encode(key).decode("utf-8"),
            "rotated_at": datetime.utcnow().isoformat(),
        }, fh)
    return key


def _privacy_key() -> bytes:
    if not is_privacy_mode():
        return b""
    key = os.environ.get("WB_PRIVACY_KEY")
    if key:
        try:
            return base64.urlsafe_b64decode(key)
        except Exception:
            pass
    derived = _load_or_rotate_key()
    os.environ["WB_PRIVACY_KEY"] = base64.urlsafe_b64encode(derived).decode("utf-8")
    return derived


def encrypt_payload(payload: Dict[str, Any]) -> str:
    data = json.dumps(payload, ensure_ascii=False)
    if not is_privacy_mode():
        return data
    key = _privacy_key()
    if not key:
        return data
    plain = data.encode("utf-8")
    cipher = bytes([plain[i] ^ key[i % len(key)] for i in range(len(plain))])
    return base64.urlsafe_b64encode(cipher).decode("utf-8")


def redact_text(text: str) -> str:
    if not text:
        return text
    scrubbed = re.sub(r"[A-Za-z0-9_.+-]+@[A-Za-z0-9-]+\.[A-Za-z0-9-.]+", "<email>", text)
    scrubbed = re.sub(r"\b\+?\d[\d\-\s]{7,}\b", "<phone>", scrubbed)
    scrubbed = re.sub(r"\b\d{4,}\b", "<number>", scrubbed)
    # Basic proper-name redaction (capitalized words that look like names)
    scrubbed = re.sub(r"\b([A-Z][a-z]{2,})\s([A-Z][a-z]{2,})\b", "<name>", scrubbed)
    return scrubbed


def redact_conversation(conversation: Conversation) -> Conversation:
    if not is_privacy_mode():
        return conversation
    redacted_messages: List[Message] = []
    for msg in conversation.messages:
        redacted_messages.append(Message(role=msg.role, content=redact_text(msg.content)))
    return Conversation(messages=redacted_messages)


def sanitize_for_logging(step_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    if not is_privacy_mode():
        return payload
    sanitized: Dict[str, Any] = {"step": step_name}
    if step_name == "reflection":
        sanitized.update({
            "signals": payload.get("output", payload.get("signals"))
        })
    elif step_name == "retrieval":
        candidates = payload.get("candidates") or payload.get("output", {}).get("candidates")
        sanitized["candidates"] = [
            {"id": c.get("id"), "score": c.get("score"), "tags": c.get("tags")}
            for c in candidates or []
        ]
    elif step_name == "plan":
        plan = payload.get("output", payload)
        sanitized["items"] = [
            {
                "content_id": item.get("content_id"),
                "duration_minutes": item.get("duration_minutes"),
                "citation": item.get("evidence_citation"),
            }
            for item in (plan.get("items") or [])
        ]
    elif step_name == "empathy":
        sanitized["text_hash"] = hash(payload.get("output", {}).get("text", ""))
    else:
        sanitized.update(payload)
    return sanitized


def sanitize_context(context_dict: Dict[str, Any]) -> Dict[str, Any]:
    if not is_privacy_mode():
        return context_dict
    keep = {k: context_dict.get(k) for k in ["user_id", "available_minutes", "focus_area", "preferences", "timezone"] if context_dict.get(k) is not None}
    mood = context_dict.get("mood")
    if mood:
        keep["mood"] = redact_text(mood)
    return keep


def sanitize_conversation_dict(conv_dict: Dict[str, Any]) -> Dict[str, Any]:
    if not is_privacy_mode():
        return conv_dict
    return {"messages": [{"role": msg.get("role"), "content": redact_text(msg.get("content", ""))[:40]} for msg in conv_dict.get("messages", [])]}


def enforce_local_only_mode() -> None:
    if not is_privacy_mode():
        return
    # In privacy mode ensure remote provider vars are unset
    for key in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "LITELLM_MODEL"]:
        os.environ.pop(key, None)


def prepare_step_storage(step_name: str, input_obj: Dict[str, Any], output_obj: Dict[str, Any]) -> Dict[str, str]:
    sanitized_input = sanitize_for_logging(step_name + "_input", input_obj)
    sanitized_output = sanitize_for_logging(step_name + "_output", output_obj)
    return {
        "input": encrypt_payload(sanitized_input),
        "output": encrypt_payload(sanitized_output),
    }


def prepare_plan_storage(plan_obj: Dict[str, Any], signals_obj: Dict[str, Any]) -> Dict[str, str]:
    if is_privacy_mode():
        plan_payload = {
            "items": [
                {
                    "content_id": item.get("content_id"),
                    "duration_minutes": item.get("duration_minutes"),
                    "citation": item.get("evidence_citation"),
                }
                for item in plan_obj.get("items", [])
            ]
        }
        signals_payload = {k: signals_obj.get(k) for k in ["sentiment", "sentiment_calibrated", "top_themes", "energy_level", "reflection_score"] if k in signals_obj}
    else:
        plan_payload = plan_obj
        signals_payload = signals_obj
    return {
        "plan": encrypt_payload(plan_payload),
        "signals": encrypt_payload(signals_payload),
    }
