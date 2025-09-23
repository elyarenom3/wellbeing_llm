from __future__ import annotations
from typing import Dict, Any, Optional
import json, os, re

# LiteLLM is a provider-agnostic client (OpenAI/Anthropic/Ollama/etc.). Optional at runtime.
try:
    import litellm
    _HAS_LITELLM = True
except Exception:
    _HAS_LITELLM = False

class BaseLLM:
    def generate_text(self, prompt: str) -> str:
        raise NotImplementedError

    def generate_json(self, prompt: str) -> Dict[str, Any]:
        """Return a JSON object parsed from model output. If JSON parse fails, try to extract with regex."""
        text = self.generate_text(prompt)
        # Try to parse JSON directly
        try:
            return json.loads(text)
        except Exception:
            # extract first {...} block
            m = re.search(r"\{.*\}", text, flags=re.S)
            if m:
                try:
                    return json.loads(m.group(0))
                except Exception:
                    pass
        return {"raw": text}

class LiteLLMClient(BaseLLM):
    """Unified client—set model via LITELLM_MODEL env var (e.g., 'gpt-4o-mini', 'ollama/llama3.1')"""
    def __init__(self, model: Optional[str] = None):
        if not _HAS_LITELLM:
            raise RuntimeError("litellm not installed. Use RuleBasedLLM or install litellm.")
        self.model = model or os.environ.get("LITELLM_MODEL", "gpt-4o-mini")

    def generate_text(self, prompt: str) -> str:
        resp = litellm.completion(model=self.model, messages=[{"role":"user","content":prompt}])
        # litellm returns an object with choices[0].message.content (OpenAI-style) or similar
        try:
            return resp.choices[0].message.content
        except Exception:
            # Best effort
            return str(resp)

class RuleBasedLLM(BaseLLM):
    """Deterministic fallback—ensures the project runs without API keys."""
    def generate_text(self, prompt: str) -> str:
        # very simple: mirror back a warm, structured response
        if "EMPATHETIC_MESSAGE" in prompt:
            return (
                "I hear that today has been a lot—thank you for sharing. "
                "Let’s keep this light and doable. I’ve picked one quick reset and one gentle practice "
                "that fit your time and energy. If anything feels off, we’ll tweak it together."
            )
        # If JSON requested for plan, return a fixed shaped object
        if "PLAN_JSON" in prompt:
            return json.dumps({
                "day": "today",
                "items": [
                    {
                        "content_id": "ritual-breathing",
                        "title": "5-Minute Breathing Reset",
                        "duration_minutes": 5,
                        "why_it_helps": "Quick downshift for the nervous system; pairs well with low energy days.",
                        "instructions": "Inhale 4, hold 4, exhale 6 for five cycles."
                    }
                ]
            })
        return "Okay."

def build_llm() -> BaseLLM:
    # If litellm is installed and a model is set, prefer it; otherwise fallback.
    if _HAS_LITELLM and os.environ.get("LITELLM_MODEL"):
        return LiteLLMClient()
    return RuleBasedLLM()
