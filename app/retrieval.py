from __future__ import annotations
import json, os, math
from typing import List
from rapidfuzz import fuzz
from .models import ContentItem, UserContext, ReflectionSignals

class ContentRepository:
    def __init__(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.items: List[ContentItem] = [ContentItem(**r) for r in raw]

    def search(self, query_text: str, themes: List[str], topk: int = 3) -> List[ContentItem]:
        results = []
        for item in self.items:
            # score components: theme overlap, fuzzy match on title/summary/body
            theme_overlap = len(set(t.lower() for t in themes) & set(t.lower() for t in item.tags))
            fuzzy_title = fuzz.token_set_ratio(query_text, item.title)
            fuzzy_summary = fuzz.token_set_ratio(query_text, item.summary)
            fuzzy_body = fuzz.token_set_ratio(query_text, item.body)
            score = 2.5 * theme_overlap + 0.3 * fuzzy_title + 0.2 * fuzzy_summary + 0.1 * fuzzy_body
            results.append(ContentItem(**item.dict(exclude={"score"}), score=float(score)))
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:topk]

def pick_durations(available_minutes: int) -> List[int]:
    """Heuristic: choose 1–2 items that sum to <= available_minutes, favor small chunks."""
    slots = [5, 10, 15, 20, 25, 30]
    chosen = []
    for s in slots:
        if s <= max(5, available_minutes):
            chosen.append(s)
        if len(chosen) >= 2:
            break
    if not chosen:
        chosen = [min(5, available_minutes)]
    return chosen[:2]
