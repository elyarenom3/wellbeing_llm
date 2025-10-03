from __future__ import annotations
import json, math, os, re
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
try:
    from rapidfuzz import fuzz
except Exception:  # pragma: no cover - optional dependency fallback
    from difflib import SequenceMatcher

    class _Fuzz:
        @staticmethod
        def partial_ratio(a: str, b: str) -> float:
            return SequenceMatcher(None, a, b).ratio() * 100

    fuzz = _Fuzz()
from .models import ContentExplanation, ContentItem

def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z']+", text.lower())

class EmbeddingBackend:
    """Lightweight embeddings with optional SentenceTransformer fallback."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or os.environ.get("WB_EMBED_MODEL", "all-MiniLM-L6-v2")
        self._st_model = None
        self.mode = "bow"
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            self._st_model = SentenceTransformer(self.model_name)
            self.mode = "sentence_transformers"
        except Exception:
            self._st_model = None
        self.vocab: Dict[str, int] = {}
        self.idf: np.ndarray | None = None

    def fit_transform(self, texts: Sequence[str]) -> np.ndarray:
        if self._st_model:
            emb = self._st_model.encode(
                list(texts),
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            return np.asarray(emb, dtype=np.float32)
        return self._fit_bow(texts)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if self._st_model:
            emb = self._st_model.encode(
                list(texts),
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            return np.asarray(emb, dtype=np.float32)
        if self.idf is None:
            raise RuntimeError("Embedding backend not fitted")
        doc_tokens = [_tokenize(t) for t in texts]
        return self._transform_tokens(doc_tokens)

    def _fit_bow(self, texts: Sequence[str]) -> np.ndarray:
        doc_tokens = []
        df_counts: List[int] = []
        vocab: Dict[str, int] = {}
        for text in texts:
            tokens = _tokenize(text)
            doc_tokens.append(tokens)
            unique_tokens = set(tokens)
            for token in tokens:
                if token not in vocab:
                    vocab[token] = len(vocab)
                    df_counts.append(0)
            for token in unique_tokens:
                idx = vocab[token]
                df_counts[idx] += 1
        self.vocab = vocab
        n_docs = max(1, len(texts))
        idf = np.log((1 + n_docs) / (1 + np.asarray(df_counts, dtype=np.float32))) + 1.0
        self.idf = idf
        return self._transform_tokens(doc_tokens)

    def _transform_tokens(self, doc_tokens: Sequence[Sequence[str]]) -> np.ndarray:
        if not self.vocab:
            return np.zeros((len(doc_tokens), 1), dtype=np.float32)
        idf = self.idf
        vocab_size = len(self.vocab)
        mat = np.zeros((len(doc_tokens), vocab_size), dtype=np.float32)
        for row, tokens in enumerate(doc_tokens):
            if not tokens:
                continue
            counts = Counter(tokens)
            total = float(sum(counts.values())) or 1.0
            for token, count in counts.items():
                idx = self.vocab.get(token)
                if idx is None:
                    continue
                tf = count / total
                weight = tf
                if idf is not None:
                    weight *= idf[idx]
                mat[row, idx] = weight
            norm = float(np.linalg.norm(mat[row]))
            if norm:
                mat[row] /= norm
        return mat


class ContentRepository:
    def __init__(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.items: List[ContentItem] = [ContentItem(**r) for r in raw]
        self.encoder = EmbeddingBackend()
        self.documents = [self._compose_doc(item) for item in self.items]
        embeddings = self.encoder.fit_transform(self.documents) if self.documents else np.zeros((0, 1), dtype=np.float32)
        self.vector_store = LocalVectorStore()
        for item, embedding in zip(self.items, embeddings):
            self.vector_store.add(item.id, embedding, {
                "content": item,
                "tags": item.tags,
                "summary": item.summary,
            })
        self.sentences: Dict[str, List[str]] = {
            item.id: self._split_sentences(item.body) for item in self.items
        }

    def search(self, query_text: str, themes: List[str], topk: int = 3) -> List[ContentItem]:
        if not self.items:
            return []
        query_vec = self.encoder.encode([query_text])[0]
        normalized_themes = {t.lower() for t in themes}
        hits = self.vector_store.query(query_vec, topk=topk, theme_boost=normalized_themes)
        results: List[ContentItem] = []
        for score, meta in hits:
            base_item = meta["content"]
            results.append(ContentItem(**base_item.model_dump(exclude={"score"}), score=score))
        return results

    def explain(self, query_text: str, items: Sequence[ContentItem]) -> List[ContentExplanation]:
        explanations: List[ContentExplanation] = []
        for item in items:
            snippet, snippet_score = self._best_snippet(query_text, item)
            explanations.append(
                ContentExplanation(
                    content_id=item.id,
                    snippet=snippet,
                    citation=self._citation_text(item),
                    score=snippet_score,
                    url=item.source_url,
                )
            )
        return explanations

    def _compose_doc(self, item: ContentItem) -> str:
        return " \n".join([item.title, item.summary, item.body])

    def _split_sentences(self, text: str) -> List[str]:
        chunks = re.split(r"(?<=[.!?])\s+", text.strip())
        return [c.strip() for c in chunks if c.strip()]

    def _best_snippet(self, query_text: str, item: ContentItem) -> Tuple[str, float]:
        sentences = self.sentences.get(item.id) or [item.summary or item.body]
        best = ""
        best_score = -math.inf
        for sent in sentences:
            score = float(fuzz.partial_ratio(query_text, sent) / 100)
            if score > best_score:
                best_score = score
                best = sent
        if not best:
            best = item.summary or item.body
            best_score = 0.0
        return best, best_score

    def _citation_text(self, item: ContentItem) -> str:
        if item.source_title and item.source_url:
            return f"{item.source_title}"
        primary_tag = item.tags[0] if item.tags else "general"
        return f"{item.id}:{primary_tag}"

    def citation_for(self, item: ContentItem) -> tuple[str, Optional[str]]:
        return self._citation_text(item), item.source_url


class LocalVectorStore:
    def __init__(self) -> None:
        self.ids: List[str] = []
        self.embeddings: np.ndarray = np.zeros((0, 1), dtype=np.float32)
        self.meta: Dict[str, Dict[str, Any]] = {}

    def add(self, doc_id: str, embedding: np.ndarray, metadata: Dict[str, Any]) -> None:
        embedding = np.asarray(embedding, dtype=np.float32)
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        if not self.ids:
            self.embeddings = embedding
        else:
            self.embeddings = np.vstack([self.embeddings, embedding])
        self.ids.append(doc_id)
        self.meta[doc_id] = metadata

    def query(self, query_vec: np.ndarray, topk: int = 3, theme_boost: Sequence[str] | None = None) -> List[Tuple[float, Dict[str, Any]]]:
        if not self.ids:
            return []
        query_vec = np.asarray(query_vec, dtype=np.float32)
        sims = self.embeddings @ query_vec
        theme_boost = set(theme_boost or [])
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for idx, doc_id in enumerate(self.ids):
            meta = self.meta[doc_id]
            tags = {t.lower() for t in meta.get("content").tags}
            boost = 0.12 * len(theme_boost & tags)
            scored.append((float(sims[idx] + boost), meta))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:topk]


def pick_durations(available_minutes: int) -> List[int]:
    """Heuristic: choose 1–2 items that sum to <= available_minutes, favor small chunks."""
    slots = [5, 10, 15, 20, 25, 30]
    chosen: List[int] = []
    for s in slots:
        if s <= max(5, available_minutes):
            chosen.append(s)
        if len(chosen) >= 2:
            break
    if not chosen:
        chosen = [min(5, available_minutes)]
    return chosen[:2]
