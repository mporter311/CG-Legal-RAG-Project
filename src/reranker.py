"""
reranker.py
===========
Cross-encoder reranker for pio-rag.

Takes the top-K candidates from hybrid (or dense) retrieval and reranks them
using a cross-encoder that scores (query, passage) pairs jointly.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - 6-layer MiniLM fine-tuned on MS MARCO passage ranking
  - Scores range roughly -10 to +10 (higher = more relevant)
  - ~66MB, fast on CPU (~10ms per 20 candidates)
  - No need to rebuild any index; runs over existing retrieval output

Usage (single query):
    from reranker import Reranker
    rr = Reranker()
    results = rr.rerank(query, candidates, top_k=8)

Usage (eval batch — model loaded once):
    rr = Reranker()
    for query, candidates in zip(queries, all_candidates):
        results = rr.rerank(query, candidates, top_k=8)
"""

import numpy as np
from typing import Optional


_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_MAX_LENGTH  = 512   # tokens; MiniLM supports up to 512


def _build_passage(chunk: dict) -> str:
    """
    Build the passage string for cross-encoder scoring.

    Prefixes with heading breadcrumb so the cross-encoder sees structural
    context, not just raw text. Heading path provides the legal section
    hierarchy (e.g. "Chapter 3 > Hazing and Bullying > 3.A. > Policy").
    """
    hp = chunk.get("heading_path", [])
    if isinstance(hp, str):
        import json as _json
        try:
            hp = _json.loads(hp)
        except Exception:
            hp = []
    breadcrumb = " > ".join(str(h) for h in hp) if hp else ""
    text       = chunk.get("text", "").strip()
    if breadcrumb:
        return f"{breadcrumb}\n\n{text}"
    return text


class Reranker:
    """
    Cross-encoder reranker. Load once, call rerank() repeatedly.

    Lazy-loads the model on first call to rerank() so import is always fast.
    """

    def __init__(self, model_name: str = _MODEL_NAME, max_length: int = _MAX_LENGTH):
        self.model_name = model_name
        self.max_length = max_length
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name, max_length=self.max_length)

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 8,
    ) -> list[dict]:
        """
        Rerank `candidates` for `query` and return top-k.

        Each candidate must be a chunk dict (from meta or hybrid_retrieve output).
        Returns a new list of top_k dicts with an added 'ce_score' field.

        If candidates is empty or fewer than top_k, returns as-is.
        """
        if not candidates:
            return candidates

        self._load()

        passages = [_build_passage(c) for c in candidates]
        pairs    = [(query, p) for p in passages]

        scores = self._model.predict(pairs, show_progress_bar=False)

        # Attach score and sort
        scored = sorted(
            zip(scores, candidates),
            key=lambda x: x[0],
            reverse=True,
        )

        results = []
        for ce_score, chunk in scored[:top_k]:
            c = dict(chunk)
            c["ce_score"] = float(ce_score)
            results.append(c)
        return results

    def rerank_batch(
        self,
        queries: list[str],
        all_candidates: list[list[dict]],
        top_k: int = 8,
    ) -> list[list[dict]]:
        """
        Batch rerank. Model is loaded once.
        All (query, passage) pairs across all queries are scored in one call
        for maximum throughput.

        Returns list of reranked lists, one per query.
        """
        if not queries:
            return []

        self._load()

        # Build flat list of (query, passage) pairs with offsets
        flat_pairs: list[tuple[str, str]] = []
        offsets: list[tuple[int, int]] = []   # (start, end) into flat_pairs
        for query, candidates in zip(queries, all_candidates):
            start = len(flat_pairs)
            for c in candidates:
                flat_pairs.append((query, _build_passage(c)))
            offsets.append((start, len(flat_pairs)))

        if not flat_pairs:
            return [[] for _ in queries]

        # Single batch predict — most efficient use of the model
        all_scores = self._model.predict(flat_pairs, show_progress_bar=False)

        results = []
        for (q_idx, (start, end)), candidates in zip(
            enumerate(offsets), all_candidates
        ):
            scores = all_scores[start:end]
            scored = sorted(
                zip(scores, candidates),
                key=lambda x: x[0],
                reverse=True,
            )
            reranked = []
            for ce_score, chunk in scored[:top_k]:
                c = dict(chunk)
                c["ce_score"] = float(ce_score)
                reranked.append(c)
            results.append(reranked)

        return results
