"""Top-k vector retrieval over chunked context.

`prepare(doc, doc_id)` chunks + embeds the document once per
(doc_id, chunk_size). `retrieve(query, k)` cosine-scores the query
against the cached doc matrix and returns the top-k chunks in
score-descending order.
"""

from __future__ import annotations

import time

import numpy as np

from harness.chunker import chunk
from harness.embedder import embed_query, embed_texts
from harness.retrievers.base import RetrieveResult


def _normalize_rows(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return m / norms


class TopKRetriever:
    name = "topk"

    def __init__(self, chunk_size: int) -> None:
        self.chunk_size = chunk_size
        self._matrices: dict[str, np.ndarray] = {}
        self._chunks: dict[str, list[str]] = {}

    def prepare(self, doc: str, doc_id: str) -> float:
        if doc_id in self._matrices:
            return 0.0
        pieces = chunk(doc, chunk_size_tokens=self.chunk_size)
        if not pieces:
            self._chunks[doc_id] = []
            self._matrices[doc_id] = np.zeros((0, 1536), dtype=np.float32)
            return 0.0
        matrix, cost = embed_texts(pieces)
        self._chunks[doc_id] = pieces
        self._matrices[doc_id] = _normalize_rows(matrix)
        return cost

    def retrieve(self, query: str, k: int, doc_id: str) -> RetrieveResult:
        t0 = time.monotonic()
        if doc_id not in self._matrices:
            raise KeyError(f"prepare() not called for doc_id={doc_id!r}")
        chunks_for_doc = self._chunks[doc_id]
        if not chunks_for_doc:
            return RetrieveResult(chunks=[], cost_usd=0.0, latency_ms=0)
        q_vec, q_cost = embed_query(query)
        q_norm = q_vec / max(float(np.linalg.norm(q_vec)), 1e-12)
        scores = self._matrices[doc_id] @ q_norm
        order = np.argsort(-scores)[:k]
        picked = [chunks_for_doc[i] for i in order]
        latency_ms = int((time.monotonic() - t0) * 1000)
        return RetrieveResult(chunks=picked, cost_usd=q_cost, latency_ms=latency_ms)
