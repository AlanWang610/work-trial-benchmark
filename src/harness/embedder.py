"""Cached batched embedding client for text-embedding-3-small."""

from __future__ import annotations

import hashlib
import sqlite3
import threading

import numpy as np
from openai import APIError, APITimeoutError, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from harness.reader import _get_client, cost_for
from harness.settings import settings

_EMBED_DIM = 1536
_BATCH_SIZE = 100
_lock = threading.Lock()
_conn: sqlite3.Connection | None = None


def _get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        settings.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        c = sqlite3.connect(
            str(settings.embeddings_db), check_same_thread=False, isolation_level=None
        )
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("CREATE TABLE IF NOT EXISTS embeddings (key TEXT PRIMARY KEY, vec BLOB NOT NULL)")
        _conn = c
    return _conn


def _key(text: str, model: str) -> str:
    return hashlib.sha256(f"{model}::{text}".encode()).hexdigest()


def _get(text: str, model: str) -> np.ndarray | None:
    cur = _get_conn().execute("SELECT vec FROM embeddings WHERE key = ?", (_key(text, model),))
    row = cur.fetchone()
    if row is None:
        return None
    return np.frombuffer(row[0], dtype=np.float32).copy()


def _put_many(items: list[tuple[str, np.ndarray]], model: str) -> None:
    rows = [(_key(t, model), v.astype(np.float32).tobytes()) for t, v in items]
    with _lock:
        _get_conn().executemany("INSERT OR REPLACE INTO embeddings(key, vec) VALUES (?, ?)", rows)


@retry(
    retry=retry_if_exception_type((RateLimitError, APITimeoutError, APIError)),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    stop=stop_after_attempt(5),
    reraise=True,
)
def _embed_batch(texts: list[str], model: str) -> tuple[list[np.ndarray], int]:
    resp = _get_client().embeddings.create(model=model, input=texts)
    vecs = [np.array(d.embedding, dtype=np.float32) for d in resp.data]
    tokens = resp.usage.total_tokens if resp.usage else 0
    return vecs, tokens


def embed_texts(texts: list[str], *, model: str | None = None) -> tuple[np.ndarray, float]:
    """Returns (matrix [n, dim], cost_usd). Misses are batched + cached."""
    model = model or settings.EMBED_MODEL
    if not texts:
        return np.zeros((0, _EMBED_DIM), dtype=np.float32), 0.0

    cached: list[np.ndarray | None] = [_get(t, model) for t in texts]
    miss_idx = [i for i, v in enumerate(cached) if v is None]
    miss_texts = [texts[i] for i in miss_idx]

    cost = 0.0
    if miss_texts:
        new_vecs: list[np.ndarray] = []
        for start in range(0, len(miss_texts), _BATCH_SIZE):
            batch = miss_texts[start : start + _BATCH_SIZE]
            vecs, tokens = _embed_batch(batch, model)
            new_vecs.extend(vecs)
            cost += cost_for(model, tokens, 0)
        _put_many(list(zip(miss_texts, new_vecs, strict=True)), model)
        for slot_i, vec in zip(miss_idx, new_vecs, strict=True):
            cached[slot_i] = vec

    matrix = np.stack(cached, axis=0)  # type: ignore[arg-type]
    return matrix, cost


def embed_query(text: str, *, model: str | None = None) -> tuple[np.ndarray, float]:
    matrix, cost = embed_texts([text], model=model)
    return matrix[0], cost
