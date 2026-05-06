"""Nebula memory-layer retrieval — process-global rate bucket, cached ingest+search.

Nebula ingest is asynchronous: `memories.create` returns immediately with
`ingestion_status ∈ {pending, parsing, extracting, chunking, embedding,
augmenting, storing, success, failed}`. We poll `memories.retrieve(id)`
until the memory is searchable before returning from `prepare()`, so a
subsequent `retrieve()` does not race the indexer.

Both ingest and search outcomes are cached in `cache/nebula.sqlite`, so
crash-resume and grid re-runs cost zero Nebula quota.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from collections import deque

from nebula import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    ConflictError,
    Nebula,
    RateLimitError,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from harness.chunker import count_tokens
from harness.retrievers.base import RetrieveResult
from harness.settings import settings


class _TokenBucket:
    """Sliding-window rate limiter shared across all worker threads."""

    def __init__(self, rate_per_minute: float) -> None:
        self.rate = float(rate_per_minute)
        self._window: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        if self.rate <= 0:
            return
        while True:
            with self._lock:
                now = time.monotonic()
                cutoff = now - 60.0
                while self._window and self._window[0] < cutoff:
                    self._window.popleft()
                if len(self._window) < self.rate:
                    self._window.append(now)
                    return
                wait = 60.0 - (now - self._window[0]) + 0.05
            time.sleep(max(wait, 0.05))

    def penalize(self) -> None:
        """On 429, treat the bucket as drained for a full minute."""
        with self._lock:
            now = time.monotonic()
            self._window.clear()
            for _ in range(int(self.rate)):
                self._window.append(now)


_BUCKET = _TokenBucket(settings.NEBULA_RPM)
_row_to_collection: dict[str, str] = {}
_state_lock = threading.Lock()


_cache_lock = threading.Lock()
_cache_conn: sqlite3.Connection | None = None


def _get_cache() -> sqlite3.Connection:
    global _cache_conn
    if _cache_conn is None:
        settings.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        c = sqlite3.connect(str(settings.nebula_db), check_same_thread=False, isolation_level=None)
        c.execute("PRAGMA journal_mode=WAL")
        c.execute(
            "CREATE TABLE IF NOT EXISTS ingest("
            "key TEXT PRIMARY KEY, "
            "collection_id TEXT NOT NULL, "
            "memory_id TEXT NOT NULL"
            ")"
        )
        c.execute(
            "CREATE TABLE IF NOT EXISTS search("
            "key TEXT PRIMARY KEY, "
            "chunks TEXT NOT NULL"
            ")"
        )
        _cache_conn = c
    return _cache_conn


def _ingest_cache_get(key: str) -> tuple[str, str] | None:
    cur = _get_cache().execute(
        "SELECT collection_id, memory_id FROM ingest WHERE key = ?", (key,)
    )
    row = cur.fetchone()
    return row if row else None


def _ingest_cache_put(key: str, collection_id: str, memory_id: str) -> None:
    with _cache_lock:
        _get_cache().execute(
            "INSERT OR REPLACE INTO ingest(key, collection_id, memory_id) VALUES (?, ?, ?)",
            (key, collection_id, memory_id),
        )


def _search_cache_get(key: str) -> list[str] | None:
    cur = _get_cache().execute("SELECT chunks FROM search WHERE key = ?", (key,))
    row = cur.fetchone()
    if row is None:
        return None
    return json.loads(row[0])


def _search_cache_put(key: str, chunks: list[str]) -> None:
    with _cache_lock:
        _get_cache().execute(
            "INSERT OR REPLACE INTO search(key, chunks) VALUES (?, ?)",
            (key, json.dumps(chunks, ensure_ascii=False)),
        )


def _ingest_key(text: str, doc_id: str) -> str:
    return hashlib.sha256(f"{doc_id}::{text}".encode()).hexdigest()


def _search_key(collection_id: str, query: str, effort: str) -> str:
    return hashlib.sha256(f"{collection_id}::{effort}::{query}".encode()).hexdigest()


_RETRYABLE = (RateLimitError, APIConnectionError, APITimeoutError)


def _on_retry_sleep(retry_state) -> None:
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    if isinstance(exc, RateLimitError):
        _BUCKET.penalize()


@retry(
    retry=retry_if_exception_type(_RETRYABLE),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    stop=stop_after_attempt(6),
    reraise=True,
    before_sleep=_on_retry_sleep,
)
def _client_call(fn, **kwargs):
    _BUCKET.acquire()
    return fn(**kwargs)


def _src_text(s: object) -> str:
    if isinstance(s, dict):
        return s.get("text") or ""
    return getattr(s, "text", "") or ""


_TERMINAL_OK = {"success"}
_TERMINAL_FAIL = {"failed"}

# Reader is gpt-4o-mini @ 128K tokens. Leave 28K for question + answer + system prompt.
DEFAULT_SOURCE_TOKEN_BUDGET = 100_000


class NebulaRetriever:
    name = "nebula"

    def __init__(
        self,
        effort: str = "medium",
        *,
        poll_interval_s: float = 2.0,
        poll_max_s: float = 600.0,
        source_token_budget: int = DEFAULT_SOURCE_TOKEN_BUDGET,
    ) -> None:
        self.effort = effort
        self.poll_interval_s = poll_interval_s
        self.poll_max_s = poll_max_s
        self.source_token_budget = source_token_budget
        self._client: Nebula | None = None

    @property
    def client(self) -> Nebula:
        if self._client is None:
            self._client = Nebula(
                api_key=settings.NEBULA_API_KEY,
                base_url=settings.NEBULA_BASE_URL,
                timeout=float(settings.HTTP_TIMEOUT_S),
            )
        return self._client

    def prepare(self, doc: str, doc_id: str) -> float:
        with _state_lock:
            if doc_id in _row_to_collection:
                return 0.0

        ikey = _ingest_key(doc, doc_id)
        cached = _ingest_cache_get(ikey)
        if cached is not None:
            cid, mid = cached
            # Cache says we already uploaded; the memory may still be in
            # extract=processing if a previous run wrote the cache before we
            # learned to wait on extraction too. Re-poll cheaply.
            self._wait_for_ingest(cid, mid)
            with _state_lock:
                _row_to_collection[doc_id] = cid
            return 0.0

        coll_name = f"row_{doc_id[:12]}_{ikey[:8]}"
        try:
            coll_resp = _client_call(self.client.collections.create, name=coll_name)
            cid = coll_resp.results.id
        except ConflictError:
            # Collection ID is name-derived; a prior run created it but never
            # cached the id. Fetch by name and continue.
            existing = _client_call(
                self.client.collections.retrieve_by_name, collection_name=coll_name
            )
            cid = existing.results.id

        try:
            mem_resp = _client_call(
                self.client.memories.create,
                collection_id=cid,
                raw_text=doc,
                engram_type="document",
            )
            mem_id = (
                getattr(mem_resp.results, "memory_id", None)
                or getattr(mem_resp.results, "id", None)
            )
        except ConflictError:
            # Memory id is also content-derived; reuse the existing one.
            lr = _client_call(self.client.memories.list, collection_ids=[cid])
            if not lr.results:
                raise
            mem_id = lr.results[0].id

        if not mem_id:
            raise RuntimeError(f"Nebula memories.create returned no id for doc_id={doc_id!r}")

        self._wait_for_ingest(cid, mem_id)

        _ingest_cache_put(ikey, cid, mem_id)
        with _state_lock:
            _row_to_collection[doc_id] = cid
        return 0.0

    def _wait_for_ingest(self, collection_id: str, memory_id: str) -> None:
        """Poll via memories.list until both ingestion AND extraction succeed.

        Nebula reaches `ingestion_status=success` when embeddings are stored,
        but search depends on the semantic-graph extraction
        (`extraction_status`), which finishes later. Waiting only on
        ingestion produces empty `sources` for queries that arrive before
        extraction completes.
        """
        t0 = time.monotonic()
        last_ing = last_ext = ""
        while True:
            try:
                lr = _client_call(self.client.memories.list, collection_ids=[collection_id])
                match = next((r for r in lr.results if r.id == memory_id), None)
                if match is not None:
                    last_ing = getattr(match, "ingestion_status", None) or ""
                    last_ext = getattr(match, "extraction_status", None) or ""
                    if last_ing in _TERMINAL_FAIL or last_ext in _TERMINAL_FAIL:
                        raise RuntimeError(
                            f"Nebula memory {memory_id} failed "
                            f"(ingest={last_ing}, extract={last_ext})"
                        )
                    if last_ing in _TERMINAL_OK and last_ext in _TERMINAL_OK:
                        return
            except APIStatusError as e:
                if getattr(e, "status_code", None) != 404:
                    raise
            if time.monotonic() - t0 > self.poll_max_s:
                raise TimeoutError(
                    f"Nebula memory {memory_id} not searchable after "
                    f"{self.poll_max_s}s (last ingest={last_ing!r}, extract={last_ext!r})"
                )
            time.sleep(self.poll_interval_s)

    def retrieve(self, query: str, k: int, doc_id: str) -> RetrieveResult:
        del k
        with _state_lock:
            cid = _row_to_collection.get(doc_id)
        if cid is None:
            raise KeyError(f"prepare() not called for doc_id={doc_id!r}")
        skey = _search_key(cid, query, self.effort)
        cached = _search_cache_get(skey)
        t0 = time.monotonic()
        if cached is not None:
            return RetrieveResult(
                chunks=cached,
                cost_usd=0.0,
                latency_ms=int((time.monotonic() - t0) * 1000),
            )
        resp = _client_call(
            self.client.memories.search,
            query=query,
            collection_ids=[cid],
            effort=self.effort,
        )
        sources = getattr(resp.results, "sources", None) or []
        raw_chunks = [t for t in (_src_text(s) for s in sources) if t]
        chunks = self._cap_to_budget(raw_chunks)
        _search_cache_put(skey, chunks)
        return RetrieveResult(
            chunks=chunks,
            cost_usd=0.0,
            latency_ms=int((time.monotonic() - t0) * 1000),
        )

    def _cap_to_budget(self, chunks: list[str]) -> list[str]:
        """Greedy budget cap — keep whole chunks until the next would exceed
        `source_token_budget`. If a single chunk is already larger than the
        budget, truncate that one chunk by token slice."""
        if self.source_token_budget <= 0 or not chunks:
            return chunks
        kept: list[str] = []
        used = 0
        for c in chunks:
            n = count_tokens(c)
            if used + n <= self.source_token_budget:
                kept.append(c)
                used += n
                continue
            remaining = self.source_token_budget - used
            if remaining <= 0:
                break
            # Truncate this chunk roughly proportionally on character length
            # (cheap; per-call OOM-safety, not exact token count).
            ratio = remaining / max(n, 1)
            cut = int(len(c) * ratio)
            if cut > 0:
                kept.append(c[:cut])
            break
        return kept
