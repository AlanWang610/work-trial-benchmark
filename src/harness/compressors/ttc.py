"""The Token Company extractive compression — process-global 10 RPM bucket.

TTC has a hard 10-requests-per-minute cap (user-confirmed). Concurrency
beyond 1 only burns retries against the cap, so we serialize through a
single token bucket. Every successful response is cached on
sha256(input || model || aggressiveness) so a re-run after a crash
costs zero TTC quota.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from collections import deque

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from harness.chunker import count_tokens
from harness.compressors.base import CompressResult
from harness.settings import settings

TTC_URL = "https://api.thetokencompany.com/v1/compress"
TTC_PRICE_PER_M_REMOVED = 0.05


class _TokenBucket:
    """Sliding-window rate limiter shared across all worker threads.

    Allows up to `rate` calls per 60s. acquire() blocks until a slot
    frees. Thread-safe.
    """

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


_BUCKET = _TokenBucket(settings.TTC_RPM)


# ----- on-disk result cache ------------------------------------------------

_cache_lock = threading.Lock()
_cache_conn: sqlite3.Connection | None = None


def _get_cache() -> sqlite3.Connection:
    global _cache_conn
    if _cache_conn is None:
        settings.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        c = sqlite3.connect(str(settings.ttc_db), check_same_thread=False, isolation_level=None)
        c.execute("PRAGMA journal_mode=WAL")
        c.execute(
            "CREATE TABLE IF NOT EXISTS ttc("
            "key TEXT PRIMARY KEY, "
            "output TEXT NOT NULL, "
            "output_tokens INTEGER NOT NULL, "
            "input_tokens INTEGER NOT NULL"
            ")"
        )
        _cache_conn = c
    return _cache_conn


def _cache_key(text: str, model: str, aggr: float) -> str:
    payload = f"{model}::{aggr:.4f}::{text}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cache_get(key: str) -> tuple[str, int, int] | None:
    cur = _get_cache().execute(
        "SELECT output, output_tokens, input_tokens FROM ttc WHERE key = ?", (key,)
    )
    row = cur.fetchone()
    return row if row else None


def _cache_put(key: str, output: str, output_tokens: int, input_tokens: int) -> None:
    with _cache_lock:
        _get_cache().execute(
            "INSERT OR REPLACE INTO ttc(key, output, output_tokens, input_tokens) "
            "VALUES (?, ?, ?, ?)",
            (key, output, output_tokens, input_tokens),
        )


# ----- HTTP call -----------------------------------------------------------


class _TTCError(Exception):
    pass


class _TTCRetryableError(_TTCError):
    pass


@retry(
    retry=retry_if_exception_type(_TTCRetryableError),
    wait=wait_exponential(multiplier=6, min=6, max=60),
    stop=stop_after_attempt(6),
    reraise=True,
)
def _post_compress(text: str, model: str, aggressiveness: float) -> dict:
    _BUCKET.acquire()
    try:
        resp = requests.post(
            TTC_URL,
            headers={
                "Authorization": f"Bearer {settings.TTC_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "input": text,
                "model": model,
                "compression_settings": {"aggressiveness": aggressiveness},
            },
            timeout=settings.HTTP_TIMEOUT_S,
        )
    except requests.RequestException as exc:
        raise _TTCRetryableError(f"network error: {exc}") from exc
    if resp.status_code == 429:
        _BUCKET.penalize()
        raise _TTCRetryableError(f"429 from TTC: {resp.text[:200]}")
    if resp.status_code >= 500:
        raise _TTCRetryableError(f"{resp.status_code} from TTC: {resp.text[:200]}")
    if resp.status_code >= 400:
        raise _TTCError(f"{resp.status_code} from TTC: {resp.text[:500]}")
    try:
        return resp.json()
    except json.JSONDecodeError as exc:
        raise _TTCError(f"non-JSON response: {resp.text[:200]}") from exc


# ----- public API ----------------------------------------------------------


class TTCCompressor:
    name = "ttc"

    def __init__(self, aggressiveness: float, model: str | None = None) -> None:
        self.aggressiveness = float(aggressiveness)
        self.model = model or settings.TTC_MODEL

    def compress(self, text: str, query: str | None = None) -> CompressResult:
        # query is intentionally ignored — TTC compression is query-agnostic.
        del query
        if not text:
            return CompressResult(
                output="",
                input_tokens=0,
                output_tokens=0,
                cost_usd=0.0,
                latency_ms=0,
                compressor=f"{self.name}:{self.model}:{self.aggressiveness:.2f}",
            )
        key = _cache_key(text, self.model, self.aggressiveness)
        cached = _cache_get(key)
        t0 = time.monotonic()
        if cached is not None:
            output, output_tokens, input_tokens = cached
            return CompressResult(
                output=output,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=0.0,
                latency_ms=int((time.monotonic() - t0) * 1000),
                compressor=f"{self.name}:{self.model}:{self.aggressiveness:.2f}",
            )

        resp = _post_compress(text, self.model, self.aggressiveness)
        output = resp.get("output", "")
        in_tokens = int(resp.get("original_input_tokens") or count_tokens(text))
        out_tokens = int(resp.get("output_tokens") or count_tokens(output))
        removed = max(in_tokens - out_tokens, 0)
        cost_usd = removed * TTC_PRICE_PER_M_REMOVED / 1_000_000
        _cache_put(key, output, out_tokens, in_tokens)
        return CompressResult(
            output=output,
            input_tokens=in_tokens,
            output_tokens=out_tokens,
            cost_usd=cost_usd,
            latency_ms=int((time.monotonic() - t0) * 1000),
            compressor=f"{self.name}:{self.model}:{self.aggressiveness:.2f}",
        )
