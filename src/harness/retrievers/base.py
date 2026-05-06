from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(slots=True)
class RetrieveResult:
    chunks: list[str]
    cost_usd: float
    latency_ms: int


class Retriever(Protocol):
    name: str

    def prepare(self, doc: str, doc_id: str) -> float: ...
    def retrieve(self, query: str, k: int, doc_id: str) -> RetrieveResult: ...
