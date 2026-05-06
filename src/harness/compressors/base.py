from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(slots=True)
class CompressResult:
    output: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: int
    compressor: str


class Compressor(Protocol):
    name: str

    def compress(self, text: str, query: str | None = None) -> CompressResult: ...
