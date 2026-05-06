from __future__ import annotations

from harness.chunker import count_tokens
from harness.compressors.base import CompressResult


class IdentityCompressor:
    name = "identity"

    def compress(self, text: str, query: str | None = None) -> CompressResult:
        n = count_tokens(text)
        return CompressResult(
            output=text,
            input_tokens=n,
            output_tokens=n,
            cost_usd=0.0,
            latency_ms=0,
            compressor=self.name,
        )
