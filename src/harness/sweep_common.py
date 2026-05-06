"""Shared sweep types — `Config` dataclass used by every approach.

Per-approach grids live in `harness.sweep_topk_ttc` and
`harness.sweep_nebula`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True, slots=True)
class Config:
    approach: str  # "ttc" | "identity" | "nebula"
    chunk_size: int | None  # None for nebula (Nebula auto-chunks internally)
    k: int | None  # None for nebula (Nebula chooses how many sources to return)
    aggressiveness: float | None  # None for identity / nebula
    nebula_effort: str | None = None  # "low" | "medium" | "high" for nebula, else None
    config_hash: str = field(default="", compare=False)

    @classmethod
    def make(
        cls,
        approach: str,
        chunk_size: int | None,
        k: int | None,
        aggressiveness: float | None,
        *,
        nebula_effort: str | None = None,
    ) -> Config:
        payload = json.dumps(
            {
                "approach": approach,
                "chunk_size": chunk_size,
                "k": k,
                "aggressiveness": aggressiveness,
                "nebula_effort": nebula_effort,
            },
            sort_keys=True,
        )
        h = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
        return cls(
            approach=approach,
            chunk_size=chunk_size,
            k=k,
            aggressiveness=aggressiveness,
            nebula_effort=nebula_effort,
            config_hash=h,
        )

    def to_dict(self) -> dict:
        return asdict(self)
