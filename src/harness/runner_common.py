"""Shared runner helpers used by both Approach 1 and Approach 2.

These are private to runner modules but importable across them. The two
runners (`harness.runner_topk_ttc` and `harness.runner_nebula`) share:

- `_run_key` / `_read_done_keys` — resume-from-checkpoint logic against
  the append-only `outputs/runs.jsonl`.
- `_make_compressor` — per-approach compressor selection.
- `_run_pair` / `_serialize_row` — single (config, record) execution and
  the JSONL row schema.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from harness.compressors.identity import IdentityCompressor
from harness.compressors.ttc import TTCCompressor
from harness.pipeline import run_one
from harness.sweep_common import Config


def _run_key(approach: str, config_hash: str, record_id: str) -> str:
    return f"{approach}|{config_hash}|{record_id}"


def _read_done_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            done.add(
                _run_key(
                    obj.get("approach", ""), obj.get("config_hash", ""), obj.get("record_id", "")
                )
            )
    return done


def _make_compressor(cfg: Config):
    if cfg.approach in ("identity", "nebula"):
        return IdentityCompressor()
    return TTCCompressor(aggressiveness=cfg.aggressiveness or 0.0)


def _run_pair(*, cfg: Config, rec: dict, retriever, tokens_in_source: int):
    compressor = _make_compressor(cfg)
    return run_one(
        rec,
        retriever=retriever,
        compressor=compressor,
        k=cfg.k if cfg.k is not None else 0,
        tokens_in_source=tokens_in_source,
    )


def _serialize_row(cfg: Config, rec: dict, result, tokens_in_source: dict[str, int]) -> str:
    payload = {
        "ts": datetime.now(UTC).isoformat(),
        "approach": cfg.approach,
        "config_hash": cfg.config_hash,
        "chunk_size": cfg.chunk_size,
        "k": cfg.k,
        "aggressiveness": cfg.aggressiveness,
        "nebula_effort": cfg.nebula_effort,
        "record_id": rec["record_id"],
        "row_id": rec["row_id"],
        "qa_idx": rec["qa_idx"],
        "task": rec["task"],
        "sub_dataset": rec.get("sub_dataset"),
        "question_type": rec.get("question_type"),
        "tokens_in_source": result.tokens_in_source,
        "tokens_to_reader": result.tokens_to_reader,
        "compression_ratio": result.compression_ratio,
        "answer": result.answer,
        "gold_answer": rec.get("answers"),
        "score": result.score,
        "judge_label": result.judge_label,
        "latency_ms_retrieve": result.latency_ms_retrieve,
        "latency_ms_compress": result.latency_ms_compress,
        "latency_ms_reader": result.latency_ms_reader,
        "latency_ms_judge": result.latency_ms_judge,
        "cost_usd_embed": result.cost_usd_embed,
        "cost_usd_compress": result.cost_usd_compress,
        "cost_usd_reader": result.cost_usd_reader,
        "cost_usd_judge": result.cost_usd_judge,
        "cost_usd_total": result.cost_usd_total,
        "error": result.error,
        "reader_model": result.reader_model,
        "reader_messages": result.reader_messages,
    }
    return json.dumps(payload, ensure_ascii=False) + "\n"
