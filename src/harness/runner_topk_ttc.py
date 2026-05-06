"""Approach 1 runner — top-k vector retrieval + TTC extractive compression.

Phase 1 (front-loaded): for each unique (row_id, chunk_size), call
`retriever.prepare()` so chunks + embeddings are computed once per
chunk_size. Cheap at OpenAI tier 4.

Phase 2 (TTC-throttled): for each (config, record) pair: retrieve →
compress (passes through the 10 RPM bucket) → reader.chat → judge →
append to runs.jsonl. Reader/judge calls are dispatched on a small
thread pool so they overlap each TTC 6 s wait.
"""

from __future__ import annotations

import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

from harness.chunker import count_tokens
from harness.retrievers.topk_ttc import TopKRetriever
from harness.runner_common import (
    _read_done_keys,
    _run_key,
    _run_pair,
    _serialize_row,
)
from harness.settings import settings
from harness.sweep_common import Config


def estimate(configs: list[Config], records: list[dict]) -> dict:
    n_records = len(records)
    n_ttc = sum(1 for c in configs if c.approach == "ttc") * n_records
    n_openai = (n_ttc + sum(1 for c in configs if c.approach == "identity") * n_records) * 2
    wall_clock_min = (n_ttc + settings.TTC_RPM - 1) // settings.TTC_RPM if settings.TTC_RPM else 0
    cost_reader_judge_usd = (
        n_ttc + sum(1 for c in configs if c.approach == "identity") * n_records
    ) * 0.00200
    return {
        "configs": len(configs),
        "records": n_records,
        "ttc_calls": n_ttc,
        "openai_calls": n_openai,
        "wall_clock_minutes_at_10rpm": int(wall_clock_min),
        "cost_reader_judge_usd": round(cost_reader_judge_usd, 2),
    }


def run(
    configs: list[Config],
    records: list[dict],
    *,
    threadpool_size: int = 4,
    runs_path: Path | None = None,
    phase2_label: str | None = None,
) -> None:
    runs_path = runs_path or settings.runs_jsonl
    runs_path.parent.mkdir(parents=True, exist_ok=True)
    done_keys = _read_done_keys(runs_path)

    chunk_sizes = sorted({cfg.chunk_size for cfg in configs})
    retrievers: dict[int, TopKRetriever] = {cs: TopKRetriever(cs) for cs in chunk_sizes}
    seen_doc: dict[tuple[int, str], None] = {}
    tokens_in_source: dict[str, int] = {}

    print(f"[Phase 1] preparing retrievers across {len(chunk_sizes)} chunk_size variants...")
    for rec in tqdm(records, desc="prepare", unit="rec"):
        row_id = rec["row_id"]
        if row_id not in tokens_in_source:
            tokens_in_source[row_id] = count_tokens(rec.get("context", ""))
        for cs in chunk_sizes:
            key = (cs, row_id)
            if key in seen_doc:
                continue
            retrievers[cs].prepare(rec.get("context", ""), doc_id=row_id)
            seen_doc[key] = None

    pairs = [(cfg, rec) for cfg in configs for rec in records]
    label = phase2_label or f"TTC ~{settings.TTC_RPM} RPM"
    print(
        f"[Phase 2] running {len(pairs)} (config × record) pairs "
        f"({len(done_keys)} cached) at {label}..."
    )

    pool = ThreadPoolExecutor(max_workers=threadpool_size)
    pending: list[tuple[Future, Config, dict]] = []
    bar = tqdm(total=len(pairs), desc="run", unit="pair")
    written = 0
    skipped = 0

    try:
        with runs_path.open("a", encoding="utf-8") as out:

            def drain_done(block: bool = False) -> None:
                nonlocal written
                if block:
                    for fut, cfg, rec in pending:
                        result = fut.result()
                        out.write(_serialize_row(cfg, rec, result, tokens_in_source))
                        out.flush()
                        written += 1
                        bar.update(1)
                    pending.clear()
                else:
                    still: list[tuple[Future, Config, dict]] = []
                    for fut, cfg, rec in pending:
                        if fut.done():
                            result = fut.result()
                            out.write(_serialize_row(cfg, rec, result, tokens_in_source))
                            out.flush()
                            written += 1
                            bar.update(1)
                        else:
                            still.append((fut, cfg, rec))
                    pending[:] = still

            for cfg, rec in pairs:
                key = _run_key(cfg.approach, cfg.config_hash, rec["record_id"])
                if key in done_keys:
                    skipped += 1
                    bar.update(1)
                    continue
                while len(pending) >= threadpool_size:
                    drain_done(block=False)
                    if len(pending) >= threadpool_size:
                        time.sleep(0.05)
                fut = pool.submit(
                    _run_pair,
                    cfg=cfg,
                    rec=rec,
                    retriever=retrievers[cfg.chunk_size],
                    tokens_in_source=tokens_in_source.get(rec["row_id"], 0),
                )
                pending.append((fut, cfg, rec))
            drain_done(block=True)
    finally:
        bar.close()
        pool.shutdown(wait=True)

    print(f"[Done] wrote {written} new rows, skipped {skipped} cached rows -> {runs_path}")
