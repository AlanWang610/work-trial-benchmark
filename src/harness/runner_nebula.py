"""Approach 2 runner — Nebula retrieval + identity compressor + reader/judge.

Phase 1: ingest each unique row_id once (effort doesn't matter for ingest).
Phase 2: cross product of (config, record); one NebulaRetriever per effort
sharing the module-level row→collection map; reader+judge dispatched on a
thread pool so they overlap each Nebula search's wait.

Imports shared helpers (`_run_key`, `_read_done_keys`, `_run_pair`,
`_serialize_row`) from `harness.runner_common`. Drives writes to the same
`outputs/runs.jsonl` so DuckDB queries see both approaches.
"""

from __future__ import annotations

import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

from harness.chunker import count_tokens
from harness.retrievers.base import Retriever
from harness.retrievers.nebula import NebulaRetriever
from harness.runner_common import (
    _read_done_keys,
    _run_key,
    _run_pair,
    _serialize_row,
)
from harness.settings import settings
from harness.sweep_common import Config


def estimate_nebula(configs: list[Config], records: list[dict]) -> dict:
    n_records = len(records)
    n_search = len(configs) * n_records
    n_ingest = len({r["row_id"] for r in records})
    n_openai = n_search * 2
    rpm = settings.NEBULA_RPM
    wall_min = (n_search + rpm - 1) // rpm if rpm else 0
    return {
        "configs": len(configs),
        "records": n_records,
        "nebula_searches": n_search,
        "nebula_ingests": n_ingest,
        "openai_calls": n_openai,
        "wall_clock_minutes": int(wall_min),
        "cost_reader_judge_usd": round(n_search * 0.00200, 2),
    }


def run_nebula(
    configs: list[Config],
    records: list[dict],
    *,
    threadpool_size: int = 4,
    runs_path: Path | None = None,
    prepare_wait_s: float = 0.0,
) -> None:
    runs_path = runs_path or settings.runs_jsonl
    runs_path.parent.mkdir(parents=True, exist_ok=True)
    done_keys = _read_done_keys(runs_path)

    efforts = sorted({c.nebula_effort for c in configs if c.nebula_effort})
    if not efforts:
        raise ValueError("run_nebula: no configs with a nebula_effort set")

    ingestor = NebulaRetriever(effort=efforts[0])
    tokens_in_source: dict[str, int] = {}
    unique_rows = sorted({r["row_id"] for r in records})
    print(f"[Phase 1] ingesting {len(unique_rows)} unique docs into Nebula...")
    seen: set[str] = set()
    for rec in tqdm(records, desc="ingest", unit="rec"):
        row_id = rec["row_id"]
        if row_id not in tokens_in_source:
            tokens_in_source[row_id] = count_tokens(rec.get("context", ""))
        if row_id in seen:
            continue
        ingestor.prepare(rec.get("context", ""), doc_id=row_id)
        seen.add(row_id)

    if prepare_wait_s > 0:
        print(f"[Phase 1] sleeping {prepare_wait_s:.0f}s before search phase...")
        time.sleep(prepare_wait_s)

    retrievers: dict[str, Retriever] = {e: NebulaRetriever(effort=e) for e in efforts}
    pairs = [(cfg, rec) for cfg in configs for rec in records]
    print(
        f"[Phase 2] running {len(pairs)} (config x record) pairs "
        f"({len(done_keys)} cached) at Nebula ~{settings.NEBULA_RPM} RPM..."
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
                    retriever=retrievers[cfg.nebula_effort],
                    tokens_in_source=tokens_in_source.get(rec["row_id"], 0),
                )
                pending.append((fut, cfg, rec))
            drain_done(block=True)
    finally:
        bar.close()
        pool.shutdown(wait=True)

    print(f"[Done] wrote {written} new rows, skipped {skipped} cached rows -> {runs_path}")
