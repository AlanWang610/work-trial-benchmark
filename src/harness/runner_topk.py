"""Approach 3 runner — top-k retrieval, no compression (raw chunks → reader).

Delegates Phase 1/Phase 2 dispatch to `runner_topk_ttc.run` since the
retriever pipeline is identical; the per-pair compressor selection in
`runner_common._make_compressor` already maps `cfg.approach == "identity"`
to `IdentityCompressor`. The only Approach-3-specific bits are:

- `estimate()` drops TTC accounting and assumes OpenAI-tier-4 throughput.
- The Phase 2 banner reflects "no TTC quota" instead of TTC's 10 RPM cap.
"""

from __future__ import annotations

from pathlib import Path

from harness.runner_topk_ttc import run as _run_topk_ttc
from harness.sweep_common import Config


def estimate(
    configs: list[Config], records: list[dict], *, threadpool_size: int = 16
) -> dict:
    n_records = len(records)
    n_calls = len(configs) * n_records
    n_openai = n_calls * 2  # reader + (potential) judge
    # OpenAI tier 4: ~2 s per pair (reader+judge serialised) at threadpool_size workers.
    secs = n_calls * 2 / max(threadpool_size, 1)
    wall_clock_min = max(1, int((secs + 59) // 60))
    return {
        "configs": len(configs),
        "records": n_records,
        "openai_calls": n_openai,
        "wall_clock_minutes": wall_clock_min,
        "cost_reader_judge_usd": round(n_calls * 0.00200, 2),
    }


def run(
    configs: list[Config],
    records: list[dict],
    *,
    threadpool_size: int = 16,
    runs_path: Path | None = None,
) -> None:
    _run_topk_ttc(
        configs,
        records,
        threadpool_size=threadpool_size,
        runs_path=runs_path,
        phase2_label=f"threadpool={threadpool_size} (no compression, no TTC quota)",
    )
