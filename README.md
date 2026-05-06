# work-trial-benchmark

Local MemoryAgentBench harness — Approach 1 (top-k vector retrieval +
Token Company extractive compression). Spec: `BENCHMARK_PLAN.md`.

## Setup

```
uv sync
cp .env.example .env  # then fill in OPENAI_API_KEY and TTC_API_KEY
```

## Run

```
# 1. one-time dataset pull (~76 MB) and shared subset build
python -m scripts.download_data
python -m scripts.build_subset

# 2. dry-runs (no API calls; prints config grid + cost + wall-clock)
python -m scripts.run_topk_ttc --smoke
python -m scripts.run_topk_ttc --full

# 3. real runs (gated by --execute; TTC at 10 RPM)
python -m scripts.run_topk_ttc --smoke --execute   # ~1 min, ~$0.02
python -m scripts.run_topk_ttc --mini  --execute   # ~13 min, ~$0.30
python -m scripts.run_topk_ttc --full  --execute   # ~3.4 h, ~$5

# 4. finalize → outputs/runs.parquet
python -m scripts.finalize_runs
```

## Sweep grid (Approach 1)

27 configs = `chunk_size{256, 512, 1024} × k{5, 10, 20} × aggressiveness{0.05, 0.3, 0.5}`,
TTC model pinned to `bear-1.2`. See `BENCHMARK_PLAN.md` §4.4.

## Layout

```
src/harness/        core: settings, tasks, data, eval, reader, embedder,
                    chunker, retrievers/, compressors/, pipeline, sweep, runner
scripts/            download_data, build_subset, run_topk_ttc, finalize_runs
data/               (gitignored) HF cache + subset_v1.jsonl
cache/              (gitignored) embeddings.sqlite + ttc.sqlite
outputs/            (gitignored) runs.jsonl → runs.parquet
```

Approach 2 (Nebula) will land as
`src/harness/retrievers/nebula.py` + `scripts/run_nebula.py` and reuse
the same subset, eval, reader, and runner.

## Rate limits

- **TTC: 10 RPM hard cap** → process-global token bucket; concurrency
  beyond 1 only burns retries. Wall-clock = `ceil(ttc_calls / 10)` min.
- **OpenAI tier 4** → reader/judge/embed at concurrency 32, never the
  bottleneck.
