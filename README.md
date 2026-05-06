# work-trial-benchmark

Local MemoryAgentBench harness covering three retrieval approaches over a
shared subset, eval, reader, and runner. Spec: `BENCHMARK_PLAN.md`.

- **Approach 1** — top-k vector retrieval + Token Company extractive compression (`run_topk_ttc`)
- **Approach 2** — Nebula memory layer, effort sweep, no top-k cap (`run_nebula`)
- **Approach 3** — top-k vector retrieval, no compression — isolates the TTC contribution (`run_topk`)

## Setup

```
uv sync
cp .env.example .env  # then fill in OPENAI_API_KEY, TTC_API_KEY, NEBULA_API_KEY
```

## Run

```
# 1. one-time dataset pull (~76 MB) and shared subset build
python -m scripts.download_data
python -m scripts.build_subset

# 2. dry-runs (no API calls; prints config grid + cost + wall-clock)
python -m scripts.run_topk_ttc --smoke
python -m scripts.run_topk     --smoke
python -m scripts.run_nebula   --smoke

# 3. real runs (gated by --execute)

# Approach 1 — top-k + TTC (10 RPM cap dominates wall-clock)
python -m scripts.run_topk_ttc --smoke --execute   # 1 cfg × 7 rec   (~1 min,  ~$0.02)
python -m scripts.run_topk_ttc --mini  --execute   # 6 cfg × 21 rec  (~13 min, ~$0.30)
python -m scripts.run_topk_ttc --full  --execute   # 27 cfg × 76 rec (~3.4 h,  ~$5)

# Approach 2 — Nebula (30 RPM default; effort ∈ {low, medium, high})
python -m scripts.run_nebula   --smoke --execute   # 1 cfg × 7 rec   (~1 min,  ~$0.02)
python -m scripts.run_nebula   --mini  --execute   # 1 cfg × 21 rec  (~1 min,  ~$0.05)
python -m scripts.run_nebula   --full  --execute   # 3 cfg × 76 rec  (~8 min,  ~$0.50)

# Approach 3 — top-k only, no compression (OpenAI tier 4, no rate cap)
python -m scripts.run_topk     --smoke --execute   # 1 cfg × 7 rec   (~10 s,   ~$0.03)
python -m scripts.run_topk     --mini  --execute   # 3 cfg × 21 rec  (~30 s,   ~$0.13)
python -m scripts.run_topk     --full  --execute   # 9 cfg × 76 rec  (~3 min,  ~$1.40)

# 4. finalize → outputs/runs.parquet (merges all three approaches)
python -m scripts.finalize_runs
```

All three scripts append to the same `outputs/runs.jsonl`, keyed on
`(approach, config_hash, record_id)`, so re-runs and crash-resumes skip
already-done pairs and a single `finalize_runs` call covers everything.

## Sweep grids

| Approach | Axes | Full grid |
|---|---|---|
| 1 — top-k + TTC | `chunk_size{256,512,1024} × k{5,10,20} × aggressiveness{0.05,0.3,0.5}` | 27 configs |
| 2 — Nebula     | `effort{low, medium, high}` (no top-k — Nebula chooses what to return) | 3 configs |
| 3 — top-k only | `chunk_size{256,512,1024} × k{5,10,20}` | 9 configs |

Approach 1 pins TTC to `bear-1.2`. Approach 1 add-on: pass
`--include-no-compress` to splice in 3 identity-compressor B2 baselines
(chunk=512 × k∈{5,10,20}). See `BENCHMARK_PLAN.md` §4.4.

## Layout

```
src/harness/        core: settings, tasks, data, eval, reader, embedder,
                    chunker, pipeline,
                    retrievers/{topk_ttc, nebula},
                    compressors/{identity, ttc},
                    runner_common + runner_{topk_ttc, nebula, topk},
                    sweep_common  + sweep_{topk_ttc, nebula, topk}
scripts/            download_data, build_subset, finalize_runs,
                    run_topk_ttc, run_nebula, run_topk
data/               (gitignored) HF cache + subset_v1.jsonl
cache/              (gitignored) embeddings.sqlite + ttc.sqlite + nebula.sqlite
outputs/            (gitignored) runs.jsonl → runs.parquet
```

## Rate limits & throughput

- **TTC: 10 RPM hard cap** (Approach 1) → process-global token bucket;
  concurrency beyond 1 only burns retries. Wall-clock = `ceil(ttc_calls / 10)` min.
  Reader/judge dispatched on a small thread pool so they overlap each TTC wait.
- **Nebula: 30 RPM default** (Approach 2) → same sliding-window bucket pattern;
  override with `--nebula-rpm`. Phase 1 ingests each unique doc once (sync poll
  on both `ingestion_status` and `extraction_status`); Phase 2 search results
  cached in `cache/nebula.sqlite`, so grid re-runs cost zero quota.
- **OpenAI tier 4** → reader/judge/embed at concurrency 32, never the
  bottleneck. Approach 3 runs the reader/judge thread pool at 16 workers
  by default since there is no compression step in front of it.
