# Local MemoryAgentBench harness — research & plan

Compiled 2026-05-05 from a four-agent research pass over the benchmark, both
candidate retrieval systems, and current OpenAI model pricing. Every claim
below is sourced (see §11). Where numbers or knobs are inferred rather than
confirmed in docs, the line is prefixed `ASSUMPTION:` and listed in §10.

---

## 0. TL;DR

- **Benchmark.** MemoryAgentBench (HF: `ai-hyz/MemoryAgentBench`, paper
  arXiv:2507.05257, ICLR 2026). 146 rows across four competency splits
  (Accurate Retrieval / Test-Time Learning / Long Range Understanding /
  Conflict Resolution). Each row is one massive context (273 K – 3.17 M chars)
  paired with 60–100 QA pairs. MIT-licensed.
- **Reader / judge / embedder.** Reader = `gpt-4o-mini` (matches the paper's
  default; pinned `gpt-4o-mini-2024-07-18`). Judge = `gpt-4o-2024-08-06`
  (matches the repo's hardcoded `gpt-4o`; verbatim prompt in §3.2). Embedder
  for Approach 1 = `text-embedding-3-small`.
- **Approach 1 — top-k + Token Company.** Real, self-serve, $0.05 / 1 M
  *removed* tokens. Single knob `aggressiveness ∈ [0, 1]` on models
  `bear-{1, 1.1, 1.2}`. Compression is **query-agnostic** — biggest gotcha
  vs. classic RAG-compression literature. Realized ratio is data-dependent
  (≈0–50 % reduction across the sweep). Sweep is 3-D: `k × chunk_size ×
  aggressiveness` ≈ 27 configs (reduced from 60 — see §4.4).
- **Approach 2 — Nebula.** Hosted SaaS; **pricing/quotas are not public**, so
  budget gating is the #1 access risk. Only three real knobs are exposed
  (`effort`, `semantic_weight`/`fulltext_weight`, `result_layers`) → ~64
  configs. Hierarchy depth, branching, summary length, summary-LLM, embedder
  are **all internal and non-configurable**. Recommend running Nebula at
  available knobs **plus** a parallel self-managed RAPTOR/TreeIndex baseline
  so the "hierarchical vector graph" axis has more than three dials.
- **Cost.** End-to-end sweep at 100 examples/config across 123 configs ≈ **$25
  reader + judge** with the recommended models; 300/config ≈ **$74**;
  three-pass plan (pilot 50 → main 100 → Pareto-only retest at 300) totals
  **~$65**. Token Company API spend is negligible at this scale; Nebula
  spend is unknown and must be confirmed at signup.
  *(2026-05-05 update: Approach 1 is now 27 configs run on the shared 76-record
  `data/subset_v1.jsonl`, not 100 examples — see §4.4 and the new §7.3 table.)*
- **Pareto.** Both systems can only be swept and measured — neither exposes a
  hard "compression ratio = X" knob. Plot accuracy (LLM-judge yes-rate)
  against `tokens_to_reader / tokens_in_source` and take the skyline.

---

## 1. Decisions to confirm before locking the harness

These five choices materially shape cost, runtime, and reproducibility.
Defaults are marked **bold**.

| # | Decision | Options | Default reasoning |
|---|---|---|---|
| 1 | Primary reader | **`gpt-4o-mini`** vs `gpt-5.1` (your prototype) vs `gpt-4.1-mini` | Match the paper's headline numbers; cheap; in the repo's `setup.sh`. |
| 2 | Approach 2 scope | **Nebula only** vs **Nebula + self-managed RAPTOR/TreeIndex** | Nebula gives 3 knobs; a RAPTOR baseline gives 4–6 and lets you ablate "hierarchy" cleanly. Recommend running both. |
| 3 | Subset depth | Pilot **n=50/task** → main **n=100/task** → final **n=300/task on Pareto survivors** | ±10 % CI at n=100 is enough to drop dominated configs; ±6 % at n=300 for the headline plot. |
| 4 | Tasks in scope | **Cheap recipe** (§2.4) vs full 146-row dataset | Full dataset includes 1.4 M-token rows and would blow past the budget. |
| 5 | Judge | **`gpt-4o-2024-08-06`** (repo default) vs `gpt-5.1` (stronger) | Paper-comparability; pin a date-stamped version for reproducibility. |

Continue past this section assuming the bold defaults.

---

## 2. The benchmark

### 2.1 Dataset structure

Single config, Parquet, 146 rows, 76.6 MB, MIT.

| Split (= competency) | Rows |
|---|---|
| `Accurate_Retrieval` | 22 |
| `Test_Time_Learning` | 6 |
| `Long_Range_Understanding` | 110 |
| `Conflict_Resolution` | 8 |

Per-row schema (every split):

```python
{
  "context":   str,            # one giant doc / transcript, 273k–3.17M chars
  "questions": list[str],      # 60–100 questions per row
  "answers":   list[list[str]],# acceptable answers per question
  "metadata":  dict            # qa_pair_ids, demo, haystack_sessions,
                               # keypoints, previous_events
}
```

Note the **inject-once-query-many** layout: a single context is amortised over
many QA pairs. Total QA pairs across the dataset are O(10 k); per-task QA
counts after the 2025-09-29 cleanup must be counted at load time (not
published in the paper).

### 2.2 Tasks (sub_dataset → YAML)

YAMLs live in `configs/data_conf/{competency}/`.

**Accurate_Retrieval**
- `ruler_qa` — needle-in-haystack QA over packed Wikipedia (SH ≈197 k tokens, MH ≈421 k).
- `longmemeval_s` / `longmemeval_s_star` — multi-session conversation memory; sub-types: single-session-user/assistant, multi-session, temporal-reasoning, knowledge-update, preference, abstention (≈355 k tokens, 300 QAs).
- `eventqa` (`Eventqa_64k`, `Eventqa_128k`, `Eventqa_full`) — multiple-choice "predict the next event in a novel" (≈534 k tokens, 500 QAs).

**Test_Time_Learning** (one very long ICL prompt)
- `icl_trec_coarse` (6 600 shots), `icl_trec_fine`, `icl_banking77`, `icl_clinic150`, `icl_nlu` — many-shot intent classification (103 k–1.44 M tokens).
- `recsys_redial` — recommend 20 movies from long dialogue history; Recall@1/5/10.

**Long_Range_Understanding**
- `infbench_sum_eng_shots2` — book-length English summarization, F1 vs reference keypoints (≈172 k tokens, 100 docs).
- `detective_qa` — multi-choice reasoning over detective novels (≈124 k tokens, 10 samples in default config).

**Conflict_Resolution**
- `factconsolidation_sh_{6k,32k,64k,262k}` — single-hop conflicting-fact resolution; pick the newest fact by serial number.
- `factconsolidation_mh_{6k,32k,64k,262k}` — multi-hop variant. The published bottleneck (≤7 % accuracy for nearly all systems).

### 2.3 Evaluation methodology

Three styles, all in-repo:

**A. Substring / EM** (`utils/eval_other_utils.py`) — `ruler_qa`, `eventqa`,
`factconsolidation`, ICL, `recsys` (Recall@K). Output parser looks for the
prefix `Answer:`.

**B. LLM-judge for LongMemEval** (`llm_based_eval/longmem_qa_evaluate.py`) —
hardcoded `metric_model="gpt-4o"`, `temperature=0`, `max_tokens=10`, label =
`'yes' in response.lower()`. Six branches keyed on `question_type`. Default
branch verbatim:

> *"I will give you a question, a correct answer, and a response from a
> model. Please answer yes if the response contains the correct answer.
> Otherwise, answer no. If the response is equivalent to the correct answer
> or contains all the intermediate steps to get the correct answer, you
> should also answer yes. If the response only contains a subset of the
> information required by the answer, answer no."*

Other branches: `temporal-reasoning` ("do not penalize off-by-one errors for
the number of days"), `knowledge-update` (accept old + updated alongside),
`single-session-preference` (rubric-based), abstention ("Does the model
correctly identify the question as unanswerable?"). Use the repo's prompts
verbatim per branch — do not invent a new judge prompt.

**C. LLM-judge for summarization** (`llm_based_eval/summarization_evaluate.py`)
— `metric_model="gpt-4o-2024-05-13"`, `temperature=0.1`. Computes Fluency
(0/1) × Recall (vs keypoints) × Precision (vs expert summary) → F1. Two
rubric variants for legal vs. literary content.

### 2.4 Cheap recipe for local runs

There is no shipped dev/small split. Subset by setting `max_test_samples` in
each task YAML.

> **2026-05-05 update — single shared subset.** Both Approach 1 (top-k +
> TTC) and Approach 2 (Nebula) consume the **same** `data/subset_v1.jsonl`,
> built once by `scripts/build_subset.py` with `seed=42` and a stable
> record-id sort. Smaller `--smoke` (1/task) and `--mini` (3/task) modes
> are deterministic *prefixes* of the per-task ordering — every record run
> in `--smoke` is also run in `--mini` and `--full`, so cross-mode
> accuracy numbers are directly comparable rather than independent
> samples.

| Task | Suggested cap | Approx. context |
|---|---|---|
| `ruler_qa` (SH) | **12** QAs | 197 k tok |
| `longmemeval_s` | **18** QAs (3 per `question_type` × 6 types) | 355 k tok |
| `eventqa_64k` | **5** QAs | 64 k tok |
| `detective_qa` | **5** QAs | 124 k tok |
| `infbench_sum_eng_shots2` | **skip** for pilots (1.4 M tok per row) | 172 k tok |
| `icl_trec_coarse` | **12** QAs | 103 k tok |
| `factconsolidation_sh_6k` | **12** QAs | 6 k tok |
| `factconsolidation_mh_6k` | **12** QAs | 6 k tok |
| All `*_262k` and `eventqa_full` | **skip** for pilots | 262 k–1 M+ tok |

→ **76 reader calls per config-pass** (× 27 configs = 2 052 per `--full`
run). ∞Bench-Sum and the `*_262k` variants are the budget-killers; defer
them until the Pareto frontier has stabilised.

Stratify by `(competency, task)` — 13 strata if you keep all eight tasks
above. Sample with a fixed seed; cache as `subset_v1.jsonl` so every config
sees identical examples.

### 2.5 Headline numbers (paper Table 3, overall avg %)

| System | AR | TTL | LRU | CR(SF) | Overall |
|---|---|---|---|---|---|
| GPT-4o (long ctx) | 58.1 | 50.0 | 54.9 | 32.5 | 48.8 |
| GPT-4o-mini | 49.2 | 48.6 | 46.2 | 25.0 | 42.2 |
| GPT-4.1-mini (1 M ctx) | 71.8 | 46.2 | 49.1 | 20.5 | 46.9 |
| Claude-3.7-Sonnet | 59.7 | 53.9 | 62.2 | 22.5 | 49.6 |
| BM25 RAG | 60.5 | 44.5 | 35.6 | 25.5 | — |
| Mem0 | 32.6 | 21.2 | 20.7 | 10.0 | — |
| Cognee | 28.3 | 22.8 | 16.0 | 15.5 | — |
| Zep | 37.5 | 37.5 | 16.2 | 5.0 | — |
| MemGPT / Letta | 34.3 | 40.8 | 22.4 | 15.5 | — |

Two anchor facts: BM25 hits 100 % on NIAH-MQ vs 22.8 % for vanilla
GPT-4o-mini, and multi-hop Conflict Resolution collapses to ≤7 % for nearly
everything. Use these as sanity checks for your harness output.

### 2.6 Pitfalls

- Splits are unevenly sized in **rows** but each row carries different QA
  counts — balance sampling per **task**, not per row.
- Per-task `generation_max_length` is tight (e.g. 10 for FactConsolidation,
  40 for EventQA, 50 for LongMemEval, 1 200 for InfBench-Sum). Match them or
  hurt scores.
- `use_chat_template` is `false` for ICL (newline-stop mode), `true`
  elsewhere.
- The dataset has been edited post-publication: `qa_pair_ids` was renamed
  from `uuid` (Jul 22 2025), `ruler_niah` was removed (Aug 5 2025),
  high-cost samples pruned (Sep 29 2025). Cache keys must include the
  current schema.
- Long-context mode is implemented as **chunk-and-concatenate into one big
  user turn**, not as separate API turns. The "incremental injection"
  framing is not a multi-call agent loop.

### 2.7 Key files

`main.py` (orchestrator), `agent.py` (model wrapper), `conversation_creator.py`
(loading + chunking via `get_chunks()` / `get_query_and_answers()`),
`initialization.py`, `utils/templates.py` (all prompt templates),
`utils/eval_other_utils.py`, `llm_based_eval/{longmem_qa,summarization}_evaluate.py`,
`configs/data_conf/**/*.yaml`,
`configs/agent_conf/Long_Context_Agents/Long_context_agent_{gpt-4o-mini,gpt-4o,gpt-4.1-mini,o4-mini,claude-3-7-sonnet-20250219,gemini-2.0-flash}.yaml`.

The default agent YAML is one chat completion per QA with system + (context +
query) packed as a single user turn, `temperature=0.7`, `input_length_limit`
defaulted to the model's window:

```yaml
agent_name: Long_context_agent_gpt-4o-mini
model: gpt-4o-mini
temperature: 0.7
input_length_limit: 128000
buffer_length: 4000
output_dir: ./outputs/gpt-4o-mini
```

---

## 3. Reader, judge, and embedding models

### 3.1 Reader — `gpt-4o-mini` (paper match)

| Role | Model ID | Ctx | Cutoff | Input $/MTok | Output $/MTok |
|---|---|---|---|---|---|
| **Primary** (paper match) | `gpt-4o-mini-2024-07-18` | 128 k | Oct 2023 | $0.15 | $0.60 |
| Cheap full-context baseline | `gpt-4.1-nano` | 1 M | Jun 2024 | $0.10 | $0.40 |
| Optional capability anchor | `gpt-4.1-mini` | 1 M | Jun 2024 | $0.40 | $1.60 |

Rationale: the paper's headline numbers and the repo's `LLM_MODEL=gpt-4o-mini`
default both argue for `gpt-4o-mini` as primary, even though your existing
prototype (`compress.py`) uses `gpt-5.1`. Use `gpt-4.1-nano` only for the
"no compression / full context" baseline (the 1 M window is the point).
Adding `gpt-5.1` as a third capability tier is reasonable but multiplies
sweep cost — defer to a final "capability sensitivity" pass after the
frontier is set.

### 3.2 Judge — `gpt-4o-2024-08-06` (repo match)

The MemoryAgentBench code hardcodes `gpt-4o` (latest in the family); pin the
`2024-08-06` snapshot for stability ($2.50 / $10.00 per MTok). Use the
repo's branched prompts verbatim (§2.3 above) — do not roll your own.
Judge cost is roughly $0.0014 per call at ~500 in / 30 out.

### 3.3 Embedding — `text-embedding-3-small`

| Model | Dim | MTEB | $/MTok |
|---|---|---|---|
| `text-embedding-3-small` | 1 536 (truncatable to 256/512/1 024) | 62.26 | **$0.02** |
| `text-embedding-3-large` | 3 072 | ~64.6 | $0.13 |

`-small` is the default. Switching to `-large` costs 6.5× and rarely moves
retrieval@k materially on QA. Reserve `-large` for a 1-config ablation if
the small-model frontier is dominated. Nebula uses its own internal embedder
(not configurable per docs) — the embedding model only affects Approach 1
and the optional RAPTOR baseline.

---

## 4. Approach 1 — top-k + Token Company

### 4.1 API surface

You already have a working prototype at `compress.py`:

```python
requests.post(
  "https://api.thetokencompany.com/v1/compress",
  headers={"Authorization": f"Bearer {os.environ['TTC_API_KEY']}"},
  json={
    "input": context_text,
    "model": "bear-1.2",
    "compression_settings": {"aggressiveness": 0.1}
  }
).json()  # → {"output": "...", "output_tokens": int, "original_input_tokens": int}
```

Or use the official Python SDK: `pip install tokenc`, `TokenClient.compress_input(...)`
returns `{output, output_tokens, original_input_tokens, tokens_saved,
compression_ratio}`. Sync `requests` only — no async client. No batch endpoint.

### 4.2 Hyperparameters (the entire surface)

| Param | Range | Default | Notes |
|---|---|---|---|
| `model` | `bear-1` / `bear-1.1` / `bear-1.2` | `bear-1.2` recommended | No quality dial beyond version |
| `compression_settings.aggressiveness` | float 0.0 – 1.0 | 0.5 | Vendor guidance: 0.1 light, 0.5 balanced, 0.9 aggressive |
| `<ttc_safe>…</ttc_safe>` markup | n/a | none | Wraps regions (e.g. the question) to preserve them |

There is **no** `query` parameter — compression is **query-agnostic**. To
inject query-awareness, splice the question into the input wrapped in
`<ttc_safe>` and rely on local context preservation (undocumented behavior;
test before relying on it).

### 4.3 Realized compression behaviour

You cannot fix a target ratio. The vendor's own published curves
(FinanceBench): aggressiveness 0.05 → −1.5 %, 0.1 → −3.9 %, 0.3 → −10.4 %,
0.5 → −14.4 %, 0.7 → −20 %; latency study reports ~50 % at aggressiveness
0.9. **Plan to sweep and measure the realized ratio** rather than target one.

### 4.4 Sweep plan — 27 configs (reduced)

Pre-compression retrieval × compressor knob:

- `chunk_size` (tokens) ∈ `{256, 512, 1024}` — 3 values
- `k` ∈ `{5, 10, 20}` — 3 values
- `aggressiveness` ∈ `{0.05, 0.3, 0.5}` — 3 values spanning the
  documented FinanceBench / latency curve from light (≈−1.5 % removal)
  through balanced (≈−10.4 %) to heavier (≈−14.4 %) so the Pareto
  frontier is bracketed end-to-end

→ **3 × 3 × 3 = 27 configs**. Pin `model = bear-1.2`. Use
`text-embedding-3-small` at full 1 536 dims for retrieval. Concatenate
top-k chunks with newlines and a chunk separator, compress, then append
`Question: {q}\nAnswer:` after the compressed block.

> **2026-05-05 reduction.** The original grid was `k{1,3,5,10,20} ×
> aggr{0.05,0.1,0.2,0.3,0.5}` = 60 configs. Reduced to 27 in two steps
> to keep `--full` within ~3.4 h wall-clock at TTC's 10 RPM cap. The
> aggressiveness mainline `{0.05, 0.3, 0.5}` brackets the FinanceBench
> curve end-to-end; `aggr=0.9` and the `k∈{1,3}` extremes can be added
> back as a Pareto-survivor retest if the frontier extends past the
> surveyed compression range.

### 4.5 Caveats

- **Query-agnostic** — biggest mismatch with the LLMLingua / Selective
  Context literature. Some questions will fail because the relevant
  sentence got shortened.
- Closed model, no weights → not reproducible offline; results depend on a
  live service that may change between runs (no version pin finer than
  `bear-1.x`).
- No published head-to-head vs LLMLingua / LongLLMLingua / Selective Context
  / RECOMP / PISCO. You are producing the comparison.
- No documented batch endpoint; for ~300 examples × 60 configs = 18 000
  compression calls, run a small client-side threadpool (≤16 concurrent).
- Pricing is $0.05 / 1 M *removed* tokens. For the entire 60-config × 100-
  example sweep this is on the order of a dollar.

---

## 5. Approach 2 — Nebula

### 5.1 Access status

**Hosted SaaS**. Pricing tiers, free-tier credits, rate limits are not
publicly visible (the page is JS-rendered). Likely sales / dashboard gated.
**Confirm at signup before investing harness time** — this is the #1 risk.

If access is gated or quotas are too tight for the sweep, fall back to a
self-managed **RAPTOR / TreeIndex** implementation (LlamaIndex `TreeIndex`,
LangChain `ParentDocumentRetriever` + recursive summarisation). That gives
you full control over hierarchy depth, branching, summary length, and
top-k per level — i.e. a proper Pareto frontier instead of a 3-knob slice.

### 5.2 API surface

REST `https://api.trynebula.ai/v1/`; Python SDK `pip install nebula-sdk`
(sync + async, Py 3.10+); Node SDK; MCP server.

```python
nebula = Nebula()  # NEBULA_API_KEY env var
coll   = nebula.collections.create(name=f"bench-{ex_id}").results

for chunk in document_chunks:
    nebula.memories.create(
        collection_id=coll.id,
        raw_text=chunk,
        engram_type="document",       # or "conversation"
    )
# ingest is async (HTTP 202) — poll until built

res = nebula.memories.search(
    query=question,
    collection_ids=[coll.id],
    effort="high",                    # low / medium / high / auto
    search_settings={"semantic_weight": 0.8, "fulltext_weight": 0.2},
).results

# res has 4 typed layers:
#   res.semantic    — extracted facts, with activation_score (0–1)
#   res.procedural  — preferences/habits, with confidence
#   res.episodes    — temporal event clusters
#   res.sources     — original passages, with speaker/role
```

### 5.3 Memory model

Nebula auto-builds a vector graph at ingest. The *conceptual* hierarchy is
roughly `raw text → episodes → semantic / procedural facts`, but it is
opinionated and opaque — not a configurable raw → summary → meta-summary
tree. Hierarchy depth, branching, summary length, summary-LLM, embedder are
**all internal and not configurable**. You cannot point Nebula's
summariser at your own OpenAI key.

### 5.4 Hyperparameters (the entire surface)

| Param | Range | Default | Notes |
|---|---|---|---|
| `effort` | `low` / `medium` / `high` / `auto` | `auto` | Traversal hops: low = 2 narrow, medium = 2 wide, high = 3 widest |
| `search_settings.semantic_weight` | 0 – 1 | 0.8 | Semantic vs full-text balance (`fulltext_weight` = 1 − this) |
| `filters` | Mongo-style | none | Metadata filtering only |
| `result_layers` (client-side, what you feed reader) | subset of `{semantic, procedural, episodes, sources}` | implementation choice | Closest analog to "raw chunks vs summary" |
| Client-side `activation_score` threshold | 0 – 1 | n/a | Trim before concat |

**Not exposed**: `top_k`, `max_tokens_returned`, hierarchy depth,
chunk size, branching factor, summary length, embedding model, summary LLM.
You cannot fix output token count — trim client-side after retrieval.

### 5.5 Sweep plan — 64 configs

- `effort` ∈ `{low, medium, high, auto}` — 4 values
- `semantic_weight` ∈ `{1.0, 0.8, 0.5, 0.2}` (fulltext = 1 − this) — 4 values
- `result_layers` ∈ `{sources}, {semantic}, {semantic + episodes}, {all four}` — 4 buckets

→ **4 × 4 × 4 = 64 configs**. Optional secondary axis: client-side
`activation_score` ≥ τ for τ ∈ `{0, 0.3, 0.5, 0.7}` — sweep this as a
post-hoc grid only on Pareto survivors to avoid blowing up to 256 configs.

For tasks with conversational structure (LongMemEval, RecSys-Redial), set
`engram_type="conversation"` at ingest; otherwise `"document"`.

### 5.6 Caveats

- **Pricing not public** — confirm at signup before committing. May be
  sales-gated for the volume needed (~300 long-doc ingests).
- **Async ingest (HTTP 202)** — poll for graph-build completion before
  querying. No documented "ready" webhook.
- **No batch ingest endpoint** — concurrency comes from your own client.
- **No published benchmarks** vs Mem0 / Mem0g / MemGPT / Letta / GraphRAG /
  LightRAG. You are producing the first head-to-head.
- **Opinionated retrieval format** — typed layers, not raw chunks. The
  layer choice is itself a confound: `sources` ≈ classic RAG behaviour;
  `semantic + episodes` is what makes Nebula different.
- **Reset between examples**: create + delete a collection per benchmark
  row. Delete semantics aren't fully documented; verify in pilot.

### 5.7 Recommended fallback / parallel — RAPTOR-style baseline

Whether or not Nebula access works out, build a **self-managed hierarchical
baseline** so the "hierarchical vector graph" approach has more than three
knobs to sweep:

- `chunk_size` ∈ `{256, 512, 1024}`
- `tree_depth` ∈ `{1, 2, 3}` (1 = flat top-k; 3 = leaf → mid → root)
- `branching_factor` ∈ `{4, 8, 16}` (children per parent)
- `summary_target_tokens` ∈ `{128, 256, 512}` (per-level summary length)
- `top_k_per_level` ∈ `{2, 5, 10}`
- `summary_llm` ∈ `{gpt-4o-mini, gpt-4.1-nano}` (your control over the
  summarisation step — Nebula does not let you control this)

Use LlamaIndex `TreeIndex` / `RaptorPack` or a thin custom implementation
on top of `text-embedding-3-small`. Sample 60 configs from the full grid
via Latin-hypercube to keep parity with Approach 1. This is what the user's
question really wants in axis count — Nebula alone cannot deliver a 5-D
Pareto frontier.

---

## 6. Harness architecture

### 6.1 Per-example flow

```
                       +------------------------------+
                       |  Dataset loader              |
                       |  HF: ai-hyz/MemoryAgentBench |
                       |  → subset_v1.jsonl           |
                       +--------------+---------------+
                                      |
                                      v
            +-------------------------+-------------------------+
            |                                                   |
            v                                                   v
[Approach 1: top-k + Token Company]               [Approach 2: Nebula / RAPTOR]
chunk(doc, chunk_size)                            ingest(doc) → vector graph
    → embed(text-embedding-3-small)                   → search(query, effort, weights)
    → top-k(query, k)                                 → result.{sem|proc|ep|src}
    → token_company.compress(aggr=L)                  → trim by activation_score
            |                                                   |
            +-------------------------+-------------------------+
                                      |
                                      v
                       tokens_to_reader = count(context)
                                      |
                                      v
                       reader(gpt-4o-mini, ctx + question)
                                      |
                                      v
                       answer, latency, in/out toks, $
                                      |
                                      v
                       judge(gpt-4o-2024-08-06, q, gold, answer)
                       — branched prompt per question_type
                                      |
                                      v
                       score ∈ {0, 1}, judge_cost
                                      |
                                      v
              append → runs.parquet (DuckDB-queryable)
```

### 6.2 Run schema

One row per (config, example) pair:

```
run_id, ts, task, sub_dataset, example_id, qa_idx,
approach,                         # "ttc" | "nebula" | "raptor" | "baseline_full" | "baseline_random"
hyperparams_json,                 # exact knob values
context_chars, tokens_in_source,
tokens_to_reader, compression_ratio,
answer, gold_list, judge_label, judge_reason, score,
latency_ms_retrieve, latency_ms_compress, latency_ms_reader, latency_ms_judge,
cost_usd_embed, cost_usd_compress, cost_usd_reader, cost_usd_judge, cost_usd_total,
error
```

Compression ratio = `tokens_to_reader / tokens_in_source` where
`tokens_in_source` is the **full-document token count** under
`tiktoken.encoding_for_model("gpt-4o-mini")`. A "no compression" baseline
will report ratio ≈ 1.0; aggressive sweeps sit in 0.05 – 0.5.

### 6.3 Pareto frontier

Per-config aggregate: `(mean_compression_ratio, mean_score, n, ci95)`. A
config P is Pareto-optimal iff no other config Q has both
`Q.score ≥ P.score AND Q.compression_ratio ≤ P.compression_ratio` (with at
least one strict). O(n log n) skyline scan.

Bootstrap CI95 over examples (1 000 resamples) for **both** axes per
config — the y-axis CI is on accuracy, the x-axis CI is on the realised
compression ratio (which varies per example because it's data-dependent).

Plot: x = compression ratio (lower = better), y = accuracy (higher =
better). Color by approach. Mark the Pareto skyline. Overlay the three
baselines (§7.1).

### 6.4 Storage / runtime

Plain Parquet on disk + DuckDB for ad-hoc queries (filter by
sub_dataset, group by hyperparams, export Pareto). No DB needed for a sweep
this size. Cache embeddings keyed on `(chunk_text_sha256,
embedding_model)` — Approach 1's 3 chunk_size variants × 60 configs would
re-embed the same text otherwise. Cache TTC compress outputs keyed on
`(input_sha256, model, aggressiveness)` for the same reason.

---

## 7. Sweep plan and budget

### 7.1 All configs

| ID | Approach | Configs | Notes |
|---|---|---|---|
| `B0_full` | full context, no retrieval | 1 (per reader) | Use `gpt-4.1-nano` 1 M ctx for the giant rows |
| `B1_random` | random k=5 chunks, no retrieval | 1 | Sanity floor |
| `B2_topk_no_compress` | top-k vector retrieval, skip TTC | 3 (k ∈ 5,10,20; chunk_size=512) | Isolates compression contribution |
| `A_ttc` | top-k + Token Company | 27 (3 × 3 × 3) | §4.4 |
| `A_ttc_query_safe` | A_ttc with `<ttc_safe>` around question | 9 (3 × 3 chunk × aggr; pick best k) | Optional; tests undocumented query-awareness |
| `B_nebula` | Nebula | 64 (4 × 4 × 4) | §5.5 |
| `B_raptor` | self-managed RAPTOR | 60 (Latin-hypercube on 6-D) | §5.7 |

→ Headline sweep: **165 configs** (`1 + 1 + 3 + 27 + 9 + 64 + 60`).
Drop `A_ttc_query_safe`, `B_raptor`, and one-of-{Nebula, RAPTOR} for
a **96-config lean run** (`1 + 1 + 3 + 27 + 64`) if budget is tight.

### 7.2 Subset and stratification

The shared subset is `data/subset_v1.jsonl` — **76 records, seed=42**,
built once by `scripts/build_subset.py` and consumed verbatim by both
Approach 1 and Approach 2 (Nebula) so the two tracks are directly
comparable. Modes (`--smoke` = 7 records, `--mini` = 21, `--full` = 76)
are deterministic *prefixes* of the per-task ordering, not independent
samples — a result computed in `--smoke` is also computed (identically)
in `--full`. LongMemEval is stratified to 3 records per `question_type`
× 6 types so all six judge branches are exercised.

(The original three-pass `pilot=50/main=100/final=300` plan was retired
on 2026-05-05 in favour of the smaller fixed subset plus optional
judge-majority retest on Pareto survivors — see §7.3.)

### 7.3 Cost & wall-clock back-of-envelope

Assumptions (unchanged):
- Reader = `gpt-4o-mini`: $0.15 / $0.60 per MTok in/out.
- Judge = `gpt-4o-2024-08-06`: ~500 in / 30 out per call → ~$0.0014/call.
- Mean tokens-to-reader = 4 000.
- Mean reader output = 100 tokens.
- Embedding spend negligible (~$0.0002 per example for Approach 1).
- TTC spend negligible (~$0.05 per million *removed* tokens, ~$1 total).
- Nebula spend **unknown** — gating risk; flag and confirm.

Per-example reader+judge ≈ `4 000 × 0.15/1 e6 + 100 × 0.60/1 e6 + 0.0014` ≈ **$0.00200**.

**Wall-clock is dominated by TTC's 10 RPM hard cap** (user-confirmed,
2026-05-05). OpenAI tier 4 has ~30 K RPM headroom on every model used
and is never the bottleneck. Wall-clock formula:
`ceil(ttc_calls / 10)` minutes.

| Mode | Records | Configs | TTC calls | OpenAI calls | Wall-clock | Cost (rdr + judge) |
|---|---|---|---|---|---|---|
| `--smoke` | 7 (1/task) | 1 | 7 | 14 | ≈ 1 min | ≈ $0.02 |
| `--mini` | 21 (3/task) | 6 (narrowed) | 126 | 252 | ≈ 13 min | ≈ $0.30 |
| `--full` (Approach 1 headline) | 76 | 27 | 2 052 | 4 104 | ≈ 3.4 h | ≈ $5 |
| `--full` + Pareto judge×2 retest | 76 | ~10 survivors | 0 (cached) | ~1 520 | ≈ 5 min | ≈ $2 |

Switching reader to `gpt-4.1-mini` ×2.67 reader cost. Switching judge to
`gpt-5.1` (if you go that way) ×~3 judge cost. In all realistic scenarios
the entire Approach-1 study is **under $10**, with the biggest cost being
wall-clock not dollars.

---

## 8. Open questions and risks

Listed in priority order.

1. **Nebula pricing & quotas**. Resolve at signup before committing harness
   work to Approach 2. If sales-gated and slow, run RAPTOR baseline only and
   put Nebula as one fixed-config data point if/when access lands.
2. **Token Company query-awareness**. The `<ttc_safe>` mechanism is the only
   path to query-aware compression and behaviour around it is undocumented.
   Test on a handful of LongMemEval rows in the pilot before scaling.
3. **Reader choice**. `gpt-4o-mini` for paper-comparability vs `gpt-5.1` for
   capability headroom. Recommend gpt-4o-mini as primary and treat gpt-5.1
   as an optional capability anchor (rerun the Pareto survivors only).
4. **Subset reproducibility**. Pin a `subset_v1.jsonl` early and never
   regenerate it across passes — otherwise CIs across passes are not
   comparable.
5. **Dataset drift**. The HF dataset has been edited multiple times in 2025.
   Pin a commit hash on first download.
6. **Cost overruns from large-context tasks**. ∞Bench-Sum and `*_262k` will
   blow the budget if mistakenly included in the pilot. Skip until the
   frontier is stable.
7. **TTC reproducibility**. Closed model, no weights, no version pin finer
   than `bear-1.x`. Record the exact `bear` version used per run.
8. **Judge variance**. `gpt-4o` is non-deterministic at the head; consider
   running the judge twice on Pareto survivors and taking majority for the
   final plot.

---

## 9. Implementation checklist

Order the work this way:

1. `apps/harness/data.py` — load HF dataset (commit-pinned), build
   `subset_v1.jsonl` with stratified seed sampling per §2.4.
2. `apps/harness/eval.py` — port the four eval styles from
   `utils/eval_other_utils.py` and `llm_based_eval/*.py`. Use the repo's
   prompts verbatim.
3. `apps/harness/reader.py` — single OpenAI chat-completion wrapper with
   token-count instrumentation, retry, and per-call cost log.
4. `apps/harness/retrievers/topk.py` — chunk + embed (cached) + top-k.
5. `apps/harness/compressors/ttc.py` — wrap the existing `compress.py` flow
   into a class with `aggressiveness` / `model` knobs and a result cache.
6. `apps/harness/retrievers/nebula.py` — ingest + poll + search; reset per
   example via `collections.create` / `.delete`. Stub if access blocked.
7. `apps/harness/retrievers/raptor.py` — LlamaIndex `TreeIndex` or thin
   custom; the only retriever with all six sweep knobs.
8. `apps/harness/runner.py` — `for example in subset: for config in grid:
   run -> append parquet`. Threadpool size ≤16. Resume-from-checkpoint by
   `(run_id, example_id, config_hash)`.
9. `apps/harness/pareto.py` — DuckDB query → `(approach, hyperparams_json)`
   → `(mean_ratio, mean_score, ci95)` → skyline scan → Plotly figure.
10. **Validation gate before any sweep**: reproduce one paper number (e.g.
    `gpt-4o-mini` long-context on LongMemEval-S\* AR ≈ 49 %). Do not start
    the pilot until this is within ±5 %.
11. Pilot 50/task → drop dominated → main 100/task → final 300/task on
    Pareto survivors.

---

## 10. Assumptions to verify before locking the sweep

- **A1**: Token Company `aggressiveness` ∈ `{0.05, 0.3, 0.5}` (the
  reduced 2026-05-05 mainline) is a sufficient sweep. *Verify*:
  spot-check 0.1 (interpolating gap) and 0.9 (high-end) on Pareto
  survivors (5 examples each); if the realised-ratio curve is
  non-monotonic between the sampled points or extends past 0.5, expand.
- **A2**: Nebula's `result_layers` is client-side (you choose what to feed
  the reader) — i.e. the search response always returns all four typed
  layers and you concat the subset. *Verify*: SDK first call.
- **A3**: Nebula collection delete fully releases storage for reuse so
  per-example reset is safe. *Verify*: SDK first call.
- **A4**: `text-embedding-3-small` at full 1 536 dims is good enough on
  all four competencies. *Verify*: 1-config -large ablation on
  LongMemEval AR before main pass.
- **A5**: Nebula pricing fits the sweep budget. *Verify at signup*.
- **A6**: The user's prototype pin on `gpt-5.1` was illustrative, not
  required. *Treat as confirmed* unless told otherwise; gpt-4o-mini is
  primary.

---

## 11. Sources

- HuggingFace dataset card — https://huggingface.co/datasets/ai-hyz/MemoryAgentBench
- Repo — https://github.com/HUST-AI-HYZ/MemoryAgentBench
- Paper — https://arxiv.org/abs/2507.05257 (HTML v1 / v3 used for Table 3)
- Token Company homepage / docs / benchmarks — https://thetokencompany.com/{,docs,benchmarks/squad-v2,benchmarks/financebench,benchmarks/latency}
- Token Company Python SDK — https://github.com/TheTokenCompany/tokenc-python-sdk
- Token Company YC launch — https://www.ycombinator.com/launches/Pb3-the-token-company-intelligent-compression-for-llm-context-bloat
- Nebula docs — https://docs.trynebula.ai/{introduction,llms.txt,guides/search}
- Nebula homepage / pricing — https://trynebula.ai/{,pricing}
- OpenAI pricing references — devtk.ai (2026 guide), cloudzero.com, tokenmix.ai (embedding comparison)
- LoCoMo benchmark — https://snap-research.github.io/locomo/
- Letta memory benchmarks — https://www.letta.com/blog/benchmarking-ai-agent-memory
- Mem0 graph-memory blog (Jan 2026) — https://mem0.ai/blog/graph-memory-solutions-ai-agents
