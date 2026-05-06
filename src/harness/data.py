"""Dataset download + shared subset builder.

The same `data/subset_v1.jsonl` is consumed by both Approach 1 (top-k +
TTC) and Approach 2 (Nebula). Modes (`smoke` / `mini` / `full`) are
deterministic prefixes of the per-task ordering.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_dataset
from huggingface_hub import HfApi

from harness.settings import settings
from harness.tasks import TASKS, TOTAL_SUBSET_SIZE

DATASET_NAME = "ai-hyz/MemoryAgentBench"
SCHEMA_VERSION = 1


def download() -> str:
    settings.ensure_dirs()
    api = HfApi()
    info = api.dataset_info(DATASET_NAME, revision=settings.HF_REVISION or None)
    revision = info.sha
    for task in TASKS.values():
        load_dataset(
            DATASET_NAME,
            split=task["split"],
            revision=revision,
            cache_dir=str(settings.hf_cache_dir),
        )
    settings.revision_path.write_text(revision + "\n", encoding="utf-8")
    return revision


def _read_revision() -> str | None:
    if settings.revision_path.exists():
        return settings.revision_path.read_text(encoding="utf-8").strip() or None
    return None


def _record_id(row_id: str, qa_idx: int, question: str) -> str:
    return hashlib.sha1(f"{row_id}:{qa_idx}:{question}".encode()).hexdigest()[:16]


def _meta_get(metadata: Any, key: str, qa_idx: int) -> Any:
    """Read parallel-list field from metadata; tolerate naming variations."""
    if not isinstance(metadata, dict):
        return None
    candidates = [key]
    if key == "question_type":
        candidates += ["question_types"]
    if key == "question_id":
        candidates += ["question_ids", "qa_pair_ids", "qa_pair_id"]
    if key == "source":
        candidates += ["sub_dataset", "sub_datasets"]
    for cand in candidates:
        if cand not in metadata:
            continue
        val = metadata[cand]
        if isinstance(val, list):
            if 0 <= qa_idx < len(val):
                return val[qa_idx]
            continue
        return val
    return None


def _row_sub_dataset(row: dict) -> str | None:
    md = row.get("metadata") or {}
    if not isinstance(md, dict):
        return None
    for k in ("source", "sub_dataset"):
        v = md.get(k)
        if isinstance(v, str):
            return v
    return None


def _flatten_task_rows(task_name: str, dataset_rows: list[dict]) -> list[dict]:
    cfg = TASKS[task_name]
    sd_filter = cfg["sd_filter"].lower()
    out: list[dict] = []
    for row_idx, row in enumerate(dataset_rows):
        sd = _row_sub_dataset(row)
        if not sd or sd.lower() != sd_filter:
            continue
        questions = row.get("questions") or []
        answers = row.get("answers") or []
        context = row.get("context") or ""
        row_id = f"{task_name}:{row_idx}"
        for qa_idx, q in enumerate(questions):
            ans = answers[qa_idx] if qa_idx < len(answers) else []
            qtype = _meta_get(row.get("metadata"), "question_type", qa_idx)
            qid = _meta_get(row.get("metadata"), "question_id", qa_idx)
            out.append(
                {
                    "record_id": _record_id(row_id, qa_idx, q),
                    "row_id": row_id,
                    "qa_idx": qa_idx,
                    "task": task_name,
                    "sub_dataset": sd,
                    "context": context,
                    "question": q,
                    "answers": ans,
                    "question_type": qtype,
                    "question_id": qid,
                    "gen_max_tokens": cfg["gen_max"],
                    "eval_kind": cfg["eval"],
                }
            )
    return out


def _stratified_pick(
    records: list[dict], n: int, stratify_on: str | None, rng: np.random.Generator
) -> list[dict]:
    if n >= len(records):
        shuffled = records[:]
        rng.shuffle(shuffled)
        return shuffled
    if not stratify_on:
        idx = rng.permutation(len(records))[:n]
        return [records[i] for i in sorted(idx)]
    groups: dict[str, list[dict]] = {}
    for r in records:
        key = str(r.get(stratify_on))
        groups.setdefault(key, []).append(r)
    for v in groups.values():
        rng.shuffle(v)
    keys = sorted(groups.keys())
    out: list[dict] = []
    cursors = {k: 0 for k in keys}
    while len(out) < n:
        progress = False
        for k in keys:
            if len(out) >= n:
                break
            i = cursors[k]
            if i < len(groups[k]):
                out.append(groups[k][i])
                cursors[k] = i + 1
                progress = True
        if not progress:
            break
    return out


def build_subset(*, seed: int = 42, rebuild: bool = False) -> dict:
    settings.ensure_dirs()
    if settings.subset_path.exists() and not rebuild:
        with settings.subset_path.open("r", encoding="utf-8") as f:
            header = json.loads(f.readline().lstrip("# ").strip())
            count = sum(1 for _ in f)
        return {"path": str(settings.subset_path), "header": header, "records": count}

    revision = _read_revision() or download()
    rng = np.random.default_rng(seed)

    splits_loaded: dict[str, list[dict]] = {}
    selected: list[dict] = []
    per_task_counts: dict[str, int] = {}

    for task_name, cfg in TASKS.items():
        split = cfg["split"]
        if split not in splits_loaded:
            ds = load_dataset(
                DATASET_NAME,
                split=split,
                revision=revision,
                cache_dir=str(settings.hf_cache_dir),
            )
            splits_loaded[split] = list(ds)
        rows = splits_loaded[split]
        flattened = _flatten_task_rows(task_name, rows)
        picked = _stratified_pick(flattened, cfg["samples"], cfg["stratify_on"], rng)
        picked.sort(key=lambda r: r["record_id"])
        per_task_counts[task_name] = len(picked)
        selected.extend(picked)

    header = {
        "seed": seed,
        "hf_revision": revision,
        "schema_version": SCHEMA_VERSION,
        "total": len(selected),
        "expected_total": TOTAL_SUBSET_SIZE,
        "per_task_counts": per_task_counts,
    }
    tmp = settings.subset_path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.write("# " + json.dumps(header) + "\n")
        for rec in selected:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp.replace(settings.subset_path)
    return {"path": str(settings.subset_path), "header": header, "records": len(selected)}


def load_subset_records(path: Path | None = None) -> tuple[dict, list[dict]]:
    p = Path(path) if path else settings.subset_path
    with p.open("r", encoding="utf-8") as f:
        header = json.loads(f.readline().lstrip("# ").strip())
        records = [json.loads(line) for line in f if line.strip()]
    return header, records


def load_subset(mode: str = "full", path: Path | None = None) -> tuple[dict, list[dict]]:
    """Return records for a given mode. smoke ⊂ mini ⊂ full (per-task prefix).

    mode = "full"  → all 76 records
    mode = "mini"  → first 3 per task (record_id-sorted) = 21
    mode = "smoke" → first 1 per task = 7
    """
    header, recs = load_subset_records(path)
    if mode == "full":
        return header, recs
    if mode == "mini":
        per_task = 3
    elif mode == "smoke":
        per_task = 1
    else:
        raise ValueError(f"unknown mode: {mode!r}")
    by_task: dict[str, list[dict]] = {}
    for r in recs:
        by_task.setdefault(r["task"], []).append(r)
    out: list[dict] = []
    for task_name in TASKS:
        task_recs = by_task.get(task_name, [])
        out.extend(task_recs[:per_task])
    return header, out
