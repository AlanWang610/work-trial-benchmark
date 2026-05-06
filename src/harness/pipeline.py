"""Approach 1 pipeline: top-k retrieve → compress → reader chat → judge."""

from __future__ import annotations

import time
from dataclasses import dataclass

from harness.chunker import count_tokens
from harness.compressors.base import Compressor
from harness.eval import score_longmem_judge, score_substring
from harness.reader import reader_call
from harness.retrievers.base import Retriever

CHUNK_SEPARATOR = "\n\n---\n\n"


@dataclass(slots=True)
class PipelineResult:
    answer: str
    score: float
    judge_label: str
    tokens_in_source: int
    tokens_to_reader: int
    compression_ratio: float
    latency_ms_retrieve: int
    latency_ms_compress: int
    latency_ms_reader: int
    latency_ms_judge: int
    cost_usd_embed: float
    cost_usd_compress: float
    cost_usd_reader: float
    cost_usd_judge: float
    cost_usd_total: float
    error: str | None
    reader_model: str
    reader_messages: list[dict[str, str]]


def _build_prompt(compressed_context: str, question: str) -> str:
    if compressed_context:
        return f"{compressed_context}\n\nQuestion: {question}\nAnswer:"
    return f"Question: {question}\nAnswer:"


def run_one(
    record: dict,
    *,
    retriever: Retriever,
    compressor: Compressor,
    k: int,
    tokens_in_source: int | None = None,
) -> PipelineResult:
    question = record["question"]
    eval_kind = record["eval_kind"]
    sub_dataset = record.get("sub_dataset", "")
    gen_max = int(record.get("gen_max_tokens") or 50)
    answers = record.get("answers", [])

    if tokens_in_source is None:
        tokens_in_source = count_tokens(record.get("context", ""))

    err: str | None = None
    cost_embed = cost_compress = cost_reader = cost_judge = 0.0
    lat_retrieve = lat_compress = lat_reader = lat_judge = 0
    tokens_to_reader = 0
    answer_text = ""
    judge_label = ""
    score = 0.0
    reader_model = ""
    reader_messages: list[dict[str, str]] = []

    try:
        retrieve = retriever.retrieve(question, k, doc_id=record["row_id"])
        cost_embed += retrieve.cost_usd
        lat_retrieve = retrieve.latency_ms
        joined = CHUNK_SEPARATOR.join(retrieve.chunks)

        t0 = time.monotonic()
        comp = compressor.compress(joined, query=question)
        cost_compress += comp.cost_usd
        lat_compress = comp.latency_ms or int((time.monotonic() - t0) * 1000)
        tokens_to_reader = int(comp.output_tokens or 0)

        prompt = _build_prompt(comp.output, question)
        chat = reader_call(prompt, gen_max_tokens=gen_max)
        cost_reader += chat.cost_usd
        lat_reader = chat.latency_ms
        answer_text = chat.text
        reader_model = chat.model
        reader_messages = chat.messages

        if eval_kind == "longmem_judge":
            score, judge_label, judge_cost = score_longmem_judge(
                answer_text,
                question=question,
                answers=answers,
                question_type=record.get("question_type"),
                question_id=record.get("question_id"),
            )
            cost_judge += judge_cost
            lat_judge = 0
        else:
            score = score_substring(answer_text, answers, sub_dataset=sub_dataset)
            judge_label = "score=1" if score == 1.0 else "score=0"
    except Exception as exc:  # noqa: BLE001
        err = f"{type(exc).__name__}: {exc}"

    cost_total = cost_embed + cost_compress + cost_reader + cost_judge
    ratio = (tokens_to_reader or 0) / tokens_in_source if tokens_in_source else 0.0
    return PipelineResult(
        answer=answer_text,
        score=score,
        judge_label=judge_label,
        tokens_in_source=tokens_in_source,
        tokens_to_reader=tokens_to_reader,
        compression_ratio=ratio,
        latency_ms_retrieve=lat_retrieve,
        latency_ms_compress=lat_compress,
        latency_ms_reader=lat_reader,
        latency_ms_judge=lat_judge,
        cost_usd_embed=cost_embed,
        cost_usd_compress=cost_compress,
        cost_usd_reader=cost_reader,
        cost_usd_judge=cost_judge,
        cost_usd_total=cost_total,
        error=err,
        reader_model=reader_model,
        reader_messages=reader_messages,
    )
