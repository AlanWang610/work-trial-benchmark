"""OpenAI chat-completions wrapper with retry, usage tracking, and cost computation.

OpenAI tier 4 — concurrency 32 is fine; reader/judge/embed share that pool.
"""

from dataclasses import dataclass

from openai import APIError, APITimeoutError, OpenAI, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from harness.settings import settings

# USD per 1M tokens (input, output) for the models we use.
PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o-mini-2024-07-18": (0.15, 0.60),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o-2024-08-06": (2.50, 10.00),
    "gpt-4o": (2.50, 10.00),
    "text-embedding-3-small": (0.02, 0.0),
    "text-embedding-3-large": (0.13, 0.0),
}


@dataclass(slots=True)
class ChatResult:
    text: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: int
    model: str
    messages: list[dict[str, str]]


def cost_for(model: str, input_tokens: int, output_tokens: int) -> float:
    in_rate, out_rate = PRICING.get(model, (0.0, 0.0))
    return (input_tokens * in_rate + output_tokens * out_rate) / 1_000_000


_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.OPENAI_API_KEY, timeout=settings.HTTP_TIMEOUT_S)
    return _client


@retry(
    retry=retry_if_exception_type((RateLimitError, APITimeoutError, APIError)),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    stop=stop_after_attempt(5),
    reraise=True,
)
def chat(
    messages: list[dict[str, str]],
    *,
    model: str,
    max_tokens: int,
    temperature: float = 0.7,
    seed: int | None = None,
) -> ChatResult:
    import time

    t0 = time.monotonic()
    resp = _get_client().chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        seed=seed,
    )
    latency_ms = int((time.monotonic() - t0) * 1000)
    text = resp.choices[0].message.content or ""
    usage = resp.usage
    in_tok = usage.prompt_tokens if usage else 0
    out_tok = usage.completion_tokens if usage else 0
    return ChatResult(
        text=text,
        input_tokens=in_tok,
        output_tokens=out_tok,
        cost_usd=cost_for(model, in_tok, out_tok),
        latency_ms=latency_ms,
        model=model,
        messages=messages,
    )


def reader_call(prompt: str, *, gen_max_tokens: int, system: str | None = None) -> ChatResult:
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return chat(
        messages,
        model=settings.READER_MODEL,
        max_tokens=gen_max_tokens,
        temperature=settings.READER_TEMPERATURE,
        seed=settings.READER_SEED,
    )


def judge_call(prompt: str) -> ChatResult:
    return chat(
        [{"role": "user", "content": prompt}],
        model=settings.JUDGE_MODEL,
        max_tokens=10,
        temperature=0.0,
        seed=settings.READER_SEED,
    )
