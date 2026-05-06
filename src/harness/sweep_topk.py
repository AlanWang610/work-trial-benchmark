"""Approach 3 (top-k, no compression) sweep grid: 9 configs (3 chunk × 3 k).

Mirrors the chunk_size and k axes of Approach 1 so the two can be plotted on
shared axes; drops the aggressiveness axis (no compression).
"""

from __future__ import annotations

from itertools import product

from harness.sweep_common import Config

CHUNK_SIZES = (256, 512, 1024)
TOP_KS = (5, 10, 20)

# Narrowed --mini grid: 3 configs (k axis at chunk=512).
MINI_CHUNK_SIZE = 512
MINI_TOP_KS = (5, 10, 20)

# Smoke: a single representative config (matches Approach 1's smoke chunk/k).
SMOKE_CHUNK_SIZE = 512
SMOKE_TOP_K = 5


def approach3_grid() -> list[Config]:
    return [Config.make("identity", cs, k, None) for cs, k in product(CHUNK_SIZES, TOP_KS)]


def mini_grid() -> list[Config]:
    return [Config.make("identity", MINI_CHUNK_SIZE, k, None) for k in MINI_TOP_KS]


def smoke_grid() -> list[Config]:
    return [Config.make("identity", SMOKE_CHUNK_SIZE, SMOKE_TOP_K, None)]


def grid_for_mode(mode: str) -> list[Config]:
    if mode == "smoke":
        return smoke_grid()
    if mode == "mini":
        return mini_grid()
    if mode == "full":
        return approach3_grid()
    raise ValueError(f"unknown mode: {mode!r}")
