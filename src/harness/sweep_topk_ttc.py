"""Approach 1 (top-k + TTC) sweep grid: 27 configs (3 chunk × 3 k × 3 aggr)."""

from __future__ import annotations

from itertools import product

from harness.sweep_common import Config

CHUNK_SIZES = (256, 512, 1024)
TOP_KS = (5, 10, 20)
AGGRESSIVENESS = (0.05, 0.3, 0.5)

# Narrowed --mini grid: 6 configs.
MINI_CHUNK_SIZES = (512,)
MINI_TOP_KS = (5, 10)
MINI_AGGRESSIVENESS = (0.05, 0.3, 0.5)

# Smoke: a single representative config.
SMOKE_CHUNK_SIZE = 512
SMOKE_TOP_K = 5
SMOKE_AGGR = 0.3

# B2 baseline (top-k, no compression): 3 configs.
B2_CHUNK_SIZE = 512
B2_TOP_KS = (5, 10, 20)


def approach1_grid() -> list[Config]:
    return [
        Config.make("ttc", cs, k, a) for cs, k, a in product(CHUNK_SIZES, TOP_KS, AGGRESSIVENESS)
    ]


def mini_grid() -> list[Config]:
    return [
        Config.make("ttc", cs, k, a)
        for cs, k, a in product(MINI_CHUNK_SIZES, MINI_TOP_KS, MINI_AGGRESSIVENESS)
    ]


def smoke_grid() -> list[Config]:
    return [Config.make("ttc", SMOKE_CHUNK_SIZE, SMOKE_TOP_K, SMOKE_AGGR)]


def topk_no_compress_grid() -> list[Config]:
    return [Config.make("identity", B2_CHUNK_SIZE, k, None) for k in B2_TOP_KS]


def grid_for_mode(mode: str, *, include_no_compress: bool = False) -> list[Config]:
    if mode == "smoke":
        configs = smoke_grid()
    elif mode == "mini":
        configs = mini_grid()
    elif mode == "full":
        configs = approach1_grid()
    else:
        raise ValueError(f"unknown mode: {mode!r}")
    if include_no_compress:
        configs = configs + topk_no_compress_grid()
    return configs
