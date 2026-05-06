"""Approach 2 (Nebula) sweep grid.

3 configs: effort ∈ {low, medium, high}. Top_k is intentionally not swept —
Nebula is designed to choose what to return; forcing a fixed cap fights the
system. Compare against Approach 1 on the natural axis (accuracy vs
tokens_to_reader, both recorded in the run schema).

Reuses `Config.make()` from `harness.sweep_common`.
"""

from __future__ import annotations

from harness.sweep_common import Config

NEBULA_EFFORTS = ("low", "medium", "high")


def nebula_full_grid() -> list[Config]:
    """3 configs: effort ∈ {low, medium, high}."""
    return [
        Config.make("nebula", None, None, None, nebula_effort=e) for e in NEBULA_EFFORTS
    ]


def nebula_mini_grid() -> list[Config]:
    """1 config: effort=medium."""
    return [Config.make("nebula", None, None, None, nebula_effort="medium")]


def nebula_smoke_grid() -> list[Config]:
    """1 config: effort=medium."""
    return [Config.make("nebula", None, None, None, nebula_effort="medium")]


def nebula_grid_for_mode(mode: str) -> list[Config]:
    if mode == "smoke":
        return nebula_smoke_grid()
    if mode == "mini":
        return nebula_mini_grid()
    if mode == "full":
        return nebula_full_grid()
    raise ValueError(f"unknown mode: {mode!r}")
