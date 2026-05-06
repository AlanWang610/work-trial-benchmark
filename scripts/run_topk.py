"""Approach 3 sweep CLI — top-k vector retrieval, no compression.

Same retrieval grid axes as Approach 1 (chunk_size × k) but feeds raw
top-k chunks straight to the reader. Isolates the Token Company
compression contribution against an otherwise-identical pipeline.

Without --execute, this is a dry-run: prints the config grid, subset
size, estimated OpenAI cost, and wall-clock, then exits without making
any API calls. With --execute, it kicks off the runner.
"""

from __future__ import annotations

import argparse
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from harness.data import build_subset, load_subset
from harness.runner_topk import estimate, run
from harness.settings import settings
from harness.sweep_topk import grid_for_mode


def main() -> int:
    parser = argparse.ArgumentParser(description="Approach 3 sweep — top-k, no compression")
    mode_grp = parser.add_mutually_exclusive_group()
    mode_grp.add_argument(
        "--smoke",
        action="store_const",
        dest="mode",
        const="smoke",
        help="1 config × 7 records (~10s, ~$0.03)",
    )
    mode_grp.add_argument(
        "--mini",
        action="store_const",
        dest="mode",
        const="mini",
        help="3 configs (k axis @ chunk=512) × 21 records (~30s, ~$0.13)",
    )
    mode_grp.add_argument(
        "--full",
        action="store_const",
        dest="mode",
        const="full",
        help="9 configs (chunk × k) × 76 records (~3 min, ~$1.40)",
    )
    parser.set_defaults(mode="smoke")
    parser.add_argument(
        "--threadpool-size",
        type=int,
        default=16,
        help="Reader/judge worker threads (default 16; no TTC bottleneck on tier 4)",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually call the APIs; without this flag, prints a dry-run summary and exits 0.",
    )
    args = parser.parse_args()

    info = build_subset()
    header = info["header"]
    _, records = load_subset(args.mode)
    configs = grid_for_mode(args.mode)
    est = estimate(configs, records, threadpool_size=args.threadpool_size)

    print(f"Mode:      {args.mode}")
    print(f"Subset:    {info['path']}")
    print(
        f"           total={header['total']}, seed={header['seed']}, "
        f"rev={header['hf_revision'][:8]}"
    )
    print(f"Records:   {len(records)} ({args.mode} = per-task prefix of subset)")
    print(f"Configs:   {est['configs']} (identity / no compression)")
    print()
    print("Config grid:")
    print(f"  {'approach':<10}{'chunk':>6}{'k':>4}  hash")
    for c in configs:
        print(f"  {c.approach:<10}{c.chunk_size:>6d}{c.k:>4d}  {c.config_hash}")
    print()
    print("Estimates:")
    print(f"  OpenAI calls (rdr+jg)= {est['openai_calls']:,}")
    print(
        f"  Wall-clock @ tp={args.threadpool_size} = {est['wall_clock_minutes']} min "
        f"(~{est['wall_clock_minutes'] / 60:.2f} h)"
    )
    print(f"  Reader+judge cost    ~ ${est['cost_reader_judge_usd']:.2f}")
    print()

    if not args.execute:
        print("[dry-run] No API calls made. Re-run with --execute to start the sweep.")
        return 0

    print("[execute] Starting sweep — no compression, OpenAI only.")
    print(f"[execute] Output: {settings.runs_jsonl}")
    print()
    run(configs, records, threadpool_size=args.threadpool_size)
    return 0


if __name__ == "__main__":
    sys.exit(main())
