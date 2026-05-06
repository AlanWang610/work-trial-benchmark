"""Approach 1 sweep CLI — top-k vector retrieval + Token Company compression.

Without --execute, this is a dry-run: prints the config grid, subset
size, estimated TTC calls, OpenAI cost, and wall-clock at TTC's 10 RPM
cap, then exits without making any API calls. With --execute, it kicks
off the runner.
"""

from __future__ import annotations

import argparse
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from harness.data import build_subset, load_subset
from harness.runner_topk_ttc import estimate, run
from harness.settings import settings
from harness.sweep_topk_ttc import grid_for_mode


def main() -> int:
    parser = argparse.ArgumentParser(description="Approach 1 sweep — top-k + TTC")
    mode_grp = parser.add_mutually_exclusive_group()
    mode_grp.add_argument(
        "--smoke",
        action="store_const",
        dest="mode",
        const="smoke",
        help="1/task = 7 records x 1 config (~1 min, ~$0.02)",
    )
    mode_grp.add_argument(
        "--mini",
        action="store_const",
        dest="mode",
        const="mini",
        help="3/task = 21 records x 6 narrowed configs (~13 min, ~$0.30)",
    )
    mode_grp.add_argument(
        "--full",
        action="store_const",
        dest="mode",
        const="full",
        help="76 records x 27 configs (~3.4 h, ~$5)",
    )
    parser.set_defaults(mode="smoke")
    parser.add_argument(
        "--include-no-compress",
        action="store_true",
        help="Add B2 baseline (3 identity-compressor configs, no TTC quota)",
    )
    parser.add_argument(
        "--ttc-rpm",
        type=float,
        default=None,
        help="Override TTC requests-per-minute cap (default from .env / 10)",
    )
    parser.add_argument(
        "--threadpool-size",
        type=int,
        default=4,
        help="Reader/judge worker threads (default 4; bump on tier 4)",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually call the APIs; without this flag, prints a dry-run summary and exits 0.",
    )
    args = parser.parse_args()

    if args.ttc_rpm is not None:
        settings.TTC_RPM = args.ttc_rpm

    info = build_subset()
    header = info["header"]
    _, records = load_subset(args.mode)
    configs = grid_for_mode(args.mode, include_no_compress=args.include_no_compress)
    est = estimate(configs, records)

    print(f"Mode:      {args.mode}")
    print(f"Subset:    {info['path']}")
    print(
        f"           total={header['total']}, seed={header['seed']}, "
        f"rev={header['hf_revision'][:8]}"
    )
    print(f"Records:   {len(records)} ({args.mode} = per-task prefix of subset)")
    print(f"Configs:   {est['configs']}")
    print(f"  - approach1 (TTC): {sum(1 for c in configs if c.approach == 'ttc')}")
    print(f"  - identity (B2 ): {sum(1 for c in configs if c.approach == 'identity')}")
    print()
    print("Config grid:")
    print(f"  {'approach':<10}{'chunk':>6}{'k':>4}{'aggr':>7}  hash")
    for c in configs:
        aggr = f"{c.aggressiveness:.2f}" if c.aggressiveness is not None else "-"
        print(f"  {c.approach:<10}{c.chunk_size:>6d}{c.k:>4d}{aggr:>7}  {c.config_hash}")
    print()
    print("Estimates:")
    print(f"  TTC calls            = {est['ttc_calls']:,}")
    print(f"  OpenAI calls (rdr+jg)= {est['openai_calls']:,}")
    print(
        f"  Wall-clock @ {settings.TTC_RPM:.0f} RPM = {est['wall_clock_minutes_at_10rpm']} min "
        f"(~{est['wall_clock_minutes_at_10rpm'] / 60:.2f} h)"
    )
    print(f"  Reader+judge cost    ~ ${est['cost_reader_judge_usd']:.2f}")
    print()

    if not args.execute:
        print("[dry-run] No API calls made. Re-run with --execute to start the sweep.")
        return 0

    print("[execute] Starting sweep — TTC at 10 RPM is the throughput cap.")
    print(f"[execute] Output: {settings.runs_jsonl}")
    print()
    run(configs, records, threadpool_size=args.threadpool_size)
    return 0


if __name__ == "__main__":
    sys.exit(main())
