"""Approach 2 sweep CLI - Nebula memory layer (effort sweep, no top_k).

Without --execute, this is a dry-run: prints the config grid, subset
size, estimated Nebula calls, OpenAI cost, and wall-clock at the
configured RPM, then exits without making any API calls. With --execute,
it kicks off the runner.
"""

from __future__ import annotations

import argparse
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from harness.data import build_subset, load_subset
from harness.runner_nebula import estimate_nebula, run_nebula
from harness.settings import settings
from harness.sweep_nebula import nebula_grid_for_mode


def main() -> int:
    parser = argparse.ArgumentParser(description="Approach 2 sweep - Nebula")
    mode_grp = parser.add_mutually_exclusive_group()
    mode_grp.add_argument(
        "--smoke",
        action="store_const",
        dest="mode",
        const="smoke",
        help="1/task = 7 records x 1 config (medium effort, ~1 min, ~$0.02)",
    )
    mode_grp.add_argument(
        "--mini",
        action="store_const",
        dest="mode",
        const="mini",
        help="3/task = 21 records x 1 config (medium effort, ~1 min, ~$0.05)",
    )
    mode_grp.add_argument(
        "--full",
        action="store_const",
        dest="mode",
        const="full",
        help="76 records x 3 configs (low/medium/high effort, ~8 min @ 30 RPM, ~$0.50)",
    )
    parser.set_defaults(mode="smoke")
    parser.add_argument(
        "--nebula-rpm",
        type=float,
        default=None,
        help="Override Nebula requests-per-minute cap (default from .env / 30)",
    )
    parser.add_argument(
        "--prepare-wait-s",
        type=float,
        default=0.0,
        help="Sleep N seconds after Phase 1 ingest before Phase 2 search "
        "(bump if --smoke shows empty sources)",
    )
    parser.add_argument(
        "--threadpool-size",
        type=int,
        default=4,
        help="Reader/judge worker threads (default 4)",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually call the APIs; without this flag, prints a dry-run summary and exits 0.",
    )
    args = parser.parse_args()

    if args.nebula_rpm is not None:
        settings.NEBULA_RPM = args.nebula_rpm

    info = build_subset()
    header = info["header"]
    _, records = load_subset(args.mode)
    configs = nebula_grid_for_mode(args.mode)
    est = estimate_nebula(configs, records)

    print(f"Mode:      {args.mode}")
    print(f"Subset:    {info['path']}")
    print(
        f"           total={header['total']}, seed={header['seed']}, "
        f"rev={header['hf_revision'][:8]}"
    )
    print(f"Records:   {len(records)} ({args.mode} = per-task prefix of subset)")
    print(f"Configs:   {est['configs']} (Nebula effort sweep)")
    print()
    print("Config grid:")
    print(f"  {'approach':<10}{'effort':>8}  hash")
    for c in configs:
        print(f"  {c.approach:<10}{c.nebula_effort or '-':>8}  {c.config_hash}")
    print()
    print("Estimates:")
    print(f"  Nebula ingests       = {est['nebula_ingests']:,}")
    print(f"  Nebula searches      = {est['nebula_searches']:,}")
    print(f"  OpenAI calls (rdr+jg)= {est['openai_calls']:,}")
    print(
        f"  Wall-clock @ {settings.NEBULA_RPM:.0f} RPM = {est['wall_clock_minutes']} min "
        f"(~{est['wall_clock_minutes'] / 60:.2f} h)"
    )
    print(f"  Reader+judge cost    ~ ${est['cost_reader_judge_usd']:.2f}")
    print()

    if not args.execute:
        print("[dry-run] No API calls made. Re-run with --execute to start the sweep.")
        return 0

    print("[execute] Starting Nebula sweep.")
    print(f"[execute] Output: {settings.runs_jsonl}")
    print()
    run_nebula(
        configs,
        records,
        threadpool_size=args.threadpool_size,
        prepare_wait_s=args.prepare_wait_s,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
