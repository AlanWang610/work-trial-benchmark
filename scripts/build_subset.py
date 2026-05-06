"""Build the shared deterministic subset (76 records) at data/subset_v1.jsonl.

The same file is consumed by Approach 1 (top-k + TTC) and a future
Approach 2 (Nebula) so both tracks see identical records.
"""

from __future__ import annotations

import argparse
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from harness.data import build_subset


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rebuild", action="store_true", help="Regenerate even if subset_v1.jsonl exists"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    info = build_subset(seed=args.seed, rebuild=args.rebuild)
    header = info["header"]
    print(f"Wrote: {info['path']}")
    print(f"  total records       = {header['total']}")
    print(f"  expected total      = {header['expected_total']}")
    print(f"  seed                = {header['seed']}")
    print(f"  hf_revision         = {header['hf_revision']}")
    print(f"  schema_version      = {header['schema_version']}")
    print("  per-task counts:")
    for task, n in header["per_task_counts"].items():
        print(f"    {task:30s} = {n:3d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
