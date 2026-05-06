"""Convert outputs/runs.jsonl → outputs/runs.parquet via DuckDB."""

from __future__ import annotations

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import duckdb

from harness.settings import settings


def main() -> int:
    settings.ensure_dirs()
    if not settings.runs_jsonl.exists():
        print(f"No runs found at {settings.runs_jsonl}")
        return 1
    src = str(settings.runs_jsonl).replace("\\", "/")
    dst = str(settings.runs_parquet).replace("\\", "/")
    duckdb.sql(
        f"COPY (SELECT * FROM read_json_auto('{src}', format='newline_delimited')) "
        f"TO '{dst}' (FORMAT PARQUET)"
    )
    print(f"Wrote: {settings.runs_parquet}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
