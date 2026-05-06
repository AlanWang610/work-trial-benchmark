"""One-time download of the MemoryAgentBench HF dataset.

Resolves the latest commit on the `main` branch (or `HF_REVISION` if set
in .env), pulls all four split splits to data/hf/, and writes the
resolved sha to data/dataset_revision.txt.
"""

from __future__ import annotations

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from datasets import load_dataset

from harness.data import DATASET_NAME, download
from harness.settings import settings


def main() -> int:
    settings.ensure_dirs()
    print(f"Downloading {DATASET_NAME} to {settings.hf_cache_dir}...")
    revision = download()
    print(f"Resolved revision: {revision}")
    print(f"Wrote: {settings.revision_path}")
    print()
    print("Per-split row counts:")
    for split in (
        "Accurate_Retrieval",
        "Long_Range_Understanding",
        "Test_Time_Learning",
        "Conflict_Resolution",
    ):
        ds = load_dataset(
            DATASET_NAME,
            split=split,
            revision=revision,
            cache_dir=str(settings.hf_cache_dir),
        )
        print(f"  {split:30s} = {len(ds):4d} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
