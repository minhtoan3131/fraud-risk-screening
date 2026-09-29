#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "final_pipeline" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

"""Walk through the external raw-dataset boundary without changing data."""

import argparse

from fraud_screening.replay.raw_dataset import (
    CANONICAL_RAW_DATASET,
    inspect_raw_dataset,
    iter_raw_chunks,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fast/read-only walkthrough of the canonical raw dataset boundary."
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help="Explicit path to card_transaction.v1.csv outside research/.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Also verify SHA-256 and full row count; this scans the whole dataset.",
    )
    parser.add_argument(
        "--preview-rows",
        type=int,
        default=3,
    )
    args = parser.parse_args()

    if args.preview_rows <= 0:
        raise ValueError("--preview-rows must be a positive integer.")

    print("=" * 92)
    print("01 — DATASET WALKTHROUGH")
    print("=" * 92)

    print("\n[1] Locked canonical identity")
    print(" - filename:", CANONICAL_RAW_DATASET.filename)
    print(" - size bytes:", CANONICAL_RAW_DATASET.size_bytes)
    print(" - SHA-256:", CANONICAL_RAW_DATASET.sha256)
    print(" - rows:", CANONICAL_RAW_DATASET.row_count)
    print(" - columns:", len(CANONICAL_RAW_DATASET.columns))

    inspection = inspect_raw_dataset(
        args.data_path,
        identity=CANONICAL_RAW_DATASET,
        project_root=PROJECT_ROOT,
        verify_sha256=args.full,
        verify_row_count=args.full,
    )

    print("\n[2] Inspection")
    print(" - path:", inspection.path)
    print(" - mode:", "FULL" if args.full else "FAST")
    print(" - filename matches:", inspection.filename_matches)
    print(" - size matches:", inspection.size_matches)
    print(" - schema/order matches:", inspection.schema_matches)
    print(" - SHA-256 matches:", inspection.sha256_matches)
    print(" - row-count matches:", inspection.row_count_matches)

    passed = (
        inspection.full_identity_match
        if args.full
        else (
            inspection.filename_matches
            and inspection.size_matches
            and inspection.schema_matches
        )
    )
    if not passed:
        raise RuntimeError("Dataset inspection did not match canonical contract.")

    print("\n[3] Small preview through canonical iterator")
    first_chunk = next(
        iter_raw_chunks(
            args.data_path,
            chunksize=args.preview_rows,
            project_root=PROJECT_ROOT,
        )
    )
    cols = [
        "raw_row_id", "User", "Card", "Year", "Month", "Day",
        "Time", "Amount", "Use Chip", "Merchant Name",
    ]
    for row in first_chunk[cols].to_dict(orient="records"):
        print(" -", row)

    print("\n[4] Boundary")
    print(" - bundled raw dataset: NO")
    print(" - silent research fallback: NO")
    print(" - file write: NO")
    print(" - model/preprocessing access: NO")

    print("\n01 DATASET WALKTHROUGH: PASS")


if __name__ == "__main__":
    main()
