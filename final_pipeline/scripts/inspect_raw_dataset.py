#!/usr/bin/env python3
"""Inspect a raw dataset against the locked final-pipeline artifact contract."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

SRC_ROOT = (
    PROJECT_ROOT
    / "final_pipeline"
    / "src"
)

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(SRC_ROOT),
    )

from fraud_screening.replay import (  # noqa: E402
    CANONICAL_RAW_DATASET,
    inspect_raw_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-path",
        required=True,
        help=(
            "Path to card_transaction.v1.csv outside research/."
        ),
    )

    parser.add_argument(
        "--fast",
        action="store_true",
        help=(
            "Check existence, filename, size and schema only; "
            "skip full SHA-256 and row-count scan."
        ),
    )

    parser.add_argument(
        "--chunksize",
        type=int,
        default=250_000,
    )

    args = parser.parse_args()

    inspection = inspect_raw_dataset(
        args.data_path,
        identity=CANONICAL_RAW_DATASET,
        project_root=PROJECT_ROOT,
        verify_sha256=not args.fast,
        verify_row_count=not args.fast,
        chunksize=args.chunksize,
    )

    print("=" * 92)
    print(
        "RAW DATASET REPLAY — READ-ONLY INSPECTION"
    )
    print("=" * 92)
    print(
        "Path:",
        inspection.path,
    )
    print(
        "Mode:",
        (
            "FAST — HEADER/SIZE ONLY"
            if args.fast
            else "FULL — SHA256 + ROW COUNT"
        ),
    )

    print("\n[1] Artifact identity")
    print(
        " - expected filename:",
        CANONICAL_RAW_DATASET.filename,
    )
    print(
        " - filename matches:",
        inspection.filename_matches,
    )
    print(
        " - expected size:",
        CANONICAL_RAW_DATASET.size_bytes,
    )
    print(
        " - actual size:",
        inspection.size_bytes,
    )
    print(
        " - size matches:",
        inspection.size_matches,
    )

    print("\n[2] Raw schema")
    print(
        " - expected columns:",
        len(CANONICAL_RAW_DATASET.columns),
    )
    print(
        " - actual columns:",
        len(inspection.columns),
    )
    print(
        " - exact order matches:",
        inspection.schema_matches,
    )

    print("\n[3] Full identity checks")
    print(
        " - SHA256:",
        inspection.sha256,
    )
    print(
        " - SHA256 matches:",
        inspection.sha256_matches,
    )
    print(
        " - row count:",
        inspection.row_count,
    )
    print(
        " - row count matches:",
        inspection.row_count_matches,
    )

    print("\n[4] Side-effect boundary")
    print(" - file write: NO")
    print(" - model.fit(): NO")
    print(" - preprocessing fit: NO")
    print(" - predict()/predict_proba(): NO")
    print(" - official artifact access: NO")
    print(" - research dependency: NO")

    if args.fast:
        fast_pass = (
            inspection.filename_matches
            and inspection.size_matches
            and inspection.schema_matches
        )

        print(
            "\nFAST INSPECTION RESULT:",
            "PASS" if fast_pass else "FAIL",
        )

        if not fast_pass:
            raise SystemExit(1)

        print("=" * 92)
        return

    print(
        "\nFULL INSPECTION RESULT:",
        (
            "PASS"
            if inspection.full_identity_match
            else "FAIL"
        ),
    )

    if not inspection.full_identity_match:
        raise SystemExit(1)

    print("=" * 92)


if __name__ == "__main__":
    main()
