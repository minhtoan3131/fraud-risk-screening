#!/usr/bin/env python3
"""Guarded command for creating a new W_SHORT rebuild artifact set."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys


PROJECT_ROOT = (
    Path(
        __file__
    )
    .resolve()
    .parents[2]
)

SRC_ROOT = (
    PROJECT_ROOT
    / "final_pipeline"
    / "src"
)

if str(
    SRC_ROOT
) not in sys.path:
    sys.path.insert(
        0,
        str(
            SRC_ROOT
        ),
    )

from fraud_screening.training import (  # noqa: E402
    CANONICAL_W_SHORT_PROFILE,
    create_rebuild_output_paths,
    load_training_bundle,
    run_rebuild_training,
)


OFFICIAL_MODEL = (
    PROJECT_ROOT
    / "final_pipeline"
    / "artifacts"
    / "official"
    / "model"
    / "model.joblib"
)

OFFICIAL_PREPROCESSING = (
    PROJECT_ROOT
    / "final_pipeline"
    / "artifacts"
    / "official"
    / "preprocessing"
    / "preprocessing_state.json"
)


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as handle:
        for chunk in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(
                chunk
            )

    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a canonical W_SHORT training bundle and, "
            "only with --execute, create a new rebuild artifact set."
        )
    )

    parser.add_argument(
        "--bundle-dir",
        required=True,
        help=(
            "Canonical training bundle directory. "
            "research/ paths are rejected."
        ),
    )

    parser.add_argument(
        "--run-id",
        required=True,
        help=(
            "Unique rebuild run id."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=50_000,
    )

    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Fit preprocessing and model, then write a new rebuild run."
        ),
    )

    args = parser.parse_args()

    if not OFFICIAL_MODEL.is_file():
        raise FileNotFoundError(
            "Official model is missing."
        )

    if not OFFICIAL_PREPROCESSING.is_file():
        raise FileNotFoundError(
            "Official preprocessing is missing."
        )

    if (
        args.batch_size
        <= 0
    ):
        raise ValueError(
            "--batch-size must be positive."
        )

    # Fail before bundle scan if run id/path collides.
    planned = create_rebuild_output_paths(
        PROJECT_ROOT,
        args.run_id,
        create=False,
    )

    model_sha_before = (
        sha256_file(
            OFFICIAL_MODEL
        )
    )

    preprocessing_sha_before = (
        sha256_file(
            OFFICIAL_PREPROCESSING
        )
    )

    bundle = load_training_bundle(
        args.bundle_dir,
        profile=
            CANONICAL_W_SHORT_PROFILE,
        project_root=
            PROJECT_ROOT,
    )

    print("=" * 92)
    print(
        "CANONICAL W_SHORT REBUILD TRAINING"
    )
    print("=" * 92)
    print(
        "Mode:",
        (
            "EXECUTE — LEARNED STATE + NEW REBUILD OUTPUT"
            if args.execute
            else "DRY RUN — BUNDLE VALIDATION ONLY"
        ),
    )

    print("\n[1] Input bundle")
    print(
        " - directory:",
        bundle.bundle_dir,
    )
    print(
        " - profile:",
        bundle.profile.name,
    )
    print(
        " - rows:",
        f"{bundle.row_count:,}",
    )
    print(
        " - fraud rows:",
        f"{bundle.fraud_rows:,}",
    )
    print(
        " - observed timestamp min:",
        bundle.timestamps.min(),
    )
    print(
        " - observed timestamp max:",
        bundle.timestamps.max(),
    )

    print("\n[2] Locked W_SHORT contract")
    print(
        " - start inclusive: 2018-01-01"
    )
    print(
        " - end exclusive: 2019-01-01"
    )
    print(
        " - expected rows: 1,721,615"
    )
    print(
        " - expected fraud rows: 2,491"
    )
    print(
        " - semantic width: 10"
    )
    print(
        " - encoded width after fitted preprocessing: 47"
    )

    print("\n[3] Planned output")
    print(
        " - run directory:",
        planned.run_dir.relative_to(
            PROJECT_ROOT
        ),
    )
    print(
        " - overwrite existing run: FORBIDDEN"
    )
    print(
        " - official promotion: NO"
    )
    print(
        " - M8 metric inheritance: NO"
    )

    print("\n[4] Operation boundary")
    print(
        " - preprocessing fit:",
        "YES" if args.execute else "NO",
    )
    print(
        " - model.fit():",
        "YES" if args.execute else "NO",
    )
    print(
        " - predict(): NO"
    )
    print(
        " - predict_proba(): NO"
    )
    print(
        " - rebuild artifact write:",
        "YES" if args.execute else "NO",
    )
    print(
        " - official artifact write: NO"
    )
    print(
        " - research dependency: NO"
    )

    if not args.execute:
        print(
            "\nDRY-RUN RESULT: PASS"
        )
        print(
            "Bundle is canonical and no learned state was created."
        )
        print(
            "Execute only when a real rebuild is intentionally required."
        )
        print("=" * 92)
        return

    result = run_rebuild_training(
        PROJECT_ROOT,
        bundle.bundle_dir,
        args.run_id,
        batch_size=
            args.batch_size,
        profile=
            CANONICAL_W_SHORT_PROFILE,
    )

    model_sha_after = (
        sha256_file(
            OFFICIAL_MODEL
        )
    )

    preprocessing_sha_after = (
        sha256_file(
            OFFICIAL_PREPROCESSING
        )
    )

    if (
        model_sha_after
        != model_sha_before
    ):
        raise RuntimeError(
            "Official model changed during rebuild."
        )

    if (
        preprocessing_sha_after
        != preprocessing_sha_before
    ):
        raise RuntimeError(
            "Official preprocessing changed during rebuild."
        )

    print("\n[5] Rebuild result")
    print(
        " - run directory:",
        result.artifact_set.paths.run_dir.relative_to(
            PROJECT_ROOT
        ),
    )
    print(
        " - model SHA:",
        result.artifact_set.model_sha256,
    )
    print(
        " - preprocessing SHA:",
        result.artifact_set.preprocessing_sha256,
    )
    print(
        " - model identity status:",
        result.model_fit.artifact_identity_status,
    )
    print(
        " - evaluation status:",
        result.model_fit.evaluation_status,
    )

    print("\n[6] Official immutability")
    print(
        " - model SHA unchanged:",
        True,
    )
    print(
        " - preprocessing SHA unchanged:",
        True,
    )

    print(
        "\nREBUILD RESULT: PASS"
    )
    print(
        "The rebuild artifact set is NOT an automatic official replacement."
    )
    print("=" * 92)


if __name__ == "__main__":
    main()
