from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path


MODEL_SOURCE_REL = Path(
    "research/data/processed/m8_02_final_test_artifact_audit/"
    "m8_02_selected_rf_estimator.joblib"
)

PREPROCESSING_SOURCE_REL = Path(
    "research/data/processed/m8_02_final_test_artifact_audit/"
    "m8_02_w_short_preprocessing_state.json"
)

OFFICIAL_ROOT_REL = Path(
    "final_pipeline/artifacts/official"
)

MODEL_TARGET_REL = (
    OFFICIAL_ROOT_REL
    / "model/model.joblib"
)

PREPROCESSING_TARGET_REL = (
    OFFICIAL_ROOT_REL
    / "preprocessing/preprocessing_state.json"
)

MANIFEST_TARGET_REL = (
    OFFICIAL_ROOT_REL
    / "manifest/artifact_manifest.json"
)

EXPECTED_MODEL_SHA256 = (
    "61bdeeba5fd163cce8a5a9efa028a9cdc2f95c0796822a5222c0604c0302115f"
)

EXPECTED_PREPROCESSING_SHA256 = (
    "c1dc5486acdcd43f5f75fbd0a11bd7ad3a3b00761d50ae96b2b1fbe5b5d9ef98"
)

MODEL_ID = "RF-REF-100-GINI-SQRT-UNPRUNED-CW"
THRESHOLD = 0.50
THRESHOLD_COMPARATOR = ">"
POSITIVE_CLASS = 1
POSITIVE_CLASS_INDEX = 1
ENCODED_FEATURE_COUNT = 47
SEMANTIC_FEATURE_COUNT = 10


def detect_root() -> Path:
    root = Path.cwd().resolve()

    required = [
        root / "research",
        root / "final_pipeline",
        root / "application",
    ]

    if not all(path.is_dir() for path in required):
        raise RuntimeError(
            "Run this script from repository root."
        )

    return root


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    try:
        with path.open("rb") as file:
            for chunk in iter(
                lambda: file.read(1024 * 1024),
                b"",
            ):
                digest.update(chunk)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Missing file: {path}"
        ) from exc

    return digest.hexdigest()


def build_manifest() -> dict:
    return {
        "manifest_version": "1.0",
        "model": {
            "file": "model/model.joblib",
            "model_id": MODEL_ID,
            "sha256": EXPECTED_MODEL_SHA256,
            "classes": [0, 1],
            "positive_class": POSITIVE_CLASS,
            "positive_class_index": POSITIVE_CLASS_INDEX,
        },
        "preprocessing": {
            "file":
                "preprocessing/preprocessing_state.json",
            "sha256":
                EXPECTED_PREPROCESSING_SHA256,
            "semantic_feature_count":
                SEMANTIC_FEATURE_COUNT,
            "encoded_feature_count":
                ENCODED_FEATURE_COUNT,
            "matrix_format":
                "CSR",
            "matrix_dtype":
                "float32",
        },
        "screening": {
            "threshold":
                THRESHOLD,
            "threshold_comparator":
                THRESHOLD_COMPARATOR,
            "risk_score_interface":
                "predict_proba positive-class score",
        },
        "runtime_policy": {
            "official_artifacts_read_only":
                True,
            "silent_research_fallback":
                False,
            "training_may_overwrite_official":
                False,
        },
    }


def source_identity(root: Path) -> dict:
    model_source = (
        root / MODEL_SOURCE_REL
    )
    preprocessing_source = (
        root / PREPROCESSING_SOURCE_REL
    )

    if not model_source.is_file():
        raise FileNotFoundError(
            model_source
        )

    if not preprocessing_source.is_file():
        raise FileNotFoundError(
            preprocessing_source
        )

    model_sha = sha256_file(
        model_source
    )
    preprocessing_sha = sha256_file(
        preprocessing_source
    )

    return {
        "model_source":
            model_source,
        "preprocessing_source":
            preprocessing_source,
        "model_sha":
            model_sha,
        "preprocessing_sha":
            preprocessing_sha,
        "model_ok":
            model_sha
            == EXPECTED_MODEL_SHA256,
        "preprocessing_ok":
            preprocessing_sha
            == EXPECTED_PREPROCESSING_SHA256,
    }


def target_state(root: Path) -> dict:
    paths = {
        "model":
            root / MODEL_TARGET_REL,
        "preprocessing":
            root / PREPROCESSING_TARGET_REL,
        "manifest":
            root / MANIFEST_TARGET_REL,
    }

    collisions = {
        name: path
        for name, path in paths.items()
        if path.exists()
    }

    return {
        "paths": paths,
        "collisions": collisions,
    }


def print_dry_run(
    root: Path,
    source: dict,
    targets: dict,
) -> None:
    print("=" * 92)
    print(
        "OFFICIAL ARTIFACT PROMOTION — DRY RUN"
    )
    print("=" * 92)
    print(
        "Project root:",
        root,
    )
    print(
        "Mode: READ-ONLY PRECHECK"
    )

    print("\n[1] Source artifact identity")
    print(
        " - model source:",
        MODEL_SOURCE_REL,
    )
    print(
        " - model SHA:",
        source["model_sha"],
    )
    print(
        " - model SHA match:",
        source["model_ok"],
    )
    print(
        " - preprocessing source:",
        PREPROCESSING_SOURCE_REL,
    )
    print(
        " - preprocessing SHA:",
        source[
            "preprocessing_sha"
        ],
    )
    print(
        " - preprocessing SHA match:",
        source[
            "preprocessing_ok"
        ],
    )

    print("\n[2] Planned official targets")
    for name, path in (
        targets["paths"].items()
    ):
        print(
            f" - {name}: "
            f"{path.relative_to(root)}"
        )

    print("\n[3] Collision gate")
    print(
        " - collisions:",
        len(
            targets[
                "collisions"
            ]
        ),
    )
    for name, path in (
        targets[
            "collisions"
        ].items()
    ):
        print(
            f"   * {name}: "
            f"{path.relative_to(root)}"
        )

    print("\n[4] Promotion policy")
    print(
        " - model copy: EXACT BYTES"
    )
    print(
        " - preprocessing copy: EXACT BYTES"
    )
    print(
        " - target overwrite: FORBIDDEN"
    )
    print(
        " - research fallback after promotion: FORBIDDEN"
    )
    print(
        " - official artifact mutation by training: FORBIDDEN"
    )

    print("\n[5] Scientific/runtime boundary")
    print(
        " - model fit: NO"
    )
    print(
        " - preprocessing fit: NO"
    )
    print(
        " - predict: NO"
    )
    print(
        " - predict_proba: NO"
    )
    print(
        " - threshold change: NO"
    )
    print(
        " - model bytes changed: NO"
    )
    print(
        " - preprocessing bytes changed: NO"
    )

    passed = (
        source["model_ok"]
        and source[
            "preprocessing_ok"
        ]
        and not targets[
            "collisions"
        ]
    )

    print("\n[6] Decision")
    print(
        " - PRECHECK:",
        "PASS"
        if passed
        else "FAIL",
    )

    if passed:
        print(
            "\nDRY-RUN RESULT: PASS"
        )
        print(
            "Execute with:"
        )
        print(
            "python promote_official_artifacts.py --execute"
        )
    else:
        print(
            "\nDRY-RUN RESULT: STOP"
        )
        print(
            "No files were written."
        )

    print("=" * 92)


def verify_official(
    root: Path,
) -> dict:
    model_path = (
        root / MODEL_TARGET_REL
    )
    preprocessing_path = (
        root / PREPROCESSING_TARGET_REL
    )
    manifest_path = (
        root / MANIFEST_TARGET_REL
    )

    if not all(
        path.is_file()
        for path in [
            model_path,
            preprocessing_path,
            manifest_path,
        ]
    ):
        raise FileNotFoundError(
            "Official artifact set is incomplete."
        )

    model_sha = sha256_file(
        model_path
    )
    preprocessing_sha = sha256_file(
        preprocessing_path
    )

    manifest = json.loads(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )

    expected_manifest = (
        build_manifest()
    )

    src_path = (
        root / "final_pipeline/src"
    )

    sys.path.insert(
        0,
        str(src_path),
    )

    try:
        from fraud_screening.artifacts import (
            load_verified_model,
        )
        from fraud_screening.preprocessing import (
            load_frozen_preprocessor,
        )

        model = load_verified_model(
            model_path,
            expected_sha256=
                EXPECTED_MODEL_SHA256,
        )

        preprocessor = (
            load_frozen_preprocessor(
                preprocessing_path
            )
        )

        gates = {
            "G01_MODEL_TARGET_EXISTS":
                model_path.is_file(),
            "G02_PREPROCESSING_TARGET_EXISTS":
                preprocessing_path.is_file(),
            "G03_MANIFEST_TARGET_EXISTS":
                manifest_path.is_file(),
            "G04_MODEL_SHA_PRESERVED":
                model_sha
                == EXPECTED_MODEL_SHA256,
            "G05_PREPROCESSING_SHA_PRESERVED":
                preprocessing_sha
                == EXPECTED_PREPROCESSING_SHA256,
            "G06_MANIFEST_EXACT":
                manifest
                == expected_manifest,
            "G07_MODEL_LOAD_COMPATIBLE":
                (
                    model.model_id
                    == MODEL_ID
                    and model.classes
                    == (0, 1)
                    and model.positive_class_index
                    == POSITIVE_CLASS_INDEX
                ),
            "G08_PREPROCESSING_COMPATIBLE":
                (
                    len(
                        preprocessor.feature_names
                    )
                    == ENCODED_FEATURE_COUNT
                    and preprocessor.strategy
                    == "W_SHORT"
                    and preprocessor.fit_source
                    == "W_SHORT_TRAIN_ONLY"
                ),
        }

    finally:
        try:
            sys.path.remove(
                str(src_path)
            )
        except ValueError:
            pass

    return {
        "model_sha":
            model_sha,
        "preprocessing_sha":
            preprocessing_sha,
        "manifest":
            manifest,
        "gates":
            gates,
    }


def execute_promotion(
    root: Path,
    source: dict,
    targets: dict,
) -> None:
    if not (
        source["model_ok"]
        and source[
            "preprocessing_ok"
        ]
    ):
        raise RuntimeError(
            "Source identity check failed."
        )

    if targets[
        "collisions"
    ]:
        raise RuntimeError(
            "Official target collision detected."
        )

    created_files: list[Path] = []
    created_dirs: list[Path] = []

    official_root = (
        root / OFFICIAL_ROOT_REL
    )

    target_paths = targets[
        "paths"
    ]

    try:
        for target in [
            target_paths["model"],
            target_paths[
                "preprocessing"
            ],
            target_paths[
                "manifest"
            ],
        ]:
            parent = target.parent

            missing_chain = []

            current = parent

            while (
                not current.exists()
                and
                current != root
            ):
                missing_chain.append(
                    current
                )
                current = (
                    current.parent
                )

            for directory in reversed(
                missing_chain
            ):
                directory.mkdir()
                created_dirs.append(
                    directory
                )

        shutil.copyfile(
            source["model_source"],
            target_paths["model"],
        )
        created_files.append(
            target_paths["model"]
        )

        shutil.copyfile(
            source[
                "preprocessing_source"
            ],
            target_paths[
                "preprocessing"
            ],
        )
        created_files.append(
            target_paths[
                "preprocessing"
            ]
        )

        copied_model_sha = sha256_file(
            target_paths["model"]
        )

        copied_preprocessing_sha = (
            sha256_file(
                target_paths[
                    "preprocessing"
                ]
            )
        )

        if (
            copied_model_sha
            != EXPECTED_MODEL_SHA256
        ):
            raise RuntimeError(
                "Copied model SHA mismatch."
            )

        if (
            copied_preprocessing_sha
            != EXPECTED_PREPROCESSING_SHA256
        ):
            raise RuntimeError(
                "Copied preprocessing SHA mismatch."
            )

        manifest = (
            build_manifest()
        )

        target_paths[
            "manifest"
        ].write_text(
            json.dumps(
                manifest,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        created_files.append(
            target_paths[
                "manifest"
            ]
        )

        verification = (
            verify_official(
                root
            )
        )

        if not all(
            verification[
                "gates"
            ].values()
        ):
            raise RuntimeError(
                "Official verification gates failed."
            )

    except Exception:
        for path in reversed(
            created_files
        ):
            try:
                path.unlink()
            except OSError:
                pass

        for directory in reversed(
            created_dirs
        ):
            try:
                directory.rmdir()
            except OSError:
                pass

        raise

    print("=" * 92)
    print(
        "OFFICIAL ARTIFACT PROMOTION: PASS"
    )
    print("=" * 92)

    print("\n[1] Official files")
    for name, path in (
        target_paths.items()
    ):
        print(
            f" - {name}: "
            f"{path.relative_to(root)}"
        )

    print("\n[2] Preserved fingerprints")
    print(
        " - model:",
        verification[
            "model_sha"
        ],
    )
    print(
        " - preprocessing:",
        verification[
            "preprocessing_sha"
        ],
    )

    print("\n[3] Verification gates")
    for name, passed in (
        verification[
            "gates"
        ].items()
    ):
        print(
            f" - {name}: "
            f"{'PASS' if passed else 'FAIL'}"
        )

    print("\n[4] Safety attestations")
    print(
        " - model fit: NO"
    )
    print(
        " - preprocessing fit: NO"
    )
    print(
        " - predict: NO"
    )
    print(
        " - predict_proba: NO"
    )
    print(
        " - model bytes changed: NO"
    )
    print(
        " - preprocessing bytes changed: NO"
    )
    print(
        " - official overwrite: NO"
    )

    print(
        "\nPROMOTION RESULT: PASS"
    )
    print(
        "Official artifacts are now available "
        "under final_pipeline/artifacts/official/."
    )
    print("=" * 92)


def print_verification(
    root: Path,
) -> None:
    verification = (
        verify_official(
            root
        )
    )

    print("=" * 92)
    print(
        "OFFICIAL ARTIFACT SET — READ-ONLY VERIFICATION"
    )
    print("=" * 92)

    print(
        " - model SHA:",
        verification[
            "model_sha"
        ],
    )
    print(
        " - preprocessing SHA:",
        verification[
            "preprocessing_sha"
        ],
    )

    print("\nVerification gates:")
    for name, passed in (
        verification[
            "gates"
        ].items()
    ):
        print(
            f" - {name}: "
            f"{'PASS' if passed else 'FAIL'}"
        )

    passed = all(
        verification[
            "gates"
        ].values()
    )

    print(
        "\nVERIFICATION RESULT:",
        "PASS"
        if passed
        else "FAIL",
    )

    print("=" * 92)

    if not passed:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()

    group = (
        parser.add_mutually_exclusive_group()
    )

    group.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Promote exact frozen artifacts "
            "to the official product path."
        ),
    )

    group.add_argument(
        "--verify-official",
        action="store_true",
        help=(
            "Read-only verification of the "
            "already-promoted official artifact set."
        ),
    )

    args = parser.parse_args()

    root = detect_root()

    if args.verify_official:
        print_verification(
            root
        )
        return

    source = source_identity(
        root
    )
    targets = target_state(
        root
    )

    if not args.execute:
        print_dry_run(
            root,
            source,
            targets,
        )
        return

    execute_promotion(
        root,
        source,
        targets,
    )


if __name__ == "__main__":
    main()
