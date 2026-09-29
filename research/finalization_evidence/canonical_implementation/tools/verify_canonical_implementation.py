from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


EXPECTED_TEST_COUNT = 92

EXPECTED_SEMANTIC_FEATURE_ORDER = (
    "amount_numeric",
    "time_since_previous_transaction_min",
    "transactions_last_1h",
    "amount_minus_previous_mean",
    "is_new_merchant",
    "has_prior_card_history",
    "transaction_mode",
    "location_state",
    "hour_of_day",
    "day_of_week",
)

EXPECTED_TOOL_FILES = (
    "bootstrap_data_boundary.py",
    "bootstrap_transaction_features.py",
    "bootstrap_history_features.py",
    "bootstrap_semantic_features.py",
    "inspect_preprocessing_state.py",
    "bootstrap_frozen_preprocessing.py",
    "bootstrap_model_artifact_loader.py",
    "bootstrap_inference_primitives.py",
)

EVIDENCE_REL = Path(
    "research/finalization_evidence/canonical_implementation"
)

TOOLS_TARGET_REL = EVIDENCE_REL / "tools"

REGISTRY_REL = (
    EVIDENCE_REL
    / "implementation_verification_registry.json"
)

FINALIZER_ARCHIVE_REL = (
    TOOLS_TARGET_REL
    / "verify_canonical_implementation.py"
)

PRODUCT_TEXT_EXTENSIONS = {
    ".py",
    ".md",
    ".json",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
    ".ini",
    ".cfg",
}

PROGRESS_PATTERNS = {
    "milestone_token":
        re.compile(r"(?i)\bM\d+(?:\.\d+)?\b"),
    "milestone_word":
        re.compile(r"(?i)\bmilestone\b"),
    "canon_label":
        re.compile(r"(?i)\bCANON(?:[-\s_:]|$)"),
    "gate_language":
        re.compile(
            r"(?i)\b(?:final\s+gate|technical\s+gate|gate\s+status)\b"
        ),
    "authorization_language":
        re.compile(
            r"(?i)\b(?:authorized|authorization|not\s+authorized)\b"
        ),
    "handoff_language":
        re.compile(r"(?i)\bhandoff\b"),
}


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

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def product_manifest(root: Path) -> dict[str, dict]:
    result: dict[str, dict] = {}

    include_roots = [
        root / "final_pipeline",
        root / "application",
        root / "README.md",
        root / ".gitignore",
    ]

    files: list[Path] = []

    for item in include_roots:
        if item.is_file():
            files.append(item)
        elif item.is_dir():
            files.extend(
                path
                for path in item.rglob("*")
                if path.is_file()
                and "__pycache__" not in path.parts
                and ".pytest_cache" not in path.parts
            )

    for path in sorted(set(files)):
        rel = str(path.relative_to(root))

        result[rel] = {
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    return result


def scan_progress_surface(root: Path) -> list[dict]:
    hits: list[dict] = []

    zones = [
        root / "final_pipeline",
        root / "application",
    ]

    for zone in zones:
        for path in sorted(zone.rglob("*")):
            if "__pycache__" in path.parts:
                continue

            rel = str(path.relative_to(root))

            for category, pattern in PROGRESS_PATTERNS.items():
                if pattern.search(rel):
                    hits.append(
                        {
                            "path": rel,
                            "where": "path",
                            "category": category,
                        }
                    )

            if (
                not path.is_file()
                or path.suffix.lower()
                not in PRODUCT_TEXT_EXTENSIONS
            ):
                continue

            if path.stat().st_size > 5 * 1024 * 1024:
                continue

            text = path.read_text(
                encoding="utf-8",
                errors="replace",
            )

            for category, pattern in PROGRESS_PATTERNS.items():
                if pattern.search(text):
                    hits.append(
                        {
                            "path": rel,
                            "where": "content",
                            "category": category,
                        }
                    )

    return hits


def scan_source_boundary(root: Path) -> dict:
    source_root = (
        root
        / "final_pipeline/src/fraud_screening"
    )

    source_files = sorted(
        path
        for path in source_root.rglob("*.py")
        if path.is_file()
    )

    research_refs = []
    fit_calls = []
    predict_calls = []
    predict_proba_calls = []

    for path in source_files:
        rel = str(path.relative_to(root))
        text = path.read_text(
            encoding="utf-8",
            errors="replace",
        )

        if re.search(
            r"(?i)(?:['\"]research(?:/|['\"])|Path\(\s*['\"]research['\"]\s*\))",
            text,
        ):
            research_refs.append(rel)

        if re.search(
            r"\.(?:fit|partial_fit)\s*\(",
            text,
        ):
            fit_calls.append(rel)

        if re.search(
            r"\.predict\s*\(",
            text,
        ):
            predict_calls.append(rel)

        count = len(
            re.findall(
                r"\.predict_proba\s*\(",
                text,
            )
        )

        if count:
            predict_proba_calls.append(
                {
                    "path": rel,
                    "count": count,
                }
            )

    return {
        "research_refs": research_refs,
        "fit_calls": fit_calls,
        "predict_calls": predict_calls,
        "predict_proba_calls":
            predict_proba_calls,
    }


def run_unit_tests(root: Path) -> dict:
    env = os.environ.copy()

    source_path = str(
        root / "final_pipeline/src"
    )

    existing = env.get("PYTHONPATH")

    env["PYTHONPATH"] = (
        source_path
        if not existing
        else source_path
        + os.pathsep
        + existing
    )

    command = [
        sys.executable,
        "-m",
        "unittest",
        "discover",
        "-s",
        "final_pipeline/tests",
        "-v",
    ]

    completed = subprocess.run(
        command,
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
    )

    output = (
        completed.stdout
        + completed.stderr
    )

    match = re.search(
        r"Ran\s+(\d+)\s+tests?",
        output,
    )

    test_count = (
        int(match.group(1))
        if match
        else None
    )

    return {
        "command":
            " ".join(command),
        "returncode":
            completed.returncode,
        "test_count":
            test_count,
        "ok":
            completed.returncode == 0
            and "\nOK" in output,
        "output":
            output,
    }


def import_contract_checks(root: Path) -> dict:
    source_path = (
        root / "final_pipeline/src"
    )

    sys.path.insert(
        0,
        str(source_path),
    )

    try:
        import numpy as np

        from fraud_screening.artifacts import (
            EXPECTED_MODEL_PARAMETERS,
            MODEL_ID,
            POSITIVE_CLASS,
        )
        from fraud_screening.features import (
            BOOLEAN_FEATURES,
            CATEGORICAL_FEATURES,
            NUMERIC_FEATURES,
            SEMANTIC_FEATURE_ORDER,
        )
        from fraud_screening.inference import (
            DEFAULT_THRESHOLD,
            apply_screening_threshold,
        )
        from fraud_screening.preprocessing.frozen import (
            EXPECTED_FEATURE_COUNT,
            EXPECTED_MATRIX_DTYPE,
            EXPECTED_MATRIX_FORMAT,
        )

        boundary = (
            apply_screening_threshold(
                np.asarray(
                    [0.49, 0.50, 0.51],
                    dtype=np.float32,
                )
            )
            .tolist()
        )

        return {
            "semantic_order_exact":
                tuple(
                    SEMANTIC_FEATURE_ORDER
                )
                ==
                EXPECTED_SEMANTIC_FEATURE_ORDER,
            "numeric_count":
                len(NUMERIC_FEATURES),
            "boolean_count":
                len(BOOLEAN_FEATURES),
            "categorical_count":
                len(CATEGORICAL_FEATURES),
            "encoded_feature_count":
                EXPECTED_FEATURE_COUNT,
            "matrix_format":
                EXPECTED_MATRIX_FORMAT,
            "matrix_dtype":
                EXPECTED_MATRIX_DTYPE,
            "model_id":
                MODEL_ID,
            "positive_class":
                POSITIVE_CLASS,
            "model_parameter_count":
                len(
                    EXPECTED_MODEL_PARAMETERS
                ),
            "threshold":
                DEFAULT_THRESHOLD,
            "strict_boundary":
                boundary,
        }

    finally:
        try:
            sys.path.remove(
                str(source_path)
            )
        except ValueError:
            pass


def expected_product_files(root: Path) -> list[str]:
    required = [
        "final_pipeline/pyproject.toml",
        "final_pipeline/README.md",
        "final_pipeline/src/fraud_screening/__init__.py",
        "final_pipeline/src/fraud_screening/errors.py",
        "final_pipeline/src/fraud_screening/data/schema.py",
        "final_pipeline/src/fraud_screening/data/models.py",
        "final_pipeline/src/fraud_screening/data/parsing.py",
        "final_pipeline/src/fraud_screening/features/transaction.py",
        "final_pipeline/src/fraud_screening/features/history.py",
        "final_pipeline/src/fraud_screening/features/semantic.py",
        "final_pipeline/src/fraud_screening/preprocessing/frozen.py",
        "final_pipeline/src/fraud_screening/artifacts/model_loader.py",
        "final_pipeline/src/fraud_screening/inference/scoring.py",
        "final_pipeline/tests/unit/test_data_boundary.py",
        "final_pipeline/tests/unit/test_transaction_features.py",
        "final_pipeline/tests/unit/test_history_features.py",
        "final_pipeline/tests/unit/test_semantic_features.py",
        "final_pipeline/tests/unit/test_frozen_preprocessing.py",
        "final_pipeline/tests/unit/test_model_artifact_loader.py",
        "final_pipeline/tests/unit/test_inference_primitives.py",
        "application/README.md",
    ]

    return [
        relative
        for relative in required
        if not (root / relative).is_file()
    ]


def build_review(root: Path) -> dict:
    missing_product_files = (
        expected_product_files(root)
    )

    missing_tools = [
        filename
        for filename in EXPECTED_TOOL_FILES
        if not (root / filename).is_file()
    ]

    evidence_targets = [
        root / EVIDENCE_REL,
        root / TOOLS_TARGET_REL,
        root / REGISTRY_REL,
        root / FINALIZER_ARCHIVE_REL,
    ]

    target_collisions = [
        str(path.relative_to(root))
        for path in evidence_targets
        if path.exists()
    ]

    surface_hits = scan_progress_surface(
        root
    )

    source_boundary = scan_source_boundary(
        root
    )

    tests = run_unit_tests(
        root
    )

    contracts = import_contract_checks(
        root
    )

    expected_predict_proba = [
        {
            "path":
                "final_pipeline/src/fraud_screening/inference/scoring.py",
            "count":
                1,
        }
    ]

    gates = {
        "G01_PRODUCT_FILES_PRESENT":
            len(missing_product_files)
            == 0,
        "G02_UNIT_SUITE_PASS":
            bool(tests["ok"]),
        "G03_EXACT_92_TESTS":
            tests["test_count"]
            == EXPECTED_TEST_COUNT,
        "G04_PRESENTATION_SURFACE_CLEAN":
            len(surface_hits) == 0,
        "G05_NO_RUNTIME_RESEARCH_REFERENCE":
            len(
                source_boundary[
                    "research_refs"
                ]
            )
            == 0,
        "G06_NO_FIT_IN_RUNTIME_SOURCE":
            len(
                source_boundary[
                    "fit_calls"
                ]
            )
            == 0,
        "G07_NO_PREDICT_IN_RUNTIME_SOURCE":
            len(
                source_boundary[
                    "predict_calls"
                ]
            )
            == 0,
        "G08_PREDICT_PROBA_SINGLE_PRIMITIVE":
            source_boundary[
                "predict_proba_calls"
            ]
            == expected_predict_proba,
        "G09_EXACT_10_SEMANTIC_ORDER":
            contracts[
                "semantic_order_exact"
            ],
        "G10_ROLE_COUNTS_4_2_4":
            (
                contracts[
                    "numeric_count"
                ]
                == 4
                and
                contracts[
                    "boolean_count"
                ]
                == 2
                and
                contracts[
                    "categorical_count"
                ]
                == 4
            ),
        "G11_FROZEN_47_CSR_FLOAT32":
            (
                contracts[
                    "encoded_feature_count"
                ]
                == 47
                and
                contracts[
                    "matrix_format"
                ]
                == "CSR"
                and
                contracts[
                    "matrix_dtype"
                ]
                == "float32"
            ),
        "G12_MODEL_ID_AND_POSITIVE_CLASS":
            (
                contracts[
                    "model_id"
                ]
                ==
                "RF-REF-100-GINI-SQRT-UNPRUNED-CW"
                and
                contracts[
                    "positive_class"
                ]
                == 1
                and
                contracts[
                    "model_parameter_count"
                ]
                == 12
            ),
        "G13_THRESHOLD_STRICT_GT":
            (
                contracts[
                    "threshold"
                ]
                == 0.50
                and
                contracts[
                    "strict_boundary"
                ]
                == [0, 0, 1]
            ),
        "G14_TOOLING_COMPLETE_FOR_ARCHIVE":
            len(missing_tools) == 0,
        "G15_EVIDENCE_TARGET_EMPTY":
            len(target_collisions)
            == 0,
    }

    return {
        "missing_product_files":
            missing_product_files,
        "missing_tools":
            missing_tools,
        "target_collisions":
            target_collisions,
        "surface_hits":
            surface_hits,
        "source_boundary":
            source_boundary,
        "tests":
            tests,
        "contracts":
            contracts,
        "gates":
            gates,
    }


def print_review(
    root: Path,
    review: dict,
) -> None:
    print("=" * 94)
    print(
        "CANONICAL IMPLEMENTATION — FINAL INTEGRATION REVIEW"
    )
    print("=" * 94)
    print("Project root:", root)
    print(
        "Mode: READ-ONLY REVIEW "
        "(unit tests execute; product files are not modified)"
    )

    print("\n[1] Product inventory")
    print(
        " - missing required product files:",
        len(
            review[
                "missing_product_files"
            ]
        ),
    )
    for item in review[
        "missing_product_files"
    ]:
        print("   *", item)

    print("\n[2] Unit suite")
    print(
        " - return code:",
        review["tests"][
            "returncode"
        ],
    )
    print(
        " - test count:",
        review["tests"][
            "test_count"
        ],
    )
    print(
        " - status:",
        "PASS"
        if review["tests"]["ok"]
        else "FAIL",
    )

    print("\n[3] Presentation surface")
    print(
        " - progress/governance hits:",
        len(
            review[
                "surface_hits"
            ]
        ),
    )
    for item in review[
        "surface_hits"
    ]:
        print(
            "   *",
            item,
        )

    print("\n[4] Runtime source boundary")
    boundary = review[
        "source_boundary"
    ]
    print(
        " - research refs:",
        boundary[
            "research_refs"
        ],
    )
    print(
        " - fit calls:",
        boundary[
            "fit_calls"
        ],
    )
    print(
        " - predict calls:",
        boundary[
            "predict_calls"
        ],
    )
    print(
        " - predict_proba calls:",
        boundary[
            "predict_proba_calls"
        ],
    )

    print("\n[5] Locked contracts")
    for key, value in (
        review["contracts"].items()
    ):
        print(
            f" - {key}: {value}"
        )

    print("\n[6] Tool archive readiness")
    print(
        " - missing bootstrap/inspection tools:",
        len(
            review[
                "missing_tools"
            ]
        ),
    )
    for item in review[
        "missing_tools"
    ]:
        print("   *", item)

    print(
        " - target collisions:",
        len(
            review[
                "target_collisions"
            ]
        ),
    )
    for item in review[
        "target_collisions"
    ]:
        print("   *", item)

    print("\n[7] Gates")
    for name, passed in (
        review["gates"].items()
    ):
        print(
            f" - {name}: "
            f"{'PASS' if passed else 'FAIL'}"
        )

    passed = all(
        review["gates"].values()
    )

    print("\n[8] Decision")
    print(
        " - FINAL INTEGRATION REVIEW:",
        "PASS" if passed else "FAIL",
    )

    print("=" * 94)


def execute_finalize(
    root: Path,
    review: dict,
) -> None:
    if not all(
        review["gates"].values()
    ):
        raise RuntimeError(
            "STOP: review gates are not all PASS."
        )

    evidence_dir = (
        root / EVIDENCE_REL
    )
    tools_dir = (
        root / TOOLS_TARGET_REL
    )

    evidence_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    tools_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    moved: list[
        tuple[Path, Path]
    ] = []

    try:
        tool_manifest_before = {}

        for filename in (
            EXPECTED_TOOL_FILES
        ):
            source = root / filename
            target = (
                tools_dir / filename
            )

            tool_manifest_before[
                filename
            ] = {
                "size_bytes":
                    source.stat().st_size,
                "sha256":
                    sha256_file(
                        source
                    ),
            }

            shutil.move(
                str(source),
                str(target),
            )

            moved.append(
                (
                    source,
                    target,
                )
            )

        finalizer_source = Path(
            __file__
        ).resolve()

        finalizer_target = (
            root
            / FINALIZER_ARCHIVE_REL
        )

        shutil.copy2(
            finalizer_source,
            finalizer_target,
        )

        tool_manifest_after = {}

        for filename in (
            EXPECTED_TOOL_FILES
        ):
            target = (
                tools_dir / filename
            )

            tool_manifest_after[
                filename
            ] = {
                "size_bytes":
                    target.stat().st_size,
                "sha256":
                    sha256_file(
                        target
                    ),
            }

        if (
            tool_manifest_before
            !=
            tool_manifest_after
        ):
            raise RuntimeError(
                "Archived tool bytes changed."
            )

        registry = {
            "operation":
                "canonical_implementation_verification",
            "created_at_utc":
                datetime.now(
                    timezone.utc
                ).isoformat(),
            "status":
                "PASS",
            "product_manifest":
                product_manifest(
                    root
                ),
            "unit_test_summary": {
                "expected_count":
                    EXPECTED_TEST_COUNT,
                "observed_count":
                    review[
                        "tests"
                    ][
                        "test_count"
                    ],
                "status":
                    "PASS",
            },
            "surface_hygiene": {
                "progress_hits":
                    review[
                        "surface_hits"
                    ],
                "status":
                    "PASS",
            },
            "runtime_boundary":
                review[
                    "source_boundary"
                ],
            "locked_contracts":
                review[
                    "contracts"
                ],
            "gates":
                review[
                    "gates"
                ],
            "archived_tools":
                tool_manifest_after,
            "attestations": {
                "model_fit_performed":
                    False,
                "preprocessing_fit_performed":
                    False,
                "threshold_retuned":
                    False,
                "artifact_promoted":
                    False,
                "application_modified":
                    False,
            },
        }

        registry_path = (
            root / REGISTRY_REL
        )

        registry_path.write_text(
            json.dumps(
                registry,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    except Exception:
        for source, target in reversed(
            moved
        ):
            if (
                target.exists()
                and not source.exists()
            ):
                shutil.move(
                    str(target),
                    str(source),
                )

        if evidence_dir.exists():
            shutil.rmtree(
                evidence_dir,
                ignore_errors=True,
            )

        raise

    print("\n" + "=" * 94)
    print(
        "CANONICAL IMPLEMENTATION FINALIZATION: PASS"
    )
    print("=" * 94)
    print(
        "Evidence registry:"
    )
    print(
        " -",
        REGISTRY_REL,
    )
    print(
        "Bootstrap/inspection tooling archived:"
    )
    print(
        " -",
        TOOLS_TARGET_REL,
    )
    print(
        "Product source modified: NO"
    )
    print(
        "Artifact promotion performed: NO"
    )
    print(
        "Application modified: NO"
    )
    print()
    print(
        "One root cleanup remains:"
    )
    print(
        " rm verify_canonical_implementation.py"
    )
    print(
        "Run it only after reading this PASS output."
    )
    print("=" * 94)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Archive completed bootstrap tooling "
            "and write verification evidence."
        ),
    )

    args = parser.parse_args()

    root = detect_root()

    review = build_review(
        root
    )

    print_review(
        root,
        review,
    )

    if not args.execute:
        print()
        print(
            "DRY-RUN ONLY — no product/evidence files moved or written."
        )

        if all(
            review["gates"].values()
        ):
            print(
                "Execute finalization with:"
            )
            print(
                "python verify_canonical_implementation.py --execute"
            )

        return

    execute_finalize(
        root,
        review,
    )


if __name__ == "__main__":
    main()
