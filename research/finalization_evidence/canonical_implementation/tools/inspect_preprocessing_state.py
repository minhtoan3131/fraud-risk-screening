from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_PREPROCESSING_SHA256 = (
    "c1dc5486acdcd43f5f75fbd0a11bd7ad3a3b00761d50ae96b2b1fbe5b5d9ef98"
)

PREPROCESSING_REL = Path(
    "research/data/processed/m8_02_final_test_artifact_audit/"
    "m8_02_w_short_preprocessing_state.json"
)

FEATURE_NAMES_REL = Path(
    "research/data/processed/m4_07_baseline_ready/"
    "feature_names.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def describe(
    value: Any,
    *,
    indent: int = 0,
    name: str = "<root>",
    depth: int = 0,
) -> None:
    prefix = " " * indent

    if isinstance(value, dict):
        print(
            f"{prefix}{name}: dict "
            f"({len(value)} keys)"
        )

        if depth >= 4:
            return

        for key in sorted(value):
            describe(
                value[key],
                indent=indent + 2,
                name=str(key),
                depth=depth + 1,
            )
        return

    if isinstance(value, list):
        print(
            f"{prefix}{name}: list "
            f"(len={len(value)})"
        )

        if len(value) <= 60:
            print(
                f"{prefix}  value={value!r}"
            )
        elif value:
            print(
                f"{prefix}  first_10={value[:10]!r}"
            )
            print(
                f"{prefix}  last_5={value[-5:]!r}"
            )
        return

    print(
        f"{prefix}{name}: "
        f"{type(value).__name__} "
        f"value={value!r}"
    )


def main() -> None:
    root = Path.cwd().resolve()

    required_root_dirs = [
        root / "research",
        root / "final_pipeline",
        root / "application",
    ]

    if not all(
        path.is_dir()
        for path in required_root_dirs
    ):
        raise RuntimeError(
            "Run from repository root."
        )

    preprocessing_path = (
        root / PREPROCESSING_REL
    )
    feature_names_path = (
        root / FEATURE_NAMES_REL
    )

    print("=" * 88)
    print("FROZEN PREPROCESSING STATE — READ-ONLY INSPECTION")
    print("=" * 88)
    print("Project root:", root)
    print("Mode: READ ONLY")
    print(" - preprocessing transform: NO")
    print(" - preprocessing fit: NO")
    print(" - model fit: NO")
    print(" - prediction: NO")
    print(" - file write: NO")

    print("\n[1] Required files")
    for path in [
        preprocessing_path,
        feature_names_path,
    ]:
        print(
            " -",
            path.relative_to(root),
            "| exists=",
            path.is_file(),
        )

    if not preprocessing_path.is_file():
        raise FileNotFoundError(
            preprocessing_path
        )

    if not feature_names_path.is_file():
        raise FileNotFoundError(
            feature_names_path
        )

    actual_sha = sha256_file(
        preprocessing_path
    )

    print("\n[2] Frozen preprocessing fingerprint")
    print(" - actual  :", actual_sha)
    print(
        " - expected:",
        EXPECTED_PREPROCESSING_SHA256,
    )
    print(
        " - match   :",
        actual_sha
        == EXPECTED_PREPROCESSING_SHA256,
    )

    if (
        actual_sha
        != EXPECTED_PREPROCESSING_SHA256
    ):
        raise RuntimeError(
            "Frozen preprocessing SHA-256 mismatch."
        )

    preprocessing_state = json.loads(
        preprocessing_path.read_text(
            encoding="utf-8"
        )
    )

    canonical_feature_names = json.loads(
        feature_names_path.read_text(
            encoding="utf-8"
        )
    )

    print("\n[3] Preprocessing JSON structure")
    describe(
        preprocessing_state,
        name="preprocessing_state",
    )

    print("\n[4] Canonical encoded feature names")
    print(
        " - type:",
        type(canonical_feature_names).__name__,
    )
    print(
        " - count:",
        len(canonical_feature_names)
        if isinstance(
            canonical_feature_names,
            list,
        )
        else "N/A",
    )
    print(
        " - value:",
        canonical_feature_names,
    )

    print("\n[5] Direct identity checks")

    if isinstance(
        preprocessing_state,
        dict,
    ):
        for key in [
            "feature_count",
            "feature_names",
            "numeric_columns",
            "boolean_columns",
            "categorical_columns",
            "numeric_mean",
            "numeric_scale",
            "categories",
            "categorical_categories",
            "category_vocabularies",
            "categorical_vocabularies",
            "unknown_token",
            "strategy",
            "fit_source",
            "validation_exact_reproduction",
        ]:
            if key in preprocessing_state:
                value = preprocessing_state[key]
                print(
                    f" - {key}: "
                    f"type={type(value).__name__}, "
                    f"value={value!r}"
                )

    state_feature_names = (
        preprocessing_state.get(
            "feature_names"
        )
        if isinstance(
            preprocessing_state,
            dict,
        )
        else None
    )

    print(
        " - state feature_names == "
        "canonical feature_names:",
        state_feature_names
        == canonical_feature_names,
    )

    print("\n" + "=" * 88)
    print("INSPECTION RESULT: PASS")
    print(
        "Frozen preprocessing state was read "
        "without transformation or modification."
    )
    print("=" * 88)


if __name__ == "__main__":
    main()
