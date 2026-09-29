from __future__ import annotations

import argparse
import sys
from pathlib import Path

NEW_FILES = {
    "final_pipeline/src/fraud_screening/training/__init__.py":
        '"""Rebuild-only training utilities.\n\nThis package creates new learned state only for explicit rebuild runs.\nIt never writes to the official artifact directory.\n"""\n\nfrom fraud_screening.training.factory import (\n    REBUILD_MODEL_ID,\n    build_rebuild_model,\n)\nfrom fraud_screening.training.outputs import (\n    RebuildOutputPaths,\n    create_rebuild_output_paths,\n    validate_run_id,\n)\n\n__all__ = [\n    "REBUILD_MODEL_ID",\n    "RebuildOutputPaths",\n    "build_rebuild_model",\n    "create_rebuild_output_paths",\n    "validate_run_id",\n]\n',
    "final_pipeline/src/fraud_screening/training/factory.py":
        '"""Factory for reproducing the frozen final Random Forest configuration."""\n\nfrom __future__ import annotations\n\nfrom sklearn.ensemble import RandomForestClassifier\n\nfrom fraud_screening.artifacts import (\n    EXPECTED_MODEL_PARAMETERS,\n    MODEL_ID,\n)\n\n\nREBUILD_MODEL_ID = MODEL_ID\n\n\ndef build_rebuild_model() -> RandomForestClassifier:\n    """Create an unfitted estimator matching the frozen final configuration."""\n\n    estimator = RandomForestClassifier(\n        bootstrap=True,\n        ccp_alpha=0.0,\n        class_weight="balanced",\n        criterion="gini",\n        max_depth=None,\n        max_features="sqrt",\n        max_samples=None,\n        min_samples_leaf=1,\n        min_samples_split=2,\n        n_estimators=100,\n        n_jobs=-1,\n        random_state=42,\n    )\n\n    actual_params = estimator.get_params(\n        deep=False\n    )\n\n    for key, expected_value in (\n        EXPECTED_MODEL_PARAMETERS.items()\n    ):\n        if actual_params[key] != expected_value:\n            raise RuntimeError(\n                f"Rebuild model parameter mismatch: {key}."\n            )\n\n    return estimator\n',
    "final_pipeline/src/fraud_screening/training/outputs.py":
        '"""Filesystem policy for rebuild outputs."""\n\nfrom __future__ import annotations\n\nimport re\nfrom dataclasses import dataclass\nfrom pathlib import Path\n\nfrom fraud_screening.errors import (\n    ArtifactCompatibilityError,\n)\n\n\n_RUN_ID_PATTERN = re.compile(\n    r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$"\n)\n\n_FORBIDDEN_COMPONENTS = {\n    "official",\n    "artifacts",\n    "research",\n}\n\n\n@dataclass(frozen=True, slots=True)\nclass RebuildOutputPaths:\n    """Resolved paths for one isolated rebuild run."""\n\n    run_id: str\n    run_dir: Path\n    model_path: Path\n    preprocessing_path: Path\n    manifest_path: Path\n\n    def all_files(self) -> tuple[Path, ...]:\n        return (\n            self.model_path,\n            self.preprocessing_path,\n            self.manifest_path,\n        )\n\n\ndef validate_run_id(run_id: str) -> str:\n    """Validate a human-readable, filesystem-safe rebuild run id."""\n\n    if not isinstance(run_id, str):\n        raise ValueError(\n            "run_id must be a string."\n        )\n\n    normalized = run_id.strip()\n\n    if not normalized:\n        raise ValueError(\n            "run_id must not be empty."\n        )\n\n    if not _RUN_ID_PATTERN.fullmatch(\n        normalized\n    ):\n        raise ValueError(\n            "run_id may contain only letters, numbers, \'.\', \'_\' and \'-\'."\n        )\n\n    lowered_parts = {\n        part.lower()\n        for part in re.split(\n            r"[._-]+",\n            normalized,\n        )\n        if part\n    }\n\n    if lowered_parts & _FORBIDDEN_COMPONENTS:\n        raise ValueError(\n            "run_id contains a reserved path term."\n        )\n\n    return normalized\n\n\ndef _ensure_under(\n    candidate: Path,\n    parent: Path,\n) -> None:\n    try:\n        candidate.relative_to(\n            parent\n        )\n    except ValueError as exc:\n        raise ArtifactCompatibilityError(\n            "Rebuild output escaped the allowed rebuild root."\n        ) from exc\n\n\ndef create_rebuild_output_paths(\n    project_root: str | Path,\n    run_id: str,\n    *,\n    create: bool = False,\n) -> RebuildOutputPaths:\n    """Resolve a unique rebuild directory without touching official artifacts."""\n\n    root = Path(\n        project_root\n    ).resolve()\n\n    validated_run_id = validate_run_id(\n        run_id\n    )\n\n    rebuild_root = (\n        root\n        / "final_pipeline"\n        / "outputs"\n        / "rebuilds"\n    ).resolve()\n\n    official_root = (\n        root\n        / "final_pipeline"\n        / "artifacts"\n        / "official"\n    ).resolve()\n\n    run_dir = (\n        rebuild_root\n        / validated_run_id\n    ).resolve()\n\n    _ensure_under(\n        run_dir,\n        rebuild_root,\n    )\n\n    if (\n        run_dir == official_root\n        or official_root in run_dir.parents\n    ):\n        raise ArtifactCompatibilityError(\n            "Rebuild output may not target official artifacts."\n        )\n\n    if run_dir.exists():\n        raise FileExistsError(\n            f"Rebuild run already exists: {run_dir}"\n        )\n\n    paths = RebuildOutputPaths(\n        run_id=validated_run_id,\n        run_dir=run_dir,\n        model_path=run_dir / "model.joblib",\n        preprocessing_path=\n            run_dir / "preprocessing_state.json",\n        manifest_path=\n            run_dir / "rebuild_manifest.json",\n    )\n\n    if create:\n        rebuild_root.mkdir(\n            parents=True,\n            exist_ok=True,\n        )\n        run_dir.mkdir(\n            parents=False,\n            exist_ok=False,\n        )\n\n    return paths\n',
    "final_pipeline/tests/unit/test_rebuild_model_factory.py":
        'from __future__ import annotations\n\nimport unittest\n\nfrom sklearn.ensemble import RandomForestClassifier\n\nfrom fraud_screening.artifacts import (\n    EXPECTED_MODEL_PARAMETERS,\n    MODEL_ID,\n)\nfrom fraud_screening.training import (\n    REBUILD_MODEL_ID,\n    build_rebuild_model,\n)\n\n\nclass RebuildModelFactoryTests(unittest.TestCase):\n    def test_factory_returns_random_forest(self) -> None:\n        estimator = build_rebuild_model()\n\n        self.assertIsInstance(\n            estimator,\n            RandomForestClassifier,\n        )\n\n    def test_factory_model_id_matches_official_identity(self) -> None:\n        self.assertEqual(\n            REBUILD_MODEL_ID,\n            MODEL_ID,\n        )\n\n    def test_factory_matches_guarded_final_parameters(self) -> None:\n        estimator = build_rebuild_model()\n        actual = estimator.get_params(\n            deep=False\n        )\n\n        for key, expected in (\n            EXPECTED_MODEL_PARAMETERS.items()\n        ):\n            self.assertEqual(\n                actual[key],\n                expected,\n                msg=key,\n            )\n\n    def test_factory_returns_unfitted_model(self) -> None:\n        estimator = build_rebuild_model()\n\n        self.assertFalse(\n            hasattr(\n                estimator,\n                "classes_",\n            )\n        )\n        self.assertFalse(\n            hasattr(\n                estimator,\n                "estimators_",\n            )\n        )\n\n    def test_factory_creates_fresh_instances(self) -> None:\n        first = build_rebuild_model()\n        second = build_rebuild_model()\n\n        self.assertIsNot(\n            first,\n            second,\n        )\n\n\nif __name__ == "__main__":\n    unittest.main()\n',
    "final_pipeline/tests/unit/test_rebuild_output_policy.py":
        'from __future__ import annotations\n\nimport tempfile\nimport unittest\nfrom pathlib import Path\n\nfrom fraud_screening.training import (\n    create_rebuild_output_paths,\n    validate_run_id,\n)\n\n\nclass RebuildOutputPolicyTests(unittest.TestCase):\n    def test_valid_run_id_is_preserved(self) -> None:\n        self.assertEqual(\n            validate_run_id(\n                "rebuild-20260922-001"\n            ),\n            "rebuild-20260922-001",\n        )\n\n    def test_empty_run_id_is_rejected(self) -> None:\n        with self.assertRaises(\n            ValueError\n        ):\n            validate_run_id("   ")\n\n    def test_path_separator_is_rejected(self) -> None:\n        for value in [\n            "../escape",\n            "a/b",\n            r"a\\b",\n        ]:\n            with self.subTest(\n                value=value\n            ):\n                with self.assertRaises(\n                    ValueError\n                ):\n                    validate_run_id(\n                        value\n                    )\n\n    def test_reserved_official_term_is_rejected(self) -> None:\n        with self.assertRaises(\n            ValueError\n        ):\n            validate_run_id(\n                "official"\n            )\n\n    def test_reserved_artifacts_term_is_rejected(self) -> None:\n        with self.assertRaises(\n            ValueError\n        ):\n            validate_run_id(\n                "rebuild-artifacts-001"\n            )\n\n    def test_reserved_research_term_is_rejected(self) -> None:\n        with self.assertRaises(\n            ValueError\n        ):\n            validate_run_id(\n                "research-copy"\n            )\n\n    def test_paths_are_under_rebuild_root(self) -> None:\n        with tempfile.TemporaryDirectory() as tmp:\n            root = Path(tmp)\n\n            paths = create_rebuild_output_paths(\n                root,\n                "run-001",\n            )\n\n            expected = (\n                root.resolve()\n                / "final_pipeline"\n                / "outputs"\n                / "rebuilds"\n                / "run-001"\n            )\n\n            self.assertEqual(\n                paths.run_dir,\n                expected,\n            )\n            self.assertEqual(\n                paths.model_path,\n                expected\n                / "model.joblib",\n            )\n            self.assertEqual(\n                paths.preprocessing_path,\n                expected\n                / "preprocessing_state.json",\n            )\n            self.assertEqual(\n                paths.manifest_path,\n                expected\n                / "rebuild_manifest.json",\n            )\n\n    def test_resolution_is_read_only_by_default(self) -> None:\n        with tempfile.TemporaryDirectory() as tmp:\n            root = Path(tmp)\n\n            paths = create_rebuild_output_paths(\n                root,\n                "run-001",\n            )\n\n            self.assertFalse(\n                paths.run_dir.exists()\n            )\n\n    def test_create_makes_only_rebuild_run_directory(self) -> None:\n        with tempfile.TemporaryDirectory() as tmp:\n            root = Path(tmp)\n\n            paths = create_rebuild_output_paths(\n                root,\n                "run-001",\n                create=True,\n            )\n\n            self.assertTrue(\n                paths.run_dir.is_dir()\n            )\n            self.assertFalse(\n                paths.model_path.exists()\n            )\n\n            official = (\n                root\n                / "final_pipeline"\n                / "artifacts"\n                / "official"\n            )\n\n            self.assertFalse(\n                official.exists()\n            )\n\n    def test_existing_run_is_never_overwritten(self) -> None:\n        with tempfile.TemporaryDirectory() as tmp:\n            root = Path(tmp)\n\n            first = create_rebuild_output_paths(\n                root,\n                "run-001",\n                create=True,\n            )\n\n            marker = (\n                first.run_dir\n                / "keep.txt"\n            )\n            marker.write_text(\n                "preserve",\n                encoding="utf-8",\n            )\n\n            with self.assertRaises(\n                FileExistsError\n            ):\n                create_rebuild_output_paths(\n                    root,\n                    "run-001",\n                    create=True,\n                )\n\n            self.assertEqual(\n                marker.read_text(\n                    encoding="utf-8"\n                ),\n                "preserve",\n            )\n\n    def test_returned_file_set_has_no_official_path(self) -> None:\n        with tempfile.TemporaryDirectory() as tmp:\n            root = Path(tmp)\n\n            paths = create_rebuild_output_paths(\n                root,\n                "run-001",\n            )\n\n            for path in (\n                paths.all_files()\n            ):\n                self.assertNotIn(\n                    "artifacts/official",\n                    path.as_posix(),\n                )\n\n\nif __name__ == "__main__":\n    unittest.main()\n',
}


def detect_root() -> Path:
    root = Path.cwd().resolve()

    required = [
        root / "research",
        root / "final_pipeline",
        root / "application",
        root
        / "final_pipeline"
        / "artifacts"
        / "official"
        / "model"
        / "model.joblib",
        root
        / "final_pipeline"
        / "artifacts"
        / "official"
        / "preprocessing"
        / "preprocessing_state.json",
        root
        / "final_pipeline"
        / "artifacts"
        / "official"
        / "manifest"
        / "artifact_manifest.json",
    ]

    if not all(path.exists() for path in required):
        raise RuntimeError(
            "Repository or official artifact set is incomplete."
        )

    return root


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Create rebuild model factory and output-policy modules. "
            "This does not fit any model."
        ),
    )

    args = parser.parse_args()

    root = detect_root()

    dependencies = [
        root
        / "final_pipeline/src/fraud_screening/artifacts/model_loader.py",
        root
        / "final_pipeline/src/fraud_screening/errors.py",
        root
        / "final_pipeline/tests/unit/test_model_artifact_loader.py",
    ]

    missing_dependencies = [
        str(path.relative_to(root))
        for path in dependencies
        if not path.is_file()
    ]

    collisions = [
        relative
        for relative in NEW_FILES
        if (root / relative).exists()
    ]

    print("=" * 92)
    print(
        "REBUILD FOUNDATION — MODEL FACTORY + OUTPUT POLICY"
    )
    print("=" * 92)
    print("Project root:", root)
    print(
        "Mode:",
        "EXECUTE — NEW SOURCE/TEST FILES"
        if args.execute
        else "DRY RUN — NO FILES WRITTEN",
    )

    print("\n[1] Dependency gate")
    print(
        " - missing dependencies:",
        len(missing_dependencies),
    )
    for relative in missing_dependencies:
        print("   *", relative)

    print("\n[2] Planned files")
    for relative in NEW_FILES:
        print(" -", relative)

    print("\n[3] Collision gate")
    print(
        " - collisions:",
        len(collisions),
    )
    for relative in collisions:
        print("   *", relative)

    print("\n[4] Rebuild model identity")
    print(
        " - model family: RandomForestClassifier"
    )
    print(
        " - model id: RF-REF-100-GINI-SQRT-UNPRUNED-CW"
    )
    print(
        " - n_estimators: 100"
    )
    print(
        " - criterion: gini"
    )
    print(
        " - max_depth: None"
    )
    print(
        " - max_features: sqrt"
    )
    print(
        " - class_weight: balanced"
    )
    print(
        " - random_state: 42"
    )
    print(
        " - n_jobs: -1"
    )

    print("\n[5] Filesystem policy")
    print(
        " - rebuild root: final_pipeline/outputs/rebuilds/<run_id>/"
    )
    print(
        " - existing run overwrite: FORBIDDEN"
    )
    print(
        " - official artifact target: FORBIDDEN"
    )
    print(
        " - research output target: FORBIDDEN"
    )

    print("\n[6] Scientific/runtime boundary")
    print(" - model.fit(): NO")
    print(" - preprocessing.fit(): NO")
    print(" - predict(): NO")
    print(" - predict_proba(): NO")
    print(" - official artifact write: NO")
    print(" - official artifact modification: NO")
    print(" - research modification: NO")

    if (
        missing_dependencies
        or collisions
    ):
        print("\nRESULT: STOP")
        print(
            "No files were written."
        )
        raise SystemExit(1)

    if not args.execute:
        print("\nDRY-RUN RESULT: PASS")
        print("Execute with:")
        print(
            "python bootstrap_rebuild_foundation.py --execute"
        )
        print("=" * 92)
        return

    created = []

    try:
        for relative, content in (
            NEW_FILES.items()
        ):
            path = root / relative

            path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            path.write_text(
                content,
                encoding="utf-8",
            )

            created.append(path)

    except Exception:
        for path in reversed(
            created
        ):
            try:
                path.unlink()
            except OSError:
                pass

        raise

    print("\n[7] Created")
    for path in created:
        print(
            " -",
            path.relative_to(root),
        )

    print("\nBOOTSTRAP RESULT: PASS")
    print(
        "No learned state was created."
    )
    print(
        "Run full unit suite:"
    )
    print(
        "PYTHONPATH=final_pipeline/src "
        "python -m unittest discover "
        "-s final_pipeline/tests -v"
    )
    print("=" * 92)


if __name__ == "__main__":
    main()
