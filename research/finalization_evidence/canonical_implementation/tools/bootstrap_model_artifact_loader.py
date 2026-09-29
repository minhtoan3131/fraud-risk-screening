from __future__ import annotations

import argparse
import hashlib
import importlib.util
import sys
from pathlib import Path

NEW_FILES = {
    "final_pipeline/src/fraud_screening/artifacts/__init__.py":
        '"""Verified artifact loading for the screening pipeline."""\n\nfrom fraud_screening.artifacts.model_loader import (\n    EXPECTED_MODEL_PARAMETERS,\n    MODEL_ID,\n    POSITIVE_CLASS,\n    VerifiedModelArtifact,\n    load_verified_model,\n)\n\n__all__ = [\n    "EXPECTED_MODEL_PARAMETERS",\n    "MODEL_ID",\n    "POSITIVE_CLASS",\n    "VerifiedModelArtifact",\n    "load_verified_model",\n]\n',
    "final_pipeline/src/fraud_screening/artifacts/model_loader.py":
        '"""Fingerprint-verified loading for the frozen classifier artifact."""\n\nfrom __future__ import annotations\n\nimport hashlib\nfrom dataclasses import dataclass\nfrom pathlib import Path\nfrom typing import Any\n\nimport joblib\nfrom sklearn.ensemble import RandomForestClassifier\n\nfrom fraud_screening.errors import (\n    ArtifactCompatibilityError,\n    ArtifactFingerprintError,\n    ArtifactNotFoundError,\n)\n\n\nMODEL_ID = "RF-REF-100-GINI-SQRT-UNPRUNED-CW"\nPOSITIVE_CLASS = 1\n\nEXPECTED_MODEL_PARAMETERS = {\n    "bootstrap": True,\n    "ccp_alpha": 0.0,\n    "class_weight": "balanced",\n    "criterion": "gini",\n    "max_depth": None,\n    "max_features": "sqrt",\n    "max_samples": None,\n    "min_samples_leaf": 1,\n    "min_samples_split": 2,\n    "n_estimators": 100,\n    "n_jobs": -1,\n    "random_state": 42,\n}\n\n\ndef sha256_file(\n    path: str | Path,\n    *,\n    chunk_size: int = 1024 * 1024,\n) -> str:\n    """Return SHA-256 for one artifact file."""\n\n    artifact_path = Path(path)\n    digest = hashlib.sha256()\n\n    try:\n        with artifact_path.open("rb") as file:\n            for chunk in iter(\n                lambda: file.read(chunk_size),\n                b"",\n            ):\n                digest.update(chunk)\n    except FileNotFoundError as exc:\n        raise ArtifactNotFoundError(\n            f"Artifact not found: {artifact_path}"\n        ) from exc\n\n    return digest.hexdigest()\n\n\n@dataclass(frozen=True, slots=True)\nclass VerifiedModelArtifact:\n    """Loaded estimator plus verified identity metadata."""\n\n    estimator: RandomForestClassifier\n    sha256: str\n    model_id: str\n    classes: tuple[int, ...]\n    positive_class_index: int\n\n    @property\n    def positive_class(self) -> int:\n        return POSITIVE_CLASS\n\n\ndef _validate_estimator_identity(\n    estimator: Any,\n) -> tuple[tuple[int, ...], int]:\n    if not isinstance(\n        estimator,\n        RandomForestClassifier,\n    ):\n        raise ArtifactCompatibilityError(\n            "Artifact is not a RandomForestClassifier."\n        )\n\n    actual_params = estimator.get_params(\n        deep=False\n    )\n\n    for key, expected_value in (\n        EXPECTED_MODEL_PARAMETERS.items()\n    ):\n        if key not in actual_params:\n            raise ArtifactCompatibilityError(\n                f"Estimator parameter missing: {key}."\n            )\n\n        if actual_params[key] != expected_value:\n            raise ArtifactCompatibilityError(\n                f"Estimator parameter mismatch: {key}."\n            )\n\n    if not hasattr(estimator, "classes_"):\n        raise ArtifactCompatibilityError(\n            "Estimator has no fitted classes_ attribute."\n        )\n\n    try:\n        classes = tuple(\n            int(value)\n            for value in estimator.classes_.tolist()\n        )\n    except Exception as exc:\n        raise ArtifactCompatibilityError(\n            "Estimator classes_ cannot be normalized."\n        ) from exc\n\n    if classes != (0, 1):\n        raise ArtifactCompatibilityError(\n            "Estimator classes must be exactly [0, 1]."\n        )\n\n    positive_class_index = classes.index(\n        POSITIVE_CLASS\n    )\n\n    if positive_class_index != 1:\n        raise ArtifactCompatibilityError(\n            "Positive class index must be 1."\n        )\n\n    return (\n        classes,\n        positive_class_index,\n    )\n\n\ndef load_verified_model(\n    path: str | Path,\n    *,\n    expected_sha256: str,\n) -> VerifiedModelArtifact:\n    """Load and verify the frozen model without running inference."""\n\n    artifact_path = Path(path)\n\n    actual_sha256 = sha256_file(\n        artifact_path\n    )\n\n    if actual_sha256 != expected_sha256:\n        raise ArtifactFingerprintError(\n            "Model artifact SHA-256 mismatch."\n        )\n\n    try:\n        estimator = joblib.load(\n            artifact_path\n        )\n    except FileNotFoundError as exc:\n        raise ArtifactNotFoundError(\n            f"Artifact not found: {artifact_path}"\n        ) from exc\n    except Exception as exc:\n        raise ArtifactCompatibilityError(\n            "Model artifact could not be loaded."\n        ) from exc\n\n    (\n        classes,\n        positive_class_index,\n    ) = _validate_estimator_identity(\n        estimator\n    )\n\n    return VerifiedModelArtifact(\n        estimator=estimator,\n        sha256=actual_sha256,\n        model_id=MODEL_ID,\n        classes=classes,\n        positive_class_index=\n            positive_class_index,\n    )\n',
    "final_pipeline/tests/unit/test_model_artifact_loader.py":
        'from __future__ import annotations\n\nimport tempfile\nimport unittest\nfrom pathlib import Path\n\nimport joblib\nimport numpy as np\nfrom sklearn.ensemble import RandomForestClassifier\n\nfrom fraud_screening.artifacts import (\n    EXPECTED_MODEL_PARAMETERS,\n    MODEL_ID,\n    POSITIVE_CLASS,\n    load_verified_model,\n)\nfrom fraud_screening.artifacts.model_loader import (\n    sha256_file,\n)\nfrom fraud_screening.errors import (\n    ArtifactCompatibilityError,\n    ArtifactFingerprintError,\n    ArtifactNotFoundError,\n)\n\n\ndef compatible_estimator() -> RandomForestClassifier:\n    estimator = RandomForestClassifier(\n        bootstrap=True,\n        ccp_alpha=0.0,\n        class_weight="balanced",\n        criterion="gini",\n        max_depth=None,\n        max_features="sqrt",\n        max_samples=None,\n        min_samples_leaf=1,\n        min_samples_split=2,\n        n_estimators=100,\n        n_jobs=-1,\n        random_state=42,\n    )\n\n    # Identity tests do not train. Only the fitted-class metadata required\n    # by the loader is attached to this temporary compatibility fixture.\n    estimator.classes_ = np.asarray(\n        [0, 1],\n        dtype=np.int64,\n    )\n\n    return estimator\n\n\nclass ModelArtifactLoaderTests(unittest.TestCase):\n    def dump_fixture(\n        self,\n        directory: str,\n        estimator: object,\n    ) -> Path:\n        path = Path(directory) / "model.joblib"\n        joblib.dump(estimator, path)\n        return path\n\n    def test_expected_model_id_is_stable(self) -> None:\n        self.assertEqual(\n            MODEL_ID,\n            "RF-REF-100-GINI-SQRT-UNPRUNED-CW",\n        )\n        self.assertEqual(\n            POSITIVE_CLASS,\n            1,\n        )\n\n    def test_expected_parameter_contract_is_exact(self) -> None:\n        self.assertEqual(\n            EXPECTED_MODEL_PARAMETERS,\n            {\n                "bootstrap": True,\n                "ccp_alpha": 0.0,\n                "class_weight": "balanced",\n                "criterion": "gini",\n                "max_depth": None,\n                "max_features": "sqrt",\n                "max_samples": None,\n                "min_samples_leaf": 1,\n                "min_samples_split": 2,\n                "n_estimators": 100,\n                "n_jobs": -1,\n                "random_state": 42,\n            },\n        )\n\n    def test_verified_loader_accepts_compatible_artifact(self) -> None:\n        with tempfile.TemporaryDirectory() as tmp:\n            path = self.dump_fixture(\n                tmp,\n                compatible_estimator(),\n            )\n            digest = sha256_file(path)\n\n            artifact = load_verified_model(\n                path,\n                expected_sha256=digest,\n            )\n\n            self.assertEqual(\n                artifact.model_id,\n                MODEL_ID,\n            )\n            self.assertEqual(\n                artifact.classes,\n                (0, 1),\n            )\n            self.assertEqual(\n                artifact.positive_class_index,\n                1,\n            )\n            self.assertEqual(\n                artifact.positive_class,\n                1,\n            )\n\n    def test_sha_mismatch_is_rejected_before_load(self) -> None:\n        with tempfile.TemporaryDirectory() as tmp:\n            path = self.dump_fixture(\n                tmp,\n                compatible_estimator(),\n            )\n\n            with self.assertRaises(\n                ArtifactFingerprintError\n            ):\n                load_verified_model(\n                    path,\n                    expected_sha256="0" * 64,\n                )\n\n    def test_missing_file_raises_domain_error(self) -> None:\n        with self.assertRaises(\n            ArtifactNotFoundError\n        ):\n            load_verified_model(\n                "/definitely/missing/model.joblib",\n                expected_sha256="0" * 64,\n            )\n\n    def test_wrong_estimator_type_is_rejected(self) -> None:\n        with tempfile.TemporaryDirectory() as tmp:\n            path = self.dump_fixture(\n                tmp,\n                {"not": "a model"},\n            )\n            digest = sha256_file(path)\n\n            with self.assertRaises(\n                ArtifactCompatibilityError\n            ):\n                load_verified_model(\n                    path,\n                    expected_sha256=digest,\n                )\n\n    def test_parameter_mismatch_is_rejected(self) -> None:\n        estimator = compatible_estimator()\n        estimator.set_params(\n            n_estimators=101\n        )\n\n        with tempfile.TemporaryDirectory() as tmp:\n            path = self.dump_fixture(\n                tmp,\n                estimator,\n            )\n            digest = sha256_file(path)\n\n            with self.assertRaises(\n                ArtifactCompatibilityError\n            ):\n                load_verified_model(\n                    path,\n                    expected_sha256=digest,\n                )\n\n    def test_missing_classes_is_rejected(self) -> None:\n        estimator = compatible_estimator()\n        del estimator.classes_\n\n        with tempfile.TemporaryDirectory() as tmp:\n            path = self.dump_fixture(\n                tmp,\n                estimator,\n            )\n            digest = sha256_file(path)\n\n            with self.assertRaises(\n                ArtifactCompatibilityError\n            ):\n                load_verified_model(\n                    path,\n                    expected_sha256=digest,\n                )\n\n    def test_wrong_classes_are_rejected(self) -> None:\n        estimator = compatible_estimator()\n        estimator.classes_ = np.asarray(\n            [0, 2],\n            dtype=np.int64,\n        )\n\n        with tempfile.TemporaryDirectory() as tmp:\n            path = self.dump_fixture(\n                tmp,\n                estimator,\n            )\n            digest = sha256_file(path)\n\n            with self.assertRaises(\n                ArtifactCompatibilityError\n            ):\n                load_verified_model(\n                    path,\n                    expected_sha256=digest,\n                )\n\n    def test_loader_does_not_expose_prediction_wrapper(self) -> None:\n        with tempfile.TemporaryDirectory() as tmp:\n            path = self.dump_fixture(\n                tmp,\n                compatible_estimator(),\n            )\n            digest = sha256_file(path)\n\n            artifact = load_verified_model(\n                path,\n                expected_sha256=digest,\n            )\n\n            self.assertFalse(\n                hasattr(\n                    artifact,\n                    "predict",\n                )\n            )\n            self.assertFalse(\n                hasattr(\n                    artifact,\n                    "predict_proba",\n                )\n            )\n\n\nif __name__ == "__main__":\n    unittest.main()\n',
}

PYPROJECT_PATH = Path("final_pipeline/pyproject.toml")
EXPECTED_PYPROJECT = '[build-system]\nrequires = ["setuptools>=68"]\nbuild-backend = "setuptools.build_meta"\n\n[project]\nname = "fraud-screening"\nversion = "0.1.0"\ndescription = "Transaction fraud risk screening pipeline"\nrequires-python = ">=3.11"\ndependencies = [\n    "numpy",\n    "scipy",\n]\n\n[tool.setuptools.packages.find]\nwhere = ["src"]\n'
UPDATED_PYPROJECT = '[build-system]\nrequires = ["setuptools>=68"]\nbuild-backend = "setuptools.build_meta"\n\n[project]\nname = "fraud-screening"\nversion = "0.1.0"\ndescription = "Transaction fraud risk screening pipeline"\nrequires-python = ">=3.11"\ndependencies = [\n    "joblib",\n    "numpy",\n    "scikit-learn==1.9.1",\n    "scipy",\n]\n\n[tool.setuptools.packages.find]\nwhere = ["src"]\n'

MODEL_REL = Path(
    "research/data/processed/m8_02_final_test_artifact_audit/"
    "m8_02_selected_rf_estimator.joblib"
)

EXPECTED_MODEL_SHA256 = (
    "61bdeeba5fd163cce8a5a9efa028a9cdc2f95c0796822a5222c0604c0302115f"
)


def detect_root() -> Path:
    root = Path.cwd().resolve()

    required = [
        root / "research",
        root / "final_pipeline",
        root / "application",
    ]

    if not all(path.is_dir() for path in required):
        raise RuntimeError(
            "Run this script from the repository root."
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


def package_status() -> dict[str, bool]:
    return {
        "joblib":
            importlib.util.find_spec("joblib")
            is not None,
        "sklearn":
            importlib.util.find_spec("sklearn")
            is not None,
    }


def verify_artifact(root: Path) -> None:
    model_path = root / MODEL_REL

    print("=" * 90)
    print("FINAL PIPELINE — MODEL ARTIFACT READ-ONLY VERIFICATION")
    print("=" * 90)
    print("Mode: LOAD + IDENTITY ONLY")
    print(" - model fit: NO")
    print(" - predict: NO")
    print(" - predict_proba: NO")
    print(" - preprocessing transform: NO")
    print(" - file write: NO")

    if not model_path.is_file():
        raise FileNotFoundError(model_path)

    actual_sha = sha256_file(
        model_path
    )

    print("\n[1] Frozen estimator fingerprint")
    print(" - actual  :", actual_sha)
    print(" - expected:", EXPECTED_MODEL_SHA256)
    print(
        " - match   :",
        actual_sha == EXPECTED_MODEL_SHA256,
    )

    if actual_sha != EXPECTED_MODEL_SHA256:
        raise RuntimeError(
            "Frozen estimator fingerprint mismatch."
        )

    src_path = root / "final_pipeline/src"
    sys.path.insert(0, str(src_path))

    try:
        import sklearn

        from fraud_screening.artifacts import (
            EXPECTED_MODEL_PARAMETERS,
            load_verified_model,
        )

        artifact = load_verified_model(
            model_path,
            expected_sha256=
                EXPECTED_MODEL_SHA256,
        )

        actual_params = (
            artifact.estimator.get_params(
                deep=False
            )
        )

        gates = {
            "G01_SHA256":
                artifact.sha256
                == EXPECTED_MODEL_SHA256,
            "G02_TYPE":
                type(
                    artifact.estimator
                ).__name__
                == "RandomForestClassifier",
            "G03_CLASSES":
                artifact.classes == (0, 1),
            "G04_POSITIVE_CLASS_INDEX":
                artifact.positive_class_index
                == 1,
            "G05_MODEL_ID":
                artifact.model_id
                == "RF-REF-100-GINI-SQRT-UNPRUNED-CW",
            "G06_EXPECTED_PARAMETERS":
                all(
                    actual_params[key]
                    == expected
                    for key, expected
                    in EXPECTED_MODEL_PARAMETERS.items()
                ),
            "G07_SKLEARN_VERSION":
                sklearn.__version__
                == "1.9.1",
        }

        print("\n[2] Loaded identity")
        print(
            " - estimator type:",
            type(
                artifact.estimator
            ).__name__,
        )
        print(
            " - classes:",
            list(artifact.classes),
        )
        print(
            " - positive class index:",
            artifact.positive_class_index,
        )
        print(
            " - model id:",
            artifact.model_id,
        )
        print(
            " - scikit-learn:",
            sklearn.__version__,
        )

        print("\n[3] Guarded model parameters")
        for key, expected in (
            EXPECTED_MODEL_PARAMETERS.items()
        ):
            print(
                f" - {key}: "
                f"{actual_params[key]!r}"
            )

        print("\n[4] Verification gates")
        for name, passed in gates.items():
            print(
                f" - {name}: "
                f"{'PASS' if passed else 'FAIL'}"
            )

        if not all(gates.values()):
            print("\nVERIFICATION RESULT: FAIL")
            raise SystemExit(1)

        print("\nVERIFICATION RESULT: PASS")
        print(
            "Frozen estimator loaded and identity-verified."
        )
        print(
            "No model fit, no prediction, no predict_proba."
        )
        print("=" * 90)

    finally:
        try:
            sys.path.remove(str(src_path))
        except ValueError:
            pass


def main() -> None:
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group()

    group.add_argument(
        "--execute",
        action="store_true",
        help="Create verified model-artifact loader.",
    )

    group.add_argument(
        "--verify-artifact",
        action="store_true",
        help="Verify loader against frozen research model.",
    )

    args = parser.parse_args()
    root = detect_root()

    if args.verify_artifact:
        verify_artifact(root)
        return

    pyproject = root / PYPROJECT_PATH

    dependencies = [
        root / "final_pipeline/src/fraud_screening/errors.py",
        root / "final_pipeline/src/fraud_screening/preprocessing/frozen.py",
        root / "final_pipeline/tests/unit/test_frozen_preprocessing.py",
        pyproject,
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

    pyproject_matches = (
        pyproject.is_file()
        and pyproject.read_text(
            encoding="utf-8"
        )
        == EXPECTED_PYPROJECT
    )

    packages = package_status()

    print("=" * 90)
    print("FINAL PIPELINE — VERIFIED MODEL ARTIFACT LOADER BOOTSTRAP")
    print("=" * 90)
    print("Project root:", root)
    print(
        "Mode:",
        "EXECUTE — GUARDED UPDATE + NEW FILES"
        if args.execute
        else "DRY RUN — NO FILES WRITTEN",
    )

    print("\n[1] Dependency gate")
    print(
        " - missing project dependencies:",
        len(missing_dependencies),
    )
    for relative in missing_dependencies:
        print("   *", relative)
    for package, available in packages.items():
        print(
            f" - Python package {package}: "
            f"{'AVAILABLE' if available else 'MISSING'}"
        )

    print("\n[2] pyproject guard")
    print(
        " - exact expected current content:",
        "YES" if pyproject_matches else "NO",
    )

    print("\n[3] Planned new files")
    for relative in NEW_FILES:
        print(" -", relative)

    print("\n[4] New-file collision gate")
    print(" - collisions:", len(collisions))
    for relative in collisions:
        print("   *", relative)

    if (
        missing_dependencies
        or collisions
        or not pyproject_matches
        or not all(packages.values())
    ):
        print("\nRESULT: STOP")
        print("No files were written.")
        raise SystemExit(1)

    print("\n[5] Loader contract")
    print(" - artifact SHA-256: required")
    print(" - estimator type: RandomForestClassifier")
    print(" - classes: [0, 1]")
    print(" - positive class: 1")
    print(" - positive class index: 1")
    print(" - exact guarded parameter set: 12")
    print(" - model id: RF-REF-100-GINI-SQRT-UNPRUNED-CW")
    print(" - runtime research path dependency: NO")

    print("\n[6] Scientific/runtime boundary")
    print(" - model fit: NO")
    print(" - predict: NO")
    print(" - predict_proba: NO")
    print(" - preprocessing fit: NO")
    print(" - threshold application: NO")
    print(" - artifact promotion: NO")
    print(" - research files modified: NO")

    if not args.execute:
        print("\nDRY-RUN RESULT: PASS")
        print("Execute with:")
        print(
            "python bootstrap_model_artifact_loader.py --execute"
        )
        print("=" * 90)
        return

    created = []
    original_pyproject = pyproject.read_text(
        encoding="utf-8"
    )

    try:
        for relative, content in NEW_FILES.items():
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

        pyproject.write_text(
            UPDATED_PYPROJECT,
            encoding="utf-8",
        )

    except Exception:
        for path in reversed(created):
            try:
                path.unlink()
            except OSError:
                pass

        try:
            pyproject.write_text(
                original_pyproject,
                encoding="utf-8",
            )
        except OSError:
            pass

        raise

    print("\n[7] Created")
    for path in created:
        print(" -", path.relative_to(root))

    print("\n[8] Guarded technical metadata update")
    print(" -", PYPROJECT_PATH)
    print(
        "   added runtime dependencies: "
        "joblib, scikit-learn==1.9.1"
    )

    print("\nBOOTSTRAP RESULT: PASS")
    print("Run full unit suite:")
    print(
        "PYTHONPATH=final_pipeline/src "
        "python -m unittest discover "
        "-s final_pipeline/tests -v"
    )
    print("Then verify against the frozen artifact:")
    print(
        "python bootstrap_model_artifact_loader.py --verify-artifact"
    )
    print("=" * 90)


if __name__ == "__main__":
    main()
