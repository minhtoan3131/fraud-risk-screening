from __future__ import annotations

import argparse
from pathlib import Path

INIT_REL = Path(
    "final_pipeline/src/fraud_screening/training/__init__.py"
)

EXPECTED_INIT = '"""Rebuild-only training utilities.\n\nThis package creates new learned state only for explicit rebuild runs.\nIt never writes to the official artifact directory.\n"""\n\nfrom fraud_screening.training.factory import (\n    REBUILD_MODEL_ID,\n    build_rebuild_model,\n)\nfrom fraud_screening.training.outputs import (\n    RebuildOutputPaths,\n    create_rebuild_output_paths,\n    validate_run_id,\n)\nfrom fraud_screening.training.preprocessing_fit import (\n    RebuildPreprocessingFit,\n    RebuildPreprocessingFitter,\n    RebuildPreprocessor,\n    fit_rebuild_preprocessor,\n)\n\n__all__ = [\n    "REBUILD_MODEL_ID",\n    "RebuildOutputPaths",\n    "RebuildPreprocessingFit",\n    "RebuildPreprocessingFitter",\n    "RebuildPreprocessor",\n    "build_rebuild_model",\n    "create_rebuild_output_paths",\n    "fit_rebuild_preprocessor",\n    "validate_run_id",\n]\n'
UPDATED_INIT = '"""Rebuild-only training utilities.\n\nThis package creates new learned state only for explicit rebuild runs.\nIt never writes to the official artifact directory.\n"""\n\nfrom fraud_screening.training.factory import (\n    REBUILD_MODEL_ID,\n    build_rebuild_model,\n)\nfrom fraud_screening.training.outputs import (\n    RebuildOutputPaths,\n    create_rebuild_output_paths,\n    validate_run_id,\n)\nfrom fraud_screening.training.preprocessing_fit import (\n    RebuildPreprocessingFit,\n    RebuildPreprocessingFitter,\n    RebuildPreprocessor,\n    fit_rebuild_preprocessor,\n)\nfrom fraud_screening.training.trainer import (\n    RebuildModelFit,\n    RebuildTrainingError,\n    fit_rebuild_model,\n)\n\n__all__ = [\n    "REBUILD_MODEL_ID",\n    "RebuildModelFit",\n    "RebuildOutputPaths",\n    "RebuildPreprocessingFit",\n    "RebuildPreprocessingFitter",\n    "RebuildPreprocessor",\n    "RebuildTrainingError",\n    "build_rebuild_model",\n    "create_rebuild_output_paths",\n    "fit_rebuild_model",\n    "fit_rebuild_preprocessor",\n    "validate_run_id",\n]\n'

NEW_FILES = {
    "final_pipeline/src/fraud_screening/training/trainer.py":
        '"""In-memory fitting of a rebuild Random Forest model."""\n\nfrom __future__ import annotations\n\nfrom dataclasses import dataclass\nfrom typing import Sequence\n\nimport numpy as np\nfrom scipy import sparse\nfrom sklearn.ensemble import RandomForestClassifier\n\nfrom fraud_screening.artifacts import (\n    EXPECTED_MODEL_PARAMETERS,\n)\nfrom fraud_screening.training.factory import (\n    REBUILD_MODEL_ID,\n    build_rebuild_model,\n)\n\n\nEXPECTED_FEATURE_COUNT = 47\nEXPECTED_TARGET_DTYPE = np.dtype(\n    np.int8\n)\n\n\nclass RebuildTrainingError(ValueError):\n    """Raised when rebuild training input violates the locked contract."""\n\n\n@dataclass(frozen=True, slots=True)\nclass RebuildModelFit:\n    """One newly learned rebuild estimator and descriptive training metadata."""\n\n    estimator: RandomForestClassifier\n    model_id: str\n    training_rows: int\n    fraud_rows: int\n    feature_count: int\n    classes: tuple[int, ...]\n    artifact_identity_status: str\n    evaluation_status: str\n\n\ndef _validate_training_matrix(\n    matrix: sparse.spmatrix,\n) -> sparse.csr_matrix:\n    if not sparse.isspmatrix_csr(\n        matrix\n    ):\n        raise RebuildTrainingError(\n            "Training matrix must be CSR."\n        )\n\n    if matrix.shape[0] <= 0:\n        raise RebuildTrainingError(\n            "Training matrix must contain at least one row."\n        )\n\n    if matrix.shape[1] != EXPECTED_FEATURE_COUNT:\n        raise RebuildTrainingError(\n            "Training matrix must contain exactly 47 features."\n        )\n\n    if matrix.dtype != np.float32:\n        raise RebuildTrainingError(\n            "Training matrix dtype must be float32."\n        )\n\n    if not np.isfinite(\n        matrix.data\n    ).all():\n        raise RebuildTrainingError(\n            "Training matrix contains NaN or infinity."\n        )\n\n    return matrix\n\n\ndef _validate_training_target(\n    target: Sequence[int] | np.ndarray,\n    *,\n    expected_rows: int,\n) -> np.ndarray:\n    y = np.asarray(\n        target\n    )\n\n    if y.ndim != 1:\n        raise RebuildTrainingError(\n            "Training target must be one-dimensional."\n        )\n\n    if y.shape[0] != expected_rows:\n        raise RebuildTrainingError(\n            "Training target length must match matrix row count."\n        )\n\n    if y.dtype != EXPECTED_TARGET_DTYPE:\n        raise RebuildTrainingError(\n            "Training target dtype must be int8."\n        )\n\n    unique = np.unique(\n        y\n    )\n\n    if not np.all(\n        np.isin(\n            unique,\n            np.asarray(\n                [0, 1],\n                dtype=np.int8,\n            ),\n        )\n    ):\n        raise RebuildTrainingError(\n            "Training target must contain only labels 0 and 1."\n        )\n\n    if unique.tolist() != [0, 1]:\n        raise RebuildTrainingError(\n            "Training target must contain both classes 0 and 1."\n        )\n\n    return y\n\n\ndef fit_rebuild_model(\n    matrix: sparse.spmatrix,\n    target: Sequence[int] | np.ndarray,\n) -> RebuildModelFit:\n    """Fit a new rebuild estimator in memory.\n\n    This function never writes files and never promotes the result to the\n    official artifact set.\n    """\n\n    X = _validate_training_matrix(\n        matrix\n    )\n\n    y = _validate_training_target(\n        target,\n        expected_rows=X.shape[0],\n    )\n\n    estimator = build_rebuild_model()\n\n    estimator.fit(\n        X,\n        y,\n    )\n\n    classes = tuple(\n        int(value)\n        for value in estimator.classes_.tolist()\n    )\n\n    if classes != (0, 1):\n        raise RebuildTrainingError(\n            "Fitted rebuild estimator classes must be exactly (0, 1)."\n        )\n\n    if int(\n        estimator.n_features_in_\n    ) != EXPECTED_FEATURE_COUNT:\n        raise RebuildTrainingError(\n            "Fitted rebuild estimator feature width mismatch."\n        )\n\n    if len(\n        estimator.estimators_\n    ) != 100:\n        raise RebuildTrainingError(\n            "Fitted rebuild estimator must contain exactly 100 trees."\n        )\n\n    actual_params = estimator.get_params(\n        deep=False\n    )\n\n    for key, expected_value in (\n        EXPECTED_MODEL_PARAMETERS.items()\n    ):\n        if actual_params[key] != expected_value:\n            raise RebuildTrainingError(\n                f"Fitted rebuild parameter mismatch: {key}."\n            )\n\n    return RebuildModelFit(\n        estimator=estimator,\n        model_id=REBUILD_MODEL_ID,\n        training_rows=int(\n            X.shape[0]\n        ),\n        fraud_rows=int(\n            np.count_nonzero(\n                y == 1\n            )\n        ),\n        feature_count=EXPECTED_FEATURE_COUNT,\n        classes=classes,\n        artifact_identity_status=\n            "NEW_REBUILD_MODEL",\n        evaluation_status=\n            "NOT_EVALUATED_AS_OFFICIAL",\n    )\n',
    "final_pipeline/tests/unit/test_rebuild_model_fit.py":
        'from __future__ import annotations\n\nimport unittest\n\nimport numpy as np\nfrom scipy import sparse\nfrom sklearn.ensemble import RandomForestClassifier\n\nfrom fraud_screening.artifacts import (\n    EXPECTED_MODEL_PARAMETERS,\n)\nfrom fraud_screening.training import (\n    REBUILD_MODEL_ID,\n    RebuildTrainingError,\n    fit_rebuild_model,\n)\n\n\ndef training_fixture() -> tuple[\n    sparse.csr_matrix,\n    np.ndarray,\n]:\n    rng = np.random.default_rng(\n        20260922\n    )\n\n    dense = rng.normal(\n        loc=0.0,\n        scale=1.0,\n        size=(40, 47),\n    ).astype(\n        np.float32\n    )\n\n    dense[\n        np.abs(dense) < 0.50\n    ] = 0.0\n\n    matrix = sparse.csr_matrix(\n        dense,\n        dtype=np.float32,\n    )\n\n    target = np.asarray(\n        [\n            1\n            if index % 5 == 0\n            else 0\n            for index in range(40)\n        ],\n        dtype=np.int8,\n    )\n\n    # Keep both classes present deterministically.\n    target[0] = 0\n    target[1] = 1\n\n    return matrix, target\n\n\nclass RebuildModelFitTests(unittest.TestCase):\n    def test_valid_fit_creates_learned_random_forest(self) -> None:\n        matrix, target = training_fixture()\n\n        result = fit_rebuild_model(\n            matrix,\n            target,\n        )\n\n        self.assertIsInstance(\n            result.estimator,\n            RandomForestClassifier,\n        )\n        self.assertTrue(\n            hasattr(\n                result.estimator,\n                "estimators_",\n            )\n        )\n        self.assertEqual(\n            len(\n                result.estimator.estimators_\n            ),\n            100,\n        )\n        self.assertEqual(\n            result.training_rows,\n            40,\n        )\n        self.assertEqual(\n            result.feature_count,\n            47,\n        )\n        self.assertEqual(\n            result.classes,\n            (0, 1),\n        )\n\n    def test_fit_metadata_does_not_claim_official_evaluation(self) -> None:\n        matrix, target = training_fixture()\n\n        result = fit_rebuild_model(\n            matrix,\n            target,\n        )\n\n        self.assertEqual(\n            result.model_id,\n            REBUILD_MODEL_ID,\n        )\n        self.assertEqual(\n            result.artifact_identity_status,\n            "NEW_REBUILD_MODEL",\n        )\n        self.assertEqual(\n            result.evaluation_status,\n            "NOT_EVALUATED_AS_OFFICIAL",\n        )\n\n    def test_fitted_model_keeps_guarded_parameters(self) -> None:\n        matrix, target = training_fixture()\n\n        result = fit_rebuild_model(\n            matrix,\n            target,\n        )\n\n        actual = (\n            result.estimator\n            .get_params(\n                deep=False\n            )\n        )\n\n        for key, expected in (\n            EXPECTED_MODEL_PARAMETERS.items()\n        ):\n            self.assertEqual(\n                actual[key],\n                expected,\n                msg=key,\n            )\n\n    def test_dense_matrix_is_rejected(self) -> None:\n        matrix, target = training_fixture()\n\n        with self.assertRaises(\n            RebuildTrainingError\n        ):\n            fit_rebuild_model(\n                matrix.toarray(),\n                target,\n            )\n\n    def test_wrong_width_is_rejected(self) -> None:\n        matrix, target = training_fixture()\n\n        with self.assertRaises(\n            RebuildTrainingError\n        ):\n            fit_rebuild_model(\n                matrix[:, :46],\n                target,\n            )\n\n    def test_wrong_matrix_dtype_is_rejected(self) -> None:\n        matrix, target = training_fixture()\n\n        with self.assertRaises(\n            RebuildTrainingError\n        ):\n            fit_rebuild_model(\n                matrix.astype(\n                    np.float64\n                ),\n                target,\n            )\n\n    def test_nonfinite_matrix_is_rejected(self) -> None:\n        matrix, target = training_fixture()\n\n        bad = matrix.copy()\n        bad.data[0] = np.inf\n\n        with self.assertRaises(\n            RebuildTrainingError\n        ):\n            fit_rebuild_model(\n                bad,\n                target,\n            )\n\n    def test_empty_matrix_is_rejected(self) -> None:\n        matrix = sparse.csr_matrix(\n            (0, 47),\n            dtype=np.float32,\n        )\n        target = np.asarray(\n            [],\n            dtype=np.int8,\n        )\n\n        with self.assertRaises(\n            RebuildTrainingError\n        ):\n            fit_rebuild_model(\n                matrix,\n                target,\n            )\n\n    def test_target_dtype_must_be_int8(self) -> None:\n        matrix, target = training_fixture()\n\n        with self.assertRaises(\n            RebuildTrainingError\n        ):\n            fit_rebuild_model(\n                matrix,\n                target.astype(\n                    np.int64\n                ),\n            )\n\n    def test_target_values_must_be_binary(self) -> None:\n        matrix, target = training_fixture()\n\n        bad = target.copy()\n        bad[0] = 2\n\n        with self.assertRaises(\n            RebuildTrainingError\n        ):\n            fit_rebuild_model(\n                matrix,\n                bad,\n            )\n\n    def test_both_classes_are_required(self) -> None:\n        matrix, _ = training_fixture()\n\n        target = np.zeros(\n            matrix.shape[0],\n            dtype=np.int8,\n        )\n\n        with self.assertRaises(\n            RebuildTrainingError\n        ):\n            fit_rebuild_model(\n                matrix,\n                target,\n            )\n\n    def test_target_length_must_match_rows(self) -> None:\n        matrix, target = training_fixture()\n\n        with self.assertRaises(\n            RebuildTrainingError\n        ):\n            fit_rebuild_model(\n                matrix,\n                target[:-1],\n            )\n\n\nif __name__ == "__main__":\n    unittest.main()\n',
}


def detect_root() -> Path:
    root = Path.cwd().resolve()

    required = [
        root / "research",
        root / "final_pipeline",
        root / "application",
        root
        / "final_pipeline/src/fraud_screening/training/factory.py",
        root
        / "final_pipeline/src/fraud_screening/training/preprocessing_fit.py",
        root
        / "final_pipeline/src/fraud_screening/training/outputs.py",
        root
        / "final_pipeline/artifacts/official/model/model.joblib",
        root
        / "final_pipeline/artifacts/official/preprocessing/preprocessing_state.json",
        root
        / "final_pipeline/artifacts/official/manifest/artifact_manifest.json",
    ]

    if not all(path.exists() for path in required):
        raise RuntimeError(
            "Repository foundation or official artifacts are incomplete."
        )

    return root


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Create in-memory rebuild model trainer and tests. "
            "Source creation itself does not fit a model."
        ),
    )

    args = parser.parse_args()

    root = detect_root()
    init_path = root / INIT_REL

    missing_dependencies = [
        relative
        for relative in [
            "final_pipeline/src/fraud_screening/training/factory.py",
            "final_pipeline/src/fraud_screening/training/preprocessing_fit.py",
            "final_pipeline/src/fraud_screening/training/outputs.py",
        ]
        if not (root / relative).is_file()
    ]

    collisions = [
        relative
        for relative in NEW_FILES
        if (root / relative).exists()
    ]

    init_matches = (
        init_path.is_file()
        and init_path.read_text(
            encoding="utf-8"
        )
        == EXPECTED_INIT
    )

    print("=" * 92)
    print(
        "REBUILD MODEL TRAINER — IN-MEMORY FIT CONTRACT"
    )
    print("=" * 92)
    print("Project root:", root)
    print(
        "Mode:",
        "EXECUTE — GUARDED SOURCE UPDATE"
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

    print("\n[2] training/__init__.py guard")
    print(
        " - exact expected current content:",
        "YES" if init_matches else "NO",
    )

    print("\n[3] Planned new files")
    for relative in NEW_FILES:
        print(" -", relative)

    print("\n[4] Collision gate")
    print(
        " - collisions:",
        len(collisions),
    )
    for relative in collisions:
        print("   *", relative)

    print("\n[5] Training input contract")
    print(" - matrix format: CSR")
    print(" - matrix dtype: float32")
    print(" - feature width: 47")
    print(" - target dtype: int8")
    print(" - target values: exactly binary 0/1")
    print(" - both classes required: YES")

    print("\n[6] Fitted-model contract")
    print(" - estimator: RandomForestClassifier")
    print(" - model config: frozen final 12-parameter set")
    print(" - trees after fit: 100")
    print(" - classes after fit: [0, 1]")
    print(" - n_features_in_: 47")
    print(" - artifact identity: NEW_REBUILD_MODEL")
    print(" - official evaluation status: NOT_EVALUATED_AS_OFFICIAL")

    print("\n[7] Side-effect boundary")
    print(" - dry-run model.fit(): NO")
    print(" - execute model.fit(): NO")
    print(
        " - unit tests after execute: YES, small synthetic fixture in RAM"
    )
    print(" - predict()/predict_proba(): NO")
    print(" - model artifact write: NO")
    print(" - preprocessing artifact write: NO")
    print(" - official artifact modification: NO")
    print(" - research modification: NO")

    if (
        missing_dependencies
        or collisions
        or not init_matches
    ):
        print("\nRESULT: STOP")
        print("No files were written.")
        raise SystemExit(1)

    if not args.execute:
        print("\nDRY-RUN RESULT: PASS")
        print("Execute with:")
        print(
            "python bootstrap_rebuild_model_fit.py --execute"
        )
        print("=" * 92)
        return

    created = []
    original_init = init_path.read_text(
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

        init_path.write_text(
            UPDATED_INIT,
            encoding="utf-8",
        )

    except Exception:
        for path in reversed(created):
            try:
                path.unlink()
            except OSError:
                pass

        try:
            init_path.write_text(
                original_init,
                encoding="utf-8",
            )
        except OSError:
            pass

        raise

    print("\n[8] Created")
    for path in created:
        print(
            " -",
            path.relative_to(root),
        )

    print("\n[9] Updated")
    print(" -", INIT_REL)

    print("\nBOOTSTRAP RESULT: PASS")
    print(
        "Source was created without model fitting."
    )
    print(
        "The next unit-suite run will intentionally call model.fit() "
        "on a small synthetic 40x47 CSR float32 fixture only."
    )
    print(
        "PYTHONPATH=final_pipeline/src "
        "python -m unittest discover "
        "-s final_pipeline/tests -v"
    )
    print("=" * 92)


if __name__ == "__main__":
    main()
