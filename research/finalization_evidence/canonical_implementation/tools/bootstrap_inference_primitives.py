from __future__ import annotations

import argparse
import sys
from pathlib import Path

NEW_FILES = {
    "final_pipeline/src/fraud_screening/inference/__init__.py":
        '"""Stable inference primitives for fraud-risk screening."""\n\nfrom fraud_screening.inference.scoring import (\n    DEFAULT_THRESHOLD,\n    ScreeningBatchResult,\n    apply_screening_threshold,\n    score_encoded_matrix,\n)\n\n__all__ = [\n    "DEFAULT_THRESHOLD",\n    "ScreeningBatchResult",\n    "apply_screening_threshold",\n    "score_encoded_matrix",\n]\n',
    "final_pipeline/src/fraud_screening/inference/scoring.py":
        '"""Frozen model scoring and threshold application."""\n\nfrom __future__ import annotations\n\nfrom dataclasses import dataclass\n\nimport numpy as np\nfrom scipy import sparse\n\nfrom fraud_screening.artifacts import (\n    VerifiedModelArtifact,\n)\nfrom fraud_screening.errors import (\n    InferenceContractError,\n)\n\n\nDEFAULT_THRESHOLD = 0.50\nEXPECTED_ENCODED_WIDTH = 47\n\n\n@dataclass(frozen=True, slots=True)\nclass ScreeningBatchResult:\n    """Stable screening output for one encoded batch."""\n\n    risk_score: np.ndarray\n    threshold: float\n    screening_prediction: np.ndarray\n    model_id: str\n\n    def __post_init__(self) -> None:\n        risk_score = np.asarray(\n            self.risk_score\n        )\n        prediction = np.asarray(\n            self.screening_prediction\n        )\n\n        if risk_score.ndim != 1:\n            raise InferenceContractError(\n                "risk_score must be one-dimensional."\n            )\n\n        if prediction.ndim != 1:\n            raise InferenceContractError(\n                "screening_prediction must be one-dimensional."\n            )\n\n        if risk_score.shape != prediction.shape:\n            raise InferenceContractError(\n                "risk_score and screening_prediction lengths must match."\n            )\n\n        if risk_score.dtype != np.float32:\n            raise InferenceContractError(\n                "risk_score dtype must be float32."\n            )\n\n        if prediction.dtype != np.int8:\n            raise InferenceContractError(\n                "screening_prediction dtype must be int8."\n            )\n\n        if not np.isfinite(risk_score).all():\n            raise InferenceContractError(\n                "risk_score must contain only finite values."\n            )\n\n        if not (\n            np.all(risk_score >= 0.0)\n            and np.all(risk_score <= 1.0)\n        ):\n            raise InferenceContractError(\n                "risk_score must be within [0, 1]."\n            )\n\n        if not np.isin(\n            prediction,\n            [0, 1],\n        ).all():\n            raise InferenceContractError(\n                "screening_prediction must contain only 0 or 1."\n            )\n\n        if self.threshold != DEFAULT_THRESHOLD:\n            raise InferenceContractError(\n                "threshold must equal the frozen value 0.50."\n            )\n\n        if not isinstance(\n            self.model_id,\n            str,\n        ) or not self.model_id:\n            raise InferenceContractError(\n                "model_id must be a non-empty string."\n            )\n\n\ndef apply_screening_threshold(\n    risk_score: np.ndarray,\n    *,\n    threshold: float = DEFAULT_THRESHOLD,\n) -> np.ndarray:\n    """Apply the frozen strict comparator risk_score > 0.50."""\n\n    if threshold != DEFAULT_THRESHOLD:\n        raise InferenceContractError(\n            "Only the frozen threshold 0.50 is allowed."\n        )\n\n    score = np.asarray(\n        risk_score,\n        dtype=np.float32,\n    )\n\n    if score.ndim != 1:\n        raise InferenceContractError(\n            "risk_score must be one-dimensional."\n        )\n\n    if not np.isfinite(score).all():\n        raise InferenceContractError(\n            "risk_score contains NaN or infinity."\n        )\n\n    if not (\n        np.all(score >= 0.0)\n        and np.all(score <= 1.0)\n    ):\n        raise InferenceContractError(\n            "risk_score must be within [0, 1]."\n        )\n\n    return (\n        score > threshold\n    ).astype(\n        np.int8,\n        copy=False,\n    )\n\n\ndef _validate_encoded_matrix(\n    matrix: sparse.spmatrix,\n) -> sparse.csr_matrix:\n    if not sparse.issparse(matrix):\n        raise InferenceContractError(\n            "Encoded input must be a scipy sparse matrix."\n        )\n\n    encoded = matrix.tocsr(\n        copy=False\n    )\n\n    if encoded.ndim != 2:\n        raise InferenceContractError(\n            "Encoded input must be two-dimensional."\n        )\n\n    if encoded.shape[1] != EXPECTED_ENCODED_WIDTH:\n        raise InferenceContractError(\n            "Encoded input width must be exactly 47."\n        )\n\n    if encoded.dtype != np.float32:\n        raise InferenceContractError(\n            "Encoded input dtype must be float32."\n        )\n\n    if not np.isfinite(\n        encoded.data\n    ).all():\n        raise InferenceContractError(\n            "Encoded input contains NaN or infinity."\n        )\n\n    return encoded\n\n\ndef score_encoded_matrix(\n    matrix: sparse.spmatrix,\n    model: VerifiedModelArtifact,\n) -> ScreeningBatchResult:\n    """Run frozen predict_proba and strict thresholding."""\n\n    if not isinstance(\n        model,\n        VerifiedModelArtifact,\n    ):\n        raise InferenceContractError(\n            "model must be a VerifiedModelArtifact."\n        )\n\n    encoded = _validate_encoded_matrix(\n        matrix\n    )\n\n    probability_matrix = (\n        model.estimator.predict_proba(\n            encoded\n        )\n    )\n\n    probabilities = np.asarray(\n        probability_matrix\n    )\n\n    if probabilities.ndim != 2:\n        raise InferenceContractError(\n            "predict_proba output must be two-dimensional."\n        )\n\n    if probabilities.shape != (\n        encoded.shape[0],\n        len(model.classes),\n    ):\n        raise InferenceContractError(\n            "predict_proba output shape does not match verified classes."\n        )\n\n    if not np.isfinite(\n        probabilities\n    ).all():\n        raise InferenceContractError(\n            "predict_proba output contains NaN or infinity."\n        )\n\n    risk_score = (\n        probabilities[\n            :,\n            model.positive_class_index,\n        ]\n        .astype(\n            np.float32,\n            copy=False,\n        )\n    )\n\n    if not (\n        np.all(risk_score >= 0.0)\n        and np.all(risk_score <= 1.0)\n    ):\n        raise InferenceContractError(\n            "Positive-class risk score is outside [0, 1]."\n        )\n\n    prediction = apply_screening_threshold(\n        risk_score\n    )\n\n    return ScreeningBatchResult(\n        risk_score=risk_score,\n        threshold=DEFAULT_THRESHOLD,\n        screening_prediction=prediction,\n        model_id=model.model_id,\n    )\n',
    "final_pipeline/tests/unit/test_inference_primitives.py":
        'from __future__ import annotations\n\nimport unittest\nfrom unittest.mock import Mock\n\nimport numpy as np\nfrom scipy import sparse\n\nfrom fraud_screening.artifacts import (\n    MODEL_ID,\n    VerifiedModelArtifact,\n)\nfrom fraud_screening.errors import (\n    InferenceContractError,\n)\nfrom fraud_screening.inference import (\n    DEFAULT_THRESHOLD,\n    ScreeningBatchResult,\n    apply_screening_threshold,\n    score_encoded_matrix,\n)\n\n\ndef verified_model_fixture(\n    probabilities: np.ndarray,\n) -> VerifiedModelArtifact:\n    estimator = Mock()\n    estimator.predict_proba.return_value = (\n        probabilities\n    )\n\n    return VerifiedModelArtifact(\n        estimator=estimator,\n        sha256="fixture",\n        model_id=MODEL_ID,\n        classes=(0, 1),\n        positive_class_index=1,\n    )\n\n\ndef encoded_rows(\n    count: int,\n) -> sparse.csr_matrix:\n    return sparse.csr_matrix(\n        np.zeros(\n            (count, 47),\n            dtype=np.float32,\n        )\n    )\n\n\nclass InferencePrimitiveTests(unittest.TestCase):\n    def test_frozen_threshold_constant(self) -> None:\n        self.assertEqual(\n            DEFAULT_THRESHOLD,\n            0.50,\n        )\n\n    def test_strict_boundary_examples(self) -> None:\n        score = np.asarray(\n            [0.49, 0.50, 0.51],\n            dtype=np.float32,\n        )\n\n        prediction = apply_screening_threshold(\n            score\n        )\n\n        np.testing.assert_array_equal(\n            prediction,\n            np.asarray(\n                [0, 0, 1],\n                dtype=np.int8,\n            ),\n        )\n\n    def test_threshold_override_is_rejected(self) -> None:\n        with self.assertRaises(\n            InferenceContractError\n        ):\n            apply_screening_threshold(\n                np.asarray(\n                    [0.9],\n                    dtype=np.float32,\n                ),\n                threshold=0.6,\n            )\n\n    def test_score_uses_positive_class_index(self) -> None:\n        probabilities = np.asarray(\n            [\n                [0.9, 0.1],\n                [0.2, 0.8],\n            ],\n            dtype=np.float64,\n        )\n\n        model = verified_model_fixture(\n            probabilities\n        )\n\n        result = score_encoded_matrix(\n            encoded_rows(2),\n            model,\n        )\n\n        np.testing.assert_allclose(\n            result.risk_score,\n            np.asarray(\n                [0.1, 0.8],\n                dtype=np.float32,\n            ),\n        )\n\n        model.estimator.predict_proba.assert_called_once()\n\n    def test_risk_score_is_float32(self) -> None:\n        model = verified_model_fixture(\n            np.asarray(\n                [[0.2, 0.8]],\n                dtype=np.float64,\n            )\n        )\n\n        result = score_encoded_matrix(\n            encoded_rows(1),\n            model,\n        )\n\n        self.assertEqual(\n            result.risk_score.dtype,\n            np.float32,\n        )\n\n    def test_prediction_is_int8(self) -> None:\n        model = verified_model_fixture(\n            np.asarray(\n                [[0.2, 0.8]],\n                dtype=np.float64,\n            )\n        )\n\n        result = score_encoded_matrix(\n            encoded_rows(1),\n            model,\n        )\n\n        self.assertEqual(\n            result.screening_prediction.dtype,\n            np.int8,\n        )\n\n    def test_output_contains_frozen_threshold_and_model_id(self) -> None:\n        model = verified_model_fixture(\n            np.asarray(\n                [[0.2, 0.8]],\n                dtype=np.float64,\n            )\n        )\n\n        result = score_encoded_matrix(\n            encoded_rows(1),\n            model,\n        )\n\n        self.assertEqual(\n            result.threshold,\n            0.50,\n        )\n        self.assertEqual(\n            result.model_id,\n            MODEL_ID,\n        )\n\n    def test_encoded_width_must_be_47(self) -> None:\n        model = verified_model_fixture(\n            np.asarray(\n                [[0.2, 0.8]],\n                dtype=np.float64,\n            )\n        )\n\n        wrong = sparse.csr_matrix(\n            np.zeros(\n                (1, 46),\n                dtype=np.float32,\n            )\n        )\n\n        with self.assertRaises(\n            InferenceContractError\n        ):\n            score_encoded_matrix(\n                wrong,\n                model,\n            )\n\n    def test_encoded_dtype_must_be_float32(self) -> None:\n        model = verified_model_fixture(\n            np.asarray(\n                [[0.2, 0.8]],\n                dtype=np.float64,\n            )\n        )\n\n        wrong = sparse.csr_matrix(\n            np.zeros(\n                (1, 47),\n                dtype=np.float64,\n            )\n        )\n\n        with self.assertRaises(\n            InferenceContractError\n        ):\n            score_encoded_matrix(\n                wrong,\n                model,\n            )\n\n    def test_dense_input_is_rejected(self) -> None:\n        model = verified_model_fixture(\n            np.asarray(\n                [[0.2, 0.8]],\n                dtype=np.float64,\n            )\n        )\n\n        with self.assertRaises(\n            InferenceContractError\n        ):\n            score_encoded_matrix(\n                np.zeros(\n                    (1, 47),\n                    dtype=np.float32,\n                ),\n                model,\n            )\n\n    def test_probability_shape_mismatch_is_rejected(self) -> None:\n        model = verified_model_fixture(\n            np.asarray(\n                [[0.2, 0.3, 0.5]],\n                dtype=np.float64,\n            )\n        )\n\n        with self.assertRaises(\n            InferenceContractError\n        ):\n            score_encoded_matrix(\n                encoded_rows(1),\n                model,\n            )\n\n    def test_non_finite_probability_is_rejected(self) -> None:\n        model = verified_model_fixture(\n            np.asarray(\n                [[0.2, np.nan]],\n                dtype=np.float64,\n            )\n        )\n\n        with self.assertRaises(\n            InferenceContractError\n        ):\n            score_encoded_matrix(\n                encoded_rows(1),\n                model,\n            )\n\n    def test_out_of_range_probability_is_rejected(self) -> None:\n        model = verified_model_fixture(\n            np.asarray(\n                [[-0.1, 1.1]],\n                dtype=np.float64,\n            )\n        )\n\n        with self.assertRaises(\n            InferenceContractError\n        ):\n            score_encoded_matrix(\n                encoded_rows(1),\n                model,\n            )\n\n    def test_batch_output_lengths_match_input(self) -> None:\n        model = verified_model_fixture(\n            np.asarray(\n                [\n                    [0.9, 0.1],\n                    [0.5, 0.5],\n                    [0.49, 0.51],\n                ],\n                dtype=np.float64,\n            )\n        )\n\n        result = score_encoded_matrix(\n            encoded_rows(3),\n            model,\n        )\n\n        self.assertEqual(\n            result.risk_score.shape,\n            (3,),\n        )\n        self.assertEqual(\n            result.screening_prediction.shape,\n            (3,),\n        )\n\n    def test_result_rejects_wrong_prediction_dtype(self) -> None:\n        with self.assertRaises(\n            InferenceContractError\n        ):\n            ScreeningBatchResult(\n                risk_score=np.asarray(\n                    [0.5],\n                    dtype=np.float32,\n                ),\n                threshold=0.50,\n                screening_prediction=np.asarray(\n                    [0],\n                    dtype=np.int64,\n                ),\n                model_id=MODEL_ID,\n            )\n\n\nif __name__ == "__main__":\n    unittest.main()\n',
}

MODEL_REL = Path(
    "research/data/processed/m8_02_final_test_artifact_audit/"
    "m8_02_selected_rf_estimator.joblib"
)

PREPROCESSING_REL = Path(
    "research/data/processed/m8_02_final_test_artifact_audit/"
    "m8_02_w_short_preprocessing_state.json"
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


def verify_real_artifacts(
    root: Path,
) -> None:
    model_path = root / MODEL_REL
    preprocessing_path = (
        root / PREPROCESSING_REL
    )

    print("=" * 90)
    print("FINAL PIPELINE — REAL ARTIFACT INFERENCE PRIMITIVE VERIFICATION")
    print("=" * 90)
    print("Mode: DETERMINISTIC TRANSFORM + PREDICT_PROBA")
    print(" - model fit: NO")
    print(" - preprocessing fit: NO")
    print(" - predict(): NO")
    print(" - predict_proba(): YES")
    print(" - threshold retuning: NO")
    print(" - file write: NO")

    if not model_path.is_file():
        raise FileNotFoundError(
            model_path
        )

    if not preprocessing_path.is_file():
        raise FileNotFoundError(
            preprocessing_path
        )

    src_path = root / "final_pipeline/src"
    sys.path.insert(
        0,
        str(src_path),
    )

    try:
        import numpy as np

        from fraud_screening.artifacts import (
            load_verified_model,
        )
        from fraud_screening.features import (
            SemanticFeatureRow,
        )
        from fraud_screening.inference import (
            apply_screening_threshold,
            score_encoded_matrix,
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

        semantic_rows = [
            SemanticFeatureRow(
                amount_numeric=10.0,
                time_since_previous_transaction_min=
                    None,
                transactions_last_1h=0,
                amount_minus_previous_mean=
                    None,
                is_new_merchant=True,
                has_prior_card_history=False,
                transaction_mode=
                    "Chip Transaction",
                location_state=
                    "PHYSICAL_COMPLETE",
                hour_of_day="9",
                day_of_week="0",
            ),
            SemanticFeatureRow(
                amount_numeric=100.0,
                time_since_previous_transaction_min=
                    30.0,
                transactions_last_1h=2,
                amount_minus_previous_mean=
                    40.0,
                is_new_merchant=False,
                has_prior_card_history=True,
                transaction_mode=
                    "Online Transaction",
                location_state=
                    "NON_PHYSICAL_OR_ONLINE",
                hour_of_day="14",
                day_of_week="5",
            ),
        ]

        encoded = preprocessor.transform(
            semantic_rows
        )

        result = score_encoded_matrix(
            encoded,
            model,
        )

        boundary_prediction = (
            apply_screening_threshold(
                np.asarray(
                    [0.49, 0.50, 0.51],
                    dtype=np.float32,
                )
            )
        )

        gates = {
            "G01_ENCODED_SHAPE":
                encoded.shape == (2, 47),
            "G02_RISK_SCORE_SHAPE":
                result.risk_score.shape
                == (2,),
            "G03_RISK_SCORE_FLOAT32":
                result.risk_score.dtype
                == np.float32,
            "G04_RISK_SCORE_FINITE":
                bool(
                    np.isfinite(
                        result.risk_score
                    ).all()
                ),
            "G05_RISK_SCORE_RANGE":
                bool(
                    np.all(
                        result.risk_score
                        >= 0.0
                    )
                    and
                    np.all(
                        result.risk_score
                        <= 1.0
                    )
                ),
            "G06_PREDICTION_INT8":
                result.screening_prediction.dtype
                == np.int8,
            "G07_MODEL_ID":
                result.model_id
                == "RF-REF-100-GINI-SQRT-UNPRUNED-CW",
            "G08_THRESHOLD_050":
                result.threshold == 0.50,
            "G09_STRICT_BOUNDARY":
                boundary_prediction.tolist()
                == [0, 0, 1],
        }

        print("\n[1] Encoded input")
        print(" - shape:", encoded.shape)
        print(" - dtype:", encoded.dtype)

        print("\n[2] Risk score")
        print(
            " - values:",
            [
                float(value)
                for value in result.risk_score
            ],
        )
        print(
            " - dtype:",
            result.risk_score.dtype,
        )

        print("\n[3] Screening prediction")
        print(
            " - values:",
            result.screening_prediction.tolist(),
        )
        print(
            " - threshold:",
            result.threshold,
        )
        print(
            " - model id:",
            result.model_id,
        )

        print("\n[4] Boundary fixture")
        print(
            " - score:",
            [0.49, 0.50, 0.51],
        )
        print(
            " - prediction:",
            boundary_prediction.tolist(),
        )

        print("\n[5] Verification gates")
        for name, passed in (
            gates.items()
        ):
            print(
                f" - {name}: "
                f"{'PASS' if passed else 'FAIL'}"
            )

        if not all(gates.values()):
            print(
                "\nVERIFICATION RESULT: FAIL"
            )
            raise SystemExit(1)

        print(
            "\nVERIFICATION RESULT: PASS"
        )
        print(
            "Frozen preprocessing + frozen model "
            "produced valid risk scores."
        )
        print(
            "Threshold semantics verified: "
            "0.49 -> 0, 0.50 -> 0, 0.51 -> 1."
        )
        print(
            "No fit, no predict(), no retuning, "
            "no file write."
        )
        print("=" * 90)

    finally:
        try:
            sys.path.remove(
                str(src_path)
            )
        except ValueError:
            pass


def main() -> None:
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group()

    group.add_argument(
        "--execute",
        action="store_true",
        help="Create inference primitive source and tests.",
    )

    group.add_argument(
        "--verify-artifacts",
        action="store_true",
        help="Verify inference primitives using frozen research artifacts.",
    )

    args = parser.parse_args()
    root = detect_root()

    if args.verify_artifacts:
        verify_real_artifacts(
            root
        )
        return

    dependencies = [
        root / "final_pipeline/src/fraud_screening/artifacts/model_loader.py",
        root / "final_pipeline/src/fraud_screening/preprocessing/frozen.py",
        root / "final_pipeline/src/fraud_screening/errors.py",
        root / "final_pipeline/tests/unit/test_model_artifact_loader.py",
        root / "final_pipeline/tests/unit/test_frozen_preprocessing.py",
    ]

    missing_dependencies = [
        str(
            path.relative_to(root)
        )
        for path in dependencies
        if not path.is_file()
    ]

    collisions = [
        relative
        for relative in NEW_FILES
        if (root / relative).exists()
    ]

    print("=" * 90)
    print("FINAL PIPELINE — INFERENCE PRIMITIVE BOOTSTRAP")
    print("=" * 90)
    print(
        "Project root:",
        root,
    )
    print(
        "Mode:",
        "EXECUTE — NEW FILES ONLY"
        if args.execute
        else "DRY RUN — NO FILES WRITTEN",
    )

    print("\n[1] Dependency gate")
    print(
        " - missing project dependencies:",
        len(
            missing_dependencies
        ),
    )
    for relative in (
        missing_dependencies
    ):
        print(
            "   *",
            relative,
        )

    print("\n[2] Planned new files")
    for relative in NEW_FILES:
        print(
            " -",
            relative,
        )

    print("\n[3] Collision gate")
    print(
        " - collisions:",
        len(collisions),
    )
    for relative in collisions:
        print(
            "   *",
            relative,
        )

    if (
        missing_dependencies
        or collisions
    ):
        print("\nRESULT: STOP")
        print(
            "No files were written."
        )
        raise SystemExit(1)

    print("\n[4] Frozen inference contract")
    print(
        " - probability interface: predict_proba"
    )
    print(
        " - positive class index: verified model artifact index"
    )
    print(
        " - risk_score dtype: float32"
    )
    print(
        " - risk_score range: [0, 1]"
    )
    print(
        " - threshold: 0.50"
    )
    print(
        " - comparator: >"
    )
    print(
        " - 0.49 -> 0"
    )
    print(
        " - 0.50 -> 0"
    )
    print(
        " - 0.51 -> 1"
    )
    print(
        " - prediction dtype: int8"
    )
    print(
        " - predict() usage: FORBIDDEN"
    )

    print("\n[5] Scientific/runtime boundary")
    print(
        " - model fit: NO"
    )
    print(
        " - preprocessing fit: NO"
    )
    print(
        " - threshold tuning: NO"
    )
    print(
        " - calibration: NO"
    )
    print(
        " - artifact promotion: NO"
    )
    print(
        " - research files modified: NO"
    )

    if not args.execute:
        print(
            "\nDRY-RUN RESULT: PASS"
        )
        print(
            "Execute with:"
        )
        print(
            "python bootstrap_inference_primitives.py --execute"
        )
        print("=" * 90)
        return

    created = []

    try:
        for (
            relative,
            content,
        ) in NEW_FILES.items():
            path = root / relative

            path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            path.write_text(
                content,
                encoding="utf-8",
            )

            created.append(
                path
            )

    except Exception:
        for path in reversed(
            created
        ):
            try:
                path.unlink()
            except OSError:
                pass

        raise

    print("\n[6] Created")
    for path in created:
        print(
            " -",
            path.relative_to(
                root
            ),
        )

    print(
        "\nBOOTSTRAP RESULT: PASS"
    )
    print(
        "Run full unit suite:"
    )
    print(
        "PYTHONPATH=final_pipeline/src "
        "python -m unittest discover "
        "-s final_pipeline/tests -v"
    )
    print(
        "Then verify with frozen artifacts:"
    )
    print(
        "python bootstrap_inference_primitives.py --verify-artifacts"
    )
    print("=" * 90)


if __name__ == "__main__":
    main()
