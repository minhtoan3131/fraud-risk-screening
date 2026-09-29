from __future__ import annotations

import unittest
from unittest.mock import Mock

import numpy as np
from scipy import sparse

from fraud_screening.artifacts import (
    MODEL_ID,
    VerifiedModelArtifact,
)
from fraud_screening.errors import (
    InferenceContractError,
)
from fraud_screening.inference import (
    DEFAULT_THRESHOLD,
    ScreeningBatchResult,
    apply_screening_threshold,
    score_encoded_matrix,
)


def verified_model_fixture(
    probabilities: np.ndarray,
) -> VerifiedModelArtifact:
    estimator = Mock()
    estimator.predict_proba.return_value = (
        probabilities
    )

    return VerifiedModelArtifact(
        estimator=estimator,
        sha256="fixture",
        model_id=MODEL_ID,
        classes=(0, 1),
        positive_class_index=1,
    )


def encoded_rows(
    count: int,
) -> sparse.csr_matrix:
    return sparse.csr_matrix(
        np.zeros(
            (count, 47),
            dtype=np.float32,
        )
    )


class InferencePrimitiveTests(unittest.TestCase):
    def test_frozen_threshold_constant(self) -> None:
        self.assertEqual(
            DEFAULT_THRESHOLD,
            0.50,
        )

    def test_strict_boundary_examples(self) -> None:
        score = np.asarray(
            [0.49, 0.50, 0.51],
            dtype=np.float32,
        )

        prediction = apply_screening_threshold(
            score
        )

        np.testing.assert_array_equal(
            prediction,
            np.asarray(
                [0, 0, 1],
                dtype=np.int8,
            ),
        )

    def test_threshold_override_is_rejected(self) -> None:
        with self.assertRaises(
            InferenceContractError
        ):
            apply_screening_threshold(
                np.asarray(
                    [0.9],
                    dtype=np.float32,
                ),
                threshold=0.6,
            )

    def test_score_uses_positive_class_index(self) -> None:
        probabilities = np.asarray(
            [
                [0.9, 0.1],
                [0.2, 0.8],
            ],
            dtype=np.float64,
        )

        model = verified_model_fixture(
            probabilities
        )

        result = score_encoded_matrix(
            encoded_rows(2),
            model,
        )

        np.testing.assert_allclose(
            result.risk_score,
            np.asarray(
                [0.1, 0.8],
                dtype=np.float32,
            ),
        )

        model.estimator.predict_proba.assert_called_once()

    def test_risk_score_is_float32(self) -> None:
        model = verified_model_fixture(
            np.asarray(
                [[0.2, 0.8]],
                dtype=np.float64,
            )
        )

        result = score_encoded_matrix(
            encoded_rows(1),
            model,
        )

        self.assertEqual(
            result.risk_score.dtype,
            np.float32,
        )

    def test_prediction_is_int8(self) -> None:
        model = verified_model_fixture(
            np.asarray(
                [[0.2, 0.8]],
                dtype=np.float64,
            )
        )

        result = score_encoded_matrix(
            encoded_rows(1),
            model,
        )

        self.assertEqual(
            result.screening_prediction.dtype,
            np.int8,
        )

    def test_output_contains_frozen_threshold_and_model_id(self) -> None:
        model = verified_model_fixture(
            np.asarray(
                [[0.2, 0.8]],
                dtype=np.float64,
            )
        )

        result = score_encoded_matrix(
            encoded_rows(1),
            model,
        )

        self.assertEqual(
            result.threshold,
            0.50,
        )
        self.assertEqual(
            result.model_id,
            MODEL_ID,
        )

    def test_encoded_width_must_be_47(self) -> None:
        model = verified_model_fixture(
            np.asarray(
                [[0.2, 0.8]],
                dtype=np.float64,
            )
        )

        wrong = sparse.csr_matrix(
            np.zeros(
                (1, 46),
                dtype=np.float32,
            )
        )

        with self.assertRaises(
            InferenceContractError
        ):
            score_encoded_matrix(
                wrong,
                model,
            )

    def test_encoded_dtype_must_be_float32(self) -> None:
        model = verified_model_fixture(
            np.asarray(
                [[0.2, 0.8]],
                dtype=np.float64,
            )
        )

        wrong = sparse.csr_matrix(
            np.zeros(
                (1, 47),
                dtype=np.float64,
            )
        )

        with self.assertRaises(
            InferenceContractError
        ):
            score_encoded_matrix(
                wrong,
                model,
            )

    def test_dense_input_is_rejected(self) -> None:
        model = verified_model_fixture(
            np.asarray(
                [[0.2, 0.8]],
                dtype=np.float64,
            )
        )

        with self.assertRaises(
            InferenceContractError
        ):
            score_encoded_matrix(
                np.zeros(
                    (1, 47),
                    dtype=np.float32,
                ),
                model,
            )

    def test_probability_shape_mismatch_is_rejected(self) -> None:
        model = verified_model_fixture(
            np.asarray(
                [[0.2, 0.3, 0.5]],
                dtype=np.float64,
            )
        )

        with self.assertRaises(
            InferenceContractError
        ):
            score_encoded_matrix(
                encoded_rows(1),
                model,
            )

    def test_non_finite_probability_is_rejected(self) -> None:
        model = verified_model_fixture(
            np.asarray(
                [[0.2, np.nan]],
                dtype=np.float64,
            )
        )

        with self.assertRaises(
            InferenceContractError
        ):
            score_encoded_matrix(
                encoded_rows(1),
                model,
            )

    def test_out_of_range_probability_is_rejected(self) -> None:
        model = verified_model_fixture(
            np.asarray(
                [[-0.1, 1.1]],
                dtype=np.float64,
            )
        )

        with self.assertRaises(
            InferenceContractError
        ):
            score_encoded_matrix(
                encoded_rows(1),
                model,
            )

    def test_batch_output_lengths_match_input(self) -> None:
        model = verified_model_fixture(
            np.asarray(
                [
                    [0.9, 0.1],
                    [0.5, 0.5],
                    [0.49, 0.51],
                ],
                dtype=np.float64,
            )
        )

        result = score_encoded_matrix(
            encoded_rows(3),
            model,
        )

        self.assertEqual(
            result.risk_score.shape,
            (3,),
        )
        self.assertEqual(
            result.screening_prediction.shape,
            (3,),
        )

    def test_result_rejects_wrong_prediction_dtype(self) -> None:
        with self.assertRaises(
            InferenceContractError
        ):
            ScreeningBatchResult(
                risk_score=np.asarray(
                    [0.5],
                    dtype=np.float32,
                ),
                threshold=0.50,
                screening_prediction=np.asarray(
                    [0],
                    dtype=np.int64,
                ),
                model_id=MODEL_ID,
            )


if __name__ == "__main__":
    unittest.main()
