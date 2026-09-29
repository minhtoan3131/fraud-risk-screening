from __future__ import annotations

import unittest

import numpy as np
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier

from fraud_screening.artifacts import (
    EXPECTED_MODEL_PARAMETERS,
)
from fraud_screening.training import (
    REBUILD_MODEL_ID,
    RebuildTrainingError,
    fit_rebuild_model,
)


def training_fixture() -> tuple[
    sparse.csr_matrix,
    np.ndarray,
]:
    rng = np.random.default_rng(
        20260922
    )

    dense = rng.normal(
        loc=0.0,
        scale=1.0,
        size=(40, 47),
    ).astype(
        np.float32
    )

    dense[
        np.abs(dense) < 0.50
    ] = 0.0

    matrix = sparse.csr_matrix(
        dense,
        dtype=np.float32,
    )

    target = np.asarray(
        [
            1
            if index % 5 == 0
            else 0
            for index in range(40)
        ],
        dtype=np.int8,
    )

    # Keep both classes present deterministically.
    target[0] = 0
    target[1] = 1

    return matrix, target


class RebuildModelFitTests(unittest.TestCase):
    def test_valid_fit_creates_learned_random_forest(self) -> None:
        matrix, target = training_fixture()

        result = fit_rebuild_model(
            matrix,
            target,
        )

        self.assertIsInstance(
            result.estimator,
            RandomForestClassifier,
        )
        self.assertTrue(
            hasattr(
                result.estimator,
                "estimators_",
            )
        )
        self.assertEqual(
            len(
                result.estimator.estimators_
            ),
            100,
        )
        self.assertEqual(
            result.training_rows,
            40,
        )
        self.assertEqual(
            result.feature_count,
            47,
        )
        self.assertEqual(
            result.classes,
            (0, 1),
        )

    def test_fit_metadata_does_not_claim_official_evaluation(self) -> None:
        matrix, target = training_fixture()

        result = fit_rebuild_model(
            matrix,
            target,
        )

        self.assertEqual(
            result.model_id,
            REBUILD_MODEL_ID,
        )
        self.assertEqual(
            result.artifact_identity_status,
            "NEW_REBUILD_MODEL",
        )
        self.assertEqual(
            result.evaluation_status,
            "NOT_EVALUATED_AS_OFFICIAL",
        )

    def test_fitted_model_keeps_guarded_parameters(self) -> None:
        matrix, target = training_fixture()

        result = fit_rebuild_model(
            matrix,
            target,
        )

        actual = (
            result.estimator
            .get_params(
                deep=False
            )
        )

        for key, expected in (
            EXPECTED_MODEL_PARAMETERS.items()
        ):
            self.assertEqual(
                actual[key],
                expected,
                msg=key,
            )

    def test_dense_matrix_is_rejected(self) -> None:
        matrix, target = training_fixture()

        with self.assertRaises(
            RebuildTrainingError
        ):
            fit_rebuild_model(
                matrix.toarray(),
                target,
            )

    def test_wrong_width_is_rejected(self) -> None:
        matrix, target = training_fixture()

        with self.assertRaises(
            RebuildTrainingError
        ):
            fit_rebuild_model(
                matrix[:, :46],
                target,
            )

    def test_wrong_matrix_dtype_is_rejected(self) -> None:
        matrix, target = training_fixture()

        with self.assertRaises(
            RebuildTrainingError
        ):
            fit_rebuild_model(
                matrix.astype(
                    np.float64
                ),
                target,
            )

    def test_nonfinite_matrix_is_rejected(self) -> None:
        matrix, target = training_fixture()

        bad = matrix.copy()
        bad.data[0] = np.inf

        with self.assertRaises(
            RebuildTrainingError
        ):
            fit_rebuild_model(
                bad,
                target,
            )

    def test_empty_matrix_is_rejected(self) -> None:
        matrix = sparse.csr_matrix(
            (0, 47),
            dtype=np.float32,
        )
        target = np.asarray(
            [],
            dtype=np.int8,
        )

        with self.assertRaises(
            RebuildTrainingError
        ):
            fit_rebuild_model(
                matrix,
                target,
            )

    def test_target_dtype_must_be_int8(self) -> None:
        matrix, target = training_fixture()

        with self.assertRaises(
            RebuildTrainingError
        ):
            fit_rebuild_model(
                matrix,
                target.astype(
                    np.int64
                ),
            )

    def test_target_values_must_be_binary(self) -> None:
        matrix, target = training_fixture()

        bad = target.copy()
        bad[0] = 2

        with self.assertRaises(
            RebuildTrainingError
        ):
            fit_rebuild_model(
                matrix,
                bad,
            )

    def test_both_classes_are_required(self) -> None:
        matrix, _ = training_fixture()

        target = np.zeros(
            matrix.shape[0],
            dtype=np.int8,
        )

        with self.assertRaises(
            RebuildTrainingError
        ):
            fit_rebuild_model(
                matrix,
                target,
            )

    def test_target_length_must_match_rows(self) -> None:
        matrix, target = training_fixture()

        with self.assertRaises(
            RebuildTrainingError
        ):
            fit_rebuild_model(
                matrix,
                target[:-1],
            )


if __name__ == "__main__":
    unittest.main()
