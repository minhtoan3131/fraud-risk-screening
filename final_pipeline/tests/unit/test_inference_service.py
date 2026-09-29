from __future__ import annotations

import unittest

import numpy as np
from scipy import sparse

from fraud_screening.artifacts import (
    MODEL_ID,
    VerifiedModelArtifact,
)
from fraud_screening.inference.service import (
    InferenceService,
)


class _FixedEstimator:
    def __init__(
        self,
        probabilities: list[float],
    ) -> None:
        self._probabilities = np.asarray(
            probabilities,
            dtype=np.float64,
        )

    def predict_proba(
        self,
        matrix,
    ) -> np.ndarray:
        count = matrix.shape[0]

        positive = self._probabilities[
            :count
        ]

        return np.column_stack(
            [
                1.0 - positive,
                positive,
            ]
        )


class _RecordingPreprocessor:
    def __init__(self) -> None:
        self.rows = None

    def transform(
        self,
        rows,
    ):
        self.rows = tuple(
            rows
        )

        return sparse.csr_matrix(
            np.zeros(
                (
                    len(self.rows),
                    47,
                ),
                dtype=np.float32,
            )
        )


def verified_model(
    probabilities: list[float],
) -> VerifiedModelArtifact:
    return VerifiedModelArtifact(
        estimator=_FixedEstimator(
            probabilities
        ),
        sha256="x" * 64,
        model_id=MODEL_ID,
        classes=(0, 1),
        positive_class_index=1,
    )


def raw_record(
    *,
    time: str,
    amount: str,
    merchant: int,
) -> dict[str, object]:
    return {
        "User": 1,
        "Card": 0,
        "Year": 2018,
        "Month": 1,
        "Day": 1,
        "Time": time,
        "Amount": amount,
        "Use Chip": "Chip Transaction",
        "Merchant Name": merchant,
        "Merchant City": "A",
        "Merchant State": "CA",
        "Zip": 90001,
    }


class InferenceServiceTests(
    unittest.TestCase
):
    def test_raw_records_run_through_canonical_history_pipeline(self) -> None:
        preprocessor = _RecordingPreprocessor()

        service = InferenceService(
            model=verified_model(
                [0.10, 0.90]
            ),
            preprocessor=preprocessor,
        )

        result = service.score_raw_records(
            [
                raw_record(
                    time="09:00",
                    amount="$10.00",
                    merchant=100,
                ),
                raw_record(
                    time="10:00",
                    amount="$20.00",
                    merchant=200,
                ),
            ]
        )

        self.assertEqual(
            len(preprocessor.rows),
            2,
        )

        self.assertFalse(
            preprocessor.rows[
                0
            ].has_prior_card_history
        )

        self.assertTrue(
            preprocessor.rows[
                1
            ].has_prior_card_history
        )

        np.testing.assert_array_equal(
            result.screening_prediction,
            np.asarray(
                [0, 1],
                dtype=np.int8,
            ),
        )

        self.assertEqual(
            result.model_id,
            MODEL_ID,
        )

    def test_threshold_boundary_remains_strict(self) -> None:
        service = InferenceService(
            model=verified_model(
                [0.49, 0.50, 0.51]
            ),
            preprocessor=_RecordingPreprocessor(),
        )

        result = service.score_raw_records(
            [
                raw_record(
                    time="09:00",
                    amount="$10.00",
                    merchant=100,
                ),
                raw_record(
                    time="10:00",
                    amount="$20.00",
                    merchant=200,
                ),
                raw_record(
                    time="11:00",
                    amount="$30.00",
                    merchant=300,
                ),
            ]
        )

        np.testing.assert_array_equal(
            result.screening_prediction,
            np.asarray(
                [0, 0, 1],
                dtype=np.int8,
            ),
        )

        self.assertEqual(
            result.threshold,
            0.5,
        )

    def test_repeated_inference_is_deterministic(self) -> None:
        service = InferenceService(
            model=verified_model(
                [0.25, 0.75]
            ),
            preprocessor=_RecordingPreprocessor(),
        )

        records = [
            raw_record(
                time="09:00",
                amount="$10.00",
                merchant=100,
            ),
            raw_record(
                time="10:00",
                amount="$20.00",
                merchant=200,
            ),
        ]

        first = service.score_raw_records(
            records
        )
        second = service.score_raw_records(
            records
        )

        np.testing.assert_array_equal(
            first.risk_score,
            second.risk_score,
        )

        np.testing.assert_array_equal(
            first.screening_prediction,
            second.screening_prediction,
        )

    def test_dataset_only_fields_are_dropped_by_raw_adapter(self) -> None:
        preprocessor = _RecordingPreprocessor()

        service = InferenceService(
            model=verified_model(
                [0.20]
            ),
            preprocessor=preprocessor,
        )

        record = raw_record(
            time="09:00",
            amount="$10.00",
            merchant=100,
        )

        record.update(
            {
                "MCC": 5411,
                "Errors?": None,
                "Is Fraud?": "Yes",
            }
        )

        result = service.score_raw_records(
            [record]
        )

        self.assertEqual(
            result.risk_score.shape,
            (1,),
        )
        self.assertEqual(
            len(preprocessor.rows),
            1,
        )


if __name__ == "__main__":
    unittest.main()
