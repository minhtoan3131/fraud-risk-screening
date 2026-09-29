from __future__ import annotations

import unittest

import numpy as np
from scipy import sparse

from fraud_screening.artifacts import (
    MODEL_ID,
    VerifiedModelArtifact,
)
from fraud_screening.errors import (
    HistoryValidationError,
)
from fraud_screening.inference.interactive import (
    COLD_START_WARNING,
    score_interactive_transaction,
)
from fraud_screening.inference.service import (
    InferenceService,
)


class _FixedEstimator:
    def __init__(
        self,
        positive_probability: float,
    ) -> None:
        self.positive_probability = (
            positive_probability
        )
        self.rows_scored = []

    def predict_proba(
        self,
        matrix,
    ) -> np.ndarray:
        self.rows_scored.append(
            matrix.shape[0]
        )

        positive = np.full(
            matrix.shape[0],
            self.positive_probability,
            dtype=np.float64,
        )

        return np.column_stack(
            [
                1.0 - positive,
                positive,
            ]
        )


class _RecordingPreprocessor:
    def __init__(self) -> None:
        self.calls = []

    def transform(
        self,
        rows,
    ):
        captured = tuple(
            rows
        )
        self.calls.append(
            captured
        )

        return sparse.csr_matrix(
            np.zeros(
                (
                    len(captured),
                    47,
                ),
                dtype=np.float32,
            )
        )


def make_service(
    probability: float = 0.75,
):
    estimator = _FixedEstimator(
        probability
    )

    model = VerifiedModelArtifact(
        estimator=estimator,
        sha256="x" * 64,
        model_id=MODEL_ID,
        classes=(0, 1),
        positive_class_index=1,
    )

    preprocessor = (
        _RecordingPreprocessor()
    )

    service = InferenceService(
        model=model,
        preprocessor=preprocessor,
    )

    return (
        service,
        estimator,
        preprocessor,
    )


def record(
    *,
    user: int = 1,
    card: int = 0,
    time: str,
    amount: str,
    merchant: int,
) -> dict[str, object]:
    return {
        "User": user,
        "Card": card,
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


class InteractiveInferenceTests(
    unittest.TestCase
):
    def test_no_history_uses_explicit_cold_start_semantics(self) -> None:
        (
            service,
            estimator,
            preprocessor,
        ) = make_service()

        result = score_interactive_transaction(
            service,
            record(
                time="10:00",
                amount="$20.00",
                merchant=200,
            ),
        )

        self.assertTrue(
            result.cold_start
        )
        self.assertEqual(
            result.warnings,
            (COLD_START_WARNING,),
        )

        semantic = (
            preprocessor.calls[-1][0]
        )

        self.assertFalse(
            semantic.has_prior_card_history
        )
        self.assertIsNone(
            semantic.time_since_previous_transaction_min
        )
        self.assertEqual(
            semantic.transactions_last_1h,
            0,
        )
        self.assertIsNone(
            semantic.amount_minus_previous_mean
        )
        self.assertTrue(
            semantic.is_new_merchant
        )

        self.assertEqual(
            estimator.rows_scored,
            [1],
        )

    def test_explicit_prior_history_disables_cold_start(self) -> None:
        (
            service,
            estimator,
            preprocessor,
        ) = make_service()

        result = score_interactive_transaction(
            service,
            record(
                time="10:00",
                amount="$20.00",
                merchant=200,
            ),
            history_records=[
                record(
                    time="09:00",
                    amount="$10.00",
                    merchant=100,
                ),
            ],
        )

        self.assertFalse(
            result.cold_start
        )
        self.assertEqual(
            result.warnings,
            (),
        )

        semantic = (
            preprocessor.calls[-1][0]
        )

        self.assertTrue(
            semantic.has_prior_card_history
        )
        self.assertEqual(
            semantic.time_since_previous_transaction_min,
            60.0,
        )
        self.assertEqual(
            semantic.transactions_last_1h,
            1,
        )
        self.assertEqual(
            semantic.amount_minus_previous_mean,
            10.0,
        )
        self.assertTrue(
            semantic.is_new_merchant
        )

        self.assertEqual(
            estimator.rows_scored,
            [1],
        )

    def test_history_from_other_card_is_rejected(self) -> None:
        service, _, _ = make_service()

        with self.assertRaises(
            HistoryValidationError
        ):
            score_interactive_transaction(
                service,
                record(
                    time="10:00",
                    amount="$20.00",
                    merchant=200,
                ),
                history_records=[
                    record(
                        card=9,
                        time="09:00",
                        amount="$10.00",
                        merchant=100,
                    ),
                ],
            )

    def test_same_timestamp_history_is_rejected(self) -> None:
        service, _, _ = make_service()

        with self.assertRaises(
            HistoryValidationError
        ):
            score_interactive_transaction(
                service,
                record(
                    time="10:00",
                    amount="$20.00",
                    merchant=200,
                ),
                history_records=[
                    record(
                        time="10:00",
                        amount="$10.00",
                        merchant=100,
                    ),
                ],
            )

    def test_future_history_is_rejected(self) -> None:
        service, _, _ = make_service()

        with self.assertRaises(
            HistoryValidationError
        ):
            score_interactive_transaction(
                service,
                record(
                    time="10:00",
                    amount="$20.00",
                    merchant=200,
                ),
                history_records=[
                    record(
                        time="11:00",
                        amount="$10.00",
                        merchant=100,
                    ),
                ],
            )

    def test_decreasing_history_order_is_rejected(self) -> None:
        service, _, _ = make_service()

        with self.assertRaises(
            HistoryValidationError
        ):
            score_interactive_transaction(
                service,
                record(
                    time="12:00",
                    amount="$30.00",
                    merchant=300,
                ),
                history_records=[
                    record(
                        time="10:00",
                        amount="$10.00",
                        merchant=100,
                    ),
                    record(
                        time="09:00",
                        amount="$20.00",
                        merchant=200,
                    ),
                ],
            )

    def test_only_current_transaction_is_sent_to_model(self) -> None:
        (
            service,
            estimator,
            preprocessor,
        ) = make_service()

        score_interactive_transaction(
            service,
            record(
                time="12:00",
                amount="$30.00",
                merchant=300,
            ),
            history_records=[
                record(
                    time="09:00",
                    amount="$10.00",
                    merchant=100,
                ),
                record(
                    time="10:00",
                    amount="$20.00",
                    merchant=200,
                ),
            ],
        )

        self.assertEqual(
            len(preprocessor.calls),
            1,
        )
        self.assertEqual(
            len(preprocessor.calls[0]),
            1,
        )
        self.assertEqual(
            estimator.rows_scored,
            [1],
        )

    def test_repeated_interactive_inference_is_deterministic(self) -> None:
        service, _, _ = make_service(
            0.51
        )

        current = record(
            time="10:00",
            amount="$20.00",
            merchant=200,
        )

        history = [
            record(
                time="09:00",
                amount="$10.00",
                merchant=100,
            ),
        ]

        first = score_interactive_transaction(
            service,
            current,
            history_records=history,
        )

        second = score_interactive_transaction(
            service,
            current,
            history_records=history,
        )

        self.assertEqual(
            first,
            second,
        )
        self.assertEqual(
            int(first.screening_prediction),
            1,
        )


if __name__ == "__main__":
    unittest.main()
