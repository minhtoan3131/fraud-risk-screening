from __future__ import annotations

import unittest
from datetime import datetime

from fraud_screening.data import CanonicalTransaction
from fraud_screening.errors import HistoryValidationError
from fraud_screening.features import (
    HISTORY_FEATURE_ORDER,
    compute_behavioral_features,
)


def tx(
    timestamp: str,
    amount: float,
    merchant: int,
    *,
    user: int = 1,
    card: int = 1,
) -> CanonicalTransaction:
    return CanonicalTransaction(
        user_id=user,
        card_id=card,
        timestamp=datetime.fromisoformat(timestamp),
        amount_numeric=amount,
        transaction_mode="Chip Transaction",
        merchant_id=merchant,
        merchant_city="Boston",
        merchant_state="MA",
        zip_code=2110,
        location_state="PHYSICAL_COMPLETE",
    )


class HistoryFeatureTests(unittest.TestCase):
    def test_exact_history_feature_order(self) -> None:
        self.assertEqual(
            HISTORY_FEATURE_ORDER,
            (
                "time_since_previous_transaction_min",
                "transactions_last_1h",
                "amount_minus_previous_mean",
                "is_new_merchant",
                "has_prior_card_history",
            ),
        )

    def test_cold_start_semantics(self) -> None:
        result = compute_behavioral_features(
            [tx("2018-01-01 09:00:00", 10.0, 100)]
        )[0]

        self.assertIsNone(
            result.time_since_previous_transaction_min
        )
        self.assertEqual(result.transactions_last_1h, 0)
        self.assertIsNone(
            result.amount_minus_previous_mean
        )
        self.assertTrue(result.is_new_merchant)
        self.assertFalse(
            result.has_prior_card_history
        )

    def test_regression_case_from_verified_builder(self) -> None:
        rows = [
            tx("2018-01-01 09:00:00", 10.0, 100),
            tx("2018-01-01 10:00:00", 20.0, 200),
            tx("2018-01-01 10:00:00", 30.0, 200),
            tx("2018-01-01 10:30:00", 40.0, 100),
        ]

        result = compute_behavioral_features(rows)

        self.assertEqual(
            [
                item.time_since_previous_transaction_min
                for item in result
            ],
            [None, 60.0, 60.0, 30.0],
        )
        self.assertEqual(
            [
                item.transactions_last_1h
                for item in result
            ],
            [0, 1, 1, 2],
        )
        self.assertEqual(
            [
                item.is_new_merchant
                for item in result
            ],
            [True, True, True, False],
        )

    def test_same_timestamp_peer_not_in_amount_mean(self) -> None:
        rows = [
            tx("2018-01-01 09:00:00", 10.0, 100),
            tx("2018-01-01 10:00:00", 20.0, 200),
            tx("2018-01-01 10:00:00", 30.0, 300),
        ]

        result = compute_behavioral_features(rows)

        self.assertEqual(
            result[1].amount_minus_previous_mean,
            10.0,
        )
        self.assertEqual(
            result[2].amount_minus_previous_mean,
            20.0,
        )

    def test_previous_mean_uses_full_strict_prior_history(self) -> None:
        rows = [
            tx("2018-01-01 09:00:00", 10.0, 100),
            tx("2018-01-01 10:00:00", 20.0, 200),
            tx("2018-01-01 11:00:00", 60.0, 300),
        ]

        result = compute_behavioral_features(rows)

        # Previous mean for 11:00 is mean(10, 20) = 15.
        self.assertEqual(
            result[2].amount_minus_previous_mean,
            45.0,
        )

    def test_one_hour_window_includes_exact_left_boundary(self) -> None:
        rows = [
            tx("2018-01-01 09:00:00", 10.0, 100),
            tx("2018-01-01 10:00:00", 20.0, 200),
        ]

        result = compute_behavioral_features(rows)

        self.assertEqual(
            result[1].transactions_last_1h,
            1,
        )

    def test_one_hour_window_excludes_older_history(self) -> None:
        rows = [
            tx("2018-01-01 08:59:59", 10.0, 100),
            tx("2018-01-01 10:00:00", 20.0, 200),
        ]

        result = compute_behavioral_features(rows)

        self.assertEqual(
            result[1].transactions_last_1h,
            0,
        )

    def test_same_timestamp_peers_not_in_velocity(self) -> None:
        rows = [
            tx("2018-01-01 09:30:00", 10.0, 100),
            tx("2018-01-01 10:00:00", 20.0, 200),
            tx("2018-01-01 10:00:00", 30.0, 300),
        ]

        result = compute_behavioral_features(rows)

        self.assertEqual(
            result[1].transactions_last_1h,
            1,
        )
        self.assertEqual(
            result[2].transactions_last_1h,
            1,
        )

    def test_same_timestamp_first_merchant_all_new(self) -> None:
        rows = [
            tx("2018-01-01 10:00:00", 20.0, 200),
            tx("2018-01-01 10:00:00", 30.0, 200),
        ]

        result = compute_behavioral_features(rows)

        self.assertEqual(
            [item.is_new_merchant for item in result],
            [True, True],
        )

    def test_existing_merchant_is_not_new(self) -> None:
        rows = [
            tx("2018-01-01 09:00:00", 10.0, 100),
            tx("2018-01-01 10:00:00", 20.0, 100),
        ]

        result = compute_behavioral_features(rows)

        self.assertFalse(result[1].is_new_merchant)

    def test_same_timestamp_does_not_create_prior_history(self) -> None:
        rows = [
            tx("2018-01-01 10:00:00", 10.0, 100),
            tx("2018-01-01 10:00:00", 20.0, 200),
        ]

        result = compute_behavioral_features(rows)

        self.assertEqual(
            [
                item.has_prior_card_history
                for item in result
            ],
            [False, False],
        )
        self.assertEqual(
            [
                item.time_since_previous_transaction_min
                for item in result
            ],
            [None, None],
        )

    def test_mixed_cards_are_rejected(self) -> None:
        rows = [
            tx("2018-01-01 09:00:00", 10.0, 100, card=1),
            tx("2018-01-01 10:00:00", 20.0, 200, card=2),
        ]

        with self.assertRaises(HistoryValidationError):
            compute_behavioral_features(rows)

    def test_mixed_users_are_rejected(self) -> None:
        rows = [
            tx("2018-01-01 09:00:00", 10.0, 100, user=1),
            tx("2018-01-01 10:00:00", 20.0, 200, user=2),
        ]

        with self.assertRaises(HistoryValidationError):
            compute_behavioral_features(rows)

    def test_decreasing_timestamp_is_rejected(self) -> None:
        rows = [
            tx("2018-01-01 10:00:00", 10.0, 100),
            tx("2018-01-01 09:00:00", 20.0, 200),
        ]

        with self.assertRaises(HistoryValidationError):
            compute_behavioral_features(rows)

    def test_empty_history_block_returns_empty(self) -> None:
        self.assertEqual(
            compute_behavioral_features([]),
            [],
        )

    def test_mapping_order_is_stable(self) -> None:
        result = compute_behavioral_features(
            [tx("2018-01-01 09:00:00", 10.0, 100)]
        )[0]

        self.assertEqual(
            tuple(result.as_dict().keys()),
            HISTORY_FEATURE_ORDER,
        )


if __name__ == "__main__":
    unittest.main()
