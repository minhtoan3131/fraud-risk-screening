from __future__ import annotations

import unittest
from datetime import datetime

from fraud_screening.data import CanonicalTransaction
from fraud_screening.features import (
    BOOLEAN_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    SEMANTIC_FEATURE_ORDER,
    build_semantic_feature_rows,
)


def tx(
    timestamp: str,
    amount: float,
    merchant: int,
) -> CanonicalTransaction:
    return CanonicalTransaction(
        user_id=1,
        card_id=2,
        timestamp=datetime.fromisoformat(timestamp),
        amount_numeric=amount,
        transaction_mode="Chip Transaction",
        merchant_id=merchant,
        merchant_city="Boston",
        merchant_state="MA",
        zip_code=2110,
        location_state="PHYSICAL_COMPLETE",
    )


class SemanticFeatureTests(unittest.TestCase):
    def test_exact_role_groups(self) -> None:
        self.assertEqual(
            NUMERIC_FEATURES,
            (
                "amount_numeric",
                "time_since_previous_transaction_min",
                "transactions_last_1h",
                "amount_minus_previous_mean",
            ),
        )
        self.assertEqual(
            BOOLEAN_FEATURES,
            (
                "is_new_merchant",
                "has_prior_card_history",
            ),
        )
        self.assertEqual(
            CATEGORICAL_FEATURES,
            (
                "transaction_mode",
                "location_state",
                "hour_of_day",
                "day_of_week",
            ),
        )

    def test_exact_ten_feature_order(self) -> None:
        self.assertEqual(
            SEMANTIC_FEATURE_ORDER,
            (
                "amount_numeric",
                "time_since_previous_transaction_min",
                "transactions_last_1h",
                "amount_minus_previous_mean",
                "is_new_merchant",
                "has_prior_card_history",
                "transaction_mode",
                "location_state",
                "hour_of_day",
                "day_of_week",
            ),
        )
        self.assertEqual(
            len(SEMANTIC_FEATURE_ORDER),
            10,
        )

    def test_output_mapping_exact_order(self) -> None:
        result = build_semantic_feature_rows(
            [tx("2018-01-01 09:00:00", 10.0, 100)]
        )[0]

        self.assertEqual(
            tuple(result.as_dict().keys()),
            SEMANTIC_FEATURE_ORDER,
        )

    def test_cold_start_structural_values_preserved(self) -> None:
        result = build_semantic_feature_rows(
            [tx("2018-01-01 09:00:00", 10.0, 100)]
        )[0]

        self.assertEqual(result.amount_numeric, 10.0)
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

    def test_hour_and_day_are_categorical_strings(self) -> None:
        result = build_semantic_feature_rows(
            [tx("2018-01-01 14:35:00", 10.0, 100)]
        )[0]

        self.assertEqual(result.hour_of_day, "14")
        self.assertEqual(result.day_of_week, "0")
        self.assertIsInstance(result.hour_of_day, str)
        self.assertIsInstance(result.day_of_week, str)

    def test_transaction_categorical_values_preserved(self) -> None:
        result = build_semantic_feature_rows(
            [tx("2018-01-01 09:00:00", 10.0, 100)]
        )[0]

        self.assertEqual(
            result.transaction_mode,
            "Chip Transaction",
        )
        self.assertEqual(
            result.location_state,
            "PHYSICAL_COMPLETE",
        )

    def test_row_count_is_preserved(self) -> None:
        rows = [
            tx("2018-01-01 09:00:00", 10.0, 100),
            tx("2018-01-01 10:00:00", 20.0, 200),
            tx("2018-01-01 10:00:00", 30.0, 200),
            tx("2018-01-01 10:30:00", 40.0, 100),
        ]

        result = build_semantic_feature_rows(rows)

        self.assertEqual(len(result), len(rows))

    def test_verified_regression_values_survive_merge(self) -> None:
        rows = [
            tx("2018-01-01 09:00:00", 10.0, 100),
            tx("2018-01-01 10:00:00", 20.0, 200),
            tx("2018-01-01 10:00:00", 30.0, 200),
            tx("2018-01-01 10:30:00", 40.0, 100),
        ]

        result = build_semantic_feature_rows(rows)

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

    def test_no_identifier_or_target_is_exposed(self) -> None:
        result = build_semantic_feature_rows(
            [tx("2018-01-01 09:00:00", 10.0, 100)]
        )[0].as_dict()

        prohibited = {
            "User",
            "Card",
            "Merchant Name",
            "user_id",
            "card_id",
            "merchant_id",
            "Errors?",
            "Is Fraud?",
            "Timestamp",
            "raw_row_id",
        }

        self.assertTrue(
            prohibited.isdisjoint(result.keys())
        )

    def test_no_non_core_transaction_features(self) -> None:
        result = build_semantic_feature_rows(
            [tx("2018-01-01 09:00:00", 10.0, 100)]
        )[0].as_dict()

        self.assertNotIn("MCC", result)
        self.assertNotIn("mcc_code", result)
        self.assertNotIn("month_of_year", result)
        self.assertNotIn("is_weekend", result)
        self.assertEqual(len(result), 10)

    def test_empty_block_returns_empty(self) -> None:
        self.assertEqual(
            build_semantic_feature_rows([]),
            [],
        )


if __name__ == "__main__":
    unittest.main()
