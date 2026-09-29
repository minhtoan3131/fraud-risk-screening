from __future__ import annotations

import unittest
from datetime import datetime

from fraud_screening.data import CanonicalTransaction
from fraud_screening.features import (
    TRANSACTION_FEATURE_ORDER,
    build_transaction_features,
)


def transaction_at(timestamp: datetime) -> CanonicalTransaction:
    return CanonicalTransaction(
        user_id=17,
        card_id=3,
        timestamp=timestamp,
        amount_numeric=-12.5,
        transaction_mode="Chip Transaction",
        merchant_id=998877,
        merchant_city="Boston",
        merchant_state="MA",
        zip_code=2110,
        location_state="PHYSICAL_COMPLETE",
    )


class TransactionFeatureTests(unittest.TestCase):
    def test_exact_transaction_feature_order(self) -> None:
        self.assertEqual(
            TRANSACTION_FEATURE_ORDER,
            (
                "amount_numeric",
                "transaction_mode",
                "location_state",
                "hour_of_day",
                "day_of_week",
            ),
        )

    def test_build_current_transaction_features(self) -> None:
        tx = transaction_at(
            datetime(2024, 1, 1, 14, 35)
        )

        features = build_transaction_features(tx)

        self.assertEqual(features.amount_numeric, -12.5)
        self.assertEqual(
            features.transaction_mode,
            "Chip Transaction",
        )
        self.assertEqual(
            features.location_state,
            "PHYSICAL_COMPLETE",
        )
        self.assertEqual(features.hour_of_day, 14)
        self.assertEqual(features.day_of_week, 0)

    def test_output_mapping_has_exact_fields_and_order(self) -> None:
        tx = transaction_at(
            datetime(2024, 1, 1, 14, 35)
        )

        mapping = build_transaction_features(tx).as_dict()

        self.assertEqual(
            tuple(mapping.keys()),
            TRANSACTION_FEATURE_ORDER,
        )

    def test_identifiers_are_not_exposed(self) -> None:
        tx = transaction_at(
            datetime(2024, 1, 1, 14, 35)
        )

        mapping = build_transaction_features(tx).as_dict()

        self.assertNotIn("User", mapping)
        self.assertNotIn("Card", mapping)
        self.assertNotIn("Merchant Name", mapping)
        self.assertNotIn("user_id", mapping)
        self.assertNotIn("card_id", mapping)
        self.assertNotIn("merchant_id", mapping)

    def test_amount_sign_is_preserved(self) -> None:
        tx = transaction_at(
            datetime(2024, 1, 1, 14, 35)
        )

        features = build_transaction_features(tx)

        self.assertEqual(features.amount_numeric, -12.5)

    def test_midnight_hour_is_zero(self) -> None:
        tx = transaction_at(
            datetime(2024, 1, 1, 0, 0)
        )

        features = build_transaction_features(tx)

        self.assertEqual(features.hour_of_day, 0)

    def test_last_hour_is_twenty_three(self) -> None:
        tx = transaction_at(
            datetime(2024, 1, 1, 23, 59)
        )

        features = build_transaction_features(tx)

        self.assertEqual(features.hour_of_day, 23)

    def test_monday_is_day_zero(self) -> None:
        tx = transaction_at(
            datetime(2024, 1, 1, 12, 0)
        )

        features = build_transaction_features(tx)

        self.assertEqual(features.day_of_week, 0)

    def test_sunday_is_day_six(self) -> None:
        tx = transaction_at(
            datetime(2024, 1, 7, 12, 0)
        )

        features = build_transaction_features(tx)

        self.assertEqual(features.day_of_week, 6)

    def test_no_month_or_weekend_feature_is_emitted(self) -> None:
        tx = transaction_at(
            datetime(2024, 1, 7, 12, 0)
        )

        mapping = build_transaction_features(tx).as_dict()

        self.assertNotIn("month_of_year", mapping)
        self.assertNotIn("is_weekend", mapping)
        self.assertEqual(len(mapping), 5)


if __name__ == "__main__":
    unittest.main()
