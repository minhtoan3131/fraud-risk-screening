from __future__ import annotations

import math
import unittest
from datetime import datetime

from fraud_screening.data import (
    extract_scoring_payload,
    parse_scoring_payload,
)
from fraud_screening.errors import InputValidationError


def valid_payload() -> dict:
    return {
        "User": 0,
        "Card": 0,
        "Year": 2002,
        "Month": 9,
        "Day": 1,
        "Time": "06:21",
        "Amount": "$134.09",
        "Use Chip": " Swipe Transaction ",
        "Merchant Name": 3527213246127876953,
        "Merchant City": " La Verne ",
        "Merchant State": "CA",
        "Zip": 91750.0,
    }


class DataBoundaryTests(unittest.TestCase):
    def test_valid_physical_transaction(self) -> None:
        tx = parse_scoring_payload(valid_payload())

        self.assertEqual(tx.user_id, 0)
        self.assertEqual(tx.card_id, 0)
        self.assertEqual(
            tx.timestamp,
            datetime(2002, 9, 1, 6, 21),
        )
        self.assertEqual(tx.amount_numeric, 134.09)
        self.assertEqual(
            tx.transaction_mode,
            "Swipe Transaction",
        )
        self.assertEqual(
            tx.merchant_id,
            3527213246127876953,
        )
        self.assertEqual(tx.merchant_city, "La Verne")
        self.assertEqual(tx.merchant_state, "CA")
        self.assertEqual(tx.zip_code, 91750)
        self.assertEqual(
            tx.location_state,
            "PHYSICAL_COMPLETE",
        )

    def test_online_location(self) -> None:
        payload = valid_payload()
        payload["Merchant City"] = "ONLINE"
        payload["Merchant State"] = None
        payload["Zip"] = math.nan

        tx = parse_scoring_payload(payload)

        self.assertEqual(
            tx.location_state,
            "NON_PHYSICAL_OR_ONLINE",
        )

    def test_physical_zip_unavailable(self) -> None:
        payload = valid_payload()
        payload["Zip"] = None

        tx = parse_scoring_payload(payload)

        self.assertEqual(
            tx.location_state,
            "PHYSICAL_ZIP_UNAVAILABLE",
        )

    def test_negative_amount_is_preserved(self) -> None:
        payload = valid_payload()
        payload["Amount"] = "-$12.50"

        tx = parse_scoring_payload(payload)

        self.assertEqual(tx.amount_numeric, -12.50)

    def test_zero_amount_is_preserved(self) -> None:
        payload = valid_payload()
        payload["Amount"] = "$0.00"

        tx = parse_scoring_payload(payload)

        self.assertEqual(tx.amount_numeric, 0.0)

    def test_amount_with_comma_is_parsed(self) -> None:
        payload = valid_payload()
        payload["Amount"] = "$1,234.50"

        tx = parse_scoring_payload(payload)

        self.assertEqual(tx.amount_numeric, 1234.50)

    def test_unknown_transaction_mode_is_not_rejected(self) -> None:
        payload = valid_payload()
        payload["Use Chip"] = "Future Mode"

        tx = parse_scoring_payload(payload)

        self.assertEqual(
            tx.transaction_mode,
            "Future Mode",
        )

    def test_invalid_timestamp_fails(self) -> None:
        payload = valid_payload()
        payload["Time"] = "25:99"

        with self.assertRaises(InputValidationError):
            parse_scoring_payload(payload)

    def test_invalid_amount_fails(self) -> None:
        payload = valid_payload()
        payload["Amount"] = "not-money"

        with self.assertRaises(InputValidationError):
            parse_scoring_payload(payload)

    def test_inconsistent_location_fails(self) -> None:
        payload = valid_payload()
        payload["Merchant City"] = "ONLINE"
        payload["Merchant State"] = "CA"
        payload["Zip"] = None

        with self.assertRaises(InputValidationError):
            parse_scoring_payload(payload)

    def test_fractional_zip_fails(self) -> None:
        payload = valid_payload()
        payload["Zip"] = 91750.5

        with self.assertRaises(InputValidationError):
            parse_scoring_payload(payload)

    def test_missing_required_field_fails(self) -> None:
        payload = valid_payload()
        del payload["Amount"]

        with self.assertRaises(InputValidationError):
            parse_scoring_payload(payload)

    def test_exact_scoring_payload_rejects_extra_fields(self) -> None:
        payload = valid_payload()
        payload["MCC"] = 5300

        with self.assertRaises(InputValidationError):
            parse_scoring_payload(payload)

    def test_dataset_adapter_drops_excluded_fields(self) -> None:
        raw = valid_payload()
        raw["MCC"] = 5300
        raw["Errors?"] = None
        raw["Is Fraud?"] = "No"

        scoring = extract_scoring_payload(raw)

        self.assertEqual(
            set(scoring),
            set(valid_payload()),
        )
        self.assertNotIn("MCC", scoring)
        self.assertNotIn("Errors?", scoring)
        self.assertNotIn("Is Fraud?", scoring)

    def test_dataset_adapter_requires_all_scoring_fields(self) -> None:
        raw = valid_payload()
        del raw["Merchant Name"]

        with self.assertRaises(InputValidationError):
            extract_scoring_payload(raw)


if __name__ == "__main__":
    unittest.main()
