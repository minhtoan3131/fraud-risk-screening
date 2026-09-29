from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

from fraud_screening_app.presentation import (
    format_text_result,
    result_payload,
)


CURRENT = {
    "User": 1,
    "Card": 2,
    "Year": 2018,
    "Month": 1,
    "Day": 1,
    "Time": "10:00",
    "Amount": "$20.00",
    "Use Chip": "Chip Transaction",
    "Merchant Name": 200,
    "Merchant City": "A",
    "Merchant State": "CA",
    "Zip": 90001,
}


class PresentationTests(unittest.TestCase):
    def _result(self):
        return SimpleNamespace(
            risk_score=np.float32(0.25),
            threshold=0.5,
            screening_prediction=np.int8(0),
            cold_start=False,
            warnings=(),
            model_id="MODEL",
        )

    def test_payload_preserves_result_fields(self) -> None:
        payload = result_payload(
            CURRENT,
            self._result(),
            history_count=3,
        )
        self.assertEqual(payload["risk_score"], 0.25)
        self.assertEqual(payload["threshold"], 0.5)
        self.assertEqual(payload["screening_prediction"], 0)
        self.assertFalse(payload["cold_start"])
        self.assertEqual(payload["history_count"], 3)
        self.assertEqual(payload["warnings"], [])
        self.assertEqual(payload["model_id"], "MODEL")

    def test_text_contains_interpretation_boundary(self) -> None:
        text = format_text_result(
            CURRENT,
            self._result(),
            history_count=0,
        )
        self.assertIn("Risk score:", text)
        self.assertIn("Threshold:", text)
        self.assertIn("screening support", text)
        self.assertIn("not a final fraud accusation", text)


if __name__ == "__main__":
    unittest.main()
