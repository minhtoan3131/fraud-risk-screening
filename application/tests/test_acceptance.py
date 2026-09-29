from __future__ import annotations

from pathlib import Path
import unittest

from fraud_screening.errors import (
    HistoryValidationError,
    InputValidationError,
)
from fraud_screening.inference import (
    load_official_inference_service,
    score_interactive_transaction,
)

from fraud_screening_app.input_io import (
    load_history_json,
    load_transaction_json,
)
from fraud_screening_app.screening import (
    ScreeningApplication,
)


ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_ROOT = (
    ROOT
    / "final_pipeline"
    / "artifacts"
    / "official"
)
EXAMPLES = (
    ROOT
    / "application"
    / "examples"
)


class ApplicationAcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.application = (
            ScreeningApplication
            .from_official_artifacts(
                OFFICIAL_ROOT
            )
        )

        cls.current = (
            load_transaction_json(
                EXAMPLES
                / "current_transaction.json"
            )
        )

        cls.history = (
            load_history_json(
                EXAMPLES
                / "history.json"
            )
        )

    def test_cold_start_end_to_end(self) -> None:
        result = self.application.screen(
            self.current
        )

        self.assertTrue(
            result.cold_start
        )

        self.assertTrue(
            result.warnings
        )

    def test_explicit_history_end_to_end(self) -> None:
        result = self.application.screen(
            self.current,
            history_records=self.history,
        )

        self.assertFalse(
            result.cold_start
        )

        self.assertEqual(
            result.warnings,
            (),
        )

    def test_application_equals_direct_final_pipeline(self) -> None:
        direct_service = (
            load_official_inference_service(
                OFFICIAL_ROOT
            )
        )

        direct = (
            score_interactive_transaction(
                direct_service,
                self.current,
                history_records=self.history,
            )
        )

        through_application = (
            self.application.screen(
                self.current,
                history_records=self.history,
            )
        )

        self.assertEqual(
            through_application,
            direct,
        )

    def test_invalid_amount_is_rejected(self) -> None:
        invalid = (
            load_transaction_json(
                EXAMPLES
                / "invalid_amount.json"
            )
        )

        with self.assertRaises(
            InputValidationError
        ):
            self.application.screen(
                invalid
            )

    def test_same_timestamp_history_is_rejected(self) -> None:
        history = (
            load_history_json(
                EXAMPLES
                / "same_timestamp_history.json"
            )
        )

        with self.assertRaises(
            HistoryValidationError
        ):
            self.application.screen(
                self.current,
                history_records=history,
            )

    def test_other_card_history_is_rejected(self) -> None:
        history = (
            load_history_json(
                EXAMPLES
                / "other_card_history.json"
            )
        )

        with self.assertRaises(
            HistoryValidationError
        ):
            self.application.screen(
                self.current,
                history_records=history,
            )


if __name__ == "__main__":
    unittest.main()
