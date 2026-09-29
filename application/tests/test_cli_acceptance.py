from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
CURRENT = "application/examples/current_transaction.json"
HISTORY = "application/examples/history.json"
INVALID = "application/examples/invalid_amount.json"
SAME_TIME = "application/examples/same_timestamp_history.json"


def _env() -> dict[str, str]:
    env = os.environ.copy()

    paths = [
        str(ROOT / "application" / "src"),
        str(ROOT / "final_pipeline" / "src"),
    ]

    existing = env.get("PYTHONPATH", "")
    if existing:
        paths.append(existing)

    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


class CliAcceptanceTests(unittest.TestCase):
    def _run(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "fraud_screening_app.cli",
                *args,
            ],
            cwd=ROOT,
            env=_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

    def test_cold_start_cli(self) -> None:
        result = self._run(
            "screen",
            "--transaction-json",
            CURRENT,
        )

        self.assertEqual(
            result.returncode,
            0,
        )
        self.assertIn(
            "Cold start: True",
            result.stdout,
        )
        self.assertIn(
            "COLD_START_NO_STRICT_PRIOR_CARD_HISTORY",
            result.stdout,
        )

    def test_history_cli(self) -> None:
        result = self._run(
            "screen",
            "--transaction-json",
            CURRENT,
            "--history-json",
            HISTORY,
        )

        self.assertEqual(
            result.returncode,
            0,
        )
        self.assertIn(
            "Cold start: False",
            result.stdout,
        )
        self.assertIn(
            "Warnings: NONE",
            result.stdout,
        )

    def test_json_cli(self) -> None:
        result = self._run(
            "screen",
            "--transaction-json",
            CURRENT,
            "--history-json",
            HISTORY,
            "--json",
        )

        self.assertEqual(
            result.returncode,
            0,
        )
        self.assertIn(
            '"risk_score"',
            result.stdout,
        )
        self.assertIn(
            '"model_id"',
            result.stdout,
        )

    def test_invalid_amount_cli_returns_error_code(self) -> None:
        result = self._run(
            "screen",
            "--transaction-json",
            INVALID,
        )

        self.assertEqual(
            result.returncode,
            2,
        )
        self.assertIn(
            "InputValidationError",
            result.stdout,
        )

    def test_same_timestamp_history_cli_returns_error_code(self) -> None:
        result = self._run(
            "screen",
            "--transaction-json",
            CURRENT,
            "--history-json",
            SAME_TIME,
        )

        self.assertEqual(
            result.returncode,
            2,
        )
        self.assertIn(
            "HistoryValidationError",
            result.stdout,
        )


if __name__ == "__main__":
    unittest.main()
