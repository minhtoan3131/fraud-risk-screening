from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from fraud_screening_app.input_io import (
    load_history_json,
    load_transaction_json,
)


class InputIoTests(unittest.TestCase):
    def test_transaction_object_loads(self) -> None:
        with TemporaryDirectory() as temp:
            path = Path(temp) / "current.json"
            path.write_text(
                json.dumps({"User": 1}),
                encoding="utf-8",
            )
            self.assertEqual(
                load_transaction_json(path),
                {"User": 1},
            )

    def test_transaction_list_is_rejected(self) -> None:
        with TemporaryDirectory() as temp:
            path = Path(temp) / "current.json"
            path.write_text(
                "[]",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_transaction_json(path)

    def test_history_list_loads(self) -> None:
        with TemporaryDirectory() as temp:
            path = Path(temp) / "history.json"
            path.write_text(
                json.dumps([{"User": 1}]),
                encoding="utf-8",
            )
            self.assertEqual(
                load_history_json(path),
                [{"User": 1}],
            )

    def test_history_non_object_item_is_rejected(self) -> None:
        with TemporaryDirectory() as temp:
            path = Path(temp) / "history.json"
            path.write_text(
                json.dumps([1]),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_history_json(path)


if __name__ == "__main__":
    unittest.main()
