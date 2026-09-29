from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from fraud_screening.replay import (
    RawDatasetIdentity,
    inspect_raw_dataset,
    iter_raw_chunks,
)


COLUMNS = (
    "User",
    "Card",
    "Year",
    "Month",
    "Day",
    "Time",
    "Amount",
    "Use Chip",
    "Merchant Name",
    "Merchant City",
    "Merchant State",
    "Zip",
    "MCC",
    "Errors?",
    "Is Fraud?",
)


def write_fixture(
    root: Path,
    *,
    filename: str = "card_transaction.v1.csv",
) -> tuple[Path, RawDatasetIdentity]:
    path = root / filename

    frame = pd.DataFrame(
        [
            [
                1, 0, 2018, 1, 1, "00:00",
                "$10.00", "Chip Transaction", 100,
                "A", "CA", 90001, 5411, None, "No",
            ],
            [
                1, 0, 2018, 1, 1, "01:00",
                "$20.00", "Swipe Transaction", 200,
                "B", "CA", 90002, 5812, None, "Yes",
            ],
            [
                2, 1, 2018, 1, 2, "02:00",
                "$30.00", "Online Transaction", 300,
                "ONLINE", None, None, 5999, None, "No",
            ],
        ],
        columns=COLUMNS,
    )

    frame.to_csv(
        path,
        index=False,
    )

    digest = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()

    identity = RawDatasetIdentity(
        filename=filename,
        size_bytes=path.stat().st_size,
        sha256=digest,
        row_count=3,
        columns=COLUMNS,
    )

    return path, identity


class RawDatasetReplayBoundaryTests(
    unittest.TestCase
):
    def test_full_identity_check_passes_for_exact_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path, identity = write_fixture(
                Path(tmp)
            )

            inspection = inspect_raw_dataset(
                path,
                identity=identity,
                chunksize=2,
            )

            self.assertTrue(
                inspection.full_identity_match
            )
            self.assertEqual(
                inspection.row_count,
                3,
            )

    def test_schema_order_mismatch_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, identity = write_fixture(
                root
            )

            frame = pd.read_csv(path)
            frame = frame[
                list(
                    reversed(COLUMNS)
                )
            ]
            frame.to_csv(
                path,
                index=False,
            )

            inspection = inspect_raw_dataset(
                path,
                identity=identity,
                verify_sha256=False,
                verify_row_count=False,
            )

            self.assertFalse(
                inspection.schema_matches
            )

    def test_research_path_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            research = (
                root
                / "research"
            )
            research.mkdir()

            path, identity = write_fixture(
                research
            )

            with self.assertRaises(
                ValueError
            ):
                inspect_raw_dataset(
                    path,
                    identity=identity,
                    project_root=root,
                )

    def test_iter_chunks_adds_stable_raw_row_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path, _ = write_fixture(
                Path(tmp)
            )

            chunks = list(
                iter_raw_chunks(
                    path,
                    chunksize=2,
                )
            )

            self.assertEqual(
                chunks[0]["raw_row_id"].tolist(),
                [0, 1],
            )
            self.assertEqual(
                chunks[1]["raw_row_id"].tolist(),
                [2],
            )

    def test_iter_chunks_rejects_schema_order_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, _ = write_fixture(
                root
            )

            frame = pd.read_csv(path)
            frame = frame[
                [
                    *COLUMNS[1:],
                    COLUMNS[0],
                ]
            ]
            frame.to_csv(
                path,
                index=False,
            )

            with self.assertRaises(
                ValueError
            ):
                list(
                    iter_raw_chunks(
                        path,
                        chunksize=2,
                    )
                )

    def test_fast_inspection_can_skip_full_hash_and_row_scan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path, identity = write_fixture(
                Path(tmp)
            )

            inspection = inspect_raw_dataset(
                path,
                identity=identity,
                verify_sha256=False,
                verify_row_count=False,
            )

            self.assertIsNone(
                inspection.sha256
            )
            self.assertIsNone(
                inspection.sha256_matches
            )
            self.assertIsNone(
                inspection.row_count
            )
            self.assertIsNone(
                inspection.row_count_matches
            )


if __name__ == "__main__":
    unittest.main()
