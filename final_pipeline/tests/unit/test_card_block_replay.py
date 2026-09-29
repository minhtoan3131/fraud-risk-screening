from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from fraud_screening.replay.card_blocks import (
    build_card_replay_block,
    iter_card_blocks,
    iter_semantic_card_blocks,
)


RAW_COLUMNS = (
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


def row(
    *,
    user: int,
    card: int,
    time: str,
    amount: str,
    merchant: int,
    fraud: str = "No",
    errors: str | None = None,
    mcc: int = 5411,
) -> list[object]:
    return [
        user,
        card,
        2018,
        1,
        1,
        time,
        amount,
        "Chip Transaction",
        merchant,
        "A",
        "CA",
        90001,
        mcc,
        errors,
        fraud,
    ]


def write_csv(
    root: Path,
    rows: list[list[object]],
) -> Path:
    path = (
        root
        / "card_transaction.v1.csv"
    )

    pd.DataFrame(
        rows,
        columns=RAW_COLUMNS,
    ).to_csv(
        path,
        index=False,
    )

    return path


class CardBlockReplayTests(
    unittest.TestCase
):
    def test_same_card_split_across_chunks_is_one_block(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = write_csv(
                Path(tmp),
                [
                    row(
                        user=1,
                        card=0,
                        time="09:00",
                        amount="$10.00",
                        merchant=100,
                    ),
                    row(
                        user=1,
                        card=0,
                        time="10:00",
                        amount="$20.00",
                        merchant=200,
                    ),
                    row(
                        user=1,
                        card=0,
                        time="11:00",
                        amount="$30.00",
                        merchant=300,
                    ),
                    row(
                        user=2,
                        card=0,
                        time="09:00",
                        amount="$40.00",
                        merchant=400,
                    ),
                ],
            )

            blocks = list(
                iter_card_blocks(
                    path,
                    chunksize=2,
                )
            )

            self.assertEqual(
                [key for key, _ in blocks],
                [(1, 0), (2, 0)],
            )
            self.assertEqual(
                len(blocks[0][1]),
                3,
            )
            self.assertEqual(
                blocks[0][1]["raw_row_id"].tolist(),
                [0, 1, 2],
            )

    def test_card_reappearance_after_emission_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = write_csv(
                Path(tmp),
                [
                    row(
                        user=1,
                        card=0,
                        time="09:00",
                        amount="$10.00",
                        merchant=100,
                    ),
                    row(
                        user=2,
                        card=0,
                        time="09:00",
                        amount="$20.00",
                        merchant=200,
                    ),
                    row(
                        user=1,
                        card=0,
                        time="10:00",
                        amount="$30.00",
                        merchant=300,
                    ),
                ],
            )

            with self.assertRaises(
                ValueError
            ):
                list(
                    iter_card_blocks(
                        path,
                        chunksize=1,
                    )
                )

    def test_semantic_replay_is_chunk_size_invariant(self) -> None:
        rows = [
            row(
                user=1,
                card=0,
                time="09:00",
                amount="$10.00",
                merchant=100,
            ),
            row(
                user=1,
                card=0,
                time="10:00",
                amount="$20.00",
                merchant=200,
            ),
            row(
                user=1,
                card=0,
                time="10:00",
                amount="$30.00",
                merchant=200,
            ),
            row(
                user=1,
                card=0,
                time="10:30",
                amount="$40.00",
                merchant=100,
            ),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            path = write_csv(
                Path(tmp),
                rows,
            )

            split = list(
                iter_semantic_card_blocks(
                    path,
                    chunksize=1,
                )
            )
            unsplit = list(
                iter_semantic_card_blocks(
                    path,
                    chunksize=100,
                )
            )

            self.assertEqual(
                split,
                unsplit,
            )

    def test_same_timestamp_causal_semantics_survive_csv_replay(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = write_csv(
                Path(tmp),
                [
                    row(
                        user=1,
                        card=0,
                        time="09:00",
                        amount="$10.00",
                        merchant=100,
                    ),
                    row(
                        user=1,
                        card=0,
                        time="10:00",
                        amount="$20.00",
                        merchant=200,
                    ),
                    row(
                        user=1,
                        card=0,
                        time="10:00",
                        amount="$30.00",
                        merchant=200,
                    ),
                    row(
                        user=1,
                        card=0,
                        time="10:30",
                        amount="$40.00",
                        merchant=100,
                    ),
                ],
            )

            replay = next(
                iter_semantic_card_blocks(
                    path,
                    chunksize=2,
                )
            )

            rows_out = replay.semantic_rows

            self.assertFalse(
                rows_out[0].has_prior_card_history
            )
            self.assertTrue(
                rows_out[1].has_prior_card_history
            )
            self.assertTrue(
                rows_out[1].is_new_merchant
            )
            self.assertTrue(
                rows_out[2].is_new_merchant
            )
            self.assertEqual(
                rows_out[1].transactions_last_1h,
                rows_out[2].transactions_last_1h,
            )
            self.assertFalse(
                rows_out[3].is_new_merchant
            )

    def test_excluded_dataset_fields_do_not_change_semantic_rows(self) -> None:
        base = [
            row(
                user=1,
                card=0,
                time="09:00",
                amount="$10.00",
                merchant=100,
                fraud="No",
                errors=None,
                mcc=5411,
            ),
            row(
                user=1,
                card=0,
                time="10:00",
                amount="$20.00",
                merchant=200,
                fraud="Yes",
                errors="Bad PIN,",
                mcc=5812,
            ),
        ]

        changed = [
            list(values)
            for values in base
        ]

        for values in changed:
            values[12] = 9999
            values[13] = "Technical Glitch,"
            values[14] = (
                "Yes"
                if values[14] == "No"
                else "No"
            )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path_a = root / "a"
            path_b = root / "b"
            path_a.mkdir()
            path_b.mkdir()

            csv_a = write_csv(
                path_a,
                base,
            )
            csv_b = write_csv(
                path_b,
                changed,
            )

            replay_a = next(
                iter_semantic_card_blocks(
                    csv_a,
                    chunksize=1,
                )
            )
            replay_b = next(
                iter_semantic_card_blocks(
                    csv_b,
                    chunksize=1,
                )
            )

            self.assertEqual(
                replay_a.semantic_rows,
                replay_b.semantic_rows,
            )

    def test_build_block_preserves_raw_row_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = write_csv(
                Path(tmp),
                [
                    row(
                        user=7,
                        card=2,
                        time="09:00",
                        amount="$10.00",
                        merchant=100,
                    ),
                    row(
                        user=7,
                        card=2,
                        time="10:00",
                        amount="$20.00",
                        merchant=200,
                    ),
                ],
            )

            key, raw_block = next(
                iter_card_blocks(
                    path,
                    chunksize=1,
                )
            )

            replay = build_card_replay_block(
                key,
                raw_block,
            )

            self.assertEqual(
                replay.card_key,
                (7, 2),
            )
            self.assertEqual(
                replay.raw_row_ids,
                (0, 1),
            )
            self.assertEqual(
                len(replay.transactions),
                2,
            )
            self.assertEqual(
                len(replay.semantic_rows),
                2,
            )


if __name__ == "__main__":
    unittest.main()
