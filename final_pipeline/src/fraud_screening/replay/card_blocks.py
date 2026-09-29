"""Card-block dataset replay using the canonical final-pipeline APIs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import pandas as pd

from fraud_screening.data import (
    CanonicalTransaction,
    extract_scoring_payload,
    parse_scoring_payload,
)
from fraud_screening.features import (
    SemanticFeatureRow,
    build_semantic_feature_rows,
)
from fraud_screening.replay.raw_dataset import (
    iter_raw_chunks,
)


CardKey = tuple[int, int]


@dataclass(frozen=True, slots=True)
class CardReplayBlock:
    """One complete contiguous User+Card block and its semantic rows."""

    card_key: CardKey
    raw_row_ids: tuple[int, ...]
    transactions: tuple[CanonicalTransaction, ...]
    semantic_rows: tuple[SemanticFeatureRow, ...]

    def __post_init__(self) -> None:
        row_count = len(self.raw_row_ids)

        if len(self.transactions) != row_count:
            raise ValueError(
                "Transaction count must match raw_row_ids."
            )

        if len(self.semantic_rows) != row_count:
            raise ValueError(
                "Semantic row count must match raw_row_ids."
            )


def _card_key_from_frame(
    frame: pd.DataFrame,
) -> CardKey:
    if frame.empty:
        raise ValueError(
            "Card block may not be empty."
        )

    users = frame["User"].drop_duplicates()
    cards = frame["Card"].drop_duplicates()

    if len(users) != 1 or len(cards) != 1:
        raise ValueError(
            "Card block must contain exactly one User+Card key."
        )

    return (
        int(users.iloc[0]),
        int(cards.iloc[0]),
    )


def _combine_parts(
    parts: Sequence[pd.DataFrame],
) -> pd.DataFrame:
    if not parts:
        raise ValueError(
            "Cannot combine an empty card-block part list."
        )

    if len(parts) == 1:
        return (
            parts[0]
            .reset_index(drop=True)
            .copy()
        )

    return pd.concat(
        list(parts),
        ignore_index=True,
    )


def iter_card_blocks(
    path: str | Path,
    *,
    chunksize: int = 250_000,
    project_root: str | Path | None = None,
) -> Iterator[tuple[CardKey, pd.DataFrame]]:
    """Yield complete contiguous User+Card blocks across CSV chunk boundaries."""

    pending_key: CardKey | None = None
    pending_parts: list[pd.DataFrame] = []
    emitted_keys: set[CardKey] = set()

    for chunk in iter_raw_chunks(
        path,
        chunksize=chunksize,
        project_root=project_root,
    ):
        if chunk.empty:
            continue

        users = chunk["User"].to_numpy()
        cards = chunk["Card"].to_numpy()

        change_positions = [
            index
            for index in range(1, len(chunk))
            if (
                users[index] != users[index - 1]
                or cards[index] != cards[index - 1]
            )
        ]

        starts = [
            0,
            *change_positions,
        ]
        ends = [
            *change_positions,
            len(chunk),
        ]

        for start, end in zip(
            starts,
            ends,
            strict=True,
        ):
            segment = (
                chunk
                .iloc[start:end]
                .copy()
            )
            key = (
                int(users[start]),
                int(cards[start]),
            )

            if pending_key is None:
                if key in emitted_keys:
                    raise ValueError(
                        f"User+Card key reappeared after emission: {key}"
                    )

                pending_key = key
                pending_parts = [segment]
                continue

            if key == pending_key:
                pending_parts.append(
                    segment
                )
                continue

            block = _combine_parts(
                pending_parts
            )

            observed_key = _card_key_from_frame(
                block
            )

            if observed_key != pending_key:
                raise ValueError(
                    "Internal card-block key mismatch."
                )

            emitted_keys.add(
                pending_key
            )

            yield pending_key, block

            if key in emitted_keys:
                raise ValueError(
                    f"User+Card key reappeared after emission: {key}"
                )

            pending_key = key
            pending_parts = [segment]

    if pending_key is not None:
        block = _combine_parts(
            pending_parts
        )

        observed_key = _card_key_from_frame(
            block
        )

        if observed_key != pending_key:
            raise ValueError(
                "Internal final card-block key mismatch."
            )

        if pending_key in emitted_keys:
            raise ValueError(
                f"User+Card key reappeared after emission: {pending_key}"
            )

        yield pending_key, block


def build_card_replay_block(
    card_key: CardKey,
    raw_block: pd.DataFrame,
) -> CardReplayBlock:
    """Run one complete raw card block through canonical parse/history/semantic APIs."""

    observed_key = _card_key_from_frame(
        raw_block
    )

    if observed_key != card_key:
        raise ValueError(
            "Supplied card_key does not match raw block."
        )

    if "raw_row_id" not in raw_block.columns:
        raise ValueError(
            "raw_row_id is required for replay lineage."
        )

    raw_row_ids = tuple(
        int(value)
        for value
        in raw_block["raw_row_id"].tolist()
    )

    transactions = []

    for record in raw_block.to_dict(
        orient="records"
    ):
        payload = extract_scoring_payload(
            record
        )
        transaction = parse_scoring_payload(
            payload
        )
        transactions.append(
            transaction
        )

    transaction_tuple = tuple(
        transactions
    )

    semantic_rows = tuple(
        build_semantic_feature_rows(
            transaction_tuple
        )
    )

    return CardReplayBlock(
        card_key=card_key,
        raw_row_ids=raw_row_ids,
        transactions=transaction_tuple,
        semantic_rows=semantic_rows,
    )


def iter_semantic_card_blocks(
    path: str | Path,
    *,
    chunksize: int = 250_000,
    project_root: str | Path | None = None,
) -> Iterator[CardReplayBlock]:
    """Yield canonical semantic replay blocks without preprocessing or inference."""

    for card_key, raw_block in iter_card_blocks(
        path,
        chunksize=chunksize,
        project_root=project_root,
    ):
        yield build_card_replay_block(
            card_key,
            raw_block,
        )
