"""Read-only boundary for the canonical raw transaction dataset."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Iterator

import pandas as pd


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


@dataclass(frozen=True, slots=True)
class RawDatasetIdentity:
    filename: str
    size_bytes: int
    sha256: str
    row_count: int
    columns: tuple[str, ...]


CANONICAL_RAW_DATASET = RawDatasetIdentity(
    filename="card_transaction.v1.csv",
    size_bytes=2_354_626_737,
    sha256=(
        "68c438319cf27614d5564b7b520814036f0d7e087b3a17d8c15081848e0f02de"
    ),
    row_count=24_386_900,
    columns=RAW_COLUMNS,
)


@dataclass(frozen=True, slots=True)
class RawDatasetInspection:
    path: Path
    filename_matches: bool
    size_bytes: int
    size_matches: bool
    sha256: str | None
    sha256_matches: bool | None
    columns: tuple[str, ...]
    schema_matches: bool
    row_count: int | None
    row_count_matches: bool | None

    @property
    def full_identity_match(self) -> bool:
        return (
            self.filename_matches
            and self.size_matches
            and self.sha256_matches is True
            and self.schema_matches
            and self.row_count_matches is True
        )


def _sha256_file(
    path: Path,
    *,
    chunk_size: int = 1024 * 1024,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(chunk_size),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def _guard_external_path(
    path: Path,
    project_root: Path | None,
) -> None:
    if project_root is None:
        return

    research_root = (
        project_root
        / "research"
    ).resolve()

    try:
        path.resolve().relative_to(
            research_root
        )
    except ValueError:
        return

    raise ValueError(
        "Final-pipeline dataset replay may not read from research/."
    )


def _read_header(
    path: Path,
) -> tuple[str, ...]:
    header = pd.read_csv(
        path,
        nrows=0,
    )

    return tuple(
        str(column)
        for column in header.columns
    )


def _count_rows(
    path: Path,
    *,
    chunksize: int,
) -> int:
    total = 0

    for chunk in pd.read_csv(
        path,
        usecols=["User"],
        chunksize=chunksize,
    ):
        total += len(chunk)

    return total


def inspect_raw_dataset(
    path: str | Path,
    *,
    identity: RawDatasetIdentity = CANONICAL_RAW_DATASET,
    project_root: str | Path | None = None,
    verify_sha256: bool = True,
    verify_row_count: bool = True,
    chunksize: int = 250_000,
) -> RawDatasetInspection:
    resolved = Path(path).expanduser().resolve()

    if not resolved.is_file():
        raise FileNotFoundError(
            f"Raw dataset does not exist: {resolved}"
        )

    if (
        not isinstance(chunksize, int)
        or chunksize <= 0
    ):
        raise ValueError(
            "chunksize must be a positive integer."
        )

    resolved_project_root = (
        None
        if project_root is None
        else Path(project_root).resolve()
    )

    _guard_external_path(
        resolved,
        resolved_project_root,
    )

    size_bytes = resolved.stat().st_size
    columns = _read_header(resolved)

    sha256 = (
        _sha256_file(resolved)
        if verify_sha256
        else None
    )

    row_count = (
        _count_rows(
            resolved,
            chunksize=chunksize,
        )
        if verify_row_count
        else None
    )

    return RawDatasetInspection(
        path=resolved,
        filename_matches=(
            resolved.name
            == identity.filename
        ),
        size_bytes=size_bytes,
        size_matches=(
            size_bytes
            == identity.size_bytes
        ),
        sha256=sha256,
        sha256_matches=(
            None
            if sha256 is None
            else sha256
            == identity.sha256
        ),
        columns=columns,
        schema_matches=(
            columns
            == identity.columns
        ),
        row_count=row_count,
        row_count_matches=(
            None
            if row_count is None
            else row_count
            == identity.row_count
        ),
    )


def iter_raw_chunks(
    path: str | Path,
    *,
    chunksize: int = 250_000,
    project_root: str | Path | None = None,
) -> Iterator[pd.DataFrame]:
    resolved = Path(path).expanduser().resolve()

    if not resolved.is_file():
        raise FileNotFoundError(
            f"Raw dataset does not exist: {resolved}"
        )

    if (
        not isinstance(chunksize, int)
        or chunksize <= 0
    ):
        raise ValueError(
            "chunksize must be a positive integer."
        )

    resolved_project_root = (
        None
        if project_root is None
        else Path(project_root).resolve()
    )

    _guard_external_path(
        resolved,
        resolved_project_root,
    )

    raw_row_offset = 0

    # TextFileReader must be closed even when validation raises midway.
    # This matters for long-lived dataset replay and prevents leaked
    # file descriptors when an invalid chunk aborts iteration.
    with pd.read_csv(
        resolved,
        chunksize=chunksize,
    ) as reader:
        for chunk in reader:
            actual_columns = tuple(
                str(column)
                for column in chunk.columns
            )

            if actual_columns != RAW_COLUMNS:
                raise ValueError(
                    "Raw dataset schema/order does not match the canonical contract."
                )

            chunk = chunk.copy()

            chunk["raw_row_id"] = range(
                raw_row_offset,
                raw_row_offset
                + len(chunk),
            )

            raw_row_offset += len(chunk)

            yield chunk
