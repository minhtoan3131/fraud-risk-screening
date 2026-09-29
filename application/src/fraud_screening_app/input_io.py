"""JSON input loading for the application boundary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _read_json(
    path: str | Path,
) -> Any:
    resolved = (
        Path(path)
        .expanduser()
        .resolve()
    )

    with resolved.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(
            handle
        )


def load_transaction_json(
    path: str | Path,
) -> dict[str, object]:
    value = _read_json(
        path
    )

    if not isinstance(
        value,
        dict,
    ):
        raise ValueError(
            "Transaction JSON must contain one object."
        )

    return dict(value)


def load_history_json(
    path: str | Path,
) -> list[dict[str, object]]:
    value = _read_json(
        path
    )

    if not isinstance(
        value,
        list,
    ):
        raise ValueError(
            "History JSON must contain a list of transaction objects."
        )

    records: list[
        dict[str, object]
    ] = []

    for index, item in enumerate(
        value
    ):
        if not isinstance(
            item,
            dict,
        ):
            raise ValueError(
                "History JSON item "
                f"{index} must be an object."
            )

        records.append(
            dict(item)
        )

    return records
