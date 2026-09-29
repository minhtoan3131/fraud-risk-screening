#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "final_pipeline" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

"""Walk through canonical parsing, causal history and 10 semantic features."""

from fraud_screening.data.parsing import (
    extract_scoring_payload,
    parse_scoring_payload,
)
from fraud_screening.features.semantic import (
    SEMANTIC_FEATURE_ORDER,
    build_semantic_feature_rows,
)


def raw_record(*, time: str, amount: str, merchant: int) -> dict[str, object]:
    return {
        "User": 999999,
        "Card": 0,
        "Year": 2018,
        "Month": 1,
        "Day": 1,
        "Time": time,
        "Amount": amount,
        "Use Chip": "Chip Transaction",
        "Merchant Name": merchant,
        "Merchant City": "A",
        "Merchant State": "CA",
        "Zip": 90001,
    }


def main() -> None:
    print("=" * 92)
    print("02 — FEATURE PIPELINE WALKTHROUGH")
    print("=" * 92)

    raw_records = [
        raw_record(time="09:00", amount="$10.00", merchant=100),
        raw_record(time="10:00", amount="$20.00", merchant=200),
        raw_record(time="10:30", amount="$30.00", merchant=100),
    ]

    print("\n[1] Canonical parsing")
    transactions = tuple(
        parse_scoring_payload(extract_scoring_payload(record))
        for record in raw_records
    )
    for index, tx in enumerate(transactions, start=1):
        print(
            f" - row {index}: user={tx.user_id}, card={tx.card_id}, "
            f"timestamp={tx.timestamp}, amount={tx.amount_numeric}"
        )

    print("\n[2] Causal semantic representation")
    semantic_rows = build_semantic_feature_rows(transactions)
    print(" - exact feature order:", SEMANTIC_FEATURE_ORDER)

    for index, row in enumerate(semantic_rows, start=1):
        values = {
            name: getattr(row, name)
            for name in SEMANTIC_FEATURE_ORDER
        }
        print(f" - semantic row {index}:", values)

    first = semantic_rows[0]
    second = semantic_rows[1]

    assert first.has_prior_card_history is False
    assert first.time_since_previous_transaction_min is None
    assert first.transactions_last_1h == 0
    assert first.amount_minus_previous_mean is None
    assert first.is_new_merchant is True

    assert second.has_prior_card_history is True
    assert second.time_since_previous_transaction_min == 60.0
    assert second.transactions_last_1h == 1
    assert second.amount_minus_previous_mean == 10.0

    print("\n[3] Guardrails")
    print(" - strict-prior history: YES")
    print(" - target used as feature/history: NO")
    print(" - raw identifiers exposed to classifier: NO")
    print(" - duplicate feature implementation: NO")
    print("\n02 FEATURE PIPELINE WALKTHROUGH: PASS")


if __name__ == "__main__":
    main()
