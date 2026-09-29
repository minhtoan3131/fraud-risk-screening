#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "final_pipeline" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

"""Walk through frozen official preprocessing from 10 semantic to 47 encoded features."""

from fraud_screening.artifacts.official_loader import load_official_artifacts
from fraud_screening.data.parsing import (
    extract_scoring_payload,
    parse_scoring_payload,
)
from fraud_screening.features.semantic import build_semantic_feature_rows

OFFICIAL_ROOT = PROJECT_ROOT / "final_pipeline" / "artifacts" / "official"


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
    print("03 — PREPROCESSING WALKTHROUGH")
    print("=" * 92)

    bundle = load_official_artifacts(OFFICIAL_ROOT)
    print("\n[1] Verified official bundle")
    print(" - model id:", bundle.model.model_id)
    print(" - encoded feature count:", len(bundle.preprocessor.feature_names))

    records = [
        raw_record(time="09:00", amount="$10.00", merchant=100),
        raw_record(time="10:00", amount="$20.00", merchant=200),
    ]
    transactions = tuple(
        parse_scoring_payload(extract_scoring_payload(record))
        for record in records
    )
    semantic_rows = build_semantic_feature_rows(transactions)

    matrix = bundle.preprocessor.transform(semantic_rows)

    print("\n[2] Frozen transform")
    print(" - shape:", matrix.shape)
    print(" - dtype:", matrix.dtype)
    print(" - format:", matrix.getformat())
    print(" - nnz:", matrix.nnz)

    assert matrix.shape == (len(semantic_rows), 47)
    assert str(matrix.dtype) == "float32"
    assert matrix.getformat() == "csr"

    print("\n[3] Active encoded columns")
    names = bundle.preprocessor.feature_names
    for row_index in range(matrix.shape[0]):
        active = [
            names[column_index]
            for column_index in matrix.getrow(row_index).indices.tolist()
        ]
        print(f" - row {row_index + 1}:", active)

    print("\n[4] Boundary")
    print(" - preprocessing fit: NO")
    print(" - model.fit(): NO")
    print(" - predict/predict_proba: NO")
    print(" - frozen output width: 47")
    print("\n03 PREPROCESSING WALKTHROUGH: PASS")


if __name__ == "__main__":
    main()
