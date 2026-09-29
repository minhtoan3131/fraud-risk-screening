#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "final_pipeline" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

"""Walk through official end-to-end interactive inference."""

from fraud_screening.errors import HistoryValidationError
from fraud_screening.inference.interactive import (
    COLD_START_WARNING,
    score_interactive_transaction,
)
from fraud_screening.inference.service import load_official_inference_service

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


def show_result(label: str, result) -> None:
    print(
        f" - {label}: risk_score={float(result.risk_score):.8f}, "
        f"threshold={result.threshold}, "
        f"screening_prediction={int(result.screening_prediction)}, "
        f"cold_start={result.cold_start}, warnings={result.warnings}, "
        f"model_id={result.model_id}"
    )


def main() -> None:
    print("=" * 92)
    print("05 — INFERENCE WALKTHROUGH")
    print("=" * 92)

    service = load_official_inference_service(OFFICIAL_ROOT)
    print("\n[1] Verified official service")
    print(" - model id:", service.model.model_id)

    current = raw_record(time="10:00", amount="$20.00", merchant=200)
    prior = raw_record(time="09:00", amount="$10.00", merchant=100)

    cold = score_interactive_transaction(service, current)
    print("\n[2] Cold-start")
    show_result("cold-start", cold)
    assert cold.cold_start is True
    assert cold.warnings == (COLD_START_WARNING,)

    with_history = score_interactive_transaction(
        service,
        current,
        history_records=[prior],
    )
    print("\n[3] Explicit history")
    show_result("with-history", with_history)
    assert with_history.cold_start is False
    assert with_history.warnings == ()

    repeated = score_interactive_transaction(
        service,
        current,
        history_records=[prior],
    )
    assert repeated == with_history
    print("\n[4] Deterministic repeat: PASS")

    same_time = raw_record(time="10:00", amount="$10.00", merchant=100)
    print("\n[5] Invalid same-timestamp history")
    try:
        score_interactive_transaction(
            service,
            current,
            history_records=[same_time],
        )
    except HistoryValidationError as exc:
        print(" - rejected:", type(exc).__name__, str(exc))
    else:
        raise AssertionError("Same-timestamp history was not rejected.")

    print("\n[6] Interpretation boundary")
    print(" - risk_score: positive-class model score")
    print(" - not claimed as calibrated real-world fraud probability")
    print(" - screening result is support for review, not a final accusation")
    print("\n05 INFERENCE WALKTHROUGH: PASS")


if __name__ == "__main__":
    main()
