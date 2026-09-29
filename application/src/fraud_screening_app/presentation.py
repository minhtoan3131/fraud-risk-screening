"""Presentation helpers for final-pipeline screening results."""

from __future__ import annotations

from typing import Mapping

from fraud_screening.inference import (
    InteractiveScreeningResult,
)


SUMMARY_FIELDS = (
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
)


def transaction_summary(
    current_record: Mapping[
        str,
        object,
    ],
) -> dict[str, object]:
    return {
        field: current_record.get(
            field
        )
        for field in SUMMARY_FIELDS
    }


def result_payload(
    current_record: Mapping[
        str,
        object,
    ],
    result: InteractiveScreeningResult,
    *,
    history_count: int,
) -> dict[str, object]:
    return {
        "transaction":
            transaction_summary(
                current_record
            ),
        "risk_score":
            float(
                result.risk_score
            ),
        "threshold":
            float(
                result.threshold
            ),
        "screening_prediction":
            int(
                result.screening_prediction
            ),
        "cold_start":
            bool(
                result.cold_start
            ),
        "history_count":
            int(
                history_count
            ),
        "warnings":
            list(
                result.warnings
            ),
        "model_id":
            result.model_id,
    }


def format_text_result(
    current_record: Mapping[
        str,
        object,
    ],
    result: InteractiveScreeningResult,
    *,
    history_count: int,
) -> str:
    payload = result_payload(
        current_record,
        result,
        history_count=history_count,
    )

    transaction = payload[
        "transaction"
    ]

    lines = [
        "=" * 72,
        "TRANSACTION FRAUD-RISK SCREENING RESULT",
        "=" * 72,
        "",
        "[Transaction]",
        (
            "User/Card: "
            f"{transaction['User']}/"
            f"{transaction['Card']}"
        ),
        (
            "Date/Time: "
            f"{transaction['Year']}-"
            f"{transaction['Month']}-"
            f"{transaction['Day']} "
            f"{transaction['Time']}"
        ),
        (
            "Amount: "
            f"{transaction['Amount']}"
        ),
        (
            "Mode: "
            f"{transaction['Use Chip']}"
        ),
        "",
        "[Screening]",
        (
            "Risk score: "
            f"{payload['risk_score']:.8f}"
        ),
        (
            "Threshold: "
            f"{payload['threshold']}"
        ),
        (
            "Screening prediction: "
            f"{payload['screening_prediction']}"
        ),
        (
            "Cold start: "
            f"{payload['cold_start']}"
        ),
        (
            "History records supplied: "
            f"{payload['history_count']}"
        ),
        (
            "Warnings: "
            + (
                ", ".join(
                    payload["warnings"]
                )
                if payload["warnings"]
                else "NONE"
            )
        ),
        (
            "Model id: "
            f"{payload['model_id']}"
        ),
        "",
        (
            "Interpretation: this is a screening "
            "support result, not a final fraud accusation."
        ),
    ]

    return "\n".join(
        lines
    )
