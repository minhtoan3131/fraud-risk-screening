"""Deterministic parsing and validation for scoring transactions."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from datetime import datetime
from typing import Any

from fraud_screening.data.models import CanonicalTransaction
from fraud_screening.data.schema import RAW_SCORING_FIELDS
from fraud_screening.errors import InputValidationError


_TIME_PATTERN = re.compile(r"^\d{2}:\d{2}$")


def _is_missing(value: Any) -> bool:
    if value is None:
        return True

    if isinstance(value, float):
        return math.isnan(value)

    try:
        result = value != value
    except Exception:
        return False

    try:
        return bool(result)
    except Exception:
        return False


def _required_value(payload: Mapping[str, Any], field: str) -> Any:
    if field not in payload:
        raise InputValidationError(
            f"Missing required field: {field}"
        )

    value = payload[field]

    if _is_missing(value):
        raise InputValidationError(
            f"Required field is missing: {field}"
        )

    return value


def _parse_integer_identifier(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise InputValidationError(
            f"{field} must be an integer identifier."
        )

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            raise InputValidationError(
                f"{field} must be an integer identifier."
            )
        return int(value)

    text = str(value).strip()

    if not text:
        raise InputValidationError(
            f"{field} must not be blank."
        )

    try:
        return int(text)
    except ValueError as exc:
        raise InputValidationError(
            f"{field} must be an integer identifier."
        ) from exc


def _parse_timestamp(payload: Mapping[str, Any]) -> datetime:
    year = _parse_integer_identifier(
        _required_value(payload, "Year"),
        "Year",
    )
    month = _parse_integer_identifier(
        _required_value(payload, "Month"),
        "Month",
    )
    day = _parse_integer_identifier(
        _required_value(payload, "Day"),
        "Day",
    )

    time_text = str(
        _required_value(payload, "Time")
    ).strip()

    if not _TIME_PATTERN.fullmatch(time_text):
        raise InputValidationError(
            "Time must use HH:MM format."
        )

    try:
        hour = int(time_text[:2])
        minute = int(time_text[3:5])

        return datetime(
            year,
            month,
            day,
            hour,
            minute,
        )
    except ValueError as exc:
        raise InputValidationError(
            "Invalid Year/Month/Day/Time combination."
        ) from exc


def _parse_amount(value: Any) -> float:
    text = (
        str(value)
        .replace("$", "")
        .replace(",", "")
        .strip()
    )

    if not text:
        raise InputValidationError(
            "Amount must not be blank."
        )

    try:
        amount = float(text)
    except ValueError as exc:
        raise InputValidationError(
            "Amount cannot be parsed as a number."
        ) from exc

    if not math.isfinite(amount):
        raise InputValidationError(
            "Amount must be finite."
        )

    return amount


def _parse_required_string(value: Any, field: str) -> str:
    text = str(value).strip()

    if not text:
        raise InputValidationError(
            f"{field} must not be blank."
        )

    return text


def _parse_optional_state(value: Any) -> str | None:
    if _is_missing(value):
        return None

    text = str(value).strip()

    if not text:
        raise InputValidationError(
            "Merchant State must be missing or a non-blank string."
        )

    return text


def _parse_optional_zip(value: Any) -> int | None:
    if _is_missing(value):
        return None

    if isinstance(value, bool):
        raise InputValidationError(
            "Zip must be an integer categorical code when present."
        )

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            raise InputValidationError(
                "Zip must be an integer categorical code when present."
            )
        return int(value)

    text = str(value).strip()

    if not text:
        raise InputValidationError(
            "Zip must be missing or an integer categorical code."
        )

    try:
        numeric = float(text)
    except ValueError as exc:
        raise InputValidationError(
            "Zip must be an integer categorical code when present."
        ) from exc

    if not math.isfinite(numeric) or not numeric.is_integer():
        raise InputValidationError(
            "Zip must be an integer categorical code when present."
        )

    return int(numeric)


def _derive_location_state(
    merchant_city: str,
    merchant_state: str | None,
    zip_code: int | None,
) -> str:
    city_online = merchant_city.upper() == "ONLINE"
    state_missing = merchant_state is None
    zip_missing = zip_code is None

    if city_online and state_missing and zip_missing:
        return "NON_PHYSICAL_OR_ONLINE"

    if (
        not city_online
        and not state_missing
        and not zip_missing
    ):
        return "PHYSICAL_COMPLETE"

    if (
        not city_online
        and not state_missing
        and zip_missing
    ):
        return "PHYSICAL_ZIP_UNAVAILABLE"

    raise InputValidationError(
        "Location fields form an inconsistent semantic state."
    )


def extract_scoring_payload(
    raw_record: Mapping[str, Any],
) -> dict[str, Any]:
    """Select only the permitted scoring fields from a dataset record."""

    missing = [
        field
        for field in RAW_SCORING_FIELDS
        if field not in raw_record
    ]

    if missing:
        raise InputValidationError(
            "Missing required scoring fields: "
            + ", ".join(missing)
        )

    return {
        field: raw_record[field]
        for field in RAW_SCORING_FIELDS
    }


def parse_scoring_payload(
    payload: Mapping[str, Any],
) -> CanonicalTransaction:
    """Validate an exact scoring payload and build canonical transaction state."""

    expected = set(RAW_SCORING_FIELDS)
    observed = set(payload.keys())

    missing = sorted(expected - observed)
    extra = sorted(observed - expected)

    if missing:
        raise InputValidationError(
            "Missing required scoring fields: "
            + ", ".join(missing)
        )

    if extra:
        raise InputValidationError(
            "Unexpected scoring fields: "
            + ", ".join(extra)
        )

    user_id = _parse_integer_identifier(
        _required_value(payload, "User"),
        "User",
    )

    card_id = _parse_integer_identifier(
        _required_value(payload, "Card"),
        "Card",
    )

    merchant_id = _parse_integer_identifier(
        _required_value(payload, "Merchant Name"),
        "Merchant Name",
    )

    timestamp = _parse_timestamp(payload)

    amount_numeric = _parse_amount(
        _required_value(payload, "Amount")
    )

    transaction_mode = _parse_required_string(
        _required_value(payload, "Use Chip"),
        "Use Chip",
    )

    merchant_city = _parse_required_string(
        _required_value(payload, "Merchant City"),
        "Merchant City",
    )

    merchant_state = _parse_optional_state(
        payload["Merchant State"]
    )

    zip_code = _parse_optional_zip(
        payload["Zip"]
    )

    location_state = _derive_location_state(
        merchant_city=merchant_city,
        merchant_state=merchant_state,
        zip_code=zip_code,
    )

    return CanonicalTransaction(
        user_id=user_id,
        card_id=card_id,
        timestamp=timestamp,
        amount_numeric=amount_numeric,
        transaction_mode=transaction_mode,
        merchant_id=merchant_id,
        merchant_city=merchant_city,
        merchant_state=merchant_state,
        zip_code=zip_code,
        location_state=location_state,
    )
