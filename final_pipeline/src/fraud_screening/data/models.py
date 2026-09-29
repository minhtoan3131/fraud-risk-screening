"""Canonical in-memory transaction representation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class CanonicalTransaction:
    """Validated transaction state before historical feature construction."""

    user_id: int
    card_id: int
    timestamp: datetime
    amount_numeric: float
    transaction_mode: str
    merchant_id: int
    merchant_city: str
    merchant_state: str | None
    zip_code: int | None
    location_state: str
