"""Current-transaction feature representation."""

from __future__ import annotations

from dataclasses import dataclass

from fraud_screening.data import CanonicalTransaction


TRANSACTION_FEATURE_ORDER = (
    "amount_numeric",
    "transaction_mode",
    "location_state",
    "hour_of_day",
    "day_of_week",
)


@dataclass(frozen=True, slots=True)
class TransactionFeatures:
    """Features derived only from the current transaction."""

    amount_numeric: float
    transaction_mode: str
    location_state: str
    hour_of_day: int
    day_of_week: int

    def as_dict(self) -> dict[str, object]:
        """Return fields in the stable transaction-feature order."""

        return {
            "amount_numeric": self.amount_numeric,
            "transaction_mode": self.transaction_mode,
            "location_state": self.location_state,
            "hour_of_day": self.hour_of_day,
            "day_of_week": self.day_of_week,
        }


def build_transaction_features(
    transaction: CanonicalTransaction,
) -> TransactionFeatures:
    """Build deterministic current-transaction features."""

    return TransactionFeatures(
        amount_numeric=transaction.amount_numeric,
        transaction_mode=transaction.transaction_mode,
        location_state=transaction.location_state,
        hour_of_day=transaction.timestamp.hour,
        day_of_week=transaction.timestamp.weekday(),
    )
