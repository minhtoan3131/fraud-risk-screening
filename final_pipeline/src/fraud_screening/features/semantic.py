"""Exact semantic feature interface consumed by frozen preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from fraud_screening.data import CanonicalTransaction
from fraud_screening.features.history import (
    BehavioralFeatures,
    compute_behavioral_features,
)
from fraud_screening.features.transaction import (
    TransactionFeatures,
    build_transaction_features,
)


NUMERIC_FEATURES = (
    "amount_numeric",
    "time_since_previous_transaction_min",
    "transactions_last_1h",
    "amount_minus_previous_mean",
)

BOOLEAN_FEATURES = (
    "is_new_merchant",
    "has_prior_card_history",
)

CATEGORICAL_FEATURES = (
    "transaction_mode",
    "location_state",
    "hour_of_day",
    "day_of_week",
)

SEMANTIC_FEATURE_ORDER = (
    *NUMERIC_FEATURES,
    *BOOLEAN_FEATURES,
    *CATEGORICAL_FEATURES,
)


@dataclass(frozen=True, slots=True)
class SemanticFeatureRow:
    """One exact 10-feature row before frozen preprocessing."""

    amount_numeric: float
    time_since_previous_transaction_min: float | None
    transactions_last_1h: int
    amount_minus_previous_mean: float | None
    is_new_merchant: bool
    has_prior_card_history: bool
    transaction_mode: str
    location_state: str
    hour_of_day: str
    day_of_week: str

    def as_dict(self) -> dict[str, object]:
        """Return the exact stable semantic-feature order."""

        return {
            "amount_numeric":
                self.amount_numeric,
            "time_since_previous_transaction_min":
                self.time_since_previous_transaction_min,
            "transactions_last_1h":
                self.transactions_last_1h,
            "amount_minus_previous_mean":
                self.amount_minus_previous_mean,
            "is_new_merchant":
                self.is_new_merchant,
            "has_prior_card_history":
                self.has_prior_card_history,
            "transaction_mode":
                self.transaction_mode,
            "location_state":
                self.location_state,
            "hour_of_day":
                self.hour_of_day,
            "day_of_week":
                self.day_of_week,
        }


def combine_feature_parts(
    transaction_features: TransactionFeatures,
    behavioral_features: BehavioralFeatures,
) -> SemanticFeatureRow:
    """Combine current-transaction and strict-causal feature parts."""

    return SemanticFeatureRow(
        amount_numeric=
            transaction_features.amount_numeric,
        time_since_previous_transaction_min=
            behavioral_features.time_since_previous_transaction_min,
        transactions_last_1h=
            behavioral_features.transactions_last_1h,
        amount_minus_previous_mean=
            behavioral_features.amount_minus_previous_mean,
        is_new_merchant=
            behavioral_features.is_new_merchant,
        has_prior_card_history=
            behavioral_features.has_prior_card_history,
        transaction_mode=
            transaction_features.transaction_mode,
        location_state=
            transaction_features.location_state,
        hour_of_day=
            str(transaction_features.hour_of_day),
        day_of_week=
            str(transaction_features.day_of_week),
    )


def build_semantic_feature_rows(
    transactions: Iterable[CanonicalTransaction],
) -> list[SemanticFeatureRow]:
    """Build exact 10-feature rows for one ordered User+Card block."""

    rows = list(transactions)

    behavioral_rows = compute_behavioral_features(
        rows
    )

    if len(behavioral_rows) != len(rows):
        raise RuntimeError(
            "Behavioral feature row count does not match input row count."
        )

    return [
        combine_feature_parts(
            build_transaction_features(transaction),
            behavioral,
        )
        for transaction, behavioral
        in zip(rows, behavioral_rows)
    ]
