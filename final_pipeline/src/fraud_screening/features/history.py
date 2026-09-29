"""Strict-causal behavioral features built from card history."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import timedelta
from typing import Iterable

from fraud_screening.data import CanonicalTransaction
from fraud_screening.errors import HistoryValidationError


HISTORY_FEATURE_ORDER = (
    "time_since_previous_transaction_min",
    "transactions_last_1h",
    "amount_minus_previous_mean",
    "is_new_merchant",
    "has_prior_card_history",
)


@dataclass(frozen=True, slots=True)
class BehavioralFeatures:
    """Strict-prior behavioral features for one transaction."""

    time_since_previous_transaction_min: float | None
    transactions_last_1h: int
    amount_minus_previous_mean: float | None
    is_new_merchant: bool
    has_prior_card_history: bool

    def as_dict(self) -> dict[str, object]:
        return {
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
        }


def _validate_same_card(
    transactions: list[CanonicalTransaction],
) -> None:
    if not transactions:
        return

    key = (
        transactions[0].user_id,
        transactions[0].card_id,
    )

    for transaction in transactions[1:]:
        if (
            transaction.user_id,
            transaction.card_id,
        ) != key:
            raise HistoryValidationError(
                "All transactions in a card-history block must "
                "belong to the same User + Card."
            )


def _validate_non_decreasing_time(
    transactions: list[CanonicalTransaction],
) -> None:
    for previous, current in zip(
        transactions,
        transactions[1:],
    ):
        if current.timestamp < previous.timestamp:
            raise HistoryValidationError(
                "Transactions must be ordered by non-decreasing timestamp."
            )


def compute_behavioral_features(
    transactions: Iterable[CanonicalTransaction],
) -> list[BehavioralFeatures]:
    """Compute strict-causal features for an ordered User+Card block.

    Transactions sharing one timestamp are treated as one prediction point.
    Every row in that timestamp group sees only transactions with strictly
    earlier timestamps. State is updated only after the whole timestamp group
    has been scored.
    """

    rows = list(transactions)

    if not rows:
        return []

    _validate_same_card(rows)
    _validate_non_decreasing_time(rows)

    results: list[BehavioralFeatures] = []

    prior_count = 0
    prior_amount_sum = 0.0
    seen_merchants: set[int] = set()
    previous_distinct_timestamp = None
    one_hour_window = deque()

    start = 0

    while start < len(rows):
        timestamp = rows[start].timestamp
        end = start + 1

        while (
            end < len(rows)
            and rows[end].timestamp == timestamp
        ):
            end += 1

        group = rows[start:end]
        has_prior = prior_count > 0

        if has_prior:
            assert previous_distinct_timestamp is not None
            delta = timestamp - previous_distinct_timestamp
            time_since_previous_min = (
                delta.total_seconds() / 60.0
            )
            previous_amount_mean = (
                prior_amount_sum / prior_count
            )
        else:
            time_since_previous_min = None
            previous_amount_mean = None

        window_start = timestamp - timedelta(hours=1)

        while (
            one_hour_window
            and one_hour_window[0] < window_start
        ):
            one_hour_window.popleft()

        transactions_last_1h = len(one_hour_window)

        for transaction in group:
            if previous_amount_mean is None:
                amount_minus_previous_mean = None
            else:
                amount_minus_previous_mean = (
                    transaction.amount_numeric
                    - previous_amount_mean
                )

            results.append(
                BehavioralFeatures(
                    time_since_previous_transaction_min=
                        time_since_previous_min,
                    transactions_last_1h=
                        transactions_last_1h,
                    amount_minus_previous_mean=
                        amount_minus_previous_mean,
                    is_new_merchant=(
                        transaction.merchant_id
                        not in seen_merchants
                    ),
                    has_prior_card_history=has_prior,
                )
            )

        # Strict-causal state update happens only after the whole timestamp group.
        group_count = len(group)
        prior_count += group_count
        prior_amount_sum += sum(
            transaction.amount_numeric
            for transaction in group
        )
        seen_merchants.update(
            transaction.merchant_id
            for transaction in group
        )
        one_hour_window.extend(
            transaction.timestamp
            for transaction in group
        )
        previous_distinct_timestamp = timestamp
        start = end

    return results
