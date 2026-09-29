"""Interactive single-transaction inference with explicit history context."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from fraud_screening.data import (
    CanonicalTransaction,
    extract_scoring_payload,
    parse_scoring_payload,
)
from fraud_screening.errors import (
    HistoryValidationError,
    InferenceContractError,
)
from fraud_screening.features import (
    build_semantic_feature_rows,
)
from fraud_screening.inference.service import (
    InferenceService,
)


COLD_START_WARNING = (
    "COLD_START_NO_STRICT_PRIOR_CARD_HISTORY"
)


@dataclass(frozen=True, slots=True)
class InteractiveScreeningResult:
    """Stable result for one interactive current transaction."""

    risk_score: np.float32
    threshold: float
    screening_prediction: np.int8
    model_id: str
    cold_start: bool
    warnings: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(
            self.risk_score,
            np.float32,
        ):
            raise InferenceContractError(
                "Interactive risk_score must be numpy.float32."
            )

        if not np.isfinite(
            self.risk_score
        ):
            raise InferenceContractError(
                "Interactive risk_score must be finite."
            )

        if not (
            0.0
            <= float(self.risk_score)
            <= 1.0
        ):
            raise InferenceContractError(
                "Interactive risk_score must be within [0, 1]."
            )

        if not isinstance(
            self.screening_prediction,
            np.int8,
        ):
            raise InferenceContractError(
                "Interactive screening_prediction must be numpy.int8."
            )

        if int(
            self.screening_prediction
        ) not in (0, 1):
            raise InferenceContractError(
                "Interactive screening_prediction must be 0 or 1."
            )

        if self.threshold != 0.50:
            raise InferenceContractError(
                "Interactive threshold must equal 0.50."
            )

        if not isinstance(
            self.model_id,
            str,
        ) or not self.model_id:
            raise InferenceContractError(
                "Interactive model_id must be a non-empty string."
            )

        if not isinstance(
            self.cold_start,
            bool,
        ):
            raise InferenceContractError(
                "cold_start must be boolean."
            )

        if not isinstance(
            self.warnings,
            tuple,
        ) or not all(
            isinstance(value, str)
            and value
            for value in self.warnings
        ):
            raise InferenceContractError(
                "warnings must be a tuple of non-empty strings."
            )

        expected_warning = (
            (COLD_START_WARNING,)
            if self.cold_start
            else ()
        )

        if self.warnings != expected_warning:
            raise InferenceContractError(
                "Interactive warnings do not match cold-start state."
            )


def _parse_record(
    raw_record: Mapping[str, Any],
) -> CanonicalTransaction:
    return parse_scoring_payload(
        extract_scoring_payload(
            raw_record
        )
    )


def _validate_explicit_history(
    current: CanonicalTransaction,
    history: Sequence[CanonicalTransaction],
) -> None:
    previous_timestamp = None

    for index, transaction in enumerate(
        history
    ):
        if (
            transaction.user_id
            != current.user_id
            or
            transaction.card_id
            != current.card_id
        ):
            raise HistoryValidationError(
                "Interactive history must contain only the current User+Card."
            )

        if (
            transaction.timestamp
            >= current.timestamp
        ):
            raise HistoryValidationError(
                "Interactive history timestamps must be strictly earlier "
                "than the current transaction."
            )

        if (
            previous_timestamp
            is not None
            and transaction.timestamp
            < previous_timestamp
        ):
            raise HistoryValidationError(
                "Interactive history must be in non-decreasing timestamp order."
            )

        previous_timestamp = (
            transaction.timestamp
        )


def score_interactive_transaction(
    service: InferenceService,
    current_record: Mapping[str, Any],
    *,
    history_records: Sequence[
        Mapping[str, Any]
    ] = (),
) -> InteractiveScreeningResult:
    """Score one current transaction using only explicit strict-prior history."""

    if not isinstance(
        service,
        InferenceService,
    ):
        raise InferenceContractError(
            "service must be an InferenceService."
        )

    current = _parse_record(
        current_record
    )

    history = tuple(
        _parse_record(record)
        for record in history_records
    )

    _validate_explicit_history(
        current,
        history,
    )

    transactions = (
        *history,
        current,
    )

    semantic_rows = (
        build_semantic_feature_rows(
            transactions
        )
    )

    if len(semantic_rows) != len(
        transactions
    ):
        raise InferenceContractError(
            "Semantic row count changed during interactive inference."
        )

    current_semantic = (
        semantic_rows[-1]
    )

    cold_start = (
        not current_semantic
        .has_prior_card_history
    )

    if cold_start != (
        len(history) == 0
    ):
        raise InferenceContractError(
            "Cold-start state does not match explicit history context."
        )

    batch_result = (
        service.score_semantic_rows(
            [current_semantic]
        )
    )

    if (
        batch_result.risk_score.shape
        != (1,)
        or
        batch_result.screening_prediction.shape
        != (1,)
    ):
        raise InferenceContractError(
            "Interactive inference must score exactly one current transaction."
        )

    warnings = (
        (COLD_START_WARNING,)
        if cold_start
        else ()
    )

    return InteractiveScreeningResult(
        risk_score=np.float32(
            batch_result.risk_score[0]
        ),
        threshold=batch_result.threshold,
        screening_prediction=np.int8(
            batch_result.screening_prediction[0]
        ),
        model_id=batch_result.model_id,
        cold_start=cold_start,
        warnings=warnings,
    )
