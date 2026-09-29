"""Stable inference primitives and reusable service for fraud-risk screening."""

from fraud_screening.inference.interactive import (
    COLD_START_WARNING,
    InteractiveScreeningResult,
    score_interactive_transaction,
)
from fraud_screening.inference.scoring import (
    DEFAULT_THRESHOLD,
    ScreeningBatchResult,
    apply_screening_threshold,
    score_encoded_matrix,
)
from fraud_screening.inference.service import (
    InferenceService,
    load_official_inference_service,
)

__all__ = [
    "DEFAULT_THRESHOLD",
    "ScreeningBatchResult",
    "apply_screening_threshold",
    "score_encoded_matrix",
    "InferenceService",
    "load_official_inference_service",
    "COLD_START_WARNING",
    "InteractiveScreeningResult",
    "score_interactive_transaction",
]
