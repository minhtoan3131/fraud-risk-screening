"""Frozen model scoring and threshold application."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from fraud_screening.artifacts import (
    VerifiedModelArtifact,
)
from fraud_screening.errors import (
    InferenceContractError,
)


DEFAULT_THRESHOLD = 0.50
EXPECTED_ENCODED_WIDTH = 47


@dataclass(frozen=True, slots=True)
class ScreeningBatchResult:
    """Stable screening output for one encoded batch."""

    risk_score: np.ndarray
    threshold: float
    screening_prediction: np.ndarray
    model_id: str

    def __post_init__(self) -> None:
        risk_score = np.asarray(
            self.risk_score
        )
        prediction = np.asarray(
            self.screening_prediction
        )

        if risk_score.ndim != 1:
            raise InferenceContractError(
                "risk_score must be one-dimensional."
            )

        if prediction.ndim != 1:
            raise InferenceContractError(
                "screening_prediction must be one-dimensional."
            )

        if risk_score.shape != prediction.shape:
            raise InferenceContractError(
                "risk_score and screening_prediction lengths must match."
            )

        if risk_score.dtype != np.float32:
            raise InferenceContractError(
                "risk_score dtype must be float32."
            )

        if prediction.dtype != np.int8:
            raise InferenceContractError(
                "screening_prediction dtype must be int8."
            )

        if not np.isfinite(risk_score).all():
            raise InferenceContractError(
                "risk_score must contain only finite values."
            )

        if not (
            np.all(risk_score >= 0.0)
            and np.all(risk_score <= 1.0)
        ):
            raise InferenceContractError(
                "risk_score must be within [0, 1]."
            )

        if not np.isin(
            prediction,
            [0, 1],
        ).all():
            raise InferenceContractError(
                "screening_prediction must contain only 0 or 1."
            )

        if self.threshold != DEFAULT_THRESHOLD:
            raise InferenceContractError(
                "threshold must equal the frozen value 0.50."
            )

        if not isinstance(
            self.model_id,
            str,
        ) or not self.model_id:
            raise InferenceContractError(
                "model_id must be a non-empty string."
            )


def apply_screening_threshold(
    risk_score: np.ndarray,
    *,
    threshold: float = DEFAULT_THRESHOLD,
) -> np.ndarray:
    """Apply the frozen strict comparator risk_score > 0.50."""

    if threshold != DEFAULT_THRESHOLD:
        raise InferenceContractError(
            "Only the frozen threshold 0.50 is allowed."
        )

    score = np.asarray(
        risk_score,
        dtype=np.float32,
    )

    if score.ndim != 1:
        raise InferenceContractError(
            "risk_score must be one-dimensional."
        )

    if not np.isfinite(score).all():
        raise InferenceContractError(
            "risk_score contains NaN or infinity."
        )

    if not (
        np.all(score >= 0.0)
        and np.all(score <= 1.0)
    ):
        raise InferenceContractError(
            "risk_score must be within [0, 1]."
        )

    return (
        score > threshold
    ).astype(
        np.int8,
        copy=False,
    )


def _validate_encoded_matrix(
    matrix: sparse.spmatrix,
) -> sparse.csr_matrix:
    if not sparse.issparse(matrix):
        raise InferenceContractError(
            "Encoded input must be a scipy sparse matrix."
        )

    encoded = matrix.tocsr(
        copy=False
    )

    if encoded.ndim != 2:
        raise InferenceContractError(
            "Encoded input must be two-dimensional."
        )

    if encoded.shape[1] != EXPECTED_ENCODED_WIDTH:
        raise InferenceContractError(
            "Encoded input width must be exactly 47."
        )

    if encoded.dtype != np.float32:
        raise InferenceContractError(
            "Encoded input dtype must be float32."
        )

    if not np.isfinite(
        encoded.data
    ).all():
        raise InferenceContractError(
            "Encoded input contains NaN or infinity."
        )

    return encoded


def score_encoded_matrix(
    matrix: sparse.spmatrix,
    model: VerifiedModelArtifact,
) -> ScreeningBatchResult:
    """Run frozen predict_proba and strict thresholding."""

    if not isinstance(
        model,
        VerifiedModelArtifact,
    ):
        raise InferenceContractError(
            "model must be a VerifiedModelArtifact."
        )

    encoded = _validate_encoded_matrix(
        matrix
    )

    probability_matrix = (
        model.estimator.predict_proba(
            encoded
        )
    )

    probabilities = np.asarray(
        probability_matrix
    )

    if probabilities.ndim != 2:
        raise InferenceContractError(
            "predict_proba output must be two-dimensional."
        )

    if probabilities.shape != (
        encoded.shape[0],
        len(model.classes),
    ):
        raise InferenceContractError(
            "predict_proba output shape does not match verified classes."
        )

    if not np.isfinite(
        probabilities
    ).all():
        raise InferenceContractError(
            "predict_proba output contains NaN or infinity."
        )

    risk_score = (
        probabilities[
            :,
            model.positive_class_index,
        ]
        .astype(
            np.float32,
            copy=False,
        )
    )

    if not (
        np.all(risk_score >= 0.0)
        and np.all(risk_score <= 1.0)
    ):
        raise InferenceContractError(
            "Positive-class risk score is outside [0, 1]."
        )

    prediction = apply_screening_threshold(
        risk_score
    )

    return ScreeningBatchResult(
        risk_score=risk_score,
        threshold=DEFAULT_THRESHOLD,
        screening_prediction=prediction,
        model_id=model.model_id,
    )
