"""In-memory fitting of a rebuild Random Forest model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier

from fraud_screening.artifacts import (
    EXPECTED_MODEL_PARAMETERS,
)
from fraud_screening.training.factory import (
    REBUILD_MODEL_ID,
    build_rebuild_model,
)


EXPECTED_FEATURE_COUNT = 47
EXPECTED_TARGET_DTYPE = np.dtype(
    np.int8
)


class RebuildTrainingError(ValueError):
    """Raised when rebuild training input violates the locked contract."""


@dataclass(frozen=True, slots=True)
class RebuildModelFit:
    """One newly learned rebuild estimator and descriptive training metadata."""

    estimator: RandomForestClassifier
    model_id: str
    training_rows: int
    fraud_rows: int
    feature_count: int
    classes: tuple[int, ...]
    artifact_identity_status: str
    evaluation_status: str


def _validate_training_matrix(
    matrix: sparse.spmatrix,
) -> sparse.csr_matrix:
    if not sparse.isspmatrix_csr(
        matrix
    ):
        raise RebuildTrainingError(
            "Training matrix must be CSR."
        )

    if matrix.shape[0] <= 0:
        raise RebuildTrainingError(
            "Training matrix must contain at least one row."
        )

    if matrix.shape[1] != EXPECTED_FEATURE_COUNT:
        raise RebuildTrainingError(
            "Training matrix must contain exactly 47 features."
        )

    if matrix.dtype != np.float32:
        raise RebuildTrainingError(
            "Training matrix dtype must be float32."
        )

    if not np.isfinite(
        matrix.data
    ).all():
        raise RebuildTrainingError(
            "Training matrix contains NaN or infinity."
        )

    return matrix


def _validate_training_target(
    target: Sequence[int] | np.ndarray,
    *,
    expected_rows: int,
) -> np.ndarray:
    y = np.asarray(
        target
    )

    if y.ndim != 1:
        raise RebuildTrainingError(
            "Training target must be one-dimensional."
        )

    if y.shape[0] != expected_rows:
        raise RebuildTrainingError(
            "Training target length must match matrix row count."
        )

    if y.dtype != EXPECTED_TARGET_DTYPE:
        raise RebuildTrainingError(
            "Training target dtype must be int8."
        )

    unique = np.unique(
        y
    )

    if not np.all(
        np.isin(
            unique,
            np.asarray(
                [0, 1],
                dtype=np.int8,
            ),
        )
    ):
        raise RebuildTrainingError(
            "Training target must contain only labels 0 and 1."
        )

    if unique.tolist() != [0, 1]:
        raise RebuildTrainingError(
            "Training target must contain both classes 0 and 1."
        )

    return y


def fit_rebuild_model(
    matrix: sparse.spmatrix,
    target: Sequence[int] | np.ndarray,
) -> RebuildModelFit:
    """Fit a new rebuild estimator in memory.

    This function never writes files and never promotes the result to the
    official artifact set.
    """

    X = _validate_training_matrix(
        matrix
    )

    y = _validate_training_target(
        target,
        expected_rows=X.shape[0],
    )

    estimator = build_rebuild_model()

    estimator.fit(
        X,
        y,
    )

    classes = tuple(
        int(value)
        for value in estimator.classes_.tolist()
    )

    if classes != (0, 1):
        raise RebuildTrainingError(
            "Fitted rebuild estimator classes must be exactly (0, 1)."
        )

    if int(
        estimator.n_features_in_
    ) != EXPECTED_FEATURE_COUNT:
        raise RebuildTrainingError(
            "Fitted rebuild estimator feature width mismatch."
        )

    if len(
        estimator.estimators_
    ) != 100:
        raise RebuildTrainingError(
            "Fitted rebuild estimator must contain exactly 100 trees."
        )

    actual_params = estimator.get_params(
        deep=False
    )

    for key, expected_value in (
        EXPECTED_MODEL_PARAMETERS.items()
    ):
        if actual_params[key] != expected_value:
            raise RebuildTrainingError(
                f"Fitted rebuild parameter mismatch: {key}."
            )

    return RebuildModelFit(
        estimator=estimator,
        model_id=REBUILD_MODEL_ID,
        training_rows=int(
            X.shape[0]
        ),
        fraud_rows=int(
            np.count_nonzero(
                y == 1
            )
        ),
        feature_count=EXPECTED_FEATURE_COUNT,
        classes=classes,
        artifact_identity_status=
            "NEW_REBUILD_MODEL",
        evaluation_status=
            "NOT_EVALUATED_AS_OFFICIAL",
    )
