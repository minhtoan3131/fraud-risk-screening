"""Fingerprint-verified loading for the frozen classifier artifact."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
from sklearn.ensemble import RandomForestClassifier

from fraud_screening.errors import (
    ArtifactCompatibilityError,
    ArtifactFingerprintError,
    ArtifactNotFoundError,
)


MODEL_ID = "RF-REF-100-GINI-SQRT-UNPRUNED-CW"
POSITIVE_CLASS = 1

EXPECTED_MODEL_PARAMETERS = {
    "bootstrap": True,
    "ccp_alpha": 0.0,
    "class_weight": "balanced",
    "criterion": "gini",
    "max_depth": None,
    "max_features": "sqrt",
    "max_samples": None,
    "min_samples_leaf": 1,
    "min_samples_split": 2,
    "n_estimators": 100,
    "n_jobs": -1,
    "random_state": 42,
}


def sha256_file(
    path: str | Path,
    *,
    chunk_size: int = 1024 * 1024,
) -> str:
    """Return SHA-256 for one artifact file."""

    artifact_path = Path(path)
    digest = hashlib.sha256()

    try:
        with artifact_path.open("rb") as file:
            for chunk in iter(
                lambda: file.read(chunk_size),
                b"",
            ):
                digest.update(chunk)
    except FileNotFoundError as exc:
        raise ArtifactNotFoundError(
            f"Artifact not found: {artifact_path}"
        ) from exc

    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class VerifiedModelArtifact:
    """Loaded estimator plus verified identity metadata."""

    estimator: RandomForestClassifier
    sha256: str
    model_id: str
    classes: tuple[int, ...]
    positive_class_index: int

    @property
    def positive_class(self) -> int:
        return POSITIVE_CLASS


def _validate_estimator_identity(
    estimator: Any,
) -> tuple[tuple[int, ...], int]:
    if not isinstance(
        estimator,
        RandomForestClassifier,
    ):
        raise ArtifactCompatibilityError(
            "Artifact is not a RandomForestClassifier."
        )

    actual_params = estimator.get_params(
        deep=False
    )

    for key, expected_value in (
        EXPECTED_MODEL_PARAMETERS.items()
    ):
        if key not in actual_params:
            raise ArtifactCompatibilityError(
                f"Estimator parameter missing: {key}."
            )

        if actual_params[key] != expected_value:
            raise ArtifactCompatibilityError(
                f"Estimator parameter mismatch: {key}."
            )

    if not hasattr(estimator, "classes_"):
        raise ArtifactCompatibilityError(
            "Estimator has no fitted classes_ attribute."
        )

    try:
        classes = tuple(
            int(value)
            for value in estimator.classes_.tolist()
        )
    except Exception as exc:
        raise ArtifactCompatibilityError(
            "Estimator classes_ cannot be normalized."
        ) from exc

    if classes != (0, 1):
        raise ArtifactCompatibilityError(
            "Estimator classes must be exactly [0, 1]."
        )

    positive_class_index = classes.index(
        POSITIVE_CLASS
    )

    if positive_class_index != 1:
        raise ArtifactCompatibilityError(
            "Positive class index must be 1."
        )

    return (
        classes,
        positive_class_index,
    )


def load_verified_model(
    path: str | Path,
    *,
    expected_sha256: str,
) -> VerifiedModelArtifact:
    """Load and verify the frozen model without running inference."""

    artifact_path = Path(path)

    actual_sha256 = sha256_file(
        artifact_path
    )

    if actual_sha256 != expected_sha256:
        raise ArtifactFingerprintError(
            "Model artifact SHA-256 mismatch."
        )

    try:
        estimator = joblib.load(
            artifact_path
        )
    except FileNotFoundError as exc:
        raise ArtifactNotFoundError(
            f"Artifact not found: {artifact_path}"
        ) from exc
    except Exception as exc:
        raise ArtifactCompatibilityError(
            "Model artifact could not be loaded."
        ) from exc

    (
        classes,
        positive_class_index,
    ) = _validate_estimator_identity(
        estimator
    )

    return VerifiedModelArtifact(
        estimator=estimator,
        sha256=actual_sha256,
        model_id=MODEL_ID,
        classes=classes,
        positive_class_index=
            positive_class_index,
    )
