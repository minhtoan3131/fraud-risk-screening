"""Factory for reproducing the frozen final Random Forest configuration."""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier

from fraud_screening.artifacts import (
    EXPECTED_MODEL_PARAMETERS,
    MODEL_ID,
)


REBUILD_MODEL_ID = MODEL_ID


def build_rebuild_model() -> RandomForestClassifier:
    """Create an unfitted estimator matching the frozen final configuration."""

    estimator = RandomForestClassifier(
        bootstrap=True,
        ccp_alpha=0.0,
        class_weight="balanced",
        criterion="gini",
        max_depth=None,
        max_features="sqrt",
        max_samples=None,
        min_samples_leaf=1,
        min_samples_split=2,
        n_estimators=100,
        n_jobs=-1,
        random_state=42,
    )

    actual_params = estimator.get_params(
        deep=False
    )

    for key, expected_value in (
        EXPECTED_MODEL_PARAMETERS.items()
    ):
        if actual_params[key] != expected_value:
            raise RuntimeError(
                f"Rebuild model parameter mismatch: {key}."
            )

    return estimator
