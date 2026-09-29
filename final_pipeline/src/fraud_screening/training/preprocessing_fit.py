"""TRAIN-only learned preprocessing for rebuild runs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Sequence

import numpy as np
from scipy import sparse
from sklearn.preprocessing import StandardScaler

from fraud_screening.errors import PreprocessingContractError
from fraud_screening.features import (
    BOOLEAN_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    SemanticFeatureRow,
)


UNKNOWN_TOKEN = "__UNKNOWN__"
REBUILD_STRATEGY = "W_SHORT"
REBUILD_FIT_SOURCE = "W_SHORT_TRAIN_ONLY"

EXPECTED_CATEGORY_VOCAB = {
    "transaction_mode": (
        "Chip Transaction",
        "Online Transaction",
        "Swipe Transaction",
    ),
    "location_state": (
        "NON_PHYSICAL_OR_ONLINE",
        "PHYSICAL_COMPLETE",
        "PHYSICAL_ZIP_UNAVAILABLE",
    ),
    "hour_of_day": tuple(
        str(value)
        for value in range(24)
    ),
    "day_of_week": tuple(
        str(value)
        for value in range(7)
    ),
}

_STRUCTURAL_NA_FEATURES = {
    "time_since_previous_transaction_min",
    "amount_minus_previous_mean",
}


def _stable_sort_categories(
    column: str,
    values: Iterable[str],
) -> tuple[str, ...]:
    normalized = [
        str(value)
        for value in values
    ]

    if column in {
        "hour_of_day",
        "day_of_week",
    }:
        try:
            return tuple(
                sorted(
                    normalized,
                    key=lambda value: int(value),
                )
            )
        except ValueError as exc:
            raise PreprocessingContractError(
                f"Non-integer categorical value for {column}."
            ) from exc

    return tuple(
        sorted(normalized)
    )


def _build_feature_names(
    category_vocab: dict[str, tuple[str, ...]],
) -> tuple[str, ...]:
    names = [
        f"num__{name}"
        for name in NUMERIC_FEATURES
    ]

    names.extend(
        f"bool__{name}"
        for name in BOOLEAN_FEATURES
    )

    for column in CATEGORICAL_FEATURES:
        names.extend(
            f"cat__{column}_{category}"
            for category in (
                *category_vocab[column],
                UNKNOWN_TOKEN,
            )
        )

    return tuple(names)


def _format_timestamp(
    value: datetime,
) -> str:
    return value.isoformat(
        sep=" ",
        timespec="seconds",
    )


@dataclass(frozen=True, slots=True)
class RebuildPreprocessor:
    """Transformer backed by preprocessing state learned in a rebuild run."""

    numeric_mean: np.ndarray
    numeric_scale: np.ndarray
    category_vocab: dict[str, tuple[str, ...]]
    feature_names: tuple[str, ...]

    def transform(
        self,
        rows: Sequence[SemanticFeatureRow],
    ) -> sparse.csr_matrix:
        if not rows:
            return sparse.csr_matrix(
                (0, len(self.feature_names)),
                dtype=np.float32,
            )

        numeric = np.full(
            (
                len(rows),
                len(NUMERIC_FEATURES),
            ),
            np.nan,
            dtype=np.float64,
        )

        boolean = np.empty(
            (
                len(rows),
                len(BOOLEAN_FEATURES),
            ),
            dtype=np.float32,
        )

        categorical_parts: list[
            sparse.csr_matrix
        ] = []

        mappings = []

        for row_index, row in enumerate(rows):
            if not isinstance(
                row,
                SemanticFeatureRow,
            ):
                raise PreprocessingContractError(
                    "Rebuild transform accepts only SemanticFeatureRow values."
                )

            mapping = row.as_dict()
            mappings.append(mapping)

            for column_index, column in enumerate(
                NUMERIC_FEATURES
            ):
                value = mapping[column]

                if value is None:
                    if column not in _STRUCTURAL_NA_FEATURES:
                        raise PreprocessingContractError(
                            f"Unexpected missing numeric feature: {column}."
                        )
                    continue

                numeric_value = float(
                    value
                )

                if not np.isfinite(
                    numeric_value
                ):
                    raise PreprocessingContractError(
                        f"Non-finite numeric feature: {column}."
                    )

                numeric[
                    row_index,
                    column_index,
                ] = numeric_value

            for column_index, column in enumerate(
                BOOLEAN_FEATURES
            ):
                value = mapping[column]

                if not isinstance(
                    value,
                    bool,
                ):
                    raise PreprocessingContractError(
                        f"Boolean feature must be bool: {column}."
                    )

                boolean[
                    row_index,
                    column_index,
                ] = float(value)

        numeric_scaled = (
            numeric
            - self.numeric_mean
        ) / self.numeric_scale

        numeric_scaled = np.nan_to_num(
            numeric_scaled,
            nan=0.0,
            posinf=np.inf,
            neginf=-np.inf,
        ).astype(
            np.float32
        )

        if not np.isfinite(
            numeric_scaled
        ).all():
            raise PreprocessingContractError(
                "Non-finite numeric value after rebuild scaling."
            )

        for column in CATEGORICAL_FEATURES:
            observed = self.category_vocab[
                column
            ]

            categories = (
                *observed,
                UNKNOWN_TOKEN,
            )

            category_to_index = {
                category: index
                for index, category
                in enumerate(categories)
            }

            row_indices = np.arange(
                len(rows),
                dtype=np.int32,
            )

            column_indices = np.empty(
                len(rows),
                dtype=np.int32,
            )

            for row_index, mapping in enumerate(
                mappings
            ):
                value = mapping[column]

                if not isinstance(
                    value,
                    str,
                ) or not value:
                    raise PreprocessingContractError(
                        f"Invalid categorical feature: {column}."
                    )

                mapped = (
                    value
                    if value in category_to_index
                    and value != UNKNOWN_TOKEN
                    else UNKNOWN_TOKEN
                )

                column_indices[
                    row_index
                ] = category_to_index[
                    mapped
                ]

            categorical_parts.append(
                sparse.csr_matrix(
                    (
                        np.ones(
                            len(rows),
                            dtype=np.float32,
                        ),
                        (
                            row_indices,
                            column_indices,
                        ),
                    ),
                    shape=(
                        len(rows),
                        len(categories),
                    ),
                    dtype=np.float32,
                )
            )

        matrix = sparse.hstack(
            [
                sparse.csr_matrix(
                    numeric_scaled,
                    dtype=np.float32,
                ),
                sparse.csr_matrix(
                    boolean,
                    dtype=np.float32,
                ),
                *categorical_parts,
            ],
            format="csr",
            dtype=np.float32,
        )

        if matrix.shape[1] != 47:
            raise PreprocessingContractError(
                "Rebuild transform must emit exactly 47 columns."
            )

        if not np.isfinite(
            matrix.data
        ).all():
            raise PreprocessingContractError(
                "Rebuild transform emitted NaN or infinity."
            )

        return matrix


@dataclass(frozen=True, slots=True)
class RebuildPreprocessingFit:
    """Learned rebuild state plus the transformer that uses it."""

    state: dict
    preprocessor: RebuildPreprocessor


class RebuildPreprocessingFitter:
    """Incrementally fit W_SHORT preprocessing from TRAIN-only semantic rows."""

    def __init__(self) -> None:
        self._scalers = [
            StandardScaler()
            for _ in NUMERIC_FEATURES
        ]

        self._category_values = {
            column: set()
            for column in CATEGORICAL_FEATURES
        }

        self._fit_row_count = 0
        self._fit_min_timestamp: datetime | None = None
        self._fit_max_timestamp: datetime | None = None

    @property
    def fit_row_count(self) -> int:
        return self._fit_row_count

    def partial_fit(
        self,
        rows: Sequence[SemanticFeatureRow],
        timestamps: Sequence[datetime],
    ) -> "RebuildPreprocessingFitter":
        if len(rows) != len(timestamps):
            raise PreprocessingContractError(
                "rows and timestamps must have the same length."
            )

        if not rows:
            return self

        for timestamp in timestamps:
            if not isinstance(
                timestamp,
                datetime,
            ):
                raise PreprocessingContractError(
                    "Every training timestamp must be a datetime."
                )

        batch_min = min(timestamps)
        batch_max = max(timestamps)

        # No global chronological-batch requirement here.
        #
        # Canonical replay is ordered by contiguous User+Card blocks.
        # Timestamp monotonicity belongs to each history block before
        # semantic features are emitted. Preprocessing learned statistics
        # are order-invariant across already-canonical semantic rows.

        numeric_matrix = np.full(
            (
                len(rows),
                len(NUMERIC_FEATURES),
            ),
            np.nan,
            dtype=np.float64,
        )

        for row_index, row in enumerate(rows):
            if not isinstance(
                row,
                SemanticFeatureRow,
            ):
                raise PreprocessingContractError(
                    "Rebuild preprocessing accepts only SemanticFeatureRow values."
                )

            mapping = row.as_dict()

            for column_index, column in enumerate(
                NUMERIC_FEATURES
            ):
                value = mapping[column]

                if value is None:
                    if column not in _STRUCTURAL_NA_FEATURES:
                        raise PreprocessingContractError(
                            f"Unexpected missing numeric feature: {column}."
                        )
                    continue

                numeric = float(value)

                if not np.isfinite(numeric):
                    raise PreprocessingContractError(
                        f"Non-finite numeric feature: {column}."
                    )

                numeric_matrix[
                    row_index,
                    column_index,
                ] = numeric

            for column in CATEGORICAL_FEATURES:
                value = mapping[column]

                if not isinstance(
                    value,
                    str,
                ) or not value:
                    raise PreprocessingContractError(
                        f"Invalid categorical feature: {column}."
                    )

                if value == UNKNOWN_TOKEN:
                    raise PreprocessingContractError(
                        f"Reserved token observed in TRAIN: {column}."
                    )

                self._category_values[
                    column
                ].add(value)

        for column_index, scaler in enumerate(
            self._scalers
        ):
            values = numeric_matrix[
                :,
                column_index,
            ]

            finite_mask = np.isfinite(
                values
            )

            if not finite_mask.any():
                continue

            scaler.partial_fit(
                values[
                    finite_mask
                ].reshape(
                    -1,
                    1,
                )
            )

        self._fit_row_count += len(rows)

        if (
            self._fit_min_timestamp is None
            or batch_min
            < self._fit_min_timestamp
        ):
            self._fit_min_timestamp = (
                batch_min
            )

        if (
            self._fit_max_timestamp is None
            or batch_max
            > self._fit_max_timestamp
        ):
            self._fit_max_timestamp = (
                batch_max
            )

        return self

    def finalize(
        self,
    ) -> RebuildPreprocessingFit:
        if self._fit_row_count <= 0:
            raise PreprocessingContractError(
                "Cannot finalize preprocessing without TRAIN rows."
            )

        if (
            self._fit_min_timestamp is None
            or self._fit_max_timestamp is None
        ):
            raise PreprocessingContractError(
                "Training timestamp range is incomplete."
            )

        numeric_mean: list[float] = []
        numeric_var: list[float] = []
        numeric_scale: list[float] = []
        numeric_n_samples_seen: list[int] = []

        for column, scaler in zip(
            NUMERIC_FEATURES,
            self._scalers,
        ):
            if not hasattr(
                scaler,
                "n_samples_seen_",
            ):
                raise PreprocessingContractError(
                    f"No observed TRAIN value for numeric feature: {column}."
                )

            count = int(
                np.asarray(
                    scaler.n_samples_seen_
                ).reshape(-1)[0]
            )
            mean = float(
                np.asarray(
                    scaler.mean_
                ).reshape(-1)[0]
            )
            variance = float(
                np.asarray(
                    scaler.var_
                ).reshape(-1)[0]
            )
            scale = float(
                np.asarray(
                    scaler.scale_
                ).reshape(-1)[0]
            )

            if (
                count <= 0
                or not np.isfinite(mean)
                or not np.isfinite(variance)
                or not np.isfinite(scale)
                or scale <= 0.0
            ):
                raise PreprocessingContractError(
                    f"Invalid learned numeric state: {column}."
                )

            numeric_n_samples_seen.append(
                count
            )
            numeric_mean.append(mean)
            numeric_var.append(
                variance
            )
            numeric_scale.append(
                scale
            )

        category_vocab: dict[
            str,
            tuple[str, ...],
        ] = {}

        for column in CATEGORICAL_FEATURES:
            observed = _stable_sort_categories(
                column,
                self._category_values[
                    column
                ],
            )

            expected = (
                EXPECTED_CATEGORY_VOCAB[
                    column
                ]
            )

            if observed != expected:
                raise PreprocessingContractError(
                    f"TRAIN categorical vocabulary mismatch: {column}."
                )

            category_vocab[
                column
            ] = observed

        feature_names = (
            _build_feature_names(
                category_vocab
            )
        )

        if len(feature_names) != 47:
            raise PreprocessingContractError(
                "Rebuild preprocessing schema must contain exactly 47 features."
            )

        state = {
            "analysis_version":
                "rebuild-preprocessing-v1",
            "categorical_columns":
                list(
                    CATEGORICAL_FEATURES
                ),
            "category_vocab": {
                column: list(
                    category_vocab[
                        column
                    ]
                )
                for column
                in CATEGORICAL_FEATURES
            },
            "feature_count":
                47,
            "feature_names":
                list(feature_names),
            "fit_max_timestamp":
                _format_timestamp(
                    self._fit_max_timestamp
                ),
            "fit_min_timestamp":
                _format_timestamp(
                    self._fit_min_timestamp
                ),
            "fit_row_count":
                self._fit_row_count,
            "fit_source":
                REBUILD_FIT_SOURCE,
            "matrix_dtype":
                "float32",
            "matrix_format":
                "CSR",
            "numeric_columns":
                list(
                    NUMERIC_FEATURES
                ),
            "numeric_mean":
                numeric_mean,
            "numeric_n_samples_seen":
                numeric_n_samples_seen,
            "numeric_scale":
                numeric_scale,
            "numeric_var":
                numeric_var,
            "strategy":
                REBUILD_STRATEGY,
            "unknown_token":
                UNKNOWN_TOKEN,
            "rebuild_validation_status":
                "NOT_COMPARED_TO_OFFICIAL_ARTIFACT",
        }

        preprocessor = (
            RebuildPreprocessor(
                numeric_mean=np.asarray(
                    numeric_mean,
                    dtype=np.float64,
                ),
                numeric_scale=np.asarray(
                    numeric_scale,
                    dtype=np.float64,
                ),
                category_vocab=
                    category_vocab,
                feature_names=
                    feature_names,
            )
        )

        return RebuildPreprocessingFit(
            state=state,
            preprocessor=preprocessor,
        )


def fit_rebuild_preprocessor(
    rows: Sequence[SemanticFeatureRow],
    timestamps: Sequence[datetime],
) -> RebuildPreprocessingFit:
    """Fit one TRAIN-only preprocessing state in a single batch."""

    fitter = (
        RebuildPreprocessingFitter()
    )

    fitter.partial_fit(
        rows,
        timestamps,
    )

    return fitter.finalize()
