"""Read-only transformation using an already-frozen preprocessing state."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy import sparse

from fraud_screening.errors import PreprocessingContractError
from fraud_screening.features import (
    BOOLEAN_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    SEMANTIC_FEATURE_ORDER,
    SemanticFeatureRow,
)


EXPECTED_FEATURE_COUNT = 47
EXPECTED_MATRIX_FORMAT = "CSR"
EXPECTED_MATRIX_DTYPE = "float32"
EXPECTED_STRATEGY = "W_SHORT"
EXPECTED_FIT_SOURCE = "W_SHORT_TRAIN_ONLY"

_STRUCTURAL_NA_FEATURES = {
    "time_since_previous_transaction_min",
    "amount_minus_previous_mean",
}


def _require_exact_list(
    state: dict[str, Any],
    key: str,
    expected: tuple[str, ...],
) -> None:
    value = state.get(key)

    if value != list(expected):
        raise PreprocessingContractError(
            f"{key} does not match the frozen semantic contract."
        )


def _as_finite_vector(
    state: dict[str, Any],
    key: str,
    width: int,
) -> np.ndarray:
    value = state.get(key)

    if not isinstance(value, list) or len(value) != width:
        raise PreprocessingContractError(
            f"{key} must contain exactly {width} values."
        )

    array = np.asarray(value, dtype=np.float64)

    if not np.isfinite(array).all():
        raise PreprocessingContractError(
            f"{key} must contain only finite values."
        )

    return array


@dataclass(frozen=True, slots=True)
class FrozenPreprocessor:
    """Immutable transform state for the exact 10-to-47 representation."""

    numeric_mean: np.ndarray
    numeric_scale: np.ndarray
    category_vocab: dict[str, tuple[str, ...]]
    unknown_token: str
    feature_names: tuple[str, ...]
    strategy: str
    fit_source: str

    @classmethod
    def from_state_dict(
        cls,
        state: dict[str, Any],
    ) -> "FrozenPreprocessor":
        if not isinstance(state, dict):
            raise PreprocessingContractError(
                "Preprocessing state must be a JSON object."
            )

        _require_exact_list(
            state,
            "numeric_columns",
            NUMERIC_FEATURES,
        )
        _require_exact_list(
            state,
            "categorical_columns",
            CATEGORICAL_FEATURES,
        )

        if state.get("feature_count") != EXPECTED_FEATURE_COUNT:
            raise PreprocessingContractError(
                "Frozen preprocessing feature_count must be 47."
            )

        if state.get("matrix_format") != EXPECTED_MATRIX_FORMAT:
            raise PreprocessingContractError(
                "Frozen preprocessing matrix format must be CSR."
            )

        if state.get("matrix_dtype") != EXPECTED_MATRIX_DTYPE:
            raise PreprocessingContractError(
                "Frozen preprocessing matrix dtype must be float32."
            )

        if state.get("strategy") != EXPECTED_STRATEGY:
            raise PreprocessingContractError(
                "Unexpected preprocessing strategy."
            )

        if state.get("fit_source") != EXPECTED_FIT_SOURCE:
            raise PreprocessingContractError(
                "Unexpected preprocessing fit source."
            )

        if state.get("validation_exact_reproduction") is not True:
            raise PreprocessingContractError(
                "Frozen state is not marked as exact validation reproduction."
            )

        numeric_mean = _as_finite_vector(
            state,
            "numeric_mean",
            len(NUMERIC_FEATURES),
        )
        numeric_scale = _as_finite_vector(
            state,
            "numeric_scale",
            len(NUMERIC_FEATURES),
        )

        if (numeric_scale <= 0.0).any():
            raise PreprocessingContractError(
                "All numeric scales must be strictly positive."
            )

        unknown_token = state.get("unknown_token")

        if (
            not isinstance(unknown_token, str)
            or not unknown_token
        ):
            raise PreprocessingContractError(
                "unknown_token must be a non-empty string."
            )

        raw_vocab = state.get("category_vocab")

        if not isinstance(raw_vocab, dict):
            raise PreprocessingContractError(
                "category_vocab must be a JSON object."
            )

        if set(raw_vocab) != set(CATEGORICAL_FEATURES):
            raise PreprocessingContractError(
                "category_vocab keys do not match categorical features."
            )

        category_vocab: dict[str, tuple[str, ...]] = {}

        for column in CATEGORICAL_FEATURES:
            values = raw_vocab.get(column)

            if (
                not isinstance(values, list)
                or not values
                or not all(
                    isinstance(value, str)
                    for value in values
                )
            ):
                raise PreprocessingContractError(
                    f"Invalid category vocabulary for {column}."
                )

            if len(values) != len(set(values)):
                raise PreprocessingContractError(
                    f"Duplicate category in vocabulary for {column}."
                )

            if unknown_token in values:
                raise PreprocessingContractError(
                    f"Reserved unknown token appears in observed vocabulary: {column}."
                )

            category_vocab[column] = tuple(values)

        feature_names_value = state.get("feature_names")

        if (
            not isinstance(feature_names_value, list)
            or not all(
                isinstance(name, str)
                for name in feature_names_value
            )
        ):
            raise PreprocessingContractError(
                "feature_names must be a list of strings."
            )

        feature_names = tuple(feature_names_value)

        expected_names = cls._build_feature_names(
            category_vocab=category_vocab,
            unknown_token=unknown_token,
        )

        if feature_names != expected_names:
            raise PreprocessingContractError(
                "Frozen feature_names do not match vocabulary-derived schema."
            )

        if len(feature_names) != EXPECTED_FEATURE_COUNT:
            raise PreprocessingContractError(
                "Frozen output schema must contain exactly 47 features."
            )

        return cls(
            numeric_mean=numeric_mean,
            numeric_scale=numeric_scale,
            category_vocab=category_vocab,
            unknown_token=unknown_token,
            feature_names=feature_names,
            strategy=state["strategy"],
            fit_source=state["fit_source"],
        )

    @staticmethod
    def _build_feature_names(
        *,
        category_vocab: dict[str, tuple[str, ...]],
        unknown_token: str,
    ) -> tuple[str, ...]:
        names: list[str] = [
            f"num__{name}"
            for name in NUMERIC_FEATURES
        ]

        names.extend(
            f"bool__{name}"
            for name in BOOLEAN_FEATURES
        )

        for column in CATEGORICAL_FEATURES:
            for category in (
                *category_vocab[column],
                unknown_token,
            ):
                names.append(
                    f"cat__{column}_{category}"
                )

        return tuple(names)

    @classmethod
    def from_json(
        cls,
        path: str | Path,
    ) -> "FrozenPreprocessor":
        state_path = Path(path)

        try:
            state = json.loads(
                state_path.read_text(
                    encoding="utf-8"
                )
            )
        except FileNotFoundError:
            raise
        except Exception as exc:
            raise PreprocessingContractError(
                "Unable to read frozen preprocessing JSON."
            ) from exc

        return cls.from_state_dict(state)

    def _numeric_matrix(
        self,
        rows: list[SemanticFeatureRow],
    ) -> np.ndarray:
        matrix = np.empty(
            (len(rows), len(NUMERIC_FEATURES)),
            dtype=np.float64,
        )

        for row_index, row in enumerate(rows):
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

                    matrix[
                        row_index,
                        column_index,
                    ] = np.nan
                    continue

                if isinstance(value, bool):
                    raise PreprocessingContractError(
                        f"Boolean value supplied for numeric feature: {column}."
                    )

                try:
                    numeric_value = float(value)
                except (TypeError, ValueError) as exc:
                    raise PreprocessingContractError(
                        f"Non-numeric value for feature: {column}."
                    ) from exc

                if not math.isfinite(numeric_value):
                    raise PreprocessingContractError(
                        f"Non-finite numeric feature: {column}."
                    )

                matrix[
                    row_index,
                    column_index,
                ] = numeric_value

        finite_mask = np.isfinite(matrix)

        transformed = np.full(
            matrix.shape,
            np.nan,
            dtype=np.float64,
        )

        if finite_mask.any():
            for index in range(
                len(NUMERIC_FEATURES)
            ):
                column_mask = finite_mask[:, index]

                if not column_mask.any():
                    continue

                transformed[
                    column_mask,
                    index,
                ] = (
                    matrix[column_mask, index]
                    - self.numeric_mean[index]
                ) / self.numeric_scale[index]

        return np.nan_to_num(
            transformed,
            nan=0.0,
            posinf=np.inf,
            neginf=-np.inf,
        ).astype(np.float32)

    @staticmethod
    def _boolean_matrix(
        rows: list[SemanticFeatureRow],
    ) -> np.ndarray:
        matrix = np.empty(
            (len(rows), len(BOOLEAN_FEATURES)),
            dtype=np.float32,
        )

        for row_index, row in enumerate(rows):
            mapping = row.as_dict()

            for column_index, column in enumerate(
                BOOLEAN_FEATURES
            ):
                value = mapping[column]

                if not isinstance(value, bool):
                    raise PreprocessingContractError(
                        f"Boolean feature is not bool: {column}."
                    )

                matrix[
                    row_index,
                    column_index,
                ] = 1.0 if value else 0.0

        return matrix

    def _categorical_matrix(
        self,
        rows: list[SemanticFeatureRow],
    ) -> sparse.csr_matrix:
        total_width = sum(
            len(self.category_vocab[column]) + 1
            for column in CATEGORICAL_FEATURES
        )

        row_indices: list[int] = []
        column_indices: list[int] = []
        data: list[float] = []

        branch_offset = 0

        for column in CATEGORICAL_FEATURES:
            known_values = self.category_vocab[column]
            lookup = {
                value: index
                for index, value in enumerate(
                    known_values
                )
            }
            unknown_index = len(known_values)

            for row_index, row in enumerate(rows):
                value = row.as_dict()[column]

                if value is None:
                    raise PreprocessingContractError(
                        f"Unexpected missing categorical feature: {column}."
                    )

                value = str(value)

                local_index = lookup.get(
                    value,
                    unknown_index,
                )

                row_indices.append(row_index)
                column_indices.append(
                    branch_offset + local_index
                )
                data.append(1.0)

            branch_offset += (
                len(known_values) + 1
            )

        return sparse.csr_matrix(
            (
                np.asarray(
                    data,
                    dtype=np.float32,
                ),
                (
                    np.asarray(
                        row_indices,
                        dtype=np.int64,
                    ),
                    np.asarray(
                        column_indices,
                        dtype=np.int64,
                    ),
                ),
            ),
            shape=(len(rows), total_width),
            dtype=np.float32,
        )

    def transform(
        self,
        rows: Iterable[SemanticFeatureRow],
    ) -> sparse.csr_matrix:
        """Transform exact semantic rows using frozen learned state only."""

        semantic_rows = list(rows)

        if not semantic_rows:
            return sparse.csr_matrix(
                (0, EXPECTED_FEATURE_COUNT),
                dtype=np.float32,
            )

        for row in semantic_rows:
            if not isinstance(
                row,
                SemanticFeatureRow,
            ):
                raise PreprocessingContractError(
                    "transform expects SemanticFeatureRow instances."
                )

            if tuple(row.as_dict()) != SEMANTIC_FEATURE_ORDER:
                raise PreprocessingContractError(
                    "Semantic feature order mismatch."
                )

        numeric_matrix = self._numeric_matrix(
            semantic_rows
        )

        boolean_matrix = self._boolean_matrix(
            semantic_rows
        )

        categorical_matrix = self._categorical_matrix(
            semantic_rows
        )

        matrix = sparse.hstack(
            [
                sparse.csr_matrix(
                    numeric_matrix,
                    dtype=np.float32,
                ),
                sparse.csr_matrix(
                    boolean_matrix,
                    dtype=np.float32,
                ),
                categorical_matrix,
            ],
            format="csr",
            dtype=np.float32,
        )

        if matrix.shape[1] != EXPECTED_FEATURE_COUNT:
            raise PreprocessingContractError(
                "Preprocessing output width is not 47."
            )

        if matrix.shape[1] != len(
            self.feature_names
        ):
            raise PreprocessingContractError(
                "Output width does not match frozen feature names."
            )

        if not np.isfinite(matrix.data).all():
            raise PreprocessingContractError(
                "Non-finite value produced by frozen preprocessing."
            )

        return matrix


def load_frozen_preprocessor(
    path: str | Path,
) -> FrozenPreprocessor:
    """Load a frozen preprocessing JSON without fitting any learned state."""

    return FrozenPreprocessor.from_json(path)
