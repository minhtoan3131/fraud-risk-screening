from __future__ import annotations

import argparse
from pathlib import Path

INIT_REL = Path(
    "final_pipeline/src/fraud_screening/training/__init__.py"
)

EXPECTED_INIT = '"""Rebuild-only training utilities.\n\nThis package creates new learned state only for explicit rebuild runs.\nIt never writes to the official artifact directory.\n"""\n\nfrom fraud_screening.training.factory import (\n    REBUILD_MODEL_ID,\n    build_rebuild_model,\n)\nfrom fraud_screening.training.outputs import (\n    RebuildOutputPaths,\n    create_rebuild_output_paths,\n    validate_run_id,\n)\n\n__all__ = [\n    "REBUILD_MODEL_ID",\n    "RebuildOutputPaths",\n    "build_rebuild_model",\n    "create_rebuild_output_paths",\n    "validate_run_id",\n]\n'
UPDATED_INIT = '"""Rebuild-only training utilities.\n\nThis package creates new learned state only for explicit rebuild runs.\nIt never writes to the official artifact directory.\n"""\n\nfrom fraud_screening.training.factory import (\n    REBUILD_MODEL_ID,\n    build_rebuild_model,\n)\nfrom fraud_screening.training.outputs import (\n    RebuildOutputPaths,\n    create_rebuild_output_paths,\n    validate_run_id,\n)\nfrom fraud_screening.training.preprocessing_fit import (\n    RebuildPreprocessingFit,\n    RebuildPreprocessingFitter,\n    RebuildPreprocessor,\n    fit_rebuild_preprocessor,\n)\n\n__all__ = [\n    "REBUILD_MODEL_ID",\n    "RebuildOutputPaths",\n    "RebuildPreprocessingFit",\n    "RebuildPreprocessingFitter",\n    "RebuildPreprocessor",\n    "build_rebuild_model",\n    "create_rebuild_output_paths",\n    "fit_rebuild_preprocessor",\n    "validate_run_id",\n]\n'

NEW_FILES = {
    "final_pipeline/src/fraud_screening/training/preprocessing_fit.py":
        '"""TRAIN-only learned preprocessing for rebuild runs."""\n\nfrom __future__ import annotations\n\nfrom dataclasses import dataclass\nfrom datetime import datetime\nfrom typing import Iterable, Sequence\n\nimport numpy as np\nfrom scipy import sparse\nfrom sklearn.preprocessing import StandardScaler\n\nfrom fraud_screening.errors import PreprocessingContractError\nfrom fraud_screening.features import (\n    BOOLEAN_FEATURES,\n    CATEGORICAL_FEATURES,\n    NUMERIC_FEATURES,\n    SemanticFeatureRow,\n)\n\n\nUNKNOWN_TOKEN = "__UNKNOWN__"\nREBUILD_STRATEGY = "W_SHORT"\nREBUILD_FIT_SOURCE = "W_SHORT_TRAIN_ONLY"\n\nEXPECTED_CATEGORY_VOCAB = {\n    "transaction_mode": (\n        "Chip Transaction",\n        "Online Transaction",\n        "Swipe Transaction",\n    ),\n    "location_state": (\n        "NON_PHYSICAL_OR_ONLINE",\n        "PHYSICAL_COMPLETE",\n        "PHYSICAL_ZIP_UNAVAILABLE",\n    ),\n    "hour_of_day": tuple(\n        str(value)\n        for value in range(24)\n    ),\n    "day_of_week": tuple(\n        str(value)\n        for value in range(7)\n    ),\n}\n\n_STRUCTURAL_NA_FEATURES = {\n    "time_since_previous_transaction_min",\n    "amount_minus_previous_mean",\n}\n\n\ndef _stable_sort_categories(\n    column: str,\n    values: Iterable[str],\n) -> tuple[str, ...]:\n    normalized = [\n        str(value)\n        for value in values\n    ]\n\n    if column in {\n        "hour_of_day",\n        "day_of_week",\n    }:\n        try:\n            return tuple(\n                sorted(\n                    normalized,\n                    key=lambda value: int(value),\n                )\n            )\n        except ValueError as exc:\n            raise PreprocessingContractError(\n                f"Non-integer categorical value for {column}."\n            ) from exc\n\n    return tuple(\n        sorted(normalized)\n    )\n\n\ndef _build_feature_names(\n    category_vocab: dict[str, tuple[str, ...]],\n) -> tuple[str, ...]:\n    names = [\n        f"num__{name}"\n        for name in NUMERIC_FEATURES\n    ]\n\n    names.extend(\n        f"bool__{name}"\n        for name in BOOLEAN_FEATURES\n    )\n\n    for column in CATEGORICAL_FEATURES:\n        names.extend(\n            f"cat__{column}_{category}"\n            for category in (\n                *category_vocab[column],\n                UNKNOWN_TOKEN,\n            )\n        )\n\n    return tuple(names)\n\n\ndef _format_timestamp(\n    value: datetime,\n) -> str:\n    return value.isoformat(\n        sep=" ",\n        timespec="seconds",\n    )\n\n\n@dataclass(frozen=True, slots=True)\nclass RebuildPreprocessor:\n    """Transformer backed by preprocessing state learned in a rebuild run."""\n\n    numeric_mean: np.ndarray\n    numeric_scale: np.ndarray\n    category_vocab: dict[str, tuple[str, ...]]\n    feature_names: tuple[str, ...]\n\n    def transform(\n        self,\n        rows: Sequence[SemanticFeatureRow],\n    ) -> sparse.csr_matrix:\n        if not rows:\n            return sparse.csr_matrix(\n                (0, len(self.feature_names)),\n                dtype=np.float32,\n            )\n\n        numeric = np.full(\n            (\n                len(rows),\n                len(NUMERIC_FEATURES),\n            ),\n            np.nan,\n            dtype=np.float64,\n        )\n\n        boolean = np.empty(\n            (\n                len(rows),\n                len(BOOLEAN_FEATURES),\n            ),\n            dtype=np.float32,\n        )\n\n        categorical_parts: list[\n            sparse.csr_matrix\n        ] = []\n\n        mappings = []\n\n        for row_index, row in enumerate(rows):\n            if not isinstance(\n                row,\n                SemanticFeatureRow,\n            ):\n                raise PreprocessingContractError(\n                    "Rebuild transform accepts only SemanticFeatureRow values."\n                )\n\n            mapping = row.as_dict()\n            mappings.append(mapping)\n\n            for column_index, column in enumerate(\n                NUMERIC_FEATURES\n            ):\n                value = mapping[column]\n\n                if value is None:\n                    if column not in _STRUCTURAL_NA_FEATURES:\n                        raise PreprocessingContractError(\n                            f"Unexpected missing numeric feature: {column}."\n                        )\n                    continue\n\n                numeric_value = float(\n                    value\n                )\n\n                if not np.isfinite(\n                    numeric_value\n                ):\n                    raise PreprocessingContractError(\n                        f"Non-finite numeric feature: {column}."\n                    )\n\n                numeric[\n                    row_index,\n                    column_index,\n                ] = numeric_value\n\n            for column_index, column in enumerate(\n                BOOLEAN_FEATURES\n            ):\n                value = mapping[column]\n\n                if not isinstance(\n                    value,\n                    bool,\n                ):\n                    raise PreprocessingContractError(\n                        f"Boolean feature must be bool: {column}."\n                    )\n\n                boolean[\n                    row_index,\n                    column_index,\n                ] = float(value)\n\n        numeric_scaled = (\n            numeric\n            - self.numeric_mean\n        ) / self.numeric_scale\n\n        numeric_scaled = np.nan_to_num(\n            numeric_scaled,\n            nan=0.0,\n            posinf=np.inf,\n            neginf=-np.inf,\n        ).astype(\n            np.float32\n        )\n\n        if not np.isfinite(\n            numeric_scaled\n        ).all():\n            raise PreprocessingContractError(\n                "Non-finite numeric value after rebuild scaling."\n            )\n\n        for column in CATEGORICAL_FEATURES:\n            observed = self.category_vocab[\n                column\n            ]\n\n            categories = (\n                *observed,\n                UNKNOWN_TOKEN,\n            )\n\n            category_to_index = {\n                category: index\n                for index, category\n                in enumerate(categories)\n            }\n\n            row_indices = np.arange(\n                len(rows),\n                dtype=np.int32,\n            )\n\n            column_indices = np.empty(\n                len(rows),\n                dtype=np.int32,\n            )\n\n            for row_index, mapping in enumerate(\n                mappings\n            ):\n                value = mapping[column]\n\n                if not isinstance(\n                    value,\n                    str,\n                ) or not value:\n                    raise PreprocessingContractError(\n                        f"Invalid categorical feature: {column}."\n                    )\n\n                mapped = (\n                    value\n                    if value in category_to_index\n                    and value != UNKNOWN_TOKEN\n                    else UNKNOWN_TOKEN\n                )\n\n                column_indices[\n                    row_index\n                ] = category_to_index[\n                    mapped\n                ]\n\n            categorical_parts.append(\n                sparse.csr_matrix(\n                    (\n                        np.ones(\n                            len(rows),\n                            dtype=np.float32,\n                        ),\n                        (\n                            row_indices,\n                            column_indices,\n                        ),\n                    ),\n                    shape=(\n                        len(rows),\n                        len(categories),\n                    ),\n                    dtype=np.float32,\n                )\n            )\n\n        matrix = sparse.hstack(\n            [\n                sparse.csr_matrix(\n                    numeric_scaled,\n                    dtype=np.float32,\n                ),\n                sparse.csr_matrix(\n                    boolean,\n                    dtype=np.float32,\n                ),\n                *categorical_parts,\n            ],\n            format="csr",\n            dtype=np.float32,\n        )\n\n        if matrix.shape[1] != 47:\n            raise PreprocessingContractError(\n                "Rebuild transform must emit exactly 47 columns."\n            )\n\n        if not np.isfinite(\n            matrix.data\n        ).all():\n            raise PreprocessingContractError(\n                "Rebuild transform emitted NaN or infinity."\n            )\n\n        return matrix\n\n\n@dataclass(frozen=True, slots=True)\nclass RebuildPreprocessingFit:\n    """Learned rebuild state plus the transformer that uses it."""\n\n    state: dict\n    preprocessor: RebuildPreprocessor\n\n\nclass RebuildPreprocessingFitter:\n    """Incrementally fit W_SHORT preprocessing from TRAIN-only semantic rows."""\n\n    def __init__(self) -> None:\n        self._scalers = [\n            StandardScaler()\n            for _ in NUMERIC_FEATURES\n        ]\n\n        self._category_values = {\n            column: set()\n            for column in CATEGORICAL_FEATURES\n        }\n\n        self._fit_row_count = 0\n        self._fit_min_timestamp: datetime | None = None\n        self._fit_max_timestamp: datetime | None = None\n\n    @property\n    def fit_row_count(self) -> int:\n        return self._fit_row_count\n\n    def partial_fit(\n        self,\n        rows: Sequence[SemanticFeatureRow],\n        timestamps: Sequence[datetime],\n    ) -> "RebuildPreprocessingFitter":\n        if len(rows) != len(timestamps):\n            raise PreprocessingContractError(\n                "rows and timestamps must have the same length."\n            )\n\n        if not rows:\n            return self\n\n        for timestamp in timestamps:\n            if not isinstance(\n                timestamp,\n                datetime,\n            ):\n                raise PreprocessingContractError(\n                    "Every training timestamp must be a datetime."\n                )\n\n        batch_min = min(timestamps)\n        batch_max = max(timestamps)\n\n        if (\n            self._fit_max_timestamp is not None\n            and batch_min < self._fit_max_timestamp\n        ):\n            raise PreprocessingContractError(\n                "Rebuild preprocessing batches must be chronological."\n            )\n\n        numeric_matrix = np.full(\n            (\n                len(rows),\n                len(NUMERIC_FEATURES),\n            ),\n            np.nan,\n            dtype=np.float64,\n        )\n\n        for row_index, row in enumerate(rows):\n            if not isinstance(\n                row,\n                SemanticFeatureRow,\n            ):\n                raise PreprocessingContractError(\n                    "Rebuild preprocessing accepts only SemanticFeatureRow values."\n                )\n\n            mapping = row.as_dict()\n\n            for column_index, column in enumerate(\n                NUMERIC_FEATURES\n            ):\n                value = mapping[column]\n\n                if value is None:\n                    if column not in _STRUCTURAL_NA_FEATURES:\n                        raise PreprocessingContractError(\n                            f"Unexpected missing numeric feature: {column}."\n                        )\n                    continue\n\n                numeric = float(value)\n\n                if not np.isfinite(numeric):\n                    raise PreprocessingContractError(\n                        f"Non-finite numeric feature: {column}."\n                    )\n\n                numeric_matrix[\n                    row_index,\n                    column_index,\n                ] = numeric\n\n            for column in CATEGORICAL_FEATURES:\n                value = mapping[column]\n\n                if not isinstance(\n                    value,\n                    str,\n                ) or not value:\n                    raise PreprocessingContractError(\n                        f"Invalid categorical feature: {column}."\n                    )\n\n                if value == UNKNOWN_TOKEN:\n                    raise PreprocessingContractError(\n                        f"Reserved token observed in TRAIN: {column}."\n                    )\n\n                self._category_values[\n                    column\n                ].add(value)\n\n        for column_index, scaler in enumerate(\n            self._scalers\n        ):\n            values = numeric_matrix[\n                :,\n                column_index,\n            ]\n\n            finite_mask = np.isfinite(\n                values\n            )\n\n            if not finite_mask.any():\n                continue\n\n            scaler.partial_fit(\n                values[\n                    finite_mask\n                ].reshape(\n                    -1,\n                    1,\n                )\n            )\n\n        self._fit_row_count += len(rows)\n\n        if (\n            self._fit_min_timestamp is None\n            or batch_min\n            < self._fit_min_timestamp\n        ):\n            self._fit_min_timestamp = (\n                batch_min\n            )\n\n        if (\n            self._fit_max_timestamp is None\n            or batch_max\n            > self._fit_max_timestamp\n        ):\n            self._fit_max_timestamp = (\n                batch_max\n            )\n\n        return self\n\n    def finalize(\n        self,\n    ) -> RebuildPreprocessingFit:\n        if self._fit_row_count <= 0:\n            raise PreprocessingContractError(\n                "Cannot finalize preprocessing without TRAIN rows."\n            )\n\n        if (\n            self._fit_min_timestamp is None\n            or self._fit_max_timestamp is None\n        ):\n            raise PreprocessingContractError(\n                "Training timestamp range is incomplete."\n            )\n\n        numeric_mean: list[float] = []\n        numeric_var: list[float] = []\n        numeric_scale: list[float] = []\n        numeric_n_samples_seen: list[int] = []\n\n        for column, scaler in zip(\n            NUMERIC_FEATURES,\n            self._scalers,\n        ):\n            if not hasattr(\n                scaler,\n                "n_samples_seen_",\n            ):\n                raise PreprocessingContractError(\n                    f"No observed TRAIN value for numeric feature: {column}."\n                )\n\n            count = int(\n                np.asarray(\n                    scaler.n_samples_seen_\n                ).reshape(-1)[0]\n            )\n            mean = float(\n                np.asarray(\n                    scaler.mean_\n                ).reshape(-1)[0]\n            )\n            variance = float(\n                np.asarray(\n                    scaler.var_\n                ).reshape(-1)[0]\n            )\n            scale = float(\n                np.asarray(\n                    scaler.scale_\n                ).reshape(-1)[0]\n            )\n\n            if (\n                count <= 0\n                or not np.isfinite(mean)\n                or not np.isfinite(variance)\n                or not np.isfinite(scale)\n                or scale <= 0.0\n            ):\n                raise PreprocessingContractError(\n                    f"Invalid learned numeric state: {column}."\n                )\n\n            numeric_n_samples_seen.append(\n                count\n            )\n            numeric_mean.append(mean)\n            numeric_var.append(\n                variance\n            )\n            numeric_scale.append(\n                scale\n            )\n\n        category_vocab: dict[\n            str,\n            tuple[str, ...],\n        ] = {}\n\n        for column in CATEGORICAL_FEATURES:\n            observed = _stable_sort_categories(\n                column,\n                self._category_values[\n                    column\n                ],\n            )\n\n            expected = (\n                EXPECTED_CATEGORY_VOCAB[\n                    column\n                ]\n            )\n\n            if observed != expected:\n                raise PreprocessingContractError(\n                    f"TRAIN categorical vocabulary mismatch: {column}."\n                )\n\n            category_vocab[\n                column\n            ] = observed\n\n        feature_names = (\n            _build_feature_names(\n                category_vocab\n            )\n        )\n\n        if len(feature_names) != 47:\n            raise PreprocessingContractError(\n                "Rebuild preprocessing schema must contain exactly 47 features."\n            )\n\n        state = {\n            "analysis_version":\n                "rebuild-preprocessing-v1",\n            "categorical_columns":\n                list(\n                    CATEGORICAL_FEATURES\n                ),\n            "category_vocab": {\n                column: list(\n                    category_vocab[\n                        column\n                    ]\n                )\n                for column\n                in CATEGORICAL_FEATURES\n            },\n            "feature_count":\n                47,\n            "feature_names":\n                list(feature_names),\n            "fit_max_timestamp":\n                _format_timestamp(\n                    self._fit_max_timestamp\n                ),\n            "fit_min_timestamp":\n                _format_timestamp(\n                    self._fit_min_timestamp\n                ),\n            "fit_row_count":\n                self._fit_row_count,\n            "fit_source":\n                REBUILD_FIT_SOURCE,\n            "matrix_dtype":\n                "float32",\n            "matrix_format":\n                "CSR",\n            "numeric_columns":\n                list(\n                    NUMERIC_FEATURES\n                ),\n            "numeric_mean":\n                numeric_mean,\n            "numeric_n_samples_seen":\n                numeric_n_samples_seen,\n            "numeric_scale":\n                numeric_scale,\n            "numeric_var":\n                numeric_var,\n            "strategy":\n                REBUILD_STRATEGY,\n            "unknown_token":\n                UNKNOWN_TOKEN,\n            "rebuild_validation_status":\n                "NOT_COMPARED_TO_OFFICIAL_ARTIFACT",\n        }\n\n        preprocessor = (\n            RebuildPreprocessor(\n                numeric_mean=np.asarray(\n                    numeric_mean,\n                    dtype=np.float64,\n                ),\n                numeric_scale=np.asarray(\n                    numeric_scale,\n                    dtype=np.float64,\n                ),\n                category_vocab=\n                    category_vocab,\n                feature_names=\n                    feature_names,\n            )\n        )\n\n        return RebuildPreprocessingFit(\n            state=state,\n            preprocessor=preprocessor,\n        )\n\n\ndef fit_rebuild_preprocessor(\n    rows: Sequence[SemanticFeatureRow],\n    timestamps: Sequence[datetime],\n) -> RebuildPreprocessingFit:\n    """Fit one TRAIN-only preprocessing state in a single batch."""\n\n    fitter = (\n        RebuildPreprocessingFitter()\n    )\n\n    fitter.partial_fit(\n        rows,\n        timestamps,\n    )\n\n    return fitter.finalize()\n',
    "final_pipeline/tests/unit/test_rebuild_preprocessing_fit.py":
        'from __future__ import annotations\n\nimport unittest\nfrom datetime import datetime, timedelta\n\nimport numpy as np\nfrom scipy import sparse\nfrom sklearn.preprocessing import StandardScaler\n\nfrom fraud_screening.errors import PreprocessingContractError\nfrom fraud_screening.features import SemanticFeatureRow\nfrom fraud_screening.training import (\n    RebuildPreprocessingFitter,\n    fit_rebuild_preprocessor,\n)\n\n\ndef training_fixture(\n    count: int = 24,\n) -> tuple[\n    list[SemanticFeatureRow],\n    list[datetime],\n]:\n    modes = [\n        "Chip Transaction",\n        "Online Transaction",\n        "Swipe Transaction",\n    ]\n\n    locations = [\n        "NON_PHYSICAL_OR_ONLINE",\n        "PHYSICAL_COMPLETE",\n        "PHYSICAL_ZIP_UNAVAILABLE",\n    ]\n\n    rows = []\n    timestamps = []\n\n    start = datetime(\n        2018,\n        1,\n        1,\n        0,\n        0,\n    )\n\n    for index in range(count):\n        rows.append(\n            SemanticFeatureRow(\n                amount_numeric=\n                    float(index + 1),\n                time_since_previous_transaction_min=\n                    None\n                    if index == 0\n                    else float(index * 10),\n                transactions_last_1h=\n                    index % 4,\n                amount_minus_previous_mean=\n                    None\n                    if index == 0\n                    else float(index - 5),\n                is_new_merchant=\n                    index < 3,\n                has_prior_card_history=\n                    index != 0,\n                transaction_mode=\n                    modes[\n                        index % 3\n                    ],\n                location_state=\n                    locations[\n                        index % 3\n                    ],\n                hour_of_day=\n                    str(index % 24),\n                day_of_week=\n                    str(index % 7),\n            )\n        )\n\n        timestamps.append(\n            start\n            + timedelta(\n                hours=index\n            )\n        )\n\n    return rows, timestamps\n\n\nclass RebuildPreprocessingFitTests(unittest.TestCase):\n    def test_fit_produces_exact_47_column_contract(self) -> None:\n        rows, timestamps = training_fixture()\n\n        result = fit_rebuild_preprocessor(\n            rows,\n            timestamps,\n        )\n\n        self.assertEqual(\n            result.state[\n                "feature_count"\n            ],\n            47,\n        )\n        self.assertEqual(\n            len(\n                result.state[\n                    "feature_names"\n                ]\n            ),\n            47,\n        )\n        self.assertEqual(\n            result.state[\n                "matrix_format"\n            ],\n            "CSR",\n        )\n        self.assertEqual(\n            result.state[\n                "matrix_dtype"\n            ],\n            "float32",\n        )\n\n    def test_rebuild_state_does_not_claim_official_validation(self) -> None:\n        rows, timestamps = training_fixture()\n\n        result = fit_rebuild_preprocessor(\n            rows,\n            timestamps,\n        )\n\n        self.assertNotIn(\n            "validation_exact_reproduction",\n            result.state,\n        )\n        self.assertEqual(\n            result.state[\n                "rebuild_validation_status"\n            ],\n            "NOT_COMPARED_TO_OFFICIAL_ARTIFACT",\n        )\n\n    def test_transform_after_fit_returns_csr_float32(self) -> None:\n        rows, timestamps = training_fixture()\n\n        result = fit_rebuild_preprocessor(\n            rows,\n            timestamps,\n        )\n\n        matrix = (\n            result.preprocessor\n            .transform(\n                rows\n            )\n        )\n\n        self.assertTrue(\n            sparse.isspmatrix_csr(\n                matrix\n            )\n        )\n        self.assertEqual(\n            matrix.shape,\n            (24, 47),\n        )\n        self.assertEqual(\n            matrix.dtype,\n            np.float32,\n        )\n\n    def test_structural_na_is_excluded_from_numeric_fit(self) -> None:\n        rows, timestamps = training_fixture()\n\n        result = fit_rebuild_preprocessor(\n            rows,\n            timestamps,\n        )\n\n        self.assertEqual(\n            result.state[\n                "numeric_n_samples_seen"\n            ],\n            [24, 23, 24, 23],\n        )\n\n    def test_numeric_state_matches_independent_standard_scaler(self) -> None:\n        rows, timestamps = training_fixture()\n\n        result = fit_rebuild_preprocessor(\n            rows,\n            timestamps,\n        )\n\n        amount_values = np.asarray(\n            [\n                row.amount_numeric\n                for row in rows\n            ],\n            dtype=np.float64,\n        ).reshape(-1, 1)\n\n        reference = (\n            StandardScaler()\n            .fit(\n                amount_values\n            )\n        )\n\n        self.assertAlmostEqual(\n            result.state[\n                "numeric_mean"\n            ][0],\n            float(\n                reference.mean_[0]\n            ),\n            places=12,\n        )\n        self.assertAlmostEqual(\n            result.state[\n                "numeric_var"\n            ][0],\n            float(\n                reference.var_[0]\n            ),\n            places=12,\n        )\n        self.assertAlmostEqual(\n            result.state[\n                "numeric_scale"\n            ][0],\n            float(\n                reference.scale_[0]\n            ),\n            places=12,\n        )\n\n    def test_two_chronological_batches_match_single_batch_state(self) -> None:\n        rows, timestamps = training_fixture()\n\n        single = fit_rebuild_preprocessor(\n            rows,\n            timestamps,\n        )\n\n        fitter = (\n            RebuildPreprocessingFitter()\n        )\n\n        fitter.partial_fit(\n            rows[:12],\n            timestamps[:12],\n        )\n        fitter.partial_fit(\n            rows[12:],\n            timestamps[12:],\n        )\n\n        batched = fitter.finalize()\n\n        np.testing.assert_allclose(\n            batched.state[\n                "numeric_mean"\n            ],\n            single.state[\n                "numeric_mean"\n            ],\n            rtol=0.0,\n            atol=1e-12,\n        )\n\n        np.testing.assert_allclose(\n            batched.state[\n                "numeric_var"\n            ],\n            single.state[\n                "numeric_var"\n            ],\n            rtol=0.0,\n            atol=1e-12,\n        )\n\n        np.testing.assert_allclose(\n            batched.state[\n                "numeric_scale"\n            ],\n            single.state[\n                "numeric_scale"\n            ],\n            rtol=0.0,\n            atol=1e-12,\n        )\n\n        self.assertEqual(\n            batched.state[\n                "category_vocab"\n            ],\n            single.state[\n                "category_vocab"\n            ],\n        )\n\n    def test_non_chronological_batch_is_rejected(self) -> None:\n        rows, timestamps = training_fixture()\n\n        fitter = (\n            RebuildPreprocessingFitter()\n        )\n\n        fitter.partial_fit(\n            rows[12:],\n            timestamps[12:],\n        )\n\n        with self.assertRaises(\n            PreprocessingContractError\n        ):\n            fitter.partial_fit(\n                rows[:12],\n                timestamps[:12],\n            )\n\n    def test_missing_expected_category_is_rejected(self) -> None:\n        rows, timestamps = training_fixture(\n            count=23\n        )\n\n        with self.assertRaises(\n            PreprocessingContractError\n        ):\n            fit_rebuild_preprocessor(\n                rows,\n                timestamps,\n            )\n\n    def test_reserved_unknown_token_is_rejected_in_train(self) -> None:\n        rows, timestamps = training_fixture()\n\n        row = rows[0]\n\n        rows[0] = SemanticFeatureRow(\n            amount_numeric=\n                row.amount_numeric,\n            time_since_previous_transaction_min=\n                row.time_since_previous_transaction_min,\n            transactions_last_1h=\n                row.transactions_last_1h,\n            amount_minus_previous_mean=\n                row.amount_minus_previous_mean,\n            is_new_merchant=\n                row.is_new_merchant,\n            has_prior_card_history=\n                row.has_prior_card_history,\n            transaction_mode=\n                "__UNKNOWN__",\n            location_state=\n                row.location_state,\n            hour_of_day=\n                row.hour_of_day,\n            day_of_week=\n                row.day_of_week,\n        )\n\n        with self.assertRaises(\n            PreprocessingContractError\n        ):\n            fit_rebuild_preprocessor(\n                rows,\n                timestamps,\n            )\n\n    def test_all_missing_structural_numeric_feature_is_rejected(self) -> None:\n        rows, timestamps = training_fixture()\n\n        rows = [\n            SemanticFeatureRow(\n                amount_numeric=\n                    row.amount_numeric,\n                time_since_previous_transaction_min=\n                    None,\n                transactions_last_1h=\n                    row.transactions_last_1h,\n                amount_minus_previous_mean=\n                    row.amount_minus_previous_mean,\n                is_new_merchant=\n                    row.is_new_merchant,\n                has_prior_card_history=\n                    row.has_prior_card_history,\n                transaction_mode=\n                    row.transaction_mode,\n                location_state=\n                    row.location_state,\n                hour_of_day=\n                    row.hour_of_day,\n                day_of_week=\n                    row.day_of_week,\n            )\n            for row in rows\n        ]\n\n        with self.assertRaises(\n            PreprocessingContractError\n        ):\n            fit_rebuild_preprocessor(\n                rows,\n                timestamps,\n            )\n\n    def test_fit_timestamp_metadata_is_recorded(self) -> None:\n        rows, timestamps = training_fixture()\n\n        result = fit_rebuild_preprocessor(\n            rows,\n            timestamps,\n        )\n\n        self.assertEqual(\n            result.state[\n                "fit_row_count"\n            ],\n            24,\n        )\n        self.assertEqual(\n            result.state[\n                "fit_min_timestamp"\n            ],\n            "2018-01-01 00:00:00",\n        )\n        self.assertEqual(\n            result.state[\n                "fit_max_timestamp"\n            ],\n            "2018-01-01 23:00:00",\n        )\n\n\nif __name__ == "__main__":\n    unittest.main()\n',
}


def detect_root() -> Path:
    root = Path.cwd().resolve()

    required = [
        root / "research",
        root / "final_pipeline",
        root / "application",
        root
        / "final_pipeline/src/fraud_screening/training/factory.py",
        root
        / "final_pipeline/src/fraud_screening/training/outputs.py",
        root
        / "final_pipeline/src/fraud_screening/preprocessing/frozen.py",
        root
        / "final_pipeline/artifacts/official/model/model.joblib",
        root
        / "final_pipeline/artifacts/official/preprocessing/preprocessing_state.json",
        root
        / "final_pipeline/artifacts/official/manifest/artifact_manifest.json",
    ]

    if not all(path.exists() for path in required):
        raise RuntimeError(
            "Repository foundation or official artifacts are incomplete."
        )

    return root


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Create rebuild preprocessing fitter and tests. "
            "Source creation itself performs no fitting."
        ),
    )

    args = parser.parse_args()

    root = detect_root()
    init_path = root / INIT_REL

    missing_dependencies = [
        relative
        for relative in [
            "final_pipeline/src/fraud_screening/features/semantic.py",
            "final_pipeline/src/fraud_screening/errors.py",
        ]
        if not (root / relative).is_file()
    ]

    collisions = [
        relative
        for relative in NEW_FILES
        if (root / relative).exists()
    ]

    init_matches = (
        init_path.is_file()
        and init_path.read_text(
            encoding="utf-8"
        )
        == EXPECTED_INIT
    )

    print("=" * 92)
    print(
        "REBUILD PREPROCESSING — TRAIN-ONLY LEARNED STATE FITTER"
    )
    print("=" * 92)
    print("Project root:", root)
    print(
        "Mode:",
        "EXECUTE — GUARDED SOURCE UPDATE"
        if args.execute
        else "DRY RUN — NO FILES WRITTEN",
    )

    print("\n[1] Dependency gate")
    print(
        " - missing dependencies:",
        len(missing_dependencies),
    )
    for relative in missing_dependencies:
        print("   *", relative)

    print("\n[2] training/__init__.py guard")
    print(
        " - exact expected current content:",
        "YES" if init_matches else "NO",
    )

    print("\n[3] Planned new files")
    for relative in NEW_FILES:
        print(" -", relative)

    print("\n[4] Collision gate")
    print(
        " - collisions:",
        len(collisions),
    )
    for relative in collisions:
        print("   *", relative)

    print("\n[5] Learned preprocessing contract")
    print(" - strategy: W_SHORT")
    print(" - fit source: TRAIN ONLY")
    print(
        " - numeric learner: one StandardScaler per numeric feature"
    )
    print(
        " - numeric update: partial_fit on finite observations only"
    )
    print(
        " - categorical vocab: observed TRAIN vocab, exact 47-column domain required"
    )
    print(" - reserved unknown token: __UNKNOWN__")
    print(" - output schema: exact 47 features")
    print(" - transform format: CSR float32")
    print(" - chronological batch order: required")
    print(
        " - rebuild state does NOT claim official validation equivalence"
    )

    print("\n[6] Side-effect boundary")
    print(
        " - dry-run preprocessing fit: NO"
    )
    print(
        " - execute preprocessing fit: NO"
    )
    print(
        " - unit tests after execute: YES, synthetic in-memory only"
    )
    print(" - model.fit(): NO")
    print(" - predict()/predict_proba(): NO")
    print(" - rebuild artifact write: NO")
    print(" - official artifact modification: NO")
    print(" - research modification: NO")

    if (
        missing_dependencies
        or collisions
        or not init_matches
    ):
        print("\nRESULT: STOP")
        print(
            "No files were written."
        )
        raise SystemExit(1)

    if not args.execute:
        print("\nDRY-RUN RESULT: PASS")
        print("Execute with:")
        print(
            "python bootstrap_rebuild_preprocessing.py --execute"
        )
        print("=" * 92)
        return

    created = []
    original_init = init_path.read_text(
        encoding="utf-8"
    )

    try:
        for relative, content in NEW_FILES.items():
            path = root / relative

            path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            path.write_text(
                content,
                encoding="utf-8",
            )

            created.append(
                path
            )

        init_path.write_text(
            UPDATED_INIT,
            encoding="utf-8",
        )

    except Exception:
        for path in reversed(created):
            try:
                path.unlink()
            except OSError:
                pass

        try:
            init_path.write_text(
                original_init,
                encoding="utf-8",
            )
        except OSError:
            pass

        raise

    print("\n[7] Created")
    for path in created:
        print(
            " -",
            path.relative_to(root),
        )

    print("\n[8] Updated")
    print(
        " -",
        INIT_REL,
    )

    print("\nBOOTSTRAP RESULT: PASS")
    print(
        "Source was created without fitting."
    )
    print(
        "Run full unit suite next; "
        "those tests intentionally fit preprocessing "
        "on small synthetic TRAIN fixtures."
    )
    print(
        "PYTHONPATH=final_pipeline/src "
        "python -m unittest discover "
        "-s final_pipeline/tests -v"
    )
    print("=" * 92)


if __name__ == "__main__":
    main()
