from __future__ import annotations

import argparse
import hashlib
import importlib.util
import sys
from pathlib import Path

NEW_FILES = {
    "final_pipeline/src/fraud_screening/preprocessing/__init__.py":
        '"""Frozen preprocessing for the screening feature interface."""\n\nfrom fraud_screening.preprocessing.frozen import (\n    FrozenPreprocessor,\n    load_frozen_preprocessor,\n)\n\n__all__ = [\n    "FrozenPreprocessor",\n    "load_frozen_preprocessor",\n]\n',
    "final_pipeline/src/fraud_screening/preprocessing/frozen.py":
        '"""Read-only transformation using an already-frozen preprocessing state."""\n\nfrom __future__ import annotations\n\nimport json\nimport math\nfrom dataclasses import dataclass\nfrom pathlib import Path\nfrom typing import Any, Iterable\n\nimport numpy as np\nfrom scipy import sparse\n\nfrom fraud_screening.errors import PreprocessingContractError\nfrom fraud_screening.features import (\n    BOOLEAN_FEATURES,\n    CATEGORICAL_FEATURES,\n    NUMERIC_FEATURES,\n    SEMANTIC_FEATURE_ORDER,\n    SemanticFeatureRow,\n)\n\n\nEXPECTED_FEATURE_COUNT = 47\nEXPECTED_MATRIX_FORMAT = "CSR"\nEXPECTED_MATRIX_DTYPE = "float32"\nEXPECTED_STRATEGY = "W_SHORT"\nEXPECTED_FIT_SOURCE = "W_SHORT_TRAIN_ONLY"\n\n_STRUCTURAL_NA_FEATURES = {\n    "time_since_previous_transaction_min",\n    "amount_minus_previous_mean",\n}\n\n\ndef _require_exact_list(\n    state: dict[str, Any],\n    key: str,\n    expected: tuple[str, ...],\n) -> None:\n    value = state.get(key)\n\n    if value != list(expected):\n        raise PreprocessingContractError(\n            f"{key} does not match the frozen semantic contract."\n        )\n\n\ndef _as_finite_vector(\n    state: dict[str, Any],\n    key: str,\n    width: int,\n) -> np.ndarray:\n    value = state.get(key)\n\n    if not isinstance(value, list) or len(value) != width:\n        raise PreprocessingContractError(\n            f"{key} must contain exactly {width} values."\n        )\n\n    array = np.asarray(value, dtype=np.float64)\n\n    if not np.isfinite(array).all():\n        raise PreprocessingContractError(\n            f"{key} must contain only finite values."\n        )\n\n    return array\n\n\n@dataclass(frozen=True, slots=True)\nclass FrozenPreprocessor:\n    """Immutable transform state for the exact 10-to-47 representation."""\n\n    numeric_mean: np.ndarray\n    numeric_scale: np.ndarray\n    category_vocab: dict[str, tuple[str, ...]]\n    unknown_token: str\n    feature_names: tuple[str, ...]\n    strategy: str\n    fit_source: str\n\n    @classmethod\n    def from_state_dict(\n        cls,\n        state: dict[str, Any],\n    ) -> "FrozenPreprocessor":\n        if not isinstance(state, dict):\n            raise PreprocessingContractError(\n                "Preprocessing state must be a JSON object."\n            )\n\n        _require_exact_list(\n            state,\n            "numeric_columns",\n            NUMERIC_FEATURES,\n        )\n        _require_exact_list(\n            state,\n            "categorical_columns",\n            CATEGORICAL_FEATURES,\n        )\n\n        if state.get("feature_count") != EXPECTED_FEATURE_COUNT:\n            raise PreprocessingContractError(\n                "Frozen preprocessing feature_count must be 47."\n            )\n\n        if state.get("matrix_format") != EXPECTED_MATRIX_FORMAT:\n            raise PreprocessingContractError(\n                "Frozen preprocessing matrix format must be CSR."\n            )\n\n        if state.get("matrix_dtype") != EXPECTED_MATRIX_DTYPE:\n            raise PreprocessingContractError(\n                "Frozen preprocessing matrix dtype must be float32."\n            )\n\n        if state.get("strategy") != EXPECTED_STRATEGY:\n            raise PreprocessingContractError(\n                "Unexpected preprocessing strategy."\n            )\n\n        if state.get("fit_source") != EXPECTED_FIT_SOURCE:\n            raise PreprocessingContractError(\n                "Unexpected preprocessing fit source."\n            )\n\n        if state.get("validation_exact_reproduction") is not True:\n            raise PreprocessingContractError(\n                "Frozen state is not marked as exact validation reproduction."\n            )\n\n        numeric_mean = _as_finite_vector(\n            state,\n            "numeric_mean",\n            len(NUMERIC_FEATURES),\n        )\n        numeric_scale = _as_finite_vector(\n            state,\n            "numeric_scale",\n            len(NUMERIC_FEATURES),\n        )\n\n        if (numeric_scale <= 0.0).any():\n            raise PreprocessingContractError(\n                "All numeric scales must be strictly positive."\n            )\n\n        unknown_token = state.get("unknown_token")\n\n        if (\n            not isinstance(unknown_token, str)\n            or not unknown_token\n        ):\n            raise PreprocessingContractError(\n                "unknown_token must be a non-empty string."\n            )\n\n        raw_vocab = state.get("category_vocab")\n\n        if not isinstance(raw_vocab, dict):\n            raise PreprocessingContractError(\n                "category_vocab must be a JSON object."\n            )\n\n        if set(raw_vocab) != set(CATEGORICAL_FEATURES):\n            raise PreprocessingContractError(\n                "category_vocab keys do not match categorical features."\n            )\n\n        category_vocab: dict[str, tuple[str, ...]] = {}\n\n        for column in CATEGORICAL_FEATURES:\n            values = raw_vocab.get(column)\n\n            if (\n                not isinstance(values, list)\n                or not values\n                or not all(\n                    isinstance(value, str)\n                    for value in values\n                )\n            ):\n                raise PreprocessingContractError(\n                    f"Invalid category vocabulary for {column}."\n                )\n\n            if len(values) != len(set(values)):\n                raise PreprocessingContractError(\n                    f"Duplicate category in vocabulary for {column}."\n                )\n\n            if unknown_token in values:\n                raise PreprocessingContractError(\n                    f"Reserved unknown token appears in observed vocabulary: {column}."\n                )\n\n            category_vocab[column] = tuple(values)\n\n        feature_names_value = state.get("feature_names")\n\n        if (\n            not isinstance(feature_names_value, list)\n            or not all(\n                isinstance(name, str)\n                for name in feature_names_value\n            )\n        ):\n            raise PreprocessingContractError(\n                "feature_names must be a list of strings."\n            )\n\n        feature_names = tuple(feature_names_value)\n\n        expected_names = cls._build_feature_names(\n            category_vocab=category_vocab,\n            unknown_token=unknown_token,\n        )\n\n        if feature_names != expected_names:\n            raise PreprocessingContractError(\n                "Frozen feature_names do not match vocabulary-derived schema."\n            )\n\n        if len(feature_names) != EXPECTED_FEATURE_COUNT:\n            raise PreprocessingContractError(\n                "Frozen output schema must contain exactly 47 features."\n            )\n\n        return cls(\n            numeric_mean=numeric_mean,\n            numeric_scale=numeric_scale,\n            category_vocab=category_vocab,\n            unknown_token=unknown_token,\n            feature_names=feature_names,\n            strategy=state["strategy"],\n            fit_source=state["fit_source"],\n        )\n\n    @staticmethod\n    def _build_feature_names(\n        *,\n        category_vocab: dict[str, tuple[str, ...]],\n        unknown_token: str,\n    ) -> tuple[str, ...]:\n        names: list[str] = [\n            f"num__{name}"\n            for name in NUMERIC_FEATURES\n        ]\n\n        names.extend(\n            f"bool__{name}"\n            for name in BOOLEAN_FEATURES\n        )\n\n        for column in CATEGORICAL_FEATURES:\n            for category in (\n                *category_vocab[column],\n                unknown_token,\n            ):\n                names.append(\n                    f"cat__{column}_{category}"\n                )\n\n        return tuple(names)\n\n    @classmethod\n    def from_json(\n        cls,\n        path: str | Path,\n    ) -> "FrozenPreprocessor":\n        state_path = Path(path)\n\n        try:\n            state = json.loads(\n                state_path.read_text(\n                    encoding="utf-8"\n                )\n            )\n        except FileNotFoundError:\n            raise\n        except Exception as exc:\n            raise PreprocessingContractError(\n                "Unable to read frozen preprocessing JSON."\n            ) from exc\n\n        return cls.from_state_dict(state)\n\n    def _numeric_matrix(\n        self,\n        rows: list[SemanticFeatureRow],\n    ) -> np.ndarray:\n        matrix = np.empty(\n            (len(rows), len(NUMERIC_FEATURES)),\n            dtype=np.float64,\n        )\n\n        for row_index, row in enumerate(rows):\n            mapping = row.as_dict()\n\n            for column_index, column in enumerate(\n                NUMERIC_FEATURES\n            ):\n                value = mapping[column]\n\n                if value is None:\n                    if column not in _STRUCTURAL_NA_FEATURES:\n                        raise PreprocessingContractError(\n                            f"Unexpected missing numeric feature: {column}."\n                        )\n\n                    matrix[\n                        row_index,\n                        column_index,\n                    ] = np.nan\n                    continue\n\n                if isinstance(value, bool):\n                    raise PreprocessingContractError(\n                        f"Boolean value supplied for numeric feature: {column}."\n                    )\n\n                try:\n                    numeric_value = float(value)\n                except (TypeError, ValueError) as exc:\n                    raise PreprocessingContractError(\n                        f"Non-numeric value for feature: {column}."\n                    ) from exc\n\n                if not math.isfinite(numeric_value):\n                    raise PreprocessingContractError(\n                        f"Non-finite numeric feature: {column}."\n                    )\n\n                matrix[\n                    row_index,\n                    column_index,\n                ] = numeric_value\n\n        finite_mask = np.isfinite(matrix)\n\n        transformed = np.full(\n            matrix.shape,\n            np.nan,\n            dtype=np.float64,\n        )\n\n        if finite_mask.any():\n            for index in range(\n                len(NUMERIC_FEATURES)\n            ):\n                column_mask = finite_mask[:, index]\n\n                if not column_mask.any():\n                    continue\n\n                transformed[\n                    column_mask,\n                    index,\n                ] = (\n                    matrix[column_mask, index]\n                    - self.numeric_mean[index]\n                ) / self.numeric_scale[index]\n\n        return np.nan_to_num(\n            transformed,\n            nan=0.0,\n            posinf=np.inf,\n            neginf=-np.inf,\n        ).astype(np.float32)\n\n    @staticmethod\n    def _boolean_matrix(\n        rows: list[SemanticFeatureRow],\n    ) -> np.ndarray:\n        matrix = np.empty(\n            (len(rows), len(BOOLEAN_FEATURES)),\n            dtype=np.float32,\n        )\n\n        for row_index, row in enumerate(rows):\n            mapping = row.as_dict()\n\n            for column_index, column in enumerate(\n                BOOLEAN_FEATURES\n            ):\n                value = mapping[column]\n\n                if not isinstance(value, bool):\n                    raise PreprocessingContractError(\n                        f"Boolean feature is not bool: {column}."\n                    )\n\n                matrix[\n                    row_index,\n                    column_index,\n                ] = 1.0 if value else 0.0\n\n        return matrix\n\n    def _categorical_matrix(\n        self,\n        rows: list[SemanticFeatureRow],\n    ) -> sparse.csr_matrix:\n        total_width = sum(\n            len(self.category_vocab[column]) + 1\n            for column in CATEGORICAL_FEATURES\n        )\n\n        row_indices: list[int] = []\n        column_indices: list[int] = []\n        data: list[float] = []\n\n        branch_offset = 0\n\n        for column in CATEGORICAL_FEATURES:\n            known_values = self.category_vocab[column]\n            lookup = {\n                value: index\n                for index, value in enumerate(\n                    known_values\n                )\n            }\n            unknown_index = len(known_values)\n\n            for row_index, row in enumerate(rows):\n                value = row.as_dict()[column]\n\n                if value is None:\n                    raise PreprocessingContractError(\n                        f"Unexpected missing categorical feature: {column}."\n                    )\n\n                value = str(value)\n\n                local_index = lookup.get(\n                    value,\n                    unknown_index,\n                )\n\n                row_indices.append(row_index)\n                column_indices.append(\n                    branch_offset + local_index\n                )\n                data.append(1.0)\n\n            branch_offset += (\n                len(known_values) + 1\n            )\n\n        return sparse.csr_matrix(\n            (\n                np.asarray(\n                    data,\n                    dtype=np.float32,\n                ),\n                (\n                    np.asarray(\n                        row_indices,\n                        dtype=np.int64,\n                    ),\n                    np.asarray(\n                        column_indices,\n                        dtype=np.int64,\n                    ),\n                ),\n            ),\n            shape=(len(rows), total_width),\n            dtype=np.float32,\n        )\n\n    def transform(\n        self,\n        rows: Iterable[SemanticFeatureRow],\n    ) -> sparse.csr_matrix:\n        """Transform exact semantic rows using frozen learned state only."""\n\n        semantic_rows = list(rows)\n\n        if not semantic_rows:\n            return sparse.csr_matrix(\n                (0, EXPECTED_FEATURE_COUNT),\n                dtype=np.float32,\n            )\n\n        for row in semantic_rows:\n            if not isinstance(\n                row,\n                SemanticFeatureRow,\n            ):\n                raise PreprocessingContractError(\n                    "transform expects SemanticFeatureRow instances."\n                )\n\n            if tuple(row.as_dict()) != SEMANTIC_FEATURE_ORDER:\n                raise PreprocessingContractError(\n                    "Semantic feature order mismatch."\n                )\n\n        numeric_matrix = self._numeric_matrix(\n            semantic_rows\n        )\n\n        boolean_matrix = self._boolean_matrix(\n            semantic_rows\n        )\n\n        categorical_matrix = self._categorical_matrix(\n            semantic_rows\n        )\n\n        matrix = sparse.hstack(\n            [\n                sparse.csr_matrix(\n                    numeric_matrix,\n                    dtype=np.float32,\n                ),\n                sparse.csr_matrix(\n                    boolean_matrix,\n                    dtype=np.float32,\n                ),\n                categorical_matrix,\n            ],\n            format="csr",\n            dtype=np.float32,\n        )\n\n        if matrix.shape[1] != EXPECTED_FEATURE_COUNT:\n            raise PreprocessingContractError(\n                "Preprocessing output width is not 47."\n            )\n\n        if matrix.shape[1] != len(\n            self.feature_names\n        ):\n            raise PreprocessingContractError(\n                "Output width does not match frozen feature names."\n            )\n\n        if not np.isfinite(matrix.data).all():\n            raise PreprocessingContractError(\n                "Non-finite value produced by frozen preprocessing."\n            )\n\n        return matrix\n\n\ndef load_frozen_preprocessor(\n    path: str | Path,\n) -> FrozenPreprocessor:\n    """Load a frozen preprocessing JSON without fitting any learned state."""\n\n    return FrozenPreprocessor.from_json(path)\n',
    "final_pipeline/tests/unit/test_frozen_preprocessing.py":
        'from __future__ import annotations\n\nimport json\nimport tempfile\nimport unittest\nfrom pathlib import Path\n\nimport numpy as np\nfrom scipy import sparse\n\nfrom fraud_screening.errors import PreprocessingContractError\nfrom fraud_screening.features import SemanticFeatureRow\nfrom fraud_screening.preprocessing import (\n    FrozenPreprocessor,\n    load_frozen_preprocessor,\n)\n\n\ndef frozen_state() -> dict:\n    category_vocab = {\n        "transaction_mode": [\n            "Chip Transaction",\n            "Online Transaction",\n            "Swipe Transaction",\n        ],\n        "location_state": [\n            "NON_PHYSICAL_OR_ONLINE",\n            "PHYSICAL_COMPLETE",\n            "PHYSICAL_ZIP_UNAVAILABLE",\n        ],\n        "hour_of_day": [\n            str(value)\n            for value in range(24)\n        ],\n        "day_of_week": [\n            str(value)\n            for value in range(7)\n        ],\n    }\n\n    unknown = "__UNKNOWN__"\n\n    feature_names = [\n        "num__amount_numeric",\n        "num__time_since_previous_transaction_min",\n        "num__transactions_last_1h",\n        "num__amount_minus_previous_mean",\n        "bool__is_new_merchant",\n        "bool__has_prior_card_history",\n    ]\n\n    for column in (\n        "transaction_mode",\n        "location_state",\n        "hour_of_day",\n        "day_of_week",\n    ):\n        feature_names.extend(\n            f"cat__{column}_{category}"\n            for category in (\n                *category_vocab[column],\n                unknown,\n            )\n        )\n\n    return {\n        "analysis_version":\n            "frozen-evaluation-state",\n        "categorical_columns": [\n            "transaction_mode",\n            "location_state",\n            "hour_of_day",\n            "day_of_week",\n        ],\n        "category_vocab":\n            category_vocab,\n        "feature_count":\n            47,\n        "feature_names":\n            feature_names,\n        "fit_max_timestamp":\n            "2018-12-31 23:58:00",\n        "fit_min_timestamp":\n            "2018-01-01 00:03:00",\n        "fit_row_count":\n            1721615,\n        "fit_source":\n            "W_SHORT_TRAIN_ONLY",\n        "matrix_dtype":\n            "float32",\n        "matrix_format":\n            "CSR",\n        "numeric_columns": [\n            "amount_numeric",\n            "time_since_previous_transaction_min",\n            "transactions_last_1h",\n            "amount_minus_previous_mean",\n        ],\n        "numeric_mean": [\n            42.879027959212905,\n            1197.3147570599144,\n            0.2759606532238619,\n            -0.4259839479714494,\n        ],\n        "numeric_n_samples_seen": [\n            1721615,\n            1721515,\n            1721615,\n            1721515,\n        ],\n        "numeric_scale": [\n            80.55065611888665,\n            2113.3233635919028,\n            0.6581493130130939,\n            78.17323847945735,\n        ],\n        "numeric_var": [\n            6488.408201183131,\n            4466135.639103394,\n            0.43316051821960744,\n            6111.0552143661125,\n        ],\n        "strategy":\n            "W_SHORT",\n        "unknown_token":\n            unknown,\n        "validation_exact_reproduction":\n            True,\n    }\n\n\ndef semantic_row(\n    *,\n    amount: float = 42.879027959212905,\n    recency: float | None = None,\n    velocity: int = 0,\n    amount_delta: float | None = None,\n    new_merchant: bool = True,\n    has_history: bool = False,\n    mode: str = "Chip Transaction",\n    location: str = "PHYSICAL_COMPLETE",\n    hour: str = "14",\n    day: str = "0",\n) -> SemanticFeatureRow:\n    return SemanticFeatureRow(\n        amount_numeric=amount,\n        time_since_previous_transaction_min=\n            recency,\n        transactions_last_1h=velocity,\n        amount_minus_previous_mean=\n            amount_delta,\n        is_new_merchant=new_merchant,\n        has_prior_card_history=has_history,\n        transaction_mode=mode,\n        location_state=location,\n        hour_of_day=hour,\n        day_of_week=day,\n    )\n\n\nclass FrozenPreprocessingTests(unittest.TestCase):\n    def test_state_loads_with_exact_47_schema(self) -> None:\n        preprocessor = FrozenPreprocessor.from_state_dict(\n            frozen_state()\n        )\n\n        self.assertEqual(\n            len(preprocessor.feature_names),\n            47,\n        )\n\n    def test_transform_returns_csr_float32(self) -> None:\n        preprocessor = FrozenPreprocessor.from_state_dict(\n            frozen_state()\n        )\n\n        matrix = preprocessor.transform(\n            [semantic_row()]\n        )\n\n        self.assertTrue(\n            sparse.isspmatrix_csr(matrix)\n        )\n        self.assertEqual(\n            matrix.shape,\n            (1, 47),\n        )\n        self.assertEqual(\n            matrix.dtype,\n            np.float32,\n        )\n\n    def test_numeric_scaling_uses_frozen_mean_and_scale(self) -> None:\n        state = frozen_state()\n        preprocessor = FrozenPreprocessor.from_state_dict(\n            state\n        )\n\n        row = semantic_row(\n            amount=state["numeric_mean"][0]\n            + state["numeric_scale"][0],\n            recency=state["numeric_mean"][1],\n            velocity=0,\n            amount_delta=state["numeric_mean"][3],\n        )\n\n        dense = preprocessor.transform(\n            [row]\n        ).toarray()[0]\n\n        self.assertAlmostEqual(\n            float(dense[0]),\n            1.0,\n            places=6,\n        )\n        self.assertAlmostEqual(\n            float(dense[1]),\n            0.0,\n            places=6,\n        )\n        self.assertAlmostEqual(\n            float(dense[3]),\n            0.0,\n            places=6,\n        )\n\n    def test_structural_missing_maps_to_standardized_zero(self) -> None:\n        preprocessor = FrozenPreprocessor.from_state_dict(\n            frozen_state()\n        )\n\n        dense = preprocessor.transform(\n            [semantic_row(\n                recency=None,\n                amount_delta=None,\n            )]\n        ).toarray()[0]\n\n        self.assertEqual(\n            float(dense[1]),\n            0.0,\n        )\n        self.assertEqual(\n            float(dense[3]),\n            0.0,\n        )\n\n    def test_boolean_features_are_passthrough_float32(self) -> None:\n        preprocessor = FrozenPreprocessor.from_state_dict(\n            frozen_state()\n        )\n\n        dense = preprocessor.transform(\n            [semantic_row(\n                new_merchant=True,\n                has_history=False,\n            )]\n        ).toarray()[0]\n\n        self.assertEqual(float(dense[4]), 1.0)\n        self.assertEqual(float(dense[5]), 0.0)\n\n    def test_known_category_activates_expected_column(self) -> None:\n        preprocessor = FrozenPreprocessor.from_state_dict(\n            frozen_state()\n        )\n\n        dense = preprocessor.transform(\n            [semantic_row(\n                mode="Chip Transaction",\n            )]\n        ).toarray()[0]\n\n        chip_index = preprocessor.feature_names.index(\n            "cat__transaction_mode_Chip Transaction"\n        )\n        unknown_index = preprocessor.feature_names.index(\n            "cat__transaction_mode___UNKNOWN__"\n        )\n\n        self.assertEqual(\n            float(dense[chip_index]),\n            1.0,\n        )\n        self.assertEqual(\n            float(dense[unknown_index]),\n            0.0,\n        )\n\n    def test_unknown_category_activates_explicit_unknown(self) -> None:\n        preprocessor = FrozenPreprocessor.from_state_dict(\n            frozen_state()\n        )\n\n        dense = preprocessor.transform(\n            [semantic_row(\n                mode="Future Transaction Mode",\n            )]\n        ).toarray()[0]\n\n        unknown_index = preprocessor.feature_names.index(\n            "cat__transaction_mode___UNKNOWN__"\n        )\n\n        self.assertEqual(\n            float(dense[unknown_index]),\n            1.0,\n        )\n\n    def test_unknown_hour_and_day_use_unknown_columns(self) -> None:\n        preprocessor = FrozenPreprocessor.from_state_dict(\n            frozen_state()\n        )\n\n        dense = preprocessor.transform(\n            [semantic_row(\n                hour="99",\n                day="9",\n            )]\n        ).toarray()[0]\n\n        hour_unknown = preprocessor.feature_names.index(\n            "cat__hour_of_day___UNKNOWN__"\n        )\n        day_unknown = preprocessor.feature_names.index(\n            "cat__day_of_week___UNKNOWN__"\n        )\n\n        self.assertEqual(\n            float(dense[hour_unknown]),\n            1.0,\n        )\n        self.assertEqual(\n            float(dense[day_unknown]),\n            1.0,\n        )\n\n    def test_each_categorical_branch_has_exactly_one_hot(self) -> None:\n        preprocessor = FrozenPreprocessor.from_state_dict(\n            frozen_state()\n        )\n\n        dense = preprocessor.transform(\n            [semantic_row()]\n        ).toarray()[0]\n\n        categorical = dense[6:]\n\n        self.assertEqual(\n            int(np.count_nonzero(categorical)),\n            4,\n        )\n        self.assertEqual(\n            float(categorical.sum()),\n            4.0,\n        )\n\n    def test_empty_input_returns_zero_by_47_csr(self) -> None:\n        preprocessor = FrozenPreprocessor.from_state_dict(\n            frozen_state()\n        )\n\n        matrix = preprocessor.transform([])\n\n        self.assertTrue(\n            sparse.isspmatrix_csr(matrix)\n        )\n        self.assertEqual(\n            matrix.shape,\n            (0, 47),\n        )\n        self.assertEqual(\n            matrix.dtype,\n            np.float32,\n        )\n\n    def test_non_positive_scale_is_rejected(self) -> None:\n        state = frozen_state()\n        state["numeric_scale"][0] = 0.0\n\n        with self.assertRaises(\n            PreprocessingContractError\n        ):\n            FrozenPreprocessor.from_state_dict(\n                state\n            )\n\n    def test_feature_name_mismatch_is_rejected(self) -> None:\n        state = frozen_state()\n        state["feature_names"][0] = "wrong"\n\n        with self.assertRaises(\n            PreprocessingContractError\n        ):\n            FrozenPreprocessor.from_state_dict(\n                state\n            )\n\n    def test_category_vocab_shape_mismatch_is_rejected(self) -> None:\n        state = frozen_state()\n        del state["category_vocab"][\n            "location_state"\n        ]\n\n        with self.assertRaises(\n            PreprocessingContractError\n        ):\n            FrozenPreprocessor.from_state_dict(\n                state\n            )\n\n    def test_non_finite_numeric_input_is_rejected(self) -> None:\n        preprocessor = FrozenPreprocessor.from_state_dict(\n            frozen_state()\n        )\n\n        with self.assertRaises(\n            PreprocessingContractError\n        ):\n            preprocessor.transform(\n                [semantic_row(\n                    amount=float("inf")\n                )]\n            )\n\n    def test_json_loader_reads_state_without_fitting(self) -> None:\n        state = frozen_state()\n\n        with tempfile.TemporaryDirectory() as tmp:\n            path = Path(tmp) / "state.json"\n            path.write_text(\n                json.dumps(state),\n                encoding="utf-8",\n            )\n\n            preprocessor = load_frozen_preprocessor(\n                path\n            )\n\n            self.assertEqual(\n                len(preprocessor.feature_names),\n                47,\n            )\n\n\nif __name__ == "__main__":\n    unittest.main()\n',
}

PYPROJECT_PATH = Path("final_pipeline/pyproject.toml")
EXPECTED_PYPROJECT = '[build-system]\nrequires = ["setuptools>=68"]\nbuild-backend = "setuptools.build_meta"\n\n[project]\nname = "fraud-screening"\nversion = "0.1.0"\ndescription = "Transaction fraud risk screening pipeline"\nrequires-python = ">=3.11"\n\n[tool.setuptools.packages.find]\nwhere = ["src"]\n'
UPDATED_PYPROJECT = '[build-system]\nrequires = ["setuptools>=68"]\nbuild-backend = "setuptools.build_meta"\n\n[project]\nname = "fraud-screening"\nversion = "0.1.0"\ndescription = "Transaction fraud risk screening pipeline"\nrequires-python = ">=3.11"\ndependencies = [\n    "numpy",\n    "scipy",\n]\n\n[tool.setuptools.packages.find]\nwhere = ["src"]\n'

PREPROCESSING_STATE_REL = Path(
    "research/data/processed/m8_02_final_test_artifact_audit/"
    "m8_02_w_short_preprocessing_state.json"
)

EXPECTED_PREPROCESSING_SHA256 = (
    "c1dc5486acdcd43f5f75fbd0a11bd7ad3a3b00761d50ae96b2b1fbe5b5d9ef98"
)


def detect_root() -> Path:
    root = Path.cwd().resolve()

    required = [
        root / "research",
        root / "final_pipeline",
        root / "application",
    ]

    if not all(path.is_dir() for path in required):
        raise RuntimeError(
            "Run this script from the repository root."
        )

    return root


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def dependency_status() -> dict[str, bool]:
    return {
        "numpy":
            importlib.util.find_spec("numpy")
            is not None,
        "scipy":
            importlib.util.find_spec("scipy")
            is not None,
    }


def verify_artifact(root: Path) -> None:
    state_path = root / PREPROCESSING_STATE_REL

    print("=" * 90)
    print("FINAL PIPELINE — FROZEN PREPROCESSING ARTIFACT VERIFICATION")
    print("=" * 90)
    print("Mode: READ-ONLY TRANSFORM VERIFICATION")
    print(" - preprocessing fit: NO")
    print(" - model fit: NO")
    print(" - prediction: NO")
    print(" - file write: NO")

    if not state_path.is_file():
        raise FileNotFoundError(state_path)

    actual_sha = sha256_file(state_path)

    print("\n[1] Frozen state fingerprint")
    print(" - actual  :", actual_sha)
    print(" - expected:", EXPECTED_PREPROCESSING_SHA256)
    print(
        " - match   :",
        actual_sha
        == EXPECTED_PREPROCESSING_SHA256,
    )

    if actual_sha != EXPECTED_PREPROCESSING_SHA256:
        raise RuntimeError(
            "Frozen preprocessing fingerprint mismatch."
        )

    src_path = root / "final_pipeline/src"
    sys.path.insert(0, str(src_path))

    try:
        import numpy as np
        from scipy import sparse

        from fraud_screening.features import (
            SemanticFeatureRow,
        )
        from fraud_screening.preprocessing import (
            load_frozen_preprocessor,
        )

        preprocessor = load_frozen_preprocessor(
            state_path
        )

        rows = [
            SemanticFeatureRow(
                amount_numeric=
                    42.879027959212905,
                time_since_previous_transaction_min=
                    None,
                transactions_last_1h=0,
                amount_minus_previous_mean=
                    None,
                is_new_merchant=True,
                has_prior_card_history=False,
                transaction_mode=
                    "Chip Transaction",
                location_state=
                    "PHYSICAL_COMPLETE",
                hour_of_day="14",
                day_of_week="0",
            ),
            SemanticFeatureRow(
                amount_numeric=
                    42.879027959212905
                    + 80.55065611888665,
                time_since_previous_transaction_min=
                    1197.3147570599144,
                transactions_last_1h=1,
                amount_minus_previous_mean=
                    -0.4259839479714494,
                is_new_merchant=False,
                has_prior_card_history=True,
                transaction_mode=
                    "UNSEEN MODE",
                location_state=
                    "PHYSICAL_COMPLETE",
                hour_of_day="14",
                day_of_week="0",
            ),
        ]

        matrix = preprocessor.transform(rows)
        dense = matrix.toarray()

        print("\n[2] Loaded frozen identity")
        print(
            " - feature count:",
            len(preprocessor.feature_names),
        )
        print(
            " - strategy:",
            preprocessor.strategy,
        )
        print(
            " - fit source:",
            preprocessor.fit_source,
        )

        print("\n[3] Transform checks")
        print(" - shape:", matrix.shape)
        print(" - dtype:", matrix.dtype)
        print(
            " - CSR:",
            sparse.isspmatrix_csr(matrix),
        )
        print(
            " - finite:",
            np.isfinite(matrix.data).all(),
        )

        recency_index = preprocessor.feature_names.index(
            "num__time_since_previous_transaction_min"
        )
        delta_index = preprocessor.feature_names.index(
            "num__amount_minus_previous_mean"
        )
        amount_index = preprocessor.feature_names.index(
            "num__amount_numeric"
        )
        unknown_mode_index = (
            preprocessor.feature_names.index(
                "cat__transaction_mode___UNKNOWN__"
            )
        )

        gates = {
            "G01_FEATURE_COUNT_47":
                len(preprocessor.feature_names)
                == 47,
            "G02_MATRIX_SHAPE":
                matrix.shape == (2, 47),
            "G03_CSR_FLOAT32":
                sparse.isspmatrix_csr(matrix)
                and matrix.dtype == np.float32,
            "G04_FINITE_OUTPUT":
                bool(
                    np.isfinite(
                        matrix.data
                    ).all()
                ),
            "G05_COLD_START_RECENCY_ZERO":
                float(
                    dense[0, recency_index]
                ) == 0.0,
            "G06_COLD_START_AMOUNT_DELTA_ZERO":
                float(
                    dense[0, delta_index]
                ) == 0.0,
            "G07_FROZEN_NUMERIC_SCALE":
                abs(
                    float(
                        dense[1, amount_index]
                    )
                    - 1.0
                )
                < 1e-6,
            "G08_UNKNOWN_CATEGORY_COLUMN":
                float(
                    dense[
                        1,
                        unknown_mode_index,
                    ]
                )
                == 1.0,
        }

        print("\n[4] Verification gates")
        for name, passed in gates.items():
            print(
                f" - {name}: "
                f"{'PASS' if passed else 'FAIL'}"
            )

        if not all(gates.values()):
            print("\nVERIFICATION RESULT: FAIL")
            raise SystemExit(1)

        print("\nVERIFICATION RESULT: PASS")
        print(
            "Frozen state transformed 10 semantic "
            "features into exact 47-column CSR float32."
        )
        print(
            "No fit, no learned-state mutation, "
            "no prediction."
        )
        print("=" * 90)

    finally:
        try:
            sys.path.remove(str(src_path))
        except ValueError:
            pass


def main() -> None:
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group()

    group.add_argument(
        "--execute",
        action="store_true",
        help="Create frozen preprocessing source and tests.",
    )

    group.add_argument(
        "--verify-artifact",
        action="store_true",
        help="Verify implementation against the frozen research artifact.",
    )

    args = parser.parse_args()
    root = detect_root()

    if args.verify_artifact:
        verify_artifact(root)
        return

    pyproject = root / PYPROJECT_PATH

    dependencies = [
        root / "final_pipeline/src/fraud_screening/features/semantic.py",
        root / "final_pipeline/src/fraud_screening/errors.py",
        root / "final_pipeline/tests/unit/test_semantic_features.py",
        pyproject,
    ]

    missing_dependencies = [
        str(path.relative_to(root))
        for path in dependencies
        if not path.is_file()
    ]

    collisions = [
        relative
        for relative in NEW_FILES
        if (root / relative).exists()
    ]

    pyproject_matches = (
        pyproject.is_file()
        and pyproject.read_text(
            encoding="utf-8"
        )
        == EXPECTED_PYPROJECT
    )

    packages = dependency_status()

    print("=" * 90)
    print("FINAL PIPELINE — FROZEN PREPROCESSING BOOTSTRAP")
    print("=" * 90)
    print("Project root:", root)
    print(
        "Mode:",
        "EXECUTE — GUARDED UPDATE + NEW FILES"
        if args.execute
        else "DRY RUN — NO FILES WRITTEN",
    )

    print("\n[1] Dependency gate")
    print(" - missing project dependencies:", len(missing_dependencies))
    for relative in missing_dependencies:
        print("   *", relative)
    for package, available in packages.items():
        print(
            f" - Python package {package}: "
            f"{'AVAILABLE' if available else 'MISSING'}"
        )

    print("\n[2] pyproject guard")
    print(
        " - exact expected current content:",
        "YES" if pyproject_matches else "NO",
    )

    print("\n[3] Planned new files")
    for relative in NEW_FILES:
        print(" -", relative)

    print("\n[4] New-file collision gate")
    print(" - collisions:", len(collisions))
    for relative in collisions:
        print("   *", relative)

    if (
        missing_dependencies
        or collisions
        or not pyproject_matches
        or not all(packages.values())
    ):
        print("\nRESULT: STOP")
        print("No files were written.")
        raise SystemExit(1)

    print("\n[5] Frozen transform contract")
    print(" - input semantic width: 10")
    print(" - output encoded width: 47")
    print(" - numeric learned state: LOAD ONLY")
    print(" - categorical vocabulary: LOAD ONLY")
    print(" - unknown category: explicit __UNKNOWN__")
    print(" - structural NA after scaling: 0.0")
    print(" - boolean branch: float32 passthrough")
    print(" - output: CSR float32")
    print(" - sklearn fit/encoder reconstruction: NOT REQUIRED")

    print("\n[6] Scientific/runtime boundary")
    print(" - preprocessing fit: NO")
    print(" - model fit: NO")
    print(" - prediction: NO")
    print(" - threshold change: NO")
    print(" - research files modified: NO")
    print(" - official artifacts modified: NO")

    if not args.execute:
        print("\nDRY-RUN RESULT: PASS")
        print("Execute with:")
        print(
            "python bootstrap_frozen_preprocessing.py --execute"
        )
        print("=" * 90)
        return

    created = []
    original_pyproject = pyproject.read_text(
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
            created.append(path)

        pyproject.write_text(
            UPDATED_PYPROJECT,
            encoding="utf-8",
        )

    except Exception:
        for path in reversed(created):
            try:
                path.unlink()
            except OSError:
                pass

        try:
            pyproject.write_text(
                original_pyproject,
                encoding="utf-8",
            )
        except OSError:
            pass

        raise

    print("\n[7] Created")
    for path in created:
        print(" -", path.relative_to(root))

    print("\n[8] Guarded technical metadata update")
    print(" -", PYPROJECT_PATH)
    print("   added runtime dependencies: numpy, scipy")

    print("\nBOOTSTRAP RESULT: PASS")
    print("Run full unit suite:")
    print(
        "PYTHONPATH=final_pipeline/src "
        "python -m unittest discover "
        "-s final_pipeline/tests -v"
    )
    print("Then verify against the frozen artifact:")
    print(
        "python bootstrap_frozen_preprocessing.py --verify-artifact"
    )
    print("=" * 90)


if __name__ == "__main__":
    main()
