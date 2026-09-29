from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy import sparse

from fraud_screening.errors import PreprocessingContractError
from fraud_screening.features import SemanticFeatureRow
from fraud_screening.preprocessing import (
    FrozenPreprocessor,
    load_frozen_preprocessor,
)


def frozen_state() -> dict:
    category_vocab = {
        "transaction_mode": [
            "Chip Transaction",
            "Online Transaction",
            "Swipe Transaction",
        ],
        "location_state": [
            "NON_PHYSICAL_OR_ONLINE",
            "PHYSICAL_COMPLETE",
            "PHYSICAL_ZIP_UNAVAILABLE",
        ],
        "hour_of_day": [
            str(value)
            for value in range(24)
        ],
        "day_of_week": [
            str(value)
            for value in range(7)
        ],
    }

    unknown = "__UNKNOWN__"

    feature_names = [
        "num__amount_numeric",
        "num__time_since_previous_transaction_min",
        "num__transactions_last_1h",
        "num__amount_minus_previous_mean",
        "bool__is_new_merchant",
        "bool__has_prior_card_history",
    ]

    for column in (
        "transaction_mode",
        "location_state",
        "hour_of_day",
        "day_of_week",
    ):
        feature_names.extend(
            f"cat__{column}_{category}"
            for category in (
                *category_vocab[column],
                unknown,
            )
        )

    return {
        "analysis_version":
            "frozen-evaluation-state",
        "categorical_columns": [
            "transaction_mode",
            "location_state",
            "hour_of_day",
            "day_of_week",
        ],
        "category_vocab":
            category_vocab,
        "feature_count":
            47,
        "feature_names":
            feature_names,
        "fit_max_timestamp":
            "2018-12-31 23:58:00",
        "fit_min_timestamp":
            "2018-01-01 00:03:00",
        "fit_row_count":
            1721615,
        "fit_source":
            "W_SHORT_TRAIN_ONLY",
        "matrix_dtype":
            "float32",
        "matrix_format":
            "CSR",
        "numeric_columns": [
            "amount_numeric",
            "time_since_previous_transaction_min",
            "transactions_last_1h",
            "amount_minus_previous_mean",
        ],
        "numeric_mean": [
            42.879027959212905,
            1197.3147570599144,
            0.2759606532238619,
            -0.4259839479714494,
        ],
        "numeric_n_samples_seen": [
            1721615,
            1721515,
            1721615,
            1721515,
        ],
        "numeric_scale": [
            80.55065611888665,
            2113.3233635919028,
            0.6581493130130939,
            78.17323847945735,
        ],
        "numeric_var": [
            6488.408201183131,
            4466135.639103394,
            0.43316051821960744,
            6111.0552143661125,
        ],
        "strategy":
            "W_SHORT",
        "unknown_token":
            unknown,
        "validation_exact_reproduction":
            True,
    }


def semantic_row(
    *,
    amount: float = 42.879027959212905,
    recency: float | None = None,
    velocity: int = 0,
    amount_delta: float | None = None,
    new_merchant: bool = True,
    has_history: bool = False,
    mode: str = "Chip Transaction",
    location: str = "PHYSICAL_COMPLETE",
    hour: str = "14",
    day: str = "0",
) -> SemanticFeatureRow:
    return SemanticFeatureRow(
        amount_numeric=amount,
        time_since_previous_transaction_min=
            recency,
        transactions_last_1h=velocity,
        amount_minus_previous_mean=
            amount_delta,
        is_new_merchant=new_merchant,
        has_prior_card_history=has_history,
        transaction_mode=mode,
        location_state=location,
        hour_of_day=hour,
        day_of_week=day,
    )


class FrozenPreprocessingTests(unittest.TestCase):
    def test_state_loads_with_exact_47_schema(self) -> None:
        preprocessor = FrozenPreprocessor.from_state_dict(
            frozen_state()
        )

        self.assertEqual(
            len(preprocessor.feature_names),
            47,
        )

    def test_transform_returns_csr_float32(self) -> None:
        preprocessor = FrozenPreprocessor.from_state_dict(
            frozen_state()
        )

        matrix = preprocessor.transform(
            [semantic_row()]
        )

        self.assertTrue(
            sparse.isspmatrix_csr(matrix)
        )
        self.assertEqual(
            matrix.shape,
            (1, 47),
        )
        self.assertEqual(
            matrix.dtype,
            np.float32,
        )

    def test_numeric_scaling_uses_frozen_mean_and_scale(self) -> None:
        state = frozen_state()
        preprocessor = FrozenPreprocessor.from_state_dict(
            state
        )

        row = semantic_row(
            amount=state["numeric_mean"][0]
            + state["numeric_scale"][0],
            recency=state["numeric_mean"][1],
            velocity=0,
            amount_delta=state["numeric_mean"][3],
        )

        dense = preprocessor.transform(
            [row]
        ).toarray()[0]

        self.assertAlmostEqual(
            float(dense[0]),
            1.0,
            places=6,
        )
        self.assertAlmostEqual(
            float(dense[1]),
            0.0,
            places=6,
        )
        self.assertAlmostEqual(
            float(dense[3]),
            0.0,
            places=6,
        )

    def test_structural_missing_maps_to_standardized_zero(self) -> None:
        preprocessor = FrozenPreprocessor.from_state_dict(
            frozen_state()
        )

        dense = preprocessor.transform(
            [semantic_row(
                recency=None,
                amount_delta=None,
            )]
        ).toarray()[0]

        self.assertEqual(
            float(dense[1]),
            0.0,
        )
        self.assertEqual(
            float(dense[3]),
            0.0,
        )

    def test_boolean_features_are_passthrough_float32(self) -> None:
        preprocessor = FrozenPreprocessor.from_state_dict(
            frozen_state()
        )

        dense = preprocessor.transform(
            [semantic_row(
                new_merchant=True,
                has_history=False,
            )]
        ).toarray()[0]

        self.assertEqual(float(dense[4]), 1.0)
        self.assertEqual(float(dense[5]), 0.0)

    def test_known_category_activates_expected_column(self) -> None:
        preprocessor = FrozenPreprocessor.from_state_dict(
            frozen_state()
        )

        dense = preprocessor.transform(
            [semantic_row(
                mode="Chip Transaction",
            )]
        ).toarray()[0]

        chip_index = preprocessor.feature_names.index(
            "cat__transaction_mode_Chip Transaction"
        )
        unknown_index = preprocessor.feature_names.index(
            "cat__transaction_mode___UNKNOWN__"
        )

        self.assertEqual(
            float(dense[chip_index]),
            1.0,
        )
        self.assertEqual(
            float(dense[unknown_index]),
            0.0,
        )

    def test_unknown_category_activates_explicit_unknown(self) -> None:
        preprocessor = FrozenPreprocessor.from_state_dict(
            frozen_state()
        )

        dense = preprocessor.transform(
            [semantic_row(
                mode="Future Transaction Mode",
            )]
        ).toarray()[0]

        unknown_index = preprocessor.feature_names.index(
            "cat__transaction_mode___UNKNOWN__"
        )

        self.assertEqual(
            float(dense[unknown_index]),
            1.0,
        )

    def test_unknown_hour_and_day_use_unknown_columns(self) -> None:
        preprocessor = FrozenPreprocessor.from_state_dict(
            frozen_state()
        )

        dense = preprocessor.transform(
            [semantic_row(
                hour="99",
                day="9",
            )]
        ).toarray()[0]

        hour_unknown = preprocessor.feature_names.index(
            "cat__hour_of_day___UNKNOWN__"
        )
        day_unknown = preprocessor.feature_names.index(
            "cat__day_of_week___UNKNOWN__"
        )

        self.assertEqual(
            float(dense[hour_unknown]),
            1.0,
        )
        self.assertEqual(
            float(dense[day_unknown]),
            1.0,
        )

    def test_each_categorical_branch_has_exactly_one_hot(self) -> None:
        preprocessor = FrozenPreprocessor.from_state_dict(
            frozen_state()
        )

        dense = preprocessor.transform(
            [semantic_row()]
        ).toarray()[0]

        categorical = dense[6:]

        self.assertEqual(
            int(np.count_nonzero(categorical)),
            4,
        )
        self.assertEqual(
            float(categorical.sum()),
            4.0,
        )

    def test_empty_input_returns_zero_by_47_csr(self) -> None:
        preprocessor = FrozenPreprocessor.from_state_dict(
            frozen_state()
        )

        matrix = preprocessor.transform([])

        self.assertTrue(
            sparse.isspmatrix_csr(matrix)
        )
        self.assertEqual(
            matrix.shape,
            (0, 47),
        )
        self.assertEqual(
            matrix.dtype,
            np.float32,
        )

    def test_non_positive_scale_is_rejected(self) -> None:
        state = frozen_state()
        state["numeric_scale"][0] = 0.0

        with self.assertRaises(
            PreprocessingContractError
        ):
            FrozenPreprocessor.from_state_dict(
                state
            )

    def test_feature_name_mismatch_is_rejected(self) -> None:
        state = frozen_state()
        state["feature_names"][0] = "wrong"

        with self.assertRaises(
            PreprocessingContractError
        ):
            FrozenPreprocessor.from_state_dict(
                state
            )

    def test_category_vocab_shape_mismatch_is_rejected(self) -> None:
        state = frozen_state()
        del state["category_vocab"][
            "location_state"
        ]

        with self.assertRaises(
            PreprocessingContractError
        ):
            FrozenPreprocessor.from_state_dict(
                state
            )

    def test_non_finite_numeric_input_is_rejected(self) -> None:
        preprocessor = FrozenPreprocessor.from_state_dict(
            frozen_state()
        )

        with self.assertRaises(
            PreprocessingContractError
        ):
            preprocessor.transform(
                [semantic_row(
                    amount=float("inf")
                )]
            )

    def test_json_loader_reads_state_without_fitting(self) -> None:
        state = frozen_state()

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            path.write_text(
                json.dumps(state),
                encoding="utf-8",
            )

            preprocessor = load_frozen_preprocessor(
                path
            )

            self.assertEqual(
                len(preprocessor.feature_names),
                47,
            )


if __name__ == "__main__":
    unittest.main()
