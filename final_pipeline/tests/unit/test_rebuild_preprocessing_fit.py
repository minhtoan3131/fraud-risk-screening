from __future__ import annotations

import unittest
from datetime import datetime, timedelta

import numpy as np
from scipy import sparse
from sklearn.preprocessing import StandardScaler

from fraud_screening.errors import PreprocessingContractError
from fraud_screening.features import SemanticFeatureRow
from fraud_screening.training import (
    RebuildPreprocessingFitter,
    fit_rebuild_preprocessor,
)


def training_fixture(
    count: int = 24,
) -> tuple[
    list[SemanticFeatureRow],
    list[datetime],
]:
    modes = [
        "Chip Transaction",
        "Online Transaction",
        "Swipe Transaction",
    ]

    locations = [
        "NON_PHYSICAL_OR_ONLINE",
        "PHYSICAL_COMPLETE",
        "PHYSICAL_ZIP_UNAVAILABLE",
    ]

    rows = []
    timestamps = []

    start = datetime(
        2018,
        1,
        1,
        0,
        0,
    )

    for index in range(count):
        rows.append(
            SemanticFeatureRow(
                amount_numeric=
                    float(index + 1),
                time_since_previous_transaction_min=
                    None
                    if index == 0
                    else float(index * 10),
                transactions_last_1h=
                    index % 4,
                amount_minus_previous_mean=
                    None
                    if index == 0
                    else float(index - 5),
                is_new_merchant=
                    index < 3,
                has_prior_card_history=
                    index != 0,
                transaction_mode=
                    modes[
                        index % 3
                    ],
                location_state=
                    locations[
                        index % 3
                    ],
                hour_of_day=
                    str(index % 24),
                day_of_week=
                    str(index % 7),
            )
        )

        timestamps.append(
            start
            + timedelta(
                hours=index
            )
        )

    return rows, timestamps


class RebuildPreprocessingFitTests(unittest.TestCase):
    def test_fit_produces_exact_47_column_contract(self) -> None:
        rows, timestamps = training_fixture()

        result = fit_rebuild_preprocessor(
            rows,
            timestamps,
        )

        self.assertEqual(
            result.state[
                "feature_count"
            ],
            47,
        )
        self.assertEqual(
            len(
                result.state[
                    "feature_names"
                ]
            ),
            47,
        )
        self.assertEqual(
            result.state[
                "matrix_format"
            ],
            "CSR",
        )
        self.assertEqual(
            result.state[
                "matrix_dtype"
            ],
            "float32",
        )

    def test_rebuild_state_does_not_claim_official_validation(self) -> None:
        rows, timestamps = training_fixture()

        result = fit_rebuild_preprocessor(
            rows,
            timestamps,
        )

        self.assertNotIn(
            "validation_exact_reproduction",
            result.state,
        )
        self.assertEqual(
            result.state[
                "rebuild_validation_status"
            ],
            "NOT_COMPARED_TO_OFFICIAL_ARTIFACT",
        )

    def test_transform_after_fit_returns_csr_float32(self) -> None:
        rows, timestamps = training_fixture()

        result = fit_rebuild_preprocessor(
            rows,
            timestamps,
        )

        matrix = (
            result.preprocessor
            .transform(
                rows
            )
        )

        self.assertTrue(
            sparse.isspmatrix_csr(
                matrix
            )
        )
        self.assertEqual(
            matrix.shape,
            (24, 47),
        )
        self.assertEqual(
            matrix.dtype,
            np.float32,
        )

    def test_structural_na_is_excluded_from_numeric_fit(self) -> None:
        rows, timestamps = training_fixture()

        result = fit_rebuild_preprocessor(
            rows,
            timestamps,
        )

        self.assertEqual(
            result.state[
                "numeric_n_samples_seen"
            ],
            [24, 23, 24, 23],
        )

    def test_numeric_state_matches_independent_standard_scaler(self) -> None:
        rows, timestamps = training_fixture()

        result = fit_rebuild_preprocessor(
            rows,
            timestamps,
        )

        amount_values = np.asarray(
            [
                row.amount_numeric
                for row in rows
            ],
            dtype=np.float64,
        ).reshape(-1, 1)

        reference = (
            StandardScaler()
            .fit(
                amount_values
            )
        )

        self.assertAlmostEqual(
            result.state[
                "numeric_mean"
            ][0],
            float(
                reference.mean_[0]
            ),
            places=12,
        )
        self.assertAlmostEqual(
            result.state[
                "numeric_var"
            ][0],
            float(
                reference.var_[0]
            ),
            places=12,
        )
        self.assertAlmostEqual(
            result.state[
                "numeric_scale"
            ][0],
            float(
                reference.scale_[0]
            ),
            places=12,
        )

    def test_two_chronological_batches_match_single_batch_state(self) -> None:
        rows, timestamps = training_fixture()

        single = fit_rebuild_preprocessor(
            rows,
            timestamps,
        )

        fitter = (
            RebuildPreprocessingFitter()
        )

        fitter.partial_fit(
            rows[:12],
            timestamps[:12],
        )
        fitter.partial_fit(
            rows[12:],
            timestamps[12:],
        )

        batched = fitter.finalize()

        np.testing.assert_allclose(
            batched.state[
                "numeric_mean"
            ],
            single.state[
                "numeric_mean"
            ],
            rtol=0.0,
            atol=1e-12,
        )

        np.testing.assert_allclose(
            batched.state[
                "numeric_var"
            ],
            single.state[
                "numeric_var"
            ],
            rtol=0.0,
            atol=1e-12,
        )

        np.testing.assert_allclose(
            batched.state[
                "numeric_scale"
            ],
            single.state[
                "numeric_scale"
            ],
            rtol=0.0,
            atol=1e-12,
        )

        self.assertEqual(
            batched.state[
                "category_vocab"
            ],
            single.state[
                "category_vocab"
            ],
        )

    def test_batch_order_does_not_require_global_timestamp_sort(self) -> None:
        rows, timestamps = training_fixture()

        single = fit_rebuild_preprocessor(
            rows,
            timestamps,
        )

        fitter = (
            RebuildPreprocessingFitter()
        )

        # Deliberately process the later half first. This represents
        # order-invariant preprocessing over already-canonical rows;
        # history causality was enforced before this stage.
        fitter.partial_fit(
            rows[12:],
            timestamps[12:],
        )
        fitter.partial_fit(
            rows[:12],
            timestamps[:12],
        )

        reordered = fitter.finalize()

        np.testing.assert_allclose(
            reordered.state[
                "numeric_mean"
            ],
            single.state[
                "numeric_mean"
            ],
            rtol=0.0,
            atol=1e-12,
        )

        np.testing.assert_allclose(
            reordered.state[
                "numeric_var"
            ],
            single.state[
                "numeric_var"
            ],
            rtol=0.0,
            atol=1e-12,
        )

        self.assertEqual(
            reordered.state[
                "fit_row_count"
            ],
            24,
        )
        self.assertEqual(
            reordered.state[
                "fit_min_timestamp"
            ],
            "2018-01-01 00:00:00",
        )
        self.assertEqual(
            reordered.state[
                "fit_max_timestamp"
            ],
            "2018-01-01 23:00:00",
        )

    def test_missing_expected_category_is_rejected(self) -> None:
        rows, timestamps = training_fixture(
            count=23
        )

        with self.assertRaises(
            PreprocessingContractError
        ):
            fit_rebuild_preprocessor(
                rows,
                timestamps,
            )

    def test_reserved_unknown_token_is_rejected_in_train(self) -> None:
        rows, timestamps = training_fixture()

        row = rows[0]

        rows[0] = SemanticFeatureRow(
            amount_numeric=
                row.amount_numeric,
            time_since_previous_transaction_min=
                row.time_since_previous_transaction_min,
            transactions_last_1h=
                row.transactions_last_1h,
            amount_minus_previous_mean=
                row.amount_minus_previous_mean,
            is_new_merchant=
                row.is_new_merchant,
            has_prior_card_history=
                row.has_prior_card_history,
            transaction_mode=
                "__UNKNOWN__",
            location_state=
                row.location_state,
            hour_of_day=
                row.hour_of_day,
            day_of_week=
                row.day_of_week,
        )

        with self.assertRaises(
            PreprocessingContractError
        ):
            fit_rebuild_preprocessor(
                rows,
                timestamps,
            )

    def test_all_missing_structural_numeric_feature_is_rejected(self) -> None:
        rows, timestamps = training_fixture()

        rows = [
            SemanticFeatureRow(
                amount_numeric=
                    row.amount_numeric,
                time_since_previous_transaction_min=
                    None,
                transactions_last_1h=
                    row.transactions_last_1h,
                amount_minus_previous_mean=
                    row.amount_minus_previous_mean,
                is_new_merchant=
                    row.is_new_merchant,
                has_prior_card_history=
                    row.has_prior_card_history,
                transaction_mode=
                    row.transaction_mode,
                location_state=
                    row.location_state,
                hour_of_day=
                    row.hour_of_day,
                day_of_week=
                    row.day_of_week,
            )
            for row in rows
        ]

        with self.assertRaises(
            PreprocessingContractError
        ):
            fit_rebuild_preprocessor(
                rows,
                timestamps,
            )

    def test_fit_timestamp_metadata_is_recorded(self) -> None:
        rows, timestamps = training_fixture()

        result = fit_rebuild_preprocessor(
            rows,
            timestamps,
        )

        self.assertEqual(
            result.state[
                "fit_row_count"
            ],
            24,
        )
        self.assertEqual(
            result.state[
                "fit_min_timestamp"
            ],
            "2018-01-01 00:00:00",
        )
        self.assertEqual(
            result.state[
                "fit_max_timestamp"
            ],
            "2018-01-01 23:00:00",
        )


if __name__ == "__main__":
    unittest.main()
