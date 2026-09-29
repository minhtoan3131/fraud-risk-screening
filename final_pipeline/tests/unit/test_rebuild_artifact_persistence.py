from __future__ import annotations

from datetime import datetime, timedelta
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np

from fraud_screening.features import (
    SemanticFeatureRow,
)
from fraud_screening.training import (
    fit_rebuild_model,
    fit_rebuild_preprocessor,
    persist_rebuild_artifacts,
)


def semantic_fixture() -> tuple[
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
    )

    for index in range(24):
        rows.append(
            SemanticFeatureRow(
                amount_numeric=
                    float(index + 1),
                time_since_previous_transaction_min=
                    None
                    if index == 0
                    else float(index + 1),
                transactions_last_1h=
                    index % 3,
                amount_minus_previous_mean=
                    None
                    if index == 0
                    else float(index - 3),
                is_new_merchant=
                    index < 4,
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
                    str(index),
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


def fitted_pair():
    rows, timestamps = (
        semantic_fixture()
    )

    preprocessing_fit = (
        fit_rebuild_preprocessor(
            rows,
            timestamps,
        )
    )

    matrix = (
        preprocessing_fit
        .preprocessor
        .transform(
            rows
        )
    )

    target = np.asarray(
        [
            1
            if index % 6 == 0
            else 0
            for index
            in range(len(rows))
        ],
        dtype=np.int8,
    )

    target[0] = 0
    target[1] = 1

    model_fit = fit_rebuild_model(
        matrix,
        target,
    )

    return (
        model_fit,
        preprocessing_fit,
    )


class RebuildArtifactPersistenceTests(
    unittest.TestCase
):
    def test_persist_writes_exact_three_files_under_rebuild(self) -> None:
        model_fit, preprocessing_fit = (
            fitted_pair()
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            result = (
                persist_rebuild_artifacts(
                    root,
                    "run-001",
                    model_fit=
                        model_fit,
                    preprocessing_fit=
                        preprocessing_fit,
                )
            )

            self.assertEqual(
                {
                    path.name
                    for path
                    in result.paths.run_dir.iterdir()
                    if path.is_file()
                },
                {
                    "model.joblib",
                    "preprocessing_state.json",
                    "rebuild_manifest.json",
                },
            )

            self.assertIn(
                "final_pipeline/outputs/rebuilds/run-001",
                result.paths.run_dir.as_posix(),
            )

    def test_manifest_explicitly_denies_official_identity_and_metrics(self) -> None:
        model_fit, preprocessing_fit = (
            fitted_pair()
        )

        with tempfile.TemporaryDirectory() as tmp:
            result = (
                persist_rebuild_artifacts(
                    tmp,
                    "run-001",
                    model_fit=
                        model_fit,
                    preprocessing_fit=
                        preprocessing_fit,
                )
            )

            manifest = result.manifest

            self.assertFalse(
                manifest[
                    "official_artifact_replacement"
                ]
            )
            self.assertFalse(
                manifest[
                    "official_metrics_inherited"
                ]
            )
            self.assertEqual(
                manifest[
                    "model"
                ][
                    "evaluation_status"
                ],
                "NOT_EVALUATED_AS_OFFICIAL",
            )

    def test_manifest_fingerprints_match_persisted_files(self) -> None:
        model_fit, preprocessing_fit = (
            fitted_pair()
        )

        with tempfile.TemporaryDirectory() as tmp:
            result = (
                persist_rebuild_artifacts(
                    tmp,
                    "run-001",
                    model_fit=
                        model_fit,
                    preprocessing_fit=
                        preprocessing_fit,
                )
            )

            manifest = json.loads(
                result.paths.manifest_path
                .read_text(
                    encoding="utf-8"
                )
            )

            self.assertEqual(
                manifest[
                    "model"
                ][
                    "sha256"
                ],
                result.model_sha256,
            )
            self.assertEqual(
                manifest[
                    "preprocessing"
                ][
                    "sha256"
                ],
                result.preprocessing_sha256,
            )

    def test_persisted_model_round_trips_as_fitted_estimator(self) -> None:
        model_fit, preprocessing_fit = (
            fitted_pair()
        )

        with tempfile.TemporaryDirectory() as tmp:
            result = (
                persist_rebuild_artifacts(
                    tmp,
                    "run-001",
                    model_fit=
                        model_fit,
                    preprocessing_fit=
                        preprocessing_fit,
                )
            )

            loaded = joblib.load(
                result.paths.model_path
            )

            self.assertEqual(
                loaded.classes_.tolist(),
                [0, 1],
            )
            self.assertEqual(
                int(
                    loaded.n_features_in_
                ),
                47,
            )
            self.assertEqual(
                len(
                    loaded.estimators_
                ),
                100,
            )

    def test_existing_run_is_not_overwritten(self) -> None:
        model_fit, preprocessing_fit = (
            fitted_pair()
        )

        with tempfile.TemporaryDirectory() as tmp:
            first = (
                persist_rebuild_artifacts(
                    tmp,
                    "run-001",
                    model_fit=
                        model_fit,
                    preprocessing_fit=
                        preprocessing_fit,
                )
            )

            original_manifest = (
                first.paths.manifest_path
                .read_text(
                    encoding="utf-8"
                )
            )

            with self.assertRaises(
                FileExistsError
            ):
                persist_rebuild_artifacts(
                    tmp,
                    "run-001",
                    model_fit=
                        model_fit,
                    preprocessing_fit=
                        preprocessing_fit,
                )

            self.assertEqual(
                first.paths.manifest_path
                .read_text(
                    encoding="utf-8"
                ),
                original_manifest,
            )

    def test_official_directory_is_never_created(self) -> None:
        model_fit, preprocessing_fit = (
            fitted_pair()
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            persist_rebuild_artifacts(
                root,
                "run-001",
                model_fit=
                    model_fit,
                preprocessing_fit=
                    preprocessing_fit,
            )

            self.assertFalse(
                (
                    root
                    / "final_pipeline"
                    / "artifacts"
                    / "official"
                ).exists()
            )

    def test_persistence_failure_rolls_back_run_directory(self) -> None:
        model_fit, preprocessing_fit = (
            fitted_pair()
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            with patch(
                "fraud_screening.training.persistence.joblib.dump",
                side_effect=RuntimeError(
                    "synthetic write failure"
                ),
            ):
                with self.assertRaises(
                    RuntimeError
                ):
                    persist_rebuild_artifacts(
                        root,
                        "run-001",
                        model_fit=
                            model_fit,
                        preprocessing_fit=
                            preprocessing_fit,
                    )

            self.assertFalse(
                (
                    root
                    / "final_pipeline"
                    / "outputs"
                    / "rebuilds"
                    / "run-001"
                ).exists()
            )

    def test_rebuild_preprocessing_status_is_preserved_in_manifest(self) -> None:
        model_fit, preprocessing_fit = (
            fitted_pair()
        )

        with tempfile.TemporaryDirectory() as tmp:
            result = (
                persist_rebuild_artifacts(
                    tmp,
                    "run-001",
                    model_fit=
                        model_fit,
                    preprocessing_fit=
                        preprocessing_fit,
                )
            )

            self.assertEqual(
                result.manifest[
                    "preprocessing"
                ][
                    "rebuild_validation_status"
                ],
                "NOT_COMPARED_TO_OFFICIAL_ARTIFACT",
            )


if __name__ == "__main__":
    unittest.main()
