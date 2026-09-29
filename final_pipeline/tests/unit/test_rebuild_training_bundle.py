from __future__ import annotations

from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from fraud_screening.features import (
    BOOLEAN_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    SEMANTIC_FEATURE_ORDER,
)
from fraud_screening.training.bundle import (
    BUNDLE_ROLE,
    BUNDLE_VERSION,
    CATEGORY_CODEBOOK,
    BOOLEAN_FILE,
    CATEGORICAL_FILE,
    MANIFEST_FILE,
    NUMERIC_FILE,
    TARGET_FILE,
    TIMESTAMP_FILE,
    TrainingBundleProfile,
    load_training_bundle,
)


TEST_PROFILE = TrainingBundleProfile(
    name="W_SHORT_TEST",
    row_count=24,
    fraud_rows=4,
    start_inclusive=np.datetime64(
        "2018-01-01T00:00:00",
        "ns",
    ),
    end_exclusive=np.datetime64(
        "2019-01-01T00:00:00",
        "ns",
    ),
)


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        digest.update(
            handle.read()
        )

    return digest.hexdigest()


def write_bundle(
    root: Path,
) -> Path:
    bundle = root / "bundle"
    bundle.mkdir(
        parents=True,
        exist_ok=False,
    )

    numeric = np.zeros(
        (24, 4),
        dtype=np.float64,
    )

    numeric[:, 0] = np.arange(
        1,
        25,
        dtype=np.float64,
    )
    numeric[:, 1] = np.arange(
        10,
        250,
        10,
        dtype=np.float64,
    )
    numeric[:, 2] = (
        np.arange(
            24,
            dtype=np.float64,
        )
        % 4
    )
    numeric[:, 3] = (
        np.arange(
            24,
            dtype=np.float64,
        )
        - 5
    )
    numeric[0, 1] = np.nan
    numeric[0, 3] = np.nan

    boolean = np.zeros(
        (24, 2),
        dtype=np.bool_,
    )
    boolean[:, 0] = (
        np.arange(24)
        < 3
    )
    boolean[:, 1] = True
    boolean[0, 1] = False

    categorical = np.zeros(
        (24, 4),
        dtype=np.int8,
    )
    categorical[:, 0] = (
        np.arange(24)
        % 3
    )
    categorical[:, 1] = (
        np.arange(24)
        % 3
    )
    categorical[:, 2] = (
        np.arange(24)
    )
    categorical[:, 3] = (
        np.arange(24)
        % 7
    )

    timestamps = np.asarray(
        [
            np.datetime64(
                datetime(
                    2018,
                    1,
                    1,
                )
                + timedelta(
                    hours=index
                ),
                "ns",
            )
            for index in range(24)
        ],
        dtype="datetime64[ns]",
    )

    target = np.zeros(
        24,
        dtype=np.int8,
    )
    target[
        [
            1,
            6,
            12,
            18,
        ]
    ] = 1

    arrays = {
        NUMERIC_FILE:
            numeric,
        BOOLEAN_FILE:
            boolean,
        CATEGORICAL_FILE:
            categorical,
        TIMESTAMP_FILE:
            timestamps,
        TARGET_FILE:
            target,
    }

    for filename, array in arrays.items():
        np.save(
            bundle / filename,
            array,
            allow_pickle=False,
        )

    manifest = {
        "bundle_version":
            BUNDLE_VERSION,
        "role":
            BUNDLE_ROLE,
        "profile":
            TEST_PROFILE.name,
        "row_count":
            TEST_PROFILE.row_count,
        "fraud_rows":
            TEST_PROFILE.fraud_rows,
        "start_inclusive":
            "2018-01-01T00:00:00",
        "end_exclusive":
            "2019-01-01T00:00:00",
        "semantic_feature_order":
            list(
                SEMANTIC_FEATURE_ORDER
            ),
        "numeric_feature_order":
            list(
                NUMERIC_FEATURES
            ),
        "boolean_feature_order":
            list(
                BOOLEAN_FEATURES
            ),
        "categorical_feature_order":
            list(
                CATEGORICAL_FEATURES
            ),
        "category_codebook": {
            key:
                list(values)
            for key, values
            in CATEGORY_CODEBOOK.items()
        },
        "files": {},
    }

    for filename, array in arrays.items():
        manifest[
            "files"
        ][
            filename
        ] = {
            "sha256":
                sha256_file(
                    bundle
                    / filename
                ),
            "dtype":
                str(
                    array.dtype
                ),
            "shape":
                list(
                    array.shape
                ),
        }

    (
        bundle
        / MANIFEST_FILE
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return bundle


class TrainingBundleTests(
    unittest.TestCase
):
    def test_valid_bundle_loads_and_maps_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle_dir = write_bundle(
                Path(tmp)
            )

            bundle = load_training_bundle(
                bundle_dir,
                profile=
                    TEST_PROFILE,
            )

            self.assertEqual(
                bundle.row_count,
                24,
            )
            self.assertEqual(
                bundle.fraud_rows,
                4,
            )

            rows, timestamps = next(
                bundle.iter_semantic_batches(
                    10
                )
            )

            self.assertEqual(
                len(rows),
                10,
            )
            self.assertEqual(
                len(timestamps),
                10,
            )
            self.assertIsNone(
                rows[0]
                .time_since_previous_transaction_min
            )
            self.assertEqual(
                rows[1]
                .transaction_mode,
                "Online Transaction",
            )

    def test_hash_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle_dir = write_bundle(
                Path(tmp)
            )

            with (
                bundle_dir
                / TARGET_FILE
            ).open(
                "ab"
            ) as handle:
                handle.write(
                    b"x"
                )

            with self.assertRaises(
                ValueError
            ):
                load_training_bundle(
                    bundle_dir,
                    profile=
                        TEST_PROFILE,
                )

    def test_fraud_count_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle_dir = write_bundle(
                Path(tmp)
            )

            manifest_path = (
                bundle_dir
                / MANIFEST_FILE
            )

            manifest = json.loads(
                manifest_path.read_text(
                    encoding="utf-8"
                )
            )
            manifest[
                "fraud_rows"
            ] = 5

            manifest_path.write_text(
                json.dumps(
                    manifest
                ),
                encoding="utf-8",
            )

            with self.assertRaises(
                ValueError
            ):
                load_training_bundle(
                    bundle_dir,
                    profile=
                        TEST_PROFILE,
                )

    def test_research_location_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            research = (
                root
                / "research"
            )
            research.mkdir()

            bundle_dir = write_bundle(
                research
            )

            with self.assertRaises(
                ValueError
            ):
                load_training_bundle(
                    bundle_dir,
                    profile=
                        TEST_PROFILE,
                    project_root=
                        root,
                )

    def test_global_timestamp_reordering_is_allowed_when_hashes_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle_dir = write_bundle(
                root
            )

            timestamp_path = (
                bundle_dir
                / TIMESTAMP_FILE
            )

            values = np.load(
                timestamp_path,
                allow_pickle=False,
            )
            values = values.copy()
            values[
                5
            ], values[
                6
            ] = (
                values[
                    6
                ],
                values[
                    5
                ],
            )

            np.save(
                timestamp_path,
                values,
                allow_pickle=False,
            )

            manifest_path = (
                bundle_dir
                / MANIFEST_FILE
            )
            manifest = json.loads(
                manifest_path.read_text(
                    encoding="utf-8"
                )
            )
            manifest[
                "files"
            ][
                TIMESTAMP_FILE
            ][
                "sha256"
            ] = sha256_file(
                timestamp_path
            )

            manifest_path.write_text(
                json.dumps(
                    manifest
                ),
                encoding="utf-8",
            )

            bundle = load_training_bundle(
                bundle_dir,
                profile=
                    TEST_PROFILE,
            )

            self.assertEqual(
                bundle.row_count,
                24,
            )


if __name__ == "__main__":
    unittest.main()
