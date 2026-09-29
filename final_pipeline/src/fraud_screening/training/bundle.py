"""Validated canonical semantic input bundle for W_SHORT rebuild training."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Iterator

import numpy as np

from fraud_screening.errors import PreprocessingContractError
from fraud_screening.features import (
    BOOLEAN_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    SEMANTIC_FEATURE_ORDER,
    SemanticFeatureRow,
)


BUNDLE_VERSION = "1.0"
BUNDLE_ROLE = "W_SHORT_CANONICAL_SEMANTIC_TRAINING_INPUT"

NUMERIC_FILE = "semantic_numeric.npy"
BOOLEAN_FILE = "semantic_boolean.npy"
CATEGORICAL_FILE = "semantic_categorical.npy"
TIMESTAMP_FILE = "timestamps.npy"
TARGET_FILE = "target.npy"
MANIFEST_FILE = "bundle_manifest.json"

CATEGORY_CODEBOOK = {
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

_STRUCTURAL_NA_INDICES = {
    1,
    3,
}


@dataclass(frozen=True, slots=True)
class TrainingBundleProfile:
    """Expected identity of a canonical training population."""

    name: str
    row_count: int
    fraud_rows: int
    start_inclusive: np.datetime64
    end_exclusive: np.datetime64


CANONICAL_W_SHORT_PROFILE = TrainingBundleProfile(
    name="W_SHORT",
    row_count=1_721_615,
    fraud_rows=2_491,
    start_inclusive=np.datetime64(
        "2018-01-01T00:00:00",
        "ns",
    ),
    end_exclusive=np.datetime64(
        "2019-01-01T00:00:00",
        "ns",
    ),
)


def _sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def _expected_files() -> tuple[str, ...]:
    return (
        NUMERIC_FILE,
        BOOLEAN_FILE,
        CATEGORICAL_FILE,
        TIMESTAMP_FILE,
        TARGET_FILE,
    )


def _load_manifest(
    bundle_dir: Path,
) -> dict:
    path = (
        bundle_dir
        / MANIFEST_FILE
    )

    if not path.is_file():
        raise FileNotFoundError(
            f"Missing training bundle manifest: {path}"
        )

    try:
        manifest = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Training bundle manifest is not valid JSON."
        ) from exc

    if not isinstance(
        manifest,
        dict,
    ):
        raise ValueError(
            "Training bundle manifest root must be an object."
        )

    return manifest


def _guard_manifest_contract(
    manifest: dict,
    profile: TrainingBundleProfile,
) -> None:
    expected_top = {
        "bundle_version",
        "role",
        "profile",
        "row_count",
        "fraud_rows",
        "start_inclusive",
        "end_exclusive",
        "semantic_feature_order",
        "numeric_feature_order",
        "boolean_feature_order",
        "categorical_feature_order",
        "category_codebook",
        "files",
    }

    if set(
        manifest
    ) != expected_top:
        raise ValueError(
            "Training bundle manifest keys do not match the locked contract."
        )

    if (
        manifest[
            "bundle_version"
        ]
        != BUNDLE_VERSION
    ):
        raise ValueError(
            "Unsupported training bundle version."
        )

    if (
        manifest[
            "role"
        ]
        != BUNDLE_ROLE
    ):
        raise ValueError(
            "Training bundle role mismatch."
        )

    if (
        manifest[
            "profile"
        ]
        != profile.name
    ):
        raise ValueError(
            "Training bundle profile mismatch."
        )

    if (
        int(
            manifest[
                "row_count"
            ]
        )
        != profile.row_count
    ):
        raise ValueError(
            "Training bundle row count mismatch."
        )

    if (
        int(
            manifest[
                "fraud_rows"
            ]
        )
        != profile.fraud_rows
    ):
        raise ValueError(
            "Training bundle fraud count mismatch."
        )

    if (
        np.datetime64(
            manifest[
                "start_inclusive"
            ],
            "ns",
        )
        != profile.start_inclusive
    ):
        raise ValueError(
            "Training bundle start boundary mismatch."
        )

    if (
        np.datetime64(
            manifest[
                "end_exclusive"
            ],
            "ns",
        )
        != profile.end_exclusive
    ):
        raise ValueError(
            "Training bundle end boundary mismatch."
        )

    if (
        manifest[
            "semantic_feature_order"
        ]
        != list(
            SEMANTIC_FEATURE_ORDER
        )
    ):
        raise ValueError(
            "Semantic feature order mismatch."
        )

    if (
        manifest[
            "numeric_feature_order"
        ]
        != list(
            NUMERIC_FEATURES
        )
    ):
        raise ValueError(
            "Numeric feature order mismatch."
        )

    if (
        manifest[
            "boolean_feature_order"
        ]
        != list(
            BOOLEAN_FEATURES
        )
    ):
        raise ValueError(
            "Boolean feature order mismatch."
        )

    if (
        manifest[
            "categorical_feature_order"
        ]
        != list(
            CATEGORICAL_FEATURES
        )
    ):
        raise ValueError(
            "Categorical feature order mismatch."
        )

    expected_codebook = {
        key: list(values)
        for key, values
        in CATEGORY_CODEBOOK.items()
    }

    if (
        manifest[
            "category_codebook"
        ]
        != expected_codebook
    ):
        raise ValueError(
            "Categorical codebook mismatch."
        )

    files = manifest[
        "files"
    ]

    if not isinstance(
        files,
        dict,
    ):
        raise ValueError(
            "Training bundle files entry must be an object."
        )

    if set(
        files
    ) != set(
        _expected_files()
    ):
        raise ValueError(
            "Training bundle file registry mismatch."
        )

    for filename in _expected_files():
        entry = files[
            filename
        ]

        if not isinstance(
            entry,
            dict,
        ):
            raise ValueError(
                f"Invalid file registry entry: {filename}."
            )

        if set(
            entry
        ) != {
            "sha256",
            "dtype",
            "shape",
        }:
            raise ValueError(
                f"File registry keys mismatch: {filename}."
            )


@dataclass(frozen=True, slots=True)
class CanonicalTrainingBundle:
    """Memory-mapped canonical semantic training representation."""

    bundle_dir: Path
    manifest: dict
    profile: TrainingBundleProfile
    numeric: np.ndarray
    boolean: np.ndarray
    categorical: np.ndarray
    timestamps: np.ndarray
    target: np.ndarray

    @property
    def row_count(self) -> int:
        return int(
            self.target.shape[0]
        )

    @property
    def fraud_rows(self) -> int:
        return int(
            np.count_nonzero(
                self.target == 1
            )
        )

    def iter_semantic_batches(
        self,
        batch_size: int,
    ) -> Iterator[
        tuple[
            list[SemanticFeatureRow],
            list[datetime],
        ]
    ]:
        if (
            not isinstance(
                batch_size,
                int,
            )
            or batch_size <= 0
        ):
            raise ValueError(
                "batch_size must be a positive integer."
            )

        for start in range(
            0,
            self.row_count,
            batch_size,
        ):
            stop = min(
                start + batch_size,
                self.row_count,
            )

            numeric = self.numeric[
                start:stop
            ]
            boolean = self.boolean[
                start:stop
            ]
            categorical = self.categorical[
                start:stop
            ]

            timestamp_values = (
                self.timestamps[
                    start:stop
                ]
                .astype(
                    "datetime64[us]"
                )
                .astype(
                    datetime
                )
            )

            rows: list[
                SemanticFeatureRow
            ] = []

            for offset in range(
                stop - start
            ):
                category_values = {}

                for column_index, column in enumerate(
                    CATEGORICAL_FEATURES
                ):
                    code = int(
                        categorical[
                            offset,
                            column_index,
                        ]
                    )

                    category_values[
                        column
                    ] = (
                        CATEGORY_CODEBOOK[
                            column
                        ][
                            code
                        ]
                    )

                rows.append(
                    SemanticFeatureRow(
                        amount_numeric=float(
                            numeric[
                                offset,
                                0,
                            ]
                        ),
                        time_since_previous_transaction_min=(
                            None
                            if np.isnan(
                                numeric[
                                    offset,
                                    1,
                                ]
                            )
                            else float(
                                numeric[
                                    offset,
                                    1,
                                ]
                            )
                        ),
                        transactions_last_1h=int(
                            numeric[
                                offset,
                                2,
                            ]
                        ),
                        amount_minus_previous_mean=(
                            None
                            if np.isnan(
                                numeric[
                                    offset,
                                    3,
                                ]
                            )
                            else float(
                                numeric[
                                    offset,
                                    3,
                                ]
                            )
                        ),
                        is_new_merchant=bool(
                            boolean[
                                offset,
                                0,
                            ]
                        ),
                        has_prior_card_history=bool(
                            boolean[
                                offset,
                                1,
                            ]
                        ),
                        transaction_mode=
                            category_values[
                                "transaction_mode"
                            ],
                        location_state=
                            category_values[
                                "location_state"
                            ],
                        hour_of_day=
                            category_values[
                                "hour_of_day"
                            ],
                        day_of_week=
                            category_values[
                                "day_of_week"
                            ],
                    )
                )

            yield (
                rows,
                list(
                    timestamp_values
                ),
            )


def _guard_bundle_location(
    bundle_dir: Path,
    project_root: Path | None,
) -> None:
    if project_root is None:
        return

    research_root = (
        project_root
        / "research"
    ).resolve()

    try:
        bundle_dir.relative_to(
            research_root
        )
    except ValueError:
        return

    raise ValueError(
        "Final-pipeline rebuild input may not be loaded from research/."
    )


def load_training_bundle(
    bundle_dir: str | Path,
    *,
    profile: TrainingBundleProfile = CANONICAL_W_SHORT_PROFILE,
    project_root: str | Path | None = None,
) -> CanonicalTrainingBundle:
    """Load and fully validate a canonical semantic training bundle."""

    root = Path(
        bundle_dir
    ).resolve()

    if not root.is_dir():
        raise FileNotFoundError(
            f"Training bundle directory does not exist: {root}"
        )

    resolved_project_root = (
        None
        if project_root is None
        else Path(
            project_root
        ).resolve()
    )

    _guard_bundle_location(
        root,
        resolved_project_root,
    )

    manifest = _load_manifest(
        root
    )

    _guard_manifest_contract(
        manifest,
        profile,
    )

    for filename in _expected_files():
        path = root / filename

        if not path.is_file():
            raise FileNotFoundError(
                f"Training bundle file missing: {path}"
            )

        actual_sha = _sha256_file(
            path
        )

        expected_sha = (
            manifest[
                "files"
            ][
                filename
            ][
                "sha256"
            ]
        )

        if (
            actual_sha
            != expected_sha
        ):
            raise ValueError(
                f"Training bundle SHA-256 mismatch: {filename}."
            )

    numeric = np.load(
        root / NUMERIC_FILE,
        mmap_mode="r",
        allow_pickle=False,
    )
    boolean = np.load(
        root / BOOLEAN_FILE,
        mmap_mode="r",
        allow_pickle=False,
    )
    categorical = np.load(
        root / CATEGORICAL_FILE,
        mmap_mode="r",
        allow_pickle=False,
    )
    timestamps = np.load(
        root / TIMESTAMP_FILE,
        mmap_mode="r",
        allow_pickle=False,
    )
    target = np.load(
        root / TARGET_FILE,
        mmap_mode="r",
        allow_pickle=False,
    )

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

    expected_dtypes = {
        NUMERIC_FILE:
            "float64",
        BOOLEAN_FILE:
            "bool",
        CATEGORICAL_FILE:
            "int8",
        TIMESTAMP_FILE:
            "datetime64[ns]",
        TARGET_FILE:
            "int8",
    }

    expected_shapes = {
        NUMERIC_FILE:
            (
                profile.row_count,
                len(
                    NUMERIC_FEATURES
                ),
            ),
        BOOLEAN_FILE:
            (
                profile.row_count,
                len(
                    BOOLEAN_FEATURES
                ),
            ),
        CATEGORICAL_FILE:
            (
                profile.row_count,
                len(
                    CATEGORICAL_FEATURES
                ),
            ),
        TIMESTAMP_FILE:
            (
                profile.row_count,
            ),
        TARGET_FILE:
            (
                profile.row_count,
            ),
    }

    for filename, array in arrays.items():
        expected_dtype = (
            expected_dtypes[
                filename
            ]
        )

        if (
            str(
                array.dtype
            )
            != expected_dtype
        ):
            raise ValueError(
                f"Training bundle dtype mismatch: {filename}."
            )

        if (
            tuple(
                array.shape
            )
            != expected_shapes[
                filename
            ]
        ):
            raise ValueError(
                f"Training bundle shape mismatch: {filename}."
            )

        registry = (
            manifest[
                "files"
            ][
                filename
            ]
        )

        if (
            registry[
                "dtype"
            ]
            != expected_dtype
        ):
            raise ValueError(
                f"Manifest dtype mismatch: {filename}."
            )

        if (
            registry[
                "shape"
            ]
            != list(
                expected_shapes[
                    filename
                ]
            )
        ):
            raise ValueError(
                f"Manifest shape mismatch: {filename}."
            )

    if (
        np.isinf(
            numeric
        ).any()
    ):
        raise PreprocessingContractError(
            "Semantic numeric bundle contains infinity."
        )

    for column_index in range(
        numeric.shape[1]
    ):
        missing = np.isnan(
            numeric[
                :,
                column_index,
            ]
        )

        if (
            column_index
            not in _STRUCTURAL_NA_INDICES
            and missing.any()
        ):
            raise PreprocessingContractError(
                "Unexpected missing numeric value in canonical bundle."
            )

    for column_index, column in enumerate(
        CATEGORICAL_FEATURES
    ):
        codes = categorical[
            :,
            column_index,
        ]

        if (
            (
                codes < 0
            ).any()
            or (
                codes
                >= len(
                    CATEGORY_CODEBOOK[
                        column
                    ]
                )
            ).any()
        ):
            raise ValueError(
                f"Categorical code out of range: {column}."
            )

    unique_target = np.unique(
        target
    )

    if (
        unique_target.tolist()
        != [0, 1]
    ):
        raise ValueError(
            "Canonical training target must contain exactly classes 0 and 1."
        )

    actual_fraud_rows = int(
        np.count_nonzero(
            target == 1
        )
    )

    if (
        actual_fraud_rows
        != profile.fraud_rows
    ):
        raise ValueError(
            "Canonical training fraud-row count mismatch."
        )

    if (
        np.isnat(
            timestamps
        ).any()
    ):
        raise ValueError(
            "Canonical training timestamp contains NaT."
        )

    observed_min = timestamps.min()
    observed_max = timestamps.max()

    if (
        observed_min
        < profile.start_inclusive
    ):
        raise ValueError(
            "Canonical training bundle starts before W_SHORT."
        )

    if (
        observed_max
        >= profile.end_exclusive
    ):
        raise ValueError(
            "Canonical training bundle crosses TRAIN_END."
        )

    # Global timestamp sorting is intentionally NOT required here.
    #
    # A canonical semantic bundle may preserve verified contiguous
    # User+Card replay order. History causality and within-card ordering
    # belong to bundle production. Per-file SHA-256 values lock row order.

    return CanonicalTrainingBundle(
        bundle_dir=root,
        manifest=manifest,
        profile=profile,
        numeric=numeric,
        boolean=boolean,
        categorical=categorical,
        timestamps=timestamps,
        target=target,
    )
