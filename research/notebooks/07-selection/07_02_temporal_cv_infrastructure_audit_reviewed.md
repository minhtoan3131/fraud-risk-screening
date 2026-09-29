# M7.2 — Artifact audit + fold-safe temporal-CV infrastructure

Milestone:

`M7 — MODEL SELECTION / ROBUSTNESS / TUNING`

Substep:

`M7.2 — Artifact audit + fold-safe temporal-CV infrastructure`

Work type:

`RUNTIME INFRASTRUCTURE AUDIT`

Primary question:

> Làm thế nào tái sử dụng canonical M4/M5 project assets để tạo temporal-CV folds mà không dùng preprocessing state học từ tương lai?

M7.2 được phép:

- audit M4.7/M6 handoff artifacts;
- rebuild row-to-Timestamp lineage;
- verify Q2/Q3/Q4-2018 temporal folds;
- rebuild strict-causal semantic features;
- fit fold-local scaler/vocabulary từ `fold-TRAIN only`;
- transform fold-validation;
- verify canonical 47-column output schema;
- verify sampler insertion point;
- define shared M7 fold runner;
- unit-test metric/runner contract;
- persist infrastructure contract.

M7.2 không được:

- chọn W_SHORT/W_LONG;
- shortlist LR/DT/RF;
- chạy model-selection comparison;
- tuning;
- imbalance selection;
- threshold optimization;
- access FINAL TEST.

Runtime-dependent state trước khi chạy:

`NOT YET VERIFIED`

## 1. CANON contract

Primary temporal CV:

```text
FORWARD / EXPANDING

Fold 1
train < 2018-04-01
validation 2018-04-01 → 2018-07-01

Fold 2
train < 2018-07-01
validation 2018-07-01 → 2018-10-01

Fold 3
train < 2018-10-01
validation 2018-10-01 → 2019-01-01
```

Training-window starts:

```text
W_LONG:
2015-01-01

W_SHORT:
2018-01-01
```

Required preprocessing rule:

`LEARNED PREPROCESSING STATE = FOLD-TRAIN ONLY`

Fold-validation:

`TRANSFORM ONLY`

Sampler:

`FOLD-TRAIN ONLY`

FINAL TEST:

`PROTECTED`

## 2. Vì sao M4.7 full-year matrices không được dùng trực tiếp cho temporal CV?

M4.7 baseline matrices là canonical cho M5/M6 baseline experiment.

Nhưng M4.7 preprocessing state được fit trên toàn classifier TRAIN tương ứng tới:

`2018-12-31`.

Trong temporal CV:

```text
Fold 1 train end:
2018-04-01

Fold 2 train end:
2018-07-01

Fold 3 train end:
2018-10-01
```

Nếu slice trực tiếp `X_train_w_long.npz` hoặc `X_train_w_short.npz` thành CV folds thì scaler/vocabulary đã nhìn thấy rows nằm sau fold-training cutoff.

Đó là:

`PREPROCESSING LEAKAGE FOR TEMPORAL CV`.

Vì vậy M7.2 phải dùng canonical semantic contract để **refit preprocessing per fold**.

M4.7 transformed matrices vẫn:

`VALID CANONICAL BASELINE ARTIFACTS`

nhưng:

`NOT SAFE AS DIRECT TEMPORAL-CV INPUT MATRICES`.


```python

from pathlib import Path
from collections import Counter
from dataclasses import dataclass
import copy
import hashlib
import json
import platform
import sys
import time
import warnings

import numpy as np
import pandas as pd

from scipy import sparse

from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    clone,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)


print("Python:")
print(sys.version)

print("\nExecutable:")
print(sys.executable)

print("\nPlatform:")
print(platform.platform())

print("\nNumPy:")
print(np.__version__)

print("\npandas:")
print(pd.__version__)

```

    Python:
    3.14.6 (main, Jun 10 2026, 10:03:53) [Clang 21.0.0 (clang-2100.0.123.102)]
    
    Executable:
    /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/.venv/bin/python
    
    Platform:
    macOS-26.6.2-arm64-arm-64bit-Mach-O
    
    NumPy:
    2.5.3
    
    pandas:
    3.0.5


## 3. Locate project artifacts


```python

RAW_REL = (
    Path("data")
    / "raw"
    / "ibm_tabformer"
    / "card_transaction.v1.csv"
)

M4_REL = (
    Path("data")
    / "processed"
    / "m4_07_baseline_ready"
)

M6_02_REL = (
    Path("data")
    / "processed"
    / "m6_02_evaluation_artifact_audit"
)

M7_02_REL = (
    Path("data")
    / "processed"
    / "m7_02_temporal_cv_infrastructure_audit"
)


candidate_roots = [
    Path.cwd(),
    *list(Path.cwd().parents)[:6],
]


PROJECT_ROOT = None

required_rel_paths = [
    RAW_REL,

    M4_REL / "manifest.json",
    M4_REL / "feature_names.json",

    M4_REL / "y_train_w_long.npy",
    M4_REL / "y_train_w_short.npy",

    M4_REL / "row_id_train_w_long.npy",
    M4_REL / "row_id_train_w_short.npy",

    M6_02_REL
    / "m6_02_evaluation_registry.json",
]


for candidate in candidate_roots:
    candidate = candidate.resolve()

    if all(
        (
            candidate
            / rel_path
        ).exists()
        for rel_path
        in required_rel_paths
    ):
        PROJECT_ROOT = candidate
        break


if PROJECT_ROOT is None:
    raise FileNotFoundError(
        "Không tìm thấy PROJECT_ROOT chứa đầy đủ "
        "raw + M4.7 + M6.2 artifacts."
    )


RAW_PATH = PROJECT_ROOT / RAW_REL
M4_DIR = PROJECT_ROOT / M4_REL
M6_02_DIR = PROJECT_ROOT / M6_02_REL
OUTPUT_DIR = PROJECT_ROOT / M7_02_REL

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


M4_MANIFEST_PATH = (
    M4_DIR
    / "manifest.json"
)

M4_FEATURE_NAMES_PATH = (
    M4_DIR
    / "feature_names.json"
)

M6_02_REGISTRY_PATH = (
    M6_02_DIR
    / "m6_02_evaluation_registry.json"
)


EXPECTED_RAW_FILE_SIZE = 2_354_626_737
EXPECTED_RAW_ROWS = 24_386_900
EXPECTED_CARD_COUNT = 6_139

EXPECTED_W_LONG_ROWS = 6_855_270
EXPECTED_W_SHORT_ROWS = 1_721_615

EXPECTED_W_LONG_FRAUD = 9_606
EXPECTED_W_SHORT_FRAUD = 2_491

EXPECTED_OUTPUT_WIDTH = 47

PIPELINE_VERSION = "M4.7-baseline-v1"

M7_RUNNER_VERSION = (
    "M7.2-shared-temporal-runner-v1"
)

M7_CV_SPEC_ID = (
    "M7-TEMPORAL-CV-Q2-Q3-Q4-2018-v1"
)

M7_FOLD_TEMPLATE_ID = (
    "EXPANDING-Q2-Q3-Q4-2018-v1"
)

M7_METRIC_CONTRACT_VERSION = (
    "M7.1-metric-contract-v1"
)

RANDOM_STATE = 42

CHUNK_SIZE = 500_000

# Khoảng 0.1% rows cho transform smoke audit.
AUDIT_SAMPLE_MOD = 997


assert RAW_PATH.exists()

assert (
    RAW_PATH.stat().st_size
    == EXPECTED_RAW_FILE_SIZE
)


print("PROJECT_ROOT:")
print(PROJECT_ROOT)

print("\nRAW_PATH:")
print(RAW_PATH)

print("\nOUTPUT_DIR:")
print(OUTPUT_DIR)

print(
    "\nM7.2 SOURCE LOCATION GATE: PASS"
)

```

    PROJECT_ROOT:
    /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening
    
    RAW_PATH:
    /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/raw/ibm_tabformer/card_transaction.v1.csv
    
    OUTPUT_DIR:
    /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/processed/m7_02_temporal_cv_infrastructure_audit
    
    M7.2 SOURCE LOCATION GATE: PASS


## 4. Load and audit canonical M4.7 / M6.2 handoff


```python

def sha256_file(
    path,
    chunk_size=1024 * 1024,
):
    digest = hashlib.sha256()

    with open(
        path,
        "rb",
    ) as file:
        while True:
            chunk = file.read(
                chunk_size
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


with open(
    M4_MANIFEST_PATH,
    "r",
    encoding="utf-8",
) as file:
    m4_manifest = json.load(
        file
    )


with open(
    M4_FEATURE_NAMES_PATH,
    "r",
    encoding="utf-8",
) as file:
    canonical_feature_names = json.load(
        file
    )


with open(
    M6_02_REGISTRY_PATH,
    "r",
    encoding="utf-8",
) as file:
    m6_02_registry = json.load(
        file
    )


assert (
    m4_manifest[
        "pipeline_version"
    ]
    == PIPELINE_VERSION
)

assert (
    m4_manifest[
        "output_width"
    ]
    == EXPECTED_OUTPUT_WIDTH
)

assert (
    m4_manifest[
        "matrix_format"
    ]
    == "CSR"
)

assert (
    m4_manifest[
        "matrix_dtype"
    ]
    == "float32"
)

assert (
    m4_manifest[
        "raw_identifiers_in_X"
    ]
    is False
)

assert (
    m4_manifest[
        "target_in_X"
    ]
    is False
)

assert (
    m4_manifest[
        "final_test_used"
    ]
    is False
)

assert len(
    canonical_feature_names
) == EXPECTED_OUTPUT_WIDTH


assert (
    m6_02_registry[
        "official_run_count"
    ]
    == 6
)

assert (
    m6_02_registry[
        "final_test_accessed"
    ]
    is False
)


print(
    "M4 pipeline:",
    m4_manifest[
        "pipeline_version"
    ],
)

print(
    "Canonical output width:",
    len(
        canonical_feature_names
    ),
)

print(
    "M6 official runs:",
    m6_02_registry[
        "official_run_count"
    ],
)

print(
    "\nM7.2 M4/M6 HANDOFF GATE: PASS"
)

```

    M4 pipeline: M4.7-baseline-v1
    Canonical output width: 47
    M6 official runs: 6
    
    M7.2 M4/M6 HANDOFF GATE: PASS


## 5. Load canonical TRAIN target / lineage arrays


```python

Y_LONG_PATH = (
    M4_DIR
    / "y_train_w_long.npy"
)

Y_SHORT_PATH = (
    M4_DIR
    / "y_train_w_short.npy"
)

ROW_LONG_PATH = (
    M4_DIR
    / "row_id_train_w_long.npy"
)

ROW_SHORT_PATH = (
    M4_DIR
    / "row_id_train_w_short.npy"
)


y_long = np.load(
    Y_LONG_PATH,
    allow_pickle=False,
)

y_short = np.load(
    Y_SHORT_PATH,
    allow_pickle=False,
)

row_long = np.load(
    ROW_LONG_PATH,
    allow_pickle=False,
)

row_short = np.load(
    ROW_SHORT_PATH,
    allow_pickle=False,
)


assert y_long.ndim == 1
assert y_short.ndim == 1
assert row_long.ndim == 1
assert row_short.ndim == 1

assert len(
    y_long
) == EXPECTED_W_LONG_ROWS

assert len(
    row_long
) == EXPECTED_W_LONG_ROWS

assert len(
    y_short
) == EXPECTED_W_SHORT_ROWS

assert len(
    row_short
) == EXPECTED_W_SHORT_ROWS

assert int(
    y_long.sum()
) == EXPECTED_W_LONG_FRAUD

assert int(
    y_short.sum()
) == EXPECTED_W_SHORT_FRAUD

assert np.all(
    np.diff(
        row_long
    ) > 0
)

assert np.all(
    np.diff(
        row_short
    ) > 0
)


short_positions_in_long = (
    np.searchsorted(
        row_long,
        row_short,
    )
)

assert np.all(
    short_positions_in_long
    < len(
        row_long
    )
)

np.testing.assert_array_equal(
    row_long[
        short_positions_in_long
    ],
    row_short,
)

np.testing.assert_array_equal(
    y_long[
        short_positions_in_long
    ],
    y_short,
)


print(
    "W_LONG rows / fraud:",
    len(
        row_long
    ),
    "/",
    int(
        y_long.sum()
    ),
)

print(
    "W_SHORT rows / fraud:",
    len(
        row_short
    ),
    "/",
    int(
        y_short.sum()
    ),
)

print(
    "\nM7.2 TRAIN LINEAGE GATE: PASS"
)

```

    W_LONG rows / fraud: 6855270 / 9606
    W_SHORT rows / fraud: 1721615 / 2491
    
    M7.2 TRAIN LINEAGE GATE: PASS


## 6. Reject direct reuse of M4.7 full-year transformed matrices for CV

M4.7 manifest phải cho thấy learned preprocessing fit tới cuối 2018.

Nếu vậy:

```text
M4.7 X_train_*:
valid baseline artifacts

direct temporal-CV slicing:
PROHIBITED
```


```python

m4_long_fit_end = pd.Timestamp(
    m4_manifest[
        "W_LONG"
    ][
        "fit_end_observed"
    ]
)

m4_short_fit_end = pd.Timestamp(
    m4_manifest[
        "W_SHORT"
    ][
        "fit_end_observed"
    ]
)


assert (
    m4_long_fit_end
    >= pd.Timestamp(
        "2018-12-31"
    )
)

assert (
    m4_short_fit_end
    >= pd.Timestamp(
        "2018-12-31"
    )
)


M4_FULL_YEAR_MATRICES_SAFE_FOR_DIRECT_CV = (
    False
)


assert (
    M4_FULL_YEAR_MATRICES_SAFE_FOR_DIRECT_CV
    is False
)


print(
    "M4.7 W_LONG preprocessing fit end:",
    m4_long_fit_end,
)

print(
    "M4.7 W_SHORT preprocessing fit end:",
    m4_short_fit_end,
)

print(
    "\nDirect slicing of M4.7 transformed "
    "TRAIN matrices for temporal CV:"
)

print(
    "PROHIBITED — PREPROCESSING STATE "
    "SEES FUTURE FOLD ROWS"
)

print(
    "\nM7.2 FULL-YEAR MATRIX CV-SAFETY GATE: PASS"
)

```

    M4.7 W_LONG preprocessing fit end: 2018-12-31 23:58:00
    M4.7 W_SHORT preprocessing fit end: 2018-12-31 23:58:00
    
    Direct slicing of M4.7 transformed TRAIN matrices for temporal CV:
    PROHIBITED — PREPROCESSING STATE SEES FUTURE FOLD ROWS
    
    M7.2 FULL-YEAR MATRIX CV-SAFETY GATE: PASS


## 7. Lock temporal fold definitions


```python

W_LONG_START = pd.Timestamp(
    "2015-01-01"
)

W_SHORT_START = pd.Timestamp(
    "2018-01-01"
)

TRAIN_END = pd.Timestamp(
    "2019-01-01"
)


FOLD_SPECS = [
    {
        "fold_id":
            "FOLD_Q2_2018",

        "train_end_exclusive":
            pd.Timestamp(
                "2018-04-01"
            ),

        "validation_start":
            pd.Timestamp(
                "2018-04-01"
            ),

        "validation_end_exclusive":
            pd.Timestamp(
                "2018-07-01"
            ),
    },

    {
        "fold_id":
            "FOLD_Q3_2018",

        "train_end_exclusive":
            pd.Timestamp(
                "2018-07-01"
            ),

        "validation_start":
            pd.Timestamp(
                "2018-07-01"
            ),

        "validation_end_exclusive":
            pd.Timestamp(
                "2018-10-01"
            ),
    },

    {
        "fold_id":
            "FOLD_Q4_2018",

        "train_end_exclusive":
            pd.Timestamp(
                "2018-10-01"
            ),

        "validation_start":
            pd.Timestamp(
                "2018-10-01"
            ),

        "validation_end_exclusive":
            pd.Timestamp(
                "2019-01-01"
            ),
    },
]


WINDOW_STARTS = {
    "W_LONG":
        W_LONG_START,

    "W_SHORT":
        W_SHORT_START,
}


assert len(
    FOLD_SPECS
) == 3


for fold in FOLD_SPECS:
    assert (
        fold[
            "train_end_exclusive"
        ]
        == fold[
            "validation_start"
        ]
    )

    assert (
        fold[
            "validation_start"
        ]
        < fold[
            "validation_end_exclusive"
        ]
    )


print(
    "CV spec:",
    M7_CV_SPEC_ID,
)

for fold in FOLD_SPECS:
    print(
        fold[
            "fold_id"
        ],
        "train end:",
        fold[
            "train_end_exclusive"
        ],
        "| validation:",
        fold[
            "validation_start"
        ],
        "→",
        fold[
            "validation_end_exclusive"
        ],
    )


print(
    "\nM7.2 FOLD SPECIFICATION GATE: PASS"
)

```

    CV spec: M7-TEMPORAL-CV-Q2-Q3-Q4-2018-v1
    FOLD_Q2_2018 train end: 2018-04-01 00:00:00 | validation: 2018-04-01 00:00:00 → 2018-07-01 00:00:00
    FOLD_Q3_2018 train end: 2018-07-01 00:00:00 | validation: 2018-07-01 00:00:00 → 2018-10-01 00:00:00
    FOLD_Q4_2018 train end: 2018-10-01 00:00:00 | validation: 2018-10-01 00:00:00 → 2019-01-01 00:00:00
    
    M7.2 FOLD SPECIFICATION GATE: PASS


## 8. Exact M4 semantic contract

Core pre-encoding features:

```text
Numeric:
amount_numeric
time_since_previous_transaction_min
transactions_last_1h
amount_minus_previous_mean

Boolean:
is_new_merchant
has_prior_card_history

Categorical:
transaction_mode
location_state
hour_of_day
day_of_week
```

Prohibited direct classifier columns:

```text
User
Card
Merchant Name
Errors?
Is Fraud?
raw_row_id
Timestamp
```


```python

NUMERIC_COLUMNS = [
    "amount_numeric",
    "time_since_previous_transaction_min",
    "transactions_last_1h",
    "amount_minus_previous_mean",
]

BOOLEAN_COLUMNS = [
    "is_new_merchant",
    "has_prior_card_history",
]

CATEGORICAL_COLUMNS = [
    "transaction_mode",
    "location_state",
    "hour_of_day",
    "day_of_week",
]

CORE_FEATURE_COLUMNS = (
    NUMERIC_COLUMNS
    + BOOLEAN_COLUMNS
    + CATEGORICAL_COLUMNS
)

PROHIBITED_DIRECT_COLUMNS = {
    "User",
    "Card",
    "Merchant Name",
    "Errors?",
    "Is Fraud?",
    "raw_row_id",
    "Timestamp",
}

UNKNOWN_TOKEN = "__UNKNOWN__"


assert len(
    CORE_FEATURE_COLUMNS
) == 10

assert set(
    CORE_FEATURE_COLUMNS
).isdisjoint(
    PROHIBITED_DIRECT_COLUMNS
)


print(
    "Core semantic feature count:",
    len(
        CORE_FEATURE_COLUMNS
    ),
)

print(
    "M7.2 FEATURE CONTRACT GATE: PASS"
)

```

    Core semantic feature count: 10
    M7.2 FEATURE CONTRACT GATE: PASS


## 9. Raw deterministic representation helpers


```python

def parse_amount(
    amount_series,
):
    cleaned = (
        amount_series
        .astype("string")
        .str.replace(
            "$",
            "",
            regex=False,
        )
        .str.replace(
            ",",
            "",
            regex=False,
        )
        .str.strip()
    )

    return pd.to_numeric(
        cleaned,
        errors="coerce",
    )


def build_timestamp(
    df,
):
    date_part = pd.to_datetime(
        {
            "year":
                df["Year"],

            "month":
                df["Month"],

            "day":
                df["Day"],
        },
        errors="coerce",
    )

    time_part = pd.to_timedelta(
        (
            df["Time"]
            .astype("string")
            + ":00"
        ),
        errors="coerce",
    )

    return (
        date_part
        + time_part
    )


def normalize_string(
    series,
):
    return (
        series
        .astype("string")
        .str.strip()
    )


def assign_location_state(
    df,
):
    city = normalize_string(
        df[
            "Merchant City"
        ]
    )

    city_online = (
        city
        .eq("ONLINE")
        .fillna(False)
    )

    state_missing = (
        df[
            "Merchant State"
        ]
        .isna()
    )

    zip_missing = (
        df["Zip"]
        .isna()
    )

    nonphysical = (
        city_online
        & state_missing
        & zip_missing
    )

    physical_complete = (
        (~city_online)
        & (~state_missing)
        & (~zip_missing)
    )

    physical_zip_unavailable = (
        (~city_online)
        & (~state_missing)
        & zip_missing
    )

    result = np.select(
        [
            nonphysical,
            physical_complete,
            physical_zip_unavailable,
        ],
        [
            "NON_PHYSICAL_OR_ONLINE",
            "PHYSICAL_COMPLETE",
            "PHYSICAL_ZIP_UNAVAILABLE",
        ],
        default="OTHER_INCONSISTENT",
    )

    return pd.Series(
        result,
        index=df.index,
        dtype="string",
    )


print(
    "M7.2 DETERMINISTIC REPRESENTATION HELPERS: DEFINED"
)

```

    M7.2 DETERMINISTIC REPRESENTATION HELPERS: DEFINED


## 10. Card-block streaming helper


```python

SEMANTIC_USECOLS = [
    "User",
    "Card",
    "Year",
    "Month",
    "Day",
    "Time",
    "Amount",
    "Use Chip",
    "Merchant Name",
    "Merchant City",
    "Merchant State",
    "Zip",
]


assert (
    "Is Fraud?"
    not in SEMANTIC_USECOLS
)

assert (
    "Errors?"
    not in SEMANTIC_USECOLS
)


def iter_card_blocks(
    data_path,
    chunksize,
    usecols,
):
    pending_key = None
    pending_parts = []

    raw_row_offset = 0

    for chunk in pd.read_csv(
        data_path,
        usecols=usecols,
        chunksize=chunksize,
    ):
        chunk_length = len(
            chunk
        )

        chunk[
            "raw_row_id"
        ] = np.arange(
            raw_row_offset,
            raw_row_offset
            + chunk_length,
            dtype=np.int64,
        )

        raw_row_offset += (
            chunk_length
        )

        chunk[
            "Timestamp"
        ] = build_timestamp(
            chunk
        )

        chunk[
            "Amount_numeric"
        ] = parse_amount(
            chunk[
                "Amount"
            ]
        )

        if (
            chunk[
                "Timestamp"
            ]
            .isna()
            .any()
        ):
            raise ValueError(
                "Timestamp parse failure."
            )

        if (
            chunk[
                "Amount_numeric"
            ]
            .isna()
            .any()
        ):
            raise ValueError(
                "Amount parse failure."
            )

        if (
            chunk[
                "Merchant Name"
            ]
            .isna()
            .any()
        ):
            raise ValueError(
                "Merchant Name missing."
            )

        working = chunk[
            [
                "raw_row_id",
                "User",
                "Card",
                "Timestamp",
                "Amount_numeric",
                "Use Chip",
                "Merchant Name",
                "Merchant City",
                "Merchant State",
                "Zip",
            ]
        ]

        users = (
            working[
                "User"
            ]
            .to_numpy()
        )

        cards = (
            working[
                "Card"
            ]
            .to_numpy()
        )

        change_positions = (
            np.flatnonzero(
                (
                    users[1:]
                    != users[:-1]
                )
                |
                (
                    cards[1:]
                    != cards[:-1]
                )
            )
            + 1
        )

        starts = np.concatenate(
            [
                [0],
                change_positions,
            ]
        )

        ends = np.concatenate(
            [
                change_positions,
                [len(working)],
            ]
        )

        for start, end in zip(
            starts,
            ends,
        ):
            key = (
                int(
                    users[start]
                ),
                int(
                    cards[start]
                ),
            )

            segment = (
                working
                .iloc[
                    start:end
                ]
                .copy()
            )

            if pending_key is None:
                pending_key = key
                pending_parts = [
                    segment
                ]
                continue

            if key == pending_key:
                pending_parts.append(
                    segment
                )
                continue

            card_block = (
                pending_parts[0]
                .reset_index(
                    drop=True
                )
                if len(
                    pending_parts
                )
                == 1
                else pd.concat(
                    pending_parts,
                    ignore_index=True,
                )
            )

            yield (
                pending_key,
                card_block,
            )

            pending_key = key
            pending_parts = [
                segment
            ]

    if (
        pending_key
        is not None
    ):
        card_block = (
            pending_parts[0]
            .reset_index(
                drop=True
            )
            if len(
                pending_parts
            )
            == 1
            else pd.concat(
                pending_parts,
                ignore_index=True,
            )
        )

        yield (
            pending_key,
            card_block,
        )


print(
    "M7.2 CARD STREAMING HELPER: DEFINED"
)

```

    M7.2 CARD STREAMING HELPER: DEFINED


## 11. Strict-causal behavioral feature builder


```python

ONE_HOUR_NS = int(
    pd.Timedelta(
        hours=1
    ).value
)


def compute_causal_behavioral_features(
    card_df,
):
    timestamp_ns = (
        card_df[
            "Timestamp"
        ]
        .to_numpy(
            dtype="datetime64[ns]"
        )
        .astype(
            "int64"
        )
    )

    if (
        np.diff(
            timestamp_ns
        )
        < 0
    ).any():
        raise ValueError(
            "Timestamp giảm trong Card block."
        )

    group_starts = np.concatenate(
        [
            [0],
            (
                np.flatnonzero(
                    timestamp_ns[1:]
                    != timestamp_ns[:-1]
                )
                + 1
            ),
        ]
    )

    group_ends = np.concatenate(
        [
            group_starts[1:],
            [len(card_df)],
        ]
    )

    group_lengths = (
        group_ends
        - group_starts
    )

    group_timestamp_ns = (
        timestamp_ns[
            group_starts
        ]
    )

    prior_count = np.repeat(
        group_starts,
        group_lengths,
    )

    has_prior_card_history = (
        prior_count
        > 0
    )

    time_since_group = np.full(
        len(
            group_starts
        ),
        np.nan,
        dtype="float64",
    )

    if (
        len(
            group_starts
        )
        > 1
    ):
        time_since_group[
            1:
        ] = (
            (
                group_timestamp_ns[
                    1:
                ]
                -
                group_timestamp_ns[
                    :-1
                ]
            )
            / 60_000_000_000
        )

    time_since_previous_min = (
        np.repeat(
            time_since_group,
            group_lengths,
        )
    )

    left_1h = np.searchsorted(
        timestamp_ns,
        (
            group_timestamp_ns
            - ONE_HOUR_NS
        ),
        side="left",
    )

    transactions_last_1h_group = (
        group_starts
        - left_1h
    )

    transactions_last_1h = np.repeat(
        transactions_last_1h_group,
        group_lengths,
    )

    amount = (
        card_df[
            "Amount_numeric"
        ]
        .to_numpy(
            dtype="float64"
        )
    )

    amount_prefix_sum = np.concatenate(
        [
            [0.0],
            np.cumsum(
                amount
            ),
        ]
    )

    previous_amount_mean_group = np.full(
        len(
            group_starts
        ),
        np.nan,
        dtype="float64",
    )

    has_prior_group = (
        group_starts
        > 0
    )

    previous_amount_mean_group[
        has_prior_group
    ] = (
        amount_prefix_sum[
            group_starts[
                has_prior_group
            ]
        ]
        /
        group_starts[
            has_prior_group
        ]
    )

    previous_amount_mean = np.repeat(
        previous_amount_mean_group,
        group_lengths,
    )

    amount_minus_previous_mean = (
        amount
        - previous_amount_mean
    )

    merchant_values = (
        card_df[
            "Merchant Name"
        ]
        .to_numpy()
    )

    is_new_merchant = np.empty(
        len(
            card_df
        ),
        dtype=bool,
    )

    seen_merchants = set()

    for start, end in zip(
        group_starts,
        group_ends,
    ):
        current_group_merchants = (
            merchant_values[
                start:end
            ]
        )

        is_new_merchant[
            start:end
        ] = np.array(
            [
                merchant
                not in seen_merchants
                for merchant
                in current_group_merchants
            ],
            dtype=bool,
        )

        seen_merchants.update(
            current_group_merchants
            .tolist()
        )

    return pd.DataFrame(
        {
            "raw_row_id":
                card_df[
                    "raw_row_id"
                ].to_numpy(),

            "Timestamp":
                card_df[
                    "Timestamp"
                ].to_numpy(),

            "amount_numeric":
                amount,

            "time_since_previous_transaction_min":
                time_since_previous_min,

            "transactions_last_1h":
                transactions_last_1h,

            "amount_minus_previous_mean":
                amount_minus_previous_mean,

            "is_new_merchant":
                is_new_merchant,

            "has_prior_card_history":
                has_prior_card_history,
        }
    )


print(
    "M7.2 STRICT-CAUSAL FEATURE BUILDER: DEFINED"
)

```

    M7.2 STRICT-CAUSAL FEATURE BUILDER: DEFINED


## 12. Strict-causal unit test


```python

causal_test_df = pd.DataFrame(
    {
        "raw_row_id":
            [0, 1, 2, 3],

        "Timestamp":
            pd.to_datetime(
                [
                    "2018-01-01 09:00:00",
                    "2018-01-01 10:00:00",
                    "2018-01-01 10:00:00",
                    "2018-01-01 10:30:00",
                ]
            ),

        "Amount_numeric":
            [10.0, 20.0, 30.0, 40.0],

        "Merchant Name":
            [100, 200, 200, 100],
    }
)


causal_test = (
    compute_causal_behavioral_features(
        causal_test_df
    )
)


np.testing.assert_allclose(
    causal_test[
        "time_since_previous_transaction_min"
    ].to_numpy(),
    np.array(
        [
            np.nan,
            60.0,
            60.0,
            30.0,
        ]
    ),
    equal_nan=True,
)


np.testing.assert_array_equal(
    causal_test[
        "transactions_last_1h"
    ].to_numpy(),
    np.array(
        [
            0,
            1,
            1,
            2,
        ]
    ),
)


np.testing.assert_array_equal(
    causal_test[
        "is_new_merchant"
    ].to_numpy(),
    np.array(
        [
            True,
            True,
            True,
            False,
        ]
    ),
)


print(
    "M7.2 STRICT-CAUSAL REGRESSION GATE: PASS"
)

```

    M7.2 STRICT-CAUSAL REGRESSION GATE: PASS


## 13. Build canonical core feature frame


```python

def build_core_feature_frame(
    context_card_df,
    behavioral_df,
):
    timestamp = (
        context_card_df[
            "Timestamp"
        ]
    )

    frame = pd.DataFrame(
        {
            "raw_row_id":
                context_card_df[
                    "raw_row_id"
                ].to_numpy(),

            "Timestamp":
                timestamp.to_numpy(),

            "amount_numeric":
                behavioral_df[
                    "amount_numeric"
                ].to_numpy(),

            "time_since_previous_transaction_min":
                behavioral_df[
                    "time_since_previous_transaction_min"
                ].to_numpy(),

            "transactions_last_1h":
                behavioral_df[
                    "transactions_last_1h"
                ].to_numpy(),

            "amount_minus_previous_mean":
                behavioral_df[
                    "amount_minus_previous_mean"
                ].to_numpy(),

            "is_new_merchant":
                behavioral_df[
                    "is_new_merchant"
                ].to_numpy(),

            "has_prior_card_history":
                behavioral_df[
                    "has_prior_card_history"
                ].to_numpy(),

            "transaction_mode":
                normalize_string(
                    context_card_df[
                        "Use Chip"
                    ]
                ).to_numpy(),

            "location_state":
                assign_location_state(
                    context_card_df
                ).to_numpy(),

            "hour_of_day":
                (
                    timestamp
                    .dt.hour
                    .astype("Int8")
                    .astype("string")
                    .to_numpy()
                ),

            "day_of_week":
                (
                    timestamp
                    .dt.dayofweek
                    .astype("Int8")
                    .astype("string")
                    .to_numpy()
                ),
        }
    )

    allowed_na = {
        "time_since_previous_transaction_min",
        "amount_minus_previous_mean",
    }

    unexpected_missing = {
        column:
            int(
                frame[
                    column
                ]
                .isna()
                .sum()
            )
        for column
        in CORE_FEATURE_COLUMNS
        if (
            frame[
                column
            ]
            .isna()
            .any()
            and
            column
            not in allowed_na
        )
    }

    if unexpected_missing:
        raise ValueError(
            "Unexpected missing: "
            f"{unexpected_missing}"
        )

    if (
        frame[
            "location_state"
        ]
        .eq(
            "OTHER_INCONSISTENT"
        )
        .any()
    ):
        raise ValueError(
            "OTHER_INCONSISTENT location state."
        )

    return frame


print(
    "M7.2 CORE FEATURE FRAME HELPER: DEFINED"
)

```

    M7.2 CORE FEATURE FRAME HELPER: DEFINED


## 14. Fold-local preprocessing implementation

Contract:

```text
numeric:
FeaturewiseStandardScaler
fit only on finite fold-TRAIN values

boolean:
passthrough float32

categorical:
fold-TRAIN vocabulary
+ explicit __UNKNOWN__

validation:
transform only

output:
CSR float32
```


```python

def make_one_hot_encoder(
    categories,
):
    kwargs = dict(
        categories=categories,
        handle_unknown="error",
        dtype=np.float32,
    )

    try:
        return OneHotEncoder(
            sparse_output=True,
            **kwargs,
        )

    except TypeError:
        return OneHotEncoder(
            sparse=True,
            **kwargs,
        )


def stable_sort_categories(
    column,
    values,
):
    values = [
        str(
            value
        )
        for value
        in values
    ]

    if column in {
        "hour_of_day",
        "day_of_week",
    }:
        return sorted(
            values,
            key=lambda x:
                int(
                    x
                ),
        )

    return sorted(
        values
    )


class FeaturewiseStandardScaler:
    def __init__(
        self,
    ):
        self._scalers = None
        self.n_features_in_ = None
        self.mean_ = None
        self.var_ = None
        self.scale_ = None
        self.n_samples_seen_ = None

    def _initialize(
        self,
        n_features,
    ):
        self.n_features_in_ = int(
            n_features
        )

        self._scalers = [
            StandardScaler()
            for _ in range(
                n_features
            )
        ]

    def _sync_public_state(
        self,
    ):
        self.mean_ = np.array(
            [
                scaler.mean_[0]
                for scaler
                in self._scalers
            ],
            dtype=np.float64,
        )

        self.var_ = np.array(
            [
                scaler.var_[0]
                for scaler
                in self._scalers
            ],
            dtype=np.float64,
        )

        self.scale_ = np.array(
            [
                scaler.scale_[0]
                for scaler
                in self._scalers
            ],
            dtype=np.float64,
        )

        self.n_samples_seen_ = np.array(
            [
                int(
                    scaler.n_samples_seen_
                )
                for scaler
                in self._scalers
            ],
            dtype=np.int64,
        )

    def partial_fit(
        self,
        X,
    ):
        X = np.asarray(
            X,
            dtype=np.float64,
        )

        if X.ndim != 2:
            raise ValueError(
                "X numeric phải là 2D."
            )

        if self._scalers is None:
            self._initialize(
                X.shape[1]
            )

        if (
            X.shape[1]
            != self.n_features_in_
        ):
            raise ValueError(
                "Numeric width mismatch."
            )

        for index, scaler in enumerate(
            self._scalers
        ):
            values = X[
                :,
                index,
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

        # Chỉ sync khi mọi feature đã nhìn thấy ít nhất một observation.
        if all(
            hasattr(
                scaler,
                "mean_",
            )
            for scaler
            in self._scalers
        ):
            self._sync_public_state()

        return self

    def fit(
        self,
        X,
    ):
        return self.partial_fit(
            X
        )

    def transform(
        self,
        X,
    ):
        if self._scalers is None:
            raise RuntimeError(
                "Scaler chưa được fit."
            )

        X = np.asarray(
            X,
            dtype=np.float64,
        )

        if (
            X.ndim != 2
            or
            X.shape[1]
            != self.n_features_in_
        ):
            raise ValueError(
                "Numeric transform width mismatch."
            )

        transformed = np.full(
            X.shape,
            np.nan,
            dtype=np.float64,
        )

        for index, scaler in enumerate(
            self._scalers
        ):
            values = X[
                :,
                index,
            ]

            finite_mask = np.isfinite(
                values
            )

            if finite_mask.any():
                transformed[
                    finite_mask,
                    index,
                ] = (
                    scaler.transform(
                        values[
                            finite_mask
                        ].reshape(
                            -1,
                            1,
                        )
                    )
                    .ravel()
                )

        return transformed


@dataclass
class FoldPreprocessingBundle:
    state_id: str
    training_window_id: str
    fold_id: str
    scaler: FeaturewiseStandardScaler
    category_vocab: dict
    encoder: OneHotEncoder
    feature_names: list
    fit_row_count: int
    fit_min_timestamp: pd.Timestamp
    fit_max_timestamp: pd.Timestamp


def build_encoder_from_train_vocab(
    category_vocab,
):
    categories = []

    for column in (
        CATEGORICAL_COLUMNS
    ):
        train_categories = (
            stable_sort_categories(
                column,
                category_vocab[
                    column
                ],
            )
        )

        if (
            UNKNOWN_TOKEN
            in train_categories
        ):
            raise ValueError(
                "Reserved unknown token "
                "xuất hiện trong TRAIN."
            )

        categories.append(
            train_categories
            + [
                UNKNOWN_TOKEN
            ]
        )

    encoder = make_one_hot_encoder(
        categories=categories
    )

    max_len = max(
        len(
            values
        )
        for values
        in categories
    )

    synthetic = {}

    for column, values in zip(
        CATEGORICAL_COLUMNS,
        categories,
    ):
        synthetic[
            column
        ] = [
            values[
                index
                % len(
                    values
                )
            ]
            for index
            in range(
                max_len
            )
        ]

    encoder.fit(
        pd.DataFrame(
            synthetic
        )
    )

    return encoder


def map_unknown_categories(
    frame,
    bundle,
):
    mapped = pd.DataFrame(
        index=frame.index
    )

    for column in (
        CATEGORICAL_COLUMNS
    ):
        values = (
            frame[
                column
            ]
            .astype(
                "string"
            )
        )

        if (
            values
            .isna()
            .any()
        ):
            raise ValueError(
                "Unexpected missing categorical: "
                f"{column}"
            )

        known = set(
            bundle
            .category_vocab[
                column
            ]
        )

        mapped[
            column
        ] = values.where(
            values.isin(
                known
            ),
            UNKNOWN_TOKEN,
        )

    return mapped


def build_feature_names(
    encoder,
):
    categorical_names = [
        f"cat__{name}"
        for name
        in encoder.get_feature_names_out(
            CATEGORICAL_COLUMNS
        )
    ]

    return (
        [
            f"num__{name}"
            for name
            in NUMERIC_COLUMNS
        ]
        +
        [
            f"bool__{name}"
            for name
            in BOOLEAN_COLUMNS
        ]
        +
        categorical_names
    )


def transform_with_bundle(
    frame,
    bundle,
):
    numeric = (
        frame[
            NUMERIC_COLUMNS
        ]
        .astype(
            "float64"
        )
        .to_numpy()
    )

    numeric_scaled = (
        bundle.scaler
        .transform(
            numeric
        )
    )

    numeric_scaled = np.nan_to_num(
        numeric_scaled,
        nan=0.0,
        posinf=np.inf,
        neginf=-np.inf,
    ).astype(
        np.float32
    )

    boolean_matrix = (
        frame[
            BOOLEAN_COLUMNS
        ]
        .astype(
            np.float32
        )
        .to_numpy()
    )

    mapped_categorical = (
        map_unknown_categories(
            frame,
            bundle,
        )
    )

    categorical_matrix = (
        bundle.encoder
        .transform(
            mapped_categorical
        )
    )

    X = sparse.hstack(
        [
            sparse.csr_matrix(
                numeric_scaled,
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

    if (
        X.shape[1]
        != len(
            bundle.feature_names
        )
    ):
        raise AssertionError(
            "Feature width mismatch."
        )

    if not np.isfinite(
        X.data
    ).all():
        raise AssertionError(
            "NaN/inf sau preprocessing."
        )

    return X


print(
    "M7.2 FOLD-LOCAL PREPROCESSING HELPERS: DEFINED"
)

```

    M7.2 FOLD-LOCAL PREPROCESSING HELPERS: DEFINED


## 15. Prepare six fold-local preprocessing states

Có:

```text
2 training-window strategies
×
3 temporal folds
=
6 preprocessing states
```

Mỗi state có scaler/vocabulary riêng.

Không state nào được học từ fold-validation.


```python

STATE_KEYS = [
    (
        window_id,
        fold[
            "fold_id"
        ],
    )
    for window_id
    in WINDOW_STARTS
    for fold
    in FOLD_SPECS
]


fold_scalers = {
    key:
        FeaturewiseStandardScaler()
    for key
    in STATE_KEYS
}

fold_category_counters = {
    key: {
        column:
            Counter()
        for column
        in CATEGORICAL_COLUMNS
    }
    for key
    in STATE_KEYS
}

fold_fit_row_counts = Counter()

fold_fit_min_timestamp = {
    key: None
    for key
    in STATE_KEYS
}

fold_fit_max_timestamp = {
    key: None
    for key
    in STATE_KEYS
}

train_audit_parts = {
    key: []
    for key
    in STATE_KEYS
}

validation_audit_parts = {
    fold[
        "fold_id"
    ]: []
    for fold
    in FOLD_SPECS
}

long_row_parts = []
long_timestamp_parts = []


assert len(
    STATE_KEYS
) == 6


print(
    "Fold-local preprocessing states:",
    len(
        STATE_KEYS
    ),
)

print(
    "M7.2 PREPROCESSING STATE REGISTRY: READY"
)

```

    Fold-local preprocessing states: 6
    M7.2 PREPROCESSING STATE REGISTRY: READY


## 16. Full raw pass — fit fold-local preprocessing + rebuild Timestamp lineage

Pass này:

- stream full raw artifact;
- không đọc raw target;
- không đọc `Errors?`;
- giữ full strict-prior card history trước 2019;
- fit 6 preprocessing states từ đúng fold-TRAIN;
- collect deterministic train/validation transform samples;
- rebuild `raw_row_id ↔ Timestamp` cho canonical W_LONG rows.

Expected runtime:

`vài phút tùy máy`.

Không model selection xảy ra trong pass này.


```python

stream_start = (
    time.perf_counter()
)

card_count = 0
raw_rows_seen = 0

seen_card_keys = set()


for card_key, card_df in (
    iter_card_blocks(
        RAW_PATH,
        CHUNK_SIZE,
        SEMANTIC_USECOLS,
    )
):
    card_count += 1
    raw_rows_seen += len(
        card_df
    )

    if card_key in seen_card_keys:
        raise RuntimeError(
            "Card block reappearance: "
            f"{card_key}"
        )

    seen_card_keys.add(
        card_key
    )

    context_card_df = (
        card_df.loc[
            card_df[
                "Timestamp"
            ]
            < TRAIN_END
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    if context_card_df.empty:
        continue

    behavior_input = (
        context_card_df[
            [
                "raw_row_id",
                "Timestamp",
                "Amount_numeric",
                "Merchant Name",
            ]
        ]
        .copy()
    )

    assert (
        "Is Fraud?"
        not in behavior_input.columns
    )

    behavioral_df = (
        compute_causal_behavioral_features(
            behavior_input
        )
    )

    core_df = (
        build_core_feature_frame(
            context_card_df,
            behavioral_df,
        )
    )

    # ----------------------------------------------------------
    # Canonical W_LONG row/Timestamp mapping
    # ----------------------------------------------------------

    long_mask = (
        (
            core_df[
                "Timestamp"
            ]
            >= W_LONG_START
        )
        &
        (
            core_df[
                "Timestamp"
            ]
            < TRAIN_END
        )
    )

    if long_mask.any():
        long_row_parts.append(
            core_df.loc[
                long_mask,
                "raw_row_id",
            ].to_numpy(
                dtype=np.int64
            )
        )

        long_timestamp_parts.append(
            core_df.loc[
                long_mask,
                "Timestamp",
            ].to_numpy(
                dtype="datetime64[ns]"
            )
        )

    # ----------------------------------------------------------
    # Six fold-local preprocessing fit states
    # ----------------------------------------------------------

    timestamps = (
        core_df[
            "Timestamp"
        ]
    )

    for window_id, window_start in (
        WINDOW_STARTS.items()
    ):
        for fold in (
            FOLD_SPECS
        ):
            key = (
                window_id,
                fold[
                    "fold_id"
                ],
            )

            train_mask = (
                (
                    timestamps
                    >= window_start
                )
                &
                (
                    timestamps
                    <
                    fold[
                        "train_end_exclusive"
                    ]
                )
            )

            if (
                train_mask.any()
            ):
                train_sub = (
                    core_df.loc[
                        train_mask
                    ]
                )

                numeric_train = (
                    train_sub[
                        NUMERIC_COLUMNS
                    ]
                    .astype(
                        "float64"
                    )
                    .to_numpy()
                )

                fold_scalers[
                    key
                ].partial_fit(
                    numeric_train
                )

                fold_fit_row_counts[
                    key
                ] += len(
                    train_sub
                )

                current_min = (
                    train_sub[
                        "Timestamp"
                    ]
                    .min()
                )

                current_max = (
                    train_sub[
                        "Timestamp"
                    ]
                    .max()
                )

                if (
                    fold_fit_min_timestamp[
                        key
                    ]
                    is None
                    or
                    current_min
                    <
                    fold_fit_min_timestamp[
                        key
                    ]
                ):
                    fold_fit_min_timestamp[
                        key
                    ] = current_min

                if (
                    fold_fit_max_timestamp[
                        key
                    ]
                    is None
                    or
                    current_max
                    >
                    fold_fit_max_timestamp[
                        key
                    ]
                ):
                    fold_fit_max_timestamp[
                        key
                    ] = current_max

                for column in (
                    CATEGORICAL_COLUMNS
                ):
                    fold_category_counters[
                        key
                    ][
                        column
                    ].update(
                        train_sub[
                            column
                        ]
                        .astype(
                            "string"
                        )
                        .value_counts()
                        .to_dict()
                    )

                audit_mask = (
                    (
                        train_sub[
                            "raw_row_id"
                        ]
                        .to_numpy(
                            dtype=np.int64
                        )
                        % AUDIT_SAMPLE_MOD
                    )
                    == 0
                )

                if audit_mask.any():
                    train_audit_parts[
                        key
                    ].append(
                        train_sub.loc[
                            audit_mask,
                            [
                                "raw_row_id",
                                "Timestamp",
                                *CORE_FEATURE_COLUMNS,
                            ],
                        ].copy()
                    )

    # ----------------------------------------------------------
    # Shared fold-validation audit samples
    # ----------------------------------------------------------

    for fold in (
        FOLD_SPECS
    ):
        validation_mask = (
            (
                timestamps
                >= fold[
                    "validation_start"
                ]
            )
            &
            (
                timestamps
                <
                fold[
                    "validation_end_exclusive"
                ]
            )
        )

        if not validation_mask.any():
            continue

        validation_sub = (
            core_df.loc[
                validation_mask
            ]
        )

        audit_mask = (
            (
                validation_sub[
                    "raw_row_id"
                ]
                .to_numpy(
                    dtype=np.int64
                )
                % AUDIT_SAMPLE_MOD
            )
            == 0
        )

        if audit_mask.any():
            validation_audit_parts[
                fold[
                    "fold_id"
                ]
            ].append(
                validation_sub.loc[
                    audit_mask,
                    [
                        "raw_row_id",
                        "Timestamp",
                        *CORE_FEATURE_COLUMNS,
                    ],
                ].copy()
            )

    if (
        card_count
        % 500
        == 0
    ):
        print(
            "Card blocks:",
            f"{card_count:,}",
            "| raw rows:",
            f"{raw_rows_seen:,}",
        )


stream_elapsed = (
    time.perf_counter()
    - stream_start
)


assert (
    card_count
    == EXPECTED_CARD_COUNT
)

assert (
    raw_rows_seen
    == EXPECTED_RAW_ROWS
)


print(
    "\nCard blocks:",
    f"{card_count:,}",
)

print(
    "Raw rows streamed:",
    f"{raw_rows_seen:,}",
)

print(
    "Elapsed seconds:",
    round(
        stream_elapsed,
        2,
    ),
)

print(
    "\nM7.2 FULL RAW PREPROCESSING FIT PASS: COMPLETE"
)

```

    Card blocks: 500 | raw rows: 2,021,584
    Card blocks: 1,000 | raw rows: 4,066,794
    Card blocks: 1,500 | raw rows: 5,915,634
    Card blocks: 2,000 | raw rows: 7,883,446
    Card blocks: 2,500 | raw rows: 9,842,479
    Card blocks: 3,000 | raw rows: 11,897,259
    Card blocks: 3,500 | raw rows: 13,997,374
    Card blocks: 4,000 | raw rows: 16,040,304
    Card blocks: 4,500 | raw rows: 18,115,490
    Card blocks: 5,000 | raw rows: 19,945,815
    Card blocks: 5,500 | raw rows: 21,956,418
    Card blocks: 6,000 | raw rows: 23,844,247
    
    Card blocks: 6,139
    Raw rows streamed: 24,386,900
    Elapsed seconds: 187.17
    
    M7.2 FULL RAW PREPROCESSING FIT PASS: COMPLETE


## 17. Exact raw-row / Timestamp lineage gate


```python

rebuilt_long_row_id = np.concatenate(
    long_row_parts
)

rebuilt_long_timestamp = np.concatenate(
    long_timestamp_parts
)


assert (
    len(
        rebuilt_long_row_id
    )
    == EXPECTED_W_LONG_ROWS
)

assert (
    len(
        rebuilt_long_timestamp
    )
    == EXPECTED_W_LONG_ROWS
)


np.testing.assert_array_equal(
    rebuilt_long_row_id,
    row_long,
)


rebuilt_short_mask = (
    rebuilt_long_timestamp
    >= np.datetime64(
        "2018-01-01"
    )
)

rebuilt_short_row_id = (
    rebuilt_long_row_id[
        rebuilt_short_mask
    ]
)

rebuilt_short_timestamp = (
    rebuilt_long_timestamp[
        rebuilt_short_mask
    ]
)


np.testing.assert_array_equal(
    rebuilt_short_row_id,
    row_short,
)


assert (
    len(
        rebuilt_short_timestamp
    )
    == EXPECTED_W_SHORT_ROWS
)


print(
    "W_LONG exact row/Timestamp mapping:",
    len(
        rebuilt_long_row_id
    ),
)

print(
    "W_SHORT exact subset mapping:",
    len(
        rebuilt_short_row_id
    ),
)

print(
    "\nM7.2 ROW/TIMESTAMP LINEAGE GATE: PASS"
)

```

    W_LONG exact row/Timestamp mapping: 6855270
    W_SHORT exact subset mapping: 1721615
    
    M7.2 ROW/TIMESTAMP LINEAGE GATE: PASS


## 18. Full fold membership / positive-support audit


```python

def build_fold_registry(
    window_id,
    row_ids,
    timestamps,
    y,
):
    window_start = (
        WINDOW_STARTS[
            window_id
        ]
    )

    records = []

    timestamps_pd = pd.DatetimeIndex(
        timestamps
    )

    for fold in FOLD_SPECS:
        train_mask = (
            (
                timestamps_pd
                >= window_start
            )
            &
            (
                timestamps_pd
                <
                fold[
                    "train_end_exclusive"
                ]
            )
        )

        validation_mask = (
            (
                timestamps_pd
                >= fold[
                    "validation_start"
                ]
            )
            &
            (
                timestamps_pd
                <
                fold[
                    "validation_end_exclusive"
                ]
            )
        )

        train_rows = int(
            train_mask.sum()
        )

        validation_rows = int(
            validation_mask.sum()
        )

        train_fraud = int(
            y[
                train_mask
            ].sum()
        )

        validation_fraud = int(
            y[
                validation_mask
            ].sum()
        )

        assert train_rows > 0
        assert validation_rows > 0
        assert train_fraud > 0
        assert validation_fraud > 0

        train_row_ids = (
            row_ids[
                train_mask
            ]
        )

        validation_row_ids = (
            row_ids[
                validation_mask
            ]
        )

        assert np.intersect1d(
            train_row_ids,
            validation_row_ids,
            assume_unique=True,
        ).size == 0

        train_max = (
            timestamps_pd[
                train_mask
            ]
            .max()
        )

        validation_min = (
            timestamps_pd[
                validation_mask
            ]
            .min()
        )

        assert (
            train_max
            < validation_min
        )

        records.append(
            {
                "training_window_id":
                    window_id,

                "fold_id":
                    fold[
                        "fold_id"
                    ],

                "train_start":
                    str(
                        window_start
                    ),

                "train_end_exclusive":
                    str(
                        fold[
                            "train_end_exclusive"
                        ]
                    ),

                "validation_start":
                    str(
                        fold[
                            "validation_start"
                        ]
                    ),

                "validation_end_exclusive":
                    str(
                        fold[
                            "validation_end_exclusive"
                        ]
                    ),

                "train_rows":
                    train_rows,

                "validation_rows":
                    validation_rows,

                "train_fraud_rows":
                    train_fraud,

                "validation_fraud_rows":
                    validation_fraud,

                "train_max_timestamp":
                    str(
                        train_max
                    ),

                "validation_min_timestamp":
                    str(
                        validation_min
                    ),

                "row_overlap":
                    0,

                "positive_support":
                    "PASS",
            }
        )

    return records


fold_registry = []


fold_registry.extend(
    build_fold_registry(
        "W_LONG",
        rebuilt_long_row_id,
        rebuilt_long_timestamp,
        y_long,
    )
)


fold_registry.extend(
    build_fold_registry(
        "W_SHORT",
        rebuilt_short_row_id,
        rebuilt_short_timestamp,
        y_short,
    )
)


assert len(
    fold_registry
) == 6


for record in fold_registry:
    print(
        record[
            "training_window_id"
        ],
        record[
            "fold_id"
        ],
        "| train rows/fraud:",
        record[
            "train_rows"
        ],
        "/",
        record[
            "train_fraud_rows"
        ],
        "| validation rows/fraud:",
        record[
            "validation_rows"
        ],
        "/",
        record[
            "validation_fraud_rows"
        ],
    )


print(
    "\nM7.2 FOLD ORDER / OVERLAP / POSITIVE SUPPORT GATE: PASS"
)

```

    W_LONG FOLD_Q2_2018 | train rows/fraud: 5557560 / 7672 | validation rows/fraud: 428953 / 590
    W_LONG FOLD_Q3_2018 | train rows/fraud: 5986513 / 8262 | validation rows/fraud: 435178 / 634
    W_LONG FOLD_Q4_2018 | train rows/fraud: 6421691 / 8896 | validation rows/fraud: 433579 / 710
    W_SHORT FOLD_Q2_2018 | train rows/fraud: 423905 / 557 | validation rows/fraud: 428953 / 590
    W_SHORT FOLD_Q3_2018 | train rows/fraud: 852858 / 1147 | validation rows/fraud: 435178 / 634
    W_SHORT FOLD_Q4_2018 | train rows/fraud: 1288036 / 1781 | validation rows/fraud: 433579 / 710
    
    M7.2 FOLD ORDER / OVERLAP / POSITIVE SUPPORT GATE: PASS


## 19. Verify fold-local preprocessing fit populations


```python

fold_registry_by_key = {
    (
        record[
            "training_window_id"
        ],
        record[
            "fold_id"
        ],
    ):
        record
    for record
    in fold_registry
}


for key in STATE_KEYS:
    record = (
        fold_registry_by_key[
            key
        ]
    )

    assert (
        fold_fit_row_counts[
            key
        ]
        == record[
            "train_rows"
        ]
    )

    assert (
        fold_fit_min_timestamp[
            key
        ]
        >= WINDOW_STARTS[
            key[0]
        ]
    )

    assert (
        fold_fit_max_timestamp[
            key
        ]
        <
        pd.Timestamp(
            record[
                "validation_start"
            ]
        )
    )

    scaler = (
        fold_scalers[
            key
        ]
    )

    assert (
        scaler.mean_
        is not None
    )

    assert np.isfinite(
        scaler.mean_
    ).all()

    assert np.isfinite(
        scaler.scale_
    ).all()

    assert np.all(
        scaler.scale_
        > 0
    )


for key in STATE_KEYS:
    print(
        key,
        "| fit rows:",
        fold_fit_row_counts[
            key
        ],
        "| range:",
        fold_fit_min_timestamp[
            key
        ],
        "→",
        fold_fit_max_timestamp[
            key
        ],
    )


print(
    "\nM7.2 FOLD-TRAIN PREPROCESSING FIT-SOURCE GATE: PASS"
)

```

    ('W_LONG', 'FOLD_Q2_2018') | fit rows: 5557560 | range: 2015-01-01 00:01:00 → 2018-03-31 23:59:00
    ('W_LONG', 'FOLD_Q3_2018') | fit rows: 5986513 | range: 2015-01-01 00:01:00 → 2018-06-30 23:59:00
    ('W_LONG', 'FOLD_Q4_2018') | fit rows: 6421691 | range: 2015-01-01 00:01:00 → 2018-09-30 23:57:00
    ('W_SHORT', 'FOLD_Q2_2018') | fit rows: 423905 | range: 2018-01-01 00:03:00 → 2018-03-31 23:59:00
    ('W_SHORT', 'FOLD_Q3_2018') | fit rows: 852858 | range: 2018-01-01 00:03:00 → 2018-06-30 23:59:00
    ('W_SHORT', 'FOLD_Q4_2018') | fit rows: 1288036 | range: 2018-01-01 00:03:00 → 2018-09-30 23:57:00
    
    M7.2 FOLD-TRAIN PREPROCESSING FIT-SOURCE GATE: PASS


## 20. Freeze six fold-local preprocessing bundles


```python

fold_bundles = {}


for key in STATE_KEYS:
    window_id, fold_id = key

    category_vocab = {
        column:
            set(
                fold_category_counters[
                    key
                ][
                    column
                ]
                .keys()
            )
        for column
        in CATEGORICAL_COLUMNS
    }

    encoder = (
        build_encoder_from_train_vocab(
            category_vocab
        )
    )

    feature_names = (
        build_feature_names(
            encoder
        )
    )

    fold_bundles[
        key
    ] = FoldPreprocessingBundle(
        state_id=(
            f"M7.2-"
            f"{window_id}-"
            f"{fold_id}-"
            "PREPROCESSING-v1"
        ),

        training_window_id=
            window_id,

        fold_id=
            fold_id,

        scaler=
            fold_scalers[
                key
            ],

        category_vocab=
            category_vocab,

        encoder=
            encoder,

        feature_names=
            feature_names,

        fit_row_count=
            int(
                fold_fit_row_counts[
                    key
                ]
            ),

        fit_min_timestamp=
            fold_fit_min_timestamp[
                key
            ],

        fit_max_timestamp=
            fold_fit_max_timestamp[
                key
            ],
    )


assert len(
    fold_bundles
) == 6


for key, bundle in (
    fold_bundles.items()
):
    assert (
        bundle.feature_names
        == canonical_feature_names
    )

    assert (
        len(
            bundle.feature_names
        )
        == EXPECTED_OUTPUT_WIDTH
    )

    for column in (
        CATEGORICAL_COLUMNS
    ):
        assert (
            UNKNOWN_TOKEN
            not in bundle
            .category_vocab[
                column
            ]
        )


print(
    "Fold-local bundles:",
    len(
        fold_bundles
    ),
)

print(
    "All schemas match canonical M4.7 "
    "47-column feature names."
)

print(
    "\nM7.2 FOLD-LOCAL BUNDLE FREEZE GATE: PASS"
)

```

    Fold-local bundles: 6
    All schemas match canonical M4.7 47-column feature names.
    
    M7.2 FOLD-LOCAL BUNDLE FREEZE GATE: PASS


## 21. Transform-only audit samples

Mục tiêu:

- transform deterministic fold-TRAIN samples;
- transform deterministic fold-validation samples;
- xác minh validation transform không thay scaler state;
- output CSR float32;
- width 47;
- finite;
- unknown-category route hoạt động.

Các sample này chỉ dùng audit hạ tầng, không dùng làm selection evidence.


```python

transform_audit = []


for key, bundle in (
    fold_bundles.items()
):
    window_id, fold_id = key

    train_parts = (
        train_audit_parts[
            key
        ]
    )

    validation_parts = (
        validation_audit_parts[
            fold_id
        ]
    )

    assert train_parts
    assert validation_parts

    train_sample = pd.concat(
        train_parts,
        ignore_index=True,
    )

    validation_sample = pd.concat(
        validation_parts,
        ignore_index=True,
    )

    scaler_mean_before = (
        bundle.scaler
        .mean_
        .copy()
    )

    scaler_scale_before = (
        bundle.scaler
        .scale_
        .copy()
    )

    X_train_sample = (
        transform_with_bundle(
            train_sample,
            bundle,
        )
    )

    X_validation_sample = (
        transform_with_bundle(
            validation_sample,
            bundle,
        )
    )

    np.testing.assert_allclose(
        bundle.scaler.mean_,
        scaler_mean_before,
    )

    np.testing.assert_allclose(
        bundle.scaler.scale_,
        scaler_scale_before,
    )

    assert sparse.isspmatrix_csr(
        X_train_sample
    )

    assert sparse.isspmatrix_csr(
        X_validation_sample
    )

    assert (
        X_train_sample.dtype
        == np.float32
    )

    assert (
        X_validation_sample.dtype
        == np.float32
    )

    assert (
        X_train_sample.shape[1]
        == EXPECTED_OUTPUT_WIDTH
    )

    assert (
        X_validation_sample.shape[1]
        == EXPECTED_OUTPUT_WIDTH
    )

    assert np.isfinite(
        X_train_sample.data
    ).all()

    assert np.isfinite(
        X_validation_sample.data
    ).all()

    # Explicit unknown-category route stress test.
    stress = (
        validation_sample
        .iloc[
            [0]
        ]
        .copy()
    )

    stress.loc[
        stress.index[0],
        "transaction_mode",
    ] = "UNSEEN_MODE_M7_AUDIT"

    X_stress = (
        transform_with_bundle(
            stress,
            bundle,
        )
    )

    unknown_candidates = [
        index
        for index, name
        in enumerate(
            bundle.feature_names
        )
        if (
            name.startswith(
                "cat__transaction_mode_"
            )
            and
            UNKNOWN_TOKEN
            in name
        )
    ]

    assert len(
        unknown_candidates
    ) == 1

    assert (
        X_stress[
            0,
            unknown_candidates[
                0
            ],
        ]
        == 1.0
    )

    transform_audit.append(
        {
            "training_window_id":
                window_id,

            "fold_id":
                fold_id,

            "train_audit_rows":
                int(
                    len(
                        train_sample
                    )
                ),

            "validation_audit_rows":
                int(
                    len(
                        validation_sample
                    )
                ),

            "output_width":
                int(
                    X_validation_sample
                    .shape[1]
                ),

            "matrix_format":
                "CSR",

            "matrix_dtype":
                str(
                    X_validation_sample
                    .dtype
                ),

            "validation_transform_mutated_scaler":
                False,

            "unknown_route":
                "PASS",

            "status":
                "PASS",
        }
    )


for item in transform_audit:
    print(
        item[
            "training_window_id"
        ],
        item[
            "fold_id"
        ],
        "| train sample:",
        item[
            "train_audit_rows"
        ],
        "| validation sample:",
        item[
            "validation_audit_rows"
        ],
        "| width:",
        item[
            "output_width"
        ],
    )


print(
    "\nM7.2 TRANSFORM-ONLY / SCHEMA GATE: PASS"
)

```

    W_LONG FOLD_Q2_2018 | train sample: 5623 | validation sample: 428 | width: 47
    W_LONG FOLD_Q3_2018 | train sample: 6051 | validation sample: 424 | width: 47
    W_LONG FOLD_Q4_2018 | train sample: 6475 | validation sample: 463 | width: 47
    W_SHORT FOLD_Q2_2018 | train sample: 429 | validation sample: 428 | width: 47
    W_SHORT FOLD_Q3_2018 | train sample: 857 | validation sample: 424 | width: 47
    W_SHORT FOLD_Q4_2018 | train sample: 1281 | validation sample: 463 | width: 47
    
    M7.2 TRANSFORM-ONLY / SCHEMA GATE: PASS


## 22. Sampler insertion-point contract

Shared runner phải có thứ tự:

```text
fold semantic data
        ↓
fit fold preprocessing on fold-TRAIN
        ↓
transform fold-TRAIN
transform fold-validation
        ↓
optional sampler:
TRAIN MATRIX / TRAIN LABEL ONLY
        ↓
fit estimator
        ↓
predict untouched fold-validation
```

M7.2 chỉ unit-test placement bằng identity sampler.

Không có real imbalance experiment trong M7.2.


```python
CANONICAL_CANDIDATE_IDENTITY_FIELDS = [
    "candidate_id",
    "milestone_substep",
    "dataset_artifact_identity",
    "feature_version",
    "preprocessing_version",
    "training_window_id",
    "model_family",
    "model_config_id",
    "hyperparameters",
    "imbalance_strategy",
    "class_weight",
    "sampling_method",
    "sampling_ratio",
    "random_state",
    "cv_spec_id",
    "fold_template_id",
    "threshold_policy",
    "probability_interface",
    "metric_contract_version",
]

CANONICAL_FOLD_IDENTITY_FIELDS = [
    "fold_id",
    "train_start",
    "train_end_exclusive",
    "validation_start",
    "validation_end_exclusive",
    "train_rows",
    "validation_rows",
    "train_fraud_rows",
    "validation_fraud_rows",
]

CANONICAL_FOLD_RESULT_FIELDS = [
    "candidate_id",
    "fold_id",
    "train_start",
    "train_end_exclusive",
    "validation_start",
    "validation_end_exclusive",
    "train_rows",
    "validation_rows",
    "train_fraud_rows",
    "validation_fraud_rows",
    "feature_version",
    "preprocessing_version",
    "model_family",
    "model_config_id",
    "hyperparameters",
    "imbalance_strategy",
    "random_state",
    "threshold_policy",
    "prediction_artifact",
    "risk_score_artifact",
    "F1_fraud",
    "Recall_fraud",
    "Precision_fraud",
    "TP",
    "FP",
    "FN",
    "TN",
    "predicted_positive_count",
    "predicted_positive_rate",
    "Accuracy_reference",
    "fit_seconds",
    "prediction_seconds",
    "warning_count",
    "warnings",
    "integrity_status",
]

CANONICAL_AGGREGATE_FIELDS = [
    "candidate_id",
    "valid_fold_count",
    "expected_fold_count",
    "fold_ids",
    "mean_F1",
    "std_F1",
    "mean_Recall",
    "mean_Precision",
    "foldwise_metrics",
    "pooled_OOF_metrics",
    "total_fit_seconds",
    "total_prediction_seconds",
    "warnings_summary",
    "stability_findings",
    "tradeoff_findings",
    "selection_status",
    "decision_reason",
    "next_action",
]


def validate_required_fields(
    payload,
    required_fields,
    payload_name,
):
    missing = [
        field
        for field in required_fields
        if field not in payload
    ]

    if missing:
        raise ValueError(
            f"{payload_name} thiếu field bắt buộc: {missing}"
        )


def validate_candidate_identity(
    candidate_identity,
):
    validate_required_fields(
        candidate_identity,
        CANONICAL_CANDIDATE_IDENTITY_FIELDS,
        "candidate_identity",
    )

    if (
        candidate_identity.get(
            "final_test_accessed",
            None,
        )
        is not False
    ):
        raise AssertionError(
            "candidate_identity.final_test_accessed phải là False."
        )


def validate_fold_identity(
    fold_identity,
    X_train,
    y_train,
    X_validation,
    y_validation,
):
    validate_required_fields(
        fold_identity,
        CANONICAL_FOLD_IDENTITY_FIELDS,
        "fold_identity",
    )

    y_train = np.asarray(
        y_train,
        dtype=np.int8,
    )

    y_validation = np.asarray(
        y_validation,
        dtype=np.int8,
    )

    assert (
        int(fold_identity["train_rows"])
        == int(X_train.shape[0])
        == len(y_train)
    )

    assert (
        int(fold_identity["validation_rows"])
        == int(X_validation.shape[0])
        == len(y_validation)
    )

    assert (
        int(fold_identity["train_fraud_rows"])
        == int(y_train.sum())
    )

    assert (
        int(fold_identity["validation_fraud_rows"])
        == int(y_validation.sum())
    )


def validate_fold_result_schema(
    fold_result,
):
    validate_required_fields(
        fold_result,
        CANONICAL_FOLD_RESULT_FIELDS,
        "fold_result",
    )

    if set(fold_result) != set(
        CANONICAL_FOLD_RESULT_FIELDS
    ):
        extra = sorted(
            set(fold_result)
            - set(CANONICAL_FOLD_RESULT_FIELDS)
        )

        raise ValueError(
            "fold_result có field ngoài canonical schema: "
            f"{extra}"
        )


class IdentityAuditSampler:
    """
    Sampler audit tối giản chỉ expose fit_resample(X, y).

    Mục đích:
    kiểm tra shared runner không phụ thuộc metadata riêng
    của một sampler implementation cụ thể.
    """

    def fit_resample(
        self,
        X,
        y,
    ):
        return (
            X,
            np.asarray(y),
        )


def compute_metric_bundle(
    y_true,
    y_pred,
):
    y_true = np.asarray(
        y_true,
        dtype=np.int8,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=np.int8,
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    tn, fp, fn, tp = cm.ravel()

    predicted_positive_count = int(
        tp + fp
    )

    return {
        "F1_fraud":
            float(
                f1_score(
                    y_true,
                    y_pred,
                    pos_label=1,
                    zero_division=0,
                )
            ),

        "Recall_fraud":
            float(
                recall_score(
                    y_true,
                    y_pred,
                    pos_label=1,
                    zero_division=0,
                )
            ),

        "Precision_fraud":
            float(
                precision_score(
                    y_true,
                    y_pred,
                    pos_label=1,
                    zero_division=0,
                )
            ),

        "Accuracy_reference":
            float(
                accuracy_score(
                    y_true,
                    y_pred,
                )
            ),

        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),

        "predicted_positive_count":
            predicted_positive_count,

        "predicted_positive_rate":
            float(
                predicted_positive_count
                / len(y_true)
            ),
    }


def run_m7_fold_candidate(
    *,
    estimator,
    X_train,
    y_train,
    X_validation,
    y_validation,
    candidate_identity,
    fold_identity,
    sampler=None,
    prediction_artifact=None,
    risk_score_artifact=None,
):
    """
    Shared M7 fold runner.

    - validate canonical candidate/fold identity trước fit;
    - sampler chỉ nhận fold-TRAIN;
    - runner tự ghi sampler input/output row counts;
    - capture warnings từ sampler/model;
    - fraud probability được xác định bằng classes_ == 1;
    - exact canonical fold-result schema;
    - FINAL TEST access phải False.
    """

    validate_candidate_identity(
        candidate_identity
    )

    validate_fold_identity(
        fold_identity,
        X_train,
        y_train,
        X_validation,
        y_validation,
    )

    X_fit = X_train
    y_fit = np.asarray(
        y_train,
        dtype=np.int8,
    )

    sampler_input_rows = None
    sampler_output_rows = None
    risk_score = None
    positive_class_index = None

    with warnings.catch_warnings(
        record=True
    ) as captured_warnings:
        warnings.simplefilter(
            "always"
        )

        if sampler is not None:
            sampler_input_rows = int(
                X_fit.shape[0]
            )

            X_fit, y_fit = sampler.fit_resample(
                X_fit,
                y_fit,
            )

            y_fit = np.asarray(
                y_fit,
                dtype=np.int8,
            )

            sampler_output_rows = int(
                X_fit.shape[0]
            )

        fit_start = time.perf_counter()

        fitted = clone(
            estimator
        )

        fitted.fit(
            X_fit,
            y_fit,
        )

        fit_seconds = (
            time.perf_counter()
            - fit_start
        )

        prediction_start = (
            time.perf_counter()
        )

        y_pred = fitted.predict(
            X_validation
        )

        prediction_seconds = (
            time.perf_counter()
            - prediction_start
        )

        if hasattr(
            fitted,
            "predict_proba",
        ):
            if not hasattr(
                fitted,
                "classes_",
            ):
                raise AttributeError(
                    "Estimator có predict_proba nhưng không có classes_."
                )

            classes = np.asarray(
                fitted.classes_
            )

            positive_matches = (
                np.flatnonzero(
                    classes == 1
                )
            )

            if len(
                positive_matches
            ) != 1:
                raise ValueError(
                    "Không xác định duy nhất positive class = 1. "
                    f"classes_={classes.tolist()}"
                )

            positive_class_index = int(
                positive_matches[0]
            )

            proba = np.asarray(
                fitted.predict_proba(
                    X_validation
                ),
                dtype=np.float64,
            )

            if (
                proba.ndim != 2
                or proba.shape[0]
                != X_validation.shape[0]
                or proba.shape[1]
                != len(classes)
            ):
                raise ValueError(
                    "predict_proba shape không tương thích classes_."
                )

            risk_score = proba[
                :,
                positive_class_index
            ]

            if not np.isfinite(
                risk_score
            ).all():
                raise ValueError(
                    "Risk score chứa NaN/inf."
                )

            if (
                (risk_score < 0.0).any()
                or
                (risk_score > 1.0).any()
            ):
                raise ValueError(
                    "Risk score ngoài [0, 1]."
                )

    warning_records = [
        {
            "category":
                warning.category.__name__,

            "message":
                str(warning.message),
        }
        for warning
        in captured_warnings
    ]

    metrics = compute_metric_bundle(
        y_validation,
        y_pred,
    )

    fold_result = {
        "candidate_id":
            candidate_identity[
                "candidate_id"
            ],

        "fold_id":
            fold_identity[
                "fold_id"
            ],

        "train_start":
            fold_identity[
                "train_start"
            ],

        "train_end_exclusive":
            fold_identity[
                "train_end_exclusive"
            ],

        "validation_start":
            fold_identity[
                "validation_start"
            ],

        "validation_end_exclusive":
            fold_identity[
                "validation_end_exclusive"
            ],

        "train_rows":
            int(
                fold_identity[
                    "train_rows"
                ]
            ),

        "validation_rows":
            int(
                fold_identity[
                    "validation_rows"
                ]
            ),

        "train_fraud_rows":
            int(
                fold_identity[
                    "train_fraud_rows"
                ]
            ),

        "validation_fraud_rows":
            int(
                fold_identity[
                    "validation_fraud_rows"
                ]
            ),

        "feature_version":
            candidate_identity[
                "feature_version"
            ],

        "preprocessing_version":
            candidate_identity[
                "preprocessing_version"
            ],

        "model_family":
            candidate_identity[
                "model_family"
            ],

        "model_config_id":
            candidate_identity[
                "model_config_id"
            ],

        "hyperparameters":
            copy.deepcopy(
                candidate_identity[
                    "hyperparameters"
                ]
            ),

        "imbalance_strategy":
            candidate_identity[
                "imbalance_strategy"
            ],

        "random_state":
            candidate_identity[
                "random_state"
            ],

        "threshold_policy":
            candidate_identity[
                "threshold_policy"
            ],

        "prediction_artifact":
            prediction_artifact,

        "risk_score_artifact":
            risk_score_artifact,

        "F1_fraud":
            metrics[
                "F1_fraud"
            ],

        "Recall_fraud":
            metrics[
                "Recall_fraud"
            ],

        "Precision_fraud":
            metrics[
                "Precision_fraud"
            ],

        "TP":
            metrics["TP"],

        "FP":
            metrics["FP"],

        "FN":
            metrics["FN"],

        "TN":
            metrics["TN"],

        "predicted_positive_count":
            metrics[
                "predicted_positive_count"
            ],

        "predicted_positive_rate":
            metrics[
                "predicted_positive_rate"
            ],

        "Accuracy_reference":
            metrics[
                "Accuracy_reference"
            ],

        "fit_seconds":
            float(fit_seconds),

        "prediction_seconds":
            float(
                prediction_seconds
            ),

        "warning_count":
            int(
                len(
                    warning_records
                )
            ),

        "warnings":
            warning_records,

        "integrity_status":
            "PASS",
    }

    validate_fold_result_schema(
        fold_result
    )

    return {
        "runner_version":
            M7_RUNNER_VERSION,

        "candidate_identity":
            copy.deepcopy(
                candidate_identity
            ),

        "fold_identity":
            copy.deepcopy(
                fold_identity
            ),

        "fold_result":
            fold_result,

        "sampler_applied":
            sampler is not None,

        "sampler_input_rows":
            sampler_input_rows,

        "sampler_output_rows":
            sampler_output_rows,

        "risk_score_available":
            risk_score is not None,

        "positive_class_index":
            positive_class_index,

        "risk_score":
            risk_score,

        "y_pred":
            np.asarray(
                y_pred,
                dtype=np.int8,
            ),

        "final_test_accessed":
            False,
    }


print(
    "M7.2 SAMPLER / METRIC / RUNNER HELPERS: DEFINED"
)
```

    M7.2 SAMPLER / METRIC / RUNNER HELPERS: DEFINED


## 23. Synthetic shared-runner unit test

Đây là:

`INFRASTRUCTURE UNIT TEST ONLY`.

Không dùng project fraud data.

Không tạo model-selection evidence.

Mục tiêu:

- estimator fit/predict path;
- sampler chỉ nhận train rows;
- validation không bị resample;
- metric bundle arithmetic;
- risk-score interface;
- FINAL TEST guardrail.


```python
X_toy_train = sparse.csr_matrix(
    np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.2, 0.1],
            [0.8, 0.9],
            [0.9, 1.0],
            [1.0, 1.1],
            [0.3, 0.2],
            [0.7, 0.8],
        ],
        dtype=np.float32,
    )
)

y_toy_train = np.array(
    [0, 0, 0, 1, 1, 1, 0, 1],
    dtype=np.int8,
)

X_toy_validation = sparse.csr_matrix(
    np.array(
        [
            [0.05, 0.0],
            [0.25, 0.1],
            [0.75, 0.8],
            [0.95, 1.0],
        ],
        dtype=np.float32,
    )
)

y_toy_validation = np.array(
    [0, 0, 1, 1],
    dtype=np.int8,
)


toy_candidate = {
    "candidate_id":
        "M7.2-SYNTHETIC-LR-RUNNER-UNIT",

    "milestone_substep":
        "M7.2",

    "dataset_artifact_identity":
        "SYNTHETIC_IN_MEMORY_ONLY",

    "feature_version":
        "SYNTHETIC_2COL-v1",

    "preprocessing_version":
        "SYNTHETIC_NO_PREPROCESSING-v1",

    "training_window_id":
        "SYNTHETIC_ONLY",

    "model_family":
        "Logistic Regression",

    "model_config_id":
        "UNIT_TEST-LBFGS",

    "hyperparameters": {
        "solver":
            "lbfgs",

        "max_iter":
            200,
    },

    "imbalance_strategy":
        "IDENTITY_AUDIT_SAMPLER",

    "class_weight":
        None,

    "sampling_method":
        "IDENTITY_AUDIT",

    "sampling_ratio":
        None,

    "random_state":
        RANDOM_STATE,

    "cv_spec_id":
        M7_CV_SPEC_ID,

    "fold_template_id":
        M7_FOLD_TEMPLATE_ID,

    "threshold_policy":
        "DEFAULT_MODEL_DECISION_RULE",

    "probability_interface":
        "predict_proba / positive class = 1",

    "metric_contract_version":
        M7_METRIC_CONTRACT_VERSION,

    "final_test_accessed":
        False,
}


toy_fold_identity = {
    "fold_id":
        "SYNTHETIC_UNIT_TEST",

    "train_start":
        "SYNTHETIC",

    "train_end_exclusive":
        "SYNTHETIC",

    "validation_start":
        "SYNTHETIC",

    "validation_end_exclusive":
        "SYNTHETIC",

    "train_rows":
        int(
            len(
                y_toy_train
            )
        ),

    "validation_rows":
        int(
            len(
                y_toy_validation
            )
        ),

    "train_fraud_rows":
        int(
            y_toy_train.sum()
        ),

    "validation_fraud_rows":
        int(
            y_toy_validation.sum()
        ),
}


# Test A — generic sampler compatibility + canonical schema
toy_result = run_m7_fold_candidate(
    estimator=
        LogisticRegression(
            solver="lbfgs",
            max_iter=200,
            random_state=RANDOM_STATE,
        ),

    X_train=
        X_toy_train,

    y_train=
        y_toy_train,

    X_validation=
        X_toy_validation,

    y_validation=
        y_toy_validation,

    candidate_identity=
        toy_candidate,

    fold_identity=
        toy_fold_identity,

    sampler=
        IdentityAuditSampler(),

    prediction_artifact=
        "IN_MEMORY_UNIT_TEST",

    risk_score_artifact=
        "IN_MEMORY_UNIT_TEST",
)


assert (
    toy_result[
        "sampler_input_rows"
    ]
    == len(
        y_toy_train
    )
)

assert (
    toy_result[
        "sampler_output_rows"
    ]
    == len(
        y_toy_train
    )
)

assert (
    toy_result[
        "fold_result"
    ][
        "validation_rows"
    ]
    == len(
        y_toy_validation
    )
)

assert (
    toy_result[
        "risk_score_available"
    ]
    is True
)

assert (
    toy_result[
        "positive_class_index"
    ]
    == 1
)

assert (
    set(
        toy_result[
            "fold_result"
        ]
    )
    == set(
        CANONICAL_FOLD_RESULT_FIELDS
    )
)


toy_metrics = toy_result[
    "fold_result"
]

assert (
    toy_metrics["TP"]
    + toy_metrics["FN"]
    == int(
        y_toy_validation.sum()
    )
)

assert (
    toy_metrics["TP"]
    + toy_metrics["FP"]
    == toy_metrics[
        "predicted_positive_count"
    ]
)


print(
    "Generic sampler compatibility: PASS"
)


# Test B — warning capture + reversed classes_ probability mapping
class ReversedClassWarningAuditEstimator(
    ClassifierMixin,
    BaseEstimator,
):
    def fit(
        self,
        X,
        y,
    ):
        warnings.warn(
            "M7.2 intentional warning audit",
            UserWarning,
        )

        self.classes_ = np.array(
            [1, 0],
            dtype=np.int8,
        )

        return self

    def _positive_score(
        self,
        X,
    ):
        if sparse.issparse(X):
            values = (
                X[:, 0]
                .toarray()
                .ravel()
            )
        else:
            values = np.asarray(X)[:, 0]

        return np.clip(
            values,
            0.0,
            1.0,
        ).astype(np.float64)

    def predict_proba(
        self,
        X,
    ):
        p1 = self._positive_score(X)

        return np.column_stack(
            [
                p1,
                1.0 - p1,
            ]
        )

    def predict(
        self,
        X,
    ):
        return (
            self._positive_score(X)
            >= 0.5
        ).astype(np.int8)


audit_candidate = copy.deepcopy(
    toy_candidate
)

audit_candidate[
    "candidate_id"
] = (
    "M7.2-REVERSED-CLASS-WARNING-AUDIT"
)

audit_candidate[
    "model_family"
] = (
    "RunnerContractAuditEstimator"
)

audit_candidate[
    "model_config_id"
] = (
    "REVERSED_CLASSES_WARNING_UNIT"
)

audit_candidate[
    "hyperparameters"
] = {}


warning_result = run_m7_fold_candidate(
    estimator=
        ReversedClassWarningAuditEstimator(),

    X_train=
        X_toy_train,

    y_train=
        y_toy_train,

    X_validation=
        X_toy_validation,

    y_validation=
        y_toy_validation,

    candidate_identity=
        audit_candidate,

    fold_identity=
        toy_fold_identity,

    sampler=
        IdentityAuditSampler(),

    prediction_artifact=
        "IN_MEMORY_WARNING_UNIT_TEST",

    risk_score_artifact=
        "IN_MEMORY_WARNING_UNIT_TEST",
)


assert (
    warning_result[
        "positive_class_index"
    ]
    == 0
)


expected_positive_score = (
    X_toy_validation[
        :,
        0,
    ]
    .toarray()
    .ravel()
    .astype(np.float64)
)


np.testing.assert_allclose(
    warning_result[
        "risk_score"
    ],
    expected_positive_score,
)


assert (
    warning_result[
        "fold_result"
    ][
        "warning_count"
    ]
    >= 1
)


assert any(
    "M7.2 intentional warning audit"
    in item["message"]
    for item
    in warning_result[
        "fold_result"
    ][
        "warnings"
    ]
)


print(
    "Warning capture: PASS"
)

print(
    "Positive-class probability mapping: PASS"
)


# Test C — missing candidate field must be rejected before fit
bad_candidate = copy.deepcopy(
    toy_candidate
)

bad_candidate.pop(
    "metric_contract_version"
)

schema_rejection_observed = False

try:
    run_m7_fold_candidate(
        estimator=
            LogisticRegression(
                solver="lbfgs",
                max_iter=50,
                random_state=RANDOM_STATE,
            ),

        X_train=
            X_toy_train,

        y_train=
            y_toy_train,

        X_validation=
            X_toy_validation,

        y_validation=
            y_toy_validation,

        candidate_identity=
            bad_candidate,

        fold_identity=
            toy_fold_identity,

        sampler=
            None,
    )

except ValueError as error:
    assert (
        "metric_contract_version"
        in str(error)
    )

    schema_rejection_observed = True


assert (
    schema_rejection_observed
    is True
)


print(
    "Candidate schema enforcement: PASS"
)

print(
    "Fold-result schema enforcement: PASS"
)

print(
    "\nSynthetic runner fold_result:"
)

print(
    toy_result[
        "fold_result"
    ]
)

print(
    "\nM7.2 GENERIC SAMPLER COMPATIBILITY GATE: PASS"
)

print(
    "M7.2 WARNING CAPTURE GATE: PASS"
)

print(
    "M7.2 POSITIVE-CLASS PROBABILITY MAPPING GATE: PASS"
)

print(
    "M7.2 CANONICAL SCHEMA ENFORCEMENT GATE: PASS"
)

print(
    "M7.2 SHARED RUNNER UNIT TEST GATE: PASS"
)
```

    Generic sampler compatibility: PASS
    Warning capture: PASS
    Positive-class probability mapping: PASS
    Candidate schema enforcement: PASS
    Fold-result schema enforcement: PASS
    
    Synthetic runner fold_result:
    {'candidate_id': 'M7.2-SYNTHETIC-LR-RUNNER-UNIT', 'fold_id': 'SYNTHETIC_UNIT_TEST', 'train_start': 'SYNTHETIC', 'train_end_exclusive': 'SYNTHETIC', 'validation_start': 'SYNTHETIC', 'validation_end_exclusive': 'SYNTHETIC', 'train_rows': 8, 'validation_rows': 4, 'train_fraud_rows': 4, 'validation_fraud_rows': 2, 'feature_version': 'SYNTHETIC_2COL-v1', 'preprocessing_version': 'SYNTHETIC_NO_PREPROCESSING-v1', 'model_family': 'Logistic Regression', 'model_config_id': 'UNIT_TEST-LBFGS', 'hyperparameters': {'solver': 'lbfgs', 'max_iter': 200}, 'imbalance_strategy': 'IDENTITY_AUDIT_SAMPLER', 'random_state': 42, 'threshold_policy': 'DEFAULT_MODEL_DECISION_RULE', 'prediction_artifact': 'IN_MEMORY_UNIT_TEST', 'risk_score_artifact': 'IN_MEMORY_UNIT_TEST', 'F1_fraud': 1.0, 'Recall_fraud': 1.0, 'Precision_fraud': 1.0, 'TP': 2, 'FP': 0, 'FN': 0, 'TN': 2, 'predicted_positive_count': 2, 'predicted_positive_rate': 0.5, 'Accuracy_reference': 1.0, 'fit_seconds': 0.0167526250006631, 'prediction_seconds': 0.00021500000730156898, 'warning_count': 0, 'warnings': [], 'integrity_status': 'PASS'}
    
    M7.2 GENERIC SAMPLER COMPATIBILITY GATE: PASS
    M7.2 WARNING CAPTURE GATE: PASS
    M7.2 POSITIVE-CLASS PROBABILITY MAPPING GATE: PASS
    M7.2 CANONICAL SCHEMA ENFORCEMENT GATE: PASS
    M7.2 SHARED RUNNER UNIT TEST GATE: PASS


## 24. Lock candidate / fold-result schemas


```python
CANDIDATE_IDENTITY_FIELDS = list(
    CANONICAL_CANDIDATE_IDENTITY_FIELDS
)

FOLD_RESULT_FIELDS = list(
    CANONICAL_FOLD_RESULT_FIELDS
)

AGGREGATE_FIELDS = list(
    CANONICAL_AGGREGATE_FIELDS
)


assert "F1_fraud" in FOLD_RESULT_FIELDS
assert "Recall_fraud" in FOLD_RESULT_FIELDS
assert "Precision_fraud" in FOLD_RESULT_FIELDS
assert "Accuracy_reference" in FOLD_RESULT_FIELDS

assert "mean_F1" in AGGREGATE_FIELDS
assert "std_F1" in AGGREGATE_FIELDS
assert "mean_Recall" in AGGREGATE_FIELDS
assert "mean_Precision" in AGGREGATE_FIELDS

assert "f1_fraud" not in FOLD_RESULT_FIELDS
assert "mean_f1" not in AGGREGATE_FIELDS


validate_candidate_identity(
    toy_candidate
)

validate_fold_identity(
    toy_fold_identity,
    X_toy_train,
    y_toy_train,
    X_toy_validation,
    y_toy_validation,
)

validate_fold_result_schema(
    toy_result[
        "fold_result"
    ]
)


print(
    "Candidate identity fields:",
    len(
        CANDIDATE_IDENTITY_FIELDS
    ),
)

print(
    "Fold-result fields:",
    len(
        FOLD_RESULT_FIELDS
    ),
)

print(
    "Aggregate fields:",
    len(
        AGGREGATE_FIELDS
    ),
)

print(
    "Canonical metric fields:"
)

print(
    "F1_fraud / Recall_fraud / "
    "Precision_fraud / Accuracy_reference"
)

print(
    "Canonical aggregate fields:"
)

print(
    "mean_F1 / std_F1 / "
    "mean_Recall / mean_Precision"
)

print(
    "\nM7.2 RESULT SCHEMA GATE: PASS"
)
```

    Candidate identity fields: 19
    Fold-result fields: 35
    Aggregate fields: 18
    Canonical metric fields:
    F1_fraud / Recall_fraud / Precision_fraud / Accuracy_reference
    Canonical aggregate fields:
    mean_F1 / std_F1 / mean_Recall / mean_Precision
    
    M7.2 RESULT SCHEMA GATE: PASS


## 25. Build infrastructure audit records


```python

preprocessing_state_audit = []


for key, bundle in (
    fold_bundles.items()
):
    window_id, fold_id = key

    record = (
        fold_registry_by_key[
            key
        ]
    )

    scaler_seen = np.asarray(
        bundle
        .scaler
        .n_samples_seen_,
        dtype=np.int64,
    )

    preprocessing_state_audit.append(
        {
            "state_id":
                bundle.state_id,

            "training_window_id":
                window_id,

            "fold_id":
                fold_id,

            "fit_row_count":
                int(
                    bundle.fit_row_count
                ),

            "expected_fold_train_rows":
                int(
                    record[
                        "train_rows"
                    ]
                ),

            "fit_min_timestamp":
                str(
                    bundle
                    .fit_min_timestamp
                ),

            "fit_max_timestamp":
                str(
                    bundle
                    .fit_max_timestamp
                ),

            "validation_start":
                record[
                    "validation_start"
                ],

            "numeric_mean":
                [
                    float(
                        value
                    )
                    for value
                    in bundle
                    .scaler
                    .mean_
                ],

            "numeric_scale":
                [
                    float(
                        value
                    )
                    for value
                    in bundle
                    .scaler
                    .scale_
                ],

            "numeric_observed_rows":
                [
                    int(
                        value
                    )
                    for value
                    in scaler_seen
                ],

            "category_vocab_size": {
                column:
                    int(
                        len(
                            bundle
                            .category_vocab[
                                column
                            ]
                        )
                    )
                for column
                in CATEGORICAL_COLUMNS
            },

            "feature_count":
                int(
                    len(
                        bundle
                        .feature_names
                    )
                ),

            "feature_names_match_m4_7":
                (
                    bundle.feature_names
                    == canonical_feature_names
                ),

            "fit_source":
                "FOLD_TRAIN_ONLY",

            "validation_fit_used":
                False,

            "status":
                "PASS",
        }
    )


assert len(
    preprocessing_state_audit
) == 6


print(
    "Preprocessing states audited:",
    len(
        preprocessing_state_audit
    ),
)

print(
    "M7.2 PREPROCESSING STATE AUDIT BUILD: PASS"
)

```

    Preprocessing states audited: 6
    M7.2 PREPROCESSING STATE AUDIT BUILD: PASS


## 26. Persist M7.2 infrastructure contract


```python

CONTRACT_PATH = (
    OUTPUT_DIR
    / "m7_02_cv_infrastructure_contract.json"
)

PREPROCESSING_AUDIT_PATH = (
    OUTPUT_DIR
    / "m7_02_preprocessing_state_audit.json"
)

MANIFEST_PATH = (
    OUTPUT_DIR
    / "m7_02_infrastructure_manifest.json"
)


contract_payload = {
    "m7_substep":
        "M7.2",

    "runner_version":
        M7_RUNNER_VERSION,

    "cv_spec_id":
        M7_CV_SPEC_ID,

    "fold_template_id":
        M7_FOLD_TEMPLATE_ID,

    "metric_contract_version":
        M7_METRIC_CONTRACT_VERSION,

    "random_state":
        RANDOM_STATE,

    "primary_cv":
        "FORWARD_EXPANDING_TEMPORAL",

    "fold_specs": [
        {
            key:
                (
                    str(
                        value
                    )
                    if isinstance(
                        value,
                        pd.Timestamp,
                    )
                    else value
                )
            for key, value
            in fold.items()
        }
        for fold
        in FOLD_SPECS
    ],

    "window_starts": {
        key:
            str(
                value
            )
        for key, value
        in WINDOW_STARTS.items()
    },

    "source_artifacts": {
        "m4_manifest":
            str(
                M4_MANIFEST_PATH
                .relative_to(
                    PROJECT_ROOT
                )
            ),

        "m4_manifest_sha256":
            sha256_file(
                M4_MANIFEST_PATH
            ),

        "m4_feature_names":
            str(
                M4_FEATURE_NAMES_PATH
                .relative_to(
                    PROJECT_ROOT
                )
            ),

        "m4_feature_names_sha256":
            sha256_file(
                M4_FEATURE_NAMES_PATH
            ),

        "row_id_train_w_long_sha256":
            sha256_file(
                ROW_LONG_PATH
            ),

        "row_id_train_w_short_sha256":
            sha256_file(
                ROW_SHORT_PATH
            ),

        "y_train_w_long_sha256":
            sha256_file(
                Y_LONG_PATH
            ),

        "y_train_w_short_sha256":
            sha256_file(
                Y_SHORT_PATH
            ),

        "m6_02_registry_sha256":
            sha256_file(
                M6_02_REGISTRY_PATH
            ),
    },

    "m4_full_year_transformed_matrices": {
        "canonical_baseline_artifacts":
            True,

        "safe_for_direct_temporal_cv_slicing":
            False,

        "reason":
            (
                "M4.7 learned preprocessing state "
                "extends through 2018-12-31; "
                "fold-local refit is required."
            ),
    },

    "fold_registry":
        fold_registry,

    "preprocessing_contract": {
        "numeric":
            "FeaturewiseStandardScaler",

        "numeric_fit_source":
            "FOLD_TRAIN_ONLY_FINITE_VALUES",

        "boolean":
            "PASSTHROUGH_FLOAT32",

        "categorical":
            "OneHotEncoder",

        "categorical_vocab_source":
            "FOLD_TRAIN_ONLY",

        "unknown_token":
            UNKNOWN_TOKEN,

        "validation":
            "TRANSFORM_ONLY",

        "output_format":
            "CSR",

        "output_dtype":
            "float32",

        "output_width":
            EXPECTED_OUTPUT_WIDTH,

        "feature_names_match_m4_7":
            True,
    },

    "sampler_contract": {
        "placement":
            "AFTER_FOLD_PREPROCESSING_BEFORE_MODEL_FIT",

        "fit_resample_source":
            "FOLD_TRAIN_ONLY",

        "validation_resampling":
            False,

        "m7_02_real_resampling_experiment":
            False,
    },

    "runner_contract": {
        "runner_version":
            M7_RUNNER_VERSION,

        "risk_score_when_supported":
            True,

        "metric_bundle": [
            "F1_fraud",
            "Recall_fraud",
            "Precision_fraud",
            "TP",
            "FP",
            "FN",
            "TN",
            "predicted_positive_count",
            "predicted_positive_rate",
            "Accuracy_reference",
        ],

        "generic_sampler_compatibility":
            "PASS",

        "warning_capture":
            "PASS",

        "positive_class_probability_mapping":
            "PASS",

        "candidate_schema_enforcement":
            "PASS",

        "fold_result_schema_enforcement":
            "PASS",

        "synthetic_unit_test":
            "PASS",
    },

    "candidate_identity_fields":
        CANDIDATE_IDENTITY_FIELDS,

    "fold_result_fields":
        FOLD_RESULT_FIELDS,

    "aggregate_fields":
        AGGREGATE_FIELDS,

    "selection_state": {
        "training_window_winner":
            "OPEN",

        "model_family_winner":
            "OPEN",

        "final_model":
            "OPEN",

        "final_imbalance_strategy":
            "OPEN",

        "final_threshold":
            "OPEN",

        "m7_02_selection_authorized":
            False,
    },

    "final_test_accessed":
        False,
}


with open(
    CONTRACT_PATH,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        contract_payload,
        file,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )


with open(
    PREPROCESSING_AUDIT_PATH,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        preprocessing_state_audit,
        file,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )


manifest_payload = {
    "m7_substep":
        "M7.2",

    "runner_version":
        M7_RUNNER_VERSION,

    "cv_spec_id":
        M7_CV_SPEC_ID,

    "fold_states_expected":
        6,

    "fold_states_built":
        len(
            fold_bundles
        ),

    "fold_registry_records":
        len(
            fold_registry
        ),

    "row_timestamp_lineage":
        "PASS",

    "fold_order_overlap_positive_support":
        "PASS",

    "fold_safe_preprocessing":
        "PASS",

    "canonical_feature_schema":
        "PASS",

    "transform_only_validation":
        "PASS",

    "unknown_category_route":
        "PASS",

    "sampler_placement_contract":
        "PASS",

    "shared_runner_unit_test":
        "PASS",

    "generic_sampler_compatibility":
        "PASS",

    "warning_capture":
        "PASS",

    "positive_class_probability_mapping":
        "PASS",

    "canonical_schema_enforcement":
        "PASS",

    "real_model_selection_performed":
        False,

    "threshold_optimization_performed":
        False,

    "final_test_accessed":
        False,

    "decision":
        "OPEN — REQUIRES M7.2 RUNTIME REVIEW",
}


with open(
    MANIFEST_PATH,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        manifest_payload,
        file,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )


for path in [
    CONTRACT_PATH,
    PREPROCESSING_AUDIT_PATH,
    MANIFEST_PATH,
]:
    assert path.exists()
    assert path.stat().st_size > 0


print("Contract:")
print(CONTRACT_PATH)

print("\nPreprocessing audit:")
print(PREPROCESSING_AUDIT_PATH)

print("\nManifest:")
print(MANIFEST_PATH)

print(
    "\nM7.2 INFRASTRUCTURE PERSISTENCE GATE: PASS"
)

```

    Contract:
    /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/processed/m7_02_temporal_cv_infrastructure_audit/m7_02_cv_infrastructure_contract.json
    
    Preprocessing audit:
    /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/processed/m7_02_temporal_cv_infrastructure_audit/m7_02_preprocessing_state_audit.json
    
    Manifest:
    /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/processed/m7_02_temporal_cv_infrastructure_audit/m7_02_infrastructure_manifest.json
    
    M7.2 INFRASTRUCTURE PERSISTENCE GATE: PASS


## 27. Persistence round-trip


```python

with open(
    CONTRACT_PATH,
    "r",
    encoding="utf-8",
) as file:
    contract_roundtrip = json.load(
        file
    )


with open(
    PREPROCESSING_AUDIT_PATH,
    "r",
    encoding="utf-8",
) as file:
    preprocessing_roundtrip = json.load(
        file
    )


with open(
    MANIFEST_PATH,
    "r",
    encoding="utf-8",
) as file:
    manifest_roundtrip = json.load(
        file
    )


assert (
    contract_roundtrip[
        "runner_version"
    ]
    == M7_RUNNER_VERSION
)

assert (
    contract_roundtrip[
        "cv_spec_id"
    ]
    == M7_CV_SPEC_ID
)

assert (
    len(
        contract_roundtrip[
            "fold_registry"
        ]
    )
    == 6
)

assert (
    contract_roundtrip[
        "m4_full_year_transformed_matrices"
    ][
        "safe_for_direct_temporal_cv_slicing"
    ]
    is False
)

assert (
    contract_roundtrip[
        "preprocessing_contract"
    ][
        "validation"
    ]
    == "TRANSFORM_ONLY"
)

assert (
    contract_roundtrip[
        "sampler_contract"
    ][
        "validation_resampling"
    ]
    is False
)

assert (
    contract_roundtrip[
        "selection_state"
    ][
        "final_model"
    ]
    == "OPEN"
)

assert (
    contract_roundtrip[
        "selection_state"
    ][
        "m7_02_selection_authorized"
    ]
    is False
)

assert (
    contract_roundtrip[
        "final_test_accessed"
    ]
    is False
)


assert len(
    preprocessing_roundtrip
) == 6

assert (
    manifest_roundtrip[
        "fold_states_built"
    ]
    == 6
)

assert (
    manifest_roundtrip[
        "real_model_selection_performed"
    ]
    is False
)

assert (
    manifest_roundtrip[
        "final_test_accessed"
    ]
    is False
)


assert (
    contract_roundtrip[
        "runner_contract"
    ][
        "generic_sampler_compatibility"
    ]
    == "PASS"
)

assert (
    contract_roundtrip[
        "runner_contract"
    ][
        "warning_capture"
    ]
    == "PASS"
)

assert (
    contract_roundtrip[
        "runner_contract"
    ][
        "positive_class_probability_mapping"
    ]
    == "PASS"
)

assert (
    contract_roundtrip[
        "runner_contract"
    ][
        "candidate_schema_enforcement"
    ]
    == "PASS"
)

assert (
    contract_roundtrip[
        "runner_contract"
    ][
        "fold_result_schema_enforcement"
    ]
    == "PASS"
)

assert (
    manifest_roundtrip[
        "generic_sampler_compatibility"
    ]
    == "PASS"
)

assert (
    manifest_roundtrip[
        "warning_capture"
    ]
    == "PASS"
)

assert (
    manifest_roundtrip[
        "positive_class_probability_mapping"
    ]
    == "PASS"
)

assert (
    manifest_roundtrip[
        "canonical_schema_enforcement"
    ]
    == "PASS"
)


print(
    "M7.2 INFRASTRUCTURE ROUND-TRIP GATE: PASS"
)

```

    M7.2 INFRASTRUCTURE ROUND-TRIP GATE: PASS


## 28. FINAL TEST / selection-boundary gate


```python

selection_state = (
    contract_payload[
        "selection_state"
    ]
)


assert (
    selection_state[
        "training_window_winner"
    ]
    == "OPEN"
)

assert (
    selection_state[
        "model_family_winner"
    ]
    == "OPEN"
)

assert (
    selection_state[
        "final_model"
    ]
    == "OPEN"
)

assert (
    selection_state[
        "final_imbalance_strategy"
    ]
    == "OPEN"
)

assert (
    selection_state[
        "final_threshold"
    ]
    == "OPEN"
)

assert (
    selection_state[
        "m7_02_selection_authorized"
    ]
    is False
)

assert (
    "Is Fraud?"
    not in SEMANTIC_USECOLS
)

assert (
    "Errors?"
    not in SEMANTIC_USECOLS
)

assert (
    contract_payload[
        "final_test_accessed"
    ]
    is False
)


print(
    "Training-window winner:",
    selection_state[
        "training_window_winner"
    ],
)

print(
    "Model-family winner:",
    selection_state[
        "model_family_winner"
    ],
)

print(
    "Final model:",
    selection_state[
        "final_model"
    ],
)

print(
    "Final threshold:",
    selection_state[
        "final_threshold"
    ],
)

print(
    "\nM7.2 FINAL TEST / SELECTION-BOUNDARY GATE: PASS"
)

```

    Training-window winner: OPEN
    Model-family winner: OPEN
    Final model: OPEN
    Final threshold: OPEN
    
    M7.2 FINAL TEST / SELECTION-BOUNDARY GATE: PASS


## 29. M7.2 overall technical gate


```python
m7_02_gates = {
    "G01_SOURCE_LOCATION": True,
    "G02_M4_M6_HANDOFF": True,
    "G03_TRAIN_LINEAGE": True,
    "G04_FULL_YEAR_MATRIX_CV_SAFETY": True,
    "G05_FOLD_SPECIFICATION": True,
    "G06_FEATURE_CONTRACT": True,
    "G07_STRICT_CAUSAL_REGRESSION": True,
    "G08_FULL_RAW_FIT_PASS": True,
    "G09_ROW_TIMESTAMP_LINEAGE": True,
    "G10_FOLD_ORDER_OVERLAP_POSITIVE_SUPPORT": True,
    "G11_FOLD_TRAIN_PREPROCESSING_FIT_SOURCE": True,
    "G12_FOLD_LOCAL_BUNDLE_FREEZE": True,
    "G13_TRANSFORM_ONLY_SCHEMA": True,
    "G14_SAMPLER_PLACEMENT": True,
    "G15_GENERIC_SAMPLER_COMPATIBILITY": True,
    "G16_WARNING_CAPTURE": True,
    "G17_POSITIVE_CLASS_PROBABILITY_MAPPING": True,
    "G18_CANONICAL_SCHEMA_ENFORCEMENT": True,
    "G19_SHARED_RUNNER_UNIT_TEST": True,
    "G20_RESULT_SCHEMA": True,
    "G21_INFRASTRUCTURE_PERSISTENCE": True,
    "G22_INFRASTRUCTURE_ROUND_TRIP": True,
    "G23_FINAL_TEST_SELECTION_BOUNDARY": True,
}


for gate_name, gate_value in (
    m7_02_gates.items()
):
    print(
        gate_name,
        "→",
        (
            "PASS"
            if gate_value
            else "FAIL"
        ),
    )


assert all(
    m7_02_gates.values()
)

assert len(
    m7_02_gates
) == 23


print(
    "\nM7.2 OVERALL TECHNICAL GATE: PASS"
)
```

    G01_SOURCE_LOCATION → PASS
    G02_M4_M6_HANDOFF → PASS
    G03_TRAIN_LINEAGE → PASS
    G04_FULL_YEAR_MATRIX_CV_SAFETY → PASS
    G05_FOLD_SPECIFICATION → PASS
    G06_FEATURE_CONTRACT → PASS
    G07_STRICT_CAUSAL_REGRESSION → PASS
    G08_FULL_RAW_FIT_PASS → PASS
    G09_ROW_TIMESTAMP_LINEAGE → PASS
    G10_FOLD_ORDER_OVERLAP_POSITIVE_SUPPORT → PASS
    G11_FOLD_TRAIN_PREPROCESSING_FIT_SOURCE → PASS
    G12_FOLD_LOCAL_BUNDLE_FREEZE → PASS
    G13_TRANSFORM_ONLY_SCHEMA → PASS
    G14_SAMPLER_PLACEMENT → PASS
    G15_GENERIC_SAMPLER_COMPATIBILITY → PASS
    G16_WARNING_CAPTURE → PASS
    G17_POSITIVE_CLASS_PROBABILITY_MAPPING → PASS
    G18_CANONICAL_SCHEMA_ENFORCEMENT → PASS
    G19_SHARED_RUNNER_UNIT_TEST → PASS
    G20_RESULT_SCHEMA → PASS
    G21_INFRASTRUCTURE_PERSISTENCE → PASS
    G22_INFRASTRUCTURE_ROUND_TRIP → PASS
    G23_FINAL_TEST_SELECTION_BOUNDARY → PASS
    
    M7.2 OVERALL TECHNICAL GATE: PASS


# 30. Runtime review và Findings M7.2

## 30.1. Execution integrity

Observed:

```text
Code cells:
28 / 28

Execution count:
1 → 28 liên tục

Runtime errors:
0

stderr:
0
```

Runtime environment:

```text
Python:
3.14.6

NumPy:
2.5.3

pandas:
3.0.5

Platform:
macOS arm64
```

Interpretation:

Notebook đã chạy đầy đủ từ đầu đến cuối.

Không có partial execution, stale cell hoặc runtime exception.

Status:

`VERIFIED`

---

## 30.2. M4 / M6 handoff integrity

Observed:

```text
M4 pipeline:
M4.7-baseline-v1

Canonical output width:
47

M6 official runs:
6
```

Runtime gate:

`M7.2 M4/M6 HANDOFF GATE: PASS`

Status:

`VERIFIED`

---

## 30.3. M4.7 full-year matrices không safe cho direct temporal-CV slicing

Observed:

```text
M4.7 W_LONG preprocessing fit end:
2018-12-31 23:58:00

M4.7 W_SHORT preprocessing fit end:
2018-12-31 23:58:00
```

Trong khi temporal folds kết thúc train tại:

```text
Q2 fold:
2018-04-01

Q3 fold:
2018-07-01

Q4 fold:
2018-10-01
```

Runtime conclusion:

```text
Direct slicing of M4.7 transformed TRAIN matrices:
PROHIBITED

Reason:
preprocessing state sees future fold rows
```

Status:

`VERIFIED`

---

## 30.4. Full raw rebuild

Observed:

```text
Card blocks:
6,139

Raw rows streamed:
24,386,900

Elapsed seconds:
187.17
```

Runtime gate:

`M7.2 FULL RAW PREPROCESSING FIT PASS: COMPLETE`

Interpretation:

Fold-local preprocessing states được build từ canonical raw semantics thay vì reuse future-informed transformed matrices.

Status:

`VERIFIED`

---

## 30.5. Exact TRAIN row / Timestamp lineage

Observed:

```text
W_LONG exact row/Timestamp mapping:
6,855,270

W_SHORT exact subset mapping:
1,721,615
```

Runtime gate:

`M7.2 ROW/TIMESTAMP LINEAGE GATE: PASS`

Interpretation:

Canonical M4.7 TRAIN row IDs đã map exact về Timestamp cần cho temporal fold membership.

Status:

`VERIFIED`

---

## 30.6. Fold population và positive support

Observed:

```text
W_LONG / Q2

train:
5,557,560 rows
7,672 fraud

validation:
428,953 rows
590 fraud
```

```text
W_LONG / Q3

train:
5,986,513 rows
8,262 fraud

validation:
435,178 rows
634 fraud
```

```text
W_LONG / Q4

train:
6,421,691 rows
8,896 fraud

validation:
433,579 rows
710 fraud
```

```text
W_SHORT / Q2

train:
423,905 rows
557 fraud

validation:
428,953 rows
590 fraud
```

```text
W_SHORT / Q3

train:
852,858 rows
1,147 fraud

validation:
435,178 rows
634 fraud
```

```text
W_SHORT / Q4

train:
1,288,036 rows
1,781 fraud

validation:
433,579 rows
710 fraud
```

All six states satisfy:

```text
train rows > 0
validation rows > 0
train fraud > 0
validation fraud > 0
```

Status:

`VERIFIED`

---

## 30.7. Expanding-window arithmetic consistency

W_SHORT:

```text
Q3 train rows:
423,905 + 428,953
= 852,858

Q3 train fraud:
557 + 590
= 1,147
```

```text
Q4 train rows:
852,858 + 435,178
= 1,288,036

Q4 train fraud:
1,147 + 634
= 1,781
```

```text
Full W_SHORT rows:
1,288,036 + 433,579
= 1,721,615

Full W_SHORT fraud:
1,781 + 710
= 2,491
```

Interpretation:

Observed fold counts khớp expanding-window design và canonical W_SHORT totals.

Status:

`VERIFIED`

---

## 30.8. Temporal order và overlap

Runtime gate:

`M7.2 FOLD ORDER / OVERLAP / POSITIVE SUPPORT GATE: PASS`

Required conditions đã PASS cho cả 6 states:

```text
max(train Timestamp)
<
min(validation Timestamp)

train / validation row overlap:
0
```

Status:

`VERIFIED`

---

## 30.9. Fold-local preprocessing fit source

Observed:

```text
W_LONG Q2:
5,557,560 fit rows
2015-01-01 00:01
→
2018-03-31 23:59

W_LONG Q3:
5,986,513 fit rows
2015-01-01 00:01
→
2018-06-30 23:59

W_LONG Q4:
6,421,691 fit rows
2015-01-01 00:01
→
2018-09-30 23:57
```

```text
W_SHORT Q2:
423,905 fit rows
2018-01-01 00:03
→
2018-03-31 23:59

W_SHORT Q3:
852,858 fit rows
2018-01-01 00:03
→
2018-06-30 23:59

W_SHORT Q4:
1,288,036 fit rows
2018-01-01 00:03
→
2018-09-30 23:57
```

Mỗi state có:

```text
fit row count
==
canonical fold-training row count
```

và:

```text
fit max Timestamp
<
fold validation start
```

Runtime gate:

`M7.2 FOLD-TRAIN PREPROCESSING FIT-SOURCE GATE: PASS`

Status:

`VERIFIED`

---

## 30.10. Canonical 47-column schema

Observed:

```text
Fold-local bundles:
6

All feature schemas:
exact M4.7 47-column match
```

Runtime gate:

`M7.2 FOLD-LOCAL BUNDLE FREEZE GATE: PASS`

Status:

`VERIFIED`

---

## 30.11. Transform-only validation

Observed audit samples:

```text
W_LONG Q2:
train sample 5,623
validation sample 428

W_LONG Q3:
train sample 6,051
validation sample 424

W_LONG Q4:
train sample 6,475
validation sample 463

W_SHORT Q2:
train sample 429
validation sample 428

W_SHORT Q3:
train sample 857
validation sample 424

W_SHORT Q4:
train sample 1,281
validation sample 463
```

For all six states:

```text
output width:
47

matrix format:
CSR

dtype:
float32

validation transform mutates scaler:
NO

unknown-category route:
PASS

NaN / inf:
NONE
```

Runtime gate:

`M7.2 TRANSFORM-ONLY / SCHEMA GATE: PASS`

Status:

`VERIFIED`

---

## 30.12. Generic sampler compatibility

Observed:

`Generic sampler compatibility: PASS`

Runner tự ghi:

```text
sampler_input_rows
sampler_output_rows
```

Runner chỉ yêu cầu sampler interface:

`fit_resample(X, y)`.

Không phụ thuộc metadata riêng của một sampler implementation.

Runtime gate:

`M7.2 GENERIC SAMPLER COMPATIBILITY GATE: PASS`

Status:

`VERIFIED`

---

## 30.13. Warning capture

Observed:

`Warning capture: PASS`

Audit estimator cố ý phát warning và runner đã capture warning trong fold-result contract.

Runtime gate:

`M7.2 WARNING CAPTURE GATE: PASS`

Lưu ý:

Toy Logistic Regression fold-result có:

```text
warning_count:
0
```

đây là kết quả của toy LR bình thường.

Warning-capture behavior được kiểm chứng bằng audit estimator riêng.

Status:

`VERIFIED`

---

## 30.14. Positive-class probability mapping

Observed:

`Positive-class probability mapping: PASS`

Audit estimator dùng:

```text
classes_ = [1, 0]
```

Runner vẫn lấy đúng fraud risk score từ class:

`1`

thay vì hard-code probability column thứ hai.

Runtime gate:

`M7.2 POSITIVE-CLASS PROBABILITY MAPPING GATE: PASS`

Status:

`VERIFIED`

---

## 30.15. Canonical candidate / fold-result schema enforcement

Observed:

```text
Candidate identity fields:
19

Fold-result fields:
35

Aggregate fields:
18
```

Canonical metric fields:

```text
F1_fraud
Recall_fraud
Precision_fraud
Accuracy_reference
```

Canonical aggregate fields:

```text
mean_F1
std_F1
mean_Recall
mean_Precision
```

Runtime also verified:

```text
candidate missing required field
→ REJECT

fold-result exact schema
→ PASS
```

Runtime gate:

`M7.2 CANONICAL SCHEMA ENFORCEMENT GATE: PASS`

Status:

`VERIFIED`

---

## 30.16. Shared M7 runner

Observed runtime gates:

```text
Generic sampler compatibility:
PASS

Warning capture:
PASS

Positive-class mapping:
PASS

Candidate schema enforcement:
PASS

Fold-result schema enforcement:
PASS

Synthetic shared-runner unit test:
PASS
```

Interpretation:

Shared runner đã đủ contract để handoff sang real temporal-CV experiments.

Status:

`LOCKABLE`

---

## 30.17. Persistence / round-trip

Persisted:

```text
data/processed/m7_02_temporal_cv_infrastructure_audit/
    m7_02_cv_infrastructure_contract.json
    m7_02_preprocessing_state_audit.json
    m7_02_infrastructure_manifest.json
```

Runtime gates:

```text
M7.2 INFRASTRUCTURE PERSISTENCE GATE:
PASS

M7.2 INFRASTRUCTURE ROUND-TRIP GATE:
PASS
```

Strengthened runner-contract fields cũng survive round-trip.

Status:

`VERIFIED`

---

## 30.18. Selection boundary / FINAL TEST

Observed:

```text
Training-window winner:
OPEN

Model-family winner:
OPEN

Final model:
OPEN

Final threshold:
OPEN
```

Contract additionally keeps:

```text
Final imbalance strategy:
OPEN

M7.2 selection authorized:
False

FINAL TEST:
NO ACCESS
```

Runtime gate:

`M7.2 FINAL TEST / SELECTION-BOUNDARY GATE: PASS`

Status:

`VERIFIED`

---

## 30.19. Overall technical result

Observed:

```text
G01_SOURCE_LOCATION                         → PASS
G02_M4_M6_HANDOFF                           → PASS
G03_TRAIN_LINEAGE                           → PASS
G04_FULL_YEAR_MATRIX_CV_SAFETY              → PASS
G05_FOLD_SPECIFICATION                      → PASS
G06_FEATURE_CONTRACT                        → PASS
G07_STRICT_CAUSAL_REGRESSION                → PASS
G08_FULL_RAW_FIT_PASS                       → PASS
G09_ROW_TIMESTAMP_LINEAGE                   → PASS
G10_FOLD_ORDER_OVERLAP_POSITIVE_SUPPORT     → PASS
G11_FOLD_TRAIN_PREPROCESSING_FIT_SOURCE     → PASS
G12_FOLD_LOCAL_BUNDLE_FREEZE                → PASS
G13_TRANSFORM_ONLY_SCHEMA                    → PASS
G14_SAMPLER_PLACEMENT                       → PASS
G15_GENERIC_SAMPLER_COMPATIBILITY            → PASS
G16_WARNING_CAPTURE                          → PASS
G17_POSITIVE_CLASS_PROBABILITY_MAPPING      → PASS
G18_CANONICAL_SCHEMA_ENFORCEMENT             → PASS
G19_SHARED_RUNNER_UNIT_TEST                  → PASS
G20_RESULT_SCHEMA                            → PASS
G21_INFRASTRUCTURE_PERSISTENCE               → PASS
G22_INFRASTRUCTURE_ROUND_TRIP                → PASS
G23_FINAL_TEST_SELECTION_BOUNDARY            → PASS
```

Overall:

`M7.2 OVERALL TECHNICAL GATE: PASS`

Blocking issue:

`NONE`

---

# 31. Findings M7.2

## M7.2-F01 — M4.7 full-year transformed matrices không thể dùng trực tiếp cho temporal CV

Reason:

learned preprocessing state nhìn đến cuối 2018.

Status:

`VERIFIED INFRASTRUCTURE FINDING`

---

## M7.2-F02 — Canonical raw-row lineage đủ để reconstruct temporal folds

Observed:

```text
W_LONG:
6,855,270 exact rows

W_SHORT:
1,721,615 exact rows
```

Status:

`VERIFIED`

---

## M7.2-F03 — Q2/Q3/Q4-2018 fold template có positive support đầy đủ

Observed:

`6 / 6 fold states`

đều có fraud trong train và validation.

Status:

`VERIFIED`

---

## M7.2-F04 — Expanding temporal order hoạt động đúng

Observed:

- train end luôn trước validation start;
- row overlap bằng 0;
- W_SHORT fold arithmetic khớp full canonical totals.

Status:

`VERIFIED`

---

## M7.2-F05 — Fold-safe preprocessing có thể tái tạo canonical feature schema

Observed:

```text
6 / 6 fold-local bundles

feature width:
47

feature names:
exact M4.7 match
```

Status:

`VERIFIED`

---

## M7.2-F06 — Validation transform không làm thay đổi learned preprocessing state

Observed:

`TRANSFORM ONLY`

scaler state unchanged.

Status:

`VERIFIED`

---

## M7.2-F07 — Unknown-category path hoạt động

Observed:

explicit `__UNKNOWN__` route PASS.

Status:

`VERIFIED`

---

## M7.2-F08 — Strict-causal behavioral semantics tiếp tục hợp lệ

Observed:

strict-causal regression gate PASS.

Status:

`VERIFIED`

---

## M7.2-F09 — Shared runner tương thích generic sampler contract

Required sampler interface:

`fit_resample(X, y)`.

Status:

`VERIFIED`

---

## M7.2-F10 — Warning evidence có thể được giữ trong fold-result registry

Observed:

warning-capture audit PASS.

Status:

`VERIFIED`

---

## M7.2-F11 — Fraud probability mapping độc lập với class-column order

Observed:

audit `classes_ = [1, 0]` vẫn map đúng positive class `1`.

Status:

`VERIFIED`

---

## M7.2-F12 — Candidate/result schema được enforce trước khi downstream selection

Observed:

- missing candidate field bị reject;
- fold-result exact schema PASS;
- canonical case-sensitive field names được giữ.

Status:

`VERIFIED`

---

## M7.2-F13 — M7.2 không tạo selection evidence

Observed:

```text
Training-window winner:
OPEN

Model-family winner:
OPEN

Final model:
OPEN

Final imbalance strategy:
OPEN

Final threshold:
OPEN
```

Status:

`BOUNDARY PRESERVED`

---

## M7.2-F14 — FINAL TEST vẫn được bảo vệ

Observed:

`FINAL TEST ACCESS = NO`

Status:

`VERIFIED`

---

## M7.2-F15 — Temporal-CV infrastructure đã đủ để chạy M7.3

Necessary components đã verified:

- temporal row lineage;
- fold definitions;
- positive support;
- fold-safe preprocessing;
- canonical 47-column output;
- sampler boundary;
- runner;
- metric contract;
- warning capture;
- probability mapping;
- persistence.

Status:

`READY FOR M7.3`

# 32. Decision Log M7.2 — sau runtime review

## M7.2-D01 — M4.7 transformed matrices

Decision:

Canonical M4.7 transformed matrices tiếp tục là official baseline artifacts cho M5/M6.

Nhưng:

`DO NOT SLICE DIRECTLY FOR TEMPORAL CV`.

Status:

`LOCKED / VERIFIED`

---

## M7.2-D02 — Temporal-CV source path

Decision:

Temporal CV rebuild semantic rows và fit fold-local preprocessing state từ canonical raw + M4 semantics.

Status:

`LOCKED / VERIFIED`

---

## M7.2-D03 — Fold template

Decision:

```text
Q2 / Q3 / Q4 2018
FORWARD / EXPANDING
```

Status:

`INHERITED — VERIFIED — LOCKED`

---

## M7.2-D04 — Row lineage

Decision:

M4.7 raw-row IDs là canonical training lineage.

Observed:

exact W_LONG / W_SHORT mapping PASS.

Status:

`LOCKED / VERIFIED`

---

## M7.2-D05 — Target source

Decision:

Use canonical M4.7 `y_train_*` arrays sau exact row alignment.

Raw `Is Fraud?` không được đọc trong semantic preprocessing pass.

Status:

`LOCKED / VERIFIED`

---

## M7.2-D06 — Behavioral semantics

Decision:

Reuse strict-causal M4.5/M4.7 semantics.

Observed:

regression gate PASS.

Status:

`INHERITED — VERIFIED — LOCKED`

---

## M7.2-D07 — Numeric preprocessing

Decision:

Featurewise scaler fit từ:

`FOLD-TRAIN ONLY`.

Observed:

6 / 6 states PASS.

Status:

`LOCKED / VERIFIED`

---

## M7.2-D08 — Categorical preprocessing

Decision:

Vocabulary fit từ:

`FOLD-TRAIN ONLY`

và explicit:

`__UNKNOWN__`.

Observed:

unknown route PASS.

Status:

`LOCKED / VERIFIED`

---

## M7.2-D09 — Fold-validation

Decision:

`TRANSFORM ONLY`.

Observed:

scaler state unchanged after validation transform.

Status:

`LOCKED / VERIFIED`

---

## M7.2-D10 — Output schema

Decision:

M7 temporal-CV preprocessing phải giữ:

```text
M4.7 canonical feature order
47 columns
CSR
float32
```

Observed:

6 / 6 bundles PASS.

Status:

`LOCKED / VERIFIED`

---

## M7.2-D11 — Sampler placement

Decision:

Sampler nằm:

```text
after fold preprocessing
before estimator fit
fold-TRAIN only
```

Validation resampling:

`NO`.

Status:

`LOCKED / VERIFIED`

---

## M7.2-D12 — Generic sampler interface

Decision:

Shared runner chỉ phụ thuộc:

`fit_resample(X, y)`.

Runner tự ghi input/output row counts.

Observed:

runtime audit PASS.

Status:

`LOCKED / VERIFIED`

---

## M7.2-D13 — Warning capture

Decision:

Warnings từ sampler / fit / predict / probability path phải được persist trong fold-result.

Observed:

runtime warning audit PASS.

Status:

`LOCKED / VERIFIED`

---

## M7.2-D14 — Positive-class probability

Decision:

Fraud risk score lấy theo:

`fitted.classes_ == 1`.

Không hard-code probability column.

Observed:

reversed-class audit PASS.

Status:

`LOCKED / VERIFIED`

---

## M7.2-D15 — Candidate identity

Decision:

19-field immutable candidate identity theo M7.1.

Observed:

missing required field bị reject.

Status:

`LOCKED / VERIFIED`

---

## M7.2-D16 — Fold-result schema

Decision:

35-field canonical fold-result schema.

Case-sensitive metric names:

```text
F1_fraud
Recall_fraud
Precision_fraud
Accuracy_reference
```

Observed:

exact schema validation PASS.

Status:

`LOCKED / VERIFIED`

---

## M7.2-D17 — Aggregate schema

Decision:

18-field aggregate schema.

Case-sensitive aggregate names:

```text
mean_F1
std_F1
mean_Recall
mean_Precision
```

Status:

`LOCKED`

---

## M7.2-D18 — Shared M7 runner

Decision:

Lock:

`M7.2-shared-temporal-runner-v1`

Observed:

- generic sampler audit PASS;
- warning capture PASS;
- positive-class mapping PASS;
- schema enforcement PASS;
- synthetic unit test PASS.

Status:

`LOCKED / VERIFIED`

---

## M7.2-D19 — Infrastructure artifacts

Decision:

Persist:

```text
m7_02_cv_infrastructure_contract.json
m7_02_preprocessing_state_audit.json
m7_02_infrastructure_manifest.json
```

Observed:

persistence + round-trip PASS.

Status:

`LOCKED / VERIFIED`

---

## M7.2-D20 — Selection authority

Decision:

M7.2 không có authority để chọn:

- training window;
- model family;
- imbalance strategy;
- final model;
- threshold.

Observed:

all selection states OPEN.

Status:

`LOCKED / VERIFIED`

---

## M7.2-D21 — FINAL TEST

Decision:

No development access.

Observed:

selection-boundary gate PASS.

Status:

`INHERITED — VERIFIED — LOCKED`

---

## M7.2-D22 — M7.3 handoff

Decision:

M7.3 được phép bắt đầu controlled training-window robustness trên infrastructure đã verify.

Status:

`READY`

# 33. M7.2 Gate

## Technical runtime gates

```text
G01_SOURCE_LOCATION                         → PASS
G02_M4_M6_HANDOFF                           → PASS
G03_TRAIN_LINEAGE                           → PASS
G04_FULL_YEAR_MATRIX_CV_SAFETY              → PASS
G05_FOLD_SPECIFICATION                      → PASS
G06_FEATURE_CONTRACT                        → PASS
G07_STRICT_CAUSAL_REGRESSION                → PASS
G08_FULL_RAW_FIT_PASS                       → PASS
G09_ROW_TIMESTAMP_LINEAGE                   → PASS
G10_FOLD_ORDER_OVERLAP_POSITIVE_SUPPORT     → PASS
G11_FOLD_TRAIN_PREPROCESSING_FIT_SOURCE     → PASS
G12_FOLD_LOCAL_BUNDLE_FREEZE                → PASS
G13_TRANSFORM_ONLY_SCHEMA                    → PASS
G14_SAMPLER_PLACEMENT                       → PASS
G15_GENERIC_SAMPLER_COMPATIBILITY            → PASS
G16_WARNING_CAPTURE                          → PASS
G17_POSITIVE_CLASS_PROBABILITY_MAPPING      → PASS
G18_CANONICAL_SCHEMA_ENFORCEMENT             → PASS
G19_SHARED_RUNNER_UNIT_TEST                  → PASS
G20_RESULT_SCHEMA                            → PASS
G21_INFRASTRUCTURE_PERSISTENCE               → PASS
G22_INFRASTRUCTURE_ROUND_TRIP                → PASS
G23_FINAL_TEST_SELECTION_BOUNDARY            → PASS
```

Technical gates:

`23 / 23 PASS`

---

## R01 — Execution complete?

Evidence:

```text
28 / 28 code cells
execution_count = 1 → 28
errors = 0
stderr = 0
```

Result:

`PASS`

---

## R02 — M4/M6 handoff valid?

Evidence:

```text
M4.7-baseline-v1
47 canonical features
6 official M6 runs
```

Result:

`PASS`

---

## R03 — Direct M4.7 matrix slicing rejected?

Evidence:

M4 preprocessing state reaches:

`2018-12-31`.

Result:

`PASS`

---

## R04 — Full raw reconstruction complete?

Evidence:

```text
6,139 Card blocks
24,386,900 raw rows
```

Result:

`PASS`

---

## R05 — TRAIN row/Timestamp lineage exact?

Evidence:

```text
W_LONG:
6,855,270 exact

W_SHORT:
1,721,615 exact
```

Result:

`PASS`

---

## R06 — Fold positive support valid?

Evidence:

All six W_LONG/W_SHORT × Q2/Q3/Q4 states have positive train and validation support.

Result:

`PASS`

---

## R07 — Temporal order / overlap valid?

Evidence:

```text
max(train Timestamp)
<
min(validation Timestamp)

row overlap:
0
```

Result:

`PASS`

---

## R08 — Fold-local preprocessing fit source valid?

Evidence:

```text
fit rows
==
fold-training rows

fit max timestamp
<
validation start
```

for 6 / 6 states.

Result:

`PASS`

---

## R09 — Canonical 47-column output preserved?

Evidence:

```text
6 / 6 bundles
exact M4.7 feature names
CSR float32
```

Result:

`PASS`

---

## R10 — Validation transform-only valid?

Evidence:

- scaler state unchanged;
- no NaN/inf;
- explicit unknown route PASS.

Result:

`PASS`

---

## R11 — Generic sampler compatibility valid?

Evidence:

Runner requires only:

`fit_resample(X, y)`.

Result:

`PASS`

---

## R12 — Warning capture valid?

Evidence:

Intentional warning audit successfully captured.

Result:

`PASS`

---

## R13 — Positive-class probability mapping valid?

Evidence:

`classes_ = [1, 0]`

still maps fraud probability from class `1`.

Result:

`PASS`

---

## R14 — Canonical schema enforcement valid?

Evidence:

```text
candidate identity:
19 fields

fold result:
35 fields

aggregate schema:
18 fields
```

Missing candidate field rejection PASS.

Result:

`PASS`

---

## R15 — Shared M7 runner lockable?

Evidence:

All runner contract audits PASS.

Result:

`PASS`

---

## R16 — Persistence / round-trip valid?

Evidence:

3 infrastructure artifacts persisted and reloaded successfully.

Result:

`PASS`

---

## R17 — Selection boundary / FINAL TEST preserved?

Evidence:

```text
Training-window winner:
OPEN

Model-family winner:
OPEN

Final model:
OPEN

Final imbalance strategy:
OPEN

Final threshold:
OPEN

M7.2 selection authorized:
False

FINAL TEST:
NO ACCESS
```

Result:

`PASS`

---

## Overall M7.2 Gate

```text
Technical gates:
23 / 23 PASS

Runtime review gates:
17 / 17 PASS

Blocking issue:
NONE
```

Final:

`M7.2 — PASS`

Handoff:

`READY FOR M7.3`

# 34. Kết luận M7.2

M7.2 đã hoàn thành nhiệm vụ audit và khóa hạ tầng temporal-CV cho M7.

Kết quả quan trọng nhất:

```text
M4.7 full-year transformed matrices:
VALID BASELINE ARTIFACTS

Direct temporal-CV slicing:
PROHIBITED
```

Reason:

```text
M4.7 learned preprocessing state
extends through 2018-12-31
```

Temporal CV hiện có implementation path hợp lệ:

```text
canonical raw + M4 semantics
        ↓
exact row/Timestamp lineage
        ↓
Q2/Q3/Q4 expanding folds
        ↓
fold-local causal semantic rows
        ↓
fold-TRAIN-only scaler/vocabulary
        ↓
transform-only fold-validation
        ↓
canonical 47-column CSR float32
        ↓
optional TRAIN-only sampler
        ↓
shared M7 runner
        ↓
canonical fold-result schema
```

Verified fold states:

```text
W_LONG:
Q2 / Q3 / Q4

W_SHORT:
Q2 / Q3 / Q4

Total:
6 / 6
```

Infrastructure integrity:

```text
Temporal order:
VERIFIED

Row overlap:
NONE

Positive support:
VERIFIED

Strict-causal history:
VERIFIED

Fold-safe preprocessing:
VERIFIED

Unknown-category handling:
VERIFIED

Generic sampler compatibility:
VERIFIED

Warning capture:
VERIFIED

Positive-class probability mapping:
VERIFIED

Candidate schema:
LOCKED

Fold-result schema:
LOCKED

Shared M7 Runner:
LOCKED
```

Persisted artifacts:

```text
data/processed/m7_02_temporal_cv_infrastructure_audit/
    m7_02_cv_infrastructure_contract.json
    m7_02_preprocessing_state_audit.json
    m7_02_infrastructure_manifest.json
```

Selection boundary remains:

```text
Training-window Winner:
OPEN

Model-family Winner:
OPEN

Final Model:
OPEN

Final Imbalance Strategy:
OPEN

Final Threshold:
OPEN

Selection Winner:
NOT YET PRODUCED

FINAL TEST:
PROTECTED
```

Final state:

```text
M7.2 — PASS

Technical Gates:
23 / 23 PASS

Runtime Review Gates:
17 / 17 PASS

Temporal CV Infrastructure:
VERIFIED

Fold-safe Preprocessing:
VERIFIED

Shared M7 Runner:
LOCKED

Candidate Result Schema:
LOCKED

Blocking Issue:
NONE

READY FOR M7.3
```

Bước tiếp theo:

`M7.3 — Controlled training-window robustness`

Primary question:

> W_SHORT provisional direction từ M6 có còn ổn định qua Q2/Q3/Q4 temporal folds khi preprocessing được fit đúng từng fold hay không?

M7.3 phải giữ:

```text
same model/config
same feature semantics
same fold-safe preprocessing
same imbalance strategy
same threshold policy
same metric code
same random-state policy

variable:
training window only
```

FINAL TEST tiếp tục:

`PROTECTED`
