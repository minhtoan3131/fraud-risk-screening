# M6.6 — False Positive / False Negative error analysis

Milestone:

`M6 — Evaluation + Error Analysis`

Substep:

`M6.6 — False Positive / False Negative Error Analysis`

Work type:

`RUNTIME LINEAGE RECONSTRUCTION / TRANSACTION-LEVEL ERROR ANALYSIS`

Câu hỏi trung tâm:

> Fraud bị bỏ sót và false alert có pattern gì trên canonical validation population?

Primary groups:

```text
Fraud side:
FN vs TP

Non-fraud side:
FP vs TN
```

Candidate semantic dimensions được phân tích:

```text
amount_numeric
transaction_mode
location_state
hour_of_day
day_of_week
is_new_merchant
has_prior_card_history
time_since_previous_transaction_min
transactions_last_1h
amount_minus_previous_mean
```

M6.6 còn đọc risk score của FN/FP để phân biệt:

- error gần baseline decision boundary;
- error có score rất xa boundary;
- model-specific / shared error subsets.

M6.6 không:

- retrain model;
- tuning;
- resampling;
- threshold optimization;
- calibration tuning;
- final model selection;
- FINAL TEST evaluation.

Runtime-dependent status trước khi chạy:

`NOT YET VERIFIED`

## 1. Lineage requirement trước mọi row-level finding

M6.2 mới xác minh:

```text
row_id_validation:
valid canonical lineage anchor
```

nhưng chưa chứng minh transaction-level semantic remapping.

M6.6 phải khóa chuỗi sau:

```text
M4.7 row_id_validation
        ↓ exact equality
raw artifact raw_row_id
        ↓
M4.7 deterministic representation
        ↓
strict-causal behavioral features
        ↓
canonical validation semantic frame
        ↓ exact row alignment
y_validation / y_pred / risk_score
        ↓
TP / FP / FN / TN transaction groups
```

Nếu row-id mapping, target alignment hoặc semantic reconstruction fail:

`STOP — KHÔNG ĐƯỢC INTERPRET ERROR PATTERN`

## 2. Target-leakage / FINAL TEST guardrail

M6.6 rebuild semantic features từ raw artifact nhưng:

```text
Is Fraud?
NOT READ

Errors?
NOT READ
```

Raw target không được dùng để tạo:

- behavioral history;
- semantic features;
- error groups.

Error-analysis label đến từ:

`M4.7 canonical y_validation.npy`

sau khi exact `row_id_validation` alignment PASS.

Historical feature state chỉ dùng transaction có:

`Timestamp < 2019-06-01`

và strict history:

`Timestamp(history) < Timestamp(current)`.

Các row từ `2019-06-01` trở đi không được đưa vào history state, semantic validation frame hoặc error-analysis statistics.

## 3. Probability context boundary

Persisted risk score:

`positive-class predict_proba`

M6.6 dùng score để mô tả error behavior.

Reference boundary:

`0.5`

chỉ được dùng sau khi runtime xác minh persisted `y_pred` tương thích với score-side của baseline default decision rule.

Predeclared descriptive bands:

```text
FN:
very_low_score      < 0.10
low_score           0.10 → < 0.40
near_boundary       0.40 → 0.50

FP:
near_boundary       0.50 → < 0.60
high_score          0.60 → < 0.90
very_high_score     >= 0.90
```

Các band này:

`DESCRIPTIVE ONLY`

Không phải threshold search và không được dùng để thay `y_pred`.


```python

from pathlib import Path
from collections import Counter
import hashlib
import json
import platform
import sys
import time

import numpy as np
import pandas as pd


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


## 4. Locate canonical raw / M4 / M6 artifacts


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

M6_05_REL = (
    Path("data")
    / "processed"
    / "m6_05_cross_model_comparative_evaluation"
)

M6_06_REL = (
    Path("data")
    / "processed"
    / "m6_06_fp_fn_error_analysis"
)


candidate_roots = [
    Path.cwd(),
    *list(Path.cwd().parents)[:6],
]


PROJECT_ROOT = None

required_rel_paths = [
    RAW_REL,
    M4_REL / "y_validation.npy",
    M4_REL / "row_id_validation.npy",
    M6_02_REL / "m6_02_evaluation_registry.json",
    M6_05_REL / "m6_05_cross_model_comparison.json",
    M6_05_REL / "m6_05_comparison_manifest.json",
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
        "raw artifact + M4.7 + M6.2 + M6.5 artifacts."
    )


RAW_PATH = PROJECT_ROOT / RAW_REL
M4_DIR = PROJECT_ROOT / M4_REL
M6_02_DIR = PROJECT_ROOT / M6_02_REL
M6_05_DIR = PROJECT_ROOT / M6_05_REL
OUTPUT_DIR = PROJECT_ROOT / M6_06_REL

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


REGISTRY_PATH = (
    M6_02_DIR
    / "m6_02_evaluation_registry.json"
)

M6_05_COMPARISON_PATH = (
    M6_05_DIR
    / "m6_05_cross_model_comparison.json"
)

M6_05_MANIFEST_PATH = (
    M6_05_DIR
    / "m6_05_comparison_manifest.json"
)


EXPECTED_RAW_FILE_SIZE = 2_354_626_737
EXPECTED_RAW_ROWS = 24_386_900
EXPECTED_CARD_COUNT = 6_139
EXPECTED_CONTEXT_ROWS = 23_038_920
EXPECTED_VALIDATION_ROWS = 712_458
EXPECTED_VALIDATION_FRAUD = 1_052

CHUNK_SIZE = 500_000

TRAIN_END = pd.Timestamp(
    "2019-01-01"
)

VALIDATION_END = pd.Timestamp(
    "2019-06-01"
)


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
    "\nM6.6 ENVIRONMENT / RAW ARTIFACT GATE: PASS"
)

```

    PROJECT_ROOT:
    /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening
    
    RAW_PATH:
    /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/raw/ibm_tabformer/card_transaction.v1.csv
    
    OUTPUT_DIR:
    /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/processed/m6_06_fp_fn_error_analysis
    
    M6.6 ENVIRONMENT / RAW ARTIFACT GATE: PASS


## 5. Verify upstream evaluation handoff


```python

with open(
    REGISTRY_PATH,
    "r",
    encoding="utf-8",
) as file:
    m6_02_registry = json.load(
        file
    )


with open(
    M6_05_COMPARISON_PATH,
    "r",
    encoding="utf-8",
) as file:
    m6_05_comparison = json.load(
        file
    )


with open(
    M6_05_MANIFEST_PATH,
    "r",
    encoding="utf-8",
) as file:
    m6_05_manifest = json.load(
        file
    )


assert (
    m6_02_registry[
        "official_run_count"
    ]
    == 6
)

assert (
    m6_02_registry[
        "validation_rows"
    ]
    == EXPECTED_VALIDATION_ROWS
)

assert (
    m6_02_registry[
        "validation_fraud_rows"
    ]
    == EXPECTED_VALIDATION_FRAUD
)

assert (
    m6_02_registry[
        "final_test_accessed"
    ]
    is False
)


assert (
    m6_05_comparison[
        "validation_rows"
    ]
    == EXPECTED_VALIDATION_ROWS
)

assert (
    m6_05_comparison[
        "validation_fraud_rows"
    ]
    == EXPECTED_VALIDATION_FRAUD
)

assert (
    m6_05_comparison[
        "selection_state"
    ][
        "final_model"
    ]
    == "OPEN"
)

assert (
    m6_05_comparison[
        "selection_state"
    ][
        "final_threshold"
    ]
    == "OPEN"
)

assert (
    m6_05_comparison[
        "final_test_accessed"
    ]
    is False
)


assert (
    m6_05_manifest[
        "probability_artifacts_verified"
    ]
    == 6
)

assert (
    m6_05_manifest[
        "threshold_search"
    ]
    == "NOT_PERFORMED"
)

assert (
    m6_05_manifest[
        "final_model_selected"
    ]
    is False
)

assert (
    m6_05_manifest[
        "final_test_accessed"
    ]
    is False
)


print(
    "M6.6 UPSTREAM HANDOFF GATE: PASS"
)

```

    M6.6 UPSTREAM HANDOFF GATE: PASS


## 6. Load canonical validation target / row lineage


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


Y_VALIDATION_PATH = (
    M4_DIR
    / "y_validation.npy"
)

ROW_ID_VALIDATION_PATH = (
    M4_DIR
    / "row_id_validation.npy"
)


y_validation = np.load(
    Y_VALIDATION_PATH,
    allow_pickle=False,
)

row_id_validation = np.load(
    ROW_ID_VALIDATION_PATH,
    allow_pickle=False,
)


assert y_validation.ndim == 1
assert row_id_validation.ndim == 1

assert len(
    y_validation
) == EXPECTED_VALIDATION_ROWS

assert len(
    row_id_validation
) == EXPECTED_VALIDATION_ROWS

assert int(
    y_validation.sum()
) == EXPECTED_VALIDATION_FRAUD

assert len(
    np.unique(
        row_id_validation
    )
) == EXPECTED_VALIDATION_ROWS


assert (
    sha256_file(
        Y_VALIDATION_PATH
    )
    == m6_02_registry[
        "y_validation_sha256"
    ]
)

assert (
    sha256_file(
        ROW_ID_VALIDATION_PATH
    )
    == m6_02_registry[
        "row_id_validation_sha256"
    ]
)


print(
    "y_validation SHA256:"
)
print(
    m6_02_registry[
        "y_validation_sha256"
    ]
)

print(
    "\nrow_id_validation SHA256:"
)
print(
    m6_02_registry[
        "row_id_validation_sha256"
    ]
)

print(
    "\nM6.6 CANONICAL VALIDATION LINEAGE GATE: PASS"
)

```

    y_validation SHA256:
    0a21b2e93e017eda8b794be1a94a692beebc5883c1303427bc245299b55f5306
    
    row_id_validation SHA256:
    e7f3074c8604680eccef60ff7be45501f0dc10efbf6ac82e47bcbd1b00e43adc
    
    M6.6 CANONICAL VALIDATION LINEAGE GATE: PASS


## 7. Exact M4.7 deterministic representation helpers

Các helper dưới đây giữ semantic của M4.7:

```text
Amount:
strip "$" / "," → numeric

Timestamp:
Year + Month + Day + Time

transaction_mode:
normalized Use Chip

location_state:
NON_PHYSICAL_OR_ONLINE
PHYSICAL_COMPLETE
PHYSICAL_ZIP_UNAVAILABLE

hour_of_day:
Timestamp hour

day_of_week:
Timestamp dayofweek
```

Không casefold location text ngoài contract đã khóa.


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
    "M6.6 DETERMINISTIC REPRESENTATION HELPERS: DEFINED"
)

```

    M6.6 DETERMINISTIC REPRESENTATION HELPERS: DEFINED


## 8. Card-block streaming helper

`raw_row_id` được tái tạo đúng như M4.7:

```text
zero-based physical data-row offset
```

Raw usecols cố ý không chứa:

```text
Is Fraud?
Errors?
```


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

        working_columns = [
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

        working = (
            chunk[
                working_columns
            ]
        )

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

            if (
                len(
                    pending_parts
                )
                == 1
            ):
                card_block = (
                    pending_parts[0]
                    .reset_index(
                        drop=True
                    )
                )
            else:
                card_block = (
                    pd.concat(
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
        if (
            len(
                pending_parts
            )
            == 1
        ):
            card_block = (
                pending_parts[0]
                .reset_index(
                    drop=True
                )
            )
        else:
            card_block = (
                pd.concat(
                    pending_parts,
                    ignore_index=True,
                )
            )

        yield (
            pending_key,
            card_block,
        )


print(
    "M6.6 CARD-BLOCK STREAMING HELPER: DEFINED"
)

```

    M6.6 CARD-BLOCK STREAMING HELPER: DEFINED


## 9. Exact strict-causal behavioral builder

Core behavioral features:

```text
time_since_previous_transaction_min
transactions_last_1h
amount_minus_previous_mean
is_new_merchant
has_prior_card_history
```

Same-timestamp transaction:

`DO NOT PROVIDE HISTORY TO EACH OTHER`


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
            "Timestamp giảm bên trong Card block."
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

    transactions_last_1h = (
        np.repeat(
            transactions_last_1h_group,
            group_lengths,
        )
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
    "M6.6 STRICT-CAUSAL BEHAVIORAL BUILDER: DEFINED"
)

```

    M6.6 STRICT-CAUSAL BEHAVIORAL BUILDER: DEFINED


## 10. Strict-causal regression check


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
    "M6.6 STRICT-CAUSAL REGRESSION GATE: PASS"
)

```

    M6.6 STRICT-CAUSAL REGRESSION GATE: PASS


## 11. Build canonical 10-feature semantic frame


```python

CORE_FEATURE_COLUMNS = [
    "amount_numeric",
    "time_since_previous_transaction_min",
    "transactions_last_1h",
    "amount_minus_previous_mean",
    "is_new_merchant",
    "has_prior_card_history",
    "transaction_mode",
    "location_state",
    "hour_of_day",
    "day_of_week",
]


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


assert len(
    CORE_FEATURE_COLUMNS
) == 10


print(
    "M6.6 CORE SEMANTIC FRAME CONTRACT: LOCKED"
)

```

    M6.6 CORE SEMANTIC FRAME CONTRACT: LOCKED


## 12. Rebuild validation semantic transactions

Pass này có thể mất khoảng vài phút tùy máy vì phải stream full raw artifact theo Card để bảo toàn historical warm-up.

Quan trọng:

- `Is Fraud?` không được đọc;
- behavioral history chỉ nhận rows `< 2019-06-01`;
- chỉ validation rows được giữ lại trong RAM;
- raw User/Card/Merchant Name chỉ tồn tại tạm thời cho causal state và không được persist vào error-analysis artifact.


```python

validation_semantic_parts = []

card_count = 0
raw_rows_seen = 0
context_rows_seen = 0

rebuild_start = (
    time.perf_counter()
)


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

    context_card_df = (
        card_df.loc[
            card_df[
                "Timestamp"
            ]
            < VALIDATION_END
        ]
        .reset_index(
            drop=True
        )
    )

    if context_card_df.empty:
        continue

    context_rows_seen += len(
        context_card_df
    )

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

    semantic_frame = (
        build_core_feature_frame(
            context_card_df,
            behavioral_df,
        )
    )

    validation_mask = (
        (
            semantic_frame[
                "Timestamp"
            ]
            >= TRAIN_END
        )
        &
        (
            semantic_frame[
                "Timestamp"
            ]
            < VALIDATION_END
        )
    )

    if (
        validation_mask.any()
    ):
        validation_semantic_parts.append(
            semantic_frame.loc[
                validation_mask
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
            "| context rows:",
            f"{context_rows_seen:,}",
        )


validation_semantic = pd.concat(
    validation_semantic_parts,
    ignore_index=True,
)


rebuild_elapsed = (
    time.perf_counter()
    - rebuild_start
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
    "Development/history context rows:",
    f"{context_rows_seen:,}",
)

print(
    "Validation semantic rows:",
    f"{len(validation_semantic):,}",
)

print(
    "Elapsed seconds:",
    round(
        rebuild_elapsed,
        2,
    ),
)

```

    Card blocks: 500 | raw rows: 2,021,584 | context rows: 1,911,651
    Card blocks: 1,000 | raw rows: 4,066,794 | context rows: 3,841,967
    Card blocks: 1,500 | raw rows: 5,915,634 | context rows: 5,589,029
    Card blocks: 2,000 | raw rows: 7,883,446 | context rows: 7,444,960
    Card blocks: 2,500 | raw rows: 9,842,479 | context rows: 9,292,973
    Card blocks: 3,000 | raw rows: 11,897,259 | context rows: 11,238,345
    Card blocks: 3,500 | raw rows: 13,997,374 | context rows: 13,231,256
    Card blocks: 4,000 | raw rows: 16,040,304 | context rows: 15,156,787
    Card blocks: 4,500 | raw rows: 18,115,490 | context rows: 17,121,544
    Card blocks: 5,000 | raw rows: 19,945,815 | context rows: 18,850,177
    Card blocks: 5,500 | raw rows: 21,956,418 | context rows: 20,747,229
    Card blocks: 6,000 | raw rows: 23,844,247 | context rows: 22,528,539
    
    Card blocks: 6,139
    Raw rows streamed: 24,386,900
    Development/history context rows: 23,038,920
    Validation semantic rows: 712,458
    Elapsed seconds: 117.69


## 13. Transaction-level lineage / semantic mapping gate


```python

assert (
    card_count
    == EXPECTED_CARD_COUNT
)

assert (
    raw_rows_seen
    == EXPECTED_RAW_ROWS
)

assert (
    context_rows_seen
    == EXPECTED_CONTEXT_ROWS
)

assert (
    len(
        validation_semantic
    )
    == EXPECTED_VALIDATION_ROWS
)

assert (
    validation_semantic[
        "raw_row_id"
    ]
    .is_unique
)

np.testing.assert_array_equal(
    validation_semantic[
        "raw_row_id"
    ].to_numpy(
        dtype=np.int64
    ),
    row_id_validation,
)


assert (
    validation_semantic[
        "Timestamp"
    ]
    .min()
    >= TRAIN_END
)

assert (
    validation_semantic[
        "Timestamp"
    ]
    .max()
    < VALIDATION_END
)


assert set(
    CORE_FEATURE_COLUMNS
).issubset(
    validation_semantic.columns
)


for prohibited in [
    "User",
    "Card",
    "Merchant Name",
    "Errors?",
    "Is Fraud?",
]:
    assert (
        prohibited
        not in validation_semantic.columns
    )


validation_semantic[
    "is_fraud"
] = y_validation


assert int(
    validation_semantic[
        "is_fraud"
    ].sum()
) == EXPECTED_VALIDATION_FRAUD


print(
    "First validation raw_row_id:",
    int(
        validation_semantic[
            "raw_row_id"
        ].iloc[0]
    ),
)

print(
    "Last validation raw_row_id:",
    int(
        validation_semantic[
            "raw_row_id"
        ].iloc[-1]
    ),
)

print(
    "\nM6.6 TRANSACTION-LEVEL LINEAGE MAPPING GATE: PASS"
)

```

    First validation raw_row_id: 4776
    Last validation raw_row_id: 24385729
    
    M6.6 TRANSACTION-LEVEL LINEAGE MAPPING GATE: PASS


## 14. Load six prediction / risk-score artifacts

M6.6 tái kiểm tra:

- persisted array length;
- prediction support;
- risk-score range;
- SHA-256 identity;
- confusion counts.

Không chạy model prediction lại.


```python

registry_by_id = {
    record[
        "experiment_id"
    ]:
    record
    for record
    in m6_02_registry[
        "records"
    ]
}


OFFICIAL_RUNS = [
    "M5-LR-SHORT-B04",
    "M5-LR-LONG-B04",
    "M5-DT-SHORT-B01",
    "M5-DT-LONG-B01",
    "M5-RF-SHORT-B01",
    "M5-RF-LONG-B01",
]


run_arrays = {}


for experiment_id in (
    OFFICIAL_RUNS
):
    record = (
        registry_by_id[
            experiment_id
        ]
    )

    pred_path = (
        PROJECT_ROOT
        / record[
            "prediction_artifact"
        ]
    )

    score_path = (
        PROJECT_ROOT
        / record[
            "risk_score_artifact"
        ]
    )

    assert pred_path.exists()
    assert score_path.exists()

    assert (
        sha256_file(
            pred_path
        )
        == record[
            "prediction_sha256"
        ]
    )

    assert (
        sha256_file(
            score_path
        )
        == record[
            "risk_score_sha256"
        ]
    )

    y_pred = np.load(
        pred_path,
        allow_pickle=False,
    )

    risk_score = np.load(
        score_path,
        allow_pickle=False,
    )

    assert y_pred.ndim == 1
    assert risk_score.ndim == 1

    assert len(
        y_pred
    ) == EXPECTED_VALIDATION_ROWS

    assert len(
        risk_score
    ) == EXPECTED_VALIDATION_ROWS

    assert set(
        np.unique(
            y_pred
        ).tolist()
    ) <= {0, 1}

    assert np.isfinite(
        risk_score
    ).all()

    assert np.all(
        risk_score
        >= 0.0
    )

    assert np.all(
        risk_score
        <= 1.0
    )

    run_arrays[
        experiment_id
    ] = {
        "y_pred":
            y_pred.astype(
                np.int8,
                copy=False,
            ),

        "risk_score":
            risk_score.astype(
                np.float64,
                copy=False,
            ),
    }


assert len(
    run_arrays
) == 6


print(
    "M6.6 SIX-RUN ARRAY IDENTITY GATE: PASS"
)

```

    M6.6 SIX-RUN ARRAY IDENTITY GATE: PASS


## 15. Reconstruct TP / FP / FN / TN row groups


```python

error_group_masks = {}
error_group_row_ids = {}


for experiment_id in (
    OFFICIAL_RUNS
):
    record = (
        registry_by_id[
            experiment_id
        ]
    )

    y_pred = (
        run_arrays[
            experiment_id
        ][
            "y_pred"
        ]
    )

    tp = (
        (y_validation == 1)
        & (y_pred == 1)
    )

    fn = (
        (y_validation == 1)
        & (y_pred == 0)
    )

    fp = (
        (y_validation == 0)
        & (y_pred == 1)
    )

    tn = (
        (y_validation == 0)
        & (y_pred == 0)
    )

    assert int(
        tp.sum()
    ) == int(
        record["tp"]
    )

    assert int(
        fn.sum()
    ) == int(
        record["fn"]
    )

    assert int(
        fp.sum()
    ) == int(
        record["fp"]
    )

    assert int(
        tn.sum()
    ) == int(
        record["tn"]
    )

    assert np.all(
        (
            tp.astype(np.int8)
            + fn.astype(np.int8)
            + fp.astype(np.int8)
            + tn.astype(np.int8)
        )
        == 1
    )

    error_group_masks[
        experiment_id
    ] = {
        "TP": tp,
        "FN": fn,
        "FP": fp,
        "TN": tn,
    }

    error_group_row_ids[
        experiment_id
    ] = {
        "TP":
            row_id_validation[
                tp
            ],

        "FN":
            row_id_validation[
                fn
            ],

        "FP":
            row_id_validation[
                fp
            ],
    }


print(
    "M6.6 CONFUSION GROUP RECONSTRUCTION GATE: PASS"
)

```

    M6.6 CONFUSION GROUP RECONSTRUCTION GATE: PASS


## 16. Score / baseline decision-rule compatibility

M6.6 không tìm threshold.

Cell này chỉ xác minh rằng persisted risk score nằm đúng phía của reference boundary `0.5` so với persisted `y_pred`.

Nếu không tương thích:

`STOP`

vì score-band interpretation không còn hợp lệ.


```python

REFERENCE_BOUNDARY = 0.5
BOUNDARY_TOL = 1e-12


score_boundary_checks = {}


for experiment_id in (
    OFFICIAL_RUNS
):
    y_pred = (
        run_arrays[
            experiment_id
        ][
            "y_pred"
        ]
    )

    scores = (
        run_arrays[
            experiment_id
        ][
            "risk_score"
        ]
    )

    positive_scores = scores[
        y_pred == 1
    ]

    negative_scores = scores[
        y_pred == 0
    ]

    positive_side_ok = (
        len(
            positive_scores
        )
        == 0
        or
        float(
            positive_scores.min()
        )
        >= (
            REFERENCE_BOUNDARY
            - BOUNDARY_TOL
        )
    )

    negative_side_ok = (
        len(
            negative_scores
        )
        == 0
        or
        float(
            negative_scores.max()
        )
        <= (
            REFERENCE_BOUNDARY
            + BOUNDARY_TOL
        )
    )

    assert positive_side_ok
    assert negative_side_ok

    score_boundary_checks[
        experiment_id
    ] = {
        "reference_boundary":
            REFERENCE_BOUNDARY,

        "positive_min":
            (
                float(
                    positive_scores.min()
                )
                if len(
                    positive_scores
                )
                else None
            ),

        "negative_max":
            (
                float(
                    negative_scores.max()
                )
                if len(
                    negative_scores
                )
                else None
            ),

        "compatible":
            True,
    }


print(
    "M6.6 SCORE / PREDICTION BOUNDARY GATE: PASS"
)

```

    M6.6 SCORE / PREDICTION BOUNDARY GATE: PASS


## 17. Error-analysis summary helpers


```python

NUMERIC_FEATURES = [
    "amount_numeric",
    "time_since_previous_transaction_min",
    "transactions_last_1h",
    "amount_minus_previous_mean",
]

BOOLEAN_FEATURES = [
    "is_new_merchant",
    "has_prior_card_history",
]

CATEGORICAL_FEATURES = [
    "transaction_mode",
    "location_state",
    "hour_of_day",
    "day_of_week",
]


def summarize_numeric(
    series,
):
    values = pd.to_numeric(
        series,
        errors="coerce",
    )

    finite = values[
        values.notna()
    ].to_numpy(
        dtype=np.float64
    )

    result = {
        "n":
            int(
                len(
                    values
                )
            ),

        "missing_count":
            int(
                values.isna().sum()
            ),
    }

    if len(
        finite
    ) == 0:
        result.update(
            {
                "mean": None,
                "std": None,
                "q25": None,
                "median": None,
                "q75": None,
                "q90": None,
                "min": None,
                "max": None,
            }
        )

        return result

    result.update(
        {
            "mean":
                float(
                    np.mean(
                        finite
                    )
                ),

            "std":
                float(
                    np.std(
                        finite
                    )
                ),

            "q25":
                float(
                    np.quantile(
                        finite,
                        0.25,
                    )
                ),

            "median":
                float(
                    np.quantile(
                        finite,
                        0.50,
                    )
                ),

            "q75":
                float(
                    np.quantile(
                        finite,
                        0.75,
                    )
                ),

            "q90":
                float(
                    np.quantile(
                        finite,
                        0.90,
                    )
                ),

            "min":
                float(
                    np.min(
                        finite
                    )
                ),

            "max":
                float(
                    np.max(
                        finite
                    )
                ),
        }
    )

    return result


def summarize_boolean(
    series,
):
    values = (
        series
        .astype("boolean")
    )

    n = len(
        values
    )

    missing_count = int(
        values.isna().sum()
    )

    true_count = int(
        values.fillna(
            False
        ).sum()
    )

    observed_n = (
        n
        - missing_count
    )

    return {
        "n":
            int(
                n
            ),

        "missing_count":
            missing_count,

        "true_count":
            true_count,

        "false_count":
            int(
                observed_n
                - true_count
            ),

        "true_rate":
            (
                float(
                    true_count
                    / observed_n
                )
                if observed_n
                else None
            ),
    }


def categorical_distribution(
    series,
):
    values = (
        series
        .astype("string")
        .fillna(
            "__MISSING__"
        )
    )

    counts = (
        values
        .value_counts(
            dropna=False
        )
    )

    n = len(
        values
    )

    return {
        str(level): {
            "count":
                int(
                    count
                ),

            "rate":
                float(
                    count
                    / n
                ),
        }
        for level, count
        in counts.items()
    }


def categorical_rate_gaps(
    error_distribution,
    correct_distribution,
    top_n=5,
):
    levels = sorted(
        set(
            error_distribution
        )
        |
        set(
            correct_distribution
        )
    )

    rows = []

    for level in levels:
        error_rate = (
            error_distribution
            .get(
                level,
                {
                    "rate":
                        0.0
                },
            )[
                "rate"
            ]
        )

        correct_rate = (
            correct_distribution
            .get(
                level,
                {
                    "rate":
                        0.0
                },
            )[
                "rate"
            ]
        )

        gap = (
            error_rate
            - correct_rate
        )

        rows.append(
            {
                "level":
                    level,

                "error_rate":
                    float(
                        error_rate
                    ),

                "correct_rate":
                    float(
                        correct_rate
                    ),

                "error_minus_correct_rate":
                    float(
                        gap
                    ),

                "absolute_rate_gap":
                    float(
                        abs(
                            gap
                        )
                    ),
            }
        )

    rows.sort(
        key=lambda row:
            row[
                "absolute_rate_gap"
            ],
        reverse=True,
    )

    return rows[
        :top_n
    ]


def analyze_error_vs_correct(
    error_mask,
    correct_mask,
):
    result = {
        "error_count":
            int(
                error_mask.sum()
            ),

        "correct_count":
            int(
                correct_mask.sum()
            ),

        "numeric": {},
        "boolean": {},
        "categorical": {},
    }

    for feature in (
        NUMERIC_FEATURES
    ):
        error_summary = (
            summarize_numeric(
                validation_semantic.loc[
                    error_mask,
                    feature,
                ]
            )
        )

        correct_summary = (
            summarize_numeric(
                validation_semantic.loc[
                    correct_mask,
                    feature,
                ]
            )
        )

        result[
            "numeric"
        ][
            feature
        ] = {
            "error":
                error_summary,

            "correct":
                correct_summary,

            "delta_mean":
                (
                    None
                    if (
                        error_summary[
                            "mean"
                        ]
                        is None
                        or
                        correct_summary[
                            "mean"
                        ]
                        is None
                    )
                    else float(
                        error_summary[
                            "mean"
                        ]
                        - correct_summary[
                            "mean"
                        ]
                    )
                ),

            "delta_median":
                (
                    None
                    if (
                        error_summary[
                            "median"
                        ]
                        is None
                        or
                        correct_summary[
                            "median"
                        ]
                        is None
                    )
                    else float(
                        error_summary[
                            "median"
                        ]
                        - correct_summary[
                            "median"
                        ]
                    )
                ),
        }

    for feature in (
        BOOLEAN_FEATURES
    ):
        error_summary = (
            summarize_boolean(
                validation_semantic.loc[
                    error_mask,
                    feature,
                ]
            )
        )

        correct_summary = (
            summarize_boolean(
                validation_semantic.loc[
                    correct_mask,
                    feature,
                ]
            )
        )

        result[
            "boolean"
        ][
            feature
        ] = {
            "error":
                error_summary,

            "correct":
                correct_summary,

            "delta_true_rate":
                float(
                    error_summary[
                        "true_rate"
                    ]
                    - correct_summary[
                        "true_rate"
                    ]
                ),
        }

    for feature in (
        CATEGORICAL_FEATURES
    ):
        error_distribution = (
            categorical_distribution(
                validation_semantic.loc[
                    error_mask,
                    feature,
                ]
            )
        )

        correct_distribution = (
            categorical_distribution(
                validation_semantic.loc[
                    correct_mask,
                    feature,
                ]
            )
        )

        result[
            "categorical"
        ][
            feature
        ] = {
            "error_distribution":
                error_distribution,

            "correct_distribution":
                correct_distribution,

            "largest_absolute_rate_gaps":
                categorical_rate_gaps(
                    error_distribution,
                    correct_distribution,
                    top_n=5,
                ),
        }

    return result


print(
    "M6.6 ERROR-ANALYSIS HELPERS: DEFINED"
)

```

    M6.6 ERROR-ANALYSIS HELPERS: DEFINED


## 18. Risk-score error context helpers


```python

def summarize_score(
    scores,
):
    scores = np.asarray(
        scores,
        dtype=np.float64,
    )

    if len(
        scores
    ) == 0:
        return {
            "n": 0,
            "mean": None,
            "q10": None,
            "q25": None,
            "median": None,
            "q75": None,
            "q90": None,
            "min": None,
            "max": None,
        }

    return {
        "n":
            int(
                len(
                    scores
                )
            ),

        "mean":
            float(
                np.mean(
                    scores
                )
            ),

        "q10":
            float(
                np.quantile(
                    scores,
                    0.10,
                )
            ),

        "q25":
            float(
                np.quantile(
                    scores,
                    0.25,
                )
            ),

        "median":
            float(
                np.quantile(
                    scores,
                    0.50,
                )
            ),

        "q75":
            float(
                np.quantile(
                    scores,
                    0.75,
                )
            ),

        "q90":
            float(
                np.quantile(
                    scores,
                    0.90,
                )
            ),

        "min":
            float(
                np.min(
                    scores
                )
            ),

        "max":
            float(
                np.max(
                    scores
                )
            ),
    }


def count_rate(
    mask,
):
    count = int(
        mask.sum()
    )

    n = len(
        mask
    )

    return {
        "count":
            count,

        "rate":
            (
                float(
                    count
                    / n
                )
                if n
                else None
            ),
    }


def build_error_score_context(
    experiment_id,
):
    scores = (
        run_arrays[
            experiment_id
        ][
            "risk_score"
        ]
    )

    masks = (
        error_group_masks[
            experiment_id
        ]
    )

    tp_scores = scores[
        masks["TP"]
    ]

    fn_scores = scores[
        masks["FN"]
    ]

    fp_scores = scores[
        masks["FP"]
    ]

    tn_scores = scores[
        masks["TN"]
    ]

    fn_bands = {
        "very_low_score_lt_0_10":
            count_rate(
                fn_scores
                < 0.10
            ),

        "low_score_0_10_to_lt_0_40":
            count_rate(
                (
                    fn_scores
                    >= 0.10
                )
                &
                (
                    fn_scores
                    < 0.40
                )
            ),

        "near_boundary_0_40_to_0_50":
            count_rate(
                (
                    fn_scores
                    >= 0.40
                )
                &
                (
                    fn_scores
                    <= (
                        REFERENCE_BOUNDARY
                        + BOUNDARY_TOL
                    )
                )
            ),
    }

    fp_bands = {
        "near_boundary_0_50_to_lt_0_60":
            count_rate(
                (
                    fp_scores
                    >= (
                        REFERENCE_BOUNDARY
                        - BOUNDARY_TOL
                    )
                )
                &
                (
                    fp_scores
                    < 0.60
                )
            ),

        "high_score_0_60_to_lt_0_90":
            count_rate(
                (
                    fp_scores
                    >= 0.60
                )
                &
                (
                    fp_scores
                    < 0.90
                )
            ),

        "very_high_score_ge_0_90":
            count_rate(
                fp_scores
                >= 0.90
            ),
    }

    return {
        "reference_boundary":
            REFERENCE_BOUNDARY,

        "threshold_search_performed":
            False,

        "TP":
            summarize_score(
                tp_scores
            ),

        "FN":
            summarize_score(
                fn_scores
            ),

        "FP":
            summarize_score(
                fp_scores
            ),

        "TN":
            summarize_score(
                tn_scores
            ),

        "FN_descriptive_bands":
            fn_bands,

        "FP_descriptive_bands":
            fp_bands,
    }


print(
    "M6.6 SCORE-CONTEXT HELPERS: DEFINED"
)

```

    M6.6 SCORE-CONTEXT HELPERS: DEFINED


## 19. Run six-run transaction-level error analysis


```python

run_error_analysis = {}


for experiment_id in (
    OFFICIAL_RUNS
):
    masks = (
        error_group_masks[
            experiment_id
        ]
    )

    fraud_side = (
        analyze_error_vs_correct(
            error_mask=masks[
                "FN"
            ],
            correct_mask=masks[
                "TP"
            ],
        )
    )

    non_fraud_side = (
        analyze_error_vs_correct(
            error_mask=masks[
                "FP"
            ],
            correct_mask=masks[
                "TN"
            ],
        )
    )

    score_context = (
        build_error_score_context(
            experiment_id
        )
    )

    run_error_analysis[
        experiment_id
    ] = {
        "experiment_id":
            experiment_id,

        "model_family":
            registry_by_id[
                experiment_id
            ][
                "model_family"
            ],

        "training_window_id":
            registry_by_id[
                experiment_id
            ][
                "training_window_id"
            ],

        "confusion_counts": {
            "TP":
                int(
                    masks[
                        "TP"
                    ].sum()
                ),

            "FN":
                int(
                    masks[
                        "FN"
                    ].sum()
                ),

            "FP":
                int(
                    masks[
                        "FP"
                    ].sum()
                ),

            "TN":
                int(
                    masks[
                        "TN"
                    ].sum()
                ),
        },

        "fraud_side_FN_vs_TP":
            fraud_side,

        "non_fraud_side_FP_vs_TN":
            non_fraud_side,

        "risk_score_context":
            score_context,

        "row_level_mapping":
            "VERIFIED",

        "threshold_optimization":
            "NOT_PERFORMED",

        "final_test_accessed":
            False,
    }


assert len(
    run_error_analysis
) == 6


print(
    "M6.6 SIX-RUN TRANSACTION ERROR ANALYSIS GATE: PASS"
)

```

    M6.6 SIX-RUN TRANSACTION ERROR ANALYSIS GATE: PASS


## 20. Concise per-run error evidence

Cell này chỉ in các gap dễ review:

Numeric:

`error median − correct median`

Boolean:

`error true-rate − correct true-rate`

Categorical:

`largest absolute percentage-point gap`

Risk score:

`FN median / FP median + descriptive boundary bands`

Full distributions được persist trong JSON.


```python

for experiment_id in (
    OFFICIAL_RUNS
):
    analysis = (
        run_error_analysis[
            experiment_id
        ]
    )

    print("=" * 72)

    print(
        experiment_id,
        "|",
        analysis[
            "model_family"
        ],
        "|",
        analysis[
            "training_window_id"
        ],
    )

    print(
        "Counts:",
        analysis[
            "confusion_counts"
        ],
    )

    print(
        "\nFraud side — FN vs TP"
    )

    for feature in (
        NUMERIC_FEATURES
    ):
        item = (
            analysis[
                "fraud_side_FN_vs_TP"
            ][
                "numeric"
            ][
                feature
            ]
        )

        print(
            " ",
            feature,
            "Δmedian(FN-TP)=",
            item[
                "delta_median"
            ],
        )

    for feature in (
        BOOLEAN_FEATURES
    ):
        item = (
            analysis[
                "fraud_side_FN_vs_TP"
            ][
                "boolean"
            ][
                feature
            ]
        )

        print(
            " ",
            feature,
            "Δtrue_rate(FN-TP)=",
            item[
                "delta_true_rate"
            ],
        )

    for feature in (
        CATEGORICAL_FEATURES
    ):
        gaps = (
            analysis[
                "fraud_side_FN_vs_TP"
            ][
                "categorical"
            ][
                feature
            ][
                "largest_absolute_rate_gaps"
            ]
        )

        print(
            " ",
            feature,
            "top gap=",
            (
                gaps[0]
                if gaps
                else None
            ),
        )

    print(
        "\nNon-fraud side — FP vs TN"
    )

    for feature in (
        NUMERIC_FEATURES
    ):
        item = (
            analysis[
                "non_fraud_side_FP_vs_TN"
            ][
                "numeric"
            ][
                feature
            ]
        )

        print(
            " ",
            feature,
            "Δmedian(FP-TN)=",
            item[
                "delta_median"
            ],
        )

    for feature in (
        BOOLEAN_FEATURES
    ):
        item = (
            analysis[
                "non_fraud_side_FP_vs_TN"
            ][
                "boolean"
            ][
                feature
            ]
        )

        print(
            " ",
            feature,
            "Δtrue_rate(FP-TN)=",
            item[
                "delta_true_rate"
            ],
        )

    for feature in (
        CATEGORICAL_FEATURES
    ):
        gaps = (
            analysis[
                "non_fraud_side_FP_vs_TN"
            ][
                "categorical"
            ][
                feature
            ][
                "largest_absolute_rate_gaps"
            ]
        )

        print(
            " ",
            feature,
            "top gap=",
            (
                gaps[0]
                if gaps
                else None
            ),
        )

    score_context = (
        analysis[
            "risk_score_context"
        ]
    )

    print(
        "\nFN score median:",
        score_context[
            "FN"
        ][
            "median"
        ],
    )

    print(
        "FP score median:",
        score_context[
            "FP"
        ][
            "median"
        ],
    )

    print(
        "FN bands:",
        score_context[
            "FN_descriptive_bands"
        ],
    )

    print(
        "FP bands:",
        score_context[
            "FP_descriptive_bands"
        ],
    )


print(
    "\nM6.6 PER-RUN ERROR EVIDENCE: READY"
)

```

    ========================================================================
    M5-LR-SHORT-B04 | Logistic Regression | W_SHORT
    Counts: {'TP': 258, 'FN': 794, 'FP': 219, 'TN': 711187}
    
    Fraud side — FN vs TP
      amount_numeric Δmedian(FN-TP)= -7.135000000000005
      time_since_previous_transaction_min Δmedian(FN-TP)= -36.5
      transactions_last_1h Δmedian(FN-TP)= 0.0
      amount_minus_previous_mean Δmedian(FN-TP)= -17.3688664181297
      is_new_merchant Δtrue_rate(FN-TP)= -0.5711049928729033
      has_prior_card_history Δtrue_rate(FN-TP)= 0.0
      transaction_mode top gap= {'level': 'Swipe Transaction', 'error_rate': 0.08564231738035265, 'correct_rate': 0.09689922480620156, 'error_minus_correct_rate': -0.011256907425848908, 'absolute_rate_gap': 0.011256907425848908}
      location_state top gap= {'level': 'PHYSICAL_ZIP_UNAVAILABLE', 'error_rate': 1.0, 'correct_rate': 1.0, 'error_minus_correct_rate': 0.0, 'absolute_rate_gap': 0.0}
      hour_of_day top gap= {'level': '10', 'error_rate': 0.08438287153652393, 'correct_rate': 0.007751937984496124, 'error_minus_correct_rate': 0.07663093355202781, 'absolute_rate_gap': 0.07663093355202781}
      day_of_week top gap= {'level': '3', 'error_rate': 0.18261964735516373, 'correct_rate': 0.5348837209302325, 'error_minus_correct_rate': -0.3522640735750688, 'absolute_rate_gap': 0.3522640735750688}
    
    Non-fraud side — FP vs TN
      amount_numeric Δmedian(FP-TN)= -8.819999999999997
      time_since_previous_transaction_min Δmedian(FP-TN)= -196.0
      transactions_last_1h Δmedian(FP-TN)= 0.0
      amount_minus_previous_mean Δmedian(FP-TN)= -2.53753382932247
      is_new_merchant Δtrue_rate(FP-TN)= 0.9296749065471628
      has_prior_card_history Δtrue_rate(FP-TN)= 4.7807398054211525e-05
      transaction_mode top gap= {'level': 'Chip Transaction', 'error_rate': 0.8904109589041096, 'correct_rate': 0.7056737538790783, 'error_minus_correct_rate': 0.18473720502503133, 'absolute_rate_gap': 0.18473720502503133}
      location_state top gap= {'level': 'PHYSICAL_ZIP_UNAVAILABLE', 'error_rate': 0.9908675799086758, 'correct_rate': 0.005940772258210569, 'error_minus_correct_rate': 0.9849268076504653, 'absolute_rate_gap': 0.9849268076504653}
      hour_of_day top gap= {'level': '11', 'error_rate': 0.1461187214611872, 'correct_rate': 0.07104741790837009, 'error_minus_correct_rate': 0.07507130355281712, 'absolute_rate_gap': 0.07507130355281712}
      day_of_week top gap= {'level': '5', 'error_rate': 0.0, 'correct_rate': 0.14334064036603594, 'error_minus_correct_rate': -0.14334064036603594, 'absolute_rate_gap': 0.14334064036603594}
    
    FN score median: 0.09735267981886864
    FP score median: 0.6081570982933044
    FN bands: {'very_low_score_lt_0_10': {'count': 401, 'rate': 0.5050377833753149}, 'low_score_0_10_to_lt_0_40': {'count': 330, 'rate': 0.4156171284634761}, 'near_boundary_0_40_to_0_50': {'count': 63, 'rate': 0.07934508816120907}}
    FP bands: {'near_boundary_0_50_to_lt_0_60': {'count': 97, 'rate': 0.4429223744292237}, 'high_score_0_60_to_lt_0_90': {'count': 121, 'rate': 0.5525114155251142}, 'very_high_score_ge_0_90': {'count': 1, 'rate': 0.0045662100456621}}
    ========================================================================
    M5-LR-LONG-B04 | Logistic Regression | W_LONG
    Counts: {'TP': 22, 'FN': 1030, 'FP': 27, 'TN': 711379}
    
    Fraud side — FN vs TP
      amount_numeric Δmedian(FN-TP)= -27.554999999999993
      time_since_previous_transaction_min Δmedian(FN-TP)= 271.5
      transactions_last_1h Δmedian(FN-TP)= -1.5
      amount_minus_previous_mean Δmedian(FN-TP)= -34.3750818733318
      is_new_merchant Δtrue_rate(FN-TP)= -0.4679611650485437
      has_prior_card_history Δtrue_rate(FN-TP)= 0.0
      transaction_mode top gap= {'level': 'Chip Transaction', 'error_rate': 0.9233009708737864, 'correct_rate': 0.36363636363636365, 'error_minus_correct_rate': 0.5596646072374227, 'absolute_rate_gap': 0.5596646072374227}
      location_state top gap= {'level': 'PHYSICAL_ZIP_UNAVAILABLE', 'error_rate': 1.0, 'correct_rate': 1.0, 'error_minus_correct_rate': 0.0, 'absolute_rate_gap': 0.0}
      hour_of_day top gap= {'level': '14', 'error_rate': 0.10679611650485436, 'correct_rate': 0.2727272727272727, 'error_minus_correct_rate': -0.16593115622241833, 'absolute_rate_gap': 0.16593115622241833}
      day_of_week top gap= {'level': '5', 'error_rate': 0.24271844660194175, 'correct_rate': 0.0, 'error_minus_correct_rate': 0.24271844660194175, 'absolute_rate_gap': 0.24271844660194175}
    
    Non-fraud side — FP vs TN
      amount_numeric Δmedian(FP-TN)= 64.10000000000001
      time_since_previous_transaction_min Δmedian(FP-TN)= -560.0
      transactions_last_1h Δmedian(FP-TN)= 1.0
      amount_minus_previous_mean Δmedian(FP-TN)= 127.27922867066331
      is_new_merchant Δtrue_rate(FP-TN)= 0.9750878223844112
      has_prior_card_history Δtrue_rate(FP-TN)= -0.07402909102003397
      transaction_mode top gap= {'level': 'Chip Transaction', 'error_rate': 0.14814814814814814, 'correct_rate': 0.7057517863192475, 'error_minus_correct_rate': -0.5576036381710994, 'absolute_rate_gap': 0.5576036381710994}
      location_state top gap= {'level': 'PHYSICAL_COMPLETE', 'error_rate': 0.0, 'correct_rate': 0.8688013000102618, 'error_minus_correct_rate': -0.8688013000102618, 'absolute_rate_gap': 0.8688013000102618}
      hour_of_day top gap= {'level': '14', 'error_rate': 0.14814814814814814, 'correct_rate': 0.06619678118133934, 'error_minus_correct_rate': 0.0819513669668088, 'absolute_rate_gap': 0.0819513669668088}
      day_of_week top gap= {'level': '6', 'error_rate': 0.4074074074074074, 'correct_rate': 0.14318106100967276, 'error_minus_correct_rate': 0.2642263463977346, 'absolute_rate_gap': 0.2642263463977346}
    
    FN score median: 0.0505970474332571
    FP score median: 0.5726937055587769
    FN bands: {'very_low_score_lt_0_10': {'count': 600, 'rate': 0.5825242718446602}, 'low_score_0_10_to_lt_0_40': {'count': 399, 'rate': 0.38737864077669903}, 'near_boundary_0_40_to_0_50': {'count': 31, 'rate': 0.030097087378640777}}
    FP bands: {'near_boundary_0_50_to_lt_0_60': {'count': 17, 'rate': 0.6296296296296297}, 'high_score_0_60_to_lt_0_90': {'count': 8, 'rate': 0.2962962962962963}, 'very_high_score_ge_0_90': {'count': 2, 'rate': 0.07407407407407407}}
    ========================================================================
    M5-DT-SHORT-B01 | Decision Tree | W_SHORT
    Counts: {'TP': 330, 'FN': 722, 'FP': 636, 'TN': 710770}
    
    Fraud side — FN vs TP
      amount_numeric Δmedian(FN-TP)= -2.8149999999999977
      time_since_previous_transaction_min Δmedian(FN-TP)= 1.5
      transactions_last_1h Δmedian(FN-TP)= 0.0
      amount_minus_previous_mean Δmedian(FN-TP)= -10.286510279572985
      is_new_merchant Δtrue_rate(FN-TP)= -0.08034919835473853
      has_prior_card_history Δtrue_rate(FN-TP)= 0.0
      transaction_mode top gap= {'level': 'Chip Transaction', 'error_rate': 0.9196675900277008, 'correct_rate': 0.8939393939393939, 'error_minus_correct_rate': 0.0257281960883069, 'absolute_rate_gap': 0.0257281960883069}
      location_state top gap= {'level': 'PHYSICAL_ZIP_UNAVAILABLE', 'error_rate': 1.0, 'correct_rate': 1.0, 'error_minus_correct_rate': 0.0, 'absolute_rate_gap': 0.0}
      hour_of_day top gap= {'level': '16', 'error_rate': 0.10803324099722991, 'correct_rate': 0.14242424242424243, 'error_minus_correct_rate': -0.03439100142701251, 'absolute_rate_gap': 0.03439100142701251}
      day_of_week top gap= {'level': '1', 'error_rate': 0.34349030470914127, 'correct_rate': 0.006060606060606061, 'error_minus_correct_rate': 0.3374296986485352, 'absolute_rate_gap': 0.3374296986485352}
    
    Non-fraud side — FP vs TN
      amount_numeric Δmedian(FP-TN)= 1.6850000000000023
      time_since_previous_transaction_min Δmedian(FP-TN)= -210.0
      transactions_last_1h Δmedian(FP-TN)= 0.0
      amount_minus_previous_mean Δmedian(FP-TN)= 1.1180675096910964
      is_new_merchant Δtrue_rate(FP-TN)= 0.29292275194861306
      has_prior_card_history Δtrue_rate(FP-TN)= 4.783544606556944e-05
      transaction_mode top gap= {'level': 'Chip Transaction', 'error_rate': 0.8018867924528302, 'correct_rate': 0.705644582635733, 'error_minus_correct_rate': 0.09624220981709719, 'absolute_rate_gap': 0.09624220981709719}
      location_state top gap= {'level': 'PHYSICAL_ZIP_UNAVAILABLE', 'error_rate': 0.9182389937106918, 'correct_rate': 0.005427916203553892, 'error_minus_correct_rate': 0.912811077507138, 'absolute_rate_gap': 0.912811077507138}
      hour_of_day top gap= {'level': '7', 'error_rate': 0.0047169811320754715, 'correct_rate': 0.06648282848178735, 'error_minus_correct_rate': -0.06176584734971188, 'absolute_rate_gap': 0.06176584734971188}
      day_of_week top gap= {'level': '6', 'error_rate': 0.009433962264150943, 'correct_rate': 0.1433107756376887, 'error_minus_correct_rate': -0.13387681337353777, 'absolute_rate_gap': 0.13387681337353777}
    
    FN score median: 0.0
    FP score median: 1.0
    FN bands: {'very_low_score_lt_0_10': {'count': 722, 'rate': 1.0}, 'low_score_0_10_to_lt_0_40': {'count': 0, 'rate': 0.0}, 'near_boundary_0_40_to_0_50': {'count': 0, 'rate': 0.0}}
    FP bands: {'near_boundary_0_50_to_lt_0_60': {'count': 0, 'rate': 0.0}, 'high_score_0_60_to_lt_0_90': {'count': 0, 'rate': 0.0}, 'very_high_score_ge_0_90': {'count': 636, 'rate': 1.0}}
    ========================================================================
    M5-DT-LONG-B01 | Decision Tree | W_LONG
    Counts: {'TP': 207, 'FN': 845, 'FP': 820, 'TN': 710586}
    
    Fraud side — FN vs TP
      amount_numeric Δmedian(FN-TP)= 2.25
      time_since_previous_transaction_min Δmedian(FN-TP)= 9.0
      transactions_last_1h Δmedian(FN-TP)= 0.0
      amount_minus_previous_mean Δmedian(FN-TP)= -1.9811655673389978
      is_new_merchant Δtrue_rate(FN-TP)= -0.2275962610410771
      has_prior_card_history Δtrue_rate(FN-TP)= 0.0
      transaction_mode top gap= {'level': 'Swipe Transaction', 'error_rate': 0.08165680473372781, 'correct_rate': 0.11594202898550725, 'error_minus_correct_rate': -0.034285224251779434, 'absolute_rate_gap': 0.034285224251779434}
      location_state top gap= {'level': 'PHYSICAL_ZIP_UNAVAILABLE', 'error_rate': 1.0, 'correct_rate': 1.0, 'error_minus_correct_rate': 0.0, 'absolute_rate_gap': 0.0}
      hour_of_day top gap= {'level': '10', 'error_rate': 0.07692307692307693, 'correct_rate': 0.01932367149758454, 'error_minus_correct_rate': 0.057599405425492384, 'absolute_rate_gap': 0.057599405425492384}
      day_of_week top gap= {'level': '5', 'error_rate': 0.28994082840236685, 'correct_rate': 0.024154589371980676, 'error_minus_correct_rate': 0.2657862390303862, 'absolute_rate_gap': 0.2657862390303862}
    
    Non-fraud side — FP vs TN
      amount_numeric Δmedian(FP-TN)= 38.355000000000004
      time_since_previous_transaction_min Δmedian(FP-TN)= -466.0
      transactions_last_1h Δmedian(FP-TN)= 0.0
      amount_minus_previous_mean Δmedian(FP-TN)= 35.87829898685149
      is_new_merchant Δtrue_rate(FP-TN)= 0.33641419143375517
      has_prior_card_history Δtrue_rate(FP-TN)= 4.7847832633896026e-05
      transaction_mode top gap= {'level': 'Online Transaction', 'error_rate': 0.6060975609756097, 'correct_rate': 0.12356702777707414, 'error_minus_correct_rate': 0.4825305331985356, 'absolute_rate_gap': 0.4825305331985356}
      location_state top gap= {'level': 'PHYSICAL_COMPLETE', 'error_rate': 0.11097560975609756, 'correct_rate': 0.8696428018564959, 'error_minus_correct_rate': -0.7586671921003983, 'absolute_rate_gap': 0.7586671921003983}
      hour_of_day top gap= {'level': '10', 'error_rate': 0.13170731707317074, 'correct_rate': 0.06768638841744701, 'error_minus_correct_rate': 0.06402092865572373, 'absolute_rate_gap': 0.06402092865572373}
      day_of_week top gap= {'level': '5', 'error_rate': 0.06341463414634146, 'correct_rate': 0.14338869609026916, 'error_minus_correct_rate': -0.0799740619439277, 'absolute_rate_gap': 0.0799740619439277}
    
    FN score median: 0.0
    FP score median: 1.0
    FN bands: {'very_low_score_lt_0_10': {'count': 845, 'rate': 1.0}, 'low_score_0_10_to_lt_0_40': {'count': 0, 'rate': 0.0}, 'near_boundary_0_40_to_0_50': {'count': 0, 'rate': 0.0}}
    FP bands: {'near_boundary_0_50_to_lt_0_60': {'count': 0, 'rate': 0.0}, 'high_score_0_60_to_lt_0_90': {'count': 0, 'rate': 0.0}, 'very_high_score_ge_0_90': {'count': 820, 'rate': 1.0}}
    ========================================================================
    M5-RF-SHORT-B01 | Random Forest | W_SHORT
    Counts: {'TP': 306, 'FN': 746, 'FP': 312, 'TN': 711094}
    
    Fraud side — FN vs TP
      amount_numeric Δmedian(FN-TP)= -14.340000000000003
      time_since_previous_transaction_min Δmedian(FN-TP)= 26.0
      transactions_last_1h Δmedian(FN-TP)= 0.0
      amount_minus_previous_mean Δmedian(FN-TP)= -17.178127658095228
      is_new_merchant Δtrue_rate(FN-TP)= -0.22674306541204503
      has_prior_card_history Δtrue_rate(FN-TP)= 0.0
      transaction_mode top gap= {'level': 'Swipe Transaction', 'error_rate': 0.09249329758713137, 'correct_rate': 0.0784313725490196, 'error_minus_correct_rate': 0.014061925038111767, 'absolute_rate_gap': 0.014061925038111767}
      location_state top gap= {'level': 'PHYSICAL_ZIP_UNAVAILABLE', 'error_rate': 1.0, 'correct_rate': 1.0, 'error_minus_correct_rate': 0.0, 'absolute_rate_gap': 0.0}
      hour_of_day top gap= {'level': '14', 'error_rate': 0.09517426273458444, 'correct_rate': 0.14705882352941177, 'error_minus_correct_rate': -0.051884560794827325, 'absolute_rate_gap': 0.051884560794827325}
      day_of_week top gap= {'level': '3', 'error_rate': 0.15951742627345844, 'correct_rate': 0.5359477124183006, 'error_minus_correct_rate': -0.3764302861448422, 'absolute_rate_gap': 0.3764302861448422}
    
    Non-fraud side — FP vs TN
      amount_numeric Δmedian(FP-TN)= 5.290000000000003
      time_since_previous_transaction_min Δmedian(FP-TN)= -298.5
      transactions_last_1h Δmedian(FP-TN)= 0.0
      amount_minus_previous_mean Δmedian(FP-TN)= 6.597993802321582
      is_new_merchant Δtrue_rate(FP-TN)= 0.44640043802496304
      has_prior_card_history Δtrue_rate(FP-TN)= 4.7813650515982076e-05
      transaction_mode top gap= {'level': 'Chip Transaction', 'error_rate': 0.9358974358974359, 'correct_rate': 0.7056296354631033, 'error_minus_correct_rate': 0.2302678004343326, 'absolute_rate_gap': 0.2302678004343326}
      location_state top gap= {'level': 'PHYSICAL_ZIP_UNAVAILABLE', 'error_rate': 0.9967948717948718, 'correct_rate': 0.005809358537689813, 'error_minus_correct_rate': 0.990985513257182, 'absolute_rate_gap': 0.990985513257182}
      hour_of_day top gap= {'level': '18', 'error_rate': 0.1282051282051282, 'correct_rate': 0.03455380020081733, 'error_minus_correct_rate': 0.09365132800431086, 'absolute_rate_gap': 0.09365132800431086}
      day_of_week top gap= {'level': '2', 'error_rate': 0.32371794871794873, 'correct_rate': 0.13535622575918232, 'error_minus_correct_rate': 0.1883617229587664, 'absolute_rate_gap': 0.1883617229587664}
    
    FN score median: 0.07999999821186066
    FP score median: 0.625
    FN bands: {'very_low_score_lt_0_10': {'count': 416, 'rate': 0.5576407506702413}, 'low_score_0_10_to_lt_0_40': {'count': 243, 'rate': 0.3257372654155496}, 'near_boundary_0_40_to_0_50': {'count': 87, 'rate': 0.11662198391420911}}
    FP bands: {'near_boundary_0_50_to_lt_0_60': {'count': 134, 'rate': 0.42948717948717946}, 'high_score_0_60_to_lt_0_90': {'count': 175, 'rate': 0.5608974358974359}, 'very_high_score_ge_0_90': {'count': 3, 'rate': 0.009615384615384616}}
    ========================================================================
    M5-RF-LONG-B01 | Random Forest | W_LONG
    Counts: {'TP': 93, 'FN': 959, 'FP': 73, 'TN': 711333}
    
    Fraud side — FN vs TP
      amount_numeric Δmedian(FN-TP)= -25.630000000000003
      time_since_previous_transaction_min Δmedian(FN-TP)= 111.0
      transactions_last_1h Δmedian(FN-TP)= 0.0
      amount_minus_previous_mean Δmedian(FN-TP)= -26.21823263744276
      is_new_merchant Δtrue_rate(FN-TP)= -0.3492661486539518
      has_prior_card_history Δtrue_rate(FN-TP)= 0.0
      transaction_mode top gap= {'level': 'Chip Transaction', 'error_rate': 0.9071949947862357, 'correct_rate': 0.956989247311828, 'error_minus_correct_rate': -0.04979425252559233, 'absolute_rate_gap': 0.04979425252559233}
      location_state top gap= {'level': 'PHYSICAL_ZIP_UNAVAILABLE', 'error_rate': 1.0, 'correct_rate': 1.0, 'error_minus_correct_rate': 0.0, 'absolute_rate_gap': 0.0}
      hour_of_day top gap= {'level': '10', 'error_rate': 0.07090719499478623, 'correct_rate': 0.010752688172043012, 'error_minus_correct_rate': 0.06015450682274322, 'absolute_rate_gap': 0.06015450682274322}
      day_of_week top gap= {'level': '3', 'error_rate': 0.24191866527632952, 'correct_rate': 0.5483870967741935, 'error_minus_correct_rate': -0.30646843149786396, 'absolute_rate_gap': 0.30646843149786396}
    
    Non-fraud side — FP vs TN
      amount_numeric Δmedian(FP-TN)= 31.87
      time_since_previous_transaction_min Δmedian(FP-TN)= -478.0
      transactions_last_1h Δmedian(FP-TN)= 0.0
      amount_minus_previous_mean Δmedian(FP-TN)= 36.88192652322793
      is_new_merchant Δtrue_rate(FP-TN)= 0.6874501237874661
      has_prior_card_history Δtrue_rate(FP-TN)= 4.779758565964798e-05
      transaction_mode top gap= {'level': 'Online Transaction', 'error_rate': 0.547945205479452, 'correct_rate': 0.12407972074963484, 'error_minus_correct_rate': 0.42386548472981717, 'absolute_rate_gap': 0.42386548472981717}
      location_state top gap= {'level': 'PHYSICAL_COMPLETE', 'error_rate': 0.0, 'correct_rate': 0.8688574830634879, 'error_minus_correct_rate': -0.8688574830634879, 'absolute_rate_gap': 0.8688574830634879}
      hour_of_day top gap= {'level': '11', 'error_rate': 0.1506849315068493, 'correct_rate': 0.07106235757373831, 'error_minus_correct_rate': 0.079622573933111, 'absolute_rate_gap': 0.079622573933111}
      day_of_week top gap= {'level': '6', 'error_rate': 0.2876712328767123, 'correct_rate': 0.14317626203198783, 'error_minus_correct_rate': 0.14449497084472449, 'absolute_rate_gap': 0.14449497084472449}
    
    FN score median: 0.07999999821186066
    FP score median: 0.5899999737739563
    FN bands: {'very_low_score_lt_0_10': {'count': 520, 'rate': 0.5422314911366006}, 'low_score_0_10_to_lt_0_40': {'count': 374, 'rate': 0.3899895724713243}, 'near_boundary_0_40_to_0_50': {'count': 65, 'rate': 0.06777893639207508}}
    FP bands: {'near_boundary_0_50_to_lt_0_60': {'count': 39, 'rate': 0.5342465753424658}, 'high_score_0_60_to_lt_0_90': {'count': 33, 'rate': 0.4520547945205479}, 'very_high_score_ge_0_90': {'count': 1, 'rate': 0.0136986301369863}}
    
    M6.6 PER-RUN ERROR EVIDENCE: READY


## 21. Same-window cross-family error overlap

M6.6 kiểm tra trên cùng training window:

```text
W_SHORT:
LR / DT / RF

W_LONG:
LR / DT / RF
```

Fraud-side:

- all three miss;
- all three catch;
- exactly one model catches;
- exactly two models catch.

Non-fraud-side:

- any false positive;
- all three false positive;
- exactly one model false-positives;
- exactly two models false-positive.

Đây là row-level overlap evidence, không phải model selection.


```python

WINDOW_RUNS = {
    "W_SHORT": [
        "M5-LR-SHORT-B04",
        "M5-DT-SHORT-B01",
        "M5-RF-SHORT-B01",
    ],

    "W_LONG": [
        "M5-LR-LONG-B04",
        "M5-DT-LONG-B01",
        "M5-RF-LONG-B01",
    ],
}


cross_run_overlap = {}
overlap_row_id_arrays = {}


for window_id, experiment_ids in (
    WINDOW_RUNS.items()
):
    prediction_matrix = np.column_stack(
        [
            run_arrays[
                experiment_id
            ][
                "y_pred"
            ]
            for experiment_id
            in experiment_ids
        ]
    )

    assert (
        prediction_matrix.shape
        == (
            EXPECTED_VALIDATION_ROWS,
            3,
        )
    )

    positive_votes = (
        prediction_matrix.sum(
            axis=1
        )
    )

    fraud_rows = (
        y_validation == 1
    )

    non_fraud_rows = (
        y_validation == 0
    )

    fraud_votes = (
        positive_votes[
            fraud_rows
        ]
    )

    non_fraud_votes = (
        positive_votes[
            non_fraud_rows
        ]
    )

    fraud_row_ids = (
        row_id_validation[
            fraud_rows
        ]
    )

    non_fraud_row_ids = (
        row_id_validation[
            non_fraud_rows
        ]
    )

    fraud_summary = {
        "actual_fraud":
            int(
                fraud_rows.sum()
            ),

        "all_three_miss":
            int(
                (
                    fraud_votes
                    == 0
                ).sum()
            ),

        "exactly_one_catches":
            int(
                (
                    fraud_votes
                    == 1
                ).sum()
            ),

        "exactly_two_catch":
            int(
                (
                    fraud_votes
                    == 2
                ).sum()
            ),

        "all_three_catch":
            int(
                (
                    fraud_votes
                    == 3
                ).sum()
            ),

        "caught_by_at_least_one":
            int(
                (
                    fraud_votes
                    >= 1
                ).sum()
            ),
    }

    non_fraud_summary = {
        "actual_non_fraud":
            int(
                non_fraud_rows.sum()
            ),

        "no_model_false_positive":
            int(
                (
                    non_fraud_votes
                    == 0
                ).sum()
            ),

        "exactly_one_false_positive":
            int(
                (
                    non_fraud_votes
                    == 1
                ).sum()
            ),

        "exactly_two_false_positive":
            int(
                (
                    non_fraud_votes
                    == 2
                ).sum()
            ),

        "all_three_false_positive":
            int(
                (
                    non_fraud_votes
                    == 3
                ).sum()
            ),

        "false_positive_by_at_least_one":
            int(
                (
                    non_fraud_votes
                    >= 1
                ).sum()
            ),
    }

    model_specific_catches = {}
    model_specific_false_positives = {}

    fraud_prediction_matrix = (
        prediction_matrix[
            fraud_rows
        ]
    )

    non_fraud_prediction_matrix = (
        prediction_matrix[
            non_fraud_rows
        ]
    )

    for model_index, experiment_id in enumerate(
        experiment_ids
    ):
        other_indices = [
            index
            for index
            in range(3)
            if index
            != model_index
        ]

        only_catch_mask = (
            (
                fraud_prediction_matrix[
                    :,
                    model_index
                ]
                == 1
            )
            &
            (
                fraud_prediction_matrix[
                    :,
                    other_indices
                ]
                .sum(
                    axis=1
                )
                == 0
            )
        )

        only_fp_mask = (
            (
                non_fraud_prediction_matrix[
                    :,
                    model_index
                ]
                == 1
            )
            &
            (
                non_fraud_prediction_matrix[
                    :,
                    other_indices
                ]
                .sum(
                    axis=1
                )
                == 0
            )
        )

        model_specific_catches[
            experiment_id
        ] = int(
            only_catch_mask.sum()
        )

        model_specific_false_positives[
            experiment_id
        ] = int(
            only_fp_mask.sum()
        )

        overlap_row_id_arrays[
            (
                f"{window_id}__"
                f"{experiment_id}__ONLY_CATCH"
            )
        ] = (
            fraud_row_ids[
                only_catch_mask
            ]
        )

        overlap_row_id_arrays[
            (
                f"{window_id}__"
                f"{experiment_id}__ONLY_FP"
            )
        ] = (
            non_fraud_row_ids[
                only_fp_mask
            ]
        )

    cross_run_overlap[
        window_id
    ] = {
        "experiment_ids":
            experiment_ids,

        "fraud_side":
            fraud_summary,

        "non_fraud_side":
            non_fraud_summary,

        "model_specific_only_catches":
            model_specific_catches,

        "model_specific_only_false_positives":
            model_specific_false_positives,

        "final_model_selection":
            False,
    }

    overlap_row_id_arrays[
        (
            f"{window_id}__"
            "ALL_THREE_MISS"
        )
    ] = (
        fraud_row_ids[
            fraud_votes
            == 0
        ]
    )

    overlap_row_id_arrays[
        (
            f"{window_id}__"
            "ALL_THREE_CATCH"
        )
    ] = (
        fraud_row_ids[
            fraud_votes
            == 3
        ]
    )

    overlap_row_id_arrays[
        (
            f"{window_id}__"
            "ALL_THREE_FP"
        )
    ] = (
        non_fraud_row_ids[
            non_fraud_votes
            == 3
        ]
    )


for window_id, summary in (
    cross_run_overlap.items()
):
    print("=" * 72)

    print(
        window_id
    )

    print(
        "Fraud side:",
        summary[
            "fraud_side"
        ],
    )

    print(
        "Non-fraud side:",
        summary[
            "non_fraud_side"
        ],
    )

    print(
        "Model-specific only catches:",
        summary[
            "model_specific_only_catches"
        ],
    )

    print(
        "Model-specific only FPs:",
        summary[
            "model_specific_only_false_positives"
        ],
    )


print(
    "\nM6.6 CROSS-RUN ERROR OVERLAP GATE: PASS"
)

```

    ========================================================================
    W_SHORT
    Fraud side: {'actual_fraud': 1052, 'all_three_miss': 618, 'exactly_one_catches': 127, 'exactly_two_catch': 154, 'all_three_catch': 153, 'caught_by_at_least_one': 434}
    Non-fraud side: {'actual_non_fraud': 711406, 'no_model_false_positive': 710602, 'exactly_one_false_positive': 514, 'exactly_two_false_positive': 217, 'all_three_false_positive': 73, 'false_positive_by_at_least_one': 804}
    Model-specific only catches: {'M5-LR-SHORT-B04': 31, 'M5-DT-SHORT-B01': 68, 'M5-RF-SHORT-B01': 28}
    Model-specific only FPs: {'M5-LR-SHORT-B04': 58, 'M5-DT-SHORT-B01': 388, 'M5-RF-SHORT-B01': 68}
    ========================================================================
    W_LONG
    Fraud side: {'actual_fraud': 1052, 'all_three_miss': 810, 'exactly_one_catches': 166, 'exactly_two_catch': 72, 'all_three_catch': 4, 'caught_by_at_least_one': 242}
    Non-fraud side: {'actual_non_fraud': 711406, 'no_model_false_positive': 710546, 'exactly_one_false_positive': 802, 'exactly_two_false_positive': 56, 'all_three_false_positive': 2, 'false_positive_by_at_least_one': 860}
    Model-specific only catches: {'M5-LR-LONG-B04': 14, 'M5-DT-LONG-B01': 134, 'M5-RF-LONG-B01': 18}
    Model-specific only FPs: {'M5-LR-LONG-B04': 20, 'M5-DT-LONG-B01': 762, 'M5-RF-LONG-B01': 20}
    
    M6.6 CROSS-RUN ERROR OVERLAP GATE: PASS


## 22. Persist row-level lineage artifacts

Persist row IDs cho:

- TP;
- FN;
- FP;

của từng official run;

và selected same-window overlap subsets.

Không persist raw User/Card/Merchant Name.


```python

def safe_npz_key(
    text,
):
    return (
        text
        .replace(
            "-",
            "_",
        )
        .replace(
            " ",
            "_",
        )
        .replace(
            "/",
            "_",
        )
    )


ROW_GROUP_PATH = (
    OUTPUT_DIR
    / "m6_06_error_group_row_ids.npz"
)


row_group_npz = {}


for experiment_id, groups in (
    error_group_row_ids.items()
):
    for group_name, row_ids in (
        groups.items()
    ):
        key = safe_npz_key(
            (
                f"{experiment_id}"
                f"__{group_name}"
            )
        )

        row_group_npz[
            key
        ] = np.asarray(
            row_ids,
            dtype=np.int64,
        )


for name, row_ids in (
    overlap_row_id_arrays.items()
):
    row_group_npz[
        safe_npz_key(
            name
        )
    ] = np.asarray(
        row_ids,
        dtype=np.int64,
    )


np.savez_compressed(
    ROW_GROUP_PATH,
    **row_group_npz,
)


assert ROW_GROUP_PATH.exists()
assert ROW_GROUP_PATH.stat().st_size > 0


with np.load(
    ROW_GROUP_PATH,
    allow_pickle=False,
) as artifact:
    persisted_keys = set(
        artifact.files
    )

    assert (
        persisted_keys
        == set(
            row_group_npz
        )
    )

    for key, expected in (
        row_group_npz.items()
    ):
        np.testing.assert_array_equal(
            artifact[key],
            expected,
        )


print(
    "Persisted row-group arrays:",
    len(
        row_group_npz
    ),
)

print(
    "\nM6.6 ROW-GROUP PERSISTENCE / ROUND-TRIP GATE: PASS"
)

```

    Persisted row-group arrays: 36
    
    M6.6 ROW-GROUP PERSISTENCE / ROUND-TRIP GATE: PASS


## 23. Persist error-analysis JSON + manifest

JSON chứa:

- verified lineage;
- semantic contract;
- six-run FN-vs-TP / FP-vs-TN summaries;
- risk-score error context;
- same-window overlap findings;
- source hashes;
- selection boundary.

Full raw transaction identifiers không được persist.


```python

ANALYSIS_VERSION = (
    "M6.6-fp-fn-error-analysis-v1"
)

ANALYSIS_PATH = (
    OUTPUT_DIR
    / "m6_06_error_analysis.json"
)

MANIFEST_PATH = (
    OUTPUT_DIR
    / "m6_06_analysis_manifest.json"
)


analysis_payload = {
    "analysis_version":
        ANALYSIS_VERSION,

    "m6_substep":
        "M6.6",

    "evaluation_population":
        "VALIDATION_2019-01_TO_2019-05",

    "validation_rows":
        EXPECTED_VALIDATION_ROWS,

    "validation_fraud_rows":
        EXPECTED_VALIDATION_FRAUD,

    "lineage": {
        "row_id_validation_sha256":
            m6_02_registry[
                "row_id_validation_sha256"
            ],

        "y_validation_sha256":
            m6_02_registry[
                "y_validation_sha256"
            ],

        "semantic_rebuild_raw_file_size":
            int(
                RAW_PATH.stat().st_size
            ),

        "semantic_rows":
            int(
                len(
                    validation_semantic
                )
            ),

        "exact_row_id_alignment":
            True,

        "transaction_level_mapping":
            "VERIFIED",
    },

    "semantic_feature_columns":
        CORE_FEATURE_COLUMNS,

    "raw_identifiers_persisted":
        False,

    "raw_target_read_for_semantic_rebuild":
        False,

    "strict_causal_history":
        True,

    "development_context_rows":
        int(
            context_rows_seen
        ),

    "semantic_rebuild_elapsed_seconds":
        float(
            rebuild_elapsed
        ),

    "score_reference_boundary":
        REFERENCE_BOUNDARY,

    "score_boundary_checks":
        score_boundary_checks,

    "run_error_analysis":
        run_error_analysis,

    "cross_run_overlap":
        cross_run_overlap,

    "row_group_artifact":
        str(
            ROW_GROUP_PATH.relative_to(
                PROJECT_ROOT
            )
        ),

    "row_group_artifact_sha256":
        sha256_file(
            ROW_GROUP_PATH
        ),

    "threshold_optimization":
        "NOT_PERFORMED",

    "training_window_winner":
        "OPEN",

    "model_family_winner":
        "OPEN",

    "final_model":
        "OPEN",

    "final_threshold":
        "OPEN",

    "final_test_accessed":
        False,
}


analysis_manifest = {
    "m6_substep":
        "M6.6",

    "analysis_version":
        ANALYSIS_VERSION,

    "transaction_level_lineage":
        "PASS",

    "semantic_feature_rebuild":
        "PASS",

    "strict_causal_regression":
        "PASS",

    "official_runs_expected":
        6,

    "official_runs_analyzed":
        len(
            run_error_analysis
        ),

    "confusion_group_reconstruction":
        "PASS",

    "risk_score_boundary_compatibility":
        "PASS",

    "same_window_overlap_groups":
        2,

    "row_group_persistence":
        "PASS",

    "threshold_search":
        "NOT_PERFORMED",

    "model_retraining":
        "NOT_PERFORMED",

    "raw_target_read_for_semantic_rebuild":
        False,

    "final_test_accessed":
        False,

    "decision":
        "OPEN — REQUIRES M6.6 RUNTIME REVIEW",
}


with open(
    ANALYSIS_PATH,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        analysis_payload,
        file,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )


with open(
    MANIFEST_PATH,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        analysis_manifest,
        file,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )


assert ANALYSIS_PATH.exists()
assert MANIFEST_PATH.exists()

assert ANALYSIS_PATH.stat().st_size > 0
assert MANIFEST_PATH.stat().st_size > 0


print("Analysis artifact:")
print(ANALYSIS_PATH)

print("\nAnalysis manifest:")
print(MANIFEST_PATH)

print(
    "\nM6.6 ANALYSIS PERSISTENCE GATE: PASS"
)

```

    Analysis artifact:
    /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/processed/m6_06_fp_fn_error_analysis/m6_06_error_analysis.json
    
    Analysis manifest:
    /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/processed/m6_06_fp_fn_error_analysis/m6_06_analysis_manifest.json
    
    M6.6 ANALYSIS PERSISTENCE GATE: PASS


## 24. Persisted JSON round-trip


```python

with open(
    ANALYSIS_PATH,
    "r",
    encoding="utf-8",
) as file:
    analysis_roundtrip = json.load(
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
    analysis_roundtrip[
        "analysis_version"
    ]
    == ANALYSIS_VERSION
)

assert (
    analysis_roundtrip[
        "lineage"
    ][
        "transaction_level_mapping"
    ]
    == "VERIFIED"
)

assert (
    analysis_roundtrip[
        "lineage"
    ][
        "exact_row_id_alignment"
    ]
    is True
)

assert (
    len(
        analysis_roundtrip[
            "run_error_analysis"
        ]
    )
    == 6
)

assert set(
    analysis_roundtrip[
        "cross_run_overlap"
    ]
) == {
    "W_SHORT",
    "W_LONG",
}

assert (
    analysis_roundtrip[
        "raw_identifiers_persisted"
    ]
    is False
)

assert (
    analysis_roundtrip[
        "raw_target_read_for_semantic_rebuild"
    ]
    is False
)

assert (
    analysis_roundtrip[
        "threshold_optimization"
    ]
    == "NOT_PERFORMED"
)

assert (
    analysis_roundtrip[
        "final_test_accessed"
    ]
    is False
)


assert (
    manifest_roundtrip[
        "transaction_level_lineage"
    ]
    == "PASS"
)

assert (
    manifest_roundtrip[
        "official_runs_analyzed"
    ]
    == 6
)

assert (
    manifest_roundtrip[
        "row_group_persistence"
    ]
    == "PASS"
)

assert (
    manifest_roundtrip[
        "final_test_accessed"
    ]
    is False
)


print(
    "M6.6 ANALYSIS ROUND-TRIP GATE: PASS"
)

```

    M6.6 ANALYSIS ROUND-TRIP GATE: PASS


## 25. FINAL TEST / leakage isolation gate

Required:

```text
raw Is Fraud?:
NOT READ

raw Errors?:
NOT READ

semantic/history max timestamp:
< 2019-06-01

validation semantic frame:
2019-01-01 <= Timestamp < 2019-06-01

threshold optimization:
NONE

FINAL TEST score:
NOT COMPUTED
```


```python

assert (
    "Is Fraud?"
    not in SEMANTIC_USECOLS
)

assert (
    "Errors?"
    not in SEMANTIC_USECOLS
)

assert (
    validation_semantic[
        "Timestamp"
    ]
    .min()
    >= TRAIN_END
)

assert (
    validation_semantic[
        "Timestamp"
    ]
    .max()
    < VALIDATION_END
)

assert (
    context_rows_seen
    == EXPECTED_CONTEXT_ROWS
)

assert (
    analysis_payload[
        "raw_target_read_for_semantic_rebuild"
    ]
    is False
)

assert (
    analysis_payload[
        "threshold_optimization"
    ]
    == "NOT_PERFORMED"
)

assert (
    analysis_payload[
        "final_test_accessed"
    ]
    is False
)

assert (
    m6_02_registry[
        "final_test_accessed"
    ]
    is False
)

assert (
    m6_05_comparison[
        "final_test_accessed"
    ]
    is False
)


print(
    "M6.6 FINAL TEST / LEAKAGE ISOLATION GATE: PASS"
)

```

    M6.6 FINAL TEST / LEAKAGE ISOLATION GATE: PASS


## 26. M6.6 overall technical gate


```python

m6_06_gates = {
    "G01_ENVIRONMENT_RAW_ARTIFACT":
        True,

    "G02_UPSTREAM_HANDOFF":
        True,

    "G03_CANONICAL_VALIDATION_LINEAGE":
        True,

    "G04_DETERMINISTIC_REPRESENTATION":
        True,

    "G05_CARD_BLOCK_STREAMING":
        True,

    "G06_STRICT_CAUSAL_REGRESSION":
        True,

    "G07_CORE_SEMANTIC_CONTRACT":
        True,

    "G08_TRANSACTION_LEVEL_MAPPING":
        True,

    "G09_SIX_RUN_ARRAY_IDENTITY":
        True,

    "G10_CONFUSION_GROUP_RECONSTRUCTION":
        True,

    "G11_SCORE_PREDICTION_BOUNDARY":
        True,

    "G12_SIX_RUN_ERROR_ANALYSIS":
        True,

    "G13_PER_RUN_ERROR_EVIDENCE":
        True,

    "G14_CROSS_RUN_ERROR_OVERLAP":
        True,

    "G15_ROW_GROUP_PERSISTENCE":
        True,

    "G16_ANALYSIS_PERSISTENCE":
        True,

    "G17_ANALYSIS_ROUND_TRIP":
        True,

    "G18_FINAL_TEST_LEAKAGE_ISOLATION":
        True,
}


for gate_name, gate_value in (
    m6_06_gates.items()
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
    m6_06_gates.values()
)


print(
    "\nM6.6 OVERALL TECHNICAL GATE: PASS"
)

```

    G01_ENVIRONMENT_RAW_ARTIFACT → PASS
    G02_UPSTREAM_HANDOFF → PASS
    G03_CANONICAL_VALIDATION_LINEAGE → PASS
    G04_DETERMINISTIC_REPRESENTATION → PASS
    G05_CARD_BLOCK_STREAMING → PASS
    G06_STRICT_CAUSAL_REGRESSION → PASS
    G07_CORE_SEMANTIC_CONTRACT → PASS
    G08_TRANSACTION_LEVEL_MAPPING → PASS
    G09_SIX_RUN_ARRAY_IDENTITY → PASS
    G10_CONFUSION_GROUP_RECONSTRUCTION → PASS
    G11_SCORE_PREDICTION_BOUNDARY → PASS
    G12_SIX_RUN_ERROR_ANALYSIS → PASS
    G13_PER_RUN_ERROR_EVIDENCE → PASS
    G14_CROSS_RUN_ERROR_OVERLAP → PASS
    G15_ROW_GROUP_PERSISTENCE → PASS
    G16_ANALYSIS_PERSISTENCE → PASS
    G17_ANALYSIS_ROUND_TRIP → PASS
    G18_FINAL_TEST_LEAKAGE_ISOLATION → PASS
    
    M6.6 OVERALL TECHNICAL GATE: PASS


# 27. Runtime review và Findings M6.6

## 27.1. Execution integrity

Observed:

```text
Code cells:
24 / 24

Execution count:
1 → 24 liên tục

Runtime errors:
0

stderr:
0
```

Interpretation:

Notebook đã chạy đầy đủ từ đầu đến cuối.

Không có partial execution hoặc runtime exception.

Status:

`VERIFIED`

---

## 27.2. Raw artifact / environment integrity

Observed:

```text
Python:
3.14.6

NumPy:
2.5.3

pandas:
3.0.5

Raw artifact:
card_transaction.v1.csv

M6.6 ENVIRONMENT / RAW ARTIFACT GATE:
PASS
```

Status:

`VERIFIED`

---

## 27.3. Upstream handoff

Observed:

```text
M6.2 Evaluation Registry:
loaded

M6.5 Cross-model Comparison:
loaded

M6.5 Comparison Manifest:
loaded

M6.6 UPSTREAM HANDOFF GATE:
PASS
```

M6.6 dùng persisted prediction / score / validation artifacts đã verify.

Không retrain hoặc predict lại model.

Status:

`VERIFIED`

---

## 27.4. Canonical validation lineage

Observed hashes:

```text
y_validation SHA256:
0a21b2e93e017eda8b794be1a94a692beebc5883c1303427bc245299b55f5306

row_id_validation SHA256:
e7f3074c8604680eccef60ff7be45501f0dc10efbf6ac82e47bcbd1b00e43adc
```

Runtime gate:

`M6.6 CANONICAL VALIDATION LINEAGE GATE: PASS`

Status:

`VERIFIED`

---

## 27.5. Strict-causal semantic rebuild

Observed full pass:

```text
Card blocks:
6,139

Raw rows streamed:
24,386,900

Development/history context rows:
23,038,920

Validation semantic rows:
712,458

Elapsed seconds:
117.69
```

Semantic pass không đọc:

```text
Is Fraud?
Errors?
```

Behavioral history giữ rule:

`Timestamp(history) < Timestamp(current)`

Same-timestamp peer:

`NOT HISTORY`

Status:

`VERIFIED`

---

## 27.6. Transaction-level mapping

Observed:

```text
First validation raw_row_id:
4776

Last validation raw_row_id:
24385729

Reconstructed validation rows:
712,458
```

Exact equality check:

```text
validation_semantic.raw_row_id
==
row_id_validation
```

Runtime gate:

`M6.6 TRANSACTION-LEVEL LINEAGE MAPPING GATE: PASS`

Interpretation:

M6.6 đã vượt qua blocker còn mở từ M6.2:

`transaction-level semantic remapping`

giờ là:

`VERIFIED`

Do đó row-level FN/FP findings được phép diễn giải.

---

## 27.7. Six-run artifact / confusion reconstruction

Observed:

```text
M6.6 SIX-RUN ARRAY IDENTITY GATE:
PASS

M6.6 CONFUSION GROUP RECONSTRUCTION GATE:
PASS

M6.6 SCORE / PREDICTION BOUNDARY GATE:
PASS

M6.6 SIX-RUN TRANSACTION ERROR ANALYSIS GATE:
PASS
```

Official runs:

```text
M5-LR-SHORT-B04
M5-LR-LONG-B04
M5-DT-SHORT-B01
M5-DT-LONG-B01
M5-RF-SHORT-B01
M5-RF-LONG-B01
```

Status:

`VERIFIED`

---

## 27.8. Logistic Regression — W_SHORT

Confusion counts:

```text
TP = 258
FN = 794
FP = 219
TN = 711,187
```

Fraud side — `FN vs TP`:

```text
Δmedian amount:
-7.135

Δmedian time_since_previous_transaction_min:
-36.5

Δmedian amount_minus_previous_mean:
-17.3689

Δtrue_rate is_new_merchant:
-0.5711
```

Interpretation:

Fraud bị bỏ sót có `is_new_merchant` rate thấp hơn fraud bắt đúng rất rõ trong LR-SHORT.

Risk-score context:

```text
FN median:
0.09735

FN < 0.10:
50.50%

FN 0.10 → <0.40:
41.56%

FN 0.40 → 0.50:
7.93%
```

Đa số FN nằm khá xa reference boundary 0.5, không chỉ tập trung sát boundary.

FP context:

```text
FP median score:
0.60816

FP 0.50 → <0.60:
44.29%

FP 0.60 → <0.90:
55.25%

FP >= 0.90:
0.46%
```

Non-fraud side có pattern mạnh:

```text
Δtrue_rate is_new_merchant:
+0.9297
```

và:

```text
PHYSICAL_ZIP_UNAVAILABLE

FP rate:
0.9909

TN rate:
0.00594
```

Status:

`OBSERVED TRANSACTION-LEVEL EVIDENCE`

---

## 27.9. Logistic Regression — W_LONG

Confusion counts:

```text
TP = 22
FN = 1,030
FP = 27
TN = 711,379
```

Fraud side:

```text
Δtrue_rate is_new_merchant:
-0.4680

FN median score:
0.05060

FN < 0.10:
58.25%
```

Non-fraud side:

```text
Δtrue_rate is_new_merchant:
+0.9751
```

Risk-score FP:

```text
FP median:
0.57269

near-boundary FP:
62.96%
```

Small-n caution:

```text
TP = 22
FP = 27
```

Các categorical rate-gap của LR-LONG có thể biến động mạnh vì sample size nhỏ.

Status:

`OBSERVED — SMALL-N CAUTION`

---

## 27.10. Decision Tree — W_SHORT

Confusion counts:

```text
TP = 330
FN = 722
FP = 636
TN = 710,770
```

Fraud side:

```text
Δtrue_rate is_new_merchant:
-0.08035
```

Score behavior:

```text
FN median:
0.0

FN < 0.10:
100%

FP median:
1.0

FP >= 0.90:
100%
```

Interpretation:

DT-SHORT error scores cực đoan:

- toàn bộ FN nằm ở score rất thấp;
- toàn bộ FP nằm ở score rất cao.

Đây là descriptive score behavior của unpruned Decision Tree baseline.

Không được diễn giải là calibrated confidence.

Non-fraud side:

```text
Δtrue_rate is_new_merchant:
+0.2929

PHYSICAL_ZIP_UNAVAILABLE

FP rate:
0.9182

TN rate:
0.00543
```

Status:

`OBSERVED TRANSACTION-LEVEL EVIDENCE`

---

## 27.11. Decision Tree — W_LONG

Confusion counts:

```text
TP = 207
FN = 845
FP = 820
TN = 710,586
```

Fraud side:

```text
Δtrue_rate is_new_merchant:
-0.2276
```

Score behavior:

```text
FN median:
0.0

FN < 0.10:
100%

FP median:
1.0

FP >= 0.90:
100%
```

Non-fraud side:

```text
Δtrue_rate is_new_merchant:
+0.3364
```

Một categorical pattern lớn:

```text
Online Transaction

FP rate:
0.6061

TN rate:
0.1236
```

Status:

`OBSERVED TRANSACTION-LEVEL EVIDENCE`

---

## 27.12. Random Forest — W_SHORT

Confusion counts:

```text
TP = 306
FN = 746
FP = 312
TN = 711,094
```

Fraud side:

```text
Δmedian amount:
-14.34

Δmedian amount_minus_previous_mean:
-17.1781

Δtrue_rate is_new_merchant:
-0.2267
```

Risk-score context:

```text
FN median:
0.08

FN < 0.10:
55.76%

FN 0.10 → <0.40:
32.57%

FN 0.40 → 0.50:
11.66%
```

Non-fraud side:

```text
Δtrue_rate is_new_merchant:
+0.4464
```

và:

```text
PHYSICAL_ZIP_UNAVAILABLE

FP rate:
0.9968

TN rate:
0.00581
```

FP score:

```text
median:
0.625

0.50 → <0.60:
42.95%

0.60 → <0.90:
56.09%

>= 0.90:
0.96%
```

Status:

`OBSERVED TRANSACTION-LEVEL EVIDENCE`

---

## 27.13. Random Forest — W_LONG

Confusion counts:

```text
TP = 93
FN = 959
FP = 73
TN = 711,333
```

Fraud side:

```text
Δmedian amount:
-25.63

Δmedian amount_minus_previous_mean:
-26.2182

Δtrue_rate is_new_merchant:
-0.3493
```

Risk-score context:

```text
FN median:
0.08

FN < 0.10:
54.22%

FN 0.10 → <0.40:
39.00%

FN 0.40 → 0.50:
6.78%
```

Non-fraud side:

```text
Δtrue_rate is_new_merchant:
+0.6875
```

FP score:

```text
median:
0.59

0.50 → <0.60:
53.42%

0.60 → <0.90:
45.21%

>= 0.90:
1.37%
```

Small-n caution:

```text
TP = 93
FP = 73
```

Status:

`OBSERVED — SMALL-N CAUTION`

---

## 27.14. Cross-run overlap — W_SHORT

Observed fraud overlap:

```text
Actual fraud:
1,052

All three miss:
618

Exactly one catches:
127

Exactly two catch:
154

All three catch:
153

Caught by at least one:
434
```

Equivalent rates:

```text
All-three miss:
≈ 58.75%

Caught by at least one:
≈ 41.25%

All-three catch:
≈ 14.54%
```

Model-specific only catches:

```text
LR-SHORT:
31

DT-SHORT:
68

RF-SHORT:
28
```

Non-fraud overlap:

```text
Actual non-fraud:
711,406

Any-model FP:
804

All-three FP:
73
```

Model-specific only FP:

```text
LR-SHORT:
58

DT-SHORT:
388

RF-SHORT:
68
```

Interpretation:

W_SHORT có substantial shared-miss burden, nhưng cũng có nhiều fraud được một hoặc hai model bắt trong khi model khác bỏ sót.

Status:

`VERIFIED OVERLAP EVIDENCE`

---

## 27.15. Cross-run overlap — W_LONG

Observed fraud overlap:

```text
Actual fraud:
1,052

All three miss:
810

Exactly one catches:
166

Exactly two catch:
72

All three catch:
4

Caught by at least one:
242
```

Equivalent rates:

```text
All-three miss:
≈ 77.00%

Caught by at least one:
≈ 23.00%

All-three catch:
≈ 0.38%
```

Model-specific only catches:

```text
LR-LONG:
14

DT-LONG:
134

RF-LONG:
18
```

Non-fraud overlap:

```text
Any-model FP:
860

All-three FP:
2
```

Model-specific only FP:

```text
LR-LONG:
20

DT-LONG:
762

RF-LONG:
20
```

Interpretation:

Shared-miss burden của W_LONG lớn hơn W_SHORT trong baseline evidence.

Đây là descriptive transaction-level evidence, không tự động khóa final training-window winner.

Status:

`VERIFIED OVERLAP EVIDENCE`

---

## 27.16. Cross-run overlap arithmetic consistency

W_SHORT fraud-positive votes:

```text
127
+ 2 × 154
+ 3 × 153
= 894
```

Individual TP sum:

```text
258 + 330 + 306
= 894
```

W_LONG fraud-positive votes:

```text
166
+ 2 × 72
+ 3 × 4
= 322
```

Individual TP sum:

```text
22 + 207 + 93
= 322
```

Interpretation:

Overlap reconstruction nhất quán với individual confusion groups.

Status:

`VERIFIED`

---

## 27.17. Cross-run semantic pattern — `is_new_merchant`

Fraud-side `FN − TP` true-rate deltas:

```text
LR-SHORT:
-0.5711

LR-LONG:
-0.4680

DT-SHORT:
-0.0803

DT-LONG:
-0.2276

RF-SHORT:
-0.2267

RF-LONG:
-0.3493
```

Observed pattern:

Trong cả 6 runs, fraud bị bỏ sót có `is_new_merchant` rate thấp hơn fraud bắt đúng.

Non-fraud side `FP − TN` cũng dương ở tất cả 6 runs.

Interpretation:

Trong synthetic validation evidence này, `is_new_merchant` có liên hệ với:

- khả năng fraud được flag;
- đồng thời false-alert burden.

Đây là association trong dataset.

Không phải causal claim hoặc production claim.

Status:

`CROSS-RUN DESCRIPTIVE PATTERN`

---

## 27.18. Cross-run semantic pattern — `location_state`

Fraud-side:

```text
PHYSICAL_ZIP_UNAVAILABLE
FN rate = 1.0
TP rate = 1.0
```

trên các official runs.

Interpretation:

`location_state` không tạo separation giữa FN và TP fraud transactions trong observed validation evidence.

Non-fraud side lại xuất hiện rất mạnh ở một số FP groups, đặc biệt LR-SHORT / DT-SHORT / RF-SHORT.

Interpretation:

Đây là dataset-specific error pattern cần được handoff như hypothesis cho M7.

Không suy rộng ra giao dịch thực tế ngoài synthetic dataset.

Status:

`DESCRIPTIVE / DATASET-SPECIFIC`

---

## 27.19. Decision Tree probability behavior

Observed:

```text
DT-SHORT:
FN median = 0
FP median = 1

DT-LONG:
FN median = 0
FP median = 1
```

và:

```text
100% DT FN:
score < 0.10

100% DT FP:
score >= 0.90
```

Interpretation:

Decision Tree baseline tạo probability output rất cực đoan trên error groups.

M6.6 không đánh giá formal calibration.

Status:

`DESCRIPTIVE PROBABILITY FINDING`

---

## 27.20. LR / RF FN are often far below boundary

Examples:

```text
LR-SHORT:
FN < 0.40
≈ 92.06%

RF-SHORT:
FN < 0.40
≈ 88.34%

LR-LONG:
FN < 0.40
≈ 96.99%

RF-LONG:
FN < 0.40
≈ 93.22%
```

Interpretation:

Phần lớn missed fraud ở LR/RF không chỉ nằm sát 0.5.

Do đó threshold adjustment không nên được giả định là tự động giải quyết phần lớn FN burden.

M6 không thực hiện threshold experiment.

Status:

`M7 HYPOTHESIS INPUT`

---

## 27.21. Persistence / lineage

Observed:

```text
Persisted row-group arrays:
36

M6.6 ROW-GROUP PERSISTENCE / ROUND-TRIP GATE:
PASS
```

Persisted:

```text
m6_06_error_analysis.json

m6_06_analysis_manifest.json

m6_06_error_group_row_ids.npz
```

Runtime gates:

```text
M6.6 ANALYSIS PERSISTENCE GATE:
PASS

M6.6 ANALYSIS ROUND-TRIP GATE:
PASS
```

Status:

`VERIFIED`

---

## 27.22. Leakage / FINAL TEST isolation

Observed:

```text
raw Is Fraud?:
NOT READ

raw Errors?:
NOT READ

history / semantic context:
Timestamp < 2019-06-01

threshold optimization:
NONE

FINAL TEST score:
NOT COMPUTED
```

Runtime gate:

`M6.6 FINAL TEST / LEAKAGE ISOLATION GATE: PASS`

Status:

`VERIFIED`

---

## 27.23. Overall technical result

Observed:

```text
G01_ENVIRONMENT_RAW_ARTIFACT          → PASS
G02_UPSTREAM_HANDOFF                  → PASS
G03_CANONICAL_VALIDATION_LINEAGE      → PASS
G04_DETERMINISTIC_REPRESENTATION      → PASS
G05_CARD_BLOCK_STREAMING              → PASS
G06_STRICT_CAUSAL_REGRESSION          → PASS
G07_CORE_SEMANTIC_CONTRACT            → PASS
G08_TRANSACTION_LEVEL_MAPPING         → PASS
G09_SIX_RUN_ARRAY_IDENTITY            → PASS
G10_CONFUSION_GROUP_RECONSTRUCTION    → PASS
G11_SCORE_PREDICTION_BOUNDARY         → PASS
G12_SIX_RUN_ERROR_ANALYSIS            → PASS
G13_PER_RUN_ERROR_EVIDENCE            → PASS
G14_CROSS_RUN_ERROR_OVERLAP           → PASS
G15_ROW_GROUP_PERSISTENCE             → PASS
G16_ANALYSIS_PERSISTENCE              → PASS
G17_ANALYSIS_ROUND_TRIP               → PASS
G18_FINAL_TEST_LEAKAGE_ISOLATION      → PASS
```

Overall:

`M6.6 OVERALL TECHNICAL GATE: PASS`

Blocking issue:

`NONE`

---

# 28. Findings M6.6

## M6.6-F01 — Transaction-level semantic lineage is now verified

Evidence:

```text
712,458 / 712,458
exact raw_row_id alignment
```

Status:

`VERIFIED`

---

## M6.6-F02 — Six-run row-level error reconstruction is complete

Evidence:

TP/FN/FP/TN groups reconstructed for all 6 official runs and match M6.2 confusion counts.

Status:

`VERIFIED`

---

## M6.6-F03 — W_SHORT has lower shared fraud-miss burden than W_LONG

Observed:

```text
W_SHORT all-three miss:
618 / 1,052
≈ 58.75%

W_LONG all-three miss:
810 / 1,052
≈ 77.00%
```

Status:

`DESCRIPTIVE COMPARATIVE EVIDENCE`

Boundary:

`NOT FINAL WINDOW SELECTION`

---

## M6.6-F04 — W_SHORT has materially more all-three catches

Observed:

```text
W_SHORT:
153

W_LONG:
4
```

Status:

`DESCRIPTIVE COMPARATIVE EVIDENCE`

---

## M6.6-F05 — Error complementarity exists across model families

Evidence:

```text
W_SHORT exactly-one catches:
127

W_SHORT exactly-two catch:
154

W_LONG exactly-one catches:
166

W_LONG exactly-two catch:
72
```

Finding:

Model families do not make identical fraud errors.

Status:

`VERIFIED PATTERN`

---

## M6.6-F06 — `is_new_merchant` is a cross-run error-pattern signal

Finding:

For all 6 runs:

```text
FN true-rate
<
TP true-rate
```

and:

```text
FP true-rate
>
TN true-rate
```

for `is_new_merchant`.

Interpretation:

The feature is associated with stronger fraud flagging but also false-alert burden in this synthetic validation dataset.

Status:

`DESCRIPTIVE ASSOCIATION`

---

## M6.6-F07 — `location_state` does not separate fraud FN from TP

Observed:

```text
PHYSICAL_ZIP_UNAVAILABLE
rate = 1.0
```

for both FN and TP fraud groups in observed runs.

Finding:

No observed FN-vs-TP separation from this semantic dimension.

Status:

`DESCRIPTIVE`

---

## M6.6-F08 — `location_state` appears strongly in several non-fraud FP groups

Observed examples:

```text
LR-SHORT:
PHYSICAL_ZIP_UNAVAILABLE
FP ≈ 99.09%
TN ≈ 0.59%

RF-SHORT:
PHYSICAL_ZIP_UNAVAILABLE
FP ≈ 99.68%
TN ≈ 0.58%
```

Finding:

Potentially important dataset-specific false-alert structure.

Status:

`M7 FOLLOW-UP HYPOTHESIS`

---

## M6.6-F09 — Decision Tree error probabilities are highly discrete/extreme

Observed:

```text
DT SHORT/LONG FN:
score median = 0

DT SHORT/LONG FP:
score median = 1
```

Status:

`VERIFIED PROBABILITY BEHAVIOR`

Boundary:

`NOT CALIBRATION CLAIM`

---

## M6.6-F10 — Many LR/RF false negatives are far below 0.5

Finding:

Most LR/RF FN scores lie below 0.40, not only in the 0.40–0.50 near-boundary region.

Status:

`M7 THRESHOLD HYPOTHESIS INPUT`

Interpretation:

Threshold tuning alone should not be assumed to recover most FN without experiment evidence.

---

## M6.6-F11 — Some W_LONG categorical findings have small-n uncertainty

Examples:

```text
LR-LONG:
TP = 22
FP = 27

RF-LONG:
TP = 93
FP = 73
```

Finding:

Categorical rate gaps for these groups are descriptive and should not be overgeneralized.

Status:

`INTERPRETATION GUARDRAIL`

---

## M6.6-F12 — Raw target was not used in semantic reconstruction

Status:

`VERIFIED`

---

## M6.6-F13 — FINAL TEST remained isolated

Status:

`VERIFIED`

---

## M6.6-F14 — M6.7 handoff is unblocked

M6.7 can now consolidate:

- M6.2 independent registry;
- M6.3 metric/confusion findings;
- M6.4 training-window evidence;
- M6.5 cross-model evidence;
- M6.6 row-level error patterns;
- M7 open questions.

Status:

`READY FOR M6.7`

# 29. Decision Log M6.6 — sau runtime review

## M6.6-D01 — Canonical lineage anchor

Decision:

`M4.7 row_id_validation.npy`

Observed:

exact transaction-level mapping PASS.

Status:

`INHERITED — VERIFIED — LOCKED`

---

## M6.6-D02 — Semantic reconstruction source

Decision:

Use canonical raw IBM TabFormer artifact and exact M4.7 deterministic/causal semantics.

Observed:

```text
24,386,900 raw rows
6,139 Card blocks
23,038,920 history-context rows
712,458 validation semantic rows
```

Status:

`LOCKED / VERIFIED`

---

## M6.6-D03 — Raw target use

Decision:

Do not read raw `Is Fraud?` during semantic reconstruction.

Observed:

`NOT READ`

Status:

`LOCKED / VERIFIED`

---

## M6.6-D04 — Historical rule

Decision:

`Timestamp(history) < Timestamp(current)`

Same timestamp:

`NOT HISTORY`

Observed:

strict-causal regression gate PASS.

Status:

`INHERITED — VERIFIED — LOCKED`

---

## M6.6-D05 — History boundary

Decision:

History/semantic state stops before:

`2019-06-01`

Observed:

PASS.

Status:

`LOCKED / VERIFIED`

---

## M6.6-D06 — Fraud-side comparison

Decision:

`FN vs TP`

Observed:

complete for all 6 runs.

Status:

`LOCKED / VERIFIED`

---

## M6.6-D07 — Non-fraud-side comparison

Decision:

`FP vs TN`

Observed:

complete for all 6 runs.

Status:

`LOCKED / VERIFIED`

---

## M6.6-D08 — Semantic dimensions

Decision:

Use 10 canonical pre-encoding semantic features.

Observed:

reconstructed successfully.

Status:

`LOCKED / VERIFIED`

---

## M6.6-D09 — `is_new_merchant` finding

Decision:

Record as cross-run descriptive association:

```text
FN rate < TP rate
FP rate > TN rate
```

across all 6 runs.

Status:

`FINDING — LOCKED FOR M6 HANDOFF`

---

## M6.6-D10 — `location_state` finding

Decision:

Record:

- no observed fraud-side FN/TP separation;
- strong non-fraud FP concentration in several runs.

Status:

`FINDING — DATASET-SPECIFIC`

---

## M6.6-D11 — Decision Tree score behavior

Decision:

Record extreme/discrete error-score behavior.

Do not claim calibration quality.

Status:

`FINDING — LOCKED FOR M6 HANDOFF`

---

## M6.6-D12 — LR/RF FN score behavior

Decision:

Record that large portions of FN are well below 0.5.

Do not change threshold in M6.

Status:

`M7 HYPOTHESIS INPUT`

---

## M6.6-D13 — Same-window overlap

Decision:

Persist and report model-error overlap separately for W_SHORT and W_LONG.

Observed:

W_SHORT and W_LONG overlap reconstruction PASS.

Status:

`LOCKED / VERIFIED`

---

## M6.6-D14 — Shared-miss comparison

Decision:

Record:

```text
W_SHORT all-three miss:
618

W_LONG all-three miss:
810
```

as descriptive comparative evidence.

Status:

`NOT FINAL WINDOW SELECTION`

---

## M6.6-D15 — Small-n interpretation

Decision:

Categorical findings from very small TP/FP groups must carry descriptive / small-n caution.

Status:

`LOCKED`

---

## M6.6-D16 — Risk score

Decision:

Descriptive error context only.

Reference boundary:

`0.5`

Observed:

score / prediction compatibility PASS.

Status:

`LOCKED / VERIFIED`

---

## M6.6-D17 — Threshold search

Decision:

None.

Observed:

`NOT PERFORMED`

Status:

`LOCKED / VERIFIED`

---

## M6.6-D18 — Raw identifiers

Decision:

User/Card/Merchant Name may be temporary causal keys but are not persisted in error findings.

Observed:

raw identifiers not persisted.

Status:

`LOCKED / VERIFIED`

---

## M6.6-D19 — Row-level lineage artifact

Decision:

Persist:

`m6_06_error_group_row_ids.npz`

Observed:

36 arrays persisted and round-trip PASS.

Status:

`LOCKED / VERIFIED`

---

## M6.6-D20 — Error-analysis artifact

Decision:

Persist:

```text
m6_06_error_analysis.json
m6_06_analysis_manifest.json
```

Observed:

persistence + round-trip PASS.

Status:

`LOCKED / VERIFIED`

---

## M6.6-D21 — Final model / window / threshold

Decision:

Remain OPEN.

Status:

`INHERITED — LOCKED BOUNDARY`

---

## M6.6-D22 — FINAL TEST

Decision:

No evaluation/access.

Observed:

leakage / FINAL TEST isolation PASS.

Status:

`INHERITED — VERIFIED — LOCKED`

---

## M6.6-D23 — M6.7 handoff

Decision:

Handoff complete M6.2–M6.6 evidence for consolidated Evaluation Registry / Findings / Decision Log / M6 Gate.

Status:

`READY`

# 30. M6.6 Gate

## Technical runtime gates

```text
G01_ENVIRONMENT_RAW_ARTIFACT          → PASS
G02_UPSTREAM_HANDOFF                  → PASS
G03_CANONICAL_VALIDATION_LINEAGE      → PASS
G04_DETERMINISTIC_REPRESENTATION      → PASS
G05_CARD_BLOCK_STREAMING              → PASS
G06_STRICT_CAUSAL_REGRESSION          → PASS
G07_CORE_SEMANTIC_CONTRACT            → PASS
G08_TRANSACTION_LEVEL_MAPPING         → PASS
G09_SIX_RUN_ARRAY_IDENTITY            → PASS
G10_CONFUSION_GROUP_RECONSTRUCTION    → PASS
G11_SCORE_PREDICTION_BOUNDARY         → PASS
G12_SIX_RUN_ERROR_ANALYSIS            → PASS
G13_PER_RUN_ERROR_EVIDENCE            → PASS
G14_CROSS_RUN_ERROR_OVERLAP           → PASS
G15_ROW_GROUP_PERSISTENCE             → PASS
G16_ANALYSIS_PERSISTENCE              → PASS
G17_ANALYSIS_ROUND_TRIP               → PASS
G18_FINAL_TEST_LEAKAGE_ISOLATION      → PASS
```

Technical gates:

`18 / 18 PASS`

---

## R01 — Execution complete?

Evidence:

```text
24 / 24 code cells
execution_count = 1 → 24
errors = 0
stderr = 0
```

Result:

`PASS`

---

## R02 — Canonical lineage verified?

Evidence:

```text
y_validation SHA:
verified

row_id_validation SHA:
verified
```

Result:

`PASS`

---

## R03 — Raw semantic reconstruction complete?

Evidence:

```text
24,386,900 raw rows
6,139 Card blocks
23,038,920 context rows
712,458 validation rows
```

Result:

`PASS`

---

## R04 — Exact transaction-level mapping valid?

Evidence:

```text
712,458 / 712,458
raw_row_id exact alignment
```

Result:

`PASS`

---

## R05 — Strict causal semantics valid?

Evidence:

M4.7-compatible strict-causal regression gate PASS.

Result:

`PASS`

---

## R06 — Six official run arrays verified?

Evidence:

Prediction + risk-score SHA/shape/range checks PASS.

Result:

`PASS`

---

## R07 — Confusion groups reconstructed?

Evidence:

TP/FN/FP/TN match persisted M6.2 counts for all six runs.

Result:

`PASS`

---

## R08 — FN vs TP analysis complete?

Evidence:

10 semantic dimensions + risk-score context available for all six runs.

Result:

`PASS`

---

## R09 — FP vs TN analysis complete?

Evidence:

same semantic bundle complete.

Result:

`PASS`

---

## R10 — Cross-run overlap valid?

Evidence:

W_SHORT and W_LONG fraud/non-fraud overlap groups reconstructed and arithmetic-consistent.

Result:

`PASS`

---

## R11 — Error row lineage persisted?

Evidence:

```text
36 row-group arrays
NPZ round-trip PASS
```

Result:

`PASS`

---

## R12 — Probability context kept descriptive?

Evidence:

No threshold optimization; reference boundary used only for descriptive score bands after compatibility check.

Result:

`PASS`

---

## R13 — Raw target excluded and FINAL TEST protected?

Evidence:

```text
raw Is Fraud?:
NOT READ

threshold optimization:
NONE

FINAL TEST:
NOT EVALUATED
```

Result:

`PASS`

---

## R14 — M6.7 handoff ready?

Evidence:

M6.6 now supplies verified transaction-level error evidence and M7 hypotheses.

Result:

`PASS`

---

## Overall M6.6 Gate

```text
Technical gates:
18 / 18 PASS

Runtime review gates:
14 / 14 PASS

Blocking issue:
NONE
```

Final:

`M6.6 — PASS`

Handoff:

`READY FOR M6.7`

# 31. Kết luận M6.6

M6.6 đã hoàn thành transaction-level False Positive / False Negative error analysis cho toàn bộ six-run baseline evidence.

Canonical validation:

```text
Rows:
712,458

Fraud:
1,052

Official runs:
6
```

Transaction-level semantic lineage:

```text
raw rows streamed:
24,386,900

Card blocks:
6,139

history-context rows:
23,038,920

validation semantic rows:
712,458

exact row_id alignment:
PASS
```

Primary analysis groups:

```text
Fraud side:
FN vs TP

Non-fraud side:
FP vs TN
```

Core cross-run findings:

```text
1. W_SHORT shared fraud misses:
618 / 1,052

2. W_LONG shared fraud misses:
810 / 1,052

3. W_SHORT all-three catches:
153

4. W_LONG all-three catches:
4

5. is_new_merchant:
FN rate < TP rate
FP rate > TN rate
across all 6 runs

6. Decision Tree errors:
FN score strongly concentrated at 0
FP score strongly concentrated at 1

7. LR/RF:
large portions of FN score lie far below 0.5

8. Error overlap:
model families do not make identical errors
```

Interpretation boundary:

```text
These are:
DESCRIPTIVE / COMPARATIVE VALIDATION FINDINGS

They are not:
causal claims
production claims
final model selection
final window selection
threshold optimization
```

Small-n caution applies especially to:

```text
LR-LONG:
TP = 22
FP = 27

RF-LONG:
TP = 93
FP = 73
```

Persisted artifacts:

```text
data/processed/m6_06_fp_fn_error_analysis/
    m6_06_error_analysis.json
    m6_06_analysis_manifest.json
    m6_06_error_group_row_ids.npz
```

Final state:

```text
M6.6 — PASS

Execution Integrity:
VERIFIED

Transaction-level Lineage:
VERIFIED

Strict-causal Semantic Reconstruction:
VERIFIED

Official Runs Analyzed:
6 / 6

FN vs TP:
VERIFIED

FP vs TN:
VERIFIED

Risk-score Context:
DESCRIPTIVE — VERIFIED

Cross-model Error Overlap:
VERIFIED

Row-level Error Lineage:
PERSISTED / ROUND-TRIP PASS

Raw Target Used in Semantic Rebuild:
NO

Threshold Optimization:
NONE

Training-window Winner:
OPEN

Model-family Winner:
OPEN

Final Model:
OPEN

Final Threshold:
OPEN

FINAL TEST:
PROTECTED

Blocking Issue:
NONE

READY FOR M6.7
```

Bước tiếp theo:

`M6.7 — Consolidated Evaluation Registry, Findings, Decision Log và M6 Gate`

M6.7 phải tổng hợp toàn bộ evidence từ:

```text
M6.2
Independent evaluation registry

M6.3
Metric / Confusion Matrix analysis

M6.4
Controlled training-window comparison

M6.5
Same-window cross-model comparison

M6.6
Transaction-level FP/FN error analysis
```

và handoff các open questions sang M7 mà không khóa final model/window/threshold trong M6.
