# M4.7 — Tạo baseline-ready modeling matrix và pipeline audit

## Vai trò

M4.6 đã khóa leakage-safe preprocessing contract.

M4.7 là Gate kỹ thuật tạo representation mà milestone modeling có thể sử dụng trực tiếp:

- `X_train`;
- `y_train`;
- `X_validation`;
- `y_validation`.

Do project vẫn giữ hai training-window candidate, M4.7 phải tạo:

### W_LONG

- `X_train_w_long`
- `y_train_w_long`
- `X_validation_w_long`
- `y_validation`

### W_SHORT

- `X_train_w_short`
- `y_train_w_short`
- `X_validation_w_short`
- `y_validation`

Hai validation feature matrices khác nhau về learned preprocessing state vì scaler/vocabulary phải được fit từ TRAIN strategy tương ứng.

Target VALIDATION là cùng một vector.

## Câu hỏi trung tâm

> Có thể tạo full baseline-ready matrices cho W_LONG và W_SHORT bằng cùng feature contract, đúng temporal boundary, đúng learned preprocessing state và không tạo leakage hay không?

## Những việc M4.7 phải chứng minh

- row count đúng;
- temporal boundary đúng;
- target mapping đúng trên development population;
- X không chứa target;
- X không chứa raw User/Card/Merchant Name/Errors?;
- behavioral history vẫn strict causal;
- preprocessing state chỉ fit từ TRAIN;
- VALIDATION chỉ transform;
- unknown category xử lý được;
- NaN/inf không còn trong matrix;
- feature width/order ổn định;
- W_LONG/W_SHORT dùng cùng output schema;
- X/y/lineage alignment đúng;
- repeated transform deterministic;
- artifact modeling matrix có thể được lưu và audit.

## Không thuộc M4.7

- model fitting;
- class weighting;
- resampling;
- feature-performance comparison;
- hyperparameter tuning;
- threshold selection;
- FINAL TEST evaluation.

## Trạng thái khi bắt đầu

M4.6:

`PASS`

Preprocessing Specification:

`LOCKED FOR BASELINE V1`

M4.7 Gate:

`OPEN`

# 1. Contract kế thừa từ M4.6

## Core pre-encoding features

### Numerical

- `amount_numeric`
- `time_since_previous_transaction_min`
- `transactions_last_1h`
- `amount_minus_previous_mean`

### Boolean

- `is_new_merchant`
- `has_prior_card_history`

### Categorical

- `transaction_mode`
- `location_state`
- `hour_of_day`
- `day_of_week`

Tổng:

`10 pre-encoding features`

## Numeric preprocessing

`StandardScaler`

Learned state:

`TRAIN only`

Structural cold-start NA:

- scaler không học từ NA;
- transform giữ NA;
- sau scaling, NA → `0.0`;
- `has_prior_card_history` giữ semantic history availability.

## Categorical preprocessing

`OneHotEncoder`

Vocabulary:

`TRAIN only`

Unknown:

explicit token:

`__UNKNOWN__`

VALIDATION:

`transform only`

## Output contract

Representation:

`CSR sparse matrix`

Dtype:

`float32`

Expected width:

`47`

W_LONG và W_SHORT:

- learned state riêng;
- cùng procedure;
- cùng feature names/order.

## Conditional / experiment features

MCC, raw location, month, Amount alternatives và support states không được M4.7 tự động đưa vào baseline matrix.

# 2. Dependency và môi trường


```python
from pathlib import Path
from collections import Counter
from dataclasses import dataclass
import json
import sys
import time

import numpy as np
import pandas as pd

try:
    from scipy import sparse
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Thiếu scipy. Chạy `%pip install scipy scikit-learn`, "
        "sau đó Restart Kernel và Run All."
    ) from exc

try:
    from sklearn.preprocessing import (
        StandardScaler,
        OneHotEncoder,
    )
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Thiếu scikit-learn. Chạy `%pip install scipy scikit-learn`, "
        "sau đó Restart Kernel và Run All."
    ) from exc


DATA_RELATIVE_PATH = (
    Path("data")
    / "raw"
    / "ibm_tabformer"
    / "card_transaction.v1.csv"
)

OUTPUT_RELATIVE_DIR = (
    Path("data")
    / "processed"
    / "m4_07_baseline_ready"
)

candidate_roots = [
    Path.cwd(),
    Path.cwd().parent,
    Path.cwd().parent.parent,
]

PROJECT_ROOT = None

for candidate in candidate_roots:
    candidate = candidate.resolve()

    if (
        candidate
        / DATA_RELATIVE_PATH
    ).exists():
        PROJECT_ROOT = candidate
        break

if PROJECT_ROOT is None:
    raise FileNotFoundError(
        "Không xác định được PROJECT_ROOT."
    )

DATA_PATH = PROJECT_ROOT / DATA_RELATIVE_PATH
OUTPUT_DIR = PROJECT_ROOT / OUTPUT_RELATIVE_DIR

EXPECTED_FILE_SIZE = 2_354_626_737

CHUNK_SIZE = 500_000

UNKNOWN_TOKEN = "__UNKNOWN__"

PIPELINE_VERSION = "M4.7-baseline-v1"

SAVE_ARTIFACTS = True

EXPECTED_OUTPUT_WIDTH = 47


print("Python executable:")
print(sys.executable)

print("\nPROJECT_ROOT:")
print(PROJECT_ROOT)

print("\nDATA_PATH:")
print(DATA_PATH)

print("\nOUTPUT_DIR:")
print(OUTPUT_DIR)

print("\nRaw file exists:")
print(DATA_PATH.exists())

print("\nRaw file size:")
print(DATA_PATH.stat().st_size)

print("\nSAVE_ARTIFACTS:")
print(SAVE_ARTIFACTS)
```

    Python executable:
    /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/.venv/bin/python
    
    PROJECT_ROOT:
    /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening
    
    DATA_PATH:
    /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/raw/ibm_tabformer/card_transaction.v1.csv
    
    OUTPUT_DIR:
    /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/processed/m4_07_baseline_ready
    
    Raw file exists:
    True
    
    Raw file size:
    2354626737
    
    SAVE_ARTIFACTS:
    True



```python
assert DATA_PATH.exists()

assert (
    DATA_PATH.stat().st_size
    == EXPECTED_FILE_SIZE
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

print(
    "M4.7 ENVIRONMENT / ARTIFACT GATE: PASS"
)
```

    M4.7 ENVIRONMENT / ARTIFACT GATE: PASS


# 3. Temporal và target contract


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

VALIDATION_END = pd.Timestamp(
    "2019-06-01"
)


EXPECTED_W_LONG_ROWS = 6_855_270

EXPECTED_W_SHORT_ROWS = 1_721_615

EXPECTED_VALIDATION_ROWS = 712_458

EXPECTED_CONTEXT_ROWS_BEFORE_VALIDATION_END = (
    23_038_920
)

EXPECTED_CARD_COUNT = 6_139


EXPECTED_W_LONG_FRAUD = 9_606

EXPECTED_W_SHORT_FRAUD = 2_491

EXPECTED_VALIDATION_FRAUD = 1_052


def map_development_target(
    target_series,
):
    target = (
        target_series
        .astype("string")
        .str.strip()
    )

    observed = set(
        target.dropna().unique()
    )

    unexpected = (
        observed
        - {"Yes", "No"}
    )

    if unexpected:
        raise ValueError(
            "Unexpected target values: "
            f"{sorted(unexpected)}"
        )

    if target.isna().any():
        raise ValueError(
            "Missing target trong development rows."
        )

    return (
        target
        .eq("Yes")
        .astype(np.int8)
        .to_numpy()
    )
```

# 4. Canonical deterministic representation helpers


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
            "year": df["Year"],
            "month": df["Month"],
            "day": df["Day"],
        },
        errors="coerce",
    )

    time_part = pd.to_timedelta(
        df["Time"]
        .astype("string")
        + ":00",
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
        df["Merchant City"]
    )

    # M4.3 contract:
    # strip deterministic nhưng không auto casefold.
    city_online = (
        city
        .eq("ONLINE")
        .fillna(False)
    )

    state_missing = (
        df["Merchant State"]
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
```

# 5. Card-block streaming helper


```python
FIT_USECOLS = [
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


MATRIX_USECOLS = [
    *FIT_USECOLS,
    "Is Fraud?",
]


assert "Is Fraud?" not in FIT_USECOLS

assert "Errors?" not in FIT_USECOLS

assert "Errors?" not in MATRIX_USECOLS


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
            chunk["Amount"]
        )

        if (
            chunk["Timestamp"]
            .isna()
            .any()
        ):
            raise ValueError(
                "Timestamp parse failure."
            )

        if (
            chunk["Amount_numeric"]
            .isna()
            .any()
        ):
            raise ValueError(
                "Amount parse failure."
            )

        if (
            chunk["Merchant Name"]
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

        if (
            "Is Fraud?"
            in usecols
        ):
            working_columns.append(
                "Is Fraud?"
            )

        working = (
            chunk[
                working_columns
            ]
        )

        users = (
            working["User"]
            .to_numpy()
        )

        cards = (
            working["Card"]
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
```

# 6. Strict-causal behavioral builder kế thừa M4.5


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
        len(group_starts),
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
        len(group_starts),
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
        len(card_df),
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
```

# 7. Strict-causal regression unit test


```python
causal_test_df = pd.DataFrame(
    {
        "raw_row_id": [
            0,
            1,
            2,
            3,
        ],
        "Timestamp": pd.to_datetime(
            [
                "2018-01-01 09:00:00",
                "2018-01-01 10:00:00",
                "2018-01-01 10:00:00",
                "2018-01-01 10:30:00",
            ]
        ),
        "Amount_numeric": [
            10.0,
            20.0,
            30.0,
            40.0,
        ],
        "Merchant Name": [
            100,
            200,
            200,
            100,
        ],
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
    "M4.7 STRICT CAUSAL REGRESSION GATE: PASS"
)
```

    M4.7 STRICT CAUSAL REGRESSION GATE: PASS


# 8. Core feature frame


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
                    timestamp.dt.hour
                    .astype("Int8")
                    .astype("string")
                    .to_numpy()
                ),

            "day_of_week":
                (
                    timestamp.dt.dayofweek
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
                frame[column]
                .isna()
                .sum()
            )
        for column in CORE_FEATURE_COLUMNS
        if (
            frame[column]
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

    assert (
        set(
            CORE_FEATURE_COLUMNS
        )
        .isdisjoint(
            PROHIBITED_DIRECT_COLUMNS
        )
    )

    return frame


print(
    "Core pre-encoding feature count:",
    len(
        CORE_FEATURE_COLUMNS
    ),
)

print(
    "Core feature columns:"
)

for column in (
    CORE_FEATURE_COLUMNS
):
    print(column)
```

    Core pre-encoding feature count: 10
    Core feature columns:
    amount_numeric
    time_since_previous_transaction_min
    transactions_last_1h
    amount_minus_previous_mean
    is_new_merchant
    has_prior_card_history
    transaction_mode
    location_state
    hour_of_day
    day_of_week


# 9. Leakage-safe preprocessing helpers


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
        str(value)
        for value in values
    ]

    if column in {
        "hour_of_day",
        "day_of_week",
    }:
        return sorted(
            values,
            key=lambda x: int(x),
        )

    return sorted(
        values
    )


@dataclass
class PreprocessingBundle:
    strategy: str
    scaler: StandardScaler
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
                "đã xuất hiện trong TRAIN."
            )

        categories.append(
            train_categories
            + [
                UNKNOWN_TOKEN
            ]
        )

    encoder = (
        make_one_hot_encoder(
            categories
        )
    )

    max_len = max(
        len(values)
        for values
        in categories
    )

    synthetic = {}

    for column, values in zip(
        CATEGORICAL_COLUMNS,
        categories,
    ):
        synthetic[column] = [
            values[
                index
                % len(values)
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
            frame[column]
            .astype("string")
        )

        if (
            values
            .isna()
            .any()
        ):
            raise ValueError(
                "Unexpected missing "
                f"categorical: {column}"
            )

        known = set(
            bundle
            .category_vocab[
                column
            ]
        )

        mapped[column] = (
            values.where(
                values.isin(
                    known
                ),
                UNKNOWN_TOKEN,
            )
        )

    return mapped


def build_feature_names(
    encoder,
):
    categorical_names = [
        f"cat__{name}"
        for name in (
            encoder
            .get_feature_names_out(
                CATEGORICAL_COLUMNS
            )
        )
    ]

    return (
        [
            f"num__{name}"
            for name in NUMERIC_COLUMNS
        ]
        +
        [
            f"bool__{name}"
            for name in BOOLEAN_COLUMNS
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

    numeric_scaled = (
        np.nan_to_num(
            numeric_scaled,
            nan=0.0,
            posinf=np.inf,
            neginf=-np.inf,
        )
        .astype(
            np.float32
        )
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
```

# 10. M4.7.1 — Fit preprocessing state lại từ TRAIN-only source

M4.7 refit preprocessing state từ raw artifact bằng đúng M4.6 procedure để kiểm tra end-to-end reproducibility.

Pass này:

- không đọc target;
- không fit từ VALIDATION;
- không dùng FINAL TEST;
- chỉ học scaler/vocabulary từ TRAIN strategy tương ứng.


```python
STRATEGIES = [
    "W_LONG",
    "W_SHORT",
]

feature_scalers = {
    strategy: {
        column: StandardScaler()
        for column in NUMERIC_COLUMNS
    }
    for strategy in STRATEGIES
}

category_counters = {
    strategy: {
        column:
            Counter()
        for column
        in CATEGORICAL_COLUMNS
    }
    for strategy
    in STRATEGIES
}

fit_row_counts = Counter()

fit_min_timestamp = {
    strategy: None
    for strategy
    in STRATEGIES
}

fit_max_timestamp = {
    strategy: None
    for strategy
    in STRATEGIES
}

fit_card_count = 0

fit_raw_rows_seen = 0

fit_context_rows_seen = 0

fit_seen_card_keys = set()

fit_start = (
    time.perf_counter()
)


for card_key, card_df in (
    iter_card_blocks(
        DATA_PATH,
        CHUNK_SIZE,
        FIT_USECOLS,
    )
):
    fit_card_count += 1

    fit_raw_rows_seen += (
        len(
            card_df
        )
    )

    if (
        card_key
        in fit_seen_card_keys
    ):
        raise RuntimeError(
            "Card block reappearance "
            f"during fit pass: "
            f"{card_key}"
        )

    fit_seen_card_keys.add(
        card_key
    )

    context_card_df = (
        card_df.loc[
            card_df[
                "Timestamp"
            ]
            < VALIDATION_END
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    if (
        context_card_df
        .empty
    ):
        continue

    fit_context_rows_seen += (
        len(
            context_card_df
        )
    )

    behavioral_df = (
        compute_causal_behavioral_features(
            context_card_df
        )
    )

    core_df = (
        build_core_feature_frame(
            context_card_df,
            behavioral_df,
        )
    )

    timestamp = (
        core_df[
            "Timestamp"
        ]
    )

    masks = {
        "W_LONG": (
            (
                timestamp
                >= W_LONG_START
            )
            &
            (
                timestamp
                < TRAIN_END
            )
        ),

        "W_SHORT": (
            (
                timestamp
                >= W_SHORT_START
            )
            &
            (
                timestamp
                < TRAIN_END
            )
        ),
    }

    for strategy in (
        STRATEGIES
    ):
        mask = masks[
            strategy
        ]

        if not (
            mask.any()
        ):
            continue

        train_sub = (
            core_df.loc[
                mask
            ]
        )

        for column in NUMERIC_COLUMNS:
            values = (
                train_sub[
                    column
                ]
                .astype("float64")
                .to_numpy()
            )

            valid_values = (
                values[
                    np.isfinite(values)
                ]
            )

            if len(valid_values) > 0:
                feature_scalers[
                    strategy
                ][
                    column
                ].partial_fit(
                    valid_values.reshape(
                        -1,
                        1,
                    )
                )

        fit_row_counts[
            strategy
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
            fit_min_timestamp[
                strategy
            ]
            is None
            or
            current_min
            <
            fit_min_timestamp[
                strategy
            ]
        ):
            fit_min_timestamp[
                strategy
            ] = current_min

        if (
            fit_max_timestamp[
                strategy
            ]
            is None
            or
            current_max
            >
            fit_max_timestamp[
                strategy
            ]
        ):
            fit_max_timestamp[
                strategy
            ] = current_max

        for column in (
            CATEGORICAL_COLUMNS
        ):
            category_counters[
                strategy
            ][column].update(
                train_sub[
                    column
                ]
                .astype(
                    "string"
                )
                .value_counts()
                .to_dict()
            )

    if (
        fit_card_count
        % 500
        == 0
    ):
        print(
            "Fit-pass cards:",
            f"{fit_card_count:,}",
            "| raw rows:",
            f"{fit_raw_rows_seen:,}",
        )

# ============================================================
# Ghép các feature-wise scaler thành một StandardScaler
# 4-feature để giữ nguyên preprocessing contract M4.6.
# ============================================================

scalers = {}

for strategy in STRATEGIES:
    combined_scaler = StandardScaler()

    means = []
    variances = []
    scales = []
    sample_counts = []

    for column in NUMERIC_COLUMNS:
        feature_scaler = (
            feature_scalers[
                strategy
            ][
                column
            ]
        )

        if not hasattr(
            feature_scaler,
            "mean_",
        ):
            raise RuntimeError(
                f"{strategy} / {column}: "
                "không có observed TRAIN value."
            )

        means.append(
            float(
                feature_scaler.mean_[0]
            )
        )

        variances.append(
            float(
                feature_scaler.var_[0]
            )
        )

        scales.append(
            float(
                feature_scaler.scale_[0]
            )
        )

        sample_counts.append(
            int(
                np.asarray(
                    feature_scaler.n_samples_seen_
                ).reshape(-1)[0]
            )
        )

    combined_scaler.mean_ = np.array(
        means,
        dtype=np.float64,
    )

    combined_scaler.var_ = np.array(
        variances,
        dtype=np.float64,
    )

    combined_scaler.scale_ = np.array(
        scales,
        dtype=np.float64,
    )

    combined_scaler.n_samples_seen_ = np.array(
        sample_counts,
        dtype=np.int64,
    )

    combined_scaler.n_features_in_ = len(
        NUMERIC_COLUMNS
    )

    scalers[
        strategy
    ] = combined_scaler

fit_elapsed = (
    time.perf_counter()
    - fit_start
)


print(
    "\nPreprocessing fit pass hoàn tất."
)

print(
    "Card blocks:",
    f"{fit_card_count:,}",
)

print(
    "Raw rows seen:",
    f"{fit_raw_rows_seen:,}",
)

print(
    "Context rows:",
    f"{fit_context_rows_seen:,}",
)

print(
    "W_LONG fit rows:",
    f"{fit_row_counts['W_LONG']:,}",
)

print(
    "W_SHORT fit rows:",
    f"{fit_row_counts['W_SHORT']:,}",
)

print(
    "Elapsed seconds:",
    round(
        fit_elapsed,
        2,
    ),
)
```

    Fit-pass cards: 500 | raw rows: 2,021,584
    Fit-pass cards: 1,000 | raw rows: 4,066,794
    Fit-pass cards: 1,500 | raw rows: 5,915,634
    Fit-pass cards: 2,000 | raw rows: 7,883,446
    Fit-pass cards: 2,500 | raw rows: 9,842,479
    Fit-pass cards: 3,000 | raw rows: 11,897,259
    Fit-pass cards: 3,500 | raw rows: 13,997,374
    Fit-pass cards: 4,000 | raw rows: 16,040,304
    Fit-pass cards: 4,500 | raw rows: 18,115,490
    Fit-pass cards: 5,000 | raw rows: 19,945,815
    Fit-pass cards: 5,500 | raw rows: 21,956,418
    Fit-pass cards: 6,000 | raw rows: 23,844,247
    
    Preprocessing fit pass hoàn tất.
    Card blocks: 6,139
    Raw rows seen: 24,386,900
    Context rows: 23,038,920
    W_LONG fit rows: 6,855,270
    W_SHORT fit rows: 1,721,615
    Elapsed seconds: 137.27


# 11. M4.7.2 — Fit-source integrity gate


```python
assert (
    fit_card_count
    == EXPECTED_CARD_COUNT
)

assert (
    len(
        fit_seen_card_keys
    )
    == EXPECTED_CARD_COUNT
)

assert (
    fit_raw_rows_seen
    == 24_386_900
)

assert (
    fit_context_rows_seen
    ==
    EXPECTED_CONTEXT_ROWS_BEFORE_VALIDATION_END
)

assert (
    fit_row_counts[
        "W_LONG"
    ]
    == EXPECTED_W_LONG_ROWS
)

assert (
    fit_row_counts[
        "W_SHORT"
    ]
    == EXPECTED_W_SHORT_ROWS
)

assert (
    fit_min_timestamp[
        "W_LONG"
    ]
    >= W_LONG_START
)

assert (
    fit_max_timestamp[
        "W_LONG"
    ]
    < TRAIN_END
)

assert (
    fit_min_timestamp[
        "W_SHORT"
    ]
    >= W_SHORT_START
)

assert (
    fit_max_timestamp[
        "W_SHORT"
    ]
    < TRAIN_END
)

assert (
    "Is Fraud?"
    not in FIT_USECOLS
)


print(
    "M4.7 TRAIN-ONLY PREPROCESSING FIT GATE: PASS"
)

print(
    "W_LONG fit range:",
    fit_min_timestamp[
        "W_LONG"
    ],
    "→",
    fit_max_timestamp[
        "W_LONG"
    ],
)

print(
    "W_SHORT fit range:",
    fit_min_timestamp[
        "W_SHORT"
    ],
    "→",
    fit_max_timestamp[
        "W_SHORT"
    ],
)
```

    M4.7 TRAIN-ONLY PREPROCESSING FIT GATE: PASS
    W_LONG fit range: 2015-01-01 00:01:00 → 2018-12-31 23:58:00
    W_SHORT fit range: 2018-01-01 00:03:00 → 2018-12-31 23:58:00


# 12. M4.7.3 — Freeze preprocessing bundles


```python
bundles = {}

for strategy in (
    STRATEGIES
):
    vocab = {
        column:
            set(
                category_counters[
                    strategy
                ][
                    column
                ].keys()
            )
        for column
        in CATEGORICAL_COLUMNS
    }

    encoder = (
        build_encoder_from_train_vocab(
            vocab
        )
    )

    feature_names = (
        build_feature_names(
            encoder
        )
    )

    bundles[
        strategy
    ] = PreprocessingBundle(
        strategy=strategy,

        scaler=
            scalers[
                strategy
            ],

        category_vocab=
            vocab,

        encoder=
            encoder,

        feature_names=
            feature_names,

        fit_row_count=
            fit_row_counts[
                strategy
            ],

        fit_min_timestamp=
            fit_min_timestamp[
                strategy
            ],

        fit_max_timestamp=
            fit_max_timestamp[
                strategy
            ],
    )

# ============================================================
# Numeric scaler integrity gate
# ============================================================

for strategy in STRATEGIES:
    scaler = scalers[
        strategy
    ]

    print(
        "\nNumeric scaler:",
        strategy,
    )

    for index, column in enumerate(
        NUMERIC_COLUMNS
    ):
        print(
            column,
            "| mean =",
            scaler.mean_[index],
            "| var =",
            scaler.var_[index],
            "| scale =",
            scaler.scale_[index],
            "| observed rows =",
            scaler.n_samples_seen_[index],
        )

    assert np.isfinite(
        scaler.mean_
    ).all()

    assert np.isfinite(
        scaler.var_
    ).all()

    assert np.isfinite(
        scaler.scale_
    ).all()

    assert (
        scaler.scale_ > 0
    ).all()


print(
    "\nM4.7 NUMERIC SCALER STATE GATE: PASS"
)

PREPROCESSING_BUNDLES_FROZEN = True


for strategy in (
    STRATEGIES
):
    bundle = bundles[
        strategy
    ]

    print(
        "\n",
        strategy,
    )

    print(
        "Output width:",
        len(
            bundle
            .feature_names
        ),
    )

    for column in (
        CATEGORICAL_COLUMNS
    ):
        print(
            column,
            "TRAIN vocab size:",
            len(
                bundle
                .category_vocab[
                    column
                ]
            ),
        )


assert (
    len(
        bundles[
            "W_LONG"
        ].feature_names
    )
    ==
    EXPECTED_OUTPUT_WIDTH
)

assert (
    bundles[
        "W_LONG"
    ].feature_names
    ==
    bundles[
        "W_SHORT"
    ].feature_names
)

print(
    "\nM4.7 PREPROCESSING BUNDLE FREEZE GATE: PASS"
)
```

    
    Numeric scaler: W_LONG
    amount_numeric | mean = 42.94575655079967 | var = 6557.464502523365 | scale = 80.97817300065101 | observed rows = 6855270
    time_since_previous_transaction_min | mean = 1197.180492606929 | var = 4634911.122458403 | scale = 2152.8843727563267 | observed rows = 6854796
    transactions_last_1h | mean = 0.2794063545272463 | var = 0.4424090359825917 | scale = 0.6651383585259474 | observed rows = 6855270
    amount_minus_previous_mean | mean = -0.4736466100841145 | var = 6177.3873871467 | scale = 78.59635734018912 | observed rows = 6854796
    
    Numeric scaler: W_SHORT
    amount_numeric | mean = 42.879027959212905 | var = 6488.408201183131 | scale = 80.55065611888665 | observed rows = 1721615
    time_since_previous_transaction_min | mean = 1197.3147570599144 | var = 4466135.639103394 | scale = 2113.3233635919028 | observed rows = 1721515
    transactions_last_1h | mean = 0.2759606532238619 | var = 0.43316051821960744 | scale = 0.6581493130130939 | observed rows = 1721615
    amount_minus_previous_mean | mean = -0.4259839479714494 | var = 6111.0552143661125 | scale = 78.17323847945735 | observed rows = 1721515
    
    M4.7 NUMERIC SCALER STATE GATE: PASS
    
     W_LONG
    Output width: 47
    transaction_mode TRAIN vocab size: 3
    location_state TRAIN vocab size: 3
    hour_of_day TRAIN vocab size: 24
    day_of_week TRAIN vocab size: 7
    
     W_SHORT
    Output width: 47
    transaction_mode TRAIN vocab size: 3
    location_state TRAIN vocab size: 3
    hour_of_day TRAIN vocab size: 24
    day_of_week TRAIN vocab size: 7
    
    M4.7 PREPROCESSING BUNDLE FREEZE GATE: PASS


# 13. M4.7.4 — Full baseline-ready matrix construction

## Quan trọng

Target chỉ được đọc ở pass này, sau khi preprocessing bundles đã freeze.

Target không được dùng để:

- fit scaler;
- tạo category vocabulary;
- quyết định feature;
- thay đổi feature order.

Mỗi development row được transform bằng state tương ứng:

W_LONG TRAIN
→ W_LONG bundle

W_SHORT TRAIN
→ W_SHORT bundle

VALIDATION
→ transform hai lần:
- W_LONG bundle;
- W_SHORT bundle.

FINAL TEST rows bị loại khỏi matrix construction.


```python
assert (
    PREPROCESSING_BUNDLES_FROZEN
    is True
)


matrix_parts = {
    "W_LONG": {
        "train": [],
        "validation": [],
    },

    "W_SHORT": {
        "train": [],
        "validation": [],
    },
}


target_parts = {
    "W_LONG": [],
    "W_SHORT": [],
    "VALIDATION": [],
}


row_id_parts = {
    "W_LONG": [],
    "W_SHORT": [],
    "VALIDATION": [],
}


timestamp_range_observed = {
    "W_LONG": {
        "min": None,
        "max": None,
    },

    "W_SHORT": {
        "min": None,
        "max": None,
    },

    "VALIDATION": {
        "min": None,
        "max": None,
    },
}


validation_repeat_sample_parts = []


build_card_count = 0

build_raw_rows_seen = 0

build_context_rows_seen = 0

build_seen_card_keys = set()


build_start = (
    time.perf_counter()
)


def update_range(
    state,
    timestamp_series,
):
    if (
        timestamp_series
        .empty
    ):
        return

    current_min = (
        timestamp_series
        .min()
    )

    current_max = (
        timestamp_series
        .max()
    )

    if (
        state["min"]
        is None
        or
        current_min
        < state["min"]
    ):
        state[
            "min"
        ] = current_min

    if (
        state["max"]
        is None
        or
        current_max
        > state["max"]
    ):
        state[
            "max"
        ] = current_max


for card_key, card_df in (
    iter_card_blocks(
        DATA_PATH,
        CHUNK_SIZE,
        MATRIX_USECOLS,
    )
):
    build_card_count += 1

    build_raw_rows_seen += (
        len(
            card_df
        )
    )

    if (
        card_key
        in build_seen_card_keys
    ):
        raise RuntimeError(
            "Card block reappearance "
            "during matrix pass: "
            f"{card_key}"
        )

    build_seen_card_keys.add(
        card_key
    )

    # Drop protected/future rows before
    # target mapping or matrix creation.
    context_card_df = (
        card_df.loc[
            card_df[
                "Timestamp"
            ]
            < VALIDATION_END
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    if (
        context_card_df
        .empty
    ):
        continue

    build_context_rows_seen += (
        len(
            context_card_df
        )
    )

    behavioral_df = (
        compute_causal_behavioral_features(
            context_card_df
        )
    )

    core_df = (
        build_core_feature_frame(
            context_card_df,
            behavioral_df,
        )
    )

    timestamp = (
        core_df[
            "Timestamp"
        ]
    )

    w_long_mask = (
        (
            timestamp
            >= W_LONG_START
        )
        &
        (
            timestamp
            < TRAIN_END
        )
    )

    w_short_mask = (
        (
            timestamp
            >= W_SHORT_START
        )
        &
        (
            timestamp
            < TRAIN_END
        )
    )

    validation_mask = (
        (
            timestamp
            >= TRAIN_END
        )
        &
        (
            timestamp
            < VALIDATION_END
        )
    )


    # ------------------------
    # W_LONG TRAIN
    # ------------------------

    if (
        w_long_mask.any()
    ):
        sub = (
            core_df.loc[
                w_long_mask
            ]
        )

        X_part = (
            transform_with_bundle(
                sub,
                bundles[
                    "W_LONG"
                ],
            )
        )

        y_part = (
            map_development_target(
                context_card_df.loc[
                    w_long_mask,
                    "Is Fraud?",
                ]
            )
        )

        matrix_parts[
            "W_LONG"
        ][
            "train"
        ].append(
            X_part
        )

        target_parts[
            "W_LONG"
        ].append(
            y_part
        )

        row_id_parts[
            "W_LONG"
        ].append(
            sub[
                "raw_row_id"
            ]
            .to_numpy(
                dtype=np.int64
            )
        )

        update_range(
            timestamp_range_observed[
                "W_LONG"
            ],
            sub[
                "Timestamp"
            ],
        )


    # ------------------------
    # W_SHORT TRAIN
    # ------------------------

    if (
        w_short_mask.any()
    ):
        sub = (
            core_df.loc[
                w_short_mask
            ]
        )

        X_part = (
            transform_with_bundle(
                sub,
                bundles[
                    "W_SHORT"
                ],
            )
        )

        y_part = (
            map_development_target(
                context_card_df.loc[
                    w_short_mask,
                    "Is Fraud?",
                ]
            )
        )

        matrix_parts[
            "W_SHORT"
        ][
            "train"
        ].append(
            X_part
        )

        target_parts[
            "W_SHORT"
        ].append(
            y_part
        )

        row_id_parts[
            "W_SHORT"
        ].append(
            sub[
                "raw_row_id"
            ]
            .to_numpy(
                dtype=np.int64
            )
        )

        update_range(
            timestamp_range_observed[
                "W_SHORT"
            ],
            sub[
                "Timestamp"
            ],
        )


    # ------------------------
    # VALIDATION
    # ------------------------

    if (
        validation_mask.any()
    ):
        sub = (
            core_df.loc[
                validation_mask
            ]
        )

        X_val_long_part = (
            transform_with_bundle(
                sub,
                bundles[
                    "W_LONG"
                ],
            )
        )

        X_val_short_part = (
            transform_with_bundle(
                sub,
                bundles[
                    "W_SHORT"
                ],
            )
        )

        y_val_part = (
            map_development_target(
                context_card_df.loc[
                    validation_mask,
                    "Is Fraud?",
                ]
            )
        )

        matrix_parts[
            "W_LONG"
        ][
            "validation"
        ].append(
            X_val_long_part
        )

        matrix_parts[
            "W_SHORT"
        ][
            "validation"
        ].append(
            X_val_short_part
        )

        target_parts[
            "VALIDATION"
        ].append(
            y_val_part
        )

        row_id_parts[
            "VALIDATION"
        ].append(
            sub[
                "raw_row_id"
            ]
            .to_numpy(
                dtype=np.int64
            )
        )

        update_range(
            timestamp_range_observed[
                "VALIDATION"
            ],
            sub[
                "Timestamp"
            ],
        )

        repeat_mask = (
            sub[
                "raw_row_id"
            ]
            % 10_000
            == 0
        )

        if (
            repeat_mask.any()
        ):
            validation_repeat_sample_parts.append(
                sub.loc[
                    repeat_mask
                ].copy()
            )


    if (
        build_card_count
        % 500
        == 0
    ):
        print(
            "Matrix-pass cards:",
            f"{build_card_count:,}",
            "| raw rows:",
            f"{build_raw_rows_seen:,}",
        )


build_elapsed = (
    time.perf_counter()
    - build_start
)


print(
    "\nMatrix part construction hoàn tất."
)

print(
    "Card blocks:",
    f"{build_card_count:,}",
)

print(
    "Raw rows seen:",
    f"{build_raw_rows_seen:,}",
)

print(
    "Context rows seen:",
    f"{build_context_rows_seen:,}",
)

print(
    "Elapsed seconds:",
    round(
        build_elapsed,
        2,
    ),
)
```

    Matrix-pass cards: 500 | raw rows: 2,021,584
    Matrix-pass cards: 1,000 | raw rows: 4,066,794
    Matrix-pass cards: 1,500 | raw rows: 5,915,634
    Matrix-pass cards: 2,000 | raw rows: 7,883,446
    Matrix-pass cards: 2,500 | raw rows: 9,842,479
    Matrix-pass cards: 3,000 | raw rows: 11,897,259
    Matrix-pass cards: 3,500 | raw rows: 13,997,374
    Matrix-pass cards: 4,000 | raw rows: 16,040,304
    Matrix-pass cards: 4,500 | raw rows: 18,115,490
    Matrix-pass cards: 5,000 | raw rows: 19,945,815
    Matrix-pass cards: 5,500 | raw rows: 21,956,418
    Matrix-pass cards: 6,000 | raw rows: 23,844,247
    
    Matrix part construction hoàn tất.
    Card blocks: 6,139
    Raw rows seen: 24,386,900
    Context rows seen: 23,038,920
    Elapsed seconds: 174.01


# 14. M4.7.5 — Assemble full X/y/lineage


```python
X_train_w_long = sparse.vstack(
    matrix_parts[
        "W_LONG"
    ][
        "train"
    ],
    format="csr",
    dtype=np.float32,
)

X_train_w_short = sparse.vstack(
    matrix_parts[
        "W_SHORT"
    ][
        "train"
    ],
    format="csr",
    dtype=np.float32,
)

X_validation_w_long = sparse.vstack(
    matrix_parts[
        "W_LONG"
    ][
        "validation"
    ],
    format="csr",
    dtype=np.float32,
)

X_validation_w_short = sparse.vstack(
    matrix_parts[
        "W_SHORT"
    ][
        "validation"
    ],
    format="csr",
    dtype=np.float32,
)


y_train_w_long = np.concatenate(
    target_parts[
        "W_LONG"
    ]
).astype(
    np.int8,
    copy=False,
)

y_train_w_short = np.concatenate(
    target_parts[
        "W_SHORT"
    ]
).astype(
    np.int8,
    copy=False,
)

y_validation = np.concatenate(
    target_parts[
        "VALIDATION"
    ]
).astype(
    np.int8,
    copy=False,
)


row_id_train_w_long = np.concatenate(
    row_id_parts[
        "W_LONG"
    ]
).astype(
    np.int64,
    copy=False,
)

row_id_train_w_short = np.concatenate(
    row_id_parts[
        "W_SHORT"
    ]
).astype(
    np.int64,
    copy=False,
)

row_id_validation = np.concatenate(
    row_id_parts[
        "VALIDATION"
    ]
).astype(
    np.int64,
    copy=False,
)


feature_names = (
    bundles[
        "W_LONG"
    ]
    .feature_names
)


print(
    "X_train_w_long:",
    X_train_w_long.shape,
    X_train_w_long.dtype,
)

print(
    "X_train_w_short:",
    X_train_w_short.shape,
    X_train_w_short.dtype,
)

print(
    "X_validation_w_long:",
    X_validation_w_long.shape,
    X_validation_w_long.dtype,
)

print(
    "X_validation_w_short:",
    X_validation_w_short.shape,
    X_validation_w_short.dtype,
)

print(
    "y_train_w_long:",
    y_train_w_long.shape,
    y_train_w_long.dtype,
)

print(
    "y_train_w_short:",
    y_train_w_short.shape,
    y_train_w_short.dtype,
)

print(
    "y_validation:",
    y_validation.shape,
    y_validation.dtype,
)

print(
    "Feature width:",
    len(
        feature_names
    ),
)
```

    X_train_w_long: (6855270, 47) float32
    X_train_w_short: (1721615, 47) float32
    X_validation_w_long: (712458, 47) float32
    X_validation_w_short: (712458, 47) float32
    y_train_w_long: (6855270,) int8
    y_train_w_short: (1721615,) int8
    y_validation: (712458,) int8
    Feature width: 47


# 15. M4.7.6 — Matrix row / boundary / target gate


```python
assert (
    build_card_count
    == EXPECTED_CARD_COUNT
)

assert (
    len(
        build_seen_card_keys
    )
    == EXPECTED_CARD_COUNT
)

assert (
    build_raw_rows_seen
    == 24_386_900
)

assert (
    build_context_rows_seen
    ==
    EXPECTED_CONTEXT_ROWS_BEFORE_VALIDATION_END
)


assert (
    X_train_w_long
    .shape
    ==
    (
        EXPECTED_W_LONG_ROWS,
        EXPECTED_OUTPUT_WIDTH,
    )
)

assert (
    X_train_w_short
    .shape
    ==
    (
        EXPECTED_W_SHORT_ROWS,
        EXPECTED_OUTPUT_WIDTH,
    )
)

assert (
    X_validation_w_long
    .shape
    ==
    (
        EXPECTED_VALIDATION_ROWS,
        EXPECTED_OUTPUT_WIDTH,
    )
)

assert (
    X_validation_w_short
    .shape
    ==
    (
        EXPECTED_VALIDATION_ROWS,
        EXPECTED_OUTPUT_WIDTH,
    )
)


assert (
    len(
        y_train_w_long
    )
    == EXPECTED_W_LONG_ROWS
)

assert (
    len(
        y_train_w_short
    )
    == EXPECTED_W_SHORT_ROWS
)

assert (
    len(
        y_validation
    )
    == EXPECTED_VALIDATION_ROWS
)


assert (
    int(
        y_train_w_long
        .sum()
    )
    == EXPECTED_W_LONG_FRAUD
)

assert (
    int(
        y_train_w_short
        .sum()
    )
    == EXPECTED_W_SHORT_FRAUD
)

assert (
    int(
        y_validation
        .sum()
    )
    == EXPECTED_VALIDATION_FRAUD
)


assert (
    set(
        np.unique(
            y_train_w_long
        )
    )
    <= {0, 1}
)

assert (
    set(
        np.unique(
            y_train_w_short
        )
    )
    <= {0, 1}
)

assert (
    set(
        np.unique(
            y_validation
        )
    )
    <= {0, 1}
)


assert (
    timestamp_range_observed[
        "W_LONG"
    ][
        "min"
    ]
    >= W_LONG_START
)

assert (
    timestamp_range_observed[
        "W_LONG"
    ][
        "max"
    ]
    < TRAIN_END
)

assert (
    timestamp_range_observed[
        "W_SHORT"
    ][
        "min"
    ]
    >= W_SHORT_START
)

assert (
    timestamp_range_observed[
        "W_SHORT"
    ][
        "max"
    ]
    < TRAIN_END
)

assert (
    timestamp_range_observed[
        "VALIDATION"
    ][
        "min"
    ]
    >= TRAIN_END
)

assert (
    timestamp_range_observed[
        "VALIDATION"
    ][
        "max"
    ]
    < VALIDATION_END
)


print(
    "M4.7 ROW / BOUNDARY / TARGET GATE: PASS"
)

print(
    "W_LONG fraud:",
    int(
        y_train_w_long
        .sum()
    ),
)

print(
    "W_SHORT fraud:",
    int(
        y_train_w_short
        .sum()
    ),
)

print(
    "VALIDATION fraud:",
    int(
        y_validation
        .sum()
    ),
)
```

    M4.7 ROW / BOUNDARY / TARGET GATE: PASS
    W_LONG fraud: 9606
    W_SHORT fraud: 2491
    VALIDATION fraud: 1052


# 16. M4.7.7 — X/y/lineage alignment gate


```python
assert (
    X_train_w_long
    .shape[0]
    ==
    len(
        y_train_w_long
    )
    ==
    len(
        row_id_train_w_long
    )
)

assert (
    X_train_w_short
    .shape[0]
    ==
    len(
        y_train_w_short
    )
    ==
    len(
        row_id_train_w_short
    )
)

assert (
    X_validation_w_long
    .shape[0]
    ==
    X_validation_w_short
    .shape[0]
    ==
    len(
        y_validation
    )
    ==
    len(
        row_id_validation
    )
)


assert (
    len(
        np.unique(
            row_id_train_w_long
        )
    )
    ==
    len(
        row_id_train_w_long
    )
)

assert (
    len(
        np.unique(
            row_id_train_w_short
        )
    )
    ==
    len(
        row_id_train_w_short
    )
)

assert (
    len(
        np.unique(
            row_id_validation
        )
    )
    ==
    len(
        row_id_validation
    )
)


# W_SHORT phải là subset row-level của W_LONG.
positions = np.searchsorted(
    row_id_train_w_long,
    row_id_train_w_short,
)

assert (
    positions
    < len(
        row_id_train_w_long
    )
).all()

assert np.array_equal(
    row_id_train_w_long[
        positions
    ],
    row_id_train_w_short,
)


print(
    "M4.7 X/Y/LINEAGE ALIGNMENT GATE: PASS"
)
```

    M4.7 X/Y/LINEAGE ALIGNMENT GATE: PASS


# 17. M4.7.8 — Schema, dtype, NaN/inf và sparsity audit


```python
def csr_memory_bytes(
    matrix,
):
    return (
        matrix.data.nbytes
        +
        matrix.indices.nbytes
        +
        matrix.indptr.nbytes
    )


matrix_audit_rows = []


for name, matrix in [
    (
        "X_train_w_long",
        X_train_w_long,
    ),
    (
        "X_train_w_short",
        X_train_w_short,
    ),
    (
        "X_validation_w_long",
        X_validation_w_long,
    ),
    (
        "X_validation_w_short",
        X_validation_w_short,
    ),
]:
    assert (
        sparse.isspmatrix_csr(
            matrix
        )
    )

    assert (
        matrix.dtype
        == np.float32
    )

    assert np.isfinite(
        matrix.data
    ).all()

    assert (
        matrix.shape[1]
        ==
        EXPECTED_OUTPUT_WIDTH
    )

    matrix_audit_rows.append(
        {
            "matrix":
                name,

            "rows":
                matrix.shape[0],

            "columns":
                matrix.shape[1],

            "nnz":
                matrix.nnz,

            "density_pct":
                (
                    matrix.nnz
                    /
                    (
                        matrix.shape[0]
                        *
                        matrix.shape[1]
                    )
                    * 100
                ),

            "csr_memory_mb":
                (
                    csr_memory_bytes(
                        matrix
                    )
                    / 1024**2
                ),
        }
    )


matrix_audit_df = pd.DataFrame(
    matrix_audit_rows
)


print(
    matrix_audit_df.to_string(
        index=False
    )
)


assert (
    len(
        feature_names
    )
    ==
    EXPECTED_OUTPUT_WIDTH
)

assert (
    len(
        set(
            feature_names
        )
    )
    ==
    EXPECTED_OUTPUT_WIDTH
)

assert (
    feature_names
    ==
    bundles[
        "W_SHORT"
    ]
    .feature_names
)


print(
    "M4.7 MATRIX SCHEMA / FINITE / SPARSITY GATE: PASS"
)
```

                  matrix    rows  columns      nnz  density_pct  csr_memory_mb
          X_train_w_long 6855270       47 61900193    19.211867     498.411777
         X_train_w_short 1721615       47 15542381    19.208067     125.146400
     X_validation_w_long  712458       47  6430339    19.203339      51.777409
    X_validation_w_short  712458       47  6430339    19.203339      51.777409
    M4.7 MATRIX SCHEMA / FINITE / SPARSITY GATE: PASS


# 18. M4.7.9 — Prohibited-feature and target-isolation audit


```python
for prohibited in [
    "User",
    "Card",
    "Merchant Name",
    "Errors?",
    "Is Fraud?",
    "raw_row_id",
    "Timestamp",
]:
    assert not any(
        prohibited
        in feature_name
        for feature_name
        in feature_names
    )


TARGET_USED_FOR_PREPROCESSING_FIT = False

TARGET_USED_FOR_FEATURE_ENGINEERING = False

FINAL_TEST_USED_FOR_MATRIX_DECISION = False

VALIDATION_USED_FOR_PREPROCESSING_FIT = False


assert (
    TARGET_USED_FOR_PREPROCESSING_FIT
    is False
)

assert (
    TARGET_USED_FOR_FEATURE_ENGINEERING
    is False
)

assert (
    FINAL_TEST_USED_FOR_MATRIX_DECISION
    is False
)

assert (
    VALIDATION_USED_FOR_PREPROCESSING_FIT
    is False
)


print(
    "M4.7 TARGET / IDENTIFIER ISOLATION GATE: PASS"
)
```

    M4.7 TARGET / IDENTIFIER ISOLATION GATE: PASS


# 19. M4.7.10 — Repeat-transform determinism


```python
validation_repeat_sample = (
    pd.concat(
        validation_repeat_sample_parts,
        ignore_index=True,
    )
    if validation_repeat_sample_parts
    else pd.DataFrame()
)


print(
    "Repeat-transform validation sample rows:",
    len(
        validation_repeat_sample
    ),
)


assert not (
    validation_repeat_sample
    .empty
)


for strategy in (
    STRATEGIES
):
    bundle = bundles[
        strategy
    ]

    X_repeat_a = (
        transform_with_bundle(
            validation_repeat_sample,
            bundle,
        )
    )

    X_repeat_b = (
        transform_with_bundle(
            validation_repeat_sample,
            bundle,
        )
    )

    difference = (
        X_repeat_a
        != X_repeat_b
    )

    assert (
        difference.nnz
        == 0
    )

    assert (
        X_repeat_a.shape[1]
        ==
        EXPECTED_OUTPUT_WIDTH
    )

    print(
        strategy,
        "repeat-transform identical:",
        True,
    )


print(
    "M4.7 REPEAT-TRANSFORM DETERMINISM GATE: PASS"
)
```

    Repeat-transform validation sample rows: 88
    W_LONG repeat-transform identical: True
    W_SHORT repeat-transform identical: True
    M4.7 REPEAT-TRANSFORM DETERMINISM GATE: PASS


# 20. M4.7.11 — Feature-name contract


```python
print(
    "Feature count:",
    len(
        feature_names
    ),
)

for index, name in enumerate(
    feature_names
):
    print(
        index,
        name,
    )


assert (
    feature_names[
        :4
    ]
    ==
    [
        "num__amount_numeric",
        "num__time_since_previous_transaction_min",
        "num__transactions_last_1h",
        "num__amount_minus_previous_mean",
    ]
)

assert (
    feature_names[
        4:6
    ]
    ==
    [
        "bool__is_new_merchant",
        "bool__has_prior_card_history",
    ]
)


print(
    "M4.7 FEATURE-NAME CONTRACT GATE: PASS"
)
```

    Feature count: 47
    0 num__amount_numeric
    1 num__time_since_previous_transaction_min
    2 num__transactions_last_1h
    3 num__amount_minus_previous_mean
    4 bool__is_new_merchant
    5 bool__has_prior_card_history
    6 cat__transaction_mode_Chip Transaction
    7 cat__transaction_mode_Online Transaction
    8 cat__transaction_mode_Swipe Transaction
    9 cat__transaction_mode___UNKNOWN__
    10 cat__location_state_NON_PHYSICAL_OR_ONLINE
    11 cat__location_state_PHYSICAL_COMPLETE
    12 cat__location_state_PHYSICAL_ZIP_UNAVAILABLE
    13 cat__location_state___UNKNOWN__
    14 cat__hour_of_day_0
    15 cat__hour_of_day_1
    16 cat__hour_of_day_2
    17 cat__hour_of_day_3
    18 cat__hour_of_day_4
    19 cat__hour_of_day_5
    20 cat__hour_of_day_6
    21 cat__hour_of_day_7
    22 cat__hour_of_day_8
    23 cat__hour_of_day_9
    24 cat__hour_of_day_10
    25 cat__hour_of_day_11
    26 cat__hour_of_day_12
    27 cat__hour_of_day_13
    28 cat__hour_of_day_14
    29 cat__hour_of_day_15
    30 cat__hour_of_day_16
    31 cat__hour_of_day_17
    32 cat__hour_of_day_18
    33 cat__hour_of_day_19
    34 cat__hour_of_day_20
    35 cat__hour_of_day_21
    36 cat__hour_of_day_22
    37 cat__hour_of_day_23
    38 cat__hour_of_day___UNKNOWN__
    39 cat__day_of_week_0
    40 cat__day_of_week_1
    41 cat__day_of_week_2
    42 cat__day_of_week_3
    43 cat__day_of_week_4
    44 cat__day_of_week_5
    45 cat__day_of_week_6
    46 cat__day_of_week___UNKNOWN__
    M4.7 FEATURE-NAME CONTRACT GATE: PASS


# 21. M4.7.12 — Save baseline-ready artifacts


```python
artifact_paths = {
    "X_train_w_long":
        OUTPUT_DIR
        / "X_train_w_long.npz",

    "y_train_w_long":
        OUTPUT_DIR
        / "y_train_w_long.npy",

    "row_id_train_w_long":
        OUTPUT_DIR
        / "row_id_train_w_long.npy",

    "X_train_w_short":
        OUTPUT_DIR
        / "X_train_w_short.npz",

    "y_train_w_short":
        OUTPUT_DIR
        / "y_train_w_short.npy",

    "row_id_train_w_short":
        OUTPUT_DIR
        / "row_id_train_w_short.npy",

    "X_validation_w_long":
        OUTPUT_DIR
        / "X_validation_w_long.npz",

    "X_validation_w_short":
        OUTPUT_DIR
        / "X_validation_w_short.npz",

    "y_validation":
        OUTPUT_DIR
        / "y_validation.npy",

    "row_id_validation":
        OUTPUT_DIR
        / "row_id_validation.npy",

    "feature_names":
        OUTPUT_DIR
        / "feature_names.json",

    "manifest":
        OUTPUT_DIR
        / "manifest.json",
}


if SAVE_ARTIFACTS:
    sparse.save_npz(
        artifact_paths[
            "X_train_w_long"
        ],
        X_train_w_long,
        compressed=True,
    )

    np.save(
        artifact_paths[
            "y_train_w_long"
        ],
        y_train_w_long,
        allow_pickle=False,
    )

    np.save(
        artifact_paths[
            "row_id_train_w_long"
        ],
        row_id_train_w_long,
        allow_pickle=False,
    )

    sparse.save_npz(
        artifact_paths[
            "X_train_w_short"
        ],
        X_train_w_short,
        compressed=True,
    )

    np.save(
        artifact_paths[
            "y_train_w_short"
        ],
        y_train_w_short,
        allow_pickle=False,
    )

    np.save(
        artifact_paths[
            "row_id_train_w_short"
        ],
        row_id_train_w_short,
        allow_pickle=False,
    )

    sparse.save_npz(
        artifact_paths[
            "X_validation_w_long"
        ],
        X_validation_w_long,
        compressed=True,
    )

    sparse.save_npz(
        artifact_paths[
            "X_validation_w_short"
        ],
        X_validation_w_short,
        compressed=True,
    )

    np.save(
        artifact_paths[
            "y_validation"
        ],
        y_validation,
        allow_pickle=False,
    )

    np.save(
        artifact_paths[
            "row_id_validation"
        ],
        row_id_validation,
        allow_pickle=False,
    )

    with open(
        artifact_paths[
            "feature_names"
        ],
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            feature_names,
            file,
            ensure_ascii=False,
            indent=2,
        )


    manifest = {
        "pipeline_version":
            PIPELINE_VERSION,

        "output_width":
            EXPECTED_OUTPUT_WIDTH,

        "matrix_dtype":
            "float32",

        "matrix_format":
            "CSR",

        "target_dtype":
            "int8",

        "feature_names_file":
            artifact_paths[
                "feature_names"
            ].name,

        "W_LONG": {
            "train_rows":
                int(
                    X_train_w_long
                    .shape[0]
                ),

            "validation_rows":
                int(
                    X_validation_w_long
                    .shape[0]
                ),

            "train_fraud":
                int(
                    y_train_w_long
                    .sum()
                ),

            "fit_start":
                str(
                    fit_min_timestamp[
                        "W_LONG"
                    ]
                ),

            "fit_end_observed":
                str(
                    fit_max_timestamp[
                        "W_LONG"
                    ]
                ),
        },

        "W_SHORT": {
            "train_rows":
                int(
                    X_train_w_short
                    .shape[0]
                ),

            "validation_rows":
                int(
                    X_validation_w_short
                    .shape[0]
                ),

            "train_fraud":
                int(
                    y_train_w_short
                    .sum()
                ),

            "fit_start":
                str(
                    fit_min_timestamp[
                        "W_SHORT"
                    ]
                ),

            "fit_end_observed":
                str(
                    fit_max_timestamp[
                        "W_SHORT"
                    ]
                ),
        },

        "VALIDATION": {
            "rows":
                int(
                    len(
                        y_validation
                    )
                ),

            "fraud":
                int(
                    y_validation
                    .sum()
                ),

            "start":
                str(
                    timestamp_range_observed[
                        "VALIDATION"
                    ][
                        "min"
                    ]
                ),

            "end_observed":
                str(
                    timestamp_range_observed[
                        "VALIDATION"
                    ][
                        "max"
                    ]
                ),
        },

        "raw_identifiers_in_X":
            False,

        "target_in_X":
            False,

        "final_test_used":
            False,
    }


    with open(
        artifact_paths[
            "manifest"
        ],
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            ensure_ascii=False,
            indent=2,
        )


    print(
        "Saved artifacts:"
    )

    for name, path in (
        artifact_paths.items()
    ):
        assert (
            path.exists()
        )

        assert (
            path.stat().st_size
            > 0
        )

        print(
            name,
            "→",
            path,
            "|",
            f"{path.stat().st_size:,}",
            "bytes",
        )

else:
    print(
        "SAVE_ARTIFACTS=False — "
        "không ghi matrix ra disk."
    )
```

    Saved artifacts:
    X_train_w_long → /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/processed/m4_07_baseline_ready/X_train_w_long.npz | 105,780,141 bytes
    y_train_w_long → /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/processed/m4_07_baseline_ready/y_train_w_long.npy | 6,855,398 bytes
    row_id_train_w_long → /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/processed/m4_07_baseline_ready/row_id_train_w_long.npy | 54,842,288 bytes
    X_train_w_short → /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/processed/m4_07_baseline_ready/X_train_w_short.npz | 26,875,005 bytes
    y_train_w_short → /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/processed/m4_07_baseline_ready/y_train_w_short.npy | 1,721,743 bytes
    row_id_train_w_short → /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/processed/m4_07_baseline_ready/row_id_train_w_short.npy | 13,773,048 bytes
    X_validation_w_long → /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/processed/m4_07_baseline_ready/X_validation_w_long.npz | 11,281,966 bytes
    X_validation_w_short → /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/processed/m4_07_baseline_ready/X_validation_w_short.npz | 11,270,182 bytes
    y_validation → /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/processed/m4_07_baseline_ready/y_validation.npy | 712,586 bytes
    row_id_validation → /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/processed/m4_07_baseline_ready/row_id_validation.npy | 5,699,792 bytes
    feature_names → /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/processed/m4_07_baseline_ready/feature_names.json | 1,380 bytes
    manifest → /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/processed/m4_07_baseline_ready/manifest.json | 795 bytes


# 22. M4.7.13 — Saved-artifact manifest audit


```python
if SAVE_ARTIFACTS:
    with open(
        artifact_paths[
            "manifest"
        ],
        "r",
        encoding="utf-8",
    ) as file:
        saved_manifest = (
            json.load(
                file
            )
        )

    with open(
        artifact_paths[
            "feature_names"
        ],
        "r",
        encoding="utf-8",
    ) as file:
        saved_feature_names = (
            json.load(
                file
            )
        )

    assert (
        saved_manifest[
            "pipeline_version"
        ]
        == PIPELINE_VERSION
    )

    assert (
        saved_manifest[
            "output_width"
        ]
        == EXPECTED_OUTPUT_WIDTH
    )

    assert (
        saved_feature_names
        == feature_names
    )

    assert (
        saved_manifest[
            "W_LONG"
        ][
            "train_rows"
        ]
        == EXPECTED_W_LONG_ROWS
    )

    assert (
        saved_manifest[
            "W_SHORT"
        ][
            "train_rows"
        ]
        == EXPECTED_W_SHORT_ROWS
    )

    assert (
        saved_manifest[
            "VALIDATION"
        ][
            "rows"
        ]
        == EXPECTED_VALIDATION_ROWS
    )

    assert (
        saved_manifest[
            "raw_identifiers_in_X"
        ]
        is False
    )

    assert (
        saved_manifest[
            "target_in_X"
        ]
        is False
    )

    assert (
        saved_manifest[
            "final_test_used"
        ]
        is False
    )


print(
    "M4.7 SAVED-ARTIFACT MANIFEST GATE: PASS"
)
```

    M4.7 SAVED-ARTIFACT MANIFEST GATE: PASS


# 23. M4.7.14 — Overall modeling-matrix safety gate


```python
assert (
    PREPROCESSING_BUNDLES_FROZEN
    is True
)

assert (
    X_train_w_long
    .shape[1]
    ==
    X_train_w_short
    .shape[1]
    ==
    X_validation_w_long
    .shape[1]
    ==
    X_validation_w_short
    .shape[1]
    ==
    EXPECTED_OUTPUT_WIDTH
)

assert (
    bundles[
        "W_LONG"
    ].feature_names
    ==
    bundles[
        "W_SHORT"
    ].feature_names
    ==
    feature_names
)

assert (
    int(
        y_train_w_long
        .sum()
    )
    == EXPECTED_W_LONG_FRAUD
)

assert (
    int(
        y_train_w_short
        .sum()
    )
    == EXPECTED_W_SHORT_FRAUD
)

assert (
    int(
        y_validation
        .sum()
    )
    == EXPECTED_VALIDATION_FRAUD
)

assert np.isfinite(
    X_train_w_long.data
).all()

assert np.isfinite(
    X_train_w_short.data
).all()

assert np.isfinite(
    X_validation_w_long.data
).all()

assert np.isfinite(
    X_validation_w_short.data
).all()

assert (
    "Is Fraud?"
    not in FIT_USECOLS
)

assert (
    "Errors?"
    not in MATRIX_USECOLS
)


PREPROCESSING_FIT_SOURCE_TRAIN_ONLY = True

VALIDATION_TRANSFORM_ONLY = True

FINAL_TEST_USED = False

TARGET_IN_X = False

RAW_IDENTIFIER_IN_X = False

FUTURE_HISTORY_USED = False

SAME_TIMESTAMP_HISTORY_USED = False


assert (
    PREPROCESSING_FIT_SOURCE_TRAIN_ONLY
    is True
)

assert (
    VALIDATION_TRANSFORM_ONLY
    is True
)

assert (
    FINAL_TEST_USED
    is False
)

assert (
    TARGET_IN_X
    is False
)

assert (
    RAW_IDENTIFIER_IN_X
    is False
)

assert (
    FUTURE_HISTORY_USED
    is False
)

assert (
    SAME_TIMESTAMP_HISTORY_USED
    is False
)


print(
    "M4.7 MODELING MATRIX SAFETY GATE: PASS"
)
```

    M4.7 MODELING MATRIX SAFETY GATE: PASS


# 24. Baseline-ready Matrix Registry — runtime review

## MR01 — W_LONG training matrix

Observed:

- rows: `6,855,270`;
- columns: `47`;
- representation: `CSR sparse matrix`;
- dtype: `float32`;
- target dtype: `int8`;
- fraud rows: `9,606`;
- CSR in-memory footprint: khoảng `498.41 MB`;
- density: `19.211867%`.

Nhận xét:

W_LONG full training population được transform thành công bằng preprocessing state fit riêng trên W_LONG TRAIN. Shape của `X`, độ dài `y` và lineage array khớp chính xác.

Status:

`VERIFIED — BASELINE-READY`

---

## MR02 — W_SHORT training matrix

Observed:

- rows: `1,721,615`;
- columns: `47`;
- representation: `CSR sparse matrix`;
- dtype: `float32`;
- target dtype: `int8`;
- fraud rows: `2,491`;
- CSR in-memory footprint: khoảng `125.15 MB`;
- density: `19.208067%`.

Nhận xét:

W_SHORT full training population được tạo bằng preprocessing state riêng của W_SHORT nhưng cùng feature contract với W_LONG.

Status:

`VERIFIED — BASELINE-READY`

---

## MR03 — W_LONG validation transform

Observed:

- rows: `712,458`;
- columns: `47`;
- preprocessing state: W_LONG TRAIN;
- target fraud rows: `1,052`;
- CSR in-memory footprint: khoảng `51.78 MB`;
- density: `19.203339%`.

Nhận xét:

VALIDATION chỉ được transform bằng state đã học từ W_LONG TRAIN. Không refit scaler hoặc vocabulary trên VALIDATION.

Status:

`VERIFIED — TRANSFORM ONLY`

---

## MR04 — W_SHORT validation transform

Observed:

- rows: `712,458`;
- columns: `47`;
- preprocessing state: W_SHORT TRAIN;
- dùng cùng `y_validation`;
- target fraud rows: `1,052`;
- CSR in-memory footprint: khoảng `51.78 MB`;
- density: `19.203339%`.

Nhận xét:

VALIDATION được transform lần thứ hai bằng state học từ W_SHORT TRAIN. Hai validation matrices có cùng row population, cùng target và cùng schema, nhưng numerical values có thể khác do scaler state của hai training window khác nhau.

Status:

`VERIFIED — TRANSFORM ONLY`

---

## MR05 — Feature schema

Observed:

- `47` unique output features;
- `4` numerical features;
- `2` boolean features;
- `41` one-hot categorical columns;
- W_LONG và W_SHORT có cùng feature names/order;
- explicit `__UNKNOWN__` column tồn tại cho mọi categorical field;
- không có target trong X;
- không có raw User/Card/Merchant Name;
- không có Errors?;
- không có raw Timestamp;
- không có lineage metadata trong X.

Feature order bắt đầu bằng:

- `num__amount_numeric`;
- `num__time_since_previous_transaction_min`;
- `num__transactions_last_1h`;
- `num__amount_minus_previous_mean`;
- `bool__is_new_merchant`;
- `bool__has_prior_card_history`.

Nhận xét:

Schema đã ổn định và có thể được coi là baseline matrix schema v1 của current feature contract.

Status:

`VERIFIED — LOCKED FOR BASELINE V1`

---

## MR06 — Lineage

Observed:

Lineage được lưu riêng bằng:

- `row_id_train_w_long.npy`;
- `row_id_train_w_short.npy`;
- `row_id_validation.npy`.

Runtime gate xác nhận:

- số lineage row khớp X/y;
- mỗi lineage array không có duplicate row id;
- W_SHORT row ids là subset của W_LONG row ids.

Nhận xét:

`raw_row_id` tiếp tục phục vụ audit/reproducibility nhưng không trở thành classifier feature.

Status:

`VERIFIED — SEPARATE FROM X`

---

## MR07 — Artifact persistence

Output directory:

`data/processed/m4_07_baseline_ready`

Artifacts đã được tạo:

- `X_train_w_long.npz`;
- `X_train_w_short.npz`;
- `X_validation_w_long.npz`;
- `X_validation_w_short.npz`;
- `y_train_w_long.npy`;
- `y_train_w_short.npy`;
- `y_validation.npy`;
- `row_id_train_w_long.npy`;
- `row_id_train_w_short.npy`;
- `row_id_validation.npy`;
- `feature_names.json`;
- `manifest.json`.

Manifest gate:

`PASS`

Bổ sung sau Run All, saved-artifact round-trip verification cũng trả về:

`M4.7 SAVED ARTIFACT ROUND-TRIP GATE: PASS`

Điều này xác nhận artifact đã ghi xuống disk có thể load lại và vẫn khớp dữ liệu trong memory về shape, dtype, target, lineage và matrix content.

Status:

`VERIFIED — PERSISTED AND ROUND-TRIP CHECKED`

# 25. M4.7 Findings — runtime review

## M4.7-F01 — Full matrix construction integrity

Observed fact:

Full matrix pass xử lý đủ:

- `6,139` User+Card blocks;
- `24,386,900` raw rows;
- `23,038,920` history-context rows trước `2019-06-01`.

Matrix construction pass hoàn tất trong khoảng:

`174.01 seconds`

trên current environment.

Interpretation:

Pipeline có thể tái tạo full baseline-ready matrices từ raw artifact mà không cần materialize toàn bộ intermediate feature dataframe cùng lúc.

Implication:

End-to-end representation pipeline đủ khả thi về mặt tính toán cho current coursework environment.

Status:

`CONFIRMED`

---

## M4.7-F02 — W_LONG / W_SHORT row and target consistency

Observed fact:

W_LONG:

- X rows: `6,855,270`;
- fraud: `9,606`.

W_SHORT:

- X rows: `1,721,615`;
- fraud: `2,491`.

VALIDATION:

- rows: `712,458`;
- fraud: `1,052`.

Tất cả khớp canonical temporal/target contract từ M3/M4.2.

Interpretation:

Matrix assembly không làm mất row, nhân row hoặc lệch target population.

Implication:

Hai training-window candidates có thể tiếp tục được so sánh ở modeling stage bằng cùng downstream protocol.

Status:

`CONFIRMED`

---

## M4.7-F03 — Schema compatibility

Observed fact:

Cả bốn matrices đều có:

`47 columns`

W_LONG và W_SHORT dùng cùng feature names/order.

Feature names là unique và explicit.

Interpretation:

Training window thay đổi learned preprocessing state nhưng không làm thay đổi semantic/output schema.

Implication:

Downstream model code không cần viết hai feature pipelines khác nhau cho W_LONG và W_SHORT.

Status:

`CONFIRMED`

---

## M4.7-F04 — Preprocessing leakage audit

Observed fact:

Preprocessing fit pass:

W_LONG:

`6,855,270` TRAIN rows

range:

`2015-01-01 00:01` → `2018-12-31 23:58`

W_SHORT:

`1,721,615` TRAIN rows

range:

`2018-01-01 00:03` → `2018-12-31 23:58`

Target không nằm trong preprocessing fit usecols.

VALIDATION không gọi scaler/vocabulary fit.

Interpretation:

Scaler statistics và category vocabulary chỉ được học từ đúng TRAIN population của mỗi strategy.

Implication:

Không phát hiện preprocessing leakage từ VALIDATION vào TRAIN representation.

Status:

`CONFIRMED`

---

## M4.7-F05 — Numeric scaler integrity

Observed fact:

Sau khi sửa feature-wise incremental fitting, toàn bộ scaler means/variances/scales đều finite và scale > 0.

Đặc biệt hai historical numeric features có structural cold-start NA vẫn giữ số observed TRAIN rows hợp lý:

W_LONG:

- recency observed rows: `6,854,796`;
- amount deviation observed rows: `6,854,796`.

W_SHORT:

- recency observed rows: `1,721,515`;
- amount deviation observed rows: `1,721,515`.

Matrix density sau sửa đạt khoảng:

`19.20%`

thay vì trạng thái lỗi trước đó khoảng `14.95%`.

Interpretation:

Structural NA không còn làm hỏng incremental scaler state; recency và amount-deviation information được giữ trong matrix thay vì bị collapse thành zero columns.

Implication:

Current full matrices mới là artifact hợp lệ để downstream modeling sử dụng. Artifact từ lần chạy trước khi sửa scaler không được dùng.

Status:

`CONFIRMED`

---

## M4.7-F06 — Causal-history preservation

Observed fact:

Strict-causal regression test PASS.

Behavioral implementation tiếp tục áp dụng:

`Timestamp(history) < Timestamp(current)`

Same-timestamp peers không cung cấp history cho nhau.

Merchant novelty state chỉ update sau current timestamp group.

Interpretation:

Việc chuyển từ behavioral features sang full matrix không làm thay đổi causal semantics đã khóa ở M4.5.

Implication:

Không phát hiện future-history hoặc same-timestamp leakage trong matrix generation.

Status:

`CONFIRMED`

---

## M4.7-F07 — Target / raw-identifier isolation

Observed fact:

Output feature schema không chứa:

- `Is Fraud?`;
- User;
- Card;
- Merchant Name;
- Errors?;
- raw_row_id;
- Timestamp.

`raw_row_id` chỉ được lưu trong lineage arrays riêng.

Target chỉ được đọc ở matrix construction pass sau khi preprocessing bundles đã freeze.

Interpretation:

Target và raw identifiers được tách khỏi classifier feature matrix.

Implication:

Không phát hiện target leakage hoặc direct identifier exposure trong baseline X.

Status:

`CONFIRMED`

---

## M4.7-F08 — Matrix sparsity and memory footprint

Observed fact:

Density:

- W_LONG TRAIN: `19.211867%`;
- W_SHORT TRAIN: `19.208067%`;
- W_LONG VALIDATION: `19.203339%`;
- W_SHORT VALIDATION: `19.203339%`.

Approximate CSR memory:

- W_LONG TRAIN: `498.41 MB`;
- W_SHORT TRAIN: `125.15 MB`;
- mỗi VALIDATION matrix: `51.78 MB`.

Interpretation:

One-hot representation vẫn đủ sparse để CSR hợp lý hơn dense representation.

Implication:

CSR `float32` tiếp tục là representation phù hợp cho baseline-ready artifacts hiện tại.

Status:

`CONFIRMED`

---

## M4.7-F09 — Deterministic transform

Observed fact:

Repeat-transform audit trên `88` deterministic VALIDATION rows cho kết quả:

W_LONG:

`identical = True`

W_SHORT:

`identical = True`

Interpretation:

Với cùng input và cùng frozen preprocessing bundle, transform cho kết quả tái hiện được.

Implication:

Representation không phụ thuộc hidden mutable state trong transform stage.

Status:

`CONFIRMED`

---

## M4.7-F10 — Artifact persistence and round-trip integrity

Observed fact:

Toàn bộ sparse matrices, targets, lineage arrays, feature names và manifest đã được ghi xuống canonical processed directory.

Manifest audit PASS.

Bổ sung round-trip audit cũng PASS:

`M4.7 SAVED ARTIFACT ROUND-TRIP GATE: PASS`

Interpretation:

Artifact trên disk không chỉ tồn tại mà còn có thể load lại và khớp dữ liệu trong memory.

Implication:

M4.7 đã tạo được reusable modeling artifacts thay vì chỉ tạo temporary in-memory objects.

Status:

`CONFIRMED`

---

## M4.7-F11 — Readiness for downstream baseline modeling

Observed fact:

Đã có đầy đủ:

- X/y cho W_LONG;
- X/y cho W_SHORT;
- hai VALIDATION transforms;
- common feature schema;
- lineage;
- manifest;
- persisted artifacts.

Interpretation:

Downstream baseline model có thể sử dụng artifacts này mà không phải tự xây lại feature engineering hoặc preprocessing.

Implication:

M4.7 hoàn thành technical modeling-matrix gate.

Điều này không có nghĩa W_LONG hoặc W_SHORT đã thắng; training-window winner vẫn phải được quyết định bằng experiment protocol sau.

Status:

`CONFIRMED`

# 26. Decision Log M4.7 — runtime review

## M4.7-D01 — Baseline matrix schema

Decision:

Baseline output schema được khóa ở:

`47 features`

Representation:

`CSR sparse matrix`

Dtype:

`float32`

Feature names/order:

stable và identical giữa W_LONG/W_SHORT.

Status:

`LOCKED FOR BASELINE V1`

---

## M4.7-D02 — W_LONG matrix readiness

Decision:

`X_train_w_long`

+

`y_train_w_long`

được chấp nhận là baseline-ready W_LONG training artifacts.

Rows:

`6,855,270`

Fraud:

`9,606`

Status:

`VERIFIED — READY`

---

## M4.7-D03 — W_SHORT matrix readiness

Decision:

`X_train_w_short`

+

`y_train_w_short`

được chấp nhận là baseline-ready W_SHORT training artifacts.

Rows:

`1,721,615`

Fraud:

`2,491`

Status:

`VERIFIED — READY`

---

## M4.7-D04 — Validation transform policy

Decision:

Giữ hai validation matrices:

- `X_validation_w_long`;
- `X_validation_w_short`.

Hai matrices dùng cùng validation rows/target nhưng mỗi matrix được transform bằng learned state của training strategy tương ứng.

VALIDATION không refit preprocessing state.

Status:

`LOCKED`

---

## M4.7-D05 — Target representation

Decision:

Binary target:

- fraud = `1`;
- non-fraud = `0`.

Dtype:

`int8`

Target lưu riêng khỏi X.

Status:

`LOCKED`

---

## M4.7-D06 — Lineage policy

Decision:

`raw_row_id` được lưu riêng dưới dạng lineage arrays.

Không đưa lineage vào classifier matrix.

Status:

`LOCKED`

---

## M4.7-D07 — Artifact format

Decision:

Matrices:

`CSR float32 .npz`

Targets:

`int8 .npy`

Lineage:

`int64 .npy`

Feature names:

`.json`

Manifest:

`.json`

Saved artifacts phải qua existence/manifest check và round-trip verification trước khi được coi là reusable.

Status:

`LOCKED`

---

## M4.7-D08 — Numeric scaler integrity

Decision:

Numeric incremental fit phải bỏ qua batch không có finite observed values cho từng feature.

Scaler state phải thỏa:

- finite mean;
- finite variance;
- finite positive scale.

Status:

`LOCKED — FAIL LOUDLY`

---

## M4.7-D09 — Preprocessing fit boundary

Decision:

W_LONG preprocessing state:

fit từ W_LONG TRAIN only.

W_SHORT preprocessing state:

fit từ W_SHORT TRAIN only.

VALIDATION:

transform only.

FINAL TEST:

không tham gia M4.7.

Status:

`INHERITED — VERIFIED — LOCKED`

---

## M4.7-D10 — W_SHORT subset invariant

Decision:

W_SHORT training rows phải là row-level subset của W_LONG training rows dưới canonical temporal contract.

Runtime lineage assertion đã PASS.

Status:

`VERIFIED — LOCKED`

---

## M4.7-D11 — Training-window winner

Decision:

Không chọn W_LONG hoặc W_SHORT winner tại M4.7.

M4.7 chỉ chứng minh cả hai representations đều baseline-ready.

Winner:

`OPEN — REQUIRES MODELING EXPERIMENT`

Status:

`OPEN BY DESIGN`

---

## M4.7-D12 — M4 handoff readiness

Decision:

M4.7 artifacts đủ điều kiện chuyển sang M4.8 để tổng hợp:

- Feature Specification;
- Preprocessing Specification;
- Behavioral Feature Contract;
- pipeline version;
- integrity/leakage audit;
- Decision Log;
- Open Questions;
- M4 Gate.

Status:

`READY FOR M4.8`

# 27. M4.7 Gate — runtime review

## G01 — Raw artifact and dependency gate

Result:

`PASS`

Current raw artifact tồn tại, file size khớp canonical artifact và required Python dependencies đã load thành công.

---

## G02 — Strict-causal behavioral regression gate

Result:

`PASS`

Behavioral builder giữ strict prior-history semantics và same-timestamp exclusion.

---

## G03 — TRAIN-only preprocessing refit

Result:

`PASS`

W_LONG và W_SHORT preprocessing states chỉ được fit từ TRAIN population tương ứng.

---

## G04 — W_LONG full matrix row count

Result:

`PASS`

Observed:

`6,855,270 × 47`

---

## G05 — W_SHORT full matrix row count

Result:

`PASS`

Observed:

`1,721,615 × 47`

---

## G06 — VALIDATION full matrix row count

Result:

`PASS`

Cả hai validation transforms:

`712,458 × 47`

---

## G07 — Canonical fraud counts

Result:

`PASS`

Observed:

- W_LONG: `9,606`;
- W_SHORT: `2,491`;
- VALIDATION: `1,052`.

---

## G08 — Temporal boundaries

Result:

`PASS`

W_LONG, W_SHORT và VALIDATION đều nằm đúng canonical half-open intervals.

---

## G09 — X/y/lineage alignment

Result:

`PASS`

Row counts giữa X, y và lineage arrays khớp chính xác.

---

## G10 — W_SHORT subset invariant

Result:

`PASS`

W_SHORT lineage được xác minh là row-level subset của W_LONG lineage.

---

## G11 — No NaN/inf

Result:

`PASS`

Toàn bộ sparse matrix data đều finite.

Numeric scaler state cũng đã được kiểm tra finite trước khi matrix construction.

---

## G12 — CSR float32 output

Result:

`PASS`

Cả bốn matrices là CSR và dtype `float32`.

---

## G13 — Stable feature width/order

Result:

`PASS`

Feature count:

`47`

W_LONG/W_SHORT dùng cùng names/order.

---

## G14 — No prohibited raw fields in X

Result:

`PASS`

Không có User, Card, Merchant Name, Errors?, raw_row_id hoặc Timestamp trong X.

---

## G15 — Target isolated from X and preprocessing fit

Result:

`PASS`

Target không tham gia feature engineering hoặc preprocessing fit và không xuất hiện trong X.

---

## G16 — FINAL TEST protected

Result:

`PASS`

Matrix construction và preprocessing decision chỉ sử dụng history/development rows trước `2019-06-01`; FINAL TEST không được dùng.

---

## G17 — Repeat transform deterministic

Result:

`PASS`

W_LONG và W_SHORT repeated transform trên deterministic validation sample cho identical outputs.

---

## G18 — Saved artifacts complete

Result:

`PASS`

Manifest audit:

`PASS`

Bổ sung saved-artifact round-trip verification:

`M4.7 SAVED ARTIFACT ROUND-TRIP GATE: PASS`

---

## G19 — W_LONG/W_SHORT baseline-ready

Result:

`PASS`

Cả hai training-window candidates đã có full X/y artifacts và corresponding validation transforms theo cùng feature/preprocessing contract.

---

## G20 — Ready for M4.8 consolidation

Result:

`PASS`

Không còn blocking technical issue trong M4.7.

---

# M4.7 Gate

Overall:

`PASS`

Blocking issue:

`NONE`

M4.7 Status:

`PASS — BASELINE-READY MATRICES VERIFIED`

Handoff:

`READY FOR M4.8`

# 28. Kết luận M4.7

## Mục tiêu đã hoàn thành

M4.7 đã tạo full baseline-ready modeling representation cho cả hai training-window candidates:

W_LONG:

- `X_train_w_long`;
- `y_train_w_long`;
- `X_validation_w_long`.

W_SHORT:

- `X_train_w_short`;
- `y_train_w_short`;
- `X_validation_w_short`.

Shared:

- `y_validation`;
- feature names;
- row lineage;
- artifact manifest.

---

## Full matrix integrity

Observed shapes:

W_LONG TRAIN:

`6,855,270 × 47`

W_SHORT TRAIN:

`1,721,615 × 47`

VALIDATION transformed by W_LONG:

`712,458 × 47`

VALIDATION transformed by W_SHORT:

`712,458 × 47`

Target vectors có đúng số row tương ứng.

Canonical fraud counts cũng khớp:

- W_LONG: `9,606`;
- W_SHORT: `2,491`;
- VALIDATION: `1,052`.

Không phát hiện row-loss, row-duplication hoặc X/y misalignment.

---

## Preprocessing leakage boundary

Preprocessing states đã được fit lại end-to-end từ TRAIN-only source.

W_LONG fit range kết thúc trước:

`2019-01-01`

W_SHORT fit range kết thúc trước:

`2019-01-01`

VALIDATION chỉ transform.

Target không được đọc trong preprocessing fit pass.

FINAL TEST không được dùng để fit, transform-decision hoặc audit M4.7.

Kết luận:

`TRAIN-ONLY LEARNED PREPROCESSING STATE — VERIFIED`

---

## Numeric preprocessing integrity

Lỗi incremental `StandardScaler` phát hiện trong lần chạy trước đã được sửa bằng feature-wise finite-value fitting.

Current run xác nhận:

- mọi mean finite;
- mọi variance finite;
- mọi scale finite;
- mọi scale > 0;
- recency và amount-deviation giữ observed support đúng structural cold-start semantics.

Matrix density quay về khoảng:

`19.2%`

phù hợp với M4.6 preprocessing contract.

Kết luận:

`NUMERIC PREPROCESSING INTEGRITY — VERIFIED`

Chỉ current regenerated artifacts sau fix này được coi là valid baseline artifacts.

---

## Causal behavioral contract

Strict-causal regression test PASS.

Behavioral history tiếp tục tuân thủ:

`Timestamp(history) < Timestamp(current)`

Same-timestamp transaction không cung cấp history cho nhau.

Merchant novelty không sử dụng future transaction.

Kết luận:

`M4.5 CAUSAL CONTRACT — PRESERVED`

---

## Feature schema

Final baseline matrix schema:

`47 features`

bao gồm:

- 4 scaled numeric columns;
- 2 boolean columns;
- 41 one-hot categorical columns.

W_LONG và W_SHORT dùng cùng feature names/order.

Explicit unknown-category columns được giữ trong schema.

Không có:

- target;
- raw User;
- raw Card;
- raw Merchant Name;
- Errors?;
- raw Timestamp;
- raw_row_id.

Kết luận:

`BASELINE MATRIX SCHEMA V1 — LOCKED`

---

## Lineage

Mỗi matrix population có lineage array riêng.

`raw_row_id` không xuất hiện trong X.

W_SHORT được runtime-verified là subset của W_LONG theo row lineage.

Kết luận:

`LINEAGE CONTRACT — VERIFIED`

---

## Sparse representation và computational note

Cả bốn matrices được lưu dưới dạng:

`CSR float32`

Observed density khoảng:

`19.20%`

Approximate in-memory CSR footprint:

- W_LONG TRAIN: `498.41 MB`;
- W_SHORT TRAIN: `125.15 MB`;
- mỗi VALIDATION matrix: `51.78 MB`.

Fit preprocessing pass mất khoảng:

`137.27 seconds`

Matrix construction pass mất khoảng:

`174.01 seconds`

trên current environment.

Đây là workload lớn hơn các notebook audit trước vì M4.7 thực sự tạo representation cho hàng triệu transaction.

Không phát hiện computational blocker cho current project environment.

---

## Artifact persistence

Artifacts đã được lưu tại:

`data/processed/m4_07_baseline_ready`

Bao gồm:

- four sparse X matrices;
- training/validation target vectors;
- lineage vectors;
- feature names;
- manifest.

Manifest gate PASS.

Bổ sung round-trip verification cũng PASS:

`M4.7 SAVED ARTIFACT ROUND-TRIP GATE: PASS`

Do đó artifact trên disk đã được xác minh có thể load lại và vẫn giữ đúng matrix/target/lineage content.

Kết luận:

`PERSISTED MODELING ARTIFACTS — VERIFIED`

---

## Những điều M4.7 chưa kết luận

M4.7 không chứng minh:

- W_LONG tốt hơn W_SHORT hoặc ngược lại;
- feature hiện tại cải thiện F1_fraud bao nhiêu;
- model nào tốt nhất;
- class-imbalance strategy nào tốt nhất;
- hyperparameter nào tối ưu;
- threshold nào nên dùng.

Những câu hỏi đó phải được giải quyết bằng modeling experiment theo thứ tự đã khóa từ M3.

Vì vậy:

`BASELINE-READY ≠ BEST MODEL`

và:

`W_LONG/W_SHORT WINNER = OPEN`

---

## Blocking issue

`NONE`

---

## M4.7 Gate

`PASS`

M4.7 đã chứng minh rằng cả W_LONG và W_SHORT có thể được chuyển từ raw transaction artifact thành full baseline-ready X/y matrices bằng cùng feature contract, strict-causal behavioral contract và leakage-safe preprocessing procedure.

---

## Trạng thái cuối

`M4.7 — PASS`

`Baseline-ready Modeling Matrix Contract — LOCKED FOR BASELINE V1`

`W_LONG — READY FOR MODELING EXPERIMENT`

`W_SHORT — READY FOR MODELING EXPERIMENT`

`FINAL TEST — STILL PROTECTED`

`READY FOR M4.8`

---

## Handoff

Next:

`M4.8 — Tổng hợp Feature Specification, Preprocessing Specification, Behavioral Feature Contract, Decision Log và M4 Gate`

M4.8 cần tổng hợp lại các quyết định đã khóa ở M4.1–M4.7 và xác nhận Milestone 4 có đủ điều kiện đóng gate trước khi chuyển sang modeling experiments.
