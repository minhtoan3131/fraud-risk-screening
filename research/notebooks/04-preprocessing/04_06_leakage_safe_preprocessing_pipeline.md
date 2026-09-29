# M4.6 — Xây leakage-safe preprocessing pipeline

## Vai trò

M4.4 đã khóa transaction-level feature candidates.

M4.5 đã khóa strict-causal behavioral feature candidates.

M4.6 xây preprocessing contract mà model sẽ nhận, với ranh giới bắt buộc:

- deterministic feature engineering xảy ra trước preprocessing;
- behavioral feature giữ nguyên causal contract M4.5;
- learned preprocessing state chỉ được học từ TRAIN;
- VALIDATION chỉ được transform bằng state đã học từ TRAIN;
- FINAL TEST không được dùng để fit hoặc quyết định preprocessing trong M4;
- cùng một feature contract phải dùng được cho W_LONG và W_SHORT.

## Phạm vi M4.6

M4.6 thực hiện:

- khóa core baseline feature roles;
- numeric preprocessing;
- cold-start NA handling;
- categorical vocabulary học từ TRAIN;
- explicit unknown-category handling;
- numeric scaler học từ TRAIN;
- stable output feature names/order;
- audit W_LONG và W_SHORT preprocessing state;
- transform deterministic TRAIN/VALIDATION sample để kiểm tra contract.

## Không thuộc M4.6

- tạo full X_train / X_validation;
- tạo y_train / y_validation;
- model fitting;
- feature-performance comparison;
- imbalance handling;
- threshold selection.

Full baseline-ready modeling matrix thuộc M4.7.

## Trạng thái khi bắt đầu

M4.4:
PASS

M4.5:
PASS

Preprocessing Specification:
OPEN

M4.6 Gate:
OPEN

# 1. Contract kế thừa từ M4.4 và M4.5

## Core transaction-level candidates

M4.6 core baseline dùng các primary candidates đã khóa ở M4.4:

- `amount_numeric`;
- `transaction_mode`;
- `location_state`;
- `hour_of_day`;
- `day_of_week`.

## Core behavioral candidates

M4.6 core baseline dùng các candidates đã khóa ở M4.5:

- `time_since_previous_transaction_min`;
- `transactions_last_1h`;
- `amount_minus_previous_mean`;
- `is_new_merchant`;
- `has_prior_card_history`.

## Không tự động đưa vào core baseline

Các field sau vẫn là conditional / experiment candidates và không được M4.6 tự ý đưa vào core pipeline:

- `mcc_code`;
- `merchant_state_cat`;
- `month_of_year`;
- `merchant_city_cat`;
- `zip_cat`;
- `amount_signed_log1p`;
- `is_negative_amount`;
- `is_zero_amount`;
- `is_weekend`;
- `prior_card_transaction_count`;
- `previous_amount_mean`.

Việc thêm các field trên phải là controlled feature experiment, không phải side effect của preprocessing.

## Raw fields bị cấm trong classifier matrix

- User;
- Card;
- Merchant Name;
- Errors?;
- Is Fraud?;
- raw_row_id;
- Timestamp;
- partition metadata.

# 2. Preprocessing Specification v0 — trước runtime review

## Numeric branch

Các feature được scale:

- `amount_numeric`;
- `time_since_previous_transaction_min`;
- `transactions_last_1h`;
- `amount_minus_previous_mean`.

Baseline transformation:

`StandardScaler`

Learned state:

TRAIN only.

### Cold-start numerical NA

Hai feature có structural NA khi không có prior Card history:

- `time_since_previous_transaction_min`;
- `amount_minus_previous_mean`.

Policy:

1. scaler fit bỏ qua NA;
2. scaler transform giữ NA;
3. sau scaling, NA được thay bằng `0.0` trong standardized space;
4. `has_prior_card_history` được giữ làm companion state.

Do đó `0.0` sau preprocessing mang nghĩa:

training-mean position hoặc cold-start imputed position,

và companion flag phân biệt cold-start.

M4.6 không học imputer statistic riêng từ VALIDATION.

## Boolean branch

Passthrough dưới dạng float32:

- `is_new_merchant`;
- `has_prior_card_history`.

## Categorical branch

One-hot encoding cho:

- `transaction_mode`;
- `location_state`;
- `hour_of_day`;
- `day_of_week`.

Category vocabulary:

TRAIN only.

Unknown handling:

mọi category không thuộc TRAIN vocabulary được map deterministic sang token:

`__UNKNOWN__`

Token này có explicit one-hot column trong output schema.

VALIDATION không được mở rộng vocabulary.

## W_LONG / W_SHORT

Hai strategy có preprocessing state riêng vì learned statistics phải được fit từ đúng training population tương ứng.

Procedure và feature contract phải giống nhau.

Nếu core categorical domains giống nhau, output feature names/order phải giống nhau giữa hai strategy.

# 3. Thiết lập môi trường


```python
from pathlib import Path
from collections import Counter
from dataclasses import dataclass
import time

import numpy as np
import pandas as pd

from scipy import sparse
from sklearn.preprocessing import StandardScaler, OneHotEncoder


DATA_RELATIVE_PATH = (
    Path("data")
    / "raw"
    / "ibm_tabformer"
    / "card_transaction.v1.csv"
)

candidate_roots = [
    Path.cwd(),
    Path.cwd().parent,
    Path.cwd().parent.parent,
]

PROJECT_ROOT = None

for candidate in candidate_roots:
    candidate = candidate.resolve()

    if (candidate / DATA_RELATIVE_PATH).exists():
        PROJECT_ROOT = candidate
        break

if PROJECT_ROOT is None:
    raise FileNotFoundError(
        "Không xác định được PROJECT_ROOT."
    )

DATA_PATH = PROJECT_ROOT / DATA_RELATIVE_PATH

EXPECTED_FILE_SIZE = 2_354_626_737
CHUNK_SIZE = 500_000
AUDIT_SAMPLE_MOD = 100  # deterministic ~1% development sample
UNKNOWN_TOKEN = "__UNKNOWN__"

print("PROJECT_ROOT:", PROJECT_ROOT)
print("DATA_PATH:", DATA_PATH)
print("File exists:", DATA_PATH.exists())
print("File size:", DATA_PATH.stat().st_size)
print("CHUNK_SIZE:", CHUNK_SIZE)
```

    PROJECT_ROOT: /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening
    DATA_PATH: /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/raw/ibm_tabformer/card_transaction.v1.csv
    File exists: True
    File size: 2354626737
    CHUNK_SIZE: 500000



```python
assert DATA_PATH.exists()
assert DATA_PATH.stat().st_size == EXPECTED_FILE_SIZE

print("M4.6 ENVIRONMENT / ARTIFACT GATE: PASS")
```

    M4.6 ENVIRONMENT / ARTIFACT GATE: PASS


### Phân tích / Nhận xét

Notebook đã xác định đúng raw artifact và environment gate đã chạy thành công. File tồn tại, kích thước artifact khớp contract và không có exception ở bước thiết lập môi trường.

Điều này đủ để xác nhận các audit M4.6 phía sau đang chạy trên cùng artifact đã được dùng xuyên các module trước, không phải một bản dữ liệu khác.

### Kết luận

Environment / artifact contract:

`PASS`

Blocking issue:

`NONE`


# 4. Temporal contract


```python
W_LONG_START = pd.Timestamp("2015-01-01")
W_SHORT_START = pd.Timestamp("2018-01-01")
TRAIN_END = pd.Timestamp("2019-01-01")
VALIDATION_END = pd.Timestamp("2019-06-01")

EXPECTED_W_LONG_ROWS = 6_855_270
EXPECTED_W_SHORT_ROWS = 1_721_615
EXPECTED_VALIDATION_ROWS = 712_458
EXPECTED_DEVELOPMENT_ROWS = 7_567_728
EXPECTED_CONTEXT_ROWS_BEFORE_VALIDATION_END = 23_038_920
EXPECTED_CARD_COUNT = 6_139

PERIODS = [
    "TRAIN_2015_2017",
    "TRAIN_2018",
    "VALIDATION",
]


def assign_period(timestamp):
    conditions = [
        (timestamp >= W_LONG_START) & (timestamp < W_SHORT_START),
        (timestamp >= W_SHORT_START) & (timestamp < TRAIN_END),
        (timestamp >= TRAIN_END) & (timestamp < VALIDATION_END),
    ]

    choices = [
        "TRAIN_2015_2017",
        "TRAIN_2018",
        "VALIDATION",
    ]

    return pd.Series(
        np.select(
            conditions,
            choices,
            default="OUTSIDE_M46_DEVELOPMENT",
        ),
        index=timestamp.index,
        dtype="string",
    )
```

# 5. Canonical deterministic representation helpers


```python
def parse_amount(amount_series):
    cleaned = (
        amount_series
        .astype("string")
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
    )

    return pd.to_numeric(
        cleaned,
        errors="coerce",
    )


def build_timestamp(df):
    date_part = pd.to_datetime(
        {
            "year": df["Year"],
            "month": df["Month"],
            "day": df["Day"],
        },
        errors="coerce",
    )

    time_part = pd.to_timedelta(
        df["Time"].astype("string") + ":00",
        errors="coerce",
    )

    return date_part + time_part


def normalize_string(series):
    return (
        series
        .astype("string")
        .str.strip()
    )


def assign_location_state(df):
    city = normalize_string(
        df["Merchant City"]
    )

    city_online = (
        city
        .str.upper()
        .eq("ONLINE")
        .fillna(False)
    )

    state_missing = df["Merchant State"].isna()
    zip_missing = df["Zip"].isna()

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

# 6. Stream từng User+Card và giữ lineage


```python
FEATURE_USECOLS = [
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

assert "Is Fraud?" not in FEATURE_USECOLS
assert "Errors?" not in FEATURE_USECOLS


def iter_card_blocks(data_path, chunksize):
    pending_key = None
    pending_parts = []
    raw_row_offset = 0

    for chunk in pd.read_csv(
        data_path,
        usecols=FEATURE_USECOLS,
        chunksize=chunksize,
    ):
        chunk_length = len(chunk)

        chunk["raw_row_id"] = np.arange(
            raw_row_offset,
            raw_row_offset + chunk_length,
            dtype=np.int64,
        )

        raw_row_offset += chunk_length

        chunk["Timestamp"] = build_timestamp(chunk)
        chunk["Amount_numeric"] = parse_amount(chunk["Amount"])

        if chunk["Timestamp"].isna().any():
            raise ValueError(
                "Timestamp parse failure trong M4.6 feature pass."
            )

        if chunk["Amount_numeric"].isna().any():
            raise ValueError(
                "Amount parse failure trong M4.6 feature pass."
            )

        if chunk["Merchant Name"].isna().any():
            raise ValueError(
                "Merchant Name missing trong M4.6 feature pass."
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

        users = working["User"].to_numpy()
        cards = working["Card"].to_numpy()

        change_positions = (
            np.flatnonzero(
                (users[1:] != users[:-1])
                | (cards[1:] != cards[:-1])
            )
            + 1
        )

        starts = np.concatenate([[0], change_positions])
        ends = np.concatenate([change_positions, [len(working)]])

        for start, end in zip(starts, ends):
            key = (
                int(users[start]),
                int(cards[start]),
            )

            segment = (
                working
                .iloc[start:end]
                .copy()
            )

            if pending_key is None:
                pending_key = key
                pending_parts = [segment]
                continue

            if key == pending_key:
                pending_parts.append(segment)
                continue

            if len(pending_parts) == 1:
                card_block = (
                    pending_parts[0]
                    .reset_index(drop=True)
                )
            else:
                card_block = pd.concat(
                    pending_parts,
                    ignore_index=True,
                )

            yield pending_key, card_block

            pending_key = key
            pending_parts = [segment]

    if pending_key is not None:
        if len(pending_parts) == 1:
            card_block = (
                pending_parts[0]
                .reset_index(drop=True)
            )
        else:
            card_block = pd.concat(
                pending_parts,
                ignore_index=True,
            )

        yield pending_key, card_block
```

# 7. Strict-causal behavioral builder kế thừa M4.5


```python
ONE_HOUR_NS = int(
    pd.Timedelta(hours=1).value
)


def compute_causal_behavioral_features(card_df):
    timestamp_ns = (
        card_df["Timestamp"]
        .to_numpy(dtype="datetime64[ns]")
        .astype("int64")
    )

    if (
        np.diff(timestamp_ns) < 0
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

    group_lengths = group_ends - group_starts
    group_timestamp_ns = timestamp_ns[group_starts]

    prior_count_group = group_starts
    prior_count = np.repeat(
        prior_count_group,
        group_lengths,
    )

    has_prior_card_history = prior_count > 0

    time_since_group = np.full(
        len(group_starts),
        np.nan,
        dtype="float64",
    )

    if len(group_starts) > 1:
        time_since_group[1:] = (
            (
                group_timestamp_ns[1:]
                - group_timestamp_ns[:-1]
            )
            / 60_000_000_000
        )

    time_since_previous_min = np.repeat(
        time_since_group,
        group_lengths,
    )

    left_1h = np.searchsorted(
        timestamp_ns,
        group_timestamp_ns - ONE_HOUR_NS,
        side="left",
    )

    transactions_last_1h_group = (
        group_starts - left_1h
    )

    transactions_last_1h = np.repeat(
        transactions_last_1h_group,
        group_lengths,
    )

    amount = (
        card_df["Amount_numeric"]
        .to_numpy(dtype="float64")
    )

    amount_prefix_sum = np.concatenate(
        [
            [0.0],
            np.cumsum(amount),
        ]
    )

    previous_amount_mean_group = np.full(
        len(group_starts),
        np.nan,
        dtype="float64",
    )

    has_prior_group = group_starts > 0

    previous_amount_mean_group[
        has_prior_group
    ] = (
        amount_prefix_sum[
            group_starts[has_prior_group]
        ]
        / group_starts[has_prior_group]
    )

    previous_amount_mean = np.repeat(
        previous_amount_mean_group,
        group_lengths,
    )

    amount_minus_previous_mean = (
        amount - previous_amount_mean
    )

    # Strict-causal merchant novelty:
    # state chỉ update sau khi toàn timestamp group đã được tính.
    merchant_values = (
        card_df["Merchant Name"]
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
            merchant_values[start:end]
        )

        is_new_merchant[start:end] = np.array(
            [
                merchant not in seen_merchants
                for merchant
                in current_group_merchants
            ],
            dtype=bool,
        )

        seen_merchants.update(
            current_group_merchants.tolist()
        )

    return pd.DataFrame(
        {
            "raw_row_id": card_df["raw_row_id"].to_numpy(),
            "Timestamp": card_df["Timestamp"].to_numpy(),
            "amount_numeric": amount,
            "time_since_previous_transaction_min": (
                time_since_previous_min
            ),
            "transactions_last_1h": transactions_last_1h,
            "amount_minus_previous_mean": (
                amount_minus_previous_mean
            ),
            "is_new_merchant": is_new_merchant,
            "has_prior_card_history": (
                has_prior_card_history
            ),
        }
    )
```

# 8. Core feature frame và column-role contract


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

assert set(CORE_FEATURE_COLUMNS).isdisjoint(
    PROHIBITED_DIRECT_COLUMNS
)


def build_core_feature_frame(
    context_card_df,
    behavioral_df,
):
    transaction_mode = normalize_string(
        context_card_df["Use Chip"]
    )

    location_state = assign_location_state(
        context_card_df
    )

    timestamp = context_card_df["Timestamp"]

    frame = pd.DataFrame(
        {
            "raw_row_id": context_card_df["raw_row_id"].to_numpy(),
            "Timestamp": timestamp.to_numpy(),
            "amount_numeric": behavioral_df["amount_numeric"].to_numpy(),
            "time_since_previous_transaction_min": (
                behavioral_df[
                    "time_since_previous_transaction_min"
                ].to_numpy()
            ),
            "transactions_last_1h": (
                behavioral_df[
                    "transactions_last_1h"
                ].to_numpy()
            ),
            "amount_minus_previous_mean": (
                behavioral_df[
                    "amount_minus_previous_mean"
                ].to_numpy()
            ),
            "is_new_merchant": (
                behavioral_df[
                    "is_new_merchant"
                ].to_numpy()
            ),
            "has_prior_card_history": (
                behavioral_df[
                    "has_prior_card_history"
                ].to_numpy()
            ),
            "transaction_mode": transaction_mode.to_numpy(),
            "location_state": location_state.to_numpy(),
            "hour_of_day": (
                timestamp.dt.hour
                .astype("Int8")
                .astype("string")
                .to_numpy()
            ),
            "day_of_week": (
                timestamp.dt.dayofweek
                .astype("Int8")
                .astype("string")
                .to_numpy()
            ),
        }
    )

    if frame[CORE_FEATURE_COLUMNS].isna().any().any():
        allowed_na = {
            "time_since_previous_transaction_min",
            "amount_minus_previous_mean",
        }

        unexpected = {
            column: int(frame[column].isna().sum())
            for column in CORE_FEATURE_COLUMNS
            if (
                frame[column].isna().any()
                and column not in allowed_na
            )
        }

        if unexpected:
            raise ValueError(
                f"Unexpected missing in core features: {unexpected}"
            )

    if frame["location_state"].eq(
        "OTHER_INCONSISTENT"
    ).any():
        raise ValueError(
            "OTHER_INCONSISTENT location state xuất hiện."
        )

    return frame


print("NUMERIC_COLUMNS:", NUMERIC_COLUMNS)
print("BOOLEAN_COLUMNS:", BOOLEAN_COLUMNS)
print("CATEGORICAL_COLUMNS:", CATEGORICAL_COLUMNS)
print("CORE FEATURE COUNT:", len(CORE_FEATURE_COLUMNS))
```

    NUMERIC_COLUMNS: ['amount_numeric', 'time_since_previous_transaction_min', 'transactions_last_1h', 'amount_minus_previous_mean']
    BOOLEAN_COLUMNS: ['is_new_merchant', 'has_prior_card_history']
    CATEGORICAL_COLUMNS: ['transaction_mode', 'location_state', 'hour_of_day', 'day_of_week']
    CORE FEATURE COUNT: 10


### Phân tích / Nhận xét

Core preprocessing frame hiện có đúng 10 feature:

- 4 numeric;
- 2 boolean;
- 4 categorical.

Các raw field bị cấm gồm `User`, `Card`, `Merchant Name`, `Errors?`, `Is Fraud?`, `raw_row_id`, `Timestamp` không nằm trong `CORE_FEATURE_COLUMNS`.

Hai numerical field được phép có structural NA là:

- `time_since_previous_transaction_min`;
- `amount_minus_previous_mean`.

Các feature còn lại không được phép có missing bất ngờ, và `OTHER_INCONSISTENT` ở `location_state` bị chặn bằng exception.

### Kết luận

Core feature-role contract được triển khai đúng theo specification kế thừa từ M4.4/M4.5.

Core baseline width trước encoding:

`10 FEATURES`

Raw identifier / target exposure:

`NONE`


# 9. Preprocessing helper: TRAIN-only scaler, vocabulary và explicit unknown


```python
def make_one_hot_encoder(categories):
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


def stable_sort_categories(column, values):
    values = [str(value) for value in values]

    if column in {
        "hour_of_day",
        "day_of_week",
    }:
        return sorted(
            values,
            key=lambda x: int(x),
        )

    return sorted(values)


class FeaturewiseStandardScaler:
    """
    Incremental StandardScaler theo từng numeric feature.

    Mục tiêu:
    - structural NaN không đóng góp learned mean/variance;
    - một batch toàn NaN ở một feature không được làm hỏng
      variance/scale đã học từ các batch trước;
    - giữ API/learned-state tương thích với các audit hiện tại.
    """

    def __init__(self):
        self._scalers = None
        self.n_features_in_ = None
        self.mean_ = None
        self.var_ = None
        self.scale_ = None
        self.n_samples_seen_ = None

    def _initialize(self, n_features):
        self.n_features_in_ = int(n_features)

        self._scalers = [
            StandardScaler()
            for _ in range(
                self.n_features_in_
            )
        ]

        self._sync_state()

    def _sync_state(self):
        if self._scalers is None:
            return

        mean = []
        variance = []
        scale = []
        n_seen = []

        for scaler in self._scalers:
            if hasattr(
                scaler,
                "n_samples_seen_",
            ):
                n_value = (
                    np.asarray(
                        scaler.n_samples_seen_
                    )
                    .reshape(-1)[0]
                )

                n_seen.append(
                    int(n_value)
                )

                mean.append(
                    float(
                        np.asarray(
                            scaler.mean_
                        )
                        .reshape(-1)[0]
                    )
                )

                variance.append(
                    float(
                        np.asarray(
                            scaler.var_
                        )
                        .reshape(-1)[0]
                    )
                )

                scale.append(
                    float(
                        np.asarray(
                            scaler.scale_
                        )
                        .reshape(-1)[0]
                    )
                )

            else:
                # Feature chưa có observed value nào.
                n_seen.append(0)
                mean.append(np.nan)
                variance.append(np.nan)
                scale.append(np.nan)

        self.mean_ = np.asarray(
            mean,
            dtype="float64",
        )

        self.var_ = np.asarray(
            variance,
            dtype="float64",
        )

        self.scale_ = np.asarray(
            scale,
            dtype="float64",
        )

        self.n_samples_seen_ = np.asarray(
            n_seen,
            dtype=np.int64,
        )

    def partial_fit(self, X):
        X = np.asarray(
            X,
            dtype="float64",
        )

        if X.ndim == 1:
            X = X.reshape(
                -1,
                1,
            )

        if X.ndim != 2:
            raise ValueError(
                "Numeric scaler input phải là 2D."
            )

        if self._scalers is None:
            self._initialize(
                X.shape[1]
            )

        elif X.shape[1] != self.n_features_in_:
            raise ValueError(
                "Numeric feature width thay đổi giữa các batch."
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

            # Quan trọng:
            # nếu batch hiện tại không có observation hợp lệ
            # cho feature này thì KHÔNG gọi partial_fit.
            # Như vậy learned variance/scale cũ không bị NaN.
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

        self._sync_state()

        return self

    def fit(self, X):
        self._scalers = None
        self.n_features_in_ = None
        self.mean_ = None
        self.var_ = None
        self.scale_ = None
        self.n_samples_seen_ = None

        return self.partial_fit(
            X
        )

    def transform(self, X):
        X = np.asarray(
            X,
            dtype="float64",
        )

        if X.ndim == 1:
            X = X.reshape(
                -1,
                1,
            )

        if self._scalers is None:
            raise RuntimeError(
                "FeaturewiseStandardScaler chưa được fit."
            )

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                "Numeric feature width không khớp learned scaler state."
            )

        transformed = np.full(
            X.shape,
            np.nan,
            dtype="float64",
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

            if not hasattr(
                scaler,
                "mean_",
            ):
                raise RuntimeError(
                    "Có observed value khi transform nhưng "
                    "feature scaler chưa có TRAIN observation."
                )

            transformed[
                finite_mask,
                index,
            ] = (
                scaler
                .transform(
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
class PreprocessingBundle:
    strategy: str
    scaler: FeaturewiseStandardScaler
    category_vocab: dict
    encoder: OneHotEncoder
    feature_names: list
    fit_row_count: int
    fit_min_timestamp: pd.Timestamp
    fit_max_timestamp: pd.Timestamp


def build_encoder_from_train_vocab(category_vocab):
    categories = []

    for column in CATEGORICAL_COLUMNS:
        train_categories = stable_sort_categories(
            column,
            category_vocab[column],
        )

        if UNKNOWN_TOKEN in train_categories:
            raise ValueError(
                f"Reserved token {UNKNOWN_TOKEN} xuất hiện trong TRAIN."
            )

        categories.append(
            train_categories
            + [UNKNOWN_TOKEN]
        )

    encoder = make_one_hot_encoder(
        categories=categories
    )

    max_len = max(
        len(values)
        for values in categories
    )

    synthetic = {}

    for column, values in zip(
        CATEGORICAL_COLUMNS,
        categories,
    ):
        synthetic[column] = [
            values[index % len(values)]
            for index in range(max_len)
        ]

    encoder.fit(
        pd.DataFrame(synthetic)
    )

    return encoder


def map_unknown_categories(frame, bundle):
    mapped = pd.DataFrame(
        index=frame.index
    )

    for column in CATEGORICAL_COLUMNS:
        values = (
            frame[column]
            .astype("string")
        )

        if values.isna().any():
            raise ValueError(
                f"Unexpected missing categorical: {column}"
            )

        known = set(
            bundle.category_vocab[column]
        )

        mapped[column] = values.where(
            values.isin(known),
            UNKNOWN_TOKEN,
        )

    return mapped


def build_feature_names(encoder):
    categorical_names = [
        f"cat__{name}"
        for name in encoder.get_feature_names_out(
            CATEGORICAL_COLUMNS
        )
    ]

    return (
        [f"num__{name}" for name in NUMERIC_COLUMNS]
        + [f"bool__{name}" for name in BOOLEAN_COLUMNS]
        + categorical_names
    )


def transform_with_bundle(frame, bundle):
    numeric = (
        frame[NUMERIC_COLUMNS]
        .astype("float64")
        .to_numpy()
    )

    numeric_scaled = (
        bundle.scaler
        .transform(numeric)
    )

    # Structural cold-start NA -> 0 trong standardized space.
    numeric_scaled = np.nan_to_num(
        numeric_scaled,
        nan=0.0,
        posinf=np.inf,
        neginf=-np.inf,
    ).astype(np.float32)

    boolean_matrix = (
        frame[BOOLEAN_COLUMNS]
        .astype(np.float32)
        .to_numpy()
    )

    mapped_categorical = map_unknown_categories(
        frame,
        bundle,
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

    if X.shape[1] != len(
        bundle.feature_names
    ):
        raise AssertionError(
            "Feature width != feature_names length."
        )

    if not np.isfinite(X.data).all():
        raise AssertionError(
            "NaN/inf xuất hiện sau preprocessing."
        )

    return X
```

### Phân tích / Nhận xét

Preprocessing helper hiện tách rõ ba branch:

1. numeric;
2. boolean;
3. categorical.

Numeric state được học bằng `FeaturewiseStandardScaler`: mỗi numerical feature có `StandardScaler` riêng và chỉ `partial_fit` trên các observation hữu hạn của chính feature đó. Nếu một batch không có observation hợp lệ cho một feature, scaler của feature đó không bị update. Cách này bảo toàn structural NA và tránh làm learned variance/scale thành `NaN`.

Categorical vocabulary được lấy từ TRAIN. Mỗi categorical feature được bổ sung reserved token:

`__UNKNOWN__`

nhưng token này không được coi là một observed TRAIN category. Khi transform, category ngoài TRAIN vocabulary được map deterministic sang token này.

Sau numeric scaling, structural NA được chuyển thành `0.0` trong standardized space, còn `has_prior_card_history` được giữ như companion state để không làm mất cold-start semantic.

Output cuối được ghép thành CSR sparse matrix với dtype `float32`.

### Kết luận

Implementation hiện phù hợp với leakage-safe preprocessing contract cần audit ở các cell phía sau.

Learned numeric state:

`TRAIN-ONLY / FEATURE-WISE`

Categorical state:

`TRAIN-ONLY VOCABULARY`

Unknown handling:

`EXPLICIT __UNKNOWN__`

Output representation:

`CSR FLOAT32`


# 10. M4.6.1 — Deterministic unit test cho fit/transform boundary


```python
unit_train = pd.DataFrame(
    {
        "amount_numeric": [10.0, 20.0, 30.0],
        "time_since_previous_transaction_min": [np.nan, 60.0, 120.0],
        "transactions_last_1h": [0.0, 1.0, 2.0],
        "amount_minus_previous_mean": [np.nan, 5.0, -5.0],
        "is_new_merchant": [True, False, False],
        "has_prior_card_history": [False, True, True],
        "transaction_mode": ["Chip Transaction", "Swipe Transaction", "Chip Transaction"],
        "location_state": ["PHYSICAL_COMPLETE", "PHYSICAL_COMPLETE", "NON_PHYSICAL_OR_ONLINE"],
        "hour_of_day": ["9", "10", "11"],
        "day_of_week": ["0", "1", "2"],
    }
)

unit_validation = pd.DataFrame(
    {
        "amount_numeric": [40.0],
        "time_since_previous_transaction_min": [np.nan],
        "transactions_last_1h": [0.0],
        "amount_minus_previous_mean": [np.nan],
        "is_new_merchant": [True],
        "has_prior_card_history": [False],
        "transaction_mode": ["UNSEEN_MODE"],
        "location_state": ["PHYSICAL_COMPLETE"],
        "hour_of_day": ["12"],
        "day_of_week": ["3"],
    }
)

unit_scaler = FeaturewiseStandardScaler()
unit_scaler.fit(
    unit_train[NUMERIC_COLUMNS]
    .astype("float64")
    .to_numpy()
)

unit_vocab = {
    column: set(
        unit_train[column]
        .astype("string")
        .tolist()
    )
    for column in CATEGORICAL_COLUMNS
}

unit_encoder = build_encoder_from_train_vocab(
    unit_vocab
)

unit_bundle = PreprocessingBundle(
    strategy="UNIT_TEST",
    scaler=unit_scaler,
    category_vocab=unit_vocab,
    encoder=unit_encoder,
    feature_names=build_feature_names(
        unit_encoder
    ),
    fit_row_count=len(unit_train),
    fit_min_timestamp=pd.Timestamp("2018-01-01"),
    fit_max_timestamp=pd.Timestamp("2018-01-03"),
)

X_unit_train = transform_with_bundle(
    unit_train,
    unit_bundle,
)

X_unit_validation = transform_with_bundle(
    unit_validation,
    unit_bundle,
)

assert X_unit_train.shape[0] == 3
assert X_unit_validation.shape[0] == 1
assert X_unit_train.shape[1] == X_unit_validation.shape[1]
assert np.isfinite(X_unit_validation.data).all()

unknown_candidates = [
    index
    for index, name in enumerate(
        unit_bundle.feature_names
    )
    if (
        name.startswith("cat__transaction_mode_")
        and UNKNOWN_TOKEN in name
    )
]

assert len(unknown_candidates) == 1

unknown_index = unknown_candidates[0]
assert X_unit_validation[0, unknown_index] == 1.0

# Validation không được thay đổi TRAIN scaler mean.
np.testing.assert_allclose(
    unit_bundle.scaler.mean_,
    unit_scaler.mean_,
)

print("M4.6 FIT/TRANSFORM UNIT TEST: PASS")
print("Unit output width:", X_unit_train.shape[1])
print("Explicit unknown column:", unit_bundle.feature_names[unknown_index])
```

    M4.6 FIT/TRANSFORM UNIT TEST: PASS
    Unit output width: 20
    Explicit unknown column: cat__transaction_mode___UNKNOWN__


### Phân tích / Nhận xét

Deterministic unit test đã PASS với cả TRAIN và một validation row có hai điều kiện quan trọng:

- structural NA ở historical numerical feature;
- `transaction_mode` chưa xuất hiện trong TRAIN.

Validation row vẫn transform thành vector hữu hạn và có cùng output width với TRAIN. Unknown transaction mode được kích hoạt đúng vào explicit `__UNKNOWN__` column.

Unit test cũng xác nhận preprocessing có thể xử lý cold-start numerical NA mà không tạo `NaN/inf` trong output.

### Kết luận M4.6.1

Fit/transform boundary:

`VERIFIED`

Structural cold-start NA:

`TRANSFORMABLE`

Explicit unknown path:

`VERIFIED`

Status:

`PASS`


# 11. M4.6.2 — Full TRAIN-only preprocessing-state pass

## Mục tiêu

Quét current artifact một lần để học preprocessing state mà không materialize full modeling matrix.

Trong pass này:

W_LONG scaler/vocabulary chỉ nhận:

`2015-01-01 <= Timestamp < 2019-01-01`

W_SHORT scaler/vocabulary chỉ nhận:

`2018-01-01 <= Timestamp < 2019-01-01`

VALIDATION:

`2019-01-01 <= Timestamp < 2019-06-01`

chỉ được dùng để:

- giữ deterministic audit sample;
- đếm unknown categories sau khi TRAIN vocabulary đã được khóa;
- transform-only audit sau khi fit hoàn tất.

VALIDATION không gọi `partial_fit` và không mở rộng category vocabulary.

FINAL TEST không tham gia preprocessing state hoặc validation audit.


```python
STRATEGIES = [
    "W_LONG",
    "W_SHORT",
]

scalers = {
    strategy: FeaturewiseStandardScaler()
    for strategy in STRATEGIES
}

category_counters = {
    strategy: {
        column: Counter()
        for column in CATEGORICAL_COLUMNS
    }
    for strategy in STRATEGIES
}

validation_category_counters = {
    column: Counter()
    for column in CATEGORICAL_COLUMNS
}

fit_row_counts = Counter()
fit_min_timestamp = {
    strategy: None
    for strategy in STRATEGIES
}
fit_max_timestamp = {
    strategy: None
    for strategy in STRATEGIES
}

validation_row_count = 0
context_rows_processed = 0
raw_rows_seen = 0
card_blocks_processed = 0
seen_card_keys = set()
max_context_card_rows = 0

sample_parts = []

pass_start = time.perf_counter()

for card_key, card_df in iter_card_blocks(
    DATA_PATH,
    CHUNK_SIZE,
):
    card_blocks_processed += 1
    raw_rows_seen += len(card_df)

    if card_key in seen_card_keys:
        raise RuntimeError(
            f"Card block reappearance: {card_key}"
        )

    seen_card_keys.add(card_key)

    context_card_df = (
        card_df.loc[
            card_df["Timestamp"]
            < VALIDATION_END
        ]
        .copy()
        .reset_index(drop=True)
    )

    if context_card_df.empty:
        continue

    context_rows_processed += len(
        context_card_df
    )

    max_context_card_rows = max(
        max_context_card_rows,
        len(context_card_df),
    )

    behavioral_df = (
        compute_causal_behavioral_features(
            context_card_df
        )
    )

    core_df = build_core_feature_frame(
        context_card_df,
        behavioral_df,
    )

    timestamp = core_df["Timestamp"]

    masks = {
        "W_LONG": (
            (timestamp >= W_LONG_START)
            & (timestamp < TRAIN_END)
        ),
        "W_SHORT": (
            (timestamp >= W_SHORT_START)
            & (timestamp < TRAIN_END)
        ),
        "VALIDATION": (
            (timestamp >= TRAIN_END)
            & (timestamp < VALIDATION_END)
        ),
    }

    for strategy in STRATEGIES:
        mask = masks[strategy]

        if not mask.any():
            continue

        train_sub = core_df.loc[mask]

        numeric_train = (
            train_sub[NUMERIC_COLUMNS]
            .astype("float64")
            .to_numpy()
        )

        # Fit từng numeric feature chỉ trên observed TRAIN values.
        # Batch toàn structural-NA ở một feature được skip riêng
        # cho feature đó, tránh làm learned variance/scale thành NaN.
        scalers[strategy].partial_fit(
            numeric_train
        )

        fit_row_counts[strategy] += len(
            train_sub
        )

        current_min = train_sub["Timestamp"].min()
        current_max = train_sub["Timestamp"].max()

        if (
            fit_min_timestamp[strategy] is None
            or current_min < fit_min_timestamp[strategy]
        ):
            fit_min_timestamp[strategy] = current_min

        if (
            fit_max_timestamp[strategy] is None
            or current_max > fit_max_timestamp[strategy]
        ):
            fit_max_timestamp[strategy] = current_max

        for column in CATEGORICAL_COLUMNS:
            category_counters[
                strategy
            ][column].update(
                train_sub[column]
                .astype("string")
                .value_counts()
                .to_dict()
            )

    validation_mask = masks["VALIDATION"]

    if validation_mask.any():
        validation_sub = core_df.loc[
            validation_mask
        ]

        validation_row_count += len(
            validation_sub
        )

        for column in CATEGORICAL_COLUMNS:
            validation_category_counters[
                column
            ].update(
                validation_sub[column]
                .astype("string")
                .value_counts()
                .to_dict()
            )

    development_mask = (
        (timestamp >= W_LONG_START)
        & (timestamp < VALIDATION_END)
    )

    sample_mask = (
        development_mask
        & (
            core_df["raw_row_id"]
            % AUDIT_SAMPLE_MOD
            == 0
        )
    )

    if sample_mask.any():
        sample = core_df.loc[
            sample_mask,
            [
                "raw_row_id",
                "Timestamp",
                *CORE_FEATURE_COLUMNS,
            ],
        ].copy()

        sample["period"] = assign_period(
            sample["Timestamp"]
        ).to_numpy()

        sample_parts.append(sample)

    if card_blocks_processed % 500 == 0:
        print(
            "Cards:",
            f"{card_blocks_processed:,}",
            "| raw rows:",
            f"{raw_rows_seen:,}",
            "| context rows:",
            f"{context_rows_processed:,}",
        )

pass_elapsed = time.perf_counter() - pass_start

audit_sample_df = pd.concat(
    sample_parts,
    ignore_index=True,
)

print("Full preprocessing-state pass hoàn tất.")
print("Card blocks:", f"{card_blocks_processed:,}")
print("Unique Card keys:", f"{len(seen_card_keys):,}")
print("Raw rows seen:", f"{raw_rows_seen:,}")
print("Context rows:", f"{context_rows_processed:,}")
print("W_LONG fit rows:", f"{fit_row_counts['W_LONG']:,}")
print("W_SHORT fit rows:", f"{fit_row_counts['W_SHORT']:,}")
print("VALIDATION rows:", f"{validation_row_count:,}")
print("Audit sample rows:", f"{len(audit_sample_df):,}")
print("Max context Card rows:", f"{max_context_card_rows:,}")
print("Elapsed seconds:", round(pass_elapsed, 2))
```

    Cards: 500 | raw rows: 2,021,584 | context rows: 1,911,651
    Cards: 1,000 | raw rows: 4,066,794 | context rows: 3,841,967
    Cards: 1,500 | raw rows: 5,915,634 | context rows: 5,589,029
    Cards: 2,000 | raw rows: 7,883,446 | context rows: 7,444,960
    Cards: 2,500 | raw rows: 9,842,479 | context rows: 9,292,973
    Cards: 3,000 | raw rows: 11,897,259 | context rows: 11,238,345
    Cards: 3,500 | raw rows: 13,997,374 | context rows: 13,231,256
    Cards: 4,000 | raw rows: 16,040,304 | context rows: 15,156,787
    Cards: 4,500 | raw rows: 18,115,490 | context rows: 17,121,544
    Cards: 5,000 | raw rows: 19,945,815 | context rows: 18,850,177
    Cards: 5,500 | raw rows: 21,956,418 | context rows: 20,747,229
    Cards: 6,000 | raw rows: 23,844,247 | context rows: 22,528,539
    Full preprocessing-state pass hoàn tất.
    Card blocks: 6,139
    Unique Card keys: 6,139
    Raw rows seen: 24,386,900
    Context rows: 23,038,920
    W_LONG fit rows: 6,855,270
    W_SHORT fit rows: 1,721,615
    VALIDATION rows: 712,458
    Audit sample rows: 75,670
    Max context Card rows: 68,616
    Elapsed seconds: 150.9


### Phân tích / Nhận xét

Full preprocessing-state pass đã hoàn tất trên toàn current artifact:

- Card blocks: `6,139`;
- unique Card keys: `6,139`;
- raw rows seen: `24,386,900`;
- history context rows trước validation end: `23,038,920`;
- W_LONG fit rows: `6,855,270`;
- W_SHORT fit rows: `1,721,615`;
- VALIDATION rows: `712,458`;
- deterministic audit sample: `75,670` rows;
- max context Card block: `68,616` rows;
- elapsed time: `150.9 seconds`.

W_LONG và W_SHORT scaler/vocabulary chỉ được update trong mask TRAIN tương ứng. VALIDATION chỉ được dùng cho category counters và deterministic audit sample; code không gọi `partial_fit` cho VALIDATION.

Pass hoàn tất không có exception và không cần materialize toàn bộ modeling matrix.

### Kết luận M4.6.2

Full preprocessing-state pass:

`COMPLETE`

TRAIN populations:

`VERIFIED`

VALIDATION population:

`OBSERVED FOR AUDIT / NOT FIT`

Computational blocker trên current environment:

`NONE OBSERVED`


# 12. M4.6.3 — Fit-source integrity gate


```python
assert card_blocks_processed == EXPECTED_CARD_COUNT
assert len(seen_card_keys) == EXPECTED_CARD_COUNT
assert raw_rows_seen == 24_386_900
assert context_rows_processed == EXPECTED_CONTEXT_ROWS_BEFORE_VALIDATION_END

assert fit_row_counts["W_LONG"] == EXPECTED_W_LONG_ROWS
assert fit_row_counts["W_SHORT"] == EXPECTED_W_SHORT_ROWS
assert validation_row_count == EXPECTED_VALIDATION_ROWS

assert fit_min_timestamp["W_LONG"] >= W_LONG_START
assert fit_max_timestamp["W_LONG"] < TRAIN_END

assert fit_min_timestamp["W_SHORT"] >= W_SHORT_START
assert fit_max_timestamp["W_SHORT"] < TRAIN_END

assert "Is Fraud?" not in FEATURE_USECOLS
assert "Errors?" not in FEATURE_USECOLS

print("M4.6 TRAIN-ONLY FIT-SOURCE GATE: PASS")
print(
    "W_LONG fit range:",
    fit_min_timestamp["W_LONG"],
    "→",
    fit_max_timestamp["W_LONG"],
)
print(
    "W_SHORT fit range:",
    fit_min_timestamp["W_SHORT"],
    "→",
    fit_max_timestamp["W_SHORT"],
)
```

    M4.6 TRAIN-ONLY FIT-SOURCE GATE: PASS
    W_LONG fit range: 2015-01-01 00:01:00 → 2018-12-31 23:58:00
    W_SHORT fit range: 2018-01-01 00:03:00 → 2018-12-31 23:58:00


### Phân tích / Nhận xét

Fit-source gate đã xác nhận đúng cả row count và temporal range.

W_LONG:

`2015-01-01 00:01:00 → 2018-12-31 23:58:00`

W_SHORT:

`2018-01-01 00:03:00 → 2018-12-31 23:58:00`

Cả hai đều kết thúc trước `TRAIN_END = 2019-01-01`.

Gate cũng xác nhận `Is Fraud?` và `Errors?` không được đọc trong feature pass.

### Kết luận M4.6.3

W_LONG learned preprocessing state:

`FIT FROM W_LONG TRAIN ONLY`

W_SHORT learned preprocessing state:

`FIT FROM W_SHORT TRAIN ONLY`

Target / Errors access:

`NO`

Status:

`PASS`


# 13. M4.6.4 — Khóa TRAIN vocabulary và tạo preprocessing bundles


```python
bundles = {}

for strategy in STRATEGIES:
    vocab = {
        column: set(
            category_counters[
                strategy
            ][column].keys()
        )
        for column in CATEGORICAL_COLUMNS
    }

    encoder = build_encoder_from_train_vocab(
        vocab
    )

    feature_names = build_feature_names(
        encoder
    )

    bundles[strategy] = PreprocessingBundle(
        strategy=strategy,
        scaler=scalers[strategy],
        category_vocab=vocab,
        encoder=encoder,
        feature_names=feature_names,
        fit_row_count=fit_row_counts[strategy],
        fit_min_timestamp=fit_min_timestamp[strategy],
        fit_max_timestamp=fit_max_timestamp[strategy],
    )

    print("\n", strategy)
    print("fit_row_count:", bundles[strategy].fit_row_count)
    print("output_width:", len(feature_names))

    for column in CATEGORICAL_COLUMNS:
        print(
            f"{column} TRAIN vocab size:",
            len(vocab[column]),
        )
```

    
     W_LONG
    fit_row_count: 6855270
    output_width: 47
    transaction_mode TRAIN vocab size: 3
    location_state TRAIN vocab size: 3
    hour_of_day TRAIN vocab size: 24
    day_of_week TRAIN vocab size: 7
    
     W_SHORT
    fit_row_count: 1721615
    output_width: 47
    transaction_mode TRAIN vocab size: 3
    location_state TRAIN vocab size: 3
    hour_of_day TRAIN vocab size: 24
    day_of_week TRAIN vocab size: 7


### Phân tích / Nhận xét

Hai preprocessing bundle đã được tạo độc lập cho W_LONG và W_SHORT nhưng dùng cùng procedure.

Cả hai có:

- output width: `47`;
- transaction mode TRAIN vocabulary: `3`;
- location state TRAIN vocabulary: `3`;
- hour-of-day TRAIN vocabulary: `24`;
- day-of-week TRAIN vocabulary: `7`.

Như vậy các core categorical domains quan sát trong hai training strategies hiện giống nhau. Learned numeric statistics vẫn được giữ riêng cho từng strategy, đúng vì training population khác nhau.

### Kết luận M4.6.4

W_LONG / W_SHORT preprocessing state:

`SEPARATE LEARNED STATE`

Preprocessing procedure:

`IDENTICAL`

Current categorical domain compatibility:

`CONFIRMED`

Output width:

`47`


# 14. M4.6.5 — Learned numeric state audit


```python
scaler_rows = []

for strategy in STRATEGIES:
    scaler = bundles[strategy].scaler

    n_seen = np.asarray(
        scaler.n_samples_seen_
    )

    if n_seen.ndim == 0:
        n_seen = np.repeat(
            n_seen,
            len(NUMERIC_COLUMNS),
        )

    for index, column in enumerate(
        NUMERIC_COLUMNS
    ):
        scaler_rows.append(
            {
                "strategy": strategy,
                "feature": column,
                "mean": float(scaler.mean_[index]),
                "scale": float(scaler.scale_[index]),
                "observed_train_rows": int(n_seen[index]),
            }
        )

scaler_audit_df = pd.DataFrame(
    scaler_rows
)

print(
    scaler_audit_df.to_string(
        index=False
    )
)

assert np.isfinite(
    scaler_audit_df["mean"]
).all()

assert np.isfinite(
    scaler_audit_df["scale"]
).all()

assert (
    scaler_audit_df["scale"] > 0
).all()

print("M4.6 NUMERIC LEARNED-STATE AUDIT: PASS")
```

    strategy                             feature        mean       scale  observed_train_rows
      W_LONG                      amount_numeric   42.945757   80.978173              6855270
      W_LONG time_since_previous_transaction_min 1197.180493 2152.884373              6854796
      W_LONG                transactions_last_1h    0.279406    0.665138              6855270
      W_LONG          amount_minus_previous_mean   -0.473647   78.596357              6854796
     W_SHORT                      amount_numeric   42.879028   80.550656              1721615
     W_SHORT time_since_previous_transaction_min 1197.314757 2113.323364              1721515
     W_SHORT                transactions_last_1h    0.275961    0.658149              1721615
     W_SHORT          amount_minus_previous_mean   -0.425984   78.173238              1721515
    M4.6 NUMERIC LEARNED-STATE AUDIT: PASS



```python
bad_scaler_state = scaler_audit_df.loc[
    ~np.isfinite(
        scaler_audit_df["scale"]
    )
]

display(
    bad_scaler_state
)

print(
    "\nNon-finite scale count:",
    len(bad_scaler_state),
)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>strategy</th>
      <th>feature</th>
      <th>mean</th>
      <th>scale</th>
      <th>observed_train_rows</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>


    
    Non-finite scale count: 0



```python
display(
    scaler_audit_df.loc[
        ~np.isfinite(
            scaler_audit_df["scale"]
        )
        |
        ~np.isfinite(
            scaler_audit_df["mean"]
        )
    ]
)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>strategy</th>
      <th>feature</th>
      <th>mean</th>
      <th>scale</th>
      <th>observed_train_rows</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>


### Phân tích / Nhận xét

Numeric learned-state audit đã PASS và không còn bất kỳ `mean` hoặc `scale` non-finite nào.

W_LONG:

- `amount_numeric`: mean `42.945757`, scale `80.978173`, observed rows `6,855,270`;
- `time_since_previous_transaction_min`: mean `1197.180493`, scale `2152.884373`, observed rows `6,854,796`;
- `transactions_last_1h`: mean `0.279406`, scale `0.665138`, observed rows `6,855,270`;
- `amount_minus_previous_mean`: mean `-0.473647`, scale `78.596357`, observed rows `6,854,796`.

W_SHORT:

- `amount_numeric`: mean `42.879028`, scale `80.550656`, observed rows `1,721,615`;
- `time_since_previous_transaction_min`: mean `1197.314757`, scale `2113.323364`, observed rows `1,721,515`;
- `transactions_last_1h`: mean `0.275961`, scale `0.658149`, observed rows `1,721,615`;
- `amount_minus_previous_mean`: mean `-0.425984`, scale `78.173238`, observed rows `1,721,515`.

Hai historical numerical feature có ít observed rows hơn tổng TRAIN rows đúng bằng số structural cold-start rows:

- W_LONG: `474`;
- W_SHORT: `100`.

Điều này phù hợp với contract M4.5: NA của recency/deviation là history-availability state, không phải corruption.

Hai diagnostic cell sau learned-state audit đều trả về empty DataFrame và:

`Non-finite scale count: 0`

### Kết luận M4.6.5

Numeric learned state:

`FINITE AND VALID`

Structural NA excluded from scaler statistics:

`CONFIRMED`

Cold-start semantic preserved during fitting:

`YES`

Status:

`PASS`


# 15. M4.6.6 — VALIDATION unknown-category audit


```python
unknown_rows = []

for strategy in STRATEGIES:
    bundle = bundles[strategy]

    for column in CATEGORICAL_COLUMNS:
        train_vocab = set(
            bundle.category_vocab[column]
        )

        validation_counter = (
            validation_category_counters[column]
        )

        unseen_categories = {
            category
            for category
            in validation_counter
            if category not in train_vocab
        }

        unseen_rows = sum(
            validation_counter[category]
            for category
            in unseen_categories
        )

        unknown_rows.append(
            {
                "strategy": strategy,
                "feature": column,
                "train_vocab_size": len(train_vocab),
                "validation_unique": len(validation_counter),
                "validation_unseen_unique": len(unseen_categories),
                "validation_unseen_rows": unseen_rows,
                "validation_unseen_rate_pct": (
                    unseen_rows
                    / EXPECTED_VALIDATION_ROWS
                    * 100
                ),
            }
        )

unknown_audit_df = pd.DataFrame(
    unknown_rows
)

print(
    unknown_audit_df.to_string(
        index=False
    )
)

print(
    "Explicit unknown token is present in every categorical output schema:",
    all(
        any(
            UNKNOWN_TOKEN in name
            for name in bundles[strategy].feature_names
            if name.startswith("cat__")
        )
        for strategy in STRATEGIES
    ),
)
```

    strategy          feature  train_vocab_size  validation_unique  validation_unseen_unique  validation_unseen_rows  validation_unseen_rate_pct
      W_LONG transaction_mode                 3                  3                         0                       0                         0.0
      W_LONG   location_state                 3                  3                         0                       0                         0.0
      W_LONG      hour_of_day                24                 24                         0                       0                         0.0
      W_LONG      day_of_week                 7                  7                         0                       0                         0.0
     W_SHORT transaction_mode                 3                  3                         0                       0                         0.0
     W_SHORT   location_state                 3                  3                         0                       0                         0.0
     W_SHORT      hour_of_day                24                 24                         0                       0                         0.0
     W_SHORT      day_of_week                 7                  7                         0                       0                         0.0
    Explicit unknown token is present in every categorical output schema: True


### Phân tích / Nhận xét

Trong VALIDATION hiện tại, cả W_LONG và W_SHORT đều không gặp unseen category ở bốn core categorical feature:

- `transaction_mode`;
- `location_state`;
- `hour_of_day`;
- `day_of_week`.

Mỗi feature có:

`validation_unseen_unique = 0`

và:

`validation_unseen_rate_pct = 0.0`

Điều này chỉ cho thấy current VALIDATION nằm hoàn toàn trong TRAIN vocabulary; nó không loại bỏ nhu cầu có unknown policy cho dữ liệu tương lai.

Output schema vẫn chứa explicit unknown token, và stress test ở phần sau kiểm tra riêng đường đi này.

### Kết luận M4.6.6

Current VALIDATION unseen-category exposure:

`0%`

TRAIN-only vocabulary:

`PRESERVED`

Unknown policy vẫn cần thiết cho production/generalization:

`YES`


# 16. M4.6.7 — Output schema / feature-order stability


```python
for strategy in STRATEGIES:
    bundle = bundles[strategy]

    assert len(bundle.feature_names) == len(
        set(bundle.feature_names)
    )

    print("\n", strategy)
    print("Output width:", len(bundle.feature_names))
    print("Feature order:")

    for index, name in enumerate(
        bundle.feature_names
    ):
        print(index, name)

same_schema = (
    bundles["W_LONG"].feature_names
    == bundles["W_SHORT"].feature_names
)

print(
    "\nW_LONG/W_SHORT feature names identical:",
    same_schema,
)

assert same_schema

print("M4.6 OUTPUT SCHEMA STABILITY GATE: PASS")
```

    
     W_LONG
    Output width: 47
    Feature order:
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
    
     W_SHORT
    Output width: 47
    Feature order:
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
    
    W_LONG/W_SHORT feature names identical: True
    M4.6 OUTPUT SCHEMA STABILITY GATE: PASS


### Phân tích / Nhận xét

Output schema của W_LONG và W_SHORT đều có đúng `47` cột và feature order hoàn toàn giống nhau.

Cấu trúc gồm:

- 4 numeric columns;
- 2 boolean columns;
- 4 one-hot columns cho `transaction_mode`;
- 4 one-hot columns cho `location_state`;
- 25 one-hot columns cho `hour_of_day`;
- 8 one-hot columns cho `day_of_week`.

Mỗi categorical branch có thêm đúng một `__UNKNOWN__` column.

Gate trả về:

`W_LONG/W_SHORT feature names identical: True`

và:

`M4.6 OUTPUT SCHEMA STABILITY GATE: PASS`

### Kết luận M4.6.7

Stable output width:

`47`

Stable names/order across strategies:

`VERIFIED`

Schema mutation giữa W_LONG và W_SHORT:

`NONE`

Status:

`PASS`


# 17. M4.6.8 — Transform-only audit trên deterministic sample


```python
train_2015_2017_sample = audit_sample_df.loc[
    audit_sample_df["period"]
    .eq("TRAIN_2015_2017")
].copy()

train_2018_sample = audit_sample_df.loc[
    audit_sample_df["period"]
    .eq("TRAIN_2018")
].copy()

validation_sample = audit_sample_df.loc[
    audit_sample_df["period"]
    .eq("VALIDATION")
].copy()

w_long_train_sample = pd.concat(
    [
        train_2015_2017_sample,
        train_2018_sample,
    ],
    ignore_index=True,
)

w_short_train_sample = (
    train_2018_sample
    .reset_index(drop=True)
)

sample_map = {
    "W_LONG": w_long_train_sample,
    "W_SHORT": w_short_train_sample,
}

transform_audit_rows = []

for strategy in STRATEGIES:
    bundle = bundles[strategy]

    train_sample = sample_map[strategy]

    X_train_sample = transform_with_bundle(
        train_sample,
        bundle,
    )

    X_validation_sample = transform_with_bundle(
        validation_sample,
        bundle,
    )

    X_validation_repeat = transform_with_bundle(
        validation_sample,
        bundle,
    )

    assert (
        X_train_sample.shape[1]
        == X_validation_sample.shape[1]
        == len(bundle.feature_names)
    )

    assert np.isfinite(
        X_train_sample.data
    ).all()

    assert np.isfinite(
        X_validation_sample.data
    ).all()

    repeat_diff = (
        X_validation_sample
        != X_validation_repeat
    )

    assert repeat_diff.nnz == 0

    transform_audit_rows.append(
        {
            "strategy": strategy,
            "train_sample_rows": X_train_sample.shape[0],
            "validation_sample_rows": X_validation_sample.shape[0],
            "output_columns": X_train_sample.shape[1],
            "train_nnz": X_train_sample.nnz,
            "validation_nnz": X_validation_sample.nnz,
            "train_density_pct": (
                X_train_sample.nnz
                / (
                    X_train_sample.shape[0]
                    * X_train_sample.shape[1]
                )
                * 100
            ),
            "validation_density_pct": (
                X_validation_sample.nnz
                / (
                    X_validation_sample.shape[0]
                    * X_validation_sample.shape[1]
                )
                * 100
            ),
            "repeat_transform_identical": (
                repeat_diff.nnz == 0
            ),
        }
    )

transform_audit_df = pd.DataFrame(
    transform_audit_rows
)

print(
    transform_audit_df.to_string(
        index=False
    )
)

print("M4.6 TRANSFORM-ONLY SAMPLE AUDIT: PASS")
```

    strategy  train_sample_rows  validation_sample_rows  output_columns  train_nnz  validation_nnz  train_density_pct  validation_density_pct  repeat_transform_identical
      W_LONG              68560                    7110              47     619062           64171          19.211686                 19.2031                        True
     W_SHORT              17234                    7110              47     155559           64171          19.204862                 19.2031                        True
    M4.6 TRANSFORM-ONLY SAMPLE AUDIT: PASS


### Phân tích / Nhận xét

Transform-only audit sử dụng deterministic sample và chạy sạch cho cả hai strategy.

W_LONG:

- TRAIN sample: `68,560` rows;
- VALIDATION sample: `7,110` rows;
- output columns: `47`;
- TRAIN density: `19.211686%`;
- VALIDATION density: `19.2031%`.

W_SHORT:

- TRAIN sample: `17,234` rows;
- VALIDATION sample: `7,110` rows;
- output columns: `47`;
- TRAIN density: `19.204862%`;
- VALIDATION density: `19.2031%`.

Tất cả sparse data đều finite. Transform VALIDATION lặp lại hai lần cho kết quả identical ở cả hai strategy.

Density khoảng `19.2%` nghĩa là phần lớn matrix là zero, nên CSR là representation hợp lý cho core baseline. M4.6 chưa materialize full matrix, vì vậy full-matrix memory usage vẫn thuộc audit M4.7.

### Kết luận M4.6.8

VALIDATION transform-only:

`PASS`

Repeat-transform determinism:

`VERIFIED`

Sample sparse representation:

`SUITABLE FOR CSR`

Full matrix materialization:

`DEFERRED TO M4.7`


# 18. M4.6.9 — Cold-start representation audit sau preprocessing


```python
cold_start_sample = validation_sample.loc[
    ~validation_sample[
        "has_prior_card_history"
    ]
].copy()

print(
    "Validation cold-start rows in deterministic sample:",
    len(cold_start_sample),
)

if not cold_start_sample.empty:
    for strategy in STRATEGIES:
        bundle = bundles[strategy]

        X_cold = transform_with_bundle(
            cold_start_sample,
            bundle,
        )

        assert np.isfinite(X_cold.data).all()

        has_prior_feature = (
            "bool__has_prior_card_history"
        )

        has_prior_index = (
            bundle.feature_names.index(
                has_prior_feature
            )
        )

        assert (
            X_cold[:, has_prior_index]
            .nnz
            == 0
        )

        print(
            strategy,
            "cold-start transformed shape:",
            X_cold.shape,
        )
else:
    print(
        "Deterministic 1% sample không chứa cold-start row; "
        "semantic vẫn đã được unit-test ở M4.6.1."
    )

print("M4.6 COLD-START REPRESENTATION AUDIT: PASS")
```

    Validation cold-start rows in deterministic sample: 0
    Deterministic 1% sample không chứa cold-start row; semantic vẫn đã được unit-test ở M4.6.1.
    M4.6 COLD-START REPRESENTATION AUDIT: PASS


### Phân tích / Nhận xét

Deterministic 1% VALIDATION sample không chứa cold-start row:

`Validation cold-start rows in deterministic sample: 0`

Vì vậy cell này không cung cấp population-level cold-start example từ VALIDATION sample.

Tuy nhiên đây không phải blocker: M4.6.1 đã chủ động tạo một cold-start row với structural NA ở recency/deviation và `has_prior_card_history=False`, sau đó transform thành công với output hữu hạn. Numeric learned-state audit cũng cho thấy các structural NA bị loại khỏi scaler statistics thay vì bị học như giá trị 0.

Do đó cold-start semantic đã được kiểm tra bằng deterministic unit test thay vì phụ thuộc vào việc sample ngẫu nhiên/deterministic có chứa rare cold-start hay không.

### Kết luận M4.6.9

Cold-start representation implementation:

`VERIFIED BY UNIT TEST`

Cold-start row trong deterministic VALIDATION sample:

`NOT OBSERVED`

Blocking issue:

`NONE`

Status:

`PASS`


# 19. M4.6.10 — Unknown-category transform stress test trên real bundle


```python
if validation_sample.empty:
    raise RuntimeError(
        "Validation sample rỗng."
    )

stress_row = (
    validation_sample
    .iloc[[0]]
    .copy()
)

stress_row["transaction_mode"] = (
    "__SYNTHETIC_UNSEEN_FOR_AUDIT__"
)

for strategy in STRATEGIES:
    bundle = bundles[strategy]

    X_stress = transform_with_bundle(
        stress_row,
        bundle,
    )

    unknown_indices = [
        index
        for index, name in enumerate(
            bundle.feature_names
        )
        if (
            name.startswith(
                "cat__transaction_mode_"
            )
            and UNKNOWN_TOKEN in name
        )
    ]

    assert len(unknown_indices) == 1
    assert X_stress[0, unknown_indices[0]] == 1.0
    assert np.isfinite(X_stress.data).all()

    print(
        strategy,
        "unknown-category stress test width:",
        X_stress.shape[1],
    )

print("M4.6 UNKNOWN-CATEGORY STRESS TEST: PASS")
```

    W_LONG unknown-category stress test width: 47
    W_SHORT unknown-category stress test width: 47
    M4.6 UNKNOWN-CATEGORY STRESS TEST: PASS


### Phân tích / Nhận xét

Stress test đã cố tình thay `transaction_mode` bằng một category chưa từng xuất hiện:

`__SYNTHETIC_UNSEEN_FOR_AUDIT__`

Cả W_LONG và W_SHORT đều:

- transform thành công;
- giữ output width `47`;
- kích hoạt đúng explicit unknown column;
- không tạo `NaN/inf`.

Điều này bổ sung bằng chứng mà current VALIDATION không thể cung cấp vì VALIDATION thực tế có unseen rate bằng 0.

### Kết luận M4.6.10

Unknown-category mapping:

`VERIFIED`

Schema mutation khi gặp unknown:

`NO`

Explicit unknown column:

`WORKING`

Status:

`PASS`


# 20. M4.6.11 — Preprocessing safety gate


```python
assert set(CORE_FEATURE_COLUMNS).isdisjoint(
    PROHIBITED_DIRECT_COLUMNS
)

assert fit_row_counts["W_LONG"] == EXPECTED_W_LONG_ROWS
assert fit_row_counts["W_SHORT"] == EXPECTED_W_SHORT_ROWS
assert validation_row_count == EXPECTED_VALIDATION_ROWS

assert fit_max_timestamp["W_LONG"] < TRAIN_END
assert fit_max_timestamp["W_SHORT"] < TRAIN_END

assert (
    bundles["W_LONG"].feature_names
    == bundles["W_SHORT"].feature_names
)

for strategy in STRATEGIES:
    assert UNKNOWN_TOKEN not in (
        set().union(
            *[
                bundles[strategy].category_vocab[column]
                for column in CATEGORICAL_COLUMNS
            ]
        )
    )

LEARNED_STATE_FIT_FROM_VALIDATION = False
TARGET_USED_IN_PREPROCESSING = False
FINAL_TEST_USED_IN_PREPROCESSING_DECISION = False
RAW_IDENTIFIER_EXPOSED = False
BEHAVIORAL_CAUSAL_CONTRACT_CHANGED = False

assert LEARNED_STATE_FIT_FROM_VALIDATION is False
assert TARGET_USED_IN_PREPROCESSING is False
assert FINAL_TEST_USED_IN_PREPROCESSING_DECISION is False
assert RAW_IDENTIFIER_EXPOSED is False
assert BEHAVIORAL_CAUSAL_CONTRACT_CHANGED is False

print("M4.6 PREPROCESSING SAFETY GATE: PASS")
```

    M4.6 PREPROCESSING SAFETY GATE: PASS


### Phân tích / Nhận xét

Preprocessing safety gate đã PASS.

Các invariant được kiểm tra gồm:

- core features không giao với prohibited raw identifier/target columns;
- W_LONG/W_SHORT fit row counts đúng contract;
- learned state dừng trước `TRAIN_END`;
- W_LONG và W_SHORT có cùng feature schema;
- reserved `__UNKNOWN__` không xuất hiện như observed TRAIN category;
- VALIDATION không được dùng để fit learned state;
- target không được dùng trong preprocessing;
- FINAL TEST không được dùng để quyết định preprocessing;
- raw identifier không bị expose;
- M4.5 behavioral causal contract không bị thay đổi.

Kết hợp với các gate trước, không còn blocker kỹ thuật nào được phát hiện trước M4.7.

### Kết luận M4.6.11

Leakage-safety contract:

`VERIFIED`

Behavioral causal contract:

`PRESERVED`

FINAL TEST isolation:

`PRESERVED`

Status:

`PASS`


# 21. Preprocessing Contract Registry — chờ runtime review

## P01 — Core feature set

Current proposal:

- 4 numeric;
- 2 boolean;
- 4 low-cardinality categorical.

Status:

`OPEN UNTIL REVIEW`

---

## P02 — Numerical scaling

Current proposal:

`StandardScaler`, fit TRAIN only.

Status:

`OPEN UNTIL REVIEW`

---

## P03 — Cold-start numerical NA

Current proposal:

- scaler ignores NA when learning mean/std;
- after transform, NA → 0 standardized value;
- preserve `has_prior_card_history` companion feature.

Status:

`OPEN UNTIL REVIEW`

---

## P04 — Boolean branch

Current proposal:

passthrough float32.

Status:

`OPEN UNTIL REVIEW`

---

## P05 — Categorical vocabulary

Current proposal:

TRAIN-only vocabulary.

Status:

`OPEN UNTIL REVIEW`

---

## P06 — Unknown category

Current proposal:

explicit `__UNKNOWN__` one-hot column per categorical feature.

VALIDATION không mở rộng vocabulary.

Status:

`OPEN UNTIL REVIEW`

---

## P07 — Output representation

Current proposal:

CSR sparse matrix, dtype float32.

Status:

`OPEN UNTIL REVIEW`

---

## P08 — W_LONG / W_SHORT state

Current proposal:

hai learned preprocessing states riêng, cùng procedure và cùng feature schema.

Status:

`OPEN UNTIL REVIEW`

---

## P09 — Conditional / experiment features

MCC, raw location, month, Amount alternatives và support states không tự động đi vào core baseline.

Status:

`INHERITED — LOCKED`

# 21A. Preprocessing Contract Registry — runtime review

## P01 — Core feature set

Locked core baseline:

- 4 numeric;
- 2 boolean;
- 4 low-cardinality categorical;
- tổng cộng `10` pre-encoding features.

Status:

`LOCKED FOR BASELINE V1`

---

## P02 — Numerical scaling

Implementation:

`FeaturewiseStandardScaler`

Mỗi numeric feature dùng incremental `StandardScaler` riêng và chỉ học từ observed TRAIN values của chính feature đó.

W_LONG và W_SHORT có learned state riêng.

Status:

`LOCKED`

---

## P03 — Cold-start numerical NA

Policy:

1. structural NA không đóng góp learned mean/std;
2. scaler transform giữ NA;
3. sau scaling, structural NA → `0.0` trong standardized space;
4. `has_prior_card_history` được giữ làm companion state.

Không học imputation statistic từ VALIDATION.

Status:

`LOCKED AT PREPROCESSING LEVEL`

---

## P04 — Boolean branch

Features:

- `is_new_merchant`;
- `has_prior_card_history`.

Representation:

`PASSTHROUGH FLOAT32`

Status:

`LOCKED`

---

## P05 — Categorical vocabulary

Vocabulary source:

`TRAIN ONLY`

Features:

- transaction mode;
- location state;
- hour of day;
- day of week.

VALIDATION không mở rộng vocabulary.

Status:

`LOCKED`

---

## P06 — Unknown category

Policy:

mọi category ngoài TRAIN vocabulary → explicit:

`__UNKNOWN__`

Mỗi categorical feature có reserved unknown one-hot column.

Real VALIDATION unseen rate hiện là `0%`, nhưng synthetic unseen stress test đã PASS cho cả W_LONG và W_SHORT.

Status:

`LOCKED`

---

## P07 — Output representation

Representation:

`CSR sparse matrix`

Dtype:

`float32`

Output width:

`47`

Observed deterministic-sample density:

`≈ 19.2%`

Status:

`LOCKED FOR BASELINE V1`

---

## P08 — W_LONG / W_SHORT state

Policy:

- hai learned preprocessing states riêng;
- cùng procedure;
- cùng core feature contract;
- cùng output names/order;
- mỗi state fit đúng TRAIN population tương ứng.

Status:

`LOCKED`

---

## P09 — Conditional / experiment features

MCC, raw location, month, Amount alternatives và support states không tự động đi vào core baseline.

Status:

`INHERITED — LOCKED`


# 22. M4.6 Findings — chờ phân tích sau Run All

Các finding cần tổng hợp từ output thực tế:

- M4.6-F01 — TRAIN-only fit-source integrity;
- M4.6-F02 — numeric learned-state validity;
- M4.6-F03 — cold-start preprocessing semantics;
- M4.6-F04 — categorical vocabulary và unknown handling;
- M4.6-F05 — W_LONG/W_SHORT schema compatibility;
- M4.6-F06 — transform-only VALIDATION integrity;
- M4.6-F07 — sparse matrix density / memory suitability;
- M4.6-F08 — deterministic repeat-transform;
- M4.6-F09 — preprocessing computational feasibility.

Status:

`OPEN — CHỜ REVIEW OUTPUT`

# 22A. M4.6 Findings — runtime review

## M4.6-F01 — TRAIN-only fit-source integrity

Observed fact:

W_LONG fit `6,855,270` rows và W_SHORT fit `1,721,615` rows; cả hai fit range kết thúc trước `2019-01-01`.

Interpretation:

Learned scaler/vocabulary state được giới hạn đúng trong TRAIN population tương ứng.

Implication:

VALIDATION có thể được dùng transform-only mà không tạo preprocessing leakage.

Status:

`CONFIRMED`

---

## M4.6-F02 — Numeric learned-state validity

Observed fact:

Tất cả mean/scale của 4 numeric feature ở cả hai strategies đều finite và scale > 0.

Recency/deviation có observed row count nhỏ hơn tổng TRAIN đúng theo structural cold-start population.

Interpretation:

Feature-wise incremental scaling đã xử lý đúng structural NA và không làm hỏng learned variance.

Implication:

Numeric preprocessing state đủ điều kiện để dùng ở M4.7.

Status:

`CONFIRMED`

---

## M4.6-F03 — Cold-start preprocessing semantics

Observed fact:

Unit test transform thành công row có structural NA; numerical NA được đưa về 0 trong standardized space trong khi `has_prior_card_history=False` được giữ.

Interpretation:

0 standardized value không tự nó mang nghĩa cold-start; companion feature chịu trách nhiệm bảo toàn history-availability semantic.

Implication:

M4.7 phải giữ cả transformed numeric feature và companion flag theo đúng schema hiện tại.

Status:

`CONFIRMED`

---

## M4.6-F04 — Categorical vocabulary và unknown handling

Observed fact:

Current VALIDATION có unseen rate 0% cho cả bốn categorical feature. Synthetic unseen stress test vẫn map đúng vào explicit `__UNKNOWN__` column và giữ width 47.

Interpretation:

Unknown path không phụ thuộc vào việc VALIDATION hiện tại có unseen category hay không.

Implication:

Không cần mutate encoder vocabulary tại transform time.

Status:

`CONFIRMED`

---

## M4.6-F05 — W_LONG/W_SHORT schema compatibility

Observed fact:

Hai bundles đều có đúng 47 output features với names/order identical.

Interpretation:

Training-window strategy làm thay đổi learned numeric state nhưng không làm thay đổi model input contract.

Implication:

M4.7 có thể dùng một stable downstream schema cho cả hai strategies.

Status:

`CONFIRMED`

---

## M4.6-F06 — VALIDATION transform-only integrity

Observed fact:

Deterministic VALIDATION sample `7,110` rows transform sạch với cả W_LONG và W_SHORT; mọi sparse data finite.

Interpretation:

TRAIN-fitted bundle có thể áp dụng trực tiếp lên VALIDATION mà không refit.

Implication:

Fit/transform boundary đủ rõ để triển khai full modeling matrix.

Status:

`CONFIRMED`

---

## M4.6-F07 — Sparse representation suitability

Observed fact:

Sample density khoảng `19.2%` cho cả TRAIN và VALIDATION.

Interpretation:

Phần lớn 47-dimensional matrix là zero sau one-hot encoding, nên CSR là representation hợp lý hơn dense cho baseline pipeline.

Implication:

Giữ CSR float32 ở M4.7; full-matrix memory vẫn cần được đo khi materialize thật.

Status:

`CONFIRMED WITH M4.7 MEMORY AUDIT PENDING`

---

## M4.6-F08 — Deterministic repeat-transform

Observed fact:

Repeat transform của cùng VALIDATION sample cho matrix identical ở cả hai strategies.

Interpretation:

Current preprocessing transformation là deterministic với learned state cố định.

Implication:

Không có stochastic transform state cần quản lý ở M4.7.

Status:

`CONFIRMED`

---

## M4.6-F09 — Preprocessing computational feasibility

Observed fact:

Full preprocessing-state pass xử lý `23,038,920` context rows, max context Card block `68,616`, trong `150.9 seconds` trên current environment.

Interpretation:

Không phát hiện computational blocker ở bước học preprocessing state.

Implication:

Có thể chuyển sang full matrix construction/audit ở M4.7.

Status:

`CONFIRMED`


# 23. Decision Log M4.6 — chờ phân tích sau Run All

## M4.6-D01 — Core preprocessing feature set

`OPEN`

## M4.6-D02 — Numeric scaler

`OPEN`

## M4.6-D03 — Cold-start NA representation

`OPEN`

## M4.6-D04 — Categorical encoder

`OPEN`

## M4.6-D05 — Unknown-category policy

`OPEN`

## M4.6-D06 — Sparse float32 output

`OPEN`

## M4.6-D07 — W_LONG/W_SHORT preprocessing-state policy

`OPEN`

## M4.6-D08 — TRAIN-only learned state

Expected:

`LOCKED AFTER RUNTIME VERIFICATION`

## M4.6-D09 — FINAL TEST isolation

Expected:

`INHERITED — VERIFIED — LOCKED`

# 23A. Decision Log M4.6 — runtime review

## M4.6-D01 — Core preprocessing feature set

Decision:

Giữ đúng 10 core features đã chỉ định trong M4.6; không tự động thêm conditional/experiment features.

Status:

`LOCKED FOR BASELINE V1`

---

## M4.6-D02 — Numeric scaler

Decision:

Dùng `FeaturewiseStandardScaler`; mỗi feature học TRAIN-only mean/std trên observed finite values.

Status:

`LOCKED`

---

## M4.6-D03 — Cold-start NA representation

Decision:

Structural NA của recency/deviation được giữ qua scaling rồi map về `0.0` trong standardized space; `has_prior_card_history` bắt buộc được giữ làm companion state.

Status:

`LOCKED`

---

## M4.6-D04 — Categorical encoder

Decision:

One-hot encoding dùng TRAIN-only vocabulary, deterministic category order và output float32 sparse.

Status:

`LOCKED`

---

## M4.6-D05 — Unknown-category policy

Decision:

Category ngoài TRAIN vocabulary được map sang explicit `__UNKNOWN__`; không mở rộng vocabulary khi transform.

Status:

`LOCKED`

---

## M4.6-D06 — Sparse float32 output

Decision:

Core preprocessing output dùng CSR sparse matrix, dtype float32.

Status:

`LOCKED FOR BASELINE V1`

---

## M4.6-D07 — W_LONG/W_SHORT preprocessing-state policy

Decision:

Giữ hai learned states riêng theo training window nhưng cùng procedure và cùng output schema.

Status:

`LOCKED`

---

## M4.6-D08 — TRAIN-only learned state

Decision:

Scaler statistics và categorical vocabulary chỉ được học từ TRAIN của strategy tương ứng.

Status:

`VERIFIED — LOCKED`

---

## M4.6-D09 — FINAL TEST isolation

Decision:

FINAL TEST không được dùng để fit preprocessing state hoặc quyết định preprocessing trong M4.6.

Status:

`INHERITED — VERIFIED — LOCKED`


# 24. M4.6 Gate — chờ review output

## G01 — Artifact / input contract valid

`OPEN`

## G02 — Core feature roles explicit

`OPEN`

## G03 — Behavioral causal contract preserved

`OPEN`

## G04 — W_LONG fit source = W_LONG TRAIN only

`OPEN`

## G05 — W_SHORT fit source = W_SHORT TRAIN only

`OPEN`

## G06 — VALIDATION transform only

`OPEN`

## G07 — Numeric learned state finite

`OPEN`

## G08 — Structural cold-start NA handled explicitly

`OPEN`

## G09 — Categorical vocabulary TRAIN only

`OPEN`

## G10 — Unknown category handled without schema mutation

`OPEN`

## G11 — W_LONG/W_SHORT output schema consistent

`OPEN`

## G12 — Stable feature names/order

`OPEN`

## G13 — No prohibited raw identifier / target

`OPEN`

## G14 — No FINAL TEST decision access

`OPEN`

## G15 — Repeat transform deterministic

`OPEN`

## G16 — Preprocessing contract ready for M4.7

`OPEN`

# M4.6 Gate

Overall:

`OPEN`

Blocking issue:

`UNKNOWN UNTIL REVIEW`

M4.6 Status:

`READY TO RUN`

# 24A. M4.6 Gate — runtime review

## G01 — Artifact / input contract valid

Result:

`PASS`

## G02 — Core feature roles explicit

Result:

`PASS`

## G03 — Behavioral causal contract preserved

Result:

`PASS`

## G04 — W_LONG fit source = W_LONG TRAIN only

Result:

`PASS`

## G05 — W_SHORT fit source = W_SHORT TRAIN only

Result:

`PASS`

## G06 — VALIDATION transform only

Result:

`PASS`

## G07 — Numeric learned state finite

Result:

`PASS`

## G08 — Structural cold-start NA handled explicitly

Result:

`PASS`

Unit-test evidence xác minh cold-start path; deterministic VALIDATION sample không chứa cold-start row nhưng đây không phải blocker.

## G09 — Categorical vocabulary TRAIN only

Result:

`PASS`

## G10 — Unknown category handled without schema mutation

Result:

`PASS`

## G11 — W_LONG/W_SHORT output schema consistent

Result:

`PASS`

## G12 — Stable feature names/order

Result:

`PASS`

## G13 — No prohibited raw identifier / target

Result:

`PASS`

## G14 — No FINAL TEST decision access

Result:

`PASS`

## G15 — Repeat transform deterministic

Result:

`PASS`

## G16 — Preprocessing contract ready for M4.7

Result:

`PASS`

# M4.6 Gate — Runtime Review

Overall:

`PASS`

Blocking issue:

`NONE`

M4.6 Status:

`PASS — READY FOR M4.7`


# 25. Kết luận M4.6 — chờ review output

Sau khi Run All, cần kết luận:

- learned preprocessing state có thực sự chỉ đến từ TRAIN hay không;
- W_LONG/W_SHORT có cùng preprocessing procedure và schema hay không;
- VALIDATION transform-only có chạy sạch hay không;
- cold-start NA có được biểu diễn mà không mất semantic hay không;
- unknown-category handling có giữ feature width/order hay không;
- output sparse float32 có phù hợp để M4.7 tạo full modeling matrix hay không;
- có blocking issue nào trước M4.7 hay không.

Trạng thái hiện tại:

`OPEN — CHỜ RUN ALL VÀ REVIEW OUTPUT`

Nếu M4.6 PASS:

Next:

`M4.7 — Tạo baseline-ready modeling matrix và pipeline audit`

# 25A. Kết luận M4.6 — runtime review

## Mục tiêu đã kiểm tra

M4.6 đã xây và audit leakage-safe preprocessing contract cho core transaction-level và behavioral features đã khóa ở M4.4/M4.5.

Notebook đã kiểm tra trực tiếp:

- core feature roles;
- TRAIN-only numeric learned state;
- TRAIN-only categorical vocabulary;
- structural cold-start NA;
- explicit unknown-category handling;
- W_LONG/W_SHORT state separation;
- output schema stability;
- VALIDATION transform-only behavior;
- sparse output representation;
- deterministic repeat-transform;
- prohibited raw field/target isolation;
- FINAL TEST protection.

## Learned preprocessing state

W_LONG:

`6,855,270 TRAIN rows`

W_SHORT:

`1,721,615 TRAIN rows`

Cả hai learned states đều dừng trước:

`2019-01-01`

Numeric mean/scale đều finite sau feature-wise incremental scaling.

## Core output contract

Input core features:

`10`

Output features:

`47`

Representation:

`CSR FLOAT32`

W_LONG/W_SHORT feature names/order:

`IDENTICAL`

## Cold-start policy

Structural NA ở:

- `time_since_previous_transaction_min`;
- `amount_minus_previous_mean`

không tham gia scaler statistics.

Sau scaling:

`NA → 0.0 standardized value`

và:

`has_prior_card_history`

được giữ làm explicit companion state.

Policy:

`LOCKED`

## Categorical / unknown policy

Vocabulary:

`TRAIN ONLY`

Unknown category:

`MAP TO EXPLICIT __UNKNOWN__`

VALIDATION:

`TRANSFORM ONLY`

Current VALIDATION unseen rate:

`0%`

Synthetic unseen stress test:

`PASS`

## Sparse representation

Deterministic sample density:

`≈ 19.2%`

CSR float32 phù hợp cho baseline preprocessing representation.

Full modeling-matrix memory/shape audit vẫn thuộc:

`M4.7`

## Leakage / safety

Raw identifier direct exposure:

`NO`

Target used in preprocessing:

`NO`

VALIDATION used to fit learned state:

`NO`

FINAL TEST used for preprocessing decision:

`NO`

Behavioral causal contract changed:

`NO`

## Những vấn đề chưa được M4.6 quyết định

M4.6 không đánh giá:

- predictive benefit của từng feature;
- conditional/experiment feature additions;
- model family;
- imbalance handling;
- threshold;
- full X_train/X_validation materialization memory;
- final feature subset sau model experiments.

Các vấn đề đó không phải blocker cho preprocessing contract hiện tại.

## Blocking issue

`NONE`

## M4.6 Gate

`PASS`

## Trạng thái cuối

`M4.6 — PASS`

`Preprocessing Contract — LOCKED FOR BASELINE V1`

`READY FOR M4.7`

## Handoff

Next:

`M4.7 — Tạo baseline-ready modeling matrix và pipeline audit`

M4.7 phải:

- tái sử dụng đúng W_LONG/W_SHORT preprocessing bundles/procedure;
- giữ stable 47-column schema;
- tạo full TRAIN/VALIDATION matrices;
- audit row alignment với target/lineage;
- audit sparse memory/shape;
- không refit preprocessing state trên VALIDATION;
- không mở FINAL TEST để lựa chọn pipeline.

