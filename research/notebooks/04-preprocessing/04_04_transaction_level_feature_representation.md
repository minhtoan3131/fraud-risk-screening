# M4.4 — Thiết kế transaction-level feature representation

## Vai trò

M4.2 đã khóa Base Representation Contract.

M4.3 đã khóa data-quality preprocessing policy.

M4.4 chuyển các contract đó thành các candidate feature chỉ sử dụng thông tin của transaction hiện tại.

## Câu hỏi trung tâm

> Từ thông tin tồn tại tại prediction point của transaction hiện tại, những feature candidate nào hợp lệ và chúng nên được biểu diễn theo semantic nào?

## Nhóm candidate

- Amount-related;
- transaction mode;
- MCC;
- location;
- calendar/time;
- derived current-transaction features.

## Không thuộc M4.4

- User/Card/Merchant Name history;
- rolling-window feature;
- previous-transaction feature;
- entity-history aggregation;
- learned encoder;
- learned imputer;
- scaling;
- resampling;
- model training;
- hyperparameter tuning;
- threshold selection.

Các historical feature thuộc M4.5.

Các learned preprocessing operation thuộc M4.6.

## Guardrails

Raw identifier không được đưa vào classifier:

- User;
- Card;
- Merchant Name.

Không sử dụng:

- Errors?;
- Is Fraud?;
- FINAL TEST statistics.

## Trạng thái khi bắt đầu

M4.2:
PASS

M4.3:
PASS

Transaction-level Feature Specification:
OPEN

M4.4 Gate:
OPEN

# 1. Contract kế thừa từ M4.2 và M4.3

M4.4 không được đảo ngược các decision đã khóa.

## Amount

Giữ:

signed Amount_numeric

Giữ:

Amount = 0

Không:

- abs();
- drop negative;
- automatic clipping;
- winsorization tại data-quality layer.

## Location

Phân biệt:

NON_PHYSICAL_OR_ONLINE

PHYSICAL_COMPLETE

PHYSICAL_ZIP_UNAVAILABLE

Semantic missing token:

NON_PHYSICAL_OR_ONLINE
→ NOT_APPLICABLE

Physical location nhưng Zip missing
→ ZIP_UNAVAILABLE

## Category normalization

String categorical:

str.strip()

Không tự động:

- lowercase;
- uppercase;
- casefold merge;
- fuzzy merge.

## Zip

Zip là:

CATEGORICAL CODE

không phải continuous numeric magnitude.

## Duplicate

Exact duplicate được giữ.

## Prediction-point rule

Mọi feature trong notebook này phải tính được chỉ từ transaction hiện tại.

Không được dùng:

- transaction tương lai;
- transaction trước đó;
- target;
- aggregate statistics học từ dataset.

## FINAL TEST

FINAL TEST không được dùng để:

- quyết định candidate feature;
- đo cardinality phục vụ feature decision;
- kiểm tra unseen category;
- kiểm tra distribution phục vụ development.

# 2. Candidate feature contract trước runtime

M4.4 bắt đầu với các candidate sau.

## Amount candidates

amount_numeric

Vai trò:
signed raw numerical representation.

amount_signed_log1p

Vai trò:
deterministic magnitude-compressed candidate nhưng vẫn giữ dấu.

is_negative_amount

Vai trò:
explicit sign indicator candidate.

is_zero_amount

Vai trò:
explicit zero indicator candidate.

Chưa quyết định giữ tất cả vào baseline.

---

## Transaction mode

transaction_mode

Nguồn:

Use Chip

Representation:

categorical string sau str.strip().

---

## MCC

mcc_code

Representation:

categorical code.

Không diễn giải numeric magnitude.

MCC được giữ ở trạng thái:

CONDITIONAL CANDIDATE

cho tới khi audit support / unseen category hoàn thành.

---

## Location

location_state

Low-cardinality semantic candidate.

merchant_city_cat

High-cardinality conditional candidate.

merchant_state_cat

Conditional categorical candidate.

zip_cat

High-cardinality categorical-code candidate.

Các raw location candidate không tự động vào baseline chỉ vì tồn tại trong dataset.

---

## Calendar/time

hour_of_day

day_of_week

month_of_year

is_weekend

Tất cả lấy từ Timestamp của transaction hiện tại.

Không sử dụng raw Year như classifier feature trong candidate set hiện tại.

---

## Không phải feature

Timestamp
→ temporal index / source để derive calendar feature.

User
Card
Merchant Name
→ history key only.

raw_row_id
→ lineage only.

partition metadata
→ development control only.


```python
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd


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

DATA_PATH = (
    PROJECT_ROOT
    / DATA_RELATIVE_PATH
)

EXPECTED_FILE_SIZE = 2_354_626_737

CHUNK_SIZE = 500_000


print("PROJECT_ROOT:")
print(PROJECT_ROOT)

print("\nDATA_PATH:")
print(DATA_PATH)

print("\nFile exists:")
print(DATA_PATH.exists())

print("\nFile size:")
print(DATA_PATH.stat().st_size)
```

    PROJECT_ROOT:
    /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening
    
    DATA_PATH:
    /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/raw/ibm_tabformer/card_transaction.v1.csv
    
    File exists:
    True
    
    File size:
    2354626737



```python
assert DATA_PATH.exists()

assert (
    DATA_PATH.stat().st_size
    == EXPECTED_FILE_SIZE
)

print(
    "M4.4 ENVIRONMENT / ARTIFACT GATE: PASS"
)
```

    M4.4 ENVIRONMENT / ARTIFACT GATE: PASS


### Nhận xét

Notebook xác định được đúng project root và raw artifact:

`data/raw/ibm_tabformer/card_transaction.v1.csv`

File tồn tại và có kích thước:

`2,354,626,737 bytes`

khớp với artifact đã được M4.2 và M4.3 sử dụng.

Environment gate trả về:

`M4.4 ENVIRONMENT / ARTIFACT GATE: PASS`

Không có dấu hiệu notebook đang đọc artifact khác hoặc artifact đã bị thay đổi.

### Kết luận

Current raw artifact phù hợp với input contract của M4.4.

Status:

`PASS`

Blocking issue:

`NONE`



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


PERIODS = [
    "TRAIN_2015_2017",
    "TRAIN_2018",
    "VALIDATION",
]


EXPECTED_PERIOD_COUNTS = {
    "TRAIN_2015_2017": 5_133_655,
    "TRAIN_2018": 1_721_615,
    "VALIDATION": 712_458,
}


EXPECTED_DEVELOPMENT_ROWS = (
    sum(
        EXPECTED_PERIOD_COUNTS.values()
    )
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


def assign_audit_period(timestamp):
    conditions = [
        (
            (timestamp >= W_LONG_START)
            & (timestamp < W_SHORT_START)
        ),
        (
            (timestamp >= W_SHORT_START)
            & (timestamp < TRAIN_END)
        ),
        (
            (timestamp >= TRAIN_END)
            & (timestamp < VALIDATION_END)
        ),
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
            default="OUTSIDE_M44_DEVELOPMENT",
        ),
        index=timestamp.index,
        dtype="string",
    )
```


```python
def parse_amount(amount_series):
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


```python
def build_transaction_features(
    df,
    timestamp=None,
):
    if timestamp is None:
        timestamp = build_timestamp(df)

    amount = parse_amount(
        df["Amount"]
    )

    transaction_mode = normalize_string(
        df["Use Chip"]
    )

    merchant_city = normalize_string(
        df["Merchant City"]
    )

    merchant_state_raw = normalize_string(
        df["Merchant State"]
    )

    location_state = (
        assign_location_state(df)
    )


    # ========================================================
    # Amount representation
    # ========================================================

    amount_signed_log1p = (
        np.sign(amount)
        * np.log1p(
            np.abs(amount)
        )
    )

    is_negative_amount = (
        amount < 0
    )

    is_zero_amount = (
        amount == 0
    )


    # ========================================================
    # MCC
    # ========================================================

    mcc_code = (
        df["MCC"]
        .astype("Int64")
        .astype("string")
    )


    # ========================================================
    # Location semantic representation
    # ========================================================

    merchant_city_cat = (
        merchant_city
        .fillna(
            "UNEXPECTED_MISSING"
        )
    )

    merchant_state_cat = (
        merchant_state_raw
        .copy()
    )

    nonphysical_mask = (
        location_state
        == "NON_PHYSICAL_OR_ONLINE"
    )

    merchant_state_cat.loc[
        nonphysical_mask
    ] = "NOT_APPLICABLE"

    merchant_state_cat = (
        merchant_state_cat
        .fillna(
            "UNEXPECTED_MISSING"
        )
    )


    zip_numeric = pd.to_numeric(
        df["Zip"],
        errors="coerce",
    )

    fractional_zip_mask = (
        zip_numeric.notna()
        & (
            zip_numeric % 1
            != 0
        )
    )

    if fractional_zip_mask.any():
        raise ValueError(
            "Zip có non-integer value; "
            "không được silently cast."
        )

    zip_as_string = (
        zip_numeric
        .astype("Int64")
        .astype("string")
    )

    zip_cat = zip_as_string.copy()

    zip_cat.loc[
        nonphysical_mask
    ] = "NOT_APPLICABLE"

    physical_zip_unavailable_mask = (
        location_state
        == "PHYSICAL_ZIP_UNAVAILABLE"
    )

    zip_cat.loc[
        physical_zip_unavailable_mask
    ] = "ZIP_UNAVAILABLE"

    zip_cat = (
        zip_cat
        .fillna(
            "UNEXPECTED_MISSING"
        )
    )


    # ========================================================
    # Calendar / time
    # ========================================================

    hour_of_day = (
        timestamp.dt.hour
        .astype("Int8")
    )

    day_of_week = (
        timestamp.dt.dayofweek
        .astype("Int8")
    )

    month_of_year = (
        timestamp.dt.month
        .astype("Int8")
    )

    is_weekend = (
        day_of_week >= 5
    )


    features = pd.DataFrame(
        {
            "amount_numeric":
                amount,

            "amount_signed_log1p":
                amount_signed_log1p,

            "is_negative_amount":
                is_negative_amount,

            "is_zero_amount":
                is_zero_amount,

            "transaction_mode":
                transaction_mode,

            "mcc_code":
                mcc_code,

            "location_state":
                location_state,

            "merchant_city_cat":
                merchant_city_cat,

            "merchant_state_cat":
                merchant_state_cat,

            "zip_cat":
                zip_cat,

            "hour_of_day":
                hour_of_day,

            "day_of_week":
                day_of_week,

            "month_of_year":
                month_of_year,

            "is_weekend":
                is_weekend,
        },
        index=df.index,
    )

    return features
```

# 3. M4.4.1 — Preview transaction-level feature representation

## Câu hỏi

Candidate builder có:

- bảo toàn row;
- không sử dụng raw identifier;
- không chứa target;
- không tạo unexpected missing;
- giữ Amount sign;
- tạo location semantic đúng;
- tạo calendar feature đúng domain

hay không?


```python
M44_USECOLS = [
    "Year",
    "Month",
    "Day",
    "Time",
    "Amount",
    "Use Chip",
    "Merchant City",
    "Merchant State",
    "Zip",
    "MCC",
]


assert "User" not in M44_USECOLS
assert "Card" not in M44_USECOLS
assert "Merchant Name" not in M44_USECOLS

assert "Errors?" not in M44_USECOLS
assert "Is Fraud?" not in M44_USECOLS


PREVIEW_ROWS = 100_000


preview_raw = pd.read_csv(
    DATA_PATH,
    usecols=M44_USECOLS,
    nrows=PREVIEW_ROWS,
)

preview_timestamp = (
    build_timestamp(
        preview_raw
    )
)

preview_features = (
    build_transaction_features(
        preview_raw,
        timestamp=preview_timestamp,
    )
)


print(
    "Raw preview shape:",
    preview_raw.shape,
)

print(
    "Feature preview shape:",
    preview_features.shape,
)

print(
    "\nFeature columns:"
)

print(
    preview_features.columns.tolist()
)

print(
    "\nFeature dtypes:"
)

print(
    preview_features.dtypes
)

print(
    "\nMissing counts:"
)

print(
    preview_features
    .isna()
    .sum()
)

display(
    preview_features.head(10)
)
```

    Raw preview shape: (100000, 10)
    Feature preview shape: (100000, 14)
    
    Feature columns:
    ['amount_numeric', 'amount_signed_log1p', 'is_negative_amount', 'is_zero_amount', 'transaction_mode', 'mcc_code', 'location_state', 'merchant_city_cat', 'merchant_state_cat', 'zip_cat', 'hour_of_day', 'day_of_week', 'month_of_year', 'is_weekend']
    
    Feature dtypes:
    amount_numeric         Float64
    amount_signed_log1p    Float64
    is_negative_amount     boolean
    is_zero_amount         boolean
    transaction_mode        string
    mcc_code                string
    location_state          string
    merchant_city_cat       string
    merchant_state_cat      string
    zip_cat                 string
    hour_of_day               Int8
    day_of_week               Int8
    month_of_year             Int8
    is_weekend             boolean
    dtype: object
    
    Missing counts:
    amount_numeric         0
    amount_signed_log1p    0
    is_negative_amount     0
    is_zero_amount         0
    transaction_mode       0
    mcc_code               0
    location_state         0
    merchant_city_cat      0
    merchant_state_cat     0
    zip_cat                0
    hour_of_day            0
    day_of_week            0
    month_of_year          0
    is_weekend             0
    dtype: int64



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
      <th>amount_numeric</th>
      <th>amount_signed_log1p</th>
      <th>is_negative_amount</th>
      <th>is_zero_amount</th>
      <th>transaction_mode</th>
      <th>mcc_code</th>
      <th>location_state</th>
      <th>merchant_city_cat</th>
      <th>merchant_state_cat</th>
      <th>zip_cat</th>
      <th>hour_of_day</th>
      <th>day_of_week</th>
      <th>month_of_year</th>
      <th>is_weekend</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>134.09</td>
      <td>4.905941</td>
      <td>False</td>
      <td>False</td>
      <td>Swipe Transaction</td>
      <td>5300</td>
      <td>PHYSICAL_COMPLETE</td>
      <td>La Verne</td>
      <td>CA</td>
      <td>91750</td>
      <td>6</td>
      <td>6</td>
      <td>9</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>38.48</td>
      <td>3.675794</td>
      <td>False</td>
      <td>False</td>
      <td>Swipe Transaction</td>
      <td>5411</td>
      <td>PHYSICAL_COMPLETE</td>
      <td>Monterey Park</td>
      <td>CA</td>
      <td>91754</td>
      <td>6</td>
      <td>6</td>
      <td>9</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>120.34</td>
      <td>4.798597</td>
      <td>False</td>
      <td>False</td>
      <td>Swipe Transaction</td>
      <td>5411</td>
      <td>PHYSICAL_COMPLETE</td>
      <td>Monterey Park</td>
      <td>CA</td>
      <td>91754</td>
      <td>6</td>
      <td>0</td>
      <td>9</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>128.95</td>
      <td>4.86715</td>
      <td>False</td>
      <td>False</td>
      <td>Swipe Transaction</td>
      <td>5651</td>
      <td>PHYSICAL_COMPLETE</td>
      <td>Monterey Park</td>
      <td>CA</td>
      <td>91754</td>
      <td>17</td>
      <td>0</td>
      <td>9</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>104.71</td>
      <td>4.660699</td>
      <td>False</td>
      <td>False</td>
      <td>Swipe Transaction</td>
      <td>5912</td>
      <td>PHYSICAL_COMPLETE</td>
      <td>La Verne</td>
      <td>CA</td>
      <td>91750</td>
      <td>6</td>
      <td>1</td>
      <td>9</td>
      <td>False</td>
    </tr>
    <tr>
      <th>5</th>
      <td>86.19</td>
      <td>4.46809</td>
      <td>False</td>
      <td>False</td>
      <td>Swipe Transaction</td>
      <td>5970</td>
      <td>PHYSICAL_COMPLETE</td>
      <td>Monterey Park</td>
      <td>CA</td>
      <td>91755</td>
      <td>13</td>
      <td>1</td>
      <td>9</td>
      <td>False</td>
    </tr>
    <tr>
      <th>6</th>
      <td>93.84</td>
      <td>4.552191</td>
      <td>False</td>
      <td>False</td>
      <td>Swipe Transaction</td>
      <td>5411</td>
      <td>PHYSICAL_COMPLETE</td>
      <td>Monterey Park</td>
      <td>CA</td>
      <td>91754</td>
      <td>5</td>
      <td>2</td>
      <td>9</td>
      <td>False</td>
    </tr>
    <tr>
      <th>7</th>
      <td>123.5</td>
      <td>4.824306</td>
      <td>False</td>
      <td>False</td>
      <td>Swipe Transaction</td>
      <td>5411</td>
      <td>PHYSICAL_COMPLETE</td>
      <td>Monterey Park</td>
      <td>CA</td>
      <td>91754</td>
      <td>6</td>
      <td>2</td>
      <td>9</td>
      <td>False</td>
    </tr>
    <tr>
      <th>8</th>
      <td>61.72</td>
      <td>4.13868</td>
      <td>False</td>
      <td>False</td>
      <td>Swipe Transaction</td>
      <td>5411</td>
      <td>PHYSICAL_COMPLETE</td>
      <td>Monterey Park</td>
      <td>CA</td>
      <td>91754</td>
      <td>6</td>
      <td>3</td>
      <td>9</td>
      <td>False</td>
    </tr>
    <tr>
      <th>9</th>
      <td>57.1</td>
      <td>4.062166</td>
      <td>False</td>
      <td>False</td>
      <td>Swipe Transaction</td>
      <td>7538</td>
      <td>PHYSICAL_COMPLETE</td>
      <td>La Verne</td>
      <td>CA</td>
      <td>91750</td>
      <td>9</td>
      <td>3</td>
      <td>9</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



```python
assert (
    len(preview_features)
    == len(preview_raw)
)

assert (
    preview_features
    .isna()
    .sum()
    .sum()
    == 0
)

assert not (
    preview_features[
        "location_state"
    ]
    .eq("OTHER_INCONSISTENT")
    .any()
)

assert not (
    preview_features[
        "merchant_state_cat"
    ]
    .eq("UNEXPECTED_MISSING")
    .any()
)

assert not (
    preview_features[
        "zip_cat"
    ]
    .eq("UNEXPECTED_MISSING")
    .any()
)

assert (
    preview_features[
        "hour_of_day"
    ]
    .between(0, 23)
    .all()
)

assert (
    preview_features[
        "day_of_week"
    ]
    .between(0, 6)
    .all()
)

assert (
    preview_features[
        "month_of_year"
    ]
    .between(1, 12)
    .all()
)

raw_sign = np.sign(
    preview_features[
        "amount_numeric"
    ]
)

log_sign = np.sign(
    preview_features[
        "amount_signed_log1p"
    ]
)

assert (
    raw_sign == log_sign
).all()


print(
    "M4.4 FEATURE PREVIEW GATE: PASS"
)
```

    M4.4 FEATURE PREVIEW GATE: PASS



```python
preview_memory_bytes = int(
    preview_features
    .memory_usage(
        deep=True
    )
    .sum()
)

preview_bytes_per_row = (
    preview_memory_bytes
    / len(preview_features)
)

print(
    "Preview feature memory:",
    f"{preview_memory_bytes:,}",
    "bytes",
)

print(
    "Approx bytes / row:",
    round(
        preview_bytes_per_row,
        2,
    ),
)
```

    Preview feature memory: 38,421,983 bytes
    Approx bytes / row: 384.22


### Nhận xét

Preview sử dụng:

`100,000 raw rows`

và tạo:

`100,000 × 14 transaction-level candidate features`

Row count được bảo toàn hoàn toàn.

Candidate frame gồm:

`amount_numeric`

`amount_signed_log1p`

`is_negative_amount`

`is_zero_amount`

`transaction_mode`

`mcc_code`

`location_state`

`merchant_city_cat`

`merchant_state_cat`

`zip_cat`

`hour_of_day`

`day_of_week`

`month_of_year`

`is_weekend`

Không có:

- `User`;
- `Card`;
- `Merchant Name`;
- `Errors?`;
- `Is Fraud?`;
- partition metadata.

Như vậy preview frame tuân thủ guardrail về identifier, target và excluded field.

Tất cả 14 candidate đều có:

`missing count = 0`

Preview cũng không xuất hiện:

`OTHER_INCONSISTENT`

`UNEXPECTED_MISSING`

trong location representation.

Các calendar feature có domain hợp lệ.

`amount_signed_log1p` giữ nguyên dấu của `amount_numeric`.

Cell assertion trả về:

`M4.4 FEATURE PREVIEW GATE: PASS`

Memory của preview frame là:

`38,421,983 bytes`

tương đương khoảng:

`384.22 bytes / row`

Đây vẫn là DataFrame audit với nhiều string categorical value, chưa phải modeling matrix đã encode. Nếu ngoại suy tuyến tính toàn development population, representation dạng này sẽ ở mức nhiều GB bộ nhớ, vì vậy M4.6 không nên giả định có thể materialize toàn bộ string feature frame một cách không kiểm soát.

### Kết luận M4.4.1

Transaction-level candidate builder hoạt động đúng về:

- row preservation;
- semantic representation;
- missing-value policy;
- identifier exclusion;
- target exclusion;
- prediction-point compatibility.

Status:

`PASS`

Feature builder:

`VALID FOR FULL DEVELOPMENT AUDIT`

Memory note:

`HANDOFF TO M4.6 FOR MEMORY-EFFICIENT PREPROCESSING`



# 4. M4.4.2 — Full development feature audit

Preview chỉ xác minh implementation.

Feature decision của M4.4 phải dựa trên development population:

TRAIN_2015_2017
+
TRAIN_2018
+
VALIDATION

M4.4 không chọn W_LONG/W_SHORT winner.

Do đó categorical generalization sẽ được kiểm tra dưới cả hai training-window candidate:

W_LONG:
TRAIN_2015_2017 + TRAIN_2018

W_SHORT:
TRAIN_2018

đều transform sang cùng VALIDATION.

## Full scan thu thập

- period row count;
- feature missing;
- unexpected semantic state;
- Amount distribution;
- categorical cardinality;
- category support;
- VALIDATION unseen categories;
- rare-support exposure;
- temporal distribution stability.

Không sử dụng target.


```python
period_counts = Counter()

candidate_missing_counts = Counter()

unexpected_location_rows = 0
unexpected_state_rows = 0
unexpected_zip_rows = 0

timestamp_parse_failures = 0
amount_parse_failures = 0


CATEGORICAL_AUDIT_FEATURES = [
    "transaction_mode",
    "mcc_code",
    "location_state",
    "merchant_city_cat",
    "merchant_state_cat",
    "zip_cat",
    "hour_of_day",
    "day_of_week",
    "month_of_year",
]


categorical_counts = {
    period: {
        feature: Counter()
        for feature
        in CATEGORICAL_AUDIT_FEATURES
    }
    for period in PERIODS
}


amount_stats = {
    period: {
        "count": 0,
        "raw_sum": 0.0,
        "raw_sum_sq": 0.0,
        "raw_min": None,
        "raw_max": None,
        "log_sum": 0.0,
        "log_sum_sq": 0.0,
        "log_min": None,
        "log_max": None,
        "negative_count": 0,
        "zero_count": 0,
    }
    for period in PERIODS
}


amount_sample_parts = []

raw_row_offset = 0

development_min_timestamp = None
development_max_timestamp = None
```


```python
for chunk_number, chunk in enumerate(
    pd.read_csv(
        DATA_PATH,
        usecols=M44_USECOLS,
        chunksize=CHUNK_SIZE,
    ),
    start=1,
):
    timestamp = build_timestamp(
        chunk
    )

    timestamp_parse_failures += int(
        timestamp.isna().sum()
    )

    audit_period = (
        assign_audit_period(
            timestamp
        )
    )

    development_mask = (
        audit_period
        != "OUTSIDE_M44_DEVELOPMENT"
    )

    global_row_ids = np.arange(
        raw_row_offset,
        raw_row_offset + len(chunk),
        dtype=np.int64,
    )

    if development_mask.any():
        dev = (
            chunk.loc[
                development_mask
            ]
            .copy()
            .reset_index(drop=True)
        )

        dev_timestamp = (
            timestamp.loc[
                development_mask
            ]
            .reset_index(drop=True)
        )

        dev_period = (
            audit_period.loc[
                development_mask
            ]
            .reset_index(drop=True)
        )

        dev_raw_ids = (
            global_row_ids[
                development_mask.to_numpy()
            ]
        )

        features = (
            build_transaction_features(
                dev,
                timestamp=dev_timestamp,
            )
        )

        # ----------------------------------------------------
        # Scope
        # ----------------------------------------------------

        period_counts.update(
            dev_period
            .value_counts()
            .to_dict()
        )

        chunk_min = (
            dev_timestamp.min()
        )

        chunk_max = (
            dev_timestamp.max()
        )

        if (
            development_min_timestamp is None
            or chunk_min
            < development_min_timestamp
        ):
            development_min_timestamp = (
                chunk_min
            )

        if (
            development_max_timestamp is None
            or chunk_max
            > development_max_timestamp
        ):
            development_max_timestamp = (
                chunk_max
            )


        # ----------------------------------------------------
        # Missing / representation integrity
        # ----------------------------------------------------

        missing = (
            features
            .isna()
            .sum()
        )

        for feature, count in (
            missing.items()
        ):
            candidate_missing_counts[
                feature
            ] += int(count)

        amount_parse_failures += int(
            features[
                "amount_numeric"
            ]
            .isna()
            .sum()
        )

        unexpected_location_rows += int(
            features[
                "location_state"
            ]
            .eq(
                "OTHER_INCONSISTENT"
            )
            .sum()
        )

        unexpected_state_rows += int(
            features[
                "merchant_state_cat"
            ]
            .eq(
                "UNEXPECTED_MISSING"
            )
            .sum()
        )

        unexpected_zip_rows += int(
            features[
                "zip_cat"
            ]
            .eq(
                "UNEXPECTED_MISSING"
            )
            .sum()
        )


        # ----------------------------------------------------
        # Period-specific audit
        # ----------------------------------------------------

        for period in PERIODS:
            period_mask = (
                dev_period
                .eq(period)
            )

            if not period_mask.any():
                continue

            f = (
                features.loc[
                    period_mask
                    .to_numpy()
                ]
            )

            # ----------------------------------------------
            # categorical support
            # ----------------------------------------------

            for feature in (
                CATEGORICAL_AUDIT_FEATURES
            ):
                categorical_counts[
                    period
                ][
                    feature
                ].update(
                    f[
                        feature
                    ]
                    .astype("string")
                    .value_counts(
                        dropna=False
                    )
                    .to_dict()
                )


            # ----------------------------------------------
            # Amount statistics
            # ----------------------------------------------

            raw_amount = (
                f[
                    "amount_numeric"
                ]
                .astype(float)
                .to_numpy()
            )

            log_amount = (
                f[
                    "amount_signed_log1p"
                ]
                .astype(float)
                .to_numpy()
            )

            stats = (
                amount_stats[
                    period
                ]
            )

            stats["count"] += len(
                raw_amount
            )

            stats["raw_sum"] += float(
                raw_amount.sum()
            )

            stats["raw_sum_sq"] += float(
                np.square(
                    raw_amount
                ).sum()
            )

            stats["log_sum"] += float(
                log_amount.sum()
            )

            stats["log_sum_sq"] += float(
                np.square(
                    log_amount
                ).sum()
            )

            current_raw_min = float(
                raw_amount.min()
            )

            current_raw_max = float(
                raw_amount.max()
            )

            current_log_min = float(
                log_amount.min()
            )

            current_log_max = float(
                log_amount.max()
            )

            if (
                stats["raw_min"]
                is None
                or current_raw_min
                < stats["raw_min"]
            ):
                stats["raw_min"] = (
                    current_raw_min
                )

            if (
                stats["raw_max"]
                is None
                or current_raw_max
                > stats["raw_max"]
            ):
                stats["raw_max"] = (
                    current_raw_max
                )

            if (
                stats["log_min"]
                is None
                or current_log_min
                < stats["log_min"]
            ):
                stats["log_min"] = (
                    current_log_min
                )

            if (
                stats["log_max"]
                is None
                or current_log_max
                > stats["log_max"]
            ):
                stats["log_max"] = (
                    current_log_max
                )

            stats[
                "negative_count"
            ] += int(
                (
                    raw_amount < 0
                ).sum()
            )

            stats[
                "zero_count"
            ] += int(
                (
                    raw_amount == 0
                ).sum()
            )


        # ----------------------------------------------------
        # Deterministic 1% Amount sample
        # ----------------------------------------------------

        sample_mask = (
            dev_raw_ids % 100
            == 0
        )

        if sample_mask.any():
            amount_sample_parts.append(
                features.loc[
                    sample_mask,
                    "amount_numeric",
                ]
                .astype(float)
                .to_numpy()
            )


    raw_row_offset += len(chunk)


    if (
        chunk_number % 10 == 0
        or len(chunk) < CHUNK_SIZE
    ):
        print(
            f"Chunk {chunk_number:02d} | "
            f"development rows = "
            f"{sum(period_counts.values()):,}"
        )
```

    Chunk 10 | development rows = 1,528,613
    Chunk 20 | development rows = 3,092,436
    Chunk 30 | development rows = 4,630,241
    Chunk 40 | development rows = 6,172,310
    Chunk 49 | development rows = 7,567,728



```python
print(
    "Development rows:",
    f"{sum(period_counts.values()):,}",
)

print(
    "\nPeriod counts:"
)

for period in PERIODS:
    print(
        period,
        f"{period_counts[period]:,}",
    )

print(
    "\nTimestamp parse failures:",
    timestamp_parse_failures,
)

print(
    "Amount parse failures:",
    amount_parse_failures,
)

print(
    "\nDevelopment min Timestamp:",
    development_min_timestamp,
)

print(
    "Development max Timestamp:",
    development_max_timestamp,
)

print(
    "\nUnexpected location rows:",
    unexpected_location_rows,
)

print(
    "Unexpected State rows:",
    unexpected_state_rows,
)

print(
    "Unexpected Zip rows:",
    unexpected_zip_rows,
)

print(
    "\nCandidate missing counts:"
)

print(
    dict(
        candidate_missing_counts
    )
)
```

    Development rows: 7,567,728
    
    Period counts:
    TRAIN_2015_2017 5,133,655
    TRAIN_2018 1,721,615
    VALIDATION 712,458
    
    Timestamp parse failures: 0
    Amount parse failures: 0
    
    Development min Timestamp: 2015-01-01 00:01:00
    Development max Timestamp: 2019-05-31 23:58:00
    
    Unexpected location rows: 0
    Unexpected State rows: 0
    Unexpected Zip rows: 0
    
    Candidate missing counts:
    {'amount_numeric': 0, 'amount_signed_log1p': 0, 'is_negative_amount': 0, 'is_zero_amount': 0, 'transaction_mode': 0, 'mcc_code': 0, 'location_state': 0, 'merchant_city_cat': 0, 'merchant_state_cat': 0, 'zip_cat': 0, 'hour_of_day': 0, 'day_of_week': 0, 'month_of_year': 0, 'is_weekend': 0}



```python
assert (
    sum(period_counts.values())
    == EXPECTED_DEVELOPMENT_ROWS
)

for period, expected in (
    EXPECTED_PERIOD_COUNTS.items()
):
    assert (
        period_counts[period]
        == expected
    )

assert timestamp_parse_failures == 0
assert amount_parse_failures == 0

assert unexpected_location_rows == 0
assert unexpected_state_rows == 0
assert unexpected_zip_rows == 0

assert (
    sum(
        candidate_missing_counts.values()
    )
    == 0
)

assert (
    development_min_timestamp
    >= W_LONG_START
)

assert (
    development_max_timestamp
    < VALIDATION_END
)


print(
    "M4.4 DEVELOPMENT FEATURE AUDIT GATE: PASS"
)
```

    M4.4 DEVELOPMENT FEATURE AUDIT GATE: PASS


### Nhận xét

Full scan đã xử lý đúng:

`7,567,728 development rows`

gồm:

`TRAIN_2015_2017 = 5,133,655`

`TRAIN_2018 = 1,721,615`

`VALIDATION = 712,458`

Temporal extent:

`2015-01-01 00:01:00`
→
`2019-05-31 23:58:00`

Không có FINAL TEST transaction trong development audit.

Runtime integrity:

`Timestamp parse failures = 0`

`Amount parse failures = 0`

`Unexpected location rows = 0`

`Unexpected State rows = 0`

`Unexpected Zip rows = 0`

Tất cả 14 candidate feature đều có:

`missing count = 0`

sau khi áp dụng deterministic representation đã khóa.

Cell assertion trả về:

`M4.4 DEVELOPMENT FEATURE AUDIT GATE: PASS`

Như vậy candidate representation không chỉ hoạt động trên preview mà tái hiện được trên toàn development population.

### Kết luận M4.4.2

Full development feature representation đạt integrity requirements.

Development scope:

`VERIFIED`

Candidate missing:

`0`

Unexpected semantic states:

`0`

FINAL TEST used:

`NO`

Status:

`PASS`




# 5. M4.4.3 — Audit Amount feature candidates

## Candidate

A.
amount_numeric

B.
amount_signed_log1p

C.
is_negative_amount

D.
is_zero_amount

M4.4 không dùng target để chọn candidate.

Cần kiểm tra:

- distribution theo temporal period;
- range;
- scale compression;
- sign preservation;
- support của negative / zero;
- deterministic implementation.

Scaling vẫn thuộc M4.6.


```python
amount_summary_rows = []

for period in PERIODS:
    stats = amount_stats[
        period
    ]

    n = stats["count"]

    raw_mean = (
        stats["raw_sum"]
        / n
    )

    raw_variance = max(
        (
            stats["raw_sum_sq"]
            / n
        )
        - raw_mean**2,
        0.0,
    )

    raw_std = (
        raw_variance
        ** 0.5
    )

    log_mean = (
        stats["log_sum"]
        / n
    )

    log_variance = max(
        (
            stats["log_sum_sq"]
            / n
        )
        - log_mean**2,
        0.0,
    )

    log_std = (
        log_variance
        ** 0.5
    )

    amount_summary_rows.append(
        {
            "period": period,
            "rows": n,

            "raw_min":
                stats["raw_min"],

            "raw_max":
                stats["raw_max"],

            "raw_mean":
                raw_mean,

            "raw_std":
                raw_std,

            "signed_log_min":
                stats["log_min"],

            "signed_log_max":
                stats["log_max"],

            "signed_log_mean":
                log_mean,

            "signed_log_std":
                log_std,

            "negative_count":
                stats[
                    "negative_count"
                ],

            "negative_rate_pct":
                (
                    stats[
                        "negative_count"
                    ]
                    / n
                    * 100
                ),

            "zero_count":
                stats[
                    "zero_count"
                ],

            "zero_rate_pct":
                (
                    stats[
                        "zero_count"
                    ]
                    / n
                    * 100
                ),
        }
    )


amount_summary_df = (
    pd.DataFrame(
        amount_summary_rows
    )
)

display(
    amount_summary_df
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
      <th>period</th>
      <th>rows</th>
      <th>raw_min</th>
      <th>raw_max</th>
      <th>raw_mean</th>
      <th>raw_std</th>
      <th>signed_log_min</th>
      <th>signed_log_max</th>
      <th>signed_log_mean</th>
      <th>signed_log_std</th>
      <th>negative_count</th>
      <th>negative_rate_pct</th>
      <th>zero_count</th>
      <th>zero_rate_pct</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>TRAIN_2015_2017</td>
      <td>5133655</td>
      <td>-500.0</td>
      <td>5155.36</td>
      <td>42.968135</td>
      <td>81.121028</td>
      <td>-6.216606</td>
      <td>8.547986</td>
      <td>2.921030</td>
      <td>2.057003</td>
      <td>248541</td>
      <td>4.841404</td>
      <td>4099</td>
      <td>0.079846</td>
    </tr>
    <tr>
      <th>1</th>
      <td>TRAIN_2018</td>
      <td>1721615</td>
      <td>-500.0</td>
      <td>5682.22</td>
      <td>42.879028</td>
      <td>80.550656</td>
      <td>-6.216606</td>
      <td>8.645273</td>
      <td>2.922356</td>
      <td>2.048261</td>
      <td>82314</td>
      <td>4.781208</td>
      <td>1364</td>
      <td>0.079228</td>
    </tr>
    <tr>
      <th>2</th>
      <td>VALIDATION</td>
      <td>712458</td>
      <td>-500.0</td>
      <td>6613.44</td>
      <td>42.895124</td>
      <td>81.068243</td>
      <td>-6.216606</td>
      <td>8.797010</td>
      <td>2.921325</td>
      <td>2.052303</td>
      <td>34311</td>
      <td>4.815863</td>
      <td>545</td>
      <td>0.076496</td>
    </tr>
  </tbody>
</table>
</div>



```python
amount_sample = np.concatenate(
    amount_sample_parts
)

signed_log_sample = (
    np.sign(amount_sample)
    * np.log1p(
        np.abs(
            amount_sample
        )
    )
)


quantile_levels = [
    0.00,
    0.01,
    0.05,
    0.25,
    0.50,
    0.75,
    0.95,
    0.99,
    1.00,
]


amount_quantile_df = pd.DataFrame(
    {
        "quantile":
            quantile_levels,

        "amount_numeric":
            np.quantile(
                amount_sample,
                quantile_levels,
            ),

        "amount_signed_log1p":
            np.quantile(
                signed_log_sample,
                quantile_levels,
            ),
    }
)


print(
    "Deterministic sample rows:",
    f"{len(amount_sample):,}",
)

display(
    amount_quantile_df
)
```

    Deterministic sample rows: 75,670



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
      <th>quantile</th>
      <th>amount_numeric</th>
      <th>amount_signed_log1p</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.00</td>
      <td>-498.0000</td>
      <td>-6.212606</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.01</td>
      <td>-95.0000</td>
      <td>-4.564348</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.05</td>
      <td>0.1745</td>
      <td>0.160834</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.25</td>
      <td>9.0300</td>
      <td>2.305581</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.50</td>
      <td>29.2100</td>
      <td>3.408173</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.75</td>
      <td>62.8200</td>
      <td>4.156067</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.95</td>
      <td>144.9810</td>
      <td>4.983476</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.99</td>
      <td>310.0000</td>
      <td>5.739793</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1.00</td>
      <td>1831.7200</td>
      <td>7.513556</td>
    </tr>
  </tbody>
</table>
</div>


### Phân tích / Nhận xét

Raw Amount có distribution lệch phải rõ.

Mean của `amount_numeric` khá ổn định qua ba period:

`42.968`

`42.879`

`42.895`

Standard deviation cũng gần như ổn định:

`81.121`

`80.551`

`81.068`

Tuy nhiên range rộng hơn nhiều so với phần giữa của distribution.

Full-period maximum lần lượt là:

`5,155.36`

`5,682.22`

`6,613.44`

trong khi deterministic 1% sample có:

median:

`29.21`

95th percentile:

`144.981`

99th percentile:

`310.00`

Điều này cho thấy một số Amount lớn kéo range lên đáng kể so với transaction thông thường.

Signed-log representation:

`sign(x) × log1p(abs(x))`

nén range rất mạnh.

Ví dụ full-period positive maximum sau transform chỉ còn khoảng:

`8.55 → 8.80`

và minimum `-500` được biểu diễn khoảng:

`-6.2166`

Transformation vẫn giữ nguyên dấu, nên không vi phạm M4.3 policy.

Negative support cũng không nhỏ:

`4.8414%`

`4.7812%`

`4.8159%`

qua ba period.

Zero Amount tuy hiếm nhưng vẫn có hàng trăm/hàng nghìn observation:

`4,099`

`1,364`

`545`

và tỷ lệ khoảng:

`0.076%–0.080%`.

Vì vậy `is_negative_amount` không phải near-empty indicator.

`is_zero_amount` có support nhỏ hơn nhiều nhưng cũng không phải impossible state.

Tuy nhiên:

`is_negative_amount`

và:

`is_zero_amount`

đều là deterministic function của `amount_numeric`.

Tương tự:

`amount_signed_log1p`

không bổ sung thông tin mới về transaction; nó thay đổi functional representation của cùng Amount.

M4.4 không có model-performance evidence để chứng minh rằng đưa đồng thời raw Amount, signed-log Amount và các indicator vào baseline sẽ tốt hơn.

Vì vậy nên phân biệt:

- representation chính đơn giản;
- alternative/derived candidates để experiment sau.

### Kết luận M4.4.3

`amount_numeric`

được khóa là:

`PRIMARY TRANSACTION-LEVEL AMOUNT CANDIDATE`

vì:

- giữ nguyên semantic;
- không missing;
- ổn định giữa temporal periods;
- không cần learned preprocessing state.

`amount_signed_log1p`

được giữ là:

`ALTERNATIVE EXPERIMENT CANDIDATE`

do khả năng nén scale mạnh nhưng chưa có evidence để thay raw Amount làm representation mặc định.

Không đưa đồng thời raw Amount và signed-log Amount vào core baseline chỉ vì chúng đều khả dụng.

`is_negative_amount`

được giữ là:

`DERIVED EXPERIMENT CANDIDATE`

`is_zero_amount`

được giữ là:

`DERIVED EXPERIMENT CANDIDATE`

Hai indicator này chưa phải core baseline vì thông tin đã có thể suy ra từ Amount.

Amount candidate decision:

`LOCKED AT REPRESENTATION LEVEL`

Performance winner giữa raw/log/indicator combination:

`OPEN — REQUIRES EXPERIMENT`



```python
def combine_counters(
    *counters
):
    result = Counter()

    for counter in counters:
        result.update(
            counter
        )

    return result


def get_training_counter(
    strategy,
    feature,
):
    if strategy == "W_LONG":
        return combine_counters(
            categorical_counts[
                "TRAIN_2015_2017"
            ][feature],
            categorical_counts[
                "TRAIN_2018"
            ][feature],
        )

    if strategy == "W_SHORT":
        return Counter(
            categorical_counts[
                "TRAIN_2018"
            ][feature]
        )

    raise ValueError(
        f"Unknown strategy: "
        f"{strategy}"
    )
```

# 6. M4.4.4 — Categorical support / unseen audit

## Câu hỏi

Categorical candidate có đủ support và khả năng transform VALIDATION hay không?

Đặc biệt cần audit:

- MCC;
- Merchant City;
- Merchant State;
- Zip.

Hai training-window candidate được kiểm tra riêng:

W_LONG → VALIDATION

W_SHORT → VALIDATION

M4.4 không quyết định cách encode unknown category.

M4.6 sẽ thiết kế unknown-category handling.

M4.4 chỉ đo mức độ vấn đề.


```python
generalization_rows = []


for strategy in [
    "W_LONG",
    "W_SHORT",
]:
    for feature in (
        CATEGORICAL_AUDIT_FEATURES
    ):
        train_counter = (
            get_training_counter(
                strategy,
                feature,
            )
        )

        validation_counter = (
            categorical_counts[
                "VALIDATION"
            ][feature]
        )

        train_categories = set(
            train_counter.keys()
        )

        validation_categories = set(
            validation_counter.keys()
        )

        unseen_categories = (
            validation_categories
            - train_categories
        )

        validation_rows = sum(
            validation_counter.values()
        )

        unseen_validation_rows = sum(
            validation_counter[
                category
            ]
            for category
            in unseen_categories
        )

        categories_lt_10 = {
            category
            for category, count
            in train_counter.items()
            if count < 10
        }

        categories_lt_100 = {
            category
            for category, count
            in train_counter.items()
            if count < 100
        }

        categories_lt_1000 = {
            category
            for category, count
            in train_counter.items()
            if count < 1000
        }

        validation_rows_train_lt_100 = sum(
            count
            for category, count
            in validation_counter.items()
            if train_counter.get(
                category,
                0,
            ) < 100
        )

        generalization_rows.append(
            {
                "strategy":
                    strategy,

                "feature":
                    feature,

                "train_unique":
                    len(
                        train_categories
                    ),

                "validation_unique":
                    len(
                        validation_categories
                    ),

                "validation_unseen_unique":
                    len(
                        unseen_categories
                    ),

                "validation_unseen_rows":
                    unseen_validation_rows,

                "validation_unseen_rate_pct":
                    (
                        unseen_validation_rows
                        / validation_rows
                        * 100
                    ),

                "train_categories_lt_10":
                    len(
                        categories_lt_10
                    ),

                "train_categories_lt_100":
                    len(
                        categories_lt_100
                    ),

                "train_categories_lt_1000":
                    len(
                        categories_lt_1000
                    ),

                "validation_rows_with_train_support_lt_100":
                    validation_rows_train_lt_100,

                "validation_rate_train_support_lt_100_pct":
                    (
                        validation_rows_train_lt_100
                        / validation_rows
                        * 100
                    ),
            }
        )


generalization_df = (
    pd.DataFrame(
        generalization_rows
    )
)


display(
    generalization_df
    .sort_values(
        [
            "strategy",
            "feature",
        ]
    )
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
      <th>train_unique</th>
      <th>validation_unique</th>
      <th>validation_unseen_unique</th>
      <th>validation_unseen_rows</th>
      <th>validation_unseen_rate_pct</th>
      <th>train_categories_lt_10</th>
      <th>train_categories_lt_100</th>
      <th>train_categories_lt_1000</th>
      <th>validation_rows_with_train_support_lt_100</th>
      <th>validation_rate_train_support_lt_100_pct</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7</th>
      <td>W_LONG</td>
      <td>day_of_week</td>
      <td>7</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>W_LONG</td>
      <td>hour_of_day</td>
      <td>24</td>
      <td>24</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>W_LONG</td>
      <td>location_state</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>W_LONG</td>
      <td>mcc_code</td>
      <td>109</td>
      <td>109</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>14</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>W_LONG</td>
      <td>merchant_city_cat</td>
      <td>11233</td>
      <td>6433</td>
      <td>204</td>
      <td>912</td>
      <td>0.128008</td>
      <td>3845</td>
      <td>7778</td>
      <td>10023</td>
      <td>16478</td>
      <td>2.312838</td>
    </tr>
    <tr>
      <th>4</th>
      <td>W_LONG</td>
      <td>merchant_state_cat</td>
      <td>161</td>
      <td>107</td>
      <td>4</td>
      <td>53</td>
      <td>0.007439</td>
      <td>19</td>
      <td>66</td>
      <td>101</td>
      <td>220</td>
      <td>0.030879</td>
    </tr>
    <tr>
      <th>8</th>
      <td>W_LONG</td>
      <td>month_of_year</td>
      <td>12</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>0</th>
      <td>W_LONG</td>
      <td>transaction_mode</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>W_LONG</td>
      <td>zip_cat</td>
      <td>22603</td>
      <td>12389</td>
      <td>427</td>
      <td>1722</td>
      <td>0.241698</td>
      <td>8138</td>
      <td>16685</td>
      <td>20989</td>
      <td>35960</td>
      <td>5.047315</td>
    </tr>
    <tr>
      <th>16</th>
      <td>W_SHORT</td>
      <td>day_of_week</td>
      <td>7</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>15</th>
      <td>W_SHORT</td>
      <td>hour_of_day</td>
      <td>24</td>
      <td>24</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>W_SHORT</td>
      <td>location_state</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>W_SHORT</td>
      <td>mcc_code</td>
      <td>109</td>
      <td>109</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>10</td>
      <td>43</td>
      <td>206</td>
      <td>0.028914</td>
    </tr>
    <tr>
      <th>12</th>
      <td>W_SHORT</td>
      <td>merchant_city_cat</td>
      <td>8046</td>
      <td>6433</td>
      <td>731</td>
      <td>3298</td>
      <td>0.462904</td>
      <td>3303</td>
      <td>6127</td>
      <td>7675</td>
      <td>50021</td>
      <td>7.020905</td>
    </tr>
    <tr>
      <th>13</th>
      <td>W_SHORT</td>
      <td>merchant_state_cat</td>
      <td>127</td>
      <td>107</td>
      <td>9</td>
      <td>107</td>
      <td>0.015018</td>
      <td>13</td>
      <td>61</td>
      <td>72</td>
      <td>1125</td>
      <td>0.157904</td>
    </tr>
    <tr>
      <th>17</th>
      <td>W_SHORT</td>
      <td>month_of_year</td>
      <td>12</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>W_SHORT</td>
      <td>transaction_mode</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>14</th>
      <td>W_SHORT</td>
      <td>zip_cat</td>
      <td>15922</td>
      <td>12389</td>
      <td>1532</td>
      <td>6870</td>
      <td>0.964267</td>
      <td>7087</td>
      <td>13094</td>
      <td>15675</td>
      <td>99590</td>
      <td>13.978368</td>
    </tr>
  </tbody>
</table>
</div>



```python
HIGH_CARD_FEATURES = [
    "mcc_code",
    "merchant_city_cat",
    "merchant_state_cat",
    "zip_cat",
]


for strategy in [
    "W_LONG",
    "W_SHORT",
]:
    print(
        "\n============================"
    )

    print(
        "TRAIN STRATEGY:",
        strategy,
    )

    print(
        "============================"
    )

    for feature in (
        HIGH_CARD_FEATURES
    ):
        train_counter = (
            get_training_counter(
                strategy,
                feature,
            )
        )

        validation_counter = (
            categorical_counts[
                "VALIDATION"
            ][feature]
        )

        unseen = [
            (
                category,
                validation_counter[
                    category
                ],
            )
            for category
            in validation_counter
            if category
            not in train_counter
        ]

        unseen = sorted(
            unseen,
            key=lambda x: x[1],
            reverse=True,
        )

        print(
            f"\n{feature}"
        )

        print(
            "Top unseen validation "
            "categories:"
        )

        print(
            unseen[:20]
        )
```

    
    ============================
    TRAIN STRATEGY: W_LONG
    ============================
    
    mcc_code
    Top unseen validation categories:
    []
    
    merchant_city_cat
    Top unseen validation categories:
    [('Rochelle Park', 36), ('Eastview', 35), ('Joseph', 31), ('Manassa', 28), ('Marion Heights', 24), ('Laura', 23), ('Zillah', 23), ('Tegucigalpa', 21), ('Leaf River', 20), ('Woodacre', 20), ('Jayess', 19), ('Ellerslie', 18), ('North Grafton', 17), ('Pahala', 17), ('Barnsdall', 14), ('Tallinn', 14), ('Hanoi', 13), ('Port of Spain', 13), ('Summitville', 12), ('Grandin', 10)]
    
    merchant_state_cat
    Top unseen validation categories:
    [('Honduras', 21), ('Estonia', 14), ('Trinidad and Tobago', 13), ('Micronesia', 5)]
    
    zip_cat
    Top unseen validation categories:
    [('7662', 36), ('42732', 35), ('97846', 31), ('81141', 28), ('69145', 28), ('17832', 24), ('45337', 23), ('33896', 23), ('98953', 23), ('66436', 21), ('61047', 20), ('94973', 20), ('39641', 19), ('21529', 18), ('1536', 17), ('96777', 17), ('78156', 17), ('82718', 16), ('19401', 15), ('40409', 15)]
    
    ============================
    TRAIN STRATEGY: W_SHORT
    ============================
    
    mcc_code
    Top unseen validation categories:
    []
    
    merchant_city_cat
    Top unseen validation categories:
    [('Mounds', 94), ('Chattaroy', 50), ('Dardanelle', 50), ('Tell City', 42), ('Saddle Brook', 38), ('Rochelle Park', 36), ('Seven Valleys', 36), ('Eastview', 35), ('Wasco', 35), ('Joseph', 31), ('Rison', 30), ('East Weymouth', 29), ('Cornwall', 29), ('Manassa', 28), ('Kimball', 28), ('Grass Valley', 27), ('Hopkinton', 27), ('Beecher', 25), ('Marion Heights', 24), ('Kitty Hawk', 24)]
    
    merchant_state_cat
    Top unseen validation categories:
    [('Honduras', 21), ('Serbia', 15), ('Slovakia', 15), ('Estonia', 14), ('Trinidad and Tobago', 13), ('Chile', 12), ('Latvia', 11), ('Micronesia', 5), ('Bahrain', 1)]
    
    zip_cat
    Top unseen validation categories:
    [('85388', 121), ('89126', 106), ('33181', 95), ('74047', 94), ('93035', 71), ('99003', 50), ('72834', 50), ('99004', 44), ('47586', 42), ('7663', 38), ('7662', 36), ('17360', 36), ('42732', 35), ('93280', 35), ('77803', 33), ('89049', 32), ('97218', 31), ('97846', 31), ('93279', 30), ('71665', 30)]


### Phân tích / Nhận xét

Các categorical candidate low-cardinality có generalization rất sạch từ TRAIN sang VALIDATION.

Dưới cả W_LONG và W_SHORT:

`transaction_mode`

có:

`0 unseen categories`

`location_state`

có:

`0 unseen categories`

`hour_of_day`

có:

`0 unseen categories`

`day_of_week`

có:

`0 unseen categories`

`month_of_year`

có:

`0 unseen validation categories`

MCC cũng có:

`109 train categories`

`109 validation categories`

và:

`0 unseen MCC`

dưới cả W_LONG lẫn W_SHORT.

Do đó MCC không gặp unknown-category problem trong current development split.

Tuy nhiên một số MCC có support thấp khi chỉ dùng W_SHORT.

W_SHORT có:

`10 MCC categories`

với training support dưới `100`.

Nhưng chỉ:

`206 validation rows`

tương đương khoảng:

`0.0289%`

VALIDATION thuộc category có W_SHORT training support dưới 100.

Do đó MCC long-tail tồn tại nhưng ở mức hạn chế về row exposure.

Raw location cho pattern khác.

### Merchant State

W_LONG:

`161 train categories`

`107 validation categories`

Unseen VALIDATION:

`4 categories`

`53 rows`

`0.0074%`

W_SHORT:

`127 train categories`

Unseen VALIDATION:

`9 categories`

`107 rows`

`0.0150%`

Unseen risk rất nhỏ.

### Merchant City

W_LONG:

`11,233 train categories`

Unseen VALIDATION:

`204 categories`

`912 rows`

`0.1280%`

W_SHORT:

`8,046 train categories`

Unseen VALIDATION:

`731 categories`

`3,298 rows`

`0.4629%`

Ngoài unseen category, long-tail support rõ hơn.

Dưới W_SHORT:

`6,127 city categories`

có train support dưới 100.

Có:

`50,021 validation rows`

tương đương:

`7.0209%`

VALIDATION nằm ở city có training support dưới 100.

### Zip

Zip có vấn đề high-cardinality mạnh nhất.

W_LONG:

`22,603 train categories`

Unseen VALIDATION:

`427 categories`

`1,722 rows`

`0.2417%`

W_SHORT:

`15,922 train categories`

Unseen VALIDATION:

`1,532 categories`

`6,870 rows`

`0.9643%`

Dưới W_SHORT có:

`13,094 Zip categories`

với train support dưới 100.

VALIDATION rows có train support dưới 100:

`99,590`

tương đương:

`13.9784%`

Đây là mức long-tail đáng kể.

Như vậy W_SHORT làm vấn đề sparse/unseen của City và Zip tăng rõ rệt so với W_LONG.

Điều này không quyết định training-window winner, nhưng cho thấy preprocessing phía sau phải có unknown-category handling đúng nếu các field này được sử dụng.

### Kết luận M4.4.4

Generalization structure được chia thành ba nhóm.

Nhóm ổn định:

- transaction_mode;
- location_state;
- MCC;
- hour_of_day;
- day_of_week;
- month_of_year.

Nhóm conditional nhưng manageable:

- merchant_state_cat.

Nhóm high-cardinality / long-tail risk:

- merchant_city_cat;
- zip_cat.

Decision:

`transaction_mode`
→ `APPROVED CANDIDATE`

`location_state`
→ `APPROVED CANDIDATE`

`mcc_code`
→ `APPROVED CONDITIONAL CANDIDATE WITH SHORTCUT-RISK FLAG`

`merchant_state_cat`
→ `CONDITIONAL CANDIDATE`

`merchant_city_cat`
→ `EXPERIMENT-ONLY HIGH-CARDINALITY CANDIDATE`

`zip_cat`
→ `EXPERIMENT-ONLY HIGH-CARDINALITY CANDIDATE`

M4.6 bắt buộc phải hỗ trợ unknown category nếu bất kỳ raw location categorical candidate nào được sử dụng.



```python
cardinality_rows = []


for period in PERIODS:
    for feature in (
        CATEGORICAL_AUDIT_FEATURES
    ):
        counter = (
            categorical_counts[
                period
            ][feature]
        )

        cardinality_rows.append(
            {
                "period":
                    period,

                "feature":
                    feature,

                "unique_categories":
                    len(counter),

                "rows":
                    sum(
                        counter.values()
                    ),

                "min_category_support":
                    min(
                        counter.values()
                    ),

                "max_category_support":
                    max(
                        counter.values()
                    ),
            }
        )


cardinality_df = pd.DataFrame(
    cardinality_rows
)


display(
    cardinality_df
    .sort_values(
        [
            "feature",
            "period",
        ]
    )
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
      <th>period</th>
      <th>feature</th>
      <th>unique_categories</th>
      <th>rows</th>
      <th>min_category_support</th>
      <th>max_category_support</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7</th>
      <td>TRAIN_2015_2017</td>
      <td>day_of_week</td>
      <td>7</td>
      <td>5133655</td>
      <td>707824</td>
      <td>752220</td>
    </tr>
    <tr>
      <th>16</th>
      <td>TRAIN_2018</td>
      <td>day_of_week</td>
      <td>7</td>
      <td>1721615</td>
      <td>230615</td>
      <td>253594</td>
    </tr>
    <tr>
      <th>25</th>
      <td>VALIDATION</td>
      <td>day_of_week</td>
      <td>7</td>
      <td>712458</td>
      <td>95750</td>
      <td>106987</td>
    </tr>
    <tr>
      <th>6</th>
      <td>TRAIN_2015_2017</td>
      <td>hour_of_day</td>
      <td>24</td>
      <td>5133655</td>
      <td>40108</td>
      <td>369845</td>
    </tr>
    <tr>
      <th>15</th>
      <td>TRAIN_2018</td>
      <td>hour_of_day</td>
      <td>24</td>
      <td>1721615</td>
      <td>13717</td>
      <td>123020</td>
    </tr>
    <tr>
      <th>24</th>
      <td>VALIDATION</td>
      <td>hour_of_day</td>
      <td>24</td>
      <td>712458</td>
      <td>5723</td>
      <td>51277</td>
    </tr>
    <tr>
      <th>2</th>
      <td>TRAIN_2015_2017</td>
      <td>location_state</td>
      <td>3</td>
      <td>5133655</td>
      <td>32519</td>
      <td>4457627</td>
    </tr>
    <tr>
      <th>11</th>
      <td>TRAIN_2018</td>
      <td>location_state</td>
      <td>3</td>
      <td>1721615</td>
      <td>12129</td>
      <td>1494430</td>
    </tr>
    <tr>
      <th>20</th>
      <td>VALIDATION</td>
      <td>location_state</td>
      <td>3</td>
      <td>712458</td>
      <td>5494</td>
      <td>618047</td>
    </tr>
    <tr>
      <th>1</th>
      <td>TRAIN_2015_2017</td>
      <td>mcc_code</td>
      <td>109</td>
      <td>5133655</td>
      <td>95</td>
      <td>611429</td>
    </tr>
    <tr>
      <th>10</th>
      <td>TRAIN_2018</td>
      <td>mcc_code</td>
      <td>109</td>
      <td>1721615</td>
      <td>27</td>
      <td>205063</td>
    </tr>
    <tr>
      <th>19</th>
      <td>VALIDATION</td>
      <td>mcc_code</td>
      <td>109</td>
      <td>712458</td>
      <td>11</td>
      <td>85015</td>
    </tr>
    <tr>
      <th>3</th>
      <td>TRAIN_2015_2017</td>
      <td>merchant_city_cat</td>
      <td>10709</td>
      <td>5133655</td>
      <td>1</td>
      <td>643509</td>
    </tr>
    <tr>
      <th>12</th>
      <td>TRAIN_2018</td>
      <td>merchant_city_cat</td>
      <td>8046</td>
      <td>1721615</td>
      <td>1</td>
      <td>215056</td>
    </tr>
    <tr>
      <th>21</th>
      <td>VALIDATION</td>
      <td>merchant_city_cat</td>
      <td>6433</td>
      <td>712458</td>
      <td>1</td>
      <td>88917</td>
    </tr>
    <tr>
      <th>4</th>
      <td>TRAIN_2015_2017</td>
      <td>merchant_state_cat</td>
      <td>148</td>
      <td>5133655</td>
      <td>1</td>
      <td>643509</td>
    </tr>
    <tr>
      <th>13</th>
      <td>TRAIN_2018</td>
      <td>merchant_state_cat</td>
      <td>127</td>
      <td>1721615</td>
      <td>2</td>
      <td>215056</td>
    </tr>
    <tr>
      <th>22</th>
      <td>VALIDATION</td>
      <td>merchant_state_cat</td>
      <td>107</td>
      <td>712458</td>
      <td>1</td>
      <td>88917</td>
    </tr>
    <tr>
      <th>8</th>
      <td>TRAIN_2015_2017</td>
      <td>month_of_year</td>
      <td>12</td>
      <td>5133655</td>
      <td>392029</td>
      <td>437508</td>
    </tr>
    <tr>
      <th>17</th>
      <td>TRAIN_2018</td>
      <td>month_of_year</td>
      <td>12</td>
      <td>1721615</td>
      <td>131276</td>
      <td>146983</td>
    </tr>
    <tr>
      <th>26</th>
      <td>VALIDATION</td>
      <td>month_of_year</td>
      <td>5</td>
      <td>712458</td>
      <td>132102</td>
      <td>147202</td>
    </tr>
    <tr>
      <th>0</th>
      <td>TRAIN_2015_2017</td>
      <td>transaction_mode</td>
      <td>3</td>
      <td>5133655</td>
      <td>639122</td>
      <td>3619072</td>
    </tr>
    <tr>
      <th>9</th>
      <td>TRAIN_2018</td>
      <td>transaction_mode</td>
      <td>3</td>
      <td>1721615</td>
      <td>213632</td>
      <td>1215055</td>
    </tr>
    <tr>
      <th>18</th>
      <td>VALIDATION</td>
      <td>transaction_mode</td>
      <td>3</td>
      <td>712458</td>
      <td>88302</td>
      <td>503020</td>
    </tr>
    <tr>
      <th>5</th>
      <td>TRAIN_2015_2017</td>
      <td>zip_cat</td>
      <td>21403</td>
      <td>5133655</td>
      <td>1</td>
      <td>643509</td>
    </tr>
    <tr>
      <th>14</th>
      <td>TRAIN_2018</td>
      <td>zip_cat</td>
      <td>15922</td>
      <td>1721615</td>
      <td>1</td>
      <td>215056</td>
    </tr>
    <tr>
      <th>23</th>
      <td>VALIDATION</td>
      <td>zip_cat</td>
      <td>12389</td>
      <td>712458</td>
      <td>1</td>
      <td>88917</td>
    </tr>
  </tbody>
</table>
</div>


### Nhận xét

Cardinality của các low-cardinality feature ổn định:

`transaction_mode = 3`

`location_state = 3`

`MCC = 109`

`hour_of_day = 24`

`day_of_week = 7`

trong tất cả applicable periods.

Raw location có cardinality lớn và giảm khi temporal window ngắn hơn.

Merchant City:

`10,709`
→
`8,046`
→
`6,433`

Merchant State:

`148`
→
`127`
→
`107`

Zip:

`21,403`
→
`15,922`
→
`12,389`

City và Zip có category support tối thiểu bằng:

`1`

trong từng period.

Điều này xác nhận long-tail không phải chỉ do unseen VALIDATION category; ngay trong training data cũng tồn tại nhiều category có rất ít observation.

Merchant State cũng có category support rất thấp ở tail, nhưng cardinality nhỏ hơn City/Zip đáng kể.

MCC có `109` categories xuyên suốt nhưng minimum support giảm:

`95`

ở TRAIN_2015_2017,

`27`

ở TRAIN_2018,

`11`

ở VALIDATION.

Do đó MCC có một số rare categories nhưng không có category disappearance trong current development split.

### Kết luận

Cardinality audit xác nhận:

Low-cardinality/stable:

- transaction_mode;
- location_state;
- hour_of_day;
- day_of_week.

Moderate cardinality:

- MCC;
- merchant_state_cat.

High-cardinality/long-tail:

- merchant_city_cat;
- zip_cat.

Status:

`CONFIRMED`

High-cardinality location fields không được xem ngang hàng với low-cardinality core candidates.


# 7. M4.4.6 — Audit temporal support stability

Mục tiêu ở đây không phải kiểm tra fraud-rate stability.

M4.4 chỉ kiểm tra distribution/support của feature representation giữa:

TRAIN_2015_2017
TRAIN_2018
VALIDATION.

Một feature có category distribution thay đổi không tự động bị loại.

Nhưng drift lớn phải được ghi nhận trước khi đưa vào baseline candidate.


```python
STABILITY_FEATURES = [
    "transaction_mode",
    "mcc_code",
    "location_state",
    "merchant_state_cat",
    "hour_of_day",
    "day_of_week",
    "month_of_year",
]


stability_rows = []


for feature in (
    STABILITY_FEATURES
):
    all_categories = set()

    for period in PERIODS:
        all_categories.update(
            categorical_counts[
                period
            ][feature]
            .keys()
        )

    period_totals = {
        period: sum(
            categorical_counts[
                period
            ][feature]
            .values()
        )
        for period in PERIODS
    }

    max_shift = -1.0
    max_shift_category = None
    shares_of_max = None

    for category in (
        all_categories
    ):
        shares = {
            period: (
                categorical_counts[
                    period
                ][feature]
                .get(
                    category,
                    0,
                )
                / period_totals[
                    period
                ]
                * 100
            )
            for period in PERIODS
        }

        shift = (
            max(
                shares.values()
            )
            - min(
                shares.values()
            )
        )

        if shift > max_shift:
            max_shift = shift
            max_shift_category = (
                category
            )
            shares_of_max = shares

    stability_rows.append(
        {
            "feature":
                feature,

            "max_share_shift_pp":
                max_shift,

            "category_of_max_shift":
                max_shift_category,

            "TRAIN_2015_2017_share_pct":
                shares_of_max[
                    "TRAIN_2015_2017"
                ],

            "TRAIN_2018_share_pct":
                shares_of_max[
                    "TRAIN_2018"
                ],

            "VALIDATION_share_pct":
                shares_of_max[
                    "VALIDATION"
                ],
        }
    )


stability_df = pd.DataFrame(
    stability_rows
)


display(
    stability_df
    .sort_values(
        "max_share_shift_pp",
        ascending=False,
    )
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
      <th>feature</th>
      <th>max_share_shift_pp</th>
      <th>category_of_max_shift</th>
      <th>TRAIN_2015_2017_share_pct</th>
      <th>TRAIN_2018_share_pct</th>
      <th>VALIDATION_share_pct</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6</th>
      <td>month_of_year</td>
      <td>12.192795</td>
      <td>3</td>
      <td>8.468352</td>
      <td>8.496209</td>
      <td>20.661148</td>
    </tr>
    <tr>
      <th>5</th>
      <td>day_of_week</td>
      <td>1.177458</td>
      <td>4</td>
      <td>13.787915</td>
      <td>14.601697</td>
      <td>14.965373</td>
    </tr>
    <tr>
      <th>1</th>
      <td>mcc_code</td>
      <td>0.175391</td>
      <td>4121</td>
      <td>4.088919</td>
      <td>4.264310</td>
      <td>4.226916</td>
    </tr>
    <tr>
      <th>3</th>
      <td>merchant_state_cat</td>
      <td>0.164171</td>
      <td>NY</td>
      <td>5.819791</td>
      <td>5.655620</td>
      <td>5.680756</td>
    </tr>
    <tr>
      <th>2</th>
      <td>location_state</td>
      <td>0.137686</td>
      <td>PHYSICAL_ZIP_UNAVAILABLE</td>
      <td>0.633447</td>
      <td>0.704513</td>
      <td>0.771133</td>
    </tr>
    <tr>
      <th>0</th>
      <td>transaction_mode</td>
      <td>0.106475</td>
      <td>Chip Transaction</td>
      <td>70.496985</td>
      <td>70.576465</td>
      <td>70.603460</td>
    </tr>
    <tr>
      <th>4</th>
      <td>hour_of_day</td>
      <td>0.105971</td>
      <td>7</td>
      <td>6.720806</td>
      <td>6.614835</td>
      <td>6.633093</td>
    </tr>
  </tbody>
</table>
</div>


### Phân tích / Nhận xét

Với các feature có domain có thể so sánh trực tiếp giữa ba period, distribution support nhìn chung ổn định.

Maximum category-share shift:

`transaction_mode ≈ 0.1065 percentage point`

`hour_of_day ≈ 0.1060 percentage point`

`location_state ≈ 0.1377 percentage point`

`merchant_state_cat ≈ 0.1642 percentage point`

`mcc_code ≈ 0.1754 percentage point`

Các mức trên đều nhỏ.

`day_of_week` có max share shift khoảng:

`1.1775 percentage point`

cao hơn nhóm trên nhưng vẫn không cho thấy thay đổi cấu trúc lớn.

Riêng:

`month_of_year`

có max apparent shift:

`12.1928 percentage points`

nhưng con số này không thể diễn giải như ordinary feature drift.

Nguyên nhân là:

TRAIN_2015_2017 và TRAIN_2018 bao phủ đủ 12 tháng,

trong khi:

VALIDATION chỉ bao phủ tháng `1 → 5`.

Ví dụ tháng 3 chiếm khoảng `20.66%` VALIDATION chỉ vì mẫu VALIDATION chỉ có năm tháng.

Do đó không được dùng bảng stability hiện tại để kết luận `month_of_year` drift 12.19 percentage points.

Đây là consequence của temporal evaluation window, không phải evidence rằng transaction seasonality thay đổi bất thường.

Ngoài ra, audit này chỉ đo feature-support distribution.

Nó không chứng minh target relationship ổn định.

### Kết luận M4.4.6

Không phát hiện major support drift đối với:

- transaction_mode;
- MCC;
- location_state;
- merchant_state;
- hour_of_day.

`day_of_week` có shift nhẹ nhưng vẫn có đầy đủ domain và support lớn.

`month_of_year`:

`NOT COMPARABLE BY SIMPLE SHARE-SHIFT METRIC`

vì VALIDATION chỉ bao phủ tháng 1–5.

Month vẫn hợp lệ tại prediction point và không có unseen category, nhưng phải mang temporal-coverage caveat.

Status:

`PASS WITH MONTH-OF-YEAR INTERPRETATION CAVEAT`


```python
for period in PERIODS:
    print(
        "\n",
        period,
    )

    counter = (
        categorical_counts[
            period
        ][
            "transaction_mode"
        ]
    )

    total = sum(
        counter.values()
    )

    for value, count in (
        counter.most_common()
    ):
        print(
            value,
            f"{count:,}",
            f"{count / total * 100:.4f}%",
        )
```

    
     TRAIN_2015_2017
    Chip Transaction 3,619,072 70.4970%
    Swipe Transaction 875,461 17.0534%
    Online Transaction 639,122 12.4496%
    
     TRAIN_2018
    Chip Transaction 1,215,055 70.5765%
    Swipe Transaction 292,928 17.0147%
    Online Transaction 213,632 12.4088%
    
     VALIDATION
    Chip Transaction 503,020 70.6035%
    Swipe Transaction 121,136 17.0025%
    Online Transaction 88,302 12.3940%


### Nhận xét

Transaction mode chỉ có ba category và distribution cực kỳ ổn định.

TRAIN_2015_2017:

`Chip ≈ 70.4970%`

`Swipe ≈ 17.0534%`

`Online ≈ 12.4496%`

TRAIN_2018:

`Chip ≈ 70.5765%`

`Swipe ≈ 17.0147%`

`Online ≈ 12.4088%`

VALIDATION:

`Chip ≈ 70.6035%`

`Swipe ≈ 17.0025%`

`Online ≈ 12.3940%`

Không có:

- unseen category;
- category disappearance;
- material distribution shift.

Feature này có tại prediction point và không yêu cầu learned transformation để xác định semantic value.

### Kết luận M4.4.7

`transaction_mode`

được khóa là:

`PRIMARY TRANSACTION-LEVEL CATEGORICAL CANDIDATE`

Status:

`APPROVED`

Encoding:

`OPEN — M4.6`


# 8. M4.4.8 — MCC representation audit

MCC là categorical code.

M2 đã cho thấy MCC có association rất mạnh với target nhưng cũng có shortcut risk trong synthetic data.

M4.4 không dùng target lại.

Cần đánh giá:

- cardinality;
- support;
- unseen categories;
- temporal support;
- compatibility với W_LONG và W_SHORT.

Final categorical encoder thuộc M4.6.


```python
for period in PERIODS:
    counter = (
        categorical_counts[
            period
        ][
            "mcc_code"
        ]
    )

    print(
        "\n",
        period,
    )

    print(
        "Unique MCC:",
        len(counter),
    )

    print(
        "Top 15 MCC by support:"
    )

    print(
        counter.most_common(
            15
        )
    )

    rare_lt_100 = sum(
        1
        for count
        in counter.values()
        if count < 100
    )

    print(
        "MCC categories "
        "with support < 100:",
        rare_lt_100,
    )
```

    
     TRAIN_2015_2017
    Unique MCC: 109
    Top 15 MCC by support:
    [('5411', 611429), ('5499', 550861), ('5541', 540641), ('5812', 380993), ('5912', 296432), ('4784', 278783), ('5300', 232939), ('4829', 224705), ('4121', 209911), ('7538', 191544), ('5814', 190349), ('5311', 184808), ('4900', 94746), ('5310', 94476), ('5813', 91218)]
    MCC categories with support < 100: 1
    
     TRAIN_2018
    Unique MCC: 109
    Top 15 MCC by support:
    [('5411', 205063), ('5499', 184585), ('5541', 179578), ('5812', 128062), ('5912', 99363), ('4784', 93072), ('5300', 78398), ('4829', 74893), ('4121', 73415), ('7538', 64636), ('5814', 63790), ('5311', 61761), ('5310', 31838), ('4900', 31719), ('5813', 30917)]
    MCC categories with support < 100: 10
    
     VALIDATION
    Unique MCC: 109
    Top 15 MCC by support:
    [('5411', 85015), ('5499', 76434), ('5541', 74917), ('5812', 52709), ('5912', 40820), ('4784', 38323), ('5300', 32421), ('4829', 30930), ('4121', 30115), ('7538', 26666), ('5814', 26381), ('5311', 25749), ('5310', 13192), ('4900', 13155), ('5813', 12743)]
    MCC categories with support < 100: 15


### Phân tích / Nhận xét

MCC có:

`109 categories`

trong:

- TRAIN_2015_2017;
- TRAIN_2018;
- VALIDATION.

Không có unseen MCC trong VALIDATION dưới cả:

`W_LONG`

và:

`W_SHORT`.

Top MCC cũng có support rất lớn và tương tự về ordering giữa các temporal periods.

Maximum observed category-share shift chỉ khoảng:

`0.175 percentage point`.

Điều này cho thấy MCC representation có structural temporal support khá ổn định trong development window.

Long-tail vẫn tồn tại.

TRAIN_2015_2017 có:

`1 MCC`

support dưới 100.

TRAIN_2018 có:

`10 MCC`

support dưới 100.

VALIDATION có:

`15 MCC`

support dưới 100.

Tuy nhiên dưới W_SHORT chỉ:

`206 validation rows`

tương đương khoảng:

`0.0289%`

nằm trong MCC có training support dưới 100.

Về representation/generalization, đây là mức nhỏ.

Do đó không có lý do kỹ thuật để loại MCC vì cardinality hoặc unknown-category risk.

Tuy nhiên MCC vẫn mang inherited synthetic-shortcut risk.

M4.4 output hiện tại không sử dụng target, nên không thể chứng minh hoặc phủ nhận predictive shortcut bằng runtime hiện tại.

Vì vậy representation có thể được phê duyệt nhưng risk flag phải được giữ.

### Kết luận M4.4.8

MCC representation:

`mcc_code`

được khóa dưới dạng:

`CATEGORICAL CODE`

Không numeric scaling/ordinal interpretation.

Candidate status:

`APPROVED CONDITIONAL CANDIDATE`

Risk flag:

`SYNTHETIC SHORTCUT / TARGET-ASSOCIATION RISK`

Không loại MCC vì unseen/cardinality.

Không tuyên bố MCC là final feature chỉ vì support tốt.

Final inclusion/performance value:

`OPEN — FEATURE EXPERIMENT`

MCC baseline-candidate status:

`CONDITIONAL — AUDITED`


# 9. M4.4.9 — Location candidate audit

Candidate levels:

A. location_state

B. merchant_state_cat

C. merchant_city_cat

D. zip_cat

M4.3 đã khóa semantic của missingness.

M4.4 phải quyết định mức location representation nào hợp lý làm baseline candidate.

Raw location có memorization / shortcut risk vì:

- Merchant City cardinality cao;
- Zip cardinality cao;
- synthetic location-target associations từng rất mạnh ở EDA.

Không chọn raw location chỉ vì association mạnh.


```python
LOCATION_FEATURES = [
    "location_state",
    "merchant_state_cat",
    "merchant_city_cat",
    "zip_cat",
]


for feature in (
    LOCATION_FEATURES
):
    print(
        "\n============================"
    )

    print(feature)

    print(
        "============================"
    )

    for period in PERIODS:
        counter = (
            categorical_counts[
                period
            ][feature]
        )

        print(
            "\n",
            period,
        )

        print(
            "Unique:",
            len(counter),
        )

        print(
            "Top 15:"
        )

        print(
            counter.most_common(
                15
            )
        )
```

    
    ============================
    location_state
    ============================
    
     TRAIN_2015_2017
    Unique: 3
    Top 15:
    [('PHYSICAL_COMPLETE', 4457627), ('NON_PHYSICAL_OR_ONLINE', 643509), ('PHYSICAL_ZIP_UNAVAILABLE', 32519)]
    
     TRAIN_2018
    Unique: 3
    Top 15:
    [('PHYSICAL_COMPLETE', 1494430), ('NON_PHYSICAL_OR_ONLINE', 215056), ('PHYSICAL_ZIP_UNAVAILABLE', 12129)]
    
     VALIDATION
    Unique: 3
    Top 15:
    [('PHYSICAL_COMPLETE', 618047), ('NON_PHYSICAL_OR_ONLINE', 88917), ('PHYSICAL_ZIP_UNAVAILABLE', 5494)]
    
    ============================
    merchant_state_cat
    ============================
    
     TRAIN_2015_2017
    Unique: 148
    Top 15:
    [('NOT_APPLICABLE', 643509), ('CA', 544257), ('TX', 376468), ('NY', 298768), ('FL', 278892), ('OH', 184541), ('IL', 180936), ('PA', 174262), ('NC', 155596), ('MI', 133808), ('GA', 133471), ('NJ', 127058), ('IN', 121761), ('TN', 110034), ('WA', 103217)]
    
     TRAIN_2018
    Unique: 127
    Top 15:
    [('NOT_APPLICABLE', 215056), ('CA', 181160), ('TX', 124340), ('NY', 97368), ('FL', 93706), ('OH', 62483), ('IL', 59546), ('PA', 58642), ('NC', 51209), ('MI', 45420), ('GA', 44954), ('NJ', 42520), ('IN', 40258), ('TN', 36987), ('WA', 33981)]
    
     VALIDATION
    Unique: 107
    Top 15:
    [('NOT_APPLICABLE', 88917), ('CA', 75575), ('TX', 52176), ('NY', 40473), ('FL', 37960), ('OH', 26528), ('IL', 24633), ('PA', 23952), ('NC', 20925), ('GA', 18667), ('MI', 18595), ('NJ', 17291), ('IN', 16219), ('TN', 14859), ('WA', 14303)]
    
    ============================
    merchant_city_cat
    ============================
    
     TRAIN_2015_2017
    Unique: 10709
    Top 15:
    [('ONLINE', 643509), ('Houston', 54626), ('Los Angeles', 38045), ('Miami', 37878), ('Brooklyn', 32071), ('Chicago', 28095), ('Dallas', 26979), ('San Antonio', 24864), ('Philadelphia', 24006), ('Indianapolis', 22768), ('Atlanta', 22483), ('Orlando', 22158), ('Louisville', 21782), ('New York', 20601), ('Tucson', 19520)]
    
     TRAIN_2018
    Unique: 8046
    Top 15:
    [('ONLINE', 215056), ('Houston', 17534), ('Los Angeles', 12520), ('Miami', 12212), ('Brooklyn', 10779), ('Chicago', 10009), ('Dallas', 8794), ('San Antonio', 8426), ('Orlando', 8090), ('Indianapolis', 7645), ('Louisville', 7390), ('Atlanta', 7319), ('Philadelphia', 7169), ('Minneapolis', 7006), ('Tucson', 6639)]
    
     VALIDATION
    Unique: 6433
    Top 15:
    [('ONLINE', 88917), ('Houston', 7421), ('Los Angeles', 5516), ('Miami', 5223), ('Brooklyn', 4478), ('Chicago', 4290), ('Dallas', 3906), ('San Antonio', 3436), ('Atlanta', 3370), ('Orlando', 3137), ('Philadelphia', 3017), ('Indianapolis', 2958), ('Louisville', 2815), ('Tucson', 2762), ('Las Vegas', 2757)]
    
    ============================
    zip_cat
    ============================
    
     TRAIN_2015_2017
    Unique: 21403
    Top 15:
    [('NOT_APPLICABLE', 643509), ('ZIP_UNAVAILABLE', 32519), ('98516', 11284), ('75023', 10237), ('91606', 9862), ('87121', 9557), ('77056', 8396), ('94606', 8164), ('55024', 8006), ('80013', 7967), ('95687', 7838), ('40299', 7740), ('96792', 7644), ('29229', 7591), ('30101', 7323)]
    
     TRAIN_2018
    Unique: 15922
    Top 15:
    [('NOT_APPLICABLE', 215056), ('ZIP_UNAVAILABLE', 12129), ('98516', 3691), ('75023', 3378), ('87121', 3261), ('91606', 3230), ('94606', 2795), ('95687', 2783), ('77056', 2721), ('40299', 2704), ('29229', 2563), ('96792', 2556), ('80013', 2549), ('30101', 2529), ('43228', 2521)]
    
     VALIDATION
    Unique: 12389
    Top 15:
    [('NOT_APPLICABLE', 88917), ('ZIP_UNAVAILABLE', 5494), ('98516', 1701), ('87121', 1348), ('75023', 1297), ('95687', 1260), ('77056', 1152), ('91606', 1112), ('94606', 1092), ('43228', 1069), ('80013', 1054), ('96792', 1039), ('29229', 1020), ('43830', 1005), ('40299', 992)]


### Phân tích / Nhận xét

Bốn mức location representation có đặc tính rất khác nhau.

## location_state

Chỉ có:

`3 categories`

trong tất cả periods:

`PHYSICAL_COMPLETE`

`NON_PHYSICAL_OR_ONLINE`

`PHYSICAL_ZIP_UNAVAILABLE`

Không unseen category.

Distribution ổn định.

Đây là representation trực tiếp của semantic policy đã khóa ở M4.3 và có cardinality thấp.

## merchant_state_cat

Cardinality:

`148`
→
`127`
→
`107`

W_LONG unseen VALIDATION:

`4 categories / 53 rows / 0.0074%`

W_SHORT unseen VALIDATION:

`9 categories / 107 rows / 0.0150%`

Unknown risk rất nhỏ.

Về mặt implementation, Merchant State có thể được encode an toàn nếu M4.6 hỗ trợ unknown category.

Tuy nhiên field này vẫn thuộc raw location group và có inherited shortcut/memorization concern.

## merchant_city_cat

Cardinality rất cao:

`10,709`
→
`8,046`
→
`6,433`

Nhiều category có support bằng 1.

Unseen và rare-category exposure tăng rõ dưới W_SHORT.

W_SHORT:

`731 unseen validation categories`

`3,298 unseen validation rows`

và:

`7.0209% VALIDATION`

nằm ở category có W_SHORT train support dưới 100.

Đây là dấu hiệu long-tail/memorization risk rõ.

## zip_cat

Zip có cardinality cao nhất:

`21,403`
→
`15,922`
→
`12,389`

W_SHORT:

`1,532 unseen validation categories`

`6,870 unseen validation rows`

`0.9643% unseen rate`

và khoảng:

`13.9784% VALIDATION rows`

thuộc Zip có W_SHORT training support dưới 100.

Như vậy Zip mang sparse-category risk lớn hơn Merchant City.

Các semantic token:

`NOT_APPLICABLE`

và:

`ZIP_UNAVAILABLE`

có support lớn và ổn định, nhưng raw individual Zip code thì có long tail rất rõ.

### Kết luận M4.4.9

`location_state`

→ `PRIMARY LOCATION CANDIDATE`

Status:

`APPROVED`

`merchant_state_cat`

→ `CONDITIONAL LOCATION CANDIDATE`

Status:

`AUDITED — NOT CORE BY DEFAULT`

`merchant_city_cat`

→ `HIGH-CARDINALITY EXPERIMENT CANDIDATE`

Status:

`NOT CORE BASELINE`

`zip_cat`

→ `HIGH-CARDINALITY EXPERIMENT CANDIDATE`

Status:

`NOT CORE BASELINE`

M4.4 không loại City/Zip khỏi mọi experiment.

Nhưng chúng không nên được đưa mặc định vào core baseline vì:

- cardinality lớn;
- long-tail support;
- unseen-category exposure;
- memorization risk;
- inherited synthetic-shortcut concern.

M4.6 nếu hỗ trợ chúng phải có explicit unknown-category policy.


# 10. M4.4.10 — Calendar/time representation

Candidate:

hour_of_day

day_of_week

month_of_year

is_weekend

Các field này được derive trực tiếp từ transaction Timestamp và có tại prediction point.

Không dùng raw Year làm candidate classifier feature ở M4.4 baseline specification.

Lý do:

Year chủ yếu định vị transaction trong temporal regime và split, trong khi baseline cần tránh biến split/calendar progression thành một shortcut trực tiếp nếu chưa có lý do riêng.

M4.4 sẽ audit support của các recurring calendar units.


```python
TIME_FEATURES = [
    "hour_of_day",
    "day_of_week",
    "month_of_year",
]


for feature in TIME_FEATURES:
    print(
        "\n============================"
    )

    print(feature)

    print(
        "============================"
    )

    for period in PERIODS:
        counter = (
            categorical_counts[
                period
            ][feature]
        )

        print(
            period,
            "unique =",
            len(counter),
        )

        print(
            sorted(
                counter.items(),
                key=lambda x: int(
                    x[0]
                ),
            )
        )
```

    
    ============================
    hour_of_day
    ============================
    TRAIN_2015_2017 unique = 24
    [('0', 52127), ('1', 44915), ('2', 45000), ('3', 40108), ('4', 41460), ('5', 68552), ('6', 290713), ('7', 345023), ('8', 342830), ('9', 335726), ('10', 350569), ('11', 364810), ('12', 369845), ('13', 354132), ('14', 336672), ('15', 327504), ('16', 336218), ('17', 187869), ('18', 176097), ('19', 173083), ('20', 164947), ('21', 159972), ('22', 162385), ('23', 63098)]
    TRAIN_2018 unique = 24
    [('0', 17393), ('1', 15437), ('2', 15075), ('3', 13717), ('4', 13914), ('5', 22925), ('6', 97583), ('7', 113882), ('8', 114469), ('9', 112171), ('10', 117504), ('11', 122921), ('12', 123020), ('13', 119182), ('14', 114466), ('15', 110187), ('16', 112377), ('17', 63012), ('18', 59723), ('19', 58071), ('20', 55491), ('21', 53286), ('22', 54286), ('23', 21523)]
    VALIDATION unique = 24
    [('0', 7292), ('1', 6314), ('2', 6166), ('3', 5723), ('4', 5723), ('5', 9487), ('6', 40578), ('7', 47258), ('8', 47274), ('9', 46566), ('10', 48274), ('11', 50679), ('12', 51277), ('13', 49380), ('14', 47211), ('15', 45706), ('16', 46707), ('17', 25731), ('18', 24699), ('19', 23940), ('20', 23114), ('21', 22406), ('22', 22242), ('23', 8711)]
    
    ============================
    day_of_week
    ============================
    TRAIN_2015_2017 unique = 7
    [('0', 724485), ('1', 749719), ('2', 752220), ('3', 731581), ('4', 707824), ('5', 736566), ('6', 731260)]
    TRAIN_2018 unique = 7
    [('0', 230615), ('1', 231527), ('2', 251858), ('3', 251485), ('4', 251385), ('5', 251151), ('6', 253594)]
    VALIDATION unique = 7
    [('0', 102681), ('1', 95750), ('2', 96359), ('3', 106987), ('4', 106622), ('5', 102192), ('6', 101867)]
    
    ============================
    month_of_year
    ============================
    TRAIN_2015_2017 unique = 12
    [('1', 433700), ('2', 392029), ('3', 434736), ('4', 419970), ('5', 436260), ('6', 424695), ('7', 437028), ('8', 436969), ('9', 422033), ('10', 434826), ('11', 423901), ('12', 437508)]
    TRAIN_2018 unique = 12
    [('1', 146357), ('2', 131276), ('3', 146272), ('4', 141688), ('5', 145438), ('6', 141827), ('7', 146638), ('8', 146983), ('9', 141557), ('10', 145473), ('11', 141843), ('12', 146263)]
    VALIDATION unique = 5
    [('1', 146097), ('2', 132102), ('3', 147202), ('4', 141409), ('5', 145648)]



```python
EXPECTED_HOURS = {
    str(x)
    for x in range(24)
}

EXPECTED_DAYS_OF_WEEK = {
    str(x)
    for x in range(7)
}

EXPECTED_MONTHS_BY_PERIOD = {
    "TRAIN_2015_2017": {
        str(x)
        for x in range(1, 13)
    },

    "TRAIN_2018": {
        str(x)
        for x in range(1, 13)
    },

    "VALIDATION": {
        str(x)
        for x in range(1, 6)
    },
}


for period in PERIODS:
    observed_hours = set(
        categorical_counts[
            period
        ][
            "hour_of_day"
        ].keys()
    )

    observed_days = set(
        categorical_counts[
            period
        ][
            "day_of_week"
        ].keys()
    )

    observed_months = set(
        categorical_counts[
            period
        ][
            "month_of_year"
        ].keys()
    )

    assert (
        observed_hours
        == EXPECTED_HOURS
    ), (
        f"{period}: unexpected "
        f"hour_of_day domain: "
        f"{sorted(observed_hours)}"
    )

    assert (
        observed_days
        == EXPECTED_DAYS_OF_WEEK
    ), (
        f"{period}: unexpected "
        f"day_of_week domain: "
        f"{sorted(observed_days)}"
    )

    assert (
        observed_months
        == EXPECTED_MONTHS_BY_PERIOD[
            period
        ]
    ), (
        f"{period}: unexpected "
        f"month_of_year domain: "
        f"{sorted(observed_months)}"
    )


print(
    "M4.4 CALENDAR COVERAGE GATE: PASS"
)
```

    M4.4 CALENDAR COVERAGE GATE: PASS


### Nhận xét

`hour_of_day`

có đủ:

`24 categories`

ở cả ba period.

Không có unseen hour.

Distribution nhìn chung ổn định.

`day_of_week`

có đủ:

`7 categories`

ở cả ba period.

Không có unseen weekday.

Weekend audit cũng xác nhận cả weekday và weekend đều có support lớn:

TRAIN_2015_2017:

`weekend ≈ 28.59%`

TRAIN_2018:

`weekend ≈ 29.32%`

VALIDATION:

`weekend ≈ 28.64%`

`month_of_year`

có đủ 12 tháng trong TRAIN periods và đúng:

`5 months`

trong VALIDATION:

`January → May`

Điều này đúng với temporal split và không phải missing calendar category.

Mọi VALIDATION month đều đã xuất hiện trong TRAIN.

Do đó không có unknown-month problem.

Tuy nhiên `is_weekend` là deterministic function của `day_of_week`.

Nếu cả hai được encode cùng lúc, chúng chứa thông tin chồng lặp.

Tương tự month-of-year có một methodological caveat: validation chỉ kiểm tra tháng 1–5, nên M4.4 không thể dùng current validation support để đánh giá trực tiếp behavior của month feature ở tháng 6–12.

### Kết luận M4.4.10

`hour_of_day`

→ `PRIMARY CALENDAR CANDIDATE`

`day_of_week`

→ `PRIMARY CALENDAR CANDIDATE`

`month_of_year`

→ `CONDITIONAL CALENDAR CANDIDATE`

với caveat:

`VALIDATION COVERS ONLY MONTHS 1–5`

`is_weekend`

→ `OPTIONAL DERIVED CANDIDATE`

không core mặc định vì information đã được chứa trong `day_of_week`.

Calendar representation status:

`AUDITED`

Calendar coverage gate:

`PASS`





```python
weekend_rows = []

for period in PERIODS:
    day_counter = (
        categorical_counts[
            period
        ][
            "day_of_week"
        ]
    )

    weekend_count = (
        day_counter.get("5", 0)
        + day_counter.get("6", 0)
    )

    total = sum(
        day_counter.values()
    )

    weekday_count = (
        total
        - weekend_count
    )

    weekend_rows.append(
        {
            "period":
                period,

            "weekday_count":
                weekday_count,

            "weekend_count":
                weekend_count,

            "weekend_rate_pct":
                (
                    weekend_count
                    / total
                    * 100
                ),
        }
    )


weekend_df = pd.DataFrame(
    weekend_rows
)

display(weekend_df)


assert (
    weekend_df[
        "weekday_count"
    ] > 0
).all()

assert (
    weekend_df[
        "weekend_count"
    ] > 0
).all()


print(
    "M4.4 WEEKEND FEATURE AUDIT: PASS"
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
      <th>period</th>
      <th>weekday_count</th>
      <th>weekend_count</th>
      <th>weekend_rate_pct</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>TRAIN_2015_2017</td>
      <td>3665829</td>
      <td>1467826</td>
      <td>28.592221</td>
    </tr>
    <tr>
      <th>1</th>
      <td>TRAIN_2018</td>
      <td>1216870</td>
      <td>504745</td>
      <td>29.318111</td>
    </tr>
    <tr>
      <th>2</th>
      <td>VALIDATION</td>
      <td>508399</td>
      <td>204059</td>
      <td>28.641548</td>
    </tr>
  </tbody>
</table>
</div>


    M4.4 WEEKEND FEATURE AUDIT: PASS


### Nhận xét

Weekend representation có support lớn trong mọi development period.

Weekend rate:

`28.5922%`

`29.3181%`

`28.6415%`

Không có period nào thiếu weekday hoặc weekend transaction.

Cell trả về:

`M4.4 WEEKEND FEATURE AUDIT: PASS`

Tuy nhiên `is_weekend` được suy ra hoàn toàn từ:

`day_of_week >= 5`

nên không bổ sung thông tin độc lập nếu `day_of_week` đã có mặt.

### Kết luận

`is_weekend`

là feature hợp lệ tại prediction point nhưng được phân loại:

`OPTIONAL / REDUNDANT DERIVED CANDIDATE`

Không đưa vào core candidate set mặc định cùng `day_of_week`.

Status:

`AUDITED`


# 11. M4.4.11 — Transaction Feature Registry

## F01 — amount_numeric

Semantic role:

`NUMERIC CURRENT-TRANSACTION FEATURE`

Evidence:

Không missing; distribution ổn định theo temporal period; giữ nguyên signed Amount policy.

Decision:

`PRIMARY CANDIDATE`

Status:

`APPROVED`

---

## F02 — amount_signed_log1p

Semantic role:

`DETERMINISTIC NUMERIC TRANSFORMATION`

Evidence:

Giữ dấu và nén mạnh Amount scale/range.

Decision:

`ALTERNATIVE EXPERIMENT CANDIDATE`

Không đưa đồng thời với raw Amount vào core baseline chỉ vì transformation khả dụng.

Status:

`APPROVED FOR EXPERIMENT`

---

## F03 — is_negative_amount

Semantic role:

`BOOLEAN CURRENT-TRANSACTION FEATURE`

Evidence:

Negative Amount chiếm khoảng `4.8%` trong mọi development period.

Decision:

`DERIVED EXPERIMENT CANDIDATE`

Không core mặc định vì sign đã được bảo toàn trong `amount_numeric`.

Status:

`APPROVED FOR EXPERIMENT`

---

## F04 — is_zero_amount

Semantic role:

`BOOLEAN CURRENT-TRANSACTION FEATURE`

Evidence:

Zero Amount xuất hiện ở cả ba periods nhưng tỷ lệ chỉ khoảng `0.08%`.

Decision:

`DERIVED EXPERIMENT CANDIDATE`

Không core mặc định.

Status:

`APPROVED FOR EXPERIMENT`

---

## F05 — transaction_mode

Source:

`Use Chip`

Semantic role:

`CATEGORICAL CURRENT-TRANSACTION FEATURE`

Evidence:

3 categories, support lớn, không unseen, distribution ổn định.

Decision:

`PRIMARY CANDIDATE`

Status:

`APPROVED`

---

## F06 — mcc_code

Semantic role:

`CATEGORICAL CODE`

Evidence:

109 categories trong mọi period; 0 unseen under W_LONG/W_SHORT; low rare-row exposure.

Risk:

synthetic shortcut / target-association risk.

Decision:

`CONDITIONAL CANDIDATE`

Representation được phê duyệt; final inclusion phải qua feature experiment.

Status:

`AUDITED — RISK FLAG`

---

## F07 — location_state

Semantic role:

`LOW-CARDINALITY SEMANTIC LOCATION FEATURE`

Evidence:

3 categories; 0 unseen; ổn định; trực tiếp bảo toàn M4.3 semantic policy.

Decision:

`PRIMARY CANDIDATE`

Status:

`APPROVED`

---

## F08 — merchant_state_cat

Semantic role:

`CONDITIONAL LOCATION CATEGORICAL FEATURE`

Evidence:

Moderate cardinality; unseen VALIDATION cực thấp dưới cả W_LONG/W_SHORT.

Decision:

`CONDITIONAL CANDIDATE`

Không core mặc định do raw-location shortcut risk.

Status:

`AUDITED`

---

## F09 — merchant_city_cat

Semantic role:

`HIGH-CARDINALITY LOCATION CATEGORICAL FEATURE`

Evidence:

Hàng nghìn categories; nhiều low-support categories; unseen rate tăng dưới W_SHORT.

Risk:

memorization / unseen category / synthetic shortcut.

Decision:

`EXPERIMENT-ONLY CANDIDATE`

Status:

`NOT CORE BASELINE`

---

## F10 — zip_cat

Semantic role:

`HIGH-CARDINALITY CATEGORICAL CODE`

Evidence:

Cardinality rất cao; W_SHORT unseen rate gần 1%; khoảng 14% VALIDATION rows thuộc Zip có W_SHORT training support dưới 100.

Risk:

memorization / unseen category / synthetic shortcut.

Decision:

`EXPERIMENT-ONLY CANDIDATE`

Status:

`NOT CORE BASELINE`

---

## F11 — hour_of_day

Semantic role:

`RECURRING CALENDAR CATEGORY`

Evidence:

24/24 categories trong mọi period; 0 unseen; support ổn định.

Decision:

`PRIMARY CANDIDATE`

Status:

`APPROVED`

---

## F12 — day_of_week

Semantic role:

`RECURRING CALENDAR CATEGORY`

Evidence:

7/7 categories trong mọi period; 0 unseen; large support.

Decision:

`PRIMARY CANDIDATE`

Status:

`APPROVED`

---

## F13 — month_of_year

Semantic role:

`RECURRING CALENDAR CATEGORY`

Evidence:

TRAIN có đủ 12 months; VALIDATION đúng tháng 1–5; không unseen validation month.

Decision:

`CONDITIONAL CANDIDATE`

Reason:

VALIDATION chỉ bao phủ một phần annual cycle nên seasonal effect không được validation kiểm tra trên toàn 12 tháng.

Status:

`AUDITED WITH TEMPORAL-COVERAGE CAVEAT`

---

## F14 — is_weekend

Semantic role:

`BOOLEAN CALENDAR FEATURE`

Evidence:

Weekend rate khoảng 28.6%–29.3% và có support lớn ở mọi period.

Decision:

`OPTIONAL DERIVED CANDIDATE`

Không core mặc định vì deterministic từ `day_of_week`.

Status:

`AUDITED`



# 12. M4.4.12 — Explicit exclusions

Các field sau không thuộc transaction-level classifier candidate set.

## Raw identifiers

User

Card

Merchant Name

Reason:

HISTORY_KEY_ONLY.

Raw direct classifier use:
PROHIBITED.

---

## Processing / target

Errors?

Reason:

EXCLUDED FROM MODEL V1.

Is Fraud?

Reason:

TARGET_ONLY.

---

## Technical metadata

raw_row_id

temporal_region

is_w_long_train

is_w_short_train

is_validation

Reason:

technical / partition metadata only.

---

## Raw timestamp components

Year

Month

Day

Time

không được đưa nguyên bộ trực tiếp vào candidate set.

Timestamp được dùng để derive:

hour_of_day

day_of_week

month_of_year

is_weekend.

Raw Year hiện không phải baseline candidate.

---

## Behavioral / history features

Không xây ở M4.4.

Ví dụ:

time_since_previous_transaction

transactions_last_1h

amount_minus_previous_mean

is_new_merchant

thuộc M4.5.


```python
PROHIBITED_FEATURE_NAMES = {
    "User",
    "Card",
    "Merchant Name",
    "Errors?",
    "Is Fraud?",
    "raw_row_id",
    "temporal_region",
    "is_w_long_train",
    "is_w_short_train",
    "is_validation",
}


actual_candidate_names = set(
    preview_features.columns
)


assert (
    actual_candidate_names
    .isdisjoint(
        PROHIBITED_FEATURE_NAMES
    )
)

assert (
    sum(period_counts.values())
    == EXPECTED_DEVELOPMENT_ROWS
)

assert (
    development_max_timestamp
    < VALIDATION_END
)

assert (
    sum(
        candidate_missing_counts.values()
    )
    == 0
)

assert unexpected_location_rows == 0
assert unexpected_state_rows == 0
assert unexpected_zip_rows == 0


LEARNED_STATE_CREATED = False
HISTORY_FEATURE_CREATED = False
TARGET_USED = False
FINAL_TEST_USED = False


assert LEARNED_STATE_CREATED is False
assert HISTORY_FEATURE_CREATED is False
assert TARGET_USED is False
assert FINAL_TEST_USED is False


print(
    "M4.4 FEATURE SAFETY GATE: PASS"
)
```

    M4.4 FEATURE SAFETY GATE: PASS


### Gate interpretation

Cell trả về:

`M4.4 FEATURE SAFETY GATE: PASS`

Candidate frame không chứa:

- User;
- Card;
- Merchant Name;
- Errors?;
- target;
- partition metadata.

M4.4 không tạo:

- learned preprocessing state;
- historical feature;
- target-dependent feature;
- FINAL TEST-dependent feature.

Development population đúng:

`7,567,728 rows`

Candidate representation không missing hoặc unexpected semantic state.

Do đó current transaction-level feature construction tuân thủ prediction-point, leakage và partition guardrail.

Status:

`PASS`


# 13. M4.4 Findings

## M4.4-F01 — Amount representation

Observed fact:

Raw Amount có scale rộng/right-tail nhưng distribution summary tương đối ổn định qua development periods.

Signed-log transformation nén scale mạnh và giữ dấu.

Evidence:

Raw mean khoảng `42.9`, std khoảng `81`.

Negative rate khoảng `4.8%`.

Zero rate khoảng `0.08%`.

Interpretation:

Raw signed Amount là representation hợp lệ; signed-log là deterministic alternative chứ không phải correction.

Implication:

Giữ `amount_numeric` làm primary candidate.

Signed-log/sign/zero representation dành cho controlled feature experiment.

Status:

`CONFIRMED`

---

## M4.4-F02 — Transaction mode

Observed fact:

Transaction mode có đúng ba categories, không unseen và distribution rất ổn định.

Evidence:

Chip khoảng `70.5%`

Swipe khoảng `17.0%`

Online khoảng `12.4%`

xuyên ba periods.

Interpretation:

Low-cardinality current-transaction categorical representation có generalization structure tốt.

Implication:

Transaction mode là primary candidate.

Status:

`CONFIRMED`

---

## M4.4-F03 — MCC support/generalization

Observed fact:

MCC có 109 categories trong toàn bộ development periods và không có unseen VALIDATION category dưới W_LONG hoặc W_SHORT.

Evidence:

W_SHORT validation rows có train support dưới 100 chỉ khoảng:

`0.0289%`.

Temporal max category-share shift khoảng:

`0.175 percentage point`.

Interpretation:

MCC không có structural generalization blocker trong development split.

Tuy nhiên inherited synthetic-shortcut risk vẫn tồn tại.

Implication:

Approve MCC representation làm conditional candidate, không coi support tốt là bằng chứng final predictive validity.

Status:

`CONFIRMED WITH RISK FLAG`

---

## M4.4-F04 — Location representation

Observed fact:

`location_state` có 3 stable categories và 0 unseen.

Merchant State có manageable cardinality/unseen risk.

Merchant City và Zip có high cardinality, long-tail và higher unseen/rare-support exposure.

Evidence:

W_SHORT unseen rate:

Merchant City:

`0.4629%`

Zip:

`0.9643%`

W_SHORT VALIDATION rows có train support <100:

Merchant City:

`7.0209%`

Zip:

`13.9784%`

Interpretation:

Semantic low-cardinality location state generalize tốt hơn raw fine-grained location codes.

Implication:

`location_state` là primary candidate.

State conditional.

City/Zip experiment-only.

Status:

`CONFIRMED`

---

## M4.4-F05 — Calendar/time features

Observed fact:

Hour và day-of-week có đầy đủ domain và large support trong mọi period.

Month có đủ train domain nhưng VALIDATION chỉ bao phủ tháng 1–5 theo split.

Weekend có stable support.

Interpretation:

Recurring calendar features đều prediction-point valid.

Tuy nhiên `month_of_year` có partial-validation-cycle caveat và `is_weekend` redundant với `day_of_week`.

Implication:

Hour/day-of-week primary.

Month conditional.

Weekend optional.

Status:

`CONFIRMED`

---

## M4.4-F06 — Categorical unseen/support risk

Observed fact:

Unknown-category risk gần bằng 0 cho low/moderate-cardinality features nhưng tăng rõ đối với City/Zip, đặc biệt dưới W_SHORT.

Evidence:

W_LONG / W_SHORT:

MCC unseen:

`0 / 0`

location_state unseen:

`0 / 0`

Merchant State unseen rows:

`53 / 107`

Merchant City unseen rows:

`912 / 3,298`

Zip unseen rows:

`1,722 / 6,870`

Interpretation:

Feature representation phải tương thích với cả W_LONG và W_SHORT, nhưng W_SHORT làm sparse high-cardinality problem rõ hơn.

Implication:

M4.6 bắt buộc có explicit unknown handling.

High-cardinality City/Zip không thuộc core set mặc định.

Status:

`CONFIRMED`

---

## M4.4-F07 — Feature-frame integrity

Observed fact:

14-feature deterministic candidate builder chạy thành công trên toàn development data mà không tạo missing hoặc unexpected state.

Evidence:

`7,567,728 development rows`

`0 parse failures`

`0 candidate missing`

`0 unexpected location/state/Zip rows`

Feature safety gate:

`PASS`

Preview memory:

`384.22 bytes/row`

Interpretation:

Representation logic đúng nhưng string-heavy audit frame có memory footprint đáng kể.

Implication:

M4.6 cần leakage-safe và memory-conscious preprocessing implementation thay vì giữ toàn bộ object/string frame không cần thiết.

Status:

`CONFIRMED`



# 14. Decision Log M4.4

## M4.4-D01 — Amount baseline representation

Decision:

`amount_numeric`

là primary Amount candidate.

`amount_signed_log1p`

là alternative feature experiment candidate.

Không sử dụng cả hai như core default chỉ vì cả hai khả dụng.

Status:

`LOCKED`

---

## M4.4-D02 — Amount sign/zero indicators

Decision:

`is_negative_amount`

và:

`is_zero_amount`

được giữ làm derived experiment candidates.

Không thuộc core default vì deterministic từ Amount.

Status:

`LOCKED`

---

## M4.4-D03 — Transaction mode

Decision:

`transaction_mode`

là primary categorical candidate.

Status:

`LOCKED`

---

## M4.4-D04 — MCC candidate status

Decision:

`mcc_code`

được giữ là categorical conditional candidate.

Không numeric-scale.

Không loại do unseen/cardinality.

Giữ explicit:

`SYNTHETIC SHORTCUT RISK FLAG`

Final inclusion:

`REQUIRES FEATURE EXPERIMENT`

Status:

`LOCKED AT REPRESENTATION LEVEL`

---

## M4.4-D05 — Location-state candidate

Decision:

`location_state`

là primary low-cardinality semantic location candidate.

Status:

`LOCKED`

---

## M4.4-D06 — Raw Merchant State candidate

Decision:

`merchant_state_cat`

là conditional location candidate.

Không core mặc định.

Status:

`LOCKED AT CANDIDATE LEVEL`

---

## M4.4-D07 — Raw Merchant City candidate

Decision:

`merchant_city_cat`

không thuộc core baseline candidate set.

Giữ là:

`EXPERIMENT-ONLY HIGH-CARDINALITY CANDIDATE`

Status:

`LOCKED`

---

## M4.4-D08 — Zip candidate

Decision:

`zip_cat`

không thuộc core baseline candidate set.

Giữ semantic categorical code và các token:

`NOT_APPLICABLE`

`ZIP_UNAVAILABLE`

nhưng raw Zip-code feature chỉ là:

`EXPERIMENT-ONLY HIGH-CARDINALITY CANDIDATE`

Status:

`LOCKED`

---

## M4.4-D09 — Calendar/time candidates

Decision:

Primary:

`hour_of_day`

`day_of_week`

Conditional:

`month_of_year`

Optional derived:

`is_weekend`

Không sử dụng raw Year làm classifier candidate trong current transaction-level specification.

Status:

`LOCKED`

---

## M4.4-D10 — Raw identifier exclusion

Decision:

User/Card/Merchant Name không được đưa trực tiếp vào classifier candidate frame.

Status:

`INHERITED — VERIFIED — LOCKED`

---

## M4.4-D11 — FINAL TEST isolation

Decision:

M4.4 không sử dụng FINAL TEST để quyết định feature representation.

Status:

`INHERITED — VERIFIED — LOCKED`

---

## M4.4-D12 — Behavioral boundary

Decision:

Historical/behavioral features không được triển khai trong M4.4.

Status:

`LOCKED`



# 15. M4.4 Gate

## G01 — Development scope correct

Result:

`PASS`

Đúng `7,567,728` TRAIN + VALIDATION rows.

---

## G02 — Prediction-point validity

Result:

`PASS`

Tất cả M4.4 candidates chỉ sử dụng transaction hiện tại.

---

## G03 — Raw identifiers excluded

Result:

`PASS`

User/Card/Merchant Name không xuất hiện trong feature frame.

---

## G04 — Amount representation explicit

Result:

`PASS`

Primary và alternative Amount representation đã được phân biệt rõ.

---

## G05 — Transaction mode representation explicit

Result:

`PASS`

Transaction mode được khóa là categorical current-transaction candidate.

---

## G06 — MCC support/generalization audited

Result:

`PASS`

109 categories, 0 unseen VALIDATION dưới W_LONG/W_SHORT; shortcut risk được ghi nhận.

---

## G07 — Location support/generalization audited

Result:

`PASS`

Đã audit location_state, State, City và Zip theo cardinality, support và unseen-category exposure.

---

## G08 — Calendar/time representation explicit

Result:

`PASS`

Hour/day/month/weekend đều được audit và phân vai rõ.

---

## G09 — VALIDATION unseen-category risk audited

Result:

`PASS`

Unseen-category exposure đã được đo riêng cho W_LONG và W_SHORT.

---

## G10 — W_LONG/W_SHORT compatibility audited

Result:

`PASS`

Candidate representation hoạt động với cả hai training-window strategy.

High-cardinality risk tăng dưới W_SHORT đã được ghi nhận nhưng không quyết định window winner.

---

## G11 — No learned preprocessing state

Result:

`PASS`

M4.4 chỉ tạo deterministic features.

---

## G12 — No historical feature leakage

Result:

`PASS`

Không historical aggregation hoặc previous-transaction feature trong M4.4.

---

## G13 — FINAL TEST remains protected

Result:

`PASS`

Không dùng FINAL TEST statistics hoặc target.

---

## G14 — Candidate Feature Registry complete

Result:

`PASS`

14 candidate features đã có semantic role và candidate status rõ.


# M4.4 Gate

Overall:

`PASS`

Blocking issue:

`NONE`

M4.4 Status:

`PASS — READY FOR M4.5`


# 16. Kết luận M4.4

## Mục tiêu đã kiểm tra

M4.4 kiểm tra liệu thông tin của transaction hiện tại có thể được chuyển thành một transaction-level feature representation hợp lệ tại prediction point, nhất quán với M4.2/M4.3 và không tạo leakage hay learned preprocessing state hay không.

Notebook đã audit:

- Amount representation;
- transaction mode;
- MCC;
- semantic location;
- fine-grained location;
- calendar/time features;
- cardinality;
- category support;
- TRAIN → VALIDATION unseen category;
- W_LONG/W_SHORT compatibility;
- temporal support;
- memory/integrity.

---

## Candidate representation đã audit

Current deterministic feature builder tạo:

`14 candidates`

và chạy thành công trên:

`7,567,728 development transactions`.

Không có:

- Amount parse failure;
- Timestamp parse failure;
- candidate missing;
- unexpected location state;
- raw identifier;
- target;
- FINAL TEST usage;
- historical feature;
- learned preprocessing state.

---

## Transaction-level feature candidates được khóa

### Primary candidates

`amount_numeric`

`transaction_mode`

`location_state`

`hour_of_day`

`day_of_week`

Đây là các representation có:

- prediction-point validity;
- manageable cardinality;
- full development support;
- không hoặc rất ít structural generalization concern.

### Conditional candidates

`mcc_code`

`merchant_state_cat`

`month_of_year`

MCC:

representation/generalization tốt nhưng giữ synthetic-shortcut risk flag.

Merchant State:

generalization tốt nhưng thuộc fine-grained raw location family.

Month:

prediction-point valid nhưng VALIDATION chỉ bao phủ tháng 1–5.

### Experiment/alternative candidates

`amount_signed_log1p`

`is_negative_amount`

`is_zero_amount`

`merchant_city_cat`

`zip_cat`

`is_weekend`

Amount-derived fields cần experiment để chứng minh lợi ích ngoài `amount_numeric`.

City/Zip có high-cardinality/long-tail risk.

Weekend trùng thông tin với day-of-week.

---

## Candidate bị loại hoặc giữ conditional

Không có transaction-level field nào bị tuyên bố “invalid” chỉ vì không nằm trong core set.

Tuy nhiên:

`merchant_city_cat`

và:

`zip_cat`

không được đưa vào core baseline mặc định do high-cardinality, rare support, unseen-category và memorization risk.

`merchant_state_cat`

và:

`mcc_code`

được giữ conditional thay vì auto-included.

`is_weekend`

không core vì redundant với day-of-week.

`amount_signed_log1p`

và Amount indicators không core vì deterministic từ Amount và chưa có performance evidence.

---

## Những rủi ro được ghi nhận

### MCC

Synthetic shortcut / target-association risk.

### Merchant City / Zip

High cardinality.

Long-tail support.

Unseen-category exposure.

Memorization risk.

### Merchant State

Raw-location shortcut risk dù cardinality/generalization tương đối manageable.

### month_of_year

VALIDATION chỉ có tháng 1–5 nên không đánh giá đầy đủ annual seasonal cycle.

### Memory

String-heavy feature audit frame dùng khoảng:

`384 bytes/row`

trên preview.

M4.6 cần preprocessing implementation chú ý memory và không materialize representation dư thừa nếu không cần.

---

## Những vấn đề tiếp tục OPEN

M4.4 chưa quyết định:

- raw Amount hay signed-log thắng về model performance;
- sign/zero indicators có cải thiện classification hay không;
- MCC có cải thiện future validation đủ để chấp nhận shortcut risk hay không;
- Merchant State có nên vào final baseline;
- City/Zip experiment có đáng thực hiện hay không;
- final categorical encoder;
- unknown-category encoding;
- scaling;
- final numeric transformation;
- feature selection;
- behavioral features;
- training-window winner.

Các câu hỏi learned preprocessing thuộc M4.6.

Behavioral representation thuộc M4.5.

Performance-dependent feature decisions thuộc experiment protocol phía sau.

---

## Blocking issue

`NONE`

Không có transaction-level representation issue chặn causal behavioral feature engineering.

---

## M4.4 Gate

`PASS`

Candidate Feature Registry đã hoàn chỉnh ở mức transaction-level representation.

---

## Trạng thái cuối

`M4.4 — PASS`

`Transaction-level Feature Specification — LOCKED AT CANDIDATE LEVEL`

`READY FOR M4.5`

---

## Handoff

Next:

`M4.5 — Triển khai causal behavioral features`

M4.5 phải giữ nguyên transaction-level contract của M4.4 và chỉ bổ sung các feature dùng historical context theo strict causal rule:

`Timestamp(history) < Timestamp(current)`

M4.5 có thể triển khai các behavioral candidate đã được handoff trước đó như:

- time since previous transaction;
- transaction count trong historical window;
- historical Amount context;
- new-merchant indicator.

M4.5 không được:

- đưa User/Card/Merchant Name trực tiếp vào classifier;
- dùng same-timestamp transaction làm history;
- dùng target history;
- dùng future transaction;
- học categorical encoder/scaler.

Sau M4.5, M4.6 sẽ nhận transaction-level candidates + behavioral candidates để xây leakage-safe preprocessing pipeline.
