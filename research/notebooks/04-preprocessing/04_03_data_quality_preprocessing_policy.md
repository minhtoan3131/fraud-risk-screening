# M4.3 — Khóa policy xử lý các vấn đề data-quality trong preprocessing

## Vai trò

M4.2 đã khóa Base Representation Contract.

M4.3 không thiết kế lại:

- raw schema;
- Amount parser;
- Timestamp;
- target mapping;
- temporal split;
- W_LONG / W_SHORT boundary;
- identifier/history-key contract.

M4.3 trả lời câu hỏi:

> Các pattern chất lượng dữ liệu đã phát hiện ở M2 phải được xử lý thế nào trong preprocessing mà không phá semantic của transaction và không tạo leakage?

Các vấn đề trọng tâm:

- negative Amount;
- zero Amount;
- Amount cực trị;
- structural location missingness;
- Online / non-physical location semantics;
- residual Zip missing;
- exact duplicate;
- category consistency;
- các representation anomaly liên quan trực tiếp đến preprocessing.

## M4.3 không làm

- categorical encoding cuối;
- feature selection;
- behavioral feature;
- scaling;
- model training;
- resampling;
- tuning;
- threshold selection;
- sử dụng FINAL TEST để quyết định preprocessing.

## Trạng thái khi bắt đầu

M4.2:
PASS

Base Representation Contract:
LOCKED

M4.3 policy:
OPEN

M4.3 Gate:
OPEN

# 1. Evidence kế thừa và nguyên tắc quyết định

M4.3 không khám phá data-quality từ đầu.

Các finding M2 được sử dụng để xác định những câu hỏi cần kiểm chứng, không được tự động biến thành preprocessing action.

## Amount

M2 đã ghi nhận trên toàn artifact:

Negative:
1,244,683
≈ 5.1039%

Zero:
20,213
≈ 0.0829%

Positive:
23,122,004
≈ 94.8132%

Negative và zero Amount có pattern rõ theo transaction mode.

M2 không có bằng chứng đủ để coi:

- negative Amount;
- zero Amount;
- Amount = -500;
- extreme positive Amount

là corruption cần sửa hoặc xóa.

Guardrail hiện tại:

Không tự động:

- abs();
- drop negative;
- replace zero;
- clip / winsorize extreme values.

## Location

M2 đã ghi nhận:

Online Transaction
→ Merchant City = ONLINE
→ Merchant State missing
→ Zip missing

trên toàn bộ Online Transaction.

Ngoài Online mode còn tồn tại:

- Chip Transaction có Merchant City = ONLINE;
- residual Zip missing ở physical/non-online merchant;
- Merchant State đôi khi chứa country name.

Do đó location missing không phải một cơ chế missing ngẫu nhiên duy nhất.

## Exact duplicate

M2 xác nhận:

66 duplicate groups
132 duplicate-member rows
66 duplicate extra rows
largest group = 2

Dataset không có transaction_id duy nhất.

Exact equality trên 15 raw fields không đủ chứng minh hai row là cùng một transaction bị copy lỗi.

## Evidence rule của M4.3

M2 evidence:
→ dùng để hình thành hypothesis / guardrail.

M4.3 current runtime:
→ chỉ sử dụng development data để kiểm chứng policy.

Development audit window:

2015-01-01
<= Timestamp
< 2019-06-01

Tách thành:

TRAIN_2015_2017
TRAIN_2018
VALIDATION

FINAL TEST:
không aggregate feature/data-quality statistics phục vụ quyết định M4.3.

# 2. Thiết lập môi trường

M4.3 tiếp tục sử dụng đúng raw artifact đã được M4.2 xác minh.

Mục tiêu của cell này chỉ là bảo đảm notebook chạy độc lập và không vô tình dùng artifact khác.


```python
from pathlib import Path
from collections import Counter, defaultdict

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
        "Không xác định được PROJECT_ROOT. "
        "Không tìm thấy raw artifact."
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
    "M4.3 ENVIRONMENT / ARTIFACT GATE: PASS"
)
```

    M4.3 ENVIRONMENT / ARTIFACT GATE: PASS


### Nhận xét

Notebook đã xác định được đúng project root và raw artifact:

`data/raw/ibm_tabformer/card_transaction.v1.csv`

File tồn tại tại thời điểm thực thi và có kích thước:

`2,354,626,737 bytes`

Kích thước này khớp artifact đã được xác minh ở M4.2.

Cell kiểm tra trả về:

`M4.3 ENVIRONMENT / ARTIFACT GATE: PASS`

M4.3 do đó đang làm việc trên cùng raw artifact đã được khóa ở bước trước, không có dấu hiệu artifact bị thay đổi hoặc notebook đọc nhầm nguồn dữ liệu.

### Kết luận

Current raw artifact phù hợp với input contract của M4.3.

Environment / artifact status:

`PASS`

Blocking issue:

`NONE`



# 3. Canonical transformation và development boundary

M4.3 kế thừa trực tiếp transformation đã khóa ở M4.2.

Không thiết kế parser mới.

## Development window

M4.3 chỉ sử dụng:

2015-01-01
<= Timestamp
< 2019-06-01

để đưa ra policy.

Ba period được audit riêng:

TRAIN_2015_2017

TRAIN_2018

VALIDATION

Mục đích:

- không chọn W_LONG/W_SHORT winner;
- kiểm tra pattern có nhất quán giữa các temporal region hay không;
- không sử dụng FINAL TEST.


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
            default="OUTSIDE_M43_DEVELOPMENT",
        ),
        index=timestamp.index,
        dtype="string",
    )
```

# 4. M4.3.1 — Full development-data quality scan

## Câu hỏi

Các pattern data-quality quan trọng có còn xuất hiện nhất quán trong development population hiện tại hay không?

## Scope

Chỉ aggregate:

TRAIN_2015_2017
TRAIN_2018
VALIDATION

Không aggregate FINAL TEST.

## Trong một scan thu thập

Amount:
- negative / zero / positive;
- distribution theo period;
- distribution theo Use Chip;
- min / max;
- count Amount = -500.

Location:
- City = ONLINE;
- State missing;
- Zip missing;
- structural location state;
- residual Zip missing;
- country/state value của residual Zip missing.

Category hygiene:
- blank string;
- leading/trailing whitespace;
- raw unique count;
- stripped unique count;
- case-normalized unique count.

Representation anomaly:
- fractional Zip code;
- unexpected Use Chip.


```python
M43_USECOLS = [
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

assert "Is Fraud?" not in M43_USECOLS
assert "Errors?" not in M43_USECOLS

EXPECTED_USE_CHIP = {
    "Chip Transaction",
    "Online Transaction",
    "Swipe Transaction",
}

EXPECTED_DEVELOPMENT_ROWS = (
    6_855_270
    + 712_458
)

print(
    "Expected development rows:",
    f"{EXPECTED_DEVELOPMENT_ROWS:,}",
)
```

    Expected development rows: 7,567,728



```python
development_rows = 0

period_counts = Counter()

amount_sign_counts = Counter()
amount_sign_by_period = Counter()
amount_sign_by_period_mode = Counter()

amount_min = None
amount_max = None
amount_eq_minus_500 = 0

amount_parse_failures = 0
timestamp_parse_failures = 0

location_state_counts = Counter()
location_state_by_period = Counter()

city_online_by_period_mode = Counter()

state_missing_by_period_mode = Counter()
zip_missing_by_period_mode = Counter()

residual_zip_state_counts = Counter()
residual_zip_city_counts = Counter()

use_chip_values = set()

zip_fractional_count = 0


# ------------------------------------------------------------
# Category-hygiene audit
# ------------------------------------------------------------

STRING_AUDIT_COLUMNS = [
    "Use Chip",
    "Merchant City",
    "Merchant State",
]

category_blank_counts = Counter()
category_whitespace_counts = Counter()

category_raw_values = {
    column: set()
    for column in STRING_AUDIT_COLUMNS
}

category_stripped_values = {
    column: set()
    for column in STRING_AUDIT_COLUMNS
}

category_casefold_values = {
    column: set()
    for column in STRING_AUDIT_COLUMNS
}


development_min_timestamp = None
development_max_timestamp = None
```

## Location semantic state dùng cho audit

M4.3 tạm phân biệt bốn trạng thái kỹ thuật:

NON_PHYSICAL_OR_ONLINE

Merchant City = ONLINE
và State/Zip unavailable.

PHYSICAL_COMPLETE

Merchant City != ONLINE
State available
Zip available.

PHYSICAL_ZIP_UNAVAILABLE

Merchant City != ONLINE
State available
Zip missing.

OTHER_INCONSISTENT

Các cấu hình còn lại.

Đây chưa phải final feature.

Mục tiêu là kiểm tra xem một quy tắc imputation duy nhất có làm mất semantic hay không.


```python
def assign_location_state(df):
    city = (
        df["Merchant City"]
        .astype("string")
        .str.strip()
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

    conditions = [
        nonphysical,
        physical_complete,
        physical_zip_unavailable,
    ]

    choices = [
        "NON_PHYSICAL_OR_ONLINE",
        "PHYSICAL_COMPLETE",
        "PHYSICAL_ZIP_UNAVAILABLE",
    ]

    result = np.select(
        conditions,
        choices,
        default="OTHER_INCONSISTENT",
    )

    return pd.Series(
        result,
        index=df.index,
        dtype="string",
    )
```


```python
for chunk_number, chunk in enumerate(
    pd.read_csv(
        DATA_PATH,
        usecols=M43_USECOLS,
        chunksize=CHUNK_SIZE,
    ),
    start=1,
):
    timestamp = build_timestamp(chunk)

    timestamp_parse_failures += int(
        timestamp.isna().sum()
    )

    audit_period = (
        assign_audit_period(timestamp)
    )

    development_mask = (
        audit_period
        != "OUTSIDE_M43_DEVELOPMENT"
    )

    if not development_mask.any():
        continue

    dev = (
        chunk.loc[
            development_mask
        ]
        .copy()
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

    dev = dev.reset_index(drop=True)

    amount = parse_amount(
        dev["Amount"]
    )

    amount_parse_failures += int(
        amount.isna().sum()
    )

    development_rows += len(dev)

    period_counts.update(
        dev_period
        .value_counts()
        .to_dict()
    )


    # ========================================================
    # Temporal extent of inspected development data
    # ========================================================

    chunk_min = dev_timestamp.min()
    chunk_max = dev_timestamp.max()

    if pd.notna(chunk_min):
        if (
            development_min_timestamp is None
            or chunk_min
            < development_min_timestamp
        ):
            development_min_timestamp = chunk_min

    if pd.notna(chunk_max):
        if (
            development_max_timestamp is None
            or chunk_max
            > development_max_timestamp
        ):
            development_max_timestamp = chunk_max


    # ========================================================
    # Amount
    # ========================================================

    sign = pd.Series(
        np.select(
            [
                amount < 0,
                amount == 0,
                amount > 0,
            ],
            [
                "negative",
                "zero",
                "positive",
            ],
            default="unparsed",
        ),
        index=dev.index,
        dtype="string",
    )

    amount_sign_counts.update(
        sign
        .value_counts()
        .to_dict()
    )

    for period, sign_value in zip(
        dev_period,
        sign,
    ):
        amount_sign_by_period[
            (
                str(period),
                str(sign_value),
            )
        ] += 1

    use_chip = (
        dev["Use Chip"]
        .astype("string")
    )

    for period, mode, sign_value in zip(
        dev_period,
        use_chip,
        sign,
    ):
        amount_sign_by_period_mode[
            (
                str(period),
                str(mode),
                str(sign_value),
            )
        ] += 1

    valid_amount = amount.dropna()

    if len(valid_amount) > 0:
        chunk_amount_min = valid_amount.min()
        chunk_amount_max = valid_amount.max()

        if (
            amount_min is None
            or chunk_amount_min < amount_min
        ):
            amount_min = chunk_amount_min

        if (
            amount_max is None
            or chunk_amount_max > amount_max
        ):
            amount_max = chunk_amount_max

    amount_eq_minus_500 += int(
        amount.eq(-500).sum()
    )


    # ========================================================
    # Use Chip domain
    # ========================================================

    use_chip_values.update(
        use_chip
        .dropna()
        .unique()
        .tolist()
    )


    # ========================================================
    # Location
    # ========================================================

    location_state = (
        assign_location_state(dev)
    )

    location_state_counts.update(
        location_state
        .value_counts()
        .to_dict()
    )

    for period, state in zip(
        dev_period,
        location_state,
    ):
        location_state_by_period[
            (
                str(period),
                str(state),
            )
        ] += 1

    city_online = (
        dev["Merchant City"]
        .astype("string")
        .str.strip()
        .str.upper()
        .eq("ONLINE")
        .fillna(False)
    )

    state_missing = (
        dev["Merchant State"]
        .isna()
    )

    zip_missing = (
        dev["Zip"]
        .isna()
    )

    for period, mode, is_city_online in zip(
        dev_period,
        use_chip,
        city_online,
    ):
        if is_city_online:
            city_online_by_period_mode[
                (
                    str(period),
                    str(mode),
                )
            ] += 1

    for period, mode, is_missing in zip(
        dev_period,
        use_chip,
        state_missing,
    ):
        if is_missing:
            state_missing_by_period_mode[
                (
                    str(period),
                    str(mode),
                )
            ] += 1

    for period, mode, is_missing in zip(
        dev_period,
        use_chip,
        zip_missing,
    ):
        if is_missing:
            zip_missing_by_period_mode[
                (
                    str(period),
                    str(mode),
                )
            ] += 1


    # --------------------------------------------------------
    # residual physical Zip missing
    # --------------------------------------------------------

    residual_zip_mask = (
        location_state
        == "PHYSICAL_ZIP_UNAVAILABLE"
    )

    residual_zip_state_counts.update(
        dev.loc[
            residual_zip_mask,
            "Merchant State",
        ]
        .fillna("<MISSING>")
        .astype(str)
        .value_counts()
        .to_dict()
    )

    residual_zip_city_counts.update(
        dev.loc[
            residual_zip_mask,
            "Merchant City",
        ]
        .fillna("<MISSING>")
        .astype(str)
        .value_counts()
        .to_dict()
    )


    # ========================================================
    # Category hygiene
    # ========================================================

    for column in STRING_AUDIT_COLUMNS:
        raw = (
            dev[column]
            .dropna()
            .astype("string")
        )

        stripped = raw.str.strip()

        category_blank_counts[
            column
        ] += int(
            stripped.eq("").sum()
        )

        category_whitespace_counts[
            column
        ] += int(
            raw.ne(stripped).sum()
        )

        category_raw_values[
            column
        ].update(
            raw.unique().tolist()
        )

        category_stripped_values[
            column
        ].update(
            stripped.unique().tolist()
        )

        category_casefold_values[
            column
        ].update(
            stripped
            .str.casefold()
            .unique()
            .tolist()
        )


    # ========================================================
    # Zip representation anomaly
    # ========================================================

    zip_nonmissing = (
        dev["Zip"]
        .dropna()
    )

    zip_fractional_count += int(
        (
            zip_nonmissing
            % 1
            != 0
        ).sum()
    )


    if (
        chunk_number % 10 == 0
        or len(chunk) < CHUNK_SIZE
    ):
        print(
            f"Chunk {chunk_number:02d} | "
            f"development rows = "
            f"{development_rows:,}"
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
    f"{development_rows:,}",
)

print(
    "Timestamp parse failures:",
    timestamp_parse_failures,
)

print(
    "Amount parse failures:",
    amount_parse_failures,
)

print(
    "Development min Timestamp:",
    development_min_timestamp,
)

print(
    "Development max Timestamp:",
    development_max_timestamp,
)

print(
    "\nPeriod counts:"
)

for key in sorted(period_counts):
    print(
        key,
        f"{period_counts[key]:,}",
    )

print(
    "\nUse Chip values:"
)

print(
    sorted(use_chip_values)
)
```

    Development rows: 7,567,728
    Timestamp parse failures: 0
    Amount parse failures: 0
    Development min Timestamp: 2015-01-01 00:01:00
    Development max Timestamp: 2019-05-31 23:58:00
    
    Period counts:
    TRAIN_2015_2017 5,133,655
    TRAIN_2018 1,721,615
    VALIDATION 712,458
    
    Use Chip values:
    ['Chip Transaction', 'Online Transaction', 'Swipe Transaction']



```python
assert (
    development_rows
    == EXPECTED_DEVELOPMENT_ROWS
)

assert timestamp_parse_failures == 0
assert amount_parse_failures == 0

assert (
    period_counts[
        "TRAIN_2015_2017"
    ]
    == 5_133_655
)

assert (
    period_counts[
        "TRAIN_2018"
    ]
    == 1_721_615
)

assert (
    period_counts[
        "VALIDATION"
    ]
    == 712_458
)

assert (
    development_min_timestamp
    >= W_LONG_START
)

assert (
    development_max_timestamp
    < VALIDATION_END
)

assert (
    use_chip_values
    == EXPECTED_USE_CHIP
)

print(
    "M4.3 DEVELOPMENT SCOPE GATE: PASS"
)
```

    M4.3 DEVELOPMENT SCOPE GATE: PASS


### Nhận xét

Full scan đã xử lý đủ:

`7,567,728 development rows`

đúng bằng:

`W_LONG eligible TRAIN + VALIDATION`

Ba temporal period được tái tạo đúng:

`TRAIN_2015_2017 = 5,133,655`

`TRAIN_2018 = 1,721,615`

`VALIDATION = 712,458`

Development temporal extent quan sát được là:

`2015-01-01 00:01:00`
→
`2019-05-31 23:58:00`

Do đó không có transaction thuộc FINAL TEST hoặc post-validation period được đưa vào các statistics phục vụ data-quality decision.

Timestamp parse failures:

`0`

Amount parse failures:

`0`

Use Chip domain vẫn chỉ gồm:

`Chip Transaction`

`Online Transaction`

`Swipe Transaction`

Ngoài ra, danh sách cột dùng cho M4.3 không chứa:

`Is Fraud?`

hay:

`Errors?`

Vì vậy các policy ở M4.3 được hình thành mà không sử dụng target và không phụ thuộc FINAL TEST statistics.

Cell assertion trả về:

`M4.3 DEVELOPMENT SCOPE GATE: PASS`

### Kết luận

Development audit scope của M4.3 được triển khai đúng temporal boundary và đúng test-isolation protocol.

Development rows:

`7,567,728`

FINAL TEST used for data-quality decision:

`NO`

Target used:

`NO`

Status:

`PASS`

M4.3 có đủ development evidence để tiếp tục đánh giá các data-quality policy.


# 5. M4.3.2 — Audit negative / zero / extreme Amount

## Câu hỏi

Negative Amount và zero Amount có nên được coi là dữ liệu lỗi cần sửa/xóa hay không?

Các pattern có ổn định giữa:

TRAIN_2015_2017
TRAIN_2018
VALIDATION

hay không?

## Candidate policy cần đánh giá

A.
negative → abs()

B.
negative → drop

C.
negative → clip

D.
giữ signed Amount_numeric

Zero Amount:

A.
coi là missing / invalid

B.
giữ nguyên zero

Extreme Amount:

A.
clip / winsorize ngay

B.
giữ nguyên ở data-quality layer;
transformation candidate để M4.4/M4.6 quyết định.


```python
amount_sign_summary = (
    pd.DataFrame(
        [
            {
                "amount_sign": key,
                "transaction_count": value,
            }
            for key, value
            in amount_sign_counts.items()
        ]
    )
)

amount_sign_summary[
    "share_pct"
] = (
    amount_sign_summary[
        "transaction_count"
    ]
    / development_rows
    * 100
)

display(
    amount_sign_summary
    .sort_values("amount_sign")
)

print(
    "\nDevelopment Amount min:",
    amount_min,
)

print(
    "Development Amount max:",
    amount_max,
)

print(
    "Amount == -500 count:",
    f"{amount_eq_minus_500:,}",
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
      <th>amount_sign</th>
      <th>transaction_count</th>
      <th>share_pct</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>negative</td>
      <td>365166</td>
      <td>4.825306</td>
    </tr>
    <tr>
      <th>0</th>
      <td>positive</td>
      <td>7196554</td>
      <td>95.095305</td>
    </tr>
    <tr>
      <th>2</th>
      <td>zero</td>
      <td>6008</td>
      <td>0.079390</td>
    </tr>
  </tbody>
</table>
</div>


    
    Development Amount min: -500.0
    Development Amount max: 6613.44
    Amount == -500 count: 98



```python
amount_period_df = pd.DataFrame(
    [
        {
            "period": key[0],
            "amount_sign": key[1],
            "transaction_count": value,
        }
        for key, value
        in amount_sign_by_period.items()
    ]
)

period_total_map = dict(
    period_counts
)

amount_period_df[
    "period_total"
] = (
    amount_period_df[
        "period"
    ]
    .map(period_total_map)
)

amount_period_df[
    "rate_within_period_pct"
] = (
    amount_period_df[
        "transaction_count"
    ]
    / amount_period_df[
        "period_total"
    ]
    * 100
)

display(
    amount_period_df
    .sort_values(
        [
            "period",
            "amount_sign",
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
      <th>amount_sign</th>
      <th>transaction_count</th>
      <th>period_total</th>
      <th>rate_within_period_pct</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>TRAIN_2015_2017</td>
      <td>negative</td>
      <td>248541</td>
      <td>5133655</td>
      <td>4.841404</td>
    </tr>
    <tr>
      <th>0</th>
      <td>TRAIN_2015_2017</td>
      <td>positive</td>
      <td>4881015</td>
      <td>5133655</td>
      <td>95.078750</td>
    </tr>
    <tr>
      <th>6</th>
      <td>TRAIN_2015_2017</td>
      <td>zero</td>
      <td>4099</td>
      <td>5133655</td>
      <td>0.079846</td>
    </tr>
    <tr>
      <th>3</th>
      <td>TRAIN_2018</td>
      <td>negative</td>
      <td>82314</td>
      <td>1721615</td>
      <td>4.781208</td>
    </tr>
    <tr>
      <th>2</th>
      <td>TRAIN_2018</td>
      <td>positive</td>
      <td>1637937</td>
      <td>1721615</td>
      <td>95.139564</td>
    </tr>
    <tr>
      <th>7</th>
      <td>TRAIN_2018</td>
      <td>zero</td>
      <td>1364</td>
      <td>1721615</td>
      <td>0.079228</td>
    </tr>
    <tr>
      <th>5</th>
      <td>VALIDATION</td>
      <td>negative</td>
      <td>34311</td>
      <td>712458</td>
      <td>4.815863</td>
    </tr>
    <tr>
      <th>4</th>
      <td>VALIDATION</td>
      <td>positive</td>
      <td>677602</td>
      <td>712458</td>
      <td>95.107641</td>
    </tr>
    <tr>
      <th>8</th>
      <td>VALIDATION</td>
      <td>zero</td>
      <td>545</td>
      <td>712458</td>
      <td>0.076496</td>
    </tr>
  </tbody>
</table>
</div>



```python
amount_period_mode_df = pd.DataFrame(
    [
        {
            "period": key[0],
            "Use Chip": key[1],
            "amount_sign": key[2],
            "transaction_count": value,
        }
        for key, value
        in amount_sign_by_period_mode.items()
    ]
)

mode_totals = (
    amount_period_mode_df
    .groupby(
        [
            "period",
            "Use Chip",
        ]
    )[
        "transaction_count"
    ]
    .sum()
)

amount_period_mode_df[
    "period_mode_total"
] = [
    mode_totals.loc[
        (
            row["period"],
            row["Use Chip"],
        )
    ]
    for _, row
    in amount_period_mode_df.iterrows()
]

amount_period_mode_df[
    "rate_within_period_mode_pct"
] = (
    amount_period_mode_df[
        "transaction_count"
    ]
    / amount_period_mode_df[
        "period_mode_total"
    ]
    * 100
)

display(
    amount_period_mode_df
    .sort_values(
        [
            "period",
            "Use Chip",
            "amount_sign",
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
      <th>Use Chip</th>
      <th>amount_sign</th>
      <th>transaction_count</th>
      <th>period_mode_total</th>
      <th>rate_within_period_mode_pct</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>TRAIN_2015_2017</td>
      <td>Chip Transaction</td>
      <td>negative</td>
      <td>197765</td>
      <td>3619072</td>
      <td>5.464522</td>
    </tr>
    <tr>
      <th>0</th>
      <td>TRAIN_2015_2017</td>
      <td>Chip Transaction</td>
      <td>positive</td>
      <td>3417944</td>
      <td>3619072</td>
      <td>94.442553</td>
    </tr>
    <tr>
      <th>18</th>
      <td>TRAIN_2015_2017</td>
      <td>Chip Transaction</td>
      <td>zero</td>
      <td>3363</td>
      <td>3619072</td>
      <td>0.092924</td>
    </tr>
    <tr>
      <th>5</th>
      <td>TRAIN_2015_2017</td>
      <td>Online Transaction</td>
      <td>negative</td>
      <td>1454</td>
      <td>639122</td>
      <td>0.227500</td>
    </tr>
    <tr>
      <th>1</th>
      <td>TRAIN_2015_2017</td>
      <td>Online Transaction</td>
      <td>positive</td>
      <td>637668</td>
      <td>639122</td>
      <td>99.772500</td>
    </tr>
    <tr>
      <th>4</th>
      <td>TRAIN_2015_2017</td>
      <td>Swipe Transaction</td>
      <td>negative</td>
      <td>49322</td>
      <td>875461</td>
      <td>5.633832</td>
    </tr>
    <tr>
      <th>3</th>
      <td>TRAIN_2015_2017</td>
      <td>Swipe Transaction</td>
      <td>positive</td>
      <td>825403</td>
      <td>875461</td>
      <td>94.282098</td>
    </tr>
    <tr>
      <th>19</th>
      <td>TRAIN_2015_2017</td>
      <td>Swipe Transaction</td>
      <td>zero</td>
      <td>736</td>
      <td>875461</td>
      <td>0.084070</td>
    </tr>
    <tr>
      <th>8</th>
      <td>TRAIN_2018</td>
      <td>Chip Transaction</td>
      <td>negative</td>
      <td>65528</td>
      <td>1215055</td>
      <td>5.393007</td>
    </tr>
    <tr>
      <th>6</th>
      <td>TRAIN_2018</td>
      <td>Chip Transaction</td>
      <td>positive</td>
      <td>1148422</td>
      <td>1215055</td>
      <td>94.516051</td>
    </tr>
    <tr>
      <th>22</th>
      <td>TRAIN_2018</td>
      <td>Chip Transaction</td>
      <td>zero</td>
      <td>1105</td>
      <td>1215055</td>
      <td>0.090942</td>
    </tr>
    <tr>
      <th>15</th>
      <td>TRAIN_2018</td>
      <td>Online Transaction</td>
      <td>negative</td>
      <td>457</td>
      <td>213632</td>
      <td>0.213919</td>
    </tr>
    <tr>
      <th>7</th>
      <td>TRAIN_2018</td>
      <td>Online Transaction</td>
      <td>positive</td>
      <td>213175</td>
      <td>213632</td>
      <td>99.786081</td>
    </tr>
    <tr>
      <th>14</th>
      <td>TRAIN_2018</td>
      <td>Swipe Transaction</td>
      <td>negative</td>
      <td>16329</td>
      <td>292928</td>
      <td>5.574407</td>
    </tr>
    <tr>
      <th>9</th>
      <td>TRAIN_2018</td>
      <td>Swipe Transaction</td>
      <td>positive</td>
      <td>276340</td>
      <td>292928</td>
      <td>94.337175</td>
    </tr>
    <tr>
      <th>20</th>
      <td>TRAIN_2018</td>
      <td>Swipe Transaction</td>
      <td>zero</td>
      <td>259</td>
      <td>292928</td>
      <td>0.088418</td>
    </tr>
    <tr>
      <th>11</th>
      <td>VALIDATION</td>
      <td>Chip Transaction</td>
      <td>negative</td>
      <td>27442</td>
      <td>503020</td>
      <td>5.455449</td>
    </tr>
    <tr>
      <th>10</th>
      <td>VALIDATION</td>
      <td>Chip Transaction</td>
      <td>positive</td>
      <td>475127</td>
      <td>503020</td>
      <td>94.454892</td>
    </tr>
    <tr>
      <th>23</th>
      <td>VALIDATION</td>
      <td>Chip Transaction</td>
      <td>zero</td>
      <td>451</td>
      <td>503020</td>
      <td>0.089658</td>
    </tr>
    <tr>
      <th>17</th>
      <td>VALIDATION</td>
      <td>Online Transaction</td>
      <td>negative</td>
      <td>164</td>
      <td>88302</td>
      <td>0.185726</td>
    </tr>
    <tr>
      <th>12</th>
      <td>VALIDATION</td>
      <td>Online Transaction</td>
      <td>positive</td>
      <td>88138</td>
      <td>88302</td>
      <td>99.814274</td>
    </tr>
    <tr>
      <th>16</th>
      <td>VALIDATION</td>
      <td>Swipe Transaction</td>
      <td>negative</td>
      <td>6705</td>
      <td>121136</td>
      <td>5.535101</td>
    </tr>
    <tr>
      <th>13</th>
      <td>VALIDATION</td>
      <td>Swipe Transaction</td>
      <td>positive</td>
      <td>114337</td>
      <td>121136</td>
      <td>94.387300</td>
    </tr>
    <tr>
      <th>21</th>
      <td>VALIDATION</td>
      <td>Swipe Transaction</td>
      <td>zero</td>
      <td>94</td>
      <td>121136</td>
      <td>0.077599</td>
    </tr>
  </tbody>
</table>
</div>


### Phân tích / Nhận xét

Trên toàn development population:

`Negative = 365,166 — 4.8253%`

`Zero = 6,008 — 0.0794%`

`Positive = 7,196,554 — 95.0953%`

Negative Amount vì vậy không phải một số lượng nhỏ các record bất thường. Gần 5% development transactions mang Amount âm.

Quan trọng hơn, tỷ lệ negative Amount khá ổn định giữa ba temporal period:

`TRAIN_2015_2017 ≈ 4.8414%`

`TRAIN_2018 ≈ 4.7812%`

`VALIDATION ≈ 4.8159%`

Không xuất hiện hiện tượng negative Amount chỉ tập trung ở một giai đoạn lịch sử rồi biến mất ở VALIDATION.

Zero Amount cũng lặp lại qua cả ba period:

`TRAIN_2015_2017 ≈ 0.07985%`

`TRAIN_2018 ≈ 0.07923%`

`VALIDATION ≈ 0.07650%`

Tỷ lệ rất nhỏ nhưng ổn định về bậc độ lớn.

Phân tích theo transaction mode cho thấy sign của Amount tiếp tục có cấu trúc rõ.

Đối với Chip Transaction, negative rate lần lượt khoảng:

`5.46%`
→
`5.39%`
→
`5.46%`

Đối với Swipe Transaction:

`5.63%`
→
`5.57%`
→
`5.54%`

Đối với Online Transaction:

`0.228%`
→
`0.214%`
→
`0.186%`

Như vậy negative Amount xuất hiện ở cả ba mode nhưng Online Transaction có tỷ lệ thấp hơn rõ rệt so với Chip và Swipe.

Pattern này tồn tại xuyên suốt TRAIN và VALIDATION, do đó không phù hợp với giả thuyết negative Amount chỉ là corruption ngẫu nhiên đơn giản.

Zero Amount trong development window được quan sát ở Chip và Swipe Transaction. Không có dòng `zero` cho Online Transaction trong bảng development period × mode, tương ứng không quan sát zero Amount ở Online Transaction trong phạm vi M4.3 hiện tại.

Điều này tiếp tục cho thấy Amount sign/zero có liên hệ với transaction semantics thay vì chỉ là formatting error.

Development Amount range là:

`-500.00`
→
`6,613.44`

Giá trị:

`Amount = -500`

xuất hiện:

`98 transactions`

Việc minimum `-500` lặp lại nhiều lần củng cố finding trước rằng đây có thể là một boundary/convention của synthetic dataset. Tuy nhiên output không cung cấp evidence để xác định chính xác semantic của boundary này.

Development maximum `6,613.44` cũng không đi kèm evidence nào chứng minh đây là lỗi dữ liệu.

Do đó current output không tạo căn cứ để:

- dùng `abs()` cho negative Amount;
- drop negative transaction;
- thay zero bằng missing;
- drop zero;
- clip/winsorize extreme Amount ở data-quality layer.

Các transformation phục vụ model như signed transformation, log-like transform, scaling hoặc additional sign indicator có thể được xem xét ở M4.4/M4.6, nhưng không phải data cleaning decision của M4.3.

### Kết luận M4.3.2

Negative Amount là một phần có cấu trúc và có quy mô đáng kể trong development data.

Zero Amount tuy hiếm nhưng lặp lại nhất quán qua các temporal period và không có evidence để coi là missing/corruption.

Extreme Amount chưa có evidence để coi là lỗi cần clip hoặc xóa.

Policy:

Amount sign:

`PRESERVE SIGN`

Không:

`abs()`

Không:

`drop negative`

Zero Amount:

`PRESERVE AS VALID NUMERIC VALUE`

Extreme Amount:

`NO CLIPPING / WINSORIZATION AT DATA-QUALITY LAYER`

Status:

`LOCKED`

Việc tạo Amount-derived model features:

`OPEN — M4.4/M4.6`


# 6. M4.3.3 — Audit structural location missingness

## Câu hỏi

Location missing trong development data có tiếp tục gồm nhiều semantic state khác nhau hay không?

Nếu có, một generic imputation policy sẽ không phù hợp.

## Cần phân biệt

NON_PHYSICAL_OR_ONLINE

PHYSICAL_COMPLETE

PHYSICAL_ZIP_UNAVAILABLE

OTHER_INCONSISTENT

M4.3 chưa quyết định final location feature encoding.


```python
location_state_df = pd.DataFrame(
    [
        {
            "location_state": key,
            "transaction_count": value,
        }
        for key, value
        in location_state_counts.items()
    ]
)

location_state_df[
    "share_pct"
] = (
    location_state_df[
        "transaction_count"
    ]
    / development_rows
    * 100
)

display(
    location_state_df
    .sort_values(
        "transaction_count",
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
      <th>location_state</th>
      <th>transaction_count</th>
      <th>share_pct</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PHYSICAL_COMPLETE</td>
      <td>6570104</td>
      <td>86.817391</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NON_PHYSICAL_OR_ONLINE</td>
      <td>947482</td>
      <td>12.520032</td>
    </tr>
    <tr>
      <th>2</th>
      <td>PHYSICAL_ZIP_UNAVAILABLE</td>
      <td>50142</td>
      <td>0.662577</td>
    </tr>
  </tbody>
</table>
</div>



```python
location_period_df = pd.DataFrame(
    [
        {
            "period": key[0],
            "location_state": key[1],
            "transaction_count": value,
        }
        for key, value
        in location_state_by_period.items()
    ]
)

location_period_df[
    "period_total"
] = (
    location_period_df[
        "period"
    ]
    .map(period_total_map)
)

location_period_df[
    "rate_within_period_pct"
] = (
    location_period_df[
        "transaction_count"
    ]
    / location_period_df[
        "period_total"
    ]
    * 100
)

display(
    location_period_df
    .sort_values(
        [
            "period",
            "transaction_count",
        ],
        ascending=[
            True,
            False,
        ],
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
      <th>location_state</th>
      <th>transaction_count</th>
      <th>period_total</th>
      <th>rate_within_period_pct</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>TRAIN_2015_2017</td>
      <td>PHYSICAL_COMPLETE</td>
      <td>4457627</td>
      <td>5133655</td>
      <td>86.831449</td>
    </tr>
    <tr>
      <th>1</th>
      <td>TRAIN_2015_2017</td>
      <td>NON_PHYSICAL_OR_ONLINE</td>
      <td>643509</td>
      <td>5133655</td>
      <td>12.535104</td>
    </tr>
    <tr>
      <th>2</th>
      <td>TRAIN_2015_2017</td>
      <td>PHYSICAL_ZIP_UNAVAILABLE</td>
      <td>32519</td>
      <td>5133655</td>
      <td>0.633447</td>
    </tr>
    <tr>
      <th>3</th>
      <td>TRAIN_2018</td>
      <td>PHYSICAL_COMPLETE</td>
      <td>1494430</td>
      <td>1721615</td>
      <td>86.803960</td>
    </tr>
    <tr>
      <th>4</th>
      <td>TRAIN_2018</td>
      <td>NON_PHYSICAL_OR_ONLINE</td>
      <td>215056</td>
      <td>1721615</td>
      <td>12.491527</td>
    </tr>
    <tr>
      <th>8</th>
      <td>TRAIN_2018</td>
      <td>PHYSICAL_ZIP_UNAVAILABLE</td>
      <td>12129</td>
      <td>1721615</td>
      <td>0.704513</td>
    </tr>
    <tr>
      <th>5</th>
      <td>VALIDATION</td>
      <td>PHYSICAL_COMPLETE</td>
      <td>618047</td>
      <td>712458</td>
      <td>86.748552</td>
    </tr>
    <tr>
      <th>6</th>
      <td>VALIDATION</td>
      <td>NON_PHYSICAL_OR_ONLINE</td>
      <td>88917</td>
      <td>712458</td>
      <td>12.480315</td>
    </tr>
    <tr>
      <th>7</th>
      <td>VALIDATION</td>
      <td>PHYSICAL_ZIP_UNAVAILABLE</td>
      <td>5494</td>
      <td>712458</td>
      <td>0.771133</td>
    </tr>
  </tbody>
</table>
</div>



```python
def counter_to_frame(
    counter,
    value_name,
):
    return pd.DataFrame(
        [
            {
                "period": key[0],
                "Use Chip": key[1],
                value_name: value,
            }
            for key, value
            in counter.items()
        ]
    )


city_online_df = counter_to_frame(
    city_online_by_period_mode,
    "city_online_count",
)

state_missing_df = counter_to_frame(
    state_missing_by_period_mode,
    "state_missing_count",
)

zip_missing_df = counter_to_frame(
    zip_missing_by_period_mode,
    "zip_missing_count",
)

print("Merchant City = ONLINE:")
display(city_online_df)

print("\nMerchant State missing:")
display(state_missing_df)

print("\nZip missing:")
display(zip_missing_df)
```

    Merchant City = ONLINE:



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
      <th>Use Chip</th>
      <th>city_online_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>TRAIN_2015_2017</td>
      <td>Online Transaction</td>
      <td>639122</td>
    </tr>
    <tr>
      <th>1</th>
      <td>TRAIN_2015_2017</td>
      <td>Chip Transaction</td>
      <td>4387</td>
    </tr>
    <tr>
      <th>2</th>
      <td>TRAIN_2018</td>
      <td>Online Transaction</td>
      <td>213632</td>
    </tr>
    <tr>
      <th>3</th>
      <td>VALIDATION</td>
      <td>Online Transaction</td>
      <td>88302</td>
    </tr>
    <tr>
      <th>4</th>
      <td>TRAIN_2018</td>
      <td>Chip Transaction</td>
      <td>1424</td>
    </tr>
    <tr>
      <th>5</th>
      <td>VALIDATION</td>
      <td>Chip Transaction</td>
      <td>615</td>
    </tr>
  </tbody>
</table>
</div>


    
    Merchant State missing:



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
      <th>Use Chip</th>
      <th>state_missing_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>TRAIN_2015_2017</td>
      <td>Online Transaction</td>
      <td>639122</td>
    </tr>
    <tr>
      <th>1</th>
      <td>TRAIN_2015_2017</td>
      <td>Chip Transaction</td>
      <td>4387</td>
    </tr>
    <tr>
      <th>2</th>
      <td>TRAIN_2018</td>
      <td>Online Transaction</td>
      <td>213632</td>
    </tr>
    <tr>
      <th>3</th>
      <td>VALIDATION</td>
      <td>Online Transaction</td>
      <td>88302</td>
    </tr>
    <tr>
      <th>4</th>
      <td>TRAIN_2018</td>
      <td>Chip Transaction</td>
      <td>1424</td>
    </tr>
    <tr>
      <th>5</th>
      <td>VALIDATION</td>
      <td>Chip Transaction</td>
      <td>615</td>
    </tr>
  </tbody>
</table>
</div>


    
    Zip missing:



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
      <th>Use Chip</th>
      <th>zip_missing_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>TRAIN_2015_2017</td>
      <td>Online Transaction</td>
      <td>639122</td>
    </tr>
    <tr>
      <th>1</th>
      <td>TRAIN_2015_2017</td>
      <td>Chip Transaction</td>
      <td>32262</td>
    </tr>
    <tr>
      <th>2</th>
      <td>TRAIN_2018</td>
      <td>Online Transaction</td>
      <td>213632</td>
    </tr>
    <tr>
      <th>3</th>
      <td>VALIDATION</td>
      <td>Online Transaction</td>
      <td>88302</td>
    </tr>
    <tr>
      <th>4</th>
      <td>VALIDATION</td>
      <td>Chip Transaction</td>
      <td>5386</td>
    </tr>
    <tr>
      <th>5</th>
      <td>TRAIN_2015_2017</td>
      <td>Swipe Transaction</td>
      <td>4644</td>
    </tr>
    <tr>
      <th>6</th>
      <td>VALIDATION</td>
      <td>Swipe Transaction</td>
      <td>723</td>
    </tr>
    <tr>
      <th>7</th>
      <td>TRAIN_2018</td>
      <td>Chip Transaction</td>
      <td>11944</td>
    </tr>
    <tr>
      <th>8</th>
      <td>TRAIN_2018</td>
      <td>Swipe Transaction</td>
      <td>1609</td>
    </tr>
  </tbody>
</table>
</div>


### Phân tích / Nhận xét

Development data được phân chia hoàn toàn thành ba location semantic state:

`PHYSICAL_COMPLETE = 6,570,104 — 86.8174%`

`NON_PHYSICAL_OR_ONLINE = 947,482 — 12.5200%`

`PHYSICAL_ZIP_UNAVAILABLE = 50,142 — 0.6626%`

Không quan sát:

`OTHER_INCONSISTENT`

tức:

`0 rows`

Ba state này bao phủ toàn bộ development population.

Structural pattern cũng xuất hiện trong cả ba temporal period.

`NON_PHYSICAL_OR_ONLINE` chiếm:

`12.5351%` trong TRAIN_2015_2017

`12.4915%` trong TRAIN_2018

`12.4803%` trong VALIDATION

Tỷ lệ gần như ổn định.

`PHYSICAL_ZIP_UNAVAILABLE` cũng xuất hiện trong cả ba period:

`0.6334%`

`0.7045%`

`0.7711%`

Tỷ lệ có tăng nhẹ theo thời gian nhưng cùng semantic pattern vẫn tồn tại ở TRAIN và VALIDATION.

Một quan hệ đặc biệt rõ xuất hiện giữa `Merchant City = ONLINE` và `Merchant State missing`.

Trong từng period, count của hai hiện tượng này khớp chính xác theo transaction mode.

TRAIN_2015_2017:

Online Transaction:

`639,122`

Chip Transaction:

`4,387`

TRAIN_2018:

Online Transaction:

`213,632`

Chip Transaction:

`1,424`

VALIDATION:

Online Transaction:

`88,302`

Chip Transaction:

`615`

Tổng cộng:

`947,482`

transaction mang non-physical representation.

Điểm quan trọng là `Merchant City = ONLINE` không đồng nghĩa hoàn toàn với `Use Chip = Online Transaction`.

Trong development data còn:

`6,426`

Chip Transactions có cùng `Merchant City = ONLINE` representation.

Vì vậy semantic của `Merchant City = ONLINE` phù hợp hơn với:

`non-physical / online-style merchant location representation`

chứ không nên được thay thế máy móc bằng transaction mode.

Đối với Zip, missingness rộng hơn State missing.

Ngoài `947,482` non-physical rows còn có:

`50,142`

physical transactions có State/City information nhưng không có Zip.

Do đó Zip missing tồn tại ít nhất hai cơ chế:

`NON_PHYSICAL_OR_ONLINE`
→ Zip không áp dụng như physical postal code

và:

`PHYSICAL_ZIP_UNAVAILABLE`
→ physical location tồn tại nhưng Zip không được biểu diễn.

Nếu drop mọi row có State/Zip missing, preprocessing sẽ loại bỏ có hệ thống toàn bộ non-physical transaction population và thêm một nhóm physical transaction riêng biệt.

Nếu dùng một global mode/median imputation, hai semantic mechanism khác nhau cũng sẽ bị trộn thành cùng một trạng thái.

Cả hai cách xử lý đều làm mất thông tin cấu trúc đã được quan sát trực tiếp.

### Kết luận M4.3.3

Location missing trong development data là structural, không phải một generic missing-value problem.

Policy được khóa:

Không:

`drop rows because State/Zip is missing`

Không:

`global mode/median imputation`

Phải phân biệt tối thiểu:

`NON_PHYSICAL_OR_ONLINE`

`PHYSICAL_COMPLETE`

`PHYSICAL_ZIP_UNAVAILABLE`

Đối với non-physical representation:

`Merchant City = ONLINE`

được giữ như semantic category.

Missing State/Zip của nhóm này được diễn giải là:

`NOT_APPLICABLE`

thay vì generic unknown missing.

Đối với physical transaction bị Zip missing:

Zip được diễn giải riêng là:

`ZIP_UNAVAILABLE`

không được gộp semantic với non-physical `NOT_APPLICABLE`.

Location missing policy:

`LOCKED`

Final categorical encoding / final feature inclusion:

`OPEN — M4.4/M4.6`



# 7. M4.3.4 — Audit residual Zip missing

## Câu hỏi

Các transaction physical/non-online bị Zip missing có đặc điểm gì?

Mục tiêu không phải xác định country hoàn hảo.

Mục tiêu là quyết định:

Zip missing có thể được xử lý bằng một global imputation value hay không?

## Guardrail

Merchant State không được giả định là US-state-only field.


```python
residual_zip_state_df = pd.DataFrame(
    residual_zip_state_counts.most_common(
        30
    ),
    columns=[
        "Merchant State",
        "transaction_count",
    ],
)

residual_zip_city_df = pd.DataFrame(
    residual_zip_city_counts.most_common(
        30
    ),
    columns=[
        "Merchant City",
        "transaction_count",
    ],
)

print(
    "Top Merchant State values "
    "among PHYSICAL_ZIP_UNAVAILABLE:"
)

display(
    residual_zip_state_df
)

print(
    "\nTop Merchant City values "
    "among PHYSICAL_ZIP_UNAVAILABLE:"
)

display(
    residual_zip_city_df
)

print(
    "\nResidual physical Zip-missing rows:"
)

print(
    f"{sum(residual_zip_state_counts.values()):,}"
)
```

    Top Merchant State values among PHYSICAL_ZIP_UNAVAILABLE:



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
      <th>Merchant State</th>
      <th>transaction_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Mexico</td>
      <td>14249</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Canada</td>
      <td>6522</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Italy</td>
      <td>5031</td>
    </tr>
    <tr>
      <th>3</th>
      <td>United Kingdom</td>
      <td>2828</td>
    </tr>
    <tr>
      <th>4</th>
      <td>France</td>
      <td>1712</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Germany</td>
      <td>1672</td>
    </tr>
    <tr>
      <th>6</th>
      <td>China</td>
      <td>1636</td>
    </tr>
    <tr>
      <th>7</th>
      <td>India</td>
      <td>1126</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Japan</td>
      <td>1111</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Dominican Republic</td>
      <td>871</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Netherlands</td>
      <td>810</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Spain</td>
      <td>801</td>
    </tr>
    <tr>
      <th>12</th>
      <td>The Bahamas</td>
      <td>609</td>
    </tr>
    <tr>
      <th>13</th>
      <td>South Korea</td>
      <td>597</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Jamaica</td>
      <td>589</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Ireland</td>
      <td>554</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Taiwan</td>
      <td>498</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Switzerland</td>
      <td>449</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Costa Rica</td>
      <td>447</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Colombia</td>
      <td>447</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Philippines</td>
      <td>424</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Australia</td>
      <td>398</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Austria</td>
      <td>343</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Hong Kong</td>
      <td>334</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Peru</td>
      <td>334</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Thailand</td>
      <td>330</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Israel</td>
      <td>309</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Pakistan</td>
      <td>302</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Norway</td>
      <td>242</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Brazil</td>
      <td>235</td>
    </tr>
  </tbody>
</table>
</div>


    
    Top Merchant City values among PHYSICAL_ZIP_UNAVAILABLE:



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
      <th>Merchant City</th>
      <th>transaction_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Rome</td>
      <td>5031</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Cancun</td>
      <td>4689</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Mexico City</td>
      <td>2671</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Toronto</td>
      <td>2647</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Cabo San Lucas</td>
      <td>2610</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Puerto Vallarta</td>
      <td>2099</td>
    </tr>
    <tr>
      <th>6</th>
      <td>London</td>
      <td>1969</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Paris</td>
      <td>1712</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Berlin</td>
      <td>1672</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Guadalajara</td>
      <td>1665</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Vancouver</td>
      <td>1381</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Montreal</td>
      <td>1259</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Tokyo</td>
      <td>1111</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Beijing</td>
      <td>1017</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Santo Domingo</td>
      <td>871</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Edinburgh</td>
      <td>859</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Amsterdam</td>
      <td>810</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Calgary</td>
      <td>642</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Shanghai</td>
      <td>619</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Nassau</td>
      <td>609</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Seoul</td>
      <td>597</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Edmonton</td>
      <td>593</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Kingston</td>
      <td>589</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Dublin</td>
      <td>554</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Acapulco</td>
      <td>515</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Tapei</td>
      <td>498</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Madrid</td>
      <td>474</td>
    </tr>
    <tr>
      <th>27</th>
      <td>San Jose</td>
      <td>447</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Bogota</td>
      <td>447</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Manila</td>
      <td>424</td>
    </tr>
  </tbody>
</table>
</div>


    
    Residual physical Zip-missing rows:
    50,142


### Phân tích / Nhận xét

Development data có:

`50,142`

transaction thuộc:

`PHYSICAL_ZIP_UNAVAILABLE`

tương đương khoảng:

`0.6626%`

development population.

Tổng Zip-missing development rows gồm:

`947,482` non-physical rows

và:

`50,142` physical Zip-unavailable rows

tức khoảng:

`997,624`

Zip-missing transactions.

Residual physical Zip missing do đó chiếm khoảng:

`5.03%`

toàn bộ Zip-missing development rows.

Nhóm này không đủ lớn để có thể bỏ qua như một vài anomaly riêng lẻ.

Các giá trị `Merchant State` phổ biến nhất trong nhóm này bao gồm:

`Mexico`

`Canada`

`Italy`

`United Kingdom`

`France`

`Germany`

`China`

`India`

`Japan`

và nhiều tên quốc gia khác.

Các Merchant City tương ứng cũng gồm nhiều location quốc tế như:

`Rome`

`Cancun`

`Mexico City`

`Toronto`

`London`

`Paris`

`Berlin`

`Tokyo`

`Beijing`

...

Evidence này phù hợp với finding M2 rằng `Merchant State` không phải một field chỉ chứa US-state code.

Ít nhất một phần đáng kể residual Zip missing phản ánh representation của merchant quốc tế: physical location vẫn tồn tại nhưng Zip theo representation hiện tại không có.

Vì vậy residual Zip missing không có cùng semantic với Zip missing ở `Merchant City = ONLINE`.

Một global replacement duy nhất cho mọi Zip missing sẽ xóa khác biệt này.

### Kết luận M4.3.4

Residual physical Zip missing là một semantic state riêng và phải được bảo toàn.

Policy:

Nếu:

`NON_PHYSICAL_OR_ONLINE`

thì Zip missing:

`NOT_APPLICABLE`

Nếu:

`PHYSICAL location + Zip missing`

thì:

`ZIP_UNAVAILABLE`

Không:

- mode/median imputation;
- suy diễn Zip từ Merchant City/State;
- coi mọi Zip missing là cùng một loại;
- drop transaction chỉ vì Zip missing.

`Merchant State` phải được xử lý như generic geographical categorical label có thể chứa state/province/country-like values, không phải US-state-only field.

Status:

`LOCKED`



# 8. M4.3.5 — Audit category consistency

## Câu hỏi

Các categorical string field có representation anomaly đơn giản như:

- blank string;
- leading/trailing whitespace;
- duplicate category chỉ khác whitespace;
- duplicate category chỉ khác letter case

hay không?

## Phạm vi

Use Chip
Merchant City
Merchant State

Normalization chỉ được dùng để AUDIT.

Không mutate raw value trước khi xem output.


```python
category_audit_rows = []

for column in STRING_AUDIT_COLUMNS:
    raw_unique = len(
        category_raw_values[column]
    )

    stripped_unique = len(
        category_stripped_values[column]
    )

    casefold_unique = len(
        category_casefold_values[column]
    )

    category_audit_rows.append(
        {
            "column": column,
            "blank_after_strip_count":
                category_blank_counts[
                    column
                ],
            "whitespace_affected_rows":
                category_whitespace_counts[
                    column
                ],
            "raw_unique":
                raw_unique,
            "stripped_unique":
                stripped_unique,
            "casefold_unique":
                casefold_unique,
            "raw_minus_stripped":
                (
                    raw_unique
                    - stripped_unique
                ),
            "stripped_minus_casefold":
                (
                    stripped_unique
                    - casefold_unique
                ),
        }
    )

category_audit_df = pd.DataFrame(
    category_audit_rows
)

display(category_audit_df)
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
      <th>column</th>
      <th>blank_after_strip_count</th>
      <th>whitespace_affected_rows</th>
      <th>raw_unique</th>
      <th>stripped_unique</th>
      <th>casefold_unique</th>
      <th>raw_minus_stripped</th>
      <th>stripped_minus_casefold</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Use Chip</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Merchant City</td>
      <td>0</td>
      <td>947482</td>
      <td>11437</td>
      <td>11437</td>
      <td>11437</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Merchant State</td>
      <td>0</td>
      <td>0</td>
      <td>164</td>
      <td>164</td>
      <td>164</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



```python
print(
    "Zip non-integer values:",
    zip_fractional_count,
)

print(
    "Use Chip unexpected values:"
)

print(
    sorted(
        use_chip_values
        - EXPECTED_USE_CHIP
    )
)
```

    Zip non-integer values: 0
    Use Chip unexpected values:
    []



```python
affected_city_values = sorted(
    value
    for value in category_raw_values[
        "Merchant City"
    ]
    if value != value.strip()
)

print(
    "Merchant City unique values "
    "affected by whitespace:",
    len(affected_city_values),
)

city_whitespace_examples = pd.DataFrame(
    {
        "raw_value":
            affected_city_values[:30],
        "stripped_value": [
            value.strip()
            for value
            in affected_city_values[:30]
        ],
    }
)

display(city_whitespace_examples)

print(
    "\nOTHER_INCONSISTENT "
    "location rows:"
)

print(
    location_state_counts.get(
        "OTHER_INCONSISTENT",
        0,
    )
)
```

    Merchant City unique values affected by whitespace: 1



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
      <th>raw_value</th>
      <th>stripped_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ONLINE</td>
      <td>ONLINE</td>
    </tr>
  </tbody>
</table>
</div>


    
    OTHER_INCONSISTENT location rows:
    0


### Phân tích / Nhận xét

Category audit không phát hiện blank-after-strip ở ba string categorical fields:

`Use Chip = 0`

`Merchant City = 0`

`Merchant State = 0`

`Use Chip` không có row bị ảnh hưởng bởi leading/trailing whitespace.

`Merchant State` cũng không có row bị ảnh hưởng bởi whitespace.

Riêng `Merchant City` có:

`947,482`

rows bị ảnh hưởng bởi whitespace.

Audit bổ sung xác nhận chỉ có:

`1`

unique raw Merchant City value thuộc trường hợp này, và sau `strip()` giá trị đó trở thành:

`ONLINE`

Con số `947,482` cũng đúng bằng số row thuộc `NON_PHYSICAL_OR_ONLINE`, cho thấy whitespace anomaly tập trung vào representation `ONLINE`, không phân tán ngẫu nhiên trên nhiều city.

Cardinality của Merchant City:

raw:

`11,437`

sau strip:

`11,437`

sau casefold:

`11,437`

Như vậy việc strip whitespace không làm merge hai category đang phân biệt thành một category duy nhất; nó chỉ chuẩn hóa representation của cùng một giá trị.

Đối với tất cả ba audited fields:

`raw_unique == stripped_unique == casefold_unique`

sau khi xét cardinality.

Không có evidence về duplicate category do letter case trong development data.

Vì vậy case normalization như `.lower()` hoặc `.casefold()` không đem lại lợi ích quan sát được ở current artifact và không cần áp dụng máy móc.

Zip audit cho thấy:

`Zip non-integer values = 0`

Điều này xác nhận các Zip nonmissing hiện tại có representation integer-like, dù Pandas có thể lưu field dưới dạng float vì missing values.

Điều đó không thay đổi semantic contract:

`Zip = categorical code`

chứ không phải numerical magnitude.

Use Chip unexpected values:

`[]`

Ngoài ra:

`OTHER_INCONSISTENT location rows = 0`

nên category normalization hiện tại không làm lộ thêm cấu hình location bất thường.

### Kết luận M4.3.5

Category normalization policy được khóa như sau:

Đối với string categorical fields:

`strip leading/trailing whitespace`

là deterministic normalization hợp lệ.

Trong current artifact, thay đổi thực tế chỉ ảnh hưởng Merchant City representation của `ONLINE`.

Không tự động:

- lowercase;
- uppercase toàn bộ category;
- casefold-based merging;
- fuzzy category merging.

Nếu sau normalization xuất hiện blank category hoặc unexpected domain trong artifact khác, pipeline phải xem đó là data-quality anomaly thay vì silently impute.

Zip:

`CATEGORICAL CODE`

Không dùng numeric magnitude, distance, scaling hoặc ordinal interpretation.

Use Chip domain:

`LOCKED TO THREE KNOWN CATEGORIES FOR CURRENT ARTIFACT`

Category normalization status:

`LOCKED`

Final encoding:

`OPEN — M4.4/M4.6`



# 9. M4.3.6 — Exact duplicate policy

## Evidence kế thừa

M2.5 đã thực hiện full-artifact exact duplicate audit:

24,386,900 row hashes

66 duplicate hash candidates

132 candidate rows

Sau exact verification trên toàn bộ 15 raw columns:

66 duplicate groups

132 duplicate-member rows

66 duplicate extra rows

largest group = 2

Dataset không có unique transaction_id.

Do đó exact equality trên raw public fields không chứng minh chắc chắn rằng một row là bản sao lỗi của cùng một transaction.

Quy mô duplicate extra row chỉ khoảng 0.000271% toàn artifact.

M4.3 không chạy lại expensive duplicate audit vì:

- artifact đã được M4.2 xác minh nhất quán;
- M2.5 đã có exact verification;
- không có evidence mới yêu cầu tái tính.

## Câu hỏi policy

A.
drop_duplicates() mặc định

B.
giữ tất cả transaction

C.
cách khác

## Nguyên tắc

Không được xóa transaction nếu chưa có semantic evidence rằng row đó là corruption.

### Phân tích / Nhận xét

M2.5 đã thực hiện duplicate audit trên toàn raw artifact bằng hai bước:

- hash toàn bộ row để tìm candidate;
- exact comparison lại trên đầy đủ 15 raw fields.

Kết quả đã xác nhận:

`66 exact duplicate groups`

`132 duplicate-member rows`

`66 duplicate extra rows`

`largest group = 2`

Quy mô extra row chỉ khoảng:

`0.000271%`

toàn artifact.

Tuy nhiên dataset không có unique transaction identifier.

Do đó hai row giống nhau trên toàn bộ observable raw fields chưa đủ để chứng minh chúng là cùng một transaction bị copy lỗi.

Nếu thực hiện:

`drop_duplicates()`

ta sẽ đưa vào pipeline một giả định về transaction identity mà dataset không hỗ trợ.

M4.3 cũng không có evidence mới làm thay đổi kết luận này.

Vì quy mô duplicate cực nhỏ, việc giữ chúng không tạo ra một data-quality blocking issue đáng kể; ngược lại, xóa chúng sẽ là một transformation không có semantic justification đủ mạnh.

### Kết luận M4.3.6

Default duplicate policy:

`KEEP EXACT DUPLICATE ROWS`

Không sử dụng:

`drop_duplicates()`

trong preprocessing mặc định.

Các row exact-identical được xem là observed transactions trừ khi sau này có evidence/provenance hoặc unique transaction identifier chứng minh chúng là copy error.

Duplicate policy:

`LOCKED`

Blocking issue:

`NONE`


# 10. M4.3.7 — Data-quality preprocessing policy

## P01 — Negative Amount

Observed evidence:

Development data có:

`365,166 negative transactions`

tương đương:

`4.8253%`

Negative rate ổn định qua TRAIN_2015_2017, TRAIN_2018 và VALIDATION, đồng thời xuất hiện ở cả Chip, Swipe và Online Transaction.

Policy:

Giữ nguyên signed `Amount_numeric`.

Không:

- `abs()`;
- drop negative rows;
- replace negative bằng positive;
- coi negative là missing.

Có thể xem xét sign-derived feature ở M4.4 nhưng đó không phải data cleaning.

Status:

`LOCKED`

---

## P02 — Zero Amount

Observed evidence:

Development data có:

`6,008 zero-Amount transactions`

tương đương:

`0.0794%`.

Zero xuất hiện lặp lại ở cả ba temporal periods và được quan sát trong Chip/Swipe transactions.

Policy:

Giữ:

`Amount_numeric = 0`

như observed numeric value.

Không replace bằng missing và không drop transaction.

Nếu cần `is_zero_amount`, đó là candidate feature của M4.4.

Status:

`LOCKED`

---

## P03 — Extreme Amount

Observed evidence:

Development range:

`-500.00`
→
`6,613.44`

`Amount = -500` xuất hiện:

`98 rows`

Không có runtime evidence chứng minh minimum, maximum hoặc extreme Amount là corruption.

Policy:

Không clip, winsorize hoặc drop extreme Amount tại data-quality layer.

Giữ observed `Amount_numeric`.

Các model-oriented transformations tiếp tục được đánh giá ở M4.4/M4.6.

Status:

`LOCKED`

---

## P04 — Merchant City = ONLINE

Observed evidence:

`947,482`

development rows thuộc non-physical representation.

`Merchant City = ONLINE` xuất hiện ở toàn bộ Online Transaction tương ứng và thêm:

`6,426 Chip Transactions`

trong development population.

Raw `ONLINE` representation cũng là unique Merchant City value duy nhất bị ảnh hưởng bởi whitespace.

Policy:

Áp dụng `strip()` cho categorical string representation trước khi semantic comparison.

Sau normalization, giữ:

`ONLINE`

như explicit semantic category.

Không thay `Merchant City = ONLINE` đơn giản bằng `Use Chip = Online Transaction`, vì hai khái niệm không hoàn toàn tương đương.

Status:

`LOCKED`

---

## P05 — Merchant State missing

Observed evidence:

Development State missing khớp hoàn toàn với:

`Merchant City = ONLINE`

trong current audit.

Không xuất hiện `OTHER_INCONSISTENT` location state.

Policy:

Không mode-impute Merchant State.

Đối với:

`NON_PHYSICAL_OR_ONLINE`

State missing được biểu diễn semantic là:

`NOT_APPLICABLE`

Không coi đây là generic unknown missing.

Status:

`LOCKED`

---

## P06 — Zip missing

Observed evidence:

Zip missing gồm ít nhất hai mechanism:

`947,482 NON_PHYSICAL_OR_ONLINE rows`

và:

`50,142 PHYSICAL_ZIP_UNAVAILABLE rows`.

Residual physical Zip missing chứa nhiều international merchant locations.

Policy:

Không dùng một global missing representation cho cả hai mechanism.

Semantic mapping:

`NON_PHYSICAL_OR_ONLINE`
→
`Zip = NOT_APPLICABLE`

`PHYSICAL_ZIP_UNAVAILABLE`
→
`Zip = ZIP_UNAVAILABLE`

Không mode/median-impute, suy diễn Zip hoặc drop transaction.

Status:

`LOCKED`

---

## P07 — Exact duplicate

Inherited evidence:

`66 exact duplicate groups`

`132 duplicate-member rows`

`66 duplicate extra rows`

Dataset không có unique transaction_id.

Policy:

Giữ exact duplicate rows.

Không chạy `drop_duplicates()` trong default preprocessing pipeline.

Chỉ thay đổi policy nếu có evidence mới về transaction identity/provenance.

Status:

`LOCKED`

---

## P08 — Category normalization

Observed evidence:

Blank after strip:

`0`

Whitespace-affected rows:

`Use Chip = 0`

`Merchant State = 0`

`Merchant City = 947,482`

Chỉ một unique Merchant City raw value bị ảnh hưởng và normalized value là:

`ONLINE`

Không có cardinality reduction sau strip/casefold.

Policy:

Áp dụng:

`str.strip()`

cho string categorical representation.

Không tự động:

- lowercase;
- uppercase;
- casefold merge;
- fuzzy merge.

Unexpected/blank category sau deterministic normalization phải được flag thay vì silently impute.

Status:

`LOCKED`

---

## P09 — Zip semantic type

Inherited contract:

`Zip = categorical code`

Observed runtime evidence:

`Zip non-integer values = 0`

trong development data.

Policy:

Zip không được xử lý như continuous numerical magnitude.

Không:

- scale Zip;
- standardize Zip numerically;
- sử dụng arithmetic distance;
- giả định ordinal relation.

Cách chuyển nonmissing Zip sang categorical representation và encoding cụ thể được triển khai ở M4.4/M4.6.

Status:

`LOCKED`


# 11. M4.3.8 — Policy safety checks


```python
# ============================================================
# Development/test isolation
# ============================================================

assert (
    development_rows
    == 7_567_728
)

assert (
    development_max_timestamp
    < VALIDATION_END
)

assert "Is Fraud?" not in M43_USECOLS


# ============================================================
# Representation integrity
# ============================================================

assert amount_parse_failures == 0

assert timestamp_parse_failures == 0

assert (
    use_chip_values
    == EXPECTED_USE_CHIP
)


# ============================================================
# M4.3 does not learn preprocessing state
# ============================================================

LEARNED_PREPROCESSING_STATE_CREATED = False

assert (
    LEARNED_PREPROCESSING_STATE_CREATED
    is False
)


print(
    "M4.3 POLICY SAFETY GATE: PASS"
)
```

    M4.3 POLICY SAFETY GATE: PASS


### Gate interpretation

Cell policy safety check trả về:

`M4.3 POLICY SAFETY GATE: PASS`

Runtime xác nhận:

- đúng `7,567,728` development rows;
- development boundary kết thúc trước `2019-06-01`;
- không sử dụng `Is Fraud?`;
- Amount parse failures bằng `0`;
- Timestamp parse failures bằng `0`;
- Use Chip domain đúng canonical values;
- M4.3 không tạo learned preprocessing state.

Các policy được quyết định từ TRAIN + VALIDATION development representation và không sử dụng FINAL TEST feature statistics hoặc target.

M4.3 cũng chưa thực hiện:

- one-hot encoding;
- frequency encoding;
- scaling;
- learned imputation;
- target encoding;
- feature selection;
- model fitting.

Do đó policy audit vẫn nằm đúng phạm vi data-quality preprocessing decision và chưa vượt sang learned preprocessing/modeling.

Safety Gate:

`PASS`

Blocking issue:

`NONE`



# 12. M4.3 Findings

## M4.3-F01 — Amount sign / zero

Observed fact:

Negative Amount và zero Amount tồn tại lặp lại và có temporal/mode structure trong development data.

Evidence:

Negative:

`365,166 — 4.8253%`

Zero:

`6,008 — 0.0794%`

Negative rate theo period:

`4.8414%`
→
`4.7812%`
→
`4.8159%`

Zero rate theo period:

`0.07985%`
→
`0.07923%`
→
`0.07650%`

Negative rate cũng khác rõ theo transaction mode, nhưng pattern này tồn tại ở cả TRAIN và VALIDATION.

Interpretation:

Negative/zero Amount không phù hợp với giả thuyết corruption ngẫu nhiên đơn giản.

Sign có khả năng mang semantic của transaction representation và không nên bị xóa ở cleaning layer.

Implication:

Giữ nguyên signed Amount và zero Amount.

Không `abs()`, drop hoặc replace.

Derived sign/zero features có thể được xem xét sau.

Status:

`CONFIRMED`

---

## M4.3-F02 — Amount extremes

Observed fact:

Development Amount range:

`-500.00`
→
`6,613.44`

Minimum `-500` xuất hiện:

`98 times`

Evidence:

Không có parse failure và không có runtime evidence cho thấy các extreme value là malformed numeric representation.

Interpretation:

Repeated `-500` có thể phản ánh một dataset convention/boundary, nhưng semantic chính xác chưa được chứng minh.

Maximum development Amount cũng chưa có evidence để gọi là corruption.

Implication:

Không clip/winsorize/drop Amount extremes ở data-quality layer.

Model-oriented transformation tiếp tục OPEN.

Status:

`CONFIRMED`

---

## M4.3-F03 — Structural location missingness

Observed fact:

Development location representation được phân chia thành:

`PHYSICAL_COMPLETE = 86.8174%`

`NON_PHYSICAL_OR_ONLINE = 12.5200%`

`PHYSICAL_ZIP_UNAVAILABLE = 0.6626%`

`OTHER_INCONSISTENT = 0`

Evidence:

Các state tồn tại qua TRAIN_2015_2017, TRAIN_2018 và VALIDATION.

Merchant City = ONLINE và State missing có count khớp chính xác trong current development audit.

Interpretation:

Location missing là structural.

Missing State/Zip ở non-physical representation không mang cùng semantic với residual Zip missing ở physical merchants.

Implication:

Không drop missing-location rows và không global-impute.

Preprocessing phải bảo toàn semantic state.

Status:

`CONFIRMED`

---

## M4.3-F04 — Residual Zip missing

Observed fact:

Có:

`50,142`

physical transactions không có Zip.

Evidence:

Top Merchant State values gồm nhiều country names như Mexico, Canada, Italy, United Kingdom, France, Germany, China, India và Japan.

Interpretation:

Ít nhất một phần đáng kể residual Zip missing gắn với international physical merchant representation.

Merchant State không phải US-state-only field.

Implication:

Physical Zip missing phải được phân biệt với non-physical Zip not-applicable.

Không suy diễn Zip và không áp dụng US-only state/Zip assumptions.

Status:

`CONFIRMED`

---

## M4.3-F05 — Category consistency

Observed fact:

Không có blank-after-strip category.

Use Chip và Merchant State không có whitespace anomaly.

Merchant City có:

`947,482 whitespace-affected rows`

nhưng chỉ:

`1 unique affected raw value`

và stripped representation là:

`ONLINE`

Cardinality không thay đổi sau strip.

Casefold cũng không làm giảm cardinality.

Zip nonmissing values đều integer-like:

`fractional count = 0`

Interpretation:

Current artifact có một deterministic whitespace representation issue ở Merchant City ONLINE nhưng không có evidence về broader case inconsistency.

Implication:

`str.strip()` là normalization hợp lệ.

Không cần casefold/lowercase merge.

Zip tiếp tục là categorical code.

Status:

`CONFIRMED`

---

## M4.3-F06 — Exact duplicate

Observed fact:

M2.5 confirmed:

`66 exact duplicate groups`

`132 member rows`

`66 extra rows`

Interpretation:

Exact equality không đủ chứng minh duplicate corruption vì dataset không có unique transaction identifier.

Quy mô duplicate rất nhỏ và không phải blocking data-quality issue.

Implication:

Giữ exact duplicate rows trong default preprocessing.

Không `drop_duplicates()` nếu không có provenance evidence mới.

Status:

`CONFIRMED`


# 13. Decision Log M4.3

## M4.3-D01 — Negative Amount

Decision:

Giữ nguyên signed `Amount_numeric`.

Không:

- `abs()`;
- drop negative;
- clip vì sign;
- replace negative bằng missing.

Sign-derived feature có thể được đánh giá ở M4.4.

Status:

`LOCKED`

---

## M4.3-D02 — Zero Amount

Decision:

Giữ `Amount_numeric = 0` như observed value.

Không coi zero là missing và không drop zero transaction.

Status:

`LOCKED`

---

## M4.3-D03 — Extreme Amount

Decision:

Không clip, winsorize hoặc drop Amount extremes tại data-quality layer.

Giữ observed Amount.

Transformation phục vụ model tiếp tục được đánh giá ở M4.4/M4.6.

Status:

`LOCKED`

---

## M4.3-D04 — Structural location missing

Decision:

Không drop row và không generic-impute State/Zip.

Phân biệt ít nhất:

`NON_PHYSICAL_OR_ONLINE`

`PHYSICAL_COMPLETE`

`PHYSICAL_ZIP_UNAVAILABLE`

Non-physical missing location được diễn giải là:

`NOT_APPLICABLE`

Status:

`LOCKED`

---

## M4.3-D05 — Residual Zip missing

Decision:

Physical Zip missing được giữ như semantic state riêng:

`ZIP_UNAVAILABLE`

Không gộp với non-physical:

`NOT_APPLICABLE`

Không mode/median-impute hoặc suy diễn Zip.

Status:

`LOCKED`

---

## M4.3-D06 — Merchant State semantics

Decision:

`Merchant State` là generic geographical categorical field.

Không giả định field chỉ chứa US state code.

Ở non-physical representation, missing State mang semantic:

`NOT_APPLICABLE`

Status:

`LOCKED`

---

## M4.3-D07 — Exact duplicate

Decision:

Giữ exact duplicate rows.

Không áp dụng default:

`drop_duplicates()`

Quyết định chỉ được mở lại nếu có evidence mới về transaction identity/provenance.

Status:

`LOCKED`

---

## M4.3-D08 — Category normalization

Decision:

Áp dụng deterministic:

`str.strip()`

cho string categorical representation.

Không tự động:

- lowercase;
- uppercase;
- casefold merge;
- fuzzy category merge.

Current Merchant City whitespace anomaly được chuẩn hóa về:

`ONLINE`

Status:

`LOCKED`

---

## M4.3-D09 — FINAL TEST isolation

Decision:

M4.3 data-quality policy không sử dụng FINAL TEST statistics hoặc target.

Current implementation chỉ aggregate:

`TRAIN_2015_2017`

`TRAIN_2018`

`VALIDATION`

cho policy decisions.

Status:

`INHERITED — VERIFIED — LOCKED`

# 14. M4.3 Gate

## G01 — Development-data isolation

Result:

`PASS`

Audit sử dụng đúng:

`7,567,728`

development rows và dừng trước `2019-06-01`.

---

## G02 — Amount evidence sufficient

Result:

`PASS`

Amount sign/zero được audit overall, theo temporal period và theo transaction mode.

Development min/max và `Amount = -500` cũng đã được kiểm tra.

---

## G03 — Negative/zero policy explicit

Result:

`PASS`

Negative:

`PRESERVE SIGN`

Zero:

`PRESERVE AS OBSERVED VALUE`

Không abs/drop/replace.

---

## G04 — Extreme Amount policy explicit

Result:

`PASS`

Không clip/winsorize/drop extreme Amount ở data-quality layer.

---

## G05 — Structural location policy explicit

Result:

`PASS`

Location được phân biệt thành semantic states và không sử dụng generic missing treatment.

---

## G06 — Residual Zip policy explicit

Result:

`PASS`

Non-physical Zip:

`NOT_APPLICABLE`

Physical missing Zip:

`ZIP_UNAVAILABLE`

Không gộp hai mechanism.

---

## G07 — Duplicate policy explicit

Result:

`PASS`

Giữ exact duplicate rows.

Không `drop_duplicates()` theo mặc định.

---

## G08 — Category consistency audited

Result:

`PASS`

Đã kiểm tra:

- blank;
- whitespace;
- stripped cardinality;
- casefold cardinality;
- Use Chip domain;
- Zip fractional representation.

---

## G09 — No semantic-destroying generic imputation

Result:

`PASS`

Không global mode/median-impute structural location missingness.

Không replace negative/zero Amount bằng generic missing value.

---

## G10 — No premature encoding

Result:

`PASS`

M4.3 chỉ khóa semantic preprocessing policy.

Chưa thực hiện:

- one-hot;
- ordinal encoding;
- frequency encoding;
- target encoding;
- scaling.

---

## G11 — No learned preprocessing state

Result:

`PASS`

Notebook không tạo learned preprocessing state.

Policy safety assertion:

`M4.3 POLICY SAFETY GATE: PASS`

---

## G12 — FINAL TEST remains protected

Result:

`PASS`

Không sử dụng FINAL TEST feature distribution hoặc target để chọn data-quality policy.


# M4.3 Gate

Overall:

`PASS`

Blocking issue:

`NONE`

M4.3 Status:

`PASS — READY FOR M4.4`


# 15. Kết luận M4.3

## Mục tiêu đã kiểm tra

M4.3 kiểm tra cách xử lý các vấn đề data-quality đã được phát hiện từ M2 trên canonical representation đã khóa ở M4.2.

Mục tiêu không phải làm sạch dữ liệu theo checklist chung mà là xác định preprocessing policy bảo toàn transaction semantics và temporal/test guardrail.

Các nhóm được kiểm tra gồm:

- negative Amount;
- zero Amount;
- Amount extremes;
- structural location missingness;
- residual physical Zip missing;
- category consistency;
- exact duplicate;
- FINAL TEST isolation.

---

## Những gì được xác minh

Development audit sử dụng:

`7,567,728 transactions`

gồm:

`TRAIN_2015_2017 = 5,133,655`

`TRAIN_2018 = 1,721,615`

`VALIDATION = 712,458`

Không sử dụng FINAL TEST statistics hoặc target.

Negative Amount chiếm:

`4.8253%`

development data và xuất hiện ổn định qua ba temporal periods.

Zero Amount chiếm:

`0.0794%`

và cũng xuất hiện lặp lại xuyên temporal development data.

Không có evidence để coi negative, zero hoặc observed extreme Amount là corruption cần sửa/xóa.

Location audit xác định ba semantic states bao phủ toàn development population:

`PHYSICAL_COMPLETE`

`NON_PHYSICAL_OR_ONLINE`

`PHYSICAL_ZIP_UNAVAILABLE`

Không xuất hiện:

`OTHER_INCONSISTENT`

State missing gắn hoàn toàn với `Merchant City = ONLINE` trong development representation.

Zip missing gồm cả non-physical not-applicable và physical unavailable mechanisms.

Residual physical Zip missing có:

`50,142 transactions`

và nhiều Merchant State values là country names, xác nhận không thể xem Merchant State như US-state-only field.

Category audit phát hiện deterministic whitespace issue ở Merchant City ONLINE nhưng không phát hiện blank, case-duplicate hoặc unexpected Use Chip category.

Exact duplicate evidence không đủ để chứng minh copy corruption vì dataset không có unique transaction identifier.

---

## Data-quality policy được khóa

### Amount

Giữ signed `Amount_numeric`.

Giữ zero Amount.

Không:

- `abs()`;
- drop negative;
- replace zero;
- clip/winsorize extreme Amount ở cleaning layer.

### Location

Không drop transaction vì State/Zip missing.

Không generic mode/median imputation.

Phân biệt:

`NON_PHYSICAL_OR_ONLINE`

`PHYSICAL_COMPLETE`

`PHYSICAL_ZIP_UNAVAILABLE`

Non-physical State/Zip:

`NOT_APPLICABLE`

Physical missing Zip:

`ZIP_UNAVAILABLE`

### Merchant State

Xử lý như generic geographical categorical value.

Không áp đặt US-state-only semantics.

### Category normalization

Áp dụng deterministic:

`str.strip()`

Không tự động casefold/lowercase/fuzzy merge.

### Zip

Giữ semantic:

`CATEGORICAL CODE`

Không coi Zip là continuous numeric feature.

### Duplicate

Giữ exact duplicate rows.

Không default `drop_duplicates()`.

---

## Những policy bị loại bỏ

M4.3 loại bỏ các default cleaning rule sau:

`negative Amount → abs()`

`negative Amount → drop`

`zero Amount → missing`

`extreme Amount → automatic clipping`

`missing location → drop row`

`missing State/Zip → global mode/median imputation`

`all Zip missing → same semantic category`

`Merchant State → US-state-only interpretation`

`exact duplicate → automatic drop`

`categorical string → automatic casefold/fuzzy merge`

`Zip → continuous numeric treatment`

Các policy trên hoặc phá semantic quan sát được, hoặc đưa vào giả định không được evidence hỗ trợ.

---

## Những vấn đề tiếp tục OPEN

M4.3 chưa quyết định:

- có tạo `is_negative_amount` hay không;
- có tạo `is_zero_amount` hay không;
- Amount transformation cho model;
- Amount scaling;
- final transaction-level location features;
- có đưa `location_state` vào model hay chỉ dùng trong preprocessing;
- final Merchant City/State/Zip inclusion;
- final MCC representation;
- categorical encoding;
- rare-category handling;
- learned preprocessing state;
- feature selection.

Các câu hỏi này thuộc M4.4 và M4.6.

---

## Blocking issue

`NONE`

Không phát hiện data-quality pattern nào yêu cầu thay đổi M4.2 Base Representation Contract hoặc chặn transaction-level feature engineering.

---

## M4.3 Gate

`PASS`

Các data-quality policy cần thiết cho bước tiếp theo đã được xác định rõ và không vi phạm FINAL TEST isolation.

---

## Trạng thái cuối

`M4.3 — PASS`

`Data-quality preprocessing policy — LOCKED`

`READY FOR M4.4`

---

## Handoff

Next:

`M4.4 — Thiết kế transaction-level feature representation`

M4.4 kế thừa các policy đã khóa:

- preserve signed Amount;
- preserve zero Amount;
- no automatic clipping;
- semantic location-state separation;
- `NOT_APPLICABLE` khác `ZIP_UNAVAILABLE`;
- deterministic string strip;
- Zip là categorical code;
- exact duplicates được giữ.

M4.4 có thể tập trung vào việc chuyển các policy này thành candidate transaction-level features, ví dụ:

- Amount-derived representation;
- transaction mode;
- MCC representation;
- location representation;
- calendar/time features.

M4.4 không được đảo ngược các data-quality decisions của M4.3 chỉ vì một representation khác thuận tiện hơn cho model.
