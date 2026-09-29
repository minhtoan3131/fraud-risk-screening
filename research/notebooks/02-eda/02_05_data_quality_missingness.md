# M2.5 — Phân tích chất lượng dữ liệu và missingness

## Mục tiêu

Sau khi M2.4 đã mô tả distribution của các feature, M2.5 tập trung vào những giá trị hoặc pattern có thể khiến ta hiểu sai dữ liệu hoặc đưa ra quyết định preprocessing sai. <br>

Các câu hỏi trung tâm gồm: <br>
- missing value thực sự nằm ở những cột nào? <br>
- missing của location là ngẫu nhiên hay phản ánh cấu trúc giao dịch? <br>
- `Merchant City = ONLINE` liên hệ thế nào với `Online Transaction` và location missing? <br>
- Amount âm và Amount bằng 0 xuất hiện theo cấu trúc nào? <br>
- các Amount cực trị là những transaction như thế nào? <br>
- exact duplicate đã biết ở M1 thực sự phân bố ra sao? <br>

## Tư duy phân tích

M2.5 không bắt đầu bằng câu hỏi: <br>
`Ta nên sửa dữ liệu thế nào?` <br>

Mà bắt đầu bằng: <br>
`Hiện tượng này là lỗi dữ liệu, giá trị hợp lệ có semantic đặc biệt, hay pattern có nguyên nhân cấu trúc?` <br>

Chỉ sau khi hiểu được hiện tượng mới chuyển quyết định xử lý sang M4 — Preprocessing. <br>

## Ranh giới

Trong M2.5: <br>
- không fill missing; <br>
- không drop missing; <br>
- không drop duplicate; <br>
- không dùng `abs()` cho Amount âm; <br>
- không clip Amount cực trị; <br>
- không quyết định encoding; <br>
- không quyết định final feature set; <br>
- không thực hiện phân tích feature ↔ fraud nói chung. <br>

Target chỉ được sử dụng ở phần duplicate để kiểm tra liệu duplicate-member rows có chứa positive class hay không. <br>

Quan hệ `Amount / Use Chip / MCC / time / location ↔ fraud` được dành cho M2.6. <br>

## Thiết lập môi trường thực thi

Notebook này được thiết kế để `chạy độc lập`, không phụ thuộc kernel hoặc biến của các notebook M2 trước. <br>

Do M2.5 cần đồng thời kiểm tra missing, Amount và exact duplicate, dataset sẽ được quét theo chunk với kích thước nhỏ hơn M2.4 để giới hạn mức sử dụng RAM. <br>

Trong lần quét đầu, notebook chỉ thu thập thống kê và row hash; không giữ toàn bộ 24 triệu transaction trong bộ nhớ. <br>


```python
from pathlib import Path
from collections import Counter
import gc

import numpy as np
import pandas as pd

from IPython.display import display


# ============================================================
# Xác định thư mục gốc của project
# ============================================================

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

    if (
        candidate
        / DATA_RELATIVE_PATH
    ).exists():
        PROJECT_ROOT = candidate
        break

if PROJECT_ROOT is None:
    raise FileNotFoundError(
        "Không xác định được PROJECT_ROOT. "
        "Không tìm thấy file dữ liệu."
    )


DATA_PATH = (
    PROJECT_ROOT
    / DATA_RELATIVE_PATH
)


# ============================================================
# Cấu hình M2.5
# ============================================================

CHUNK_SIZE = 250_000

EXTREME_PREVIEW_N = 10

CATEGORY_PREVIEW_N = 15


EXPECTED_COLUMNS = [
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
    "MCC",
    "Errors?",
    "Is Fraud?",
]


print("PROJECT_ROOT:")
print(PROJECT_ROOT)

print("\nDATA_PATH:")
print(DATA_PATH)

print("\nFile tồn tại:")
print(DATA_PATH.exists())

print("\nCHUNK_SIZE:")
print(CHUNK_SIZE)
```

    PROJECT_ROOT:
    /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening
    
    DATA_PATH:
    /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/raw/ibm_tabformer/card_transaction.v1.csv
    
    File tồn tại:
    True
    
    CHUNK_SIZE:
    250000


### Nhận xét thiết lập môi trường

Notebook đã xác định được đúng `PROJECT_ROOT` và raw artifact. <br>

Kết quả `File tồn tại: True` xác nhận M2.5 có thể truy cập trực tiếp dataset và chạy độc lập, không phụ thuộc vào kernel hoặc biến được tạo ở các notebook M2 trước. <br>

`CHUNK_SIZE = 250,000` được sử dụng để giảm áp lực bộ nhớ trong bước quality audit, đặc biệt vì notebook đồng thời thu thập row hash phục vụ kiểm tra exact duplicate. <br>

`Kết luận: PASS`

## M2.5.1 — Quét toàn bộ dataset để thu thập bằng chứng chất lượng dữ liệu

### Vì sao kiểm tra này tồn tại?

M2.5 cần kiểm tra nhiều hiện tượng cùng lúc. <br>

Nếu đọc file riêng cho missing, Amount, location và duplicate thì cùng một file hơn 2 GB sẽ bị quét nhiều lần không cần thiết. <br>

Vì vậy một full scan được dùng để thu thập đồng thời các bằng chứng cần thiết. <br>

### Phương pháp

Trong mỗi chunk sẽ thu thập: <br>
- missing count của toàn bộ 15 cột; <br>
- Amount âm / zero / dương theo `Use Chip`; <br>
- các transaction có Amount thấp nhất / cao nhất; <br>
- quan hệ giữa `Use Chip`, `Merchant City = ONLINE`, State missing và Zip missing; <br>
- các location pattern còn missing ngoài Online Transaction; <br>
- row hash phục vụ tìm exact duplicate mà không giữ toàn dataset trong RAM. <br>


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


# ============================================================
# Tổng số dòng / missing
# ============================================================

total_rows = 0
chunk_count = 0

missing_counts = Counter()


# ============================================================
# Amount
# ============================================================

amount_sign_by_mode_counts = Counter()

amount_negative_count = 0
amount_zero_count = 0
amount_positive_count = 0

extreme_low_parts = []
extreme_high_parts = []


# ============================================================
# Location / transaction mode
# ============================================================

use_chip_counts = Counter()

city_online_by_mode = Counter()
state_missing_by_mode = Counter()
zip_missing_by_mode = Counter()
both_missing_by_mode = Counter()

location_pattern_counts = Counter()

city_online_count = 0

online_and_city_online_count = 0
online_and_state_missing_count = 0
online_and_zip_missing_count = 0

city_online_and_state_missing_count = 0
city_online_and_zip_missing_count = 0


# Missing còn lại ngoài Online Transaction
physical_state_missing_city_counts = Counter()
physical_zip_missing_city_counts = Counter()

zip_missing_state_present_state_counts = Counter()


# ============================================================
# Hash phục vụ duplicate audit
# ============================================================

row_hash_parts = []


AMOUNT_PREVIEW_COLUMNS = [
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
    "MCC",
]


# ============================================================
# Full scan
# ============================================================

for chunk_number, chunk in enumerate(
    pd.read_csv(
        DATA_PATH,
        usecols=EXPECTED_COLUMNS,
        chunksize=CHUNK_SIZE,
    ),
    start=1,
):
    chunk_count = chunk_number
    total_rows += len(chunk)

    # --------------------------------------------------------
    # 1. Missing count
    # --------------------------------------------------------

    for column in EXPECTED_COLUMNS:
        missing_counts[column] += int(
            chunk[column]
            .isna()
            .sum()
        )

    # --------------------------------------------------------
    # 2. Amount
    # --------------------------------------------------------

    amount_numeric = parse_amount(
        chunk["Amount"]
    )

    amount_negative_count += int(
        (amount_numeric < 0).sum()
    )

    amount_zero_count += int(
        (amount_numeric == 0).sum()
    )

    amount_positive_count += int(
        (amount_numeric > 0).sum()
    )


    amount_sign = np.select(
        [
            amount_numeric < 0,
            amount_numeric == 0,
            amount_numeric > 0,
        ],
        [
            "negative",
            "zero",
            "positive",
        ],
        default="missing_or_invalid",
    )


    amount_sign_mode_df = pd.DataFrame(
        {
            "Use Chip":
                chunk["Use Chip"],

            "amount_sign":
                amount_sign,
        }
    )


    sign_mode_counts = (
        amount_sign_mode_df
        .value_counts(
            dropna=False
        )
    )


    amount_sign_by_mode_counts.update(
        {
            tuple(key): int(value)
            for key, value
            in sign_mode_counts.items()
        }
    )


    # --------------------------------------------------------
    # Extreme Amount candidate rows
    # --------------------------------------------------------

    amount_preview = (
        chunk[
            AMOUNT_PREVIEW_COLUMNS
        ]
        .copy()
    )

    amount_preview[
        "Amount_numeric"
    ] = amount_numeric


    extreme_low_parts.append(
        amount_preview.nsmallest(
            EXTREME_PREVIEW_N,
            "Amount_numeric",
        )
    )

    extreme_high_parts.append(
        amount_preview.nlargest(
            EXTREME_PREVIEW_N,
            "Amount_numeric",
        )
    )

    # --------------------------------------------------------
    # 3. Transaction mode + location
    # --------------------------------------------------------

    use_chip = (
        chunk["Use Chip"]
        .astype("string")
    )

    online_mode = (
        use_chip
        .eq(
            "Online Transaction"
        )
        .fillna(False)
    )

    city_online = (
        chunk["Merchant City"]
        .astype("string")
        .str.strip()
        .str.upper()
        .eq("ONLINE")
        .fillna(False)
    )

    state_missing = (
        chunk["Merchant State"]
        .isna()
    )

    zip_missing = (
        chunk["Zip"]
        .isna()
    )

    both_missing = (
        state_missing
        & zip_missing
    )


    use_chip_counts.update(
        use_chip
        .dropna()
        .value_counts()
        .to_dict()
    )

    city_online_by_mode.update(
        use_chip[
            city_online
        ]
        .dropna()
        .value_counts()
        .to_dict()
    )

    state_missing_by_mode.update(
        use_chip[
            state_missing
        ]
        .dropna()
        .value_counts()
        .to_dict()
    )

    zip_missing_by_mode.update(
        use_chip[
            zip_missing
        ]
        .dropna()
        .value_counts()
        .to_dict()
    )

    both_missing_by_mode.update(
        use_chip[
            both_missing
        ]
        .dropna()
        .value_counts()
        .to_dict()
    )


    city_online_count += int(
        city_online.sum()
    )

    online_and_city_online_count += int(
        (
            online_mode
            & city_online
        ).sum()
    )

    online_and_state_missing_count += int(
        (
            online_mode
            & state_missing
        ).sum()
    )

    online_and_zip_missing_count += int(
        (
            online_mode
            & zip_missing
        ).sum()
    )

    city_online_and_state_missing_count += int(
        (
            city_online
            & state_missing
        ).sum()
    )

    city_online_and_zip_missing_count += int(
        (
            city_online
            & zip_missing
        ).sum()
    )


    # --------------------------------------------------------
    # Joint location pattern
    # --------------------------------------------------------

    location_pattern_df = pd.DataFrame(
        {
            "Use Chip":
                use_chip,

            "city_online":
                city_online,

            "state_missing":
                state_missing,

            "zip_missing":
                zip_missing,
        }
    )


    location_pattern_chunk_counts = (
        location_pattern_df
        .value_counts(
            dropna=False
        )
    )


    location_pattern_counts.update(
        {
            tuple(key): int(value)
            for key, value
            in location_pattern_chunk_counts.items()
        }
    )


    # --------------------------------------------------------
    # 4. Residual missingness ngoài Online Transaction
    # --------------------------------------------------------

    physical_mode = (
        ~online_mode
    )


    physical_state_missing_city_counts.update(
        chunk.loc[
            physical_mode
            & state_missing,
            "Merchant City",
        ]
        .fillna(
            "<MISSING>"
        )
        .astype(str)
        .value_counts()
        .to_dict()
    )


    physical_zip_missing_city_counts.update(
        chunk.loc[
            physical_mode
            & zip_missing,
            "Merchant City",
        ]
        .fillna(
            "<MISSING>"
        )
        .astype(str)
        .value_counts()
        .to_dict()
    )


    zip_missing_state_present_state_counts.update(
        chunk.loc[
            physical_mode
            & zip_missing
            & (~state_missing),
            "Merchant State",
        ]
        .fillna(
            "<MISSING>"
        )
        .astype(str)
        .value_counts()
        .to_dict()
    )


    # --------------------------------------------------------
    # 5. Hash toàn row phục vụ duplicate audit
    # --------------------------------------------------------

    row_hash = (
        pd.util
        .hash_pandas_object(
            chunk[
                EXPECTED_COLUMNS
            ],
            index=False,
        )
        .to_numpy(
            dtype=np.uint64,
            copy=True,
        )
    )


    row_hash_parts.append(
        row_hash
    )


    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    if (
        chunk_number % 10 == 0
        or len(chunk) < CHUNK_SIZE
    ):
        print(
            f"Đã xử lý {chunk_number} chunk "
            f"- tổng số dòng: {total_rows:,}"
        )


# ============================================================
# Global extreme rows
# ============================================================

extreme_low_amount_df = (
    pd.concat(
        extreme_low_parts,
        ignore_index=True,
    )
    .nsmallest(
        EXTREME_PREVIEW_N,
        "Amount_numeric",
    )
    .reset_index(drop=True)
)


extreme_high_amount_df = (
    pd.concat(
        extreme_high_parts,
        ignore_index=True,
    )
    .nlargest(
        EXTREME_PREVIEW_N,
        "Amount_numeric",
    )
    .reset_index(drop=True)
)


del extreme_low_parts
del extreme_high_parts

gc.collect()


print("\nHoàn tất full scan M2.5.")

print(
    "Số chunk:",
    chunk_count,
)

print(
    "Tổng số dòng:",
    f"{total_rows:,}",
)

print(
    "Số phần row hash:",
    len(row_hash_parts),
)
```

    Đã xử lý 10 chunk - tổng số dòng: 2,500,000
    Đã xử lý 20 chunk - tổng số dòng: 5,000,000
    Đã xử lý 30 chunk - tổng số dòng: 7,500,000
    Đã xử lý 40 chunk - tổng số dòng: 10,000,000
    Đã xử lý 50 chunk - tổng số dòng: 12,500,000
    Đã xử lý 60 chunk - tổng số dòng: 15,000,000
    Đã xử lý 70 chunk - tổng số dòng: 17,500,000
    Đã xử lý 80 chunk - tổng số dòng: 20,000,000
    Đã xử lý 90 chunk - tổng số dòng: 22,500,000
    Đã xử lý 98 chunk - tổng số dòng: 24,386,900
    
    Hoàn tất full scan M2.5.
    Số chunk: 98
    Tổng số dòng: 24,386,900
    Số phần row hash: 98


### Nhận xét M2.5.1

Toàn bộ dataset đã được quét thành công qua `98 chunk`, với tổng cộng `24,386,900 transaction`. <br>

Trong cùng một full scan, notebook đã thu thập được các bằng chứng cần thiết cho: <br>
- missingness toàn schema; <br>
- Amount âm / zero / dương; <br>
- Amount cực trị; <br>
- quan hệ giữa transaction mode và location missing; <br>
- các location pattern còn lại ngoài Online Transaction; <br>
- row hash phục vụ exact duplicate audit. <br>

Tổng số dòng được xử lý khớp với raw artifact đã xác minh ở các bước trước. <br>

Notebook cũng tạo đủ `98 phần row hash`, tương ứng với 98 chunk, để tiếp tục duplicate audit mà không phải giữ toàn bộ DataFrame 24 triệu dòng trong RAM. <br>

`Kết luận: PASS`

## M2.5.2 — Missingness trên toàn bộ schema

### Câu hỏi

Missing thực tế tập trung ở những cột nào? <br>

Có cột cốt lõi nào xuất hiện missing mới ngoài những gì đã biết từ M1 hay không? <br>

### Phương pháp

Từ full scan, tính với từng cột: <br>
`missing_count` <br>
`missing_rate_pct` <br>
`non_missing_count` <br>

Bảng được sắp theo tỷ lệ missing giảm dần. <br>

`Errors?` vẫn được thống kê vì nó là một phần raw schema, nhưng M2.5 không nghiên cứu nó như feature candidate vì Model V1 đã loại field này. <br>


```python
missingness_df = pd.DataFrame(
    [
        {
            "column":
                column,

            "missing_count":
                int(
                    missing_counts.get(
                        column,
                        0,
                    )
                ),
        }
        for column
        in EXPECTED_COLUMNS
    ]
)


missingness_df[
    "missing_rate_pct"
] = (
    missingness_df[
        "missing_count"
    ]
    / total_rows
    * 100
)


missingness_df[
    "non_missing_count"
] = (
    total_rows
    - missingness_df[
        "missing_count"
    ]
)


missingness_df = (
    missingness_df
    .sort_values(
        "missing_count",
        ascending=False,
    )
    .reset_index(drop=True)
)


display(
    missingness_df
)


print(
    "Số cột có missing:",
    (
        missingness_df[
            "missing_count"
        ]
        > 0
    ).sum(),
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
      <th>column</th>
      <th>missing_count</th>
      <th>missing_rate_pct</th>
      <th>non_missing_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Errors?</td>
      <td>23998469</td>
      <td>98.407215</td>
      <td>388431</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Zip</td>
      <td>2878135</td>
      <td>11.801972</td>
      <td>21508765</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Merchant State</td>
      <td>2720821</td>
      <td>11.156896</td>
      <td>21666079</td>
    </tr>
    <tr>
      <th>3</th>
      <td>User</td>
      <td>0</td>
      <td>0.000000</td>
      <td>24386900</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Card</td>
      <td>0</td>
      <td>0.000000</td>
      <td>24386900</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Year</td>
      <td>0</td>
      <td>0.000000</td>
      <td>24386900</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Month</td>
      <td>0</td>
      <td>0.000000</td>
      <td>24386900</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Day</td>
      <td>0</td>
      <td>0.000000</td>
      <td>24386900</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Time</td>
      <td>0</td>
      <td>0.000000</td>
      <td>24386900</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Amount</td>
      <td>0</td>
      <td>0.000000</td>
      <td>24386900</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Use Chip</td>
      <td>0</td>
      <td>0.000000</td>
      <td>24386900</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Merchant Name</td>
      <td>0</td>
      <td>0.000000</td>
      <td>24386900</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Merchant City</td>
      <td>0</td>
      <td>0.000000</td>
      <td>24386900</td>
    </tr>
    <tr>
      <th>13</th>
      <td>MCC</td>
      <td>0</td>
      <td>0.000000</td>
      <td>24386900</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Is Fraud?</td>
      <td>0</td>
      <td>0.000000</td>
      <td>24386900</td>
    </tr>
  </tbody>
</table>
</div>


    Số cột có missing: 3


### Nhận xét M2.5.2

Missing value chỉ xuất hiện ở `3 / 15 cột`: <br>
`Errors?: 23,998,469 — 98.407215%` <br>
`Zip: 2,878,135 — 11.801972%` <br>
`Merchant State: 2,720,821 — 11.156896%` <br>

Tất cả các trường còn lại đều có `0 missing`, bao gồm: <br>
`User` <br>
`Card` <br>
`Year` <br>
`Month` <br>
`Day` <br>
`Time` <br>
`Amount` <br>
`Use Chip` <br>
`Merchant Name` <br>
`Merchant City` <br>
`MCC` <br>
`Is Fraud?` <br>

Như vậy missingness không phân bố rộng trên toàn schema mà tập trung vào một số trường cụ thể. <br>

Đặc biệt, các trường cốt lõi phục vụ target, transaction value, transaction mode và thời gian đều đầy đủ. <br>

`Errors?` có tỷ lệ missing rất cao nhưng field này đã bị loại khỏi Model V1 vì prediction-time availability chưa được chứng minh, nên M2.5 không cần xây chiến lược imputation cho Errors? như một model feature. <br>

Một điểm cần lưu ý là `Merchant City` có missing count bằng 0 nhưng điều đó không đồng nghĩa mọi transaction đều có physical city hợp lệ. <br>

M2.4 đã phát hiện giá trị `Merchant City = ONLINE`, do đó sự vắng mặt của physical location có thể được biểu diễn bằng một category thay vì bằng NaN. <br>

Vì vậy chỉ nhìn raw missing count chưa đủ để hiểu semantic missingness của location. <br>

`Kết luận: missingness tập trung ở Errors?, Merchant State và Zip; location cần được phân tích theo semantic transaction mode thay vì xem như missing ngẫu nhiên.`

## M2.5.3 — Kiểm tra structural missingness của location

### Vì sao kiểm tra này tồn tại?

Nếu missing là ngẫu nhiên, ta có thể xem nó như một vấn đề chất lượng dữ liệu thông thường. <br>

Nếu missing gần như được quyết định bởi loại transaction, missing lại mang `ý nghĩa cấu trúc` và việc drop / impute máy móc có thể phá dữ liệu. <br>

### Câu hỏi

State và Zip missing bao nhiêu trong từng loại `Use Chip`? <br>

`Merchant City = ONLINE` có đồng nhất với `Online Transaction` hay không? <br>

Missing State / Zip và category ONLINE chồng lấp với nhau đến mức nào? <br>

### Phương pháp

Với từng transaction mode, tính: <br>
- số transaction; <br>
- số và tỷ lệ `Merchant City = ONLINE`; <br>
- số và tỷ lệ State missing; <br>
- số và tỷ lệ Zip missing; <br>
- số và tỷ lệ State + Zip cùng missing. <br>


```python
location_mode_rows = []


for mode, transaction_count in (
    use_chip_counts
    .most_common()
):
    city_online_n = int(
        city_online_by_mode.get(
            mode,
            0,
        )
    )

    state_missing_n = int(
        state_missing_by_mode.get(
            mode,
            0,
        )
    )

    zip_missing_n = int(
        zip_missing_by_mode.get(
            mode,
            0,
        )
    )

    both_missing_n = int(
        both_missing_by_mode.get(
            mode,
            0,
        )
    )


    location_mode_rows.append(
        {
            "Use Chip":
                mode,

            "transaction_count":
                transaction_count,

            "city_online_count":
                city_online_n,

            "city_online_rate_pct":
                city_online_n
                / transaction_count
                * 100,

            "state_missing_count":
                state_missing_n,

            "state_missing_rate_pct":
                state_missing_n
                / transaction_count
                * 100,

            "zip_missing_count":
                zip_missing_n,

            "zip_missing_rate_pct":
                zip_missing_n
                / transaction_count
                * 100,

            "both_missing_count":
                both_missing_n,

            "both_missing_rate_pct":
                both_missing_n
                / transaction_count
                * 100,
        }
    )


location_by_mode_df = pd.DataFrame(
    location_mode_rows
)


display(
    location_by_mode_df
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
      <th>Use Chip</th>
      <th>transaction_count</th>
      <th>city_online_count</th>
      <th>city_online_rate_pct</th>
      <th>state_missing_count</th>
      <th>state_missing_rate_pct</th>
      <th>zip_missing_count</th>
      <th>zip_missing_rate_pct</th>
      <th>both_missing_count</th>
      <th>both_missing_rate_pct</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Swipe Transaction</td>
      <td>15386082</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.000000</td>
      <td>105793</td>
      <td>0.687589</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Chip Transaction</td>
      <td>6287598</td>
      <td>7601</td>
      <td>0.120889</td>
      <td>7601</td>
      <td>0.120889</td>
      <td>59122</td>
      <td>0.940295</td>
      <td>7601</td>
      <td>0.120889</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Online Transaction</td>
      <td>2713220</td>
      <td>2713220</td>
      <td>100.000000</td>
      <td>2713220</td>
      <td>100.000000</td>
      <td>2713220</td>
      <td>100.000000</td>
      <td>2713220</td>
      <td>100.000000</td>
    </tr>
  </tbody>
</table>
</div>



```python
online_transaction_count = int(
    use_chip_counts.get(
        "Online Transaction",
        0,
    )
)


state_missing_count = int(
    missing_counts.get(
        "Merchant State",
        0,
    )
)

zip_missing_count = int(
    missing_counts.get(
        "Zip",
        0,
    )
)


def safe_pct(
    numerator,
    denominator,
):
    if denominator == 0:
        return np.nan

    return (
        numerator
        / denominator
        * 100
    )


online_alignment_summary = pd.Series(
    {
        "Online Transaction count":
            online_transaction_count,

        "Merchant City = ONLINE count":
            city_online_count,

        "State missing count":
            state_missing_count,

        "Zip missing count":
            zip_missing_count,

        "P(City=ONLINE | Online Transaction) (%)":
            safe_pct(
                online_and_city_online_count,
                online_transaction_count,
            ),

        "P(Online Transaction | City=ONLINE) (%)":
            safe_pct(
                online_and_city_online_count,
                city_online_count,
            ),

        "P(State missing | Online Transaction) (%)":
            safe_pct(
                online_and_state_missing_count,
                online_transaction_count,
            ),

        "P(Online Transaction | State missing) (%)":
            safe_pct(
                online_and_state_missing_count,
                state_missing_count,
            ),

        "P(Zip missing | Online Transaction) (%)":
            safe_pct(
                online_and_zip_missing_count,
                online_transaction_count,
            ),

        "P(Online Transaction | Zip missing) (%)":
            safe_pct(
                online_and_zip_missing_count,
                zip_missing_count,
            ),

        "P(State missing | City=ONLINE) (%)":
            safe_pct(
                city_online_and_state_missing_count,
                city_online_count,
            ),

        "P(City=ONLINE | State missing) (%)":
            safe_pct(
                city_online_and_state_missing_count,
                state_missing_count,
            ),

        "P(Zip missing | City=ONLINE) (%)":
            safe_pct(
                city_online_and_zip_missing_count,
                city_online_count,
            ),
    }
)


display(
    online_alignment_summary
)
```


    Online Transaction count                     2.713220e+06
    Merchant City = ONLINE count                 2.720821e+06
    State missing count                          2.720821e+06
    Zip missing count                            2.878135e+06
    P(City=ONLINE | Online Transaction) (%)      1.000000e+02
    P(Online Transaction | City=ONLINE) (%)      9.972064e+01
    P(State missing | Online Transaction) (%)    1.000000e+02
    P(Online Transaction | State missing) (%)    9.972064e+01
    P(Zip missing | Online Transaction) (%)      1.000000e+02
    P(Online Transaction | Zip missing) (%)      9.427007e+01
    P(State missing | City=ONLINE) (%)           1.000000e+02
    P(City=ONLINE | State missing) (%)           1.000000e+02
    P(Zip missing | City=ONLINE) (%)             1.000000e+02
    dtype: float64



```python
location_pattern_df = pd.DataFrame(
    [
        {
            "Use Chip":
                key[0],

            "city_online":
                key[1],

            "state_missing":
                key[2],

            "zip_missing":
                key[3],

            "transaction_count":
                count,
        }
        for key, count
        in location_pattern_counts.items()
    ]
)


location_pattern_df[
    "transaction_share_pct"
] = (
    location_pattern_df[
        "transaction_count"
    ]
    / total_rows
    * 100
)


location_pattern_df = (
    location_pattern_df
    .sort_values(
        "transaction_count",
        ascending=False,
    )
    .reset_index(drop=True)
)


display(
    location_pattern_df
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
      <th>Use Chip</th>
      <th>city_online</th>
      <th>state_missing</th>
      <th>zip_missing</th>
      <th>transaction_count</th>
      <th>transaction_share_pct</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Swipe Transaction</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>15280289</td>
      <td>62.657775</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Chip Transaction</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>6228476</td>
      <td>25.540253</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Online Transaction</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>2713220</td>
      <td>11.125727</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Swipe Transaction</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>105793</td>
      <td>0.433811</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Chip Transaction</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>51521</td>
      <td>0.211265</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Chip Transaction</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>7601</td>
      <td>0.031168</td>
    </tr>
  </tbody>
</table>
</div>


### Nhận xét M2.5.3

Quan hệ giữa transaction mode và location missing thể hiện cấu trúc rất rõ. <br>

Đối với `Online Transaction`: <br>
`2,713,220 transaction` <br>
`Merchant City = ONLINE: 100%` <br>
`Merchant State missing: 100%` <br>
`Zip missing: 100%` <br>
`State + Zip cùng missing: 100%` <br>

Như vậy với toàn bộ Online Transaction trong artifact, physical merchant location không được biểu diễn bằng State / Zip; Merchant City được thay bằng category `ONLINE`. <br>

Tuy nhiên chiều ngược lại không hoàn toàn giống nhau. <br>

Toàn dataset có: <br>
`Online Transaction: 2,713,220` <br>
`Merchant City = ONLINE: 2,720,821` <br>
`Merchant State missing: 2,720,821` <br>

Có thêm `7,601 transaction` không thuộc Online Transaction nhưng vẫn có `Merchant City = ONLINE` và State / Zip cùng missing. <br>

Các transaction này đều nằm trong `Chip Transaction`. <br>

Do đó: <br>
`P(City=ONLINE | Online Transaction) = 100%` <br>
`P(Online Transaction | City=ONLINE) ≈ 99.72%` <br>

Nói cách khác: `Online Transaction → City=ONLINE` là quan hệ hoàn toàn trong artifact, nhưng `City=ONLINE → Online Transaction` không hoàn toàn đúng vì tồn tại 7,601 Chip Transaction mang cùng representation location. <br>

Một quan hệ còn mạnh hơn xuất hiện giữa Merchant City và Merchant State: <br>
`Merchant City = ONLINE count = 2,720,821` <br>
`Merchant State missing count = 2,720,821` <br>
`P(State missing | City=ONLINE) = 100%` <br>
`P(City=ONLINE | State missing) = 100%` <br>

Trong artifact hiện tại, `Merchant State missing` và `Merchant City = ONLINE` là hai representation tương đương về tập transaction. <br>

Điều này là bằng chứng rất mạnh rằng State missing không phải missing ngẫu nhiên mà phản ánh một cấu trúc location cụ thể. <br>

Đối với `Zip`, quan hệ rộng hơn: <br>
`P(Zip missing | Online Transaction) = 100%` <br>
nhưng <br>
`P(Online Transaction | Zip missing) ≈ 94.27%`. <br>

Do đó phần lớn Zip missing được giải thích bởi Online Transaction, nhưng vẫn còn một nhóm non-online transaction bị Zip missing cần điều tra riêng. <br>

Các pattern tổng hợp xác nhận sáu cấu hình chính, trong đó ba cấu hình lớn nhất là: <br>
`Swipe + physical location đầy đủ: 62.66% dataset` <br>
`Chip + physical location đầy đủ: 25.54%` <br>
`Online + City=ONLINE + State missing + Zip missing: 11.13%` <br>

Như vậy missing location có `structural missingness` rất rõ. <br>

Việc drop mọi row thiếu State hoặc Zip sẽ không phải một thao tác cleaning trung tính; nó sẽ loại bỏ có hệ thống một loại transaction cụ thể. <br>

Ngoài ra, việc tạo missing indicator cho State cũng có nguy cơ cung cấp thông tin gần trùng với `Merchant City = ONLINE` và rất mạnh với transaction mode. <br>

Điều này phải được cân nhắc ở M4 thay vì mặc định xem missing indicator là một feature độc lập. <br>

`Kết luận: Merchant State missing là structural missingness gần như hoàn toàn gắn với representation Merchant City = ONLINE; Zip missing cũng chủ yếu do Online Transaction nhưng còn một residual pattern ngoài Online cần được phân tích riêng.`

## M2.5.4 — Điều tra missing location còn lại ngoài Online Transaction

### Vì sao kiểm tra này tồn tại?

Nếu transaction online giải thích phần lớn missing location nhưng không giải thích `100% tổng missing`, phần dư vẫn cần được hiểu trước khi preprocessing. <br>

### Câu hỏi

Trong các transaction không phải Online: <br>
- State missing xuất hiện ở những Merchant City nào? <br>
- Zip missing xuất hiện ở những Merchant City nào? <br>
- khi Zip missing nhưng State vẫn tồn tại, State nào xuất hiện nhiều? <br>

### Phương pháp

Chỉ phân tích phần `Use Chip != Online Transaction`. <br>

Các bảng dưới đây là frequency preview để xác định liệu residual missingness có tạo ra pattern mới cần đào sâu hay không. <br>


```python
def counter_to_frequency_df(
    counter,
    column_name,
):
    df = pd.DataFrame(
        [
            {
                column_name:
                    value,

                "transaction_count":
                    count,
            }
            for value, count
            in counter.items()
        ]
    )

    if df.empty:
        return df

    return (
        df
        .sort_values(
            "transaction_count",
            ascending=False,
        )
        .reset_index(drop=True)
    )

physical_state_missing_city_df = (
    counter_to_frequency_df(
        physical_state_missing_city_counts,
        "Merchant City",
    )
)


physical_zip_missing_city_df = (
    counter_to_frequency_df(
        physical_zip_missing_city_counts,
        "Merchant City",
    )
)


zip_missing_state_present_df = (
    counter_to_frequency_df(
        zip_missing_state_present_state_counts,
        "Merchant State",
    )
)


physical_state_missing_count = (
    physical_state_missing_city_df[
        "transaction_count"
    ].sum()
    if not physical_state_missing_city_df.empty
    else 0
)


physical_zip_missing_count = (
    physical_zip_missing_city_df[
        "transaction_count"
    ].sum()
    if not physical_zip_missing_city_df.empty
    else 0
)


residual_location_summary = pd.Series(
    {
        "Non-online State missing":
            physical_state_missing_count,

        "Non-online State missing rate on full dataset (%)":
            physical_state_missing_count
            / total_rows
            * 100,

        "Non-online Zip missing":
            physical_zip_missing_count,

        "Non-online Zip missing rate on full dataset (%)":
            physical_zip_missing_count
            / total_rows
            * 100,
    }
)


display(
    residual_location_summary
)
```


    Non-online State missing                               7601.000000
    Non-online State missing rate on full dataset (%)         0.031168
    Non-online Zip missing                               164915.000000
    Non-online Zip missing rate on full dataset (%)           0.676244
    dtype: float64



```python
state_city_preview_n = min(
    CATEGORY_PREVIEW_N,
    len(
        physical_state_missing_city_df
    ),
)

zip_city_preview_n = min(
    CATEGORY_PREVIEW_N,
    len(
        physical_zip_missing_city_df
    ),
)

state_preview_n = min(
    CATEGORY_PREVIEW_N,
    len(
        zip_missing_state_present_df
    ),
)


print(
    f"{state_city_preview_n} Merchant City "
    "phổ biến nhất trong non-online transaction "
    "bị State missing:"
)

display(
    physical_state_missing_city_df.head(
        state_city_preview_n
    )
)


print(
    f"\n{zip_city_preview_n} Merchant City "
    "phổ biến nhất trong non-online transaction "
    "bị Zip missing:"
)

display(
    physical_zip_missing_city_df.head(
        zip_city_preview_n
    )
)


print(
    f"\n{state_preview_n} Merchant State "
    "phổ biến nhất khi Zip missing "
    "nhưng State vẫn tồn tại:"
)

display(
    zip_missing_state_present_df.head(
        state_preview_n
    )
)
```

    1 Merchant City phổ biến nhất trong non-online transaction bị State missing:



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
      <td>ONLINE</td>
      <td>7601</td>
    </tr>
  </tbody>
</table>
</div>


    
    15 Merchant City phổ biến nhất trong non-online transaction bị Zip missing:



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
      <td>Cancun</td>
      <td>16977</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Mexico City</td>
      <td>8878</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Rome</td>
      <td>8730</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Toronto</td>
      <td>7783</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Cabo San Lucas</td>
      <td>7659</td>
    </tr>
    <tr>
      <th>5</th>
      <td>ONLINE</td>
      <td>7601</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Puerto Vallarta</td>
      <td>7105</td>
    </tr>
    <tr>
      <th>7</th>
      <td>London</td>
      <td>6092</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Paris</td>
      <td>5311</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Berlin</td>
      <td>5263</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Guadalajara</td>
      <td>5031</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Montreal</td>
      <td>4714</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Vancouver</td>
      <td>4222</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Tokyo</td>
      <td>3955</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Beijing</td>
      <td>3141</td>
    </tr>
  </tbody>
</table>
</div>


    
    15 Merchant State phổ biến nhất khi Zip missing nhưng State vẫn tồn tại:



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
      <td>47152</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Canada</td>
      <td>20148</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Italy</td>
      <td>8730</td>
    </tr>
    <tr>
      <th>3</th>
      <td>United Kingdom</td>
      <td>8055</td>
    </tr>
    <tr>
      <th>4</th>
      <td>France</td>
      <td>5311</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Germany</td>
      <td>5263</td>
    </tr>
    <tr>
      <th>6</th>
      <td>China</td>
      <td>4996</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Japan</td>
      <td>3955</td>
    </tr>
    <tr>
      <th>8</th>
      <td>India</td>
      <td>3482</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Spain</td>
      <td>3203</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Dominican Republic</td>
      <td>2754</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Netherlands</td>
      <td>2616</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Jamaica</td>
      <td>2184</td>
    </tr>
    <tr>
      <th>13</th>
      <td>South Korea</td>
      <td>2178</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Switzerland</td>
      <td>1947</td>
    </tr>
  </tbody>
</table>
</div>


### Nhận xét M2.5.4

Sau khi loại `Online Transaction`, vẫn còn: <br>
`7,601 transaction bị Merchant State missing` <br>
`164,915 transaction bị Zip missing` <br>

Tương ứng trên toàn dataset: <br>
`Non-online State missing ≈ 0.0312%` <br>
`Non-online Zip missing ≈ 0.6762%` <br>

Toàn bộ `7,601` trường hợp State missing ngoài Online Transaction có: <br>
`Merchant City = ONLINE`. <br>

Kết hợp với kết quả M2.5.3, đây chính là nhóm `Chip Transaction + Merchant City = ONLINE + State missing + Zip missing`. <br>

Do đó residual State missing không tạo ra một nhóm physical-city mới; nó vẫn thuộc representation `ONLINE`, chỉ khác ở transaction mode. <br>

Residual Zip missing có cấu trúc khác. <br>

Các Merchant City xuất hiện nhiều nhất trong non-online transaction bị Zip missing gồm: <br>
`Cancun` <br>
`Mexico City` <br>
`Rome` <br>
`Toronto` <br>
`Cabo San Lucas` <br>
`Puerto Vallarta` <br>
`London` <br>
`Paris` <br>
`Berlin` <br>
`Montreal` <br>
`Vancouver` <br>
`Tokyo` <br>
`Beijing` <br>

Khi Zip missing nhưng Merchant State vẫn tồn tại, các giá trị Merchant State phổ biến gồm: <br>
`Mexico` <br>
`Canada` <br>
`Italy` <br>
`United Kingdom` <br>
`France` <br>
`Germany` <br>
`China` <br>
`Japan` <br>
`India` <br>
`Spain` <br>
và nhiều quốc gia khác. <br>

Output này cho thấy một phần đáng kể residual Zip missing liên quan đến merchant location ngoài Hoa Kỳ. <br>

Đây là bằng chứng quan trọng về semantic của field `Zip`: việc một merchant quốc tế không có mã ZIP theo representation của dataset có thể là `not applicable` hơn là dữ liệu bị mất ngẫu nhiên. <br>

Đây vẫn là một diễn giải dựa trên pattern quan sát được, chưa phải bằng chứng tài liệu chính thức về cơ chế tạo dữ liệu. <br>

Output cũng cho thấy `Merchant State` không luôn chứa mã state của Hoa Kỳ; đối với nhiều merchant quốc tế, field này chứa tên quốc gia như `Mexico`, `Canada`, `Italy` hoặc `United Kingdom`. <br>

Do đó tên cột `Merchant State` không nên được hiểu quá hẹp như một US-state field khi thiết kế preprocessing. <br>

Có thể phân biệt ít nhất hai cơ chế Zip missing: <br>
`Online transaction → không có physical location → Zip missing` <br>
và <br>
`physical / non-online international merchant → State/country tồn tại nhưng Zip không tồn tại trong representation`. <br>

`Kết luận: phần lớn location missing là structural. State missing ngoài Online mode vẫn thuộc representation ONLINE, trong khi residual Zip missing chủ yếu xuất hiện ở các merchant quốc tế, cho thấy Zip missing có nhiều cơ chế semantic khác nhau và không nên được xử lý bằng một quy tắc imputation duy nhất.`

## M2.5.5 — Điều tra Amount âm, Amount bằng 0 và các giá trị cực trị

### Câu hỏi

Amount âm và Amount bằng 0 phân bố như thế nào giữa các transaction mode? <br>

Amount âm có chỉ xuất hiện ở một loại transaction hay xuất hiện trên nhiều mode? <br>

Những Amount thấp nhất và cao nhất trong dataset trông như thế nào? <br>

### Phương pháp

Tạo ba nhóm: <br>
`negative` <br>
`zero` <br>
`positive` <br>

Sau đó tính count và tỷ lệ của từng nhóm trong từng `Use Chip`. <br>

Đồng thời hiển thị một số transaction có Amount thấp nhất và cao nhất để audit trực tiếp. <br>

Các transaction cực trị chỉ được xem là `candidate cần kiểm tra`, không tự gọi là lỗi hoặc outlier cần xóa. <br>


```python
amount_sign_by_mode_df = pd.DataFrame(
    [
        {
            "Use Chip":
                key[0],

            "amount_sign":
                key[1],

            "transaction_count":
                count,
        }
        for key, count
        in amount_sign_by_mode_counts.items()
    ]
)


amount_sign_by_mode_df[
    "mode_total"
] = (
    amount_sign_by_mode_df[
        "Use Chip"
    ]
    .map(
        use_chip_counts
    )
)


amount_sign_by_mode_df[
    "rate_within_mode_pct"
] = (
    amount_sign_by_mode_df[
        "transaction_count"
    ]
    / amount_sign_by_mode_df[
        "mode_total"
    ]
    * 100
)


amount_sign_by_mode_df = (
    amount_sign_by_mode_df
    .sort_values(
        [
            "Use Chip",
            "amount_sign",
        ]
    )
    .reset_index(drop=True)
)


display(
    amount_sign_by_mode_df
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
      <th>Use Chip</th>
      <th>amount_sign</th>
      <th>transaction_count</th>
      <th>mode_total</th>
      <th>rate_within_mode_pct</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Chip Transaction</td>
      <td>negative</td>
      <td>342120</td>
      <td>6287598</td>
      <td>5.441188</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Chip Transaction</td>
      <td>positive</td>
      <td>5939649</td>
      <td>6287598</td>
      <td>94.466106</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Chip Transaction</td>
      <td>zero</td>
      <td>5829</td>
      <td>6287598</td>
      <td>0.092706</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Online Transaction</td>
      <td>negative</td>
      <td>13499</td>
      <td>2713220</td>
      <td>0.497527</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Online Transaction</td>
      <td>positive</td>
      <td>2699712</td>
      <td>2713220</td>
      <td>99.502141</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Online Transaction</td>
      <td>zero</td>
      <td>9</td>
      <td>2713220</td>
      <td>0.000332</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Swipe Transaction</td>
      <td>negative</td>
      <td>889064</td>
      <td>15386082</td>
      <td>5.778365</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Swipe Transaction</td>
      <td>positive</td>
      <td>14482643</td>
      <td>15386082</td>
      <td>94.128206</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Swipe Transaction</td>
      <td>zero</td>
      <td>14375</td>
      <td>15386082</td>
      <td>0.093429</td>
    </tr>
  </tbody>
</table>
</div>



```python
amount_sign_summary = pd.DataFrame(
    [
        {
            "amount_sign":
                "negative",

            "transaction_count":
                amount_negative_count,
        },
        {
            "amount_sign":
                "zero",

            "transaction_count":
                amount_zero_count,
        },
        {
            "amount_sign":
                "positive",

            "transaction_count":
                amount_positive_count,
        },
    ]
)


amount_sign_summary[
    "transaction_share_pct"
] = (
    amount_sign_summary[
        "transaction_count"
    ]
    / total_rows
    * 100
)


display(
    amount_sign_summary
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
      <th>transaction_share_pct</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>negative</td>
      <td>1244683</td>
      <td>5.103900</td>
    </tr>
    <tr>
      <th>1</th>
      <td>zero</td>
      <td>20213</td>
      <td>0.082885</td>
    </tr>
    <tr>
      <th>2</th>
      <td>positive</td>
      <td>23122004</td>
      <td>94.813215</td>
    </tr>
  </tbody>
</table>
</div>



```python
low_preview_n = min(
    EXTREME_PREVIEW_N,
    len(
        extreme_low_amount_df
    ),
)

high_preview_n = min(
    EXTREME_PREVIEW_N,
    len(
        extreme_high_amount_df
    ),
)


print(
    f"{low_preview_n} transaction "
    "có Amount thấp nhất:"
)

display(
    extreme_low_amount_df.head(
        low_preview_n
    )
)


print(
    f"\n{high_preview_n} transaction "
    "có Amount cao nhất:"
)

display(
    extreme_high_amount_df.head(
        high_preview_n
    )
)
```

    10 transaction có Amount thấp nhất:



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
      <th>Year</th>
      <th>Month</th>
      <th>Day</th>
      <th>Time</th>
      <th>Amount</th>
      <th>Use Chip</th>
      <th>Merchant Name</th>
      <th>Merchant City</th>
      <th>Merchant State</th>
      <th>Zip</th>
      <th>MCC</th>
      <th>Amount_numeric</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2020</td>
      <td>1</td>
      <td>6</td>
      <td>17:46</td>
      <td>$-500.00</td>
      <td>Swipe Transaction</td>
      <td>-1007477596717646975</td>
      <td>Anchorage</td>
      <td>AK</td>
      <td>99504.0</td>
      <td>3780</td>
      <td>-500.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2009</td>
      <td>11</td>
      <td>27</td>
      <td>10:44</td>
      <td>$-500.00</td>
      <td>Swipe Transaction</td>
      <td>-6161792371494728879</td>
      <td>Pirtleville</td>
      <td>AZ</td>
      <td>85626.0</td>
      <td>3389</td>
      <td>-500.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2015</td>
      <td>12</td>
      <td>9</td>
      <td>10:45</td>
      <td>$-500.00</td>
      <td>Swipe Transaction</td>
      <td>483490033258680568</td>
      <td>Walnut Creek</td>
      <td>CA</td>
      <td>94597.0</td>
      <td>3504</td>
      <td>-500.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2010</td>
      <td>10</td>
      <td>16</td>
      <td>12:25</td>
      <td>$-500.00</td>
      <td>Swipe Transaction</td>
      <td>483490033258680568</td>
      <td>Randolph</td>
      <td>MA</td>
      <td>2368.0</td>
      <td>3504</td>
      <td>-500.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2018</td>
      <td>6</td>
      <td>26</td>
      <td>04:11</td>
      <td>$-500.00</td>
      <td>Chip Transaction</td>
      <td>-112121233619748226</td>
      <td>Marion</td>
      <td>NC</td>
      <td>28752.0</td>
      <td>3509</td>
      <td>-500.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2017</td>
      <td>9</td>
      <td>23</td>
      <td>21:31</td>
      <td>$-500.00</td>
      <td>Chip Transaction</td>
      <td>4552887027432897467</td>
      <td>Oakland</td>
      <td>CA</td>
      <td>94606.0</td>
      <td>3596</td>
      <td>-500.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2001</td>
      <td>11</td>
      <td>15</td>
      <td>04:54</td>
      <td>$-500.00</td>
      <td>Swipe Transaction</td>
      <td>4552887027432897467</td>
      <td>Oakland</td>
      <td>CA</td>
      <td>94606.0</td>
      <td>3596</td>
      <td>-500.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2017</td>
      <td>2</td>
      <td>3</td>
      <td>10:12</td>
      <td>$-500.00</td>
      <td>Online Transaction</td>
      <td>-6796080605569913348</td>
      <td>ONLINE</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4722</td>
      <td>-500.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2007</td>
      <td>2</td>
      <td>13</td>
      <td>23:08</td>
      <td>$-500.00</td>
      <td>Swipe Transaction</td>
      <td>3991321433720498482</td>
      <td>Calexico</td>
      <td>CA</td>
      <td>92231.0</td>
      <td>3405</td>
      <td>-500.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2006</td>
      <td>1</td>
      <td>6</td>
      <td>02:54</td>
      <td>$-500.00</td>
      <td>Swipe Transaction</td>
      <td>4522145123748518450</td>
      <td>Rogersville</td>
      <td>TN</td>
      <td>37857.0</td>
      <td>7011</td>
      <td>-500.0</td>
    </tr>
  </tbody>
</table>
</div>


    
    10 transaction có Amount cao nhất:



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
      <th>Year</th>
      <th>Month</th>
      <th>Day</th>
      <th>Time</th>
      <th>Amount</th>
      <th>Use Chip</th>
      <th>Merchant Name</th>
      <th>Merchant City</th>
      <th>Merchant State</th>
      <th>Zip</th>
      <th>MCC</th>
      <th>Amount_numeric</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2004</td>
      <td>1</td>
      <td>17</td>
      <td>06:57</td>
      <td>$12390.50</td>
      <td>Online Transaction</td>
      <td>-7421991366943579652</td>
      <td>ONLINE</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4411</td>
      <td>12390.5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2010</td>
      <td>9</td>
      <td>22</td>
      <td>06:37</td>
      <td>$6820.20</td>
      <td>Swipe Transaction</td>
      <td>-2910169010079677451</td>
      <td>Staten Island</td>
      <td>NY</td>
      <td>10302.0</td>
      <td>5712</td>
      <td>6820.2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2019</td>
      <td>1</td>
      <td>27</td>
      <td>17:52</td>
      <td>$6613.44</td>
      <td>Online Transaction</td>
      <td>-7576589894704936102</td>
      <td>ONLINE</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4411</td>
      <td>6613.44</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2001</td>
      <td>1</td>
      <td>23</td>
      <td>11:03</td>
      <td>$6261.69</td>
      <td>Swipe Transaction</td>
      <td>4872340518840476610</td>
      <td>Stamford</td>
      <td>CT</td>
      <td>6907.0</td>
      <td>5732</td>
      <td>6261.69</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2012</td>
      <td>4</td>
      <td>10</td>
      <td>11:05</td>
      <td>$5913.37</td>
      <td>Swipe Transaction</td>
      <td>6584510127902958451</td>
      <td>Wilton</td>
      <td>CT</td>
      <td>6897.0</td>
      <td>5932</td>
      <td>5913.37</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2000</td>
      <td>1</td>
      <td>3</td>
      <td>11:46</td>
      <td>$5878.31</td>
      <td>Swipe Transaction</td>
      <td>-168344065522689764</td>
      <td>Houston</td>
      <td>TX</td>
      <td>77027.0</td>
      <td>5932</td>
      <td>5878.31</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2013</td>
      <td>5</td>
      <td>22</td>
      <td>17:28</td>
      <td>$5813.78</td>
      <td>Online Transaction</td>
      <td>-7576589894704936102</td>
      <td>ONLINE</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4411</td>
      <td>5813.78</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2003</td>
      <td>10</td>
      <td>15</td>
      <td>11:17</td>
      <td>$5717.28</td>
      <td>Swipe Transaction</td>
      <td>4872340518840476610</td>
      <td>Stamford</td>
      <td>CT</td>
      <td>6907.0</td>
      <td>5732</td>
      <td>5717.28</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2019</td>
      <td>11</td>
      <td>22</td>
      <td>14:49</td>
      <td>$5712.06</td>
      <td>Chip Transaction</td>
      <td>-7162568984096711351</td>
      <td>Memphis</td>
      <td>TN</td>
      <td>38107.0</td>
      <td>5733</td>
      <td>5712.06</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2014</td>
      <td>10</td>
      <td>24</td>
      <td>13:11</td>
      <td>$5696.78</td>
      <td>Online Transaction</td>
      <td>-7902814680636443920</td>
      <td>ONLINE</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4411</td>
      <td>5696.78</td>
    </tr>
  </tbody>
</table>
</div>


### Nhận xét M2.5.5

Phân bố dấu của Amount trên toàn dataset tiếp tục xác nhận: <br>
`Negative: 1,244,683 — 5.1039%` <br>
`Zero: 20,213 — 0.0829%` <br>
`Positive: 23,122,004 — 94.8132%` <br>

Amount âm xuất hiện ở cả ba transaction mode, nhưng tỷ lệ khác nhau rõ rệt. <br>

`Swipe Transaction`: <br>
`negative ≈ 5.7784%` <br>
`zero ≈ 0.0934%` <br>
`positive ≈ 94.1282%` <br>

`Chip Transaction`: <br>
`negative ≈ 5.4412%` <br>
`zero ≈ 0.0927%` <br>
`positive ≈ 94.4661%` <br>

`Online Transaction`: <br>
`negative ≈ 0.4975%` <br>
`zero ≈ 0.00033%` <br>
`positive ≈ 99.5021%` <br>

Như vậy Amount âm không phải hiện tượng chỉ xảy ra ở một channel. <br>

Tuy nhiên tỷ lệ negative Amount ở Swipe và Chip lớn hơn khoảng một bậc độ lớn so với Online Transaction. <br>

Zero Amount cũng xuất hiện với tỷ lệ gần nhau ở Swipe và Chip nhưng gần như không xuất hiện ở Online Transaction, chỉ có `9 transaction`. <br>

Điều này cho thấy sign của Amount có cấu trúc theo transaction mode và không giống một dạng corruption ngẫu nhiên phân bố đều trên dataset. <br>

Kết quả này củng cố guardrail rằng không nên tự động xóa, lấy trị tuyệt đối hoặc clip Amount âm trước khi hiểu semantic của nó. <br>

Đối với cực trị âm, cả `10 transaction thấp nhất` đều có đúng giá trị `-500.00`. <br>

Các transaction -500 xuất hiện ở nhiều năm, nhiều merchant, nhiều MCC và cả `Swipe`, `Chip`, `Online`. <br>

Việc minimum lặp lại chính xác ở `-500` trên nhiều context khác nhau gợi ý rằng `-500` có thể là một boundary hoặc convention của dữ liệu / synthetic generator thay vì một extreme value hoàn toàn ngẫu nhiên. <br>

Tuy nhiên output hiện tại chưa đủ để xác định chính xác nguyên nhân hoặc semantic của boundary này. <br>

Đối với cực trị dương, transaction lớn nhất là: <br>
`12,390.50` <br>

Các Amount cao nhất còn lại nằm trong khoảng khoảng `5,700–6,800`, và xuất hiện ở cả Online, Swipe và Chip Transaction. <br>

Extreme positive Amount vì vậy không chỉ thuộc một transaction mode duy nhất. <br>

Các giá trị lớn cũng xuất hiện ở nhiều merchant và MCC khác nhau, mặc dù một số merchant lặp lại trong danh sách top extremes. <br>

Không có bằng chứng ở M2.5 để xem các Amount cực trị này là lỗi dữ liệu. <br>

`Kết luận: Amount âm và zero có pattern rõ theo transaction mode, vì vậy không phù hợp với giả thuyết corruption ngẫu nhiên đơn giản. Minimum -500 có dấu hiệu giống một boundary cần lưu ý, còn extreme positive values xuất hiện trên nhiều context. Semantic chính xác của Amount âm vẫn là limitation mở và quyết định preprocessing phải được trì hoãn sang M4.`

## M2.5.6 — Audit exact duplicate

### Vì sao kiểm tra này tồn tại?

Exact duplicate có thể là lỗi sao chép dữ liệu, nhưng cũng có thể là hai transaction khác nhau vô tình có cùng toàn bộ các field công khai. <br>

Dataset không có `transaction_id` duy nhất nên không được tự động đồng nhất hai khái niệm này. <br>

### Câu hỏi

Có bao nhiêu nhóm row giống hệt trên toàn bộ 15 cột? <br>

Mỗi nhóm có bao nhiêu thành viên? <br>

Duplicate xuất hiện ở những năm nào? <br>

Các duplicate-member rows có chứa fraud hay không? <br>

### Phương pháp

Bước 1: dùng row hash để tìm những hash xuất hiện nhiều hơn một lần. <br>

Bước 2: đọc lại dataset và chỉ giữ các row có duplicate hash. <br>

Bước 3: trên tập candidate nhỏ, kiểm tra exact equality trên đủ 15 cột để loại khả năng hash collision. <br>

Hash chỉ dùng để giảm không gian tìm kiếm; kết luận duplicate cuối cùng vẫn dựa trên exact row comparison. <br>


```python
all_row_hashes = np.concatenate(
    row_hash_parts
)


del row_hash_parts

gc.collect()


print(
    "Số row hash:",
    f"{len(all_row_hashes):,}",
)


# Sort in-place để giảm memory phụ
all_row_hashes.sort()


same_as_previous = (
    all_row_hashes[1:]
    == all_row_hashes[:-1]
)


duplicate_hashes = np.unique(
    all_row_hashes[1:][
        same_as_previous
    ]
)


print(
    "Số duplicate hash candidate:",
    len(
        duplicate_hashes
    ),
)


del same_as_previous
del all_row_hashes

gc.collect()
```

    Số row hash: 24,386,900
    Số duplicate hash candidate: 66





    0




```python
duplicate_candidate_parts = []


if len(duplicate_hashes) > 0:

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            DATA_PATH,
            usecols=EXPECTED_COLUMNS,
            chunksize=CHUNK_SIZE,
        ),
        start=1,
    ):
        row_hash = (
            pd.util
            .hash_pandas_object(
                chunk[
                    EXPECTED_COLUMNS
                ],
                index=False,
            )
            .to_numpy(
                dtype=np.uint64,
                copy=False,
            )
        )


        duplicate_candidate_mask = (
            np.isin(
                row_hash,
                duplicate_hashes,
            )
        )


        if (
            duplicate_candidate_mask
            .any()
        ):
            candidate_rows = (
                chunk.loc[
                    duplicate_candidate_mask
                ]
                .copy()
            )

            candidate_rows[
                "__row_hash"
            ] = (
                row_hash[
                    duplicate_candidate_mask
                ]
            )

            duplicate_candidate_parts.append(
                candidate_rows
            )


        if (
            chunk_number % 20 == 0
            or len(chunk) < CHUNK_SIZE
        ):
            print(
                f"Duplicate pass: "
                f"đã xử lý {chunk_number} chunk"
            )


if duplicate_candidate_parts:

    duplicate_candidates_df = (
        pd.concat(
            duplicate_candidate_parts,
            ignore_index=True,
        )
    )

else:

    duplicate_candidates_df = pd.DataFrame(
        columns=(
            EXPECTED_COLUMNS
            + ["__row_hash"]
        )
    )


print(
    "\nSố candidate rows sau hash filter:",
    len(
        duplicate_candidates_df
    ),
)
```

    Duplicate pass: đã xử lý 20 chunk
    Duplicate pass: đã xử lý 40 chunk
    Duplicate pass: đã xử lý 60 chunk
    Duplicate pass: đã xử lý 80 chunk
    Duplicate pass: đã xử lý 98 chunk
    
    Số candidate rows sau hash filter: 132



```python
if not duplicate_candidates_df.empty:

    exact_duplicate_mask = (
        duplicate_candidates_df
        .duplicated(
            subset=EXPECTED_COLUMNS,
            keep=False,
        )
    )


    exact_duplicate_members_df = (
        duplicate_candidates_df
        .loc[
            exact_duplicate_mask,
            EXPECTED_COLUMNS,
        ]
        .copy()
        .reset_index(drop=True)
    )

else:

    exact_duplicate_members_df = pd.DataFrame(
        columns=EXPECTED_COLUMNS
    )


if not exact_duplicate_members_df.empty:

    duplicate_group_object = (
        exact_duplicate_members_df
        .groupby(
            EXPECTED_COLUMNS,
            dropna=False,
            sort=False,
        )
    )


    exact_duplicate_members_df[
        "duplicate_group_id"
    ] = (
        duplicate_group_object
        .ngroup()
        + 1
    )


    group_size_map = (
        exact_duplicate_members_df
        .groupby(
            "duplicate_group_id"
        )
        .size()
    )


    exact_duplicate_members_df[
        "group_size"
    ] = (
        exact_duplicate_members_df[
            "duplicate_group_id"
        ]
        .map(
            group_size_map
        )
    )


    duplicate_group_count = int(
        group_size_map.shape[0]
    )

    duplicate_member_rows = int(
        len(
            exact_duplicate_members_df
        )
    )

    duplicate_extra_rows = int(
        (
            group_size_map
            - 1
        ).sum()
    )

    largest_duplicate_group = int(
        group_size_map.max()
    )

else:

    duplicate_group_count = 0
    duplicate_member_rows = 0
    duplicate_extra_rows = 0
    largest_duplicate_group = 0
```


```python
duplicate_fraud_count = int(
    (
        exact_duplicate_members_df[
            "Is Fraud?"
        ]
        .astype("string")
        .str.strip()
        .eq("Yes")
    ).sum()
)


duplicate_summary = pd.Series(
    {
        "duplicate_group_count":
            duplicate_group_count,

        "duplicate_member_rows":
            duplicate_member_rows,

        "duplicate_extra_rows":
            duplicate_extra_rows,

        "duplicate_extra_rate_pct":
            (
                duplicate_extra_rows
                / total_rows
                * 100
            ),

        "largest_duplicate_group":
            largest_duplicate_group,

        "fraud_among_duplicate_members":
            duplicate_fraud_count,
    }
)


display(
    duplicate_summary
)
```


    duplicate_group_count             66.000000
    duplicate_member_rows            132.000000
    duplicate_extra_rows              66.000000
    duplicate_extra_rate_pct           0.000271
    largest_duplicate_group            2.000000
    fraud_among_duplicate_members      0.000000
    dtype: float64



```python
if not exact_duplicate_members_df.empty:

    duplicate_year_df = (
        exact_duplicate_members_df[
            "Year"
        ]
        .value_counts()
        .sort_index()
        .rename_axis(
            "Year"
        )
        .reset_index(
            name="duplicate_member_count"
        )
    )


    duplicate_target_df = (
        exact_duplicate_members_df[
            "Is Fraud?"
        ]
        .value_counts(
            dropna=False
        )
        .rename_axis(
            "Is Fraud?"
        )
        .reset_index(
            name="duplicate_member_count"
        )
    )


    print(
        "Duplicate-member rows theo năm:"
    )

    display(
        duplicate_year_df
    )


    print(
        "\nDuplicate-member rows theo target:"
    )

    display(
        duplicate_target_df
    )
```

    Duplicate-member rows theo năm:



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
      <th>Year</th>
      <th>duplicate_member_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1999</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2001</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2002</td>
      <td>6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2003</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2004</td>
      <td>6</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2005</td>
      <td>8</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2006</td>
      <td>10</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2007</td>
      <td>4</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2008</td>
      <td>6</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2009</td>
      <td>8</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2010</td>
      <td>2</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2011</td>
      <td>10</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2012</td>
      <td>14</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2013</td>
      <td>8</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2014</td>
      <td>4</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2015</td>
      <td>6</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2016</td>
      <td>6</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2017</td>
      <td>14</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2018</td>
      <td>2</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2019</td>
      <td>10</td>
    </tr>
  </tbody>
</table>
</div>


    
    Duplicate-member rows theo target:



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
      <th>Is Fraud?</th>
      <th>duplicate_member_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>No</td>
      <td>132</td>
    </tr>
  </tbody>
</table>
</div>



```python
duplicate_preview_n = min(
    20,
    len(
        exact_duplicate_members_df
    ),
)


print(
    f"{duplicate_preview_n} duplicate-member rows đầu tiên:"
)


display(
    exact_duplicate_members_df[
        [
            "duplicate_group_id",
            "group_size",
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
            "MCC",
            "Errors?",
            "Is Fraud?",
        ]
    ]
    .sort_values(
        [
            "duplicate_group_id",
        ]
    )
    .head(
        duplicate_preview_n
    )
)
```

    20 duplicate-member rows đầu tiên:



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
      <th>duplicate_group_id</th>
      <th>group_size</th>
      <th>User</th>
      <th>Card</th>
      <th>Year</th>
      <th>Month</th>
      <th>Day</th>
      <th>Time</th>
      <th>Amount</th>
      <th>Use Chip</th>
      <th>Merchant Name</th>
      <th>Merchant City</th>
      <th>Merchant State</th>
      <th>Zip</th>
      <th>MCC</th>
      <th>Errors?</th>
      <th>Is Fraud?</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>13</td>
      <td>0</td>
      <td>2015</td>
      <td>4</td>
      <td>28</td>
      <td>06:36</td>
      <td>$100.00</td>
      <td>Chip Transaction</td>
      <td>-4282466774399734331</td>
      <td>Boyne City</td>
      <td>MI</td>
      <td>49712.0</td>
      <td>4829</td>
      <td>Insufficient Balance,</td>
      <td>No</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2</td>
      <td>13</td>
      <td>0</td>
      <td>2015</td>
      <td>4</td>
      <td>28</td>
      <td>06:36</td>
      <td>$100.00</td>
      <td>Chip Transaction</td>
      <td>-4282466774399734331</td>
      <td>Boyne City</td>
      <td>MI</td>
      <td>49712.0</td>
      <td>4829</td>
      <td>Insufficient Balance,</td>
      <td>No</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>2</td>
      <td>13</td>
      <td>1</td>
      <td>2006</td>
      <td>7</td>
      <td>19</td>
      <td>07:34</td>
      <td>$100.00</td>
      <td>Swipe Transaction</td>
      <td>-4282466774399734331</td>
      <td>Boyne City</td>
      <td>MI</td>
      <td>49712.0</td>
      <td>4829</td>
      <td>Insufficient Balance,</td>
      <td>No</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>2</td>
      <td>13</td>
      <td>1</td>
      <td>2006</td>
      <td>7</td>
      <td>19</td>
      <td>07:34</td>
      <td>$100.00</td>
      <td>Swipe Transaction</td>
      <td>-4282466774399734331</td>
      <td>Boyne City</td>
      <td>MI</td>
      <td>49712.0</td>
      <td>4829</td>
      <td>Insufficient Balance,</td>
      <td>No</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>2</td>
      <td>13</td>
      <td>1</td>
      <td>2006</td>
      <td>10</td>
      <td>15</td>
      <td>07:11</td>
      <td>$100.00</td>
      <td>Swipe Transaction</td>
      <td>-4282466774399734331</td>
      <td>Boyne City</td>
      <td>MI</td>
      <td>49712.0</td>
      <td>4829</td>
      <td>Insufficient Balance,</td>
      <td>No</td>
    </tr>
    <tr>
      <th>5</th>
      <td>3</td>
      <td>2</td>
      <td>13</td>
      <td>1</td>
      <td>2006</td>
      <td>10</td>
      <td>15</td>
      <td>07:11</td>
      <td>$100.00</td>
      <td>Swipe Transaction</td>
      <td>-4282466774399734331</td>
      <td>Boyne City</td>
      <td>MI</td>
      <td>49712.0</td>
      <td>4829</td>
      <td>Insufficient Balance,</td>
      <td>No</td>
    </tr>
    <tr>
      <th>6</th>
      <td>4</td>
      <td>2</td>
      <td>17</td>
      <td>2</td>
      <td>2011</td>
      <td>12</td>
      <td>12</td>
      <td>16:26</td>
      <td>$120.00</td>
      <td>Swipe Transaction</td>
      <td>-4282466774399734331</td>
      <td>Prescott Valley</td>
      <td>AZ</td>
      <td>86314.0</td>
      <td>4829</td>
      <td>Insufficient Balance,</td>
      <td>No</td>
    </tr>
    <tr>
      <th>7</th>
      <td>4</td>
      <td>2</td>
      <td>17</td>
      <td>2</td>
      <td>2011</td>
      <td>12</td>
      <td>12</td>
      <td>16:26</td>
      <td>$120.00</td>
      <td>Swipe Transaction</td>
      <td>-4282466774399734331</td>
      <td>Prescott Valley</td>
      <td>AZ</td>
      <td>86314.0</td>
      <td>4829</td>
      <td>Insufficient Balance,</td>
      <td>No</td>
    </tr>
    <tr>
      <th>8</th>
      <td>5</td>
      <td>2</td>
      <td>109</td>
      <td>2</td>
      <td>1999</td>
      <td>12</td>
      <td>9</td>
      <td>04:16</td>
      <td>$120.00</td>
      <td>Swipe Transaction</td>
      <td>-4282466774399734331</td>
      <td>Hammond</td>
      <td>IN</td>
      <td>46323.0</td>
      <td>4829</td>
      <td>Insufficient Balance,</td>
      <td>No</td>
    </tr>
    <tr>
      <th>9</th>
      <td>5</td>
      <td>2</td>
      <td>109</td>
      <td>2</td>
      <td>1999</td>
      <td>12</td>
      <td>9</td>
      <td>04:16</td>
      <td>$120.00</td>
      <td>Swipe Transaction</td>
      <td>-4282466774399734331</td>
      <td>Hammond</td>
      <td>IN</td>
      <td>46323.0</td>
      <td>4829</td>
      <td>Insufficient Balance,</td>
      <td>No</td>
    </tr>
    <tr>
      <th>10</th>
      <td>6</td>
      <td>2</td>
      <td>109</td>
      <td>2</td>
      <td>2019</td>
      <td>3</td>
      <td>8</td>
      <td>04:22</td>
      <td>$120.00</td>
      <td>Chip Transaction</td>
      <td>-4282466774399734331</td>
      <td>Hammond</td>
      <td>IN</td>
      <td>46323.0</td>
      <td>4829</td>
      <td>Insufficient Balance,</td>
      <td>No</td>
    </tr>
    <tr>
      <th>11</th>
      <td>6</td>
      <td>2</td>
      <td>109</td>
      <td>2</td>
      <td>2019</td>
      <td>3</td>
      <td>8</td>
      <td>04:22</td>
      <td>$120.00</td>
      <td>Chip Transaction</td>
      <td>-4282466774399734331</td>
      <td>Hammond</td>
      <td>IN</td>
      <td>46323.0</td>
      <td>4829</td>
      <td>Insufficient Balance,</td>
      <td>No</td>
    </tr>
    <tr>
      <th>12</th>
      <td>7</td>
      <td>2</td>
      <td>205</td>
      <td>2</td>
      <td>2012</td>
      <td>2</td>
      <td>8</td>
      <td>17:33</td>
      <td>$82.00</td>
      <td>Swipe Transaction</td>
      <td>2027553650310142703</td>
      <td>Huntington Station</td>
      <td>NY</td>
      <td>11746.0</td>
      <td>5541</td>
      <td>NaN</td>
      <td>No</td>
    </tr>
    <tr>
      <th>13</th>
      <td>7</td>
      <td>2</td>
      <td>205</td>
      <td>2</td>
      <td>2012</td>
      <td>2</td>
      <td>8</td>
      <td>17:33</td>
      <td>$82.00</td>
      <td>Swipe Transaction</td>
      <td>2027553650310142703</td>
      <td>Huntington Station</td>
      <td>NY</td>
      <td>11746.0</td>
      <td>5541</td>
      <td>NaN</td>
      <td>No</td>
    </tr>
    <tr>
      <th>15</th>
      <td>8</td>
      <td>2</td>
      <td>235</td>
      <td>1</td>
      <td>2008</td>
      <td>5</td>
      <td>22</td>
      <td>10:23</td>
      <td>$83.00</td>
      <td>Swipe Transaction</td>
      <td>1799189980464955940</td>
      <td>Garland</td>
      <td>TX</td>
      <td>75042.0</td>
      <td>5499</td>
      <td>NaN</td>
      <td>No</td>
    </tr>
    <tr>
      <th>14</th>
      <td>8</td>
      <td>2</td>
      <td>235</td>
      <td>1</td>
      <td>2008</td>
      <td>5</td>
      <td>22</td>
      <td>10:23</td>
      <td>$83.00</td>
      <td>Swipe Transaction</td>
      <td>1799189980464955940</td>
      <td>Garland</td>
      <td>TX</td>
      <td>75042.0</td>
      <td>5499</td>
      <td>NaN</td>
      <td>No</td>
    </tr>
    <tr>
      <th>16</th>
      <td>9</td>
      <td>2</td>
      <td>300</td>
      <td>1</td>
      <td>2017</td>
      <td>8</td>
      <td>25</td>
      <td>07:15</td>
      <td>$63.00</td>
      <td>Chip Transaction</td>
      <td>1799189980464955940</td>
      <td>Kingman</td>
      <td>AZ</td>
      <td>86401.0</td>
      <td>5499</td>
      <td>NaN</td>
      <td>No</td>
    </tr>
    <tr>
      <th>17</th>
      <td>9</td>
      <td>2</td>
      <td>300</td>
      <td>1</td>
      <td>2017</td>
      <td>8</td>
      <td>25</td>
      <td>07:15</td>
      <td>$63.00</td>
      <td>Chip Transaction</td>
      <td>1799189980464955940</td>
      <td>Kingman</td>
      <td>AZ</td>
      <td>86401.0</td>
      <td>5499</td>
      <td>NaN</td>
      <td>No</td>
    </tr>
    <tr>
      <th>18</th>
      <td>10</td>
      <td>2</td>
      <td>301</td>
      <td>2</td>
      <td>2017</td>
      <td>6</td>
      <td>18</td>
      <td>13:21</td>
      <td>$80.00</td>
      <td>Swipe Transaction</td>
      <td>-4282466774399734331</td>
      <td>Cookeville</td>
      <td>TN</td>
      <td>38501.0</td>
      <td>4829</td>
      <td>Insufficient Balance,</td>
      <td>No</td>
    </tr>
    <tr>
      <th>19</th>
      <td>10</td>
      <td>2</td>
      <td>301</td>
      <td>2</td>
      <td>2017</td>
      <td>6</td>
      <td>18</td>
      <td>13:21</td>
      <td>$80.00</td>
      <td>Swipe Transaction</td>
      <td>-4282466774399734331</td>
      <td>Cookeville</td>
      <td>TN</td>
      <td>38501.0</td>
      <td>4829</td>
      <td>Insufficient Balance,</td>
      <td>No</td>
    </tr>
  </tbody>
</table>
</div>



```python
del duplicate_hashes
del duplicate_candidate_parts

gc.collect()
```




    0



### Nhận xét M2.5.6

Row-hash audit tạo đúng `24,386,900 hash`, tương ứng với toàn bộ transaction. <br>

Sau khi sort hash, phát hiện `66 duplicate hash candidate`. <br>

Lần đọc thứ hai thu được `132 candidate rows`. <br>

Exact verification trên toàn bộ 15 raw columns xác nhận: <br>
`66 duplicate groups` <br>
`132 duplicate-member rows` <br>
`66 duplicate extra rows` <br>
`largest duplicate group = 2` <br>

Như vậy mọi duplicate group đều là một cặp gồm đúng hai row giống hệt nhau; không xuất hiện group có ba row hoặc nhiều hơn. <br>

Tỷ lệ duplicate extra row chỉ khoảng `0.000271%` toàn dataset. <br>

Đây là quy mô cực nhỏ so với `24,386,900 transaction`. <br>

Duplicate-member rows xuất hiện rải rác từ `1999` đến `2019`, thay vì tập trung tại một năm duy nhất. <br>

Một số năm có nhiều duplicate member hơn, nhưng số tuyệt đối vẫn rất nhỏ; lớn nhất trong output chỉ là `14 duplicate-member rows` ở một năm. <br>

Toàn bộ `132 duplicate-member rows` đều có target: <br>
`Is Fraud? = No` <br>

Không có fraud transaction nào nằm trong duplicate groups. <br>

Preview xác nhận các row trong cùng một group thực sự giống nhau trên toàn bộ các field công khai, bao gồm User, Card, timestamp, Amount, transaction mode, merchant, location, MCC, Errors? và target. <br>

Tuy nhiên dataset không có unique transaction identifier. <br>

Do đó exact equality trên toàn bộ raw schema vẫn chưa chứng minh chắc chắn rằng một row là bản sao lỗi của cùng một transaction thực. <br>

Về mặt chất lượng dữ liệu, duplicate có quy mô quá nhỏ để trở thành blocking issue hoặc ảnh hưởng đáng kể tới class distribution. <br>

Không có lý do EDA nào buộc phải `drop_duplicates()` ngay tại M2.5. <br>

Quyết định giữ hoặc loại các exact duplicate có thể được chuyển sang M4, với lưu ý rằng tác động định lượng của quyết định này sẽ rất nhỏ. <br>

`Kết luận: exact duplicate tồn tại nhưng cực hiếm, chỉ gồm 66 cặp và không chứa fraud. Đây là finding cần ghi nhận chứ không phải vấn đề chất lượng dữ liệu nghiêm trọng hoặc blocking issue.`

## M2.5.7 — Đối chiếu các phát hiện chất lượng dữ liệu với Milestone 1

### Câu hỏi

Các kết quả được tính độc lập ở M2.5 có nhất quán với các mốc đã audit trong M1 hay không? <br>

### Phương pháp

Đối chiếu: <br>
- tổng số transaction; <br>
- missing Merchant State; <br>
- missing Zip; <br>
- missing Errors?; <br>
- Negative Amount; <br>
- Zero Amount; <br>
- State / Zip missing của Online Transaction; <br>
- duplicate groups / members / extra rows; <br>
- fraud trong duplicate-member rows. <br>

Các giá trị M1 chỉ dùng để cross-check sau khi M2.5 đã tự tính kết quả. <br>


```python
online_state_missing_rate = (
    location_by_mode_df
    .loc[
        location_by_mode_df[
            "Use Chip"
        ]
        == "Online Transaction",
        "state_missing_rate_pct",
    ]
    .iloc[0]
)


online_zip_missing_rate = (
    location_by_mode_df
    .loc[
        location_by_mode_df[
            "Use Chip"
        ]
        == "Online Transaction",
        "zip_missing_rate_pct",
    ]
    .iloc[0]
)


m25_consistency_checks = pd.Series(
    {
        "Tổng transaction = 24,386,900":
            total_rows
            == 24_386_900,

        "Merchant State missing = 2,720,821":
            missing_counts[
                "Merchant State"
            ]
            == 2_720_821,

        "Zip missing = 2,878,135":
            missing_counts[
                "Zip"
            ]
            == 2_878_135,

        "Errors? missing = 23,998,469":
            missing_counts[
                "Errors?"
            ]
            == 23_998_469,

        "Negative Amount = 1,244,683":
            amount_negative_count
            == 1_244_683,

        "Zero Amount = 20,213":
            amount_zero_count
            == 20_213,

        "Online State missing = 100%":
            np.isclose(
                online_state_missing_rate,
                100.0,
            ),

        "Online Zip missing = 100%":
            np.isclose(
                online_zip_missing_rate,
                100.0,
            ),

        "Duplicate groups = 66":
            duplicate_group_count
            == 66,

        "Duplicate-member rows = 132":
            duplicate_member_rows
            == 132,

        "Duplicate extra rows = 66":
            duplicate_extra_rows
            == 66,

        "Largest duplicate group = 2":
            largest_duplicate_group
            == 2,

        "Fraud among duplicate members = 0":
            duplicate_fraud_count
            == 0,
    }
)


display(
    m25_consistency_checks
)


print(
    "Tất cả phép đối chiếu đều khớp:",
    m25_consistency_checks.all(),
)
```


    Tổng transaction = 24,386,900         True
    Merchant State missing = 2,720,821    True
    Zip missing = 2,878,135               True
    Errors? missing = 23,998,469          True
    Negative Amount = 1,244,683           True
    Zero Amount = 20,213                  True
    Online State missing = 100%           True
    Online Zip missing = 100%             True
    Duplicate groups = 66                 True
    Duplicate-member rows = 132           True
    Duplicate extra rows = 66             True
    Largest duplicate group = 2           True
    Fraud among duplicate members = 0     True
    dtype: bool


    Tất cả phép đối chiếu đều khớp: True


### Nhận xét M2.5.7

Tất cả các phép đối chiếu giữa kết quả được tính độc lập trong M2.5 và các mốc đã xác minh ở M1 đều trả về `True`. <br>

Các giá trị khớp gồm: <br>
`Tổng transaction = 24,386,900` <br>
`Merchant State missing = 2,720,821` <br>
`Zip missing = 2,878,135` <br>
`Errors? missing = 23,998,469` <br>
`Negative Amount = 1,244,683` <br>
`Zero Amount = 20,213` <br>
`Online State missing = 100%` <br>
`Online Zip missing = 100%` <br>
`Duplicate groups = 66` <br>
`Duplicate-member rows = 132` <br>
`Duplicate extra rows = 66` <br>
`Largest duplicate group = 2` <br>
`Fraud among duplicate members = 0` <br>

Không phát hiện bất nhất giữa quality audit thực hiện trong M2.5 và Technical / Risk Audit của M1. <br>

Quan trọng hơn, M2.5 không chỉ tái lập các con số cũ mà còn làm rõ thêm semantic structure của location missing và Amount sign. <br>

`Kết luận: PASS`

# Tổng kết M2.5 — Phân tích chất lượng dữ liệu và missingness

## Các phát hiện chính

Missingness của dataset không phải một hiện tượng phân bố ngẫu nhiên trên toàn schema. <br>

Chỉ ba cột có missing: <br>
`Errors?` <br>
`Merchant State` <br>
`Zip` <br>

Location missing có cấu trúc rất rõ theo transaction mode và representation của merchant location. <br>

Amount âm / zero cũng có pattern rõ theo transaction mode thay vì phân bố đồng đều. <br>

Exact duplicate tồn tại nhưng ở quy mô cực nhỏ và không chứa fraud. <br>

Không phát hiện vấn đề chất lượng dữ liệu nào buộc phải sửa raw dataset ngay tại M2.5. <br>

## Kết luận về missingness toàn dataset

Các field cốt lõi phục vụ transaction, target và thời gian đều không có missing. <br>

`Errors?` missing khoảng `98.41%`, nhưng field này đã bị loại khỏi Model V1 nên không cần xây preprocessing strategy như một feature đầu vào. <br>

`Merchant State` missing khoảng `11.16%` và `Zip` missing khoảng `11.80%`. <br>

Hai trường này cần được xử lý theo semantic location thay vì bằng một quy tắc missing-value chung. <br>

## Kết luận về structural missingness của location

Toàn bộ `Online Transaction` có: <br>
`Merchant City = ONLINE` <br>
`Merchant State missing` <br>
`Zip missing` <br>

Do đó location missing của Online Transaction là `structural missingness`, không phải bằng chứng về corruption ngẫu nhiên. <br>

Đặc biệt, trong artifact hiện tại: <br>
`Merchant State missing ↔ Merchant City = ONLINE` <br>
là một quan hệ hai chiều hoàn toàn về tập transaction. <br>

Tuy nhiên `Merchant City = ONLINE` không hoàn toàn đồng nghĩa với `Use Chip = Online Transaction`, vì còn `7,601 Chip Transaction` có representation location ONLINE. <br>

Điều này có nghĩa các feature `Use Chip`, `Merchant City`, `Merchant State missing indicator` có mức chồng lấp thông tin rất mạnh nhưng không hoàn toàn giống nhau. <br>

M4 cần cân nhắc redundancy khi lựa chọn representation. <br>

## Kết luận về Merchant City = ONLINE

`Merchant City = ONLINE` không nên được xem là một city địa lý bình thường. <br>

Nó hoạt động như một category semantic biểu diễn nhóm transaction không có physical merchant location trong artifact. <br>

Vì Merchant City không có NaN nhưng chứa category `ONLINE`, raw missing count bằng 0 không có nghĩa field này luôn chứa physical city. <br>

Đây là một ví dụ cho thấy semantic missingness có thể được encode bằng category thay vì null value. <br>

## Kết luận về residual location missingness

Sau khi loại Online Transaction, State missing gần như không còn ngoài nhóm `Chip + City=ONLINE`. <br>

Zip missing thì vẫn còn `164,915` non-online transaction. <br>

Phần lớn pattern quan sát được liên quan đến các merchant quốc tế như Mexico, Canada, Italy, United Kingdom, France, Germany, Japan và nhiều quốc gia khác. <br>

Điều này cho thấy Zip missing có ít nhất hai cơ chế khác nhau: <br>
`Online → không có physical location` <br>
và <br>
`international physical merchant → location tồn tại nhưng Zip không áp dụng / không được representation cung cấp`. <br>

Do đó Zip không nên được impute theo một cơ chế duy nhất mà không xét context. <br>

Output cũng cho thấy `Merchant State` có thể chứa tên quốc gia đối với merchant quốc tế, vì vậy semantic thực tế của field rộng hơn tên cột State gợi ý. <br>

## Kết luận về Amount âm và Amount bằng 0

Amount âm chiếm khoảng `5.10%` dataset và xuất hiện trên cả Swipe, Chip và Online Transaction. <br>

Tỷ lệ negative Amount ở Swipe / Chip khoảng `5.4–5.8%`, trong khi Online chỉ khoảng `0.50%`. <br>

Zero Amount cũng tập trung chủ yếu ở Swipe / Chip và gần như không xuất hiện ở Online. <br>

Như vậy sign của Amount có cấu trúc theo transaction mode và không phù hợp với giả thuyết lỗi parsing hoặc corruption ngẫu nhiên đơn giản. <br>

Semantic chính xác của Amount âm vẫn chưa được xác định. <br>

Do đó guardrail `không abs / drop / clip negative Amount` tiếp tục có hiệu lực. <br>

## Kết luận về Amount cực trị

Minimum Amount của dataset là `-500`, và nhiều transaction ở nhiều context khác nhau cùng đạt đúng boundary này. <br>

Điều này tạo dấu hiệu rằng `-500` có thể phản ánh một boundary hoặc convention của synthetic data, nhưng M2.5 chưa đủ bằng chứng để khóa diễn giải đó. <br>

Extreme positive Amount lên tới `12,390.50` và xuất hiện trên nhiều transaction mode / merchant context. <br>

Không có bằng chứng để tự động coi các extreme Amount là lỗi dữ liệu. <br>

M4 không được loại hoặc clip các extreme values chỉ vì chúng nằm xa phần lớn distribution nếu chưa có quyết định preprocessing có căn cứ. <br>

## Kết luận về exact duplicate

Dataset có `66 exact duplicate groups`, tương ứng `132 duplicate-member rows` và `66 extra rows`. <br>

Mỗi group chỉ gồm hai row. <br>

Duplicate extra rows chỉ chiếm khoảng `0.000271%` dataset. <br>

Duplicate-member rows phân bố qua nhiều năm và toàn bộ đều thuộc class `No`. <br>

Không có unique transaction ID để chứng minh chắc chắn exact duplicate row là cùng một transaction bị ghi hai lần. <br>

Do đó duplicate không phải blocking issue và chưa có lý do phải xóa tại M2.5. <br>

## Các vấn đề cần chuyển sang M2.6

M2.6 cần kiểm tra quan hệ giữa target và các feature / pattern đã được hiểu semantic ở M2.4–M2.5. <br>

Đặc biệt cần kiểm tra: <br>
`Amount ↔ fraud` <br>
`Amount sign ↔ fraud` <br>
`Amount sign × Use Chip ↔ fraud` <br>
`Use Chip ↔ fraud` <br>
`MCC ↔ fraud` <br>
`hour_of_day / day_of_week / month ↔ fraud` <br>
`location / location missingness ↔ fraud` <br>

Khi phân tích missing location ↔ fraud, không được diễn giải missing như một nguyên nhân độc lập vì missing đã được chứng minh có quan hệ rất mạnh với transaction mode và location representation. <br>

M2.6 cũng phải luôn hiển thị `transaction_count`, `fraud_count` và `fraud_rate` cùng nhau để tránh săn category có fraud rate cao nhưng support nhỏ. <br>

## Các vấn đề cần chuyển sang M4

M4 phải quyết định chiến lược representation / missing-value handling cho location dựa trên semantic đã phát hiện. <br>

Không nên dùng một imputation strategy duy nhất cho mọi Zip missing nếu Online và international merchant có cơ chế missing khác nhau. <br>

Cần cân nhắc redundancy giữa: <br>
`Use Chip = Online Transaction` <br>
`Merchant City = ONLINE` <br>
`Merchant State missing` <br>
`Zip missing indicator` <br>

M4 phải tiếp tục giữ dấu của Amount cho đến khi có lý do rõ ràng để biến đổi. <br>

Quyết định có xử lý exact duplicate hay không cũng được chuyển sang M4; nếu loại, tác động định lượng dự kiến rất nhỏ. <br>

## Các limitation còn mở

`Exact semantic của negative Amount` vẫn chưa được khóa. <br>

`Nguyên nhân chính xác của boundary -500` chưa được chứng minh. <br>

`Merchant City = ONLINE` có thêm 7,601 Chip Transaction ngoài Online mode; cơ chế generator chính xác tạo ra nhóm này chưa được biết. <br>

Diễn giải residual Zip missing là liên quan đến international merchant được hỗ trợ mạnh bởi pattern dữ liệu, nhưng chưa phải mô tả chính thức từ generator. <br>

Dataset là synthetic nên một số quality pattern có thể phản ánh logic của generator thay vì behavior của transaction banking thực tế. <br>

## Quyết định M2.5

`Decision ID: M2.5-D01` <br>
`Full-scan quality audit: PASS` <br>
`Schema missingness analysis: PASS` <br>
`Structural location missingness: CONFIRMED` <br>
`Multiple Zip-missing mechanisms: DETECTED` <br>
`Amount-sign structure: CONFIRMED` <br>
`Negative Amount semantic: OPEN LIMITATION` <br>
`Extreme Amount corruption: NOT ESTABLISHED` <br>
`Exact duplicate audit: PASS WITH FINDING` <br>
`Consistency with M1: PASS` <br>
`Blocking data-quality issue: NONE` <br>
`Status: PASS WITH FINDINGS` <br>

## Bước tiếp theo

`Next: M2.6 — Phân tích mối quan hệ giữa feature và target`
