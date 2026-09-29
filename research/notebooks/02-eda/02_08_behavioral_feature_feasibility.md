# M2.8 — Khảo sát behavioral-feature feasibility và temporal windows

## Mục tiêu

M2.7 đã xác nhận phần lớn transaction có prior User/Card history và entity history nhìn chung rất sâu. <br>

M2.8 chuyển sang câu hỏi khó hơn: <br>
`Lịch sử đó có thể được chuyển thành behavioral feature đúng causal, đủ coverage và với chi phí tính toán chấp nhận được hay không?` <br>

M2.8 chỉ prototype một số feature đại diện: <br>
- `time_since_previous_transaction`; <br>
- `transactions_last_1h`; <br>
- `amount_vs_previous_history`; <br>
- `is_new_merchant`. <br>

Đồng thời M2.8 so sánh một số temporal window khả thi để tạo bằng chứng cho M3: <br>
- full raw dataset `1991–2020`; <br>
- full history trước 2020 `1991–2019`; <br>
- recent window `2015–2019`; <br>
- recent window `2018–2019`; <br>
- pre-break window `2018 → 2019-10`. <br>

Window cuối được bổ sung vì M2.3 đã phát hiện fraud regime thay đổi từ khoảng `11/2019`. <br>

## Quy tắc causal bắt buộc

Đối với transaction T: <br>

`historical feature(T)` <br>
chỉ được sử dụng transaction có: <br>
`timestamp(history) < timestamp(T)` <br>

Transaction có cùng Timestamp với T không được dùng làm history cho T. <br>

Không được dùng: <br>
- current transaction; <br>
- future transaction; <br>
- lifetime aggregate chứa tương lai; <br>
- full-dataset statistics gắn ngược về quá khứ. <br>

## Phạm vi

M2.8 là `feasibility study`. <br>

Mục tiêu chưa phải chứng minh behavioral feature cải thiện model. <br>

M2.8 chỉ cần trả lời: <br>
- feature có tính causal được không; <br>
- coverage thực tế bao nhiêu; <br>
- distribution có usable hay quá cực đoan; <br>
- chi phí tính toán có khả thi; <br>
- recent temporal subset có giữ đủ dữ liệu / fraud hay không; <br>
- older history có hữu ích như warm-up cho recent modeling window hay không. <br>

## Ranh giới

Trong M2.8: <br>
- chưa train model; <br>
- chưa feature selection; <br>
- chưa khóa training window; <br>
- chưa khóa final split; <br>
- chưa tạo production feature pipeline; <br>
- chưa dùng raw User / Card / Merchant Name trực tiếp làm classifier feature. <br>

## Thiết lập môi trường thực thi

Notebook được thiết kế để `chạy độc lập`. <br>

M2.8 sử dụng hai lần đọc chính: <br>

`Pass 1` <br>
→ audit physical order của User+Card và tổng hợp temporal-window statistics. <br>

`Pass 2` <br>
→ nếu order audit an toàn, stream từng Card block và prototype các behavioral feature causal. <br>

Không giữ toàn bộ 24 triệu transaction trong RAM. <br>


```python
from pathlib import Path
import time

import numpy as np
import pandas as pd

from IPython.display import display


# ============================================================
# Xác định project root
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
        "Không xác định được PROJECT_ROOT."
    )


DATA_PATH = (
    PROJECT_ROOT
    / DATA_RELATIVE_PATH
)


# ============================================================
# Cấu hình
# ============================================================

CHUNK_SIZE = 500_000

RANDOM_STATE = 42

FEATURE_SAMPLE_FRAC = 0.01


WINDOW_ORDER = [
    "FULL_1991_2020",
    "FULL_1991_2019",
    "RECENT_2015_2019",
    "RECENT_2018_2019",
    "PRE_BREAK_2018_TO_2019_10",
]


print("PROJECT_ROOT:")
print(PROJECT_ROOT)

print("\nDATA_PATH:")
print(DATA_PATH)

print("\nFile tồn tại:")
print(DATA_PATH.exists())
```

    PROJECT_ROOT:
    /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening
    
    DATA_PATH:
    /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/raw/ibm_tabformer/card_transaction.v1.csv
    
    File tồn tại:
    True


### Nhận xét thiết lập môi trường

Notebook đã xác định được đúng `PROJECT_ROOT` và đường dẫn tới raw artifact. <br>

Kết quả `File tồn tại: True` xác nhận M2.8 có thể truy cập trực tiếp dataset và chạy độc lập. <br>

Thiết kế sử dụng hai pass chính: một pass để audit physical order và tổng hợp temporal-window statistics, sau đó một pass để tính behavioral feature theo từng Card block. <br>

Cách tổ chức này tránh phải giữ toàn bộ hơn 24 triệu transaction trong RAM. <br>

`Kết luận: PASS`


```python
def build_timestamp(df):
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
        df["Time"]
        .astype("string")
        + ":00",
        errors="coerce",
    )

    return (
        date_part
        + time_part
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


def build_window_masks(df):
    year = df["Year"]
    month = df["Month"]

    return {
        "FULL_1991_2020":
            pd.Series(
                True,
                index=df.index,
            ),

        "FULL_1991_2019":
            year <= 2019,

        "RECENT_2015_2019":
            year.between(
                2015,
                2019,
            ),

        "RECENT_2018_2019":
            year.between(
                2018,
                2019,
            ),

        "PRE_BREAK_2018_TO_2019_10":
            (
                year.eq(2018)
                |
                (
                    year.eq(2019)
                    & month.le(10)
                )
            ),
    }
```

## M2.8.1 — Audit physical order phục vụ causal streaming

### Vì sao kiểm tra này tồn tại?

M2.2 đã xác nhận raw CSV không tăng dần theo Timestamp trên toàn dataset. <br>

Điều đó chưa trả lời hai câu hỏi quan trọng hơn cho historical feature: <br>
- transaction của một `User+Card` có nằm trong một block liên tục hay không; <br>
- Timestamp bên trong từng Card có không giảm theo raw order hay không. <br>

Nếu cả hai điều kiện đúng, ta có thể stream từng Card theo raw file mà không phải sort toàn bộ dataset. <br>

Nếu một trong hai điều kiện sai, raw order không được dùng trực tiếp để tính history và M2.8 phải yêu cầu explicit sort theo: <br>
`User + Card + Timestamp`. <br>

### Đồng thời

Pass này cũng tổng hợp quy mô và số fraud của các temporal window candidate để tránh thêm một full scan riêng. <br>


```python
ORDER_USECOLS = [
    "User",
    "Card",
    "Year",
    "Month",
    "Day",
    "Time",
    "Is Fraud?",
]


total_rows = 0
timestamp_parse_failures = 0

within_card_timestamp_decreases = 0
within_card_equal_adjacent_timestamps = 0

card_block_count = 0
card_block_reappearance_count = 0

current_block_key = None
closed_block_keys = set()

last_timestamp_by_card = {}

all_card_keys = set()


window_state = {
    window: {
        "transaction_count": 0,
        "fraud_count": 0,
        "users": set(),
        "cards": set(),
    }

    for window
    in WINDOW_ORDER
}


for chunk_number, chunk in enumerate(
    pd.read_csv(
        DATA_PATH,
        usecols=ORDER_USECOLS,
        chunksize=CHUNK_SIZE,
    ),
    start=1,
):
    total_rows += len(chunk)

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    timestamp = build_timestamp(
        chunk
    )

    timestamp_parse_failures += int(
        timestamp
        .isna()
        .sum()
    )

    audit_df = (
        chunk[
            [
                "User",
                "Card",
            ]
        ]
        .copy()
    )

    audit_df[
        "Timestamp"
    ] = timestamp


    # --------------------------------------------------------
    # Internal order trong mỗi chunk
    # --------------------------------------------------------

    timestamp_diff = (
        audit_df
        .groupby(
            [
                "User",
                "Card",
            ],
            sort=False,
        )["Timestamp"]
        .diff()
    )

    within_card_timestamp_decreases += int(
        (
            timestamp_diff
            < pd.Timedelta(0)
        ).sum()
    )

    within_card_equal_adjacent_timestamps += int(
        (
            timestamp_diff
            == pd.Timedelta(0)
        ).sum()
    )


    # --------------------------------------------------------
    # Cross-chunk order
    # --------------------------------------------------------

    first_rows = (
        audit_df
        .groupby(
            [
                "User",
                "Card",
            ],
            sort=False,
        )
        .head(1)
    )

    for row in first_rows.itertuples(
        index=False,
    ):
        key = (
            int(row.User),
            int(row.Card),
        )

        previous_timestamp = (
            last_timestamp_by_card
            .get(key)
        )

        if previous_timestamp is not None:

            if (
                row.Timestamp
                < previous_timestamp
            ):
                within_card_timestamp_decreases += 1

            elif (
                row.Timestamp
                == previous_timestamp
            ):
                within_card_equal_adjacent_timestamps += 1


    last_rows = (
        audit_df
        .groupby(
            [
                "User",
                "Card",
            ],
            sort=False,
        )
        .tail(1)
    )

    for row in last_rows.itertuples(
        index=False,
    ):
        key = (
            int(row.User),
            int(row.Card),
        )

        last_timestamp_by_card[
            key
        ] = row.Timestamp

        all_card_keys.add(
            key
        )


    # --------------------------------------------------------
    # Kiểm tra contiguous Card blocks
    # --------------------------------------------------------

    users = (
        chunk["User"]
        .to_numpy()
    )

    cards = (
        chunk["Card"]
        .to_numpy()
    )

    if len(chunk) > 0:

        change = np.ones(
            len(chunk),
            dtype=bool,
        )

        if len(chunk) > 1:
            change[1:] = (
                (users[1:] != users[:-1])
                |
                (cards[1:] != cards[:-1])
            )

        first_key = (
            int(users[0]),
            int(cards[0]),
        )

        if (
            current_block_key
            is not None
            and first_key
            == current_block_key
        ):
            change[0] = False

        block_starts = np.flatnonzero(
            change
        )

        for index in block_starts:

            new_key = (
                int(
                    users[index]
                ),
                int(
                    cards[index]
                ),
            )

            if current_block_key is None:

                current_block_key = (
                    new_key
                )

                card_block_count += 1

                continue


            if (
                new_key
                != current_block_key
            ):
                closed_block_keys.add(
                    current_block_key
                )

                if (
                    new_key
                    in closed_block_keys
                ):
                    card_block_reappearance_count += 1

                current_block_key = (
                    new_key
                )

                card_block_count += 1


    # --------------------------------------------------------
    # Temporal-window statistics
    # --------------------------------------------------------

    target = (
        chunk["Is Fraud?"]
        .astype("string")
        .str.strip()
    )

    fraud_mask = (
        target
        .eq("Yes")
        .fillna(False)
    )

    window_masks = (
        build_window_masks(
            chunk
        )
    )

    for window_name, mask in (
        window_masks.items()
    ):
        if not mask.any():
            continue

        state = (
            window_state[
                window_name
            ]
        )

        state[
            "transaction_count"
        ] += int(
            mask.sum()
        )

        state[
            "fraud_count"
        ] += int(
            fraud_mask.loc[
                mask
            ].sum()
        )

        state[
            "users"
        ].update(
            int(value)

            for value
            in chunk.loc[
                mask,
                "User",
            ].unique()
        )

        card_pairs = (
            chunk.loc[
                mask,
                [
                    "User",
                    "Card",
                ],
            ]
            .drop_duplicates()
        )

        state[
            "cards"
        ].update(
            (
                int(user),
                int(card),
            )

            for user, card
            in card_pairs.itertuples(
                index=False,
                name=None,
            )
        )


    if (
        chunk_number % 10 == 0
        or len(chunk) < CHUNK_SIZE
    ):
        print(
            f"Đã audit {chunk_number} chunk "
            f"- tổng số dòng: {total_rows:,}"
        )


print("\nHoàn tất order audit.")

print(
    "Tổng số dòng:",
    f"{total_rows:,}",
)

print(
    "Timestamp parse failures:",
    timestamp_parse_failures,
)

print(
    "Unique User+Card:",
    len(
        all_card_keys
    ),
)

print(
    "Card block count:",
    card_block_count,
)

print(
    "Card block reappearance:",
    card_block_reappearance_count,
)

print(
    "Within-card timestamp decreases:",
    within_card_timestamp_decreases,
)

print(
    "Adjacent equal timestamps within Card:",
    within_card_equal_adjacent_timestamps,
)
```

    Đã audit 10 chunk - tổng số dòng: 5,000,000
    Đã audit 20 chunk - tổng số dòng: 10,000,000
    Đã audit 30 chunk - tổng số dòng: 15,000,000
    Đã audit 40 chunk - tổng số dòng: 20,000,000
    Đã audit 49 chunk - tổng số dòng: 24,386,900
    
    Hoàn tất order audit.
    Tổng số dòng: 24,386,900
    Timestamp parse failures: 0
    Unique User+Card: 6139
    Card block count: 6139
    Card block reappearance: 0
    Within-card timestamp decreases: 0
    Adjacent equal timestamps within Card: 142010


### Nhận xét M2.8.1

Order audit đã xử lý đầy đủ `24,386,900 transaction` qua `49 chunk`. <br>

Không có lỗi Timestamp: <br>
`Timestamp parse failures = 0` <br>

Dataset có đúng: <br>
`6,139 User+Card` <br>
và <br>
`6,139 Card block`. <br>

Không phát hiện Card nào xuất hiện trở lại sau khi block của nó đã kết thúc: <br>
`Card block reappearance = 0`. <br>

Quan trọng hơn, không phát hiện Timestamp giảm bên trong bất kỳ User+Card nào: <br>
`Within-card timestamp decreases = 0`. <br>

Như vậy raw CSV tuy không được sắp xếp tăng dần theo thời gian trên toàn dataset, nhưng có một cấu trúc hữu ích hơn cho bài toán history: <br>
`mỗi User+Card nằm trong một contiguous block và Timestamp không giảm bên trong block đó`. <br>

Điều này có nghĩa card-level historical feature có thể được tính bằng streaming trên artifact hiện tại mà không cần sort toàn bộ 24 triệu dòng trước. <br>

Audit cũng phát hiện `142,010 adjacent equal-timestamp pairs` bên trong Card. <br>

Đây không phải lỗi ordering vì Timestamp không giảm, nhưng nó xác nhận rằng nhiều transaction có thể xảy ra cùng một phút. <br>

Do guardrail yêu cầu `timestamp(history) < timestamp(current)`, transaction cùng Timestamp không được phép cung cấp history cho nhau. <br>

Vì vậy việc prototype M2.8 xử lý theo các nhóm Timestamp khác nhau, thay vì đơn giản dùng raw-row `shift(1)`, là cần thiết để tránh leakage cùng thời điểm. <br>

`Kết luận: raw artifact hiện tại hỗ trợ causal streaming ở cấp User+Card, nhưng pipeline M4 không nên mặc định tin vào physical order của mọi artifact tương lai; phải giữ order assertion hoặc explicit sort User + Card + Timestamp.`

## M2.8.2 — So sánh quy mô các temporal window candidate

### Câu hỏi

Nếu không nhất thiết dùng toàn bộ 1991–2020 cho experiment, các recent temporal window còn giữ: <br>
- bao nhiêu transaction; <br>
- bao nhiêu fraud; <br>
- bao nhiêu User; <br>
- bao nhiêu Card? <br>

### Mục tiêu

Không chọn final split tại M2.8. <br>

Chỉ đánh giá trade-off giữa: <br>
`nhiều lịch sử hơn` <br>
và <br>
`dữ liệu gần future regime hơn`. <br>

`FULL_1991_2020` được giữ như raw baseline nhưng đã biết 2020 là partial year và có 0 fraud. <br>

`FULL_1991_2019` loại riêng phần 2020 nhưng vẫn bao phủ gần toàn lịch sử. <br>


```python
window_rows = []


for window_name in WINDOW_ORDER:

    state = (
        window_state[
            window_name
        ]
    )

    transaction_count = (
        state[
            "transaction_count"
        ]
    )

    fraud_count = (
        state[
            "fraud_count"
        ]
    )

    window_rows.append(
        {
            "window":
                window_name,

            "transaction_count":
                transaction_count,

            "share_of_full_dataset_pct":
                transaction_count
                / total_rows
                * 100,

            "fraud_count":
                fraud_count,

            "fraud_rate_pct":
                fraud_count
                / transaction_count
                * 100,

            "unique_users":
                len(
                    state[
                        "users"
                    ]
                ),

            "unique_user_card_pairs":
                len(
                    state[
                        "cards"
                    ]
                ),
        }
    )


window_summary_df = pd.DataFrame(
    window_rows
)


display(
    window_summary_df
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
      <th>window</th>
      <th>transaction_count</th>
      <th>share_of_full_dataset_pct</th>
      <th>fraud_count</th>
      <th>fraud_rate_pct</th>
      <th>unique_users</th>
      <th>unique_user_card_pairs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>FULL_1991_2020</td>
      <td>24386900</td>
      <td>100.000000</td>
      <td>29757</td>
      <td>0.122020</td>
      <td>2000</td>
      <td>6139</td>
    </tr>
    <tr>
      <th>1</th>
      <td>FULL_1991_2019</td>
      <td>24050400</td>
      <td>98.620161</td>
      <td>29757</td>
      <td>0.123728</td>
      <td>1657</td>
      <td>5028</td>
    </tr>
    <tr>
      <th>2</th>
      <td>RECENT_2015_2019</td>
      <td>8579208</td>
      <td>35.179576</td>
      <td>11693</td>
      <td>0.136295</td>
      <td>1621</td>
      <td>4435</td>
    </tr>
    <tr>
      <th>3</th>
      <td>RECENT_2018_2019</td>
      <td>3445553</td>
      <td>14.128704</td>
      <td>4578</td>
      <td>0.132867</td>
      <td>1587</td>
      <td>4152</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PRE_BREAK_2018_TO_2019_10</td>
      <td>3157028</td>
      <td>12.945590</td>
      <td>4578</td>
      <td>0.145010</td>
      <td>1541</td>
      <td>4078</td>
    </tr>
  </tbody>
</table>
</div>


### Nhận xét M2.8.2

Các temporal window candidate đều còn quy mô đủ lớn cho experiment, nhưng trade-off giữa recency và sample size rất rõ. <br>

Toàn bộ `1991–2020`: <br>
`24,386,900 transaction` <br>
`29,757 fraud` <br>
`fraud rate ≈ 0.1220%` <br>
`2,000 User` <br>
`6,139 User+Card`. <br>

Nếu chỉ dùng dữ liệu đến hết 2019: <br>
`24,050,400 transaction` <br>
`29,757 fraud` <br>
`fraud rate ≈ 0.1237%`. <br>

Như vậy `336,500 transaction` của 2020 không bổ sung fraud nào. <br>

Đồng thời số unique entity tăng từ `1,657 → 2,000 User` và từ `5,028 → 6,139 User+Card` khi thêm 2020. <br>

Do đó trong artifact hiện tại có một số lượng đáng kể entity chỉ xuất hiện ở phần 2020, trong khi giai đoạn này lại không chứa positive class. <br>

Finding này tiếp tục củng cố việc không sử dụng 2020 làm final fraud-performance test. <br>

Window `2015–2019` còn: <br>
`8,579,208 transaction` <br>
`11,693 fraud` <br>
`1,621 User` <br>
`4,435 User+Card`. <br>

Dù chỉ chứa khoảng `35.18%` số transaction toàn dataset, window này vẫn giữ hơn 11 nghìn fraud và phần lớn User population. <br>

Window `2018–2019` còn: <br>
`3,445,553 transaction` <br>
`4,578 fraud` <br>
`1,587 User` <br>
`4,152 User+Card`. <br>

Window `2018 → 2019-10` còn: <br>
`3,157,028 transaction` <br>
`4,578 fraud` <br>
`1,541 User` <br>
`4,078 User+Card`. <br>

Điểm đáng chú ý là khi loại `2019-11 → 2019-12`, số transaction giảm `288,525` nhưng fraud count vẫn giữ nguyên `4,578`. <br>

Do đó fraud rate tăng từ khoảng `0.1329%` ở 2018–2019 lên khoảng `0.1450%` ở pre-break window. <br>

Điều này hoàn toàn nhất quán với finding M2.3 rằng hai tháng cuối 2019 thuộc zero-fraud regime. <br>

Các recent window vì vậy vẫn đủ lớn cho modeling experiment và đồng thời giảm đáng kể lượng dữ liệu phải xử lý. <br>

`Kết luận: không có yêu cầu về sample size buộc M3 phải train trên toàn bộ 1991–2019. Các recent temporal window vẫn giữ hàng triệu transaction, hàng nghìn fraud và hàng nghìn Card; pre-break window tránh trực tiếp zero-fraud regime cuối 2019.`

## M2.8.3 — Gate cho causal streaming

Feature prototype bên dưới chỉ được chạy nếu raw artifact thỏa đồng thời: <br>
- mỗi User+Card nằm trong một contiguous block; <br>
- Timestamp không giảm bên trong User+Card; <br>
- Timestamp parse không lỗi. <br>

Nếu gate fail, M2.8 không được tự sửa bằng raw row order. <br>

Khi đó pipeline chính thức phải explicit sort theo: <br>
`User + Card + Timestamp`. <br>


```python
causal_streaming_gate = pd.Series(
    {
        "Timestamp parse failures = 0":
            timestamp_parse_failures
            == 0,

        "Card block reappearance = 0":
            card_block_reappearance_count
            == 0,

        "Within-card timestamp decreases = 0":
            within_card_timestamp_decreases
            == 0,

        "Card block count = unique cards":
            card_block_count
            == len(
                all_card_keys
            ),
    }
)


display(
    causal_streaming_gate
)


CAUSAL_STREAMING_SAFE = bool(
    causal_streaming_gate.all()
)


print(
    "CAUSAL_STREAMING_SAFE:",
    CAUSAL_STREAMING_SAFE,
)


if not CAUSAL_STREAMING_SAFE:
    raise RuntimeError(
        "Raw order không đủ an toàn để causal streaming. "
        "Cần explicit sort User + Card + Timestamp "
        "trước khi tiếp tục M2.8."
    )
```


    Timestamp parse failures = 0           True
    Card block reappearance = 0            True
    Within-card timestamp decreases = 0    True
    Card block count = unique cards        True
    dtype: bool


    CAUSAL_STREAMING_SAFE: True


### Nhận xét M2.8.3

Tất cả bốn điều kiện của causal-streaming gate đều trả về `True`: <br>

`Timestamp parse failures = 0` <br>
`Card block reappearance = 0` <br>
`Within-card timestamp decreases = 0` <br>
`Card block count = unique cards` <br>

Kết quả cuối cùng: <br>
`CAUSAL_STREAMING_SAFE = True` <br>

Vì vậy artifact hiện tại đủ điều kiện để prototype card-level historical feature trực tiếp bằng streaming. <br>

Đây là một finding về physical representation của artifact hiện tại, không phải assumption được phép áp dụng vô điều kiện cho dữ liệu khác. <br>

Pipeline sau này phải tiếp tục kiểm tra invariant này hoặc explicit sort trước khi tính history. <br>

`Kết luận: PASS — causal streaming được xác nhận an toàn trên raw artifact hiện tại.`

## M2.8.4 — Xây cơ chế stream từng User+Card

Do M2.8.3 đã khóa gate order, raw file có thể được đọc thành từng Card block. <br>

Chỉ một Card block được giữ trong bộ nhớ tại một thời điểm. <br>

Nếu block bị cắt bởi chunk boundary, phần cuối chunk trước và đầu chunk sau được ghép lại trước khi xử lý. <br>


```python
FEATURE_USECOLS = [
    "User",
    "Card",
    "Year",
    "Month",
    "Day",
    "Time",
    "Amount",
    "Merchant Name",
]


def iter_card_blocks(
    data_path,
    chunksize,
):
    pending_key = None
    pending_parts = []

    for chunk in pd.read_csv(
        data_path,
        usecols=FEATURE_USECOLS,
        chunksize=chunksize,
    ):
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

        if chunk["Amount_numeric"].isna().any():
            raise ValueError(
                "Phát hiện Amount parse failure "
                "trong causal feature pass."
            )

        working = (
            chunk[
                [
                    "User",
                    "Card",
                    "Year",
                    "Month",
                    "Timestamp",
                    "Amount_numeric",
                    "Merchant Name",
                ]
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


            if len(
                pending_parts
            ) == 1:

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


    if pending_key is not None:

        if len(
            pending_parts
        ) == 1:

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

## M2.8.5 — Prototype behavioral feature theo strict causal rule

### Feature 1 — time_since_previous_transaction

Khoảng thời gian tới timestamp transaction trước gần nhất của cùng Card. <br>

Chỉ timestamp nhỏ hơn current Timestamp được sử dụng. <br>

Nếu nhiều transaction cùng timestamp, chúng không được coi là history của nhau. <br>

### Feature 2 — transactions_last_1h

Số transaction trước đó của Card nằm trong khoảng: <br>
`[T - 1 giờ, T)` <br>

Current transaction và transaction cùng Timestamp T bị loại. <br>

### Feature 3 — amount_vs_previous_history

Đầu tiên tính: <br>
`previous_amount_mean` <br>
từ toàn bộ transaction có Timestamp nhỏ hơn T. <br>

Sau đó prototype: <br>
`amount_minus_previous_mean = current Amount - previous_amount_mean` <br>

Không dùng absolute value và không sửa dấu Amount. <br>

### Feature 4 — is_new_merchant

`True` nếu Merchant Name chưa từng xuất hiện ở Card tại Timestamp nhỏ hơn T. <br>

Nếu nhiều transaction tới cùng merchant tại first timestamp của merchant, tất cả đều được xem là new vì không transaction nào được phép dùng peer cùng Timestamp làm history. <br>

### Lưu ý

Các feature này được tạo để kiểm tra feasibility. <br>

Chúng chưa phải final feature set. <br>


```python
ONE_HOUR_NS = int(
    pd.Timedelta(
        hours=1
    ).value
)


def compute_causal_prototype_features(
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


    # --------------------------------------------------------
    # Các timestamp group khác nhau
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Strict prior transaction count
    # --------------------------------------------------------

    prior_transaction_count_group = (
        group_starts
    )


    prior_transaction_count = np.repeat(
        prior_transaction_count_group,
        group_lengths,
    )


    # --------------------------------------------------------
    # Feature 1:
    # time_since_previous_transaction
    # --------------------------------------------------------

    time_since_previous_group = np.full(
        len(
            group_starts
        ),
        np.nan,
        dtype="float64",
    )


    if len(
        group_starts
    ) > 1:

        time_since_previous_group[
            1:
        ] = (
            (
                group_timestamp_ns[1:]
                - group_timestamp_ns[:-1]
            )
            / 60_000_000_000
        )


    time_since_previous_min = np.repeat(
        time_since_previous_group,
        group_lengths,
    )


    # --------------------------------------------------------
    # Feature 2:
    # transactions_last_1h
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Feature 3:
    # previous_amount_mean
    # amount_minus_previous_mean
    # --------------------------------------------------------

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
        / group_starts[
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


    # --------------------------------------------------------
    # Feature 4:
    # is_new_merchant
    # --------------------------------------------------------

    first_merchant_timestamp = (
        card_df
        .groupby(
            "Merchant Name",
            sort=False,
        )[
            "Timestamp"
        ]
        .transform(
            "min"
        )
    )


    is_new_merchant = (
        card_df[
            "Timestamp"
        ]
        .eq(
            first_merchant_timestamp
        )
        .to_numpy()
    )


    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    feature_df = pd.DataFrame(
        {
            "Year":
                card_df[
                    "Year"
                ]
                .to_numpy(),

            "Month":
                card_df[
                    "Month"
                ]
                .to_numpy(),

            "prior_transaction_count":
                prior_transaction_count,

            "time_since_previous_transaction_min":
                time_since_previous_min,

            "transactions_last_1h":
                transactions_last_1h,

            "previous_amount_mean":
                previous_amount_mean,

            "amount_minus_previous_mean":
                amount_minus_previous_mean,

            "is_new_merchant":
                is_new_merchant,
        }
    )


    return feature_df
```

## M2.8.6 — Chạy causal prototype trên toàn dataset

### Mục tiêu

Đo chính xác: <br>
- bao nhiêu transaction có strict prior Card history; <br>
- bao nhiêu transaction có ít nhất một transaction trong 1 giờ trước; <br>
- bao nhiêu transaction có previous Amount history; <br>
- bao nhiêu transaction gặp merchant mới; <br>
- các tỷ lệ trên thay đổi ra sao theo temporal window. <br>

Đồng thời lưu sample 1% để khảo sát distribution mà không giữ toàn feature matrix trong RAM. <br>


```python
feature_window_state = {
    window: {
        "row_count": 0,
        "prior_history_count": 0,
        "last_1h_positive_count": 0,
        "amount_history_available_count": 0,
        "new_merchant_count": 0,
        "window_only_cold_start_count": 0,
        "global_history_cold_start_count": 0,
    }

    for window
    in WINDOW_ORDER
}


feature_sample_parts = []


feature_rows_processed = 0
feature_cards_processed = 0

max_card_block_rows = 0


feature_pass_start = (
    time.perf_counter()
)


for card_key, card_df in (
    iter_card_blocks(
        DATA_PATH,
        CHUNK_SIZE,
    )
):
    feature_cards_processed += 1

    feature_rows_processed += len(
        card_df
    )

    max_card_block_rows = max(
        max_card_block_rows,
        len(
            card_df
        ),
    )


    feature_df = (
        compute_causal_prototype_features(
            card_df
        )
    )


    # --------------------------------------------------------
    # Exact window-level coverage
    # --------------------------------------------------------

    window_masks = (
        build_window_masks(
            feature_df
        )
    )


    for window_name, mask in (
        window_masks.items()
    ):
        if not mask.any():
            continue

        state = (
            feature_window_state[
                window_name
            ]
        )

        sub = (
            feature_df.loc[
                mask
            ]
        )


        state[
            "row_count"
        ] += len(
            sub
        )


        state[
            "prior_history_count"
        ] += int(
            (
                sub[
                    "prior_transaction_count"
                ]
                > 0
            ).sum()
        )


        state[
            "last_1h_positive_count"
        ] += int(
            (
                sub[
                    "transactions_last_1h"
                ]
                > 0
            ).sum()
        )


        state[
            "amount_history_available_count"
        ] += int(
            sub[
                "previous_amount_mean"
            ]
            .notna()
            .sum()
        )


        state[
            "new_merchant_count"
        ] += int(
            sub[
                "is_new_merchant"
            ]
            .sum()
        )


        # ----------------------------------------------------
        # Global history cold-start
        # ----------------------------------------------------

        state[
            "global_history_cold_start_count"
        ] += int(
            (
                sub[
                    "prior_transaction_count"
                ]
                == 0
            ).sum()
        )


        # ----------------------------------------------------
        # Nếu reset history tại đầu window:
        # tất cả transaction ở first timestamp
        # của Card trong window sẽ cold-start
        # ----------------------------------------------------

        card_window_timestamps = (
            card_df.loc[
                mask,
                "Timestamp",
            ]
        )

        first_window_timestamp = (
            card_window_timestamps
            .min()
        )

        state[
            "window_only_cold_start_count"
        ] += int(
            card_window_timestamps
            .eq(
                first_window_timestamp
            )
            .sum()
        )


    # --------------------------------------------------------
    # Reproducible 1% sample
    # --------------------------------------------------------

    seed = (
        RANDOM_STATE
        + card_key[0] * 1009
        + card_key[1] * 9176
    ) % (
        2 ** 32
    )


    rng = np.random.default_rng(
        seed
    )


    sample_mask = (
        rng.random(
            len(
                feature_df
            )
        )
        < FEATURE_SAMPLE_FRAC
    )


    if sample_mask.any():

        feature_sample_parts.append(
            feature_df.loc[
                sample_mask,
                [
                    "Year",
                    "Month",
                    "time_since_previous_transaction_min",
                    "transactions_last_1h",
                    "previous_amount_mean",
                    "amount_minus_previous_mean",
                    "is_new_merchant",
                ],
            ]
            .copy()
        )


    if (
        feature_cards_processed
        % 500
        == 0
    ):
        print(
            f"Đã xử lý {feature_cards_processed:,} Card "
            f"- {feature_rows_processed:,} transaction"
        )


feature_pass_elapsed_seconds = (
    time.perf_counter()
    - feature_pass_start
)


feature_sample_df = pd.concat(
    feature_sample_parts,
    ignore_index=True,
)


print("\nHoàn tất causal feature prototype.")

print(
    "Card processed:",
    f"{feature_cards_processed:,}",
)

print(
    "Rows processed:",
    f"{feature_rows_processed:,}",
)

print(
    "Max Card block rows:",
    f"{max_card_block_rows:,}",
)

print(
    "Feature sample rows:",
    f"{len(feature_sample_df):,}",
)

print(
    "Elapsed seconds:",
    round(
        feature_pass_elapsed_seconds,
        2,
    ),
)
```

    Đã xử lý 500 Card - 2,021,584 transaction
    Đã xử lý 1,000 Card - 4,066,794 transaction
    Đã xử lý 1,500 Card - 5,915,634 transaction
    Đã xử lý 2,000 Card - 7,883,446 transaction
    Đã xử lý 2,500 Card - 9,842,479 transaction
    Đã xử lý 3,000 Card - 11,897,259 transaction
    Đã xử lý 3,500 Card - 13,997,374 transaction
    Đã xử lý 4,000 Card - 16,040,304 transaction
    Đã xử lý 4,500 Card - 18,115,490 transaction
    Đã xử lý 5,000 Card - 19,945,815 transaction
    Đã xử lý 5,500 Card - 21,956,418 transaction
    Đã xử lý 6,000 Card - 23,844,247 transaction
    
    Hoàn tất causal feature prototype.
    Card processed: 6,139
    Rows processed: 24,386,900
    Max Card block rows: 70,008
    Feature sample rows: 244,285
    Elapsed seconds: 90.67


### Nhận xét M2.8.6

Causal feature pass đã xử lý thành công toàn bộ: <br>
`6,139 Card` <br>
`24,386,900 transaction`. <br>

Không cần tạo một full feature DataFrame gồm 24 triệu dòng; mỗi lần chỉ xử lý một Card block. <br>

Card block lớn nhất có `70,008 transaction`, phù hợp với quy mô card history đã phát hiện ở M2.7. <br>

Sample phục vụ distribution chứa `244,285 transaction`, tương đương xấp xỉ `1%` toàn dataset như thiết kế. <br>

Bốn prototype feature đều được tạo theo strict causal rule: <br>
- `time_since_previous_transaction`; <br>
- `transactions_last_1h`; <br>
- `previous_amount_mean / amount_minus_previous_mean`; <br>
- `is_new_merchant`. <br>

Current transaction và các transaction có cùng Timestamp không được dùng làm history của nhau. <br>

Feature generation cũng không sử dụng target, vì vậy không tạo direct target leakage trong quá trình tính historical feature. <br>

Toàn bộ pass hoàn thành trong khoảng `90.67 giây` trên môi trường hiện tại. <br>

`Kết luận: bốn loại behavioral feature đại diện đều có thể được tính causal trên full artifact bằng streaming với chi phí thực thi thực tế chấp nhận được.`

## M2.8.7 — Coverage của historical feature

### Câu hỏi

`Có prior history` ở M2.7 có chuyển thành feature availability thực tế hay không? <br>

Đặc biệt: <br>
- time-since-previous có available không; <br>
- trong 1 giờ gần nhất có transaction nào không; <br>
- previous Amount mean có available không; <br>
- merchant mới xuất hiện thường xuyên đến đâu? <br>

### Lưu ý

`transactions_last_1h = 0` vẫn là một feature value hợp lệ. <br>

Bảng dưới đây dùng `last_1h_positive_rate` để đo mật độ history ngắn hạn, không phải feature missingness. <br>


```python
feature_coverage_rows = []


for window_name in WINDOW_ORDER:

    state = (
        feature_window_state[
            window_name
        ]
    )

    row_count = (
        state[
            "row_count"
        ]
    )


    feature_coverage_rows.append(
        {
            "window":
                window_name,

            "row_count":
                row_count,

            "prior_card_history_pct":
                (
                    state[
                        "prior_history_count"
                    ]
                    / row_count
                    * 100
                ),

            "has_transaction_last_1h_pct":
                (
                    state[
                        "last_1h_positive_count"
                    ]
                    / row_count
                    * 100
                ),

            "previous_amount_mean_available_pct":
                (
                    state[
                        "amount_history_available_count"
                    ]
                    / row_count
                    * 100
                ),

            "is_new_merchant_rate_pct":
                (
                    state[
                        "new_merchant_count"
                    ]
                    / row_count
                    * 100
                ),
        }
    )


feature_coverage_df = pd.DataFrame(
    feature_coverage_rows
)


display(
    feature_coverage_df
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
      <th>window</th>
      <th>row_count</th>
      <th>prior_card_history_pct</th>
      <th>has_transaction_last_1h_pct</th>
      <th>previous_amount_mean_available_pct</th>
      <th>is_new_merchant_rate_pct</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>FULL_1991_2020</td>
      <td>24386900</td>
      <td>99.974745</td>
      <td>20.934092</td>
      <td>99.974745</td>
      <td>4.015980</td>
    </tr>
    <tr>
      <th>1</th>
      <td>FULL_1991_2019</td>
      <td>24050400</td>
      <td>99.979023</td>
      <td>20.955394</td>
      <td>99.979023</td>
      <td>3.945211</td>
    </tr>
    <tr>
      <th>2</th>
      <td>RECENT_2015_2019</td>
      <td>8579208</td>
      <td>99.992762</td>
      <td>19.478733</td>
      <td>99.992762</td>
      <td>2.912903</td>
    </tr>
    <tr>
      <th>3</th>
      <td>RECENT_2018_2019</td>
      <td>3445553</td>
      <td>99.992831</td>
      <td>19.408641</td>
      <td>99.992831</td>
      <td>2.724236</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PRE_BREAK_2018_TO_2019_10</td>
      <td>3157028</td>
      <td>99.994520</td>
      <td>19.400683</td>
      <td>99.994520</td>
      <td>2.746285</td>
    </tr>
  </tbody>
</table>
</div>


### Nhận xét M2.8.7

Kết quả xác nhận sự khác biệt quan trọng giữa: <br>
`có prior Card history` <br>
và <br>
`có history trong một cửa sổ thời gian ngắn`. <br>

Trên toàn dataset, khoảng `99.9747% transaction` có ít nhất một Card transaction với Timestamp nhỏ hơn current Timestamp. <br>

Ở các recent window, tỷ lệ này còn cao hơn khi cho phép sử dụng historical warm-up: <br>
`2015–2019: ≈ 99.9928%` <br>
`2018–2019: ≈ 99.9928%` <br>
`2018 → 2019-10: ≈ 99.9945%`. <br>

`previous_amount_mean` có availability chính xác bằng prior-card-history coverage. <br>

Điều này cho thấy với artifact hiện tại, hầu như mọi transaction có prior Card history đều có thể tính được running previous Amount mean. <br>

Tuy nhiên `transactions_last_1h` kể một câu chuyện khác. <br>

Chỉ khoảng: <br>
`20.93%` full dataset <br>
và khoảng `19.4%` ở các recent window <br>
có ít nhất một prior transaction trong một giờ gần nhất. <br>

Do đó khoảng `79–81% transaction` có: <br>
`transactions_last_1h = 0`. <br>

Đây không phải missing value. <br>
Zero là một giá trị behavioral hợp lệ và cho biết Card không có transaction trong rolling 1-hour window. <br>

Kết quả này xác nhận M2.7 không thể suy ra short-window density chỉ từ lifetime history depth. <br>

Đối với `is_new_merchant`, tỷ lệ transaction gặp merchant chưa từng xuất hiện trước đó trên cùng Card là: <br>
`≈ 4.02%` trên full history; <br>
`≈ 2.91%` trong 2015–2019; <br>
`≈ 2.72%` trong 2018–2019; <br>
`≈ 2.75%` trong pre-break window. <br>

Như vậy merchant novelty không quá hiếm và không phải feature gần như luôn False. <br>

Tỷ lệ thấp hơn trong recent windows cũng phù hợp với việc các Card đã tích lũy nhiều merchant history trước đó, nhưng M2.8 chưa phân tích relationship với fraud nên chưa thể kết luận predictive value. <br>

`Kết luận: historical feature coverage rất cao đối với lifetime/running-history feature, trong khi short-window velocity feature có cấu trúc zero-heavy rõ ràng. Cả hai dạng đều khả thi nhưng biểu diễn những khía cạnh hành vi khác nhau.`

## M2.8.8 — Phân bố xấp xỉ của behavioral feature

### Phương pháp

Sử dụng sample 1% có thể tái hiện từ causal feature pass. <br>

Đối với mỗi temporal window, tính: <br>
- median; <br>
- P90; <br>
- P95; <br>
- P99; <br>
- min; <br>
- max. <br>

Các quantile này là `xấp xỉ từ sample`, không phải population statistic chính xác. <br>


```python
DISTRIBUTION_FEATURES = [
    "time_since_previous_transaction_min",
    "transactions_last_1h",
    "amount_minus_previous_mean",
]


feature_distribution_rows = []


sample_window_masks = (
    build_window_masks(
        feature_sample_df
    )
)


for window_name in WINDOW_ORDER:

    window_mask = (
        sample_window_masks[
            window_name
        ]
    )

    for feature_name in (
        DISTRIBUTION_FEATURES
    ):
        values = (
            feature_sample_df
            .loc[
                window_mask,
                feature_name,
            ]
            .dropna()
        )

        if values.empty:
            continue


        feature_distribution_rows.append(
            {
                "window":
                    window_name,

                "feature":
                    feature_name,

                "sample_non_missing_count":
                    len(
                        values
                    ),

                "min":
                    float(
                        values.min()
                    ),

                "median":
                    float(
                        values.quantile(
                            0.50
                        )
                    ),

                "p90":
                    float(
                        values.quantile(
                            0.90
                        )
                    ),

                "p95":
                    float(
                        values.quantile(
                            0.95
                        )
                    ),

                "p99":
                    float(
                        values.quantile(
                            0.99
                        )
                    ),

                "max":
                    float(
                        values.max()
                    ),
            }
        )


feature_distribution_df = pd.DataFrame(
    feature_distribution_rows
)


display(
    feature_distribution_df
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
      <th>window</th>
      <th>feature</th>
      <th>sample_non_missing_count</th>
      <th>min</th>
      <th>median</th>
      <th>p90</th>
      <th>p95</th>
      <th>p99</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>FULL_1991_2020</td>
      <td>time_since_previous_transaction_min</td>
      <td>244230</td>
      <td>1.000</td>
      <td>521.000000</td>
      <td>2705.000000</td>
      <td>4003.000000</td>
      <td>7980.130000</td>
      <td>176694.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>FULL_1991_2020</td>
      <td>transactions_last_1h</td>
      <td>244285</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>9.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>FULL_1991_2020</td>
      <td>amount_minus_previous_mean</td>
      <td>244230</td>
      <td>-1215.048</td>
      <td>-8.833947</td>
      <td>56.680624</td>
      <td>91.808285</td>
      <td>260.946215</td>
      <td>3879.610428</td>
    </tr>
    <tr>
      <th>3</th>
      <td>FULL_1991_2019</td>
      <td>time_since_previous_transaction_min</td>
      <td>240895</td>
      <td>1.000</td>
      <td>520.000000</td>
      <td>2700.000000</td>
      <td>3995.000000</td>
      <td>7968.000000</td>
      <td>176694.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>FULL_1991_2019</td>
      <td>transactions_last_1h</td>
      <td>240941</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>9.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>FULL_1991_2019</td>
      <td>amount_minus_previous_mean</td>
      <td>240895</td>
      <td>-1215.048</td>
      <td>-8.844531</td>
      <td>56.702102</td>
      <td>91.866682</td>
      <td>261.408921</td>
      <td>3879.610428</td>
    </tr>
    <tr>
      <th>6</th>
      <td>RECENT_2015_2019</td>
      <td>time_since_previous_transaction_min</td>
      <td>86056</td>
      <td>1.000</td>
      <td>612.000000</td>
      <td>2884.000000</td>
      <td>4317.000000</td>
      <td>8678.450000</td>
      <td>65147.000000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>RECENT_2015_2019</td>
      <td>transactions_last_1h</td>
      <td>86064</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>8.000000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>RECENT_2015_2019</td>
      <td>amount_minus_previous_mean</td>
      <td>86056</td>
      <td>-847.215</td>
      <td>-9.103737</td>
      <td>56.526972</td>
      <td>91.896017</td>
      <td>263.327623</td>
      <td>3115.315693</td>
    </tr>
    <tr>
      <th>9</th>
      <td>RECENT_2018_2019</td>
      <td>time_since_previous_transaction_min</td>
      <td>34554</td>
      <td>1.000</td>
      <td>609.000000</td>
      <td>2873.000000</td>
      <td>4309.000000</td>
      <td>8627.410000</td>
      <td>65147.000000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>RECENT_2018_2019</td>
      <td>transactions_last_1h</td>
      <td>34556</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>8.000000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>RECENT_2018_2019</td>
      <td>amount_minus_previous_mean</td>
      <td>34554</td>
      <td>-847.215</td>
      <td>-8.862781</td>
      <td>56.920874</td>
      <td>92.251803</td>
      <td>270.948319</td>
      <td>1651.782283</td>
    </tr>
    <tr>
      <th>12</th>
      <td>PRE_BREAK_2018_TO_2019_10</td>
      <td>time_since_previous_transaction_min</td>
      <td>31591</td>
      <td>1.000</td>
      <td>611.000000</td>
      <td>2873.000000</td>
      <td>4308.000000</td>
      <td>8624.000000</td>
      <td>65147.000000</td>
    </tr>
    <tr>
      <th>13</th>
      <td>PRE_BREAK_2018_TO_2019_10</td>
      <td>transactions_last_1h</td>
      <td>31593</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>8.000000</td>
    </tr>
    <tr>
      <th>14</th>
      <td>PRE_BREAK_2018_TO_2019_10</td>
      <td>amount_minus_previous_mean</td>
      <td>31591</td>
      <td>-847.215</td>
      <td>-8.833805</td>
      <td>57.060658</td>
      <td>92.775606</td>
      <td>272.685498</td>
      <td>1651.782283</td>
    </tr>
  </tbody>
</table>
</div>


### Nhận xét M2.8.8

Sample 1% cho thấy ba behavioral feature số có distribution rất khác nhau. <br>

Đối với `time_since_previous_transaction`: <br>

Full dataset có: <br>
`median ≈ 521 phút` <br>
`P90 ≈ 2,705 phút` <br>
`P95 ≈ 4,003 phút` <br>
`P99 ≈ 7,980 phút`. <br>

Median khoảng `8.7 giờ`, trong khi P99 đã khoảng `5.5 ngày`. <br>

Ở recent windows, median tăng lên khoảng `609–612 phút`, tức khoảng `10.2 giờ`; P99 khoảng `8,624–8,678 phút`, gần `6 ngày`. <br>

Maximum rất lớn: khoảng `176,694 phút` trên full history và `65,147 phút` trong recent windows. <br>

Như vậy time-since-previous có distribution lệch phải mạnh và trải trên nhiều thang thời gian. <br>

Feature này không phải constant hoặc gần constant, nhưng nếu đưa vào model tuyến tính / scale-sensitive thì M4 có thể cần cân nhắc transformation phù hợp, ví dụ một phép biến đổi monotonic như `log1p`, chỉ sau khi fit pipeline đúng trên train. <br>

Đối với `transactions_last_1h`: <br>
`median = 0` <br>
`P90 = 1` <br>
`P95 = 2` <br>
`P99 = 3` <br>
`max = 8–9`. <br>

Đây là một feature count rời rạc, zero-heavy nhưng vẫn có variation rõ ở upper tail. <br>

Kết quả phù hợp với coverage analysis rằng chỉ khoảng một phần năm transaction có prior activity trong một giờ. <br>

Đối với `amount_minus_previous_mean`: <br>
median ổn định quanh `-8.8 đến -9.1`; <br>
P90 khoảng `56–57`; <br>
P95 khoảng `92`; <br>
P99 khoảng `261–273`. <br>

Full sample có khoảng giá trị từ xấp xỉ `-1,215` tới `3,880`, cho thấy distribution khá rộng và có extreme deviation. <br>

Mean-based historical baseline có thể bị ảnh hưởng bởi skewness và extreme Amount đã phát hiện ở M2.4. <br>

Do đó M4 có thể cân nhắc so sánh running-mean deviation với một robust alternative, nhưng M2.8 chưa có lý do để thay thế feature hiện tại hoặc sửa Amount. <br>

Quan trọng hơn, distribution của ba feature đều có variation thực sự và không cho thấy prototype nào bị collapse thành một constant vô dụng. <br>

`Kết luận: các behavioral prototype đều tạo distribution usable ở mức feasibility; time-since và amount-deviation có tail dài, còn 1-hour velocity có cấu trúc discrete zero-heavy.`

## M2.8.9 — Tác động của historical warm-up

### Vì sao kiểm tra này tồn tại?

Nếu M3 chọn một recent modeling window như `2018–2019`, có hai chiến lược khác nhau: <br>

`Strategy A — history warm-up` <br>
→ transaction model nằm trong recent window; <br>
→ nhưng behavioral feature được phép sử dụng raw history xảy ra trước window nếu history đó nằm trước prediction point. <br>

`Strategy B — reset history tại window start` <br>
→ mọi Card bị xem như chưa có history khi bước vào đầu window. <br>

Hai chiến lược tạo mức cold-start rất khác nhau. <br>

M2.8 chỉ đo sự khác biệt; chưa khóa strategy cuối cùng. <br>


```python
warmup_rows = []


for window_name in WINDOW_ORDER:

    state = (
        feature_window_state[
            window_name
        ]
    )

    row_count = (
        state[
            "row_count"
        ]
    )

    global_cold = (
        state[
            "global_history_cold_start_count"
        ]
    )

    reset_cold = (
        state[
            "window_only_cold_start_count"
        ]
    )


    warmup_rows.append(
        {
            "window":
                window_name,

            "row_count":
                row_count,

            "cold_start_with_prior_history_warmup":
                global_cold,

            "cold_start_with_prior_history_warmup_pct":
                global_cold
                / row_count
                * 100,

            "cold_start_if_history_reset_at_window_start":
                reset_cold,

            "cold_start_if_history_reset_pct":
                reset_cold
                / row_count
                * 100,

            "transactions_helped_by_pre_window_history":
                reset_cold
                - global_cold,

            "transactions_helped_by_pre_window_history_pct":
                (
                    reset_cold
                    - global_cold
                )
                / row_count
                * 100,
        }
    )


history_warmup_df = pd.DataFrame(
    warmup_rows
)


display(
    history_warmup_df
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
      <th>window</th>
      <th>row_count</th>
      <th>cold_start_with_prior_history_warmup</th>
      <th>cold_start_with_prior_history_warmup_pct</th>
      <th>cold_start_if_history_reset_at_window_start</th>
      <th>cold_start_if_history_reset_pct</th>
      <th>transactions_helped_by_pre_window_history</th>
      <th>transactions_helped_by_pre_window_history_pct</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>FULL_1991_2020</td>
      <td>24386900</td>
      <td>6159</td>
      <td>0.025255</td>
      <td>6159</td>
      <td>0.025255</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>FULL_1991_2019</td>
      <td>24050400</td>
      <td>5045</td>
      <td>0.020977</td>
      <td>5045</td>
      <td>0.020977</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>RECENT_2015_2019</td>
      <td>8579208</td>
      <td>621</td>
      <td>0.007238</td>
      <td>4455</td>
      <td>0.051928</td>
      <td>3834</td>
      <td>0.044689</td>
    </tr>
    <tr>
      <th>3</th>
      <td>RECENT_2018_2019</td>
      <td>3445553</td>
      <td>247</td>
      <td>0.007169</td>
      <td>4159</td>
      <td>0.120706</td>
      <td>3912</td>
      <td>0.113538</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PRE_BREAK_2018_TO_2019_10</td>
      <td>3157028</td>
      <td>173</td>
      <td>0.005480</td>
      <td>4085</td>
      <td>0.129394</td>
      <td>3912</td>
      <td>0.123914</td>
    </tr>
  </tbody>
</table>
</div>


### Nhận xét M2.8.9

Historical warm-up làm giảm đáng kể cold-start theo nghĩa tương đối khi sử dụng recent modeling window. <br>

Đối với full-history window, warm-up và reset giống nhau vì window bắt đầu từ đầu history. <br>

Nhưng ở recent windows, khác biệt trở nên rõ. <br>

Với `2015–2019`: <br>
nếu giữ prior history từ trước 2015, chỉ có `621 transaction` cold-start, khoảng `0.00724%`. <br>

Nếu reset toàn bộ history tại đầu 2015, cold-start tăng lên `4,455 transaction`, khoảng `0.05193%`. <br>

Pre-window history giúp `3,834 transaction`. <br>

Với `2018–2019`: <br>
warm-up → `247 cold-start transaction`, khoảng `0.00717%`; <br>
reset → `4,159`, khoảng `0.12071%`. <br>

Pre-2018 history giúp thêm `3,912 transaction`. <br>

Tính theo số cold-start transaction, historical warm-up làm giảm khoảng `94%` cold-start so với reset ở window 2018–2019. <br>

Với `2018 → 2019-10`: <br>
warm-up chỉ còn `173 cold-start transaction`, khoảng `0.00548%`; <br>
reset history tạo `4,085 cold-start transaction`, khoảng `0.12939%`. <br>

Historical warm-up làm giảm khoảng `95.8%` số cold-start transaction so với reset. <br>

Tuy nhiên cần nhìn cả absolute effect: số transaction được pre-window history hỗ trợ chỉ chiếm khoảng `0.12%` toàn pre-break window. <br>

Do đó warm-up tạo cải thiện rất lớn về `tỷ lệ cold-start tương đối`, nhưng vì cold-start vốn đã hiếm nên tác động lên tổng số transaction là nhỏ. <br>

Finding này vẫn quan trọng về thiết kế pipeline: nếu chọn recent modeling rows, không có lý do causal nào buộc phải quên lịch sử hợp lệ xảy ra trước window start. <br>

Có thể dùng older transaction như `history warm-up` mà không dùng chúng làm training rows, miễn mọi feature tại T chỉ sử dụng dữ liệu xảy ra trước T và không sử dụng future labels. <br>

`Kết luận: historical warm-up là một thiết kế hợp lệ và hữu ích, đặc biệt để giảm card-level cold-start trong recent modeling windows; nó cho phép tách modeling-window recency khỏi history availability.`

## M2.8.10 — Chi phí tính toán của causal prototype

### Câu hỏi

Causal behavioral feature có thể được tính trên toàn artifact bằng streaming với tài nguyên notebook thông thường hay không? <br>

### Chỉ số

Ghi nhận: <br>
- tổng số rows processed; <br>
- số Card processed; <br>
- Card block lớn nhất; <br>
- thời gian feature pass; <br>
- throughput rows/second; <br>
- kích thước sample giữ trong RAM. <br>

Runtime phụ thuộc máy đang chạy nên chỉ là benchmark cho project environment hiện tại, không phải đặc tính cố định của dataset. <br>


```python
performance_summary = pd.Series(
    {
        "feature_rows_processed":
            feature_rows_processed,

        "feature_cards_processed":
            feature_cards_processed,

        "max_card_block_rows":
            max_card_block_rows,

        "feature_sample_rows":
            len(
                feature_sample_df
            ),

        "elapsed_seconds":
            feature_pass_elapsed_seconds,

        "rows_per_second":
            (
                feature_rows_processed
                / feature_pass_elapsed_seconds
            ),
    }
)


display(
    performance_summary
)
```


    feature_rows_processed     2.438690e+07
    feature_cards_processed    6.139000e+03
    max_card_block_rows        7.000800e+04
    feature_sample_rows        2.442850e+05
    elapsed_seconds            9.067388e+01
    rows_per_second            2.689518e+05
    dtype: float64


### Nhận xét M2.8.10

Causal prototype đã xử lý: <br>
`24,386,900 transaction` <br>
`6,139 Card`. <br>

Card block lớn nhất chứa `70,008 transaction`. <br>

Sample distribution chứa `244,285 row`, xấp xỉ 1% dataset. <br>

Full causal feature pass hoàn thành trong khoảng: <br>
`90.67 giây` <br>

Throughput đạt khoảng: <br>
`268,952 transaction / giây`. <br>

Benchmark này phụ thuộc vào máy và software environment hiện tại, nên không được xem là runtime cố định của pipeline. <br>

Tuy nhiên kết quả đủ để trả lời câu hỏi feasibility: với representation của artifact hiện tại, việc tính card-level causal behavioral feature trên full dataset không phải computational blocker. <br>

Streaming theo Card còn tránh yêu cầu phải giữ toàn bộ feature matrix hoặc toàn bộ raw dataset trong RAM cùng lúc. <br>

`Kết luận: computational feasibility được xác nhận; behavioral feature engineering ở quy mô full artifact là khả thi trong phạm vi project.`

## M2.8.11 — Kiểm tra tính nhất quán

### Mục tiêu

Xác nhận: <br>
- feature prototype đã xử lý đúng toàn bộ raw artifact; <br>
- mỗi Card được xử lý đúng một block; <br>
- các temporal-window count khớp các finding trước; <br>
- không có dấu hiệu raw-order assumption bị vi phạm. <br>


```python
window_check = (
    window_summary_df
    .set_index(
        "window"
    )
)


m28_consistency_checks = pd.Series(
    {
        "Tổng transaction = 24,386,900":
            total_rows
            == 24_386_900,

        "Unique User+Card = 6,139":
            len(
                all_card_keys
            )
            == 6_139,

        "Timestamp parse failures = 0":
            timestamp_parse_failures
            == 0,

        "Within-card timestamp decreases = 0":
            within_card_timestamp_decreases
            == 0,

        "Card block reappearance = 0":
            card_block_reappearance_count
            == 0,

        "Card blocks = 6,139":
            card_block_count
            == 6_139,

        "Feature rows processed = full dataset":
            feature_rows_processed
            == total_rows,

        "Feature cards processed = 6,139":
            feature_cards_processed
            == 6_139,

        "2015-2019 rows = 8,579,208":
            int(
                window_check.loc[
                    "RECENT_2015_2019",
                    "transaction_count",
                ]
            )
            == 8_579_208,

        "2015-2019 fraud = 11,693":
            int(
                window_check.loc[
                    "RECENT_2015_2019",
                    "fraud_count",
                ]
            )
            == 11_693,

        "2018-2019 rows = 3,445,553":
            int(
                window_check.loc[
                    "RECENT_2018_2019",
                    "transaction_count",
                ]
            )
            == 3_445_553,

        "2018-2019 fraud = 4,578":
            int(
                window_check.loc[
                    "RECENT_2018_2019",
                    "fraud_count",
                ]
            )
            == 4_578,

        "Pre-break 2018 to 2019-10 rows = 3,157,028":
            int(
                window_check.loc[
                    "PRE_BREAK_2018_TO_2019_10",
                    "transaction_count",
                ]
            )
            == 3_157_028,

        "Pre-break 2018 to 2019-10 fraud = 4,578":
            int(
                window_check.loc[
                    "PRE_BREAK_2018_TO_2019_10",
                    "fraud_count",
                ]
            )
            == 4_578,
    }
)


display(
    m28_consistency_checks
)


print(
    "Tất cả phép kiểm tra đều khớp:",
    m28_consistency_checks.all(),
)
```


    Tổng transaction = 24,386,900                 True
    Unique User+Card = 6,139                      True
    Timestamp parse failures = 0                  True
    Within-card timestamp decreases = 0           True
    Card block reappearance = 0                   True
    Card blocks = 6,139                           True
    Feature rows processed = full dataset         True
    Feature cards processed = 6,139               True
    2015-2019 rows = 8,579,208                    True
    2015-2019 fraud = 11,693                      True
    2018-2019 rows = 3,445,553                    True
    2018-2019 fraud = 4,578                       True
    Pre-break 2018 to 2019-10 rows = 3,157,028    True
    Pre-break 2018 to 2019-10 fraud = 4,578       True
    dtype: bool


    Tất cả phép kiểm tra đều khớp: True


### Nhận xét M2.8.11

Tất cả consistency check của M2.8 đều trả về `True`. <br>

Các mốc được xác nhận gồm: <br>
`Tổng transaction = 24,386,900` <br>
`Unique User+Card = 6,139` <br>
`Timestamp parse failures = 0` <br>
`Within-card timestamp decreases = 0` <br>
`Card block reappearance = 0` <br>
`Card blocks = 6,139` <br>
`Feature rows processed = full dataset` <br>
`Feature cards processed = 6,139` <br>

Temporal-window cross-check cũng khớp: <br>
`2015–2019 = 8,579,208 transaction / 11,693 fraud` <br>
`2018–2019 = 3,445,553 transaction / 4,578 fraud` <br>
`2018 → 2019-10 = 3,157,028 transaction / 4,578 fraud`. <br>

Không phát hiện bất nhất giữa causal prototype, temporal-window aggregation và các finding M2 trước. <br>

`Kết luận: PASS`

# Tổng kết M2.8 — Behavioral-feature feasibility và temporal windows

## Các phát hiện chính

M2.8 đã xác nhận rằng behavioral feature engineering không chỉ khả thi về mặt cấu trúc history mà còn khả thi về mặt causal computation và chi phí thực thi. <br>

Raw artifact có cấu trúc đặc biệt thuận lợi: mỗi `User+Card` nằm trong một contiguous block và Timestamp không giảm bên trong Card. <br>

Nhờ đó toàn bộ hơn 24 triệu transaction có thể được xử lý bằng causal streaming mà không cần sort toàn dataset trong notebook hiện tại. <br>

Bốn prototype đại diện cho `recency`, `velocity`, `historical amount deviation` và `merchant novelty` đều được tính thành công theo strict causal rule. <br>

Recent temporal windows vẫn giữ hàng triệu transaction và hàng nghìn fraud, nên dataset size không buộc project phải huấn luyện trên toàn bộ lịch sử 1991–2019. <br>

## Kết luận về raw entity order

Raw CSV không chronological trên toàn dataset nhưng có order phù hợp ở cấp User+Card. <br>

Có đúng `6,139 Card block`, không có block reappearance và không có Timestamp decrease bên trong Card. <br>

Do đó raw artifact hiện tại cho phép card-level causal streaming. <br>

Tuy nhiên có `142,010 adjacent equal-timestamp pairs`, nên transaction cùng timestamp phải tiếp tục được xử lý như cùng prediction time và không được dùng làm history cho nhau. <br>

Pipeline M4 phải giữ explicit order check hoặc explicit sort; không được phụ thuộc âm thầm vào physical CSV order. <br>

## Kết luận về khả năng causal streaming

Causal-streaming gate đạt toàn bộ điều kiện và trả về: <br>
`CAUSAL_STREAMING_SAFE = True`. <br>

Toàn bộ `24,386,900 transaction` và `6,139 Card` đã được feature pass xử lý thành công. <br>

Causal streaming vì vậy được xác nhận khả thi trên artifact hiện tại. <br>

## Kết luận về time_since_previous_transaction

Feature `time_since_previous_transaction` có availability gần tương đương prior-card-history coverage. <br>

Distribution lệch phải rõ rệt: median khoảng `8.7 giờ` trên full data và khoảng `10.2 giờ` ở recent windows, trong khi P99 gần `5.5–6 ngày`. <br>

Feature có variation lớn và đáng mang sang M4 để tiếp tục experiment. <br>

Nếu cần transformation, quyết định phải được thực hiện trong train-only preprocessing. <br>

## Kết luận về transactions_last_1h

Khoảng `19–21%` transaction có ít nhất một prior transaction trong một giờ. <br>

Do đó phần lớn transaction có `transactions_last_1h = 0`. <br>

Zero ở đây là behavioral signal hợp lệ chứ không phải missing. <br>

Feature có distribution rời rạc: median 0, P90 1, P95 2 và P99 3. <br>

Velocity feature vì vậy khả thi nhưng có cấu trúc zero-heavy và cần được model đánh giá thay vì giả định có predictive power. <br>

## Kết luận về amount_vs_previous_history

`previous_amount_mean` có availability trên gần như toàn bộ transaction có prior Card history. <br>

`amount_minus_previous_mean` tạo distribution rộng và asymmetric, với median khoảng `-9` nhưng upper tail lên hàng trăm hoặc hàng nghìn đơn vị Amount. <br>

Feature có thể tính causal hiệu quả bằng running sum / count. <br>

Tuy nhiên mean lịch sử nhạy với tail và extreme Amount, nên M4 có thể cân nhắc robust historical baseline khác nếu model experiment cho thấy cần thiết. <br>

Negative Amount tiếp tục được giữ nguyên; M2.8 không dùng abs / clip. <br>

## Kết luận về is_new_merchant

`is_new_merchant` có tỷ lệ True khoảng `4.02%` trên full history và khoảng `2.7–2.9%` trong recent windows. <br>

Feature không quá hiếm và có thể được tính causal bằng merchant history của từng Card. <br>

Raw Merchant Name vẫn không được dùng trực tiếp làm classifier feature; chỉ trạng thái novelty dẫn xuất từ past history được xem là candidate. <br>

M2.8 chưa kiểm tra quan hệ của novelty với fraud nên chưa kết luận predictive usefulness. <br>

## Kết luận về historical-feature coverage

Prior Card history coverage rất cao: khoảng `99.97%` trên full dataset và trên `99.99%` ở recent windows khi dùng historical warm-up. <br>

Running-history feature như previous Amount mean vì vậy có coverage gần hoàn chỉnh. <br>

Short-window activity lại thưa hơn nhiều; chỉ khoảng một phần năm transaction có prior activity trong một giờ. <br>

Do đó lifetime history depth và short-window density là hai khái niệm khác nhau và phải được biểu diễn bằng các feature khác nhau. <br>

## Kết luận về feature distribution

Không prototype nào bị collapse thành constant. <br>

`time_since_previous_transaction` có long right tail. <br>

`transactions_last_1h` là count zero-heavy nhưng có variation. <br>

`amount_minus_previous_mean` có distribution rộng và extreme deviations. <br>

Các distribution này đủ để tiếp tục modeling experiment nhưng có thể cần representation / transformation khác nhau ở M4. <br>

## Kết luận về historical warm-up

Older history có thể được sử dụng như warm-up cho recent modeling window mà không tạo leakage nếu chỉ sử dụng transaction xảy ra trước prediction point. <br>

Ở window 2018–2019, giữ pre-2018 history giảm cold-start từ `4,159` xuống còn `247 transaction`, tương đương giảm khoảng `94%` số cold-start. <br>

Ở pre-break window, cold-start giảm từ `4,085` xuống `173`, tương đương giảm khoảng `95.8%`. <br>

Tuy nhiên số transaction được warm-up hỗ trợ chỉ chiếm khoảng `0.1%` tổng rows vì cold-start vốn đã rất hiếm. <br>

Finding quan trọng là modeling rows và history warm-up rows không cần phải là cùng một tập dữ liệu. <br>

## Kết luận về computational feasibility

Full causal feature pass hoàn thành trong khoảng `90.67 giây`, với throughput khoảng `269 nghìn transaction/giây` trên environment hiện tại. <br>

Card block lớn nhất chỉ cần xử lý `70,008 transaction` tại một thời điểm. <br>

Không phát hiện computational blocker đối với card-level behavioral feature trong phạm vi project. <br>

## So sánh các temporal window

`2015–2019` giữ khoảng `8.58 triệu transaction` và `11,693 fraud`. <br>

`2018–2019` giữ khoảng `3.45 triệu transaction` và `4,578 fraud`. <br>

`2018 → 2019-10` vẫn giữ toàn bộ `4,578 fraud` của window 2018–2019 nhưng loại `288,525 transaction` thuộc zero-fraud regime cuối 2019. <br>

Do đó các recent windows vẫn có sample size đủ lớn cho experiment. <br>

Full historical data có thể vẫn có giá trị như `history warm-up`, nhưng M2.6 đã chứng minh full-history feature-target association không ổn định theo thời gian, nên không có bằng chứng buộc phải dùng mọi historical row làm model-training row. <br>

## Khuyến nghị chuyển sang M3

M3 nên ưu tiên thiết kế evaluation theo thời gian và so sánh một số recent training-window strategy thay vì mặc định dùng toàn bộ 1991–2019. <br>

Ít nhất nên giữ hai hướng candidate để experiment: <br>
- một window dài hơn như `2015–2019` nhằm giữ nhiều fraud và nhiều lịch sử gần; <br>
- một window recent / pre-break gần 2018–2019-10 nhằm giảm ảnh hưởng của các temporal regime cũ và late-2019 zero-fraud regime. <br>

Final train / validation / test boundary vẫn phải được quyết định ở M3 dựa trên past → future ordering và positive-class support. <br>

`2020` tiếp tục không phù hợp làm final fraud-performance test. <br>

Older transactions có thể được dùng làm causal history warm-up dù không được chọn làm model-training rows. <br>

M3 cần ghi rõ sự khác biệt giữa: <br>
`history available before prediction` <br>
và <br>
`rows được dùng để fit classifier`. <br>

## Khuyến nghị chuyển sang M4

M4 có thể mang forward ít nhất bốn behavioral candidate: <br>
`time_since_previous_transaction` <br>
`transactions_last_1h` <br>
`amount_minus_previous_mean` <br>
`is_new_merchant`. <br>

Feature pipeline phải đảm bảo strict causal ordering và xử lý transaction cùng Timestamp như cùng prediction point. <br>

Không được tính lifetime aggregate từ cả tương lai rồi gắn lại cho transaction quá khứ. <br>

Nếu pipeline tiếp tục tận dụng raw physical order, phải giữ order assertions; cách an toàn tổng quát hơn là explicit sort theo User + Card + Timestamp. <br>

Có thể nghiên cứu thêm các candidate như `transactions_last_10m`, amount rolling windows, `is_new_mcc` hoặc location novelty sau khi bốn prototype cơ bản được tích hợp đúng. <br>

Các transformation học từ dữ liệu vẫn phải fit chỉ trên training partition. <br>

## Các limitation còn mở

M2.8 mới chứng minh `computability`, `coverage` và `distribution`; chưa chứng minh behavioral feature cải thiện classification performance. <br>

Chưa kiểm tra relationship giữa các behavioral feature prototype và target. <br>

Chưa so sánh model có / không có behavioral feature. <br>

Chưa khóa final temporal training window. <br>

Runtime hiện tại chỉ là benchmark trên project environment, không phải cam kết performance trên mọi máy. <br>

Physical-order finding chỉ được xác minh trên raw artifact hiện tại. <br>

`previous_amount_mean` mới là một prototype; chưa so sánh với robust historical baseline như median. <br>

Cold-start hoàn toàn mới vẫn là evaluation segment nhỏ và cần được ghi rõ như limitation. <br>

## Quyết định M2.8

`Decision ID: M2.8-D01` <br>
`Raw User+Card block structure: VERIFIED` <br>
`Within-card temporal order: VERIFIED` <br>
`Strict same-timestamp handling: REQUIRED` <br>
`Causal streaming feasibility: CONFIRMED` <br>
`Behavioral-feature coverage: HIGH FOR RUNNING HISTORY` <br>
`Short-window activity density: SPARSE BUT USABLE` <br>
`Merchant novelty feasibility: CONFIRMED` <br>
`Historical warm-up value: CONFIRMED` <br>
`Recent temporal-subset feasibility: CONFIRMED` <br>
`Computational feasibility: CONFIRMED` <br>
`Final training window: NOT YET LOCKED` <br>
`Consistency checks: PASS` <br>
`Blocking issue: NONE` <br>
`Status: PASS WITH FINDINGS` <br>

## Bước tiếp theo

`Next: M2.9 — Tổng hợp EDA findings, decision log và M2 Gate`
