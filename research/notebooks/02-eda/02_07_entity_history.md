# M2.7 — Phân tích entity và lịch sử User / Card

## Mục tiêu

M2.7 đánh giá liệu cấu trúc `User + Card + Timestamp` của dataset có cung cấp đủ lịch sử để xây behavioral feature theo quan hệ nhân quả hay không. <br>

Câu hỏi trung tâm là: <br>
`Tại thời điểm một transaction cần được screening, entity liên quan thường đã có lịch sử trước đó hay đang ở trạng thái cold-start?` <br>

Các nhiệm vụ chính gồm: <br>
- phân tích số transaction trên mỗi User; <br>
- phân tích số Card trên mỗi User; <br>
- phân tích số transaction trên mỗi `User + Card`; <br>
- đo khoảng thời gian hoạt động của User / Card; <br>
- xác định số transaction thực sự không có lịch sử trước đó; <br>
- kiểm tra mức entity overlap giữa `past <= 2018` và `future = 2019`; <br>
- định lượng new-user / new-card cold-start trong 2019; <br>
- đánh giá liệu cấu trúc history có đủ cơ sở để sang M2.8 thử behavioral feature hay không. <br>

## Hai khái niệm cold-start cần phân biệt

`Strict transaction-level cold-start` <br>
→ transaction xảy ra tại timestamp đầu tiên của entity; <br>
→ không tồn tại transaction nào của entity có `timestamp < timestamp hiện tại`. <br>

`Evaluation-period entity cold-start` <br>
→ entity xuất hiện trong future evaluation period nhưng hoàn toàn chưa xuất hiện trong historical training period. <br>

Hai khái niệm này trả lời hai câu hỏi khác nhau và không được trộn lẫn. <br>

## Guardrail

`User`, `Card` và `Merchant Name` là identifier / history key. <br>

Không sử dụng giá trị raw của chúng trực tiếp làm classifier feature. <br>

Mọi behavioral feature sau này phải tuân thủ: <br>
`timestamp(history) < timestamp(current transaction)` <br>

Không sử dụng transaction hiện tại hoặc tương lai để xây feature cho transaction hiện tại. <br>

## Ranh giới

Trong M2.7: <br>
- chưa tạo rolling-window feature cuối cùng; <br>
- chưa tạo `transactions_last_10m`; <br>
- chưa tạo `amount_sum_last_1h`; <br>
- chưa tạo previous-history mean / median; <br>
- chưa quyết định training window; <br>
- chưa huấn luyện model. <br>

M2.7 chỉ đánh giá `cấu trúc và độ sâu của history`. <br>

Việc prototype behavioral feature causal được dành cho M2.8. <br>

## Thiết lập môi trường thực thi

Notebook này được thiết kế để `chạy độc lập`. <br>

Do số User và User+Card tương đối nhỏ so với số transaction, M2.7 không cần giữ toàn bộ 24 triệu dòng trong RAM. <br>

Dataset được đọc theo chunk. <br>

Trong mỗi chunk, notebook chỉ giữ các thống kê aggregate của từng entity: <br>
- transaction count; <br>
- timestamp đầu tiên; <br>
- số transaction tại timestamp đầu tiên; <br>
- timestamp cuối cùng. <br>

Các thống kê `first / last / lifetime count` trong notebook này chỉ dùng để đánh giá feasibility ở mức EDA. <br>

Chúng không được phép sử dụng trực tiếp như per-transaction model feature vì có chứa thông tin từ toàn bộ lifetime của entity. <br>


```python
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

from IPython.display import display


# ============================================================
# Xác định thư mục gốc project
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
# Cấu hình M2.7
# ============================================================

CHUNK_SIZE = 500_000

M27_USECOLS = [
    "User",
    "Card",
    "Year",
    "Month",
    "Day",
    "Time",
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
    500000


### Nhận xét thiết lập môi trường

Notebook đã xác định được đúng `PROJECT_ROOT` và đường dẫn tới raw artifact. <br>

Kết quả `File tồn tại: True` xác nhận M2.7 có thể chạy độc lập, không phụ thuộc vào kernel hoặc biến của các notebook M2 trước. <br>

`CHUNK_SIZE = 500,000` được sử dụng để quét toàn bộ dataset theo từng phần mà không cần giữ 24 triệu transaction trong bộ nhớ. <br>

`Kết luận: PASS`

## M2.7.1 — Quét toàn bộ dataset và xây entity-history summary

### Câu hỏi

Mỗi User và mỗi User+Card có: <br>
- bao nhiêu transaction; <br>
- timestamp đầu tiên; <br>
- timestamp cuối cùng; <br>
- bao nhiêu transaction xảy ra tại timestamp đầu tiên? <br>

### Vì sao cần đếm số transaction tại timestamp đầu tiên?

Timestamp của dataset có độ phân giải tới phút. <br>

Nếu nhiều transaction của cùng một entity xảy ra tại cùng timestamp đầu tiên, tất cả các transaction đó đều không có history thỏa điều kiện: <br>
`timestamp(history) < timestamp(current transaction)` <br>

Do đó strict cold-start phải được xác định theo timestamp chứ không theo raw row order. <br>

### Phương pháp

Trong một full scan: <br>
- tạo Timestamp từ Year / Month / Day / Time; <br>
- aggregate theo User; <br>
- aggregate theo `User + Card`; <br>
- lưu count, first timestamp, last timestamp và first-timestamp count; <br>
- đồng thời thu thập entity overlap giữa `past <= 2018` và `2019`. <br>

Raw row order không được sử dụng làm thứ tự lịch sử. <br>


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


def aggregate_entity_chunk(
    entity_df,
    group_columns,
):
    grouped_timestamp = (
        entity_df
        .groupby(
            group_columns,
            sort=False,
        )["Timestamp"]
    )

    aggregate_df = (
        grouped_timestamp
        .agg(
            transaction_count="size",
            first_timestamp="min",
            last_timestamp="max",
        )
        .reset_index()
    )

    first_timestamp_in_chunk = (
        grouped_timestamp
        .transform("min")
    )

    first_mask = (
        entity_df["Timestamp"]
        .eq(
            first_timestamp_in_chunk
        )
    )

    first_count_df = (
        entity_df
        .loc[
            first_mask,
            group_columns,
        ]
        .groupby(
            group_columns,
            sort=False,
        )
        .size()
        .rename(
            "first_timestamp_transaction_count"
        )
        .reset_index()
    )

    return (
        aggregate_df
        .merge(
            first_count_df,
            on=group_columns,
            how="left",
        )
    )


def update_entity_state(
    state,
    aggregate_df,
    group_columns,
):
    key_size = len(
        group_columns
    )

    expected_columns = (
        group_columns
        + [
            "transaction_count",
            "first_timestamp",
            "last_timestamp",
            "first_timestamp_transaction_count",
        ]
    )

    aggregate_df = (
        aggregate_df[
            expected_columns
        ]
    )

    for values in (
        aggregate_df
        .itertuples(
            index=False,
            name=None,
        )
    ):
        if key_size == 1:
            key = int(
                values[0]
            )
        else:
            key = tuple(
                int(value)
                for value
                in values[:key_size]
            )

        transaction_count = int(
            values[
                key_size
            ]
        )

        first_timestamp = (
            values[
                key_size + 1
            ]
        )

        last_timestamp = (
            values[
                key_size + 2
            ]
        )

        first_timestamp_count = int(
            values[
                key_size + 3
            ]
        )

        if key not in state:
            state[key] = {
                "transaction_count":
                    transaction_count,

                "first_timestamp":
                    first_timestamp,

                "last_timestamp":
                    last_timestamp,

                "first_timestamp_transaction_count":
                    first_timestamp_count,
            }

            continue

        current = state[key]

        current[
            "transaction_count"
        ] += transaction_count

        if (
            first_timestamp
            < current[
                "first_timestamp"
            ]
        ):
            current[
                "first_timestamp"
            ] = first_timestamp

            current[
                "first_timestamp_transaction_count"
            ] = first_timestamp_count

        elif (
            first_timestamp
            == current[
                "first_timestamp"
            ]
        ):
            current[
                "first_timestamp_transaction_count"
            ] += first_timestamp_count

        if (
            last_timestamp
            > current[
                "last_timestamp"
            ]
        ):
            current[
                "last_timestamp"
            ] = last_timestamp
```


```python
total_rows = 0
timestamp_parse_failures = 0


# ============================================================
# Entity lifetime state
# ============================================================

user_state = {}

card_state = {}


# ============================================================
# Past <= 2018
# ============================================================

past_users = set()

past_cards = set()


# ============================================================
# Future = 2019
# ============================================================

users_2019 = set()

cards_2019 = set()

transactions_2019_by_user = Counter()

transactions_2019_by_card = Counter()
```


```python
for chunk_number, chunk in enumerate(
    pd.read_csv(
        DATA_PATH,
        usecols=M27_USECOLS,
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

    failures_in_chunk = int(
        timestamp
        .isna()
        .sum()
    )

    timestamp_parse_failures += (
        failures_in_chunk
    )

    if failures_in_chunk > 0:
        raise ValueError(
            "Phát hiện Timestamp không hợp lệ "
            f"trong chunk {chunk_number}."
        )

    entity_df = (
        chunk[
            [
                "User",
                "Card",
            ]
        ]
        .copy()
    )

    entity_df[
        "Timestamp"
    ] = timestamp


    # --------------------------------------------------------
    # User history
    # --------------------------------------------------------

    user_chunk_summary = (
        aggregate_entity_chunk(
            entity_df,
            ["User"],
        )
    )

    update_entity_state(
        user_state,
        user_chunk_summary,
        ["User"],
    )


    # --------------------------------------------------------
    # User + Card history
    # --------------------------------------------------------

    card_chunk_summary = (
        aggregate_entity_chunk(
            entity_df,
            [
                "User",
                "Card",
            ],
        )
    )

    update_entity_state(
        card_state,
        card_chunk_summary,
        [
            "User",
            "Card",
        ],
    )


    # --------------------------------------------------------
    # Past <= 2018
    # --------------------------------------------------------

    past_mask = (
        chunk["Year"]
        <= 2018
    )

    if past_mask.any():

        past_users.update(
            int(value)
            for value
            in chunk.loc[
                past_mask,
                "User",
            ]
            .unique()
        )

        past_card_pairs = (
            chunk.loc[
                past_mask,
                [
                    "User",
                    "Card",
                ],
            ]
            .drop_duplicates()
        )

        past_cards.update(
            (
                int(user),
                int(card),
            )
            for user, card
            in past_card_pairs
            .itertuples(
                index=False,
                name=None,
            )
        )


    # --------------------------------------------------------
    # Future = 2019
    # --------------------------------------------------------

    year_2019_mask = (
        chunk["Year"]
        == 2019
    )

    if year_2019_mask.any():

        chunk_2019 = (
            chunk.loc[
                year_2019_mask,
                [
                    "User",
                    "Card",
                ],
            ]
        )

        users_2019.update(
            int(value)
            for value
            in chunk_2019[
                "User"
            ].unique()
        )

        cards_2019.update(
            (
                int(user),
                int(card),
            )
            for user, card
            in chunk_2019[
                [
                    "User",
                    "Card",
                ]
            ]
            .drop_duplicates()
            .itertuples(
                index=False,
                name=None,
            )
        )


        transactions_2019_by_user.update(
            {
                int(key):
                    int(value)

                for key, value
                in chunk_2019[
                    "User"
                ]
                .value_counts()
                .items()
            }
        )


        card_count_2019 = (
            chunk_2019
            .groupby(
                [
                    "User",
                    "Card",
                ]
            )
            .size()
        )

        transactions_2019_by_card.update(
            {
                (
                    int(key[0]),
                    int(key[1]),
                ):
                    int(value)

                for key, value
                in card_count_2019.items()
            }
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


print("\nHoàn tất full scan M2.7.")

print(
    "Tổng số dòng:",
    f"{total_rows:,}",
)

print(
    "Timestamp parse failures:",
    timestamp_parse_failures,
)

print(
    "Unique User:",
    len(
        user_state
    ),
)

print(
    "Unique User+Card:",
    len(
        card_state
    ),
)
```

    Đã xử lý 10 chunk - tổng số dòng: 5,000,000
    Đã xử lý 20 chunk - tổng số dòng: 10,000,000
    Đã xử lý 30 chunk - tổng số dòng: 15,000,000
    Đã xử lý 40 chunk - tổng số dòng: 20,000,000
    Đã xử lý 49 chunk - tổng số dòng: 24,386,900
    
    Hoàn tất full scan M2.7.
    Tổng số dòng: 24,386,900
    Timestamp parse failures: 0
    Unique User: 2000
    Unique User+Card: 6139


### Nhận xét M2.7.1

Toàn bộ dataset đã được quét thành công qua `49 chunk`, với tổng cộng `24,386,900 transaction`. <br>

Không có lỗi parse Timestamp: <br>
`Timestamp parse failures = 0` <br>

Full scan xác nhận: <br>
`Unique User = 2,000` <br>
`Unique User+Card = 6,139` <br>

Các con số này nhất quán với những mốc đã được audit trước đó. <br>

Notebook đã xây được state ở hai cấp entity: <br>
`User` <br>
và <br>
`User + Card` <br>

Mỗi entity được lưu transaction count, first timestamp, last timestamp và số transaction tại first timestamp. <br>

Cách làm này phù hợp với mục tiêu M2.7 vì cho phép đánh giá history depth và strict cold-start mà không cần dựa vào raw row order. <br>

`Kết luận: PASS`

## M2.7.2 — Độ sâu lịch sử ở cấp User

### Câu hỏi

Một User điển hình có bao nhiêu transaction? <br>

Lịch sử của User kéo dài trong bao lâu? <br>

Một User thường có bao nhiêu Card? <br>

### Phương pháp

Chuyển state aggregate thành bảng User-level. <br>

Các thống kê lifetime trong bảng này chỉ dùng để mô tả cấu trúc dataset. <br>

Không được gắn lifetime transaction count hoặc lifetime span vào từng transaction làm model feature. <br>

Thay vì tự đặt ngưỡng tùy ý, distribution được mô tả bằng các quantile. <br>


```python
user_history_df = pd.DataFrame(
    [
        {
            "User":
                user,

            "transaction_count":
                stats[
                    "transaction_count"
                ],

            "first_timestamp":
                stats[
                    "first_timestamp"
                ],

            "last_timestamp":
                stats[
                    "last_timestamp"
                ],

            "first_timestamp_transaction_count":
                stats[
                    "first_timestamp_transaction_count"
                ],
        }

        for user, stats
        in user_state.items()
    ]
)


user_history_df[
    "active_span_days"
] = (
    (
        user_history_df[
            "last_timestamp"
        ]
        - user_history_df[
            "first_timestamp"
        ]
    )
    .dt.total_seconds()
    / 86_400
)


cards_per_user = (
    pd.Series(
        [
            user
            for user, card
            in card_state.keys()
        ]
    )
    .value_counts()
    .rename(
        "card_count"
    )
)


user_history_df[
    "card_count"
] = (
    user_history_df[
        "User"
    ]
    .map(
        cards_per_user
    )
    .astype(int)
)


display(
    user_history_df.head()
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
      <th>User</th>
      <th>transaction_count</th>
      <th>first_timestamp</th>
      <th>last_timestamp</th>
      <th>first_timestamp_transaction_count</th>
      <th>active_span_days</th>
      <th>card_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>19963</td>
      <td>2002-09-01 06:21:00</td>
      <td>2020-02-28 07:36:00</td>
      <td>1</td>
      <td>6389.052083</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>8919</td>
      <td>2003-07-01 06:45:00</td>
      <td>2020-02-27 11:23:00</td>
      <td>1</td>
      <td>6085.193056</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>41978</td>
      <td>2002-03-01 06:59:00</td>
      <td>2020-02-28 23:49:00</td>
      <td>1</td>
      <td>6573.701389</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>10117</td>
      <td>2007-02-01 13:25:00</td>
      <td>2020-02-28 22:58:00</td>
      <td>1</td>
      <td>4775.397917</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>18542</td>
      <td>1999-11-26 15:03:00</td>
      <td>2020-02-28 20:29:00</td>
      <td>1</td>
      <td>7399.226389</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



```python
HISTORY_QUANTILES = [
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


def build_quantile_table(
    series,
    metric_name,
):
    result = (
        series
        .quantile(
            HISTORY_QUANTILES
        )
        .rename(
            metric_name
        )
        .reset_index()
        .rename(
            columns={
                "index":
                    "quantile"
            }
        )
    )

    return result
```


```python
print(
    "Phân bố transaction count theo User:"
)

display(
    build_quantile_table(
        user_history_df[
            "transaction_count"
        ],
        "transaction_count",
    )
)


print(
    "\nPhân bố active span theo User:"
)

display(
    build_quantile_table(
        user_history_df[
            "active_span_days"
        ],
        "active_span_days",
    )
)


print(
    "\nPhân bố số Card trên mỗi User:"
)

display(
    build_quantile_table(
        user_history_df[
            "card_count"
        ],
        "card_count",
    )
)
```

    Phân bố transaction count theo User:



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
      <th>transaction_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.00</td>
      <td>15.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.01</td>
      <td>42.00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.05</td>
      <td>76.95</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.25</td>
      <td>4008.00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.50</td>
      <td>10860.50</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.75</td>
      <td>17425.75</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.95</td>
      <td>31473.10</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.99</td>
      <td>47095.89</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1.00</td>
      <td>82355.00</td>
    </tr>
  </tbody>
</table>
</div>


    
    Phân bố active span theo User:



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
      <th>active_span_days</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.00</td>
      <td>24.803472</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.01</td>
      <td>26.656625</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.05</td>
      <td>27.586007</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.25</td>
      <td>2123.690278</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.50</td>
      <td>4682.021528</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.75</td>
      <td>6024.473785</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.95</td>
      <td>7819.351701</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.99</td>
      <td>9022.700208</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1.00</td>
      <td>10649.286806</td>
    </tr>
  </tbody>
</table>
</div>


    
    Phân bố số Card trên mỗi User:



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
      <th>card_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.00</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.01</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.05</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.25</td>
      <td>2.00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.50</td>
      <td>3.00</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.75</td>
      <td>4.00</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.95</td>
      <td>6.00</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.99</td>
      <td>7.01</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1.00</td>
      <td>9.00</td>
    </tr>
  </tbody>
</table>
</div>


### Nhận xét M2.7.2

Độ sâu lịch sử giữa các User rất không đồng đều nhưng nhìn chung khá lớn. <br>

Số transaction trên mỗi User có distribution: <br>
`min = 15` <br>
`P25 = 4,008` <br>
`median = 10,860.5` <br>
`P75 = 17,425.75` <br>
`P95 ≈ 31,473` <br>
`max = 82,355` <br>

Như vậy một User điển hình có hàng nghìn đến hàng chục nghìn transaction trong dataset. <br>

Ngay tại median, một User đã có hơn `10 nghìn transaction`. <br>

Active span của User cũng thường rất dài: <br>
`P25 ≈ 2,123.7 ngày` <br>
`median ≈ 4,682 ngày` <br>
`P75 ≈ 6,024.5 ngày` <br>
`P95 ≈ 7,819.4 ngày` <br>
`max ≈ 10,649.3 ngày` <br>

Median khoảng `4,682 ngày`, tương đương gần `12.8 năm`. <br>

Điều này cho thấy phần lớn User được quan sát qua một khoảng thời gian dài, tạo cơ sở tốt cho các feature mô tả historical behavior. <br>

Tuy nhiên distribution không đồng nhất. <br>

`P5 active span chỉ khoảng 27.6 ngày`, trong khi `P25 đã trên 2,100 ngày`. <br>

Khoảng nhảy lớn này cho thấy tồn tại một nhóm User có history rất ngắn bên cạnh nhóm có history nhiều năm. <br>

Do đó không nên giả định mọi User đều có cùng mức historical context. <br>

Số Card trên mỗi User tương đối nhỏ: <br>
`median = 3` <br>
`P75 = 4` <br>
`P95 = 6` <br>
`max = 9`. <br>

Điều này cho thấy user-level history có thể tổng hợp thông tin từ nhiều Card nhưng số lượng Card trên một User vẫn ở quy mô quản lý được. <br>

`Kết luận: dataset có user-level history rất sâu đối với phần lớn User, nhưng tồn tại một nhóm User có lịch sử ngắn. User-level behavioral feature có cơ sở về mặt dữ liệu, song availability thực tế tại từng prediction point vẫn phải tuân thủ causal history.`

## M2.7.3 — Độ sâu lịch sử ở cấp User + Card

### Câu hỏi

Một Card điển hình có bao nhiêu transaction? <br>

Card được quan sát trong khoảng thời gian bao lâu? <br>

Distribution của history depth có đủ sâu để behavioral feature có ý nghĩa hay không? <br>

### Phương pháp

Entity Card được định nghĩa bằng cặp: <br>
`User + Card` <br>

Không sử dụng `Card` một mình vì Card là identifier trong phạm vi User. <br>

Tương tự User analysis, lifetime statistics chỉ phục vụ EDA feasibility. <br>


```python
card_history_df = pd.DataFrame(
    [
        {
            "User":
                key[0],

            "Card":
                key[1],

            "transaction_count":
                stats[
                    "transaction_count"
                ],

            "first_timestamp":
                stats[
                    "first_timestamp"
                ],

            "last_timestamp":
                stats[
                    "last_timestamp"
                ],

            "first_timestamp_transaction_count":
                stats[
                    "first_timestamp_transaction_count"
                ],
        }

        for key, stats
        in card_state.items()
    ]
)


card_history_df[
    "active_span_days"
] = (
    (
        card_history_df[
            "last_timestamp"
        ]
        - card_history_df[
            "first_timestamp"
        ]
    )
    .dt.total_seconds()
    / 86_400
)


card_history_df[
    "has_later_history"
] = (
    card_history_df[
        "transaction_count"
    ]
    > card_history_df[
        "first_timestamp_transaction_count"
    ]
)


display(
    card_history_df.head()
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
      <th>User</th>
      <th>Card</th>
      <th>transaction_count</th>
      <th>first_timestamp</th>
      <th>last_timestamp</th>
      <th>first_timestamp_transaction_count</th>
      <th>active_span_days</th>
      <th>has_later_history</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>5011</td>
      <td>2002-09-01 06:21:00</td>
      <td>2020-02-27 12:41:00</td>
      <td>1</td>
      <td>6388.263889</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>1203</td>
      <td>2014-04-09 13:53:00</td>
      <td>2020-02-26 20:46:00</td>
      <td>1</td>
      <td>2149.286806</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>2</td>
      <td>4332</td>
      <td>2003-07-02 06:05:00</td>
      <td>2020-02-28 07:36:00</td>
      <td>1</td>
      <td>6085.063194</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>3</td>
      <td>9391</td>
      <td>2003-01-01 06:25:00</td>
      <td>2020-02-28 06:53:00</td>
      <td>1</td>
      <td>6267.019444</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>4</td>
      <td>26</td>
      <td>2008-09-03 14:07:00</td>
      <td>2009-03-31 13:17:00</td>
      <td>1</td>
      <td>208.965278</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>



```python
print(
    "Phân bố transaction count theo User+Card:"
)

display(
    build_quantile_table(
        card_history_df[
            "transaction_count"
        ],
        "transaction_count",
    )
)


print(
    "\nPhân bố active span theo User+Card:"
)

display(
    build_quantile_table(
        card_history_df[
            "active_span_days"
        ],
        "active_span_days",
    )
)


print(
    "\nSố transaction tại timestamp đầu tiên "
    "của mỗi User+Card:"
)

display(
    build_quantile_table(
        card_history_df[
            "first_timestamp_transaction_count"
        ],
        "first_timestamp_transaction_count",
    )
)
```

    Phân bố transaction count theo User+Card:



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
      <th>transaction_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.00</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.01</td>
      <td>11.00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.05</td>
      <td>26.00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.25</td>
      <td>461.00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.50</td>
      <td>2602.00</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.75</td>
      <td>5518.50</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.95</td>
      <td>13015.60</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.99</td>
      <td>21962.92</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1.00</td>
      <td>70008.00</td>
    </tr>
  </tbody>
</table>
</div>


    
    Phân bố active span theo User+Card:



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
      <th>active_span_days</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.01</td>
      <td>22.984639</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.05</td>
      <td>26.684931</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.25</td>
      <td>575.892708</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.50</td>
      <td>3188.861111</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.75</td>
      <td>4568.652778</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.95</td>
      <td>6573.122361</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.99</td>
      <td>8048.788889</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1.00</td>
      <td>10649.286806</td>
    </tr>
  </tbody>
</table>
</div>


    
    Số transaction tại timestamp đầu tiên của mỗi User+Card:



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
      <th>first_timestamp_transaction_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.01</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.05</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.25</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.50</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.75</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.95</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.99</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1.00</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
</div>


### Nhận xét M2.7.3

History ở cấp `User + Card` cũng rất sâu đối với phần lớn entity. <br>

Số transaction trên mỗi Card có distribution: <br>
`min = 1` <br>
`P1 = 11` <br>
`P5 = 26` <br>
`P25 = 461` <br>
`median = 2,602` <br>
`P75 ≈ 5,518.5` <br>
`P95 ≈ 13,015.6` <br>
`P99 ≈ 21,962.9` <br>
`max = 70,008` <br>

Median `2,602 transaction/Card` cho thấy một Card điển hình có lịch sử giao dịch rất lớn. <br>

Khoảng 75% Card có ít nhất xấp xỉ `461 transaction`, và khoảng 95% có ít nhất xấp xỉ `26 transaction`. <br>

Do đó dataset không phải cấu trúc mà phần lớn Card chỉ xuất hiện một vài lần. <br>

Active span ở cấp Card cũng dài: <br>
`P25 ≈ 575.9 ngày` <br>
`median ≈ 3,188.9 ngày` <br>
`P75 ≈ 4,568.7 ngày` <br>
`P95 ≈ 6,573.1 ngày` <br>
`max ≈ 10,649.3 ngày` <br>

Median khoảng `3,189 ngày`, tương đương gần `8.7 năm`. <br>

Điều này cung cấp nền tảng rất tốt cho việc nghiên cứu historical / behavioral feature ở card level. <br>

Tương tự User, history depth vẫn không đồng nhất. <br>

`P5 active span chỉ khoảng 26.7 ngày`, trong khi `P25 khoảng 576 ngày`. <br>

Vì vậy một nhóm nhỏ Card có lịch sử ngắn đáng kể so với phần lớn Card. <br>

Số transaction tại timestamp đầu tiên gần như luôn bằng 1: <br>
từ `P0 đến P99 đều bằng 1`, và giá trị lớn nhất chỉ là `3`. <br>

Điều này cho thấy timestamp tie tại thời điểm Card xuất hiện lần đầu tồn tại nhưng rất hiếm. <br>

`Kết luận: User+Card history có độ sâu và active span đủ lớn để behavioral feature trở thành một hướng khả thi. Tuy nhiên history availability không đồng đều giữa các Card và cần được kiểm chứng trực tiếp ở các rolling window trong M2.8.`

## M2.7.4 — Strict transaction-level cold-start

### Câu hỏi

Bao nhiêu transaction xảy ra khi User hoặc Card chưa có bất kỳ transaction nào có timestamp nhỏ hơn? <br>

Phần lớn transaction có historical context hay không? <br>

### Định nghĩa

Một transaction thuộc strict cold-start của entity nếu: <br>
`timestamp(transaction) = first_timestamp(entity)` <br>

Do historical feature bắt buộc chỉ dùng: <br>
`timestamp(history) < timestamp(current)` <br>

nên tất cả transaction tại timestamp đầu tiên đều không có lịch sử trước đó. <br>

Định nghĩa này xử lý đúng cả trường hợp nhiều transaction cùng xảy ra trong phút đầu tiên. <br>


```python
user_strict_cold_start_transactions = int(
    user_history_df[
        "first_timestamp_transaction_count"
    ].sum()
)


card_strict_cold_start_transactions = int(
    card_history_df[
        "first_timestamp_transaction_count"
    ].sum()
)


users_with_later_history = int(
    (
        user_history_df[
            "transaction_count"
        ]
        > user_history_df[
            "first_timestamp_transaction_count"
        ]
    ).sum()
)


cards_with_later_history = int(
    (
        card_history_df[
            "transaction_count"
        ]
        > card_history_df[
            "first_timestamp_transaction_count"
        ]
    ).sum()
)


strict_cold_start_df = pd.DataFrame(
    [
        {
            "entity_level":
                "User",

            "entity_count":
                len(
                    user_history_df
                ),

            "entities_with_later_history":
                users_with_later_history,

            "entities_with_later_history_pct":
                users_with_later_history
                / len(
                    user_history_df
                )
                * 100,

            "strict_cold_start_transactions":
                user_strict_cold_start_transactions,

            "strict_cold_start_rate_pct":
                user_strict_cold_start_transactions
                / total_rows
                * 100,

            "transactions_with_prior_history":
                total_rows
                - user_strict_cold_start_transactions,

            "transactions_with_prior_history_pct":
                (
                    total_rows
                    - user_strict_cold_start_transactions
                )
                / total_rows
                * 100,
        },
        {
            "entity_level":
                "User+Card",

            "entity_count":
                len(
                    card_history_df
                ),

            "entities_with_later_history":
                cards_with_later_history,

            "entities_with_later_history_pct":
                cards_with_later_history
                / len(
                    card_history_df
                )
                * 100,

            "strict_cold_start_transactions":
                card_strict_cold_start_transactions,

            "strict_cold_start_rate_pct":
                card_strict_cold_start_transactions
                / total_rows
                * 100,

            "transactions_with_prior_history":
                total_rows
                - card_strict_cold_start_transactions,

            "transactions_with_prior_history_pct":
                (
                    total_rows
                    - card_strict_cold_start_transactions
                )
                / total_rows
                * 100,
        },
    ]
)


display(
    strict_cold_start_df
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
      <th>entity_level</th>
      <th>entity_count</th>
      <th>entities_with_later_history</th>
      <th>entities_with_later_history_pct</th>
      <th>strict_cold_start_transactions</th>
      <th>strict_cold_start_rate_pct</th>
      <th>transactions_with_prior_history</th>
      <th>transactions_with_prior_history_pct</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>User</td>
      <td>2000</td>
      <td>2000</td>
      <td>100.000000</td>
      <td>2008</td>
      <td>0.008234</td>
      <td>24384892</td>
      <td>99.991766</td>
    </tr>
    <tr>
      <th>1</th>
      <td>User+Card</td>
      <td>6139</td>
      <td>6137</td>
      <td>99.967421</td>
      <td>6159</td>
      <td>0.025255</td>
      <td>24380741</td>
      <td>99.974745</td>
    </tr>
  </tbody>
</table>
</div>



```python
first_timestamp_tie_summary = pd.Series(
    {
        "Cards có >1 transaction tại first timestamp":
            int(
                (
                    card_history_df[
                        "first_timestamp_transaction_count"
                    ]
                    > 1
                ).sum()
            ),

        "Tỷ lệ Card có >1 transaction tại first timestamp (%)":
            (
                card_history_df[
                    "first_timestamp_transaction_count"
                ]
                .gt(1)
                .mean()
                * 100
            ),

        "Max transaction cùng first timestamp trên một Card":
            int(
                card_history_df[
                    "first_timestamp_transaction_count"
                ]
                .max()
            ),
    }
)


display(
    first_timestamp_tie_summary
)
```


    Cards có >1 transaction tại first timestamp             19.000000
    Tỷ lệ Card có >1 transaction tại first timestamp (%)     0.309497
    Max transaction cùng first timestamp trên một Card       3.000000
    dtype: float64


### Nhận xét M2.7.4

Strict transaction-level cold-start chiếm tỷ lệ rất nhỏ trên toàn dataset. <br>

Ở cấp `User`: <br>
`2,008 transaction` xảy ra tại first timestamp của User <br>
→ `0.008234%` toàn dataset. <br>

Khoảng `99.9918%` transaction có ít nhất một User-level timestamp xảy ra trước timestamp hiện tại. <br>

Tất cả `2,000 / 2,000 User` đều có transaction tại một timestamp muộn hơn first timestamp. <br>

Ở cấp `User + Card`: <br>
`6,159 transaction` thuộc strict cold-start <br>
→ `0.025255%` toàn dataset. <br>

Khoảng `99.9747%` transaction có ít nhất một Card-level transaction với timestamp trước timestamp hiện tại. <br>

`6,137 / 6,139 Card`, tương đương khoảng `99.967%`, có history tiếp tục sau first timestamp. <br>

Chỉ có `2 Card` không có transaction tại timestamp muộn hơn first timestamp. <br>

Strict cold-start transaction count lớn hơn số Card `6,139` vì một số Card có nhiều transaction cùng timestamp đầu tiên. <br>

Tuy nhiên timestamp tie rất hiếm: <br>
chỉ `19 Card`, tương đương khoảng `0.31%`, có hơn một transaction tại first timestamp; <br>
giá trị lớn nhất chỉ là `3 transaction` cùng first timestamp. <br>

Do đó việc xác định cold-start theo điều kiện strict `history timestamp < current timestamp` thay vì raw row order là cần thiết về nguyên tắc, nhưng timestamp tie không phải vấn đề quy mô lớn trong dataset này. <br>

Kết quả này cho thấy phần áp đảo transaction có ít nhất một historical observation ở cấp Card trước prediction point. <br>

Tuy nhiên `có ít nhất một prior transaction` chưa đồng nghĩa có đủ history trong các cửa sổ ngắn như `10 phút`, `1 giờ` hoặc `30 ngày`. <br>

`Kết luận: strict card-level cold-start cực hiếm trên toàn dataset. Historical feature có coverage tiềm năng rất cao, nhưng rolling-window coverage cụ thể phải được kiểm tra trong M2.8.`

## M2.7.5 — Entity overlap và cold-start trong evaluation period 2019

### Vì sao kiểm tra này tồn tại?

Strict transaction-level cold-start cho biết một transaction có lịch sử trước timestamp hiện tại hay không. <br>

Nhưng khi thiết kế temporal evaluation, còn một câu hỏi khác: <br>
`Entity xuất hiện trong future test period đã từng xuất hiện trong training past hay chưa?` <br>

Đây là evaluation cold-start. <br>

### Thiết kế stress test

`Past: Year <= 2018` <br>
`Future: Year = 2019` <br>

Đối với User và User+Card, đo: <br>
- số entity xuất hiện năm 2019; <br>
- số entity đã từng tồn tại trong past; <br>
- số entity hoàn toàn mới; <br>
- số transaction 2019 thuộc entity mới; <br>
- tỷ lệ transaction cold-start. <br>


```python
new_users_2019 = (
    users_2019
    - past_users
)


new_cards_2019 = (
    cards_2019
    - past_cards
)


seen_users_2019 = (
    users_2019
    & past_users
)


seen_cards_2019 = (
    cards_2019
    & past_cards
)


total_transactions_2019 = int(
    sum(
        transactions_2019_by_user
        .values()
    )
)


new_user_transactions_2019 = int(
    sum(
        transactions_2019_by_user[
            user
        ]

        for user
        in new_users_2019
    )
)


new_card_transactions_2019 = int(
    sum(
        transactions_2019_by_card[
            card_key
        ]

        for card_key
        in new_cards_2019
    )
)


entity_overlap_2019_df = pd.DataFrame(
    [
        {
            "entity_level":
                "User",

            "entities_in_2019":
                len(
                    users_2019
                ),

            "seen_in_past":
                len(
                    seen_users_2019
                ),

            "new_entities":
                len(
                    new_users_2019
                ),

            "entity_overlap_pct":
                len(
                    seen_users_2019
                )
                / len(
                    users_2019
                )
                * 100,

            "transactions_2019":
                total_transactions_2019,

            "new_entity_transactions":
                new_user_transactions_2019,

            "new_entity_transaction_rate_pct":
                new_user_transactions_2019
                / total_transactions_2019
                * 100,
        },
        {
            "entity_level":
                "User+Card",

            "entities_in_2019":
                len(
                    cards_2019
                ),

            "seen_in_past":
                len(
                    seen_cards_2019
                ),

            "new_entities":
                len(
                    new_cards_2019
                ),

            "entity_overlap_pct":
                len(
                    seen_cards_2019
                )
                / len(
                    cards_2019
                )
                * 100,

            "transactions_2019":
                total_transactions_2019,

            "new_entity_transactions":
                new_card_transactions_2019,

            "new_entity_transaction_rate_pct":
                new_card_transactions_2019
                / total_transactions_2019
                * 100,
        },
    ]
)


display(
    entity_overlap_2019_df
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
      <th>entity_level</th>
      <th>entities_in_2019</th>
      <th>seen_in_past</th>
      <th>new_entities</th>
      <th>entity_overlap_pct</th>
      <th>transactions_2019</th>
      <th>new_entity_transactions</th>
      <th>new_entity_transaction_rate_pct</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>User</td>
      <td>1579</td>
      <td>1528</td>
      <td>51</td>
      <td>96.770108</td>
      <td>1723938</td>
      <td>3255</td>
      <td>0.188812</td>
    </tr>
    <tr>
      <th>1</th>
      <td>User+Card</td>
      <td>4029</td>
      <td>3883</td>
      <td>146</td>
      <td>96.376272</td>
      <td>1723938</td>
      <td>18544</td>
      <td>1.075677</td>
    </tr>
  </tbody>
</table>
</div>


## M2.7.6 — Phân rã new-card cold-start trong 2019

### Câu hỏi

Các Card mới trong 2019 chủ yếu thuộc: <br>
- User đã có lịch sử trước 2019; <br>
hay <br>
- User cũng hoàn toàn mới? <br>

### Vì sao điều này quan trọng?

`New Card + Existing User` <br>
→ không có card-level history; <br>
→ nhưng vẫn có thể tồn tại user-level history. <br>

`New Card + New User` <br>
→ không có cả card-level lẫn user-level history từ past. <br>

Hai trường hợp có mức cold-start khác nhau và có thể cần chiến lược feature khác nhau sau này. <br>


```python
new_cards_existing_user_2019 = {
    card_key

    for card_key
    in new_cards_2019

    if (
        card_key[0]
        in past_users
    )
}


new_cards_new_user_2019 = (
    new_cards_2019
    - new_cards_existing_user_2019
)


new_card_existing_user_transactions = int(
    sum(
        transactions_2019_by_card[
            card_key
        ]

        for card_key
        in new_cards_existing_user_2019
    )
)


new_card_new_user_transactions = int(
    sum(
        transactions_2019_by_card[
            card_key
        ]

        for card_key
        in new_cards_new_user_2019
    )
)


new_card_breakdown_df = pd.DataFrame(
    [
        {
            "segment":
                "New Card + Existing User",

            "new_card_count":
                len(
                    new_cards_existing_user_2019
                ),

            "transaction_count":
                new_card_existing_user_transactions,

            "share_of_2019_transactions_pct":
                new_card_existing_user_transactions
                / total_transactions_2019
                * 100,
        },
        {
            "segment":
                "New Card + New User",

            "new_card_count":
                len(
                    new_cards_new_user_2019
                ),

            "transaction_count":
                new_card_new_user_transactions,

            "share_of_2019_transactions_pct":
                new_card_new_user_transactions
                / total_transactions_2019
                * 100,
        },
    ]
)


display(
    new_card_breakdown_df
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
      <th>segment</th>
      <th>new_card_count</th>
      <th>transaction_count</th>
      <th>share_of_2019_transactions_pct</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>New Card + Existing User</td>
      <td>93</td>
      <td>15289</td>
      <td>0.886865</td>
    </tr>
    <tr>
      <th>1</th>
      <td>New Card + New User</td>
      <td>53</td>
      <td>3255</td>
      <td>0.188812</td>
    </tr>
  </tbody>
</table>
</div>



```python
print(
    "New-card transaction decomposition khớp:",
    (
        new_card_existing_user_transactions
        + new_card_new_user_transactions
    )
    == new_card_transactions_2019,
)
```

    New-card transaction decomposition khớp: True


### Nhận xét M2.7.5–M2.7.6

Trong năm 2019 có: <br>
`1,579 User` <br>
`4,029 User+Card` <br>
`1,723,938 transaction`. <br>

Ở cấp User: <br>
`1,528 / 1,579 User` đã xuất hiện trong past `<= 2018` <br>
→ entity overlap khoảng `96.77%`. <br>

Chỉ có `51 User` hoàn toàn mới trong 2019. <br>

Các User mới tạo ra `3,255 transaction`, chỉ chiếm khoảng `0.1888%` transaction năm 2019. <br>

Ở cấp User+Card: <br>
`3,883 / 4,029 Card` đã xuất hiện trong past <br>
→ entity overlap khoảng `96.38%`. <br>

Có `146 Card` mới hoàn toàn trong 2019. <br>

Các Card mới tạo ra `18,544 transaction`, chiếm khoảng `1.0757%` transaction năm 2019. <br>

Như vậy cold-start nhìn theo số entity khoảng 3–4%, nhưng nhìn theo transaction volume lại nhỏ hơn nhiều. <br>

Evaluation năm 2019 vì thế chủ yếu đánh giá transaction của những User/Card đã tồn tại trong historical period. <br>

Phân rã 146 Card mới cho thấy: <br>
`93 Card mới` thuộc `User đã tồn tại`; <br>
`53 Card mới` thuộc `User cũng mới`. <br>

Nhóm `New Card + Existing User` tạo ra `15,289 transaction`, tương đương khoảng `0.8869%` toàn bộ transaction 2019. <br>

Nhóm `New Card + New User` tạo ra `3,255 transaction`, tương đương khoảng `0.1888%`. <br>

Như vậy khoảng `82.45%` transaction thuộc new Card vẫn đến từ User đã có lịch sử trước 2019. <br>

Điều này rất quan trọng về mặt feature engineering: <br>
khi card-level history chưa tồn tại, user-level history vẫn có thể khả dụng cho phần lớn new-card transaction. <br>

Toàn bộ `3,255 new-user transaction` cũng nằm trong nhóm `New Card + New User`, phù hợp logic cold-start. <br>

Phép decomposition trả về `True`, xác nhận hai segment new-card cộng lại đúng bằng toàn bộ `18,544 new-card transaction`. <br>

`Kết luận: evaluation năm 2019 có entity overlap rất cao và cold-start transaction volume thấp. Model evaluation sau này chủ yếu đo khả năng screening transaction mới của existing User/Card; khả năng generalize tới User/Card hoàn toàn mới chỉ được kiểm tra trên một phần rất nhỏ dữ liệu.`

## M2.7.7 — Tóm tắt bằng chứng về khả năng xây historical feature

### Câu hỏi

Các bằng chứng cấu trúc hiện tại có đủ để tiếp tục prototype behavioral feature ở M2.8 hay không? <br>

### Phương pháp

Tổng hợp một số thống kê trực tiếp từ các bảng trước: <br>
- số User / Card; <br>
- median transaction count; <br>
- median active span; <br>
- tỷ lệ transaction có prior Card history; <br>
- tỷ lệ 2019 transaction thuộc new User / new Card. <br>

Bảng này chỉ là `feasibility snapshot`, không phải feature matrix. <br>


```python
card_median_transaction_count = float(
    card_history_df[
        "transaction_count"
    ].median()
)


card_median_active_span_days = float(
    card_history_df[
        "active_span_days"
    ].median()
)


user_median_transaction_count = float(
    user_history_df[
        "transaction_count"
    ].median()
)


card_transactions_with_prior_history_pct = float(
    (
        total_rows
        - card_strict_cold_start_transactions
    )
    / total_rows
    * 100
)


history_feasibility_summary = pd.Series(
    {
        "unique_users":
            len(
                user_history_df
            ),

        "unique_user_card_pairs":
            len(
                card_history_df
            ),

        "median_transactions_per_user":
            user_median_transaction_count,

        "median_transactions_per_card":
            card_median_transaction_count,

        "median_card_active_span_days":
            card_median_active_span_days,

        "transactions_with_prior_card_history_pct":
            card_transactions_with_prior_history_pct,

        "2019_new_user_transaction_rate_pct":
            (
                new_user_transactions_2019
                / total_transactions_2019
                * 100
            ),

        "2019_new_card_transaction_rate_pct":
            (
                new_card_transactions_2019
                / total_transactions_2019
                * 100
            ),
    }
)


display(
    history_feasibility_summary
)
```


    unique_users                                 2000.000000
    unique_user_card_pairs                       6139.000000
    median_transactions_per_user                10860.500000
    median_transactions_per_card                 2602.000000
    median_card_active_span_days                 3188.861111
    transactions_with_prior_card_history_pct       99.974745
    2019_new_user_transaction_rate_pct              0.188812
    2019_new_card_transaction_rate_pct              1.075677
    dtype: float64


### Nhận xét M2.7.7

Feasibility snapshot tổng hợp các bằng chứng chính: <br>

`Unique User = 2,000` <br>
`Unique User+Card = 6,139` <br>
`Median transaction/User = 10,860.5` <br>
`Median transaction/Card = 2,602` <br>
`Median Card active span ≈ 3,188.9 ngày` <br>
`Transaction có prior Card history ≈ 99.9747%` <br>
`2019 new-user transaction rate ≈ 0.1888%` <br>
`2019 new-card transaction rate ≈ 1.0757%` <br>

Các con số này tạo bằng chứng mạnh rằng dataset có đủ repeated entity history để nghiên cứu behavioral feature ở cả user level và card level. <br>

Đặc biệt, Card điển hình có hàng nghìn transaction và được quan sát trong nhiều năm, trong khi phần áp đảo transaction không thuộc strict cold-start. <br>

Vì vậy các candidate như: <br>
`time_since_previous_transaction` <br>
`transactions_last_10m` <br>
`transactions_last_1h` <br>
`previous_amount_mean` <br>
`is_new_merchant` <br>
có cơ sở dữ liệu để được prototype. <br>

Tuy nhiên M2.7 mới chứng minh `history tồn tại`, chưa chứng minh `history trong một rolling window cụ thể đủ dày`. <br>

Một Card có 2,000 transaction trong 10 năm vẫn có thể không có transaction nào trong 10 phút trước transaction hiện tại. <br>

Do đó feasibility của từng candidate feature phải được kiểm tra bằng strict causal computation ở M2.8. <br>

`Kết luận: behavioral feature engineering là khả thi về mặt cấu trúc entity-history và đáng tiếp tục sang M2.8.`

## M2.7.8 — Đối chiếu kết quả entity-history với Milestone 1

### Câu hỏi

Các thống kê entity và cold-start được tính độc lập trong M2.7 có nhất quán với Risk Audit của M1 hay không? <br>

### Các mốc đối chiếu

`Total rows = 24,386,900` <br>
`Unique Users = 2,000` <br>
`Unique User+Card = 6,139` <br>

Stress test 2019: <br>
`2019 Users = 1,579` <br>
`Users seen in past = 1,528` <br>
`2019 User+Card = 4,029` <br>
`Cards seen in past = 3,883` <br>
`New-user transactions = 3,255` <br>
`New-card transactions = 18,544` <br>


```python
m27_consistency_checks = pd.Series(
    {
        "Tổng transaction = 24,386,900":
            total_rows
            == 24_386_900,

        "Timestamp parse failures = 0":
            timestamp_parse_failures
            == 0,

        "Unique Users = 2,000":
            len(
                user_history_df
            )
            == 2_000,

        "Unique User+Card = 6,139":
            len(
                card_history_df
            )
            == 6_139,

        "2019 Users = 1,579":
            len(
                users_2019
            )
            == 1_579,

        "2019 Users seen in past = 1,528":
            len(
                seen_users_2019
            )
            == 1_528,

        "2019 User+Card = 4,029":
            len(
                cards_2019
            )
            == 4_029,

        "2019 Cards seen in past = 3,883":
            len(
                seen_cards_2019
            )
            == 3_883,

        "2019 new-user transactions = 3,255":
            new_user_transactions_2019
            == 3_255,

        "2019 new-card transactions = 18,544":
            new_card_transactions_2019
            == 18_544,
    }
)


display(
    m27_consistency_checks
)


print(
    "Tất cả phép đối chiếu đều khớp:",
    m27_consistency_checks.all(),
)
```


    Tổng transaction = 24,386,900          True
    Timestamp parse failures = 0           True
    Unique Users = 2,000                   True
    Unique User+Card = 6,139               True
    2019 Users = 1,579                     True
    2019 Users seen in past = 1,528        True
    2019 User+Card = 4,029                 True
    2019 Cards seen in past = 3,883        True
    2019 new-user transactions = 3,255     True
    2019 new-card transactions = 18,544    True
    dtype: bool


    Tất cả phép đối chiếu đều khớp: True


### Nhận xét M2.7.8

Tất cả phép đối chiếu giữa kết quả được tính độc lập trong M2.7 và các mốc của Milestone 1 đều trả về `True`. <br>

Các giá trị khớp gồm: <br>
`Tổng transaction = 24,386,900` <br>
`Timestamp parse failures = 0` <br>
`Unique Users = 2,000` <br>
`Unique User+Card = 6,139` <br>
`2019 Users = 1,579` <br>
`2019 Users seen in past = 1,528` <br>
`2019 User+Card = 4,029` <br>
`2019 Cards seen in past = 3,883` <br>
`2019 new-user transactions = 3,255` <br>
`2019 new-card transactions = 18,544` <br>

Không phát hiện bất nhất giữa entity-history analysis ở M2.7 và Risk Audit của M1. <br>

M2.7 còn mở rộng kết quả M1 bằng cách định lượng strict transaction-level cold-start, history depth, active span và phân rã new-card transaction theo existing/new User. <br>

`Kết luận: PASS`

# Tổng kết M2.7 — Phân tích entity và lịch sử User / Card

## Các phát hiện chính

Dataset có repeated entity history rất sâu ở cả cấp User và User+Card. <br>

Median User có hơn `10,860 transaction`; median Card có `2,602 transaction`. <br>

Median active span của Card khoảng `3,189 ngày`, tương đương gần `8.7 năm`. <br>

Strict transaction-level cold-start chỉ chiếm khoảng `0.0253%` ở card level. <br>

Trong stress test 2019, new-user transaction chỉ chiếm khoảng `0.1888%`, còn new-card transaction khoảng `1.0757%`. <br>

Như vậy phần lớn transaction có historical context và dataset có cấu trúc rất phù hợp để nghiên cứu behavioral features. <br>

## Kết luận về độ sâu lịch sử User

User history nhìn chung rất sâu. <br>

Median User có hơn `10 nghìn transaction` và active span gần `12.8 năm`. <br>

Số Card trên một User tương đối nhỏ, với median `3 Card`. <br>

Điều này cung cấp cơ sở cho cả user-level aggregation và fallback history khi một Card còn mới. <br>

Tuy nhiên một nhóm nhỏ User có active span rất ngắn, nên historical context không đồng đều trên toàn population. <br>

## Kết luận về độ sâu lịch sử User + Card

User+Card là entity phù hợp để xây behavioral features ở mức giao dịch. <br>

Median Card có `2,602 transaction`; P25 vẫn có khoảng `461 transaction`. <br>

Median active span gần `8.7 năm`. <br>

Chỉ một phần nhỏ Card có history rất ngắn hoặc rất ít transaction. <br>

Những con số này cho thấy card-level behavioral feature có nền tảng dữ liệu tốt. <br>

## Kết luận về strict transaction-level cold-start

Chỉ `6,159 / 24,386,900 transaction`, khoảng `0.0253%`, xảy ra tại first timestamp của Card. <br>

Khoảng `99.9747%` transaction có ít nhất một Card-level timestamp xảy ra trước timestamp hiện tại. <br>

Timestamp tie tại first timestamp chỉ xuất hiện ở `19 Card`, khoảng `0.31%`, và tối đa chỉ có 3 transaction cùng first timestamp. <br>

Do đó strict causal history có coverage tiềm năng rất cao. <br>

Tuy nhiên đây mới là bằng chứng về `có prior history`, không phải bằng chứng rằng mọi rolling window đều chứa prior transaction. <br>

## Kết luận về entity overlap trong 2019

Entity overlap giữa historical period `<= 2018` và future period `2019` rất cao. <br>

Khoảng `96.77% User` và `96.38% User+Card` xuất hiện trong 2019 đã từng tồn tại trong past. <br>

Khi xét theo transaction volume, cold-start còn nhỏ hơn: <br>
`new-user transaction ≈ 0.1888%` <br>
`new-card transaction ≈ 1.0757%`. <br>

Do đó evaluation 2019 chủ yếu phản ánh performance trên existing entities có historical context. <br>

## Kết luận về new-user / new-card limitation

Cold-start hoàn toàn không được đại diện mạnh trong evaluation 2019. <br>

Trong `18,544 new-card transaction`, có `15,289`, khoảng `82.45%`, thuộc User đã tồn tại trong past. <br>

Do đó phần lớn trường hợp không có card history vẫn có khả năng sử dụng user-level history. <br>

Chỉ `3,255 transaction`, khoảng `0.1888%` năm 2019, thuộc cả new User và new Card. <br>

Vì vậy khả năng generalize tới entity hoàn toàn mới sẽ không được final evaluation kiểm chứng mạnh nếu sử dụng cấu trúc holdout tương tự. <br>

Đây là một evaluation limitation cần được giữ trong báo cáo. <br>

## Kết luận về khả năng xây behavioral feature

M2.7 cung cấp bằng chứng mạnh rằng behavioral feature engineering là khả thi về mặt cấu trúc dữ liệu. <br>

Các entity có repeated history lớn, active span dài và strict cold-start transaction rate rất thấp. <br>

Do đó M2.8 có đủ cơ sở để prototype một số feature causal đại diện. <br>

Tuy nhiên M2.8 phải kiểm tra availability thực tế của từng loại history thay vì suy ra từ lifetime count. <br>

Ví dụ: <br>
`có prior Card history` <br>
không đồng nghĩa <br>
`có prior transaction trong 10 phút`. <br>

## Các vấn đề cần chuyển sang M2.8

M2.8 cần prototype một số historical feature đại diện với strict causal rule, ví dụ: <br>
`time_since_previous_transaction` <br>
`transactions_last_10m` <br>
`transactions_last_1h` <br>
`amount_sum_last_1h` hoặc một previous-history Amount feature <br>
`is_new_merchant` hoặc `is_new_mcc` <br>

M2.8 cần đo tỷ lệ transaction mà từng feature thực sự có historical support. <br>

M2.8 cũng cần kiểm tra computational feasibility của việc tính rolling history trên quy mô dataset lớn. <br>

Quan trọng nhất, mọi feature cho transaction T chỉ được dùng transaction có timestamp nhỏ hơn T. <br>

Transaction cùng timestamp không được phép cung cấp history cho nhau nếu giữ strict causal rule. <br>

## Các vấn đề cần chuyển sang M3

M3 phải thiết kế temporal evaluation sao cho past được dùng để tạo history cho future mà không có future leakage. <br>

Entity overlap giữa train và test không phải leakage; ngược lại, đây là điều cần thiết để đánh giá behavioral feature trên existing User/Card. <br>

Tuy nhiên evaluation report phải nói rõ phần lớn test transaction thuộc existing entities và cold-start được đại diện yếu. <br>

Nếu cần đánh giá cold-start riêng, nên xem nó như một evaluation segment thay vì giả định final test hiện tại kiểm tra tốt tình huống đó. <br>

## Các vấn đề cần chuyển sang M4

M4 cần triển khai historical feature theo causal computation, không sử dụng lifetime aggregate được tính từ toàn bộ dataset. <br>

Không được lấy full-card mean, total count hoặc first/last statistics của cả tương lai rồi gắn ngược vào transaction quá khứ. <br>

Cần có chiến lược rõ cho transaction chưa có history, ví dụ indicator hoặc giá trị representation phù hợp, thay vì drop transaction cold-start. <br>

Có thể cân nhắc user-level fallback khi Card mới nhưng User đã có lịch sử, vì phần lớn new-card transaction 2019 thuộc trường hợp này. <br>

## Các limitation còn mở

History depth rất không đồng đều giữa entity. <br>

Một nhóm nhỏ User/Card có active span chỉ khoảng vài tuần dù phần lớn entity có nhiều năm lịch sử. <br>

M2.7 chưa đo mật độ transaction trong các cửa sổ thời gian ngắn. <br>

Do đó chưa thể kết luận `transactions_last_10m`, `transactions_last_1h` hoặc các rolling feature cụ thể có coverage cao. <br>

M2.7 cũng chưa kiểm tra predictive relationship của các behavioral feature vì các feature đó chưa được tạo causal. <br>

Các câu hỏi này thuộc M2.8. <br>

## Quyết định M2.7

`Decision ID: M2.7-D01` <br>
`Full-scan entity audit: PASS` <br>
`User history depth: SUFFICIENT WITH HETEROGENEITY` <br>
`User+Card history depth: SUFFICIENT` <br>
`Strict card-level cold-start: VERY LOW` <br>
`2019 entity overlap: HIGH` <br>
`2019 new-user transaction rate: LOW` <br>
`2019 new-card transaction rate: LOW` <br>
`Behavioral-feature structural feasibility: CONFIRMED` <br>
`Cold-start evaluation weakness: CONFIRMED` <br>
`Consistency with M1: PASS` <br>
`Blocking issue: NONE` <br>
`Status: PASS WITH FINDINGS` <br>

## Bước tiếp theo

`Next: M2.8 — Khảo sát behavioral-feature feasibility và temporal windows`
