# M3.3 — Audit candidate temporal train / validation / final-test split

## Mục tiêu

M3.3 không train model.

Mục tiêu là kiểm tra trên **raw artifact thật** xem các candidate temporal split có đủ điều kiện để trở thành experiment split hay không.

Các câu hỏi chính:

- Validation và final test chứa bao nhiêu transaction?
- Mỗi partition chứa bao nhiêu fraud?
- Fraud rate là bao nhiêu?
- Temporal boundary có đúng và không overlap không?
- User/Card coverage của từng partition như thế nào?
- Future partition có bao nhiêu User/Card chưa từng xuất hiện trong toàn bộ lịch sử trước partition đó?
- Future partition có bao nhiêu entity đã xuất hiện trong classifier-training rows?
- Các candidate có giữ nguyên toàn bộ positive-bearing future reservoir `2019-01 → 2019-10` hay không?
- Có candidate nào vi phạm guardrail temporal hoặc vô tình đi vào zero-fraud regime từ `2019-11` không?

## Quy tắc

M3.3 chỉ làm `split audit`.

Chưa:

- train model;
- fit preprocessing;
- resample;
- chọn threshold;
- chọn training-window winner;
- dùng final test performance;
- khóa split trước khi đọc output thật.

## Candidate boundary

Giữ cố định:

`TRAIN kết thúc trước 2019-01`

và:

`future evaluation reservoir = 2019-01 → 2019-10`.

Audit ba cách chia reservoir này:

```text
S1_4M_6M
Validation : 2019-01 → 2019-04
Final test : 2019-05 → 2019-10

S2_5M_5M
Validation : 2019-01 → 2019-05
Final test : 2019-06 → 2019-10

S3_6M_4M
Validation : 2019-01 → 2019-06
Final test : 2019-07 → 2019-10
```

Ba candidate này chỉ nhằm kiểm tra trade-off giữa độ dài validation/test và positive support.

Không candidate nào được xem là tốt nhất trước khi có output.

## Training-window family

M3.2 giữ hai họ training window để so sánh sau này:

```text
W_LONG
2015-01 → 2018-12

W_SHORT
2018-01 → 2018-12
```

M3.3 chỉ audit quy mô/entity coverage của hai training family.

Việc window nào cho model tốt hơn thuộc experiment sau; M3.3 không được chọn winner.

## Historical context

Older transaction trước modeling-window start vẫn có thể đóng vai trò causal historical warm-up.

Vì vậy notebook phân biệt:

`training rows`

và:

`all prior history before partition start`.

## Trạng thái trước khi chạy

```text
Final split: OPEN
Training-window winner: OPEN
M3.3 Decision: OPEN — chờ output thực tế
```


## 1. Thiết lập môi trường

Notebook được thiết kế để chạy độc lập từ project.

Chỉ đọc các cột cần thiết và sử dụng chunk để không giữ toàn bộ hơn 24 triệu transaction trong RAM.



```python
from pathlib import Path
from collections import Counter

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
    Path.cwd().parent.parent.parent,
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
        "Không xác định được PROJECT_ROOT "
        "chứa data/raw/ibm_tabformer/card_transaction.v1.csv"
    )

DATA_PATH = (
    PROJECT_ROOT
    / DATA_RELATIVE_PATH
)

CHUNK_SIZE = 500_000

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

#### Phân tích

Notebook đã xác định thành công `PROJECT_ROOT` của project và truy cập đúng raw artifact:

`data/raw/ibm_tabformer/card_transaction.v1.csv`.

Kết quả `File tồn tại: True` xác nhận notebook đang làm việc trực tiếp với CSV thật trong project, không sử dụng dữ liệu giả lập hoặc artifact trung gian khác.

`CHUNK_SIZE = 500,000` phù hợp với mục tiêu của M3.3 là audit toàn bộ dataset nhưng không cần giữ hơn 24 triệu transaction trong RAM cùng lúc.

Notebook vì vậy có thể chạy độc lập trong project và thực hiện split audit trực tiếp trên raw artifact.

#### Kết luận

`Kết luận: PASS`

Môi trường thực thi và đường dẫn dữ liệu hợp lệ. M3.3 có thể tiếp tục audit trên raw artifact th

## 2. Khóa candidate specification trước khi nhìn output M3.3

Mục đích của cell này là làm cho boundary có thể kiểm tra và tái hiện.

Tất cả khoảng thời gian dùng quy ước:

`[start, end)`

tức là có `start` và không chứa `end`.



```python
# ============================================================
# Temporal constants
# ============================================================

PRE_2019_START = pd.Timestamp("2019-01-01")
PRE_BREAK_END = pd.Timestamp("2019-11-01")

POST_BREAK_START = pd.Timestamp("2019-11-01")
ARTIFACT_END_EXCLUSIVE = pd.Timestamp("2020-03-01")


TRAIN_WINDOWS = {
    "W_LONG_2015_TO_2018": {
        "start": pd.Timestamp("2015-01-01"),
        "end": pd.Timestamp("2019-01-01"),
    },

    "W_SHORT_2018_ONLY": {
        "start": pd.Timestamp("2018-01-01"),
        "end": pd.Timestamp("2019-01-01"),
    },
}


SPLIT_CANDIDATES = {
    "S1_4M_6M": {
        "validation_start": pd.Timestamp("2019-01-01"),
        "test_start": pd.Timestamp("2019-05-01"),
        "test_end": pd.Timestamp("2019-11-01"),
    },

    "S2_5M_5M": {
        "validation_start": pd.Timestamp("2019-01-01"),
        "test_start": pd.Timestamp("2019-06-01"),
        "test_end": pd.Timestamp("2019-11-01"),
    },

    "S3_6M_4M": {
        "validation_start": pd.Timestamp("2019-01-01"),
        "test_start": pd.Timestamp("2019-07-01"),
        "test_end": pd.Timestamp("2019-11-01"),
    },
}


candidate_spec_rows = []

for candidate_name, spec in SPLIT_CANDIDATES.items():
    candidate_spec_rows.append(
        {
            "candidate": candidate_name,
            "validation_start": spec["validation_start"],
            "validation_end_exclusive": spec["test_start"],
            "test_start": spec["test_start"],
            "test_end_exclusive": spec["test_end"],
        }
    )

candidate_spec_df = pd.DataFrame(
    candidate_spec_rows
)

display(candidate_spec_df)

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
      <th>candidate</th>
      <th>validation_start</th>
      <th>validation_end_exclusive</th>
      <th>test_start</th>
      <th>test_end_exclusive</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>S1_4M_6M</td>
      <td>2019-01-01</td>
      <td>2019-05-01</td>
      <td>2019-05-01</td>
      <td>2019-11-01</td>
    </tr>
    <tr>
      <th>1</th>
      <td>S2_5M_5M</td>
      <td>2019-01-01</td>
      <td>2019-06-01</td>
      <td>2019-06-01</td>
      <td>2019-11-01</td>
    </tr>
    <tr>
      <th>2</th>
      <td>S3_6M_4M</td>
      <td>2019-01-01</td>
      <td>2019-07-01</td>
      <td>2019-07-01</td>
      <td>2019-11-01</td>
    </tr>
  </tbody>
</table>
</div>


### Nhận xét candidate specification

#### Phân tích

Ba candidate đều sử dụng cùng một future-evaluation reservoir:

`2019-01-01 ≤ Timestamp < 2019-11-01`.

Điểm khác nhau duy nhất giữa ba candidate là vị trí boundary giữa validation và final test:

`S1_4M_6M`

* validation: 4 tháng;
* final test: 6 tháng.

`S2_5M_5M`

* validation: 5 tháng;
* final test: 5 tháng.

`S3_6M_4M`

* validation: 6 tháng;
* final test: 4 tháng.

Cả ba candidate đều giữ validation bắt đầu tại `2019-01-01` và final test kết thúc trước `2019-11-01`, tức không đưa zero-fraud regime từ tháng 11/2019 vào candidate final fraud-performance test.

Việc định nghĩa khoảng theo `[start, end)` cũng làm boundary rõ ràng: một transaction chỉ thuộc đúng một partition và không có overlap tại ngày chuyển giao.

Ở thời điểm specification, chưa có cơ sở để chọn candidate nào; đây chỉ là ba giả thuyết thiết kế cần được kiểm tra bằng output thực tế.

#### Kết luận

Ba candidate được định nghĩa hợp lệ và đủ rõ để audit.

`S1_4M_6M`, `S2_5M_5M` và `S3_6M_4M` đều được giữ ở trạng thái `CANDIDATE` trước khi xem kết quả.

Chưa khóa final split tại bước specification.

## 3. Helper

`User + Card` được dùng làm Card key vì `Card` là index trong phạm vi User.

State của mỗi partition giữ:

- row count;
- fraud count;
- unique User;
- unique User+Card;
- transaction count theo User/Card để tính transaction share trên new/seen entity;
- min/max Timestamp thực tế.



```python
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
        df["Time"]
        .astype("string")
        .str.strip()
        + ":00",
        errors="coerce",
    )

    return (
        date_part
        + time_part
    )


def init_partition_state():
    return {
        "row_count": 0,
        "fraud_count": 0,
        "users": set(),
        "cards": set(),
        "user_tx_counts": Counter(),
        "card_tx_counts": Counter(),
        "min_timestamp": None,
        "max_timestamp": None,
    }


def update_partition_state(
    state,
    chunk,
    timestamp,
    fraud_mask,
    mask,
):
    if not mask.any():
        return

    sub_users = (
        chunk.loc[
            mask,
            "User",
        ]
        .astype("int64")
    )

    sub_cards = (
        chunk.loc[
            mask,
            [
                "User",
                "Card",
            ],
        ]
        .astype("int64")
    )

    sub_timestamp = (
        timestamp.loc[
            mask
        ]
    )

    state["row_count"] += int(
        mask.sum()
    )

    state["fraud_count"] += int(
        fraud_mask.loc[
            mask
        ].sum()
    )

    state["users"].update(
        int(value)
        for value
        in sub_users.unique()
    )

    card_keys = list(
        zip(
            sub_cards["User"].tolist(),
            sub_cards["Card"].tolist(),
        )
    )

    state["cards"].update(
        card_keys
    )

    state[
        "user_tx_counts"
    ].update(
        int(value)
        for value
        in sub_users.tolist()
    )

    state[
        "card_tx_counts"
    ].update(
        card_keys
    )

    chunk_min = sub_timestamp.min()
    chunk_max = sub_timestamp.max()

    if (
        state["min_timestamp"] is None
        or chunk_min < state["min_timestamp"]
    ):
        state["min_timestamp"] = chunk_min

    if (
        state["max_timestamp"] is None
        or chunk_max > state["max_timestamp"]
    ):
        state["max_timestamp"] = chunk_max


def update_entity_set(
    entity_state,
    chunk,
    mask,
):
    if not mask.any():
        return

    users = (
        chunk.loc[
            mask,
            "User",
        ]
        .astype("int64")
    )

    cards = (
        chunk.loc[
            mask,
            [
                "User",
                "Card",
            ],
        ]
        .astype("int64")
    )

    entity_state["users"].update(
        int(value)
        for value
        in users.unique()
    )

    entity_state["cards"].update(
        zip(
            cards["User"].tolist(),
            cards["Card"].tolist(),
        )
    )


def safe_pct(
    numerator,
    denominator,
):
    if denominator == 0:
        return float("nan")

    return (
        numerator
        / denominator
        * 100
    )

```

## 4. Full-artifact split audit

Cell này thực hiện **một full scan**.

Nó đồng thời thu:

1. global integrity statistics;
2. W_LONG / W_SHORT training-window statistics;
3. validation/test statistics cho ba candidate;
4. entity history có trước validation/test;
5. pre-break/post-break consistency statistics.

Không train model và không tính behavioral feature.



```python
AUDIT_USECOLS = [
    "User",
    "Card",
    "Year",
    "Month",
    "Day",
    "Time",
    "Is Fraud?",
]


# ============================================================
# Global integrity
# ============================================================

total_rows = 0
total_fraud = 0
timestamp_parse_failures = 0


# ============================================================
# Training-window state
# ============================================================

train_states = {
    window_name:
        init_partition_state()

    for window_name
    in TRAIN_WINDOWS
}


# ============================================================
# Candidate partition state
# ============================================================

candidate_states = {
    candidate_name: {
        "VALIDATION":
            init_partition_state(),

        "FINAL_TEST":
            init_partition_state(),
    }

    for candidate_name
    in SPLIT_CANDIDATES
}


# ============================================================
# All-prior-history entity state
#
# PRE_VALIDATION:
#   toàn bộ entity trước 2019-01-01
#
# PRE_TEST:
#   toàn bộ entity trước test_start của từng candidate
# ============================================================

pre_validation_entities = {
    "users": set(),
    "cards": set(),
}

pre_test_entities = {
    candidate_name: {
        "users": set(),
        "cards": set(),
    }

    for candidate_name
    in SPLIT_CANDIDATES
}


# ============================================================
# Cross-check regions
# ============================================================

pre_break_2019_state = (
    init_partition_state()
)

post_break_state = (
    init_partition_state()
)


for chunk_number, chunk in enumerate(
    pd.read_csv(
        DATA_PATH,
        usecols=AUDIT_USECOLS,
        chunksize=CHUNK_SIZE,
    ),
    start=1,
):
    total_rows += len(
        chunk
    )

    timestamp = build_timestamp(
        chunk
    )

    timestamp_parse_failures += int(
        timestamp
        .isna()
        .sum()
    )

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

    total_fraud += int(
        fraud_mask.sum()
    )


    # --------------------------------------------------------
    # Prior history before validation
    # --------------------------------------------------------

    pre_validation_mask = (
        timestamp
        < PRE_2019_START
    )

    update_entity_set(
        pre_validation_entities,
        chunk,
        pre_validation_mask,
    )


    # --------------------------------------------------------
    # Training-window families
    # --------------------------------------------------------

    for window_name, spec in (
        TRAIN_WINDOWS.items()
    ):
        mask = (
            (timestamp >= spec["start"])
            &
            (timestamp < spec["end"])
        )

        update_partition_state(
            train_states[
                window_name
            ],
            chunk,
            timestamp,
            fraud_mask,
            mask,
        )


    # --------------------------------------------------------
    # Candidate validation/test
    # --------------------------------------------------------

    for candidate_name, spec in (
        SPLIT_CANDIDATES.items()
    ):
        validation_mask = (
            (timestamp >= spec["validation_start"])
            &
            (timestamp < spec["test_start"])
        )

        test_mask = (
            (timestamp >= spec["test_start"])
            &
            (timestamp < spec["test_end"])
        )

        update_partition_state(
            candidate_states[
                candidate_name
            ]["VALIDATION"],
            chunk,
            timestamp,
            fraud_mask,
            validation_mask,
        )

        update_partition_state(
            candidate_states[
                candidate_name
            ]["FINAL_TEST"],
            chunk,
            timestamp,
            fraud_mask,
            test_mask,
        )


        # Toàn bộ history trước test-start,
        # gồm cả validation period.
        prior_test_mask = (
            timestamp
            < spec["test_start"]
        )

        update_entity_set(
            pre_test_entities[
                candidate_name
            ],
            chunk,
            prior_test_mask,
        )


    # --------------------------------------------------------
    # Pre-break future reservoir 2019-01 -> 2019-10
    # --------------------------------------------------------

    pre_break_mask = (
        (timestamp >= PRE_2019_START)
        &
        (timestamp < PRE_BREAK_END)
    )

    update_partition_state(
        pre_break_2019_state,
        chunk,
        timestamp,
        fraud_mask,
        pre_break_mask,
    )


    # --------------------------------------------------------
    # Post-break region 2019-11 -> 2020-02
    # --------------------------------------------------------

    post_break_mask = (
        (timestamp >= POST_BREAK_START)
        &
        (timestamp < ARTIFACT_END_EXCLUSIVE)
    )

    update_partition_state(
        post_break_state,
        chunk,
        timestamp,
        fraud_mask,
        post_break_mask,
    )


    if (
        chunk_number % 10 == 0
        or len(chunk) < CHUNK_SIZE
    ):
        print(
            f"Đã audit {chunk_number} chunk "
            f"- {total_rows:,} transaction"
        )


print("\nHoàn tất full-artifact split audit.")

print(
    "Total rows:",
    f"{total_rows:,}",
)

print(
    "Total fraud:",
    f"{total_fraud:,}",
)

print(
    "Timestamp parse failures:",
    timestamp_parse_failures,
)

```

    Đã audit 10 chunk - 5,000,000 transaction
    Đã audit 20 chunk - 10,000,000 transaction
    Đã audit 30 chunk - 15,000,000 transaction
    Đã audit 40 chunk - 20,000,000 transaction
    Đã audit 49 chunk - 24,386,900 transaction
    
    Hoàn tất full-artifact split audit.
    Total rows: 24,386,900
    Total fraud: 29,757
    Timestamp parse failures: 0


### Nhận xét full-artifact audit

#### Phân tích

Full-artifact audit đã hoàn tất toàn bộ `49 chunk` và xử lý đúng:

`24,386,900 transaction`.

Tổng fraud thu được là:

`29,757`.

Đây là các mốc đã được xác nhận từ các bước EDA trước, nên việc M3.3 tái tạo đúng hai giá trị này là dấu hiệu đầu tiên cho thấy notebook đang đọc đúng raw artifact và target được parse đúng.

Đồng thời:

`Timestamp parse failures = 0`.

Như vậy toàn bộ transaction được sử dụng trong temporal split audit đều tạo được Timestamp hợp lệ. Không có row nào bị mất temporal identity do lỗi parse ngày/giờ.

Full scan cũng hoàn tất mà không có exception, cho thấy các state dùng để thu training-window statistics, partition statistics và entity-history statistics đã được tính trên toàn bộ artifact.

#### Kết luận

`Kết luận: PASS`

Full-artifact audit đã chạy thành công trên toàn bộ dataset thật.

Không phát hiện:

* thiếu row;
* sai tổng fraud;
* lỗi parse Timestamp;
* hoặc lỗi thực thi blocking.

Các output partition phía sau đủ điều kiện để tiếp tục kiểm tra consistency và diễn giải.

## 5. Training-window summary

M3.3 chưa chọn W_LONG hay W_SHORT.

Bảng này chỉ xác nhận quy mô và entity coverage thực tế của hai họ training window dùng chung một cutoff `2019-01-01`.



```python
train_summary_rows = []

for window_name, spec in (
    TRAIN_WINDOWS.items()
):
    state = (
        train_states[
            window_name
        ]
    )

    train_summary_rows.append(
        {
            "train_window":
                window_name,

            "configured_start":
                spec["start"],

            "configured_end_exclusive":
                spec["end"],

            "actual_min_timestamp":
                state["min_timestamp"],

            "actual_max_timestamp":
                state["max_timestamp"],

            "transaction_count":
                state["row_count"],

            "fraud_count":
                state["fraud_count"],

            "fraud_rate_pct":
                safe_pct(
                    state["fraud_count"],
                    state["row_count"],
                ),

            "unique_users":
                len(
                    state["users"]
                ),

            "unique_user_card_pairs":
                len(
                    state["cards"]
                ),
        }
    )


train_summary_df = pd.DataFrame(
    train_summary_rows
)

display(
    train_summary_df
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
      <th>train_window</th>
      <th>configured_start</th>
      <th>configured_end_exclusive</th>
      <th>actual_min_timestamp</th>
      <th>actual_max_timestamp</th>
      <th>transaction_count</th>
      <th>fraud_count</th>
      <th>fraud_rate_pct</th>
      <th>unique_users</th>
      <th>unique_user_card_pairs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>W_LONG_2015_TO_2018</td>
      <td>2015-01-01</td>
      <td>2019-01-01</td>
      <td>2015-01-01 00:01:00</td>
      <td>2018-12-31 23:58:00</td>
      <td>6855270</td>
      <td>9606</td>
      <td>0.140126</td>
      <td>1567</td>
      <td>4289</td>
    </tr>
    <tr>
      <th>1</th>
      <td>W_SHORT_2018_ONLY</td>
      <td>2018-01-01</td>
      <td>2019-01-01</td>
      <td>2018-01-01 00:03:00</td>
      <td>2018-12-31 23:58:00</td>
      <td>1721615</td>
      <td>2491</td>
      <td>0.144690</td>
      <td>1532</td>
      <td>4006</td>
    </tr>
  </tbody>
</table>
</div>


### Nhận xét training-window summary

#### Phân tích

Hai training-window family cùng kết thúc tại `2019-01-01` nhưng khác độ dài lịch sử classifier-training.

`W_LONG_2015_TO_2018` chứa:

`6,855,270 transaction`
`9,606 fraud`
`fraud rate ≈ 0.140126%`
`1,567 User`
`4,289 User+Card`.

`W_SHORT_2018_ONLY` chứa:

`1,721,615 transaction`
`2,491 fraud`
`fraud rate ≈ 0.144690%`
`1,532 User`
`4,006 User+Card`.

Như vậy W_LONG có khoảng bốn lần số transaction và gần bốn lần số fraud của W_SHORT.

Tuy nhiên fraud rate của hai window khá gần nhau ở mức tổng hợp. Điều này chỉ mô tả class prevalence, không chứng minh distribution của feature hoặc relationship với target là giống nhau.

W_LONG cũng chứa nhiều User và Card hơn W_SHORT, nhưng chênh lệch entity coverage nhỏ hơn rất nhiều so với chênh lệch số transaction. Điều này cho thấy phần dữ liệu 2015–2017 chủ yếu bổ sung thêm lịch sử và nhiều transaction trên population entity tương đối tương đồng, thay vì tạo ra một population hoàn toàn khác về số lượng entity.

Quan trọng nhất, các số trên chỉ chứng minh cả hai training window đều có quy mô thực tế đủ lớn để tiếp tục làm candidate.

Không có model experiment nào được thực hiện trong M3.3, nên chưa thể kết luận W_LONG hay W_SHORT tạo classifier tốt hơn.

#### Kết luận

Cả hai training-window family đều khả thi về sample size và positive support:

`W_LONG_2015_TO_2018: VERIFIED AS FEASIBLE`

`W_SHORT_2018_ONLY: VERIFIED AS FEASIBLE`

W_LONG cung cấp nhiều training data và nhiều fraud hơn đáng kể.

W_SHORT nhỏ hơn nhưng vẫn giữ hơn `1.72 triệu transaction` và `2,491 fraud`.

`Training-window winner: OPEN`

M3.3 không chọn W_LONG hay W_SHORT. Việc lựa chọn phải dựa trên model experiment công bằng ở bước thích hợp sau này.


## 6. Candidate validation / final-test summary

Đây là output chính để xem mỗi boundary tạo:

- bao nhiêu row;
- bao nhiêu fraud;
- fraud rate;
- User/Card coverage;
- actual timestamp span.

Chưa đặt một ngưỡng tùy ý như “ít nhất X fraud là đủ”.

Positive support sẽ được đọc từ output thực tế rồi mới quyết định.



```python
split_summary_rows = []

for candidate_name, spec in (
    SPLIT_CANDIDATES.items()
):
    partition_specs = {
        "VALIDATION": {
            "start":
                spec[
                    "validation_start"
                ],

            "end":
                spec[
                    "test_start"
                ],
        },

        "FINAL_TEST": {
            "start":
                spec[
                    "test_start"
                ],

            "end":
                spec[
                    "test_end"
                ],
        },
    }

    for partition_name, partition_spec in (
        partition_specs.items()
    ):
        state = (
            candidate_states[
                candidate_name
            ][
                partition_name
            ]
        )

        split_summary_rows.append(
            {
                "candidate":
                    candidate_name,

                "partition":
                    partition_name,

                "configured_start":
                    partition_spec[
                        "start"
                    ],

                "configured_end_exclusive":
                    partition_spec[
                        "end"
                    ],

                "actual_min_timestamp":
                    state[
                        "min_timestamp"
                    ],

                "actual_max_timestamp":
                    state[
                        "max_timestamp"
                    ],

                "transaction_count":
                    state[
                        "row_count"
                    ],

                "fraud_count":
                    state[
                        "fraud_count"
                    ],

                "fraud_rate_pct":
                    safe_pct(
                        state[
                            "fraud_count"
                        ],
                        state[
                            "row_count"
                        ],
                    ),

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


split_summary_df = pd.DataFrame(
    split_summary_rows
)

display(
    split_summary_df
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
      <th>candidate</th>
      <th>partition</th>
      <th>configured_start</th>
      <th>configured_end_exclusive</th>
      <th>actual_min_timestamp</th>
      <th>actual_max_timestamp</th>
      <th>transaction_count</th>
      <th>fraud_count</th>
      <th>fraud_rate_pct</th>
      <th>unique_users</th>
      <th>unique_user_card_pairs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>S1_4M_6M</td>
      <td>VALIDATION</td>
      <td>2019-01-01</td>
      <td>2019-05-01</td>
      <td>2019-01-01 00:02:00</td>
      <td>2019-04-30 23:58:00</td>
      <td>566810</td>
      <td>792</td>
      <td>0.139729</td>
      <td>1528</td>
      <td>3912</td>
    </tr>
    <tr>
      <th>1</th>
      <td>S1_4M_6M</td>
      <td>FINAL_TEST</td>
      <td>2019-05-01</td>
      <td>2019-11-01</td>
      <td>2019-05-01 00:02:00</td>
      <td>2019-10-31 23:59:00</td>
      <td>868603</td>
      <td>1295</td>
      <td>0.149090</td>
      <td>1526</td>
      <td>3913</td>
    </tr>
    <tr>
      <th>2</th>
      <td>S2_5M_5M</td>
      <td>VALIDATION</td>
      <td>2019-01-01</td>
      <td>2019-06-01</td>
      <td>2019-01-01 00:02:00</td>
      <td>2019-05-31 23:58:00</td>
      <td>712458</td>
      <td>1052</td>
      <td>0.147658</td>
      <td>1528</td>
      <td>3915</td>
    </tr>
    <tr>
      <th>3</th>
      <td>S2_5M_5M</td>
      <td>FINAL_TEST</td>
      <td>2019-06-01</td>
      <td>2019-11-01</td>
      <td>2019-06-01 00:02:00</td>
      <td>2019-10-31 23:59:00</td>
      <td>722955</td>
      <td>1035</td>
      <td>0.143162</td>
      <td>1525</td>
      <td>3908</td>
    </tr>
    <tr>
      <th>4</th>
      <td>S3_6M_4M</td>
      <td>VALIDATION</td>
      <td>2019-01-01</td>
      <td>2019-07-01</td>
      <td>2019-01-01 00:02:00</td>
      <td>2019-06-30 23:59:00</td>
      <td>854857</td>
      <td>1267</td>
      <td>0.148212</td>
      <td>1528</td>
      <td>3920</td>
    </tr>
    <tr>
      <th>5</th>
      <td>S3_6M_4M</td>
      <td>FINAL_TEST</td>
      <td>2019-07-01</td>
      <td>2019-11-01</td>
      <td>2019-07-01 00:00:00</td>
      <td>2019-10-31 23:59:00</td>
      <td>580556</td>
      <td>820</td>
      <td>0.141244</td>
      <td>1522</td>
      <td>3897</td>
    </tr>
  </tbody>
</table>
</div>


### Nhận xét candidate partition summary

#### Phân tích

Cả ba candidate đều tạo validation và final test có quy mô lớn, hàng trăm nghìn transaction và hàng trăm đến hơn một nghìn fraud.

##### S1_4M_6M

Validation:

`566,810 transaction`
`792 fraud`
`fraud rate ≈ 0.139729%`.

Final test:

`868,603 transaction`
`1,295 fraud`
`fraud rate ≈ 0.149090%`.

S1 dành phần nhỏ hơn của future reservoir cho validation và phần lớn hơn cho final test.

Do đó final test lớn hơn validation khoảng `301,793 transaction` và có thêm `503 fraud`.

Validation vẫn có positive support đáng kể, nhưng allocation giữa hai partition tương đối lệch.

##### S2_5M_5M

Validation:

`712,458 transaction`
`1,052 fraud`
`fraud rate ≈ 0.147658%`.

Final test:

`722,955 transaction`
`1,035 fraud`
`fraud rate ≈ 0.143162%`.

Hai partition đều kéo dài 5 tháng.

Chênh lệch chỉ:

`10,497 transaction`

và:

`17 fraud`.

Do đó S2 tạo allocation cân bằng nhất trong ba candidate cả về thời lượng, transaction volume và positive support.

Fraud rate giữa validation và final test cũng không chênh lệch lớn ở mức tổng hợp. Đây là một finding mô tả hữu ích, nhưng không được hiểu là hai partition có distribution hoàn toàn giống nhau.

##### S3_6M_4M

Validation:

`854,857 transaction`
`1,267 fraud`
`fraud rate ≈ 0.148212%`.

Final test:

`580,556 transaction`
`820 fraud`
`fraud rate ≈ 0.141244%`.

S3 dành nhiều dữ liệu hơn cho validation nhưng làm final test ngắn nhất trong ba candidate.

Final test vẫn có `820 fraud`, tức không phải một partition thiếu positive class, nhưng positive support và transaction volume của final test thấp nhất trong ba phương án.

##### So sánh chung

Cả ba candidate đều đủ lớn về mặt tuyệt đối để tiếp tục được xem là temporal split hợp lệ.

Sự khác biệt chính không phải candidate nào “có fraud hay không”, mà là cách phân bổ `1,435,413 transaction / 2,087 fraud` của future reservoir giữa validation và final test.

S1 nghiêng về final test.

S3 nghiêng về validation.

S2 chia gần cân bằng nhất.

Trong bối cảnh project cần:

* validation đủ mạnh cho các quyết định lặp lại;
* final test đủ mạnh cho đánh giá cuối;
* một boundary đơn giản, dễ giải thích;
* và không có lý do nghiệp vụ nào hiện tại buộc một partition phải dài hơn partition còn lại;

S2 là candidate cân bằng nhất về experiment design.

#### Kết luận

Cả ba candidate đều có positive support thực tế và không bị loại vì thiếu dữ liệu.

Tuy nhiên:

`S1_4M_6M`
→ hợp lệ nhưng allocation nghiêng về final test.

`S3_6M_4M`
→ hợp lệ nhưng allocation nghiêng về validation và để lại final test nhỏ nhất.

`S2_5M_5M`
→ tạo allocation cân bằng nhất giữa validation và final test.

`Candidate ưu tiên sau partition-size audit: S2_5M_5M`.

Quyết định cuối vẫn phải được đối chiếu với cold-start, entity overlap và structural gate trước khi LOCKED.

## 7. Strict prior-history / cold-start audit

Ở đây `new User/Card` có nghĩa:

> entity xuất hiện trong partition nhưng **chưa từng xuất hiện ở bất kỳ transaction nào trước partition start**.

Đây là audit causal-history availability.

Đối với final test, history trước test-start có thể bao gồm transaction trong validation period vì chúng xảy ra trong quá khứ so với test.

Không sử dụng fraud label của history để tạo feature.



```python
cold_start_rows = []


for candidate_name in (
    SPLIT_CANDIDATES
):
    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    validation_state = (
        candidate_states[
            candidate_name
        ]["VALIDATION"]
    )

    validation_new_users = (
        validation_state[
            "users"
        ]
        - pre_validation_entities[
            "users"
        ]
    )

    validation_new_cards = (
        validation_state[
            "cards"
        ]
        - pre_validation_entities[
            "cards"
        ]
    )

    validation_new_user_tx = sum(
        validation_state[
            "user_tx_counts"
        ][user]
        for user
        in validation_new_users
    )

    validation_new_card_tx = sum(
        validation_state[
            "card_tx_counts"
        ][card]
        for card
        in validation_new_cards
    )

    cold_start_rows.append(
        {
            "candidate":
                candidate_name,

            "partition":
                "VALIDATION",

            "new_user_count":
                len(
                    validation_new_users
                ),

            "new_user_unique_pct":
                safe_pct(
                    len(
                        validation_new_users
                    ),
                    len(
                        validation_state[
                            "users"
                        ]
                    ),
                ),

            "transactions_on_new_users":
                validation_new_user_tx,

            "transactions_on_new_users_pct":
                safe_pct(
                    validation_new_user_tx,
                    validation_state[
                        "row_count"
                    ],
                ),

            "new_card_count":
                len(
                    validation_new_cards
                ),

            "new_card_unique_pct":
                safe_pct(
                    len(
                        validation_new_cards
                    ),
                    len(
                        validation_state[
                            "cards"
                        ]
                    ),
                ),

            "transactions_on_new_cards":
                validation_new_card_tx,

            "transactions_on_new_cards_pct":
                safe_pct(
                    validation_new_card_tx,
                    validation_state[
                        "row_count"
                    ],
                ),
        }
    )


    # --------------------------------------------------------
    # Final test
    # --------------------------------------------------------

    test_state = (
        candidate_states[
            candidate_name
        ]["FINAL_TEST"]
    )

    prior_test = (
        pre_test_entities[
            candidate_name
        ]
    )

    test_new_users = (
        test_state[
            "users"
        ]
        - prior_test[
            "users"
        ]
    )

    test_new_cards = (
        test_state[
            "cards"
        ]
        - prior_test[
            "cards"
        ]
    )

    test_new_user_tx = sum(
        test_state[
            "user_tx_counts"
        ][user]
        for user
        in test_new_users
    )

    test_new_card_tx = sum(
        test_state[
            "card_tx_counts"
        ][card]
        for card
        in test_new_cards
    )

    cold_start_rows.append(
        {
            "candidate":
                candidate_name,

            "partition":
                "FINAL_TEST",

            "new_user_count":
                len(
                    test_new_users
                ),

            "new_user_unique_pct":
                safe_pct(
                    len(
                        test_new_users
                    ),
                    len(
                        test_state[
                            "users"
                        ]
                    ),
                ),

            "transactions_on_new_users":
                test_new_user_tx,

            "transactions_on_new_users_pct":
                safe_pct(
                    test_new_user_tx,
                    test_state[
                        "row_count"
                    ],
                ),

            "new_card_count":
                len(
                    test_new_cards
                ),

            "new_card_unique_pct":
                safe_pct(
                    len(
                        test_new_cards
                    ),
                    len(
                        test_state[
                            "cards"
                        ]
                    ),
                ),

            "transactions_on_new_cards":
                test_new_card_tx,

            "transactions_on_new_cards_pct":
                safe_pct(
                    test_new_card_tx,
                    test_state[
                        "row_count"
                    ],
                ),
        }
    )


cold_start_df = pd.DataFrame(
    cold_start_rows
)

display(
    cold_start_df
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
      <th>candidate</th>
      <th>partition</th>
      <th>new_user_count</th>
      <th>new_user_unique_pct</th>
      <th>transactions_on_new_users</th>
      <th>transactions_on_new_users_pct</th>
      <th>new_card_count</th>
      <th>new_card_unique_pct</th>
      <th>transactions_on_new_cards</th>
      <th>transactions_on_new_cards_pct</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>S1_4M_6M</td>
      <td>VALIDATION</td>
      <td>3</td>
      <td>0.196335</td>
      <td>632</td>
      <td>0.111501</td>
      <td>31</td>
      <td>0.792434</td>
      <td>2861</td>
      <td>0.504755</td>
    </tr>
    <tr>
      <th>1</th>
      <td>S1_4M_6M</td>
      <td>FINAL_TEST</td>
      <td>4</td>
      <td>0.262123</td>
      <td>536</td>
      <td>0.061708</td>
      <td>41</td>
      <td>1.047789</td>
      <td>3388</td>
      <td>0.390052</td>
    </tr>
    <tr>
      <th>2</th>
      <td>S2_5M_5M</td>
      <td>VALIDATION</td>
      <td>3</td>
      <td>0.196335</td>
      <td>794</td>
      <td>0.111445</td>
      <td>33</td>
      <td>0.842912</td>
      <td>3975</td>
      <td>0.557928</td>
    </tr>
    <tr>
      <th>3</th>
      <td>S2_5M_5M</td>
      <td>FINAL_TEST</td>
      <td>4</td>
      <td>0.262295</td>
      <td>536</td>
      <td>0.074140</td>
      <td>39</td>
      <td>0.997953</td>
      <td>3060</td>
      <td>0.423263</td>
    </tr>
    <tr>
      <th>4</th>
      <td>S3_6M_4M</td>
      <td>VALIDATION</td>
      <td>3</td>
      <td>0.196335</td>
      <td>954</td>
      <td>0.111598</td>
      <td>37</td>
      <td>0.943878</td>
      <td>5178</td>
      <td>0.605715</td>
    </tr>
    <tr>
      <th>5</th>
      <td>S3_6M_4M</td>
      <td>FINAL_TEST</td>
      <td>4</td>
      <td>0.262812</td>
      <td>536</td>
      <td>0.092325</td>
      <td>35</td>
      <td>0.898127</td>
      <td>2670</td>
      <td>0.459904</td>
    </tr>
  </tbody>
</table>
</div>


### Nhận xét strict prior-history / cold-start

#### Phân tích

Cold-start audit sử dụng định nghĩa nghiêm ngặt:

một User hoặc User+Card chỉ được coi là `new` nếu entity đó chưa từng xuất hiện ở bất kỳ transaction nào xảy ra trước partition start.

Do đó đây là phép đo về causal historical availability, không chỉ là overlap với classifier-training rows.

##### Validation

Cả ba candidate validation đều chỉ có:

`3 new User`.

Tỷ lệ unique User mới khoảng:

`0.1963%`.

Tỷ lệ transaction thuộc User hoàn toàn mới cũng rất nhỏ:

xấp xỉ `0.1115%`.

Đối với Card:

S1 có `31 new Card`.

S2 có `33 new Card`.

S3 có `37 new Card`.

Transaction trên Card hoàn toàn mới chỉ chiếm khoảng:

`0.50% → 0.61%`.

##### Final test

Cả ba candidate final test đều chỉ có:

`4 new User`.

Transaction trên User mới chiếm dưới `0.1%`.

Số new Card lần lượt là:

S1: `41`
S2: `39`
S3: `35`.

Transaction thuộc Card hoàn toàn mới chỉ chiếm khoảng:

`0.39% → 0.46%`.

##### Đối với S2

Candidate đang được ưu tiên có:

Validation:

* `3 new User`;
* `33 new Card`;
* `0.111445%` transaction thuộc new User;
* `0.557928%` transaction thuộc new Card.

Final test:

* `4 new User`;
* `39 new Card`;
* `0.074140%` transaction thuộc new User;
* `0.423263%` transaction thuộc new Card.

Như vậy hơn 99% transaction của cả validation và final test không phải strict card cold-start.

Finding này phù hợp với M2: future evaluation của dataset chủ yếu là transaction trên entity đã tồn tại và có prior history.

Điều đó có hai hệ quả.

Thứ nhất, historical feature có khả năng có prior context trên phần lớn future transaction.

Thứ hai, final model performance sau này không nên được diễn giải như bằng chứng mạnh về khả năng xử lý hoàn toàn-new User/Card, vì segment đó rất nhỏ trong evaluation population.

Không có candidate nào xuất hiện cold-start rate đủ lớn để trở thành blocker cho split.

#### Kết luận

`Strict prior-history coverage: HIGH`

`Cold-start prevalence: LOW`

Cold-start không phải blocking issue cho S1, S2 hoặc S3.

Đối với S2, main validation/final-test population chủ yếu gồm existing User/Card có historical context trước prediction period.

Completely-new User/Card phải được ghi nhận như một segment nhỏ và limitation của evaluation, không phải population chính.

`Cold-start audit: PASS WITH FINDINGS`

## 8. Future entity overlap với classifier-training rows

Phân tích này khác cold-start.

Một future Card có thể:

- đã có lịch sử trước partition;
- nhưng không xuất hiện trong một recent classifier-training window cụ thể.

Vì raw User/Card không được đưa trực tiếp vào classifier, overlap không tự động là leakage.

Bảng này chỉ mô tả mức độ future entity đã xuất hiện trong training rows của W_LONG hoặc W_SHORT.



```python
entity_overlap_rows = []


for candidate_name in (
    SPLIT_CANDIDATES
):
    for partition_name in [
        "VALIDATION",
        "FINAL_TEST",
    ]:
        future_state = (
            candidate_states[
                candidate_name
            ][
                partition_name
            ]
        )

        for train_window_name in (
            TRAIN_WINDOWS
        ):
            train_state = (
                train_states[
                    train_window_name
                ]
            )

            overlapping_users = (
                future_state[
                    "users"
                ]
                & train_state[
                    "users"
                ]
            )

            overlapping_cards = (
                future_state[
                    "cards"
                ]
                & train_state[
                    "cards"
                ]
            )

            tx_on_seen_users = sum(
                future_state[
                    "user_tx_counts"
                ][user]
                for user
                in overlapping_users
            )

            tx_on_seen_cards = sum(
                future_state[
                    "card_tx_counts"
                ][card]
                for card
                in overlapping_cards
            )

            entity_overlap_rows.append(
                {
                    "candidate":
                        candidate_name,

                    "partition":
                        partition_name,

                    "train_window":
                        train_window_name,

                    "future_unique_users":
                        len(
                            future_state[
                                "users"
                            ]
                        ),

                    "user_overlap_count":
                        len(
                            overlapping_users
                        ),

                    "user_overlap_unique_pct":
                        safe_pct(
                            len(
                                overlapping_users
                            ),
                            len(
                                future_state[
                                    "users"
                                ]
                            ),
                        ),

                    "future_tx_on_train_seen_users_pct":
                        safe_pct(
                            tx_on_seen_users,
                            future_state[
                                "row_count"
                            ],
                        ),

                    "future_unique_cards":
                        len(
                            future_state[
                                "cards"
                            ]
                        ),

                    "card_overlap_count":
                        len(
                            overlapping_cards
                        ),

                    "card_overlap_unique_pct":
                        safe_pct(
                            len(
                                overlapping_cards
                            ),
                            len(
                                future_state[
                                    "cards"
                                ]
                            ),
                        ),

                    "future_tx_on_train_seen_cards_pct":
                        safe_pct(
                            tx_on_seen_cards,
                            future_state[
                                "row_count"
                            ],
                        ),
                }
            )


entity_overlap_df = pd.DataFrame(
    entity_overlap_rows
)

display(
    entity_overlap_df
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
      <th>candidate</th>
      <th>partition</th>
      <th>train_window</th>
      <th>future_unique_users</th>
      <th>user_overlap_count</th>
      <th>user_overlap_unique_pct</th>
      <th>future_tx_on_train_seen_users_pct</th>
      <th>future_unique_cards</th>
      <th>card_overlap_count</th>
      <th>card_overlap_unique_pct</th>
      <th>future_tx_on_train_seen_cards_pct</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>S1_4M_6M</td>
      <td>VALIDATION</td>
      <td>W_LONG_2015_TO_2018</td>
      <td>1528</td>
      <td>1523</td>
      <td>99.672775</td>
      <td>99.824633</td>
      <td>3912</td>
      <td>3881</td>
      <td>99.207566</td>
      <td>99.495245</td>
    </tr>
    <tr>
      <th>1</th>
      <td>S1_4M_6M</td>
      <td>VALIDATION</td>
      <td>W_SHORT_2018_ONLY</td>
      <td>1528</td>
      <td>1523</td>
      <td>99.672775</td>
      <td>99.824633</td>
      <td>3912</td>
      <td>3881</td>
      <td>99.207566</td>
      <td>99.495245</td>
    </tr>
    <tr>
      <th>2</th>
      <td>S1_4M_6M</td>
      <td>FINAL_TEST</td>
      <td>W_LONG_2015_TO_2018</td>
      <td>1526</td>
      <td>1518</td>
      <td>99.475754</td>
      <td>99.705734</td>
      <td>3913</td>
      <td>3842</td>
      <td>98.185535</td>
      <td>98.851834</td>
    </tr>
    <tr>
      <th>3</th>
      <td>S1_4M_6M</td>
      <td>FINAL_TEST</td>
      <td>W_SHORT_2018_ONLY</td>
      <td>1526</td>
      <td>1518</td>
      <td>99.475754</td>
      <td>99.705734</td>
      <td>3913</td>
      <td>3842</td>
      <td>98.185535</td>
      <td>98.851834</td>
    </tr>
    <tr>
      <th>4</th>
      <td>S2_5M_5M</td>
      <td>VALIDATION</td>
      <td>W_LONG_2015_TO_2018</td>
      <td>1528</td>
      <td>1523</td>
      <td>99.672775</td>
      <td>99.812340</td>
      <td>3915</td>
      <td>3882</td>
      <td>99.157088</td>
      <td>99.442072</td>
    </tr>
    <tr>
      <th>5</th>
      <td>S2_5M_5M</td>
      <td>VALIDATION</td>
      <td>W_SHORT_2018_ONLY</td>
      <td>1528</td>
      <td>1523</td>
      <td>99.672775</td>
      <td>99.812340</td>
      <td>3915</td>
      <td>3882</td>
      <td>99.157088</td>
      <td>99.442072</td>
    </tr>
    <tr>
      <th>6</th>
      <td>S2_5M_5M</td>
      <td>FINAL_TEST</td>
      <td>W_LONG_2015_TO_2018</td>
      <td>1525</td>
      <td>1517</td>
      <td>99.475410</td>
      <td>99.693895</td>
      <td>3908</td>
      <td>3837</td>
      <td>98.183214</td>
      <td>98.774613</td>
    </tr>
    <tr>
      <th>7</th>
      <td>S2_5M_5M</td>
      <td>FINAL_TEST</td>
      <td>W_SHORT_2018_ONLY</td>
      <td>1525</td>
      <td>1517</td>
      <td>99.475410</td>
      <td>99.693895</td>
      <td>3908</td>
      <td>3837</td>
      <td>98.183214</td>
      <td>98.774613</td>
    </tr>
    <tr>
      <th>8</th>
      <td>S3_6M_4M</td>
      <td>VALIDATION</td>
      <td>W_LONG_2015_TO_2018</td>
      <td>1528</td>
      <td>1523</td>
      <td>99.672775</td>
      <td>99.806634</td>
      <td>3920</td>
      <td>3883</td>
      <td>99.056122</td>
      <td>99.394285</td>
    </tr>
    <tr>
      <th>9</th>
      <td>S3_6M_4M</td>
      <td>VALIDATION</td>
      <td>W_SHORT_2018_ONLY</td>
      <td>1528</td>
      <td>1523</td>
      <td>99.672775</td>
      <td>99.806634</td>
      <td>3920</td>
      <td>3883</td>
      <td>99.056122</td>
      <td>99.394285</td>
    </tr>
    <tr>
      <th>10</th>
      <td>S3_6M_4M</td>
      <td>FINAL_TEST</td>
      <td>W_LONG_2015_TO_2018</td>
      <td>1522</td>
      <td>1514</td>
      <td>99.474376</td>
      <td>99.673244</td>
      <td>3897</td>
      <td>3827</td>
      <td>98.203746</td>
      <td>98.681264</td>
    </tr>
    <tr>
      <th>11</th>
      <td>S3_6M_4M</td>
      <td>FINAL_TEST</td>
      <td>W_SHORT_2018_ONLY</td>
      <td>1522</td>
      <td>1514</td>
      <td>99.474376</td>
      <td>99.673244</td>
      <td>3897</td>
      <td>3827</td>
      <td>98.203746</td>
      <td>98.681264</td>
    </tr>
  </tbody>
</table>
</div>


### Nhận xét entity overlap

#### Phân tích

Entity-overlap audit trả lời một câu hỏi khác với strict cold-start:

future entity có xuất hiện trong classifier-training rows của W_LONG hoặc W_SHORT hay không?

Kết quả cho thấy overlap rất cao ở cả User và Card.

Đáng chú ý hơn, đối với mọi candidate và partition, các overlap statistic của:

`W_LONG_2015_TO_2018`

và:

`W_SHORT_2018_ONLY`

là giống nhau.

Ví dụ với S2 validation:

`1,523 / 1,528 User`

đã xuất hiện trong cả W_LONG và W_SHORT:

`user overlap ≈ 99.672775%`.

Khoảng:

`99.812340%`

validation transaction thuộc các User đã xuất hiện trong training rows.

Đối với Card:

`3,882 / 3,915 Card`

đã xuất hiện trong training:

`card overlap ≈ 99.157088%`.

Khoảng:

`99.442072%`

validation transaction nằm trên Card đã xuất hiện trong training rows.

Với S2 final test:

`1,517 / 1,525 User`

overlap với training:

`≈ 99.475410%`.

Khoảng:

`99.693895%`

final-test transaction thuộc User đã thấy trong training.

Đối với Card:

`3,837 / 3,908 Card`

overlap:

`≈ 98.183214%`.

Khoảng:

`98.774613%`

final-test transaction nằm trên Card đã xuất hiện trong training.

Việc W_LONG và W_SHORT cho cùng overlap statistic là một finding quan trọng.

W_LONG có thêm hơn 5 triệu training transaction so với W_SHORT, nhưng phần dữ liệu cũ hơn đó không làm tăng coverage của future User/Card theo các overlap metric đang audit.

Do đó không thể dùng lập luận:

`W_LONG tốt hơn vì nó bao phủ nhiều future entity hơn`

để chọn training window.

Nếu W_LONG hoặc W_SHORT cho model performance khác nhau sau này, nguyên nhân phải được tìm ở lượng training evidence, temporal recency, distribution hoặc learning behavior, chứ không phải ở future-entity overlap được đo trong M3.3.

Ngoài ra, overlap cao không được xem là leakage.

Raw User/Card không được dùng trực tiếp làm classifier feature, và temporal validity vẫn phụ thuộc việc mọi feature tại transaction T chỉ sử dụng thông tin xảy ra trước T.

#### Kết luận

Future entity overlap với cả hai training-window candidate đều rất cao.

Đối với S2:

Validation:

* User overlap ≈ `99.67%`;
* Card overlap ≈ `99.16%`.

Final test:

* User overlap ≈ `99.48%`;
* Card overlap ≈ `98.18%`.

`W_LONG` không tạo thêm future-entity overlap so với `W_SHORT` trong các partition đã audit.

Vì vậy:

`Entity coverage does not select the training-window winner.`

Việc chọn W_LONG hay W_SHORT tiếp tục phải được để `OPEN` cho model experiment.

`Entity overlap audit: PASS WITH FINDINGS`

## 9. Candidate boundary gate

Gate này chỉ kiểm tra correctness có thể xác định trước.

Nó **không tự quyết định** positive support đã “đủ tốt” hay chưa vì chưa khóa một ngưỡng tùy ý.

Câu hỏi “792 fraud có đủ hay không?”, “820 có đủ hay không?”... sẽ được thảo luận sau khi có output.



```python
gate_rows = []


for candidate_name, spec in (
    SPLIT_CANDIDATES.items()
):
    validation_state = (
        candidate_states[
            candidate_name
        ]["VALIDATION"]
    )

    test_state = (
        candidate_states[
            candidate_name
        ]["FINAL_TEST"]
    )

    validation_before_test = (
        spec[
            "validation_start"
        ]
        < spec[
            "test_start"
        ]
    )

    test_before_zero_fraud_break = (
        spec[
            "test_end"
        ]
        <= POST_BREAK_START
    )

    validation_nonempty = (
        validation_state[
            "row_count"
        ]
        > 0
    )

    test_nonempty = (
        test_state[
            "row_count"
        ]
        > 0
    )

    validation_has_positive = (
        validation_state[
            "fraud_count"
        ]
        > 0
    )

    test_has_positive = (
        test_state[
            "fraud_count"
        ]
        > 0
    )

    reservoir_rows_preserved = (
        validation_state[
            "row_count"
        ]
        + test_state[
            "row_count"
        ]
        == pre_break_2019_state[
            "row_count"
        ]
    )

    reservoir_fraud_preserved = (
        validation_state[
            "fraud_count"
        ]
        + test_state[
            "fraud_count"
        ]
        == pre_break_2019_state[
            "fraud_count"
        ]
    )

    configured_boundary_contiguous = (
        spec[
            "validation_start"
        ]
        == PRE_2019_START
        and spec[
            "test_end"
        ]
        == PRE_BREAK_END
    )

    gate_rows.append(
        {
            "candidate":
                candidate_name,

            "validation_before_test":
                validation_before_test,

            "test_ends_before_zero_fraud_regime":
                test_before_zero_fraud_break,

            "validation_nonempty":
                validation_nonempty,

            "test_nonempty":
                test_nonempty,

            "validation_has_positive":
                validation_has_positive,

            "test_has_positive":
                test_has_positive,

            "validation_plus_test_preserves_prebreak_rows":
                reservoir_rows_preserved,

            "validation_plus_test_preserves_prebreak_fraud":
                reservoir_fraud_preserved,

            "candidate_covers_full_2019_01_to_10_reservoir":
                configured_boundary_contiguous,
        }
    )


candidate_gate_df = pd.DataFrame(
    gate_rows
)

display(
    candidate_gate_df
)


BOOLEAN_GATE_COLUMNS = [
    column
    for column
    in candidate_gate_df.columns
    if column != "candidate"
]


candidate_gate_df[
    "all_structural_checks_pass"
] = (
    candidate_gate_df[
        BOOLEAN_GATE_COLUMNS
    ]
    .all(
        axis=1
    )
)


print("\nCandidate structural gate:")
display(
    candidate_gate_df[
        [
            "candidate",
            "all_structural_checks_pass",
        ]
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
      <th>candidate</th>
      <th>validation_before_test</th>
      <th>test_ends_before_zero_fraud_regime</th>
      <th>validation_nonempty</th>
      <th>test_nonempty</th>
      <th>validation_has_positive</th>
      <th>test_has_positive</th>
      <th>validation_plus_test_preserves_prebreak_rows</th>
      <th>validation_plus_test_preserves_prebreak_fraud</th>
      <th>candidate_covers_full_2019_01_to_10_reservoir</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>S1_4M_6M</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>S2_5M_5M</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>S3_6M_4M</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>


    
    Candidate structural gate:



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
      <th>candidate</th>
      <th>all_structural_checks_pass</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>S1_4M_6M</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>S2_5M_5M</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>S3_6M_4M</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>


### Nhận xét candidate boundary gate

#### Phân tích

Tất cả structural check của cả ba candidate đều trả về `True`.

Cụ thể, với S1, S2 và S3:

`validation_before_test = True`

xác nhận validation luôn xảy ra trước final test.

`test_ends_before_zero_fraud_regime = True`

xác nhận final test kết thúc trước `2019-11`, không đi vào zero-fraud regime đã phát hiện.

`validation_nonempty = True`

và:

`test_nonempty = True`

xác nhận không có partition rỗng.

`validation_has_positive = True`

và:

`test_has_positive = True`

xác nhận cả hai evaluation partition đều chứa fraud positive.

`validation_plus_test_preserves_prebreak_rows = True`

xác nhận hai partition cộng lại giữ đúng toàn bộ transaction trong `2019-01 → 2019-10`.

`validation_plus_test_preserves_prebreak_fraud = True`

xác nhận không fraud nào trong reservoir bị bỏ hoặc double-count do boundary.

`candidate_covers_full_2019_01_to_10_reservoir = True`

xác nhận mỗi candidate đều chia liên tục toàn bộ positive-bearing future reservoir.

Cuối cùng:

`all_structural_checks_pass = True`

cho cả S1, S2 và S3.

Do đó structural gate không loại candidate nào.

Việc chọn giữa ba candidate phải dựa trên experiment-design trade-off như validation/test duration, volume và positive support, chứ không phải lỗi temporal correctness.

#### Kết luận

```text
S1_4M_6M structural gate: PASS
S2_5M_5M structural gate: PASS
S3_6M_4M structural gate: PASS
```

Không candidate nào vi phạm temporal ordering, positive-class availability hoặc reservoir integrity.

Structural validity vì vậy không tạo khác biệt giữa ba candidate.

S2 chỉ được ưu tiên vì allocation validation/final-test cân bằng hơn, không phải vì S1 hoặc S3 sai.

## 10. Cross-milestone consistency gate

Các mốc dưới đây đã được M2 xác nhận trên artifact hiện tại.

M3.3 dùng chúng như regression/integrity check để bảo đảm notebook đang đọc đúng artifact và không có lỗi boundary.

Nếu một check fail, **dừng diễn giải M3.3** và kiểm tra lại code/data trước.



```python
consistency_checks = pd.Series(
    {
        "Total rows = 24,386,900":
            total_rows
            == 24_386_900,

        "Total fraud = 29,757":
            total_fraud
            == 29_757,

        "Timestamp parse failures = 0":
            timestamp_parse_failures
            == 0,

        "2019-01_to_2019-10 rows = 1,435,413":
            pre_break_2019_state[
                "row_count"
            ]
            == 1_435_413,

        "2019-01_to_2019-10 fraud = 2,087":
            pre_break_2019_state[
                "fraud_count"
            ]
            == 2_087,

        "2019-11_to_2020-02 rows = 625,025":
            post_break_state[
                "row_count"
            ]
            == 625_025,

        "2019-11_to_2020-02 fraud = 0":
            post_break_state[
                "fraud_count"
            ]
            == 0,

        "W_LONG 2015-2018 rows = 6,855,270":
            train_states[
                "W_LONG_2015_TO_2018"
            ][
                "row_count"
            ]
            == 6_855_270,

        "W_LONG 2015-2018 fraud = 9,606":
            train_states[
                "W_LONG_2015_TO_2018"
            ][
                "fraud_count"
            ]
            == 9_606,

        "W_SHORT 2018 rows = 1,721,615":
            train_states[
                "W_SHORT_2018_ONLY"
            ][
                "row_count"
            ]
            == 1_721_615,

        "W_SHORT 2018 fraud = 2,491":
            train_states[
                "W_SHORT_2018_ONLY"
            ][
                "fraud_count"
            ]
            == 2_491,
    }
)


display(
    consistency_checks
)


ALL_CONSISTENCY_CHECKS_PASS = bool(
    consistency_checks.all()
)


print(
    "\nALL_CONSISTENCY_CHECKS_PASS:",
    ALL_CONSISTENCY_CHECKS_PASS,
)


if not ALL_CONSISTENCY_CHECKS_PASS:
    raise RuntimeError(
        "M3.3 consistency gate FAIL. "
        "Không diễn giải split trước khi kiểm tra lại "
        "artifact / boundary / code."
    )

```


    Total rows = 24,386,900                True
    Total fraud = 29,757                   True
    Timestamp parse failures = 0           True
    2019-01_to_2019-10 rows = 1,435,413    True
    2019-01_to_2019-10 fraud = 2,087       True
    2019-11_to_2020-02 rows = 625,025      True
    2019-11_to_2020-02 fraud = 0           True
    W_LONG 2015-2018 rows = 6,855,270      True
    W_LONG 2015-2018 fraud = 9,606         True
    W_SHORT 2018 rows = 1,721,615          True
    W_SHORT 2018 fraud = 2,491             True
    dtype: bool


    
    ALL_CONSISTENCY_CHECKS_PASS: True


### Nhận xét consistency gate

#### Phân tích

Toàn bộ cross-milestone consistency check đều trả về `True`.

M3.3 tái xác nhận đúng các mốc dataset đã được khóa ở M2:

`Total rows = 24,386,900`

`Total fraud = 29,757`

`Timestamp parse failures = 0`

Vùng pre-break:

### `2019-01 → 2019-10`

`1,435,413 transaction / 2,087 fraud`.

Vùng post-break:

### `2019-11 → 2020-02`

`625,025 transaction / 0 fraud`.

Hai training-window family cũng khớp chính xác:

### `W_LONG 2015–2018`

`6,855,270 transaction / 9,606 fraud`.

### `W_SHORT 2018`

`1,721,615 transaction / 2,491 fraud`.

Kết quả cuối:

`ALL_CONSISTENCY_CHECKS_PASS = True`.

Điều này đặc biệt quan trọng vì các quyết định M3.3 phụ thuộc vào boundary theo thời gian.

Nếu một trong các mốc trên sai, ta phải nghi ngờ code mask, timestamp hoặc artifact trước khi diễn giải candidate split.

Nhưng output hiện tại không cho thấy bất kỳ inconsistency nào như vậy.

#### Kết luận

`Consistency gate: PASS`

Notebook M3.3 đang đọc đúng artifact và tái tạo đúng các finding temporal/training-window quan trọng từ M2.

Không phát hiện regression hoặc boundary-count inconsistency.

Output M3.3 đủ độ tin cậy để sử dụng làm evidence cho final split decision.


## 11. Compact handoff output

Cell này gom các bảng cần gửi lại để phân tích M3.3.

Sau khi `Run All`, gửi notebook đã chạy hoặc copy nguyên output của các phần:

1. Training-window summary
2. Candidate validation/final-test summary
3. Strict prior-history/cold-start
4. Entity overlap
5. Candidate structural gate
6. Consistency gate



```python
print(
    "=== TRAINING WINDOW SUMMARY ==="
)
display(
    train_summary_df
)

print(
    "\n=== CANDIDATE PARTITION SUMMARY ==="
)
display(
    split_summary_df
)

print(
    "\n=== STRICT PRIOR-HISTORY / COLD-START ==="
)
display(
    cold_start_df
)

print(
    "\n=== FUTURE ENTITY OVERLAP WITH TRAIN ==="
)
display(
    entity_overlap_df
)

print(
    "\n=== CANDIDATE STRUCTURAL GATE ==="
)
display(
    candidate_gate_df
)

print(
    "\n=== CONSISTENCY GATE ==="
)
display(
    consistency_checks
)

print(
    "\nALL_CONSISTENCY_CHECKS_PASS:",
    ALL_CONSISTENCY_CHECKS_PASS,
)

```

    === TRAINING WINDOW SUMMARY ===



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
      <th>train_window</th>
      <th>configured_start</th>
      <th>configured_end_exclusive</th>
      <th>actual_min_timestamp</th>
      <th>actual_max_timestamp</th>
      <th>transaction_count</th>
      <th>fraud_count</th>
      <th>fraud_rate_pct</th>
      <th>unique_users</th>
      <th>unique_user_card_pairs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>W_LONG_2015_TO_2018</td>
      <td>2015-01-01</td>
      <td>2019-01-01</td>
      <td>2015-01-01 00:01:00</td>
      <td>2018-12-31 23:58:00</td>
      <td>6855270</td>
      <td>9606</td>
      <td>0.140126</td>
      <td>1567</td>
      <td>4289</td>
    </tr>
    <tr>
      <th>1</th>
      <td>W_SHORT_2018_ONLY</td>
      <td>2018-01-01</td>
      <td>2019-01-01</td>
      <td>2018-01-01 00:03:00</td>
      <td>2018-12-31 23:58:00</td>
      <td>1721615</td>
      <td>2491</td>
      <td>0.144690</td>
      <td>1532</td>
      <td>4006</td>
    </tr>
  </tbody>
</table>
</div>


    
    === CANDIDATE PARTITION SUMMARY ===



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
      <th>candidate</th>
      <th>partition</th>
      <th>configured_start</th>
      <th>configured_end_exclusive</th>
      <th>actual_min_timestamp</th>
      <th>actual_max_timestamp</th>
      <th>transaction_count</th>
      <th>fraud_count</th>
      <th>fraud_rate_pct</th>
      <th>unique_users</th>
      <th>unique_user_card_pairs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>S1_4M_6M</td>
      <td>VALIDATION</td>
      <td>2019-01-01</td>
      <td>2019-05-01</td>
      <td>2019-01-01 00:02:00</td>
      <td>2019-04-30 23:58:00</td>
      <td>566810</td>
      <td>792</td>
      <td>0.139729</td>
      <td>1528</td>
      <td>3912</td>
    </tr>
    <tr>
      <th>1</th>
      <td>S1_4M_6M</td>
      <td>FINAL_TEST</td>
      <td>2019-05-01</td>
      <td>2019-11-01</td>
      <td>2019-05-01 00:02:00</td>
      <td>2019-10-31 23:59:00</td>
      <td>868603</td>
      <td>1295</td>
      <td>0.149090</td>
      <td>1526</td>
      <td>3913</td>
    </tr>
    <tr>
      <th>2</th>
      <td>S2_5M_5M</td>
      <td>VALIDATION</td>
      <td>2019-01-01</td>
      <td>2019-06-01</td>
      <td>2019-01-01 00:02:00</td>
      <td>2019-05-31 23:58:00</td>
      <td>712458</td>
      <td>1052</td>
      <td>0.147658</td>
      <td>1528</td>
      <td>3915</td>
    </tr>
    <tr>
      <th>3</th>
      <td>S2_5M_5M</td>
      <td>FINAL_TEST</td>
      <td>2019-06-01</td>
      <td>2019-11-01</td>
      <td>2019-06-01 00:02:00</td>
      <td>2019-10-31 23:59:00</td>
      <td>722955</td>
      <td>1035</td>
      <td>0.143162</td>
      <td>1525</td>
      <td>3908</td>
    </tr>
    <tr>
      <th>4</th>
      <td>S3_6M_4M</td>
      <td>VALIDATION</td>
      <td>2019-01-01</td>
      <td>2019-07-01</td>
      <td>2019-01-01 00:02:00</td>
      <td>2019-06-30 23:59:00</td>
      <td>854857</td>
      <td>1267</td>
      <td>0.148212</td>
      <td>1528</td>
      <td>3920</td>
    </tr>
    <tr>
      <th>5</th>
      <td>S3_6M_4M</td>
      <td>FINAL_TEST</td>
      <td>2019-07-01</td>
      <td>2019-11-01</td>
      <td>2019-07-01 00:00:00</td>
      <td>2019-10-31 23:59:00</td>
      <td>580556</td>
      <td>820</td>
      <td>0.141244</td>
      <td>1522</td>
      <td>3897</td>
    </tr>
  </tbody>
</table>
</div>


    
    === STRICT PRIOR-HISTORY / COLD-START ===



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
      <th>candidate</th>
      <th>partition</th>
      <th>new_user_count</th>
      <th>new_user_unique_pct</th>
      <th>transactions_on_new_users</th>
      <th>transactions_on_new_users_pct</th>
      <th>new_card_count</th>
      <th>new_card_unique_pct</th>
      <th>transactions_on_new_cards</th>
      <th>transactions_on_new_cards_pct</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>S1_4M_6M</td>
      <td>VALIDATION</td>
      <td>3</td>
      <td>0.196335</td>
      <td>632</td>
      <td>0.111501</td>
      <td>31</td>
      <td>0.792434</td>
      <td>2861</td>
      <td>0.504755</td>
    </tr>
    <tr>
      <th>1</th>
      <td>S1_4M_6M</td>
      <td>FINAL_TEST</td>
      <td>4</td>
      <td>0.262123</td>
      <td>536</td>
      <td>0.061708</td>
      <td>41</td>
      <td>1.047789</td>
      <td>3388</td>
      <td>0.390052</td>
    </tr>
    <tr>
      <th>2</th>
      <td>S2_5M_5M</td>
      <td>VALIDATION</td>
      <td>3</td>
      <td>0.196335</td>
      <td>794</td>
      <td>0.111445</td>
      <td>33</td>
      <td>0.842912</td>
      <td>3975</td>
      <td>0.557928</td>
    </tr>
    <tr>
      <th>3</th>
      <td>S2_5M_5M</td>
      <td>FINAL_TEST</td>
      <td>4</td>
      <td>0.262295</td>
      <td>536</td>
      <td>0.074140</td>
      <td>39</td>
      <td>0.997953</td>
      <td>3060</td>
      <td>0.423263</td>
    </tr>
    <tr>
      <th>4</th>
      <td>S3_6M_4M</td>
      <td>VALIDATION</td>
      <td>3</td>
      <td>0.196335</td>
      <td>954</td>
      <td>0.111598</td>
      <td>37</td>
      <td>0.943878</td>
      <td>5178</td>
      <td>0.605715</td>
    </tr>
    <tr>
      <th>5</th>
      <td>S3_6M_4M</td>
      <td>FINAL_TEST</td>
      <td>4</td>
      <td>0.262812</td>
      <td>536</td>
      <td>0.092325</td>
      <td>35</td>
      <td>0.898127</td>
      <td>2670</td>
      <td>0.459904</td>
    </tr>
  </tbody>
</table>
</div>


    
    === FUTURE ENTITY OVERLAP WITH TRAIN ===



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
      <th>candidate</th>
      <th>partition</th>
      <th>train_window</th>
      <th>future_unique_users</th>
      <th>user_overlap_count</th>
      <th>user_overlap_unique_pct</th>
      <th>future_tx_on_train_seen_users_pct</th>
      <th>future_unique_cards</th>
      <th>card_overlap_count</th>
      <th>card_overlap_unique_pct</th>
      <th>future_tx_on_train_seen_cards_pct</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>S1_4M_6M</td>
      <td>VALIDATION</td>
      <td>W_LONG_2015_TO_2018</td>
      <td>1528</td>
      <td>1523</td>
      <td>99.672775</td>
      <td>99.824633</td>
      <td>3912</td>
      <td>3881</td>
      <td>99.207566</td>
      <td>99.495245</td>
    </tr>
    <tr>
      <th>1</th>
      <td>S1_4M_6M</td>
      <td>VALIDATION</td>
      <td>W_SHORT_2018_ONLY</td>
      <td>1528</td>
      <td>1523</td>
      <td>99.672775</td>
      <td>99.824633</td>
      <td>3912</td>
      <td>3881</td>
      <td>99.207566</td>
      <td>99.495245</td>
    </tr>
    <tr>
      <th>2</th>
      <td>S1_4M_6M</td>
      <td>FINAL_TEST</td>
      <td>W_LONG_2015_TO_2018</td>
      <td>1526</td>
      <td>1518</td>
      <td>99.475754</td>
      <td>99.705734</td>
      <td>3913</td>
      <td>3842</td>
      <td>98.185535</td>
      <td>98.851834</td>
    </tr>
    <tr>
      <th>3</th>
      <td>S1_4M_6M</td>
      <td>FINAL_TEST</td>
      <td>W_SHORT_2018_ONLY</td>
      <td>1526</td>
      <td>1518</td>
      <td>99.475754</td>
      <td>99.705734</td>
      <td>3913</td>
      <td>3842</td>
      <td>98.185535</td>
      <td>98.851834</td>
    </tr>
    <tr>
      <th>4</th>
      <td>S2_5M_5M</td>
      <td>VALIDATION</td>
      <td>W_LONG_2015_TO_2018</td>
      <td>1528</td>
      <td>1523</td>
      <td>99.672775</td>
      <td>99.812340</td>
      <td>3915</td>
      <td>3882</td>
      <td>99.157088</td>
      <td>99.442072</td>
    </tr>
    <tr>
      <th>5</th>
      <td>S2_5M_5M</td>
      <td>VALIDATION</td>
      <td>W_SHORT_2018_ONLY</td>
      <td>1528</td>
      <td>1523</td>
      <td>99.672775</td>
      <td>99.812340</td>
      <td>3915</td>
      <td>3882</td>
      <td>99.157088</td>
      <td>99.442072</td>
    </tr>
    <tr>
      <th>6</th>
      <td>S2_5M_5M</td>
      <td>FINAL_TEST</td>
      <td>W_LONG_2015_TO_2018</td>
      <td>1525</td>
      <td>1517</td>
      <td>99.475410</td>
      <td>99.693895</td>
      <td>3908</td>
      <td>3837</td>
      <td>98.183214</td>
      <td>98.774613</td>
    </tr>
    <tr>
      <th>7</th>
      <td>S2_5M_5M</td>
      <td>FINAL_TEST</td>
      <td>W_SHORT_2018_ONLY</td>
      <td>1525</td>
      <td>1517</td>
      <td>99.475410</td>
      <td>99.693895</td>
      <td>3908</td>
      <td>3837</td>
      <td>98.183214</td>
      <td>98.774613</td>
    </tr>
    <tr>
      <th>8</th>
      <td>S3_6M_4M</td>
      <td>VALIDATION</td>
      <td>W_LONG_2015_TO_2018</td>
      <td>1528</td>
      <td>1523</td>
      <td>99.672775</td>
      <td>99.806634</td>
      <td>3920</td>
      <td>3883</td>
      <td>99.056122</td>
      <td>99.394285</td>
    </tr>
    <tr>
      <th>9</th>
      <td>S3_6M_4M</td>
      <td>VALIDATION</td>
      <td>W_SHORT_2018_ONLY</td>
      <td>1528</td>
      <td>1523</td>
      <td>99.672775</td>
      <td>99.806634</td>
      <td>3920</td>
      <td>3883</td>
      <td>99.056122</td>
      <td>99.394285</td>
    </tr>
    <tr>
      <th>10</th>
      <td>S3_6M_4M</td>
      <td>FINAL_TEST</td>
      <td>W_LONG_2015_TO_2018</td>
      <td>1522</td>
      <td>1514</td>
      <td>99.474376</td>
      <td>99.673244</td>
      <td>3897</td>
      <td>3827</td>
      <td>98.203746</td>
      <td>98.681264</td>
    </tr>
    <tr>
      <th>11</th>
      <td>S3_6M_4M</td>
      <td>FINAL_TEST</td>
      <td>W_SHORT_2018_ONLY</td>
      <td>1522</td>
      <td>1514</td>
      <td>99.474376</td>
      <td>99.673244</td>
      <td>3897</td>
      <td>3827</td>
      <td>98.203746</td>
      <td>98.681264</td>
    </tr>
  </tbody>
</table>
</div>


    
    === CANDIDATE STRUCTURAL GATE ===



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
      <th>candidate</th>
      <th>validation_before_test</th>
      <th>test_ends_before_zero_fraud_regime</th>
      <th>validation_nonempty</th>
      <th>test_nonempty</th>
      <th>validation_has_positive</th>
      <th>test_has_positive</th>
      <th>validation_plus_test_preserves_prebreak_rows</th>
      <th>validation_plus_test_preserves_prebreak_fraud</th>
      <th>candidate_covers_full_2019_01_to_10_reservoir</th>
      <th>all_structural_checks_pass</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>S1_4M_6M</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>S2_5M_5M</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>S3_6M_4M</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>


    
    === CONSISTENCY GATE ===



    Total rows = 24,386,900                True
    Total fraud = 29,757                   True
    Timestamp parse failures = 0           True
    2019-01_to_2019-10 rows = 1,435,413    True
    2019-01_to_2019-10 fraud = 2,087       True
    2019-11_to_2020-02 rows = 625,025      True
    2019-11_to_2020-02 fraud = 0           True
    W_LONG 2015-2018 rows = 6,855,270      True
    W_LONG 2015-2018 fraud = 9,606         True
    W_SHORT 2018 rows = 1,721,615          True
    W_SHORT 2018 fraud = 2,491             True
    dtype: bool


    
    ALL_CONSISTENCY_CHECKS_PASS: True


# 12. Tổng kết M3.3

## Observed facts

### Phân tích

M3.3 đã hoàn thành full-artifact temporal split audit trên toàn bộ `24,386,900 transaction`.

Tổng target được xác nhận lại là `29,757 fraud`, không có Timestamp parse failure.

Future positive-bearing reservoir trước temporal break được xác nhận:

`2019-01 → 2019-10`

gồm:

`1,435,413 transaction`
`2,087 fraud`.

Vùng:

`2019-11 → 2020-02`

gồm:

`625,025 transaction`
`0 fraud`.

Ba candidate S1/S2/S3 đều chia toàn bộ pre-break reservoir thành validation → final test theo đúng thứ tự thời gian.

### Kết luận

M3.3 có đầy đủ output thực tế để đánh giá candidate temporal split.

`Evidence sufficiency: CONFIRMED`

Không cần chạy thêm split audit trước khi đưa ra final boundary decision.

---

## So sánh positive support giữa các candidate

### Phân tích

Positive support của ba candidate là:

S1:

* validation: `792 fraud`;
* final test: `1,295 fraud`.

S2:

* validation: `1,052 fraud`;
* final test: `1,035 fraud`.

S3:

* validation: `1,267 fraud`;
* final test: `820 fraud`.

Tất cả partition đều có hàng trăm positive nên không có candidate nào mất khả năng đánh giá fraud class.

S2 phân bổ `2,087 fraud` gần cân bằng nhất giữa validation và final test.

### Kết luận

Cả ba candidate đều đạt yêu cầu cơ bản về positive support.

S2 cung cấp allocation cân bằng nhất:

`1,052 fraud validation / 1,035 fraud final test`.

`Positive-support preference: S2_5M_5M`

---

## So sánh transaction volume

### Phân tích

S1:

`566,810 validation / 868,603 final test`.

S2:

`712,458 validation / 722,955 final test`.

S3:

`854,857 validation / 580,556 final test`.

S2 có chênh lệch volume nhỏ nhất và đồng thời chia future reservoir thành hai khoảng 5 tháng bằng nhau.

### Kết luận

`S2_5M_5M` tạo allocation cân bằng nhất cả về thời lượng và transaction volume.

Không có bằng chứng hiện tại yêu cầu validation hoặc final test phải dài hơn bên còn lại.

---

## So sánh User/Card coverage

### Phân tích

Unique User và User+Card coverage giữa các candidate đều ở cùng quy mô.

S2 validation có:

`1,528 User`
`3,915 User+Card`.

S2 final test có:

`1,525 User`
`3,908 User+Card`.

Không xuất hiện sự suy giảm entity coverage đáng kể khi chuyển từ validation sang final test.

### Kết luận

S2 giữ entity coverage lớn và tương đối ổn định ở cả hai evaluation partition.

`Entity coverage: SUFFICIENT`

Không có entity-volume issue ngăn cản việc chọn S2.

---

## Cold-start / strict prior-history finding

### Phân tích

Strict cold-start là segment rất nhỏ.

Với S2 validation:

`3 new User`
`33 new Card`.

Chỉ khoảng:

`0.111445%` transaction thuộc new User

và:

`0.557928%` transaction thuộc new Card.

Với S2 final test:

`4 new User`
`39 new Card`.

Chỉ khoảng:

`0.074140%` transaction thuộc new User

và:

`0.423263%` transaction thuộc new Card.

### Kết luận

Main evaluation của project chủ yếu đo performance trên existing User/Card có prior historical context.

Completely-new User/Card là segment nhỏ và phải được ghi nhận như limitation.

`Cold-start blocking issue: NONE`

---

## Entity overlap finding

### Phân tích

S2 có future entity overlap rất cao với classifier-training rows.

Validation:

User overlap ≈ `99.672775%`.

Card overlap ≈ `99.157088%`.

Final test:

User overlap ≈ `99.475410%`.

Card overlap ≈ `98.183214%`.

Các giá trị này giống nhau giữa W_LONG và W_SHORT.

### Kết luận

W_LONG không cung cấp thêm future-entity coverage so với W_SHORT theo các overlap metric của M3.3.

Không được chọn training window dựa trên entity overlap.

`Training-window winner remains OPEN`.

---

## Temporal validity

### Phân tích

Cả ba candidate đều thỏa toàn bộ structural gate:

* validation trước final test;
* test kết thúc trước zero-fraud regime;
* không partition nào rỗng;
* cả validation và test đều có positive;
* không mất hoặc double-count transaction/fraud trong pre-break reservoir.

Consistency gate cũng PASS hoàn toàn.

### Kết luận

`Temporal validity: VERIFIED`

Không phát hiện temporal leakage hoặc partition-integrity issue trong candidate specification.

---

## Candidate bị loại

### Phân tích

S1 và S3 không bị loại do correctness failure.

S1 dành 4 tháng cho validation và 6 tháng cho final test, tạo allocation nghiêng nhiều về final test.

S3 làm điều ngược lại: validation dài 6 tháng và final test chỉ còn 4 tháng.

Cả hai vẫn là temporal split hợp lệ.

Tuy nhiên hiện không có yêu cầu phương pháp hoặc nghiệp vụ nào buộc phải ưu tiên validation hoặc final test theo hai hướng mất cân bằng này.

S2 đạt cùng structural validity nhưng chia đều 5 tháng/5 tháng và đồng thời tạo số transaction/fraud gần cân bằng.

### Kết luận

S1 và S3 được loại khỏi **final boundary selection**, nhưng không bị đánh dấu là invalid.

```text
S1_4M_6M:
VALID BUT NOT SELECTED

S3_6M_4M:
VALID BUT NOT SELECTED
```

Lý do không chọn là allocation kém cân bằng hơn S2, không phải do lỗi dữ liệu hoặc temporal correctness.

---

## Candidate còn đủ điều kiện

### Phân tích

S2 thỏa toàn bộ structural gate, consistency gate, positive support, entity coverage và cold-start requirements.

Nó tạo:

Validation:
`2019-01 → 2019-05`
`712,458 transaction`
`1,052 fraud`.

Final test:
`2019-06 → 2019-10`
`722,955 transaction`
`1,035 fraud`.

Đây là candidate cân bằng nhất trong candidate set đã được định nghĩa trước khi xem output.

### Kết luận

`S2_5M_5M: SELECTED`

S2 đủ evidence để trở thành final shared evaluation boundary của project.

---

## Final split decision

### Phân tích

Việc chọn S2 dựa trên:

* strict temporal ordering;
* tránh zero-fraud regime;
* validation/test đều có positive support lớn;
* 5 tháng cho mỗi partition;
* transaction volume gần cân bằng;
* fraud count gần cân bằng;
* entity coverage cao;
* cold-start nhỏ;
* structural gate PASS;
* consistency gate PASS.

Không có model nào được train và không có model-performance score nào được sử dụng để lựa chọn boundary.

Do đó việc chọn split không tiêu hao final test theo nghĩa model selection.

### Kết luận

```text
M3.3-D01

Selected split:
S2_5M_5M

TRAIN:
Timestamp < 2019-01-01
với training-window start vẫn OPEN giữa W_LONG và W_SHORT.

VALIDATION:
2019-01-01 ≤ Timestamp < 2019-06-01

FINAL TEST:
2019-06-01 ≤ Timestamp < 2019-11-01

Status:
LOCKED
```

---

## Training-window winner

### Phân tích

M3.3 xác nhận cả W_LONG và W_SHORT đều khả thi.

W_LONG có nhiều data/fraud hơn.

W_SHORT recent hơn và vẫn giữ sample size lớn.

Entity-overlap audit không tạo khác biệt giữa hai window.

Không có model-performance evidence trong M3.3.

### Kết luận

```text
W_LONG_2015_TO_2018:
CANDIDATE

W_SHORT_2018_ONLY:
CANDIDATE

Training-window winner:
OPEN
```

Winner chỉ được xác định qua experiment công bằng ở bước sau.

---

## M3.3 Gate

### Phân tích

M3.3 đã hoàn tất:

* full-artifact execution;
* Timestamp integrity;
* training-window size audit;
* candidate partition audit;
* positive support audit;
* strict cold-start audit;
* entity-overlap audit;
* structural gate;
* cross-milestone consistency gate;
* final boundary comparison.

Không còn câu hỏi bắt buộc nào của M3.3 cần thêm một lần quét dữ liệu trước khi khóa evaluation boundary.

Training-window winner còn OPEN là có chủ ý và nằm ngoài nhiệm vụ phải giải quyết tại M3.3.

### Kết luận

```text
Full-artifact execution:
PASS

Timestamp integrity:
PASS

Candidate structural gate:
PASS

Cross-milestone consistency:
PASS

Positive support:
PASS

Entity / cold-start audit:
PASS WITH FINDINGS

Final evaluation boundary:
LOCKED

Training-window winner:
OPEN — đúng phạm vi

Blocking issue:
NONE

M3.3 Gate:
PASS
```

