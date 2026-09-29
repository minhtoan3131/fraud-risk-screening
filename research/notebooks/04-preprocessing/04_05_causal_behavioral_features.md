# M4.5 — Triển khai causal behavioral features

## Vai trò

M4.4 đã khóa Transaction-level Feature Specification ở mức candidate.

M4.5 bổ sung nhóm feature mô tả hành vi lịch sử của transaction hiện tại.

## Câu hỏi trung tâm

> Những historical feature nào có thể được triển khai ổn định, causal và đủ rõ về semantic để đưa vào behavioral candidate set?

## Behavioral prototype kế thừa từ M2.8

1. time_since_previous_transaction_min

2. transactions_last_1h

3. amount_minus_previous_mean

4. is_new_merchant

Các feature trên đều được xây ở cấp:

User + Card

Merchant Name chỉ được sử dụng làm history state cho merchant novelty.

## Strict causal rule

History hợp lệ khi và chỉ khi:

Timestamp(history) < Timestamp(current)

Không được dùng:

- current transaction làm history của chính nó;
- transaction cùng Timestamp;
- future transaction;
- target label của transaction trước;
- target label của transaction tương lai;
- lifetime aggregate có chứa future rows.

## Historical warm-up

Classifier-training rows và history-context rows là hai khái niệm khác nhau.

Transaction trước modeling window có thể được dùng làm history nếu:

Timestamp(history) < Timestamp(current)

Do đó M4.5 không reset history tại đầu W_LONG, W_SHORT hoặc VALIDATION.

## Không thuộc M4.5

- categorical encoding;
- scaling;
- learned imputation;
- resampling;
- model training;
- feature-performance comparison;
- threshold;
- FINAL TEST feature analysis.

Các learned preprocessing operation thuộc M4.6.

## Trạng thái khi bắt đầu

M4.4:
PASS

Transaction-level Feature Specification:
LOCKED AT CANDIDATE LEVEL

Behavioral Feature Contract:
OPEN

M4.5 Gate:
OPEN

# 1. Evidence kế thừa và ranh giới M4.5

M4.5 không khám phá behavioral feature từ đầu.

## Evidence M2.8

M2.8 đã xác nhận:

- raw artifact có 6,139 User+Card;
- User+Card nằm trong contiguous block trên current artifact;
- Timestamp không giảm bên trong Card;
- same-timestamp transaction thực sự tồn tại;
- causal streaming khả thi;
- bốn behavioral prototype có thể tính đúng;
- historical warm-up làm giảm artificial cold-start;
- prior Card history coverage rất cao;
- short-window activity có cấu trúc zero-heavy;
- merchant novelty không phải feature gần constant.

M2.8 mới chứng minh:

computability
+
coverage
+
distribution

M2.8 chưa chứng minh:

classification benefit.

## Evidence M3

M3 đã khóa:

Historical warm-up:
ALLOWED

Classifier-training membership:
INDEPENDENT FROM HISTORY AVAILABILITY

Strict history:

Timestamp(history) < Timestamp(current)

Same-timestamp history:
PROHIBITED

Target-label history:
PROHIBITED

## Evidence M4.4

M4.4 đã khóa transaction-level candidate frame.

M4.5 không thay đổi các feature M4.4.

M4.5 chỉ bổ sung behavioral candidates.

## Phạm vi behavioral v1

M4.5 ưu tiên bốn prototype đã có bằng chứng implementation từ M2.8.

Không tự động mở rộng sang:

- user-level lifetime aggregates;
- transactions_last_10m;
- rolling Amount 30d;
- is_new_mcc;
- location novelty;
- các feature khác

trước khi bốn prototype lõi được tích hợp và audit đúng.

## Exact historical lookback

M3 vẫn để câu hỏi historical lookback tổng quát ở trạng thái OPEN.

Current M4.5 baseline candidate sử dụng:

time_since_previous_transaction
→ previous strict timestamp gần nhất

transactions_last_1h
→ [T - 1 giờ, T)

previous_amount_mean
→ toàn bộ strict prior Card history

is_new_merchant
→ toàn bộ strict prior Merchant history trên Card

Đây là baseline behavioral contract cần audit.

Nó không phải tuyên bố rằng full prior history là lookback tối ưu về model performance.

# 2. Thiết lập môi trường


```python
from pathlib import Path
from collections import Counter
import time

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


EXPECTED_FILE_SIZE = (
    2_354_626_737
)

CHUNK_SIZE = 500_000

# Deterministic ~1% sample:
# raw_row_id chia hết cho 100.
FEATURE_SAMPLE_MOD = 100


print("PROJECT_ROOT:")
print(PROJECT_ROOT)

print("\nDATA_PATH:")
print(DATA_PATH)

print("\nFile exists:")
print(DATA_PATH.exists())

print("\nFile size:")
print(DATA_PATH.stat().st_size)

print("\nCHUNK_SIZE:")
print(CHUNK_SIZE)
```

    PROJECT_ROOT:
    /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening
    
    DATA_PATH:
    /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/raw/ibm_tabformer/card_transaction.v1.csv
    
    File exists:
    True
    
    File size:
    2354626737
    
    CHUNK_SIZE:
    500000



```python
assert DATA_PATH.exists()

assert (
    DATA_PATH.stat().st_size
    == EXPECTED_FILE_SIZE
)


print(
    "M4.5 ENVIRONMENT / ARTIFACT GATE: PASS"
)
```

    M4.5 ENVIRONMENT / ARTIFACT GATE: PASS


### Nhận xét

Notebook xác định đúng project root và raw artifact:

`data/raw/ibm_tabformer/card_transaction.v1.csv`

File tồn tại và có kích thước:

`2,354,626,737 bytes`

khớp artifact đã được sử dụng xuyên M4.2–M4.4.

Chunk size:

`500,000`

Environment gate trả về:

`M4.5 ENVIRONMENT / ARTIFACT GATE: PASS`

### Kết luận

M4.5 đang làm việc trên đúng raw artifact đã khóa.

Status:

`PASS`

Blocking issue:

`NONE`

# 3. Temporal contract


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
    "TRAIN_2015_2017":
        5_133_655,

    "TRAIN_2018":
        1_721_615,

    "VALIDATION":
        712_458,
}


EXPECTED_DEVELOPMENT_ROWS = (
    7_567_728
)

EXPECTED_CONTEXT_ROWS_BEFORE_VALIDATION_END = (
    23_038_920
)

EXPECTED_PRE_W_LONG_HISTORY_ROWS = (
    15_471_192
)


def assign_development_period(
    timestamp,
):
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
            default="OUTSIDE_M45_DEVELOPMENT",
        ),
        index=timestamp.index,
        dtype="string",
    )
```

# 4. Canonical parsing


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
```

# 5. M4.5.1 — Audit physical order cho causal streaming

## Câu hỏi

Current raw artifact có còn đáp ứng các invariant cần thiết để stream theo User+Card hay không?

Cần kiểm tra:

1. Timestamp parse không lỗi.

2. Mỗi User+Card tạo đúng một contiguous block.

3. Card block đã đóng không xuất hiện trở lại.

4. Timestamp không giảm bên trong User+Card.

5. Ghi nhận same-timestamp adjacency.

## Quan trọng

Raw CSV không globally chronological.

M4.5 chỉ được tận dụng physical order vì current artifact có cấu trúc thuận lợi ở cấp User+Card.

Nếu gate fail:

STOP

Không được sửa bằng cách giả định raw row order đúng.

Implementation tổng quát phải explicit sort:

User
+
Card
+
Timestamp


```python
ORDER_USECOLS = [
    "User",
    "Card",
    "Year",
    "Month",
    "Day",
    "Time",
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


for chunk_number, chunk in enumerate(
    pd.read_csv(
        DATA_PATH,
        usecols=ORDER_USECOLS,
        chunksize=CHUNK_SIZE,
    ),
    start=1,
):
    total_rows += len(chunk)

    timestamp = (
        build_timestamp(
            chunk
        )
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


    # ========================================================
    # Internal order trong từng chunk
    # ========================================================

    timestamp_diff = (
        audit_df
        .groupby(
            [
                "User",
                "Card",
            ],
            sort=False,
        )[
            "Timestamp"
        ]
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


    # ========================================================
    # Cross-chunk temporal order
    # ========================================================

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


    # ========================================================
    # Contiguous Card blocks
    # ========================================================

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

        block_starts = (
            np.flatnonzero(
                change
            )
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


    if (
        chunk_number % 10 == 0
        or len(chunk) < CHUNK_SIZE
    ):
        print(
            f"Chunk {chunk_number:02d} | "
            f"rows = {total_rows:,}"
        )


print(
    "\nOrder audit hoàn tất."
)

print(
    "Total rows:",
    f"{total_rows:,}",
)

print(
    "Timestamp parse failures:",
    timestamp_parse_failures,
)

print(
    "Unique User+Card:",
    len(all_card_keys),
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

    Chunk 10 | rows = 5,000,000
    Chunk 20 | rows = 10,000,000
    Chunk 30 | rows = 15,000,000
    Chunk 40 | rows = 20,000,000
    Chunk 49 | rows = 24,386,900
    
    Order audit hoàn tất.
    Total rows: 24,386,900
    Timestamp parse failures: 0
    Unique User+Card: 6139
    Card block count: 6139
    Card block reappearance: 0
    Within-card timestamp decreases: 0
    Adjacent equal timestamps within Card: 142010



```python
assert total_rows == 24_386_900

assert timestamp_parse_failures == 0

assert len(
    all_card_keys
) == 6_139

assert (
    card_block_count
    == 6_139
)

assert (
    card_block_reappearance_count
    == 0
)

assert (
    within_card_timestamp_decreases
    == 0
)


CAUSAL_STREAMING_SAFE = True


print(
    "M4.5 CAUSAL STREAMING ORDER GATE: PASS"
)
```

    M4.5 CAUSAL STREAMING ORDER GATE: PASS


### Nhận xét

Full order audit đã quét đủ:

`24,386,900 transactions`

và xác nhận:

`Timestamp parse failures = 0`

`Unique User+Card = 6,139`

`Card block count = 6,139`

`Card block reappearance = 0`

`Within-card timestamp decreases = 0`

Như vậy current artifact vẫn giữ đúng cấu trúc vật lý đã quan sát ở M2.8:

mỗi `User+Card` tạo một contiguous block và Timestamp không giảm bên trong block.

Điều này cho phép causal streaming trên current artifact mà không phải sort toàn bộ dataset.

Tuy nhiên audit cũng xác nhận:

`142,010 adjacent equal-timestamp pairs`

trong Card history.

Con số này cho thấy same-timestamp handling không phải edge case lý thuyết. Dataset thực sự chứa nhiều transaction có cùng prediction time.

Do đó nếu dùng row order hoặc `shift()` đơn giản mà không group theo timestamp, pipeline có thể vô tình cho transaction cùng timestamp làm history của nhau.

Cell gate trả về:

`M4.5 CAUSAL STREAMING ORDER GATE: PASS`

### Kết luận M4.5.1

Current artifact đủ điều kiện causal streaming ở cấp:

`User + Card`

với điều kiện bắt buộc:

`Timestamp(history) < Timestamp(current)`

Same-timestamp peer:

`EXCLUDED`

Raw physical-order optimization:

`VALID FOR CURRENT VERIFIED ARTIFACT`

Không được coi đây là assumption chung cho artifact khác.

Status:

`PASS`

# 6. Stream từng User+Card


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


assert "Is Fraud?" not in FEATURE_USECOLS
assert "Errors?" not in FEATURE_USECOLS


def iter_card_blocks(
    data_path,
    chunksize,
):
    pending_key = None
    pending_parts = []

    raw_row_offset = 0

    for chunk in pd.read_csv(
        data_path,
        usecols=FEATURE_USECOLS,
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
            chunk[
                "Timestamp"
            ]
            .isna()
            .any()
        ):
            raise ValueError(
                "Timestamp parse failure "
                "trong causal feature pass."
            )

        if (
            chunk[
                "Amount_numeric"
            ]
            .isna()
            .any()
        ):
            raise ValueError(
                "Amount parse failure "
                "trong causal feature pass."
            )
        if (
            chunk[
                "Merchant Name"
            ]
            .isna()
            .any()
        ):
            raise ValueError(
                "Merchant Name missing "
                "trong causal feature pass."
            )

        working = (
            chunk[
                [
                    "raw_row_id",
                    "User",
                    "Card",
                    "Timestamp",
                    "Amount_numeric",
                    "Merchant Name",
                ]
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


    if pending_key is not None:
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

# 7. Strict-causal behavioral builder

## Feature semantics

### prior_card_transaction_count

Số transaction của cùng Card có:

Timestamp < T

Field này chủ yếu dùng để audit history depth / cold-start.

Chưa mặc định coi là classifier candidate.

### has_prior_card_history

True nếu tồn tại ít nhất một Card transaction với:

Timestamp < T

Đây là companion state dùng để biểu diễn cold-start rõ ràng.

### time_since_previous_transaction_min

Khoảng cách phút tới distinct previous timestamp gần nhất.

Same-timestamp peer bị loại.

Cold-start:
NA.

### transactions_last_1h

Số prior Card transaction thuộc:

[T - 1 giờ, T)

Zero là behavioral value hợp lệ.

### previous_amount_mean

Mean Amount của toàn bộ Card transaction có:

Timestamp < T

Dùng làm support state để xây Amount deviation.

### amount_minus_previous_mean

current Amount
-
previous_amount_mean

Cold-start:
NA.

### is_new_merchant

True khi Merchant Name hiện tại chưa xuất hiện trên Card tại bất kỳ:

Timestamp < T

Nếu nhiều transaction cùng merchant xuất hiện tại first timestamp của merchant:

tất cả đều True.

Peer cùng timestamp không được dùng làm history.


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


    # ========================================================
    # Distinct timestamp groups
    # ========================================================

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


    # ========================================================
    # Strict prior count
    # ========================================================

    prior_transaction_count_group = (
        group_starts
    )

    prior_transaction_count = (
        np.repeat(
            prior_transaction_count_group,
            group_lengths,
        )
    )

    has_prior_card_history = (
        prior_transaction_count
        > 0
    )


    # ========================================================
    # Recency
    # ========================================================

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

    time_since_previous_min = (
        np.repeat(
            time_since_previous_group,
            group_lengths,
        )
    )


    # ========================================================
    # 1-hour velocity
    # [T - 1h, T)
    # ========================================================

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


    # ========================================================
    # Previous Amount mean
    # ========================================================

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

    previous_amount_mean = (
        np.repeat(
            previous_amount_mean_group,
            group_lengths,
        )
    )

    amount_minus_previous_mean = (
        amount
        - previous_amount_mean
    )


    # ========================================================
    # Merchant novelty — strict causal implementation
    # ========================================================

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

        # Quan trọng:
        # chỉ kiểm tra state được tích lũy
        # từ timestamp NHỎ HƠN current timestamp.
        #
        # Không update seen_merchants trước khi
        # tính feature cho toàn bộ timestamp group.
        # Vì vậy các peer cùng timestamp không
        # được làm history cho nhau.
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

        # Chỉ sau khi feature của toàn group
        # đã được tính xong mới cập nhật history.
        seen_merchants.update(
            current_group_merchants.tolist()
        )


    # ========================================================
    # Same-timestamp audit metadata
    # ========================================================

    timestamp_group_size = (
        np.repeat(
            group_lengths,
            group_lengths,
        )
    )


    feature_df = pd.DataFrame(
        {
            "raw_row_id":
                card_df[
                    "raw_row_id"
                ]
                .to_numpy(),

            "Timestamp":
                card_df[
                    "Timestamp"
                ]
                .to_numpy(),

            "Amount_numeric":
                amount,

            "prior_card_transaction_count":
                prior_transaction_count,

            "has_prior_card_history":
                has_prior_card_history,

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

            "timestamp_group_size":
                timestamp_group_size,
        }
    )

    return feature_df
```

# 8. M4.5.2 — Deterministic causal unit test

Test nhỏ được tạo thủ công để xác minh strict-causal semantics.

Timeline:

09:00
Merchant A
Amount 10

10:00
Merchant B
Amount 20

10:00
Merchant B
Amount 30

10:30
Merchant A
Amount 40

Hai transaction 10:00:

- không được nhìn thấy nhau;
- đều chỉ được nhìn thấy 09:00;
- Merchant B phải được coi là new cho cả hai.

Transaction 10:30:

- được nhìn thấy cả hai transaction 10:00;
- Merchant A không còn new.


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


causal_test_features = (
    compute_causal_behavioral_features(
        causal_test_df
    )
)


display(
    causal_test_features
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
      <th>raw_row_id</th>
      <th>Timestamp</th>
      <th>Amount_numeric</th>
      <th>prior_card_transaction_count</th>
      <th>has_prior_card_history</th>
      <th>time_since_previous_transaction_min</th>
      <th>transactions_last_1h</th>
      <th>previous_amount_mean</th>
      <th>amount_minus_previous_mean</th>
      <th>is_new_merchant</th>
      <th>timestamp_group_size</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>2018-01-01 09:00:00</td>
      <td>10.0</td>
      <td>0</td>
      <td>False</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2018-01-01 10:00:00</td>
      <td>20.0</td>
      <td>1</td>
      <td>True</td>
      <td>60.0</td>
      <td>1</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>True</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>2018-01-01 10:00:00</td>
      <td>30.0</td>
      <td>1</td>
      <td>True</td>
      <td>60.0</td>
      <td>1</td>
      <td>10.0</td>
      <td>20.0</td>
      <td>True</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>2018-01-01 10:30:00</td>
      <td>40.0</td>
      <td>3</td>
      <td>True</td>
      <td>30.0</td>
      <td>2</td>
      <td>20.0</td>
      <td>20.0</td>
      <td>False</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



```python
np.testing.assert_array_equal(
    causal_test_features[
        "prior_card_transaction_count"
    ]
    .to_numpy(),
    np.array(
        [
            0,
            1,
            1,
            3,
        ]
    ),
)


np.testing.assert_allclose(
    causal_test_features[
        "time_since_previous_transaction_min"
    ]
    .to_numpy(),
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
    causal_test_features[
        "transactions_last_1h"
    ]
    .to_numpy(),
    np.array(
        [
            0,
            1,
            1,
            2,
        ]
    ),
)


np.testing.assert_allclose(
    causal_test_features[
        "previous_amount_mean"
    ]
    .to_numpy(),
    np.array(
        [
            np.nan,
            10.0,
            10.0,
            20.0,
        ]
    ),
    equal_nan=True,
)


np.testing.assert_allclose(
    causal_test_features[
        "amount_minus_previous_mean"
    ]
    .to_numpy(),
    np.array(
        [
            np.nan,
            10.0,
            20.0,
            20.0,
        ]
    ),
    equal_nan=True,
)


np.testing.assert_array_equal(
    causal_test_features[
        "is_new_merchant"
    ]
    .to_numpy(),
    np.array(
        [
            True,
            True,
            True,
            False,
        ]
    ),
)


assert (
    causal_test_features
    .loc[
        1,
        "previous_amount_mean"
    ]
    ==
    causal_test_features
    .loc[
        2,
        "previous_amount_mean"
    ]
)


M45_CAUSAL_UNIT_TEST_PASS = True


print(
    "M4.5 STRICT CAUSAL UNIT TEST: PASS"
)
```

    M4.5 STRICT CAUSAL UNIT TEST: PASS


### Nhận xét

Unit test chủ động tạo hai transaction tại cùng:

`10:00`

để kiểm tra strict-causal semantics.

Hai transaction 10:00 đều nhận:

`prior_card_transaction_count = 1`

thay vì một transaction nhìn thấy transaction 10:00 còn lại.

Cả hai có:

`time_since_previous_transaction_min = 60`

`transactions_last_1h = 1`

`previous_amount_mean = 10`

Điều này xác nhận toàn bộ historical state của chúng chỉ được tính từ transaction 09:00.

Đối với Merchant B, cả hai transaction 10:00 đều có:

`is_new_merchant = True`

Đây là behavior đúng.

Dù hai transaction cùng merchant xảy ra tại cùng timestamp đầu tiên, chúng không được dùng làm history của nhau.

Transaction 10:30 có:

`prior_card_transaction_count = 3`

`transactions_last_1h = 2`

`previous_amount_mean = 20`

`is_new_merchant = False`

cho Merchant A vì Merchant A đã thực sự tồn tại tại timestamp nhỏ hơn 10:30.

Cell assertion trả về:

`M4.5 STRICT CAUSAL UNIT TEST: PASS`

Đặc biệt, implementation `is_new_merchant` hiện cập nhật `seen_merchants` chỉ sau khi feature cho toàn timestamp group đã được tính xong.

Do đó implementation hiện tại không dùng future-inclusive `first timestamp` aggregate và không cho same-timestamp peer làm history.

### Kết luận M4.5.2

Strict causal semantics đã được xác minh trực tiếp bằng deterministic unit test.

Current transaction:

`EXCLUDED FROM HISTORY`

Same-timestamp peer:

`EXCLUDED FROM HISTORY`

Future transaction:

`EXCLUDED FROM HISTORY`

Merchant novelty:

`STRICT-CAUSAL IMPLEMENTATION VERIFIED`

Status:

`PASS`

# 9. Full causal pass — accumulators


```python
coverage_state = {
    period: {
        "row_count": 0,

        "prior_history_count": 0,

        "time_since_available_count": 0,

        "last_1h_positive_count": 0,

        "previous_amount_mean_available_count": 0,

        "amount_deviation_available_count": 0,

        "new_merchant_count": 0,

        "same_timestamp_row_count": 0,
    }

    for period in PERIODS
}


WARMUP_WINDOWS = {
    "W_LONG_TRAIN": (
        W_LONG_START,
        TRAIN_END,
    ),

    "W_SHORT_TRAIN": (
        W_SHORT_START,
        TRAIN_END,
    ),

    "VALIDATION": (
        TRAIN_END,
        VALIDATION_END,
    ),
}


warmup_state = {
    name: {
        "row_count": 0,

        "cold_start_with_warmup": 0,

        "cold_start_if_reset": 0,
    }

    for name
    in WARMUP_WINDOWS
}


feature_sample_parts = []


feature_cards_processed = 0

feature_raw_rows_seen = 0

history_context_rows_processed = 0

pre_w_long_history_rows = 0

development_rows_emitted = 0


max_full_card_block_rows = 0

max_context_card_block_rows = 0


development_max_feature_timestamp = None

development_min_feature_timestamp = None
```

# 10. M4.5.3 — Full causal behavioral pass

## Cách sử dụng history

M4.5 phải đọc history cũ để tạo state cho development rows.

History context:

Timestamp < 2019-06-01

Development feature rows:

2015-01-01 <= Timestamp < 2019-06-01

Do đó các transaction trước 2015:

- không phải classifier-development rows;
- nhưng được phép tham gia historical warm-up.

FINAL TEST:

Timestamp >= 2019-06-01

không được đưa vào behavioral feature computation phục vụ M4.5 development decision.

## Quan trọng

Trong VALIDATION, một transaction có thể sử dụng các transaction VALIDATION xảy ra sớm hơn làm causal event history.

Không sử dụng label của các transaction đó.

Đây là event-history availability, không phải model fitting.


```python
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

    feature_raw_rows_seen += len(
        card_df
    )

    max_full_card_block_rows = max(
        max_full_card_block_rows,
        len(
            card_df
        ),
    )


    # ========================================================
    # Chỉ history trước VALIDATION_END
    # ========================================================

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

    if context_card_df.empty:
        continue


    history_context_rows_processed += len(
        context_card_df
    )

    max_context_card_block_rows = max(
        max_context_card_block_rows,
        len(
            context_card_df
        ),
    )

    pre_w_long_history_rows += int(
        (
            context_card_df[
                "Timestamp"
            ]
            < W_LONG_START
        ).sum()
    )


    # ========================================================
    # Compute strict-causal feature
    # ========================================================

    feature_df = (
        compute_causal_behavioral_features(
            context_card_df
        )
    )


    prior_count = (
        feature_df[
            "prior_card_transaction_count"
        ]
    )

    has_prior = (
        feature_df[
            "has_prior_card_history"
        ]
    )

    time_since = (
        feature_df[
            "time_since_previous_transaction_min"
        ]
    )

    last_1h = (
        feature_df[
            "transactions_last_1h"
        ]
    )

    previous_mean = (
        feature_df[
            "previous_amount_mean"
        ]
    )

    amount_deviation = (
        feature_df[
            "amount_minus_previous_mean"
        ]
    )


    # ========================================================
    # Per-card causal assertions
    # ========================================================

    assert (
        prior_count
        >= 0
    ).all()

    assert (
        last_1h
        >= 0
    ).all()

    assert (
        last_1h
        <= prior_count
    ).all()


    # Cold-start semantics
    no_prior = (
        ~has_prior
    )

    assert (
        prior_count.loc[
            no_prior
        ]
        == 0
    ).all()

    assert (
        time_since.loc[
            no_prior
        ]
        .isna()
        .all()
    )

    assert (
        previous_mean.loc[
            no_prior
        ]
        .isna()
        .all()
    )

    assert (
        amount_deviation.loc[
            no_prior
        ]
        .isna()
        .all()
    )

    assert (
        last_1h.loc[
            no_prior
        ]
        == 0
    ).all()

    assert (
        feature_df.loc[
            no_prior,
            "is_new_merchant",
        ]
        .all()
    )


    # Prior-history semantics
    if has_prior.any():
        assert (
            time_since.loc[
                has_prior
            ]
            .notna()
            .all()
        )

        assert (
            time_since.loc[
                has_prior
            ]
            > 0
        ).all()

        assert (
            previous_mean.loc[
                has_prior
            ]
            .notna()
            .all()
        )

        assert (
            amount_deviation.loc[
                has_prior
            ]
            .notna()
            .all()
        )


    # Amount-deviation identity
    if has_prior.any():
        expected_deviation = (
            feature_df.loc[
                has_prior,
                "Amount_numeric",
            ]
            -
            feature_df.loc[
                has_prior,
                "previous_amount_mean",
            ]
        )

        np.testing.assert_allclose(
            feature_df.loc[
                has_prior,
                "amount_minus_previous_mean",
            ]
            .to_numpy(),
            expected_deviation
            .to_numpy(),
        )


    # ========================================================
    # Historical warm-up audit
    # ========================================================

    for (
        window_name,
        (
            window_start,
            window_end,
        ),
    ) in (
        WARMUP_WINDOWS.items()
    ):
        window_mask = (
            (
                feature_df[
                    "Timestamp"
                ]
                >= window_start
            )
            &
            (
                feature_df[
                    "Timestamp"
                ]
                < window_end
            )
        )

        if not window_mask.any():
            continue

        window_timestamp = (
            feature_df.loc[
                window_mask,
                "Timestamp",
            ]
        )

        first_timestamp_in_window = (
            window_timestamp.min()
        )

        reset_cold_count = int(
            window_timestamp
            .eq(
                first_timestamp_in_window
            )
            .sum()
        )

        actual_cold_count = int(
            (
                feature_df.loc[
                    window_mask,
                    "prior_card_transaction_count",
                ]
                == 0
            ).sum()
        )

        state = (
            warmup_state[
                window_name
            ]
        )

        state[
            "row_count"
        ] += int(
            window_mask.sum()
        )

        state[
            "cold_start_with_warmup"
        ] += (
            actual_cold_count
        )

        state[
            "cold_start_if_reset"
        ] += (
            reset_cold_count
        )


    # ========================================================
    # Development rows
    # ========================================================

    period = (
        assign_development_period(
            feature_df[
                "Timestamp"
            ]
        )
    )

    development_mask = (
        period
        != "OUTSIDE_M45_DEVELOPMENT"
    )

    if not development_mask.any():
        continue


    dev_feature_df = (
        feature_df.loc[
            development_mask
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    dev_period = (
        period.loc[
            development_mask
        ]
        .reset_index(
            drop=True
        )
    )

    development_rows_emitted += len(
        dev_feature_df
    )


    current_min_timestamp = (
        dev_feature_df[
            "Timestamp"
        ]
        .min()
    )

    current_max_timestamp = (
        dev_feature_df[
            "Timestamp"
        ]
        .max()
    )

    if (
        development_min_feature_timestamp
        is None
        or current_min_timestamp
        < development_min_feature_timestamp
    ):
        development_min_feature_timestamp = (
            current_min_timestamp
        )

    if (
        development_max_feature_timestamp
        is None
        or current_max_timestamp
        > development_max_feature_timestamp
    ):
        development_max_feature_timestamp = (
            current_max_timestamp
        )


    # ========================================================
    # Coverage theo period
    # ========================================================

    for period_name in PERIODS:
        mask = (
            dev_period
            .eq(
                period_name
            )
        )

        if not mask.any():
            continue

        sub = (
            dev_feature_df.loc[
                mask
                .to_numpy()
            ]
        )

        state = (
            coverage_state[
                period_name
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
            sub[
                "has_prior_card_history"
            ]
            .sum()
        )

        state[
            "time_since_available_count"
        ] += int(
            sub[
                "time_since_previous_transaction_min"
            ]
            .notna()
            .sum()
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
            "previous_amount_mean_available_count"
        ] += int(
            sub[
                "previous_amount_mean"
            ]
            .notna()
            .sum()
        )

        state[
            "amount_deviation_available_count"
        ] += int(
            sub[
                "amount_minus_previous_mean"
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

        state[
            "same_timestamp_row_count"
        ] += int(
            (
                sub[
                    "timestamp_group_size"
                ]
                > 1
            ).sum()
        )


    # ========================================================
    # Deterministic ~1% sample
    # ========================================================

    sample_mask = (
        (
            dev_feature_df[
                "raw_row_id"
            ]
            % FEATURE_SAMPLE_MOD
        )
        == 0
    )

    if sample_mask.any():
        sample_part = (
            dev_feature_df.loc[
                sample_mask,
                [
                    "raw_row_id",
                    "Timestamp",
                    "prior_card_transaction_count",
                    "has_prior_card_history",
                    "time_since_previous_transaction_min",
                    "transactions_last_1h",
                    "previous_amount_mean",
                    "amount_minus_previous_mean",
                    "is_new_merchant",
                    "timestamp_group_size",
                ],
            ]
            .copy()
        )

        sample_part[
            "period"
        ] = (
            dev_period.loc[
                sample_mask
            ]
            .to_numpy()
        )

        feature_sample_parts.append(
            sample_part
        )


    if (
        feature_cards_processed
        % 500
        == 0
    ):
        print(
            "Cards processed:",
            f"{feature_cards_processed:,}",
            "| raw rows seen:",
            f"{feature_raw_rows_seen:,}",
            "| development rows:",
            f"{development_rows_emitted:,}",
        )


feature_pass_elapsed_seconds = (
    time.perf_counter()
    - feature_pass_start
)


feature_sample_df = pd.concat(
    feature_sample_parts,
    ignore_index=True,
)


print(
    "\nFull causal pass hoàn tất."
)

print(
    "Cards processed:",
    f"{feature_cards_processed:,}",
)

print(
    "Raw rows seen:",
    f"{feature_raw_rows_seen:,}",
)

print(
    "History-context rows (< validation end):",
    f"{history_context_rows_processed:,}",
)

print(
    "Pre-W_LONG warm-up rows:",
    f"{pre_w_long_history_rows:,}",
)

print(
    "Development rows emitted:",
    f"{development_rows_emitted:,}",
)

print(
    "Max full Card block rows:",
    f"{max_full_card_block_rows:,}",
)

print(
    "Max context Card block rows:",
    f"{max_context_card_block_rows:,}",
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

print(
    "Context throughput rows/s:",
    round(
        history_context_rows_processed
        / feature_pass_elapsed_seconds,
        2,
    ),
)
```

    Cards processed: 500 | raw rows seen: 2,021,584 | development rows: 611,634
    Cards processed: 1,000 | raw rows seen: 4,066,794 | development rows: 1,255,463
    Cards processed: 1,500 | raw rows seen: 5,915,634 | development rows: 1,805,952
    Cards processed: 2,000 | raw rows seen: 7,883,446 | development rows: 2,428,739
    Cards processed: 2,500 | raw rows seen: 9,842,479 | development rows: 3,053,449
    Cards processed: 3,000 | raw rows seen: 11,897,259 | development rows: 3,673,054
    Cards processed: 3,500 | raw rows seen: 13,997,374 | development rows: 4,302,317
    Cards processed: 4,000 | raw rows seen: 16,040,304 | development rows: 4,961,502
    Cards processed: 4,500 | raw rows seen: 18,115,490 | development rows: 5,590,903
    Cards processed: 5,000 | raw rows seen: 19,945,815 | development rows: 6,156,873
    Cards processed: 6,000 | raw rows seen: 23,844,247 | development rows: 7,387,772
    
    Full causal pass hoàn tất.
    Cards processed: 6,139
    Raw rows seen: 24,386,900
    History-context rows (< validation end): 23,038,920
    Pre-W_LONG warm-up rows: 15,471,192
    Development rows emitted: 7,567,728
    Max full Card block rows: 70,008
    Max context Card block rows: 68,616
    Feature sample rows: 75,670
    Elapsed seconds: 106.24
    Context throughput rows/s: 216858.24


# 11. M4.5.4 — Full causal integrity gate


```python
assert (
    CAUSAL_STREAMING_SAFE
    is True
)

assert (
    M45_CAUSAL_UNIT_TEST_PASS
    is True
)

assert (
    feature_cards_processed
    == 6_139
)

assert (
    feature_raw_rows_seen
    == 24_386_900
)

assert (
    history_context_rows_processed
    ==
    EXPECTED_CONTEXT_ROWS_BEFORE_VALIDATION_END
)

assert (
    pre_w_long_history_rows
    ==
    EXPECTED_PRE_W_LONG_HISTORY_ROWS
)

assert (
    development_rows_emitted
    ==
    EXPECTED_DEVELOPMENT_ROWS
)


for (
    period_name,
    expected_count,
) in (
    EXPECTED_PERIOD_COUNTS.items()
):
    assert (
        coverage_state[
            period_name
        ][
            "row_count"
        ]
        == expected_count
    )


assert (
    development_min_feature_timestamp
    >= W_LONG_START
)

assert (
    development_max_feature_timestamp
    < VALIDATION_END
)


# History context phải lớn hơn
# classifier-development population.
assert (
    history_context_rows_processed
    >
    development_rows_emitted
)


# Không đọc target trong feature pass.
assert (
    "Is Fraud?"
    not in FEATURE_USECOLS
)


print(
    "M4.5 FULL CAUSAL INTEGRITY GATE: PASS"
)
```

    M4.5 FULL CAUSAL INTEGRITY GATE: PASS


### Nhận xét

Full causal pass xử lý:

`6,139 User+Card`

và stream qua đủ:

`24,386,900 raw rows`

History context thực sự được sử dụng cho development feature generation là:

`23,038,920 rows`

tức toàn bộ transaction có:

`Timestamp < 2019-06-01`

Trong đó:

`15,471,192 rows`

nằm trước `2015-01-01` và chỉ đóng vai trò historical warm-up.

Development feature rows được emit là:

`7,567,728`

đúng bằng:

`TRAIN_2015_2017 + TRAIN_2018 + VALIDATION`.

Điều này xác nhận rõ:

`history-context rows != classifier-development rows`

Các transaction trước W_LONG có thể đóng góp causal history nhưng không trở thành classifier rows.

Card block lớn nhất trên toàn artifact:

`70,008 rows`

Card context block lớn nhất trước validation end:

`68,616 rows`

Deterministic feature sample:

`75,670 rows`

xấp xỉ 1% development population.

Feature pass hoàn tất trong:

`106.24 seconds`

với throughput khoảng:

`216,858 context rows / second`

trên current environment.

Feature pass không đọc:

`Is Fraud?`

và context dùng cho M4.5 feature statistics dừng trước:

`2019-06-01`

Cell trả về:

`M4.5 FULL CAUSAL INTEGRITY GATE: PASS`

### Kết luận M4.5.4

Full behavioral feature implementation tái hiện được trên toàn development population.

History warm-up:

`ACTIVE`

Development rows:

`7,567,728 — VERIFIED`

Pre-W_LONG history:

`15,471,192 — HISTORY CONTEXT ONLY`

Target used:

`NO`

FINAL TEST feature state used for development decision:

`NO`

Status:

`PASS`

# 12. M4.5.5 — Behavioral coverage


```python
coverage_rows = []


for period_name in PERIODS:
    state = (
        coverage_state[
            period_name
        ]
    )

    n = (
        state[
            "row_count"
        ]
    )

    coverage_rows.append(
        {
            "period":
                period_name,

            "row_count":
                n,

            "prior_card_history_count":
                state[
                    "prior_history_count"
                ],

            "prior_card_history_pct":
                (
                    state[
                        "prior_history_count"
                    ]
                    / n
                    * 100
                ),

            "time_since_available_pct":
                (
                    state[
                        "time_since_available_count"
                    ]
                    / n
                    * 100
                ),

            "has_transaction_last_1h_pct":
                (
                    state[
                        "last_1h_positive_count"
                    ]
                    / n
                    * 100
                ),

            "previous_amount_mean_available_pct":
                (
                    state[
                        "previous_amount_mean_available_count"
                    ]
                    / n
                    * 100
                ),

            "amount_deviation_available_pct":
                (
                    state[
                        "amount_deviation_available_count"
                    ]
                    / n
                    * 100
                ),

            "is_new_merchant_rate_pct":
                (
                    state[
                        "new_merchant_count"
                    ]
                    / n
                    * 100
                ),

            "same_timestamp_row_rate_pct":
                (
                    state[
                        "same_timestamp_row_count"
                    ]
                    / n
                    * 100
                ),
        }
    )


behavioral_coverage_df = pd.DataFrame(
    coverage_rows
)


display(
    behavioral_coverage_df
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
      <th>row_count</th>
      <th>prior_card_history_count</th>
      <th>prior_card_history_pct</th>
      <th>time_since_available_pct</th>
      <th>has_transaction_last_1h_pct</th>
      <th>previous_amount_mean_available_pct</th>
      <th>amount_deviation_available_pct</th>
      <th>is_new_merchant_rate_pct</th>
      <th>same_timestamp_row_rate_pct</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>TRAIN_2015_2017</td>
      <td>5133655</td>
      <td>5133281</td>
      <td>99.992715</td>
      <td>99.992715</td>
      <td>19.525776</td>
      <td>99.992715</td>
      <td>99.992715</td>
      <td>3.039530</td>
      <td>1.086438</td>
    </tr>
    <tr>
      <th>1</th>
      <td>TRAIN_2018</td>
      <td>1721615</td>
      <td>1721515</td>
      <td>99.994192</td>
      <td>99.994192</td>
      <td>19.333939</td>
      <td>99.994192</td>
      <td>99.994192</td>
      <td>2.796560</td>
      <td>1.081543</td>
    </tr>
    <tr>
      <th>2</th>
      <td>VALIDATION</td>
      <td>712458</td>
      <td>712424</td>
      <td>99.995228</td>
      <td>99.995228</td>
      <td>19.410407</td>
      <td>99.995228</td>
      <td>99.995228</td>
      <td>2.571239</td>
      <td>1.099574</td>
    </tr>
  </tbody>
</table>
</div>



```python
for period_name in PERIODS:
    state = coverage_state[
        period_name
    ]

    assert (
        state[
            "prior_history_count"
        ]
        ==
        state[
            "time_since_available_count"
        ]
        ==
        state[
            "previous_amount_mean_available_count"
        ]
        ==
        state[
            "amount_deviation_available_count"
        ]
    )

    assert (
        state[
            "same_timestamp_row_count"
        ] > 0
    )

    assert (
        0
        < state[
            "last_1h_positive_count"
        ]
        < state[
            "row_count"
        ]
    )

    assert (
        0
        < state[
            "new_merchant_count"
        ]
        < state[
            "row_count"
        ]
    )


print(
    "M4.5 BEHAVIORAL EVIDENCE COMPLETENESS GATE: PASS"
)
```

    M4.5 BEHAVIORAL EVIDENCE COMPLETENESS GATE: PASS


### Phân tích / Nhận xét

Prior Card history coverage rất cao trong cả ba development periods.

TRAIN_2015_2017:

`99.992715%`

TRAIN_2018:

`99.994192%`

VALIDATION:

`99.995228%`

Số strict cold-start transaction tương ứng chỉ còn:

`374`

`100`

`34`

transaction.

Availability của:

`time_since_previous_transaction_min`

`previous_amount_mean`

và:

`amount_minus_previous_mean`

khớp chính xác với prior-card-history availability.

Cell completeness gate cũng đã assertion trực tiếp equality này và trả về:

`M4.5 BEHAVIORAL EVIDENCE COMPLETENESS GATE: PASS`

Như vậy missing của recency và historical Amount feature không phải data-quality missing ngẫu nhiên.

Nó có semantic rõ:

`không tồn tại strict prior Card history`.

---

### 1-hour velocity

Tỷ lệ transaction có ít nhất một prior transaction trong 1 giờ là:

TRAIN_2015_2017:

`19.525776%`

TRAIN_2018:

`19.333939%`

VALIDATION:

`19.410407%`

Tức khoảng:

`80.5%`

transaction có:

`transactions_last_1h = 0`.

Zero ở đây không phải missing.

Nó biểu diễn:

`không có activity trong rolling 1-hour history window`.

Pattern này ổn định giữa TRAIN và VALIDATION.

---

### Merchant novelty

Population rate của:

`is_new_merchant = True`

là:

TRAIN_2015_2017:

`3.039530%`

TRAIN_2018:

`2.796560%`

VALIDATION:

`2.571239%`

Feature không gần như luôn False.

Merchant novelty có đủ support để trở thành behavioral candidate.

Tỷ lệ giảm dần nhẹ theo thời gian cũng phù hợp với việc Card tích lũy merchant history ngày càng nhiều.

Tuy nhiên M4.5 không sử dụng target nên không suy ra predictive benefit từ pattern này.

---

### Same-timestamp exposure

Tỷ lệ row nằm trong timestamp group có nhiều hơn một transaction là khoảng:

TRAIN_2015_2017:

`1.086438%`

TRAIN_2018:

`1.081543%`

VALIDATION:

`1.099574%`

Tương ứng xấp xỉ:

`55,774`

`18,620`

`7,834`

development transactions.

Do đó strict same-timestamp handling ảnh hưởng tới một lượng transaction thực tế đáng kể, không phải chỉ một vài trường hợp ngoại lệ.

### Kết luận M4.5.5

Historical coverage:

`NEAR-COMPLETE`

Recency availability:

`HIGH`

Historical Amount availability:

`HIGH`

1-hour velocity:

`ZERO-HEAVY BUT WELL-SUPPORTED`

Merchant novelty:

`NON-TRIVIAL SUPPORT`

Same-timestamp handling:

`MATERIALLY REQUIRED`

Status:

`PASS`

# 13. M4.5.6 — Historical warm-up impact


```python
warmup_rows = []


for (
    window_name,
    state,
) in (
    warmup_state.items()
):
    n = (
        state[
            "row_count"
        ]
    )

    actual_cold = (
        state[
            "cold_start_with_warmup"
        ]
    )

    reset_cold = (
        state[
            "cold_start_if_reset"
        ]
    )

    helped = (
        reset_cold
        - actual_cold
    )

    warmup_rows.append(
        {
            "window":
                window_name,

            "row_count":
                n,

            "cold_start_with_warmup":
                actual_cold,

            "cold_start_with_warmup_pct":
                (
                    actual_cold
                    / n
                    * 100
                ),

            "cold_start_if_reset":
                reset_cold,

            "cold_start_if_reset_pct":
                (
                    reset_cold
                    / n
                    * 100
                ),

            "transactions_helped_by_prior_history":
                helped,

            "transactions_helped_by_prior_history_pct":
                (
                    helped
                    / n
                    * 100
                ),
        }
    )


history_warmup_df = (
    pd.DataFrame(
        warmup_rows
    )
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
      <th>cold_start_with_warmup</th>
      <th>cold_start_with_warmup_pct</th>
      <th>cold_start_if_reset</th>
      <th>cold_start_if_reset_pct</th>
      <th>transactions_helped_by_prior_history</th>
      <th>transactions_helped_by_prior_history_pct</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>W_LONG_TRAIN</td>
      <td>6855270</td>
      <td>474</td>
      <td>0.006914</td>
      <td>4308</td>
      <td>0.062842</td>
      <td>3834</td>
      <td>0.055928</td>
    </tr>
    <tr>
      <th>1</th>
      <td>W_SHORT_TRAIN</td>
      <td>1721615</td>
      <td>100</td>
      <td>0.005808</td>
      <td>4012</td>
      <td>0.233037</td>
      <td>3912</td>
      <td>0.227229</td>
    </tr>
    <tr>
      <th>2</th>
      <td>VALIDATION</td>
      <td>712458</td>
      <td>34</td>
      <td>0.004772</td>
      <td>3924</td>
      <td>0.550769</td>
      <td>3890</td>
      <td>0.545997</td>
    </tr>
  </tbody>
</table>
</div>



```python
assert (
    history_warmup_df[
        "transactions_helped_by_prior_history"
    ]
    >= 0
).all()


print(
    "M4.5 HISTORICAL WARM-UP AUDIT: PASS"
)
```

    M4.5 HISTORICAL WARM-UP AUDIT: PASS


### Phân tích / Nhận xét

Historical warm-up tạo khác biệt rõ ràng ở cả hai training-window candidate và VALIDATION.

## W_LONG TRAIN

Rows:

`6,855,270`

Cold-start khi giữ causal pre-window history:

`474`

tương đương:

`0.006914%`

Nếu reset history tại `2015-01-01`:

`4,308`

tương đương:

`0.062842%`

Warm-up hỗ trợ:

`3,834 transactions`

---

## W_SHORT TRAIN

Rows:

`1,721,615`

Cold-start với warm-up:

`100`

`0.005808%`

Nếu reset history tại `2018-01-01`:

`4,012`

`0.233037%`

Warm-up hỗ trợ:

`3,912 transactions`

Tức phần lớn artificial cold-start do reset W_SHORT được loại bỏ khi sử dụng causal history trước 2018.

---

## VALIDATION

Rows:

`712,458`

Cold-start với causal history:

`34`

`0.004772%`

Nếu reset history tại đầu VALIDATION:

`3,924`

`0.550769%`

Warm-up giúp:

`3,890 validation transactions`

tương đương:

`0.545997%`

toàn validation population.

Như vậy việc reset history ở partition boundary sẽ tạo ra một population cold-start nhân tạo lớn hơn nhiều so với trạng thái history thực tế tại prediction time.

Cell audit trả về:

`M4.5 HISTORICAL WARM-UP AUDIT: PASS`

### Kết luận M4.5.6

Historical warm-up không chỉ hợp lệ về causal protocol mà còn cần thiết để representation phản ánh đúng history availability tại prediction point.

Policy:

`PRESERVE CAUSAL PRE-WINDOW HISTORY`

Không:

`reset entity history at TRAIN / VALIDATION boundary`

Classifier membership:

`INDEPENDENT FROM HISTORY CONTEXT`

Status:

`LOCKED`

# 14. M4.5.7 — Behavioral distribution audit


```python
DISTRIBUTION_FEATURES = [
    "prior_card_transaction_count",
    "time_since_previous_transaction_min",
    "transactions_last_1h",
    "amount_minus_previous_mean",
]


QUANTILES = [
    0.00,
    0.50,
    0.90,
    0.95,
    0.99,
    1.00,
]


distribution_rows = []


for period_name in PERIODS:
    period_mask = (
        feature_sample_df[
            "period"
        ]
        .eq(
            period_name
        )
    )

    for feature_name in (
        DISTRIBUTION_FEATURES
    ):
        values = (
            feature_sample_df.loc[
                period_mask,
                feature_name,
            ]
            .dropna()
            .astype(float)
        )

        if values.empty:
            continue

        quantiles = (
            values
            .quantile(
                QUANTILES
            )
        )

        distribution_rows.append(
            {
                "period":
                    period_name,

                "feature":
                    feature_name,

                "sample_count":
                    len(
                        values
                    ),

                "min":
                    quantiles.loc[
                        0.00
                    ],

                "median":
                    quantiles.loc[
                        0.50
                    ],

                "p90":
                    quantiles.loc[
                        0.90
                    ],

                "p95":
                    quantiles.loc[
                        0.95
                    ],

                "p99":
                    quantiles.loc[
                        0.99
                    ],

                "max":
                    quantiles.loc[
                        1.00
                    ],
            }
        )


behavioral_distribution_df = (
    pd.DataFrame(
        distribution_rows
    )
)


display(
    behavioral_distribution_df
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
      <th>sample_count</th>
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
      <td>TRAIN_2015_2017</td>
      <td>prior_card_transaction_count</td>
      <td>51326</td>
      <td>0.000000</td>
      <td>4398.000000</td>
      <td>13539.000000</td>
      <td>18025.000000</td>
      <td>30756.000000</td>
      <td>65499.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>TRAIN_2015_2017</td>
      <td>time_since_previous_transaction_min</td>
      <td>51323</td>
      <td>1.000000</td>
      <td>609.000000</td>
      <td>2886.800000</td>
      <td>4330.000000</td>
      <td>8675.780000</td>
      <td>377970.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>TRAIN_2015_2017</td>
      <td>transactions_last_1h</td>
      <td>51326</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>10.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>TRAIN_2015_2017</td>
      <td>amount_minus_previous_mean</td>
      <td>51323</td>
      <td>-862.160000</td>
      <td>-9.136520</td>
      <td>55.753014</td>
      <td>89.935893</td>
      <td>257.653394</td>
      <td>1701.109714</td>
    </tr>
    <tr>
      <th>4</th>
      <td>TRAIN_2018</td>
      <td>prior_card_transaction_count</td>
      <td>17234</td>
      <td>0.000000</td>
      <td>5203.000000</td>
      <td>15349.100000</td>
      <td>19745.150000</td>
      <td>32819.680000</td>
      <td>67699.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>TRAIN_2018</td>
      <td>time_since_previous_transaction_min</td>
      <td>17233</td>
      <td>1.000000</td>
      <td>619.000000</td>
      <td>2895.800000</td>
      <td>4333.400000</td>
      <td>8930.480000</td>
      <td>36727.000000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>TRAIN_2018</td>
      <td>transactions_last_1h</td>
      <td>17234</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>6.000000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>TRAIN_2018</td>
      <td>amount_minus_previous_mean</td>
      <td>17233</td>
      <td>-549.050451</td>
      <td>-9.112634</td>
      <td>55.384990</td>
      <td>89.549788</td>
      <td>221.610969</td>
      <td>1490.821916</td>
    </tr>
    <tr>
      <th>8</th>
      <td>VALIDATION</td>
      <td>prior_card_transaction_count</td>
      <td>7110</td>
      <td>1.000000</td>
      <td>5599.500000</td>
      <td>16295.600000</td>
      <td>20886.100000</td>
      <td>34330.730000</td>
      <td>68599.000000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>VALIDATION</td>
      <td>time_since_previous_transaction_min</td>
      <td>7110</td>
      <td>1.000000</td>
      <td>611.500000</td>
      <td>2870.100000</td>
      <td>4321.000000</td>
      <td>8470.640000</td>
      <td>64123.000000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>VALIDATION</td>
      <td>transactions_last_1h</td>
      <td>7110</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>2.910000</td>
      <td>6.000000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>VALIDATION</td>
      <td>amount_minus_previous_mean</td>
      <td>7110</td>
      <td>-560.446965</td>
      <td>-8.792889</td>
      <td>57.051582</td>
      <td>92.181248</td>
      <td>266.944955</td>
      <td>1765.549761</td>
    </tr>
  </tbody>
</table>
</div>



```python
merchant_novelty_sample_rows = []


for period_name in PERIODS:
    period_mask = (
        feature_sample_df[
            "period"
        ]
        .eq(
            period_name
        )
    )

    sub = (
        feature_sample_df.loc[
            period_mask
        ]
    )

    merchant_novelty_sample_rows.append(
        {
            "period":
                period_name,

            "sample_rows":
                len(sub),

            "is_new_merchant_count":
                int(
                    sub[
                        "is_new_merchant"
                    ]
                    .sum()
                ),

            "is_new_merchant_rate_pct":
                (
                    sub[
                        "is_new_merchant"
                    ]
                    .mean()
                    * 100
                ),
        }
    )


merchant_novelty_sample_df = (
    pd.DataFrame(
        merchant_novelty_sample_rows
    )
)


display(
    merchant_novelty_sample_df
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
      <th>sample_rows</th>
      <th>is_new_merchant_count</th>
      <th>is_new_merchant_rate_pct</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>TRAIN_2015_2017</td>
      <td>51326</td>
      <td>1578</td>
      <td>3.074465</td>
    </tr>
    <tr>
      <th>1</th>
      <td>TRAIN_2018</td>
      <td>17234</td>
      <td>456</td>
      <td>2.645932</td>
    </tr>
    <tr>
      <th>2</th>
      <td>VALIDATION</td>
      <td>7110</td>
      <td>181</td>
      <td>2.545710</td>
    </tr>
  </tbody>
</table>
</div>


### Phân tích / Nhận xét

## prior_card_transaction_count

Median history depth tăng theo thời gian:

TRAIN_2015_2017:

`4,398`

TRAIN_2018:

`5,203`

VALIDATION:

`5,599.5`

Điều này phù hợp với accumulated Card history.

Feature có variation rất lớn, nhưng hiện tại nó chỉ được sử dụng làm audit/support state chứ chưa được đưa vào baseline classifier candidate.

---

## time_since_previous_transaction_min

Median:

TRAIN_2015_2017:

`609 minutes`

TRAIN_2018:

`619 minutes`

VALIDATION:

`611.5 minutes`

Tức khoảng:

`10.2 giờ`

và khá ổn định giữa temporal periods.

P90 khoảng:

`2,870–2,896 minutes`

P95 khoảng:

`4,321–4,333 minutes`

P99 khoảng:

`8,471–8,930 minutes`

Feature vì vậy có long right tail rõ rệt.

Sample maximum lên tới:

`377,970 minutes`

ở TRAIN_2015_2017.

Điều này cho thấy raw recency scale trải trên nhiều bậc độ lớn.

M4.5 chỉ khóa semantic feature.

Nếu cần `log1p` hoặc scaling, việc đó thuộc M4.6/feature experiment.

---

## transactions_last_1h

Cả ba period đều có:

`median = 0`

`P90 = 1`

`P95 = 2`

P99 xấp xỉ:

`3`

Maximum sample:

`10`

ở TRAIN_2015_2017 và:

`6`

ở TRAIN_2018 / VALIDATION.

Distribution rời rạc, zero-heavy nhưng không collapse thành constant.

Pattern cũng ổn định giữa TRAIN và VALIDATION.

---

## amount_minus_previous_mean

Median:

TRAIN_2015_2017:

`-9.14`

TRAIN_2018:

`-9.11`

VALIDATION:

`-8.79`

P90 khoảng:

`55–57`

P95 khoảng:

`90–92`

P99:

`~222–267`

Sample range trải từ hàng trăm âm tới hơn:

`1,400–1,700`

ở phía dương.

Feature có distribution rộng và asymmetric.

Điều này phù hợp với việc Amount itself có tail và historical mean nhạy với extreme values.

Không có evidence ở M4.5 để clip hoặc thay running mean bằng robust estimator.

Những lựa chọn như median historical baseline hoặc transformed deviation phải là controlled experiment sau.

---

## Merchant novelty sample cross-check

Deterministic sample cho novelty rate:

`3.0745%`

`2.6459%`

`2.5457%`

theo ba periods.

Các mức này gần population rates từ full pass, nên sample phục vụ distribution không cho thấy bất nhất rõ rệt với population aggregation.

### Kết luận M4.5.7

Không behavioral prototype nào collapse thành constant.

`time_since_previous_transaction_min`

→ long-tailed continuous behavioral candidate.

`transactions_last_1h`

→ discrete zero-heavy behavioral candidate.

`amount_minus_previous_mean`

→ wide/asymmetric continuous behavioral candidate.

`is_new_merchant`

→ low-frequency nhưng well-supported boolean candidate.

Feature transformation:

`OPEN — M4.6 / CONTROLLED EXPERIMENT`

Behavioral semantic validity:

`CONFIRMED`

# 15. M4.5.8 — Exact lookback contract cho baseline candidate

M3 để historical lookback tối ưu ở trạng thái OPEN.

M4.5 cần phân biệt:

baseline feature semantics

với:

lookback strategy tối ưu về classification.

## Baseline candidate semantics hiện tại

### time_since_previous_transaction_min

Entity:

User + Card

History:

previous distinct Timestamp gần nhất.

Lookback:

không đặt fixed horizon.

History condition:

Timestamp(history) < T.

### transactions_last_1h

Entity:

User + Card

Exact window:

[T - 1 hour, T)

### amount_minus_previous_mean

Entity:

User + Card

Historical baseline:

mean của toàn bộ strict prior history có sẵn.

Lookback:

full prior history.

### is_new_merchant

Entity:

User + Card + Merchant Name history state

Definition:

True nếu không tồn tại cùng Merchant Name tại Timestamp < T.

Lookback:

full prior history.

## Interpretation

Full prior history ở đây là baseline implementation candidate.

M4.5 không tuyên bố:

full prior history > 1-year history

hoặc:

full prior history > rolling-history alternative

về model performance.

Các alternative lookback chỉ được mở như controlled feature experiment sau khi baseline implementation đã ổn định.


```python
lookback_contract_df = pd.DataFrame(
    [
        {
            "feature":
                "time_since_previous_transaction_min",

            "entity":
                "User+Card",

            "lookback":
                "nearest strict prior timestamp",

            "same_timestamp":
                "excluded",

            "warmup":
                "allowed",
        },

        {
            "feature":
                "transactions_last_1h",

            "entity":
                "User+Card",

            "lookback":
                "[T-1h, T)",

            "same_timestamp":
                "excluded",

            "warmup":
                "allowed",
        },

        {
            "feature":
                "amount_minus_previous_mean",

            "entity":
                "User+Card",

            "lookback":
                "full strict prior history",

            "same_timestamp":
                "excluded",

            "warmup":
                "allowed",
        },

        {
            "feature":
                "is_new_merchant",

            "entity":
                "User+Card+Merchant-history",

            "lookback":
                "full strict prior history",

            "same_timestamp":
                "excluded",

            "warmup":
                "allowed",
        },
    ]
)


display(
    lookback_contract_df
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
      <th>entity</th>
      <th>lookback</th>
      <th>same_timestamp</th>
      <th>warmup</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>time_since_previous_transaction_min</td>
      <td>User+Card</td>
      <td>nearest strict prior timestamp</td>
      <td>excluded</td>
      <td>allowed</td>
    </tr>
    <tr>
      <th>1</th>
      <td>transactions_last_1h</td>
      <td>User+Card</td>
      <td>[T-1h, T)</td>
      <td>excluded</td>
      <td>allowed</td>
    </tr>
    <tr>
      <th>2</th>
      <td>amount_minus_previous_mean</td>
      <td>User+Card</td>
      <td>full strict prior history</td>
      <td>excluded</td>
      <td>allowed</td>
    </tr>
    <tr>
      <th>3</th>
      <td>is_new_merchant</td>
      <td>User+Card+Merchant-history</td>
      <td>full strict prior history</td>
      <td>excluded</td>
      <td>allowed</td>
    </tr>
  </tbody>
</table>
</div>


### Nhận xét

Current baseline implementation phân biệt rõ từng loại historical scope.

`time_since_previous_transaction_min`

không dùng fixed historical horizon.

Nó chỉ tìm distinct strict-prior timestamp gần nhất.

`transactions_last_1h`

có exact rolling window:

`[T - 1 hour, T)`

`amount_minus_previous_mean`

dùng mean từ:

`full strict prior Card history`

`is_new_merchant`

kiểm tra merchant trên:

`full strict prior Card history`

Trong tất cả trường hợp:

same timestamp bị loại

và historical warm-up được phép.

M4.5 chỉ khóa đây là baseline semantics để pipeline có một definition tái hiện được.

Current evidence không so sánh:

full history

với:

1-year history

hoặc:

rolling 30/90/365-day history.

Do đó không có căn cứ để gọi full-history lookback là tối ưu về classification.

### Kết luận M4.5.8

Baseline lookback contract:

`LOCKED`

Recency:

`NEAREST STRICT PRIOR TIMESTAMP`

Velocity:

`[T - 1h, T)`

Historical Amount mean:

`FULL STRICT PRIOR CARD HISTORY`

Merchant novelty:

`FULL STRICT PRIOR CARD HISTORY`

Optimal historical lookback:

`OPEN — REQUIRES FEATURE EXPERIMENT`

Baseline semantic contract và optimal-performance choice được giữ là hai quyết định riêng biệt.

# 16. M4.5.9 — Cold-start semantic contract

### Nhận xét / Kết luận

Runtime evidence xác nhận cold-start tồn tại nhưng rất hiếm trong development population khi historical warm-up được sử dụng.

Cold-start representation phải giữ đúng semantic:

`has_prior_card_history = False`

`prior_card_transaction_count = 0`

`time_since_previous_transaction_min = NA`

`transactions_last_1h = 0`

`previous_amount_mean = NA`

`amount_minus_previous_mean = NA`

`is_new_merchant = True`

Điểm quan trọng:

`transactions_last_1h = 0`

và:

`time_since_previous_transaction_min = NA`

không được coi là cùng một loại missing.

Zero velocity nghĩa là:

`không có transaction trong một giờ trước`

trong khi NA recency/deviation nghĩa là:

`không tồn tại prior Card history`.

M4.5 không impute các NA này.

`has_prior_card_history` cần được giữ làm companion state để M4.6 có thể bảo toàn semantic cold-start khi numerical missing được xử lý.

Cold-start policy:

`LOCKED AT SEMANTIC LEVEL`

Imputation/encoding:

`OPEN — M4.6`

# 17. M4.5.10 — Behavioral Feature Registry

## B01 — time_since_previous_transaction_min

Entity:

`User + Card`

Semantic:

`RECENCY`

History rule:

nearest distinct `Timestamp < current Timestamp`

Cold-start:

`NA`

Prediction-point valid:

`YES`

Evidence:

Availability trên `99.99%` development rows; distribution có variation rõ và ổn định tương đối qua TRAIN/VALIDATION.

Decision:

`PRIMARY BEHAVIORAL CANDIDATE`

Status:

`APPROVED`

---

## B02 — transactions_last_1h

Entity:

`User + Card`

Semantic:

`SHORT-WINDOW VELOCITY`

Window:

`[T - 1 hour, T)`

Cold-start:

`0`

Zero meaning:

`VALID BEHAVIORAL STATE`

Prediction-point valid:

`YES`

Evidence:

Khoảng `19.3–19.5%` transaction có positive 1-hour history; median 0, P90 1, P95 2, P99 khoảng 3.

Decision:

`PRIMARY BEHAVIORAL CANDIDATE`

Status:

`APPROVED`

---

## B03 — amount_minus_previous_mean

Entity:

`User + Card`

Semantic:

`CURRENT AMOUNT DEVIATION FROM PRIOR CARD HISTORY`

Historical state:

`previous_amount_mean` chỉ từ `Timestamp < T`

Cold-start:

`NA`

Prediction-point valid:

`YES`

Evidence:

Availability trên `99.99%` development rows có prior history; distribution rộng và non-constant.

Decision:

`PRIMARY BEHAVIORAL CANDIDATE`

Status:

`APPROVED`

---

## B04 — is_new_merchant

Entity:

`User + Card + Merchant Name history`

Semantic:

`MERCHANT NOVELTY`

Definition:

True nếu Merchant Name chưa xuất hiện tại bất kỳ `Timestamp < T`.

Same first timestamp:

`ALL PEERS ARE NEW`

Prediction-point valid:

`YES`

Evidence:

Population True rate khoảng `2.57–3.04%`; strict-causal unit test và full implementation đều xử lý timestamp group đúng.

Decision:

`PRIMARY BEHAVIORAL CANDIDATE`

Status:

`APPROVED`

---

## B05 — has_prior_card_history

Entity:

`User + Card`

Semantic:

`COLD-START / HISTORY-AVAILABILITY COMPANION STATE`

Definition:

`prior_card_transaction_count > 0`

Evidence:

Strict cold-start hiếm nhưng tồn tại; state này phân biệt semantic NA với observed zero.

Decision:

`APPROVED COMPANION CANDIDATE`

Status:

`APPROVED`

---

## Audit/support state — prior_card_transaction_count

Vai trò:

`HISTORY DEPTH AUDIT STATE`

Evidence:

History depth có variation lớn và tăng theo accumulated history.

Decision:

Không đưa mặc định vào baseline classifier set ở M4.5.

Có thể được mở lại như controlled feature candidate nếu experiment plan cần history-depth feature.

Status:

`AUDIT / SUPPORT ONLY`

---

## Audit/support state — previous_amount_mean

Vai trò:

`INTERMEDIATE CAUSAL STATE`

Dùng để tính:

`amount_minus_previous_mean`

Decision:

Không đưa đồng thời previous mean và deviation vào core baseline nếu chưa có experiment justification.

Status:

`SUPPORT STATE`

# 18. Explicit exclusions

## Raw identifiers

User

Card

Merchant Name

Được sử dụng để:

- grouping;
- history state;
- merchant novelty.

Không được xuất hiện trực tiếp trong classifier feature set.

---

## Target

Is Fraud?

Không được đọc trong behavioral feature pass.

Không được sử dụng dưới bất kỳ dạng:

- previous fraud count;
- previous fraud rate;
- fraud history;
- target encoding;
- future label state.

---

## Same-timestamp peer

Transaction có:

Timestamp == current Timestamp

không được tính là history.

---

## Future transaction

Transaction có:

Timestamp > current Timestamp

không được dùng.

---

## FINAL TEST

Không dùng FINAL TEST distribution hoặc target để quyết định Behavioral Feature Contract.

---

## Learned state

M4.5 không học:

- scaler;
- encoder;
- imputer;
- model parameter.

Các bước đó thuộc M4.6 hoặc modeling experiment.

# 19. M4.5.11 — Behavioral feature safety gate


```python
BEHAVIORAL_MODEL_CANDIDATES = {
    "time_since_previous_transaction_min",
    "transactions_last_1h",
    "amount_minus_previous_mean",
    "is_new_merchant",
    "has_prior_card_history",
}


PROHIBITED_DIRECT_FEATURES = {
    "User",
    "Card",
    "Merchant Name",
    "Errors?",
    "Is Fraud?",
    "raw_row_id",
    "Timestamp",
}


assert (
    BEHAVIORAL_MODEL_CANDIDATES
    .isdisjoint(
        PROHIBITED_DIRECT_FEATURES
    )
)


assert (
    CAUSAL_STREAMING_SAFE
    is True
)

assert (
    M45_CAUSAL_UNIT_TEST_PASS
    is True
)

assert (
    feature_cards_processed
    == 6_139
)

assert (
    development_rows_emitted
    == EXPECTED_DEVELOPMENT_ROWS
)

assert (
    history_context_rows_processed
    ==
    EXPECTED_CONTEXT_ROWS_BEFORE_VALIDATION_END
)

assert (
    development_max_feature_timestamp
    < VALIDATION_END
)

assert (
    "Is Fraud?"
    not in FEATURE_USECOLS
)


LEARNED_PREPROCESSING_STATE_CREATED = False

TARGET_HISTORY_USED = False

FINAL_TEST_STATISTICS_USED_FOR_DECISION = False

RAW_IDENTIFIER_EXPOSED_AS_DIRECT_FEATURE = False


assert (
    LEARNED_PREPROCESSING_STATE_CREATED
    is False
)

assert (
    TARGET_HISTORY_USED
    is False
)

assert (
    FINAL_TEST_STATISTICS_USED_FOR_DECISION
    is False
)

assert (
    RAW_IDENTIFIER_EXPOSED_AS_DIRECT_FEATURE
    is False
)


print(
    "M4.5 BEHAVIORAL FEATURE SAFETY GATE: PASS"
)
```

    M4.5 BEHAVIORAL FEATURE SAFETY GATE: PASS


### Gate interpretation

Cell trả về:

`M4.5 BEHAVIORAL FEATURE SAFETY GATE: PASS`

Behavioral model candidates không chứa trực tiếp:

- User;
- Card;
- Merchant Name;
- Errors?;
- Is Fraud?;
- raw_row_id;
- Timestamp.

User/Card/Merchant Name chỉ được dùng làm:

`history key / state key`

Không sử dụng target trong feature pass.

Không tạo:

- target-history state;
- learned encoder;
- scaler;
- learned imputer;
- future-label feature.

Feature state phục vụ development dừng trước FINAL TEST boundary.

Strict-causal unit test và full causal integrity gate đều đã PASS.

Status:

`PASS`

# 20. M4.5 Findings

## M4.5-F01 — Causal streaming integrity

Observed fact:

Current artifact có đúng `6,139 User+Card` contiguous blocks, không block reappearance và không timestamp decrease.

Evidence:

`24,386,900 rows`

`Timestamp parse failures = 0`

`Card block reappearance = 0`

`Within-card timestamp decreases = 0`

Interpretation:

Physical order của current artifact đủ điều kiện cho memory-efficient card-level causal streaming.

Implication:

Có thể tiếp tục sử dụng streaming implementation với explicit order gate.

Không được bỏ gate hoặc giả định artifact khác có cùng order.

Status:

`CONFIRMED`

---

## M4.5-F02 — Same-timestamp handling

Observed fact:

Có `142,010` adjacent equal-timestamp pairs trên raw artifact và khoảng `1.08–1.10%` development rows thuộc multi-row timestamp group.

Evidence:

Strict causal unit test xác nhận same-timestamp peer không ảnh hưởng:

- prior count;
- recency;
- velocity;
- previous Amount mean;
- merchant novelty.

Interpretation:

Same-timestamp exclusion là requirement thực tế, không phải theoretical edge case.

Implication:

Mọi historical feature implementation phải update history state sau khi toàn current timestamp group đã được scored.

Status:

`CONFIRMED`

---

## M4.5-F03 — Historical warm-up

Observed fact:

Giữ pre-window history giảm artificial cold-start rõ rệt.

Evidence:

W_SHORT:

`4,012 → 100`

VALIDATION:

`3,924 → 34`

khi chuyển từ reset-history sang causal warm-up.

Interpretation:

Partition boundary không phải entity-history reset boundary.

Implication:

Older causal transactions được phép và cần tiếp tục làm history context dù không phải classifier rows.

Status:

`CONFIRMED`

---

## M4.5-F04 — Recency feature

Observed fact:

Recency available trên hơn `99.99%` development transactions.

Median khoảng:

`609–619 phút`

và P99 khoảng:

`8,471–8,930 phút`.

Interpretation:

Feature có variation thực sự và long right tail.

Implication:

Giữ làm primary behavioral candidate.

Transformation nếu cần thuộc preprocessing/experiment sau.

Status:

`CONFIRMED`

---

## M4.5-F05 — 1-hour velocity

Observed fact:

Khoảng `19.3–19.5%` transaction có prior activity trong một giờ.

Median:

`0`

P90:

`1`

P95:

`2`

P99:

khoảng `3`.

Interpretation:

Velocity có zero-heavy discrete distribution nhưng không constant.

Implication:

Giữ `transactions_last_1h` làm primary behavioral candidate.

Zero phải được bảo toàn như observed behavioral value.

Status:

`CONFIRMED`

---

## M4.5-F06 — Amount historical deviation

Observed fact:

`amount_minus_previous_mean` available trên gần toàn bộ rows có prior Card history và có distribution rộng/asymmetric.

Evidence:

Median khoảng:

`-9`

P95 khoảng:

`90–92`

P99 lên khoảng:

`222–267`

và sample chứa extreme deviations lớn hơn nhiều.

Interpretation:

Current Amount thường có thể được contextualize bằng historical Card Amount behavior.

Running mean vẫn nhạy với historical extreme Amount.

Implication:

Giữ deviation làm primary candidate.

Không clip và không thay running mean bằng robust estimator tại M4.5.

Status:

`CONFIRMED`

---

## M4.5-F07 — Merchant novelty

Observed fact:

`is_new_merchant = True`

khoảng:

`3.04%`

`2.80%`

`2.57%`

qua ba development periods.

Evidence:

Strict causal implementation cập nhật `seen_merchants` sau từng timestamp group.

Sample cross-check cũng cho mức tương tự.

Interpretation:

Merchant novelty có support thực tế và không gần constant.

Implication:

Giữ làm primary behavioral candidate.

Raw Merchant Name vẫn không được đưa trực tiếp vào classifier.

Status:

`CONFIRMED`

---

## M4.5-F08 — Cold-start semantics

Observed fact:

Strict Card cold-start chỉ còn:

`374`

`100`

`34`

rows trong ba development periods khi historical warm-up được giữ.

Evidence:

Availability của recency, previous mean và amount deviation khớp chính xác prior-history availability.

Interpretation:

NA của historical numerical feature là structural cold-start state, không phải data-quality missing.

Implication:

Giữ `has_prior_card_history` làm companion candidate.

M4.6 phải xử lý NA mà không làm mất cold-start semantic.

Status:

`CONFIRMED`

---

## M4.5-F09 — Computational feasibility

Observed fact:

Full causal feature pass xử lý:

`23,038,920 history-context rows`

trong:

`106.24 seconds`

với throughput khoảng:

`216,858 rows/second`.

Max full Card block:

`70,008 rows`

Max context block:

`68,616 rows`.

Interpretation:

Card-level streaming không tạo computational blocker trong current project environment.

Implication:

Behavioral feature generation có thể được tích hợp vào pipeline mà không cần materialize toàn bộ 24 triệu-row feature frame trong memory.

Status:

`CONFIRMED`

# 21. Decision Log M4.5

## M4.5-D01 — History entity

Decision:

Bốn baseline behavioral prototype sử dụng:

`User + Card`

làm primary history entity.

Merchant novelty bổ sung:

`Merchant Name`

chỉ làm history-state key bên trong Card.

User-level fallback chưa thuộc Behavioral v1.

Status:

`LOCKED FOR BASELINE V1`

---

## M4.5-D02 — Strict causal rule

Decision:

`Timestamp(history) < Timestamp(current)`

Same timestamp:

`EXCLUDED`

Status:

`INHERITED — VERIFIED — LOCKED`

---

## M4.5-D03 — Historical warm-up

Decision:

Giữ causal historical state xuyên qua modeling-window và partition boundaries.

Không reset history tại:

- W_LONG start;
- W_SHORT start;
- VALIDATION start.

Classifier-training membership không giới hạn causal history availability.

Status:

`LOCKED`

---

## M4.5-D04 — Recency candidate

Decision:

`time_since_previous_transaction_min`

là primary behavioral candidate.

Cold-start:

`NA`

Status:

`LOCKED AT CANDIDATE LEVEL`

---

## M4.5-D05 — Velocity candidate

Decision:

`transactions_last_1h`

là primary behavioral candidate.

Exact window:

`[T - 1h, T)`

Zero:

`VALID OBSERVED VALUE`

Status:

`LOCKED AT CANDIDATE LEVEL`

---

## M4.5-D06 — Amount-deviation candidate

Decision:

`amount_minus_previous_mean`

là primary behavioral candidate.

Historical baseline:

strict-prior Card running mean.

Cold-start:

`NA`

Status:

`LOCKED AT CANDIDATE LEVEL`

---

## M4.5-D07 — Merchant-novelty candidate

Decision:

`is_new_merchant`

là primary behavioral candidate.

History state chỉ được cập nhật sau current timestamp group.

Raw Merchant Name không được exposed cho classifier.

Status:

`LOCKED AT CANDIDATE LEVEL`

---

## M4.5-D08 — Cold-start companion state

Decision:

`has_prior_card_history`

được giữ làm explicit companion candidate.

Mục đích:

bảo toàn semantic giữa structural NA và observed zero khi sang preprocessing.

Status:

`LOCKED AT CANDIDATE LEVEL`

---

## M4.5-D09 — Baseline lookback semantics

Decision:

Baseline semantics:

Recency:

`nearest strict prior timestamp`

Velocity:

`[T - 1h, T)`

Historical Amount:

`full strict prior Card history`

Merchant novelty:

`full strict prior Card history`

Status:

`LOCKED FOR BASELINE V1`

Optimal historical lookback:

`OPEN — REQUIRES FEATURE EXPERIMENT`

---

## M4.5-D10 — Raw identifier role

Decision:

User/Card/Merchant Name chỉ dùng làm history key/state.

Không direct classifier feature.

Status:

`INHERITED — VERIFIED — LOCKED`

---

## M4.5-D11 — Target-history prohibition

Decision:

Không sử dụng target history.

Status:

`INHERITED — VERIFIED — LOCKED`

---

## M4.5-D12 — FINAL TEST isolation

Decision:

Behavioral feature statistics và state phục vụ M4.5 development decision dừng trước:

`2019-06-01`

Không dùng FINAL TEST target hoặc feature distribution để lựa chọn behavioral contract.

Status:

`INHERITED — VERIFIED — LOCKED`

# 22. M4.5 Gate

## G01 — Artifact / causal order verified

Result:

`PASS`

Full raw artifact đã được order-audit.

---

## G02 — Card block integrity verified

Result:

`PASS`

`6,139 User+Card`

`6,139 blocks`

`0 reappearance`

`0 timestamp decrease`

---

## G03 — Strict causal computation verified

Result:

`PASS`

Deterministic unit test và full-pass assertions đều thành công.

---

## G04 — Same-timestamp peer exclusion verified

Result:

`PASS`

Same-timestamp peers không được đưa vào prior count, recency, velocity, previous Amount mean hoặc merchant-history state.

---

## G05 — Historical warm-up audited

Result:

`PASS`

Warm-up được đo riêng cho W_LONG, W_SHORT và VALIDATION.

---

## G06 — History rows separated from classifier rows

Result:

`PASS`

History context:

`23,038,920 rows`

Development classifier population:

`7,567,728 rows`

Pre-W_LONG warm-up:

`15,471,192 rows`

---

## G07 — Recency candidate audited

Result:

`PASS`

Coverage và distribution đã được kiểm tra.

---

## G08 — Velocity candidate audited

Result:

`PASS`

1-hour window semantics, zero-heavy distribution và support đã được kiểm tra.

---

## G09 — Historical Amount candidate audited

Result:

`PASS`

Strict-prior mean và Amount deviation đã được kiểm tra về availability, identity và distribution.

---

## G10 — Merchant novelty candidate audited

Result:

`PASS`

Strict causal implementation đã được sửa và unit-tested.

---

## G11 — Cold-start semantics explicit

Result:

`PASS`

NA và zero được phân biệt semantic rõ ràng.

---

## G12 — Raw identifiers not exposed

Result:

`PASS`

User/Card/Merchant Name chỉ làm history key/state.

---

## G13 — No target-history leakage

Result:

`PASS`

`Is Fraud?` không xuất hiện trong behavioral feature pass.

---

## G14 — No learned preprocessing state

Result:

`PASS`

M4.5 chỉ tạo deterministic causal history features.

---

## G15 — FINAL TEST remains protected

Result:

`PASS`

Development behavioral state/statistics dừng trước `2019-06-01`.

---

## G16 — Behavioral Feature Registry complete

Result:

`PASS`

Bốn core behavioral candidates và cold-start companion state đã có semantic contract rõ.


# M4.5 Gate

Overall:

`PASS`

Blocking issue:

`NONE`

M4.5 Status:

`PASS — READY FOR M4.6`

# 23. Kết luận M4.5

## Mục tiêu đã kiểm tra

M4.5 triển khai và audit behavioral feature representation dựa trên historical transaction context nhưng phải giữ strict causal ordering.

Notebook kiểm tra:

- physical Card ordering;
- contiguous entity blocks;
- same-timestamp handling;
- strict prior-event computation;
- historical warm-up;
- cold-start;
- recency;
- short-window velocity;
- historical Amount context;
- merchant novelty;
- distribution;
- computational feasibility;
- target/FINAL TEST protection.

---

## Causal implementation đã xác minh

Full raw artifact gồm:

`24,386,900 transactions`

và:

`6,139 User+Card`

được stream thành công.

Current artifact có:

`0 timestamp decrease`

`0 Card block reappearance`

nhưng có:

`142,010 adjacent equal-timestamp pairs`.

Do đó implementation bắt buộc xử lý cùng timestamp như cùng prediction point.

Deterministic unit test xác nhận:

`Timestamp(history) < Timestamp(current)`

được áp dụng nhất quán cho tất cả behavioral states.

Merchant novelty hiện cũng được tính bằng accumulated past state thay vì future-inclusive aggregate.

---

## Behavioral candidates được khóa

### Primary behavioral candidates

`time_since_previous_transaction_min`

`transactions_last_1h`

`amount_minus_previous_mean`

`is_new_merchant`

### Cold-start companion candidate

`has_prior_card_history`

### Audit/support states

`prior_card_transaction_count`

`previous_amount_mean`

Hai state này chưa mặc định trở thành classifier feature.

---

## Same-timestamp policy

Transaction có cùng Timestamp:

`DO NOT PROVIDE HISTORY TO EACH OTHER`

History state chỉ được update sau khi toàn bộ transaction trong current timestamp group đã được xử lý.

Policy:

`LOCKED`

---

## Historical warm-up policy

Older transaction được phép làm history nếu:

`Timestamp(history) < Timestamp(current)`

dù chúng nằm ngoài classifier training window.

Không reset history tại:

- W_LONG start;
- W_SHORT start;
- VALIDATION start.

Runtime audit cho thấy reset history sẽ tạo nhiều artificial cold-start hơn.

Policy:

`LOCKED`

---

## Cold-start policy

Khi không có prior Card history:

`has_prior_card_history = False`

`time_since_previous_transaction_min = NA`

`transactions_last_1h = 0`

`amount_minus_previous_mean = NA`

`is_new_merchant = True`

NA không được tự động hiểu là data-quality missing.

M4.6 phải xử lý representation này mà vẫn giữ cold-start semantic.

Policy:

`LOCKED AT SEMANTIC LEVEL`

---

## Baseline lookback contract

Recency:

`nearest strict prior timestamp`

Velocity:

`[T - 1h, T)`

Historical Amount:

`full strict prior Card history`

Merchant novelty:

`full strict prior Card history`

Đây là baseline implementation contract.

Không phải bằng chứng rằng full-history lookback là tối ưu.

Optimal lookback:

`OPEN — REQUIRES FEATURE EXPERIMENT`

---

## Những vấn đề tiếp tục OPEN

M4.5 chưa chứng minh:

- behavioral features cải thiện F1_fraud;
- feature nào trong bốn candidate mang predictive value lớn nhất;
- full prior Amount mean tốt hơn rolling/robust alternative;
- full-history merchant novelty tốt hơn fixed lookback;
- recency cần raw hay log transform;
- amount deviation cần transform hay scaling;
- prior_card_transaction_count có nên trở thành model feature;
- user-level fallback có đáng bổ sung cho new-card cold-start;
- behavioral-feature subset cuối cùng.

Đặc biệt, user-level fallback có thể là future candidate cho new-card/existing-user cases nhưng không phải blocker cho Behavioral v1.

---

## Computational / memory note

Full causal pass:

`23,038,920 history-context rows`

Runtime:

`106.24 seconds`

Throughput:

`~216,858 rows/second`

Max Card block:

`70,008 rows`

Implementation xử lý từng Card block nên không cần giữ toàn behavioral feature matrix trong RAM.

Không phát hiện computational blocker trong current project environment.

---

## Blocking issue

`NONE`

---

## M4.5 Gate

`PASS`

Strict-causal behavioral feature representation đã được triển khai và audit đầy đủ.

---

## Trạng thái cuối

`M4.5 — PASS`

`Behavioral Feature Contract — LOCKED AT CANDIDATE LEVEL`

`READY FOR M4.6`

---

## Handoff

Next:

`M4.6 — Xây leakage-safe preprocessing pipeline`

M4.6 nhận hai nhóm feature:

`M4.4 transaction-level candidates`

+

`M4.5 behavioral candidates`

M4.6 phải xử lý:

- numerical branches;
- categorical branches;
- cold-start NA;
- `has_prior_card_history`;
- categorical unknown handling;
- scaling/transformation nếu cần;
- fit state chỉ từ TRAIN;
- transform VALIDATION;
- stable output feature names/order.

M4.6 không được tính lại behavioral history theo cách phá contract M4.5.

Đặc biệt:

- behavioral feature phải được tạo causal trước preprocessing;
- preprocessing state và historical event state là hai khái niệm khác nhau;
- learned preprocessing state chỉ được fit từ TRAIN;
- causal history vẫn được phép chứa transaction trước prediction point ngoài classifier-training membership.
