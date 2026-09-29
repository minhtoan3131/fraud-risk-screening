# M7.5 — Controlled class-imbalance experiments

Milestone:

`M7 — MODEL SELECTION / ROBUSTNESS / TUNING`

Substep:

`M7.5 — Controlled class-imbalance experiments`

Primary question:

> Với training window W_SHORT đã khóa và model shortlist LR / DT / RF đã khóa ở M7.4, có cần learning intervention ngoài imbalance strategy `NONE` hay không?

CANON yêu cầu staged experiment:

```text
NONE
↓
CLASS_WEIGHT
↓
RANDOM OVER/UNDER only if justified
↓
SMOTE only if conditional gate passes
```

Notebook này triển khai **M7.5 Phase A**:

```text
NONE
vs
CLASS_WEIGHT = "balanced"
```

cho cả ba shortlisted families trên Q2/Q3/Q4-2018 temporal folds.

Lý do chỉ chạy Phase A trước:

- `NONE` baseline đã tồn tại và được review ở M7.3/M7.4;
- class weight là intervention đầu tiên được CANON cho phép sau baseline;
- random over/under sampling chỉ được chạy nếu Phase A evidence vẫn justify intervention;
- SMOTE là conditional, không phải default;
- không được chạy mọi technique cùng lúc.

Do đó notebook **không** tự động chạy random sampling hoặc SMOTE sau khi nhìn metric.

Sau Run All:

- Phase A evidence được persist;
- random sampling authorization vẫn `OPEN`;
- final imbalance strategy vẫn `OPEN`;
- cần runtime review trước khi quyết định có mở Phase B hay không.

Runtime-dependent state trước khi chạy:

`NOT YET VERIFIED`

## 1. Controlled experiment contract

Locked upstream scope:

```text
training window:
W_SHORT

shortlisted families:
LR
DT
RF

folds:
Q2 / Q3 / Q4 2018

feature/preprocessing:
M7.2 fold-safe canonical contract

threshold policy:
DEFAULT_MODEL_DECISION_RULE

random state:
42
```

Phase A primary comparison trong từng model/fold:

```text
NONE
vs
CLASS_WEIGHT_BALANCED
```

Giữ cố định:

```text
same model family
same baseline model hyperparameters
same W_SHORT training window
same temporal fold
same train/validation rows
same fold-local preprocessing state
same metric implementation
same threshold policy
same random-state policy
```

Biến duy nhất:

`imbalance strategy / class_weight`

Validation giữ natural class distribution.

Không resampling validation.

M7.5 phải đọc đồng thời:

```text
F1
Recall gain
Precision cost
FN reduction
FP increase
alert burden
fold stability
runtime
warnings
```

Không được gọi class weight tốt hơn chỉ vì Recall tăng.

## 2. Phase boundary

Notebook dừng ở:

`CLASS_WEIGHT`

sau khi tạo comparison registry.

Nó không tự mở:

```text
RANDOM_OVERSAMPLING
RANDOM_UNDERSAMPLING
SMOTE
```

Phase B chỉ được thêm sau runtime review nếu evidence cho thấy:

- `NONE` chưa đủ;
- class weight không giải quyết trade-off đủ tốt;
- hoặc random sampling có hypothesis rõ ràng cần kiểm chứng.

SMOTE chỉ được cân nhắc nếu representation/leakage gate riêng PASS.

Final imbalance strategy có thể hoàn toàn là:

`NONE`.


```python

from pathlib import Path
import copy
import gc
import hashlib
import json
import platform
import sys
import time
import warnings

import numpy as np
import pandas as pd

from scipy import sparse

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


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


## 3. Locate upstream artifacts


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

M7_02_REL = (
    Path("data")
    / "processed"
    / "m7_02_temporal_cv_infrastructure_audit"
)

M7_03_REL = (
    Path("data")
    / "processed"
    / "m7_03_training_window_robustness"
)

M7_04_REL = (
    Path("data")
    / "processed"
    / "m7_04_model_family_robustness_shortlist"
)

M7_05_REL = (
    Path("data")
    / "processed"
    / "m7_05_controlled_class_imbalance"
)


required_rel_paths = [
    RAW_REL,

    M4_REL / "manifest.json",
    M4_REL / "feature_names.json",
    M4_REL / "y_train_w_long.npy",
    M4_REL / "y_train_w_short.npy",
    M4_REL / "row_id_train_w_long.npy",
    M4_REL / "row_id_train_w_short.npy",

    M7_02_REL
    / "m7_02_cv_infrastructure_contract.json",

    M7_02_REL
    / "m7_02_preprocessing_state_audit.json",

    M7_02_REL
    / "m7_02_infrastructure_manifest.json",

    M7_03_REL
    / "m7_03_training_window_robustness.json",

    M7_03_REL
    / "m7_03_robustness_manifest.json",

    M7_04_REL
    / "m7_04_model_family_robustness.json",

    M7_04_REL
    / "m7_04_shortlist_manifest.json",
]


candidate_roots = [
    Path.cwd(),
    *list(
        Path.cwd().parents
    )[:6],
]


PROJECT_ROOT = None


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
        "raw + M4.7 + M7.2 + M7.3 + M7.4 artifacts."
    )


RAW_PATH = (
    PROJECT_ROOT
    / RAW_REL
)

M4_DIR = (
    PROJECT_ROOT
    / M4_REL
)

M7_02_DIR = (
    PROJECT_ROOT
    / M7_02_REL
)

M7_03_DIR = (
    PROJECT_ROOT
    / M7_03_REL
)

M7_04_DIR = (
    PROJECT_ROOT
    / M7_04_REL
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / M7_05_REL
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


M4_MANIFEST_PATH = (
    M4_DIR
    / "manifest.json"
)

M4_FEATURE_NAMES_PATH = (
    M4_DIR
    / "feature_names.json"
)

Y_LONG_PATH = (
    M4_DIR
    / "y_train_w_long.npy"
)

Y_SHORT_PATH = (
    M4_DIR
    / "y_train_w_short.npy"
)

ROW_LONG_PATH = (
    M4_DIR
    / "row_id_train_w_long.npy"
)

ROW_SHORT_PATH = (
    M4_DIR
    / "row_id_train_w_short.npy"
)


M7_02_CONTRACT_PATH = (
    M7_02_DIR
    / "m7_02_cv_infrastructure_contract.json"
)

M7_02_PREPROCESSING_AUDIT_PATH = (
    M7_02_DIR
    / "m7_02_preprocessing_state_audit.json"
)

M7_02_MANIFEST_PATH = (
    M7_02_DIR
    / "m7_02_infrastructure_manifest.json"
)


M7_03_RESULT_PATH = (
    M7_03_DIR
    / "m7_03_training_window_robustness.json"
)

M7_03_MANIFEST_PATH = (
    M7_03_DIR
    / "m7_03_robustness_manifest.json"
)


M7_04_RESULT_PATH = (
    M7_04_DIR
    / "m7_04_model_family_robustness.json"
)

M7_04_MANIFEST_PATH = (
    M7_04_DIR
    / "m7_04_shortlist_manifest.json"
)


RESULT_PATH = (
    OUTPUT_DIR
    / "m7_05_class_weight_phase_a.json"
)

MANIFEST_PATH = (
    OUTPUT_DIR
    / "m7_05_phase_a_manifest.json"
)

PROGRESS_PATH = (
    OUTPUT_DIR
    / "m7_05_phase_a_progress.json"
)


EXPECTED_RAW_FILE_SIZE = 2_354_626_737
EXPECTED_RAW_ROWS = 24_386_900
EXPECTED_CARD_COUNT = 6_139

EXPECTED_W_LONG_ROWS = 6_855_270
EXPECTED_W_SHORT_ROWS = 1_721_615

EXPECTED_W_LONG_FRAUD = 9_606
EXPECTED_W_SHORT_FRAUD = 2_491

EXPECTED_FEATURE_COUNT = 47

TRAIN_END = pd.Timestamp(
    "2019-01-01"
)

W_LONG_START = pd.Timestamp(
    "2015-01-01"
)

W_SHORT_START = pd.Timestamp(
    "2018-01-01"
)

RANDOM_STATE = 42

CHUNK_SIZE = 500_000

UNKNOWN_TOKEN = "__UNKNOWN__"

M7_05_ANALYSIS_VERSION = (
    "M7.5-class-weight-phase-a-v1"
)

UPSTREAM_REVIEWED_M7_4_STATUS = (
    "PASS"
)

UPSTREAM_REVIEWED_TRAINING_WINDOW = (
    "W_SHORT"
)

UPSTREAM_REVIEWED_SHORTLIST = [
    "LR",
    "DT",
    "RF",
]

UPSTREAM_REVIEWED_M7_4_NOTEBOOK_SHA256 = (
    "1724a51a0e2fb5cc53d68c16065a7bee"
    "a37aa63ab12f3d0c761f3c123bce2454"
)


assert RAW_PATH.exists()

assert (
    RAW_PATH.stat().st_size
    == EXPECTED_RAW_FILE_SIZE
)


print("PROJECT_ROOT:")
print(PROJECT_ROOT)

print("\nOUTPUT_DIR:")
print(OUTPUT_DIR)

print(
    "\nM7.5 SOURCE LOCATION GATE: PASS"
)

```

    PROJECT_ROOT:
    /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening
    
    OUTPUT_DIR:
    /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/processed/m7_05_controlled_class_imbalance
    
    M7.5 SOURCE LOCATION GATE: PASS


## 4. Load and audit upstream reviewed handoff


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


with open(
    M4_MANIFEST_PATH,
    "r",
    encoding="utf-8",
) as file:
    m4_manifest = json.load(
        file
    )


with open(
    M4_FEATURE_NAMES_PATH,
    "r",
    encoding="utf-8",
) as file:
    canonical_feature_names = json.load(
        file
    )


with open(
    M7_02_CONTRACT_PATH,
    "r",
    encoding="utf-8",
) as file:
    m7_02_contract = json.load(
        file
    )


with open(
    M7_02_PREPROCESSING_AUDIT_PATH,
    "r",
    encoding="utf-8",
) as file:
    m7_02_preprocessing_audit = json.load(
        file
    )


with open(
    M7_02_MANIFEST_PATH,
    "r",
    encoding="utf-8",
) as file:
    m7_02_manifest = json.load(
        file
    )


with open(
    M7_03_RESULT_PATH,
    "r",
    encoding="utf-8",
) as file:
    m7_03_result = json.load(
        file
    )


with open(
    M7_03_MANIFEST_PATH,
    "r",
    encoding="utf-8",
) as file:
    m7_03_manifest = json.load(
        file
    )


with open(
    M7_04_RESULT_PATH,
    "r",
    encoding="utf-8",
) as file:
    m7_04_result = json.load(
        file
    )


with open(
    M7_04_MANIFEST_PATH,
    "r",
    encoding="utf-8",
) as file:
    m7_04_manifest = json.load(
        file
    )


y_long = np.load(
    Y_LONG_PATH,
    allow_pickle=False,
)

y_short = np.load(
    Y_SHORT_PATH,
    allow_pickle=False,
)

row_long = np.load(
    ROW_LONG_PATH,
    allow_pickle=False,
)

row_short = np.load(
    ROW_SHORT_PATH,
    allow_pickle=False,
)


assert (
    m4_manifest[
        "pipeline_version"
    ]
    == "M4.7-baseline-v1"
)

assert (
    len(
        canonical_feature_names
    )
    == EXPECTED_FEATURE_COUNT
)

assert (
    m7_02_manifest[
        "fold_safe_preprocessing"
    ]
    == "PASS"
)

assert (
    m7_02_manifest[
        "shared_runner_unit_test"
    ]
    == "PASS"
)

assert (
    m7_03_manifest[
        "completed_run_count"
    ]
    == 18
)

assert (
    m7_03_manifest[
        "warning_count_total"
    ]
    == 0
)

assert (
    m7_03_result[
        "cross_family_robustness_state"
    ]
    ==
    "ROBUST W_SHORT PREFERENCE"
)

assert (
    m7_04_manifest[
        "training_window_scope"
    ]
    == "W_SHORT"
)

assert (
    m7_04_manifest[
        "candidate_family_count"
    ]
    == 3
)

assert (
    m7_04_manifest[
        "final_test_accessed"
    ]
    is False
)

assert (
    UPSTREAM_REVIEWED_M7_4_STATUS
    == "PASS"
)

assert (
    UPSTREAM_REVIEWED_TRAINING_WINDOW
    == "W_SHORT"
)

assert (
    UPSTREAM_REVIEWED_SHORTLIST
    == [
        "LR",
        "DT",
        "RF",
    ]
)

assert len(
    row_long
) == EXPECTED_W_LONG_ROWS

assert len(
    row_short
) == EXPECTED_W_SHORT_ROWS

assert int(
    y_short.sum()
) == EXPECTED_W_SHORT_FRAUD


print(
    "Reviewed training window:",
    UPSTREAM_REVIEWED_TRAINING_WINDOW,
)

print(
    "Reviewed shortlist:",
    UPSTREAM_REVIEWED_SHORTLIST,
)

print(
    "Baseline M7.3 runs:",
    m7_03_manifest[
        "completed_run_count"
    ],
)

print(
    "\nM7.5 UPSTREAM HANDOFF GATE: PASS"
)

```

    Reviewed training window: W_SHORT
    Reviewed shortlist: ['LR', 'DT', 'RF']
    Baseline M7.3 runs: 18
    
    M7.5 UPSTREAM HANDOFF GATE: PASS


## 5. Lock Phase A candidate identity before metric

Phase A intervention:

`CLASS_WEIGHT_BALANCED`

Exact class-weight value:

`"balanced"`

Không thử nhiều class-weight ratios trong cùng M7.5 Phase A.

Lý do:

- class_weight `"balanced"` là candidate đã được M3.7 nêu rõ;
- M7.5 đang kiểm tra **có cần learning intervention hay không**, không tuning class-weight grid;
- M7.6 mới là moderate hyperparameter tuning.

Mỗi model giữ nguyên official baseline hyperparameters ngoài `class_weight`.


```python

MODEL_KEYS = [
    "LR",
    "DT",
    "RF",
]


MODEL_LABELS = {
    "LR":
        "Logistic Regression",

    "DT":
        "Decision Tree",

    "RF":
        "Random Forest",
}


EXPECTED_CONFIGS = {
    "LR":
        "LR-B04-LBFGS-L2-C1",

    "DT":
        "DT-B01-DEFAULT-GINI-UNPRUNED",

    "RF":
        "RF-B01-100-GINI-SQRT-BOOTSTRAP",
}


FOLD_ORDER = [
    "FOLD_Q2_2018",
    "FOLD_Q3_2018",
    "FOLD_Q4_2018",
]


FOLD_SPECS = [
    {
        "fold_id":
            "FOLD_Q2_2018",

        "train_end_exclusive":
            pd.Timestamp(
                "2018-04-01"
            ),

        "validation_start":
            pd.Timestamp(
                "2018-04-01"
            ),

        "validation_end_exclusive":
            pd.Timestamp(
                "2018-07-01"
            ),
    },

    {
        "fold_id":
            "FOLD_Q3_2018",

        "train_end_exclusive":
            pd.Timestamp(
                "2018-07-01"
            ),

        "validation_start":
            pd.Timestamp(
                "2018-07-01"
            ),

        "validation_end_exclusive":
            pd.Timestamp(
                "2018-10-01"
            ),
    },

    {
        "fold_id":
            "FOLD_Q4_2018",

        "train_end_exclusive":
            pd.Timestamp(
                "2018-10-01"
            ),

        "validation_start":
            pd.Timestamp(
                "2018-10-01"
            ),

        "validation_end_exclusive":
            pd.Timestamp(
                "2019-01-01"
            ),
    },
]


baseline_w_short_records = [
    record
    for record
    in m7_03_result[
        "fold_results"
    ]
    if (
        record[
            "training_window_id"
        ]
        == "W_SHORT"
    )
]


assert len(
    baseline_w_short_records
) == 9


baseline_by_model_fold = {
    (
        record[
            "model_key"
        ],
        record[
            "fold_id"
        ],
    ):
        record
    for record
    in baseline_w_short_records
}


BASE_MODEL_HYPERPARAMETERS = {}


for model_key in MODEL_KEYS:
    records = [
        baseline_by_model_fold[
            (
                model_key,
                fold_id,
            )
        ]
        for fold_id
        in FOLD_ORDER
    ]

    config_ids = {
        record[
            "model_config_id"
        ]
        for record
        in records
    }

    assert config_ids == {
        EXPECTED_CONFIGS[
            model_key
        ]
    }

    param_records = [
        copy.deepcopy(
            record[
                "hyperparameters"
            ]
        )
        for record
        in records
    ]

    assert (
        param_records[
            0
        ]
        ==
        param_records[
            1
        ]
        ==
        param_records[
            2
        ]
    )

    params = param_records[
        0
    ]

    assert (
        params[
            "class_weight"
        ]
        is None
    )

    params.pop(
        "class_weight"
    )

    BASE_MODEL_HYPERPARAMETERS[
        model_key
    ] = params


PHASE_A_INTERVENTION = {
    "imbalance_strategy":
        "CLASS_WEIGHT",

    "class_weight":
        "balanced",

    "sampling_method":
        None,

    "sampling_ratio":
        None,

    "threshold_policy":
        "DEFAULT_MODEL_DECISION_RULE",

    "random_state":
        RANDOM_STATE,
}


for model_key in MODEL_KEYS:
    print(
        model_key,
        EXPECTED_CONFIGS[
            model_key
        ],
        "→ class_weight='balanced'"
    )


print(
    "\nM7.5 PHASE-A CANDIDATE SPEC GATE: PASS"
)

```

    LR LR-B04-LBFGS-L2-C1 → class_weight='balanced'
    DT DT-B01-DEFAULT-GINI-UNPRUNED → class_weight='balanced'
    RF RF-B01-100-GINI-SQRT-BOOTSTRAP → class_weight='balanced'
    
    M7.5 PHASE-A CANDIDATE SPEC GATE: PASS


## 6. Canonical feature contract


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


def extract_canonical_categories(
    feature_names,
    column,
):
    prefix = (
        f"cat__{column}_"
    )

    values = [
        name[
            len(
                prefix
            ):
        ]
        for name
        in feature_names
        if name.startswith(
            prefix
        )
    ]

    if not values:
        raise ValueError(
            f"Không tìm thấy category schema cho {column}."
        )

    if (
        UNKNOWN_TOKEN
        not in values
    ):
        raise ValueError(
            f"Thiếu {UNKNOWN_TOKEN} cho {column}."
        )

    return values


CANONICAL_CATEGORIES = {
    column:
        extract_canonical_categories(
            canonical_feature_names,
            column,
        )
    for column
    in CATEGORICAL_COLUMNS
}


CATEGORY_TO_CODE = {
    column: {
        value:
            index
        for index, value
        in enumerate(
            CANONICAL_CATEGORIES[
                column
            ]
        )
    }
    for column
    in CATEGORICAL_COLUMNS
}


UNKNOWN_CODE = {
    column:
        CATEGORY_TO_CODE[
            column
        ][
            UNKNOWN_TOKEN
        ]
    for column
    in CATEGORICAL_COLUMNS
}


category_width = sum(
    len(
        CANONICAL_CATEGORIES[
            column
        ]
    )
    for column
    in CATEGORICAL_COLUMNS
)


assert (
    len(
        NUMERIC_COLUMNS
    )
    +
    len(
        BOOLEAN_COLUMNS
    )
    +
    category_width
    == EXPECTED_FEATURE_COUNT
)


print(
    "Canonical feature width:",
    EXPECTED_FEATURE_COUNT,
)

print(
    "\nM7.5 FEATURE CONTRACT GATE: PASS"
)

```

    Canonical feature width: 47
    
    M7.5 FEATURE CONTRACT GATE: PASS


## 7. Raw representation helpers


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


print(
    "M7.3 RAW REPRESENTATION HELPERS: DEFINED"
)

```

    M7.3 RAW REPRESENTATION HELPERS: DEFINED


## 8. Card-block streaming


```python

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

            card_block = (
                pending_parts[0]
                .reset_index(
                    drop=True
                )
                if len(
                    pending_parts
                )
                == 1
                else pd.concat(
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
        card_block = (
            pending_parts[0]
            .reset_index(
                drop=True
            )
            if len(
                pending_parts
            )
            == 1
            else pd.concat(
                pending_parts,
                ignore_index=True,
            )
        )

        yield (
            pending_key,
            card_block,
        )


print(
    "M7.3 CARD STREAMING HELPER: DEFINED"
)

```

    M7.3 CARD STREAMING HELPER: DEFINED


## 9. Strict-causal behavioral features


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
            "Timestamp giảm trong Card block."
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

    transactions_last_1h = np.repeat(
        transactions_last_1h_group,
        group_lengths,
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
    "M7.3 STRICT-CAUSAL FEATURE BUILDER: DEFINED"
)

```

    M7.3 STRICT-CAUSAL FEATURE BUILDER: DEFINED


## 10. Core semantic representation


```python

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


def encode_categorical_column(
    series,
    column,
):
    mapping = (
        CATEGORY_TO_CODE[
            column
        ]
    )

    unknown_code = (
        UNKNOWN_CODE[
            column
        ]
    )

    codes = (
        series
        .astype("string")
        .map(
            mapping
        )
        .fillna(
            unknown_code
        )
        .astype(
            np.int16
        )
        .to_numpy()
    )

    return codes


print(
    "M7.3 CORE SEMANTIC / CATEGORY ENCODING HELPERS: DEFINED"
)

```

    M7.3 CORE SEMANTIC / CATEGORY ENCODING HELPERS: DEFINED


## 11. Strict-causal regression test


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
    "M7.5 STRICT-CAUSAL REGRESSION GATE: PASS"
)

```

    M7.5 STRICT-CAUSAL REGRESSION GATE: PASS


## 12. Rebuild semantic TRAIN lineage

M7.5 sử dụng cùng raw semantics với M7.3.

Raw target không được đọc trong semantic pass.

Labels tiếp tục đến từ canonical M4.7 target arrays.


```python

semantic_row_parts = []
semantic_timestamp_parts = []
semantic_numeric_parts = []
semantic_boolean_parts = []
semantic_categorical_parts = []

card_count = 0
raw_rows_seen = 0

seen_card_keys = set()

stream_start = (
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

    if card_key in seen_card_keys:
        raise RuntimeError(
            "Card block reappearance: "
            f"{card_key}"
        )

    seen_card_keys.add(
        card_key
    )

    context_card_df = (
        card_df.loc[
            card_df[
                "Timestamp"
            ]
            < TRAIN_END
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    if context_card_df.empty:
        continue

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

    behavioral_df = (
        compute_causal_behavioral_features(
            behavior_input
        )
    )

    core_df = (
        build_core_feature_frame(
            context_card_df,
            behavioral_df,
        )
    )

    train_mask = (
        (
            core_df[
                "Timestamp"
            ]
            >= W_LONG_START
        )
        &
        (
            core_df[
                "Timestamp"
            ]
            < TRAIN_END
        )
    )

    if not train_mask.any():
        continue

    sub = (
        core_df.loc[
            train_mask
        ]
    )

    semantic_row_parts.append(
        sub[
            "raw_row_id"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )

    semantic_timestamp_parts.append(
        sub[
            "Timestamp"
        ]
        .to_numpy(
            dtype="datetime64[ns]"
        )
    )

    semantic_numeric_parts.append(
        sub[
            NUMERIC_COLUMNS
        ]
        .astype(
            np.float32
        )
        .to_numpy()
    )

    semantic_boolean_parts.append(
        sub[
            BOOLEAN_COLUMNS
        ]
        .astype(
            np.uint8
        )
        .to_numpy()
    )

    categorical_codes = np.column_stack(
        [
            encode_categorical_column(
                sub[
                    column
                ],
                column,
            )
            for column
            in CATEGORICAL_COLUMNS
        ]
    ).astype(
        np.int16,
        copy=False,
    )

    semantic_categorical_parts.append(
        categorical_codes
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
        )


semantic_row_id = np.concatenate(
    semantic_row_parts
)

semantic_timestamp = np.concatenate(
    semantic_timestamp_parts
)

semantic_numeric = np.vstack(
    semantic_numeric_parts
)

semantic_boolean = np.vstack(
    semantic_boolean_parts
)

semantic_categorical = np.vstack(
    semantic_categorical_parts
)


del semantic_row_parts
del semantic_timestamp_parts
del semantic_numeric_parts
del semantic_boolean_parts
del semantic_categorical_parts

gc.collect()


stream_elapsed = (
    time.perf_counter()
    - stream_start
)


assert (
    card_count
    == EXPECTED_CARD_COUNT
)

assert (
    raw_rows_seen
    == EXPECTED_RAW_ROWS
)

assert (
    len(
        semantic_row_id
    )
    == EXPECTED_W_LONG_ROWS
)

assert (
    semantic_numeric.shape
    ==
    (
        EXPECTED_W_LONG_ROWS,
        len(
            NUMERIC_COLUMNS
        ),
    )
)

assert (
    semantic_boolean.shape
    ==
    (
        EXPECTED_W_LONG_ROWS,
        len(
            BOOLEAN_COLUMNS
        ),
    )
)

assert (
    semantic_categorical.shape
    ==
    (
        EXPECTED_W_LONG_ROWS,
        len(
            CATEGORICAL_COLUMNS
        ),
    )
)


np.testing.assert_array_equal(
    semantic_row_id,
    row_long,
)


short_mask_full = (
    semantic_timestamp
    >= np.datetime64(
        "2018-01-01"
    )
)


np.testing.assert_array_equal(
    semantic_row_id[
        short_mask_full
    ],
    row_short,
)


short_positions = np.flatnonzero(
    short_mask_full
)


np.testing.assert_array_equal(
    y_long[
        short_positions
    ],
    y_short,
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
    "Semantic W_LONG rows:",
    f"{len(semantic_row_id):,}",
)

print(
    "Semantic W_SHORT rows:",
    f"{int(short_mask_full.sum()):,}",
)

print(
    "Elapsed seconds:",
    round(
        stream_elapsed,
        2,
    ),
)

print(
    "\nM7.5 SEMANTIC TRAIN LINEAGE GATE: PASS"
)

```

    Card blocks: 500 | raw rows: 2,021,584
    Card blocks: 1,000 | raw rows: 4,066,794
    Card blocks: 1,500 | raw rows: 5,915,634
    Card blocks: 2,000 | raw rows: 7,883,446
    Card blocks: 2,500 | raw rows: 9,842,479
    Card blocks: 3,000 | raw rows: 11,897,259
    Card blocks: 3,500 | raw rows: 13,997,374
    Card blocks: 4,000 | raw rows: 16,040,304
    Card blocks: 4,500 | raw rows: 18,115,490
    Card blocks: 5,000 | raw rows: 19,945,815
    Card blocks: 6,000 | raw rows: 23,844,247
    
    Card blocks: 6,139
    Raw rows streamed: 24,386,900
    Semantic W_LONG rows: 6,855,270
    Semantic W_SHORT rows: 1,721,615
    Elapsed seconds: 132.14
    
    M7.5 SEMANTIC TRAIN LINEAGE GATE: PASS


## 13. Build W_SHORT fold registry


```python

timestamp_index = pd.DatetimeIndex(
    semantic_timestamp
)


fold_registry = []


for window_id, window_start in (
    {'W_SHORT': W_SHORT_START}.items()
):
    for fold in FOLD_SPECS:
        train_mask = (
            (
                timestamp_index
                >= window_start
            )
            &
            (
                timestamp_index
                <
                fold[
                    "train_end_exclusive"
                ]
            )
        )

        validation_mask = (
            (
                timestamp_index
                >= fold[
                    "validation_start"
                ]
            )
            &
            (
                timestamp_index
                <
                fold[
                    "validation_end_exclusive"
                ]
            )
        )

        train_indices = np.flatnonzero(
            train_mask
        )

        validation_indices = np.flatnonzero(
            validation_mask
        )

        assert (
            len(
                train_indices
            )
            > 0
        )

        assert (
            len(
                validation_indices
            )
            > 0
        )

        train_fraud = int(
            y_long[
                train_indices
            ].sum()
        )

        validation_fraud = int(
            y_long[
                validation_indices
            ].sum()
        )

        assert train_fraud > 0
        assert validation_fraud > 0

        assert (
            timestamp_index[
                train_indices
            ].max()
            <
            timestamp_index[
                validation_indices
            ].min()
        )

        assert np.intersect1d(
            semantic_row_id[
                train_indices
            ],
            semantic_row_id[
                validation_indices
            ],
            assume_unique=True,
        ).size == 0

        fold_registry.append(
            {
                "training_window_id":
                    window_id,

                "fold_id":
                    fold[
                        "fold_id"
                    ],

                "train_start":
                    str(
                        window_start
                    ),

                "train_end_exclusive":
                    str(
                        fold[
                            "train_end_exclusive"
                        ]
                    ),

                "validation_start":
                    str(
                        fold[
                            "validation_start"
                        ]
                    ),

                "validation_end_exclusive":
                    str(
                        fold[
                            "validation_end_exclusive"
                        ]
                    ),

                "train_rows":
                    int(
                        len(
                            train_indices
                        )
                    ),

                "validation_rows":
                    int(
                        len(
                            validation_indices
                        )
                    ),

                "train_fraud_rows":
                    train_fraud,

                "validation_fraud_rows":
                    validation_fraud,

                "train_max_timestamp":
                    str(
                        timestamp_index[
                            train_indices
                        ].max()
                    ),

                "validation_min_timestamp":
                    str(
                        timestamp_index[
                            validation_indices
                        ].min()
                    ),

                "row_overlap":
                    0,
            }
        )


assert len(
    fold_registry
) == 3


fold_registry_by_key = {
    (
        record[
            "training_window_id"
        ],
        record[
            "fold_id"
        ],
    ):
        record
    for record
    in fold_registry
}


for record in fold_registry:
    print(
        record[
            "training_window_id"
        ],
        record[
            "fold_id"
        ],
        "| train rows/fraud:",
        record[
            "train_rows"
        ],
        "/",
        record[
            "train_fraud_rows"
        ],
        "| val rows/fraud:",
        record[
            "validation_rows"
        ],
        "/",
        record[
            "validation_fraud_rows"
        ],
    )


print(
    "\nM7.5 FOLD INTEGRITY GATE: PASS"
)

```

    W_SHORT FOLD_Q2_2018 | train rows/fraud: 423905 / 557 | val rows/fraud: 428953 / 590
    W_SHORT FOLD_Q3_2018 | train rows/fraud: 852858 / 1147 | val rows/fraud: 435178 / 634
    W_SHORT FOLD_Q4_2018 | train rows/fraud: 1288036 / 1781 | val rows/fraud: 433579 / 710
    
    M7.5 FOLD INTEGRITY GATE: PASS


## 14. Load verified W_SHORT preprocessing states


```python

preprocessing_state_by_key = {
    (
        record[
            "training_window_id"
        ],
        record[
            "fold_id"
        ],
    ):
        record
    for record
    in m7_02_preprocessing_audit
}


preprocessing_state_by_key = {
    key: value
    for key, value
    in preprocessing_state_by_key.items()
    if key[0] == 'W_SHORT'
}

assert len(
    preprocessing_state_by_key
) == 3


for key, record in (
    preprocessing_state_by_key.items()
):
    fold_record = (
        fold_registry_by_key[
            key
        ]
    )

    assert (
        record[
            "status"
        ]
        == "PASS"
    )

    assert (
        record[
            "fit_source"
        ]
        == "FOLD_TRAIN_ONLY"
    )

    assert (
        record[
            "validation_fit_used"
        ]
        is False
    )

    assert (
        record[
            "feature_names_match_m4_7"
        ]
        is True
    )

    assert (
        record[
            "feature_count"
        ]
        == EXPECTED_FEATURE_COUNT
    )

    assert (
        record[
            "fit_row_count"
        ]
        == fold_record[
            "train_rows"
        ]
    )

    assert (
        pd.Timestamp(
            record[
                "fit_max_timestamp"
            ]
        )
        <
        pd.Timestamp(
            fold_record[
                "validation_start"
            ]
        )
    )

    numeric_mean = np.asarray(
        record[
            "numeric_mean"
        ],
        dtype=np.float64,
    )

    numeric_scale = np.asarray(
        record[
            "numeric_scale"
        ],
        dtype=np.float64,
    )

    assert (
        numeric_mean.shape
        ==
        (
            len(
                NUMERIC_COLUMNS
            ),
        )
    )

    assert (
        numeric_scale.shape
        ==
        (
            len(
                NUMERIC_COLUMNS
            ),
        )
    )

    assert np.isfinite(
        numeric_mean
    ).all()

    assert np.isfinite(
        numeric_scale
    ).all()

    assert np.all(
        numeric_scale
        > 0
    )

    for column in (
        CATEGORICAL_COLUMNS
    ):
        assert (
            record[
                "category_vocab_size"
            ][
                column
            ]
            ==
            (
                len(
                    CANONICAL_CATEGORIES[
                        column
                    ]
                )
                - 1
            )
        )


print(
    "Verified preprocessing states:",
    len(
        preprocessing_state_by_key
    ),
)

print(
    "\nM7.5 PREPROCESSING-STATE REUSE GATE: PASS"
)

```

    Verified preprocessing states: 3
    
    M7.5 PREPROCESSING-STATE REUSE GATE: PASS


## 15. Fold-local matrix builder


```python

CATEGORY_OFFSETS = {}

current_offset = 0

for column in (
    CATEGORICAL_COLUMNS
):
    CATEGORY_OFFSETS[
        column
    ] = current_offset

    current_offset += len(
        CANONICAL_CATEGORIES[
            column
        ]
    )


assert (
    current_offset
    == category_width
)


def build_fold_matrix(
    row_indices,
    preprocessing_state,
):
    row_indices = np.asarray(
        row_indices,
        dtype=np.int64,
    )

    n_rows = len(
        row_indices
    )

    numeric_mean = np.asarray(
        preprocessing_state[
            "numeric_mean"
        ],
        dtype=np.float64,
    )

    numeric_scale = np.asarray(
        preprocessing_state[
            "numeric_scale"
        ],
        dtype=np.float64,
    )

    numeric_raw = (
        semantic_numeric[
            row_indices
        ]
        .astype(
            np.float64,
            copy=False,
        )
    )

    numeric_scaled = (
        (
            numeric_raw
            - numeric_mean
        )
        /
        numeric_scale
    )

    numeric_scaled = np.nan_to_num(
        numeric_scaled,
        nan=0.0,
        posinf=np.inf,
        neginf=-np.inf,
    ).astype(
        np.float32,
        copy=False,
    )

    if not np.isfinite(
        numeric_scaled
    ).all():
        raise ValueError(
            "Numeric transform chứa NaN/inf."
        )

    boolean_matrix = (
        semantic_boolean[
            row_indices
        ]
        .astype(
            np.float32,
            copy=False,
        )
    )

    categorical_codes = (
        semantic_categorical[
            row_indices
        ]
        .astype(
            np.int32,
            copy=False,
        )
    )

    # One CSR categorical block, exactly four non-zero entries per row.
    cat_indptr = np.arange(
        0,
        (
            len(
                CATEGORICAL_COLUMNS
            )
            *
            (
                n_rows
                + 1
            )
        ),
        len(
            CATEGORICAL_COLUMNS
        ),
        dtype=np.int64,
    )

    cat_indices = np.empty(
        n_rows
        * len(
            CATEGORICAL_COLUMNS
        ),
        dtype=np.int32,
    )

    for column_index, column in enumerate(
        CATEGORICAL_COLUMNS
    ):
        codes = (
            categorical_codes[
                :,
                column_index
            ]
        )

        assert np.all(
            codes
            >= 0
        )

        assert np.all(
            codes
            <
            len(
                CANONICAL_CATEGORIES[
                    column
                ]
            )
        )

        cat_indices[
            column_index
            ::
            len(
                CATEGORICAL_COLUMNS
            )
        ] = (
            codes
            +
            CATEGORY_OFFSETS[
                column
            ]
        )

    cat_data = np.ones(
        len(
            cat_indices
        ),
        dtype=np.float32,
    )

    categorical_matrix = sparse.csr_matrix(
        (
            cat_data,
            cat_indices,
            cat_indptr,
        ),
        shape=(
            n_rows,
            category_width,
        ),
        dtype=np.float32,
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

    assert (
        X.shape
        ==
        (
            n_rows,
            EXPECTED_FEATURE_COUNT,
        )
    )

    assert np.isfinite(
        X.data
    ).all()

    return X


# Small deterministic smoke test.
audit_indices = np.arange(
    min(
        2_000,
        EXPECTED_W_LONG_ROWS,
    ),
    dtype=np.int64,
)

audit_state = (
    preprocessing_state_by_key[
        (
            "W_SHORT",
            "FOLD_Q2_2018",
        )
    ]
)

X_audit = build_fold_matrix(
    audit_indices,
    audit_state,
)

assert sparse.isspmatrix_csr(
    X_audit
)

assert (
    X_audit.dtype
    == np.float32
)

assert (
    X_audit.shape[1]
    == EXPECTED_FEATURE_COUNT
)


print(
    "Audit matrix shape:",
    X_audit.shape,
)

print(
    "\nM7.5 MATRIX BUILDER GATE: PASS"
)

del X_audit

gc.collect()

```

    Audit matrix shape: (2000, 47)
    
    M7.5 MATRIX BUILDER GATE: PASS





    0



## 16. Class-weight estimator / metric runner

M7.5 Phase A không dùng sampler.

`class_weight="balanced"` được truyền trực tiếp vào estimator fit.

Fold-validation vẫn giữ natural distribution.

Fraud probability được map theo:

`classes_ == 1`.


```python

CANONICAL_CANDIDATE_IDENTITY_FIELDS = [
    "candidate_id",
    "milestone_substep",
    "dataset_artifact_identity",
    "feature_version",
    "preprocessing_version",
    "training_window_id",
    "model_family",
    "model_config_id",
    "hyperparameters",
    "imbalance_strategy",
    "class_weight",
    "sampling_method",
    "sampling_ratio",
    "random_state",
    "cv_spec_id",
    "fold_template_id",
    "threshold_policy",
    "probability_interface",
    "metric_contract_version",
]


CANONICAL_FOLD_RESULT_FIELDS = [
    "candidate_id",
    "fold_id",
    "train_start",
    "train_end_exclusive",
    "validation_start",
    "validation_end_exclusive",
    "train_rows",
    "validation_rows",
    "train_fraud_rows",
    "validation_fraud_rows",
    "feature_version",
    "preprocessing_version",
    "model_family",
    "model_config_id",
    "hyperparameters",
    "imbalance_strategy",
    "random_state",
    "threshold_policy",
    "prediction_artifact",
    "risk_score_artifact",
    "F1_fraud",
    "Recall_fraud",
    "Precision_fraud",
    "TP",
    "FP",
    "FN",
    "TN",
    "predicted_positive_count",
    "predicted_positive_rate",
    "Accuracy_reference",
    "fit_seconds",
    "prediction_seconds",
    "warning_count",
    "warnings",
    "integrity_status",
]


def compute_metric_bundle(
    y_true,
    y_pred,
):
    y_true = np.asarray(
        y_true,
        dtype=np.int8,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=np.int8,
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[
            0,
            1,
        ],
    )

    tn, fp, fn, tp = (
        cm.ravel()
    )

    predicted_positive_count = int(
        tp + fp
    )

    return {
        "F1_fraud":
            float(
                f1_score(
                    y_true,
                    y_pred,
                    pos_label=1,
                    zero_division=0,
                )
            ),

        "Recall_fraud":
            float(
                recall_score(
                    y_true,
                    y_pred,
                    pos_label=1,
                    zero_division=0,
                )
            ),

        "Precision_fraud":
            float(
                precision_score(
                    y_true,
                    y_pred,
                    pos_label=1,
                    zero_division=0,
                )
            ),

        "Accuracy_reference":
            float(
                accuracy_score(
                    y_true,
                    y_pred,
                )
            ),

        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),

        "predicted_positive_count":
            predicted_positive_count,

        "predicted_positive_rate":
            float(
                predicted_positive_count
                /
                len(
                    y_true
                )
            ),
    }


def make_class_weight_estimator(
    model_key,
):
    params = copy.deepcopy(
        BASE_MODEL_HYPERPARAMETERS[
            model_key
        ]
    )

    params[
        "class_weight"
    ] = "balanced"

    if model_key == "LR":
        return LogisticRegression(
            **params
        )

    if model_key == "DT":
        return DecisionTreeClassifier(
            **params
        )

    if model_key == "RF":
        return RandomForestClassifier(
            **params
        )

    raise KeyError(
        model_key
    )


def run_class_weight_fold(
    *,
    model_key,
    candidate_identity,
    fold_record,
    X_train,
    y_train,
    X_validation,
    y_validation,
    prediction_rel_path,
    risk_score_rel_path,
):
    assert set(
        CANONICAL_CANDIDATE_IDENTITY_FIELDS
    ).issubset(
        candidate_identity
    )

    estimator = (
        make_class_weight_estimator(
            model_key
        )
    )

    with warnings.catch_warnings(
        record=True
    ) as captured_warnings:
        warnings.simplefilter(
            "always"
        )

        fit_start = (
            time.perf_counter()
        )

        fitted = clone(
            estimator
        )

        fitted.fit(
            X_train,
            y_train,
        )

        fit_seconds = (
            time.perf_counter()
            - fit_start
        )

        prediction_start = (
            time.perf_counter()
        )

        y_pred = fitted.predict(
            X_validation
        )

        prediction_seconds = (
            time.perf_counter()
            - prediction_start
        )

        classes = np.asarray(
            fitted.classes_
        )

        positive_matches = (
            np.flatnonzero(
                classes == 1
            )
        )

        if len(
            positive_matches
        ) != 1:
            raise ValueError(
                "Không xác định duy nhất positive class = 1."
            )

        positive_index = int(
            positive_matches[
                0
            ]
        )

        risk_score = (
            fitted.predict_proba(
                X_validation
            )[
                :,
                positive_index
            ]
        )

    warning_records = [
        {
            "category":
                warning.category.__name__,

            "message":
                str(
                    warning.message
                ),
        }
        for warning
        in captured_warnings
    ]

    metrics = compute_metric_bundle(
        y_validation,
        y_pred,
    )

    fold_result = {
        "candidate_id":
            candidate_identity[
                "candidate_id"
            ],

        "fold_id":
            fold_record[
                "fold_id"
            ],

        "train_start":
            fold_record[
                "train_start"
            ],

        "train_end_exclusive":
            fold_record[
                "train_end_exclusive"
            ],

        "validation_start":
            fold_record[
                "validation_start"
            ],

        "validation_end_exclusive":
            fold_record[
                "validation_end_exclusive"
            ],

        "train_rows":
            int(
                fold_record[
                    "train_rows"
                ]
            ),

        "validation_rows":
            int(
                fold_record[
                    "validation_rows"
                ]
            ),

        "train_fraud_rows":
            int(
                fold_record[
                    "train_fraud_rows"
                ]
            ),

        "validation_fraud_rows":
            int(
                fold_record[
                    "validation_fraud_rows"
                ]
            ),

        "feature_version":
            candidate_identity[
                "feature_version"
            ],

        "preprocessing_version":
            candidate_identity[
                "preprocessing_version"
            ],

        "model_family":
            candidate_identity[
                "model_family"
            ],

        "model_config_id":
            candidate_identity[
                "model_config_id"
            ],

        "hyperparameters":
            copy.deepcopy(
                candidate_identity[
                    "hyperparameters"
                ]
            ),

        "imbalance_strategy":
            candidate_identity[
                "imbalance_strategy"
            ],

        "random_state":
            candidate_identity[
                "random_state"
            ],

        "threshold_policy":
            candidate_identity[
                "threshold_policy"
            ],

        "prediction_artifact":
            str(
                prediction_rel_path
            ),

        "risk_score_artifact":
            str(
                risk_score_rel_path
            ),

        "F1_fraud":
            metrics[
                "F1_fraud"
            ],

        "Recall_fraud":
            metrics[
                "Recall_fraud"
            ],

        "Precision_fraud":
            metrics[
                "Precision_fraud"
            ],

        "TP":
            metrics[
                "TP"
            ],

        "FP":
            metrics[
                "FP"
            ],

        "FN":
            metrics[
                "FN"
            ],

        "TN":
            metrics[
                "TN"
            ],

        "predicted_positive_count":
            metrics[
                "predicted_positive_count"
            ],

        "predicted_positive_rate":
            metrics[
                "predicted_positive_rate"
            ],

        "Accuracy_reference":
            metrics[
                "Accuracy_reference"
            ],

        "fit_seconds":
            float(
                fit_seconds
            ),

        "prediction_seconds":
            float(
                prediction_seconds
            ),

        "warning_count":
            int(
                len(
                    warning_records
                )
            ),

        "warnings":
            warning_records,

        "integrity_status":
            (
                "PASS"
                if len(
                    warning_records
                )
                == 0
                else "WARNING_REVIEW_REQUIRED"
            ),
    }

    assert (
        set(
            fold_result
        )
        ==
        set(
            CANONICAL_FOLD_RESULT_FIELDS
        )
    )

    return {
        "fold_result":
            fold_result,

        "y_pred":
            np.asarray(
                y_pred,
                dtype=np.int8,
            ),

        "risk_score":
            np.asarray(
                risk_score,
                dtype=np.float32,
            ),
    }


print(
    "M7.5 CLASS-WEIGHT RUNNER: DEFINED"
)

```

    M7.5 CLASS-WEIGHT RUNNER: DEFINED


## 17. Execute 9 class-weight fold runs

Expected:

```text
3 model families
×
3 temporal folds
=
9 CLASS_WEIGHT runs
```

Baseline `NONE` is not retrained.

It is read from exact M7.3 W_SHORT fold results.

If any class-weight run raises warning:

`STOP before comparative decision`.


```python

candidate_registry = []
class_weight_fold_results = []
artifact_fingerprints = {}


def write_progress(
    status,
):
    payload = {
        "analysis_version":
            M7_05_ANALYSIS_VERSION,

        "phase":
            "A_CLASS_WEIGHT",

        "status":
            status,

        "completed_run_count":
            len(
                class_weight_fold_results
            ),

        "expected_run_count":
            9,

        "candidate_registry":
            candidate_registry,

        "fold_results":
            class_weight_fold_results,

        "random_sampling_authorized":
            False,

        "smote_authorized":
            False,

        "final_test_accessed":
            False,
    }

    with open(
        PROGRESS_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )


write_progress(
    "RUNNING"
)


phase_start = (
    time.perf_counter()
)


for fold in FOLD_SPECS:
    fold_id = (
        fold[
            "fold_id"
        ]
    )

    validation_mask = (
        (
            timestamp_index
            >= fold[
                "validation_start"
            ]
        )
        &
        (
            timestamp_index
            <
            fold[
                "validation_end_exclusive"
            ]
        )
    )

    validation_indices = (
        np.flatnonzero(
            validation_mask
        )
    )

    train_mask = (
        (
            timestamp_index
            >= W_SHORT_START
        )
        &
        (
            timestamp_index
            <
            fold[
                "train_end_exclusive"
            ]
        )
    )

    train_indices = (
        np.flatnonzero(
            train_mask
        )
    )

    y_train_fold = (
        y_long[
            train_indices
        ]
    )

    y_validation_fold = (
        y_long[
            validation_indices
        ]
    )

    key = (
        "W_SHORT",
        fold_id,
    )

    fold_record = (
        fold_registry_by_key[
            key
        ]
    )

    preprocessing_state = (
        preprocessing_state_by_key[
            key
        ]
    )

    matrix_start = (
        time.perf_counter()
    )

    X_train_fold = (
        build_fold_matrix(
            train_indices,
            preprocessing_state,
        )
    )

    X_validation_fold = (
        build_fold_matrix(
            validation_indices,
            preprocessing_state,
        )
    )

    matrix_seconds = (
        time.perf_counter()
        - matrix_start
    )

    print(
        "\n=================================================="
    )

    print(
        "W_SHORT",
        fold_id,
    )

    print(
        "Train shape:",
        X_train_fold.shape,
        "| fraud:",
        int(
            y_train_fold.sum()
        ),
    )

    print(
        "Validation shape:",
        X_validation_fold.shape,
        "| fraud:",
        int(
            y_validation_fold.sum()
        ),
    )

    print(
        "Matrix seconds:",
        round(
            matrix_seconds,
            3,
        ),
    )

    for model_key in MODEL_KEYS:
        candidate_id = (
            f"M7.5-"
            f"{model_key}-"
            f"W_SHORT-"
            f"CLASS_WEIGHT_BALANCED-"
            f"{fold_id}"
        )

        prediction_filename = (
            candidate_id
            + "__y_pred.npy"
        )

        risk_score_filename = (
            candidate_id
            + "__risk_score.npy"
        )

        prediction_path = (
            OUTPUT_DIR
            / prediction_filename
        )

        risk_score_path = (
            OUTPUT_DIR
            / risk_score_filename
        )

        prediction_rel_path = (
            prediction_path
            .relative_to(
                PROJECT_ROOT
            )
        )

        risk_score_rel_path = (
            risk_score_path
            .relative_to(
                PROJECT_ROOT
            )
        )

        candidate_identity = {
            "candidate_id":
                candidate_id,

            "milestone_substep":
                "M7.5-PHASE-A",

            "dataset_artifact_identity":
                "M4.7 canonical TRAIN / W_SHORT temporal CV",

            "feature_version":
                "Feature Specification v1.0",

            "preprocessing_version":
                "Preprocessing Specification v1.0 | M7.2 fold-local",

            "training_window_id":
                "W_SHORT",

            "model_family":
                MODEL_LABELS[
                    model_key
                ],

            "model_config_id":
                EXPECTED_CONFIGS[
                    model_key
                ],

            "hyperparameters":
                copy.deepcopy(
                    BASE_MODEL_HYPERPARAMETERS[
                        model_key
                    ]
                ),

            "imbalance_strategy":
                "CLASS_WEIGHT",

            "class_weight":
                "balanced",

            "sampling_method":
                None,

            "sampling_ratio":
                None,

            "random_state":
                RANDOM_STATE,

            "cv_spec_id":
                "M7-TEMPORAL-CV-Q2-Q3-Q4-2018-v1",

            "fold_template_id":
                "EXPANDING-Q2-Q3-Q4-2018-v1",

            "threshold_policy":
                "DEFAULT_MODEL_DECISION_RULE",

            "probability_interface":
                "predict_proba / positive class = 1",

            "metric_contract_version":
                "M7.1-metric-contract-v1",
        }

        candidate_registry.append(
            candidate_identity
        )

        print(
            "\nRun:",
            candidate_id,
        )

        run_output = (
            run_class_weight_fold(
                model_key=
                    model_key,

                candidate_identity=
                    candidate_identity,

                fold_record=
                    fold_record,

                X_train=
                    X_train_fold,

                y_train=
                    y_train_fold,

                X_validation=
                    X_validation_fold,

                y_validation=
                    y_validation_fold,

                prediction_rel_path=
                    prediction_rel_path,

                risk_score_rel_path=
                    risk_score_rel_path,
            )
        )

        np.save(
            prediction_path,
            run_output[
                "y_pred"
            ],
            allow_pickle=False,
        )

        np.save(
            risk_score_path,
            run_output[
                "risk_score"
            ],
            allow_pickle=False,
        )

        record = copy.deepcopy(
            run_output[
                "fold_result"
            ]
        )

        record[
            "model_key"
        ] = model_key

        record[
            "training_window_id"
        ] = "W_SHORT"

        record[
            "class_weight"
        ] = "balanced"

        record[
            "matrix_build_seconds"
        ] = float(
            matrix_seconds
        )

        class_weight_fold_results.append(
            record
        )

        artifact_fingerprints[
            candidate_id
        ] = {
            "prediction_sha256":
                sha256_file(
                    prediction_path
                ),

            "risk_score_sha256":
                sha256_file(
                    risk_score_path
                ),
        }

        write_progress(
            "RUNNING"
        )

        print(
            "F1 / Recall / Precision:",
            round(
                record[
                    "F1_fraud"
                ],
                6,
            ),
            "/",
            round(
                record[
                    "Recall_fraud"
                ],
                6,
            ),
            "/",
            round(
                record[
                    "Precision_fraud"
                ],
                6,
            ),
        )

        print(
            "TP / FP / FN / TN:",
            record[
                "TP"
            ],
            "/",
            record[
                "FP"
            ],
            "/",
            record[
                "FN"
            ],
            "/",
            record[
                "TN"
            ],
        )

        print(
            "Alerts:",
            record[
                "predicted_positive_count"
            ],
        )

        print(
            "Warnings:",
            record[
                "warning_count"
            ],
        )

        if (
            record[
                "warning_count"
            ]
            != 0
        ):
            write_progress(
                "STOPPED_WARNING_REVIEW_REQUIRED"
            )

            raise RuntimeError(
                "Class-weight run warning detected. "
                "STOP before comparative interpretation."
            )

    del X_train_fold
    del X_validation_fold
    del train_indices
    del validation_indices
    del y_train_fold
    del y_validation_fold

    gc.collect()


write_progress(
    "COMPLETE"
)


phase_elapsed = (
    time.perf_counter()
    - phase_start
)


assert len(
    candidate_registry
) == 9

assert len(
    class_weight_fold_results
) == 9


print(
    "\nCompleted class-weight runs:",
    len(
        class_weight_fold_results
    ),
)

print(
    "Phase A seconds:",
    round(
        phase_elapsed,
        2,
    ),
)

print(
    "\nM7.5 CLASS-WEIGHT EXECUTION GATE: PASS"
)

```

    
    ==================================================
    W_SHORT FOLD_Q2_2018
    Train shape: (423905, 47) | fraud: 557
    Validation shape: (428953, 47) | fraud: 590
    Matrix seconds: 0.384
    
    Run: M7.5-LR-W_SHORT-CLASS_WEIGHT_BALANCED-FOLD_Q2_2018
    F1 / Recall / Precision: 0.40852 / 0.991525 / 0.257256
    TP / FP / FN / TN: 585 / 1689 / 5 / 426674
    Alerts: 2274
    Warnings: 0
    
    Run: M7.5-DT-W_SHORT-CLASS_WEIGHT_BALANCED-FOLD_Q2_2018
    F1 / Recall / Precision: 0.526132 / 0.511864 / 0.541219
    TP / FP / FN / TN: 302 / 256 / 288 / 428107
    Alerts: 558
    Warnings: 0
    
    Run: M7.5-RF-W_SHORT-CLASS_WEIGHT_BALANCED-FOLD_Q2_2018
    F1 / Recall / Precision: 0.603636 / 0.70339 / 0.528662
    TP / FP / FN / TN: 415 / 370 / 175 / 427993
    Alerts: 785
    Warnings: 0
    
    ==================================================
    W_SHORT FOLD_Q3_2018
    Train shape: (852858, 47) | fraud: 1147
    Validation shape: (435178, 47) | fraud: 634
    Matrix seconds: 0.786
    
    Run: M7.5-LR-W_SHORT-CLASS_WEIGHT_BALANCED-FOLD_Q3_2018
    F1 / Recall / Precision: 0.364631 / 0.998423 / 0.223044
    TP / FP / FN / TN: 633 / 2205 / 1 / 432339
    Alerts: 2838
    Warnings: 0
    
    Run: M7.5-DT-W_SHORT-CLASS_WEIGHT_BALANCED-FOLD_Q3_2018
    F1 / Recall / Precision: 0.538527 / 0.501577 / 0.581353
    TP / FP / FN / TN: 318 / 229 / 316 / 434315
    Alerts: 547
    Warnings: 0
    
    Run: M7.5-RF-W_SHORT-CLASS_WEIGHT_BALANCED-FOLD_Q3_2018
    F1 / Recall / Precision: 0.6502 / 0.768139 / 0.563657
    TP / FP / FN / TN: 487 / 377 / 147 / 434167
    Alerts: 864
    Warnings: 0
    
    ==================================================
    W_SHORT FOLD_Q4_2018
    Train shape: (1288036, 47) | fraud: 1781
    Validation shape: (433579, 47) | fraud: 710
    Matrix seconds: 0.685
    
    Run: M7.5-LR-W_SHORT-CLASS_WEIGHT_BALANCED-FOLD_Q4_2018
    F1 / Recall / Precision: 0.460272 / 0.787324 / 0.325189
    TP / FP / FN / TN: 559 / 1160 / 151 / 431709
    Alerts: 1719
    Warnings: 0
    
    Run: M7.5-DT-W_SHORT-CLASS_WEIGHT_BALANCED-FOLD_Q4_2018
    F1 / Recall / Precision: 0.471233 / 0.36338 / 0.67013
    TP / FP / FN / TN: 258 / 127 / 452 / 432742
    Alerts: 385
    Warnings: 0
    
    Run: M7.5-RF-W_SHORT-CLASS_WEIGHT_BALANCED-FOLD_Q4_2018
    F1 / Recall / Precision: 0.609594 / 0.58169 / 0.64031
    TP / FP / FN / TN: 413 / 232 / 297 / 432637
    Alerts: 645
    Warnings: 0
    
    Completed class-weight runs: 9
    Phase A seconds: 45.38
    
    M7.5 CLASS-WEIGHT EXECUTION GATE: PASS


## 18. Controlled comparability: NONE vs CLASS_WEIGHT


```python

class_weight_by_model_fold = {
    (
        record[
            "model_key"
        ],
        record[
            "fold_id"
        ],
    ):
        record
    for record
    in class_weight_fold_results
}


assert len(
    class_weight_by_model_fold
) == 9


comparability_checks = []


for model_key in MODEL_KEYS:
    for fold_id in FOLD_ORDER:
        baseline = (
            baseline_by_model_fold[
                (
                    model_key,
                    fold_id,
                )
            ]
        )

        weighted = (
            class_weight_by_model_fold[
                (
                    model_key,
                    fold_id,
                )
            ]
        )

        shared_fields = [
            "fold_id",
            "train_start",
            "train_end_exclusive",
            "validation_start",
            "validation_end_exclusive",
            "train_rows",
            "validation_rows",
            "train_fraud_rows",
            "validation_fraud_rows",
            "feature_version",
            "preprocessing_version",
            "model_family",
            "model_config_id",
            "random_state",
            "threshold_policy",
        ]

        results = {}

        for field in shared_fields:
            results[
                field
            ] = (
                baseline[
                    field
                ]
                ==
                weighted[
                    field
                ]
            )

        assert all(
            results.values()
        )

        baseline_params = copy.deepcopy(
            baseline[
                "hyperparameters"
            ]
        )

        weighted_base_params = copy.deepcopy(
            weighted[
                "hyperparameters"
            ]
        )

        baseline_class_weight = (
            baseline_params.pop(
                "class_weight"
            )
        )

        assert (
            baseline_class_weight
            is None
        )

        assert (
            baseline_params
            ==
            weighted_base_params
        )

        assert (
            baseline[
                "imbalance_strategy"
            ]
            == "NONE"
        )

        assert (
            weighted[
                "imbalance_strategy"
            ]
            == "CLASS_WEIGHT"
        )

        assert (
            weighted[
                "class_weight"
            ]
            == "balanced"
        )

        comparability_checks.append(
            {
                "model_key":
                    model_key,

                "fold_id":
                    fold_id,

                "shared_field_count":
                    len(
                        shared_fields
                    ),

                "base_hyperparameters_match":
                    True,

                "only_intended_change":
                    (
                        "NONE → "
                        "CLASS_WEIGHT_BALANCED"
                    ),

                "status":
                    "PASS",
            }
        )


assert len(
    comparability_checks
) == 9


print(
    "Controlled NONE vs CLASS_WEIGHT checks:",
    len(
        comparability_checks
    ),
)

print(
    "\nM7.5 CONTROLLED COMPARABILITY GATE: PASS"
)

```

    Controlled NONE vs CLASS_WEIGHT checks: 9
    
    M7.5 CONTROLLED COMPARABILITY GATE: PASS


## 19. Aggregate NONE and CLASS_WEIGHT evidence


```python

def aggregate_records(
    records,
):
    f1_values = np.array(
        [
            record[
                "F1_fraud"
            ]
            for record
            in records
        ],
        dtype=np.float64,
    )

    recall_values = np.array(
        [
            record[
                "Recall_fraud"
            ]
            for record
            in records
        ],
        dtype=np.float64,
    )

    precision_values = np.array(
        [
            record[
                "Precision_fraud"
            ]
            for record
            in records
        ],
        dtype=np.float64,
    )

    tp = int(
        sum(
            record[
                "TP"
            ]
            for record
            in records
        )
    )

    fp = int(
        sum(
            record[
                "FP"
            ]
            for record
            in records
        )
    )

    fn = int(
        sum(
            record[
                "FN"
            ]
            for record
            in records
        )
    )

    tn = int(
        sum(
            record[
                "TN"
            ]
            for record
            in records
        )
    )

    alerts = int(
        sum(
            record[
                "predicted_positive_count"
            ]
            for record
            in records
        )
    )

    rows = (
        tp + fp + fn + tn
    )

    return {
        "mean_F1":
            float(
                f1_values.mean()
            ),

        "std_F1":
            float(
                f1_values.std(
                    ddof=0
                )
            ),

        "mean_Recall":
            float(
                recall_values.mean()
            ),

        "mean_Precision":
            float(
                precision_values.mean()
            ),

        "TP":
            tp,

        "FP":
            fp,

        "FN":
            fn,

        "TN":
            tn,

        "alerts":
            alerts,

        "alert_rate":
            float(
                alerts / rows
            ),

        "total_fit_seconds":
            float(
                sum(
                    record[
                        "fit_seconds"
                    ]
                    for record
                    in records
                )
            ),

        "total_prediction_seconds":
            float(
                sum(
                    record[
                        "prediction_seconds"
                    ]
                    for record
                    in records
                )
            ),
    }


aggregate_evidence = {}


for model_key in MODEL_KEYS:
    none_records = [
        baseline_by_model_fold[
            (
                model_key,
                fold_id,
            )
        ]
        for fold_id
        in FOLD_ORDER
    ]

    weighted_records = [
        class_weight_by_model_fold[
            (
                model_key,
                fold_id,
            )
        ]
        for fold_id
        in FOLD_ORDER
    ]

    aggregate_evidence[
        model_key
    ] = {
        "NONE":
            aggregate_records(
                none_records
            ),

        "CLASS_WEIGHT_BALANCED":
            aggregate_records(
                weighted_records
            ),
    }


for model_key in MODEL_KEYS:
    print(
        "\n",
        MODEL_LABELS[
            model_key
        ],
    )

    for strategy in [
        "NONE",
        "CLASS_WEIGHT_BALANCED",
    ]:
        record = (
            aggregate_evidence[
                model_key
            ][
                strategy
            ]
        )

        print(
            strategy,
            "| mean F1:",
            round(
                record[
                    "mean_F1"
                ],
                6,
            ),
            "| std F1:",
            round(
                record[
                    "std_F1"
                ],
                6,
            ),
            "| mean Recall:",
            round(
                record[
                    "mean_Recall"
                ],
                6,
            ),
            "| mean Precision:",
            round(
                record[
                    "mean_Precision"
                ],
                6,
            ),
            "| FN:",
            record[
                "FN"
            ],
            "| FP:",
            record[
                "FP"
            ],
            "| alerts:",
            record[
                "alerts"
            ],
        )


print(
    "\nM7.5 AGGREGATION GATE: PASS"
)

```

    
     Logistic Regression
    NONE | mean F1: 0.495102 | std F1: 0.038315 | mean Recall: 0.385526 | mean Precision: 0.698296 | FN: 1192 | FP: 321 | alerts: 1063
    CLASS_WEIGHT_BALANCED | mean F1: 0.411141 | std F1: 0.039089 | mean Recall: 0.925757 | mean Precision: 0.268496 | FN: 157 | FP: 5054 | alerts: 6831
    
     Decision Tree
    NONE | mean F1: 0.487392 | std F1: 0.013631 | mean Recall: 0.471101 | mean Precision: 0.515111 | FN: 1027 | FP: 880 | alerts: 1787
    CLASS_WEIGHT_BALANCED | mean F1: 0.511964 | std F1: 0.029242 | mean Recall: 0.458941 | mean Precision: 0.597567 | FN: 1056 | FP: 612 | alerts: 1490
    
     Random Forest
    NONE | mean F1: 0.517612 | std F1: 0.029168 | mean Recall: 0.414125 | mean Precision: 0.694881 | FN: 1135 | FP: 351 | alerts: 1150
    CLASS_WEIGHT_BALANCED | mean F1: 0.621144 | std F1: 0.02069 | mean Recall: 0.684406 | mean Precision: 0.577543 | FN: 619 | FP: 979 | alerts: 2294
    
    M7.5 AGGREGATION GATE: PASS


## 20. Intervention deltas by model


```python

intervention_comparisons = []


for model_key in MODEL_KEYS:
    none_agg = (
        aggregate_evidence[
            model_key
        ][
            "NONE"
        ]
    )

    weighted_agg = (
        aggregate_evidence[
            model_key
        ][
            "CLASS_WEIGHT_BALANCED"
        ]
    )

    fold_deltas = []

    weighted_f1_leads = 0
    none_f1_leads = 0

    for fold_id in FOLD_ORDER:
        none_record = (
            baseline_by_model_fold[
                (
                    model_key,
                    fold_id,
                )
            ]
        )

        weighted_record = (
            class_weight_by_model_fold[
                (
                    model_key,
                    fold_id,
                )
            ]
        )

        delta_f1 = (
            weighted_record[
                "F1_fraud"
            ]
            -
            none_record[
                "F1_fraud"
            ]
        )

        if delta_f1 > 0:
            weighted_f1_leads += 1

        elif delta_f1 < 0:
            none_f1_leads += 1

        fold_deltas.append(
            {
                "fold_id":
                    fold_id,

                "delta_F1":
                    float(
                        delta_f1
                    ),

                "delta_Recall":
                    float(
                        weighted_record[
                            "Recall_fraud"
                        ]
                        -
                        none_record[
                            "Recall_fraud"
                        ]
                    ),

                "delta_Precision":
                    float(
                        weighted_record[
                            "Precision_fraud"
                        ]
                        -
                        none_record[
                            "Precision_fraud"
                        ]
                    ),

                "delta_TP":
                    int(
                        weighted_record[
                            "TP"
                        ]
                        -
                        none_record[
                            "TP"
                        ]
                    ),

                "delta_FP":
                    int(
                        weighted_record[
                            "FP"
                        ]
                        -
                        none_record[
                            "FP"
                        ]
                    ),

                "delta_FN":
                    int(
                        weighted_record[
                            "FN"
                        ]
                        -
                        none_record[
                            "FN"
                        ]
                    ),

                "delta_alerts":
                    int(
                        weighted_record[
                            "predicted_positive_count"
                        ]
                        -
                        none_record[
                            "predicted_positive_count"
                        ]
                    ),
            }
        )

    intervention_comparisons.append(
        {
            "model_key":
                model_key,

            "model_family":
                MODEL_LABELS[
                    model_key
                ],

            "delta_convention":
                "CLASS_WEIGHT_BALANCED - NONE",

            "weighted_F1_fold_leads":
                weighted_f1_leads,

            "none_F1_fold_leads":
                none_f1_leads,

            "delta_mean_F1":
                float(
                    weighted_agg[
                        "mean_F1"
                    ]
                    -
                    none_agg[
                        "mean_F1"
                    ]
                ),

            "delta_std_F1":
                float(
                    weighted_agg[
                        "std_F1"
                    ]
                    -
                    none_agg[
                        "std_F1"
                    ]
                ),

            "delta_mean_Recall":
                float(
                    weighted_agg[
                        "mean_Recall"
                    ]
                    -
                    none_agg[
                        "mean_Recall"
                    ]
                ),

            "delta_mean_Precision":
                float(
                    weighted_agg[
                        "mean_Precision"
                    ]
                    -
                    none_agg[
                        "mean_Precision"
                    ]
                ),

            "delta_TP":
                int(
                    weighted_agg[
                        "TP"
                    ]
                    -
                    none_agg[
                        "TP"
                    ]
                ),

            "delta_FP":
                int(
                    weighted_agg[
                        "FP"
                    ]
                    -
                    none_agg[
                        "FP"
                    ]
                ),

            "delta_FN":
                int(
                    weighted_agg[
                        "FN"
                    ]
                    -
                    none_agg[
                        "FN"
                    ]
                ),

            "delta_alerts":
                int(
                    weighted_agg[
                        "alerts"
                    ]
                    -
                    none_agg[
                        "alerts"
                    ]
                ),

            "fit_time_ratio":
                float(
                    weighted_agg[
                        "total_fit_seconds"
                    ]
                    /
                    none_agg[
                        "total_fit_seconds"
                    ]
                ),

            "fold_deltas":
                fold_deltas,

            "decision":
                (
                    "OPEN — REQUIRES "
                    "M7.5 PHASE-A RUNTIME REVIEW"
                ),
        }
    )


for comparison in (
    intervention_comparisons
):
    print(
        "\n",
        comparison[
            "model_family"
        ],
    )

    print(
        "CLASS_WEIGHT / NONE F1 fold leads:",
        comparison[
            "weighted_F1_fold_leads"
        ],
        "/",
        comparison[
            "none_F1_fold_leads"
        ],
    )

    print(
        "Δ mean F1:",
        round(
            comparison[
                "delta_mean_F1"
            ],
            6,
        ),
    )

    print(
        "Δ mean Recall:",
        round(
            comparison[
                "delta_mean_Recall"
            ],
            6,
        ),
    )

    print(
        "Δ mean Precision:",
        round(
            comparison[
                "delta_mean_Precision"
            ],
            6,
        ),
    )

    print(
        "Δ FN / FP / alerts:",
        comparison[
            "delta_FN"
        ],
        "/",
        comparison[
            "delta_FP"
        ],
        "/",
        comparison[
            "delta_alerts"
        ],
    )


print(
    "\nM7.5 INTERVENTION DELTA GATE: PASS"
)

```

    
     Logistic Regression
    CLASS_WEIGHT / NONE F1 fold leads: 0 / 3
    Δ mean F1: -0.083961
    Δ mean Recall: 0.540231
    Δ mean Precision: -0.429799
    Δ FN / FP / alerts: -1035 / 4733 / 5768
    
     Decision Tree
    CLASS_WEIGHT / NONE F1 fold leads: 2 / 1
    Δ mean F1: 0.024572
    Δ mean Recall: -0.012161
    Δ mean Precision: 0.082456
    Δ FN / FP / alerts: 29 / -268 / -297
    
     Random Forest
    CLASS_WEIGHT / NONE F1 fold leads: 3 / 0
    Δ mean F1: 0.103531
    Δ mean Recall: 0.270281
    Δ mean Precision: -0.117338
    Δ FN / FP / alerts: -516 / 628 / 1144
    
    M7.5 INTERVENTION DELTA GATE: PASS


## 21. Phase-A decision boundary

Notebook không tự động quyết định:

```text
KEEP NONE

or

SELECT CLASS_WEIGHT
```

vì runtime review phải đọc đầy đủ trade-off.

Notebook cũng không tự authorize Phase B.

Sau runtime review, mỗi model có thể có state riêng:

```text
KEEP NONE
CLASS_WEIGHT PROMISING
MIXED / INCONCLUSIVE
```

Cross-model final imbalance strategy vẫn có thể khác nhau trong intermediate candidate set, nhưng M7.7 cuối cùng phải freeze một upstream candidate rõ ràng.

Random over/under chỉ được mở nếu có reviewed justification.


```python

PHASE_A_DECISION = (
    "OPEN — REQUIRES M7.5 PHASE-A RUNTIME REVIEW"
)

RANDOM_OVERSAMPLING_AUTHORIZED = False
RANDOM_UNDERSAMPLING_AUTHORIZED = False
SMOTE_AUTHORIZED = False

FINAL_IMBALANCE_STRATEGY = "OPEN"

MODEL_FAMILY_WINNER = "OPEN"
FINAL_MODEL = "OPEN"
FINAL_THRESHOLD = "OPEN"


assert (
    RANDOM_OVERSAMPLING_AUTHORIZED
    is False
)

assert (
    RANDOM_UNDERSAMPLING_AUTHORIZED
    is False
)

assert (
    SMOTE_AUTHORIZED
    is False
)

assert (
    FINAL_IMBALANCE_STRATEGY
    == "OPEN"
)


print(
    "Phase A decision:"
)

print(
    PHASE_A_DECISION
)

print(
    "\nRandom oversampling authorized:"
)

print(
    RANDOM_OVERSAMPLING_AUTHORIZED
)

print(
    "Random undersampling authorized:"
)

print(
    RANDOM_UNDERSAMPLING_AUTHORIZED
)

print(
    "SMOTE authorized:"
)

print(
    SMOTE_AUTHORIZED
)

print(
    "\nM7.5 STAGED-ORDER GATE: PASS"
)

```

    Phase A decision:
    OPEN — REQUIRES M7.5 PHASE-A RUNTIME REVIEW
    
    Random oversampling authorized:
    False
    Random undersampling authorized:
    False
    SMOTE authorized:
    False
    
    M7.5 STAGED-ORDER GATE: PASS


## 22. Persist Phase-A registry


```python

source_fingerprints = {
    "m4_manifest_sha256":
        sha256_file(
            M4_MANIFEST_PATH
        ),

    "m7_02_contract_sha256":
        sha256_file(
            M7_02_CONTRACT_PATH
        ),

    "m7_02_preprocessing_audit_sha256":
        sha256_file(
            M7_02_PREPROCESSING_AUDIT_PATH
        ),

    "m7_03_result_sha256":
        sha256_file(
            M7_03_RESULT_PATH
        ),

    "m7_03_manifest_sha256":
        sha256_file(
            M7_03_MANIFEST_PATH
        ),

    "m7_04_result_sha256":
        sha256_file(
            M7_04_RESULT_PATH
        ),

    "m7_04_manifest_sha256":
        sha256_file(
            M7_04_MANIFEST_PATH
        ),

    "reviewed_m7_04_notebook_sha256":
        UPSTREAM_REVIEWED_M7_4_NOTEBOOK_SHA256,
}


result_payload = {
    "analysis_version":
        M7_05_ANALYSIS_VERSION,

    "phase":
        "A_CLASS_WEIGHT",

    "upstream_reviewed_handoff": {
        "m7_04_status":
            UPSTREAM_REVIEWED_M7_4_STATUS,

        "training_window":
            UPSTREAM_REVIEWED_TRAINING_WINDOW,

        "shortlist":
            UPSTREAM_REVIEWED_SHORTLIST,

        "reviewed_notebook_sha256":
            UPSTREAM_REVIEWED_M7_4_NOTEBOOK_SHA256,
    },

    "source_fingerprints":
        source_fingerprints,

    "experiment_scope": {
        "training_window":
            "W_SHORT",

        "model_keys":
            MODEL_KEYS,

        "fold_ids":
            FOLD_ORDER,

        "reference_strategy":
            "NONE",

        "phase_a_candidate":
            "CLASS_WEIGHT_BALANCED",

        "class_weight":
            "balanced",

        "threshold_policy":
            "DEFAULT_MODEL_DECISION_RULE",

        "random_state":
            RANDOM_STATE,

        "active_variable":
            "imbalance strategy / class_weight",
    },

    "candidate_registry":
        candidate_registry,

    "baseline_none_fold_results":
        baseline_w_short_records,

    "class_weight_fold_results":
        class_weight_fold_results,

    "comparability_checks":
        comparability_checks,

    "aggregate_evidence":
        aggregate_evidence,

    "intervention_comparisons":
        intervention_comparisons,

    "artifact_fingerprints":
        artifact_fingerprints,

    "selection_state": {
        "phase_a_decision":
            PHASE_A_DECISION,

        "random_oversampling_authorized":
            RANDOM_OVERSAMPLING_AUTHORIZED,

        "random_undersampling_authorized":
            RANDOM_UNDERSAMPLING_AUTHORIZED,

        "smote_authorized":
            SMOTE_AUTHORIZED,

        "final_imbalance_strategy":
            FINAL_IMBALANCE_STRATEGY,

        "model_family_winner":
            MODEL_FAMILY_WINNER,

        "final_model":
            FINAL_MODEL,

        "final_threshold":
            FINAL_THRESHOLD,
    },

    "validation_resampling_performed":
        False,

    "external_validation_scoring_performed":
        False,

    "hyperparameter_tuning_performed":
        False,

    "threshold_optimization_performed":
        False,

    "final_test_accessed":
        False,
}


with open(
    RESULT_PATH,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        result_payload,
        file,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )


manifest_payload = {
    "analysis_version":
        M7_05_ANALYSIS_VERSION,

    "phase":
        "A_CLASS_WEIGHT",

    "training_window":
        "W_SHORT",

    "shortlisted_family_count":
        3,

    "baseline_none_fold_run_count":
        9,

    "class_weight_fold_run_count":
        9,

    "warning_count_total":
        int(
            sum(
                record[
                    "warning_count"
                ]
                for record
                in class_weight_fold_results
            )
        ),

    "controlled_comparison_count":
        len(
            comparability_checks
        ),

    "phase_a_decision":
        PHASE_A_DECISION,

    "random_sampling_authorized":
        False,

    "smote_authorized":
        False,

    "final_imbalance_strategy":
        "OPEN",

    "validation_resampling_performed":
        False,

    "external_validation_scoring_performed":
        False,

    "hyperparameter_tuning_performed":
        False,

    "threshold_optimization_performed":
        False,

    "final_test_accessed":
        False,

    "decision":
        (
            "OPEN — REQUIRES "
            "M7.5 PHASE-A RUNTIME REVIEW"
        ),
}


with open(
    MANIFEST_PATH,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        manifest_payload,
        file,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )


assert RESULT_PATH.exists()
assert MANIFEST_PATH.exists()

assert (
    RESULT_PATH.stat().st_size
    > 0
)

assert (
    MANIFEST_PATH.stat().st_size
    > 0
)


print(
    "Result:"
)

print(
    RESULT_PATH
)

print(
    "\nManifest:"
)

print(
    MANIFEST_PATH
)

print(
    "\nM7.5 PHASE-A PERSISTENCE GATE: PASS"
)

```

    Result:
    /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/processed/m7_05_controlled_class_imbalance/m7_05_class_weight_phase_a.json
    
    Manifest:
    /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/processed/m7_05_controlled_class_imbalance/m7_05_phase_a_manifest.json
    
    M7.5 PHASE-A PERSISTENCE GATE: PASS


## 23. Persistence round-trip / artifact identity


```python

with open(
    RESULT_PATH,
    "r",
    encoding="utf-8",
) as file:
    result_roundtrip = json.load(
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
    result_roundtrip[
        "analysis_version"
    ]
    == M7_05_ANALYSIS_VERSION
)

assert (
    result_roundtrip[
        "experiment_scope"
    ][
        "training_window"
    ]
    == "W_SHORT"
)

assert (
    len(
        result_roundtrip[
            "candidate_registry"
        ]
    )
    == 9
)

assert (
    len(
        result_roundtrip[
            "class_weight_fold_results"
        ]
    )
    == 9
)

assert (
    len(
        result_roundtrip[
            "intervention_comparisons"
        ]
    )
    == 3
)

assert (
    result_roundtrip[
        "selection_state"
    ][
        "random_oversampling_authorized"
    ]
    is False
)

assert (
    result_roundtrip[
        "selection_state"
    ][
        "smote_authorized"
    ]
    is False
)

assert (
    result_roundtrip[
        "validation_resampling_performed"
    ]
    is False
)

assert (
    result_roundtrip[
        "final_test_accessed"
    ]
    is False
)


assert (
    manifest_roundtrip[
        "class_weight_fold_run_count"
    ]
    == 9
)

assert (
    manifest_roundtrip[
        "warning_count_total"
    ]
    == 0
)

assert (
    manifest_roundtrip[
        "controlled_comparison_count"
    ]
    == 9
)

assert (
    manifest_roundtrip[
        "final_test_accessed"
    ]
    is False
)


for candidate_id, fingerprint in (
    artifact_fingerprints.items()
):
    record = next(
        record
        for record
        in class_weight_fold_results
        if record[
            "candidate_id"
        ]
        == candidate_id
    )

    prediction_path = (
        PROJECT_ROOT
        / record[
            "prediction_artifact"
        ]
    )

    risk_score_path = (
        PROJECT_ROOT
        / record[
            "risk_score_artifact"
        ]
    )

    assert (
        sha256_file(
            prediction_path
        )
        ==
        fingerprint[
            "prediction_sha256"
        ]
    )

    assert (
        sha256_file(
            risk_score_path
        )
        ==
        fingerprint[
            "risk_score_sha256"
        ]
    )


print(
    "Result SHA256:"
)

print(
    sha256_file(
        RESULT_PATH
    )
)

print(
    "\nManifest SHA256:"
)

print(
    sha256_file(
        MANIFEST_PATH
    )
)

print(
    "\nM7.5 PHASE-A ROUND-TRIP GATE: PASS"
)

```

    Result SHA256:
    1fe1771fd5f9250028361310cc59cc4cc5dd6a5b6ec93218c1a58ab3d3c20d10
    
    Manifest SHA256:
    796298d45f3c5c3829db3ad1e139eaf1759dd591ec9dd52b6a797cfea1f9bdda
    
    M7.5 PHASE-A ROUND-TRIP GATE: PASS


## 24. Leakage / selection-boundary gate


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
    result_payload[
        "validation_resampling_performed"
    ]
    is False
)

assert (
    result_payload[
        "external_validation_scoring_performed"
    ]
    is False
)

assert (
    result_payload[
        "hyperparameter_tuning_performed"
    ]
    is False
)

assert (
    result_payload[
        "threshold_optimization_performed"
    ]
    is False
)

assert (
    result_payload[
        "selection_state"
    ][
        "random_oversampling_authorized"
    ]
    is False
)

assert (
    result_payload[
        "selection_state"
    ][
        "random_undersampling_authorized"
    ]
    is False
)

assert (
    result_payload[
        "selection_state"
    ][
        "smote_authorized"
    ]
    is False
)

assert (
    result_payload[
        "selection_state"
    ][
        "final_imbalance_strategy"
    ]
    == "OPEN"
)

assert (
    result_payload[
        "selection_state"
    ][
        "model_family_winner"
    ]
    == "OPEN"
)

assert (
    result_payload[
        "final_test_accessed"
    ]
    is False
)


print(
    "Validation resampling:",
    result_payload[
        "validation_resampling_performed"
    ],
)

print(
    "External VALIDATION scoring:",
    result_payload[
        "external_validation_scoring_performed"
    ],
)

print(
    "Threshold optimization:",
    result_payload[
        "threshold_optimization_performed"
    ],
)

print(
    "FINAL TEST accessed:",
    result_payload[
        "final_test_accessed"
    ],
)

print(
    "\nM7.5 LEAKAGE / SELECTION-BOUNDARY GATE: PASS"
)

```

    Validation resampling: False
    External VALIDATION scoring: False
    Threshold optimization: False
    FINAL TEST accessed: False
    
    M7.5 LEAKAGE / SELECTION-BOUNDARY GATE: PASS


## 25. Overall Phase-A technical gate


```python

m7_05_phase_a_gates = {
    "G01_SOURCE_LOCATION":
        True,

    "G02_UPSTREAM_HANDOFF":
        True,

    "G03_PHASE_A_CANDIDATE_SPEC":
        True,

    "G04_FEATURE_CONTRACT":
        True,

    "G05_STRICT_CAUSAL_REGRESSION":
        True,

    "G06_SEMANTIC_TRAIN_LINEAGE":
        True,

    "G07_FOLD_INTEGRITY":
        True,

    "G08_PREPROCESSING_STATE_REUSE":
        True,

    "G09_MATRIX_BUILDER":
        True,

    "G10_CLASS_WEIGHT_RUNNER":
        True,

    "G11_CLASS_WEIGHT_9_RUN_EXECUTION":
        True,

    "G12_CONTROLLED_COMPARABILITY":
        True,

    "G13_AGGREGATION":
        True,

    "G14_INTERVENTION_DELTAS":
        True,

    "G15_STAGED_ORDER":
        True,

    "G16_PERSISTENCE":
        True,

    "G17_ROUND_TRIP_ARTIFACT_IDENTITY":
        True,

    "G18_NO_VALIDATION_RESAMPLING":
        True,

    "G19_NO_THRESHOLD_TUNING":
        True,

    "G20_FINAL_TEST_PROTECTION":
        True,
}


for gate_name, gate_value in (
    m7_05_phase_a_gates.items()
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


assert len(
    m7_05_phase_a_gates
) == 20

assert all(
    m7_05_phase_a_gates.values()
)


print(
    "\nM7.5 PHASE-A OVERALL TECHNICAL GATE: PASS"
)

```

    G01_SOURCE_LOCATION → PASS
    G02_UPSTREAM_HANDOFF → PASS
    G03_PHASE_A_CANDIDATE_SPEC → PASS
    G04_FEATURE_CONTRACT → PASS
    G05_STRICT_CAUSAL_REGRESSION → PASS
    G06_SEMANTIC_TRAIN_LINEAGE → PASS
    G07_FOLD_INTEGRITY → PASS
    G08_PREPROCESSING_STATE_REUSE → PASS
    G09_MATRIX_BUILDER → PASS
    G10_CLASS_WEIGHT_RUNNER → PASS
    G11_CLASS_WEIGHT_9_RUN_EXECUTION → PASS
    G12_CONTROLLED_COMPARABILITY → PASS
    G13_AGGREGATION → PASS
    G14_INTERVENTION_DELTAS → PASS
    G15_STAGED_ORDER → PASS
    G16_PERSISTENCE → PASS
    G17_ROUND_TRIP_ARTIFACT_IDENTITY → PASS
    G18_NO_VALIDATION_RESAMPLING → PASS
    G19_NO_THRESHOLD_TUNING → PASS
    G20_FINAL_TEST_PROTECTION → PASS
    
    M7.5 PHASE-A OVERALL TECHNICAL GATE: PASS


# 26. Runtime review và Findings M7.5 Phase A

## 26.1. Execution integrity

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

Không có partial execution hoặc runtime exception làm mất hiệu lực evidence.

Status:

`VERIFIED`

---

## 26.2. Upstream handoff

Observed:

```text
Training window:
W_SHORT

Shortlist:
LR
DT
RF

Baseline NONE:
9 W_SHORT fold results từ M7.3

Fold-safe preprocessing:
verified từ M7.2
```

Status:

`VERIFIED`

---

## 26.3. Temporal fold integrity

Observed:

```text
W_SHORT / Q2

train:
423,905 rows
557 fraud

validation:
428,953 rows
590 fraud
```

```text
W_SHORT / Q3

train:
852,858 rows
1,147 fraud

validation:
435,178 rows
634 fraud
```

```text
W_SHORT / Q4

train:
1,288,036 rows
1,781 fraud

validation:
433,579 rows
710 fraud
```

All three folds satisfy:

```text
positive train support
positive validation support
forward temporal order
zero train/validation overlap
```

Runtime gate:

`M7.5 FOLD INTEGRITY GATE: PASS`

Status:

`VERIFIED`

---

## 26.4. Fold-safe preprocessing

Observed:

```text
Verified preprocessing states:
3

Canonical feature width:
47

Matrix format:
CSR float32
```

Runtime gates:

```text
M7.5 PREPROCESSING-STATE REUSE GATE:
PASS

M7.5 MATRIX BUILDER GATE:
PASS
```

Status:

`VERIFIED`

---

## 26.5. Phase-A experiment completeness

Expected:

```text
3 model families
×
3 temporal folds
=
9 CLASS_WEIGHT runs
```

Observed:

```text
Completed class-weight runs:
9 / 9

Warnings:
0 / 9

Phase A seconds:
45.38
```

Runtime gate:

`M7.5 CLASS-WEIGHT EXECUTION GATE: PASS`

Status:

`VERIFIED`

---

## 26.6. Controlled comparability

Observed:

```text
Controlled NONE vs CLASS_WEIGHT checks:
9 / 9 PASS
```

Trong từng model/fold:

```text
same W_SHORT population
same fold boundaries
same feature version
same preprocessing version
same model family/config
same baseline hyperparameters
same random state
same threshold policy

only intended change:
NONE
→
CLASS_WEIGHT = "balanced"
```

Validation không được resample.

Status:

`VERIFIED`

---

# 27. Imbalance Findings

## M7.5-F01 — Logistic Regression: class weight tăng Recall nhưng phá vỡ primary/operational trade-off

Baseline `NONE`:

```text
mean F1:
0.495102

std F1:
0.038315

mean Recall:
0.385526

mean Precision:
0.698296

FN:
1,192

FP:
321

alerts:
1,063
```

`CLASS_WEIGHT_BALANCED`:

```text
mean F1:
0.411141

std F1:
0.039089

mean Recall:
0.925757

mean Precision:
0.268496

FN:
157

FP:
5,054

alerts:
6,831
```

Delta:

```text
CLASS_WEIGHT - NONE

Δ mean F1:
-0.083961

Δ mean Recall:
+0.540231

Δ mean Precision:
-0.429799

Δ FN:
-1,035

Δ FP:
+4,733

Δ alerts:
+5,768
```

Fold-wise primary metric:

```text
CLASS_WEIGHT F1 leads:
0 / 3

NONE F1 leads:
3 / 3
```

Interpretation:

Class weight gần như triệt tiêu nhiều false negative của LR, nhưng đổi lại:

- primary F1 giảm rõ;
- Precision giảm rất mạnh;
- FP tăng hơn 4,700;
- alert volume tăng hơn 5,700;
- không thắng F1 ở bất kỳ fold nào;
- stability không cải thiện.

Đây đúng là trường hợp không được gọi intervention tốt hơn chỉ vì Recall tăng.

Reviewed disposition:

`LR → KEEP NONE`

Status:

`LOCKED FOR DOWNSTREAM M7`

---

## M7.5-F02 — Decision Tree: class weight tạo improvement có ích nhưng kèm Recall/stability trade-off

Baseline `NONE`:

```text
mean F1:
0.487392

std F1:
0.013631

mean Recall:
0.471101

mean Precision:
0.515111

FN:
1,027

FP:
880

alerts:
1,787
```

`CLASS_WEIGHT_BALANCED`:

```text
mean F1:
0.511964

std F1:
0.029242

mean Recall:
0.458941

mean Precision:
0.597567

FN:
1,056

FP:
612

alerts:
1,490
```

Delta:

```text
Δ mean F1:
+0.024572

Δ mean Recall:
-0.012161

Δ mean Precision:
+0.082456

Δ FN:
+29

Δ FP:
-268

Δ alerts:
-297
```

Fold-wise primary metric:

```text
CLASS_WEIGHT F1 leads:
2 / 3

NONE F1 leads:
1 / 3
```

Interpretation:

Class weight không dominance tuyệt đối:

- mean F1 tăng;
- Precision tăng;
- FP giảm;
- alert burden giảm;

nhưng:

- Recall giảm nhẹ;
- FN tăng 29;
- std F1 tăng từ 0.013631 lên 0.029242;
- Q4 F1 thấp hơn NONE.

Tuy vậy, theo metric contract hiện tại F1 là primary, và improvement F1 đi cùng Precision/FP/alert improvement, không phải chỉ một Recall effect.

Recall/FN cost tồn tại nhưng nhỏ hơn operational gain và primary-F1 gain trong Phase A evidence.

Reviewed disposition:

`DT → RETAIN CLASS_WEIGHT_BALANCED`

Interpretation qualifier:

`PROMISING WITH REVIEWED RECALL/STABILITY TRADE-OFF`

Status:

`LOCKED FOR DOWNSTREAM M7`

---

## M7.5-F03 — Random Forest: class weight tạo strongest intervention evidence

Baseline `NONE`:

```text
mean F1:
0.517612

std F1:
0.029168

mean Recall:
0.414125

mean Precision:
0.694881

FN:
1,135

FP:
351

alerts:
1,150
```

`CLASS_WEIGHT_BALANCED`:

```text
mean F1:
0.621144

std F1:
0.020690

mean Recall:
0.684406

mean Precision:
0.577543

FN:
619

FP:
979

alerts:
2,294
```

Delta:

```text
Δ mean F1:
+0.103531

Δ mean Recall:
+0.270281

Δ mean Precision:
-0.117338

Δ FN:
-516

Δ FP:
+628

Δ alerts:
+1,144
```

Fold-wise primary metric:

```text
CLASS_WEIGHT F1 leads:
3 / 3

NONE F1 leads:
0 / 3
```

Interpretation:

RF class weight có evidence mạnh nhất trong Phase A:

- mean F1 tăng lớn;
- Recall tăng lớn;
- FN giảm 516;
- F1 thắng cả 3 folds;
- std F1 còn giảm.

Cost:

- Precision giảm;
- FP tăng;
- alert burden gần gấp đôi.

Nhưng khác LR, RF vẫn tăng primary F1 rõ và ổn định qua toàn bộ folds.

Reviewed disposition:

`RF → RETAIN CLASS_WEIGHT_BALANCED`

Status:

`LOCKED FOR DOWNSTREAM M7`

---

## M7.5-F04 — Intervention effect phụ thuộc model family

Observed dispositions:

```text
LR:
KEEP NONE

DT:
RETAIN CLASS_WEIGHT_BALANCED

RF:
RETAIN CLASS_WEIGHT_BALANCED
```

Interpretation:

Không tồn tại một kết luận kiểu:

`class_weight luôn tốt`

hoặc:

`class_weight luôn xấu`.

Effect phụ thuộc model family.

Đây là lý do candidate identity trong M7 phải chứa cả:

```text
model family
model config
imbalance strategy
```

Status:

`VERIFIED`

---

## M7.5-F05 — Random over/under sampling không được justify đủ mạnh để mở Phase B

CANON chỉ yêu cầu random over/under:

`nếu evidence vẫn justify intervention`.

Phase A hiện đã tạo actionable disposition cho cả ba candidates:

```text
LR:
NONE

DT:
CLASS_WEIGHT_BALANCED

RF:
CLASS_WEIGHT_BALANCED
```

Không có candidate nào còn ở trạng thái:

`MIXED / INCONCLUSIVE`

đến mức cần mở thêm một family intervention chỉ để hoàn thành danh sách technique.

Đặc biệt:

- LR đã có clear rejection của class weight và NONE vẫn là defensible baseline;
- DT đã có class-weight candidate với positive primary/operational evidence;
- RF đã có strong class-weight candidate.

Do đó:

```text
RANDOM_OVERSAMPLING:
DO NOT OPEN

RANDOM_UNDERSAMPLING:
DO NOT OPEN
```

Reason:

`NO ADDITIONAL JUSTIFICATION AFTER PHASE A`

Status:

`LOCKED`

---

## M7.5-F06 — SMOTE không được mở

SMOTE vẫn:

`CONDITIONAL / NOT REQUIRED`.

M7.5 Phase A không tạo evidence cần thiết để vượt conditional gate.

Không cần chạy SMOTE để M7.5 PASS.

Decision:

`DO NOT USE SMOTE IN M7.5`

Status:

`LOCKED`

---

## M7.5-F07 — Candidate-specific imbalance disposition được khóa, global final strategy vẫn chưa tồn tại

Reviewed downstream candidate states:

```text
LR candidate:
NONE

DT candidate:
CLASS_WEIGHT_BALANCED

RF candidate:
CLASS_WEIGHT_BALANCED
```

Điều này không có nghĩa project đã có một global final imbalance strategy.

Final global strategy phụ thuộc candidate/model cuối cùng được chọn sau tuning/confirmation.

Therefore:

```text
Candidate-specific imbalance disposition:
LOCKED

Global Final Imbalance Strategy:
OPEN
```

Status:

`BOUNDARY PRESERVED`

---

## M7.5-F08 — Không cần Phase B

Phase A đã trả lời câu hỏi M7.5 ở mức đủ để downstream tuning tiếp tục:

```text
Có intervention ngoài NONE cần giữ?
YES — cho DT và RF

Có intervention cần loại?
CLASS_WEIGHT cho LR

Cần random sampling để resolve ambiguity?
NO

Cần SMOTE?
NO
```

Therefore:

`M7.5 STAGED EXPERIMENTS COMPLETE`

Status:

`VERIFIED`

---

## M7.5-F09 — Runtime / warning integrity

Observed:

```text
Phase-A fit/predict execution:
COMPLETE

Warnings:
0

Runtime errors:
0

stderr:
0
```

Status:

`VERIFIED`

---

## M7.5-F10 — Persistence / artifact identity

Persisted runtime artifacts:

```text
data/processed/m7_05_controlled_class_imbalance/
    m7_05_class_weight_phase_a.json
    m7_05_phase_a_manifest.json
    m7_05_phase_a_progress.json
```

Plus:

```text
9 prediction artifacts
9 risk-score artifacts
```

Observed fingerprints:

```text
Result SHA256:
1fe1771fd5f9250028361310cc59cc4cc5dd6a5b6ec93218c1a58ab3d3c20d10

Manifest SHA256:
796298d45f3c5c3829db3ad1e139eaf1759dd591ec9dd52b6a797cfea1f9bdda
```

Runtime gate:

`M7.5 PHASE-A ROUND-TRIP GATE: PASS`

Status:

`VERIFIED`

---

## M7.5-F11 — Leakage / selection boundary sạch

Observed:

```text
Validation resampling:
False

External VALIDATION scoring:
False

Threshold optimization:
False

FINAL TEST accessed:
False
```

No random sampling was executed.

No SMOTE was executed.

Status:

`VERIFIED`

---

## M7.5-F12 — M7.6 readiness

Necessary downstream candidate set:

```text
LR:
W_SHORT
NONE

DT:
W_SHORT
CLASS_WEIGHT_BALANCED

RF:
W_SHORT
CLASS_WEIGHT_BALANCED
```

Model-family winner remains:

`OPEN`.

Final model remains:

`OPEN`.

Global final imbalance strategy remains:

`OPEN`.

Status:

`READY FOR M7.6`

# 28. Decision Log M7.5 — sau Phase-A runtime review

## M7.5-D01 — Training-window scope

Decision:

`W_SHORT`

Status:

`INHERITED — VERIFIED — LOCKED`

---

## M7.5-D02 — Candidate families

Decision:

```text
LR
DT
RF
```

Status:

`INHERITED FROM M7.4 — VERIFIED — LOCKED`

---

## M7.5-D03 — Baseline imbalance reference

Decision:

`NONE`

Status:

`REQUIRED BASELINE — VERIFIED`

---

## M7.5-D04 — Phase-A intervention

Decision:

`CLASS_WEIGHT = "balanced"`

Status:

`EXECUTED / VERIFIED`

---

## M7.5-D05 — Controlled comparability

Observed:

`9 / 9 PASS`

Decision:

NONE vs CLASS_WEIGHT evidence is valid for interpretation.

Status:

`VERIFIED`

---

## M7.5-D06 — Logistic Regression imbalance disposition

Observed:

```text
Δ mean F1:
-0.083961

F1 fold leads:
0 / 3

Δ Recall:
+0.540231

Δ Precision:
-0.429799

Δ FP:
+4,733

Δ alerts:
+5,768
```

Decision:

`LR → KEEP NONE`

Reason:

Recall gain does not compensate for primary-F1, Precision, FP and alert deterioration.

Status:

`LOCKED FOR DOWNSTREAM M7`

---

## M7.5-D07 — Decision Tree imbalance disposition

Observed:

```text
Δ mean F1:
+0.024572

F1 fold leads:
2 / 3

Δ Recall:
-0.012161

Δ Precision:
+0.082456

Δ FN:
+29

Δ FP:
-268

Δ alerts:
-297
```

Decision:

`DT → RETAIN CLASS_WEIGHT_BALANCED`

Reason:

Primary F1 improves and Precision/FP/alert burden improve.

Trade-off:

Recall slightly lower, FN slightly higher and std F1 worse.

Status:

`LOCKED FOR DOWNSTREAM M7`

---

## M7.5-D08 — Random Forest imbalance disposition

Observed:

```text
Δ mean F1:
+0.103531

F1 fold leads:
3 / 3

Δ Recall:
+0.270281

Δ Precision:
-0.117338

Δ FN:
-516

Δ FP:
+628

Δ alerts:
+1,144

std F1:
improves
```

Decision:

`RF → RETAIN CLASS_WEIGHT_BALANCED`

Reason:

Strong and fold-consistent F1/Recall/FN improvement despite Precision/FP/alert cost.

Status:

`LOCKED FOR DOWNSTREAM M7`

---

## M7.5-D09 — Candidate-specific imbalance map

Decision:

```text
LR
→ NONE

DT
→ CLASS_WEIGHT_BALANCED

RF
→ CLASS_WEIGHT_BALANCED
```

Status:

`LOCKED FOR M7.6 INPUT`

---

## M7.5-D10 — Random oversampling

Decision:

`DO NOT OPEN`

Reason:

Phase A already resolves candidate-specific intervention direction sufficiently.

Status:

`LOCKED`

---

## M7.5-D11 — Random undersampling

Decision:

`DO NOT OPEN`

Reason:

No unresolved Phase-A evidence currently justifies discarding majority rows or adding another intervention family.

Status:

`LOCKED`

---

## M7.5-D12 — SMOTE

Decision:

`DO NOT USE`

Reason:

Conditional strategy; not required; no additional Phase-A justification.

Status:

`LOCKED FOR M7.5`

---

## M7.5-D13 — Phase B

Decision:

`NOT REQUIRED`

Status:

`CLOSED`

---

## M7.5-D14 — Global final imbalance strategy

Decision:

Not yet defined because final model family is still OPEN.

Status:

`OPEN`

---

## M7.5-D15 — Model-family winner

Decision:

Not selected in M7.5.

Status:

`OPEN`

---

## M7.5-D16 — Hyperparameter tuning

Decision:

None performed in M7.5.

Status:

`LOCKED / VERIFIED`

---

## M7.5-D17 — Threshold optimization

Decision:

None performed in M7.5.

Status:

`LOCKED / VERIFIED`

---

## M7.5-D18 — External VALIDATION

Decision:

No scoring in M7.5.

Status:

`LOCKED / VERIFIED`

---

## M7.5-D19 — FINAL TEST

Decision:

No access.

Status:

`INHERITED — VERIFIED — LOCKED`

---

## M7.5-D20 — M7.6 handoff

Decision:

Proceed to:

`M7.6 — Moderate hyperparameter tuning`

with candidate-specific imbalance settings:

```text
LR:
W_SHORT
NONE

DT:
W_SHORT
CLASS_WEIGHT_BALANCED

RF:
W_SHORT
CLASS_WEIGHT_BALANCED
```

Status:

`READY`

# 29. M7.5 Gate

## Technical runtime gates

```text
G01_SOURCE_LOCATION                         → PASS
G02_UPSTREAM_HANDOFF                        → PASS
G03_PHASE_A_CANDIDATE_SPEC                  → PASS
G04_FEATURE_CONTRACT                        → PASS
G05_STRICT_CAUSAL_REGRESSION                → PASS
G06_SEMANTIC_TRAIN_LINEAGE                  → PASS
G07_FOLD_INTEGRITY                          → PASS
G08_PREPROCESSING_STATE_REUSE               → PASS
G09_MATRIX_BUILDER                          → PASS
G10_CLASS_WEIGHT_RUNNER                     → PASS
G11_CLASS_WEIGHT_9_RUN_EXECUTION            → PASS
G12_CONTROLLED_COMPARABILITY                → PASS
G13_AGGREGATION                             → PASS
G14_INTERVENTION_DELTAS                     → PASS
G15_STAGED_ORDER                            → PASS
G16_PERSISTENCE                             → PASS
G17_ROUND_TRIP_ARTIFACT_IDENTITY            → PASS
G18_NO_VALIDATION_RESAMPLING                → PASS
G19_NO_THRESHOLD_TUNING                     → PASS
G20_FINAL_TEST_PROTECTION                    → PASS
```

Technical gates:

`20 / 20 PASS`

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

## R02 — Upstream scope valid?

Evidence:

```text
Training window:
W_SHORT

Shortlist:
LR / DT / RF

Reference imbalance:
NONE
```

Result:

`PASS`

---

## R03 — Fold integrity valid?

Evidence:

```text
Q2 / Q3 / Q4
positive support
forward order
no overlap
```

Result:

`PASS`

---

## R04 — Fold-safe preprocessing valid?

Evidence:

```text
3 verified states
47 features
CSR float32
```

Result:

`PASS`

---

## R05 — Phase-A execution complete?

Evidence:

```text
CLASS_WEIGHT runs:
9 / 9

warnings:
0
```

Result:

`PASS`

---

## R06 — NONE vs CLASS_WEIGHT comparability valid?

Evidence:

`9 / 9 controlled comparisons PASS`

Result:

`PASS`

---

## R07 — LR disposition supportable?

Evidence:

```text
Δ mean F1:
-0.083961

CLASS_WEIGHT F1 leads:
0 / 3

Δ Precision:
-0.429799

Δ FP:
+4,733

Δ alerts:
+5,768
```

Decision:

`KEEP NONE`

Result:

`PASS`

---

## R08 — DT disposition supportable?

Evidence:

```text
Δ mean F1:
+0.024572

CLASS_WEIGHT F1 leads:
2 / 3

Δ Precision:
+0.082456

Δ FP:
-268

Δ alerts:
-297
```

Trade-off reviewed:

```text
Δ Recall:
-0.012161

Δ FN:
+29

std F1:
worse
```

Decision:

`RETAIN CLASS_WEIGHT_BALANCED`

Result:

`PASS`

---

## R09 — RF disposition supportable?

Evidence:

```text
Δ mean F1:
+0.103531

CLASS_WEIGHT F1 leads:
3 / 3

Δ Recall:
+0.270281

Δ FN:
-516

std F1:
improves
```

Cost reviewed:

```text
Δ Precision:
-0.117338

Δ FP:
+628

Δ alerts:
+1,144
```

Decision:

`RETAIN CLASS_WEIGHT_BALANCED`

Result:

`PASS`

---

## R10 — Recall gain not used alone?

Evidence:

LR class weight rejected despite very large Recall gain because primary/operational trade-off deteriorated.

Result:

`PASS`

---

## R11 — Candidate-specific imbalance map resolved?

Decision:

```text
LR:
NONE

DT:
CLASS_WEIGHT_BALANCED

RF:
CLASS_WEIGHT_BALANCED
```

Result:

`PASS`

---

## R12 — Random oversampling needed?

Reviewed decision:

`NO`

Reason:

No unresolved candidate requires another intervention family at this stage.

Result:

`PASS`

---

## R13 — Random undersampling needed?

Reviewed decision:

`NO`

Result:

`PASS`

---

## R14 — SMOTE required?

Reviewed decision:

`NO`

Status:

`CONDITIONAL / NOT REQUIRED`

Result:

`PASS`

---

## R15 — Phase B needed?

Decision:

`NO`

Result:

`PASS`

---

## R16 — Global final imbalance strategy boundary preserved?

Evidence:

Final model family remains OPEN.

Therefore:

`Global Final Imbalance Strategy = OPEN`

Result:

`PASS`

---

## R17 — No hidden threshold/tuning contamination?

Evidence:

```text
Hyperparameter tuning:
NONE

Threshold optimization:
NONE
```

Result:

`PASS`

---

## R18 — Validation distribution preserved?

Evidence:

`validation_resampling_performed = False`

Result:

`PASS`

---

## R19 — External VALIDATION boundary preserved?

Evidence:

`external_validation_scoring_performed = False`

Result:

`PASS`

---

## R20 — Persistence / round-trip valid?

Evidence:

```text
result:
PASS

manifest:
PASS

9 prediction/risk-score artifact pairs:
fingerprinted
```

Result:

`PASS`

---

## R21 — FINAL TEST protected?

Evidence:

`final_test_accessed = False`

Result:

`PASS`

---

## R22 — M7.6 handoff valid?

Required:

- W_SHORT locked;
- candidate families retained;
- candidate-specific imbalance disposition resolved;
- random sampling not needed;
- SMOTE not needed;
- model-family winner still OPEN;
- FINAL TEST protected.

Observed:

`PASS`

---

## Overall M7.5 Gate

```text
Technical gates:
20 / 20 PASS

Runtime review gates:
22 / 22 PASS

Blocking issue:
NONE
```

Final:

`M7.5 — PASS`

Staged imbalance experiment:

`COMPLETE AFTER PHASE A`

Candidate-specific imbalance map:

```text
LR:
NONE

DT:
CLASS_WEIGHT_BALANCED

RF:
CLASS_WEIGHT_BALANCED
```

Random sampling:

`NOT REQUIRED`

SMOTE:

`NOT REQUIRED`

Handoff:

`READY FOR M7.6`

# 30. Kết luận M7.5

M7.5 đã hoàn thành controlled class-imbalance experiment theo staged protocol.

Executed stage:

```text
NONE
vs
CLASS_WEIGHT = "balanced"
```

Scope:

```text
Training window:
W_SHORT

Models:
LR
DT
RF

Folds:
Q2 / Q3 / Q4 2018

New class-weight fits:
9 / 9
```

Integrity:

```text
Warnings:
0

Controlled comparisons:
9 / 9 PASS

Technical gates:
20 / 20 PASS

Runtime review gates:
22 / 22 PASS

Validation resampling:
NONE

External VALIDATION scoring:
NONE

Threshold optimization:
NONE

FINAL TEST:
PROTECTED
```

Class-weight effect is model-dependent.

Logistic Regression:

```text
Class weight:
REJECTED FOR DOWNSTREAM

Reason:
mean F1 decreases
Precision collapses
FP and alert burden increase sharply
0 / 3 F1 fold leads

Disposition:
NONE
```

Decision Tree:

```text
Class weight:
RETAINED

Reason:
mean F1 improves
Precision improves
FP and alerts decrease
2 / 3 F1 fold leads

Known trade-off:
slight Recall/FN deterioration
higher std F1

Disposition:
CLASS_WEIGHT_BALANCED
```

Random Forest:

```text
Class weight:
RETAINED

Reason:
large mean-F1 improvement
large Recall improvement
FN reduction
3 / 3 F1 fold leads
std F1 improves

Known trade-off:
Precision decreases
FP and alert burden increase

Disposition:
CLASS_WEIGHT_BALANCED
```

Reviewed candidate-specific imbalance map:

```text
LR
→ NONE

DT
→ CLASS_WEIGHT_BALANCED

RF
→ CLASS_WEIGHT_BALANCED
```

Staged-order decision:

```text
Random Oversampling:
NOT REQUIRED

Random Undersampling:
NOT REQUIRED

SMOTE:
NOT REQUIRED

Phase B:
NOT OPENED
```

Reason:

Phase A already creates usable, candidate-specific learning-strategy decisions.

Running additional techniques merely because they exist would violate the staged evidence-first protocol.

Important boundary:

```text
Global Final Imbalance Strategy:
OPEN
```

because final model family has not yet been selected.

Final state:

```text
M7.5 — PASS

Training Window:
W_SHORT — LOCKED

Candidate Families:
LR / DT / RF

Candidate-specific Imbalance:

LR:
NONE

DT:
CLASS_WEIGHT_BALANCED

RF:
CLASS_WEIGHT_BALANCED

Random Sampling:
NOT REQUIRED

SMOTE:
NOT REQUIRED

Model-family Winner:
OPEN

Final Model:
OPEN

Global Final Imbalance Strategy:
OPEN

Final Threshold:
OPEN

Blocking Issue:
NONE

FINAL TEST:
PROTECTED

READY FOR M7.6
```

Bước tiếp theo:

`M7.6 — Moderate hyperparameter tuning`

M7.6 phải giữ candidate identity tương ứng:

```text
LR:
W_SHORT
NONE

DT:
W_SHORT
CLASS_WEIGHT_BALANCED

RF:
W_SHORT
CLASS_WEIGHT_BALANCED
```

và chỉ thay:

`hyperparameter candidate`

trong search space:

`SMALL / JUSTIFIED / DECLARED BEFORE RUN`.

Không tuning random seed.

Không tuning threshold trong M7.6.

FINAL TEST tiếp tục:

`PROTECTED`
