# M7.3 — Controlled training-window robustness

Milestone:

`M7 — MODEL SELECTION / ROBUSTNESS / TUNING`

Substep:

`M7.3 — Controlled training-window robustness`

Primary question:

> Hướng ưu tiên tạm thời W_SHORT từ M6 có còn ổn định qua Q2 / Q3 / Q4-2018 temporal folds khi preprocessing được xử lý fold-safe hay không?

M7.3 chạy ba **controlled training-window pairs độc lập**:

```text
TW-LR-B04
W_SHORT vs W_LONG
Logistic Regression
LR-B04-LBFGS-L2-C1

TW-DT-B01
W_SHORT vs W_LONG
Decision Tree
DT-B01-DEFAULT-GINI-UNPRUNED

TW-RF-B01
W_SHORT vs W_LONG
Random Forest
RF-B01-100-GINI-SQRT-BOOTSTRAP
```

Trong từng pair:

`ONLY CLASSIFIER-TRAINING WINDOW CHANGES`.

M7.3 không:

- tune hyperparameter;
- thay model config riêng theo window;
- dùng class weight/resampling;
- tối ưu threshold;
- dùng external VALIDATION 2019 để tune;
- dùng FINAL TEST;
- so model family để shortlist.

Model-family shortlist thuộc:

`M7.4`.

Runtime-dependent state trước khi chạy:

`NOT YET VERIFIED`

## 1. CANON comparison contract

Giữ cùng trong mỗi pair:

```text
feature semantics
fold-safe preprocessing procedure
model family
exact model config
imbalance strategy = NONE
threshold policy = DEFAULT_MODEL_DECISION_RULE
Q2/Q3/Q4 temporal folds
metric implementation
random_state = 42
warning/error handling
```

Biến duy nhất được chủ động thay đổi:

`training_window_id`.

Primary temporal aggregate:

`mean_F1`

Mandatory stability:

`std_F1`

Mandatory secondary:

```text
mean Recall_fraud
mean Precision_fraud
```

Fold-wise evidence bắt buộc:

```text
F1_fraud
Recall_fraud
Precision_fraud
TP / FP / FN / TN
predicted-positive count/rate
runtime
warnings
```

Không dùng arbitrary epsilon để tạo tie.

Predeclared descriptive delta:

`Δ = W_SHORT − W_LONG`

nên:

```text
Δ > 0:
W_SHORT cao hơn

Δ < 0:
W_LONG cao hơn
```

Strict automated robustness label chỉ dùng làm descriptive summary:

```text
ROBUST W_SHORT PREFERENCE
khi W_SHORT có F1 cao hơn ở cả 3 folds
và mean F1 cao hơn

ROBUST W_LONG PREFERENCE
khi W_LONG có F1 cao hơn ở cả 3 folds
và mean F1 cao hơn

còn lại:
MIXED / INCONCLUSIVE
```

Label tự động này **không tự khóa final training-window winner**.

Quyết định cuối M7.3 chỉ được khóa sau runtime review đầy đủ.

## 2. Runtime/resource note

Notebook thực hiện:

```text
6 fold-local preprocessing states
×
3 baseline model families
=
18 model fits
```

Trong đó Random Forest W_LONG là phần có chi phí lớn nhất.

Không được giảm fold, subsample hoặc bỏ W_LONG riêng cho model đắt.

Nếu compute blocker xảy ra:

```text
STOP
→ giữ output hiện có
→ review computational finding
→ thiết kế workaround có kiểm soát
```


```python

from pathlib import Path
from collections import defaultdict
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


## 3. Locate canonical artifacts


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
        "raw + M4.7 + M7.2 artifacts."
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

OUTPUT_DIR = (
    PROJECT_ROOT
    / M7_03_REL
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


EXPECTED_RAW_FILE_SIZE = 2_354_626_737
EXPECTED_RAW_ROWS = 24_386_900
EXPECTED_CARD_COUNT = 6_139

EXPECTED_W_LONG_ROWS = 6_855_270
EXPECTED_W_SHORT_ROWS = 1_721_615

EXPECTED_W_LONG_FRAUD = 9_606
EXPECTED_W_SHORT_FRAUD = 2_491

EXPECTED_FEATURE_COUNT = 47

M4_PIPELINE_VERSION = (
    "M4.7-baseline-v1"
)

M7_RUNNER_VERSION = (
    "M7.2-shared-temporal-runner-v1"
)

M7_CV_SPEC_ID = (
    "M7-TEMPORAL-CV-Q2-Q3-Q4-2018-v1"
)

M7_FOLD_TEMPLATE_ID = (
    "EXPANDING-Q2-Q3-Q4-2018-v1"
)

M7_METRIC_CONTRACT_VERSION = (
    "M7.1-metric-contract-v1"
)

M7_03_ANALYSIS_VERSION = (
    "M7.3-training-window-robustness-v1"
)

FEATURE_VERSION = (
    "Feature Specification v1.0"
)

PREPROCESSING_VERSION = (
    "Preprocessing Specification v1.0 | "
    "M7.2 fold-local"
)

IMBALANCE_STRATEGY = "NONE"

THRESHOLD_POLICY = (
    "DEFAULT_MODEL_DECISION_RULE"
)

RANDOM_STATE = 42

CHUNK_SIZE = 500_000

UNKNOWN_TOKEN = "__UNKNOWN__"


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
    "\nM7.3 SOURCE LOCATION GATE: PASS"
)

```

    PROJECT_ROOT:
    /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening
    
    OUTPUT_DIR:
    /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/processed/m7_03_training_window_robustness
    
    M7.3 SOURCE LOCATION GATE: PASS


## 4. Load M4.7 + M7.2 handoff


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
    == M4_PIPELINE_VERSION
)

assert (
    m4_manifest[
        "output_width"
    ]
    == EXPECTED_FEATURE_COUNT
)

assert (
    m4_manifest[
        "final_test_used"
    ]
    is False
)

assert (
    len(
        canonical_feature_names
    )
    == EXPECTED_FEATURE_COUNT
)


assert (
    m7_02_contract[
        "runner_version"
    ]
    == M7_RUNNER_VERSION
)

assert (
    m7_02_contract[
        "cv_spec_id"
    ]
    == M7_CV_SPEC_ID
)

assert (
    m7_02_contract[
        "fold_template_id"
    ]
    == M7_FOLD_TEMPLATE_ID
)

assert (
    m7_02_contract[
        "metric_contract_version"
    ]
    == M7_METRIC_CONTRACT_VERSION
)

assert (
    m7_02_contract[
        "m4_full_year_transformed_matrices"
    ][
        "safe_for_direct_temporal_cv_slicing"
    ]
    is False
)

assert (
    m7_02_contract[
        "preprocessing_contract"
    ][
        "validation"
    ]
    == "TRANSFORM_ONLY"
)

assert (
    m7_02_contract[
        "preprocessing_contract"
    ][
        "output_width"
    ]
    == EXPECTED_FEATURE_COUNT
)

assert (
    m7_02_contract[
        "selection_state"
    ][
        "training_window_winner"
    ]
    == "OPEN"
)

assert (
    m7_02_contract[
        "selection_state"
    ][
        "m7_02_selection_authorized"
    ]
    is False
)

assert (
    m7_02_contract[
        "final_test_accessed"
    ]
    is False
)


assert (
    m7_02_manifest[
        "fold_states_built"
    ]
    == 6
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
    m7_02_manifest[
        "generic_sampler_compatibility"
    ]
    == "PASS"
)

assert (
    m7_02_manifest[
        "warning_capture"
    ]
    == "PASS"
)

assert (
    m7_02_manifest[
        "positive_class_probability_mapping"
    ]
    == "PASS"
)

assert (
    m7_02_manifest[
        "canonical_schema_enforcement"
    ]
    == "PASS"
)

assert (
    m7_02_manifest[
        "real_model_selection_performed"
    ]
    is False
)

assert (
    m7_02_manifest[
        "final_test_accessed"
    ]
    is False
)


assert len(
    row_long
) == EXPECTED_W_LONG_ROWS

assert int(
    y_long.sum()
) == EXPECTED_W_LONG_FRAUD

assert len(
    row_short
) == EXPECTED_W_SHORT_ROWS

assert int(
    y_short.sum()
) == EXPECTED_W_SHORT_FRAUD


print(
    "M7.2 fold states:",
    m7_02_manifest[
        "fold_states_built"
    ],
)

print(
    "Canonical features:",
    len(
        canonical_feature_names
    ),
)

print(
    "\nM7.3 UPSTREAM HANDOFF GATE: PASS"
)

```

    M7.2 fold states: 6
    Canonical features: 47
    
    M7.3 UPSTREAM HANDOFF GATE: PASS


## 5. Lock folds and official baseline configs before result

M7.3 uses exactly the M5 official baseline configurations.

No tuning is performed here.


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


WINDOW_STARTS = {
    "W_LONG":
        W_LONG_START,

    "W_SHORT":
        W_SHORT_START,
}


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


MODEL_SPECS = {
    "LR": {
        "family":
            "Logistic Regression",

        "model_id":
            "SKLEARN_LOGISTIC_REGRESSION",

        "config_id":
            "LR-B04-LBFGS-L2-C1",

        "hyperparameters": {
            "C":
                1.0,

            "solver":
                "lbfgs",

            "max_iter":
                200,

            "tol":
                1e-4,

            "fit_intercept":
                True,

            "class_weight":
                None,

            "random_state":
                RANDOM_STATE,

            "l1_ratio":
                0.0,
        },
    },

    "DT": {
        "family":
            "Decision Tree",

        "model_id":
            "SKLEARN_DECISION_TREE_CLASSIFIER",

        "config_id":
            "DT-B01-DEFAULT-GINI-UNPRUNED",

        "hyperparameters": {
            "criterion":
                "gini",

            "splitter":
                "best",

            "max_depth":
                None,

            "min_samples_split":
                2,

            "min_samples_leaf":
                1,

            "max_features":
                None,

            "class_weight":
                None,

            "ccp_alpha":
                0.0,

            "random_state":
                RANDOM_STATE,
        },
    },

    "RF": {
        "family":
            "Random Forest",

        "model_id":
            "SKLEARN_RANDOM_FOREST_CLASSIFIER",

        "config_id":
            "RF-B01-100-GINI-SQRT-BOOTSTRAP",

        "hyperparameters": {
            "n_estimators":
                100,

            "criterion":
                "gini",

            "max_depth":
                None,

            "min_samples_split":
                2,

            "min_samples_leaf":
                1,

            "max_features":
                "sqrt",

            "bootstrap":
                True,

            "class_weight":
                None,

            "ccp_alpha":
                0.0,

            "max_samples":
                None,

            "random_state":
                RANDOM_STATE,

            "n_jobs":
                -1,
        },
    },
}


PAIR_SPECS = {
    "TW-LR-B04": {
        "model_key":
            "LR",

        "config_id":
            "LR-B04-LBFGS-L2-C1",
    },

    "TW-DT-B01": {
        "model_key":
            "DT",

        "config_id":
            "DT-B01-DEFAULT-GINI-UNPRUNED",
    },

    "TW-RF-B01": {
        "model_key":
            "RF",

        "config_id":
            "RF-B01-100-GINI-SQRT-BOOTSTRAP",
    },
}


DELTA_CONVENTION = (
    "W_SHORT_MINUS_W_LONG"
)


assert len(
    FOLD_SPECS
) == 3

assert len(
    MODEL_SPECS
) == 3

assert len(
    PAIR_SPECS
) == 3


print(
    "Controlled training-window pairs:"
)

for pair_id, pair in (
    PAIR_SPECS.items()
):
    print(
        pair_id,
        "→",
        MODEL_SPECS[
            pair[
                "model_key"
            ]
        ][
            "family"
        ],
    )


print(
    "\nDelta convention:",
    DELTA_CONVENTION,
)

print(
    "\nM7.3 PRE-RESULT EXPERIMENT SPEC GATE: PASS"
)

```

    Controlled training-window pairs:
    TW-LR-B04 → Logistic Regression
    TW-DT-B01 → Decision Tree
    TW-RF-B01 → Random Forest
    
    Delta convention: W_SHORT_MINUS_W_LONG
    
    M7.3 PRE-RESULT EXPERIMENT SPEC GATE: PASS


## 6. Controlled-pair invariants


```python

SHARED_PAIR_FIELDS = {
    "feature_version":
        FEATURE_VERSION,

    "preprocessing_version":
        PREPROCESSING_VERSION,

    "imbalance_strategy":
        IMBALANCE_STRATEGY,

    "threshold_policy":
        THRESHOLD_POLICY,

    "cv_spec_id":
        M7_CV_SPEC_ID,

    "fold_template_id":
        M7_FOLD_TEMPLATE_ID,

    "metric_contract_version":
        M7_METRIC_CONTRACT_VERSION,

    "random_state":
        RANDOM_STATE,

    "validation_resampling":
        False,

    "final_test_accessed":
        False,
}


for pair_id, pair in (
    PAIR_SPECS.items()
):
    model_spec = (
        MODEL_SPECS[
            pair[
                "model_key"
            ]
        ]
    )

    assert (
        pair[
            "config_id"
        ]
        == model_spec[
            "config_id"
        ]
    )

    assert (
        model_spec[
            "hyperparameters"
        ][
            "class_weight"
        ]
        is None
    )


assert (
    IMBALANCE_STRATEGY
    == "NONE"
)

assert (
    THRESHOLD_POLICY
    == "DEFAULT_MODEL_DECISION_RULE"
)


print(
    "Only intended active variable "
    "inside each pair:"
)

print(
    "training_window_id"
)

print(
    "\nM7.3 CONTROLLED-PAIR CONTRACT GATE: PASS"
)

```

    Only intended active variable inside each pair:
    training_window_id
    
    M7.3 CONTROLLED-PAIR CONTRACT GATE: PASS


## 7. Canonical semantic feature contract

M7.3 rebuilds semantic TRAIN rows once.

It does **not** read raw `Is Fraud?`.

Labels come from canonical M4.7 `y_train_w_long.npy` after exact row-id alignment.

Preprocessing state is reused from verified M7.2 fold-local audit:

- numeric mean/scale are fold-TRAIN learned state;
- categorical schema is exact canonical M4.7 schema;
- M7.2 already verified all 6 fold vocabularies produce the same 47-column feature schema.

This avoids refitting state from future rows while also avoiding repeated raw passes.


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
    "Canonical category widths:"
)

for column in (
    CATEGORICAL_COLUMNS
):
    print(
        column,
        len(
            CANONICAL_CATEGORIES[
                column
            ]
        ),
    )


print(
    "\nM7.3 CANONICAL FEATURE-SCHEMA GATE: PASS"
)

```

    Canonical category widths:
    transaction_mode 4
    location_state 4
    hour_of_day 25
    day_of_week 8
    
    M7.3 CANONICAL FEATURE-SCHEMA GATE: PASS


## 8. Raw representation helpers


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


## 9. Card-block streaming


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


## 10. Strict-causal behavioral features


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


## 11. Core semantic frame + category-code mapping


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


## 12. Strict-causal regression unit test


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
    "M7.3 STRICT-CAUSAL REGRESSION GATE: PASS"
)

```

    M7.3 STRICT-CAUSAL REGRESSION GATE: PASS


## 13. Rebuild canonical semantic TRAIN arrays

Một raw pass duy nhất sẽ tạo compact arrays cho toàn bộ W_LONG TRAIN.

Memory-efficient representation:

```text
row_id:
int64

Timestamp:
datetime64[ns]

4 numeric features:
float32

2 boolean features:
uint8

4 categorical semantic codes:
int16
```

Không lưu raw User/Card/Merchant Name trong classifier representation.

Không đọc raw target.


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
    "\nM7.3 SEMANTIC TRAIN LINEAGE GATE: PASS"
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
    Elapsed seconds: 124.99
    
    M7.3 SEMANTIC TRAIN LINEAGE GATE: PASS


## 14. Build fold registry from exact lineage


```python

timestamp_index = pd.DatetimeIndex(
    semantic_timestamp
)


fold_registry = []


for window_id, window_start in (
    WINDOW_STARTS.items()
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
) == 6


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
    "\nM7.3 FOLD INTEGRITY GATE: PASS"
)

```

    W_LONG FOLD_Q2_2018 | train rows/fraud: 5557560 / 7672 | val rows/fraud: 428953 / 590
    W_LONG FOLD_Q3_2018 | train rows/fraud: 5986513 / 8262 | val rows/fraud: 435178 / 634
    W_LONG FOLD_Q4_2018 | train rows/fraud: 6421691 / 8896 | val rows/fraud: 433579 / 710
    W_SHORT FOLD_Q2_2018 | train rows/fraud: 423905 / 557 | val rows/fraud: 428953 / 590
    W_SHORT FOLD_Q3_2018 | train rows/fraud: 852858 / 1147 | val rows/fraud: 435178 / 634
    W_SHORT FOLD_Q4_2018 | train rows/fraud: 1288036 / 1781 | val rows/fraud: 433579 / 710
    
    M7.3 FOLD INTEGRITY GATE: PASS


## 15. Load verified M7.2 fold-local preprocessing states

M7.2 persisted audit state includes:

```text
numeric_mean
numeric_scale
fit_row_count
fit_min_timestamp
fit_max_timestamp
category_vocab_size
feature_names_match_m4_7
```

M7.3 reuses those verified fold-TRAIN states.

No new preprocessing learning occurs from fold-validation.


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


assert len(
    preprocessing_state_by_key
) == 6


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
    "\nM7.3 PREPROCESSING-STATE REUSE GATE: PASS"
)

```

    Verified preprocessing states: 6
    
    M7.3 PREPROCESSING-STATE REUSE GATE: PASS


## 16. Fold-local matrix builder


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
            "W_LONG",
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
    "\nM7.3 MATRIX BUILDER GATE: PASS"
)

del X_audit

gc.collect()

```

    Audit matrix shape: (2000, 47)
    
    M7.3 MATRIX BUILDER GATE: PASS





    0



## 17. Shared M7 runner contract

Runner semantics are carried forward from M7.2:

- warning capture;
- positive class located through `classes_ == 1`;
- no sampler in M7.3;
- default model decision rule;
- canonical metric bundle;
- exact fold identity;
- FINAL TEST access = false.


```python

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
        tp
        + fp
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

        "TP":
            int(
                tp
            ),

        "FP":
            int(
                fp
            ),

        "FN":
            int(
                fn
            ),

        "TN":
            int(
                tn
            ),

        "predicted_positive_count":
            predicted_positive_count,

        "predicted_positive_rate":
            float(
                predicted_positive_count
                / len(
                    y_true
                )
            ),
    }


def make_estimator(
    model_key,
):
    spec = (
        MODEL_SPECS[
            model_key
        ]
    )

    params = copy.deepcopy(
        spec[
            "hyperparameters"
        ]
    )

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


def run_fold_candidate(
    *,
    model_key,
    candidate_id,
    window_id,
    fold_record,
    X_train,
    y_train,
    X_validation,
    y_validation,
    prediction_rel_path,
    risk_score_rel_path,
):
    model_spec = (
        MODEL_SPECS[
            model_key
        ]
    )

    estimator = (
        make_estimator(
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
                classes
                == 1
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

    metrics = (
        compute_metric_bundle(
            y_validation,
            y_pred,
        )
    )

    fold_result = {
        "candidate_id":
            candidate_id,

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
            FEATURE_VERSION,

        "preprocessing_version":
            PREPROCESSING_VERSION,

        "model_family":
            model_spec[
                "family"
            ],

        "model_config_id":
            model_spec[
                "config_id"
            ],

        "hyperparameters":
            copy.deepcopy(
                model_spec[
                    "hyperparameters"
                ]
            ),

        "imbalance_strategy":
            IMBALANCE_STRATEGY,

        "random_state":
            RANDOM_STATE,

        "threshold_policy":
            THRESHOLD_POLICY,

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
    "M7.3 SHARED RUNNER CONTRACT: DEFINED"
)

```

    M7.3 SHARED RUNNER CONTRACT: DEFINED


## 18. Pre-run artifact identities

M7.3 persists every fold prediction/risk-score artifact.

A progress registry is updated after every completed run so interrupted compute still leaves an audit trail.


```python

RESULT_PATH = (
    OUTPUT_DIR
    / "m7_03_training_window_robustness.json"
)

MANIFEST_PATH = (
    OUTPUT_DIR
    / "m7_03_robustness_manifest.json"
)

PROGRESS_PATH = (
    OUTPUT_DIR
    / "m7_03_progress_registry.json"
)


source_fingerprints = {
    "m4_manifest_sha256":
        sha256_file(
            M4_MANIFEST_PATH
        ),

    "feature_names_sha256":
        sha256_file(
            M4_FEATURE_NAMES_PATH
        ),

    "y_train_w_long_sha256":
        sha256_file(
            Y_LONG_PATH
        ),

    "y_train_w_short_sha256":
        sha256_file(
            Y_SHORT_PATH
        ),

    "row_id_train_w_long_sha256":
        sha256_file(
            ROW_LONG_PATH
        ),

    "row_id_train_w_short_sha256":
        sha256_file(
            ROW_SHORT_PATH
        ),

    "m7_02_contract_sha256":
        sha256_file(
            M7_02_CONTRACT_PATH
        ),

    "m7_02_preprocessing_audit_sha256":
        sha256_file(
            M7_02_PREPROCESSING_AUDIT_PATH
        ),

    "m7_02_manifest_sha256":
        sha256_file(
            M7_02_MANIFEST_PATH
        ),
}


print(
    "Result path:"
)

print(
    RESULT_PATH
)

print(
    "\nProgress path:"
)

print(
    PROGRESS_PATH
)

print(
    "\nM7.3 ARTIFACT SPECIFICATION GATE: PASS"
)

```

    Result path:
    /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/processed/m7_03_training_window_robustness/m7_03_training_window_robustness.json
    
    Progress path:
    /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/processed/m7_03_training_window_robustness/m7_03_progress_registry.json
    
    M7.3 ARTIFACT SPECIFICATION GATE: PASS


## 19. Execute 18 controlled temporal-CV runs

Execution order minimizes matrix rebuild:

```text
for each fold:
    for W_SHORT / W_LONG:
        build one fold-local train/validation matrix pair
        run LR
        run DT
        run RF
        free matrices
```

No model config is changed after seeing an earlier score.

If any warning occurs:

- result is persisted in progress registry;
- notebook stops;
- metric must not be used for selection until warning is reviewed.


```python

fold_results = []
artifact_fingerprints = {}

experiment_start = (
    time.perf_counter()
)


def write_progress_registry(
    records,
    status,
):
    payload = {
        "analysis_version":
            M7_03_ANALYSIS_VERSION,

        "status":
            status,

        "completed_run_count":
            len(
                records
            ),

        "expected_run_count":
            18,

        "fold_results":
            records,

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


write_progress_registry(
    fold_results,
    "RUNNING",
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

    y_validation_fold = (
        y_long[
            validation_indices
        ]
    )

    for window_id in [
        "W_SHORT",
        "W_LONG",
    ]:
        window_start = (
            WINDOW_STARTS[
                window_id
            ]
        )

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

        key = (
            window_id,
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
            window_id,
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
            "Matrix build seconds:",
            round(
                matrix_seconds,
                2,
            ),
        )

        for model_key in [
            "LR",
            "DT",
            "RF",
        ]:
            model_spec = (
                MODEL_SPECS[
                    model_key
                ]
            )

            candidate_id = (
                f"M7.3-"
                f"{model_key}-"
                f"{window_id}-"
                f"{fold_id}"
            )

            prediction_filename = (
                f"{candidate_id}"
                "__y_pred.npy"
            )

            risk_score_filename = (
                f"{candidate_id}"
                "__risk_score.npy"
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

            print(
                "\nRun:",
                candidate_id,
            )

            run_output = (
                run_fold_candidate(
                    model_key=
                        model_key,

                    candidate_id=
                        candidate_id,

                    window_id=
                        window_id,

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

            record = (
                run_output[
                    "fold_result"
                ]
            )

            record[
                "training_window_id"
            ] = window_id

            record[
                "model_key"
            ] = model_key

            record[
                "pair_id"
            ] = next(
                pair_id
                for pair_id, pair
                in PAIR_SPECS.items()
                if pair[
                    "model_key"
                ]
                == model_key
            )

            record[
                "matrix_build_seconds"
            ] = float(
                matrix_seconds
            )

            fold_results.append(
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

            write_progress_registry(
                fold_results,
                "RUNNING",
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
                write_progress_registry(
                    fold_results,
                    "STOPPED_WARNING_REVIEW_REQUIRED",
                )

                raise RuntimeError(
                    "Model warning detected. "
                    "STOP before comparative interpretation."
                )

        del X_train_fold
        del X_validation_fold
        del train_indices
        del y_train_fold

        gc.collect()


write_progress_registry(
    fold_results,
    "COMPLETE",
)


experiment_elapsed = (
    time.perf_counter()
    - experiment_start
)


assert len(
    fold_results
) == 18


print(
    "\nCompleted runs:",
    len(
        fold_results
    ),
)

print(
    "Total experiment seconds:",
    round(
        experiment_elapsed,
        2,
    ),
)

print(
    "\nM7.3 18-RUN EXECUTION GATE: PASS"
)

```

    
    ==================================================
    W_SHORT FOLD_Q2_2018
    Train shape: (423905, 47) | fraud: 557
    Validation shape: (428953, 47) | fraud: 590
    Matrix build seconds: 0.22
    
    Run: M7.3-LR-W_SHORT-FOLD_Q2_2018
    F1 / Recall / Precision: 0.476291 / 0.383051 / 0.629526
    TP / FP / FN / TN: 226 / 133 / 364 / 428230
    Warnings: 0
    
    Run: M7.3-DT-W_SHORT-FOLD_Q2_2018
    F1 / Recall / Precision: 0.468365 / 0.483051 / 0.454545
    TP / FP / FN / TN: 285 / 342 / 305 / 428021
    Warnings: 0
    
    Run: M7.3-RF-W_SHORT-FOLD_Q2_2018
    F1 / Recall / Precision: 0.490176 / 0.401695 / 0.628647
    TP / FP / FN / TN: 237 / 140 / 353 / 428223
    Warnings: 0
    
    ==================================================
    W_LONG FOLD_Q2_2018
    Train shape: (5557560, 47) | fraud: 7672
    Validation shape: (428953, 47) | fraud: 590
    Matrix build seconds: 2.19
    
    Run: M7.3-LR-W_LONG-FOLD_Q2_2018
    F1 / Recall / Precision: 0.0 / 0.0 / 0.0
    TP / FP / FN / TN: 0 / 7 / 590 / 428356
    Warnings: 0
    
    Run: M7.3-DT-W_LONG-FOLD_Q2_2018
    F1 / Recall / Precision: 0.140138 / 0.137288 / 0.14311
    TP / FP / FN / TN: 81 / 485 / 509 / 427878
    Warnings: 0
    
    Run: M7.3-RF-W_LONG-FOLD_Q2_2018
    F1 / Recall / Precision: 0.049459 / 0.027119 / 0.280702
    TP / FP / FN / TN: 16 / 41 / 574 / 428322
    Warnings: 0
    
    ==================================================
    W_SHORT FOLD_Q3_2018
    Train shape: (852858, 47) | fraud: 1147
    Validation shape: (435178, 47) | fraud: 634
    Matrix build seconds: 0.4
    
    Run: M7.3-LR-W_SHORT-FOLD_Q3_2018
    F1 / Recall / Precision: 0.548515 / 0.436909 / 0.736702
    TP / FP / FN / TN: 277 / 99 / 357 / 434445
    Warnings: 0
    
    Run: M7.3-DT-W_SHORT-FOLD_Q3_2018
    F1 / Recall / Precision: 0.494226 / 0.506309 / 0.482707
    TP / FP / FN / TN: 321 / 344 / 313 / 434200
    Warnings: 0
    
    Run: M7.3-RF-W_SHORT-FOLD_Q3_2018
    F1 / Recall / Precision: 0.558006 / 0.458991 / 0.711491
    TP / FP / FN / TN: 291 / 118 / 343 / 434426
    Warnings: 0
    
    ==================================================
    W_LONG FOLD_Q3_2018
    Train shape: (5986513, 47) | fraud: 8262
    Validation shape: (435178, 47) | fraud: 634
    Matrix build seconds: 1.79
    
    Run: M7.3-LR-W_LONG-FOLD_Q3_2018
    F1 / Recall / Precision: 0.02118 / 0.011041 / 0.259259
    TP / FP / FN / TN: 7 / 20 / 627 / 434524
    Warnings: 0
    
    Run: M7.3-DT-W_LONG-FOLD_Q3_2018
    F1 / Recall / Precision: 0.188737 / 0.195584 / 0.182353
    TP / FP / FN / TN: 124 / 556 / 510 / 433988
    Warnings: 0
    
    Run: M7.3-RF-W_LONG-FOLD_Q3_2018
    F1 / Recall / Precision: 0.103967 / 0.059937 / 0.391753
    TP / FP / FN / TN: 38 / 59 / 596 / 434485
    Warnings: 0
    
    ==================================================
    W_SHORT FOLD_Q4_2018
    Train shape: (1288036, 47) | fraud: 1781
    Validation shape: (433579, 47) | fraud: 710
    Matrix build seconds: 0.66
    
    Run: M7.3-LR-W_SHORT-FOLD_Q4_2018
    F1 / Recall / Precision: 0.460501 / 0.33662 / 0.728659
    TP / FP / FN / TN: 239 / 89 / 471 / 432780
    Warnings: 0
    
    Run: M7.3-DT-W_SHORT-FOLD_Q4_2018
    F1 / Recall / Precision: 0.499585 / 0.423944 / 0.608081
    TP / FP / FN / TN: 301 / 194 / 409 / 432675
    Warnings: 0
    
    Run: M7.3-RF-W_SHORT-FOLD_Q4_2018
    F1 / Recall / Precision: 0.504655 / 0.38169 / 0.744505
    TP / FP / FN / TN: 271 / 93 / 439 / 432776
    Warnings: 0
    
    ==================================================
    W_LONG FOLD_Q4_2018
    Train shape: (6421691, 47) | fraud: 8896
    Validation shape: (433579, 47) | fraud: 710
    Matrix build seconds: 2.29
    
    Run: M7.3-LR-W_LONG-FOLD_Q4_2018
    F1 / Recall / Precision: 0.01387 / 0.007042 / 0.454545
    TP / FP / FN / TN: 5 / 6 / 705 / 432863
    Warnings: 0
    
    Run: M7.3-DT-W_LONG-FOLD_Q4_2018
    F1 / Recall / Precision: 0.256908 / 0.242254 / 0.27345
    TP / FP / FN / TN: 172 / 457 / 538 / 432412
    Warnings: 0
    
    Run: M7.3-RF-W_LONG-FOLD_Q4_2018
    F1 / Recall / Precision: 0.163017 / 0.094366 / 0.598214
    TP / FP / FN / TN: 67 / 45 / 643 / 432824
    Warnings: 0
    
    Completed runs: 18
    Total experiment seconds: 2589.22
    
    M7.3 18-RUN EXECUTION GATE: PASS


## 20. Fold-result integrity and controlled-pair checks


```python

records_by_candidate = {
    record[
        "candidate_id"
    ]:
        record
    for record
    in fold_results
}


assert len(
    records_by_candidate
) == 18


for record in fold_results:
    assert (
        record[
            "integrity_status"
        ]
        == "PASS"
    )

    assert (
        record[
            "warning_count"
        ]
        == 0
    )

    assert (
        record[
            "imbalance_strategy"
        ]
        == "NONE"
    )

    assert (
        record[
            "threshold_policy"
        ]
        == THRESHOLD_POLICY
    )

    assert (
        record[
            "random_state"
        ]
        == RANDOM_STATE
    )

    assert (
        record[
            "TP"
        ]
        +
        record[
            "FN"
        ]
        ==
        record[
            "validation_fraud_rows"
        ]
    )

    assert (
        record[
            "TN"
        ]
        +
        record[
            "FP"
        ]
        ==
        (
            record[
                "validation_rows"
            ]
            -
            record[
                "validation_fraud_rows"
            ]
        )
    )

    assert (
        record[
            "TP"
        ]
        +
        record[
            "FP"
        ]
        ==
        record[
            "predicted_positive_count"
        ]
    )


for pair_id, pair in (
    PAIR_SPECS.items()
):
    model_key = (
        pair[
            "model_key"
        ]
    )

    for fold in FOLD_SPECS:
        fold_id = (
            fold[
                "fold_id"
            ]
        )

        short_id = (
            f"M7.3-"
            f"{model_key}-"
            f"W_SHORT-"
            f"{fold_id}"
        )

        long_id = (
            f"M7.3-"
            f"{model_key}-"
            f"W_LONG-"
            f"{fold_id}"
        )

        short_record = (
            records_by_candidate[
                short_id
            ]
        )

        long_record = (
            records_by_candidate[
                long_id
            ]
        )

        assert (
            short_record[
                "fold_id"
            ]
            ==
            long_record[
                "fold_id"
            ]
        )

        assert (
            short_record[
                "validation_start"
            ]
            ==
            long_record[
                "validation_start"
            ]
        )

        assert (
            short_record[
                "validation_end_exclusive"
            ]
            ==
            long_record[
                "validation_end_exclusive"
            ]
        )

        assert (
            short_record[
                "validation_rows"
            ]
            ==
            long_record[
                "validation_rows"
            ]
        )

        assert (
            short_record[
                "validation_fraud_rows"
            ]
            ==
            long_record[
                "validation_fraud_rows"
            ]
        )

        assert (
            short_record[
                "model_family"
            ]
            ==
            long_record[
                "model_family"
            ]
        )

        assert (
            short_record[
                "model_config_id"
            ]
            ==
            long_record[
                "model_config_id"
            ]
        )

        assert (
            short_record[
                "hyperparameters"
            ]
            ==
            long_record[
                "hyperparameters"
            ]
        )

        assert (
            short_record[
                "feature_version"
            ]
            ==
            long_record[
                "feature_version"
            ]
        )

        assert (
            short_record[
                "preprocessing_version"
            ]
            ==
            long_record[
                "preprocessing_version"
            ]
        )

        assert (
            short_record[
                "imbalance_strategy"
            ]
            ==
            long_record[
                "imbalance_strategy"
            ]
            == "NONE"
        )

        assert (
            short_record[
                "threshold_policy"
            ]
            ==
            long_record[
                "threshold_policy"
            ]
        )

        assert (
            short_record[
                "random_state"
            ]
            ==
            long_record[
                "random_state"
            ]
            == RANDOM_STATE
        )


print(
    "Fold results:",
    len(
        fold_results
    ),
)

print(
    "Controlled pair/fold checks:",
    3 * 3,
)

print(
    "\nM7.3 CONTROLLED COMPARABILITY GATE: PASS"
)

```

    Fold results: 18
    Controlled pair/fold checks: 9
    
    M7.3 CONTROLLED COMPARABILITY GATE: PASS


## 21. Aggregate temporal-CV evidence


```python

def pooled_metrics_from_records(
    records,
):
    tp = sum(
        int(
            record[
                "TP"
            ]
        )
        for record
        in records
    )

    fp = sum(
        int(
            record[
                "FP"
            ]
        )
        for record
        in records
    )

    fn = sum(
        int(
            record[
                "FN"
            ]
        )
        for record
        in records
    )

    tn = sum(
        int(
            record[
                "TN"
            ]
        )
        for record
        in records
    )

    precision = (
        tp
        /
        (
            tp
            + fp
        )
        if (
            tp
            + fp
        )
        else 0.0
    )

    recall = (
        tp
        /
        (
            tp
            + fn
        )
        if (
            tp
            + fn
        )
        else 0.0
    )

    f1 = (
        2.0
        * precision
        * recall
        /
        (
            precision
            + recall
        )
        if (
            precision
            + recall
        )
        else 0.0
    )

    return {
        "F1_fraud":
            float(
                f1
            ),

        "Recall_fraud":
            float(
                recall
            ),

        "Precision_fraud":
            float(
                precision
            ),

        "TP":
            int(
                tp
            ),

        "FP":
            int(
                fp
            ),

        "FN":
            int(
                fn
            ),

        "TN":
            int(
                tn
            ),

        "predicted_positive_count":
            int(
                tp
                + fp
            ),

        "validation_rows":
            int(
                tp
                + fp
                + fn
                + tn
            ),
    }


aggregate_records = []


for model_key, model_spec in (
    MODEL_SPECS.items()
):
    for window_id in [
        "W_SHORT",
        "W_LONG",
    ]:
        records = [
            record
            for record
            in fold_results
            if (
                record[
                    "model_key"
                ]
                == model_key
                and
                record[
                    "training_window_id"
                ]
                == window_id
            )
        ]

        records = sorted(
            records,
            key=lambda record:
                [
                    fold[
                        "fold_id"
                    ]
                    for fold
                    in FOLD_SPECS
                ].index(
                    record[
                        "fold_id"
                    ]
                ),
        )

        assert len(
            records
        ) == 3

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

        aggregate_records.append(
            {
                "model_key":
                    model_key,

                "model_family":
                    model_spec[
                        "family"
                    ],

                "model_config_id":
                    model_spec[
                        "config_id"
                    ],

                "training_window_id":
                    window_id,

                "valid_fold_count":
                    3,

                "expected_fold_count":
                    3,

                "fold_ids": [
                    record[
                        "fold_id"
                    ]
                    for record
                    in records
                ],

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

                "foldwise_metrics": [
                    {
                        "fold_id":
                            record[
                                "fold_id"
                            ],

                        "F1_fraud":
                            record[
                                "F1_fraud"
                            ],

                        "Recall_fraud":
                            record[
                                "Recall_fraud"
                            ],

                        "Precision_fraud":
                            record[
                                "Precision_fraud"
                            ],

                        "TP":
                            record[
                                "TP"
                            ],

                        "FP":
                            record[
                                "FP"
                            ],

                        "FN":
                            record[
                                "FN"
                            ],

                        "TN":
                            record[
                                "TN"
                            ],

                        "predicted_positive_count":
                            record[
                                "predicted_positive_count"
                            ],

                        "predicted_positive_rate":
                            record[
                                "predicted_positive_rate"
                            ],

                        "fit_seconds":
                            record[
                                "fit_seconds"
                            ],

                        "prediction_seconds":
                            record[
                                "prediction_seconds"
                            ],
                    }
                    for record
                    in records
                ],

                "pooled_OOF_metrics":
                    pooled_metrics_from_records(
                        records
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

                "warnings_summary":
                    [],

                "stability_findings":
                    "PENDING_PAIR_COMPARISON",

                "tradeoff_findings":
                    "PENDING_PAIR_COMPARISON",

                "selection_status":
                    "REFERENCE",

                "decision_reason":
                    (
                        "M7.3 robustness evidence; "
                        "selection requires pair review."
                    ),

                "next_action":
                    "PAIR_COMPARISON",
            }
        )


assert len(
    aggregate_records
) == 6


for record in aggregate_records:
    print(
        record[
            "model_family"
        ],
        record[
            "training_window_id"
        ],
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
    )


print(
    "\nM7.3 AGGREGATION GATE: PASS"
)

```

    Logistic Regression W_SHORT | mean F1: 0.495102 | std F1: 0.038315 | mean Recall: 0.385526 | mean Precision: 0.698296
    Logistic Regression W_LONG | mean F1: 0.011683 | std F1: 0.008784 | mean Recall: 0.006028 | mean Precision: 0.237935
    Decision Tree W_SHORT | mean F1: 0.487392 | std F1: 0.013631 | mean Recall: 0.471101 | mean Precision: 0.515111
    Decision Tree W_LONG | mean F1: 0.195261 | std F1: 0.047894 | mean Recall: 0.191708 | mean Precision: 0.199637
    Random Forest W_SHORT | mean F1: 0.517612 | std F1: 0.029168 | mean Recall: 0.414125 | mean Precision: 0.694881
    Random Forest W_LONG | mean F1: 0.105481 | std F1: 0.046372 | mean Recall: 0.060474 | mean Precision: 0.423556
    
    M7.3 AGGREGATION GATE: PASS


## 22. Pair-wise W_SHORT vs W_LONG robustness evidence


```python

aggregate_by_key = {
    (
        record[
            "model_key"
        ],
        record[
            "training_window_id"
        ],
    ):
        record
    for record
    in aggregate_records
}


pair_comparisons = []


for pair_id, pair in (
    PAIR_SPECS.items()
):
    model_key = (
        pair[
            "model_key"
        ]
    )

    short_aggregate = (
        aggregate_by_key[
            (
                model_key,
                "W_SHORT",
            )
        ]
    )

    long_aggregate = (
        aggregate_by_key[
            (
                model_key,
                "W_LONG",
            )
        ]
    )

    fold_deltas = []

    short_fold_leads = 0
    long_fold_leads = 0
    equal_fold_f1 = 0

    for fold in FOLD_SPECS:
        fold_id = (
            fold[
                "fold_id"
            ]
        )

        short_record = (
            records_by_candidate[
                f"M7.3-"
                f"{model_key}-"
                f"W_SHORT-"
                f"{fold_id}"
            ]
        )

        long_record = (
            records_by_candidate[
                f"M7.3-"
                f"{model_key}-"
                f"W_LONG-"
                f"{fold_id}"
            ]
        )

        delta_f1 = (
            short_record[
                "F1_fraud"
            ]
            -
            long_record[
                "F1_fraud"
            ]
        )

        if delta_f1 > 0:
            short_fold_leads += 1

        elif delta_f1 < 0:
            long_fold_leads += 1

        else:
            equal_fold_f1 += 1

        fold_deltas.append(
            {
                "fold_id":
                    fold_id,

                "delta_F1_SHORT_MINUS_LONG":
                    float(
                        delta_f1
                    ),

                "delta_Recall_SHORT_MINUS_LONG":
                    float(
                        short_record[
                            "Recall_fraud"
                        ]
                        -
                        long_record[
                            "Recall_fraud"
                        ]
                    ),

                "delta_Precision_SHORT_MINUS_LONG":
                    float(
                        short_record[
                            "Precision_fraud"
                        ]
                        -
                        long_record[
                            "Precision_fraud"
                        ]
                    ),

                "delta_TP_SHORT_MINUS_LONG":
                    int(
                        short_record[
                            "TP"
                        ]
                        -
                        long_record[
                            "TP"
                        ]
                    ),

                "delta_FP_SHORT_MINUS_LONG":
                    int(
                        short_record[
                            "FP"
                        ]
                        -
                        long_record[
                            "FP"
                        ]
                    ),

                "delta_FN_SHORT_MINUS_LONG":
                    int(
                        short_record[
                            "FN"
                        ]
                        -
                        long_record[
                            "FN"
                        ]
                    ),

                "delta_alert_count_SHORT_MINUS_LONG":
                    int(
                        short_record[
                            "predicted_positive_count"
                        ]
                        -
                        long_record[
                            "predicted_positive_count"
                        ]
                    ),
            }
        )

    delta_mean_f1 = (
        short_aggregate[
            "mean_F1"
        ]
        -
        long_aggregate[
            "mean_F1"
        ]
    )

    if (
        short_fold_leads
        == 3
        and
        delta_mean_f1
        > 0
    ):
        automated_state = (
            "ROBUST W_SHORT PREFERENCE"
        )

    elif (
        long_fold_leads
        == 3
        and
        delta_mean_f1
        < 0
    ):
        automated_state = (
            "ROBUST W_LONG PREFERENCE"
        )

    else:
        automated_state = (
            "MIXED / INCONCLUSIVE"
        )

    pair_comparisons.append(
        {
            "pair_id":
                pair_id,

            "model_key":
                model_key,

            "model_family":
                MODEL_SPECS[
                    model_key
                ][
                    "family"
                ],

            "model_config_id":
                MODEL_SPECS[
                    model_key
                ][
                    "config_id"
                ],

            "delta_convention":
                DELTA_CONVENTION,

            "short_fold_F1_leads":
                int(
                    short_fold_leads
                ),

            "long_fold_F1_leads":
                int(
                    long_fold_leads
                ),

            "equal_fold_F1":
                int(
                    equal_fold_f1
                ),

            "delta_mean_F1_SHORT_MINUS_LONG":
                float(
                    delta_mean_f1
                ),

            "delta_std_F1_SHORT_MINUS_LONG":
                float(
                    short_aggregate[
                        "std_F1"
                    ]
                    -
                    long_aggregate[
                        "std_F1"
                    ]
                ),

            "delta_mean_Recall_SHORT_MINUS_LONG":
                float(
                    short_aggregate[
                        "mean_Recall"
                    ]
                    -
                    long_aggregate[
                        "mean_Recall"
                    ]
                ),

            "delta_mean_Precision_SHORT_MINUS_LONG":
                float(
                    short_aggregate[
                        "mean_Precision"
                    ]
                    -
                    long_aggregate[
                        "mean_Precision"
                    ]
                ),

            "fit_time_ratio_LONG_over_SHORT":
                float(
                    long_aggregate[
                        "total_fit_seconds"
                    ]
                    /
                    short_aggregate[
                        "total_fit_seconds"
                    ]
                ),

            "fold_deltas":
                fold_deltas,

            "automated_robustness_state":
                automated_state,

            "selection_decision":
                (
                    "OPEN — REQUIRES M7.3 RUNTIME REVIEW"
                ),
        }
    )


assert len(
    pair_comparisons
) == 3


for pair in pair_comparisons:
    print(
        "\n",
        pair[
            "pair_id"
        ],
        pair[
            "model_family"
        ],
    )

    print(
        "SHORT / LONG fold F1 leads:",
        pair[
            "short_fold_F1_leads"
        ],
        "/",
        pair[
            "long_fold_F1_leads"
        ],
    )

    print(
        "Δ mean F1:",
        round(
            pair[
                "delta_mean_F1_SHORT_MINUS_LONG"
            ],
            6,
        ),
    )

    print(
        "Δ mean Recall:",
        round(
            pair[
                "delta_mean_Recall_SHORT_MINUS_LONG"
            ],
            6,
        ),
    )

    print(
        "Δ mean Precision:",
        round(
            pair[
                "delta_mean_Precision_SHORT_MINUS_LONG"
            ],
            6,
        ),
    )

    print(
        "Automated robustness state:",
        pair[
            "automated_robustness_state"
        ],
    )


print(
    "\nM7.3 PAIR ROBUSTNESS EVIDENCE GATE: PASS"
)

```

    
     TW-LR-B04 Logistic Regression
    SHORT / LONG fold F1 leads: 3 / 0
    Δ mean F1: 0.483419
    Δ mean Recall: 0.379499
    Δ mean Precision: 0.460361
    Automated robustness state: ROBUST W_SHORT PREFERENCE
    
     TW-DT-B01 Decision Tree
    SHORT / LONG fold F1 leads: 3 / 0
    Δ mean F1: 0.292131
    Δ mean Recall: 0.279393
    Δ mean Precision: 0.315474
    Automated robustness state: ROBUST W_SHORT PREFERENCE
    
     TW-RF-B01 Random Forest
    SHORT / LONG fold F1 leads: 3 / 0
    Δ mean F1: 0.412131
    Δ mean Recall: 0.353651
    Δ mean Precision: 0.271325
    Automated robustness state: ROBUST W_SHORT PREFERENCE
    
    M7.3 PAIR ROBUSTNESS EVIDENCE GATE: PASS


## 23. Cross-family synthesis — descriptive only


```python

pair_states = [
    pair[
        "automated_robustness_state"
    ]
    for pair
    in pair_comparisons
]


if all(
    state
    ==
    "ROBUST W_SHORT PREFERENCE"
    for state
    in pair_states
):
    cross_family_robustness_state = (
        "ROBUST W_SHORT PREFERENCE"
    )

elif all(
    state
    ==
    "ROBUST W_LONG PREFERENCE"
    for state
    in pair_states
):
    cross_family_robustness_state = (
        "ROBUST W_LONG PREFERENCE"
    )

else:
    cross_family_robustness_state = (
        "MIXED / INCONCLUSIVE"
    )


print(
    "Pair states:"
)

for pair in pair_comparisons:
    print(
        pair[
            "pair_id"
        ],
        "→",
        pair[
            "automated_robustness_state"
        ],
    )


print(
    "\nCross-family descriptive robustness state:"
)

print(
    cross_family_robustness_state
)


TRAINING_WINDOW_DECISION = (
    "OPEN — REQUIRES M7.3 RUNTIME REVIEW"
)


print(
    "\nTraining-window decision:"
)

print(
    TRAINING_WINDOW_DECISION
)


assert (
    TRAINING_WINDOW_DECISION
    != "W_SHORT"
)

assert (
    TRAINING_WINDOW_DECISION
    != "W_LONG"
)


print(
    "\nM7.3 NO-AUTO-SELECTION GATE: PASS"
)

```

    Pair states:
    TW-LR-B04 → ROBUST W_SHORT PREFERENCE
    TW-DT-B01 → ROBUST W_SHORT PREFERENCE
    TW-RF-B01 → ROBUST W_SHORT PREFERENCE
    
    Cross-family descriptive robustness state:
    ROBUST W_SHORT PREFERENCE
    
    Training-window decision:
    OPEN — REQUIRES M7.3 RUNTIME REVIEW
    
    M7.3 NO-AUTO-SELECTION GATE: PASS


## 24. Persist M7.3 registry and manifest


```python

result_payload = {
    "analysis_version":
        M7_03_ANALYSIS_VERSION,

    "source_fingerprints":
        source_fingerprints,

    "comparison_scope": {
        "training_windows": [
            "W_SHORT",
            "W_LONG",
        ],

        "model_families": [
            MODEL_SPECS[
                key
            ][
                "family"
            ]
            for key
            in [
                "LR",
                "DT",
                "RF",
            ]
        ],

        "pair_ids":
            list(
                PAIR_SPECS
            ),

        "fold_ids": [
            fold[
                "fold_id"
            ]
            for fold
            in FOLD_SPECS
        ],

        "active_variable":
            "training_window_id",

        "imbalance_strategy":
            IMBALANCE_STRATEGY,

        "threshold_policy":
            THRESHOLD_POLICY,

        "random_state":
            RANDOM_STATE,

        "delta_convention":
            DELTA_CONVENTION,
    },

    "fold_registry":
        fold_registry,

    "fold_results":
        fold_results,

    "aggregate_records":
        aggregate_records,

    "pair_comparisons":
        pair_comparisons,

    "cross_family_robustness_state":
        cross_family_robustness_state,

    "artifact_fingerprints":
        artifact_fingerprints,

    "selection_state": {
        "training_window_decision":
            TRAINING_WINDOW_DECISION,

        "model_family_winner":
            "OPEN",

        "final_model":
            "OPEN",

        "final_imbalance_strategy":
            "OPEN",

        "final_threshold":
            "OPEN",

        "m7_03_runtime_review_required":
            True,
    },

    "external_validation_used":
        False,

    "threshold_optimization_performed":
        False,

    "imbalance_intervention_performed":
        False,

    "hyperparameter_tuning_performed":
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
        M7_03_ANALYSIS_VERSION,

    "expected_run_count":
        18,

    "completed_run_count":
        len(
            fold_results
        ),

    "controlled_pair_count":
        len(
            pair_comparisons
        ),

    "fold_count":
        len(
            FOLD_SPECS
        ),

    "warning_count_total":
        int(
            sum(
                record[
                    "warning_count"
                ]
                for record
                in fold_results
            )
        ),

    "integrity_pass_run_count":
        int(
            sum(
                record[
                    "integrity_status"
                ]
                == "PASS"
                for record
                in fold_results
            )
        ),

    "fold_safe_preprocessing_source":
        "M7.2 VERIFIED STATES",

    "only_active_variable":
        "training_window_id",

    "external_validation_used":
        False,

    "threshold_optimization_performed":
        False,

    "imbalance_intervention_performed":
        False,

    "hyperparameter_tuning_performed":
        False,

    "training_window_decision":
        TRAINING_WINDOW_DECISION,

    "model_selection_performed":
        False,

    "final_test_accessed":
        False,

    "decision":
        "OPEN — REQUIRES M7.3 RUNTIME REVIEW",
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

assert RESULT_PATH.stat().st_size > 0
assert MANIFEST_PATH.stat().st_size > 0


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
    "\nM7.3 PERSISTENCE GATE: PASS"
)

```

    Result:
    /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/processed/m7_03_training_window_robustness/m7_03_training_window_robustness.json
    
    Manifest:
    /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/processed/m7_03_training_window_robustness/m7_03_robustness_manifest.json
    
    M7.3 PERSISTENCE GATE: PASS


## 25. Persistence round-trip and artifact identity


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
    == M7_03_ANALYSIS_VERSION
)

assert (
    len(
        result_roundtrip[
            "fold_results"
        ]
    )
    == 18
)

assert (
    len(
        result_roundtrip[
            "aggregate_records"
        ]
    )
    == 6
)

assert (
    len(
        result_roundtrip[
            "pair_comparisons"
        ]
    )
    == 3
)

assert (
    result_roundtrip[
        "selection_state"
    ][
        "training_window_decision"
    ]
    ==
    "OPEN — REQUIRES M7.3 RUNTIME REVIEW"
)

assert (
    result_roundtrip[
        "external_validation_used"
    ]
    is False
)

assert (
    result_roundtrip[
        "threshold_optimization_performed"
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
        "completed_run_count"
    ]
    == 18
)

assert (
    manifest_roundtrip[
        "warning_count_total"
    ]
    == 0
)

assert (
    manifest_roundtrip[
        "integrity_pass_run_count"
    ]
    == 18
)

assert (
    manifest_roundtrip[
        "only_active_variable"
    ]
    == "training_window_id"
)

assert (
    manifest_roundtrip[
        "final_test_accessed"
    ]
    is False
)


for candidate_id, fingerprints in (
    artifact_fingerprints.items()
):
    record = (
        records_by_candidate[
            candidate_id
        ]
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
        fingerprints[
            "prediction_sha256"
        ]
    )

    assert (
        sha256_file(
            risk_score_path
        )
        ==
        fingerprints[
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
    "\nM7.3 ROUND-TRIP / ARTIFACT IDENTITY GATE: PASS"
)

```

    Result SHA256:
    2588836b8af6c2f5061ed5f8f076facbbe287538776a929f09aaf929b029ff6a
    
    Manifest SHA256:
    7b34c1b5a2d20efeb8013a97055a77e6f1bd91c095eda4dfa4b3c72afa013dff
    
    M7.3 ROUND-TRIP / ARTIFACT IDENTITY GATE: PASS


## 26. Leakage / selection-boundary gate


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
        "external_validation_used"
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
        "imbalance_intervention_performed"
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
        "selection_state"
    ][
        "model_family_winner"
    ]
    == "OPEN"
)

assert (
    result_payload[
        "selection_state"
    ][
        "final_model"
    ]
    == "OPEN"
)

assert (
    result_payload[
        "selection_state"
    ][
        "final_threshold"
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
    "External VALIDATION used:",
    result_payload[
        "external_validation_used"
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
    "\nM7.3 LEAKAGE / SELECTION-BOUNDARY GATE: PASS"
)

```

    External VALIDATION used: False
    Threshold optimization: False
    FINAL TEST accessed: False
    
    M7.3 LEAKAGE / SELECTION-BOUNDARY GATE: PASS


## 27. Overall technical gate


```python

m7_03_gates = {
    "G01_SOURCE_LOCATION":
        True,

    "G02_UPSTREAM_M7_2_HANDOFF":
        True,

    "G03_PRE_RESULT_EXPERIMENT_SPEC":
        True,

    "G04_CONTROLLED_PAIR_CONTRACT":
        True,

    "G05_CANONICAL_FEATURE_SCHEMA":
        True,

    "G06_STRICT_CAUSAL_REGRESSION":
        True,

    "G07_SEMANTIC_TRAIN_LINEAGE":
        True,

    "G08_FOLD_INTEGRITY":
        True,

    "G09_PREPROCESSING_STATE_REUSE":
        True,

    "G10_MATRIX_BUILDER":
        True,

    "G11_SHARED_RUNNER_CONTRACT":
        True,

    "G12_ARTIFACT_SPECIFICATION":
        True,

    "G13_18_RUN_EXECUTION":
        True,

    "G14_CONTROLLED_COMPARABILITY":
        True,

    "G15_AGGREGATION":
        True,

    "G16_PAIR_ROBUSTNESS_EVIDENCE":
        True,

    "G17_NO_AUTO_SELECTION":
        True,

    "G18_PERSISTENCE":
        True,

    "G19_ROUND_TRIP_ARTIFACT_IDENTITY":
        True,

    "G20_LEAKAGE_SELECTION_BOUNDARY":
        True,
}


for gate_name, gate_value in (
    m7_03_gates.items()
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
    m7_03_gates
) == 20

assert all(
    m7_03_gates.values()
)


print(
    "\nM7.3 OVERALL TECHNICAL GATE: PASS"
)

```

    G01_SOURCE_LOCATION → PASS
    G02_UPSTREAM_M7_2_HANDOFF → PASS
    G03_PRE_RESULT_EXPERIMENT_SPEC → PASS
    G04_CONTROLLED_PAIR_CONTRACT → PASS
    G05_CANONICAL_FEATURE_SCHEMA → PASS
    G06_STRICT_CAUSAL_REGRESSION → PASS
    G07_SEMANTIC_TRAIN_LINEAGE → PASS
    G08_FOLD_INTEGRITY → PASS
    G09_PREPROCESSING_STATE_REUSE → PASS
    G10_MATRIX_BUILDER → PASS
    G11_SHARED_RUNNER_CONTRACT → PASS
    G12_ARTIFACT_SPECIFICATION → PASS
    G13_18_RUN_EXECUTION → PASS
    G14_CONTROLLED_COMPARABILITY → PASS
    G15_AGGREGATION → PASS
    G16_PAIR_ROBUSTNESS_EVIDENCE → PASS
    G17_NO_AUTO_SELECTION → PASS
    G18_PERSISTENCE → PASS
    G19_ROUND_TRIP_ARTIFACT_IDENTITY → PASS
    G20_LEAKAGE_SELECTION_BOUNDARY → PASS
    
    M7.3 OVERALL TECHNICAL GATE: PASS


# 28. Runtime review và Findings M7.3

## 28.1. Execution integrity

Observed:

```text
Code cells:
26 / 26

Execution count:
1 → 26 liên tục

Runtime errors:
0

stderr:
0
```

Interpretation:

Notebook đã chạy đầy đủ từ đầu đến cuối.

Không có partial execution hoặc runtime exception làm mất hiệu lực comparative evidence.

Status:

`VERIFIED`

---

## 28.2. Upstream M7.2 handoff

Observed runtime gates:

```text
M7.2 contract:
LOADED

Verified preprocessing states:
6

Fold-safe preprocessing:
PASS

Shared runner contract:
AVAILABLE

Canonical feature width:
47
```

M7.3 không slice trực tiếp full-year transformed matrices của M4.7.

Status:

`VERIFIED`

---

## 28.3. Semantic TRAIN lineage

Observed:

```text
Card blocks:
6,139

Raw rows streamed:
24,386,900

Semantic W_LONG rows:
6,855,270

Semantic W_SHORT rows:
1,721,615

Raw-pass elapsed:
124.99 s
```

Runtime gate:

`M7.3 SEMANTIC TRAIN LINEAGE GATE: PASS`

Exact row-id alignment với canonical M4.7:

`PASS`

Status:

`VERIFIED`

---

## 28.4. Six temporal fold populations

Observed:

```text
W_LONG / Q2

train:
5,557,560 rows
7,672 fraud

validation:
428,953 rows
590 fraud
```

```text
W_LONG / Q3

train:
5,986,513 rows
8,262 fraud

validation:
435,178 rows
634 fraud
```

```text
W_LONG / Q4

train:
6,421,691 rows
8,896 fraud

validation:
433,579 rows
710 fraud
```

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

All six states satisfy:

```text
train rows > 0
validation rows > 0
train fraud > 0
validation fraud > 0
temporal order valid
row overlap = 0
```

Runtime gate:

`M7.3 FOLD INTEGRITY GATE: PASS`

Status:

`VERIFIED`

---

## 28.5. Preprocessing-state reuse

Observed:

```text
Verified preprocessing states:
6
```

For each state:

```text
fit source:
FOLD_TRAIN_ONLY

validation fit used:
False

feature count:
47

feature names:
exact M4.7 match
```

Runtime gate:

`M7.3 PREPROCESSING-STATE REUSE GATE: PASS`

Status:

`VERIFIED`

---

## 28.6. Controlled experiment completeness

Expected:

```text
3 model configs
×
2 training windows
×
3 temporal folds
=
18 runs
```

Observed:

```text
Completed runs:
18 / 18

Warnings:
0 / 18
```

Total experiment runtime:

```text
2,589.22 seconds
≈ 43.15 minutes
```

Runtime gate:

`M7.3 18-RUN EXECUTION GATE: PASS`

Status:

`VERIFIED`

---

## 28.7. Controlled comparability

Observed runtime check:

```text
Controlled pair/fold checks:
9 / 9
```

Trong từng model-family/fold pair:

```text
same validation population
same validation fraud support
same model family
same exact model config
same hyperparameters
same feature version
same preprocessing protocol
same imbalance strategy = NONE
same threshold policy
same random_state = 42

only active variable:
training_window_id
```

Runtime gate:

`M7.3 CONTROLLED COMPARABILITY GATE: PASS`

Status:

`VERIFIED`

---

# 29. Predictive robustness findings

## M7.3-F01 — Logistic Regression: W_SHORT robustly dominates W_LONG on F1

Observed aggregate:

```text
W_SHORT

mean F1:
0.495102

std F1:
0.038315

mean Recall:
0.385526

mean Precision:
0.698296
```

```text
W_LONG

mean F1:
0.011683

std F1:
0.008784

mean Recall:
0.006028

mean Precision:
0.237935
```

Delta convention:

`W_SHORT − W_LONG`

Observed:

```text
Δ mean F1:
+0.483419

Δ mean Recall:
+0.379499

Δ mean Precision:
+0.460361

W_SHORT F1 fold leads:
3 / 3

W_LONG F1 fold leads:
0 / 3
```

Fold-wise F1:

```text
Q2:
SHORT 0.476291
LONG  0.000000

Q3:
SHORT 0.548515
LONG  0.021180

Q4:
SHORT 0.460501
LONG  0.013870
```

Pooled raw error evidence across Q2/Q3/Q4:

```text
W_SHORT

TP:
742

FP:
321

FN:
1,192

Predicted positive:
1,063
```

```text
W_LONG

TP:
12

FP:
33

FN:
1,922

Predicted positive:
45
```

Interpretation:

W_LONG tạo ít alert hơn, nhưng phần lớn là vì model gần như không phát hiện fraud dưới default decision rule.

Low alert burden này không thể override mức suy giảm rất lớn về F1 và Recall.

Robustness state:

`ROBUST W_SHORT PREFERENCE`

Status:

`VERIFIED`

---

## M7.3-F02 — Decision Tree: W_SHORT robustly dominates W_LONG và còn giảm FP

Observed aggregate:

```text
W_SHORT

mean F1:
0.487392

std F1:
0.013631

mean Recall:
0.471101

mean Precision:
0.515111
```

```text
W_LONG

mean F1:
0.195261

std F1:
0.047894

mean Recall:
0.191708

mean Precision:
0.199637
```

Observed deltas:

```text
Δ mean F1:
+0.292131

Δ mean Recall:
+0.279393

Δ mean Precision:
+0.315474

W_SHORT F1 fold leads:
3 / 3
```

Fold-wise F1:

```text
Q2:
SHORT 0.468365
LONG  0.140138

Q3:
SHORT 0.494226
LONG  0.188737

Q4:
SHORT 0.499585
LONG  0.256908
```

Pooled raw error evidence:

```text
W_SHORT

TP:
907

FP:
880

FN:
1,027

Predicted positive:
1,787
```

```text
W_LONG

TP:
377

FP:
1,498

FN:
1,557

Predicted positive:
1,875
```

Difference:

```text
W_SHORT vs W_LONG

TP:
+530

FP:
-618

FN:
-530

alerts:
-88
```

Interpretation:

Trong Decision Tree pair, W_SHORT không chỉ tăng fraud detection mà đồng thời giảm false positive và tổng số alert.

Đây là direction rất rõ theo primary metric và raw error burden.

Robustness state:

`ROBUST W_SHORT PREFERENCE`

Status:

`VERIFIED`

---

## M7.3-F03 — Random Forest: W_SHORT robustly dominates W_LONG on F1 / Recall / Precision

Observed aggregate:

```text
W_SHORT

mean F1:
0.517612

std F1:
0.029168

mean Recall:
0.414125

mean Precision:
0.694881
```

```text
W_LONG

mean F1:
0.105481

std F1:
0.046372

mean Recall:
0.060474

mean Precision:
0.423556
```

Observed deltas:

```text
Δ mean F1:
+0.412131

Δ mean Recall:
+0.353651

Δ mean Precision:
+0.271325

W_SHORT F1 fold leads:
3 / 3
```

Fold-wise F1:

```text
Q2:
SHORT 0.490176
LONG  0.049459

Q3:
SHORT 0.558006
LONG  0.103967

Q4:
SHORT 0.504655
LONG  0.163017
```

Pooled raw error evidence:

```text
W_SHORT

TP:
799

FP:
351

FN:
1,135

Predicted positive:
1,150
```

```text
W_LONG

TP:
121

FP:
145

FN:
1,813

Predicted positive:
266
```

Interpretation:

RF-W_LONG có ít alert và FP hơn nhưng bỏ sót thêm rất nhiều fraud.

W_SHORT có advantage lớn và nhất quán trên F1, Recall và Precision qua cả ba folds.

Robustness state:

`ROBUST W_SHORT PREFERENCE`

Status:

`VERIFIED`

---

## M7.3-F04 — Cross-family direction nhất quán

Observed pair states:

```text
TW-LR-B04
→ ROBUST W_SHORT PREFERENCE

TW-DT-B01
→ ROBUST W_SHORT PREFERENCE

TW-RF-B01
→ ROBUST W_SHORT PREFERENCE
```

Cross-family descriptive state:

`ROBUST W_SHORT PREFERENCE`

Tổng số model-fold F1 comparisons:

```text
W_SHORT leads:
9 / 9

W_LONG leads:
0 / 9
```

Interpretation:

M6 provisional W_SHORT direction không chỉ lặp lại trên một external validation snapshot.

Nó tiếp tục xuất hiện nhất quán qua ba forward temporal folds và cả ba official baseline model configurations.

Status:

`VERIFIED`

---

## M7.3-F05 — Không cần dùng parsimony fallback

M7.1 chỉ cho phép:

`W_SHORT = PARSIMONY FALLBACK`

sau khi predictive robustness vẫn không tạo stable preference.

Observed M7.3 evidence không rơi vào trường hợp đó.

Predictive evidence đã phân biệt hai window rõ ràng:

```text
LR:
3 / 3 SHORT F1 leads

DT:
3 / 3 SHORT F1 leads

RF:
3 / 3 SHORT F1 leads
```

Do đó:

`PARSIMONY FALLBACK NOT INVOKED`

Status:

`VERIFIED`

---

## M7.3-F06 — Stability phải đọc cùng performance level

Observed:

```text
LR std F1

SHORT:
0.038315

LONG:
0.008784
```

LR-W_LONG có std F1 nhỏ hơn.

Nhưng absolute F1/Recall của LR-W_LONG gần 0:

```text
mean F1:
0.011683

mean Recall:
0.006028
```

Interpretation:

Low variance quanh một predictive level rất thấp không phải evidence đủ để override primary performance.

Với DT và RF, W_SHORT còn có std F1 thấp hơn W_LONG.

Status:

`REVIEWED`

---

## M7.3-F07 — Computational evidence không quyết định winner

Observed:

W_LONG matrices lớn hơn rõ rệt và M7.3 toàn experiment mất:

`2,589.22 s`.

Nhưng W_SHORT preference không được khóa vì nhanh hơn.

Nó được khóa vì predictive robustness evidence rõ ràng.

Computational evidence tiếp tục chỉ là:

`SUPPORTING EVIDENCE`

Status:

`PROTOCOL-CONSISTENT`

---

## M7.3-F08 — Scope của training-window decision

M7.3 evidence được tạo dưới:

```text
Feature Specification v1.0

M7.2 fold-local preprocessing

official LR / DT / RF baseline configs

imbalance:
NONE

threshold:
DEFAULT_MODEL_DECISION_RULE

random_state:
42
```

Do đó decision đúng là:

> W_SHORT là training-window preference được khóa cho downstream M7 dưới current canonical feature/preprocessing contract và baseline robustness evidence.

Không được suy rộng thành:

> W_SHORT luôn tốt hơn cho mọi future representation/modeling architecture.

Nếu một future step thay đổi modeling assumptions đủ lớn để làm window conclusion không còn comparable, cần đánh giá lại applicability của decision.

Status:

`SCOPE LOCKED`

---

## M7.3-F09 — Model family vẫn OPEN

M7.3 không được dùng aggregate values để chọn LR, DT hoặc RF.

Ví dụ RF-W_SHORT có mean F1 cao nhất trong output hiện tại, nhưng model-family robustness/shortlist thuộc:

`M7.4`.

Status:

`BOUNDARY PRESERVED`

---

## M7.3-F10 — External VALIDATION và FINAL TEST vẫn được bảo vệ

Observed:

```text
External VALIDATION used:
False

Threshold optimization:
False

FINAL TEST accessed:
False
```

Also:

```text
Imbalance intervention:
NONE

Hyperparameter tuning:
NONE
```

Runtime gate:

`M7.3 LEAKAGE / SELECTION-BOUNDARY GATE: PASS`

Status:

`VERIFIED`

---

## M7.3-F11 — Persistence và artifact identity

Persisted:

```text
data/processed/m7_03_training_window_robustness/
    m7_03_training_window_robustness.json
    m7_03_robustness_manifest.json
    m7_03_progress_registry.json
    18 prediction artifacts
    18 risk-score artifacts
```

Observed fingerprints:

```text
Result SHA256:
2588836b8af6c2f5061ed5f8f076facbbe287538776a929f09aaf929b029ff6a

Manifest SHA256:
7b34c1b5a2d20efeb8013a97055a77e6f1bd91c095eda4dfa4b3c72afa013dff
```

Runtime gate:

`M7.3 ROUND-TRIP / ARTIFACT IDENTITY GATE: PASS`

Status:

`VERIFIED`

---

## M7.3-F12 — M7.4 readiness

Necessary evidence now exists:

```text
18 / 18 controlled runs
0 warnings
9 / 9 pair/fold comparability checks
3 / 3 model-family pair states = ROBUST W_SHORT PREFERENCE
external validation untouched
FINAL TEST protected
```

Training-window scope can now be narrowed to:

`W_SHORT`

for M7.4.

Status:

`READY FOR M7.4`

# 30. Decision Log M7.3 — sau runtime review

## M7.3-D01 — Primary question

Decision:

Temporal robustness của:

`W_SHORT vs W_LONG`

đã được thực nghiệm trên Q2/Q3/Q4-2018.

Status:

`COMPLETED`

---

## M7.3-D02 — Model configurations

Decision:

Giữ exact official M5 baseline configs:

```text
LR-B04-LBFGS-L2-C1

DT-B01-DEFAULT-GINI-UNPRUNED

RF-B01-100-GINI-SQRT-BOOTSTRAP
```

No tuning occurred.

Status:

`LOCKED / VERIFIED`

---

## M7.3-D03 — Fold template

Decision:

```text
Q2 / Q3 / Q4 2018

FORWARD / EXPANDING
```

Status:

`INHERITED — VERIFIED — LOCKED`

---

## M7.3-D04 — Active variable

Decision:

Trong mỗi model/fold controlled pair:

`training_window_id`

là biến chủ động duy nhất.

Observed:

`9 / 9 comparability checks PASS`

Status:

`LOCKED / VERIFIED`

---

## M7.3-D05 — Imbalance strategy

Decision:

`NONE`

for all 18 runs.

Status:

`LOCKED / VERIFIED`

---

## M7.3-D06 — Threshold policy

Decision:

`DEFAULT_MODEL_DECISION_RULE`

No numerical threshold optimization.

Status:

`LOCKED / VERIFIED`

---

## M7.3-D07 — Random state

Decision:

`42`

for all stochastic official baseline configs.

Status:

`LOCKED / VERIFIED`

---

## M7.3-D08 — Metric / aggregation contract

Decision:

Primary:

`mean_F1`

Mandatory stability:

`std_F1`

Mandatory secondary:

```text
mean_Recall
mean_Precision
```

plus fold-wise F1/Recall/Precision, confusion counts, alert burden and runtime.

Status:

`LOCKED / VERIFIED`

---

## M7.3-D09 — Delta convention

Decision:

`W_SHORT − W_LONG`

Status:

`LOCKED BEFORE RESULT / VERIFIED`

---

## M7.3-D10 — Logistic Regression window robustness

Observed:

```text
SHORT fold F1 leads:
3 / 3

Δ mean F1:
+0.483419

Δ mean Recall:
+0.379499

Δ mean Precision:
+0.460361
```

Decision:

`ROBUST W_SHORT PREFERENCE`

Status:

`LOCKED`

---

## M7.3-D11 — Decision Tree window robustness

Observed:

```text
SHORT fold F1 leads:
3 / 3

Δ mean F1:
+0.292131

Δ mean Recall:
+0.279393

Δ mean Precision:
+0.315474
```

Decision:

`ROBUST W_SHORT PREFERENCE`

Status:

`LOCKED`

---

## M7.3-D12 — Random Forest window robustness

Observed:

```text
SHORT fold F1 leads:
3 / 3

Δ mean F1:
+0.412131

Δ mean Recall:
+0.353651

Δ mean Precision:
+0.271325
```

Decision:

`ROBUST W_SHORT PREFERENCE`

Status:

`LOCKED`

---

## M7.3-D13 — Cross-family training-window direction

Observed:

```text
LR:
ROBUST W_SHORT PREFERENCE

DT:
ROBUST W_SHORT PREFERENCE

RF:
ROBUST W_SHORT PREFERENCE
```

Decision:

`ROBUST W_SHORT PREFERENCE`

Status:

`LOCKED`

---

## M7.3-D14 — Parsimony fallback

Decision:

Không dùng parsimony để tạo M7.3 preference.

Reason:

predictive robustness đã phân biệt W_SHORT và W_LONG rõ ràng.

Status:

`NOT INVOKED`

---

## M7.3-D15 — Downstream training-window scope

Decision:

`W_SHORT — LOCKED FOR DOWNSTREAM M7`

Meaning:

M7.4 model-family robustness/shortlist phải dùng same training-window scope:

`W_SHORT`.

Scope note:

Decision này thuộc canonical feature/preprocessing + current M7 robustness protocol.

Status:

`LOCKED`

---

## M7.3-D16 — Model-family selection

Decision:

Không chọn LR/DT/RF trong M7.3.

Status:

`DEFERRED TO M7.4`

---

## M7.3-D17 — External VALIDATION

Decision:

Not used in M7.3.

Observed:

`False`

Status:

`LOCKED / VERIFIED`

---

## M7.3-D18 — Hyperparameter tuning

Decision:

None in M7.3.

Status:

`LOCKED / VERIFIED`

---

## M7.3-D19 — Threshold optimization

Decision:

None in M7.3.

Status:

`LOCKED / VERIFIED`

---

## M7.3-D20 — FINAL TEST

Decision:

No access.

Observed:

`False`

Status:

`INHERITED — VERIFIED — LOCKED`

---

## M7.3-D21 — M7.4 handoff

Decision:

Proceed to:

`M7.4 — Model-family robustness / shortlist`

with:

```text
training-window scope:
W_SHORT
```

Status:

`READY`

# 31. M7.3 Gate

## Technical runtime gates

```text
G01_SOURCE_LOCATION                       → PASS
G02_UPSTREAM_M7_2_HANDOFF                 → PASS
G03_PRE_RESULT_EXPERIMENT_SPEC            → PASS
G04_CONTROLLED_PAIR_CONTRACT              → PASS
G05_CANONICAL_FEATURE_SCHEMA              → PASS
G06_STRICT_CAUSAL_REGRESSION              → PASS
G07_SEMANTIC_TRAIN_LINEAGE                → PASS
G08_FOLD_INTEGRITY                        → PASS
G09_PREPROCESSING_STATE_REUSE             → PASS
G10_MATRIX_BUILDER                        → PASS
G11_SHARED_RUNNER_CONTRACT                → PASS
G12_ARTIFACT_SPECIFICATION                → PASS
G13_18_RUN_EXECUTION                      → PASS
G14_CONTROLLED_COMPARABILITY              → PASS
G15_AGGREGATION                           → PASS
G16_PAIR_ROBUSTNESS_EVIDENCE              → PASS
G17_NO_AUTO_SELECTION                     → PASS
G18_PERSISTENCE                           → PASS
G19_ROUND_TRIP_ARTIFACT_IDENTITY          → PASS
G20_LEAKAGE_SELECTION_BOUNDARY            → PASS
```

Technical gates:

`20 / 20 PASS`

---

## R01 — Execution complete?

Evidence:

```text
26 / 26 code cells
execution_count = 1 → 26
errors = 0
stderr = 0
```

Result:

`PASS`

---

## R02 — Upstream M7.2 infrastructure valid?

Evidence:

```text
6 verified preprocessing states
47 canonical features
fold-safe preprocessing
shared runner contract
```

Result:

`PASS`

---

## R03 — Semantic lineage exact?

Evidence:

```text
W_LONG:
6,855,270 rows

W_SHORT:
1,721,615 rows

exact M4.7 row alignment:
PASS
```

Result:

`PASS`

---

## R04 — Six folds valid?

Evidence:

All six window/fold states have:

```text
positive train support
positive validation support
valid temporal ordering
zero train/validation overlap
```

Result:

`PASS`

---

## R05 — Experiment completeness?

Evidence:

```text
18 / 18 runs complete

warnings:
0
```

Result:

`PASS`

---

## R06 — Controlled comparability valid?

Evidence:

```text
9 / 9 pair/fold checks PASS

only active variable:
training_window_id
```

Result:

`PASS`

---

## R07 — LR robustness reviewed?

Evidence:

```text
SHORT F1 leads:
3 / 3

Δ mean F1:
+0.483419
```

Result:

`PASS`

Decision:

`ROBUST W_SHORT PREFERENCE`

---

## R08 — DT robustness reviewed?

Evidence:

```text
SHORT F1 leads:
3 / 3

Δ mean F1:
+0.292131
```

Result:

`PASS`

Decision:

`ROBUST W_SHORT PREFERENCE`

---

## R09 — RF robustness reviewed?

Evidence:

```text
SHORT F1 leads:
3 / 3

Δ mean F1:
+0.412131
```

Result:

`PASS`

Decision:

`ROBUST W_SHORT PREFERENCE`

---

## R10 — Recall / Precision trade-offs reviewed?

Evidence:

All three model families have:

```text
Δ mean Recall:
positive

Δ mean Precision:
positive
```

for W_SHORT − W_LONG.

Result:

`PASS`

---

## R11 — Raw confusion / alert evidence reviewed?

Evidence:

LR, DT and RF pooled Q2/Q3/Q4 TP/FP/FN/alert counts reviewed.

Result:

`PASS`

---

## R12 — Stability reviewed?

Evidence:

mean F1 and std F1 reviewed for all six model-window aggregates.

Result:

`PASS`

---

## R13 — Parsimony used only if needed?

Evidence:

Predictive robustness produced clear W_SHORT preference.

Parsimony fallback:

`NOT USED`

Result:

`PASS`

---

## R14 — Training-window decision supportable?

Evidence:

```text
W_SHORT leads:
9 / 9 model-fold F1 comparisons

3 / 3 model-family robustness states:
ROBUST W_SHORT PREFERENCE
```

Result:

`PASS`

Decision:

`W_SHORT — LOCKED FOR DOWNSTREAM M7`

---

## R15 — Model-family boundary preserved?

Evidence:

No LR/DT/RF shortlist or winner selected in M7.3.

Result:

`PASS`

---

## R16 — External VALIDATION untouched?

Evidence:

`external_validation_used = False`

Result:

`PASS`

---

## R17 — No hidden tuning / imbalance intervention?

Evidence:

```text
Hyperparameter tuning:
NONE

Imbalance intervention:
NONE

Threshold optimization:
NONE
```

Result:

`PASS`

---

## R18 — Persistence / round-trip valid?

Evidence:

```text
result registry:
PASS

manifest:
PASS

18 prediction artifacts:
fingerprinted

18 risk-score artifacts:
fingerprinted
```

Result:

`PASS`

---

## R19 — FINAL TEST protected?

Evidence:

`final_test_accessed = False`

Result:

`PASS`

---

## R20 — M7.4 handoff valid?

Required:

- M7.3 evidence reviewed;
- downstream training-window scope resolved;
- model-family selection still OPEN;
- FINAL TEST protected.

Observed:

`PASS`

---

## Overall M7.3 Gate

```text
Technical gates:
20 / 20 PASS

Runtime review gates:
20 / 20 PASS

Blocking issue:
NONE
```

Final:

`M7.3 — PASS`

Training-window robustness:

`ROBUST W_SHORT PREFERENCE`

Downstream training-window scope:

`W_SHORT — LOCKED FOR DOWNSTREAM M7`

Handoff:

`READY FOR M7.4`

# 32. Kết luận M7.3

M7.3 đã hoàn thành controlled training-window robustness experiment trên:

```text
3 official baseline model configurations

× 2 training windows

× 3 forward temporal folds

=
18 controlled runs
```

Experiment integrity:

```text
Runs:
18 / 18 COMPLETE

Warnings:
0

Controlled pair/fold checks:
9 / 9 PASS

Technical gates:
20 / 20 PASS

Runtime review gates:
20 / 20 PASS
```

Predictive result:

```text
Logistic Regression:
ROBUST W_SHORT PREFERENCE

Decision Tree:
ROBUST W_SHORT PREFERENCE

Random Forest:
ROBUST W_SHORT PREFERENCE
```

Across all model/fold comparisons:

```text
W_SHORT F1 leads:
9 / 9

W_LONG F1 leads:
0 / 9
```

Aggregate F1:

```text
Logistic Regression

W_SHORT:
0.495102

W_LONG:
0.011683
```

```text
Decision Tree

W_SHORT:
0.487392

W_LONG:
0.195261
```

```text
Random Forest

W_SHORT:
0.517612

W_LONG:
0.105481
```

Trong cả ba model families:

```text
Δ mean F1:
positive

Δ mean Recall:
positive

Δ mean Precision:
positive
```

Vì vậy M7.3 không cần dùng parsimony fallback.

Training-window decision được khóa bằng predictive robustness evidence:

```text
Training-window Robustness:
ROBUST W_SHORT PREFERENCE

Training-window Preference:
W_SHORT — LOCKED FOR DOWNSTREAM M7
```

Scope của decision:

```text
Feature:
Feature Specification v1.0

Preprocessing:
M7.2 fold-local canonical preprocessing

Model evidence:
official LR / DT / RF baseline configs

Imbalance:
NONE

Threshold:
DEFAULT_MODEL_DECISION_RULE

Random state:
42
```

M7.3 không lựa chọn model family.

Do đó:

```text
Model-family Winner:
OPEN

Final Model:
OPEN

Final Imbalance Strategy:
OPEN

Final Threshold:
OPEN
```

Boundary protection:

```text
External VALIDATION:
NOT USED

Hyperparameter tuning:
NONE

Threshold optimization:
NONE

FINAL TEST:
PROTECTED
```

Final state:

```text
M7.3 — PASS

Execution Integrity:
VERIFIED

Temporal Fold Integrity:
VERIFIED

Fold-safe Preprocessing:
VERIFIED

Controlled Runs:
18 / 18 COMPLETE

Warnings:
0

Training-window Robustness:
ROBUST W_SHORT PREFERENCE

Training-window Preference:
W_SHORT — LOCKED FOR DOWNSTREAM M7

Model-family Winner:
OPEN

Final Model:
OPEN

Final Imbalance Strategy:
OPEN

Final Threshold:
OPEN

Blocking Issue:
NONE

READY FOR M7.4
```

Bước tiếp theo:

`M7.4 — Model-family robustness / shortlist`

M7.4 phải dùng:

```text
same training-window scope:
W_SHORT

same feature/preprocessing protocol

same folds

same imbalance strategy:
NONE

same threshold policy:
DEFAULT_MODEL_DECISION_RULE
```

Variable:

`model family / baseline config`

M7.4 mới có authority để đánh giá:

```text
Logistic Regression
vs
Decision Tree
vs
Random Forest
```

cho shortlist.

FINAL TEST tiếp tục:

`PROTECTED`
