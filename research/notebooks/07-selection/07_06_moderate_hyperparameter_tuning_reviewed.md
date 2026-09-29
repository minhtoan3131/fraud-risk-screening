# M7.6 — Moderate hyperparameter tuning

Milestone:

`M7 — MODEL SELECTION / ROBUSTNESS / TUNING`

Substep:

`M7.6 — Moderate hyperparameter tuning`

Primary question:

> Với W_SHORT và candidate-specific imbalance strategy đã được khóa qua M7.5, một search space nhỏ, có lý do và khai báo trước có tạo configuration tốt hơn baseline/reference configuration hay không?

CANON yêu cầu:

```text
SMALL
JUSTIFIED
DECLARED BEFORE RUN
TEMPORAL-CV-BASED
```

M7.6 không:

- exhaustive search;
- broad random search;
- mở rộng grid sau khi nhìn result để săn score;
- tuning random seed;
- tuning threshold;
- thay imbalance strategy;
- dùng FINAL TEST.

Upstream candidate identities:

```text
LR
W_SHORT
imbalance = NONE

DT
W_SHORT
imbalance = CLASS_WEIGHT_BALANCED

RF
W_SHORT
imbalance = CLASS_WEIGHT_BALANCED
```

M7.6 chỉ thay:

`hyperparameter candidate`

trong từng model family.

Runtime-dependent tuning decision trước khi chạy:

`OPEN — REQUIRES M7.6 RUNTIME REVIEW`

## 1. Search-space policy

CANON không khóa exact parameter grid.

Vì vậy các giá trị dưới đây là **pre-runtime M7.6 design choices**, không phải giá trị được trích nguyên văn từ CANON.

Chúng được freeze trong notebook này trước khi có M7.6 output.

### Logistic Regression

Reference:

```text
C = 1.0
solver = lbfgs
penalty behavior = L2
imbalance = NONE
```

New candidates:

```text
C = 0.1
C = 10.0
```

Justification:

- chỉ tune một regularization-strength dimension;
- một mức mạnh hơn và một mức yếu hơn reference theo một bậc độ lớn;
- giữ solver / preprocessing / imbalance / threshold cố định.

### Decision Tree

Reference:

```text
max_depth = None
min_samples_leaf = 1
class_weight = balanced
```

M5 từng ghi nhận W_SHORT baseline tree depth khoảng 44, nên M7.6 thử hai regularization directions riêng biệt:

```text
Candidate A:
max_depth = 20

Candidate B:
min_samples_leaf = 5
```

Mỗi candidate chỉ thay một complexity control so với reference.

### Random Forest

Reference:

```text
n_estimators = 100
max_depth = None
min_samples_leaf = 1
max_features = sqrt
class_weight = balanced
```

New candidates:

```text
Candidate A:
max_depth = 20

Candidate B:
min_samples_leaf = 2
```

Justification:

- giữ `n_estimators = 100` để không biến estimator count/runtime thành tuning dimension;
- thử hai regularization controls nhẹ;
- mỗi candidate chỉ thay một dimension so với reference.

Tổng search:

```text
3 families
×
2 new configs
×
3 folds
=
18 new fits
```

Reference configs không retrain; chúng được tái sử dụng từ M7.3/M7.5.

## 2. Tuning comparison contract

Trong từng model family:

Giữ cố định:

```text
training window:
W_SHORT

imbalance strategy:
candidate-specific M7.5 decision

temporal folds:
Q2 / Q3 / Q4 2018

feature/preprocessing:
M7.2 fold-safe canonical contract

threshold policy:
DEFAULT_MODEL_DECISION_RULE

metric code:
same

random state:
42
```

Biến:

`hyperparameter candidate`.

Primary tuning evidence:

```text
mean F1
fold-wise F1
std F1
```

Mandatory supporting evidence:

```text
mean Recall
mean Precision
TP / FP / FN / TN
alert burden
runtime
warnings
failed candidates
```

Possible reviewed outcome per family:

```text
SELECT CONFIG
or
KEEP REFERENCE CONFIG
```

M7.6 không chọn final model family.

Model-family winner vẫn:

`OPEN`.


```python

from pathlib import Path
import copy
import gc
import hashlib
import json
import platform
import sys
import time
import traceback
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

M7_05_REL = (
    Path("data")
    / "processed"
    / "m7_05_controlled_class_imbalance"
)

M7_06_REL = (
    Path("data")
    / "processed"
    / "m7_06_moderate_hyperparameter_tuning"
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

    M7_05_REL
    / "m7_05_class_weight_phase_a.json",

    M7_05_REL
    / "m7_05_phase_a_manifest.json",
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
        "raw + M4.7 + M7.2 + M7.3 + M7.5 artifacts."
    )


RAW_PATH = PROJECT_ROOT / RAW_REL
M4_DIR = PROJECT_ROOT / M4_REL
M7_02_DIR = PROJECT_ROOT / M7_02_REL
M7_03_DIR = PROJECT_ROOT / M7_03_REL
M7_05_DIR = PROJECT_ROOT / M7_05_REL

OUTPUT_DIR = PROJECT_ROOT / M7_06_REL

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


M7_05_RESULT_PATH = (
    M7_05_DIR
    / "m7_05_class_weight_phase_a.json"
)

M7_05_MANIFEST_PATH = (
    M7_05_DIR
    / "m7_05_phase_a_manifest.json"
)


RESULT_PATH = (
    OUTPUT_DIR
    / "m7_06_tuning_registry.json"
)

MANIFEST_PATH = (
    OUTPUT_DIR
    / "m7_06_tuning_manifest.json"
)

PROGRESS_PATH = (
    OUTPUT_DIR
    / "m7_06_tuning_progress.json"
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

M7_06_ANALYSIS_VERSION = (
    "M7.6-moderate-hyperparameter-tuning-v1"
)

UPSTREAM_REVIEWED_M7_5_STATUS = (
    "PASS"
)

UPSTREAM_REVIEWED_TRAINING_WINDOW = (
    "W_SHORT"
)

UPSTREAM_REVIEWED_IMBALANCE_MAP = {
    "LR":
        "NONE",

    "DT":
        "CLASS_WEIGHT_BALANCED",

    "RF":
        "CLASS_WEIGHT_BALANCED",
}

UPSTREAM_REVIEWED_M7_5_NOTEBOOK_SHA256 = (
    "a7d50b9b35a2c8406e893d7f1ee87ff"
    "4133d1159d8f6f58ddd41202ea2a39060"
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
    "\nM7.6 SOURCE LOCATION GATE: PASS"
)

```

    PROJECT_ROOT:
    /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening
    
    OUTPUT_DIR:
    /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/processed/m7_06_moderate_hyperparameter_tuning
    
    M7.6 SOURCE LOCATION GATE: PASS


## 4. Load and verify upstream handoff


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
    M7_05_RESULT_PATH,
    "r",
    encoding="utf-8",
) as file:
    m7_05_result = json.load(
        file
    )


with open(
    M7_05_MANIFEST_PATH,
    "r",
    encoding="utf-8",
) as file:
    m7_05_manifest = json.load(
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
    m7_05_manifest[
        "phase"
    ]
    == "A_CLASS_WEIGHT"
)

assert (
    m7_05_manifest[
        "class_weight_fold_run_count"
    ]
    == 9
)

assert (
    m7_05_manifest[
        "warning_count_total"
    ]
    == 0
)

assert (
    m7_05_manifest[
        "final_test_accessed"
    ]
    is False
)

assert (
    UPSTREAM_REVIEWED_M7_5_STATUS
    == "PASS"
)

assert (
    UPSTREAM_REVIEWED_TRAINING_WINDOW
    == "W_SHORT"
)

assert (
    UPSTREAM_REVIEWED_IMBALANCE_MAP
    ==
    {
        "LR":
            "NONE",

        "DT":
            "CLASS_WEIGHT_BALANCED",

        "RF":
            "CLASS_WEIGHT_BALANCED",
    }
)

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
    "Reviewed imbalance map:",
    UPSTREAM_REVIEWED_IMBALANCE_MAP,
)

print(
    "\nM7.6 UPSTREAM HANDOFF GATE: PASS"
)

```

    Reviewed training window: W_SHORT
    Reviewed imbalance map: {'LR': 'NONE', 'DT': 'CLASS_WEIGHT_BALANCED', 'RF': 'CLASS_WEIGHT_BALANCED'}
    
    M7.6 UPSTREAM HANDOFF GATE: PASS


## 5. Freeze exact tuning search space before runtime

The following search is now immutable for this notebook run.

No candidate may be added after seeing output.


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


# Reference estimator parameters from reviewed upstream artifacts.
m73_lr_reference = next(
    record
    for record
    in m7_03_result[
        "fold_results"
    ]
    if (
        record[
            "model_key"
        ]
        == "LR"
        and
        record[
            "training_window_id"
        ]
        == "W_SHORT"
    )
)


m75_cw_by_model = {}

for model_key in [
    "DT",
    "RF",
]:
    m75_cw_by_model[
        model_key
    ] = next(
        record
        for record
        in m7_05_result[
            "class_weight_fold_results"
        ]
        if (
            record[
                "model_key"
            ]
            == model_key
        )
    )


LR_BASE_PARAMS = copy.deepcopy(
    m73_lr_reference[
        "hyperparameters"
    ]
)

DT_BASE_PARAMS = copy.deepcopy(
    m75_cw_by_model[
        "DT"
    ][
        "hyperparameters"
    ]
)

RF_BASE_PARAMS = copy.deepcopy(
    m75_cw_by_model[
        "RF"
    ][
        "hyperparameters"
    ]
)


assert (
    LR_BASE_PARAMS[
        "class_weight"
    ]
    is None
)

DT_BASE_PARAMS[
    "class_weight"
] = "balanced"

RF_BASE_PARAMS[
    "class_weight"
] = "balanced"


TUNING_CONFIGS = {
    "LR": [
        {
            "config_id":
                "LR-REF-LBFGS-L2-C1",

            "role":
                "REFERENCE",

            "params":
                copy.deepcopy(
                    LR_BASE_PARAMS
                ),

            "source":
                "M7.3 W_SHORT NONE",
        },

        {
            "config_id":
                "LR-T01-LBFGS-L2-C0.1",

            "role":
                "TUNING_CANDIDATE",

            "params": {
                **copy.deepcopy(
                    LR_BASE_PARAMS
                ),

                "C":
                    0.1,
            },

            "source":
                "M7.6 PREDECLARED",
        },

        {
            "config_id":
                "LR-T02-LBFGS-L2-C10",

            "role":
                "TUNING_CANDIDATE",

            "params": {
                **copy.deepcopy(
                    LR_BASE_PARAMS
                ),

                "C":
                    10.0,
            },

            "source":
                "M7.6 PREDECLARED",
        },
    ],

    "DT": [
        {
            "config_id":
                "DT-REF-GINI-UNPRUNED-CW",

            "role":
                "REFERENCE",

            "params":
                copy.deepcopy(
                    DT_BASE_PARAMS
                ),

            "source":
                "M7.5 W_SHORT CLASS_WEIGHT",
        },

        {
            "config_id":
                "DT-T01-GINI-MAXDEPTH20-CW",

            "role":
                "TUNING_CANDIDATE",

            "params": {
                **copy.deepcopy(
                    DT_BASE_PARAMS
                ),

                "max_depth":
                    20,
            },

            "source":
                "M7.6 PREDECLARED",
        },

        {
            "config_id":
                "DT-T02-GINI-MINLEAF5-CW",

            "role":
                "TUNING_CANDIDATE",

            "params": {
                **copy.deepcopy(
                    DT_BASE_PARAMS
                ),

                "min_samples_leaf":
                    5,
            },

            "source":
                "M7.6 PREDECLARED",
        },
    ],

    "RF": [
        {
            "config_id":
                "RF-REF-100-GINI-SQRT-UNPRUNED-CW",

            "role":
                "REFERENCE",

            "params":
                copy.deepcopy(
                    RF_BASE_PARAMS
                ),

            "source":
                "M7.5 W_SHORT CLASS_WEIGHT",
        },

        {
            "config_id":
                "RF-T01-100-GINI-SQRT-MAXDEPTH20-CW",

            "role":
                "TUNING_CANDIDATE",

            "params": {
                **copy.deepcopy(
                    RF_BASE_PARAMS
                ),

                "max_depth":
                    20,
            },

            "source":
                "M7.6 PREDECLARED",
        },

        {
            "config_id":
                "RF-T02-100-GINI-SQRT-MINLEAF2-CW",

            "role":
                "TUNING_CANDIDATE",

            "params": {
                **copy.deepcopy(
                    RF_BASE_PARAMS
                ),

                "min_samples_leaf":
                    2,
            },

            "source":
                "M7.6 PREDECLARED",
        },
    ],
}


assert all(
    len(
        configs
    )
    == 3
    for configs
    in TUNING_CONFIGS.values()
)

assert sum(
    sum(
        config[
            "role"
        ]
        ==
        "TUNING_CANDIDATE"
        for config
        in configs
    )
    for configs
    in TUNING_CONFIGS.values()
) == 6


for model_key in MODEL_KEYS:
    print(
        "\n",
        model_key,
        MODEL_LABELS[
            model_key
        ],
    )

    for config in (
        TUNING_CONFIGS[
            model_key
        ]
    ):
        print(
            " ",
            config[
                "config_id"
            ],
            "|",
            config[
                "role"
            ],
        )


print(
    "\nNew configs:",
    6,
)

print(
    "Expected new fits:",
    18,
)

print(
    "\nM7.6 PRE-RUNTIME SEARCH SPACE GATE: PASS"
)

```

    
     LR Logistic Regression
      LR-REF-LBFGS-L2-C1 | REFERENCE
      LR-T01-LBFGS-L2-C0.1 | TUNING_CANDIDATE
      LR-T02-LBFGS-L2-C10 | TUNING_CANDIDATE
    
     DT Decision Tree
      DT-REF-GINI-UNPRUNED-CW | REFERENCE
      DT-T01-GINI-MAXDEPTH20-CW | TUNING_CANDIDATE
      DT-T02-GINI-MINLEAF5-CW | TUNING_CANDIDATE
    
     RF Random Forest
      RF-REF-100-GINI-SQRT-UNPRUNED-CW | REFERENCE
      RF-T01-100-GINI-SQRT-MAXDEPTH20-CW | TUNING_CANDIDATE
      RF-T02-100-GINI-SQRT-MINLEAF2-CW | TUNING_CANDIDATE
    
    New configs: 6
    Expected new fits: 18
    
    M7.6 PRE-RUNTIME SEARCH SPACE GATE: PASS


## 6. Single-factor search-space audit

New candidates must not accidentally modify unrelated upstream choices.

Expected active dimensions:

```text
LR:
C only

DT-T01:
max_depth only

DT-T02:
min_samples_leaf only

RF-T01:
max_depth only

RF-T02:
min_samples_leaf only
```

Random seed remains 42.

Threshold remains unchanged.

Imbalance strategy remains candidate-specific M7.5 decision.


```python

def changed_keys(
    reference,
    candidate,
):
    keys = set(
        reference
    ) | set(
        candidate
    )

    return sorted(
        key
        for key
        in keys
        if (
            reference.get(
                key
            )
            !=
            candidate.get(
                key
            )
        )
    )


EXPECTED_ACTIVE_KEYS = {
    "LR-T01-LBFGS-L2-C0.1":
        ["C"],

    "LR-T02-LBFGS-L2-C10":
        ["C"],

    "DT-T01-GINI-MAXDEPTH20-CW":
        ["max_depth"],

    "DT-T02-GINI-MINLEAF5-CW":
        ["min_samples_leaf"],

    "RF-T01-100-GINI-SQRT-MAXDEPTH20-CW":
        ["max_depth"],

    "RF-T02-100-GINI-SQRT-MINLEAF2-CW":
        ["min_samples_leaf"],
}


for model_key in MODEL_KEYS:
    reference = (
        TUNING_CONFIGS[
            model_key
        ][
            0
        ][
            "params"
        ]
    )

    for candidate in (
        TUNING_CONFIGS[
            model_key
        ][
            1:
        ]
    ):
        changes = changed_keys(
            reference,
            candidate[
                "params"
            ],
        )

        assert (
            changes
            ==
            EXPECTED_ACTIVE_KEYS[
                candidate[
                    "config_id"
                ]
            ]
        )

        assert (
            candidate[
                "params"
            ][
                "random_state"
            ]
            == RANDOM_STATE
        )

        print(
            candidate[
                "config_id"
            ],
            "→ changed:",
            changes,
        )


print(
    "\nM7.6 SEARCH-SPACE ISOLATION GATE: PASS"
)

```

    LR-T01-LBFGS-L2-C0.1 → changed: ['C']
    LR-T02-LBFGS-L2-C10 → changed: ['C']
    DT-T01-GINI-MAXDEPTH20-CW → changed: ['max_depth']
    DT-T02-GINI-MINLEAF5-CW → changed: ['min_samples_leaf']
    RF-T01-100-GINI-SQRT-MAXDEPTH20-CW → changed: ['max_depth']
    RF-T02-100-GINI-SQRT-MINLEAF2-CW → changed: ['min_samples_leaf']
    
    M7.6 SEARCH-SPACE ISOLATION GATE: PASS


## 7. Canonical feature contract


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
    "\nM7.6 FEATURE CONTRACT GATE: PASS"
)

```

    Canonical feature width: 47
    
    M7.6 FEATURE CONTRACT GATE: PASS


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


## 11. Core semantic representation


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


## 12. Strict-causal regression test


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
    "M7.6 STRICT-CAUSAL REGRESSION GATE: PASS"
)

```

    M7.6 STRICT-CAUSAL REGRESSION GATE: PASS


## 13. Rebuild semantic TRAIN lineage

Same verified semantics as M7.3/M7.5.

Raw target is not read during feature reconstruction.

Labels come from canonical M4.7 target arrays.


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
    "\nM7.6 SEMANTIC TRAIN LINEAGE GATE: PASS"
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
    Elapsed seconds: 134.16
    
    M7.6 SEMANTIC TRAIN LINEAGE GATE: PASS


## 14. Build W_SHORT temporal-fold registry


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
    "\nM7.6 FOLD INTEGRITY GATE: PASS"
)

```

    W_SHORT FOLD_Q2_2018 | train rows/fraud: 423905 / 557 | val rows/fraud: 428953 / 590
    W_SHORT FOLD_Q3_2018 | train rows/fraud: 852858 / 1147 | val rows/fraud: 435178 / 634
    W_SHORT FOLD_Q4_2018 | train rows/fraud: 1288036 / 1781 | val rows/fraud: 433579 / 710
    
    M7.6 FOLD INTEGRITY GATE: PASS


## 15. Load verified W_SHORT preprocessing states


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
    "\nM7.6 PREPROCESSING-STATE REUSE GATE: PASS"
)

```

    Verified preprocessing states: 3
    
    M7.6 PREPROCESSING-STATE REUSE GATE: PASS


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
    "\nM7.6 MATRIX BUILDER GATE: PASS"
)

del X_audit

gc.collect()

```

    Audit matrix shape: (2000, 47)
    
    M7.6 MATRIX BUILDER GATE: PASS





    0



## 17. Reference fold evidence

M7.6 does not retrain reference configs.

Reference source by family:

```text
LR:
M7.3 W_SHORT / NONE

DT:
M7.5 W_SHORT / CLASS_WEIGHT_BALANCED

RF:
M7.5 W_SHORT / CLASS_WEIGHT_BALANCED
```

This makes tuning incremental rather than duplicating already-verified runs.


```python

m73_w_short = [
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


m75_class_weight = (
    m7_05_result[
        "class_weight_fold_results"
    ]
)


reference_fold_records = {
    "LR": {
        record[
            "fold_id"
        ]:
            record
        for record
        in m73_w_short
        if (
            record[
                "model_key"
            ]
            == "LR"
        )
    },

    "DT": {
        record[
            "fold_id"
        ]:
            record
        for record
        in m75_class_weight
        if (
            record[
                "model_key"
            ]
            == "DT"
        )
    },

    "RF": {
        record[
            "fold_id"
        ]:
            record
        for record
        in m75_class_weight
        if (
            record[
                "model_key"
            ]
            == "RF"
        )
    },
}


for model_key in MODEL_KEYS:
    assert set(
        reference_fold_records[
            model_key
        ]
    ) == set(
        FOLD_ORDER
    )


assert all(
    reference_fold_records[
        "LR"
    ][
        fold_id
    ][
        "imbalance_strategy"
    ]
    == "NONE"
    for fold_id
    in FOLD_ORDER
)


for model_key in [
    "DT",
    "RF",
]:
    assert all(
        reference_fold_records[
            model_key
        ][
            fold_id
        ][
            "imbalance_strategy"
        ]
        == "CLASS_WEIGHT"
        for fold_id
        in FOLD_ORDER
    )


print(
    "Reference folds:",
    sum(
        len(
            records
        )
        for records
        in reference_fold_records.values()
    ),
)

print(
    "\nM7.6 REFERENCE EVIDENCE GATE: PASS"
)

```

    Reference folds: 9
    
    M7.6 REFERENCE EVIDENCE GATE: PASS


## 18. Generic tuning runner


```python

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


def make_estimator(
    model_key,
    params,
):
    params = copy.deepcopy(
        params
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


def run_tuning_fold(
    *,
    model_key,
    config,
    fold_record,
    X_train,
    y_train,
    X_validation,
    y_validation,
    prediction_rel_path,
    risk_score_rel_path,
):
    estimator = make_estimator(
        model_key,
        config[
            "params"
        ],
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

        positive_matches = np.flatnonzero(
            classes == 1
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

    return {
        "candidate_id":
            config[
                "config_id"
            ],

        "model_key":
            model_key,

        "model_family":
            MODEL_LABELS[
                model_key
            ],

        "fold_id":
            fold_record[
                "fold_id"
            ],

        "training_window_id":
            "W_SHORT",

        "imbalance_strategy":
            UPSTREAM_REVIEWED_IMBALANCE_MAP[
                model_key
            ],

        "hyperparameters":
            copy.deepcopy(
                config[
                    "params"
                ]
            ),

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

        "threshold_policy":
            "DEFAULT_MODEL_DECISION_RULE",

        "random_state":
            RANDOM_STATE,

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
            len(
                warning_records
            ),

        "warnings":
            warning_records,

        "prediction_artifact":
            str(
                prediction_rel_path
            ),

        "risk_score_artifact":
            str(
                risk_score_rel_path
            ),

        "integrity_status":
            (
                "PASS"
                if len(
                    warning_records
                )
                == 0
                else
                "WARNING_REVIEW_REQUIRED"
            ),

        "_y_pred":
            np.asarray(
                y_pred,
                dtype=np.int8,
            ),

        "_risk_score":
            np.asarray(
                risk_score,
                dtype=np.float32,
            ),
    }


print(
    "M7.6 GENERIC TUNING RUNNER: DEFINED"
)

```

    M7.6 GENERIC TUNING RUNNER: DEFINED


## 19. Execute 18 predeclared tuning fits

Only non-reference candidates are fit.

Execution continues through candidate failures so failed candidates can be persisted.

A failure is never silently dropped.


```python

tuning_fold_results = []
failed_candidate_registry = []
artifact_fingerprints = []


def write_progress(
    status,
):
    payload = {
        "analysis_version":
            M7_06_ANALYSIS_VERSION,

        "status":
            status,

        "completed_fold_results":
            len(
                tuning_fold_results
            ),

        "failed_fold_runs":
            len(
                failed_candidate_registry
            ),

        "expected_new_fold_runs":
            18,

        "tuning_fold_results":
            tuning_fold_results,

        "failed_candidate_registry":
            failed_candidate_registry,

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


run_start = time.perf_counter()


for fold in FOLD_SPECS:
    fold_id = fold[
        "fold_id"
    ]

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

    y_train_fold = y_long[
        train_indices
    ]

    y_validation_fold = y_long[
        validation_indices
    ]

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

    matrix_start = time.perf_counter()

    X_train_fold = build_fold_matrix(
        train_indices,
        preprocessing_state,
    )

    X_validation_fold = build_fold_matrix(
        validation_indices,
        preprocessing_state,
    )

    matrix_seconds = (
        time.perf_counter()
        - matrix_start
    )

    print(
        "\n=================================================="
    )

    print(
        "Fold:",
        fold_id,
    )

    print(
        "Train:",
        X_train_fold.shape,
        "| fraud:",
        int(
            y_train_fold.sum()
        ),
    )

    print(
        "Validation:",
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
        new_configs = [
            config
            for config
            in TUNING_CONFIGS[
                model_key
            ]
            if (
                config[
                    "role"
                ]
                ==
                "TUNING_CANDIDATE"
            )
        ]

        for config in new_configs:
            candidate_id = (
                config[
                    "config_id"
                ]
            )

            run_id = (
                f"M7.6-"
                f"{candidate_id}-"
                f"{fold_id}"
            )

            prediction_path = (
                OUTPUT_DIR
                /
                (
                    run_id
                    + "__y_pred.npy"
                )
            )

            risk_score_path = (
                OUTPUT_DIR
                /
                (
                    run_id
                    + "__risk_score.npy"
                )
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
                run_id,
            )

            try:
                result = run_tuning_fold(
                    model_key=
                        model_key,

                    config=
                        config,

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

                y_pred = result.pop(
                    "_y_pred"
                )

                risk_score = result.pop(
                    "_risk_score"
                )

                result[
                    "run_id"
                ] = run_id

                result[
                    "matrix_build_seconds"
                ] = float(
                    matrix_seconds
                )

                np.save(
                    prediction_path,
                    y_pred,
                    allow_pickle=False,
                )

                np.save(
                    risk_score_path,
                    risk_score,
                    allow_pickle=False,
                )

                artifact_fingerprints.append(
                    {
                        "run_id":
                            run_id,

                        "prediction_sha256":
                            sha256_file(
                                prediction_path
                            ),

                        "risk_score_sha256":
                            sha256_file(
                                risk_score_path
                            ),
                    }
                )

                tuning_fold_results.append(
                    result
                )

                print(
                    "F1 / Recall / Precision:",
                    round(
                        result[
                            "F1_fraud"
                        ],
                        6,
                    ),
                    "/",
                    round(
                        result[
                            "Recall_fraud"
                        ],
                        6,
                    ),
                    "/",
                    round(
                        result[
                            "Precision_fraud"
                        ],
                        6,
                    ),
                )

                print(
                    "TP / FP / FN / TN:",
                    result[
                        "TP"
                    ],
                    "/",
                    result[
                        "FP"
                    ],
                    "/",
                    result[
                        "FN"
                    ],
                    "/",
                    result[
                        "TN"
                    ],
                )

                print(
                    "Warnings:",
                    result[
                        "warning_count"
                    ],
                )

            except Exception as exc:
                failure = {
                    "run_id":
                        run_id,

                    "candidate_id":
                        candidate_id,

                    "model_key":
                        model_key,

                    "fold_id":
                        fold_id,

                    "error_type":
                        type(
                            exc
                        ).__name__,

                    "error_message":
                        str(
                            exc
                        ),

                    "traceback":
                        traceback.format_exc(),

                    "status":
                        "FAILED",
                }

                failed_candidate_registry.append(
                    failure
                )

                print(
                    "FAILED:",
                    failure[
                        "error_type"
                    ],
                    failure[
                        "error_message"
                    ],
                )

            write_progress(
                "RUNNING"
            )

    del X_train_fold
    del X_validation_fold
    del train_indices
    del validation_indices
    del y_train_fold
    del y_validation_fold

    gc.collect()


run_elapsed = (
    time.perf_counter()
    - run_start
)


write_progress(
    "COMPLETE"
)


print(
    "\nCompleted new fold results:",
    len(
        tuning_fold_results
    ),
)

print(
    "Failed fold runs:",
    len(
        failed_candidate_registry
    ),
)

print(
    "Expected new fold runs:",
    18,
)

print(
    "Runtime seconds:",
    round(
        run_elapsed,
        2,
    ),
)


print(
    "\nM7.6 TUNING EXECUTION GATE:",
    (
        "PASS"
        if (
            len(
                tuning_fold_results
            )
            +
            len(
                failed_candidate_registry
            )
            ==
            18
        )
        else "FAIL"
    ),
)


assert (
    len(
        tuning_fold_results
    )
    +
    len(
        failed_candidate_registry
    )
    ==
    18
)

```

    
    ==================================================
    Fold: FOLD_Q2_2018
    Train: (423905, 47) | fraud: 557
    Validation: (428953, 47) | fraud: 590
    Matrix seconds: 0.197
    
    Run: M7.6-LR-T01-LBFGS-L2-C0.1-FOLD_Q2_2018
    F1 / Recall / Precision: 0.361355 / 0.244068 / 0.695652
    TP / FP / FN / TN: 144 / 63 / 446 / 428300
    Warnings: 0
    
    Run: M7.6-LR-T02-LBFGS-L2-C10-FOLD_Q2_2018
    F1 / Recall / Precision: 0.515362 / 0.440678 / 0.620525
    TP / FP / FN / TN: 260 / 159 / 330 / 428204
    Warnings: 0
    
    Run: M7.6-DT-T01-GINI-MAXDEPTH20-CW-FOLD_Q2_2018
    F1 / Recall / Precision: 0.565302 / 0.983051 / 0.396717
    TP / FP / FN / TN: 580 / 882 / 10 / 427481
    Warnings: 0
    
    Run: M7.6-DT-T02-GINI-MINLEAF5-CW-FOLD_Q2_2018
    F1 / Recall / Precision: 0.587302 / 0.815254 / 0.458969
    TP / FP / FN / TN: 481 / 567 / 109 / 427796
    Warnings: 0
    
    Run: M7.6-RF-T01-100-GINI-SQRT-MAXDEPTH20-CW-FOLD_Q2_2018
    F1 / Recall / Precision: 0.565175 / 0.984746 / 0.396317
    TP / FP / FN / TN: 581 / 885 / 9 / 427478
    Warnings: 0
    
    Run: M7.6-RF-T02-100-GINI-SQRT-MINLEAF2-CW-FOLD_Q2_2018
    F1 / Recall / Precision: 0.592324 / 0.967797 / 0.426756
    TP / FP / FN / TN: 571 / 767 / 19 / 427596
    Warnings: 0
    
    ==================================================
    Fold: FOLD_Q3_2018
    Train: (852858, 47) | fraud: 1147
    Validation: (435178, 47) | fraud: 634
    Matrix seconds: 0.676
    
    Run: M7.6-LR-T01-LBFGS-L2-C0.1-FOLD_Q3_2018
    F1 / Recall / Precision: 0.421296 / 0.287066 / 0.791304
    TP / FP / FN / TN: 182 / 48 / 452 / 434496
    Warnings: 0
    
    Run: M7.6-LR-T02-LBFGS-L2-C10-FOLD_Q3_2018
    F1 / Recall / Precision: 0.564547 / 0.462145 / 0.725248
    TP / FP / FN / TN: 293 / 111 / 341 / 434433
    Warnings: 0
    
    Run: M7.6-DT-T01-GINI-MAXDEPTH20-CW-FOLD_Q3_2018
    F1 / Recall / Precision: 0.555261 / 0.990536 / 0.385749
    TP / FP / FN / TN: 628 / 1000 / 6 / 433544
    Warnings: 0
    
    Run: M7.6-DT-T02-GINI-MINLEAF5-CW-FOLD_Q3_2018
    F1 / Recall / Precision: 0.612022 / 0.794953 / 0.497532
    TP / FP / FN / TN: 504 / 509 / 130 / 434035
    Warnings: 0
    
    Run: M7.6-RF-T01-100-GINI-SQRT-MAXDEPTH20-CW-FOLD_Q3_2018
    F1 / Recall / Precision: 0.582985 / 0.988959 / 0.413316
    TP / FP / FN / TN: 627 / 890 / 7 / 433654
    Warnings: 0
    
    Run: M7.6-RF-T02-100-GINI-SQRT-MINLEAF2-CW-FOLD_Q3_2018
    F1 / Recall / Precision: 0.607833 / 0.966877 / 0.443239
    TP / FP / FN / TN: 613 / 770 / 21 / 433774
    Warnings: 0
    
    ==================================================
    Fold: FOLD_Q4_2018
    Train: (1288036, 47) | fraud: 1781
    Validation: (433579, 47) | fraud: 710
    Matrix seconds: 0.543
    
    Run: M7.6-LR-T01-LBFGS-L2-C0.1-FOLD_Q4_2018
    F1 / Recall / Precision: 0.400822 / 0.274648 / 0.741445
    TP / FP / FN / TN: 195 / 68 / 515 / 432801
    Warnings: 0
    
    Run: M7.6-LR-T02-LBFGS-L2-C10-FOLD_Q4_2018
    F1 / Recall / Precision: 0.493075 / 0.376056 / 0.715818
    TP / FP / FN / TN: 267 / 106 / 443 / 432763
    Warnings: 0
    
    Run: M7.6-DT-T01-GINI-MAXDEPTH20-CW-FOLD_Q4_2018
    F1 / Recall / Precision: 0.610188 / 0.776056 / 0.502737
    TP / FP / FN / TN: 551 / 545 / 159 / 432324
    Warnings: 0
    
    Run: M7.6-DT-T02-GINI-MINLEAF5-CW-FOLD_Q4_2018
    F1 / Recall / Precision: 0.617385 / 0.635211 / 0.600533
    TP / FP / FN / TN: 451 / 300 / 259 / 432569
    Warnings: 0
    
    Run: M7.6-RF-T01-100-GINI-SQRT-MAXDEPTH20-CW-FOLD_Q4_2018
    F1 / Recall / Precision: 0.634062 / 0.776056 / 0.535992
    TP / FP / FN / TN: 551 / 477 / 159 / 432392
    Warnings: 0
    
    Run: M7.6-RF-T02-100-GINI-SQRT-MINLEAF2-CW-FOLD_Q4_2018
    F1 / Recall / Precision: 0.649112 / 0.746479 / 0.574215
    TP / FP / FN / TN: 530 / 393 / 180 / 432476
    Warnings: 0
    
    Completed new fold results: 18
    Failed fold runs: 0
    Expected new fold runs: 18
    Runtime seconds: 61.47
    
    M7.6 TUNING EXECUTION GATE: PASS


## 20. Normalize reference + tuning fold registry


```python

def normalize_reference_record(
    model_key,
    config,
    source_record,
):
    params = copy.deepcopy(
        config[
            "params"
        ]
    )

    return {
        "candidate_id":
            config[
                "config_id"
            ],

        "model_key":
            model_key,

        "model_family":
            MODEL_LABELS[
                model_key
            ],

        "fold_id":
            source_record[
                "fold_id"
            ],

        "training_window_id":
            "W_SHORT",

        "imbalance_strategy":
            UPSTREAM_REVIEWED_IMBALANCE_MAP[
                model_key
            ],

        "hyperparameters":
            params,

        "F1_fraud":
            float(
                source_record[
                    "F1_fraud"
                ]
            ),

        "Recall_fraud":
            float(
                source_record[
                    "Recall_fraud"
                ]
            ),

        "Precision_fraud":
            float(
                source_record[
                    "Precision_fraud"
                ]
            ),

        "TP":
            int(
                source_record[
                    "TP"
                ]
            ),

        "FP":
            int(
                source_record[
                    "FP"
                ]
            ),

        "FN":
            int(
                source_record[
                    "FN"
                ]
            ),

        "TN":
            int(
                source_record[
                    "TN"
                ]
            ),

        "predicted_positive_count":
            int(
                source_record[
                    "predicted_positive_count"
                ]
            ),

        "predicted_positive_rate":
            float(
                source_record[
                    "predicted_positive_rate"
                ]
            ),

        "Accuracy_reference":
            float(
                source_record[
                    "Accuracy_reference"
                ]
            ),

        "fit_seconds":
            float(
                source_record[
                    "fit_seconds"
                ]
            ),

        "prediction_seconds":
            float(
                source_record[
                    "prediction_seconds"
                ]
            ),

        "warning_count":
            int(
                source_record[
                    "warning_count"
                ]
            ),

        "warnings":
            copy.deepcopy(
                source_record[
                    "warnings"
                ]
            ),

        "integrity_status":
            source_record[
                "integrity_status"
            ],

        "source_stage":
            (
                "M7.3"
                if model_key
                == "LR"
                else "M7.5"
            ),
    }


reference_normalized = []


for model_key in MODEL_KEYS:
    reference_config = (
        TUNING_CONFIGS[
            model_key
        ][
            0
        ]
    )

    for fold_id in FOLD_ORDER:
        reference_normalized.append(
            normalize_reference_record(
                model_key,
                reference_config,
                reference_fold_records[
                    model_key
                ][
                    fold_id
                ],
            )
        )


all_fold_evidence = (
    reference_normalized
    +
    tuning_fold_results
)


print(
    "Reference fold records:",
    len(
        reference_normalized
    ),
)

print(
    "New tuning fold records:",
    len(
        tuning_fold_results
    ),
)

print(
    "Failed fold runs:",
    len(
        failed_candidate_registry
    ),
)

print(
    "\nM7.6 FOLD REGISTRY GATE: PASS"
)

```

    Reference fold records: 9
    New tuning fold records: 18
    Failed fold runs: 0
    
    M7.6 FOLD REGISTRY GATE: PASS


## 21. Aggregate candidate metrics


```python

def aggregate_candidate_records(
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

    return {
        "valid_fold_count":
            len(
                records
            ),

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
            int(
                sum(
                    record[
                        "TP"
                    ]
                    for record
                    in records
                )
            ),

        "FP":
            int(
                sum(
                    record[
                        "FP"
                    ]
                    for record
                    in records
                )
            ),

        "FN":
            int(
                sum(
                    record[
                        "FN"
                    ]
                    for record
                    in records
                )
            ),

        "TN":
            int(
                sum(
                    record[
                        "TN"
                    ]
                    for record
                    in records
                )
            ),

        "alerts":
            int(
                sum(
                    record[
                        "predicted_positive_count"
                    ]
                    for record
                    in records
                )
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

        "warning_count_total":
            int(
                sum(
                    record[
                        "warning_count"
                    ]
                    for record
                    in records
                )
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

                "alerts":
                    record[
                        "predicted_positive_count"
                    ],
            }
            for record
            in records
        ],
    }


candidate_aggregates = []


for model_key in MODEL_KEYS:
    for config in (
        TUNING_CONFIGS[
            model_key
        ]
    ):
        records = [
            record
            for record
            in all_fold_evidence
            if (
                record[
                    "model_key"
                ]
                == model_key
                and
                record[
                    "candidate_id"
                ]
                ==
                config[
                    "config_id"
                ]
            )
        ]

        failed_folds = [
            failure
            for failure
            in failed_candidate_registry
            if (
                failure[
                    "model_key"
                ]
                == model_key
                and
                failure[
                    "candidate_id"
                ]
                ==
                config[
                    "config_id"
                ]
            )
        ]

        if len(
            records
        ) == 3:
            aggregate = (
                aggregate_candidate_records(
                    records
                )
            )

            aggregate[
                "candidate_id"
            ] = config[
                "config_id"
            ]

            aggregate[
                "model_key"
            ] = model_key

            aggregate[
                "model_family"
            ] = MODEL_LABELS[
                model_key
            ]

            aggregate[
                "role"
            ] = config[
                "role"
            ]

            aggregate[
                "hyperparameters"
            ] = copy.deepcopy(
                config[
                    "params"
                ]
            )

            aggregate[
                "imbalance_strategy"
            ] = (
                UPSTREAM_REVIEWED_IMBALANCE_MAP[
                    model_key
                ]
            )

            aggregate[
                "expected_fold_count"
            ] = 3

            aggregate[
                "failed_fold_count"
            ] = len(
                failed_folds
            )

            aggregate[
                "integrity_status"
            ] = (
                "PASS"
                if (
                    aggregate[
                        "warning_count_total"
                    ]
                    == 0
                    and
                    len(
                        failed_folds
                    )
                    == 0
                )
                else
                "REVIEW_REQUIRED"
            )

        else:
            aggregate = {
                "candidate_id":
                    config[
                        "config_id"
                    ],

                "model_key":
                    model_key,

                "model_family":
                    MODEL_LABELS[
                        model_key
                    ],

                "role":
                    config[
                        "role"
                    ],

                "hyperparameters":
                    copy.deepcopy(
                        config[
                            "params"
                        ]
                    ),

                "imbalance_strategy":
                    UPSTREAM_REVIEWED_IMBALANCE_MAP[
                        model_key
                    ],

                "valid_fold_count":
                    len(
                        records
                    ),

                "expected_fold_count":
                    3,

                "failed_fold_count":
                    len(
                        failed_folds
                    ),

                "integrity_status":
                    "INCOMPLETE",
            }

        candidate_aggregates.append(
            aggregate
        )


assert len(
    candidate_aggregates
) == 9


for aggregate in candidate_aggregates:
    print(
        aggregate[
            "candidate_id"
        ],
        "| folds:",
        aggregate[
            "valid_fold_count"
        ],
        "/",
        aggregate[
            "expected_fold_count"
        ],
        "| status:",
        aggregate[
            "integrity_status"
        ],
        end="",
    )

    if (
        aggregate[
            "valid_fold_count"
        ]
        == 3
    ):
        print(
            "| mean F1:",
            round(
                aggregate[
                    "mean_F1"
                ],
                6,
            ),
            "| std:",
            round(
                aggregate[
                    "std_F1"
                ],
                6,
            ),
        )

    else:
        print()


print(
    "\nM7.6 AGGREGATION GATE: PASS"
)

```

    LR-REF-LBFGS-L2-C1 | folds: 3 / 3 | status: PASS| mean F1: 0.495102 | std: 0.038315
    LR-T01-LBFGS-L2-C0.1 | folds: 3 / 3 | status: PASS| mean F1: 0.394491 | std: 0.024877
    LR-T02-LBFGS-L2-C10 | folds: 3 / 3 | status: PASS| mean F1: 0.524328 | std: 0.029859
    DT-REF-GINI-UNPRUNED-CW | folds: 3 / 3 | status: PASS| mean F1: 0.511964 | std: 0.029242
    DT-T01-GINI-MAXDEPTH20-CW | folds: 3 / 3 | status: PASS| mean F1: 0.576917 | std: 0.023881
    DT-T02-GINI-MINLEAF5-CW | folds: 3 / 3 | status: PASS| mean F1: 0.60557 | std: 0.013102
    RF-REF-100-GINI-SQRT-UNPRUNED-CW | folds: 3 / 3 | status: PASS| mean F1: 0.621144 | std: 0.02069
    RF-T01-100-GINI-SQRT-MAXDEPTH20-CW | folds: 3 / 3 | status: PASS| mean F1: 0.594074 | std: 0.029196
    RF-T02-100-GINI-SQRT-MINLEAF2-CW | folds: 3 / 3 | status: PASS| mean F1: 0.616423 | std: 0.023966
    
    M7.6 AGGREGATION GATE: PASS


## 22. Compare each tuning candidate with its family reference


```python

aggregate_by_id = {
    record[
        "candidate_id"
    ]:
        record
    for record
    in candidate_aggregates
}


tuning_comparisons = []


for model_key in MODEL_KEYS:
    reference_config = (
        TUNING_CONFIGS[
            model_key
        ][
            0
        ]
    )

    reference_agg = (
        aggregate_by_id[
            reference_config[
                "config_id"
            ]
        ]
    )

    for config in (
        TUNING_CONFIGS[
            model_key
        ][
            1:
        ]
    ):
        candidate_agg = (
            aggregate_by_id[
                config[
                    "config_id"
                ]
            ]
        )

        comparison = {
            "model_key":
                model_key,

            "reference_config_id":
                reference_config[
                    "config_id"
                ],

            "candidate_config_id":
                config[
                    "config_id"
                ],

            "candidate_integrity_status":
                candidate_agg[
                    "integrity_status"
                ],

            "decision":
                (
                    "OPEN — REQUIRES "
                    "M7.6 RUNTIME REVIEW"
                ),
        }

        if (
            candidate_agg[
                "valid_fold_count"
            ]
            == 3
        ):
            candidate_fold_leads = 0
            reference_fold_leads = 0

            fold_deltas = []

            for fold_id in FOLD_ORDER:
                ref_fold = next(
                    record
                    for record
                    in all_fold_evidence
                    if (
                        record[
                            "candidate_id"
                        ]
                        ==
                        reference_config[
                            "config_id"
                        ]
                        and
                        record[
                            "fold_id"
                        ]
                        == fold_id
                    )
                )

                cand_fold = next(
                    record
                    for record
                    in all_fold_evidence
                    if (
                        record[
                            "candidate_id"
                        ]
                        ==
                        config[
                            "config_id"
                        ]
                        and
                        record[
                            "fold_id"
                        ]
                        == fold_id
                    )
                )

                delta_f1 = (
                    cand_fold[
                        "F1_fraud"
                    ]
                    -
                    ref_fold[
                        "F1_fraud"
                    ]
                )

                if delta_f1 > 0:
                    candidate_fold_leads += 1

                elif delta_f1 < 0:
                    reference_fold_leads += 1

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
                                cand_fold[
                                    "Recall_fraud"
                                ]
                                -
                                ref_fold[
                                    "Recall_fraud"
                                ]
                            ),

                        "delta_Precision":
                            float(
                                cand_fold[
                                    "Precision_fraud"
                                ]
                                -
                                ref_fold[
                                    "Precision_fraud"
                                ]
                            ),
                    }
                )

            comparison.update(
                {
                    "candidate_F1_fold_leads":
                        candidate_fold_leads,

                    "reference_F1_fold_leads":
                        reference_fold_leads,

                    "delta_mean_F1":
                        float(
                            candidate_agg[
                                "mean_F1"
                            ]
                            -
                            reference_agg[
                                "mean_F1"
                            ]
                        ),

                    "delta_std_F1":
                        float(
                            candidate_agg[
                                "std_F1"
                            ]
                            -
                            reference_agg[
                                "std_F1"
                            ]
                        ),

                    "delta_mean_Recall":
                        float(
                            candidate_agg[
                                "mean_Recall"
                            ]
                            -
                            reference_agg[
                                "mean_Recall"
                            ]
                        ),

                    "delta_mean_Precision":
                        float(
                            candidate_agg[
                                "mean_Precision"
                            ]
                            -
                            reference_agg[
                                "mean_Precision"
                            ]
                        ),

                    "delta_TP":
                        int(
                            candidate_agg[
                                "TP"
                            ]
                            -
                            reference_agg[
                                "TP"
                            ]
                        ),

                    "delta_FP":
                        int(
                            candidate_agg[
                                "FP"
                            ]
                            -
                            reference_agg[
                                "FP"
                            ]
                        ),

                    "delta_FN":
                        int(
                            candidate_agg[
                                "FN"
                            ]
                            -
                            reference_agg[
                                "FN"
                            ]
                        ),

                    "delta_alerts":
                        int(
                            candidate_agg[
                                "alerts"
                            ]
                            -
                            reference_agg[
                                "alerts"
                            ]
                        ),

                    "fit_time_ratio":
                        float(
                            candidate_agg[
                                "total_fit_seconds"
                            ]
                            /
                            reference_agg[
                                "total_fit_seconds"
                            ]
                        ),

                    "fold_deltas":
                        fold_deltas,
                }
            )

        tuning_comparisons.append(
            comparison
        )


assert len(
    tuning_comparisons
) == 6


for comparison in tuning_comparisons:
    print(
        "\n",
        comparison[
            "candidate_config_id"
        ],
    )

    print(
        "Status:",
        comparison[
            "candidate_integrity_status"
        ],
    )

    if (
        "delta_mean_F1"
        in comparison
    ):
        print(
            "F1 fold leads candidate/reference:",
            comparison[
                "candidate_F1_fold_leads"
            ],
            "/",
            comparison[
                "reference_F1_fold_leads"
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
            "Δ Recall / Precision:",
            round(
                comparison[
                    "delta_mean_Recall"
                ],
                6,
            ),
            "/",
            round(
                comparison[
                    "delta_mean_Precision"
                ],
                6,
            ),
        )


print(
    "\nM7.6 REFERENCE COMPARISON GATE: PASS"
)

```

    
     LR-T01-LBFGS-L2-C0.1
    Status: PASS
    F1 fold leads candidate/reference: 0 / 3
    Δ mean F1: -0.100611
    Δ Recall / Precision: -0.116932 / 0.044505
    
     LR-T02-LBFGS-L2-C10
    Status: PASS
    F1 fold leads candidate/reference: 3 / 0
    Δ mean F1: 0.029226
    Δ Recall / Precision: 0.040767 / -0.011099
    
     DT-T01-GINI-MAXDEPTH20-CW
    Status: PASS
    F1 fold leads candidate/reference: 3 / 0
    Δ mean F1: 0.064953
    Δ Recall / Precision: 0.457607 / -0.169166
    
     DT-T02-GINI-MINLEAF5-CW
    Status: PASS
    F1 fold leads candidate/reference: 3 / 0
    Δ mean F1: 0.093606
    Δ Recall / Precision: 0.289532 / -0.078556
    
     RF-T01-100-GINI-SQRT-MAXDEPTH20-CW
    Status: PASS
    F1 fold leads candidate/reference: 1 / 2
    Δ mean F1: -0.02707
    Δ Recall / Precision: 0.232181 / -0.129002
    
     RF-T02-100-GINI-SQRT-MINLEAF2-CW
    Status: PASS
    F1 fold leads candidate/reference: 1 / 2
    Δ mean F1: -0.004721
    Δ Recall / Precision: 0.209311 / -0.09614
    
    M7.6 REFERENCE COMPARISON GATE: PASS


## 23. Tuning decision boundary

No configuration is auto-selected.

Runtime review must decide separately for each model family:

```text
SELECT CONFIG
or
KEEP REFERENCE CONFIG
```

Selection reading order:

```text
mean F1
fold-wise F1
std F1
Recall / Precision
FP / FN
alerts
runtime
warnings / failures
```

No arbitrary epsilon.

No post-result grid expansion in this notebook.

Model-family winner remains OPEN.


```python

TUNING_DECISIONS = {
    "LR":
        "OPEN — REQUIRES M7.6 RUNTIME REVIEW",

    "DT":
        "OPEN — REQUIRES M7.6 RUNTIME REVIEW",

    "RF":
        "OPEN — REQUIRES M7.6 RUNTIME REVIEW",
}


MODEL_FAMILY_WINNER = "OPEN"
FINAL_MODEL = "OPEN"
FINAL_THRESHOLD = "OPEN"


assert all(
    value.startswith(
        "OPEN"
    )
    for value
    in TUNING_DECISIONS.values()
)

assert (
    MODEL_FAMILY_WINNER
    == "OPEN"
)

assert (
    FINAL_MODEL
    == "OPEN"
)

assert (
    FINAL_THRESHOLD
    == "OPEN"
)


print(
    "Tuning decisions:",
    TUNING_DECISIONS,
)

print(
    "\nModel-family winner:",
    MODEL_FAMILY_WINNER,
)

print(
    "\nM7.6 NO-AUTO-SELECTION GATE: PASS"
)

```

    Tuning decisions: {'LR': 'OPEN — REQUIRES M7.6 RUNTIME REVIEW', 'DT': 'OPEN — REQUIRES M7.6 RUNTIME REVIEW', 'RF': 'OPEN — REQUIRES M7.6 RUNTIME REVIEW'}
    
    Model-family winner: OPEN
    
    M7.6 NO-AUTO-SELECTION GATE: PASS


## 24. Persist tuning registry


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

    "m7_05_result_sha256":
        sha256_file(
            M7_05_RESULT_PATH
        ),

    "m7_05_manifest_sha256":
        sha256_file(
            M7_05_MANIFEST_PATH
        ),

    "reviewed_m7_05_notebook_sha256":
        UPSTREAM_REVIEWED_M7_5_NOTEBOOK_SHA256,
}


result_payload = {
    "analysis_version":
        M7_06_ANALYSIS_VERSION,

    "upstream_reviewed_handoff": {
        "m7_05_status":
            UPSTREAM_REVIEWED_M7_5_STATUS,

        "training_window":
            UPSTREAM_REVIEWED_TRAINING_WINDOW,

        "candidate_specific_imbalance":
            UPSTREAM_REVIEWED_IMBALANCE_MAP,

        "reviewed_notebook_sha256":
            UPSTREAM_REVIEWED_M7_5_NOTEBOOK_SHA256,
    },

    "source_fingerprints":
        source_fingerprints,

    "search_policy": {
        "scope":
            "SMALL / JUSTIFIED / DECLARED BEFORE RUN",

        "new_config_count":
            6,

        "new_fold_fit_count":
            18,

        "temporal_cv_required":
            True,

        "random_seed_tuning":
            False,

        "adaptive_grid_expansion":
            False,

        "threshold_tuning":
            False,
    },

    "tuning_configs":
        TUNING_CONFIGS,

    "reference_fold_records":
        reference_normalized,

    "tuning_fold_results":
        tuning_fold_results,

    "failed_candidate_registry":
        failed_candidate_registry,

    "candidate_aggregates":
        candidate_aggregates,

    "tuning_comparisons":
        tuning_comparisons,

    "artifact_fingerprints":
        artifact_fingerprints,

    "selection_state": {
        "family_tuning_decisions":
            TUNING_DECISIONS,

        "model_family_winner":
            MODEL_FAMILY_WINNER,

        "final_model":
            FINAL_MODEL,

        "final_threshold":
            FINAL_THRESHOLD,
    },

    "external_validation_scoring_performed":
        False,

    "imbalance_strategy_changed":
        False,

    "threshold_optimization_performed":
        False,

    "random_seed_tuned":
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
        M7_06_ANALYSIS_VERSION,

    "training_window":
        "W_SHORT",

    "family_count":
        3,

    "reference_config_count":
        3,

    "new_config_count":
        6,

    "expected_new_fold_runs":
        18,

    "completed_new_fold_runs":
        len(
            tuning_fold_results
        ),

    "failed_fold_runs":
        len(
            failed_candidate_registry
        ),

    "warning_count_total":
        int(
            sum(
                record[
                    "warning_count"
                ]
                for record
                in tuning_fold_results
            )
        ),

    "candidate_aggregate_count":
        len(
            candidate_aggregates
        ),

    "family_tuning_decisions":
        TUNING_DECISIONS,

    "model_family_winner":
        "OPEN",

    "external_validation_scoring_performed":
        False,

    "imbalance_strategy_changed":
        False,

    "threshold_optimization_performed":
        False,

    "random_seed_tuned":
        False,

    "final_test_accessed":
        False,

    "decision":
        "OPEN — REQUIRES M7.6 RUNTIME REVIEW",
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
    "\nM7.6 PERSISTENCE GATE: PASS"
)

```

    Result:
    /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/processed/m7_06_moderate_hyperparameter_tuning/m7_06_tuning_registry.json
    
    Manifest:
    /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/processed/m7_06_moderate_hyperparameter_tuning/m7_06_tuning_manifest.json
    
    M7.6 PERSISTENCE GATE: PASS


## 25. Persistence round-trip / artifact identity


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
    == M7_06_ANALYSIS_VERSION
)

assert (
    result_roundtrip[
        "search_policy"
    ][
        "new_config_count"
    ]
    == 6
)

assert (
    result_roundtrip[
        "search_policy"
    ][
        "new_fold_fit_count"
    ]
    == 18
)

assert (
    result_roundtrip[
        "search_policy"
    ][
        "random_seed_tuning"
    ]
    is False
)

assert (
    result_roundtrip[
        "search_policy"
    ][
        "adaptive_grid_expansion"
    ]
    is False
)

assert (
    result_roundtrip[
        "external_validation_scoring_performed"
    ]
    is False
)

assert (
    result_roundtrip[
        "imbalance_strategy_changed"
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
        "expected_new_fold_runs"
    ]
    == 18
)

assert (
    manifest_roundtrip[
        "candidate_aggregate_count"
    ]
    == 9
)

assert (
    manifest_roundtrip[
        "model_family_winner"
    ]
    == "OPEN"
)

assert (
    manifest_roundtrip[
        "final_test_accessed"
    ]
    is False
)


for artifact in artifact_fingerprints:
    run_id = artifact[
        "run_id"
    ]

    record = next(
        record
        for record
        in tuning_fold_results
        if record[
            "run_id"
        ]
        == run_id
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
        artifact[
            "prediction_sha256"
        ]
    )

    assert (
        sha256_file(
            risk_score_path
        )
        ==
        artifact[
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
    "\nM7.6 ROUND-TRIP GATE: PASS"
)

```

    Result SHA256:
    0450890acd3be9c18290208d08455bec3268d12733bb5bc4b99737c9cda13693
    
    Manifest SHA256:
    88a3dc042d3517cd43df492c1c90e9ec158566d74e6759382e0d971c1b261e67
    
    M7.6 ROUND-TRIP GATE: PASS


## 26. Selection-boundary / leakage gate


```python

assert (
    result_payload[
        "external_validation_scoring_performed"
    ]
    is False
)

assert (
    result_payload[
        "imbalance_strategy_changed"
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
        "random_seed_tuned"
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
    "External VALIDATION scoring:",
    result_payload[
        "external_validation_scoring_performed"
    ],
)

print(
    "Imbalance strategy changed:",
    result_payload[
        "imbalance_strategy_changed"
    ],
)

print(
    "Threshold optimization:",
    result_payload[
        "threshold_optimization_performed"
    ],
)

print(
    "Random seed tuned:",
    result_payload[
        "random_seed_tuned"
    ],
)

print(
    "FINAL TEST accessed:",
    result_payload[
        "final_test_accessed"
    ],
)

print(
    "\nM7.6 SELECTION-BOUNDARY GATE: PASS"
)

```

    External VALIDATION scoring: False
    Imbalance strategy changed: False
    Threshold optimization: False
    Random seed tuned: False
    FINAL TEST accessed: False
    
    M7.6 SELECTION-BOUNDARY GATE: PASS


## 27. Overall M7.6 technical gate


```python

m7_06_gates = {
    "G01_SOURCE_LOCATION":
        True,

    "G02_UPSTREAM_HANDOFF":
        True,

    "G03_PRE_RUNTIME_SEARCH_SPACE":
        True,

    "G04_SEARCH_SPACE_ISOLATION":
        True,

    "G05_FEATURE_CONTRACT":
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

    "G11_REFERENCE_EVIDENCE":
        True,

    "G12_GENERIC_TUNING_RUNNER":
        True,

    "G13_TUNING_EXECUTION_ACCOUNTING":
        (
            len(
                tuning_fold_results
            )
            +
            len(
                failed_candidate_registry
            )
            ==
            18
        ),

    "G14_FOLD_REGISTRY":
        True,

    "G15_AGGREGATION":
        True,

    "G16_REFERENCE_COMPARISON":
        True,

    "G17_NO_AUTO_SELECTION":
        True,

    "G18_PERSISTENCE":
        True,

    "G19_ROUND_TRIP":
        True,

    "G20_SELECTION_BOUNDARY":
        True,
}


for gate_name, gate_value in (
    m7_06_gates.items()
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
    m7_06_gates
) == 20

assert all(
    m7_06_gates.values()
)


print(
    "\nM7.6 OVERALL TECHNICAL GATE: PASS"
)

```

    G01_SOURCE_LOCATION → PASS
    G02_UPSTREAM_HANDOFF → PASS
    G03_PRE_RUNTIME_SEARCH_SPACE → PASS
    G04_SEARCH_SPACE_ISOLATION → PASS
    G05_FEATURE_CONTRACT → PASS
    G06_STRICT_CAUSAL_REGRESSION → PASS
    G07_SEMANTIC_TRAIN_LINEAGE → PASS
    G08_FOLD_INTEGRITY → PASS
    G09_PREPROCESSING_STATE_REUSE → PASS
    G10_MATRIX_BUILDER → PASS
    G11_REFERENCE_EVIDENCE → PASS
    G12_GENERIC_TUNING_RUNNER → PASS
    G13_TUNING_EXECUTION_ACCOUNTING → PASS
    G14_FOLD_REGISTRY → PASS
    G15_AGGREGATION → PASS
    G16_REFERENCE_COMPARISON → PASS
    G17_NO_AUTO_SELECTION → PASS
    G18_PERSISTENCE → PASS
    G19_ROUND_TRIP → PASS
    G20_SELECTION_BOUNDARY → PASS
    
    M7.6 OVERALL TECHNICAL GATE: PASS


# 28. Kiểm tra runtime và các phát hiện M7.6

## 28.1. Tính toàn vẹn thực thi

Quan sát:

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

Diễn giải:

Notebook đã chạy đầy đủ từ đầu đến cuối.

Không có thực thi dở dang, exception hoặc stderr làm mất hiệu lực evidence tuning.

Trạng thái:

`VERIFIED`

---

## 28.2. Handoff từ M7.5

Upstream candidate identity đã được giữ nguyên:

```text
LR:
W_SHORT
imbalance = NONE

DT:
W_SHORT
imbalance = CLASS_WEIGHT_BALANCED

RF:
W_SHORT
imbalance = CLASS_WEIGHT_BALANCED
```

Không thay đổi:

```text
training window
imbalance strategy
threshold policy
random state
feature/preprocessing contract
temporal folds
```

Trạng thái:

`VERIFIED`

---

## 28.3. Search space đã được khóa trước runtime

Search space thực tế:

```text
Logistic Regression

reference:
C = 1.0

candidate:
C = 0.1

candidate:
C = 10.0
```

```text
Decision Tree

reference:
max_depth = None
min_samples_leaf = 1

candidate:
max_depth = 20

candidate:
min_samples_leaf = 5
```

```text
Random Forest

reference:
max_depth = None
min_samples_leaf = 1
n_estimators = 100

candidate:
max_depth = 20

candidate:
min_samples_leaf = 2
```

Mỗi tuning candidate chỉ thay đúng một hyperparameter dimension đã khai báo.

Không có:

```text
adaptive grid expansion
random-seed tuning
threshold tuning
imbalance-strategy change
```

Runtime gates:

```text
M7.6 PRE-RUNTIME SEARCH SPACE GATE:
PASS

M7.6 SEARCH-SPACE ISOLATION GATE:
PASS
```

Trạng thái:

`VERIFIED`

---

## 28.4. Fold / preprocessing integrity

Observed:

```text
Reference fold records:
9

W_SHORT temporal folds:
Q2 / Q3 / Q4 2018

Fold-safe preprocessing:
VERIFIED

Feature width:
47
```

Các fold sử dụng cùng causal semantics và fold-local preprocessing đã được khóa từ M7.2.

Trạng thái:

`VERIFIED`

---

## 28.5. Tuning execution completeness

Expected:

```text
6 new configs
×
3 folds
=
18 new fold runs
```

Observed:

```text
Completed new fold results:
18 / 18

Failed fold runs:
0

Warnings:
0

Runtime:
61.47 seconds
```

Runtime gate:

`M7.6 TUNING EXECUTION GATE: PASS`

Trạng thái:

`VERIFIED`

---

# 29. Phân tích tuning theo model family

## M7.6-F01 — Logistic Regression: C = 0.1 bị loại

Reference:

```text
C = 1.0

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

Candidate:

```text
C = 0.1

mean F1:
0.394491

std F1:
0.024877

mean Recall:
0.268594

mean Precision:
0.742800

FN:
1,413

FP:
179

alerts:
700
```

So với reference:

```text
F1 fold leads:
0 / 3

Δ mean F1:
-0.100611

Δ mean Recall:
-0.116932

Δ mean Precision:
+0.044505
```

Diễn giải:

`C = 0.1` tạo regularization mạnh hơn và giúp Precision/FP/alert burden, nhưng:

- primary F1 giảm rất lớn;
- Recall giảm;
- FN tăng;
- thua reference ở cả 3 folds.

Vì primary metric và fold consistency đều xấu hơn, candidate này không đủ evidence để giữ.

Quyết định:

`LR C=0.1 → REJECT`

Trạng thái:

`LOCKED`

---

## M7.6-F02 — Logistic Regression: C = 10 được chọn

Candidate:

```text
C = 10.0

mean F1:
0.524328

std F1:
0.029859

mean Recall:
0.426293

mean Precision:
0.687197

TP:
820

FP:
376

FN:
1,114

alerts:
1,196
```

So với reference `C = 1.0`:

```text
F1 fold leads:
3 / 3

Δ mean F1:
+0.029226

Δ std F1:
-0.008456

Δ mean Recall:
+0.040767

Δ mean Precision:
-0.011099

Δ TP:
+78

Δ FN:
-78

Δ FP:
+55

Δ alerts:
+133
```

Diễn giải:

`C = 10` có improvement tương đối nhất quán:

- mean F1 tăng;
- thắng F1 ở cả 3 folds;
- Recall tăng;
- FN giảm;
- std F1 còn thấp hơn reference.

Cost:

- Precision giảm nhẹ;
- FP và alerts tăng nhẹ.

Đây là trade-off có thể bảo vệ được vì primary metric, fold consistency và Recall đều cải thiện.

Quyết định:

```text
LR:
SELECT CONFIG

LR-T02-LBFGS-L2-C10
```

Trạng thái:

`LOCKED FOR M7.7`

---

## M7.6-F03 — Decision Tree: max_depth = 20 tốt hơn reference nhưng không phải candidate mạnh nhất

Reference:

```text
mean F1:
0.511964

std F1:
0.029242
```

Candidate:

```text
max_depth = 20

mean F1:
0.576917

std F1:
0.023881

mean Recall:
0.916548

mean Precision:
0.428401

TP:
1,759

FP:
2,427

FN:
175

alerts:
4,186
```

So với reference:

```text
F1 fold leads:
3 / 3

Δ mean F1:
+0.064953

Δ mean Recall:
+0.457607

Δ mean Precision:
-0.169166
```

Diễn giải:

Giới hạn độ sâu ở 20 tạo improvement F1 mạnh và tăng Recall rất lớn, nhưng Precision/FP/alert burden chịu cost đáng kể.

Candidate này là improvement thật so với reference, nhưng cần so tiếp với `min_samples_leaf = 5`.

Trạng thái:

`COMPETITIVE BUT NOT SELECTED`

---

## M7.6-F04 — Decision Tree: min_samples_leaf = 5 được chọn

Candidate:

```text
min_samples_leaf = 5

mean F1:
0.605570

std F1:
0.013102

mean Recall:
0.748473

mean Precision:
0.519011

TP:
1,436

FP:
1,376

FN:
498

alerts:
2,812
```

So với reference:

```text
F1 fold leads:
3 / 3

Δ mean F1:
+0.093606

Δ std F1:
-0.016140

Δ mean Recall:
+0.289532

Δ mean Precision:
-0.078556

Δ TP:
+558

Δ FN:
-558

Δ FP:
+764

Δ alerts:
+1,322
```

So với `max_depth = 20`:

```text
mean F1:
0.605570 > 0.576917

std F1:
0.013102 < 0.023881

mean Precision:
0.519011 > 0.428401

FP:
1,376 < 2,427

alerts:
2,812 < 4,186
```

`max_depth = 20` có Recall cao hơn và FN thấp hơn, nhưng `min_samples_leaf = 5` có:

- primary F1 cao hơn;
- stability tốt hơn;
- Precision tốt hơn;
- FP và alert burden thấp hơn đáng kể.

Theo metric contract hiện tại, đây là tuning candidate cân bằng hơn.

Quyết định:

```text
DT:
SELECT CONFIG

DT-T02-GINI-MINLEAF5-CW
```

Trạng thái:

`LOCKED FOR M7.7`

---

## M7.6-F05 — Random Forest: max_depth = 20 không vượt reference

Reference:

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

Candidate:

```text
max_depth = 20

mean F1:
0.594074

std F1:
0.029196

mean Recall:
0.916587

mean Precision:
0.448542

TP:
1,759

FP:
2,252

FN:
175

alerts:
4,011
```

So với reference:

```text
F1 fold leads:
1 / 2

Δ mean F1:
-0.027070

Δ std F1:
+0.008506

Δ mean Recall:
+0.232181

Δ mean Precision:
-0.129002
```

Diễn giải:

Candidate tăng Recall và giảm FN rất mạnh, nhưng:

- mean F1 giảm;
- stability xấu hơn;
- Precision giảm;
- FP/alerts tăng mạnh;
- chỉ thắng F1 ở 1/3 folds.

Không đủ evidence để thay reference.

Quyết định:

`RF max_depth=20 → REJECT`

Trạng thái:

`LOCKED`

---

## M7.6-F06 — Random Forest: min_samples_leaf = 2 vẫn không đủ để thay reference

Candidate:

```text
min_samples_leaf = 2

mean F1:
0.616423

std F1:
0.023966

mean Recall:
0.893718

mean Precision:
0.481403

TP:
1,714

FP:
1,930

FN:
220

alerts:
3,644
```

So với reference:

```text
F1 fold leads:
1 / 2

Δ mean F1:
-0.004721

Δ std F1:
+0.003276

Δ mean Recall:
+0.209311

Δ mean Precision:
-0.096140

Δ TP:
+399

Δ FN:
-399

Δ FP:
+951

Δ alerts:
+1,350
```

Diễn giải:

Candidate này gần reference hơn `max_depth=20` về F1 và tăng Recall lớn.

Tuy nhiên:

- mean F1 vẫn thấp hơn reference;
- thua reference 2/3 folds;
- std F1 xấu hơn;
- Precision giảm;
- FP/alert burden tăng đáng kể.

Không dùng arbitrary epsilon để coi mức giảm F1 là hòa.

M7.6 cũng chưa phải bước threshold tuning; Recall cao hơn không đủ để override primary F1 + stability + Precision/FP evidence.

Quyết định:

`RF min_samples_leaf=2 → REJECT`

Trạng thái:

`LOCKED`

---

## M7.6-F07 — Random Forest giữ reference configuration

Cả hai tuning candidates đều không vượt reference trên primary aggregate F1:

```text
reference:
0.621144

max_depth=20:
0.594074

min_samples_leaf=2:
0.616423
```

Reference cũng có:

```text
lowest std F1 trong 3 RF configs:
0.020690

higher Precision

lower FP

lower alert burden
```

Do đó tuning không tạo configuration tốt hơn reference theo evidence bundle hiện tại.

Quyết định:

```text
RF:
KEEP REFERENCE CONFIG

RF-REF-100-GINI-SQRT-UNPRUNED-CW
```

Trạng thái:

`LOCKED FOR M7.7`

---

## M7.6-F08 — Tuning outcome theo family

Reviewed configuration map:

```text
LR:
LR-T02-LBFGS-L2-C10

DT:
DT-T02-GINI-MINLEAF5-CW

RF:
RF-REF-100-GINI-SQRT-UNPRUNED-CW
```

Tức là:

```text
LR:
SELECT CONFIG

DT:
SELECT CONFIG

RF:
KEEP REFERENCE CONFIG
```

Đây là family-specific configuration selection.

Nó chưa phải final model-family selection.

Trạng thái:

`LOCKED FOR M7.7 INPUT`

---

## M7.6-F09 — Tuning có giá trị nhưng không bắt buộc phải thắng ở mọi family

Observed:

```text
LR:
tuning improvement exists

DT:
tuning improvement exists

RF:
reference remains better
```

Điều này phù hợp với protocol:

`tuning outcome có thể KEEP REFERENCE CONFIG`.

Không có yêu cầu tuning phải luôn tìm ra score cao hơn.

Trạng thái:

`PROTOCOL-CONSISTENT`

---

## M7.6-F10 — Failed candidate / warning audit

Observed:

```text
Expected new fold runs:
18

Completed:
18

Failed:
0

Warnings:
0
```

Không có candidate bị loại âm thầm vì lỗi runtime.

Trạng thái:

`VERIFIED`

---

## M7.6-F11 — Search space không được mở rộng sau kết quả

Observed notebook search:

```text
6 new configs
18 new fold fits
```

Không có candidate mới được thêm sau khi xem kết quả.

Decision:

`NO ADAPTIVE GRID EXPANSION`

Trạng thái:

`VERIFIED`

---

## M7.6-F12 — Selection boundary sạch

Observed:

```text
External VALIDATION scoring:
False

Imbalance strategy changed:
False

Threshold optimization:
False

Random seed tuned:
False

FINAL TEST accessed:
False
```

M7.6 chỉ chọn configuration bên trong từng family.

Model-family winner tiếp tục:

`OPEN`.

Trạng thái:

`VERIFIED`

---

## M7.6-F13 — M7.7 readiness

M7.7 nhận ba upstream candidates đã được tuning review:

```text
LR:
W_SHORT
NONE
C = 10

DT:
W_SHORT
CLASS_WEIGHT_BALANCED
min_samples_leaf = 5

RF:
W_SHORT
CLASS_WEIGHT_BALANCED
reference RF config
```

M7.7 mới có authority để:

- đối chiếu temporal-CV behavior;
- dùng external VALIDATION để development confirmation;
- đối chiếu M6 baseline/error evidence;
- chọn một upstream development candidate.

Threshold vẫn chưa được tune.

FINAL TEST vẫn được bảo vệ.

Trạng thái:

`READY FOR M7.7`

# 30. Decision Log M7.6 — sau runtime review

## M7.6-D01 — Tuning scope

Decision:

`MODERATE / PREDECLARED / TEMPORAL-CV-BASED`

Status:

`VERIFIED — LOCKED`

---

## M7.6-D02 — Training window

Decision:

`W_SHORT`

Status:

`INHERITED — VERIFIED — LOCKED`

---

## M7.6-D03 — Candidate-specific imbalance

Decision:

```text
LR:
NONE

DT:
CLASS_WEIGHT_BALANCED

RF:
CLASS_WEIGHT_BALANCED
```

Status:

`INHERITED FROM M7.5 — VERIFIED — LOCKED`

---

## M7.6-D04 — Logistic Regression search

Executed:

```text
C = 1.0 reference
C = 0.1
C = 10.0
```

No post-result expansion.

Status:

`VERIFIED`

---

## M7.6-D05 — Logistic Regression tuning decision

Decision:

```text
SELECT CONFIG

LR-T02-LBFGS-L2-C10
```

Key evidence:

```text
mean F1:
0.524328

F1 fold leads vs reference:
3 / 3

Δ mean F1:
+0.029226

Δ Recall:
+0.040767

Δ Precision:
-0.011099

std F1:
0.029859
<
reference 0.038315
```

Status:

`LOCKED FOR M7.7`

---

## M7.6-D06 — Decision Tree search

Executed:

```text
reference

max_depth = 20

min_samples_leaf = 5
```

No post-result expansion.

Status:

`VERIFIED`

---

## M7.6-D07 — Decision Tree tuning decision

Decision:

```text
SELECT CONFIG

DT-T02-GINI-MINLEAF5-CW
```

Key evidence:

```text
mean F1:
0.605570

F1 fold leads vs reference:
3 / 3

Δ mean F1:
+0.093606

std F1:
0.013102

Δ Recall:
+0.289532

Δ Precision:
-0.078556
```

`min_samples_leaf=5` cũng có F1, stability, Precision và alert burden tốt hơn candidate `max_depth=20`.

Status:

`LOCKED FOR M7.7`

---

## M7.6-D08 — Random Forest search

Executed:

```text
reference

max_depth = 20

min_samples_leaf = 2
```

No post-result expansion.

Status:

`VERIFIED`

---

## M7.6-D09 — Random Forest tuning decision

Decision:

```text
KEEP REFERENCE CONFIG

RF-REF-100-GINI-SQRT-UNPRUNED-CW
```

Reason:

```text
Reference mean F1:
0.621144

max_depth=20:
0.594074

min_samples_leaf=2:
0.616423
```

Both tuning candidates:

- fail to exceed reference mean F1;
- lose F1 in 2/3 folds;
- have worse std F1;
- reduce Precision;
- increase FP / alerts.

Status:

`LOCKED FOR M7.7`

---

## M7.6-D10 — Reviewed family-specific configuration map

Decision:

```text
LR:
LR-T02-LBFGS-L2-C10

DT:
DT-T02-GINI-MINLEAF5-CW

RF:
RF-REF-100-GINI-SQRT-UNPRUNED-CW
```

Status:

`LOCKED FOR M7.7 INPUT`

---

## M7.6-D11 — Adaptive grid expansion

Decision:

`DO NOT EXPAND`

Reason:

Current M7.6 search đã hoàn thành đúng declared budget.

Status:

`LOCKED`

---

## M7.6-D12 — Failed / invalid candidates

Observed:

```text
Failed fold runs:
0

Warnings:
0
```

Decision:

No failed candidate requires remediation.

Status:

`VERIFIED`

---

## M7.6-D13 — Random seed

Decision:

`42`

Not tuned.

Status:

`VERIFIED — LOCKED`

---

## M7.6-D14 — Threshold

Decision:

`DEFAULT_MODEL_DECISION_RULE`

No numerical threshold tuning in M7.6.

Status:

`LOCKED`

---

## M7.6-D15 — Model-family winner

Decision:

Not selected in M7.6.

Status:

`OPEN`

---

## M7.6-D16 — External VALIDATION

Decision:

No scoring in M7.6.

Status:

`VERIFIED — LOCKED`

---

## M7.6-D17 — FINAL TEST

Decision:

No access.

Status:

`INHERITED — VERIFIED — LOCKED`

---

## M7.6-D18 — M7.7 handoff

Decision:

Proceed to:

`M7.7 — Candidate selection + external VALIDATION confirmation`

with:

```text
LR candidate:
W_SHORT
NONE
C = 10

DT candidate:
W_SHORT
CLASS_WEIGHT_BALANCED
min_samples_leaf = 5

RF candidate:
W_SHORT
CLASS_WEIGHT_BALANCED
reference RF hyperparameters
```

Status:

`READY`

# 31. M7.6 Gate

## Technical runtime gates

```text
G01_SOURCE_LOCATION                         → PASS
G02_UPSTREAM_HANDOFF                        → PASS
G03_PRE_RUNTIME_SEARCH_SPACE                → PASS
G04_SEARCH_SPACE_ISOLATION                  → PASS
G05_FEATURE_CONTRACT                        → PASS
G06_STRICT_CAUSAL_REGRESSION                → PASS
G07_SEMANTIC_TRAIN_LINEAGE                  → PASS
G08_FOLD_INTEGRITY                          → PASS
G09_PREPROCESSING_STATE_REUSE               → PASS
G10_MATRIX_BUILDER                          → PASS
G11_REFERENCE_EVIDENCE                      → PASS
G12_GENERIC_TUNING_RUNNER                   → PASS
G13_TUNING_EXECUTION_ACCOUNTING             → PASS
G14_FOLD_REGISTRY                           → PASS
G15_AGGREGATION                             → PASS
G16_REFERENCE_COMPARISON                    → PASS
G17_NO_AUTO_SELECTION                       → PASS
G18_PERSISTENCE                             → PASS
G19_ROUND_TRIP                              → PASS
G20_SELECTION_BOUNDARY                      → PASS
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

## R02 — Upstream candidate identity preserved?

Evidence:

```text
LR:
W_SHORT / NONE

DT:
W_SHORT / CLASS_WEIGHT_BALANCED

RF:
W_SHORT / CLASS_WEIGHT_BALANCED
```

Result:

`PASS`

---

## R03 — Search space frozen before runtime?

Evidence:

```text
3 reference configs
6 new configs
18 expected new fits
```

No adaptive expansion.

Result:

`PASS`

---

## R04 — Single-factor tuning isolation valid?

Evidence:

```text
LR:
C only

DT:
max_depth or min_samples_leaf only

RF:
max_depth or min_samples_leaf only
```

Result:

`PASS`

---

## R05 — Fold/preprocessing integrity valid?

Evidence:

```text
Q2 / Q3 / Q4 2018
W_SHORT
fold-safe preprocessing
47 features
```

Result:

`PASS`

---

## R06 — All tuning runs accounted?

Evidence:

```text
18 / 18 completed

failed:
0

warnings:
0
```

Result:

`PASS`

---

## R07 — LR C=0.1 reviewed?

Decision:

`REJECT`

Evidence:

```text
Δ mean F1:
-0.100611

F1 fold leads:
0 / 3
```

Result:

`PASS`

---

## R08 — LR C=10 reviewed?

Decision:

`SELECT CONFIG`

Evidence:

```text
mean F1:
0.524328

F1 fold leads:
3 / 3

Δ mean F1:
+0.029226

std F1:
improves
```

Result:

`PASS`

---

## R09 — DT max_depth=20 reviewed?

Decision:

`NOT SELECTED`

Evidence:

Improves reference but remains weaker overall than `min_samples_leaf=5` on primary/stability/Precision-operational balance.

Result:

`PASS`

---

## R10 — DT min_samples_leaf=5 reviewed?

Decision:

`SELECT CONFIG`

Evidence:

```text
mean F1:
0.605570

F1 fold leads:
3 / 3

Δ mean F1:
+0.093606

std F1:
0.013102
```

Result:

`PASS`

---

## R11 — RF max_depth=20 reviewed?

Decision:

`REJECT`

Evidence:

```text
Δ mean F1:
-0.027070

F1 fold leads candidate/reference:
1 / 2
```

Result:

`PASS`

---

## R12 — RF min_samples_leaf=2 reviewed?

Decision:

`REJECT`

Evidence:

```text
Δ mean F1:
-0.004721

F1 fold leads candidate/reference:
1 / 2

std F1:
worse

Precision/FP/alerts:
worse
```

Result:

`PASS`

---

## R13 — RF reference retained?

Decision:

`KEEP REFERENCE CONFIG`

Result:

`PASS`

---

## R14 — Family-specific configuration map resolved?

Decision:

```text
LR:
LR-T02-LBFGS-L2-C10

DT:
DT-T02-GINI-MINLEAF5-CW

RF:
RF-REF-100-GINI-SQRT-UNPRUNED-CW
```

Result:

`PASS`

---

## R15 — Failed/warning candidates resolved?

Evidence:

```text
failed:
0

warnings:
0
```

Result:

`PASS`

---

## R16 — No post-result grid expansion?

Decision:

`NO EXPANSION`

Result:

`PASS`

---

## R17 — Random seed boundary preserved?

Evidence:

`random_seed_tuned = False`

Result:

`PASS`

---

## R18 — Threshold boundary preserved?

Evidence:

`threshold_optimization_performed = False`

Result:

`PASS`

---

## R19 — External VALIDATION boundary preserved?

Evidence:

`external_validation_scoring_performed = False`

Result:

`PASS`

---

## R20 — Model-family selection boundary preserved?

Evidence:

```text
Model-family Winner:
OPEN

Final Model:
OPEN
```

Result:

`PASS`

---

## R21 — Persistence / round-trip valid?

Evidence:

```text
result registry:
PASS

manifest:
PASS

prediction/risk-score artifacts:
fingerprinted
```

Result:

`PASS`

---

## R22 — FINAL TEST protected?

Evidence:

`final_test_accessed = False`

Result:

`PASS`

---

## R23 — M7.7 handoff valid?

Required:

- family-specific configs resolved;
- model-family winner still OPEN;
- external VALIDATION untouched in M7.6;
- threshold still untuned;
- FINAL TEST protected.

Observed:

`PASS`

---

## Overall M7.6 Gate

```text
Technical gates:
20 / 20 PASS

Runtime review gates:
23 / 23 PASS

Blocking issue:
NONE
```

Final:

`M7.6 — PASS`

Tuned configuration map:

```text
LR:
LR-T02-LBFGS-L2-C10

DT:
DT-T02-GINI-MINLEAF5-CW

RF:
RF-REF-100-GINI-SQRT-UNPRUNED-CW
```

Model-family winner:

`OPEN`

Handoff:

`READY FOR M7.7`

# 32. Kết luận M7.6

M7.6 đã hoàn thành moderate hyperparameter tuning theo đúng search policy:

```text
SMALL
JUSTIFIED
DECLARED BEFORE RUN
TEMPORAL-CV-BASED
```

Quy mô thực nghiệm:

```text
3 reference configs

6 tuning configs mới

18 / 18 tuning fold runs hoàn thành

failed runs:
0

warnings:
0
```

Search space không được mở rộng sau khi xem kết quả.

Random seed không được dùng làm tuning dimension.

Imbalance strategy không bị thay đổi.

Threshold không được tune.

External VALIDATION chưa được dùng.

FINAL TEST chưa được truy cập.

Reviewed tuning decisions:

```text
Logistic Regression

SELECT:
LR-T02-LBFGS-L2-C10

W_SHORT
imbalance = NONE
C = 10
```

```text
Decision Tree

SELECT:
DT-T02-GINI-MINLEAF5-CW

W_SHORT
imbalance = CLASS_WEIGHT_BALANCED
min_samples_leaf = 5
```

```text
Random Forest

KEEP REFERENCE:
RF-REF-100-GINI-SQRT-UNPRUNED-CW

W_SHORT
imbalance = CLASS_WEIGHT_BALANCED
n_estimators = 100
max_depth = None
min_samples_leaf = 1
max_features = sqrt
```

Tuning outcome cho thấy:

```text
LR:
tuning tạo improvement đủ rõ

DT:
tuning tạo improvement đủ rõ

RF:
không cần thay reference config
```

Điều này phù hợp với protocol vì tuning không bắt buộc phải thắng baseline/reference ở mọi family.

Final M7.6 state:

```text
M7.6 — PASS

Training Window:
W_SHORT — LOCKED

LR Config:
LR-T02-LBFGS-L2-C10 — LOCKED

DT Config:
DT-T02-GINI-MINLEAF5-CW — LOCKED

RF Config:
RF-REF-100-GINI-SQRT-UNPRUNED-CW — LOCKED

Model-family Winner:
OPEN

Final Model:
OPEN

Final Threshold:
OPEN

External VALIDATION:
NOT USED

FINAL TEST:
PROTECTED

Blocking Issue:
NONE

READY FOR M7.7
```

Bước tiếp theo:

`M7.7 — Candidate selection + external VALIDATION confirmation`

M7.7 phải mang đúng ba candidate identities đã khóa:

```text
LR:
W_SHORT
NONE
C = 10

DT:
W_SHORT
CLASS_WEIGHT_BALANCED
min_samples_leaf = 5

RF:
W_SHORT
CLASS_WEIGHT_BALANCED
reference RF config
```

M7.7 sẽ đối chiếu:

```text
temporal-CV evidence
external VALIDATION confirmation
M6 baseline/error evidence
computational feasibility
```

để chọn một upstream development candidate trước khi bước sang threshold selection.

Threshold vẫn:

`OPEN`

FINAL TEST tiếp tục:

`PROTECTED`
