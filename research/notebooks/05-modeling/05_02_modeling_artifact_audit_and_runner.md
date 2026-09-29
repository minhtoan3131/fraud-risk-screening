# M5.2 — Audit modeling artifacts và khóa shared experiment runner

Milestone:

`M5 — Modeling baseline`

Substep:

`M5.2 — Artifact Audit + Shared Baseline Runner`

Mục tiêu:

1. xác minh lại persisted modeling artifacts được handoff từ M4.7;
2. phát hiện stale/corrupt artifact trước khi fit model;
3. khóa một metric implementation dùng chung;
4. khóa một shared experiment runner dùng chung cho M5.3–M5.5;
5. khóa experiment-result schema;
6. bảo vệ FINAL TEST khỏi M5 development code path.

Notebook này **không fit Logistic Regression / Decision Tree / Random Forest trên dữ liệu project**.

Model baseline thật bắt đầu từ M5.3.

## 1. Contract kế thừa

M5.2 kế thừa trực tiếp:

- `CANON-M5.1 — Khóa Modeling Charter và baseline protocol`;
- `CANON-Kế hoạch Milestone 5 — Modeling baseline`;
- `CANON-M4.8`;
- `CANON-M4.7`;
- Experiment Specification từ M3.

Các invariant chính:

```text
Feature schema:
47-column CSR float32

Target:
int8
fraud = 1

Training windows:
W_LONG
W_SHORT

Validation:
2019-01 → 2019-05

Imbalance strategy:
NONE

Primary metric:
F1_fraud

Secondary:
Recall_fraud
Precision_fraud

Mandatory:
TP / FP / FN / TN
Confusion Matrix

Operational:
predicted-positive count/rate

Accuracy:
reference only

Probability/risk score:
preserve when available

FINAL TEST:
NO ACCESS
```

## 2. Runtime protocol

M5.2 là công việc runtime-dependent.

Quy trình:

```text
locate M4.7 artifacts
        ↓
file / manifest audit
        ↓
load matrices / target / lineage
        ↓
shape / dtype / finite / sparsity audit
        ↓
target / lineage / feature-name audit
        ↓
stale-artifact signature audit
        ↓
FINAL TEST isolation gate
        ↓
canonical metric implementation
        ↓
shared runner implementation
        ↓
toy infrastructure self-test
        ↓
persist M5.2 contract metadata
        ↓
M5.2 Gate
```

Nếu một assertion fail:

`STOP — không chuyển sang M5.3`.


```python

from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import json
import platform
import sys
import tempfile
import time
import warnings

import numpy as np
import pandas as pd

try:
    from scipy import sparse
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Thiếu scipy. Chạy `%pip install scipy scikit-learn`, "
        "Restart Kernel rồi Run All."
    ) from exc

try:
    import sklearn
    from sklearn.base import BaseEstimator, ClassifierMixin, clone
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Thiếu scikit-learn. Chạy `%pip install scipy scikit-learn`, "
        "Restart Kernel rồi Run All."
    ) from exc


print("Python:")
print(sys.version)

print("\nExecutable:")
print(sys.executable)

print("\nPlatform:")
print(platform.platform())

print("\nNumPy:")
print(np.__version__)

print("\nPandas:")
print(pd.__version__)

print("\nSciPy sparse available:")
print(True)

print("\nscikit-learn:")
print(sklearn.__version__)

```

    Python:
    3.14.6 (main, Jun 10 2026, 10:03:53) [Clang 21.0.0 (clang-2100.0.123.102)]
    
    Executable:
    /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/.venv/bin/python
    
    Platform:
    macOS-26.6.2-arm64-arm-64bit-Mach-O
    
    NumPy:
    2.5.3
    
    Pandas:
    3.0.5
    
    SciPy sparse available:
    True
    
    scikit-learn:
    1.9.1


## 3. Locate canonical M4.7 artifact directory

M5.2 không cần raw CSV.

Notebook chỉ tìm:

`data/processed/m4_07_baseline_ready`

Nếu không tìm thấy `manifest.json` trong canonical directory:

`FAIL LOUDLY`.


```python

ARTIFACT_RELATIVE_DIR = (
    Path("data")
    / "processed"
    / "m4_07_baseline_ready"
)

M5_02_RELATIVE_DIR = (
    Path("data")
    / "processed"
    / "m5_02_modeling_contract"
)

candidate_roots = [
    Path.cwd(),
    *list(Path.cwd().parents)[:5],
]

PROJECT_ROOT = None

for candidate in candidate_roots:
    candidate = candidate.resolve()

    if (
        candidate
        / ARTIFACT_RELATIVE_DIR
        / "manifest.json"
    ).exists():
        PROJECT_ROOT = candidate
        break

if PROJECT_ROOT is None:
    raise FileNotFoundError(
        "Không xác định được PROJECT_ROOT có "
        "data/processed/m4_07_baseline_ready/manifest.json"
    )

ARTIFACT_DIR = (
    PROJECT_ROOT
    / ARTIFACT_RELATIVE_DIR
)

M5_02_OUTPUT_DIR = (
    PROJECT_ROOT
    / M5_02_RELATIVE_DIR
)

M5_02_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

print("PROJECT_ROOT:")
print(PROJECT_ROOT)

print("\nARTIFACT_DIR:")
print(ARTIFACT_DIR)

print("\nM5_02_OUTPUT_DIR:")
print(M5_02_OUTPUT_DIR)

```

    PROJECT_ROOT:
    /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening
    
    ARTIFACT_DIR:
    /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/processed/m4_07_baseline_ready
    
    M5_02_OUTPUT_DIR:
    /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/processed/m5_02_modeling_contract


## 4. Expected canonical artifact signature

Các số dưới đây đến từ M4.7 runtime-reviewed artifacts.

M5.2 dùng chúng như integrity signature.

Đặc biệt, `nnz` giúp phát hiện stale artifact trước scaler fix.


```python

M4_PIPELINE_VERSION = "M4.7-baseline-v1"
M5_RUNNER_VERSION = "M5.2-shared-runner-v1"

EXPECTED_OUTPUT_WIDTH = 47

EXPECTED_W_LONG_ROWS = 6_855_270
EXPECTED_W_SHORT_ROWS = 1_721_615
EXPECTED_VALIDATION_ROWS = 712_458

EXPECTED_W_LONG_FRAUD = 9_606
EXPECTED_W_SHORT_FRAUD = 2_491
EXPECTED_VALIDATION_FRAUD = 1_052

EXPECTED_NNZ = {
    "X_train_w_long": 61_900_193,
    "X_train_w_short": 15_542_381,
    "X_validation_w_long": 6_430_339,
    "X_validation_w_short": 6_430_339,
}

EXPECTED_SHAPES = {
    "X_train_w_long": (
        EXPECTED_W_LONG_ROWS,
        EXPECTED_OUTPUT_WIDTH,
    ),
    "X_train_w_short": (
        EXPECTED_W_SHORT_ROWS,
        EXPECTED_OUTPUT_WIDTH,
    ),
    "X_validation_w_long": (
        EXPECTED_VALIDATION_ROWS,
        EXPECTED_OUTPUT_WIDTH,
    ),
    "X_validation_w_short": (
        EXPECTED_VALIDATION_ROWS,
        EXPECTED_OUTPUT_WIDTH,
    ),
}

IMBALANCE_STRATEGY = "NONE"
DEFAULT_RANDOM_STATE = 42

BASELINE_THRESHOLD_POLICY = (
    "DEFAULT_MODEL_DECISION_RULE"
)

FINAL_TEST_ACCESS_ALLOWED = False

print("M4 pipeline version:")
print(M4_PIPELINE_VERSION)

print("\nM5 runner version:")
print(M5_RUNNER_VERSION)

print("\nImbalance strategy:")
print(IMBALANCE_STRATEGY)

print("\nBaseline threshold policy:")
print(BASELINE_THRESHOLD_POLICY)

print("\nFINAL TEST access allowed:")
print(FINAL_TEST_ACCESS_ALLOWED)

```

    M4 pipeline version:
    M4.7-baseline-v1
    
    M5 runner version:
    M5.2-shared-runner-v1
    
    Imbalance strategy:
    NONE
    
    Baseline threshold policy:
    DEFAULT_MODEL_DECISION_RULE
    
    FINAL TEST access allowed:
    False


## 5. Required-file gate


```python

artifact_paths = {
    "X_train_w_long":
        ARTIFACT_DIR
        / "X_train_w_long.npz",

    "X_train_w_short":
        ARTIFACT_DIR
        / "X_train_w_short.npz",

    "X_validation_w_long":
        ARTIFACT_DIR
        / "X_validation_w_long.npz",

    "X_validation_w_short":
        ARTIFACT_DIR
        / "X_validation_w_short.npz",

    "y_train_w_long":
        ARTIFACT_DIR
        / "y_train_w_long.npy",

    "y_train_w_short":
        ARTIFACT_DIR
        / "y_train_w_short.npy",

    "y_validation":
        ARTIFACT_DIR
        / "y_validation.npy",

    "row_id_train_w_long":
        ARTIFACT_DIR
        / "row_id_train_w_long.npy",

    "row_id_train_w_short":
        ARTIFACT_DIR
        / "row_id_train_w_short.npy",

    "row_id_validation":
        ARTIFACT_DIR
        / "row_id_validation.npy",

    "feature_names":
        ARTIFACT_DIR
        / "feature_names.json",

    "manifest":
        ARTIFACT_DIR
        / "manifest.json",
}


file_audit_rows = []

for name, path in artifact_paths.items():
    exists = path.exists()
    size_bytes = (
        path.stat().st_size
        if exists
        else 0
    )

    file_audit_rows.append(
        {
            "artifact": name,
            "exists": exists,
            "size_bytes": size_bytes,
            "filename": path.name,
        }
    )

    assert exists, (
        f"Thiếu artifact: {path}"
    )

    assert size_bytes > 0, (
        f"Artifact rỗng: {path}"
    )


file_audit_df = pd.DataFrame(
    file_audit_rows
)

print(
    file_audit_df.to_string(
        index=False
    )
)

print(
    "\nM5.2 REQUIRED-FILE GATE: PASS"
)

```

                artifact  exists  size_bytes                 filename
          X_train_w_long    True   105780141       X_train_w_long.npz
         X_train_w_short    True    26875005      X_train_w_short.npz
     X_validation_w_long    True    11281966  X_validation_w_long.npz
    X_validation_w_short    True    11270182 X_validation_w_short.npz
          y_train_w_long    True     6855398       y_train_w_long.npy
         y_train_w_short    True     1721743      y_train_w_short.npy
            y_validation    True      712586         y_validation.npy
     row_id_train_w_long    True    54842288  row_id_train_w_long.npy
    row_id_train_w_short    True    13773048 row_id_train_w_short.npy
       row_id_validation    True     5699792    row_id_validation.npy
           feature_names    True        1380       feature_names.json
                manifest    True         795            manifest.json
    
    M5.2 REQUIRED-FILE GATE: PASS


## 6. Manifest + feature-name contract


```python

with open(
    artifact_paths["manifest"],
    "r",
    encoding="utf-8",
) as file:
    manifest = json.load(file)


with open(
    artifact_paths["feature_names"],
    "r",
    encoding="utf-8",
) as file:
    feature_names = json.load(file)


assert (
    manifest["pipeline_version"]
    == M4_PIPELINE_VERSION
)

assert (
    manifest["output_width"]
    == EXPECTED_OUTPUT_WIDTH
)

assert (
    manifest["matrix_dtype"]
    == "float32"
)

assert (
    str(
        manifest["matrix_format"]
    ).upper()
    == "CSR"
)

assert (
    manifest["target_dtype"]
    == "int8"
)

assert (
    manifest["feature_names_file"]
    == artifact_paths[
        "feature_names"
    ].name
)

assert (
    manifest["W_LONG"]["train_rows"]
    == EXPECTED_W_LONG_ROWS
)

assert (
    manifest["W_SHORT"]["train_rows"]
    == EXPECTED_W_SHORT_ROWS
)

assert (
    manifest["VALIDATION"]["rows"]
    == EXPECTED_VALIDATION_ROWS
)

assert (
    manifest["W_LONG"]["train_fraud"]
    == EXPECTED_W_LONG_FRAUD
)

assert (
    manifest["W_SHORT"]["train_fraud"]
    == EXPECTED_W_SHORT_FRAUD
)

assert (
    manifest["VALIDATION"]["fraud"]
    == EXPECTED_VALIDATION_FRAUD
)

assert (
    manifest["raw_identifiers_in_X"]
    is False
)

assert (
    manifest["target_in_X"]
    is False
)

assert (
    manifest["final_test_used"]
    is False
)


assert isinstance(
    feature_names,
    list,
)

assert (
    len(feature_names)
    == EXPECTED_OUTPUT_WIDTH
)

assert (
    len(set(feature_names))
    == EXPECTED_OUTPUT_WIDTH
)


print(
    json.dumps(
        manifest,
        ensure_ascii=False,
        indent=2,
    )
)

print(
    "\nFeature count:",
    len(feature_names),
)

print(
    "\nFirst 10 features:"
)

for name in feature_names[:10]:
    print(name)


print(
    "\nM5.2 MANIFEST / FEATURE-NAME GATE: PASS"
)

```

    {
      "pipeline_version": "M4.7-baseline-v1",
      "output_width": 47,
      "matrix_dtype": "float32",
      "matrix_format": "CSR",
      "target_dtype": "int8",
      "feature_names_file": "feature_names.json",
      "W_LONG": {
        "train_rows": 6855270,
        "validation_rows": 712458,
        "train_fraud": 9606,
        "fit_start": "2015-01-01 00:01:00",
        "fit_end_observed": "2018-12-31 23:58:00"
      },
      "W_SHORT": {
        "train_rows": 1721615,
        "validation_rows": 712458,
        "train_fraud": 2491,
        "fit_start": "2018-01-01 00:03:00",
        "fit_end_observed": "2018-12-31 23:58:00"
      },
      "VALIDATION": {
        "rows": 712458,
        "fraud": 1052,
        "start": "2019-01-01 00:02:00",
        "end_observed": "2019-05-31 23:58:00"
      },
      "raw_identifiers_in_X": false,
      "target_in_X": false,
      "final_test_used": false
    }
    
    Feature count: 47
    
    First 10 features:
    num__amount_numeric
    num__time_since_previous_transaction_min
    num__transactions_last_1h
    num__amount_minus_previous_mean
    bool__is_new_merchant
    bool__has_prior_card_history
    cat__transaction_mode_Chip Transaction
    cat__transaction_mode_Online Transaction
    cat__transaction_mode_Swipe Transaction
    cat__transaction_mode___UNKNOWN__
    
    M5.2 MANIFEST / FEATURE-NAME GATE: PASS


## 7. Load canonical modeling artifacts

Bước này có thể dùng đáng kể RAM vì W_LONG matrix có hơn 6.8 triệu rows.

Không convert sparse matrix sang dense.


```python

load_start = time.perf_counter()

X_train_w_long = sparse.load_npz(
    artifact_paths[
        "X_train_w_long"
    ]
)

X_train_w_short = sparse.load_npz(
    artifact_paths[
        "X_train_w_short"
    ]
)

X_validation_w_long = sparse.load_npz(
    artifact_paths[
        "X_validation_w_long"
    ]
)

X_validation_w_short = sparse.load_npz(
    artifact_paths[
        "X_validation_w_short"
    ]
)


y_train_w_long = np.load(
    artifact_paths[
        "y_train_w_long"
    ],
    allow_pickle=False,
)

y_train_w_short = np.load(
    artifact_paths[
        "y_train_w_short"
    ],
    allow_pickle=False,
)

y_validation = np.load(
    artifact_paths[
        "y_validation"
    ],
    allow_pickle=False,
)


row_id_train_w_long = np.load(
    artifact_paths[
        "row_id_train_w_long"
    ],
    allow_pickle=False,
)

row_id_train_w_short = np.load(
    artifact_paths[
        "row_id_train_w_short"
    ],
    allow_pickle=False,
)

row_id_validation = np.load(
    artifact_paths[
        "row_id_validation"
    ],
    allow_pickle=False,
)


load_elapsed = (
    time.perf_counter()
    - load_start
)


print(
    "Artifact load seconds:",
    round(
        load_elapsed,
        2,
    ),
)

print(
    "\nM5.2 ARTIFACT LOAD: COMPLETE"
)

```

    Artifact load seconds: 0.8
    
    M5.2 ARTIFACT LOAD: COMPLETE


## 8. Sparse-matrix integrity + stale-artifact detection

Current canonical artifacts sau scaler fix có exact `nnz` signature đã được M4.7 review.

Nếu `nnz` khác:

`STOP`

vì có thể đang load stale/corrupt/re-generated artifact khác contract.


```python

def csr_memory_bytes(
    matrix,
):
    return (
        matrix.data.nbytes
        + matrix.indices.nbytes
        + matrix.indptr.nbytes
    )


matrices = {
    "X_train_w_long":
        X_train_w_long,

    "X_train_w_short":
        X_train_w_short,

    "X_validation_w_long":
        X_validation_w_long,

    "X_validation_w_short":
        X_validation_w_short,
}


matrix_audit_rows = []


for name, matrix in matrices.items():
    assert sparse.isspmatrix_csr(
        matrix
    ), (
        f"{name}: không phải CSR."
    )

    assert (
        matrix.dtype
        == np.float32
    ), (
        f"{name}: dtype != float32."
    )

    assert (
        matrix.shape
        == EXPECTED_SHAPES[
            name
        ]
    ), (
        f"{name}: shape mismatch "
        f"{matrix.shape}"
    )

    assert np.isfinite(
        matrix.data
    ).all(), (
        f"{name}: có NaN/inf."
    )

    assert (
        matrix.nnz
        == EXPECTED_NNZ[
            name
        ]
    ), (
        f"{name}: nnz mismatch. "
        "Có thể là stale artifact "
        "trước M4.7 scaler fix."
    )

    density_pct = (
        matrix.nnz
        /
        (
            matrix.shape[0]
            *
            matrix.shape[1]
        )
        * 100
    )

    matrix_audit_rows.append(
        {
            "matrix": name,
            "rows": matrix.shape[0],
            "columns": matrix.shape[1],
            "dtype": str(
                matrix.dtype
            ),
            "nnz": matrix.nnz,
            "density_pct":
                density_pct,
            "csr_memory_mb":
                csr_memory_bytes(
                    matrix
                )
                / 1024**2,
        }
    )


matrix_audit_df = pd.DataFrame(
    matrix_audit_rows
)


print(
    matrix_audit_df.to_string(
        index=False
    )
)


print(
    "\nM5.2 MATRIX / STALE-ARTIFACT GATE: PASS"
)

```

                  matrix    rows  columns   dtype      nnz  density_pct  csr_memory_mb
          X_train_w_long 6855270       47 float32 61900193    19.211867     498.411777
         X_train_w_short 1721615       47 float32 15542381    19.208067     125.146400
     X_validation_w_long  712458       47 float32  6430339    19.203339      51.777409
    X_validation_w_short  712458       47 float32  6430339    19.203339      51.777409
    
    M5.2 MATRIX / STALE-ARTIFACT GATE: PASS


## 9. Target integrity


```python

target_specs = {
    "y_train_w_long": {
        "array":
            y_train_w_long,
        "rows":
            EXPECTED_W_LONG_ROWS,
        "fraud":
            EXPECTED_W_LONG_FRAUD,
    },

    "y_train_w_short": {
        "array":
            y_train_w_short,
        "rows":
            EXPECTED_W_SHORT_ROWS,
        "fraud":
            EXPECTED_W_SHORT_FRAUD,
    },

    "y_validation": {
        "array":
            y_validation,
        "rows":
            EXPECTED_VALIDATION_ROWS,
        "fraud":
            EXPECTED_VALIDATION_FRAUD,
    },
}


target_audit_rows = []


for name, spec in (
    target_specs.items()
):
    array = spec["array"]

    assert (
        array.dtype
        == np.int8
    ), (
        f"{name}: dtype != int8"
    )

    assert (
        len(array)
        == spec["rows"]
    ), (
        f"{name}: row mismatch"
    )

    unique_values = (
        set(
            np.unique(
                array
            ).tolist()
        )
    )

    assert (
        unique_values
        <= {0, 1}
    ), (
        f"{name}: unexpected labels "
        f"{unique_values}"
    )

    fraud_count = int(
        array.sum()
    )

    assert (
        fraud_count
        == spec["fraud"]
    ), (
        f"{name}: fraud count mismatch"
    )

    target_audit_rows.append(
        {
            "target": name,
            "rows": len(array),
            "fraud": fraud_count,
            "non_fraud":
                len(array)
                - fraud_count,
            "dtype":
                str(array.dtype),
        }
    )


target_audit_df = pd.DataFrame(
    target_audit_rows
)


print(
    target_audit_df.to_string(
        index=False
    )
)


print(
    "\nM5.2 TARGET INTEGRITY GATE: PASS"
)

```

             target    rows  fraud  non_fraud dtype
     y_train_w_long 6855270   9606    6845664  int8
    y_train_w_short 1721615   2491    1719124  int8
       y_validation  712458   1052     711406  int8
    
    M5.2 TARGET INTEGRITY GATE: PASS


## 10. X/y alignment


```python

assert (
    X_train_w_long.shape[0]
    == len(
        y_train_w_long
    )
)

assert (
    X_train_w_short.shape[0]
    == len(
        y_train_w_short
    )
)

assert (
    X_validation_w_long.shape[0]
    == len(
        y_validation
    )
)

assert (
    X_validation_w_short.shape[0]
    == len(
        y_validation
    )
)


assert (
    X_train_w_long.shape[1]
    ==
    X_train_w_short.shape[1]
    ==
    X_validation_w_long.shape[1]
    ==
    X_validation_w_short.shape[1]
    ==
    len(feature_names)
    ==
    EXPECTED_OUTPUT_WIDTH
)


print(
    "M5.2 X/Y ALIGNMENT GATE: PASS"
)

```

    M5.2 X/Y ALIGNMENT GATE: PASS


## 11. Lineage integrity

Lineage không đi vào classifier.

M5.2 chỉ dùng lineage để kiểm tra row identity/alignment contract.


```python

lineage_specs = {
    "row_id_train_w_long": {
        "array":
            row_id_train_w_long,
        "rows":
            EXPECTED_W_LONG_ROWS,
    },

    "row_id_train_w_short": {
        "array":
            row_id_train_w_short,
        "rows":
            EXPECTED_W_SHORT_ROWS,
    },

    "row_id_validation": {
        "array":
            row_id_validation,
        "rows":
            EXPECTED_VALIDATION_ROWS,
    },
}


for name, spec in (
    lineage_specs.items()
):
    array = spec["array"]

    assert (
        array.dtype
        == np.int64
    ), (
        f"{name}: dtype != int64"
    )

    assert (
        len(array)
        == spec["rows"]
    ), (
        f"{name}: row mismatch"
    )

    if len(array) > 1:
        assert np.all(
            np.diff(
                array
            )
            > 0
        ), (
            f"{name}: row_id không "
            "strictly increasing / unique."
        )


positions = np.searchsorted(
    row_id_train_w_long,
    row_id_train_w_short,
)

assert np.all(
    positions
    < len(
        row_id_train_w_long
    )
)

assert np.array_equal(
    row_id_train_w_long[
        positions
    ],
    row_id_train_w_short,
), (
    "W_SHORT lineage không phải "
    "subset của W_LONG."
)


print(
    "W_LONG lineage rows:",
    len(
        row_id_train_w_long
    ),
)

print(
    "W_SHORT lineage rows:",
    len(
        row_id_train_w_short
    ),
)

print(
    "VALIDATION lineage rows:",
    len(
        row_id_validation
    ),
)

print(
    "\nM5.2 LINEAGE INTEGRITY GATE: PASS"
)

```

    W_LONG lineage rows: 6855270
    W_SHORT lineage rows: 1721615
    VALIDATION lineage rows: 712458
    
    M5.2 LINEAGE INTEGRITY GATE: PASS


## 12. Feature-name safety gate


```python
prohibited_fragments = [
    "is fraud",
    "target",
    "raw_row_id",
    "merchant name",
    "errors",
    "timestamp",
]

normalized_feature_names = [
    str(name).strip().lower()
    for name
    in feature_names
]


for fragment in (
    prohibited_fragments
):
    offending = [
        name
        for name
        in normalized_feature_names
        if fragment
        in name
    ]

    assert not offending, (
        f"Prohibited fragment "
        f"{fragment!r} trong features: "
        f"{offending[:10]}"
    )


# Chỉ cấm raw identifier nếu feature name chính xác là
# raw User / raw Card theo naming convention của output schema.
#
# Không được cấm mọi token "card" vì các behavioral features hợp lệ như:
# - bool__has_prior_card_history
# - num__time_since_previous_transaction_min
# - num__transactions_last_1h
# đều có semantic dựa trên User+Card history nhưng KHÔNG chứa raw identifier value.
raw_identifier_feature_names = {
    "user",
    "card",
    "raw__user",
    "raw__card",
    "id__user",
    "id__card",
    "identifier__user",
    "identifier__card",
}


raw_identifier_offending = [
    name
    for name
    in normalized_feature_names
    if name
    in raw_identifier_feature_names
]


assert not raw_identifier_offending, (
    "Raw identifier feature xuất hiện trong X: "
    f"{raw_identifier_offending}"
)


# Positive contract check:
# behavioral companion feature này PHẢI được phép tồn tại.
assert (
    "bool__has_prior_card_history"
    in normalized_feature_names
), (
    "Thiếu behavioral companion feature "
    "bool__has_prior_card_history"
)


print(
    "Feature count:",
    len(feature_names),
)

print(
    "Unique feature count:",
    len(set(feature_names)),
)

print(
    "\nM5.2 FEATURE SAFETY GATE: PASS"
)
```

    Feature count: 47
    Unique feature count: 47
    
    M5.2 FEATURE SAFETY GATE: PASS


## 13. FINAL TEST isolation gate

M5.2 không có final-test matrix trong required artifact registry.

M5 runner contract cũng không nhận final-test argument.


```python

assert (
    FINAL_TEST_ACCESS_ALLOWED
    is False
)

assert (
    manifest[
        "final_test_used"
    ]
    is False
)

for name, path in (
    artifact_paths.items()
):
    lower_name = (
        name.lower()
        + " "
        + path.name.lower()
    )

    assert (
        "final_test"
        not in lower_name
    ), (
        f"FINAL TEST artifact "
        f"xuất hiện trong M5.2 registry: "
        f"{name}"
    )


print(
    "FINAL TEST artifact registered:",
    False,
)

print(
    "Manifest final_test_used:",
    manifest[
        "final_test_used"
    ],
)

print(
    "\nM5.2 FINAL TEST ISOLATION GATE: PASS"
)

```

    FINAL TEST artifact registered: False
    Manifest final_test_used: False
    
    M5.2 FINAL TEST ISOLATION GATE: PASS


## 14. Canonical metric implementation

Một implementation duy nhất sẽ được dùng cho M5.3–M5.5.

Positive class:

`1 = fraud`

Metric:

- F1_fraud;
- Recall_fraud;
- Precision_fraud;
- Accuracy reference;
- TP / FP / FN / TN;
- predicted-positive count/rate.


```python

def compute_binary_metrics(
    y_true,
    y_pred,
):
    y_true = np.asarray(
        y_true
    )

    y_pred = np.asarray(
        y_pred
    )

    assert (
        y_true.ndim
        == 1
    )

    assert (
        y_pred.ndim
        == 1
    )

    assert (
        len(y_true)
        == len(y_pred)
    )

    assert (
        set(
            np.unique(
                y_true
            ).tolist()
        )
        <= {0, 1}
    )

    assert (
        set(
            np.unique(
                y_pred
            ).tolist()
        )
        <= {0, 1}
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

    predicted_positive_rate = (
        predicted_positive_count
        / len(y_true)
        if len(y_true)
        else np.nan
    )


    return {
        "f1_fraud":
            float(
                f1_score(
                    y_true,
                    y_pred,
                    pos_label=1,
                    zero_division=0,
                )
            ),

        "recall_fraud":
            float(
                recall_score(
                    y_true,
                    y_pred,
                    pos_label=1,
                    zero_division=0,
                )
            ),

        "precision_fraud":
            float(
                precision_score(
                    y_true,
                    y_pred,
                    pos_label=1,
                    zero_division=0,
                )
            ),

        "accuracy_reference":
            float(
                accuracy_score(
                    y_true,
                    y_pred,
                )
            ),

        "tn":
            int(tn),

        "fp":
            int(fp),

        "fn":
            int(fn),

        "tp":
            int(tp),

        "predicted_positive_count":
            predicted_positive_count,

        "predicted_positive_rate":
            float(
                predicted_positive_rate
            ),

        "validation_rows":
            int(
                len(y_true)
            ),
    }

```

## 15. Metric unit test

Đây là infrastructure test bằng dữ liệu toy.

Không phải model-performance experiment.


```python

toy_y_true = np.array(
    [
        0,
        0,
        1,
        1,
    ],
    dtype=np.int8,
)

toy_y_pred = np.array(
    [
        0,
        1,
        1,
        0,
    ],
    dtype=np.int8,
)


toy_metrics = (
    compute_binary_metrics(
        toy_y_true,
        toy_y_pred,
    )
)


assert (
    toy_metrics["tp"]
    == 1
)

assert (
    toy_metrics["fp"]
    == 1
)

assert (
    toy_metrics["fn"]
    == 1
)

assert (
    toy_metrics["tn"]
    == 1
)

assert np.isclose(
    toy_metrics[
        "precision_fraud"
    ],
    0.5,
)

assert np.isclose(
    toy_metrics[
        "recall_fraud"
    ],
    0.5,
)

assert np.isclose(
    toy_metrics[
        "f1_fraud"
    ],
    0.5,
)

assert np.isclose(
    toy_metrics[
        "accuracy_reference"
    ],
    0.5,
)

assert (
    toy_metrics[
        "predicted_positive_count"
    ]
    == 2
)

assert np.isclose(
    toy_metrics[
        "predicted_positive_rate"
    ],
    0.5,
)


print(
    json.dumps(
        toy_metrics,
        ensure_ascii=False,
        indent=2,
    )
)

print(
    "\nM5.2 METRIC UNIT TEST: PASS"
)

```

    {
      "f1_fraud": 0.5,
      "recall_fraud": 0.5,
      "precision_fraud": 0.5,
      "accuracy_reference": 0.5,
      "tn": 1,
      "fp": 1,
      "fn": 1,
      "tp": 1,
      "predicted_positive_count": 2,
      "predicted_positive_rate": 0.5,
      "validation_rows": 4
    }
    
    M5.2 METRIC UNIT TEST: PASS


## 16. Shared experiment runner v1

Runner contract:

- caller truyền đúng một estimator/config;
- runner clone estimator trước fit;
- runner không resample;
- runner dùng `predict()` cho baseline class decision;
- giữ `predict_proba()` hoặc `decision_function()` khi available;
- ghi warning;
- ghi fit/prediction time;
- tính canonical metric bundle;
- không có FINAL TEST argument.


```python

def _json_safe_value(
    value,
):
    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
            type(None),
        ),
    ):
        return value

    if isinstance(
        value,
        np.generic,
    ):
        return value.item()

    if isinstance(
        value,
        (
            list,
            tuple,
        ),
    ):
        return [
            _json_safe_value(
                item
            )
            for item
            in value
        ]

    if isinstance(
        value,
        dict,
    ):
        return {
            str(key):
                _json_safe_value(
                    item
                )
            for key, item
            in value.items()
        }

    return repr(value)


def extract_positive_score(
    fitted_estimator,
    X,
):
    if hasattr(
        fitted_estimator,
        "predict_proba",
    ):
        probabilities = (
            fitted_estimator
            .predict_proba(
                X
            )
        )

        classes = np.asarray(
            fitted_estimator
            .classes_
        )

        positive_positions = np.flatnonzero(
            classes
            == 1
        )

        if len(
            positive_positions
        ) != 1:
            raise RuntimeError(
                "Không xác định được "
                "positive class = 1 "
                "trong predict_proba."
            )

        score = (
            probabilities[
                :,
                positive_positions[
                    0
                ],
            ]
        )

        return (
            np.asarray(
                score,
                dtype=np.float64,
            ),
            "predict_proba",
        )


    if hasattr(
        fitted_estimator,
        "decision_function",
    ):
        score = (
            fitted_estimator
            .decision_function(
                X
            )
        )

        score = np.asarray(
            score,
            dtype=np.float64,
        )

        if (
            score.ndim
            != 1
        ):
            raise RuntimeError(
                "decision_function không "
                "phải binary 1-D score."
            )

        return (
            score,
            "decision_function",
        )


    return (
        None,
        None,
    )


@dataclass
class ExperimentRunOutput:
    summary: Dict[str, Any]
    y_pred: np.ndarray
    risk_score: Optional[np.ndarray]
    risk_score_kind: Optional[str]
    warning_messages: List[str]


def run_baseline_experiment(
    *,
    experiment_id,
    estimator,
    model_family,
    model_id,
    model_config_id,
    X_train,
    y_train,
    X_validation,
    y_validation,
    training_window_id,
    random_state=None,
    feature_version=(
        "Feature Specification v1.0"
    ),
    preprocessing_version=(
        "Preprocessing Specification v1.0"
    ),
    matrix_schema_version=(
        "Baseline Matrix Schema v1.0"
    ),
    imbalance_strategy=(
        IMBALANCE_STRATEGY
    ),
    threshold_policy=(
        BASELINE_THRESHOLD_POLICY
    ),
    decision=(
        "OPEN — REQUIRES RUNTIME REVIEW"
    ),
    next_action=(
        "REVIEW RUN OUTPUT"
    ),
):
    assert (
        imbalance_strategy
        == "NONE"
    ), (
        "M5 core baseline không cho phép "
        "imbalance intervention."
    )

    assert training_window_id in {
        "W_LONG",
        "W_SHORT",
    }

    assert isinstance(
        model_family,
        str,
    ) and model_family.strip()

    assert isinstance(
        model_id,
        str,
    ) and model_id.strip()

    assert isinstance(
        model_config_id,
        str,
    ) and model_config_id.strip()

    if random_state is not None:
        assert isinstance(
            random_state,
            (int, np.integer),
        )

    assert sparse.isspmatrix_csr(
        X_train
    )

    assert sparse.isspmatrix_csr(
        X_validation
    )

    assert (
        X_train.shape[1]
        == EXPECTED_OUTPUT_WIDTH
    )

    assert (
        X_validation.shape[1]
        == EXPECTED_OUTPUT_WIDTH
    )

    assert (
        X_train.shape[0]
        == len(y_train)
    )

    assert (
        X_validation.shape[0]
        == len(y_validation)
    )

    assert np.isfinite(
        X_train.data
    ).all()

    assert np.isfinite(
        X_validation.data
    ).all()

    assert (
        set(
            np.unique(
                y_train
            ).tolist()
        )
        <= {0, 1}
    )

    assert (
        set(
            np.unique(
                y_validation
            ).tolist()
        )
        <= {0, 1}
    )


    model = clone(
        estimator
    )


    with warnings.catch_warnings(
        record=True
    ) as caught_warnings:
        warnings.simplefilter(
            "always"
        )

        fit_start = (
            time.perf_counter()
        )

        model.fit(
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

        y_pred = model.predict(
            X_validation
        )

        risk_score, (
            risk_score_kind
        ) = extract_positive_score(
            model,
            X_validation,
        )

        prediction_seconds = (
            time.perf_counter()
            - prediction_start
        )


    y_pred = np.asarray(
        y_pred,
        dtype=np.int8,
    )

    metrics = (
        compute_binary_metrics(
            y_validation,
            y_pred,
        )
    )

    warning_messages = [
        (
            f"{item.category.__name__}: "
            f"{item.message}"
        )
        for item
        in caught_warnings
    ]


    model_params = {}

    if hasattr(
        model,
        "get_params",
    ):
        model_params = {
            key:
                _json_safe_value(
                    value
                )
            for key, value
            in model.get_params(
                deep=False
            ).items()
        }


    summary = {
        "runner_version":
            M5_RUNNER_VERSION,

        "experiment_id":
            str(
                experiment_id
            ),

        "run_status":
            "COMPLETED",

        "model_family":
            str(
                model_family
            ),

        "model_id":
            str(
                model_id
            ),

        "model_config_id":
            str(
                model_config_id
            ),

        "model_class":
            model.__class__.__name__,

        "model_module":
            model.__class__.__module__,

        "model_params":
            model_params,

        "training_window_id":
            training_window_id,

        "feature_version":
            feature_version,

        "preprocessing_version":
            preprocessing_version,

        "matrix_schema_version":
            matrix_schema_version,

        "feature_count":
            int(
                X_train.shape[1]
            ),

        "training_rows":
            int(
                X_train.shape[0]
            ),

        "training_fraud_rows":
            int(
                np.asarray(
                    y_train
                ).sum()
            ),

        "validation_rows":
            int(
                X_validation.shape[0]
            ),

        "validation_fraud_rows":
            int(
                np.asarray(
                    y_validation
                ).sum()
            ),

        "imbalance_strategy":
            imbalance_strategy,

        "random_state":
            (
                None
                if random_state is None
                else int(
                    random_state
                )
            ),

        "threshold_policy":
            threshold_policy,

        "fit_seconds":
            float(
                fit_seconds
            ),

        "prediction_seconds":
            float(
                prediction_seconds
            ),

        "risk_score_available":
            (
                risk_score
                is not None
            ),

        "risk_score_kind":
            risk_score_kind,

        "warning_count":
            len(
                warning_messages
            ),

        "warnings":
            warning_messages,

        "errors":
            [],

        "final_test_accessed":
            False,

        "integrity_gate_result":
            "PASS",

        "decision":
            str(
                decision
            ),

        "next_action":
            str(
                next_action
            ),

        **metrics,
    }


    return ExperimentRunOutput(
        summary=summary,
        y_pred=y_pred,
        risk_score=(
            None
            if risk_score is None
            else np.asarray(
                risk_score,
                dtype=np.float32,
            )
        ),
        risk_score_kind=
            risk_score_kind,
        warning_messages=
            warning_messages,
    )

```

## 17. Prediction artifact persistence helper

M5.3–M5.5 có thể dùng helper này để lưu:

- prediction;
- probability/risk score;
- experiment summary.

FINAL TEST không có trong interface.


```python

def save_experiment_output(
    run_output,
    output_dir,
):
    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    experiment_id = (
        run_output
        .summary[
            "experiment_id"
        ]
    )


    prediction_path = (
        output_dir
        / (
            experiment_id
            + "__y_pred.npy"
        )
    )

    summary_path = (
        output_dir
        / (
            experiment_id
            + "__summary.json"
        )
    )


    np.save(
        prediction_path,
        run_output.y_pred,
        allow_pickle=False,
    )


    risk_score_path = None

    if (
        run_output
        .risk_score
        is not None
    ):
        risk_score_path = (
            output_dir
            / (
                experiment_id
                + "__risk_score.npy"
            )
        )

        np.save(
            risk_score_path,
            run_output.risk_score,
            allow_pickle=False,
        )


    persisted_summary = dict(
        run_output.summary
    )

    persisted_summary[
        "prediction_file"
    ] = prediction_path.name

    persisted_summary[
        "risk_score_file"
    ] = (
        None
        if risk_score_path
        is None
        else risk_score_path.name
    )


    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            persisted_summary,
            file,
            ensure_ascii=False,
            indent=2,
        )


    return {
        "summary_path":
            summary_path,

        "prediction_path":
            prediction_path,

        "risk_score_path":
            risk_score_path,
    }

```

## 18. Shared runner infrastructure self-test

Test dưới đây dùng một estimator stub trên toy data.

Mục đích:

- kiểm tra clone/fit/predict flow;
- kiểm tra probability extraction;
- kiểm tra canonical metrics;
- kiểm tra persistence helper.

Đây **không phải** baseline model của project.


```python

class RunnerContractStub(
    BaseEstimator,
    ClassifierMixin,
):
    def fit(
        self,
        X,
        y,
    ):
        self.classes_ = np.array(
            [
                0,
                1,
            ]
        )

        return self


    def predict(
        self,
        X,
    ):
        return np.zeros(
            X.shape[0],
            dtype=np.int8,
        )


    def predict_proba(
        self,
        X,
    ):
        result = np.empty(
            (
                X.shape[0],
                2,
            ),
            dtype=np.float64,
        )

        result[
            :,
            0
        ] = 0.9

        result[
            :,
            1
        ] = 0.1

        return result


toy_X_train = sparse.csr_matrix(
    np.array(
        [
            [0.0] * EXPECTED_OUTPUT_WIDTH,
            [1.0] * EXPECTED_OUTPUT_WIDTH,
            [0.5] * EXPECTED_OUTPUT_WIDTH,
            [0.2] * EXPECTED_OUTPUT_WIDTH,
        ],
        dtype=np.float32,
    )
)

toy_X_validation = sparse.csr_matrix(
    np.array(
        [
            [0.1] * EXPECTED_OUTPUT_WIDTH,
            [0.9] * EXPECTED_OUTPUT_WIDTH,
        ],
        dtype=np.float32,
    )
)

toy_train_y = np.array(
    [
        0,
        1,
        0,
        1,
    ],
    dtype=np.int8,
)

toy_validation_y = np.array(
    [
        0,
        1,
    ],
    dtype=np.int8,
)


toy_run = run_baseline_experiment(
    experiment_id=(
        "M5.2-RUNNER-CONTRACT-SELF-TEST"
    ),
    estimator=RunnerContractStub(),
    model_family=(
        "CONTRACT_STUB"
    ),
    model_id=(
        "RUNNER_CONTRACT_STUB"
    ),
    model_config_id=(
        "SELF_TEST_V1"
    ),
    random_state=(
        DEFAULT_RANDOM_STATE
    ),
    X_train=toy_X_train,
    y_train=toy_train_y,
    X_validation=
        toy_X_validation,
    y_validation=
        toy_validation_y,
    training_window_id=
        "W_SHORT",
)


assert (
    toy_run
    .summary[
        "run_status"
    ]
    == "COMPLETED"
)

assert (
    toy_run
    .summary[
        "risk_score_available"
    ]
    is True
)

assert (
    toy_run
    .risk_score_kind
    == "predict_proba"
)

assert (
    toy_run
    .summary[
        "final_test_accessed"
    ]
    is False
)


assert (
    toy_run
    .summary[
        "model_family"
    ]
    == "CONTRACT_STUB"
)

assert (
    toy_run
    .summary[
        "model_id"
    ]
    == "RUNNER_CONTRACT_STUB"
)

assert (
    toy_run
    .summary[
        "model_config_id"
    ]
    == "SELF_TEST_V1"
)

assert (
    toy_run
    .summary[
        "random_state"
    ]
    == DEFAULT_RANDOM_STATE
)

assert (
    toy_run
    .summary[
        "errors"
    ]
    == []
)

assert (
    toy_run
    .summary[
        "integrity_gate_result"
    ]
    == "PASS"
)

assert (
    toy_run
    .summary[
        "decision"
    ]
    == "OPEN — REQUIRES RUNTIME REVIEW"
)

assert (
    toy_run
    .summary[
        "next_action"
    ]
    == "REVIEW RUN OUTPUT"
)

assert np.array_equal(
    toy_run.y_pred,
    np.array(
        [
            0,
            0,
        ],
        dtype=np.int8,
    ),
)

assert np.allclose(
    toy_run.risk_score,
    np.array(
        [
            0.1,
            0.1,
        ],
        dtype=np.float32,
    ),
)


with tempfile.TemporaryDirectory() as (
    temp_dir
):
    saved = save_experiment_output(
        toy_run,
        temp_dir,
    )

    assert (
        saved[
            "summary_path"
        ].exists()
    )

    assert (
        saved[
            "prediction_path"
        ].exists()
    )

    assert (
        saved[
            "risk_score_path"
        ].exists()
    )


print(
    json.dumps(
        toy_run.summary,
        ensure_ascii=False,
        indent=2,
    )
)

print(
    "\nM5.2 SHARED RUNNER SELF-TEST: PASS"
)

```

    {
      "runner_version": "M5.2-shared-runner-v1",
      "experiment_id": "M5.2-RUNNER-CONTRACT-SELF-TEST",
      "run_status": "COMPLETED",
      "model_family": "CONTRACT_STUB",
      "model_id": "RUNNER_CONTRACT_STUB",
      "model_config_id": "SELF_TEST_V1",
      "model_class": "RunnerContractStub",
      "model_module": "__main__",
      "model_params": {},
      "training_window_id": "W_SHORT",
      "feature_version": "Feature Specification v1.0",
      "preprocessing_version": "Preprocessing Specification v1.0",
      "matrix_schema_version": "Baseline Matrix Schema v1.0",
      "feature_count": 47,
      "training_rows": 4,
      "training_fraud_rows": 2,
      "validation_rows": 2,
      "validation_fraud_rows": 1,
      "imbalance_strategy": "NONE",
      "random_state": 42,
      "threshold_policy": "DEFAULT_MODEL_DECISION_RULE",
      "fit_seconds": 3.7080026231706142e-06,
      "prediction_seconds": 2.3917004000395536e-05,
      "risk_score_available": true,
      "risk_score_kind": "predict_proba",
      "warning_count": 0,
      "warnings": [],
      "errors": [],
      "final_test_accessed": false,
      "integrity_gate_result": "PASS",
      "decision": "OPEN — REQUIRES RUNTIME REVIEW",
      "next_action": "REVIEW RUN OUTPUT",
      "f1_fraud": 0.0,
      "recall_fraud": 0.0,
      "precision_fraud": 0.0,
      "accuracy_reference": 0.5,
      "tn": 1,
      "fp": 0,
      "fn": 1,
      "tp": 0,
      "predicted_positive_count": 0,
      "predicted_positive_rate": 0.0
    }
    
    M5.2 SHARED RUNNER SELF-TEST: PASS


## 19. Experiment result schema contract

M5.3–M5.5 phải dùng cùng field semantics.

Không được để mỗi model tự phát minh một kiểu log khác nhau.


```python

EXPERIMENT_RESULT_REQUIRED_FIELDS = [
    "runner_version",
    "experiment_id",
    "run_status",

    "model_family",
    "model_id",
    "model_config_id",
    "model_class",
    "model_module",
    "model_params",

    "training_window_id",

    "feature_version",
    "preprocessing_version",
    "matrix_schema_version",
    "feature_count",

    "training_rows",
    "training_fraud_rows",
    "validation_rows",
    "validation_fraud_rows",

    "imbalance_strategy",
    "random_state",
    "threshold_policy",

    "fit_seconds",
    "prediction_seconds",

    "risk_score_available",
    "risk_score_kind",

    "warning_count",
    "warnings",
    "errors",

    "final_test_accessed",
    "integrity_gate_result",

    "f1_fraud",
    "recall_fraud",
    "precision_fraud",
    "accuracy_reference",

    "tn",
    "fp",
    "fn",
    "tp",

    "predicted_positive_count",
    "predicted_positive_rate",

    "decision",
    "next_action",
]


missing_self_test_fields = [
    field
    for field
    in EXPERIMENT_RESULT_REQUIRED_FIELDS
    if field
    not in toy_run.summary
]


assert not (
    missing_self_test_fields
), (
    "Runner thiếu fields: "
    + repr(
        missing_self_test_fields
    )
)


M5_1_CANONICAL_LOG_FIELDS = {
    "experiment_id",
    "run_status",
    "model_family",
    "model_id",
    "model_config_id",
    "training_window_id",
    "feature_version",
    "preprocessing_version",
    "matrix_schema_version",
    "training_rows",
    "training_fraud_rows",
    "validation_rows",
    "validation_fraud_rows",
    "imbalance_strategy",
    "random_state",
    "threshold_policy",
    "f1_fraud",
    "recall_fraud",
    "precision_fraud",
    "accuracy_reference",
    "tn",
    "fp",
    "fn",
    "tp",
    "predicted_positive_count",
    "predicted_positive_rate",
    "risk_score_available",
    "fit_seconds",
    "prediction_seconds",
    "warnings",
    "errors",
    "final_test_accessed",
    "integrity_gate_result",
    "decision",
    "next_action",
}


missing_vs_m5_1 = sorted(
    M5_1_CANONICAL_LOG_FIELDS
    - set(
        EXPERIMENT_RESULT_REQUIRED_FIELDS
    )
)


assert not missing_vs_m5_1, (
    "M5.2 experiment schema chưa đủ "
    "M5.1 canonical contract: "
    + repr(
        missing_vs_m5_1
    )
)


print(
    "Experiment result field count:",
    len(
        EXPERIMENT_RESULT_REQUIRED_FIELDS
    ),
)

for field in (
    EXPERIMENT_RESULT_REQUIRED_FIELDS
):
    print(field)


print(
    "\nM5.2 EXPERIMENT-SCHEMA GATE: PASS"
)

```

    Experiment result field count: 42
    runner_version
    experiment_id
    run_status
    model_family
    model_id
    model_config_id
    model_class
    model_module
    model_params
    training_window_id
    feature_version
    preprocessing_version
    matrix_schema_version
    feature_count
    training_rows
    training_fraud_rows
    validation_rows
    validation_fraud_rows
    imbalance_strategy
    random_state
    threshold_policy
    fit_seconds
    prediction_seconds
    risk_score_available
    risk_score_kind
    warning_count
    warnings
    errors
    final_test_accessed
    integrity_gate_result
    f1_fraud
    recall_fraud
    precision_fraud
    accuracy_reference
    tn
    fp
    fn
    tp
    predicted_positive_count
    predicted_positive_rate
    decision
    next_action
    
    M5.2 EXPERIMENT-SCHEMA GATE: PASS


## 20. Shared input registry cho M5.3–M5.5

Registry này chỉ map W_LONG/W_SHORT tới canonical matrices.

Không có FINAL TEST entry.


```python

MODELING_INPUTS = {
    "W_LONG": {
        "X_train":
            X_train_w_long,

        "y_train":
            y_train_w_long,

        "X_validation":
            X_validation_w_long,

        "y_validation":
            y_validation,

        "row_id_train":
            row_id_train_w_long,

        "row_id_validation":
            row_id_validation,
    },

    "W_SHORT": {
        "X_train":
            X_train_w_short,

        "y_train":
            y_train_w_short,

        "X_validation":
            X_validation_w_short,

        "y_validation":
            y_validation,

        "row_id_train":
            row_id_train_w_short,

        "row_id_validation":
            row_id_validation,
    },
}


assert set(
    MODELING_INPUTS.keys()
) == {
    "W_LONG",
    "W_SHORT",
}


for strategy, item in (
    MODELING_INPUTS.items()
):
    assert (
        item[
            "X_train"
        ].shape[0]
        ==
        len(
            item[
                "y_train"
            ]
        )
    )

    assert (
        item[
            "X_validation"
        ].shape[0]
        ==
        len(
            item[
                "y_validation"
            ]
        )
    )

    assert (
        item[
            "X_train"
        ].shape[1]
        ==
        EXPECTED_OUTPUT_WIDTH
    )

    assert (
        item[
            "X_validation"
        ].shape[1]
        ==
        EXPECTED_OUTPUT_WIDTH
    )


print(
    "MODELING_INPUTS keys:"
)

print(
    sorted(
        MODELING_INPUTS.keys()
    )
)

print(
    "\nM5.2 MODELING-INPUT REGISTRY GATE: PASS"
)

```

    MODELING_INPUTS keys:
    ['W_LONG', 'W_SHORT']
    
    M5.2 MODELING-INPUT REGISTRY GATE: PASS


## 21. Persist M5.2 audit/runner contract metadata

File metadata này không chứa model score.

Nó ghi lại:

- artifact version;
- exact shapes;
- exact nnz;
- runner version;
- metric policy;
- baseline imbalance policy;
- threshold policy;
- FINAL TEST isolation.


```python

m5_02_contract = {
    "m5_substep":
        "M5.2",

    "runner_version":
        M5_RUNNER_VERSION,

    "input_pipeline_version":
        manifest[
            "pipeline_version"
        ],

    "feature_count":
        len(feature_names),

    "matrix_format":
        "CSR",

    "matrix_dtype":
        "float32",

    "target_dtype":
        "int8",

    "imbalance_strategy":
        IMBALANCE_STRATEGY,

    "random_state_default":
        DEFAULT_RANDOM_STATE,

    "random_state_role":
        "REPRODUCIBILITY_NOT_TUNING",

    "threshold_policy":
        BASELINE_THRESHOLD_POLICY,

    "primary_metric":
        "f1_fraud",

    "secondary_metrics": [
        "recall_fraud",
        "precision_fraud",
    ],

    "mandatory_diagnostics": [
        "tn",
        "fp",
        "fn",
        "tp",
        "predicted_positive_count",
        "predicted_positive_rate",
    ],

    "accuracy_role":
        "REFERENCE_ONLY",

    "probability_policy":
        "PRESERVE_WHEN_AVAILABLE",

    "final_test_access_allowed":
        False,

    "matrices": {
        name: {
            "shape": [
                int(
                    matrix.shape[0]
                ),
                int(
                    matrix.shape[1]
                ),
            ],

            "nnz":
                int(
                    matrix.nnz
                ),

            "dtype":
                str(
                    matrix.dtype
                ),
        }
        for name, matrix
        in matrices.items()
    },

    "experiment_result_required_fields":
        EXPERIMENT_RESULT_REQUIRED_FIELDS,

    "m5_1_canonical_log_fields":
        sorted(
            M5_1_CANONICAL_LOG_FIELDS
        ),

    "experiment_schema_covers_m5_1":
        (
            set(
                M5_1_CANONICAL_LOG_FIELDS
            )
            <= set(
                EXPERIMENT_RESULT_REQUIRED_FIELDS
            )
        ),
}


contract_path = (
    M5_02_OUTPUT_DIR
    / "m5_02_modeling_contract.json"
)


with open(
    contract_path,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        m5_02_contract,
        file,
        ensure_ascii=False,
        indent=2,
    )


assert contract_path.exists()

assert (
    contract_path.stat().st_size
    > 0
)


with open(
    contract_path,
    "r",
    encoding="utf-8",
) as file:
    reloaded_contract = (
        json.load(file)
    )


assert (
    reloaded_contract[
        "runner_version"
    ]
    == M5_RUNNER_VERSION
)

assert (
    reloaded_contract[
        "final_test_access_allowed"
    ]
    is False
)

assert (
    reloaded_contract[
        "experiment_schema_covers_m5_1"
    ]
    is True
)


print(
    "Saved M5.2 contract:"
)

print(
    contract_path
)

print(
    "\nM5.2 CONTRACT PERSISTENCE GATE: PASS"
)

```

    Saved M5.2 contract:
    /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/processed/m5_02_modeling_contract/m5_02_modeling_contract.json
    
    M5.2 CONTRACT PERSISTENCE GATE: PASS


## 22. M5.2 overall gate


```python

G01_REQUIRED_FILES = True
G02_MANIFEST_SCHEMA = True
G03_MATRIX_INTEGRITY = True
G04_STALE_ARTIFACT_PROTECTION = True
G05_TARGET_INTEGRITY = True
G06_ALIGNMENT = True
G07_LINEAGE = True
G08_FEATURE_SAFETY = True
G09_FINAL_TEST_ISOLATION = True
G10_METRIC_IMPLEMENTATION = True
G11_SHARED_RUNNER = True
G12_EXPERIMENT_SCHEMA = True
G13_MODELING_INPUT_REGISTRY = True
G14_CONTRACT_PERSISTENCE = True


m5_02_gates = {
    "G01_REQUIRED_FILES":
        G01_REQUIRED_FILES,

    "G02_MANIFEST_SCHEMA":
        G02_MANIFEST_SCHEMA,

    "G03_MATRIX_INTEGRITY":
        G03_MATRIX_INTEGRITY,

    "G04_STALE_ARTIFACT_PROTECTION":
        G04_STALE_ARTIFACT_PROTECTION,

    "G05_TARGET_INTEGRITY":
        G05_TARGET_INTEGRITY,

    "G06_ALIGNMENT":
        G06_ALIGNMENT,

    "G07_LINEAGE":
        G07_LINEAGE,

    "G08_FEATURE_SAFETY":
        G08_FEATURE_SAFETY,

    "G09_FINAL_TEST_ISOLATION":
        G09_FINAL_TEST_ISOLATION,

    "G10_METRIC_IMPLEMENTATION":
        G10_METRIC_IMPLEMENTATION,

    "G11_SHARED_RUNNER":
        G11_SHARED_RUNNER,

    "G12_EXPERIMENT_SCHEMA":
        G12_EXPERIMENT_SCHEMA,

    "G13_MODELING_INPUT_REGISTRY":
        G13_MODELING_INPUT_REGISTRY,

    "G14_CONTRACT_PERSISTENCE":
        G14_CONTRACT_PERSISTENCE,
}


for gate_name, gate_value in (
    m5_02_gates.items()
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


assert all(
    m5_02_gates.values()
)


print(
    "\nM5.2 OVERALL TECHNICAL GATE: PASS"
)

```

    G01_REQUIRED_FILES → PASS
    G02_MANIFEST_SCHEMA → PASS
    G03_MATRIX_INTEGRITY → PASS
    G04_STALE_ARTIFACT_PROTECTION → PASS
    G05_TARGET_INTEGRITY → PASS
    G06_ALIGNMENT → PASS
    G07_LINEAGE → PASS
    G08_FEATURE_SAFETY → PASS
    G09_FINAL_TEST_ISOLATION → PASS
    G10_METRIC_IMPLEMENTATION → PASS
    G11_SHARED_RUNNER → PASS
    G12_EXPERIMENT_SCHEMA → PASS
    G13_MODELING_INPUT_REGISTRY → PASS
    G14_CONTRACT_PERSISTENCE → PASS
    
    M5.2 OVERALL TECHNICAL GATE: PASS


# 23. Phân tích và nhận xét runtime M5.2

## 23.1. Execution integrity

Notebook đã được thực thi đầy đủ:

`21 / 21 code cells`

Execution count:

`1 → 21 liên tục`

Không ghi nhận:

- exception;
- `output_type = error`;
- `stderr`.

Điều này cho phép sử dụng toàn bộ output hiện tại làm runtime evidence cho M5.2.

Nhận xét:

M5.2 không còn ở trạng thái “code đã viết nhưng chưa được xác minh”. Toàn bộ chuỗi audit từ artifact discovery đến overall technical gate đã được thực thi trong cùng notebook.

---

## 23.2. Artifact discovery và required-file integrity

Canonical artifact directory được xác định thành công:

`data/processed/m4_07_baseline_ready`

Toàn bộ 12 artifact bắt buộc đều tồn tại và có kích thước khác 0:

- 4 sparse matrices;
- 3 target arrays;
- 3 lineage arrays;
- `feature_names.json`;
- `manifest.json`.

Runtime gate:

`M5.2 REQUIRED-FILE GATE: PASS`

Nhận xét:

M5.2 đang sử dụng đúng persisted handoff từ M4.7 thay vì tự rebuild feature/preprocessing representation. Điều này giữ nguyên separation of responsibility:

`M4 = xây representation`

`M5 = modeling trên representation đã khóa`

---

## 23.3. Manifest và feature-name contract

Manifest runtime xác nhận:

`pipeline_version = M4.7-baseline-v1`

`output_width = 47`

`matrix_dtype = float32`

`matrix_format = CSR`

`target_dtype = int8`

`raw_identifiers_in_X = false`

`target_in_X = false`

`final_test_used = false`

Feature registry có:

`47 features`

và toàn bộ tên feature là unique.

Các feature đầu tiên khớp baseline contract, gồm:

- `num__amount_numeric`;
- `num__time_since_previous_transaction_min`;
- `num__transactions_last_1h`;
- `num__amount_minus_previous_mean`;
- `bool__is_new_merchant`;
- `bool__has_prior_card_history`;
- categorical one-hot features.

Runtime gate:

`M5.2 MANIFEST / FEATURE-NAME GATE: PASS`

Nhận xét:

Modeling code ở M5 có thể dựa vào một schema đã định danh rõ, thay vì suy đoán số cột hoặc tự dựng tên feature mới.

---

## 23.4. Matrix integrity và stale-artifact protection

Runtime matrix evidence:

```text
X_train_w_long
rows      = 6,855,270
columns   = 47
dtype     = float32
nnz       = 61,900,193
density   = 19.211867%
CSR memory ≈ 498.41 MB

X_train_w_short
rows      = 1,721,615
columns   = 47
dtype     = float32
nnz       = 15,542,381
density   = 19.208067%
CSR memory ≈ 125.15 MB

X_validation_w_long
rows      = 712,458
columns   = 47
dtype     = float32
nnz       = 6,430,339
density   = 19.203339%
CSR memory ≈ 51.78 MB

X_validation_w_short
rows      = 712,458
columns   = 47
dtype     = float32
nnz       = 6,430,339
density   = 19.203339%
CSR memory ≈ 51.78 MB
```

Runtime gate:

`M5.2 MATRIX / STALE-ARTIFACT GATE: PASS`

Nhận xét:

Exact `nnz` signature khớp artifacts M4.7 đã được regenerate sau numeric-scaler fix.

Đây là bằng chứng quan trọng vì artifact M4.7 cũ từng có density thấp bất thường do hai numeric feature bị suy biến. M5.2 hiện không load lại bộ stale artifact đó.

Matrix vẫn giữ sparse CSR; notebook không dense hóa W_LONG. Đây là điều kiện thực tế quan trọng cho M5.3–M5.5 vì W_LONG có hơn 6.8 triệu rows.

---

## 23.5. Target integrity

Runtime target evidence:

```text
y_train_w_long
rows      = 6,855,270
fraud     = 9,606
non-fraud = 6,845,664
dtype     = int8

y_train_w_short
rows      = 1,721,615
fraud     = 2,491
non-fraud = 1,719,124
dtype     = int8

y_validation
rows      = 712,458
fraud     = 1,052
non-fraud = 711,406
dtype     = int8
```

Runtime gate:

`M5.2 TARGET INTEGRITY GATE: PASS`

Nhận xét:

Target support khớp canonical counts từ M3/M4.

M5.2 không phát hiện label ngoài `{0, 1}` và không phát hiện mismatch giữa row count của X và y.

---

## 23.6. X/y alignment

Runtime gate:

`M5.2 X/Y ALIGNMENT GATE: PASS`

Đã xác minh:

- `X_train_w_long` khớp `y_train_w_long`;
- `X_train_w_short` khớp `y_train_w_short`;
- cả hai validation matrices khớp cùng `y_validation`;
- tất cả matrices có đúng 47 columns.

Nhận xét:

Điều kiện cơ bản để fit/predict ở M5.3–M5.5 đã được xác minh trước khi bất kỳ classifier thật nào được huấn luyện.

---

## 23.7. Lineage integrity

Runtime evidence:

```text
W_LONG lineage rows:
6,855,270

W_SHORT lineage rows:
1,721,615

VALIDATION lineage rows:
712,458
```

Runtime gate:

`M5.2 LINEAGE INTEGRITY GATE: PASS`

Ngoài row count, code còn xác minh:

- lineage dtype = `int64`;
- row IDs strictly increasing/unique;
- W_SHORT lineage là subset của W_LONG lineage.

Nhận xét:

Lineage vẫn được giữ riêng khỏi classifier matrix nhưng đủ để audit row identity khi cần.

---

## 23.8. Feature safety

Runtime evidence:

`Feature count = 47`

`Unique feature count = 47`

Runtime gate:

`M5.2 FEATURE SAFETY GATE: PASS`

Không phát hiện prohibited feature fragments như:

- target;
- raw row id;
- Merchant Name;
- Errors?;
- raw Timestamp.

Raw identifier gate đã được sửa để phân biệt đúng:

`raw Card identifier`

với behavioral semantic feature hợp lệ:

`bool__has_prior_card_history`.

Nhận xét:

Feature-safety check hiện phù hợp với M4 Behavioral Feature Contract và không còn false-positive đối với companion behavioral feature.

---

## 23.9. FINAL TEST isolation

Runtime evidence:

`FINAL TEST artifact registered: False`

`Manifest final_test_used: False`

Runtime gate:

`M5.2 FINAL TEST ISOLATION GATE: PASS`

Nhận xét:

Shared M5 modeling input path chỉ chứa W_LONG/W_SHORT TRAIN và external VALIDATION.

Không có FINAL TEST entry trong artifact registry hoặc shared modeling registry.

Điều này giữ đúng M3/M5.1 partition-rights contract:

`FINAL TEST ACCESS DURING DEVELOPMENT = PROHIBITED`

---

## 23.10. Canonical metric implementation

Metric unit test tạo deliberate confusion matrix:

```text
TN = 1
FP = 1
FN = 1
TP = 1
```

và runtime trả:

```text
F1_fraud        = 0.5
Recall_fraud    = 0.5
Precision_fraud = 0.5
Accuracy        = 0.5

predicted_positive_count = 2
predicted_positive_rate  = 0.5
```

Runtime gate:

`M5.2 METRIC UNIT TEST: PASS`

Nhận xét:

Canonical metric function đang tính đúng positive class `fraud = 1` và cung cấp đủ metric bundle đã khóa từ M3/M5.1.

M5.3–M5.5 không cần viết lại metric code riêng cho từng model.

---

## 23.11. Shared baseline runner

Runtime self-test xác nhận runner có thể thực hiện đầy đủ:

```text
clone estimator
→ fit
→ predict
→ extract risk score
→ canonical metrics
→ warning capture
→ experiment summary
→ persistence helper
```

Self-test metadata xác nhận:

- runner version = `M5.2-shared-runner-v1`;
- imbalance strategy = `NONE`;
- random state = `42`;
- threshold policy = `DEFAULT_MODEL_DECISION_RULE`;
- risk score được lấy bằng `predict_proba`;
- warning count = `0`;
- errors = `[]`;
- `final_test_accessed = false`;
- `integrity_gate_result = PASS`.

Runtime gate:

`M5.2 SHARED RUNNER SELF-TEST: PASS`

Nhận xét quan trọng:

Toy self-test có:

`F1_fraud = 0.0`

nhưng đây **không phải model-performance result của project**.

Estimator được dùng chỉ là `RunnerContractStub`, được thiết kế để kiểm tra mechanics của infrastructure. Không được dùng score toy này để diễn giải dữ liệu fraud hoặc chất lượng model.

---

## 23.12. Experiment-result schema

Runtime output:

`Experiment result field count = 42`

Schema hiện bao gồm đầy đủ các nhóm:

- experiment identity;
- model family/id/config;
- data/version contract;
- row/fraud counts;
- imbalance strategy;
- random state;
- threshold policy;
- runtime;
- probability/risk-score information;
- warnings;
- errors;
- FINAL TEST access flag;
- integrity gate;
- canonical metrics;
- decision;
- next action.

Runtime gate:

`M5.2 EXPERIMENT-SCHEMA GATE: PASS`

Đặc biệt, schema đã được sửa để bao phủ các field governance do M5.1 yêu cầu:

- `model_family`;
- `model_id`;
- `model_config_id`;
- `random_state`;
- `errors`;
- `integrity_gate_result`;
- `decision`;
- `next_action`.

Nhận xét:

Blocker contract-completeness phát hiện ở lần review trước đã được xử lý.

---

## 23.13. Shared modeling-input registry

Runtime registry chỉ chứa:

```text
W_LONG
W_SHORT
```

Runtime gate:

`M5.2 MODELING-INPUT REGISTRY GATE: PASS`

Không có FINAL TEST entry.

Nhận xét:

M5.3–M5.5 có thể lấy input qua cùng một registry thay vì tự tìm file hoặc tự gắn validation population.

---

## 23.14. Contract persistence

M5.2 đã persist:

`data/processed/m5_02_modeling_contract/m5_02_modeling_contract.json`

Runtime gate:

`M5.2 CONTRACT PERSISTENCE GATE: PASS`

Contract persisted gồm:

- input pipeline version;
- matrix signature;
- runner version;
- metric policy;
- no-imbalance policy;
- random-state policy;
- threshold policy;
- FINAL TEST prohibition;
- experiment-result schema;
- explicit check rằng schema bao phủ M5.1 canonical log fields.

Nhận xét:

Model-specific notebooks phía sau có thể dùng cùng một machine-readable contract.

---

## 23.15. Overall technical gate

Runtime output:

```text
G01_REQUIRED_FILES            → PASS
G02_MANIFEST_SCHEMA           → PASS
G03_MATRIX_INTEGRITY          → PASS
G04_STALE_ARTIFACT_PROTECTION → PASS
G05_TARGET_INTEGRITY          → PASS
G06_ALIGNMENT                 → PASS
G07_LINEAGE                   → PASS
G08_FEATURE_SAFETY            → PASS
G09_FINAL_TEST_ISOLATION      → PASS
G10_METRIC_IMPLEMENTATION     → PASS
G11_SHARED_RUNNER             → PASS
G12_EXPERIMENT_SCHEMA         → PASS
G13_MODELING_INPUT_REGISTRY   → PASS
G14_CONTRACT_PERSISTENCE      → PASS
```

Overall runtime result:

`M5.2 OVERALL TECHNICAL GATE: PASS`

Blocking issue sau runtime review:

`NONE`

# 24. Findings M5.2

## M5.2-F01 — Canonical artifact handoff is usable

Observed:

Toàn bộ canonical M4.7 modeling artifacts tồn tại, load thành công và khớp manifest.

Interpretation:

M5 không cần rebuild raw feature/preprocessing pipeline trước baseline modeling.

Implication:

M5.3 có thể bắt đầu trực tiếp từ persisted matrices.

Status:

`VERIFIED`

---

## M5.2-F02 — Current artifacts are post-scaler-fix artifacts

Observed:

Exact `nnz`/density signature khớp runtime-reviewed M4.7 artifacts:

- W_LONG density ≈ `19.211867%`;
- W_SHORT density ≈ `19.208067%`;
- VALIDATION density ≈ `19.203339%`.

Interpretation:

Không có evidence rằng notebook đang dùng stale matrix từng tồn tại trước numeric-scaler fix.

Implication:

Hai historical numeric features không bị silent-collapse theo lỗi M4.7 cũ.

Status:

`VERIFIED`

---

## M5.2-F03 — Modeling schema is stable

Observed:

Tất cả matrices có:

`47 columns`

`CSR`

`float32`

Feature names:

`47 / 47 unique`

Interpretation:

W_LONG, W_SHORT và corresponding validation representations dùng cùng baseline schema.

Implication:

Các baseline model có thể được so sánh mà không thay feature width/order.

Status:

`VERIFIED`

---

## M5.2-F04 — Target and row alignment are intact

Observed:

Target row/fraud counts khớp canonical contract.

X/y alignment:

`PASS`

Lineage:

`PASS`

W_SHORT lineage subset W_LONG:

`PASS`

Interpretation:

Không phát hiện row-shift, target-shift hoặc subset inconsistency trong handoff.

Status:

`VERIFIED`

---

## M5.2-F05 — Feature safety remains compatible with M4

Observed:

Raw target, raw identifiers, Errors? và raw Timestamp không xuất hiện trong feature schema.

Behavioral companion:

`bool__has_prior_card_history`

được giữ hợp lệ.

Interpretation:

M5.2 không làm suy yếu M4 leakage/identifier guardrails.

Status:

`VERIFIED`

---

## M5.2-F06 — FINAL TEST remains isolated

Observed:

- không đăng ký FINAL TEST artifact;
- `manifest.final_test_used = false`;
- shared modeling registry chỉ có W_LONG/W_SHORT.

Interpretation:

M5.2 không sử dụng FINAL TEST trong infrastructure development.

Status:

`VERIFIED`

---

## M5.2-F07 — Canonical metric implementation is correct on controlled test

Observed:

Toy confusion matrix cho expected metric values chính xác.

Interpretation:

Metric function có semantics phù hợp positive class fraud.

Implication:

M5.3–M5.5 có thể dùng chung metric implementation, giảm nguy cơ mỗi notebook tính metric khác nhau.

Status:

`VERIFIED`

---

## M5.2-F08 — Shared runner mechanics are valid

Observed:

Runner self-test hoàn thành fit/predict/probability/metrics/warning capture/persistence.

Interpretation:

Shared runner đủ chức năng kỹ thuật cho model-specific baseline notebooks.

Caveat:

Toy score không phải project performance.

Status:

`VERIFIED`

---

## M5.2-F09 — M5.1 experiment-log contract is covered

Observed:

Experiment-result schema có `42` fields và bao gồm các governance fields bắt buộc từ M5.1.

Interpretation:

Model-specific experiments sẽ không thiếu model ID/config, random state, errors, integrity result, decision hoặc next action.

Status:

`VERIFIED`

---

## M5.2-F10 — Baseline policy is encoded in the runner

Observed:

`IMBALANCE_STRATEGY = NONE`

`DEFAULT_RANDOM_STATE = 42`

`BASELINE_THRESHOLD_POLICY = DEFAULT_MODEL_DECISION_RULE`

Interpretation:

Core M5 runner không tự đưa class-weight/resampling/threshold tuning vào baseline.

Status:

`VERIFIED`

---

## M5.2-F11 — Infrastructure is ready for Logistic Regression baseline

Observed:

Artifacts, inputs, metrics, runner, schema và persistence đều PASS.

Interpretation:

Không còn infrastructure blocker trước M5.3.

Status:

`READY FOR M5.3`

# 25. Decision Log M5.2

## M5.2-D01 — Canonical modeling artifact source

Decision:

M5 baseline modeling sử dụng canonical M4.7 persisted artifacts tại:

`data/processed/m4_07_baseline_ready`

Không rebuild representation bằng logic khác.

Status:

`LOCKED`

---

## M5.2-D02 — Current artifact validity

Decision:

Bộ artifacts có exact shape/dtype/nnz signature hiện tại được công nhận là canonical baseline-ready artifacts cho M5.

Stale pre-scaler-fix artifacts:

`NOT AUTHORIZED`

Status:

`LOCKED`

---

## M5.2-D03 — Matrix schema

Decision:

Baseline matrix contract:

`47-column CSR float32`

Feature names/order kế thừa M4.7.

Status:

`LOCKED`

---

## M5.2-D04 — Target contract

Decision:

Target:

`int8`

Positive class:

`fraud = 1`

Target luôn lưu riêng khỏi X.

Status:

`INHERITED — VERIFIED — LOCKED`

---

## M5.2-D05 — Lineage policy

Decision:

Lineage tiếp tục lưu riêng khỏi classifier matrix.

Không đưa `raw_row_id` vào X.

Status:

`INHERITED — VERIFIED — LOCKED`

---

## M5.2-D06 — Shared metric implementation

Decision:

M5.3–M5.5 sử dụng chung canonical function:

`compute_binary_metrics`

Metric bundle:

- F1_fraud;
- Recall_fraud;
- Precision_fraud;
- Accuracy reference;
- TP/FP/FN/TN;
- predicted-positive count/rate.

Status:

`LOCKED FOR M5 BASELINE`

---

## M5.2-D07 — Shared baseline runner

Decision:

Runner version:

`M5.2-shared-runner-v1`

M5.3–M5.5 phải sử dụng cùng runner semantics hoặc implementation tương đương được audit.

Status:

`LOCKED FOR M5 BASELINE`

---

## M5.2-D08 — Probability/risk-score extraction

Decision:

Khi estimator hỗ trợ:

ưu tiên `predict_proba` và lấy score của positive class `1`.

Nếu không có `predict_proba` nhưng có `decision_function`:

giữ decision score.

Status:

`LOCKED`

---

## M5.2-D09 — Imbalance baseline

Decision:

Core baseline runner chỉ cho phép:

`IMBALANCE_STRATEGY = NONE`

Class weight/resampling không thuộc M5 core baseline.

Status:

`INHERITED — LOCKED`

---

## M5.2-D10 — Random-state policy

Decision:

Default:

`RANDOM_STATE = 42`

khi operation/model có randomness.

Random state là reproducibility metadata, không phải tuning target.

Status:

`INHERITED — LOCKED`

---

## M5.2-D11 — Baseline threshold policy

Decision:

`DEFAULT_MODEL_DECISION_RULE`

được dùng làm baseline class-decision policy.

Đây:

`NOT FINAL THRESHOLD`

Final threshold vẫn:

`OPEN`

Status:

`LOCKED FOR BASELINE COMPARISON`

---

## M5.2-D12 — FINAL TEST isolation

Decision:

M5 modeling registry không có FINAL TEST input.

`FINAL TEST ACCESS = NO`

Status:

`INHERITED — VERIFIED — LOCKED`

---

## M5.2-D13 — Experiment result schema

Decision:

Official M5 baseline run schema gồm `42` required fields và phải bao phủ M5.1 canonical experiment-log contract.

Status:

`LOCKED`

---

## M5.2-D14 — Shared modeling inputs

Decision:

Authorized modeling-input registry keys:

- `W_LONG`;
- `W_SHORT`.

Không có FINAL TEST key.

Status:

`LOCKED`

---

## M5.2-D15 — Toy self-test interpretation

Decision:

Runner self-test chỉ là infrastructure verification.

Các metric từ `RunnerContractStub`:

`NOT MODEL PERFORMANCE EVIDENCE`

Status:

`LOCKED`

---

## M5.2-D16 — M5.3 handoff

Decision:

Không còn artifact/runner/metric/schema blocker trước Logistic Regression baseline.

Status:

`READY`

# 26. M5.2 Gate

## G01 — Required artifacts available

Evidence:

12/12 required files tồn tại và non-empty.

Result:

`PASS`

---

## G02 — Manifest/schema compatibility

Evidence:

- pipeline version đúng;
- 47 columns;
- CSR float32;
- target int8;
- feature-name file đúng;
- canonical row/fraud counts đúng.

Result:

`PASS`

---

## G03 — Matrix integrity

Evidence:

- expected shapes;
- CSR format;
- float32;
- finite sparse data.

Result:

`PASS`

---

## G04 — Stale-artifact protection

Evidence:

Exact current `nnz` signatures khớp post-scaler-fix artifacts.

Result:

`PASS`

---

## G05 — Target integrity

Evidence:

Binary `{0,1}`, int8, canonical row/fraud counts.

Result:

`PASS`

---

## G06 — X/y alignment

Evidence:

Mọi TRAIN/VALIDATION matrix khớp target length và 47-column schema.

Result:

`PASS`

---

## G07 — Lineage integrity

Evidence:

- int64 lineage;
- unique/strict increasing;
- W_SHORT subset W_LONG.

Result:

`PASS`

---

## G08 — Feature safety

Evidence:

Không phát hiện target/raw identifier/Errors?/raw Timestamp trong classifier schema.

Result:

`PASS`

---

## G09 — FINAL TEST isolation

Evidence:

- không đăng ký FINAL TEST artifact;
- manifest `final_test_used = false`;
- modeling registry không có FINAL TEST.

Result:

`PASS`

---

## G10 — Canonical metric implementation

Evidence:

Controlled metric unit test trả đúng expected values.

Result:

`PASS`

---

## G11 — Shared runner

Evidence:

Runner infrastructure self-test PASS, bao gồm risk-score extraction và persistence.

Result:

`PASS`

---

## G12 — Experiment schema

Evidence:

42 required fields và M5.1 canonical governance fields đều được cover.

Result:

`PASS`

---

## G13 — Modeling-input registry

Evidence:

Authorized keys chỉ gồm:

`W_LONG`

`W_SHORT`

Result:

`PASS`

---

## G14 — Contract persistence

Evidence:

`m5_02_modeling_contract.json`

được save và reload thành công; schema coverage flag = true.

Result:

`PASS`

---

## Overall M5.2 Gate

```text
G01  PASS
G02  PASS
G03  PASS
G04  PASS
G05  PASS
G06  PASS
G07  PASS
G08  PASS
G09  PASS
G10  PASS
G11  PASS
G12  PASS
G13  PASS
G14  PASS
```

Overall:

`M5.2 — PASS`

Blocking issue:

`NONE`

# 27. Kết luận M5.2

M5.2 đã hoàn thành mục tiêu chuyển handoff M4 thành một modeling infrastructure có thể tái sử dụng cho các baseline classifier.

Runtime evidence xác nhận:

- canonical M4.7 artifacts tồn tại và load được;
- current matrices là bộ post-scaler-fix hợp lệ;
- W_LONG/W_SHORT/VALIDATION shapes và target counts khớp contract;
- 47-column CSR float32 schema ổn định;
- X/y/lineage alignment đúng;
- raw target/identifier leakage không xuất hiện trong feature schema;
- FINAL TEST chưa tham gia M5 development;
- canonical fraud metric implementation đã được unit-test;
- shared experiment runner đã được infrastructure self-test;
- experiment-result schema đã bao phủ M5.1 canonical log contract;
- shared modeling registry chỉ chứa W_LONG và W_SHORT;
- M5.2 machine-readable contract đã được persist thành công.

M5.2 không tạo model-performance claim.

Đặc biệt:

`RunnerContractStub F1 = 0.0`

chỉ là kết quả toy infrastructure self-test, không phải evidence về Logistic Regression, Decision Tree, Random Forest hoặc chất lượng dataset.

M5.2 cũng chưa quyết định:

- W_LONG hay W_SHORT tốt hơn;
- Logistic Regression có tốt hơn model khác hay không;
- Decision Tree có tốt hơn hay không;
- Random Forest có tốt hơn hay không;
- class-weight/resampling có cần hay không;
- hyperparameter nào tốt nhất;
- final threshold;
- final model;
- FINAL TEST performance.

Các câu hỏi này vẫn cần model experiment theo các milestone tiếp theo.

Trạng thái cuối:

```text
M5.2 — PASS

Artifact Handoff — VERIFIED

Current M4.7 Artifacts — VERIFIED

Canonical Metric Contract — LOCKED FOR M5 BASELINE

Shared Baseline Runner v1 — LOCKED

Experiment Result Schema — LOCKED

W_LONG Modeling Input — READY

W_SHORT Modeling Input — READY

Imbalance Strategy — NONE

Baseline Threshold Policy — DEFAULT MODEL DECISION RULE

Final Threshold — OPEN

FINAL TEST — PROTECTED

Blocking Issue — NONE

READY FOR M5.3
```

Bước tiếp theo:

`M5.3 — Logistic Regression baseline`

M5.3 phải khóa exact Logistic Regression baseline configuration trước khi đọc comparative validation result và chạy controlled pair:

`LR-LONG`

`LR-SHORT`.
