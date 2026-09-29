# M5.5 — Random Forest baseline

Milestone:

`M5 — Modeling baseline`

Substep:

`M5.5 — Random Forest baseline`

Official runs:

`M5-RF-SHORT-B01`

`M5-RF-LONG-B01`

Mục tiêu:

> Tạo một Random Forest baseline reproducible trên cùng canonical M4.7 representation, cùng W_SHORT/W_LONG protocol, cùng metric contract và không can thiệp class imbalance.

M5.5:

- khóa exact Random Forest baseline config **trước runtime result**;
- dùng cùng config cho W_SHORT và W_LONG;
- dùng `IMBALANCE_STRATEGY = NONE`;
- giữ `predict_proba` làm probability/risk-score evidence;
- dùng estimator default `predict()` làm baseline decision rule;
- không search `n_estimators`;
- không tuning `max_depth`;
- không tuning `max_features`;
- không tuning `min_samples_leaf`;
- không class weighting;
- không threshold tuning;
- không temporal CV;
- không dùng FINAL TEST;
- không tự chọn training-window winner;
- không tự chọn model-family winner.

M5.5 chỉ được khóa sau runtime review.

## 1. Contract kế thừa

M5.5 kế thừa trực tiếp:

```text
M4.7:
canonical persisted baseline-ready matrices

M5.1:
Modeling Charter / baseline protocol

M5.2:
artifact + metric + experiment-result contract

M5.3:
Logistic Regression baseline evidence

M5.4:
Decision Tree baseline evidence
```

Canonical invariants:

```text
Feature schema:
47-column CSR float32

Target:
int8
fraud = 1

Training windows:
W_SHORT
W_LONG

Validation:
same external VALIDATION

Imbalance strategy:
NONE

Primary metric:
F1_fraud

Secondary:
Recall_fraud
Precision_fraud

Mandatory:
TP / FP / FN / TN
predicted-positive count/rate

Accuracy:
reference only

Probability:
preserve when model supports

FINAL TEST:
NO ACCESS
```

M5.1 để exact Random Forest configuration ở trạng thái `OPEN — M5.5`; M5.5 phải khóa config trước comparative result.

## 2. Exact Random Forest baseline configuration

M5.5 khóa trước result:

```text
Estimator:
sklearn.ensemble.RandomForestClassifier

n_estimators:
100

criterion:
gini

max_depth:
None

min_samples_split:
2

min_samples_leaf:
1

min_weight_fraction_leaf:
0.0

max_features:
sqrt

max_leaf_nodes:
None

min_impurity_decrease:
0.0

bootstrap:
True

oob_score:
False

class_weight:
None

ccp_alpha:
0.0

max_samples:
None

random_state:
42

n_jobs:
-1

verbose:
0

warm_start:
False
```

### Vai trò của cấu hình này

Đây là một baseline gần mặc định của scikit-learn, không phải tuned Random Forest.

`n_estimators=100` được khóa như ensemble-size reference trước khi đọc validation result.

`max_depth=None` và `min_samples_leaf=1` giữ tree complexity ở baseline unpruned state.

`max_features="sqrt"` và `bootstrap=True` giữ cơ chế Random Forest điển hình: mỗi cây nhận bootstrap sample và chỉ xem một subset feature tại split.

`class_weight=None` giữ đúng no-intervention baseline.

`n_jobs=-1` là execution policy để tận dụng các CPU core có sẵn. Nó không được xem là model-quality tuning.

Không được chạy 20/100/300 trees rồi chọn score cao nhất trong M5.5.

Systematic hyperparameter tuning:

`DEFERRED TO M7`

## 3. Computational guardrail

Random Forest có thể nặng hơn Decision Tree đáng kể vì fit nhiều cây.

W_LONG có hơn:

`6.8 triệu rows`

Protocol:

```text
Nếu resource issue:
STOP

Không:
- subsample W_LONG;
- chỉ giữ W_SHORT;
- giảm matrix rows;
- đổi feature set;
- dùng config khác nhau giữa hai windows.

Sau đó:
ghi computational finding
→ đề xuất controlled workaround
→ khóa workaround trước result
→ rerun pair công bằng nếu cần
```

Notebook chạy W_SHORT trước để phát hiện implementation/resource issue sớm.

Nếu W_SHORT hoàn thành sạch, notebook mới chạy W_LONG.

## 4. Runtime order

```text
environment / CPU visibility
        ↓
locate M4.7 + M5.2 contract
        ↓
load canonical matrices
        ↓
input-integrity preflight
        ↓
define canonical metrics
        ↓
define RF runner
        ↓
freeze RF-B01 config
        ↓
W_SHORT official run
        ↓
technical + forest-structure + probability gate
        ↓
W_LONG official run
        ↓
technical + forest-structure + probability gate
        ↓
controlled-pair gate
        ↓
metric/result integrity
        ↓
persist predictions/probabilities/summaries
        ↓
round-trip + pair manifest
        ↓
FINAL TEST isolation
        ↓
M5.5 technical gate
        ↓
runtime review
```


```python

from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import hashlib
import json
import os
import platform
import sys
import time
import warnings

import numpy as np

try:
    from scipy import sparse
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Thiếu scipy. Cài scipy + scikit-learn, "
        "Restart Kernel rồi Run All."
    ) from exc

try:
    import sklearn
    from sklearn.base import clone
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Thiếu scikit-learn. Cài scipy + scikit-learn, "
        "Restart Kernel rồi Run All."
    ) from exc


print("Python:")
print(sys.version)

print("\nExecutable:")
print(sys.executable)

print("\nPlatform:")
print(platform.platform())

print("\nLogical CPU count:")
print(os.cpu_count())

print("\nNumPy:")
print(np.__version__)

print("\nscikit-learn:")
print(sklearn.__version__)

```

    Python:
    3.14.6 (main, Jun 10 2026, 10:03:53) [Clang 21.0.0 (clang-2100.0.123.102)]
    
    Executable:
    /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/.venv/bin/python
    
    Platform:
    macOS-26.6.2-arm64-arm-64bit-Mach-O
    
    Logical CPU count:
    8
    
    NumPy:
    2.5.3
    
    scikit-learn:
    1.9.1


## 5. Locate canonical artifacts và M5.2 contract


```python

M4_ARTIFACT_RELATIVE_DIR = (
    Path("data")
    / "processed"
    / "m4_07_baseline_ready"
)

M5_02_CONTRACT_RELATIVE_PATH = (
    Path("data")
    / "processed"
    / "m5_02_modeling_contract"
    / "m5_02_modeling_contract.json"
)

M5_05_OUTPUT_RELATIVE_DIR = (
    Path("data")
    / "processed"
    / "m5_05_random_forest_baseline"
)


candidate_roots = [
    Path.cwd(),
    *list(Path.cwd().parents)[:5],
]

PROJECT_ROOT = None

for candidate in candidate_roots:
    candidate = candidate.resolve()

    if (
        (
            candidate
            / M4_ARTIFACT_RELATIVE_DIR
            / "manifest.json"
        ).exists()
        and
        (
            candidate
            / M5_02_CONTRACT_RELATIVE_PATH
        ).exists()
    ):
        PROJECT_ROOT = candidate
        break


if PROJECT_ROOT is None:
    raise FileNotFoundError(
        "Không tìm thấy PROJECT_ROOT chứa "
        "M4.7 manifest và M5.2 modeling contract."
    )


ARTIFACT_DIR = (
    PROJECT_ROOT
    / M4_ARTIFACT_RELATIVE_DIR
)

M5_02_CONTRACT_PATH = (
    PROJECT_ROOT
    / M5_02_CONTRACT_RELATIVE_PATH
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / M5_05_OUTPUT_RELATIVE_DIR
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


print("PROJECT_ROOT:")
print(PROJECT_ROOT)

print("\nARTIFACT_DIR:")
print(ARTIFACT_DIR)

print("\nM5_02_CONTRACT_PATH:")
print(M5_02_CONTRACT_PATH)

print("\nOUTPUT_DIR:")
print(OUTPUT_DIR)

```

    PROJECT_ROOT:
    /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening
    
    ARTIFACT_DIR:
    /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/processed/m4_07_baseline_ready
    
    M5_02_CONTRACT_PATH:
    /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/processed/m5_02_modeling_contract/m5_02_modeling_contract.json
    
    OUTPUT_DIR:
    /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/processed/m5_05_random_forest_baseline


## 6. Canonical constants


```python

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

M4_PIPELINE_VERSION = "M4.7-baseline-v1"
M5_RUNNER_VERSION = "M5.2-shared-runner-v1"

IMBALANCE_STRATEGY = "NONE"
DEFAULT_RANDOM_STATE = 42

BASELINE_THRESHOLD_POLICY = (
    "DEFAULT_MODEL_DECISION_RULE"
)

FEATURE_VERSION = "Feature Specification v1.0"
PREPROCESSING_VERSION = "Preprocessing Specification v1.0"
MATRIX_SCHEMA_VERSION = "Baseline Matrix Schema v1.0"

RF_MODEL_FAMILY = "Random Forest"
RF_MODEL_ID = "SKLEARN_RANDOM_FOREST_CLASSIFIER"

RF_CONFIG_ID = (
    "RF-B01-100-GINI-SQRT-BOOTSTRAP"
)

RF_SHORT_EXPERIMENT_ID = "M5-RF-SHORT-B01"
RF_LONG_EXPERIMENT_ID = "M5-RF-LONG-B01"


print("Model family:")
print(RF_MODEL_FAMILY)

print("\nConfig ID:")
print(RF_CONFIG_ID)

print("\nImbalance strategy:")
print(IMBALANCE_STRATEGY)

print("\nThreshold policy:")
print(BASELINE_THRESHOLD_POLICY)

```

    Model family:
    Random Forest
    
    Config ID:
    RF-B01-100-GINI-SQRT-BOOTSTRAP
    
    Imbalance strategy:
    NONE
    
    Threshold policy:
    DEFAULT_MODEL_DECISION_RULE


## 7. Load manifest, feature names và M5.2 contract


```python

with open(
    ARTIFACT_DIR / "manifest.json",
    "r",
    encoding="utf-8",
) as file:
    manifest = json.load(file)


with open(
    ARTIFACT_DIR / "feature_names.json",
    "r",
    encoding="utf-8",
) as file:
    feature_names = json.load(file)


with open(
    M5_02_CONTRACT_PATH,
    "r",
    encoding="utf-8",
) as file:
    m5_02_contract = json.load(file)


assert manifest["pipeline_version"] == M4_PIPELINE_VERSION
assert manifest["output_width"] == EXPECTED_OUTPUT_WIDTH
assert manifest["matrix_dtype"] == "float32"
assert str(manifest["matrix_format"]).upper() == "CSR"
assert manifest["target_dtype"] == "int8"
assert manifest["raw_identifiers_in_X"] is False
assert manifest["target_in_X"] is False
assert manifest["final_test_used"] is False

assert len(feature_names) == EXPECTED_OUTPUT_WIDTH
assert len(set(feature_names)) == EXPECTED_OUTPUT_WIDTH

assert m5_02_contract["runner_version"] == M5_RUNNER_VERSION
assert m5_02_contract["input_pipeline_version"] == M4_PIPELINE_VERSION
assert m5_02_contract["feature_count"] == EXPECTED_OUTPUT_WIDTH
assert m5_02_contract["matrix_format"] == "CSR"
assert m5_02_contract["matrix_dtype"] == "float32"
assert m5_02_contract["target_dtype"] == "int8"
assert m5_02_contract["imbalance_strategy"] == IMBALANCE_STRATEGY
assert m5_02_contract["random_state_default"] == DEFAULT_RANDOM_STATE

assert (
    m5_02_contract["threshold_policy"]
    == BASELINE_THRESHOLD_POLICY
)

assert m5_02_contract["final_test_access_allowed"] is False
assert m5_02_contract["experiment_schema_covers_m5_1"] is True


print("M4 pipeline:")
print(manifest["pipeline_version"])

print("\nM5.2 runner:")
print(m5_02_contract["runner_version"])

print("\nFeature count:")
print(len(feature_names))

print("\nM5.5 MANIFEST / CONTRACT GATE: PASS")

```

    M4 pipeline:
    M4.7-baseline-v1
    
    M5.2 runner:
    M5.2-shared-runner-v1
    
    Feature count:
    47
    
    M5.5 MANIFEST / CONTRACT GATE: PASS


## 8. Load modeling matrices và targets

Không load FINAL TEST.

Không rebuild preprocessing.


```python

load_start = time.perf_counter()

X_train_w_short = sparse.load_npz(
    ARTIFACT_DIR / "X_train_w_short.npz"
)

X_train_w_long = sparse.load_npz(
    ARTIFACT_DIR / "X_train_w_long.npz"
)

X_validation_w_short = sparse.load_npz(
    ARTIFACT_DIR / "X_validation_w_short.npz"
)

X_validation_w_long = sparse.load_npz(
    ARTIFACT_DIR / "X_validation_w_long.npz"
)

y_train_w_short = np.load(
    ARTIFACT_DIR / "y_train_w_short.npy",
    allow_pickle=False,
)

y_train_w_long = np.load(
    ARTIFACT_DIR / "y_train_w_long.npy",
    allow_pickle=False,
)

y_validation = np.load(
    ARTIFACT_DIR / "y_validation.npy",
    allow_pickle=False,
)

load_seconds = (
    time.perf_counter()
    - load_start
)


print("Artifact load seconds:")
print(round(load_seconds, 2))

print("\nM5.5 ARTIFACT LOAD: COMPLETE")

```

    Artifact load seconds:
    0.75
    
    M5.5 ARTIFACT LOAD: COMPLETE


## 9. Input-integrity preflight


```python

matrices = {
    "X_train_w_short": X_train_w_short,
    "X_train_w_long": X_train_w_long,
    "X_validation_w_short": X_validation_w_short,
    "X_validation_w_long": X_validation_w_long,
}

expected_shapes = {
    "X_train_w_short": (
        EXPECTED_W_SHORT_ROWS,
        EXPECTED_OUTPUT_WIDTH,
    ),
    "X_train_w_long": (
        EXPECTED_W_LONG_ROWS,
        EXPECTED_OUTPUT_WIDTH,
    ),
    "X_validation_w_short": (
        EXPECTED_VALIDATION_ROWS,
        EXPECTED_OUTPUT_WIDTH,
    ),
    "X_validation_w_long": (
        EXPECTED_VALIDATION_ROWS,
        EXPECTED_OUTPUT_WIDTH,
    ),
}


for name, matrix in matrices.items():
    assert sparse.isspmatrix_csr(matrix)
    assert matrix.dtype == np.float32
    assert matrix.shape == expected_shapes[name]
    assert matrix.nnz == EXPECTED_NNZ[name]
    assert np.isfinite(matrix.data).all()


assert y_train_w_short.dtype == np.int8
assert y_train_w_long.dtype == np.int8
assert y_validation.dtype == np.int8

assert len(y_train_w_short) == EXPECTED_W_SHORT_ROWS
assert len(y_train_w_long) == EXPECTED_W_LONG_ROWS
assert len(y_validation) == EXPECTED_VALIDATION_ROWS

assert int(y_train_w_short.sum()) == EXPECTED_W_SHORT_FRAUD
assert int(y_train_w_long.sum()) == EXPECTED_W_LONG_FRAUD
assert int(y_validation.sum()) == EXPECTED_VALIDATION_FRAUD

assert set(np.unique(y_train_w_short).tolist()) <= {0, 1}
assert set(np.unique(y_train_w_long).tolist()) <= {0, 1}
assert set(np.unique(y_validation).tolist()) <= {0, 1}


print("W_SHORT:")
print(
    X_train_w_short.shape,
    X_train_w_short.dtype,
    "fraud=",
    int(y_train_w_short.sum()),
)

print("\nW_LONG:")
print(
    X_train_w_long.shape,
    X_train_w_long.dtype,
    "fraud=",
    int(y_train_w_long.sum()),
)

print("\nVALIDATION:")
print(
    X_validation_w_short.shape,
    "fraud=",
    int(y_validation.sum()),
)

print("\nM5.5 INPUT INTEGRITY GATE: PASS")

```

    W_SHORT:
    (1721615, 47) float32 fraud= 2491
    
    W_LONG:
    (6855270, 47) float32 fraud= 9606
    
    VALIDATION:
    (712458, 47) fraud= 1052
    
    M5.5 INPUT INTEGRITY GATE: PASS


## 10. Canonical metric implementation


```python

def compute_binary_metrics(
    y_true,
    y_pred,
):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    assert y_true.ndim == 1
    assert y_pred.ndim == 1
    assert len(y_true) == len(y_pred)

    assert set(np.unique(y_true).tolist()) <= {0, 1}
    assert set(np.unique(y_pred).tolist()) <= {0, 1}

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    tn, fp, fn, tp = cm.ravel()

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

        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),

        "predicted_positive_count":
            predicted_positive_count,

        "predicted_positive_rate":
            float(
                predicted_positive_rate
            ),

        "validation_rows":
            int(len(y_true)),
    }

```

## 11. Random Forest runner

Ngoài canonical metric bundle, runner ghi thêm forest diagnostics:

```text
estimator_count
total_nodes
mean_nodes_per_tree
min / mean / max tree depth
total_leaves
mean_leaves_per_tree
nonzero_feature_importance_count
max_feature_importance
```

Diagnostics mô tả computational/model complexity.

Không dùng chúng để tự tuning trong M5.5.


```python

def _json_safe_value(value):
    if isinstance(
        value,
        (str, int, float, bool, type(None)),
    ):
        return value

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, (list, tuple)):
        return [
            _json_safe_value(item)
            for item
            in value
        ]

    if isinstance(value, dict):
        return {
            str(key): _json_safe_value(item)
            for key, item
            in value.items()
        }

    return repr(value)


@dataclass
class ExperimentRunOutput:
    summary: Dict[str, Any]
    y_pred: np.ndarray
    risk_score: Optional[np.ndarray]
    warning_messages: List[str]


def run_rf_baseline(
    *,
    experiment_id,
    estimator,
    X_train,
    y_train,
    X_validation,
    y_validation,
    training_window_id,
):
    assert training_window_id in {
        "W_SHORT",
        "W_LONG",
    }

    assert sparse.isspmatrix_csr(X_train)
    assert sparse.isspmatrix_csr(X_validation)

    model = clone(estimator)

    with warnings.catch_warnings(
        record=True
    ) as caught_warnings:
        warnings.simplefilter("always")

        fit_start = time.perf_counter()

        model.fit(
            X_train,
            y_train,
        )

        fit_seconds = (
            time.perf_counter()
            - fit_start
        )

        prediction_start = time.perf_counter()

        y_pred = model.predict(
            X_validation
        ).astype(
            np.int8
        )

        probability = model.predict_proba(
            X_validation
        )

        prediction_seconds = (
            time.perf_counter()
            - prediction_start
        )

    classes = np.asarray(
        model.classes_
    )

    positive_positions = np.flatnonzero(
        classes == 1
    )

    assert len(positive_positions) == 1

    risk_score = probability[
        :,
        positive_positions[0],
    ].astype(
        np.float32
    )

    warning_messages = [
        (
            f"{item.category.__name__}: "
            f"{item.message}"
        )
        for item
        in caught_warnings
    ]

    metrics = compute_binary_metrics(
        y_validation,
        y_pred,
    )

    estimators = list(
        model.estimators_
    )

    tree_depths = np.asarray(
        [
            tree.get_depth()
            for tree
            in estimators
        ],
        dtype=np.int64,
    )

    tree_nodes = np.asarray(
        [
            tree.tree_.node_count
            for tree
            in estimators
        ],
        dtype=np.int64,
    )

    tree_leaves = np.asarray(
        [
            tree.get_n_leaves()
            for tree
            in estimators
        ],
        dtype=np.int64,
    )

    importances = np.asarray(
        model.feature_importances_,
        dtype=np.float64,
    )

    forest_diagnostics = {
        "estimator_count":
            int(len(estimators)),

        "total_nodes":
            int(tree_nodes.sum()),

        "mean_nodes_per_tree":
            float(tree_nodes.mean()),

        "min_tree_depth":
            int(tree_depths.min()),

        "mean_tree_depth":
            float(tree_depths.mean()),

        "max_tree_depth":
            int(tree_depths.max()),

        "total_leaves":
            int(tree_leaves.sum()),

        "mean_leaves_per_tree":
            float(tree_leaves.mean()),

        "nonzero_feature_importance_count":
            int(
                np.count_nonzero(
                    importances
                )
            ),

        "max_feature_importance":
            float(importances.max()),

        "feature_importance_sum":
            float(importances.sum()),
    }

    model_params = {
        key: _json_safe_value(value)
        for key, value
        in model.get_params(
            deep=False
        ).items()
    }

    summary = {
        "runner_version":
            M5_RUNNER_VERSION,

        "experiment_id":
            str(experiment_id),

        "run_status":
            "COMPLETED",

        "model_family":
            RF_MODEL_FAMILY,

        "model_id":
            RF_MODEL_ID,

        "model_config_id":
            RF_CONFIG_ID,

        "model_class":
            model.__class__.__name__,

        "model_module":
            model.__class__.__module__,

        "model_params":
            model_params,

        "model_diagnostics":
            forest_diagnostics,

        "training_window_id":
            training_window_id,

        "feature_version":
            FEATURE_VERSION,

        "preprocessing_version":
            PREPROCESSING_VERSION,

        "matrix_schema_version":
            MATRIX_SCHEMA_VERSION,

        "feature_count":
            int(X_train.shape[1]),

        "training_rows":
            int(X_train.shape[0]),

        "training_fraud_rows":
            int(
                np.asarray(
                    y_train
                ).sum()
            ),

        "validation_rows":
            int(X_validation.shape[0]),

        "validation_fraud_rows":
            int(
                np.asarray(
                    y_validation
                ).sum()
            ),

        "imbalance_strategy":
            IMBALANCE_STRATEGY,

        "random_state":
            DEFAULT_RANDOM_STATE,

        "threshold_policy":
            BASELINE_THRESHOLD_POLICY,

        "fit_seconds":
            float(fit_seconds),

        "prediction_seconds":
            float(prediction_seconds),

        "risk_score_available":
            True,

        "risk_score_kind":
            "predict_proba",

        "warning_count":
            len(warning_messages),

        "warnings":
            warning_messages,

        "errors":
            [],

        "final_test_accessed":
            False,

        "integrity_gate_result":
            "PASS",

        "decision":
            "OPEN — REQUIRES M5.5 RUNTIME REVIEW",

        "next_action":
            "REVIEW RANDOM FOREST BASELINE EVIDENCE",

        **metrics,
    }

    return ExperimentRunOutput(
        summary=summary,
        y_pred=y_pred,
        risk_score=risk_score,
        warning_messages=warning_messages,
    )


def assert_clean_rf_run(
    output,
    *,
    expected_estimators,
):
    summary = output.summary

    assert summary[
        "run_status"
    ] == "COMPLETED"

    if summary["warnings"]:
        for warning_message in summary["warnings"]:
            print(
                " -",
                warning_message,
            )

    assert summary[
        "warning_count"
    ] == 0, (
        "Random Forest run có warning cần runtime review."
    )

    diagnostics = summary[
        "model_diagnostics"
    ]

    assert (
        diagnostics[
            "estimator_count"
        ]
        == expected_estimators
    )

    assert diagnostics["total_nodes"] >= expected_estimators
    assert diagnostics["total_leaves"] >= expected_estimators
    assert diagnostics["max_tree_depth"] >= 0
    assert diagnostics["min_tree_depth"] >= 0

    assert np.isclose(
        diagnostics[
            "feature_importance_sum"
        ],
        1.0,
        atol=1e-6,
    ), (
        "Feature importance sum không bằng 1."
    )

    assert summary[
        "final_test_accessed"
    ] is False

    assert summary[
        "errors"
    ] == []

```

## 12. Define Random Forest B01

Baseline config được explicit hóa đầy đủ trước result.


```python

RF_BASELINE_CONFIG = {
    "n_estimators": 100,
    "criterion": "gini",
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "min_weight_fraction_leaf": 0.0,
    "max_features": "sqrt",
    "max_leaf_nodes": None,
    "min_impurity_decrease": 0.0,
    "bootstrap": True,
    "oob_score": False,
    "n_jobs": -1,
    "random_state": DEFAULT_RANDOM_STATE,
    "verbose": 0,
    "warm_start": False,
    "class_weight": None,
    "ccp_alpha": 0.0,
    "max_samples": None,
}


rf_template = RandomForestClassifier(
    **RF_BASELINE_CONFIG
)


actual_params = rf_template.get_params(
    deep=False
)


for key, expected_value in (
    RF_BASELINE_CONFIG.items()
):
    assert (
        actual_params[key]
        == expected_value
    ), (
        f"RF param mismatch: {key}"
    )


assert actual_params["n_estimators"] == 100
assert actual_params["class_weight"] is None
assert actual_params["max_depth"] is None
assert actual_params["max_features"] == "sqrt"
assert actual_params["bootstrap"] is True
assert actual_params["random_state"] == DEFAULT_RANDOM_STATE
assert actual_params["n_jobs"] == -1


print("Random Forest baseline config:")

for key in sorted(
    RF_BASELINE_CONFIG
):
    print(
        f"{key}: "
        f"{RF_BASELINE_CONFIG[key]}"
    )


print(
    "\nExecution parallelism:"
)

print(
    "n_jobs =",
    RF_BASELINE_CONFIG[
        "n_jobs"
    ],
)

print(
    "logical CPU count =",
    os.cpu_count(),
)

print(
    "\nM5.5 RF CONFIG DEFINITION GATE: PASS"
)

```

    Random Forest baseline config:
    bootstrap: True
    ccp_alpha: 0.0
    class_weight: None
    criterion: gini
    max_depth: None
    max_features: sqrt
    max_leaf_nodes: None
    max_samples: None
    min_impurity_decrease: 0.0
    min_samples_leaf: 1
    min_samples_split: 2
    min_weight_fraction_leaf: 0.0
    n_estimators: 100
    n_jobs: -1
    oob_score: False
    random_state: 42
    verbose: 0
    warm_start: False
    
    Execution parallelism:
    n_jobs = -1
    logical CPU count = 8
    
    M5.5 RF CONFIG DEFINITION GATE: PASS


## 13. Freeze B01 config trước result

Config fingerprint được persist trước W_SHORT và W_LONG.


```python

config_lock_payload = {
    "m5_substep":
        "M5.5",

    "model_family":
        RF_MODEL_FAMILY,

    "model_id":
        RF_MODEL_ID,

    "model_config_id":
        RF_CONFIG_ID,

    "config":
        RF_BASELINE_CONFIG,

    "config_role":
        "DEFAULT_STYLE_RANDOM_FOREST_BASELINE",

    "ensemble_size_tuning":
        False,

    "complexity_tuning":
        False,

    "parallelism_policy":
        "ALL_AVAILABLE_CORES_N_JOBS_MINUS_1",

    "feature_version":
        FEATURE_VERSION,

    "preprocessing_version":
        PREPROCESSING_VERSION,

    "matrix_schema_version":
        MATRIX_SCHEMA_VERSION,

    "imbalance_strategy":
        IMBALANCE_STRATEGY,

    "random_state":
        DEFAULT_RANDOM_STATE,

    "threshold_policy":
        BASELINE_THRESHOLD_POLICY,

    "training_windows": [
        "W_SHORT",
        "W_LONG",
    ],

    "validation_role":
        "EXTERNAL_VALIDATION",

    "final_test_access":
        False,

    "systematic_tuning":
        False,
}


config_canonical_json = json.dumps(
    config_lock_payload,
    ensure_ascii=False,
    sort_keys=True,
    separators=(",", ":"),
)


config_fingerprint = hashlib.sha256(
    config_canonical_json.encode(
        "utf-8"
    )
).hexdigest()


config_lock_payload[
    "config_fingerprint_sha256"
] = config_fingerprint


CONFIG_LOCK_PATH = (
    OUTPUT_DIR
    / "m5_05_rf_baseline_config_lock.json"
)


with open(
    CONFIG_LOCK_PATH,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        config_lock_payload,
        file,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )


with open(
    CONFIG_LOCK_PATH,
    "r",
    encoding="utf-8",
) as file:
    reloaded_config = json.load(file)


assert (
    reloaded_config[
        "config_fingerprint_sha256"
    ]
    == config_fingerprint
)

assert (
    reloaded_config[
        "ensemble_size_tuning"
    ]
    is False
)

assert (
    reloaded_config[
        "complexity_tuning"
    ]
    is False
)

assert (
    reloaded_config[
        "systematic_tuning"
    ]
    is False
)

assert (
    reloaded_config[
        "final_test_access"
    ]
    is False
)

assert (
    reloaded_config[
        "imbalance_strategy"
    ]
    == "NONE"
)


print("Config lock:")
print(CONFIG_LOCK_PATH)

print("\nConfig SHA256:")
print(config_fingerprint)

print(
    "\nM5.5 PRE-RESULT CONFIG LOCK GATE: PASS"
)

```

    Config lock:
    /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/processed/m5_05_random_forest_baseline/m5_05_rf_baseline_config_lock.json
    
    Config SHA256:
    99b6610f7fa37cc9274998c3c253c40a46112d6d5a6e2cb525a7e9eae509bef9
    
    M5.5 PRE-RESULT CONFIG LOCK GATE: PASS


# 14. W_SHORT official run

W_SHORT là official baseline run, đồng thời là implementation/resource preflight.

Nếu cell fail do RAM/CPU/resource:

`STOP`

Không giảm số rows hoặc đổi config âm thầm.


```python

print("=" * 72)
print(
    "Starting:",
    RF_SHORT_EXPERIMENT_ID,
)

print(
    "Training rows:",
    f"{EXPECTED_W_SHORT_ROWS:,}",
)

print(
    "Training fraud:",
    f"{EXPECTED_W_SHORT_FRAUD:,}",
)

print(
    "n_estimators:",
    RF_BASELINE_CONFIG[
        "n_estimators"
    ],
)

print(
    "n_jobs:",
    RF_BASELINE_CONFIG[
        "n_jobs"
    ],
)


short_output = run_rf_baseline(
    experiment_id=
        RF_SHORT_EXPERIMENT_ID,
    estimator=
        rf_template,
    X_train=
        X_train_w_short,
    y_train=
        y_train_w_short,
    X_validation=
        X_validation_w_short,
    y_validation=
        y_validation,
    training_window_id=
        "W_SHORT",
)


short_summary = (
    short_output.summary
)

short_diag = short_summary[
    "model_diagnostics"
]


print(
    "\nCompleted:",
    RF_SHORT_EXPERIMENT_ID,
)

print(
    "fit_seconds:",
    round(
        short_summary[
            "fit_seconds"
        ],
        3,
    ),
)

print(
    "prediction_seconds:",
    round(
        short_summary[
            "prediction_seconds"
        ],
        3,
    ),
)

print(
    "warning_count:",
    short_summary[
        "warning_count"
    ],
)

print(
    "estimator_count:",
    short_diag[
        "estimator_count"
    ],
)

print(
    "total_nodes:",
    short_diag[
        "total_nodes"
    ],
)

print(
    "mean_nodes_per_tree:",
    round(
        short_diag[
            "mean_nodes_per_tree"
        ],
        2,
    ),
)

print(
    "tree_depth min/mean/max:",
    (
        short_diag[
            "min_tree_depth"
        ],
        round(
            short_diag[
                "mean_tree_depth"
            ],
            2,
        ),
        short_diag[
            "max_tree_depth"
        ],
    ),
)

print(
    "total_leaves:",
    short_diag[
        "total_leaves"
    ],
)


assert_clean_rf_run(
    short_output,
    expected_estimators=
        RF_BASELINE_CONFIG[
            "n_estimators"
        ],
)


print(
    "\nM5.5 W_SHORT TECHNICAL GATE: PASS"
)

```

    ========================================================================
    Starting: M5-RF-SHORT-B01
    Training rows: 1,721,615
    Training fraud: 2,491
    n_estimators: 100
    n_jobs: -1
    
    Completed: M5-RF-SHORT-B01
    fit_seconds: 40.698
    prediction_seconds: 0.868
    warning_count: 0
    estimator_count: 100
    total_nodes: 451532
    mean_nodes_per_tree: 4515.32
    tree_depth min/mean/max: (33, 41.43, 50)
    total_leaves: 225816
    
    M5.5 W_SHORT TECHNICAL GATE: PASS


## 15. W_SHORT probability/default-rule integrity

Random Forest class prediction được kiểm tra bằng:

`predict() == classes[argmax(predict_proba)]`

Final threshold vẫn OPEN.


```python

short_probability = np.column_stack(
    [
        1.0
        - short_output.risk_score,

        short_output.risk_score,
    ]
)


assert np.isfinite(
    short_output.risk_score
).all()

assert np.all(
    short_output.risk_score >= 0.0
)

assert np.all(
    short_output.risk_score <= 1.0
)


short_argmax_pred = (
    np.argmax(
        short_probability,
        axis=1,
    )
).astype(
    np.int8
)


assert np.array_equal(
    short_argmax_pred,
    short_output.y_pred,
)


print(
    "Risk-score kind:",
    short_summary[
        "risk_score_kind"
    ],
)

print(
    "Final threshold:",
    "OPEN",
)

print(
    "\nM5.5 W_SHORT PROBABILITY / DEFAULT-RULE GATE: PASS"
)

```

    Risk-score kind: predict_proba
    Final threshold: OPEN
    
    M5.5 W_SHORT PROBABILITY / DEFAULT-RULE GATE: PASS


# 16. W_LONG official run

Chỉ chạy sau khi W_SHORT technical/probability gates PASS.

Exact RF-B01 config không đổi.

Nếu W_LONG gặp resource issue:

`STOP`

Không subsample hoặc đổi forest config riêng cho W_LONG.


```python

print("=" * 72)
print(
    "Starting:",
    RF_LONG_EXPERIMENT_ID,
)

print(
    "Training rows:",
    f"{EXPECTED_W_LONG_ROWS:,}",
)

print(
    "Training fraud:",
    f"{EXPECTED_W_LONG_FRAUD:,}",
)

print(
    "n_estimators:",
    RF_BASELINE_CONFIG[
        "n_estimators"
    ],
)

print(
    "n_jobs:",
    RF_BASELINE_CONFIG[
        "n_jobs"
    ],
)


long_output = run_rf_baseline(
    experiment_id=
        RF_LONG_EXPERIMENT_ID,
    estimator=
        rf_template,
    X_train=
        X_train_w_long,
    y_train=
        y_train_w_long,
    X_validation=
        X_validation_w_long,
    y_validation=
        y_validation,
    training_window_id=
        "W_LONG",
)


long_summary = (
    long_output.summary
)

long_diag = long_summary[
    "model_diagnostics"
]


print(
    "\nCompleted:",
    RF_LONG_EXPERIMENT_ID,
)

print(
    "fit_seconds:",
    round(
        long_summary[
            "fit_seconds"
        ],
        3,
    ),
)

print(
    "prediction_seconds:",
    round(
        long_summary[
            "prediction_seconds"
        ],
        3,
    ),
)

print(
    "warning_count:",
    long_summary[
        "warning_count"
    ],
)

print(
    "estimator_count:",
    long_diag[
        "estimator_count"
    ],
)

print(
    "total_nodes:",
    long_diag[
        "total_nodes"
    ],
)

print(
    "mean_nodes_per_tree:",
    round(
        long_diag[
            "mean_nodes_per_tree"
        ],
        2,
    ),
)

print(
    "tree_depth min/mean/max:",
    (
        long_diag[
            "min_tree_depth"
        ],
        round(
            long_diag[
                "mean_tree_depth"
            ],
            2,
        ),
        long_diag[
            "max_tree_depth"
        ],
    ),
)

print(
    "total_leaves:",
    long_diag[
        "total_leaves"
    ],
)


assert_clean_rf_run(
    long_output,
    expected_estimators=
        RF_BASELINE_CONFIG[
            "n_estimators"
        ],
)


print(
    "\nM5.5 W_LONG TECHNICAL GATE: PASS"
)

```

    ========================================================================
    Starting: M5-RF-LONG-B01
    Training rows: 6,855,270
    Training fraud: 9,606
    n_estimators: 100
    n_jobs: -1
    
    Completed: M5-RF-LONG-B01
    fit_seconds: 627.714
    prediction_seconds: 1.494
    warning_count: 0
    estimator_count: 100
    total_nodes: 3066586
    mean_nodes_per_tree: 30665.86
    tree_depth min/mean/max: (44, 49.06, 57)
    total_leaves: 1533343
    
    M5.5 W_LONG TECHNICAL GATE: PASS


## 17. W_LONG probability/default-rule integrity


```python

long_probability = np.column_stack(
    [
        1.0
        - long_output.risk_score,

        long_output.risk_score,
    ]
)


assert np.isfinite(
    long_output.risk_score
).all()

assert np.all(
    long_output.risk_score >= 0.0
)

assert np.all(
    long_output.risk_score <= 1.0
)


long_argmax_pred = (
    np.argmax(
        long_probability,
        axis=1,
    )
).astype(
    np.int8
)


assert np.array_equal(
    long_argmax_pred,
    long_output.y_pred,
)


print(
    "M5.5 W_LONG PROBABILITY / DEFAULT-RULE GATE: PASS"
)

```

    M5.5 W_LONG PROBABILITY / DEFAULT-RULE GATE: PASS


## 18. Controlled-pair configuration equivalence


```python

controlled_equal_fields = [
    "runner_version",
    "model_family",
    "model_id",
    "model_config_id",
    "model_class",
    "model_module",
    "model_params",
    "feature_version",
    "preprocessing_version",
    "matrix_schema_version",
    "feature_count",
    "validation_rows",
    "validation_fraud_rows",
    "imbalance_strategy",
    "random_state",
    "threshold_policy",
    "risk_score_kind",
    "final_test_accessed",
]


for field in controlled_equal_fields:
    assert (
        short_summary[field]
        == long_summary[field]
    ), (
        f"Controlled field mismatch: "
        f"{field}"
    )


assert (
    short_summary[
        "training_window_id"
    ]
    == "W_SHORT"
)

assert (
    long_summary[
        "training_window_id"
    ]
    == "W_LONG"
)

assert (
    short_summary[
        "training_rows"
    ]
    == EXPECTED_W_SHORT_ROWS
)

assert (
    long_summary[
        "training_rows"
    ]
    == EXPECTED_W_LONG_ROWS
)

assert (
    short_summary[
        "training_fraud_rows"
    ]
    == EXPECTED_W_SHORT_FRAUD
)

assert (
    long_summary[
        "training_fraud_rows"
    ]
    == EXPECTED_W_LONG_FRAUD
)


print(
    "Controlled equal fields:",
    len(controlled_equal_fields),
)

print(
    "\nM5.5 CONTROLLED-PAIR CONFIG GATE: PASS"
)

```

    Controlled equal fields: 18
    
    M5.5 CONTROLLED-PAIR CONFIG GATE: PASS


## 19. Metric/result integrity

Chỉ kiểm tra arithmetic/internal consistency.

Không chọn winner.


```python

for summary in [
    short_summary,
    long_summary,
]:
    assert (
        summary[
            "validation_rows"
        ]
        == EXPECTED_VALIDATION_ROWS
    )

    assert (
        summary[
            "validation_fraud_rows"
        ]
        == EXPECTED_VALIDATION_FRAUD
    )

    assert (
        summary["tn"]
        + summary["fp"]
        + summary["fn"]
        + summary["tp"]
        == EXPECTED_VALIDATION_ROWS
    )

    assert (
        summary[
            "predicted_positive_count"
        ]
        == summary["tp"]
        + summary["fp"]
    )

    expected_positive_rate = (
        summary[
            "predicted_positive_count"
        ]
        / EXPECTED_VALIDATION_ROWS
    )

    assert np.isclose(
        summary[
            "predicted_positive_rate"
        ],
        expected_positive_rate,
    )

    for metric_name in [
        "f1_fraud",
        "recall_fraud",
        "precision_fraud",
        "accuracy_reference",
        "predicted_positive_rate",
    ]:
        value = summary[
            metric_name
        ]

        assert np.isfinite(value)
        assert 0.0 <= value <= 1.0

    assert (
        summary[
            "integrity_gate_result"
        ]
        == "PASS"
    )

    assert summary["errors"] == []

    assert (
        summary[
            "final_test_accessed"
        ]
        is False
    )


print(
    "M5.5 METRIC / RESULT INTEGRITY GATE: PASS"
)

```

    M5.5 METRIC / RESULT INTEGRITY GATE: PASS


## 20. Baseline evidence output

In descriptive evidence phục vụ runtime review.

Không gắn nhãn winner.


```python

def print_rf_evidence(
    summary,
):
    diagnostics = summary[
        "model_diagnostics"
    ]

    print(
        "Experiment:",
        summary[
            "experiment_id"
        ],
    )

    print(
        "Training window:",
        summary[
            "training_window_id"
        ],
    )

    print(
        "Training rows:",
        f"{summary['training_rows']:,}",
    )

    print(
        "Training fraud:",
        f"{summary['training_fraud_rows']:,}",
    )

    print(
        "Estimators:",
        diagnostics[
            "estimator_count"
        ],
    )

    print(
        "Total nodes:",
        diagnostics[
            "total_nodes"
        ],
    )

    print(
        "Mean nodes/tree:",
        diagnostics[
            "mean_nodes_per_tree"
        ],
    )

    print(
        "Depth min/mean/max:",
        (
            diagnostics[
                "min_tree_depth"
            ],
            diagnostics[
                "mean_tree_depth"
            ],
            diagnostics[
                "max_tree_depth"
            ],
        ),
    )

    print(
        "Total leaves:",
        diagnostics[
            "total_leaves"
        ],
    )

    print(
        "Nonzero feature importances:",
        diagnostics[
            "nonzero_feature_importance_count"
        ],
    )

    print(
        "F1_fraud:",
        summary[
            "f1_fraud"
        ],
    )

    print(
        "Recall_fraud:",
        summary[
            "recall_fraud"
        ],
    )

    print(
        "Precision_fraud:",
        summary[
            "precision_fraud"
        ],
    )

    print(
        "Accuracy reference:",
        summary[
            "accuracy_reference"
        ],
    )

    print(
        "TP / FP / FN / TN:",
        summary["tp"],
        summary["fp"],
        summary["fn"],
        summary["tn"],
    )

    print(
        "Predicted positive count:",
        summary[
            "predicted_positive_count"
        ],
    )

    print(
        "Predicted positive rate:",
        summary[
            "predicted_positive_rate"
        ],
    )

    print(
        "Fit seconds:",
        summary[
            "fit_seconds"
        ],
    )

    print(
        "Prediction seconds:",
        summary[
            "prediction_seconds"
        ],
    )


print("=" * 72)
print_rf_evidence(
    short_summary
)

print("\n" + "=" * 72)
print_rf_evidence(
    long_summary
)


print("\n" + "=" * 72)

print(
    "Descriptive metric deltas: "
    "W_LONG - W_SHORT"
)

for metric_name in [
    "f1_fraud",
    "recall_fraud",
    "precision_fraud",
    "accuracy_reference",
    "predicted_positive_rate",
]:
    print(
        metric_name,
        (
            long_summary[
                metric_name
            ]
            - short_summary[
                metric_name
            ]
        ),
    )


print(
    "\nTraining-window winner:",
    "OPEN — M5.5 DOES NOT SELECT",
)

print(
    "Model-family winner:",
    "OPEN — M5.5 DOES NOT SELECT",
)

```

    ========================================================================
    Experiment: M5-RF-SHORT-B01
    Training window: W_SHORT
    Training rows: 1,721,615
    Training fraud: 2,491
    Estimators: 100
    Total nodes: 451532
    Mean nodes/tree: 4515.32
    Depth min/mean/max: (33, 41.43, 50)
    Total leaves: 225816
    Nonzero feature importances: 43
    F1_fraud: 0.3664670658682635
    Recall_fraud: 0.2908745247148289
    Precision_fraud: 0.49514563106796117
    Accuracy reference: 0.9985150001824669
    TP / FP / FN / TN: 306 312 746 711094
    Predicted positive count: 618
    Predicted positive rate: 0.0008674195531526069
    Fit seconds: 40.69802879198687
    Prediction seconds: 0.8680003330227919
    
    ========================================================================
    Experiment: M5-RF-LONG-B01
    Training window: W_LONG
    Training rows: 6,855,270
    Training fraud: 9,606
    Estimators: 100
    Total nodes: 3066586
    Mean nodes/tree: 30665.86
    Depth min/mean/max: (44, 49.06, 57)
    Total leaves: 1533343
    Nonzero feature importances: 43
    F1_fraud: 0.15270935960591134
    Recall_fraud: 0.08840304182509506
    Precision_fraud: 0.5602409638554217
    Accuracy reference: 0.9985514935617258
    TP / FP / FN / TN: 93 73 959 711333
    Predicted positive count: 166
    Predicted positive rate: 0.000232996190652642
    Fit seconds: 627.7140797079774
    Prediction seconds: 1.4944120000000112
    
    ========================================================================
    Descriptive metric deltas: W_LONG - W_SHORT
    f1_fraud -0.21375770626235216
    recall_fraud -0.20247148288973382
    precision_fraud 0.06509533278746049
    accuracy_reference 3.649337925892837e-05
    predicted_positive_rate -0.0006344233624999649
    
    Training-window winner: OPEN — M5.5 DOES NOT SELECT
    Model-family winner: OPEN — M5.5 DOES NOT SELECT


## 21. Persist Random Forest baseline evidence


```python

def save_experiment_output(
    output,
    output_dir,
):
    output_dir = Path(
        output_dir
    )

    experiment_id = output.summary[
        "experiment_id"
    ]

    prediction_path = (
        output_dir
        / f"{experiment_id}__y_pred.npy"
    )

    risk_score_path = (
        output_dir
        / f"{experiment_id}__risk_score.npy"
    )

    summary_path = (
        output_dir
        / f"{experiment_id}__summary.json"
    )

    np.save(
        prediction_path,
        output.y_pred,
        allow_pickle=False,
    )

    np.save(
        risk_score_path,
        output.risk_score,
        allow_pickle=False,
    )

    persisted_summary = dict(
        output.summary
    )

    persisted_summary[
        "prediction_file"
    ] = prediction_path.name

    persisted_summary[
        "risk_score_file"
    ] = risk_score_path.name

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


saved_outputs = {
    "W_SHORT":
        save_experiment_output(
            short_output,
            OUTPUT_DIR,
        ),

    "W_LONG":
        save_experiment_output(
            long_output,
            OUTPUT_DIR,
        ),
}


for window_id, paths in (
    saved_outputs.items()
):
    print(
        "\n",
        window_id,
        sep="",
    )

    for key, path in (
        paths.items()
    ):
        print(
            key,
            "→",
            path,
        )

        assert path.exists()
        assert (
            path.stat().st_size
            > 0
        )


print(
    "\nM5.5 PERSISTENCE GATE: PASS"
)

```

    
    W_SHORT
    summary_path → /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/processed/m5_05_random_forest_baseline/M5-RF-SHORT-B01__summary.json
    prediction_path → /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/processed/m5_05_random_forest_baseline/M5-RF-SHORT-B01__y_pred.npy
    risk_score_path → /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/processed/m5_05_random_forest_baseline/M5-RF-SHORT-B01__risk_score.npy
    
    W_LONG
    summary_path → /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/processed/m5_05_random_forest_baseline/M5-RF-LONG-B01__summary.json
    prediction_path → /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/processed/m5_05_random_forest_baseline/M5-RF-LONG-B01__y_pred.npy
    risk_score_path → /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/processed/m5_05_random_forest_baseline/M5-RF-LONG-B01__risk_score.npy
    
    M5.5 PERSISTENCE GATE: PASS


## 22. Saved-artifact round-trip


```python

for window_id, output in [
    ("W_SHORT", short_output),
    ("W_LONG", long_output),
]:
    paths = saved_outputs[
        window_id
    ]

    reloaded_pred = np.load(
        paths[
            "prediction_path"
        ],
        allow_pickle=False,
    )

    reloaded_score = np.load(
        paths[
            "risk_score_path"
        ],
        allow_pickle=False,
    )

    assert np.array_equal(
        reloaded_pred,
        output.y_pred,
    )

    assert np.array_equal(
        reloaded_score,
        output.risk_score,
    )

    assert (
        reloaded_pred.dtype
        == np.int8
    )

    assert (
        reloaded_score.dtype
        == np.float32
    )

    assert (
        len(reloaded_pred)
        == EXPECTED_VALIDATION_ROWS
    )

    assert (
        len(reloaded_score)
        == EXPECTED_VALIDATION_ROWS
    )

    with open(
        paths[
            "summary_path"
        ],
        "r",
        encoding="utf-8",
    ) as file:
        reloaded_summary = (
            json.load(file)
        )

    assert (
        reloaded_summary[
            "experiment_id"
        ]
        == output.summary[
            "experiment_id"
        ]
    )

    assert (
        reloaded_summary[
            "model_config_id"
        ]
        == RF_CONFIG_ID
    )

    assert (
        reloaded_summary[
            "final_test_accessed"
        ]
        is False
    )


print(
    "M5.5 SAVED ARTIFACT ROUND-TRIP GATE: PASS"
)

```

    M5.5 SAVED ARTIFACT ROUND-TRIP GATE: PASS


## 23. Pair manifest


```python

pair_manifest = {
    "m5_substep":
        "M5.5",

    "model_family":
        RF_MODEL_FAMILY,

    "model_id":
        RF_MODEL_ID,

    "model_config_id":
        RF_CONFIG_ID,

    "config_fingerprint_sha256":
        config_fingerprint,

    "runner_version":
        M5_RUNNER_VERSION,

    "feature_version":
        FEATURE_VERSION,

    "preprocessing_version":
        PREPROCESSING_VERSION,

    "matrix_schema_version":
        MATRIX_SCHEMA_VERSION,

    "imbalance_strategy":
        IMBALANCE_STRATEGY,

    "random_state":
        DEFAULT_RANDOM_STATE,

    "threshold_policy":
        BASELINE_THRESHOLD_POLICY,

    "final_threshold_status":
        "OPEN",

    "training_window_winner":
        "OPEN",

    "model_family_winner":
        "OPEN",

    "ensemble_size_tuning":
        False,

    "complexity_tuning":
        False,

    "final_test_accessed":
        False,

    "runs": {
        "W_SHORT": {
            "experiment_id":
                RF_SHORT_EXPERIMENT_ID,

            "summary_file":
                saved_outputs[
                    "W_SHORT"
                ][
                    "summary_path"
                ].name,

            "prediction_file":
                saved_outputs[
                    "W_SHORT"
                ][
                    "prediction_path"
                ].name,

            "risk_score_file":
                saved_outputs[
                    "W_SHORT"
                ][
                    "risk_score_path"
                ].name,
        },

        "W_LONG": {
            "experiment_id":
                RF_LONG_EXPERIMENT_ID,

            "summary_file":
                saved_outputs[
                    "W_LONG"
                ][
                    "summary_path"
                ].name,

            "prediction_file":
                saved_outputs[
                    "W_LONG"
                ][
                    "prediction_path"
                ].name,

            "risk_score_file":
                saved_outputs[
                    "W_LONG"
                ][
                    "risk_score_path"
                ].name,
        },
    },
}


PAIR_MANIFEST_PATH = (
    OUTPUT_DIR
    / "m5_05_rf_pair_manifest.json"
)


with open(
    PAIR_MANIFEST_PATH,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        pair_manifest,
        file,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )


with open(
    PAIR_MANIFEST_PATH,
    "r",
    encoding="utf-8",
) as file:
    reloaded_pair_manifest = (
        json.load(file)
    )


assert (
    reloaded_pair_manifest[
        "config_fingerprint_sha256"
    ]
    == config_fingerprint
)

assert (
    reloaded_pair_manifest[
        "training_window_winner"
    ]
    == "OPEN"
)

assert (
    reloaded_pair_manifest[
        "model_family_winner"
    ]
    == "OPEN"
)

assert (
    reloaded_pair_manifest[
        "ensemble_size_tuning"
    ]
    is False
)

assert (
    reloaded_pair_manifest[
        "complexity_tuning"
    ]
    is False
)

assert (
    reloaded_pair_manifest[
        "final_test_accessed"
    ]
    is False
)


print("Pair manifest:")
print(PAIR_MANIFEST_PATH)

print(
    "\nM5.5 PAIR MANIFEST GATE: PASS"
)

```

    Pair manifest:
    /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/processed/m5_05_random_forest_baseline/m5_05_rf_pair_manifest.json
    
    M5.5 PAIR MANIFEST GATE: PASS


## 24. FINAL TEST isolation


```python

assert manifest[
    "final_test_used"
] is False

assert m5_02_contract[
    "final_test_access_allowed"
] is False

assert short_summary[
    "final_test_accessed"
] is False

assert long_summary[
    "final_test_accessed"
] is False

assert pair_manifest[
    "final_test_accessed"
] is False


for path in OUTPUT_DIR.iterdir():
    assert (
        "final_test"
        not in path.name.lower()
    ), (
        "Phát hiện artifact name liên quan FINAL TEST: "
        f"{path.name}"
    )


print(
    "M5.5 FINAL TEST ISOLATION GATE: PASS"
)

```

    M5.5 FINAL TEST ISOLATION GATE: PASS


## 25. M5.5 overall technical gate


```python

m5_05_gates = {
    "G01_MANIFEST_CONTRACT":
        True,

    "G02_INPUT_INTEGRITY":
        True,

    "G03_RF_CONFIG_DEFINED":
        True,

    "G04_PRE_RESULT_CONFIG_LOCK":
        True,

    "G05_W_SHORT_TECHNICAL":
        True,

    "G06_W_SHORT_PROBABILITY":
        True,

    "G07_W_LONG_TECHNICAL":
        True,

    "G08_W_LONG_PROBABILITY":
        True,

    "G09_CONTROLLED_PAIR":
        True,

    "G10_METRIC_RESULT_INTEGRITY":
        True,

    "G11_PERSISTENCE":
        True,

    "G12_ARTIFACT_ROUND_TRIP":
        True,

    "G13_PAIR_MANIFEST":
        True,

    "G14_FINAL_TEST_ISOLATION":
        True,
}


for gate_name, gate_value in (
    m5_05_gates.items()
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
    m5_05_gates.values()
)


print(
    "\nM5.5 OVERALL TECHNICAL GATE: PASS"
)

```

    G01_MANIFEST_CONTRACT → PASS
    G02_INPUT_INTEGRITY → PASS
    G03_RF_CONFIG_DEFINED → PASS
    G04_PRE_RESULT_CONFIG_LOCK → PASS
    G05_W_SHORT_TECHNICAL → PASS
    G06_W_SHORT_PROBABILITY → PASS
    G07_W_LONG_TECHNICAL → PASS
    G08_W_LONG_PROBABILITY → PASS
    G09_CONTROLLED_PAIR → PASS
    G10_METRIC_RESULT_INTEGRITY → PASS
    G11_PERSISTENCE → PASS
    G12_ARTIFACT_ROUND_TRIP → PASS
    G13_PAIR_MANIFEST → PASS
    G14_FINAL_TEST_ISOLATION → PASS
    
    M5.5 OVERALL TECHNICAL GATE: PASS


# 26. Phân tích và nhận xét runtime M5.5

## 26.1. Execution integrity

Notebook đã chạy đầy đủ:

```text
22 / 22 code cells
execution_count = 1 → 22 liên tục
runtime error = 0
stderr = 0
```

Do đó output hiện tại đủ điều kiện dùng làm runtime evidence cho M5.5.

---

## 26.2. Môi trường và parallelism

Observed runtime environment:

```text
Python:
3.14.6

NumPy:
2.5.3

scikit-learn:
1.9.1

Platform:
macOS arm64

Logical CPU count:
8
```

Random Forest config:

```text
n_jobs = -1
```

Interpretation:

Notebook được cấu hình để scikit-learn sử dụng các CPU core có sẵn cho Random Forest.

Trong quá trình runtime review, người chạy cũng quan sát toàn bộ CPU được sử dụng.

Điều này phù hợp với execution policy đã khóa trước result.

`n_jobs=-1` được xem là execution/parallelism setting, không phải validation-score tuning.

---

## 26.3. Canonical input / contract integrity

Observed:

```text
W_SHORT:
1,721,615 rows
2,491 fraud
47 features
float32 CSR

W_LONG:
6,855,270 rows
9,606 fraud
47 features
float32 CSR

VALIDATION:
712,458 rows
1,052 fraud
47 features
```

Runtime gates:

`M5.5 MANIFEST / CONTRACT GATE: PASS`

`M5.5 INPUT INTEGRITY GATE: PASS`

Nhận xét:

Random Forest được fit trên full canonical matrices.

Không có evidence cho:

- subsampling;
- giảm matrix rows;
- đổi feature set;
- rebuild preprocessing;
- chỉ giữ W_SHORT.

---

## 26.4. Pre-result configuration lock

Official config:

`RF-B01-100-GINI-SQRT-BOOTSTRAP`

```text
n_estimators         = 100
criterion            = gini
max_depth            = None
min_samples_split    = 2
min_samples_leaf     = 1
min_weight_fraction_leaf = 0.0
max_features         = sqrt
max_leaf_nodes       = None
min_impurity_decrease= 0.0
bootstrap            = True
oob_score            = False
class_weight         = None
ccp_alpha            = 0.0
max_samples          = None
random_state         = 42
n_jobs               = -1
verbose              = 0
warm_start           = False
```

Config fingerprint:

`99b6610f7fa37cc9274998c3c253c40a46112d6d5a6e2cb525a7e9eae509bef9`

Runtime gate:

`M5.5 PRE-RESULT CONFIG LOCK GATE: PASS`

Interpretation:

M5.5 không thử nhiều số cây/depth/max_features rồi chọn validation score tốt nhất.

`100 trees` là baseline ensemble-size reference, không phải tuned optimum.

---

## 26.5. W_SHORT technical behavior

Observed:

```text
Experiment:
M5-RF-SHORT-B01

Training rows:
1,721,615

Training fraud:
2,491

n_estimators:
100

fit_seconds:
40.698

prediction_seconds:
0.868

warning_count:
0

estimator_count:
100

total_nodes:
451,532

mean_nodes_per_tree:
4,515.32

tree depth min / mean / max:
33 / 41.43 / 50

total_leaves:
225,816

nonzero feature importances:
43 / 47
```

Gate:

`M5.5 W_SHORT TECHNICAL GATE: PASS`

Nhận xét:

Toàn bộ 100 cây đã được fit.

Không warning.

Forest có cấu trúc lớn nhưng vẫn hoàn thành trong khoảng 41 giây trên full W_SHORT.

---

## 26.6. W_LONG technical behavior

Observed:

```text
Experiment:
M5-RF-LONG-B01

Training rows:
6,855,270

Training fraud:
9,606

n_estimators:
100

fit_seconds:
627.714

prediction_seconds:
1.494

warning_count:
0

estimator_count:
100

total_nodes:
3,066,586

mean_nodes_per_tree:
30,665.86

tree depth min / mean / max:
44 / 49.06 / 57

total_leaves:
1,533,343

nonzero feature importances:
43 / 47
```

Gate:

`M5.5 W_LONG TECHNICAL GATE: PASS`

Nhận xét:

W_LONG cũng fit đủ 100 cây, không warning và không cần controlled workaround.

Runtime khoảng:

`10 phút 28 giây`

chỉ riêng phần W_LONG fit.

---

## 26.7. Vì sao W_LONG tăng thời gian mạnh hơn tỷ lệ số rows

Observed ratios:

```text
Training rows:
W_LONG / W_SHORT ≈ 3.98×

Fit time:
W_LONG / W_SHORT ≈ 15.42×

Total nodes:
W_LONG / W_SHORT ≈ 6.79×

Total leaves:
W_LONG / W_SHORT ≈ 6.79×
```

Interpretation:

Random Forest training cost không chỉ tăng theo số rows.

W_LONG còn tạo các tree lớn hơn rõ rệt:

```text
Mean nodes/tree:

W_SHORT:
4,515.32

W_LONG:
30,665.86
```

và tree depth distribution cũng dịch lên:

```text
W_SHORT:
33 / 41.43 / 50

W_LONG:
44 / 49.06 / 57
```

Do đó việc W_LONG tốn hơn 10 phút không phải dấu hiệu notebook bị treo.

Nó phù hợp với:

- dataset lớn hơn;
- nhiều candidate split phải xét hơn;
- forest lớn hơn nhiều;
- 100 cây đều được fit đầy đủ.

---

## 26.8. Computational feasibility

Observed:

```text
W_SHORT fit:
40.698 s

W_LONG fit:
627.714 s

Warnings:
0 / 0

Estimator count:
100 / 100
```

Finding-level interpretation:

Random Forest B01 **khả thi về mặt tính toán** trên current environment, nhưng có chi phí rõ rệt.

Không phát sinh resource failure nên computational workaround policy **không được kích hoạt**.

Không cần:

- subsample;
- giảm số cây;
- chỉ train W_SHORT;
- dùng config khác nhau giữa windows.

---

## 26.9. Probability / default-decision-rule integrity

Runtime gates:

`M5.5 W_SHORT PROBABILITY / DEFAULT-RULE GATE: PASS`

`M5.5 W_LONG PROBABILITY / DEFAULT-RULE GATE: PASS`

Đã xác minh:

- `predict_proba` tồn tại;
- positive-class probability finite;
- probability nằm trong `[0,1]`;
- `predict()` khớp với probability argmax.

Final threshold:

`OPEN`

M5.5 không thực hiện threshold tuning.

---

## 26.10. Controlled-pair integrity

Runtime:

`Controlled equal fields: 18`

Gate:

`M5.5 CONTROLLED-PAIR CONFIG GATE: PASS`

Hai official runs dùng cùng:

- model family;
- exact RF-B01 config;
- `n_estimators=100`;
- feature/preprocessing version;
- matrix schema;
- validation population;
- imbalance strategy;
- random-state policy;
- probability semantics;
- threshold policy;
- warning/error handling.

Biến chủ động khác:

`TRAINING WINDOW`

Do đó W_SHORT/W_LONG differences là valid controlled Random Forest baseline evidence.

---

## 26.11. W_SHORT fraud-class baseline behavior

Observed:

```text
F1_fraud:
0.3664670659

Recall_fraud:
0.2908745247

Precision_fraud:
0.4951456311

Accuracy reference:
0.9985150002

TP:
306

FP:
312

FN:
746

TN:
711,094

Predicted positive count:
618

Predicted positive rate:
0.0008674196
```

Interpretation:

W_SHORT bắt được:

`306 / 1,052`

fraud trong validation.

Trong 618 positive predictions:

- 306 là true positive;
- 312 là false positive.

Recall vẫn thấp vì:

`746 fraud`

bị bỏ sót ở baseline operating rule.

---

## 26.12. W_LONG fraud-class baseline behavior

Observed:

```text
F1_fraud:
0.1527093596

Recall_fraud:
0.0884030418

Precision_fraud:
0.5602409639

Accuracy reference:
0.9985514936

TP:
93

FP:
73

FN:
959

TN:
711,333

Predicted positive count:
166

Predicted positive rate:
0.0002329962
```

Interpretation:

W_LONG có Precision cao hơn W_SHORT nhưng chỉ phát:

`166`

positive predictions trên toàn validation.

Nó bắt được:

`93 / 1,052`

fraud và bỏ sót:

`959 fraud`.

Do đó Precision cao hơn không được đọc riêng lẻ như bằng chứng model tốt hơn.

---

## 26.13. Descriptive W_LONG − W_SHORT deltas

Runtime output:

```text
F1_fraud:
-0.2137577063

Recall_fraud:
-0.2024714829

Precision_fraud:
+0.0650953328

Accuracy reference:
+0.0000364934

Predicted-positive rate:
-0.0006344234
```

Interpretation:

Trong Random Forest B01 cụ thể này:

```text
W_SHORT:
F1 cao hơn
Recall cao hơn

W_LONG:
Precision cao hơn
nhưng Recall thấp hơn rất mạnh
```

W_LONG có behavior bảo thủ hơn rõ rệt:

```text
Predicted positive:

W_SHORT:
618

W_LONG:
166
```

M5.5 chỉ ghi nhận đây là:

`DESCRIPTIVE BASELINE EVIDENCE`

không khóa final training-window winner.

---

## 26.14. Accuracy vẫn chỉ là reference metric

Observed:

```text
W_SHORT Accuracy:
≈ 0.998515

W_LONG Accuracy:
≈ 0.998551
```

Hai giá trị gần như tương đương và W_LONG thậm chí hơi cao hơn.

Nhưng fraud Recall:

```text
W_SHORT:
≈ 0.291

W_LONG:
≈ 0.088
```

Do đó Accuracy không phản ánh đầy đủ fraud-detection behavior trong bài toán mất cân bằng.

Điều này tiếp tục hỗ trợ metric policy:

```text
Primary:
F1_fraud

Secondary:
Recall_fraud
Precision_fraud

Accuracy:
reference only
```

---

## 26.15. Forest-complexity evidence

Observed:

```text
W_SHORT:
100 trees
451,532 total nodes
225,816 total leaves
mean depth = 41.43

W_LONG:
100 trees
3,066,586 total nodes
1,533,343 total leaves
mean depth = 49.06
```

Cả hai forest có:

`43 / 47`

features với non-zero impurity-based importance.

Interpretation:

Baseline forest sử dụng phần lớn representation và có complexity đáng kể.

Không được kết luận:

`overfitting confirmed`

chỉ từ depth/node count.

Đúng boundary:

`complexity/generalization/pruning/hyperparameter question → later evaluation/M7`

---

## 26.16. Persistence / reproducibility

Runtime gates:

`M5.5 PERSISTENCE GATE: PASS`

`M5.5 SAVED ARTIFACT ROUND-TRIP GATE: PASS`

`M5.5 PAIR MANIFEST GATE: PASS`

Persisted cho cả W_SHORT và W_LONG:

- `y_pred`;
- positive-class risk score;
- summary JSON;
- config lock;
- pair manifest.

Round-trip xác minh persisted arrays và metadata khớp runtime evidence.

---

## 26.17. FINAL TEST isolation

Runtime gate:

`M5.5 FINAL TEST ISOLATION GATE: PASS`

Evidence:

- M4 manifest: no final-test use;
- M5.2 contract: final-test access prohibited;
- both run summaries: `final_test_accessed=false`;
- pair manifest: `final_test_accessed=false`;
- output directory không chứa final-test artifact.

M5.5 không dùng FINAL TEST để chọn:

- số cây;
- training window;
- model family;
- imbalance strategy;
- threshold.

---

## 26.18. Overall technical result

Runtime:

```text
G01_MANIFEST_CONTRACT          → PASS
G02_INPUT_INTEGRITY            → PASS
G03_RF_CONFIG_DEFINED          → PASS
G04_PRE_RESULT_CONFIG_LOCK     → PASS
G05_W_SHORT_TECHNICAL          → PASS
G06_W_SHORT_PROBABILITY        → PASS
G07_W_LONG_TECHNICAL           → PASS
G08_W_LONG_PROBABILITY         → PASS
G09_CONTROLLED_PAIR            → PASS
G10_METRIC_RESULT_INTEGRITY    → PASS
G11_PERSISTENCE                → PASS
G12_ARTIFACT_ROUND_TRIP        → PASS
G13_PAIR_MANIFEST              → PASS
G14_FINAL_TEST_ISOLATION       → PASS
```

Overall runtime:

`M5.5 OVERALL TECHNICAL GATE: PASS`

Blocking issue:

`NONE`

# 27. Findings M5.5

## M5.5-F01 — Official Random Forest baseline pair is technically valid

Evidence:

```text
M5-RF-SHORT-B01:
completed
100 / 100 trees
warning_count = 0

M5-RF-LONG-B01:
completed
100 / 100 trees
warning_count = 0
```

Finding:

```text
RF-SHORT-B01 = VALID
RF-LONG-B01  = VALID
```

Status:

`VERIFIED`

---

## M5.5-F02 — Full Random Forest baseline is computationally feasible

Evidence:

```text
W_SHORT fit:
~40.7 s

W_LONG fit:
~627.7 s
```

Both full canonical windows completed without subsampling or config divergence.

Finding:

RF-B01 is operationally feasible in the current environment.

Caveat:

W_LONG training cost is substantial.

Status:

`VERIFIED`

---

## M5.5-F03 — Parallel execution policy worked as intended

Evidence:

```text
logical CPU count = 8
n_jobs = -1
```

Runtime observation during execution:

all CPU cores were utilized.

Finding:

Random Forest parallelism was active in the current execution environment.

Status:

`OBSERVED / CONSISTENT WITH CONFIG`

---

## M5.5-F04 — W_LONG forest is substantially larger than W_SHORT forest

Evidence:

```text
Total nodes:

W_SHORT:
451,532

W_LONG:
3,066,586

Ratio:
~6.79×

Total leaves:

W_SHORT:
225,816

W_LONG:
1,533,343

Ratio:
~6.79×
```

Finding:

Longer training history yields a materially larger RF-B01 forest structure.

Status:

`OBSERVED`

---

## M5.5-F05 — Runtime grows faster than row-count ratio

Evidence:

```text
Row ratio:
~3.98×

Fit-time ratio:
~15.42×

Node ratio:
~6.79×
```

Finding:

W_LONG computational cost cannot be explained by row count alone; forest structure is also much larger.

Status:

`OBSERVED`

---

## M5.5-F06 — W_SHORT has higher fraud F1 and Recall in RF-B01

Observed:

```text
W_SHORT:
F1       ≈ 0.3665
Recall   ≈ 0.2909
Precision≈ 0.4951

W_LONG:
F1       ≈ 0.1527
Recall   ≈ 0.0884
Precision≈ 0.5602
```

Finding:

Within this controlled Random Forest baseline pair, W_SHORT produces substantially higher fraud F1 and Recall.

Status:

`OBSERVED BASELINE EVIDENCE`

---

## M5.5-F07 — W_LONG is more conservative at the baseline decision rule

Evidence:

```text
Predicted positive:

W_SHORT:
618

W_LONG:
166

TP:

W_SHORT:
306

W_LONG:
93

FN:

W_SHORT:
746

W_LONG:
959
```

Finding:

W_LONG emits far fewer fraud flags, producing higher Precision but much lower Recall.

Status:

`OBSERVED`

---

## M5.5-F08 — Higher Precision alone does not establish better fraud detection

Evidence:

W_LONG Precision is higher:

`~0.5602 vs ~0.4951`

but Recall is much lower:

`~0.0884 vs ~0.2909`

and F1 is lower:

`~0.1527 vs ~0.3665`.

Finding:

Precision must be interpreted jointly with Recall/F1 and confusion-matrix evidence.

Status:

`CONSISTENT WITH METRIC CONTRACT`

---

## M5.5-F09 — Accuracy is not decision-sufficient

Evidence:

Both runs have Accuracy around `0.9985` despite large differences in fraud Recall/F1.

Finding:

Accuracy remains reference-only.

Status:

`VERIFIED`

---

## M5.5-F10 — Probability evidence is available for both official RF runs

Evidence:

Both probability/default-rule gates PASS and both risk-score artifacts survive round-trip.

Finding:

Later evaluation/threshold work can consume persisted Random Forest probability evidence.

Status:

`VERIFIED`

---

## M5.5-F11 — Resource workaround was not required

Evidence:

Both full-window runs completed successfully.

Finding:

No authorization is needed for subsampling, reduced tree count or asymmetric window treatment.

Status:

`NOT TRIGGERED`

---

## M5.5-F12 — Training-window winner remains open

Finding:

M5.5 records strong RF-specific W_SHORT/W_LONG differences but does not declare a final training-window winner.

Status:

`OPEN`

---

## M5.5-F13 — Model-family winner remains open

Finding:

Logistic Regression, Decision Tree and Random Forest baseline evidence now exist, but M5.5 does not perform final model-family selection.

Status:

`OPEN — M6/M7`

---

## M5.5-F14 — FINAL TEST remains protected

Evidence:

FINAL TEST isolation gate PASS.

Status:

`VERIFIED`

---

## M5.5-F15 — M5.6 handoff is unblocked

Evidence:

A complete persisted baseline pair now exists for all three core model families across M5.3–M5.5.

Status:

`READY FOR M5.6`

# 28. Decision Log M5.5

## M5.5-D01 — Official Random Forest baseline config

Decision:

`RF-B01-100-GINI-SQRT-BOOTSTRAP`

```text
n_estimators = 100
criterion = gini
max_depth = None
min_samples_split = 2
min_samples_leaf = 1
max_features = sqrt
bootstrap = True
class_weight = None
ccp_alpha = 0.0
max_samples = None
random_state = 42
n_jobs = -1
```

Status:

`LOCKED FOR M5 BASELINE`

---

## M5.5-D02 — Official Random Forest runs

Decision:

```text
M5-RF-SHORT-B01
M5-RF-LONG-B01
```

Status:

`RUNTIME VERIFIED`

---

## M5.5-D03 — Ensemble size

Decision:

`n_estimators = 100`

Role:

baseline ensemble-size reference.

Not interpreted as tuned optimum.

Status:

`LOCKED FOR M5 BASELINE`

---

## M5.5-D04 — Parallelism

Decision:

`n_jobs = -1`

Role:

execution policy to use available CPU cores.

Status:

`LOCKED`

---

## M5.5-D05 — Complexity policy

Decision:

Keep default-style unpruned forest as baseline evidence.

Do not change:

- max_depth;
- min_samples_leaf;
- max_features;
- tree count;
- pruning;

after reading validation results.

Status:

`LOCKED`

Tuning:

`DEFERRED TO M7`

---

## M5.5-D06 — Imbalance strategy

Decision:

```text
IMBALANCE_STRATEGY = NONE
class_weight = None
```

Status:

`INHERITED — VERIFIED — LOCKED`

---

## M5.5-D07 — Controlled training-window pair

Decision:

W_SHORT and W_LONG use the same exact RF-B01 config.

Status:

`VERIFIED`

Training-window winner:

`OPEN`

---

## M5.5-D08 — Metric contract

Decision:

Continue using:

- F1_fraud primary;
- Recall_fraud;
- Precision_fraud;
- TP/FP/FN/TN;
- predicted-positive count/rate;
- Accuracy reference-only.

Status:

`INHERITED — VERIFIED — LOCKED`

---

## M5.5-D09 — Probability evidence

Decision:

Persist positive-class `predict_proba` output for both official RF runs.

Status:

`LOCKED`

---

## M5.5-D10 — Baseline decision rule

Decision:

Use estimator default `predict()`, verified against probability argmax.

Final threshold:

`OPEN`

Status:

`LOCKED FOR BASELINE / FINAL OPEN`

---

## M5.5-D11 — Resource policy outcome

Decision:

No controlled workaround required.

Reason:

Both full-window runs completed successfully.

Status:

`NO WORKAROUND`

---

## M5.5-D12 — Training-window interpretation

Decision:

M5.5 may record RF-specific differences between W_SHORT and W_LONG.

M5.5 does not make final training-window selection.

Status:

`WINNER OPEN`

---

## M5.5-D13 — Model-family interpretation

Decision:

M5.5 does not select Logistic Regression, Decision Tree or Random Forest as final model.

Status:

`OPEN — M6/M7`

---

## M5.5-D14 — Hyperparameter / imbalance / threshold tuning

Decision:

Not performed in M5.5.

Status:

`DEFERRED TO M7`

---

## M5.5-D15 — FINAL TEST

Decision:

No access.

Status:

`INHERITED — VERIFIED — LOCKED`

---

## M5.5-D16 — Artifact handoff

Decision:

Retain predictions, probabilities, summaries, config lock and pair manifest as official M5.5 baseline evidence.

Status:

`LOCKED`

---

## M5.5-D17 — M5.6 handoff

Decision:

No M5.5 blocker remains before Baseline Model Registry / M5 Gate.

Status:

`READY`

# 29. M5.5 Gate

## Technical runtime gates

```text
G01_MANIFEST_CONTRACT          → PASS
G02_INPUT_INTEGRITY            → PASS
G03_RF_CONFIG_DEFINED          → PASS
G04_PRE_RESULT_CONFIG_LOCK     → PASS
G05_W_SHORT_TECHNICAL          → PASS
G06_W_SHORT_PROBABILITY        → PASS
G07_W_LONG_TECHNICAL           → PASS
G08_W_LONG_PROBABILITY         → PASS
G09_CONTROLLED_PAIR            → PASS
G10_METRIC_RESULT_INTEGRITY    → PASS
G11_PERSISTENCE                → PASS
G12_ARTIFACT_ROUND_TRIP        → PASS
G13_PAIR_MANIFEST              → PASS
G14_FINAL_TEST_ISOLATION       → PASS
```

---

## Review gate R01 — Runtime execution complete?

Evidence:

`22 / 22 code cells`

Result:

`PASS`

---

## Review gate R02 — Both official RF runs valid?

Evidence:

```text
RF-SHORT-B01:
100 / 100 estimators
warnings = 0

RF-LONG-B01:
100 / 100 estimators
warnings = 0
```

Result:

`PASS`

---

## Review gate R03 — Full-window execution preserved?

Evidence:

Canonical full W_SHORT/W_LONG row counts were used.

No subsampling or asymmetric config.

Result:

`PASS`

---

## Review gate R04 — Controlled comparison intact?

Evidence:

18 controlled fields match.

Training window is the intentional varying factor.

Result:

`PASS`

---

## Review gate R05 — No imbalance intervention?

Evidence:

```text
class_weight = None
IMBALANCE_STRATEGY = NONE
```

Result:

`PASS`

---

## Review gate R06 — No post-result RF tuning?

Evidence:

Official config remains pre-result locked B01.

Result:

`PASS`

---

## Review gate R07 — Probability evidence available?

Evidence:

Both probability/default-rule gates and persistence round-trip PASS.

Result:

`PASS`

---

## Review gate R08 — FINAL TEST protected?

Evidence:

FINAL TEST isolation gate PASS.

Result:

`PASS`

---

## Overall M5.5 Gate

```text
Technical gates:
14 / 14 PASS

Runtime review gates:
8 / 8 PASS

Blocking issue:
NONE
```

Final:

`M5.5 — PASS`

# 30. Kết luận M5.5

M5.5 đã tạo thành công controlled Random Forest baseline pair trên canonical M4.7 representation.

Official model config:

`RF-B01-100-GINI-SQRT-BOOTSTRAP`

Official runs:

```text
M5-RF-SHORT-B01
M5-RF-LONG-B01
```

Technical behavior:

```text
W_SHORT:
100 trees
fit ≈ 40.70 s
451,532 nodes
225,816 leaves
depth mean ≈ 41.43
warnings = 0

W_LONG:
100 trees
fit ≈ 627.71 s
3,066,586 nodes
1,533,343 leaves
depth mean ≈ 49.06
warnings = 0
```

Fraud-class baseline evidence:

```text
W_SHORT:
F1_fraud        ≈ 0.3665
Recall_fraud    ≈ 0.2909
Precision_fraud ≈ 0.4951
TP / FP         = 306 / 312
FN / TN         = 746 / 711,094
Predicted +     = 618

W_LONG:
F1_fraud        ≈ 0.1527
Recall_fraud    ≈ 0.0884
Precision_fraud ≈ 0.5602
TP / FP         = 93 / 73
FN / TN         = 959 / 711,333
Predicted +     = 166
```

Được phép kết luận ở M5.5:

- cả hai Random Forest baseline runs đều technically valid;
- full-data RF-B01 khả thi trong current environment;
- `n_jobs=-1` execution policy cho phép sử dụng nhiều CPU cores;
- W_LONG forest lớn hơn và tốn runtime hơn đáng kể;
- W_SHORT có F1/Recall cao hơn trong RF-B01;
- W_LONG có Precision cao hơn nhưng rất thấp Recall và phát ít positive hơn;
- probability evidence đã được persist;
- Accuracy tiếp tục không đủ để đánh giá fraud behavior;
- không cần resource workaround;
- FINAL TEST vẫn được bảo vệ.

M5.5 **không** khóa:

- optimal tree count;
- optimal max_depth;
- optimal max_features;
- optimal min_samples_leaf;
- optimal class-weight/resampling strategy;
- final threshold;
- W_SHORT là final training-window winner;
- Random Forest là final model;
- final-test performance.

Trạng thái cuối:

```text
M5.5 — PASS

Official Random Forest Baseline:
RF-B01-100-GINI-SQRT-BOOTSTRAP

RF-SHORT-B01:
VALID

RF-LONG-B01:
VALID

Estimator Count:
100 / 100 for both

Controlled Pair:
VERIFIED

Forest Complexity Evidence:
RECORDED

Computational Feasibility:
VERIFIED

Resource Workaround:
NOT REQUIRED

Imbalance Strategy:
NONE

Probability Evidence:
PERSISTED

Training-window Winner:
OPEN

Model-family Winner:
OPEN

Hyperparameter Tuning:
DEFERRED TO M7

Final Threshold:
OPEN

FINAL TEST:
PROTECTED

Blocking Issue:
NONE

READY FOR M5.6
```

Bước tiếp theo:

`M5.6 — Baseline Model Registry / M5 Decision Log / M5 Gate`

M5.6 sẽ tổng hợp official baseline evidence đã được runtime-verified từ:

```text
M5.3:
Logistic Regression

M5.4:
Decision Tree

M5.5:
Random Forest
```

M5.6 vẫn không được sử dụng FINAL TEST và không biến M5 thành final model-selection milestone.
