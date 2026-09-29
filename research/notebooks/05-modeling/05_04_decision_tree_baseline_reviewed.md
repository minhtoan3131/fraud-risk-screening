# M5.4 — Decision Tree baseline

Milestone:

`M5 — Modeling baseline`

Substep:

`M5.4 — Decision Tree baseline`

Official runs:

`M5-DT-SHORT-B01`

`M5-DT-LONG-B01`

Mục tiêu:

> Tạo một Decision Tree baseline reproducible trên cùng canonical M4.7 representation, cùng W_SHORT/W_LONG protocol, cùng metric contract và không can thiệp class imbalance.

M5.4:

- khóa exact Decision Tree baseline config **trước runtime result**;
- dùng cùng config cho W_SHORT và W_LONG;
- dùng `IMBALANCE_STRATEGY = NONE`;
- giữ `predict_proba` làm risk/probability evidence;
- dùng `predict()` của estimator làm baseline decision rule;
- không tuning `max_depth`;
- không tuning `min_samples_leaf`;
- không pruning search;
- không class weighting;
- không temporal CV;
- không chọn final threshold;
- không dùng FINAL TEST;
- không tự chọn training-window winner;
- không so model-family winner với Logistic Regression trong notebook này.

M5.4 chỉ được khóa sau runtime review.

## 1. Contract kế thừa

M5.4 kế thừa trực tiếp:

```text
M4.7:
canonical persisted baseline-ready matrices

M5.1:
Modeling Charter / baseline protocol

M5.2:
artifact + metric + experiment-result contract

M5.3:
không thay đổi M5-wide contract;
chỉ cung cấp Logistic Regression baseline evidence
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

Decision Tree là một trong ba model family core của M5. fileciteturn66file0L144-L176

## 2. Exact Decision Tree baseline configuration

M5.1 để exact Decision Tree config ở trạng thái:

`OPEN — M5.4`

M5.4 khóa trước result:

```text
Estimator:
sklearn.tree.DecisionTreeClassifier

criterion:
gini

splitter:
best

max_depth:
None

min_samples_split:
2

min_samples_leaf:
1

max_features:
None

class_weight:
None

ccp_alpha:
0.0

random_state:
42
```

### Vì sao không tự chọn `max_depth=5`, `10`, `20`, ...?

Decision Tree quá sâu có thể overfit, và `max_depth` / `min_samples_leaf` là các hyperparameter chính kiểm soát độ phức tạp. Nhưng không có một độ sâu “thần kỳ” đúng cho mọi dataset; lựa chọn tốt cần validation/CV thay vì cảm giác. fileciteturn67file2L177-L199

Do đó M5.4 dùng **unpruned/default-complexity tree** như một baseline reference, không coi đây là final tree configuration.

Nếu cây rất sâu hoặc validation behavior kém:

`ghi nhận baseline evidence`

không âm thầm sửa depth rồi chạy lại để lấy score tốt hơn.

Complexity tuning:

`DEFERRED TO M7`

## 3. Runtime order

```text
environment
        ↓
locate M4.7 + M5.2 contract
        ↓
load W_SHORT / W_LONG / VALIDATION
        ↓
input-integrity preflight
        ↓
define metric + DT runner
        ↓
freeze DT-B01 config
        ↓
W_SHORT official run
        ↓
technical / probability / tree-structure gate
        ↓
W_LONG official run
        ↓
technical / probability / tree-structure gate
        ↓
controlled-pair equivalence
        ↓
persist predictions / probabilities / summaries
        ↓
round-trip
        ↓
FINAL TEST isolation
        ↓
M5.4 technical gate
        ↓
runtime review
```

W_SHORT chạy trước để phát hiện lỗi implementation/resource sớm.

W_SHORT score không được dùng để thay đổi config trước W_LONG.


```python

from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import hashlib
import json
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
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )
    from sklearn.tree import DecisionTreeClassifier
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
    
    NumPy:
    2.5.3
    
    scikit-learn:
    1.9.1


## 4. Locate canonical artifacts và M5.2 contract


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

M5_04_OUTPUT_RELATIVE_DIR = (
    Path("data")
    / "processed"
    / "m5_04_decision_tree_baseline"
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


ARTIFACT_DIR = PROJECT_ROOT / M4_ARTIFACT_RELATIVE_DIR

M5_02_CONTRACT_PATH = (
    PROJECT_ROOT
    / M5_02_CONTRACT_RELATIVE_PATH
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / M5_04_OUTPUT_RELATIVE_DIR
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
    /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/processed/m5_04_decision_tree_baseline


## 5. Canonical constants


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

DT_MODEL_FAMILY = "Decision Tree"
DT_MODEL_ID = "SKLEARN_DECISION_TREE_CLASSIFIER"
DT_CONFIG_ID = "DT-B01-DEFAULT-GINI-UNPRUNED"

DT_SHORT_EXPERIMENT_ID = "M5-DT-SHORT-B01"
DT_LONG_EXPERIMENT_ID = "M5-DT-LONG-B01"


print("Model family:")
print(DT_MODEL_FAMILY)

print("\nConfig ID:")
print(DT_CONFIG_ID)

print("\nImbalance strategy:")
print(IMBALANCE_STRATEGY)

print("\nThreshold policy:")
print(BASELINE_THRESHOLD_POLICY)

```

    Model family:
    Decision Tree
    
    Config ID:
    DT-B01-DEFAULT-GINI-UNPRUNED
    
    Imbalance strategy:
    NONE
    
    Threshold policy:
    DEFAULT_MODEL_DECISION_RULE


## 6. Load manifest, feature names và M5.2 contract


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

print("\nM5.4 MANIFEST / CONTRACT GATE: PASS")

```

    M4 pipeline:
    M4.7-baseline-v1
    
    M5.2 runner:
    M5.2-shared-runner-v1
    
    Feature count:
    47
    
    M5.4 MANIFEST / CONTRACT GATE: PASS


## 7. Load modeling matrices và targets

Không load lineage vào classifier.

Không có FINAL TEST path.


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

load_seconds = time.perf_counter() - load_start


print("Artifact load seconds:")
print(round(load_seconds, 2))

print("\nM5.4 ARTIFACT LOAD: COMPLETE")

```

    Artifact load seconds:
    0.77
    
    M5.4 ARTIFACT LOAD: COMPLETE


## 8. Input-integrity preflight


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

print("\nM5.4 INPUT INTEGRITY GATE: PASS")

```

    W_SHORT:
    (1721615, 47) float32 fraud= 2491
    
    W_LONG:
    (6855270, 47) float32 fraud= 9606
    
    VALIDATION:
    (712458, 47) fraud= 1052
    
    M5.4 INPUT INTEGRITY GATE: PASS


## 9. Canonical metric implementation


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

    predicted_positive_count = int(tp + fp)

    predicted_positive_rate = (
        predicted_positive_count / len(y_true)
        if len(y_true)
        else np.nan
    )

    return {
        "f1_fraud": float(
            f1_score(
                y_true,
                y_pred,
                pos_label=1,
                zero_division=0,
            )
        ),
        "recall_fraud": float(
            recall_score(
                y_true,
                y_pred,
                pos_label=1,
                zero_division=0,
            )
        ),
        "precision_fraud": float(
            precision_score(
                y_true,
                y_pred,
                pos_label=1,
                zero_division=0,
            )
        ),
        "accuracy_reference": float(
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
            float(predicted_positive_rate),
        "validation_rows":
            int(len(y_true)),
    }

```

## 10. Decision Tree runner

Runner giữ semantics của M5.2 và bổ sung tree diagnostics:

```text
depth
node_count
n_leaves
nonzero_feature_importance_count
max_feature_importance
```

Các diagnostics này chỉ mô tả complexity của baseline tree.

Không dùng chúng để tự tuning trong M5.4.


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


def run_dt_baseline(
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

    importances = np.asarray(
        model.feature_importances_,
        dtype=np.float64,
    )

    tree_diagnostics = {
        "tree_depth":
            int(model.get_depth()),

        "node_count":
            int(model.tree_.node_count),

        "n_leaves":
            int(model.get_n_leaves()),

        "nonzero_feature_importance_count":
            int(np.count_nonzero(importances)),

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
            DT_MODEL_FAMILY,

        "model_id":
            DT_MODEL_ID,

        "model_config_id":
            DT_CONFIG_ID,

        "model_class":
            model.__class__.__name__,

        "model_module":
            model.__class__.__module__,

        "model_params":
            model_params,

        "model_diagnostics":
            tree_diagnostics,

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
            int(np.asarray(y_train).sum()),

        "validation_rows":
            int(X_validation.shape[0]),

        "validation_fraud_rows":
            int(np.asarray(y_validation).sum()),

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
            "OPEN — REQUIRES M5.4 RUNTIME REVIEW",

        "next_action":
            "REVIEW DECISION TREE BASELINE EVIDENCE",

        **metrics,
    }

    return ExperimentRunOutput(
        summary=summary,
        y_pred=y_pred,
        risk_score=risk_score,
        warning_messages=warning_messages,
    )


def assert_clean_dt_run(
    output,
):
    summary = output.summary

    assert summary["run_status"] == "COMPLETED"

    if summary["warnings"]:
        for warning_message in summary["warnings"]:
            print(
                " -",
                warning_message,
            )

    assert summary["warning_count"] == 0, (
        "Decision Tree run có warning cần runtime review."
    )

    diagnostics = summary[
        "model_diagnostics"
    ]

    assert diagnostics["tree_depth"] >= 0
    assert diagnostics["node_count"] >= 1
    assert diagnostics["n_leaves"] >= 1

    assert np.isclose(
        diagnostics[
            "feature_importance_sum"
        ],
        1.0,
        atol=1e-6,
    ), (
        "Feature importance sum không bằng 1."
    )

    assert summary["final_test_accessed"] is False
    assert summary["errors"] == []

```

## 11. Define Decision Tree B01

Đây là baseline tree gần sklearn defaults, không complexity tuning.


```python

DT_BASELINE_CONFIG = {
    "criterion": "gini",
    "splitter": "best",
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": None,
    "class_weight": None,
    "ccp_alpha": 0.0,
    "random_state": DEFAULT_RANDOM_STATE,
}


dt_template = DecisionTreeClassifier(
    **DT_BASELINE_CONFIG
)


actual_params = dt_template.get_params(
    deep=False
)


for key, expected_value in DT_BASELINE_CONFIG.items():
    assert actual_params[key] == expected_value


assert actual_params["class_weight"] is None
assert actual_params["max_depth"] is None
assert actual_params["criterion"] == "gini"
assert actual_params["splitter"] == "best"
assert actual_params["random_state"] == DEFAULT_RANDOM_STATE
assert actual_params["ccp_alpha"] == 0.0


print("Decision Tree baseline config:")

for key in sorted(DT_BASELINE_CONFIG):
    print(
        f"{key}: "
        f"{DT_BASELINE_CONFIG[key]}"
    )


print("\nM5.4 DT CONFIG DEFINITION GATE: PASS")

```

    Decision Tree baseline config:
    ccp_alpha: 0.0
    class_weight: None
    criterion: gini
    max_depth: None
    max_features: None
    min_samples_leaf: 1
    min_samples_split: 2
    random_state: 42
    splitter: best
    
    M5.4 DT CONFIG DEFINITION GATE: PASS


## 12. Freeze B01 config trước model result

Config fingerprint được persist trước W_SHORT/W_LONG.


```python

config_lock_payload = {
    "m5_substep":
        "M5.4",

    "model_family":
        DT_MODEL_FAMILY,

    "model_id":
        DT_MODEL_ID,

    "model_config_id":
        DT_CONFIG_ID,

    "config":
        DT_BASELINE_CONFIG,

    "config_role":
        "UNPRUNED_DEFAULT_COMPLEXITY_BASELINE",

    "complexity_tuning":
        False,

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
    config_canonical_json.encode("utf-8")
).hexdigest()


config_lock_payload[
    "config_fingerprint_sha256"
] = config_fingerprint


CONFIG_LOCK_PATH = (
    OUTPUT_DIR
    / "m5_04_dt_baseline_config_lock.json"
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

assert reloaded_config["complexity_tuning"] is False
assert reloaded_config["systematic_tuning"] is False
assert reloaded_config["final_test_access"] is False
assert reloaded_config["imbalance_strategy"] == "NONE"


print("Config lock:")
print(CONFIG_LOCK_PATH)

print("\nConfig SHA256:")
print(config_fingerprint)

print("\nM5.4 PRE-RESULT CONFIG LOCK GATE: PASS")

```

    Config lock:
    /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/processed/m5_04_decision_tree_baseline/m5_04_dt_baseline_config_lock.json
    
    Config SHA256:
    b83ba67a2cbac701c8959882020305ac3f6a60470a337645065500226b86c1f4
    
    M5.4 PRE-RESULT CONFIG LOCK GATE: PASS


# 13. W_SHORT official run

W_SHORT chạy trước như computational/implementation preflight.

Nếu cell fail:

`STOP`

Không sửa hyperparameter theo score rồi tiếp tục W_LONG.


```python

print("=" * 72)
print("Starting:", DT_SHORT_EXPERIMENT_ID)

print(
    "Training rows:",
    f"{EXPECTED_W_SHORT_ROWS:,}",
)

print(
    "Training fraud:",
    f"{EXPECTED_W_SHORT_FRAUD:,}",
)


short_output = run_dt_baseline(
    experiment_id=
        DT_SHORT_EXPERIMENT_ID,
    estimator=
        dt_template,
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


short_summary = short_output.summary


print("\nCompleted:", DT_SHORT_EXPERIMENT_ID)

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
    "tree_depth:",
    short_summary[
        "model_diagnostics"
    ][
        "tree_depth"
    ],
)

print(
    "node_count:",
    short_summary[
        "model_diagnostics"
    ][
        "node_count"
    ],
)

print(
    "n_leaves:",
    short_summary[
        "model_diagnostics"
    ][
        "n_leaves"
    ],
)


assert_clean_dt_run(
    short_output
)


print(
    "\nM5.4 W_SHORT TECHNICAL GATE: PASS"
)

```

    ========================================================================
    Starting: M5-DT-SHORT-B01
    Training rows: 1,721,615
    Training fraud: 2,491
    
    Completed: M5-DT-SHORT-B01
    fit_seconds: 6.662
    prediction_seconds: 0.048
    warning_count: 0
    tree_depth: 44
    node_count: 3547
    n_leaves: 1774
    
    M5.4 W_SHORT TECHNICAL GATE: PASS


## 14. W_SHORT probability/default-rule integrity

Decision Tree baseline decision rule được kiểm tra bằng:

`predict() == classes[argmax(predict_proba)]`

Không giả định một numerical threshold riêng.

Final threshold vẫn OPEN.


```python

short_probability = np.column_stack(
    [
        1.0 - short_output.risk_score,
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
    "\nM5.4 W_SHORT PROBABILITY / DEFAULT-RULE GATE: PASS"
)

```

    Risk-score kind: predict_proba
    Final threshold: OPEN
    
    M5.4 W_SHORT PROBABILITY / DEFAULT-RULE GATE: PASS


# 15. W_LONG official run

Chỉ chạy sau khi W_SHORT technical/probability gates PASS.

Exact config không đổi.


```python

print("=" * 72)
print("Starting:", DT_LONG_EXPERIMENT_ID)

print(
    "Training rows:",
    f"{EXPECTED_W_LONG_ROWS:,}",
)

print(
    "Training fraud:",
    f"{EXPECTED_W_LONG_FRAUD:,}",
)


long_output = run_dt_baseline(
    experiment_id=
        DT_LONG_EXPERIMENT_ID,
    estimator=
        dt_template,
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


long_summary = long_output.summary


print("\nCompleted:", DT_LONG_EXPERIMENT_ID)

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
    "tree_depth:",
    long_summary[
        "model_diagnostics"
    ][
        "tree_depth"
    ],
)

print(
    "node_count:",
    long_summary[
        "model_diagnostics"
    ][
        "node_count"
    ],
)

print(
    "n_leaves:",
    long_summary[
        "model_diagnostics"
    ][
        "n_leaves"
    ],
)


assert_clean_dt_run(
    long_output
)


print(
    "\nM5.4 W_LONG TECHNICAL GATE: PASS"
)

```

    ========================================================================
    Starting: M5-DT-LONG-B01
    Training rows: 6,855,270
    Training fraud: 9,606
    
    Completed: M5-DT-LONG-B01
    fit_seconds: 116.47
    prediction_seconds: 0.102
    warning_count: 0
    tree_depth: 41
    node_count: 28651
    n_leaves: 14326
    
    M5.4 W_LONG TECHNICAL GATE: PASS


## 16. W_LONG probability/default-rule integrity


```python

long_probability = np.column_stack(
    [
        1.0 - long_output.risk_score,
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
    "M5.4 W_LONG PROBABILITY / DEFAULT-RULE GATE: PASS"
)

```

    M5.4 W_LONG PROBABILITY / DEFAULT-RULE GATE: PASS


## 17. Controlled-pair configuration equivalence


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
    "\nM5.4 CONTROLLED-PAIR CONFIG GATE: PASS"
)

```

    Controlled equal fields: 18
    
    M5.4 CONTROLLED-PAIR CONFIG GATE: PASS


## 18. Metric/result integrity

Chỉ kiểm tra arithmetic/internal consistency.

Không chọn winner.


```python

for summary in [
    short_summary,
    long_summary,
]:
    assert (
        summary["validation_rows"]
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

    assert summary[
        "integrity_gate_result"
    ] == "PASS"

    assert summary["errors"] == []

    assert summary[
        "final_test_accessed"
    ] is False


print(
    "M5.4 METRIC / RESULT INTEGRITY GATE: PASS"
)

```

    M5.4 METRIC / RESULT INTEGRITY GATE: PASS


## 19. Baseline evidence output

In raw descriptive evidence cho runtime review.

Không gắn nhãn winner.


```python

def print_dt_evidence(
    summary,
):
    diagnostics = summary[
        "model_diagnostics"
    ]

    print(
        "Experiment:",
        summary["experiment_id"],
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
        "Tree depth:",
        diagnostics[
            "tree_depth"
        ],
    )

    print(
        "Node count:",
        diagnostics[
            "node_count"
        ],
    )

    print(
        "Leaves:",
        diagnostics[
            "n_leaves"
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
        summary["f1_fraud"],
    )

    print(
        "Recall_fraud:",
        summary["recall_fraud"],
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
print_dt_evidence(
    short_summary
)

print("\n" + "=" * 72)
print_dt_evidence(
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
    "OPEN — M5.4 DOES NOT SELECT",
)

print(
    "Model-family winner:",
    "OPEN — M5.4 DOES NOT SELECT",
)

```

    ========================================================================
    Experiment: M5-DT-SHORT-B01
    Training window: W_SHORT
    Training rows: 1,721,615
    Training fraud: 2,491
    Tree depth: 44
    Node count: 3547
    Leaves: 1774
    Nonzero feature importances: 40
    F1_fraud: 0.3270564915758176
    Recall_fraud: 0.31368821292775667
    Precision_fraud: 0.3416149068322981
    Accuracy reference: 0.9980939227294802
    TP / FP / FN / TN: 330 636 722 710770
    Predicted positive count: 966
    Predicted positive rate: 0.0013558693986171816
    Fit seconds: 6.661822540976573
    Prediction seconds: 0.048459709039889276
    
    ========================================================================
    Experiment: M5-DT-LONG-B01
    Training window: W_LONG
    Training rows: 6,855,270
    Training fraud: 9,606
    Tree depth: 41
    Node count: 28651
    Leaves: 14326
    Nonzero feature importances: 40
    F1_fraud: 0.19913419913419914
    Recall_fraud: 0.1967680608365019
    Precision_fraud: 0.20155793573515093
    Accuracy reference: 0.9976630201359238
    TP / FP / FN / TN: 207 820 845 710586
    Predicted positive count: 1027
    Predicted positive rate: 0.0014414884807244777
    Fit seconds: 116.46980562497629
    Prediction seconds: 0.10240820900071412
    
    ========================================================================
    Descriptive metric deltas: W_LONG - W_SHORT
    f1_fraud -0.12792229244161848
    recall_fraud -0.11692015209125478
    precision_fraud -0.14005697109714718
    accuracy_reference -0.0004309025935563815
    predicted_positive_rate 8.561908210729611e-05
    
    Training-window winner: OPEN — M5.4 DOES NOT SELECT
    Model-family winner: OPEN — M5.4 DOES NOT SELECT


## 20. Persist Decision Tree baseline evidence


```python

def save_experiment_output(
    output,
    output_dir,
):
    output_dir = Path(output_dir)

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


for window_id, paths in saved_outputs.items():
    print("\n", window_id, sep="")

    for key, path in paths.items():
        print(
            key,
            "→",
            path,
        )

        assert path.exists()
        assert path.stat().st_size > 0


print(
    "\nM5.4 PERSISTENCE GATE: PASS"
)

```

    
    W_SHORT
    summary_path → /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/processed/m5_04_decision_tree_baseline/M5-DT-SHORT-B01__summary.json
    prediction_path → /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/processed/m5_04_decision_tree_baseline/M5-DT-SHORT-B01__y_pred.npy
    risk_score_path → /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/processed/m5_04_decision_tree_baseline/M5-DT-SHORT-B01__risk_score.npy
    
    W_LONG
    summary_path → /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/processed/m5_04_decision_tree_baseline/M5-DT-LONG-B01__summary.json
    prediction_path → /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/processed/m5_04_decision_tree_baseline/M5-DT-LONG-B01__y_pred.npy
    risk_score_path → /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/processed/m5_04_decision_tree_baseline/M5-DT-LONG-B01__risk_score.npy
    
    M5.4 PERSISTENCE GATE: PASS


## 21. Saved-artifact round-trip


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

    assert reloaded_pred.dtype == np.int8
    assert reloaded_score.dtype == np.float32

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
        reloaded_summary = json.load(
            file
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
        == DT_CONFIG_ID
    )

    assert (
        reloaded_summary[
            "final_test_accessed"
        ]
        is False
    )


print(
    "M5.4 SAVED ARTIFACT ROUND-TRIP GATE: PASS"
)

```

    M5.4 SAVED ARTIFACT ROUND-TRIP GATE: PASS


## 22. Pair manifest


```python

pair_manifest = {
    "m5_substep":
        "M5.4",

    "model_family":
        DT_MODEL_FAMILY,

    "model_id":
        DT_MODEL_ID,

    "model_config_id":
        DT_CONFIG_ID,

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

    "complexity_tuning":
        False,

    "final_test_accessed":
        False,

    "runs": {
        "W_SHORT": {
            "experiment_id":
                DT_SHORT_EXPERIMENT_ID,

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
                DT_LONG_EXPERIMENT_ID,

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
    / "m5_04_dt_pair_manifest.json"
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
    reloaded_pair_manifest = json.load(
        file
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
    "\nM5.4 PAIR MANIFEST GATE: PASS"
)

```

    Pair manifest:
    /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/processed/m5_04_decision_tree_baseline/m5_04_dt_pair_manifest.json
    
    M5.4 PAIR MANIFEST GATE: PASS


## 23. FINAL TEST isolation


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
    "M5.4 FINAL TEST ISOLATION GATE: PASS"
)

```

    M5.4 FINAL TEST ISOLATION GATE: PASS


## 24. M5.4 overall technical gate


```python

m5_04_gates = {
    "G01_MANIFEST_CONTRACT":
        True,

    "G02_INPUT_INTEGRITY":
        True,

    "G03_DT_CONFIG_DEFINED":
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
    m5_04_gates.items()
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
    m5_04_gates.values()
)


print(
    "\nM5.4 OVERALL TECHNICAL GATE: PASS"
)

```

    G01_MANIFEST_CONTRACT → PASS
    G02_INPUT_INTEGRITY → PASS
    G03_DT_CONFIG_DEFINED → PASS
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
    
    M5.4 OVERALL TECHNICAL GATE: PASS


# 25. Phân tích và nhận xét runtime M5.4

## 25.1. Execution integrity

Notebook đã thực thi đầy đủ:

```text
22 / 22 code cells
execution_count = 1 → 22 liên tục
runtime error = 0
stderr = 0
```

Do đó output hiện tại đủ điều kiện dùng làm runtime evidence cho M5.4.

---

## 25.2. Canonical input / contract integrity

Runtime xác nhận M5.4 dùng đúng persisted M4.7 artifacts và M5.2 modeling contract.

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

Gate:

`M5.4 INPUT INTEGRITY GATE: PASS`

Nhận xét:

Decision Tree không được fit trên sample rút gọn, schema khác hoặc feature set mới. Hai training windows tiếp tục dùng cùng canonical baseline representation.

---

## 25.3. Config được khóa trước result

Official baseline config:

`DT-B01-DEFAULT-GINI-UNPRUNED`

```text
criterion         = gini
splitter          = best
max_depth         = None
min_samples_split = 2
min_samples_leaf  = 1
max_features      = None
class_weight      = None
ccp_alpha         = 0.0
random_state      = 42
```

Config fingerprint:

`b83ba67a2cbac701c8959882020305ac3f6a60470a337645065500226b86c1f4`

Runtime gate:

`M5.4 PRE-RESULT CONFIG LOCK GATE: PASS`

Interpretation:

M5.4 không thử nhiều depth/leaf/pruning config rồi chọn validation score tốt nhất.

Tree complexity được để ở baseline unpruned/default-complexity state để tạo reference evidence; complexity tuning vẫn nằm ngoài M5.4.

---

## 25.4. W_SHORT technical behavior

Observed:

```text
Experiment:
M5-DT-SHORT-B01

Training rows:
1,721,615

Training fraud:
2,491

fit_seconds:
6.662

prediction_seconds:
0.048

warning_count:
0

tree_depth:
44

node_count:
3,547

n_leaves:
1,774

nonzero feature importances:
40
```

Gate:

`M5.4 W_SHORT TECHNICAL GATE: PASS`

Nhận xét:

Run hoàn thành sạch, không warning và tạo một cây sâu. Đây là expected baseline complexity evidence khi `max_depth=None`, không phải bằng chứng tự thân rằng model đã overfit.

M5.4 không có training-performance comparison để khẳng định overfitting; câu hỏi đó cần được đánh giá ở bước evaluation/model-selection sau.

---

## 25.5. W_LONG technical behavior

Observed:

```text
Experiment:
M5-DT-LONG-B01

Training rows:
6,855,270

Training fraud:
9,606

fit_seconds:
116.470

prediction_seconds:
0.102

warning_count:
0

tree_depth:
41

node_count:
28,651

n_leaves:
14,326

nonzero feature importances:
40
```

Gate:

`M5.4 W_LONG TECHNICAL GATE: PASS`

Nhận xét:

W_LONG tạo cây không sâu hơn W_SHORT về maximum depth, nhưng lớn hơn rất nhiều về số node và leaf.

```text
W_SHORT:
3,547 nodes
1,774 leaves

W_LONG:
28,651 nodes
14,326 leaves
```

Điều này cho thấy full unpruned tree sử dụng W_LONG tạo cấu trúc phân nhánh rộng/lớn hơn đáng kể.

Đây là complexity evidence cần giữ lại cho M6/M7; M5.4 không tự pruning hoặc giảm depth sau khi nhìn thấy kết quả này.

---

## 25.6. Computational behavior

Observed fit time:

```text
W_SHORT:
~6.66 s

W_LONG:
~116.47 s
```

W_LONG có gần 4 lần số training rows nhưng fit time tăng mạnh hơn tỷ lệ row count.

Interpretation:

Decision Tree training cost không chỉ phụ thuộc số rows mà còn phụ thuộc quá trình tìm split và cấu trúc cây được tạo ra. W_LONG cũng tạo số node/leaves lớn hơn rất nhiều.

Runtime khoảng hai phút của notebook vì vậy phù hợp với observed full-data tree construction; không có evidence notebook bỏ qua phần lớn dữ liệu.

---

## 25.7. Probability / default-decision-rule integrity

Runtime gates:

`M5.4 W_SHORT PROBABILITY / DEFAULT-RULE GATE: PASS`

`M5.4 W_LONG PROBABILITY / DEFAULT-RULE GATE: PASS`

Đã xác minh:

- `predict_proba` tồn tại;
- positive-class risk score finite;
- risk score nằm trong `[0, 1]`;
- class prediction khớp default argmax probability rule.

Final threshold vẫn:

`OPEN`

M5.4 không thực hiện threshold tuning.

---

## 25.8. Controlled-pair integrity

Runtime output:

`Controlled equal fields: 18`

Gate:

`M5.4 CONTROLLED-PAIR CONFIG GATE: PASS`

Hai run dùng cùng:

- Decision Tree family;
- exact estimator config;
- feature version;
- preprocessing version;
- matrix schema;
- validation population;
- imbalance strategy;
- random state;
- threshold policy;
- probability semantics.

Biến chủ động khác:

`training window`

Do đó W_SHORT/W_LONG differences có thể được giữ như controlled Decision Tree baseline evidence.

Training-window winner vẫn:

`OPEN`

---

## 25.9. W_SHORT fraud-class baseline behavior

Observed:

```text
F1_fraud:
0.3270564916

Recall_fraud:
0.3136882129

Precision_fraud:
0.3416149068

Accuracy reference:
0.9980939227

TP:
330

FP:
636

FN:
722

TN:
710,770

Predicted positive count:
966

Predicted positive rate:
0.0013558694
```

Interpretation:

Ở baseline Decision Tree B01, W_SHORT bắt được:

`330 / 1,052`

fraud trong validation.

Model đồng thời tạo:

`636 false positives`

và bỏ sót:

`722 fraud`.

Đây là baseline evidence, không phải final operating point.

---

## 25.10. W_LONG fraud-class baseline behavior

Observed:

```text
F1_fraud:
0.1991341991

Recall_fraud:
0.1967680608

Precision_fraud:
0.2015579357

Accuracy reference:
0.9976630201

TP:
207

FP:
820

FN:
845

TN:
710,586

Predicted positive count:
1,027

Predicted positive rate:
0.0014414885
```

Interpretation:

W_LONG dự đoán positive nhiều hơn W_SHORT một chút:

```text
W_LONG:
1,027

W_SHORT:
966
```

nhưng:

```text
TP:
207 vs 330

FP:
820 vs 636
```

Do đó trong Decision Tree B01, khác biệt giữa hai windows không phải đơn thuần do một window phát ít cảnh báo hơn.

---

## 25.11. Descriptive W_LONG − W_SHORT deltas

Runtime output:

```text
F1_fraud:
-0.1279222924

Recall_fraud:
-0.1169201521

Precision_fraud:
-0.1400569711

Accuracy reference:
-0.0004309026

Predicted-positive rate:
+0.0000856191
```

Interpretation:

Trong **Decision Tree B01 cụ thể này**, W_SHORT có fraud-class F1/Recall/Precision cao hơn W_LONG trên cùng validation.

Tuy nhiên M5.4 không chuyển finding này thành:

`W_SHORT = FINAL TRAINING-WINDOW WINNER`

vì training-window selection và cross-model evidence còn chưa hoàn tất.

---

## 25.12. Accuracy vẫn chỉ là reference metric

Cả hai run có Accuracy gần 1:

```text
W_SHORT ≈ 0.99809
W_LONG  ≈ 0.99766
```

nhưng fraud-class Recall chỉ khoảng:

```text
W_SHORT ≈ 0.314
W_LONG  ≈ 0.197
```

Điều này tiếp tục cho thấy Accuracy không đủ để mô tả chất lượng fraud detection trong dataset mất cân bằng.

Primary/secondary fraud metrics vẫn phải được ưu tiên theo contract.

---

## 25.13. Tree-complexity evidence

Observed:

```text
W_SHORT:
depth = 44
nodes = 3,547
leaves = 1,774

W_LONG:
depth = 41
nodes = 28,651
leaves = 14,326
```

Cả hai run đều có:

`40 / 47 features`

với non-zero impurity-based feature importance.

Interpretation:

Unpruned baseline sử dụng phần lớn feature representation và tạo cây có complexity đáng kể.

Không được kết luận:

`overfitting confirmed`

chỉ từ depth/node count.

Đúng kết luận ở M5.4 là:

`complexity risk / pruning question requires later evaluation`

---

## 25.14. Persistence / reproducibility

Runtime gates:

`M5.4 PERSISTENCE GATE: PASS`

`M5.4 SAVED ARTIFACT ROUND-TRIP GATE: PASS`

`M5.4 PAIR MANIFEST GATE: PASS`

Đã persist cho cả hai run:

- prediction;
- positive-class risk score;
- summary JSON;
- pair manifest;
- config lock.

Round-trip xác minh persisted arrays/metadata khớp runtime evidence.

---

## 25.15. FINAL TEST isolation

Runtime gate:

`M5.4 FINAL TEST ISOLATION GATE: PASS`

Evidence:

- M4 manifest vẫn ghi `final_test_used = false`;
- M5.2 contract không cho phép final-test access;
- cả hai summaries ghi `final_test_accessed = false`;
- pair manifest ghi `final_test_accessed = false`;
- output directory không chứa final-test artifact.

M5.4 không mở FINAL TEST để chọn depth, pruning, training window hoặc model family.

---

## 25.16. Overall technical result

Runtime:

```text
G01_MANIFEST_CONTRACT          → PASS
G02_INPUT_INTEGRITY            → PASS
G03_DT_CONFIG_DEFINED          → PASS
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

`M5.4 OVERALL TECHNICAL GATE: PASS`

Blocking issue:

`NONE`

# 26. Findings M5.4

## M5.4-F01 — Official Decision Tree baseline pair is technically valid

Evidence:

- W_SHORT completed;
- W_LONG completed;
- warning count = 0 for both;
- input/metric/probability/persistence gates PASS.

Finding:

```text
M5-DT-SHORT-B01 = VALID
M5-DT-LONG-B01  = VALID
```

Status:

`VERIFIED`

---

## M5.4-F02 — Full unpruned Decision Tree is computationally feasible

Evidence:

```text
W_SHORT fit ≈ 6.66 s
W_LONG fit  ≈ 116.47 s
```

Finding:

Decision Tree B01 can be trained on both full canonical windows in the current environment without subsampling or schema reduction.

Status:

`VERIFIED`

---

## M5.4-F03 — W_LONG tree is structurally much larger

Evidence:

```text
W_SHORT:
3,547 nodes
1,774 leaves

W_LONG:
28,651 nodes
14,326 leaves
```

Finding:

Longer classifier-training history substantially increases the size of the learned unpruned tree, even though maximum depth is slightly lower.

Status:

`OBSERVED`

---

## M5.4-F04 — Complexity is a later evaluation/tuning question

Evidence:

Depth is 44 for W_SHORT and 41 for W_LONG under `max_depth=None`.

Finding:

Tree complexity is high enough to justify later examination of generalization/pruning behavior.

Boundary:

M5.4 does not have sufficient evidence to claim overfitting or to select a pruning configuration.

Status:

`OPEN — LATER EVALUATION / M7`

---

## M5.4-F05 — W_SHORT shows stronger fraud-class baseline metrics within DT-B01

Observed:

```text
W_SHORT:
F1       ≈ 0.3271
Recall   ≈ 0.3137
Precision≈ 0.3416

W_LONG:
F1       ≈ 0.1991
Recall   ≈ 0.1968
Precision≈ 0.2016
```

Finding:

Within this controlled Decision Tree baseline pair, W_SHORT produces higher fraud-class F1, Recall and Precision on the shared validation set.

Status:

`OBSERVED BASELINE EVIDENCE`

---

## M5.4-F06 — W_LONG does not simply predict fewer positives

Evidence:

```text
Predicted positive:

W_SHORT = 966
W_LONG  = 1,027
```

Yet:

```text
TP:
330 vs 207

FP:
636 vs 820
```

Finding:

The W_LONG degradation in DT-B01 is associated with less precise positive classification, not merely a lower positive-prediction rate.

Status:

`OBSERVED`

---

## M5.4-F07 — Accuracy is insufficient for this task

Evidence:

Accuracy remains around `0.998` for both runs while fraud Recall differs materially and remains well below 1.

Finding:

The runtime evidence is consistent with the project rule that Accuracy is reference-only.

Status:

`VERIFIED`

---

## M5.4-F08 — Probability evidence is available for both trees

Evidence:

Both probability/default-rule gates PASS and both risk-score artifacts survive round-trip.

Finding:

Later threshold/evaluation work can consume persisted Decision Tree probability evidence.

Status:

`VERIFIED`

---

## M5.4-F09 — Controlled pair remains methodologically valid

Evidence:

18 controlled fields match between W_SHORT and W_LONG.

Finding:

Observed metric/structure differences are valid Decision Tree training-window baseline evidence under the current contract.

Status:

`VERIFIED`

---

## M5.4-F10 — Training-window winner remains open

Finding:

M5.4 records DT-specific evidence but does not declare W_SHORT or W_LONG the final training window.

Status:

`OPEN`

---

## M5.4-F11 — Model-family winner remains open

Finding:

M5.4 does not rank Decision Tree against Logistic Regression or future Random Forest as final model.

Status:

`OPEN — M6/M7`

---

## M5.4-F12 — FINAL TEST remains protected

Evidence:

FINAL TEST isolation gate PASS.

Status:

`VERIFIED`

---

## M5.4-F13 — M5.5 handoff is unblocked

Evidence:

A complete, persisted, controlled Decision Tree baseline pair now exists.

Status:

`READY FOR M5.5`

# 27. Decision Log M5.4

## M5.4-D01 — Official Decision Tree baseline config

Decision:

`DT-B01-DEFAULT-GINI-UNPRUNED`

```text
criterion         = gini
splitter          = best
max_depth         = None
min_samples_split = 2
min_samples_leaf  = 1
max_features      = None
class_weight      = None
ccp_alpha         = 0.0
random_state      = 42
```

Status:

`LOCKED FOR M5 BASELINE`

---

## M5.4-D02 — Official Decision Tree runs

Decision:

```text
M5-DT-SHORT-B01
M5-DT-LONG-B01
```

Status:

`RUNTIME VERIFIED`

---

## M5.4-D03 — Complexity policy

Decision:

Observed unpruned depth/node/leaf counts are baseline evidence.

M5.4 does not modify depth, leaf-size or pruning parameters after reading validation result.

Status:

`LOCKED`

Complexity tuning:

`DEFERRED TO M7`

---

## M5.4-D04 — Imbalance strategy

Decision:

`IMBALANCE_STRATEGY = NONE`

`class_weight = None`

Status:

`INHERITED — VERIFIED — LOCKED`

---

## M5.4-D05 — Training-window comparison

Decision:

W_SHORT/W_LONG pair is a controlled comparison using the same Decision Tree config.

Status:

`VERIFIED`

Training-window winner:

`OPEN`

---

## M5.4-D06 — Metric contract

Decision:

Continue using:

- F1_fraud as primary;
- Recall_fraud;
- Precision_fraud;
- TP/FP/FN/TN;
- predicted-positive count/rate;
- Accuracy as reference only.

Status:

`INHERITED — VERIFIED — LOCKED`

---

## M5.4-D07 — Probability evidence

Decision:

Persist positive-class `predict_proba` score for both official runs.

Status:

`LOCKED`

---

## M5.4-D08 — Baseline decision rule

Decision:

Use estimator default `predict()` behavior, verified against probability argmax.

Final threshold:

`OPEN`

Status:

`LOCKED FOR BASELINE / FINAL OPEN`

---

## M5.4-D09 — Training-window interpretation

Decision:

M5.4 may record that W_SHORT has stronger DT-B01 fraud metrics on current validation.

M5.4 does not convert that observation into the final training-window selection.

Status:

`WINNER OPEN`

---

## M5.4-D10 — Model-family interpretation

Decision:

M5.4 does not select Decision Tree, Logistic Regression, or any future Random Forest as final model.

Status:

`OPEN — M6/M7`

---

## M5.4-D11 — FINAL TEST

Decision:

No access.

Status:

`INHERITED — VERIFIED — LOCKED`

---

## M5.4-D12 — Artifact handoff

Decision:

Retain predictions, probability/risk scores, summaries, config lock and pair manifest as official M5.4 baseline evidence.

Status:

`LOCKED`

---

## M5.4-D13 — M5.5 handoff

Decision:

No M5.4 blocker remains before Random Forest baseline.

Status:

`READY`

# 28. M5.4 Gate

## Technical runtime gates

```text
G01_MANIFEST_CONTRACT          → PASS
G02_INPUT_INTEGRITY            → PASS
G03_DT_CONFIG_DEFINED          → PASS
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

## Review gate R02 — Both official DT runs valid?

Evidence:

```text
DT-SHORT-B01:
completed
warning_count = 0

DT-LONG-B01:
completed
warning_count = 0
```

Result:

`PASS`

---

## Review gate R03 — Controlled comparison intact?

Evidence:

18 controlled fields match; training window is the intentional varying factor.

Result:

`PASS`

---

## Review gate R04 — No imbalance intervention?

Evidence:

```text
class_weight = None
IMBALANCE_STRATEGY = NONE
```

Result:

`PASS`

---

## Review gate R05 — No post-result complexity tuning?

Evidence:

Official config remains the pre-result locked B01 unpruned baseline.

Result:

`PASS`

---

## Review gate R06 — Probability evidence available?

Evidence:

Both probability/default-rule gates and persistence round-trip PASS.

Result:

`PASS`

---

## Review gate R07 — FINAL TEST protected?

Evidence:

Final-test isolation runtime gate PASS.

Result:

`PASS`

---

## Overall M5.4 Gate

```text
Technical gates:
14 / 14 PASS

Runtime review gates:
7 / 7 PASS

Blocking issue:
NONE
```

Final:

`M5.4 — PASS`

# 29. Kết luận M5.4

M5.4 đã tạo thành công một controlled Decision Tree baseline pair trên canonical M4.7 representation.

Official model config:

`DT-B01-DEFAULT-GINI-UNPRUNED`

Official runs:

```text
M5-DT-SHORT-B01
M5-DT-LONG-B01
```

Technical behavior:

```text
W_SHORT:
fit ≈ 6.66 s
depth = 44
nodes = 3,547
leaves = 1,774
warnings = 0

W_LONG:
fit ≈ 116.47 s
depth = 41
nodes = 28,651
leaves = 14,326
warnings = 0
```

Fraud-class baseline evidence:

```text
W_SHORT:
F1_fraud        ≈ 0.3271
Recall_fraud    ≈ 0.3137
Precision_fraud ≈ 0.3416
TP / FP         = 330 / 636
FN / TN         = 722 / 710,770
Predicted +     = 966

W_LONG:
F1_fraud        ≈ 0.1991
Recall_fraud    ≈ 0.1968
Precision_fraud ≈ 0.2016
TP / FP         = 207 / 820
FN / TN         = 845 / 710,586
Predicted +     = 1,027
```

Được phép kết luận ở M5.4:

- cả hai Decision Tree baseline runs đều technically valid;
- full-data Decision Tree baseline khả thi trong current environment;
- unpruned W_LONG tree lớn hơn rất nhiều về node/leaf count;
- W_SHORT có fraud-class F1/Recall/Precision cao hơn W_LONG trong chính DT-B01 controlled pair;
- probability evidence đã được persist;
- Accuracy tiếp tục không đủ để đánh giá fraud-class behavior;
- FINAL TEST vẫn được bảo vệ.

M5.4 **không** khóa:

- overfitting diagnosis chỉ từ tree depth;
- optimal `max_depth`;
- optimal `min_samples_leaf`;
- pruning strategy;
- W_SHORT là final training-window winner;
- Decision Tree là final model;
- final imbalance strategy;
- final threshold;
- final-test performance.

Trạng thái cuối:

```text
M5.4 — PASS

Official Decision Tree Baseline:
DT-B01-DEFAULT-GINI-UNPRUNED

DT-SHORT-B01:
VALID

DT-LONG-B01:
VALID

Controlled Pair:
VERIFIED

Tree Complexity Evidence:
RECORDED

Imbalance Strategy:
NONE

Probability Evidence:
PERSISTED

Training-window Winner:
OPEN

Model-family Winner:
OPEN

Complexity Tuning:
DEFERRED TO M7

Final Threshold:
OPEN

FINAL TEST:
PROTECTED

Blocking Issue:
NONE

READY FOR M5.5
```

Bước tiếp theo:

`M5.5 — Random Forest baseline`

M5.5 phải khóa exact Random Forest baseline config trước khi đọc validation result và chạy controlled pair:

```text
RF-SHORT
RF-LONG
```
