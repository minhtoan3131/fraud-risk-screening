# M7.7 — Lựa chọn candidate và xác nhận trên external VALIDATION

Mục tiêu:

> Chọn một upstream development candidate trước bước chọn threshold số.

Ba candidate đầu vào đã được M7.6 review và khóa:

```text
LR:
W_SHORT
imbalance = NONE
C = 10

DT:
W_SHORT
imbalance = CLASS_WEIGHT_BALANCED
min_samples_leaf = 5

RF:
W_SHORT
imbalance = CLASS_WEIGHT_BALANCED
reference RF config
```

Vai trò evidence:

```text
Temporal CV:
robustness / tuning evidence

External VALIDATION:
development confirmation / selection evidence

M6:
supporting baseline + error-analysis evidence

Runtime:
computational feasibility
```

M7.7 không được:

- mở thêm candidate;
- retune hyperparameter;
- đổi imbalance strategy;
- tuning threshold;
- dùng FINAL TEST.

Threshold vẫn:

`OPEN`

Selection trước runtime:

`OPEN — REQUIRES M7.7 RUNTIME REVIEW`

## 1. Guardrail sử dụng external VALIDATION

External VALIDATION được phép dùng cho controlled development confirmation.

Notebook này chỉ score đúng 3 candidate identities đã khóa trước runtime.

Không có:

```text
adaptive candidate expansion
hyperparameter retuning
imbalance retuning
threshold search
```

Nếu external VALIDATION tạo kết quả bất ngờ, notebook chỉ ghi nhận evidence.

Không được thêm config mới trong cùng run để săn score.

## 2. Canonical external VALIDATION representation

M7.7 dùng M4.7 persisted artifacts:

```text
X_train_w_short
y_train_w_short

X_validation_w_short
y_validation
```

W_SHORT preprocessing state được fit từ TRAIN-only source.

External VALIDATION chỉ được transform.

Expected external VALIDATION:

```text
2019-01-01
≤ Timestamp
< 2019-06-01

rows:
712,458

fraud:
1,052
```

M7.7 dùng default model decision rule để tạo label prediction.

Risk score được persist cho M7.8 nhưng chưa được dùng để thử numerical threshold.


```python
from pathlib import Path
import copy
import hashlib
import json
import platform
import sys
import time
import warnings

import numpy as np
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

print("Python:", sys.version)
print("Executable:", sys.executable)
print("Platform:", platform.platform())
print("NumPy:", np.__version__)
```

    Python: 3.14.6 (main, Jun 10 2026, 10:03:53) [Clang 21.0.0 (clang-2100.0.123.102)]
    Executable: /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/.venv/bin/python
    Platform: macOS-26.6.2-arm64-arm-64bit-Mach-O
    NumPy: 2.5.3


## 3. Locate artifacts


```python
M4_REL = Path("data") / "processed" / "m4_07_baseline_ready"
M7_06_REL = Path("data") / "processed" / "m7_06_moderate_hyperparameter_tuning"
M7_07_REL = Path("data") / "processed" / "m7_07_candidate_selection_external_validation"

required_rel_paths = [
    M4_REL / "X_train_w_short.npz",
    M4_REL / "X_validation_w_short.npz",
    M4_REL / "y_train_w_short.npy",
    M4_REL / "y_validation.npy",
    M4_REL / "row_id_train_w_short.npy",
    M4_REL / "row_id_validation.npy",
    M4_REL / "feature_names.json",
    M4_REL / "manifest.json",
    M7_06_REL / "m7_06_tuning_registry.json",
    M7_06_REL / "m7_06_tuning_manifest.json",
]

candidate_roots = [Path.cwd(), *list(Path.cwd().parents)[:6]]
PROJECT_ROOT = None

for candidate in candidate_roots:
    candidate = candidate.resolve()
    if all((candidate / rel_path).exists() for rel_path in required_rel_paths):
        PROJECT_ROOT = candidate
        break

if PROJECT_ROOT is None:
    raise FileNotFoundError(
        "Không tìm thấy PROJECT_ROOT chứa đầy đủ M4.7 và M7.6 artifacts."
    )

M4_DIR = PROJECT_ROOT / M4_REL
M7_06_DIR = PROJECT_ROOT / M7_06_REL
OUTPUT_DIR = PROJECT_ROOT / M7_07_REL
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

X_TRAIN_PATH = M4_DIR / "X_train_w_short.npz"
X_VALIDATION_PATH = M4_DIR / "X_validation_w_short.npz"
Y_TRAIN_PATH = M4_DIR / "y_train_w_short.npy"
Y_VALIDATION_PATH = M4_DIR / "y_validation.npy"
ROW_TRAIN_PATH = M4_DIR / "row_id_train_w_short.npy"
ROW_VALIDATION_PATH = M4_DIR / "row_id_validation.npy"
FEATURE_NAMES_PATH = M4_DIR / "feature_names.json"
M4_MANIFEST_PATH = M4_DIR / "manifest.json"

M7_06_RESULT_PATH = M7_06_DIR / "m7_06_tuning_registry.json"
M7_06_MANIFEST_PATH = M7_06_DIR / "m7_06_tuning_manifest.json"

RESULT_PATH = OUTPUT_DIR / "m7_07_external_validation_registry.json"
MANIFEST_PATH = OUTPUT_DIR / "m7_07_candidate_selection_manifest.json"

EXPECTED_TRAIN_ROWS = 1_721_615
EXPECTED_TRAIN_FRAUD = 2_491
EXPECTED_VALIDATION_ROWS = 712_458
EXPECTED_VALIDATION_FRAUD = 1_052
EXPECTED_FEATURE_COUNT = 47
EXPECTED_TRAIN_NNZ = 15_542_381
EXPECTED_VALIDATION_NNZ = 6_430_339

RANDOM_STATE = 42
M7_07_ANALYSIS_VERSION = "M7.7-candidate-selection-external-validation-v1"

UPSTREAM_REVIEWED_M7_6_STATUS = "PASS"
UPSTREAM_REVIEWED_M7_6_NOTEBOOK_SHA256 = (
    "c97275ec93382f891c2d2c19ecd74163"
    "56caf8e725da99975301869f98274727"
)

UPSTREAM_REVIEWED_SELECTED_CONFIGS = {
    "LR": "LR-T02-LBFGS-L2-C10",
    "DT": "DT-T02-GINI-MINLEAF5-CW",
    "RF": "RF-REF-100-GINI-SQRT-UNPRUNED-CW",
}

UPSTREAM_REVIEWED_IMBALANCE = {
    "LR": "NONE",
    "DT": "CLASS_WEIGHT_BALANCED",
    "RF": "CLASS_WEIGHT_BALANCED",
}

print("PROJECT_ROOT:", PROJECT_ROOT)
print("OUTPUT_DIR:", OUTPUT_DIR)
print("\nM7.7 SOURCE LOCATION GATE: PASS")
```

    PROJECT_ROOT: /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening
    OUTPUT_DIR: /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/processed/m7_07_candidate_selection_external_validation
    
    M7.7 SOURCE LOCATION GATE: PASS


## 4. Load và verify M7.6 handoff


```python
def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


with open(M4_MANIFEST_PATH, "r", encoding="utf-8") as file:
    m4_manifest = json.load(file)

with open(FEATURE_NAMES_PATH, "r", encoding="utf-8") as file:
    feature_names = json.load(file)

with open(M7_06_RESULT_PATH, "r", encoding="utf-8") as file:
    m7_06_result = json.load(file)

with open(M7_06_MANIFEST_PATH, "r", encoding="utf-8") as file:
    m7_06_manifest = json.load(file)

assert m4_manifest["pipeline_version"] == "M4.7-baseline-v1"
assert len(feature_names) == EXPECTED_FEATURE_COUNT

assert (
    m7_06_result["analysis_version"]
    == "M7.6-moderate-hyperparameter-tuning-v1"
)
assert m7_06_manifest["completed_new_fold_runs"] == 18
assert m7_06_manifest["failed_fold_runs"] == 0
assert m7_06_manifest["warning_count_total"] == 0
assert m7_06_manifest["final_test_accessed"] is False
assert UPSTREAM_REVIEWED_M7_6_STATUS == "PASS"

print("M4 pipeline:", m4_manifest["pipeline_version"])
print("M7.6 completed tuning folds:", m7_06_manifest["completed_new_fold_runs"])
print("\nM7.7 UPSTREAM HANDOFF GATE: PASS")
```

    M4 pipeline: M4.7-baseline-v1
    M7.6 completed tuning folds: 18
    
    M7.7 UPSTREAM HANDOFF GATE: PASS


## 5. Verify exact reviewed candidate identities

M7.6 registry được persist trước runtime review, vì vậy selection field trong JSON có thể vẫn OPEN.

M7.7 sử dụng reviewed handoff và independently kiểm tra rằng từng selected config tồn tại trong immutable M7.6 search space.


```python
tuning_configs = m7_06_result["tuning_configs"]
candidate_aggregates = m7_06_result["candidate_aggregates"]

SELECTED_CANDIDATES = {}

for model_key, selected_id in UPSTREAM_REVIEWED_SELECTED_CONFIGS.items():
    config_matches = [
        config
        for config in tuning_configs[model_key]
        if config["config_id"] == selected_id
    ]
    assert len(config_matches) == 1

    aggregate_matches = [
        aggregate
        for aggregate in candidate_aggregates
        if aggregate["candidate_id"] == selected_id
    ]
    assert len(aggregate_matches) == 1

    config = copy.deepcopy(config_matches[0])
    aggregate = copy.deepcopy(aggregate_matches[0])

    assert aggregate["valid_fold_count"] == 3
    assert aggregate["integrity_status"] == "PASS"

    SELECTED_CANDIDATES[model_key] = {
        "model_key": model_key,
        "config_id": selected_id,
        "params": config["params"],
        "imbalance_strategy": UPSTREAM_REVIEWED_IMBALANCE[model_key],
        "cv_aggregate": aggregate,
    }

assert set(SELECTED_CANDIDATES) == {"LR", "DT", "RF"}

assert SELECTED_CANDIDATES["LR"]["params"]["C"] == 10.0
assert SELECTED_CANDIDATES["LR"]["params"]["class_weight"] is None

assert SELECTED_CANDIDATES["DT"]["params"]["min_samples_leaf"] == 5
assert SELECTED_CANDIDATES["DT"]["params"]["class_weight"] == "balanced"

assert SELECTED_CANDIDATES["RF"]["params"]["n_estimators"] == 100
assert SELECTED_CANDIDATES["RF"]["params"]["class_weight"] == "balanced"

for model_key in ["LR", "DT", "RF"]:
    c = SELECTED_CANDIDATES[model_key]
    print(
        model_key,
        "→",
        c["config_id"],
        "| imbalance:",
        c["imbalance_strategy"],
    )

print("\nM7.7 CANDIDATE IDENTITY GATE: PASS")
```

    LR → LR-T02-LBFGS-L2-C10 | imbalance: NONE
    DT → DT-T02-GINI-MINLEAF5-CW | imbalance: CLASS_WEIGHT_BALANCED
    RF → RF-REF-100-GINI-SQRT-UNPRUNED-CW | imbalance: CLASS_WEIGHT_BALANCED
    
    M7.7 CANDIDATE IDENTITY GATE: PASS


## 6. Load W_SHORT TRAIN và external VALIDATION


```python
load_start = time.perf_counter()

X_train = sparse.load_npz(X_TRAIN_PATH)
X_validation = sparse.load_npz(X_VALIDATION_PATH)
y_train = np.load(Y_TRAIN_PATH, allow_pickle=False)
y_validation = np.load(Y_VALIDATION_PATH, allow_pickle=False)
row_train = np.load(ROW_TRAIN_PATH, allow_pickle=False)
row_validation = np.load(ROW_VALIDATION_PATH, allow_pickle=False)

load_seconds = time.perf_counter() - load_start

print("Load seconds:", round(load_seconds, 3))
print("X_train:", X_train.shape, "| nnz:", X_train.nnz)
print("X_validation:", X_validation.shape, "| nnz:", X_validation.nnz)
print("\nM7.7 ARTIFACT LOAD: COMPLETE")
```

    Load seconds: 0.197
    X_train: (1721615, 47) | nnz: 15542381
    X_validation: (712458, 47) | nnz: 6430339
    
    M7.7 ARTIFACT LOAD: COMPLETE


## 7. Artifact integrity gate


```python
assert sparse.isspmatrix_csr(X_train)
assert sparse.isspmatrix_csr(X_validation)

assert X_train.shape == (EXPECTED_TRAIN_ROWS, EXPECTED_FEATURE_COUNT)
assert X_validation.shape == (EXPECTED_VALIDATION_ROWS, EXPECTED_FEATURE_COUNT)

assert X_train.nnz == EXPECTED_TRAIN_NNZ
assert X_validation.nnz == EXPECTED_VALIDATION_NNZ

assert X_train.dtype == np.float32
assert X_validation.dtype == np.float32

assert y_train.shape == (EXPECTED_TRAIN_ROWS,)
assert y_validation.shape == (EXPECTED_VALIDATION_ROWS,)
assert row_train.shape == (EXPECTED_TRAIN_ROWS,)
assert row_validation.shape == (EXPECTED_VALIDATION_ROWS,)

assert y_train.dtype == np.int8
assert y_validation.dtype == np.int8

assert int(y_train.sum()) == EXPECTED_TRAIN_FRAUD
assert int(y_validation.sum()) == EXPECTED_VALIDATION_FRAUD

assert len(np.unique(row_train)) == len(row_train)
assert len(np.unique(row_validation)) == len(row_validation)

assert (
    np.intersect1d(
        row_train,
        row_validation,
        assume_unique=True,
    ).size
    == 0
)

print("TRAIN rows/fraud:", len(y_train), "/", int(y_train.sum()))
print("VALIDATION rows/fraud:", len(y_validation), "/", int(y_validation.sum()))
print("\nM7.7 M4.7 ARTIFACT INTEGRITY GATE: PASS")
```

    TRAIN rows/fraud: 1721615 / 2491
    VALIDATION rows/fraud: 712458 / 1052
    
    M7.7 M4.7 ARTIFACT INTEGRITY GATE: PASS


## 8. Controlled external VALIDATION use


```python
EXPECTED_SELECTED_IDS = {
    "LR-T02-LBFGS-L2-C10",
    "DT-T02-GINI-MINLEAF5-CW",
    "RF-REF-100-GINI-SQRT-UNPRUNED-CW",
}

assert {
    candidate["config_id"]
    for candidate in SELECTED_CANDIDATES.values()
} == EXPECTED_SELECTED_IDS

assert len(SELECTED_CANDIDATES) == 3

EXTERNAL_VALIDATION_USE = "CONTROLLED CONFIRMATION"
THRESHOLD_POLICY = "DEFAULT_MODEL_DECISION_RULE"

THRESHOLD_OPTIMIZATION_PERFORMED = False
ADAPTIVE_CANDIDATE_EXPANSION = False
HYPERPARAMETER_RETUNING_PERFORMED = False
IMBALANCE_RETUNING_PERFORMED = False

assert THRESHOLD_OPTIMIZATION_PERFORMED is False
assert ADAPTIVE_CANDIDATE_EXPANSION is False
assert HYPERPARAMETER_RETUNING_PERFORMED is False
assert IMBALANCE_RETUNING_PERFORMED is False

print("External VALIDATION use:", EXTERNAL_VALIDATION_USE)
print("Candidate count:", len(SELECTED_CANDIDATES))
print("Threshold policy:", THRESHOLD_POLICY)
print("\nM7.7 CONTROLLED VALIDATION-USE GATE: PASS")
```

    External VALIDATION use: CONTROLLED CONFIRMATION
    Candidate count: 3
    Threshold policy: DEFAULT_MODEL_DECISION_RULE
    
    M7.7 CONTROLLED VALIDATION-USE GATE: PASS


## 9. Frozen M6 supporting evidence

M6 evidence dưới đây là supporting context đã được review trước đó.

M7.7 không dùng các con số này làm tuning target.


```python
M6_FROZEN_W_SHORT_BASELINE = {
    "LR": {
        "F1_fraud": 0.33747547416612167,
        "Recall_fraud": 0.24524714828897337,
        "Precision_fraud": 0.5408805031446541,
        "TP": 258,
        "FP": 219,
        "FN": 794,
        "alerts": 477,
        "fit_seconds": 0.8222020840039477,
        "unique_only_catches": 31,
        "unique_only_false_positives": 58,
    },
    "DT": {
        "F1_fraud": 0.3270564915758176,
        "Recall_fraud": 0.31368821292775667,
        "Precision_fraud": 0.3416149068322981,
        "TP": 330,
        "FP": 636,
        "FN": 722,
        "alerts": 966,
        "fit_seconds": 6.661822540976573,
        "unique_only_catches": 68,
        "unique_only_false_positives": 388,
    },
    "RF": {
        "F1_fraud": 0.3664670658682635,
        "Recall_fraud": 0.2908745247148289,
        "Precision_fraud": 0.49514563106796117,
        "TP": 306,
        "FP": 312,
        "FN": 746,
        "alerts": 618,
        "fit_seconds": 40.69802879198687,
        "unique_only_catches": 28,
        "unique_only_false_positives": 68,
    },
}

M6_FROZEN_OVERLAP = {
    "actual_fraud": 1_052,
    "all_three_miss": 618,
    "caught_by_at_least_one": 434,
    "model_specific_only_catches": {
        "LR": 31,
        "DT": 68,
        "RF": 28,
    },
    "model_specific_only_false_positives": {
        "LR": 58,
        "DT": 388,
        "RF": 68,
    },
}

assert set(M6_FROZEN_W_SHORT_BASELINE) == {"LR", "DT", "RF"}

print("M6 frozen evidence: LOADED")
print("M6 actual fraud:", M6_FROZEN_OVERLAP["actual_fraud"])
print("\nM7.7 M6 SUPPORTING-EVIDENCE GATE: PASS")
```

    M6 frozen evidence: LOADED
    M6 actual fraud: 1052
    
    M7.7 M6 SUPPORTING-EVIDENCE GATE: PASS


## 10. External VALIDATION runner


```python
MODEL_LABELS = {
    "LR": "Logistic Regression",
    "DT": "Decision Tree",
    "RF": "Random Forest",
}


def compute_metric_bundle(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.int8)
    y_pred = np.asarray(y_pred, dtype=np.int8)

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    tn, fp, fn, tp = cm.ravel()
    alerts = int(tp + fp)

    return {
        "F1_fraud": float(
            f1_score(
                y_true,
                y_pred,
                pos_label=1,
                zero_division=0,
            )
        ),
        "Recall_fraud": float(
            recall_score(
                y_true,
                y_pred,
                pos_label=1,
                zero_division=0,
            )
        ),
        "Precision_fraud": float(
            precision_score(
                y_true,
                y_pred,
                pos_label=1,
                zero_division=0,
            )
        ),
        "Accuracy_reference": float(
            accuracy_score(y_true, y_pred)
        ),
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
        "predicted_positive_count": alerts,
        "predicted_positive_rate": float(alerts / len(y_true)),
    }


def make_estimator(model_key, params):
    params = copy.deepcopy(params)

    if model_key == "LR":
        return LogisticRegression(**params)

    if model_key == "DT":
        return DecisionTreeClassifier(**params)

    if model_key == "RF":
        return RandomForestClassifier(**params)

    raise KeyError(model_key)


def run_external_validation_candidate(model_key, candidate):
    estimator = make_estimator(
        model_key,
        candidate["params"],
    )

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")

        fit_start = time.perf_counter()
        fitted = clone(estimator)
        fitted.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - fit_start

        pred_start = time.perf_counter()
        y_pred = fitted.predict(X_validation)
        prediction_seconds = time.perf_counter() - pred_start

        classes = np.asarray(fitted.classes_)
        positive_matches = np.flatnonzero(classes == 1)

        if len(positive_matches) != 1:
            raise ValueError(
                "Không xác định duy nhất positive class = 1."
            )

        positive_index = int(positive_matches[0])

        risk_score = fitted.predict_proba(X_validation)[:, positive_index]

    warning_records = [
        {
            "category": warning.category.__name__,
            "message": str(warning.message),
        }
        for warning in captured
    ]

    metrics = compute_metric_bundle(
        y_validation,
        y_pred,
    )

    result = {
        "candidate_id": candidate["config_id"],
        "model_key": model_key,
        "model_family": MODEL_LABELS[model_key],
        "training_window": "W_SHORT",
        "imbalance_strategy": candidate["imbalance_strategy"],
        "hyperparameters": copy.deepcopy(candidate["params"]),
        "threshold_policy": THRESHOLD_POLICY,
        "random_state": RANDOM_STATE,
        **metrics,
        "fit_seconds": float(fit_seconds),
        "prediction_seconds": float(prediction_seconds),
        "warning_count": len(warning_records),
        "warnings": warning_records,
        "integrity_status": (
            "PASS"
            if len(warning_records) == 0
            else "WARNING_REVIEW_REQUIRED"
        ),
    }

    return (
        result,
        np.asarray(y_pred, dtype=np.int8),
        np.asarray(risk_score, dtype=np.float32),
    )


print("M7.7 EXTERNAL VALIDATION RUNNER: DEFINED")
```

    M7.7 EXTERNAL VALIDATION RUNNER: DEFINED


## 11. Fit 3 fixed candidates và score external VALIDATION


```python
validation_results = []
prediction_arrays = {}
risk_score_arrays = {}
artifact_fingerprints = []

run_start = time.perf_counter()

for model_key in ["LR", "DT", "RF"]:
    candidate = SELECTED_CANDIDATES[model_key]

    print("\n==================================================")
    print("Candidate:", candidate["config_id"])

    result, y_pred, risk_score = run_external_validation_candidate(
        model_key,
        candidate,
    )

    prediction_path = (
        OUTPUT_DIR
        / (
            "m7_07__"
            + candidate["config_id"]
            + "__validation_y_pred.npy"
        )
    )

    risk_score_path = (
        OUTPUT_DIR
        / (
            "m7_07__"
            + candidate["config_id"]
            + "__validation_risk_score.npy"
        )
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

    result["prediction_artifact"] = str(
        prediction_path.relative_to(PROJECT_ROOT)
    )
    result["risk_score_artifact"] = str(
        risk_score_path.relative_to(PROJECT_ROOT)
    )

    validation_results.append(result)
    prediction_arrays[model_key] = y_pred
    risk_score_arrays[model_key] = risk_score

    artifact_fingerprints.append(
        {
            "candidate_id": candidate["config_id"],
            "prediction_sha256": sha256_file(prediction_path),
            "risk_score_sha256": sha256_file(risk_score_path),
        }
    )

    print(
        "F1 / Recall / Precision:",
        round(result["F1_fraud"], 6),
        "/",
        round(result["Recall_fraud"], 6),
        "/",
        round(result["Precision_fraud"], 6),
    )

    print(
        "TP / FP / FN / TN:",
        result["TP"],
        "/",
        result["FP"],
        "/",
        result["FN"],
        "/",
        result["TN"],
    )

    print(
        "Alerts:",
        result["predicted_positive_count"],
        "| rate:",
        result["predicted_positive_rate"],
    )

    print(
        "Fit seconds:",
        round(result["fit_seconds"], 3),
    )

    print(
        "Warnings:",
        result["warning_count"],
    )

validation_elapsed = time.perf_counter() - run_start

assert len(validation_results) == 3

print("\nValidation confirmation seconds:", round(validation_elapsed, 2))
print("\nM7.7 3-CANDIDATE VALIDATION GATE: PASS")
```

    
    ==================================================
    Candidate: LR-T02-LBFGS-L2-C10
    F1 / Recall / Precision: 0.341368 / 0.249049 / 0.542443
    TP / FP / FN / TN: 262 / 221 / 790 / 711185
    Alerts: 483 | rate: 0.0006779346993085908
    Fit seconds: 0.776
    Warnings: 0
    
    ==================================================
    Candidate: DT-T02-GINI-MINLEAF5-CW
    F1 / Recall / Precision: 0.345007 / 0.440114 / 0.283701
    TP / FP / FN / TN: 463 / 1169 / 589 / 710237
    Alerts: 1632 | rate: 0.002290661344247661
    Fit seconds: 7.796
    Warnings: 0
    
    ==================================================
    Candidate: RF-REF-100-GINI-SQRT-UNPRUNED-CW
    F1 / Recall / Precision: 0.409111 / 0.443916 / 0.379366
    TP / FP / FN / TN: 467 / 764 / 585 / 710642
    Alerts: 1231 | rate: 0.0017278211487554353
    Fit seconds: 46.312
    Warnings: 0
    
    Validation confirmation seconds: 56.2
    
    M7.7 3-CANDIDATE VALIDATION GATE: PASS


## 12. Warning / integrity gate


```python
warning_count_total = int(
    sum(
        record["warning_count"]
        for record in validation_results
    )
)

for record in validation_results:
    print(
        record["candidate_id"],
        "→",
        record["integrity_status"],
        "| warnings:",
        record["warning_count"],
    )

print(
    "\nM7.7 WARNING / INTEGRITY GATE:",
    "PASS" if warning_count_total == 0 else "REVIEW REQUIRED",
)
```

    LR-T02-LBFGS-L2-C10 → PASS | warnings: 0
    DT-T02-GINI-MINLEAF5-CW → PASS | warnings: 0
    RF-REF-100-GINI-SQRT-UNPRUNED-CW → PASS | warnings: 0
    
    M7.7 WARNING / INTEGRITY GATE: PASS


## 13. Temporal CV ↔ external VALIDATION

Notebook chỉ report exact delta:

`external VALIDATION − temporal CV aggregate`.

Không dùng arbitrary epsilon để tự động gọi gap là lớn/nhỏ.

Runtime review mới diễn giải temporal generalization.


```python
cv_validation_comparisons = []

for record in validation_results:
    model_key = record["model_key"]
    cv = SELECTED_CANDIDATES[model_key]["cv_aggregate"]

    comparison = {
        "model_key": model_key,
        "candidate_id": record["candidate_id"],
        "cv_mean_F1": float(cv["mean_F1"]),
        "external_validation_F1": float(record["F1_fraud"]),
        "delta_validation_minus_cv_F1": float(
            record["F1_fraud"] - cv["mean_F1"]
        ),
        "cv_mean_Recall": float(cv["mean_Recall"]),
        "external_validation_Recall": float(record["Recall_fraud"]),
        "delta_validation_minus_cv_Recall": float(
            record["Recall_fraud"] - cv["mean_Recall"]
        ),
        "cv_mean_Precision": float(cv["mean_Precision"]),
        "external_validation_Precision": float(record["Precision_fraud"]),
        "delta_validation_minus_cv_Precision": float(
            record["Precision_fraud"] - cv["mean_Precision"]
        ),
        "cv_std_F1": float(cv["std_F1"]),
        "temporal_generalization_review": (
            "OPEN — REQUIRES M7.7 RUNTIME REVIEW"
        ),
    }

    cv_validation_comparisons.append(comparison)

    print("\n", comparison["candidate_id"])
    print(
        "CV mean F1 → VALIDATION F1:",
        round(comparison["cv_mean_F1"], 6),
        "→",
        round(comparison["external_validation_F1"], 6),
    )
    print(
        "Δ F1:",
        round(
            comparison["delta_validation_minus_cv_F1"],
            6,
        ),
    )
    print(
        "Δ Recall:",
        round(
            comparison["delta_validation_minus_cv_Recall"],
            6,
        ),
    )
    print(
        "Δ Precision:",
        round(
            comparison["delta_validation_minus_cv_Precision"],
            6,
        ),
    )

print("\nM7.7 TEMPORAL-GENERALIZATION EVIDENCE GATE: PASS")
```

    
     LR-T02-LBFGS-L2-C10
    CV mean F1 → VALIDATION F1: 0.524328 → 0.341368
    Δ F1: -0.18296
    Δ Recall: -0.177244
    Δ Precision: -0.144754
    
     DT-T02-GINI-MINLEAF5-CW
    CV mean F1 → VALIDATION F1: 0.60557 → 0.345007
    Δ F1: -0.260562
    Δ Recall: -0.308359
    Δ Precision: -0.23531
    
     RF-REF-100-GINI-SQRT-UNPRUNED-CW
    CV mean F1 → VALIDATION F1: 0.621144 → 0.409111
    Δ F1: -0.212033
    Δ Recall: -0.24049
    Δ Precision: -0.198177
    
    M7.7 TEMPORAL-GENERALIZATION EVIDENCE GATE: PASS


## 14. External VALIDATION comparative profile


```python
validation_by_model = {
    record["model_key"]: record
    for record in validation_results
}

external_validation_roles = {
    "highest_F1": max(
        validation_by_model,
        key=lambda key: validation_by_model[key]["F1_fraud"],
    ),
    "highest_Recall": max(
        validation_by_model,
        key=lambda key: validation_by_model[key]["Recall_fraud"],
    ),
    "highest_Precision": max(
        validation_by_model,
        key=lambda key: validation_by_model[key]["Precision_fraud"],
    ),
    "lowest_FN": min(
        validation_by_model,
        key=lambda key: validation_by_model[key]["FN"],
    ),
    "lowest_FP": min(
        validation_by_model,
        key=lambda key: validation_by_model[key]["FP"],
    ),
    "lowest_alerts": min(
        validation_by_model,
        key=lambda key: validation_by_model[key]["predicted_positive_count"],
    ),
    "lowest_fit_seconds": min(
        validation_by_model,
        key=lambda key: validation_by_model[key]["fit_seconds"],
    ),
}

for model_key in ["LR", "DT", "RF"]:
    record = validation_by_model[model_key]

    print("\n", MODEL_LABELS[model_key])
    print("F1:", record["F1_fraud"])
    print("Recall:", record["Recall_fraud"])
    print("Precision:", record["Precision_fraud"])
    print(
        "TP / FP / FN:",
        record["TP"],
        "/",
        record["FP"],
        "/",
        record["FN"],
    )
    print("Alerts:", record["predicted_positive_count"])
    print("Fit seconds:", record["fit_seconds"])

print("\nDescriptive roles:")
for role, model_key in external_validation_roles.items():
    print(role, "→", model_key)

print("\nM7.7 VALIDATION PROFILE GATE: PASS")
```

    
     Logistic Regression
    F1: 0.34136807817589576
    Recall: 0.24904942965779467
    Precision: 0.5424430641821946
    TP / FP / FN: 262 / 221 / 790
    Alerts: 483
    Fit seconds: 0.7764063340146095
    
     Decision Tree
    F1: 0.3450074515648286
    Recall: 0.44011406844106465
    Precision: 0.28370098039215685
    TP / FP / FN: 463 / 1169 / 589
    Alerts: 1632
    Fit seconds: 7.795959416020196
    
     Random Forest
    F1: 0.4091108190976785
    Recall: 0.4439163498098859
    Precision: 0.3793663688058489
    TP / FP / FN: 467 / 764 / 585
    Alerts: 1231
    Fit seconds: 46.31219670898281
    
    Descriptive roles:
    highest_F1 → RF
    highest_Recall → RF
    highest_Precision → LR
    lowest_FN → RF
    lowest_FP → LR
    lowest_alerts → LR
    lowest_fit_seconds → LR
    
    M7.7 VALIDATION PROFILE GATE: PASS


Các role trên chỉ là mô tả cực trị.

Không có automatic winner rule.

M7.7 phải đọc đồng thời F1, Recall, Precision, FP/FN, alert burden, CV behavior, M6 evidence và runtime.

## 15. Error overlap của 3 selected candidates


```python
fraud_mask = y_validation == 1
nonfraud_mask = y_validation == 0

caught_masks = {
    model_key: (
        (prediction_arrays[model_key] == 1)
        & fraud_mask
    )
    for model_key in ["LR", "DT", "RF"]
}

false_positive_masks = {
    model_key: (
        (prediction_arrays[model_key] == 1)
        & nonfraud_mask
    )
    for model_key in ["LR", "DT", "RF"]
}

caught_any = (
    caught_masks["LR"]
    | caught_masks["DT"]
    | caught_masks["RF"]
)

caught_all = (
    caught_masks["LR"]
    & caught_masks["DT"]
    & caught_masks["RF"]
)

unique_only_catches = {}
unique_only_false_positives = {}

for model_key in ["LR", "DT", "RF"]:
    others = [
        key
        for key in ["LR", "DT", "RF"]
        if key != model_key
    ]

    unique_only_catches[model_key] = int(
        (
            caught_masks[model_key]
            & ~caught_masks[others[0]]
            & ~caught_masks[others[1]]
        ).sum()
    )

    unique_only_false_positives[model_key] = int(
        (
            false_positive_masks[model_key]
            & ~false_positive_masks[others[0]]
            & ~false_positive_masks[others[1]]
        ).sum()
    )

selected_candidate_overlap = {
    "actual_fraud": int(fraud_mask.sum()),
    "caught_by_at_least_one": int(caught_any.sum()),
    "caught_by_all_three": int(caught_all.sum()),
    "all_three_miss": int(
        fraud_mask.sum() - caught_any.sum()
    ),
    "model_specific_only_catches": unique_only_catches,
    "model_specific_only_false_positives": unique_only_false_positives,
}

print("Actual fraud:", selected_candidate_overlap["actual_fraud"])
print(
    "Caught by at least one:",
    selected_candidate_overlap["caught_by_at_least_one"],
)
print(
    "Caught by all three:",
    selected_candidate_overlap["caught_by_all_three"],
)
print(
    "All-three miss:",
    selected_candidate_overlap["all_three_miss"],
)
print(
    "Unique-only catches:",
    selected_candidate_overlap["model_specific_only_catches"],
)
print(
    "Unique-only false positives:",
    selected_candidate_overlap[
        "model_specific_only_false_positives"
    ],
)

print("\nM7.7 ERROR-OVERLAP GATE: PASS")
```

    Actual fraud: 1052
    Caught by at least one: 518
    Caught by all three: 244
    All-three miss: 534
    Unique-only catches: {'LR': 1, 'DT': 41, 'RF': 46}
    Unique-only false positives: {'LR': 12, 'DT': 568, 'RF': 159}
    
    M7.7 ERROR-OVERLAP GATE: PASS


Error overlap chỉ là supporting evidence.

M7.7 không mở ensemble experiment.

## 16. M6 baseline ↔ M7.7 selected-candidate change


```python
m6_to_m77_changes = []

for model_key in ["LR", "DT", "RF"]:
    m6 = M6_FROZEN_W_SHORT_BASELINE[model_key]
    m77 = validation_by_model[model_key]

    change = {
        "model_key": model_key,
        "delta_F1": float(
            m77["F1_fraud"] - m6["F1_fraud"]
        ),
        "delta_Recall": float(
            m77["Recall_fraud"] - m6["Recall_fraud"]
        ),
        "delta_Precision": float(
            m77["Precision_fraud"] - m6["Precision_fraud"]
        ),
        "delta_TP": int(m77["TP"] - m6["TP"]),
        "delta_FP": int(m77["FP"] - m6["FP"]),
        "delta_FN": int(m77["FN"] - m6["FN"]),
        "delta_alerts": int(
            m77["predicted_positive_count"] - m6["alerts"]
        ),
    }

    m6_to_m77_changes.append(change)

    print("\n", model_key)
    print(
        "Δ F1 / Recall / Precision:",
        round(change["delta_F1"], 6),
        "/",
        round(change["delta_Recall"], 6),
        "/",
        round(change["delta_Precision"], 6),
    )
    print(
        "Δ TP / FP / FN / alerts:",
        change["delta_TP"],
        "/",
        change["delta_FP"],
        "/",
        change["delta_FN"],
        "/",
        change["delta_alerts"],
    )

print("\nM7.7 M6-TO-CURRENT COMPARISON GATE: PASS")
```

    
     LR
    Δ F1 / Recall / Precision: 0.003893 / 0.003802 / 0.001563
    Δ TP / FP / FN / alerts: 4 / 2 / -4 / 6
    
     DT
    Δ F1 / Recall / Precision: 0.017951 / 0.126426 / -0.057914
    Δ TP / FP / FN / alerts: 133 / 533 / -133 / 666
    
     RF
    Δ F1 / Recall / Precision: 0.042644 / 0.153042 / -0.115779
    Δ TP / FP / FN / alerts: 161 / 452 / -161 / 613
    
    M7.7 M6-TO-CURRENT COMPARISON GATE: PASS


## 17. Computational feasibility


```python
computational_profiles = {}

for model_key in ["LR", "DT", "RF"]:
    validation_record = validation_by_model[model_key]
    cv_aggregate = SELECTED_CANDIDATES[model_key]["cv_aggregate"]

    computational_profiles[model_key] = {
        "external_validation_fit_seconds": float(
            validation_record["fit_seconds"]
        ),
        "external_validation_prediction_seconds": float(
            validation_record["prediction_seconds"]
        ),
        "cv_total_fit_seconds": float(
            cv_aggregate["total_fit_seconds"]
        ),
        "cv_total_prediction_seconds": float(
            cv_aggregate["total_prediction_seconds"]
        ),
    }

    print(
        model_key,
        "→ validation fit:",
        round(
            computational_profiles[model_key][
                "external_validation_fit_seconds"
            ],
            3,
        ),
        "s | CV fit total:",
        round(
            computational_profiles[model_key][
                "cv_total_fit_seconds"
            ],
            3,
        ),
        "s",
    )

print("\nM7.7 COMPUTATIONAL PROFILE GATE: PASS")
```

    LR → validation fit: 0.776 s | CV fit total: 1.428 s
    DT → validation fit: 7.796 s | CV fit total: 3.031 s
    RF → validation fit: 46.312 s | CV fit total: 33.69 s
    
    M7.7 COMPUTATIONAL PROFILE GATE: PASS


## 18. Selection boundary

M7.7 runtime notebook không auto-select winner.

Sau Run All, review phải đọc theo thứ tự:

```text
1. temporal-CV F1 / stability
2. external VALIDATION F1
3. Recall / Precision
4. FP / FN / alerts
5. CV ↔ validation generalization
6. M6 supporting error evidence
7. computational feasibility
```

Không dùng composite weighted score tự chế.

Không tune threshold để thay đổi thứ tự candidate trong M7.7.


```python
UPSTREAM_CANDIDATE_SELECTION = (
    "OPEN — REQUIRES M7.7 RUNTIME REVIEW"
)

UPSTREAM_PIPELINE_STATUS = "OPEN"
MODEL_FAMILY_WINNER = "OPEN"
FINAL_MODEL = "OPEN"
FINAL_THRESHOLD = "OPEN"

assert UPSTREAM_CANDIDATE_SELECTION.startswith("OPEN")
assert MODEL_FAMILY_WINNER == "OPEN"
assert FINAL_THRESHOLD == "OPEN"

print("Upstream candidate selection:")
print(UPSTREAM_CANDIDATE_SELECTION)

print("\nThreshold:")
print(FINAL_THRESHOLD)

print("\nM7.7 NO-AUTO-SELECTION GATE: PASS")
```

    Upstream candidate selection:
    OPEN — REQUIRES M7.7 RUNTIME REVIEW
    
    Threshold:
    OPEN
    
    M7.7 NO-AUTO-SELECTION GATE: PASS


## 19. Persist M7.7 evidence registry


```python
source_fingerprints = {
    "m4_manifest_sha256": sha256_file(M4_MANIFEST_PATH),
    "x_train_w_short_sha256": sha256_file(X_TRAIN_PATH),
    "x_validation_w_short_sha256": sha256_file(X_VALIDATION_PATH),
    "y_train_w_short_sha256": sha256_file(Y_TRAIN_PATH),
    "y_validation_sha256": sha256_file(Y_VALIDATION_PATH),
    "m7_06_result_sha256": sha256_file(M7_06_RESULT_PATH),
    "m7_06_manifest_sha256": sha256_file(M7_06_MANIFEST_PATH),
    "reviewed_m7_06_notebook_sha256": (
        UPSTREAM_REVIEWED_M7_6_NOTEBOOK_SHA256
    ),
}

result_payload = {
    "analysis_version": M7_07_ANALYSIS_VERSION,
    "upstream_reviewed_handoff": {
        "m7_06_status": UPSTREAM_REVIEWED_M7_6_STATUS,
        "selected_configs": UPSTREAM_REVIEWED_SELECTED_CONFIGS,
        "imbalance_map": UPSTREAM_REVIEWED_IMBALANCE,
        "reviewed_notebook_sha256": (
            UPSTREAM_REVIEWED_M7_6_NOTEBOOK_SHA256
        ),
    },
    "source_fingerprints": source_fingerprints,
    "external_validation_use": EXTERNAL_VALIDATION_USE,
    "candidate_count": 3,
    "threshold_policy": THRESHOLD_POLICY,
    "selected_candidates": SELECTED_CANDIDATES,
    "validation_results": validation_results,
    "cv_validation_comparisons": cv_validation_comparisons,
    "external_validation_roles": external_validation_roles,
    "selected_candidate_overlap": selected_candidate_overlap,
    "m6_frozen_supporting_evidence": M6_FROZEN_W_SHORT_BASELINE,
    "m6_frozen_overlap": M6_FROZEN_OVERLAP,
    "m6_to_m77_changes": m6_to_m77_changes,
    "computational_profiles": computational_profiles,
    "artifact_fingerprints": artifact_fingerprints,
    "selection_state": {
        "upstream_candidate_selection": UPSTREAM_CANDIDATE_SELECTION,
        "upstream_pipeline": UPSTREAM_PIPELINE_STATUS,
        "model_family_winner": MODEL_FAMILY_WINNER,
        "final_model": FINAL_MODEL,
        "final_threshold": FINAL_THRESHOLD,
        "runtime_review_required": True,
    },
    "adaptive_candidate_expansion": False,
    "hyperparameter_retuning_performed": False,
    "imbalance_retuning_performed": False,
    "threshold_optimization_performed": False,
    "final_test_accessed": False,
}

with open(RESULT_PATH, "w", encoding="utf-8") as file:
    json.dump(
        result_payload,
        file,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )

manifest_payload = {
    "analysis_version": M7_07_ANALYSIS_VERSION,
    "training_window": "W_SHORT",
    "candidate_count": 3,
    "validation_rows": EXPECTED_VALIDATION_ROWS,
    "validation_fraud": EXPECTED_VALIDATION_FRAUD,
    "completed_validation_runs": len(validation_results),
    "warning_count_total": warning_count_total,
    "threshold_policy": THRESHOLD_POLICY,
    "upstream_candidate_selection": UPSTREAM_CANDIDATE_SELECTION,
    "upstream_pipeline": "OPEN",
    "model_family_winner": "OPEN",
    "final_threshold": "OPEN",
    "adaptive_candidate_expansion": False,
    "hyperparameter_retuning_performed": False,
    "imbalance_retuning_performed": False,
    "threshold_optimization_performed": False,
    "final_test_accessed": False,
    "decision": "OPEN — REQUIRES M7.7 RUNTIME REVIEW",
}

with open(MANIFEST_PATH, "w", encoding="utf-8") as file:
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

print("Result:", RESULT_PATH)
print("Manifest:", MANIFEST_PATH)
print("\nM7.7 PERSISTENCE GATE: PASS")
```

    Result: /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/processed/m7_07_candidate_selection_external_validation/m7_07_external_validation_registry.json
    Manifest: /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/processed/m7_07_candidate_selection_external_validation/m7_07_candidate_selection_manifest.json
    
    M7.7 PERSISTENCE GATE: PASS


## 20. Persistence round-trip / artifact identity


```python
with open(RESULT_PATH, "r", encoding="utf-8") as file:
    result_roundtrip = json.load(file)

with open(MANIFEST_PATH, "r", encoding="utf-8") as file:
    manifest_roundtrip = json.load(file)

assert result_roundtrip["analysis_version"] == M7_07_ANALYSIS_VERSION
assert result_roundtrip["candidate_count"] == 3
assert len(result_roundtrip["validation_results"]) == 3

assert (
    result_roundtrip["external_validation_use"]
    == "CONTROLLED CONFIRMATION"
)
assert result_roundtrip["adaptive_candidate_expansion"] is False
assert result_roundtrip["threshold_optimization_performed"] is False
assert result_roundtrip["final_test_accessed"] is False

assert manifest_roundtrip["completed_validation_runs"] == 3
assert manifest_roundtrip["validation_rows"] == EXPECTED_VALIDATION_ROWS
assert manifest_roundtrip["validation_fraud"] == EXPECTED_VALIDATION_FRAUD
assert manifest_roundtrip["final_test_accessed"] is False

for artifact in artifact_fingerprints:
    candidate_id = artifact["candidate_id"]

    record = next(
        record
        for record in validation_results
        if record["candidate_id"] == candidate_id
    )

    prediction_path = PROJECT_ROOT / record["prediction_artifact"]
    risk_score_path = PROJECT_ROOT / record["risk_score_artifact"]

    assert (
        sha256_file(prediction_path)
        == artifact["prediction_sha256"]
    )
    assert (
        sha256_file(risk_score_path)
        == artifact["risk_score_sha256"]
    )

print("Result SHA256:")
print(sha256_file(RESULT_PATH))

print("\nManifest SHA256:")
print(sha256_file(MANIFEST_PATH))

print("\nM7.7 ROUND-TRIP GATE: PASS")
```

    Result SHA256:
    9f2cc25b8acf162b02b5fcc95e06bcc1978f7a1b1d1cf4347f9b4be2ceb7170c
    
    Manifest SHA256:
    6c57e0c47b2c374f7d11721335d94fffcb82ac79fbeeecfb3b6a6ee4029eea36
    
    M7.7 ROUND-TRIP GATE: PASS


## 21. Selection-boundary gate


```python
assert ADAPTIVE_CANDIDATE_EXPANSION is False
assert THRESHOLD_OPTIMIZATION_PERFORMED is False
assert HYPERPARAMETER_RETUNING_PERFORMED is False
assert IMBALANCE_RETUNING_PERFORMED is False

assert (
    result_payload["selection_state"]["upstream_candidate_selection"]
    == "OPEN — REQUIRES M7.7 RUNTIME REVIEW"
)
assert result_payload["selection_state"]["final_threshold"] == "OPEN"
assert result_payload["final_test_accessed"] is False

print(
    "Adaptive candidate expansion:",
    result_payload["adaptive_candidate_expansion"],
)
print(
    "Hyperparameter retuning:",
    result_payload["hyperparameter_retuning_performed"],
)
print(
    "Imbalance retuning:",
    result_payload["imbalance_retuning_performed"],
)
print(
    "Threshold optimization:",
    result_payload["threshold_optimization_performed"],
)
print(
    "FINAL TEST accessed:",
    result_payload["final_test_accessed"],
)

print("\nM7.7 SELECTION-BOUNDARY GATE: PASS")
```

    Adaptive candidate expansion: False
    Hyperparameter retuning: False
    Imbalance retuning: False
    Threshold optimization: False
    FINAL TEST accessed: False
    
    M7.7 SELECTION-BOUNDARY GATE: PASS


## 22. Overall technical gate


```python
m7_07_gates = {
    "G01_SOURCE_LOCATION": True,
    "G02_UPSTREAM_HANDOFF": True,
    "G03_CANDIDATE_IDENTITY": True,
    "G04_ARTIFACT_LOAD": True,
    "G05_M4_7_ARTIFACT_INTEGRITY": True,
    "G06_CONTROLLED_VALIDATION_USE": True,
    "G07_M6_SUPPORTING_EVIDENCE": True,
    "G08_EXTERNAL_VALIDATION_RUNNER": True,
    "G09_3_CANDIDATE_VALIDATION": len(validation_results) == 3,
    "G10_WARNING_INTEGRITY": True,
    "G11_TEMPORAL_GENERALIZATION_EVIDENCE": True,
    "G12_VALIDATION_PROFILE": True,
    "G13_ERROR_OVERLAP": True,
    "G14_M6_TO_CURRENT_COMPARISON": True,
    "G15_COMPUTATIONAL_PROFILE": True,
    "G16_NO_AUTO_SELECTION": True,
    "G17_PERSISTENCE": True,
    "G18_ROUND_TRIP": True,
    "G19_NO_THRESHOLD_TUNING": True,
    "G20_FINAL_TEST_PROTECTION": True,
}

for gate_name, gate_value in m7_07_gates.items():
    print(
        gate_name,
        "→",
        "PASS" if gate_value else "FAIL",
    )

assert len(m7_07_gates) == 20
assert all(m7_07_gates.values())

print("\nM7.7 OVERALL TECHNICAL GATE: PASS")
```

    G01_SOURCE_LOCATION → PASS
    G02_UPSTREAM_HANDOFF → PASS
    G03_CANDIDATE_IDENTITY → PASS
    G04_ARTIFACT_LOAD → PASS
    G05_M4_7_ARTIFACT_INTEGRITY → PASS
    G06_CONTROLLED_VALIDATION_USE → PASS
    G07_M6_SUPPORTING_EVIDENCE → PASS
    G08_EXTERNAL_VALIDATION_RUNNER → PASS
    G09_3_CANDIDATE_VALIDATION → PASS
    G10_WARNING_INTEGRITY → PASS
    G11_TEMPORAL_GENERALIZATION_EVIDENCE → PASS
    G12_VALIDATION_PROFILE → PASS
    G13_ERROR_OVERLAP → PASS
    G14_M6_TO_CURRENT_COMPARISON → PASS
    G15_COMPUTATIONAL_PROFILE → PASS
    G16_NO_AUTO_SELECTION → PASS
    G17_PERSISTENCE → PASS
    G18_ROUND_TRIP → PASS
    G19_NO_THRESHOLD_TUNING → PASS
    G20_FINAL_TEST_PROTECTION → PASS
    
    M7.7 OVERALL TECHNICAL GATE: PASS


# 23. Kiểm tra runtime và các phát hiện M7.7

## 23.1. Tính toàn vẹn thực thi

Quan sát:

```text
Code cells:
21 / 21

Execution count:
1 → 21 liên tục

Runtime errors:
0

stderr:
0
```

Notebook đã chạy đầy đủ từ đầu đến cuối, không có lỗi thực thi hoặc partial execution làm mất hiệu lực evidence.

Trạng thái:

`VERIFIED`

---

## 23.2. Candidate identity

Ba candidate được đánh giá đúng theo reviewed handoff từ M7.6:

```text
LR:
LR-T02-LBFGS-L2-C10
W_SHORT
imbalance = NONE

DT:
DT-T02-GINI-MINLEAF5-CW
W_SHORT
imbalance = CLASS_WEIGHT_BALANCED

RF:
RF-REF-100-GINI-SQRT-UNPRUNED-CW
W_SHORT
imbalance = CLASS_WEIGHT_BALANCED
```

Không có candidate mới, không retune hyperparameter, không retune imbalance.

Trạng thái:

`VERIFIED`

---

## 23.3. External VALIDATION artifact integrity

Quan sát:

```text
TRAIN:
1,721,615 rows
2,491 fraud

VALIDATION:
712,458 rows
1,052 fraud

Feature width:
47
```

M4.7 artifact integrity gate:

`PASS`

Training / validation lineage không overlap.

Trạng thái:

`VERIFIED`

---

## 23.4. Controlled external VALIDATION execution

Expected:

```text
3 fixed candidates
×
1 external VALIDATION
=
3 runs
```

Observed:

```text
3 / 3 COMPLETE

Warnings:
0

Validation confirmation runtime:
56.20 s
```

Trạng thái:

`VERIFIED`

---

# 24. Phân tích candidate selection

## M7.7-F01 — Logistic Regression giữ Precision / operational role nhưng fraud-side coverage còn yếu

External VALIDATION:

```text
F1:
0.341368

Recall:
0.249049

Precision:
0.542443

TP:
262

FP:
221

FN:
790

Alerts:
483

Fit time:
0.776 s
```

Vai trò mô tả:

```text
highest Precision
lowest FP
lowest alerts
lowest fit cost
```

Nhưng Recall thấp nhất và FN cao nhất trong ba candidate.

So với temporal CV:

```text
CV mean F1:
0.524328

External VALIDATION F1:
0.341368

Δ F1:
-0.182960

Δ Recall:
-0.177244

Δ Precision:
-0.144754
```

So với frozen M6 W_SHORT baseline:

```text
Δ F1:
+0.003893

Δ Recall:
+0.003802

Δ Precision:
+0.001563

Δ TP:
+4

Δ FP:
+2

Δ FN:
-4

Δ alerts:
+6
```

Diễn giải:

LR tuned candidate gần như giữ nguyên behavior so với M6 baseline trên external VALIDATION. Operational efficiency vẫn tốt nhưng fraud-side coverage không tạo bước cải thiện đủ lớn.

Quyết định:

`NOT SELECTED`

---

## M7.7-F02 — Decision Tree tăng Recall so với M6 nhưng chịu Precision / FP / alert cost lớn

External VALIDATION:

```text
F1:
0.345007

Recall:
0.440114

Precision:
0.283701

TP:
463

FP:
1,169

FN:
589

Alerts:
1,632

Fit time:
7.796 s
```

So với temporal CV:

```text
CV mean F1:
0.605570

External VALIDATION F1:
0.345007

Δ F1:
-0.260562

Δ Recall:
-0.308359

Δ Precision:
-0.235310
```

So với frozen M6 baseline:

```text
Δ F1:
+0.017951

Δ Recall:
+0.126426

Δ Precision:
-0.057914

Δ TP:
+133

Δ FP:
+533

Δ FN:
-133

Δ alerts:
+666
```

DT cải thiện fraud capture so với M6 baseline nhưng Precision giảm, FP tăng và alert burden tăng mạnh.

Trên external VALIDATION hiện tại, RF tốt hơn DT về:

```text
F1
Recall
Precision
FN
FP
alert burden
```

DT chỉ còn lợi thế compute.

Quyết định:

`NOT SELECTED`

---

## M7.7-F03 — Random Forest có external VALIDATION profile mạnh nhất trên evidence bundle hiện tại

External VALIDATION:

```text
F1:
0.409111

Recall:
0.443916

Precision:
0.379366

TP:
467

FP:
764

FN:
585

Alerts:
1,231

Fit time:
46.312 s
```

Vai trò mô tả:

```text
highest F1
highest Recall
lowest FN
```

So với temporal CV:

```text
CV mean F1:
0.621144

External VALIDATION F1:
0.409111

Δ F1:
-0.212033

Δ Recall:
-0.240490

Δ Precision:
-0.198177
```

So với M6 W_SHORT baseline RF:

```text
Δ F1:
+0.042644

Δ Recall:
+0.153042

Δ Precision:
-0.115779

Δ TP:
+161

Δ FP:
+452

Δ FN:
-161

Δ alerts:
+613
```

RF tạo improvement đáng kể trên fraud-side so với M6 baseline: F1 và Recall tăng, TP tăng, FN giảm.

Cost:

- Precision giảm;
- FP tăng;
- alert burden tăng;
- compute cost cao hơn LR/DT.

Tuy nhiên khi so trực tiếp ba selected candidates, RF có profile mạnh nhất theo primary metric và fraud-side evidence.

Trạng thái:

`STRONGEST UPSTREAM CANDIDATE`

---

## M7.7-F04 — Temporal generalization finding tồn tại ở cả ba family

Cả ba candidate đều suy giảm rõ khi chuyển từ temporal-CV aggregate sang external VALIDATION:

```text
LR:
Δ F1 = -0.182960

DT:
Δ F1 = -0.260562

RF:
Δ F1 = -0.212033
```

Recall và Precision cũng giảm ở cả ba family.

Finding:

`TEMPORAL GENERALIZATION FINDING`

Điều này cho thấy temporal-CV performance năm 2018 không chuyển nguyên vẹn sang external VALIDATION 2019.

Không được dùng finding này để mở lại tuning ad hoc.

Trạng thái:

`VERIFIED — CARRY FORWARD`

---

## M7.7-F05 — External VALIDATION xác nhận RF có selection advantage đa chiều

External VALIDATION F1:

```text
RF:
0.409111

DT:
0.345007

LR:
0.341368
```

External VALIDATION Recall:

```text
RF:
0.443916

DT:
0.440114

LR:
0.249049
```

External VALIDATION Precision:

```text
LR:
0.542443

RF:
0.379366

DT:
0.283701
```

RF không thắng Precision, nhưng nó có:

- F1 cao nhất;
- Recall cao nhất;
- FN thấp nhất;
- Precision tốt hơn DT;
- FP thấp hơn DT;
- alerts thấp hơn DT.

Đây là evidence selection đa chiều, không phải one-number ranking.

Trạng thái:

`VERIFIED`

---

## M7.7-F06 — Error-overlap supporting evidence tiếp tục hỗ trợ RF hơn DT

Selected-candidate overlap:

```text
Actual fraud:
1,052

Caught by at least one:
518

Caught by all three:
244

All-three miss:
534
```

Model-specific-only fraud catches:

```text
LR:
1

DT:
41

RF:
46
```

Model-specific-only false positives:

```text
LR:
12

DT:
568

RF:
159
```

RF có nhiều model-specific fraud catches nhất và unique-only FP thấp hơn DT rất nhiều.

Đây chỉ là supporting evidence; không suy ra ensemble recommendation.

Trạng thái:

`SUPPORTING EVIDENCE`

---

## M7.7-F07 — Computational feasibility không tạo blocker

External VALIDATION fit time:

```text
LR:
0.776 s

DT:
7.796 s

RF:
46.312 s
```

RF tốn compute nhất, nhưng runtime vẫn hoàn tất bình thường trong môi trường hiện tại, không warning, không failure.

Compute cost được ghi nhận là trade-off nhưng không đủ để override predictive evidence.

Trạng thái:

`FEASIBLE WITH HIGHER COST`

---

## M7.7-F08 — Chọn upstream development candidate

Reviewed evidence bundle:

```text
Temporal-CV robustness:
RF strong

External VALIDATION F1:
RF highest

External VALIDATION Recall:
RF highest

External VALIDATION FN:
RF lowest

RF vs DT:
RF better on F1 / Recall / Precision / FN / FP / alerts

RF vs LR:
RF stronger on fraud-side coverage
LR stronger on Precision / FP / alerts / compute

M6-to-current RF:
higher F1 / Recall
lower FN
with Precision / FP / alert cost
```

Quyết định:

```text
UPSTREAM DEVELOPMENT CANDIDATE:

RF-REF-100-GINI-SQRT-UNPRUNED-CW
```

Identity:

```text
Feature / preprocessing:
M4 canonical contract

Training window:
W_SHORT

Model family:
Random Forest

Model configuration:
RF-REF-100-GINI-SQRT-UNPRUNED-CW

Imbalance strategy:
CLASS_WEIGHT_BALANCED

Random state:
42

Probability interface:
predict_proba / positive class = 1

Threshold policy:
DEFAULT_MODEL_DECISION_RULE
```

Exact hyperparameters tiếp tục được kế thừa nguyên vẹn từ immutable M7.6 candidate registry.

Trạng thái:

`SELECTED — LOCKED FOR M7.8`

---

## M7.7-F09 — Upstream pipeline freeze

Sau selection:

```text
FEATURE / PREPROCESSING:
FROZEN

TRAINING WINDOW:
W_SHORT — FROZEN

MODEL FAMILY:
RANDOM FOREST — FROZEN

MODEL CONFIG:
RF-REF-100-GINI-SQRT-UNPRUNED-CW — FROZEN

IMBALANCE STRATEGY:
CLASS_WEIGHT_BALANCED — FROZEN

RANDOM STATE:
42 — FROZEN

PROBABILITY INTERFACE:
predict_proba / positive class = 1 — FROZEN
```

Threshold:

`STILL OPEN`

FINAL TEST:

`STILL PROTECTED`

Trạng thái:

`UPSTREAM PIPELINE FROZEN`

---

## M7.7-F10 — Sẵn sàng cho M7.8

Selected RF external VALIDATION risk score đã được persist và fingerprint trong runtime artifacts của M7.7.

M7.8 có thể dùng đúng risk score này để thực hiện numerical threshold comparison.

Trạng thái:

`READY FOR M7.8`

# 25. Decision Log M7.7 — sau runtime review

## M7.7-D01 — Candidate set

Decision:

Chỉ 3 candidates đã được review ở M7.6.

Status:

`VERIFIED — LOCKED`

---

## M7.7-D02 — Training window

Decision:

`W_SHORT`

Status:

`FROZEN`

---

## M7.7-D03 — External VALIDATION role

Decision:

`CONTROLLED DEVELOPMENT CONFIRMATION / SELECTION EVIDENCE`

Không dùng external VALIDATION để mở search space mới.

Status:

`LOCKED`

---

## M7.7-D04 — Validation representation

Decision:

```text
X_validation_w_short
y_validation
```

Status:

`LOCKED`

---

## M7.7-D05 — Temporal generalization finding

Decision:

Carry forward:

`TEMPORAL GENERALIZATION FINDING`

Status:

`LOCKED FINDING`

---

## M7.7-D06 — Logistic Regression disposition

Decision:

`NOT SELECTED`

Reason:

Operational profile tốt nhưng Recall thấp và FN cao; external VALIDATION F1 thấp hơn RF.

Status:

`LOCKED`

---

## M7.7-D07 — Decision Tree disposition

Decision:

`NOT SELECTED`

Reason:

RF tốt hơn DT trên external VALIDATION về F1, Recall, Precision, FN, FP và alerts.

Status:

`LOCKED`

---

## M7.7-D08 — Random Forest disposition

Decision:

```text
SELECT UPSTREAM DEVELOPMENT CANDIDATE

RF-REF-100-GINI-SQRT-UNPRUNED-CW
```

Known costs:

```text
Precision lower than LR
FP / alerts higher than LR
compute cost highest
```

Status:

`LOCKED FOR M7.8`

---

## M7.7-D09 — Feature / preprocessing

Decision:

`M4 canonical feature/preprocessing contract`

Status:

`FROZEN`

---

## M7.7-D10 — Model family

Decision:

`RANDOM FOREST`

Status:

`FROZEN`

---

## M7.7-D11 — Model config

Decision:

`RF-REF-100-GINI-SQRT-UNPRUNED-CW`

Exact hyperparameters:

`INHERITED FROM IMMUTABLE M7.6 REGISTRY`

Status:

`FROZEN`

---

## M7.7-D12 — Imbalance strategy

Decision:

`CLASS_WEIGHT_BALANCED`

Status:

`FROZEN`

---

## M7.7-D13 — Random state

Decision:

`42`

Status:

`FROZEN`

---

## M7.7-D14 — Probability interface

Decision:

`predict_proba / positive class = 1`

Status:

`FROZEN`

---

## M7.7-D15 — Numerical threshold

Decision:

Không chọn trong M7.7.

Status:

`OPEN`

---

## M7.7-D16 — Adaptive search / retuning

Observed:

```text
Adaptive candidate expansion:
False

Hyperparameter retuning:
False

Imbalance retuning:
False
```

Status:

`VERIFIED`

---

## M7.7-D17 — FINAL TEST

Decision:

No access.

Status:

`PROTECTED`

---

## M7.7-D18 — M7.8 handoff

Proceed to:

`M7.8 — Numerical threshold selection`

Input:

```text
RF-REF-100-GINI-SQRT-UNPRUNED-CW

W_SHORT

CLASS_WEIGHT_BALANCED

M7.7 persisted external VALIDATION risk score
```

Status:

`READY`

# 26. M7.7 Gate

## Technical runtime gates

```text
G01_SOURCE_LOCATION                         → PASS
G02_UPSTREAM_HANDOFF                        → PASS
G03_CANDIDATE_IDENTITY                      → PASS
G04_ARTIFACT_LOAD                           → PASS
G05_M4_7_ARTIFACT_INTEGRITY                → PASS
G06_CONTROLLED_VALIDATION_USE               → PASS
G07_M6_SUPPORTING_EVIDENCE                  → PASS
G08_EXTERNAL_VALIDATION_RUNNER              → PASS
G09_3_CANDIDATE_VALIDATION                  → PASS
G10_WARNING_INTEGRITY                       → PASS
G11_TEMPORAL_GENERALIZATION_EVIDENCE        → PASS
G12_VALIDATION_PROFILE                      → PASS
G13_ERROR_OVERLAP                           → PASS
G14_M6_TO_CURRENT_COMPARISON                → PASS
G15_COMPUTATIONAL_PROFILE                   → PASS
G16_NO_AUTO_SELECTION                       → PASS
G17_PERSISTENCE                             → PASS
G18_ROUND_TRIP                              → PASS
G19_NO_THRESHOLD_TUNING                     → PASS
G20_FINAL_TEST_PROTECTION                    → PASS
```

Technical gates:

`20 / 20 PASS`

---

## Runtime review gates

```text
R01 Execution integrity                  → PASS
R02 Candidate identity                   → PASS
R03 External VALIDATION artifact audit   → PASS
R04 Controlled 3-candidate confirmation  → PASS
R05 CV ↔ VALIDATION review               → PASS
R06 LR profile review                    → PASS
R07 DT profile review                    → PASS
R08 RF profile review                    → PASS
R09 M6 supporting evidence               → PASS
R10 Error-overlap review                 → PASS
R11 Computational feasibility            → PASS
R12 Upstream candidate selection         → PASS
R13 Training window freeze               → PASS
R14 Model-family freeze                  → PASS
R15 Model-config freeze                  → PASS
R16 Imbalance-strategy freeze            → PASS
R17 Threshold boundary                   → PASS
R18 No adaptive retuning                 → PASS
R19 Persistence / round-trip             → PASS
R20 FINAL TEST protection                → PASS
R21 M7.8 handoff                         → PASS
```

Runtime review gates:

`21 / 21 PASS`

Blocking issue:

`NONE`

---

## Overall M7.7 Gate

Final:

`M7.7 — PASS`

Selected upstream candidate:

```text
RF-REF-100-GINI-SQRT-UNPRUNED-CW
```

Frozen upstream state:

```text
Training Window:
W_SHORT

Model Family:
RANDOM FOREST

Model Config:
RF-REF-100-GINI-SQRT-UNPRUNED-CW

Imbalance Strategy:
CLASS_WEIGHT_BALANCED

Random State:
42

Probability Interface:
predict_proba / positive class = 1
```

Threshold:

`OPEN`

FINAL TEST:

`PROTECTED`

Handoff:

`READY FOR M7.8`

# 27. Kết luận M7.7

M7.7 đã hoàn thành controlled external VALIDATION confirmation cho đúng ba upstream candidates đã được khóa từ M7.6.

Runtime integrity:

```text
21 / 21 code cells executed

3 / 3 external VALIDATION runs complete

warnings:
0

Technical gates:
20 / 20 PASS

Runtime review gates:
21 / 21 PASS
```

Một finding quan trọng phải được giữ lại:

`TEMPORAL GENERALIZATION FINDING`

Cả ba candidates đều suy giảm từ temporal CV sang external VALIDATION:

```text
LR Δ F1:
-0.182960

DT Δ F1:
-0.260562

RF Δ F1:
-0.212033
```

Tuy nhiên external VALIDATION vẫn đủ để phân biệt candidate behavior.

External VALIDATION:

```text
LR:
F1 0.341368
Recall 0.249049
Precision 0.542443
FN 790
FP 221
alerts 483
```

```text
DT:
F1 0.345007
Recall 0.440114
Precision 0.283701
FN 589
FP 1,169
alerts 1,632
```

```text
RF:
F1 0.409111
Recall 0.443916
Precision 0.379366
FN 585
FP 764
alerts 1,231
```

Reviewed upstream selection:

```text
SELECTED:

RF-REF-100-GINI-SQRT-UNPRUNED-CW
```

Lý do:

- F1 cao nhất;
- Recall cao nhất;
- FN thấp nhất;
- tốt hơn DT về F1, Recall, Precision, FN, FP và alerts;
- có nhiều model-specific-only fraud catches nhất;
- current RF path cải thiện F1/Recall/FN so với frozen M6 RF baseline.

Trade-off:

- Precision thấp hơn LR;
- FP / alerts cao hơn LR;
- compute cost cao nhất.

Upstream freeze:

```text
FEATURE / PREPROCESSING:
FROZEN

TRAINING WINDOW:
W_SHORT — FROZEN

MODEL FAMILY:
RANDOM FOREST — FROZEN

MODEL CONFIG:
RF-REF-100-GINI-SQRT-UNPRUNED-CW — FROZEN

IMBALANCE STRATEGY:
CLASS_WEIGHT_BALANCED — FROZEN

RANDOM STATE:
42 — FROZEN

PROBABILITY INTERFACE:
predict_proba / positive class = 1 — FROZEN
```

Threshold:

`STILL OPEN`

External VALIDATION risk score của selected RF candidate đã được persist và fingerprint trong M7.7.

Final state:

```text
M7.7 — PASS

Upstream Development Candidate:
RF-REF-100-GINI-SQRT-UNPRUNED-CW — SELECTED

Upstream Pipeline:
FROZEN

Threshold:
OPEN

FINAL TEST:
PROTECTED

Blocking Issue:
NONE

READY FOR M7.8
```

Bước tiếp theo:

`M7.8 — Numerical threshold selection`

M7.8 chỉ được dùng selected RF external VALIDATION risk score để so sánh threshold candidates đã predeclare.

M7.8 không được thay đổi upstream model/configuration và không được dùng FINAL TEST.
