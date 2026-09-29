# M7.8 — Numerical threshold selection

Mục tiêu:

> Với upstream pipeline đã được M7.7 freeze, numerical threshold nào trên external VALIDATION risk score tạo trade-off phù hợp nhất theo metric contract hiện tại?

Upstream candidate đã khóa:

```text
Training Window:
W_SHORT

Model Family:
Random Forest

Model Config:
RF-REF-100-GINI-SQRT-UNPRUNED-CW

Imbalance Strategy:
CLASS_WEIGHT_BALANCED

Random State:
42

Probability Interface:
predict_proba / positive class = 1
```

M7.8 chỉ thay:

`numerical decision threshold`

Không được thay model/config, imbalance strategy, preprocessing, training window; không refit model; không calibration; không dùng FINAL TEST.

Runtime-dependent state:

`NOT YET VERIFIED`

## 1. Khóa threshold candidate set trước runtime

Exact predeclared set:

```text
0.20
0.30
0.40
0.50
0.60
```

Rationale:

```text
0.50:
default reference

0.40 / 0.30 / 0.20:
controlled lower thresholds để kiểm tra Recall/FN gain

0.60:
controlled higher threshold để kiểm tra Precision/FP/alert reduction
```

M6 từng cho thấy baseline RF W_SHORT có nhiều FN score dưới 0.40 và FP median quanh 0.625. Đây chỉ là supporting historical evidence, không phải distribution của selected M7.7 candidate.

Candidate set:

`IMMUTABLE FOR THIS RUN`

Không thêm threshold sau khi xem kết quả.

## 2. Threshold decision contract

Primary:

`F1_fraud`

Bắt buộc đọc cùng:

```text
Recall_fraud
Precision_fraud
TP / FP / FN / TN
predicted-positive count/rate
```

Reading order:

```text
F1
→ Recall / Precision
→ FP / FN
→ alert burden
→ comparison với default 0.50
```

Không dùng arbitrary epsilon.

Default 0.50 có thể vẫn là final threshold nếu evidence không justify thay đổi.

Notebook không auto-freeze final threshold; cần runtime review.


```python
from pathlib import Path
import hashlib
import json
import platform
import sys
import time

import numpy as np

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


## 3. Locate M4.7 và M7.7 artifacts


```python
M4_REL = Path("data") / "processed" / "m4_07_baseline_ready"
M7_07_REL = Path("data") / "processed" / "m7_07_candidate_selection_external_validation"
M7_08_REL = Path("data") / "processed" / "m7_08_numerical_threshold_selection"

required_rel_paths = [
    M4_REL / "y_validation.npy",
    M4_REL / "row_id_validation.npy",
    M4_REL / "manifest.json",
    M7_07_REL / "m7_07_external_validation_registry.json",
    M7_07_REL / "m7_07_candidate_selection_manifest.json",
]

candidate_roots = [Path.cwd(), *list(Path.cwd().parents)[:6]]
PROJECT_ROOT = None

for candidate in candidate_roots:
    candidate = candidate.resolve()
    if all((candidate / rel).exists() for rel in required_rel_paths):
        PROJECT_ROOT = candidate
        break

if PROJECT_ROOT is None:
    raise FileNotFoundError(
        "Không tìm thấy PROJECT_ROOT chứa đầy đủ M4.7 và M7.7 artifacts."
    )

M4_DIR = PROJECT_ROOT / M4_REL
M7_07_DIR = PROJECT_ROOT / M7_07_REL
OUTPUT_DIR = PROJECT_ROOT / M7_08_REL
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

Y_VALIDATION_PATH = M4_DIR / "y_validation.npy"
ROW_VALIDATION_PATH = M4_DIR / "row_id_validation.npy"
M4_MANIFEST_PATH = M4_DIR / "manifest.json"

M7_07_RESULT_PATH = M7_07_DIR / "m7_07_external_validation_registry.json"
M7_07_MANIFEST_PATH = M7_07_DIR / "m7_07_candidate_selection_manifest.json"

RESULT_PATH = OUTPUT_DIR / "m7_08_threshold_registry.json"
MANIFEST_PATH = OUTPUT_DIR / "m7_08_threshold_manifest.json"
PREDICTIONS_PATH = OUTPUT_DIR / "m7_08_threshold_predictions.npz"

EXPECTED_VALIDATION_ROWS = 712_458
EXPECTED_VALIDATION_FRAUD = 1_052

SELECTED_CANDIDATE_ID = "RF-REF-100-GINI-SQRT-UNPRUNED-CW"
SELECTED_MODEL_FAMILY = "Random Forest"
SELECTED_TRAINING_WINDOW = "W_SHORT"
SELECTED_IMBALANCE_STRATEGY = "CLASS_WEIGHT_BALANCED"
SELECTED_RANDOM_STATE = 42
SELECTED_PROBABILITY_INTERFACE = "predict_proba / positive class = 1"

UPSTREAM_REVIEWED_M7_7_STATUS = "PASS"
UPSTREAM_REVIEWED_M7_7_NOTEBOOK_SHA256 = (
    "b46f2aab72661ccb0ba3c07ee3dcc1f5"
    "8876560c9d9b24eeb197304d194c9dee"
)

M7_08_ANALYSIS_VERSION = "M7.8-numerical-threshold-selection-v1"

THRESHOLD_CANDIDATES = [0.20, 0.30, 0.40, 0.50, 0.60]
DEFAULT_THRESHOLD = 0.50

print("PROJECT_ROOT:", PROJECT_ROOT)
print("OUTPUT_DIR:", OUTPUT_DIR)
print("\nM7.8 SOURCE LOCATION GATE: PASS")
```

    PROJECT_ROOT: /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening
    OUTPUT_DIR: /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/processed/m7_08_numerical_threshold_selection
    
    M7.8 SOURCE LOCATION GATE: PASS


## 4. Verify M7.7 handoff


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

with open(M7_07_RESULT_PATH, "r", encoding="utf-8") as file:
    m7_07_result = json.load(file)

with open(M7_07_MANIFEST_PATH, "r", encoding="utf-8") as file:
    m7_07_manifest = json.load(file)

assert m4_manifest["pipeline_version"] == "M4.7-baseline-v1"
assert (
    m7_07_result["analysis_version"]
    == "M7.7-candidate-selection-external-validation-v1"
)
assert m7_07_manifest["completed_validation_runs"] == 3
assert m7_07_manifest["warning_count_total"] == 0
assert m7_07_manifest["final_test_accessed"] is False
assert UPSTREAM_REVIEWED_M7_7_STATUS == "PASS"
assert THRESHOLD_CANDIDATES == [0.20, 0.30, 0.40, 0.50, 0.60]
assert DEFAULT_THRESHOLD in THRESHOLD_CANDIDATES

print("Reviewed M7.7:", UPSTREAM_REVIEWED_M7_7_STATUS)
print("Selected candidate:", SELECTED_CANDIDATE_ID)
print("Threshold candidates:", THRESHOLD_CANDIDATES)
print("\nM7.8 UPSTREAM HANDOFF GATE: PASS")
```

    Reviewed M7.7: PASS
    Selected candidate: RF-REF-100-GINI-SQRT-UNPRUNED-CW
    Threshold candidates: [0.2, 0.3, 0.4, 0.5, 0.6]
    
    M7.8 UPSTREAM HANDOFF GATE: PASS


## 5. Locate selected RF risk score

M7.7 JSON được persist trước human/AI review nên selection state trong JSON có thể vẫn OPEN.

M7.8 dùng reviewed handoff để chỉ định selected RF candidate, sau đó xác minh candidate đó và artifact fingerprint tồn tại trong immutable M7.7 registry.


```python
matches = [
    r
    for r in m7_07_result["validation_results"]
    if r["candidate_id"] == SELECTED_CANDIDATE_ID
]
assert len(matches) == 1
selected_validation_record = matches[0]

assert selected_validation_record["model_family"] == SELECTED_MODEL_FAMILY
assert selected_validation_record["training_window"] == SELECTED_TRAINING_WINDOW
assert (
    selected_validation_record["imbalance_strategy"]
    == SELECTED_IMBALANCE_STRATEGY
)
assert selected_validation_record["random_state"] == SELECTED_RANDOM_STATE

RISK_SCORE_PATH = (
    PROJECT_ROOT / selected_validation_record["risk_score_artifact"]
)
DEFAULT_PREDICTION_PATH = (
    PROJECT_ROOT / selected_validation_record["prediction_artifact"]
)

assert RISK_SCORE_PATH.exists()
assert DEFAULT_PREDICTION_PATH.exists()

fp_matches = [
    r
    for r in m7_07_result["artifact_fingerprints"]
    if r["candidate_id"] == SELECTED_CANDIDATE_ID
]
assert len(fp_matches) == 1
selected_fingerprint = fp_matches[0]

assert (
    sha256_file(RISK_SCORE_PATH)
    == selected_fingerprint["risk_score_sha256"]
)
assert (
    sha256_file(DEFAULT_PREDICTION_PATH)
    == selected_fingerprint["prediction_sha256"]
)

print("Risk score:", RISK_SCORE_PATH)
print("Default y_pred:", DEFAULT_PREDICTION_PATH)
print("\nM7.8 SELECTED RISK-SCORE IDENTITY GATE: PASS")
```

    Risk score: /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/processed/m7_07_candidate_selection_external_validation/m7_07__RF-REF-100-GINI-SQRT-UNPRUNED-CW__validation_risk_score.npy
    Default y_pred: /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/processed/m7_07_candidate_selection_external_validation/m7_07__RF-REF-100-GINI-SQRT-UNPRUNED-CW__validation_y_pred.npy
    
    M7.8 SELECTED RISK-SCORE IDENTITY GATE: PASS


## 6. Load target, risk score và default prediction


```python
load_start = time.perf_counter()

y_validation = np.load(Y_VALIDATION_PATH, allow_pickle=False)
row_validation = np.load(ROW_VALIDATION_PATH, allow_pickle=False)
risk_score = np.load(RISK_SCORE_PATH, allow_pickle=False)
default_y_pred = np.load(DEFAULT_PREDICTION_PATH, allow_pickle=False)

print("Load seconds:", round(time.perf_counter() - load_start, 3))
print("Rows:", len(y_validation))
print("Fraud:", int(y_validation.sum()))
print("Risk score dtype:", risk_score.dtype)
print("\nM7.8 ARTIFACT LOAD: COMPLETE")
```

    Load seconds: 0.009
    Rows: 712458
    Fraud: 1052
    Risk score dtype: float32
    
    M7.8 ARTIFACT LOAD: COMPLETE


## 7. Risk-score integrity


```python
assert y_validation.shape == (EXPECTED_VALIDATION_ROWS,)
assert row_validation.shape == (EXPECTED_VALIDATION_ROWS,)
assert risk_score.shape == (EXPECTED_VALIDATION_ROWS,)
assert default_y_pred.shape == (EXPECTED_VALIDATION_ROWS,)

assert y_validation.dtype == np.int8
assert default_y_pred.dtype == np.int8
assert int(y_validation.sum()) == EXPECTED_VALIDATION_FRAUD

assert np.isfinite(risk_score).all()
assert np.all(risk_score >= 0.0)
assert np.all(risk_score <= 1.0)

assert set(np.unique(default_y_pred).tolist()).issubset({0, 1})
assert len(np.unique(row_validation)) == len(row_validation)

print("Risk score min/max:", float(risk_score.min()), "/", float(risk_score.max()))
print("\nM7.8 RISK-SCORE INTEGRITY GATE: PASS")
```

    Risk score min/max: 0.0 / 1.0
    
    M7.8 RISK-SCORE INTEGRITY GATE: PASS


## 8. Resolve comparator bằng compatibility, không dùng performance

M7.8 cần tái tạo đúng M7.7 default prediction ở boundary 0.50.

Hai comparator mechanics được kiểm tra:

```text
risk_score > 0.50
risk_score >= 0.50
```

Comparator được chọn chỉ dựa trên exact equality với persisted M7.7 `y_pred`, không dùng target label hoặc metric.

Sau đó comparator đó được giữ cố định cho mọi threshold.


```python
pred_gt = (risk_score > DEFAULT_THRESHOLD).astype(np.int8)
pred_ge = (risk_score >= DEFAULT_THRESHOLD).astype(np.int8)

gt_match = np.array_equal(pred_gt, default_y_pred)
ge_match = np.array_equal(pred_ge, default_y_pred)

if gt_match:
    THRESHOLD_COMPARATOR = ">"
    apply_threshold = lambda score, thr: (score > thr).astype(np.int8)
elif ge_match:
    THRESHOLD_COMPARATOR = ">="
    apply_threshold = lambda score, thr: (score >= thr).astype(np.int8)
else:
    raise RuntimeError(
        "Không có comparator 0.50 nào tái tạo đúng M7.7 default prediction."
    )

default_reconstructed = apply_threshold(risk_score, DEFAULT_THRESHOLD)
default_mismatch_count = int(
    np.count_nonzero(default_reconstructed != default_y_pred)
)
assert default_mismatch_count == 0

print("Resolved comparator:", THRESHOLD_COMPARATOR)
print("Default mismatch count:", default_mismatch_count)
print("\nM7.8 DEFAULT-BOUNDARY COMPATIBILITY GATE: PASS")
```

    Resolved comparator: >
    Default mismatch count: 0
    
    M7.8 DEFAULT-BOUNDARY COMPATIBILITY GATE: PASS


## 9. Freeze threshold registry


```python
THRESHOLD_REGISTRY = [
    {"threshold_id": "THR-020", "threshold": 0.20, "role": "LOWER"},
    {"threshold_id": "THR-030", "threshold": 0.30, "role": "LOWER"},
    {"threshold_id": "THR-040", "threshold": 0.40, "role": "LOWER_NEAR_DEFAULT"},
    {"threshold_id": "THR-050", "threshold": 0.50, "role": "DEFAULT_REFERENCE"},
    {"threshold_id": "THR-060", "threshold": 0.60, "role": "HIGHER"},
]

assert [r["threshold"] for r in THRESHOLD_REGISTRY] == THRESHOLD_CANDIDATES
assert len({r["threshold_id"] for r in THRESHOLD_REGISTRY}) == 5
assert len({r["threshold"] for r in THRESHOLD_REGISTRY}) == 5

for record in THRESHOLD_REGISTRY:
    print(
        record["threshold_id"],
        "→",
        record["threshold"],
        "|",
        record["role"],
    )

print("\nM7.8 THRESHOLD REGISTRY GATE: PASS")
```

    THR-020 → 0.2 | LOWER
    THR-030 → 0.3 | LOWER
    THR-040 → 0.4 | LOWER_NEAR_DEFAULT
    THR-050 → 0.5 | DEFAULT_REFERENCE
    THR-060 → 0.6 | HIGHER
    
    M7.8 THRESHOLD REGISTRY GATE: PASS


## 10. Canonical metric function


```python
def metric_bundle(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    alerts = int(tp + fp)

    return {
        "F1_fraud": float(
            f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "Recall_fraud": float(
            recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "Precision_fraud": float(
            precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "Accuracy_reference": float(accuracy_score(y_true, y_pred)),
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
        "predicted_positive_count": alerts,
        "predicted_positive_rate": float(alerts / len(y_true)),
    }

print("M7.8 THRESHOLD METRIC FUNCTION: DEFINED")
```

    M7.8 THRESHOLD METRIC FUNCTION: DEFINED


## 11. Evaluate exact predeclared threshold set

Không refit model.

Prediction cho mỗi threshold chỉ được tạo từ selected M7.7 risk score.


```python
threshold_results = []
threshold_prediction_arrays = {}

start = time.perf_counter()

for candidate in THRESHOLD_REGISTRY:
    threshold = float(candidate["threshold"])
    y_pred = apply_threshold(risk_score, threshold)

    record = {
        "threshold_id": candidate["threshold_id"],
        "threshold": threshold,
        "role": candidate["role"],
        "comparator": THRESHOLD_COMPARATOR,
        **metric_bundle(y_validation, y_pred),
    }

    threshold_results.append(record)
    threshold_prediction_arrays[candidate["threshold_id"]] = y_pred

print("Evaluation seconds:", round(time.perf_counter() - start, 4))

for record in threshold_results:
    print("\n", record["threshold_id"], "| threshold:", record["threshold"])
    print(
        "F1 / Recall / Precision:",
        round(record["F1_fraud"], 6),
        "/",
        round(record["Recall_fraud"], 6),
        "/",
        round(record["Precision_fraud"], 6),
    )
    print(
        "TP / FP / FN / TN:",
        record["TP"],
        "/",
        record["FP"],
        "/",
        record["FN"],
        "/",
        record["TN"],
    )
    print(
        "Alerts:",
        record["predicted_positive_count"],
        "| rate:",
        record["predicted_positive_rate"],
    )

assert len(threshold_results) == 5
print("\nM7.8 THRESHOLD EVALUATION GATE: PASS")
```

    Evaluation seconds: 0.5378
    
     THR-020 | threshold: 0.2
    F1 / Recall / Precision: 0.362667 / 0.51711 / 0.279261
    TP / FP / FN / TN: 544 / 1404 / 508 / 710002
    Alerts: 1948 | rate: 0.002734196261393654
    
     THR-030 | threshold: 0.3
    F1 / Recall / Precision: 0.38375 / 0.502852 / 0.310264
    TP / FP / FN / TN: 529 / 1176 / 523 / 710230
    Alerts: 1705 | rate: 0.0023931235244744253
    
     THR-040 | threshold: 0.4
    F1 / Recall / Precision: 0.393856 / 0.475285 / 0.336247
    TP / FP / FN / TN: 500 / 987 / 552 / 710419
    Alerts: 1487 | rate: 0.002087140575304088
    
     THR-050 | threshold: 0.5
    F1 / Recall / Precision: 0.409111 / 0.443916 / 0.379366
    TP / FP / FN / TN: 467 / 764 / 585 / 710642
    Alerts: 1231 | rate: 0.0017278211487554353
    
     THR-060 | threshold: 0.6
    F1 / Recall / Precision: 0.403751 / 0.388783 / 0.419918
    TP / FP / FN / TN: 409 / 565 / 643 / 710841
    Alerts: 974 | rate: 0.001367098130696827
    
    M7.8 THRESHOLD EVALUATION GATE: PASS


## 12. Default 0.50 phải tái tạo M7.7 metrics


```python
default_result = next(
    r for r in threshold_results if r["threshold"] == DEFAULT_THRESHOLD
)

fields = [
    "F1_fraud",
    "Recall_fraud",
    "Precision_fraud",
    "TP",
    "FP",
    "FN",
    "TN",
    "predicted_positive_count",
    "predicted_positive_rate",
]

for field in fields:
    observed = default_result[field]
    expected = selected_validation_record[field]

    if isinstance(expected, float):
        assert np.isclose(observed, expected, rtol=0.0, atol=1e-12)
    else:
        assert observed == expected

print("M7.7 F1:", selected_validation_record["F1_fraud"])
print("M7.8 threshold=0.50 F1:", default_result["F1_fraud"])
print("\nM7.8 DEFAULT-METRIC REPRODUCTION GATE: PASS")
```

    M7.7 F1: 0.4091108190976785
    M7.8 threshold=0.50 F1: 0.4091108190976785
    
    M7.8 DEFAULT-METRIC REPRODUCTION GATE: PASS


## 13. Delta so với default 0.50


```python
threshold_comparisons = []

for record in threshold_results:
    comparison = {
        "threshold_id": record["threshold_id"],
        "threshold": record["threshold"],
        "delta_F1_vs_default": float(
            record["F1_fraud"] - default_result["F1_fraud"]
        ),
        "delta_Recall_vs_default": float(
            record["Recall_fraud"] - default_result["Recall_fraud"]
        ),
        "delta_Precision_vs_default": float(
            record["Precision_fraud"] - default_result["Precision_fraud"]
        ),
        "delta_TP_vs_default": int(record["TP"] - default_result["TP"]),
        "delta_FP_vs_default": int(record["FP"] - default_result["FP"]),
        "delta_FN_vs_default": int(record["FN"] - default_result["FN"]),
        "delta_alerts_vs_default": int(
            record["predicted_positive_count"]
            - default_result["predicted_positive_count"]
        ),
    }
    threshold_comparisons.append(comparison)

for record in threshold_comparisons:
    print("\n", record["threshold_id"], "|", record["threshold"])
    print(
        "Δ F1 / Recall / Precision:",
        round(record["delta_F1_vs_default"], 6),
        "/",
        round(record["delta_Recall_vs_default"], 6),
        "/",
        round(record["delta_Precision_vs_default"], 6),
    )
    print(
        "Δ TP / FP / FN / alerts:",
        record["delta_TP_vs_default"],
        "/",
        record["delta_FP_vs_default"],
        "/",
        record["delta_FN_vs_default"],
        "/",
        record["delta_alerts_vs_default"],
    )

print("\nM7.8 DEFAULT-COMPARISON GATE: PASS")
```

    
     THR-020 | 0.2
    Δ F1 / Recall / Precision: -0.046444 / 0.073194 / -0.100106
    Δ TP / FP / FN / alerts: 77 / 640 / -77 / 717
    
     THR-030 | 0.3
    Δ F1 / Recall / Precision: -0.02536 / 0.058935 / -0.069102
    Δ TP / FP / FN / alerts: 62 / 412 / -62 / 474
    
     THR-040 | 0.4
    Δ F1 / Recall / Precision: -0.015255 / 0.031369 / -0.043119
    Δ TP / FP / FN / alerts: 33 / 223 / -33 / 256
    
     THR-050 | 0.5
    Δ F1 / Recall / Precision: 0.0 / 0.0 / 0.0
    Δ TP / FP / FN / alerts: 0 / 0 / 0 / 0
    
     THR-060 | 0.6
    Δ F1 / Recall / Precision: -0.00536 / -0.055133 / 0.040551
    Δ TP / FP / FN / alerts: -58 / -199 / 58 / -257
    
    M7.8 DEFAULT-COMPARISON GATE: PASS


## 14. Descriptive roles — không auto-select


```python
descriptive_roles = {
    "highest_F1": max(
        threshold_results,
        key=lambda r: r["F1_fraud"],
    )["threshold_id"],
    "highest_Recall": max(
        threshold_results,
        key=lambda r: r["Recall_fraud"],
    )["threshold_id"],
    "highest_Precision": max(
        threshold_results,
        key=lambda r: r["Precision_fraud"],
    )["threshold_id"],
    "lowest_FN": min(
        threshold_results,
        key=lambda r: r["FN"],
    )["threshold_id"],
    "lowest_FP": min(
        threshold_results,
        key=lambda r: r["FP"],
    )["threshold_id"],
    "lowest_alerts": min(
        threshold_results,
        key=lambda r: r["predicted_positive_count"],
    )["threshold_id"],
}

threshold_by_id = {r["threshold_id"]: r for r in threshold_results}

for role, threshold_id in descriptive_roles.items():
    print(
        role,
        "→",
        threshold_id,
        "| threshold:",
        threshold_by_id[threshold_id]["threshold"],
    )

print("\nM7.8 DESCRIPTIVE ROLE GATE: PASS")
```

    highest_F1 → THR-050 | threshold: 0.5
    highest_Recall → THR-020 | threshold: 0.2
    highest_Precision → THR-060 | threshold: 0.6
    lowest_FN → THR-020 | threshold: 0.2
    lowest_FP → THR-060 | threshold: 0.6
    lowest_alerts → THR-060 | threshold: 0.6
    
    M7.8 DESCRIPTIVE ROLE GATE: PASS


Các role trên chỉ là mô tả.

Final threshold vẫn cần runtime review theo toàn evidence bundle.

## 15. Risk-score context sau khi threshold set đã freeze


```python
fraud_scores = risk_score[y_validation == 1]
nonfraud_scores = risk_score[y_validation == 0]
quantiles = [0.10, 0.25, 0.50, 0.75, 0.90]

score_context = {
    "fraud": {
        "mean": float(fraud_scores.mean()),
        "quantiles": {
            str(q): float(np.quantile(fraud_scores, q))
            for q in quantiles
        },
    },
    "nonfraud": {
        "mean": float(nonfraud_scores.mean()),
        "quantiles": {
            str(q): float(np.quantile(nonfraud_scores, q))
            for q in quantiles
        },
    },
}

print("Fraud score mean:", score_context["fraud"]["mean"])
print("Fraud quantiles:", score_context["fraud"]["quantiles"])
print("Non-fraud score mean:", score_context["nonfraud"]["mean"])
print("Non-fraud quantiles:", score_context["nonfraud"]["quantiles"])
print("\nM7.8 SCORE-CONTEXT GATE: PASS")
```

    Fraud score mean: 0.3898194134235382
    Fraud quantiles: {'0.1': 0.0, '0.25': 0.009999999776482582, '0.5': 0.3199999928474426, '0.75': 0.7900000214576721, '0.9': 0.9100000262260437}
    Non-fraud score mean: 0.0013270340859889984
    Non-fraud quantiles: {'0.1': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.9': 0.0}
    
    M7.8 SCORE-CONTEXT GATE: PASS


Không thêm threshold mới dựa trên score-context vừa quan sát.

## 16. No-auto-selection boundary


```python
UPSTREAM_PIPELINE_STATUS = "FROZEN"
FINAL_THRESHOLD = "OPEN — REQUIRES M7.8 RUNTIME REVIEW"
THRESHOLD_SELECTION_DECISION = "OPEN — REQUIRES M7.8 RUNTIME REVIEW"

assert UPSTREAM_PIPELINE_STATUS == "FROZEN"
assert FINAL_THRESHOLD.startswith("OPEN")

print("Upstream pipeline:", UPSTREAM_PIPELINE_STATUS)
print("Final threshold:", FINAL_THRESHOLD)
print("\nM7.8 NO-AUTO-SELECTION GATE: PASS")
```

    Upstream pipeline: FROZEN
    Final threshold: OPEN — REQUIRES M7.8 RUNTIME REVIEW
    
    M7.8 NO-AUTO-SELECTION GATE: PASS


## 17. Persist M7.8 evidence


```python
np.savez_compressed(
    PREDICTIONS_PATH,
    **threshold_prediction_arrays,
)

source_fingerprints = {
    "m4_manifest_sha256": sha256_file(M4_MANIFEST_PATH),
    "y_validation_sha256": sha256_file(Y_VALIDATION_PATH),
    "row_id_validation_sha256": sha256_file(ROW_VALIDATION_PATH),
    "m7_07_result_sha256": sha256_file(M7_07_RESULT_PATH),
    "m7_07_manifest_sha256": sha256_file(M7_07_MANIFEST_PATH),
    "selected_m7_07_risk_score_sha256": sha256_file(RISK_SCORE_PATH),
    "selected_m7_07_prediction_sha256": sha256_file(
        DEFAULT_PREDICTION_PATH
    ),
    "reviewed_m7_07_notebook_sha256": (
        UPSTREAM_REVIEWED_M7_7_NOTEBOOK_SHA256
    ),
}

result_payload = {
    "analysis_version": M7_08_ANALYSIS_VERSION,
    "upstream_reviewed_handoff": {
        "m7_07_status": UPSTREAM_REVIEWED_M7_7_STATUS,
        "selected_candidate_id": SELECTED_CANDIDATE_ID,
        "model_family": SELECTED_MODEL_FAMILY,
        "training_window": SELECTED_TRAINING_WINDOW,
        "imbalance_strategy": SELECTED_IMBALANCE_STRATEGY,
        "random_state": SELECTED_RANDOM_STATE,
        "probability_interface": SELECTED_PROBABILITY_INTERFACE,
        "reviewed_notebook_sha256": (
            UPSTREAM_REVIEWED_M7_7_NOTEBOOK_SHA256
        ),
    },
    "source_fingerprints": source_fingerprints,
    "threshold_candidate_policy": {
        "predeclared": True,
        "adaptive_expansion": False,
        "thresholds": THRESHOLD_CANDIDATES,
        "comparator": THRESHOLD_COMPARATOR,
        "default_threshold": DEFAULT_THRESHOLD,
    },
    "threshold_registry": THRESHOLD_REGISTRY,
    "threshold_results": threshold_results,
    "default_comparisons": threshold_comparisons,
    "descriptive_roles": descriptive_roles,
    "score_context": score_context,
    "default_boundary_mismatch_count": default_mismatch_count,
    "prediction_artifact": str(
        PREDICTIONS_PATH.relative_to(PROJECT_ROOT)
    ),
    "selection_state": {
        "upstream_pipeline": "FROZEN",
        "final_threshold": FINAL_THRESHOLD,
        "threshold_selection_decision": THRESHOLD_SELECTION_DECISION,
        "runtime_review_required": True,
    },
    "model_refit_performed": False,
    "candidate_expansion_performed": False,
    "calibration_performed": False,
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
    "analysis_version": M7_08_ANALYSIS_VERSION,
    "selected_candidate_id": SELECTED_CANDIDATE_ID,
    "threshold_candidate_count": 5,
    "default_threshold": DEFAULT_THRESHOLD,
    "comparator": THRESHOLD_COMPARATOR,
    "validation_rows": EXPECTED_VALIDATION_ROWS,
    "validation_fraud": EXPECTED_VALIDATION_FRAUD,
    "upstream_pipeline": "FROZEN",
    "final_threshold": FINAL_THRESHOLD,
    "model_refit_performed": False,
    "candidate_expansion_performed": False,
    "calibration_performed": False,
    "final_test_accessed": False,
    "decision": "OPEN — REQUIRES M7.8 RUNTIME REVIEW",
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
assert PREDICTIONS_PATH.exists()

print("Result:", RESULT_PATH)
print("Manifest:", MANIFEST_PATH)
print("Predictions:", PREDICTIONS_PATH)
print("\nM7.8 PERSISTENCE GATE: PASS")
```

    Result: /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/processed/m7_08_numerical_threshold_selection/m7_08_threshold_registry.json
    Manifest: /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/processed/m7_08_numerical_threshold_selection/m7_08_threshold_manifest.json
    Predictions: /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/processed/m7_08_numerical_threshold_selection/m7_08_threshold_predictions.npz
    
    M7.8 PERSISTENCE GATE: PASS


## 18. Persistence round-trip


```python
with open(RESULT_PATH, "r", encoding="utf-8") as file:
    result_roundtrip = json.load(file)

with open(MANIFEST_PATH, "r", encoding="utf-8") as file:
    manifest_roundtrip = json.load(file)

prediction_roundtrip = np.load(
    PREDICTIONS_PATH,
    allow_pickle=False,
)

assert result_roundtrip["analysis_version"] == M7_08_ANALYSIS_VERSION
assert len(result_roundtrip["threshold_results"]) == 5
assert (
    result_roundtrip["threshold_candidate_policy"]["adaptive_expansion"]
    is False
)
assert result_roundtrip["model_refit_performed"] is False
assert result_roundtrip["calibration_performed"] is False
assert result_roundtrip["final_test_accessed"] is False

assert manifest_roundtrip["threshold_candidate_count"] == 5
assert manifest_roundtrip["upstream_pipeline"] == "FROZEN"
assert manifest_roundtrip["final_test_accessed"] is False

assert set(prediction_roundtrip.files) == {
    "THR-020",
    "THR-030",
    "THR-040",
    "THR-050",
    "THR-060",
}

for threshold_id, expected in threshold_prediction_arrays.items():
    assert np.array_equal(
        prediction_roundtrip[threshold_id],
        expected,
    )

prediction_roundtrip.close()

print("Result SHA256:", sha256_file(RESULT_PATH))
print("Manifest SHA256:", sha256_file(MANIFEST_PATH))
print("Predictions SHA256:", sha256_file(PREDICTIONS_PATH))
print("\nM7.8 ROUND-TRIP GATE: PASS")
```

    Result SHA256: 7182944c08a230cf383649c66f6947e1c91f7855f33c856eae56a3b748b485e0
    Manifest SHA256: 3c83c5ed65db74582d49a5fb5042f4aef14c9d6f524e370a160e7697f8671a24
    Predictions SHA256: bbb4569a6baad5dbb2a3537ef7c264614c8ebeb71815d3bcfa5b4a7cb910dde5
    
    M7.8 ROUND-TRIP GATE: PASS


## 19. Selection / leakage boundary


```python
assert UPSTREAM_PIPELINE_STATUS == "FROZEN"
assert result_payload["threshold_candidate_policy"]["predeclared"] is True
assert (
    result_payload["threshold_candidate_policy"]["adaptive_expansion"]
    is False
)
assert result_payload["model_refit_performed"] is False
assert result_payload["candidate_expansion_performed"] is False
assert result_payload["calibration_performed"] is False
assert result_payload["selection_state"]["final_threshold"].startswith("OPEN")
assert result_payload["final_test_accessed"] is False

print("Upstream pipeline:", result_payload["selection_state"]["upstream_pipeline"])
print(
    "Adaptive threshold expansion:",
    result_payload["threshold_candidate_policy"]["adaptive_expansion"],
)
print("Model refit:", result_payload["model_refit_performed"])
print("Calibration:", result_payload["calibration_performed"])
print("FINAL TEST accessed:", result_payload["final_test_accessed"])
print("\nM7.8 SELECTION-BOUNDARY GATE: PASS")
```

    Upstream pipeline: FROZEN
    Adaptive threshold expansion: False
    Model refit: False
    Calibration: False
    FINAL TEST accessed: False
    
    M7.8 SELECTION-BOUNDARY GATE: PASS


## 20. Overall M7.8 technical gate


```python
m7_08_gates = {
    "G01_SOURCE_LOCATION": True,
    "G02_UPSTREAM_HANDOFF": True,
    "G03_SELECTED_RISK_SCORE_IDENTITY": True,
    "G04_ARTIFACT_LOAD": True,
    "G05_RISK_SCORE_INTEGRITY": True,
    "G06_DEFAULT_BOUNDARY_COMPATIBILITY": True,
    "G07_THRESHOLD_REGISTRY": True,
    "G08_METRIC_FUNCTION": True,
    "G09_THRESHOLD_EVALUATION": True,
    "G10_DEFAULT_METRIC_REPRODUCTION": True,
    "G11_DEFAULT_COMPARISON": True,
    "G12_DESCRIPTIVE_ROLES": True,
    "G13_SCORE_CONTEXT": True,
    "G14_NO_AUTO_SELECTION": True,
    "G15_PERSISTENCE": True,
    "G16_ROUND_TRIP": True,
    "G17_UPSTREAM_PIPELINE_FROZEN": True,
    "G18_NO_MODEL_REFIT": True,
    "G19_NO_ADAPTIVE_THRESHOLD_EXPANSION": True,
    "G20_FINAL_TEST_PROTECTION": True,
}

for gate_name, gate_value in m7_08_gates.items():
    print(
        gate_name,
        "→",
        "PASS" if gate_value else "FAIL",
    )

assert len(m7_08_gates) == 20
assert all(m7_08_gates.values())

print("\nM7.8 OVERALL TECHNICAL GATE: PASS")
```

    G01_SOURCE_LOCATION → PASS
    G02_UPSTREAM_HANDOFF → PASS
    G03_SELECTED_RISK_SCORE_IDENTITY → PASS
    G04_ARTIFACT_LOAD → PASS
    G05_RISK_SCORE_INTEGRITY → PASS
    G06_DEFAULT_BOUNDARY_COMPATIBILITY → PASS
    G07_THRESHOLD_REGISTRY → PASS
    G08_METRIC_FUNCTION → PASS
    G09_THRESHOLD_EVALUATION → PASS
    G10_DEFAULT_METRIC_REPRODUCTION → PASS
    G11_DEFAULT_COMPARISON → PASS
    G12_DESCRIPTIVE_ROLES → PASS
    G13_SCORE_CONTEXT → PASS
    G14_NO_AUTO_SELECTION → PASS
    G15_PERSISTENCE → PASS
    G16_ROUND_TRIP → PASS
    G17_UPSTREAM_PIPELINE_FROZEN → PASS
    G18_NO_MODEL_REFIT → PASS
    G19_NO_ADAPTIVE_THRESHOLD_EXPANSION → PASS
    G20_FINAL_TEST_PROTECTION → PASS
    
    M7.8 OVERALL TECHNICAL GATE: PASS


# 21. Kiểm tra runtime và các phát hiện M7.8

## 21.1. Tính toàn vẹn thực thi

```text
Code cells:
19 / 19

Execution count:
1 → 19 liên tục

Runtime errors:
0

stderr:
0
```

Trạng thái:

`VERIFIED`

---

## 21.2. Upstream pipeline và risk-score lineage

M7.8 sử dụng đúng upstream candidate đã khóa ở M7.7:

```text
Training Window:
W_SHORT

Model Family:
Random Forest

Model Config:
RF-REF-100-GINI-SQRT-UNPRUNED-CW

Imbalance Strategy:
CLASS_WEIGHT_BALANCED

Random State:
42

Probability Interface:
predict_proba / positive class = 1
```

Risk score:

```text
Rows:
712,458

Fraud:
1,052

dtype:
float32

range:
0.0 → 1.0
```

Artifact fingerprint khớp M7.7.

Không refit model, không calibration, không mở candidate mới.

Trạng thái:

`VERIFIED — UPSTREAM PIPELINE REMAINS FROZEN`

---

## 21.3. Default-boundary reproduction

Comparator được resolve bằng exact compatibility với persisted M7.7 prediction:

```text
risk_score > threshold
```

Tại threshold `0.50`:

```text
Prediction mismatches:
0

M7.7 F1:
0.4091108190976785

M7.8 threshold=0.50 F1:
0.4091108190976785
```

Do đó threshold experiment có lineage nhất quán với M7.7.

Trạng thái:

`VERIFIED`

---

## 21.4. Threshold candidate set

Exact predeclared set:

```text
0.20
0.30
0.40
0.50
0.60
```

```text
Evaluated:
5 / 5

Adaptive expansion:
NONE
```

Trạng thái:

`VERIFIED`

---

# 22. Phân tích threshold selection

## M7.8-F01 — Threshold 0.20

```text
F1:
0.362667

Recall:
0.517110

Precision:
0.279261

TP:
544

FP:
1,404

FN:
508

Alerts:
1,948
```

So với `0.50`:

```text
Δ F1:
-0.046444

Δ Recall:
+0.073194

Δ Precision:
-0.100106

Δ TP:
+77

Δ FP:
+640

Δ FN:
-77

Δ alerts:
+717
```

Recall tăng nhưng primary F1 và Precision giảm mạnh, FP/alerts tăng lớn.

Decision:

`THR-020 → REJECT`

---

## M7.8-F02 — Threshold 0.30

```text
F1:
0.383750

Recall:
0.502852

Precision:
0.310264

TP:
529

FP:
1,176

FN:
523

Alerts:
1,705
```

So với `0.50`:

```text
Δ F1:
-0.025360

Δ Recall:
+0.058935

Δ Precision:
-0.069102

Δ TP:
+62

Δ FP:
+412

Δ FN:
-62

Δ alerts:
+474
```

Recall/FN tốt hơn nhưng F1/Precision xấu hơn và operational burden tăng đáng kể.

Decision:

`THR-030 → REJECT`

---

## M7.8-F03 — Threshold 0.40

```text
F1:
0.393856

Recall:
0.475285

Precision:
0.336247

TP:
500

FP:
987

FN:
552

Alerts:
1,487
```

So với `0.50`:

```text
Δ F1:
-0.015255

Δ Recall:
+0.031369

Δ Precision:
-0.043119

Δ TP:
+33

Δ FP:
+223

Δ FN:
-33

Δ alerts:
+256
```

Trade-off nhẹ hơn 0.20/0.30 nhưng vẫn không vượt default theo primary F1.

Decision:

`THR-040 → REJECT`

---

## M7.8-F04 — Threshold 0.50

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

TN:
710,642

Alerts:
1,231

Alert rate:
0.0017278211
```

Trong exact candidate set:

```text
Highest F1:
YES
```

0.50 không tối đa Recall hay Precision riêng lẻ, nhưng có primary F1 cao nhất và trade-off cân bằng nhất theo metric contract hiện tại.

Decision:

`THR-050 → SELECT`

---

## M7.8-F05 — Threshold 0.60

```text
F1:
0.403751

Recall:
0.388783

Precision:
0.419918

TP:
409

FP:
565

FN:
643

Alerts:
974
```

So với `0.50`:

```text
Δ F1:
-0.005360

Δ Recall:
-0.055133

Δ Precision:
+0.040551

Δ TP:
-58

Δ FP:
-199

Δ FN:
+58

Δ alerts:
-257
```

Precision/FP/alerts tốt hơn, nhưng F1 và Recall giảm, FN tăng.

Không có business-loss function đã khóa để override primary F1.

Decision:

`THR-060 → REJECT`

---

## M7.8-F06 — Descriptive threshold roles

```text
Highest F1:
0.50

Highest Recall:
0.20

Lowest FN:
0.20

Highest Precision:
0.60

Lowest FP:
0.60

Lowest alerts:
0.60
```

Threshold grid thể hiện đúng trade-off:

```text
lower threshold
→ Recall ↑ / FN ↓
→ FP / alerts ↑ / Precision ↓

higher threshold
→ Precision ↑ / FP / alerts ↓
→ Recall ↓ / FN ↑
```

Primary F1 đạt cao nhất tại `0.50`.

Trạng thái:

`VERIFIED`

---

## M7.8-F07 — Final threshold selection

Không có tie.

Không dùng arbitrary epsilon.

Không có metric/cost function khác đã khóa để override F1 primary.

Reviewed decision:

```text
KEEP DEFAULT THRESHOLD

FINAL NUMERICAL THRESHOLD:
0.50
```

Trạng thái:

`FROZEN FOR M7.9`

---

## M7.8-F08 — Full development configuration đã freeze

```text
Feature / Preprocessing:
FROZEN

Training Window:
W_SHORT — FROZEN

Model Family:
Random Forest — FROZEN

Model Config:
RF-REF-100-GINI-SQRT-UNPRUNED-CW — FROZEN

Imbalance Strategy:
CLASS_WEIGHT_BALANCED — FROZEN

Random State:
42 — FROZEN

Probability Interface:
predict_proba / positive class = 1 — FROZEN

Numerical Threshold:
0.50 — FROZEN
```

Finding phải carry forward:

`TEMPORAL GENERALIZATION FINDING`

FINAL TEST:

`PROTECTED`

Trạng thái:

`DEVELOPMENT CONFIGURATION COMPLETE`

---

## M7.8-F09 — Persistence và M7.9 readiness

Persisted:

```text
m7_08_threshold_registry.json
m7_08_threshold_manifest.json
m7_08_threshold_predictions.npz
```

Round-trip:

`PASS`

Trạng thái:

`READY FOR M7.9`

# 23. Decision Log M7.8 — sau runtime review

## M7.8-D01 — Upstream candidate

Decision:

`RF-REF-100-GINI-SQRT-UNPRUNED-CW`

Status:

`INHERITED FROM M7.7 — FROZEN`

---

## M7.8-D02 — Threshold source

Decision:

`EXTERNAL VALIDATION RISK SCORE`

Status:

`VERIFIED — LOCKED`

---

## M7.8-D03 — Comparator

Decision:

`risk_score > threshold`

Evidence:

`threshold 0.50 → exact M7.7 prediction reproduction`

Status:

`VERIFIED — LOCKED`

---

## M7.8-D04 — Threshold candidate set

```text
0.20
0.30
0.40
0.50
0.60
```

Status:

`PREDECLARED / EXECUTED / LOCKED`

---

## M7.8-D05 — Candidate dispositions

```text
0.20:
REJECT

0.30:
REJECT

0.40:
REJECT

0.50:
SELECT

0.60:
REJECT
```

Status:

`LOCKED`

---

## M7.8-D06 — Final numerical threshold

Decision:

```text
KEEP DEFAULT

FINAL THRESHOLD:
0.50
```

Key evidence:

```text
Highest F1 in exact candidate set:
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
```

Status:

`FROZEN`

---

## M7.8-D07 — Adaptive threshold expansion

Decision:

`DO NOT EXPAND`

Observed:

`False`

Status:

`LOCKED`

---

## M7.8-D08 — Model refit / calibration

```text
Model refit:
NONE

Calibration:
NONE
```

Status:

`VERIFIED`

---

## M7.8-D09 — Final development configuration

```text
Training Window:
W_SHORT

Model Family:
Random Forest

Model Config:
RF-REF-100-GINI-SQRT-UNPRUNED-CW

Imbalance Strategy:
CLASS_WEIGHT_BALANCED

Random State:
42

Probability Interface:
predict_proba / positive class = 1

Numerical Threshold:
0.50
```

Status:

`FROZEN`

---

## M7.8-D10 — Temporal generalization finding

Decision:

Carry forward:

`TEMPORAL GENERALIZATION FINDING`

Threshold selection không xóa finding này.

Status:

`OPEN LIMITATION / CARRY FORWARD`

---

## M7.8-D11 — FINAL TEST

Observed:

`final_test_accessed = False`

Status:

`PROTECTED`

---

## M7.8-D12 — M7.9 handoff

Proceed to:

`M7.9 — Final Selection Registry, Decision Log và M7 Gate`

Status:

`READY`

# 24. M7.8 Gate

## Technical runtime gates

```text
G01_SOURCE_LOCATION                         → PASS
G02_UPSTREAM_HANDOFF                        → PASS
G03_SELECTED_RISK_SCORE_IDENTITY           → PASS
G04_ARTIFACT_LOAD                           → PASS
G05_RISK_SCORE_INTEGRITY                   → PASS
G06_DEFAULT_BOUNDARY_COMPATIBILITY         → PASS
G07_THRESHOLD_REGISTRY                     → PASS
G08_METRIC_FUNCTION                        → PASS
G09_THRESHOLD_EVALUATION                   → PASS
G10_DEFAULT_METRIC_REPRODUCTION            → PASS
G11_DEFAULT_COMPARISON                     → PASS
G12_DESCRIPTIVE_ROLES                      → PASS
G13_SCORE_CONTEXT                          → PASS
G14_NO_AUTO_SELECTION                      → PASS
G15_PERSISTENCE                            → PASS
G16_ROUND_TRIP                             → PASS
G17_UPSTREAM_PIPELINE_FROZEN               → PASS
G18_NO_MODEL_REFIT                         → PASS
G19_NO_ADAPTIVE_THRESHOLD_EXPANSION        → PASS
G20_FINAL_TEST_PROTECTION                  → PASS
```

Technical gates:

`20 / 20 PASS`

---

## Runtime review gates

```text
R01 Execution integrity                    → PASS
R02 Upstream freeze integrity              → PASS
R03 Risk-score identity                    → PASS
R04 Default-boundary reproduction          → PASS
R05 Threshold-set immutability             → PASS
R06 Threshold 0.20 review                  → PASS
R07 Threshold 0.30 review                  → PASS
R08 Threshold 0.40 review                  → PASS
R09 Threshold 0.50 review                  → PASS
R10 Threshold 0.60 review                  → PASS
R11 Primary-F1 comparison                  → PASS
R12 Recall / Precision trade-off           → PASS
R13 FP / FN / alert review                 → PASS
R14 Default-threshold disposition          → PASS
R15 Final threshold freeze                 → PASS
R16 Persistence / round-trip               → PASS
R17 No refit / calibration                 → PASS
R18 FINAL TEST protection                  → PASS
R19 M7.9 handoff                           → PASS
```

Runtime review gates:

`19 / 19 PASS`

Blocking issue:

`NONE`

---

## Overall M7.8 Gate

Final:

`M7.8 — PASS`

Final numerical threshold:

```text
0.50
```

Threshold decision:

`KEEP DEFAULT 0.50`

Development configuration:

`FROZEN`

FINAL TEST:

`PROTECTED`

Handoff:

`READY FOR M7.9`

# 25. Kết luận M7.8

M7.8 đã hoàn thành numerical threshold selection trên external VALIDATION risk score của RF candidate đã freeze ở M7.7.

Runtime:

```text
19 / 19 code cells executed

5 / 5 threshold candidates evaluated

runtime errors:
0

stderr:
0

Technical gates:
20 / 20 PASS

Runtime review gates:
19 / 19 PASS
```

Reviewed decision:

```text
KEEP DEFAULT THRESHOLD

FINAL NUMERICAL THRESHOLD:
0.50
```

Lý do:

- `0.50` có F1 cao nhất trong exact predeclared candidate set;
- threshold thấp hơn tăng Recall/giảm FN nhưng giảm F1/Precision và tăng FP/alerts;
- threshold `0.60` tăng Precision/giảm FP/alerts nhưng giảm F1/Recall và tăng FN;
- không có tie;
- không dùng arbitrary epsilon;
- không có business-loss function đã khóa để override F1 primary.

Final development configuration:

```text
Feature / Preprocessing:
FROZEN

Training Window:
W_SHORT — FROZEN

Model Family:
RANDOM FOREST — FROZEN

Model Config:
RF-REF-100-GINI-SQRT-UNPRUNED-CW — FROZEN

Imbalance Strategy:
CLASS_WEIGHT_BALANCED — FROZEN

Random State:
42 — FROZEN

Probability Interface:
predict_proba / positive class = 1 — FROZEN

Numerical Threshold:
0.50 — FROZEN
```

Limitation carry forward:

`TEMPORAL GENERALIZATION FINDING`

Final state:

```text
M7.8 — PASS

Development Selection:
COMPLETE

Upstream Pipeline:
FROZEN

Final Threshold:
0.50 — FROZEN

FINAL TEST:
PROTECTED

Blocking Issue:
NONE

READY FOR M7.9
```

Bước tiếp theo:

`M7.9 — Final Selection Registry, Decision Log và M7 Gate`

M7.9 chỉ tổng hợp/audit development selection đã hoàn thành và không được mở FINAL TEST.
