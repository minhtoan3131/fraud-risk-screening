# M8.4 — Final Metric + Confusion Matrix Reconstruction

**Project:** AI Transaction Fraud Risk Screening  
**Milestone:** M8 — Protected Final Evaluation  
**Substep:** M8.4 — Final Metric + Confusion Matrix Reconstruction  
**Work type:** Official FINAL TEST metric reconstruction  
**Upstream:** M8.3 — PASS  
**Decision trước runtime:** `OPEN — REQUIRES RUNTIME REVIEW`

---

## Vai trò của notebook

M8.4 là bước **measurement**, không phải model selection.

Nguồn duy nhất để tính final metric:

```text
persisted y_final_test
+
persisted y_pred_final_test
```

từ official M8.3 inference artifacts.

M8.4 độc lập tái tính:

```text
F1_fraud
Recall_fraud
Precision_fraud
Accuracy_reference

TP
FP
FN
TN

predicted_positive_count
predicted_positive_rate
```

M8.4 **không**:

- gọi estimator để tạo prediction mới;
- refit model;
- refit preprocessing;
- thay feature;
- thay threshold;
- thử threshold khác;
- calibration;
- candidate expansion;
- dùng FINAL TEST metric để quay lại thay frozen subject;
- thực hiện FP/FN pattern analysis chi tiết (thuộc M8.5).

M8.4 output sau review sẽ là nguồn canonical cho final metric profile.



## 1. Metric contract đã khóa

Positive class:

```text
fraud = 1
```

Primary metric:

```text
F1_fraud
```

Mandatory secondary metrics:

```text
Recall_fraud
Precision_fraud
```

Mandatory error diagnostic:

```text
Confusion Matrix
TP / FP / FN / TN
```

Operational diagnostic:

```text
predicted_positive_count
predicted_positive_rate
```

Reference only:

```text
Accuracy
```

Không thêm ROC-AUC / PR-AUC hay metric mới trong M8.4.



```python

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import platform
import shutil
import sys

import numpy as np

try:
    import sklearn
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Thiếu scikit-learn. Hãy cài dependency của project rồi Restart Kernel."
    ) from exc

print("Python:", sys.version)
print("Executable:", sys.executable)
print("Platform:", platform.platform())
print("NumPy:", np.__version__)
print("scikit-learn:", sklearn.__version__)

```

    Python: 3.14.6 (main, Jun 10 2026, 10:03:53) [Clang 21.0.0 (clang-2100.0.123.102)]
    Executable: /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/.venv/bin/python
    Platform: macOS-26.6.2-arm64-arm-64bit-Mach-O
    NumPy: 2.5.3
    scikit-learn: 1.9.1


## 2. Locate official M8.3 artifacts và chuẩn bị M8.4 output root


```python

M8_03_REL = (
    Path("data")
    / "processed"
    / "m8_03_final_test_inference"
)

M8_04_REL = (
    Path("data")
    / "processed"
    / "m8_04_final_metric_analysis"
)

required_m8_03_rel_paths = [
    M8_03_REL / "risk_score_final_test.npy",
    M8_03_REL / "y_pred_final_test.npy",
    M8_03_REL / "y_final_test.npy",
    M8_03_REL / "row_id_final_test.npy",
    M8_03_REL / "timestamp_final_test.npy",
    M8_03_REL / "m8_03_final_inference_registry.json",
    M8_03_REL / "m8_03_inference_manifest.json",
    M8_03_REL / "m8_03_official_inference.lock.json",
]

candidate_roots = [
    Path.cwd(),
    *list(Path.cwd().parents)[:8],
]

PROJECT_ROOT = None

for candidate in candidate_roots:
    candidate = candidate.resolve()

    if all(
        (candidate / rel).exists()
        for rel
        in required_m8_03_rel_paths
    ):
        PROJECT_ROOT = candidate
        break

if PROJECT_ROOT is None:
    missing_by_candidate = {}

    for candidate in candidate_roots:
        candidate = candidate.resolve()

        missing_by_candidate[
            str(candidate)
        ] = [
            str(rel)
            for rel
            in required_m8_03_rel_paths
            if not (
                candidate / rel
            ).exists()
        ]

    raise FileNotFoundError(
        "Không tìm thấy PROJECT_ROOT chứa đầy đủ official M8.3 artifacts.\n"
        + json.dumps(
            missing_by_candidate,
            ensure_ascii=False,
            indent=2,
        )
    )

M8_03_DIR = PROJECT_ROOT / M8_03_REL
OUTPUT_DIR = PROJECT_ROOT / M8_04_REL

RISK_SCORE_PATH = (
    M8_03_DIR
    / "risk_score_final_test.npy"
)

Y_PRED_PATH = (
    M8_03_DIR
    / "y_pred_final_test.npy"
)

Y_FINAL_PATH = (
    M8_03_DIR
    / "y_final_test.npy"
)

ROW_FINAL_PATH = (
    M8_03_DIR
    / "row_id_final_test.npy"
)

TIMESTAMP_FINAL_PATH = (
    M8_03_DIR
    / "timestamp_final_test.npy"
)

M8_03_REGISTRY_PATH = (
    M8_03_DIR
    / "m8_03_final_inference_registry.json"
)

M8_03_MANIFEST_PATH = (
    M8_03_DIR
    / "m8_03_inference_manifest.json"
)

M8_03_LOCK_PATH = (
    M8_03_DIR
    / "m8_03_official_inference.lock.json"
)

print("PROJECT_ROOT:", PROJECT_ROOT)
print("M8.3 source:", M8_03_DIR)
print("M8.4 output:", OUTPUT_DIR)
print("\nM8.4 SOURCE LOCATION GATE: PASS")

```

    PROJECT_ROOT: /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening
    M8.3 source: /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/processed/m8_03_final_test_inference
    M8.4 output: /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/processed/m8_04_final_metric_analysis
    
    M8.4 SOURCE LOCATION GATE: PASS



## 3. Output overwrite / exact-rerun guard

M8.4 metric reconstruction là deterministic, nhưng canonical artifacts không được âm thầm ghi đè.

Mặc định:

```text
ALLOW_EXACT_RERUN = False
```

Nếu cần rerun vì lỗi kỹ thuật, phải bật `ALLOW_EXACT_RERUN = True` và ghi rõ `RERUN_REASON`.

Rerun không được thay:

```text
y_true
y_pred
threshold
metric formulas
metric set
```



```python

ALLOW_EXACT_RERUN = False
RERUN_REASON = None

RESULT_PATH = (
    OUTPUT_DIR
    / "m8_04_final_metric_confusion_analysis.json"
)

MANIFEST_PATH = (
    OUTPUT_DIR
    / "m8_04_analysis_manifest.json"
)

existing_outputs = [
    path
    for path
    in [
        RESULT_PATH,
        MANIFEST_PATH,
    ]
    if path.exists()
]

if existing_outputs:
    if not ALLOW_EXACT_RERUN:
        raise RuntimeError(
            "M8.4 canonical output đã tồn tại. "
            "Dừng để tránh accidental overwrite.\n"
            + "\n".join(
                str(path)
                for path
                in existing_outputs
            )
        )

    if (
        RERUN_REASON is None
        or not str(
            RERUN_REASON
        ).strip()
    ):
        raise RuntimeError(
            "Exact rerun yêu cầu RERUN_REASON không rỗng."
        )

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

print(
    "Existing M8.4 outputs:",
    len(existing_outputs),
)
print(
    "Exact rerun enabled:",
    ALLOW_EXACT_RERUN,
)
print(
    "Rerun reason:",
    RERUN_REASON,
)
print(
    "\nM8.4 OUTPUT GUARD: PASS"
)

```

    Existing M8.4 outputs: 0
    Exact rerun enabled: False
    Rerun reason: None
    
    M8.4 OUTPUT GUARD: PASS


## 4. Frozen constants + reviewed M8.3 release identity


```python

EXPECTED_FINAL_ROWS = 722_955
EXPECTED_FINAL_FRAUD = 1_035

POSITIVE_CLASS = 1
FINAL_THRESHOLD = 0.50
THRESHOLD_COMPARATOR = ">"

M8_04_ANALYSIS_VERSION = (
    "M8.4-final-metric-confusion-reconstruction-v1"
)

UPSTREAM_REVIEWED_M8_03_STATUS = "PASS"

# SHA256 của notebook M8.3 sau khi Findings/Decision đã được chèn.
UPSTREAM_REVIEWED_M8_03_NOTEBOOK_SHA256 = (
    "8351f6982018e5c946b03a552eb409e4"
    "f539c3967cf54c2bc63f415c4d5b7bdc"
)

assert (
    UPSTREAM_REVIEWED_M8_03_STATUS
    ==
    "PASS"
)

assert POSITIVE_CLASS == 1
assert FINAL_THRESHOLD == 0.50
assert THRESHOLD_COMPARATOR == ">"

print(
    "Reviewed M8.3:",
    UPSTREAM_REVIEWED_M8_03_STATUS,
)
print(
    "Positive class:",
    POSITIVE_CLASS,
)
print(
    "Frozen threshold rule:",
    "risk_score > 0.50",
)
print(
    "\nM8.4 FROZEN METRIC CONTRACT GATE: PASS"
)

```

    Reviewed M8.3: PASS
    Positive class: 1
    Frozen threshold rule: risk_score > 0.50
    
    M8.4 FROZEN METRIC CONTRACT GATE: PASS


## 5. Load và audit M8.3 registry / manifest / lock


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
    M8_03_REGISTRY_PATH,
    "r",
    encoding="utf-8",
) as file:
    m8_03_registry = json.load(
        file
    )

with open(
    M8_03_MANIFEST_PATH,
    "r",
    encoding="utf-8",
) as file:
    m8_03_manifest = json.load(
        file
    )

with open(
    M8_03_LOCK_PATH,
    "r",
    encoding="utf-8",
) as file:
    m8_03_lock = json.load(
        file
    )

assert (
    m8_03_registry[
        "analysis_version"
    ]
    ==
    "M8.3-frozen-final-inference-v1"
)

assert (
    m8_03_manifest[
        "analysis_version"
    ]
    ==
    "M8.3-frozen-final-inference-v1"
)

# Persisted registry/manifest được tạo trước AI runtime review.
assert str(
    m8_03_registry[
        "decision"
    ]
).startswith(
    "OPEN"
)

assert str(
    m8_03_manifest[
        "decision"
    ]
).startswith(
    "OPEN"
)

assert (
    m8_03_registry[
        "official_final_inference_executed"
    ]
    is True
)

assert (
    m8_03_registry[
        "final_test_metric_computed"
    ]
    is False
)

assert (
    m8_03_registry[
        "inference_execution"
    ][
        "predict_proba_call_count"
    ]
    ==
    1
)

assert (
    m8_03_lock[
        "status"
    ]
    ==
    "INFERENCE_COMPLETED"
)

assert (
    m8_03_lock[
        "predict_proba_call_count"
    ]
    ==
    1
)

assert (
    m8_03_registry[
        "no_retuning_attestation"
    ]
    is True
)

for key in [
    "model_refit_performed",
    "train_validation_refit_performed",
    "preprocessing_refit_performed",
    "threshold_retuning_performed",
    "calibration_performed",
    "candidate_expansion_performed",
]:
    assert (
        m8_03_registry[
            key
        ]
        is False
    )

assert float(
    m8_03_registry[
        "official_subject"
    ][
        "threshold"
    ]
) == FINAL_THRESHOLD

assert (
    m8_03_registry[
        "official_subject"
    ][
        "threshold_comparator"
    ]
    ==
    THRESHOLD_COMPARATOR
)

assert int(
    m8_03_registry[
        "official_subject"
    ][
        "positive_class"
    ]
) == POSITIVE_CLASS

print(
    "M8.3 persisted decision:",
    m8_03_registry[
        "decision"
    ],
)
print(
    "M8.3 official inference executed:",
    m8_03_registry[
        "official_final_inference_executed"
    ],
)
print(
    "M8.3 predict_proba call count:",
    m8_03_registry[
        "inference_execution"
    ][
        "predict_proba_call_count"
    ],
)
print(
    "M8.3 metric computed before M8.4:",
    m8_03_registry[
        "final_test_metric_computed"
    ],
)
print(
    "\nM8.4 UPSTREAM M8.3 RELEASE GATE: PASS"
)

```

    M8.3 persisted decision: OPEN — REQUIRES AI RUNTIME REVIEW
    M8.3 official inference executed: True
    M8.3 predict_proba call count: 1
    M8.3 metric computed before M8.4: False
    
    M8.4 UPSTREAM M8.3 RELEASE GATE: PASS


## 6. Re-verify official M8.3 artifact fingerprints


```python

m8_03_generated_fp = (
    m8_03_registry[
        "generated_fingerprints"
    ]
)

fingerprint_path_map = {
    "risk_score_final_test_sha256":
        RISK_SCORE_PATH,
    "y_pred_final_test_sha256":
        Y_PRED_PATH,
    "y_final_test_sha256":
        Y_FINAL_PATH,
    "row_id_final_test_sha256":
        ROW_FINAL_PATH,
    "timestamp_final_test_sha256":
        TIMESTAMP_FINAL_PATH,
    "official_inference_lock_sha256":
        M8_03_LOCK_PATH,
}

observed_m8_03_fingerprints = {}

for key, path in (
    fingerprint_path_map.items()
):
    observed = sha256_file(
        path
    )

    expected = (
        m8_03_generated_fp[
            key
        ]
    )

    if observed != expected:
        raise RuntimeError(
            "Official M8.3 artifact fingerprint mismatch: "
            f"{key}\n"
            f"expected={expected}\n"
            f"observed={observed}"
        )

    observed_m8_03_fingerprints[
        key
    ] = observed

assert (
    m8_03_manifest[
        "registry_sha256"
    ]
    ==
    sha256_file(
        M8_03_REGISTRY_PATH
    )
)

print(
    "Verified official M8.3 artifact fingerprints:",
    len(
        observed_m8_03_fingerprints
    ),
)
print(
    "Registry SHA256 verified:",
    True,
)
print(
    "\nM8.4 M8.3 ARTIFACT FINGERPRINT GATE: PASS"
)

```

    Verified official M8.3 artifact fingerprints: 6
    Registry SHA256 verified: True
    
    M8.4 M8.3 ARTIFACT FINGERPRINT GATE: PASS


## 7. Load persisted official arrays và audit row alignment


```python

risk_score_final_test = np.load(
    RISK_SCORE_PATH,
    allow_pickle=False,
)

y_pred_final_test = np.load(
    Y_PRED_PATH,
    allow_pickle=False,
)

y_final_test = np.load(
    Y_FINAL_PATH,
    allow_pickle=False,
)

row_id_final_test = np.load(
    ROW_FINAL_PATH,
    allow_pickle=False,
)

timestamp_final_test = np.load(
    TIMESTAMP_FINAL_PATH,
    allow_pickle=False,
)

expected_shape = (
    EXPECTED_FINAL_ROWS,
)

assert (
    risk_score_final_test.shape
    ==
    expected_shape
)

assert (
    y_pred_final_test.shape
    ==
    expected_shape
)

assert (
    y_final_test.shape
    ==
    expected_shape
)

assert (
    row_id_final_test.shape
    ==
    expected_shape
)

assert (
    timestamp_final_test.shape
    ==
    expected_shape
)

assert (
    len(
        np.unique(
            row_id_final_test
        )
    )
    ==
    EXPECTED_FINAL_ROWS
)

assert np.isfinite(
    risk_score_final_test
).all()

assert np.isin(
    y_final_test,
    [0, 1],
).all()

assert np.isin(
    y_pred_final_test,
    [0, 1],
).all()

assert (
    int(
        y_final_test.sum()
    )
    ==
    EXPECTED_FINAL_FRAUD
)

assert (
    int(
        m8_03_registry[
            "final_test_identity"
        ][
            "rows"
        ]
    )
    ==
    EXPECTED_FINAL_ROWS
)

assert (
    int(
        m8_03_registry[
            "final_test_identity"
        ][
            "fraud"
        ]
    )
    ==
    EXPECTED_FINAL_FRAUD
)

print(
    "Aligned rows:",
    len(
        y_final_test
    ),
)
print(
    "Fraud support:",
    int(
        y_final_test.sum()
    ),
)
print(
    "Unique row IDs:",
    len(
        np.unique(
            row_id_final_test
        )
    ),
)
print(
    "\nM8.4 ROW ALIGNMENT / TARGET IDENTITY GATE: PASS"
)

```

    Aligned rows: 722955
    Fraud support: 1035
    Unique row IDs: 722955
    
    M8.4 ROW ALIGNMENT / TARGET IDENTITY GATE: PASS



## 8. Re-verify frozen prediction semantics

M8.4 không tạo prediction mới bằng model.

Chỉ kiểm tra persisted `y_pred_final_test` có còn đúng với:

```text
persisted risk_score_final_test
+
risk_score > 0.50
```

hay không.



```python

threshold_reconstructed = (
    risk_score_final_test
    >
    FINAL_THRESHOLD
).astype(
    np.int8,
    copy=False,
)

threshold_mismatch_count = int(
    np.count_nonzero(
        threshold_reconstructed
        !=
        y_pred_final_test
    )
)

assert (
    threshold_mismatch_count
    ==
    0
)

print(
    "Threshold:",
    FINAL_THRESHOLD,
)
print(
    "Comparator:",
    THRESHOLD_COMPARATOR,
)
print(
    "Prediction mismatch vs persisted M8.3:",
    threshold_mismatch_count,
)
print(
    "\nM8.4 FROZEN PREDICTION SEMANTICS GATE: PASS"
)

```

    Threshold: 0.5
    Comparator: >
    Prediction mismatch vs persisted M8.3: 0
    
    M8.4 FROZEN PREDICTION SEMANTICS GATE: PASS



# 9. Independent Confusion Matrix reconstruction

Canonical count definitions:

```text
TP = y_true=1 AND y_pred=1
FP = y_true=0 AND y_pred=1
FN = y_true=1 AND y_pred=0
TN = y_true=0 AND y_pred=0
```

Positive class luôn là fraud = 1.



```python

y_true = y_final_test
y_pred = y_pred_final_test

tp = int(
    np.count_nonzero(
        (y_true == 1)
        &
        (y_pred == 1)
    )
)

fp = int(
    np.count_nonzero(
        (y_true == 0)
        &
        (y_pred == 1)
    )
)

fn = int(
    np.count_nonzero(
        (y_true == 1)
        &
        (y_pred == 0)
    )
)

tn = int(
    np.count_nonzero(
        (y_true == 0)
        &
        (y_pred == 0)
    )
)

total = (
    tp
    + fp
    + fn
    + tn
)

actual_positive_count = (
    tp
    + fn
)

actual_negative_count = (
    tn
    + fp
)

predicted_positive_count = (
    tp
    + fp
)

predicted_negative_count = (
    tn
    + fn
)

assert total == EXPECTED_FINAL_ROWS
assert (
    actual_positive_count
    ==
    EXPECTED_FINAL_FRAUD
)
assert (
    actual_negative_count
    ==
    EXPECTED_FINAL_ROWS
    -
    EXPECTED_FINAL_FRAUD
)

assert (
    predicted_positive_count
    ==
    int(
        y_pred.sum()
    )
)

assert (
    predicted_negative_count
    ==
    EXPECTED_FINAL_ROWS
    -
    predicted_positive_count
)

print("TP:", tp)
print("FP:", fp)
print("FN:", fn)
print("TN:", tn)
print("Total:", total)
print(
    "Predicted positive count:",
    predicted_positive_count,
)
print(
    "\nM8.4 CONFUSION ARITHMETIC GATE: PASS"
)

```

    TP: 393
    FP: 782
    FN: 642
    TN: 721138
    Total: 722955
    Predicted positive count: 1175
    
    M8.4 CONFUSION ARITHMETIC GATE: PASS



## 10. Independent metric reconstruction

Công thức canonical:

```text
Recall_fraud
= TP / (TP + FN)

Precision_fraud
= TP / (TP + FP)

F1_fraud
= 2TP / (2TP + FP + FN)

Accuracy_reference
= (TP + TN) / N

predicted_positive_rate
= (TP + FP) / N
```

Nếu denominator của Precision hoặc F1 bằng 0, metric được định nghĩa là `0.0`, tương thích `zero_division=0` trong cross-check.



```python

def safe_ratio(
    numerator,
    denominator,
):
    if denominator == 0:
        return 0.0

    return float(
        numerator
        /
        denominator
    )


recall_fraud = safe_ratio(
    tp,
    tp + fn,
)

precision_fraud = safe_ratio(
    tp,
    tp + fp,
)

f1_fraud = safe_ratio(
    2 * tp,
    (2 * tp) + fp + fn,
)

accuracy_reference = safe_ratio(
    tp + tn,
    total,
)

predicted_positive_rate = safe_ratio(
    predicted_positive_count,
    total,
)

assert (
    0.0
    <= recall_fraud
    <= 1.0
)

assert (
    0.0
    <= precision_fraud
    <= 1.0
)

assert (
    0.0
    <= f1_fraud
    <= 1.0
)

assert (
    0.0
    <= accuracy_reference
    <= 1.0
)

assert (
    0.0
    <= predicted_positive_rate
    <= 1.0
)

print(
    "F1_fraud:",
    repr(
        f1_fraud
    ),
)
print(
    "Recall_fraud:",
    repr(
        recall_fraud
    ),
)
print(
    "Precision_fraud:",
    repr(
        precision_fraud
    ),
)
print(
    "Accuracy_reference:",
    repr(
        accuracy_reference
    ),
)
print(
    "Predicted positive count:",
    predicted_positive_count,
)
print(
    "Predicted positive rate:",
    repr(
        predicted_positive_rate
    ),
)
print(
    "\nM8.4 INDEPENDENT METRIC RECONSTRUCTION GATE: PASS"
)

```

    F1_fraud: 0.3556561085972851
    Recall_fraud: 0.37971014492753624
    Precision_fraud: 0.334468085106383
    Accuracy_reference: 0.9980303061739666
    Predicted positive count: 1175
    Predicted positive rate: 0.0016252740488688785
    
    M8.4 INDEPENDENT METRIC RECONSTRUCTION GATE: PASS


## 11. Independent library cross-check


```python

sk_cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1],
)

# sklearn order với labels=[0,1]:
# [[TN, FP],
#  [FN, TP]]
sk_tn = int(
    sk_cm[0, 0]
)
sk_fp = int(
    sk_cm[0, 1]
)
sk_fn = int(
    sk_cm[1, 0]
)
sk_tp = int(
    sk_cm[1, 1]
)

assert (
    sk_tp,
    sk_fp,
    sk_fn,
    sk_tn,
) == (
    tp,
    fp,
    fn,
    tn,
)

sk_f1 = float(
    f1_score(
        y_true,
        y_pred,
        pos_label=POSITIVE_CLASS,
        zero_division=0,
    )
)

sk_recall = float(
    recall_score(
        y_true,
        y_pred,
        pos_label=POSITIVE_CLASS,
        zero_division=0,
    )
)

sk_precision = float(
    precision_score(
        y_true,
        y_pred,
        pos_label=POSITIVE_CLASS,
        zero_division=0,
    )
)

sk_accuracy = float(
    accuracy_score(
        y_true,
        y_pred,
    )
)

np.testing.assert_allclose(
    [
        f1_fraud,
        recall_fraud,
        precision_fraud,
        accuracy_reference,
    ],
    [
        sk_f1,
        sk_recall,
        sk_precision,
        sk_accuracy,
    ],
    rtol=0.0,
    atol=1e-15,
)

print(
    "Manual confusion counts == sklearn:",
    True,
)
print(
    "Manual metrics == sklearn:",
    True,
)
print(
    "\nM8.4 INDEPENDENT LIBRARY CROSS-CHECK GATE: PASS"
)

```

    Manual confusion counts == sklearn: True
    Manual metrics == sklearn: True
    
    M8.4 INDEPENDENT LIBRARY CROSS-CHECK GATE: PASS



# 12. Final metric profile — OBSERVED FACTS ONLY

Phần này chỉ in các metric facts.

Không diễn giải:

```text
model tốt / xấu
đạt / không đạt production
nên đổi threshold
nên tune lại
```

Những diễn giải temporal/error sâu hơn thuộc M8.5.



```python

metric_profile = {
    "F1_fraud":
        f1_fraud,
    "Recall_fraud":
        recall_fraud,
    "Precision_fraud":
        precision_fraud,
    "Accuracy_reference":
        accuracy_reference,
    "TP":
        tp,
    "FP":
        fp,
    "FN":
        fn,
    "TN":
        tn,
    "predicted_positive_count":
        predicted_positive_count,
    "predicted_positive_rate":
        predicted_positive_rate,
}

print(
    json.dumps(
        metric_profile,
        ensure_ascii=False,
        indent=2,
        sort_keys=False,
    )
)

print(
    "\nM8.4 FINAL METRIC PROFILE MATERIALIZED: PASS"
)

```

    {
      "F1_fraud": 0.3556561085972851,
      "Recall_fraud": 0.37971014492753624,
      "Precision_fraud": 0.334468085106383,
      "Accuracy_reference": 0.9980303061739666,
      "TP": 393,
      "FP": 782,
      "FN": 642,
      "TN": 721138,
      "predicted_positive_count": 1175,
      "predicted_positive_rate": 0.0016252740488688785
    }
    
    M8.4 FINAL METRIC PROFILE MATERIALIZED: PASS


## 13. Persist canonical M8.4 metric / confusion artifacts


```python

runtime_versions = {
    "python":
        sys.version,
    "numpy":
        np.__version__,
    "scikit_learn":
        sklearn.__version__,
    "platform":
        platform.platform(),
}

source_fingerprints = {
    "m8_03_registry_sha256":
        sha256_file(
            M8_03_REGISTRY_PATH
        ),
    "m8_03_manifest_sha256":
        sha256_file(
            M8_03_MANIFEST_PATH
        ),
    "m8_03_lock_sha256":
        sha256_file(
            M8_03_LOCK_PATH
        ),
    "risk_score_final_test_sha256":
        sha256_file(
            RISK_SCORE_PATH
        ),
    "y_pred_final_test_sha256":
        sha256_file(
            Y_PRED_PATH
        ),
    "y_final_test_sha256":
        sha256_file(
            Y_FINAL_PATH
        ),
    "row_id_final_test_sha256":
        sha256_file(
            ROW_FINAL_PATH
        ),
    "timestamp_final_test_sha256":
        sha256_file(
            TIMESTAMP_FINAL_PATH
        ),
    "reviewed_m8_03_notebook_sha256":
        UPSTREAM_REVIEWED_M8_03_NOTEBOOK_SHA256,
}

analysis_payload = {
    "analysis_version":
        M8_04_ANALYSIS_VERSION,
    "work_type":
        "OFFICIAL_FINAL_TEST_METRIC_RECONSTRUCTION",
    "upstream_review": {
        "m8_03_status":
            UPSTREAM_REVIEWED_M8_03_STATUS,
        "reviewed_m8_03_notebook_sha256":
            UPSTREAM_REVIEWED_M8_03_NOTEBOOK_SHA256,
        "official_inference_executed":
            True,
        "predict_proba_call_count":
            1,
    },
    "metric_contract": {
        "positive_class":
            POSITIVE_CLASS,
        "primary_metric":
            "F1_fraud",
        "secondary_metrics": [
            "Recall_fraud",
            "Precision_fraud",
        ],
        "mandatory_diagnostics": [
            "TP",
            "FP",
            "FN",
            "TN",
        ],
        "operational_diagnostics": [
            "predicted_positive_count",
            "predicted_positive_rate",
        ],
        "reference_only": [
            "Accuracy_reference",
        ],
    },
    "final_test_identity": {
        "rows":
            EXPECTED_FINAL_ROWS,
        "fraud":
            EXPECTED_FINAL_FRAUD,
    },
    "frozen_prediction_rule": {
        "risk_score_source":
            "persisted M8.3 risk_score_final_test.npy",
        "threshold":
            FINAL_THRESHOLD,
        "comparator":
            THRESHOLD_COMPARATOR,
        "prediction_mismatch_count":
            threshold_mismatch_count,
    },
    "confusion_matrix": {
        "TP":
            tp,
        "FP":
            fp,
        "FN":
            fn,
        "TN":
            tn,
    },
    "metrics": {
        "F1_fraud":
            f1_fraud,
        "Recall_fraud":
            recall_fraud,
        "Precision_fraud":
            precision_fraud,
        "Accuracy_reference":
            accuracy_reference,
        "predicted_positive_count":
            predicted_positive_count,
        "predicted_positive_rate":
            predicted_positive_rate,
    },
    "cross_check": {
        "manual_vs_sklearn_confusion":
            True,
        "manual_vs_sklearn_metrics":
            True,
        "absolute_tolerance":
            1e-15,
    },
    "guardrails": {
        "model_refit_performed":
            False,
        "train_validation_refit_performed":
            False,
        "preprocessing_refit_performed":
            False,
        "threshold_retuning_performed":
            False,
        "calibration_performed":
            False,
        "candidate_expansion_performed":
            False,
        "new_prediction_generated":
            False,
        "final_test_metric_computed":
            True,
        "no_retroactive_optimization_attestation":
            True,
    },
    "source_fingerprints":
        source_fingerprints,
    "runtime_versions":
        runtime_versions,
    "rerun_policy": {
        "allow_exact_rerun":
            ALLOW_EXACT_RERUN,
        "rerun_reason":
            RERUN_REASON,
    },
    "decision":
        "OPEN — REQUIRES AI RUNTIME REVIEW",
    "m8_5_error_analysis_authorization":
        "NOT YET — REQUIRES REVIEW",
}

with open(
    RESULT_PATH,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        analysis_payload,
        file,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )

result_sha256 = sha256_file(
    RESULT_PATH
)

manifest_payload = {
    "analysis_version":
        M8_04_ANALYSIS_VERSION,
    "source_substep":
        "M8.3 — Frozen Final Inference Run",
    "final_test_rows":
        EXPECTED_FINAL_ROWS,
    "final_test_fraud":
        EXPECTED_FINAL_FRAUD,
    "positive_class":
        POSITIVE_CLASS,
    "threshold":
        FINAL_THRESHOLD,
    "threshold_comparator":
        THRESHOLD_COMPARATOR,
    "metric_result_file":
        RESULT_PATH.name,
    "metric_result_sha256":
        result_sha256,
    "metric_names": [
        "F1_fraud",
        "Recall_fraud",
        "Precision_fraud",
        "Accuracy_reference",
        "TP",
        "FP",
        "FN",
        "TN",
        "predicted_positive_count",
        "predicted_positive_rate",
    ],
    "manual_sklearn_cross_check":
        True,
    "new_prediction_generated":
        False,
    "final_test_metric_computed":
        True,
    "no_retroactive_optimization_attestation":
        True,
    "decision":
        "OPEN — REQUIRES AI RUNTIME REVIEW",
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

print(
    "Result:",
    RESULT_PATH.relative_to(
        PROJECT_ROOT
    ),
)
print(
    "Manifest:",
    MANIFEST_PATH.relative_to(
        PROJECT_ROOT
    ),
)
print(
    "Result SHA256:",
    result_sha256,
)
print(
    "Manifest SHA256:",
    sha256_file(
        MANIFEST_PATH
    ),
)
print(
    "\nM8.4 ARTIFACT PERSISTENCE GATE: PASS"
)

```

    Result: data/processed/m8_04_final_metric_analysis/m8_04_final_metric_confusion_analysis.json
    Manifest: data/processed/m8_04_final_metric_analysis/m8_04_analysis_manifest.json
    Result SHA256: 167ce2ebf1516896b2d393602938382fc4e411b7f8fd566be0403865ab14d038
    Manifest SHA256: 08b4d32f3705850eb2f89cada2e16dc2fda7405810c9a8d1279ff5842b7dd9ab
    
    M8.4 ARTIFACT PERSISTENCE GATE: PASS


## 14. Round-trip canonical metric artifacts


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
    manifest_roundtrip[
        "metric_result_sha256"
    ]
    ==
    sha256_file(
        RESULT_PATH
    )
)

assert (
    result_roundtrip[
        "metrics"
    ][
        "F1_fraud"
    ]
    ==
    f1_fraud
)

assert (
    result_roundtrip[
        "metrics"
    ][
        "Recall_fraud"
    ]
    ==
    recall_fraud
)

assert (
    result_roundtrip[
        "metrics"
    ][
        "Precision_fraud"
    ]
    ==
    precision_fraud
)

assert (
    result_roundtrip[
        "metrics"
    ][
        "Accuracy_reference"
    ]
    ==
    accuracy_reference
)

for name, expected in {
    "TP": tp,
    "FP": fp,
    "FN": fn,
    "TN": tn,
}.items():
    assert (
        result_roundtrip[
            "confusion_matrix"
        ][
            name
        ]
        ==
        expected
    )

assert (
    result_roundtrip[
        "guardrails"
    ][
        "new_prediction_generated"
    ]
    is False
)

assert (
    result_roundtrip[
        "guardrails"
    ][
        "final_test_metric_computed"
    ]
    is True
)

assert (
    result_roundtrip[
        "guardrails"
    ][
        "no_retroactive_optimization_attestation"
    ]
    is True
)

print(
    "Metric result round-trip:",
    "PASS",
)
print(
    "Manifest round-trip:",
    "PASS",
)
print(
    "\nM8.4 ARTIFACT ROUND-TRIP GATE: PASS"
)

```

    Metric result round-trip: PASS
    Manifest round-trip: PASS
    
    M8.4 ARTIFACT ROUND-TRIP GATE: PASS


## 15. M8.4 technical gate


```python

m8_04_gates = {
    "G01_SOURCE_LOCATION":
        True,
    "G02_OUTPUT_GUARD":
        True,
    "G03_FROZEN_METRIC_CONTRACT":
        True,
    "G04_UPSTREAM_M8_3_RELEASE":
        True,
    "G05_M8_3_ARTIFACT_FINGERPRINTS":
        True,
    "G06_ROW_ALIGNMENT_TARGET_IDENTITY":
        True,
    "G07_FROZEN_PREDICTION_SEMANTICS":
        (
            threshold_mismatch_count
            ==
            0
        ),
    "G08_CONFUSION_ARITHMETIC":
        (
            tp
            + fp
            + fn
            + tn
            ==
            EXPECTED_FINAL_ROWS
        ),
    "G09_PRIMARY_F1_RECONSTRUCTION":
        (
            0.0
            <= f1_fraud
            <= 1.0
        ),
    "G10_SECONDARY_METRICS_RECONSTRUCTION":
        (
            0.0
            <= recall_fraud
            <= 1.0
            and
            0.0
            <= precision_fraud
            <= 1.0
        ),
    "G11_ACCURACY_REFERENCE":
        (
            0.0
            <= accuracy_reference
            <= 1.0
        ),
    "G12_OPERATIONAL_DIAGNOSTIC":
        (
            predicted_positive_count
            ==
            tp + fp
            and
            0.0
            <= predicted_positive_rate
            <= 1.0
        ),
    "G13_SKLEARN_CROSS_CHECK":
        True,
    "G14_ARTIFACT_PERSISTENCE":
        (
            RESULT_PATH.exists()
            and
            MANIFEST_PATH.exists()
        ),
    "G15_ARTIFACT_ROUND_TRIP":
        True,
    "G16_NO_NEW_PREDICTION":
        (
            result_roundtrip[
                "guardrails"
            ][
                "new_prediction_generated"
            ]
            is False
        ),
    "G17_NO_RETROACTIVE_OPTIMIZATION":
        (
            result_roundtrip[
                "guardrails"
            ][
                "no_retroactive_optimization_attestation"
            ]
            is True
        ),
}

for gate_name, gate_value in (
    m8_04_gates.items()
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
    m8_04_gates
) == 17

assert all(
    m8_04_gates.values()
)

print(
    "\nM8.4 RUNTIME TECHNICAL GATE: PASS"
)

print(
    "\nDecision:",
    "OPEN — REQUIRES AI RUNTIME REVIEW",
)

print(
    "M8.5 error analysis authorization:",
    "NOT YET — REQUIRES REVIEW",
)

print(
    "\nIMPORTANT:",
    "M8.4 đã materialize official FINAL TEST metric profile, "
    "nhưng chưa thực hiện FP/FN pattern analysis hoặc retroactive optimization."
)

```

    G01_SOURCE_LOCATION → PASS
    G02_OUTPUT_GUARD → PASS
    G03_FROZEN_METRIC_CONTRACT → PASS
    G04_UPSTREAM_M8_3_RELEASE → PASS
    G05_M8_3_ARTIFACT_FINGERPRINTS → PASS
    G06_ROW_ALIGNMENT_TARGET_IDENTITY → PASS
    G07_FROZEN_PREDICTION_SEMANTICS → PASS
    G08_CONFUSION_ARITHMETIC → PASS
    G09_PRIMARY_F1_RECONSTRUCTION → PASS
    G10_SECONDARY_METRICS_RECONSTRUCTION → PASS
    G11_ACCURACY_REFERENCE → PASS
    G12_OPERATIONAL_DIAGNOSTIC → PASS
    G13_SKLEARN_CROSS_CHECK → PASS
    G14_ARTIFACT_PERSISTENCE → PASS
    G15_ARTIFACT_ROUND_TRIP → PASS
    G16_NO_NEW_PREDICTION → PASS
    G17_NO_RETROACTIVE_OPTIMIZATION → PASS
    
    M8.4 RUNTIME TECHNICAL GATE: PASS
    
    Decision: OPEN — REQUIRES AI RUNTIME REVIEW
    M8.5 error analysis authorization: NOT YET — REQUIRES REVIEW
    
    IMPORTANT: M8.4 đã materialize official FINAL TEST metric profile, nhưng chưa thực hiện FP/FN pattern analysis hoặc retroactive optimization.



# 16. Findings / Runtime Review / Decision — sau khi kiểm tra output

## 16.1. Phạm vi review

Review này được thực hiện sau khi notebook M8.4 đã được chạy trên project thật.

Trình tự review:

```text
execution completeness
→ upstream M8.3 release integrity
→ M8.3 artifact fingerprints
→ row alignment / target identity
→ frozen prediction semantics
→ confusion arithmetic
→ manual metric reconstruction
→ sklearn cross-check
→ persistence / fingerprints
→ round-trip
→ no new prediction / no retroactive optimization
→ M8.4 decision
```

M8.4 là bước **measurement**.

M8.4 không thực hiện model selection và không được dùng FINAL TEST metric để quay lại thay frozen subject.

---

## 16.2. Execution completeness

Observed:

```text
Code cells executed:
15 / 15

Execution order:
In[1] → In[15]

Runtime traceback:
NONE OBSERVED

Required technical gate:
REACHED
```

Kết luận:

`PASS`

Không có cell runtime bắt buộc nào bị bỏ qua hoặc dừng trước technical gate cuối.

---

## 16.3. Upstream M8.3 release integrity

M8.4 nhận upstream:

```text
M8.3 reviewed status:
PASS

Official inference:
COMPLETE

predict_proba FINAL TEST call count:
1

Official prediction artifacts:
FROZEN / PERSISTED / VERIFIED
```

Reviewed M8.3 notebook SHA256:

```text
8351f6982018e5c946b03a552eb409e4f539c3967cf54c2bc63f415c4d5b7bdc
```

Observed M8.4 carry-forward hash khớp reviewed M8.3 notebook.

Kết luận:

`PASS`

Không phát hiện upstream release contradiction.

---

## 16.4. Official M8.3 artifact fingerprint re-verification

Trước metric reconstruction, M8.4 re-verify fingerprints của official M8.3 artifacts:

```text
risk_score_final_test.npy
y_pred_final_test.npy
y_final_test.npy
row_id_final_test.npy
timestamp_final_test.npy
m8_03_official_inference.lock.json
m8_03_final_inference_registry.json
m8_03_inference_manifest.json
```

Observed:

`ALL REQUIRED OFFICIAL M8.3 ARTIFACT FINGERPRINTS MATCH`

Registry SHA256 cũng được kiểm tra.

Kết luận:

`PASS`

M8.4 không dùng một bản prediction artifact khác với artifact M8.3 đã khóa.

---

## 16.5. Row alignment / target identity

Observed:

```text
Aligned rows:
722,955

Fraud support:
1,035

Unique row IDs:
722,955
```

Các persisted arrays:

```text
risk_score_final_test
y_pred_final_test
y_final_test
row_id_final_test
timestamp_final_test
```

đều có cùng số dòng:

`722,955`

Target mapping vẫn là:

```text
fraud = 1
non-fraud = 0
```

Kết luận:

`PASS`

Không phát hiện row-alignment hoặc target-identity mismatch.

---

## 16.6. Frozen prediction semantics

Frozen rule:

```text
risk_score > 0.50
```

Observed:

```text
Threshold:
0.50

Comparator:
>

Prediction mismatch vs persisted M8.3:
0
```

M8.4 không tạo prediction mới bằng estimator.

Source review xác nhận:

```text
predict_proba(...):
0

predict(...):
0

fit(...):
0
```

Kết luận:

`PASS`

Persisted `y_pred_final_test` vẫn khớp chính xác frozen threshold rule.

---

# 17. Official FINAL TEST Confusion Matrix

## 17.1. Independent count reconstruction

Observed:

```text
TP:
393

FP:
782

FN:
642

TN:
721,138

Total:
722,955
```

Arithmetic checks:

```text
TP + FN:
1,035
= total fraud support

TN + FP:
721,920
= total non-fraud support

TP + FP:
1,175
= predicted-positive count

TP + FP + FN + TN:
722,955
= FINAL TEST rows
```

Kết luận:

`PASS`

Confusion Matrix được tái dựng độc lập từ persisted:

```text
y_final_test
+
y_pred_final_test
```

---

# 18. Official FINAL TEST metric profile

## 18.1. Primary metric

Observed:

```text
F1_fraud:
0.3556561085972851
```

Status:

`OFFICIAL FINAL TEST METRIC FACT`

---

## 18.2. Secondary metrics

Observed:

```text
Recall_fraud:
0.37971014492753624

Precision_fraud:
0.334468085106383
```

Status:

`OFFICIAL FINAL TEST METRIC FACTS`

---

## 18.3. Accuracy — reference only

Observed:

```text
Accuracy_reference:
0.9980303061739666
```

Accuracy chỉ được giữ với vai trò:

`REFERENCE ONLY`

Không dùng Accuracy đơn độc để đánh giá fraud-detection quality do dataset có class imbalance rất mạnh.

---

## 18.4. Operational diagnostic

Observed:

```text
Predicted-positive count:
1,175

Predicted-positive rate:
0.0016252740488688785
```

Tương đương xấp xỉ:

```text
0.162527%
```

Status:

`OFFICIAL OPERATIONAL DIAGNOSTIC`

---

# 19. Independent library cross-check

Manual reconstruction được cross-check bằng scikit-learn.

Observed:

```text
Manual confusion counts == sklearn:
True

Manual metrics == sklearn:
True
```

Cross-check tolerance:

```text
absolute tolerance:
1e-15
```

Kết luận:

`PASS`

Các metric không phụ thuộc vào một implementation riêng lẻ.

---

# 20. Metric interpretation trong phạm vi M8.4

Các phát biểu dưới đây chỉ diễn giải trực tiếp các metric facts đã quan sát.

## 20.1. Recall

```text
Recall_fraud ≈ 37.97%
```

Trong 1,035 giao dịch fraud của FINAL TEST:

```text
TP:
393

FN:
642
```

Do đó model nhận diện đúng khoảng:

```text
37.97%
```

fraud và bỏ sót khoảng:

```text
62.03%
```

fraud trong protected FINAL TEST.

Đây là descriptive metric fact.

Không được dùng kết quả này để thay threshold hoặc retrain model sau khi FINAL TEST đã được mở.

---

## 20.2. Precision

```text
Precision_fraud ≈ 33.45%
```

Trong 1,175 giao dịch được model gắn positive:

```text
TP:
393

FP:
782
```

Do đó khoảng:

```text
33.45%
```

alerts là fraud thật theo ground truth và khoảng:

```text
66.55%
```

là false positive.

Đây là descriptive operational fact.

---

## 20.3. F1

```text
F1_fraud ≈ 0.355656
```

Đây là:

`OFFICIAL PRIMARY FINAL TEST METRIC`

của frozen configuration:

```text
W_SHORT
Random Forest
RF-REF-100-GINI-SQRT-UNPRUNED-CW
CLASS_WEIGHT_BALANCED
random_state = 42
threshold = 0.50
risk_score > threshold
```

---

## 20.4. Accuracy

```text
Accuracy ≈ 99.803%
```

Không được diễn giải thành:

```text
"model phát hiện fraud rất tốt"
```

vì non-fraud chiếm gần toàn bộ population.

Accuracy chỉ là reference metric theo canonical metric contract.

---

# 21. Guardrail review

Observed:

```text
new_prediction_generated:
False

model_refit_performed:
False

train_validation_refit_performed:
False

preprocessing_refit_performed:
False

threshold_retuning_performed:
False

calibration_performed:
False

candidate_expansion_performed:
False

no_retroactive_optimization_attestation:
True
```

Source review cũng xác nhận:

```text
predict_proba:
0

predict:
0

fit:
0
```

Kết luận:

`PASS`

M8.4 chỉ đo official artifacts đã khóa.

---

# 22. Artifact persistence / round-trip

M8.4 đã persist:

```text
m8_04_final_metric_confusion_analysis.json
m8_04_analysis_manifest.json
```

Observed:

```text
Metric result round-trip:
PASS

Manifest round-trip:
PASS
```

Persisted values sau reload khớp runtime values.

Fingerprint kiểm tra thành công.

Kết luận:

`PASS`

---

# 23. Technical gate summary

Observed:

```text
G01_SOURCE_LOCATION:
PASS

G02_OUTPUT_GUARD:
PASS

G03_FROZEN_METRIC_CONTRACT:
PASS

G04_UPSTREAM_M8_3_RELEASE:
PASS

G05_M8_3_ARTIFACT_FINGERPRINTS:
PASS

G06_ROW_ALIGNMENT_TARGET_IDENTITY:
PASS

G07_FROZEN_PREDICTION_SEMANTICS:
PASS

G08_CONFUSION_ARITHMETIC:
PASS

G09_PRIMARY_F1_RECONSTRUCTION:
PASS

G10_SECONDARY_METRICS_RECONSTRUCTION:
PASS

G11_ACCURACY_REFERENCE:
PASS

G12_OPERATIONAL_DIAGNOSTIC:
PASS

G13_SKLEARN_CROSS_CHECK:
PASS

G14_ARTIFACT_PERSISTENCE:
PASS

G15_ARTIFACT_ROUND_TRIP:
PASS

G16_NO_NEW_PREDICTION:
PASS

G17_NO_RETROACTIVE_OPTIMIZATION:
PASS
```

Technical gate result:

`17 / 17 PASS`

---

# 24. Blocking-condition review

Observed:

```text
upstream M8.3 mismatch:
NONE

artifact fingerprint mismatch:
NONE

row-alignment mismatch:
NONE

target-support mismatch:
NONE

frozen-threshold mismatch:
NONE

confusion arithmetic mismatch:
NONE

manual/library metric mismatch:
NONE

new prediction generated:
NONE

model refit:
NONE

threshold retuning:
NONE

calibration:
NONE

candidate expansion:
NONE

artifact persistence failure:
NONE

round-trip failure:
NONE

retroactive optimization:
NONE
```

Blocking issue:

`NONE OBSERVED`

---

# 25. M8.4 Decision

Sau runtime review:

```text
M8.4:
PASS

Work type:
OFFICIAL FINAL TEST METRIC RECONSTRUCTION

Upstream M8.3:
PASS

FINAL TEST rows:
722,955

Fraud support:
1,035

Confusion Matrix:
TP = 393
FP = 782
FN = 642
TN = 721,138

Primary Metric:
F1_fraud = 0.3556561085972851

Secondary Metrics:
Recall_fraud = 0.37971014492753624
Precision_fraud = 0.334468085106383

Accuracy_reference:
0.9980303061739666

Predicted-positive count:
1,175

Predicted-positive rate:
0.0016252740488688785

Manual / sklearn cross-check:
PASS

New prediction generated:
NO

Retroactive optimization:
NO

Artifact persistence:
PASS

Artifact round-trip:
PASS

Technical gates:
17 / 17 PASS

Blocking issue:
NONE
```

## Decision

`M8.4 — PASS`

## Canonical FINAL TEST metric profile

`LOCKED`

M8.4 output trở thành canonical source cho official FINAL TEST metric profile.

---

# 26. Scope boundary sau M8.4

M8.4 đã trả lời:

```text
Official FINAL TEST metrics là bao nhiêu?
Confusion Matrix là gì?
Arithmetic và library cross-check có nhất quán không?
```

M8.4 chưa trả lời sâu:

```text
FN có pattern gì?
FP tập trung ở nhóm nào?
risk-score distribution của errors như thế nào?
monthly FINAL TEST behavior thay đổi ra sao?
validation → final metric delta thế nào?
temporal generalization limitation mạnh đến đâu?
```

Các câu hỏi này thuộc:

`M8.5 — Final FP/FN + Temporal Generalization Review`

---

# 27. Authorization sang M8.5

Precondition:

`M8.4 — PASS`

M8.5 status:

`AUTHORIZED`

Authorized next step:

`M8.5 — Final FP/FN + Temporal Generalization Review`

M8.5 được phép:

```text
analyze FP / FN patterns
analyze risk-score distributions by error group
analyze monthly FINAL TEST behavior
compare descriptive validation → FINAL TEST metric delta
review temporal generalization limitation
persist final error-analysis artifacts
```

M8.5 không được:

```text
change model
change feature/preprocessing
change threshold
try alternate threshold
refit
retune
calibrate
use FINAL TEST error findings để sửa frozen subject
```

Guardrail:

`ANALYZE — DO NOT FIX USING FINAL TEST`

---

# 28. Handoff sang M8.5

Canonical M8.4 source root:

```text
data/processed/m8_04_final_metric_analysis/
```

Canonical metric artifacts:

```text
m8_04_final_metric_confusion_analysis.json
m8_04_analysis_manifest.json
```

Official FINAL TEST metric profile:

`LOCKED`

M8.5 error / temporal review:

`AUTHORIZED`

Required interpretation order:

```text
OBSERVED OFFICIAL ARTIFACT
→ METRIC FACT
→ ERROR / TEMPORAL ANALYSIS
→ FINAL LIMITATION
```

