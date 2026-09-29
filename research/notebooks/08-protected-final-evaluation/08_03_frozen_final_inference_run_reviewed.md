# M8.3 — Frozen Final Inference Run

**Project:** AI Transaction Fraud Risk Screening  
**Milestone:** M8 — Protected Final Evaluation  
**Substep:** M8.3 — Frozen Final Inference Run  
**Work type:** Official inference artifact generation  
**Upstream:** M8.2 — PASS  
**Decision trước runtime:** `OPEN — REQUIRES RUNTIME REVIEW`

---

## Vai trò của notebook

M8.3 thực hiện **official inference đầu tiên trên protected FINAL TEST** bằng đúng subject đã được M8.2 audit và release.

Official sequence:

```text
load audited reconstructed selected estimator
→ load audited FINAL TEST representation
→ re-verify artifact fingerprints / feature schema / model identity
→ predict_proba exactly once
→ select positive class = 1
→ cast risk_score theo frozen validation-compatible semantics
→ apply risk_score > 0.50
→ persist official prediction artifacts
→ fingerprint
→ round-trip
```

M8.3 **không**:

- thay model;
- thay feature/preprocessing;
- fit lại estimator;
- TRAIN+VALIDATION refit;
- thay random seed;
- thử threshold khác;
- calibration;
- compute F1 / Recall / Precision / Confusion Matrix trên FINAL TEST;
- diễn giải model “tốt/xấu”;
- dùng FINAL TEST output để sửa frozen subject.

FINAL TEST performance interpretation vẫn thuộc M8.4/M8.5.



## 1. Frozen subject được release từ M8.2

```text
Training Window:
W_SHORT

Model:
Random Forest

Config:
RF-REF-100-GINI-SQRT-UNPRUNED-CW

Imbalance Strategy:
CLASS_WEIGHT_BALANCED

random_state:
42

Risk Score:
predict_proba / positive class = 1

Threshold:
0.50

Comparator:
risk_score > 0.50

FINAL TEST:
2019-06-01 <= Timestamp < 2019-11-01

Expected rows:
722,955

Expected fraud:
1,035

Representation:
47-column CSR float32
```

M8.2 evidence trước khi release:

```text
VALIDATION matrix exact reproduction:
difference nnz = 0

M7.7 validation prediction reproduction:
mismatch = 0

M7.7 validation risk-score reproduction:
mismatch = 0

threshold > 0.50 reproduction:
mismatch = 0

M8.2 technical gates:
21 / 21 PASS

M8.2 Decision:
PASS
```



```python

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import platform
import shutil
import sys
import time

import joblib
import numpy as np

try:
    import scipy
    from scipy import sparse
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Thiếu scipy. Hãy cài dependency của project rồi Restart Kernel."
    ) from exc

try:
    import sklearn
    from sklearn.ensemble import RandomForestClassifier
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Thiếu scikit-learn. Hãy cài dependency của project rồi Restart Kernel."
    ) from exc

print("Python:", sys.version)
print("Executable:", sys.executable)
print("Platform:", platform.platform())
print("NumPy:", np.__version__)
print("SciPy:", scipy.__version__)
print("scikit-learn:", sklearn.__version__)
print("joblib:", joblib.__version__)

```

    Python: 3.14.6 (main, Jun 10 2026, 10:03:53) [Clang 21.0.0 (clang-2100.0.123.102)]
    Executable: /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/.venv/bin/python
    Platform: macOS-26.6.2-arm64-arm-64bit-Mach-O
    NumPy: 2.5.3
    SciPy: 1.18.1
    scikit-learn: 1.9.1
    joblib: 1.6.0


## 2. Locate M8.2 audited artifacts và chuẩn bị M8.3 output root


```python

M4_REL = (
    Path("data")
    / "processed"
    / "m4_07_baseline_ready"
)

M8_02_REL = (
    Path("data")
    / "processed"
    / "m8_02_final_test_artifact_audit"
)

M8_03_REL = (
    Path("data")
    / "processed"
    / "m8_03_final_test_inference"
)

required_rel_paths = [
    M4_REL / "feature_names.json",
    M8_02_REL / "X_final_test_w_short.npz",
    M8_02_REL / "y_final_test.npy",
    M8_02_REL / "row_id_final_test.npy",
    M8_02_REL / "timestamp_final_test.npy",
    M8_02_REL / "m8_02_w_short_preprocessing_state.json",
    M8_02_REL / "m8_02_selected_rf_estimator.joblib",
    M8_02_REL / "m8_02_audit_registry.json",
    M8_02_REL / "m8_02_audit_manifest.json",
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
        in required_rel_paths
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
            in required_rel_paths
            if not (
                candidate / rel
            ).exists()
        ][:10]

    raise FileNotFoundError(
        "Không tìm thấy PROJECT_ROOT chứa đầy đủ audited M8.2 artifacts.\n"
        + json.dumps(
            missing_by_candidate,
            ensure_ascii=False,
            indent=2,
        )
    )

M4_DIR = PROJECT_ROOT / M4_REL
M8_02_DIR = PROJECT_ROOT / M8_02_REL
OUTPUT_DIR = PROJECT_ROOT / M8_03_REL

FEATURE_NAMES_PATH = (
    M4_DIR
    / "feature_names.json"
)

X_FINAL_SOURCE_PATH = (
    M8_02_DIR
    / "X_final_test_w_short.npz"
)

Y_FINAL_SOURCE_PATH = (
    M8_02_DIR
    / "y_final_test.npy"
)

ROW_FINAL_SOURCE_PATH = (
    M8_02_DIR
    / "row_id_final_test.npy"
)

TIMESTAMP_FINAL_SOURCE_PATH = (
    M8_02_DIR
    / "timestamp_final_test.npy"
)

PREPROCESSING_STATE_PATH = (
    M8_02_DIR
    / "m8_02_w_short_preprocessing_state.json"
)

ESTIMATOR_SOURCE_PATH = (
    M8_02_DIR
    / "m8_02_selected_rf_estimator.joblib"
)

M8_02_REGISTRY_PATH = (
    M8_02_DIR
    / "m8_02_audit_registry.json"
)

M8_02_MANIFEST_PATH = (
    M8_02_DIR
    / "m8_02_audit_manifest.json"
)

print("PROJECT_ROOT:", PROJECT_ROOT)
print("M8.2 source:", M8_02_DIR)
print("M8.3 output:", OUTPUT_DIR)
print("\nM8.3 SOURCE LOCATION GATE: PASS")

```

    PROJECT_ROOT: /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening
    M8.2 source: /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/processed/m8_02_final_test_artifact_audit
    M8.3 output: /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/processed/m8_03_final_test_inference
    
    M8.3 SOURCE LOCATION GATE: PASS



## 3. One-shot / rerun guard

Official FINAL TEST inference không được chạy lại tùy tiện.

Mặc định:

```text
ALLOW_EXACT_RERUN = False
```

Nếu một technical failure xảy ra **sau khi official inference đã bắt đầu**, chỉ được rerun cùng exact subject khi:

- không thay model/config/data/feature/threshold/seed;
- đặt `ALLOW_EXACT_RERUN = True`;
- ghi `RERUN_REASON` rõ ràng;
- giữ lại reason trong registry.

Không dùng rerun để thử cải thiện performance.



```python

ALLOW_EXACT_RERUN = False
RERUN_REASON = None

OFFICIAL_LOCK_PATH = (
    OUTPUT_DIR
    / "m8_03_official_inference.lock.json"
)

RISK_SCORE_PATH = (
    OUTPUT_DIR
    / "risk_score_final_test.npy"
)

Y_PRED_PATH = (
    OUTPUT_DIR
    / "y_pred_final_test.npy"
)

Y_FINAL_OUTPUT_PATH = (
    OUTPUT_DIR
    / "y_final_test.npy"
)

ROW_FINAL_OUTPUT_PATH = (
    OUTPUT_DIR
    / "row_id_final_test.npy"
)

TIMESTAMP_FINAL_OUTPUT_PATH = (
    OUTPUT_DIR
    / "timestamp_final_test.npy"
)

REGISTRY_PATH = (
    OUTPUT_DIR
    / "m8_03_final_inference_registry.json"
)

MANIFEST_PATH = (
    OUTPUT_DIR
    / "m8_03_inference_manifest.json"
)

official_output_paths = [
    OFFICIAL_LOCK_PATH,
    RISK_SCORE_PATH,
    Y_PRED_PATH,
    Y_FINAL_OUTPUT_PATH,
    ROW_FINAL_OUTPUT_PATH,
    TIMESTAMP_FINAL_OUTPUT_PATH,
    REGISTRY_PATH,
    MANIFEST_PATH,
]

existing_official_paths = [
    path
    for path
    in official_output_paths
    if path.exists()
]

if existing_official_paths:
    if not ALLOW_EXACT_RERUN:
        raise RuntimeError(
            "M8.3 official output đã tồn tại. "
            "Dừng để tránh accidental repeated FINAL TEST inference / overwrite.\n"
            + "\n".join(
                str(path)
                for path
                in existing_official_paths
            )
        )

    if (
        RERUN_REASON is None
        or not str(
            RERUN_REASON
        ).strip()
    ):
        raise RuntimeError(
            "Exact technical rerun yêu cầu RERUN_REASON không rỗng."
        )

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

print(
    "Existing official outputs:",
    len(existing_official_paths),
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
    "\nM8.3 ONE-SHOT / RERUN GUARD: PASS"
)

```

    Existing official outputs: 0
    Exact rerun enabled: False
    Rerun reason: None
    
    M8.3 ONE-SHOT / RERUN GUARD: PASS


## 4. Frozen constants + reviewed M8.2 release identity


```python

EXPECTED_FINAL_ROWS = 722_955
EXPECTED_FINAL_FRAUD = 1_035
EXPECTED_FEATURE_COUNT = 47

SELECTED_TRAINING_WINDOW = "W_SHORT"
SELECTED_MODEL_FAMILY = "Random Forest"
SELECTED_CANDIDATE_ID = "RF-REF-100-GINI-SQRT-UNPRUNED-CW"
SELECTED_IMBALANCE_STRATEGY = "CLASS_WEIGHT_BALANCED"
SELECTED_RANDOM_STATE = 42

POSITIVE_CLASS = 1
FINAL_THRESHOLD = 0.50
THRESHOLD_COMPARATOR = ">"

M8_03_ANALYSIS_VERSION = (
    "M8.3-frozen-final-inference-v1"
)

UPSTREAM_REVIEWED_M8_2_STATUS = "PASS"

# SHA256 của reviewed M8.2 notebook sau khi Findings / Decision
# đã được chèn và M8.2 được release sang M8.3.
UPSTREAM_REVIEWED_M8_2_NOTEBOOK_SHA256 = (
    "d4a378c084cd3c6bc029b821503e2cf1"
    "3c81535f617931f6209abac0231c41a5"
)

assert (
    UPSTREAM_REVIEWED_M8_2_STATUS
    == "PASS"
)

assert FINAL_THRESHOLD == 0.50
assert THRESHOLD_COMPARATOR == ">"

print(
    "Reviewed M8.2:",
    UPSTREAM_REVIEWED_M8_2_STATUS,
)
print(
    "Frozen candidate:",
    SELECTED_CANDIDATE_ID,
)
print(
    "Frozen threshold:",
    FINAL_THRESHOLD,
)
print(
    "Comparator:",
    THRESHOLD_COMPARATOR,
)
print(
    "\nM8.3 FROZEN SUBJECT GATE: PASS"
)

```

    Reviewed M8.2: PASS
    Frozen candidate: RF-REF-100-GINI-SQRT-UNPRUNED-CW
    Frozen threshold: 0.5
    Comparator: >
    
    M8.3 FROZEN SUBJECT GATE: PASS


## 5. Load M8.2 registry / manifest / preprocessing identity


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
    M8_02_REGISTRY_PATH,
    "r",
    encoding="utf-8",
) as file:
    m8_02_registry = json.load(
        file
    )

with open(
    M8_02_MANIFEST_PATH,
    "r",
    encoding="utf-8",
) as file:
    m8_02_manifest = json.load(
        file
    )

with open(
    PREPROCESSING_STATE_PATH,
    "r",
    encoding="utf-8",
) as file:
    preprocessing_state = json.load(
        file
    )

with open(
    FEATURE_NAMES_PATH,
    "r",
    encoding="utf-8",
) as file:
    canonical_feature_names = json.load(
        file
    )

assert (
    m8_02_registry[
        "analysis_version"
    ]
    ==
    "M8.2-final-test-artifact-audit-v1"
)

assert (
    m8_02_manifest[
        "analysis_version"
    ]
    ==
    "M8.2-final-test-artifact-audit-v1"
)

# Persisted M8.2 registry được tạo trước AI review nên decision
# lịch sử vẫn OPEN. Reviewed notebook sau đó đã khóa M8.2 PASS.
assert str(
    m8_02_registry[
        "decision"
    ]
).startswith(
    "OPEN"
)

assert str(
    m8_02_manifest[
        "decision"
    ]
).startswith(
    "OPEN"
)

assert all(
    m8_02_registry[
        "technical_gates"
    ].values()
)

assert (
    m8_02_manifest[
        "technical_gate_count"
    ]
    ==
    21
)

assert (
    m8_02_manifest[
        "technical_gate_pass_count"
    ]
    ==
    21
)

assert (
    m8_02_registry[
        "final_test_prediction_created"
    ]
    is False
)

assert (
    m8_02_registry[
        "final_test_metric_computed"
    ]
    is False
)

assert (
    m8_02_registry[
        "train_validation_refit_performed"
    ]
    is False
)

print(
    "M8.2 technical gates:",
    m8_02_manifest[
        "technical_gate_pass_count"
    ],
    "/",
    m8_02_manifest[
        "technical_gate_count"
    ],
)
print(
    "M8.2 persisted decision state:",
    m8_02_registry["decision"],
)
print(
    "Reviewed release state:",
    UPSTREAM_REVIEWED_M8_2_STATUS,
)
print(
    "\nM8.3 UPSTREAM M8.2 RELEASE GATE: PASS"
)

```

    M8.2 technical gates: 21 / 21
    M8.2 persisted decision state: OPEN — REQUIRES AI RUNTIME REVIEW
    Reviewed release state: PASS
    
    M8.3 UPSTREAM M8.2 RELEASE GATE: PASS


## 6. Re-verify M8.2 artifact fingerprints trước official inference


```python

generated_fp = (
    m8_02_registry[
        "generated_fingerprints"
    ]
)

fingerprint_checks = {
    "X_final_test_w_short_sha256": (
        X_FINAL_SOURCE_PATH
    ),
    "y_final_test_sha256": (
        Y_FINAL_SOURCE_PATH
    ),
    "row_id_final_test_sha256": (
        ROW_FINAL_SOURCE_PATH
    ),
    "timestamp_final_test_sha256": (
        TIMESTAMP_FINAL_SOURCE_PATH
    ),
    "preprocessing_state_sha256": (
        PREPROCESSING_STATE_PATH
    ),
    "selected_rf_estimator_sha256": (
        ESTIMATOR_SOURCE_PATH
    ),
}

observed_source_fingerprints = {}

for key, path in (
    fingerprint_checks.items()
):
    observed = sha256_file(
        path
    )

    expected = generated_fp[
        key
    ]

    if observed != expected:
        raise RuntimeError(
            "M8.2 artifact fingerprint mismatch: "
            f"{key}\n"
            f"expected={expected}\n"
            f"observed={observed}"
        )

    observed_source_fingerprints[
        key
    ] = observed

print(
    "Verified M8.2 generated artifact fingerprints:",
    len(
        observed_source_fingerprints
    ),
)
print(
    "\nM8.3 M8.2 ARTIFACT FINGERPRINT GATE: PASS"
)

```

    Verified M8.2 generated artifact fingerprints: 6
    
    M8.3 M8.2 ARTIFACT FINGERPRINT GATE: PASS


## 7. Load audited FINAL TEST representation + lineage + target identity


```python

X_final_test = sparse.load_npz(
    X_FINAL_SOURCE_PATH
)

y_final_test = np.load(
    Y_FINAL_SOURCE_PATH,
    allow_pickle=False,
)

row_id_final_test = np.load(
    ROW_FINAL_SOURCE_PATH,
    allow_pickle=False,
)

timestamp_final_test = np.load(
    TIMESTAMP_FINAL_SOURCE_PATH,
    allow_pickle=False,
)

assert sparse.isspmatrix_csr(
    X_final_test
)

assert (
    X_final_test.shape
    ==
    (
        EXPECTED_FINAL_ROWS,
        EXPECTED_FEATURE_COUNT,
    )
)

assert (
    X_final_test.dtype
    == np.float32
)

assert np.isfinite(
    X_final_test.data
).all()

assert (
    y_final_test.shape
    ==
    (
        EXPECTED_FINAL_ROWS,
    )
)

assert (
    y_final_test.dtype
    == np.int8
)

assert (
    int(
        y_final_test.sum()
    )
    ==
    EXPECTED_FINAL_FRAUD
)

assert (
    row_id_final_test.shape
    ==
    (
        EXPECTED_FINAL_ROWS,
    )
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

assert (
    timestamp_final_test.shape
    ==
    (
        EXPECTED_FINAL_ROWS,
    )
)

print(
    "FINAL TEST matrix:",
    X_final_test.shape,
    X_final_test.dtype,
    "| nnz:",
    X_final_test.nnz,
)
print(
    "FINAL TEST target identity:",
    len(y_final_test),
    "rows /",
    int(
        y_final_test.sum()
    ),
    "fraud",
)
print(
    "FINAL TEST lineage rows:",
    len(
        row_id_final_test
    ),
)
print(
    "\nM8.3 FINAL TEST INPUT IDENTITY GATE: PASS"
)

```

    FINAL TEST matrix: (722955, 47) float32 | nnz: 6526714
    FINAL TEST target identity: 722955 rows / 1035 fraud
    FINAL TEST lineage rows: 722955
    
    M8.3 FINAL TEST INPUT IDENTITY GATE: PASS


## 8. Feature-name / preprocessing identity


```python

assert (
    preprocessing_state[
        "strategy"
    ]
    ==
    "W_SHORT"
)

assert (
    preprocessing_state[
        "fit_source"
    ]
    ==
    "W_SHORT_TRAIN_ONLY"
)

assert (
    preprocessing_state[
        "feature_count"
    ]
    ==
    EXPECTED_FEATURE_COUNT
)

assert (
    preprocessing_state[
        "matrix_format"
    ]
    ==
    "CSR"
)

assert (
    preprocessing_state[
        "matrix_dtype"
    ]
    ==
    "float32"
)

assert (
    preprocessing_state[
        "validation_exact_reproduction"
    ]
    is True
)

assert (
    preprocessing_state[
        "feature_names"
    ]
    ==
    canonical_feature_names
)

assert (
    len(
        canonical_feature_names
    )
    ==
    EXPECTED_FEATURE_COUNT
)

print(
    "Feature names:",
    len(
        canonical_feature_names
    ),
)
print(
    "Preprocessing strategy:",
    preprocessing_state[
        "strategy"
    ],
)
print(
    "Fit source:",
    preprocessing_state[
        "fit_source"
    ],
)
print(
    "\nM8.3 FEATURE / PREPROCESSING IDENTITY GATE: PASS"
)

```

    Feature names: 47
    Preprocessing strategy: W_SHORT
    Fit source: W_SHORT_TRAIN_ONLY
    
    M8.3 FEATURE / PREPROCESSING IDENTITY GATE: PASS


## 9. Load audited reconstructed estimator và verify exact frozen identity


```python

selected_rf_estimator = joblib.load(
    ESTIMATOR_SOURCE_PATH
)

assert isinstance(
    selected_rf_estimator,
    RandomForestClassifier,
)

official_subject = (
    m8_02_registry[
        "official_subject"
    ]
)

assert (
    official_subject[
        "training_window"
    ]
    ==
    SELECTED_TRAINING_WINDOW
)

assert (
    official_subject[
        "model_family"
    ]
    ==
    SELECTED_MODEL_FAMILY
)

assert (
    official_subject[
        "config_id"
    ]
    ==
    SELECTED_CANDIDATE_ID
)

assert (
    official_subject[
        "imbalance_strategy"
    ]
    ==
    SELECTED_IMBALANCE_STRATEGY
)

assert int(
    official_subject[
        "random_state"
    ]
) == SELECTED_RANDOM_STATE

assert float(
    official_subject[
        "threshold"
    ]
) == FINAL_THRESHOLD

assert (
    official_subject[
        "threshold_comparator"
    ]
    ==
    THRESHOLD_COMPARATOR
)

estimator_params = (
    selected_rf_estimator
    .get_params()
)

expected_param_subset = (
    official_subject[
        "hyperparameters"
    ]
)

for key, expected_value in (
    expected_param_subset.items()
):
    observed_value = (
        estimator_params[
            key
        ]
    )

    if observed_value != expected_value:
        raise RuntimeError(
            "Estimator parameter mismatch: "
            f"{key} | "
            f"expected={expected_value!r} | "
            f"observed={observed_value!r}"
        )

assert (
    estimator_params[
        "n_estimators"
    ]
    ==
    100
)

assert (
    estimator_params[
        "class_weight"
    ]
    ==
    "balanced"
)

assert (
    estimator_params[
        "random_state"
    ]
    ==
    SELECTED_RANDOM_STATE
)

classes = np.asarray(
    selected_rf_estimator.classes_
)

positive_positions = np.flatnonzero(
    classes
    ==
    POSITIVE_CLASS
)

assert (
    len(
        positive_positions
    )
    ==
    1
)

positive_index = int(
    positive_positions[
        0
    ]
)

print(
    "Estimator:",
    type(
        selected_rf_estimator
    ).__name__,
)
print(
    "Classes:",
    classes.tolist(),
)
print(
    "Positive-class index:",
    positive_index,
)
print(
    "\nM8.3 ESTIMATOR / POSITIVE-CLASS IDENTITY GATE: PASS"
)

```

    Estimator: RandomForestClassifier
    Classes: [0, 1]
    Positive-class index: 1
    
    M8.3 ESTIMATOR / POSITIVE-CLASS IDENTITY GATE: PASS



## 10. Pre-inference freeze attestation

Đây là gate cuối trước official access.

Notebook phải xác nhận:

```text
no model fit/refit
no preprocessing fit/update
no threshold change
no calibration
no alternate candidate
no alternate feature representation
no performance-driven change
```



```python

NO_MODEL_REFIT = True
NO_TRAIN_VALIDATION_REFIT = True
NO_PREPROCESSING_REFIT = True
NO_THRESHOLD_RETUNING = True
NO_CALIBRATION = True
NO_CANDIDATE_EXPANSION = True
NO_FEATURE_CHANGE = True
NO_SEED_CHANGE = True
NO_FINAL_TEST_METRIC_BEFORE_INFERENCE = True

freeze_attestation = {
    "no_model_refit":
        NO_MODEL_REFIT,
    "no_train_validation_refit":
        NO_TRAIN_VALIDATION_REFIT,
    "no_preprocessing_refit":
        NO_PREPROCESSING_REFIT,
    "no_threshold_retuning":
        NO_THRESHOLD_RETUNING,
    "no_calibration":
        NO_CALIBRATION,
    "no_candidate_expansion":
        NO_CANDIDATE_EXPANSION,
    "no_feature_change":
        NO_FEATURE_CHANGE,
    "no_seed_change":
        NO_SEED_CHANGE,
    "no_final_test_metric_before_inference":
        NO_FINAL_TEST_METRIC_BEFORE_INFERENCE,
}

assert all(
    freeze_attestation.values()
)

print(
    json.dumps(
        freeze_attestation,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
)
print(
    "\nM8.3 PRE-INFERENCE FREEZE ATTESTATION: PASS"
)

```

    {
      "no_calibration": true,
      "no_candidate_expansion": true,
      "no_feature_change": true,
      "no_final_test_metric_before_inference": true,
      "no_model_refit": true,
      "no_preprocessing_refit": true,
      "no_seed_change": true,
      "no_threshold_retuning": true,
      "no_train_validation_refit": true
    }
    
    M8.3 PRE-INFERENCE FREEZE ATTESTATION: PASS



# 11. OFFICIAL FINAL TEST INFERENCE — ONE SHOT

**Đây là cell duy nhất được phép gọi `predict_proba(X_final_test)`.**

Không gọi `predict()`.

Binary prediction được tạo **chỉ** từ frozen rule:

```text
risk_score > 0.50
```

Risk score được cast `float32` để giữ semantics tương thích với exact M7.7/M8.2 validation reproduction.



```python

if (
    "OFFICIAL_INFERENCE_EXECUTED"
    in globals()
    and OFFICIAL_INFERENCE_EXECUTED
):
    raise RuntimeError(
        "Official inference đã được execute trong kernel này. "
        "Không được chạy lại cell."
    )

OFFICIAL_INFERENCE_EXECUTED = False
OFFICIAL_INFERENCE_CALL_COUNT = 0

inference_started_at_utc = (
    datetime.now(
        timezone.utc
    ).isoformat()
)

lock_payload = {
    "analysis_version":
        M8_03_ANALYSIS_VERSION,
    "status":
        "INFERENCE_STARTED",
    "started_at_utc":
        inference_started_at_utc,
    "selected_candidate_id":
        SELECTED_CANDIDATE_ID,
    "threshold":
        FINAL_THRESHOLD,
    "threshold_comparator":
        THRESHOLD_COMPARATOR,
    "allow_exact_rerun":
        ALLOW_EXACT_RERUN,
    "rerun_reason":
        RERUN_REASON,
}

with open(
    OFFICIAL_LOCK_PATH,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        lock_payload,
        file,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )

inference_start = (
    time.perf_counter()
)

# ============================================================
# OFFICIAL FINAL TEST MODEL ACCESS — EXACTLY ONE predict_proba
# ============================================================
official_probability_matrix = (
    selected_rf_estimator
    .predict_proba(
        X_final_test
    )
)

OFFICIAL_INFERENCE_CALL_COUNT += 1
OFFICIAL_INFERENCE_EXECUTED = True

inference_seconds = (
    time.perf_counter()
    - inference_start
)

assert (
    OFFICIAL_INFERENCE_CALL_COUNT
    ==
    1
)

assert (
    official_probability_matrix.shape[0]
    ==
    EXPECTED_FINAL_ROWS
)

assert (
    official_probability_matrix.shape[1]
    ==
    len(
        classes
    )
)

assert np.isfinite(
    official_probability_matrix
).all()

risk_score_final_test = (
    official_probability_matrix[
        :,
        positive_index,
    ]
    .astype(
        np.float32,
        copy=False,
    )
)

del official_probability_matrix

assert (
    risk_score_final_test.shape
    ==
    (
        EXPECTED_FINAL_ROWS,
    )
)

assert (
    risk_score_final_test.dtype
    ==
    np.float32
)

assert np.isfinite(
    risk_score_final_test
).all()

assert np.all(
    risk_score_final_test
    >= 0.0
)

assert np.all(
    risk_score_final_test
    <= 1.0
)

# Frozen comparator — không thử comparator khác.
y_pred_final_test = (
    risk_score_final_test
    >
    FINAL_THRESHOLD
).astype(
    np.int8,
    copy=False,
)

assert (
    y_pred_final_test.shape
    ==
    (
        EXPECTED_FINAL_ROWS,
    )
)

assert (
    y_pred_final_test.dtype
    ==
    np.int8
)

assert np.isin(
    y_pred_final_test,
    [0, 1],
).all()

inference_completed_at_utc = (
    datetime.now(
        timezone.utc
    ).isoformat()
)

lock_payload[
    "status"
] = "INFERENCE_COMPLETED"

lock_payload[
    "completed_at_utc"
] = (
    inference_completed_at_utc
)

lock_payload[
    "predict_proba_call_count"
] = (
    OFFICIAL_INFERENCE_CALL_COUNT
)

with open(
    OFFICIAL_LOCK_PATH,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        lock_payload,
        file,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )

print(
    "Official risk-score rows:",
    len(
        risk_score_final_test
    ),
)
print(
    "Official prediction rows:",
    len(
        y_pred_final_test
    ),
)
print(
    "predict_proba call count:",
    OFFICIAL_INFERENCE_CALL_COUNT,
)
print(
    "Inference seconds:",
    round(
        inference_seconds,
        3,
    ),
)
print(
    "\nM8.3 OFFICIAL INFERENCE EXECUTION GATE: PASS"
)

```

    Official risk-score rows: 722955
    Official prediction rows: 722955
    predict_proba call count: 1
    Inference seconds: 0.445
    
    M8.3 OFFICIAL INFERENCE EXECUTION GATE: PASS



## 12. Prediction artifact integrity — không compute target-dependent performance

Cell này chỉ kiểm tra artifact mechanics:

- shape;
- dtype;
- finite;
- score range;
- binary prediction;
- threshold rule consistency.

Không dùng `y_final_test` để tính metric.



```python

assert (
    len(
        risk_score_final_test
    )
    ==
    len(
        y_pred_final_test
    )
    ==
    len(
        row_id_final_test
    )
    ==
    EXPECTED_FINAL_ROWS
)

threshold_reconstructed = (
    risk_score_final_test
    >
    FINAL_THRESHOLD
).astype(
    np.int8
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

assert np.isfinite(
    risk_score_final_test
).all()

assert np.isin(
    y_pred_final_test,
    [0, 1],
).all()

print(
    "Rows:",
    EXPECTED_FINAL_ROWS,
)
print(
    "Risk-score dtype:",
    risk_score_final_test.dtype,
)
print(
    "Prediction dtype:",
    y_pred_final_test.dtype,
)
print(
    "Threshold reconstruction mismatch:",
    threshold_mismatch_count,
)
print(
    "\nM8.3 PREDICTION ARTIFACT INTEGRITY GATE: PASS"
)

```

    Rows: 722955
    Risk-score dtype: float32
    Prediction dtype: int8
    Threshold reconstruction mismatch: 0
    
    M8.3 PREDICTION ARTIFACT INTEGRITY GATE: PASS


## 13. Persist official M8.3 inference artifacts


```python

np.save(
    RISK_SCORE_PATH,
    risk_score_final_test,
    allow_pickle=False,
)

np.save(
    Y_PRED_PATH,
    y_pred_final_test,
    allow_pickle=False,
)

# Target / lineage được copy nguyên byte từ audited M8.2 artifacts.
shutil.copy2(
    Y_FINAL_SOURCE_PATH,
    Y_FINAL_OUTPUT_PATH,
)

shutil.copy2(
    ROW_FINAL_SOURCE_PATH,
    ROW_FINAL_OUTPUT_PATH,
)

shutil.copy2(
    TIMESTAMP_FINAL_SOURCE_PATH,
    TIMESTAMP_FINAL_OUTPUT_PATH,
)

for path in [
    RISK_SCORE_PATH,
    Y_PRED_PATH,
    Y_FINAL_OUTPUT_PATH,
    ROW_FINAL_OUTPUT_PATH,
    TIMESTAMP_FINAL_OUTPUT_PATH,
    OFFICIAL_LOCK_PATH,
]:
    assert path.exists()
    assert path.stat().st_size > 0

print("Persisted official artifacts:")

for path in [
    RISK_SCORE_PATH,
    Y_PRED_PATH,
    Y_FINAL_OUTPUT_PATH,
    ROW_FINAL_OUTPUT_PATH,
    TIMESTAMP_FINAL_OUTPUT_PATH,
    OFFICIAL_LOCK_PATH,
]:
    print(
        " ",
        path.relative_to(
            PROJECT_ROOT
        ),
        "|",
        f"{path.stat().st_size:,}",
        "bytes",
    )

print(
    "\nM8.3 OFFICIAL ARTIFACT PERSISTENCE GATE: PASS"
)

```

    Persisted official artifacts:
      data/processed/m8_03_final_test_inference/risk_score_final_test.npy | 2,891,948 bytes
      data/processed/m8_03_final_test_inference/y_pred_final_test.npy | 723,083 bytes
      data/processed/m8_03_final_test_inference/y_final_test.npy | 723,083 bytes
      data/processed/m8_03_final_test_inference/row_id_final_test.npy | 5,783,768 bytes
      data/processed/m8_03_final_test_inference/timestamp_final_test.npy | 5,783,768 bytes
      data/processed/m8_03_final_test_inference/m8_03_official_inference.lock.json | 408 bytes
    
    M8.3 OFFICIAL ARTIFACT PERSISTENCE GATE: PASS


## 14. Fingerprint official artifacts + persist registry / manifest


```python

source_fingerprints = {
    "m8_02_registry_sha256":
        sha256_file(
            M8_02_REGISTRY_PATH
        ),
    "m8_02_manifest_sha256":
        sha256_file(
            M8_02_MANIFEST_PATH
        ),
    "m8_02_x_final_test_sha256":
        sha256_file(
            X_FINAL_SOURCE_PATH
        ),
    "m8_02_y_final_test_sha256":
        sha256_file(
            Y_FINAL_SOURCE_PATH
        ),
    "m8_02_row_id_final_test_sha256":
        sha256_file(
            ROW_FINAL_SOURCE_PATH
        ),
    "m8_02_timestamp_final_test_sha256":
        sha256_file(
            TIMESTAMP_FINAL_SOURCE_PATH
        ),
    "m8_02_preprocessing_state_sha256":
        sha256_file(
            PREPROCESSING_STATE_PATH
        ),
    "m8_02_selected_estimator_sha256":
        sha256_file(
            ESTIMATOR_SOURCE_PATH
        ),
    "m4_feature_names_sha256":
        sha256_file(
            FEATURE_NAMES_PATH
        ),
    "reviewed_m8_02_notebook_sha256":
        UPSTREAM_REVIEWED_M8_2_NOTEBOOK_SHA256,
}

generated_fingerprints = {
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
            Y_FINAL_OUTPUT_PATH
        ),
    "row_id_final_test_sha256":
        sha256_file(
            ROW_FINAL_OUTPUT_PATH
        ),
    "timestamp_final_test_sha256":
        sha256_file(
            TIMESTAMP_FINAL_OUTPUT_PATH
        ),
    "official_inference_lock_sha256":
        sha256_file(
            OFFICIAL_LOCK_PATH
        ),
}

# Byte-identical lineage copies.
assert (
    generated_fingerprints[
        "y_final_test_sha256"
    ]
    ==
    source_fingerprints[
        "m8_02_y_final_test_sha256"
    ]
)

assert (
    generated_fingerprints[
        "row_id_final_test_sha256"
    ]
    ==
    source_fingerprints[
        "m8_02_row_id_final_test_sha256"
    ]
)

assert (
    generated_fingerprints[
        "timestamp_final_test_sha256"
    ]
    ==
    source_fingerprints[
        "m8_02_timestamp_final_test_sha256"
    ]
)

runtime_versions = {
    "python":
        sys.version,
    "numpy":
        np.__version__,
    "scipy":
        scipy.__version__,
    "scikit_learn":
        sklearn.__version__,
    "joblib":
        joblib.__version__,
    "platform":
        platform.platform(),
}

registry_payload = {
    "analysis_version":
        M8_03_ANALYSIS_VERSION,
    "work_type":
        "OFFICIAL_FROZEN_FINAL_TEST_INFERENCE",
    "upstream_review": {
        "m8_02_status":
            UPSTREAM_REVIEWED_M8_2_STATUS,
        "reviewed_m8_02_notebook_sha256":
            UPSTREAM_REVIEWED_M8_2_NOTEBOOK_SHA256,
        "m8_02_technical_gate_count":
            int(
                m8_02_manifest[
                    "technical_gate_count"
                ]
            ),
        "m8_02_technical_gate_pass_count":
            int(
                m8_02_manifest[
                    "technical_gate_pass_count"
                ]
            ),
    },
    "official_subject": {
        "training_window":
            SELECTED_TRAINING_WINDOW,
        "model_family":
            SELECTED_MODEL_FAMILY,
        "config_id":
            SELECTED_CANDIDATE_ID,
        "imbalance_strategy":
            SELECTED_IMBALANCE_STRATEGY,
        "random_state":
            SELECTED_RANDOM_STATE,
        "hyperparameters":
            expected_param_subset,
        "probability_interface":
            "predict_proba / positive class = 1",
        "positive_class":
            POSITIVE_CLASS,
        "positive_class_index":
            positive_index,
        "threshold":
            FINAL_THRESHOLD,
        "threshold_comparator":
            THRESHOLD_COMPARATOR,
    },
    "final_test_identity": {
        "rows":
            EXPECTED_FINAL_ROWS,
        "fraud":
            EXPECTED_FINAL_FRAUD,
        "feature_count":
            EXPECTED_FEATURE_COUNT,
        "matrix_format":
            "CSR",
        "matrix_dtype":
            "float32",
    },
    "inference_execution": {
        "started_at_utc":
            inference_started_at_utc,
        "completed_at_utc":
            inference_completed_at_utc,
        "seconds":
            float(
                inference_seconds
            ),
        "predict_proba_call_count":
            OFFICIAL_INFERENCE_CALL_COUNT,
        "risk_score_dtype":
            str(
                risk_score_final_test.dtype
            ),
        "prediction_dtype":
            str(
                y_pred_final_test.dtype
            ),
        "risk_score_finite":
            bool(
                np.isfinite(
                    risk_score_final_test
                ).all()
            ),
        "risk_score_within_0_1":
            bool(
                np.all(
                    risk_score_final_test
                    >= 0.0
                )
                and
                np.all(
                    risk_score_final_test
                    <= 1.0
                )
            ),
        "threshold_reconstruction_mismatch_count":
            threshold_mismatch_count,
    },
    "rerun_policy": {
        "allow_exact_rerun":
            ALLOW_EXACT_RERUN,
        "rerun_reason":
            RERUN_REASON,
    },
    "freeze_attestation":
        freeze_attestation,
    "source_fingerprints":
        source_fingerprints,
    "generated_fingerprints":
        generated_fingerprints,
    "runtime_versions":
        runtime_versions,
    "final_test_accessed":
        True,
    "official_final_inference_executed":
        True,
    "final_test_metric_computed":
        False,
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
    "no_retuning_attestation":
        True,
    "decision":
        "OPEN — REQUIRES AI RUNTIME REVIEW",
    "m8_4_metric_analysis_authorization":
        "NOT YET — REQUIRES REVIEW",
}

with open(
    REGISTRY_PATH,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        registry_payload,
        file,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        default=str,
    )

registry_sha256 = sha256_file(
    REGISTRY_PATH
)

manifest_payload = {
    "analysis_version":
        M8_03_ANALYSIS_VERSION,
    "selected_candidate_id":
        SELECTED_CANDIDATE_ID,
    "final_test_rows":
        EXPECTED_FINAL_ROWS,
    "feature_count":
        EXPECTED_FEATURE_COUNT,
    "positive_class":
        POSITIVE_CLASS,
    "threshold":
        FINAL_THRESHOLD,
    "threshold_comparator":
        THRESHOLD_COMPARATOR,
    "predict_proba_call_count":
        OFFICIAL_INFERENCE_CALL_COUNT,
    "final_test_accessed":
        True,
    "official_final_inference_executed":
        True,
    "final_test_metric_computed":
        False,
    "no_retuning_attestation":
        True,
    "risk_score_file":
        RISK_SCORE_PATH.name,
    "prediction_file":
        Y_PRED_PATH.name,
    "target_file":
        Y_FINAL_OUTPUT_PATH.name,
    "row_id_file":
        ROW_FINAL_OUTPUT_PATH.name,
    "timestamp_file":
        TIMESTAMP_FINAL_OUTPUT_PATH.name,
    "registry_file":
        REGISTRY_PATH.name,
    "registry_sha256":
        registry_sha256,
    "generated_fingerprints":
        generated_fingerprints,
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
        default=str,
    )

assert REGISTRY_PATH.exists()
assert MANIFEST_PATH.exists()

print(
    "Registry:",
    REGISTRY_PATH.relative_to(
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
    "Registry SHA256:",
    registry_sha256,
)
print(
    "Manifest SHA256:",
    sha256_file(
        MANIFEST_PATH
    ),
)
print(
    "\nM8.3 REGISTRY / MANIFEST PERSISTENCE GATE: PASS"
)

```

    Registry: data/processed/m8_03_final_test_inference/m8_03_final_inference_registry.json
    Manifest: data/processed/m8_03_final_test_inference/m8_03_inference_manifest.json
    Registry SHA256: 7a03495f08ff1dd0f5d6488f67d8439500699adabfea28c9724620ffe67436ce
    Manifest SHA256: 922722f935236ce4a4dc0bc440c83d79d365a2eab8a8ce36a31417154fdfaea7
    
    M8.3 REGISTRY / MANIFEST PERSISTENCE GATE: PASS


## 15. Official artifact round-trip


```python

risk_score_roundtrip = np.load(
    RISK_SCORE_PATH,
    allow_pickle=False,
)

y_pred_roundtrip = np.load(
    Y_PRED_PATH,
    allow_pickle=False,
)

y_final_roundtrip = np.load(
    Y_FINAL_OUTPUT_PATH,
    allow_pickle=False,
)

row_id_roundtrip = np.load(
    ROW_FINAL_OUTPUT_PATH,
    allow_pickle=False,
)

timestamp_roundtrip = np.load(
    TIMESTAMP_FINAL_OUTPUT_PATH,
    allow_pickle=False,
)

with open(
    REGISTRY_PATH,
    "r",
    encoding="utf-8",
) as file:
    registry_roundtrip = json.load(
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

np.testing.assert_array_equal(
    risk_score_roundtrip,
    risk_score_final_test,
)

np.testing.assert_array_equal(
    y_pred_roundtrip,
    y_pred_final_test,
)

np.testing.assert_array_equal(
    y_final_roundtrip,
    y_final_test,
)

np.testing.assert_array_equal(
    row_id_roundtrip,
    row_id_final_test,
)

np.testing.assert_array_equal(
    timestamp_roundtrip,
    timestamp_final_test,
)

assert (
    registry_roundtrip[
        "official_final_inference_executed"
    ]
    is True
)

assert (
    registry_roundtrip[
        "final_test_metric_computed"
    ]
    is False
)

assert (
    registry_roundtrip[
        "inference_execution"
    ][
        "predict_proba_call_count"
    ]
    ==
    1
)

assert (
    manifest_roundtrip[
        "registry_sha256"
    ]
    ==
    sha256_file(
        REGISTRY_PATH
    )
)

for key, expected_hash in (
    generated_fingerprints.items()
):
    if key == "risk_score_final_test_sha256":
        path = RISK_SCORE_PATH
    elif key == "y_pred_final_test_sha256":
        path = Y_PRED_PATH
    elif key == "y_final_test_sha256":
        path = Y_FINAL_OUTPUT_PATH
    elif key == "row_id_final_test_sha256":
        path = ROW_FINAL_OUTPUT_PATH
    elif key == "timestamp_final_test_sha256":
        path = TIMESTAMP_FINAL_OUTPUT_PATH
    elif key == "official_inference_lock_sha256":
        path = OFFICIAL_LOCK_PATH
    else:
        raise AssertionError(
            f"Unexpected fingerprint key: {key}"
        )

    assert (
        sha256_file(
            path
        )
        ==
        expected_hash
    )

print(
    "Official risk-score round-trip rows:",
    len(
        risk_score_roundtrip
    ),
)
print(
    "Official prediction round-trip rows:",
    len(
        y_pred_roundtrip
    ),
)
print(
    "Registry round-trip:",
    "PASS",
)
print(
    "Manifest round-trip:",
    "PASS",
)
print(
    "\nM8.3 OFFICIAL ARTIFACT ROUND-TRIP GATE: PASS"
)

```

    Official risk-score round-trip rows: 722955
    Official prediction round-trip rows: 722955
    Registry round-trip: PASS
    Manifest round-trip: PASS
    
    M8.3 OFFICIAL ARTIFACT ROUND-TRIP GATE: PASS



## 16. M8.3 technical gate

Technical gate chỉ xác nhận inference artifact integrity.

Nó **không** phải performance conclusion.



```python

m8_03_gates = {
    "G01_SOURCE_LOCATION":
        True,
    "G02_ONE_SHOT_RERUN_GUARD":
        True,
    "G03_FROZEN_SUBJECT":
        True,
    "G04_UPSTREAM_M8_2_RELEASE":
        True,
    "G05_M8_2_ARTIFACT_FINGERPRINTS":
        True,
    "G06_FINAL_TEST_INPUT_IDENTITY":
        True,
    "G07_FEATURE_PREPROCESSING_IDENTITY":
        True,
    "G08_ESTIMATOR_IDENTITY":
        True,
    "G09_POSITIVE_CLASS_IDENTITY":
        True,
    "G10_PRE_INFERENCE_FREEZE_ATTESTATION":
        all(
            freeze_attestation.values()
        ),
    "G11_OFFICIAL_INFERENCE_EXECUTED_ONCE":
        (
            OFFICIAL_INFERENCE_EXECUTED
            and
            OFFICIAL_INFERENCE_CALL_COUNT
            == 1
        ),
    "G12_RISK_SCORE_INTEGRITY":
        (
            risk_score_final_test.shape
            ==
            (
                EXPECTED_FINAL_ROWS,
            )
            and
            risk_score_final_test.dtype
            ==
            np.float32
            and
            np.isfinite(
                risk_score_final_test
            ).all()
        ),
    "G13_THRESHOLD_RULE_INTEGRITY":
        (
            threshold_mismatch_count
            ==
            0
        ),
    "G14_OFFICIAL_ARTIFACT_PERSISTENCE":
        all(
            path.exists()
            and
            path.stat().st_size
            > 0
            for path
            in [
                RISK_SCORE_PATH,
                Y_PRED_PATH,
                Y_FINAL_OUTPUT_PATH,
                ROW_FINAL_OUTPUT_PATH,
                TIMESTAMP_FINAL_OUTPUT_PATH,
                REGISTRY_PATH,
                MANIFEST_PATH,
                OFFICIAL_LOCK_PATH,
            ]
        ),
    "G15_OFFICIAL_ARTIFACT_ROUND_TRIP":
        True,
    "G16_NO_FINAL_TEST_METRIC":
        (
            registry_roundtrip[
                "final_test_metric_computed"
            ]
            is False
        ),
    "G17_NO_REFIT_RETUNING_CALIBRATION":
        (
            registry_roundtrip[
                "model_refit_performed"
            ]
            is False
            and
            registry_roundtrip[
                "train_validation_refit_performed"
            ]
            is False
            and
            registry_roundtrip[
                "preprocessing_refit_performed"
            ]
            is False
            and
            registry_roundtrip[
                "threshold_retuning_performed"
            ]
            is False
            and
            registry_roundtrip[
                "calibration_performed"
            ]
            is False
            and
            registry_roundtrip[
                "candidate_expansion_performed"
            ]
            is False
        ),
}

for gate_name, gate_value in (
    m8_03_gates.items()
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
    m8_03_gates
) == 17

assert all(
    m8_03_gates.values()
)

print(
    "\nM8.3 RUNTIME TECHNICAL GATE: PASS"
)

print(
    "\nDecision:",
    "OPEN — REQUIRES AI RUNTIME REVIEW",
)

print(
    "M8.4 metric analysis authorization:",
    "NOT YET — REQUIRES REVIEW",
)

print(
    "\nIMPORTANT:",
    "M8.3 đã tạo official prediction artifacts "
    "nhưng CHƯA tính FINAL TEST performance metrics."
)

```

    G01_SOURCE_LOCATION → PASS
    G02_ONE_SHOT_RERUN_GUARD → PASS
    G03_FROZEN_SUBJECT → PASS
    G04_UPSTREAM_M8_2_RELEASE → PASS
    G05_M8_2_ARTIFACT_FINGERPRINTS → PASS
    G06_FINAL_TEST_INPUT_IDENTITY → PASS
    G07_FEATURE_PREPROCESSING_IDENTITY → PASS
    G08_ESTIMATOR_IDENTITY → PASS
    G09_POSITIVE_CLASS_IDENTITY → PASS
    G10_PRE_INFERENCE_FREEZE_ATTESTATION → PASS
    G11_OFFICIAL_INFERENCE_EXECUTED_ONCE → PASS
    G12_RISK_SCORE_INTEGRITY → PASS
    G13_THRESHOLD_RULE_INTEGRITY → PASS
    G14_OFFICIAL_ARTIFACT_PERSISTENCE → PASS
    G15_OFFICIAL_ARTIFACT_ROUND_TRIP → PASS
    G16_NO_FINAL_TEST_METRIC → PASS
    G17_NO_REFIT_RETUNING_CALIBRATION → PASS
    
    M8.3 RUNTIME TECHNICAL GATE: PASS
    
    Decision: OPEN — REQUIRES AI RUNTIME REVIEW
    M8.4 metric analysis authorization: NOT YET — REQUIRES REVIEW
    
    IMPORTANT: M8.3 đã tạo official prediction artifacts nhưng CHƯA tính FINAL TEST performance metrics.



# 17. Findings / Runtime Review / Decision — sau khi kiểm tra output

## 17.1. Phạm vi review

Review này được thực hiện sau khi notebook M8.3 đã được chạy trên project thật.

Trình tự review:

```text
execution completeness
→ one-shot / rerun guard
→ upstream M8.2 release identity
→ M8.2 artifact fingerprints
→ FINAL TEST input identity
→ feature / preprocessing identity
→ estimator / positive-class identity
→ pre-inference freeze
→ official predict_proba execution
→ threshold-rule integrity
→ persistence / fingerprints
→ round-trip
→ no metric / no refit / no retuning
→ M8.3 decision
```

M8.3 chỉ xác nhận **official frozen inference artifact integrity**.

M8.3 **không đánh giá predictive performance của FINAL TEST**.

---

## 17.2. Execution completeness

Observed:

```text
Code cells executed:
16 / 16

Execution order:
In[1] → In[16]

Runtime traceback:
NONE OBSERVED

Required technical gate:
REACHED
```

Kết luận:

`PASS`

Không có cell runtime bắt buộc nào bị bỏ qua hoặc dừng trước technical gate cuối.

---

## 17.3. One-shot / rerun guard

Observed:

```text
Existing official outputs trước run:
0

ALLOW_EXACT_RERUN:
False

RERUN_REASON:
None
```

Do đó runtime này được xác định là:

`OFFICIAL FIRST INFERENCE RUN`

Không phải technical rerun.

One-shot guard không bị bypass.

Kết luận:

`PASS`

---

## 17.4. Upstream M8.2 release identity

M8.3 nhận upstream:

```text
M8.2 reviewed status:
PASS

M8.2 technical gates:
21 / 21 PASS
```

Persisted M8.2 registry vẫn giữ historical pre-review state:

`OPEN — REQUIRES AI RUNTIME REVIEW`

nhưng reviewed M8.2 notebook sau đó đã khóa:

`M8.2 — PASS`

Reviewed M8.2 notebook SHA256:

```text
d4a378c084cd3c6bc029b821503e2cf13c81535f617931f6209abac0231c41a5
```

Observed M8.3 carry-forward hash khớp reviewed M8.2 notebook.

Kết luận:

`PASS`

Không phát hiện upstream release contradiction.

---

## 17.5. M8.2 artifact fingerprint re-verification

Trước official inference, M8.3 re-verify fingerprints của audited M8.2 artifacts.

Verified source artifacts gồm:

```text
X_final_test_w_short.npz
y_final_test.npy
row_id_final_test.npy
timestamp_final_test.npy
m8_02_w_short_preprocessing_state.json
m8_02_selected_rf_estimator.joblib
```

Observed:

`ALL REQUIRED M8.2 GENERATED ARTIFACT FINGERPRINTS MATCH`

Kết luận:

`PASS`

Official inference không dùng một bản artifact khác với artifact đã được M8.2 audit.

---

## 17.6. FINAL TEST input identity

Observed:

```text
FINAL TEST matrix:
(722955, 47)

dtype:
float32

Sparse format:
CSR

nnz:
6,526,714

Target rows:
722,955

Target fraud support:
1,035

Lineage rows:
722,955
```

Integrity checks:

```text
finite matrix values:
PASS

unique row_id:
PASS

expected feature width:
PASS

expected population size:
PASS
```

Kết luận:

`PASS`

---

## 17.7. Feature / preprocessing identity

Observed:

```text
Preprocessing strategy:
W_SHORT

Fit source:
W_SHORT_TRAIN_ONLY

Feature count:
47

Matrix format:
CSR

Matrix dtype:
float32

VALIDATION exact reproduction flag:
True
```

Canonical feature-name sequence khớp persisted preprocessing state.

Không có learned preprocessing state nào được fit/update trong M8.3.

Kết luận:

`PASS`

---

## 17.8. Estimator / positive-class identity

Observed estimator:

```text
RandomForestClassifier
```

Frozen subject:

```text
Training Window:
W_SHORT

Config:
RF-REF-100-GINI-SQRT-UNPRUNED-CW

Imbalance:
CLASS_WEIGHT_BALANCED

random_state:
42
```

Estimator classes:

```text
[0, 1]
```

Positive class:

`1`

Positive-class probability index:

`1`

Exact parameter subset được re-verify với frozen M8.2 subject.

Kết luận:

`PASS`

Không phát hiện model/config/class-weight/seed/positive-class mismatch.

---

## 17.9. Pre-inference freeze attestation

Observed:

```text
no_model_refit:
True

no_train_validation_refit:
True

no_preprocessing_refit:
True

no_threshold_retuning:
True

no_calibration:
True

no_candidate_expansion:
True

no_feature_change:
True

no_seed_change:
True

no_final_test_metric_before_inference:
True
```

Source review cũng không phát hiện `.fit(...)` trong official M8.3 path.

Kết luận:

`PASS`

Frozen subject không bị thay đổi trước official FINAL TEST access.

---

# 18. Official FINAL TEST inference review

## 18.1. Official model access

Observed:

```text
Official risk-score rows:
722,955

Official prediction rows:
722,955

predict_proba call count:
1
```

Source review xác nhận:

```text
predict_proba(X_final_test):
EXACTLY ONE OFFICIAL CALL

predict(X_final_test):
NONE

fit(...):
NONE
```

Kết luận:

`PASS — EXACTLY ONE OFFICIAL INFERENCE CALL`

Official inference đã được thực hiện đúng theo frozen protocol.

---

## 18.2. Risk-score integrity

Observed:

```text
risk_score shape:
(722955,)

risk_score dtype:
float32

finite:
PASS

range:
0.0 <= risk_score <= 1.0
```

Risk score được lấy từ:

`predict_proba / positive class = 1`

Không claim đây là calibrated confidence/probability.

Kết luận:

`PASS`

---

## 18.3. Frozen threshold rule

Frozen rule:

```text
risk_score > 0.50
```

Observed:

```text
Prediction dtype:
int8

Threshold reconstruction mismatch:
0
```

Binary prediction artifact được tạo trực tiếp từ frozen comparator.

Không thử:

```text
>= 0.50
alternate threshold
adaptive threshold
calibration
```

Kết luận:

`PASS — EXACT THRESHOLD RULE`

---

## 18.4. FINAL TEST performance separation

Source review không phát hiện các metric call:

```text
f1_score
recall_score
precision_score
confusion_matrix
sklearn.metrics
```

M8.3 không compute target-dependent performance.

Observed registry state:

```text
final_test_metric_computed:
False
```

Kết luận:

`PASS`

Separation được giữ đúng:

```text
M8.3:
official prediction artifacts

→

M8.4:
final metric + confusion matrix

→

M8.5:
error / temporal-generalization interpretation
```

---

# 19. Official artifact persistence review

## 19.1. Persisted artifacts

M8.3 đã persist:

```text
risk_score_final_test.npy
y_pred_final_test.npy
y_final_test.npy
row_id_final_test.npy
timestamp_final_test.npy
m8_03_final_inference_registry.json
m8_03_inference_manifest.json
m8_03_official_inference.lock.json
```

Target / lineage artifacts được copy từ audited M8.2 source.

Risk score và prediction là official M8.3 outputs.

Kết luận:

`PASS`

---

## 19.2. Fingerprints

Source và generated artifact fingerprints được persist trong registry/manifest.

Byte-identity assertions cho:

```text
y_final_test.npy
row_id_final_test.npy
timestamp_final_test.npy
```

khớp M8.2 source artifacts.

Registry và manifest cũng có fingerprint.

Kết luận:

`PASS`

---

## 19.3. Round-trip

Observed:

```text
Official risk-score round-trip rows:
722,955

Official prediction round-trip rows:
722,955

Registry round-trip:
PASS

Manifest round-trip:
PASS
```

Các arrays sau reload khớp arrays trong memory.

Fingerprint re-check sau persistence:

`PASS`

Kết luận:

`PASS`

---

# 20. Technical gate summary

Observed:

```text
G01_SOURCE_LOCATION:
PASS

G02_ONE_SHOT_RERUN_GUARD:
PASS

G03_FROZEN_SUBJECT:
PASS

G04_UPSTREAM_M8_2_RELEASE:
PASS

G05_M8_2_ARTIFACT_FINGERPRINTS:
PASS

G06_FINAL_TEST_INPUT_IDENTITY:
PASS

G07_FEATURE_PREPROCESSING_IDENTITY:
PASS

G08_ESTIMATOR_IDENTITY:
PASS

G09_POSITIVE_CLASS_IDENTITY:
PASS

G10_PRE_INFERENCE_FREEZE_ATTESTATION:
PASS

G11_OFFICIAL_INFERENCE_EXECUTED_ONCE:
PASS

G12_RISK_SCORE_INTEGRITY:
PASS

G13_THRESHOLD_RULE_INTEGRITY:
PASS

G14_OFFICIAL_ARTIFACT_PERSISTENCE:
PASS

G15_OFFICIAL_ARTIFACT_ROUND_TRIP:
PASS

G16_NO_FINAL_TEST_METRIC:
PASS

G17_NO_REFIT_RETUNING_CALIBRATION:
PASS
```

Technical gate result:

`17 / 17 PASS`

---

# 21. Blocking-condition review

Observed:

```text
upstream M8.2 mismatch:
NONE

artifact fingerprint mismatch:
NONE

FINAL TEST input identity mismatch:
NONE

feature/preprocessing mismatch:
NONE

estimator identity mismatch:
NONE

positive-class mismatch:
NONE

unauthorized refit:
NONE

threshold retuning:
NONE

calibration:
NONE

candidate expansion:
NONE

multiple official inference calls:
NONE

prediction artifact corruption:
NONE

persistence failure:
NONE

round-trip failure:
NONE

FINAL TEST metric computed in M8.3:
NONE
```

Blocking issue:

`NONE OBSERVED`

---

# 22. M8.3 Decision

Sau runtime review:

```text
M8.3:
PASS

Work type:
OFFICIAL FROZEN FINAL TEST INFERENCE

Upstream M8.2:
PASS

Official inference run:
FIRST / ONE-SHOT

predict_proba FINAL TEST call count:
1

FINAL TEST rows:
722,955

Risk-score artifact:
PERSISTED / VERIFIED

Prediction artifact:
PERSISTED / VERIFIED

Threshold:
0.50

Comparator:
risk_score > 0.50

Threshold reconstruction mismatch:
0

Model refit:
NONE

TRAIN+VALIDATION refit:
NONE

Preprocessing refit:
NONE

Threshold retuning:
NONE

Calibration:
NONE

Candidate expansion:
NONE

FINAL TEST metric in M8.3:
NONE

Artifact fingerprints:
PASS

Artifact round-trip:
PASS

Technical gates:
17 / 17 PASS

Blocking issue:
NONE
```

## Decision

`M8.3 — PASS`

## Official prediction state

`FROZEN / PERSISTED / VERIFIED`

## FINAL TEST performance status

`NOT YET INTERPRETED`

M8.3 không kết luận model tốt/xấu.

M8.3 chỉ xác nhận official prediction artifacts đã được tạo đúng bằng frozen subject.

---

# 23. Authorization sang M8.4

Precondition:

`M8.3 — PASS`

M8.4 status:

`AUTHORIZED`

Authorized next step:

`M8.4 — Final Metric + Confusion Matrix Reconstruction`

M8.4 được phép:

```text
load official M8.3 artifacts
verify fingerprints / lineage
use y_final_test + y_pred_final_test
reconstruct confusion matrix
compute F1_fraud
compute Recall_fraud
compute Precision_fraud
compute Accuracy as reference only
compute predicted-positive count/rate
persist metric/confusion artifacts
fingerprint
round-trip
```

M8.4 không được:

```text
change model
change training population
refit estimator
change feature/preprocessing
change threshold
try alternate threshold
calibrate probability
use metric result để quay lại thay frozen subject
```

M8.4 là bước **measurement**, không phải model selection.

---

# 24. Handoff sang M8.4

Expected M8.3 source root:

```text
data/processed/m8_03_final_test_inference/
```

Required source artifacts:

```text
risk_score_final_test.npy
y_pred_final_test.npy
y_final_test.npy
row_id_final_test.npy
timestamp_final_test.npy
m8_03_final_inference_registry.json
m8_03_inference_manifest.json
m8_03_official_inference.lock.json
```

Official FINAL TEST inference:

`COMPLETE`

Official prediction artifacts:

`FROZEN`

M8.4 metric reconstruction:

`AUTHORIZED`

Performance interpretation vẫn phải giữ đúng thứ tự:

```text
OBSERVED OFFICIAL ARTIFACT
→ METRIC FACT
→ INTERPRETATION
→ FINAL LIMITATION
```

