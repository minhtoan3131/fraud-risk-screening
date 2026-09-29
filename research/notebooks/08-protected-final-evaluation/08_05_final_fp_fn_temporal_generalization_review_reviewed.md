# M8.5 — Final FP/FN + Temporal Generalization Review

**Project:** AI Transaction Fraud Risk Screening  
**Milestone:** M8 — Protected Final Evaluation  
**Substep:** M8.5 — Final FP/FN + Temporal Generalization Review  
**Work type:** Descriptive FINAL TEST error analysis + temporal generalization review  
**Upstream:** M8.4 — PASS / canonical FINAL TEST metric profile locked  
**Decision trước runtime:** `OPEN — REQUIRES RUNTIME REVIEW`

---

## Vai trò của notebook

M8.5 chỉ bắt đầu sau khi:

```text
official prediction artifacts:
FROZEN

official FINAL TEST metrics:
LOCKED
```

Các câu hỏi được phép:

```text
fraud bị bỏ sót có pattern gì?
FP có concentration đáng chú ý không?
risk score của FN/FP phân bố thế nào?
monthly FINAL TEST behavior thay đổi ra sao?
VALIDATION → FINAL TEST delta theo F1 / Recall / Precision / alerts ra sao?
temporal generalization limitation có bằng chứng gì?
```

Guardrail bắt buộc:

```text
ANALYZE — DO NOT FIX USING FINAL TEST
```

M8.5 **không**:

- gọi estimator để tạo prediction mới;
- refit model;
- refit preprocessing;
- thay feature;
- thay threshold;
- thử threshold khác;
- calibration;
- candidate expansion;
- dùng FP/FN hoặc monthly result để quay lại tối ưu frozen subject;
- biến descriptive pattern thành causal claim;
- suy rộng trực tiếp sang ngân hàng thật / production.

M8.5 chỉ tạo evidence mô tả để khóa limitation và handoff sang M8.6.



## 1. Nguồn evidence

M8.5 dùng ba lớp artifact đã được khóa trước đó:

```text
M8.2:
audited FINAL TEST representation + preprocessing state

M8.3:
official FINAL TEST risk_score / y_pred / target / lineage / timestamps

M8.4:
canonical FINAL TEST metrics + confusion matrix
```

Để so sánh temporal generalization, M8.5 dùng thêm:

```text
M7.7:
selected RF official VALIDATION y_pred + risk_score

M4.7:
canonical y_validation
```

Không chạy model lại trên VALIDATION hoặc FINAL TEST.



```python

from pathlib import Path
import hashlib
import json
import platform
import sys

import numpy as np

try:
    import scipy
    from scipy import sparse
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Thiếu scipy. Hãy cài dependency của project rồi Restart Kernel."
    ) from exc

print("Python:", sys.version)
print("Executable:", sys.executable)
print("Platform:", platform.platform())
print("NumPy:", np.__version__)
print("SciPy:", scipy.__version__)

```

    Python: 3.14.6 (main, Jun 10 2026, 10:03:53) [Clang 21.0.0 (clang-2100.0.123.102)]
    Executable: /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/.venv/bin/python
    Platform: macOS-26.6.2-arm64-arm-64bit-Mach-O
    NumPy: 2.5.3
    SciPy: 1.18.1


## 2. Locate canonical upstream artifacts


```python

M4_REL = (
    Path("data")
    / "processed"
    / "m4_07_baseline_ready"
)

M7_07_REL = (
    Path("data")
    / "processed"
    / "m7_07_candidate_selection_external_validation"
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

M8_04_REL = (
    Path("data")
    / "processed"
    / "m8_04_final_metric_analysis"
)

M8_05_REL = (
    Path("data")
    / "processed"
    / "m8_05_final_error_analysis"
)

SELECTED_CANDIDATE_ID = (
    "RF-REF-100-GINI-SQRT-UNPRUNED-CW"
)

m7_pred_name = (
    "m7_07__"
    + SELECTED_CANDIDATE_ID
    + "__validation_y_pred.npy"
)

m7_score_name = (
    "m7_07__"
    + SELECTED_CANDIDATE_ID
    + "__validation_risk_score.npy"
)

required_rel_paths = [
    M4_REL / "y_validation.npy",
    M4_REL / "feature_names.json",
    M7_07_REL / m7_pred_name,
    M7_07_REL / m7_score_name,
    M7_07_REL / "m7_07_external_validation_registry.json",
    M7_07_REL / "m7_07_candidate_selection_manifest.json",
    M8_02_REL / "X_final_test_w_short.npz",
    M8_02_REL / "m8_02_w_short_preprocessing_state.json",
    M8_03_REL / "risk_score_final_test.npy",
    M8_03_REL / "y_pred_final_test.npy",
    M8_03_REL / "y_final_test.npy",
    M8_03_REL / "row_id_final_test.npy",
    M8_03_REL / "timestamp_final_test.npy",
    M8_03_REL / "m8_03_final_inference_registry.json",
    M8_03_REL / "m8_03_inference_manifest.json",
    M8_04_REL / "m8_04_final_metric_confusion_analysis.json",
    M8_04_REL / "m8_04_analysis_manifest.json",
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
        ]

    raise FileNotFoundError(
        "Không tìm thấy PROJECT_ROOT chứa đầy đủ M4.7 / M7.7 / M8.2–M8.4 artifacts.\n"
        + json.dumps(
            missing_by_candidate,
            ensure_ascii=False,
            indent=2,
        )
    )

M4_DIR = PROJECT_ROOT / M4_REL
M7_07_DIR = PROJECT_ROOT / M7_07_REL
M8_02_DIR = PROJECT_ROOT / M8_02_REL
M8_03_DIR = PROJECT_ROOT / M8_03_REL
M8_04_DIR = PROJECT_ROOT / M8_04_REL
OUTPUT_DIR = PROJECT_ROOT / M8_05_REL

Y_VALIDATION_PATH = (
    M4_DIR
    / "y_validation.npy"
)

FEATURE_NAMES_PATH = (
    M4_DIR
    / "feature_names.json"
)

M7_VALIDATION_PRED_PATH = (
    M7_07_DIR
    / m7_pred_name
)

M7_VALIDATION_SCORE_PATH = (
    M7_07_DIR
    / m7_score_name
)

M7_07_REGISTRY_PATH = (
    M7_07_DIR
    / "m7_07_external_validation_registry.json"
)

M7_07_MANIFEST_PATH = (
    M7_07_DIR
    / "m7_07_candidate_selection_manifest.json"
)

X_FINAL_PATH = (
    M8_02_DIR
    / "X_final_test_w_short.npz"
)

PREPROCESSING_STATE_PATH = (
    M8_02_DIR
    / "m8_02_w_short_preprocessing_state.json"
)

RISK_SCORE_FINAL_PATH = (
    M8_03_DIR
    / "risk_score_final_test.npy"
)

Y_PRED_FINAL_PATH = (
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

M8_04_RESULT_PATH = (
    M8_04_DIR
    / "m8_04_final_metric_confusion_analysis.json"
)

M8_04_MANIFEST_PATH = (
    M8_04_DIR
    / "m8_04_analysis_manifest.json"
)

print("PROJECT_ROOT:", PROJECT_ROOT)
print("M8.5 output:", OUTPUT_DIR)
print("\nM8.5 SOURCE LOCATION GATE: PASS")

```

    PROJECT_ROOT: /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening
    M8.5 output: /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/processed/m8_05_final_error_analysis
    
    M8.5 SOURCE LOCATION GATE: PASS


## 3. Output guard


```python

ALLOW_EXACT_RERUN = False
RERUN_REASON = None

RESULT_PATH = (
    OUTPUT_DIR
    / "m8_05_final_error_analysis.json"
)

MANIFEST_PATH = (
    OUTPUT_DIR
    / "m8_05_analysis_manifest.json"
)

ROW_GROUP_PATH = (
    OUTPUT_DIR
    / "m8_05_error_group_row_ids.npz"
)

existing_outputs = [
    path
    for path
    in [
        RESULT_PATH,
        MANIFEST_PATH,
        ROW_GROUP_PATH,
    ]
    if path.exists()
]

if existing_outputs:
    if not ALLOW_EXACT_RERUN:
        raise RuntimeError(
            "M8.5 canonical output đã tồn tại. "
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
    "Existing M8.5 outputs:",
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
    "\nM8.5 OUTPUT GUARD: PASS"
)

```

    Existing M8.5 outputs: 0
    Exact rerun enabled: False
    Rerun reason: None
    
    M8.5 OUTPUT GUARD: PASS


## 4. Frozen analysis contract + reviewed M8.4 release


```python

EXPECTED_FINAL_ROWS = 722_955
EXPECTED_FINAL_FRAUD = 1_035

EXPECTED_VALIDATION_ROWS = 712_458
EXPECTED_VALIDATION_FRAUD = 1_052

FINAL_THRESHOLD = 0.50
THRESHOLD_COMPARATOR = ">"

# Reviewed M8.4 canonical confusion identity.
EXPECTED_FINAL_TP = 393
EXPECTED_FINAL_FP = 782
EXPECTED_FINAL_FN = 642
EXPECTED_FINAL_TN = 721_138

# Reviewed M7.7 selected-RF VALIDATION identity.
EXPECTED_VALIDATION_TP = 467
EXPECTED_VALIDATION_FP = 764
EXPECTED_VALIDATION_FN = 585
EXPECTED_VALIDATION_TN = 710_642

EXPECTED_FINAL_MONTHS = [
    "2019-06",
    "2019-07",
    "2019-08",
    "2019-09",
    "2019-10",
]

M8_05_ANALYSIS_VERSION = (
    "M8.5-final-fp-fn-temporal-review-v1"
)

UPSTREAM_REVIEWED_M8_04_STATUS = "PASS"

UPSTREAM_REVIEWED_M8_04_NOTEBOOK_SHA256 = (
    "8882b37791a5b540d4e3fb5002b746ac"
    "c3092637077fb6cbf103cb2659e989cd"
)

ANALYZE_DO_NOT_FIX_USING_FINAL_TEST = True
NO_MODEL_CALL = True
NO_NEW_PREDICTION = True
NO_REFIT = True
NO_RETUNING = True
NO_THRESHOLD_CHANGE = True
NO_CALIBRATION = True
NO_CANDIDATE_EXPANSION = True

assert (
    UPSTREAM_REVIEWED_M8_04_STATUS
    ==
    "PASS"
)

assert FINAL_THRESHOLD == 0.50
assert THRESHOLD_COMPARATOR == ">"

assert all([
    ANALYZE_DO_NOT_FIX_USING_FINAL_TEST,
    NO_MODEL_CALL,
    NO_NEW_PREDICTION,
    NO_REFIT,
    NO_RETUNING,
    NO_THRESHOLD_CHANGE,
    NO_CALIBRATION,
    NO_CANDIDATE_EXPANSION,
])

print(
    "Reviewed M8.4:",
    UPSTREAM_REVIEWED_M8_04_STATUS,
)
print(
    "Guardrail:",
    "ANALYZE — DO NOT FIX USING FINAL TEST",
)
print(
    "\nM8.5 FROZEN ANALYSIS CONTRACT GATE: PASS"
)

```

    Reviewed M8.4: PASS
    Guardrail: ANALYZE — DO NOT FIX USING FINAL TEST
    
    M8.5 FROZEN ANALYSIS CONTRACT GATE: PASS


## 5. Load canonical upstream registries / manifests và verify fingerprints


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


def load_json(path):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(
            file
        )


m7_07_registry = load_json(
    M7_07_REGISTRY_PATH
)

m7_07_manifest = load_json(
    M7_07_MANIFEST_PATH
)

m8_03_registry = load_json(
    M8_03_REGISTRY_PATH
)

m8_03_manifest = load_json(
    M8_03_MANIFEST_PATH
)

m8_04_result = load_json(
    M8_04_RESULT_PATH
)

m8_04_manifest = load_json(
    M8_04_MANIFEST_PATH
)

preprocessing_state = load_json(
    PREPROCESSING_STATE_PATH
)

canonical_feature_names = load_json(
    FEATURE_NAMES_PATH
)

assert (
    m7_07_registry[
        "analysis_version"
    ]
    ==
    "M7.7-candidate-selection-external-validation-v1"
)

assert (
    m7_07_manifest[
        "final_test_accessed"
    ]
    is False
)

assert (
    m8_03_registry[
        "official_final_inference_executed"
    ]
    is True
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
    m8_03_registry[
        "final_test_metric_computed"
    ]
    is False
)

assert (
    m8_04_result[
        "analysis_version"
    ]
    ==
    "M8.4-final-metric-confusion-reconstruction-v1"
)

assert str(
    m8_04_result[
        "decision"
    ]
).startswith(
    "OPEN"
)

assert (
    m8_04_result[
        "guardrails"
    ][
        "new_prediction_generated"
    ]
    is False
)

assert (
    m8_04_result[
        "guardrails"
    ][
        "no_retroactive_optimization_attestation"
    ]
    is True
)

assert (
    m8_04_manifest[
        "metric_result_sha256"
    ]
    ==
    sha256_file(
        M8_04_RESULT_PATH
    )
)

# Re-verify official M8.3 generated artifacts through its persisted fingerprints.
m8_03_fp = (
    m8_03_registry[
        "generated_fingerprints"
    ]
)

m8_03_fp_path_map = {
    "risk_score_final_test_sha256":
        RISK_SCORE_FINAL_PATH,
    "y_pred_final_test_sha256":
        Y_PRED_FINAL_PATH,
    "y_final_test_sha256":
        Y_FINAL_PATH,
    "row_id_final_test_sha256":
        ROW_FINAL_PATH,
    "timestamp_final_test_sha256":
        TIMESTAMP_FINAL_PATH,
}

for key, path in (
    m8_03_fp_path_map.items()
):
    assert (
        sha256_file(
            path
        )
        ==
        m8_03_fp[
            key
        ]
    )

# X_final / preprocessing identity được M8.3 carry từ M8.2.
assert (
    sha256_file(
        X_FINAL_PATH
    )
    ==
    m8_03_registry[
        "source_fingerprints"
    ][
        "m8_02_x_final_test_sha256"
    ]
)

assert (
    sha256_file(
        PREPROCESSING_STATE_PATH
    )
    ==
    m8_03_registry[
        "source_fingerprints"
    ][
        "m8_02_preprocessing_state_sha256"
    ]
)

print(
    "Official M8.3 fingerprints:",
    "VERIFIED",
)
print(
    "Canonical M8.4 metric artifact:",
    "VERIFIED",
)
print(
    "Audited M8.2 semantic representation:",
    "VERIFIED",
)
print(
    "\nM8.5 UPSTREAM ARTIFACT INTEGRITY GATE: PASS"
)

```

    Official M8.3 fingerprints: VERIFIED
    Canonical M8.4 metric artifact: VERIFIED
    Audited M8.2 semantic representation: VERIFIED
    
    M8.5 UPSTREAM ARTIFACT INTEGRITY GATE: PASS


## 6. Load official FINAL TEST arrays và reconstruct error groups


```python

risk_score_final = np.load(
    RISK_SCORE_FINAL_PATH,
    allow_pickle=False,
)

y_pred_final = np.load(
    Y_PRED_FINAL_PATH,
    allow_pickle=False,
)

y_final = np.load(
    Y_FINAL_PATH,
    allow_pickle=False,
)

row_id_final = np.load(
    ROW_FINAL_PATH,
    allow_pickle=False,
)

timestamp_final = np.load(
    TIMESTAMP_FINAL_PATH,
    allow_pickle=False,
)

X_final = sparse.load_npz(
    X_FINAL_PATH
)

expected_shape = (
    EXPECTED_FINAL_ROWS,
)

for array in [
    risk_score_final,
    y_pred_final,
    y_final,
    row_id_final,
    timestamp_final,
]:
    assert (
        array.shape
        ==
        expected_shape
    )

assert (
    X_final.shape
    ==
    (
        EXPECTED_FINAL_ROWS,
        47,
    )
)

assert sparse.isspmatrix_csr(
    X_final
)

assert (
    X_final.dtype
    ==
    np.float32
)

assert (
    int(
        y_final.sum()
    )
    ==
    EXPECTED_FINAL_FRAUD
)

tp_mask = (
    (y_final == 1)
    &
    (y_pred_final == 1)
)

fp_mask = (
    (y_final == 0)
    &
    (y_pred_final == 1)
)

fn_mask = (
    (y_final == 1)
    &
    (y_pred_final == 0)
)

tn_mask = (
    (y_final == 0)
    &
    (y_pred_final == 0)
)

tp = int(
    np.count_nonzero(
        tp_mask
    )
)

fp = int(
    np.count_nonzero(
        fp_mask
    )
)

fn = int(
    np.count_nonzero(
        fn_mask
    )
)

tn = int(
    np.count_nonzero(
        tn_mask
    )
)

assert (
    tp,
    fp,
    fn,
    tn,
) == (
    EXPECTED_FINAL_TP,
    EXPECTED_FINAL_FP,
    EXPECTED_FINAL_FN,
    EXPECTED_FINAL_TN,
)

assert (
    tp
    + fp
    + fn
    + tn
    ==
    EXPECTED_FINAL_ROWS
)

threshold_reconstructed = (
    risk_score_final
    >
    FINAL_THRESHOLD
).astype(
    np.int8,
    copy=False,
)

assert (
    np.count_nonzero(
        threshold_reconstructed
        !=
        y_pred_final
    )
    ==
    0
)

print("TP:", tp)
print("FP:", fp)
print("FN:", fn)
print("TN:", tn)
print(
    "\nM8.5 FINAL ERROR-GROUP IDENTITY GATE: PASS"
)

```

    TP: 393
    FP: 782
    FN: 642
    TN: 721138
    
    M8.5 FINAL ERROR-GROUP IDENTITY GATE: PASS


## 7. Persist row-level error lineage


```python

tp_row_ids = row_id_final[
    tp_mask
].astype(
    np.int64,
    copy=False,
)

fp_row_ids = row_id_final[
    fp_mask
].astype(
    np.int64,
    copy=False,
)

fn_row_ids = row_id_final[
    fn_mask
].astype(
    np.int64,
    copy=False,
)

tn_row_ids = row_id_final[
    tn_mask
].astype(
    np.int64,
    copy=False,
)

assert len(
    tp_row_ids
) == tp

assert len(
    fp_row_ids
) == fp

assert len(
    fn_row_ids
) == fn

assert len(
    tn_row_ids
) == tn

all_group_rows = np.concatenate([
    tp_row_ids,
    fp_row_ids,
    fn_row_ids,
    tn_row_ids,
])

assert (
    len(
        all_group_rows
    )
    ==
    EXPECTED_FINAL_ROWS
)

assert (
    len(
        np.unique(
            all_group_rows
        )
    )
    ==
    EXPECTED_FINAL_ROWS
)

np.savez_compressed(
    ROW_GROUP_PATH,
    tp_row_id=tp_row_ids,
    fp_row_id=fp_row_ids,
    fn_row_id=fn_row_ids,
    tn_row_id=tn_row_ids,
)

assert ROW_GROUP_PATH.exists()

print(
    "TP row ids:",
    len(
        tp_row_ids
    ),
)
print(
    "FP row ids:",
    len(
        fp_row_ids
    ),
)
print(
    "FN row ids:",
    len(
        fn_row_ids
    ),
)
print(
    "TN row ids:",
    len(
        tn_row_ids
    ),
)
print(
    "\nM8.5 ERROR ROW-LINEAGE GATE: PASS"
)

```

    TP row ids: 393
    FP row ids: 782
    FN row ids: 642
    TN row ids: 721138
    
    M8.5 ERROR ROW-LINEAGE GATE: PASS



## 8. Risk-score distribution theo TP / FP / FN / TN

Đây là descriptive score analysis.

Risk score **không** được gọi là calibrated confidence.



```python

def quantile_summary(
    values,
):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if values.size == 0:
        return {
            "count": 0,
        }

    return {
        "count":
            int(
                values.size
            ),
        "min":
            float(
                np.min(
                    values
                )
            ),
        "p10":
            float(
                np.quantile(
                    values,
                    0.10,
                )
            ),
        "p25":
            float(
                np.quantile(
                    values,
                    0.25,
                )
            ),
        "median":
            float(
                np.quantile(
                    values,
                    0.50,
                )
            ),
        "p75":
            float(
                np.quantile(
                    values,
                    0.75,
                )
            ),
        "p90":
            float(
                np.quantile(
                    values,
                    0.90,
                )
            ),
        "max":
            float(
                np.max(
                    values
                )
            ),
        "mean":
            float(
                np.mean(
                    values
                )
            ),
    }


score_summary = {
    "TP":
        quantile_summary(
            risk_score_final[
                tp_mask
            ]
        ),
    "FP":
        quantile_summary(
            risk_score_final[
                fp_mask
            ]
        ),
    "FN":
        quantile_summary(
            risk_score_final[
                fn_mask
            ]
        ),
    "TN":
        quantile_summary(
            risk_score_final[
                tn_mask
            ]
        ),
}

score_bins = np.array([
    0.0,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0000001,
], dtype=np.float64)

score_band_labels = [
    "[0.0,0.1)",
    "[0.1,0.2)",
    "[0.2,0.3)",
    "[0.3,0.4)",
    "[0.4,0.5)",
    "[0.5,0.6)",
    "[0.6,0.7)",
    "[0.7,0.8)",
    "[0.8,0.9)",
    "[0.9,1.0]",
]

score_band_counts = {}

for group_name, mask in {
    "TP": tp_mask,
    "FP": fp_mask,
    "FN": fn_mask,
    "TN": tn_mask,
}.items():
    counts, _ = np.histogram(
        risk_score_final[
            mask
        ].astype(
            np.float64
        ),
        bins=score_bins,
    )

    score_band_counts[
        group_name
    ] = {
        label:
            int(
                count
            )
        for label, count
        in zip(
            score_band_labels,
            counts,
        )
    }

boundary_exact_050 = {
    "fraud_score_eq_0_50":
        int(
            np.count_nonzero(
                (y_final == 1)
                &
                (
                    risk_score_final
                    ==
                    np.float32(
                        0.50
                    )
                )
            )
        ),
    "nonfraud_score_eq_0_50":
        int(
            np.count_nonzero(
                (y_final == 0)
                &
                (
                    risk_score_final
                    ==
                    np.float32(
                        0.50
                    )
                )
            )
        ),
}

print(
    json.dumps(
        {
            "quantiles":
                score_summary,
            "bands":
                score_band_counts,
            "boundary_exact_0_50":
                boundary_exact_050,
        },
        ensure_ascii=False,
        indent=2,
    )
)

print(
    "\nM8.5 RISK-SCORE ERROR PROFILE GATE: PASS"
)

```

    {
      "quantiles": {
        "TP": {
          "count": 393,
          "min": 0.5099999904632568,
          "p10": 0.6000000238418579,
          "p25": 0.699999988079071,
          "median": 0.8299999833106995,
          "p75": 0.9100000262260437,
          "p90": 0.9599999785423279,
          "max": 1.0,
          "mean": 0.7981424936811432
        },
        "FP": {
          "count": 782,
          "min": 0.5099999904632568,
          "p10": 0.550000011920929,
          "p25": 0.6100000143051147,
          "median": 0.7200000286102295,
          "p75": 0.8299999833106995,
          "p90": 0.9100000262260437,
          "max": 1.0,
          "mean": 0.7247058819322025
        },
        "FN": {
          "count": 642,
          "min": 0.0,
          "p10": 0.0,
          "p25": 0.0,
          "median": 0.009999999776482582,
          "p75": 0.03999999910593033,
          "p90": 0.24899999946355797,
          "max": 0.5,
          "mean": 0.06260124607899478
        },
        "TN": {
          "count": 721138,
          "min": 0.0,
          "p10": 0.0,
          "p25": 0.0,
          "median": 0.0,
          "p75": 0.0,
          "p90": 0.0,
          "max": 0.5,
          "mean": 0.0005939917160870857
        }
      },
      "bands": {
        "TP": {
          "[0.0,0.1)": 0,
          "[0.1,0.2)": 0,
          "[0.2,0.3)": 0,
          "[0.3,0.4)": 0,
          "[0.4,0.5)": 0,
          "[0.5,0.6)": 39,
          "[0.6,0.7)": 68,
          "[0.7,0.8)": 58,
          "[0.8,0.9)": 117,
          "[0.9,1.0]": 111
        },
        "FP": {
          "[0.0,0.1)": 0,
          "[0.1,0.2)": 0,
          "[0.2,0.3)": 0,
          "[0.3,0.4)": 0,
          "[0.4,0.5)": 0,
          "[0.5,0.6)": 169,
          "[0.6,0.7)": 196,
          "[0.7,0.8)": 158,
          "[0.8,0.9)": 171,
          "[0.9,1.0]": 88
        },
        "FN": {
          "[0.0,0.1)": 538,
          "[0.1,0.2)": 30,
          "[0.2,0.3)": 18,
          "[0.3,0.4)": 23,
          "[0.4,0.5)": 28,
          "[0.5,0.6)": 5,
          "[0.6,0.7)": 0,
          "[0.7,0.8)": 0,
          "[0.8,0.9)": 0,
          "[0.9,1.0]": 0
        },
        "TN": {
          "[0.0,0.1)": 719924,
          "[0.1,0.2)": 489,
          "[0.2,0.3)": 263,
          "[0.3,0.4)": 241,
          "[0.4,0.5)": 198,
          "[0.5,0.6)": 23,
          "[0.6,0.7)": 0,
          "[0.7,0.8)": 0,
          "[0.8,0.9)": 0,
          "[0.9,1.0]": 0
        }
      },
      "boundary_exact_0_50": {
        "fraud_score_eq_0_50": 5,
        "nonfraud_score_eq_0_50": 23
      }
    }
    
    M8.5 RISK-SCORE ERROR PROFILE GATE: PASS



## 9. Khôi phục các semantic features có thể diễn giải từ audited 47-column matrix

M8.5 không rebuild raw pipeline.

Các giá trị dưới đây được khôi phục từ **chính matrix FINAL TEST đã audit ở M8.2** và frozen preprocessing state.

Phạm vi semantic reconstruction:

```text
amount_numeric
time_since_previous_transaction_min
transactions_last_1h
amount_minus_previous_mean
is_new_merchant
has_prior_card_history
transaction_mode
location_state
```

Structural missing của hai history-dependent numeric feature được khôi phục về `NaN` khi `has_prior_card_history = 0`.



```python

assert (
    preprocessing_state[
        "feature_count"
    ]
    ==
    47
)

assert (
    preprocessing_state[
        "feature_names"
    ]
    ==
    canonical_feature_names
)

feature_index = {
    name: idx
    for idx, name
    in enumerate(
        canonical_feature_names
    )
}

numeric_columns = (
    preprocessing_state[
        "numeric_columns"
    ]
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

assert numeric_columns == [
    "amount_numeric",
    "time_since_previous_transaction_min",
    "transactions_last_1h",
    "amount_minus_previous_mean",
]

def dense_column(
    feature_name,
):
    idx = feature_index[
        feature_name
    ]

    return (
        X_final[
            :,
            idx,
        ]
        .toarray()
        .ravel()
        .astype(
            np.float64,
            copy=False,
        )
    )


numeric_semantic = {}

for j, column in enumerate(
    numeric_columns
):
    scaled = dense_column(
        "num__"
        + column
    )

    semantic = (
        scaled
        * numeric_scale[j]
        + numeric_mean[j]
    )

    numeric_semantic[
        column
    ] = semantic


is_new_merchant = (
    dense_column(
        "bool__is_new_merchant"
    )
    >
    0.5
)

has_prior_card_history = (
    dense_column(
        "bool__has_prior_card_history"
    )
    >
    0.5
)

numeric_semantic[
    "time_since_previous_transaction_min"
] = (
    numeric_semantic[
        "time_since_previous_transaction_min"
    ].copy()
)

numeric_semantic[
    "amount_minus_previous_mean"
] = (
    numeric_semantic[
        "amount_minus_previous_mean"
    ].copy()
)

numeric_semantic[
    "time_since_previous_transaction_min"
][
    ~has_prior_card_history
] = np.nan

numeric_semantic[
    "amount_minus_previous_mean"
][
    ~has_prior_card_history
] = np.nan


def decode_categorical(
    column_name,
):
    prefix = (
        "cat__"
        + column_name
        + "_"
    )

    pairs = [
        (
            idx,
            name[
                len(
                    prefix
                ):
            ],
        )
        for idx, name
        in enumerate(
            canonical_feature_names
        )
        if name.startswith(
            prefix
        )
    ]

    assert pairs

    indices = [
        idx
        for idx, _
        in pairs
    ]

    labels = np.asarray([
        label
        for _, label
        in pairs
    ], dtype=object)

    block = (
        X_final[
            :,
            indices,
        ]
        .toarray()
    )

    row_sums = block.sum(
        axis=1
    )

    np.testing.assert_allclose(
        row_sums,
        1.0,
        rtol=0.0,
        atol=1e-6,
    )

    decoded = labels[
        np.argmax(
            block,
            axis=1,
        )
    ]

    return decoded


transaction_mode = decode_categorical(
    "transaction_mode"
)

location_state = decode_categorical(
    "location_state"
)

assert len(
    transaction_mode
) == EXPECTED_FINAL_ROWS

assert len(
    location_state
) == EXPECTED_FINAL_ROWS

print(
    "Semantic rows:",
    EXPECTED_FINAL_ROWS,
)
print(
    "transaction_mode values:",
    sorted(
        set(
            transaction_mode.tolist()
        )
    ),
)
print(
    "location_state values:",
    sorted(
        set(
            location_state.tolist()
        )
    ),
)
print(
    "\nM8.5 SEMANTIC FEATURE RECONSTRUCTION GATE: PASS"
)

```

    Semantic rows: 722955
    transaction_mode values: ['Chip Transaction', 'Online Transaction', 'Swipe Transaction']
    location_state values: ['NON_PHYSICAL_OR_ONLINE', 'PHYSICAL_COMPLETE', 'PHYSICAL_ZIP_UNAVAILABLE']
    
    M8.5 SEMANTIC FEATURE RECONSTRUCTION GATE: PASS


## 10. Descriptive semantic profile theo error group


```python

def safe_float(
    value,
):
    if value is None:
        return None

    value = float(
        value
    )

    if not np.isfinite(
        value
    ):
        return None

    return value


def numeric_group_summary(
    values,
    mask,
):
    selected = np.asarray(
        values
    )[
        mask
    ]

    finite = selected[
        np.isfinite(
            selected
        )
    ]

    if finite.size == 0:
        return {
            "finite_count": 0,
        }

    return {
        "finite_count":
            int(
                finite.size
            ),
        "median":
            safe_float(
                np.median(
                    finite
                )
            ),
        "p25":
            safe_float(
                np.quantile(
                    finite,
                    0.25,
                )
            ),
        "p75":
            safe_float(
                np.quantile(
                    finite,
                    0.75,
                )
            ),
        "mean":
            safe_float(
                np.mean(
                    finite
                )
            ),
    }


def categorical_rate_summary(
    values,
    mask,
):
    selected = np.asarray(
        values,
        dtype=object,
    )[
        mask
    ]

    unique, counts = np.unique(
        selected,
        return_counts=True,
    )

    total = int(
        selected.size
    )

    return {
        str(
            value
        ): {
            "count":
                int(
                    count
                ),
            "rate":
                (
                    float(
                        count
                        /
                        total
                    )
                    if total
                    else 0.0
                ),
        }
        for value, count
        in zip(
            unique,
            counts,
        )
    }


error_group_masks = {
    "TP": tp_mask,
    "FP": fp_mask,
    "FN": fn_mask,
    "TN": tn_mask,
}

semantic_group_profile = {}

for group_name, mask in (
    error_group_masks.items()
):
    count = int(
        np.count_nonzero(
            mask
        )
    )

    semantic_group_profile[
        group_name
    ] = {
        "count":
            count,
        "risk_score":
            score_summary[
                group_name
            ],
        "amount_numeric":
            numeric_group_summary(
                numeric_semantic[
                    "amount_numeric"
                ],
                mask,
            ),
        "time_since_previous_transaction_min":
            numeric_group_summary(
                numeric_semantic[
                    "time_since_previous_transaction_min"
                ],
                mask,
            ),
        "transactions_last_1h":
            numeric_group_summary(
                numeric_semantic[
                    "transactions_last_1h"
                ],
                mask,
            ),
        "amount_minus_previous_mean":
            numeric_group_summary(
                numeric_semantic[
                    "amount_minus_previous_mean"
                ],
                mask,
            ),
        "is_new_merchant_rate":
            (
                float(
                    np.mean(
                        is_new_merchant[
                            mask
                        ]
                    )
                )
                if count
                else 0.0
            ),
        "has_prior_card_history_rate":
            (
                float(
                    np.mean(
                        has_prior_card_history[
                            mask
                        ]
                    )
                )
                if count
                else 0.0
            ),
        "transaction_mode":
            categorical_rate_summary(
                transaction_mode,
                mask,
            ),
        "location_state":
            categorical_rate_summary(
                location_state,
                mask,
            ),
    }

print(
    json.dumps(
        semantic_group_profile,
        ensure_ascii=False,
        indent=2,
    )
)

print(
    "\nM8.5 SEMANTIC ERROR-GROUP PROFILE GATE: PASS"
)

```

    {
      "TP": {
        "count": 393,
        "risk_score": {
          "count": 393,
          "min": 0.5099999904632568,
          "p10": 0.6000000238418579,
          "p25": 0.699999988079071,
          "median": 0.8299999833106995,
          "p75": 0.9100000262260437,
          "p90": 0.9599999785423279,
          "max": 1.0,
          "mean": 0.7981424936811432
        },
        "amount_numeric": {
          "finite_count": 393,
          "median": 44.950000009550294,
          "p25": 9.190000798391218,
          "p75": 123.60000160725107,
          "mean": 100.62923675431877
        },
        "time_since_previous_transaction_min": {
          "finite_count": 393,
          "median": 306.9999785506918,
          "p25": 76.9999954754469,
          "p75": 1320.9999980652206,
          "mean": 992.7633581238733
        },
        "transactions_last_1h": {
          "finite_count": 393,
          "median": 6.291368037647516e-09,
          "p25": 6.291368037647516e-09,
          "p75": 6.291368037647516e-09,
          "mean": 0.2391857555201368
        },
        "amount_minus_previous_mean": {
          "finite_count": 393,
          "median": 1.0988230523446842,
          "p25": -30.1098051053162,
          "p75": 83.04333122662995,
          "mean": 56.58387116897805
        },
        "is_new_merchant_rate": 0.6106870229007634,
        "has_prior_card_history_rate": 1.0,
        "transaction_mode": {
          "Chip Transaction": {
            "count": 358,
            "rate": 0.910941475826972
          },
          "Swipe Transaction": {
            "count": 35,
            "rate": 0.089058524173028
          }
        },
        "location_state": {
          "PHYSICAL_ZIP_UNAVAILABLE": {
            "count": 393,
            "rate": 1.0
          }
        }
      },
      "FP": {
        "count": 782,
        "risk_score": {
          "count": 782,
          "min": 0.5099999904632568,
          "p10": 0.550000011920929,
          "p25": 0.6100000143051147,
          "median": 0.7200000286102295,
          "p75": 0.8299999833106995,
          "p90": 0.9100000262260437,
          "max": 1.0,
          "mean": 0.7247058819322025
        },
        "amount_numeric": {
          "finite_count": 782,
          "median": 23.994999896639214,
          "p25": 10.817500282137503,
          "p75": 77.99999900219106,
          "mean": 60.38693100427287
        },
        "time_since_previous_transaction_min": {
          "finite_count": 782,
          "median": 363.50000999157913,
          "p25": 84.25003601713018,
          "p75": 1444.7500041910125,
          "mean": 1056.9322284894795
        },
        "transactions_last_1h": {
          "finite_count": 782,
          "median": 6.291368037647516e-09,
          "p25": 6.291368037647516e-09,
          "p75": 6.291368037647516e-09,
          "mean": 0.2647058876076443
        },
        "amount_minus_previous_mean": {
          "finite_count": 782,
          "median": -10.919025366660879,
          "p25": -30.663323753480107,
          "p75": 31.686786425989144,
          "mean": 13.339581619055297
        },
        "is_new_merchant_rate": 0.37595907928388744,
        "has_prior_card_history_rate": 1.0,
        "transaction_mode": {
          "Chip Transaction": {
            "count": 710,
            "rate": 0.907928388746803
          },
          "Online Transaction": {
            "count": 3,
            "rate": 0.0038363171355498722
          },
          "Swipe Transaction": {
            "count": 69,
            "rate": 0.08823529411764706
          }
        },
        "location_state": {
          "NON_PHYSICAL_OR_ONLINE": {
            "count": 3,
            "rate": 0.0038363171355498722
          },
          "PHYSICAL_ZIP_UNAVAILABLE": {
            "count": 779,
            "rate": 0.9961636828644501
          }
        }
      },
      "FN": {
        "count": 642,
        "risk_score": {
          "count": 642,
          "min": 0.0,
          "p10": 0.0,
          "p25": 0.0,
          "median": 0.009999999776482582,
          "p75": 0.03999999910593033,
          "p90": 0.24899999946355797,
          "max": 0.5,
          "mean": 0.06260124607899478
        },
        "amount_numeric": {
          "finite_count": 642,
          "median": 30.474999976233413,
          "p25": 6.774999996313445,
          "p75": 82.70250011854856,
          "mean": 65.2853893309618
        },
        "time_since_previous_transaction_min": {
          "finite_count": 642,
          "median": 289.5000044835535,
          "p25": 84.9999620266151,
          "p75": 1374.249999794609,
          "mean": 1017.0545200051564
        },
        "transactions_last_1h": {
          "finite_count": 642,
          "median": 6.291368037647516e-09,
          "p25": 6.291368037647516e-09,
          "p75": 6.291368037647516e-09,
          "mean": 0.28816200106947604
        },
        "amount_minus_previous_mean": {
          "finite_count": 642,
          "median": -9.926681952609286,
          "p25": -42.35673958701905,
          "p75": 42.16856469704344,
          "mean": 19.014504737778832
        },
        "is_new_merchant_rate": 0.45638629283489096,
        "has_prior_card_history_rate": 1.0,
        "transaction_mode": {
          "Chip Transaction": {
            "count": 578,
            "rate": 0.9003115264797508
          },
          "Swipe Transaction": {
            "count": 64,
            "rate": 0.09968847352024922
          }
        },
        "location_state": {
          "PHYSICAL_ZIP_UNAVAILABLE": {
            "count": 642,
            "rate": 1.0
          }
        }
      },
      "TN": {
        "count": 721138,
        "risk_score": {
          "count": 721138,
          "min": 0.0,
          "p10": 0.0,
          "p25": 0.0,
          "median": 0.0,
          "p75": 0.0,
          "p90": 0.0,
          "max": 0.5,
          "mean": 0.0005939917160870857
        },
        "amount_numeric": {
          "finite_count": 721138,
          "median": 29.20999958467597,
          "p25": 9.030001033521891,
          "p75": 62.890000502198205,
          "mean": 42.53208886638418
        },
        "time_since_previous_transaction_min": {
          "finite_count": 721099,
          "median": 601.0000089450022,
          "p25": 116.99995419517563,
          "p75": 1432.9999942734119,
          "mean": 1183.4689939757632
        },
        "transactions_last_1h": {
          "finite_count": 721138,
          "median": 6.291368037647516e-09,
          "p25": 6.291368037647516e-09,
          "p75": 6.291368037647516e-09,
          "mean": 0.2791573930133223
        },
        "amount_minus_previous_mean": {
          "finite_count": 721099,
          "median": -8.899868810388165,
          "p25": -27.670638796146036,
          "p75": 19.261623029443214,
          "mean": -0.6357468671519378
        },
        "is_new_merchant_rate": 0.02691440473251999,
        "has_prior_card_history_rate": 0.9999459188116561,
        "transaction_mode": {
          "Chip Transaction": {
            "count": 509614,
            "rate": 0.7066802747879046
          },
          "Online Transaction": {
            "count": 89681,
            "rate": 0.12436038594554717
          },
          "Swipe Transaction": {
            "count": 121843,
            "rate": 0.16895933926654816
          }
        },
        "location_state": {
          "NON_PHYSICAL_OR_ONLINE": {
            "count": 90393,
            "rate": 0.12534771430710903
          },
          "PHYSICAL_COMPLETE": {
            "count": 626381,
            "rate": 0.8686007393869135
          },
          "PHYSICAL_ZIP_UNAVAILABLE": {
            "count": 4364,
            "rate": 0.0060515463059774965
          }
        }
      }
    }
    
    M8.5 SEMANTIC ERROR-GROUP PROFILE GATE: PASS



## 11. Error-rate concentration theo semantic subgroup

Mục tiêu là mô tả:

```text
fraud miss rate trong từng subgroup
non-fraud false-positive rate trong từng subgroup
```

Không dùng subgroup result để chọn lại feature/threshold.



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


def subgroup_error_profile(
    values,
):
    values = np.asarray(
        values,
        dtype=object,
    )

    result = {}

    for value in sorted(
        set(
            values.tolist()
        ),
        key=str,
    ):
        subgroup = (
            values
            ==
            value
        )

        fraud_count = int(
            np.count_nonzero(
                subgroup
                &
                (y_final == 1)
            )
        )

        fraud_fn = int(
            np.count_nonzero(
                subgroup
                &
                fn_mask
            )
        )

        nonfraud_count = int(
            np.count_nonzero(
                subgroup
                &
                (y_final == 0)
            )
        )

        nonfraud_fp = int(
            np.count_nonzero(
                subgroup
                &
                fp_mask
            )
        )

        result[
            str(
                value
            )
        ] = {
            "rows":
                int(
                    np.count_nonzero(
                        subgroup
                    )
                ),
            "fraud_count":
                fraud_count,
            "fraud_fn":
                fraud_fn,
            "fraud_miss_rate":
                safe_ratio(
                    fraud_fn,
                    fraud_count,
                ),
            "nonfraud_count":
                nonfraud_count,
            "nonfraud_fp":
                nonfraud_fp,
            "nonfraud_fp_rate":
                safe_ratio(
                    nonfraud_fp,
                    nonfraud_count,
                ),
        }

    return result


subgroup_profiles = {
    "is_new_merchant":
        subgroup_error_profile(
            is_new_merchant.astype(
                str
            )
        ),
    "has_prior_card_history":
        subgroup_error_profile(
            has_prior_card_history.astype(
                str
            )
        ),
    "transaction_mode":
        subgroup_error_profile(
            transaction_mode
        ),
    "location_state":
        subgroup_error_profile(
            location_state
        ),
}

# Historical M6.6 descriptive direction check:
# is_new_merchant rate: FN < TP and FP > TN across six validation runs.
final_new_merchant_direction = {
    "fn_rate":
        float(
            np.mean(
                is_new_merchant[
                    fn_mask
                ]
            )
        ),
    "tp_rate":
        float(
            np.mean(
                is_new_merchant[
                    tp_mask
                ]
            )
        ),
    "fp_rate":
        float(
            np.mean(
                is_new_merchant[
                    fp_mask
                ]
            )
        ),
    "tn_rate":
        float(
            np.mean(
                is_new_merchant[
                    tn_mask
                ]
            )
        ),
}

final_new_merchant_direction[
    "m6_6_fn_lt_tp_direction_persists"
] = (
    final_new_merchant_direction[
        "fn_rate"
    ]
    <
    final_new_merchant_direction[
        "tp_rate"
    ]
)

final_new_merchant_direction[
    "m6_6_fp_gt_tn_direction_persists"
] = (
    final_new_merchant_direction[
        "fp_rate"
    ]
    >
    final_new_merchant_direction[
        "tn_rate"
    ]
)

print(
    json.dumps(
        {
            "subgroup_profiles":
                subgroup_profiles,
            "is_new_merchant_direction_check":
                final_new_merchant_direction,
        },
        ensure_ascii=False,
        indent=2,
    )
)

print(
    "\nM8.5 ERROR CONCENTRATION / SUBGROUP GATE: PASS"
)

```

    {
      "subgroup_profiles": {
        "is_new_merchant": {
          "False": {
            "rows": 702719,
            "fraud_count": 502,
            "fraud_fn": 349,
            "fraud_miss_rate": 0.6952191235059761,
            "nonfraud_count": 702217,
            "nonfraud_fp": 488,
            "nonfraud_fp_rate": 0.0006949418769411735
          },
          "True": {
            "rows": 20236,
            "fraud_count": 533,
            "fraud_fn": 293,
            "fraud_miss_rate": 0.549718574108818,
            "nonfraud_count": 19703,
            "nonfraud_fp": 294,
            "nonfraud_fp_rate": 0.014921585545348424
          }
        },
        "has_prior_card_history": {
          "False": {
            "rows": 39,
            "fraud_count": 0,
            "fraud_fn": 0,
            "fraud_miss_rate": 0.0,
            "nonfraud_count": 39,
            "nonfraud_fp": 0,
            "nonfraud_fp_rate": 0.0
          },
          "True": {
            "rows": 722916,
            "fraud_count": 1035,
            "fraud_fn": 642,
            "fraud_miss_rate": 0.6202898550724638,
            "nonfraud_count": 721881,
            "nonfraud_fp": 782,
            "nonfraud_fp_rate": 0.0010832810393956898
          }
        },
        "transaction_mode": {
          "Chip Transaction": {
            "rows": 511260,
            "fraud_count": 936,
            "fraud_fn": 578,
            "fraud_miss_rate": 0.6175213675213675,
            "nonfraud_count": 510324,
            "nonfraud_fp": 710,
            "nonfraud_fp_rate": 0.0013912729951952093
          },
          "Online Transaction": {
            "rows": 89684,
            "fraud_count": 0,
            "fraud_fn": 0,
            "fraud_miss_rate": 0.0,
            "nonfraud_count": 89684,
            "nonfraud_fp": 3,
            "nonfraud_fp_rate": 3.345078274831631e-05
          },
          "Swipe Transaction": {
            "rows": 122011,
            "fraud_count": 99,
            "fraud_fn": 64,
            "fraud_miss_rate": 0.6464646464646465,
            "nonfraud_count": 121912,
            "nonfraud_fp": 69,
            "nonfraud_fp_rate": 0.0005659820198175733
          }
        },
        "location_state": {
          "NON_PHYSICAL_OR_ONLINE": {
            "rows": 90396,
            "fraud_count": 0,
            "fraud_fn": 0,
            "fraud_miss_rate": 0.0,
            "nonfraud_count": 90396,
            "nonfraud_fp": 3,
            "nonfraud_fp_rate": 3.318730917297226e-05
          },
          "PHYSICAL_COMPLETE": {
            "rows": 626381,
            "fraud_count": 0,
            "fraud_fn": 0,
            "fraud_miss_rate": 0.0,
            "nonfraud_count": 626381,
            "nonfraud_fp": 0,
            "nonfraud_fp_rate": 0.0
          },
          "PHYSICAL_ZIP_UNAVAILABLE": {
            "rows": 6178,
            "fraud_count": 1035,
            "fraud_fn": 642,
            "fraud_miss_rate": 0.6202898550724638,
            "nonfraud_count": 5143,
            "nonfraud_fp": 779,
            "nonfraud_fp_rate": 0.1514680147773673
          }
        }
      },
      "is_new_merchant_direction_check": {
        "fn_rate": 0.45638629283489096,
        "tp_rate": 0.6106870229007634,
        "fp_rate": 0.37595907928388744,
        "tn_rate": 0.02691440473251999,
        "m6_6_fn_lt_tp_direction_persists": true,
        "m6_6_fp_gt_tn_direction_persists": true
      }
    }
    
    M8.5 ERROR CONCENTRATION / SUBGROUP GATE: PASS


## 12. Monthly FINAL TEST behavior — 2019-06 → 2019-10


```python

month_values = (
    timestamp_final
    .astype(
        "datetime64[M]"
    )
    .astype(
        str
    )
)

observed_months = sorted(
    np.unique(
        month_values
    ).tolist()
)

assert (
    observed_months
    ==
    EXPECTED_FINAL_MONTHS
)

def metric_from_counts(
    tp_value,
    fp_value,
    fn_value,
    tn_value,
):
    total_value = (
        tp_value
        + fp_value
        + fn_value
        + tn_value
    )

    return {
        "F1_fraud":
            safe_ratio(
                2 * tp_value,
                (
                    2 * tp_value
                    + fp_value
                    + fn_value
                ),
            ),
        "Recall_fraud":
            safe_ratio(
                tp_value,
                tp_value
                + fn_value,
            ),
        "Precision_fraud":
            safe_ratio(
                tp_value,
                tp_value
                + fp_value,
            ),
        "Accuracy_reference":
            safe_ratio(
                tp_value
                + tn_value,
                total_value,
            ),
        "predicted_positive_count":
            int(
                tp_value
                + fp_value
            ),
        "predicted_positive_rate":
            safe_ratio(
                tp_value
                + fp_value,
                total_value,
            ),
    }


monthly_profile = []

monthly_sum = {
    "rows": 0,
    "fraud": 0,
    "TP": 0,
    "FP": 0,
    "FN": 0,
    "TN": 0,
}

for month in observed_months:
    mask = (
        month_values
        ==
        month
    )

    month_rows = int(
        np.count_nonzero(
            mask
        )
    )

    month_tp = int(
        np.count_nonzero(
            mask
            &
            tp_mask
        )
    )

    month_fp = int(
        np.count_nonzero(
            mask
            &
            fp_mask
        )
    )

    month_fn = int(
        np.count_nonzero(
            mask
            &
            fn_mask
        )
    )

    month_tn = int(
        np.count_nonzero(
            mask
            &
            tn_mask
        )
    )

    month_fraud = (
        month_tp
        + month_fn
    )

    metrics = metric_from_counts(
        month_tp,
        month_fp,
        month_fn,
        month_tn,
    )

    monthly_profile.append({
        "month":
            month,
        "rows":
            month_rows,
        "fraud":
            month_fraud,
        "fraud_rate":
            safe_ratio(
                month_fraud,
                month_rows,
            ),
        "TP":
            month_tp,
        "FP":
            month_fp,
        "FN":
            month_fn,
        "TN":
            month_tn,
        **metrics,
        "FN_risk_score_median":
            (
                safe_float(
                    np.median(
                        risk_score_final[
                            mask
                            &
                            fn_mask
                        ]
                    )
                )
                if np.any(
                    mask
                    &
                    fn_mask
                )
                else None
            ),
        "FP_risk_score_median":
            (
                safe_float(
                    np.median(
                        risk_score_final[
                            mask
                            &
                            fp_mask
                        ]
                    )
                )
                if np.any(
                    mask
                    &
                    fp_mask
                )
                else None
            ),
    })

    monthly_sum[
        "rows"
    ] += month_rows

    monthly_sum[
        "fraud"
    ] += month_fraud

    monthly_sum[
        "TP"
    ] += month_tp

    monthly_sum[
        "FP"
    ] += month_fp

    monthly_sum[
        "FN"
    ] += month_fn

    monthly_sum[
        "TN"
    ] += month_tn


assert monthly_sum == {
    "rows": EXPECTED_FINAL_ROWS,
    "fraud": EXPECTED_FINAL_FRAUD,
    "TP": EXPECTED_FINAL_TP,
    "FP": EXPECTED_FINAL_FP,
    "FN": EXPECTED_FINAL_FN,
    "TN": EXPECTED_FINAL_TN,
}

monthly_extrema = {
    "min_F1_month":
        min(
            monthly_profile,
            key=lambda record:
                record[
                    "F1_fraud"
                ],
        )[
            "month"
        ],
    "max_F1_month":
        max(
            monthly_profile,
            key=lambda record:
                record[
                    "F1_fraud"
                ],
        )[
            "month"
        ],
    "min_Recall_month":
        min(
            monthly_profile,
            key=lambda record:
                record[
                    "Recall_fraud"
                ],
        )[
            "month"
        ],
    "max_Recall_month":
        max(
            monthly_profile,
            key=lambda record:
                record[
                    "Recall_fraud"
                ],
        )[
            "month"
        ],
    "min_Precision_month":
        min(
            monthly_profile,
            key=lambda record:
                record[
                    "Precision_fraud"
                ],
        )[
            "month"
        ],
    "max_Precision_month":
        max(
            monthly_profile,
            key=lambda record:
                record[
                    "Precision_fraud"
                ],
        )[
            "month"
        ],
}

print(
    json.dumps(
        {
            "monthly_profile":
                monthly_profile,
            "monthly_extrema":
                monthly_extrema,
        },
        ensure_ascii=False,
        indent=2,
    )
)

print(
    "\nM8.5 MONTHLY FINAL-TEST PROFILE GATE: PASS"
)

```

    {
      "monthly_profile": [
        {
          "month": "2019-06",
          "rows": 142399,
          "fraud": 215,
          "fraud_rate": 0.001509842063497637,
          "TP": 79,
          "FP": 212,
          "FN": 136,
          "TN": 141972,
          "F1_fraud": 0.31225296442687744,
          "Recall_fraud": 0.3674418604651163,
          "Precision_fraud": 0.27147766323024053,
          "Accuracy_reference": 0.9975561626135016,
          "predicted_positive_count": 291,
          "predicted_positive_rate": 0.0020435536766409876,
          "FN_risk_score_median": 0.009999999776482582,
          "FP_risk_score_median": 0.7150000333786011
        },
        {
          "month": "2019-07",
          "rows": 146599,
          "fraud": 156,
          "fraud_rate": 0.001064127313283174,
          "TP": 51,
          "FP": 155,
          "FN": 105,
          "TN": 146288,
          "F1_fraud": 0.281767955801105,
          "Recall_fraud": 0.3269230769230769,
          "Precision_fraud": 0.24757281553398058,
          "Accuracy_reference": 0.9982264544778614,
          "predicted_positive_count": 206,
          "predicted_positive_rate": 0.0014051937598482936,
          "FN_risk_score_median": 0.009999999776482582,
          "FP_risk_score_median": 0.7300000190734863
        },
        {
          "month": "2019-08",
          "rows": 147139,
          "fraud": 247,
          "fraud_rate": 0.0016786847810573676,
          "TP": 106,
          "FP": 170,
          "FN": 141,
          "TN": 146722,
          "F1_fraud": 0.40535372848948376,
          "Recall_fraud": 0.4291497975708502,
          "Precision_fraud": 0.38405797101449274,
          "Accuracy_reference": 0.9978863523606929,
          "predicted_positive_count": 276,
          "predicted_positive_rate": 0.0018757773262017548,
          "FN_risk_score_median": 0.009999999776482582,
          "FP_risk_score_median": 0.7099999785423279
        },
        {
          "month": "2019-09",
          "rows": 141744,
          "fraud": 142,
          "fraud_rate": 0.0010018060729201942,
          "TP": 53,
          "FP": 96,
          "FN": 89,
          "TN": 141506,
          "F1_fraud": 0.3642611683848797,
          "Recall_fraud": 0.3732394366197183,
          "Precision_fraud": 0.35570469798657717,
          "Accuracy_reference": 0.9986948301162659,
          "predicted_positive_count": 149,
          "predicted_positive_rate": 0.001051190879331753,
          "FN_risk_score_median": 0.019999999552965164,
          "FP_risk_score_median": 0.7250000238418579
        },
        {
          "month": "2019-10",
          "rows": 145074,
          "fraud": 275,
          "fraud_rate": 0.001895584322483698,
          "TP": 104,
          "FP": 149,
          "FN": 171,
          "TN": 144650,
          "F1_fraud": 0.3939393939393939,
          "Recall_fraud": 0.3781818181818182,
          "Precision_fraud": 0.41106719367588934,
          "Accuracy_reference": 0.997794229152019,
          "predicted_positive_count": 253,
          "predicted_positive_rate": 0.0017439375766850021,
          "FN_risk_score_median": 0.009999999776482582,
          "FP_risk_score_median": 0.7200000286102295
        }
      ],
      "monthly_extrema": {
        "min_F1_month": "2019-07",
        "max_F1_month": "2019-08",
        "min_Recall_month": "2019-07",
        "max_Recall_month": "2019-08",
        "min_Precision_month": "2019-07",
        "max_Precision_month": "2019-10"
      }
    }
    
    M8.5 MONTHLY FINAL-TEST PROFILE GATE: PASS



## 13. Reconstruct selected RF VALIDATION profile từ persisted M7.7 artifacts

Không chạy model lại.

Nguồn:

```text
M4.7 y_validation
M7.7 selected-candidate y_pred
M7.7 selected-candidate risk_score
```



```python

y_validation = np.load(
    Y_VALIDATION_PATH,
    allow_pickle=False,
)

validation_pred = np.load(
    M7_VALIDATION_PRED_PATH,
    allow_pickle=False,
)

validation_score = np.load(
    M7_VALIDATION_SCORE_PATH,
    allow_pickle=False,
)

assert (
    y_validation.shape
    ==
    (
        EXPECTED_VALIDATION_ROWS,
    )
)

assert (
    validation_pred.shape
    ==
    (
        EXPECTED_VALIDATION_ROWS,
    )
)

assert (
    validation_score.shape
    ==
    (
        EXPECTED_VALIDATION_ROWS,
    )
)

assert (
    int(
        y_validation.sum()
    )
    ==
    EXPECTED_VALIDATION_FRAUD
)

assert np.isfinite(
    validation_score
).all()

validation_threshold_reconstructed = (
    validation_score
    >
    FINAL_THRESHOLD
).astype(
    np.int8,
    copy=False,
)

validation_threshold_mismatch = int(
    np.count_nonzero(
        validation_threshold_reconstructed
        !=
        validation_pred
    )
)

assert (
    validation_threshold_mismatch
    ==
    0
)

validation_tp = int(
    np.count_nonzero(
        (y_validation == 1)
        &
        (validation_pred == 1)
    )
)

validation_fp = int(
    np.count_nonzero(
        (y_validation == 0)
        &
        (validation_pred == 1)
    )
)

validation_fn = int(
    np.count_nonzero(
        (y_validation == 1)
        &
        (validation_pred == 0)
    )
)

validation_tn = int(
    np.count_nonzero(
        (y_validation == 0)
        &
        (validation_pred == 0)
    )
)

assert (
    validation_tp,
    validation_fp,
    validation_fn,
    validation_tn,
) == (
    EXPECTED_VALIDATION_TP,
    EXPECTED_VALIDATION_FP,
    EXPECTED_VALIDATION_FN,
    EXPECTED_VALIDATION_TN,
)

validation_metrics = metric_from_counts(
    validation_tp,
    validation_fp,
    validation_fn,
    validation_tn,
)

validation_profile = {
    "rows":
        EXPECTED_VALIDATION_ROWS,
    "fraud":
        EXPECTED_VALIDATION_FRAUD,
    "fraud_rate":
        safe_ratio(
            EXPECTED_VALIDATION_FRAUD,
            EXPECTED_VALIDATION_ROWS,
        ),
    "TP":
        validation_tp,
    "FP":
        validation_fp,
    "FN":
        validation_fn,
    "TN":
        validation_tn,
    **validation_metrics,
    "threshold_reconstruction_mismatch":
        validation_threshold_mismatch,
}

print(
    json.dumps(
        validation_profile,
        ensure_ascii=False,
        indent=2,
    )
)

print(
    "\nM8.5 SELECTED-RF VALIDATION PROFILE GATE: PASS"
)

```

    {
      "rows": 712458,
      "fraud": 1052,
      "fraud_rate": 0.0014765782684733697,
      "TP": 467,
      "FP": 764,
      "FN": 585,
      "TN": 710642,
      "F1_fraud": 0.4091108190976785,
      "Recall_fraud": 0.4439163498098859,
      "Precision_fraud": 0.3793663688058489,
      "Accuracy_reference": 0.9981065550530698,
      "predicted_positive_count": 1231,
      "predicted_positive_rate": 0.0017278211487554353,
      "threshold_reconstruction_mismatch": 0
    }
    
    M8.5 SELECTED-RF VALIDATION PROFILE GATE: PASS


## 14. VALIDATION → FINAL TEST delta


```python

final_metrics = (
    m8_04_result[
        "metrics"
    ]
)

final_confusion = (
    m8_04_result[
        "confusion_matrix"
    ]
)

assert final_confusion == {
    "TP": EXPECTED_FINAL_TP,
    "FP": EXPECTED_FINAL_FP,
    "FN": EXPECTED_FINAL_FN,
    "TN": EXPECTED_FINAL_TN,
}

assert (
    int(
        final_metrics[
            "predicted_positive_count"
        ]
    )
    ==
    EXPECTED_FINAL_TP
    +
    EXPECTED_FINAL_FP
)

temporal_delta = {
    "F1_fraud": {
        "validation":
            validation_metrics[
                "F1_fraud"
            ],
        "final":
            float(
                final_metrics[
                    "F1_fraud"
                ]
            ),
    },
    "Recall_fraud": {
        "validation":
            validation_metrics[
                "Recall_fraud"
            ],
        "final":
            float(
                final_metrics[
                    "Recall_fraud"
                ]
            ),
    },
    "Precision_fraud": {
        "validation":
            validation_metrics[
                "Precision_fraud"
            ],
        "final":
            float(
                final_metrics[
                    "Precision_fraud"
                ]
            ),
    },
    "predicted_positive_count": {
        "validation":
            int(
                validation_metrics[
                    "predicted_positive_count"
                ]
            ),
        "final":
            int(
                final_metrics[
                    "predicted_positive_count"
                ]
            ),
    },
    "predicted_positive_rate": {
        "validation":
            validation_metrics[
                "predicted_positive_rate"
            ],
        "final":
            float(
                final_metrics[
                    "predicted_positive_rate"
                ]
            ),
    },
    "fraud_rate": {
        "validation":
            safe_ratio(
                EXPECTED_VALIDATION_FRAUD,
                EXPECTED_VALIDATION_ROWS,
            ),
        "final":
            safe_ratio(
                EXPECTED_FINAL_FRAUD,
                EXPECTED_FINAL_ROWS,
            ),
    },
}

for metric_name, record in (
    temporal_delta.items()
):
    record[
        "absolute_delta_final_minus_validation"
    ] = (
        record[
            "final"
        ]
        -
        record[
            "validation"
        ]
    )

for metric_name in [
    "F1_fraud",
    "Recall_fraud",
    "Precision_fraud",
]:
    validation_value = (
        temporal_delta[
            metric_name
        ][
            "validation"
        ]
    )

    final_value = (
        temporal_delta[
            metric_name
        ][
            "final"
        ]
    )

    temporal_delta[
        metric_name
    ][
        "relative_delta_vs_validation"
    ] = (
        safe_ratio(
            final_value
            -
            validation_value,
            validation_value,
        )
        if validation_value
        else None
    )

classification_metrics_all_lower = all(
    temporal_delta[
        metric
    ][
        "final"
    ]
    <
    temporal_delta[
        metric
    ][
        "validation"
    ]
    for metric
    in [
        "F1_fraud",
        "Recall_fraud",
        "Precision_fraud",
    ]
)

monthly_f1_values = [
    record[
        "F1_fraud"
    ]
    for record
    in monthly_profile
]

monthly_recall_values = [
    record[
        "Recall_fraud"
    ]
    for record
    in monthly_profile
]

monthly_precision_values = [
    record[
        "Precision_fraud"
    ]
    for record
    in monthly_profile
]

temporal_generalization_evidence = {
    "classification_metrics_all_lower_on_final_vs_validation":
        classification_metrics_all_lower,
    "monthly_F1_min":
        float(
            min(
                monthly_f1_values
            )
        ),
    "monthly_F1_max":
        float(
            max(
                monthly_f1_values
            )
        ),
    "monthly_Recall_min":
        float(
            min(
                monthly_recall_values
            )
        ),
    "monthly_Recall_max":
        float(
            max(
                monthly_recall_values
            )
        ),
    "monthly_Precision_min":
        float(
            min(
                monthly_precision_values
            )
        ),
    "monthly_Precision_max":
        float(
            max(
                monthly_precision_values
            )
        ),
    "interpretation_status":
        "DESCRIPTIVE EVIDENCE ONLY — FINAL REVIEW REQUIRED",
}

print(
    json.dumps(
        {
            "validation_to_final_delta":
                temporal_delta,
            "temporal_generalization_evidence":
                temporal_generalization_evidence,
        },
        ensure_ascii=False,
        indent=2,
    )
)

print(
    "\nM8.5 VALIDATION → FINAL TEMPORAL DELTA GATE: PASS"
)

```

    {
      "validation_to_final_delta": {
        "F1_fraud": {
          "validation": 0.4091108190976785,
          "final": 0.3556561085972851,
          "absolute_delta_final_minus_validation": -0.05345471050039341,
          "relative_delta_vs_validation": -0.13066071099828497
        },
        "Recall_fraud": {
          "validation": 0.4439163498098859,
          "final": 0.37971014492753624,
          "absolute_delta_final_minus_validation": -0.06420620488234968,
          "relative_delta_vs_validation": -0.14463581913540013
        },
        "Precision_fraud": {
          "validation": 0.3793663688058489,
          "final": 0.334468085106383,
          "absolute_delta_final_minus_validation": -0.04489828369946591,
          "relative_delta_vs_validation": -0.11835072212857074
        },
        "predicted_positive_count": {
          "validation": 1231,
          "final": 1175,
          "absolute_delta_final_minus_validation": -56
        },
        "predicted_positive_rate": {
          "validation": 0.0017278211487554353,
          "final": 0.0016252740488688785,
          "absolute_delta_final_minus_validation": -0.00010254709988655685
        },
        "fraud_rate": {
          "validation": 0.0014765782684733697,
          "final": 0.0014316243749610972,
          "absolute_delta_final_minus_validation": -4.4953893512272554e-05
        }
      },
      "temporal_generalization_evidence": {
        "classification_metrics_all_lower_on_final_vs_validation": true,
        "monthly_F1_min": 0.281767955801105,
        "monthly_F1_max": 0.40535372848948376,
        "monthly_Recall_min": 0.3269230769230769,
        "monthly_Recall_max": 0.4291497975708502,
        "monthly_Precision_min": 0.24757281553398058,
        "monthly_Precision_max": 0.41106719367588934,
        "interpretation_status": "DESCRIPTIVE EVIDENCE ONLY — FINAL REVIEW REQUIRED"
      }
    }
    
    M8.5 VALIDATION → FINAL TEMPORAL DELTA GATE: PASS



## 15. Persist M8.5 analysis artifacts

Persist:

```text
m8_05_final_error_analysis.json
m8_05_analysis_manifest.json
m8_05_error_group_row_ids.npz
```

Decision vẫn giữ `OPEN` cho tới AI runtime review.



```python

source_fingerprints = {
    "m7_07_registry_sha256":
        sha256_file(
            M7_07_REGISTRY_PATH
        ),
    "m7_07_manifest_sha256":
        sha256_file(
            M7_07_MANIFEST_PATH
        ),
    "m7_07_validation_pred_sha256":
        sha256_file(
            M7_VALIDATION_PRED_PATH
        ),
    "m7_07_validation_score_sha256":
        sha256_file(
            M7_VALIDATION_SCORE_PATH
        ),
    "m4_y_validation_sha256":
        sha256_file(
            Y_VALIDATION_PATH
        ),
    "m8_02_x_final_test_sha256":
        sha256_file(
            X_FINAL_PATH
        ),
    "m8_02_preprocessing_state_sha256":
        sha256_file(
            PREPROCESSING_STATE_PATH
        ),
    "m8_03_registry_sha256":
        sha256_file(
            M8_03_REGISTRY_PATH
        ),
    "m8_03_manifest_sha256":
        sha256_file(
            M8_03_MANIFEST_PATH
        ),
    "m8_03_risk_score_final_sha256":
        sha256_file(
            RISK_SCORE_FINAL_PATH
        ),
    "m8_03_y_pred_final_sha256":
        sha256_file(
            Y_PRED_FINAL_PATH
        ),
    "m8_03_y_final_sha256":
        sha256_file(
            Y_FINAL_PATH
        ),
    "m8_03_row_id_final_sha256":
        sha256_file(
            ROW_FINAL_PATH
        ),
    "m8_03_timestamp_final_sha256":
        sha256_file(
            TIMESTAMP_FINAL_PATH
        ),
    "m8_04_result_sha256":
        sha256_file(
            M8_04_RESULT_PATH
        ),
    "m8_04_manifest_sha256":
        sha256_file(
            M8_04_MANIFEST_PATH
        ),
    "reviewed_m8_04_notebook_sha256":
        UPSTREAM_REVIEWED_M8_04_NOTEBOOK_SHA256,
}

row_group_sha256 = sha256_file(
    ROW_GROUP_PATH
)

analysis_payload = {
    "analysis_version":
        M8_05_ANALYSIS_VERSION,
    "work_type":
        "DESCRIPTIVE_FINAL_ERROR_AND_TEMPORAL_GENERALIZATION_REVIEW",
    "guardrail":
        "ANALYZE — DO NOT FIX USING FINAL TEST",
    "upstream_review": {
        "m8_04_status":
            UPSTREAM_REVIEWED_M8_04_STATUS,
        "reviewed_m8_04_notebook_sha256":
            UPSTREAM_REVIEWED_M8_04_NOTEBOOK_SHA256,
    },
    "final_test_identity": {
        "rows":
            EXPECTED_FINAL_ROWS,
        "fraud":
            EXPECTED_FINAL_FRAUD,
        "threshold":
            FINAL_THRESHOLD,
        "comparator":
            THRESHOLD_COMPARATOR,
    },
    "final_confusion_matrix": {
        "TP":
            tp,
        "FP":
            fp,
        "FN":
            fn,
        "TN":
            tn,
    },
    "final_metrics":
        final_metrics,
    "risk_score_error_profile": {
        "quantiles":
            score_summary,
        "bands":
            score_band_counts,
        "boundary_exact_0_50":
            boundary_exact_050,
    },
    "semantic_error_group_profile":
        semantic_group_profile,
    "subgroup_error_profiles":
        subgroup_profiles,
    "historical_m6_6_is_new_merchant_direction_check":
        final_new_merchant_direction,
    "monthly_final_test_profile":
        monthly_profile,
    "monthly_extrema":
        monthly_extrema,
    "selected_rf_validation_profile":
        validation_profile,
    "validation_to_final_delta":
        temporal_delta,
    "temporal_generalization_evidence":
        temporal_generalization_evidence,
    "error_row_lineage": {
        "artifact":
            ROW_GROUP_PATH.name,
        "sha256":
            row_group_sha256,
        "TP_rows":
            len(
                tp_row_ids
            ),
        "FP_rows":
            len(
                fp_row_ids
            ),
        "FN_rows":
            len(
                fn_row_ids
            ),
        "TN_rows":
            len(
                tn_row_ids
            ),
    },
    "source_fingerprints":
        source_fingerprints,
    "runtime_versions": {
        "python":
            sys.version,
        "numpy":
            np.__version__,
        "scipy":
            scipy.__version__,
        "platform":
            platform.platform(),
    },
    "attestations": {
        "model_called":
            False,
        "new_prediction_generated":
            False,
        "model_refit_performed":
            False,
        "preprocessing_refit_performed":
            False,
        "threshold_changed":
            False,
        "threshold_retuning_performed":
            False,
        "calibration_performed":
            False,
        "candidate_expansion_performed":
            False,
        "final_test_used_for_optimization":
            False,
        "descriptive_not_causal":
            True,
    },
    "decision":
        "OPEN — REQUIRES AI RUNTIME REVIEW",
    "m8_6_gate_authorization":
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
        M8_05_ANALYSIS_VERSION,
    "final_test_rows":
        EXPECTED_FINAL_ROWS,
    "final_test_fraud":
        EXPECTED_FINAL_FRAUD,
    "result_file":
        RESULT_PATH.name,
    "result_sha256":
        result_sha256,
    "error_row_lineage_file":
        ROW_GROUP_PATH.name,
    "error_row_lineage_sha256":
        row_group_sha256,
    "monthly_profile_months":
        EXPECTED_FINAL_MONTHS,
    "validation_comparator_source":
        "persisted M7.7 selected-candidate artifacts",
    "model_called":
        False,
    "new_prediction_generated":
        False,
    "final_test_used_for_optimization":
        False,
    "guardrail":
        "ANALYZE — DO NOT FIX USING FINAL TEST",
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
assert ROW_GROUP_PATH.exists()

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
    "Row lineage:",
    ROW_GROUP_PATH.relative_to(
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
    "Row-lineage SHA256:",
    row_group_sha256,
)

print(
    "\nM8.5 ARTIFACT PERSISTENCE GATE: PASS"
)

```

    Result: data/processed/m8_05_final_error_analysis/m8_05_final_error_analysis.json
    Manifest: data/processed/m8_05_final_error_analysis/m8_05_analysis_manifest.json
    Row lineage: data/processed/m8_05_final_error_analysis/m8_05_error_group_row_ids.npz
    Result SHA256: 2775a51e3d97205d964ee32775cc782419796f783f2bbc65c81c067bf6cb823d
    Manifest SHA256: 665d44b11fa5759e6efb2de78d375414dff688614f22d39bd4a571c99d23237e
    Row-lineage SHA256: 26da360a056293448f54384bab0293b727259d4b416439a6f49e4adf4576b5c9
    
    M8.5 ARTIFACT PERSISTENCE GATE: PASS


## 16. Round-trip M8.5 artifacts


```python

result_roundtrip = load_json(
    RESULT_PATH
)

manifest_roundtrip = load_json(
    MANIFEST_PATH
)

row_group_roundtrip = np.load(
    ROW_GROUP_PATH,
    allow_pickle=False,
)

assert (
    manifest_roundtrip[
        "result_sha256"
    ]
    ==
    sha256_file(
        RESULT_PATH
    )
)

assert (
    manifest_roundtrip[
        "error_row_lineage_sha256"
    ]
    ==
    sha256_file(
        ROW_GROUP_PATH
    )
)

np.testing.assert_array_equal(
    row_group_roundtrip[
        "tp_row_id"
    ],
    tp_row_ids,
)

np.testing.assert_array_equal(
    row_group_roundtrip[
        "fp_row_id"
    ],
    fp_row_ids,
)

np.testing.assert_array_equal(
    row_group_roundtrip[
        "fn_row_id"
    ],
    fn_row_ids,
)

np.testing.assert_array_equal(
    row_group_roundtrip[
        "tn_row_id"
    ],
    tn_row_ids,
)

assert (
    result_roundtrip[
        "final_confusion_matrix"
    ]
    ==
    {
        "TP": EXPECTED_FINAL_TP,
        "FP": EXPECTED_FINAL_FP,
        "FN": EXPECTED_FINAL_FN,
        "TN": EXPECTED_FINAL_TN,
    }
)

assert (
    result_roundtrip[
        "attestations"
    ][
        "model_called"
    ]
    is False
)

assert (
    result_roundtrip[
        "attestations"
    ][
        "new_prediction_generated"
    ]
    is False
)

assert (
    result_roundtrip[
        "attestations"
    ][
        "final_test_used_for_optimization"
    ]
    is False
)

print(
    "Result round-trip:",
    "PASS",
)
print(
    "Manifest round-trip:",
    "PASS",
)
print(
    "Error-row lineage round-trip:",
    "PASS",
)
print(
    "\nM8.5 ARTIFACT ROUND-TRIP GATE: PASS"
)

```

    Result round-trip: PASS
    Manifest round-trip: PASS
    Error-row lineage round-trip: PASS
    
    M8.5 ARTIFACT ROUND-TRIP GATE: PASS


## 17. M8.5 technical gate


```python

m8_05_gates = {
    "G01_SOURCE_LOCATION":
        True,
    "G02_OUTPUT_GUARD":
        True,
    "G03_FROZEN_ANALYSIS_CONTRACT":
        True,
    "G04_UPSTREAM_ARTIFACT_INTEGRITY":
        True,
    "G05_FINAL_ERROR_GROUP_IDENTITY":
        (
            tp
            + fp
            + fn
            + tn
            ==
            EXPECTED_FINAL_ROWS
        ),
    "G06_ERROR_ROW_LINEAGE":
        (
            len(
                np.unique(
                    all_group_rows
                )
            )
            ==
            EXPECTED_FINAL_ROWS
        ),
    "G07_RISK_SCORE_ERROR_PROFILE":
        True,
    "G08_SEMANTIC_FEATURE_RECONSTRUCTION":
        (
            len(
                transaction_mode
            )
            ==
            EXPECTED_FINAL_ROWS
            and
            len(
                location_state
            )
            ==
            EXPECTED_FINAL_ROWS
        ),
    "G09_ERROR_CONCENTRATION_SUBGROUPS":
        True,
    "G10_MONTHLY_FINAL_TEST_COVERAGE":
        (
            observed_months
            ==
            EXPECTED_FINAL_MONTHS
        ),
    "G11_MONTHLY_ARITHMETIC":
        (
            monthly_sum[
                "rows"
            ]
            ==
            EXPECTED_FINAL_ROWS
            and
            monthly_sum[
                "TP"
            ]
            ==
            EXPECTED_FINAL_TP
            and
            monthly_sum[
                "FP"
            ]
            ==
            EXPECTED_FINAL_FP
            and
            monthly_sum[
                "FN"
            ]
            ==
            EXPECTED_FINAL_FN
            and
            monthly_sum[
                "TN"
            ]
            ==
            EXPECTED_FINAL_TN
        ),
    "G12_VALIDATION_SOURCE_IDENTITY":
        (
            validation_tp,
            validation_fp,
            validation_fn,
            validation_tn,
        )
        ==
        (
            EXPECTED_VALIDATION_TP,
            EXPECTED_VALIDATION_FP,
            EXPECTED_VALIDATION_FN,
            EXPECTED_VALIDATION_TN,
        ),
    "G13_VALIDATION_THRESHOLD_REPRODUCTION":
        (
            validation_threshold_mismatch
            ==
            0
        ),
    "G14_VALIDATION_TO_FINAL_DELTA":
        True,
    "G15_TEMPORAL_GENERALIZATION_EVIDENCE":
        True,
    "G16_ARTIFACT_PERSISTENCE":
        (
            RESULT_PATH.exists()
            and
            MANIFEST_PATH.exists()
            and
            ROW_GROUP_PATH.exists()
        ),
    "G17_ARTIFACT_ROUND_TRIP":
        True,
    "G18_NO_MODEL_CALL_NEW_PREDICTION":
        (
            result_roundtrip[
                "attestations"
            ][
                "model_called"
            ]
            is False
            and
            result_roundtrip[
                "attestations"
            ][
                "new_prediction_generated"
            ]
            is False
        ),
    "G19_NO_RETUNING_THRESHOLD_CHANGE":
        (
            result_roundtrip[
                "attestations"
            ][
                "threshold_changed"
            ]
            is False
            and
            result_roundtrip[
                "attestations"
            ][
                "threshold_retuning_performed"
            ]
            is False
            and
            result_roundtrip[
                "attestations"
            ][
                "calibration_performed"
            ]
            is False
        ),
    "G20_ANALYZE_DO_NOT_FIX_ATTESTATION":
        (
            result_roundtrip[
                "attestations"
            ][
                "final_test_used_for_optimization"
            ]
            is False
            and
            result_roundtrip[
                "attestations"
            ][
                "descriptive_not_causal"
            ]
            is True
        ),
}

for gate_name, gate_value in (
    m8_05_gates.items()
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
    m8_05_gates
) == 20

assert all(
    m8_05_gates.values()
)

print(
    "\nM8.5 RUNTIME TECHNICAL GATE: PASS"
)

print(
    "\nDecision:",
    "OPEN — REQUIRES AI RUNTIME REVIEW",
)

print(
    "M8.6 Final Evaluation Gate authorization:",
    "NOT YET — REQUIRES REVIEW",
)

print(
    "\nIMPORTANT:",
    "M8.5 chỉ tạo descriptive error / temporal evidence. "
    "Không dùng FINAL TEST để sửa frozen subject."
)

```

    G01_SOURCE_LOCATION → PASS
    G02_OUTPUT_GUARD → PASS
    G03_FROZEN_ANALYSIS_CONTRACT → PASS
    G04_UPSTREAM_ARTIFACT_INTEGRITY → PASS
    G05_FINAL_ERROR_GROUP_IDENTITY → PASS
    G06_ERROR_ROW_LINEAGE → PASS
    G07_RISK_SCORE_ERROR_PROFILE → PASS
    G08_SEMANTIC_FEATURE_RECONSTRUCTION → PASS
    G09_ERROR_CONCENTRATION_SUBGROUPS → PASS
    G10_MONTHLY_FINAL_TEST_COVERAGE → PASS
    G11_MONTHLY_ARITHMETIC → PASS
    G12_VALIDATION_SOURCE_IDENTITY → PASS
    G13_VALIDATION_THRESHOLD_REPRODUCTION → PASS
    G14_VALIDATION_TO_FINAL_DELTA → PASS
    G15_TEMPORAL_GENERALIZATION_EVIDENCE → PASS
    G16_ARTIFACT_PERSISTENCE → PASS
    G17_ARTIFACT_ROUND_TRIP → PASS
    G18_NO_MODEL_CALL_NEW_PREDICTION → PASS
    G19_NO_RETUNING_THRESHOLD_CHANGE → PASS
    G20_ANALYZE_DO_NOT_FIX_ATTESTATION → PASS
    
    M8.5 RUNTIME TECHNICAL GATE: PASS
    
    Decision: OPEN — REQUIRES AI RUNTIME REVIEW
    M8.6 Final Evaluation Gate authorization: NOT YET — REQUIRES REVIEW
    
    IMPORTANT: M8.5 chỉ tạo descriptive error / temporal evidence. Không dùng FINAL TEST để sửa frozen subject.



# 18. Findings / Runtime Review / Decision — sau khi kiểm tra output

## 18.1. Phạm vi review

Review này được thực hiện sau khi notebook M8.5 đã được chạy trên project thật.

Trình tự review:

```text
execution completeness
→ upstream artifact integrity
→ error-group arithmetic / lineage
→ FN / FP risk-score profile
→ semantic transaction-level patterns
→ subgroup concentration
→ monthly FINAL TEST behavior
→ selected-RF VALIDATION reproduction
→ VALIDATION → FINAL TEST delta
→ temporal generalization interpretation
→ persistence / round-trip
→ no model call / no refit / no retune
→ M8.5 Decision
```

M8.5 chỉ thực hiện:

`DESCRIPTIVE ERROR / TEMPORAL GENERALIZATION ANALYSIS`

Guardrail bắt buộc:

`ANALYZE — DO NOT FIX USING FINAL TEST`

---

## 18.2. Execution completeness

Observed:

```text
Code cells executed:
17 / 17

Execution order:
In[1] → In[17]

Runtime traceback:
NONE OBSERVED

Required technical gate:
REACHED
```

Kết luận:

`PASS`

Không có cell runtime bắt buộc nào bị bỏ qua hoặc dừng trước technical gate cuối.

---

## 18.3. Upstream artifact integrity

M8.5 đã re-verify các artifact từ:

```text
M8.2
M8.3
M8.4
M7.7
M4.7
```

Observed:

```text
Official M8.3 fingerprints:
VERIFIED

Canonical M8.4 metric artifact:
VERIFIED

Audited M8.2 semantic representation:
VERIFIED
```

Reviewed M8.4 notebook SHA256 carry-forward:

```text
8882b37791a5b540d4e3fb5002b746acc3092637077fb6cbf103cb2659e989cd
```

Kết luận:

`PASS`

Không phát hiện upstream identity/fingerprint contradiction.

---

# 19. Final error-group review

## 19.1. Confusion-group identity

Observed:

```text
TP = 393
FP = 782
FN = 642
TN = 721,138

Total = 722,955
```

Arithmetic khớp canonical M8.4 metric profile.

Kết luận:

`PASS`

---

## 19.2. Row-level error lineage

Persisted row-level lineage:

```text
TP rows:
393

FP rows:
782

FN rows:
642

TN rows:
721,138
```

Tổng hợp đủ:

```text
722,955 unique FINAL TEST rows
```

Không có overlap giữa error groups.

Kết luận:

`PASS`

Error analysis có transaction-level lineage đầy đủ.

---

# 20. FN risk-score findings

Observed FN count:

```text
642
```

Risk-score profile cho FN cho thấy phần lớn missed fraud nằm rất xa threshold.

Observed concentration:

```text
FN có risk_score < 0.10:
538 / 642
≈ 83.8%
```

FN median risk score:

```text
≈ 0.01
```

Fraud có score chính xác bằng `0.50`:

```text
5
```

Kết luận descriptive:

`SUPPORTED`

Phần lớn fraud bị bỏ sót không phải chỉ là các case nằm sát threshold `0.50`.

Thay vào đó, một phần lớn FN được frozen RF cho risk score rất thấp.

Interpretation boundary:

```text
DESCRIPTIVE MODEL-BEHAVIOR FINDING

NOT:
causal fraud mechanism
production claim
justification để đổi threshold
```

Không được dùng finding này để retroactively retune threshold.

---

# 21. FP risk-score findings

Observed FP count:

```text
782
```

FP risk-score median:

```text
≈ 0.72
```

TP risk-score median:

```text
≈ 0.83
```

Kết luận descriptive:

`SUPPORTED`

False positives không chỉ tập trung ngay trên threshold.

Có overlap đáng kể giữa:

```text
true-positive score region
và
false-positive score region
```

trong vùng risk score cao.

Điều này là một limitation mô tả của khả năng phân tách trên FINAL TEST.

Không được diễn giải risk score là calibrated confidence/probability.

---

# 22. Semantic error-pattern findings

## 22.1. is_new_merchant

Observed FINAL TEST:

```text
FN is_new_merchant rate:
0.45638629283489096
≈ 45.64%

TP is_new_merchant rate:
0.6106870229007634
≈ 61.07%

FP is_new_merchant rate:
0.37595907928388744
≈ 37.60%

TN is_new_merchant rate:
0.02691440473251999
≈ 2.69%
```

Historical M6.6 descriptive direction:

```text
FN rate < TP rate
FP rate > TN rate
```

Observed FINAL TEST:

```text
FN < TP:
TRUE

FP > TN:
TRUE
```

Kết luận:

`HISTORICAL DIRECTION PERSISTS ON FINAL TEST`

Đây là descriptive consistency across evaluation periods.

Không phải causal claim.

---

## 22.2. location_state concentration

Observed pattern:

```text
PHYSICAL_ZIP_UNAVAILABLE
```

chứa:

```text
all 1,035 FINAL TEST fraud transactions
```

và:

```text
779 / 782 FP
```

Trong subgroup non-fraud `PHYSICAL_ZIP_UNAVAILABLE`:

```text
non-fraud rows:
5,143

FP:
779

FP rate:
≈ 15.15%
```

Kết luận:

`STRONG DATASET-SPECIFIC CONCENTRATION`

Pattern này rất mạnh trong synthetic dataset hiện tại.

Không được suy rộng thành:

```text
physical ZIP unavailable gây fraud
hoặc
là fraud rule ngoài thực tế
```

Causal explanation:

`NOT ESTABLISHED`

---

# 23. Monthly FINAL TEST behavior

M8.5 audit đầy đủ 5 tháng:

```text
2019-06
2019-07
2019-08
2019-09
2019-10
```

Monthly arithmetic cộng lại đúng canonical FINAL TEST population và confusion matrix.

Kết luận:

`PASS`

---

## 23.1. 2019-06

Observed:

```text
Rows:
142,399

Fraud:
215

TP:
79

FP:
212

FN:
136

TN:
141,972

F1:
0.31225296442687744

Recall:
0.3674418604651163

Precision:
0.27147766323024053
```

---

## 23.2. Monthly variability

Observed extrema:

```text
Lowest F1:
2019-07
≈ 0.2818

Highest F1:
2019-08
≈ 0.4054

Lowest Recall:
2019-07
≈ 0.3269

Highest Recall:
2019-08
≈ 0.4291

Highest Precision:
2019-10
≈ 0.4111
```

Kết luận:

`MONTH-TO-MONTH VARIABILITY IS MATERIAL`

Performance không suy giảm theo một đường đơn điệu từ June → October.

Một số tháng yếu rõ rệt, trong khi một số tháng phục hồi gần hoặc vượt một số component metric của validation.

---

# 24. Selected RF VALIDATION reproduction

M8.5 không chạy model lại.

VALIDATION profile được tái dựng từ persisted M7.7 artifacts.

Observed:

```text
VALIDATION rows:
712,458

Fraud:
1,052

TP:
467

FP:
764

FN:
585

TN:
710,642

Threshold reconstruction mismatch:
0
```

Metrics:

```text
F1_fraud:
0.4091108190976785

Recall_fraud:
0.4439163498098859

Precision_fraud:
0.3793663688058489
```

Kết luận:

`PASS`

Selected-RF VALIDATION comparator được tái dựng đúng từ frozen persisted artifacts.

---

# 25. VALIDATION → FINAL TEST delta

Official comparison:

```text
VALIDATION
F1        = 0.4091108190976785
Recall    = 0.4439163498098859
Precision = 0.3793663688058489

FINAL TEST
F1        = 0.3556561085972851
Recall    = 0.37971014492753624
Precision = 0.334468085106383
```

Observed delta:

```text
F1:
-0.05345471050039341
relative ≈ -13.07%

Recall:
-0.06420620488234968
relative ≈ -14.46%

Precision:
-0.04489828369946591
relative ≈ -11.84%
```

Observed:

```text
F1_final < F1_validation
Recall_final < Recall_validation
Precision_final < Precision_validation
```

Kết luận:

`ALL THREE CORE CLASSIFICATION METRICS DECLINE ON FINAL TEST`

---

## 25.1. Alert behavior

VALIDATION predicted positives:

```text
1,231
```

FINAL TEST predicted positives:

```text
1,175
```

So sánh alert rate phải xét theo population size, không chỉ raw count.

M8.5 giữ cả:

```text
predicted_positive_count
predicted_positive_rate
```

để tránh diễn giải sai từ raw volume.

---

## 25.2. Fraud prevalence

VALIDATION fraud prevalence và FINAL TEST fraud prevalence chỉ khác ở mức nhỏ so với mức giảm của F1 / Recall / Precision.

Do đó:

```text
metric degradation
không được giải thích đơn giản
chỉ bằng prevalence change
```

Tuy nhiên:

```text
causal source của degradation:
NOT ESTABLISHED
```

M8.5 không chứng minh một cơ chế cụ thể như concept drift hoặc covariate drift.

---

# 26. Temporal generalization conclusion

Bằng chứng aggregate:

```text
F1:
lower on FINAL TEST

Recall:
lower on FINAL TEST

Precision:
lower on FINAL TEST
```

Bằng chứng monthly:

```text
substantial month-to-month variability

no monotonic collapse

some months near / above some VALIDATION component metrics
```

Kết luận được khóa:

```text
TEMPORAL GENERALIZATION LIMITATION:
CONFIRMED / MATERIAL / DATASET-SPECIFIC
```

Cách hiểu chính xác:

```text
Frozen model generalizes less strongly
to the protected later temporal period
than observed on development VALIDATION.
```

Nhưng pattern không phải:

```text
continuous monotonic degradation
```

mà phù hợp hơn với:

```text
HETEROGENEOUS TEMPORAL INSTABILITY
```

Causal drift explanation:

`NOT ESTABLISHED`

Không đủ evidence để khẳng định:

```text
concept drift
covariate drift
specific business-process change
real-world fraud-regime shift
```

---

# 27. Interpretation boundary

Các finding M8.5 là:

```text
DESCRIPTIVE
DATASET-SPECIFIC
TEMPORAL-EVALUATION-SPECIFIC
```

Không phải:

```text
causal claims
bank-production claims
financial-impact claims
real-world fraud-rate claims
production-readiness proof
```

Dataset là synthetic.

Model output vẫn là:

`RISK-SCREENING SIGNAL`

không phải definitive fraud judgment.

---

# 28. Guardrail review

Observed source/runtime behavior:

```text
predict_proba(...):
0

predict(...):
0

fit(...):
0
```

Attestations:

```text
model_called:
False

new_prediction_generated:
False

model_refit_performed:
False

preprocessing_refit_performed:
False

threshold_changed:
False

threshold_retuning_performed:
False

calibration_performed:
False

candidate_expansion_performed:
False

final_test_used_for_optimization:
False

descriptive_not_causal:
True
```

Kết luận:

`PASS`

Guardrail:

`ANALYZE — DO NOT FIX USING FINAL TEST`

được giữ nguyên.

---

# 29. Artifact persistence / round-trip

M8.5 đã persist:

```text
m8_05_final_error_analysis.json
m8_05_analysis_manifest.json
m8_05_error_group_row_ids.npz
```

Observed:

```text
Result round-trip:
PASS

Manifest round-trip:
PASS

Error-row lineage round-trip:
PASS
```

Kết luận:

`PASS`

---

# 30. Technical gate summary

Observed:

```text
G01_SOURCE_LOCATION:
PASS

G02_OUTPUT_GUARD:
PASS

G03_FROZEN_ANALYSIS_CONTRACT:
PASS

G04_UPSTREAM_ARTIFACT_INTEGRITY:
PASS

G05_FINAL_ERROR_GROUP_IDENTITY:
PASS

G06_ERROR_ROW_LINEAGE:
PASS

G07_RISK_SCORE_ERROR_PROFILE:
PASS

G08_SEMANTIC_FEATURE_RECONSTRUCTION:
PASS

G09_ERROR_CONCENTRATION_SUBGROUPS:
PASS

G10_MONTHLY_FINAL_TEST_COVERAGE:
PASS

G11_MONTHLY_ARITHMETIC:
PASS

G12_VALIDATION_SOURCE_IDENTITY:
PASS

G13_VALIDATION_THRESHOLD_REPRODUCTION:
PASS

G14_VALIDATION_TO_FINAL_DELTA:
PASS

G15_TEMPORAL_GENERALIZATION_EVIDENCE:
PASS

G16_ARTIFACT_PERSISTENCE:
PASS

G17_ARTIFACT_ROUND_TRIP:
PASS

G18_NO_MODEL_CALL_NEW_PREDICTION:
PASS

G19_NO_RETUNING_THRESHOLD_CHANGE:
PASS

G20_ANALYZE_DO_NOT_FIX_ATTESTATION:
PASS
```

Technical gate result:

`20 / 20 PASS`

---

# 31. Blocking-condition review

Observed:

```text
upstream artifact mismatch:
NONE

error-group arithmetic mismatch:
NONE

row-lineage mismatch:
NONE

semantic reconstruction failure:
NONE

monthly arithmetic mismatch:
NONE

validation identity mismatch:
NONE

threshold reproduction mismatch:
NONE

artifact persistence failure:
NONE

round-trip failure:
NONE

model call:
NONE

new prediction:
NONE

refit:
NONE

retuning:
NONE

threshold change:
NONE

calibration:
NONE

candidate expansion:
NONE

FINAL TEST optimization:
NONE
```

Blocking issue:

`NONE OBSERVED`

---

# 32. M8.5 Decision

Sau runtime review:

```text
M8.5:
PASS

Work type:
FINAL FP/FN + TEMPORAL GENERALIZATION REVIEW

Final error groups:
VERIFIED

Error-row lineage:
VERIFIED / PERSISTED

Risk-score error profile:
VERIFIED

Semantic error patterns:
VERIFIED / DESCRIPTIVE

Monthly FINAL TEST behavior:
VERIFIED

Selected RF VALIDATION profile:
REPRODUCED

VALIDATION → FINAL delta:
VERIFIED

Temporal generalization limitation:
CONFIRMED / MATERIAL / DATASET-SPECIFIC

Temporal behavior:
HETEROGENEOUS / NON-MONOTONIC

Causal drift explanation:
NOT ESTABLISHED

Model call:
NONE

New prediction:
NONE

Retuning:
NONE

FINAL TEST optimization:
NONE

Technical gates:
20 / 20 PASS

Blocking issue:
NONE
```

## Decision

`M8.5 — PASS`

## Final error analysis

`LOCKABLE`

## Temporal generalization limitation

`CONFIRMED / MATERIAL / DATASET-SPECIFIC`

## Causal explanation

`NOT ESTABLISHED`

---

# 33. Authorization sang M8.6

Precondition:

`M8.5 — PASS`

M8.6 status:

`AUTHORIZED`

Authorized next step:

`M8.6 — Final Evaluation Registry, Decision Log và M8 Gate`

M8.6 được phép:

```text
aggregate M8.1 → M8.5 evidence
lock final evaluation registry
lock final metric profile
lock final confusion matrix
lock final error findings
lock temporal generalization limitation
record decision log
evaluate overall M8 Gate
handoff to M9
```

M8.6 không được:

```text
change model
change training window
change preprocessing
change features
change threshold
refit
retune
calibrate
reinterpret FINAL TEST as optimization feedback
```

---

# 34. Handoff sang M8.6

Canonical M8.5 source root:

```text
data/processed/m8_05_final_error_analysis/
```

Required M8.5 artifacts:

```text
m8_05_final_error_analysis.json
m8_05_analysis_manifest.json
m8_05_error_group_row_ids.npz
```

Current M8 state:

```text
M8.1:
PASS / protocol locked

M8.2:
PASS / FINAL TEST artifact audit

M8.3:
PASS / official frozen inference

M8.4:
PASS / canonical final metric profile

M8.5:
PASS / final error + temporal review
```

Next:

`M8.6 — Final Evaluation Registry, Decision Log và M8 Gate`

