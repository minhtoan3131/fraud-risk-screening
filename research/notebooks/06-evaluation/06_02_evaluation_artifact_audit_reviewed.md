# M6.2 — Evaluation artifact audit và independent metric reconstruction

Milestone:

`M6 — Evaluation + Error Analysis`

Substep:

`M6.2 — Evaluation Artifact Audit + Independent Metric Reconstruction`

Work type:

`RUNTIME AUDIT / INTEGRITY / RECONSTRUCTION`

Mục tiêu:

> Xác minh rằng toàn bộ persisted evaluation evidence từ M5 đủ integrity để M6 đánh giá độc lập mà không retrain model.

M6.2 không:

- fit classifier;
- retrain baseline;
- tuning;
- resampling;
- threshold optimization;
- model selection;
- training-window selection;
- FINAL TEST access.

M6.2 phải audit:

```text
M4.7:
y_validation
row_id_validation
manifest

M5:
6 y_pred
6 risk_score
6 summary JSON
3 config locks
3 pair manifests
```

Sau đó:

```text
y_validation + persisted y_pred
        ↓
independent metric reconstruction
        ↓
compare with M5 persisted summaries
        ↓
persist M6 Evaluation Registry
```

Runtime-dependent status trước khi chạy:

`NOT YET VERIFIED`

## 1. Contract kế thừa

M6.2 kế thừa:

- `CANON-Kế hoạch Milestone 6 — Evaluation và Error Analysis`;
- `CANON-M6.1 — Evaluation Charter / Scope / Guardrails`;
- `CANON-M5.6 — Baseline Model Registry / M5 Gate`;
- canonical metric strategy từ M3.4;
- controlled training-window protocol từ M3.6.

Canonical evaluation population:

```text
VALIDATION:
2019-01-01 <= Timestamp < 2019-06-01

Rows:
712,458

Fraud:
1,052

Positive class:
fraud = 1
```

M6.2 phải tái tính metric từ prediction artifact thật; không chỉ copy metric trong summary.

## 2. Official evaluation subjects

```text
Logistic Regression
M5-LR-SHORT-B04
M5-LR-LONG-B04

Decision Tree
M5-DT-SHORT-B01
M5-DT-LONG-B01

Random Forest
M5-RF-SHORT-B01
M5-RF-LONG-B01
```

Expected result của M6.2 sau runtime review:

```text
6 / 6 prediction artifacts:
VERIFIED

6 / 6 risk-score artifacts:
VERIFIED

6 / 6 summaries:
VERIFIED

Independent metric reconstruction:
PASS

Summary agreement:
PASS

Validation lineage:
VERIFIED

FINAL TEST:
PROTECTED
```

## 3. Audit flow

```text
environment
    ↓
locate project root
    ↓
load M4.7 manifest + validation target + row_id
    ↓
audit validation identity / lineage
    ↓
locate M5 family artifacts
    ↓
audit config locks + pair manifests
    ↓
load 6 predictions + 6 risk scores + 6 summaries
    ↓
array integrity
    ↓
experiment identity
    ↓
independent confusion / metric reconstruction
    ↓
compare with persisted summaries
    ↓
controlled-pair metadata check
    ↓
persist Evaluation Registry
    ↓
round-trip
    ↓
FINAL TEST isolation
    ↓
M6.2 technical gate
    ↓
runtime review
```


```python

from pathlib import Path
from typing import Any, Dict
import hashlib
import json
import platform
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
        "Thiếu scikit-learn. Cài scikit-learn, "
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


## 4. Locate canonical M4/M5 artifact directories


```python

M4_REL = (
    Path("data")
    / "processed"
    / "m4_07_baseline_ready"
)

M5_LR_REL = (
    Path("data")
    / "processed"
    / "m5_03_logistic_regression_baseline_b04"
)

M5_DT_REL = (
    Path("data")
    / "processed"
    / "m5_04_decision_tree_baseline"
)

M5_RF_REL = (
    Path("data")
    / "processed"
    / "m5_05_random_forest_baseline"
)

M6_02_REL = (
    Path("data")
    / "processed"
    / "m6_02_evaluation_artifact_audit"
)


candidate_roots = [
    Path.cwd(),
    *list(Path.cwd().parents)[:6],
]

PROJECT_ROOT = None

required_rel_paths = [
    M4_REL / "manifest.json",
    M4_REL / "y_validation.npy",
    M4_REL / "row_id_validation.npy",
    M5_LR_REL / "m5_03_lr_b04_pair_manifest.json",
    M5_DT_REL / "m5_04_dt_pair_manifest.json",
    M5_RF_REL / "m5_05_rf_pair_manifest.json",
]


for candidate in candidate_roots:
    candidate = candidate.resolve()

    if all(
        (candidate / rel_path).exists()
        for rel_path
        in required_rel_paths
    ):
        PROJECT_ROOT = candidate
        break


if PROJECT_ROOT is None:
    raise FileNotFoundError(
        "Không tìm thấy PROJECT_ROOT chứa đầy đủ "
        "M4.7 validation artifacts và M5 pair manifests."
    )


M4_DIR = PROJECT_ROOT / M4_REL
M5_LR_DIR = PROJECT_ROOT / M5_LR_REL
M5_DT_DIR = PROJECT_ROOT / M5_DT_REL
M5_RF_DIR = PROJECT_ROOT / M5_RF_REL

OUTPUT_DIR = PROJECT_ROOT / M6_02_REL
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


print("PROJECT_ROOT:")
print(PROJECT_ROOT)

print("\nM4_DIR:")
print(M4_DIR)

print("\nM5_LR_DIR:")
print(M5_LR_DIR)

print("\nM5_DT_DIR:")
print(M5_DT_DIR)

print("\nM5_RF_DIR:")
print(M5_RF_DIR)

print("\nOUTPUT_DIR:")
print(OUTPUT_DIR)

```

    PROJECT_ROOT:
    /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening
    
    M4_DIR:
    /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/processed/m4_07_baseline_ready
    
    M5_LR_DIR:
    /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/processed/m5_03_logistic_regression_baseline_b04
    
    M5_DT_DIR:
    /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/processed/m5_04_decision_tree_baseline
    
    M5_RF_DIR:
    /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/processed/m5_05_random_forest_baseline
    
    OUTPUT_DIR:
    /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/processed/m6_02_evaluation_artifact_audit


## 5. Canonical constants và official run specification


```python

EXPECTED_VALIDATION_ROWS = 712_458
EXPECTED_VALIDATION_FRAUD = 1_052
EXPECTED_FEATURE_COUNT = 47

M4_PIPELINE_VERSION = "M4.7-baseline-v1"
M5_RUNNER_VERSION = "M5.2-shared-runner-v1"

FEATURE_VERSION = "Feature Specification v1.0"
PREPROCESSING_VERSION = "Preprocessing Specification v1.0"
MATRIX_SCHEMA_VERSION = "Baseline Matrix Schema v1.0"

IMBALANCE_STRATEGY = "NONE"
THRESHOLD_POLICY = "DEFAULT_MODEL_DECISION_RULE"


OFFICIAL_RUNS = {
    "M5-LR-SHORT-B04": {
        "family": "Logistic Regression",
        "model_id": "SKLEARN_LOGISTIC_REGRESSION",
        "config_id": "LR-B04-LBFGS-L2-C1",
        "window": "W_SHORT",
        "dir": M5_LR_DIR,
    },
    "M5-LR-LONG-B04": {
        "family": "Logistic Regression",
        "model_id": "SKLEARN_LOGISTIC_REGRESSION",
        "config_id": "LR-B04-LBFGS-L2-C1",
        "window": "W_LONG",
        "dir": M5_LR_DIR,
    },
    "M5-DT-SHORT-B01": {
        "family": "Decision Tree",
        "model_id": "SKLEARN_DECISION_TREE_CLASSIFIER",
        "config_id": "DT-B01-DEFAULT-GINI-UNPRUNED",
        "window": "W_SHORT",
        "dir": M5_DT_DIR,
    },
    "M5-DT-LONG-B01": {
        "family": "Decision Tree",
        "model_id": "SKLEARN_DECISION_TREE_CLASSIFIER",
        "config_id": "DT-B01-DEFAULT-GINI-UNPRUNED",
        "window": "W_LONG",
        "dir": M5_DT_DIR,
    },
    "M5-RF-SHORT-B01": {
        "family": "Random Forest",
        "model_id": "SKLEARN_RANDOM_FOREST_CLASSIFIER",
        "config_id": "RF-B01-100-GINI-SQRT-BOOTSTRAP",
        "window": "W_SHORT",
        "dir": M5_RF_DIR,
    },
    "M5-RF-LONG-B01": {
        "family": "Random Forest",
        "model_id": "SKLEARN_RANDOM_FOREST_CLASSIFIER",
        "config_id": "RF-B01-100-GINI-SQRT-BOOTSTRAP",
        "window": "W_LONG",
        "dir": M5_RF_DIR,
    },
}


FAMILY_ARTIFACTS = {
    "Logistic Regression": {
        "dir": M5_LR_DIR,
        "config_lock":
            "m5_03_lr_b04_config_lock.json",
        "pair_manifest":
            "m5_03_lr_b04_pair_manifest.json",
        "config_id":
            "LR-B04-LBFGS-L2-C1",
    },
    "Decision Tree": {
        "dir": M5_DT_DIR,
        "config_lock":
            "m5_04_dt_baseline_config_lock.json",
        "pair_manifest":
            "m5_04_dt_pair_manifest.json",
        "config_id":
            "DT-B01-DEFAULT-GINI-UNPRUNED",
    },
    "Random Forest": {
        "dir": M5_RF_DIR,
        "config_lock":
            "m5_05_rf_baseline_config_lock.json",
        "pair_manifest":
            "m5_05_rf_pair_manifest.json",
        "config_id":
            "RF-B01-100-GINI-SQRT-BOOTSTRAP",
    },
}


assert len(OFFICIAL_RUNS) == 6
assert len(FAMILY_ARTIFACTS) == 3

print("Official runs:")
for experiment_id in OFFICIAL_RUNS:
    print(" -", experiment_id)

print("\nM6.2 OFFICIAL SUBJECT SPEC: LOCKED")

```

    Official runs:
     - M5-LR-SHORT-B04
     - M5-LR-LONG-B04
     - M5-DT-SHORT-B01
     - M5-DT-LONG-B01
     - M5-RF-SHORT-B01
     - M5-RF-LONG-B01
    
    M6.2 OFFICIAL SUBJECT SPEC: LOCKED


## 6. Load M4.7 validation target / lineage / manifest


```python

with open(
    M4_DIR / "manifest.json",
    "r",
    encoding="utf-8",
) as file:
    m4_manifest = json.load(file)


y_validation = np.load(
    M4_DIR / "y_validation.npy",
    allow_pickle=False,
)

row_id_validation = np.load(
    M4_DIR / "row_id_validation.npy",
    allow_pickle=False,
)


assert (
    m4_manifest["pipeline_version"]
    == M4_PIPELINE_VERSION
)

assert (
    m4_manifest["output_width"]
    == EXPECTED_FEATURE_COUNT
)

assert m4_manifest["target_dtype"] == "int8"
assert m4_manifest["final_test_used"] is False


print("M4 pipeline:")
print(m4_manifest["pipeline_version"])

print("\ny_validation:")
print(
    y_validation.shape,
    y_validation.dtype,
)

print("\nrow_id_validation:")
print(
    row_id_validation.shape,
    row_id_validation.dtype,
)

```

    M4 pipeline:
    M4.7-baseline-v1
    
    y_validation:
    (712458,) int8
    
    row_id_validation:
    (712458,) int64


## 7. Validation target và lineage integrity


```python

assert y_validation.ndim == 1
assert len(y_validation) == EXPECTED_VALIDATION_ROWS
assert y_validation.dtype == np.int8

assert (
    int(y_validation.sum())
    == EXPECTED_VALIDATION_FRAUD
)

assert (
    set(
        np.unique(
            y_validation
        ).tolist()
    )
    <= {0, 1}
)


assert row_id_validation.ndim == 1
assert (
    len(row_id_validation)
    == EXPECTED_VALIDATION_ROWS
)

assert (
    len(
        np.unique(
            row_id_validation
        )
    )
    == EXPECTED_VALIDATION_ROWS
), (
    "row_id_validation phải unique để làm lineage anchor."
)


print(
    "Validation rows:",
    len(y_validation),
)

print(
    "Validation fraud:",
    int(y_validation.sum()),
)

print(
    "Unique row IDs:",
    len(
        np.unique(
            row_id_validation
        )
    ),
)

print(
    "\nM6.2 VALIDATION TARGET / LINEAGE GATE: PASS"
)

```

    Validation rows: 712458
    Validation fraud: 1052
    Unique row IDs: 712458
    
    M6.2 VALIDATION TARGET / LINEAGE GATE: PASS


## 8. Audit config locks và pair manifests

Mục tiêu:

- đúng family/config identity;
- no FINAL TEST access;
- pair vẫn giữ W_SHORT/W_LONG;
- winner vẫn OPEN ở M5 handoff;
- không phát hiện config/pair artifact bị trộn.


```python

family_metadata = {}


for family, spec in (
    FAMILY_ARTIFACTS.items()
):
    family_dir = spec["dir"]

    config_path = (
        family_dir
        / spec["config_lock"]
    )

    pair_path = (
        family_dir
        / spec["pair_manifest"]
    )

    assert config_path.exists()
    assert pair_path.exists()

    with open(
        config_path,
        "r",
        encoding="utf-8",
    ) as file:
        config_lock = json.load(file)

    with open(
        pair_path,
        "r",
        encoding="utf-8",
    ) as file:
        pair_manifest = json.load(file)

    assert (
        config_lock["model_config_id"]
        == spec["config_id"]
    )

    assert (
        pair_manifest["model_config_id"]
        == spec["config_id"]
    )

    assert (
        config_lock["model_family"]
        == family
    )

    assert (
        pair_manifest["model_family"]
        == family
    )

    assert (
        config_lock["final_test_access"]
        is False
    )

    assert (
        pair_manifest["final_test_accessed"]
        is False
    )

    assert (
        pair_manifest[
            "training_window_winner"
        ]
        == "OPEN"
    )

    family_metadata[family] = {
        "config_lock":
            config_lock,
        "pair_manifest":
            pair_manifest,
        "config_path":
            config_path,
        "pair_path":
            pair_path,
    }


print("Families audited:")
for family in family_metadata:
    print(" -", family)


print(
    "\nM6.2 CONFIG LOCK / PAIR MANIFEST GATE: PASS"
)

```

    Families audited:
     - Logistic Regression
     - Decision Tree
     - Random Forest
    
    M6.2 CONFIG LOCK / PAIR MANIFEST GATE: PASS


## 9. Verify pair-manifest run identities


```python

def get_pair_experiment_ids(
    pair_manifest,
):
    runs = pair_manifest["runs"]

    short_value = runs["W_SHORT"]
    long_value = runs["W_LONG"]

    if isinstance(
        short_value,
        dict,
    ):
        short_id = short_value[
            "experiment_id"
        ]
        long_id = long_value[
            "experiment_id"
        ]
    else:
        short_id = short_value
        long_id = long_value

    return {
        "W_SHORT": short_id,
        "W_LONG": long_id,
    }


expected_pair_ids = {
    "Logistic Regression": {
        "W_SHORT":
            "M5-LR-SHORT-B04",
        "W_LONG":
            "M5-LR-LONG-B04",
    },
    "Decision Tree": {
        "W_SHORT":
            "M5-DT-SHORT-B01",
        "W_LONG":
            "M5-DT-LONG-B01",
    },
    "Random Forest": {
        "W_SHORT":
            "M5-RF-SHORT-B01",
        "W_LONG":
            "M5-RF-LONG-B01",
    },
}


for family, metadata in (
    family_metadata.items()
):
    actual_ids = (
        get_pair_experiment_ids(
            metadata[
                "pair_manifest"
            ]
        )
    )

    assert (
        actual_ids
        == expected_pair_ids[
            family
        ]
    ), (
        f"Pair-manifest run mismatch: "
        f"{family}"
    )


print(
    "M6.2 PAIR RUN IDENTITY GATE: PASS"
)

```

    M6.2 PAIR RUN IDENTITY GATE: PASS


## 10. Canonical metric implementation cho independent reconstruction


```python

def compute_canonical_metrics(
    y_true,
    y_pred,
):
    y_true = np.asarray(
        y_true
    )

    y_pred = np.asarray(
        y_pred
    )

    assert y_true.ndim == 1
    assert y_pred.ndim == 1
    assert len(y_true) == len(y_pred)

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
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
    }


print(
    "M6.2 CANONICAL METRIC FUNCTION: DEFINED"
)

```

    M6.2 CANONICAL METRIC FUNCTION: DEFINED


## 11. Artifact fingerprint helper

M6.2 ghi SHA-256 của persisted arrays để Evaluation Registry có identity evidence.

Hash không dùng để đánh giá model quality.

Hash chỉ dùng cho:

`artifact identity / reproducibility`


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


y_validation_sha256 = (
    sha256_file(
        M4_DIR
        / "y_validation.npy"
    )
)

row_id_validation_sha256 = (
    sha256_file(
        M4_DIR
        / "row_id_validation.npy"
    )
)


print("y_validation SHA256:")
print(y_validation_sha256)

print("\nrow_id_validation SHA256:")
print(row_id_validation_sha256)

```

    y_validation SHA256:
    0a21b2e93e017eda8b794be1a94a692beebc5883c1303427bc245299b55f5306
    
    row_id_validation SHA256:
    e7f3074c8604680eccef60ff7be45501f0dc10efbf6ac82e47bcbd1b00e43adc


## 12. Single-run audit function

Audit gồm:

- file existence;
- array shape/dtype/support;
- score range;
- summary identity;
- canonical contract identity;
- final-test flag;
- independent metric reconstruction;
- exact confusion-count agreement;
- numerical metric agreement;
- prediction/risk artifact filename agreement.


```python

FLOAT_METRIC_FIELDS = [
    "f1_fraud",
    "recall_fraud",
    "precision_fraud",
    "accuracy_reference",
    "predicted_positive_rate",
]

COUNT_FIELDS = [
    "tn",
    "fp",
    "fn",
    "tp",
    "predicted_positive_count",
]


def audit_run(
    experiment_id,
    spec,
):
    family_dir = spec["dir"]

    prediction_path = (
        family_dir
        / (
            f"{experiment_id}"
            "__y_pred.npy"
        )
    )

    risk_score_path = (
        family_dir
        / (
            f"{experiment_id}"
            "__risk_score.npy"
        )
    )

    summary_path = (
        family_dir
        / (
            f"{experiment_id}"
            "__summary.json"
        )
    )

    assert prediction_path.exists()
    assert risk_score_path.exists()
    assert summary_path.exists()

    y_pred = np.load(
        prediction_path,
        allow_pickle=False,
    )

    risk_score = np.load(
        risk_score_path,
        allow_pickle=False,
    )

    with open(
        summary_path,
        "r",
        encoding="utf-8",
    ) as file:
        summary = json.load(file)

    # Array integrity.
    assert y_pred.ndim == 1
    assert len(y_pred) == EXPECTED_VALIDATION_ROWS

    assert (
        set(
            np.unique(
                y_pred
            ).tolist()
        )
        <= {0, 1}
    )

    assert np.isfinite(
        y_pred
    ).all()

    assert risk_score.ndim == 1
    assert (
        len(risk_score)
        == EXPECTED_VALIDATION_ROWS
    )

    assert np.isfinite(
        risk_score
    ).all()

    assert np.all(
        risk_score >= 0.0
    )

    assert np.all(
        risk_score <= 1.0
    )

    # Summary identity.
    assert (
        summary["experiment_id"]
        == experiment_id
    )

    assert (
        summary["model_family"]
        == spec["family"]
    )

    assert (
        summary["model_id"]
        == spec["model_id"]
    )

    assert (
        summary["model_config_id"]
        == spec["config_id"]
    )

    assert (
        summary[
            "training_window_id"
        ]
        == spec["window"]
    )

    assert (
        summary["runner_version"]
        == M5_RUNNER_VERSION
    )

    assert (
        summary["feature_version"]
        == FEATURE_VERSION
    )

    assert (
        summary[
            "preprocessing_version"
        ]
        == PREPROCESSING_VERSION
    )

    assert (
        summary[
            "matrix_schema_version"
        ]
        == MATRIX_SCHEMA_VERSION
    )

    assert (
        summary["feature_count"]
        == EXPECTED_FEATURE_COUNT
    )

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
        summary["imbalance_strategy"]
        == IMBALANCE_STRATEGY
    )

    assert (
        summary["threshold_policy"]
        == THRESHOLD_POLICY
    )

    assert (
        summary["risk_score_available"]
        is True
    )

    assert (
        summary["risk_score_kind"]
        == "predict_proba"
    )

    assert (
        summary["final_test_accessed"]
        is False
    )

    assert (
        summary["integrity_gate_result"]
        == "PASS"
    )

    # Persisted filename metadata compatibility.
    #
    # M5.4 / M5.5 summaries persist prediction_file và
    # risk_score_file trực tiếp trong summary JSON.
    #
    # M5.3 Logistic Regression B04 là legacy schema:
    # summary JSON được dump trực tiếp từ output.summary,
    # nên hai metadata field này không tồn tại dù các
    # artifacts đã được persist và round-trip verified.
    #
    # Vì vậy filename metadata là optional ở M6.2.
    # Nếu field tồn tại -> phải khớp chính xác.
    # Nếu field không tồn tại -> canonical filename +
    # file existence + experiment identity + pair manifest
    # vẫn là evidence identity bắt buộc.

    if "prediction_file" in summary:
        assert (
            summary["prediction_file"]
            == prediction_path.name
        )
        prediction_filename_metadata_check = "PASS"
    else:
        prediction_filename_metadata_check = (
            "NOT_PRESENT_IN_SOURCE_SUMMARY_SCHEMA"
        )

    if "risk_score_file" in summary:
        assert (
            summary["risk_score_file"]
            == risk_score_path.name
        )
        risk_score_filename_metadata_check = "PASS"
    else:
        risk_score_filename_metadata_check = (
            "NOT_PRESENT_IN_SOURCE_SUMMARY_SCHEMA"
        )

    # Canonical filenames themselves remain mandatory.
    assert (
        prediction_path.name
        == f"{experiment_id}__y_pred.npy"
    )

    assert (
        risk_score_path.name
        == f"{experiment_id}__risk_score.npy"
    )

    assert (
        summary_path.name
        == f"{experiment_id}__summary.json"
    )

    # Independent metric reconstruction.
    recomputed = (
        compute_canonical_metrics(
            y_validation,
            y_pred,
        )
    )

    for field in COUNT_FIELDS:
        assert (
            recomputed[field]
            == int(
                summary[field]
            )
        ), (
            f"{experiment_id}: "
            f"count mismatch {field}"
        )

    for field in FLOAT_METRIC_FIELDS:
        assert np.isclose(
            recomputed[field],
            float(
                summary[field]
            ),
            rtol=1e-12,
            atol=1e-12,
        ), (
            f"{experiment_id}: "
            f"metric mismatch {field}"
        )

    # Confusion arithmetic.
    assert (
        recomputed["tp"]
        + recomputed["fn"]
        == EXPECTED_VALIDATION_FRAUD
    )

    assert (
        recomputed[
            "predicted_positive_count"
        ]
        == recomputed["tp"]
        + recomputed["fp"]
    )

    assert (
        recomputed["tn"]
        + recomputed["fp"]
        + recomputed["fn"]
        + recomputed["tp"]
        == EXPECTED_VALIDATION_ROWS
    )

    return {
        "experiment_id":
            experiment_id,

        "model_family":
            spec["family"],

        "model_id":
            spec["model_id"],

        "model_config_id":
            spec["config_id"],

        "training_window_id":
            spec["window"],

        "feature_version":
            summary[
                "feature_version"
            ],

        "preprocessing_version":
            summary[
                "preprocessing_version"
            ],

        "matrix_schema_version":
            summary[
                "matrix_schema_version"
            ],

        "validation_period":
            "2019-01-01_to_2019-06-01_exclusive",

        "validation_rows":
            EXPECTED_VALIDATION_ROWS,

        "validation_fraud_rows":
            EXPECTED_VALIDATION_FRAUD,

        "imbalance_strategy":
            summary[
                "imbalance_strategy"
            ],

        "threshold_policy":
            summary[
                "threshold_policy"
            ],

        "prediction_artifact":
            str(
                prediction_path.relative_to(
                    PROJECT_ROOT
                )
            ),

        "risk_score_artifact":
            str(
                risk_score_path.relative_to(
                    PROJECT_ROOT
                )
            ),

        "summary_artifact":
            str(
                summary_path.relative_to(
                    PROJECT_ROOT
                )
            ),

        "prediction_filename_metadata_check":
            prediction_filename_metadata_check,

        "risk_score_filename_metadata_check":
            risk_score_filename_metadata_check,

        "prediction_sha256":
            sha256_file(
                prediction_path
            ),

        "risk_score_sha256":
            sha256_file(
                risk_score_path
            ),

        "f1_fraud":
            recomputed[
                "f1_fraud"
            ],

        "recall_fraud":
            recomputed[
                "recall_fraud"
            ],

        "precision_fraud":
            recomputed[
                "precision_fraud"
            ],

        "accuracy_reference":
            recomputed[
                "accuracy_reference"
            ],

        "tn":
            recomputed["tn"],

        "fp":
            recomputed["fp"],

        "fn":
            recomputed["fn"],

        "tp":
            recomputed["tp"],

        "predicted_positive_count":
            recomputed[
                "predicted_positive_count"
            ],

        "predicted_positive_rate":
            recomputed[
                "predicted_positive_rate"
            ],

        "risk_score_available":
            True,

        "risk_score_kind":
            "predict_proba",

        "independent_metric_check":
            "PASS",

        "summary_match_check":
            "PASS",

        "artifact_integrity_check":
            "PASS",

        "final_test_accessed":
            False,

        "evaluation_status":
            "VERIFIED_FOR_M6_EVALUATION",

        "notes":
            (
                "Independent reconstruction "
                "from canonical y_validation "
                "and persisted y_pred."
            ),
    }

```

### 12.1. Compatibility note — M5.3 summary schema

`M5.3 Logistic Regression B04` dùng persisted summary schema cũ hơn M5.4/M5.5.

Cụ thể:

```text
M5.3 summary JSON:
không bắt buộc có
prediction_file
risk_score_file
```

Trong M5.3, ba artifact vẫn được persist riêng:

```text
{experiment_id}__y_pred.npy
{experiment_id}__risk_score.npy
{experiment_id}__summary.json
```

và M5.3 đã có persistence + round-trip gate riêng.

Do đó M6.2 xử lý hai metadata field này như sau:

```text
Nếu field tồn tại:
phải match canonical filename

Nếu field không tồn tại:
không fail chỉ vì schema legacy;
thay vào đó bắt buộc kiểm tra:
- canonical filename;
- file existence;
- experiment identity;
- pair-manifest identity;
- SHA-256;
- independent metric reconstruction.
```

Đây là compatibility handling, không phải nới lỏng prediction/metric integrity.

# 13. Audit all six official runs


```python

evaluation_registry = []


for experiment_id, spec in (
    OFFICIAL_RUNS.items()
):
    print(
        "=" * 72
    )

    print(
        "Auditing:",
        experiment_id,
    )

    record = audit_run(
        experiment_id,
        spec,
    )

    evaluation_registry.append(
        record
    )

    print(
        "F1:",
        record["f1_fraud"],
    )

    print(
        "Recall:",
        record["recall_fraud"],
    )

    print(
        "Precision:",
        record["precision_fraud"],
    )

    print(
        "TP / FP / FN / TN:",
        record["tp"],
        record["fp"],
        record["fn"],
        record["tn"],
    )

    print(
        "Predicted positive:",
        record[
            "predicted_positive_count"
        ],
    )

    print(
        "Artifact integrity:",
        record[
            "artifact_integrity_check"
        ],
    )

    print(
        "Summary match:",
        record[
            "summary_match_check"
        ],
    )

    print(
        "Filename metadata:",
        (
            record[
                "prediction_filename_metadata_check"
            ],
            record[
                "risk_score_filename_metadata_check"
            ],
        ),
    )


assert len(
    evaluation_registry
) == 6


print(
    "\nM6.2 SIX-RUN ARTIFACT / METRIC AUDIT: PASS"
)

```

    ========================================================================
    Auditing: M5-LR-SHORT-B04
    F1: 0.33747547416612167
    Recall: 0.24524714828897337
    Precision: 0.5408805031446541
    TP / FP / FN / TN: 258 219 794 711187
    Predicted positive: 477
    Artifact integrity: PASS
    Summary match: PASS
    Filename metadata: ('NOT_PRESENT_IN_SOURCE_SUMMARY_SCHEMA', 'NOT_PRESENT_IN_SOURCE_SUMMARY_SCHEMA')
    ========================================================================
    Auditing: M5-LR-LONG-B04
    F1: 0.03996366939146231
    Recall: 0.02091254752851711
    Precision: 0.4489795918367347
    TP / FP / FN / TN: 22 27 1030 711379
    Predicted positive: 49
    Artifact integrity: PASS
    Summary match: PASS
    Filename metadata: ('NOT_PRESENT_IN_SOURCE_SUMMARY_SCHEMA', 'NOT_PRESENT_IN_SOURCE_SUMMARY_SCHEMA')
    ========================================================================
    Auditing: M5-DT-SHORT-B01
    F1: 0.3270564915758176
    Recall: 0.31368821292775667
    Precision: 0.3416149068322981
    TP / FP / FN / TN: 330 636 722 710770
    Predicted positive: 966
    Artifact integrity: PASS
    Summary match: PASS
    Filename metadata: ('PASS', 'PASS')
    ========================================================================
    Auditing: M5-DT-LONG-B01
    F1: 0.19913419913419914
    Recall: 0.1967680608365019
    Precision: 0.20155793573515093
    TP / FP / FN / TN: 207 820 845 710586
    Predicted positive: 1027
    Artifact integrity: PASS
    Summary match: PASS
    Filename metadata: ('PASS', 'PASS')
    ========================================================================
    Auditing: M5-RF-SHORT-B01
    F1: 0.3664670658682635
    Recall: 0.2908745247148289
    Precision: 0.49514563106796117
    TP / FP / FN / TN: 306 312 746 711094
    Predicted positive: 618
    Artifact integrity: PASS
    Summary match: PASS
    Filename metadata: ('PASS', 'PASS')
    ========================================================================
    Auditing: M5-RF-LONG-B01
    F1: 0.15270935960591134
    Recall: 0.08840304182509506
    Precision: 0.5602409638554217
    TP / FP / FN / TN: 93 73 959 711333
    Predicted positive: 166
    Artifact integrity: PASS
    Summary match: PASS
    Filename metadata: ('PASS', 'PASS')
    
    M6.2 SIX-RUN ARTIFACT / METRIC AUDIT: PASS


## 14. Registry-level uniqueness và completeness

Một official experiment chỉ được xuất hiện một lần.

Phải có đúng:

- 3 families;
- 2 windows mỗi family;
- 6 unique experiment IDs.


```python

experiment_ids = [
    record[
        "experiment_id"
    ]
    for record
    in evaluation_registry
]

assert len(
    experiment_ids
) == 6

assert len(
    set(
        experiment_ids
    )
) == 6


families = {
    record[
        "model_family"
    ]
    for record
    in evaluation_registry
}

assert families == {
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
}


for family in families:
    family_records = [
        record
        for record
        in evaluation_registry
        if (
            record[
                "model_family"
            ]
            == family
        )
    ]

    assert (
        len(
            family_records
        )
        == 2
    )

    assert {
        record[
            "training_window_id"
        ]
        for record
        in family_records
    } == {
        "W_SHORT",
        "W_LONG",
    }


print(
    "Families:",
    sorted(
        families
    ),
)

print(
    "\nM6.2 REGISTRY COMPLETENESS GATE: PASS"
)

```

    Families: ['Decision Tree', 'Logistic Regression', 'Random Forest']
    
    M6.2 REGISTRY COMPLETENESS GATE: PASS


## 15. Controlled-pair metadata integrity

M6.2 chưa so metric để chọn window.

Cell này chỉ xác minh controlled metadata của SHORT/LONG pair.


```python

CONTROLLED_FIELDS = [
    "model_family",
    "model_id",
    "model_config_id",
    "feature_version",
    "preprocessing_version",
    "matrix_schema_version",
    "validation_period",
    "validation_rows",
    "validation_fraud_rows",
    "imbalance_strategy",
    "threshold_policy",
    "risk_score_kind",
]


for family in sorted(
    families
):
    family_records = {
        record[
            "training_window_id"
        ]:
        record
        for record
        in evaluation_registry
        if (
            record[
                "model_family"
            ]
            == family
        )
    }

    short_record = (
        family_records[
            "W_SHORT"
        ]
    )

    long_record = (
        family_records[
            "W_LONG"
        ]
    )

    for field in (
        CONTROLLED_FIELDS
    ):
        assert (
            short_record[field]
            == long_record[field]
        ), (
            f"{family}: "
            f"controlled field mismatch "
            f"{field}"
        )


print(
    "Controlled fields checked:",
    len(
        CONTROLLED_FIELDS
    ),
)

print(
    "\nM6.2 CONTROLLED-PAIR METADATA GATE: PASS"
)

```

    Controlled fields checked: 12
    
    M6.2 CONTROLLED-PAIR METADATA GATE: PASS


## 16. Compact independent Evaluation Registry evidence

Đây là evidence output để runtime review.

Không chọn winner.


```python

for record in evaluation_registry:
    print("=" * 72)

    print(
        "Experiment:",
        record[
            "experiment_id"
        ],
    )

    print(
        "Family / Window:",
        record[
            "model_family"
        ],
        "/",
        record[
            "training_window_id"
        ],
    )

    print(
        "F1 / Recall / Precision:",
        record[
            "f1_fraud"
        ],
        record[
            "recall_fraud"
        ],
        record[
            "precision_fraud"
        ],
    )

    print(
        "TP / FP / FN / TN:",
        record["tp"],
        record["fp"],
        record["fn"],
        record["tn"],
    )

    print(
        "Predicted positive:",
        record[
            "predicted_positive_count"
        ],
        record[
            "predicted_positive_rate"
        ],
    )

    print(
        "Independent metric:",
        record[
            "independent_metric_check"
        ],
    )

    print(
        "Summary match:",
        record[
            "summary_match_check"
        ],
    )


print(
    "\nTraining-window winner:",
    "OPEN — NOT SELECTED IN M6.2",
)

print(
    "Model-family winner:",
    "OPEN — NOT SELECTED IN M6.2",
)

```

    ========================================================================
    Experiment: M5-LR-SHORT-B04
    Family / Window: Logistic Regression / W_SHORT
    F1 / Recall / Precision: 0.33747547416612167 0.24524714828897337 0.5408805031446541
    TP / FP / FN / TN: 258 219 794 711187
    Predicted positive: 477 0.0006695131502488568
    Independent metric: PASS
    Summary match: PASS
    ========================================================================
    Experiment: M5-LR-LONG-B04
    Family / Window: Logistic Regression / W_LONG
    F1 / Recall / Precision: 0.03996366939146231 0.02091254752851711 0.4489795918367347
    TP / FP / FN / TN: 22 27 1030 711379
    Predicted positive: 49 6.877598398782806e-05
    Independent metric: PASS
    Summary match: PASS
    ========================================================================
    Experiment: M5-DT-SHORT-B01
    Family / Window: Decision Tree / W_SHORT
    F1 / Recall / Precision: 0.3270564915758176 0.31368821292775667 0.3416149068322981
    TP / FP / FN / TN: 330 636 722 710770
    Predicted positive: 966 0.0013558693986171816
    Independent metric: PASS
    Summary match: PASS
    ========================================================================
    Experiment: M5-DT-LONG-B01
    Family / Window: Decision Tree / W_LONG
    F1 / Recall / Precision: 0.19913419913419914 0.1967680608365019 0.20155793573515093
    TP / FP / FN / TN: 207 820 845 710586
    Predicted positive: 1027 0.0014414884807244777
    Independent metric: PASS
    Summary match: PASS
    ========================================================================
    Experiment: M5-RF-SHORT-B01
    Family / Window: Random Forest / W_SHORT
    F1 / Recall / Precision: 0.3664670658682635 0.2908745247148289 0.49514563106796117
    TP / FP / FN / TN: 306 312 746 711094
    Predicted positive: 618 0.0008674195531526069
    Independent metric: PASS
    Summary match: PASS
    ========================================================================
    Experiment: M5-RF-LONG-B01
    Family / Window: Random Forest / W_LONG
    F1 / Recall / Precision: 0.15270935960591134 0.08840304182509506 0.5602409638554217
    TP / FP / FN / TN: 93 73 959 711333
    Predicted positive: 166 0.000232996190652642
    Independent metric: PASS
    Summary match: PASS
    
    Training-window winner: OPEN — NOT SELECTED IN M6.2
    Model-family winner: OPEN — NOT SELECTED IN M6.2


## 17. Persist independent Evaluation Registry


```python

REGISTRY_VERSION = (
    "M6.2-evaluation-registry-v1"
)

REGISTRY_PATH = (
    OUTPUT_DIR
    / "m6_02_evaluation_registry.json"
)

AUDIT_MANIFEST_PATH = (
    OUTPUT_DIR
    / "m6_02_audit_manifest.json"
)


registry_payload = {
    "registry_version":
        REGISTRY_VERSION,

    "m6_substep":
        "M6.2",

    "evaluation_population":
        "VALIDATION_2019-01_TO_2019-05",

    "validation_rows":
        EXPECTED_VALIDATION_ROWS,

    "validation_fraud_rows":
        EXPECTED_VALIDATION_FRAUD,

    "positive_class":
        1,

    "y_validation_artifact":
        str(
            (
                M4_DIR
                / "y_validation.npy"
            ).relative_to(
                PROJECT_ROOT
            )
        ),

    "row_id_validation_artifact":
        str(
            (
                M4_DIR
                / "row_id_validation.npy"
            ).relative_to(
                PROJECT_ROOT
            )
        ),

    "y_validation_sha256":
        y_validation_sha256,

    "row_id_validation_sha256":
        row_id_validation_sha256,

    "official_run_count":
        6,

    "records":
        evaluation_registry,

    "training_window_winner":
        "OPEN",

    "model_family_winner":
        "OPEN",

    "final_threshold":
        "OPEN",

    "final_test_accessed":
        False,
}


audit_manifest = {
    "m6_substep":
        "M6.2",

    "registry_version":
        REGISTRY_VERSION,

    "official_runs_expected":
        6,

    "official_runs_verified":
        len(
            evaluation_registry
        ),

    "prediction_artifacts_verified":
        6,

    "risk_score_artifacts_verified":
        6,

    "summary_artifacts_verified":
        6,

    "config_locks_verified":
        3,

    "pair_manifests_verified":
        3,

    "independent_metric_reconstruction":
        "PASS",

    "summary_agreement":
        "PASS",

    "validation_lineage":
        "PASS",

    "controlled_pair_metadata":
        "PASS",

    "final_test_accessed":
        False,

    "decision":
        "OPEN — REQUIRES M6.2 RUNTIME REVIEW",
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
    )


with open(
    AUDIT_MANIFEST_PATH,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        audit_manifest,
        file,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )


assert REGISTRY_PATH.exists()
assert AUDIT_MANIFEST_PATH.exists()

assert REGISTRY_PATH.stat().st_size > 0
assert AUDIT_MANIFEST_PATH.stat().st_size > 0


print("Registry:")
print(REGISTRY_PATH)

print("\nAudit manifest:")
print(AUDIT_MANIFEST_PATH)

print(
    "\nM6.2 REGISTRY PERSISTENCE GATE: PASS"
)

```

    Registry:
    /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/processed/m6_02_evaluation_artifact_audit/m6_02_evaluation_registry.json
    
    Audit manifest:
    /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/processed/m6_02_evaluation_artifact_audit/m6_02_audit_manifest.json
    
    M6.2 REGISTRY PERSISTENCE GATE: PASS


## 18. Persisted registry round-trip


```python

with open(
    REGISTRY_PATH,
    "r",
    encoding="utf-8",
) as file:
    registry_roundtrip = json.load(
        file
    )


with open(
    AUDIT_MANIFEST_PATH,
    "r",
    encoding="utf-8",
) as file:
    manifest_roundtrip = json.load(
        file
    )


assert (
    registry_roundtrip[
        "registry_version"
    ]
    == REGISTRY_VERSION
)

assert (
    registry_roundtrip[
        "official_run_count"
    ]
    == 6
)

assert (
    len(
        registry_roundtrip[
            "records"
        ]
    )
    == 6
)

assert (
    registry_roundtrip[
        "y_validation_sha256"
    ]
    == y_validation_sha256
)

assert (
    registry_roundtrip[
        "row_id_validation_sha256"
    ]
    == row_id_validation_sha256
)

assert (
    registry_roundtrip[
        "final_test_accessed"
    ]
    is False
)

assert (
    manifest_roundtrip[
        "official_runs_verified"
    ]
    == 6
)

assert (
    manifest_roundtrip[
        "independent_metric_reconstruction"
    ]
    == "PASS"
)

assert (
    manifest_roundtrip[
        "summary_agreement"
    ]
    == "PASS"
)


roundtrip_ids = {
    record[
        "experiment_id"
    ]
    for record
    in registry_roundtrip[
        "records"
    ]
}

assert roundtrip_ids == set(
    OFFICIAL_RUNS.keys()
)


print(
    "M6.2 REGISTRY ROUND-TRIP GATE: PASS"
)

```

    M6.2 REGISTRY ROUND-TRIP GATE: PASS


## 19. FINAL TEST isolation


```python

assert (
    m4_manifest[
        "final_test_used"
    ]
    is False
)


for record in (
    evaluation_registry
):
    assert (
        record[
            "final_test_accessed"
        ]
        is False
    )


for family, metadata in (
    family_metadata.items()
):
    assert (
        metadata[
            "config_lock"
        ][
            "final_test_access"
        ]
        is False
    )

    assert (
        metadata[
            "pair_manifest"
        ][
            "final_test_accessed"
        ]
        is False
    )


audit_input_paths = [
    M4_DIR
    / "y_validation.npy",

    M4_DIR
    / "row_id_validation.npy",

    *[
        spec["dir"]
        / (
            f"{experiment_id}"
            "__y_pred.npy"
        )
        for experiment_id, spec
        in OFFICIAL_RUNS.items()
    ],

    *[
        spec["dir"]
        / (
            f"{experiment_id}"
            "__risk_score.npy"
        )
        for experiment_id, spec
        in OFFICIAL_RUNS.items()
    ],

    *[
        spec["dir"]
        / (
            f"{experiment_id}"
            "__summary.json"
        )
        for experiment_id, spec
        in OFFICIAL_RUNS.items()
    ],
]


for path in audit_input_paths:
    assert (
        "final_test"
        not in path.name.lower()
    ), (
        "Phát hiện input path "
        "liên quan FINAL TEST: "
        f"{path}"
    )


assert (
    registry_payload[
        "final_test_accessed"
    ]
    is False
)

assert (
    audit_manifest[
        "final_test_accessed"
    ]
    is False
)


print(
    "M6.2 FINAL TEST ISOLATION GATE: PASS"
)

```

    M6.2 FINAL TEST ISOLATION GATE: PASS


## 20. M6.2 overall technical gate


```python

m6_02_gates = {
    "G01_M4_VALIDATION_IDENTITY":
        True,

    "G02_ROW_ID_VALIDATION_IDENTITY":
        True,

    "G03_SIX_PREDICTION_ARTIFACTS":
        True,

    "G04_SIX_RISK_SCORE_ARTIFACTS":
        True,

    "G05_SIX_SUMMARY_ARTIFACTS":
        True,

    "G06_THREE_CONFIG_LOCKS":
        True,

    "G07_THREE_PAIR_MANIFESTS":
        True,

    "G08_ARRAY_INTEGRITY":
        True,

    "G09_EXPERIMENT_IDENTITY":
        True,

    "G10_INDEPENDENT_METRIC_RECONSTRUCTION":
        True,

    "G11_SUMMARY_AGREEMENT":
        True,

    "G12_CONFUSION_ARITHMETIC":
        True,

    "G13_FINAL_TEST_ISOLATION":
        True,

    "G14_EVALUATION_REGISTRY_PERSISTENCE":
        True,

    "G15_REGISTRY_ROUND_TRIP":
        True,
}


for gate_name, gate_value in (
    m6_02_gates.items()
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
    m6_02_gates.values()
)


print(
    "\nM6.2 OVERALL TECHNICAL GATE: PASS"
)

```

    G01_M4_VALIDATION_IDENTITY → PASS
    G02_ROW_ID_VALIDATION_IDENTITY → PASS
    G03_SIX_PREDICTION_ARTIFACTS → PASS
    G04_SIX_RISK_SCORE_ARTIFACTS → PASS
    G05_SIX_SUMMARY_ARTIFACTS → PASS
    G06_THREE_CONFIG_LOCKS → PASS
    G07_THREE_PAIR_MANIFESTS → PASS
    G08_ARRAY_INTEGRITY → PASS
    G09_EXPERIMENT_IDENTITY → PASS
    G10_INDEPENDENT_METRIC_RECONSTRUCTION → PASS
    G11_SUMMARY_AGREEMENT → PASS
    G12_CONFUSION_ARITHMETIC → PASS
    G13_FINAL_TEST_ISOLATION → PASS
    G14_EVALUATION_REGISTRY_PERSISTENCE → PASS
    G15_REGISTRY_ROUND_TRIP → PASS
    
    M6.2 OVERALL TECHNICAL GATE: PASS


# 21. Runtime review và Findings M6.2

## 21.1. Execution integrity

Observed:

```text
Code cells:
18 / 18

Execution count:
1 → 18 liên tục

Runtime errors:
0

stderr:
0
```

Interpretation:

Notebook đã chạy đầy đủ từ đầu đến cuối.

Không có evidence của partial execution hoặc hidden exception.

Status:

`VERIFIED`

---

## 21.2. Runtime environment

Observed:

```text
Python:
3.14.6

Platform:
macOS-26.6.2-arm64

NumPy:
2.5.3

scikit-learn:
1.9.1
```

M6.2 không fit model nên environment evidence ở đây chủ yếu phục vụ reproducibility của evaluation reconstruction.

Status:

`RECORDED`

---

## 21.3. Canonical validation target

Observed:

```text
M4 pipeline:
M4.7-baseline-v1

y_validation:
shape = (712,458,)
dtype = int8

Validation rows:
712,458

Validation fraud:
1,052

Target support:
{0, 1}
```

Runtime gate:

`M6.2 VALIDATION TARGET / LINEAGE GATE: PASS`

Interpretation:

M6.2 đang đánh giá đúng canonical validation population đã khóa.

Status:

`VERIFIED`

---

## 21.4. Validation lineage anchor

Observed:

```text
row_id_validation:
shape = (712,458,)
dtype = int64

Unique row IDs:
712,458
```

SHA-256:

```text
y_validation:
0a21b2e93e017eda8b794be1a94a692beebc5883c1303427bc245299b55f5306

row_id_validation:
e7f3074c8604680eccef60ff7be45501f0dc10efbf6ac82e47bcbd1b00e43adc
```

Interpretation:

`row_id_validation` đủ điều kiện làm canonical lineage anchor cho M6.

Giới hạn cần giữ rõ:

M6.2 mới xác minh:

- đúng length;
- uniqueness;
- identity fingerprint;
- alignment contract với canonical validation artifacts.

M6.2 **chưa** chứng minh mapping implementation từ `row_id_validation` trở lại transaction-level semantic representation.

Mapping đó phải được audit riêng trước khi M6.6 đưa ra row-level FP/FN findings.

Status:

`ANCHOR VERIFIED / TRANSACTION-LEVEL MAPPING DEFERRED`

---

## 21.5. Config locks và pair manifests

Observed:

```text
Families audited:
Logistic Regression
Decision Tree
Random Forest

Config locks:
3 / 3

Pair manifests:
3 / 3
```

Runtime gates:

```text
M6.2 CONFIG LOCK / PAIR MANIFEST GATE:
PASS

M6.2 PAIR RUN IDENTITY GATE:
PASS
```

Official run identities:

```text
M5-LR-SHORT-B04
M5-LR-LONG-B04

M5-DT-SHORT-B01
M5-DT-LONG-B01

M5-RF-SHORT-B01
M5-RF-LONG-B01
```

Status:

`VERIFIED`

---

## 21.6. M5.3 legacy-summary compatibility

Observed for Logistic Regression B04:

```text
prediction_file metadata:
NOT_PRESENT_IN_SOURCE_SUMMARY_SCHEMA

risk_score_file metadata:
NOT_PRESENT_IN_SOURCE_SUMMARY_SCHEMA
```

Observed for Decision Tree / Random Forest:

```text
prediction_file metadata:
PASS

risk_score_file metadata:
PASS
```

Interpretation:

Đây không phải integrity failure.

M5.3 B04 sử dụng summary schema cũ hơn và không persist hai filename fields trực tiếp trong summary JSON.

M6.2 vẫn xác minh Logistic Regression artifacts bằng:

- canonical filename;
- file existence;
- experiment identity;
- pair-manifest identity;
- SHA-256;
- independent metric reconstruction;
- persisted-summary metric agreement.

Do đó compatibility handling này không làm yếu prediction/metric integrity.

Status:

`LEGACY SCHEMA COMPATIBILITY VERIFIED`

---

## 21.7. Six-run prediction / probability artifact audit

Observed:

```text
Official runs:
6 / 6

Prediction artifacts:
6 / 6 verified

Risk-score artifacts:
6 / 6 verified

Summary artifacts:
6 / 6 verified

Artifact integrity:
PASS for all six runs
```

Probability contract:

```text
risk_score_available:
TRUE

risk_score_kind:
predict_proba

finite:
TRUE

range:
[0, 1]
```

Status:

`VERIFIED`

---

## 21.8. Independent metric reconstruction — Logistic Regression

### M5-LR-SHORT-B04

```text
F1_fraud:
0.33747547416612167

Recall_fraud:
0.24524714828897337

Precision_fraud:
0.5408805031446541

TP / FP / FN / TN:
258 / 219 / 794 / 711187

Predicted positive:
477

Predicted-positive rate:
0.0006695131502488568
```

Independent metric check:

`PASS`

Persisted summary match:

`PASS`

---

### M5-LR-LONG-B04

```text
F1_fraud:
0.03996366939146231

Recall_fraud:
0.02091254752851711

Precision_fraud:
0.4489795918367347

TP / FP / FN / TN:
22 / 27 / 1030 / 711379

Predicted positive:
49

Predicted-positive rate:
0.00006877598398782806
```

Independent metric check:

`PASS`

Persisted summary match:

`PASS`

---

## 21.9. Independent metric reconstruction — Decision Tree

### M5-DT-SHORT-B01

```text
F1_fraud:
0.3270564915758176

Recall_fraud:
0.31368821292775667

Precision_fraud:
0.3416149068322981

TP / FP / FN / TN:
330 / 636 / 722 / 710770

Predicted positive:
966

Predicted-positive rate:
0.0013558693986171816
```

Independent metric check:

`PASS`

Persisted summary match:

`PASS`

---

### M5-DT-LONG-B01

```text
F1_fraud:
0.19913419913419914

Recall_fraud:
0.1967680608365019

Precision_fraud:
0.20155793573515093

TP / FP / FN / TN:
207 / 820 / 845 / 710586

Predicted positive:
1,027

Predicted-positive rate:
0.0014414884807244777
```

Independent metric check:

`PASS`

Persisted summary match:

`PASS`

---

## 21.10. Independent metric reconstruction — Random Forest

### M5-RF-SHORT-B01

```text
F1_fraud:
0.3664670658682635

Recall_fraud:
0.2908745247148289

Precision_fraud:
0.49514563106796117

TP / FP / FN / TN:
306 / 312 / 746 / 711094

Predicted positive:
618

Predicted-positive rate:
0.0008674195531526069
```

Independent metric check:

`PASS`

Persisted summary match:

`PASS`

---

### M5-RF-LONG-B01

```text
F1_fraud:
0.15270935960591134

Recall_fraud:
0.08840304182509506

Precision_fraud:
0.5602409638554217

TP / FP / FN / TN:
93 / 73 / 959 / 711333

Predicted positive:
166

Predicted-positive rate:
0.000232996190652642
```

Independent metric check:

`PASS`

Persisted summary match:

`PASS`

---

## 21.11. Ý nghĩa của independent reconstruction

M6.2 đã chứng minh rằng six-run metric evidence không chỉ tồn tại trong M5 summary JSON.

Các metric có thể được dựng lại độc lập từ:

```text
canonical y_validation
+
persisted y_pred
```

và kết quả khớp M5 summary.

Điều này giảm rủi ro:

- stale metric summary;
- accidental run mixing;
- confusion-label order mismatch;
- metric implementation drift;
- summary-only dependency.

Status:

`INDEPENDENTLY REPRODUCIBLE`

---

## 21.12. Confusion arithmetic

Với cả 6 runs:

```text
TP + FN:
1,052

TP + FP:
predicted_positive_count

TP + FP + FN + TN:
712,458
```

Runtime gate:

`G12_CONFUSION_ARITHMETIC → PASS`

Status:

`VERIFIED`

---

## 21.13. Registry completeness

Observed:

```text
Unique experiment IDs:
6

Families:
3

Windows per family:
2

Required windows:
W_SHORT
W_LONG
```

Runtime gate:

`M6.2 REGISTRY COMPLETENESS GATE: PASS`

Status:

`VERIFIED`

---

## 21.14. Controlled-pair metadata

Observed:

```text
Controlled fields checked:
12
```

Checked fields include:

- model family;
- model ID;
- model config ID;
- feature version;
- preprocessing version;
- matrix schema;
- validation period;
- validation rows;
- validation fraud rows;
- imbalance strategy;
- threshold policy;
- risk-score kind.

Runtime gate:

`M6.2 CONTROLLED-PAIR METADATA GATE: PASS`

Interpretation:

Ba SHORT/LONG pairs đủ metadata integrity để M6.4 tiếp tục controlled training-window comparison.

M6.2 chưa chọn window winner.

Status:

`VERIFIED`

---

## 21.15. Evaluation Registry persistence

Persisted:

```text
data/processed/m6_02_evaluation_artifact_audit/
    m6_02_evaluation_registry.json
    m6_02_audit_manifest.json
```

Runtime gates:

```text
M6.2 REGISTRY PERSISTENCE GATE:
PASS

M6.2 REGISTRY ROUND-TRIP GATE:
PASS
```

Evaluation Registry chứa:

- canonical validation identity;
- six official records;
- independent metrics;
- artifact paths;
- artifact SHA-256;
- integrity statuses;
- winner fields giữ OPEN;
- final-test flag false.

Status:

`VERIFIED`

---

## 21.16. FINAL TEST isolation

Observed:

```text
M4 manifest:
final_test_used = false

6 evaluation records:
final_test_accessed = false

3 config locks:
final_test_access = false

3 pair manifests:
final_test_accessed = false
```

Runtime gate:

`M6.2 FINAL TEST ISOLATION GATE: PASS`

Status:

`VERIFIED`

---

## 21.17. Overall technical result

Observed:

```text
G01_M4_VALIDATION_IDENTITY             → PASS
G02_ROW_ID_VALIDATION_IDENTITY         → PASS
G03_SIX_PREDICTION_ARTIFACTS           → PASS
G04_SIX_RISK_SCORE_ARTIFACTS           → PASS
G05_SIX_SUMMARY_ARTIFACTS              → PASS
G06_THREE_CONFIG_LOCKS                 → PASS
G07_THREE_PAIR_MANIFESTS               → PASS
G08_ARRAY_INTEGRITY                    → PASS
G09_EXPERIMENT_IDENTITY                → PASS
G10_INDEPENDENT_METRIC_RECONSTRUCTION  → PASS
G11_SUMMARY_AGREEMENT                  → PASS
G12_CONFUSION_ARITHMETIC               → PASS
G13_FINAL_TEST_ISOLATION               → PASS
G14_EVALUATION_REGISTRY_PERSISTENCE    → PASS
G15_REGISTRY_ROUND_TRIP                → PASS
```

Overall:

`M6.2 OVERALL TECHNICAL GATE: PASS`

Blocking issue:

`NONE`

---

# 22. Findings M6.2

## M6.2-F01 — Notebook execution is complete

Evidence:

```text
18 / 18 code cells executed
execution_count = 1 → 18
runtime error = 0
stderr = 0
```

Status:

`VERIFIED`

---

## M6.2-F02 — Canonical validation population is intact

Evidence:

```text
712,458 rows
1,052 fraud
target dtype = int8
```

Status:

`VERIFIED`

---

## M6.2-F03 — Validation row-ID anchor is valid

Evidence:

```text
712,458 row IDs
712,458 unique
```

Finding:

`row_id_validation` is valid as the canonical lineage anchor.

Boundary:

transaction-level semantic remapping is still deferred to M6.6.

Status:

`VERIFIED WITH SCOPE LIMIT`

---

## M6.2-F04 — All six official prediction artifacts are evaluation-ready

Evidence:

`6 / 6 artifact integrity checks PASS`

Status:

`VERIFIED`

---

## M6.2-F05 — All six probability artifacts are evaluation-ready

Evidence:

```text
6 / 6 risk scores loaded
finite
within [0,1]
predict_proba semantics verified
```

Status:

`VERIFIED`

---

## M6.2-F06 — M5.3 legacy summary schema is not an evaluation blocker

Evidence:

LR B04 summary lacks filename metadata but canonical artifacts, pair identity, hashes and reconstructed metrics all verify correctly.

Status:

`VERIFIED COMPATIBILITY`

---

## M6.2-F07 — Canonical metrics are independently reproducible

Evidence:

`6 / 6 independent metric reconstructions PASS`

Status:

`VERIFIED`

---

## M6.2-F08 — M5 persisted summaries agree with independent M6 reconstruction

Evidence:

```text
count fields:
exact match

floating metrics:
match within declared tolerance
```

Status:

`VERIFIED`

---

## M6.2-F09 — Confusion arithmetic is internally consistent

Evidence:

All six runs satisfy canonical row/fraud/count equations.

Status:

`VERIFIED`

---

## M6.2-F10 — Three controlled SHORT/LONG pairs are metadata-consistent

Evidence:

12 controlled fields match within each family.

Status:

`VERIFIED`

---

## M6.2-F11 — Independent M6 Evaluation Registry is persisted

Evidence:

Registry + audit manifest persistence and round-trip PASS.

Status:

`VERIFIED`

---

## M6.2-F12 — FINAL TEST remains protected

Evidence:

FINAL TEST isolation gate PASS.

Status:

`VERIFIED`

---

## M6.2-F13 — No model/window winner is selected in M6.2

Observed:

```text
Training-window winner:
OPEN

Model-family winner:
OPEN

Final threshold:
OPEN
```

Status:

`CORRECT BOUNDARY`

---

## M6.2-F14 — M6.3 handoff is unblocked

M6.3 can now consume an independently verified six-run Evaluation Registry for aggregate metric and Confusion Matrix analysis.

Status:

`READY FOR M6.3`

# 23. Decision Log M6.2 — sau runtime review

## M6.2-D01 — Evaluation source

Decision:

Use persisted M5 official baseline artifacts.

Observed:

`6 / 6 official runs verified`

Status:

`LOCKED`

---

## M6.2-D02 — Retraining

Decision:

No model fit/retraining.

Observed:

M6.2 completed without retraining.

Status:

`LOCKED / VERIFIED`

---

## M6.2-D03 — Canonical target

Decision:

`M4.7 y_validation.npy`

Observed:

```text
712,458 rows
1,052 fraud
dtype int8
```

Status:

`LOCKED / VERIFIED`

---

## M6.2-D04 — Canonical lineage anchor

Decision:

`M4.7 row_id_validation.npy`

Observed:

```text
712,458 entries
712,458 unique
```

Status:

`LOCKED AS ANCHOR / VERIFIED`

Transaction-level mapping:

`DEFERRED TO M6.6`

---

## M6.2-D05 — Independent metric reconstruction

Decision:

Recompute from:

`y_validation + persisted y_pred`

Observed:

`6 / 6 PASS`

Status:

`LOCKED / VERIFIED`

---

## M6.2-D06 — Persisted summary role

Decision:

M5 summary is comparison/reference evidence, not the sole metric source in M6.

Observed:

`6 / 6 summaries agree with independent reconstruction`

Status:

`LOCKED / VERIFIED`

---

## M6.2-D07 — Legacy M5.3 summary compatibility

Decision:

Missing `prediction_file` / `risk_score_file` metadata in LR-B04 summary is accepted only when canonical filename, existence, experiment identity, pair identity, SHA-256 and independent reconstruction all pass.

Observed:

All required substitute identity checks PASS.

Status:

`LOCKED / VERIFIED`

---

## M6.2-D08 — Probability evidence

Decision:

Use persisted positive-class `predict_proba` score for later descriptive M6 analysis.

Observed:

`6 / 6 risk-score artifacts verified`

Threshold optimization:

`NOT AUTHORIZED IN M6`

Status:

`LOCKED`

---

## M6.2-D09 — Evaluation Registry

Decision:

Persist:

`m6_02_evaluation_registry.json`

Observed:

Persistence + round-trip PASS.

Status:

`LOCKED / VERIFIED`

---

## M6.2-D10 — Audit manifest

Decision:

Persist:

`m6_02_audit_manifest.json`

Observed:

Persistence + round-trip PASS.

Status:

`LOCKED / VERIFIED`

---

## M6.2-D11 — Artifact fingerprints

Decision:

Use SHA-256 as identity/reproducibility evidence only.

Observed:

Canonical target/lineage and run artifact hashes persisted in registry.

Status:

`LOCKED / VERIFIED`

---

## M6.2-D12 — Controlled pair metadata

Decision:

SHORT/LONG pair must match controlled metadata before M6.4 comparative interpretation.

Observed:

```text
LR pair:
PASS

DT pair:
PASS

RF pair:
PASS
```

Status:

`VERIFIED`

---

## M6.2-D13 — Winner decisions

Decision:

M6.2 does not select training window or model family.

Status:

```text
Training-window winner:
OPEN

Model-family winner:
OPEN
```

`LOCKED BOUNDARY`

---

## M6.2-D14 — Final threshold

Decision:

No threshold optimization in M6.2.

Status:

`OPEN — DEFERRED TO M7`

---

## M6.2-D15 — FINAL TEST

Decision:

No access.

Observed:

FINAL TEST isolation PASS.

Status:

`INHERITED — VERIFIED — LOCKED`

---

## M6.2-D16 — M6.3 handoff

Decision:

Use verified M6.2 Evaluation Registry as canonical aggregate-evaluation input for M6.3.

Status:

`READY`

# 24. M6.2 Gate

## Technical runtime gates

```text
G01_M4_VALIDATION_IDENTITY             → PASS
G02_ROW_ID_VALIDATION_IDENTITY         → PASS
G03_SIX_PREDICTION_ARTIFACTS           → PASS
G04_SIX_RISK_SCORE_ARTIFACTS           → PASS
G05_SIX_SUMMARY_ARTIFACTS              → PASS
G06_THREE_CONFIG_LOCKS                 → PASS
G07_THREE_PAIR_MANIFESTS               → PASS
G08_ARRAY_INTEGRITY                    → PASS
G09_EXPERIMENT_IDENTITY                → PASS
G10_INDEPENDENT_METRIC_RECONSTRUCTION  → PASS
G11_SUMMARY_AGREEMENT                  → PASS
G12_CONFUSION_ARITHMETIC               → PASS
G13_FINAL_TEST_ISOLATION               → PASS
G14_EVALUATION_REGISTRY_PERSISTENCE    → PASS
G15_REGISTRY_ROUND_TRIP                → PASS
```

Technical gates:

`15 / 15 PASS`

---

## R01 — Execution complete?

Evidence:

```text
18 / 18 code cells
execution_count = 1 → 18
errors = 0
stderr = 0
```

Result:

`PASS`

---

## R02 — Canonical validation identity verified?

Evidence:

```text
712,458 rows
1,052 fraud
M4.7-baseline-v1
```

Result:

`PASS`

---

## R03 — Validation lineage anchor verified?

Evidence:

```text
712,458 row IDs
712,458 unique
SHA-256 recorded
```

Result:

`PASS`

Scope note:

transaction-level remapping remains M6.6 work.

---

## R04 — All official prediction/risk-score artifacts verified?

Evidence:

```text
Predictions:
6 / 6

Risk scores:
6 / 6
```

Result:

`PASS`

---

## R05 — Independent metrics reconstructed?

Evidence:

`6 / 6 PASS`

Result:

`PASS`

---

## R06 — Persisted summaries agree?

Evidence:

`6 / 6 PASS`

Result:

`PASS`

---

## R07 — Controlled-pair metadata intact?

Evidence:

```text
3 / 3 model-family pairs
12 controlled fields
```

Result:

`PASS`

---

## R08 — Evaluation Registry persisted and reloadable?

Evidence:

Persistence and round-trip gates PASS.

Result:

`PASS`

---

## R09 — M5.3 legacy summary compatibility handled safely?

Evidence:

Missing filename metadata does not bypass canonical filename, file existence, pair identity, hash or metric reconstruction checks.

Result:

`PASS`

---

## R10 — FINAL TEST protected?

Evidence:

FINAL TEST isolation gate PASS.

Result:

`PASS`

---

## Overall M6.2 Gate

```text
Technical gates:
15 / 15 PASS

Runtime review gates:
10 / 10 PASS

Blocking issue:
NONE
```

Final:

`M6.2 — PASS`

Handoff:

`READY FOR M6.3`

# 25. Kết luận M6.2

M6.2 đã hoàn thành evaluation-artifact audit và independent metric reconstruction cho toàn bộ six-run baseline registry.

Canonical validation:

```text
Rows:
712,458

Fraud:
1,052

Positive class:
fraud = 1
```

Official evaluation subjects:

```text
M5-LR-SHORT-B04
M5-LR-LONG-B04

M5-DT-SHORT-B01
M5-DT-LONG-B01

M5-RF-SHORT-B01
M5-RF-LONG-B01
```

Đã xác minh:

```text
Prediction artifacts:
6 / 6 VERIFIED

Risk-score artifacts:
6 / 6 VERIFIED

Summary artifacts:
6 / 6 VERIFIED

Config locks:
3 / 3 VERIFIED

Pair manifests:
3 / 3 VERIFIED

Independent metric reconstruction:
6 / 6 PASS

Persisted-summary agreement:
6 / 6 PASS

Confusion arithmetic:
PASS

Controlled-pair metadata:
PASS

Evaluation Registry persistence:
PASS

Evaluation Registry round-trip:
PASS

FINAL TEST:
PROTECTED
```

M6.2 đã tạo independent M6 evaluation evidence tại:

```text
data/processed/m6_02_evaluation_artifact_audit/
    m6_02_evaluation_registry.json
    m6_02_audit_manifest.json
```

Điểm lineage được khóa đúng scope:

```text
row_id_validation as canonical anchor:
VERIFIED

transaction-level semantic remapping:
NOT YET VERIFIED
DEFERRED TO M6.6
```

M6.2 không khóa:

```text
Training-window winner:
OPEN

Model-family winner:
OPEN

Final threshold:
OPEN
```

Final state:

```text
M6.2 — PASS

Execution Integrity:
VERIFIED

Canonical Validation:
VERIFIED

Validation Lineage Anchor:
VERIFIED

Independent Evaluation Registry:
VERIFIED

Six-run Metric Reconstruction:
VERIFIED

Summary Agreement:
VERIFIED

Controlled Pair Metadata:
VERIFIED

FINAL TEST:
PROTECTED

Blocking Issue:
NONE

READY FOR M6.3
```

Bước tiếp theo:

`M6.3 — Baseline metric và Confusion Matrix analysis`

M6.3 sẽ sử dụng Evaluation Registry đã verify để diễn giải fraud F1, Recall, Precision, TP/FP/FN/TN và predicted-positive behavior cho cả 6 official runs mà không retrain model.
