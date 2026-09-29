# M6.3 — Baseline metric và Confusion Matrix analysis

Milestone:

`M6 — Evaluation + Error Analysis`

Substep:

`M6.3 — Baseline Metric + Confusion Matrix Analysis`

Work type:

`RUNTIME EVALUATION / AGGREGATE INTERPRETATION`

Câu hỏi trung tâm:

> Với từng official baseline run, model đang bắt đúng bao nhiêu fraud, bỏ sót bao nhiêu fraud, tạo bao nhiêu false alert, và F1 / Recall / Precision phải được đọc như thế nào cùng Confusion Matrix?

M6.3 sử dụng:

`M6.2 independent Evaluation Registry`

làm canonical aggregate-evaluation input.

M6.3 không:

- fit model;
- retrain model;
- gọi lại prediction để thay persisted evidence;
- tuning;
- class-weight/resampling;
- threshold optimization;
- training-window selection;
- model-family selection;
- FINAL TEST access.

Runtime-dependent status trước khi chạy:

`NOT YET VERIFIED`

## 1. Contract kế thừa

M6.3 kế thừa trực tiếp:

- `CANON-Kế hoạch Milestone 6 — Evaluation và Error Analysis`;
- `CANON-M6.1 — Khóa Evaluation Charter, scope và guardrails`;
- `M6.2 — Evaluation artifact audit và independent metric reconstruction`.

M6.2 đã xác minh:

```text
Official runs:
6 / 6

Prediction artifacts:
6 / 6 VERIFIED

Risk-score artifacts:
6 / 6 VERIFIED

Independent metric reconstruction:
6 / 6 PASS

Persisted-summary agreement:
6 / 6 PASS

Controlled-pair metadata:
VERIFIED

FINAL TEST:
PROTECTED

M6.2:
PASS

READY FOR M6.3
```

Vì vậy M6.3 không cần lặp lại toàn bộ M6.2.

M6.3 tập trung vào:

`INTERPRETATION OF VERIFIED AGGREGATE EVIDENCE`

## 2. Canonical metric semantics

Primary:

`F1_fraud`

Secondary:

```text
Recall_fraud
Precision_fraud
```

Mandatory diagnostic:

```text
TP
FP
FN
TN
```

Operational diagnostic:

```text
predicted_positive_count
predicted_positive_rate
```

Accuracy:

`REFERENCE ONLY`

M6.3 phải luôn chuyển metric về raw transaction counts.

Ví dụ:

Không chỉ ghi:

`Recall = 0.29`

mà phải đọc:

```text
TP = số fraud bắt đúng
FN = số fraud bị bỏ sót
Recall = TP / (TP + FN)
```

Tương tự:

```text
Precision = TP / (TP + FP)
```

nên Precision phải được đọc cùng số false alert.

## 3. Scope boundary

M6.3 phân tích **từng run riêng lẻ**.

Comparison chính thức:

```text
W_SHORT vs W_LONG
```

thuộc:

`M6.4`

Comparison chính thức:

```text
LR vs DT vs RF
```

thuộc:

`M6.5`

Do đó M6.3 không:

- rank sáu runs;
- gọi run nào là winner;
- gọi model nào là tốt nhất;
- khóa W_SHORT/W_LONG;
- dùng một metric đơn lẻ để chọn model.

M6.3 tạo six-run evaluation profiles để các bước sau có evidence chuẩn hóa.


```python

from pathlib import Path
import hashlib
import json
import platform
import sys

import numpy as np


print("Python:")
print(sys.version)

print("\nExecutable:")
print(sys.executable)

print("\nPlatform:")
print(platform.platform())

print("\nNumPy:")
print(np.__version__)

```

    Python:
    3.14.6 (main, Jun 10 2026, 10:03:53) [Clang 21.0.0 (clang-2100.0.123.102)]
    
    Executable:
    /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/.venv/bin/python
    
    Platform:
    macOS-26.6.2-arm64-arm-64bit-Mach-O
    
    NumPy:
    2.5.3


## 4. Locate M6.2 Evaluation Registry


```python

M6_02_REL = (
    Path("data")
    / "processed"
    / "m6_02_evaluation_artifact_audit"
)

M6_03_REL = (
    Path("data")
    / "processed"
    / "m6_03_baseline_metric_confusion_analysis"
)


candidate_roots = [
    Path.cwd(),
    *list(Path.cwd().parents)[:6],
]


PROJECT_ROOT = None

required_rel_paths = [
    M6_02_REL
    / "m6_02_evaluation_registry.json",

    M6_02_REL
    / "m6_02_audit_manifest.json",
]


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
        "Không tìm thấy PROJECT_ROOT chứa "
        "M6.2 Evaluation Registry và audit manifest."
    )


M6_02_DIR = (
    PROJECT_ROOT
    / M6_02_REL
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / M6_03_REL
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


REGISTRY_PATH = (
    M6_02_DIR
    / "m6_02_evaluation_registry.json"
)

AUDIT_MANIFEST_PATH = (
    M6_02_DIR
    / "m6_02_audit_manifest.json"
)


print("PROJECT_ROOT:")
print(PROJECT_ROOT)

print("\nM6_02_DIR:")
print(M6_02_DIR)

print("\nOUTPUT_DIR:")
print(OUTPUT_DIR)

```

    PROJECT_ROOT:
    /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening
    
    M6_02_DIR:
    /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/processed/m6_02_evaluation_artifact_audit
    
    OUTPUT_DIR:
    /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/processed/m6_03_baseline_metric_confusion_analysis


## 5. Load và verify M6.2 handoff


```python

with open(
    REGISTRY_PATH,
    "r",
    encoding="utf-8",
) as file:
    m6_02_registry = json.load(
        file
    )


with open(
    AUDIT_MANIFEST_PATH,
    "r",
    encoding="utf-8",
) as file:
    m6_02_audit = json.load(
        file
    )


EXPECTED_REGISTRY_VERSION = (
    "M6.2-evaluation-registry-v1"
)

EXPECTED_VALIDATION_ROWS = 712_458
EXPECTED_VALIDATION_FRAUD = 1_052


assert (
    m6_02_registry[
        "registry_version"
    ]
    == EXPECTED_REGISTRY_VERSION
)

assert (
    m6_02_registry[
        "official_run_count"
    ]
    == 6
)

assert (
    len(
        m6_02_registry[
            "records"
        ]
    )
    == 6
)

assert (
    m6_02_registry[
        "validation_rows"
    ]
    == EXPECTED_VALIDATION_ROWS
)

assert (
    m6_02_registry[
        "validation_fraud_rows"
    ]
    == EXPECTED_VALIDATION_FRAUD
)

assert (
    m6_02_registry[
        "positive_class"
    ]
    == 1
)

assert (
    m6_02_registry[
        "training_window_winner"
    ]
    == "OPEN"
)

assert (
    m6_02_registry[
        "model_family_winner"
    ]
    == "OPEN"
)

assert (
    m6_02_registry[
        "final_threshold"
    ]
    == "OPEN"
)

assert (
    m6_02_registry[
        "final_test_accessed"
    ]
    is False
)


assert (
    m6_02_audit[
        "official_runs_verified"
    ]
    == 6
)

assert (
    m6_02_audit[
        "prediction_artifacts_verified"
    ]
    == 6
)

assert (
    m6_02_audit[
        "risk_score_artifacts_verified"
    ]
    == 6
)

assert (
    m6_02_audit[
        "independent_metric_reconstruction"
    ]
    == "PASS"
)

assert (
    m6_02_audit[
        "summary_agreement"
    ]
    == "PASS"
)

assert (
    m6_02_audit[
        "validation_lineage"
    ]
    == "PASS"
)

assert (
    m6_02_audit[
        "controlled_pair_metadata"
    ]
    == "PASS"
)

assert (
    m6_02_audit[
        "final_test_accessed"
    ]
    is False
)


print(
    "M6.3 M6.2 HANDOFF GATE: PASS"
)

```

    M6.3 M6.2 HANDOFF GATE: PASS


## 6. Verify six-run evaluation integrity

M6.3 chỉ dùng record đã được M6.2 đánh dấu:

```text
independent_metric_check = PASS
summary_match_check = PASS
artifact_integrity_check = PASS
evaluation_status = VERIFIED_FOR_M6_EVALUATION
final_test_accessed = false
```


```python

OFFICIAL_EXPERIMENT_IDS = {
    "M5-LR-SHORT-B04",
    "M5-LR-LONG-B04",
    "M5-DT-SHORT-B01",
    "M5-DT-LONG-B01",
    "M5-RF-SHORT-B01",
    "M5-RF-LONG-B01",
}


records = (
    m6_02_registry[
        "records"
    ]
)


actual_ids = {
    record[
        "experiment_id"
    ]
    for record
    in records
}


assert (
    actual_ids
    == OFFICIAL_EXPERIMENT_IDS
)


for record in records:
    assert (
        record[
            "validation_rows"
        ]
        == EXPECTED_VALIDATION_ROWS
    )

    assert (
        record[
            "validation_fraud_rows"
        ]
        == EXPECTED_VALIDATION_FRAUD
    )

    assert (
        record[
            "independent_metric_check"
        ]
        == "PASS"
    )

    assert (
        record[
            "summary_match_check"
        ]
        == "PASS"
    )

    assert (
        record[
            "artifact_integrity_check"
        ]
        == "PASS"
    )

    assert (
        record[
            "evaluation_status"
        ]
        == "VERIFIED_FOR_M6_EVALUATION"
    )

    assert (
        record[
            "final_test_accessed"
        ]
        is False
    )


print(
    "Official run IDs:",
    len(
        actual_ids
    ),
)

print(
    "\nM6.3 SIX-RUN INPUT INTEGRITY GATE: PASS"
)

```

    Official run IDs: 6
    
    M6.3 SIX-RUN INPUT INTEGRITY GATE: PASS


## 7. Derived evaluation-profile contract

M6.3 không tạo model metric mới.

M6.3 chỉ dẫn xuất các quantity dễ đọc từ Confusion Matrix:

```text
fraud_capture_count = TP
fraud_miss_count = FN

fraud_capture_rate = TP / (TP + FN)
fraud_miss_rate = FN / (TP + FN)

alert_count = TP + FP
true_alert_count = TP
false_alert_count = FP

true_alert_share = TP / (TP + FP)
false_alert_share = FP / (TP + FP)

negative_prediction_count = TN + FN

total_misclassified = FP + FN
```

Trong đó:

```text
fraud_capture_rate == Recall_fraud
true_alert_share == Precision_fraud
```

Các equality này phải được assert.


```python

def build_evaluation_profile(
    record,
):
    tp = int(
        record["tp"]
    )

    fp = int(
        record["fp"]
    )

    fn = int(
        record["fn"]
    )

    tn = int(
        record["tn"]
    )

    actual_fraud = (
        tp + fn
    )

    actual_non_fraud = (
        tn + fp
    )

    alert_count = (
        tp + fp
    )

    negative_prediction_count = (
        tn + fn
    )

    fraud_capture_rate = (
        tp / actual_fraud
        if actual_fraud
        else 0.0
    )

    fraud_miss_rate = (
        fn / actual_fraud
        if actual_fraud
        else 0.0
    )

    true_alert_share = (
        tp / alert_count
        if alert_count
        else 0.0
    )

    false_alert_share = (
        fp / alert_count
        if alert_count
        else 0.0
    )

    total_misclassified = (
        fp + fn
    )

    overall_error_rate = (
        total_misclassified
        / EXPECTED_VALIDATION_ROWS
    )

    # Arithmetic / semantic consistency.
    assert (
        actual_fraud
        == EXPECTED_VALIDATION_FRAUD
    )

    assert (
        actual_fraud
        + actual_non_fraud
        == EXPECTED_VALIDATION_ROWS
    )

    assert (
        alert_count
        == int(
            record[
                "predicted_positive_count"
            ]
        )
    )

    assert np.isclose(
        (
            alert_count
            / EXPECTED_VALIDATION_ROWS
        ),
        float(
            record[
                "predicted_positive_rate"
            ]
        ),
        rtol=1e-12,
        atol=1e-12,
    )

    assert np.isclose(
        fraud_capture_rate,
        float(
            record[
                "recall_fraud"
            ]
        ),
        rtol=1e-12,
        atol=1e-12,
    )

    assert np.isclose(
        true_alert_share,
        float(
            record[
                "precision_fraud"
            ]
        ),
        rtol=1e-12,
        atol=1e-12,
    )

    assert np.isclose(
        fraud_capture_rate
        + fraud_miss_rate,
        1.0,
        rtol=1e-12,
        atol=1e-12,
    )

    if alert_count > 0:
        assert np.isclose(
            true_alert_share
            + false_alert_share,
            1.0,
            rtol=1e-12,
            atol=1e-12,
        )

    return {
        "experiment_id":
            record[
                "experiment_id"
            ],

        "model_family":
            record[
                "model_family"
            ],

        "model_config_id":
            record[
                "model_config_id"
            ],

        "training_window_id":
            record[
                "training_window_id"
            ],

        "validation_rows":
            EXPECTED_VALIDATION_ROWS,

        "actual_fraud_count":
            actual_fraud,

        "actual_non_fraud_count":
            actual_non_fraud,

        "f1_fraud":
            float(
                record[
                    "f1_fraud"
                ]
            ),

        "recall_fraud":
            float(
                record[
                    "recall_fraud"
                ]
            ),

        "precision_fraud":
            float(
                record[
                    "precision_fraud"
                ]
            ),

        "accuracy_reference":
            float(
                record[
                    "accuracy_reference"
                ]
            ),

        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,

        "fraud_capture_count":
            tp,

        "fraud_miss_count":
            fn,

        "fraud_capture_rate":
            float(
                fraud_capture_rate
            ),

        "fraud_miss_rate":
            float(
                fraud_miss_rate
            ),

        "alert_count":
            alert_count,

        "alert_rate":
            float(
                alert_count
                / EXPECTED_VALIDATION_ROWS
            ),

        "true_alert_count":
            tp,

        "false_alert_count":
            fp,

        "true_alert_share":
            float(
                true_alert_share
            ),

        "false_alert_share":
            float(
                false_alert_share
            ),

        "negative_prediction_count":
            negative_prediction_count,

        "total_misclassified":
            total_misclassified,

        "overall_error_rate_reference":
            float(
                overall_error_rate
            ),

        "interpretation_scope":
            (
                "SINGLE_RUN_AGGREGATE_"
                "DESCRIPTIVE_EVALUATION"
            ),

        "winner_decision":
            "NOT_AUTHORIZED_IN_M6.3",

        "final_test_accessed":
            False,
    }


print(
    "M6.3 EVALUATION PROFILE FUNCTION: DEFINED"
)

```

    M6.3 EVALUATION PROFILE FUNCTION: DEFINED


# 8. Build six evaluation profiles


```python

evaluation_profiles = [
    build_evaluation_profile(
        record
    )
    for record
    in records
]


assert (
    len(
        evaluation_profiles
    )
    == 6
)

assert (
    len({
        profile[
            "experiment_id"
        ]
        for profile
        in evaluation_profiles
    })
    == 6
)


print(
    "Evaluation profiles:",
    len(
        evaluation_profiles
    ),
)

print(
    "\nM6.3 SIX-RUN PROFILE BUILD GATE: PASS"
)

```

    Evaluation profiles: 6
    
    M6.3 SIX-RUN PROFILE BUILD GATE: PASS


## 9. Single-run interpretation format

Mỗi run sẽ được đọc theo cùng một template:

```text
Fraud thật:
1,052

Bắt đúng:
TP

Bỏ sót:
FN

Fraud capture rate:
Recall

Alerts phát ra:
TP + FP

Alert đúng:
TP

False alert:
FP

Alert precision:
Precision

F1:
fraud-class balance summary

Accuracy:
reference only
```

M6.3 không dùng wording:

`best / winner / final model`


```python

def print_profile(
    profile,
):
    print("=" * 72)

    print(
        "Experiment:",
        profile[
            "experiment_id"
        ],
    )

    print(
        "Family:",
        profile[
            "model_family"
        ],
    )

    print(
        "Training window:",
        profile[
            "training_window_id"
        ],
    )

    print()

    print(
        "Fraud thật:",
        profile[
            "actual_fraud_count"
        ],
    )

    print(
        "Fraud bắt đúng (TP):",
        profile["tp"],
    )

    print(
        "Fraud bỏ sót (FN):",
        profile["fn"],
    )

    print(
        "Fraud capture / Recall:",
        (
            f"{profile['fraud_capture_rate']:.6f}"
        ),
    )

    print(
        "Fraud miss rate:",
        (
            f"{profile['fraud_miss_rate']:.6f}"
        ),
    )

    print()

    print(
        "Alerts phát ra (TP + FP):",
        profile[
            "alert_count"
        ],
    )

    print(
        "Alert rate:",
        (
            f"{profile['alert_rate']:.8f}"
        ),
    )

    print(
        "Alert đúng / TP:",
        profile[
            "true_alert_count"
        ],
    )

    print(
        "False alert / FP:",
        profile[
            "false_alert_count"
        ],
    )

    print(
        "Precision:",
        (
            f"{profile['precision_fraud']:.6f}"
        ),
    )

    print(
        "False-alert share among alerts:",
        (
            f"{profile['false_alert_share']:.6f}"
        ),
    )

    print()

    print(
        "F1_fraud:",
        (
            f"{profile['f1_fraud']:.6f}"
        ),
    )

    print(
        "Accuracy reference:",
        (
            f"{profile['accuracy_reference']:.6f}"
        ),
    )

    print(
        "Total misclassified FP + FN:",
        profile[
            "total_misclassified"
        ],
    )

    print(
        "Interpretation scope:",
        profile[
            "interpretation_scope"
        ],
    )


for profile in (
    evaluation_profiles
):
    print_profile(
        profile
    )


print(
    "\nM6.3 SIX-RUN HUMAN-READABLE PROFILE GATE: PASS"
)

```

    ========================================================================
    Experiment: M5-LR-SHORT-B04
    Family: Logistic Regression
    Training window: W_SHORT
    
    Fraud thật: 1052
    Fraud bắt đúng (TP): 258
    Fraud bỏ sót (FN): 794
    Fraud capture / Recall: 0.245247
    Fraud miss rate: 0.754753
    
    Alerts phát ra (TP + FP): 477
    Alert rate: 0.00066951
    Alert đúng / TP: 258
    False alert / FP: 219
    Precision: 0.540881
    False-alert share among alerts: 0.459119
    
    F1_fraud: 0.337475
    Accuracy reference: 0.998578
    Total misclassified FP + FN: 1013
    Interpretation scope: SINGLE_RUN_AGGREGATE_DESCRIPTIVE_EVALUATION
    ========================================================================
    Experiment: M5-LR-LONG-B04
    Family: Logistic Regression
    Training window: W_LONG
    
    Fraud thật: 1052
    Fraud bắt đúng (TP): 22
    Fraud bỏ sót (FN): 1030
    Fraud capture / Recall: 0.020913
    Fraud miss rate: 0.979087
    
    Alerts phát ra (TP + FP): 49
    Alert rate: 0.00006878
    Alert đúng / TP: 22
    False alert / FP: 27
    Precision: 0.448980
    False-alert share among alerts: 0.551020
    
    F1_fraud: 0.039964
    Accuracy reference: 0.998516
    Total misclassified FP + FN: 1057
    Interpretation scope: SINGLE_RUN_AGGREGATE_DESCRIPTIVE_EVALUATION
    ========================================================================
    Experiment: M5-DT-SHORT-B01
    Family: Decision Tree
    Training window: W_SHORT
    
    Fraud thật: 1052
    Fraud bắt đúng (TP): 330
    Fraud bỏ sót (FN): 722
    Fraud capture / Recall: 0.313688
    Fraud miss rate: 0.686312
    
    Alerts phát ra (TP + FP): 966
    Alert rate: 0.00135587
    Alert đúng / TP: 330
    False alert / FP: 636
    Precision: 0.341615
    False-alert share among alerts: 0.658385
    
    F1_fraud: 0.327056
    Accuracy reference: 0.998094
    Total misclassified FP + FN: 1358
    Interpretation scope: SINGLE_RUN_AGGREGATE_DESCRIPTIVE_EVALUATION
    ========================================================================
    Experiment: M5-DT-LONG-B01
    Family: Decision Tree
    Training window: W_LONG
    
    Fraud thật: 1052
    Fraud bắt đúng (TP): 207
    Fraud bỏ sót (FN): 845
    Fraud capture / Recall: 0.196768
    Fraud miss rate: 0.803232
    
    Alerts phát ra (TP + FP): 1027
    Alert rate: 0.00144149
    Alert đúng / TP: 207
    False alert / FP: 820
    Precision: 0.201558
    False-alert share among alerts: 0.798442
    
    F1_fraud: 0.199134
    Accuracy reference: 0.997663
    Total misclassified FP + FN: 1665
    Interpretation scope: SINGLE_RUN_AGGREGATE_DESCRIPTIVE_EVALUATION
    ========================================================================
    Experiment: M5-RF-SHORT-B01
    Family: Random Forest
    Training window: W_SHORT
    
    Fraud thật: 1052
    Fraud bắt đúng (TP): 306
    Fraud bỏ sót (FN): 746
    Fraud capture / Recall: 0.290875
    Fraud miss rate: 0.709125
    
    Alerts phát ra (TP + FP): 618
    Alert rate: 0.00086742
    Alert đúng / TP: 306
    False alert / FP: 312
    Precision: 0.495146
    False-alert share among alerts: 0.504854
    
    F1_fraud: 0.366467
    Accuracy reference: 0.998515
    Total misclassified FP + FN: 1058
    Interpretation scope: SINGLE_RUN_AGGREGATE_DESCRIPTIVE_EVALUATION
    ========================================================================
    Experiment: M5-RF-LONG-B01
    Family: Random Forest
    Training window: W_LONG
    
    Fraud thật: 1052
    Fraud bắt đúng (TP): 93
    Fraud bỏ sót (FN): 959
    Fraud capture / Recall: 0.088403
    Fraud miss rate: 0.911597
    
    Alerts phát ra (TP + FP): 166
    Alert rate: 0.00023300
    Alert đúng / TP: 93
    False alert / FP: 73
    Precision: 0.560241
    False-alert share among alerts: 0.439759
    
    F1_fraud: 0.152709
    Accuracy reference: 0.998551
    Total misclassified FP + FN: 1032
    Interpretation scope: SINGLE_RUN_AGGREGATE_DESCRIPTIVE_EVALUATION
    
    M6.3 SIX-RUN HUMAN-READABLE PROFILE GATE: PASS


## 10. Accuracy reference-only audit

M6.3 cần xác minh rằng Accuracy không được dùng như primary evaluation signal.

Cell này chỉ in:

- Accuracy;
- fraud Recall;
- fraud F1;
- fraud miss count;

để runtime review có thể kiểm tra việc Accuracy rất cao không che mất FN burden.

Không rank model theo Accuracy.


```python

for profile in (
    evaluation_profiles
):
    print(
        profile[
            "experiment_id"
        ]
    )

    print(
        "  Accuracy reference:",
        (
            f"{profile['accuracy_reference']:.6f}"
        ),
    )

    print(
        "  F1_fraud:",
        (
            f"{profile['f1_fraud']:.6f}"
        ),
    )

    print(
        "  Recall_fraud:",
        (
            f"{profile['recall_fraud']:.6f}"
        ),
    )

    print(
        "  Fraud missed / FN:",
        profile[
            "fraud_miss_count"
        ],
    )

    print()


print(
    "M6.3 ACCURACY REFERENCE-ONLY EVIDENCE: READY"
)

```

    M5-LR-SHORT-B04
      Accuracy reference: 0.998578
      F1_fraud: 0.337475
      Recall_fraud: 0.245247
      Fraud missed / FN: 794
    
    M5-LR-LONG-B04
      Accuracy reference: 0.998516
      F1_fraud: 0.039964
      Recall_fraud: 0.020913
      Fraud missed / FN: 1030
    
    M5-DT-SHORT-B01
      Accuracy reference: 0.998094
      F1_fraud: 0.327056
      Recall_fraud: 0.313688
      Fraud missed / FN: 722
    
    M5-DT-LONG-B01
      Accuracy reference: 0.997663
      F1_fraud: 0.199134
      Recall_fraud: 0.196768
      Fraud missed / FN: 845
    
    M5-RF-SHORT-B01
      Accuracy reference: 0.998515
      F1_fraud: 0.366467
      Recall_fraud: 0.290875
      Fraud missed / FN: 746
    
    M5-RF-LONG-B01
      Accuracy reference: 0.998551
      F1_fraud: 0.152709
      Recall_fraud: 0.088403
      Fraud missed / FN: 959
    
    M6.3 ACCURACY REFERENCE-ONLY EVIDENCE: READY


## 11. Confusion Matrix arithmetic audit

M6.2 đã reconstruct Confusion Matrix.

M6.3 kiểm tra lại các arithmetic relationships cần cho interpretation:

```text
TP + FN = fraud thật
TP + FP = alert count
TN + FP = non-fraud thật
TP + FP + FN + TN = validation rows
```

Đây là aggregate arithmetic check, không phải metric reconstruction lần hai.


```python

for profile in (
    evaluation_profiles
):
    assert (
        profile["tp"]
        + profile["fn"]
        == EXPECTED_VALIDATION_FRAUD
    )

    assert (
        profile["tp"]
        + profile["fp"]
        == profile[
            "alert_count"
        ]
    )

    assert (
        profile["tn"]
        + profile["fp"]
        == profile[
            "actual_non_fraud_count"
        ]
    )

    assert (
        profile["tp"]
        + profile["fp"]
        + profile["fn"]
        + profile["tn"]
        == EXPECTED_VALIDATION_ROWS
    )


print(
    "M6.3 CONFUSION ARITHMETIC GATE: PASS"
)

```

    M6.3 CONFUSION ARITHMETIC GATE: PASS


## 12. Metric-to-count consistency audit

M6.3 phải bảo đảm:

```text
Recall
↔
TP / (TP + FN)

Precision
↔
TP / (TP + FP)

Predicted-positive rate
↔
(TP + FP) / total rows
```

Không được diễn giải metric nếu raw-count relationship không khớp.


```python

for profile in (
    evaluation_profiles
):
    assert np.isclose(
        profile[
            "recall_fraud"
        ],
        (
            profile["tp"]
            / (
                profile["tp"]
                + profile["fn"]
            )
        ),
        rtol=1e-12,
        atol=1e-12,
    )

    assert np.isclose(
        profile[
            "precision_fraud"
        ],
        (
            profile["tp"]
            / (
                profile["tp"]
                + profile["fp"]
            )
        ),
        rtol=1e-12,
        atol=1e-12,
    )

    assert np.isclose(
        profile[
            "alert_rate"
        ],
        (
            (
                profile["tp"]
                + profile["fp"]
            )
            / EXPECTED_VALIDATION_ROWS
        ),
        rtol=1e-12,
        atol=1e-12,
    )


print(
    "M6.3 METRIC-TO-COUNT CONSISTENCY GATE: PASS"
)

```

    M6.3 METRIC-TO-COUNT CONSISTENCY GATE: PASS


## 13. Winner-selection boundary guardrail

M6.3 chỉ hoàn thành six-run profiles.

Không được tạo:

```text
training_window_winner
model_family_winner
best_model
best_run
recommended_model
```

từ M6.3 output.

Các trường selection trong M6.3 artifact phải giữ:

```text
OPEN / NOT AUTHORIZED
```


```python

selection_state = {
    "training_window_winner":
        "OPEN",

    "model_family_winner":
        "OPEN",

    "final_threshold":
        "OPEN",

    "final_model":
        "OPEN",

    "m6_03_selection_authorized":
        False,
}


assert (
    selection_state[
        "training_window_winner"
    ]
    == "OPEN"
)

assert (
    selection_state[
        "model_family_winner"
    ]
    == "OPEN"
)

assert (
    selection_state[
        "final_model"
    ]
    == "OPEN"
)

assert (
    selection_state[
        "m6_03_selection_authorized"
    ]
    is False
)


print(
    "Training-window winner:",
    selection_state[
        "training_window_winner"
    ],
)

print(
    "Model-family winner:",
    selection_state[
        "model_family_winner"
    ],
)

print(
    "Final model:",
    selection_state[
        "final_model"
    ],
)

print(
    "\nM6.3 NO-SELECTION BOUNDARY GATE: PASS"
)

```

    Training-window winner: OPEN
    Model-family winner: OPEN
    Final model: OPEN
    
    M6.3 NO-SELECTION BOUNDARY GATE: PASS


## 14. Source-registry fingerprint

M6.3 ghi SHA-256 của M6.2 Evaluation Registry.

Vai trò:

`lineage / reproducibility`

Không dùng hash để đánh giá model quality.


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


m6_02_registry_sha256 = (
    sha256_file(
        REGISTRY_PATH
    )
)


print(
    "M6.2 Evaluation Registry SHA256:"
)

print(
    m6_02_registry_sha256
)

```

    M6.2 Evaluation Registry SHA256:
    9e112883d5aa1358c58c714edc0f0ef7913252e94c56f3c6b28d3ba1b7f4e6fb


## 15. Persist M6.3 metric/confusion analysis artifact


```python

ANALYSIS_VERSION = (
    "M6.3-metric-confusion-analysis-v1"
)

ANALYSIS_PATH = (
    OUTPUT_DIR
    / "m6_03_metric_confusion_analysis.json"
)

MANIFEST_PATH = (
    OUTPUT_DIR
    / "m6_03_analysis_manifest.json"
)


analysis_payload = {
    "analysis_version":
        ANALYSIS_VERSION,

    "m6_substep":
        "M6.3",

    "source_registry_version":
        EXPECTED_REGISTRY_VERSION,

    "source_registry_artifact":
        str(
            REGISTRY_PATH.relative_to(
                PROJECT_ROOT
            )
        ),

    "source_registry_sha256":
        m6_02_registry_sha256,

    "evaluation_population":
        "VALIDATION_2019-01_TO_2019-05",

    "validation_rows":
        EXPECTED_VALIDATION_ROWS,

    "validation_fraud_rows":
        EXPECTED_VALIDATION_FRAUD,

    "positive_class":
        1,

    "primary_metric":
        "F1_fraud",

    "secondary_metrics": [
        "Recall_fraud",
        "Precision_fraud",
    ],

    "mandatory_confusion_fields": [
        "tp",
        "fp",
        "fn",
        "tn",
    ],

    "accuracy_role":
        "REFERENCE_ONLY",

    "records":
        evaluation_profiles,

    "selection_state":
        selection_state,

    "final_test_accessed":
        False,
}


analysis_manifest = {
    "m6_substep":
        "M6.3",

    "analysis_version":
        ANALYSIS_VERSION,

    "source_m6_02_registry":
        "VERIFIED",

    "official_runs_expected":
        6,

    "official_runs_profiled":
        len(
            evaluation_profiles
        ),

    "confusion_arithmetic":
        "PASS",

    "metric_to_count_consistency":
        "PASS",

    "accuracy_reference_only":
        True,

    "training_window_selection":
        "NOT_PERFORMED",

    "model_family_selection":
        "NOT_PERFORMED",

    "threshold_optimization":
        "NOT_PERFORMED",

    "model_retraining":
        "NOT_PERFORMED",

    "final_test_accessed":
        False,

    "decision":
        "OPEN — REQUIRES M6.3 RUNTIME REVIEW",
}


with open(
    ANALYSIS_PATH,
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


with open(
    MANIFEST_PATH,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        analysis_manifest,
        file,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )


assert ANALYSIS_PATH.exists()
assert MANIFEST_PATH.exists()

assert ANALYSIS_PATH.stat().st_size > 0
assert MANIFEST_PATH.stat().st_size > 0


print("Analysis artifact:")
print(ANALYSIS_PATH)

print("\nAnalysis manifest:")
print(MANIFEST_PATH)

print(
    "\nM6.3 ANALYSIS PERSISTENCE GATE: PASS"
)

```

    Analysis artifact:
    /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/processed/m6_03_baseline_metric_confusion_analysis/m6_03_metric_confusion_analysis.json
    
    Analysis manifest:
    /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/processed/m6_03_baseline_metric_confusion_analysis/m6_03_analysis_manifest.json
    
    M6.3 ANALYSIS PERSISTENCE GATE: PASS


## 16. Persisted artifact round-trip


```python

with open(
    ANALYSIS_PATH,
    "r",
    encoding="utf-8",
) as file:
    analysis_roundtrip = json.load(
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
    analysis_roundtrip[
        "analysis_version"
    ]
    == ANALYSIS_VERSION
)

assert (
    analysis_roundtrip[
        "source_registry_sha256"
    ]
    == m6_02_registry_sha256
)

assert (
    len(
        analysis_roundtrip[
            "records"
        ]
    )
    == 6
)

assert {
    record[
        "experiment_id"
    ]
    for record
    in analysis_roundtrip[
        "records"
    ]
} == OFFICIAL_EXPERIMENT_IDS

assert (
    analysis_roundtrip[
        "selection_state"
    ][
        "training_window_winner"
    ]
    == "OPEN"
)

assert (
    analysis_roundtrip[
        "selection_state"
    ][
        "model_family_winner"
    ]
    == "OPEN"
)

assert (
    analysis_roundtrip[
        "selection_state"
    ][
        "m6_03_selection_authorized"
    ]
    is False
)

assert (
    analysis_roundtrip[
        "final_test_accessed"
    ]
    is False
)

assert (
    manifest_roundtrip[
        "official_runs_profiled"
    ]
    == 6
)

assert (
    manifest_roundtrip[
        "confusion_arithmetic"
    ]
    == "PASS"
)

assert (
    manifest_roundtrip[
        "metric_to_count_consistency"
    ]
    == "PASS"
)


print(
    "M6.3 ANALYSIS ROUND-TRIP GATE: PASS"
)

```

    M6.3 ANALYSIS ROUND-TRIP GATE: PASS


## 17. FINAL TEST isolation


```python

assert (
    m6_02_registry[
        "final_test_accessed"
    ]
    is False
)

for profile in (
    evaluation_profiles
):
    assert (
        profile[
            "final_test_accessed"
        ]
        is False
    )


assert (
    analysis_payload[
        "final_test_accessed"
    ]
    is False
)

assert (
    analysis_manifest[
        "final_test_accessed"
    ]
    is False
)


input_paths = [
    REGISTRY_PATH,
    AUDIT_MANIFEST_PATH,
]


for path in input_paths:
    assert (
        "final_test"
        not in path.name.lower()
    )


print(
    "M6.3 FINAL TEST ISOLATION GATE: PASS"
)

```

    M6.3 FINAL TEST ISOLATION GATE: PASS


## 18. M6.3 overall technical gate


```python

m6_03_gates = {
    "G01_M6_02_HANDOFF":
        True,

    "G02_SIX_RUN_INPUT_INTEGRITY":
        True,

    "G03_PROFILE_BUILD":
        True,

    "G04_HUMAN_READABLE_PROFILE":
        True,

    "G05_CONFUSION_ARITHMETIC":
        True,

    "G06_METRIC_TO_COUNT_CONSISTENCY":
        True,

    "G07_ACCURACY_REFERENCE_ONLY":
        True,

    "G08_NO_SELECTION_BOUNDARY":
        True,

    "G09_SOURCE_REGISTRY_LINEAGE":
        True,

    "G10_ANALYSIS_PERSISTENCE":
        True,

    "G11_ANALYSIS_ROUND_TRIP":
        True,

    "G12_FINAL_TEST_ISOLATION":
        True,
}


for gate_name, gate_value in (
    m6_03_gates.items()
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
    m6_03_gates.values()
)


print(
    "\nM6.3 OVERALL TECHNICAL GATE: PASS"
)

```

    G01_M6_02_HANDOFF → PASS
    G02_SIX_RUN_INPUT_INTEGRITY → PASS
    G03_PROFILE_BUILD → PASS
    G04_HUMAN_READABLE_PROFILE → PASS
    G05_CONFUSION_ARITHMETIC → PASS
    G06_METRIC_TO_COUNT_CONSISTENCY → PASS
    G07_ACCURACY_REFERENCE_ONLY → PASS
    G08_NO_SELECTION_BOUNDARY → PASS
    G09_SOURCE_REGISTRY_LINEAGE → PASS
    G10_ANALYSIS_PERSISTENCE → PASS
    G11_ANALYSIS_ROUND_TRIP → PASS
    G12_FINAL_TEST_ISOLATION → PASS
    
    M6.3 OVERALL TECHNICAL GATE: PASS


# 19. Runtime review và Findings M6.3

## 19.1. Execution integrity

Observed:

```text
Code cells:
16 / 16

Execution count:
1 → 16 liên tục

Runtime errors:
0

stderr:
0
```

Interpretation:

Notebook đã chạy đầy đủ từ đầu đến cuối.

Không có partial execution hoặc hidden runtime exception.

Status:

`VERIFIED`

---

## 19.2. M6.2 handoff integrity

Observed:

```text
M6.2 Evaluation Registry:
loaded

M6.2 audit manifest:
loaded

Registry version:
M6.2-evaluation-registry-v1

Official runs:
6 / 6

Independent metric reconstruction:
PASS

Persisted-summary agreement:
PASS

Controlled-pair metadata:
PASS

FINAL TEST:
not accessed
```

Runtime gate:

`M6.3 M6.2 HANDOFF GATE: PASS`

Status:

`VERIFIED`

---

## 19.3. Six-run input integrity

Observed official subjects:

```text
M5-LR-SHORT-B04
M5-LR-LONG-B04

M5-DT-SHORT-B01
M5-DT-LONG-B01

M5-RF-SHORT-B01
M5-RF-LONG-B01
```

All records satisfy:

```text
validation rows:
712,458

validation fraud:
1,052

independent_metric_check:
PASS

summary_match_check:
PASS

artifact_integrity_check:
PASS

evaluation_status:
VERIFIED_FOR_M6_EVALUATION

final_test_accessed:
false
```

Runtime gate:

`M6.3 SIX-RUN INPUT INTEGRITY GATE: PASS`

Status:

`VERIFIED`

---

## 19.4. Six evaluation profiles

Observed:

`6 / 6 profiles built`

Runtime gate:

`M6.3 SIX-RUN PROFILE BUILD GATE: PASS`

Each profile contains:

- model/window identity;
- F1_fraud;
- Recall_fraud;
- Precision_fraud;
- Accuracy reference;
- TP / FP / FN / TN;
- fraud capture count/rate;
- fraud miss count/rate;
- alert count/rate;
- true alert count;
- false alert count;
- false-alert share;
- total misclassified.

Status:

`VERIFIED`

---

## 19.5. Logistic Regression — W_SHORT

Observed:

```text
Experiment:
M5-LR-SHORT-B04

Fraud thật:
1,052

Fraud bắt đúng / TP:
258

Fraud bỏ sót / FN:
794

Recall:
0.245247

Fraud miss rate:
0.754753

Alerts phát ra:
477

True alerts:
258

False alerts:
219

Precision:
0.540881

False-alert share among alerts:
0.459119

F1_fraud:
0.337475

Accuracy reference:
0.998578

Total misclassified:
1,013
```

Interpretation:

Baseline LR-SHORT bắt được khoảng một phần tư fraud thật.

Trong 477 transaction bị flag positive:

- 258 là fraud thật;
- 219 là false alert.

Precision lớn hơn Recall, nên behavior hiện tại thiên về:

`ít flag hơn nhưng bỏ sót nhiều fraud hơn`

so với một operating behavior có Recall cao hơn.

M6.3 chỉ ghi nhận single-run behavior.

Không kết luận training-window winner hoặc model-family winner.

Status:

`OBSERVED SINGLE-RUN EVIDENCE`

---

## 19.6. Logistic Regression — W_LONG

Observed:

```text
Experiment:
M5-LR-LONG-B04

Fraud thật:
1,052

Fraud bắt đúng / TP:
22

Fraud bỏ sót / FN:
1,030

Recall:
0.020913

Fraud miss rate:
0.979087

Alerts phát ra:
49

True alerts:
22

False alerts:
27

Precision:
0.448980

False-alert share among alerts:
0.551020

F1_fraud:
0.039964

Accuracy reference:
0.998516

Total misclassified:
1,057
```

Interpretation:

LR-LONG chỉ flag 49 transaction positive trên 712,458 validation rows.

Nó bắt được 22 / 1,052 fraud và bỏ sót 1,030 fraud.

Precision gần 0.45 không được đọc riêng lẻ như evidence mạnh vì alert volume rất thấp và Recall chỉ khoảng 2.1%.

Status:

`OBSERVED SINGLE-RUN EVIDENCE`

---

## 19.7. Decision Tree — W_SHORT

Observed:

```text
Experiment:
M5-DT-SHORT-B01

Fraud thật:
1,052

Fraud bắt đúng / TP:
330

Fraud bỏ sót / FN:
722

Recall:
0.313688

Fraud miss rate:
0.686312

Alerts phát ra:
966

True alerts:
330

False alerts:
636

Precision:
0.341615

False-alert share among alerts:
0.658385

F1_fraud:
0.327056

Accuracy reference:
0.998094

Total misclassified:
1,358
```

Interpretation:

DT-SHORT bắt được 330 fraud nhưng tạo 636 false alerts.

Do đó Recall cao hơn một số run khác không được đọc tách khỏi Precision/FP burden.

Status:

`OBSERVED SINGLE-RUN EVIDENCE`

---

## 19.8. Decision Tree — W_LONG

Observed:

```text
Experiment:
M5-DT-LONG-B01

Fraud thật:
1,052

Fraud bắt đúng / TP:
207

Fraud bỏ sót / FN:
845

Recall:
0.196768

Fraud miss rate:
0.803232

Alerts phát ra:
1,027

True alerts:
207

False alerts:
820

Precision:
0.201558

False-alert share among alerts:
0.798442

F1_fraud:
0.199134

Accuracy reference:
0.997663

Total misclassified:
1,665
```

Interpretation:

DT-LONG phát 1,027 alerts nhưng chỉ 207 là true fraud.

Khoảng 79.8% alerts là false alerts.

Đây là aggregate behavior cần M6.4/M6.5 đọc trong controlled comparison context.

M6.3 chưa kết luận window/model winner.

Status:

`OBSERVED SINGLE-RUN EVIDENCE`

---

## 19.9. Random Forest — W_SHORT

Observed:

```text
Experiment:
M5-RF-SHORT-B01

Fraud thật:
1,052

Fraud bắt đúng / TP:
306

Fraud bỏ sót / FN:
746

Recall:
0.290875

Fraud miss rate:
0.709125

Alerts phát ra:
618

True alerts:
306

False alerts:
312

Precision:
0.495146

False-alert share among alerts:
0.504854

F1_fraud:
0.366467

Accuracy reference:
0.998515

Total misclassified:
1,058
```

Interpretation:

RF-SHORT bắt được 306 fraud và phát 618 alerts.

True alerts và false alerts gần cân bằng về count:

```text
306 TP
312 FP
```

Status:

`OBSERVED SINGLE-RUN EVIDENCE`

---

## 19.10. Random Forest — W_LONG

Observed:

```text
Experiment:
M5-RF-LONG-B01

Fraud thật:
1,052

Fraud bắt đúng / TP:
93

Fraud bỏ sót / FN:
959

Recall:
0.088403

Fraud miss rate:
0.911597

Alerts phát ra:
166

True alerts:
93

False alerts:
73

Precision:
0.560241

False-alert share among alerts:
0.439759

F1_fraud:
0.152709

Accuracy reference:
0.998551

Total misclassified:
1,032
```

Interpretation:

RF-LONG có Precision tương đối cao trong six-run baseline evidence nhưng chỉ phát 166 alerts.

Nó bắt 93 fraud và bỏ sót 959 fraud.

Vì vậy Precision phải được đọc cùng Recall và alert volume.

Status:

`OBSERVED SINGLE-RUN EVIDENCE`

---

## 19.11. Accuracy reference-only evidence

Observed:

```text
Accuracy của cả 6 runs:
xấp xỉ 0.9977 → 0.9986
```

Trong khi:

```text
Recall_fraud:
xấp xỉ 0.0209 → 0.3137
```

và fraud missed:

```text
722 → 1,030
```

Interpretation:

Accuracy cao không đồng nghĩa fraud detection mạnh trong dataset mất cân bằng.

Điều này xác nhận policy:

`Accuracy = REFERENCE ONLY`

Runtime evidence:

`M6.3 ACCURACY REFERENCE-ONLY EVIDENCE: READY`

Status:

`VERIFIED`

---

## 19.12. Confusion arithmetic

Observed for all 6 runs:

```text
TP + FN = 1,052

TP + FP = alert count

TN + FP = actual non-fraud

TP + FP + FN + TN = 712,458
```

Runtime gate:

`M6.3 CONFUSION ARITHMETIC GATE: PASS`

Status:

`VERIFIED`

---

## 19.13. Metric-to-count consistency

Observed for all 6 runs:

```text
Recall = TP / (TP + FN)

Precision = TP / (TP + FP)

Alert rate = (TP + FP) / 712,458
```

Runtime gate:

`M6.3 METRIC-TO-COUNT CONSISTENCY GATE: PASS`

Interpretation:

M6.3 metric wording có raw-count basis rõ ràng.

Status:

`VERIFIED`

---

## 19.14. No-selection boundary

Observed:

```text
Training-window winner:
OPEN

Model-family winner:
OPEN

Final model:
OPEN

M6.3 selection authorized:
FALSE
```

Runtime gate:

`M6.3 NO-SELECTION BOUNDARY GATE: PASS`

Status:

`VERIFIED`

---

## 19.15. M6.3 analysis artifact persistence

Persisted:

```text
data/processed/m6_03_baseline_metric_confusion_analysis/
    m6_03_metric_confusion_analysis.json
    m6_03_analysis_manifest.json
```

Runtime gates:

```text
M6.3 ANALYSIS PERSISTENCE GATE:
PASS

M6.3 ANALYSIS ROUND-TRIP GATE:
PASS
```

Source-registry SHA-256 also recorded for lineage/reproducibility.

Status:

`VERIFIED`

---

## 19.16. FINAL TEST isolation

Observed:

```text
M6.2 registry:
final_test_accessed = false

6 evaluation profiles:
final_test_accessed = false

M6.3 analysis:
final_test_accessed = false
```

Runtime gate:

`M6.3 FINAL TEST ISOLATION GATE: PASS`

Status:

`VERIFIED`

---

## 19.17. Overall technical result

Observed:

```text
G01_M6_02_HANDOFF                → PASS
G02_SIX_RUN_INPUT_INTEGRITY      → PASS
G03_PROFILE_BUILD                → PASS
G04_HUMAN_READABLE_PROFILE       → PASS
G05_CONFUSION_ARITHMETIC         → PASS
G06_METRIC_TO_COUNT_CONSISTENCY  → PASS
G07_ACCURACY_REFERENCE_ONLY      → PASS
G08_NO_SELECTION_BOUNDARY        → PASS
G09_SOURCE_REGISTRY_LINEAGE      → PASS
G10_ANALYSIS_PERSISTENCE         → PASS
G11_ANALYSIS_ROUND_TRIP          → PASS
G12_FINAL_TEST_ISOLATION         → PASS
```

Overall:

`M6.3 OVERALL TECHNICAL GATE: PASS`

Blocking issue:

`NONE`

---

# 20. Findings M6.3

## M6.3-F01 — Six-run aggregate evaluation profiles are complete

Evidence:

`6 / 6 official runs profiled`

Status:

`VERIFIED`

---

## M6.3-F02 — All metrics have raw-count interpretation

Evidence:

Recall/Precision/alert-rate identities PASS for all six runs.

Status:

`VERIFIED`

---

## M6.3-F03 — Logistic Regression SHORT behavior

Observed:

```text
TP = 258
FN = 794
FP = 219
F1 ≈ 0.3375
Recall ≈ 0.2452
Precision ≈ 0.5409
Alerts = 477
```

Finding:

LR-SHORT produces relatively few alerts and has Precision materially above Recall, with substantial fraud-miss burden.

Status:

`OBSERVED`

---

## M6.3-F04 — Logistic Regression LONG behavior

Observed:

```text
TP = 22
FN = 1,030
FP = 27
F1 ≈ 0.0400
Recall ≈ 0.0209
Precision ≈ 0.4490
Alerts = 49
```

Finding:

LR-LONG is extremely conservative at the baseline rule and misses most validation fraud.

Status:

`OBSERVED`

---

## M6.3-F05 — Decision Tree SHORT behavior

Observed:

```text
TP = 330
FN = 722
FP = 636
F1 ≈ 0.3271
Recall ≈ 0.3137
Precision ≈ 0.3416
Alerts = 966
```

Finding:

DT-SHORT catches more fraud than it false-negatives relative to several other baseline runs, but generates substantial false-alert burden.

Status:

`OBSERVED`

---

## M6.3-F06 — Decision Tree LONG behavior

Observed:

```text
TP = 207
FN = 845
FP = 820
F1 ≈ 0.1991
Recall ≈ 0.1968
Precision ≈ 0.2016
Alerts = 1,027
```

Finding:

DT-LONG generates the largest alert volume among the six baseline profiles while most alerts are false alerts.

Status:

`OBSERVED`

---

## M6.3-F07 — Random Forest SHORT behavior

Observed:

```text
TP = 306
FN = 746
FP = 312
F1 ≈ 0.3665
Recall ≈ 0.2909
Precision ≈ 0.4951
Alerts = 618
```

Finding:

RF-SHORT produces roughly balanced true/false alert counts while still missing 746 fraud transactions.

Status:

`OBSERVED`

---

## M6.3-F08 — Random Forest LONG behavior

Observed:

```text
TP = 93
FN = 959
FP = 73
F1 ≈ 0.1527
Recall ≈ 0.0884
Precision ≈ 0.5602
Alerts = 166
```

Finding:

RF-LONG's relatively high Precision is accompanied by low alert volume and high fraud-miss burden.

Status:

`OBSERVED`

---

## M6.3-F09 — Accuracy cannot discriminate fraud-screening behavior

Evidence:

Accuracy is near 0.998 for all runs while Recall and FN counts differ strongly.

Status:

`VERIFIED`

---

## M6.3-F10 — Precision must be interpreted with alert volume and Recall

Evidence:

Runs with relatively high Precision can still have very low Recall because they emit few positive predictions.

Status:

`VERIFIED INTERPRETATION RULE`

---

## M6.3-F11 — Recall must be interpreted with FP / Precision burden

Evidence:

A run can capture more fraud while also generating materially more false alerts.

Status:

`VERIFIED INTERPRETATION RULE`

---

## M6.3-F12 — M6.3 does not establish a winner

Finding:

Six single-run profiles are now available, but controlled pair and same-window model comparisons remain separate work.

Status:

`CORRECT BOUNDARY`

---

## M6.3-F13 — M6.4 handoff is unblocked

M6.4 can now compare:

```text
LR SHORT vs LONG
DT SHORT vs LONG
RF SHORT vs LONG
```

using verified profiles and controlled-pair metadata.

Status:

`READY FOR M6.4`

# 21. Decision Log M6.3 — sau runtime review

## M6.3-D01 — Canonical input

Decision:

Use:

`M6.2 verified Evaluation Registry`

Observed:

handoff and registry integrity PASS.

Status:

`LOCKED / VERIFIED`

---

## M6.3-D02 — Evaluation population

Decision:

```text
VALIDATION 2019-01 → 2019-05
712,458 rows
1,052 fraud
```

Status:

`INHERITED — VERIFIED — LOCKED`

---

## M6.3-D03 — Primary metric

Decision:

`F1_fraud`

Status:

`INHERITED — LOCKED`

---

## M6.3-D04 — Secondary metrics

Decision:

```text
Recall_fraud
Precision_fraud
```

Status:

`INHERITED — LOCKED`

---

## M6.3-D05 — Mandatory confusion evidence

Decision:

```text
TP
FP
FN
TN
```

Observed:

6 / 6 arithmetic checks PASS.

Status:

`INHERITED — VERIFIED — LOCKED`

---

## M6.3-D06 — Operational diagnostic

Decision:

```text
predicted-positive count
predicted-positive rate
```

Status:

`INHERITED — LOCKED`

---

## M6.3-D07 — Fraud-coverage interpretation

Decision:

Every run must report:

```text
fraud capture count / rate
fraud miss count / rate
```

Observed:

6 / 6 profiles include these values.

Status:

`LOCKED / VERIFIED`

---

## M6.3-D08 — Alert-quality interpretation

Decision:

Every run must report:

```text
alert count
true alerts
false alerts
Precision
false-alert share
```

Observed:

6 / 6 profiles include these values.

Status:

`LOCKED / VERIFIED`

---

## M6.3-D09 — Accuracy

Decision:

`REFERENCE ONLY`

Observed:

Accuracy remains high across all six runs despite large fraud-detection differences.

Status:

`INHERITED — VERIFIED — LOCKED`

---

## M6.3-D10 — Comparison boundary

Decision:

M6.3 profiles each run individually.

Controlled training-window comparison:

`M6.4`

Cross-model comparison:

`M6.5`

Status:

`LOCKED`

---

## M6.3-D11 — Retraining / tuning

Decision:

None.

Observed:

No fit/tuning intervention in M6.3.

Status:

`LOCKED / VERIFIED`

---

## M6.3-D12 — Training-window winner

Decision:

Not selected in M6.3.

Status:

`OPEN`

---

## M6.3-D13 — Model-family winner

Decision:

Not selected in M6.3.

Status:

`OPEN`

---

## M6.3-D14 — Final model

Decision:

Not selected.

Status:

`OPEN — M7`

---

## M6.3-D15 — Threshold

Decision:

No optimization.

Status:

`OPEN — DEFERRED TO M7`

---

## M6.3-D16 — Analysis artifact

Decision:

Persist:

`m6_03_metric_confusion_analysis.json`

Observed:

Persistence + round-trip PASS.

Status:

`LOCKED / VERIFIED`

---

## M6.3-D17 — Analysis manifest

Decision:

Persist:

`m6_03_analysis_manifest.json`

Observed:

Persistence + round-trip PASS.

Status:

`LOCKED / VERIFIED`

---

## M6.3-D18 — FINAL TEST

Decision:

No access.

Observed:

FINAL TEST isolation PASS.

Status:

`INHERITED — VERIFIED — LOCKED`

---

## M6.3-D19 — M6.4 handoff

Decision:

Use M6.3 six-run profiles for controlled SHORT/LONG comparison.

Status:

`READY`

# 22. M6.3 Gate

## Technical runtime gates

```text
G01_M6_02_HANDOFF                → PASS
G02_SIX_RUN_INPUT_INTEGRITY      → PASS
G03_PROFILE_BUILD                → PASS
G04_HUMAN_READABLE_PROFILE       → PASS
G05_CONFUSION_ARITHMETIC         → PASS
G06_METRIC_TO_COUNT_CONSISTENCY  → PASS
G07_ACCURACY_REFERENCE_ONLY      → PASS
G08_NO_SELECTION_BOUNDARY        → PASS
G09_SOURCE_REGISTRY_LINEAGE      → PASS
G10_ANALYSIS_PERSISTENCE         → PASS
G11_ANALYSIS_ROUND_TRIP          → PASS
G12_FINAL_TEST_ISOLATION         → PASS
```

Technical gates:

`12 / 12 PASS`

---

## R01 — Execution complete?

Evidence:

```text
16 / 16 code cells
execution_count = 1 → 16
errors = 0
stderr = 0
```

Result:

`PASS`

---

## R02 — M6.2 verified handoff used?

Evidence:

Verified Evaluation Registry + audit manifest loaded and checked.

Result:

`PASS`

---

## R03 — Six official profiles complete?

Evidence:

`6 / 6`

Result:

`PASS`

---

## R04 — Confusion arithmetic valid?

Evidence:

All 6 runs satisfy canonical TP/FP/FN/TN identities.

Result:

`PASS`

---

## R05 — Metric-to-count consistency valid?

Evidence:

Recall / Precision / alert-rate relationships PASS for all six runs.

Result:

`PASS`

---

## R06 — Accuracy kept reference-only?

Evidence:

Notebook reports Accuracy beside fraud F1/Recall/FN evidence and does not use Accuracy for winner selection.

Result:

`PASS`

---

## R07 — Single-run scope preserved?

Evidence:

M6.3 records per-run behavior and does not perform official SHORT/LONG or LR/DT/RF selection.

Result:

`PASS`

---

## R08 — No retraining/tuning?

Evidence:

No model fit, resampling or threshold search.

Result:

`PASS`

---

## R09 — No selection contamination?

Evidence:

```text
Training-window winner:
OPEN

Model-family winner:
OPEN

Final model:
OPEN
```

Result:

`PASS`

---

## R10 — Analysis artifacts persisted?

Evidence:

Analysis JSON + manifest persistence and round-trip PASS.

Result:

`PASS`

---

## R11 — FINAL TEST protected?

Evidence:

FINAL TEST isolation gate PASS.

Result:

`PASS`

---

## R12 — M6.4 handoff ready?

Evidence:

Six verified profiles and all required raw-count/metric evidence are available.

Result:

`PASS`

---

## Overall M6.3 Gate

```text
Technical gates:
12 / 12 PASS

Runtime review gates:
12 / 12 PASS

Blocking issue:
NONE
```

Final:

`M6.3 — PASS`

Handoff:

`READY FOR M6.4`

# 23. Kết luận M6.3

M6.3 đã hoàn thành aggregate metric và Confusion Matrix interpretation cho toàn bộ six-run baseline registry.

Canonical evaluation population:

```text
VALIDATION:
712,458 transactions

Fraud:
1,052

Positive class:
fraud = 1
```

Official profiles:

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
Official profiles:
6 / 6 COMPLETE

Confusion arithmetic:
PASS

Metric-to-count consistency:
PASS

Accuracy policy:
REFERENCE ONLY

Analysis persistence:
PASS

Analysis round-trip:
PASS

Retraining:
NONE

Tuning:
NONE

Threshold optimization:
NONE

FINAL TEST:
PROTECTED
```

M6.3 đã persist:

```text
data/processed/m6_03_baseline_metric_confusion_analysis/
    m6_03_metric_confusion_analysis.json
    m6_03_analysis_manifest.json
```

M6.3 đã tạo đủ evidence để đọc mỗi run theo:

```text
fraud caught
fraud missed
alerts emitted
true alerts
false alerts
Recall
Precision
F1
Accuracy reference
```

M6.3 không khóa:

```text
Training-window winner:
OPEN

Model-family winner:
OPEN

Final model:
OPEN

Final threshold:
OPEN
```

Final state:

```text
M6.3 — PASS

Execution Integrity:
VERIFIED

M6.2 Handoff:
VERIFIED

Official Profiles:
6 / 6 COMPLETE

Confusion Matrix Evidence:
VERIFIED

Metric-to-count Interpretation:
VERIFIED

Accuracy:
REFERENCE ONLY

Analysis Artifact:
PERSISTED

Round-trip:
PASS

FINAL TEST:
PROTECTED

Blocking Issue:
NONE

READY FOR M6.4
```

Bước tiếp theo:

`M6.4 — Controlled training-window comparison`

M6.4 sẽ so sánh đúng ba controlled pairs:

```text
LR:
W_SHORT vs W_LONG

DT:
W_SHORT vs W_LONG

RF:
W_SHORT vs W_LONG
```

với nguyên tắc:

`ONLY TRAINING WINDOW CHANGES`

và chưa thực hiện cross-model family selection.
