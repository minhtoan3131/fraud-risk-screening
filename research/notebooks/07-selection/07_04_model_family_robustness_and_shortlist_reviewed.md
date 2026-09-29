# M7.4 — Model-family robustness và shortlist

Milestone:

`M7 — MODEL SELECTION / ROBUSTNESS / TUNING`

Substep:

`M7.4 — Model-family robustness / shortlist`

Primary question:

> Sau khi M7.3 đã khóa training-window scope là W_SHORT, trade-off giữa Logistic Regression, Decision Tree và Random Forest có ổn định qua Q2/Q3/Q4-2018 temporal folds không, và evidence hiện tại hỗ trợ shortlist nào cho downstream M7?

M7.4 dùng lại **exact runtime evidence** từ M7.3.

Không retrain lại 9 W_SHORT fold runs vì M7.3 đã thực hiện đúng:

```text
3 model families
×
3 temporal folds
=
9 W_SHORT fold runs
```

với fold-safe preprocessing và controlled comparison contract.

M7.4 được phép:

- audit upstream M7.3 handoff;
- chỉ dùng W_SHORT;
- so LR / DT / RF trên cùng temporal folds;
- đọc mean F1 / std F1;
- đọc fold-wise F1 / Recall / Precision;
- đọc TP / FP / FN / TN;
- đọc alert burden;
- đọc runtime;
- đối chiếu M6 frozen supporting evidence;
- tạo evidence cho `MODEL SHORTLIST`.

M7.4 không được:

- đổi training window;
- tune hyperparameters;
- thử class-weight/resampling;
- tối ưu threshold;
- chọn final model;
- mở FINAL TEST.

Runtime-dependent shortlist:

`OPEN — REQUIRES M7.4 RUNTIME REVIEW`

## 1. CANON comparison contract

Training-window scope đã được M7.3 review và khóa:

`W_SHORT — LOCKED FOR DOWNSTREAM M7`

M7.4 primary comparison giữ:

```text
same training window:
W_SHORT

same feature/preprocessing protocol:
YES

same temporal folds:
Q2 / Q3 / Q4 2018

same imbalance strategy:
NONE

same threshold policy:
DEFAULT_MODEL_DECISION_RULE

same metric implementation:
YES

same random-state policy:
42
```

Biến khác nhau:

`model family / official baseline config`

Candidates:

```text
Logistic Regression
LR-B04-LBFGS-L2-C1

Decision Tree
DT-B01-DEFAULT-GINI-UNPRUNED

Random Forest
RF-B01-100-GINI-SQRT-BOOTSTRAP
```

Shortlist không được tạo chỉ bằng one-number ranking.

Evidence phải đọc cùng nhau:

```text
F1 primary
Recall / Precision
fold stability
FP / FN
alert burden
computational cost
M6 error findings
```

## 2. Predeclared shortlist reading policy

M7.4 không dùng weighted score tự chế.

Không đặt arbitrary epsilon kiểu:

`ΔF1 < x → coi như hòa`.

Reading order:

```text
1.
mean F1
+ fold-wise F1

2.
std F1
+ fold consistency

3.
Recall / FN

4.
Precision / FP

5.
alert burden

6.
computational cost

7.
M6 error complementarity / score behavior

8.
shortlist review
```

Một model có thể được giữ trong shortlist dù không dẫn primary F1 nếu nó có compensating evidence rõ, ví dụ:

- Recall/FN strength;
- Precision/FP/alert strength;
- computational advantage;
- complementary fraud catches.

Một model chỉ nên bị loại nếu reviewed evidence cho thấy nó không còn competitive và không có compensating role đủ rõ.

M7.4 notebook chỉ tạo evidence package.

Final shortlist vẫn:

`OPEN — REQUIRES RUNTIME REVIEW`.


```python

from pathlib import Path
from collections import Counter, defaultdict
import copy
import hashlib
import json
import platform
import sys

import numpy as np
import pandas as pd


print("Python:")
print(sys.version)

print("\nExecutable:")
print(sys.executable)

print("\nPlatform:")
print(platform.platform())

print("\nNumPy:")
print(np.__version__)

print("\npandas:")
print(pd.__version__)

```

    Python:
    3.14.6 (main, Jun 10 2026, 10:03:53) [Clang 21.0.0 (clang-2100.0.123.102)]
    
    Executable:
    /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/.venv/bin/python
    
    Platform:
    macOS-26.6.2-arm64-arm-64bit-Mach-O
    
    NumPy:
    2.5.3
    
    pandas:
    3.0.5


## 3. Locate M7.3 reviewed handoff artifacts


```python

M7_03_REL = (
    Path("data")
    / "processed"
    / "m7_03_training_window_robustness"
)

M7_04_REL = (
    Path("data")
    / "processed"
    / "m7_04_model_family_robustness_shortlist"
)


required_rel_paths = [
    M7_03_REL
    / "m7_03_training_window_robustness.json",

    M7_03_REL
    / "m7_03_robustness_manifest.json",
]


candidate_roots = [
    Path.cwd(),
    *list(
        Path.cwd().parents
    )[:6],
]


PROJECT_ROOT = None


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
        "M7.3 result + manifest artifacts."
    )


M7_03_DIR = (
    PROJECT_ROOT
    / M7_03_REL
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / M7_04_REL
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


M7_03_RESULT_PATH = (
    M7_03_DIR
    / "m7_03_training_window_robustness.json"
)

M7_03_MANIFEST_PATH = (
    M7_03_DIR
    / "m7_03_robustness_manifest.json"
)


M7_04_RESULT_PATH = (
    OUTPUT_DIR
    / "m7_04_model_family_robustness.json"
)

M7_04_MANIFEST_PATH = (
    OUTPUT_DIR
    / "m7_04_shortlist_manifest.json"
)


M7_04_ANALYSIS_VERSION = (
    "M7.4-model-family-robustness-shortlist-v1"
)

UPSTREAM_REVIEWED_M7_3_STATUS = (
    "PASS"
)

UPSTREAM_REVIEWED_TRAINING_WINDOW = (
    "W_SHORT"
)

UPSTREAM_REVIEWED_M7_3_NOTEBOOK_SHA256 = (
    "da1b8d14954a6c79825e81071950aea29"
    "b113f7c6d0d84989d0e5f473c6198d4"
)


print("PROJECT_ROOT:")
print(PROJECT_ROOT)

print("\nM7.3 result:")
print(M7_03_RESULT_PATH)

print("\nM7.4 output:")
print(OUTPUT_DIR)

print(
    "\nM7.4 SOURCE LOCATION GATE: PASS"
)

```

    PROJECT_ROOT:
    /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening
    
    M7.3 result:
    /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/processed/m7_03_training_window_robustness/m7_03_training_window_robustness.json
    
    M7.4 output:
    /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/processed/m7_04_model_family_robustness_shortlist
    
    M7.4 SOURCE LOCATION GATE: PASS


## 4. Important handoff note

M7.3 runtime JSON được persist **trước** human/AI runtime review nên field:

`training_window_decision`

vẫn có thể ghi:

`OPEN — REQUIRES M7.3 RUNTIME REVIEW`.

Sau đó reviewed notebook đã khóa:

`W_SHORT — LOCKED FOR DOWNSTREAM M7`.

M7.4 không sửa ngược runtime artifact M7.3.

Thay vào đó M7.4:

1. giữ reviewed handoff `W_SHORT`;
2. independently verify raw M7.3 runtime evidence:
   - 18/18 runs;
   - 0 warnings;
   - 3/3 pair states = ROBUST W_SHORT PREFERENCE;
   - cross-family state = ROBUST W_SHORT PREFERENCE.

Nếu raw evidence không khớp reviewed handoff:

`STOP`.


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
    M7_03_RESULT_PATH,
    "r",
    encoding="utf-8",
) as file:
    m7_03_result = json.load(
        file
    )


with open(
    M7_03_MANIFEST_PATH,
    "r",
    encoding="utf-8",
) as file:
    m7_03_manifest = json.load(
        file
    )


assert (
    m7_03_result[
        "analysis_version"
    ]
    == "M7.3-training-window-robustness-v1"
)

assert (
    m7_03_manifest[
        "completed_run_count"
    ]
    == 18
)

assert (
    m7_03_manifest[
        "warning_count_total"
    ]
    == 0
)

assert (
    m7_03_manifest[
        "integrity_pass_run_count"
    ]
    == 18
)

assert (
    m7_03_manifest[
        "only_active_variable"
    ]
    == "training_window_id"
)

assert (
    m7_03_manifest[
        "model_selection_performed"
    ]
    is False
)

assert (
    m7_03_manifest[
        "final_test_accessed"
    ]
    is False
)


pair_comparisons = (
    m7_03_result[
        "pair_comparisons"
    ]
)

assert len(
    pair_comparisons
) == 3

assert all(
    pair[
        "automated_robustness_state"
    ]
    ==
    "ROBUST W_SHORT PREFERENCE"
    for pair
    in pair_comparisons
)

assert (
    m7_03_result[
        "cross_family_robustness_state"
    ]
    ==
    "ROBUST W_SHORT PREFERENCE"
)

assert (
    UPSTREAM_REVIEWED_M7_3_STATUS
    == "PASS"
)

assert (
    UPSTREAM_REVIEWED_TRAINING_WINDOW
    == "W_SHORT"
)


print(
    "M7.3 completed runs:",
    m7_03_manifest[
        "completed_run_count"
    ],
)

print(
    "M7.3 warnings:",
    m7_03_manifest[
        "warning_count_total"
    ],
)

print(
    "Cross-family training-window evidence:"
)

print(
    m7_03_result[
        "cross_family_robustness_state"
    ]
)

print(
    "Reviewed downstream window:"
)

print(
    UPSTREAM_REVIEWED_TRAINING_WINDOW
)

print(
    "\nM7.4 M7.3 REVIEWED HANDOFF GATE: PASS"
)

```

    M7.3 completed runs: 18
    M7.3 warnings: 0
    Cross-family training-window evidence:
    ROBUST W_SHORT PREFERENCE
    Reviewed downstream window:
    W_SHORT
    
    M7.4 M7.3 REVIEWED HANDOFF GATE: PASS


## 5. Filter exact W_SHORT model-family evidence


```python

fold_results = (
    m7_03_result[
        "fold_results"
    ]
)

aggregate_records = (
    m7_03_result[
        "aggregate_records"
    ]
)


w_short_fold_results = [
    record
    for record
    in fold_results
    if (
        record[
            "training_window_id"
        ]
        == "W_SHORT"
    )
]


w_short_aggregates = [
    record
    for record
    in aggregate_records
    if (
        record[
            "training_window_id"
        ]
        == "W_SHORT"
    )
]


assert len(
    w_short_fold_results
) == 9

assert len(
    w_short_aggregates
) == 3


MODEL_KEYS = [
    "LR",
    "DT",
    "RF",
]


MODEL_LABELS = {
    "LR":
        "Logistic Regression",

    "DT":
        "Decision Tree",

    "RF":
        "Random Forest",
}


EXPECTED_CONFIGS = {
    "LR":
        "LR-B04-LBFGS-L2-C1",

    "DT":
        "DT-B01-DEFAULT-GINI-UNPRUNED",

    "RF":
        "RF-B01-100-GINI-SQRT-BOOTSTRAP",
}


FOLD_ORDER = [
    "FOLD_Q2_2018",
    "FOLD_Q3_2018",
    "FOLD_Q4_2018",
]


for model_key in MODEL_KEYS:
    model_fold_records = [
        record
        for record
        in w_short_fold_results
        if (
            record[
                "model_key"
            ]
            == model_key
        )
    ]

    assert len(
        model_fold_records
    ) == 3

    assert {
        record[
            "fold_id"
        ]
        for record
        in model_fold_records
    } == set(
        FOLD_ORDER
    )

    assert all(
        record[
            "model_config_id"
        ]
        ==
        EXPECTED_CONFIGS[
            model_key
        ]
        for record
        in model_fold_records
    )

    assert all(
        record[
            "imbalance_strategy"
        ]
        == "NONE"
        for record
        in model_fold_records
    )

    assert all(
        record[
            "threshold_policy"
        ]
        ==
        "DEFAULT_MODEL_DECISION_RULE"
        for record
        in model_fold_records
    )

    assert all(
        record[
            "random_state"
        ]
        == 42
        for record
        in model_fold_records
    )


print(
    "W_SHORT fold results:",
    len(
        w_short_fold_results
    ),
)

print(
    "W_SHORT aggregate records:",
    len(
        w_short_aggregates
    ),
)

print(
    "\nM7.4 W_SHORT SCOPE GATE: PASS"
)

```

    W_SHORT fold results: 9
    W_SHORT aggregate records: 3
    
    M7.4 W_SHORT SCOPE GATE: PASS


## 6. Same-window / same-fold comparability audit


```python

records_by_model_fold = {
    (
        record[
            "model_key"
        ],
        record[
            "fold_id"
        ],
    ):
        record
    for record
    in w_short_fold_results
}


comparability_checks = []


for fold_id in FOLD_ORDER:
    fold_records = [
        records_by_model_fold[
            (
                model_key,
                fold_id,
            )
        ]
        for model_key
        in MODEL_KEYS
    ]

    shared_fields = [
        "training_window_id",
        "fold_id",
        "train_start",
        "train_end_exclusive",
        "validation_start",
        "validation_end_exclusive",
        "train_rows",
        "validation_rows",
        "train_fraud_rows",
        "validation_fraud_rows",
        "feature_version",
        "preprocessing_version",
        "imbalance_strategy",
        "random_state",
        "threshold_policy",
    ]

    field_results = {}

    for field in shared_fields:
        values = [
            record[
                field
            ]
            for record
            in fold_records
        ]

        field_results[
            field
        ] = (
            values[
                0
            ]
            == values[
                1
            ]
            == values[
                2
            ]
        )

    assert all(
        field_results.values()
    )

    assert (
        fold_records[
            0
        ][
            "model_config_id"
        ]
        !=
        fold_records[
            1
        ][
            "model_config_id"
        ]
    )

    assert (
        fold_records[
            1
        ][
            "model_config_id"
        ]
        !=
        fold_records[
            2
        ][
            "model_config_id"
        ]
    )

    comparability_checks.append(
        {
            "fold_id":
                fold_id,

            "shared_field_count":
                len(
                    shared_fields
                ),

            "all_shared_fields_match":
                True,

            "active_variable":
                "model_family / model_config",
        }
    )


assert len(
    comparability_checks
) == 3


for check in comparability_checks:
    print(
        check[
            "fold_id"
        ],
        "→ PASS | shared fields:",
        check[
            "shared_field_count"
        ],
    )


print(
    "\nM7.4 SAME-WINDOW COMPARABILITY GATE: PASS"
)

```

    FOLD_Q2_2018 → PASS | shared fields: 15
    FOLD_Q3_2018 → PASS | shared fields: 15
    FOLD_Q4_2018 → PASS | shared fields: 15
    
    M7.4 SAME-WINDOW COMPARABILITY GATE: PASS


## 7. Reconstruct and verify aggregate records


```python

aggregate_by_model = {
    record[
        "model_key"
    ]:
        record
    for record
    in w_short_aggregates
}


assert set(
    aggregate_by_model
) == set(
    MODEL_KEYS
)


def reconstruct_model_aggregate(
    model_key,
):
    records = [
        records_by_model_fold[
            (
                model_key,
                fold_id,
            )
        ]
        for fold_id
        in FOLD_ORDER
    ]

    f1_values = np.array(
        [
            record[
                "F1_fraud"
            ]
            for record
            in records
        ],
        dtype=np.float64,
    )

    recall_values = np.array(
        [
            record[
                "Recall_fraud"
            ]
            for record
            in records
        ],
        dtype=np.float64,
    )

    precision_values = np.array(
        [
            record[
                "Precision_fraud"
            ]
            for record
            in records
        ],
        dtype=np.float64,
    )

    return {
        "mean_F1":
            float(
                f1_values.mean()
            ),

        "std_F1":
            float(
                f1_values.std(
                    ddof=0
                )
            ),

        "mean_Recall":
            float(
                recall_values.mean()
            ),

        "mean_Precision":
            float(
                precision_values.mean()
            ),

        "TP":
            int(
                sum(
                    record[
                        "TP"
                    ]
                    for record
                    in records
                )
            ),

        "FP":
            int(
                sum(
                    record[
                        "FP"
                    ]
                    for record
                    in records
                )
            ),

        "FN":
            int(
                sum(
                    record[
                        "FN"
                    ]
                    for record
                    in records
                )
            ),

        "TN":
            int(
                sum(
                    record[
                        "TN"
                    ]
                    for record
                    in records
                )
            ),

        "predicted_positive_count":
            int(
                sum(
                    record[
                        "predicted_positive_count"
                    ]
                    for record
                    in records
                )
            ),

        "total_fit_seconds":
            float(
                sum(
                    record[
                        "fit_seconds"
                    ]
                    for record
                    in records
                )
            ),

        "total_prediction_seconds":
            float(
                sum(
                    record[
                        "prediction_seconds"
                    ]
                    for record
                    in records
                )
            ),
    }


reconstructed_aggregates = {
    model_key:
        reconstruct_model_aggregate(
            model_key
        )
    for model_key
    in MODEL_KEYS
}


for model_key in MODEL_KEYS:
    official = (
        aggregate_by_model[
            model_key
        ]
    )

    rebuilt = (
        reconstructed_aggregates[
            model_key
        ]
    )

    np.testing.assert_allclose(
        rebuilt[
            "mean_F1"
        ],
        official[
            "mean_F1"
        ],
        rtol=0.0,
        atol=1e-15,
    )

    np.testing.assert_allclose(
        rebuilt[
            "std_F1"
        ],
        official[
            "std_F1"
        ],
        rtol=0.0,
        atol=1e-15,
    )

    np.testing.assert_allclose(
        rebuilt[
            "mean_Recall"
        ],
        official[
            "mean_Recall"
        ],
        rtol=0.0,
        atol=1e-15,
    )

    np.testing.assert_allclose(
        rebuilt[
            "mean_Precision"
        ],
        official[
            "mean_Precision"
        ],
        rtol=0.0,
        atol=1e-15,
    )

    pooled = (
        official[
            "pooled_OOF_metrics"
        ]
    )

    assert (
        rebuilt[
            "TP"
        ]
        == pooled[
            "TP"
        ]
    )

    assert (
        rebuilt[
            "FP"
        ]
        == pooled[
            "FP"
        ]
    )

    assert (
        rebuilt[
            "FN"
        ]
        == pooled[
            "FN"
        ]
    )

    assert (
        rebuilt[
            "TN"
        ]
        == pooled[
            "TN"
        ]
    )

    assert (
        rebuilt[
            "predicted_positive_count"
        ]
        == pooled[
            "predicted_positive_count"
        ]
    )


print(
    "Aggregate reconstruction:"
)

for model_key in MODEL_KEYS:
    record = (
        reconstructed_aggregates[
            model_key
        ]
    )

    print(
        model_key,
        "| mean F1:",
        round(
            record[
                "mean_F1"
            ],
            6,
        ),
        "| std F1:",
        round(
            record[
                "std_F1"
            ],
            6,
        ),
        "| mean Recall:",
        round(
            record[
                "mean_Recall"
            ],
            6,
        ),
        "| mean Precision:",
        round(
            record[
                "mean_Precision"
            ],
            6,
        ),
    )


print(
    "\nM7.4 AGGREGATE RECONSTRUCTION GATE: PASS"
)

```

    Aggregate reconstruction:
    LR | mean F1: 0.495102 | std F1: 0.038315 | mean Recall: 0.385526 | mean Precision: 0.698296
    DT | mean F1: 0.487392 | std F1: 0.013631 | mean Recall: 0.471101 | mean Precision: 0.515111
    RF | mean F1: 0.517612 | std F1: 0.029168 | mean Recall: 0.414125 | mean Precision: 0.694881
    
    M7.4 AGGREGATE RECONSTRUCTION GATE: PASS


## 8. Fold-wise F1 robustness


```python

fold_f1_records = []


for fold_id in FOLD_ORDER:
    values = {
        model_key:
            records_by_model_fold[
                (
                    model_key,
                    fold_id,
                )
            ][
                "F1_fraud"
            ]
        for model_key
        in MODEL_KEYS
    }

    leader = max(
        values,
        key=values.get,
    )

    ordered = sorted(
        values.items(),
        key=lambda item:
            item[
                1
            ],
        reverse=True,
    )

    fold_f1_records.append(
        {
            "fold_id":
                fold_id,

            "values":
                values,

            "leader":
                leader,

            "ordered":
                ordered,
        }
    )


f1_lead_counts = Counter(
    record[
        "leader"
    ]
    for record
    in fold_f1_records
)


for record in fold_f1_records:
    print(
        record[
            "fold_id"
        ],
        "→",
        record[
            "ordered"
        ],
    )


print(
    "\nF1 fold lead counts:"
)

print(
    dict(
        f1_lead_counts
    )
)


print(
    "\nM7.4 FOLD-WISE F1 GATE: PASS"
)

```

    FOLD_Q2_2018 → [('RF', 0.4901758014477766), ('LR', 0.47629083245521603), ('DT', 0.4683648315529992)]
    FOLD_Q3_2018 → [('RF', 0.5580057526366251), ('LR', 0.5485148514851486), ('DT', 0.4942263279445728)]
    FOLD_Q4_2018 → [('RF', 0.5046554934823091), ('DT', 0.4995850622406639), ('LR', 0.4605009633911368)]
    
    F1 fold lead counts:
    {'RF': 3}
    
    M7.4 FOLD-WISE F1 GATE: PASS


## 9. Fold-wise Recall robustness


```python

fold_recall_records = []


for fold_id in FOLD_ORDER:
    values = {
        model_key:
            records_by_model_fold[
                (
                    model_key,
                    fold_id,
                )
            ][
                "Recall_fraud"
            ]
        for model_key
        in MODEL_KEYS
    }

    leader = max(
        values,
        key=values.get,
    )

    fold_recall_records.append(
        {
            "fold_id":
                fold_id,

            "values":
                values,

            "leader":
                leader,
        }
    )


recall_lead_counts = Counter(
    record[
        "leader"
    ]
    for record
    in fold_recall_records
)


for record in fold_recall_records:
    print(
        record[
            "fold_id"
        ],
        "→",
        sorted(
            record[
                "values"
            ].items(),
            key=lambda item:
                item[
                    1
                ],
            reverse=True,
        ),
    )


print(
    "\nRecall fold lead counts:"
)

print(
    dict(
        recall_lead_counts
    )
)


print(
    "\nM7.4 FOLD-WISE RECALL GATE: PASS"
)

```

    FOLD_Q2_2018 → [('DT', 0.4830508474576271), ('RF', 0.4016949152542373), ('LR', 0.38305084745762713)]
    FOLD_Q3_2018 → [('DT', 0.5063091482649842), ('RF', 0.4589905362776025), ('LR', 0.43690851735015773)]
    FOLD_Q4_2018 → [('DT', 0.423943661971831), ('RF', 0.3816901408450704), ('LR', 0.33661971830985915)]
    
    Recall fold lead counts:
    {'DT': 3}
    
    M7.4 FOLD-WISE RECALL GATE: PASS


## 10. Fold-wise Precision robustness


```python

fold_precision_records = []


for fold_id in FOLD_ORDER:
    values = {
        model_key:
            records_by_model_fold[
                (
                    model_key,
                    fold_id,
                )
            ][
                "Precision_fraud"
            ]
        for model_key
        in MODEL_KEYS
    }

    leader = max(
        values,
        key=values.get,
    )

    fold_precision_records.append(
        {
            "fold_id":
                fold_id,

            "values":
                values,

            "leader":
                leader,
        }
    )


precision_lead_counts = Counter(
    record[
        "leader"
    ]
    for record
    in fold_precision_records
)


for record in fold_precision_records:
    print(
        record[
            "fold_id"
        ],
        "→",
        sorted(
            record[
                "values"
            ].items(),
            key=lambda item:
                item[
                    1
                ],
            reverse=True,
        ),
    )


print(
    "\nPrecision fold lead counts:"
)

print(
    dict(
        precision_lead_counts
    )
)


print(
    "\nM7.4 FOLD-WISE PRECISION GATE: PASS"
)

```

    FOLD_Q2_2018 → [('LR', 0.6295264623955432), ('RF', 0.6286472148541115), ('DT', 0.45454545454545453)]
    FOLD_Q3_2018 → [('LR', 0.7367021276595744), ('RF', 0.7114914425427873), ('DT', 0.48270676691729325)]
    FOLD_Q4_2018 → [('RF', 0.7445054945054945), ('LR', 0.7286585365853658), ('DT', 0.6080808080808081)]
    
    Precision fold lead counts:
    {'LR': 2, 'RF': 1}
    
    M7.4 FOLD-WISE PRECISION GATE: PASS


## 11. Pairwise F1 deltas across folds


```python

PAIRWISE_MODEL_COMPARISONS = [
    (
        "RF",
        "LR",
    ),
    (
        "RF",
        "DT",
    ),
    (
        "LR",
        "DT",
    ),
]


pairwise_f1_evidence = []


for left, right in (
    PAIRWISE_MODEL_COMPARISONS
):
    fold_deltas = []

    left_leads = 0
    right_leads = 0

    for fold_id in FOLD_ORDER:
        left_f1 = (
            records_by_model_fold[
                (
                    left,
                    fold_id,
                )
            ][
                "F1_fraud"
            ]
        )

        right_f1 = (
            records_by_model_fold[
                (
                    right,
                    fold_id,
                )
            ][
                "F1_fraud"
            ]
        )

        delta = (
            left_f1
            - right_f1
        )

        if delta > 0:
            left_leads += 1

        elif delta < 0:
            right_leads += 1

        fold_deltas.append(
            {
                "fold_id":
                    fold_id,

                "delta_F1":
                    float(
                        delta
                    ),
            }
        )

    mean_delta = (
        reconstructed_aggregates[
            left
        ][
            "mean_F1"
        ]
        -
        reconstructed_aggregates[
            right
        ][
            "mean_F1"
        ]
    )

    pairwise_f1_evidence.append(
        {
            "left":
                left,

            "right":
                right,

            "delta_convention":
                f"{left} - {right}",

            "fold_deltas":
                fold_deltas,

            "left_fold_leads":
                left_leads,

            "right_fold_leads":
                right_leads,

            "delta_mean_F1":
                float(
                    mean_delta
                ),
        }
    )


for comparison in pairwise_f1_evidence:
    print(
        "\n",
        comparison[
            "delta_convention"
        ],
    )

    print(
        "Fold leads:",
        comparison[
            "left_fold_leads"
        ],
        "/",
        comparison[
            "right_fold_leads"
        ],
    )

    print(
        "Δ mean F1:",
        round(
            comparison[
                "delta_mean_F1"
            ],
            6,
        ),
    )


print(
    "\nM7.4 PAIRWISE F1 ROBUSTNESS GATE: PASS"
)

```

    
     RF - LR
    Fold leads: 3 / 0
    Δ mean F1: 0.02251
    
     RF - DT
    Fold leads: 3 / 0
    Δ mean F1: 0.03022
    
     LR - DT
    Fold leads: 2 / 1
    Δ mean F1: 0.00771
    
    M7.4 PAIRWISE F1 ROBUSTNESS GATE: PASS


## 12. Raw error burden and operational output


```python

operational_profiles = {}


for model_key in MODEL_KEYS:
    aggregate = (
        reconstructed_aggregates[
            model_key
        ]
    )

    total_rows = (
        aggregate[
            "TP"
        ]
        +
        aggregate[
            "FP"
        ]
        +
        aggregate[
            "FN"
        ]
        +
        aggregate[
            "TN"
        ]
    )

    operational_profiles[
        model_key
    ] = {
        "TP":
            aggregate[
                "TP"
            ],

        "FP":
            aggregate[
                "FP"
            ],

        "FN":
            aggregate[
                "FN"
            ],

        "TN":
            aggregate[
                "TN"
            ],

        "predicted_positive_count":
            aggregate[
                "predicted_positive_count"
            ],

        "predicted_positive_rate":
            float(
                aggregate[
                    "predicted_positive_count"
                ]
                /
                total_rows
            ),

        "total_fit_seconds":
            aggregate[
                "total_fit_seconds"
            ],

        "total_prediction_seconds":
            aggregate[
                "total_prediction_seconds"
            ],
    }


for model_key in MODEL_KEYS:
    profile = (
        operational_profiles[
            model_key
        ]
    )

    print(
        "\n",
        MODEL_LABELS[
            model_key
        ],
    )

    print(
        "TP / FP / FN / TN:",
        profile[
            "TP"
        ],
        "/",
        profile[
            "FP"
        ],
        "/",
        profile[
            "FN"
        ],
        "/",
        profile[
            "TN"
        ],
    )

    print(
        "Alerts:",
        profile[
            "predicted_positive_count"
        ],
        "| rate:",
        profile[
            "predicted_positive_rate"
        ],
    )

    print(
        "Fit seconds:",
        round(
            profile[
                "total_fit_seconds"
            ],
            3,
        ),
    )


print(
    "\nM7.4 OPERATIONAL EVIDENCE GATE: PASS"
)

```

    
     Logistic Regression
    TP / FP / FN / TN: 742 / 321 / 1192 / 1295455
    Alerts: 1063 | rate: 0.0008191352459332209
    Fit seconds: 1.24
    
     Decision Tree
    TP / FP / FN / TN: 907 / 880 / 1027 / 1294896
    Alerts: 1787 | rate: 0.0013770410954681708
    Fit seconds: 2.668
    
     Random Forest
    TP / FP / FN / TN: 799 / 351 / 1135 / 1295425
    Alerts: 1150 | rate: 0.0008861764184602107
    Fit seconds: 24.115
    
    M7.4 OPERATIONAL EVIDENCE GATE: PASS


## 13. Frozen M6 supporting evidence

M7.4 phải đọc M6 error findings nhưng không chạy lại external VALIDATION.

Frozen W_SHORT M6 evidence:

```text
Logistic Regression

F1:
0.337475

Recall:
0.245247

Precision:
0.540881

TP / FP / FN:
258 / 219 / 794

alerts:
477

baseline fit seconds:
0.8222
```

```text
Decision Tree

F1:
0.327056

Recall:
0.313688

Precision:
0.341615

TP / FP / FN:
330 / 636 / 722

alerts:
966

baseline fit seconds:
6.6618
```

```text
Random Forest

F1:
0.366467

Recall:
0.290875

Precision:
0.495146

TP / FP / FN:
306 / 312 / 746

alerts:
618

baseline fit seconds:
40.6980
```

M6 descriptive roles:

```text
highest F1:
RF

highest Recall / lowest FN:
DT

highest Precision:
LR

lowest FP / alerts / fit cost:
LR
```

M6 W_SHORT overlap:

```text
actual fraud:
1,052

all-three miss:
618

caught by at least one:
434

model-specific only catches:
LR 31
DT 68
RF 28
```

```text
model-specific only false positives:
LR 58
DT 388
RF 68
```

Important:

Đây là **frozen supporting evidence** từ M6.

M7.4 không tải lại external-validation labels và không tạo new validation scoring.


```python

M6_FROZEN_W_SHORT_EVIDENCE = {
    "LR": {
        "F1_fraud":
            0.33747547416612167,

        "Recall_fraud":
            0.24524714828897337,

        "Precision_fraud":
            0.5408805031446541,

        "TP":
            258,

        "FP":
            219,

        "FN":
            794,

        "alerts":
            477,

        "fit_seconds":
            0.8222020840039477,

        "unique_only_catches":
            31,

        "unique_only_false_positives":
            58,
    },

    "DT": {
        "F1_fraud":
            0.3270564915758176,

        "Recall_fraud":
            0.31368821292775667,

        "Precision_fraud":
            0.3416149068322981,

        "TP":
            330,

        "FP":
            636,

        "FN":
            722,

        "alerts":
            966,

        "fit_seconds":
            6.661822540976573,

        "unique_only_catches":
            68,

        "unique_only_false_positives":
            388,
    },

    "RF": {
        "F1_fraud":
            0.3664670658682635,

        "Recall_fraud":
            0.2908745247148289,

        "Precision_fraud":
            0.49514563106796117,

        "TP":
            306,

        "FP":
            312,

        "FN":
            746,

        "alerts":
            618,

        "fit_seconds":
            40.69802879198687,

        "unique_only_catches":
            28,

        "unique_only_false_positives":
            68,
    },
}


M6_W_SHORT_OVERLAP = {
    "actual_fraud":
        1_052,

    "all_three_miss":
        618,

    "caught_by_at_least_one":
        434,

    "model_specific_only_catches": {
        "LR":
            31,

        "DT":
            68,

        "RF":
            28,
    },

    "model_specific_only_false_positives": {
        "LR":
            58,

        "DT":
            388,

        "RF":
            68,
    },
}


assert set(
    M6_FROZEN_W_SHORT_EVIDENCE
) == set(
    MODEL_KEYS
)


print(
    "M6 frozen evidence candidates:",
    len(
        M6_FROZEN_W_SHORT_EVIDENCE
    ),
)

print(
    "M6 all-three miss:",
    M6_W_SHORT_OVERLAP[
        "all_three_miss"
    ],
)

print(
    "\nM7.4 M6 SUPPORTING-EVIDENCE GATE: PASS"
)

```

    M6 frozen evidence candidates: 3
    M6 all-three miss: 618
    
    M7.4 M6 SUPPORTING-EVIDENCE GATE: PASS


## 14. Cross-stage role consistency


```python

# M7.3 temporal-CV extrema.
m7_mean_f1_leader = max(
    MODEL_KEYS,
    key=lambda key:
        reconstructed_aggregates[
            key
        ][
            "mean_F1"
        ],
)

m7_mean_recall_leader = max(
    MODEL_KEYS,
    key=lambda key:
        reconstructed_aggregates[
            key
        ][
            "mean_Recall"
        ],
)

m7_mean_precision_leader = max(
    MODEL_KEYS,
    key=lambda key:
        reconstructed_aggregates[
            key
        ][
            "mean_Precision"
        ],
)

m7_lowest_fn = min(
    MODEL_KEYS,
    key=lambda key:
        reconstructed_aggregates[
            key
        ][
            "FN"
        ],
)

m7_lowest_fp = min(
    MODEL_KEYS,
    key=lambda key:
        reconstructed_aggregates[
            key
        ][
            "FP"
        ],
)

m7_lowest_alerts = min(
    MODEL_KEYS,
    key=lambda key:
        reconstructed_aggregates[
            key
        ][
            "predicted_positive_count"
        ],
)

m7_lowest_fit_seconds = min(
    MODEL_KEYS,
    key=lambda key:
        reconstructed_aggregates[
            key
        ][
            "total_fit_seconds"
        ],
)


# Frozen M6 W_SHORT extrema.
m6_f1_leader = max(
    MODEL_KEYS,
    key=lambda key:
        M6_FROZEN_W_SHORT_EVIDENCE[
            key
        ][
            "F1_fraud"
        ],
)

m6_recall_leader = max(
    MODEL_KEYS,
    key=lambda key:
        M6_FROZEN_W_SHORT_EVIDENCE[
            key
        ][
            "Recall_fraud"
        ],
)

m6_precision_leader = max(
    MODEL_KEYS,
    key=lambda key:
        M6_FROZEN_W_SHORT_EVIDENCE[
            key
        ][
            "Precision_fraud"
        ],
)

m6_lowest_fn = min(
    MODEL_KEYS,
    key=lambda key:
        M6_FROZEN_W_SHORT_EVIDENCE[
            key
        ][
            "FN"
        ],
)

m6_lowest_fp = min(
    MODEL_KEYS,
    key=lambda key:
        M6_FROZEN_W_SHORT_EVIDENCE[
            key
        ][
            "FP"
        ],
)

m6_lowest_alerts = min(
    MODEL_KEYS,
    key=lambda key:
        M6_FROZEN_W_SHORT_EVIDENCE[
            key
        ][
            "alerts"
        ],
)

m6_lowest_fit_seconds = min(
    MODEL_KEYS,
    key=lambda key:
        M6_FROZEN_W_SHORT_EVIDENCE[
            key
        ][
            "fit_seconds"
        ],
)


role_consistency = {
    "F1_leader": {
        "M6":
            m6_f1_leader,

        "M7_temporal_CV":
            m7_mean_f1_leader,

        "consistent":
            (
                m6_f1_leader
                ==
                m7_mean_f1_leader
            ),
    },

    "Recall_leader": {
        "M6":
            m6_recall_leader,

        "M7_temporal_CV":
            m7_mean_recall_leader,

        "consistent":
            (
                m6_recall_leader
                ==
                m7_mean_recall_leader
            ),
    },

    "Precision_leader": {
        "M6":
            m6_precision_leader,

        "M7_temporal_CV":
            m7_mean_precision_leader,

        "consistent":
            (
                m6_precision_leader
                ==
                m7_mean_precision_leader
            ),
    },

    "lowest_FN": {
        "M6":
            m6_lowest_fn,

        "M7_temporal_CV":
            m7_lowest_fn,

        "consistent":
            (
                m6_lowest_fn
                ==
                m7_lowest_fn
            ),
    },

    "lowest_FP": {
        "M6":
            m6_lowest_fp,

        "M7_temporal_CV":
            m7_lowest_fp,

        "consistent":
            (
                m6_lowest_fp
                ==
                m7_lowest_fp
            ),
    },

    "lowest_alerts": {
        "M6":
            m6_lowest_alerts,

        "M7_temporal_CV":
            m7_lowest_alerts,

        "consistent":
            (
                m6_lowest_alerts
                ==
                m7_lowest_alerts
            ),
    },

    "lowest_fit_seconds": {
        "M6":
            m6_lowest_fit_seconds,

        "M7_temporal_CV":
            m7_lowest_fit_seconds,

        "consistent":
            (
                m6_lowest_fit_seconds
                ==
                m7_lowest_fit_seconds
            ),
    },
}


for role, evidence in (
    role_consistency.items()
):
    print(
        role,
        "→ M6:",
        evidence[
            "M6"
        ],
        "| M7 temporal CV:",
        evidence[
            "M7_temporal_CV"
        ],
        "| consistent:",
        evidence[
            "consistent"
        ],
    )


print(
    "\nM7.4 CROSS-STAGE ROLE CONSISTENCY GATE: PASS"
)

```

    F1_leader → M6: RF | M7 temporal CV: RF | consistent: True
    Recall_leader → M6: DT | M7 temporal CV: DT | consistent: True
    Precision_leader → M6: LR | M7 temporal CV: LR | consistent: True
    lowest_FN → M6: DT | M7 temporal CV: DT | consistent: True
    lowest_FP → M6: LR | M7 temporal CV: LR | consistent: True
    lowest_alerts → M6: LR | M7 temporal CV: LR | consistent: True
    lowest_fit_seconds → M6: LR | M7 temporal CV: LR | consistent: True
    
    M7.4 CROSS-STAGE ROLE CONSISTENCY GATE: PASS


## 15. Candidate evidence profiles


```python

candidate_profiles = {}


for model_key in MODEL_KEYS:
    aggregate = (
        reconstructed_aggregates[
            model_key
        ]
    )

    candidate_profiles[
        model_key
    ] = {
        "model_family":
            MODEL_LABELS[
                model_key
            ],

        "model_config_id":
            EXPECTED_CONFIGS[
                model_key
            ],

        "mean_F1":
            aggregate[
                "mean_F1"
            ],

        "std_F1":
            aggregate[
                "std_F1"
            ],

        "mean_Recall":
            aggregate[
                "mean_Recall"
            ],

        "mean_Precision":
            aggregate[
                "mean_Precision"
            ],

        "F1_fold_leads":
            int(
                f1_lead_counts[
                    model_key
                ]
            ),

        "Recall_fold_leads":
            int(
                recall_lead_counts[
                    model_key
                ]
            ),

        "Precision_fold_leads":
            int(
                precision_lead_counts[
                    model_key
                ]
            ),

        "TP":
            aggregate[
                "TP"
            ],

        "FP":
            aggregate[
                "FP"
            ],

        "FN":
            aggregate[
                "FN"
            ],

        "alerts":
            aggregate[
                "predicted_positive_count"
            ],

        "total_fit_seconds":
            aggregate[
                "total_fit_seconds"
            ],

        "M6_F1":
            M6_FROZEN_W_SHORT_EVIDENCE[
                model_key
            ][
                "F1_fraud"
            ],

        "M6_unique_only_catches":
            M6_FROZEN_W_SHORT_EVIDENCE[
                model_key
            ][
                "unique_only_catches"
            ],

        "M6_unique_only_false_positives":
            M6_FROZEN_W_SHORT_EVIDENCE[
                model_key
            ][
                "unique_only_false_positives"
            ],
    }


for model_key in MODEL_KEYS:
    profile = (
        candidate_profiles[
            model_key
        ]
    )

    print(
        "\n",
        model_key,
        profile[
            "model_family"
        ],
    )

    for key in [
        "mean_F1",
        "std_F1",
        "mean_Recall",
        "mean_Precision",
        "F1_fold_leads",
        "Recall_fold_leads",
        "Precision_fold_leads",
        "TP",
        "FP",
        "FN",
        "alerts",
        "total_fit_seconds",
        "M6_unique_only_catches",
        "M6_unique_only_false_positives",
    ]:
        print(
            " ",
            key,
            "→",
            profile[
                key
            ],
        )


print(
    "\nM7.4 CANDIDATE PROFILE GATE: PASS"
)

```

    
     LR Logistic Regression
      mean_F1 → 0.4951022157771671
      std_F1 → 0.03831459453193906
      mean_Recall → 0.3855263610392147
      mean_Precision → 0.6982957088801611
      F1_fold_leads → 0
      Recall_fold_leads → 0
      Precision_fold_leads → 2
      TP → 742
      FP → 321
      FN → 1192
      alerts → 1063
      total_fit_seconds → 1.2401174160186201
      M6_unique_only_catches → 31
      M6_unique_only_false_positives → 58
    
     DT Decision Tree
      mean_F1 → 0.4873920739127453
      std_F1 → 0.013630993413942437
      mean_Recall → 0.4711012192314808
      mean_Precision → 0.515111009847852
      F1_fold_leads → 0
      Recall_fold_leads → 3
      Precision_fold_leads → 0
      TP → 907
      FP → 880
      FN → 1027
      alerts → 1787
      total_fit_seconds → 2.667644833913073
      M6_unique_only_catches → 68
      M6_unique_only_false_positives → 388
    
     RF Random Forest
      mean_F1 → 0.5176123491889036
      std_F1 → 0.02916774076408759
      mean_Recall → 0.4141251974589701
      mean_Precision → 0.6948813839674645
      F1_fold_leads → 3
      Recall_fold_leads → 0
      Precision_fold_leads → 1
      TP → 799
      FP → 351
      FN → 1135
      alerts → 1150
      total_fit_seconds → 24.114745874947403
      M6_unique_only_catches → 28
      M6_unique_only_false_positives → 68
    
    M7.4 CANDIDATE PROFILE GATE: PASS


## 16. Descriptive role map — không phải shortlist decision

Các role dưới đây chỉ mô tả evidence extrema.

Chúng không phải score/rank tổng hợp.


```python

descriptive_roles = {
    "primary_F1_leader":
        m7_mean_f1_leader,

    "F1_fold_leader":
        max(
            MODEL_KEYS,
            key=lambda key:
                f1_lead_counts[
                    key
                ],
        ),

    "Recall_leader":
        m7_mean_recall_leader,

    "lowest_FN":
        m7_lowest_fn,

    "Precision_leader":
        m7_mean_precision_leader,

    "lowest_FP":
        m7_lowest_fp,

    "lowest_alert_burden":
        m7_lowest_alerts,

    "lowest_fit_cost":
        m7_lowest_fit_seconds,

    "lowest_std_F1":
        min(
            MODEL_KEYS,
            key=lambda key:
                reconstructed_aggregates[
                    key
                ][
                    "std_F1"
                ],
        ),

    "most_M6_unique_only_catches":
        max(
            MODEL_KEYS,
            key=lambda key:
                M6_FROZEN_W_SHORT_EVIDENCE[
                    key
                ][
                    "unique_only_catches"
                ],
        ),
}


for role, model_key in (
    descriptive_roles.items()
):
    print(
        role,
        "→",
        model_key,
        MODEL_LABELS[
            model_key
        ],
    )


print(
    "\nM7.4 DESCRIPTIVE ROLE MAP GATE: PASS"
)

```

    primary_F1_leader → RF Random Forest
    F1_fold_leader → RF Random Forest
    Recall_leader → DT Decision Tree
    lowest_FN → DT Decision Tree
    Precision_leader → LR Logistic Regression
    lowest_FP → LR Logistic Regression
    lowest_alert_burden → LR Logistic Regression
    lowest_fit_cost → LR Logistic Regression
    lowest_std_F1 → DT Decision Tree
    most_M6_unique_only_catches → DT Decision Tree
    
    M7.4 DESCRIPTIVE ROLE MAP GATE: PASS


## 17. Why shortlist cannot be auto-generated from one metric

M7.3 W_SHORT evidence is expected to reveal different roles:

```text
RF:
primary F1 strength

DT:
Recall / FN / stability strength

LR:
Precision / FP / alert / compute strength
```

M6 also showed model-specific fraud catches.

Therefore:

- `highest mean F1` alone is insufficient;
- `highest Recall` alone is insufficient;
- `lowest runtime` alone is insufficient;
- a weighted composite score is not authorized.

Notebook will persist all evidence and leave shortlist OPEN until runtime review.


```python

MODEL_SHORTLIST_DECISION = (
    "OPEN — REQUIRES M7.4 RUNTIME REVIEW"
)

MODEL_FAMILY_WINNER = "OPEN"

FINAL_MODEL = "OPEN"

FINAL_IMBALANCE_STRATEGY = "OPEN"

FINAL_THRESHOLD = "OPEN"


assert (
    MODEL_SHORTLIST_DECISION
    != ["LR"]
)

assert (
    MODEL_SHORTLIST_DECISION
    != ["DT"]
)

assert (
    MODEL_SHORTLIST_DECISION
    != ["RF"]
)

assert (
    MODEL_FAMILY_WINNER
    == "OPEN"
)

assert (
    FINAL_MODEL
    == "OPEN"
)


print(
    "Model shortlist:"
)

print(
    MODEL_SHORTLIST_DECISION
)

print(
    "\nModel-family winner:"
)

print(
    MODEL_FAMILY_WINNER
)

print(
    "\nM7.4 NO-AUTO-SHORTLIST GATE: PASS"
)

```

    Model shortlist:
    OPEN — REQUIRES M7.4 RUNTIME REVIEW
    
    Model-family winner:
    OPEN
    
    M7.4 NO-AUTO-SHORTLIST GATE: PASS


## 18. Persist M7.4 evidence registry


```python

source_fingerprints = {
    "m7_03_result_sha256":
        sha256_file(
            M7_03_RESULT_PATH
        ),

    "m7_03_manifest_sha256":
        sha256_file(
            M7_03_MANIFEST_PATH
        ),

    "reviewed_m7_03_notebook_sha256":
        UPSTREAM_REVIEWED_M7_3_NOTEBOOK_SHA256,
}


result_payload = {
    "analysis_version":
        M7_04_ANALYSIS_VERSION,

    "upstream_reviewed_handoff": {
        "m7_03_status":
            UPSTREAM_REVIEWED_M7_3_STATUS,

        "training_window":
            UPSTREAM_REVIEWED_TRAINING_WINDOW,

        "reviewed_notebook_sha256":
            UPSTREAM_REVIEWED_M7_3_NOTEBOOK_SHA256,

        "raw_m7_03_cross_family_state":
            m7_03_result[
                "cross_family_robustness_state"
            ],
    },

    "source_fingerprints":
        source_fingerprints,

    "comparison_scope": {
        "training_window":
            "W_SHORT",

        "model_keys":
            MODEL_KEYS,

        "model_configs":
            EXPECTED_CONFIGS,

        "fold_ids":
            FOLD_ORDER,

        "imbalance_strategy":
            "NONE",

        "threshold_policy":
            "DEFAULT_MODEL_DECISION_RULE",

        "random_state":
            42,

        "active_variable":
            "model_family / model_config",
    },

    "comparability_checks":
        comparability_checks,

    "w_short_fold_results":
        w_short_fold_results,

    "reconstructed_aggregates":
        reconstructed_aggregates,

    "fold_f1_records":
        fold_f1_records,

    "fold_recall_records":
        fold_recall_records,

    "fold_precision_records":
        fold_precision_records,

    "pairwise_f1_evidence":
        pairwise_f1_evidence,

    "operational_profiles":
        operational_profiles,

    "m6_frozen_supporting_evidence":
        M6_FROZEN_W_SHORT_EVIDENCE,

    "m6_overlap_supporting_evidence":
        M6_W_SHORT_OVERLAP,

    "role_consistency":
        role_consistency,

    "candidate_profiles":
        candidate_profiles,

    "descriptive_roles":
        descriptive_roles,

    "selection_state": {
        "training_window":
            "W_SHORT — LOCKED FOR DOWNSTREAM M7",

        "model_shortlist":
            MODEL_SHORTLIST_DECISION,

        "model_family_winner":
            MODEL_FAMILY_WINNER,

        "final_model":
            FINAL_MODEL,

        "final_imbalance_strategy":
            FINAL_IMBALANCE_STRATEGY,

        "final_threshold":
            FINAL_THRESHOLD,

        "m7_04_runtime_review_required":
            True,
    },

    "new_external_validation_scoring_performed":
        False,

    "m6_frozen_validation_evidence_consulted":
        True,

    "retraining_performed":
        False,

    "hyperparameter_tuning_performed":
        False,

    "imbalance_intervention_performed":
        False,

    "threshold_optimization_performed":
        False,

    "final_test_accessed":
        False,
}


with open(
    M7_04_RESULT_PATH,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        result_payload,
        file,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )


manifest_payload = {
    "analysis_version":
        M7_04_ANALYSIS_VERSION,

    "training_window_scope":
        "W_SHORT",

    "candidate_family_count":
        3,

    "fold_count":
        3,

    "w_short_fold_run_count":
        len(
            w_short_fold_results
        ),

    "comparability_fold_count":
        len(
            comparability_checks
        ),

    "aggregate_reconstruction":
        "PASS",

    "foldwise_f1_analysis":
        "PASS",

    "foldwise_recall_analysis":
        "PASS",

    "foldwise_precision_analysis":
        "PASS",

    "operational_evidence":
        "PASS",

    "m6_supporting_evidence":
        "LOADED_AS_FROZEN_CONTEXT",

    "model_shortlist":
        MODEL_SHORTLIST_DECISION,

    "model_family_winner":
        "OPEN",

    "new_external_validation_scoring_performed":
        False,

    "retraining_performed":
        False,

    "hyperparameter_tuning_performed":
        False,

    "imbalance_intervention_performed":
        False,

    "threshold_optimization_performed":
        False,

    "final_test_accessed":
        False,

    "decision":
        "OPEN — REQUIRES M7.4 RUNTIME REVIEW",
}


with open(
    M7_04_MANIFEST_PATH,
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


assert M7_04_RESULT_PATH.exists()
assert M7_04_MANIFEST_PATH.exists()

assert (
    M7_04_RESULT_PATH.stat().st_size
    > 0
)

assert (
    M7_04_MANIFEST_PATH.stat().st_size
    > 0
)


print(
    "M7.4 result:"
)

print(
    M7_04_RESULT_PATH
)

print(
    "\nM7.4 manifest:"
)

print(
    M7_04_MANIFEST_PATH
)

print(
    "\nM7.4 PERSISTENCE GATE: PASS"
)

```

    M7.4 result:
    /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/processed/m7_04_model_family_robustness_shortlist/m7_04_model_family_robustness.json
    
    M7.4 manifest:
    /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/processed/m7_04_model_family_robustness_shortlist/m7_04_shortlist_manifest.json
    
    M7.4 PERSISTENCE GATE: PASS


## 19. Persistence round-trip


```python

with open(
    M7_04_RESULT_PATH,
    "r",
    encoding="utf-8",
) as file:
    result_roundtrip = json.load(
        file
    )


with open(
    M7_04_MANIFEST_PATH,
    "r",
    encoding="utf-8",
) as file:
    manifest_roundtrip = json.load(
        file
    )


assert (
    result_roundtrip[
        "analysis_version"
    ]
    == M7_04_ANALYSIS_VERSION
)

assert (
    result_roundtrip[
        "comparison_scope"
    ][
        "training_window"
    ]
    == "W_SHORT"
)

assert (
    len(
        result_roundtrip[
            "w_short_fold_results"
        ]
    )
    == 9
)

assert (
    len(
        result_roundtrip[
            "candidate_profiles"
        ]
    )
    == 3
)

assert (
    result_roundtrip[
        "selection_state"
    ][
        "model_shortlist"
    ]
    ==
    "OPEN — REQUIRES M7.4 RUNTIME REVIEW"
)

assert (
    result_roundtrip[
        "selection_state"
    ][
        "model_family_winner"
    ]
    == "OPEN"
)

assert (
    result_roundtrip[
        "new_external_validation_scoring_performed"
    ]
    is False
)

assert (
    result_roundtrip[
        "m6_frozen_validation_evidence_consulted"
    ]
    is True
)

assert (
    result_roundtrip[
        "final_test_accessed"
    ]
    is False
)


assert (
    manifest_roundtrip[
        "w_short_fold_run_count"
    ]
    == 9
)

assert (
    manifest_roundtrip[
        "candidate_family_count"
    ]
    == 3
)

assert (
    manifest_roundtrip[
        "model_family_winner"
    ]
    == "OPEN"
)

assert (
    manifest_roundtrip[
        "final_test_accessed"
    ]
    is False
)


print(
    "Result SHA256:"
)

print(
    sha256_file(
        M7_04_RESULT_PATH
    )
)

print(
    "\nManifest SHA256:"
)

print(
    sha256_file(
        M7_04_MANIFEST_PATH
    )
)

print(
    "\nM7.4 ROUND-TRIP GATE: PASS"
)

```

    Result SHA256:
    65f353209a55e7d754d6b5551b3d35f52bf91c8df12bb5cb53d5ed739cfacb58
    
    Manifest SHA256:
    3278f88fe29326f01caae0687efddd107d0ec0f362456069a39fd952dfed9762
    
    M7.4 ROUND-TRIP GATE: PASS


## 20. Selection-boundary / leakage gate


```python

assert (
    UPSTREAM_REVIEWED_TRAINING_WINDOW
    == "W_SHORT"
)

assert (
    result_payload[
        "new_external_validation_scoring_performed"
    ]
    is False
)

assert (
    result_payload[
        "retraining_performed"
    ]
    is False
)

assert (
    result_payload[
        "hyperparameter_tuning_performed"
    ]
    is False
)

assert (
    result_payload[
        "imbalance_intervention_performed"
    ]
    is False
)

assert (
    result_payload[
        "threshold_optimization_performed"
    ]
    is False
)

assert (
    result_payload[
        "selection_state"
    ][
        "model_family_winner"
    ]
    == "OPEN"
)

assert (
    result_payload[
        "selection_state"
    ][
        "final_model"
    ]
    == "OPEN"
)

assert (
    result_payload[
        "selection_state"
    ][
        "final_imbalance_strategy"
    ]
    == "OPEN"
)

assert (
    result_payload[
        "selection_state"
    ][
        "final_threshold"
    ]
    == "OPEN"
)

assert (
    result_payload[
        "final_test_accessed"
    ]
    is False
)


print(
    "Training-window scope:"
)

print(
    UPSTREAM_REVIEWED_TRAINING_WINDOW
)

print(
    "\nNew external VALIDATION scoring:"
)

print(
    result_payload[
        "new_external_validation_scoring_performed"
    ]
)

print(
    "\nRetraining:"
)

print(
    result_payload[
        "retraining_performed"
    ]
)

print(
    "\nFINAL TEST accessed:"
)

print(
    result_payload[
        "final_test_accessed"
    ]
)

print(
    "\nM7.4 SELECTION-BOUNDARY GATE: PASS"
)

```

    Training-window scope:
    W_SHORT
    
    New external VALIDATION scoring:
    False
    
    Retraining:
    False
    
    FINAL TEST accessed:
    False
    
    M7.4 SELECTION-BOUNDARY GATE: PASS


## 21. Overall technical gate


```python

m7_04_gates = {
    "G01_SOURCE_LOCATION":
        True,

    "G02_M7_3_REVIEWED_HANDOFF":
        True,

    "G03_W_SHORT_SCOPE":
        True,

    "G04_SAME_WINDOW_COMPARABILITY":
        True,

    "G05_AGGREGATE_RECONSTRUCTION":
        True,

    "G06_FOLDWISE_F1":
        True,

    "G07_FOLDWISE_RECALL":
        True,

    "G08_FOLDWISE_PRECISION":
        True,

    "G09_PAIRWISE_F1_ROBUSTNESS":
        True,

    "G10_OPERATIONAL_EVIDENCE":
        True,

    "G11_M6_SUPPORTING_EVIDENCE":
        True,

    "G12_CROSS_STAGE_ROLE_CONSISTENCY":
        True,

    "G13_CANDIDATE_PROFILES":
        True,

    "G14_DESCRIPTIVE_ROLE_MAP":
        True,

    "G15_NO_AUTO_SHORTLIST":
        True,

    "G16_PERSISTENCE":
        True,

    "G17_ROUND_TRIP":
        True,

    "G18_NO_NEW_EXTERNAL_VALIDATION_SCORING":
        True,

    "G19_NO_TUNING_IMBALANCE_THRESHOLD":
        True,

    "G20_FINAL_TEST_PROTECTION":
        True,
}


for gate_name, gate_value in (
    m7_04_gates.items()
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
    m7_04_gates
) == 20

assert all(
    m7_04_gates.values()
)


print(
    "\nM7.4 OVERALL TECHNICAL GATE: PASS"
)

```

    G01_SOURCE_LOCATION → PASS
    G02_M7_3_REVIEWED_HANDOFF → PASS
    G03_W_SHORT_SCOPE → PASS
    G04_SAME_WINDOW_COMPARABILITY → PASS
    G05_AGGREGATE_RECONSTRUCTION → PASS
    G06_FOLDWISE_F1 → PASS
    G07_FOLDWISE_RECALL → PASS
    G08_FOLDWISE_PRECISION → PASS
    G09_PAIRWISE_F1_ROBUSTNESS → PASS
    G10_OPERATIONAL_EVIDENCE → PASS
    G11_M6_SUPPORTING_EVIDENCE → PASS
    G12_CROSS_STAGE_ROLE_CONSISTENCY → PASS
    G13_CANDIDATE_PROFILES → PASS
    G14_DESCRIPTIVE_ROLE_MAP → PASS
    G15_NO_AUTO_SHORTLIST → PASS
    G16_PERSISTENCE → PASS
    G17_ROUND_TRIP → PASS
    G18_NO_NEW_EXTERNAL_VALIDATION_SCORING → PASS
    G19_NO_TUNING_IMBALANCE_THRESHOLD → PASS
    G20_FINAL_TEST_PROTECTION → PASS
    
    M7.4 OVERALL TECHNICAL GATE: PASS


# 22. Runtime review và Findings M7.4

## 22.1. Execution integrity

Observed:

```text
Code cells:
20 / 20

Execution count:
1 → 20 liên tục

Runtime errors:
0

stderr:
0
```

Interpretation:

Notebook đã chạy đầy đủ từ đầu đến cuối.

Không có partial execution hoặc runtime exception làm mất hiệu lực comparative evidence.

Status:

`VERIFIED`

---

## 22.2. M7.3 reviewed handoff

Observed:

```text
M7.3 completed runs:
18 / 18

M7.3 warnings:
0

Cross-family training-window evidence:
ROBUST W_SHORT PREFERENCE

Reviewed downstream window:
W_SHORT
```

M7.4 chỉ đọc đúng downstream training-window scope đã được review:

`W_SHORT`.

Runtime gate:

`M7.4 M7.3 REVIEWED HANDOFF GATE: PASS`

Status:

`VERIFIED`

---

## 22.3. W_SHORT-only model-family scope

Observed:

```text
W_SHORT fold results:
9

W_SHORT aggregate records:
3
```

Candidate families:

```text
LR
DT
RF
```

Temporal folds:

```text
Q2 2018
Q3 2018
Q4 2018
```

Không có mixed-window family comparison.

Runtime gate:

`M7.4 W_SHORT SCOPE GATE: PASS`

Status:

`VERIFIED`

---

## 22.4. Same-window comparability

Observed:

```text
Q2:
PASS — 15 shared fields

Q3:
PASS — 15 shared fields

Q4:
PASS — 15 shared fields
```

Trong từng fold, ba model dùng cùng:

```text
training window
train/validation population
feature version
preprocessing version
imbalance strategy
threshold policy
random state
temporal boundaries
```

Biến khác nhau:

`model family / model config`.

Runtime gate:

`M7.4 SAME-WINDOW COMPARABILITY GATE: PASS`

Status:

`VERIFIED`

---

# 23. Predictive / operational Findings

## M7.4-F01 — Random Forest có primary-F1 robustness mạnh nhất

Temporal-CV aggregate:

```text
Random Forest

mean F1:
0.517612

std F1:
0.029168

mean Recall:
0.414125

mean Precision:
0.694881
```

Fold-wise F1:

```text
Q2:
0.490176

Q3:
0.558006

Q4:
0.504655
```

F1 lead count:

`3 / 3 folds`

Pairwise F1 evidence:

```text
RF − LR

fold leads:
3 / 0

Δ mean F1:
+0.022510
```

```text
RF − DT

fold leads:
3 / 0

Δ mean F1:
+0.030220
```

Interpretation:

RF có primary-metric advantage nhất quán qua toàn bộ ba temporal folds.

Đây là robustness evidence mạnh nhất cho vai trò:

`PRIMARY F1 LEADER`.

Status:

`VERIFIED`

---

## M7.4-F02 — Decision Tree có Recall / FN advantage mạnh nhất

Temporal-CV aggregate:

```text
Decision Tree

mean F1:
0.487392

std F1:
0.013631

mean Recall:
0.471101

mean Precision:
0.515111
```

Recall fold leads:

`3 / 3`

Pooled temporal-CV errors:

```text
TP:
907

FN:
1,027
```

So với:

```text
RF FN:
1,135

LR FN:
1,192
```

DT còn có:

`lowest std F1 = 0.013631`

trong ba candidates.

Interpretation:

DT không dẫn primary F1, nhưng có evidence rõ về:

```text
highest Recall
lowest FN
lowest F1 variation
```

Đây là compensating role quan trọng trong fraud screening, nơi bỏ sót fraud là một failure mode chính.

Status:

`VERIFIED`

---

## M7.4-F03 — Logistic Regression có Precision / FP / alert / compute advantage mạnh nhất

Temporal-CV aggregate:

```text
Logistic Regression

mean F1:
0.495102

std F1:
0.038315

mean Recall:
0.385526

mean Precision:
0.698296
```

Precision fold leads:

`2 / 3`

Pooled temporal-CV operational evidence:

```text
FP:
321

alerts:
1,063

alert rate:
0.0008191352

total fit seconds:
1.240
```

So với RF:

```text
FP:
351

alerts:
1,150

fit seconds:
24.115
```

và DT:

```text
FP:
880

alerts:
1,787

fit seconds:
2.668
```

Interpretation:

LR có operational role rõ nhất:

```text
highest mean Precision
lowest FP
lowest alert burden
lowest fit cost
```

Dù primary F1 thấp hơn RF, LR vẫn còn competitive nhờ precision/operational-efficiency trade-off.

Status:

`VERIFIED`

---

## M7.4-F04 — LR vs DT không có dominance một chiều

Pairwise primary F1:

```text
LR − DT

F1 fold leads:
2 / 1

Δ mean F1:
+0.007710
```

Nhưng:

```text
DT:
higher Recall
lower FN
lower std F1

LR:
higher Precision
lower FP
lower alert burden
lower compute cost
```

Interpretation:

Không có bằng chứng đủ để coi một trong hai family là redundant chỉ dựa trên aggregate F1.

Status:

`REVIEWED`

---

## M7.4-F05 — Cross-stage role consistency rất mạnh

Observed:

```text
F1 leader
M6:
RF

M7 temporal CV:
RF
```

```text
Recall leader
M6:
DT

M7 temporal CV:
DT
```

```text
lowest FN
M6:
DT

M7 temporal CV:
DT
```

```text
Precision leader
M6:
LR

M7 temporal CV:
LR
```

```text
lowest FP
M6:
LR

M7 temporal CV:
LR
```

```text
lowest alerts
M6:
LR

M7 temporal CV:
LR
```

```text
lowest fit cost
M6:
LR

M7 temporal CV:
LR
```

Interpretation:

Các model-family roles không chỉ xuất hiện trong một evaluation snapshot.

Chúng lặp lại giữa:

- M6 frozen external-validation evidence;
- M7 temporal-CV robustness evidence.

Status:

`VERIFIED`

---

## M7.4-F06 — M6 complementarity cho thấy cả ba family còn có non-redundant signal

Frozen M6 W_SHORT overlap:

```text
actual fraud:
1,052

all-three miss:
618

caught by at least one:
434
```

Model-specific only catches:

```text
LR:
31

DT:
68

RF:
28
```

Model-specific only false positives:

```text
LR:
58

DT:
388

RF:
68
```

Interpretation:

Cả ba model đều có fraud cases mà hai model còn lại không bắt được trong frozen M6 evidence.

Đặc biệt DT có nhiều model-specific-only catches nhất.

Điều này không phải ensemble recommendation.

Nó chỉ cho thấy chưa có bằng chứng đủ mạnh để coi một family là hoàn toàn redundant trước M7.5/M7.6.

Status:

`SUPPORTING EVIDENCE`

---

## M7.4-F07 — Computational cost là supporting evidence, không phải winner criterion

Temporal-CV total fit time:

```text
LR:
1.240 s

DT:
2.668 s

RF:
24.115 s
```

RF đắt hơn rõ rệt.

Nhưng RF cũng có:

```text
highest mean F1
3 / 3 F1 fold leads
```

Vì vậy runtime không thể override predictive evidence.

Ngược lại, LR computational advantage là legitimate supporting factor khi downstream tuning budget được thiết kế.

Status:

`PROTOCOL-CONSISTENT`

---

## M7.4-F08 — Không candidate nào đủ evidence để bị loại ở M7.4

Evidence roles:

```text
RF:
primary F1 leader
3 / 3 F1 fold leads

DT:
Recall leader
lowest FN
lowest std F1
most M6 unique-only fraud catches

LR:
Precision leader
lowest FP
lowest alerts
lowest fit cost
```

M7.4 policy cho phép loại candidate chỉ khi reviewed evidence cho thấy candidate không còn competitive và Decision Log ghi rõ lý do.

Observed evidence hiện tại cho thấy:

`ALL THREE REMAIN COMPETITIVE FOR DISTINCT REASONS`.

Do đó không có đủ căn cứ để drop LR, DT hoặc RF khỏi downstream candidate set ở bước này.

Status:

`VERIFIED`

---

## M7.4-F09 — Reviewed model shortlist

Decision:

```text
MODEL SHORTLIST

LR
DT
RF
```

Meaning:

`RETAIN ALL 3 FAMILIES`

Reason:

- RF giữ primary-F1 advantage;
- DT giữ Recall/FN/stability advantage;
- LR giữ Precision/FP/alert/compute advantage;
- M6 complementarity cho thấy cả ba còn non-redundant evidence.

Đây là shortlist decision.

Nó **không** phải final model-family selection.

Status:

`LOCKED FOR DOWNSTREAM M7`

---

## M7.4-F10 — Model-family winner vẫn OPEN

M7.4 không freeze final family.

Observed evidence chỉ đủ để khóa candidate set cho downstream controlled interventions/tuning.

Therefore:

```text
Model-family Winner:
OPEN

Final Model:
OPEN
```

Status:

`BOUNDARY PRESERVED`

---

## M7.4-F11 — External VALIDATION / tuning / FINAL TEST boundary sạch

Observed:

```text
New external VALIDATION scoring:
False

Retraining:
False

Hyperparameter tuning:
False

Imbalance intervention:
False

Threshold optimization:
False

FINAL TEST accessed:
False
```

Runtime gate:

`M7.4 SELECTION-BOUNDARY GATE: PASS`

Status:

`VERIFIED`

---

## M7.4-F12 — Persistence / round-trip

Persisted pre-review evidence artifacts:

```text
data/processed/m7_04_model_family_robustness_shortlist/
    m7_04_model_family_robustness.json
    m7_04_shortlist_manifest.json
```

Observed fingerprints:

```text
Result SHA256:
65f353209a55e7d754d6b5551b3d35f52bf91c8df12bb5cb53d5ed739cfacb58

Manifest SHA256:
3278f88fe29326f01caae0687efddd107d0ec0f362456069a39fd952dfed9762
```

Lưu ý:

Các runtime artifacts được persist trước human/AI review nên field shortlist trong JSON vẫn là:

`OPEN — REQUIRES M7.4 RUNTIME REVIEW`.

Reviewed shortlist được khóa trong notebook review này; không rewrite ngược runtime artifact.

Status:

`VERIFIED`

---

## M7.4-F13 — M7.5 readiness

Necessary conditions:

```text
M7.4 technical gates:
PASS

W_SHORT scope:
LOCKED

Model shortlist:
LR / DT / RF

Model-family winner:
OPEN

Imbalance strategy:
NONE baseline reference

FINAL TEST:
PROTECTED
```

Status:

`READY FOR M7.5`

# 24. Decision Log M7.4 — sau runtime review

## M7.4-D01 — Training-window scope

Decision:

`W_SHORT`

Status:

`INHERITED FROM REVIEWED M7.3 — VERIFIED — LOCKED`

---

## M7.4-D02 — Candidate families

Reviewed candidates:

```text
Logistic Regression
LR-B04-LBFGS-L2-C1

Decision Tree
DT-B01-DEFAULT-GINI-UNPRUNED

Random Forest
RF-B01-100-GINI-SQRT-BOOTSTRAP
```

Status:

`VERIFIED`

---

## M7.4-D03 — Comparison population

Decision:

Q2/Q3/Q4-2018 temporal-CV W_SHORT folds from M7.3.

Observed:

`9 / 9 W_SHORT fold results available`

Status:

`LOCKED / VERIFIED`

---

## M7.4-D04 — Retraining

Decision:

No retraining in M7.4.

Exact M7.3 W_SHORT fold results reused.

Observed:

`retraining_performed = False`

Status:

`LOCKED / VERIFIED`

---

## M7.4-D05 — Same-window comparability

Observed:

```text
Q2:
PASS

Q3:
PASS

Q4:
PASS
```

Each fold:

`15 / 15 shared fields match`.

Decision:

Model-family effect is compared only within same W_SHORT fold population.

Status:

`LOCKED / VERIFIED`

---

## M7.4-D06 — Imbalance strategy

Decision:

`NONE`

for current shortlist evidence.

Status:

`INHERITED — LOCKED`

---

## M7.4-D07 — Threshold policy

Decision:

`DEFAULT_MODEL_DECISION_RULE`

No threshold optimization.

Status:

`LOCKED / VERIFIED`

---

## M7.4-D08 — Random state

Decision:

`42`

Status:

`LOCKED / VERIFIED`

---

## M7.4-D09 — Primary F1 role

Observed:

```text
RF mean F1:
0.517612

RF F1 fold leads:
3 / 3
```

Decision:

`RF = PRIMARY F1 LEADER`

Status:

`DESCRIPTIVE ROLE — LOCKED`

---

## M7.4-D10 — Recall/FN role

Observed:

```text
DT mean Recall:
0.471101

DT Recall fold leads:
3 / 3

DT pooled FN:
1,027

DT std F1:
0.013631
```

Decision:

`DT = RECALL / FN / STABILITY ROLE`

Status:

`DESCRIPTIVE ROLE — LOCKED`

---

## M7.4-D11 — Precision/operational role

Observed:

```text
LR mean Precision:
0.698296

LR FP:
321

LR alerts:
1,063

LR total fit seconds:
1.240
```

Decision:

`LR = PRECISION / FP / ALERT / COMPUTE ROLE`

Status:

`DESCRIPTIVE ROLE — LOCKED`

---

## M7.4-D12 — M6 supporting evidence

Decision:

Frozen M6 external-validation/error findings are supporting evidence only.

Observed role consistency with temporal CV:

```text
F1 leader:
RF / RF

Recall leader:
DT / DT

Precision leader:
LR / LR

lowest FN:
DT / DT

lowest FP:
LR / LR

lowest alerts:
LR / LR

lowest fit cost:
LR / LR
```

Status:

`VERIFIED`

---

## M7.4-D13 — Complementarity evidence

Observed M6 model-specific-only catches:

```text
LR:
31

DT:
68

RF:
28
```

Decision:

Use as non-redundancy supporting evidence only.

Do not infer ensemble recommendation.

Status:

`LOCKED INTERPRETATION`

---

## M7.4-D14 — Model shortlist

Decision:

```text
MODEL SHORTLIST

LR
DT
RF
```

Disposition:

`RETAIN ALL 3`

Reason:

Không candidate nào vừa mất competitiveness trên primary/secondary/operational evidence vừa thiếu compensating role.

Status:

`LOCKED FOR DOWNSTREAM M7`

---

## M7.4-D15 — Candidate elimination

Decision:

```text
LR:
NOT ELIMINATED

DT:
NOT ELIMINATED

RF:
NOT ELIMINATED
```

Reason:

M7.4 evidence không đủ justify drop bất kỳ family nào.

Status:

`LOCKED`

---

## M7.4-D16 — Model-family winner

Decision:

Not selected in M7.4.

Status:

`OPEN`

---

## M7.4-D17 — Hyperparameter tuning

Decision:

None in M7.4.

Status:

`LOCKED / VERIFIED`

---

## M7.4-D18 — Imbalance intervention

Decision:

None in M7.4.

M7.5 starts from:

`NONE`

and tests interventions only under controlled design.

Status:

`LOCKED / VERIFIED`

---

## M7.4-D19 — External VALIDATION

Decision:

No new scoring in M7.4.

Frozen M6 evidence only.

Status:

`LOCKED / VERIFIED`

---

## M7.4-D20 — Threshold optimization

Decision:

None in M7.4.

Status:

`LOCKED / VERIFIED`

---

## M7.4-D21 — FINAL TEST

Decision:

No access.

Status:

`INHERITED — VERIFIED — LOCKED`

---

## M7.4-D22 — M7.5 handoff

Decision:

Proceed to:

`M7.5 — Controlled class-imbalance experiments`

with:

```text
training-window:
W_SHORT

shortlisted families:
LR
DT
RF

initial imbalance reference:
NONE
```

Status:

`READY`

# 25. M7.4 Gate

## Technical runtime gates

```text
G01_SOURCE_LOCATION                         → PASS
G02_M7_3_REVIEWED_HANDOFF                   → PASS
G03_W_SHORT_SCOPE                           → PASS
G04_SAME_WINDOW_COMPARABILITY               → PASS
G05_AGGREGATE_RECONSTRUCTION                → PASS
G06_FOLDWISE_F1                             → PASS
G07_FOLDWISE_RECALL                         → PASS
G08_FOLDWISE_PRECISION                      → PASS
G09_PAIRWISE_F1_ROBUSTNESS                  → PASS
G10_OPERATIONAL_EVIDENCE                    → PASS
G11_M6_SUPPORTING_EVIDENCE                  → PASS
G12_CROSS_STAGE_ROLE_CONSISTENCY            → PASS
G13_CANDIDATE_PROFILES                      → PASS
G14_DESCRIPTIVE_ROLE_MAP                    → PASS
G15_NO_AUTO_SHORTLIST                       → PASS
G16_PERSISTENCE                             → PASS
G17_ROUND_TRIP                              → PASS
G18_NO_NEW_EXTERNAL_VALIDATION_SCORING      → PASS
G19_NO_TUNING_IMBALANCE_THRESHOLD           → PASS
G20_FINAL_TEST_PROTECTION                    → PASS
```

Technical gates:

`20 / 20 PASS`

---

## R01 — Execution complete?

Evidence:

```text
20 / 20 code cells
execution_count = 1 → 20
errors = 0
stderr = 0
```

Result:

`PASS`

---

## R02 — Reviewed M7.3 handoff valid?

Evidence:

```text
M7.3 runs:
18 / 18

warnings:
0

training-window direction:
ROBUST W_SHORT PREFERENCE

downstream window:
W_SHORT
```

Result:

`PASS`

---

## R03 — W_SHORT-only family scope valid?

Evidence:

```text
W_SHORT fold results:
9

W_SHORT aggregate records:
3

mixed-window comparison:
NONE
```

Result:

`PASS`

---

## R04 — Same-fold comparability valid?

Evidence:

```text
Q2:
15 shared fields PASS

Q3:
15 shared fields PASS

Q4:
15 shared fields PASS
```

Result:

`PASS`

---

## R05 — Aggregate reconstruction valid?

Evidence:

Exact reconstruction of:

```text
mean F1
std F1
mean Recall
mean Precision
pooled TP/FP/FN/TN
alerts
runtime
```

Result:

`PASS`

---

## R06 — F1 robustness reviewed?

Evidence:

```text
RF F1 fold leads:
3 / 3

RF mean F1:
0.517612

LR mean F1:
0.495102

DT mean F1:
0.487392
```

Result:

`PASS`

---

## R07 — Recall/FN evidence reviewed?

Evidence:

```text
DT Recall fold leads:
3 / 3

DT mean Recall:
0.471101

DT pooled FN:
1,027
```

Result:

`PASS`

---

## R08 — Precision/FP/alert evidence reviewed?

Evidence:

```text
LR mean Precision:
0.698296

LR FP:
321

LR alerts:
1,063
```

Result:

`PASS`

---

## R09 — Stability reviewed?

Evidence:

```text
std F1

DT:
0.013631

RF:
0.029168

LR:
0.038315
```

Result:

`PASS`

---

## R10 — Computational evidence reviewed?

Evidence:

```text
total temporal-CV fit time

LR:
1.240 s

DT:
2.668 s

RF:
24.115 s
```

Result:

`PASS`

---

## R11 — Cross-stage M6/M7 role consistency reviewed?

Evidence:

All seven tracked roles are consistent:

```text
F1 leader
Recall leader
Precision leader
lowest FN
lowest FP
lowest alerts
lowest fit cost
```

Result:

`PASS`

---

## R12 — Complementarity evidence reviewed?

Evidence:

M6 model-specific-only fraud catches:

```text
LR:
31

DT:
68

RF:
28
```

Interpretation boundary:

`NO ENSEMBLE CLAIM`

Result:

`PASS`

---

## R13 — Candidate competitiveness reviewed?

Evidence:

```text
RF:
F1 role

DT:
Recall/FN/stability role

LR:
Precision/FP/alert/compute role
```

Result:

`PASS`

---

## R14 — Shortlist decision supportable?

Decision:

```text
MODEL SHORTLIST:
LR / DT / RF
```

Disposition:

`RETAIN ALL 3`

Reason:

Each candidate has distinct reviewed competitive evidence.

Result:

`PASS`

---

## R15 — Candidate elimination justified?

Decision:

`NONE ELIMINATED`

Reason:

No family lacks both predictive competitiveness and compensating role.

Result:

`PASS`

---

## R16 — Model-family winner boundary preserved?

Evidence:

```text
Model-family Winner:
OPEN

Final Model:
OPEN
```

Result:

`PASS`

---

## R17 — No hidden tuning / imbalance / threshold search?

Evidence:

```text
Retraining:
NONE

Hyperparameter tuning:
NONE

Imbalance intervention:
NONE

Threshold optimization:
NONE
```

Result:

`PASS`

---

## R18 — External VALIDATION boundary preserved?

Evidence:

```text
new external VALIDATION scoring:
False

M6 frozen evidence:
supporting context only
```

Result:

`PASS`

---

## R19 — Persistence / round-trip valid?

Evidence:

```text
result artifact:
PASS

manifest:
PASS

SHA fingerprints:
recorded
```

Result:

`PASS`

---

## R20 — FINAL TEST protected?

Evidence:

`final_test_accessed = False`

Result:

`PASS`

---

## R21 — M7.5 handoff valid?

Required:

- W_SHORT locked;
- shortlist locked;
- NONE remains baseline imbalance reference;
- final family still OPEN;
- FINAL TEST protected.

Observed:

`PASS`

---

## Overall M7.4 Gate

```text
Technical gates:
20 / 20 PASS

Runtime review gates:
21 / 21 PASS

Blocking issue:
NONE
```

Final:

`M7.4 — PASS`

Model shortlist:

`LR / DT / RF — RETAIN ALL 3`

Model-family winner:

`OPEN`

Handoff:

`READY FOR M7.5`

# 26. Kết luận M7.4

M7.4 đã hoàn thành model-family robustness / shortlist review trên đúng training-window scope đã được M7.3 khóa:

`W_SHORT`.

Primary temporal-CV comparison:

```text
3 model families
×
3 temporal folds
=
9 W_SHORT fold results
```

Experiment/review integrity:

```text
Technical gates:
20 / 20 PASS

Runtime review gates:
21 / 21 PASS

Same-fold comparability:
3 / 3 PASS

New external VALIDATION scoring:
NONE

Retraining:
NONE

Hyperparameter tuning:
NONE

Imbalance intervention:
NONE

Threshold optimization:
NONE

FINAL TEST:
PROTECTED
```

Model-family evidence không cho một dominance profile duy nhất.

Thay vào đó ba family có ba competitive roles khác nhau:

```text
Random Forest

mean F1:
0.517612

F1 fold leads:
3 / 3

role:
PRIMARY F1 LEADER
```

```text
Decision Tree

mean Recall:
0.471101

Recall fold leads:
3 / 3

FN:
1,027

std F1:
0.013631

role:
RECALL / FN / STABILITY
```

```text
Logistic Regression

mean Precision:
0.698296

FP:
321

alerts:
1,063

total fit seconds:
1.240

role:
PRECISION / FP / ALERT / COMPUTE
```

M6 supporting evidence củng cố cùng role structure:

```text
F1 leader:
RF

Recall / lowest FN:
DT

Precision / lowest FP / lowest alerts / lowest fit cost:
LR
```

M6 complementarity còn cho thấy model-specific-only fraud catches:

```text
LR:
31

DT:
68

RF:
28
```

Vì vậy reviewed evidence chưa justify loại bất kỳ family nào khỏi downstream candidate set.

Shortlist decision:

```text
MODEL SHORTLIST

Logistic Regression
Decision Tree
Random Forest
```

Disposition:

`RETAIN ALL 3`

Đây không phải final model selection.

Final state:

```text
M7.4 — PASS

Training-window Scope:
W_SHORT — LOCKED

Model Shortlist:
LR / DT / RF — LOCKED

Candidate Elimination:
NONE

Model-family Winner:
OPEN

Final Model:
OPEN

Final Imbalance Strategy:
OPEN

Final Threshold:
OPEN

Blocking Issue:
NONE

FINAL TEST:
PROTECTED

READY FOR M7.5
```

Bước tiếp theo:

`M7.5 — Controlled class-imbalance experiments`

M7.5 phải bắt đầu từ:

`NONE`

và chỉ thử intervention có kiểm soát theo canonical order:

```text
NONE
→
CLASS_WEIGHT
→
RANDOM OVER/UNDER if justified
→
SMOTE only if conditional gate passes
```

Primary experiment phải giữ:

```text
training window:
W_SHORT

shortlisted families:
LR / DT / RF

same folds
same preprocessing
same threshold policy
same metric code
same random-state policy
```

Variable:

`imbalance strategy`

M7.5 phải đặc biệt đọc:

```text
Recall gain
Precision cost
FN reduction
FP increase
alert burden
fold stability
```

Model-family winner tiếp tục:

`OPEN`

FINAL TEST tiếp tục:

`PROTECTED`
