# M6.5 — Cross-model comparative evaluation

Milestone:

`M6 — Evaluation + Error Analysis`

Substep:

`M6.5 — Cross-model Comparative Evaluation`

Work type:

`RUNTIME COMPARATIVE EVALUATION`

Câu hỏi trung tâm:

> Trên cùng training-window population, Logistic Regression / Decision Tree / Random Forest thể hiện trade-off khác nhau như thế nào?

Primary comparison groups:

```text
W_SHORT:
LR vs DT vs RF

W_LONG:
LR vs DT vs RF
```

M6.5 phân tích:

- F1_fraud;
- Recall_fraud;
- Precision_fraud;
- TP / FP / FN / TN;
- alert count / rate;
- descriptive probability behavior;
- computational evidence.

M6.5 không:

- fit/retrain model;
- tuning;
- resampling;
- class-weight experiment;
- threshold optimization;
- so model bằng cách trộn training window;
- khóa final model;
- mở FINAL TEST.

Runtime-dependent status trước khi chạy:

`NOT YET VERIFIED`

## 1. Contract kế thừa

M6.5 kế thừa:

- `CANON-Kế hoạch Milestone 6 — Evaluation và Error Analysis`;
- `CANON-M6.1 — Evaluation Charter`;
- `M6.2 — verified Evaluation Registry`;
- `M6.3 — verified metric/confusion profiles`;
- `M6.4 — verified controlled training-window comparison`.

Cross-model rule:

```text
MODEL FAMILY CHANGES
TRAINING WINDOW STAYS FIXED
```

Valid:

```text
W_SHORT:
LR-SHORT vs DT-SHORT vs RF-SHORT

W_LONG:
LR-LONG vs DT-LONG vs RF-LONG
```

Invalid để kết luận về model family:

```text
LR-SHORT vs RF-LONG
```

vì comparison đó thay đồng thời:

```text
model family
+
training window
```

## 2. Metric / diagnostic contract

Primary metric:

`F1_fraud`

Mandatory secondary evidence:

```text
Recall_fraud
Precision_fraud
TP
FP
FN
TN
predicted-positive count
predicted-positive rate
```

Accuracy:

`REFERENCE ONLY`

Probability behavior:

`DESCRIPTIVE ONLY`

Computational evidence:

```text
fit_seconds
prediction_seconds
warning_count
```

M6.5 có thể ghi:

`metric-specific leader`

trong từng same-window group.

Ví dụ:

`highest F1 within W_SHORT`

Nhưng điều đó không được đổi tên thành:

`final model winner`.

## 3. Probability-analysis boundary

Persisted `risk_score` là positive-class fraud probability từ M5.

M6.5 được phép mô tả:

```text
overall score distribution
fraud score distribution
non-fraud score distribution
quantiles
mean/median separation
```

M6.5 không:

- search threshold;
- tìm F1-max threshold;
- chọn threshold riêng cho model;
- calibration tuning;
- dùng probability analysis để sửa `y_pred`.

Probability evidence chỉ giúp hiểu:

> Model đang phân bố risk score khác nhau như thế nào trên cùng validation population?


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


## 4. Locate M6.2 / M6.3 / M6.4 artifacts


```python

M4_REL = (
    Path("data")
    / "processed"
    / "m4_07_baseline_ready"
)

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

M6_04_REL = (
    Path("data")
    / "processed"
    / "m6_04_training_window_comparison"
)

M6_05_REL = (
    Path("data")
    / "processed"
    / "m6_05_cross_model_comparative_evaluation"
)


candidate_roots = [
    Path.cwd(),
    *list(Path.cwd().parents)[:6],
]


PROJECT_ROOT = None

required_rel_paths = [
    M4_REL
    / "y_validation.npy",

    M6_02_REL
    / "m6_02_evaluation_registry.json",

    M6_03_REL
    / "m6_03_metric_confusion_analysis.json",

    M6_04_REL
    / "m6_04_training_window_comparison.json",

    M6_04_REL
    / "m6_04_comparison_manifest.json",
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
        "Không tìm thấy PROJECT_ROOT chứa đầy đủ "
        "M4.7 + M6.2 + M6.3 + M6.4 artifacts."
    )


M4_DIR = (
    PROJECT_ROOT
    / M4_REL
)

M6_02_DIR = (
    PROJECT_ROOT
    / M6_02_REL
)

M6_03_DIR = (
    PROJECT_ROOT
    / M6_03_REL
)

M6_04_DIR = (
    PROJECT_ROOT
    / M6_04_REL
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / M6_05_REL
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


M6_02_REGISTRY_PATH = (
    M6_02_DIR
    / "m6_02_evaluation_registry.json"
)

M6_03_ANALYSIS_PATH = (
    M6_03_DIR
    / "m6_03_metric_confusion_analysis.json"
)

M6_04_COMPARISON_PATH = (
    M6_04_DIR
    / "m6_04_training_window_comparison.json"
)

M6_04_MANIFEST_PATH = (
    M6_04_DIR
    / "m6_04_comparison_manifest.json"
)


print("PROJECT_ROOT:")
print(PROJECT_ROOT)

print("\nOUTPUT_DIR:")
print(OUTPUT_DIR)

```

    PROJECT_ROOT:
    /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening
    
    OUTPUT_DIR:
    /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/processed/m6_05_cross_model_comparative_evaluation


## 5. Verify upstream handoff


```python

with open(
    M6_02_REGISTRY_PATH,
    "r",
    encoding="utf-8",
) as file:
    m6_02_registry = json.load(
        file
    )


with open(
    M6_03_ANALYSIS_PATH,
    "r",
    encoding="utf-8",
) as file:
    m6_03_analysis = json.load(
        file
    )


with open(
    M6_04_COMPARISON_PATH,
    "r",
    encoding="utf-8",
) as file:
    m6_04_comparison = json.load(
        file
    )


with open(
    M6_04_MANIFEST_PATH,
    "r",
    encoding="utf-8",
) as file:
    m6_04_manifest = json.load(
        file
    )


EXPECTED_VALIDATION_ROWS = 712_458
EXPECTED_VALIDATION_FRAUD = 1_052


assert (
    m6_02_registry[
        "official_run_count"
    ]
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
        "final_test_accessed"
    ]
    is False
)


assert (
    m6_03_analysis[
        "validation_rows"
    ]
    == EXPECTED_VALIDATION_ROWS
)

assert (
    len(
        m6_03_analysis[
            "records"
        ]
    )
    == 6
)

assert (
    m6_03_analysis[
        "final_test_accessed"
    ]
    is False
)


assert (
    len(
        m6_04_comparison[
            "comparison_pairs"
        ]
    )
    == 3
)

assert (
    m6_04_comparison[
        "selection_state"
    ][
        "training_window_winner"
    ]
    == "OPEN"
)

assert (
    m6_04_comparison[
        "selection_state"
    ][
        "model_family_winner"
    ]
    == "OPEN"
)

assert (
    m6_04_comparison[
        "final_test_accessed"
    ]
    is False
)


assert (
    m6_04_manifest[
        "controlled_pairs_built"
    ]
    == 3
)

assert (
    m6_04_manifest[
        "controlled_pair_integrity"
    ]
    == "PASS"
)

assert (
    m6_04_manifest[
        "final_training_window_selected"
    ]
    is False
)

assert (
    m6_04_manifest[
        "final_test_accessed"
    ]
    is False
)


print(
    "M6.5 UPSTREAM HANDOFF GATE: PASS"
)

```

    M6.5 UPSTREAM HANDOFF GATE: PASS


## 6. Load canonical validation target

Probability behavior phải dùng cùng target/order với M6.2 registry.

M6.5 verify SHA-256 của `y_validation.npy` trước khi dùng class-conditioned score summaries.


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


Y_VALIDATION_PATH = (
    M4_DIR
    / "y_validation.npy"
)


y_validation = np.load(
    Y_VALIDATION_PATH,
    allow_pickle=False,
)


assert y_validation.ndim == 1

assert (
    len(
        y_validation
    )
    == EXPECTED_VALIDATION_ROWS
)

assert (
    int(
        y_validation.sum()
    )
    == EXPECTED_VALIDATION_FRAUD
)

assert (
    sha256_file(
        Y_VALIDATION_PATH
    )
    == m6_02_registry[
        "y_validation_sha256"
    ]
)


fraud_mask = (
    y_validation == 1
)

non_fraud_mask = (
    y_validation == 0
)


assert int(
    fraud_mask.sum()
) == EXPECTED_VALIDATION_FRAUD

assert int(
    non_fraud_mask.sum()
) == (
    EXPECTED_VALIDATION_ROWS
    - EXPECTED_VALIDATION_FRAUD
)


print(
    "M6.5 VALIDATION TARGET GATE: PASS"
)

```

    M6.5 VALIDATION TARGET GATE: PASS


## 7. Build same-window comparison groups

Canonical groups:

```text
W_SHORT:
M5-LR-SHORT-B04
M5-DT-SHORT-B01
M5-RF-SHORT-B01

W_LONG:
M5-LR-LONG-B04
M5-DT-LONG-B01
M5-RF-LONG-B01
```

M6.5 không tạo mixed-window group.


```python

GROUP_SPECS = {
    "W_SHORT": [
        "M5-LR-SHORT-B04",
        "M5-DT-SHORT-B01",
        "M5-RF-SHORT-B01",
    ],

    "W_LONG": [
        "M5-LR-LONG-B04",
        "M5-DT-LONG-B01",
        "M5-RF-LONG-B01",
    ],
}


profiles_by_id = {
    record[
        "experiment_id"
    ]:
    record
    for record
    in m6_03_analysis[
        "records"
    ]
}


registry_by_id = {
    record[
        "experiment_id"
    ]:
    record
    for record
    in m6_02_registry[
        "records"
    ]
}


assert (
    set(
        profiles_by_id
    )
    == set(
        registry_by_id
    )
)


for window_id, experiment_ids in (
    GROUP_SPECS.items()
):
    assert len(
        experiment_ids
    ) == 3

    for experiment_id in (
        experiment_ids
    ):
        assert (
            experiment_id
            in profiles_by_id
        )

        assert (
            profiles_by_id[
                experiment_id
            ][
                "training_window_id"
            ]
            == window_id
        )

        assert (
            registry_by_id[
                experiment_id
            ][
                "training_window_id"
            ]
            == window_id
        )


print(
    "W_SHORT group:",
    GROUP_SPECS[
        "W_SHORT"
    ],
)

print(
    "\nW_LONG group:",
    GROUP_SPECS[
        "W_LONG"
    ],
)

print(
    "\nM6.5 SAME-WINDOW GROUP GATE: PASS"
)

```

    W_SHORT group: ['M5-LR-SHORT-B04', 'M5-DT-SHORT-B01', 'M5-RF-SHORT-B01']
    
    W_LONG group: ['M5-LR-LONG-B04', 'M5-DT-LONG-B01', 'M5-RF-LONG-B01']
    
    M6.5 SAME-WINDOW GROUP GATE: PASS


## 8. Same-window comparability audit

Trong từng group, các run phải giống:

- validation period;
- validation rows;
- validation fraud rows;
- feature version;
- preprocessing version;
- matrix schema;
- imbalance strategy;
- threshold policy;
- risk-score kind.

Model family/config **được phép khác** vì đó chính là variable đang so.


```python

SAME_WINDOW_CONTROLLED_FIELDS = [
    "training_window_id",
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


group_integrity = []


for window_id, experiment_ids in (
    GROUP_SPECS.items()
):
    group_records = [
        registry_by_id[
            experiment_id
        ]
        for experiment_id
        in experiment_ids
    ]

    reference = group_records[0]

    mismatches = []

    for record in (
        group_records[1:]
    ):
        for field in (
            SAME_WINDOW_CONTROLLED_FIELDS
        ):
            if (
                record[field]
                != reference[field]
            ):
                mismatches.append(
                    (
                        record[
                            "experiment_id"
                        ],
                        field,
                    )
                )

    assert not mismatches, (
        f"{window_id}: "
        f"same-window mismatch: "
        f"{mismatches}"
    )

    families = {
        record[
            "model_family"
        ]
        for record
        in group_records
    }

    assert families == {
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
    }

    group_integrity.append(
        {
            "training_window_id":
                window_id,

            "models":
                sorted(
                    families
                ),

            "controlled_fields_checked":
                len(
                    SAME_WINDOW_CONTROLLED_FIELDS
                ),

            "integrity_status":
                "PASS",
        }
    )


for item in group_integrity:
    print(
        item[
            "training_window_id"
        ],
        "→",
        item[
            "integrity_status"
        ],
        "| fields:",
        item[
            "controlled_fields_checked"
        ],
    )


print(
    "\nM6.5 SAME-WINDOW COMPARABILITY GATE: PASS"
)

```

    W_SHORT → PASS | fields: 10
    W_LONG → PASS | fields: 10
    
    M6.5 SAME-WINDOW COMPARABILITY GATE: PASS


## 9. Load computational context

Computational evidence được đọc từ persisted M5 summary.

M6.5 không chạy lại training.

Các field:

```text
fit_seconds
prediction_seconds
warning_count
```

chỉ là supporting evidence.


```python

runtime_context = {}


for experiment_id, record in (
    registry_by_id.items()
):
    summary_path = (
        PROJECT_ROOT
        / record[
            "summary_artifact"
        ]
    )

    assert summary_path.exists()

    with open(
        summary_path,
        "r",
        encoding="utf-8",
    ) as file:
        summary = json.load(
            file
        )

    assert (
        summary[
            "experiment_id"
        ]
        == experiment_id
    )

    assert (
        summary[
            "final_test_accessed"
        ]
        is False
    )

    runtime_context[
        experiment_id
    ] = {
        "fit_seconds":
            float(
                summary[
                    "fit_seconds"
                ]
            ),

        "prediction_seconds":
            float(
                summary[
                    "prediction_seconds"
                ]
            ),

        "warning_count":
            int(
                summary[
                    "warning_count"
                ]
            ),
    }


assert len(
    runtime_context
) == 6


print(
    "M6.5 COMPUTATIONAL CONTEXT GATE: PASS"
)

```

    M6.5 COMPUTATIONAL CONTEXT GATE: PASS


## 10. Probability summary contract

M6.5 dùng class-conditioned risk-score summaries.

Cho mỗi run:

```text
overall:
mean / median / q90 / q95 / q99

fraud:
mean / q25 / median / q75 / q90

non-fraud:
mean / median / q90 / q95 / q99

descriptive separation:
fraud_mean - nonfraud_mean
fraud_median - nonfraud_median
```

Không có threshold search.


```python

def summarize_scores(
    scores,
):
    scores = np.asarray(
        scores,
        dtype=np.float64,
    )

    assert scores.ndim == 1
    assert np.isfinite(
        scores
    ).all()

    return {
        "mean":
            float(
                np.mean(
                    scores
                )
            ),

        "q25":
            float(
                np.quantile(
                    scores,
                    0.25,
                )
            ),

        "median":
            float(
                np.quantile(
                    scores,
                    0.50,
                )
            ),

        "q75":
            float(
                np.quantile(
                    scores,
                    0.75,
                )
            ),

        "q90":
            float(
                np.quantile(
                    scores,
                    0.90,
                )
            ),

        "q95":
            float(
                np.quantile(
                    scores,
                    0.95,
                )
            ),

        "q99":
            float(
                np.quantile(
                    scores,
                    0.99,
                )
            ),

        "min":
            float(
                np.min(
                    scores
                )
            ),

        "max":
            float(
                np.max(
                    scores
                )
            ),
    }


def build_probability_summary(
    experiment_id,
):
    registry_record = (
        registry_by_id[
            experiment_id
        ]
    )

    risk_score_path = (
        PROJECT_ROOT
        / registry_record[
            "risk_score_artifact"
        ]
    )

    assert risk_score_path.exists()

    assert (
        sha256_file(
            risk_score_path
        )
        == registry_record[
            "risk_score_sha256"
        ]
    )

    scores = np.load(
        risk_score_path,
        allow_pickle=False,
    )

    assert scores.ndim == 1

    assert len(
        scores
    ) == EXPECTED_VALIDATION_ROWS

    assert np.isfinite(
        scores
    ).all()

    assert np.all(
        scores >= 0.0
    )

    assert np.all(
        scores <= 1.0
    )

    overall = summarize_scores(
        scores
    )

    fraud = summarize_scores(
        scores[
            fraud_mask
        ]
    )

    non_fraud = summarize_scores(
        scores[
            non_fraud_mask
        ]
    )

    return {
        "experiment_id":
            experiment_id,

        "risk_score_kind":
            registry_record[
                "risk_score_kind"
            ],

        "overall":
            overall,

        "fraud":
            fraud,

        "non_fraud":
            non_fraud,

        "fraud_minus_nonfraud_mean":
            float(
                fraud["mean"]
                - non_fraud["mean"]
            ),

        "fraud_minus_nonfraud_median":
            float(
                fraud["median"]
                - non_fraud["median"]
            ),

        "threshold_search_performed":
            False,

        "final_test_accessed":
            False,
    }


probability_summaries = {
    experiment_id:
        build_probability_summary(
            experiment_id
        )
    for experiment_id
    in registry_by_id
}


assert len(
    probability_summaries
) == 6


print(
    "M6.5 PROBABILITY SUMMARY BUILD GATE: PASS"
)

```

    M6.5 PROBABILITY SUMMARY BUILD GATE: PASS


## 11. Build same-window model records

Mỗi model record kết hợp:

- M6.3 aggregate profile;
- M5 runtime context;
- M6.5 descriptive probability summary.

Không thay persisted predictions.


```python

def build_model_record(
    experiment_id,
):
    profile = (
        profiles_by_id[
            experiment_id
        ]
    )

    registry = (
        registry_by_id[
            experiment_id
        ]
    )

    return {
        "experiment_id":
            experiment_id,

        "model_family":
            profile[
                "model_family"
            ],

        "model_config_id":
            profile[
                "model_config_id"
            ],

        "training_window_id":
            profile[
                "training_window_id"
            ],

        "f1_fraud":
            float(
                profile[
                    "f1_fraud"
                ]
            ),

        "recall_fraud":
            float(
                profile[
                    "recall_fraud"
                ]
            ),

        "precision_fraud":
            float(
                profile[
                    "precision_fraud"
                ]
            ),

        "accuracy_reference":
            float(
                profile[
                    "accuracy_reference"
                ]
            ),

        "tp":
            int(
                profile["tp"]
            ),

        "fp":
            int(
                profile["fp"]
            ),

        "fn":
            int(
                profile["fn"]
            ),

        "tn":
            int(
                profile["tn"]
            ),

        "alert_count":
            int(
                profile[
                    "alert_count"
                ]
            ),

        "alert_rate":
            float(
                profile[
                    "alert_rate"
                ]
            ),

        "fit_seconds":
            runtime_context[
                experiment_id
            ][
                "fit_seconds"
            ],

        "prediction_seconds":
            runtime_context[
                experiment_id
            ][
                "prediction_seconds"
            ],

        "warning_count":
            runtime_context[
                experiment_id
            ][
                "warning_count"
            ],

        "probability_summary":
            probability_summaries[
                experiment_id
            ],

        "artifact_integrity_check":
            registry[
                "artifact_integrity_check"
            ],

        "final_test_accessed":
            False,
    }


same_window_records = {
    window_id: [
        build_model_record(
            experiment_id
        )
        for experiment_id
        in experiment_ids
    ]
    for window_id, experiment_ids
    in GROUP_SPECS.items()
}


assert set(
    same_window_records
) == {
    "W_SHORT",
    "W_LONG",
}


for window_id, group in (
    same_window_records.items()
):
    assert len(
        group
    ) == 3

    assert {
        record[
            "model_family"
        ]
        for record
        in group
    } == {
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
    }


print(
    "M6.5 SAME-WINDOW MODEL RECORD GATE: PASS"
)

```

    M6.5 SAME-WINDOW MODEL RECORD GATE: PASS


## 12. Metric-specific descriptive extrema

Trong mỗi training window, M6.5 ghi model có:

- highest F1;
- highest Recall;
- highest Precision;
- lowest FN;
- lowest FP;
- lowest alert count;
- lowest fit time.

Đây là:

`DESCRIPTIVE EXTREMA`

không phải:

`FINAL MODEL RANKING`.


```python

def model_name_for_max(
    group,
    field,
):
    return max(
        group,
        key=lambda record:
            record[field],
    )[
        "model_family"
    ]


def model_name_for_min(
    group,
    field,
):
    return min(
        group,
        key=lambda record:
            record[field],
    )[
        "model_family"
    ]


window_extrema = {}


for window_id, group in (
    same_window_records.items()
):
    window_extrema[
        window_id
    ] = {
        "highest_f1":
            model_name_for_max(
                group,
                "f1_fraud",
            ),

        "highest_recall":
            model_name_for_max(
                group,
                "recall_fraud",
            ),

        "highest_precision":
            model_name_for_max(
                group,
                "precision_fraud",
            ),

        "lowest_fn":
            model_name_for_min(
                group,
                "fn",
            ),

        "lowest_fp":
            model_name_for_min(
                group,
                "fp",
            ),

        "lowest_alert_count":
            model_name_for_min(
                group,
                "alert_count",
            ),

        "lowest_fit_seconds":
            model_name_for_min(
                group,
                "fit_seconds",
            ),

        "interpretation":
            "DESCRIPTIVE_EXTREMA_NOT_FINAL_SELECTION",
    }


for window_id, extrema in (
    window_extrema.items()
):
    print("=" * 72)
    print(window_id)

    for key, value in (
        extrema.items()
    ):
        print(
            key,
            "→",
            value,
        )


print(
    "\nM6.5 METRIC-SPECIFIC EXTREMA GATE: PASS"
)

```

    ========================================================================
    W_SHORT
    highest_f1 → Random Forest
    highest_recall → Decision Tree
    highest_precision → Logistic Regression
    lowest_fn → Decision Tree
    lowest_fp → Logistic Regression
    lowest_alert_count → Logistic Regression
    lowest_fit_seconds → Logistic Regression
    interpretation → DESCRIPTIVE_EXTREMA_NOT_FINAL_SELECTION
    ========================================================================
    W_LONG
    highest_f1 → Decision Tree
    highest_recall → Decision Tree
    highest_precision → Random Forest
    lowest_fn → Decision Tree
    lowest_fp → Logistic Regression
    lowest_alert_count → Logistic Regression
    lowest_fit_seconds → Logistic Regression
    interpretation → DESCRIPTIVE_EXTREMA_NOT_FINAL_SELECTION
    
    M6.5 METRIC-SPECIFIC EXTREMA GATE: PASS


## 13. Pairwise same-window deltas

Để không chỉ nhìn extrema, M6.5 tạo pairwise deltas:

```text
DT - LR
RF - LR
RF - DT
```

cho từng window.

Fields:

```text
F1
Recall
Precision
TP
FP
FN
alert count
fit seconds
prediction seconds
fraud-score mean
nonfraud-score mean
score mean separation
```

Không có mixed-window delta.


```python

FAMILY_KEY = {
    "Logistic Regression":
        "LR",

    "Decision Tree":
        "DT",

    "Random Forest":
        "RF",
}


PAIRWISE_SPECS = [
    (
        "DT_MINUS_LR",
        "Decision Tree",
        "Logistic Regression",
    ),
    (
        "RF_MINUS_LR",
        "Random Forest",
        "Logistic Regression",
    ),
    (
        "RF_MINUS_DT",
        "Random Forest",
        "Decision Tree",
    ),
]


def build_pairwise_delta(
    window_id,
    group,
    delta_id,
    left_family,
    right_family,
):
    by_family = {
        record[
            "model_family"
        ]:
        record
        for record
        in group
    }

    left = by_family[
        left_family
    ]

    right = by_family[
        right_family
    ]

    assert (
        left[
            "training_window_id"
        ]
        == window_id
    )

    assert (
        right[
            "training_window_id"
        ]
        == window_id
    )

    return {
        "delta_id":
            delta_id,

        "training_window_id":
            window_id,

        "delta_convention":
            (
                f"{FAMILY_KEY[left_family]}"
                "_MINUS_"
                f"{FAMILY_KEY[right_family]}"
            ),

        "left_family":
            left_family,

        "right_family":
            right_family,

        "delta_f1":
            float(
                left["f1_fraud"]
                - right["f1_fraud"]
            ),

        "delta_recall":
            float(
                left["recall_fraud"]
                - right["recall_fraud"]
            ),

        "delta_precision":
            float(
                left["precision_fraud"]
                - right["precision_fraud"]
            ),

        "delta_tp":
            int(
                left["tp"]
                - right["tp"]
            ),

        "delta_fp":
            int(
                left["fp"]
                - right["fp"]
            ),

        "delta_fn":
            int(
                left["fn"]
                - right["fn"]
            ),

        "delta_alert_count":
            int(
                left["alert_count"]
                - right["alert_count"]
            ),

        "delta_fit_seconds":
            float(
                left["fit_seconds"]
                - right["fit_seconds"]
            ),

        "delta_prediction_seconds":
            float(
                left[
                    "prediction_seconds"
                ]
                - right[
                    "prediction_seconds"
                ]
            ),

        "delta_fraud_score_mean":
            float(
                left[
                    "probability_summary"
                ][
                    "fraud"
                ][
                    "mean"
                ]
                - right[
                    "probability_summary"
                ][
                    "fraud"
                ][
                    "mean"
                ]
            ),

        "delta_nonfraud_score_mean":
            float(
                left[
                    "probability_summary"
                ][
                    "non_fraud"
                ][
                    "mean"
                ]
                - right[
                    "probability_summary"
                ][
                    "non_fraud"
                ][
                    "mean"
                ]
            ),

        "delta_score_mean_separation":
            float(
                left[
                    "probability_summary"
                ][
                    "fraud_minus_nonfraud_mean"
                ]
                - right[
                    "probability_summary"
                ][
                    "fraud_minus_nonfraud_mean"
                ]
            ),

        "mixed_training_window":
            False,

        "final_model_selection":
            False,
    }


pairwise_deltas = {}


for window_id, group in (
    same_window_records.items()
):
    pairwise_deltas[
        window_id
    ] = [
        build_pairwise_delta(
            window_id,
            group,
            delta_id,
            left_family,
            right_family,
        )
        for (
            delta_id,
            left_family,
            right_family,
        )
        in PAIRWISE_SPECS
    ]


assert len(
    pairwise_deltas[
        "W_SHORT"
    ]
) == 3

assert len(
    pairwise_deltas[
        "W_LONG"
    ]
) == 3


print(
    "M6.5 PAIRWISE SAME-WINDOW DELTA GATE: PASS"
)

```

    M6.5 PAIRWISE SAME-WINDOW DELTA GATE: PASS


## 14. Print W_SHORT cross-model evidence

M6.5 in đủ:

- metric;
- confusion counts;
- alert volume;
- computational context;
- probability summary.

Không in final model winner.


```python

def print_model_record(
    record,
):
    probability = record[
        "probability_summary"
    ]

    print(
        "Model:",
        record[
            "model_family"
        ],
    )

    print(
        "  Experiment:",
        record[
            "experiment_id"
        ],
    )

    print(
        "  F1:",
        record[
            "f1_fraud"
        ],
    )

    print(
        "  Recall:",
        record[
            "recall_fraud"
        ],
    )

    print(
        "  Precision:",
        record[
            "precision_fraud"
        ],
    )

    print(
        "  TP / FP / FN / TN:",
        record["tp"],
        record["fp"],
        record["fn"],
        record["tn"],
    )

    print(
        "  Alerts:",
        record[
            "alert_count"
        ],
        "| rate:",
        record[
            "alert_rate"
        ],
    )

    print(
        "  Fit / predict seconds:",
        record[
            "fit_seconds"
        ],
        "/",
        record[
            "prediction_seconds"
        ],
    )

    print(
        "  Fraud score mean / median:",
        probability[
            "fraud"
        ][
            "mean"
        ],
        "/",
        probability[
            "fraud"
        ][
            "median"
        ],
    )

    print(
        "  Non-fraud score mean / median:",
        probability[
            "non_fraud"
        ][
            "mean"
        ],
        "/",
        probability[
            "non_fraud"
        ][
            "median"
        ],
    )

    print(
        "  Mean score separation:",
        probability[
            "fraud_minus_nonfraud_mean"
        ],
    )

    print()


print(
    "=" * 72
)

print(
    "W_SHORT — LR vs DT vs RF"
)

print(
    "=" * 72
)


for record in (
    same_window_records[
        "W_SHORT"
    ]
):
    print_model_record(
        record
    )


print(
    "W_SHORT descriptive extrema:"
)

for key, value in (
    window_extrema[
        "W_SHORT"
    ].items()
):
    print(
        " ",
        key,
        "→",
        value,
    )


print(
    "\nM6.5 W_SHORT CROSS-MODEL EVIDENCE: READY"
)

```

    ========================================================================
    W_SHORT — LR vs DT vs RF
    ========================================================================
    Model: Logistic Regression
      Experiment: M5-LR-SHORT-B04
      F1: 0.33747547416612167
      Recall: 0.24524714828897337
      Precision: 0.5408805031446541
      TP / FP / FN / TN: 258 219 794 711187
      Alerts: 477 | rate: 0.0006695131502488568
      Fit / predict seconds: 0.8222020840039477 / 0.03208825003821403
      Fraud score mean / median: 0.27809304784108735 / 0.2046126425266266
      Non-fraud score mean / median: 0.0009943300135354998 / 3.985931925853947e-06
      Mean score separation: 0.27709871782755185
    
    Model: Decision Tree
      Experiment: M5-DT-SHORT-B01
      F1: 0.3270564915758176
      Recall: 0.31368821292775667
      Precision: 0.3416149068322981
      TP / FP / FN / TN: 330 636 722 710770
      Alerts: 966 | rate: 0.0013558693986171816
      Fit / predict seconds: 6.661822540976573 / 0.048459709039889276
      Fraud score mean / median: 0.31368821292775667 / 0.0
      Non-fraud score mean / median: 0.0008940042676052775 / 0.0
      Mean score separation: 0.3127942086601514
    
    Model: Random Forest
      Experiment: M5-RF-SHORT-B01
      F1: 0.3664670658682635
      Recall: 0.2908745247148289
      Precision: 0.49514563106796117
      TP / FP / FN / TN: 306 312 746 711094
      Alerts: 618 | rate: 0.0008674195531526069
      Fit / predict seconds: 40.69802879198687 / 0.8680003330227919
      Fraud score mean / median: 0.3034885932156682 / 0.1899999976158142
      Non-fraud score mean / median: 0.0009097336788646354 / 0.0
      Mean score separation: 0.3025788595368035
    
    W_SHORT descriptive extrema:
      highest_f1 → Random Forest
      highest_recall → Decision Tree
      highest_precision → Logistic Regression
      lowest_fn → Decision Tree
      lowest_fp → Logistic Regression
      lowest_alert_count → Logistic Regression
      lowest_fit_seconds → Logistic Regression
      interpretation → DESCRIPTIVE_EXTREMA_NOT_FINAL_SELECTION
    
    M6.5 W_SHORT CROSS-MODEL EVIDENCE: READY


## 15. Print W_LONG cross-model evidence


```python

print(
    "=" * 72
)

print(
    "W_LONG — LR vs DT vs RF"
)

print(
    "=" * 72
)


for record in (
    same_window_records[
        "W_LONG"
    ]
):
    print_model_record(
        record
    )


print(
    "W_LONG descriptive extrema:"
)

for key, value in (
    window_extrema[
        "W_LONG"
    ].items()
):
    print(
        " ",
        key,
        "→",
        value,
    )


print(
    "\nM6.5 W_LONG CROSS-MODEL EVIDENCE: READY"
)

```

    ========================================================================
    W_LONG — LR vs DT vs RF
    ========================================================================
    Model: Logistic Regression
      Experiment: M5-LR-LONG-B04
      F1: 0.03996366939146231
      Recall: 0.02091254752851711
      Precision: 0.4489795918367347
      TP / FP / FN / TN: 22 27 1030 711379
      Alerts: 49 | rate: 6.877598398782806e-05
      Fit / predict seconds: 3.6143539589829743 / 0.024729042022954673
      Fraud score mean / median: 0.12153302745182269 / 0.058560268953442574
      Non-fraud score mean / median: 0.001259920658677669 / 9.025400504469872e-05
      Mean score separation: 0.12027310679314503
    
    Model: Decision Tree
      Experiment: M5-DT-LONG-B01
      F1: 0.19913419913419914
      Recall: 0.1967680608365019
      Precision: 0.20155793573515093
      TP / FP / FN / TN: 207 820 845 710586
      Alerts: 1027 | rate: 0.0014414884807244777
      Fit / predict seconds: 116.46980562497629 / 0.10240820900071412
      Fraud score mean / median: 0.1967680608365019 / 0.0
      Non-fraud score mean / median: 0.0011526470116923389 / 0.0
      Mean score separation: 0.19561541382480954
    
    Model: Random Forest
      Experiment: M5-RF-LONG-B01
      F1: 0.15270935960591134
      Recall: 0.08840304182509506
      Precision: 0.5602409638554217
      TP / FP / FN / TN: 93 73 959 711333
      Alerts: 166 | rate: 0.000232996190652642
      Fit / predict seconds: 627.7140797079774 / 1.4944120000000112
      Fraud score mean / median: 0.17400190110598113 / 0.10000000149011612
      Non-fraud score mean / median: 0.0011014104404785828 / 0.0
      Mean score separation: 0.17290049066550253
    
    W_LONG descriptive extrema:
      highest_f1 → Decision Tree
      highest_recall → Decision Tree
      highest_precision → Random Forest
      lowest_fn → Decision Tree
      lowest_fp → Logistic Regression
      lowest_alert_count → Logistic Regression
      lowest_fit_seconds → Logistic Regression
      interpretation → DESCRIPTIVE_EXTREMA_NOT_FINAL_SELECTION
    
    M6.5 W_LONG CROSS-MODEL EVIDENCE: READY


## 16. Print pairwise deltas


```python

for window_id in [
    "W_SHORT",
    "W_LONG",
]:
    print(
        "=" * 72
    )

    print(
        window_id,
        "PAIRWISE DELTAS",
    )

    for delta in (
        pairwise_deltas[
            window_id
        ]
    ):
        print(
            delta[
                "delta_convention"
            ]
        )

        print(
            "  ΔF1:",
            delta[
                "delta_f1"
            ],
        )

        print(
            "  ΔRecall:",
            delta[
                "delta_recall"
            ],
        )

        print(
            "  ΔPrecision:",
            delta[
                "delta_precision"
            ],
        )

        print(
            "  ΔTP / ΔFP / ΔFN:",
            delta[
                "delta_tp"
            ],
            delta[
                "delta_fp"
            ],
            delta[
                "delta_fn"
            ],
        )

        print(
            "  ΔAlerts:",
            delta[
                "delta_alert_count"
            ],
        )

        print(
            "  ΔFit seconds:",
            delta[
                "delta_fit_seconds"
            ],
        )

        print(
            "  ΔFraud-score mean:",
            delta[
                "delta_fraud_score_mean"
            ],
        )

        print(
            "  ΔNonfraud-score mean:",
            delta[
                "delta_nonfraud_score_mean"
            ],
        )

        print(
            "  ΔMean separation:",
            delta[
                "delta_score_mean_separation"
            ],
        )

        print()


print(
    "M6.5 PAIRWISE DELTA EVIDENCE: READY"
)

```

    ========================================================================
    W_SHORT PAIRWISE DELTAS
    DT_MINUS_LR
      ΔF1: -0.010418982590304049
      ΔRecall: 0.0684410646387833
      ΔPrecision: -0.19926559631235596
      ΔTP / ΔFP / ΔFN: 72 417 -72
      ΔAlerts: 489
      ΔFit seconds: 5.839620456972625
      ΔFraud-score mean: 0.03559516508666932
      ΔNonfraud-score mean: -0.00010032574593022239
      ΔMean separation: 0.03569549083259954
    
    RF_MINUS_LR
      ΔF1: 0.028991591702141828
      ΔRecall: 0.045627376425855515
      ΔPrecision: -0.04573487207669291
      ΔTP / ΔFP / ΔFN: 48 93 -48
      ΔAlerts: 141
      ΔFit seconds: 39.875826707982924
      ΔFraud-score mean: 0.02539554537458083
      ΔNonfraud-score mean: -8.459633467086441e-05
      ΔMean separation: 0.02548014170925167
    
    RF_MINUS_DT
      ΔF1: 0.03941057429244588
      ΔRecall: -0.022813688212927785
      ΔPrecision: 0.15353072423566305
      ΔTP / ΔFP / ΔFN: -24 -324 24
      ΔAlerts: -348
      ΔFit seconds: 34.0362062510103
      ΔFraud-score mean: -0.010199619712088492
      ΔNonfraud-score mean: 1.5729411259357974e-05
      ΔMean separation: -0.010215349123347872
    
    ========================================================================
    W_LONG PAIRWISE DELTAS
    DT_MINUS_LR
      ΔF1: 0.15917052974273682
      ΔRecall: 0.17585551330798477
      ΔPrecision: -0.24742165610158376
      ΔTP / ΔFP / ΔFN: 185 793 -185
      ΔAlerts: 978
      ΔFit seconds: 112.85545166599331
      ΔFraud-score mean: 0.0752350333846792
      ΔNonfraud-score mean: -0.00010727364698533007
      ΔMean separation: 0.07534230703166452
    
    RF_MINUS_LR
      ΔF1: 0.11274569021444902
      ΔRecall: 0.06749049429657795
      ΔPrecision: 0.11126137201868697
      ΔTP / ΔFP / ΔFN: 71 46 -71
      ΔAlerts: 117
      ΔFit seconds: 624.0997257489944
      ΔFraud-score mean: 0.052468873654158435
      ΔNonfraud-score mean: -0.00015851021819908613
      ΔMean separation: 0.052627383872357505
    
    RF_MINUS_DT
      ΔF1: -0.0464248395282878
      ΔRecall: -0.10836501901140683
      ΔPrecision: 0.35868302812027075
      ΔTP / ΔFP / ΔFN: -114 -747 114
      ΔAlerts: -861
      ΔFit seconds: 511.2442740830011
      ΔFraud-score mean: -0.022766159730520763
      ΔNonfraud-score mean: -5.123657121375606e-05
      ΔMean separation: -0.02271492315930701
    
    M6.5 PAIRWISE DELTA EVIDENCE: READY


## 17. Probability-behavior sanity checks

Probability comparison chỉ hợp lệ nếu:

```text
6 / 6 risk-score hashes match M6.2 registry
6 / 6 arrays finite
6 / 6 arrays in [0,1]
same y_validation order
no threshold search
```

M6.5 không đánh giá calibration quality chính thức.


```python

for experiment_id, summary in (
    probability_summaries.items()
):
    assert (
        summary[
            "risk_score_kind"
        ]
        == "predict_proba"
    )

    assert (
        summary[
            "threshold_search_performed"
        ]
        is False
    )

    assert (
        summary[
            "final_test_accessed"
        ]
        is False
    )

    for group_name in [
        "overall",
        "fraud",
        "non_fraud",
    ]:
        group = summary[
            group_name
        ]

        assert (
            0.0
            <= group["min"]
            <= 1.0
        )

        assert (
            0.0
            <= group["max"]
            <= 1.0
        )


print(
    "M6.5 PROBABILITY-BEHAVIOR SANITY GATE: PASS"
)

```

    M6.5 PROBABILITY-BEHAVIOR SANITY GATE: PASS


## 18. Selection boundary

M6.5 có thể nói:

```text
Trong W_SHORT,
model X có highest F1.

Trong W_LONG,
model Y có highest Recall.
```

M6.5 không được nói:

```text
Final model = X
```

vì final selection thuộc M7.

M6.5 cũng không dùng một same-window extremum để override M6.4 training-window evidence.


```python

selection_state = {
    "training_window_winner":
        "OPEN",

    "model_family_winner":
        "OPEN",

    "final_model":
        "OPEN",

    "final_threshold":
        "OPEN",

    "cross_model_final_selection_authorized":
        False,

    "threshold_optimization_performed":
        False,

    "mixed_window_model_comparison_used":
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
        "cross_model_final_selection_authorized"
    ]
    is False
)

assert (
    selection_state[
        "threshold_optimization_performed"
    ]
    is False
)

assert (
    selection_state[
        "mixed_window_model_comparison_used"
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
    "\nM6.5 SELECTION-BOUNDARY GATE: PASS"
)

```

    Training-window winner: OPEN
    Model-family winner: OPEN
    Final model: OPEN
    
    M6.5 SELECTION-BOUNDARY GATE: PASS


## 19. Source-artifact fingerprints


```python

m6_02_registry_sha256 = (
    sha256_file(
        M6_02_REGISTRY_PATH
    )
)

m6_03_analysis_sha256 = (
    sha256_file(
        M6_03_ANALYSIS_PATH
    )
)

m6_04_comparison_sha256 = (
    sha256_file(
        M6_04_COMPARISON_PATH
    )
)


print(
    "M6.2 registry SHA256:"
)
print(
    m6_02_registry_sha256
)

print(
    "\nM6.3 analysis SHA256:"
)
print(
    m6_03_analysis_sha256
)

print(
    "\nM6.4 comparison SHA256:"
)
print(
    m6_04_comparison_sha256
)

```

    M6.2 registry SHA256:
    9e112883d5aa1358c58c714edc0f0ef7913252e94c56f3c6b28d3ba1b7f4e6fb
    
    M6.3 analysis SHA256:
    290f348ba20a4a19f659e24bb517ca8cce2229605af3f7ee762520e5b1d7b168
    
    M6.4 comparison SHA256:
    0ca60389901a5c619ddf92913b4fdfeefbb99a6ba6ccee9cecb717235452b5c6


## 20. Persist M6.5 cross-model artifact


```python

COMPARISON_VERSION = (
    "M6.5-cross-model-comparison-v1"
)

COMPARISON_PATH = (
    OUTPUT_DIR
    / "m6_05_cross_model_comparison.json"
)

MANIFEST_PATH = (
    OUTPUT_DIR
    / "m6_05_comparison_manifest.json"
)


comparison_payload = {
    "comparison_version":
        COMPARISON_VERSION,

    "m6_substep":
        "M6.5",

    "evaluation_population":
        "VALIDATION_2019-01_TO_2019-05",

    "validation_rows":
        EXPECTED_VALIDATION_ROWS,

    "validation_fraud_rows":
        EXPECTED_VALIDATION_FRAUD,

    "primary_groups": {
        "W_SHORT":
            GROUP_SPECS[
                "W_SHORT"
            ],

        "W_LONG":
            GROUP_SPECS[
                "W_LONG"
            ],
    },

    "primary_metric":
        "F1_fraud",

    "secondary_metrics": [
        "Recall_fraud",
        "Precision_fraud",
    ],

    "source_artifacts": {
        "m6_02_registry":
            str(
                M6_02_REGISTRY_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),

        "m6_02_registry_sha256":
            m6_02_registry_sha256,

        "m6_03_analysis":
            str(
                M6_03_ANALYSIS_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),

        "m6_03_analysis_sha256":
            m6_03_analysis_sha256,

        "m6_04_comparison":
            str(
                M6_04_COMPARISON_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),

        "m6_04_comparison_sha256":
            m6_04_comparison_sha256,
    },

    "group_integrity":
        group_integrity,

    "same_window_records":
        same_window_records,

    "window_extrema":
        window_extrema,

    "pairwise_deltas":
        pairwise_deltas,

    "selection_state":
        selection_state,

    "probability_analysis_scope":
        "DESCRIPTIVE_ONLY",

    "final_test_accessed":
        False,
}


comparison_manifest = {
    "m6_substep":
        "M6.5",

    "comparison_version":
        COMPARISON_VERSION,

    "same_window_groups_expected":
        2,

    "same_window_groups_built":
        2,

    "models_per_group":
        3,

    "same_window_comparability":
        "PASS",

    "probability_artifacts_verified":
        6,

    "probability_analysis_scope":
        "DESCRIPTIVE_ONLY",

    "threshold_search":
        "NOT_PERFORMED",

    "mixed_window_model_comparison":
        "NOT_PERFORMED",

    "final_model_selected":
        False,

    "training_window_selected":
        False,

    "final_test_accessed":
        False,

    "decision":
        "OPEN — REQUIRES M6.5 RUNTIME REVIEW",
}


with open(
    COMPARISON_PATH,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        comparison_payload,
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
        comparison_manifest,
        file,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )


assert COMPARISON_PATH.exists()
assert MANIFEST_PATH.exists()

assert COMPARISON_PATH.stat().st_size > 0
assert MANIFEST_PATH.stat().st_size > 0


print("Cross-model artifact:")
print(COMPARISON_PATH)

print("\nComparison manifest:")
print(MANIFEST_PATH)

print(
    "\nM6.5 COMPARISON PERSISTENCE GATE: PASS"
)

```

    Cross-model artifact:
    /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/processed/m6_05_cross_model_comparative_evaluation/m6_05_cross_model_comparison.json
    
    Comparison manifest:
    /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/processed/m6_05_cross_model_comparative_evaluation/m6_05_comparison_manifest.json
    
    M6.5 COMPARISON PERSISTENCE GATE: PASS


## 21. Persisted artifact round-trip


```python

with open(
    COMPARISON_PATH,
    "r",
    encoding="utf-8",
) as file:
    comparison_roundtrip = json.load(
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
    comparison_roundtrip[
        "comparison_version"
    ]
    == COMPARISON_VERSION
)

assert (
    set(
        comparison_roundtrip[
            "same_window_records"
        ]
    )
    == {
        "W_SHORT",
        "W_LONG",
    }
)

assert (
    len(
        comparison_roundtrip[
            "same_window_records"
        ][
            "W_SHORT"
        ]
    )
    == 3
)

assert (
    len(
        comparison_roundtrip[
            "same_window_records"
        ][
            "W_LONG"
        ]
    )
    == 3
)

assert (
    comparison_roundtrip[
        "selection_state"
    ][
        "final_model"
    ]
    == "OPEN"
)

assert (
    comparison_roundtrip[
        "selection_state"
    ][
        "mixed_window_model_comparison_used"
    ]
    is False
)

assert (
    comparison_roundtrip[
        "final_test_accessed"
    ]
    is False
)

assert (
    manifest_roundtrip[
        "same_window_groups_built"
    ]
    == 2
)

assert (
    manifest_roundtrip[
        "probability_artifacts_verified"
    ]
    == 6
)

assert (
    manifest_roundtrip[
        "threshold_search"
    ]
    == "NOT_PERFORMED"
)

assert (
    manifest_roundtrip[
        "final_model_selected"
    ]
    is False
)


print(
    "M6.5 COMPARISON ROUND-TRIP GATE: PASS"
)

```

    M6.5 COMPARISON ROUND-TRIP GATE: PASS


## 22. FINAL TEST isolation


```python

assert (
    m6_02_registry[
        "final_test_accessed"
    ]
    is False
)

assert (
    m6_03_analysis[
        "final_test_accessed"
    ]
    is False
)

assert (
    m6_04_comparison[
        "final_test_accessed"
    ]
    is False
)


for window_id, group in (
    same_window_records.items()
):
    for record in group:
        assert (
            record[
                "final_test_accessed"
            ]
            is False
        )


assert (
    comparison_payload[
        "final_test_accessed"
    ]
    is False
)

assert (
    comparison_manifest[
        "final_test_accessed"
    ]
    is False
)


for path in [
    Y_VALIDATION_PATH,
    M6_02_REGISTRY_PATH,
    M6_03_ANALYSIS_PATH,
    M6_04_COMPARISON_PATH,
    M6_04_MANIFEST_PATH,
]:
    assert (
        "final_test"
        not in path.name.lower()
    )


print(
    "M6.5 FINAL TEST ISOLATION GATE: PASS"
)

```

    M6.5 FINAL TEST ISOLATION GATE: PASS


## 23. M6.5 overall technical gate


```python

m6_05_gates = {
    "G01_UPSTREAM_HANDOFF":
        True,

    "G02_VALIDATION_TARGET":
        True,

    "G03_SAME_WINDOW_GROUPS":
        True,

    "G04_SAME_WINDOW_COMPARABILITY":
        True,

    "G05_COMPUTATIONAL_CONTEXT":
        True,

    "G06_PROBABILITY_SUMMARIES":
        True,

    "G07_MODEL_RECORDS":
        True,

    "G08_METRIC_SPECIFIC_EXTREMA":
        True,

    "G09_PAIRWISE_SAME_WINDOW_DELTAS":
        True,

    "G10_W_SHORT_EVIDENCE":
        True,

    "G11_W_LONG_EVIDENCE":
        True,

    "G12_PROBABILITY_SANITY":
        True,

    "G13_SELECTION_BOUNDARY":
        True,

    "G14_SOURCE_LINEAGE":
        True,

    "G15_COMPARISON_PERSISTENCE":
        True,

    "G16_COMPARISON_ROUND_TRIP":
        True,

    "G17_FINAL_TEST_ISOLATION":
        True,
}


for gate_name, gate_value in (
    m6_05_gates.items()
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
    m6_05_gates.values()
)


print(
    "\nM6.5 OVERALL TECHNICAL GATE: PASS"
)

```

    G01_UPSTREAM_HANDOFF → PASS
    G02_VALIDATION_TARGET → PASS
    G03_SAME_WINDOW_GROUPS → PASS
    G04_SAME_WINDOW_COMPARABILITY → PASS
    G05_COMPUTATIONAL_CONTEXT → PASS
    G06_PROBABILITY_SUMMARIES → PASS
    G07_MODEL_RECORDS → PASS
    G08_METRIC_SPECIFIC_EXTREMA → PASS
    G09_PAIRWISE_SAME_WINDOW_DELTAS → PASS
    G10_W_SHORT_EVIDENCE → PASS
    G11_W_LONG_EVIDENCE → PASS
    G12_PROBABILITY_SANITY → PASS
    G13_SELECTION_BOUNDARY → PASS
    G14_SOURCE_LINEAGE → PASS
    G15_COMPARISON_PERSISTENCE → PASS
    G16_COMPARISON_ROUND_TRIP → PASS
    G17_FINAL_TEST_ISOLATION → PASS
    
    M6.5 OVERALL TECHNICAL GATE: PASS


# 24. Runtime review và Findings M6.5

## 24.1. Execution integrity

Observed:

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

Interpretation:

Notebook đã chạy đầy đủ từ đầu đến cuối.

Không có partial execution hoặc runtime exception.

Status:

`VERIFIED`

---

## 24.2. Upstream handoff

Observed:

```text
M6.2 Evaluation Registry:
loaded

M6.3 metric/confusion analysis:
loaded

M6.4 training-window comparison:
loaded

M6.5 UPSTREAM HANDOFF GATE:
PASS
```

M6.5 dùng persisted evidence đã verify, không fit hoặc predict lại model.

Status:

`VERIFIED`

---

## 24.3. Canonical validation target

Observed:

```text
Rows:
712,458

Fraud:
1,052

Positive class:
fraud = 1
```

`y_validation.npy` được kiểm tra lại bằng SHA-256 trước khi dùng cho class-conditioned probability summaries.

Runtime gate:

`M6.5 VALIDATION TARGET GATE: PASS`

Status:

`VERIFIED`

---

## 24.4. Same-window comparison groups

Observed:

```text
W_SHORT:
M5-LR-SHORT-B04
M5-DT-SHORT-B01
M5-RF-SHORT-B01

W_LONG:
M5-LR-LONG-B04
M5-DT-LONG-B01
M5-RF-LONG-B01
```

Không có mixed-window group.

Runtime gate:

`M6.5 SAME-WINDOW GROUP GATE: PASS`

Status:

`VERIFIED`

---

## 24.5. Same-window comparability

Observed:

```text
W_SHORT:
PASS

W_LONG:
PASS

Controlled fields checked:
10 per group
```

Các field được giữ cố định trong từng window:

- training_window_id;
- feature version;
- preprocessing version;
- matrix schema;
- validation period;
- validation rows;
- validation fraud rows;
- imbalance strategy;
- threshold policy;
- risk-score kind.

Variable được phép khác:

`model family / model config`

Runtime gate:

`M6.5 SAME-WINDOW COMPARABILITY GATE: PASS`

Status:

`VERIFIED`

---

## 24.6. W_SHORT — cross-model evidence

### Logistic Regression

Observed:

```text
F1:
0.33747547416612167

Recall:
0.24524714828897337

Precision:
0.5408805031446541

TP / FP / FN / TN:
258 / 219 / 794 / 711187

Alerts:
477

Alert rate:
0.0006695131502488568

Fit / predict seconds:
0.8222020840039477
/
0.03208825003821403
```

Probability summary:

```text
Fraud score mean:
0.27809304784108735

Fraud score median:
0.2046126425266266

Non-fraud score mean:
0.0009943300135354998

Non-fraud score median:
0.000003985931925853947

Mean score separation:
0.27709871782755185
```

---

### Decision Tree

Observed:

```text
F1:
0.3270564915758176

Recall:
0.31368821292775667

Precision:
0.3416149068322981

TP / FP / FN / TN:
330 / 636 / 722 / 710770

Alerts:
966

Alert rate:
0.0013558693986171816

Fit / predict seconds:
6.661822540976573
/
0.048459709039889276
```

Probability summary:

```text
Fraud score mean:
0.31368821292775667

Fraud score median:
0.0

Non-fraud score mean:
0.0008940042676052775

Non-fraud score median:
0.0

Mean score separation:
0.3127942086601514
```

---

### Random Forest

Observed:

```text
F1:
0.3664670658682635

Recall:
0.2908745247148289

Precision:
0.49514563106796117

TP / FP / FN / TN:
306 / 312 / 746 / 711094

Alerts:
618

Alert rate:
0.0008674195531526069

Fit / predict seconds:
40.69802879198687
/
0.8680003330227919
```

Probability summary:

```text
Fraud score mean:
0.3034885932156682

Fraud score median:
0.1899999976158142

Non-fraud score mean:
0.0009097336788646354

Non-fraud score median:
0.0

Mean score separation:
0.3025788595368035
```

Descriptive extrema:

```text
Highest F1:
Random Forest

Highest Recall:
Decision Tree

Highest Precision:
Logistic Regression

Lowest FN:
Decision Tree

Lowest FP:
Logistic Regression

Lowest alert count:
Logistic Regression

Lowest fit time:
Logistic Regression
```

Interpretation:

Không có một W_SHORT model cùng lúc dẫn toàn bộ tiêu chí.

W_SHORT thể hiện trade-off rõ:

- RF có F1 cao nhất;
- DT có Recall cao nhất và FN thấp nhất;
- LR có Precision cao nhất, FP thấp nhất, alert count thấp nhất và fit time thấp nhất.

Status:

`VERIFIED SAME-WINDOW TRADE-OFF`

---

## 24.7. W_SHORT — pairwise deltas

### DT − LR

Observed:

```text
ΔF1:
-0.010418982590304049

ΔRecall:
+0.0684410646387833

ΔPrecision:
-0.19926559631235596

ΔTP / ΔFP / ΔFN:
+72 / +417 / -72

ΔAlerts:
+489

ΔFit seconds:
+5.839620456972625

ΔMean score separation:
+0.03569549083259954
```

Interpretation:

So với LR-SHORT, DT-SHORT bắt thêm 72 fraud và giảm 72 FN, nhưng tạo thêm 417 FP và Precision thấp hơn đáng kể.

---

### RF − LR

Observed:

```text
ΔF1:
+0.028991591702141828

ΔRecall:
+0.045627376425855515

ΔPrecision:
-0.04573487207669291

ΔTP / ΔFP / ΔFN:
+48 / +93 / -48

ΔAlerts:
+141

ΔFit seconds:
+39.875826707982924

ΔMean score separation:
+0.02548014170925167
```

Interpretation:

RF-SHORT có F1 và Recall cao hơn LR-SHORT, nhưng Precision thấp hơn và tạo nhiều FP/alerts hơn.

---

### RF − DT

Observed:

```text
ΔF1:
+0.03941057429244588

ΔRecall:
-0.022813688212927785

ΔPrecision:
+0.15353072423566305

ΔTP / ΔFP / ΔFN:
-24 / -324 / +24

ΔAlerts:
-348

ΔFit seconds:
+34.0362062510103

ΔMean score separation:
-0.010215349123347872
```

Interpretation:

RF-SHORT có F1/Precision cao hơn DT-SHORT và ít hơn 324 FP, nhưng Recall thấp hơn và bỏ sót thêm 24 fraud.

Status:

`VERIFIED PAIRWISE EVIDENCE`

---

## 24.8. W_LONG — cross-model evidence

### Logistic Regression

Observed:

```text
F1:
0.03996366939146231

Recall:
0.02091254752851711

Precision:
0.4489795918367347

TP / FP / FN / TN:
22 / 27 / 1030 / 711379

Alerts:
49

Alert rate:
0.00006877598398782806

Fit / predict seconds:
3.6143539589829743
/
0.024729042022954673
```

Probability summary:

```text
Fraud score mean:
0.12153302745182269

Fraud score median:
0.058560268953442574

Non-fraud score mean:
0.001259920658677669

Non-fraud score median:
0.00009025400504469872

Mean score separation:
0.12027310679314503
```

---

### Decision Tree

Observed:

```text
F1:
0.19913419913419914

Recall:
0.1967680608365019

Precision:
0.20155793573515093

TP / FP / FN / TN:
207 / 820 / 845 / 710586

Alerts:
1,027

Alert rate:
0.0014414884807244777

Fit / predict seconds:
116.46980562497629
/
0.10240820900071412
```

Probability summary:

```text
Fraud score mean:
0.1967680608365019

Fraud score median:
0.0

Non-fraud score mean:
0.0011526470116923389

Non-fraud score median:
0.0

Mean score separation:
0.19561541382480954
```

---

### Random Forest

Observed:

```text
F1:
0.15270935960591134

Recall:
0.08840304182509506

Precision:
0.5602409638554217

TP / FP / FN / TN:
93 / 73 / 959 / 711333

Alerts:
166

Alert rate:
0.000232996190652642

Fit / predict seconds:
627.7140797079774
/
1.4944120000000112
```

Probability summary:

```text
Fraud score mean:
0.17400190110598113

Fraud score median:
0.10000000149011612

Non-fraud score mean:
0.0011014104404785828

Non-fraud score median:
0.0

Mean score separation:
0.17290049066550253
```

Descriptive extrema:

```text
Highest F1:
Decision Tree

Highest Recall:
Decision Tree

Highest Precision:
Random Forest

Lowest FN:
Decision Tree

Lowest FP:
Logistic Regression

Lowest alert count:
Logistic Regression

Lowest fit time:
Logistic Regression
```

Interpretation:

W_LONG cũng không có một model dẫn toàn bộ tiêu chí.

- DT có F1/Recall cao nhất và FN thấp nhất;
- RF có Precision cao nhất;
- LR có FP thấp nhất và alert volume thấp nhất nhưng Recall rất thấp.

Status:

`VERIFIED SAME-WINDOW TRADE-OFF`

---

## 24.9. W_LONG — pairwise deltas

### DT − LR

Observed:

```text
ΔF1:
+0.15917052974273682

ΔRecall:
+0.17585551330798477

ΔPrecision:
-0.24742165610158376

ΔTP / ΔFP / ΔFN:
+185 / +793 / -185

ΔAlerts:
+978

ΔFit seconds:
+112.85545166599331

ΔMean score separation:
+0.07534230703166452
```

Interpretation:

DT-LONG bắt thêm 185 fraud so với LR-LONG nhưng tạo thêm 793 FP và Precision thấp hơn đáng kể.

---

### RF − LR

Observed:

```text
ΔF1:
+0.11274569021444902

ΔRecall:
+0.06749049429657795

ΔPrecision:
+0.11126137201868697

ΔTP / ΔFP / ΔFN:
+71 / +46 / -71

ΔAlerts:
+117

ΔFit seconds:
+624.0997257489944

ΔMean score separation:
+0.052627383872357505
```

Interpretation:

RF-LONG cao hơn LR-LONG về cả F1, Recall và Precision, nhưng computational cost lớn hơn rất nhiều.

Đây vẫn chỉ là same-window descriptive evidence, không phải final selection.

---

### RF − DT

Observed:

```text
ΔF1:
-0.0464248395282878

ΔRecall:
-0.10836501901140683

ΔPrecision:
+0.35868302812027075

ΔTP / ΔFP / ΔFN:
-114 / -747 / +114

ΔAlerts:
-861

ΔFit seconds:
+511.2442740830011

ΔMean score separation:
-0.02271492315930701
```

Interpretation:

So với DT-LONG, RF-LONG có Precision cao hơn rất mạnh và ít hơn 747 FP, nhưng F1/Recall thấp hơn và bỏ sót thêm 114 fraud.

Status:

`VERIFIED PAIRWISE EVIDENCE`

---

## 24.10. Probability behavior

Observed runtime gate:

`M6.5 PROBABILITY-BEHAVIOR SANITY GATE: PASS`

Verified:

```text
6 / 6 risk-score artifacts:
hash matched

finite:
YES

range:
[0,1]

same validation target/order:
YES

threshold search:
NO
```

Descriptive mean score separation:

```text
W_SHORT
LR:
0.27709871782755185

DT:
0.3127942086601514

RF:
0.3025788595368035

W_LONG
LR:
0.12027310679314503

DT:
0.19561541382480954

RF:
0.17290049066550253
```

Interpretation boundary:

Các score statistics này mô tả risk-score behavior.

Không được diễn giải chúng như:

- calibration quality;
- threshold quality;
- final model ranking.

Đặc biệt Decision Tree có fraud-score median = 0 trong cả W_SHORT và W_LONG, cho thấy score distribution của baseline tree cần được đọc cẩn thận hơn ở error/probability analysis kế tiếp.

Status:

`DESCRIPTIVE PROBABILITY EVIDENCE VERIFIED`

---

## 24.11. Computational evidence

Observed:

```text
W_SHORT fit seconds

LR:
0.8222020840039477

DT:
6.661822540976573

RF:
40.69802879198687
```

```text
W_LONG fit seconds

LR:
3.6143539589829743

DT:
116.46980562497629

RF:
627.7140797079774
```

Interpretation:

Computational cost khác nhau rõ giữa model families.

Đây là supporting evidence.

Không dùng runtime làm automatic final-model rule.

Status:

`VERIFIED`

---

## 24.12. Mixed-window comparison guardrail

Observed:

```text
mixed_window_model_comparison_used:
False
```

M6.5 chỉ dùng:

```text
W_SHORT:
LR vs DT vs RF

W_LONG:
LR vs DT vs RF
```

Status:

`VERIFIED`

---

## 24.13. Selection boundary

Observed:

```text
Training-window winner:
OPEN

Model-family winner:
OPEN

Final model:
OPEN

Final threshold:
OPEN

Cross-model final selection authorized:
FALSE

Threshold optimization performed:
FALSE
```

Runtime gate:

`M6.5 SELECTION-BOUNDARY GATE: PASS`

Status:

`VERIFIED`

---

## 24.14. Persistence / lineage

Persisted:

```text
data/processed/m6_05_cross_model_comparative_evaluation/
    m6_05_cross_model_comparison.json
    m6_05_comparison_manifest.json
```

Runtime gates:

```text
M6.5 COMPARISON PERSISTENCE GATE:
PASS

M6.5 COMPARISON ROUND-TRIP GATE:
PASS
```

Recorded upstream hashes:

```text
M6.2 registry SHA256:
9e112883d5aa1358c58c714edc0f0ef7913252e94c56f3c6b28d3ba1b7f4e6fb

M6.3 analysis SHA256:
290f348ba20a4a19f659e24bb517ca8cce2229605af3f7ee762520e5b1d7b168

M6.4 comparison SHA256:
0ca60389901a5c619ddf92913b4fdfeefbb99a6ba6ccee9cecb717235452b5c6
```

Status:

`VERIFIED`

---

## 24.15. FINAL TEST isolation

Observed:

`M6.5 FINAL TEST ISOLATION GATE: PASS`

No FINAL TEST artifact was used.

Status:

`VERIFIED`

---

## 24.16. Overall technical result

Observed:

```text
G01_UPSTREAM_HANDOFF              → PASS
G02_VALIDATION_TARGET             → PASS
G03_SAME_WINDOW_GROUPS            → PASS
G04_SAME_WINDOW_COMPARABILITY     → PASS
G05_COMPUTATIONAL_CONTEXT         → PASS
G06_PROBABILITY_SUMMARIES         → PASS
G07_MODEL_RECORDS                 → PASS
G08_METRIC_SPECIFIC_EXTREMA       → PASS
G09_PAIRWISE_SAME_WINDOW_DELTAS   → PASS
G10_W_SHORT_EVIDENCE              → PASS
G11_W_LONG_EVIDENCE               → PASS
G12_PROBABILITY_SANITY            → PASS
G13_SELECTION_BOUNDARY            → PASS
G14_SOURCE_LINEAGE                → PASS
G15_COMPARISON_PERSISTENCE        → PASS
G16_COMPARISON_ROUND_TRIP         → PASS
G17_FINAL_TEST_ISOLATION          → PASS
```

Overall:

`M6.5 OVERALL TECHNICAL GATE: PASS`

Blocking issue:

`NONE`

---

# 25. Findings M6.5

## M6.5-F01 — Both same-window comparison groups are valid

Evidence:

```text
W_SHORT:
PASS

W_LONG:
PASS
```

Status:

`VERIFIED`

---

## M6.5-F02 — W_SHORT has no single model dominating all fraud-screening criteria

Observed:

```text
Highest F1:
Random Forest

Highest Recall:
Decision Tree

Highest Precision:
Logistic Regression
```

Status:

`VERIFIED TRADE-OFF`

---

## M6.5-F03 — RF-SHORT improves F1 over LR-SHORT with a Precision cost

Evidence:

```text
RF - LR

ΔF1:
+0.028991591702141828

ΔRecall:
+0.045627376425855515

ΔPrecision:
-0.04573487207669291

ΔTP:
+48

ΔFP:
+93
```

Status:

`DESCRIPTIVE COMPARATIVE EVIDENCE`

---

## M6.5-F04 — RF-SHORT improves F1 and Precision over DT-SHORT with a small Recall cost

Evidence:

```text
RF - DT

ΔF1:
+0.03941057429244588

ΔRecall:
-0.022813688212927785

ΔPrecision:
+0.15353072423566305

ΔTP:
-24

ΔFP:
-324
```

Status:

`DESCRIPTIVE COMPARATIVE EVIDENCE`

---

## M6.5-F05 — DT-SHORT maximizes fraud capture but generates the most false alerts

Observed:

```text
TP:
330

FN:
722

FP:
636

Alerts:
966
```

Status:

`DESCRIPTIVE COMPARATIVE EVIDENCE`

---

## M6.5-F06 — LR-SHORT is the most conservative W_SHORT baseline

Observed:

```text
Highest Precision:
LR

Lowest FP:
LR

Lowest alerts:
LR

Lowest fit time:
LR
```

This conservatism is accompanied by lower Recall than DT/RF.

Status:

`DESCRIPTIVE COMPARATIVE EVIDENCE`

---

## M6.5-F07 — W_LONG also has no single model dominating all criteria

Observed:

```text
Highest F1:
Decision Tree

Highest Recall:
Decision Tree

Highest Precision:
Random Forest

Lowest FP:
Logistic Regression
```

Status:

`VERIFIED TRADE-OFF`

---

## M6.5-F08 — DT-LONG gains fraud capture over LR-LONG at large false-alert cost

Evidence:

```text
DT - LR

ΔTP:
+185

ΔFP:
+793

ΔFN:
-185

ΔPrecision:
-0.24742165610158376
```

Status:

`DESCRIPTIVE COMPARATIVE EVIDENCE`

---

## M6.5-F09 — RF-LONG exceeds LR-LONG on F1, Recall and Precision

Evidence:

```text
RF - LR

ΔF1:
+0.11274569021444902

ΔRecall:
+0.06749049429657795

ΔPrecision:
+0.11126137201868697
```

Computational cost is much larger, so this is not converted into final selection.

Status:

`DESCRIPTIVE COMPARATIVE EVIDENCE`

---

## M6.5-F10 — RF-LONG vs DT-LONG exposes a strong Precision/Recall trade-off

Evidence:

```text
RF - DT

ΔF1:
-0.0464248395282878

ΔRecall:
-0.10836501901140683

ΔPrecision:
+0.35868302812027075

ΔFP:
-747

ΔFN:
+114
```

Status:

`DESCRIPTIVE COMPARATIVE EVIDENCE`

---

## M6.5-F11 — Probability behavior differs materially across models/windows

Evidence:

Class-conditioned risk-score summaries and mean separation differ across all six runs.

Status:

`DESCRIPTIVE PROBABILITY EVIDENCE`

Boundary:

`NOT CALIBRATION / NOT THRESHOLD SELECTION`

---

## M6.5-F12 — Decision Tree probability output needs careful downstream interpretation

Observed:

```text
DT-SHORT fraud-score median:
0.0

DT-LONG fraud-score median:
0.0
```

while fraud-score means remain positive.

Finding:

The score distribution is sufficiently non-smooth/skewed that M6.6 should avoid treating mean probability alone as a calibrated confidence measure.

Status:

`M6.6 FOLLOW-UP EVIDENCE`

---

## M6.5-F13 — Computational cost is strongly model-family dependent

Observed:

LR is fastest to fit in both same-window groups, DT is intermediate, RF is most expensive among these baselines.

Status:

`VERIFIED COMPUTATIONAL EVIDENCE`

---

## M6.5-F14 — No mixed-window model-family conclusion was used

Status:

`VERIFIED`

---

## M6.5-F15 — M6.5 does not establish a final model winner

Finding:

Cross-model evidence is sufficient to describe trade-offs but not to perform final M7 model selection.

Status:

`CORRECT BOUNDARY`

---

## M6.5-F16 — M6.6 handoff is unblocked

M6.6 can now move from aggregate comparison to transaction-level:

```text
Fraud side:
FN vs TP

Non-fraud side:
FP vs TN
```

plus descriptive score behavior.

Status:

`READY FOR M6.6`

# 26. Decision Log M6.5 — sau runtime review

## M6.5-D01 — Comparison scope

Decision:

Model-family comparison giữ training window cố định.

Observed:

same-window comparability PASS for both groups.

Status:

`INHERITED — VERIFIED — LOCKED`

---

## M6.5-D02 — W_SHORT group

Decision:

```text
LR-SHORT
DT-SHORT
RF-SHORT
```

Status:

`LOCKED / VERIFIED`

---

## M6.5-D03 — W_LONG group

Decision:

```text
LR-LONG
DT-LONG
RF-LONG
```

Status:

`LOCKED / VERIFIED`

---

## M6.5-D04 — Primary metric

Decision:

`F1_fraud`

Status:

`INHERITED — LOCKED`

---

## M6.5-D05 — Secondary evidence

Decision:

Read with F1:

```text
Recall
Precision
TP / FP / FN / TN
alert count/rate
```

Observed:

complete for all six runs.

Status:

`INHERITED — VERIFIED — LOCKED`

---

## M6.5-D06 — W_SHORT descriptive extrema

Observed:

```text
Highest F1:
Random Forest

Highest Recall:
Decision Tree

Highest Precision:
Logistic Regression

Lowest FN:
Decision Tree

Lowest FP:
Logistic Regression

Lowest alert count:
Logistic Regression

Lowest fit time:
Logistic Regression
```

Status:

`DESCRIPTIVE EVIDENCE — NOT FINAL RANKING`

---

## M6.5-D07 — W_LONG descriptive extrema

Observed:

```text
Highest F1:
Decision Tree

Highest Recall:
Decision Tree

Highest Precision:
Random Forest

Lowest FN:
Decision Tree

Lowest FP:
Logistic Regression

Lowest alert count:
Logistic Regression

Lowest fit time:
Logistic Regression
```

Status:

`DESCRIPTIVE EVIDENCE — NOT FINAL RANKING`

---

## M6.5-D08 — Probability behavior

Decision:

Use class-conditioned persisted `predict_proba` summaries descriptively.

Observed:

6 / 6 artifacts verified.

Threshold search:

`NO`

Calibration claim:

`NO`

Status:

`LOCKED / VERIFIED`

---

## M6.5-D09 — Computational context

Decision:

Read fit/prediction runtime as supporting evidence only.

Status:

`LOCKED / VERIFIED`

---

## M6.5-D10 — Pairwise deltas

Decision:

Use:

```text
DT - LR
RF - LR
RF - DT
```

inside W_SHORT and W_LONG separately.

Observed:

all six pairwise delta records produced.

Status:

`LOCKED / VERIFIED`

---

## M6.5-D11 — Mixed-window model comparison

Decision:

Do not use mixed-window comparison for model-family conclusions.

Observed:

```text
mixed_window_model_comparison_used:
False
```

Status:

`LOCKED / VERIFIED`

---

## M6.5-D12 — Final model

Decision:

Not selected in M6.5.

Status:

`OPEN — M7`

---

## M6.5-D13 — Model-family winner

Decision:

Not locked in M6.5.

Status:

`OPEN — M7`

---

## M6.5-D14 — Final training window

Decision:

Not selected in M6.5.

Status:

`OPEN — M7 IF ROBUSTNESS / SELECTION REQUIRED`

---

## M6.5-D15 — Final threshold

Decision:

No threshold optimization.

Status:

`OPEN — M7`

---

## M6.5-D16 — Cross-model artifact

Decision:

Persist:

`m6_05_cross_model_comparison.json`

Observed:

persistence + round-trip PASS.

Status:

`LOCKED / VERIFIED`

---

## M6.5-D17 — Comparison manifest

Decision:

Persist:

`m6_05_comparison_manifest.json`

Observed:

persistence + round-trip PASS.

Status:

`LOCKED / VERIFIED`

---

## M6.5-D18 — FINAL TEST

Decision:

No access.

Observed:

FINAL TEST isolation PASS.

Status:

`INHERITED — VERIFIED — LOCKED`

---

## M6.5-D19 — M6.6 handoff

Decision:

Proceed to transaction-level FP/FN error analysis and descriptive probability context.

Status:

`READY`

# 27. M6.5 Gate

## Technical runtime gates

```text
G01_UPSTREAM_HANDOFF              → PASS
G02_VALIDATION_TARGET             → PASS
G03_SAME_WINDOW_GROUPS            → PASS
G04_SAME_WINDOW_COMPARABILITY     → PASS
G05_COMPUTATIONAL_CONTEXT         → PASS
G06_PROBABILITY_SUMMARIES         → PASS
G07_MODEL_RECORDS                 → PASS
G08_METRIC_SPECIFIC_EXTREMA       → PASS
G09_PAIRWISE_SAME_WINDOW_DELTAS   → PASS
G10_W_SHORT_EVIDENCE              → PASS
G11_W_LONG_EVIDENCE               → PASS
G12_PROBABILITY_SANITY            → PASS
G13_SELECTION_BOUNDARY            → PASS
G14_SOURCE_LINEAGE                → PASS
G15_COMPARISON_PERSISTENCE        → PASS
G16_COMPARISON_ROUND_TRIP         → PASS
G17_FINAL_TEST_ISOLATION          → PASS
```

Technical gates:

`17 / 17 PASS`

---

## R01 — Execution complete?

Evidence:

```text
21 / 21 code cells
execution_count = 1 → 21
errors = 0
stderr = 0
```

Result:

`PASS`

---

## R02 — Upstream artifacts verified?

Evidence:

M6.2 / M6.3 / M6.4 handoff gate PASS.

Result:

`PASS`

---

## R03 — Canonical validation target verified?

Evidence:

712,458 rows / 1,052 fraud / matching SHA-256.

Result:

`PASS`

---

## R04 — Same-window groups valid?

Evidence:

```text
W_SHORT:
LR / DT / RF

W_LONG:
LR / DT / RF
```

Result:

`PASS`

---

## R05 — Same-window comparability verified?

Evidence:

10 controlled fields match in each group.

Result:

`PASS`

---

## R06 — W_SHORT cross-model evidence complete?

Evidence:

metric + confusion + alerts + probability + computational context available for LR/DT/RF.

Result:

`PASS`

---

## R07 — W_LONG cross-model evidence complete?

Evidence:

same evidence bundle complete.

Result:

`PASS`

---

## R08 — Pairwise same-window deltas complete?

Evidence:

```text
DT - LR
RF - LR
RF - DT
```

for both windows.

Result:

`PASS`

---

## R09 — Probability behavior verified without threshold search?

Evidence:

6 / 6 risk-score artifacts verified; descriptive-only scope preserved.

Result:

`PASS`

---

## R10 — Computational evidence reviewed?

Evidence:

fit/prediction time available for all six runs.

Result:

`PASS`

---

## R11 — No mixed-window model comparison?

Evidence:

```text
mixed_window_model_comparison_used:
False
```

Result:

`PASS`

---

## R12 — Selection boundary preserved?

Evidence:

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

Result:

`PASS`

---

## R13 — Persistence and FINAL TEST isolation valid?

Evidence:

comparison persistence + round-trip PASS.

FINAL TEST isolation PASS.

Result:

`PASS`

---

## Overall M6.5 Gate

```text
Technical gates:
17 / 17 PASS

Runtime review gates:
13 / 13 PASS

Blocking issue:
NONE
```

Final:

`M6.5 — PASS`

Handoff:

`READY FOR M6.6`

# 28. Kết luận M6.5

M6.5 đã hoàn thành cross-model comparative evaluation trên hai same-window populations.

Canonical groups:

```text
W_SHORT:
LR vs DT vs RF

W_LONG:
LR vs DT vs RF
```

Same-window comparability:

`VERIFIED`

W_SHORT evidence:

```text
Highest F1:
Random Forest

Highest Recall:
Decision Tree

Highest Precision:
Logistic Regression
```

W_LONG evidence:

```text
Highest F1:
Decision Tree

Highest Recall:
Decision Tree

Highest Precision:
Random Forest
```

Điều quan trọng:

```text
Không có một model family
dẫn toàn bộ F1 / Recall / Precision /
FP / FN / alert burden / compute cost
trong cả hai same-window groups.
```

Vì vậy M6.5 xác nhận:

`MODEL TRADE-OFF EXISTS`

chứ không khóa:

`FINAL MODEL WINNER`.

Probability evidence:

```text
6 / 6 persisted risk-score artifacts:
VERIFIED

Analysis:
DESCRIPTIVE ONLY

Threshold search:
NONE

Calibration claim:
NONE
```

Computational evidence:

```text
LR:
lowest fit time in both windows

DT:
intermediate fit cost

RF:
highest fit cost among baseline families
```

M6.5 đã persist:

```text
data/processed/m6_05_cross_model_comparative_evaluation/
    m6_05_cross_model_comparison.json
    m6_05_comparison_manifest.json
```

Final state:

```text
M6.5 — PASS

Execution Integrity:
VERIFIED

Same-window Groups:
2 / 2 VERIFIED

W_SHORT Cross-model Evidence:
VERIFIED

W_LONG Cross-model Evidence:
VERIFIED

Pairwise Same-window Deltas:
VERIFIED

Probability Behavior:
DESCRIPTIVE — VERIFIED

Computational Context:
VERIFIED

Mixed-window Model Comparison:
NONE

Threshold Search:
NONE

Training-window Winner:
OPEN

Model-family Winner:
OPEN

Final Model:
OPEN

Final Threshold:
OPEN

FINAL TEST:
PROTECTED

Blocking Issue:
NONE

READY FOR M6.6
```

Bước tiếp theo:

`M6.6 — False Positive / False Negative error analysis`

M6.6 phải chuyển từ aggregate evidence sang actual validation transactions:

```text
Fraud side:
FN vs TP

Non-fraud side:
FP vs TN
```

và chỉ được đưa ra row-level findings sau khi xác minh mapping từ:

`row_id_validation`

về canonical transaction-level semantic data.

Probability context ở M6.6 tiếp tục là descriptive evidence; không thay threshold.
