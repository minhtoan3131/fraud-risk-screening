# M6.4 — Controlled training-window comparison

Milestone:

`M6 — Evaluation + Error Analysis`

Substep:

`M6.4 — Controlled Training-window Comparison`

Work type:

`RUNTIME COMPARATIVE EVALUATION`

Câu hỏi trung tâm:

> Trong từng model family, khi giữ model/config/features/preprocessing/validation/imbalance/threshold policy cố định và chỉ thay classifier-training window, W_SHORT và W_LONG tạo validation behavior khác nhau như thế nào?

Official controlled pairs:

```text
Logistic Regression:
M5-LR-SHORT-B04
vs
M5-LR-LONG-B04

Decision Tree:
M5-DT-SHORT-B01
vs
M5-DT-LONG-B01

Random Forest:
M5-RF-SHORT-B01
vs
M5-RF-LONG-B01
```

M6.4 không:

- fit/retrain model;
- tuning;
- imbalance intervention;
- threshold optimization;
- cross-model family selection;
- FINAL TEST access;
- khóa final training-window winner.

Runtime-dependent status trước khi chạy:

`NOT YET VERIFIED`

## 1. Contract kế thừa

M6.4 kế thừa trực tiếp:

- `CANON-M3.6 — Thiết kế protocol so sánh training window`;
- `CANON-Kế hoạch Milestone 6 — Evaluation và Error Analysis`;
- `CANON-M6.1 — Evaluation Charter / scope / guardrails`;
- `M6.2 — verified Evaluation Registry`;
- `M6.3 — verified six-run metric/confusion profiles`.

M3.6 khóa nguyên tắc:

```text
ONLY TRAINING WINDOW CHANGES
```

Validation:

```text
2019-01 → 2019-05
```

Primary comparison metric:

`F1_fraud`

Mandatory secondary evidence:

```text
Recall_fraud
Precision_fraud
TP
FP
FN
TN
predicted-positive count/rate
Accuracy — reference only
```

FINAL TEST:

`PROHIBITED FOR WINDOW SELECTION`

## 2. Interpretation boundary

M3.6 cho phép F1 tạo:

`PROVISIONAL LEADER`

nhưng không cho phép chỉ sort F1 rồi viết:

`W_SHORT/W_LONG thắng`

mà không đọc trade-off.

M6.4 sẽ:

1. kiểm tra controlled-pair identity;
2. tính delta;
3. xác định provisional F1 leader cho từng family;
4. đọc Recall / Precision;
5. đọc TP / FP / FN / TN;
6. đọc alert volume;
7. đọc computational context;
8. ghi pair-specific findings.

M6.4 **không** khóa global final training-window winner.

Final state phải giữ:

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

Nếu cần robustness/tie-break:

`HANDOFF M7`

## 3. Delta convention

Mọi delta trong notebook được định nghĩa:

```text
DELTA = W_LONG - W_SHORT
```

Do đó:

```text
ΔF1 < 0
→ W_LONG thấp hơn W_SHORT về F1

ΔRecall < 0
→ W_LONG bắt ít fraud hơn W_SHORT

ΔFP > 0
→ W_LONG tạo nhiều false alert hơn W_SHORT

ΔFN > 0
→ W_LONG bỏ sót nhiều fraud hơn W_SHORT
```

Delta convention phải giữ nhất quán trong toàn M6.4.


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


## 4. Locate M6.2 / M6.3 artifacts


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

M6_04_REL = (
    Path("data")
    / "processed"
    / "m6_04_training_window_comparison"
)


candidate_roots = [
    Path.cwd(),
    *list(Path.cwd().parents)[:6],
]


PROJECT_ROOT = None

required_rel_paths = [
    M6_02_REL
    / "m6_02_evaluation_registry.json",

    M6_03_REL
    / "m6_03_metric_confusion_analysis.json",

    M6_03_REL
    / "m6_03_analysis_manifest.json",
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
        "M6.2 registry và M6.3 analysis artifacts."
    )


M6_02_DIR = (
    PROJECT_ROOT
    / M6_02_REL
)

M6_03_DIR = (
    PROJECT_ROOT
    / M6_03_REL
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / M6_04_REL
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

M6_03_MANIFEST_PATH = (
    M6_03_DIR
    / "m6_03_analysis_manifest.json"
)


print("PROJECT_ROOT:")
print(PROJECT_ROOT)

print("\nM6_02_REGISTRY_PATH:")
print(M6_02_REGISTRY_PATH)

print("\nM6_03_ANALYSIS_PATH:")
print(M6_03_ANALYSIS_PATH)

print("\nOUTPUT_DIR:")
print(OUTPUT_DIR)

```

    PROJECT_ROOT:
    /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening
    
    M6_02_REGISTRY_PATH:
    /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/processed/m6_02_evaluation_artifact_audit/m6_02_evaluation_registry.json
    
    M6_03_ANALYSIS_PATH:
    /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/processed/m6_03_baseline_metric_confusion_analysis/m6_03_metric_confusion_analysis.json
    
    OUTPUT_DIR:
    /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/processed/m6_04_training_window_comparison


## 5. Load và verify M6.3 handoff


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
    M6_03_MANIFEST_PATH,
    "r",
    encoding="utf-8",
) as file:
    m6_03_manifest = json.load(
        file
    )


EXPECTED_M6_03_VERSION = (
    "M6.3-metric-confusion-analysis-v1"
)

EXPECTED_VALIDATION_ROWS = 712_458
EXPECTED_VALIDATION_FRAUD = 1_052


assert (
    m6_03_analysis[
        "analysis_version"
    ]
    == EXPECTED_M6_03_VERSION
)

assert (
    m6_03_analysis[
        "validation_rows"
    ]
    == EXPECTED_VALIDATION_ROWS
)

assert (
    m6_03_analysis[
        "validation_fraud_rows"
    ]
    == EXPECTED_VALIDATION_FRAUD
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
        "selection_state"
    ][
        "training_window_winner"
    ]
    == "OPEN"
)

assert (
    m6_03_analysis[
        "selection_state"
    ][
        "model_family_winner"
    ]
    == "OPEN"
)

assert (
    m6_03_analysis[
        "selection_state"
    ][
        "m6_03_selection_authorized"
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
    m6_03_manifest[
        "official_runs_profiled"
    ]
    == 6
)

assert (
    m6_03_manifest[
        "confusion_arithmetic"
    ]
    == "PASS"
)

assert (
    m6_03_manifest[
        "metric_to_count_consistency"
    ]
    == "PASS"
)

assert (
    m6_03_manifest[
        "training_window_selection"
    ]
    == "NOT_PERFORMED"
)

assert (
    m6_03_manifest[
        "final_test_accessed"
    ]
    is False
)


print(
    "M6.4 M6.3 HANDOFF GATE: PASS"
)

```

    M6.4 M6.3 HANDOFF GATE: PASS


## 6. Controlled pair definitions

Comparison IDs:

```text
TW-LR-B04
TW-DT-B01
TW-RF-B01
```

Mỗi pair phải có:

- cùng model family;
- cùng exact model config;
- cùng feature version;
- cùng preprocessing version;
- cùng matrix schema;
- cùng validation population;
- cùng imbalance strategy;
- cùng threshold policy;
- cùng risk-score semantics;
- khác `training_window_id`.


```python

PAIR_SPECS = {
    "TW-LR-B04": {
        "family":
            "Logistic Regression",

        "short_id":
            "M5-LR-SHORT-B04",

        "long_id":
            "M5-LR-LONG-B04",

        "config_id":
            "LR-B04-LBFGS-L2-C1",
    },

    "TW-DT-B01": {
        "family":
            "Decision Tree",

        "short_id":
            "M5-DT-SHORT-B01",

        "long_id":
            "M5-DT-LONG-B01",

        "config_id":
            "DT-B01-DEFAULT-GINI-UNPRUNED",
    },

    "TW-RF-B01": {
        "family":
            "Random Forest",

        "short_id":
            "M5-RF-SHORT-B01",

        "long_id":
            "M5-RF-LONG-B01",

        "config_id":
            "RF-B01-100-GINI-SQRT-BOOTSTRAP",
    },
}


assert len(
    PAIR_SPECS
) == 3


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


for pair_id, spec in (
    PAIR_SPECS.items()
):
    assert (
        spec["short_id"]
        in profiles_by_id
    )

    assert (
        spec["long_id"]
        in profiles_by_id
    )

    assert (
        spec["short_id"]
        in registry_by_id
    )

    assert (
        spec["long_id"]
        in registry_by_id
    )


print(
    "Controlled pair specs:",
    len(
        PAIR_SPECS
    ),
)

print(
    "\nM6.4 PAIR DEFINITION GATE: PASS"
)

```

    Controlled pair specs: 3
    
    M6.4 PAIR DEFINITION GATE: PASS


## 7. Controlled-pair integrity audit


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


pair_integrity_records = []


for pair_id, spec in (
    PAIR_SPECS.items()
):
    short_record = (
        registry_by_id[
            spec["short_id"]
        ]
    )

    long_record = (
        registry_by_id[
            spec["long_id"]
        ]
    )

    assert (
        short_record[
            "model_family"
        ]
        == spec["family"]
    )

    assert (
        long_record[
            "model_family"
        ]
        == spec["family"]
    )

    assert (
        short_record[
            "model_config_id"
        ]
        == spec["config_id"]
    )

    assert (
        long_record[
            "model_config_id"
        ]
        == spec["config_id"]
    )

    assert (
        short_record[
            "training_window_id"
        ]
        == "W_SHORT"
    )

    assert (
        long_record[
            "training_window_id"
        ]
        == "W_LONG"
    )

    mismatches = []

    for field in (
        CONTROLLED_FIELDS
    ):
        if (
            short_record[field]
            != long_record[field]
        ):
            mismatches.append(
                field
            )

    assert not mismatches, (
        f"{pair_id}: "
        f"controlled-field mismatch "
        f"{mismatches}"
    )

    pair_integrity_records.append(
        {
            "comparison_id":
                pair_id,

            "model_family":
                spec["family"],

            "controlled_fields_checked":
                len(
                    CONTROLLED_FIELDS
                ),

            "controlled_fields_match":
                True,

            "intended_difference":
                "training_window_id",

            "integrity_status":
                "PASS",
        }
    )


print(
    "Controlled fields checked per pair:",
    len(
        CONTROLLED_FIELDS
    ),
)

for record in (
    pair_integrity_records
):
    print(
        record[
            "comparison_id"
        ],
        "→",
        record[
            "integrity_status"
        ],
    )


print(
    "\nM6.4 CONTROLLED-PAIR INTEGRITY GATE: PASS"
)

```

    Controlled fields checked per pair: 12
    TW-LR-B04 → PASS
    TW-DT-B01 → PASS
    TW-RF-B01 → PASS
    
    M6.4 CONTROLLED-PAIR INTEGRITY GATE: PASS


## 8. Load computational context từ persisted M5 summaries

M3.6 yêu cầu đọc computational cost cùng predictive evidence.

M6.4 không train lại model.

Notebook chỉ đọc:

```text
fit_seconds
prediction_seconds
warning_count
```

từ summary artifact đã được M6.2 identity-audit.


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
    "M6.4 COMPUTATIONAL CONTEXT LOAD GATE: PASS"
)

```

    M6.4 COMPUTATIONAL CONTEXT LOAD GATE: PASS


## 9. Build controlled comparison records

Delta convention:

`W_LONG − W_SHORT`

Primary decision evidence:

`F1_fraud`

Provisional leader:

- `W_SHORT` nếu SHORT F1 > LONG F1;
- `W_LONG` nếu LONG F1 > SHORT F1;
- `TIE_EXACT` chỉ nếu numerical equality exact trong stored evidence.

Không dùng arbitrary epsilon.


```python

def build_pair_comparison(
    pair_id,
    spec,
):
    short_profile = (
        profiles_by_id[
            spec["short_id"]
        ]
    )

    long_profile = (
        profiles_by_id[
            spec["long_id"]
        ]
    )

    short_runtime = (
        runtime_context[
            spec["short_id"]
        ]
    )

    long_runtime = (
        runtime_context[
            spec["long_id"]
        ]
    )

    delta_fields = [
        "f1_fraud",
        "recall_fraud",
        "precision_fraud",
        "accuracy_reference",
        "tp",
        "fp",
        "fn",
        "tn",
        "alert_count",
        "alert_rate",
        "fraud_capture_count",
        "fraud_miss_count",
        "total_misclassified",
    ]

    deltas = {}

    for field in delta_fields:
        deltas[
            f"delta_long_minus_short__{field}"
        ] = (
            long_profile[field]
            - short_profile[field]
        )

    if (
        short_profile[
            "f1_fraud"
        ]
        > long_profile[
            "f1_fraud"
        ]
    ):
        provisional_f1_leader = (
            "W_SHORT"
        )
    elif (
        long_profile[
            "f1_fraud"
        ]
        > short_profile[
            "f1_fraud"
        ]
    ):
        provisional_f1_leader = (
            "W_LONG"
        )
    else:
        provisional_f1_leader = (
            "TIE_EXACT"
        )

    recall_leader = (
        "W_SHORT"
        if (
            short_profile[
                "recall_fraud"
            ]
            > long_profile[
                "recall_fraud"
            ]
        )
        else (
            "W_LONG"
            if (
                long_profile[
                    "recall_fraud"
                ]
                > short_profile[
                    "recall_fraud"
                ]
            )
            else "TIE_EXACT"
        )
    )

    precision_leader = (
        "W_SHORT"
        if (
            short_profile[
                "precision_fraud"
            ]
            > long_profile[
                "precision_fraud"
            ]
        )
        else (
            "W_LONG"
            if (
                long_profile[
                    "precision_fraud"
                ]
                > short_profile[
                    "precision_fraud"
                ]
            )
            else "TIE_EXACT"
        )
    )

    fit_ratio = (
        long_runtime[
            "fit_seconds"
        ]
        / short_runtime[
            "fit_seconds"
        ]
        if (
            short_runtime[
                "fit_seconds"
            ]
            > 0
        )
        else None
    )

    comparison = {
        "comparison_id":
            pair_id,

        "model_family":
            spec["family"],

        "model_config_id":
            spec["config_id"],

        "delta_convention":
            "W_LONG_MINUS_W_SHORT",

        "short_experiment_id":
            spec["short_id"],

        "long_experiment_id":
            spec["long_id"],

        "short":
            short_profile,

        "long":
            long_profile,

        "deltas":
            deltas,

        "short_runtime":
            short_runtime,

        "long_runtime":
            long_runtime,

        "fit_time_ratio_long_over_short":
            float(
                fit_ratio
            ),

        "provisional_f1_leader":
            provisional_f1_leader,

        "recall_leader":
            recall_leader,

        "precision_leader":
            precision_leader,

        "pair_integrity":
            "PASS",

        "window_specific_tuning":
            False,

        "threshold_policy_changed":
            False,

        "final_test_accessed":
            False,

        "pair_interpretation_scope":
            (
                "CONTROLLED_WITHIN_MODEL_FAMILY_"
                "VALIDATION_COMPARISON"
            ),

        "final_training_window_winner":
            "OPEN",
    }

    return comparison


pair_comparisons = [
    build_pair_comparison(
        pair_id,
        spec,
    )
    for pair_id, spec
    in PAIR_SPECS.items()
]


assert len(
    pair_comparisons
) == 3


print(
    "Pair comparisons:",
    len(
        pair_comparisons
    ),
)

print(
    "\nM6.4 COMPARISON BUILD GATE: PASS"
)

```

    Pair comparisons: 3
    
    M6.4 COMPARISON BUILD GATE: PASS


## 10. Print pair evidence

Runtime output phải hiển thị cho từng family:

- SHORT metrics;
- LONG metrics;
- LONG − SHORT deltas;
- TP/FP/FN;
- alert volume;
- provisional F1 leader;
- Recall leader;
- Precision leader;
- fit-time context.

Không in global winner.


```python

def print_pair_comparison(
    comparison,
):
    short = comparison[
        "short"
    ]

    long = comparison[
        "long"
    ]

    deltas = comparison[
        "deltas"
    ]

    print("=" * 72)

    print(
        "Comparison:",
        comparison[
            "comparison_id"
        ],
    )

    print(
        "Family:",
        comparison[
            "model_family"
        ],
    )

    print(
        "Config:",
        comparison[
            "model_config_id"
        ],
    )

    print()

    print("W_SHORT")

    print(
        "  F1:",
        short[
            "f1_fraud"
        ],
    )

    print(
        "  Recall:",
        short[
            "recall_fraud"
        ],
    )

    print(
        "  Precision:",
        short[
            "precision_fraud"
        ],
    )

    print(
        "  TP / FP / FN / TN:",
        short["tp"],
        short["fp"],
        short["fn"],
        short["tn"],
    )

    print(
        "  Alerts:",
        short[
            "alert_count"
        ],
    )

    print(
        "  Fit seconds:",
        comparison[
            "short_runtime"
        ][
            "fit_seconds"
        ],
    )

    print()

    print("W_LONG")

    print(
        "  F1:",
        long[
            "f1_fraud"
        ],
    )

    print(
        "  Recall:",
        long[
            "recall_fraud"
        ],
    )

    print(
        "  Precision:",
        long[
            "precision_fraud"
        ],
    )

    print(
        "  TP / FP / FN / TN:",
        long["tp"],
        long["fp"],
        long["fn"],
        long["tn"],
    )

    print(
        "  Alerts:",
        long[
            "alert_count"
        ],
    )

    print(
        "  Fit seconds:",
        comparison[
            "long_runtime"
        ][
            "fit_seconds"
        ],
    )

    print()

    print(
        "DELTA = W_LONG - W_SHORT"
    )

    print(
        "  ΔF1:",
        deltas[
            "delta_long_minus_short__f1_fraud"
        ],
    )

    print(
        "  ΔRecall:",
        deltas[
            "delta_long_minus_short__recall_fraud"
        ],
    )

    print(
        "  ΔPrecision:",
        deltas[
            "delta_long_minus_short__precision_fraud"
        ],
    )

    print(
        "  ΔTP:",
        deltas[
            "delta_long_minus_short__tp"
        ],
    )

    print(
        "  ΔFP:",
        deltas[
            "delta_long_minus_short__fp"
        ],
    )

    print(
        "  ΔFN:",
        deltas[
            "delta_long_minus_short__fn"
        ],
    )

    print(
        "  ΔAlerts:",
        deltas[
            "delta_long_minus_short__alert_count"
        ],
    )

    print(
        "  Fit ratio LONG / SHORT:",
        comparison[
            "fit_time_ratio_long_over_short"
        ],
    )

    print()

    print(
        "Provisional F1 leader:",
        comparison[
            "provisional_f1_leader"
        ],
    )

    print(
        "Recall leader:",
        comparison[
            "recall_leader"
        ],
    )

    print(
        "Precision leader:",
        comparison[
            "precision_leader"
        ],
    )

    print(
        "Final training-window winner:",
        comparison[
            "final_training_window_winner"
        ],
    )


for comparison in (
    pair_comparisons
):
    print_pair_comparison(
        comparison
    )


print(
    "\nM6.4 THREE CONTROLLED PAIR EVIDENCE: READY"
)

```

    ========================================================================
    Comparison: TW-LR-B04
    Family: Logistic Regression
    Config: LR-B04-LBFGS-L2-C1
    
    W_SHORT
      F1: 0.33747547416612167
      Recall: 0.24524714828897337
      Precision: 0.5408805031446541
      TP / FP / FN / TN: 258 219 794 711187
      Alerts: 477
      Fit seconds: 0.8222020840039477
    
    W_LONG
      F1: 0.03996366939146231
      Recall: 0.02091254752851711
      Precision: 0.4489795918367347
      TP / FP / FN / TN: 22 27 1030 711379
      Alerts: 49
      Fit seconds: 3.6143539589829743
    
    DELTA = W_LONG - W_SHORT
      ΔF1: -0.2975118047746594
      ΔRecall: -0.22433460076045625
      ΔPrecision: -0.09190091130791939
      ΔTP: -236
      ΔFP: -192
      ΔFN: 236
      ΔAlerts: -428
      Fit ratio LONG / SHORT: 4.395943563389971
    
    Provisional F1 leader: W_SHORT
    Recall leader: W_SHORT
    Precision leader: W_SHORT
    Final training-window winner: OPEN
    ========================================================================
    Comparison: TW-DT-B01
    Family: Decision Tree
    Config: DT-B01-DEFAULT-GINI-UNPRUNED
    
    W_SHORT
      F1: 0.3270564915758176
      Recall: 0.31368821292775667
      Precision: 0.3416149068322981
      TP / FP / FN / TN: 330 636 722 710770
      Alerts: 966
      Fit seconds: 6.661822540976573
    
    W_LONG
      F1: 0.19913419913419914
      Recall: 0.1967680608365019
      Precision: 0.20155793573515093
      TP / FP / FN / TN: 207 820 845 710586
      Alerts: 1027
      Fit seconds: 116.46980562497629
    
    DELTA = W_LONG - W_SHORT
      ΔF1: -0.12792229244161848
      ΔRecall: -0.11692015209125478
      ΔPrecision: -0.14005697109714718
      ΔTP: -123
      ΔFP: 184
      ΔFN: 123
      ΔAlerts: 61
      Fit ratio LONG / SHORT: 17.483174447919577
    
    Provisional F1 leader: W_SHORT
    Recall leader: W_SHORT
    Precision leader: W_SHORT
    Final training-window winner: OPEN
    ========================================================================
    Comparison: TW-RF-B01
    Family: Random Forest
    Config: RF-B01-100-GINI-SQRT-BOOTSTRAP
    
    W_SHORT
      F1: 0.3664670658682635
      Recall: 0.2908745247148289
      Precision: 0.49514563106796117
      TP / FP / FN / TN: 306 312 746 711094
      Alerts: 618
      Fit seconds: 40.69802879198687
    
    W_LONG
      F1: 0.15270935960591134
      Recall: 0.08840304182509506
      Precision: 0.5602409638554217
      TP / FP / FN / TN: 93 73 959 711333
      Alerts: 166
      Fit seconds: 627.7140797079774
    
    DELTA = W_LONG - W_SHORT
      ΔF1: -0.21375770626235216
      ΔRecall: -0.20247148288973382
      ΔPrecision: 0.06509533278746049
      ΔTP: -213
      ΔFP: -239
      ΔFN: 213
      ΔAlerts: -452
      Fit ratio LONG / SHORT: 15.423697371592834
    
    Provisional F1 leader: W_SHORT
    Recall leader: W_SHORT
    Precision leader: W_LONG
    Final training-window winner: OPEN
    
    M6.4 THREE CONTROLLED PAIR EVIDENCE: READY


## 11. Predictive-trade-off classification

M6.4 ghi pair behavior theo rule mô tả:

### `SHORT_DOMINATES_F1_RECALL_PRECISION`

W_SHORT cao hơn cả:

- F1;
- Recall;
- Precision.

### `SHORT_F1_RECALL__LONG_PRECISION`

W_SHORT cao hơn:

- F1;
- Recall;

nhưng W_LONG cao hơn:

- Precision.

### `LONG_DOMINATES_F1_RECALL_PRECISION`

W_LONG cao hơn cả ba.

### `MIXED_OTHER`

Các pattern khác.

Đây là descriptive classification, không phải final winner rule.


```python

def classify_tradeoff(
    comparison,
):
    f1_leader = comparison[
        "provisional_f1_leader"
    ]

    recall_leader = comparison[
        "recall_leader"
    ]

    precision_leader = comparison[
        "precision_leader"
    ]

    if (
        f1_leader == "W_SHORT"
        and recall_leader == "W_SHORT"
        and precision_leader == "W_SHORT"
    ):
        return (
            "SHORT_DOMINATES_"
            "F1_RECALL_PRECISION"
        )

    if (
        f1_leader == "W_SHORT"
        and recall_leader == "W_SHORT"
        and precision_leader == "W_LONG"
    ):
        return (
            "SHORT_F1_RECALL__"
            "LONG_PRECISION"
        )

    if (
        f1_leader == "W_LONG"
        and recall_leader == "W_LONG"
        and precision_leader == "W_LONG"
    ):
        return (
            "LONG_DOMINATES_"
            "F1_RECALL_PRECISION"
        )

    return "MIXED_OTHER"


for comparison in (
    pair_comparisons
):
    comparison[
        "tradeoff_class"
    ] = classify_tradeoff(
        comparison
    )

    print(
        comparison[
            "comparison_id"
        ],
        "→",
        comparison[
            "tradeoff_class"
        ],
    )


print(
    "\nM6.4 TRADE-OFF CLASSIFICATION GATE: PASS"
)

```

    TW-LR-B04 → SHORT_DOMINATES_F1_RECALL_PRECISION
    TW-DT-B01 → SHORT_DOMINATES_F1_RECALL_PRECISION
    TW-RF-B01 → SHORT_F1_RECALL__LONG_PRECISION
    
    M6.4 TRADE-OFF CLASSIFICATION GATE: PASS


## 12. Cross-pair consistency diagnostic

M6.4 được phép hỏi:

> Provisional F1 leader có cùng hướng trong cả 3 controlled pairs không?

Đây **không** phải cross-model family selection.

Nó chỉ tổng hợp consistency của training-window direction.

Nếu cả ba pair cùng hướng:

ghi:

`CONSISTENT PROVISIONAL DIRECTION`

nhưng final training-window winner vẫn:

`OPEN`

cho đến khi protocol selection/robustness ở giai đoạn phù hợp cho phép khóa.


```python

provisional_f1_leaders = [
    comparison[
        "provisional_f1_leader"
    ]
    for comparison
    in pair_comparisons
]


unique_provisional_f1_leaders = set(
    provisional_f1_leaders
)


if (
    len(
        unique_provisional_f1_leaders
    )
    == 1
):
    cross_pair_direction = (
        "CONSISTENT_PROVISIONAL_DIRECTION"
    )

    common_provisional_direction = (
        provisional_f1_leaders[0]
    )
else:
    cross_pair_direction = (
        "MIXED_PROVISIONAL_DIRECTION"
    )

    common_provisional_direction = (
        "NONE"
    )


print(
    "Provisional F1 leaders:",
    provisional_f1_leaders,
)

print(
    "Cross-pair direction:",
    cross_pair_direction,
)

print(
    "Common provisional direction:",
    common_provisional_direction,
)

print(
    "Final training-window winner:",
    "OPEN",
)


print(
    "\nM6.4 CROSS-PAIR DIRECTION DIAGNOSTIC: READY"
)

```

    Provisional F1 leaders: ['W_SHORT', 'W_SHORT', 'W_SHORT']
    Cross-pair direction: CONSISTENT_PROVISIONAL_DIRECTION
    Common provisional direction: W_SHORT
    Final training-window winner: OPEN
    
    M6.4 CROSS-PAIR DIRECTION DIAGNOSTIC: READY


## 13. No arbitrary epsilon / no automatic winner

M3.6 cấm rule kiểu:

```text
|ΔF1| < 0.01
→ tie
```

M6.4 không tạo epsilon.

M6.4 cũng không tự động chuyển:

`provisional F1 leader`

thành:

`final training-window winner`.

Nếu comparative evidence cần robustness:

`M7 temporal robustness / CV`

theo upstream policy.


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

    "arbitrary_f1_epsilon_used":
        False,

    "window_specific_tuning_used":
        False,

    "threshold_optimization_used":
        False,

    "m6_04_final_selection_authorized":
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
        "arbitrary_f1_epsilon_used"
    ]
    is False
)

assert (
    selection_state[
        "window_specific_tuning_used"
    ]
    is False
)

assert (
    selection_state[
        "threshold_optimization_used"
    ]
    is False
)

assert (
    selection_state[
        "m6_04_final_selection_authorized"
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
    "Arbitrary F1 epsilon used:",
    selection_state[
        "arbitrary_f1_epsilon_used"
    ],
)

print(
    "\nM6.4 SELECTION-BOUNDARY GATE: PASS"
)

```

    Training-window winner: OPEN
    Arbitrary F1 epsilon used: False
    
    M6.4 SELECTION-BOUNDARY GATE: PASS


## 14. Source-artifact fingerprints


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
        M6_02_REGISTRY_PATH
    )
)

m6_03_analysis_sha256 = (
    sha256_file(
        M6_03_ANALYSIS_PATH
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

```

    M6.2 registry SHA256:
    9e112883d5aa1358c58c714edc0f0ef7913252e94c56f3c6b28d3ba1b7f4e6fb
    
    M6.3 analysis SHA256:
    290f348ba20a4a19f659e24bb517ca8cce2229605af3f7ee762520e5b1d7b168


## 15. Persist M6.4 controlled-comparison artifact


```python

COMPARISON_VERSION = (
    "M6.4-training-window-comparison-v1"
)

COMPARISON_PATH = (
    OUTPUT_DIR
    / "m6_04_training_window_comparison.json"
)

MANIFEST_PATH = (
    OUTPUT_DIR
    / "m6_04_comparison_manifest.json"
)


comparison_payload = {
    "comparison_version":
        COMPARISON_VERSION,

    "m6_substep":
        "M6.4",

    "delta_convention":
        "W_LONG_MINUS_W_SHORT",

    "source_m6_02_registry":
        str(
            M6_02_REGISTRY_PATH.relative_to(
                PROJECT_ROOT
            )
        ),

    "source_m6_02_registry_sha256":
        m6_02_registry_sha256,

    "source_m6_03_analysis":
        str(
            M6_03_ANALYSIS_PATH.relative_to(
                PROJECT_ROOT
            )
        ),

    "source_m6_03_analysis_sha256":
        m6_03_analysis_sha256,

    "validation_rows":
        EXPECTED_VALIDATION_ROWS,

    "validation_fraud_rows":
        EXPECTED_VALIDATION_FRAUD,

    "primary_metric":
        "F1_fraud",

    "mandatory_secondary_evidence": [
        "Recall_fraud",
        "Precision_fraud",
        "TP",
        "FP",
        "FN",
        "TN",
        "predicted_positive_count",
        "predicted_positive_rate",
        "Accuracy_reference",
    ],

    "comparison_pairs":
        pair_comparisons,

    "cross_pair_direction":
        cross_pair_direction,

    "common_provisional_direction":
        common_provisional_direction,

    "selection_state":
        selection_state,

    "final_test_accessed":
        False,
}


comparison_manifest = {
    "m6_substep":
        "M6.4",

    "comparison_version":
        COMPARISON_VERSION,

    "controlled_pairs_expected":
        3,

    "controlled_pairs_built":
        len(
            pair_comparisons
        ),

    "controlled_pair_integrity":
        "PASS",

    "delta_convention":
        "W_LONG_MINUS_W_SHORT",

    "primary_metric":
        "F1_fraud",

    "arbitrary_f1_epsilon_used":
        False,

    "window_specific_tuning":
        "NOT_PERFORMED",

    "threshold_optimization":
        "NOT_PERFORMED",

    "final_training_window_selected":
        False,

    "cross_model_family_selection":
        "NOT_PERFORMED",

    "final_test_accessed":
        False,

    "decision":
        "OPEN — REQUIRES M6.4 RUNTIME REVIEW",
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


print("Comparison artifact:")
print(COMPARISON_PATH)

print("\nComparison manifest:")
print(MANIFEST_PATH)

print(
    "\nM6.4 COMPARISON PERSISTENCE GATE: PASS"
)

```

    Comparison artifact:
    /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/processed/m6_04_training_window_comparison/m6_04_training_window_comparison.json
    
    Comparison manifest:
    /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/processed/m6_04_training_window_comparison/m6_04_comparison_manifest.json
    
    M6.4 COMPARISON PERSISTENCE GATE: PASS


## 16. Persisted artifact round-trip


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
    len(
        comparison_roundtrip[
            "comparison_pairs"
        ]
    )
    == 3
)

assert {
    pair[
        "comparison_id"
    ]
    for pair
    in comparison_roundtrip[
        "comparison_pairs"
    ]
} == set(
    PAIR_SPECS.keys()
)

assert (
    comparison_roundtrip[
        "selection_state"
    ][
        "training_window_winner"
    ]
    == "OPEN"
)

assert (
    comparison_roundtrip[
        "selection_state"
    ][
        "m6_04_final_selection_authorized"
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
        "controlled_pairs_built"
    ]
    == 3
)

assert (
    manifest_roundtrip[
        "controlled_pair_integrity"
    ]
    == "PASS"
)

assert (
    manifest_roundtrip[
        "arbitrary_f1_epsilon_used"
    ]
    is False
)

assert (
    manifest_roundtrip[
        "final_training_window_selected"
    ]
    is False
)


print(
    "M6.4 COMPARISON ROUND-TRIP GATE: PASS"
)

```

    M6.4 COMPARISON ROUND-TRIP GATE: PASS


## 17. FINAL TEST isolation


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


for comparison in (
    pair_comparisons
):
    assert (
        comparison[
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
    M6_02_REGISTRY_PATH,
    M6_03_ANALYSIS_PATH,
    M6_03_MANIFEST_PATH,
]:
    assert (
        "final_test"
        not in path.name.lower()
    )


print(
    "M6.4 FINAL TEST ISOLATION GATE: PASS"
)

```

    M6.4 FINAL TEST ISOLATION GATE: PASS


## 18. M6.4 overall technical gate


```python

m6_04_gates = {
    "G01_M6_03_HANDOFF":
        True,

    "G02_PAIR_DEFINITION":
        True,

    "G03_CONTROLLED_PAIR_INTEGRITY":
        True,

    "G04_COMPUTATIONAL_CONTEXT":
        True,

    "G05_COMPARISON_BUILD":
        True,

    "G06_THREE_PAIR_EVIDENCE":
        True,

    "G07_TRADEOFF_CLASSIFICATION":
        True,

    "G08_CROSS_PAIR_DIRECTION":
        True,

    "G09_NO_ARBITRARY_EPSILON":
        True,

    "G10_SELECTION_BOUNDARY":
        True,

    "G11_SOURCE_LINEAGE":
        True,

    "G12_COMPARISON_PERSISTENCE":
        True,

    "G13_COMPARISON_ROUND_TRIP":
        True,

    "G14_FINAL_TEST_ISOLATION":
        True,
}


for gate_name, gate_value in (
    m6_04_gates.items()
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
    m6_04_gates.values()
)


print(
    "\nM6.4 OVERALL TECHNICAL GATE: PASS"
)

```

    G01_M6_03_HANDOFF → PASS
    G02_PAIR_DEFINITION → PASS
    G03_CONTROLLED_PAIR_INTEGRITY → PASS
    G04_COMPUTATIONAL_CONTEXT → PASS
    G05_COMPARISON_BUILD → PASS
    G06_THREE_PAIR_EVIDENCE → PASS
    G07_TRADEOFF_CLASSIFICATION → PASS
    G08_CROSS_PAIR_DIRECTION → PASS
    G09_NO_ARBITRARY_EPSILON → PASS
    G10_SELECTION_BOUNDARY → PASS
    G11_SOURCE_LINEAGE → PASS
    G12_COMPARISON_PERSISTENCE → PASS
    G13_COMPARISON_ROUND_TRIP → PASS
    G14_FINAL_TEST_ISOLATION → PASS
    
    M6.4 OVERALL TECHNICAL GATE: PASS


# 19. Runtime review và Findings M6.4

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

Không có partial execution, exception hoặc stderr bất thường.

Status:

`VERIFIED`

---

## 19.2. M6.3 handoff

Observed:

```text
M6.2 Evaluation Registry:
loaded

M6.3 metric/confusion analysis:
loaded

M6.3 analysis manifest:
loaded

M6.4 M6.3 HANDOFF GATE:
PASS
```

M6.4 sử dụng đúng persisted evidence từ M6.2/M6.3, không fit hoặc predict lại model.

Status:

`VERIFIED`

---

## 19.3. Controlled-pair integrity

Observed:

```text
Controlled pairs:
3

Controlled fields checked per pair:
12

TW-LR-B04:
PASS

TW-DT-B01:
PASS

TW-RF-B01:
PASS
```

Các field được kiểm soát gồm:

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

Intended difference duy nhất:

`training_window_id`

Runtime gate:

`M6.4 CONTROLLED-PAIR INTEGRITY GATE: PASS`

Status:

`VERIFIED`

---

## 19.4. Delta convention

Observed convention:

`DELTA = W_LONG - W_SHORT`

Interpretation:

```text
negative ΔF1:
W_LONG thấp hơn W_SHORT về F1

negative ΔRecall:
W_LONG bắt ít fraud hơn W_SHORT

positive ΔFN:
W_LONG bỏ sót nhiều fraud hơn W_SHORT

positive ΔFP:
W_LONG tạo nhiều false alert hơn W_SHORT
```

Status:

`LOCKED / VERIFIED`

---

## 19.5. Logistic Regression — controlled window comparison

Comparison:

`TW-LR-B04`

Config:

`LR-B04-LBFGS-L2-C1`

Observed:

```text
W_SHORT
F1        = 0.33747547416612167
Recall    = 0.24524714828897337
Precision = 0.5408805031446541
TP / FP / FN / TN
258 / 219 / 794 / 711187
Alerts    = 477
Fit sec   = 0.8222020840039477

W_LONG
F1        = 0.03996366939146231
Recall    = 0.02091254752851711
Precision = 0.4489795918367347
TP / FP / FN / TN
22 / 27 / 1030 / 711379
Alerts    = 49
Fit sec   = 3.6143539589829743
```

Delta:

```text
ΔF1        = -0.2975118047746594
ΔRecall    = -0.22433460076045625
ΔPrecision = -0.09190091130791939

ΔTP     = -236
ΔFP     = -192
ΔFN     = +236
ΔAlerts = -428

Fit ratio LONG / SHORT:
4.395943563389971
```

Pair-specific leaders:

```text
Provisional F1 leader:
W_SHORT

Recall leader:
W_SHORT

Precision leader:
W_SHORT
```

Trade-off classification:

`SHORT_DOMINATES_F1_RECALL_PRECISION`

Interpretation:

Trong LR-B04, W_SHORT cao hơn W_LONG đồng thời về F1, Recall và Precision trên canonical validation.

W_LONG tạo ít false alert hơn, nhưng đồng thời gần như không flag positive:

```text
49 alerts
22 TP
1030 FN
```

Do đó giảm FP ở W_LONG không thể được đọc tách khỏi mức giảm rất lớn của fraud capture.

Scope:

`DESCRIPTIVE COMPARATIVE EVIDENCE — LR-B04 ONLY`

Không suy rộng thành global training-window winner.

---

## 19.6. Decision Tree — controlled window comparison

Comparison:

`TW-DT-B01`

Config:

`DT-B01-DEFAULT-GINI-UNPRUNED`

Observed:

```text
W_SHORT
F1        = 0.3270564915758176
Recall    = 0.31368821292775667
Precision = 0.3416149068322981
TP / FP / FN / TN
330 / 636 / 722 / 710770
Alerts    = 966
Fit sec   = 6.661822540976573

W_LONG
F1        = 0.19913419913419914
Recall    = 0.1967680608365019
Precision = 0.20155793573515093
TP / FP / FN / TN
207 / 820 / 845 / 710586
Alerts    = 1027
Fit sec   = 116.46980562497629
```

Delta:

```text
ΔF1        = -0.12792229244161848
ΔRecall    = -0.11692015209125478
ΔPrecision = -0.14005697109714718

ΔTP     = -123
ΔFP     = +184
ΔFN     = +123
ΔAlerts = +61

Fit ratio LONG / SHORT:
17.483174447919577
```

Pair-specific leaders:

```text
Provisional F1 leader:
W_SHORT

Recall leader:
W_SHORT

Precision leader:
W_SHORT
```

Trade-off classification:

`SHORT_DOMINATES_F1_RECALL_PRECISION`

Interpretation:

Trong DT-B01, W_LONG vừa:

- bắt ít hơn 123 fraud;
- bỏ sót nhiều hơn 123 fraud;
- tạo thêm 184 false alerts;
- có F1/Recall/Precision đều thấp hơn W_SHORT.

Đây là controlled validation evidence thuận lợi cho W_SHORT trong DT-B01.

Scope:

`DESCRIPTIVE COMPARATIVE EVIDENCE — DT-B01 ONLY`

Không khóa global winner.

---

## 19.7. Random Forest — controlled window comparison

Comparison:

`TW-RF-B01`

Config:

`RF-B01-100-GINI-SQRT-BOOTSTRAP`

Observed:

```text
W_SHORT
F1        = 0.3664670658682635
Recall    = 0.2908745247148289
Precision = 0.49514563106796117
TP / FP / FN / TN
306 / 312 / 746 / 711094
Alerts    = 618
Fit sec   = 40.69802879198687

W_LONG
F1        = 0.15270935960591134
Recall    = 0.08840304182509506
Precision = 0.5602409638554217
TP / FP / FN / TN
93 / 73 / 959 / 711333
Alerts    = 166
Fit sec   = 627.7140797079774
```

Delta:

```text
ΔF1        = -0.21375770626235216
ΔRecall    = -0.20247148288973382
ΔPrecision = +0.06509533278746049

ΔTP     = -213
ΔFP     = -239
ΔFN     = +213
ΔAlerts = -452

Fit ratio LONG / SHORT:
15.423697371592834
```

Pair-specific leaders:

```text
Provisional F1 leader:
W_SHORT

Recall leader:
W_SHORT

Precision leader:
W_LONG
```

Trade-off classification:

`SHORT_F1_RECALL__LONG_PRECISION`

Interpretation:

RF-B01 có trade-off khác LR/DT.

W_LONG có Precision cao hơn, nhưng:

- chỉ bắt 93 fraud thay vì 306;
- bỏ sót 959 fraud thay vì 746;
- chỉ phát 166 alerts thay vì 618.

Vì vậy Precision advantage của W_LONG phải được đọc cùng Recall, F1 và alert volume.

Theo primary F1 evidence:

`W_SHORT = PROVISIONAL F1 LEADER`

Scope:

`DESCRIPTIVE COMPARATIVE EVIDENCE — RF-B01 ONLY`

Không khóa final winner.

---

## 19.8. Trade-off classification

Observed:

```text
TW-LR-B04
→ SHORT_DOMINATES_F1_RECALL_PRECISION

TW-DT-B01
→ SHORT_DOMINATES_F1_RECALL_PRECISION

TW-RF-B01
→ SHORT_F1_RECALL__LONG_PRECISION
```

Runtime gate:

`M6.4 TRADE-OFF CLASSIFICATION GATE: PASS`

Interpretation:

Hai family LR/DT cho cùng hướng trên cả F1, Recall và Precision.

RF tạo trade-off:

```text
W_SHORT:
F1 + Recall advantage

W_LONG:
Precision advantage
```

Status:

`VERIFIED`

---

## 19.9. Cross-pair direction

Observed:

```text
Provisional F1 leaders:
['W_SHORT', 'W_SHORT', 'W_SHORT']

Cross-pair direction:
CONSISTENT_PROVISIONAL_DIRECTION

Common provisional direction:
W_SHORT
```

Interpretation:

Ở **cả ba baseline model/config đã khóa**, W_SHORT là provisional F1 leader trên cùng canonical validation population.

Đây là evidence nhất quán ở baseline stage.

Tuy nhiên M6.4 không chuyển evidence này thành:

`FINAL TRAINING-WINDOW WINNER`

vì selection/robustness boundary vẫn phải được giữ.

Status:

`CONSISTENT PROVISIONAL DIRECTION — W_SHORT`

---

## 19.10. Computational context

Observed fit-time ratios:

```text
Logistic Regression:
LONG / SHORT ≈ 4.40x

Decision Tree:
LONG / SHORT ≈ 17.48x

Random Forest:
LONG / SHORT ≈ 15.42x
```

Interpretation:

W_LONG đòi hỏi fit-time lớn hơn trong cả ba baseline families.

Đây là:

`COMPUTATIONAL SUPPORTING EVIDENCE`

không phải predictive winner rule.

Không được kết luận:

`W_SHORT là final winner chỉ vì chạy nhanh hơn`.

Status:

`VERIFIED`

---

## 19.11. No arbitrary F1 epsilon

Observed:

```text
Arbitrary F1 epsilon used:
False
```

Không có rule kiểu:

```text
|ΔF1| < 0.01
→ tie
```

Runtime gate:

`M6.4 SELECTION-BOUNDARY GATE: PASS`

Status:

`VERIFIED`

---

## 19.12. No tuning / no threshold contamination

Observed:

```text
Window-specific tuning:
NONE

Threshold optimization:
NONE

Baseline threshold policy:
UNCHANGED
```

Không có experiment can thiệp sau khi nhìn pair result.

Status:

`VERIFIED`

---

## 19.13. Selection boundary

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

M6.4 final selection authorized:
FALSE
```

Interpretation:

Notebook giữ đúng ranh giới:

`comparative evidence ≠ final selection`

Status:

`VERIFIED`

---

## 19.14. Persistence / lineage

Persisted:

```text
data/processed/m6_04_training_window_comparison/
    m6_04_training_window_comparison.json
    m6_04_comparison_manifest.json
```

Runtime gates:

```text
M6.4 COMPARISON PERSISTENCE GATE:
PASS

M6.4 COMPARISON ROUND-TRIP GATE:
PASS
```

Recorded source hashes:

```text
M6.2 registry SHA256:
9e112883d5aa1358c58c714edc0f0ef7913252e94c56f3c6b28d3ba1b7f4e6fb

M6.3 analysis SHA256:
290f348ba20a4a19f659e24bb517ca8cce2229605af3f7ee762520e5b1d7b168
```

Status:

`VERIFIED`

---

## 19.15. FINAL TEST isolation

Observed:

`M6.4 FINAL TEST ISOLATION GATE: PASS`

No FINAL TEST artifact was used in training-window comparison.

Status:

`VERIFIED`

---

## 19.16. Overall technical result

Observed:

```text
G01_M6_03_HANDOFF               → PASS
G02_PAIR_DEFINITION             → PASS
G03_CONTROLLED_PAIR_INTEGRITY   → PASS
G04_COMPUTATIONAL_CONTEXT       → PASS
G05_COMPARISON_BUILD            → PASS
G06_THREE_PAIR_EVIDENCE         → PASS
G07_TRADEOFF_CLASSIFICATION     → PASS
G08_CROSS_PAIR_DIRECTION        → PASS
G09_NO_ARBITRARY_EPSILON        → PASS
G10_SELECTION_BOUNDARY          → PASS
G11_SOURCE_LINEAGE              → PASS
G12_COMPARISON_PERSISTENCE      → PASS
G13_COMPARISON_ROUND_TRIP       → PASS
G14_FINAL_TEST_ISOLATION        → PASS
```

Overall:

`M6.4 OVERALL TECHNICAL GATE: PASS`

Blocking issue:

`NONE`

---

# 20. Findings M6.4

## M6.4-F01 — All three comparisons are valid controlled pairs

Evidence:

12 controlled fields match in all three families.

Status:

`VERIFIED`

---

## M6.4-F02 — LR-B04 favors W_SHORT on all three fraud metrics

Evidence:

```text
ΔF1        = -0.2975118047746594
ΔRecall    = -0.22433460076045625
ΔPrecision = -0.09190091130791939
```

Finding:

W_SHORT is the pair-specific provisional leader for F1, Recall and Precision.

Status:

`DESCRIPTIVE COMPARATIVE EVIDENCE`

---

## M6.4-F03 — LR-LONG emits far fewer alerts but misses substantially more fraud

Evidence:

```text
ΔTP = -236
ΔFN = +236
ΔAlerts = -428
```

Finding:

Lower LR-LONG alert volume is accompanied by severe loss of fraud coverage.

Status:

`DESCRIPTIVE COMPARATIVE EVIDENCE`

---

## M6.4-F04 — DT-B01 favors W_SHORT on F1, Recall and Precision

Evidence:

```text
ΔF1        = -0.12792229244161848
ΔRecall    = -0.11692015209125478
ΔPrecision = -0.14005697109714718
```

Status:

`DESCRIPTIVE COMPARATIVE EVIDENCE`

---

## M6.4-F05 — DT-LONG has both higher FP and higher FN

Evidence:

```text
ΔFP = +184
ΔFN = +123
```

Finding:

Under DT-B01, W_LONG produces more false alerts while also missing more fraud than W_SHORT.

Status:

`DESCRIPTIVE COMPARATIVE EVIDENCE`

---

## M6.4-F06 — RF-B01 shows a Precision-versus-Recall trade-off

Evidence:

```text
ΔF1        = -0.21375770626235216
ΔRecall    = -0.20247148288973382
ΔPrecision = +0.06509533278746049
```

Finding:

W_LONG has higher Precision, while W_SHORT has higher F1 and Recall.

Status:

`DESCRIPTIVE COMPARATIVE EVIDENCE`

---

## M6.4-F07 — RF-LONG Precision advantage comes with much lower fraud coverage

Evidence:

```text
W_SHORT:
306 TP
746 FN
618 alerts

W_LONG:
93 TP
959 FN
166 alerts
```

Status:

`DESCRIPTIVE COMPARATIVE EVIDENCE`

---

## M6.4-F08 — W_SHORT is provisional F1 leader in all three baseline families

Evidence:

```text
LR:
W_SHORT

DT:
W_SHORT

RF:
W_SHORT
```

Cross-pair diagnostic:

`CONSISTENT_PROVISIONAL_DIRECTION`

Status:

`VERIFIED COMPARATIVE PATTERN`

---

## M6.4-F09 — W_SHORT is not yet locked as final training-window winner

Finding:

Consistent baseline direction is evidence for later selection, but M6.4 does not perform final robustness/selection.

Status:

`CORRECT BOUNDARY`

---

## M6.4-F10 — W_LONG has substantially higher fit cost in all three families

Evidence:

```text
LR:
~4.40x

DT:
~17.48x

RF:
~15.42x
```

Status:

`OBSERVED COMPUTATIONAL EVIDENCE`

---

## M6.4-F11 — No arbitrary tie threshold was introduced

Status:

`VERIFIED`

---

## M6.4-F12 — FINAL TEST remained isolated

Status:

`VERIFIED`

---

## M6.4-F13 — M6.5 handoff is unblocked

M6.5 can now perform same-window cross-model comparison using:

```text
W_SHORT:
LR vs DT vs RF

W_LONG:
LR vs DT vs RF
```

without conflating model family and training-window effects.

Status:

`READY FOR M6.5`

# 21. Decision Log M6.4 — sau runtime review

## M6.4-D01 — Comparison target

Decision:

Compare classifier-training window.

Status:

`INHERITED — LOCKED`

---

## M6.4-D02 — Official candidates

Decision:

```text
W_SHORT
W_LONG
```

Status:

`INHERITED — LOCKED`

---

## M6.4-D03 — Controlled-pair rule

Decision:

Within each model family:

`ONLY TRAINING WINDOW CHANGES`

Observed:

3 / 3 pair integrity checks PASS.

Status:

`INHERITED — VERIFIED — LOCKED`

---

## M6.4-D04 — Validation

Decision:

Same canonical validation:

```text
2019-01 → 2019-05
712,458 rows
1,052 fraud
```

Status:

`INHERITED — VERIFIED — LOCKED`

---

## M6.4-D05 — Primary comparison metric

Decision:

`F1_fraud`

Status:

`INHERITED — LOCKED`

---

## M6.4-D06 — Mandatory secondary evidence

Decision:

Read with F1:

```text
Recall
Precision
TP / FP / FN / TN
predicted-positive count/rate
Accuracy reference
```

Observed:

available for all three pairs.

Status:

`INHERITED — VERIFIED — LOCKED`

---

## M6.4-D07 — Delta convention

Decision:

`W_LONG − W_SHORT`

Observed:

used consistently.

Status:

`LOCKED / VERIFIED`

---

## M6.4-D08 — LR provisional direction

Decision:

Under LR-B04 baseline evidence:

```text
F1 leader:
W_SHORT

Recall leader:
W_SHORT

Precision leader:
W_SHORT
```

Status:

`DESCRIPTIVE COMPARATIVE EVIDENCE`

---

## M6.4-D09 — DT provisional direction

Decision:

Under DT-B01 baseline evidence:

```text
F1 leader:
W_SHORT

Recall leader:
W_SHORT

Precision leader:
W_SHORT
```

Status:

`DESCRIPTIVE COMPARATIVE EVIDENCE`

---

## M6.4-D10 — RF provisional direction

Decision:

Under RF-B01 baseline evidence:

```text
F1 leader:
W_SHORT

Recall leader:
W_SHORT

Precision leader:
W_LONG
```

Status:

`DESCRIPTIVE COMPARATIVE EVIDENCE`

---

## M6.4-D11 — Cross-pair F1 direction

Decision:

Observed:

```text
LR:
W_SHORT

DT:
W_SHORT

RF:
W_SHORT
```

Classification:

`CONSISTENT_PROVISIONAL_DIRECTION — W_SHORT`

Status:

`VERIFIED COMPARATIVE PATTERN`

---

## M6.4-D12 — Arbitrary epsilon

Decision:

No arbitrary ΔF1 epsilon.

Observed:

`False`

Status:

`INHERITED — VERIFIED — LOCKED`

---

## M6.4-D13 — Computational context

Decision:

Use runtime only as supporting evidence.

Observed:

W_LONG fit time > W_SHORT fit time in all 3 pairs.

Status:

`VERIFIED`

---

## M6.4-D14 — Window-specific tuning

Decision:

None.

Status:

`INHERITED — VERIFIED — LOCKED`

---

## M6.4-D15 — Threshold

Decision:

Same baseline threshold policy.

No optimization.

Status:

`INHERITED — VERIFIED — LOCKED`

---

## M6.4-D16 — Final global training-window winner

Decision:

Do not lock in M6.4.

Observed:

W_SHORT is consistent provisional F1 leader across all three baseline families.

Final status:

`OPEN — HANDOFF M7 IF ROBUSTNESS / SELECTION IS REQUIRED`

---

## M6.4-D17 — Cross-model family selection

Decision:

Not performed in M6.4.

Status:

`DEFERRED TO M6.5 / M7`

---

## M6.4-D18 — Comparison artifact

Decision:

Persist:

`m6_04_training_window_comparison.json`

Observed:

Persistence + round-trip PASS.

Status:

`LOCKED / VERIFIED`

---

## M6.4-D19 — Comparison manifest

Decision:

Persist:

`m6_04_comparison_manifest.json`

Observed:

Persistence + round-trip PASS.

Status:

`LOCKED / VERIFIED`

---

## M6.4-D20 — FINAL TEST

Decision:

No access.

Observed:

FINAL TEST isolation PASS.

Status:

`INHERITED — VERIFIED — LOCKED`

---

## M6.4-D21 — M6.5 handoff

Decision:

Use same-window comparisons only:

```text
W_SHORT:
LR vs DT vs RF

W_LONG:
LR vs DT vs RF
```

Status:

`READY`

# 22. M6.4 Gate

## Technical runtime gates

```text
G01_M6_03_HANDOFF               → PASS
G02_PAIR_DEFINITION             → PASS
G03_CONTROLLED_PAIR_INTEGRITY   → PASS
G04_COMPUTATIONAL_CONTEXT       → PASS
G05_COMPARISON_BUILD            → PASS
G06_THREE_PAIR_EVIDENCE         → PASS
G07_TRADEOFF_CLASSIFICATION     → PASS
G08_CROSS_PAIR_DIRECTION        → PASS
G09_NO_ARBITRARY_EPSILON        → PASS
G10_SELECTION_BOUNDARY          → PASS
G11_SOURCE_LINEAGE              → PASS
G12_COMPARISON_PERSISTENCE      → PASS
G13_COMPARISON_ROUND_TRIP       → PASS
G14_FINAL_TEST_ISOLATION        → PASS
```

Technical gates:

`14 / 14 PASS`

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

## R02 — M6.3 handoff valid?

Evidence:

M6.2 registry + M6.3 analysis/manifest loaded and verified.

Result:

`PASS`

---

## R03 — Controlled pairs valid?

Evidence:

```text
3 / 3 pairs
12 / 12 controlled fields per pair
```

Result:

`PASS`

---

## R04 — LR comparison complete?

Evidence:

F1 / Recall / Precision / TP / FP / FN / alerts / runtime all available.

Result:

`PASS`

---

## R05 — DT comparison complete?

Evidence:

Complete canonical evidence bundle.

Result:

`PASS`

---

## R06 — RF comparison complete?

Evidence:

Complete canonical evidence bundle, including Precision-vs-Recall trade-off.

Result:

`PASS`

---

## R07 — Pair-specific provisional F1 leaders identified?

Evidence:

```text
LR:
W_SHORT

DT:
W_SHORT

RF:
W_SHORT
```

Result:

`PASS`

---

## R08 — Secondary trade-offs reviewed?

Evidence:

LR/DT show SHORT dominance across F1/Recall/Precision.

RF shows SHORT F1/Recall advantage and LONG Precision advantage.

Result:

`PASS`

---

## R09 — Confusion-count deltas interpreted?

Evidence:

TP / FP / FN / alert deltas are available and consistent.

Result:

`PASS`

---

## R10 — Computational context reviewed?

Evidence:

LONG/SHORT fit-time ratios recorded for all pairs.

Result:

`PASS`

---

## R11 — No arbitrary epsilon / tuning / threshold contamination?

Evidence:

```text
arbitrary epsilon:
False

window-specific tuning:
None

threshold optimization:
None
```

Result:

`PASS`

---

## R12 — Final-selection boundary preserved?

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

Comparison persistence + round-trip PASS.

FINAL TEST isolation PASS.

Result:

`PASS`

---

## Overall M6.4 Gate

```text
Technical gates:
14 / 14 PASS

Runtime review gates:
13 / 13 PASS

Blocking issue:
NONE
```

Final:

`M6.4 — PASS`

Handoff:

`READY FOR M6.5`

# 23. Kết luận M6.4

M6.4 đã hoàn thành controlled training-window comparison cho cả ba official baseline model families.

Controlled protocol:

```text
ONLY TRAINING WINDOW CHANGES
```

Canonical validation:

```text
2019-01 → 2019-05

Rows:
712,458

Fraud:
1,052
```

Comparison pairs:

```text
LR:
W_SHORT vs W_LONG

DT:
W_SHORT vs W_LONG

RF:
W_SHORT vs W_LONG
```

Observed pair-specific F1 direction:

```text
LR:
W_SHORT provisional F1 leader

DT:
W_SHORT provisional F1 leader

RF:
W_SHORT provisional F1 leader
```

Cross-pair diagnostic:

```text
CONSISTENT_PROVISIONAL_DIRECTION

Common direction:
W_SHORT
```

Trade-off structure:

```text
LR:
W_SHORT leads F1 / Recall / Precision

DT:
W_SHORT leads F1 / Recall / Precision

RF:
W_SHORT leads F1 / Recall
W_LONG leads Precision
```

Computational context:

```text
LONG fit-time > SHORT fit-time
for all three baseline families
```

M6.4 đã xác minh:

```text
Controlled pair integrity:
PASS

Delta convention:
W_LONG - W_SHORT

No arbitrary F1 epsilon:
PASS

Window-specific tuning:
NONE

Threshold optimization:
NONE

Comparison persistence:
PASS

Comparison round-trip:
PASS

FINAL TEST:
PROTECTED
```

M6.4 chưa khóa:

```text
Final training-window winner:
OPEN

Model-family winner:
OPEN

Final model:
OPEN

Final threshold:
OPEN
```

Lý do:

M6.4 cung cấp comparative baseline evidence.

Nếu project cần final training-window selection với robustness/tie-break protocol, việc đó thuộc giai đoạn selection phù hợp, đặc biệt M7.

Final state:

```text
M6.4 — PASS

Execution Integrity:
VERIFIED

Controlled Pairs:
3 / 3 VERIFIED

Controlled Fields:
12 / 12 per pair

LR Provisional F1 Direction:
W_SHORT

DT Provisional F1 Direction:
W_SHORT

RF Provisional F1 Direction:
W_SHORT

Cross-pair Direction:
CONSISTENT PROVISIONAL W_SHORT

Trade-off Review:
VERIFIED

Computational Context:
VERIFIED

Final Training-window Winner:
OPEN

FINAL TEST:
PROTECTED

Blocking Issue:
NONE

READY FOR M6.5
```

Bước tiếp theo:

`M6.5 — Cross-model comparative evaluation`

M6.5 phải giữ training window cố định khi so model family:

```text
W_SHORT:
LR vs DT vs RF

W_LONG:
LR vs DT vs RF
```

Không dùng comparison chéo như:

`LR-SHORT vs RF-LONG`

để kết luận về model family.
