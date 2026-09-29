# M5.3 — Logistic Regression baseline B04

Milestone:

`M5 — Modeling baseline`

Substep:

`M5.3 — Logistic Regression baseline`

Official candidate runs:

`M5-LR-SHORT-B04`

`M5-LR-LONG-B04`

Lịch sử technical candidates:

```text
B01:
SAGA + max_iter=200
→ ConvergenceWarning
→ FAILED_CONVERGENCE

B02:
SAGA + max_iter=1000
→ W_LONG chạy >25 phút chưa hoàn thành
→ ABORTED / COMPUTATIONALLY INEFFICIENT

B03:
newton-cholesky
→ W_SHORT fit ≈ 2.166 giây
→ n_iter_ = 15
→ LinAlgWarning:
   Hessian singular / very ill-conditioned
→ sklearn tự fallback sang lbfgs
→ FAILED_NUMERICAL_WARNING
```

B04 dùng trực tiếp:

`solver = lbfgs`

Quyết định này dựa trên technical optimization evidence của B03, không dựa trên validation metric.

B04 vẫn giữ nguyên:

- canonical 47-column CSR representation;
- `C=1.0`;
- L2-equivalent `l1_ratio=0.0`;
- `class_weight=None`;
- `IMBALANCE_STRATEGY=NONE`;
- cùng VALIDATION;
- cùng canonical metric implementation;
- cùng default-decision threshold policy;
- không dùng FINAL TEST.

## 1. B04 exact configuration — khóa trước result

```text
Estimator:
sklearn.linear_model.LogisticRegression

solver:
lbfgs

C:
1.0

l1_ratio:
0.0

max_iter:
200

tol:
1e-4

fit_intercept:
True

class_weight:
None

random_state:
42
```

`l1_ratio=0.0` giữ L2 regularization intent trong scikit-learn runtime hiện tại mà không truyền explicit deprecated `penalty="l2"`.

B04 không thay C, feature, preprocessing, class distribution, validation population, metric hay threshold policy.

`max_iter=200` là technical iteration budget của L-BFGS. Nếu warning xuất hiện hoặc `n_iter_` chạm 200:

`STOP — không diễn giải metric`.

## 2. Runtime order

```text
load + integrity checks
        ↓
freeze B04 config
        ↓
W_SHORT official preflight
        ↓
warning / convergence gate
        ↓
nếu PASS
        ↓
W_LONG official run
        ↓
warning / convergence gate
        ↓
controlled-pair gate
        ↓
probability/default-rule gate
        ↓
persist + round-trip
        ↓
FINAL TEST isolation
        ↓
overall technical gate
```

W_SHORT chạy trước để phát hiện lỗi kỹ thuật sớm.

Không được dùng W_SHORT score để sửa B04 config trước W_LONG.


```python

from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import hashlib
import json
import platform
import sys
import time
import warnings

import numpy as np

try:
    from scipy import sparse
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Thiếu scipy. Chạy `%pip install scipy scikit-learn`, "
        "Restart Kernel rồi Run All."
    ) from exc

try:
    import sklearn
    from sklearn.base import clone
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Thiếu scikit-learn. Chạy `%pip install scipy scikit-learn`, "
        "Restart Kernel rồi Run All."
    ) from exc


print("Python:")
print(sys.version)

print("\nPlatform:")
print(platform.platform())

print("\nNumPy:")
print(np.__version__)

print("\nscikit-learn:")
print(sklearn.__version__)

```

    Python:
    3.14.6 (main, Jun 10 2026, 10:03:53) [Clang 21.0.0 (clang-2100.0.123.102)]
    
    Platform:
    macOS-26.6.2-arm64-arm-64bit-Mach-O
    
    NumPy:
    2.5.3
    
    scikit-learn:
    1.9.1


## 3. Locate canonical artifacts và M5.2 contract


```python

M4_ARTIFACT_RELATIVE_DIR = (
    Path("data")
    / "processed"
    / "m4_07_baseline_ready"
)

M5_02_CONTRACT_RELATIVE_PATH = (
    Path("data")
    / "processed"
    / "m5_02_modeling_contract"
    / "m5_02_modeling_contract.json"
)

M5_03_OUTPUT_RELATIVE_DIR = (
    Path("data")
    / "processed"
    / "m5_03_logistic_regression_baseline_b04"
)


candidate_roots = [
    Path.cwd(),
    *list(Path.cwd().parents)[:5],
]

PROJECT_ROOT = None

for candidate in candidate_roots:
    candidate = candidate.resolve()

    if (
        (
            candidate
            / M4_ARTIFACT_RELATIVE_DIR
            / "manifest.json"
        ).exists()
        and
        (
            candidate
            / M5_02_CONTRACT_RELATIVE_PATH
        ).exists()
    ):
        PROJECT_ROOT = candidate
        break


if PROJECT_ROOT is None:
    raise FileNotFoundError(
        "Không tìm thấy PROJECT_ROOT chứa "
        "M4.7 manifest và M5.2 contract."
    )


ARTIFACT_DIR = PROJECT_ROOT / M4_ARTIFACT_RELATIVE_DIR
M5_02_CONTRACT_PATH = PROJECT_ROOT / M5_02_CONTRACT_RELATIVE_PATH
OUTPUT_DIR = PROJECT_ROOT / M5_03_OUTPUT_RELATIVE_DIR

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

print("PROJECT_ROOT:")
print(PROJECT_ROOT)

print("\nARTIFACT_DIR:")
print(ARTIFACT_DIR)

print("\nOUTPUT_DIR:")
print(OUTPUT_DIR)

```

    PROJECT_ROOT:
    /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening
    
    ARTIFACT_DIR:
    /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/processed/m4_07_baseline_ready
    
    OUTPUT_DIR:
    /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/processed/m5_03_logistic_regression_baseline_b04


## 4. Canonical constants


```python

EXPECTED_OUTPUT_WIDTH = 47

EXPECTED_W_LONG_ROWS = 6_855_270
EXPECTED_W_SHORT_ROWS = 1_721_615
EXPECTED_VALIDATION_ROWS = 712_458

EXPECTED_W_LONG_FRAUD = 9_606
EXPECTED_W_SHORT_FRAUD = 2_491
EXPECTED_VALIDATION_FRAUD = 1_052

EXPECTED_NNZ = {
    "X_train_w_long": 61_900_193,
    "X_train_w_short": 15_542_381,
    "X_validation_w_long": 6_430_339,
    "X_validation_w_short": 6_430_339,
}

M4_PIPELINE_VERSION = "M4.7-baseline-v1"
M5_RUNNER_VERSION = "M5.2-shared-runner-v1"

IMBALANCE_STRATEGY = "NONE"
DEFAULT_RANDOM_STATE = 42
BASELINE_THRESHOLD_POLICY = "DEFAULT_MODEL_DECISION_RULE"

FEATURE_VERSION = "Feature Specification v1.0"
PREPROCESSING_VERSION = "Preprocessing Specification v1.0"
MATRIX_SCHEMA_VERSION = "Baseline Matrix Schema v1.0"

LR_MODEL_FAMILY = "Logistic Regression"
LR_MODEL_ID = "SKLEARN_LOGISTIC_REGRESSION"
LR_CONFIG_ID = "LR-B04-LBFGS-L2-C1"

LR_SHORT_EXPERIMENT_ID = "M5-LR-SHORT-B04"
LR_LONG_EXPERIMENT_ID = "M5-LR-LONG-B04"

```

## 5. Load contract + matrices


```python

with open(
    ARTIFACT_DIR / "manifest.json",
    "r",
    encoding="utf-8",
) as file:
    manifest = json.load(file)

with open(
    ARTIFACT_DIR / "feature_names.json",
    "r",
    encoding="utf-8",
) as file:
    feature_names = json.load(file)

with open(
    M5_02_CONTRACT_PATH,
    "r",
    encoding="utf-8",
) as file:
    m5_02_contract = json.load(file)


assert manifest["pipeline_version"] == M4_PIPELINE_VERSION
assert manifest["output_width"] == EXPECTED_OUTPUT_WIDTH
assert manifest["matrix_dtype"] == "float32"
assert str(manifest["matrix_format"]).upper() == "CSR"
assert manifest["target_dtype"] == "int8"
assert manifest["raw_identifiers_in_X"] is False
assert manifest["target_in_X"] is False
assert manifest["final_test_used"] is False

assert len(feature_names) == EXPECTED_OUTPUT_WIDTH
assert len(set(feature_names)) == EXPECTED_OUTPUT_WIDTH

assert m5_02_contract["runner_version"] == M5_RUNNER_VERSION
assert m5_02_contract["final_test_access_allowed"] is False
assert m5_02_contract["imbalance_strategy"] == "NONE"


load_start = time.perf_counter()

X_train_w_long = sparse.load_npz(
    ARTIFACT_DIR / "X_train_w_long.npz"
)

X_train_w_short = sparse.load_npz(
    ARTIFACT_DIR / "X_train_w_short.npz"
)

X_validation_w_long = sparse.load_npz(
    ARTIFACT_DIR / "X_validation_w_long.npz"
)

X_validation_w_short = sparse.load_npz(
    ARTIFACT_DIR / "X_validation_w_short.npz"
)

y_train_w_long = np.load(
    ARTIFACT_DIR / "y_train_w_long.npy",
    allow_pickle=False,
)

y_train_w_short = np.load(
    ARTIFACT_DIR / "y_train_w_short.npy",
    allow_pickle=False,
)

y_validation = np.load(
    ARTIFACT_DIR / "y_validation.npy",
    allow_pickle=False,
)


print(
    "Load seconds:",
    round(
        time.perf_counter() - load_start,
        2,
    ),
)

print("\nM5.3 B04 ARTIFACT LOAD: COMPLETE")

```

    Load seconds: 0.8
    
    M5.3 B04 ARTIFACT LOAD: COMPLETE


## 6. Input integrity gate


```python

matrices = {
    "X_train_w_long": X_train_w_long,
    "X_train_w_short": X_train_w_short,
    "X_validation_w_long": X_validation_w_long,
    "X_validation_w_short": X_validation_w_short,
}

expected_shapes = {
    "X_train_w_long": (
        EXPECTED_W_LONG_ROWS,
        EXPECTED_OUTPUT_WIDTH,
    ),
    "X_train_w_short": (
        EXPECTED_W_SHORT_ROWS,
        EXPECTED_OUTPUT_WIDTH,
    ),
    "X_validation_w_long": (
        EXPECTED_VALIDATION_ROWS,
        EXPECTED_OUTPUT_WIDTH,
    ),
    "X_validation_w_short": (
        EXPECTED_VALIDATION_ROWS,
        EXPECTED_OUTPUT_WIDTH,
    ),
}

for name, matrix in matrices.items():
    assert sparse.isspmatrix_csr(matrix)
    assert matrix.dtype == np.float32
    assert matrix.shape == expected_shapes[name]
    assert matrix.nnz == EXPECTED_NNZ[name]
    assert np.isfinite(matrix.data).all()

assert y_train_w_long.dtype == np.int8
assert y_train_w_short.dtype == np.int8
assert y_validation.dtype == np.int8

assert len(y_train_w_long) == EXPECTED_W_LONG_ROWS
assert len(y_train_w_short) == EXPECTED_W_SHORT_ROWS
assert len(y_validation) == EXPECTED_VALIDATION_ROWS

assert int(y_train_w_long.sum()) == EXPECTED_W_LONG_FRAUD
assert int(y_train_w_short.sum()) == EXPECTED_W_SHORT_FRAUD
assert int(y_validation.sum()) == EXPECTED_VALIDATION_FRAUD


print("M5.3 B04 INPUT INTEGRITY GATE: PASS")

```

    M5.3 B04 INPUT INTEGRITY GATE: PASS


## 7. Canonical metric + shared runner


```python

def compute_binary_metrics(
    y_true,
    y_pred,
):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    tn, fp, fn, tp = cm.ravel()

    predicted_positive_count = int(tp + fp)

    return {
        "f1_fraud": float(
            f1_score(
                y_true,
                y_pred,
                pos_label=1,
                zero_division=0,
            )
        ),
        "recall_fraud": float(
            recall_score(
                y_true,
                y_pred,
                pos_label=1,
                zero_division=0,
            )
        ),
        "precision_fraud": float(
            precision_score(
                y_true,
                y_pred,
                pos_label=1,
                zero_division=0,
            )
        ),
        "accuracy_reference": float(
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
                predicted_positive_count
                / len(y_true)
            ),
        "validation_rows":
            int(len(y_true)),
    }


def _json_safe_value(value):
    if isinstance(
        value,
        (str, int, float, bool, type(None)),
    ):
        return value

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, (list, tuple)):
        return [
            _json_safe_value(item)
            for item
            in value
        ]

    if isinstance(value, dict):
        return {
            str(key): _json_safe_value(item)
            for key, item
            in value.items()
        }

    return repr(value)


@dataclass
class ExperimentRunOutput:
    summary: Dict[str, Any]
    y_pred: np.ndarray
    risk_score: Optional[np.ndarray]
    warning_messages: List[str]


def run_lr_baseline(
    *,
    experiment_id,
    estimator,
    X_train,
    y_train,
    X_validation,
    y_validation,
    training_window_id,
):
    model = clone(estimator)

    with warnings.catch_warnings(
        record=True
    ) as caught_warnings:
        warnings.simplefilter("always")

        fit_start = time.perf_counter()

        model.fit(
            X_train,
            y_train,
        )

        fit_seconds = (
            time.perf_counter()
            - fit_start
        )

        pred_start = time.perf_counter()

        y_pred = model.predict(
            X_validation
        ).astype(
            np.int8
        )

        proba = model.predict_proba(
            X_validation
        )

        classes = np.asarray(
            model.classes_
        )

        positive_position = np.flatnonzero(
            classes == 1
        )

        assert len(positive_position) == 1

        risk_score = proba[
            :,
            positive_position[0],
        ].astype(
            np.float32
        )

        prediction_seconds = (
            time.perf_counter()
            - pred_start
        )

    warning_messages = [
        (
            f"{item.category.__name__}: "
            f"{item.message}"
        )
        for item
        in caught_warnings
    ]

    metrics = compute_binary_metrics(
        y_validation,
        y_pred,
    )

    model_params = {
        key: _json_safe_value(value)
        for key, value
        in model.get_params(
            deep=False
        ).items()
    }

    n_iter_values = [
        int(value)
        for value
        in np.asarray(
            model.n_iter_
        ).ravel()
    ]

    summary = {
        "runner_version":
            M5_RUNNER_VERSION,

        "experiment_id":
            experiment_id,

        "run_status":
            "COMPLETED",

        "model_family":
            LR_MODEL_FAMILY,

        "model_id":
            LR_MODEL_ID,

        "model_config_id":
            LR_CONFIG_ID,

        "model_class":
            model.__class__.__name__,

        "model_module":
            model.__class__.__module__,

        "model_params":
            model_params,

        "model_diagnostics": {
            "n_iter_":
                n_iter_values,

            "coef_l2_norm":
                float(
                    np.linalg.norm(
                        model.coef_
                    )
                ),

            "intercept":
                [
                    float(value)
                    for value
                    in np.asarray(
                        model.intercept_
                    ).ravel()
                ],
        },

        "training_window_id":
            training_window_id,

        "feature_version":
            FEATURE_VERSION,

        "preprocessing_version":
            PREPROCESSING_VERSION,

        "matrix_schema_version":
            MATRIX_SCHEMA_VERSION,

        "feature_count":
            int(
                X_train.shape[1]
            ),

        "training_rows":
            int(
                X_train.shape[0]
            ),

        "training_fraud_rows":
            int(
                np.asarray(
                    y_train
                ).sum()
            ),

        "validation_rows":
            int(
                X_validation.shape[0]
            ),

        "validation_fraud_rows":
            int(
                np.asarray(
                    y_validation
                ).sum()
            ),

        "imbalance_strategy":
            IMBALANCE_STRATEGY,

        "random_state":
            DEFAULT_RANDOM_STATE,

        "threshold_policy":
            BASELINE_THRESHOLD_POLICY,

        "fit_seconds":
            float(
                fit_seconds
            ),

        "prediction_seconds":
            float(
                prediction_seconds
            ),

        "risk_score_available":
            True,

        "risk_score_kind":
            "predict_proba",

        "warning_count":
            len(
                warning_messages
            ),

        "warnings":
            warning_messages,

        "errors":
            [],

        "final_test_accessed":
            False,

        "integrity_gate_result":
            "PASS",

        "decision":
            "OPEN — REQUIRES M5.3 RUNTIME REVIEW",

        "next_action":
            "DO NOT SELECT FINAL WINNER IN M5.3",

        **metrics,
    }

    return ExperimentRunOutput(
        summary=summary,
        y_pred=y_pred,
        risk_score=risk_score,
        warning_messages=warning_messages,
    )


def assert_clean_convergence(
    output,
    *,
    expected_max_iter,
):
    summary = output.summary

    for message in summary["warnings"]:
        print(" -", message)

    assert summary["warning_count"] == 0, (
        "Có warning cần review; "
        "không được diễn giải metric."
    )

    n_iter_values = summary[
        "model_diagnostics"
    ][
        "n_iter_"
    ]

    assert n_iter_values

    assert max(n_iter_values) < expected_max_iter, (
        "n_iter_ chạm max_iter; "
        "không được coi là converged."
    )

```

## 8. Define + freeze B04 config


```python

LR_BASELINE_CONFIG = {
    "C": 1.0,
    "solver": "lbfgs",
    "max_iter": 200,
    "tol": 1e-4,
    "fit_intercept": True,
    "class_weight": None,
    "random_state": DEFAULT_RANDOM_STATE,
    "l1_ratio": 0.0,
}


lr_template = LogisticRegression(
    **LR_BASELINE_CONFIG
)


actual_params = lr_template.get_params(
    deep=False
)

for key, expected in (
    LR_BASELINE_CONFIG.items()
):
    assert actual_params[key] == expected


config_lock_payload = {
    "m5_substep": "M5.3-B04",
    "model_family": LR_MODEL_FAMILY,
    "model_id": LR_MODEL_ID,
    "model_config_id": LR_CONFIG_ID,
    "config": LR_BASELINE_CONFIG,
    "reason_for_b04": (
        "B01 failed convergence; "
        "B02 saga was computationally inefficient "
        "after >25 minutes on W_LONG; "
        "B04 newton-cholesky emitted LinAlgWarning "
        "and internally fell back to lbfgs on W_SHORT."
    ),
    "validation_score_used_to_choose_b04": False,
    "imbalance_strategy": IMBALANCE_STRATEGY,
    "threshold_policy": BASELINE_THRESHOLD_POLICY,
    "final_test_access": False,
}


canonical = json.dumps(
    config_lock_payload,
    ensure_ascii=False,
    sort_keys=True,
    separators=(",", ":"),
)

fingerprint = hashlib.sha256(
    canonical.encode("utf-8")
).hexdigest()

config_lock_payload[
    "config_fingerprint_sha256"
] = fingerprint


CONFIG_PATH = (
    OUTPUT_DIR
    / "m5_03_lr_b04_config_lock.json"
)


with open(
    CONFIG_PATH,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        config_lock_payload,
        file,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )


print("B04 config:")
for key in sorted(LR_BASELINE_CONFIG):
    print(
        key,
        "=",
        LR_BASELINE_CONFIG[key],
    )

print("\nConfig fingerprint:")
print(fingerprint)

print("\nM5.3 B04 PRE-RESULT CONFIG LOCK: PASS")

```

    B04 config:
    C = 1.0
    class_weight = None
    fit_intercept = True
    l1_ratio = 0.0
    max_iter = 200
    random_state = 42
    solver = lbfgs
    tol = 0.0001
    
    Config fingerprint:
    b60b88e457ee01b6b84e70f20108d00539c288e6bef95050a6ab16fe4d3ff776
    
    M5.3 B04 PRE-RESULT CONFIG LOCK: PASS


# 9. W_SHORT preflight official run

Đây vẫn là official B04 run.

Nếu W_SHORT có warning/convergence issue:

`STOP`

không chạy W_LONG.


```python

print("=" * 72)
print("Starting:", LR_SHORT_EXPERIMENT_ID)
print("Training rows:", f"{EXPECTED_W_SHORT_ROWS:,}")
print("Training fraud:", f"{EXPECTED_W_SHORT_FRAUD:,}")


short_output = run_lr_baseline(
    experiment_id=
        LR_SHORT_EXPERIMENT_ID,
    estimator=
        lr_template,
    X_train=
        X_train_w_short,
    y_train=
        y_train_w_short,
    X_validation=
        X_validation_w_short,
    y_validation=
        y_validation,
    training_window_id=
        "W_SHORT",
)


print("\nCompleted:", LR_SHORT_EXPERIMENT_ID)
print(
    "fit_seconds:",
    round(
        short_output.summary[
            "fit_seconds"
        ],
        3,
    ),
)
print(
    "prediction_seconds:",
    round(
        short_output.summary[
            "prediction_seconds"
        ],
        3,
    ),
)
print(
    "warning_count:",
    short_output.summary[
        "warning_count"
    ],
)
print(
    "n_iter_:",
    short_output.summary[
        "model_diagnostics"
    ][
        "n_iter_"
    ],
)


assert_clean_convergence(
    short_output,
    expected_max_iter=
        LR_BASELINE_CONFIG[
            "max_iter"
        ],
)


print(
    "\nM5.3 B04 W_SHORT CONVERGENCE GATE: PASS"
)

```

    ========================================================================
    Starting: M5-LR-SHORT-B04
    Training rows: 1,721,615
    Training fraud: 2,491
    
    Completed: M5-LR-SHORT-B04
    fit_seconds: 0.822
    prediction_seconds: 0.032
    warning_count: 0
    n_iter_: [19]
    
    M5.3 B04 W_SHORT CONVERGENCE GATE: PASS


# 10. W_LONG official run

Chỉ chạy khi W_SHORT đã PASS convergence gate.

Config không được thay đổi.


```python

print("=" * 72)
print("Starting:", LR_LONG_EXPERIMENT_ID)
print("Training rows:", f"{EXPECTED_W_LONG_ROWS:,}")
print("Training fraud:", f"{EXPECTED_W_LONG_FRAUD:,}")


long_output = run_lr_baseline(
    experiment_id=
        LR_LONG_EXPERIMENT_ID,
    estimator=
        lr_template,
    X_train=
        X_train_w_long,
    y_train=
        y_train_w_long,
    X_validation=
        X_validation_w_long,
    y_validation=
        y_validation,
    training_window_id=
        "W_LONG",
)


print("\nCompleted:", LR_LONG_EXPERIMENT_ID)
print(
    "fit_seconds:",
    round(
        long_output.summary[
            "fit_seconds"
        ],
        3,
    ),
)
print(
    "prediction_seconds:",
    round(
        long_output.summary[
            "prediction_seconds"
        ],
        3,
    ),
)
print(
    "warning_count:",
    long_output.summary[
        "warning_count"
    ],
)
print(
    "n_iter_:",
    long_output.summary[
        "model_diagnostics"
    ][
        "n_iter_"
    ],
)


assert_clean_convergence(
    long_output,
    expected_max_iter=
        LR_BASELINE_CONFIG[
            "max_iter"
        ],
)


print(
    "\nM5.3 B04 W_LONG CONVERGENCE GATE: PASS"
)

```

    ========================================================================
    Starting: M5-LR-LONG-B04
    Training rows: 6,855,270
    Training fraud: 9,606
    
    Completed: M5-LR-LONG-B04
    fit_seconds: 3.614
    prediction_seconds: 0.025
    warning_count: 0
    n_iter_: [22]
    
    M5.3 B04 W_LONG CONVERGENCE GATE: PASS


## 11. Controlled-pair comparability


```python

short_summary = short_output.summary
long_summary = long_output.summary


controlled_fields = [
    "runner_version",
    "model_family",
    "model_id",
    "model_config_id",
    "model_class",
    "model_module",
    "model_params",
    "feature_version",
    "preprocessing_version",
    "matrix_schema_version",
    "feature_count",
    "validation_rows",
    "validation_fraud_rows",
    "imbalance_strategy",
    "random_state",
    "threshold_policy",
    "risk_score_kind",
    "final_test_accessed",
]


for field in controlled_fields:
    assert (
        short_summary[field]
        == long_summary[field]
    ), (
        f"Controlled field mismatch: {field}"
    )


assert short_summary[
    "training_window_id"
] == "W_SHORT"

assert long_summary[
    "training_window_id"
] == "W_LONG"


print(
    "M5.3 B04 CONTROLLED-PAIR GATE: PASS"
)

```

    M5.3 B04 CONTROLLED-PAIR GATE: PASS


## 12. Probability/default-rule integrity


```python

for output in [
    short_output,
    long_output,
]:
    score = output.risk_score

    assert score is not None
    assert np.isfinite(score).all()
    assert np.all(score >= 0.0)
    assert np.all(score <= 1.0)

    default_pred = (
        score >= 0.5
    ).astype(
        np.int8
    )

    assert np.array_equal(
        default_pred,
        output.y_pred,
    )


print(
    "M5.3 B04 PROBABILITY / DEFAULT-RULE GATE: PASS"
)

```

    M5.3 B04 PROBABILITY / DEFAULT-RULE GATE: PASS


## 13. Baseline evidence — descriptive only


```python

def print_summary(summary):
    print("Experiment:", summary["experiment_id"])
    print("Window:", summary["training_window_id"])
    print("Training rows:", f"{summary['training_rows']:,}")
    print("Training fraud:", f"{summary['training_fraud_rows']:,}")
    print("n_iter_:", summary["model_diagnostics"]["n_iter_"])
    print("fit_seconds:", summary["fit_seconds"])
    print("F1_fraud:", summary["f1_fraud"])
    print("Recall_fraud:", summary["recall_fraud"])
    print("Precision_fraud:", summary["precision_fraud"])
    print("Accuracy:", summary["accuracy_reference"])
    print(
        "TP/FP/FN/TN:",
        summary["tp"],
        summary["fp"],
        summary["fn"],
        summary["tn"],
    )
    print(
        "Predicted positive count:",
        summary["predicted_positive_count"],
    )
    print(
        "Predicted positive rate:",
        summary["predicted_positive_rate"],
    )


print("=" * 72)
print_summary(short_summary)

print("\n" + "=" * 72)
print_summary(long_summary)

print(
    "\nTraining-window winner:",
    "OPEN — M5.3 DOES NOT SELECT",
)

```

    ========================================================================
    Experiment: M5-LR-SHORT-B04
    Window: W_SHORT
    Training rows: 1,721,615
    Training fraud: 2,491
    n_iter_: [19]
    fit_seconds: 0.8222020840039477
    F1_fraud: 0.33747547416612167
    Recall_fraud: 0.24524714828897337
    Precision_fraud: 0.5408805031446541
    Accuracy: 0.9985781618004149
    TP/FP/FN/TN: 258 219 794 711187
    Predicted positive count: 477
    Predicted positive rate: 0.0006695131502488568
    
    ========================================================================
    Experiment: M5-LR-LONG-B04
    Window: W_LONG
    Training rows: 6,855,270
    Training fraud: 9,606
    n_iter_: [22]
    fit_seconds: 3.6143539589829743
    F1_fraud: 0.03996366939146231
    Recall_fraud: 0.02091254752851711
    Precision_fraud: 0.4489795918367347
    Accuracy: 0.9985164037739769
    TP/FP/FN/TN: 22 27 1030 711379
    Predicted positive count: 49
    Predicted positive rate: 6.877598398782806e-05
    
    Training-window winner: OPEN — M5.3 DOES NOT SELECT


## 14. Persist official B04 evidence


```python

def save_output(
    output,
):
    exp_id = output.summary[
        "experiment_id"
    ]

    pred_path = (
        OUTPUT_DIR
        / f"{exp_id}__y_pred.npy"
    )

    score_path = (
        OUTPUT_DIR
        / f"{exp_id}__risk_score.npy"
    )

    summary_path = (
        OUTPUT_DIR
        / f"{exp_id}__summary.json"
    )

    np.save(
        pred_path,
        output.y_pred,
        allow_pickle=False,
    )

    np.save(
        score_path,
        output.risk_score,
        allow_pickle=False,
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output.summary,
            file,
            ensure_ascii=False,
            indent=2,
        )

    return {
        "prediction": pred_path,
        "risk_score": score_path,
        "summary": summary_path,
    }


saved = {
    "W_SHORT": save_output(short_output),
    "W_LONG": save_output(long_output),
}


for window_id, paths in saved.items():
    print(window_id)

    for key, path in paths.items():
        print(" ", key, "→", path)
        assert path.exists()
        assert path.stat().st_size > 0


print(
    "\nM5.3 B04 PERSISTENCE GATE: PASS"
)

```

    W_SHORT
      prediction → /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/processed/m5_03_logistic_regression_baseline_b04/M5-LR-SHORT-B04__y_pred.npy
      risk_score → /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/processed/m5_03_logistic_regression_baseline_b04/M5-LR-SHORT-B04__risk_score.npy
      summary → /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/processed/m5_03_logistic_regression_baseline_b04/M5-LR-SHORT-B04__summary.json
    W_LONG
      prediction → /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/processed/m5_03_logistic_regression_baseline_b04/M5-LR-LONG-B04__y_pred.npy
      risk_score → /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/processed/m5_03_logistic_regression_baseline_b04/M5-LR-LONG-B04__risk_score.npy
      summary → /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/processed/m5_03_logistic_regression_baseline_b04/M5-LR-LONG-B04__summary.json
    
    M5.3 B04 PERSISTENCE GATE: PASS


## 15. Round-trip + pair manifest


```python

for window_id, output in [
    ("W_SHORT", short_output),
    ("W_LONG", long_output),
]:
    paths = saved[window_id]

    pred = np.load(
        paths["prediction"],
        allow_pickle=False,
    )

    score = np.load(
        paths["risk_score"],
        allow_pickle=False,
    )

    assert np.array_equal(
        pred,
        output.y_pred,
    )

    assert np.array_equal(
        score,
        output.risk_score,
    )

    with open(
        paths["summary"],
        "r",
        encoding="utf-8",
    ) as file:
        summary = json.load(file)

    assert (
        summary["experiment_id"]
        == output.summary[
            "experiment_id"
        ]
    )


pair_manifest = {
    "m5_substep": "M5.3-B04",
    "model_family": LR_MODEL_FAMILY,
    "model_config_id": LR_CONFIG_ID,
    "config_fingerprint_sha256": fingerprint,
    "b01_status": "FAILED_CONVERGENCE",
    "b02_status": "ABORTED_COMPUTATIONALLY_INEFFICIENT",
    "b04_status": "FAILED_NUMERICAL_WARNING",
    "b04_status": "COMPLETED_PENDING_RUNTIME_REVIEW",
    "training_window_winner": "OPEN",
    "final_threshold": "OPEN",
    "final_test_accessed": False,
    "runs": {
        "W_SHORT": LR_SHORT_EXPERIMENT_ID,
        "W_LONG": LR_LONG_EXPERIMENT_ID,
    },
}


PAIR_MANIFEST_PATH = (
    OUTPUT_DIR
    / "m5_03_lr_b04_pair_manifest.json"
)


with open(
    PAIR_MANIFEST_PATH,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        pair_manifest,
        file,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )


assert PAIR_MANIFEST_PATH.exists()

print(
    "M5.3 B04 ROUND-TRIP / PAIR MANIFEST GATE: PASS"
)

```

    M5.3 B04 ROUND-TRIP / PAIR MANIFEST GATE: PASS


## 16. FINAL TEST isolation


```python

assert manifest["final_test_used"] is False
assert m5_02_contract["final_test_access_allowed"] is False

assert short_summary[
    "final_test_accessed"
] is False

assert long_summary[
    "final_test_accessed"
] is False

assert pair_manifest[
    "final_test_accessed"
] is False


for path in OUTPUT_DIR.iterdir():
    assert (
        "final_test"
        not in path.name.lower()
    )


print(
    "M5.3 B04 FINAL TEST ISOLATION GATE: PASS"
)

```

    M5.3 B04 FINAL TEST ISOLATION GATE: PASS


## 17. Overall technical gate


```python

gates = {
    "G01_INPUT_CONTRACT": True,
    "G02_B04_CONFIG_LOCK": True,
    "G03_W_SHORT_CONVERGED": True,
    "G04_W_LONG_CONVERGED": True,
    "G05_CONTROLLED_PAIR": True,
    "G06_PROBABILITY_DEFAULT_RULE": True,
    "G07_PERSISTENCE": True,
    "G08_ROUND_TRIP": True,
    "G09_FINAL_TEST_ISOLATION": True,
}


for name, value in gates.items():
    print(
        name,
        "→",
        "PASS" if value else "FAIL",
    )


assert all(gates.values())


print(
    "\nM5.3 B04 OVERALL TECHNICAL GATE: PASS"
)

```

    G01_INPUT_CONTRACT → PASS
    G02_B04_CONFIG_LOCK → PASS
    G03_W_SHORT_CONVERGED → PASS
    G04_W_LONG_CONVERGED → PASS
    G05_CONTROLLED_PAIR → PASS
    G06_PROBABILITY_DEFAULT_RULE → PASS
    G07_PERSISTENCE → PASS
    G08_ROUND_TRIP → PASS
    G09_FINAL_TEST_ISOLATION → PASS
    
    M5.3 B04 OVERALL TECHNICAL GATE: PASS


# 18. Phân tích và nhận xét runtime M5.3 B04

## 18.1. Execution integrity

Notebook đã chạy đầy đủ:

`16 / 16 code cells`

Execution count:

`1 → 16 liên tục`

Không ghi nhận:

- exception;
- `output_type = error`;
- `stderr`.

Do đó toàn bộ output hiện tại đủ điều kiện dùng làm runtime evidence cho M5.3 B04.

---

## 18.2. Input / contract integrity

Runtime xác nhận:

- canonical M4.7 artifacts được load thành công;
- M5.2 modeling contract được load thành công;
- 47-column CSR float32 schema được giữ nguyên;
- target dtype/counts khớp canonical contract;
- FINAL TEST không nằm trong modeling path.

Gate:

`M5.3 B04 INPUT INTEGRITY GATE: PASS`

Nhận xét:

B04 không đạt tốc độ nhanh bằng cách giảm sample, giảm feature hoặc bỏ W_LONG.

Model vẫn nhận đầy đủ:

```text
W_SHORT:
1,721,615 training rows
2,491 fraud

W_LONG:
6,855,270 training rows
9,606 fraud

Feature width:
47
```

---

## 18.3. Pre-result configuration lock

B04 config đã được khóa trước official model runs:

```text
solver       = lbfgs
C            = 1.0
l1_ratio     = 0.0
max_iter     = 200
tol          = 1e-4
fit_intercept= True
class_weight = None
random_state = 42
```

Config fingerprint:

`b60b88e457ee01b6b84e70f20108d00539c288e6bef95050a6ab16fe4d3ff776`

Gate:

`M5.3 B04 PRE-RESULT CONFIG LOCK: PASS`

Interpretation:

B04 config không được chọn bằng cách thử nhiều cấu hình rồi lấy validation score cao nhất.

Việc chuyển solver sang `lbfgs` là technical response cho:

- B01 không hội tụ;
- B02 quá chậm;
- B03 phát LinAlgWarning và nội bộ fallback sang lbfgs.

---

## 18.4. W_SHORT optimization behavior

Runtime:

```text
Experiment:
M5-LR-SHORT-B04

Training rows:
1,721,615

Training fraud:
2,491

fit_seconds:
0.822

prediction_seconds:
0.032

warning_count:
0

n_iter_:
[19]
```

Gate:

`M5.3 B04 W_SHORT CONVERGENCE GATE: PASS`

Nhận xét:

`n_iter_=19` thấp hơn rõ ràng `max_iter=200`.

Không có warning.

Không có evidence cho thấy optimizer chạm iteration budget hoặc phải fallback sang solver khác.

---

## 18.5. W_LONG optimization behavior

Runtime:

```text
Experiment:
M5-LR-LONG-B04

Training rows:
6,855,270

Training fraud:
9,606

fit_seconds:
3.614

prediction_seconds:
0.025

warning_count:
0

n_iter_:
[22]
```

Gate:

`M5.3 B04 W_LONG CONVERGENCE GATE: PASS`

Nhận xét:

W_LONG có gần 4 lần số rows W_SHORT và fit time cũng tăng theo cùng bậc độ lớn.

Đây là consistency evidence hỗ trợ việc model thực sự xử lý full W_LONG input, không phải bằng chứng duy nhất nhưng phù hợp với input-integrity checks đã PASS.

---

## 18.6. Vì sao B04 chạy nhanh không phải tự động là lỗi

Lịch sử technical run:

```text
B01:
saga
max_iter=200
→ ~734 giây trên W_LONG
→ ConvergenceWarning
→ FAILED_CONVERGENCE

B02:
saga
max_iter=1000
→ >25 phút trên W_LONG
→ chưa hoàn thành
→ ABORTED / COMPUTATIONALLY INEFFICIENT

B03:
newton-cholesky
→ W_SHORT ≈ 2.166 giây
→ n_iter_=15
→ LinAlgWarning
→ Hessian ill-conditioned
→ sklearn fallback sang lbfgs
→ FAILED_NUMERICAL_WARNING

B04:
lbfgs trực tiếp
→ W_SHORT ≈ 0.822 giây
→ W_LONG ≈ 3.614 giây
→ warning_count=0
→ convergence gates PASS
```

Interpretation:

Sự giảm runtime không đến từ việc thay đổi dataset hay feature schema.

Nó đến từ thay đổi optimization solver.

B04 cho thấy `lbfgs` phù hợp hơn về mặt computational behavior đối với Logistic Regression trên representation hiện tại.

---

## 18.7. Controlled-pair comparability

Gate:

`M5.3 B04 CONTROLLED-PAIR GATE: PASS`

Hai run dùng cùng:

- model family;
- estimator class;
- exact config;
- feature version;
- preprocessing version;
- matrix schema;
- validation population;
- imbalance strategy;
- random-state policy;
- threshold policy;
- probability semantics.

Biến chủ động khác:

`training window`

Do đó khác biệt metric giữa hai run có thể được đọc như controlled baseline evidence về training-window effect trong Logistic Regression B04.

M5.3 vẫn không khóa training-window winner.

---

## 18.8. Probability / default-decision-rule integrity

Gate:

`M5.3 B04 PROBABILITY / DEFAULT-RULE GATE: PASS`

Runtime xác nhận:

- fraud risk score finite;
- risk score nằm trong `[0,1]`;
- `predict()` khớp với default rule `P(fraud) >= 0.5`.

Interpretation:

Probability/risk-score artifacts đủ điều kiện để chuyển sang các bước evaluation sau.

`0.5` ở đây chỉ là:

`BASELINE DEFAULT DECISION RULE`

không phải final threshold.

---

## 18.9. W_SHORT baseline evidence

Observed:

```text
F1_fraud:
0.3374754742

Recall_fraud:
0.2452471483

Precision_fraud:
0.5408805031

Accuracy:
0.9985781618

TP:
258

FP:
219

FN:
794

TN:
711,187

Predicted positive count:
477

Predicted positive rate:
0.0006695132
```

Interpretation:

Ở default decision rule, LR-SHORT flag 477 transaction là fraud.

Trong 1,052 fraud ground-truth của VALIDATION:

- bắt được 258;
- bỏ sót 794.

Precision khoảng 0.541 cho thấy hơn một nửa các prediction-positive là fraud theo dataset.

Recall khoảng 0.245 cho thấy phần lớn fraud vẫn chưa được bắt ở baseline rule.

---

## 18.10. W_LONG baseline evidence

Observed:

```text
F1_fraud:
0.0399636694

Recall_fraud:
0.0209125475

Precision_fraud:
0.4489795918

Accuracy:
0.9985164038

TP:
22

FP:
27

FN:
1,030

TN:
711,379

Predicted positive count:
49

Predicted positive rate:
0.0000687760
```

Interpretation:

Ở cùng default decision rule, LR-LONG chỉ flag 49 transaction positive.

Trong 1,052 fraud:

- bắt được 22;
- bỏ sót 1,030.

Điều này cho thấy LR-LONG baseline có hành vi rất bảo thủ ở decision threshold hiện tại.

---

## 18.11. W_LONG vs W_SHORT — descriptive comparison only

Observed trên cùng validation:

```text
W_SHORT:
F1      ≈ 0.3375
Recall  ≈ 0.2452
Precision≈0.5409
Predicted positive = 477

W_LONG:
F1      ≈ 0.0400
Recall  ≈ 0.0209
Precision≈0.4490
Predicted positive = 49
```

Nhận xét:

Trong Logistic Regression B04, training-window choice tạo ra khác biệt rất lớn ở fraud-class behavior.

W_SHORT phát ra nhiều positive hơn và bắt được nhiều fraud hơn ở default decision rule.

Tuy nhiên M5.3 không chuyển nhận xét này thành:

`W_SHORT = FINAL WINNER`

vì:

- mới chỉ có Logistic Regression baseline;
- Decision Tree và Random Forest chưa chạy;
- comparative/error interpretation sâu thuộc M6;
- final selection thuộc M7.

Training-window winner tiếp tục:

`OPEN`

---

## 18.12. Accuracy không được dùng làm kết luận chính

Cả hai run có Accuracy khoảng:

`0.9985`

nhưng fraud prevalence rất thấp.

Do đó Accuracy cao không mâu thuẫn với việc:

- W_SHORT bỏ sót 794 fraud;
- W_LONG bỏ sót 1,030 fraud.

Điều này củng cố việc project dùng:

`F1_fraud`

làm primary metric và luôn đọc thêm Recall/Precision/Confusion Matrix.

---

## 18.13. Persistence / reproducibility

Runtime gates:

`M5.3 B04 PERSISTENCE GATE: PASS`

`M5.3 B04 ROUND-TRIP / PAIR MANIFEST GATE: PASS`

Artifacts đã được persist cho cả W_SHORT và W_LONG:

- `y_pred`;
- `risk_score`;
- summary JSON;
- pair manifest;
- config lock.

Round-trip xác nhận persisted values khớp in-memory outputs.

---

## 18.14. FINAL TEST isolation

Runtime gate:

`M5.3 B04 FINAL TEST ISOLATION GATE: PASS`

Evidence:

- M4 manifest vẫn `final_test_used = false`;
- M5.2 contract không cho final-test access;
- run summaries ghi `final_test_accessed = false`;
- pair manifest ghi `final_test_accessed = false`;
- output directory không chứa final-test artifact.

Interpretation:

M5.3 không tiêu thụ FINAL TEST để chọn solver, window, metric, threshold hoặc model.

---

## 18.15. Overall technical result

Runtime:

```text
G01_INPUT_CONTRACT           → PASS
G02_B04_CONFIG_LOCK          → PASS
G03_W_SHORT_CONVERGED        → PASS
G04_W_LONG_CONVERGED         → PASS
G05_CONTROLLED_PAIR          → PASS
G06_PROBABILITY_DEFAULT_RULE → PASS
G07_PERSISTENCE              → PASS
G08_ROUND_TRIP               → PASS
G09_FINAL_TEST_ISOLATION     → PASS
```

Overall:

`M5.3 B04 OVERALL TECHNICAL GATE: PASS`

Blocking issue:

`NONE`

# 19. Findings M5.3

## M5.3-F01 — B04 is the first technically clean Logistic Regression baseline

Evidence:

- W_SHORT warning count = 0;
- W_LONG warning count = 0;
- `n_iter_ = 19 / 22`;
- both below `max_iter=200`;
- both convergence gates PASS.

Finding:

`LR-B04 = TECHNICALLY VALID BASELINE`

Status:

`VERIFIED`

---

## M5.3-F02 — Solver choice materially affects computational feasibility

Evidence:

```text
B01 saga:
failed convergence

B02 saga:
computationally inefficient / aborted

B03 newton-cholesky:
numerical warning + fallback

B04 lbfgs:
clean convergence in seconds
```

Finding:

For the current 47-feature Logistic Regression representation, `lbfgs` is operationally suitable for baseline training.

Caveat:

Runtime suitability alone does not make Logistic Regression the final model.

Status:

`VERIFIED`

---

## M5.3-F03 — Fast B04 runtime is internally consistent with full-data execution

Evidence:

- input row counts match canonical full W_SHORT/W_LONG;
- exact M4 matrix contract is preserved;
- W_LONG has ~4x rows W_SHORT;
- W_LONG fit time is also several times W_SHORT;
- predictions are non-trivial;
- convergence diagnostics are populated.

Finding:

No evidence was found that B04 achieved speed by silently reducing the modeling population.

Status:

`VERIFIED`

---

## M5.3-F04 — Controlled W_LONG/W_SHORT comparison is valid for LR-B04

Evidence:

Controlled-pair gate PASS.

Finding:

The primary changing factor between LR-LONG-B04 and LR-SHORT-B04 is classifier-training window.

Status:

`VERIFIED`

---

## M5.3-F05 — W_SHORT and W_LONG exhibit materially different fraud-class behavior

Observed:

```text
W_SHORT:
F1 ≈ 0.3375
Recall ≈ 0.2452
Precision ≈ 0.5409
TP = 258
Predicted positive = 477

W_LONG:
F1 ≈ 0.0400
Recall ≈ 0.0209
Precision ≈ 0.4490
TP = 22
Predicted positive = 49
```

Finding:

Within Logistic Regression B04 and the same default decision rule, training-window choice strongly changes validation behavior.

Status:

`OBSERVED BASELINE EVIDENCE`

---

## M5.3-F06 — W_LONG is much more conservative at the baseline decision rule

Evidence:

```text
W_LONG predicted positive:
49 / 712,458

W_SHORT predicted positive:
477 / 712,458
```

Finding:

The W_LONG baseline emits substantially fewer fraud flags and therefore has very low fraud recall at the default rule.

Status:

`OBSERVED`

---

## M5.3-F07 — Baseline still misses many fraud cases

Evidence:

```text
W_SHORT FN:
794

W_LONG FN:
1,030

Total validation fraud:
1,052
```

Finding:

No-intervention Logistic Regression baseline leaves substantial false-negative burden.

Interpretation boundary:

This finding does not authorize class-weight/resampling/threshold tuning inside M5.3.

Status:

`OBSERVED — HANDOFF TO LATER EVALUATION/SELECTION`

---

## M5.3-F08 — Accuracy is not decision-sufficient

Evidence:

Both runs have Accuracy ≈ 0.9985 despite very different fraud recall.

Finding:

Accuracy is confirmed as reference-only in this imbalanced classification problem.

Status:

`CONSISTENT WITH PROJECT METRIC POLICY`

---

## M5.3-F09 — Probability/risk-score evidence is available

Evidence:

Probability/default-rule gate PASS and risk-score artifacts persist successfully.

Finding:

LR-B04 outputs can support later threshold analysis without rerunning the baseline solely to recreate probabilities.

Status:

`VERIFIED`

---

## M5.3-F10 — FINAL TEST remains protected

Evidence:

FINAL TEST isolation gate PASS.

Finding:

No final-test performance evidence has been exposed in M5.3.

Status:

`VERIFIED`

---

## M5.3-F11 — Training-window winner remains open

Finding:

M5.3 records strong LR-specific descriptive evidence, but does not lock W_SHORT or W_LONG as final winner.

Reason:

M5.4/M5.5 and later evaluation/model-selection evidence are still pending.

Status:

`OPEN`

---

## M5.3-F12 — M5.4 handoff is unblocked

Evidence:

- technically valid LR baseline pair exists;
- predictions and probabilities are persisted;
- no warning/convergence blocker;
- no FINAL TEST violation.

Status:

`READY FOR M5.4`

# 20. Decision Log M5.3

## M5.3-D01 — B01 status

Decision:

`LR-B01 = FAILED_CONVERGENCE`

Reason:

SAGA reached `max_iter=200` and emitted ConvergenceWarning.

B01 metrics:

`NOT AUTHORIZED FOR MODEL/WINDOW DECISION`

Status:

`LOCKED`

---

## M5.3-D02 — B02 status

Decision:

`LR-B02 = ABORTED / COMPUTATIONALLY INEFFICIENT`

Reason:

W_LONG remained in SAGA fit for more than 25 minutes without completion.

Status:

`LOCKED`

---

## M5.3-D03 — B03 status

Decision:

`LR-B03 = FAILED_NUMERICAL_WARNING`

Reason:

Newton-Cholesky encountered singular/ill-conditioned Hessian and internally fell back to L-BFGS.

B03 not accepted as clean baseline.

Status:

`LOCKED`

---

## M5.3-D04 — Official Logistic Regression baseline configuration

Decision:

Official technically valid LR baseline:

`LR-B04-LBFGS-L2-C1`

```text
solver       = lbfgs
C            = 1.0
l1_ratio     = 0.0
max_iter     = 200
tol          = 1e-4
fit_intercept= True
class_weight = None
random_state = 42
```

Status:

`LOCKED FOR M5 BASELINE`

---

## M5.3-D05 — Official M5.3 baseline runs

Decision:

```text
M5-LR-SHORT-B04
M5-LR-LONG-B04
```

Status:

`VALID / RUNTIME VERIFIED`

---

## M5.3-D06 — Imbalance strategy

Decision:

`NONE`

Status:

`INHERITED — LOCKED`

No class weighting/resampling in M5.3.

---

## M5.3-D07 — Threshold policy

Decision:

`DEFAULT_MODEL_DECISION_RULE`

For Logistic Regression B04:

`P(fraud) >= 0.5`

This is baseline comparison policy only.

Final threshold:

`OPEN`

Status:

`LOCKED FOR BASELINE / FINAL OPEN`

---

## M5.3-D08 — Metric contract

Decision:

Continue using:

- F1_fraud primary;
- Recall_fraud;
- Precision_fraud;
- TP/FP/FN/TN;
- predicted-positive count/rate;
- Accuracy reference-only.

Status:

`INHERITED — VERIFIED — LOCKED`

---

## M5.3-D09 — Probability/risk score

Decision:

Persist positive-class probability for both official B04 runs.

Status:

`LOCKED`

---

## M5.3-D10 — Training-window interpretation

Decision:

M5.3 records descriptive LR-specific differences between W_SHORT and W_LONG.

M5.3 does not select final training-window winner.

Status:

`WINNER OPEN`

---

## M5.3-D11 — Model-family interpretation

Decision:

M5.3 does not claim Logistic Regression is better/worse than Decision Tree or Random Forest because those official baseline runs are not yet complete.

Status:

`OPEN`

---

## M5.3-D12 — Tuning / imbalance / threshold optimization

Decision:

Not performed in M5.3.

Status:

`DEFERRED TO M7`

---

## M5.3-D13 — FINAL TEST

Decision:

No access.

Status:

`INHERITED — VERIFIED — LOCKED`

---

## M5.3-D14 — M5.4 handoff

Decision:

Logistic Regression baseline evidence is complete enough to proceed to Decision Tree baseline.

Status:

`READY`

# 21. M5.3 Gate

## G01 — Canonical input contract

Evidence:

M4.7/M5.2 input integrity checks PASS.

Result:

`PASS`

---

## G02 — Config locked before official result

Evidence:

B04 config lock persisted with SHA256 fingerprint before W_SHORT/W_LONG runs.

Result:

`PASS`

---

## G03 — W_SHORT technical validity

Evidence:

```text
fit ≈ 0.822 s
warning_count = 0
n_iter_ = 19 < 200
```

Result:

`PASS`

---

## G04 — W_LONG technical validity

Evidence:

```text
fit ≈ 3.614 s
warning_count = 0
n_iter_ = 22 < 200
```

Result:

`PASS`

---

## G05 — Controlled training-window comparison

Evidence:

Controlled-pair gate PASS.

Result:

`PASS`

---

## G06 — Probability/default-decision-rule integrity

Evidence:

Risk scores finite/in-range and class prediction matches default `>=0.5` rule.

Result:

`PASS`

---

## G07 — Canonical metric evidence available

Evidence:

Both runs produced complete F1/Recall/Precision/CM/predicted-positive/Accuracy bundle.

Result:

`PASS`

---

## G08 — Persistence

Evidence:

Prediction/risk-score/summary artifacts saved.

Result:

`PASS`

---

## G09 — Round-trip integrity

Evidence:

Saved artifacts reload equal to in-memory outputs.

Result:

`PASS`

---

## G10 — FINAL TEST isolation

Evidence:

Final-test isolation gate PASS.

Result:

`PASS`

---

## G11 — No hidden imbalance intervention

Evidence:

`class_weight=None`

`IMBALANCE_STRATEGY=NONE`

Result:

`PASS`

---

## G12 — No systematic tuning

Evidence:

B04 changes were technical responses to convergence/computational/numerical failures, not validation-score search.

Result:

`PASS`

---

## Overall M5.3 Gate

```text
G01 PASS
G02 PASS
G03 PASS
G04 PASS
G05 PASS
G06 PASS
G07 PASS
G08 PASS
G09 PASS
G10 PASS
G11 PASS
G12 PASS
```

Overall:

`M5.3 — PASS`

Blocking issue:

`NONE`

# 22. Kết luận M5.3

M5.3 đã tạo được Logistic Regression baseline pair hợp lệ về kỹ thuật và đúng protocol.

Official baseline configuration:

`LR-B04-LBFGS-L2-C1`

Official runs:

```text
M5-LR-SHORT-B04
M5-LR-LONG-B04
```

Runtime convergence:

```text
W_SHORT:
0.822 s
n_iter_ = 19
warnings = 0

W_LONG:
3.614 s
n_iter_ = 22
warnings = 0
```

Baseline validation evidence:

```text
W_SHORT:
F1_fraud        ≈ 0.3375
Recall_fraud    ≈ 0.2452
Precision_fraud ≈ 0.5409
TP / FP         = 258 / 219
FN / TN         = 794 / 711,187
Predicted +     = 477

W_LONG:
F1_fraud        ≈ 0.0400
Recall_fraud    ≈ 0.0209
Precision_fraud ≈ 0.4490
TP / FP         = 22 / 27
FN / TN         = 1,030 / 711,379
Predicted +     = 49
```

Interpretation được phép khóa:

- Logistic Regression B04 chạy hợp lệ trên full W_SHORT và W_LONG matrices;
- solver `lbfgs` phù hợp về computational behavior với representation hiện tại;
- W_SHORT và W_LONG tạo fraud-class behavior rất khác nhau trong LR baseline;
- default no-intervention baseline vẫn bỏ sót nhiều fraud;
- Accuracy không đủ để đánh giá model trong bài toán mất cân bằng;
- probability/risk score đã được persist;
- FINAL TEST vẫn được bảo vệ.

M5.3 chưa được phép khóa:

- W_SHORT là final training-window winner;
- Logistic Regression là final model;
- class-weight/resampling strategy;
- tuned hyperparameters;
- final numerical threshold;
- final-test performance.

Trạng thái cuối:

```text
M5.3 — PASS

Official LR Baseline:
LR-B04-LBFGS-L2-C1

LR-SHORT-B04:
VALID

LR-LONG-B04:
VALID

Controlled Pair:
VERIFIED

Imbalance Strategy:
NONE

Probability Evidence:
PERSISTED

Baseline Threshold Policy:
DEFAULT MODEL DECISION RULE

Final Threshold:
OPEN

Training-window Winner:
OPEN

Final Model:
OPEN

FINAL TEST:
PROTECTED

Blocking Issue:
NONE

READY FOR M5.4
```

Bước tiếp theo:

`M5.4 — Decision Tree baseline`

M5.4 phải khóa exact Decision Tree baseline config trước khi đọc comparative validation result và chạy controlled pair:

`DT-LONG`

`DT-SHORT`.
