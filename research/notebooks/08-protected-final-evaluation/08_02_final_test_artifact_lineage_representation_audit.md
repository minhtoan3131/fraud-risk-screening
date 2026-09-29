# M8.2 — FINAL TEST artifact / lineage / representation audit

**Project:** AI Transaction Fraud Risk Screening  
**Milestone:** M8 — Protected Final Evaluation  
**Substep:** M8.2 — FINAL TEST artifact / lineage / representation audit  
**Work type:** Runtime audit / identity verification / representation materialization  
**Upstream:** M8.1 Final Evaluation Charter — PASS  
**Decision trước runtime:** `OPEN — REQUIRES RUNTIME AUDIT`

---

## Vai trò của notebook này

Notebook này **chưa phải M8.3**.

M8.2 chỉ trả lời câu hỏi:

> Project có thể chứng minh rằng FINAL TEST representation và official selected estimator thực sự tương thích đúng với frozen M7 identity trước khi tạo bất kỳ official final prediction nào hay không?

Notebook được phép:

- đọc raw dataset để audit FINAL TEST identity;
- đọc target FINAL TEST để xác minh expected population `722,955 rows / 1,035 fraud`;
- tái dựng W_SHORT preprocessing state từ **TRAIN-only** source;
- dựng lại VALIDATION representation để kiểm tra exact reproduction với M4.7;
- dựng FINAL TEST representation theo đúng M4 causal contract;
- tái dựng selected RF estimator từ exact frozen config nếu không có canonical serialized estimator;
- dùng **VALIDATION** để kiểm tra estimator reconstruction có tái tạo M7.7 hay không;
- persist audit artifacts và fingerprint.

Notebook **không được phép**:

- tạo prediction trên FINAL TEST;
- gọi `predict()` hoặc `predict_proba()` với `X_final_test`;
- tính F1 / Recall / Precision / Confusion Matrix trên FINAL TEST;
- thử threshold khác;
- thay feature/preprocessing/model/config/seed;
- refit TRAIN+VALIDATION;
- sử dụng FINAL TEST để sửa bất kỳ development decision nào.

Nếu bất kỳ identity / lineage / representation / reconstruction gate nào fail:

`STOP — KHÔNG CHUYỂN SANG M8.3`



## 1. Frozen contract kế thừa

Official final-evaluation subject:

```text
Training window:
W_SHORT

Classifier TRAIN:
2018-01-01 <= Timestamp < 2019-01-01

Model:
Random Forest

Config ID:
RF-REF-100-GINI-SQRT-UNPRUNED-CW

Imbalance:
CLASS_WEIGHT_BALANCED

random_state:
42

Risk score:
predict_proba / positive class = 1

Threshold:
0.50

Comparator:
risk_score > 0.50

FINAL TEST:
2019-06-01 <= Timestamp < 2019-11-01

Expected:
722,955 rows
1,035 fraud

Representation:
10 pre-encoding features
→ 47 encoded columns
→ CSR sparse
→ float32

Causal history:
Timestamp(history) < Timestamp(current)
```

M8.2 phải kiểm chứng runtime; các giá trị trên không được coi là PASS chỉ vì đã được ghi trong tài liệu.



```python

from pathlib import Path
from collections import Counter
from dataclasses import dataclass
import copy
import gc
import hashlib
import json
import platform
import sys
import time
import warnings

import numpy as np
import pandas as pd

try:
    from scipy import sparse
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Thiếu scipy. Hãy cài dependency của project rồi Restart Kernel."
    ) from exc

try:
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.ensemble import RandomForestClassifier
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Thiếu scikit-learn. Hãy cài dependency của project rồi Restart Kernel."
    ) from exc

try:
    import joblib
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Thiếu joblib. joblib thường đi cùng scikit-learn."
    ) from exc

print("Python:", sys.version)
print("Executable:", sys.executable)
print("Platform:", platform.platform())
print("NumPy:", np.__version__)
print("pandas:", pd.__version__)

```

    Python: 3.14.6 (main, Jun 10 2026, 10:03:53) [Clang 21.0.0 (clang-2100.0.123.102)]
    Executable: /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/.venv/bin/python
    Platform: macOS-26.6.2-arm64-arm-64bit-Mach-O
    NumPy: 2.5.3
    pandas: 3.0.5


## 2. Xác định project root và upstream artifacts


```python

RAW_REL = Path("data") / "raw" / "ibm_tabformer" / "card_transaction.v1.csv"
M4_REL = Path("data") / "processed" / "m4_07_baseline_ready"
M7_06_REL = Path("data") / "processed" / "m7_06_moderate_hyperparameter_tuning"
M7_07_REL = Path("data") / "processed" / "m7_07_candidate_selection_external_validation"
M7_08_REL = Path("data") / "processed" / "m7_08_numerical_threshold_selection"
M8_02_REL = Path("data") / "processed" / "m8_02_final_test_artifact_audit"

required_rel_paths = [
    RAW_REL,
    M4_REL / "X_train_w_short.npz",
    M4_REL / "X_validation_w_short.npz",
    M4_REL / "y_train_w_short.npy",
    M4_REL / "y_validation.npy",
    M4_REL / "row_id_train_w_short.npy",
    M4_REL / "row_id_validation.npy",
    M4_REL / "feature_names.json",
    M4_REL / "manifest.json",
    M7_06_REL / "m7_06_tuning_registry.json",
    M7_06_REL / "m7_06_tuning_manifest.json",
    M7_07_REL / "m7_07_external_validation_registry.json",
    M7_07_REL / "m7_07_candidate_selection_manifest.json",
    M7_08_REL / "m7_08_threshold_registry.json",
    M7_08_REL / "m7_08_threshold_manifest.json",
]

candidate_roots = [
    Path.cwd(),
    *list(Path.cwd().parents)[:8],
]

PROJECT_ROOT = None
for candidate in candidate_roots:
    candidate = candidate.resolve()
    if all((candidate / rel).exists() for rel in required_rel_paths):
        PROJECT_ROOT = candidate
        break

if PROJECT_ROOT is None:
    missing_by_candidate = {}
    for candidate in candidate_roots:
        candidate = candidate.resolve()
        missing = [
            str(rel)
            for rel in required_rel_paths
            if not (candidate / rel).exists()
        ]
        missing_by_candidate[str(candidate)] = missing[:8]
    raise FileNotFoundError(
        "Không tìm thấy PROJECT_ROOT chứa đầy đủ raw + M4.7 + M7.6 + M7.7 + M7.8 artifacts.\n"
        + json.dumps(missing_by_candidate, ensure_ascii=False, indent=2)
    )

RAW_PATH = PROJECT_ROOT / RAW_REL
M4_DIR = PROJECT_ROOT / M4_REL
M7_06_DIR = PROJECT_ROOT / M7_06_REL
M7_07_DIR = PROJECT_ROOT / M7_07_REL
M7_08_DIR = PROJECT_ROOT / M7_08_REL
OUTPUT_DIR = PROJECT_ROOT / M8_02_REL
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

X_TRAIN_PATH = M4_DIR / "X_train_w_short.npz"
X_VALIDATION_PATH = M4_DIR / "X_validation_w_short.npz"
Y_TRAIN_PATH = M4_DIR / "y_train_w_short.npy"
Y_VALIDATION_PATH = M4_DIR / "y_validation.npy"
ROW_TRAIN_PATH = M4_DIR / "row_id_train_w_short.npy"
ROW_VALIDATION_PATH = M4_DIR / "row_id_validation.npy"
FEATURE_NAMES_PATH = M4_DIR / "feature_names.json"
M4_MANIFEST_PATH = M4_DIR / "manifest.json"

M7_06_RESULT_PATH = M7_06_DIR / "m7_06_tuning_registry.json"
M7_06_MANIFEST_PATH = M7_06_DIR / "m7_06_tuning_manifest.json"

M7_07_RESULT_PATH = M7_07_DIR / "m7_07_external_validation_registry.json"
M7_07_MANIFEST_PATH = M7_07_DIR / "m7_07_candidate_selection_manifest.json"

M7_08_RESULT_PATH = M7_08_DIR / "m7_08_threshold_registry.json"
M7_08_MANIFEST_PATH = M7_08_DIR / "m7_08_threshold_manifest.json"

print("PROJECT_ROOT:", PROJECT_ROOT)
print("RAW_PATH:", RAW_PATH)
print("OUTPUT_DIR:", OUTPUT_DIR)
print("\nM8.2 SOURCE LOCATION GATE: PASS")

```

    PROJECT_ROOT: /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening
    RAW_PATH: /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/raw/ibm_tabformer/card_transaction.v1.csv
    OUTPUT_DIR: /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/processed/m8_02_final_test_artifact_audit
    
    M8.2 SOURCE LOCATION GATE: PASS


## 3. Frozen constants và expected runtime identity


```python

EXPECTED_RAW_FILE_SIZE = 2_354_626_737
EXPECTED_RAW_ROWS = 24_386_900
EXPECTED_CARD_COUNT = 6_139

W_SHORT_START = pd.Timestamp("2018-01-01")
TRAIN_END = pd.Timestamp("2019-01-01")
VALIDATION_END = pd.Timestamp("2019-06-01")
FINAL_TEST_END = pd.Timestamp("2019-11-01")

EXPECTED_W_SHORT_ROWS = 1_721_615
EXPECTED_W_SHORT_FRAUD = 2_491

EXPECTED_VALIDATION_ROWS = 712_458
EXPECTED_VALIDATION_FRAUD = 1_052
EXPECTED_VALIDATION_NNZ = 6_430_339

EXPECTED_FINAL_ROWS = 722_955
EXPECTED_FINAL_FRAUD = 1_035

EXPECTED_CONTEXT_ROWS_BEFORE_VALIDATION_END = 23_038_920
EXPECTED_CONTEXT_ROWS_BEFORE_FINAL_TEST_END = (
    EXPECTED_CONTEXT_ROWS_BEFORE_VALIDATION_END
    + EXPECTED_FINAL_ROWS
)

EXPECTED_FEATURE_COUNT = 47
CHUNK_SIZE = 500_000
UNKNOWN_TOKEN = "__UNKNOWN__"

SELECTED_CANDIDATE_ID = "RF-REF-100-GINI-SQRT-UNPRUNED-CW"
SELECTED_MODEL_FAMILY = "Random Forest"
SELECTED_TRAINING_WINDOW = "W_SHORT"
SELECTED_IMBALANCE_STRATEGY = "CLASS_WEIGHT_BALANCED"
SELECTED_RANDOM_STATE = 42
FINAL_THRESHOLD = 0.50
EXPECTED_THRESHOLD_COMPARATOR = ">"

M8_02_ANALYSIS_VERSION = "M8.2-final-test-artifact-audit-v1"

assert RAW_PATH.stat().st_size == EXPECTED_RAW_FILE_SIZE

print("Raw file size:", RAW_PATH.stat().st_size)
print("Expected final rows/fraud:", EXPECTED_FINAL_ROWS, "/", EXPECTED_FINAL_FRAUD)
print("Selected candidate:", SELECTED_CANDIDATE_ID)
print("\nM8.2 FROZEN CONTRACT GATE: PASS")

```

    Raw file size: 2354626737
    Expected final rows/fraud: 722955 / 1035
    Selected candidate: RF-REF-100-GINI-SQRT-UNPRUNED-CW
    
    M8.2 FROZEN CONTRACT GATE: PASS


## 4. Load và audit upstream M4 / M7 handoff

### Lưu ý về artifact M7.8

"
            "M7.8 cố ý persist registry/manifest **trước runtime review**, vì vậy "
            "`final_threshold` trong JSON lịch sử còn ở trạng thái "
            "`OPEN — REQUIRES M7.8 RUNTIME REVIEW`. Đây không phải corruption.

"
            "Quyết định `0.50` được khóa sau runtime review trong Decision Log M7.8, "
            "sau đó được M7.9 Final Selection Registry và M8.1 Charter carry forward "
            "thành frozen upstream decision. M8.2 vì thế kiểm tra:

"
            "- persisted M7.8 evidence vẫn nguyên trạng `OPEN`;
"
            "- `default_threshold == 0.50`;
"
            "- comparator là `>`;
"
            "- threshold 0.50 có record trong threshold results;
"
            "- default-boundary mismatch = 0;
"
            "- frozen threshold mang vào M8 vẫn là 0.50.

"
            "**Không sửa JSON M7.8 cũ để biến `OPEN` thành `0.50`.**


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

with open(FEATURE_NAMES_PATH, "r", encoding="utf-8") as file:
    canonical_feature_names = json.load(file)

with open(M7_06_RESULT_PATH, "r", encoding="utf-8") as file:
    m7_06_result = json.load(file)

with open(M7_06_MANIFEST_PATH, "r", encoding="utf-8") as file:
    m7_06_manifest = json.load(file)

with open(M7_07_RESULT_PATH, "r", encoding="utf-8") as file:
    m7_07_result = json.load(file)

with open(M7_07_MANIFEST_PATH, "r", encoding="utf-8") as file:
    m7_07_manifest = json.load(file)

with open(M7_08_RESULT_PATH, "r", encoding="utf-8") as file:
    m7_08_result = json.load(file)

with open(M7_08_MANIFEST_PATH, "r", encoding="utf-8") as file:
    m7_08_manifest = json.load(file)

assert m4_manifest["pipeline_version"] == "M4.7-baseline-v1"
assert m4_manifest["output_width"] == EXPECTED_FEATURE_COUNT
assert m4_manifest["matrix_dtype"] == "float32"
assert m4_manifest["matrix_format"] == "CSR"
assert m4_manifest["final_test_used"] is False
assert len(canonical_feature_names) == EXPECTED_FEATURE_COUNT

assert (
    m7_06_result["analysis_version"]
    == "M7.6-moderate-hyperparameter-tuning-v1"
)
assert m7_06_manifest["final_test_accessed"] is False

assert (
    m7_07_result["analysis_version"]
    == "M7.7-candidate-selection-external-validation-v1"
)
assert m7_07_manifest["final_test_accessed"] is False

assert (
    m7_08_result["analysis_version"]
    == "M7.8-numerical-threshold-selection-v1"
)
assert m7_08_manifest["final_test_accessed"] is False
assert m7_08_manifest["selected_candidate_id"] == SELECTED_CANDIDATE_ID

# M7.8 persist runtime evidence TRƯỚC runtime review.
# Vì vậy field final_threshold trong JSON được EXPECT là OPEN,
# không phải số 0.50. Không được sửa artifact lịch sử này.
assert float(m7_08_manifest["default_threshold"]) == FINAL_THRESHOLD
assert m7_08_manifest["comparator"] == EXPECTED_THRESHOLD_COMPARATOR

persisted_m7_08_final_threshold = m7_08_manifest["final_threshold"]
assert isinstance(persisted_m7_08_final_threshold, str)
assert persisted_m7_08_final_threshold.startswith("OPEN")

# Cross-check chính evidence đã chạy ở M7.8:
# threshold 0.50 phải tồn tại trong predeclared registry và phải
# tái tạo đúng M7.7 default prediction/metric.
threshold_results = m7_08_result["threshold_results"]

m7_08_default_records = [
    record
    for record in threshold_results
    if float(record["threshold"]) == FINAL_THRESHOLD
]
assert len(m7_08_default_records) == 1

m7_08_default_record = m7_08_default_records[0]
assert m7_08_default_record["comparator"] == EXPECTED_THRESHOLD_COMPARATOR
assert int(m7_08_result["default_boundary_mismatch_count"]) == 0

# Reviewed/frozen decision 0.50 được carry từ M7.8 Decision Log
# → M7.9 Final Selection Registry → M8.1 Charter.
# Notebook M8.2 nhận FINAL_THRESHOLD = 0.50 như frozen upstream
# decision và chỉ audit compatibility của runtime artifacts.
assert FINAL_THRESHOLD == 0.50

print("M4 pipeline:", m4_manifest["pipeline_version"])
print("M7.6 final test accessed:", m7_06_manifest["final_test_accessed"])
print("M7.7 final test accessed:", m7_07_manifest["final_test_accessed"])
print("M7.8 final test accessed:", m7_08_manifest["final_test_accessed"])
print("M7.8 persisted final_threshold state:", persisted_m7_08_final_threshold)
print("Reviewed/frozen final threshold carried into M8:", FINAL_THRESHOLD)
print("Threshold comparator:", m7_08_manifest["comparator"])
print("M7.8 default-boundary mismatch:", m7_08_result["default_boundary_mismatch_count"])
print("\nM8.2 UPSTREAM HANDOFF GATE: PASS")

```

    M4 pipeline: M4.7-baseline-v1
    M7.6 final test accessed: False
    M7.7 final test accessed: False
    M7.8 final test accessed: False
    M7.8 persisted final_threshold state: OPEN — REQUIRES M7.8 RUNTIME REVIEW
    Reviewed/frozen final threshold carried into M8: 0.5
    Threshold comparator: >
    M7.8 default-boundary mismatch: 0
    
    M8.2 UPSTREAM HANDOFF GATE: PASS



## 5. Canonical feature / causal helpers

Các helper dưới đây được giữ cùng semantics với M4.7:

- deterministic string strip;
- strict `Timestamp(history) < Timestamp(current)`;
- same-timestamp transaction không làm history cho nhau;
- `Errors?` không được đọc;
- raw User/Card/Merchant Name không đi trực tiếp vào X;
- structural NA chỉ được phép ở hai behavioral numeric feature;
- unknown category map vào `__UNKNOWN__`.



```python

def map_binary_target(target_series):
    target = (
        target_series
        .astype("string")
        .str.strip()
    )
    observed = set(target.dropna().unique())
    unexpected = observed - {"Yes", "No"}

    if unexpected:
        raise ValueError(
            "Unexpected target values: "
            + repr(sorted(unexpected))
        )

    if target.isna().any():
        raise ValueError("Missing target trong audited rows.")

    return (
        target.eq("Yes")
        .astype(np.int8)
        .to_numpy()
    )


def parse_amount(amount_series):
    cleaned = (
        amount_series
        .astype("string")
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors="coerce")


def build_timestamp(df):
    date_part = pd.to_datetime(
        {
            "year": df["Year"],
            "month": df["Month"],
            "day": df["Day"],
        },
        errors="coerce",
    )
    time_part = pd.to_timedelta(
        df["Time"].astype("string") + ":00",
        errors="coerce",
    )
    return date_part + time_part


def normalize_string(series):
    return (
        series
        .astype("string")
        .str.strip()
    )


def assign_location_state(df):
    city = normalize_string(df["Merchant City"])

    city_online = city.eq("ONLINE").fillna(False)
    state_missing = df["Merchant State"].isna()
    zip_missing = df["Zip"].isna()

    nonphysical = (
        city_online
        & state_missing
        & zip_missing
    )

    physical_complete = (
        (~city_online)
        & (~state_missing)
        & (~zip_missing)
    )

    physical_zip_unavailable = (
        (~city_online)
        & (~state_missing)
        & zip_missing
    )

    result = np.select(
        [
            nonphysical,
            physical_complete,
            physical_zip_unavailable,
        ],
        [
            "NON_PHYSICAL_OR_ONLINE",
            "PHYSICAL_COMPLETE",
            "PHYSICAL_ZIP_UNAVAILABLE",
        ],
        default="OTHER_INCONSISTENT",
    )

    return pd.Series(
        result,
        index=df.index,
        dtype="string",
    )


FIT_USECOLS = [
    "User",
    "Card",
    "Year",
    "Month",
    "Day",
    "Time",
    "Amount",
    "Use Chip",
    "Merchant Name",
    "Merchant City",
    "Merchant State",
    "Zip",
]

MATRIX_USECOLS = [
    *FIT_USECOLS,
    "Is Fraud?",
]

assert "Is Fraud?" not in FIT_USECOLS
assert "Errors?" not in FIT_USECOLS
assert "Errors?" not in MATRIX_USECOLS


def iter_card_blocks(data_path, chunksize, usecols):
    pending_key = None
    pending_parts = []
    raw_row_offset = 0

    for chunk in pd.read_csv(
        data_path,
        usecols=usecols,
        chunksize=chunksize,
    ):
        chunk_length = len(chunk)

        chunk["raw_row_id"] = np.arange(
            raw_row_offset,
            raw_row_offset + chunk_length,
            dtype=np.int64,
        )
        raw_row_offset += chunk_length

        chunk["Timestamp"] = build_timestamp(chunk)
        chunk["Amount_numeric"] = parse_amount(chunk["Amount"])

        if chunk["Timestamp"].isna().any():
            raise ValueError("Timestamp parse failure.")

        if chunk["Amount_numeric"].isna().any():
            raise ValueError("Amount parse failure.")

        if chunk["Merchant Name"].isna().any():
            raise ValueError("Merchant Name missing.")

        working_columns = [
            "raw_row_id",
            "User",
            "Card",
            "Timestamp",
            "Amount_numeric",
            "Use Chip",
            "Merchant Name",
            "Merchant City",
            "Merchant State",
            "Zip",
        ]

        if "Is Fraud?" in usecols:
            working_columns.append("Is Fraud?")

        working = chunk[working_columns]

        users = working["User"].to_numpy()
        cards = working["Card"].to_numpy()

        change_positions = (
            np.flatnonzero(
                (users[1:] != users[:-1])
                | (cards[1:] != cards[:-1])
            )
            + 1
        )

        starts = np.concatenate([[0], change_positions])
        ends = np.concatenate([change_positions, [len(working)]])

        for start, end in zip(starts, ends):
            key = (
                int(users[start]),
                int(cards[start]),
            )
            segment = working.iloc[start:end].copy()

            if pending_key is None:
                pending_key = key
                pending_parts = [segment]
                continue

            if key == pending_key:
                pending_parts.append(segment)
                continue

            card_block = (
                pending_parts[0].reset_index(drop=True)
                if len(pending_parts) == 1
                else pd.concat(pending_parts, ignore_index=True)
            )

            yield pending_key, card_block

            pending_key = key
            pending_parts = [segment]

    if pending_key is not None:
        card_block = (
            pending_parts[0].reset_index(drop=True)
            if len(pending_parts) == 1
            else pd.concat(pending_parts, ignore_index=True)
        )
        yield pending_key, card_block

```


```python

ONE_HOUR_NS = int(pd.Timedelta(hours=1).value)


def compute_causal_behavioral_features(card_df):
    timestamp_ns = (
        card_df["Timestamp"]
        .to_numpy(dtype="datetime64[ns]")
        .astype("int64")
    )

    if (np.diff(timestamp_ns) < 0).any():
        raise ValueError(
            "Timestamp giảm bên trong Card block."
        )

    group_starts = np.concatenate(
        [
            [0],
            np.flatnonzero(
                timestamp_ns[1:]
                != timestamp_ns[:-1]
            ) + 1,
        ]
    )

    group_ends = np.concatenate(
        [
            group_starts[1:],
            [len(card_df)],
        ]
    )

    group_lengths = group_ends - group_starts
    group_timestamp_ns = timestamp_ns[group_starts]

    prior_count = np.repeat(
        group_starts,
        group_lengths,
    )
    has_prior_card_history = prior_count > 0

    time_since_group = np.full(
        len(group_starts),
        np.nan,
        dtype="float64",
    )

    if len(group_starts) > 1:
        time_since_group[1:] = (
            (
                group_timestamp_ns[1:]
                - group_timestamp_ns[:-1]
            )
            / 60_000_000_000
        )

    time_since_previous_min = np.repeat(
        time_since_group,
        group_lengths,
    )

    left_1h = np.searchsorted(
        timestamp_ns,
        group_timestamp_ns - ONE_HOUR_NS,
        side="left",
    )

    transactions_last_1h_group = (
        group_starts - left_1h
    )

    transactions_last_1h = np.repeat(
        transactions_last_1h_group,
        group_lengths,
    )

    amount = (
        card_df["Amount_numeric"]
        .to_numpy(dtype="float64")
    )

    amount_prefix_sum = np.concatenate(
        [
            [0.0],
            np.cumsum(amount),
        ]
    )

    previous_amount_mean_group = np.full(
        len(group_starts),
        np.nan,
        dtype="float64",
    )

    has_prior_group = group_starts > 0

    previous_amount_mean_group[has_prior_group] = (
        amount_prefix_sum[
            group_starts[has_prior_group]
        ]
        / group_starts[has_prior_group]
    )

    previous_amount_mean = np.repeat(
        previous_amount_mean_group,
        group_lengths,
    )

    amount_minus_previous_mean = (
        amount
        - previous_amount_mean
    )

    merchant_values = (
        card_df["Merchant Name"]
        .to_numpy()
    )

    is_new_merchant = np.empty(
        len(card_df),
        dtype=bool,
    )

    seen_merchants = set()

    for start, end in zip(
        group_starts,
        group_ends,
    ):
        current_group_merchants = (
            merchant_values[start:end]
        )

        is_new_merchant[start:end] = np.array(
            [
                merchant not in seen_merchants
                for merchant
                in current_group_merchants
            ],
            dtype=bool,
        )

        seen_merchants.update(
            current_group_merchants.tolist()
        )

    return pd.DataFrame(
        {
            "raw_row_id":
                card_df["raw_row_id"].to_numpy(),
            "Timestamp":
                card_df["Timestamp"].to_numpy(),
            "amount_numeric":
                amount,
            "time_since_previous_transaction_min":
                time_since_previous_min,
            "transactions_last_1h":
                transactions_last_1h,
            "amount_minus_previous_mean":
                amount_minus_previous_mean,
            "is_new_merchant":
                is_new_merchant,
            "has_prior_card_history":
                has_prior_card_history,
        }
    )


NUMERIC_COLUMNS = [
    "amount_numeric",
    "time_since_previous_transaction_min",
    "transactions_last_1h",
    "amount_minus_previous_mean",
]

BOOLEAN_COLUMNS = [
    "is_new_merchant",
    "has_prior_card_history",
]

CATEGORICAL_COLUMNS = [
    "transaction_mode",
    "location_state",
    "hour_of_day",
    "day_of_week",
]

CORE_FEATURE_COLUMNS = (
    NUMERIC_COLUMNS
    + BOOLEAN_COLUMNS
    + CATEGORICAL_COLUMNS
)

PROHIBITED_DIRECT_COLUMNS = {
    "User",
    "Card",
    "Merchant Name",
    "Errors?",
    "Is Fraud?",
    "raw_row_id",
    "Timestamp",
}


def build_core_feature_frame(
    context_card_df,
    behavioral_df,
):
    timestamp = context_card_df["Timestamp"]

    frame = pd.DataFrame(
        {
            "raw_row_id":
                context_card_df["raw_row_id"].to_numpy(),
            "Timestamp":
                timestamp.to_numpy(),
            "amount_numeric":
                behavioral_df["amount_numeric"].to_numpy(),
            "time_since_previous_transaction_min":
                behavioral_df[
                    "time_since_previous_transaction_min"
                ].to_numpy(),
            "transactions_last_1h":
                behavioral_df[
                    "transactions_last_1h"
                ].to_numpy(),
            "amount_minus_previous_mean":
                behavioral_df[
                    "amount_minus_previous_mean"
                ].to_numpy(),
            "is_new_merchant":
                behavioral_df["is_new_merchant"].to_numpy(),
            "has_prior_card_history":
                behavioral_df[
                    "has_prior_card_history"
                ].to_numpy(),
            "transaction_mode":
                normalize_string(
                    context_card_df["Use Chip"]
                ).to_numpy(),
            "location_state":
                assign_location_state(
                    context_card_df
                ).to_numpy(),
            "hour_of_day":
                (
                    timestamp.dt.hour
                    .astype("Int8")
                    .astype("string")
                    .to_numpy()
                ),
            "day_of_week":
                (
                    timestamp.dt.dayofweek
                    .astype("Int8")
                    .astype("string")
                    .to_numpy()
                ),
        }
    )

    allowed_na = {
        "time_since_previous_transaction_min",
        "amount_minus_previous_mean",
    }

    unexpected_missing = {
        column:
            int(frame[column].isna().sum())
        for column
        in CORE_FEATURE_COLUMNS
        if (
            frame[column].isna().any()
            and column not in allowed_na
        )
    }

    if unexpected_missing:
        raise ValueError(
            "Unexpected missing: "
            + repr(unexpected_missing)
        )

    if (
        frame["location_state"]
        .eq("OTHER_INCONSISTENT")
        .any()
    ):
        raise ValueError(
            "OTHER_INCONSISTENT location state."
        )

    assert set(CORE_FEATURE_COLUMNS).isdisjoint(
        PROHIBITED_DIRECT_COLUMNS
    )

    return frame


print(
    "Core feature count:",
    len(CORE_FEATURE_COLUMNS),
)
assert len(CORE_FEATURE_COLUMNS) == 10

```

    Core feature count: 10


## 6. Strict-causal regression test trước full scan


```python

causal_test_df = pd.DataFrame(
    {
        "raw_row_id": [0, 1, 2, 3],
        "Timestamp": pd.to_datetime(
            [
                "2018-01-01 09:00:00",
                "2018-01-01 10:00:00",
                "2018-01-01 10:00:00",
                "2018-01-01 10:30:00",
            ]
        ),
        "Amount_numeric": [10.0, 20.0, 30.0, 40.0],
        "Merchant Name": [100, 200, 200, 100],
    }
)

causal_test = compute_causal_behavioral_features(
    causal_test_df
)

np.testing.assert_allclose(
    causal_test[
        "time_since_previous_transaction_min"
    ].to_numpy(),
    np.array(
        [np.nan, 60.0, 60.0, 30.0]
    ),
    equal_nan=True,
)

np.testing.assert_array_equal(
    causal_test[
        "transactions_last_1h"
    ].to_numpy(),
    np.array([0, 1, 1, 2]),
)

np.testing.assert_array_equal(
    causal_test[
        "is_new_merchant"
    ].to_numpy(),
    np.array(
        [True, True, True, False]
    ),
)

print(
    "M8.2 STRICT-CAUSAL REGRESSION GATE: PASS"
)

```

    M8.2 STRICT-CAUSAL REGRESSION GATE: PASS


## 7. Leakage-safe preprocessing helpers


```python

def make_one_hot_encoder(categories):
    kwargs = dict(
        categories=categories,
        handle_unknown="error",
        dtype=np.float32,
    )

    try:
        return OneHotEncoder(
            sparse_output=True,
            **kwargs,
        )
    except TypeError:
        return OneHotEncoder(
            sparse=True,
            **kwargs,
        )


def stable_sort_categories(column, values):
    values = [str(value) for value in values]

    if column in {
        "hour_of_day",
        "day_of_week",
    }:
        return sorted(
            values,
            key=lambda x: int(x),
        )

    return sorted(values)


@dataclass
class PreprocessingBundle:
    strategy: str
    scaler: StandardScaler
    category_vocab: dict
    encoder: OneHotEncoder
    feature_names: list
    fit_row_count: int
    fit_min_timestamp: pd.Timestamp
    fit_max_timestamp: pd.Timestamp


def build_encoder_from_train_vocab(
    category_vocab,
):
    categories = []

    for column in CATEGORICAL_COLUMNS:
        train_categories = stable_sort_categories(
            column,
            category_vocab[column],
        )

        if UNKNOWN_TOKEN in train_categories:
            raise ValueError(
                "Reserved unknown token đã xuất hiện trong TRAIN."
            )

        categories.append(
            train_categories
            + [UNKNOWN_TOKEN]
        )

    encoder = make_one_hot_encoder(
        categories
    )

    max_len = max(
        len(values)
        for values
        in categories
    )

    synthetic = {}

    for column, values in zip(
        CATEGORICAL_COLUMNS,
        categories,
    ):
        synthetic[column] = [
            values[
                index % len(values)
            ]
            for index
            in range(max_len)
        ]

    encoder.fit(
        pd.DataFrame(synthetic)
    )

    return encoder


def map_unknown_categories(
    frame,
    bundle,
):
    mapped = pd.DataFrame(
        index=frame.index
    )

    for column in CATEGORICAL_COLUMNS:
        values = (
            frame[column]
            .astype("string")
        )

        if values.isna().any():
            raise ValueError(
                f"Unexpected missing categorical: {column}"
            )

        known = set(
            bundle.category_vocab[column]
        )

        mapped[column] = values.where(
            values.isin(known),
            UNKNOWN_TOKEN,
        )

    return mapped


def count_unknown_categories(
    frame,
    bundle,
):
    result = {}
    for column in CATEGORICAL_COLUMNS:
        values = frame[column].astype("string")
        known = set(bundle.category_vocab[column])
        result[column] = int(
            (~values.isin(known)).sum()
        )
    return result


def build_feature_names(encoder):
    categorical_names = [
        f"cat__{name}"
        for name
        in encoder.get_feature_names_out(
            CATEGORICAL_COLUMNS
        )
    ]

    return (
        [
            f"num__{name}"
            for name
            in NUMERIC_COLUMNS
        ]
        +
        [
            f"bool__{name}"
            for name
            in BOOLEAN_COLUMNS
        ]
        +
        categorical_names
    )


def transform_with_bundle(
    frame,
    bundle,
):
    numeric = (
        frame[NUMERIC_COLUMNS]
        .astype("float64")
        .to_numpy()
    )

    numeric_scaled = (
        bundle.scaler
        .transform(numeric)
    )

    numeric_scaled = (
        np.nan_to_num(
            numeric_scaled,
            nan=0.0,
            posinf=np.inf,
            neginf=-np.inf,
        )
        .astype(np.float32)
    )

    boolean_matrix = (
        frame[BOOLEAN_COLUMNS]
        .astype(np.float32)
        .to_numpy()
    )

    mapped_categorical = (
        map_unknown_categories(
            frame,
            bundle,
        )
    )

    categorical_matrix = (
        bundle.encoder
        .transform(
            mapped_categorical
        )
    )

    X = sparse.hstack(
        [
            sparse.csr_matrix(
                numeric_scaled,
                dtype=np.float32,
            ),
            sparse.csr_matrix(
                boolean_matrix,
                dtype=np.float32,
            ),
            categorical_matrix,
        ],
        format="csr",
        dtype=np.float32,
    )

    if X.shape[1] != len(
        bundle.feature_names
    ):
        raise AssertionError(
            "Feature width mismatch."
        )

    if not np.isfinite(X.data).all():
        raise AssertionError(
            "NaN/inf sau preprocessing."
        )

    return X

```


## 8. Reconstruct W_SHORT preprocessing state từ TRAIN-only

Đây là full raw scan thứ nhất.

Quan trọng:

- `Is Fraud?` không được đọc;
- scaler/vocabulary chỉ học từ `2018-01-01 <= Timestamp < 2019-01-01`;
- VALIDATION và FINAL TEST không đóng góp learned state;
- causal history có thể sử dụng prior observable transactions vì đây là event state, không phải learned model state.



```python

feature_scalers = {
    column: StandardScaler()
    for column
    in NUMERIC_COLUMNS
}

category_counters = {
    column: Counter()
    for column
    in CATEGORICAL_COLUMNS
}

fit_row_count = 0
fit_min_timestamp = None
fit_max_timestamp = None

fit_card_count = 0
fit_raw_rows_seen = 0
fit_context_rows_seen = 0
fit_seen_card_keys = set()

fit_start = time.perf_counter()

for card_key, card_df in iter_card_blocks(
    RAW_PATH,
    CHUNK_SIZE,
    FIT_USECOLS,
):
    fit_card_count += 1
    fit_raw_rows_seen += len(card_df)

    if card_key in fit_seen_card_keys:
        raise RuntimeError(
            "Card block reappearance during fit pass: "
            + repr(card_key)
        )

    fit_seen_card_keys.add(card_key)

    # Giữ đúng M4.7 reconstruction boundary.
    context_card_df = (
        card_df.loc[
            card_df["Timestamp"]
            < VALIDATION_END
        ]
        .copy()
        .reset_index(drop=True)
    )

    if context_card_df.empty:
        continue

    fit_context_rows_seen += len(
        context_card_df
    )

    behavioral_df = (
        compute_causal_behavioral_features(
            context_card_df
        )
    )

    core_df = build_core_feature_frame(
        context_card_df,
        behavioral_df,
    )

    timestamp = core_df["Timestamp"]

    w_short_mask = (
        (timestamp >= W_SHORT_START)
        & (timestamp < TRAIN_END)
    )

    if not w_short_mask.any():
        continue

    train_sub = core_df.loc[
        w_short_mask
    ]

    for column in NUMERIC_COLUMNS:
        values = (
            train_sub[column]
            .astype("float64")
            .to_numpy()
        )

        valid_values = values[
            np.isfinite(values)
        ]

        if len(valid_values) > 0:
            feature_scalers[
                column
            ].partial_fit(
                valid_values.reshape(
                    -1,
                    1,
                )
            )

    fit_row_count += len(
        train_sub
    )

    current_min = (
        train_sub["Timestamp"].min()
    )
    current_max = (
        train_sub["Timestamp"].max()
    )

    if (
        fit_min_timestamp is None
        or current_min < fit_min_timestamp
    ):
        fit_min_timestamp = current_min

    if (
        fit_max_timestamp is None
        or current_max > fit_max_timestamp
    ):
        fit_max_timestamp = current_max

    for column in CATEGORICAL_COLUMNS:
        category_counters[column].update(
            train_sub[column]
            .astype("string")
            .value_counts()
            .to_dict()
        )

    if fit_card_count % 500 == 0:
        print(
            "Fit-pass cards:",
            f"{fit_card_count:,}",
            "| raw rows:",
            f"{fit_raw_rows_seen:,}",
        )

fit_elapsed = (
    time.perf_counter()
    - fit_start
)

assert fit_card_count == EXPECTED_CARD_COUNT
assert len(fit_seen_card_keys) == EXPECTED_CARD_COUNT
assert fit_raw_rows_seen == EXPECTED_RAW_ROWS
assert (
    fit_context_rows_seen
    == EXPECTED_CONTEXT_ROWS_BEFORE_VALIDATION_END
)
assert fit_row_count == EXPECTED_W_SHORT_ROWS
assert fit_min_timestamp >= W_SHORT_START
assert fit_max_timestamp < TRAIN_END

print("\nPreprocessing fit pass complete")
print("Cards:", f"{fit_card_count:,}")
print("Raw rows:", f"{fit_raw_rows_seen:,}")
print("Context rows:", f"{fit_context_rows_seen:,}")
print("W_SHORT fit rows:", f"{fit_row_count:,}")
print("Fit range:", fit_min_timestamp, "→", fit_max_timestamp)
print("Elapsed seconds:", round(fit_elapsed, 2))
print("\nM8.2 TRAIN-ONLY PREPROCESSING SOURCE GATE: PASS")

```

    Fit-pass cards: 500 | raw rows: 2,021,584
    Fit-pass cards: 1,500 | raw rows: 5,915,634
    Fit-pass cards: 2,000 | raw rows: 7,883,446
    Fit-pass cards: 2,500 | raw rows: 9,842,479
    Fit-pass cards: 3,000 | raw rows: 11,897,259
    Fit-pass cards: 3,500 | raw rows: 13,997,374
    Fit-pass cards: 4,000 | raw rows: 16,040,304
    Fit-pass cards: 4,500 | raw rows: 18,115,490
    Fit-pass cards: 5,000 | raw rows: 19,945,815
    Fit-pass cards: 6,000 | raw rows: 23,844,247
    
    Preprocessing fit pass complete
    Cards: 6,139
    Raw rows: 24,386,900
    Context rows: 23,038,920
    W_SHORT fit rows: 1,721,615
    Fit range: 2018-01-01 00:03:00 → 2018-12-31 23:58:00
    Elapsed seconds: 125.34
    
    M8.2 TRAIN-ONLY PREPROCESSING SOURCE GATE: PASS



```python

# Ghép bốn feature-wise scaler thành một StandardScaler
# đúng mechanics đã dùng ở M4.7.

means = []
variances = []
scales = []
sample_counts = []

for column in NUMERIC_COLUMNS:
    feature_scaler = (
        feature_scalers[column]
    )

    if not hasattr(
        feature_scaler,
        "mean_",
    ):
        raise RuntimeError(
            f"{column}: không có observed TRAIN value."
        )

    means.append(
        float(feature_scaler.mean_[0])
    )
    variances.append(
        float(feature_scaler.var_[0])
    )
    scales.append(
        float(feature_scaler.scale_[0])
    )
    sample_counts.append(
        int(
            np.asarray(
                feature_scaler.n_samples_seen_
            ).reshape(-1)[0]
        )
    )

combined_scaler = StandardScaler()
combined_scaler.mean_ = np.array(
    means,
    dtype=np.float64,
)
combined_scaler.var_ = np.array(
    variances,
    dtype=np.float64,
)
combined_scaler.scale_ = np.array(
    scales,
    dtype=np.float64,
)
combined_scaler.n_samples_seen_ = np.array(
    sample_counts,
    dtype=np.int64,
)
combined_scaler.n_features_in_ = len(
    NUMERIC_COLUMNS
)

category_vocab = {
    column:
        set(
            category_counters[
                column
            ].keys()
        )
    for column
    in CATEGORICAL_COLUMNS
}

encoder = build_encoder_from_train_vocab(
    category_vocab
)

reconstructed_feature_names = (
    build_feature_names(encoder)
)

w_short_bundle = PreprocessingBundle(
    strategy="W_SHORT",
    scaler=combined_scaler,
    category_vocab=category_vocab,
    encoder=encoder,
    feature_names=reconstructed_feature_names,
    fit_row_count=fit_row_count,
    fit_min_timestamp=fit_min_timestamp,
    fit_max_timestamp=fit_max_timestamp,
)

EXPECTED_W_SHORT_MEAN = np.array(
    [
        42.879027959212905,
        1197.3147570599144,
        0.2759606532238619,
        -0.4259839479714494,
    ],
    dtype=np.float64,
)

EXPECTED_W_SHORT_SCALE = np.array(
    [
        80.55065611888665,
        2113.3233635919028,
        0.6581493130130939,
        78.17323847945735,
    ],
    dtype=np.float64,
)

EXPECTED_CATEGORY_VOCAB_SIZE = {
    "transaction_mode": 3,
    "location_state": 3,
    "hour_of_day": 24,
    "day_of_week": 7,
}

np.testing.assert_allclose(
    w_short_bundle.scaler.mean_,
    EXPECTED_W_SHORT_MEAN,
    rtol=1e-12,
    atol=1e-12,
)

np.testing.assert_allclose(
    w_short_bundle.scaler.scale_,
    EXPECTED_W_SHORT_SCALE,
    rtol=1e-12,
    atol=1e-12,
)

for column, expected_size in (
    EXPECTED_CATEGORY_VOCAB_SIZE.items()
):
    assert (
        len(
            w_short_bundle.category_vocab[
                column
            ]
        )
        == expected_size
    )

assert (
    reconstructed_feature_names
    == canonical_feature_names
)
assert len(
    reconstructed_feature_names
) == EXPECTED_FEATURE_COUNT

print("Scaler mean:", w_short_bundle.scaler.mean_)
print("Scaler scale:", w_short_bundle.scaler.scale_)
print("Feature count:", len(reconstructed_feature_names))
print("\nM8.2 W_SHORT PREPROCESSING IDENTITY GATE: PASS")

```

    Scaler mean: [ 4.28790280e+01  1.19731476e+03  2.75960653e-01 -4.25983948e-01]
    Scaler scale: [8.05506561e+01 2.11332336e+03 6.58149313e-01 7.81732385e+01]
    Feature count: 47
    
    M8.2 W_SHORT PREPROCESSING IDENTITY GATE: PASS



## 9. Rebuild VALIDATION + materialize FINAL TEST representation

Đây là full raw scan thứ hai.

Cùng một causal stream được giữ tới trước `2019-11-01`.

Điều này cho phép:

- VALIDATION transaction dùng prior history;
- FINAL TEST transaction dùng toàn bộ prior observable history, bao gồm VALIDATION và các FINAL TEST transaction xảy ra sớm hơn;
- không có target label nào đi vào history;
- same-timestamp peer vẫn không làm history.

M8.2 sẽ dựng prediction-ready `X_final_test`, nhưng **không truyền matrix này vào estimator**.



```python

validation_matrix_parts = []
validation_target_parts = []
validation_row_parts = []
validation_timestamp_parts = []

final_matrix_parts = []
final_target_parts = []
final_row_parts = []
final_timestamp_parts = []

unknown_counts_validation = Counter()
unknown_counts_final = Counter()

build_card_count = 0
build_raw_rows_seen = 0
build_context_rows_seen = 0
build_seen_card_keys = set()

build_start = time.perf_counter()

for card_key, card_df in iter_card_blocks(
    RAW_PATH,
    CHUNK_SIZE,
    MATRIX_USECOLS,
):
    build_card_count += 1
    build_raw_rows_seen += len(
        card_df
    )

    if card_key in build_seen_card_keys:
        raise RuntimeError(
            "Card block reappearance during representation pass: "
            + repr(card_key)
        )

    build_seen_card_keys.add(
        card_key
    )

    context_card_df = (
        card_df.loc[
            card_df["Timestamp"]
            < FINAL_TEST_END
        ]
        .copy()
        .reset_index(drop=True)
    )

    if context_card_df.empty:
        continue

    build_context_rows_seen += len(
        context_card_df
    )

    behavioral_df = (
        compute_causal_behavioral_features(
            context_card_df
        )
    )

    core_df = build_core_feature_frame(
        context_card_df,
        behavioral_df,
    )

    timestamp = core_df["Timestamp"]

    validation_mask = (
        (timestamp >= TRAIN_END)
        & (timestamp < VALIDATION_END)
    )

    final_mask = (
        (timestamp >= VALIDATION_END)
        & (timestamp < FINAL_TEST_END)
    )

    if validation_mask.any():
        sub = core_df.loc[
            validation_mask
        ]

        X_part = transform_with_bundle(
            sub,
            w_short_bundle,
        )

        y_part = map_binary_target(
            context_card_df.loc[
                validation_mask,
                "Is Fraud?",
            ]
        )

        validation_matrix_parts.append(
            X_part
        )
        validation_target_parts.append(
            y_part
        )
        validation_row_parts.append(
            sub["raw_row_id"]
            .to_numpy(dtype=np.int64)
        )
        validation_timestamp_parts.append(
            sub["Timestamp"]
            .to_numpy(dtype="datetime64[ns]")
        )

        for key, value in (
            count_unknown_categories(
                sub,
                w_short_bundle,
            ).items()
        ):
            unknown_counts_validation[
                key
            ] += value

    if final_mask.any():
        sub = core_df.loc[
            final_mask
        ]

        X_part = transform_with_bundle(
            sub,
            w_short_bundle,
        )

        y_part = map_binary_target(
            context_card_df.loc[
                final_mask,
                "Is Fraud?",
            ]
        )

        final_matrix_parts.append(
            X_part
        )
        final_target_parts.append(
            y_part
        )
        final_row_parts.append(
            sub["raw_row_id"]
            .to_numpy(dtype=np.int64)
        )
        final_timestamp_parts.append(
            sub["Timestamp"]
            .to_numpy(dtype="datetime64[ns]")
        )

        for key, value in (
            count_unknown_categories(
                sub,
                w_short_bundle,
            ).items()
        ):
            unknown_counts_final[
                key
            ] += value

    if build_card_count % 500 == 0:
        print(
            "Representation-pass cards:",
            f"{build_card_count:,}",
            "| raw rows:",
            f"{build_raw_rows_seen:,}",
        )

build_elapsed = (
    time.perf_counter()
    - build_start
)

assert build_card_count == EXPECTED_CARD_COUNT
assert len(build_seen_card_keys) == EXPECTED_CARD_COUNT
assert build_raw_rows_seen == EXPECTED_RAW_ROWS
assert (
    build_context_rows_seen
    == EXPECTED_CONTEXT_ROWS_BEFORE_FINAL_TEST_END
)

print("\nRepresentation pass complete")
print("Cards:", f"{build_card_count:,}")
print("Raw rows:", f"{build_raw_rows_seen:,}")
print("Context rows < FINAL_TEST_END:", f"{build_context_rows_seen:,}")
print("Elapsed seconds:", round(build_elapsed, 2))
print("\nM8.2 CAUSAL STREAM / COVERAGE GATE: PASS")

```

    Representation-pass cards: 500 | raw rows: 2,021,584
    Representation-pass cards: 1,000 | raw rows: 4,066,794
    Representation-pass cards: 1,500 | raw rows: 5,915,634
    Representation-pass cards: 2,000 | raw rows: 7,883,446
    Representation-pass cards: 2,500 | raw rows: 9,842,479
    Representation-pass cards: 3,000 | raw rows: 11,897,259
    Representation-pass cards: 3,500 | raw rows: 13,997,374
    Representation-pass cards: 4,000 | raw rows: 16,040,304
    Representation-pass cards: 4,500 | raw rows: 18,115,490
    Representation-pass cards: 5,000 | raw rows: 19,945,815
    Representation-pass cards: 5,500 | raw rows: 21,956,418
    Representation-pass cards: 6,000 | raw rows: 23,844,247
    
    Representation pass complete
    Cards: 6,139
    Raw rows: 24,386,900
    Context rows < FINAL_TEST_END: 23,761,875
    Elapsed seconds: 147.12
    
    M8.2 CAUSAL STREAM / COVERAGE GATE: PASS



```python

X_validation_rebuilt = sparse.vstack(
    validation_matrix_parts,
    format="csr",
    dtype=np.float32,
)

y_validation_rebuilt = np.concatenate(
    validation_target_parts
).astype(
    np.int8,
    copy=False,
)

row_validation_rebuilt = np.concatenate(
    validation_row_parts
).astype(
    np.int64,
    copy=False,
)

timestamp_validation_rebuilt = np.concatenate(
    validation_timestamp_parts
).astype(
    "datetime64[ns]",
    copy=False,
)

X_final_test = sparse.vstack(
    final_matrix_parts,
    format="csr",
    dtype=np.float32,
)

y_final_test = np.concatenate(
    final_target_parts
).astype(
    np.int8,
    copy=False,
)

row_id_final_test = np.concatenate(
    final_row_parts
).astype(
    np.int64,
    copy=False,
)

timestamp_final_test = np.concatenate(
    final_timestamp_parts
).astype(
    "datetime64[ns]",
    copy=False,
)

print("X_validation_rebuilt:", X_validation_rebuilt.shape, "| nnz:", X_validation_rebuilt.nnz)
print("X_final_test:", X_final_test.shape, "| nnz:", X_final_test.nnz)
print("Validation fraud:", int(y_validation_rebuilt.sum()))
print("Final fraud:", int(y_final_test.sum()))

```

    X_validation_rebuilt: (712458, 47) | nnz: 6430339
    X_final_test: (722955, 47) | nnz: 6526714
    Validation fraud: 1052
    Final fraud: 1035


## 10. Exact validation reproduction — kiểm tra preprocessing/feature identity


```python

X_validation_saved = sparse.load_npz(
    X_VALIDATION_PATH
)
y_validation_saved = np.load(
    Y_VALIDATION_PATH,
    allow_pickle=False,
)
row_validation_saved = np.load(
    ROW_VALIDATION_PATH,
    allow_pickle=False,
)

assert sparse.isspmatrix_csr(
    X_validation_saved
)
assert sparse.isspmatrix_csr(
    X_validation_rebuilt
)

assert (
    X_validation_rebuilt.shape
    ==
    (
        EXPECTED_VALIDATION_ROWS,
        EXPECTED_FEATURE_COUNT,
    )
)

assert (
    X_validation_saved.shape
    ==
    X_validation_rebuilt.shape
)

assert (
    X_validation_saved.nnz
    == EXPECTED_VALIDATION_NNZ
)

assert (
    X_validation_rebuilt.nnz
    == EXPECTED_VALIDATION_NNZ
)

assert (
    X_validation_saved.dtype
    == np.float32
)

assert (
    X_validation_rebuilt.dtype
    == np.float32
)

validation_matrix_difference_nnz = int(
    (
        X_validation_saved
        != X_validation_rebuilt
    ).nnz
)

assert (
    validation_matrix_difference_nnz
    == 0
)

np.testing.assert_array_equal(
    y_validation_rebuilt,
    y_validation_saved,
)

np.testing.assert_array_equal(
    row_validation_rebuilt,
    row_validation_saved,
)

assert (
    int(y_validation_rebuilt.sum())
    == EXPECTED_VALIDATION_FRAUD
)

print(
    "Validation matrix difference nnz:",
    validation_matrix_difference_nnz,
)
print(
    "Validation rows/fraud:",
    len(y_validation_rebuilt),
    "/",
    int(y_validation_rebuilt.sum()),
)
print(
    "Unknown categories — VALIDATION:",
    dict(unknown_counts_validation),
)
print(
    "\nM8.2 VALIDATION REPRESENTATION REPRODUCTION GATE: PASS"
)

```

    Validation matrix difference nnz: 0
    Validation rows/fraud: 712458 / 1052
    Unknown categories — VALIDATION: {'transaction_mode': 0, 'location_state': 0, 'hour_of_day': 0, 'day_of_week': 0}
    
    M8.2 VALIDATION REPRESENTATION REPRODUCTION GATE: PASS


## 11. FINAL TEST population / lineage / representation gates


```python

row_train_saved = np.load(
    ROW_TRAIN_PATH,
    allow_pickle=False,
)

assert (
    X_final_test.shape
    ==
    (
        EXPECTED_FINAL_ROWS,
        EXPECTED_FEATURE_COUNT,
    )
)

assert sparse.isspmatrix_csr(
    X_final_test
)

assert (
    X_final_test.dtype
    == np.float32
)

assert np.isfinite(
    X_final_test.data
).all()

assert y_final_test.shape == (
    EXPECTED_FINAL_ROWS,
)

assert y_final_test.dtype == np.int8

assert (
    int(y_final_test.sum())
    == EXPECTED_FINAL_FRAUD
)

assert row_id_final_test.shape == (
    EXPECTED_FINAL_ROWS,
)

assert timestamp_final_test.shape == (
    EXPECTED_FINAL_ROWS,
)

assert (
    len(np.unique(row_id_final_test))
    == EXPECTED_FINAL_ROWS
)

assert (
    np.intersect1d(
        row_train_saved,
        row_id_final_test,
        assume_unique=True,
    ).size
    == 0
)

assert (
    np.intersect1d(
        row_validation_saved,
        row_id_final_test,
        assume_unique=True,
    ).size
    == 0
)

final_timestamp_index = pd.DatetimeIndex(
    timestamp_final_test
)

assert (
    final_timestamp_index.min()
    >= VALIDATION_END
)

assert (
    final_timestamp_index.max()
    < FINAL_TEST_END
)

# Không assert global chronological order:
# raw artifact được group theo User/Card, không theo global Timestamp.
# Boundary toàn partition và per-card causal ordering đã được kiểm tra.

print(
    "FINAL TEST rows/fraud:",
    len(y_final_test),
    "/",
    int(y_final_test.sum()),
)
print(
    "FINAL TEST timestamp:",
    final_timestamp_index.min(),
    "→",
    final_timestamp_index.max(),
)
print(
    "FINAL TEST matrix:",
    X_final_test.shape,
    X_final_test.dtype,
    "| nnz:",
    X_final_test.nnz,
)
print(
    "Unknown categories — FINAL TEST:",
    dict(unknown_counts_final),
)
print(
    "\nM8.2 FINAL TEST POPULATION / LINEAGE / REPRESENTATION GATE: PASS"
)

```

    FINAL TEST rows/fraud: 722955 / 1035
    FINAL TEST timestamp: 2019-06-01 00:02:00 → 2019-10-31 23:59:00
    FINAL TEST matrix: (722955, 47) float32 | nnz: 6526714
    Unknown categories — FINAL TEST: {'transaction_mode': 0, 'location_state': 0, 'hour_of_day': 0, 'day_of_week': 0}
    
    M8.2 FINAL TEST POPULATION / LINEAGE / REPRESENTATION GATE: PASS



## 12. Audit physical estimator artifact

Upstream M7.7 đã persist validation prediction/risk-score, nhưng M8.1 không cho phép **giả định** rằng fitted estimator đã được serialize.

Quy tắc notebook:

1. kiểm tra canonical M7.7 output directory;
2. nếu xuất hiện serialized estimator không được khai báo rõ → STOP để review thủ công;
3. nếu không có canonical serialized estimator → dùng deterministic reconstruction fallback đã được M8.1 cho phép;
4. reconstructed estimator phải tái tạo M7.7 VALIDATION prediction/risk-score trước khi được persist làm M8.2 physical subject.



```python

serialized_model_candidates = sorted(
    [
        *M7_07_DIR.glob("*.joblib"),
        *M7_07_DIR.glob("*.pkl"),
        *M7_07_DIR.glob("*.pickle"),
    ]
)

print(
    "Canonical M7.7 serialized model candidates:",
    [
        path.name
        for path
        in serialized_model_candidates
    ],
)

if serialized_model_candidates:
    raise RuntimeError(
        "Phát hiện serialized estimator trong M7.7 directory "
        "nhưng upstream CANON không khóa identity/path này. "
        "STOP để review thủ công trước khi sử dụng."
    )

print(
    "No canonical serialized estimator found in M7.7 output directory."
)
print(
    "Deterministic reconstruction fallback: REQUIRED"
)
print(
    "\nM8.2 PHYSICAL-ESTIMATOR DISCOVERY GATE: PASS"
)

```

    Canonical M7.7 serialized model candidates: []
    No canonical serialized estimator found in M7.7 output directory.
    Deterministic reconstruction fallback: REQUIRED
    
    M8.2 PHYSICAL-ESTIMATOR DISCOVERY GATE: PASS


## 13. Resolve exact selected RF parameters từ immutable M7.6 registry


```python

tuning_configs = m7_06_result[
    "tuning_configs"
]

rf_matches = [
    config
    for config
    in tuning_configs["RF"]
    if (
        config["config_id"]
        == SELECTED_CANDIDATE_ID
    )
]

assert len(rf_matches) == 1

selected_rf_config = copy.deepcopy(
    rf_matches[0]
)

rf_params = copy.deepcopy(
    selected_rf_config["params"]
)

assert selected_rf_config["role"] == "REFERENCE"
assert rf_params["n_estimators"] == 100
assert rf_params["criterion"] == "gini"
assert rf_params["max_depth"] is None
assert rf_params["min_samples_leaf"] == 1
assert rf_params["max_features"] == "sqrt"
assert rf_params["class_weight"] == "balanced"
assert rf_params["random_state"] == SELECTED_RANDOM_STATE

print("Selected config:", selected_rf_config["config_id"])
print(
    json.dumps(
        rf_params,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        default=str,
    )
)
print("\nM8.2 RF PARAMETER IDENTITY GATE: PASS")

```

    Selected config: RF-REF-100-GINI-SQRT-UNPRUNED-CW
    {
      "bootstrap": true,
      "ccp_alpha": 0.0,
      "class_weight": "balanced",
      "criterion": "gini",
      "max_depth": null,
      "max_features": "sqrt",
      "max_samples": null,
      "min_samples_leaf": 1,
      "min_samples_split": 2,
      "n_estimators": 100,
      "n_jobs": -1,
      "random_state": 42
    }
    
    M8.2 RF PARAMETER IDENTITY GATE: PASS



## 14. Deterministic estimator reconstruction + M7.7 VALIDATION reproduction

Cell này **chỉ fit trên W_SHORT TRAIN** và chỉ predict trên **VALIDATION**.

Nó không được truyền `X_final_test` vào model.

PASS condition mạnh:

- reconstructed class prediction phải giống M7.7 persisted prediction;
- reconstructed `predict_proba` positive-class score (float32) phải giống M7.7 persisted risk score;
- M7.8 comparator `>` tại threshold 0.50 phải tái tạo M7.7 prediction.



```python

X_train_saved = sparse.load_npz(
    X_TRAIN_PATH
)
y_train_saved = np.load(
    Y_TRAIN_PATH,
    allow_pickle=False,
)

assert sparse.isspmatrix_csr(
    X_train_saved
)
assert (
    X_train_saved.shape
    ==
    (
        EXPECTED_W_SHORT_ROWS,
        EXPECTED_FEATURE_COUNT,
    )
)
assert X_train_saved.dtype == np.float32
assert (
    int(y_train_saved.sum())
    == EXPECTED_W_SHORT_FRAUD
)

M7_07_RF_PRED_PATH = (
    M7_07_DIR
    / (
        "m7_07__"
        + SELECTED_CANDIDATE_ID
        + "__validation_y_pred.npy"
    )
)

M7_07_RF_SCORE_PATH = (
    M7_07_DIR
    / (
        "m7_07__"
        + SELECTED_CANDIDATE_ID
        + "__validation_risk_score.npy"
    )
)

assert M7_07_RF_PRED_PATH.exists()
assert M7_07_RF_SCORE_PATH.exists()

m7_07_rf_pred = np.load(
    M7_07_RF_PRED_PATH,
    allow_pickle=False,
)

m7_07_rf_score = np.load(
    M7_07_RF_SCORE_PATH,
    allow_pickle=False,
)

reconstruction_start = time.perf_counter()

selected_rf_estimator = (
    RandomForestClassifier(
        **copy.deepcopy(rf_params)
    )
)

with warnings.catch_warnings(
    record=True
) as captured_warnings:
    warnings.simplefilter("always")

    selected_rf_estimator.fit(
        X_train_saved,
        y_train_saved,
    )

reconstruction_fit_seconds = (
    time.perf_counter()
    - reconstruction_start
)

warning_records = [
    {
        "category":
            warning.category.__name__,
        "message":
            str(warning.message),
    }
    for warning
    in captured_warnings
]

assert len(warning_records) == 0

validation_pred_reconstructed = (
    selected_rf_estimator
    .predict(
        X_validation_saved
    )
    .astype(
        np.int8,
        copy=False,
    )
)

classes = np.asarray(
    selected_rf_estimator.classes_
)

positive_positions = np.flatnonzero(
    classes == 1
)

assert len(positive_positions) == 1
positive_index = int(
    positive_positions[0]
)

validation_score_reconstructed = (
    selected_rf_estimator
    .predict_proba(
        X_validation_saved
    )[
        :,
        positive_index,
    ]
    .astype(
        np.float32,
        copy=False,
    )
)

prediction_mismatch_count = int(
    np.count_nonzero(
        validation_pred_reconstructed
        != m7_07_rf_pred
    )
)

score_mismatch_count = int(
    np.count_nonzero(
        validation_score_reconstructed
        != m7_07_rf_score
    )
)

assert prediction_mismatch_count == 0
assert score_mismatch_count == 0

threshold_reconstructed = (
    validation_score_reconstructed
    > FINAL_THRESHOLD
).astype(
    np.int8
)

threshold_mismatch_count = int(
    np.count_nonzero(
        threshold_reconstructed
        != m7_07_rf_pred
    )
)

assert threshold_mismatch_count == 0

print(
    "Fit seconds:",
    round(
        reconstruction_fit_seconds,
        3,
    ),
)
print(
    "Validation prediction mismatch:",
    prediction_mismatch_count,
)
print(
    "Validation risk-score mismatch:",
    score_mismatch_count,
)
print(
    "Threshold > 0.50 mismatch:",
    threshold_mismatch_count,
)
print(
    "Warnings:",
    len(warning_records),
)
print(
    "\nM8.2 SELECTED RF RECONSTRUCTION / VALIDATION-REPRODUCTION GATE: PASS"
)

```

    Fit seconds: 36.695
    Validation prediction mismatch: 0
    Validation risk-score mismatch: 0
    Threshold > 0.50 mismatch: 0
    Warnings: 0
    
    M8.2 SELECTED RF RECONSTRUCTION / VALIDATION-REPRODUCTION GATE: PASS



## 15. Persist M8.2 audit artifacts

Chỉ sau khi VALIDATION representation và selected RF reconstruction đều PASS mới được persist M8.2 artifacts.

Artifacts này chuẩn bị cho M8.3 nhưng **M8.2 vẫn không tạo FINAL TEST prediction**.



```python

X_FINAL_PATH = (
    OUTPUT_DIR
    / "X_final_test_w_short.npz"
)

Y_FINAL_PATH = (
    OUTPUT_DIR
    / "y_final_test.npy"
)

ROW_FINAL_PATH = (
    OUTPUT_DIR
    / "row_id_final_test.npy"
)

TIMESTAMP_FINAL_PATH = (
    OUTPUT_DIR
    / "timestamp_final_test.npy"
)

PREPROCESSING_STATE_PATH = (
    OUTPUT_DIR
    / "m8_02_w_short_preprocessing_state.json"
)

ESTIMATOR_PATH = (
    OUTPUT_DIR
    / "m8_02_selected_rf_estimator.joblib"
)

REGISTRY_PATH = (
    OUTPUT_DIR
    / "m8_02_audit_registry.json"
)

MANIFEST_PATH = (
    OUTPUT_DIR
    / "m8_02_audit_manifest.json"
)

sparse.save_npz(
    X_FINAL_PATH,
    X_final_test,
    compressed=True,
)

np.save(
    Y_FINAL_PATH,
    y_final_test,
    allow_pickle=False,
)

np.save(
    ROW_FINAL_PATH,
    row_id_final_test,
    allow_pickle=False,
)

np.save(
    TIMESTAMP_FINAL_PATH,
    timestamp_final_test,
    allow_pickle=False,
)

preprocessing_state_payload = {
    "analysis_version":
        M8_02_ANALYSIS_VERSION,
    "strategy":
        "W_SHORT",
    "fit_source":
        "W_SHORT_TRAIN_ONLY",
    "fit_row_count":
        int(
            w_short_bundle.fit_row_count
        ),
    "fit_min_timestamp":
        str(
            w_short_bundle.fit_min_timestamp
        ),
    "fit_max_timestamp":
        str(
            w_short_bundle.fit_max_timestamp
        ),
    "numeric_columns":
        NUMERIC_COLUMNS,
    "numeric_mean":
        w_short_bundle.scaler.mean_.tolist(),
    "numeric_var":
        w_short_bundle.scaler.var_.tolist(),
    "numeric_scale":
        w_short_bundle.scaler.scale_.tolist(),
    "numeric_n_samples_seen":
        np.asarray(
            w_short_bundle.scaler.n_samples_seen_
        ).astype(int).tolist(),
    "categorical_columns":
        CATEGORICAL_COLUMNS,
    "category_vocab": {
        column:
            stable_sort_categories(
                column,
                values,
            )
        for column, values
        in w_short_bundle.category_vocab.items()
    },
    "unknown_token":
        UNKNOWN_TOKEN,
    "feature_names":
        reconstructed_feature_names,
    "feature_count":
        EXPECTED_FEATURE_COUNT,
    "matrix_format":
        "CSR",
    "matrix_dtype":
        "float32",
    "validation_exact_reproduction":
        True,
}

with open(
    PREPROCESSING_STATE_PATH,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        preprocessing_state_payload,
        file,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )

joblib.dump(
    selected_rf_estimator,
    ESTIMATOR_PATH,
)

for path in [
    X_FINAL_PATH,
    Y_FINAL_PATH,
    ROW_FINAL_PATH,
    TIMESTAMP_FINAL_PATH,
    PREPROCESSING_STATE_PATH,
    ESTIMATOR_PATH,
]:
    assert path.exists()
    assert path.stat().st_size > 0

print("Persisted:")
for path in [
    X_FINAL_PATH,
    Y_FINAL_PATH,
    ROW_FINAL_PATH,
    TIMESTAMP_FINAL_PATH,
    PREPROCESSING_STATE_PATH,
    ESTIMATOR_PATH,
]:
    print(
        " ",
        path.relative_to(PROJECT_ROOT),
        "|",
        f"{path.stat().st_size:,}",
        "bytes",
    )

print("\nM8.2 CORE ARTIFACT PERSISTENCE GATE: PASS")

```

    Persisted:
      data/processed/m8_02_final_test_artifact_audit/X_final_test_w_short.npz | 11,436,235 bytes
      data/processed/m8_02_final_test_artifact_audit/y_final_test.npy | 723,083 bytes
      data/processed/m8_02_final_test_artifact_audit/row_id_final_test.npy | 5,783,768 bytes
      data/processed/m8_02_final_test_artifact_audit/timestamp_final_test.npy | 5,783,768 bytes
      data/processed/m8_02_final_test_artifact_audit/m8_02_w_short_preprocessing_state.json | 3,278 bytes
      data/processed/m8_02_final_test_artifact_audit/m8_02_selected_rf_estimator.joblib | 50,564,265 bytes
    
    M8.2 CORE ARTIFACT PERSISTENCE GATE: PASS


## 16. Round-trip audit — không score FINAL TEST


```python

X_final_roundtrip = sparse.load_npz(
    X_FINAL_PATH
)

y_final_roundtrip = np.load(
    Y_FINAL_PATH,
    allow_pickle=False,
)

row_final_roundtrip = np.load(
    ROW_FINAL_PATH,
    allow_pickle=False,
)

timestamp_final_roundtrip = np.load(
    TIMESTAMP_FINAL_PATH,
    allow_pickle=False,
)

estimator_roundtrip = joblib.load(
    ESTIMATOR_PATH
)

assert sparse.isspmatrix_csr(
    X_final_roundtrip
)
assert (
    X_final_roundtrip.shape
    ==
    X_final_test.shape
)
assert (
    X_final_roundtrip.dtype
    ==
    np.float32
)
assert (
    (
        X_final_roundtrip
        != X_final_test
    ).nnz
    == 0
)

np.testing.assert_array_equal(
    y_final_roundtrip,
    y_final_test,
)

np.testing.assert_array_equal(
    row_final_roundtrip,
    row_id_final_test,
)

np.testing.assert_array_equal(
    timestamp_final_roundtrip,
    timestamp_final_test,
)

assert (
    estimator_roundtrip.get_params()
    ==
    selected_rf_estimator.get_params()
)

np.testing.assert_array_equal(
    estimator_roundtrip.classes_,
    selected_rf_estimator.classes_,
)

# Chỉ dùng VALIDATION để kiểm tra model round-trip.
roundtrip_validation_score = (
    estimator_roundtrip
    .predict_proba(
        X_validation_saved
    )[
        :,
        positive_index,
    ]
    .astype(
        np.float32,
        copy=False,
    )
)

assert np.array_equal(
    roundtrip_validation_score,
    m7_07_rf_score,
)

print(
    "FINAL matrix round-trip:",
    X_final_roundtrip.shape,
    "| nnz:",
    X_final_roundtrip.nnz,
)
print(
    "Estimator validation reproduction after round-trip: PASS"
)
print(
    "\nM8.2 ROUND-TRIP GATE: PASS"
)

```

    FINAL matrix round-trip: (722955, 47) | nnz: 6526714
    Estimator validation reproduction after round-trip: PASS
    
    M8.2 ROUND-TRIP GATE: PASS


## 17. Persist registry / manifest / fingerprints


```python

source_fingerprints = {
    "m4_manifest_sha256":
        sha256_file(M4_MANIFEST_PATH),
    "m4_feature_names_sha256":
        sha256_file(FEATURE_NAMES_PATH),
    "m4_x_train_w_short_sha256":
        sha256_file(X_TRAIN_PATH),
    "m4_x_validation_w_short_sha256":
        sha256_file(X_VALIDATION_PATH),
    "m4_y_train_w_short_sha256":
        sha256_file(Y_TRAIN_PATH),
    "m4_y_validation_sha256":
        sha256_file(Y_VALIDATION_PATH),
    "m7_06_registry_sha256":
        sha256_file(M7_06_RESULT_PATH),
    "m7_06_manifest_sha256":
        sha256_file(M7_06_MANIFEST_PATH),
    "m7_07_registry_sha256":
        sha256_file(M7_07_RESULT_PATH),
    "m7_07_manifest_sha256":
        sha256_file(M7_07_MANIFEST_PATH),
    "m7_07_selected_prediction_sha256":
        sha256_file(M7_07_RF_PRED_PATH),
    "m7_07_selected_risk_score_sha256":
        sha256_file(M7_07_RF_SCORE_PATH),
    "m7_08_registry_sha256":
        sha256_file(M7_08_RESULT_PATH),
    "m7_08_manifest_sha256":
        sha256_file(M7_08_MANIFEST_PATH),
}

generated_fingerprints = {
    "X_final_test_w_short_sha256":
        sha256_file(X_FINAL_PATH),
    "y_final_test_sha256":
        sha256_file(Y_FINAL_PATH),
    "row_id_final_test_sha256":
        sha256_file(ROW_FINAL_PATH),
    "timestamp_final_test_sha256":
        sha256_file(TIMESTAMP_FINAL_PATH),
    "preprocessing_state_sha256":
        sha256_file(PREPROCESSING_STATE_PATH),
    "selected_rf_estimator_sha256":
        sha256_file(ESTIMATOR_PATH),
}

technical_gates = {
    "G01_SOURCE_LOCATION": True,
    "G02_FROZEN_CONTRACT": True,
    "G03_UPSTREAM_HANDOFF": True,
    "G04_STRICT_CAUSAL_REGRESSION": True,
    "G05_TRAIN_ONLY_PREPROCESSING_SOURCE": True,
    "G06_W_SHORT_PREPROCESSING_IDENTITY": True,
    "G07_CAUSAL_STREAM_COVERAGE": True,
    "G08_VALIDATION_REPRESENTATION_REPRODUCTION": True,
    "G09_FINAL_TEST_POPULATION": (
        len(y_final_test)
        == EXPECTED_FINAL_ROWS
    ),
    "G10_FINAL_TEST_FRAUD_IDENTITY": (
        int(y_final_test.sum())
        == EXPECTED_FINAL_FRAUD
    ),
    "G11_FINAL_TEST_LINEAGE": (
        len(np.unique(row_id_final_test))
        == EXPECTED_FINAL_ROWS
    ),
    "G12_FINAL_TEST_REPRESENTATION": (
        sparse.isspmatrix_csr(
            X_final_test
        )
        and X_final_test.shape
        == (
            EXPECTED_FINAL_ROWS,
            EXPECTED_FEATURE_COUNT,
        )
        and X_final_test.dtype
        == np.float32
        and np.isfinite(
            X_final_test.data
        ).all()
    ),
    "G13_PHYSICAL_ESTIMATOR_DISCOVERY": (
        len(
            serialized_model_candidates
        )
        == 0
    ),
    "G14_RF_PARAMETER_IDENTITY": True,
    "G15_RF_VALIDATION_PREDICTION_REPRODUCTION": (
        prediction_mismatch_count
        == 0
    ),
    "G16_RF_VALIDATION_SCORE_REPRODUCTION": (
        score_mismatch_count
        == 0
    ),
    "G17_THRESHOLD_COMPARATOR_IDENTITY": (
        threshold_mismatch_count
        == 0
        and m7_08_manifest["comparator"]
        == EXPECTED_THRESHOLD_COMPARATOR
    ),
    "G18_CORE_ARTIFACT_PERSISTENCE": True,
    "G19_ROUND_TRIP": True,
    "G20_NO_FINAL_TEST_PREDICTION": True,
    "G21_NO_FINAL_TEST_METRIC": True,
}

assert all(
    technical_gates.values()
)

registry_payload = {
    "analysis_version":
        M8_02_ANALYSIS_VERSION,
    "work_type":
        "AUDIT_ONLY_NOT_OFFICIAL_PERFORMANCE_SCORING",
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
            rf_params,
        "probability_interface":
            "predict_proba / positive class = 1",
        "threshold":
            FINAL_THRESHOLD,
        "threshold_comparator":
            EXPECTED_THRESHOLD_COMPARATOR,
    },
    "preprocessing_audit": {
        "pipeline_version":
            m4_manifest["pipeline_version"],
        "fit_source":
            "W_SHORT_TRAIN_ONLY",
        "fit_rows":
            int(fit_row_count),
        "validation_matrix_difference_nnz":
            validation_matrix_difference_nnz,
        "feature_count":
            EXPECTED_FEATURE_COUNT,
        "feature_names_exact_match":
            (
                reconstructed_feature_names
                == canonical_feature_names
            ),
        "unknown_counts_validation":
            dict(
                unknown_counts_validation
            ),
        "unknown_counts_final_test":
            dict(
                unknown_counts_final
            ),
    },
    "final_test_identity": {
        "start_inclusive":
            str(VALIDATION_END),
        "end_exclusive":
            str(FINAL_TEST_END),
        "rows":
            int(len(y_final_test)),
        "fraud":
            int(y_final_test.sum()),
        "timestamp_min":
            str(
                final_timestamp_index.min()
            ),
        "timestamp_max":
            str(
                final_timestamp_index.max()
            ),
        "matrix_shape":
            list(
                X_final_test.shape
            ),
        "matrix_nnz":
            int(
                X_final_test.nnz
            ),
        "matrix_format":
            "CSR",
        "matrix_dtype":
            str(
                X_final_test.dtype
            ),
        "finite":
            bool(
                np.isfinite(
                    X_final_test.data
                ).all()
            ),
    },
    "model_reconstruction": {
        "canonical_serialized_model_found":
            False,
        "fallback_used":
            True,
        "fit_source":
            "M4.7 X_train_w_short / y_train_w_short",
        "validation_prediction_mismatch_count":
            prediction_mismatch_count,
        "validation_risk_score_mismatch_count":
            score_mismatch_count,
        "threshold_boundary_mismatch_count":
            threshold_mismatch_count,
        "warnings":
            warning_records,
    },
    "final_test_prediction_created":
        False,
    "final_test_metric_computed":
        False,
    "final_test_target_read_for_identity_audit":
        True,
    "train_validation_refit_performed":
        False,
    "source_fingerprints":
        source_fingerprints,
    "generated_fingerprints":
        generated_fingerprints,
    "technical_gates":
        technical_gates,
    "decision":
        "OPEN — REQUIRES AI RUNTIME REVIEW",
    "m8_3_official_scoring_authorization":
        "NOT YET — REQUIRES REVIEW",
}

manifest_payload = {
    "analysis_version":
        M8_02_ANALYSIS_VERSION,
    "final_test_rows":
        int(len(y_final_test)),
    "final_test_fraud":
        int(y_final_test.sum()),
    "feature_count":
        EXPECTED_FEATURE_COUNT,
    "matrix_format":
        "CSR",
    "matrix_dtype":
        "float32",
    "selected_candidate_id":
        SELECTED_CANDIDATE_ID,
    "threshold":
        FINAL_THRESHOLD,
    "threshold_comparator":
        EXPECTED_THRESHOLD_COMPARATOR,
    "validation_representation_reproduced_exactly":
        True,
    "selected_rf_validation_reproduced_exactly":
        True,
    "final_test_prediction_created":
        False,
    "final_test_metric_computed":
        False,
    "train_validation_refit_performed":
        False,
    "technical_gate_count":
        len(
            technical_gates
        ),
    "technical_gate_pass_count":
        int(
            sum(
                technical_gates.values()
            )
        ),
    "decision":
        "OPEN — REQUIRES AI RUNTIME REVIEW",
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

print("Registry:", REGISTRY_PATH)
print("Manifest:", MANIFEST_PATH)

print("\nGenerated fingerprints:")
for key, value in (
    generated_fingerprints.items()
):
    print(key, "→", value)

print(
    "\nM8.2 REGISTRY / MANIFEST PERSISTENCE GATE: PASS"
)

```

    Registry: /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/processed/m8_02_final_test_artifact_audit/m8_02_audit_registry.json
    Manifest: /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/processed/m8_02_final_test_artifact_audit/m8_02_audit_manifest.json
    
    Generated fingerprints:
    X_final_test_w_short_sha256 → 68b72dc7607b1c77edcd10b2e42cac0949ea2d2acd2de843925c0d9854a7c2bf
    y_final_test_sha256 → 5413d0d2934da55678faa40d1fd8abe442bd036d39f62ddd5faea19d67d682e2
    row_id_final_test_sha256 → 46a9bd1c8fd57551be17cf125c1b86a0ad4feef3d76e67e1dd389203cb1ebc36
    timestamp_final_test_sha256 → 3e87911541c2d06a2ac8a781d6834f249eaf761103d911fea5670a91119dd7f2
    preprocessing_state_sha256 → c1dc5486acdcd43f5f75fbd0a11bd7ad3a3b00761d50ae96b2b1fbe5b5d9ef98
    selected_rf_estimator_sha256 → 61bdeeba5fd163cce8a5a9efa028a9cdc2f95c0796822a5222c0604c0302115f
    
    M8.2 REGISTRY / MANIFEST PERSISTENCE GATE: PASS


## 18. Final technical gate — vẫn chưa authorize M8.3


```python

for gate_name, gate_value in (
    technical_gates.items()
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

assert len(technical_gates) == 21
assert all(
    technical_gates.values()
)

print(
    "\nM8.2 RUNTIME TECHNICAL GATE: PASS"
)

print(
    "\nDecision:",
    "OPEN — REQUIRES AI RUNTIME REVIEW",
)

print(
    "M8.3 official scoring authorization:",
    "NOT YET — REQUIRES REVIEW",
)

print(
    "\nIMPORTANT:",
    "Notebook này KHÔNG tạo FINAL TEST prediction "
    "và KHÔNG tính FINAL TEST metric."
)

```

    G01_SOURCE_LOCATION → PASS
    G02_FROZEN_CONTRACT → PASS
    G03_UPSTREAM_HANDOFF → PASS
    G04_STRICT_CAUSAL_REGRESSION → PASS
    G05_TRAIN_ONLY_PREPROCESSING_SOURCE → PASS
    G06_W_SHORT_PREPROCESSING_IDENTITY → PASS
    G07_CAUSAL_STREAM_COVERAGE → PASS
    G08_VALIDATION_REPRESENTATION_REPRODUCTION → PASS
    G09_FINAL_TEST_POPULATION → PASS
    G10_FINAL_TEST_FRAUD_IDENTITY → PASS
    G11_FINAL_TEST_LINEAGE → PASS
    G12_FINAL_TEST_REPRESENTATION → PASS
    G13_PHYSICAL_ESTIMATOR_DISCOVERY → PASS
    G14_RF_PARAMETER_IDENTITY → PASS
    G15_RF_VALIDATION_PREDICTION_REPRODUCTION → PASS
    G16_RF_VALIDATION_SCORE_REPRODUCTION → PASS
    G17_THRESHOLD_COMPARATOR_IDENTITY → PASS
    G18_CORE_ARTIFACT_PERSISTENCE → PASS
    G19_ROUND_TRIP → PASS
    G20_NO_FINAL_TEST_PREDICTION → PASS
    G21_NO_FINAL_TEST_METRIC → PASS
    
    M8.2 RUNTIME TECHNICAL GATE: PASS
    
    Decision: OPEN — REQUIRES AI RUNTIME REVIEW
    M8.3 official scoring authorization: NOT YET — REQUIRES REVIEW
    
    IMPORTANT: Notebook này KHÔNG tạo FINAL TEST prediction và KHÔNG tính FINAL TEST metric.



# 19. Findings / Runtime Review / Decision — sau khi kiểm tra output

## 19.1. Phạm vi review

Review này được thực hiện sau khi notebook M8.2 đã được chạy trên project thật.

Trình tự review:

```text
execution completeness
→ source / artifact integrity
→ preprocessing identity
→ VALIDATION exact reproduction
→ FINAL TEST population / lineage / representation
→ leakage / causal-history consistency
→ selected-estimator identity
→ M7.7 validation reproduction
→ threshold comparator identity
→ persistence / round-trip
→ M8.2 decision
```

Review **không đánh giá chất lượng predictive performance của FINAL TEST**.

M8.2 vẫn chỉ là:

`artifact / lineage / representation / model-state audit`

---

## 19.2. Execution completeness

Observed:

```text
Notebook execution:
COMPLETE

Runtime traceback:
NONE OBSERVED

Required technical sections:
EXECUTED

Final technical gate:
REACHED
```

Kết luận:

`PASS`

Không có cell runtime bắt buộc nào bị dừng trước M8.2 technical gate.

---

## 19.3. Raw / upstream artifact identity

Observed:

```text
Raw artifact:
card_transaction.v1.csv

Observed raw size:
2,354,626,737 bytes

Selected model:
Random Forest

Selected config:
RF-REF-100-GINI-SQRT-UNPRUNED-CW

Training window:
W_SHORT

Imbalance strategy:
CLASS_WEIGHT_BALANCED

random_state:
42

Frozen threshold:
0.50

Comparator:
risk_score > 0.50
```

M7.8 persisted pre-review state vẫn giữ:

`OPEN — REQUIRES M7.8 RUNTIME REVIEW`

và không bị chỉnh sửa retroactively.

Reviewed/frozen threshold được carry vào M8:

`0.50`

Kết luận:

`PASS`

Không phát hiện upstream identity contradiction.

---

## 19.4. Strict-causal behavioral contract

Strict-causal regression test:

`PASS`

Causal-history implementation tiếp tục giữ nguyên nguyên tắc:

```text
Timestamp(history) < Timestamp(current)

same-timestamp peer:
NOT HISTORY

current transaction:
NOT ITS OWN HISTORY

future transaction:
NOT HISTORY

target label:
NOT HISTORY STATE
```

Kết luận:

`PASS`

Không quan sát thấy causal-history violation trong audit path.

---

## 19.5. W_SHORT TRAIN-only preprocessing reconstruction

Observed:

```text
Cards scanned:
6,139

Raw rows scanned:
24,386,900

Context rows before VALIDATION_END:
23,038,920

W_SHORT preprocessing fit rows:
1,721,615

Fit range:
2018-01-01 00:03:00
→
2018-12-31 23:58:00
```

Learned preprocessing state được fit từ:

`W_SHORT TRAIN ONLY`

Không dùng VALIDATION để fit scaler/vocabulary.

Không dùng FINAL TEST để fit scaler/vocabulary.

Observed canonical output feature count:

`47`

Kết luận:

`PASS`

---

## 19.6. VALIDATION exact representation reproduction

Observed:

```text
VALIDATION rows:
712,458

VALIDATION fraud:
1,052

Rebuilt feature width:
47

Validation matrix difference nnz:
0
```

Unknown-category counts:

```text
transaction_mode:
0

location_state:
0

hour_of_day:
0

day_of_week:
0
```

`Validation matrix difference nnz = 0` nghĩa là rebuilt VALIDATION representation tái tạo **chính xác** persisted M4.7 matrix, không chỉ khớp row count hoặc feature width.

Kết luận:

`PASS — EXACT REPRODUCTION`

Đây là bằng chứng trực tiếp cho thấy frozen W_SHORT preprocessing / feature representation được reconstruction đúng trước khi materialize FINAL TEST.

---

## 19.7. FINAL TEST population identity

Observed:

```text
FINAL TEST rows:
722,955

FINAL TEST fraud:
1,035

Observed minimum timestamp:
2019-06-01 00:02:00

Observed maximum timestamp:
2019-10-31 23:59:00
```

Frozen partition contract:

```text
2019-06-01 <= Timestamp < 2019-11-01
```

Observed population khớp expected support.

Kết luận:

`PASS`

Không phát hiện:

```text
row-count mismatch
target-support mismatch
temporal-boundary mismatch
```

---

## 19.8. FINAL TEST lineage audit

Audit xác minh:

```text
FINAL TEST raw_row_id:
UNIQUE

Overlap với W_SHORT TRAIN:
0

Overlap với VALIDATION:
0
```

Kết luận:

`PASS`

Không phát hiện illegal development/final-test row overlap.

---

## 19.9. FINAL TEST representation audit

Observed:

```text
Shape:
(722955, 47)

Sparse format:
CSR

dtype:
float32

nnz:
6,526,714

NaN / inf:
NONE OBSERVED
```

Unknown-category counts:

```text
transaction_mode:
0

location_state:
0

hour_of_day:
0

day_of_week:
0
```

Feature contract giữ nguyên:

```text
10 pre-encoding semantic features
→
47 encoded features
→
CSR float32
```

Kết luận:

`PASS`

Không phát hiện:

```text
feature-schema mismatch
invalid representation values
unexpected unknown-category behavior
```

---

## 19.10. Physical selected-estimator audit

Canonical serialized estimator trong M7.7 output directory:

`NOT FOUND`

Do đó M8.2 sử dụng nhánh deterministic reconstruction đã được M8.1 cho phép.

Reconstruction subject:

```text
Training population:
W_SHORT TRAIN

Model:
RandomForestClassifier

Config ID:
RF-REF-100-GINI-SQRT-UNPRUNED-CW

class_weight:
balanced

random_state:
42

Risk-score interface:
predict_proba / positive class = 1
```

Exact parameter dictionary được lấy từ immutable M7.6 registry, không suy đoán từ config name.

Kết luận:

`PASS`

Physical estimator path được giải quyết bằng:

`DETERMINISTIC RECONSTRUCTION`

---

## 19.11. Selected RF validation reproduction

Observed:

```text
Fit warnings:
0

Validation prediction mismatch:
0

Validation risk-score mismatch:
0

Threshold > 0.50 mismatch:
0
```

Điều này xác minh reconstructed estimator tái tạo chính xác persisted M7.7 behavior trên VALIDATION:

```text
y_pred:
EXACT

risk_score:
EXACT

threshold comparator behavior:
EXACT
```

Kết luận:

`PASS — EXACT REPRODUCTION`

Không phát hiện model-state mismatch.

---

## 19.12. Positive-class / threshold comparator identity

Estimator `classes_` được kiểm tra runtime.

Positive class:

`1`

Risk-score source:

`predict_proba` tại vị trí class `1`

Comparator:

`risk_score > 0.50`

Observed threshold-boundary mismatch so với M7.7:

`0`

Kết luận:

`PASS`

Không phát hiện:

```text
positive-class mismatch
threshold/comparator mismatch
```

---

## 19.13. FINAL TEST protection trong M8.2

Source review xác nhận model inference chỉ được gọi trên:

`X_validation_saved`

M8.2 **không gọi**:

```text
predict(X_final_test)

predict_proba(X_final_test)
```

M8.2 cũng không compute:

```text
F1
Recall
Precision
Confusion Matrix
```

trên FINAL TEST.

Target FINAL TEST chỉ được đọc trong M8.2 để audit known expected population identity.

Observed:

```text
FINAL TEST prediction created:
FALSE

FINAL TEST performance metric computed:
FALSE
```

Kết luận:

`PASS`

M8.2 không bypass separation:

```text
M8.2
artifact / lineage / representation audit

→

M8.3
official frozen inference

→

M8.4
final metric reconstruction
```

---

## 19.14. Persistence / round-trip

Persisted M8.2 core artifacts gồm:

```text
X_final_test_w_short.npz
y_final_test.npy
row_id_final_test.npy
timestamp_final_test.npy
m8_02_w_short_preprocessing_state.json
m8_02_selected_rf_estimator.joblib
m8_02_audit_registry.json
m8_02_audit_manifest.json
```

Round-trip evidence:

```text
FINAL TEST matrix:
(722955, 47)

FINAL TEST matrix nnz:
6,526,714

Estimator validation reproduction after reload:
PASS
```

Artifact fingerprints được tạo cho generated artifacts và upstream evidence.

Kết luận:

`PASS`

Không quan sát thấy artifact persistence failure.

---

## 19.15. Technical gate summary

Observed:

```text
G01_SOURCE_LOCATION:
PASS

G02_FROZEN_CONTRACT:
PASS

G03_UPSTREAM_HANDOFF:
PASS

G04_STRICT_CAUSAL_REGRESSION:
PASS

G05_TRAIN_ONLY_PREPROCESSING_SOURCE:
PASS

G06_W_SHORT_PREPROCESSING_IDENTITY:
PASS

G07_CAUSAL_STREAM_COVERAGE:
PASS

G08_VALIDATION_REPRESENTATION_REPRODUCTION:
PASS

G09_FINAL_TEST_POPULATION:
PASS

G10_FINAL_TEST_FRAUD_IDENTITY:
PASS

G11_FINAL_TEST_LINEAGE:
PASS

G12_FINAL_TEST_REPRESENTATION:
PASS

G13_PHYSICAL_ESTIMATOR_DISCOVERY:
PASS

G14_RF_PARAMETER_IDENTITY:
PASS

G15_RF_VALIDATION_PREDICTION_REPRODUCTION:
PASS

G16_RF_VALIDATION_SCORE_REPRODUCTION:
PASS

G17_THRESHOLD_COMPARATOR_IDENTITY:
PASS

G18_CORE_ARTIFACT_PERSISTENCE:
PASS

G19_ROUND_TRIP:
PASS

G20_NO_FINAL_TEST_PREDICTION:
PASS

G21_NO_FINAL_TEST_METRIC:
PASS
```

Technical gate result:

`21 / 21 PASS`

---

## 19.16. STOP-condition review

M8.1 STOP conditions được đối chiếu với runtime evidence.

Observed:

```text
S01 Upstream handoff mismatch:
NOT TRIGGERED

S02 Model identity mismatch:
NOT TRIGGERED

S03 No trustworthy model state:
NOT TRIGGERED

S04 Unauthorized refit path:
NOT TRIGGERED

S05 FINAL TEST boundary mismatch:
NOT TRIGGERED

S06 Row-count mismatch:
NOT TRIGGERED

S07 Target-support mismatch:
NOT TRIGGERED

S08 Partition overlap:
NOT TRIGGERED

S09 Feature schema mismatch:
NOT TRIGGERED

S10 Learned-state contamination:
NOT TRIGGERED

S11 Causal-history violation:
NOT TRIGGERED

S12 Invalid representation values:
NOT TRIGGERED

S13 Positive-class mismatch:
NOT TRIGGERED

S14 Threshold/comparator mismatch:
NOT TRIGGERED

S15 Artifact persistence failure:
NOT TRIGGERED
```

Blocking issue:

`NONE OBSERVED`

---

## 19.17. Review finding — metadata cleanup

Runtime evidence đủ để quyết định M8.2.

Tuy nhiên, trước M8 final gate cần tiếp tục giữ metadata/fingerprint discipline.

Carry-forward cleanup:

```text
explicit runtime-version metadata
explicit no-retuning attestation
explicit final_test_accessed semantics by substep
canonical artifact-role documentation
```

Đây là:

`DOCUMENTATION / REGISTRY COMPLETENESS`

không phải:

`IDENTITY / LINEAGE / MODEL-STATE FAILURE`

Do đó không kích hoạt STOP condition.

Status:

`NON-BLOCKING — CARRY FORWARD`

---

# 20. M8.2 Decision

Sau runtime review:

```text
M8.2:
PASS

Work type:
FINAL TEST ARTIFACT / LINEAGE / REPRESENTATION / MODEL-STATE AUDIT

Raw / upstream identity:
PASS

W_SHORT preprocessing reconstruction:
PASS

VALIDATION representation reproduction:
EXACT

FINAL TEST population identity:
PASS

FINAL TEST lineage:
PASS

FINAL TEST representation:
PASS

Strict-causal history:
PASS

Selected RF identity:
PASS

Physical selected estimator:
DETERMINISTIC RECONSTRUCTION

M7.7 validation y_pred reproduction:
EXACT

M7.7 validation risk-score reproduction:
EXACT

Threshold comparator reproduction:
EXACT

FINAL TEST prediction in M8.2:
NONE

FINAL TEST metric in M8.2:
NONE

Artifact persistence / round-trip:
PASS

Technical gates:
21 / 21 PASS

STOP conditions triggered:
NONE

Blocking issue:
NONE
```

## Decision

`M8.2 — PASS`

## FINAL TEST performance status

`STILL UNREAD`

M8.2 không đưa ra performance conclusion.

## Frozen subject released to M8.3

```text
Training Window:
W_SHORT

Model:
Random Forest

Config:
RF-REF-100-GINI-SQRT-UNPRUNED-CW

Imbalance:
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

Representation:
47-column CSR float32
```

## M8.3 authorization

`AUTHORIZED`

Authorized next step:

`M8.3 — Frozen Final Inference Run`

M8.3 được phép:

```text
load audited FINAL TEST representation
load audited reconstructed selected estimator
assert model / feature / positive-class identity
run predict_proba exactly once under frozen protocol
apply frozen comparator risk_score > 0.50
persist risk_score / y_pred / lineage / metadata
fingerprint
round-trip
```

M8.3 không được:

```text
change model
change feature/preprocessing
change training population
TRAIN+VALIDATION refit
change random_state
change threshold
try alternate thresholds
perform calibration
use FINAL TEST result to modify the frozen subject
```

---

# 21. Handoff sang M8.3

M8.3 phải bắt đầu từ M8.2 artifacts đã audit.

Expected source artifacts:

```text
data/processed/m8_02_final_test_artifact_audit/
    X_final_test_w_short.npz
    y_final_test.npy
    row_id_final_test.npy
    timestamp_final_test.npy
    m8_02_w_short_preprocessing_state.json
    m8_02_selected_rf_estimator.joblib
    m8_02_audit_registry.json
    m8_02_audit_manifest.json
```

Precondition:

`M8.2 — PASS`

Official scoring authorization:

`YES`

M8.3 vẫn phải giữ separation:

```text
M8.3:
produce / persist official frozen prediction artifacts

M8.4:
reconstruct final metrics and confusion matrix

M8.5:
interpret final errors / temporal generalization
```

Không diễn giải final performance trong M8.3.

