# CANON-M5.6 — Tổng hợp Baseline Model Registry, Decision Log và M5 Gate

## Document status

Milestone:

`M5 — Modeling baseline`

Substep:

`M5.6 — Consolidated Baseline Modeling Registry / M5 Gate / Handoff M6`

Work type:

`CONSOLIDATION / GOVERNANCE / HANDOFF`

Runtime model experiment mới:

`NOT REQUIRED`

M5.6 không fit thêm model.

M5.6 không tuning.

M5.6 không mở FINAL TEST.

M5.6 tổng hợp và khóa những evidence đã được runtime-review ở:

- `M5.2 — Artifact Audit + Shared Baseline Runner`;
- `M5.3 — Logistic Regression baseline`;
- `M5.4 — Decision Tree baseline`;
- `M5.5 — Random Forest baseline`.

Trạng thái cuối của tài liệu này:

`M5 — PASS`

Handoff:

`READY FOR M6 — EVALUATION + ERROR ANALYSIS`

---

# 1. Câu hỏi trung tâm

M5.6 phải trả lời:

> Sau Milestone 5, project đã có đủ baseline model outputs, prediction/probability artifacts, metadata và integrity evidence để M6 phân tích metric, confusion matrix, error behavior và sự khác biệt giữa các run mà không phải train lại model chỉ vì thiếu evidence hay chưa?

Câu trả lời sau consolidation:

`YES`

M5.6 chỉ xác nhận:

`BASELINE MODELING EVIDENCE EXISTS AND IS M6-READY`

M5.6 không khóa:

- final model;
- final training window;
- final imbalance strategy;
- tuned hyperparameters;
- final numerical threshold;
- final-test performance.

---

# 2. Source hierarchy và evidence được dùng

M5.6 kế thừa trực tiếp:

1. `CANON-Kế hoạch Milestone 5 — Modeling baseline.md`
2. `CANON-M5.1 — Khóa Modeling Charter và baseline protocol.md`
3. `CANON-M5.2 — Audit modeling artifacts và khóa shared experiment runner.md`
4. `CANON-M5.3 — Logistic Regression baseline.md`
5. `CANON-M5.4 — Decision Tree baseline.md`
6. `CANON-M5.5 — Random Forest baseline.md`

Runtime evidence nguồn:

```text
05_03_logistic_regression_baseline_b04_reviewed.ipynb
05_04_decision_tree_baseline_reviewed.ipynb
05_05_random_forest_baseline_reviewed.ipynb
```

M5.6 không invent model score.

Mọi metric/runtime/config được ghi trong registry dưới đây đều lấy từ official runtime-reviewed baseline runs.

---

# 3. Contract cấp milestone được giữ nguyên

## 3.1. Canonical modeling artifacts

Artifact source:

`data/processed/m4_07_baseline_ready`

Pipeline version:

`M4.7-baseline-v1`

Matrix contract:

```text
format:
CSR

dtype:
float32

feature count:
47

target dtype:
int8

positive class:
fraud = 1
```

Training populations:

```text
W_SHORT:
1,721,615 rows
2,491 fraud

W_LONG:
6,855,270 rows
9,606 fraud

VALIDATION:
712,458 rows
1,052 fraud
```

M5 không rebuild feature representation.

M5 không thêm raw identifier vào classifier matrix.

M5 không refit preprocessing bằng logic mới.

Status:

`INHERITED — VERIFIED — LOCKED`

---

## 3.2. Model family scope

Core model families:

```text
Logistic Regression
Decision Tree
Random Forest
```

Official controlled runs required:

```text
LR-SHORT
LR-LONG

DT-SHORT
DT-LONG

RF-SHORT
RF-LONG
```

Observed:

`6 / 6 official baseline runs complete`

Status:

`COMPLETE`

---

## 3.3. Training-window contract

W_SHORT và W_LONG đều được giữ cho mỗi model family.

Trong cùng model-family pair:

`TRAINING WINDOW`

là biến chủ động thay đổi.

Các yếu tố phải giữ cùng gồm:

- validation population;
- target definition;
- feature version;
- preprocessing version;
- matrix schema;
- model config;
- imbalance strategy;
- metric semantics;
- threshold-policy class;
- warning/error handling.

Status:

`VERIFIED FOR ALL THREE MODEL FAMILIES`

Final training-window preference:

`OPEN`

---

## 3.4. Imbalance baseline

Core M5 policy:

`IMBALANCE_STRATEGY = NONE`

Observed official runs:

```text
Logistic Regression:
class_weight = None

Decision Tree:
class_weight = None

Random Forest:
class_weight = None
```

No resampling.

No SMOTE.

Status:

`VERIFIED — LOCKED FOR M5 BASELINE`

---

## 3.5. Metric contract

Primary:

`F1_fraud`

Secondary:

```text
Recall_fraud
Precision_fraud
```

Mandatory diagnostics:

```text
TP
FP
FN
TN
predicted-positive count
predicted-positive rate
```

Reference only:

`Accuracy`

Probability:

`preserve when supported`

All three model families support `predict_proba`, và positive-class score đã được persisted cho cả 6 official runs.

Status:

`VERIFIED`

---

## 3.6. Threshold policy

Baseline decision rule:

`DEFAULT_MODEL_DECISION_RULE`

M5 không tối ưu numerical threshold.

Final threshold:

`OPEN`

Status:

`INHERITED — VERIFIED — LOCKED FOR BASELINE`

---

## 3.7. FINAL TEST

M5.2–M5.5 đều có explicit FINAL TEST isolation gates.

Observed:

```text
FINAL TEST access:
NO

final_test_accessed:
false

Final-test artifacts in M5 output paths:
NONE
```

Status:

`PROTECTED`

---

# 4. Baseline Model Registry v1.0

## 4.1. Registry — identity, data contract và runtime

```text
| Experiment ID       | Family              | Config ID                         | Window  | Train rows | Train fraud | Val rows | Val fraud | Features | Imbalance | Random state | Threshold policy             | Fit s   | Predict s | Warnings | Integrity |
|---------------------|---------------------|-----------------------------------|---------|-----------:|------------:|---------:|----------:|---------:|-----------|-------------:|------------------------------|--------:|----------:|---------:|-----------|
| M5-LR-SHORT-B04     | Logistic Regression | LR-B04-LBFGS-L2-C1                | W_SHORT | 1,721,615  | 2,491       | 712,458  | 1,052     | 47       | NONE      | 42           | DEFAULT_MODEL_DECISION_RULE  | 0.822   | 0.032     | 0        | PASS      |
| M5-LR-LONG-B04      | Logistic Regression | LR-B04-LBFGS-L2-C1                | W_LONG  | 6,855,270  | 9,606       | 712,458  | 1,052     | 47       | NONE      | 42           | DEFAULT_MODEL_DECISION_RULE  | 3.614   | 0.025     | 0        | PASS      |
| M5-DT-SHORT-B01     | Decision Tree       | DT-B01-DEFAULT-GINI-UNPRUNED      | W_SHORT | 1,721,615  | 2,491       | 712,458  | 1,052     | 47       | NONE      | 42           | DEFAULT_MODEL_DECISION_RULE  | 6.662   | 0.048     | 0        | PASS      |
| M5-DT-LONG-B01      | Decision Tree       | DT-B01-DEFAULT-GINI-UNPRUNED      | W_LONG  | 6,855,270  | 9,606       | 712,458  | 1,052     | 47       | NONE      | 42           | DEFAULT_MODEL_DECISION_RULE  | 116.470 | 0.102     | 0        | PASS      |
| M5-RF-SHORT-B01     | Random Forest       | RF-B01-100-GINI-SQRT-BOOTSTRAP    | W_SHORT | 1,721,615  | 2,491       | 712,458  | 1,052     | 47       | NONE      | 42           | DEFAULT_MODEL_DECISION_RULE  | 40.698  | 0.868     | 0        | PASS      |
| M5-RF-LONG-B01      | Random Forest       | RF-B01-100-GINI-SQRT-BOOTSTRAP    | W_LONG  | 6,855,270  | 9,606       | 712,458  | 1,052     | 47       | NONE      | 42           | DEFAULT_MODEL_DECISION_RULE  | 627.714 | 1.494     | 0        | PASS      |
```

Shared versions:

```text
Feature:
Feature Specification v1.0

Preprocessing:
Preprocessing Specification v1.0

Matrix:
Baseline Matrix Schema v1.0

M4 pipeline:
M4.7-baseline-v1

M5 runner contract:
M5.2-shared-runner-v1
```

---

## 4.2. Registry — canonical metric bundle

```text
| Experiment ID       | F1_fraud | Recall_fraud | Precision_fraud | Accuracy ref | TP  | FP  | FN    | TN      | Predicted + | Predicted + rate |
|---------------------|---------:|-------------:|----------------:|-------------:|----:|----:|------:|--------:|------------:|-----------------:|
| M5-LR-SHORT-B04     | 0.337475 | 0.245247     | 0.540881        | 0.998578     | 258 | 219 | 794   | 711,187 | 477         | 0.000669513      |
| M5-LR-LONG-B04      | 0.039964 | 0.020913     | 0.448980        | 0.998516     | 22  | 27  | 1,030 | 711,379 | 49          | 0.000068776      |
| M5-DT-SHORT-B01     | 0.327056 | 0.313688     | 0.341615        | 0.998094     | 330 | 636 | 722   | 710,770 | 966         | 0.001355869      |
| M5-DT-LONG-B01      | 0.199134 | 0.196768     | 0.201558        | 0.997663     | 207 | 820 | 845   | 710,586 | 1,027       | 0.001441488      |
| M5-RF-SHORT-B01     | 0.366467 | 0.290875     | 0.495146        | 0.998515     | 306 | 312 | 746   | 711,094 | 618         | 0.000867420      |
| M5-RF-LONG-B01      | 0.152709 | 0.088403     | 0.560241        | 0.998551     | 93  | 73  | 959   | 711,333 | 166         | 0.000232996      |
```

Registry purpose:

`evidence availability`

Không phải:

`final model ranking`

M5.6 không chuyển bảng metric trên thành winner selection.

---

# 5. Official baseline configuration registry

## 5.1. Logistic Regression

Official config ID:

`LR-B04-LBFGS-L2-C1`

```text
estimator:
LogisticRegression

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

Official runs:

```text
M5-LR-SHORT-B04
M5-LR-LONG-B04
```

Technical convergence:

```text
W_SHORT:
n_iter_ = 19
warnings = 0

W_LONG:
n_iter_ = 22
warnings = 0
```

Technical history trước B04:

```text
B01:
FAILED_CONVERGENCE

B02:
ABORTED / COMPUTATIONALLY INEFFICIENT

B03:
FAILED_NUMERICAL_WARNING

B04:
TECHNICALLY VALID BASELINE
```

Các B01–B03 không được dùng làm official comparative baseline result.

---

## 5.2. Decision Tree

Official config ID:

`DT-B01-DEFAULT-GINI-UNPRUNED`

```text
criterion:
gini

splitter:
best

max_depth:
None

min_samples_split:
2

min_samples_leaf:
1

max_features:
None

class_weight:
None

ccp_alpha:
0.0

random_state:
42
```

Official runs:

```text
M5-DT-SHORT-B01
M5-DT-LONG-B01
```

Complexity evidence:

```text
W_SHORT:
depth = 44
nodes = 3,547
leaves = 1,774

W_LONG:
depth = 41
nodes = 28,651
leaves = 14,326
```

M5 không kết luận overfitting chỉ từ depth/node count.

Complexity tuning:

`DEFERRED TO M7`

---

## 5.3. Random Forest

Official config ID:

`RF-B01-100-GINI-SQRT-BOOTSTRAP`

```text
n_estimators:
100

criterion:
gini

max_depth:
None

min_samples_split:
2

min_samples_leaf:
1

max_features:
sqrt

bootstrap:
True

class_weight:
None

ccp_alpha:
0.0

max_samples:
None

random_state:
42

n_jobs:
-1
```

Official runs:

```text
M5-RF-SHORT-B01
M5-RF-LONG-B01
```

Forest-complexity evidence:

```text
W_SHORT:
100 / 100 estimators
451,532 total nodes
225,816 total leaves
depth min / mean / max = 33 / 41.43 / 50

W_LONG:
100 / 100 estimators
3,066,586 total nodes
1,533,343 total leaves
depth min / mean / max = 44 / 49.06 / 57
```

Resource workaround:

`NOT REQUIRED`

Hyperparameter tuning:

`DEFERRED TO M7`

---

# 6. Artifact Registry v1.0

M6 phải có thể phân tích mà không cần retrain chỉ vì thiếu prediction evidence.

## 6.1. Logistic Regression artifacts

Base directory:

`data/processed/m5_03_logistic_regression_baseline_b04`

```text
M5-LR-SHORT-B04__y_pred.npy
M5-LR-SHORT-B04__risk_score.npy
M5-LR-SHORT-B04__summary.json

M5-LR-LONG-B04__y_pred.npy
M5-LR-LONG-B04__risk_score.npy
M5-LR-LONG-B04__summary.json

m5_03_lr_b04_config_lock.json
m5_03_lr_b04_pair_manifest.json
```

Persistence gate:

`PASS`

Round-trip gate:

`PASS`

---

## 6.2. Decision Tree artifacts

Base directory:

`data/processed/m5_04_decision_tree_baseline`

```text
M5-DT-SHORT-B01__y_pred.npy
M5-DT-SHORT-B01__risk_score.npy
M5-DT-SHORT-B01__summary.json

M5-DT-LONG-B01__y_pred.npy
M5-DT-LONG-B01__risk_score.npy
M5-DT-LONG-B01__summary.json

m5_04_dt_baseline_config_lock.json
m5_04_dt_pair_manifest.json
```

Persistence gate:

`PASS`

Round-trip gate:

`PASS`

---

## 6.3. Random Forest artifacts

Base directory:

`data/processed/m5_05_random_forest_baseline`

```text
M5-RF-SHORT-B01__y_pred.npy
M5-RF-SHORT-B01__risk_score.npy
M5-RF-SHORT-B01__summary.json

M5-RF-LONG-B01__y_pred.npy
M5-RF-LONG-B01__risk_score.npy
M5-RF-LONG-B01__summary.json

m5_05_rf_baseline_config_lock.json
m5_05_rf_pair_manifest.json
```

Persistence gate:

`PASS`

Round-trip gate:

`PASS`

---

# 7. Metric implementation audit ở M5.6

M5.2 đã khóa canonical metric semantics:

```text
F1_fraud
Recall_fraud
Precision_fraud
Accuracy reference
TP / FP / FN / TN
predicted-positive count/rate
```

M5.3–M5.5 đều dùng:

```text
positive class = 1
confusion_matrix labels = [0, 1]
zero_division = 0
```

Decision Tree và Random Forest implementations có thêm input assertions so với Logistic Regression implementation.

Những assertions bổ sung này không thay đổi công thức metric hoặc output semantics trên validation population hợp lệ.

Observed metric arithmetic/integrity gates:

```text
M5.3:
PASS

M5.4:
PASS

M5.5:
PASS
```

Conclusion:

`NO METRIC-SEMANTIC DIVERGENCE DETECTED`

Status:

`CANONICAL METRIC CONTRACT PRESERVED`

---

# 8. Consolidated modeling findings

## M5.6-F01 — M4.7 artifacts are directly model-ready

Observed:

- canonical matrices load thành công;
- CSR/float32/47-column contract khớp;
- target int8/fraud=1 khớp;
- exact row/fraud counts khớp;
- finite checks PASS;
- stale-artifact signature checks PASS.

Finding:

M4.7 handoff đủ integrity để train cả ba core model families trực tiếp.

Status:

`VERIFIED`

---

## M5.6-F02 — All six official baseline runs exist

Observed:

```text
M5-LR-SHORT-B04
M5-LR-LONG-B04

M5-DT-SHORT-B01
M5-DT-LONG-B01

M5-RF-SHORT-B01
M5-RF-LONG-B01
```

Finding:

Model-family scope và W_LONG/W_SHORT coverage đã complete.

Status:

`VERIFIED`

---

## M5.6-F03 — All official runs are technically valid

Observed:

```text
warnings:
0 for all six official runs

integrity:
PASS for all six official runs
```

LR-specific convergence checks PASS.

DT/RF structural diagnostics present.

Finding:

Không có technical blocker còn mở trong six-run baseline registry.

Status:

`VERIFIED`

---

## M5.6-F04 — Controlled training-window pair integrity is preserved

Observed:

Mỗi family dùng same model config cho W_SHORT và W_LONG.

Finding:

Within-family window evidence đủ điều kiện để M6 đọc như controlled baseline evidence.

Status:

`VERIFIED`

---

## M5.6-F05 — No-intervention baseline was preserved

Observed:

```text
IMBALANCE_STRATEGY = NONE
```

for all six runs.

Finding:

M5 baseline không trộn class weighting/resampling vào một số model riêng lẻ.

Status:

`VERIFIED`

---

## M5.6-F06 — Probability/risk-score evidence is complete

Observed:

Positive-class risk-score artifacts tồn tại và round-trip PASS cho cả 6 runs.

Finding:

M6/M7 không cần retrain model chỉ để tái tạo probability evidence.

Status:

`VERIFIED`

---

## M5.6-F07 — Prediction evidence is complete

Observed:

`y_pred` artifacts tồn tại và round-trip PASS cho cả 6 runs.

Finding:

M6 có thể thực hiện confusion/error analysis trên persisted baseline predictions.

Status:

`VERIFIED`

---

## M5.6-F08 — Computational behavior differs substantially across model families/windows

Observed runtime evidence:

```text
Logistic Regression:
sub-second to few-second fit after technical solver remediation

Decision Tree:
~6.7 s W_SHORT
~116.5 s W_LONG

Random Forest:
~40.7 s W_SHORT
~627.7 s W_LONG
```

Finding:

Computational cost là modeling evidence cần được giữ cho later model selection.

Boundary:

Runtime không được dùng trong M5.6 để tự tuyên bố model winner.

Status:

`RECORDED`

---

## M5.6-F09 — Accuracy is not sufficient for fraud behavior interpretation

Observed:

Accuracy của các baseline runs đều rất cao do class imbalance, trong khi fraud-class Recall/F1 khác đáng kể.

Finding:

M6 phải tập trung vào canonical fraud metrics và error counts, không dựa vào Accuracy đơn lẻ.

Status:

`VERIFIED`

---

## M5.6-F10 — Baseline evidence contains material window-dependent behavior

Observed:

Trong từng model family, W_SHORT/W_LONG tạo khác biệt rõ ở F1/Recall/Precision/predicted-positive behavior.

Finding:

Training-window question có evidence để M6 phân tích.

Decision:

M5.6 không khóa winner.

Status:

`EVIDENCE AVAILABLE — PREFERENCE OPEN`

---

## M5.6-F11 — FINAL TEST remains fully protected

Observed:

M5.2, M5.3, M5.4 và M5.5 đều PASS final-test isolation.

Finding:

Không có final-test result nào được dùng để quyết định baseline config/model/window/threshold.

Status:

`VERIFIED`

---

## M5.6-F12 — M6 can proceed without baseline retraining

Observed M6 inputs now include:

- six `y_pred` artifacts;
- six risk-score artifacts;
- six summaries;
- config locks;
- pair manifests;
- canonical validation target from M4.7;
- complete metric/runtime/integrity metadata.

Finding:

M6 có đủ modeling evidence để bắt đầu evaluation/error analysis.

Status:

`READY FOR M6`

---

# 9. M5 Decision Log — consolidated

## M5-D01 — Milestone role

Decision:

M5 là:

`BASELINE MODELING MILESTONE`

Không phải:

`FINAL MODEL-SELECTION MILESTONE`

Status:

`LOCKED`

---

## M5-D02 — Canonical input source

Decision:

M5 sử dụng:

`data/processed/m4_07_baseline_ready`

Không rebuild feature/preprocessing bằng logic mới.

Status:

`INHERITED — VERIFIED — LOCKED`

---

## M5-D03 — Matrix/target contract

Decision:

```text
X:
47-column CSR float32

y:
int8

fraud:
1
```

Status:

`INHERITED — VERIFIED — LOCKED`

---

## M5-D04 — Core model families

Decision:

```text
Logistic Regression
Decision Tree
Random Forest
```

Observed:

All three families completed official W_SHORT/W_LONG baselines.

Status:

`COMPLETE — LOCKED`

---

## M5-D05 — Training-window candidates

Decision:

Giữ cả:

```text
W_SHORT
W_LONG
```

Status:

`INHERITED — VERIFIED — LOCKED`

Final preference:

`OPEN`

---

## M5-D06 — Controlled pair rule

Decision:

Within each model family, W_SHORT/W_LONG run phải dùng same baseline config.

Observed:

```text
LR pair:
VERIFIED

DT pair:
VERIFIED

RF pair:
VERIFIED
```

Status:

`LOCKED`

---

## M5-D07 — Imbalance baseline

Decision:

`IMBALANCE_STRATEGY = NONE`

Status:

`VERIFIED — LOCKED FOR M5`

Later imbalance experiments:

`DEFERRED TO M7`

---

## M5-D08 — Randomness

Decision:

`RANDOM_STATE = 42`

khi estimator/operation có randomness.

Status:

`VERIFIED — LOCKED`

---

## M5-D09 — Canonical metrics

Decision:

```text
Primary:
F1_fraud

Secondary:
Recall_fraud
Precision_fraud

Mandatory:
TP / FP / FN / TN
predicted-positive count/rate

Reference:
Accuracy
```

Status:

`VERIFIED — LOCKED`

---

## M5-D10 — Probability output

Decision:

Giữ positive-class probability/risk score khi model hỗ trợ.

Observed:

All three families support and persisted probability evidence.

Status:

`VERIFIED — LOCKED`

---

## M5-D11 — Official Logistic Regression baseline

Decision:

`LR-B04-LBFGS-L2-C1`

Official runs:

```text
M5-LR-SHORT-B04
M5-LR-LONG-B04
```

Status:

`LOCKED FOR M5 BASELINE`

---

## M5-D12 — Official Decision Tree baseline

Decision:

`DT-B01-DEFAULT-GINI-UNPRUNED`

Official runs:

```text
M5-DT-SHORT-B01
M5-DT-LONG-B01
```

Status:

`LOCKED FOR M5 BASELINE`

---

## M5-D13 — Official Random Forest baseline

Decision:

`RF-B01-100-GINI-SQRT-BOOTSTRAP`

Official runs:

```text
M5-RF-SHORT-B01
M5-RF-LONG-B01
```

Status:

`LOCKED FOR M5 BASELINE`

---

## M5-D14 — Hidden tuning prohibition

Decision:

Không chọn baseline config bằng validation search.

Observed:

- DT config khóa trước result;
- RF config khóa trước result;
- LR technical config changes B01→B04 dựa trên convergence/computational/numerical diagnostics, không dựa trên việc chọn validation metric cao nhất.

Status:

`VERIFIED`

Systematic tuning:

`DEFERRED TO M7`

---

## M5-D15 — Threshold

Decision:

M5 dùng baseline default model decision rule.

Final numerical threshold:

`OPEN`

Status:

`LOCKED FOR BASELINE / FINAL OPEN`

---

## M5-D16 — Computational constraint policy outcome

Decision:

Không silent subsampling hoặc asymmetric config.

Observed:

```text
LR:
technical solver remediation documented;
official B04 pair full-data

DT:
full-data pair complete

RF:
full-data pair complete;
resource workaround not required
```

Status:

`VERIFIED`

---

## M5-D17 — FINAL TEST

Decision:

Không access FINAL TEST trong M5.

Status:

`INHERITED — VERIFIED — LOCKED`

---

## M5-D18 — Training-window winner

Decision:

M5.6 không khóa W_SHORT hoặc W_LONG làm final winner.

Status:

`OPEN — HANDOFF M6/M7`

---

## M5-D19 — Model-family winner

Decision:

M5.6 không khóa Logistic Regression, Decision Tree hoặc Random Forest làm final model.

Status:

`OPEN — HANDOFF M6/M7`

---

## M5-D20 — M5 artifact handoff

Decision:

Six-run prediction/probability/summary artifacts là official baseline modeling evidence handoff sang M6.

Status:

`LOCKED`

---

## M5-D21 — M6 handoff

Decision:

M5 đã complete baseline modeling scope.

Next:

`M6 — Evaluation + Error Analysis`

Status:

`READY`

---

# 10. M5 Open Questions

## M5-O01 — Training-window preference

Status:

`OPEN`

M6 sẽ đọc comparative validation behavior.

Nếu cần robustness/selection protocol:

`M7`

---

## M5-O02 — Model-family preference

Status:

`OPEN`

M6:

`evaluation + error analysis`

M7:

`selection protocol`

---

## M5-O03 — Feature experiments ngoài baseline v1

Status:

`DEFERRED`

Không thuộc core M5 baseline.

---

## M5-O04 — Class imbalance strategy

Status:

`OPEN — M7`

Authorized candidates từ upstream protocol gồm:

```text
NONE
CLASS_WEIGHT
RANDOM_OVERSAMPLING
RANDOM_UNDERSAMPLING
SMOTE conditional
```

M5 chỉ dùng:

`NONE`

---

## M5-O05 — Temporal CV

Status:

`DEFERRED TO M7`

Primary strategy nếu dùng:

`FORWARD / EXPANDING TEMPORAL CV`

---

## M5-O06 — Hyperparameter tuning

Status:

`DEFERRED TO M7`

Search phải:

- nhỏ;
- justified;
- declared before result.

---

## M5-O07 — Final threshold

Status:

`DEFERRED TO M7`

Selection source:

`external VALIDATION`

FINAL TEST:

`PROHIBITED FOR THRESHOLD SELECTION`

---

## M5-O08 — Final performance

Status:

`OPEN`

FINAL TEST chỉ được mở sau khi upstream pipeline/model/threshold đã freeze theo protocol sau M7.

---

# 11. M5 Gate

M5 Gate được đánh giá theo 12 gate trong Milestone 5 plan.

## G01 — M4 artifact handoff integrity

Requirement:

- canonical M4.7 artifacts load thành công;
- manifest/schema khớp;
- stale pre-fix artifacts không được dùng.

Evidence:

M5.2 artifact audit PASS.

Exact shape/dtype/nnz signatures verified.

Result:

`PASS`

---

## G02 — Modeling contract preserved

Requirement:

- 47-column schema giữ nguyên;
- target mapping đúng;
- raw identifiers không thêm lại;
- preprocessing không refit tùy tiện.

Evidence:

M5.2–M5.5 input/integrity gates PASS.

Result:

`PASS`

---

## G03 — Model family scope complete

Requirement:

Baseline runs tồn tại cho:

- Logistic Regression;
- Decision Tree;
- Random Forest.

Evidence:

```text
LR:
2 official runs

DT:
2 official runs

RF:
2 official runs
```

Result:

`PASS`

---

## G04 — W_LONG/W_SHORT coverage

Requirement:

Mỗi family có controlled run cho cả windows.

Evidence:

```text
LR-SHORT + LR-LONG:
VALID

DT-SHORT + DT-LONG:
VALID

RF-SHORT + RF-LONG:
VALID
```

Result:

`PASS`

---

## G05 — No-intervention baseline

Requirement:

`IMBALANCE_STRATEGY = NONE`

Evidence:

All six official runs:

`NONE`

Result:

`PASS`

---

## G06 — No hidden tuning

Requirement:

Baseline config khai báo trước comparative result và không chọn từ validation search.

Evidence:

- LR official B04 technical remediation documented and warning/runtime-driven;
- DT B01 config pre-result locked;
- RF B01 config pre-result locked;
- no class-weight/resampling/threshold search;
- no systematic hyperparameter search.

Result:

`PASS`

---

## G07 — Probability/risk-score preservation

Requirement:

Probability được giữ khi model hỗ trợ.

Evidence:

Six risk-score artifacts persisted and round-trip verified.

Result:

`PASS`

---

## G08 — Canonical metric implementation

Requirement:

Same canonical fraud-class metric semantics.

Evidence:

All runs use:

```text
F1_fraud
Recall_fraud
Precision_fraud
TP / FP / FN / TN
predicted-positive count/rate
Accuracy reference
```

Positive class:

`1`

Metric arithmetic/integrity gates PASS in M5.3–M5.5.

Result:

`PASS`

---

## G09 — Reproducibility

Requirement:

- randomness controlled;
- random state logged;
- config logged;
- run identity reproducible.

Evidence:

- unique experiment IDs;
- config-lock artifacts;
- random state recorded;
- model params in summaries;
- prediction/risk-score/summary persistence;
- pair manifests.

Result:

`PASS`

---

## G10 — FINAL TEST isolation

Requirement:

`FINAL TEST ACCESS = NO`

Evidence:

M5.2–M5.5 final-test isolation gates PASS.

Result:

`PASS`

---

## G11 — Experiment Registry complete

Requirement:

Six baseline runs có metadata/output rõ.

Evidence:

`Baseline Model Registry v1.0`

contains all six official runs with:

- identity;
- data counts;
- config;
- runtime;
- warnings;
- canonical metrics;
- prediction/probability locations;
- integrity status.

Result:

`PASS`

---

## G12 — M6 readiness

Requirement:

M6 có thể phân tích:

- Precision/Recall/F1;
- Confusion Matrix;
- false positives;
- false negatives;
- predicted-positive behavior;
- window differences;
- model-family differences;

mà không phải retrain model chỉ vì thiếu prediction evidence.

Evidence:

Six persisted prediction artifacts + six probability artifacts + summaries + config/pair manifests are available.

Result:

`PASS`

---

# 12. M5 Gate summary

```text
G01 — M4 artifact handoff integrity          PASS
G02 — Modeling contract preserved           PASS
G03 — Model family scope complete           PASS
G04 — W_LONG/W_SHORT coverage               PASS
G05 — No-intervention baseline              PASS
G06 — No hidden tuning                      PASS
G07 — Probability/risk-score preservation   PASS
G08 — Canonical metric implementation       PASS
G09 — Reproducibility                       PASS
G10 — FINAL TEST isolation                  PASS
G11 — Experiment Registry complete          PASS
G12 — M6 readiness                          PASS
```

Overall:

`12 / 12 PASS`

Blocking issue:

`NONE`

Milestone result:

`M5 — PASS`

---

# 13. Những điều M5 PASS không có nghĩa là gì

M5 PASS không yêu cầu:

- F1 phải đạt một ngưỡng cụ thể;
- Recall phải đạt một ngưỡng cụ thể;
- Precision phải đạt một ngưỡng cụ thể;
- một model family phải “thắng”;
- W_SHORT hoặc W_LONG phải “thắng”;
- class weighting phải được dùng;
- resampling phải được dùng;
- SMOTE phải được dùng;
- hyperparameter phải được tối ưu;
- final threshold phải được khóa;
- FINAL TEST phải được mở.

Một baseline score thấp nhưng experiment đúng vẫn là valid modeling evidence.

M5 PASS có nghĩa:

`baseline modeling protocol + evidence + artifacts are complete`

không có nghĩa:

`final fraud model has been selected`

---

# 14. M5 completion-definition audit

M5 plan yêu cầu project có thể trả lời chắc chắn các câu hỏi sau.

## C01 — M4 artifacts có load và train model trực tiếp được không?

Answer:

`YES`

Evidence:

M5.2 artifact compatibility audit PASS và cả ba family đã fit full canonical matrices.

---

## C02 — Logistic Regression baseline đã chạy hợp lệ chưa?

Answer:

`YES`

Official config:

`LR-B04-LBFGS-L2-C1`

Official runs:

`VALID`

---

## C03 — Decision Tree baseline đã chạy hợp lệ chưa?

Answer:

`YES`

Official config:

`DT-B01-DEFAULT-GINI-UNPRUNED`

Official runs:

`VALID`

---

## C04 — Random Forest baseline đã chạy hợp lệ chưa?

Answer:

`YES`

Official config:

`RF-B01-100-GINI-SQRT-BOOTSTRAP`

Official runs:

`VALID`

---

## C05 — Cả W_LONG và W_SHORT đã được xử lý bằng controlled setup chưa?

Answer:

`YES`

Observed:

`3 / 3 model-family controlled pairs verified`

---

## C06 — Có giữ probability/risk score khi model hỗ trợ không?

Answer:

`YES`

Observed:

`6 / 6 official runs persisted probability evidence`

---

## C07 — Có đủ prediction/modeling evidence để M6 phân tích lỗi không?

Answer:

`YES`

Observed:

`6 / 6 y_pred artifacts persisted + round-trip verified`

---

## C08 — Có FINAL TEST leakage không?

Answer:

`NO`

Observed:

`FINAL TEST PROTECTED`

---

## C09 — Có hidden tuning hoặc imbalance intervention bị trộn vào baseline không?

Answer:

`NO EVIDENCE OF SUCH VIOLATION`

Observed:

- config locks trước result;
- imbalance NONE;
- no systematic validation search;
- technical remediation documented;
- final threshold still OPEN.

---

# 15. Handoff package sang M6

M6 nhận:

## 15.1. Canonical input/evaluation context

```text
M4.7 artifacts
Feature Specification v1.0
Preprocessing Specification v1.0
Baseline Matrix Schema v1.0
validation target y_validation
```

---

## 15.2. Six baseline experiment summaries

```text
M5-LR-SHORT-B04
M5-LR-LONG-B04

M5-DT-SHORT-B01
M5-DT-LONG-B01

M5-RF-SHORT-B01
M5-RF-LONG-B01
```

---

## 15.3. Six prediction artifacts

```text
LR SHORT y_pred
LR LONG y_pred

DT SHORT y_pred
DT LONG y_pred

RF SHORT y_pred
RF LONG y_pred
```

---

## 15.4. Six probability/risk-score artifacts

```text
LR SHORT risk score
LR LONG risk score

DT SHORT risk score
DT LONG risk score

RF SHORT risk score
RF LONG risk score
```

---

## 15.5. Config and pair manifests

```text
LR B04 config lock + pair manifest
DT B01 config lock + pair manifest
RF B01 config lock + pair manifest
```

---

# 16. M6 authorized questions

M6 có thể bắt đầu trả lời bằng persisted evidence:

- baseline Precision/Recall/F1 khác nhau như thế nào;
- false positive / false negative burden của từng run;
- predicted-positive behavior;
- cùng model family, W_SHORT/W_LONG khác nhau như thế nào;
- cùng training-window population, các model family có error behavior khác nhau như thế nào;
- những loại lỗi nào đáng chú ý;
- probability distribution/ranking behavior cần kiểm tra gì thêm;
- baseline nào cần được đưa sang selection/tuning protocol ở M7.

M6 phải giữ distinction:

`DESCRIPTIVE / COMPARATIVE EVALUATION`

khác với:

`FINAL MODEL SELECTION`

---

# 17. M6/M7 boundary sau M5

M6:

```text
Evaluation
Confusion Matrix
Precision
Recall
F1
Error analysis
Predicted-positive behavior
Comparative baseline behavior
```

M7:

```text
Model selection
Temporal CV
Hyperparameter tuning
Imbalance experiments
Threshold selection
Robustness checks
```

FINAL TEST:

`STILL PROTECTED`

---

# 18. Final M5 Decision Block

```text
Milestone:
M5 — MODELING BASELINE

Canonical M4 artifact handoff:
VERIFIED

Shared modeling contract:
LOCKED

Core model families:
3 / 3 COMPLETE

Official baseline runs:
6 / 6 VALID

W_SHORT/W_LONG controlled coverage:
COMPLETE

Imbalance strategy:
NONE

Probability evidence:
6 / 6 PERSISTED

Prediction evidence:
6 / 6 PERSISTED

Canonical metric evidence:
COMPLETE

Config locks:
COMPLETE

Pair manifests:
COMPLETE

Hidden tuning:
NO EVIDENCE

Resource workaround:
NOT REQUIRED FOR OFFICIAL RUNS

Training-window winner:
OPEN

Model-family winner:
OPEN

Final threshold:
OPEN

Systematic tuning:
DEFERRED TO M7

FINAL TEST:
PROTECTED

M5 Gate:
12 / 12 PASS

Blocking issue:
NONE

M5:
PASS

Handoff:
READY FOR M6 — EVALUATION + ERROR ANALYSIS
```

---

# 19. Kết luận M5.6

Milestone 5 đã hoàn thành đúng vai trò baseline modeling.

Project hiện có:

- canonical modeling artifacts đã audit;
- một modeling/metric contract thống nhất;
- ba official baseline model configurations;
- sáu controlled W_SHORT/W_LONG baseline runs;
- sáu prediction artifacts;
- sáu probability/risk-score artifacts;
- canonical fraud metric evidence;
- runtime/computational evidence;
- config locks;
- pair manifests;
- FINAL TEST isolation evidence.

M5.6 khóa:

`BASELINE MODELING EVIDENCE EXISTS`

M5.6 không khóa:

`FINAL MODEL / FINAL WINDOW / FINAL THRESHOLD`

M5 Gate:

`PASS`

Next:

`M6 — Evaluation + Error Analysis`

Status:

`READY FOR M6`
