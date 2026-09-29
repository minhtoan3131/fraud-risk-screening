# CANON-M7.9 — Final Selection Registry, Decision Log và M7 Gate

Milestone:

`M7 — MODEL SELECTION / ROBUSTNESS / TUNING`

Substep:

`M7.9 — Final Selection Registry, Decision Log và M7 Gate`

Artifact type:

`CANONICAL CONSOLIDATION / FINAL DEVELOPMENT SELECTION REGISTRY`

---

# 1. Mục tiêu và authority của M7.9

M7.9 không mở thêm experiment.

M7.9 chỉ:

- tổng hợp evidence đã được review từ M7.2 → M7.8;
- khóa Final Selection Registry của development stage;
- hợp nhất Decision Log;
- ghi các limitation/finding còn mở;
- audit artifact identity / fingerprint;
- chạy M7 Gate ở mức tổng hợp;
- xác nhận FINAL TEST vẫn chưa được dùng trong development.

Authority của M7.9:

`CONSOLIDATE FINAL DEVELOPMENT REGISTRY / M7 GATE`

M7.9 không có authority để:

- mở model family mới;
- đổi training window;
- đổi feature/preprocessing;
- retune hyperparameter;
- đổi imbalance strategy;
- mở threshold candidate mới;
- calibration tuning;
- đọc hoặc đánh giá FINAL TEST.

---

# 2. Source hierarchy được dùng cho M7.9

Nguồn canonical/planning:

```text
CANON-Kế hoạch Milestone 7
CANON-M7.1 — Selection Charter, scope và guardrails
```

Nguồn runtime/reviewed evidence:

```text
M7.2
07_02_temporal_cv_infrastructure_audit_reviewed.ipynb

M7.3
07_03_training_window_robustness_reviewed.ipynb

M7.4
07_04_model_family_robustness_and_shortlist_reviewed.ipynb

M7.5
07_05_controlled_class_imbalance_experiments_reviewed.ipynb

M7.6
07_06_moderate_hyperparameter_tuning_reviewed.ipynb

M7.7
07_07_candidate_selection_external_validation_confirmation_reviewed.ipynb

M7.8
07_08_numerical_threshold_selection_reviewed.ipynb
```

Các nguồn trên đã hoàn thành runtime review tương ứng trước khi M7.9 được lập.

M7.9 không invent thêm runtime result.

---

# 3. M7.9 Completion Questions

M7 completion definition yêu cầu project trả lời chắc chắn các câu hỏi sau.

## Q01 — Temporal-CV infrastructure có leakage-safe không?

Answer:

`YES — VERIFIED`

Evidence:

- forward / expanding temporal CV;
- Q2 / Q3 / Q4 2018;
- train trước validation;
- zero row overlap;
- positive support ở train và validation;
- fold-local preprocessing fit trên fold-TRAIN only;
- M4.7 full-window preprocessed matrices không bị dùng trực tiếp cho fold-CV;
- sampler contract khóa fold-TRAIN only.

Source:

`M7.2 — PASS`

---

## Q02 — Training-window decision dựa trên robustness evidence nào?

Answer:

`W_SHORT — ROBUST / LOCKED`

Evidence:

```text
LR:
W_SHORT mean F1 = 0.495102
W_LONG  mean F1 = 0.011683
W_SHORT leads = 3 / 3 folds

DT:
W_SHORT mean F1 = 0.487392
W_LONG  mean F1 = 0.195261
W_SHORT leads = 3 / 3 folds

RF:
W_SHORT mean F1 = 0.517612
W_LONG  mean F1 = 0.105481
W_SHORT leads = 3 / 3 folds
```

Cross-family:

`9 / 9 model-fold comparisons favor W_SHORT`

Decision:

`W_SHORT — FROZEN`

Source:

`M7.3 — PASS`

---

## Q03 — Model family/config nào được chọn và vì sao?

Answer:

```text
Model Family:
Random Forest

Model Config:
RF-REF-100-GINI-SQRT-UNPRUNED-CW
```

M7.7 external VALIDATION:

```text
RF:
F1        = 0.409111
Recall    = 0.443916
Precision = 0.379366
TP        = 467
FP        = 764
FN        = 585
TN        = 710,642
Alerts    = 1,231
```

Trong ba final upstream candidates, RF có:

- external VALIDATION F1 cao nhất;
- external VALIDATION Recall cao nhất;
- FN thấp nhất;
- tốt hơn DT về F1 / Recall / Precision / FN / FP / alerts;
- fraud-side coverage mạnh hơn LR.

Known trade-off:

- Precision thấp hơn LR;
- FP / alerts cao hơn LR;
- compute cost cao nhất.

Decision:

`RF-REF-100-GINI-SQRT-UNPRUNED-CW — FROZEN`

Source:

`M7.7 — PASS`

---

## Q04 — Class-imbalance intervention nào được chọn?

Answer:

`CLASS_WEIGHT_BALANCED`

M7.5 candidate-specific result:

```text
LR:
NONE

DT:
CLASS_WEIGHT_BALANCED

RF:
CLASS_WEIGHT_BALANCED
```

RF class-weight temporal CV effect:

```text
NONE:
mean F1        = 0.517612
std F1         = 0.029168
mean Recall    = 0.414125
mean Precision = 0.694881
FN             = 1,135
FP             = 351
Alerts         = 1,150

CLASS_WEIGHT_BALANCED:
mean F1        = 0.621144
std F1         = 0.020690
mean Recall    = 0.684406
mean Precision = 0.577543
FN             = 619
FP             = 979
Alerts         = 2,294
```

Delta:

```text
Δ mean F1:
+0.103531

Δ Recall:
+0.270281

Δ Precision:
-0.117338

Δ FN:
-516

Δ FP:
+628

Δ Alerts:
+1,144

F1 fold leads:
3 / 3
```

Decision:

`CLASS_WEIGHT_BALANCED — FROZEN`

Random over/under sampling:

`NOT REQUIRED`

SMOTE:

`NOT REQUIRED`

Source:

`M7.5 — PASS`

---

## Q05 — Moderate tuning đã được thực hiện như thế nào?

Answer:

`YES — SMALL / JUSTIFIED / PREDECLARED / TEMPORAL-CV-BASED`

Search space:

```text
LR:
C = 0.1
C = 10
reference C = 1

DT:
max_depth = 20
min_samples_leaf = 5
reference unpruned

RF:
max_depth = 20
min_samples_leaf = 2
reference unpruned
```

Reviewed family-specific configuration result:

```text
LR:
SELECT C = 10

DT:
SELECT min_samples_leaf = 5

RF:
KEEP REFERENCE CONFIG
```

Final selected RF therefore giữ:

`RF-REF-100-GINI-SQRT-UNPRUNED-CW`

Source:

`M7.6 — PASS`

---

## Q06 — Temporal-CV và external VALIDATION có đồng thuận hoàn toàn không?

Answer:

`NO`

Finding:

`TEMPORAL GENERALIZATION FINDING`

CV → external VALIDATION F1 change của final M7.6 candidates:

```text
LR:
0.524328 → 0.341368
Δ = -0.182960

DT:
0.605570 → 0.345007
Δ = -0.260562

RF:
0.621144 → 0.409111
Δ = -0.212033
```

Recall và Precision cũng giảm ở cả ba family.

Interpretation:

- temporal CV vẫn hữu ích cho robustness/tuning;
- external VALIDATION cho thấy 2019 performance thấp hơn 2018 CV;
- finding này không được xóa sau model selection;
- không mở lại tuning ad hoc chỉ vì external VALIDATION thấp hơn.

Status:

`CARRY FORWARD`

Source:

`M7.7 — PASS`

---

## Q07 — Upstream model configuration đã freeze chưa?

Answer:

`YES`

Frozen before threshold selection:

```text
Feature / Preprocessing:
M4 canonical contract

Training Window:
W_SHORT

Model Family:
Random Forest

Model Config:
RF-REF-100-GINI-SQRT-UNPRUNED-CW

Imbalance Strategy:
CLASS_WEIGHT_BALANCED

Random State:
42

Probability Interface:
predict_proba / positive class = 1
```

Source:

`M7.7 — PASS`

---

## Q08 — Threshold được chọn theo policy nào?

Answer:

`EXTERNAL VALIDATION / PREDECLARED NUMERICAL THRESHOLD COMPARISON`

Candidate set:

```text
0.20
0.30
0.40
0.50
0.60
```

Comparator:

`risk_score > threshold`

Default compatibility:

```text
threshold 0.50
prediction mismatch vs persisted M7.7 default y_pred:
0
```

Threshold results:

```text
0.20:
F1        = 0.362667
Recall    = 0.517110
Precision = 0.279261
FP        = 1,404
FN        = 508
Alerts    = 1,948

0.30:
F1        = 0.383750
Recall    = 0.502852
Precision = 0.310264
FP        = 1,176
FN        = 523
Alerts    = 1,705

0.40:
F1        = 0.393856
Recall    = 0.475285
Precision = 0.336247
FP        = 987
FN        = 552
Alerts    = 1,487

0.50:
F1        = 0.409111
Recall    = 0.443916
Precision = 0.379366
FP        = 764
FN        = 585
Alerts    = 1,231

0.60:
F1        = 0.403751
Recall    = 0.388783
Precision = 0.419918
FP        = 565
FN        = 643
Alerts    = 974
```

Decision:

`KEEP DEFAULT THRESHOLD`

Final numerical threshold:

`0.50 — FROZEN`

Source:

`M7.8 — PASS`

---

## Q09 — Selected artifacts/config có identity rõ không?

Answer:

`YES`

Artifact registry và SHA256 fingerprints được ghi ở Section 10.

---

## Q10 — FINAL TEST có được bảo vệ trong toàn bộ M7 không?

Answer:

`YES`

Observed development state xuyên M7:

```text
M7.2:
FINAL TEST protected

M7.3:
FINAL TEST protected

M7.4:
FINAL TEST protected

M7.5:
final_test_accessed = False

M7.6:
final_test_accessed = False

M7.7:
final_test_accessed = False

M7.8:
final_test_accessed = False
```

M7.9:

`NO FINAL TEST ACCESS`

Final-test performance:

`UNKNOWN / NOT A M7 DEVELOPMENT INPUT`

---

# 4. Final Selection Registry

## 4.1. Canonical development configuration

```text
Registry ID:
M7-FINAL-DEVELOPMENT-SELECTION-v1

Feature / Preprocessing:
M4 canonical contract
47-column CSR float32 representation
strict-causal behavioral features
TRAIN-only learned preprocessing state

Training Window:
W_SHORT

Classifier TRAIN interval:
2018-01-01
≤ Timestamp
< 2019-01-01

Classifier TRAIN rows:
1,721,615

Classifier TRAIN fraud:
2,491

Model Family:
Random Forest

Model Config ID:
RF-REF-100-GINI-SQRT-UNPRUNED-CW

Criterion:
gini

n_estimators:
100

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
balanced

ccp_alpha:
0.0

max_samples:
None

random_state:
42

n_jobs:
-1

Probability Interface:
predict_proba

Positive Class:
1

Risk-score interpretation:
model probability-like risk score
NOT claimed calibrated confidence

Numerical Threshold:
0.50

Threshold Comparator:
risk_score > 0.50

Primary Metric:
F1_fraud

Mandatory supporting metrics:
Recall_fraud
Precision_fraud
TP / FP / FN / TN
predicted-positive count/rate
```

Status:

`FROZEN`

---

## 4.2. External VALIDATION confirmation của final configuration

Population:

```text
2019-01-01
≤ Timestamp
< 2019-06-01

Rows:
712,458

Fraud:
1,052
```

Metrics at final threshold `0.50`:

```text
F1_fraud:
0.4091108191

Recall_fraud:
0.443916

Precision_fraud:
0.379366

TP:
467

FP:
764

FN:
585

TN:
710,642

Predicted-positive count:
1,231

Predicted-positive rate:
0.0017278211
```

Fit time observed in M7.7 external confirmation:

`46.312 s`

No warning:

`YES`

---

# 5. Evidence Chain toàn M7

## M7.2 — Temporal-CV infrastructure

Final:

```text
M7.2 — PASS

Temporal CV Infrastructure:
VERIFIED

Fold-safe Preprocessing:
VERIFIED

Shared M7 Runner:
LOCKED

Candidate Result Schema:
LOCKED

FINAL TEST:
PROTECTED
```

W_SHORT folds:

```text
Q2:
train 423,905 / fraud 557
validation 428,953 / fraud 590

Q3:
train 852,858 / fraud 1,147
validation 435,178 / fraud 634

Q4:
train 1,288,036 / fraud 1,781
validation 433,579 / fraud 710
```

---

## M7.3 — Training-window robustness

Final:

```text
M7.3 — PASS

ROBUST W_SHORT PREFERENCE

W_SHORT:
LOCKED FOR DOWNSTREAM M7
```

Cross-family:

`9 / 9 comparisons favor W_SHORT`

---

## M7.4 — Model-family robustness / shortlist

Baseline W_SHORT temporal-CV aggregate:

```text
LR:
mean F1        = 0.495102
std F1         = 0.038315
mean Recall    = 0.385526
mean Precision = 0.698296

DT:
mean F1        = 0.487392
std F1         = 0.013631
mean Recall    = 0.471101
mean Precision = 0.515111

RF:
mean F1        = 0.517612
std F1         = 0.029168
mean Recall    = 0.414125
mean Precision = 0.694881
```

Reviewed shortlist:

`LR / DT / RF — RETAIN ALL 3`

Reason:

- RF: strongest F1 role;
- DT: strongest Recall / lowest-FN role;
- LR: strongest Precision / lowest-FP / lowest-alert / fastest-fit role;
- no family met evidence standard for elimination at M7.4.

Final:

`M7.4 — PASS`

---

## M7.5 — Controlled class-imbalance experiments

Reviewed dispositions:

```text
LR:
NONE

DT:
CLASS_WEIGHT_BALANCED

RF:
CLASS_WEIGHT_BALANCED
```

Random oversampling:

`NOT REQUIRED`

Random undersampling:

`NOT REQUIRED`

SMOTE:

`NOT REQUIRED`

Phase B:

`NOT OPENED`

Final:

`M7.5 — PASS`

---

## M7.6 — Moderate hyperparameter tuning

Reviewed configuration map:

```text
LR:
LR-T02-LBFGS-L2-C10

DT:
DT-T02-GINI-MINLEAF5-CW

RF:
RF-REF-100-GINI-SQRT-UNPRUNED-CW
```

Final:

`M7.6 — PASS`

Model-family winner:

`STILL OPEN`

---

## M7.7 — Candidate selection + external VALIDATION confirmation

External VALIDATION:

```text
LR:
F1        = 0.341368
Recall    = 0.249049
Precision = 0.542443
FN        = 790
FP        = 221
Alerts    = 483

DT:
F1        = 0.345007
Recall    = 0.440114
Precision = 0.283701
FN        = 589
FP        = 1,169
Alerts    = 1,632

RF:
F1        = 0.409111
Recall    = 0.443916
Precision = 0.379366
FN        = 585
FP        = 764
Alerts    = 1,231
```

Selected:

`RF-REF-100-GINI-SQRT-UNPRUNED-CW`

Finding:

`TEMPORAL GENERALIZATION FINDING`

Final:

`M7.7 — PASS`

Upstream pipeline:

`FROZEN`

---

## M7.8 — Numerical threshold selection

Predeclared:

`0.20 / 0.30 / 0.40 / 0.50 / 0.60`

Selected:

`0.50`

Decision:

`KEEP DEFAULT THRESHOLD`

Final:

```text
M7.8 — PASS

Development Selection:
COMPLETE

Final Threshold:
0.50 — FROZEN

FINAL TEST:
PROTECTED
```

---

# 6. Computational Evidence Registry

Observed compute evidence được dùng như supporting criterion, không override clear predictive evidence.

## M6 W_SHORT baseline fit times

```text
LR:
0.822 s

DT:
6.662 s

RF:
40.698 s
```

## M7.7 final-candidate external VALIDATION fit times

```text
LR C=10:
0.776 s

DT min_samples_leaf=5:
7.796 s

RF selected:
46.312 s
```

Finding:

`RF has highest computational cost among final candidates`

Decision impact:

- cost được ghi nhận;
- không tạo resource blocker;
- không override RF predictive evidence.

---

# 7. Error / Operational Evidence Registry

M7.7 selected-candidate overlap:

```text
Actual fraud:
1,052

Caught by at least one:
518

Caught by all three:
244

All-three miss:
534
```

Model-specific-only fraud catches:

```text
LR:
1

DT:
41

RF:
46
```

Model-specific-only false positives:

```text
LR:
12

DT:
568

RF:
159
```

Interpretation:

- RF giữ fraud-side incremental value;
- DT có higher unique false-positive burden;
- LR có strong low-FP role nhưng fraud coverage yếu hơn;
- overlap evidence không được dùng để mở ensemble experiment trong M7.

---

# 8. Open Limitations / Carry-forward Findings

## L01 — Temporal generalization

Status:

`OPEN LIMITATION / CARRY FORWARD`

Finding:

Temporal-CV performance cao hơn external VALIDATION đáng kể ở cả ba final candidates.

Không được mô tả CV metric như trực tiếp đại diện cho 2019 external performance.

---

## L02 — Synthetic dataset

Dataset là synthetic / benchmark-oriented.

Không được suy rộng trực tiếp:

- fraud behavior;
- threshold operating point;
- expected production precision/recall;
- business loss;
- deployment reliability

sang hệ thống tài chính production thật.

---

## L03 — Probability calibration

Selected RF sử dụng:

`predict_proba`

Nhưng project chưa có calibration audit/tuning.

Do đó:

- được gọi `risk score`;
- không được gọi `calibrated confidence`.

Calibration:

`NOT REQUIRED FOR M7 PASS / NOT PERFORMED`

---

## L04 — External VALIDATION reuse

External VALIDATION đã được dùng cho controlled development confirmation và threshold selection.

Guardrail được giữ bằng:

- staged narrowing;
- immutable candidate identities;
- small predeclared tuning/threshold sets;
- no adaptive candidate expansion sau kết quả.

Tuy vậy đây vẫn là development validation set, không phải unbiased final-test estimate.

---

## L05 — Threshold scope

Final threshold `0.50` là selected threshold trong exact predeclared set:

`0.20 / 0.30 / 0.40 / 0.50 / 0.60`

Không claim:

`global mathematical optimum over all real-valued thresholds`.

---

## L06 — Business-cost function

Không có canonical monetary/business loss function được khóa.

Vì vậy threshold selection giữ:

`F1_fraud as primary`

và dùng Recall / Precision / FP / FN / alerts làm supporting evidence.

---

## L07 — FINAL TEST performance

Status:

`UNKNOWN`

Reason:

`PROTECTED / NOT A M7 DEVELOPMENT INPUT`

---

# 9. Artifact Registry

## 9.1. Reviewed notebooks

```text
M7.2 reviewed notebook
07_02_temporal_cv_infrastructure_audit_reviewed.ipynb
SHA256:
d737beee8c5a8c39827fcc927283397a961ecdf2400a3ba187e9bbca5b554293

M7.3 reviewed notebook
07_03_training_window_robustness_reviewed.ipynb
SHA256:
da1b8d14954a6c79825e81071950aea29b113f7c6d0d84989d0e5f473c6198d4

M7.4 reviewed notebook
07_04_model_family_robustness_and_shortlist_reviewed.ipynb
SHA256:
1724a51a0e2fb5cc53d68c16065a7beea37aa63ab12f3d0c761f3c123bce2454

M7.5 reviewed notebook
07_05_controlled_class_imbalance_experiments_reviewed.ipynb
SHA256:
a7d50b9b35a2c8406e893d7f1ee87ff4133d1159d8f6f58ddd41202ea2a39060

M7.6 reviewed notebook
07_06_moderate_hyperparameter_tuning_reviewed.ipynb
SHA256:
c97275ec93382f891c2d2c19ecd7416356caf8e725da99975301869f98274727

M7.7 reviewed notebook
07_07_candidate_selection_external_validation_confirmation_reviewed.ipynb
SHA256:
b46f2aab72661ccb0ba3c07ee3dcc1f58876560c9d9b24eeb197304d194c9dee

M7.8 reviewed notebook
07_08_numerical_threshold_selection_reviewed.ipynb
SHA256:
429fc03d17ba08479814453d62896c6ed55c4a5f95d9c80bcad2be1f1958bce0
```

---

## 9.2. Runtime result / manifest fingerprints surfaced by reviewed outputs

```text
M7.3 result SHA256:
2588836b8af6c2f5061ed5f8f076facbbe287538776a929f09aaf929b029ff6a

M7.3 manifest SHA256:
7b34c1b5a2d20efeb8013a97055a77e6f1bd91c095eda4dfa4b3c72afa013dff


M7.4 result SHA256:
65f353209a55e7d754d6b5551b3d35f52bf91c8df12bb5cb53d5ed739cfacb58

M7.4 manifest SHA256:
3278f88fe29326f01caae0687efddd107d0ec0f362456069a39fd952dfed9762


M7.5 result SHA256:
1fe1771fd5f9250028361310cc59cc4cc5dd6a5b6ec93218c1a58ab3d3c20d10

M7.5 manifest SHA256:
796298d45f3c5c3829db3ad1e139eaf1759dd591ec9dd52b6a797cfea1f9bdda


M7.6 result SHA256:
0450890acd3be9c18290208d08455bec3268d12733bb5bc4b99737c9cda13693

M7.6 manifest SHA256:
88a3dc042d3517cd43df492c1c90e9ec158566d74e6759382e0d971c1b261e67


M7.7 result SHA256:
9f2cc25b8acf162b02b5fcc95e06bcc1978f7a1b1d1cf4347f9b4be2ceb7170c

M7.7 manifest SHA256:
6c57e0c47b2c374f7d11721335d94fffcb82ac79fbeeecfb3b6a6ee4029eea36


M7.8 result SHA256:
7182944c08a230cf383649c66f6947e1c91f7855f33c856eae56a3b748b485e0

M7.8 manifest SHA256:
3c83c5ed65db74582d49a5fb5042f4aef14c9d6f524e370a160e7697f8671a24

M7.8 threshold-prediction artifact SHA256:
bbb4569a6baad5dbb2a3537ef7c264614c8ebeb71815d3bcfa5b4a7cb910dde5
```

M7.2 canonical runtime artifacts:

```text
data/processed/m7_02_temporal_cv_infrastructure_audit/
    m7_02_cv_infrastructure_contract.json
    m7_02_preprocessing_state_audit.json
    m7_02_infrastructure_manifest.json
```

M7.2 reviewed notebook là fingerprinted evidence anchor cho M7.9.

---

# 10. Consolidated Decision Log

## M7.9-D01 — Primary selection metric

Decision:

`F1_fraud`

Supporting:

`Recall / Precision / Confusion Matrix / alerts`

Status:

`INHERITED — LOCKED`

---

## M7.9-D02 — Temporal validation design

Decision:

`FORWARD / EXPANDING Q2/Q3/Q4 2018`

Status:

`VERIFIED — LOCKED`

---

## M7.9-D03 — Fold-safe preprocessing

Decision:

`FOLD-TRAIN ONLY`

Status:

`VERIFIED — LOCKED`

---

## M7.9-D04 — Random-state policy

Decision:

`42`

Seed hunting:

`PROHIBITED`

Status:

`LOCKED`

---

## M7.9-D05 — Training window

Decision:

`W_SHORT`

Reason:

`ROBUST W_SHORT PREFERENCE / 9 of 9 fold-family comparisons`

Status:

`FROZEN`

---

## M7.9-D06 — Model-family shortlist

Decision:

`LR / DT / RF retained through M7.4`

Reason:

Distinct predictive/operational roles and no evidence-based elimination at M7.4.

Status:

`HISTORICAL DECISION — CLOSED`

---

## M7.9-D07 — LR imbalance strategy

Decision:

`NONE`

Status:

`CLOSED`

---

## M7.9-D08 — DT imbalance strategy

Decision:

`CLASS_WEIGHT_BALANCED`

Status:

`CLOSED`

---

## M7.9-D09 — RF imbalance strategy

Decision:

`CLASS_WEIGHT_BALANCED`

Status:

`FROZEN`

---

## M7.9-D10 — Random over/under sampling

Decision:

`NOT REQUIRED`

Reason:

M7.5 Phase A đã tạo actionable candidate-specific dispositions.

Status:

`CLOSED`

---

## M7.9-D11 — SMOTE

Decision:

`NOT REQUIRED`

Status:

`CLOSED`

---

## M7.9-D12 — LR tuned configuration

Decision:

`LR-T02-LBFGS-L2-C10`

Status:

`HISTORICAL FINAL LR CANDIDATE`

---

## M7.9-D13 — DT tuned configuration

Decision:

`DT-T02-GINI-MINLEAF5-CW`

Status:

`HISTORICAL FINAL DT CANDIDATE`

---

## M7.9-D14 — RF tuned configuration

Decision:

`KEEP REFERENCE CONFIG`

Selected RF candidate:

`RF-REF-100-GINI-SQRT-UNPRUNED-CW`

Status:

`FROZEN`

---

## M7.9-D15 — External VALIDATION candidate selection

Decision:

`SELECT RANDOM FOREST`

Selected:

`RF-REF-100-GINI-SQRT-UNPRUNED-CW`

Status:

`FROZEN`

---

## M7.9-D16 — Temporal generalization finding

Decision:

Carry forward:

`TEMPORAL GENERALIZATION FINDING`

Status:

`OPEN LIMITATION`

---

## M7.9-D17 — Probability interface

Decision:

`predict_proba / positive class = 1`

Status:

`FROZEN`

---

## M7.9-D18 — Calibration

Decision:

`NOT PERFORMED / NOT REQUIRED FOR M7 PASS`

Therefore score language:

`risk score`

not:

`calibrated confidence`

Status:

`CLOSED FOR M7`

---

## M7.9-D19 — Threshold candidate set

Decision:

`0.20 / 0.30 / 0.40 / 0.50 / 0.60`

Status:

`HISTORICAL / IMMUTABLE`

---

## M7.9-D20 — Final numerical threshold

Decision:

`0.50`

Disposition:

`KEEP DEFAULT THRESHOLD`

Status:

`FROZEN`

---

## M7.9-D21 — Final development configuration

Decision:

```text
W_SHORT
Random Forest
RF-REF-100-GINI-SQRT-UNPRUNED-CW
CLASS_WEIGHT_BALANCED
random_state = 42
risk_score = predict_proba class 1
threshold = 0.50
```

Status:

`FROZEN`

---

## M7.9-D22 — FINAL TEST

Decision:

`NO DEVELOPMENT ACCESS`

Status:

`UNTOUCHED / PROTECTED`

---

## M7.9-D23 — M7 completion

Decision:

`M7 — PASS`

Reason:

Selection process is controlled, evidence-backed, reproducible and final-test-safe.

Status:

`LOCKED`

---

## M7.9-D24 — Handoff

Decision:

`READY FOR PROTECTED FINAL EVALUATION`

Important:

Protected final evaluation phải là bước riêng sau M7.

Không được retroactively dùng FINAL TEST để sửa M7 decisions.

Status:

`READY`

---

# 11. M7 Gate

## G01 — M6 handoff integrity

Requirement:

`M6 verified evaluation + error evidence available`

Observed:

`PASS`

Result:

`PASS`

---

## G02 — Temporal-CV infrastructure

Requirement:

`Leakage-safe forward/expanding CV infrastructure`

Observed:

`M7.2 — PASS`

Result:

`PASS`

---

## G03 — Fold order / overlap / positive support

Requirement:

`PASS`

Observed:

- temporal order verified;
- zero row overlap;
- positive train and validation fraud support.

Result:

`PASS`

---

## G04 — Fold-safe preprocessing

Requirement:

`FOLD-TRAIN ONLY`

Observed:

`VERIFIED`

Result:

`PASS`

---

## G05 — Sampler leakage isolation

Requirement:

Sampler nếu dùng chỉ được fit/apply trên fold-TRAIN.

Observed:

- sampler compatibility verified in M7.2;
- class_weight selected for final RF;
- validation not resampled.

Result:

`PASS`

---

## G06 — Training-window robustness decision

Required:

`RECORDED`

Observed:

`W_SHORT — ROBUST / FROZEN`

Result:

`PASS`

---

## G07 — Model-family/config selection evidence

Required:

`COMPLETE`

Observed:

- M7.4 shortlist complete;
- M7.6 config selection complete;
- M7.7 final RF selection complete.

Result:

`PASS`

---

## G08 — Imbalance strategy evidence

Required:

`COMPLETE`

Observed:

`CLASS_WEIGHT_BALANCED selected for final RF`

Random sampling / SMOTE not required by reviewed evidence.

Result:

`PASS`

---

## G09 — Hyperparameter tuning evidence

Required:

`COMPLETE OR EXPLICITLY NOT NEEDED`

Observed:

`M7.6 moderate tuning COMPLETE`

RF decision:

`KEEP REFERENCE CONFIG`

Result:

`PASS`

---

## G10 — External VALIDATION confirmation

Required:

`COMPLETE`

Observed:

`3 / 3 final candidates scored on external VALIDATION`

Selected RF:

`F1 = 0.409111`

Result:

`PASS`

---

## G11 — Upstream candidate freeze

Required:

`COMPLETE`

Observed:

```text
W_SHORT
Random Forest
RF-REF-100-GINI-SQRT-UNPRUNED-CW
CLASS_WEIGHT_BALANCED
random_state 42
predict_proba class 1
```

Result:

`PASS`

---

## G12 — Threshold selection

Required:

`COMPLETE`

Observed:

`0.50 — FROZEN`

Result:

`PASS`

---

## G13 — Final development configuration registry

Required:

`COMPLETE`

Observed:

`Section 4 — COMPLETE`

Result:

`PASS`

---

## G14 — FINAL TEST isolation

Required:

`PASS`

Observed:

`FINAL TEST UNTOUCHED / PROTECTED across M7`

Result:

`PASS`

---

## G15 — Decision Log / limitations / handoff

Required:

`COMPLETE`

Observed:

- consolidated Decision Log complete;
- limitations recorded;
- artifact fingerprints recorded;
- protected-final-evaluation handoff recorded.

Result:

`PASS`

---

# 12. Overall M7 Gate

```text
M7 Gate checks:
15 / 15 PASS

Development Selection:
COMPLETE

Upstream Configuration:
FROZEN

Threshold:
0.50 — FROZEN

Final Selection Registry:
COMPLETE

Decision Log:
COMPLETE

Artifact Fingerprints:
RECORDED

Temporal Generalization Finding:
CARRY FORWARD

FINAL TEST:
UNTOUCHED / PROTECTED

Blocking Issue:
NONE
```

Final:

`M7 — PASS`

---

# 13. Final Development Handoff

Canonical frozen development selection:

```text
Feature / Preprocessing:
M4 canonical contract
47-column CSR float32
strict-causal behavioral features
TRAIN-only learned preprocessing

Training Window:
W_SHORT

Model Family:
Random Forest

Model Config:
RF-REF-100-GINI-SQRT-UNPRUNED-CW

Imbalance Strategy:
CLASS_WEIGHT_BALANCED

Random State:
42

Probability Interface:
predict_proba / positive class = 1

Numerical Threshold:
0.50
```

Development validation evidence:

```text
F1:
0.409111

Recall:
0.443916

Precision:
0.379366

TP:
467

FP:
764

FN:
585

TN:
710,642

Alerts:
1,231
```

Critical limitation:

`TEMPORAL GENERALIZATION FINDING`

FINAL TEST:

`UNTOUCHED / PROTECTED`

Handoff:

`READY FOR PROTECTED FINAL EVALUATION`

---

# 14. M7.9 Final State

```text
M7.9 — PASS

M7 — PASS

Final Selection Registry:
LOCKED

Development Configuration:
FROZEN

Training Window:
W_SHORT — FROZEN

Model Family:
RANDOM FOREST — FROZEN

Model Config:
RF-REF-100-GINI-SQRT-UNPRUNED-CW — FROZEN

Imbalance Strategy:
CLASS_WEIGHT_BALANCED — FROZEN

Numerical Threshold:
0.50 — FROZEN

Temporal Generalization Finding:
CARRY FORWARD

FINAL TEST:
UNTOUCHED / PROTECTED

Blocking Issue:
NONE

Handoff:
READY FOR PROTECTED FINAL EVALUATION
```
