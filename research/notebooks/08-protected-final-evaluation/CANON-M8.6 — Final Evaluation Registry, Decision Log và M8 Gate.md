# CANON-M8.6 — Final Evaluation Registry, Decision Log và M8 Gate

## 0. Trạng thái tài liệu

Milestone:

`M8 — PROTECTED FINAL EVALUATION`

Substep:

`M8.6 — Final Evaluation Registry, Decision Log và M8 Gate`

Loại công việc:

`CANONICAL CONSOLIDATION / FINAL EVALUATION GATE / HANDOFF`

Runtime model experiment mới:

`NOT REQUIRED`

M8.6 không:

- fit lại model;
- chạy prediction mới;
- thay preprocessing;
- thay feature;
- thay training window;
- thay imbalance strategy;
- thay hyperparameter;
- thay random seed;
- calibration;
- thử threshold khác;
- dùng FINAL TEST để quay lại tối ưu subject.

M8.6 tổng hợp evidence đã được runtime-review và khóa ở M8.1–M8.5.

Final state của tài liệu này:

`M8 — PASS`

Handoff:

`READY FOR M9 — FINAL MODEL / INFERENCE PACKAGING`

---

# 1. Câu hỏi trung tâm

M8.6 phải trả lời:

> Sau toàn bộ protected final evaluation, project có đủ evidence để khóa official FINAL TEST result, khóa limitation, chứng minh không có retroactive optimization, và handoff đúng frozen evaluated subject sang M9 hay chưa?

Câu trả lời sau consolidation:

`YES`

Điều M8.6 xác nhận là:

`PROTECTED FINAL EVALUATION COMPLETED WITH AUDITABLE EVIDENCE`

Điều M8.6 không xác nhận:

```text
production readiness
real-bank performance
real-world fraud probability
causal fraud mechanism
completely-new User/Card generalization
```

---

# 2. Source hierarchy và evidence chain

M8.6 kế thừa trực tiếp:

1. `CANON-M8.1 — Final Evaluation Charter.md`
2. `CANON-M8.2 — FINAL TEST artifact : lineage : representation audit.md`
3. `CANON-M8.3 — Frozen Final Inference Run.md`
4. `CANON-M8.4 — Final Metric + Confusion Matrix Reconstruction.md`
5. `CANON-M8.5 — Final FP/FN + Temporal Generalization Review.md`

Runtime-reviewed notebook evidence:

```text
M8.2:
08_02_final_test_artifact_lineage_representation_audit_v2.ipynb

M8.3:
08_03_frozen_final_inference_run_reviewed.ipynb

M8.4:
08_04_final_metric_confusion_matrix_reconstruction_reviewed.ipynb

M8.5:
08_05_final_fp_fn_temporal_generalization_review_reviewed.ipynb
```

Source precedence:

`VERIFIED RUNTIME EVIDENCE > REVIEWED NOTEBOOK > LATEST CANON > PLAN > GENERIC ML GUIDANCE`

M8.6 không invent runtime result.

---

# 3. Canonical source fingerprints

Canonical document fingerprints:

```text
CANON-M8.1 SHA256:
3d0bc62131918439d454d9908451be178ef225e3c8589f554ecf15c5fc1c021a

CANON-M8.2 SHA256:
94ea35097e4aab425a4ad6c1b4405ad3e1a4b5072430dade19e7c7a3bf699ec7

CANON-M8.3 SHA256:
238c91e7e3475e468858db5a8ac5d58e731b3bdb644d3d7ae60179ce17976e8d

CANON-M8.4 SHA256:
cd58abfefdcbf4bee93911d9e4c8af526f7bc5d5cc7673f5082c3244156a23d1

CANON-M8.5 SHA256:
3029e43e86953f2e658d69b5de6a472094e3891a8331efdc9433859b52e454ea
```

Reviewed notebook fingerprints:

```text
M8.2 reviewed notebook SHA256:
d4a378c084cd3c6bc029b821503e2cf13c81535f617931f6209abac0231c41a5

M8.3 reviewed notebook SHA256:
8351f6982018e5c946b03a552eb409e4f539c3967cf54c2bc63f415c4d5b7bdc

M8.4 reviewed notebook SHA256:
8882b37791a5b540d4e3fb5002b746acc3092637077fb6cbf103cb2659e989cd

M8.5 reviewed notebook SHA256:
6cd77ef17164b354cd39dd082ab8a05b3e9b70ceba1c950b774374ecee9ed262
```

M8.6 dùng các fingerprint này như provenance anchors của final evidence chain.

---

# 4. Final Evaluation Registry

## 4.1. Project / task identity

Problem:

`Binary supervised classification`

Unit:

`one financial transaction`

Positive class:

`fraud = 1`

Negative class:

`non-fraud = 0`

Prediction point:

`transaction screening time`

Model role:

`transaction risk-screening component`

Output semantics:

`risk-screening signal`

Không phải:

`definitive fraud judgment`

Dataset:

`IBM Synthetic Credit Card Transactions / TabFormer`

Dataset nature:

`FULLY SYNTHETIC`

---

## 4.2. Frozen evaluated subject

Training window:

`W_SHORT`

Training boundary:

```text
2018-01-01 <= Timestamp < 2019-01-01
```

Model family:

`Random Forest`

Config ID:

`RF-REF-100-GINI-SQRT-UNPRUNED-CW`

Imbalance strategy:

`CLASS_WEIGHT_BALANCED`

random_state:

`42`

Exact model parameters:

```text
bootstrap = true
ccp_alpha = 0.0
class_weight = "balanced"
criterion = "gini"
max_depth = null
max_features = "sqrt"
max_samples = null
min_samples_leaf = 1
min_samples_split = 2
n_estimators = 100
n_jobs = -1
random_state = 42
```

Physical estimator resolution:

`DETERMINISTIC RECONSTRUCTION IN M8.2`

Reason:

`No canonical serialized M7.7 estimator was available.`

Acceptance condition:

`Exact M7.7 VALIDATION prediction/risk-score reproduction`

Observed:

```text
validation y_pred mismatch:
0

validation risk_score mismatch:
0

threshold comparator mismatch:
0
```

M8.2 persisted reconstructed estimator SHA256:

`61bdeeba5fd163cce8a5a9efa028a9cdc2f95c0796822a5222c0604c0302115f`

---

## 4.3. Feature / preprocessing identity

Semantic feature contract:

`10 pre-encoding features`

Encoded width:

`47`

Matrix:

`CSR`

dtype:

`float32`

Learned preprocessing source:

`W_SHORT TRAIN ONLY`

Causal-history rule:

`Timestamp(history) < Timestamp(current)`

No target labels in history state.

M8.2 exact VALIDATION representation reproduction:

```text
matrix difference nnz:
0
```

M8.2 FINAL TEST representation SHA256:

`68b72dc7607b1c77edcd10b2e42cac0949ea2d2acd2de843925c0d9854a7c2bf`

M8.2 preprocessing-state SHA256:

`c1dc5486acdcd43f5f75fbd0a11bd7ad3a3b00761d50ae96b2b1fbe5b5d9ef98`

---

## 4.4. FINAL TEST identity

Boundary:

```text
2019-06-01 <= Timestamp < 2019-11-01
```

Rows:

`722,955`

Fraud:

`1,035`

Non-fraud:

`721,920`

Observed timestamp support:

```text
2019-06-01 00:02
→
2019-10-31 23:59
```

Lineage:

`VERIFIED`

Development overlap:

`NONE`

Target identity:

`VERIFIED`

FINAL TEST target SHA256:

`5413d0d2934da55678faa40d1fd8abe442bd036d39f62ddd5faea19d67d682e2`

FINAL TEST row-id SHA256:

`46a9bd1c8fd57551be17cf125c1b86a0ad4feef3d76e67e1dd389203cb1ebc36`

FINAL TEST timestamp SHA256:

`3e87911541c2d06a2ac8a781d6834f249eaf761103d911fea5670a91119dd7f2`

---

## 4.5. Risk-score / threshold identity

Risk-score interface:

`predict_proba / positive class = 1`

Allowed language:

```text
risk score
positive-class score
model score
```

Prohibited official language:

```text
calibrated confidence
real-world fraud probability
"x% chắc chắn là fraud"
```

Frozen threshold:

`0.50`

Comparator:

`risk_score > 0.50`

Calibration:

`NOT PERFORMED`

Threshold change after FINAL TEST:

`NONE`

---

# 5. M8 substep registry

```text
M8.1
Role:
Final Evaluation Charter

Status:
PASS

Key result:
protocol / subject / threshold / metric / STOP rules locked
```

```text
M8.2
Role:
FINAL TEST artifact / lineage / representation / model-state audit

Technical gates:
21 / 21 PASS

STOP conditions triggered:
NONE

Decision:
PASS
```

```text
M8.3
Role:
official frozen FINAL TEST inference

Official predict_proba calls on FINAL TEST:
1

Technical gates:
17 / 17 PASS

Decision:
PASS
```

```text
M8.4
Role:
independent final metric + confusion reconstruction

Technical gates:
17 / 17 PASS

Decision:
PASS

Canonical final metric profile:
LOCKED
```

```text
M8.5
Role:
final FP/FN + temporal generalization review

Technical gates:
20 / 20 PASS

Decision:
PASS

Temporal generalization limitation:
CONFIRMED / MATERIAL / DATASET-SPECIFIC
```

---

# 6. Official M8.3 prediction-artifact registry

M8.3 official output root:

`data/processed/m8_03_final_test_inference/`

Official artifacts:

```text
risk_score_final_test.npy
y_pred_final_test.npy
y_final_test.npy
row_id_final_test.npy
timestamp_final_test.npy
m8_03_final_inference_registry.json
m8_03_inference_manifest.json
m8_03_official_inference.lock.json
```

Registry SHA256:

`7a03495f08ff1dd0f5d6488f67d8439500699adabfea28c9724620ffe67436ce`

Manifest SHA256:

`922722f935236ce4a4dc0bc440c83d79d365a2eab8a8ce36a31417154fdfaea7`

M8.3 evidence:

```text
predict_proba FINAL TEST call count:
1

threshold reconstruction mismatch:
0

artifact persistence:
PASS

artifact round-trip:
PASS

final-test metric computed in M8.3:
NO
```

Official prediction state:

`FROZEN / PERSISTED / VERIFIED`

---

# 7. Canonical FINAL TEST metric profile

Source:

`M8.4`

M8.4 result SHA256:

`167ce2ebf1516896b2d393602938382fc4e411b7f8fd566be0403865ab14d038`

M8.4 manifest SHA256:

`08b4d32f3705850eb2f89cada2e16dc2fda7405810c9a8d1279ff5842b7dd9ab`

Official confusion matrix:

```text
TP = 393
FP = 782
FN = 642
TN = 721,138

Total = 722,955
```

Primary metric:

```text
F1_fraud =
0.3556561085972851
```

Secondary metrics:

```text
Recall_fraud =
0.37971014492753624

Precision_fraud =
0.334468085106383
```

Reference metric:

```text
Accuracy_reference =
0.9980303061739666
```

Operational diagnostic:

```text
predicted_positive_count =
1,175

predicted_positive_rate =
0.0016252740488688785
≈ 0.162527%
```

Metric reconstruction:

`MANUAL + SCIKIT-LEARN CROSS-CHECK PASS`

Canonical status:

`LOCKED`

---

# 8. Final error findings

Source:

`M8.5`

M8.5 result SHA256:

`2775a51e3d97205d964ee32775cc782419796f783f2bbc65c81c067bf6cb823d`

M8.5 manifest SHA256:

`665d44b11fa5759e6efb2de78d375414dff688614f22d39bd4a571c99d23237e`

Error-row lineage SHA256:

`26da360a056293448f54384bab0293b727259d4b416439a6f49e4adf4576b5c9`

Error groups:

```text
TP:
393

FP:
782

FN:
642

TN:
721,138
```

## 8.1. False-negative finding

Observed:

```text
FN count:
642

FN score < 0.10:
538 / 642
≈ 83.8%

FN median risk score:
≈ 0.01
```

Interpretation:

`Most missed fraud is scored far below the frozen threshold, not merely clustered immediately below 0.50.`

Status:

`DESCRIPTIVE / VERIFIED`

Not authorized:

`threshold retuning from this finding`

---

## 8.2. False-positive finding

Observed:

```text
FP count:
782

FP median risk score:
≈ 0.72

TP median risk score:
≈ 0.83
```

Interpretation:

`TP and FP show material overlap in high-score regions.`

Status:

`DESCRIPTIVE / VERIFIED`

---

## 8.3. is_new_merchant finding

FINAL TEST:

```text
FN is_new_merchant rate:
≈ 45.64%

TP is_new_merchant rate:
≈ 61.07%

FP is_new_merchant rate:
≈ 37.60%

TN is_new_merchant rate:
≈ 2.69%
```

Historical M6.6 direction:

```text
FN < TP
FP > TN
```

FINAL TEST:

```text
FN < TP:
PERSISTS

FP > TN:
PERSISTS
```

Status:

`DESCRIPTIVE CROSS-PERIOD CONSISTENCY`

---

## 8.4. location_state concentration

Observed dataset-specific concentration:

```text
PHYSICAL_ZIP_UNAVAILABLE:

contains:
1,035 / 1,035 FINAL TEST fraud

contains:
779 / 782 FINAL TEST FP
```

Within non-fraud `PHYSICAL_ZIP_UNAVAILABLE`:

```text
rows:
5,143

FP:
779

FP rate:
≈ 15.15%
```

Status:

`STRONG DATASET-SPECIFIC CONCENTRATION`

Causal interpretation:

`NOT AUTHORIZED`

---

# 9. Temporal generalization registry

Selected RF VALIDATION:

```text
Rows:
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

F1:
0.4091108190976785

Recall:
0.4439163498098859

Precision:
0.3793663688058489

Alerts:
1,231
```

Protected FINAL TEST:

```text
Rows:
722,955

Fraud:
1,035

TP:
393

FP:
782

FN:
642

TN:
721,138

F1:
0.3556561085972851

Recall:
0.37971014492753624

Precision:
0.334468085106383

Alerts:
1,175
```

Delta FINAL − VALIDATION:

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

Observed aggregate direction:

```text
F1_final < F1_validation
Recall_final < Recall_validation
Precision_final < Precision_validation
```

Temporal conclusion:

`TEMPORAL GENERALIZATION LIMITATION — CONFIRMED / MATERIAL / DATASET-SPECIFIC`

Monthly behavior:

`HETEROGENEOUS / NON-MONOTONIC`

Causal drift explanation:

`NOT ESTABLISHED`

M8 evidence does not establish:

```text
concept drift as a causal mechanism
covariate drift as a causal mechanism
business-process regime shift
real-world fraud-regime shift
```

---

# 10. Population / generalization limitation

The protected future evaluation population is predominantly an existing User/Card population with historical context.

Supported claim:

> Model behavior was measured on later transactions from a population largely containing entities with prior history in the synthetic IBM dataset.

Not supported:

`strong completely-new User/Card generalization claim`

Cold-start limitation:

`MUST BE CARRIED FORWARD`

Required carry-forward:

```text
M9:
packaging / inference semantics

M13:
report limitation

M14:
defense limitation
```

---

# 11. No-retuning-after-final-test attestation

Required M8 field:

`no_retuning_after_final_test = true`

Final attestation:

```text
no_retuning_after_final_test = true

No model-family change:
CONFIRMED

No training-window change:
CONFIRMED

No feature change:
CONFIRMED

No preprocessing change:
CONFIRMED

No imbalance-strategy change:
CONFIRMED

No hyperparameter change:
CONFIRMED

No seed search:
CONFIRMED

No calibration tuning:
CONFIRMED

No threshold change:
CONFIRMED

No TRAIN+VALIDATION refit:
CONFIRMED

No candidate expansion:
CONFIRMED

No FINAL TEST driven optimization:
CONFIRMED
```

Retroactive optimization:

`NONE`

Test-isolation breach:

`NONE OBSERVED`

---

# 12. Final interpretation boundary

M8 official evidence supports:

```text
1. identity / lineage / representation integrity of protected FINAL TEST
2. exact frozen-subject inference
3. official FINAL TEST metric profile
4. final FP/FN behavior
5. temporal degradation relative to validation
6. dataset-specific limitations
```

M8 does not support:

```text
1. production-bank performance
2. calibrated real-world fraud probability
3. definitive fraud verdict
4. causal interpretation of error patterns
5. causal explanation of temporal degradation
6. completely-new User/Card generalization
7. financial cost-saving claims
```

Correct project-level wording:

> Frozen Random Forest risk-screening model đạt F1_fraud khoảng 0.3557, Recall khoảng 0.3797 và Precision khoảng 0.3345 trên protected future FINAL TEST của synthetic IBM dataset, với 393 TP, 782 FP và 642 FN. Hiệu năng aggregate thấp hơn external VALIDATION trên cả F1, Recall và Precision, cho thấy temporal generalization limitation đáng kể trong protocol hiện tại.

Không được rút gọn thành:

`model production-ready`

hoặc:

`model dự đoán fraud ngoài thực tế với độ chính xác 99.8%`

---

# 13. M8.6 Decision Log

## M8.6-D01 — M8 milestone role

Decision:

`M8 = PROTECTED FINAL EVALUATION`

Status:

`LOCKED / COMPLETED`

---

## M8.6-D02 — Official evaluated subject

Decision:

```text
W_SHORT
Random Forest
RF-REF-100-GINI-SQRT-UNPRUNED-CW
CLASS_WEIGHT_BALANCED
random_state = 42
```

Status:

`FINAL EVALUATED SUBJECT — LOCKED`

---

## M8.6-D03 — Physical model subject

Decision:

`M8.2 deterministic reconstruction accepted after exact M7.7 validation reproduction`

Status:

`VERIFIED / PERSISTED`

---

## M8.6-D04 — Feature / preprocessing identity

Decision:

`M4.7 canonical 10 → 47 CSR float32 / W_SHORT TRAIN-only learned state`

Status:

`LOCKED`

---

## M8.6-D05 — Final threshold semantics

Decision:

`risk_score > 0.50`

Status:

`LOCKED`

No post-FINAL TEST threshold change.

---

## M8.6-D06 — Official final metric profile

Decision:

```text
F1_fraud = 0.3556561085972851
Recall_fraud = 0.37971014492753624
Precision_fraud = 0.334468085106383
Accuracy_reference = 0.9980303061739666
```

Status:

`LOCKED`

---

## M8.6-D07 — Official confusion matrix

Decision:

```text
TP = 393
FP = 782
FN = 642
TN = 721,138
```

Status:

`LOCKED`

---

## M8.6-D08 — Alert behavior

Decision:

```text
predicted_positive_count = 1,175
predicted_positive_rate ≈ 0.162527%
```

Status:

`LOCKED`

---

## M8.6-D09 — Error-profile interpretation

Decision:

`DESCRIPTIVE ONLY`

Status:

`LOCKED`

No error finding may be used to reopen model/threshold selection.

---

## M8.6-D10 — Temporal generalization

Decision:

`CONFIRMED / MATERIAL / DATASET-SPECIFIC LIMITATION`

Temporal shape:

`HETEROGENEOUS / NON-MONOTONIC`

Status:

`LOCKED`

---

## M8.6-D11 — Causal drift explanation

Decision:

`NOT ESTABLISHED`

Status:

`LOCKED`

---

## M8.6-D12 — Risk-score language

Decision:

`RISK SCORE / POSITIVE-CLASS SCORE`

Not authorized:

`CALIBRATED CONFIDENCE / REAL-WORLD FRAUD PROBABILITY`

Status:

`LOCKED`

---

## M8.6-D13 — Cold-start generalization

Decision:

`LIMITED EVIDENCE`

Strong completely-new User/Card claim:

`NOT AUTHORIZED`

Status:

`CARRY FORWARD`

---

## M8.6-D14 — TRAIN+VALIDATION refit

Decision:

`NOT PERFORMED FOR FINAL EVALUATED SUBJECT`

Status:

`LOCKED`

---

## M8.6-D15 — Retroactive optimization

Decision:

`PROHIBITED / NONE OBSERVED`

Status:

`LOCKED`

---

## M8.6-D16 — M9 subject

Decision:

`M9 default packaging subject = the same M8 evaluated frozen subject`

M9 must not silently replace it with a TRAIN+VALIDATION refit model.

Status:

`LOCKED FOR HANDOFF`

---

# 14. M8 Gate

## G01 — M8.1 protocol complete

Requirement:

`Final Evaluation Charter locked before official final performance access`

Observed:

`M8.1 — PASS`

Result:

`PASS`

---

## G02 — FINAL TEST identity / lineage verified

Requirement:

```text
boundary
row count
fraud support
row lineage
no illegal development overlap
```

Observed:

`M8.2 — PASS`

Result:

`PASS`

---

## G03 — Feature / preprocessing identity verified

Requirement:

```text
W_SHORT TRAIN-only learned state
10 → 47
CSR float32
exact validation representation reproduction
strict-causal behavior
```

Observed:

`VERIFIED`

Result:

`PASS`

---

## G04 — Official selected model state verified

Requirement:

`exact frozen selected RF subject reproducible`

Observed:

```text
deterministic reconstruction
validation y_pred mismatch = 0
validation risk_score mismatch = 0
```

Result:

`PASS`

---

## G05 — Official FINAL TEST inference executed correctly

Requirement:

`one frozen official inference`

Observed:

```text
predict_proba FINAL TEST call count = 1
M8.3 — PASS
```

Result:

`PASS`

---

## G06 — Official prediction artifacts auditable

Requirement:

```text
persisted
fingerprinted
round-tripped
lineage preserved
```

Observed:

`PASS`

Result:

`PASS`

---

## G07 — Final metric profile independently reconstructed

Requirement:

`metrics from persisted y_final_test + y_pred_final_test`

Observed:

```text
manual reconstruction = sklearn cross-check
M8.4 — PASS
```

Result:

`PASS`

---

## G08 — Confusion arithmetic consistent

Requirement:

`TP + FP + FN + TN = 722,955`

Observed:

`PASS`

Result:

`PASS`

---

## G09 — Final metric profile locked

Requirement:

```text
F1
Recall
Precision
Accuracy reference
alerts
```

Observed:

`LOCKED IN M8.4`

Result:

`PASS`

---

## G10 — Error analysis has row-level evidence

Requirement:

`TP/FP/FN/TN row lineage + score/semantic descriptive analysis`

Observed:

`M8.5 — PASS`

Result:

`PASS`

---

## G11 — Temporal generalization reviewed

Requirement:

`VALIDATION → FINAL comparison + monthly FINAL TEST slices`

Observed:

`VERIFIED`

Result:

`PASS`

---

## G12 — Final limitations explicitly recorded

Requirement:

```text
synthetic-data limitation
temporal generalization limitation
risk-score calibration limitation
cold-start limitation
no causal error/drift claim
```

Observed:

`RECORDED`

Result:

`PASS`

---

## G13 — No-retuning after FINAL TEST

Requirement:

`no_retuning_after_final_test = true`

Observed:

`TRUE`

Result:

`PASS`

---

## G14 — No test-isolation breach

Requirement:

`FINAL TEST not used to alter frozen subject`

Observed:

`NONE OBSERVED`

Result:

`PASS`

---

## G15 — Required M8 artifacts persisted

Requirement:

```text
M8.2 audit artifacts
M8.3 prediction artifacts
M8.4 metric artifacts
M8.5 error artifacts
```

Observed:

`PERSISTED / FINGERPRINTED / ROUND-TRIPPED`

Result:

`PASS`

---

## G16 — Official prediction artifacts fingerprinted before Gate

Requirement:

`SHA256/fingerprint + round-trip evidence`

Observed:

`PASS`

Result:

`PASS`

---

## G17 — M8 evidence chain reviewable

Requirement:

`substep-separated source / notebook / artifact provenance`

Observed:

`YES`

Result:

`PASS`

---

## G18 — M9 handoff is unambiguous

Requirement:

```text
evaluated subject identity frozen
risk-score semantics frozen
threshold semantics frozen
limitations frozen
no silent refit
```

Observed:

`READY`

Result:

`PASS`

---

# 15. Overall M8 Gate

Gate summary:

```text
M8.1:
PASS

M8.2:
PASS
21 / 21 technical gates

M8.3:
PASS
17 / 17 technical gates

M8.4:
PASS
17 / 17 technical gates

M8.5:
PASS
20 / 20 technical gates

M8.6 Gate:
18 / 18 PASS

Blocking issue:
NONE

Test-isolation breach:
NONE OBSERVED

no_retuning_after_final_test:
true
```

Final decision:

`M8 — PASS`

Final Evaluation Registry:

`LOCKED`

Official Final Metric Profile:

`LOCKED`

Official Confusion Matrix:

`LOCKED`

Final Error Findings:

`LOCKED / DESCRIPTIVE`

Temporal Generalization Limitation:

`LOCKED / CONFIRMED / MATERIAL / DATASET-SPECIFIC`

---

# 16. M8 completion statement

M8 đã hoàn thành protected final evaluation theo đúng separation:

```text
M8.1
protocol lock

→

M8.2
artifact / lineage / representation / model-state audit

→

M8.3
one-shot official frozen inference

→

M8.4
independent final metric reconstruction

→

M8.5
descriptive final error + temporal review

→

M8.6
final registry / decision log / M8 Gate
```

Không có evidence cho thấy FINAL TEST được dùng để chọn lại:

```text
training window
model family
feature set
preprocessing
imbalance strategy
hyperparameters
seed
calibration
threshold
```

M8 đóng với trạng thái:

`PASS`

---

# 17. Handoff sang M9

M9 role:

`FINAL MODEL / INFERENCE PACKAGING`

M9 default subject:

```text
Training window:
W_SHORT

Model:
Random Forest

Config:
RF-REF-100-GINI-SQRT-UNPRUNED-CW

Imbalance:
CLASS_WEIGHT_BALANCED

random_state:
42

Feature representation:
M4.7 canonical
10 semantic features → 47 encoded features

Risk score:
predict_proba class 1

Threshold:
0.50

Comparator:
risk_score > 0.50
```

Primary physical estimator source:

```text
data/processed/m8_02_final_test_artifact_audit/
    m8_02_selected_rf_estimator.joblib
```

Frozen preprocessing evidence:

```text
data/processed/m8_02_final_test_artifact_audit/
    m8_02_w_short_preprocessing_state.json
```

Reference regression evidence:

```text
M8.3 official FINAL TEST prediction artifacts
M8.4 official metric profile
```

M9 packaging must preserve:

```text
model identity
feature order
preprocessing identity
positive-class identity
risk-score semantics
threshold comparator semantics
no-retuning history
```

M9 must not silently:

```text
TRAIN+VALIDATION refit
change estimator
change threshold
change preprocessing
change feature set
calibrate score
```

Nếu M9 muốn tạo packaging/runtime wrapper mới:

`WRAPPER MAY CHANGE — EVALUATED SUBJECT MAY NOT CHANGE`

---

# 18. Limitations bắt buộc carry sang M9 / M13 / M14

```text
1. Dataset is fully synthetic.

2. FINAL TEST performance is temporal/dataset-specific,
   not a production-bank guarantee.

3. Risk score is not calibrated probability.

4. Recall on FINAL TEST is limited:
   393 TP / 642 FN.

5. False-alert burden exists:
   782 FP among 1,175 alerts.

6. Aggregate FINAL TEST F1 / Recall / Precision
   are lower than external VALIDATION.

7. Temporal behavior is heterogeneous and non-monotonic.

8. Causal drift explanation is not established.

9. Main evaluation population is existing-entity dominant.

10. Strong completely-new User/Card generalization
    is not established.
```

---

# 19. Final M8 handoff block

```text
Milestone:
M8 — PROTECTED FINAL EVALUATION

Status:
PASS

Official evaluated subject:
W_SHORT Random Forest
RF-REF-100-GINI-SQRT-UNPRUNED-CW
CLASS_WEIGHT_BALANCED
random_state = 42

Threshold:
risk_score > 0.50

FINAL TEST:
2019-06-01 <= Timestamp < 2019-11-01
722,955 rows
1,035 fraud

Official metrics:
F1_fraud = 0.3556561085972851
Recall_fraud = 0.37971014492753624
Precision_fraud = 0.334468085106383
Accuracy_reference = 0.9980303061739666

Confusion Matrix:
TP = 393
FP = 782
FN = 642
TN = 721,138

Alerts:
1,175
0.162527%

Temporal generalization:
LIMITATION CONFIRMED
MATERIAL
DATASET-SPECIFIC
HETEROGENEOUS / NON-MONOTONIC

Causal drift explanation:
NOT ESTABLISHED

no_retuning_after_final_test:
true

Test-isolation breach:
NONE OBSERVED

M8 Gate:
18 / 18 PASS

Next:
M9 — FINAL MODEL / INFERENCE PACKAGING
```
