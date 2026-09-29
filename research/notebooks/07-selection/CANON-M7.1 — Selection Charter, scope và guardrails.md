# CANON-M7.1 — Selection Charter, scope và guardrails

## 0. Trạng thái tài liệu

Milestone:

`M7 — MODEL SELECTION / ROBUSTNESS / TUNING`

Substep:

`M7.1 — Selection Charter, scope và guardrails`

Loại công việc:

`CANONICAL PROTOCOL LOCK`

Câu hỏi trung tâm:

> Trước khi chạy bất kỳ experiment selection/tuning nào của M7, project phải khóa những luật nào để kết quả sau này có thể so sánh công bằng, tái hiện được và không làm rò rỉ FINAL TEST?

M7.1 không train model.

M7.1 không tạo runtime selection score.

M7.1 không chọn:

- training-window winner;
- model-family winner;
- hyperparameter winner;
- imbalance-strategy winner;
- numerical threshold;
- final model.

Mục tiêu duy nhất của M7.1:

`KHÓA LUẬT SELECTION TRƯỚC KHI NHÌN M7 RUNTIME RESULTS`.

Final state của tài liệu này:

```text
M7 Selection Charter:
LOCKED

Temporal CV Contract:
LOCKED

Metric / Aggregation Policy:
LOCKED

Tie-break Policy:
LOCKED

Randomness Policy:
LOCKED

Resource Policy:
LOCKED

Imbalance Experiment Order:
LOCKED

Tuning Scope Policy:
LOCKED

Threshold Order:
LOCKED

Artifact / Registry Schema:
LOCKED

Decision Log Schema:
LOCKED

FINAL TEST:
PROTECTED

Runtime Selection Result:
NOT YET PRODUCED

M7.1:
PASS
```

---

# 1. Source hierarchy

M7.1 kế thừa trực tiếp:

- `CANON-Kế hoạch Milestone 7`;
- `CANON-M6.7 — Tổng hợp Evaluation Registry, Error Findings, Decision Log và M6 Gate`;
- `CANON-M3.7 — Policy cho class imbalance, temporal validation/CV và các experiment/tuning sau`;
- `CANON-M3.6 — Thiết kế protocol so sánh training window`;
- `CANON-M3.4 — Thiết kế evaluation metric strategy`;
- `CANON-M3.8 — Tổng hợp Experiment Specification v1.0, Decision Log, Open Questions và M3 Gate`;
- `CANON-M4.8 — Tổng hợp Feature Specification v1.0, Preprocessing Specification v1.0, Behavioral Feature Contract, Decision Log và M4 Gate`;
- M5 baseline modeling CANON;
- M6 reviewed runtime evidence.

Source precedence:

`PROJECT-SPECIFIC CANON > generic ML guidance`.

Nếu một generic technique xung đột với temporal/evaluation contract đã khóa của project:

`CANON PROJECT WINS`.

---

# 2. Starting state từ M6

M6 handoff đã khóa:

```text
M6:
PASS

External VALIDATION:
2019-01-01 <= Timestamp < 2019-06-01

Validation rows:
712,458

Validation fraud:
1,052

Feature representation:
M4.7-baseline-v1

Feature count:
47

Baseline imbalance strategy:
NONE

Official baseline families:
Logistic Regression
Decision Tree
Random Forest

Training-window winner:
OPEN

Model-family winner:
OPEN

Final model:
OPEN

Final threshold:
OPEN

FINAL TEST:
PROTECTED
```

Training-window evidence:

```text
LR:
W_SHORT provisional F1 leader

DT:
W_SHORT provisional F1 leader

RF:
W_SHORT provisional F1 leader

Cross-pair direction:
CONSISTENT PROVISIONAL W_SHORT
```

Điều này là:

`M6 COMPARATIVE EVIDENCE`

không phải:

`FINAL W_SHORT SELECTION`.

Cross-model evidence:

```text
W_SHORT highest F1:
Random Forest

W_SHORT highest Recall:
Decision Tree

W_SHORT highest Precision:
Logistic Regression

W_LONG highest F1 / Recall:
Decision Tree

W_LONG highest Precision:
Random Forest
```

M6 conclusion:

`MODEL TRADE-OFF EXISTS`.

M7 phải dùng robustness/selection protocol thay vì chỉ sort một validation score.

---

# 3. M7 role

M7 là milestone đầu tiên được phép thực hiện:

- training-window robustness;
- model-family/config selection;
- temporal CV;
- controlled imbalance experiments;
- moderate hyperparameter tuning;
- external VALIDATION confirmation;
- numerical threshold selection;
- final development-configuration freeze.

M7 không phải protected final evaluation.

M7 kết thúc khi development configuration đã freeze.

FINAL TEST vẫn chưa tham gia bất kỳ lựa chọn nào.

---

# 4. Ba tầng dữ liệu và quyền sử dụng

M7 khóa ba tầng evidence.

```text
TẦNG 1
TRAIN-internal temporal CV

Vai trò:
robustness
stability
hyperparameter tuning
imbalance comparison
model/config comparison


TẦNG 2
EXTERNAL VALIDATION
2019-01 → 2019-05

Vai trò:
development confirmation
controlled comparison
final upstream candidate confirmation
numerical threshold selection


TẦNG 3
FINAL TEST
2019-06 → 2019-10

Vai trò:
protected final evaluation only
```

Không được trộn vai trò.

## M7.1-D01 — Partition rights

Decision:

`THREE-TIER ROLE SEPARATION = LOCKED`

Status:

`LOCKED`

---

# 5. FINAL TEST guardrail

Trong toàn bộ M7 development, FINAL TEST không được dùng để:

- chọn W_SHORT/W_LONG;
- chọn LR/DT/RF;
- shortlist model;
- chọn hyperparameters;
- chọn class_weight;
- chọn sampling ratio;
- chọn oversampling/undersampling strategy;
- quyết định có dùng SMOTE hay không;
- chọn random seed;
- chọn probability calibration;
- chọn numerical threshold;
- mở rộng search space;
- giải quyết tie-break;
- xác nhận một candidate sau khi validation result không như mong đợi.

M7 notebook/runtime artifact không được load FINAL TEST target/prediction artifacts trừ khi một future milestone riêng đã chính thức authorize protected evaluation.

## M7.1-D02 — FINAL TEST

Decision:

`FINAL TEST ACCESS DURING M7 DEVELOPMENT = PROHIBITED`

Status:

`INHERITED — LOCKED`

---

# 6. Candidate families ban đầu

Initial model families:

```text
Logistic Regression
Decision Tree
Random Forest
```

Initial training-window candidates:

```text
W_SHORT
W_LONG
```

Initial imbalance reference:

`NONE`

Authorized imbalance candidates:

```text
CLASS_WEIGHT

RANDOM_OVERSAMPLING

RANDOM_UNDERSAMPLING

SMOTE — CONDITIONAL
```

Initial family list không có nghĩa tất cả combinations đều phải chạy.

M7 dùng staged narrowing, không dùng full Cartesian search.

## M7.1-D03 — Initial candidate universe

Decision:

```text
Training windows:
W_SHORT / W_LONG

Model families:
LR / DT / RF

Reference imbalance:
NONE
```

Status:

`LOCKED AS STARTING CANDIDATES`

Final winners:

`OPEN`

---

# 7. Primary temporal CV contract

Primary CV:

`FORWARD / EXPANDING TEMPORAL VALIDATION`

Không shuffle.

Không dùng:

```text
Random KFold + shuffle

Random StratifiedKFold + shuffle
```

cho primary model selection.

Reason:

project objective là future transaction screening theo:

`PAST → FUTURE`.

## Fold 1

```text
Fold-training:
training-window start
<= Timestamp
< 2018-04-01

Fold-validation:
2018-04-01
<= Timestamp
< 2018-07-01
```

## Fold 2

```text
Fold-training:
training-window start
<= Timestamp
< 2018-07-01

Fold-validation:
2018-07-01
<= Timestamp
< 2018-10-01
```

## Fold 3

```text
Fold-training:
training-window start
<= Timestamp
< 2018-10-01

Fold-validation:
2018-10-01
<= Timestamp
< 2019-01-01
```

Training-window start:

```text
W_LONG:
2015-01-01

W_SHORT:
2018-01-01
```

Q1/2018 là initial W_SHORT training block, không phải validation fold.

## M7.1-D04 — Primary CV

Decision:

`FORWARD / EXPANDING TEMPORAL VALIDATION`

Status:

`INHERITED — LOCKED`

## M7.1-D05 — Fold template

Decision:

`Q2 / Q3 / Q4 2018`

Status:

`INHERITED — LOCKED`

---

# 8. Fold integrity gate

Mỗi fold trước khi đọc score phải assert:

```text
train rows > 0

validation rows > 0

train fraud > 0

validation fraud > 0

max(train Timestamp)
<
min(validation Timestamp)

train / validation row overlap:
NONE

future row in fold-training:
NO
```

Nếu một fold không có positive support:

không được sửa bằng cách shuffle.

Phải:

- dừng experiment;
- ghi finding;
- thiết kế temporal boundary/fold change có lý do;
- cập nhật Decision Log trước rerun.

## M7.1-D06 — Fold integrity

Decision:

`INTEGRITY BEFORE METRIC`

Status:

`LOCKED`

---

# 9. Fold-safe preprocessing contract

M4 đã khóa semantic feature/preprocessing contract.

M7 không được đổi feature semantics tùy candidate chỉ để tăng score.

Learned preprocessing state trong CV phải fit lại từ:

`FOLD-TRAIN ONLY`.

Bao gồm mọi learned state như:

- scaler statistics;
- categorical vocabulary;
- encoder category mapping;
- imputation state nếu có learned state;
- future learned preprocessing component nếu được authorize.

Fold-validation:

`TRANSFORM ONLY`.

Không được lấy một transformed matrix đã được fit bởi state có nhìn thấy row nằm sau fold-training cutoff rồi gọi đó là temporal CV.

M7.2 chịu trách nhiệm audit implementation path cụ thể.

## M7.1-D07 — Fold preprocessing

Decision:

`LEARNED PREPROCESSING STATE = FOLD-TRAIN ONLY`

Status:

`INHERITED — LOCKED`

---

# 10. Causal feature-history contract

Behavioral features tiếp tục tuân thủ:

`Timestamp(history) < Timestamp(current)`.

Same-timestamp peer:

`NOT HISTORY`.

Historical warm-up không được reset một cách tùy tiện chỉ vì classifier-training window bắt đầu muộn.

Raw IDs:

- có thể dùng làm history keys theo contract;
- không được trở thành classifier features.

Target history:

`PROHIBITED`.

Future history:

`PROHIBITED`.

## M7.1-D08 — Causal feature contract

Decision:

`STRICT CAUSAL HISTORY = LOCKED`

Status:

`INHERITED — LOCKED`

---

# 11. Sampler boundary

Nếu M7 thử resampling:

sampler chỉ được fit/apply trên:

`FOLD-TRAIN`.

Fold-validation:

`NO RESAMPLING`.

External VALIDATION:

`NO RESAMPLING`.

FINAL TEST:

`NO RESAMPLING`.

Synthetic sample không được xuất hiện trong validation/test population.

## M7.1-D09 — Sampler placement

Decision:

```text
Sampler:
FOLD-TRAIN ONLY

Validation resampling:
NO
```

Status:

`INHERITED — LOCKED`

---

# 12. Metric contract

Positive class:

`fraud = 1`.

Primary metric:

`F1_fraud`.

Mandatory secondary:

```text
Recall_fraud
Precision_fraud
```

Mandatory raw evidence:

```text
TP
FP
FN
TN
```

Mandatory operational diagnostic:

```text
predicted_positive_count
predicted_positive_rate
```

Reference only:

`Accuracy`.

Probability/risk score:

`REQUIRED WHEN MODEL SUPPORTS IT`.

Advanced ranking metrics:

`OPTIONAL / NOT CORE SELECTION METRIC`.

Nếu một advanced metric sau này được thêm:

- phải có question riêng;
- phải ghi Decision Log;
- phải xác định vai trò trước runtime result;
- không được thay thế F1_fraud âm thầm.

## M7.1-D10 — Metric contract

Decision:

```text
Primary:
F1_fraud

Mandatory secondary:
Recall_fraud
Precision_fraud

Mandatory diagnostics:
TP / FP / FN / TN
alert count / rate

Accuracy:
REFERENCE ONLY
```

Status:

`INHERITED — LOCKED`

---

# 13. Threshold-policy comparability

F1 / Recall / Precision phụ thuộc threshold.

Do đó primary candidate comparison trước M7.8 phải giữ:

`THRESHOLD POLICY COMPATIBLE / FIXED`.

Không được:

```text
Candidate A:
default decision rule

Candidate B:
validation-optimized threshold
```

rồi dùng F1 để kết luận B tốt hơn.

Threshold optimization là decision layer riêng ở M7.8 sau upstream freeze.

## M7.1-D11 — Threshold policy during upstream comparison

Decision:

`NO CANDIDATE-SPECIFIC THRESHOLD OPTIMIZATION`

Status:

`LOCKED`

---

# 14. Fold-wise reporting contract

Mỗi temporal-CV candidate phải lưu từng fold:

```text
fold_id

fold training boundary
fold validation boundary

train rows
validation rows

train fraud
validation fraud

F1_fraud
Recall_fraud
Precision_fraud

TP
FP
FN
TN

predicted_positive_count
predicted_positive_rate

fit_seconds
prediction_seconds

warning_count / warnings
integrity status
```

Không được chỉ báo aggregate score.

Temporal instability là evidence chính thức.

Ví dụ:

```text
Fold 1:
good

Fold 2:
good

Fold 3:
collapse
```

không được tóm tắt thành:

`mean CV tốt`.

---

# 15. Aggregation policy — khóa ở M7.1

M7.1 khóa aggregate diagnostics sau.

## Primary aggregate

`MEAN_F1_ACROSS_3_TEMPORAL_FOLDS`

là aggregate center chính cho temporal-CV comparison.

## Mandatory stability diagnostic

`STD_F1_ACROSS_3_TEMPORAL_FOLDS`

phải luôn được báo cùng mean F1.

## Mandatory secondary aggregates

```text
mean Recall_fraud
mean Precision_fraud
```

## Raw-count reading

TP / FP / FN / TN phải được giữ theo từng fold.

Không thay raw fold counts bằng average count rồi mất temporal context.

## Pooled out-of-fold metric

Có thể ghép prediction từ ba temporal validation folds để tạo:

```text
pooled F1
pooled Recall
pooled Precision
pooled Confusion Matrix
```

Nhưng pooled metrics là:

`SUPPLEMENTARY DIAGNOSTIC`

không phải primary aggregate để override fold-wise instability.

## M7.1-D12 — Aggregation

Decision:

```text
Primary CV aggregate:
mean F1

Mandatory stability:
std F1

Mandatory secondary aggregate:
mean Recall
mean Precision

Fold-wise metrics:
MANDATORY

Pooled OOF:
OPTIONAL SUPPLEMENTARY DIAGNOSTIC
```

Status:

`LOCKED`

---

# 16. Selection reading order

Sau khi integrity gate PASS, mọi primary candidate comparison đọc theo thứ tự:

```text
STEP 1
mean F1
→ provisional aggregate leader

STEP 2
fold-wise F1 direction
+ std F1
→ temporal stability

STEP 3
Recall + Precision
→ trade-off

STEP 4
TP / FP / FN / TN
→ transaction-level effect

STEP 5
predicted-positive count/rate
→ screening workload

STEP 6
runtime / warnings
→ computational feasibility

STEP 7
external VALIDATION
chỉ ở substep được authorize
→ development confirmation

STEP 8
Decision Log
→ selection / shortlist / inconclusive
```

Không metric nào ngoài contract được chèn vào giữa workflow sau khi nhìn result để cứu một candidate.

---

# 17. Tie-break policy — global M7 rule

M7 không dùng arbitrary epsilon như:

```text
|Δmean_F1| < 0.01
→ tie
```

trừ khi một future Decision Log có căn cứ mới và được khóa trước experiment liên quan.

M7.1 không định nghĩa một “magic number” để gọi score gần nhau.

## Rule T1 — Higher mean F1

Candidate có mean F1 cao hơn:

`PROVISIONAL AGGREGATE LEADER`.

Không tự động là final winner.

## Rule T2 — Fold stability

Đọc:

- F1 từng fold;
- std F1;
- fold collapse;
- direction consistency.

Nếu candidate A mean F1 cao hơn nhưng advantage đến từ một fold rất mạnh trong khi một fold collapse:

không được bỏ qua instability.

## Rule T3 — Recall / Precision

Nếu F1 leadership đi kèm trade-off:

đọc Recall / Precision.

Không tối ưu Recall một mình.

Không tối ưu Precision một mình.

## Rule T4 — Raw error / alert burden

Đọc:

- FN reduction;
- FP increase;
- TP;
- alert volume.

Không gọi intervention/model tốt hơn chỉ vì Recall tăng nếu false-alert burden tăng mạnh và F1/stability không hỗ trợ.

## Rule T5 — Computational evidence

Runtime / memory / warnings là supporting evidence.

Không được dùng computational cost để override một predictive preference rõ chỉ vì candidate nhanh hơn.

## Rule T6 — Inconclusive is allowed

Nếu evidence không tạo một preference có thể bảo vệ:

`DECISION = INCONCLUSIVE / KEEP MULTIPLE CANDIDATES`.

Không ép winner.

## Rule T7 — Training-window-specific parsimony fallback

Riêng W_SHORT vs W_LONG:

nếu sau robustness evidence vẫn không có stable predictive preference và hai candidate vẫn được xem là predictively equivalent theo documented review:

`W_SHORT = PARSIMONY FALLBACK`.

Điều này chỉ áp dụng sau predictive/robustness review.

Không được dùng parsimony trước.

## Rule T8 — Model/config comparison

Không có automatic parsimony fallback đã được khóa cho LR/DT/RF.

Nếu model/config evidence vẫn mixed:

- giữ shortlist;
- handoff external VALIDATION confirmation;
- không invent one-number tie-break.

## M7.1-D13 — Tie-break policy

Decision:

`F1 → stability → Recall/Precision → raw errors/alerts → compute → inconclusive if needed`

Training-window special fallback:

`W_SHORT only after unresolved predictive robustness`.

Status:

`LOCKED`

---

# 18. Candidate narrowing policy

M7 không chạy:

```text
2 windows
× 3 model families
× all imbalance strategies
× large hyperparameter grid
× many thresholds
```

như một Cartesian search.

M7 dùng staged narrowing:

```text
training-window robustness
        ↓
model-family robustness / shortlist
        ↓
imbalance experiment
        ↓
moderate tuning
        ↓
external VALIDATION confirmation
        ↓
freeze upstream candidate
        ↓
threshold selection
```

Mỗi narrowing decision phải:

- dựa trên reviewed evidence;
- ghi Decision Log;
- không dùng FINAL TEST;
- không thay luật sau khi thấy result bất lợi.

## M7.1-D14 — Narrowing policy

Decision:

`STAGED NARROWING`

Status:

`LOCKED`

---

# 19. Training-window selection policy

M7.3 là substep chính thức cho:

`W_SHORT vs W_LONG robustness`.

Comparison phải giữ cùng:

- feature semantics;
- fold-safe preprocessing protocol;
- model/config;
- imbalance strategy;
- threshold policy;
- temporal folds;
- metric code;
- random-state policy.

Variable:

`classifier-training window`.

Possible M7.3 state:

```text
ROBUST W_SHORT PREFERENCE

ROBUST W_LONG PREFERENCE

MIXED / INCONCLUSIVE
```

M7.3 có thể khóa downstream training-window preference nếu evidence đủ rõ.

Nếu mixed:

- không ép winner;
- giữ ambiguity;
- hoặc dùng tie-break protocol đã khóa.

## M7.1-D15 — Training-window selection

Decision:

`M7.3 OWNS ROBUSTNESS DECISION`

Current winner:

`OPEN`

Status:

`LOCKED SCOPE`

---

# 20. Model-family shortlist policy

M7.4 chịu trách nhiệm:

`LR vs DT vs RF robustness / shortlist`.

Model-family comparison phải giữ:

```text
same training-window scope
same feature/preprocessing protocol
same imbalance policy
same folds
same threshold policy
same metric implementation
```

Variable:

`model family / model config`.

Không được:

`LR-SHORT vs RF-LONG`

để kết luận family effect.

Shortlist phải đọc:

- mean F1;
- fold-wise F1;
- std F1;
- Recall;
- Precision;
- FP/FN;
- alert burden;
- computational cost;
- M6 error findings.

M7.4 có thể loại candidate khỏi downstream tuning nếu reviewed evidence cho thấy candidate không còn competitive và Decision Log ghi rõ lý do.

## M7.1-D16 — Model shortlist scope

Decision:

`M7.4 OWNS MODEL-FAMILY ROBUSTNESS / SHORTLIST`

Final family:

`OPEN`

Status:

`LOCKED SCOPE`

---

# 21. Class-imbalance experiment order

No-intervention baseline:

`REQUIRED`.

Reference:

`NONE`.

Sequence:

```text
1. NONE

2. CLASS_WEIGHT
   nếu model hỗ trợ

3. RANDOM_OVERSAMPLING
   hoặc RANDOM_UNDERSAMPLING
   nếu evidence vẫn justify intervention

4. SMOTE
   chỉ nếu conditional representation/leakage gate PASS
```

Không chạy mọi technique cùng lúc.

Primary comparison phải giữ:

- same model/config;
- same training window;
- same folds;
- same preprocessing;
- same threshold policy;
- same metric implementation;
- same random-state policy.

Variable:

`imbalance strategy`.

Outcome có thể là:

`KEEP NONE`.

## M7.1-D17 — Imbalance order

Decision:

`NONE → CLASS_WEIGHT → RANDOM OVER/UNDER if justified → SMOTE conditional`

Status:

`INHERITED — LOCKED`

---

# 22. SMOTE conditional gate

SMOTE không phải default.

Trước khi dùng phải chứng minh:

```text
feature representation:
SUITABLE

sampler:
FOLD-TRAIN ONLY

synthetic validation rows:
NONE

synthetic FINAL TEST rows:
NONE

preprocessing / sampling pipeline:
LEAKAGE-SAFE
```

Nếu không chứng minh được:

`DO NOT USE SMOTE`.

M7 PASS không yêu cầu chạy SMOTE.

## M7.1-D18 — SMOTE

Decision:

`CONDITIONAL / NOT REQUIRED`

Status:

`INHERITED — LOCKED`

---

# 23. Hyperparameter tuning scope

M7.6 chỉ tune shortlisted candidate(s).

Search space:

`SMALL / JUSTIFIED / DECLARED BEFORE RUN`.

Temporal CV:

`REQUIRED` cho multi-candidate hyperparameter tuning.

Không:

- exhaustive search lớn;
- broad random search không có budget;
- adaptive grid expansion sau khi xem result chỉ để săn score;
- tuning random seed;
- tuning trên FINAL TEST;
- thay imbalance strategy và hyperparameters tùy từng candidate mà không có experiment design rõ.

Exact parameter grids:

`NOT LOCKED IN M7.1`.

Chúng phải được:

- đề xuất;
- justify;
- ghi CANON/notebook config;
- freeze trước M7.6 runtime.

Tuning outcome có thể là:

`KEEP BASELINE CONFIG`.

## M7.1-D19 — Tuning scope

Decision:

`MODERATE / PREDECLARED / TEMPORAL-CV-BASED`

Status:

`LOCKED`

Exact search spaces:

`OPEN UNTIL PRE-RUNTIME M7.6 DECLARATION`

---

# 24. Randomness policy

Primary random state:

`42`

khi operation có randomness.

Random seed là:

`REPRODUCIBILITY CONTROL`.

Không phải tuning dimension.

Nếu stochastic stability là question:

- seed set nhỏ phải khai báo trước;
- cùng seed set cho mọi compared candidate;
- report distribution/variation;
- không chọn seed đẹp nhất.

## M7.1-D20 — Random state

Decision:

`RANDOM_STATE = 42`

Status:

`INHERITED — LOCKED`

Multi-seed:

`CONDITIONAL ROBUSTNESS ONLY`

---

# 25. Resource policy

M6 evidence:

```text
W_LONG cost > W_SHORT cost

RF cost > DT cost > LR cost
```

M7 cho phép resource-aware narrowing.

Nhưng comparison fairness phải được giữ.

Không được:

- silently subsample một candidate;
- giảm fold count chỉ cho candidate đắt;
- đổi feature set riêng cho candidate đắt;
- dùng W_SHORT chỉ vì RF-W_LONG chậm mà không ghi decision;
- bỏ failed candidate khỏi registry;
- đổi n_jobs/resource policy theo cách làm comparison mất ý nghĩa mà không ghi metadata.

Nếu compute blocker:

```text
STOP

record computational finding

design controlled workaround

apply fairly

rerun
```

Runtime là:

`SUPPORTING EVIDENCE`

không phải primary predictive metric.

## M7.1-D21 — Resource policy

Decision:

`NO SILENT COMPUTATIONAL SHORTCUT`

Status:

`LOCKED`

---

# 26. External VALIDATION reuse guardrail

External VALIDATION được dùng cho development.

Nhưng phải tránh biến nó thành hidden training/tuning set qua quá nhiều vòng.

Policy:

- temporal CV dùng chủ yếu cho tuning/robustness;
- external VALIDATION dùng controlled confirmation;
- search space không được liên tục mở rộng theo external-validation result;
- unexpected result có thể tạo hypothesis mới;
- hypothesis mới phải được ghi rồi thiết kế experiment mới rõ ràng;
- không chỉnh ad hoc rồi gọi đó là cùng experiment.

Threshold selection chỉ diễn ra sau upstream freeze.

## M7.1-D22 — External VALIDATION reuse

Decision:

`CONTROLLED DEVELOPMENT USE`

Status:

`LOCKED`

---

# 27. Upstream freeze order

Canonical order:

```text
feature / preprocessing contract
        ↓
training window
        ↓
model family / configuration
        ↓
imbalance strategy
        ↓
tuning / robustness
        ↓
freeze upstream development candidate
        ↓
external VALIDATION probability
        ↓
numerical threshold selection
        ↓
freeze threshold
        ↓
protected FINAL TEST
```

Không đảo thứ tự.

M7.7 phải freeze upstream candidate trước M7.8.

## M7.1-D23 — Freeze order

Decision:

`UPSTREAM BEFORE THRESHOLD`

Status:

`INHERITED — LOCKED`

---

# 28. Threshold selection order

Threshold là decision layer riêng.

M7.8 chỉ bắt đầu khi đã freeze:

```text
feature/preprocessing
training window
model family
hyperparameters
imbalance strategy
random-state policy
probability interface
```

Threshold source:

`EXTERNAL VALIDATION`.

FINAL TEST:

`PROHIBITED`.

Threshold comparison phải báo:

```text
threshold candidate
F1
Recall
Precision
TP
FP
FN
TN
predicted-positive count/rate
```

Exact threshold candidate set:

`NOT LOCKED IN M7.1`.

Nó phải được predeclared trước M7.8 runtime.

Default threshold có thể vẫn là final choice nếu evidence không justify thay đổi.

## M7.1-D24 — Threshold order

Decision:

`THRESHOLD AFTER UPSTREAM FREEZE`

Status:

`LOCKED`

Final numeric threshold:

`OPEN`

---

# 29. Probability calibration scope

M6 cho thấy Decision Tree score behavior extreme/discrete.

M7.1 không tự động authorize calibration như core requirement.

Calibration:

`OPTIONAL / CONDITIONAL`.

Nếu nghiên cứu:

- question riêng;
- fold-safe TRAIN-only fitting;
- không trộn với threshold comparison;
- không dùng FINAL TEST;
- phải có metric/evidence phù hợp để đánh giá calibration.

M7 có thể PASS mà không chạy calibration.

## M7.1-D25 — Calibration

Decision:

`OPTIONAL / CONDITIONAL`

Status:

`LOCKED SCOPE`

---

# 30. Error-analysis evidence role

M6 error findings được dùng như:

`SUPPORTING SELECTION EVIDENCE`.

Không phải direct tuning target.

M7 phải tiếp tục đọc:

```text
FN
FP
shared misses
model-specific catches
alert burden
risk-score depth
```

Đặc biệt M6 đã thấy:

- error complementarity;
- is_new_merchant association;
- location_state FP concentration;
- DT score extremity;
- nhiều LR/RF FN nằm xa 0.5.

Không được tune model để “khớp subgroup” nếu experiment question chưa được khóa.

Nếu aggregate F1 tốt hơn nhưng error behavior xấu đi rõ:

`Decision Log MUST RECORD TRADE-OFF`.

## M7.1-D26 — Error evidence role

Decision:

`SUPPORTING / NOT PRIMARY TUNING TARGET`

Status:

`LOCKED`

---

# 31. Candidate identity schema

Mỗi M7 candidate phải có immutable identity trước runtime result.

Minimum:

```text
candidate_id

milestone_substep

dataset / artifact identity

feature_version
preprocessing_version

training_window_id

model_family
model_config_id
hyperparameters

imbalance_strategy
class_weight if applicable
sampling_method if applicable
sampling_ratio if applicable

random_state

cv_spec_id
fold_template_id

threshold_policy

probability_interface

metric_contract_version
```

Candidate ID không được tái sử dụng cho config khác.

Nếu config thay đổi:

`NEW CANDIDATE ID`.

## M7.1-D27 — Candidate identity

Decision:

`IMMUTABLE CANDIDATE IDENTITY`

Status:

`LOCKED`

---

# 32. Fold-result schema

Minimum per candidate/fold:

```text
candidate_id
fold_id

train_start
train_end_exclusive

validation_start
validation_end_exclusive

train_rows
validation_rows

train_fraud_rows
validation_fraud_rows

feature_version
preprocessing_version

model_family
model_config_id
hyperparameters

imbalance_strategy
random_state
threshold_policy

prediction_artifact
risk_score_artifact

F1_fraud
Recall_fraud
Precision_fraud

TP
FP
FN
TN

predicted_positive_count
predicted_positive_rate

Accuracy_reference

fit_seconds
prediction_seconds

warning_count
warnings

integrity_status
```

Nếu fold invalid:

metric không được dùng cho selection.

---

# 33. Candidate-aggregate schema

Minimum per candidate:

```text
candidate_id

valid_fold_count
expected_fold_count

fold_ids

mean_F1
std_F1

mean_Recall
mean_Precision

foldwise_metrics

pooled_OOF_metrics
if produced

total_fit_seconds
total_prediction_seconds

warnings_summary

stability_findings
tradeoff_findings

selection_status
decision_reason

next_action
```

Allowed `selection_status`:

```text
REFERENCE

PROVISIONAL_LEADER

SHORTLISTED

NOT_SHORTLISTED

INCONCLUSIVE

SELECTED_UPSTREAM

REJECTED_INVALID

REJECTED_COMPUTATIONAL
```

Final labels chỉ được gán ở substep có authority tương ứng.

---

# 34. Experiment registry schema

M7 consolidated experiment registry tối thiểu chứa:

```text
registry_version

selection_charter_version

source_M6_registry_identity

candidate_registry

fold_result_registry

aggregate_result_registry

artifact_fingerprints

failed_candidate_registry

decision_log

open_questions

final_test_accessed
```

`final_test_accessed` phải là:

`False`

trong toàn bộ M7 development registry.

---

# 35. Artifact persistence contract

Mỗi runtime substep M7 phải persist:

1. experiment/candidate specification;
2. fold-level result records;
3. aggregate records;
4. manifest;
5. predictions/risk scores nếu cần cho independent verification;
6. hash/fingerprint;
7. Decision state.

Không được chỉ lưu:

`best_score.json`.

Failed/invalid runs đã thực hiện cũng phải để lại audit trail.

Artifact naming phải deterministic theo:

- substep;
- candidate ID;
- fold ID;
- artifact role.

Exact path được khóa trong từng substep trước runtime.

## M7.1-D28 — Artifact policy

Decision:

`PERSIST ENOUGH FOR INDEPENDENT REVIEW`

Status:

`LOCKED`

---

# 36. Decision Log schema

Mỗi Decision Log entry phải có tối thiểu:

```text
decision_id

substep

question

evidence_scope

candidate_set

decision

status

reason

tradeoffs

integrity_preconditions

artifacts_consulted

final_test_accessed

next_action
```

Recommended `status` vocabulary:

```text
OPEN

LOCKED

PROVISIONAL

INCONCLUSIVE

DEFERRED

REQUIRES_RERUN

BLOCKED

SELECTED

REJECTED
```

Decision Log không được chỉ ghi:

`model X tốt hơn`.

Phải ghi:

- dựa trên evidence nào;
- scope nào;
- trade-off gì;
- condition nào;
- có final-test access hay không.

## M7.1-D29 — Decision Log schema

Decision:

`STRUCTURED DECISION LOG REQUIRED`

Status:

`LOCKED`

---

# 37. Integrity-before-score principle

Mọi M7 runtime experiment phải pass integrity trước khi diễn giải metric.

Required checks:

```text
correct artifact:
PASS

candidate identity:
PASS

temporal order:
PASS

partition overlap:
NONE

positive support:
PASS

preprocessing leakage:
NO

resampling leakage:
NO

future-label leakage:
NO

same-timestamp leakage:
NO

raw identifier guardrail:
PASS

threshold policy consistency:
PASS

metric positive-class mapping:
CORRECT

FINAL TEST access:
NO
```

Nếu critical check fail:

`STOP`.

Không:

- sort metric;
- select candidate;
- write performance conclusion.

## M7.1-D30 — Integrity-before-score

Decision:

`MANDATORY`

Status:

`LOCKED`

---

# 38. Runtime review protocol

M7 tiếp tục Type-C collaboration protocol.

Cho từng notebook runtime:

```text
1. Read CANON
2. Freeze experiment question/config
3. Create notebook
4. Restart Kernel → Run All
5. Preserve outputs/errors/warnings
6. AI reviews execution/integrity/leakage/comparability
7. Missing check → add check → rerun
8. Only then write Findings / Decision Log / Gate
```

Không invent runtime output.

Không PASS notebook chưa chạy nếu substep phụ thuộc runtime.

M7.1 là document-only protocol step, nên không cần model runtime để PASS.

---

# 39. Substep authority matrix

```text
M7.1
Authority:
lock selection protocol

Cannot:
select runtime winner


M7.2
Authority:
verify CV infrastructure

Cannot:
select model/window


M7.3
Authority:
training-window robustness decision

Cannot:
final model selection
threshold optimization


M7.4
Authority:
model-family robustness / shortlist

Cannot:
final threshold selection


M7.5
Authority:
imbalance-strategy experiment / decision

Cannot:
threshold optimization in primary imbalance comparison


M7.6
Authority:
moderate hyperparameter tuning

Cannot:
use FINAL TEST


M7.7
Authority:
select and freeze upstream development candidate

Threshold:
still OPEN


M7.8
Authority:
select numerical threshold on external VALIDATION

Upstream config:
must already be FROZEN


M7.9
Authority:
consolidate final development registry / M7 Gate

FINAL TEST:
still UNTOUCHED
```

---

# 40. Prohibited actions

M7 cấm:

- random shuffled primary CV;
- fold-validation leakage vào preprocessing fit;
- fold-validation resampling;
- SMOTE validation;
- future row trong fold-training;
- candidate-specific hidden metric code;
- threshold tuning trong model/imbalance primary comparison;
- seed hunting;
- adaptive grid expansion chỉ vì score chưa đẹp;
- silent candidate deletion;
- silent row subsampling;
- mixed-window model-family comparison;
- using FINAL TEST for any development decision;
- changing tie-break after seeing results;
- changing primary metric after seeing results;
- treating Accuracy as primary;
- claiming calibrated confidence without calibration evidence;
- production generalization from synthetic error patterns.

---

# 41. M7.1 Open Questions sau charter

M7.1 khóa policy nhưng không invent future runtime choices.

## O01 — Fold-safe implementation path

Status:

`OPEN — M7.2`

Question:

M4 artifacts / source representation nào được dùng để refit preprocessing per fold mà vẫn tái tạo đúng M4.7 semantics?

---

## O02 — Training-window winner

Status:

`OPEN — M7.3`

Current evidence:

`PROVISIONAL W_SHORT DIRECTION`

---

## O03 — Model shortlist

Status:

`OPEN — M7.4`

---

## O04 — Final imbalance strategy

Status:

`OPEN — M7.5`

Could remain:

`NONE`.

---

## O05 — Exact M7.6 hyperparameter search space

Status:

`OPEN UNTIL PRE-RUNTIME DECLARATION`

---

## O06 — Need for probability calibration

Status:

`OPTIONAL / CONDITIONAL`

---

## O07 — Upstream development candidate

Status:

`OPEN — M7.7`

---

## O08 — Exact threshold candidate set

Status:

`OPEN UNTIL PRE-RUNTIME M7.8 DECLARATION`

---

## O09 — Final numerical threshold

Status:

`OPEN — M7.8`

---

## O10 — FINAL TEST performance

Status:

`NOT A M7 DEVELOPMENT INPUT`

---

# 42. M7.1 Gate

## G01 — M6 handoff state rõ?

Required:

```text
M6 PASS
selection questions OPEN
FINAL TEST protected
```

Result:

`PASS`

---

## G02 — Partition roles khóa?

Required:

TRAIN-internal temporal CV / external VALIDATION / FINAL TEST không trộn vai trò.

Result:

`PASS`

---

## G03 — Temporal CV khóa?

Required:

`FORWARD / EXPANDING`

Fold template:

`Q2 / Q3 / Q4 2018`

Result:

`PASS`

---

## G04 — Fold integrity checks khóa?

Required:

order / overlap / positive support.

Result:

`PASS`

---

## G05 — Fold-safe preprocessing khóa?

Required:

`learned state = fold-TRAIN only`.

Result:

`PASS`

---

## G06 — Sampler boundary khóa?

Required:

`fold-TRAIN only`

validation resampling:

`NO`.

Result:

`PASS`

---

## G07 — Metric contract khóa?

Required:

```text
F1 primary
Recall/Precision secondary
Confusion Matrix mandatory
alert diagnostics mandatory
Accuracy reference only
```

Result:

`PASS`

---

## G08 — Aggregation policy khóa?

Required:

```text
mean F1
std F1
mean Recall
mean Precision
fold-wise evidence mandatory
```

Result:

`PASS`

---

## G09 — Tie-break khóa?

Required:

- no arbitrary epsilon;
- F1 provisional leader;
- stability/trade-off/raw error review;
- inconclusive allowed;
- W_SHORT parsimony only as final training-window fallback after unresolved robustness.

Result:

`PASS`

---

## G10 — Randomness/resource policy khóa?

Required:

```text
random_state = 42
no seed hunting
no silent computational shortcuts
```

Result:

`PASS`

---

## G11 — Imbalance/tuning scope khóa?

Required:

```text
NONE reference
CLASS_WEIGHT authorized
random over/under conditional on need
SMOTE conditional
small predeclared tuning
```

Result:

`PASS`

---

## G12 — Threshold order khóa?

Required:

`upstream freeze → external VALIDATION threshold selection`.

Result:

`PASS`

---

## G13 — Registry/artifact schema khóa?

Required:

candidate identity + fold results + aggregate + manifest + fingerprints + failed-run trail.

Result:

`PASS`

---

## G14 — Decision Log schema khóa?

Required:

structured evidence / scope / trade-off / status / next-action fields.

Result:

`PASS`

---

## G15 — FINAL TEST isolation khóa?

Required:

`NO DEVELOPMENT ACCESS`.

Result:

`PASS`

---

# 43. Overall M7.1 Gate

```text
G01_M6_HANDOFF_STATE                  → PASS
G02_PARTITION_ROLE_SEPARATION         → PASS
G03_TEMPORAL_CV_CONTRACT              → PASS
G04_FOLD_INTEGRITY_CONTRACT           → PASS
G05_FOLD_SAFE_PREPROCESSING           → PASS
G06_SAMPLER_BOUNDARY                  → PASS
G07_METRIC_CONTRACT                   → PASS
G08_AGGREGATION_POLICY                → PASS
G09_TIE_BREAK_POLICY                  → PASS
G10_RANDOMNESS_RESOURCE_POLICY        → PASS
G11_IMBALANCE_TUNING_SCOPE            → PASS
G12_THRESHOLD_ORDER                   → PASS
G13_REGISTRY_ARTIFACT_SCHEMA          → PASS
G14_DECISION_LOG_SCHEMA               → PASS
G15_FINAL_TEST_ISOLATION              → PASS
```

Overall:

`15 / 15 PASS`

Blocking issue:

`NONE`

M7.1:

`PASS`

---

# 44. M7.1 Decision Log tổng hợp

```text
M7.1-D01
Three-tier partition rights
LOCKED

M7.1-D02
FINAL TEST development access prohibited
LOCKED

M7.1-D03
Initial candidate universe = W_SHORT/W_LONG + LR/DT/RF
LOCKED

M7.1-D04
Primary CV = forward/expanding temporal validation
LOCKED

M7.1-D05
CV folds = Q2/Q3/Q4 2018
LOCKED

M7.1-D06
Integrity before metric
LOCKED

M7.1-D07
Learned preprocessing fit on fold-TRAIN only
LOCKED

M7.1-D08
Strict causal feature history
LOCKED

M7.1-D09
Sampler fold-TRAIN only
LOCKED

M7.1-D10
F1 primary + Recall/Precision + Confusion Matrix
LOCKED

M7.1-D11
No candidate-specific threshold optimization upstream
LOCKED

M7.1-D12
Aggregation = mean F1 + std F1 + mandatory fold-wise evidence
LOCKED

M7.1-D13
Tie-break = F1 → stability → trade-off → raw error/alerts → compute → inconclusive
LOCKED

M7.1-D14
Staged candidate narrowing
LOCKED

M7.1-D15
M7.3 owns training-window robustness
LOCKED

M7.1-D16
M7.4 owns model-family robustness/shortlist
LOCKED

M7.1-D17
Imbalance order
LOCKED

M7.1-D18
SMOTE conditional
LOCKED

M7.1-D19
Moderate predeclared tuning
LOCKED

M7.1-D20
Random state = 42
LOCKED

M7.1-D21
No silent computational shortcut
LOCKED

M7.1-D22
External VALIDATION controlled development use
LOCKED

M7.1-D23
Upstream freeze before threshold
LOCKED

M7.1-D24
Threshold selection on external VALIDATION
LOCKED

M7.1-D25
Calibration optional / conditional
LOCKED

M7.1-D26
M6 error evidence = supporting evidence
LOCKED

M7.1-D27
Immutable candidate identity
LOCKED

M7.1-D28
Artifact persistence sufficient for independent review
LOCKED

M7.1-D29
Structured Decision Log
LOCKED

M7.1-D30
Integrity-before-score
LOCKED
```

---

# 45. Handoff sang M7.2

M7.2 phải chứng minh implementation thực tế của temporal-CV contract.

Primary question:

> Làm thế nào tái sử dụng canonical M4/M5 project assets để tạo fold-safe training/validation matrices mà không dùng preprocessing state học từ tương lai?

M7.2 phải audit ít nhất:

- available source artifacts;
- row lineage;
- Timestamp mapping;
- fold membership;
- causal behavioral features;
- learned preprocessing refit path;
- categorical vocabulary fit path;
- numerical scaler fit path;
- sparse output schema;
- feature-name consistency;
- sampler insertion point;
- model runner contract;
- metric runner;
- persistence;
- FINAL TEST isolation.

M7.2 không được chạy selection experiment trước khi infrastructure gate PASS.

Expected M7.2 end state:

```text
Temporal CV Infrastructure:
VERIFIED

Fold-safe Preprocessing:
VERIFIED

Shared M7 Runner:
LOCKED

Candidate Result Schema:
LOCKED

Selection Winner:
NOT YET PRODUCED

FINAL TEST:
PROTECTED

READY FOR M7.3
```

---

# 46. Final M7.1 state

```text
Milestone:
M7 — MODEL SELECTION / ROBUSTNESS / TUNING

Substep:
M7.1 — Selection Charter, scope và guardrails

Selection Charter:
LOCKED

Partition Roles:
LOCKED

Primary CV:
FORWARD / EXPANDING TEMPORAL VALIDATION

CV Folds:
Q2 / Q3 / Q4 2018

Fold-safe Preprocessing:
REQUIRED

Sampler:
FOLD-TRAIN ONLY

Primary Metric:
F1_fraud

Secondary Metrics:
Recall_fraud
Precision_fraud

Mandatory Diagnostics:
TP / FP / FN / TN
predicted-positive count/rate

Aggregation:
mean F1
std F1
mean Recall
mean Precision
+ mandatory fold-wise evidence

Tie-break:
NO ARBITRARY EPSILON
F1 → stability → trade-off → raw errors/alerts → compute
INCONCLUSIVE ALLOWED

Training-window Special Fallback:
W_SHORT parsimony
ONLY AFTER unresolved predictive robustness

Random State:
42

Imbalance Reference:
NONE

Imbalance Order:
CLASS_WEIGHT
→ random over/under if justified
→ SMOTE conditional

Tuning:
SMALL / JUSTIFIED / PREDECLARED

Threshold:
AFTER UPSTREAM FREEZE
EXTERNAL VALIDATION ONLY

Calibration:
OPTIONAL / CONDITIONAL

Training-window Winner:
OPEN

Model-family Winner:
OPEN

Final Model:
OPEN

Final Imbalance Strategy:
OPEN

Final Threshold:
OPEN

FINAL TEST:
PROTECTED

Runtime Selection Result:
NOT YET PRODUCED

M7.1 Gate:
15 / 15 PASS

Blocking Issue:
NONE

M7.1:
PASS

Next:
M7.2 — Artifact audit + fold-safe temporal-CV infrastructure
