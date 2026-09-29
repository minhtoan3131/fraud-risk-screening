# CANON-Kế hoạch Milestone 7 — Model Selection, Robustness, Tuning, Imbalance và Threshold

## 0. Trạng thái tài liệu

Milestone:

`M7 — MODEL SELECTION / ROBUSTNESS / TUNING`

Loại tài liệu:

`CANONICAL MILESTONE PLAN`

Nguồn kế thừa chính:

- `CANON-M6.7 — Tổng hợp Evaluation Registry, Error Findings, Decision Log và M6 Gate`;
- `CANON-M3.7 — Policy cho class imbalance, temporal validation/CV và các experiment/tuning sau`;
- `CANON-M3.6 — Protocol so sánh training window`;
- `CANON-M3.4 — Evaluation metric strategy`;
- M4.7 baseline-ready feature/preprocessing contract;
- M5 baseline modeling evidence.

M7 bắt đầu từ trạng thái:

```text
M6:
PASS

Training-window winner:
OPEN

Model-family winner:
OPEN

Final model:
OPEN

Final imbalance strategy:
OPEN

Final hyperparameters:
OPEN

Final threshold:
OPEN

FINAL TEST:
PROTECTED
```

Mục tiêu của M7:

> Dùng protocol selection có kiểm soát để chuyển từ baseline evidence sang một development configuration đã freeze, đủ điều kiện handoff cho protected final evaluation.

---

# 1. M7 là gì?

M5 đã trả lời:

> Các baseline model chạy như thế nào?

M6 đã trả lời:

> Các baseline model sai ở đâu, trade-off ra sao và evidence hiện tại nói gì?

M7 phải trả lời:

> Training window, model family/config, imbalance strategy và numerical threshold nào sẽ được chọn làm development configuration cuối?

M7 không phải:

`thử càng nhiều càng tốt rồi chọn validation score cao nhất`.

M7 phải:

- khóa câu hỏi experiment trước khi chạy;
- chỉ thay đúng intervention đang nghiên cứu;
- dùng temporal protocol đúng vai trò;
- giữ external VALIDATION đúng quyền sử dụng;
- không dùng FINAL TEST để lựa chọn;
- ghi đầy đủ stability / trade-off / computational evidence;
- freeze upstream decisions trước threshold;
- freeze threshold trước protected final evaluation.

---

# 2. M7 handoff từ M6

Stable facts:

```text
External VALIDATION:
2019-01-01 <= Timestamp < 2019-06-01

Rows:
712,458

Fraud:
1,052

Feature representation:
M4.7-baseline-v1

Feature count:
47

Baseline imbalance:
NONE

Official baseline model families:
Logistic Regression
Decision Tree
Random Forest

FINAL TEST:
PROTECTED
```

Training-window evidence:

```text
LR provisional F1 direction:
W_SHORT

DT provisional F1 direction:
W_SHORT

RF provisional F1 direction:
W_SHORT

Cross-pair direction:
CONSISTENT PROVISIONAL W_SHORT

Final training-window winner:
OPEN
```

Same-window model evidence:

```text
W_SHORT:
highest F1      → Random Forest
highest Recall  → Decision Tree
highest Precision → Logistic Regression

W_LONG:
highest F1 / Recall → Decision Tree
highest Precision   → Random Forest

Final model:
OPEN
```

Error evidence:

```text
W_SHORT all-three fraud miss:
618

W_LONG all-three fraud miss:
810

Error complementarity:
YES

is_new_merchant cross-run pattern:
OBSERVED

location_state false-alert concentration:
OBSERVED / DATASET-SPECIFIC
```

Probability evidence:

```text
Decision Tree error scores:
EXTREME / DISCRETE

LR/RF false negatives far below 0.5:
COMMON

Final threshold:
OPEN
```

Computational evidence:

```text
LR:
lowest baseline fit cost

DT:
intermediate

RF:
highest

W_LONG:
substantially more expensive than W_SHORT
```

---

# 3. M7 core questions

M7 phải xử lý lần lượt các câu hỏi sau.

## Q01 — Training window

> W_SHORT có đủ robust để được khóa thay cho W_LONG hay không?

M6 evidence:

`W_SHORT = consistent provisional F1 direction`

nhưng chưa phải final selection.

M7 phải dùng robustness evidence nếu cần trước khi khóa.

---

## Q02 — Model family / model configuration

> LR, DT hay RF — với configuration nào — tạo predictive trade-off đủ tốt và đủ ổn định?

Không được chọn family chỉ từ một metric hoặc một holdout result.

---

## Q03 — Class imbalance intervention

> Baseline NONE có đủ hay cần class-weight / resampling?

Authorized candidate family:

```text
NONE
CLASS_WEIGHT
RANDOM_OVERSAMPLING
RANDOM_UNDERSAMPLING
SMOTE — CONDITIONAL
```

Không strategy nào mặc định tốt hơn.

---

## Q04 — Hyperparameter tuning

> Moderate tuning có tạo improvement đủ ổn định để justify complexity tăng thêm không?

Search space phải:

- nhỏ;
- có lý do;
- khai báo trước;
- không mở rộng sau khi nhìn result chỉ để săn score.

---

## Q05 — Numerical threshold

> Sau khi upstream model/config/window/imbalance đã freeze, threshold nào tạo development trade-off phù hợp trên external VALIDATION?

Threshold là:

`DECISION LAYER RIÊNG`

không được thay đồng thời với imbalance/model config trong primary causal comparison.

---

## Q06 — Ready for protected final evaluation?

> Toàn bộ development decisions đã freeze chưa?

Chỉ khi câu trả lời là YES mới được handoff sang protected final evaluation.

---

# 4. Partition rights trong M7

Project giữ ba tầng evidence.

```text
TẦNG 1
TRAIN-internal temporal CV
→ robustness
→ tuning
→ stability evidence

TẦNG 2
External VALIDATION
2019-01 → 2019-05
→ development confirmation
→ controlled comparison
→ numerical threshold selection

TẦNG 3
FINAL TEST
2019-06 → 2019-10
→ protected final evaluation
```

M7 không được trộn vai trò ba tầng.

FINAL TEST không được dùng để:

- chọn training window;
- chọn model family;
- chọn hyperparameters;
- chọn class_weight;
- chọn sampling ratio;
- chọn imbalance strategy;
- chọn random seed;
- chọn probability calibration;
- chọn numerical threshold.

---

# 5. Temporal CV strategy chính thức

Primary CV:

`FORWARD / EXPANDING TEMPORAL VALIDATION`

Không shuffle.

Không dùng random shuffled KFold hoặc shuffled StratifiedKFold cho primary model selection.

Temporal fold template:

```text
FOLD 1

Fold-train:
training-window start
<= Timestamp
< 2018-04-01

Fold-validation:
2018-04-01
<= Timestamp
< 2018-07-01


FOLD 2

Fold-train:
training-window start
<= Timestamp
< 2018-07-01

Fold-validation:
2018-07-01
<= Timestamp
< 2018-10-01


FOLD 3

Fold-train:
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

Q1/2018 không dùng làm validation fold cho W_SHORT vì cần initial training block.

Mỗi fold phải assert:

```text
train rows > 0
validation rows > 0
train fraud > 0
validation fraud > 0

max(train Timestamp)
<
min(validation Timestamp)

fold overlap:
NONE
```

---

# 6. Fold-safe preprocessing / leakage contract

Đây là guardrail bắt buộc của M7.

M4.7 đã khóa feature/preprocessing semantics.

M7 không được thay semantic feature contract chỉ để phục vụ CV.

Tuy nhiên:

`learned preprocessing state`

phải fit lại trong từng fold từ:

`fold-TRAIN only`.

Do đó M7 không được mù quáng tái sử dụng một transformed matrix nếu scaler/encoder state của matrix đó đã học từ rows nằm sau fold-training boundary.

M7.2 phải audit rõ:

- row-level canonical inputs nào được phép tái sử dụng;
- preprocessing object nào phải clone/refit per fold;
- behavioral features nào đã causal-safe;
- learned scaler / encoder state nào phải fold-local;
- feature names / output dimension consistency;
- row alignment;
- sparse matrix contract.

Sampler:

`fold-TRAIN only`

Validation resampling:

`NO`

Fold-validation preprocessing:

`transform only`

---

# 7. Metric contract cho M7

Primary metric:

`F1_fraud`

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
predicted-positive count
predicted-positive rate
```

Reference:

`Accuracy`

Probability/risk score:

`PRESERVE WHEN MODEL SUPPORTS`

M7 phải giữ fold-wise evidence.

Không được chỉ báo:

`mean F1`

rồi che mất unstable folds.

Mỗi candidate tối thiểu phải report:

- từng fold F1 / Recall / Precision;
- từng fold TP / FP / FN / TN;
- alert count / rate;
- runtime;
- warning;
- aggregate center;
- fold-to-fold variation.

Exact aggregation / tie-break rule phải được khóa ở M7.1 trước khi selection runtime bắt đầu.

Không invent arbitrary epsilon sau khi nhìn score.

---

# 8. Randomness policy

Default:

`RANDOM_STATE = 42`

cho primary experiment nếu operation có randomness.

Random seed:

`REPRODUCIBILITY CONTROL`

không phải hyperparameter.

Không được:

```text
seed 1
seed 2
seed 3
...
→ chọn seed có validation F1 cao nhất
```

Nếu candidate phụ thuộc mạnh seed:

`STABILITY FINDING`

Multi-seed chỉ dùng khi:

- có lý do robustness rõ;
- seed set được khai báo trước;
- cùng seed set áp dụng cho mọi candidate trong comparison.

---

# 9. Class-imbalance experiment policy

No-intervention baseline:

`REQUIRED`

Baseline đã tồn tại từ M5/M6:

`IMBALANCE_STRATEGY = NONE`

Recommended intervention sequence:

```text
STEP 1
NONE baseline

        ↓

STEP 2
CLASS_WEIGHT
nếu model hỗ trợ

        ↓

STEP 3
Nếu evidence vẫn cho thấy cần:
RANDOM_OVERSAMPLING
hoặc
RANDOM_UNDERSAMPLING

        ↓

STEP 4
Chỉ khi representation phù hợp
và có lý do:
SMOTE

        ↓

STEP 5
Sau khi learning strategy ổn định:
THRESHOLD SELECTION
```

Không chạy tất cả kỹ thuật chỉ vì chúng tồn tại.

Primary imbalance comparison phải giữ cố định:

- training window;
- model family/config;
- features;
- fold definitions;
- metric code;
- threshold policy;
- random-state policy.

Không được so:

```text
NONE + threshold 0.5
vs
CLASS_WEIGHT + threshold 0.2
```

rồi gán improvement cho class_weight.

---

# 10. SMOTE policy

SMOTE:

`CONDITIONAL / NOT DEFAULT`

Chỉ được triển khai nếu trước runtime chứng minh:

```text
feature representation:
SUITABLE

sampler:
INSIDE FOLD-TRAIN ONLY

validation/test synthetic samples:
NONE

preprocessing/sampling pipeline:
NO LEAKAGE
```

Nếu chưa đủ evidence:

`DO NOT USE SMOTE`

M7 không bắt buộc phải dùng SMOTE để PASS.

---

# 11. Probability calibration policy

Probability calibration:

`OPTIONAL / CONDITIONAL`

M7 không mặc định phải dùng:

- Platt scaling;
- isotonic calibration;
- calibrated classifier.

M6 chỉ cho thấy:

`Decision Tree score behavior = extreme/discrete`

Điều đó tạo câu hỏi nghiên cứu, không tự động tạo calibration requirement.

Nếu calibration được nghiên cứu:

- phải có question riêng;
- phải được fit bằng TRAIN-only / fold-safe protocol;
- phải được tách khỏi threshold comparison;
- không dùng FINAL TEST;
- không gọi calibrated probability tốt hơn nếu không có metric/evidence phù hợp.

Core M7 có thể PASS mà không thực hiện calibration.

---

# 12. Computational-budget policy

M6 đã cho thấy:

```text
W_LONG cost > W_SHORT cost

RF cost > DT cost > LR cost
```

M7 phải dùng evidence này để thiết kế experiment có kiểm soát.

Được phép:

- khóa training window trước khi broad tuning nếu robustness evidence đủ;
- shortlist model families trước khi tuning;
- tune ít candidate có lý do;
- dùng resource budget như constraint.

Không được:

- silently subsample chỉ một candidate;
- giảm số fold cho model đắt mà giữ 3 fold cho model khác trong cùng comparison;
- đổi feature set riêng cho model đắt;
- đổi training window riêng để làm model chạy được;
- bỏ failed candidate khỏi report mà không ghi computational finding.

Nếu compute blocker xảy ra:

```text
STOP
→ record
→ design controlled workaround
→ apply fairly
→ rerun
```

---

# 13. M7 experiment sequence

M7 được chia thành 9 substeps.

```text
M7.1
Selection Charter / scope / guardrails
        ↓
M7.2
Artifact audit + fold-safe temporal-CV infrastructure
        ↓
M7.3
Training-window robustness
W_SHORT vs W_LONG
        ↓
M7.4
Model-family robustness / shortlist
LR vs DT vs RF
        ↓
M7.5
Controlled class-imbalance experiments
        ↓
M7.6
Moderate hyperparameter tuning
        ↓
M7.7
Candidate selection + external VALIDATION confirmation
+ upstream freeze
        ↓
M7.8
Numerical threshold selection on external VALIDATION
        ↓
M7.9
Final Selection Registry / Decision Log / M7 Gate
        ↓
READY FOR PROTECTED FINAL EVALUATION
```

---

# 14. M7.1 — Khóa Selection Charter, scope và guardrails

Mục tiêu:

khóa luật selection trước khi thấy M7 experiment results.

M7.1 phải khóa:

- roles của TRAIN / temporal-CV / VALIDATION / FINAL TEST;
- candidate families ban đầu;
- metric contract;
- fold template;
- aggregation policy;
- tie-break policy;
- random-state policy;
- resource policy;
- imbalance experiment order;
- tuning scope policy;
- threshold order;
- artifact/registry schema;
- Decision Log schema.

Không train candidate trong M7.1.

Expected state:

```text
M7 Selection Charter:
LOCKED

Temporal CV contract:
LOCKED

Selection metrics:
LOCKED

FINAL TEST:
PROTECTED

Runtime selection result:
NOT YET PRODUCED
```

---

# 15. M7.2 — Artifact audit và fold-safe temporal-CV infrastructure

Mục tiêu:

chứng minh infrastructure có thể chạy temporal folds mà không leakage.

M7.2 phải audit:

- M4 feature/preprocessing contract;
- M6 evidence handoff;
- training row lineage;
- timestamp boundaries;
- fold membership;
- positive support;
- preprocessing refit behavior;
- sampler placement;
- prediction/risk-score schema;
- metric reconstruction;
- persistence;
- reproducibility.

M7.2 cần một shared experiment runner cho M7.

Runner phải hỗ trợ tối thiểu:

- fold-train fit;
- fold-validation predict;
- risk score;
- metric bundle;
- confusion counts;
- runtime;
- warnings;
- candidate identity;
- random state;
- preprocessing identity;
- imbalance identity;
- fold identity.

M7.2 không chọn winner.

---

# 16. M7.3 — Controlled training-window robustness

Câu hỏi:

> W_SHORT provisional direction của M6 có ổn định qua temporal folds không?

Primary comparison:

```text
W_SHORT
vs
W_LONG
```

Trong mỗi model family/config được dùng cho robustness comparison:

`ONLY TRAINING WINDOW CHANGES`

Phải giữ cùng:

- feature semantics;
- preprocessing protocol;
- model/config;
- imbalance strategy;
- threshold policy;
- temporal folds;
- metric code;
- random-state policy.

M7.3 đọc:

- fold-wise F1;
- fold-wise Recall;
- fold-wise Precision;
- confusion counts;
- alert burden;
- fold variation;
- runtime.

Possible outcomes:

```text
ROBUST W_SHORT PREFERENCE
ROBUST W_LONG PREFERENCE
MIXED / INCONCLUSIVE
```

Nếu evidence đủ rõ:

M7.3 có thể khóa training-window preference cho downstream M7.

Nếu không:

training window vẫn OPEN và downstream design phải giữ ambiguity hoặc có controlled tie-break protocol.

Không dùng parsimony trước predictive/robustness evidence.

---

# 17. M7.4 — Model-family robustness và shortlist

Câu hỏi:

> Khi training-window scope đã được kiểm soát, LR / DT / RF model-family trade-off có ổn định qua folds không?

M7.4 không được so model family bằng mixed-window comparison.

Primary comparison:

```text
same training window
same feature/preprocessing protocol
same imbalance policy
same folds
same threshold policy

variable:
model family / baseline config
```

M7.4 có thể tạo:

`MODEL SHORTLIST`

không cần giữ cả 3 families nếu evidence cho thấy candidate nào đó không còn competitive và Decision Log giải thích rõ.

Shortlist decision phải đọc:

- F1 primary;
- Recall / Precision;
- fold stability;
- FP / FN;
- alert burden;
- computational cost;
- M6 error findings.

Không được shortlist chỉ bằng one-number ranking.

---

# 18. M7.5 — Controlled class-imbalance experiments

Mục tiêu:

xác định có cần learning intervention ngoài `NONE` hay không.

Initial reference:

`NONE`

Candidate order:

```text
CLASS_WEIGHT
then
RANDOM OVER/UNDER SAMPLING if justified
then
SMOTE only if conditional gate passes
```

Primary comparison:

```text
same model/config
same training window
same folds
same preprocessing
same threshold policy
same metrics

variable:
imbalance strategy
```

M7.5 phải đặc biệt kiểm tra:

- Recall gain;
- Precision cost;
- FN reduction;
- FP increase;
- alert burden;
- stability across folds.

Không được gọi intervention tốt hơn chỉ vì Recall tăng nếu FP/Precision/F1 hoặc stability suy giảm đáng kể.

Final imbalance strategy có thể:

`NONE`

nếu intervention không tạo evidence đủ tốt.

---

# 19. M7.6 — Moderate hyperparameter tuning

Mục tiêu:

tuning có kiểm soát trên shortlisted candidate(s).

Search space:

`SMALL / JUSTIFIED / DECLARED BEFORE RUN`

Không:

- exhaustive search lớn;
- adaptive grid expansion sau khi thấy result chỉ để săn score;
- tuning random seed;
- tuning bằng FINAL TEST.

Temporal CV:

`REQUIRED`

cho multi-candidate hyperparameter tuning.

Mỗi candidate phải có immutable identity gồm:

- model family;
- training window;
- imbalance strategy;
- hyperparameters;
- preprocessing version;
- fold template;
- random state;
- threshold policy.

Tuning report phải giữ:

- fold-wise metrics;
- aggregate metrics;
- variation;
- runtime;
- warnings;
- failed candidates.

Possible decision:

```text
SELECT CONFIG
or
KEEP BASELINE CONFIG
```

Tuning không bắt buộc phải thắng baseline.

---

# 20. M7.7 — Candidate selection + external VALIDATION confirmation

Mục tiêu:

chọn một upstream development candidate trước threshold selection.

Candidate identity phải bao gồm:

```text
feature/preprocessing contract
training window
model family
model hyperparameters
imbalance strategy
random-state policy
probability interface
```

Temporal CV cung cấp:

`robustness / tuning evidence`

External VALIDATION cung cấp:

`development confirmation / selection evidence`

M7.7 phải đối chiếu:

- CV behavior;
- external VALIDATION behavior;
- M6 baseline/error evidence;
- computational feasibility.

Nếu CV tốt nhưng external VALIDATION xấu đáng kể:

đây là:

`TEMPORAL GENERALIZATION FINDING`

Không được âm thầm bỏ external validation result.

Kết thúc M7.7 nếu evidence đủ:

```text
UPSTREAM PIPELINE:
FROZEN

TRAINING WINDOW:
FROZEN

MODEL FAMILY:
FROZEN

MODEL HYPERPARAMETERS:
FROZEN

IMBALANCE STRATEGY:
FROZEN
```

Threshold:

`STILL OPEN`

FINAL TEST:

`STILL PROTECTED`

---

# 21. M7.8 — Numerical threshold selection

Chỉ bắt đầu khi M7.7 upstream candidate đã freeze.

Sequence:

```text
frozen feature/preprocessing
        ↓
frozen training window
        ↓
frozen model/config
        ↓
frozen imbalance strategy
        ↓
external VALIDATION risk score
        ↓
threshold comparison
        ↓
threshold selection
        ↓
freeze threshold
```

Threshold selection source:

`EXTERNAL VALIDATION`

FINAL TEST:

`PROHIBITED`

Threshold experiment phải report:

- threshold candidate;
- F1;
- Recall;
- Precision;
- TP;
- FP;
- FN;
- TN;
- alert count/rate.

M7.8 không được coi threshold là cách “sửa” model một cách tự động.

M6 handoff đã chỉ ra:

- nhiều LR/RF FN nằm xa 0.5;
- DT score behavior cực đoan.

Do đó threshold candidate set phải có lý do và được khai báo trước khi đọc comparison result.

Final threshold có thể vẫn là default boundary nếu evidence không justify thay đổi.

---

# 22. M7.9 — Final Selection Registry, Decision Log và M7 Gate

M7.9 tổng hợp toàn milestone.

Expected registry phải chứa:

- selected training window;
- selected model family;
- selected model config;
- selected imbalance strategy;
- selected threshold;
- temporal-CV evidence;
- external VALIDATION confirmation;
- computational evidence;
- open limitations;
- artifact fingerprints;
- full Decision Log.

M7.9 không mở FINAL TEST.

M7 Gate PASS có nghĩa:

```text
development selection:
COMPLETE

upstream configuration:
FROZEN

threshold:
FROZEN

FINAL TEST:
STILL UNTOUCHED

ready:
PROTECTED FINAL EVALUATION
```

---

# 23. Candidate narrowing policy

M7 không bắt buộc chạy full Cartesian product:

```text
2 windows
× 3 models
× many hyperparameters
× many imbalance strategies
× many thresholds
```

Cách đó:

- tốn tài nguyên;
- tăng multiple-comparison risk;
- làm selection khó audit;
- dễ validation overfitting.

M7 dùng staged narrowing:

```text
window robustness
        ↓
model shortlist
        ↓
imbalance experiment
        ↓
moderate tuning
        ↓
upstream freeze
        ↓
threshold selection
```

Mỗi narrowing decision phải:

- dựa trên evidence đã review;
- ghi Decision Log;
- không dùng FINAL TEST;
- không thay luật sau khi thấy kết quả bất lợi.

---

# 24. External VALIDATION reuse guardrail

External VALIDATION được phép hỗ trợ development decisions.

Nhưng M7 phải tránh biến nó thành hidden training set qua quá nhiều vòng thử.

Do đó:

- tuning candidate chủ yếu dùng temporal CV;
- external VALIDATION dùng controlled confirmation;
- threshold selection diễn ra sau upstream freeze;
- không liên tục mở rộng search space theo external-validation score;
- nếu một result bất ngờ tạo hypothesis mới, hypothesis đó phải được ghi rồi thiết kế experiment mới rõ ràng, không chỉnh ad hoc.

---

# 25. Error-analysis handoff vào selection

M6 error findings không phải final selection metric.

Chúng là supporting evidence.

M7 nên theo dõi:

## Fraud-side

```text
FN count
shared-miss behavior
model-specific catches
risk-score depth of FN
```

## Non-fraud-side

```text
FP count
alert burden
model-specific FP
is_new_merchant association
location_state concentration
```

M7 không được tune trực tiếp một model để “khớp” các subgroup này nếu experiment question chưa được khóa.

Nếu M7 candidate làm aggregate F1 tốt hơn nhưng error behavior xấu đi rõ:

Decision Log phải ghi trade-off.

---

# 26. M7 artifact policy

Mỗi experiment substep phải persist đủ để independently review.

Minimum per candidate/fold:

```text
candidate_id
fold_id
training boundary
validation boundary
training rows
validation rows
training fraud
validation fraud

feature/preprocessing identity
model identity
hyperparameters
imbalance strategy
random_state
threshold policy

y_pred identity
risk_score identity

F1
Recall
Precision
TP
FP
FN
TN
alert count/rate

fit runtime
prediction runtime
warnings

integrity status
```

M7 consolidated registry phải tránh chỉ lưu “best score”.

Failed / invalid candidates cũng phải được ghi nếu đã chạy.

---

# 27. Experiment execution protocol với AI

M7 tiếp tục protocol Type C.

Cho mỗi runtime substep:

1. Xác định đúng câu hỏi.
2. Đọc CANON liên quan.
3. Khóa candidate/config/search space trước result.
4. AI tạo notebook.
5. Notebook có artifact assertions / leakage gates / metadata / persistence.
6. Người thực hiện `Restart Kernel → Run All`.
7. Giữ nguyên output / warnings / errors.
8. AI review:
   - execution completeness;
   - fold integrity;
   - preprocessing leakage;
   - sampler placement;
   - artifact identity;
   - metric consistency;
   - comparability;
   - computational behavior;
   - FINAL TEST isolation.
9. Nếu thiếu check:
   - thêm check;
   - rerun.
10. Chỉ sau runtime review mới viết Findings / Decision Log / Gate.
11. Chỉ khi substep PASS mới chuyển sang bước phụ thuộc tiếp theo.

Không invent runtime score.

Không PASS notebook chưa có output thật.

---

# 28. M7 Decision Log khởi tạo

## M7-D01 — Milestone role

Decision:

M7 là:

`MODEL SELECTION / ROBUSTNESS / TUNING`

Status:

`READY TO LOCK`

---

## M7-D02 — FINAL TEST

Decision:

Không dùng FINAL TEST trong bất kỳ development selection nào của M7.

Status:

`INHERITED — LOCKED`

---

## M7-D03 — Primary CV

Decision:

`FORWARD / EXPANDING TEMPORAL VALIDATION`

Status:

`INHERITED — READY TO LOCK`

---

## M7-D04 — CV folds

Decision:

```text
Q2 2018
Q3 2018
Q4 2018
```

Status:

`INHERITED — READY TO LOCK`

---

## M7-D05 — Metric contract

Decision:

```text
Primary:
F1_fraud

Mandatory:
Recall_fraud
Precision_fraud
TP / FP / FN / TN
alert count/rate
```

Status:

`INHERITED — READY TO LOCK`

---

## M7-D06 — Training-window candidate

Decision:

```text
W_SHORT
W_LONG
```

Current evidence:

`PROVISIONAL W_SHORT DIRECTION`

Final winner:

`OPEN`

---

## M7-D07 — Model families

Decision:

Initial candidate families:

```text
Logistic Regression
Decision Tree
Random Forest
```

Final family:

`OPEN`

---

## M7-D08 — No-intervention baseline

Decision:

`NONE baseline must remain reference`

Status:

`INHERITED — LOCKED`

---

## M7-D09 — Imbalance candidates

Decision:

```text
CLASS_WEIGHT:
AUTHORIZED

RANDOM_OVERSAMPLING:
AUTHORIZED

RANDOM_UNDERSAMPLING:
AUTHORIZED

SMOTE:
CONDITIONAL
```

Winner:

`OPEN`

---

## M7-D10 — Hyperparameter search

Decision:

`SMALL / JUSTIFIED / DECLARED BEFORE RUN`

Status:

`READY TO LOCK`

---

## M7-D11 — Random state

Decision:

`42`

for primary stochastic experiments.

Status:

`INHERITED — READY TO LOCK`

---

## M7-D12 — Threshold

Decision:

Threshold selection occurs only after upstream candidate freeze.

Selection partition:

`EXTERNAL VALIDATION`

Final threshold:

`OPEN`

---

## M7-D13 — Probability calibration

Decision:

`OPTIONAL / CONDITIONAL`

Status:

`OPEN`

---

## M7-D14 — Fold preprocessing

Decision:

Learned preprocessing state:

`FOLD-TRAIN ONLY`

Status:

`INHERITED — READY TO LOCK`

---

## M7-D15 — Sampler placement

Decision:

Sampler:

`FOLD-TRAIN ONLY`

Validation resampling:

`NO`

Status:

`INHERITED — READY TO LOCK`

---

## M7-D16 — M7 end state

Decision:

M7 phải freeze development configuration trước protected final evaluation.

Status:

`READY TO LOCK`

---

# 29. Open items khi bắt đầu M7

```text
O01
Exact metric aggregation / tie-break rule
→ LOCK IN M7.1

O02
Fold-safe implementation path từ M4 artifacts
→ AUDIT IN M7.2

O03
Final training window
→ M7.3

O04
Model shortlist / final family
→ M7.4 → M7.7

O05
Need for imbalance intervention
→ M7.5

O06
Exact hyperparameter search spaces
→ DECLARE BEFORE M7.6 RUNTIME

O07
Probability calibration need
→ CONDITIONAL

O08
Exact threshold candidate set / policy
→ DECLARE BEFORE M7.8 RUNTIME

O09
Final development configuration
→ M7.9

O10
FINAL TEST performance
→ NOT A M7 DEVELOPMENT INPUT
```

---

# 30. M7 Gate — planned checks

M7 Gate dự kiến yêu cầu:

```text
G01
M6 handoff integrity
PASS

G02
Temporal-CV infrastructure
PASS

G03
Fold order / overlap / positive support
PASS

G04
Fold-safe preprocessing
PASS

G05
Sampler leakage isolation
PASS

G06
Training-window robustness decision
RECORDED

G07
Model-family/config selection evidence
COMPLETE

G08
Imbalance strategy evidence
COMPLETE

G09
Hyperparameter tuning evidence
COMPLETE OR EXPLICITLY NOT NEEDED

G10
External VALIDATION confirmation
COMPLETE

G11
Upstream candidate freeze
COMPLETE

G12
Threshold selection
COMPLETE

G13
Final development configuration registry
COMPLETE

G14
FINAL TEST isolation
PASS

G15
Decision Log / limitations / handoff
COMPLETE
```

M7 PASS không yêu cầu:

- một intervention phải thắng NONE;
- tuning phải cải thiện baseline;
- SMOTE phải được chạy;
- calibration phải được chạy;
- RF phải được chọn;
- W_SHORT phải được chọn;
- threshold phải khác 0.5.

M7 PASS yêu cầu:

`selection process is controlled, evidence-backed, reproducible and final-test-safe`.

---

# 31. Planned substep outputs

```text
M7.1
CANON-M7.1 — Selection Charter, scope và guardrails.md

M7.2
07_02_temporal_cv_infrastructure_audit.ipynb
+
M7 experiment contract / runner artifacts

M7.3
07_03_training_window_robustness.ipynb

M7.4
07_04_model_family_robustness_and_shortlist.ipynb

M7.5
07_05_imbalance_strategy_experiments.ipynb

M7.6
07_06_moderate_hyperparameter_tuning.ipynb

M7.7
07_07_candidate_selection_and_validation_confirmation.ipynb

M7.8
07_08_threshold_selection.ipynb

M7.9
CANON-M7.9 — Final Selection Registry, Decision Log và M7 Gate.md
```

Tên artifact runtime chi tiết được khóa trong từng substep trước khi chạy.

---

# 32. Những điều M7 tuyệt đối không được làm

Không:

- random shuffle temporal data cho primary selection;
- fit preprocessing trên fold-validation;
- resample validation;
- SMOTE validation;
- dùng FINAL TEST trong tuning;
- tune threshold cùng lúc với imbalance rồi gán tác động sai;
- chọn seed thuận lợi;
- mở rộng grid vô hạn sau khi xem validation score;
- đổi metric sau khi xem result;
- drop candidate thất bại khỏi registry;
- silently subsample candidate đắt;
- so model family bằng mixed training windows;
- gọi probability là calibrated confidence nếu chưa calibration-audit;
- suy rộng synthetic fraud findings thành production behavior.

---

# 33. M7 completion definition

M7 được coi là hoàn thành khi project có thể trả lời chắc chắn:

> Temporal CV infrastructure có leakage-safe không?

> Training-window decision dựa trên robustness evidence nào?

> Model family/config nào được chọn và vì sao?

> Class imbalance intervention nào được chọn, hoặc vì sao giữ NONE?

> Tuning đã được thực hiện theo search space nào, hoặc vì sao không cần?

> Temporal-CV và external VALIDATION có đồng thuận hay có temporal-generalization warning?

> Upstream model configuration đã freeze chưa?

> Threshold được chọn từ external VALIDATION theo policy nào?

> Mọi selected artifact/config có identity và persistence rõ chưa?

> FINAL TEST có được bảo vệ hoàn toàn trong suốt M7 không?

Nếu tất cả câu trả lời đủ evidence:

```text
M7 — PASS

Development Configuration:
FROZEN

FINAL TEST:
UNTOUCHED / PROTECTED

Handoff:
READY FOR PROTECTED FINAL EVALUATION
```

---

# 34. Final planning block

```text
Milestone 7:
MODEL SELECTION / ROBUSTNESS / TUNING

Input:
M6 verified evaluation + error evidence
M4/M5 canonical modeling contracts

Primary temporal CV:
FORWARD / EXPANDING

Validation folds:
Q2 / Q3 / Q4 2018

External VALIDATION:
2019-01 → 2019-05

FINAL TEST:
PROTECTED
NO DEVELOPMENT ACCESS

Selection dimensions:
training window
model family/config
imbalance strategy
hyperparameters
numerical threshold

Imbalance order:
NONE
→ CLASS_WEIGHT
→ RANDOM OVER/UNDER if justified
→ SMOTE conditional

Threshold:
AFTER UPSTREAM FREEZE

Random state:
42

Primary metric:
F1_fraud

Mandatory:
Recall_fraud
Precision_fraud
TP / FP / FN / TN
alert count/rate

M7 target output:
FROZEN DEVELOPMENT CONFIGURATION

M7 final handoff:
READY FOR PROTECTED FINAL EVALUATION
```
