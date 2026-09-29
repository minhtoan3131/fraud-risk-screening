# M3.8 — Tổng hợp Experiment Specification v1.0, Decision Log, Open Questions và M3 Gate

## 1. Mục tiêu

M3.8 là bước tổng hợp cuối của Milestone 3.

M3.8 không tạo thêm experiment mới.

Mục tiêu là gom toàn bộ kết quả M3.1 → M3.7 thành một specification thống nhất để các milestone modeling phía sau không phải tự quyết định lại:

- prediction point;
- temporal evaluation;
- train / validation / final-test boundary;
- historical warm-up;
- metric strategy;
- quyền sử dụng từng partition;
- training-window candidate và comparison protocol;
- class-imbalance policy;
- temporal CV;
- tuning policy;
- threshold-selection boundary;
- leakage guardrail;
- các câu hỏi còn mở.

Sản phẩm của M3.8:

```text
Experiment Specification v1.0
+
Consolidated Decision Log
+
Open Questions / Handoff
+
M3 Gate
```

---

# 2. Phạm vi đã hoàn thành của Milestone 3

## Phân tích

M3 được thiết kế để tạo experiment protocol chứ không phải tìm final model.

Qua M3.1–M3.7, project đã lần lượt:

- khóa mục tiêu và guardrail;
- xây temporal map;
- audit split trên raw artifact thật;
- khóa validation/final-test boundary;
- khóa metric strategy;
- khóa quyền TRAIN / VALIDATION / FINAL TEST;
- thiết kế controlled comparison cho training window;
- thiết kế policy cho imbalance, temporal CV và tuning.

Không có bước nào được phép tự biến thành final model selection nếu chưa có model experiment thật.

## Kết luận

`Milestone 3 scope: COMPLETED AS EXPERIMENT-DESIGN MILESTONE`

Không còn thiếu protocol bắt buộc trước khi chuyển sang feature/preprocessing/modeling work.

---

# 3. Problem definition và prediction point

## Phân tích

Bài toán tiếp tục là binary supervised classification.

Positive class:

`fraud`

theo ground truth của dataset.

Model được định vị như một:

`transaction risk-screening component`

tạo risk signal / probability phục vụ screening.

Prediction point:

`thời điểm transaction cần được screening`.

Mọi feature phải:

- tồn tại tại prediction point;
- hoặc được tính hoàn toàn từ thông tin xảy ra trước prediction point.

Model không có quyền sử dụng future information hoặc information chỉ có sau khi giao dịch đã được xử lý.

## Kết luận

```text
Problem:
Binary fraud-risk screening classification

Positive class:
Fraud

Prediction point:
Transaction screening time

Status:
LOCKED
```

---

# 4. Temporal evaluation principle

## Phân tích

Dataset đã cho thấy temporal heterogeneity mạnh.

Đặc biệt:

- 2017 có fraud prevalence rất khác các năm lân cận;
- tháng 1–10/2019 vẫn có fraud;
- từ tháng 11/2019 đến tháng 2/2020 có 625,025 transaction nhưng 0 fraud.

Vì vậy random split không phải evaluation design chính phù hợp với project.

Evaluation phải mô phỏng:

`past → future`.

M3.2 đã xác định `2019-01 → 2019-10` là positive-bearing future-evaluation reservoir và tách `2019-11 → 2020-02` thành zero-fraud post-break region.

## Kết luận

`Primary evaluation direction: PAST → FUTURE — LOCKED`

`Random split as main evaluation: NOT AUTHORIZED`

---

# 5. Temporal map chính thức

```text
1991 ───────────────────────────────────────────────────────────────→ 2020

1991 → trước modeling window
HISTORICAL CONTEXT RESERVOIR
→ causal warm-up
→ không tự động là classifier-training rows

2015 → 2018
W_LONG candidate classifier-training region

2018
W_SHORT candidate classifier-training region

2019-01 → 2019-05
EXTERNAL VALIDATION

2019-06 → 2019-10
PROTECTED FINAL TEST

2019-11 → 2020-02
ZERO-FRAUD POST-BREAK REGION
→ supplementary diagnostic only
```

---

# 6. Canonical split

## Phân tích

M3.3 đã chạy full-artifact audit thật và chọn `S2_5M_5M`.

Boundary được khóa:

TRAIN:

`Timestamp < 2019-01-01`

nhưng training start vẫn phụ thuộc W_LONG/W_SHORT.

VALIDATION:

`2019-01-01 ≤ Timestamp < 2019-06-01`

FINAL TEST:

`2019-06-01 ≤ Timestamp < 2019-11-01`.

M3.3 xác nhận validation có `712,458 transaction / 1,052 fraud`, final test có `722,955 / 1,035 fraud`; structural và consistency gate đều PASS. 

## Kết luận

```text
TRAIN_END_EXCLUSIVE:
2019-01-01

VALIDATION:
2019-01-01 → 2019-06-01 exclusive

FINAL TEST:
2019-06-01 → 2019-11-01 exclusive

Split:
S2_5M_5M

Status:
LOCKED
```

---

# 7. Training-window candidates

Hai candidate chính thức:

```text
W_LONG_2015_TO_2018

2015-01-01
≤ Timestamp
< 2019-01-01

6,855,270 transaction
9,606 fraud
```

và:

```text
W_SHORT_2018_ONLY

2018-01-01
≤ Timestamp
< 2019-01-01

1,721,615 transaction
2,491 fraud
```

M3.3 xác nhận cả hai khả thi nhưng không có model-performance evidence để chọn winner.

## Kết luận

```text
W_LONG:
CANDIDATE

W_SHORT:
CANDIDATE

Training-window winner:
OPEN / REQUIRES EXPERIMENT
```

---

# 8. Historical context policy

## Phân tích

Classifier-training rows và historical-context rows là hai khái niệm khác nhau.

Older transaction có thể dùng để tạo historical feature cho một recent modeling window mà không cần trở thành classifier-training row.

Đối với transaction T:

```text
Timestamp(history) < Timestamp(T)
```

là điều kiện bắt buộc.

Không được sử dụng transaction cùng Timestamp làm history cho nhau.

Không được dùng:

- current transaction như history của chính nó;
- future transaction;
- lifetime aggregate chứa tương lai;
- target label của prior/future transaction làm behavioral feature state.

M3.1 và M3.5 đã khóa rõ distinction này.

## Kết luận

```text
Historical warm-up:
ALLOWED

Classifier-training membership:
INDEPENDENT FROM HISTORY AVAILABILITY

Strict rule:
Timestamp(history) < Timestamp(current)

Same-timestamp history:
PROHIBITED

Target-label history:
PROHIBITED

Status:
LOCKED
```

---

# 9. Feature-level guardrails mang forward

Các guardrail sau tiếp tục bắt buộc trong M4/modeling:

```text
Raw User:
DO NOT USE AS CLASSIFIER FEATURE

Raw Card:
DO NOT USE AS CLASSIFIER FEATURE

Raw Merchant Name:
DO NOT USE AS CLASSIFIER FEATURE

Errors?:
EXCLUDED FROM MODEL V1
unless prediction-time availability is later proven

Structural location missingness:
DO NOT DROP/FILL MECHANICALLY

Negative Amount:
DO NOT ABS/DROP/CLIP MECHANICALLY
before semantics/preprocessing experiment

Full-history association:
NOT PROOF OF FUTURE GENERALIZATION
```

## Kết luận

Các quyết định preprocessing/feature cụ thể vẫn có thể được nghiên cứu trong M4, nhưng không được vi phạm các guardrail trên.

---

# 10. Evaluation population limitation

## Phân tích

M3.3 xác nhận external validation/final test chủ yếu gồm existing User/Card.

Strict cold-start chỉ chiếm một phần rất nhỏ.

Do đó evaluation chính chủ yếu trả lời:

`Model hoạt động thế nào trên future transaction của population phần lớn đã có historical context?`

Không đủ bằng chứng để tuyên bố mạnh về:

`completely-new User/Card generalization`.

## Kết luận

`Main evaluation population: EXISTING-ENTITY DOMINANT — VERIFIED`

`Cold-start generalization: LIMITED EVIDENCE`

Đây phải được giữ như limitation khi viết báo cáo cuối.

---

# 11. Metric strategy

## Phân tích

Fraud class cực hiếm nên Accuracy có thể rất cao ngay cả khi model bỏ sót toàn bộ fraud.

M3.4 đã khóa metric hierarchy.

## Kết luận

```text
Positive class:
fraud

PRIMARY:
F1_fraud

MANDATORY SECONDARY:
Recall_fraud
Precision_fraud

MANDATORY DIAGNOSTIC:
TP
FP
FN
TN

OPERATIONAL DIAGNOSTIC:
predicted_positive_count
predicted_positive_rate

REFERENCE ONLY:
Accuracy

PROBABILITY OUTPUT:
retain when available

Final numeric threshold:
OPEN
```

---

# 12. Advanced ranking metrics

ROC-AUC / PR-AUC hoặc các ranking metric khác:

```text
OPTIONAL / DEFERRED
```

Chúng không thuộc core metric set của Experiment Specification v1.0.

Nếu sau này bổ sung:

- phải định nghĩa vai trò;
- phải ghi Decision Log;
- không tự thay F1_fraud làm primary metric một cách âm thầm.

---

# 13. Partition roles

## TRAIN

```text
Role:
LEARNING PARTITION

Được phép:
fit preprocessing
fit encoder/scaler/imputer
fit model
class weighting
resampling nếu protocol cho phép

Không được:
nhìn future validation/test information
```

## VALIDATION

```text
Role:
DEVELOPMENT / SELECTION PARTITION

Được phép:
predict
compute metrics
feature comparison
preprocessing comparison
model/config comparison
training-window comparison
imbalance comparison
threshold selection
development error analysis

Không được:
fit learned preprocessing/model state của candidate đang đánh giá
resample
```

## FINAL TEST

```text
Role:
PROTECTED FINAL EVALUATION

Được phép:
final frozen-pipeline evaluation
final metric report
descriptive post-evaluation analysis

Không được:
feature selection
preprocessing selection
training-window selection
model selection
hyperparameter tuning
imbalance-strategy selection
threshold selection
```

M3.5 khóa rõ các quyền này.

---

# 14. Learned state và causal stream state

Phải phân biệt:

```text
LEARNED STATE
scaler / encoder / imputer / model parameters
→ fit trên TRAIN
→ frozen khi evaluate validation/test
```

và:

```text
CAUSAL TRANSACTION-HISTORY STATE
prior observable transaction events
→ có thể tiến theo thời gian
→ nhưng không dùng label
```

Việc historical state tiến theo thời gian không được diễn giải là validation/test đang fit model.

---

# 15. Final-test isolation

Final test:

`2019-06 → 2019-10`

không được sử dụng trong bất kỳ vòng lựa chọn nào.

Không được:

```text
run test
→ thấy Recall thấp
→ đổi threshold
→ run test lại
```

hoặc:

```text
run test
→ feature A kém
→ thêm feature B
→ run lại cùng test
```

Test isolation áp dụng cho:

- code;
- notebook;
- người thực hiện;
- AI hỗ trợ.

Nếu final-test result tham gia development decision, partition đó đã bị tiêu hao như development data.

---

# 16. Final TRAIN + VALIDATION refit

Workflow:

```text
TRAIN + VALIDATION
→ refit final model
→ FINAL TEST
```

chưa được tự động cho phép.

Trạng thái:

`OPEN / NOT AUTHORIZED BY DEFAULT`.

Nếu sau này muốn sử dụng, phải có protocol riêng được khóa trước khi tiêu hao final test.

---

# 17. Training-window comparison protocol

M3.6 đã khóa controlled comparison:

```text
THAY ĐỔI:
training window

GIỮ CỐ ĐỊNH:
validation period
feature version
preprocessing procedure
model/configuration
imbalance strategy
metric implementation
threshold policy
evaluation code
historical warm-up principle
randomness policy
```

Primary comparison target:

`validation F1_fraud`

nhưng bắt buộc đọc:

- Recall;
- Precision;
- Confusion Matrix;
- predicted-positive statistics.

M3.6 không cho phép window-specific tuning trong primary comparison.

## Kết luận

`Training-window comparison protocol: LOCKED`

`Winner: REQUIRES REAL EXPERIMENT`

---

# 18. Training-window tie-break policy

Nếu F1 chênh lệch rất nhỏ hoặc trade-off khó phân biệt:

1. đọc Recall / Precision / FN / FP;
2. không đặt arbitrary epsilon kiểu `ΔF1 < 0.01`;
3. thực hiện robustness protocol nếu cần;
4. nếu vẫn không có stable predictive preference đủ mạnh, dùng W_SHORT như parsimony fallback do training footprint nhỏ hơn.

W_SHORT fallback không phải winner mặc định trước experiment.

---

# 19. Class-imbalance policy

## Baseline

Bắt buộc có:

`NONE / no-intervention baseline`.

## Authorized candidates

```text
CLASS_WEIGHT
RANDOM_OVERSAMPLING
RANDOM_UNDERSAMPLING
```

## Conditional

```text
SMOTE
```

SMOTE chỉ được dùng khi feature representation và pipeline cho phép synthetic interpolation hợp lệ và leakage-safe.

M3.7 không chọn final imbalance winner.

## Kết luận

`Final imbalance strategy: OPEN / REQUIRES EXPERIMENT`

---

# 20. Resampling policy

Resampling phải diễn ra:

```text
AFTER SPLIT
```

và:

```text
TRAINING PORTION ONLY
```

Trong Cross-validation:

```text
FOLD SPLIT
→ RESAMPLE FOLD-TRAIN ONLY
→ FIT
→ EVALUATE FOLD-VALIDATION
```

Không được resample:

- external VALIDATION;
- FINAL TEST.

Hai partition này phải giữ natural class distribution.

---

# 21. Temporal CV policy

Primary CV strategy:

`forward / expanding temporal validation`.

Không sử dụng shuffled KFold hoặc shuffled StratifiedKFold làm primary model-selection CV.

M3.7 khóa template:

```text
FOLD 1

train:
training-window start → trước 2018-04-01

validation:
2018-04-01 → trước 2018-07-01
```

```text
FOLD 2

train:
training-window start → trước 2018-07-01

validation:
2018-07-01 → trước 2018-10-01
```

```text
FOLD 3

train:
training-window start → trước 2018-10-01

validation:
2018-10-01 → trước 2019-01-01
```

M3.7 khóa forward/expanding CV, preprocessing và sampler bên trong fold, còn external validation vẫn giữ độc lập.

---

# 22. Ba tầng evaluation/development

Canonical hierarchy:

```text
TẦNG 1
TEMPORAL CV BÊN TRONG TRAIN
→ tuning
→ stability
→ robustness

        ↓

TẦNG 2
EXTERNAL VALIDATION
2019-01 → 2019-05
→ development selection
→ comparison
→ threshold selection

        ↓

TẦNG 3
FINAL TEST
2019-06 → 2019-10
→ final protected evaluation
```

Không được đảo hoặc trộn ba tầng trên.

---

# 23. Preprocessing trong CV

Mọi preprocessing có learned state phải nằm bên trong fold.

Sai:

```text
fit scaler/encoder trên full TRAIN
→ temporal CV
```

Đúng:

```text
fold split
→ fit preprocessing trên fold-training
→ transform fold-validation
```

Category vocabulary cũng là learned state.

Unseen category phải được xử lý bằng policy của preprocessing pipeline, không được học trước từ validation fold.

---

# 24. Tuning policy

Hyperparameter tuning chỉ bắt đầu khi:

- baseline chạy đúng;
- leakage gate PASS;
- feature/preprocessing đủ ổn định;
- metric implementation đã khóa;
- search question rõ.

Search space phải:

- nhỏ;
- có lý do;
- khai báo trước;
- có computational budget.

Không được dùng GridSearchCV lớn chỉ vì thư viện cho phép.

Grid Search hoặc Randomized Search chỉ là công cụ; validity phụ thuộc data-boundary protocol.

---

# 25. Random-state policy

`RANDOM_STATE = 42`

được dùng cho stochastic operation khi phù hợp để hỗ trợ reproducibility.

Không được:

```text
thử nhiều seed
→ chọn seed cho validation score đẹp nhất
```

Nếu result nhạy với seed:

đó là stability finding.

Multi-seed robustness chỉ chạy khi có câu hỏi rõ và phải dùng cùng seed set cho các candidate cần so sánh.

---

# 26. Threshold policy

Numerical threshold cuối vẫn chưa khóa.

Canonical order:

```text
feature / preprocessing
        ↓
training window
        ↓
model / configuration
        ↓
imbalance strategy
        ↓
tuning / robustness
        ↓
freeze upstream pipeline
        ↓
external VALIDATION probability
        ↓
threshold selection
        ↓
freeze threshold
        ↓
FINAL TEST
```

Threshold selection:

`VALIDATION ONLY`.

FINAL TEST không được dùng.

---

# 27. Canonical modeling workflow sau M3

```text
1. Load canonical raw artifact
        ↓
2. Build/verify Timestamp
        ↓
3. Enforce causal ordering
        ↓
4. Apply canonical temporal boundaries
        ↓
5. Build causal historical features
        ↓
6. Select classifier-training candidate
        ↓
7. Fit learned preprocessing on TRAIN / fold-TRAIN only
        ↓
8. Apply optional imbalance intervention on TRAIN / fold-TRAIN only
        ↓
9. Fit baseline/model
        ↓
10. Temporal CV if tuning/robustness is required
        ↓
11. Refit chosen config on full selected TRAIN
        ↓
12. Evaluate external VALIDATION
        ↓
13. Make development decisions
        ↓
14. Select threshold on VALIDATION
        ↓
15. Freeze complete pipeline
        ↓
16. Evaluate FINAL TEST
```

---

# 28. Required experiment metadata

Mỗi experiment modeling sau M3 phải lưu đủ để tái hiện.

Tối thiểu:

```text
Experiment ID

Dataset/artifact identity

Training-window ID
Temporal boundaries

Feature version
Preprocessing version

Model ID
Hyperparameters

Imbalance strategy
Sampling ratio / class weights

Random state

CV specification
Fold boundaries

Threshold policy
Numerical threshold nếu đã khóa

Metric implementation

F1
Recall
Precision
TP
FP
FN
TN
predicted-positive count/rate
Accuracy reference

Runtime / warnings

Decision
Next action
```

---

# 29. Integrity gate trước khi tin metric

Mỗi experiment phải kiểm tra ít nhất:

```text
Correct artifact:
PASS

Temporal order:
PASS

Partition overlap:
NONE

Positive support:
> 0 trong evaluation block

Preprocessing leakage:
NO

Resampling leakage:
NO

Future-label leakage:
NO

Same-timestamp leakage:
NO

Raw ID guardrail:
PASS

Final-test access:
AUTHORIZED FOR CURRENT STAGE

Metric positive-class mapping:
CORRECT
```

Nếu một integrity check quan trọng fail:

`STOP`.

Không diễn giải metric.

---

# 30. Consolidated Decision Log — temporal/split

```text
M3.2-D01
Full-history classifier training không phải default.
LOCKED

M3.2-D02
Historical warm-up được giữ cho recent window.
LOCKED

M3.2-D03
11/2019 là temporal break quan trọng cho design.
LOCKED

M3.2-D05
2019-11→2020-02 không phải final fraud-performance test.
LOCKED

M3.3-D01
S2_5M_5M là canonical temporal split.
LOCKED

TRAIN cutoff:
< 2019-01-01
LOCKED

VALIDATION:
2019-01 → 2019-05
LOCKED

FINAL TEST:
2019-06 → 2019-10
LOCKED

Training-window winner:
OPEN / REQUIRES EXPERIMENT
```

---

# 31. Consolidated Decision Log — metrics

```text
Positive class:
fraud
LOCKED

Primary metric:
F1_fraud
LOCKED

Secondary:
Recall_fraud
Precision_fraud
LOCKED

Confusion Matrix:
MANDATORY
LOCKED

Predicted-positive count/rate:
MANDATORY DIAGNOSTIC
LOCKED

Accuracy:
REFERENCE ONLY
LOCKED

Probability output:
RETAIN WHEN AVAILABLE
LOCKED

Numerical threshold:
OPEN

Advanced ranking metrics:
OPTIONAL / DEFERRED
```

---

# 32. Consolidated Decision Log — partition permissions

```text
Learned preprocessing:
TRAIN ONLY
LOCKED

Model fitting:
TRAIN ONLY
LOCKED

Resampling:
TRAIN ONLY
LOCKED

Development selection:
VALIDATION
LOCKED

Threshold selection:
VALIDATION
LOCKED

Final-test tuning:
PROHIBITED
LOCKED

Causal prior event history:
ALLOWED
LOCKED

Historical fraud label as feature state:
PROHIBITED
LOCKED

TRAIN+VALIDATION final refit:
OPEN / NOT AUTHORIZED BY DEFAULT
```

---

# 33. Consolidated Decision Log — training-window experiment

```text
Candidates:
W_LONG
W_SHORT
LOCKED

Only intended variable:
training window
LOCKED

Shared validation:
2019-01→05
LOCKED

Final-test selection access:
PROHIBITED
LOCKED

Primary comparison metric:
F1_fraud
LOCKED

Window-specific tuning:
PROHIBITED IN PRIMARY COMPARISON
LOCKED

Robustness:
USE IF PRIMARY EVIDENCE INCONCLUSIVE
LOCKED

Winner:
REQUIRES EXPERIMENT
```

---

# 34. Consolidated Decision Log — imbalance/CV/tuning

```text
No-intervention baseline:
REQUIRED
LOCKED

Class_weight:
AUTHORIZED CANDIDATE

Random oversampling:
AUTHORIZED CANDIDATE

Random undersampling:
AUTHORIZED CANDIDATE

SMOTE:
CONDITIONAL / DEFERRED

Primary CV:
FORWARD / EXPANDING TEMPORAL CV
LOCKED

CV folds:
Q2 / Q3 / Q4 2018
LOCKED

Shuffled StratifiedKFold as primary:
NOT AUTHORIZED
LOCKED

Preprocessing inside fold:
REQUIRED
LOCKED

Sampler inside fold:
REQUIRED
LOCKED

Tuning search:
SMALL / JUSTIFIED / PREDECLARED
LOCKED

Random seed:
REPRODUCIBILITY, NOT TUNING
LOCKED

Final imbalance winner:
REQUIRES EXPERIMENT

Final hyperparameters:
REQUIRES EXPERIMENT
```

---

# 35. Open Questions chuyển sang M4/modeling

## O01 — Final feature set

`OPEN`

M4 phải quyết định feature nào thực sự được đưa vào model candidate.

Không được suy ra từ EDA association đơn thuần.

---

## O02 — Behavioral feature set

`OPEN`

Các candidate từ M2 gồm ít nhất:

- time_since_previous_transaction;
- transactions_last_1h;
- amount_minus_previous_mean;
- is_new_merchant.

M2 mới chứng minh causal feasibility/coverage, chưa chứng minh classification benefit.

---

## O03 — Exact historical lookback

`OPEN`

Historical warm-up principle đã khóa.

Nhưng exact lookback:

- full prior history;
- fixed years;
- fixed rolling horizon;
- hoặc strategy khác

chưa được chọn.

---

## O04 — MCC/location encoding

`OPEN`

Cần xử lý high cardinality, structural missingness và leakage/memorization risk phù hợp.

---

## O05 — Amount transformation

`OPEN`

Negative Amount không được xử lý máy móc.

Transformation cuối phải dựa trên experiment/preprocessing reasoning.

---

## O06 — Training-window winner

`REQUIRES EXPERIMENT`

W_LONG vs W_SHORT.

---

## O07 — Final model family

`OPEN / REQUIRES EXPERIMENT`

M3 chưa chọn Logistic Regression, Decision Tree, Random Forest hoặc model khác làm final.

---

## O08 — Final imbalance strategy

`REQUIRES EXPERIMENT`

Không intervention / class_weight / over/undersampling phải được đánh giá thực tế.

SMOTE vẫn conditional.

---

## O09 — Final hyperparameters

`REQUIRES EXPERIMENT`

Không được khóa từ lý thuyết.

---

## O10 — Final numerical threshold

`REQUIRES VALIDATION EXPERIMENT`

Không mặc định 0.5.

---

## O11 — Final TRAIN+VALIDATION refit

`OPEN / NOT AUTHORIZED BY DEFAULT`

Chỉ thực hiện nếu protocol riêng được khóa trước final-test consumption.

---

## O12 — Probability calibration

`DEFERRED / OPTIONAL`

Không thuộc core requirement hiện tại.

---

## O13 — Cold-start / group-based evaluation

`DEFERRED / OPTIONAL`

Main evaluation không đại diện mạnh cho completely-new entities.

---

## O14 — Post-break zero-fraud diagnostic

`DEFERRED / OPTIONAL`

2019-11→2020-02 chỉ được dùng nếu có câu hỏi diagnostic/stress cụ thể.

---

# 36. Những điều không được “mang forward” như kết luận đã chứng minh

M3 không chứng minh:

- W_LONG tốt hơn W_SHORT;
- W_SHORT tốt hơn W_LONG;
- behavioral feature cải thiện F1;
- class_weight cải thiện model;
- SMOTE cần thiết;
- một model family cụ thể tốt nhất;
- threshold cụ thể tốt nhất;
- final F1/Recall/Precision đạt bao nhiêu;
- model có hiệu quả trong ngân hàng thật.

Mọi statement trên cần experiment thích hợp.

---

# 37. Handoff sang M4

M4 không cần thiết kế lại evaluation protocol.

M4 phải tiếp nhận nguyên:

```text
temporal split
metric strategy
partition permissions
causal history rules
feature guardrails
training-window candidates
temporal-CV policy
imbalance boundaries
test-isolation policy
```

Trọng tâm M4 nên chuyển sang:

```text
feature engineering
+
preprocessing representation
+
baseline-ready modeling matrix
+
leakage-safe implementation
```

M4 không được mở FINAL TEST để hỗ trợ feature engineering.

---

# 38. Handoff sang modeling experiments

Khi baseline pipeline đủ rõ, thứ tự thực nghiệm nên là:

```text
baseline feature/preprocessing implementation

        ↓

baseline model
no imbalance intervention

        ↓

controlled training-window comparison

        ↓

feature/preprocessing experiments

        ↓

model/config experiments

        ↓

imbalance experiments nếu cần

        ↓

temporal CV / tuning / robustness

        ↓

external validation comparison

        ↓

threshold selection

        ↓

freeze final pipeline

        ↓

FINAL TEST
```

Thứ tự cụ thể có thể điều chỉnh nếu evidence yêu cầu, nhưng mọi thay đổi phải có Decision Log và vẫn tuân thủ test isolation.

---

# 39. M3 Gate — GATE 01

## Câu hỏi

Final evaluation theo thời gian đã được định nghĩa chưa?

## Phân tích

Có.

Main evaluation tuân thủ past → future.

Validation và final test nằm trong positive-bearing 2019 pre-break region.

Zero-fraud post-break region đã được tách riêng.

## Kết luận

`PASS`

---

# 40. M3 Gate — GATE 02

## Câu hỏi

TRAIN / VALIDATION / FINAL TEST có boundary cụ thể chưa?

## Phân tích

Có.

```text
TRAIN:
Timestamp < 2019-01-01
training start phụ thuộc W_LONG/W_SHORT

VALIDATION:
2019-01-01 → trước 2019-06-01

FINAL TEST:
2019-06-01 → trước 2019-11-01
```

## Kết luận

`PASS`

---

# 41. M3 Gate — GATE 03

## Câu hỏi

Validation và final test có đủ fraud để đánh giá không?

## Phân tích

M3.3 actual audit xác nhận:

Validation:

`1,052 fraud`.

Final test:

`1,035 fraud`.

Cả hai partition đều có positive support thực tế và structural gate PASS.

## Kết luận

`PASS`

---

# 42. M3 Gate — GATE 04

## Câu hỏi

Final test có được bảo vệ khỏi model selection chưa?

## Phân tích

Có.

Feature, preprocessing, model, training window, hyperparameter, imbalance strategy và threshold đều bị cấm sử dụng final-test feedback để lựa chọn.

## Kết luận

`PASS`

---

# 43. M3 Gate — GATE 05

## Câu hỏi

Primary / secondary metric có lý do chưa?

## Phân tích

Có.

Primary:

`F1_fraud`.

Secondary:

`Recall_fraud + Precision_fraud`.

Confusion Matrix bắt buộc.

Accuracy reference only.

## Kết luận

`PASS`

---

# 44. M3 Gate — GATE 06

## Câu hỏi

History warm-up và classifier-training rows đã được tách khái niệm chưa?

## Phân tích

Có.

Older data có thể causal warm-up nhưng không tự động tham gia classifier fitting.

Strict prior rule đã khóa.

## Kết luận

`PASS`

---

# 45. M3 Gate — GATE 07

## Câu hỏi

Training-window comparison đã có protocol công bằng chưa?

## Phân tích

Có.

Chỉ thay training window; các yếu tố chính còn lại phải được kiểm soát.

VALIDATION được dùng để chọn.

FINAL TEST không được dùng.

## Kết luận

`PASS`

Winner chưa có là đúng phạm vi và không làm Gate fail.

---

# 46. M3 Gate — GATE 08

## Câu hỏi

Class-imbalance technique đã được giới hạn đúng partition chưa?

## Phân tích

Có.

Resampling chỉ training portion.

Trong CV chỉ fold-training.

Validation/final test giữ natural distribution.

## Kết luận

`PASS`

---

# 47. M3 Gate — GATE 09

## Câu hỏi

CV/tuning có tôn trọng temporal ordering không?

## Phân tích

Có.

Primary CV = forward/expanding temporal CV.

Không dùng shuffled StratifiedKFold làm primary model-selection CV.

## Kết luận

`PASS`

---

# 48. M3 Gate — GATE 10

## Câu hỏi

M3 có tránh premature decision về feature/model/threshold không?

## Phân tích

Có.

Final feature set, model family, hyperparameter, imbalance winner và numerical threshold vẫn OPEN / REQUIRES EXPERIMENT.

## Kết luận

`PASS`

---

# 49. M3 Gate — GATE 11

## Câu hỏi

Các quyết định chuyển sang M4/modeling có được ghi rõ chưa?

## Phân tích

Có.

Open Questions đã xác định rõ:

- feature;
- preprocessing;
- behavioral features;
- Amount;
- encoding;
- training window;
- model;
- imbalance;
- hyperparameters;
- threshold;
- optional calibration/cold-start diagnostics.

## Kết luận

`PASS`

---

# 50. M3 Gate — GATE 12

## Câu hỏi

Còn leakage issue bắt buộc nào chưa có policy xử lý không?

## Phân tích

Các leakage pathway chính đã có guardrail:

- future feature leakage;
- same-timestamp history;
- full-dataset preprocessing;
- validation/test preprocessing fit;
- resampling leakage;
- CV-fold leakage;
- final-test feedback;
- target label in history;
- raw identifier memorization guardrail.

Không có unresolved leakage issue ở mức experiment specification đang block M4.

Điều này không có nghĩa future implementation chắc chắn không có bug; mỗi notebook vẫn phải có integrity checks.

## Kết luận

`PASS`

---

# 51. Tổng hợp M3 Gate

```text
GATE-01 Temporal evaluation:
PASS

GATE-02 Concrete split:
PASS

GATE-03 Positive support:
PASS

GATE-04 Test isolation:
PASS

GATE-05 Metric strategy:
PASS

GATE-06 History/modeling distinction:
PASS

GATE-07 Training-window comparison:
PASS

GATE-08 Imbalance boundary:
PASS

GATE-09 Temporal CV/tuning:
PASS

GATE-10 No premature modeling decision:
PASS

GATE-11 Handoff/open questions:
PASS

GATE-12 Leakage protection:
PASS
```

---

# 52. Final Decision M3

## Phân tích

Milestone 3 đã hoàn thành mục tiêu ban đầu:

tạo một Experiment Specification đủ rõ để modeling phía sau không phải tự phát minh lại evaluation protocol.

Những câu hỏi còn OPEN đều thuộc nhóm cần:

- feature implementation;
- preprocessing implementation;
- model training;
- controlled comparison;
- validation experiment.

Do đó việc chưa có training-window winner, model winner hoặc threshold cuối không phải failure của M3.

Ngược lại, khóa các quyết định này khi chưa có model evidence mới là vi phạm methodology của project.

## Kết luận

```text
Experiment Specification:
v1.0 — LOCKED

Temporal evaluation:
LOCKED

Canonical split:
LOCKED

Metric strategy:
LOCKED

Partition roles:
LOCKED

History warm-up principle:
LOCKED

Training-window comparison protocol:
LOCKED

Imbalance policy:
LOCKED

Temporal-CV policy:
LOCKED

Tuning policy:
LOCKED

Test isolation:
LOCKED

Training-window winner:
OPEN / REQUIRES EXPERIMENT

Final feature/preprocessing:
OPEN

Final model:
OPEN / REQUIRES EXPERIMENT

Final imbalance strategy:
OPEN / REQUIRES EXPERIMENT

Final hyperparameters:
OPEN / REQUIRES EXPERIMENT

Final threshold:
OPEN / REQUIRES VALIDATION EXPERIMENT

Blocking issue:
NONE

M3 Gate:
PASS

Milestone 3 Status:
PASS — EXPERIMENT SPECIFICATION v1.0 LOCKED
```

---

# 53. Bước tiếp theo

`Next: Milestone 4 — Feature engineering và preprocessing pipeline theo Experiment Specification v1.0`

M4 phải bắt đầu từ các quyết định đã khóa trong M3, không thiết kế lại split/evaluation từ đầu.

Mọi thay đổi đối với Experiment Specification v1.0 sau này chỉ được thực hiện khi có evidence mới và phải được ghi bằng Decision Log.