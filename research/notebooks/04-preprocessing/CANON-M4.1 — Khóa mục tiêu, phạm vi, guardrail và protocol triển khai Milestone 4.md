# M4.1 — Khóa mục tiêu, phạm vi, guardrail và protocol triển khai Milestone 4

## 1. Vai trò của M4.1

Milestone 3 đã hoàn thành vai trò thiết kế experiment protocol. Temporal evaluation, canonical split, metric strategy, quyền sử dụng TRAIN / VALIDATION / FINAL TEST, historical warm-up, training-window comparison protocol, class-imbalance policy, temporal-CV policy, tuning policy và test isolation đã được khóa trước khi chuyển sang feature/preprocessing/modeling.

Vì vậy M4 không được bắt đầu bằng việc thiết kế lại cách chia dữ liệu hoặc thay đổi luật đánh giá.

Handoff chính thức từ M3 xác định M4 phải tiếp nhận nguyên:

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

và chuyển trọng tâm sang feature engineering, preprocessing representation, baseline-ready modeling matrix và leakage-safe implementation.

M4.1 là bước đầu tiên của Milestone 4.

M4.1 chưa xây preprocessing pipeline cụ thể và chưa tạo feature matrix cuối.

Vai trò của M4.1 là:

```text
đọc toàn bộ handoff M1–M3
        ↓
phân biệt điều đã LOCKED với điều vẫn OPEN
        ↓
khóa phạm vi của Milestone 4
        ↓
khóa guardrail triển khai
        ↓
xác định sản phẩm đầu ra của M4
        ↓
chia M4 thành các công việc có ranh giới rõ
        ↓
xác định M4 Gate
```

M4.1 thuộc nhóm công việc `Concept / protocol / guardrail` kết hợp tổng hợp evidence đã có.

Không cần quét lại raw dataset chỉ để thực hiện M4.1.

---

# 2. Trạng thái đầu vào khi bước vào Milestone 4

Project hiện đã hoàn thành:

```text
M1
→ dataset selection + audit
→ DONE

M2
→ EDA + evidence
→ PASS WITH FINDINGS

M3
→ Experiment Specification v1.0
→ PASS
```

M3 xác nhận không còn protocol bắt buộc nào thiếu trước khi chuyển sang feature/preprocessing/modeling.

Tuy nhiên một số quyết định vẫn cố ý để mở vì chưa có implementation hoặc experiment thực tế.

Các câu hỏi được handoff sang M4/modeling gồm:

```text
O01 — feature set
O02 — behavioral feature set
O03 — exact historical lookback
O04 — MCC/location encoding
O05 — Amount transformation
O06 — training-window winner
O07 — final model family
O08 — final imbalance strategy
O09 — final hyperparameters
O10 — final numerical threshold
...
```

Trong đó O01–O05 liên quan trực tiếp đến feature/preprocessing implementation; O06 trở đi phần lớn cần model experiment hoặc validation experiment sau khi M4 đã tạo được pipeline thích hợp.

---

# 3. Mục tiêu chính thức của Milestone 4

Milestone 4 có mục tiêu:

> **Xây dựng một quy trình representation, feature engineering và preprocessing có thể tái hiện, tuân thủ prediction point và temporal boundary đã khóa, tạo được đầu vào Machine Learning sẵn sàng cho baseline mà không làm rò rỉ thông tin từ tương lai hoặc từ các partition không được phép.**

M4 không chỉ có mục tiêu:

```text
"làm dữ liệu sạch"
```

M4 phải đồng thời giải quyết bốn vấn đề.

Thứ nhất là `representation`:

```text
raw transaction
→ representation nhất quán
```

để Timestamp, Amount, categorical feature, target và các key lịch sử có ý nghĩa rõ ràng cho pipeline.

Thứ hai là `feature engineering`:

```text
raw/current-transaction information
+
strict-causal historical information
→ model feature candidates
```

Thứ ba là `preprocessing`:

```text
feature candidate
→ representation mà classifier có thể nhận
```

nhưng mọi learned preprocessing state phải tôn trọng partition permissions.

Thứ tư là `implementation safety`:

```text
feature/preprocessing logic
→ reproducible
→ auditable
→ deterministic khi có thể
→ leakage-safe
→ baseline-ready
```

---

# 4. Prediction point tiếp tục giữ nguyên

Problem definition không thay đổi trong M4.

Bài toán là:

```text
binary fraud-risk screening classification
```

Positive class:

```text
fraud
```

Prediction point:

```text
thời điểm transaction cần được screening
```

Mọi feature phải:

```text
1. tồn tại tại prediction point

HOẶC

2. được tính hoàn toàn từ thông tin xảy ra
   trước prediction point.
```

Future information hoặc thông tin chỉ xuất hiện sau khi transaction đã được xử lý không được phép đi vào feature pipeline.

Quyết định:

```text
M4.1-D01
Prediction point và problem definition:
INHERITED — LOCKED
```

---

# 5. Temporal split M4 phải kế thừa

M4 không được tự tạo split mới chỉ để thuận tiện cho preprocessing.

Canonical boundary hiện tại:

```text
TRAIN:

Timestamp < 2019-01-01

training start:
phụ thuộc candidate W_LONG / W_SHORT


VALIDATION:

2019-01-01
≤ Timestamp
< 2019-06-01


FINAL TEST:

2019-06-01
≤ Timestamp
< 2019-11-01
```

Split `S2_5M_5M` đã được khóa.

Hai classifier-training window candidate chính thức là:

```text
W_LONG_2015_TO_2018

2015-01-01
≤ Timestamp
< 2019-01-01
```

và:

```text
W_SHORT_2018_ONLY

2018-01-01
≤ Timestamp
< 2019-01-01
```

Cả hai đã được audit là khả thi nhưng chưa có model-performance evidence để chọn winner.

Do đó:

```text
M4.1-D02

Canonical split:
INHERITED — LOCKED

Training-window candidates:
W_LONG / W_SHORT
INHERITED — LOCKED

Training-window winner:
OPEN / REQUIRES EXPERIMENT
```

M4 không được chọn W_LONG hoặc W_SHORT chỉ dựa vào việc window nào dễ xử lý hơn về mặt code.

---

# 6. Ba loại state M4 bắt buộc phải phân biệt

Đây là một trong những guardrail quan trọng nhất khi triển khai pipeline.

## 6.1. Model-learning state

Đây là những thứ classifier học trong quá trình `fit()`.

Ví dụ:

```text
coefficient
tree structure
split rule
model parameter
learned classifier state
```

Nguồn học:

```text
TRAIN only
```

## 6.2. Preprocessing-learning state

Đây là statistic hoặc mapping được học từ dữ liệu.

Ví dụ:

```text
scaler mean/std
imputer statistic
category vocabulary
encoder mapping
learned transformation
data-driven feature-selection state
```

Nguồn học hợp lệ:

```text
TRAIN only
```

VALIDATION và FINAL TEST không được tham gia việc fit những state này.

## 6.3. Causal historical state

Behavioral feature có thể sử dụng lịch sử thực sự xảy ra trước transaction hiện tại.

Ví dụ:

```text
previous transaction
transactions in previous 1 hour
previous amount history
merchant-seen-before state
```

Đối với transaction T:

```text
Timestamp(history) < Timestamp(T)
```

Historical context có thể bao gồm transaction nằm trước classifier-training-window start nếu transaction đó thực sự xảy ra trước prediction point.

Điều này không biến historical context thành classifier-training data.

Quyết định:

```text
M4.1-D03

Model-learning state:
TRAIN ONLY

Preprocessing-learning state:
TRAIN ONLY

Causal historical state:
có thể sử dụng valid prior history
theo strict temporal rule

Status:
INHERITED — LOCKED
```

---

# 7. Historical feature rule

Mọi behavioral feature của M4 phải tuân thủ:

```text
Timestamp(history) < Timestamp(current transaction)
```

Không phải:

```text
Timestamp(history) <= Timestamp(current transaction)
```

Không được dùng:

```text
current transaction
như history của chính nó

future transaction

transaction cùng Timestamp

lifetime aggregate chứa future information

target label của prior/future transaction
làm historical feature state
```

M3 đã khóa rõ rằng historical warm-up được phép, nhưng classifier-training membership độc lập với history availability.

Ví dụ nếu W_SHORT chỉ train classifier bằng 2018:

```text
classifier-training rows
→ chỉ 2018
```

nhưng behavioral feature của transaction đầu 2018 vẫn có thể sử dụng valid transaction trước 2018 để tạo historical context.

Không được reset history tại:

```text
2018-01-01
```

chỉ vì classifier window bắt đầu tại đó.

Quyết định:

```text
M4.1-D04

Historical warm-up:
AUTHORIZED

Strict causal rule:
MANDATORY

Same-timestamp history:
PROHIBITED

Target-label history:
PROHIBITED

Status:
INHERITED — LOCKED
```

---

# 8. Feature guardrail M4 phải giữ nguyên

Các guardrail từ M1–M3 không được xem lại chỉ vì bước sang preprocessing.

## 8.1. Raw User

```text
DO NOT USE AS CLASSIFIER FEATURE
```

Được phép dùng làm:

```text
identifier
grouping key
history key
```

## 8.2. Raw Card

```text
DO NOT USE AS CLASSIFIER FEATURE
```

Được phép dùng cùng User để xác định entity/history.

## 8.3. Raw Merchant Name

```text
DO NOT USE AS CLASSIFIER FEATURE
```

Có thể được sử dụng như historical/grouping key nếu feature engineering cụ thể đảm bảo prediction-time validity và không memorization bằng raw ID.

## 8.4. Errors?

```text
EXCLUDED FROM MODEL V1
```

trừ khi sau này xuất hiện evidence mới chứng minh field tồn tại tại prediction point.

## 8.5. Merchant State / Zip missing

```text
DO NOT DROP/FILL MECHANICALLY
```

M2 đã xác nhận location missing mang tính cấu trúc; Online Transaction có State/Zip missing 100%.

## 8.6. Negative Amount

```text
DO NOT:

abs()
drop
clip
```

một cách máy móc.

M2 xác nhận negative Amount chiếm tỷ lệ đáng kể và chưa có căn cứ coi đây là vài lỗi dữ liệu ngẫu nhiên.

## 8.7. Full-history association

Một feature/category có fraud association mạnh trên toàn lịch sử:

```text
≠ bằng chứng feature đó generalize tốt
```

MCC/location đặc biệt phải được kiểm soát vì tồn tại temporal instability và synthetic-shortcut/memorization risk.

Quyết định:

```text
M4.1-D05

Feature guardrails từ M1–M3:
INHERITED — LOCKED

M4 được phép nghiên cứu representation cụ thể,
nhưng không được đảo ngược các guardrail này
nếu không có evidence mới và Decision Log riêng.
```

---

# 9. Quyền sử dụng TRAIN trong M4

TRAIN là partition duy nhất được phép cung cấp dữ liệu cho learned preprocessing state.

TRAIN có thể được dùng để:

```text
fit imputer
fit scaler
học category vocabulary
fit encoder
fit learned transformation
fit data-driven feature-selection mechanism
```

Sau này TRAIN cũng là nơi classifier được fit và nơi các kỹ thuật thay đổi training distribution được thực hiện.

Không được:

```text
TRAIN + VALIDATION + FINAL TEST
        ↓
fit preprocessing
        ↓
chia lại
```

Không được học:

```text
category vocabulary
scaler statistics
imputer statistics
encoder mapping
```

từ full dataset rồi mới quay lại tạo train/validation.

Quyết định:

```text
M4.1-D06

Learned preprocessing fit source:
TRAIN ONLY

Status:
INHERITED — LOCKED
```

---

# 10. Quyền sử dụng VALIDATION trong M4

VALIDATION là development-decision partition.

VALIDATION được phép:

```text
nhận transformation
từ preprocessing đã fit bằng TRAIN

kiểm tra schema/compatibility

nhận feature representation

sau này nhận prediction/probability

tính development metrics
khi model experiment bắt đầu
```

VALIDATION không được tham gia vào:

```text
fit scaler
fit imputer
học vocabulary
fit encoder
fit classifier đang được đánh giá
```

M3 định nghĩa VALIDATION là nơi trả lời:

> Trong các lựa chọn đã fit từ TRAIN, lựa chọn nào nên được mang tiếp?

Trong core M4, VALIDATION chủ yếu được dùng để kiểm tra pipeline có transform future data đúng hay không, ví dụ:

```text
unknown category có xử lý được không
schema có nhất quán không
feature generation có chạy đúng không
có NaN/inf ngoài dự kiến không
historical state có đúng causal không
```

Nếu cần sử dụng model performance trên VALIDATION để chọn giữa các feature/preprocessing candidate thì công việc đó phải được ghi nhận là:

```text
model-dependent feature/preprocessing experiment
```

không được âm thầm xem là preprocessing implementation đơn thuần.

Quyết định:

```text
M4.1-D07

VALIDATION:
development partition

Core M4:
transform / integrity / compatibility checks allowed

Performance-based selection:
requires explicit model experiment

Status:
LOCKED / READY TO APPLY
```

---

# 11. FINAL TEST tuyệt đối được bảo vệ trong M4

FINAL TEST là protected evaluation partition.

Nó không được sử dụng để:

```text
chọn feature
bỏ feature
thêm feature
chọn encoding
chọn preprocessing
chọn W_LONG/W_SHORT
chọn model
chọn class weight
chọn resampling
chọn hyperparameter
chọn threshold
```



Do đó trong M4:

```text
FINAL TEST labels:
DO NOT USE

FINAL TEST metrics:
DO NOT COMPUTE

FINAL TEST performance:
DO NOT INSPECT FOR DEVELOPMENT

FINAL TEST error analysis:
DO NOT USE FOR FEATURE/PREPROCESSING DECISIONS
```

Core M4 không cần mở FINAL TEST để chứng minh feature pipeline đúng.

Quyết định:

```text
M4.1-D08

FINAL TEST consumption during core M4:
PROHIBITED

Status:
INHERITED — LOCKED
```

---

# 12. Các câu hỏi M4 trực tiếp nhận từ M3

M4 nhận năm nhóm câu hỏi chính.

## O01 — Baseline/model-candidate feature set

M3 để `OPEN`.

M4 phải quyết định feature nào đủ điều kiện đi vào một baseline/model-candidate feature version.

Không được chọn chỉ dựa trên full-history EDA association.

M4 không nhất thiết chứng minh feature version này là “final feature set tốt nhất”.

Sản phẩm hợp lý của M4 là:

```text
Baseline Feature Set v1.0
```

hoặc:

```text
Model Candidate Feature Set v1.0
```

đủ để bắt đầu controlled model experiment.

---

## O02 — Behavioral feature set

Các candidate đã có evidence feasibility gồm ít nhất:

```text
time_since_previous_transaction

transactions_last_1h

amount_minus_previous_mean

is_new_merchant
```

M2 mới chứng minh:

```text
causal feasibility
+
coverage
+
computability
```

chưa chứng minh:

```text
classification benefit
```



M4 có nhiệm vụ triển khai candidate hợp lệ.

Việc feature có cải thiện F1 hay không cần model evidence phía sau.

---

## O03 — Exact historical lookback

Historical warm-up đã được khóa.

Nhưng M3 chưa chọn:

```text
full prior history

fixed years

fixed rolling horizon

strategy khác
```



Đây là câu hỏi data/implementation-dependent của M4.

Không được khóa chỉ từ lý thuyết nếu runtime, coverage hoặc representation thực tế có ảnh hưởng tới quyết định.

---

## O04 — MCC/location representation

Trạng thái:

```text
OPEN
```

M4 phải xem xét đồng thời:

```text
cardinality
structural missingness
temporal stability
unseen category
memorization risk
synthetic-shortcut risk
pipeline cost
```

M3 không khóa encoding cuối.

---

## O05 — Amount representation

Trạng thái:

```text
OPEN
```

Negative Amount không được xử lý máy móc.

Transformation cuối phải dựa trên preprocessing reasoning và evidence triển khai.

---

# 13. Những quyết định KHÔNG thuộc core M4

M4 không được cố khóa trước các vấn đề sau chỉ để “hoàn thiện pipeline”.

```text
Training-window winner
→ REQUIRES EXPERIMENT

Final model family
→ REQUIRES EXPERIMENT

Final imbalance strategy
→ REQUIRES EXPERIMENT

Final hyperparameters
→ REQUIRES EXPERIMENT

Final numerical threshold
→ REQUIRES VALIDATION EXPERIMENT

Probability calibration
→ OPTIONAL / DEFERRED

Cold-start group evaluation
→ OPTIONAL / DEFERRED

Post-break zero-fraud diagnostic
→ OPTIONAL / DEFERRED
```



Quyết định:

```text
M4.1-D09

M4 không được mở rộng scope thành
full model-selection milestone.

Status:
READY TO LOCK
```

---

# 14. Sản phẩm bắt buộc của Milestone 4

M4 phải hướng tới ít nhất các sản phẩm sau.

```text
1. Representation Specification

2. Feature Specification / Feature Dictionary

3. Behavioral Feature Contract

4. Preprocessing Specification

5. Leakage-safe preprocessing implementation

6. Baseline Feature Set v1.0

7. Baseline-ready TRAIN representation

8. Baseline-ready VALIDATION representation

9. Integrity / Leakage Audit

10. Decision Log

11. Open Questions / Handoff

12. M4 Gate
```

M4 không bắt buộc phải tạo final classifier performance.

---

# 15. Cấu trúc Milestone 4

Từ handoff M2–M3, M4 được chia thành tám công việc.

## M4.1 — Khóa mục tiêu, phạm vi, guardrail và protocol

Câu hỏi:

> M4 được phép quyết định những gì, phải kế thừa những gì và phải tạo đầu ra nào?

Đầu ra:

```text
M4 Scope
+
M4 Guardrails
+
M4 Work Plan
+
M4 Gate Definition
```

---

## M4.2 — Xây representation cơ sở và audit input

Câu hỏi:

> Raw transaction phải được biểu diễn thế nào để những bước feature/preprocessing sau cùng làm việc trên một representation thống nhất?

Trọng tâm dự kiến:

```text
raw schema verification
Timestamp reconstruction
Amount numeric representation
target representation
semantic type
partition membership
identifier/history keys
basic invariants
```

M4.2 chưa quyết định final encoding.

---

## M4.3 — Khóa policy xử lý data-quality trong preprocessing

Câu hỏi:

> Những finding về Amount, missing location, duplicate và các giá trị đặc biệt phải được chuyển thành preprocessing policy như thế nào?

Trọng tâm:

```text
negative Amount
zero Amount
location missingness
online transaction semantics
duplicate policy
category consistency
other representation anomalies
```

Mỗi quyết định data-dependent phải dựa trên output thật.

---

## M4.4 — Thiết kế transaction-level feature representation

Câu hỏi:

> Từ thông tin của transaction hiện tại, những feature candidate nào hợp lệ tại prediction point và biểu diễn chúng thế nào?

Có thể bao gồm các nhóm:

```text
Amount-related
transaction mode
MCC
location
calendar/time
derived current-transaction features
```

Raw identifiers vẫn bị cấm.

---

## M4.5 — Triển khai causal behavioral feature

Câu hỏi:

> Những historical feature nào có thể được triển khai ổn định, causal và đủ hiệu quả tính toán để đưa vào baseline candidate?

Trọng tâm:

```text
history ordering
same-timestamp handling
warm-up
exact lookback
User/Card history
merchant-history state
prototype → reusable implementation
```

M2 đã xác nhận behavioral feature engineering khả thi và yêu cầu M4 có order assertion hoặc explicit sort.

---

## M4.6 — Xây leakage-safe preprocessing pipeline

Câu hỏi:

> Feature candidate được chuyển thành modeling matrix như thế nào mà learned state chỉ học từ TRAIN?

Trọng tâm:

```text
numeric branch
categorical branch
imputation nếu có
encoding
scaling nếu cần
unknown category handling
feature names/order
fit / transform boundary
```

---

## M4.7 — Tạo baseline-ready modeling matrix và audit pipeline

Câu hỏi:

> Pipeline hiện tại đã đủ đúng và đủ ổn định để M5 chỉ cần gắn baseline classifier vào hay chưa?

Đầu ra trọng tâm:

```text
X_train
y_train

X_validation
y_validation

feature metadata
preprocessing metadata
integrity checks
leakage checks
runtime/memory notes
reproducibility checks
```

M4.7 không sử dụng FINAL TEST performance.

---

## M4.8 — Tổng hợp M4, Decision Log và M4 Gate

Câu hỏi:

> Feature/preprocessing implementation đã đủ điều kiện chuyển sang baseline/model experiments chưa?

Đầu ra:

```text
Feature Specification v1.0
Preprocessing Specification v1.0
Decision Log
Open Questions
M4 Gate
Handoff to modeling
```

---

# 16. Quy trình thực nghiệm cho M4.2 → M4.7

Không phải mọi quyết định M4 đều có thể khóa từ lý thuyết.

Đối với các câu hỏi phụ thuộc trạng thái project thật, quy trình là:

```text
Câu hỏi
        ↓
Evidence đã có
        ↓
Thiết kế phép kiểm tra
        ↓
Code + assertion + gate
        ↓
Người thực hiện Run All
        ↓
Execution check
        ↓
Integrity check
        ↓
Consistency check
        ↓
Logic / leakage check
        ↓
Observed fact
        ↓
Interpretation
        ↓
Decision
```

Một data-dependent decision chỉ được `LOCKED` khi có output thực tế.

Nếu chưa đủ evidence:

```text
OPEN
```

hoặc:

```text
REQUIRES EXPERIMENT
```

Không viết kết luận trước rồi tìm output khớp với kết luận.

---

# 17. Nguyên tắc code của Milestone 4

Code M4 phải ưu tiên:

```text
dễ đọc

dễ giải thích khi vấn đáp

chạy độc lập

không phụ thuộc hidden notebook state

deterministic khi có thể

không thay đổi raw artifact

fail loudly khi invariant bị phá

có assertion / consistency check

chunk / stream khi dataset size yêu cầu

không tối ưu quá sớm làm code khó hiểu
```

Đặc biệt với behavioral feature:

```text
physical row order
```

không được mặc định là chronological order nếu chưa có assertion.

Feature implementation phải:

```text
explicit sort
```

hoặc:

```text
order assertion
```

phù hợp với semantics của phép tính.

---

# 18. Versioning trong M4

M4 bắt đầu quản lý version rõ ràng cho representation và feature/preprocessing.

Đề xuất:

```text
Representation Version:
REP_V1

Feature Version:
FEAT_V1

Preprocessing Version:
PREP_V1

Pipeline Version:
PIPE_V1
```

Tên cụ thể có thể thay đổi khi implementation hình thành.

Mỗi experiment quan trọng sau này phải có khả năng ghi lại:

```text
input artifact
training window
feature version
preprocessing version
model/configuration
random seed
metric
threshold nếu có
output
decision
```

Mục đích là tránh tình trạng:

```text
"model lần trước dùng những feature nào?"
```

nhưng không còn truy nguyên được.

---

# 19. M4 Gate — định nghĩa trước khi thực hiện

M4 chỉ được coi là hoàn thành nếu đáp ứng tối thiểu các gate sau.

## G01 — Protocol inheritance

```text
PASS khi:
M4 không thay đổi trái phép
temporal split / partition rights /
test isolation / causal rule.
```

## G02 — Prediction-time validity

```text
PASS khi:
mọi classifier feature tồn tại tại prediction point
hoặc chỉ được tính từ prior information.
```

## G03 — Raw identifier protection

```text
PASS khi:
raw User / Card / Merchant Name
không xuất hiện trong classifier feature matrix.
```

## G04 — Errors? protection

```text
PASS khi:
Errors? không xuất hiện trong Model V1 feature matrix.
```

## G05 — Causal-history integrity

```text
PASS khi:
future-history violation = 0

same-timestamp-as-history violation = 0

target-label history = 0
```

## G06 — Preprocessing boundary

```text
PASS khi:
mọi learned preprocessing state
được fit từ TRAIN only.
```

## G07 — FINAL TEST isolation

```text
PASS khi:
FINAL TEST performance chưa tham gia
bất kỳ quyết định M4 nào.
```

## G08 — Data-quality decisions

```text
PASS khi:
negative Amount và location missing
không bị xử lý máy móc trái guardrail.
```

## G09 — Baseline feature contract

```text
PASS khi:
có một Baseline Feature Set v1.0
được mô tả đủ rõ và tái hiện được.
```

## G10 — TRAIN/VALIDATION compatibility

```text
PASS khi:
TRAIN và VALIDATION được transform
bằng cùng feature/preprocessing contract

và pipeline xử lý được future categories/
missing representation hợp lệ.
```

## G11 — Reproducibility

```text
PASS khi:
pipeline chạy độc lập,
không phụ thuộc notebook hidden state,
và version/metadata quan trọng được ghi nhận.
```

## G12 — Baseline readiness

```text
PASS khi:
M5 có thể nhận output của M4
và fit baseline model
mà không phải tự phát minh lại
feature/preprocessing logic.
```

---

# 20. Những điều M4 Gate không yêu cầu

M4 Gate không yêu cầu:

```text
F1 phải đạt một ngưỡng cụ thể

Recall phải cao

Precision phải cao

Random Forest phải thắng

W_LONG/W_SHORT phải có winner

class_weight phải thắng baseline

SMOTE phải được sử dụng

threshold cuối phải được khóa

final-test performance phải được biết
```

Những điều trên thuộc model-dependent hoặc validation/final-test experiment.

M3 đã ghi rõ chưa có bằng chứng để kết luận behavioral feature cải thiện F1, class_weight cải thiện model, SMOTE cần thiết, model family nào tốt nhất hoặc threshold nào tốt nhất.

---

# 21. Decision Log M4.1

```text
M4.1-D01
Prediction point và problem definition
→ INHERITED — LOCKED

M4.1-D02
Canonical split và W_LONG/W_SHORT candidates
→ INHERITED — LOCKED

M4.1-D03
Model-learning/preprocessing-learning state
chỉ được học từ TRAIN
→ INHERITED — LOCKED

M4.1-D04
Historical warm-up được phép,
strict causal rule bắt buộc
→ INHERITED — LOCKED

M4.1-D05
Feature-level guardrails M1–M3 tiếp tục có hiệu lực
→ INHERITED — LOCKED

M4.1-D06
Learned preprocessing fit source = TRAIN only
→ INHERITED — LOCKED

M4.1-D07
VALIDATION là development partition;
performance-based feature/preprocessing selection
phải được ghi nhận như model experiment
→ READY TO LOCK

M4.1-D08
FINAL TEST không được sử dụng trong core M4 development
→ INHERITED — LOCKED

M4.1-D09
M4 không mở rộng thành final model-selection milestone
→ READY TO LOCK

M4.1-D10
M4 được tổ chức thành M4.1 → M4.8
theo cấu trúc của tài liệu này
→ READY TO LOCK

M4.1-D11
Output trung tâm của M4 là
baseline-ready, leakage-safe feature/preprocessing pipeline
→ READY TO LOCK

M4.1-D12
M4 Gate gồm G01 → G12
→ READY TO LOCK
```

---

# 22. M4.1 Gate

## G01 — Có xác định rõ M4 kế thừa gì từ M3 không?

```text
PASS
```

Temporal split, partition rights, causal rules, feature guardrails, training-window candidates và test isolation đã được xác định.

## G02 — Có xác định rõ phạm vi M4 không?

```text
PASS
```

M4 tập trung vào:

```text
representation
feature engineering
preprocessing
baseline-ready matrix
leakage-safe implementation
```

## G03 — Có phân biệt preprocessing state và historical state không?

```text
PASS
```

Learned preprocessing state:

```text
TRAIN only
```

Historical state:

```text
valid prior history
theo Timestamp(history) < Timestamp(current)
```

## G04 — Có xác định FINAL TEST protection không?

```text
PASS
```

FINAL TEST không tham gia feature/preprocessing development.

## G05 — Có xác định những câu hỏi vẫn OPEN không?

```text
PASS
```

Feature set, behavioral feature, historical lookback, MCC/location representation và Amount transformation vẫn cần implementation/evidence.

## G06 — Có tránh premature model decision không?

```text
PASS
```

M4 không khóa trước:

```text
training-window winner
final model
imbalance winner
hyperparameter
threshold
final performance
```

## G07 — Có kế hoạch thực thi cụ thể sau M4.1 không?

```text
PASS
```

Đã xác định M4.2 → M4.8.

## G08 — Có M4 Gate trước khi bắt đầu implementation không?

```text
PASS
```

G01 → G12 đã được định nghĩa để tránh việc cuối M4 mới thay đổi tiêu chuẩn hoàn thành.

---

# 23. Kết luận M4.1

M4.1 không tạo thêm data-dependent finding mới.

Bước này đã chuyển Experiment Specification v1.0 của M3 thành một protocol triển khai cụ thể cho feature engineering và preprocessing.

Trạng thái đề xuất:

```text
M4.1 Scope:
DEFINED

M4 Guardrails:
DEFINED

Inherited Experiment Protocol:
PRESERVED

M4 Work Plan:
DEFINED — M4.1 → M4.8

M4 Gate:
DEFINED

New data experiment:
NOT REQUIRED FOR M4.1

Blocking issue:
NONE

M4.1 Status:
READY TO LOCK
```

Sau khi M4.1 được khóa:

```text
Next:

M4.2
— Xây representation cơ sở
và audit input cho preprocessing/feature pipeline
```

M4.2 sẽ là bước đầu tiên quay lại project/data thật và do đó phải tuân thủ quy trình:

```text
câu hỏi trước
→ thiết kế check
→ code
→ người thực hiện chạy
→ kiểm tra output
→ kết luận sau
```