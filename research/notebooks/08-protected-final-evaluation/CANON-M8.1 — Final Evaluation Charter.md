# CANON-M8.1 — Final Evaluation Charter

## Document status

**Project:** AI Transaction Fraud Risk Screening  
**Tên đề tài:** Xây dựng chương trình đánh giá dấu hiệu gian lận của giao dịch tài chính bằng thuật toán học máy  
**Milestone:** `M8 — Protected Final Evaluation`  
**Substep:** `M8.1 — Final Evaluation Charter + Guardrails`  
**Work type:** `PROTOCOL / GOVERNANCE / TEST-ISOLATION RELEASE`  
**Runtime experiment mới:** `NOT REQUIRED`  
**FINAL TEST performance access trong M8.1:** `PROHIBITED`  
**Upstream entry:** `M7 — PASS / READY FOR PROTECTED FINAL EVALUATION`

M8.1 là bước khóa luật đánh giá cuối **trước khi** project thực hiện bất kỳ official FINAL TEST scoring nào.

M8.1 không nhằm tạo model mới, không nhằm cải thiện score và không nhằm audit runtime FINAL TEST. Những việc đó thuộc các substep sau.

---

# 1. Mục tiêu

M8.1 phải biến handoff sau M7 thành một **Final Evaluation Charter có thể thực thi và audit được**.

Câu hỏi trung tâm:

> Frozen development configuration sẽ được đánh giá trên protected FINAL TEST theo protocol nào để final performance phản ánh đúng một future holdout chưa từng tham gia vào development selection?

M8.1 phải khóa trước runtime:

```text
subject model identity
feature/preprocessing identity
FINAL TEST boundary
positive class
authorized model-state policy
threshold
metric contract
comparison policy
allowed diagnostics
forbidden optimization
persistence schema
STOP conditions
M8.2 release condition
```

Kết quả mong muốn:

```text
M8.1 Charter:
LOCKED

FINAL TEST performance:
STILL UNREAD

Official final scoring:
NOT YET AUTHORIZED

Next:
M8.2 — FINAL TEST artifact / lineage / representation audit
```

---

# 2. Căn cứ kế thừa

M8.1 không thiết kế lại project từ đầu.

Nguồn CANON chính:

```text
1. CANON-Handoff_M0-M7_sang_Giai_doan_hoan_thien_va_ket_thuc_du_an.md
2. CANON-Kế hoạch Giai đoạn hoàn thiện và kết thúc dự án.md
3. CANON-Quy ước cách làm việc giai đoạn hoàn thiện và kết thúc dự án.md
4. CANON-M3.5 — Khóa vai trò và quyền sử dụng TRAIN / VALIDATION / FINAL TEST.md
5. CANON-M3.8 — Experiment Specification v1.0 / M3 Gate.md
6. CANON-M4.8 — Feature / Preprocessing / Behavioral Contract / M4 Gate.md
7. CANON-M6.7 — Evaluation Registry / Error Findings / M6 Gate.md
8. CANON-M7.6 — Moderate hyperparameter tuning.md
9. CANON-M7.7 — Candidate selection + external VALIDATION confirmation.md
10. CANON-M7.8 — Numerical threshold selection.md
11. CANON-M7.9 — Final Selection Registry, Decision Log và M7 Gate.md
12. CANON-Quy trình phối hợp thực nghiệm giữa người thực hiện và AI trong dự án.md
```

Source hierarchy khi có xung đột:

```text
verified project/runtime artifact
        ↓
reviewed notebook output
        ↓
latest project-specific CANON / Decision Log / Gate
        ↓
M0–M7 handoff
        ↓
stage plan
        ↓
generic ML guidance
        ↓
intuition / convenience
```

M8.1 không được dùng generic best practice để tự mở lại một quyết định đã được project khóa bằng evidence hợp lệ.

---

# 3. Entry state từ M7

M7 handoff đã khóa:

```text
M7:
PASS

Development selection:
COMPLETE

Upstream configuration:
FROZEN

FINAL TEST:
UNTOUCHED / PROTECTED

Handoff:
READY FOR PROTECTED FINAL EVALUATION
```

M8.1 kế thừa trạng thái này như precondition.

Nếu M8.2 sau đó phát hiện artifact corruption hoặc contradiction thực tế, project phải dừng và audit nguyên nhân; M8.1 không cho phép âm thầm thay config để tiếp tục.

---

# 4. Vai trò chính thức của M8

M8 chỉ trả lời:

> Frozen development configuration hoạt động như thế nào trên future holdout đã được bảo vệ khỏi development selection?

M8 không tồn tại để trả lời:

```text
model nào tốt hơn nếu thử lại?
threshold nào đẹp hơn trên FINAL TEST?
feature nào nên thêm sau khi xem test?
hyperparameter nào cải thiện final score?
seed nào cho score tốt hơn?
refit TRAIN+VALIDATION có tăng F1 không?
```

Những câu hỏi trên thuộc development/post-evaluation research khác, không thuộc protected final evaluation hiện tại.

---

# 5. Official final-evaluation subject identity

M8 chỉ có **một official evaluation subject**.

Logical identity đã freeze:

```text
Training Window:
W_SHORT

Classifier-training boundary:
2018-01-01 <= Timestamp < 2019-01-01

Model Family:
Random Forest

Model Config ID:
RF-REF-100-GINI-SQRT-UNPRUNED-CW

Imbalance Strategy:
CLASS_WEIGHT_BALANCED

Random State:
42

Probability Interface:
predict_proba

Positive probability class:
1

Numerical Threshold:
0.50
```

Feature / preprocessing identity:

```text
M4 canonical feature/preprocessing contract

Pre-encoding features:
10

Encoded width:
47

Matrix:
CSR sparse

dtype:
float32

Behavioral rule:
strict-causal prior history

Learned preprocessing state:
TRAIN only
```

Status:

`FROZEN — NOT A CANDIDATE SET`

M8 không được tạo leaderboard mới giữa LR / DT / RF hoặc giữa W_LONG / W_SHORT.

---

# 6. Exact RF parameter identity

Config ID đã khóa:

`RF-REF-100-GINI-SQRT-UNPRUNED-CW`

M8.1 **không tái dựng exact parameter dictionary bằng trí nhớ hoặc suy đoán từ tên config**.

Exact runtime parameter identity phải được M8.2 xác minh từ:

```text
immutable M7.6 tuning registry
+
M7.6 reviewed notebook
+
M7.7 candidate identity assertions / manifest
```

Known frozen attributes tối thiểu từ upstream:

```text
n_estimators = 100
class_weight = balanced
random_state = 42
reference RF configuration retained
```

Các parameter khác phải đọc từ canonical artifact/config registry khi runtime cần.

Status:

`LOGICAL MODEL IDENTITY LOCKED / EXACT PARAM DICT REQUIRES ARTIFACT AUDIT`

---

# 7. Physical model artifact policy

M0–M7 evidence xác nhận logical selected model identity, config, validation prediction/risk-score artifacts và manifests.

M8.1 không được giả định không có bằng chứng rằng đã tồn tại một serialized fitted Random Forest object ở một path cụ thể.

Do đó physical estimator state bắt đầu M8 có trạng thái:

`OPEN — REQUIRES M8.2 ARTIFACT AUDIT`

M8.2 phải rẽ theo một trong hai nhánh.

## 7.1. Nhánh A — Serialized selected estimator tồn tại

Chỉ được sử dụng nếu M8.2 xác minh:

```text
artifact exists
model class matches RandomForestClassifier
config identity matches frozen M7 selection
training-window identity = W_SHORT
class_weight identity = balanced
random_state identity = 42
feature/preprocessing compatibility matches canonical contract
artifact fingerprint recorded
```

Nếu không đủ evidence:

`DO NOT USE AS OFFICIAL FINAL SUBJECT`

## 7.2. Nhánh B — Serialized selected estimator không tồn tại

Không được coi đây là lý do để:

```text
refit TRAIN + VALIDATION
change config
change seed
change features
change preprocessing
```

M8.2 phải thiết kế deterministic reconstruction của **cùng M7.7 subject** bằng:

```text
same W_SHORT training population
same canonical W_SHORT representation
same exact selected RF parameter dictionary
same class_weight
same random_state
same library/runtime contract as far as project can verify
```

Trước khi reconstruction đó được phép score FINAL TEST, phải thực hiện reproducibility check với persisted M7.7 external VALIDATION evidence.

Minimum acceptance requirement:

```text
same validation population
same positive class
same threshold policy
same prediction semantics
same risk-score semantics
```

Và M8.2 phải xác minh mức tái tạo của persisted M7.7 validation prediction/risk score.

Nếu exact reproduction không đạt hoặc không giải thích được:

`STOP — OFFICIAL FINAL SCORING NOT AUTHORIZED`

Quan trọng:

Deterministic reconstruction trên **cùng W_SHORT TRAIN** để tái tạo subject M7.7 không đồng nghĩa với authorized `TRAIN + VALIDATION refit`.

---

# 8. TRAIN + VALIDATION refit policy

Upstream M3 đã ghi:

```text
TRAIN + VALIDATION
→ refit final model
→ FINAL TEST
```

không được tự động cho phép.

M8.1 giữ trạng thái:

`NOT AUTHORIZED BY DEFAULT`

M8 không được train lại model trên VALIDATION chỉ vì development đã kết thúc.

Nếu một `deployment-refit` được muốn trong tương lai:

```text
phải có model ID khác
training population khác phải được ghi rõ
không được gán trực tiếp M8 final-test metric cho estimator mới
coi là post-evaluation extension / future work
```

Trong core project:

`EVALUATED SUBJECT = FROZEN M7 DEVELOPMENT SUBJECT`

---

# 9. FINAL TEST population contract

Canonical boundary:

```text
2019-06-01 <= Timestamp < 2019-11-01
```

Known upstream structural evidence:

```text
Rows:
722,955

Fraud:
1,035

Non-fraud:
721,920

Fraud rate:
≈ 0.143162%
```

Vai trò:

`PROTECTED FINAL EVALUATION ONLY`

Không được mở rộng core FINAL TEST sang:

```text
2019-11 → 2020-02
```

vì đây là post-break zero-fraud diagnostic region, không thuộc core final fraud-performance protocol.

---

# 10. Partition rights trong M8

TRAIN / temporal CV:

```text
historical development evidence
no new selection loop
```

External VALIDATION 2019-01 → 2019-05:

```text
historical development confirmation
historical threshold-selection source
allowed for reproduction/identity checks
allowed for descriptive comparison with final test
not reopened as a new tuning target
```

FINAL TEST 2019-06 → 2019-10:

```text
M8.2:
artifact / lineage / representation audit

M8.3:
official frozen inference

M8.4:
final metric reconstruction

M8.5:
error / temporal generalization interpretation
```

Không có quyền development selection mới.

---

# 11. Staged FINAL TEST release

Để giảm nguy cơ test leakage, M8 dùng staged release.

## Stage A — M8.1

```text
protocol only
no final-test performance
```

## Stage B — M8.2

```text
structural / artifact / lineage / representation audit
no model-performance interpretation
```

M8.2 được phép kiểm tra known expected target support để xác minh population identity, nhưng không được dùng target để thay đổi model/config/threshold/feature.

## Stage C — M8.3

```text
freeze subject identity
run official inference
persist risk_score + y_pred + lineage + metadata
```

Không được chọn config sau khi nhìn prediction distribution.

## Stage D — M8.4

```text
lock prediction artifacts
compute final metrics independently
read Confusion Matrix
```

## Stage E — M8.5

```text
interpret errors / temporal behavior
no model modification
```

M8.6 tổng hợp registry và Gate.

---

# 12. Positive-class contract

Raw target:

```text
Is Fraud?

Yes → 1 → fraud → positive
No  → 0 → non-fraud → negative
```

Probability extraction:

```text
predict_proba
→ locate estimator.classes_ == 1
→ use exactly that column as risk_score
```

Không được giả định cột probability thứ hai luôn là class 1 mà không kiểm tra `classes_`.

M8.2/M8.3 phải assert positive-class identity trước official scoring.

---

# 13. Risk-score language

Selected RF có `predict_proba`, nhưng project chưa thực hiện probability calibration audit/tuning.

Được dùng:

```text
risk score
model score
positive-class score
```

Không được dùng như claim chính thức:

```text
calibrated confidence
real-world fraud probability
"x% chắc chắn là fraud"
```

M8 không mở calibration tuning.

---

# 14. Numerical threshold contract

Final numerical threshold:

`0.50 — FROZEN`

M7.8 historical candidate set:

```text
0.20
0.30
0.40
0.50
0.60
```

M8 không được chạy lại candidate set trên FINAL TEST.

Official final prediction phải dùng comparator semantics tương thích đúng với M7.8/M7.7 reconstruction.

M8.2/M8.3 phải xác minh comparator implementation thay vì tự đổi `>` / `>=` theo convenience.

Nếu comparator identity không tái tạo M7.7 threshold=0.50 behavior:

`STOP`

---

# 15. Metric contract

Primary metric:

`F1_fraud`

Mandatory secondary:

```text
Recall_fraud
Precision_fraud
```

Mandatory raw counts:

```text
TP
FP
FN
TN
```

Mandatory operational diagnostics:

```text
predicted_positive_count
predicted_positive_rate
```

Accuracy:

`REFERENCE ONLY`

Reason:

Fraud class rất hiếm nên Accuracy có thể rất cao dù model bỏ sót phần lớn fraud.

M8 không được đổi primary metric sau khi xem final result.

---

# 16. Metric implementation policy

M8.4 phải tái tính metric từ persisted official artifacts, không chỉ copy số từ M8.3 output.

Canonical semantics:

```text
labels = [0, 1]
positive class = 1
zero_division = 0

TP + FN = actual fraud
TN + FP = actual non-fraud
predicted_positive_count = TP + FP
predicted_positive_rate = predicted_positive_count / N
```

M8.4 phải có arithmetic consistency assertions.

M8.6 chỉ chấp nhận final metric profile nếu independent reconstruction khớp official prediction artifacts.

---

# 17. Advanced metrics policy

ROC-AUC / PR-AUC hoặc ranking metrics khác không thuộc core M8 metric contract đã kế thừa.

M8.1 disposition:

`NOT REQUIRED / DO NOT ADD AFTER SEEING FINAL RESULT`

Nếu project sau này muốn phân tích ranking metric vì lý do học thuật riêng, phải ghi rõ:

```text
SUPPLEMENTARY DIAGNOSTIC
NOT A RETROACTIVE SELECTION CRITERION
```

Core M8 Gate không phụ thuộc chúng.

---

# 18. Allowed diagnostics

Sau khi official predictions đã được persist và frozen, M8 được phép thực hiện descriptive diagnostics để **hiểu** final behavior.

Allowed core diagnostics:

```text
Confusion Matrix
FP / FN counts
fraud capture / fraud miss interpretation
alert burden
risk-score descriptive summaries
descriptive comparison with external VALIDATION
monthly / temporal descriptive slices inside FINAL TEST
error-group analysis using canonical semantic features
known temporal-generalization review
```

Allowed purpose:

```text
explain
characterize
report limitation
support viva understanding
```

Không được biến diagnostic thành selection criterion mới.

---

# 19. Validation-vs-final comparison policy

External VALIDATION result của selected RF là historical development evidence.

Known validation profile:

```text
Rows:
712,458

Fraud:
1,052

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

M8 được phép so final-test metrics với validation theo kiểu:

```text
absolute values
absolute deltas
relative descriptive direction when useful
alert-rate difference
temporal degradation / improvement description
```

Nhưng comparison chỉ có vai trò:

`DESCRIPTIVE GENERALIZATION EVIDENCE`

Không được kết luận:

```text
"FINAL TEST cho thấy threshold khác tốt hơn nên đổi"
"FINAL TEST cho thấy DT có thể tốt hơn nên chạy lại"
```

---

# 20. Temporal generalization finding

M7 handoff mang theo:

`TEMPORAL GENERALIZATION FINDING — OPEN LIMITATION`

M8 phải xem đây là câu hỏi diễn giải, không phải tuning trigger.

M8.5 được phép hỏi:

```text
final-test behavior lệch validation theo hướng nào?
metric có suy giảm hay thay đổi đáng kể theo thời gian không?
FP/FN có temporal concentration đáng chú ý không?
monthly performance có heterogeneity không?
```

Nếu final behavior kém:

```text
ghi nhận
→ giải thích
→ limitation
→ future work
```

Không:

```text
retune
→ rerun FINAL TEST
```

---

# 21. Evaluation-population limitation

Upstream audit cho thấy main future evaluation chủ yếu là existing User/Card population với historical context.

Do đó M8 final performance chủ yếu hỗ trợ claim:

> Model hoạt động như thế nào trên future transactions của population phần lớn đã có historical context trong synthetic IBM dataset.

Không đủ căn cứ để tuyên bố mạnh:

`completely-new User/Card generalization`

Cold-start limitation phải được carry sang M8.6, M13 và M14.

---

# 22. Feature contract

M8 không feature-engineer lại.

Core 10 pre-encoding features:

```text
Numeric:
amount_numeric
time_since_previous_transaction_min
transactions_last_1h
amount_minus_previous_mean

Boolean:
is_new_merchant
has_prior_card_history

Categorical:
transaction_mode
location_state
hour_of_day
day_of_week
```

Không tự thêm:

```text
MCC
Merchant City
raw Zip
raw User
raw Card
raw Merchant Name
Errors?
new Amount transform
new calendar feature
```

Feature semantics = `M4 CANONICAL / FROZEN`.

---

# 23. Preprocessing contract

Numeric branch:

```text
StandardScaler
learned statistics = TRAIN only
```

Structural NA:

```text
NA không tham gia scaler statistics
scaled NA → standardized 0.0
has_prior_card_history giữ cold-start semantic
```

Boolean:

```text
passthrough
float32
```

Categorical:

```text
OneHotEncoder
vocabulary = TRAIN only
unknown → __UNKNOWN__
```

Output:

```text
47 columns
CSR sparse
float32
canonical feature names/order
```

M8.2 phải xác minh exact learned preprocessing identity trước scoring.

Không fit vocabulary/scaler trên FINAL TEST.

---

# 24. Learned state vs causal transaction-history state

M8 tiếp tục distinction đã khóa.

```text
LEARNED STATE
scaler / encoder / model parameters
→ học từ TRAIN
→ frozen trong FINAL TEST
```

```text
CAUSAL TRANSACTION-HISTORY STATE
prior observable transaction events
→ được tiến theo chronological stream
→ không dùng label
```

Strict rule:

`Timestamp(history) < Timestamp(current)`

Prohibited:

```text
future transaction
same-timestamp peer as history
current transaction as its own history
target-label history
future label
```

Việc causal history state tiến qua thời gian không được diễn giải là refit model trên FINAL TEST.

---

# 25. FINAL TEST representation policy

M4.7 persisted development matrices cho TRAIN/VALIDATION; FINAL TEST trước M8 chưa được materialize như development input.

M8.2 phải audit pipeline:

```text
raw chronological stream
→ canonical base representation
→ strict-causal behavioral state
→ frozen M4 feature semantics
→ frozen W_SHORT preprocessing state
→ 47-column CSR float32 FINAL TEST representation
```

M8.2 phải xác minh ít nhất:

```text
row count
row lineage
temporal boundary
feature width
feature names/order
sparse format
dtype
finite matrix values
unknown-category handling
strict-causal history invariants
no learned-state fit on FINAL TEST
```

Chỉ sau khi audit PASS mới được M8.3 score.

---

# 26. Test-isolation rules

M8.1 khóa:

```text
FINAL TEST performance không được tham gia bất kỳ development decision nào.
```

Sau khi FINAL TEST performance được nhìn thấy, các decision M7 vẫn historical/frozen.

Không được:

```text
change model family
change training window
change hyperparameters
change class_weight
change sampling strategy
change preprocessing
add/drop feature
change threshold
change probability calibration
change random seed
change primary metric
```

Nếu final score không như kỳ vọng:

```text
record
analyze
explain
report limitation
future work
```

---

# 27. Forbidden optimization — explicit list

M8 cấm:

```text
FINAL TEST threshold search
FINAL TEST hyperparameter search
FINAL TEST model-family comparison
FINAL TEST feature ablation for selection
FINAL TEST class-weight search
FINAL TEST resampling experiment
FINAL TEST seed search
FINAL TEST calibration tuning
FINAL TEST preprocessing refit
TRAIN+VALIDATION refit without prior authorized protocol
adaptive grid expansion after seeing final result
rerunning alternative candidate to replace selected RF
changing metric because final F1 is disappointing
```

Nếu một ý tưởng mới xuất hiện sau khi xem test:

`POST-EVALUATION FUTURE WORK`

không phải:

`CURRENT PROJECT FINAL SELECTION UPDATE`

---

# 28. Official-run uniqueness policy

M8 có đúng một official final-evaluation subject identity.

M8.3 có thể rerun **cùng exact deterministic code/artifact** chỉ khi cần technical recovery như:

```text
kernel crash
file write interrupted
artifact round-trip verification
identical reproducibility check
```

Những rerun này phải:

```text
không đổi config
không đổi data population
không đổi threshold
không đổi feature/preprocessing
không đổi seed
không dùng result trước để thay logic
```

Mọi rerun phải được ghi reason trong runtime metadata.

---

# 29. M8 artifact persistence schema

M8 dùng substep-separated persistence để tránh ghi đè và giữ lineage rõ.

Proposed canonical roots:

```text
data/processed/m8_02_final_test_audit/
data/processed/m8_03_final_test_inference/
data/processed/m8_04_final_metric_analysis/
data/processed/m8_05_final_error_analysis/
```

M8.2 expected artifacts tối thiểu:

```text
m8_02_final_test_audit.json
m8_02_audit_manifest.json

nếu materialize representation:
X_final_test.npz
row_id_final_test.npy
feature_names_final_test.json hoặc exact canonical feature-name fingerprint
```

M8.3 expected official inference artifacts tối thiểu:

```text
y_final_test.npy
risk_score_final_test.npy
y_pred_final_test.npy
row_id_final_test.npy
m8_03_final_inference_registry.json
m8_03_inference_manifest.json
```

M8.4 expected artifacts tối thiểu:

```text
m8_04_final_metric_confusion_analysis.json
m8_04_analysis_manifest.json
```

M8.5 expected artifacts tối thiểu:

```text
m8_05_final_error_analysis.json
m8_05_analysis_manifest.json

optional row-group lineage artifact nếu cần:
m8_05_error_group_row_ids.npz
```

M8.6 final CANON:

`CANON-M8.6 — Final Evaluation Registry, Decision Log và M8 Gate.md`

Tên file có thể được điều chỉnh ở implementation nếu project structure thực tế yêu cầu, nhưng logical artifact roles và non-overwrite policy phải giữ nguyên.

---

# 30. Required metadata / fingerprints

M8.2–M8.5 phải persist đủ metadata để independent review.

Minimum identity fields:

```text
project / milestone / substep
analysis or runner version
source artifact paths
source artifact sizes / SHA256 khi phù hợp
raw dataset identity / fingerprint khi dùng
FINAL TEST boundary
row count
fraud count khi target được authorized
row_id fingerprint
feature-name fingerprint
feature count
matrix format / dtype
preprocessing identity / fingerprint
model config ID
exact model params
model artifact fingerprint hoặc reconstruction identity
training-window ID
imbalance strategy
random_state
probability interface
positive class
threshold
threshold comparator semantics
runtime versions
warning/error summary
final_test_accessed = true chỉ ở authorized M8 substeps
no_retuning_attestation
```

Official prediction artifacts phải có SHA256/fingerprint và round-trip verification trước M8 Gate.

---

# 31. Separation giữa prediction artifact và interpretation

M8.3 nhiệm vụ chính:

```text
produce official frozen predictions
persist artifacts
verify identity
```

M8.3 không cần đưa ra narrative kiểu:

```text
"model tốt"
"model xấu"
"Recall không đủ"
```

M8.4/M8.5 mới diễn giải sau khi integrity đã PASS.

Điều này giúp tách:

```text
OBSERVED ARTIFACT
→ METRIC FACT
→ INTERPRETATION
→ FINAL LIMITATION
```

---

# 32. M8.2 audit scope

M8.1 authorize M8.2 kiểm tra:

```text
FINAL TEST boundary
expected row count = 722,955
expected fraud count = 1,035
no overlap with development partitions
target mapping
row lineage
feature width/order
CSR float32
NaN/inf
unknown category behavior
frozen W_SHORT preprocessing identity
strict-causal history behavior
selected RF logical identity
physical model artifact or deterministic reconstruction path
validation-reproduction evidence if reconstruction is needed
artifact fingerprints
```

M8.2 không được compute final model performance để bypass M8.3/M8.4 separation.

---

# 33. M8.3 official inference scope

M8.3 chỉ được bắt đầu khi:

`M8.2 — PASS`

Official inference sequence:

```text
load verified frozen/reconstructed selected estimator
→ load/use verified frozen W_SHORT preprocessing state
→ load verified FINAL TEST representation
→ assert feature schema
→ predict_proba
→ select class 1 risk score
→ apply frozen threshold 0.50 with verified comparator
→ persist y_pred / risk_score / lineage / target identity
→ fingerprint
→ round-trip
```

Không parameter selection trong M8.3.

---

# 34. M8.4 final metric scope

M8.4 độc lập tái tính:

```text
F1_fraud
Recall_fraud
Precision_fraud
Accuracy_reference
TP
FP
FN
TN
predicted_positive_count
predicted_positive_rate
```

Bắt buộc:

```text
metric reconstruction from persisted y_final_test + y_pred_final_test
confusion arithmetic assertions
row alignment assertions
comparison with M8.3 row counts
```

M8.4 output mới là nguồn canonical cho final metric profile.

---

# 35. M8.5 error / temporal review scope

M8.5 chỉ bắt đầu sau official predictions và final metrics đã khóa.

Allowed questions:

```text
fraud bị bỏ sót có pattern gì?
FP có concentration đáng chú ý không?
risk score của FN/FP phân bố thế nào?
monthly final-test behavior thay đổi ra sao?
validation → final delta theo F1/Recall/Precision/alerts ra sao?
temporal generalization limitation có mạnh hơn hay nhẹ hơn?
```

Guardrail:

`ANALYZE — DO NOT FIX USING FINAL TEST`

---

# 36. STOP conditions trước official FINAL TEST scoring

Bất kỳ điều kiện nào sau đây xảy ra trong M8.2 thì:

`STOP — DO NOT PROCEED TO M8.3`

## S01 — Upstream handoff mismatch

```text
M7.9 final configuration không tái xác minh được
```

## S02 — Model identity mismatch

```text
model/config/class_weight/random_state không khớp frozen identity
```

## S03 — No trustworthy model state

```text
serialized estimator không xác minh được
AND
deterministic reconstruction không tái tạo được upstream validation evidence
```

## S04 — Unauthorized refit path

```text
implementation yêu cầu TRAIN+VALIDATION refit hoặc training population mới
```

## S05 — FINAL TEST boundary mismatch

```text
row boundary không phải 2019-06-01 <= Timestamp < 2019-11-01
```

## S06 — Row-count mismatch

```text
expected 722,955 không khớp mà chưa giải thích được
```

## S07 — Target-support mismatch

```text
expected 1,035 fraud không khớp mà chưa giải thích được
```

## S08 — Partition overlap

```text
FINAL TEST row lineage overlap development population trái protocol
```

## S09 — Feature schema mismatch

```text
width != 47
feature names/order mismatch
wrong sparse/dtype contract
```

## S10 — Learned-state contamination

```text
scaler/encoder/model fit hoặc update bằng FINAL TEST
```

## S11 — Causal-history violation

```text
same/future transaction được dùng làm history
hoặc target history được dùng
```

## S12 — Invalid representation values

```text
unexpected NaN / inf / corruption
```

## S13 — Positive-class mismatch

```text
cannot uniquely identify class 1 probability
```

## S14 — Threshold/comparator mismatch

```text
0.50 semantics không tương thích upstream M7 threshold behavior
```

## S15 — Artifact persistence failure

```text
required audit artifact cannot be persisted/fingerprinted/round-tripped
```

Mọi STOP phải được ghi issue và giải quyết ở audit layer trước scoring.

---

# 37. Post-scoring incident policy

Nếu sau M8.3 hoặc M8.4 phát hiện integrity issue:

```text
STOP INTERPRETATION
→ preserve current artifacts
→ record incident
→ determine whether issue is technical reproduction failure or test-isolation breach
```

Nếu cần technical rerun, rerun chỉ được dùng exact same frozen subject/config/data/threshold.

Nếu incident đã làm FINAL TEST tham gia development choice:

```text
record TEST-ISOLATION BREACH
→ do not hide it
→ do not claim clean protected evaluation without qualification
```

M8.1 không cho phép sửa lịch sử âm thầm.

---

# 38. No-retuning attestation

M8.3–M8.6 phải carry field:

`no_retuning_after_final_test = true`

M8.6 phải xác nhận:

```text
No model-family change
No training-window change
No feature change
No preprocessing change
No imbalance-strategy change
No hyperparameter change
No seed search
No calibration tuning
No threshold change
```

sau khi FINAL TEST performance được mở.

---

# 39. Decision Log — M8.1

## M8.1-D01 — Milestone role

Decision:

`M8 = PROTECTED FINAL EVALUATION`

Status:

`LOCKED`

---

## M8.1-D02 — Official subject count

Decision:

`ONE OFFICIAL FINAL-EVALUATION SUBJECT`

Status:

`LOCKED`

---

## M8.1-D03 — Training window

Decision:

`W_SHORT`

Status:

`INHERITED — FROZEN`

---

## M8.1-D04 — Model family

Decision:

`Random Forest`

Status:

`INHERITED — FROZEN`

---

## M8.1-D05 — Model config

Decision:

`RF-REF-100-GINI-SQRT-UNPRUNED-CW`

Status:

`INHERITED — FROZEN`

---

## M8.1-D06 — Imbalance strategy

Decision:

`CLASS_WEIGHT_BALANCED`

Status:

`INHERITED — FROZEN`

---

## M8.1-D07 — Random state

Decision:

`42`

Status:

`INHERITED — FROZEN`

---

## M8.1-D08 — Probability interface

Decision:

`predict_proba / positive class = 1`

Status:

`INHERITED — FROZEN`

---

## M8.1-D09 — Risk-score language

Decision:

`risk score / positive-class score`

Prohibited claim:

`calibrated confidence`

Status:

`LOCKED`

---

## M8.1-D10 — Final numerical threshold

Decision:

`0.50`

Status:

`INHERITED — FROZEN`

---

## M8.1-D11 — Final-test boundary

Decision:

```text
2019-06-01 <= Timestamp < 2019-11-01
```

Status:

`INHERITED — LOCKED`

---

## M8.1-D12 — Final-test expected population

Decision:

```text
722,955 rows
1,035 fraud
```

Role:

`EXPECTED IDENTITY FOR M8.2 AUDIT`

Status:

`INHERITED — VERIFIED UPSTREAM`

---

## M8.1-D13 — Core metric

Decision:

`F1_fraud`

Status:

`INHERITED — LOCKED`

---

## M8.1-D14 — Secondary metrics

Decision:

`Recall_fraud + Precision_fraud`

Status:

`INHERITED — LOCKED`

---

## M8.1-D15 — Accuracy

Decision:

`REFERENCE ONLY`

Status:

`LOCKED`

---

## M8.1-D16 — Physical estimator artifact

Decision:

`DO NOT ASSUME; AUDIT IN M8.2`

Status:

`OPEN — REQUIRES ARTIFACT AUDIT`

---

## M8.1-D17 — Deterministic reconstruction fallback

Decision:

Authorized only if serialized subject is absent/unusable and reconstruction uses exact frozen W_SHORT contract and passes upstream validation-reproduction checks.

Status:

`CONDITIONAL — M8.2`

---

## M8.1-D18 — TRAIN+VALIDATION refit

Decision:

`NOT AUTHORIZED BY DEFAULT`

Status:

`LOCKED`

---

## M8.1-D19 — Final-test preprocessing

Decision:

`TRANSFORM WITH FROZEN W_SHORT LEARNED STATE ONLY`

Status:

`LOCKED`

---

## M8.1-D20 — Causal history

Decision:

`Timestamp(history) < Timestamp(current)`

Status:

`INHERITED — LOCKED`

---

## M8.1-D21 — Allowed final diagnostics

Decision:

`DESCRIPTIVE / ERROR / TEMPORAL ANALYSIS AFTER OFFICIAL PREDICTIONS`

Status:

`LOCKED`

---

## M8.1-D22 — Retroactive optimization

Decision:

`PROHIBITED`

Status:

`LOCKED`

---

## M8.1-D23 — Validation comparison

Decision:

`DESCRIPTIVE ONLY`

Status:

`LOCKED`

---

## M8.1-D24 — Advanced metrics

Decision:

`NOT CORE / NOT REQUIRED`

Status:

`LOCKED FOR CORE M8`

---

## M8.1-D25 — Artifact persistence

Decision:

`SUBSTEP-SEPARATED + FINGERPRINTED + ROUND-TRIPPED`

Status:

`LOCKED`

---

## M8.1-D26 — M8.2 release

Decision:

`AUTHORIZED AFTER M8.1 PASS`

Scope:

`AUDIT ONLY — NOT OFFICIAL PERFORMANCE SCORING`

Status:

`READY`

---

# 40. M8.1 Gate

M8.1 là protocol milestone nên Gate được đánh giá bằng consistency với upstream CANON, không cần runtime FINAL TEST output.

## G01 — M7 handoff available

Requirement:

`M7 PASS / development configuration frozen`

Observed:

`YES`

Result:

`PASS`

---

## G02 — Official subject identity fixed

Requirement:

```text
W_SHORT
Random Forest
RF-REF-100-GINI-SQRT-UNPRUNED-CW
CLASS_WEIGHT_BALANCED
random_state=42
```

Observed:

`LOCKED`

Result:

`PASS`

---

## G03 — Feature/preprocessing identity fixed

Requirement:

`M4 canonical / 10 → 47 / CSR float32 / TRAIN-only learned state`

Observed:

`LOCKED`

Result:

`PASS`

---

## G04 — FINAL TEST boundary fixed

Requirement:

`2019-06-01 <= Timestamp < 2019-11-01`

Observed:

`LOCKED`

Result:

`PASS`

---

## G05 — Positive class fixed

Requirement:

`fraud = 1`

Observed:

`LOCKED`

Result:

`PASS`

---

## G06 — Threshold fixed

Requirement:

`0.50`

Observed:

`FROZEN`

Result:

`PASS`

---

## G07 — Metric contract fixed

Requirement:

```text
Primary F1
Secondary Recall/Precision
CM counts mandatory
Accuracy reference only
```

Observed:

`LOCKED`

Result:

`PASS`

---

## G08 — No-refit rule fixed

Requirement:

`TRAIN+VALIDATION refit not authorized by default`

Observed:

`LOCKED`

Result:

`PASS`

---

## G09 — Physical-model uncertainty surfaced

Requirement:

`Do not invent serialized artifact identity`

Observed:

`OPEN — M8.2 AUDIT`

Result:

`PASS`

---

## G10 — Reconstruction fallback controlled

Requirement:

`same W_SHORT subject + validation reproduction before FINAL TEST`

Observed:

`DEFINED`

Result:

`PASS`

---

## G11 — Allowed diagnostics fixed

Requirement:

`descriptive only after official predictions`

Observed:

`LOCKED`

Result:

`PASS`

---

## G12 — Forbidden optimization fixed

Requirement:

`no retroactive development after FINAL TEST`

Observed:

`LOCKED`

Result:

`PASS`

---

## G13 — Persistence schema defined

Requirement:

`auditable prediction/metric/error artifacts + fingerprints`

Observed:

`DEFINED`

Result:

`PASS`

---

## G14 — STOP conditions defined

Requirement:

`identity / lineage / representation / learned-state failures stop M8.3`

Observed:

`DEFINED`

Result:

`PASS`

---

## G15 — FINAL TEST performance still unread in M8.1

Requirement:

`NO PERFORMANCE ACCESS`

Observed:

`SATISFIED BY M8.1 WORK TYPE`

Result:

`PASS`

---

# 41. M8.1 conclusion

M8.1 đã khóa Final Evaluation Charter trước runtime.

```text
M8.1 — PASS

Final Evaluation Charter:
LOCKED

Official Subject:
W_SHORT
Random Forest
RF-REF-100-GINI-SQRT-UNPRUNED-CW
CLASS_WEIGHT_BALANCED
random_state = 42

Risk Score:
predict_proba class 1

Threshold:
0.50

FINAL TEST:
2019-06 → 2019-10
722,955 rows
1,035 fraud

Core Metric:
F1_fraud

Retroactive Optimization:
PROHIBITED

TRAIN+VALIDATION Refit:
NOT AUTHORIZED BY DEFAULT

Physical Selected-Model Artifact:
OPEN — REQUIRES M8.2 AUDIT

FINAL TEST Performance:
STILL UNREAD

M8.3 Official Scoring:
NOT YET AUTHORIZED

Next:
M8.2 — FINAL TEST artifact / lineage / representation audit
```

M8.2 chỉ được chuyển sang M8.3 khi tất cả identity, lineage, feature/preprocessing, causal-history và model-state gates bắt buộc đều PASS.

---

# 42. Handoff sang M8.2

M8.2 phải bắt đầu từ câu hỏi:

> Project có thể chứng minh rằng FINAL TEST representation và official selected estimator thực sự tương thích đúng với frozen M7 identity trước khi tạo bất kỳ official final prediction nào hay không?

M8.2 cần audit tối thiểu:

```text
1. project/source artifact location
2. raw dataset identity
3. FINAL TEST boundary
4. expected 722,955 rows
5. expected 1,035 fraud
6. no illegal development overlap
7. target mapping
8. row lineage
9. strict-causal history reconstruction
10. 10-feature semantic contract
11. frozen W_SHORT preprocessing state
12. 47-column feature width/order
13. CSR float32
14. finite values
15. unknown-category handling
16. selected RF logical identity
17. exact RF params from canonical registry
18. physical estimator existence/fingerprint
19. deterministic reconstruction + validation reproduction if needed
20. threshold comparator identity
21. output artifact schema
22. STOP gate
```

Decision trước khi chạy M8.2:

`OPEN — REQUIRES RUNTIME AUDIT`

M8.2 chỉ được PASS sau khi người thực hiện chạy audit trên project thật và output được review theo sequence:

```text
execution
→ integrity
→ consistency
→ leakage / causal logic
→ artifact identity
→ decision
```

Không invent runtime result.
