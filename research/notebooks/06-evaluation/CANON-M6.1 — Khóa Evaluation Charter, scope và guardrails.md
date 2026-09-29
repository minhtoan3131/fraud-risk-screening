# CANON-M6.1 — Khóa Evaluation Charter, scope và guardrails

## Document status

Milestone:

`M6 — Evaluation + Error Analysis`

Substep:

`M6.1 — Evaluation Charter / Scope / Guardrails`

Work type:

`CONCEPT / PROTOCOL / GUARDRAIL`

Runtime model experiment:

`NOT REQUIRED IN M6.1`

Runtime evaluation artifact audit:

`DEFERRED TO M6.2`

M6.1 không fit classifier.

M6.1 không retrain baseline.

M6.1 không đọc FINAL TEST.

M6.1 không chọn final model.

M6.1 không chọn final training window.

M6.1 không tối ưu threshold.

Mục tiêu của M6.1 là khóa luật đánh giá trước khi M6.2–M6.6 bắt đầu đọc sâu các baseline outputs.

---

# 1. Câu hỏi trung tâm

M6.1 phải trả lời:

> Milestone 6 được phép đánh giá evidence nào, trên population nào, bằng metric nào, theo comparison protocol nào, được phép phân tích lỗi đến đâu, và những quyết định nào tuyệt đối phải giữ OPEN cho M7?

M6.1 phải đủ rõ để các bước sau không phải tự quyết định lại:

- nguồn prediction nào được phép dùng;
- nguồn probability/risk score nào được phép dùng;
- target nào là canonical evaluation target;
- population nào được phép đánh giá;
- positive class là gì;
- metric nào là primary;
- metric nào là secondary;
- diagnostic nào là bắt buộc;
- Accuracy có vai trò gì;
- error analysis group được định nghĩa thế nào;
- lineage mapping phải được kiểm tra ra sao;
- W_SHORT/W_LONG được so như thế nào;
- model family được so như thế nào;
- probability analysis được phép làm gì;
- threshold analysis nào bị cấm;
- FINAL TEST có quyền gì;
- khi nào phải STOP;
- M6 được phép kết luận đến mức nào;
- M7 phải nhận handoff gì.

---

# 2. Trạng thái đầu vào khi bước vào M6

Milestone 5 đã PASS.

Canonical handoff từ M5:

```text
M5:
PASS

Official model families:
3 / 3 COMPLETE

Official baseline runs:
6 / 6 VALID

Prediction evidence:
6 / 6 PERSISTED

Probability evidence:
6 / 6 PERSISTED

Controlled W_SHORT/W_LONG coverage:
COMPLETE

Imbalance strategy:
NONE

Final threshold:
OPEN

Training-window winner:
OPEN

Model-family winner:
OPEN

FINAL TEST:
PROTECTED

Handoff:
READY FOR M6 — EVALUATION + ERROR ANALYSIS
```

M6.1 không mở lại M5 modeling decisions chỉ vì evaluation bắt đầu.

---

# 3. Vai trò chính thức của M6

Cấp project đã phân chia:

```text
M5 — Modeling baseline
        ↓
M6 — Evaluation + Error Analysis
        ↓
M7 — Model Selection / Robustness / Tuning
```

M6 hỏi:

> Những prediction baseline đã tạo ra đang đúng/sai theo cách nào?

và:

> Các baseline khác nhau ra sao về fraud detection, false alert, probability behavior và error pattern?

Sản phẩm của M6:

`EVALUATION EVIDENCE`

không phải:

`FINAL MODEL`

M6 tập trung vào:

- independent metric reconstruction;
- Confusion Matrix;
- fraud Recall;
- fraud Precision;
- fraud F1;
- predicted-positive behavior;
- controlled training-window comparison;
- cross-model comparison;
- FP/FN error analysis;
- descriptive probability/risk-score analysis;
- findings;
- Decision Log;
- M7 handoff.

---

# 4. M6 được phép làm gì

M6 được phép:

- load persisted `y_pred`;
- load persisted `risk_score`;
- load persisted summary JSON;
- load config lock;
- load pair manifest;
- load canonical `y_validation`;
- load `row_id_validation`;
- load authorized validation-level semantic representation khi cần error analysis;
- tái tính metric độc lập;
- reconstruct TP / FP / FN / TN;
- tính predicted-positive count/rate;
- kiểm tra probability range;
- mô tả probability distribution;
- phân nhóm TP / FP / FN / TN;
- so sánh FN với TP;
- so sánh FP với TN;
- so sánh W_SHORT với W_LONG trong cùng model family;
- so sánh model family khi giữ training window cố định;
- ghi computational evidence đã tồn tại từ M5;
- ghi observed trade-off;
- ghi hypothesis cần M7 kiểm tra;
- tạo Evaluation Registry;
- tạo M6 Findings;
- tạo M6 Decision Log.

---

# 5. M6 không được phép làm gì

Core M6 không được:

- gọi `.fit(...)` để cải thiện baseline;
- retrain Logistic Regression;
- retrain Decision Tree;
- retrain Random Forest;
- đổi model config;
- search `C`;
- search `max_depth`;
- search `min_samples_leaf`;
- search `n_estimators`;
- search `max_features`;
- thử `class_weight`;
- random oversampling;
- random undersampling;
- SMOTE;
- temporal CV để chọn config;
- random CV để chọn config;
- probability calibration experiment có mục đích selection;
- threshold grid search;
- threshold optimization;
- threshold riêng cho từng model để cải thiện F1;
- dùng FINAL TEST;
- tuyên bố final model;
- tuyên bố final training window;
- tuyên bố final production performance.

Nếu analysis cho thấy cần một intervention:

`HANDOFF TO M7`

không tự biến M6 thành tuning milestone.

---

# 6. Problem definition và prediction context

Problem:

`Binary fraud-risk screening classification`

Positive class:

`fraud = 1`

Negative class:

`non-fraud = 0`

Prediction point:

`transaction screening time`

Interpretation scope:

Model tạo:

`risk signal / probability / binary baseline prediction`

cho một transaction cần screening.

Project không được diễn giải baseline evaluation như bằng chứng của một hệ thống production tự động chặn giao dịch ngoài đời.

---

# 7. Canonical evaluation population

Core M6 evaluation population:

`EXTERNAL VALIDATION`

Temporal boundary:

```text
2019-01-01 <= Timestamp < 2019-06-01
```

Rows:

`712,458`

Fraud positives:

`1,052`

Natural temporal/class distribution:

`PRESERVED`

M6 không:

- resample validation;
- rebalance validation;
- oversample fraud;
- undersample non-fraud;
- tạo synthetic validation rows;
- thay validation bằng training rows;
- thay validation bằng random split.

---

# 8. FINAL TEST policy

FINAL TEST temporal boundary:

```text
2019-06-01 <= Timestamp < 2019-11-01
```

M6 rights:

`NO ACCESS`

FINAL TEST không được dùng cho:

- metric reconstruction;
- training-window comparison;
- model-family comparison;
- error analysis;
- probability analysis;
- threshold analysis;
- subgroup analysis;
- robustness check;
- selection;
- confirmation của một pattern thấy trên validation.

M6 phải giữ:

`FINAL TEST PROTECTED`

Nếu một M6 notebook có code path load FINAL TEST:

`STOP`

không tiếp tục interpretation cho notebook đó cho đến khi code path được loại bỏ và rerun.

---

# 9. Official evaluation subjects

M6 chỉ đánh giá official baseline runs đã được M5 runtime-review.

## Logistic Regression

Official config:

`LR-B04-LBFGS-L2-C1`

Official runs:

```text
M5-LR-SHORT-B04
M5-LR-LONG-B04
```

---

## Decision Tree

Official config:

`DT-B01-DEFAULT-GINI-UNPRUNED`

Official runs:

```text
M5-DT-SHORT-B01
M5-DT-LONG-B01
```

---

## Random Forest

Official config:

`RF-B01-100-GINI-SQRT-BOOTSTRAP`

Official runs:

```text
M5-RF-SHORT-B01
M5-RF-LONG-B01
```

M6 không tự thêm một seventh run vào official registry.

Nếu có experiment mới:

nó thuộc protocol sau, không được âm thầm nhập vào M6 baseline evaluation.

---

# 10. Canonical evaluation artifact sources

## 10.1. M4.7 target / lineage source

Base directory:

`data/processed/m4_07_baseline_ready`

Required:

```text
y_validation.npy
row_id_validation.npy
feature_names.json
manifest.json
```

M6.2 phải audit actual file integrity trước khi downstream analysis dùng.

---

## 10.2. Logistic Regression evidence

Base directory:

`data/processed/m5_03_logistic_regression_baseline_b04`

Required:

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

---

## 10.3. Decision Tree evidence

Base directory:

`data/processed/m5_04_decision_tree_baseline`

Required:

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

---

## 10.4. Random Forest evidence

Base directory:

`data/processed/m5_05_random_forest_baseline`

Required:

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

M6.1 chỉ khóa expected source contract.

Actual existence/round-trip:

`VERIFY IN M6.2`

---

# 11. Independent evaluation principle

M6 không được chỉ copy metric từ M5 summary.

M6 phải tái tính metric từ:

```text
canonical y_validation
+
persisted y_pred
```

cho từng official run.

Sau đó mới so với persisted M5 summary.

Mục đích:

- xác minh đúng artifact;
- phát hiện stale output;
- phát hiện accidental file mix-up;
- phát hiện label-order mistake;
- phát hiện metric implementation divergence;
- tạo independent Evaluation Registry.

Nếu recomputed result không khớp persisted summary trong numerical tolerance hợp lý:

`STOP`

Không viết comparative conclusion cho run đó.

---

# 12. Canonical metric strategy

## 12.1. Primary metric

`F1_fraud`

Vai trò:

đọc cân bằng giữa Precision_fraud và Recall_fraud.

F1 không được đọc một mình.

Status:

`INHERITED — LOCKED`

---

## 12.2. Secondary metric 1

`Recall_fraud`

Question:

> Trong toàn bộ fraud thật, model tìm được bao nhiêu?

Recall giảm khi:

`FN tăng`

M6 phải luôn chuyển Recall về:

- TP count;
- FN count;
- actual fraud denominator.

Status:

`INHERITED — LOCKED`

---

## 12.3. Secondary metric 2

`Precision_fraud`

Question:

> Trong các transaction model flag positive, bao nhiêu thật sự là fraud?

Precision giảm khi:

`FP tăng`

M6 phải luôn chuyển Precision về:

- TP count;
- FP count;
- predicted-positive denominator.

Status:

`INHERITED — LOCKED`

---

# 13. Confusion Matrix contract

Mỗi official evaluation bắt buộc có:

```text
TP
FP
FN
TN
```

Arithmetic checks:

```text
TP + FN
=
actual fraud
=
1,052

TP + FP
=
predicted_positive_count

TP + FP + FN + TN
=
712,458
```

Confusion Matrix là:

`MANDATORY DIAGNOSTIC`

Không được chỉ báo F1.

---

# 14. Predicted-positive / alert-burden contract

Mỗi run phải có:

```text
predicted_positive_count
predicted_positive_rate
```

Interpretation:

`screening / alert-volume diagnostic`

Không phải primary metric.

M6 phải đọc metric này cùng Precision/Recall.

Ví dụ một model có Precision cao nhưng predicted-positive count cực thấp có thể đang bỏ sót nhiều fraud.

M6 không được tự diễn giải:

`high Precision = automatically better`

---

# 15. Accuracy policy

Accuracy:

`REFERENCE ONLY`

M6 không:

- sort model theo Accuracy;
- dùng Accuracy làm primary winner criterion;
- dùng Accuracy cao để che FN burden;
- gọi model gần hoàn hảo chỉ vì Accuracy gần 1.

M6 phải ưu tiên:

```text
F1
Recall
Precision
Confusion counts
Predicted-positive behavior
```

---

# 16. Metric implementation contract

M6 metric implementation phải giữ semantic:

```text
positive class:
1

confusion labels:
[0, 1]

F1:
positive-class fraud

Recall:
positive-class fraud

Precision:
positive-class fraud

zero division:
explicit safe handling
```

M6.2 được phép viết implementation độc lập nhưng phải cho cùng semantics với M5.

Nếu implementation khác syntax nhưng cùng output semantics:

`ALLOWED`

Nếu definition thay đổi:

`NOT ALLOWED`

---

# 17. Prediction artifact integrity guardrails

Với mỗi `y_pred`, M6.2 phải assert tối thiểu:

```text
file exists

length:
712,458

shape:
1D

dtype:
integer-compatible

unique values:
subset of {0, 1}

no NaN

experiment identity:
matches expected run

final-test flag:
false
```

Nếu fail:

`STOP`

---

# 18. Probability artifact integrity guardrails

Với mỗi `risk_score`, M6.2 phải assert tối thiểu:

```text
file exists

length:
712,458

shape:
1D

finite:
TRUE

range:
0 <= score <= 1

risk_score_kind:
predict_proba

experiment identity:
matches expected run
```

Probability artifact được hiểu là:

`positive-class fraud probability / score persisted by M5`

Không được tự giả định một artifact khác là fraud probability nếu metadata không xác nhận.

---

# 19. Evaluation identity guardrail

M6 phải ngăn accidental cross-run mixing.

Mỗi evaluation record phải gắn:

- experiment_id;
- model_family;
- model_config_id;
- training_window_id;
- feature_version;
- preprocessing_version;
- matrix_schema_version;
- threshold_policy;
- imbalance_strategy;
- validation rows;
- validation fraud rows.

Nếu summary nói:

`M5-RF-LONG-B01`

nhưng filename/config identity thuộc:

`M5-RF-SHORT-B01`

thì:

`STOP`

---

# 20. Lineage / row mapping policy

M6.2 metric reconstruction chỉ cần:

`y_validation + y_pred`

M6.6 error analysis cần quay lại transaction-level evidence.

Canonical lineage anchor:

`row_id_validation.npy`

M6 error-analysis mapping phải xác minh:

```text
row_id length:
712,458

row_id uniqueness:
as required by canonical artifact contract

ordering alignment:
verified

prediction length:
same validation order

target length:
same validation order
```

Không được:

- join chỉ dựa vào DataFrame positional index nếu lineage chưa kiểm tra;
- sort một side nhưng không sort side còn lại;
- reset index rồi giả định vẫn cùng transaction;
- map bằng raw identifier không nằm trong canonical lineage contract;
- map tới FINAL TEST rows.

Nếu lineage mapping không được chứng minh:

M6.6 không được đưa ra row-level error conclusion.

---

# 21. Error-state definition

Với từng run:

```text
TP:
y_true = 1
y_pred = 1

FN:
y_true = 1
y_pred = 0

FP:
y_true = 0
y_pred = 1

TN:
y_true = 0
y_pred = 0
```

Primary fraud-side analysis:

`FN vs TP`

Lý do:

cả hai là actual fraud.

Question:

> Fraud nào bị bỏ sót và fraud nào được bắt?

Primary non-fraud-side analysis:

`FP vs TN`

Lý do:

cả hai là actual non-fraud.

Question:

> Non-fraud nào dễ bị flag nhầm?

Status:

`LOCKED FOR CORE M6 ERROR ANALYSIS`

---

# 22. Authorized error-analysis feature scope

Core dimensions ưu tiên:

```text
amount_numeric

time_since_previous_transaction_min
transactions_last_1h
amount_minus_previous_mean

is_new_merchant
has_prior_card_history

transaction_mode
location_state
hour_of_day
day_of_week
```

M6 có thể dùng human-readable semantic values tương ứng nếu lineage/mapping verified.

M6 không được dùng:

- future information;
- post-transaction field;
- protected target-derived field;
- final-test-only information;
- arbitrary leakage-prone derived variable.

Nếu tạo evaluation-only derived diagnostic:

phải:

- mô tả công thức;
- không thay model input;
- không thay prediction;
- không gọi đó là baseline feature nếu model không dùng.

---

# 23. Subgroup-analysis guardrail

Subgroup analysis có nguy cơ data dredging.

Do đó M6 phải:

- khai báo dimension đang phân tích;
- báo subgroup denominator;
- báo fraud/non-fraud support;
- báo raw count;
- báo rate khi phù hợp;
- cảnh báo subgroup nhỏ;
- không chỉ chọn subgroup có pattern “đẹp” để báo;
- phân biệt exploratory observation với locked conclusion.

Không được:

`scan hàng trăm subgroup → lấy subgroup cực đoan nhất → gọi là root cause`

M6 tìm:

`ERROR PATTERN`

không tự động chứng minh:

`CAUSAL ROOT CAUSE`

---

# 24. Probability analysis scope

M6 được phép:

- overall score distribution;
- fraud score distribution;
- non-fraud score distribution;
- TP score distribution;
- FN score distribution;
- FP score distribution;
- TN score distribution;
- quantiles;
- near-boundary descriptive counts;
- score overlap;
- descriptive ranking behavior.

M6 không được:

- optimize threshold;
- search F1-max threshold;
- search Recall-max threshold;
- search Precision target threshold;
- choose model-specific new threshold;
- compare models sau khi mỗi model được threshold-tune riêng.

Threshold policy trong M6 core:

`M5 DEFAULT MODEL DECISION RULE`

Final threshold:

`OPEN — M7`

---

# 25. Near-boundary analysis guardrail

Nếu M6 dùng khái niệm:

`near-boundary`

thì numerical band phải:

- được khai báo trước khi đọc subgroup result;
- được xem là descriptive diagnostic;
- không được tối ưu để tạo narrative đẹp;
- không được gọi là threshold recommendation.

Nếu không có lý do rõ để định nghĩa band:

M6 có thể chỉ dùng quantile/distribution summary.

---

# 26. Controlled training-window comparison contract

Official within-family pairs:

```text
LR:
M5-LR-SHORT-B04
vs
M5-LR-LONG-B04

DT:
M5-DT-SHORT-B01
vs
M5-DT-LONG-B01

RF:
M5-RF-SHORT-B01
vs
M5-RF-LONG-B01
```

Comparison principle:

`ONLY TRAINING WINDOW CHANGES`

Phải giữ cùng:

- model family;
- exact model config;
- feature version;
- preprocessing version;
- matrix schema;
- imbalance strategy;
- threshold policy;
- validation population;
- metric semantics;
- error handling.

Trước khi đọc delta metric:

`PAIR INTEGRITY MUST PASS`

---

# 27. Training-window comparison evidence bundle

Mỗi pair tối thiểu phải đọc:

```text
F1_SHORT
F1_LONG
ΔF1

Recall_SHORT
Recall_LONG
ΔRecall

Precision_SHORT
Precision_LONG
ΔPrecision

TP delta
FP delta
FN delta
TN delta

predicted-positive count delta
predicted-positive rate delta

computational context
```

Không được chỉ lưu:

`F1_SHORT vs F1_LONG`

---

# 28. Training-window interpretation rule

M6 có thể dùng wording:

> Trong model/config X, W_SHORT có validation Recall cao hơn W_LONG.

M6 không được tự mở rộng thành:

> W_SHORT luôn tốt hơn W_LONG.

Mỗi conclusion phải scope theo:

- model family;
- config;
- validation period;
- baseline rule.

Nếu evidence mixed:

`INCONCLUSIVE`

Nếu cần robustness:

`HANDOFF M7`

---

# 29. Training-window tie-break boundary

Upstream protocol đã định nghĩa:

- F1 là primary evidence;
- Recall/Precision/raw counts phải được đọc;
- không dùng arbitrary delta-F1 epsilon;
- nếu trade-off khó phân biệt, có thể cần robustness;
- parsimony fallback chỉ dùng sau robustness khi predictive preference vẫn không ổn định.

M6 không tự động khóa tie-break winner.

M6 responsibility:

`DESCRIBE EVIDENCE + IDENTIFY WHETHER ROBUSTNESS IS NEEDED`

M7 responsibility:

`RUN ROBUSTNESS / SELECTION IF AUTHORIZED`

---

# 30. Cross-model comparison contract

Model-family comparison phải giữ training window cố định.

Valid W_SHORT group:

```text
M5-LR-SHORT-B04
M5-DT-SHORT-B01
M5-RF-SHORT-B01
```

Valid W_LONG group:

```text
M5-LR-LONG-B04
M5-DT-LONG-B01
M5-RF-LONG-B01
```

M6 không được dùng:

`LR-SHORT vs RF-LONG`

để kết luận thuần về model family.

Vì comparison đó thay:

```text
model family
+
training window
```

cùng lúc.

---

# 31. Cross-model evidence bundle

Mỗi same-window group phải đọc:

- F1;
- Recall;
- Precision;
- TP;
- FP;
- FN;
- TN;
- predicted-positive count;
- predicted-positive rate;
- probability behavior;
- computational evidence từ M5;
- error-pattern evidence khi M6.6 có.

M6 được phép mô tả:

`trade-off`

M6 không được khóa:

`FINAL MODEL`

---

# 32. Computational evidence policy

M5 đã ghi:

- fit time;
- prediction time;
- convergence/warnings;
- tree/forest complexity.

M6 được phép dùng computational evidence như một descriptive dimension.

Ví dụ hợp lệ:

> RF-LONG có computational cost cao hơn đáng kể so với LR-LONG trong baseline runs.

Không hợp lệ:

> LR nhanh nhất nên là final model.

Computational evidence:

`SELECTION INPUT FOR M7`

không phải:

`M6 WINNER RULE`

---

# 33. Findings vocabulary

M6 sử dụng bốn loại finding.

## VERIFIED

Fact đã được integrity/arithmetic check.

Ví dụ:

`TP + FN = 1,052`

---

## OBSERVED

Pattern trực tiếp trên validation.

Ví dụ:

> RF-LONG phát ít positive hơn RF-SHORT ở default rule.

---

## DESCRIPTIVE COMPARATIVE EVIDENCE

Comparison hợp lệ theo protocol.

Ví dụ:

> Trong DT-B01, W_SHORT có Recall cao hơn W_LONG trên canonical validation.

---

## HYPOTHESIS / OPEN QUESTION

Giải thích chưa được experiment tách nguyên nhân.

Ví dụ:

> Temporal nonstationarity có thể góp phần vào W_LONG behavior.

Không được nâng:

`HYPOTHESIS`

thành:

`CAUSAL CONCLUSION`

nếu chưa có experiment phù hợp.

---

# 34. Overclaim guardrail

Project sử dụng synthetic dataset.

M6 không được suy ra trực tiếp:

- hiệu năng ngân hàng thật;
- fraud prevalence ngoài đời;
- operational loss ngoài đời;
- fraud loss prevented;
- analyst workload thực tế;
- production readiness;
- customer impact;
- regulatory suitability.

Wording được phép:

> Trên temporal VALIDATION của synthetic dataset hiện tại, baseline X có metric/error behavior như sau.

---

# 35. No-retroactive-baseline-modification rule

Nếu M6 phát hiện một baseline có Recall thấp:

không được quay lại M5 notebook và sửa:

- config;
- feature;
- class_weight;
- threshold;
- number of trees;
- depth;

rồi thay official M5 output bằng run mới mà không mở một experiment protocol mới.

M5 official baselines là:

`FIXED EVALUATION SUBJECTS`

Intervention mới:

`M7 CANDIDATE EXPERIMENT`

---

# 36. No-threshold-contamination rule

M6 baseline confusion metrics phải dựa trên:

`persisted official M5 y_pred`

và default decision rule đã dùng trong M5.

Không được tạo:

```text
new y_pred at threshold 0.20
```

rồi ghi đè baseline confusion evidence.

Threshold sensitivity nếu sau này authorized:

phải là một analysis/experiment riêng và không thay canonical baseline registry.

---

# 37. No-resampling-evaluation rule

Validation phải giữ natural distribution.

M6 không được resample validation chỉ để:

- làm Confusion Matrix cân đối hơn;
- làm biểu đồ dễ nhìn hơn;
- làm metric “công bằng” hơn;
- tăng số fraud trong subgroup.

Nếu visualization cần sampling để render:

sample chỉ được dùng cho visualization convenience và không dùng để tính official metric.

Official metric:

`FULL VALIDATION ONLY`

---

# 38. Evaluation registry contract

M6.2 sẽ tạo một independent Evaluation Registry.

Mỗi row tối thiểu cần các field logic sau:

```text
experiment_id
model_family
model_config_id
training_window_id

feature_version
preprocessing_version
matrix_schema_version

validation_period
validation_rows
validation_fraud_rows

imbalance_strategy
threshold_policy

prediction_artifact
risk_score_artifact
summary_artifact

f1_fraud
recall_fraud
precision_fraud
accuracy_reference

tp
fp
fn
tn

predicted_positive_count
predicted_positive_rate

risk_score_available
risk_score_kind

independent_metric_check
summary_match_check
artifact_integrity_check
final_test_accessed

evaluation_status
notes
```

Exact serialization format được phép chốt ở M6.2.

Semantic contract:

`LOCKED`

---

# 39. M6.2 readiness checklist

Trước khi M6.2 được tạo, M6.1 yêu cầu notebook phải có gate cho:

```text
G01:
canonical M4.7 validation target identity

G02:
row_id_validation identity

G03:
6 y_pred existence

G04:
6 risk_score existence

G05:
6 summary existence

G06:
3 config lock existence

G07:
3 pair manifest existence

G08:
length / dtype / finite checks

G09:
experiment identity checks

G10:
independent metric reconstruction

G11:
summary metric agreement

G12:
confusion arithmetic

G13:
FINAL TEST isolation

G14:
evaluation registry persistence

G15:
round-trip if registry persisted
```

Actual gate names có thể thay đổi.

Required semantics:

`MUST BE COVERED`

---

# 40. Runtime-dependent protocol

M6.1 là protocol document nên không cần runtime PASS.

M6.2–M6.6 là runtime/evidence-dependent.

Workflow bắt buộc:

```text
question
    ↓
read CANON
    ↓
lock input / scope
    ↓
create notebook
    ↓
Run All on real environment
    ↓
preserve outputs
    ↓
runtime review
    ↓
integrity / leakage review
    ↓
interpret
    ↓
Findings
    ↓
Decision Log
    ↓
Gate
```

Nếu thiếu check:

`ADD CHECK + RERUN`

Không:

`ASSUME PASS`

---

# 41. M6.1 Decision Log

## M6.1-D01 — Milestone role

Decision:

M6 là:

`EVALUATION + ERROR ANALYSIS`

không phải final model selection.

Status:

`LOCKED`

---

## M6.1-D02 — Core evaluation population

Decision:

`VALIDATION 2019-01 → 2019-05`

Rows:

`712,458`

Fraud:

`1,052`

Status:

`INHERITED — LOCKED`

---

## M6.1-D03 — Positive class

Decision:

`fraud = 1`

Status:

`INHERITED — LOCKED`

---

## M6.1-D04 — Official evaluation subjects

Decision:

Six M5 official baseline runs only.

Status:

`LOCKED`

---

## M6.1-D05 — Independent metric reconstruction

Decision:

M6 phải tái tính metric từ:

`y_validation + persisted y_pred`

trước comparative interpretation.

Status:

`LOCKED`

---

## M6.1-D06 — Primary metric

Decision:

`F1_fraud`

Status:

`INHERITED — LOCKED`

---

## M6.1-D07 — Secondary metrics

Decision:

```text
Recall_fraud
Precision_fraud
```

Status:

`INHERITED — LOCKED`

---

## M6.1-D08 — Mandatory confusion evidence

Decision:

```text
TP
FP
FN
TN
```

Status:

`INHERITED — LOCKED`

---

## M6.1-D09 — Operational diagnostic

Decision:

```text
predicted_positive_count
predicted_positive_rate
```

Status:

`INHERITED — LOCKED`

---

## M6.1-D10 — Accuracy

Decision:

`REFERENCE ONLY`

Status:

`INHERITED — LOCKED`

---

## M6.1-D11 — Probability evidence

Decision:

Persisted positive-class risk scores được dùng cho descriptive analysis.

Threshold optimization:

`PROHIBITED IN M6`

Status:

`LOCKED`

---

## M6.1-D12 — Error-analysis primary groups

Decision:

```text
Fraud:
FN vs TP

Non-fraud:
FP vs TN
```

Status:

`LOCKED`

---

## M6.1-D13 — Lineage anchor

Decision:

`row_id_validation.npy`

là canonical anchor cho transaction-level error mapping.

Actual mapping:

`VERIFY IN M6.2 / M6.6`

Status:

`LOCKED AS CONTRACT`

---

## M6.1-D14 — Training-window comparison

Decision:

Only compare SHORT vs LONG within same model/config as controlled pair.

Status:

`INHERITED — LOCKED`

---

## M6.1-D15 — Cross-model comparison

Decision:

Model-family comparison phải giữ training window cố định.

Status:

`LOCKED`

---

## M6.1-D16 — Retraining

Decision:

No baseline retraining in core M6.

Status:

`LOCKED`

---

## M6.1-D17 — Tuning / imbalance intervention

Decision:

Not in core M6.

Status:

`DEFERRED TO M7`

---

## M6.1-D18 — Final threshold

Decision:

No threshold optimization in M6.

Status:

`OPEN — DEFERRED TO M7`

---

## M6.1-D19 — FINAL TEST

Decision:

No access.

Status:

`INHERITED — LOCKED`

---

## M6.1-D20 — Training-window winner

Decision:

M6.1 does not lock a winner.

Status:

`OPEN`

---

## M6.1-D21 — Model-family winner

Decision:

M6.1 does not lock a winner.

Status:

`OPEN`

---

## M6.1-D22 — Findings language

Decision:

Findings must distinguish:

```text
VERIFIED
OBSERVED
DESCRIPTIVE COMPARATIVE EVIDENCE
HYPOTHESIS / OPEN QUESTION
```

Status:

`LOCKED`

---

## M6.1-D23 — M6.2 handoff

Decision:

M6.2 must audit all persisted evaluation artifacts and independently reconstruct metrics.

Status:

`READY`

---

# 42. M6.1 Open Questions

## O01 — Do all persisted prediction artifacts reconstruct exactly?

Status:

`OPEN — M6.2`

---

## O02 — Do M6 recomputed metrics match M5 summaries?

Status:

`OPEN — M6.2`

---

## O03 — Which baseline produces the strongest/weakest fraud coverage patterns?

Status:

`OPEN — M6.3`

M6.3 may describe evidence.

Final selection:

`NOT M6.3`

---

## O04 — What is the within-family training-window behavior?

Status:

`OPEN — M6.4`

---

## O05 — What are same-window cross-family trade-offs?

Status:

`OPEN — M6.5`

---

## O06 — Which fraud transactions are systematically missed?

Status:

`OPEN — M6.6`

---

## O07 — Which non-fraud transactions are systematically false-alerted?

Status:

`OPEN — M6.6`

---

## O08 — Do error groups show repeatable semantic patterns?

Status:

`OPEN — M6.6`

---

## O09 — Does probability evidence suggest threshold/imbalance work is worth testing?

Status:

`OPEN — DESCRIPTIVE IN M6 / EXPERIMENT IN M7`

---

## O10 — Final training-window preference

Status:

`OPEN — M6 EVIDENCE / M7 SELECTION IF NEEDED`

---

## O11 — Final model-family preference

Status:

`OPEN — M6 EVIDENCE / M7 SELECTION`

---

## O12 — Final imbalance strategy

Status:

`OPEN — M7`

---

## O13 — Final hyperparameters

Status:

`OPEN — M7`

---

## O14 — Final threshold

Status:

`OPEN — M7`

---

# 43. M6.1 Gate

M6.1 là concept/protocol gate.

Không cần runtime model output mới.

## G01 — Milestone role rõ?

Requirement:

Evaluation/error analysis khác final selection.

Result:

`PASS`

---

## G02 — Canonical validation population rõ?

Requirement:

```text
2019-01 → 2019-05
712,458 rows
1,052 fraud
```

Result:

`PASS`

---

## G03 — Positive class rõ?

Requirement:

`fraud = 1`

Result:

`PASS`

---

## G04 — Official evaluation subjects rõ?

Requirement:

6 official M5 baseline runs.

Result:

`PASS`

---

## G05 — Artifact source contract rõ?

Requirement:

M4.7 target/lineage + M5 prediction/risk-score/summary/config/pair artifacts.

Result:

`PASS`

---

## G06 — Independent metric reconstruction requirement rõ?

Result:

`PASS`

---

## G07 — Metric strategy rõ?

Requirement:

F1 primary; Recall/Precision mandatory secondary; Accuracy reference.

Result:

`PASS`

---

## G08 — Confusion / alert diagnostics rõ?

Requirement:

TP/FP/FN/TN + predicted-positive count/rate.

Result:

`PASS`

---

## G09 — Error-analysis group definition rõ?

Requirement:

FN vs TP; FP vs TN.

Result:

`PASS`

---

## G10 — Lineage guardrail rõ?

Requirement:

transaction-level analysis must verify canonical row mapping.

Result:

`PASS`

---

## G11 — Training-window comparison rule rõ?

Requirement:

only training window changes within family.

Result:

`PASS`

---

## G12 — Cross-model comparison rule rõ?

Requirement:

same training window for model-family comparison.

Result:

`PASS`

---

## G13 — Probability/threshold boundary rõ?

Requirement:

descriptive probability analysis allowed; threshold optimization prohibited.

Result:

`PASS`

---

## G14 — No retraining/tuning boundary rõ?

Result:

`PASS`

---

## G15 — FINAL TEST isolation rõ?

Requirement:

`NO ACCESS`

Result:

`PASS`

---

## G16 — Findings vocabulary / overclaim guardrail rõ?

Result:

`PASS`

---

## G17 — M6.2 handoff rõ?

Requirement:

artifact audit + independent metric reconstruction.

Result:

`PASS`

---

# 44. M6.1 Gate summary

```text
G01 — Milestone role                         PASS
G02 — Validation population                  PASS
G03 — Positive class                         PASS
G04 — Evaluation subjects                    PASS
G05 — Artifact source contract               PASS
G06 — Independent metric reconstruction      PASS
G07 — Metric strategy                        PASS
G08 — Confusion / alert diagnostics          PASS
G09 — Error-analysis group definition        PASS
G10 — Lineage guardrail                      PASS
G11 — Training-window comparison             PASS
G12 — Cross-model comparison                 PASS
G13 — Probability / threshold boundary       PASS
G14 — No retraining / tuning boundary        PASS
G15 — FINAL TEST isolation                   PASS
G16 — Findings / overclaim guardrail         PASS
G17 — M6.2 handoff                           PASS
```

Overall:

`17 / 17 PASS`

Blocking issue:

`NONE`

M6.1:

`PASS`

---

# 45. Những điều M6.1 chưa chứng minh

M6.1 chưa chứng minh:

- six prediction files hiện tại vẫn còn đúng trên disk;
- six risk-score files hiện tại vẫn còn đúng trên disk;
- recomputed metrics khớp summary;
- lineage mapping implementation thực tế đúng;
- model nào có evaluation behavior phù hợp hơn;
- window nào có evidence tốt hơn;
- error groups tập trung ở feature pattern nào;
- probability distribution ra sao;
- model nào nên đi tiếp M7;
- class imbalance intervention nào tốt;
- hyperparameter nào tốt;
- threshold nào tốt.

Các câu hỏi runtime/evidence-dependent trên:

`REQUIRE M6.2 → M6.6`

M6.1 chỉ khóa protocol.

---

# 46. Handoff sang M6.2

M6.2 phải tạo:

`06_02_evaluation_artifact_audit.ipynb`

M6.2 không fit model.

M6.2 phải:

```text
locate M4.7 validation artifacts
        ↓
locate 6 M5 prediction artifacts
        ↓
locate 6 M5 risk-score artifacts
        ↓
locate 6 summaries
        ↓
locate config locks + pair manifests
        ↓
identity / shape / dtype / finite audit
        ↓
independent metric reconstruction
        ↓
confusion arithmetic
        ↓
compare recomputed vs persisted summaries
        ↓
audit row_id_validation
        ↓
FINAL TEST isolation
        ↓
persist independent Evaluation Registry
        ↓
round-trip
        ↓
runtime review
```

M6.2 expected terminal state sau runtime review:

```text
6 / 6 prediction artifacts:
VERIFIED

6 / 6 probability artifacts:
VERIFIED

6 / 6 independent metrics:
RECONSTRUCTED

6 / 6 summary matches:
VERIFIED

Validation lineage:
VERIFIED

FINAL TEST:
PROTECTED

Blocking issue:
NONE

M6.2:
PASS

READY FOR M6.3
```

---

# 47. Source hierarchy

M6.1 kế thừa trực tiếp:

- `CANON-Kế hoạch Milestone 6 — Evaluation và Error Analysis.md`
- `CANON-M5.6 — Tổng hợp Baseline Model Registry, Decision Log và M5 Gate.md`
- `CANON-M5.1 — Khóa Modeling Charter và baseline protocol.md`
- `CANON-Kế hoạch Milestone 5 — Modeling baseline.md`
- `CANON-M3.4 — Thiết kế evaluation metric strategy.md`
- `CANON-M3.6 — Thiết kế protocol so sánh training window.md`
- `CANON-M3.7 — Policy cho class imbalance, temporal validation/CV và các experiment/tuning sau.md`
- `CANON-M3.8 — Tổng hợp Experiment Specification v1.0, Decision Log, Open Questions và M3 Gate.md`

Project-specific temporal/evaluation protocol có ưu tiên cao hơn generic ML example.

---

# 48. Kết luận M6.1

M6.1 đã khóa Evaluation Charter cho Milestone 6.

```text
Milestone:
M6 — EVALUATION + ERROR ANALYSIS

Core population:
VALIDATION 2019-01 → 2019-05

Rows:
712,458

Fraud:
1,052

Positive class:
fraud = 1

Official subjects:
6 M5 baseline runs

Primary metric:
F1_fraud

Secondary:
Recall_fraud
Precision_fraud

Mandatory:
TP / FP / FN / TN

Operational:
predicted-positive count/rate

Accuracy:
REFERENCE ONLY

Independent metric reconstruction:
REQUIRED

Error analysis:
FN vs TP
FP vs TN

Training-window comparison:
CONTROLLED WITHIN MODEL FAMILY

Cross-model comparison:
SAME TRAINING WINDOW

Probability analysis:
DESCRIPTIVE ALLOWED

Threshold optimization:
PROHIBITED IN M6

Retraining:
NO

Tuning:
NO

Imbalance intervention:
NO

FINAL TEST:
PROTECTED / NO ACCESS

Final model:
OPEN

Final training window:
OPEN

M6.1 Gate:
17 / 17 PASS

Blocking issue:
NONE

M6.1:
PASS

Next:
M6.2 — Evaluation artifact audit + independent metric reconstruction
```
