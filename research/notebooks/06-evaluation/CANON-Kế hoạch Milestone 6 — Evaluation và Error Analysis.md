# CANON-Kế hoạch Milestone 6 — Evaluation và Error Analysis

## Document status

Milestone:

`M6 — Evaluation + Error Analysis`

Work type:

`EVALUATION / COMPARATIVE ANALYSIS / ERROR ANALYSIS`

Input milestone:

`M5 — PASS`

Handoff từ M5:

`READY FOR M6 — EVALUATION + ERROR ANALYSIS`

M6 bắt đầu từ sáu official baseline runs đã được runtime-review:

```text
M5-LR-SHORT-B04
M5-LR-LONG-B04

M5-DT-SHORT-B01
M5-DT-LONG-B01

M5-RF-SHORT-B01
M5-RF-LONG-B01
```

M6 không cần train lại các baseline chỉ để đánh giá.

M6 không mở FINAL TEST.

---

# 1. Câu hỏi trung tâm của Milestone 6

M5 đã trả lời:

> Các baseline model có thể fit/predict hợp lệ trên canonical modeling representation hay không?

M6 chuyển sang câu hỏi khác:

> Những prediction đã tồn tại đúng/sai như thế nào, các baseline khác nhau ra sao, fraud bị bỏ sót ở đâu, false alert xuất hiện ở đâu, và evidence nào cần được handoff sang M7?

M6 phải biến:

`prediction artifacts`

thành:

`evaluation evidence`

và tiếp tục biến:

`aggregate metrics`

thành:

`error-pattern evidence`.

---

# 2. Vai trò chính thức của M6

M6 nằm giữa:

```text
M5 — Modeling baseline
        ↓
M6 — Evaluation + Error Analysis
        ↓
M7 — Model Selection / CV / tuning / imbalance / threshold
```

M6 tập trung vào:

- xác minh prediction/evaluation artifacts;
- tái tính metric độc lập từ persisted predictions;
- đọc Confusion Matrix;
- phân tích Precision / Recall / F1;
- đọc TP / FP / FN / TN;
- đọc predicted-positive count/rate;
- so sánh W_SHORT với W_LONG trong controlled setup;
- so sánh model families trên cùng training-window population;
- phân tích false negative;
- phân tích false positive;
- đọc probability/risk-score behavior ở mức mô tả;
- tổng hợp comparative findings;
- xác định câu hỏi cần M7 giải quyết.

M6 là:

`DESCRIPTIVE / COMPARATIVE EVALUATION MILESTONE`

M6 chưa phải:

`FINAL MODEL SELECTION MILESTONE`

---

# 3. Ranh giới M6 với M7

## M6 được phép

- load M5 prediction artifacts;
- load M5 probability/risk-score artifacts;
- load validation target;
- load validation lineage / row mapping khi cần cho error analysis;
- tái tính F1 / Recall / Precision / Accuracy reference;
- tái tạo Confusion Matrix;
- phân tích TP / FP / FN / TN;
- phân tích predicted-positive count/rate;
- mô tả probability distribution/ranking behavior;
- so sánh baseline runs;
- phân tích error groups;
- ghi findings;
- ghi trade-off;
- ghi computational evidence đã có từ M5;
- xác định candidate questions cho M7.

## M6 không được phép

- retrain baseline để cải thiện metric;
- thử nhiều `C`;
- thử nhiều `max_depth`;
- thử nhiều `n_estimators`;
- class-weight experiment;
- oversampling;
- undersampling;
- SMOTE;
- temporal CV để chọn config;
- threshold optimization;
- calibration experiment có tính lựa chọn;
- chọn final model;
- chọn final training window;
- mở FINAL TEST;
- tuyên bố final project performance.

Những intervention/selection trên thuộc M7 hoặc bước sau.

---

# 4. Căn cứ kế thừa từ M3

M6 không thiết kế lại evaluation protocol.

## 4.1. Problem

```text
Binary supervised classification
Positive class = fraud = 1
Prediction point = transaction screening time
```

Model được hiểu là:

`transaction risk-screening component`

không phải hệ thống tự động ra quyết định chặn giao dịch ngoài đời.

---

## 4.2. Temporal evaluation

Primary evaluation direction:

`PAST → FUTURE`

Canonical development validation:

```text
2019-01-01 <= Timestamp < 2019-06-01
```

FINAL TEST:

```text
2019-06-01 <= Timestamp < 2019-11-01
```

M6:

`VALIDATION ONLY`

FINAL TEST:

`PROTECTED`

---

## 4.3. Primary metric

`F1_fraud`

Vai trò:

metric tổng hợp để đọc cân bằng giữa fraud Recall và fraud Precision.

M6 không được đọc F1 một mình.

---

## 4.4. Mandatory secondary metrics

```text
Recall_fraud
Precision_fraud
```

Recall trả lời:

> Trong toàn bộ fraud thật, model bắt được bao nhiêu?

Precision trả lời:

> Trong toàn bộ transaction model flag positive, bao nhiêu thật sự là fraud?

Hai metric phải được đọc cùng nhau.

---

## 4.5. Mandatory diagnostic

Confusion Matrix:

```text
TP
FP
FN
TN
```

M6 phải luôn giữ raw count vì dataset mất cân bằng mạnh.

Một thay đổi metric nhỏ có thể tương ứng với khác biệt đáng kể về số fraud bị bỏ sót hoặc số false alert.

---

## 4.6. Operational diagnostic

```text
predicted_positive_count
predicted_positive_rate
```

Hai đại lượng này giúp đọc:

> Baseline đang đưa bao nhiêu transaction vào nhóm cần chú ý?

Chúng không thay thế F1/Recall/Precision.

---

## 4.7. Accuracy

`REFERENCE ONLY`

Accuracy không được dùng làm tiêu chí chính vì negative class chiếm đa số rất lớn.

---

# 5. Căn cứ kế thừa từ M5

M5 đã khóa và verified:

```text
Canonical artifacts:
M4.7-baseline-v1

Feature count:
47

Matrix:
CSR float32

Target:
int8
fraud = 1

Imbalance:
NONE

Threshold policy:
DEFAULT_MODEL_DECISION_RULE

Official baseline runs:
6 / 6 VALID

Prediction evidence:
6 / 6 PERSISTED

Probability evidence:
6 / 6 PERSISTED

FINAL TEST:
PROTECTED
```

M6 không được thay đổi các facts này khi chỉ thực hiện evaluation.

---

# 6. Evaluation population

Core M6 evaluation population:

`VALIDATION 2019-01 → 2019-05`

Rows:

`712,458`

Fraud:

`1,052`

Natural class distribution:

`PRESERVED`

M6 không resample validation.

M6 không rebalance validation.

M6 không tạo synthetic validation rows.

---

# 7. Official M6 evaluation subjects

M6 đánh giá sáu official runs:

## Logistic Regression

```text
M5-LR-SHORT-B04
M5-LR-LONG-B04
```

Official config:

`LR-B04-LBFGS-L2-C1`

---

## Decision Tree

```text
M5-DT-SHORT-B01
M5-DT-LONG-B01
```

Official config:

`DT-B01-DEFAULT-GINI-UNPRUNED`

---

## Random Forest

```text
M5-RF-SHORT-B01
M5-RF-LONG-B01
```

Official config:

`RF-B01-100-GINI-SQRT-BOOTSTRAP`

---

# 8. Canonical evaluation artifact contract

M6 phải dùng persisted M5 evidence, không copy metric bằng tay rồi coi đó là đủ.

Required inputs:

```text
M4.7:
y_validation.npy
row_id_validation.npy
feature_names.json
manifest.json

M5.3:
2 y_pred arrays
2 risk_score arrays
2 summary JSON
config lock
pair manifest

M5.4:
2 y_pred arrays
2 risk_score arrays
2 summary JSON
config lock
pair manifest

M5.5:
2 y_pred arrays
2 risk_score arrays
2 summary JSON
config lock
pair manifest
```

Nếu error analysis cần transaction-level semantic fields:

- phải map qua canonical validation row identity;
- mapping phải được audit;
- không được dùng positional assumption không được kiểm tra;
- không được đưa FINAL TEST row vào evaluation frame.

---

# 9. Nguyên tắc independent evaluation

M6 phải tái tính canonical metrics từ:

```text
y_validation
+
persisted y_pred
```

Không được chỉ tin summary JSON.

Mục đích:

- phát hiện artifact mismatch;
- phát hiện stale prediction;
- xác minh confusion counts;
- xác minh metric arithmetic;
- tạo independent evaluation registry.

M6.2 phải chứng minh:

```text
recomputed metrics
==
M5 persisted metrics
```

trong tolerance hợp lý.

Nếu mismatch:

`STOP`

→ xác định mismatch source

→ không tiếp tục comparative conclusion cho run đó cho đến khi resolved.

---

# 10. Error-analysis unit

Core error analysis làm việc ở transaction level trên VALIDATION.

Mỗi run phân partition thành:

```text
TP
FP
FN
TN
```

Primary error groups:

```text
FN — fraud bị bỏ sót
FP — non-fraud bị flag nhầm
```

Reference groups:

```text
TP — fraud được bắt đúng
TN — non-fraud được bỏ qua đúng
```

Primary comparison cho fraud-side error:

`FN vs TP`

vì hai nhóm đều là actual fraud nhưng khác nhau ở khả năng model phát hiện.

Primary comparison cho non-fraud-side error:

`FP vs TN`

vì hai nhóm đều là actual non-fraud nhưng khác nhau ở việc model tạo false alert.

---

# 11. Feature scope cho error analysis

M6 error analysis ưu tiên các feature/semantic quantities đã được authorized trong baseline representation.

Core feature families:

```text
Amount:
amount_numeric

History / recency:
time_since_previous_transaction_min
transactions_last_1h
amount_minus_previous_mean

Cold-start / merchant behavior:
is_new_merchant
has_prior_card_history

Transaction semantics:
transaction_mode
location_state
hour_of_day
day_of_week
```

M6 có thể dùng representation dễ diễn giải tương ứng nếu mapping lineage được verified.

M6 không được:

- tạo future-dependent feature mới;
- dùng target leakage;
- dùng FINAL TEST;
- tự mở một feature-engineering experiment mới rồi trộn vào baseline error analysis.

Nếu một derived diagnostic cần thêm transformation:

- phải là evaluation-only;
- không được thay prediction;
- phải ghi rõ derivation;
- không được gọi nó là model feature nếu model chưa sử dụng.

---

# 12. Probability / risk-score analysis scope

M5 đã persist positive-class probability/risk score cho cả 6 runs.

M6 được phép phân tích mô tả:

- score distribution overall;
- score distribution fraud vs non-fraud;
- score distribution TP / FP / FN / TN;
- score concentration gần default decision boundary;
- ranking behavior;
- overlap giữa fraud và non-fraud score;
- count/rate ở một số descriptive score bands nếu declared trước analysis.

M6 không được:

- search threshold để tối đa F1;
- chọn operating threshold mới;
- thay threshold giữa runs rồi gọi là fair baseline comparison;
- dùng FINAL TEST để chọn threshold.

Final threshold:

`DEFERRED TO M7`

---

# 13. Controlled training-window comparison

M3.6 đã khóa:

`ONLY TRAINING WINDOW CHANGES`

M6 phải giữ comparison theo family:

```text
LR-SHORT vs LR-LONG

DT-SHORT vs DT-LONG

RF-SHORT vs RF-LONG
```

Đọc tối thiểu:

```text
ΔF1
ΔRecall
ΔPrecision
ΔTP
ΔFP
ΔFN
ΔTN
Δpredicted-positive count
Δpredicted-positive rate
```

Không được kết luận:

`W_SHORT always better`

hoặc:

`W_LONG always better`

chỉ từ một model family.

Interpretation phải scope theo model/config cụ thể.

Ví dụ formulation hợp lệ:

> Trong RF-B01 trên canonical VALIDATION, W_SHORT tạo Recall cao hơn W_LONG ở baseline decision rule.

Không được mở rộng ngay thành:

> W_SHORT luôn là training window tốt hơn cho mọi model.

---

# 14. Cross-model comparison principle

Model-family comparison phải giữ training-window population cố định.

Valid comparisons:

```text
LR-SHORT vs DT-SHORT vs RF-SHORT
```

và:

```text
LR-LONG vs DT-LONG vs RF-LONG
```

Không dùng comparison như:

```text
LR-SHORT vs RF-LONG
```

để kết luận thuần về model family vì lúc đó model family và training window cùng thay đổi.

Cross-model analysis đọc:

- fraud F1;
- fraud Recall;
- fraud Precision;
- TP/FP/FN/TN;
- predicted-positive behavior;
- probability behavior;
- computational evidence từ M5;
- error-pattern overlap/difference.

M6 có thể mô tả trade-off.

M6 chưa khóa final model winner.

---

# 15. Không tạo arbitrary winner rule ở M6

M6 không tự tạo rule kiểu:

```text
model có F1 cao nhất
→ final winner
```

vì M7 vẫn phải xử lý:

- temporal robustness;
- CV nếu cần;
- tuning;
- imbalance interventions;
- threshold;
- robustness checks.

M6 có thể tạo:

`EVALUATION FINDING`

không phải:

`FINAL SELECTION DECISION`

---

# 16. Proposed M6 work breakdown

Phân rã M6 dưới đây được khóa như execution plan của milestone.

## M6.1 — Evaluation Charter, scope và guardrails

### Câu hỏi

> M6 được phép đánh giá gì, không được phép thay đổi gì, và evidence nào là canonical?

### Nhiệm vụ

Khóa:

- input artifact contract;
- evaluation population;
- metric semantics;
- comparison rules;
- error-analysis scope;
- probability-analysis boundary;
- FINAL TEST prohibition;
- M6/M7 boundary;
- M6 output schema.

### Output

`CANON-M6.1 — Khóa Evaluation Charter, scope và guardrails.md`

### Gate

PASS khi mọi notebook M6 sau có thể biết chính xác:

- input nào được load;
- partition nào được đọc;
- metric nào được tái tính;
- comparison nào hợp lệ;
- M6 được phép kết luận đến đâu;
- FINAL TEST có quyền gì.

---

## M6.2 — Evaluation artifact audit và independent metric reconstruction

### Câu hỏi

> Persisted prediction/probability artifacts từ M5 có đủ integrity để evaluation độc lập không?

### Input

- six y_pred;
- six risk_score;
- six summaries;
- config locks;
- pair manifests;
- y_validation;
- row_id_validation.

### Audit bắt buộc

- file existence;
- exact row count;
- dtype;
- finite;
- binary prediction support;
- probability range;
- experiment identity;
- summary identity;
- final-test flag;
- confusion arithmetic;
- independent metric recomputation;
- equality/tolerance với persisted summary.

### Không thực hiện

- model fit;
- tuning;
- threshold selection;
- error interpretation sâu.

### Output

`06_02_evaluation_artifact_audit.ipynb`

### Gate

PASS khi:

`6 / 6 runs independently reconstructable`

và không có metric/artifact mismatch.

---

## M6.3 — Baseline metric và Confusion Matrix analysis

### Câu hỏi

> Mỗi baseline đang phát hiện fraud và mắc lỗi ở mức aggregate như thế nào?

### Nhiệm vụ

Với từng run:

- F1;
- Recall;
- Precision;
- Accuracy reference;
- TP;
- FP;
- FN;
- TN;
- predicted-positive count/rate;
- fraud captured count/rate;
- fraud missed count/rate;
- false-alert burden.

### Interpretation rule

Mọi metric phải được diễn giải cùng raw count.

Không ghi:

> Recall thấp.

mà cần ghi:

> Recall = X tương ứng TP = A và FN = B trên 1,052 fraud validation.

### Output

`06_03_baseline_metric_and_confusion_analysis.ipynb`

### Gate

PASS khi cả 6 runs có canonical metric/confusion interpretation hoàn chỉnh.

---

## M6.4 — Controlled training-window comparison

### Câu hỏi

> Trong từng model family, chỉ thay training history thì validation behavior thay đổi ra sao?

### Pairs

```text
LR SHORT vs LONG
DT SHORT vs LONG
RF SHORT vs LONG
```

### Nhiệm vụ

Tính/mô tả:

- metric deltas;
- confusion-count deltas;
- predicted-positive deltas;
- precision-recall trade-off;
- computational context;
- probability behavior difference nếu cần.

### Không được

- tune riêng từng window;
- chọn threshold riêng;
- dùng final test;
- generalize winner ngoài scope evidence.

### Output

`06_04_training_window_comparison.ipynb`

### Gate

PASS khi 3 controlled pairs có findings rõ và scope đúng.

---

## M6.5 — Cross-model comparative evaluation

### Câu hỏi

> Trên cùng training-window population, LR / DT / RF thể hiện trade-off khác nhau như thế nào?

### Primary comparison groups

```text
W_SHORT:
LR vs DT vs RF

W_LONG:
LR vs DT vs RF
```

### Nhiệm vụ

So sánh:

- F1;
- Recall;
- Precision;
- TP/FP/FN;
- alert volume;
- probability behavior;
- computational evidence.

### Output

`06_05_cross_model_comparative_evaluation.ipynb`

### Gate

PASS khi cross-family evidence được mô tả mà không khóa final model.

---

## M6.6 — False Positive / False Negative error analysis

### Câu hỏi

> Fraud bị bỏ sót và false alert có pattern gì trên canonical validation population?

### Primary groups

```text
Fraud-side:
FN vs TP

Non-fraud-side:
FP vs TN
```

### Phân tích candidate

Theo các feature/semantic dimensions đã authorized:

- Amount;
- transaction mode;
- location state;
- hour;
- day of week;
- new merchant;
- prior-card history;
- transaction recency;
- recent transaction count;
- amount deviation from historical mean.

### Cross-run analysis

Có thể kiểm tra:

- fraud nào bị cả 3 family bỏ sót;
- fraud nào một model bắt được nhưng model khác bỏ sót;
- FP overlap;
- model-specific error subsets.

### Probability context

Đọc score của FN/FP để phân biệt:

- near-boundary error;
- strongly confident error;
- potential ranking issue.

Không thay threshold.

### Output

`06_06_fp_fn_error_analysis.ipynb`

### Gate

PASS khi:

- lineage mapping verified;
- FP/FN groups reconstructed correctly;
- findings dựa trên actual transactions;
- không có leakage;
- không có FINAL TEST access.

---

## M6.7 — Consolidated Evaluation Registry, Findings, Decision Log và M6 Gate

### Câu hỏi

> M6 đã hiểu đủ baseline behavior và error pattern để handoff M7 chưa?

### Tổng hợp

- Evaluation Registry;
- six-run canonical metrics;
- confusion findings;
- training-window findings;
- cross-model findings;
- FP findings;
- FN findings;
- probability findings;
- computational observations;
- open questions;
- M6 Decision Log;
- M6 Gate;
- M7 handoff.

### Output

`CANON-M6.7 — Tổng hợp Evaluation Registry, Error Findings, Decision Log và M6 Gate.md`

### Không khóa

- final model;
- final training window;
- final imbalance strategy;
- final hyperparameters;
- final threshold;
- final-test performance.

---

# 17. Runtime execution protocol cho M6

M6.2–M6.6 là evidence-dependent analysis.

Workflow chuẩn:

```text
1. Xác định câu hỏi analysis
        ↓
2. Đọc CANON liên quan
        ↓
3. Khóa input + comparison scope trước output
        ↓
4. Tạo notebook với assertions/gates
        ↓
5. Người thực hiện Run All trên environment thật
        ↓
6. Giữ nguyên output/error/warning
        ↓
7. AI review execution/integrity/consistency/leakage
        ↓
8. Nếu thiếu evidence:
   thêm check → rerun
        ↓
9. Chỉ sau runtime review:
   viết Findings / Decision Log / Gate
        ↓
10. M6.7 consolidation
```

Không invent output.

Không PASS runtime-dependent substep khi notebook chưa chạy thực tế.

---

# 18. Canonical M6 analysis registry schema

Mỗi evaluated run tối thiểu có:

## Identity

- experiment_id;
- model_family;
- model_config_id;
- training_window_id.

## Evaluation population

- validation_rows;
- validation_fraud_rows;
- validation_period;
- target positive class.

## Metric bundle

- F1_fraud;
- Recall_fraud;
- Precision_fraud;
- Accuracy_reference.

## Confusion evidence

- TP;
- FP;
- FN;
- TN.

## Operational evidence

- predicted_positive_count;
- predicted_positive_rate;
- fraud_capture_count;
- fraud_miss_count.

## Probability evidence

- risk_score_available;
- score finite/range;
- descriptive score summaries khi dùng.

## Artifact evidence

- prediction source;
- risk-score source;
- validation target source;
- row lineage source.

## Integrity

- independent metric reproduction;
- summary match;
- final_test_accessed = false;
- evaluation gate result.

## Interpretation

- finding;
- limitation;
- next question;
- M7 relevance nếu có.

---

# 19. M6 findings classification

Để tránh overclaim, mỗi finding phải gắn một trong các scope sau.

## VERIFIED

Integrity/arithmetic/artifact fact đã được kiểm tra.

Ví dụ:

`TP + FN = 1,052`

---

## OBSERVED

Pattern trực tiếp trên validation data.

Ví dụ:

> RF-LONG phát ít positive hơn RF-SHORT ở baseline rule.

---

## DESCRIPTIVE COMPARATIVE EVIDENCE

Comparison hợp lệ theo controlled protocol.

Ví dụ:

> Trong DT-B01, W_SHORT có Recall cao hơn W_LONG trên canonical validation.

---

## HYPOTHESIS FOR M7

Giải thích chưa được causal/protocol evidence chứng minh.

Ví dụ:

> Temporal nonstationarity có thể góp phần làm W_LONG kém hơn.

Nếu không có experiment tách nguyên nhân:

không nâng hypothesis thành conclusion.

---

# 20. Error-analysis guardrails

M6 error analysis không được biến thành data dredging không kiểm soát.

Phải:

- khai báo comparison group;
- dùng canonical validation population;
- giữ lineage;
- ưu tiên feature dimensions đã được project authorize;
- phân biệt descriptive association với causal explanation;
- ghi denominator;
- báo cả count và rate khi class/group rất nhỏ;
- cảnh báo khi subgroup sample quá ít.

Không được:

- nhìn một vài anecdotal rows rồi generalize;
- tạo hàng chục subgroup rồi chỉ báo subgroup đẹp nhất;
- dùng target/future field như một explanatory feature;
- mở FINAL TEST để “xác nhận pattern”.

---

# 21. Probability-analysis guardrails

M6 probability analysis là descriptive.

Được phép:

```text
score distribution
score quantiles
fraud/non-fraud separation
TP/FP/FN/TN score summaries
near-boundary counts
```

Không được:

```text
search optimal threshold
maximize F1 over threshold grid
select threshold per model
calibrate then compare as baseline
```

Threshold experiment:

`M7`

---

# 22. Computational evidence trong M6

M5 đã ghi runtime:

- fit time;
- prediction time;
- model complexity diagnostics.

M6 được phép giữ computational evidence như một dimension mô tả.

Không được:

> model chạy nhanh nhất → model tốt nhất

Runtime là một consideration cho later selection, không thay predictive evidence.

M6.7 phải handoff computational facts sang M7 nếu chúng material.

---

# 23. Training-window preference ở M6

M3.6 cho phép evaluation evidence làm rõ W_SHORT/W_LONG.

Tuy nhiên M6 không được tự tạo arbitrary epsilon để ép winner.

Nếu evidence mixed:

`INCONCLUSIVE`

Nếu một window có F1 advantage nhưng Recall/Precision trade-off material:

phải ghi trade-off.

Nếu selection cần robustness:

handoff M7 temporal robustness/CV.

Parsimony fallback W_SHORT chỉ được dùng theo protocol khi predictive evidence/robustness không phân biệt được và project thực sự cần một fallback.

M6 không tự động áp dụng fallback trước khi M7 xem xét robustness requirement.

---

# 24. Model-family preference ở M6

M6 có thể nói:

```text
On validation:
run A has higher F1 than run B
run B has higher Recall
run C produces fewer false alerts
```

M6 không được nói:

```text
Final model = A
```

Final model selection cần xem:

- comparative evidence;
- robustness;
- temporal CV nếu cần;
- tuning response;
- imbalance intervention;
- threshold;
- computational constraints.

Những nội dung đó thuộc M7.

---

# 25. Proposed M6 outputs

```text
CANON-Kế hoạch Milestone 6 — Evaluation và Error Analysis.md

CANON-M6.1 — Khóa Evaluation Charter, scope và guardrails.md

06_02_evaluation_artifact_audit.ipynb

06_03_baseline_metric_and_confusion_analysis.ipynb

06_04_training_window_comparison.ipynb

06_05_cross_model_comparative_evaluation.ipynb

06_06_fp_fn_error_analysis.ipynb

CANON-M6.7 — Tổng hợp Evaluation Registry, Error Findings, Decision Log và M6 Gate.md
```

Nếu implementation sau gộp một số notebook nhưng giữ đầy đủ semantic/gates thì vẫn hợp lệ.

---

# 26. M6 Gate

M6 chỉ PASS khi ít nhất các gate sau đạt.

## G01 — M5 handoff integrity

PASS khi:

- six official runs được xác định đúng;
- persisted artifacts đúng source/version;
- M5 registry không bị thay bằng stale experiment.

---

## G02 — Prediction artifact integrity

PASS khi:

`6 / 6 y_pred`

load thành công, đúng length/dtype/support và identity.

---

## G03 — Risk-score artifact integrity

PASS khi:

`6 / 6 risk_score`

load thành công, đúng length, finite và hợp lệ.

---

## G04 — Independent metric reproduction

PASS khi canonical metrics được tái tính từ:

`y_validation + y_pred`

và khớp M5 summary.

---

## G05 — Confusion Matrix reconstruction

PASS khi:

`6 / 6`

TP/FP/FN/TN được reconstruct và arithmetic consistency PASS.

---

## G06 — Predicted-positive analysis

PASS khi alert count/rate được đọc cho mọi run và diễn giải theo screening context.

---

## G07 — Controlled training-window analysis

PASS khi đủ:

```text
LR SHORT vs LONG
DT SHORT vs LONG
RF SHORT vs LONG
```

với đúng comparison scope.

---

## G08 — Cross-model comparative analysis

PASS khi đủ:

```text
W_SHORT:
LR vs DT vs RF

W_LONG:
LR vs DT vs RF
```

không confound training-window effect.

---

## G09 — False-negative analysis

PASS khi FN được map về validation transactions và có evidence-based findings.

---

## G10 — False-positive analysis

PASS khi FP được map về validation transactions và có evidence-based findings.

---

## G11 — Probability behavior review

PASS khi persisted score evidence được audit và ít nhất descriptive behavior được tổng hợp mà không threshold-optimize.

---

## G12 — No retraining / no hidden tuning

PASS khi M6 core evaluation không thay model configuration hoặc train model để săn evaluation score.

---

## G13 — FINAL TEST isolation

PASS khi:

`FINAL TEST ACCESS = NO`

trong toàn M6.

---

## G14 — Evaluation Findings + Decision Log complete

PASS khi findings phân biệt rõ:

- verified fact;
- observed pattern;
- comparative evidence;
- hypothesis/open question.

---

## G15 — M7 handoff explicit

PASS khi M7 nhận được:

- unresolved training-window question;
- unresolved model-family question;
- imbalance question;
- tuning question;
- threshold question;
- robustness/CV question;
- relevant error findings;
- relevant computational constraints.

---

# 27. Những điều M6 Gate không yêu cầu

M6 PASS không yêu cầu:

- tìm được một model “tốt nhất”;
- training-window winner phải được khóa;
- imbalance strategy winner phải được khóa;
- threshold phải được tối ưu;
- hyperparameter tuning phải hoàn thành;
- FINAL TEST phải được mở;
- tất cả fraud phải được giải thích bằng một vài subgroup đơn giản.

M6 PASS yêu cầu:

`evaluation evidence is complete, coherent and M7-ready`.

---

# 28. M6 Open Questions

## O01 — Final training-window preference

Status:

`OPEN`

M6 cung cấp comparative evidence.

M7 quyết định có cần robustness/CV/tie-break protocol hay không.

---

## O02 — Final model-family preference

Status:

`OPEN`

M6 cung cấp model behavior/error trade-off.

M7 thực hiện selection.

---

## O03 — Class imbalance intervention

Status:

`OPEN — M7`

Candidates authorized upstream:

```text
NONE
CLASS_WEIGHT
RANDOM_OVERSAMPLING
RANDOM_UNDERSAMPLING
SMOTE — CONDITIONAL
```

---

## O04 — Temporal CV / robustness

Status:

`DEFERRED TO M7`

Primary strategy nếu dùng:

`FORWARD / EXPANDING TEMPORAL VALIDATION`

---

## O05 — Hyperparameter tuning

Status:

`DEFERRED TO M7`

Search space phải nhỏ, có lý do và khai báo trước.

---

## O06 — Final numerical threshold

Status:

`DEFERRED TO M7`

FINAL TEST không được dùng để chọn threshold.

---

## O07 — Probability calibration

Status:

`OPEN / NOT CORE M6`

Chỉ xem xét ở M7 hoặc sau nếu evidence cho thấy cần và protocol cho phép.

---

## O08 — Final project performance

Status:

`OPEN`

FINAL TEST vẫn chưa được mở.

---

# 29. M6 completion definition

M6 được coi là hoàn thành khi project có thể trả lời chắc chắn:

> Six M5 prediction artifacts có evaluation integrity không?

> Metrics có tái tạo độc lập được không?

> Mỗi model đang bỏ sót bao nhiêu fraud?

> Mỗi model tạo bao nhiêu false alert?

> W_SHORT/W_LONG khác nhau ra sao trong từng family?

> LR/DT/RF khác nhau ra sao khi giữ training window cố định?

> FN có pattern nào đáng chú ý?

> FP có pattern nào đáng chú ý?

> Probability/risk-score behavior có cung cấp thêm insight không?

> Có evidence nào gợi ý M7 cần class-weight/resampling/tuning/threshold work không?

> FINAL TEST có được bảo vệ hoàn toàn không?

Nếu tất cả evidence cần thiết đã có:

`M6 — PASS`

Handoff:

`READY FOR M7 — MODEL SELECTION / ROBUSTNESS / TUNING`

---

# 30. Execution sequence

```text
M5.6
Baseline Modeling Registry
M5 PASS
        ↓
M6.1
Evaluation Charter
        ↓
M6.2
Evaluation Artifact Audit
Independent Metric Reconstruction
        ↓
M6.3
Metric + Confusion Matrix Analysis
        ↓
M6.4
Training-window Comparison
        ↓
M6.5
Cross-model Comparative Evaluation
        ↓
M6.6
FP/FN Error Analysis
Probability Behavior
        ↓
M6.7
Evaluation Registry
Findings
Decision Log
M6 Gate
        ↓
M7
Model Selection
Temporal Robustness / CV
Tuning
Imbalance
Threshold
```

---

# 31. Initial M6 Decision Log

## M6-D01 — Milestone role

Decision:

M6 là:

`EVALUATION + ERROR ANALYSIS`

không phải final model selection.

Status:

`LOCKED`

---

## M6-D02 — Evaluation population

Decision:

Core M6 dùng:

`VALIDATION 2019-01 → 2019-05`

Status:

`INHERITED — LOCKED`

---

## M6-D03 — FINAL TEST

Decision:

Không access trong M6.

Status:

`INHERITED — LOCKED`

---

## M6-D04 — Metric contract

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

Accuracy:
reference only
```

Status:

`INHERITED — LOCKED`

---

## M6-D05 — Independent metric reconstruction

Decision:

M6 phải tái tính metric từ persisted predictions.

Status:

`LOCKED`

---

## M6-D06 — Training-window comparison

Decision:

Within-family pair chỉ được đọc như controlled comparison khi model config giống nhau.

Status:

`INHERITED — LOCKED`

---

## M6-D07 — Cross-model comparison

Decision:

Model family comparison phải giữ training-window population cố định.

Status:

`LOCKED`

---

## M6-D08 — Error-analysis primary groups

Decision:

```text
Fraud side:
FN vs TP

Non-fraud side:
FP vs TN
```

Status:

`LOCKED`

---

## M6-D09 — Probability analysis

Decision:

Descriptive score analysis được phép.

Threshold optimization:

`NOT AUTHORIZED IN M6`

Status:

`LOCKED`

---

## M6-D10 — Retraining/tuning

Decision:

Core M6 không retrain baseline và không tuning.

Status:

`LOCKED`

---

## M6-D11 — Final model/window

Decision:

M6 không khóa final model hoặc final training window.

Status:

`OPEN — HANDOFF M7`

---

## M6-D12 — M7 handoff

Decision:

M6 phải handoff comparative + error evidence, không chỉ một bảng metric.

Status:

`LOCKED`

---

# 32. Source hierarchy

Kế hoạch M6 này kế thừa trực tiếp:

- `CANON-M5.6 — Tổng hợp Baseline Model Registry, Decision Log và M5 Gate.md`
- `CANON-M5.1 — Khóa Modeling Charter và baseline protocol.md`
- `CANON-Kế hoạch Milestone 5 — Modeling baseline.md`
- `CANON-M3.4 — Thiết kế evaluation metric strategy.md`
- `CANON-M3.6 — Thiết kế protocol so sánh training window.md`
- `CANON-M3.7 — Policy cho class imbalance, temporal validation/CV và các experiment/tuning sau.md`
- `CANON-M3.8 — Tổng hợp Experiment Specification v1.0, Decision Log, Open Questions và M3 Gate.md`
- `Tai_lieu_hoc_Tuan_8_Danh_gia_Classification.md`

Quy tắc project-specific có ưu tiên cao hơn ví dụ generic trong tài liệu học.

Đặc biệt:

- random split không thay thế temporal protocol;
- Accuracy không thay thế fraud metrics;
- M6 không thay thế M7;
- FINAL TEST vẫn protected.

---

# 33. Final planning block

```text
Milestone:
M6 — EVALUATION + ERROR ANALYSIS

Input:
M5 official baseline evidence

Runs:
6

Evaluation population:
VALIDATION 2019-01 → 2019-05

Primary metric:
F1_fraud

Secondary:
Recall_fraud
Precision_fraud

Mandatory:
TP / FP / FN / TN

Operational:
predicted-positive count/rate

Probability:
DESCRIPTIVE ANALYSIS ALLOWED

Error analysis:
FN vs TP
FP vs TN

Window comparison:
CONTROLLED WITHIN MODEL FAMILY

Model comparison:
SAME TRAINING WINDOW ONLY

Retraining:
NO

Tuning:
NO

Imbalance intervention:
NO

Threshold optimization:
NO

FINAL TEST:
PROTECTED

M6 final model selection:
NO

Core outputs:
M6.1 → M6.7

M6 Gate:
15 checks

Target handoff:
READY FOR M7
```
