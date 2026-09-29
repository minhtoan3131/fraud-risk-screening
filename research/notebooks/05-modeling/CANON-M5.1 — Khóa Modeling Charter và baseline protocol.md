# CANON-M5.1 — Khóa Modeling Charter và baseline protocol

## Document status

Milestone:

`M5 — Modeling`

Substep:

`M5.1 — Modeling Charter / Baseline Protocol`

Work type:

`CONCEPT / PROTOCOL / GUARDRAIL`

Runtime model experiment:

`NOT REQUIRED IN M5.1`

M5.1 không fit classifier.

M5.1 không đọc validation metric để đưa ra model decision.

Mục tiêu của M5.1 là khóa luật thực nghiệm trước khi M5.2–M5.5 bắt đầu chạy model thật.

---

# 1. Câu hỏi trung tâm

M5.1 phải trả lời:

> M5 được phép làm gì, không được làm gì, và một baseline modeling run hợp lệ của project phải tuân thủ những contract nào?

M5.1 phải đủ rõ để các bước sau không phải tự quyết định lại:

- dùng artifact nào;
- model family nào thuộc baseline;
- W_LONG/W_SHORT đóng vai trò gì;
- metric nào phải báo;
- class imbalance được xử lý thế nào ở baseline;
- random state dùng thế nào;
- threshold được hiểu thế nào;
- FINAL TEST có quyền gì;
- output của một experiment phải lưu gì;
- điều kiện nào khiến một run phải dừng.

---

# 2. Trạng thái đầu vào khi bước vào M5

Milestone 4 đã PASS.

Canonical handoff từ M4:

`Feature Specification v1.0 — LOCKED FOR BASELINE IMPLEMENTATION`

`Behavioral Feature Contract v1.0 — LOCKED FOR BASELINE IMPLEMENTATION`

`Preprocessing Specification v1.0 — LOCKED`

`Baseline Matrix Schema v1.0 — LOCKED`

Output representation:

`47-column CSR sparse matrix`

Matrix dtype:

`float32`

Target dtype:

`int8`

W_LONG:

`BASELINE-READY`

W_SHORT:

`BASELINE-READY`

FINAL TEST:

`PROTECTED`

Persisted modeling artifacts:

`VERIFIED`

Saved artifact round-trip:

`PASS`

M5.1 không được mở lại các quyết định M4 chỉ vì modeling sắp bắt đầu.

---

# 3. Vai trò chính thức của M5

Cấp project đã phân chia:

`M5 — Modeling`

Câu hỏi:

> Các model học được gì?

Sản phẩm:

`Logistic Regression / Decision Tree / Random Forest baseline`

M5 nằm giữa:

`M4 — Preprocessing`

và:

`M6 — Evaluation`

sau đó:

`M7 — Model Selection`

Do đó M5 phải tập trung vào:

- model implementation;
- fit/predict;
- reproducible baseline runs;
- prediction/probability evidence;
- modeling metadata.

M5 không được tự biến thành full model-selection milestone.

---

# 4. Ranh giới giữa M5, M6 và M7

## M5 — Modeling

M5 trả lời:

> Các baseline model có thể fit và tạo prediction hợp lệ trên representation đã khóa hay không?

M5 tạo:

- fitted baseline runs;
- prediction evidence;
- probability/risk score khi có;
- run metadata;
- minimal metric bundle;
- warnings/runtime evidence.

---

## M6 — Evaluation

M6 trả lời:

> Các model đang đúng/sai theo hướng nào?

M6 tập trung:

- Confusion Matrix;
- Precision;
- Recall;
- F1;
- false positive;
- false negative;
- predicted-positive behavior;
- error analysis;
- comparative interpretation.

---

## M7 — Model Selection

M7 trả lời:

> Model/config nào phù hợp nhất với project theo protocol đã khóa?

M7 mới xử lý có hệ thống:

- temporal CV;
- hyperparameter tuning;
- class_weight;
- resampling;
- imbalance strategy;
- threshold selection;
- robustness;
- final upstream selection.

---

# 5. Problem definition kế thừa

Problem:

`Binary supervised classification`

Positive class:

`fraud = 1`

Negative class:

`non-fraud = 0`

Prediction point:

`transaction screening time`

Model role:

`transaction risk-screening component`

Model output có thể gồm:

- binary prediction;
- probability/risk score.

Không được diễn giải model như:

`fraud prevention system hoàn chỉnh`

Không được overclaim:

`model xác định được gian lận ngoài đời thực`

Ground truth chỉ là fraud label của dataset.

---

# 6. Temporal protocol kế thừa

Evaluation direction:

`PAST → FUTURE`

Random split:

`NOT AUTHORIZED AS PRIMARY PROJECT EVALUATION`

Canonical training-window candidates:

## W_LONG

Period:

`2015-01-01 <= Timestamp < 2019-01-01`

Rows:

`6,855,270`

Fraud:

`9,606`

ID:

`W_LONG_2015_TO_2018`

---

## W_SHORT

Period:

`2018-01-01 <= Timestamp < 2019-01-01`

Rows:

`1,721,615`

Fraud:

`2,491`

ID:

`W_SHORT_2018_ONLY`

---

## External VALIDATION

Period:

`2019-01-01 <= Timestamp < 2019-06-01`

Rows:

`712,458`

Fraud:

`1,052`

Role:

`DEVELOPMENT HOLDOUT`

---

## FINAL TEST

Period:

`2019-06-01 <= Timestamp < 2019-11-01`

Role:

`PROTECTED FINAL EVALUATION`

M5 access:

`PROHIBITED`

---

# 7. M5 input artifact contract

M5 không đọc raw CSV để tự xây lại modeling matrix.

M5 nhận output M4.7 tại:

`data/processed/m4_07_baseline_ready`

Required files:

- `X_train_w_long.npz`;
- `X_train_w_short.npz`;
- `X_validation_w_long.npz`;
- `X_validation_w_short.npz`;
- `y_train_w_long.npy`;
- `y_train_w_short.npy`;
- `y_validation.npy`;
- `row_id_train_w_long.npy`;
- `row_id_train_w_short.npy`;
- `row_id_validation.npy`;
- `feature_names.json`;
- `manifest.json`.

M5.2 sẽ audit runtime các file này trước khi model fit.

M5.1 khóa policy:

nếu artifact mismatch:

`STOP`

Không được:

- regenerate artifact bằng code khác;
- thay preprocessing;
- bỏ feature;
- đổi schema;
- tự sửa y;
- tự bỏ row

chỉ để model chạy được.

---

# 8. Baseline matrix contract

Feature version:

`Feature Specification v1.0`

Preprocessing version:

`Preprocessing Specification v1.0`

Behavioral contract:

`Behavioral Feature Contract v1.0`

Output width:

`47`

Representation:

`CSR`

X dtype:

`float32`

Target representation:

`0 / 1`

Fraud:

`1`

y dtype:

`int8`

Feature names/order:

`STABLE`

W_LONG và W_SHORT:

`SAME OUTPUT SCHEMA`

M5 không được tự thêm feature vào baseline matrix.

---

# 9. Những feature M5 baseline không tự thêm

Không thêm trực tiếp:

- MCC;
- Merchant State;
- Merchant City;
- Zip;
- month_of_year;
- is_weekend;
- amount_signed_log1p;
- is_negative_amount;
- is_zero_amount;
- prior_card_transaction_count;
- previous_amount_mean.

Các feature này có thể là candidate cho controlled feature experiment sau.

M5 baseline không phải nơi thử mọi feature đã từng xuất hiện trong EDA/M4.

---

# 10. Raw field protection vẫn có hiệu lực

Không đưa vào classifier:

- raw `User`;
- raw `Card`;
- raw `Merchant Name`;
- `Errors?`;
- `Is Fraud?` như feature;
- `raw_row_id`;
- raw Timestamp;
- temporal partition flags.

M5 model phải nhận đúng M4 matrix.

Không được quay lại DataFrame raw rồi vô tình thêm identifier vì model API cho phép.

---

# 11. Baseline model-family scope

Core M5 model families:

1. `Logistic Regression`
2. `Decision Tree`
3. `Random Forest`

Status:

`LOCKED FOR M5 BASELINE SCOPE`

Lý do cấp project:

- phạm vi vừa sức;
- dễ giải thích;
- bao phủ linear baseline;
- single nonlinear tree;
- ensemble tree.

M5 không cần tăng độ phức tạp để “trông giống AI hơn”.

Không core M5 nếu chưa có Decision Log mới:

- XGBoost;
- LightGBM;
- CatBoost;
- neural network;
- deep learning;
- autoencoder;
- isolation forest;
- transformer model;
- anomaly detection pipeline riêng.

---

# 12. Vai trò của Logistic Regression

Logistic Regression là:

`LINEAR BASELINE`

Mục đích:

- tạo model đơn giản dễ hiểu;
- làm điểm tham chiếu cho Tree/Forest;
- tạo probability output;
- kiểm tra liệu representation hiện tại đã chứa signal đủ cho linear classifier hay chưa.

M5.1 không khóa:

- C;
- solver;
- penalty;
- max_iter cụ thể.

Các cấu hình đó được khóa ở M5.3 trước khi đọc comparative validation result.

Status:

`MODEL FAMILY LOCKED — CONFIG OPEN UNTIL M5.3`

---

# 13. Vai trò của Decision Tree

Decision Tree là:

`SINGLE NONLINEAR TREE BASELINE`

Mục đích:

- kiểm tra nonlinear split pattern;
- tạo baseline có cấu trúc quyết định trực quan;
- cung cấp contrast với linear model và ensemble model.

M5.1 không khóa:

- max_depth;
- min_samples_split;
- min_samples_leaf;
- criterion.

Các cấu hình đó được khóa ở M5.4 trước comparative result.

Status:

`MODEL FAMILY LOCKED — CONFIG OPEN UNTIL M5.4`

---

# 14. Vai trò của Random Forest

Random Forest là:

`ENSEMBLE TREE BASELINE`

Mục đích:

- kiểm tra ensemble tree trên cùng representation;
- giảm phụ thuộc vào một single tree;
- tạo baseline model family thứ ba theo định hướng project.

M5.1 không khóa:

- n_estimators;
- max_depth;
- max_features;
- min_samples_leaf;
- bootstrap.

Các cấu hình đó được khóa ở M5.5 trước comparative result.

Status:

`MODEL FAMILY LOCKED — CONFIG OPEN UNTIL M5.5`

---

# 15. Nguyên tắc baseline configuration

Baseline configuration khác tuned configuration.

Baseline config phải:

- đơn giản;
- explicit;
- reproducible;
- được khai báo trước result;
- có lý do kỹ thuật/học thuật;
- giống nhau giữa W_LONG và W_SHORT trong cùng model family.

Không được:

- thử nhiều config;
- đọc validation F1;
- chọn config cao nhất;
- rồi gọi config đó là baseline.

Hành vi trên là tuning.

Tuning thuộc M7.

---

# 16. Training-window role trong M5

W_LONG và W_SHORT đều phải tiếp tục tồn tại.

M5 không chọn winner trước model evidence.

Mỗi model family phải được thiết kế thành pair:

`MODEL-LONG`

và:

`MODEL-SHORT`

Ví dụ:

- LR-LONG;
- LR-SHORT;
- DT-LONG;
- DT-SHORT;
- RF-LONG;
- RF-SHORT.

Trong một model-family window pair:

biến chủ động thay đổi chỉ là:

`TRAINING WINDOW`

---

# 17. Fair training-window comparison contract

Đối với W_LONG vs W_SHORT trong cùng model family, phải giữ cùng:

- validation boundary;
- target definition;
- fraud positive-class definition;
- feature version;
- feature construction logic;
- historical warm-up principle;
- preprocessing algorithm/version;
- model family;
- model hyperparameter configuration;
- imbalance strategy;
- metric implementation;
- threshold policy;
- evaluation code;
- random-state policy;
- warning/error handling.

Không được:

`W_LONG → config A`

`W_SHORT → config B`

rồi kết luận training window tạo ra khác biệt.

Nếu config khác:

đó là interaction/multi-factor experiment.

---

# 18. Historical warm-up contract

Historical features đã được tính ở M4 bằng strict causal state.

M5 không reset history.

M5 không rebuild behavioral feature state.

M5 chỉ dùng persisted matrix.

Do đó:

classifier-training membership

không được nhầm với:

historical-context membership.

Historical warm-up semantics:

`INHERITED — LOCKED`

---

# 19. Class imbalance policy cho M5

Fraud class hiếm.

Nhưng:

`class imbalance ≠ automatic data error`

và:

`class imbalance ≠ bắt buộc SMOTE`

M3.7 khóa:

`NO-INTERVENTION BASELINE = REQUIRED`

Do đó core M5:

`IMBALANCE_STRATEGY = NONE`

TRAIN giữ natural class distribution.

VALIDATION giữ natural class distribution.

M5 không dùng:

- class_weight="balanced";
- custom class weight;
- oversampling;
- undersampling;
- SMOTE.

Những kỹ thuật này thuộc experiment sau baseline.

---

# 20. Vì sao no-intervention baseline là bắt buộc

Nếu dùng class_weight/SMOTE ngay từ model đầu tiên thì project mất mốc trả lời:

> Model cơ bản đang bỏ sót fraud như thế nào nếu chưa can thiệp imbalance?

Baseline NONE giúp:

- tạo reference;
- đo lợi ích thật của intervention sau;
- tránh thêm complexity không có evidence;
- tách model-family behavior khỏi imbalance intervention.

Status:

`LOCKED`

---

# 21. Random-state policy

Randomness được dùng để reproducibility.

Default:

`RANDOM_STATE = 42`

khi model/operation có randomness.

Ví dụ:

Decision Tree:

random_state nếu relevant.

Random Forest:

random_state bắt buộc explicit.

Logistic Regression:

nếu solver/config có randomness thì explicit random state khi applicable.

Không được:

- thử seed 1, 2, 3, ...;
- chọn seed có F1 đẹp nhất.

Nếu result phụ thuộc mạnh random seed:

đó là:

`STABILITY FINDING`

không phải tuning opportunity.

Status:

`LOCKED`

---

# 22. Metric strategy cho baseline runs

Mỗi official classification run phải có canonical metric bundle.

Primary:

`F1_fraud`

Secondary:

`Recall_fraud`

`Precision_fraud`

Mandatory diagnostics:

- TP;
- FP;
- FN;
- TN;
- Confusion Matrix.

Operational diagnostics:

- predicted_positive_count;
- predicted_positive_rate.

Reference:

`Accuracy`

Không được chỉ in Accuracy rồi kết luận model tốt.

Status:

`INHERITED — LOCKED`

---

# 23. Ý nghĩa metric trong project

## Recall_fraud

Trả lời:

> Trong fraud ground-truth, model bắt được bao nhiêu?

Low Recall:

nhiều false negative.

---

## Precision_fraud

Trả lời:

> Trong các transaction model flag fraud, bao nhiêu thật sự là fraud theo dataset?

Low Precision:

nhiều false positive.

---

## F1_fraud

Dùng làm primary comparison metric theo protocol hiện tại.

F1 phải được đọc cùng:

`threshold policy`

vì threshold thay đổi prediction class.

---

## Accuracy

Reference only.

Fraud class hiếm nên Accuracy cao không đủ chứng minh model tốt.

---

# 24. Probability / risk-score policy

Nếu model hỗ trợ probability:

phải giữ:

`predict_proba()` positive-class score

hoặc score tương đương.

Probability/risk score phục vụ:

- M6 analysis;
- M7 threshold study;
- M8 application/inference.

Không được diễn giải một score như:

`0.8 = chắc chắn 80% fraud ngoài đời`

nếu calibration chưa được kiểm tra.

Probability calibration:

`OPTIONAL / DEFERRED`

Status:

`LOCKED`

---

# 25. Threshold policy cho M5

Final numerical threshold:

`OPEN`

M5 không threshold-tune.

Baseline prediction có thể sử dụng:

`default model decision rule`

nếu được khai báo trong experiment metadata.

Nếu model sử dụng default 0.5 probability threshold:

phải ghi:

`BASELINE COMPARISON THRESHOLD ONLY`

`NOT FINAL THRESHOLD`

Điều quan trọng trong W_LONG/W_SHORT pair:

`THRESHOLD POLICY MUST BE IDENTICAL`

Không được:

- threshold 0.5 cho W_LONG;
- threshold 0.2 cho W_SHORT;
- rồi kết luận W_SHORT tốt hơn.

Threshold optimization:

`DEFERRED TO M7`

---

# 26. FINAL TEST policy

FINAL TEST:

`2019-06 → 2019-10`

M5:

`NO ACCESS`

M5 code không được load final-test labels để:

- xem score;
- debug performance;
- chọn baseline config;
- chọn model;
- chọn window;
- chọn threshold;
- chọn imbalance strategy.

FINAL TEST boundary có thể tồn tại trong documentation.

FINAL TEST performance:

`DO NOT COMPUTE`

Nếu final-test performance bị mở trong M5:

`BLOCKING PROTOCOL VIOLATION`

---

# 27. External VALIDATION role

VALIDATION:

`2019-01 → 2019-05`

Role:

`DEVELOPMENT HOLDOUT`

M5 được phép dùng VALIDATION để tạo baseline run evidence.

Tuy nhiên M5 không dùng validation để systematic tuning.

M5 output metric được xem là:

`BASELINE EVIDENCE`

Comparative interpretation sâu:

`M6`

Formal model selection/tuning:

`M7`

---

# 28. Temporal CV role

Temporal CV không phải core M5.

Nếu CV được dùng sau:

primary strategy:

`FORWARD / EXPANDING TEMPORAL CV`

Validation blocks:

`Q2 / Q3 / Q4 2018`

Random shuffled StratifiedKFold:

`NOT PRIMARY`

Temporal CV thuộc M7 tuning/stability/robustness.

M5 không cần chạy CV để được coi là baseline modeling milestone hoàn chỉnh.

---

# 29. Preprocessing trong M5

M5 dùng matrices đã preprocessing.

M5 không:

- fit StandardScaler mới;
- fit OneHotEncoder mới;
- mở rộng vocabulary;
- impute lại values;
- thay cold-start policy;
- thay `__UNKNOWN__` handling;
- thêm/drop output columns.

Nếu một model không thích representation hiện tại:

đó phải được ghi như:

`MODEL/REPRESENTATION COMPATIBILITY FINDING`

không được âm thầm thay preprocessing chỉ cho riêng model đó trong core baseline.

---

# 30. Baseline modeling run definition

Một official M5 run phải có:

## Identity

- experiment_id;
- model_family;
- model_id;
- model_config_id;
- training_window_id.

## Input contract

- feature_version;
- preprocessing_version;
- matrix_schema_version;
- feature_count;
- train rows;
- train fraud count;
- validation rows;
- validation fraud count.

## Learning policy

- imbalance_strategy;
- random_state;
- threshold_policy.

## Fit evidence

- fit started;
- fit completed;
- fit time;
- warnings;
- convergence status nếu applicable;
- technical errors nếu có.

## Prediction evidence

- predicted labels;
- probability/risk score nếu available;
- prediction/evaluation time.

## Metric evidence

- F1_fraud;
- Recall_fraud;
- Precision_fraud;
- TP;
- FP;
- FN;
- TN;
- predicted_positive_count;
- predicted_positive_rate;
- Accuracy.

## Integrity

- input finite;
- X/y alignment;
- schema correct;
- correct training window;
- FINAL TEST access = NO.

---

# 31. Experiment ID convention

Đề xuất M5 baseline IDs:

Logistic Regression:

`M5-LR-LONG-B01`

`M5-LR-SHORT-B01`

Decision Tree:

`M5-DT-LONG-B01`

`M5-DT-SHORT-B01`

Random Forest:

`M5-RF-LONG-B01`

`M5-RF-SHORT-B01`

`B01`

nghĩa:

`Baseline configuration version 01`

Nếu phải sửa technical bug mà không đổi semantic config:

có thể increment run revision riêng.

Nếu đổi hyperparameter substantive:

không được coi là cùng baseline run âm thầm.

---

# 32. Shared experiment log schema

Mỗi run phải lưu tối thiểu:

```text
Experiment ID
Run status
Model family
Model ID
Model configuration
Training-window ID

Feature version
Preprocessing version
Matrix schema version

Training rows
Training fraud rows
Validation rows
Validation fraud rows

Imbalance strategy
Random state
Threshold policy

F1_fraud
Recall_fraud
Precision_fraud
Accuracy

TP
FP
FN
TN

Predicted-positive count
Predicted-positive rate

Probability output available?
Probability artifact / in-memory reference

Fit time
Prediction time
Warnings
Errors

FINAL TEST accessed?
Integrity gate result

Decision
Next action
```

M5.2 sẽ triển khai schema này thành code/runtime runner.

---

# 33. Error-handling policy

Nếu model fit lỗi:

`DO NOT HIDE ERROR`

Nếu warning xuất hiện:

`DO NOT SILENCE WITHOUT REVIEW`

Ví dụ:

- convergence warning;
- memory error;
- sparse-matrix incompatibility;
- invalid parameter;
- NaN/inf;
- dtype conversion issue.

Cần:

1. giữ error/warning;
2. xác định technical cause;
3. sửa đúng cause;
4. rerun;
5. không thay experimental factor khác nếu không cần.

Không được sửa model config chỉ để “ra được score” mà không ghi Decision Log.

---

# 34. Convergence policy

Đặc biệt với Logistic Regression:

nếu convergence warning xuất hiện:

đó là:

`TECHNICAL MODEL-FIT ISSUE`

không phải metric finding.

Không được đọc F1 trước khi xác định fitted state đủ hợp lệ.

Các điều chỉnh như `max_iter` có thể là technical convergence configuration nếu được lý giải rõ, nhưng phải được khóa trước comparative interpretation và dùng giống nhau cho W_LONG/W_SHORT.

M5.3 chịu trách nhiệm khóa chi tiết.

---

# 35. Computational constraint policy

Dataset lớn:

W_LONG hơn 6.8 triệu rows.

Random Forest có thể tốn CPU/RAM.

Nếu gặp resource issue:

không được âm thầm:

- subsample W_LONG;
- chỉ train W_SHORT;
- đổi feature set;
- giảm matrix rows;
- dùng different config giữa windows.

Đúng protocol:

`STOP`

→ ghi computational finding

→ đề xuất controlled workaround

→ khóa workaround trước result

→ áp dụng công bằng nếu comparison yêu cầu.

Runtime ngắn hơn không tự động làm model tốt hơn.

---

# 36. Fair model-family comparison principle

Nếu M6 sau này so LR vs DT vs RF trên một training window:

các model phải dùng cùng:

- same X/y population;
- same feature version;
- same preprocessing version;
- same validation set;
- same imbalance strategy;
- same threshold policy class;
- same metric code.

Khác biệt chính:

`MODEL FAMILY / BASELINE CONFIG`

M5.1 chưa quyết định model winner.

---

# 37. Những điều M5.1 được khóa ngay

Có đủ protocol evidence để khóa:

- M5 scope;
- model-family scope;
- artifact input policy;
- W_LONG/W_SHORT roles;
- no-intervention baseline;
- random-state policy;
- metric bundle;
- probability retention;
- threshold role;
- FINAL TEST prohibition;
- experiment-log schema;
- M5/M6/M7 boundary.

Những điều này không cần model output để khóa.

---

# 38. Những điều M5.1 không được khóa

M5.1 không có evidence để quyết định:

- Logistic Regression tốt hơn Tree/Forest;
- Decision Tree tốt hơn;
- Random Forest tốt hơn;
- W_LONG tốt hơn W_SHORT;
- baseline F1 thực tế;
- Recall thực tế;
- Precision thực tế;
- final feature set ngoài baseline v1;
- final imbalance strategy;
- final hyperparameters;
- final threshold;
- final test performance.

Các câu hỏi này:

`REQUIRE RUNTIME MODEL EVIDENCE`

---

# 39. Model-specific configuration status

## Logistic Regression

Family:

`LOCKED`

Exact baseline config:

`OPEN — M5.3`

---

## Decision Tree

Family:

`LOCKED`

Exact baseline config:

`OPEN — M5.4`

---

## Random Forest

Family:

`LOCKED`

Exact baseline config:

`OPEN — M5.5`

---

# 40. Modeling workflow sau M5.1

```text
M5.1
Modeling Charter
        ↓
M5.2
Audit M4 artifacts
+
Shared experiment runner
        ↓
M5.3
Logistic Regression baseline
LR-LONG / LR-SHORT
        ↓
M5.4
Decision Tree baseline
DT-LONG / DT-SHORT
        ↓
M5.5
Random Forest baseline
RF-LONG / RF-SHORT
        ↓
M5.6
Baseline Model Registry
M5 Decision Log
M5 Gate
        ↓
M6
Evaluation + Error Analysis
        ↓
M7
Model Selection / CV / tuning /
imbalance / threshold
```

---

# 41. M5.1 Decision Log

## M5.1-D01 — Milestone role

Decision:

M5 là baseline modeling milestone.

Không phải final model-selection milestone.

Status:

`LOCKED`

---

## M5.1-D02 — M5 / M6 / M7 boundary

Decision:

M5:

fit baseline models.

M6:

evaluation/error analysis.

M7:

selection/CV/tuning/imbalance/threshold.

Status:

`LOCKED`

---

## M5.1-D03 — Input artifact

Decision:

M5 sử dụng canonical M4.7 persisted artifacts.

Không rebuild feature/preprocessing bằng logic mới.

Status:

`LOCKED`

---

## M5.1-D04 — Matrix contract

Decision:

47-column CSR float32 feature matrix.

Target int8, fraud = 1.

Status:

`INHERITED — LOCKED`

---

## M5.1-D05 — Baseline model families

Decision:

Core M5 model families:

- Logistic Regression;
- Decision Tree;
- Random Forest.

Status:

`LOCKED`

---

## M5.1-D06 — Training-window candidates

Decision:

Giữ cả W_LONG và W_SHORT.

Status:

`INHERITED — LOCKED`

Winner:

`OPEN`

---

## M5.1-D07 — Window comparison

Decision:

Trong cùng model family, W_LONG/W_SHORT comparison chỉ thay training window.

Status:

`INHERITED — LOCKED`

---

## M5.1-D08 — Imbalance baseline

Decision:

`IMBALANCE_STRATEGY = NONE`

cho core M5 baseline.

Status:

`INHERITED — LOCKED`

---

## M5.1-D09 — Random state

Decision:

`RANDOM_STATE = 42`

khi randomness applicable.

Status:

`INHERITED — LOCKED`

---

## M5.1-D10 — Metric strategy

Decision:

Primary:

`F1_fraud`

Secondary:

`Recall_fraud + Precision_fraud`

Mandatory diagnostic:

`Confusion Matrix + TP/FP/FN/TN`

Operational:

`predicted-positive count/rate`

Accuracy:

`reference only`

Status:

`INHERITED — LOCKED`

---

## M5.1-D11 — Probability

Decision:

Giữ probability/risk score khi model hỗ trợ.

Status:

`INHERITED — LOCKED`

---

## M5.1-D12 — Threshold

Decision:

M5 không chọn final numerical threshold.

Baseline threshold/decision rule phải explicit và giống nhau trong controlled comparison.

Status:

`LOCKED`

Final threshold:

`OPEN`

---

## M5.1-D13 — Tuning

Decision:

Systematic hyperparameter tuning không thuộc M5.

Status:

`DEFERRED TO M7`

---

## M5.1-D14 — Temporal CV

Decision:

Không phải core M5.

Nếu dùng sau:

forward/expanding temporal CV.

Status:

`DEFERRED TO M7`

---

## M5.1-D15 — FINAL TEST

Decision:

No access in M5.

Status:

`INHERITED — LOCKED`

---

## M5.1-D16 — Exact model configuration

Decision:

M5.1 không khóa exact baseline hyperparameters.

Model-specific substeps phải khóa config trước result.

Status:

`DEFERRED TO M5.3 / M5.4 / M5.5`

---

## M5.1-D17 — Experiment log

Decision:

Mỗi run phải có complete identity/data/model/metric/runtime/integrity metadata.

Status:

`LOCKED`

---

## M5.1-D18 — Score interpretation

Decision:

M5 score là baseline modeling evidence.

Comparative/error interpretation chính thức thuộc M6.

Final selection thuộc M7.

Status:

`LOCKED`

---

# 42. Open Questions sau M5.1

## O01 — Logistic baseline configuration

Status:

`OPEN — M5.3`

---

## O02 — Decision Tree baseline configuration

Status:

`OPEN — M5.4`

---

## O03 — Random Forest baseline configuration

Status:

`OPEN — M5.5`

---

## O04 — W_LONG vs W_SHORT preference

Status:

`OPEN — REQUIRES MODEL EVIDENCE`

---

## O05 — Model-family preference

Status:

`OPEN — M6/M7`

---

## O06 — Feature experiments beyond baseline

Status:

`OPEN — CONTROLLED EXPERIMENT LATER`

---

## O07 — Imbalance strategy

Status:

`OPEN — M7`

---

## O08 — Hyperparameters

Status:

`OPEN — M7`

---

## O09 — Final threshold

Status:

`OPEN — M7`

---

## O10 — Final-test performance

Status:

`PROTECTED / NOT AVAILABLE`

---

# 43. M5.1 Gate

## G01 — M5 scope clear?

Requirement:

M5 = baseline modeling, không phải final selection.

Result:

`PASS`

---

## G02 — M5/M6/M7 boundary clear?

Requirement:

fit / evaluate / select được tách role.

Result:

`PASS`

---

## G03 — Input artifact contract clear?

Requirement:

M5 dùng canonical M4.7 artifacts.

Result:

`PASS`

---

## G04 — Feature/preprocessing contract preserved?

Requirement:

47-column CSR float32 schema không bị tự ý thay đổi.

Result:

`PASS`

---

## G05 — Model family scope clear?

Requirement:

Logistic Regression / Decision Tree / Random Forest.

Result:

`PASS`

---

## G06 — Training windows clear?

Requirement:

W_LONG và W_SHORT đều giữ làm candidates.

Result:

`PASS`

---

## G07 — Controlled window comparison clear?

Requirement:

chỉ training window thay trong same-model primary pair.

Result:

`PASS`

---

## G08 — Baseline imbalance policy clear?

Requirement:

`IMBALANCE_STRATEGY = NONE`

Result:

`PASS`

---

## G09 — Metric strategy clear?

Requirement:

F1_fraud + Recall/Precision + CM + predicted-positive diagnostics.

Result:

`PASS`

---

## G10 — Probability policy clear?

Requirement:

giữ probability/risk score khi available.

Result:

`PASS`

---

## G11 — Threshold policy clear?

Requirement:

final threshold chưa khóa; baseline rule explicit; không threshold tuning trong M5.

Result:

`PASS`

---

## G12 — Randomness policy clear?

Requirement:

random state cho reproducibility, không seed tuning.

Result:

`PASS`

---

## G13 — FINAL TEST protected?

Requirement:

FINAL TEST access trong M5 = prohibited.

Result:

`PASS`

---

## G14 — Exact hyperparameter decision deferred correctly?

Requirement:

M5.1 không chọn config dựa trên nonexistent score.

Result:

`PASS`

---

## G15 — Experiment result schema defined?

Requirement:

mọi run có enough metadata/metric/runtime/integrity evidence.

Result:

`PASS`

---

## G16 — Ready for M5.2?

Requirement:

M5.2 có thể audit artifacts và implement shared runner mà không tự phát minh lại protocol.

Result:

`PASS`

---

# 44. M5.1 Gate summary

Overall:

`PASS`

Blocking issue:

`NONE`

M5.1 status:

`MODELING CHARTER — LOCKED`

`BASELINE PROTOCOL — LOCKED`

Runtime model score:

`NOT YET GENERATED`

FINAL TEST:

`STILL PROTECTED`

Next:

`M5.2 — Audit modeling artifacts và khóa shared experiment runner`

---

# 45. Kết luận M5.1

M5.1 đã khóa đầy đủ protocol cần thiết trước baseline model training.

M5 bắt đầu từ persisted 47-column CSR matrices của M4.

Ba model family thuộc core baseline:

- Logistic Regression;
- Decision Tree;
- Random Forest.

Hai training-window candidates tiếp tục được giữ:

- W_LONG;
- W_SHORT.

No-intervention baseline:

`REQUIRED`

Imbalance strategy trong core M5:

`NONE`

Metric primary:

`F1_fraud`

Secondary:

`Recall_fraud + Precision_fraud`

Mandatory diagnostics:

`Confusion Matrix + TP/FP/FN/TN`

Operational diagnostics:

`predicted-positive count/rate`

Accuracy:

`REFERENCE ONLY`

Probability/risk score:

`PRESERVE WHEN AVAILABLE`

Random state:

`42 WHEN APPLICABLE`

Final numerical threshold:

`OPEN`

Systematic tuning:

`NOT IN M5`

Temporal CV:

`NOT CORE M5`

FINAL TEST:

`NO ACCESS`

M5.1 không tạo model-performance claim vì chưa có model experiment thật.

Trạng thái cuối:

`M5.1 — PASS`

`Modeling Charter — LOCKED`

`Baseline Experiment Protocol — LOCKED`

`READY FOR M5.2`

---

# 46. Handoff sang M5.2

M5.2 phải làm runtime work đầu tiên của Milestone 5.

M5.2 cần:

- locate canonical M4.7 artifact directory;
- load manifest;
- load feature names;
- load W_LONG/W_SHORT matrices;
- load targets;
- load lineage nếu cần audit;
- verify CSR/dtype/shape;
- verify target counts;
- verify no NaN/inf;
- verify feature width/order;
- verify artifact version;
- verify validation alignment;
- build one shared evaluation function;
- build one shared experiment runner;
- build one shared metric implementation;
- build one experiment-result schema;
- ensure FINAL TEST path/data không được dùng.

M5.2 chưa cần chọn model winner.

M5.2 Gate phải PASS trước M5.3.

---

# 47. Source hierarchy

M5.1 kế thừa trực tiếp:

- `CANON-Kế hoạch Milestone 5 — Modeling baseline.md`
- `CANON-M4.8 — Tổng hợp Feature Specification v1.0, Preprocessing Specification v1.0, Behavioral Feature Contract, Decision Log và M4 Gate.md`
- `CANON-M3.4 — Thiết kế evaluation metric strategy.md`
- `CANON-M3.6 — Thiết kế protocol so sánh training window.md`
- `CANON-M3.7 — Policy cho class imbalance, temporal validation/CV và các experiment/tuning sau.md`
- `CANON-M3.8 — Tổng hợp Experiment Specification v1.0, Decision Log, Open Questions và M3 Gate.md`
- `CANON-Quy trình thực hiện dự án.md`

Các tài liệu học tuần là background giải thích model, không ghi đè project-specific temporal/partition protocol.

---

# 48. Final M5.1 handoff block

```text
M5.1:
PASS

Work type:
PROTOCOL / GUARDRAIL

Model training:
NOT YET

Input:
M4.7 canonical baseline-ready artifacts

Feature schema:
47-column CSR float32

Models:
Logistic Regression
Decision Tree
Random Forest

Training windows:
W_LONG
W_SHORT

Imbalance strategy:
NONE

Primary metric:
F1_fraud

Secondary:
Recall_fraud
Precision_fraud

Mandatory diagnostics:
Confusion Matrix
TP / FP / FN / TN

Operational diagnostics:
Predicted-positive count/rate

Accuracy:
REFERENCE ONLY

Probability/risk score:
PRESERVE WHEN AVAILABLE

Random state:
42 WHEN APPLICABLE

Systematic tuning:
DEFERRED TO M7

Temporal CV:
DEFERRED TO M7

Final threshold:
OPEN

FINAL TEST:
PROTECTED / NO ACCESS

Exact LR config:
OPEN — M5.3

Exact DT config:
OPEN — M5.4

Exact RF config:
OPEN — M5.5

Next:
M5.2 — Artifact Audit + Shared Baseline Runner
```
