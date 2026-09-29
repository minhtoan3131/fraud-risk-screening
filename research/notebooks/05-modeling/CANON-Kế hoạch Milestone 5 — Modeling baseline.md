# CANON-Kế hoạch Milestone 5 — Modeling baseline

## Document status

Milestone:

`M5 — Modeling`

Câu hỏi trung tâm cấp project:

> Các model baseline học được gì từ representation đã khóa ở M4?

Sản phẩm cấp project đã được định hướng:

`Logistic Regression / Decision Tree / Random Forest baseline`

M5 bắt đầu từ:

`M4 — PASS`

`Feature Specification v1.0 — LOCKED FOR BASELINE IMPLEMENTATION`

`Behavioral Feature Contract v1.0 — LOCKED FOR BASELINE IMPLEMENTATION`

`Preprocessing Specification v1.0 — LOCKED`

`Baseline Matrix Schema v1.0 — 47-column CSR float32`

`W_LONG — BASELINE-READY`

`W_SHORT — BASELINE-READY`

`FINAL TEST — PROTECTED`

---

# 1. Vai trò của Milestone 5

M5 là bước đầu tiên project thực sự fit các classifier trên modeling matrix đã chuẩn bị ở M4.

M5 không còn hỏi:

> Dữ liệu phải được biểu diễn như thế nào?

Câu hỏi đó đã được M4 giải quyết.

M5 hỏi:

> Với cùng một feature/preprocessing contract, các classifier cơ bản có thể học và tạo prediction/risk score hợp lệ như thế nào?

M5 phải tạo được baseline runs cho ba model family đã định hướng ở cấp project:

- Logistic Regression;
- Decision Tree;
- Random Forest.

M5 là:

`MODELING / BASELINE IMPLEMENTATION MILESTONE`

M5 chưa phải:

`FINAL MODEL SELECTION MILESTONE`

---

# 2. Ranh giới M5 với M6 và M7

CANON cấp project tách rõ:

`M5 — Modeling`

→ Logistic / Tree / Forest baseline.

`M6 — Evaluation`

→ Confusion Matrix, Precision, Recall, F1, error analysis.

`M7 — Model Selection`

→ CV, tuning vừa phải, imbalance strategy, threshold nếu cần.

Do đó M5 không được tự mở rộng thành M6/M7.

## M5 được phép

- load baseline-ready artifacts từ M4;
- xác minh modeling input;
- định nghĩa model baseline configuration;
- fit Logistic Regression;
- fit Decision Tree;
- fit Random Forest;
- chạy trên W_LONG và W_SHORT theo controlled setup;
- tạo validation predictions;
- giữ probability/risk score khi model hỗ trợ;
- tính metric tối thiểu như run evidence nếu cần;
- ghi fit time, prediction time, warnings;
- lưu experiment/modeling registry;
- kiểm tra reproducibility/integrity;
- handoff prediction artifacts sang M6.

## M5 không được phép

- tuyên bố final model winner;
- tuning hyperparameter có hệ thống;
- dùng temporal CV để chọn config;
- chọn class_weight winner;
- oversampling / undersampling để tối ưu;
- SMOTE;
- tối ưu numerical threshold;
- dùng FINAL TEST;
- chọn final pipeline;
- tuyên bố final project performance.

Những việc đó thuộc M6/M7 hoặc bước sau theo đúng protocol.

---

# 3. Căn cứ kế thừa từ M3

M3 đã khóa experiment protocol.

M5 không được thiết kế lại các quy tắc sau.

## 3.1 Problem

`Binary supervised classification`

Positive class:

`fraud = 1`

Prediction point:

`transaction screening time`

Model được định vị là:

`transaction risk-screening component`

không phải hệ thống tự động quyết định chặn giao dịch ngoài đời.

---

## 3.2 Primary metric

`F1_fraud`

Status:

`LOCKED`

Secondary metrics:

- `Recall_fraud`;
- `Precision_fraud`.

Mandatory diagnostics:

- TP;
- FP;
- FN;
- TN;
- Confusion Matrix.

Operational diagnostic:

- predicted-positive count;
- predicted-positive rate.

Accuracy:

`REFERENCE ONLY`

Probability/risk score:

`REQUIRED WHEN AVAILABLE`

M5 có thể tính các metric trên để xác nhận run hợp lệ, nhưng phần phân tích sâu về lỗi và trade-off thuộc M6.

---

## 3.3 Training-window candidates

W_LONG:

`2015-01-01 <= Timestamp < 2019-01-01`

Rows:

`6,855,270`

Fraud:

`9,606`

W_SHORT:

`2018-01-01 <= Timestamp < 2019-01-01`

Rows:

`1,721,615`

Fraud:

`2,491`

VALIDATION:

`2019-01-01 <= Timestamp < 2019-06-01`

Rows:

`712,458`

Fraud:

`1,052`

Winner:

`OPEN`

---

## 3.4 Training-window controlled comparison

Nếu đọc W_LONG và W_SHORT như một comparison thì:

biến duy nhất được thay đổi là:

`TRAINING WINDOW`

Phải giữ cố định:

- target definition;
- positive-class definition;
- feature version;
- preprocessing procedure/version;
- historical warm-up policy;
- model family;
- model configuration;
- imbalance strategy;
- metric implementation;
- threshold policy;
- evaluation code;
- random-state policy;
- error-handling policy.

Không được tuning riêng W_LONG và W_SHORT rồi gọi đó là tác động thuần của training window.

---

## 3.5 Imbalance baseline

M3.7 đã khóa:

`NO-INTERVENTION BASELINE = REQUIRED`

M5 baseline sử dụng:

`IMBALANCE_STRATEGY = NONE`

Không:

- class_weight optimization;
- random oversampling;
- random undersampling;
- SMOTE.

Class imbalance intervention chỉ được nghiên cứu sau khi baseline đã tồn tại.

---

## 3.6 Randomness

Default random state cho operation có randomness:

`RANDOM_STATE = 42`

Random state dùng để:

`REPRODUCIBILITY`

không dùng như hyperparameter để săn validation score.

---

## 3.7 FINAL TEST

FINAL TEST:

`2019-06-01 <= Timestamp < 2019-11-01`

M5 access:

`PROHIBITED`

Không được dùng FINAL TEST cho:

- baseline config decision;
- model comparison;
- training-window decision;
- threshold;
- tuning;
- class imbalance;
- debugging performance.

---

# 4. Căn cứ kế thừa từ M4

M5 nhận trực tiếp persisted artifacts tại:

`data/processed/m4_07_baseline_ready`

Required artifacts:

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

M4.7 đã xác minh:

`SAVED ARTIFACT ROUND-TRIP GATE: PASS`

M5 không cần quét lại raw transaction dataset chỉ để train baseline model.

Nếu M5 phát hiện artifact mismatch:

`STOP`

Không được tự regenerate theo logic khác với M4 để tiếp tục lấy score.

---

# 5. Baseline feature/preprocessing contract của M5

M5 sử dụng đúng:

`Baseline Matrix Schema v1.0`

Output width:

`47`

Representation:

`CSR sparse matrix`

Dtype:

`float32`

Pre-encoding core:

`10 features`

M5 không tự thêm:

- MCC;
- Merchant State;
- Merchant City;
- Zip;
- month_of_year;
- is_weekend;
- Amount signed-log;
- sign/zero indicator;
- support state khác.

Nếu muốn thử các feature đó:

`DEFER TO CONTROLLED FEATURE EXPERIMENT`

không trộn vào baseline modeling run.

---

# 6. Model family scope

Cấp project đã định hướng phạm vi vừa sức:

- Logistic Regression;
- Decision Tree;
- Random Forest.

Đây là ba model family chính thức của M5 baseline.

Không tự thêm model phức tạp chỉ để tăng số lượng thuật toán.

Ví dụ không thuộc core M5 nếu chưa có Decision Log mới:

- XGBoost;
- LightGBM;
- neural network;
- deep learning;
- anomaly-detection model;
- transformer model.

Mục tiêu M5 là hiểu được baseline model và tạo experiment có thể bảo vệ được khi vấn đáp.

---

# 7. Baseline configuration principle

M5 cần phân biệt:

`baseline configuration`

với:

`tuned configuration`

Baseline configuration phải:

- đơn giản;
- explicit;
- có lý do;
- không được chọn bằng cách thử nhiều config rồi lấy validation score cao nhất;
- áp dụng cùng config cho W_LONG/W_SHORT trong cùng model family.

Exact baseline hyperparameters:

`OPEN UNTIL MODEL-SPECIFIC SUBSTEP`

Mỗi model-specific substep phải khóa config trước khi đọc comparative validation result.

---

# 8. Threshold policy trong M5

Final numerical threshold:

`OPEN`

M5 không tối ưu threshold.

Nếu classifier cung cấp `predict()` theo default decision rule thì có thể dùng output đó như:

`BASELINE COMPARISON DECISION RULE`

nhưng phải ghi rõ:

`NOT FINAL THRESHOLD`

Probability/risk score vẫn phải giữ khi model hỗ trợ để M6/M7 có thể phân tích sau.

Không được thay threshold giữa W_LONG và W_SHORT trong cùng controlled baseline comparison.

---

# 9. M5.1 — Khóa Modeling Charter và baseline protocol

## Câu hỏi

> M5 được phép làm gì và một baseline modeling run hợp lệ phải tuân thủ những contract nào?

## Nhiệm vụ

Tổng hợp và khóa:

- input artifact version;
- feature/preprocessing version;
- model family scope;
- W_LONG/W_SHORT roles;
- no-intervention imbalance policy;
- random-state policy;
- baseline threshold policy;
- metric implementation;
- output schema của experiment log;
- FINAL TEST prohibition.

## Không cần

- fit model;
- tuning;
- CV;
- model selection.

## Output

`M5 Modeling Charter`

+

`Baseline Experiment Protocol`

+

`M5 Guardrails`

+

`Decision / Open-question Log`

## Gate

PASS khi có thể trả lời:

- model học X/y nào;
- validation nào được dùng;
- FINAL TEST có bị chạm không;
- imbalance policy là gì;
- model families nào thuộc M5;
- M5/M6/M7 khác nhau ở đâu.

---

# 10. M5.2 — Audit modeling artifacts và khóa shared experiment runner

## Câu hỏi

> Persisted artifacts từ M4 có thể được load trực tiếp và dùng nhất quán cho tất cả baseline models hay không?

## Input

Canonical M4.7 artifacts.

## Audit bắt buộc

### File integrity

- required files tồn tại;
- manifest tồn tại;
- feature_names tồn tại;
- load không lỗi.

### Shapes

W_LONG TRAIN:

`6,855,270 × 47`

W_SHORT TRAIN:

`1,721,615 × 47`

W_LONG VALIDATION:

`712,458 × 47`

W_SHORT VALIDATION:

`712,458 × 47`

### Dtype

X:

`float32`

y:

`int8`

### Sparse representation

X:

`CSR`

### Target support

W_LONG fraud:

`9,606`

W_SHORT fraud:

`2,491`

VALIDATION fraud:

`1,052`

### Numerical integrity

- no NaN;
- no inf.

### Alignment

- X rows == y rows;
- validation targets shared correctly;
- feature width/order consistent.

## Shared runner

M5.2 phải tạo reusable functions cho:

- fit model;
- predict class;
- predict probability/risk score nếu available;
- measure fit time;
- measure prediction time;
- collect warnings;
- calculate canonical metric bundle;
- produce experiment manifest.

Không viết ba notebook với ba evaluation implementations khác nhau.

## Output

`Baseline Modeling Runner v1`

+

`Artifact Compatibility Audit`

+

`Experiment Result Schema`

## Gate

PASS khi một model baseline có thể được cắm vào runner mà không thay temporal/metric/artifact logic.

---

# 11. M5.3 — Logistic Regression baseline

## Vai trò

Logistic Regression là linear baseline đơn giản, dễ diễn giải và là điểm tham chiếu để xem model phi tuyến có thực sự tạo thêm giá trị hay không.

## Câu hỏi

> Với Baseline Feature Set v1.0 và no-imbalance intervention, Logistic Regression học được mức signal nào trên W_LONG và W_SHORT?

## Runs

### LR-LONG

TRAIN:

`X_train_w_long / y_train_w_long`

VALIDATION:

`X_validation_w_long / y_validation`

### LR-SHORT

TRAIN:

`X_train_w_short / y_train_w_short`

VALIDATION:

`X_validation_w_short / y_validation`

## Controlled variables

Hai runs phải dùng cùng:

- Logistic Regression configuration;
- Feature Specification v1.0;
- Preprocessing Specification v1.0;
- imbalance strategy = NONE;
- evaluation code;
- threshold policy;
- random-state policy.

## Output cần giữ

- fitted status;
- convergence status;
- warnings;
- fit time;
- prediction time;
- predicted class;
- fraud probability/risk score nếu available;
- F1_fraud;
- Recall_fraud;
- Precision_fraud;
- Confusion Matrix;
- TP/FP/FN/TN;
- predicted-positive count/rate;
- Accuracy reference;
- experiment metadata.

## M5 interpretation boundary

Có thể mô tả:

- model fit thành công hay không;
- numerical/convergence issue có hay không;
- prediction/probability output hợp lệ hay không.

Chưa khóa:

- Logistic là model tốt nhất;
- W_LONG/W_SHORT winner cuối;
- final threshold.

## Gate

PASS khi cả LR-LONG và LR-SHORT là reproducible valid baseline runs hoặc có documented technical reason khiến một run không hợp lệ.

---

# 12. M5.4 — Decision Tree baseline

## Vai trò

Decision Tree cung cấp một nonlinear single-model baseline.

Nó giúp project quan sát liệu một model phân nhánh có học được pattern khác linear baseline hay không.

## Câu hỏi

> Với cùng data contract và no-imbalance intervention, một Decision Tree baseline học được gì?

## Runs

### DT-LONG

W_LONG TRAIN → W_LONG VALIDATION.

### DT-SHORT

W_SHORT TRAIN → W_SHORT VALIDATION.

## Controlled variables

Hai window runs phải giữ cùng:

- Decision Tree baseline configuration;
- random state;
- features;
- preprocessing;
- imbalance strategy;
- threshold policy;
- metric code.

## Output

Giống canonical result schema của M5.2.

Ngoài ra ghi:

- learned tree depth;
- leaf count;
- model-size diagnostics nếu dễ thu thập.

Các đại lượng này là complexity diagnostics.

Không dùng riêng tree depth/leaf count để chọn winner.

## Gate

PASS khi cả DT-LONG và DT-SHORT được fit/evaluate bằng cùng experiment contract.

---

# 13. M5.5 — Random Forest baseline

## Vai trò

Random Forest cung cấp ensemble-tree baseline.

Model này giúp kiểm tra liệu tổng hợp nhiều tree có tạo prediction ổn định/hữu ích hơn single-tree baseline hay không.

## Câu hỏi

> Với cùng baseline representation và no-imbalance intervention, Random Forest tạo được baseline prediction hợp lệ trên hai training-window candidates hay không?

## Runs

### RF-LONG

W_LONG TRAIN → W_LONG VALIDATION.

### RF-SHORT

W_SHORT TRAIN → W_SHORT VALIDATION.

## Controlled variables

Hai runs giữ cùng:

- Random Forest baseline configuration;
- random state = 42 nếu operation có randomness;
- feature/preprocessing version;
- imbalance strategy = NONE;
- threshold policy;
- metric code.

## Computational note

W_LONG có hơn 6.8 triệu rows.

Random Forest có thể là bước nặng về CPU/RAM.

M5.5 phải ghi:

- fit time;
- prediction time;
- warnings;
- resource issue nếu xảy ra.

Không được âm thầm:

- giảm TRAIN rows;
- subsample khác nhau giữa W_LONG/W_SHORT;
- giảm số feature;
- đổi model config cho riêng một window

chỉ để làm run chạy được rồi vẫn gọi đó là controlled comparison.

Nếu computational constraint buộc thay đổi design:

`STOP → DOCUMENT → CREATE NEW DECISION`

## Gate

PASS khi baseline RF run có integrity rõ và mọi computational workaround nếu có đều được khai báo trước khi đọc comparative score.

---

# 14. M5.6 — Consolidated Baseline Modeling Registry và handoff M6

## Câu hỏi

> Sau M5, project đã có đủ baseline model outputs để M6 phân tích lỗi và metric một cách công bằng hay chưa?

## Modeling Registry

Mỗi run phải có unique Experiment ID.

Tối thiểu gồm:

- Experiment ID;
- model family;
- model configuration;
- training-window ID;
- training row count;
- training fraud count;
- validation row count;
- validation fraud count;
- Feature Specification version;
- Preprocessing Specification version;
- matrix schema version;
- imbalance strategy;
- random state;
- threshold policy;
- fit time;
- prediction time;
- warnings;
- prediction artifact location;
- probability/risk-score artifact location nếu có;
- canonical metric bundle;
- integrity status.

## Expected run registry

- LR-LONG;
- LR-SHORT;
- DT-LONG;
- DT-SHORT;
- RF-LONG;
- RF-SHORT.

## M5.6 không chọn final model

M5.6 chỉ xác nhận:

`BASELINE MODELING EVIDENCE EXISTS`

Nó không được viết:

- Logistic thắng;
- Tree thắng;
- Forest thắng;
- W_LONG thắng;
- W_SHORT thắng

chỉ dựa trên việc một score trông lớn hơn.

Các kết luận comparative/error-analysis chính thức thuộc M6.

Model-selection/tuning/imbalance/threshold thuộc M7.

## Output

`Baseline Model Registry v1.0`

+

`M5 Decision Log`

+

`M5 Open Questions`

+

`M5 Gate`

+

`Handoff to M6`

---

# 15. Canonical experiment result schema

Mỗi baseline run phải báo tối thiểu.

## Identity

- Experiment ID
- model ID
- model family
- model configuration
- training-window ID

## Data contract

- feature version
- preprocessing version
- matrix version
- training rows
- fraud rows
- validation rows
- validation fraud rows

## Learning policy

- imbalance strategy
- random state
- threshold policy

## Performance outputs

Primary:

`F1_fraud`

Secondary:

- `Recall_fraud`
- `Precision_fraud`

Diagnostic:

- TP
- FP
- FN
- TN
- Confusion Matrix

Operational:

- predicted-positive count
- predicted-positive rate

Reference:

- Accuracy

Probability:

- risk score / probability if available

## Operational evidence

- fit time
- prediction/evaluation time
- warnings
- convergence status nếu relevant
- memory note nếu đáng tin cậy

## Integrity

- FINAL TEST access = NO
- input matrix finite = PASS
- X/y alignment = PASS
- correct matrix/version = PASS

---

# 16. Quy tắc fair comparison trong M5

## Rule 1 — Không đổi nhiều yếu tố rồi gọi là model comparison

Nếu so Logistic và Tree:

cần dùng cùng:

- training window;
- feature version;
- preprocessing version;
- imbalance strategy;
- validation population;
- metric code;
- threshold policy.

Model family/config là biến khác biệt chủ động.

---

## Rule 2 — Window pair trong cùng model phải cùng config

Ví dụ hợp lệ:

`LR-LONG config A`

vs

`LR-SHORT config A`

Không hợp lệ:

`LR-LONG config A`

vs

`LR-SHORT config B`

rồi kết luận training window gây chênh lệch.

---

## Rule 3 — Không tuning baseline bằng VALIDATION

Không:

- thử nhiều `C` rồi lấy LR có F1 cao nhất;
- thử nhiều `max_depth` rồi lấy Tree có F1 cao nhất;
- thử nhiều `n_estimators` rồi lấy Forest có F1 cao nhất.

Đó là tuning và thuộc M7.

M5 baseline config phải được khai báo trước result.

---

## Rule 4 — Imbalance strategy cố định

M5:

`NONE`

Không model nào được class_weight/resampling riêng chỉ vì baseline metric thấp.

---

## Rule 5 — Threshold không phải tuning target

M5 giữ một baseline decision rule được khai báo trước.

Final numerical threshold:

`OPEN`

---

## Rule 6 — FINAL TEST không tồn tại trong modeling code path

M5 notebook không load final-test features/labels cho evaluation.

Nếu FINAL TEST vô tình được access:

`BLOCKING VIOLATION`

---

# 17. M5 notebook organization

M5 là công việc runtime-dependent.

Do đó M5.2 → M5.5 nên thực hiện bằng notebook.

Đề xuất:

`05_01_modeling_charter.md`

hoặc notebook markdown-only nếu muốn giữ cùng workflow.

Runtime notebooks:

`05_02_modeling_artifact_audit_and_runner.ipynb`

`05_03_logistic_regression_baseline.ipynb`

`05_04_decision_tree_baseline.ipynb`

`05_05_random_forest_baseline.ipynb`

M5 consolidation:

`CANON-M5.6 — Tổng hợp Baseline Model Registry, Decision Log và M5 Gate.md`

Không bắt buộc phải chia đúng file naming này nếu implementation sau giữ semantic tương đương.

---

# 18. M5 Gate

M5 chỉ PASS khi ít nhất các gate sau đạt.

## G01 — M4 artifact handoff integrity

PASS khi:

- canonical M4.7 artifacts load thành công;
- manifest/schema khớp;
- không sử dụng stale pre-fix artifacts.

---

## G02 — Modeling contract preserved

PASS khi:

- 47-column schema giữ nguyên;
- target mapping đúng;
- raw identifiers không được thêm lại;
- preprocessing không bị refit tùy tiện ở M5.

---

## G03 — Model family scope complete

PASS khi baseline runs tồn tại cho:

- Logistic Regression;
- Decision Tree;
- Random Forest.

---

## G04 — W_LONG/W_SHORT coverage

PASS khi mỗi baseline model family có controlled runs cho cả:

- W_LONG;
- W_SHORT

hoặc có documented blocking technical reason được review.

---

## G05 — No-intervention baseline

PASS khi:

`IMBALANCE_STRATEGY = NONE`

cho core M5 runs.

---

## G06 — No hidden tuning

PASS khi baseline configuration được khai báo trước comparative result và không chọn từ validation search.

---

## G07 — Probability/risk-score preservation

PASS khi model hỗ trợ probability và output đó được giữ cho M6/M7.

---

## G08 — Canonical metric implementation

PASS khi run evidence dùng cùng metric code:

- F1_fraud;
- Recall_fraud;
- Precision_fraud;
- Confusion Matrix;
- predicted-positive diagnostics;
- Accuracy reference.

---

## G09 — Reproducibility

PASS khi:

- randomness được kiểm soát;
- random state được ghi;
- config được log;
- same run có thể được tái tạo.

---

## G10 — FINAL TEST isolation

PASS khi:

`FINAL TEST ACCESS = NO`

trong M5.

---

## G11 — Experiment Registry complete

PASS khi sáu baseline run hoặc documented exceptions đều có metadata/output rõ.

---

## G12 — M6 readiness

PASS khi M6 có thể phân tích:

- lỗi;
- Precision/Recall/F1;
- Confusion Matrix;
- predicted-positive behavior;
- window/model differences

mà không phải train lại model chỉ vì thiếu prediction evidence.

---

# 19. Những điều M5 Gate không yêu cầu

M5 PASS không yêu cầu:

- F1 phải cao;
- Recall phải đạt một ngưỡng;
- Precision phải đạt một ngưỡng;
- Random Forest phải thắng;
- Logistic Regression phải thắng;
- Decision Tree phải thắng;
- W_LONG phải thắng;
- W_SHORT phải thắng;
- class_weight phải được dùng;
- SMOTE phải được dùng;
- hyperparameter phải tối ưu;
- threshold cuối phải được khóa;
- FINAL TEST phải được mở.

Nếu baseline score thấp nhưng experiment đúng:

M5 vẫn có thể PASS.

Score thấp là modeling evidence, không tự động là pipeline failure.

---

# 20. Open Questions được handoff sau M5

## O01 — Training-window preference

Status:

`OPEN`

M6 đọc comparative validation evidence.

Nếu cần robustness/tuning:

handoff M7.

---

## O02 — Model-family preference

Status:

`OPEN`

M6 phân tích baseline behavior.

M7 thực hiện selection protocol.

---

## O03 — Feature experiment

Status:

`DEFERRED`

Không thuộc core M5 baseline.

---

## O04 — Class imbalance strategy

Status:

`OPEN — M7`

Candidates đã được M3 authorize:

- NONE;
- CLASS_WEIGHT;
- RANDOM_OVERSAMPLING;
- RANDOM_UNDERSAMPLING;
- SMOTE conditional.

---

## O05 — Temporal CV

Status:

`DEFERRED TO M7`

Primary strategy nếu dùng:

`FORWARD / EXPANDING TEMPORAL CV`

Validation blocks:

`Q2 / Q3 / Q4 2018`

---

## O06 — Hyperparameter tuning

Status:

`DEFERRED TO M7`

Search space:

small / justified / declared before run.

---

## O07 — Final threshold

Status:

`DEFERRED TO M7`

Selection source:

external VALIDATION.

FINAL TEST:

prohibited for threshold selection.

---

## O08 — Final performance

Status:

`OPEN`

FINAL TEST chỉ mở sau upstream pipeline/model/threshold đã freeze.

---

# 21. M5 execution protocol với AI

M5.2–M5.5 là runtime experiment.

Quy trình cho mỗi step:

1. Xác định đúng câu hỏi experiment.
2. Đọc CANON liên quan.
3. Khóa config trước result.
4. AI tạo notebook với:
   - inputs;
   - artifact assertions;
   - model config;
   - fit;
   - predictions;
   - probability;
   - metric bundle;
   - runtime;
   - warnings;
   - gates;
   - Decision = OPEN.
5. Người thực hiện Run All trên environment thật.
6. Giữ nguyên output/error/warning.
7. AI kiểm tra:
   - execution;
   - artifact integrity;
   - comparability;
   - leakage/test isolation;
   - model warning/convergence;
   - output consistency.
8. Nếu thiếu evidence:
   - thêm check;
   - rerun.
9. Chỉ sau review mới viết nhận xét/kết luận.
10. M5.6 tổng hợp toàn milestone.

Không invent model score.

Không PASS nếu notebook chưa có runtime output thật.

---

# 22. Expected M5 sequence

```text
M4.8
Feature/preprocessing baseline contract
LOCKED
        ↓
M5.1
Modeling Charter
        ↓
M5.2
Artifact audit
+
Shared baseline runner
        ↓
M5.3
Logistic Regression
LR-LONG + LR-SHORT
        ↓
M5.4
Decision Tree
DT-LONG + DT-SHORT
        ↓
M5.5
Random Forest
RF-LONG + RF-SHORT
        ↓
M5.6
Baseline Model Registry
Decision Log
M5 Gate
        ↓
M6
Evaluation + Error Analysis
        ↓
M7
Model Selection
CV / tuning / imbalance / threshold
```

---

# 23. Decision Log khởi tạo cho M5

## M5-D01 — M5 scope

Decision:

M5 là baseline modeling milestone, không phải final model selection.

Status:

`READY TO LOCK`

---

## M5-D02 — Baseline model families

Decision:

Core model families:

- Logistic Regression;
- Decision Tree;
- Random Forest.

Status:

`INHERITED — READY TO LOCK`

---

## M5-D03 — M4 artifact contract

Decision:

Dùng canonical M4.7 persisted artifacts, không rebuild feature pipeline theo logic mới.

Status:

`INHERITED — READY TO LOCK`

---

## M5-D04 — Training windows

Decision:

Cả W_LONG và W_SHORT đều phải được giữ như modeling candidates.

Status:

`INHERITED — LOCKED`

Winner:

`OPEN`

---

## M5-D05 — Imbalance

Decision:

Core M5 baseline:

`IMBALANCE_STRATEGY = NONE`

Status:

`INHERITED — LOCKED`

---

## M5-D06 — Random state

Decision:

`42`

khi operation có randomness.

Status:

`INHERITED — LOCKED`

---

## M5-D07 — Metrics

Decision:

Primary:

`F1_fraud`

Secondary:

`Recall_fraud + Precision_fraud`

Mandatory:

Confusion Matrix + TP/FP/FN/TN.

Operational:

predicted-positive count/rate.

Accuracy:

reference only.

Status:

`INHERITED — LOCKED`

---

## M5-D08 — Probability output

Decision:

Giữ probability/risk score khi model hỗ trợ.

Status:

`INHERITED — LOCKED`

---

## M5-D09 — Threshold

Decision:

M5 không tối ưu final numerical threshold.

Status:

`INHERITED — LOCKED`

Final threshold:

`OPEN`

---

## M5-D10 — Tuning

Decision:

Systematic hyperparameter tuning không thuộc M5 baseline.

Status:

`DEFERRED TO M7`

---

## M5-D11 — FINAL TEST

Decision:

Không access trong M5.

Status:

`INHERITED — LOCKED`

---

## M5-D12 — Model/window winner

Decision:

M5 không khóa final winner.

Status:

`OPEN — REQUIRES M6/M7 EVIDENCE`

---

# 24. M5 completion definition

M5 được coi là hoàn thành khi project có thể trả lời chắc chắn:

> M4 artifacts có load và train model trực tiếp được không?

> Logistic Regression baseline đã chạy hợp lệ chưa?

> Decision Tree baseline đã chạy hợp lệ chưa?

> Random Forest baseline đã chạy hợp lệ chưa?

> Cả W_LONG và W_SHORT đã được xử lý bằng controlled setup chưa?

> Có giữ probability/risk score khi model hỗ trợ không?

> Có đủ prediction/modeling evidence để M6 phân tích lỗi không?

> Có bất kỳ FINAL TEST leakage nào không?

> Có hidden tuning hoặc imbalance intervention nào bị trộn vào baseline không?

Nếu tất cả câu trả lời cần thiết đều đạt:

`M5 — PASS`

Handoff:

`READY FOR M6 — EVALUATION`

---

# 25. Source hierarchy

Kế hoạch này kế thừa trực tiếp:

- `CANON-Quy trình thực hiện dự án.md`
- `CANON-M3.4 — Thiết kế evaluation metric strategy.md`
- `CANON-M3.6 — Thiết kế protocol so sánh training window.md`
- `CANON-M3.7 — Policy cho class imbalance, temporal validation/CV và các experiment/tuning sau.md`
- `CANON-M3.8 — Tổng hợp Experiment Specification v1.0, Decision Log, Open Questions và M3 Gate.md`
- `CANON-Kế hoạch Milestone 4.md`
- `CANON-M4.8 — Tổng hợp Feature Specification v1.0, Preprocessing Specification v1.0, Behavioral Feature Contract, Decision Log và M4 Gate.md`

Tài liệu học Tuần 7 được dùng làm background để giải thích trực giác model family, nhưng không được dùng để ghi đè temporal evaluation protocol riêng của project.

Đặc biệt:

các ví dụ random split / StratifiedKFold trong tài liệu học chung không thay thế project-specific rule:

`PAST → FUTURE`

và:

`FORWARD / EXPANDING TEMPORAL CV`

khi CV được sử dụng ở M7.

---

# 26. Final planning block

```text
Milestone 5:
MODELING BASELINE

Input:
M4 baseline-ready artifacts

Core model families:
Logistic Regression
Decision Tree
Random Forest

Training windows:
W_LONG
W_SHORT

Core runs:
6 controlled baseline runs

Imbalance strategy:
NONE

Primary metric:
F1_fraud

Secondary:
Recall_fraud
Precision_fraud

Diagnostics:
Confusion Matrix
TP / FP / FN / TN
Predicted-positive count/rate
Accuracy reference

Probability/risk score:
PRESERVE WHEN AVAILABLE

Systematic tuning:
NOT IN M5

Temporal CV:
NOT CORE M5
DEFER TO M7

Class-weight/resampling/SMOTE:
NOT CORE M5
DEFER TO M7

Final threshold:
OPEN
NOT SELECTED IN M5

FINAL TEST:
PROTECTED
NO ACCESS

M5 output:
Baseline Model Registry
Prediction/probability evidence
Experiment metadata
Decision Log
M5 Gate

Next:
M6 — Evaluation + Error Analysis
```
