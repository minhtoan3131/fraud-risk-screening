# M3.5 — Khóa vai trò và quyền sử dụng của TRAIN / VALIDATION / FINAL TEST

## 1. Mục tiêu

M3.3 đã trả lời:

`Các partition nằm ở đâu trên trục thời gian?`

M3.5 trả lời:

`Mỗi partition được phép làm gì và tuyệt đối không được phép làm gì?`

Mục tiêu của bước này là khóa data-boundary protocol để các milestone modeling phía sau không tự quyết định lại cách dùng TRAIN, VALIDATION và FINAL TEST.

M3.5 không:

- train model;
- chọn model;
- chọn training-window winner;
- chọn feature cuối;
- chọn imbalance technique;
- chọn hyperparameter;
- chọn numerical threshold;
- chạy final-test model performance.

M3.5 chỉ khóa **quyền sử dụng dữ liệu**.

---

# 2. Temporal partitions đã khóa

Từ M3.3:

## TRAIN boundary

Classifier-training rows phải thỏa:

`Timestamp < 2019-01-01`

Training-window start vẫn chưa khóa.

Hai candidate hiện tại:

`W_LONG_2015_TO_2018`

và:

`W_SHORT_2018_ONLY`.

M3.5 không chọn winner.

---

## VALIDATION

```text
2019-01-01
≤ Timestamp
< 2019-06-01
```

Tương ứng:

`2019-01 → 2019-05`

Đã audit:

`712,458 transaction`

`1,052 fraud`.

---

## FINAL TEST

```text
2019-06-01
≤ Timestamp
< 2019-11-01
```

Tương ứng:

`2019-06 → 2019-10`

Đã audit:

`722,955 transaction`

`1,035 fraud`.

---

## POST-BREAK REGION

```text
2019-11 → 2020-02
```

Đã xác nhận:

`625,025 transaction`

`0 fraud`.

Region này không thuộc core final fraud-performance test.

---

# 3. Ba khái niệm phải phân biệt

M3.5 khóa ba loại sử dụng dữ liệu khác nhau.

## 3.1. Model-learning state

Là những thứ thực sự được model học trong `fit()`.

Ví dụ:

- coefficient;
- tree structure;
- split rules;
- estimator parameters;
- learned category representation;
- các parameter khác của classifier.

Nguồn học hợp lệ:

`TRAIN only`.

---

## 3.2. Preprocessing-learning state

Là những statistic hoặc mapping phải được học từ dữ liệu.

Ví dụ:

- scaler mean/std;
- imputer statistic;
- category vocabulary;
- encoder mapping;
- learned transformation;
- feature-selection rule nếu rule được học từ target/data;
- các statistic tương tự.

Nguồn học hợp lệ:

`TRAIN only`

trừ khi một protocol nội bộ khác được khóa rõ ràng sau này.

VALIDATION và FINAL TEST không được tham gia việc fit các state này.

---

## 3.3. Causal historical state

Đây là khái niệm khác.

Behavioral feature có thể cần thông tin transaction đã xảy ra trước transaction hiện tại.

Ví dụ:

- transaction trước đó;
- số transaction trong 1 giờ trước;
- previous amount history;
- merchant đã từng gặp trước đây hay chưa.

Đối với transaction hiện tại T:

`historical event được dùng phải có Timestamp < Timestamp(T)`.

Historical state có thể chứa transaction nằm ngoài classifier-training window nếu transaction đó thực sự xảy ra trước prediction point.

Nhưng:

- không được dùng target label tương lai;
- không được dùng transaction tương lai;
- không được dùng current transaction như history của chính nó;
- transaction cùng Timestamp không làm history cho nhau.

## Kết luận quan trọng

`classifier-training rows ≠ historical-context rows`.

Đây là hai vai trò khác nhau.

---

# 4. Quyền của TRAIN

TRAIN là partition duy nhất được phép cung cấp dữ liệu cho việc học model và learned preprocessing state.

## TRAIN được phép

TRAIN được phép:

- fit preprocessing;
- fit scaler;
- fit imputer;
- học category vocabulary;
- học encoder;
- fit feature-selection mechanism nếu sau này experiment có bước học như vậy;
- fit classifier;
- fit model parameter;
- áp dụng `class_weight`;
- oversampling;
- undersampling;
- SMOTE nếu sau này được experiment protocol cho phép;
- các kỹ thuật làm thay đổi training distribution;
- tạo baseline;
- tạo các model/config candidate.

Resampling phải xảy ra:

`SAU split`

và:

`CHỈ bên trong training data`.

---

# 5. TRAIN không được phép nhìn gì?

Trong quá trình fit model hoặc learned preprocessing state, TRAIN không được sử dụng:

- feature statistics của VALIDATION;
- feature statistics của FINAL TEST;
- target distribution của future partition để điều chỉnh model trực tiếp;
- validation label trong `fit()`;
- final-test label trong `fit()`;
- category vocabulary học từ toàn bộ dataset;
- scaler/imputer statistic tính trên toàn bộ dataset;
- encoding mapping có future information;
- resampled validation;
- resampled final test.

Không được làm:

```text
concat TRAIN + VALIDATION + TEST
→ fit preprocessing
→ split lại
```

vì learned preprocessing state khi đó đã nhìn future data.

---

# 6. Quyền của VALIDATION

VALIDATION là **development decision partition**.

VALIDATION không dùng để học parameter của model đang được đánh giá.

Nó dùng để trả lời:

`Trong các lựa chọn đã fit từ TRAIN, lựa chọn nào nên được mang tiếp?`

---

# 7. VALIDATION được phép làm gì?

VALIDATION được phép:

- nhận transformation từ preprocessing đã fit trên TRAIN;
- nhận prediction;
- nhận probability/risk score;
- tính metric;
- tính Confusion Matrix;
- đọc TP / FP / FN / TN;
- tính F1;
- tính Recall;
- tính Precision;
- tính Accuracy tham khảo;
- đọc predicted-positive count/rate;
- thực hiện error analysis;
- so sánh feature candidate;
- so sánh preprocessing candidate;
- so sánh model family;
- so sánh model configuration;
- so sánh hyperparameter candidate;
- so sánh W_LONG và W_SHORT;
- so sánh imbalance strategy;
- hỗ trợ lựa chọn threshold;
- hỗ trợ các quyết định modeling lặp lại.

VALIDATION có thể được xem nhiều lần trong quá trình phát triển.

Đây chính là lý do nó **không phải final test**.

---

# 8. VALIDATION không được phép làm gì?

VALIDATION không được dùng để:

- fit scaler;
- fit imputer;
- học category vocabulary;
- fit encoder;
- update classifier parameter;
- resample;
- SMOTE;
- bổ sung row vào TRAIN một cách âm thầm;
- tính statistic rồi đưa statistic đó trở lại fit pipeline mà không ghi nhận đây là một thay đổi experiment;
- đóng vai trò như một phần training set trong khi vẫn gọi metric trên chính nó là validation generalization metric.

Ví dụ không hợp lệ:

```text
fit scaler trên TRAIN + VALIDATION
→ train model trên TRAIN
→ báo F1 trên VALIDATION
```

Vì preprocessing đã nhìn validation.

---

# 9. VALIDATION được dùng nhiều lần nhưng phải có kỷ luật

Việc validation được dùng để:

- chọn feature;
- chọn model;
- chọn training window;
- chọn imbalance strategy;
- chọn threshold;

có nghĩa project dần thích nghi với validation period.

Đó là hành vi có chủ ý của development partition.

Vì vậy mọi quyết định quan trọng dựa trên validation phải được ghi lại trong:

`Experiment Log / Decision Log`.

Không được thử hàng chục cấu hình nhưng chỉ lưu cấu hình cuối cùng như thể những lần thử trước chưa từng tồn tại.

---

# 10. Quyền của FINAL TEST

FINAL TEST là **protected evaluation partition**.

Nó chỉ được sử dụng sau khi các lựa chọn chính đã được khóa ở mức đủ để đánh giá cuối.

FINAL TEST trả lời:

`Pipeline đã được lựa chọn hoạt động thế nào trên future period chưa được dùng để lựa chọn model?`

FINAL TEST không phải development partition.

---

# 11. FINAL TEST được phép làm gì?

Sau khi pipeline phù hợp đã được khóa, FINAL TEST được phép:

- nhận transformation từ pipeline đã khóa;
- nhận prediction;
- nhận probability/risk score;
- tính metric set đã khóa ở M3.4;
- tính F1_fraud;
- Recall_fraud;
- Precision_fraud;
- Accuracy tham khảo;
- TP / FP / FN / TN;
- predicted-positive count/rate;
- tạo final evaluation report.

Sau final evaluation, có thể thực hiện:

`descriptive error analysis`

để giải thích model đã sai ở đâu.

Nhưng error analysis đó không được dùng để sửa model rồi quay lại đánh giá trên cùng FINAL TEST và vẫn gọi kết quả mới là một final test độc lập.

---

# 12. FINAL TEST không được phép làm gì?

FINAL TEST không được dùng để:

- chọn feature;
- bỏ feature;
- thêm feature;
- chọn encoding;
- chọn preprocessing;
- chọn W_LONG/W_SHORT;
- chọn model family;
- chọn hyperparameter;
- chọn class weight;
- chọn resampling strategy;
- chọn threshold;
- chọn random seed;
- quyết định sửa model vì score không đẹp;
- thử model A, xem test, rồi chuyển model B;
- thử threshold 0.5, xem test, rồi đổi thành 0.3;
- xem error rồi feature-engineer lại và tiếp tục gọi cùng test là “unseen test”.

Nguyên tắc:

```text
Nếu final-test result tham gia vào quyết định modeling,
final test đã bị tiêu hao như development data.
```

---

# 13. Test isolation áp dụng cho cả con người

Test leakage không chỉ xảy ra trong code.

Ví dụ code không gọi:

`model.fit(X_test, y_test)`

nhưng người làm project:

```text
chạy final test
→ thấy Recall thấp
→ giảm threshold
→ chạy final test lại
```

thì final test vẫn đã tham gia vào model selection.

Do đó test isolation áp dụng đồng thời cho:

- code;
- notebook workflow;
- quyết định của người thực hiện;
- quyết định của AI hỗ trợ.

---

# 14. Causal historical feature qua partition boundary

Đây là ngoại lệ quan trọng cần phân biệt với fitting.

Giả sử đang prediction một transaction validation tại thời điểm T.

Behavioral history được phép dùng transaction trước T, kể cả transaction xảy ra:

- trong TRAIN;
- trong historical warm-up region;
- hoặc trước đó trong VALIDATION;

nếu transaction đó thực sự đã xảy ra trước T.

Tương tự, khi prediction transaction FINAL TEST tại T, causal history có thể gồm transaction xảy ra trước T trong:

- historical reservoir;
- TRAIN;
- VALIDATION;
- và các transaction FINAL TEST xảy ra sớm hơn T.

Điều này mô phỏng việc hệ thống thực tế quan sát transaction stream theo thời gian.

---

# 15. Nhưng label không được chảy qua history state

Causal history ở phần trước chỉ nói về **transaction event information** có sẵn trước prediction point.

Không được sử dụng:

- fraud label của validation transaction trước đó để tạo feature cho validation transaction sau;
- fraud label của validation để tạo feature cho final test;
- fraud label của final-test transaction trước để tạo feature cho transaction final-test sau;
- investigation outcome hoặc thông tin hậu nghiệm khác không có tại prediction time.

Tức là:

```text
past observable transaction event:
CÓ THỂ dùng làm causal history

past/future target label:
KHÔNG dùng làm feature state
```

trừ khi sau này problem definition chứng minh label đó thực sự khả dụng tại prediction point, điều hiện tại chưa được khóa.

---

# 16. Learned state không được update bằng VALIDATION/TEST

Phải phân biệt:

```text
causal transaction-history state
```

và:

```text
learned preprocessing/model state
```

Trong validation/final-test evaluation:

Causal transaction-history state:

`có thể tiến theo thời gian`.

Learned scaler/imputer/encoder/model parameter:

`phải frozen`.

Ví dụ:

Nếu StandardScaler được fit trên TRAIN:

mean/std phải giữ nguyên khi transform validation và test.

Không được cập nhật mean/std khi transaction validation mới xuất hiện.

Nếu model được fit trên TRAIN:

coefficient/tree parameter phải giữ nguyên trong lúc validation/test.

Không được online-learning thêm từ validation/test trừ khi một experiment hoàn toàn khác được định nghĩa rõ.

---

# 17. Same-timestamp rule tiếp tục áp dụng

Nếu hai transaction có cùng Timestamp:

không transaction nào được xem là lịch sử của transaction còn lại.

Partition boundary không thay đổi guardrail này.

Feature pipeline sau này phải bảo đảm:

`Timestamp(history) < Timestamp(current)`

không phải:

`Timestamp(history) <= Timestamp(current)`.

---

# 18. Historical warm-up region

Transaction trước selected training-window start có thể được dùng làm:

`history warm-up`.

Ví dụ nếu W_SHORT được sử dụng:

classifier fit có thể chỉ dùng row từ 2018.

Nhưng behavioral feature của transaction 2018 vẫn có thể sử dụng transaction trước 2018 nếu strict causal rule được bảo đảm.

Older history không vì thế trở thành classifier-training data.

## Quyết định

`History availability không quyết định classifier-training membership.`

---

# 19. Quyền của POST-BREAK 2019-11 → 2020-02

Region:

`2019-11 → 2020-02`

không phải core final fraud-performance test vì không có fraud positive.

M3.5 không cho phép region này âm thầm thay thế final test.

Nó có thể được sử dụng sau này như:

- supplementary diagnostic;
- stress / abnormal-regime study;
- distribution-shift discussion;

nếu một experiment riêng định nghĩa rõ mục đích.

Nhưng kết quả ở region này không được dùng để:

- thay final fraud metric;
- làm bằng chứng rằng model phát hiện fraud tốt;
- tuning ngược core model một cách không ghi nhận.

`Post-break diagnostic role: OPTIONAL / DEFERRED`

---

# 20. Training-window comparison và partition rights

W_LONG và W_SHORT phải dùng cùng:

- VALIDATION;
- metric definitions;
- threshold policy;
- feature version;
- preprocessing version;
- model/configuration;

khi mục tiêu experiment là chỉ so training window.

VALIDATION được dùng để quyết định training-window candidate nào phù hợp hơn.

FINAL TEST không được dùng để chọn winner W_LONG/W_SHORT.

---

# 21. Hyperparameter selection

Hyperparameter candidate được fit bằng TRAIN theo protocol tương ứng.

VALIDATION được dùng để so sánh configuration.

FINAL TEST không tham gia hyperparameter tuning.

Nếu M3.7 sau này khóa temporal-CV bên trong training region, các fold nội bộ phải tiếp tục bảo vệ temporal direction và learned preprocessing boundary.

M3.5 chưa khóa CV strategy cụ thể.

`Temporal-CV strategy: OPEN → M3.7`

---

# 22. Class-imbalance technique

Nếu experiment sau này thử:

- class_weight;
- oversampling;
- undersampling;
- SMOTE;

thì technique đó chỉ được áp dụng bên trong training data.

Không resample:

`VALIDATION`

hoặc:

`FINAL TEST`.

Hai partition evaluation phải giữ distribution thật của chúng.

---

# 23. Threshold selection

Numerical threshold hiện vẫn:

`OPEN`.

VALIDATION là partition được phép dùng để lựa chọn threshold sau này.

FINAL TEST không được dùng để chọn threshold.

Sau khi threshold được khóa:

final test dùng đúng threshold đã khóa.

Không thay threshold vì final-test F1/Recall/Precision không như mong muốn.

---

# 24. Error analysis

## Trên TRAIN

Có thể dùng để:

- debug;
- hiểu model fit;
- kiểm tra underfit/overfit;
- kiểm tra pipeline.

Nhưng train metric không chứng minh generalization.

---

## Trên VALIDATION

Được phép error analysis trong quá trình phát triển.

Finding có thể dẫn tới:

- feature candidate mới;
- preprocessing mới;
- model/config mới;
- threshold experiment mới.

Nhưng mọi thay đổi phải được đánh giá lại theo protocol và ghi Decision Log.

---

## Trên FINAL TEST

Chỉ error analysis sau final evaluation cho mục tiêu:

- báo cáo;
- giải thích limitation;
- thảo luận future work.

Nếu error analysis final test dẫn tới sửa model thì model sửa đổi đó phải được xem là **một thế hệ experiment mới**, và cùng final test không còn giữ nguyên vai trò unseen final evaluation cho model mới.

---

# 25. Final refit trên TRAIN + VALIDATION

M3.5 **không tự động cho phép**:

```text
sau khi chọn model
→ gộp TRAIN + VALIDATION
→ refit
→ chạy FINAL TEST
```

Đây là một workflow có thể hợp lệ trong một số thiết kế ML, nhưng nó thay đổi training data của final model và cần protocol rõ ràng.

Hiện tại:

`TRAIN + VALIDATION final-refit policy: OPEN / NOT AUTHORIZED BY DEFAULT`

Nếu sau này muốn sử dụng, phải khóa trước khi tiêu hao FINAL TEST và ghi rõ:

- vì sao refit;
- những row nào được dùng;
- preprocessing được refit thế nào;
- threshold được xử lý thế nào;
- model/config đã được lựa chọn ở thời điểm nào.

Không được thực hiện âm thầm.

---

# 26. Quy tắc khi FINAL TEST phát hiện lỗi kỹ thuật

Không phải mọi lần rerun final test đều đồng nghĩa tuning.

Nếu phát hiện lỗi kỹ thuật thật sự như:

- đọc sai file;
- nhầm label mapping;
- metric implementation sai;
- code crash;
- partition mask sai;
- artifact corruption;

thì được phép sửa lỗi và chạy lại.

Nhưng phải ghi:

```text
Technical issue
Evidence
Code/data correction
Có hay không thay đổi model decision
Lý do rerun final test
```

Điều kiện:

lỗi phải là correctness issue, không phải:

`test score không đẹp`.

Không được dùng danh nghĩa “sửa bug” để tuning model theo final-test result.

---

# 27. Quy tắc boundary trong code

Các notebook modeling sau M3 phải dùng cùng canonical boundary:

```text
TRAIN_END_EXCLUSIVE
= 2019-01-01

VALIDATION_START
= 2019-01-01

VALIDATION_END_EXCLUSIVE
= 2019-06-01

FINAL_TEST_START
= 2019-06-01

FINAL_TEST_END_EXCLUSIVE
= 2019-11-01
```

Training start phụ thuộc training-window candidate.

Mọi notebook quan trọng nên có assertion kiểm tra:

```text
max(TRAIN timestamp) < min(VALIDATION timestamp)

max(VALIDATION timestamp) < min(FINAL TEST timestamp)

TRAIN ∩ VALIDATION = ∅

TRAIN ∩ FINAL TEST = ∅

VALIDATION ∩ FINAL TEST = ∅
```

Không được định nghĩa boundary thủ công khác nhau trong từng notebook.

---

# 28. Phân quyền tóm tắt

```text
TRAIN
=====

Được học:
YES

Fit preprocessing:
YES

Fit model:
YES

Resampling:
YES — nếu protocol cho phép

Model selection:
không tự quyết định;
candidate được đánh giá bằng VALIDATION

Threshold selection:
NO

Final performance claim:
NO


VALIDATION
==========

Được học model parameter:
NO

Fit preprocessing:
NO

Resampling:
NO

Predict / probability:
YES

Compute metrics:
YES

Feature/model/config comparison:
YES

Training-window comparison:
YES

Imbalance-strategy comparison:
YES

Threshold selection:
YES — ở giai đoạn thích hợp

Error analysis phục vụ development:
YES

Final performance claim:
NO


FINAL TEST
==========

Fit preprocessing:
NO

Fit model:
NO

Resampling:
NO

Feature selection:
NO

Model selection:
NO

Training-window selection:
NO

Hyperparameter tuning:
NO

Threshold selection:
NO

Final frozen-pipeline evaluation:
YES

Final metric report:
YES

Post-evaluation descriptive error analysis:
YES

Sửa model rồi test lại như chưa từng xem test:
NO
```

---

# 29. Decision Log M3.5

`M3.5-D01`  
TRAIN là nguồn duy nhất cho learned preprocessing state và classifier fitting.  
`Status: LOCKED`

`M3.5-D02`  
VALIDATION là development decision partition; không tham gia trực tiếp vào fit của model đang được đánh giá.  
`Status: LOCKED`

`M3.5-D03`  
VALIDATION được sử dụng cho feature/model/config/training-window/imbalance/threshold comparison theo protocol.  
`Status: LOCKED`

`M3.5-D04`  
FINAL TEST là protected evaluation partition và không tham gia model selection.  
`Status: LOCKED`

`M3.5-D05`  
Final-test feedback không được dùng để sửa pipeline rồi tiếp tục tuyên bố cùng test là independent final test.  
`Status: LOCKED`

`M3.5-D06`  
Learned model/preprocessing state phải frozen ngoài TRAIN.  
`Status: LOCKED`

`M3.5-D07`  
Causal historical transaction state được phép sử dụng prior observable transaction qua partition boundary nếu `Timestamp(history) < Timestamp(current)` và không dùng target label.  
`Status: LOCKED`

`M3.5-D08`  
Historical-context rows và classifier-training rows là hai khái niệm độc lập.  
`Status: LOCKED`

`M3.5-D09`  
Resampling chỉ được thực hiện trong training data.  
`Status: LOCKED`

`M3.5-D10`  
Threshold selection sử dụng VALIDATION, không sử dụng FINAL TEST.  
`Status: LOCKED`

`M3.5-D11`  
Final refit trên TRAIN + VALIDATION không được thực hiện mặc định; cần protocol riêng nếu sau này muốn áp dụng.  
`Status: OPEN / NOT AUTHORIZED BY DEFAULT`

`M3.5-D12`  
2019-11 → 2020-02 chỉ là candidate supplementary diagnostic/stress region, không phải core final fraud-performance test.  
`Status: DEFERRED / OPTIONAL`

---

# 30. M3.5 Gate

### GATE-01 — TRAIN được phép học gì đã rõ?

`PASS`

Learned preprocessing và model state chỉ được fit từ TRAIN.

### GATE-02 — VALIDATION có vai trò rõ?

`PASS`

VALIDATION phục vụ các quyết định phát triển và lựa chọn.

### GATE-03 — FINAL TEST được bảo vệ?

`PASS`

FINAL TEST không tham gia feature/model/config/training-window/threshold selection.

### GATE-04 — Resampling boundary rõ?

`PASS`

Chỉ TRAIN.

### GATE-05 — Threshold boundary rõ?

`PASS`

Validation selection, không final test.

### GATE-06 — Historical context có bị nhầm với training data?

`PASS`

Hai khái niệm đã được tách rõ.

### GATE-07 — Learned state và causal event state đã tách rõ?

`PASS`

Learned state frozen ngoài TRAIN; causal historical state có thể tiến theo thời gian bằng prior observable events.

### GATE-08 — Final-test human feedback có được kiểm soát?

`PASS`

Test isolation áp dụng cho cả code và quyết định của người thực hiện/AI.

### GATE-09 — Final-refit policy có bị tự ý giả định?

`PASS`

Chưa cho phép mặc định; giữ OPEN nếu sau này cần.

### GATE-10 — Có cần experiment mới để khóa M3.5?

`PASS`

Không.

M3.5 là protocol/data-boundary decision dựa trên split và guardrail đã được xác minh trước đó.

---

# 31. Kết luận M3.5

```text
TRAIN role:
LEARNING PARTITION — LOCKED

VALIDATION role:
DEVELOPMENT / SELECTION PARTITION — LOCKED

FINAL TEST role:
PROTECTED FINAL EVALUATION PARTITION — LOCKED

Learned preprocessing state:
TRAIN ONLY — LOCKED

Model fitting:
TRAIN ONLY — LOCKED

Resampling:
TRAIN ONLY — LOCKED

Feature/model/config selection:
VALIDATION — LOCKED

Training-window selection:
VALIDATION — LOCKED

Threshold selection:
VALIDATION — LOCKED

Final-test feedback for tuning:
PROHIBITED — LOCKED

Causal historical event state:
PRIOR OBSERVABLE EVENTS ALLOWED — LOCKED

Target label in historical feature state:
PROHIBITED — LOCKED

Final TRAIN+VALIDATION refit:
OPEN / NOT AUTHORIZED BY DEFAULT

Post-break diagnostic:
OPTIONAL / DEFERRED

Blocking issue:
NONE

M3.5 Gate:
PASS

M3.5 Status:
PASS — PARTITION ROLES AND PERMISSIONS LOCKED
```

## Bước tiếp theo

`Next: M3.6 — Thiết kế protocol so sánh W_LONG và W_SHORT`