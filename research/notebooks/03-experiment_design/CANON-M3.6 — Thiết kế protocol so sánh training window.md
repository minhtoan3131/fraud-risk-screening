# M3.6 — Thiết kế protocol so sánh training window

## 1. Mục tiêu

M3.6 trả lời câu hỏi:

`W_LONG và W_SHORT phải được so sánh như thế nào để khác biệt kết quả có thể quy chủ yếu cho training window?`

M3.6 không thực hiện model experiment ngay.

Bước này chỉ khóa:

- training-window candidate;
- biến được phép thay đổi;
- các yếu tố phải giữ cố định;
- validation dùng để lựa chọn như thế nào;
- metric dùng để đọc comparison;
- threshold policy phải được kiểm soát ra sao;
- historical warm-up phải được xử lý thế nào;
- computational budget;
- tie-break procedure;
- output cần ghi;
- điều kiện được phép khóa winner.

M3.6 không:

- chọn W_LONG;
- chọn W_SHORT;
- chọn model cuối;
- chọn feature cuối;
- chọn preprocessing cuối;
- chọn imbalance strategy cuối;
- chọn numerical threshold cuối;
- sử dụng FINAL TEST để chọn training window.

---

# 2. Evidence đầu vào

M3.3 đã xác nhận hai training-window candidate đều khả thi.

## W_LONG

```text
Tên:
W_LONG_2015_TO_2018

Classifier-training period:
2015-01-01
≤ Timestamp
< 2019-01-01

Transaction:
6,855,270

Fraud:
9,606
```

## W_SHORT

```text
Tên:
W_SHORT_2018_ONLY

Classifier-training period:
2018-01-01
≤ Timestamp
< 2019-01-01

Transaction:
1,721,615

Fraud:
2,491
```

Cả hai cùng kết thúc tại:

`2019-01-01`.

Do đó hai window có cùng future validation period.

M3.3 chưa có model-performance evidence để chọn winner.

---

# 3. Evaluation boundary dùng chung

Training-window comparison bắt buộc sử dụng validation đã khóa:

```text
VALIDATION

2019-01-01
≤ Timestamp
< 2019-06-01
```

Đã audit:

```text
712,458 transaction
1,052 fraud
```

Final test:

```text
2019-06-01
≤ Timestamp
< 2019-11-01
```

không được sử dụng để lựa chọn W_LONG/W_SHORT.

FINAL TEST chỉ tiếp tục được giữ protected cho đánh giá cuối.

---

# 4. Câu hỏi thực nghiệm chính

Experiment sau này phải trả lời:

> Với cùng feature, cùng preprocessing procedure, cùng model/configuration, cùng imbalance strategy, cùng validation period, cùng metric implementation và cùng threshold policy, việc thay classifier-training rows từ W_LONG sang W_SHORT làm validation performance thay đổi như thế nào?

Đây là một controlled comparison.

Không được biến câu hỏi thành:

`Pipeline A khác Pipeline B ở rất nhiều thứ thì pipeline nào tốt hơn?`

---

# 5. Biến duy nhất chủ động thay đổi

Trong primary comparison:

```text
Biến thay đổi:

TRAINING WINDOW
```

Cụ thể:

```text
Experiment LONG
→ classifier fit rows = 2015–2018

Experiment SHORT
→ classifier fit rows = 2018
```

Các yếu tố còn lại phải được kiểm soát.

---

# 6. Những yếu tố phải giữ cố định

Primary comparison phải giữ cùng:

```text
Validation boundary
Target definition
Positive-class definition
Feature version
Feature construction logic
Historical warm-up policy
Preprocessing algorithm/version
Model family
Model hyperparameter configuration
Class-imbalance strategy
Metric implementation
Threshold policy
Evaluation code
Random-state policy
Error-handling policy
```

Nếu nhiều yếu tố trên thay cùng training window, experiment không còn cô lập được tác động của training-window choice.

---

# 7. “Giữ preprocessing cố định” nghĩa là gì?

Không có nghĩa fitted scaler/encoder của W_LONG và W_SHORT phải có cùng giá trị.

Mỗi experiment phải:

```text
fit preprocessing procedure
trên chính TRAIN candidate của nó
```

Ví dụ:

W_LONG:

```text
fit scaler statistics
từ W_LONG TRAIN
```

W_SHORT:

```text
fit scaler statistics
từ W_SHORT TRAIN
```

Điều phải giống nhau là:

`preprocessing procedure / implementation / configuration`.

Các statistic học được khác nhau là hệ quả hợp lệ của việc training data khác nhau.

Không được fit preprocessing một lần trên W_LONG rồi tái sử dụng learned state đó cho W_SHORT.

Cũng không được fit trên:

`W_LONG + W_SHORT + VALIDATION`.

---

# 8. Feature version phải cố định

Hai window phải sử dụng cùng feature specification.

Ví dụ nếu experiment sau này khóa:

```text
FEATURE_VERSION = V1
```

thì cả W_LONG và W_SHORT phải dùng đúng V1.

Không được:

```text
W_LONG
→ feature cơ bản

W_SHORT
→ feature cơ bản + behavioral feature mới
```

rồi kết luận training window gây ra performance difference.

Nếu muốn thử feature khác, đó phải là experiment riêng.

---

# 9. Historical warm-up phải được kiểm soát

Classifier-training rows và historical-context rows tiếp tục là hai khái niệm khác nhau.

W_SHORT không được cố tình reset toàn bộ history tại:

`2018-01-01`

chỉ vì classifier fit bắt đầu từ 2018.

Nếu feature pipeline sử dụng historical feature thì cả hai window phải tuân theo cùng policy:

> Sử dụng prior observable transaction thực sự xảy ra trước transaction hiện tại theo strict causal rule.

Tức:

`Timestamp(history) < Timestamp(current)`.

Ví dụ W_SHORT có thể dùng pre-2018 transaction làm causal history cho transaction 2018 nếu feature specification cho phép.

Các row pre-2018 đó:

`không trở thành classifier-training rows`.

---

# 10. Không sử dụng label trong historical state

Historical warm-up chỉ sử dụng observable transaction information hợp lệ tại prediction point.

Không được dùng:

- historical fraud label;
- future fraud label;
- investigation outcome;
- post-event information không có ở prediction time.

Training-window experiment phải kiểm tra invariant này giống mọi modeling experiment khác.

---

# 11. Model/configuration phải cố định

Primary comparison chỉ được chạy sau khi có một baseline modeling configuration đủ rõ.

Ví dụ cần khóa trước:

```text
MODEL_ID
MODEL_VERSION
HYPERPARAMETER_CONFIGURATION
```

M3.6 không chọn model nào ở thời điểm hiện tại.

Không được làm:

```text
W_LONG
→ Logistic Regression

W_SHORT
→ Random Forest
```

rồi tuyên bố W_SHORT tốt hơn.

---

# 12. Không tuning riêng từng training window trong primary comparison

Primary comparison không được:

```text
tuning W_LONG riêng
→ chọn config tốt nhất cho W_LONG

tuning W_SHORT riêng
→ chọn config tốt nhất cho W_SHORT
```

rồi coi kết quả đó là tác động thuần của training window.

Trong primary experiment:

`model configuration phải giống nhau`.

Nếu sau này muốn nghiên cứu interaction:

`training window × model configuration`

thì đó là experiment riêng.

---

# 13. Class-imbalance strategy phải cố định

Nếu primary comparison sử dụng:

- không resampling;
- class_weight;
- oversampling;
- undersampling;
- hoặc strategy khác;

thì cùng strategy phải được áp dụng cho cả W_LONG và W_SHORT.

Không được:

```text
W_LONG
→ no class weight

W_SHORT
→ class_weight="balanced"
```

trong cùng training-window comparison.

M3.7 sẽ khóa policy chi tiết cho class imbalance.

---

# 14. Validation là nơi lựa chọn training window

W_LONG và W_SHORT được so sánh trên:

`VALIDATION 2019-01 → 2019-05`.

Được phép đọc lặp lại validation vì đây là development partition.

Không được sử dụng FINAL TEST score để:

- chọn W_LONG;
- chọn W_SHORT;
- xác nhận lại window winner;
- hoặc đảo quyết định window.

Training-window winner phải được quyết định trước khi mở final-test model performance.

---

# 15. Primary evaluation target

Theo M3.4:

```text
Primary metric:
F1_fraud
```

Do đó primary validation target của training-window comparison là:

`F1_fraud trên cùng validation protocol`.

Tuy nhiên F1 không được đọc một mình.

Mandatory secondary evidence:

```text
Recall_fraud
Precision_fraud
TP
FP
FN
TN
predicted_positive_count
predicted_positive_rate
Accuracy — reference only
```

---

# 16. Không được chỉ sort F1 rồi kết luận

Nếu:

```text
F1_LONG > F1_SHORT
```

điều đó làm W_LONG trở thành **provisional leader**, không tự động đủ để kết luận cuối.

Phải đọc thêm:

- Recall;
- Precision;
- FN;
- FP;
- số predicted positive;
- tính ổn định của result;
- integrity của experiment.

Ví dụ một F1 cao hơn rất nhỏ nhưng Recall giảm mạnh hoặc result thay đổi theo robustness check thì không được viết đơn giản:

`W_LONG thắng`.

---

# 17. Threshold policy là biến kiểm soát bắt buộc

F1, Recall và Precision phụ thuộc vào classification threshold.

Vì vậy trước khi primary window comparison thực sự được chạy, experiment phải khai báo rõ:

```text
THRESHOLD_POLICY
```

Policy đó phải giống cho W_LONG và W_SHORT.

Điều cần giống là:

`quy tắc xác định threshold`.

Không được:

```text
W_LONG
→ threshold policy A

W_SHORT
→ threshold policy B
```

---

# 18. M3.6 không khóa numerical threshold

M3.6 không tự chọn:

`0.5`

hoặc một numerical threshold khác.

Final threshold vẫn:

`OPEN`.

Nếu comparison sau này sử dụng một fixed comparison threshold, threshold đó phải:

- được khai báo trước khi đọc result;
- áp dụng giống nhau cho hai candidate;
- được ghi rõ là comparison threshold, không tự động trở thành final production/application threshold.

Nếu sử dụng validation-based threshold procedure, procedure đó cũng phải giống nhau cho hai candidate.

Không được nhìn kết quả của từng window rồi tự chọn threshold thuận lợi riêng.

---

# 19. Readiness gate trước khi chạy training-window experiment

Experiment W_LONG vs W_SHORT chỉ được chạy khi đã có đầy đủ:

```text
Feature version:
FROZEN FOR THIS EXPERIMENT

Preprocessing version:
FROZEN FOR THIS EXPERIMENT

Model/config:
FROZEN FOR THIS EXPERIMENT

Imbalance strategy:
FROZEN FOR THIS EXPERIMENT

Threshold policy:
FROZEN FOR THIS EXPERIMENT

Validation boundary:
LOCKED

Metric implementation:
LOCKED

Historical warm-up policy:
LOCKED
```

Nếu một mục còn chưa định nghĩa:

`DO NOT RUN WINDOW WINNER EXPERIMENT YET`.

---

# 20. Randomness policy

Nếu preprocessing/model có stochastic operation:

cả hai experiment phải sử dụng cùng random-state policy.

Project default:

`RANDOM_STATE = 42`

được dùng cho các thao tác ngẫu nhiên khi phù hợp.

Không được:

```text
thử nhiều seed cho W_LONG
thử một seed cho W_SHORT
```

hoặc:

```text
chọn seed đẹp nhất riêng cho từng window.
```

Nếu result có dấu hiệu nhạy với randomness, đó là robustness question và cần experiment bổ sung có thiết kế rõ.

---

# 21. Computational budget

Không khóa budget bằng số phút vì runtime phụ thuộc máy.

M3.6 khóa budget theo **số loại experiment**.

Primary comparison gồm đúng hai candidate:

```text
Run A:
W_LONG

Run B:
W_SHORT
```

với cùng pipeline configuration.

Không thực hiện trong primary comparison:

- exhaustive grid search;
- hàng chục training-window start khác nhau;
- window-specific tuning;
- nhiều model family cùng lúc.

Nếu primary comparison không đủ kết luận, chỉ bổ sung robustness experiment có lý do rõ ràng.

---

# 22. Runtime không phải primary winner criterion

Mỗi run nên ghi:

```text
training rows
fraud rows
fit time
prediction/evaluation time
memory information nếu thu được đáng tin cậy
```

Runtime là operational evidence.

Không được chọn W_SHORT chỉ vì nhanh hơn nếu validation performance của W_LONG có ưu thế rõ và có ý nghĩa cho project.

Ngược lại, nếu predictive evidence cuối cùng không cho thấy một preference ổn định, computational/data cost có thể trở thành parsimony tie-break.

---

# 23. Output bắt buộc của mỗi run

Mỗi training-window experiment phải lưu ít nhất:

```text
Experiment ID

Training-window ID
Training start
Training end
Training transaction count
Training fraud count

Validation start
Validation end
Validation transaction count
Validation fraud count

History warm-up policy

Feature version
Preprocessing version
Model ID
Model configuration
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

Fit time
Evaluation time

Integrity checks
Warnings
Notes
```

Không được chỉ lưu:

`F1_LONG` và `F1_SHORT`.

---

# 24. Experiment pair phải có chung comparison ID

Hai run phải được liên kết thành một comparison pair.

Ví dụ:

```text
Comparison ID:
TW-COMP-001

Run:
TW-COMP-001-LONG

Run:
TW-COMP-001-SHORT
```

Điều này giúp kiểm tra rằng hai kết quả thực sự thuộc cùng một controlled comparison.

---

# 25. Integrity checks trước khi đọc metric

Trước khi so F1, phải xác nhận:

```text
Validation rows giống nhau:
TRUE

Validation labels giống nhau:
TRUE

Feature columns/specification giống nhau:
TRUE

Preprocessing procedure giống nhau:
TRUE

Model/config giống nhau:
TRUE

Imbalance strategy giống nhau:
TRUE

Threshold policy giống nhau:
TRUE

Metric code giống nhau:
TRUE

Final test chưa được sử dụng:
TRUE

Temporal ordering hợp lệ:
TRUE

Leakage checks:
PASS
```

Nếu một check fail:

`STOP`.

Không được dùng metric để chọn window.

---

# 26. Primary decision rule

Sau khi integrity gate PASS:

### Bước 1 — So F1_fraud

Candidate có validation `F1_fraud` cao hơn là:

`PROVISIONAL LEADER`.

### Bước 2 — Đọc Recall và Precision

Kiểm tra F1 advantage được tạo ra bằng trade-off gì.

### Bước 3 — Đọc raw TP / FP / FN / TN

Chuyển metric difference về số transaction thật.

### Bước 4 — Kiểm tra operational output

Đọc predicted-positive rate và computational cost.

### Bước 5 — Kiểm tra stability/robustness nếu cần

Nếu evidence chưa tạo ra một preference có thể bảo vệ, không khóa winner.

---

# 27. Tie-break procedure

Không đặt một epsilon tùy ý kiểu:

`ΔF1 < 0.01 → hòa`.

Nếu hai candidate cho kết quả gần hoặc trade-off khó phân biệt:

### Tie-break 1

Ưu tiên candidate có evidence tốt hơn về việc giảm FN nếu F1 gần tương đương và Precision không collapse rõ rệt.

Lý do:

project là screening component và fraud bị bỏ sót là một loại lỗi cần được theo dõi đặc biệt.

Tuy nhiên Recall không thay thế F1 làm primary metric.

### Tie-break 2

Nếu Recall advantage phải trả giá bằng FP/Precision deterioration lớn, không tự động chọn.

Giữ:

`INCONCLUSIVE`.

### Tie-break 3

Thực hiện robustness procedure được M3.7 khóa.

Có thể gồm temporal robustness hoặc stochastic robustness tùy protocol sau này.

### Tie-break 4

Nếu sau robustness vẫn không có predictive preference ổn định và hai candidate được xem là tương đương đủ để phục vụ project, ưu tiên:

`W_SHORT`

theo nguyên tắc parsimony.

Lý do:

W_SHORT sử dụng:

`1,721,615`

training transaction thay vì:

`6,855,270`.

W_SHORT vì vậy có training-data footprint nhỏ hơn đáng kể.

Đây chỉ là fallback tie-break sau khi predictive evidence không phân biệt được candidate, không phải lý do chọn W_SHORT trước experiment.

---

# 28. Không được chọn W_LONG chỉ vì nhiều dữ liệu hơn

W_LONG có:

`6,855,270 transaction`

và:

`9,606 fraud`.

W_SHORT có:

`1,721,615 transaction`

và:

`2,491 fraud`.

Nhiều sample hơn có thể hữu ích.

Nhưng temporal EDA đã cho thấy distribution/fraud regime thay đổi theo thời gian.

Do đó:

`more data ≠ automatically better data`.

W_LONG phải chứng minh lợi ích trên future validation.

---

# 29. Không được chọn W_SHORT chỉ vì recent hơn

W_SHORT gần validation period hơn.

Nhưng recency không tự động chứng minh generalization tốt hơn.

Có thể lượng dữ liệu bổ sung của W_LONG giúp model học signal ổn định hơn.

Do đó:

`more recent ≠ automatically better`.

W_SHORT cũng phải chứng minh bằng validation experiment.

---

# 30. Interpretation scope của winner

Nếu W_LONG thắng trong controlled comparison với baseline pipeline X, kết luận đúng trước hết là:

> W_LONG cho validation performance tốt hơn W_SHORT dưới pipeline/configuration X và protocol đã khóa.

Không được ngay lập tức suy ra:

> W_LONG luôn tốt hơn với mọi model và mọi feature set.

Nếu các milestone sau thay đổi mạnh:

- feature representation;
- model family;
- imbalance treatment;
- hoặc modeling architecture;

thì phải đánh giá xem training-window decision cũ còn có thể mang forward hay cần verification lại.

---

# 31. Khi nào được khóa training-window winner?

Chỉ được khóa winner khi:

```text
Experiment thực tế đã chạy:
YES

Integrity gate:
PASS

Validation boundary:
đúng canonical boundary

Final test:
chưa dùng

Only intended variable changed:
training window

Primary metric:
đã có output thật

Secondary metrics:
đã kiểm tra

Confusion Matrix:
đã kiểm tra

Warnings / anomalies:
đã xử lý

Trade-off:
có thể giải thích

Robustness:
đủ theo protocol nếu primary result chưa rõ
```

Nếu thiếu evidence:

`Training-window winner = OPEN`.

---

# 32. Khi nào phải chạy lại?

Phải chạy lại controlled comparison nếu sau này thay đổi một yếu tố có khả năng làm training-window conclusion không còn tương đương, ví dụ:

- feature set thay đổi lớn;
- preprocessing representation thay đổi lớn;
- model family thay đổi;
- imbalance strategy thay đổi mạnh;
- threshold comparison protocol thay đổi;
- phát hiện leakage/bug.

Không nhất thiết rerun chỉ vì thay đổi tên notebook hoặc refactor không làm thay đổi semantics.

---

# 33. Final test không tham gia M3.6 winner decision

Trong training-window selection:

```text
FINAL TEST PERFORMANCE:
DO NOT COMPUTE FOR SELECTION
```

Boundary final test vẫn được giữ cố định, nhưng score của nó không được mở.

Sau khi:

- training window;
- feature/preprocessing;
- model;
- imbalance strategy;
- hyperparameter;
- threshold;

đã được khóa theo protocol tương ứng, final test mới được sử dụng cho final evaluation.

---

# 34. Trạng thái thực thi hiện tại

Hiện tại project đã có:

```text
Training-window candidates:
READY

Validation boundary:
LOCKED

Final-test boundary:
LOCKED

Metric strategy:
LOCKED

Partition permissions:
LOCKED

Historical warm-up principle:
LOCKED
```

Nhưng chưa có:

```text
Final feature version:
NOT YET LOCKED

Preprocessing version:
NOT YET LOCKED

Baseline model/config:
NOT YET LOCKED

Class-imbalance strategy:
NOT YET LOCKED

Concrete comparison threshold policy:
NOT YET LOCKED
```

Vì vậy:

`Training-window performance experiment chưa được chạy ở M3.6.`

Đây là trạng thái đúng, không phải thiếu sót.

---

# 35. Decision Log M3.6

`M3.6-D01`  
Training-window candidate chính thức là W_LONG_2015_TO_2018 và W_SHORT_2018_ONLY.  
`Status: LOCKED`

`M3.6-D02`  
Primary comparison chỉ chủ động thay classifier-training window.  
`Status: LOCKED`

`M3.6-D03`  
Validation period phải giống nhau: 2019-01 → 2019-05.  
`Status: LOCKED`

`M3.6-D04`  
FINAL TEST không được sử dụng để chọn training window.  
`Status: LOCKED`

`M3.6-D05`  
Feature/preprocessing/model/imbalance strategy/metric/threshold policy phải được kiểm soát.  
`Status: LOCKED`

`M3.6-D06`  
Historical warm-up policy phải giống về nguyên tắc causal; không reset history chỉ để làm W_SHORT.  
`Status: LOCKED`

`M3.6-D07`  
Primary comparison target = validation F1_fraud; Recall, Precision và Confusion Matrix là evidence bắt buộc đi kèm.  
`Status: LOCKED`

`M3.6-D08`  
Không định nghĩa arbitrary ΔF1 epsilon để ép tie-break.  
`Status: LOCKED`

`M3.6-D09`  
Nếu primary evidence không đủ phân biệt, dùng robustness protocol; nếu vẫn không có stable predictive preference, W_SHORT là parsimony fallback.  
`Status: LOCKED`

`M3.6-D10`  
Computational budget chính gồm hai primary run; không window-specific tuning trong primary comparison.  
`Status: LOCKED`

`M3.6-D11`  
Training-window winner chưa được khóa khi chưa có model experiment thật.  
`Status: REQUIRES EXPERIMENT`

---

# 36. M3.6 Gate

### GATE-01 — Candidate window đã rõ?

`PASS`

W_LONG và W_SHORT.

### GATE-02 — Chỉ biến nào được phép thay đã rõ?

`PASS`

Classifier-training window.

### GATE-03 — Các biến kiểm soát đã rõ?

`PASS`

Feature/preprocessing/model/imbalance/metric/threshold/evaluation protocol.

### GATE-04 — Validation dùng cho selection đã rõ?

`PASS`

2019-01 → 2019-05.

### GATE-05 — Final test có được bảo vệ?

`PASS`

Không dùng cho window selection.

### GATE-06 — Historical warm-up có được kiểm soát?

`PASS`

Strict causal history; không reset chỉ vì modeling-window start.

### GATE-07 — Primary target đã rõ?

`PASS`

F1_fraud + mandatory secondary evidence.

### GATE-08 — Tie-break đã có procedure?

`PASS`

Không arbitrary epsilon; dùng trade-off → robustness → parsimony fallback.

### GATE-09 — Computational budget đã rõ?

`PASS`

Hai primary controlled runs; robustness chỉ khi có lý do.

### GATE-10 — Winner có bị khóa khi chưa có experiment?

`PASS`

Không.

---

# 37. Kết luận M3.6

```text
Training-window candidates:
W_LONG_2015_TO_2018
W_SHORT_2018_ONLY
— LOCKED

Comparison principle:
ONLY TRAINING WINDOW CHANGES
— LOCKED

Shared validation:
2019-01 → 2019-05
— LOCKED

Final test for selection:
PROHIBITED
— LOCKED

Primary comparison metric:
F1_fraud
— LOCKED

Secondary evidence:
Recall_fraud
Precision_fraud
Confusion Matrix
— LOCKED

Historical warm-up policy:
STRICT CAUSAL / SAME PRINCIPLE
— LOCKED

Window-specific tuning in primary comparison:
PROHIBITED
— LOCKED

Training-window winner:
OPEN / REQUIRES EXPERIMENT

Blocking issue for protocol:
NONE

M3.6 Gate:
PASS

M3.6 Status:
PASS — TRAINING-WINDOW COMPARISON PROTOCOL LOCKED
```

## Handoff

Khi M4/modeling đã tạo được một baseline pipeline đủ rõ, workflow thực nghiệm phải là:

```text
Freeze feature/preprocessing/model/config
        ↓
Freeze imbalance + threshold policy
        ↓
Run W_LONG
        ↓
Run W_SHORT
        ↓
Integrity check
        ↓
Compare VALIDATION only
        ↓
F1 + Recall + Precision + CM
        ↓
Nếu chưa rõ:
robustness protocol
        ↓
Lock / keep OPEN
```

Không mở FINAL TEST trong quá trình này.

## Bước tiếp theo

`Next: M3.7 — Thiết kế policy cho class imbalance, temporal validation/CV và các experiment/tuning sau`