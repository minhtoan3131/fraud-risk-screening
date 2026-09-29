# M3.4 — Thiết kế evaluation metric strategy

## 1. Mục tiêu

M3.4 trả lời câu hỏi:

`Model sẽ được đọc, so sánh và lựa chọn bằng những tiêu chí nào?`

M3.4 không train model và không tìm metric value thực tế.

Bước này chỉ khóa:

- Positive class;
- ý nghĩa TP / TN / FP / FN trong project;
- primary metric;
- secondary metrics;
- diagnostic bắt buộc;
- vai trò của Accuracy;
- nguyên tắc sử dụng probability và threshold;
- quy tắc so sánh model;
- quy tắc báo cáo final test.

M3.4 không khóa:

- model cuối;
- training-window winner;
- class-imbalance technique;
- hyperparameter;
- numerical classification threshold cuối;
- final model performance.

---

# 2. Evidence đầu vào

M3.3 đã khóa temporal evaluation boundary:

Validation:

`2019-01-01 ≤ Timestamp < 2019-06-01`

với:

`712,458 transaction`  
`1,052 fraud`  
`fraud rate ≈ 0.147658%`.

Final test:

`2019-06-01 ≤ Timestamp < 2019-11-01`

với:

`722,955 transaction`  
`1,035 fraud`  
`fraud rate ≈ 0.143162%`.

Như vậy positive class tiếp tục rất hiếm trong cả validation và final test.

Nếu một classifier dự đoán toàn bộ validation là non-fraud, Accuracy lý thuyết vẫn xấp xỉ:

`99.852%`.

Nếu làm tương tự trên final test, Accuracy vẫn xấp xỉ:

`99.857%`.

Do đó một Accuracy rất cao không tự động chứng minh model phát hiện được fraud.

Finding này nhất quán với guardrail đã khóa từ M3.1:

`Accuracy không được sử dụng đơn độc làm cơ sở lựa chọn model.`

---

# 3. Positive class

Positive class của project được khóa là:

`Fraud / transaction có dấu hiệu gian lận theo ground truth của dataset`.

Negative class:

`Non-fraud`.

Mọi Precision, Recall, F1 và TP/FP/FN/TN trong project phải được diễn giải theo positive class này.

Không được để thư viện hoặc encoding làm thay đổi ngầm ý nghĩa của positive class.

---

# 4. Ý nghĩa TP / TN / FP / FN trong project

## True Positive — TP

Ground truth:

`Fraud`

Model:

`Positive / suspicious`.

Ý nghĩa:

Transaction có nhãn fraud được hệ thống screening phát hiện và đưa vào nhóm cần chú ý.

---

## False Negative — FN

Ground truth:

`Fraud`

Model:

`Negative / non-suspicious`.

Ý nghĩa:

Một transaction fraud trong ground truth bị hệ thống screening bỏ sót.

FN làm giảm Recall.

Đây là loại lỗi cần được theo dõi đặc biệt vì hệ thống có nhiệm vụ hỗ trợ phát hiện dấu hiệu gian lận.

Tuy nhiên project không được vì muốn giảm FN mà bỏ qua hoàn toàn FP.

---

## False Positive — FP

Ground truth:

`Non-fraud`

Model:

`Positive / suspicious`.

Ý nghĩa:

Một transaction non-fraud bị hệ thống đưa vào nhóm đáng ngờ.

FP làm giảm Precision.

Trong mô hình screening, FP không có nghĩa transaction bị khóa hoặc bị kết luận là fraud, nhưng quá nhiều FP có thể làm tăng số lượng cảnh báo hoặc khối lượng review phía sau.

Do đó Precision phải được theo dõi song song với Recall.

---

## True Negative — TN

Ground truth:

`Non-fraud`

Model:

`Negative / non-suspicious`.

Ý nghĩa:

Transaction non-fraud được model giữ ở lớp Negative.

Do Negative class chiếm áp đảo dataset, TN có thể rất lớn và làm Accuracy cao ngay cả khi khả năng phát hiện fraud yếu.

---

# 5. Vì sao không chọn Accuracy làm primary metric

Dataset có class imbalance cực mạnh.

Trong validation đã khóa:

fraud chỉ chiếm khoảng `0.147658%`.

Trong final test:

fraud chỉ chiếm khoảng `0.143162%`.

Do đó classifier luôn dự đoán:

`non-fraud`

vẫn có thể đạt Accuracy khoảng `99.85%`.

Nhưng classifier đó có:

`Recall fraud = 0`.

Vì vậy Accuracy chủ yếu bị chi phối bởi Negative class và không đủ để trả lời câu hỏi quan trọng nhất:

`Model có phát hiện được fraud hay không?`

## Kết luận

`Accuracy: REFERENCE ONLY`

Accuracy vẫn được báo cáo để mô tả tỷ lệ đúng tổng thể và hỗ trợ sanity check.

Accuracy không được:

- làm primary metric;
- dùng một mình để chọn model;
- dùng một mình để tuyên bố model tốt;
- dùng một mình để so sánh training window hoặc imbalance strategy.

---

# 6. Primary metric

## Quyết định

Primary classification metric của project:

`F1-score của fraud class`.

Viết ngắn trong experiment log:

`Primary metric = F1_fraud`.

## Lý do

Recall một mình tập trung vào FN nhưng có thể được đẩy rất cao bằng cách dự đoán quá nhiều transaction là Positive.

Precision một mình tập trung vào FP nhưng có thể được đẩy rất cao bằng cách chỉ cảnh báo một số rất ít transaction mà model chắc chắn nhất.

Project hiện chưa có:

- cost matrix nghiệp vụ;
- review-capacity constraint;
- monetary cost cho FN;
- monetary cost cho FP;
- yêu cầu kiểu “Recall phải ≥ X%”;
- hoặc “Precision phải ≥ Y%”.

Do đó chưa có bằng chứng để gán trọng số nghiệp vụ chính xác cho Recall hoặc Precision.

F1 tạo một tiêu chí cân bằng giữa hai metric và giảm nguy cơ một phía rất tốt che khuất phía còn lại.

F1 cũng không trực tiếp sử dụng TN, nên phù hợp hơn Accuracy trong dataset có Negative class áp đảo.

## Giới hạn diễn giải

Việc chọn F1 không có nghĩa:

`FP và FN có chi phí nghiệp vụ bằng nhau`.

Nó chỉ có nghĩa:

`trong phạm vi project hiện tại chưa có evidence đủ để định nghĩa cost ratio khác`.

Nếu milestone sau có bằng chứng thực tế hoặc yêu cầu nghiệp vụ mới, metric strategy có thể được điều chỉnh bằng Decision Log.

## Quyết định

`M3.4-D01: F1_fraud = PRIMARY METRIC — LOCKED`

---

# 7. Secondary metric 1 — Recall

Recall trả lời:

`Trong toàn bộ fraud thật, model tìm được bao nhiêu?`

Recall giảm khi FN tăng.

Trong project này, Recall phải luôn được báo cáo cùng F1 vì F1 có thể che mất việc model đang thiên về Precision hay Recall.

Một model có F1 tương đối tốt nhưng Recall thấp vẫn có thể bỏ sót nhiều fraud.

## Quyết định

`Recall_fraud = MANDATORY SECONDARY METRIC`

Recall không được tối ưu một mình mà không đọc Precision.

`M3.4-D02: Recall_fraud = SECONDARY / FRAUD-COVERAGE METRIC — LOCKED`

---

# 8. Secondary metric 2 — Precision

Precision trả lời:

`Trong các transaction model đưa vào nhóm Positive, bao nhiêu transaction thực sự fraud?`

Precision giảm khi FP tăng.

Đối với một screening component, Precision giúp đọc chất lượng của nhóm cảnh báo.

Nếu Precision quá thấp, hệ thống có thể tạo rất nhiều Positive prediction nhưng phần lớn là non-fraud.

Do đó model có Recall cao nhưng Precision collapse không được tự động xem là tốt.

## Quyết định

`Precision_fraud = MANDATORY SECONDARY METRIC`

`M3.4-D03: Precision_fraud = SECONDARY / ALERT-QUALITY METRIC — LOCKED`

---

# 9. Confusion Matrix là diagnostic bắt buộc

Mọi evaluation chính thức phải báo cáo tối thiểu:

`TP`  
`FP`  
`FN`  
`TN`.

Không chỉ báo:

`F1 = ...`

mà không cho biết model tạo ra loại lỗi nào.

Confusion Matrix được dùng để:

- xem số fraud phát hiện đúng;
- xem số fraud bị bỏ sót;
- xem số non-fraud bị cảnh báo nhầm;
- kiểm tra metric có hợp lý không;
- hỗ trợ error analysis sau này.

Đối với dataset rất lớn, raw count đặc biệt quan trọng.

Một thay đổi metric nhỏ có thể tương ứng với hàng nghìn FP hoặc hàng chục/hàng trăm FN khác nhau.

## Quyết định

`Confusion Matrix = MANDATORY DIAGNOSTIC`

`M3.4-D04: TP / FP / FN / TN phải có trong mọi evaluation chính — LOCKED`

---

# 10. Predicted-positive count và alert rate

Ngoài Confusion Matrix, mỗi evaluation nên báo:

`predicted_positive_count = TP + FP`

và:

`predicted_positive_rate`.

Hai đại lượng này không phải primary metric.

Chúng giúp diễn giải model theo góc độ screening:

`Model đang đưa bao nhiêu transaction vào nhóm cần chú ý?`

Ví dụ hai model có Recall gần nhau nhưng một model flag số transaction gấp nhiều lần model kia thì đó là khác biệt cần biết.

## Quyết định

`Predicted-positive count/rate = MANDATORY OPERATIONAL DIAGNOSTIC`

Không dùng riêng chúng để chọn model.

---

# 11. Metric set chuẩn của project

Mỗi experiment classification chính thức phải báo tối thiểu:

```text
Primary:
F1_fraud

Secondary:
Recall_fraud
Precision_fraud

Diagnostic:
TP
FP
FN
TN
predicted_positive_count
predicted_positive_rate

Reference:
Accuracy
```

Không được báo một con số duy nhất rồi kết luận model tốt hơn.

---

# 12. Probability output

Định hướng đề tài là screening component có thể tạo:

`probability / risk score`

trước khi chuyển sang quyết định class.

Vì vậy đối với model hỗ trợ probability, pipeline modeling nên giữ:

`predict_proba()` hoặc score tương đương.

Probability có hai vai trò:

1. cung cấp risk signal cho tầng ứng dụng;
2. cho phép nghiên cứu threshold trên validation.

M3.4 chưa đánh giá probability calibration.

Một probability `0.8` chưa được diễn giải thành:

`80% xác suất fraud thực tế`

nếu model chưa được kiểm tra calibration.

## Quyết định

`Probability output: REQUIRED WHEN MODEL SUPPORTS IT`

`Probability calibration: DEFERRED / không thuộc core M3.4`

---

# 13. Threshold policy

Classification threshold cuối:

`CHƯA KHÓA`.

M3.4 không chọn:

`0.5`

hay bất kỳ giá trị numerical nào khác làm final threshold.

Threshold ảnh hưởng trực tiếp:

`predicted Positive`

→ `TP / FP / FN / TN`

→ `Precision / Recall / F1`.

Do đó một F1 value luôn phải được đọc cùng threshold policy đã tạo ra prediction đó.

## Guardrail

Threshold cuối phải được lựa chọn bằng validation data theo protocol được khóa trước.

Final test không được dùng để:

`thử threshold → xem metric → đổi threshold → test lại`.

## Quyết định

`M3.4-D05: Final numeric threshold = OPEN`

`M3.4-D06: Threshold selection phải diễn ra trên validation, không trên final test — LOCKED`

---

# 14. Quan hệ giữa F1 và threshold

Vì F1 phụ thuộc threshold, việc khóa F1 làm primary metric không đồng nghĩa numerical threshold đã được khóa.

Trong mọi experiment dùng F1 để so sánh, experiment log phải ghi rõ:

`threshold policy`.

Ví dụ:

`default model threshold`

hoặc:

`validation-selected threshold theo protocol X`.

Hai model được so sánh bằng F1 chỉ được xem là comparison công bằng nếu threshold policy của chúng tương thích với protocol của experiment.

Không được:

- dùng threshold mặc định cho model A;
- tối ưu threshold cho model B;
- rồi kết luận B tốt hơn bằng F1.

---

# 15. Probability/ranking metrics nâng cao

Các metric như:

`ROC-AUC`

hoặc:

`PR-AUC / Average Precision`

không được đưa vào core metric set bắt buộc của phiên bản M3.4 này.

Lý do:

- tài liệu học cốt lõi hiện tập trung vào Confusion Matrix, Accuracy, Precision, Recall, F1, probability và threshold;
- mục tiêu môn học ưu tiên pipeline rõ ràng và khả năng giải thích;
- không cần mở rộng metric chỉ vì metric đó tồn tại.

Tuy nhiên các metric ranking có thể được bổ sung sau như diagnostic nếu experiment thực tế cho thấy cần đánh giá chất lượng score độc lập với threshold.

Nếu bổ sung, phải ghi Decision Log và định nghĩa rõ vai trò trước khi dùng để lựa chọn model.

## Quyết định

`Advanced ranking metric: OPTIONAL / DEFERRED`

Không phải primary metric hiện tại.

---

# 16. Quy tắc so sánh model

Một comparison chỉ hợp lệ khi các model được đánh giá trên cùng:

- validation period;
- target definition;
- positive-class definition;
- feature version nếu feature không phải biến đang thử;
- preprocessing protocol;
- metric implementation;
- threshold policy;
- evaluation code.

Nếu đang so một yếu tố cụ thể, chỉ yếu tố đó nên thay đổi khi có thể.

Ví dụ khi so training window:

```text
Thay đổi:
W_LONG vs W_SHORT

Giữ cố định:
validation
feature set
preprocessing
model configuration
metric strategy
threshold policy
```

Mọi comparison phải đọc:

`F1 + Recall + Precision + Confusion Matrix`.

Không được chỉ sort theo F1 rồi bỏ qua trade-off.

---

# 17. Cách diễn giải khi F1 gần nhau

Nếu hai candidate có F1 rất gần nhau nhưng Precision và Recall khác đáng kể, không được viết đơn giản:

`Model A thắng`.

Phải mô tả:

- model nào bỏ sót ít fraud hơn;
- model nào tạo ít false alert hơn;
- chênh lệch F1 lớn hay nhỏ;
- chênh lệch raw FP/FN là bao nhiêu;
- có đủ bằng chứng để xem khác biệt là có ý nghĩa cho project hay không.

M3.4 không định nghĩa một epsilon tùy ý như:

`ΔF1 < 0.01 = bằng nhau`.

Nếu sau này cần một tie-breaking rule định lượng, phải dựa vào evidence experiment và ghi Decision Log.

---

# 18. Validation metric policy

Validation là partition dùng cho các vòng lựa chọn sau này.

Các metric được phép đọc lặp lại trên validation theo experiment protocol:

- F1;
- Recall;
- Precision;
- Confusion Matrix;
- Accuracy;
- predicted-positive statistics;
- probability-based diagnostics nếu được bổ sung hợp lệ.

Validation có thể hỗ trợ:

- model comparison;
- training-window comparison;
- feature/preprocessing comparison;
- imbalance-strategy comparison;
- threshold selection sau này.

Chi tiết quyền hạn partition được khóa tiếp ở M3.5.

---

# 19. Final-test metric policy

Final test:

`2019-06 → 2019-10`

phải được giữ protected.

Khi model pipeline và threshold đã được khóa ở giai đoạn phù hợp, final test sẽ báo cùng metric set:

```text
F1_fraud
Recall_fraud
Precision_fraud
Accuracy
TP
FP
FN
TN
predicted_positive_count
predicted_positive_rate
```

Final test không được dùng để thay đổi:

- feature;
- preprocessing;
- training window;
- model;
- hyperparameter;
- imbalance strategy;
- threshold.

Nếu final-test result gây thất vọng, kết quả đó vẫn là final-test evidence.

Không được quay lại tuning rồi gọi lần test tiếp theo là cùng một “final test sạch”.

---

# 20. Không overclaim từ metric

Model đạt F1/Recall/Precision tốt trên artifact này chỉ cho phép nói:

`Model cho kết quả như vậy trên temporal evaluation protocol của dataset synthetic hiện tại.`

Không được suy ra trực tiếp:

- hiệu năng trong ngân hàng thật;
- tỷ lệ fraud thật ngoài đời;
- tác động tài chính thực tế;
- khả năng production;
- cost saving thực tế.

Project là một ML risk-screening study trên synthetic dataset.

---

# 21. Decision Log M3.4

`M3.4-D01`  
Positive class = fraud theo ground truth dataset.  
`Status: LOCKED`

`M3.4-D02`  
F1 của fraud class là primary classification metric.  
`Status: LOCKED`

`M3.4-D03`  
Recall và Precision của fraud class là mandatory secondary metrics.  
`Status: LOCKED`

`M3.4-D04`  
Confusion Matrix và raw TP/FP/FN/TN là mandatory diagnostic.  
`Status: LOCKED`

`M3.4-D05`  
Predicted-positive count/rate được báo cáo như operational diagnostic.  
`Status: LOCKED`

`M3.4-D06`  
Accuracy chỉ là reference metric, không được dùng đơn độc để chọn model.  
`Status: LOCKED`

`M3.4-D07`  
Probability/risk score phải được giữ khi model hỗ trợ.  
`Status: LOCKED`

`M3.4-D08`  
Final numeric threshold chưa được chọn ở M3.4.  
`Status: OPEN`

`M3.4-D09`  
Threshold selection chỉ được sử dụng validation protocol, không được dùng final test.  
`Status: LOCKED`

`M3.4-D10`  
ROC-AUC / PR-AUC hoặc ranking metric nâng cao chưa thuộc core metric set; có thể bổ sung sau bằng Decision Log nếu cần.  
`Status: DEFERRED / OPTIONAL`

---

# 22. Những điều M3.4 chưa chứng minh

M3.4 chưa chứng minh:

- model nào có F1 cao nhất;
- Recall thực tế đạt bao nhiêu;
- Precision thực tế đạt bao nhiêu;
- W_LONG hay W_SHORT tốt hơn;
- threshold nào tối ưu;
- class_weight/resampling có giúp model hay không;
- behavioral feature có cải thiện metric hay không.

Các câu hỏi trên cần model experiment thật.

Không được biến metric strategy thành model-performance claim.

---

# 23. M3.4 Gate

### GATE-01 — Positive class đã rõ?

`PASS`

Positive = fraud.

### GATE-02 — Primary metric đã rõ?

`PASS`

`F1_fraud`.

### GATE-03 — Secondary metrics đã rõ?

`PASS`

`Recall_fraud + Precision_fraud`.

### GATE-04 — Error diagnostic đã rõ?

`PASS`

Confusion Matrix + TP/FP/FN/TN.

### GATE-05 — Accuracy role đã rõ?

`PASS`

Reference only.

### GATE-06 — Probability role đã rõ?

`PASS`

Giữ probability/risk score khi model hỗ trợ.

### GATE-07 — Threshold có bị khóa sớm không?

`PASS`

Numeric threshold vẫn OPEN.

### GATE-08 — Final test có được bảo vệ không?

`PASS`

Threshold và model selection không sử dụng final-test feedback.

### GATE-09 — Có metric decision nào đang giả vờ là model result không?

`PASS`

M3.4 mới khóa protocol, chưa có model-performance claim.

---

# 24. Kết luận M3.4

`Primary metric: F1_fraud — LOCKED`

`Secondary metrics: Recall_fraud + Precision_fraud — LOCKED`

`Mandatory diagnostic: Confusion Matrix + TP/FP/FN/TN — LOCKED`

`Operational diagnostic: predicted-positive count/rate — LOCKED`

`Accuracy: REFERENCE ONLY — LOCKED`

`Probability output: REQUIRED WHEN AVAILABLE — LOCKED`

`Final numeric threshold: OPEN`

`Advanced ranking metrics: OPTIONAL / DEFERRED`

`Blocking issue: NONE`

`M3.4 Gate: PASS`

`M3.4 Status: PASS — EVALUATION METRIC STRATEGY LOCKED`

## Bước tiếp theo

`Next: M3.5 — Khóa vai trò và quyền sử dụng của TRAIN / VALIDATION / FINAL TEST`