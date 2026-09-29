# M3.7 — Policy cho class imbalance, temporal validation/CV và các experiment/tuning sau

## 1. Mục tiêu

M3.7 trả lời ba câu hỏi:

`Class imbalance được phép xử lý ở đâu và như thế nào?`

`Cross-validation của project phải được thiết kế thế nào để không phá temporal ordering?`

`Hyperparameter tuning và các robustness experiment sau này được phép sử dụng dữ liệu nào?`

M3.7 không:

- train model cuối;
- chọn imbalance strategy thắng;
- chọn class_weight cuối;
- chọn sampling ratio cuối;
- chọn hyperparameter cuối;
- chọn numerical threshold cuối;
- chọn training-window winner;
- sử dụng FINAL TEST.

M3.7 chỉ khóa **experiment policy**.

---

# 2. Evidence đầu vào

Project đã xác nhận class imbalance rất mạnh.

Fraud là positive class hiếm.

M3.4 đã khóa:

`Primary metric = F1_fraud`

với:

`Recall_fraud + Precision_fraud`

và Confusion Matrix là evidence bắt buộc.

M3.5 đã khóa:

- learned preprocessing/model state chỉ fit từ TRAIN;
- resampling chỉ được thực hiện trên TRAIN;
- VALIDATION dùng cho development/selection;
- FINAL TEST không tham gia tuning;
- threshold được lựa chọn bằng VALIDATION, không FINAL TEST.

M3.6 đã khóa:

- training-window comparison chỉ thay training window;
- robustness procedure có thể được dùng nếu evidence chưa đủ rõ;
- FINAL TEST không tham gia lựa chọn W_LONG/W_SHORT.

M3.7 kế thừa toàn bộ các quyết định trên.

---

# 3. Nguyên tắc tổng quát cho class imbalance

Class imbalance là đặc điểm của bài toán, không tự động là lỗi dữ liệu.

Không được suy luận:

```text
fraud rất hiếm
→ bắt buộc phải dùng SMOTE
```

hoặc:

```text
fraud rất hiếm
→ bắt buộc phải cân bằng 50/50
```

Cách đúng:

```text
baseline
→ đánh giá lỗi thật
→ thử một intervention có kiểm soát
→ đánh giá lại
→ quyết định từ evidence
```

## Kết luận

`Imbalance treatment is experimental, not automatic.`

---

# 4. Baseline không xử lý imbalance là bắt buộc

Trước khi đánh giá lợi ích của class_weight hoặc resampling phải có ít nhất một baseline:

```text
IMBALANCE_STRATEGY = NONE
```

Baseline dùng distribution TRAIN tự nhiên.

Mục đích:

- biết model cơ bản đang bỏ sót fraud đến mức nào;
- tạo mốc để đo intervention có thực sự cải thiện;
- tránh đưa kỹ thuật phức tạp vào pipeline mà không có bằng chứng cần thiết.

## Quyết định

`M3.7-D01: No-intervention baseline = REQUIRED`

---

# 5. Candidate class-imbalance strategy

Các strategy được phép nghiên cứu sau baseline gồm:

```text
NONE

CLASS_WEIGHT

RANDOM_OVERSAMPLING

RANDOM_UNDERSAMPLING

SMOTE — CONDITIONAL
```

Danh sách này là candidate set.

Không strategy nào được xem là winner ở M3.7.

---

# 6. Class weight

Nếu model hỗ trợ class weighting, đây là một candidate phù hợp để thử sớm vì:

- không tạo synthetic row;
- không loại bỏ majority row;
- không làm thay đổi số lượng training transaction;
- tương đối đơn giản để giải thích.

Ví dụ:

`class_weight="balanced"`

có thể là một experiment candidate.

Nhưng:

`class_weight="balanced" ≠ automatically better`.

Nó vẫn phải được so với baseline trên cùng validation protocol.

## Quyết định

`CLASS_WEIGHT: AUTHORIZED CANDIDATE`

`Final class-weight policy: OPEN / REQUIRES EXPERIMENT`

---

# 7. Random Oversampling

Random Oversampling chỉ được thực hiện trên training data.

Không được oversample trước temporal split.

Không được oversample:

- validation;
- final test.

Random Oversampling lặp lại minority samples; nó không tạo information mới.

Do đó phải theo dõi:

- overfitting;
- training-size growth;
- runtime/memory;
- validation Precision/Recall/F1.

Không mặc định oversample tới tỷ lệ 50/50.

Sampling ratio phải là parameter của experiment và được ghi lại.

## Quyết định

`RANDOM_OVERSAMPLING: AUTHORIZED CANDIDATE`

`Final oversampling ratio: OPEN`

---

# 8. Random Undersampling

Random Undersampling chỉ giảm majority rows trong training portion.

Không được undersample:

- validation;
- final test.

Dataset có majority class rất lớn, nên undersampling có thể giảm computational cost.

Nhưng nó cũng có thể loại bỏ useful majority information.

Do đó:

`training nhanh hơn`

không tự động đồng nghĩa:

`model tốt hơn`.

## Quyết định

`RANDOM_UNDERSAMPLING: AUTHORIZED CANDIDATE`

`Final undersampling ratio: OPEN`

---

# 9. SMOTE không phải default strategy

SMOTE tạo synthetic minority samples trong feature space.

Project hiện chưa khóa:

- final feature set;
- final categorical encoding;
- numerical representation;
- final preprocessing pipeline.

Dataset lại chứa nhiều categorical feature và high-cardinality feature.

Vì vậy M3.7 không cho phép mặc định thêm SMOTE vào baseline.

Trước khi SMOTE được dùng, phải chứng minh rằng:

```text
feature representation đã phù hợp
sampler nằm đúng trong training fold
không synthetic validation/test
categorical representation không bị nội suy vô nghĩa
pipeline không leakage
```

Nếu các điều kiện chưa thỏa:

`DO NOT USE SMOTE`.

## Quyết định

`SMOTE: CONDITIONAL / DEFERRED`

Không phải core baseline của Model V1.

---

# 10. Quy tắc tuyệt đối cho resampling

Workflow tối thiểu:

```text
RAW DATA
   ↓
TEMPORAL SPLIT
   ↓
TRAIN
   ↓
fold split nếu có CV
   ↓
fit preprocessing trên fold-TRAIN
   ↓
resample fold-TRAIN nếu experiment yêu cầu
   ↓
fit model
   ↓
predict fold-validation
```

Sampler không được nhìn:

- fold-validation;
- external VALIDATION;
- FINAL TEST.

---

# 11. Resampling phải nằm bên trong CV fold

Sai:

```text
resample toàn TRAIN
→ temporal CV
```

vì một sample hoặc synthetic information tạo từ phần dữ liệu sau có thể ảnh hưởng fold validation trước khi fold được tách.

Đúng:

```text
temporal fold split
→ resample chỉ fold-training portion
→ fit model
→ đánh giá fold-validation
```

## Kết luận

`Sampler boundary = fold-training only`

khi CV được sử dụng.

---

# 12. Validation/test giữ distribution tự nhiên

Không được “cân bằng” validation hoặc final test cho dễ đọc metric.

Hai partition này phải giữ fraud prevalence thực tế của temporal period tương ứng.

Mục tiêu evaluation là xem model hoạt động trên distribution thật của partition, không phải trên distribution nhân tạo thuận tiện.

## Quyết định

```text
VALIDATION resampling:
PROHIBITED

FINAL TEST resampling:
PROHIBITED
```

---

# 13. Không thay imbalance strategy và threshold cùng lúc

Class imbalance treatment và threshold là hai intervention khác nhau.

Ví dụ:

`class_weight`

thay đổi cách model học.

`threshold`

thay đổi cách probability/score được chuyển thành predicted class.

Không được:

```text
Baseline:
no class_weight + threshold 0.5

Candidate:
class_weight + threshold 0.2
```

rồi kết luận:

`class_weight giúp model tốt hơn`.

Không biết improvement đến từ class_weight hay threshold.

## Quyết định

Primary imbalance comparison:

`threshold policy phải giữ cố định`.

Threshold optimization diễn ra sau khi upstream candidate đủ ổn định.

---

# 14. Thứ tự experiment cho imbalance

Recommended experiment sequence:

```text
STEP 1
No-intervention baseline

        ↓

STEP 2
Class-weight candidate nếu model hỗ trợ

        ↓

STEP 3
Nếu evidence vẫn cho thấy cần xử lý:
Random Under/Over Sampling candidate

        ↓

STEP 4
Chỉ khi representation phù hợp và có lý do:
SMOTE experiment

        ↓

STEP 5
Sau khi learning strategy đủ ổn định:
threshold selection trên VALIDATION
```

Không chạy mọi kỹ thuật cùng lúc chỉ vì chúng tồn tại.

---

# 15. Vì sao random StratifiedKFold không phải primary CV

Trong classification thông thường, StratifiedKFold hữu ích để giữ class proportion giữa các fold.

Nhưng project này đã xác nhận:

`temporal shift`.

Main evaluation cũng đã khóa theo:

`past → future`.

Nếu dùng:

```text
StratifiedKFold
+ shuffle=True
```

thì transaction tương lai có thể nằm trong training fold trong khi transaction quá khứ nằm trong validation fold.

Điều đó không phản ánh câu hỏi triển khai của project.

## Quyết định

```text
Random shuffled KFold:
NOT AUTHORIZED FOR PRIMARY MODEL SELECTION

Random shuffled StratifiedKFold:
NOT AUTHORIZED FOR PRIMARY MODEL SELECTION
```

Class imbalance không phải lý do để phá temporal ordering.

---

# 16. CV strategy chính thức

Nếu Cross-validation được sử dụng, strategy chính của project là:

`forward / expanding temporal validation`.

Mỗi fold phải thỏa:

```text
all training timestamps
<
all validation timestamps
```

Không shuffle.

Không cho future row đi ngược vào fold-training.

---

# 17. Temporal CV template

Để cùng một protocol sử dụng được cho cả W_LONG và W_SHORT, M3.7 khóa ba temporal validation block trong năm 2018.

## Fold 1

Validation:

```text
2018-04-01
≤ Timestamp
< 2018-07-01
```

Fold-training:

```text
training-window start
≤ Timestamp
< 2018-04-01
```

---

## Fold 2

Validation:

```text
2018-07-01
≤ Timestamp
< 2018-10-01
```

Fold-training:

```text
training-window start
≤ Timestamp
< 2018-07-01
```

---

## Fold 3

Validation:

```text
2018-10-01
≤ Timestamp
< 2019-01-01
```

Fold-training:

```text
training-window start
≤ Timestamp
< 2018-10-01
```

Trong đó:

W_LONG:

```text
training-window start = 2015-01-01
```

W_SHORT:

```text
training-window start = 2018-01-01
```

Đây là expanding-window design.

---

# 18. Vì sao không dùng Q1/2018 làm validation fold?

W_SHORT bắt đầu tại:

`2018-01-01`.

Nếu Q1/2018 được dùng làm validation ngay, W_SHORT không còn modeling rows trước validation block để fit classifier.

Do đó Q1/2018 đóng vai trò initial training block cho W_SHORT.

Temporal CV bắt đầu validation từ Q2/2018.

---

# 19. Positive-support gate cho temporal fold

M2 đã xác nhận năm 2018 có fraud positive xuyên suốt các tháng.

Tuy nhiên implementation sau này vẫn phải assert cho từng fold:

```text
train rows > 0
validation rows > 0

train fraud > 0
validation fraud > 0
```

Nếu fold nào không đủ positive support:

không được sửa bằng cách shuffle dữ liệu.

Phải:

- mở rộng validation block;
- điều chỉnh temporal boundary;
- hoặc giảm số fold;

và ghi Decision Log.

---

# 20. Vai trò của temporal CV

Temporal CV **không thay thế** external VALIDATION đã khóa.

Project có ba tầng:

```text
TẦNG 1
Temporal CV bên trong TRAIN
→ tuning / stability / robustness

TẦNG 2
External VALIDATION 2019-01 → 2019-05
→ development decision / comparison / threshold

TẦNG 3
FINAL TEST 2019-06 → 2019-10
→ final protected evaluation
```

Ba tầng không được trộn vai trò.

---

# 21. Khi nào temporal CV bắt buộc?

Temporal CV phải được dùng khi:

- hyperparameter tuning có nhiều candidate;
- cần đánh giá stability của một lựa chọn;
- M3.6 training-window comparison chưa đủ rõ và cần robustness;
- model/config selection có nguy cơ phụ thuộc quá mạnh vào một holdout.

Temporal CV không bắt buộc phải chạy cho mọi sanity baseline nhỏ.

Một baseline đơn giản có thể:

```text
fit TRAIN
→ evaluate external VALIDATION
```

trước khi quyết định có cần tuning hay không.

---

# 22. Metric trong temporal CV

Primary fold metric:

`F1_fraud`.

Secondary fold metrics:

- Recall_fraud;
- Precision_fraud;
- TP;
- FP;
- FN;
- TN;
- predicted-positive rate.

Không dùng Accuracy làm scoring chính.

---

# 23. Không chỉ báo mean CV

Khi sử dụng 3 temporal folds phải lưu:

```text
F1 từng fold
Recall từng fold
Precision từng fold
TP/FP/FN/TN từng fold

mean F1
std F1

mean Recall
mean Precision
```

Mean score không được che mất temporal instability.

Ví dụ:

```text
fold 1 tốt
fold 2 tốt
fold 3 collapse
```

không nên được tóm tắt đơn giản bằng:

`mean CV khá tốt`.

Temporal variation chính là evidence cần đọc.

---

# 24. Có thể dùng pooled out-of-fold diagnostic

Ngoài fold-wise metrics, implementation có thể ghép prediction của ba validation fold thành:

`out-of-fold temporal predictions`.

Từ đó tính:

- pooled Confusion Matrix;
- pooled F1;
- pooled Recall;
- pooled Precision.

Pooled metric là diagnostic bổ sung.

Không thay thế việc xem từng fold.

---

# 25. Pipeline phải nằm bên trong fold

Nếu preprocessing có bước fit, mỗi fold phải:

```text
fit preprocessing
trên fold-training only
```

sau đó:

```text
transform fold-validation
bằng learned state của fold-training
```

Không được:

```text
fit preprocessing trên toàn TRAIN
→ rồi mới CV
```

nếu preprocessing học statistic từ dữ liệu.

---

# 26. Categorical vocabulary cũng là learned state

Nếu encoder học category vocabulary:

fold-validation category không được tham gia việc học vocabulary của fold-training.

Pipeline phải xử lý:

`unknown/unseen category`

theo rule đã định nghĩa.

Không được fit encoder trên toàn 2018 rồi quay lại CV theo quý.

---

# 27. Behavioral history trong temporal CV

Causal historical state tiếp tục tuân theo M3.5.

Khi dự đoán transaction ở fold-validation tại thời điểm T:

được phép sử dụng prior observable transaction có:

`Timestamp < T`.

Có thể bao gồm transaction đã xảy ra trước đó trong validation block nếu pipeline mô phỏng sequential screening.

Không được sử dụng:

- fraud label của các transaction trước đó;
- future transaction;
- transaction cùng timestamp.

Learned preprocessing/model state vẫn frozen trong fold-validation.

---

# 28. Hyperparameter tuning policy

Tuning chỉ được thực hiện khi:

- baseline pipeline chạy đúng;
- leakage checks PASS;
- feature/preprocessing representation đủ ổn định;
- metric implementation đã khóa;
- search question rõ.

Không được tuning chỉ vì:

`score chưa đẹp`.

---

# 29. Search space phải được định nghĩa trước

Trước khi chạy tuning phải ghi:

```text
model ID
hyperparameter được tuning
candidate range/distribution
lý do chọn range
temporal CV specification
primary scoring
computational budget
```

Không được:

```text
xem kết quả
→ mở rộng search space theo hướng score đẹp
→ không ghi lại các lần thử trước
```

Mọi adjustment phải vào Experiment/Decision Log.

---

# 30. Grid search và randomized search

Nếu search space nhỏ và có chủ đích:

`Grid Search` được phép.

Nếu search space lớn:

`Randomized Search` có thể được dùng để giới hạn ngân sách.

Không có công cụ nào tự làm experiment hợp lệ.

Điều quyết định tính hợp lệ là:

- temporal CV đúng;
- preprocessing đúng fold;
- resampling đúng fold;
- test isolation;
- scoring đúng.

---

# 31. Không exhaustive tuning ở giai đoạn sớm

Không được mặc định chạy hàng trăm/hàng nghìn configuration khi:

- baseline chưa ổn;
- feature vẫn thay đổi;
- preprocessing vẫn chưa khóa;
- leakage chưa kiểm tra;
- training-window winner còn chưa đủ evidence.

Nguyên tắc:

`small justified search space first`.

---

# 32. Tuning computational budget

Không khóa ngân sách bằng thời gian phút/giờ vì phụ thuộc máy.

Mỗi tuning experiment phải khóa trước:

```text
candidate configuration count
×
temporal fold count
```

Ví dụ:

5 cấu hình × 3 fold

→ tối thiểu 15 lượt fit để so candidate.

Nếu chi phí quá lớn:

- thu hẹp search space;
- dùng randomized search;
- hoặc dùng baseline đơn giản hơn.

Không giảm integrity của split để tiết kiệm thời gian.

---

# 33. Refit sau temporal CV

Sau khi một configuration được chọn bằng inner temporal CV:

được phép refit configuration đó trên toàn classifier TRAIN tương ứng.

Ví dụ nếu selected window là W_SHORT:

```text
refit:
2018-01-01
≤ Timestamp
< 2019-01-01
```

Sau đó mới evaluate external VALIDATION:

```text
2019-01 → 2019-05
```

External VALIDATION không tham gia fit.

---

# 34. External VALIDATION vẫn là development partition

Kết quả temporal CV giúp:

- tuning;
- stability assessment;
- robustness.

Nhưng project vẫn sử dụng external VALIDATION đã khóa để đánh giá candidate pipeline trong development.

Nếu external validation cho result không như inner CV:

đây là một finding quan trọng về temporal generalization.

Không được âm thầm bỏ external validation result.

---

# 35. Threshold selection nằm sau upstream model selection

Threshold là decision layer riêng.

Recommended sequence:

```text
feature/preprocessing candidate
        ↓
training-window candidate
        ↓
model/config
        ↓
imbalance strategy
        ↓
temporal-CV/tuning nếu cần
        ↓
freeze upstream pipeline
        ↓
evaluate probability on external VALIDATION
        ↓
select threshold
        ↓
freeze threshold
        ↓
FINAL TEST
```

Không tối ưu threshold đồng thời với mọi upstream configuration nếu mục tiêu experiment là cô lập tác động của từng yếu tố.

---

# 36. Threshold không được chọn bằng inner final-test-like fold rồi gọi là final

Temporal CV fold chỉ là development evidence.

External VALIDATION tiếp tục là partition chính thức được phép hỗ trợ numerical threshold selection.

FINAL TEST tuyệt đối không được dùng.

---

# 37. Probability calibration chưa thuộc core M3.7

M3.7 không khóa:

- Platt scaling;
- isotonic calibration;
- calibration curve strategy;
- calibrated probability model.

Các nội dung này có thể được nghiên cứu nếu sau này ứng dụng yêu cầu probability có ý nghĩa calibration rõ hơn.

Hiện tại:

`Probability calibration: DEFERRED / OPTIONAL`

---

# 38. Robustness policy cho M3.6

Nếu W_LONG và W_SHORT có external-validation result gần hoặc trade-off khó quyết định, M3.7 cho phép dùng temporal CV template này làm robustness evidence.

So sánh phải xem:

```text
W_LONG fold-wise results
vs
W_SHORT fold-wise results
```

với cùng:

- feature;
- preprocessing;
- model;
- imbalance strategy;
- threshold policy;
- folds.

Không được dùng different folds cho hai candidate.

---

# 39. Stochastic robustness

Nếu model hoặc sampler có randomness và kết luận thay đổi đáng kể theo seed:

không được chọn seed có result đẹp nhất.

Khi cần robustness:

- định nghĩa trước một tập seed nhỏ;
- chạy cùng seed set cho mọi candidate;
- báo distribution/variation;
- không chọn riêng seed thuận lợi.

Default random state cho primary experiment:

`RANDOM_STATE = 42`

nếu operation có randomness.

Multi-seed chỉ dùng khi có lý do robustness rõ ràng.

---

# 40. Random seed không phải hyperparameter để săn điểm

Không được:

```text
seed 1
seed 2
seed 3
...
→ lấy seed cho validation F1 cao nhất
```

Random state dùng cho reproducibility.

Nếu performance phụ thuộc mạnh seed thì đó là:

`stability finding`

không phải cơ hội tuning seed.

---

# 41. GroupKFold không phải primary CV

Main project objective là future temporal screening và M3.3 đã xác nhận evaluation chủ yếu gồm existing User/Card.

Do đó M3.7 không chọn:

`GroupKFold theo User/Card`

làm primary model-selection CV.

Group-based/cold-start evaluation có thể là supplementary study nếu sau này project đặt câu hỏi riêng về unseen entity generalization.

## Quyết định

`Group-based CV: OPTIONAL / DEFERRED`

---

# 42. Integrity checks cho mỗi CV/tuning experiment

Trước khi đọc score phải xác nhận:

```text
fold temporal order:
PASS

fold overlap:
NONE

fold train fraud:
> 0

fold validation fraud:
> 0

preprocessing fit:
fold-TRAIN only

sampler:
fold-TRAIN only

validation resampling:
NO

final-test access:
NO

metric:
canonical implementation

positive class:
fraud

same-timestamp causal rule:
PASS

future-label use:
NO
```

Nếu bất kỳ điều kiện nào fail:

`STOP — không diễn giải metric`.

---

# 43. Experiment Log bắt buộc

Mỗi experiment phải lưu tối thiểu:

```text
Experiment ID
Date/runtime environment nếu cần

Training-window ID

Feature version
Preprocessing version
Model ID
Hyperparameters

Imbalance strategy
Sampling ratio / class weight
Random state

CV strategy
Fold boundaries

Threshold policy

Metric implementation

Fold-wise metrics
Mean/std diagnostics

External validation metrics nếu đã chạy

Fit time
Warnings

Decision
Next action
```

---

# 44. Quy tắc thay một yếu tố

Khi mục tiêu là đánh giá imbalance strategy:

```text
chỉ thay imbalance strategy
```

Khi mục tiêu là hyperparameter:

```text
chỉ thay hyperparameter candidate
```

Khi mục tiêu là training window:

```text
chỉ thay training window
```

Nếu experiment intentionally nghiên cứu interaction giữa nhiều yếu tố thì phải nói rõ đó là:

`interaction experiment`

không được diễn giải như single-factor comparison.

---

# 45. Không dùng FINAL TEST trong CV

FINAL TEST:

```text
2019-06 → 2019-10
```

không xuất hiện trong:

- fold generation;
- tuning;
- class-weight selection;
- sampling-ratio selection;
- training-window robustness;
- model selection;
- threshold selection.

## Quyết định

`FINAL TEST ACCESS DURING DEVELOPMENT: PROHIBITED`

---

# 46. Những gì M3.7 khóa được ngay

Có đủ căn cứ để khóa:

```text
resampling boundary
temporal-CV principle
3-fold expanding template
preprocessing-inside-fold rule
sampler-inside-fold rule
tuning/search policy
random-state policy
test-isolation policy
imbalance experiment order
```

Những nội dung này là protocol decision.

Không cần model output để khóa.

---

# 47. Những gì chưa được phép kết luận

M3.7 chưa chứng minh:

- class_weight cải thiện model;
- oversampling cải thiện model;
- undersampling cải thiện model;
- SMOTE cải thiện model;
- kỹ thuật nào tốt nhất;
- sampling ratio nào tốt nhất;
- hyperparameter nào tốt nhất;
- temporal-CV F1 thực tế bao nhiêu;
- W_LONG hay W_SHORT ổn định hơn;
- numerical threshold cuối là bao nhiêu.

Các câu hỏi trên:

`REQUIRE REAL MODEL EXPERIMENT`.

---

# 48. Decision Log M3.7

`M3.7-D01`  
No-intervention baseline phải tồn tại trước imbalance intervention.  
`Status: LOCKED`

`M3.7-D02`  
Class_weight, random oversampling và random undersampling là authorized experiment candidates.  
`Status: LOCKED`

`M3.7-D03`  
SMOTE không phải default; chỉ được dùng khi feature representation/pipeline phù hợp và leakage-safe.  
`Status: CONDITIONAL / DEFERRED`

`M3.7-D04`  
Resampling chỉ xảy ra sau split và chỉ trên training portion.  
`Status: LOCKED`

`M3.7-D05`  
Trong CV, sampler chỉ được áp dụng bên trong fold-training.  
`Status: LOCKED`

`M3.7-D06`  
VALIDATION và FINAL TEST phải giữ natural class distribution.  
`Status: LOCKED`

`M3.7-D07`  
Không sử dụng shuffled KFold/StratifiedKFold làm primary model-selection CV.  
`Status: LOCKED`

`M3.7-D08`  
Primary CV strategy là forward/expanding temporal validation.  
`Status: LOCKED`

`M3.7-D09`  
Temporal-CV template gồm ba validation block: Q2, Q3 và Q4/2018.  
`Status: LOCKED`

`M3.7-D10`  
Preprocessing có learned state phải fit riêng trên fold-training.  
`Status: LOCKED`

`M3.7-D11`  
Temporal CV dùng cho tuning/stability/robustness; không thay thế external VALIDATION 2019-01→05.  
`Status: LOCKED`

`M3.7-D12`  
Primary CV scoring = F1_fraud; phải đọc thêm Recall, Precision và Confusion Matrix theo fold.  
`Status: LOCKED`

`M3.7-D13`  
Tuning phải dùng search space nhỏ, có lý do và được khai báo trước.  
`Status: LOCKED`

`M3.7-D14`  
Random state dùng để tái lập, không được tuning để săn validation score.  
`Status: LOCKED`

`M3.7-D15`  
Threshold selection là bước riêng sau upstream model selection và sử dụng external VALIDATION, không FINAL TEST.  
`Status: LOCKED`

`M3.7-D16`  
Final imbalance-strategy winner chưa được chọn.  
`Status: REQUIRES EXPERIMENT`

`M3.7-D17`  
Final hyperparameter configuration chưa được chọn.  
`Status: REQUIRES EXPERIMENT`

`M3.7-D18`  
Probability calibration chưa thuộc core experiment.  
`Status: DEFERRED / OPTIONAL`

`M3.7-D19`  
Group/cold-start CV không phải primary CV; chỉ là supplementary option nếu sau này có câu hỏi riêng.  
`Status: DEFERRED / OPTIONAL`

---

# 49. M3.7 Gate

### GATE-01 — Imbalance có bị xử lý tự động chỉ vì class hiếm?

`PASS`

Không.

Baseline trước, intervention sau.

### GATE-02 — Resampling boundary rõ?

`PASS`

Training portion only.

### GATE-03 — Validation/test có bị resample?

`PASS`

Không.

### GATE-04 — CV có giữ temporal order?

`PASS`

Forward/expanding temporal CV.

### GATE-05 — Random StratifiedKFold có bị dùng máy móc?

`PASS`

Không được dùng làm primary model-selection CV.

### GATE-06 — Fold template đã rõ?

`PASS`

Ba validation block Q2/Q3/Q4-2018.

### GATE-07 — Preprocessing/resampling trong CV có leakage-safe?

`PASS`

Fit/apply chỉ từ fold-training.

### GATE-08 — External VALIDATION có giữ đúng vai trò?

`PASS`

Temporal CV không thay thế VALIDATION 2019-01→05.

### GATE-09 — FINAL TEST có được bảo vệ?

`PASS`

Không tham gia CV/tuning/imbalance/threshold selection.

### GATE-10 — Tuning policy có kiểm soát?

`PASS`

Search space nhỏ, khai báo trước, có computational budget.

### GATE-11 — Threshold có bị trộn với imbalance strategy?

`PASS`

Được xử lý như decision layer riêng.

### GATE-12 — Có quyết định winner nào bị khóa khi chưa có evidence?

`PASS`

Không.

---

# 50. Kết luận M3.7

```text
No-intervention baseline:
REQUIRED — LOCKED

Class weight:
AUTHORIZED CANDIDATE

Random oversampling:
AUTHORIZED CANDIDATE

Random undersampling:
AUTHORIZED CANDIDATE

SMOTE:
CONDITIONAL / DEFERRED

Resampling scope:
TRAINING PORTION ONLY — LOCKED

Validation/test resampling:
PROHIBITED — LOCKED

Primary CV:
FORWARD / EXPANDING TEMPORAL CV — LOCKED

CV validation blocks:
Q2 / Q3 / Q4 2018 — LOCKED

Random shuffled StratifiedKFold:
NOT PRIMARY — LOCKED

Preprocessing inside fold:
REQUIRED — LOCKED

Sampler inside fold:
REQUIRED — LOCKED

CV primary metric:
F1_fraud — LOCKED

Fold diagnostics:
Recall + Precision + CM — LOCKED

External VALIDATION:
2019-01 → 2019-05
REMAINS DEVELOPMENT HOLDOUT — LOCKED

FINAL TEST:
PROTECTED / NO TUNING ACCESS — LOCKED

Random state:
REPRODUCIBILITY TOOL, NOT TUNING TARGET — LOCKED

Final imbalance strategy:
OPEN / REQUIRES EXPERIMENT

Final hyperparameters:
OPEN / REQUIRES EXPERIMENT

Final threshold:
OPEN / REQUIRES VALIDATION EXPERIMENT

Probability calibration:
OPTIONAL / DEFERRED

Blocking issue:
NONE

M3.7 Gate:
PASS

M3.7 Status:
PASS — IMBALANCE / TEMPORAL-CV / TUNING POLICY LOCKED
```

## Handoff

Khi modeling bắt đầu, hierarchy phải là:

```text
TRAIN
   ↓
temporal CV bên trong TRAIN
   ↓
tuning / robustness
   ↓
refit trên full selected TRAIN
   ↓
external VALIDATION 2019-01→05
   ↓
development decision
   ↓
threshold selection
   ↓
freeze pipeline
   ↓
FINAL TEST 2019-06→10
```

Không được đảo thứ tự này.

## Bước tiếp theo

`Next: M3.8 — Tổng hợp Experiment Specification v1.0, Decision Log, Open Questions và M3 Gate`