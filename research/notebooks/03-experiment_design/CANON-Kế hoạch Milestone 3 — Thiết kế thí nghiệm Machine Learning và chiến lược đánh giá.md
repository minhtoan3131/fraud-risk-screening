
# Kế hoạch Milestone 3 — Thiết kế thí nghiệm Machine Learning và chiến lược đánh giá

M3 không có nhiệm vụ “tìm model tốt nhất”. Nó phải biến toàn bộ bằng chứng từ M1–M2 thành một **experiment protocol** đủ rõ để từ M4 trở đi mọi preprocessing, baseline và model đều được thực hiện trên cùng một nền tảng đánh giá hợp lệ.

Điều này khớp với quy trình tổng thể đã khóa:

```text
M2 — EDA
    ↓
M3 — Thiết kế thí nghiệm ML
    ↓
M4 — Preprocessing / feature engineering
    ↓
M5 — Baseline
    ↓
M6 — Model ứng viên
    ↓
...
Final test
```



Sản phẩm quan trọng nhất của M3 theo tôi không phải một con số metric, mà phải là một bộ quy tắc trả lời được:

```text
Model được phép học dữ liệu nào?

Validation nằm ở đâu?

Final test nằm ở đâu?

Tại sao split đó phản ánh prediction trong tương lai?

Metric nào được dùng để lựa chọn?

Metric nào chỉ dùng để bổ trợ diễn giải?

Test set được phép nhìn bao nhiêu lần?

Các training window sẽ được so sánh như thế nào?

Historical warm-up khác modeling rows ở đâu?

Class imbalance được phép xử lý ở phần nào?

Những gì tuyệt đối không được dùng vì leakage?

Khi nào một thí nghiệm được coi là công bằng?
```

Nếu M3 chưa trả lời được các câu đó thì chưa nên đi sang preprocessing/modeling.

---

# 2. Những bằng chứng M3 bắt buộc phải kế thừa từ M2

Tôi sẽ coi các điểm sau là **đầu vào đã được xác nhận**, không làm EDA lại từ đầu.

Thứ nhất, target cực kỳ mất cân bằng: chỉ có `29,757 fraud / 24,386,900 transaction`, khoảng `0.122%`, tức xấp xỉ `818.5 non-fraud / 1 fraud`. Majority baseline luôn đoán non-fraud vẫn đạt gần `99.88% Accuracy`. Vì vậy Accuracy không thể là tiêu chí chính để lựa chọn model. 

Thứ hai, temporal behavior là vấn đề trung tâm. Fraud prevalence và nhiều quan hệ feature–target thay đổi theo thời gian; đặc biệt từ `11/2019 → 02/2020` có `625,025 transaction` nhưng không có fraud. Vì vậy random split thông thường có thể tạo đánh giá quá lạc quan; M3 phải dùng temporal evaluation. 

Thứ ba, năm 2020 không được dùng làm final fraud-performance test vì chỉ có tháng 1–2 và `0 fraud`. 

Thứ tư, association mạnh trong full history không đồng nghĩa sẽ generalize sang tương lai. M2.6 đã thấy một số quan hệ của `Use Chip`, Amount, MCC, location… thay đổi đáng kể theo temporal regime; high-cardinality location/MCC còn có nguy cơ synthetic shortcut và memorization. 

Thứ năm, User/Card history đủ sâu để tạo behavioral feature. Median Card có khoảng `2,602 transaction`, active span gần `8.7 năm`; strict cold-start rất nhỏ. 

Thứ sáu, bốn behavioral feature đại diện đã được chứng minh là có thể tính strict-causal; nhưng M2 mới chứng minh **computability**, chưa chứng minh chúng giúp model tốt hơn. 

Thứ bảy, historical feature phải giữ nguyên quy tắc:

```text
timestamp(history) < timestamp(current transaction)
```

Transaction cùng Timestamp không được làm history cho nhau. 

Thứ tám, dữ liệu cũ trước modeling window vẫn có thể dùng để warm-up lịch sử mà không nhất thiết trở thành training rows. Đây là một distinction rất quan trọng cho thiết kế M3. 

Cuối cùng, M2 đã chính thức để lại cho M3 ba quyết định mở lớn: final train/validation/test boundaries, final training window và primary/secondary metrics. 

---

# 3. Một điểm tôi muốn điều chỉnh so với cách hiểu quá máy móc về M3

Trong M2 có câu rằng các training window như `2015–2019`, `2018–2019`, pre-break window cần được “so sánh bằng experiment”. 

Nhưng quy trình tổng thể lại đặt:

```text
M3 design
→ M4 preprocessing
→ M5 baseline
```

Nếu ở M3 ta cố buộc phải chọn “window thắng” bằng performance model ngay lập tức thì sẽ xảy ra nghịch lý: muốn chạy model phải quyết định preprocessing/feature representation, nhưng đó lại là việc của M4.

Vì vậy tôi đề xuất giải quyết như sau:

**M3 sẽ thiết kế và khóa protocol cho training-window comparison, nhưng chưa nhất thiết khóa window chiến thắng.**

Tức là M3 có thể kết luận:

```text
Đây là 2–3 training-window candidates hợp lệ.

Đây là validation/test period chung.

Đây là cách giữ mọi điều kiện khác giống nhau.

Đây là metric dùng để so sánh.

Đây là feature/preprocessing version phải giữ cố định khi so sánh.

Đây là rule để chọn window.

Actual winner:
→ chỉ khóa khi baseline/preprocessing tối thiểu đã tồn tại.
```

Nếu trong quá trình thực hiện M3 chúng ta tìm ra một cách so sánh hợp lệ mà không phá ranh giới M4/M5 thì có thể thực hiện ngay. Nếu không, phần **protocol** được khóa ở M3, còn **execution** chuyển đúng sang milestone modeling tương ứng.

Tôi cho rằng đây là cách trung thành với logic dự án hơn là ép milestone phải hoàn thành một việc chỉ vì kế hoạch ban đầu ghi như vậy.

---

# 4. Kế hoạch M3 đề xuất

Tôi đề xuất chia M3 thành **8 công việc chính: M3.1 → M3.8**.

Không coi số `8` là bắt buộc. Sau mỗi bước, chúng ta có quyền gộp/tách hoặc thêm bước nếu output thực tế yêu cầu.

---

## M3.1 — Khóa mục tiêu, guardrail và protocol của Milestone 3

Đây sẽ là bước đầu tiên, tương tự vai trò M2.1.

Câu hỏi trung tâm:

> “M3 được phép quyết định gì, không được quyết định gì, và một experiment hợp lệ của project phải tuân thủ các ranh giới nào?”

M3.1 sẽ kế thừa các guardrail:

```text
Prediction point
→ thời điểm transaction cần được screening.

Target
→ Is Fraud?, Yes = positive class.

User / Card / Merchant Name
→ không dùng raw làm classifier feature.

Errors?
→ không dùng trong Model V1.

Temporal evaluation
→ evaluation chính phải past → future.

2020
→ không dùng làm final fraud test.

Historical features
→ strict causal.

Same Timestamp
→ không làm history cho nhau.

Test set
→ không dùng lặp đi lặp lại để lựa chọn model/config.

Synthetic dataset
→ không overclaim production banking performance.
```

Prediction point này đã được khóa từ M1.8: feature phải tồn tại tại prediction point hoặc được tính hoàn toàn từ quá khứ. 

Đầu ra M3.1 nên là:

```text
M3 Experiment Charter
+
M3 Guardrails
+
Decision / Open-question log
```

Gate của M3.1:

> Nếu chưa nói rõ dữ liệu nào model được phép biết ở từng thời điểm thì chưa sang M3.2.

---

# 5. M3.2 — Xây temporal map phục vụ thiết kế split

Bước này chưa train model.

Nhiệm vụ là chuyển các finding temporal từ M2 thành bản đồ thiết kế experiment.

Ta cần đặt các giai đoạn thời gian cạnh nhau, ví dụ:

```text
... 2015 ... 2016 ... 2017 ... 2018 ... 2019-01 ... 2019-10 | 2019-11 ... 2020-02
                                                          ↑
                                                zero-fraud regime bắt đầu
```

Sau đó xem xét những vùng nào có thể đóng vai trò:

```text
history warm-up
training
validation
final test
special stress-test / abnormal regime
```

Đặc biệt phải phân biệt:

```text
modeling window
≠
historical context window
```

Ví dụ:

```text
1991 ───────── 2017 | 2018 ─────── 2019
       history      |   modeling rows
       warm-up      |
```

không có nghĩa model được fit bằng toàn bộ 1991–2017.

M2 đã xác nhận older rows có thể dùng làm causal warm-up và recent subset vẫn giữ hàng triệu transaction/hàng nghìn fraud. 

Đầu ra của M3.2:

```text
Temporal regime map
+
candidate evaluation periods
+
candidate modeling windows
+
history warm-up policy v0.1
```

Chưa khóa split cuối nếu bằng chứng chưa đủ.

---

# 6. M3.3 — Thiết kế candidate train / validation / test split

Đây sẽ là một trong những bước quan trọng nhất M3.

Thay vì chọn ngay:

```text
train = ...
validation = ...
test = ...
```

ta nên lập một số candidate có lý do.

Ví dụ khái niệm:

```text
Candidate A

past
───────────────
TRAIN

later past
───────────────
VALIDATION

future
───────────────
FINAL TEST
```

Mọi candidate phải thỏa:

```text
max(train time) < validation period

validation nằm trước final test

không có tương lai quay ngược vào train

test có đủ fraud positive để tính metric

2020 không là final fraud test

không dùng test để lựa chọn split sau khi đã xem score
```

Ở bước này, chúng ta phải tính chính xác cho từng candidate:

```text
transaction count
fraud count
fraud rate
time span
User coverage
Card coverage
new-user / new-card share
```

Đây không phải EDA lại từ đầu; đây là **split audit** phục vụ experiment design.

Gate M3.3:

> Mỗi partition phải đủ dữ liệu, có ý nghĩa temporal và có khả năng đánh giá positive class.

Chỉ sau output thực tế mới khóa final boundaries.

---

# 7. M3.4 — Thiết kế evaluation metric strategy

Bước này trả lời:

> “Chúng ta dùng tiêu chí nào để đọc và lựa chọn model?”

Không nên chỉ liệt kê:

```text
Accuracy
Precision
Recall
F1
```

mà phải định nghĩa **vai trò từng metric**.

Ví dụ M3 cần trả lời:

```text
Primary metric:
metric nào chi phối lựa chọn?

Secondary metrics:
metric nào dùng để kiểm tra trade-off?

Diagnostic:
Confusion Matrix dùng để đọc loại lỗi gì?

Probability/ranking metric:
có cần ở giai đoạn này không?

Accuracy:
được giữ để tham khảo nhưng không được dùng làm bằng chứng chính?
```

Tài liệu Tuần 8 nhấn mạnh model cần được đọc từ nhiều góc nhìn, và metric phải được diễn giải thông qua TP/TN/FP/FN chứ không chỉ nhìn một số duy nhất. 

M3.4 cũng phải ghi rõ ý nghĩa nghiệp vụ của lỗi:

```text
False Negative:
fraud trong ground truth
nhưng model bỏ sót.

False Positive:
non-fraud
nhưng model đưa vào nhóm đáng ngờ.
```

Chúng ta **chưa chọn threshold cuối ở M3**.

Threshold phải được chọn sau bằng validation objective; tài liệu dự án đã khóa rằng không được nhìn test hết lần này tới lần khác để chỉnh threshold. 

---

# 8. M3.5 — Khóa vai trò của train, validation và final test

M3.3 quyết định *ranh giới thời gian*.

M3.5 quyết định *quyền hạn* của từng partition.

Tôi muốn bước này rất rõ vì nó sẽ bảo vệ toàn bộ project sau này.

Ví dụ:

```text
TRAIN

Được phép:
- fit preprocessing;
- học category vocabulary;
- fit scaler/imputer;
- fit model;
- resampling nếu experiment yêu cầu.

Không được dùng:
- future validation/test information.
```

```text
VALIDATION

Được phép:
- so sánh feature/preprocessing candidate;
- so sánh model/config;
- so sánh training window;
- hỗ trợ lựa chọn threshold sau này.

Không được:
- fit các statistic vốn phải học từ train.
```

```text
FINAL TEST

Chỉ dùng sau khi lựa chọn chính gần như hoàn tất.

Không dùng để:
- chọn feature;
- chọn training window;
- chọn hyperparameter;
- chọn threshold;
- quyết định model rồi quay lại sửa tiếp.
```

Điều này phù hợp với nguyên tắc tài liệu học: test set là “đề thi cuối”, mọi lựa chọn lặp lại phải diễn ra bên trong training/validation. 

---

# 9. M3.6 — Thiết kế protocol so sánh training window

Đây là phần nhận trực tiếp từ M2.8.

Các candidate ban đầu có bằng chứng feasibility:

```text
2015–2019
8,579,208 transaction
11,693 fraud

2018–2019
3,445,553 transaction
4,578 fraud

2018 → 2019-10
3,157,028 transaction
4,578 fraud
```



Nhưng khi bước vào M3, có thể ranh giới sẽ cần điều chỉnh để validation/test dùng chung.

Nguyên tắc của experiment sẽ là:

```text
Chỉ thay:
training window

Giữ cố định:
validation period
test period
feature version
preprocessing version
model/baseline configuration
metric definitions
threshold policy
```

Nếu thay cả training window, model, encoding và threshold cùng lúc thì ta không biết performance thay đổi do nguyên nhân nào.

M3.6 cần định nghĩa trước:

```text
candidate windows
comparison condition
primary evaluation target
tie-break rule
computational budget
cách ghi lại kết quả
```

Nhưng như tôi nói ở trên, **M3 chưa bắt buộc phải tuyên bố winner** nếu baseline/preprocessing cần thiết chưa tồn tại.

---

# 10. M3.7 — Thiết kế policy cho class imbalance, CV và các experiment sau

Bước này không có nghĩa chúng ta dùng SMOTE ngay.

Nó phải trả lời:

> “Nếu sau này cần class weighting/resampling/CV thì nó được phép xảy ra ở đâu?”

Rule phải khóa ngay:

```text
Split trước.

Sau đó mới:
class_weight / oversampling / undersampling / SMOTE...

Chỉ trên training data.

Không bao giờ resample validation/test.
```

M2 đã chuyển nguyên tắc này sang M3/M4. 

Tài liệu Tuần 10 cũng nhấn mạnh không resample test và không chọn threshold bằng việc xem test nhiều lần. 

Một điểm quan trọng khác: **không máy móc dùng StratifiedKFold chỉ vì classification bị imbalance**.

Tài liệu Tuần 9 dạy StratifiedKFold trong trường hợp classification thông thường. 

Nhưng dataset của chúng ta có temporal shift rõ ràng.

Vì vậy M3 phải quyết định CV theo **cấu trúc thực tế của project**, có khả năng là temporal/forward validation thay vì random stratified folds.

Đây chính là ví dụ rõ nhất cho nguyên tắc:

> kiến thức chung là điểm xuất phát, nhưng cấu trúc thời gian thực tế của dataset mới quyết định protocol cuối.

---

# 11. M3.8 — Tổng hợp Experiment Specification và M3 Gate

M3.8 không tạo experiment mới.

Nó sẽ tổng hợp:

```text
Prediction point
Temporal map
Final split boundaries
Roles của train / validation / test
History warm-up policy
Metric strategy
Training-window candidates
Training-window comparison protocol
Class-imbalance rules
CV/tuning rules
Leakage guardrails
Open questions chuyển sang M4/M5
```

Sản phẩm cuối nên là:

```text
Experiment Specification v1.0
+
Decision Log
+
Open Questions / Handoff
```

M3 Gate chỉ PASS nếu chúng ta trả lời được ít nhất:

```text
1. Final evaluation theo thời gian đã được định nghĩa chưa?

2. Train / validation / test có ranh giới cụ thể chưa?

3. Mỗi partition có đủ fraud để đánh giá chưa?

4. Test set có được bảo vệ khỏi model selection chưa?

5. Primary / secondary metric có lý do chưa?

6. History warm-up và modeling rows đã được tách khái niệm chưa?

7. Training-window comparison đã có protocol công bằng chưa?

8. Class imbalance techniques được giới hạn đúng training partition chưa?

9. CV/tuning sau này có tôn trọng thời gian chưa?

10. M3 có tránh quyết định premature về feature/model/threshold không?

11. Các quyết định chuyển sang M4/M5 đã ghi rõ chưa?

12. Có leakage nào chưa được giải quyết không?
```

Nếu một câu quan trọng còn “không biết”, M3 chưa nên PASS.

---

# 12. Những thứ M3 không nên làm

Tôi muốn khóa phần này ngay để tránh scope creep.

M3 không nên tự biến thành notebook modeling.

Chưa nên:

```text
chọn Logistic Regression là model cuối;

chọn Random Forest vì EDA thấy MCC mạnh;

chạy GridSearchCV hàng trăm cấu hình;

chọn threshold 0.5 hay 0.3;

SMOTE toàn dataset;

One-Hot encode full dataset rồi mới split;

fit StandardScaler trên toàn dataset;

quyết định final behavioral feature set;

dùng test 2019 rồi nhìn score nhiều lần để sửa model;

dùng 2020 làm fraud test;

dùng raw User / Card / Merchant Name làm feature.
```

Preprocessing có bước học thống kê phải fit trên training data; đây là nguyên tắc đã được tài liệu học khóa rõ. 

---

# 13. Cách chúng ta sẽ thực hiện M3 trên thực tế

Tôi đề nghị giữ đúng nhịp đã làm ở M2:

```text
Không viết toàn bộ M3 một lượt.

Mỗi lần chỉ làm một M3.x.

Mỗi M3.x:

Câu hỏi
→ xác định bằng chứng cần có
→ viết code tối thiểu cần thiết
→ chạy trên dữ liệu thật
→ đọc output
→ kiểm tra consistency
→ diễn giải
→ quyết định
→ cập nhật Decision Log
→ mới sang bước tiếp.
```

Điểm này rất quan trọng.

Ví dụ hiện tại tôi đề xuất M3.3 là thiết kế final split. Nhưng giả sử khi chạy M3.2 chúng ta phát hiện:

```text
candidate validation period
→ quá ít fraud

hoặc

candidate test
→ nằm sát zero-fraud regime quá mức

hoặc

một boundary tháng cụ thể
→ tạo distribution quá dị biệt
```

thì **không được nói “kế hoạch ghi thế rồi cứ làm tiếp”**.

Khi đó chúng ta có quyền:

```text
M3.2 mở rộng;

M3.3 đổi candidate;

gộp M3.2 + M3.3;

thêm một stress-test temporal segment;

hoặc thay đổi numbering nếu logic project yêu cầu.
```

Đó mới là cách tôi sẽ làm M3 cùng bạn.

Nguyên tắc ưu tiên xuyên suốt sẽ là:

```text
Problem statement đã khóa
        ↓
Guardrail từ M1
        ↓
Bằng chứng EDA M2
        ↓
Output thực tế của công việc M3 hiện tại
        ↓
Lập luận ML đúng
        ↓
Kế hoạch M3 ban đầu
```

Chứ **không phải kế hoạch ban đầu đứng trên dữ liệu**. Đây cũng chính là tinh thần của tài liệu dự án: cấu trúc phải phục vụ vấn đề thực tế, không được ép vấn đề theo một template. 

# 14. Thứ tự làm việc tôi đề nghị từ đây

Chúng ta sẽ đi đúng từng bước:

```text
M3.1
Khóa Experiment Charter + guardrails M3
        ↓
M3.2
Temporal map + modeling/history windows
        ↓
M3.3
Candidate train/validation/test split
→ chạy audit thực tế
→ khóa split nếu đủ bằng chứng
        ↓
M3.4
Metric strategy
        ↓
M3.5
Partition roles + test isolation policy
        ↓
M3.6
Training-window comparison protocol
        ↓
M3.7
Imbalance / CV / future experiment policy
        ↓
M3.8
M3 Summary + Decision Log + Gate
```

Tôi cho rằng đây là khung đủ chi tiết để bắt đầu nhưng vẫn đủ mềm để không rơi vào tình trạng “thiết kế M3 trên giấy quá đẹp nhưng không phù hợp dữ liệu thật”.

