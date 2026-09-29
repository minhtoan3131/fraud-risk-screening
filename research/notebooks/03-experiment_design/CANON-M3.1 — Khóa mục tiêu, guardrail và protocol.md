# M3.1 — Khóa mục tiêu, guardrail và protocol của Milestone 3

## 1. Vai trò của Milestone 3

Milestone 1 đã lựa chọn và audit dataset. Milestone 2 đã thực hiện EDA để hiểu target, chiều thời gian, chất lượng dữ liệu, feature, entity history và khả năng xây behavioral feature.

Milestone 2 kết thúc với trạng thái `PASS WITH FINDINGS`, không phát hiện blocking issue và chính thức chuyển các vấn đề về `temporal split`, `evaluation protocol` và `training-window comparison` sang M3.

Milestone 3 có vai trò:

> **Thiết kế một quy trình thí nghiệm Machine Learning có thể kiểm chứng, tái hiện và bảo vệ được, trong đó dữ liệu quá khứ được sử dụng để học và dữ liệu tương lai được sử dụng để đánh giá khả năng tổng quát hóa của mô hình.**

M3 không tồn tại để tìm model có score cao nhất càng nhanh càng tốt.

M3 tồn tại để trả lời trước các câu hỏi:

* dữ liệu nào được phép dùng để học;
* dữ liệu nào dùng để lựa chọn;
* dữ liệu nào phải được giữ lại để đánh giá cuối;
* cách chia dữ liệu phải tôn trọng thời gian như thế nào;
* các thí nghiệm phải được so sánh trong điều kiện nào;
* metric nào có vai trò gì;
* preprocessing, feature engineering và xử lý class imbalance được phép học thông tin từ partition nào;
* bằng cách nào bảo đảm test set không dần tham gia vào quá trình lựa chọn.

M3 vì vậy là giai đoạn **thiết kế luật của thí nghiệm**, trước khi đi sâu vào preprocessing và modeling.

---

## 2. Bài toán và prediction point tiếp tục được giữ nguyên

Problem statement của project tiếp tục được giữ nguyên:

> Sử dụng các thông tin có sẵn về một giao dịch tại thời điểm đánh giá để xây dựng mô hình học máy phân loại nhị phân, nhằm ước lượng khả năng giao dịch thuộc lớp có dấu hiệu gian lận theo nhãn của bộ dữ liệu; kết quả được sử dụng như một tín hiệu hỗ trợ sàng lọc ban đầu, không phải quyết định cuối cùng về gian lận.

Đơn vị dự đoán vẫn là:

`một transaction`

Target vẫn là:

`Is Fraud?`

Positive class:

`Yes`

Negative class:

`No`

Prediction point của project là:

> **thời điểm transaction cần được screening.**

Một feature chỉ hợp lệ nếu nó đã tồn tại tại prediction point hoặc được tính hoàn toàn từ thông tin lịch sử xảy ra trước transaction hiện tại. Target, future transaction, aggregate chứa future data và thông tin chỉ xuất hiện sau prediction point không được sử dụng.

`Quyết định M3.1-D01: GIỮ NGUYÊN prediction point và problem definition đã khóa từ M0/M1.`

---

## 3. Mục tiêu chính thức của Milestone 3

M3 phải tạo ra một `Experiment Specification` đủ rõ để các milestone phía sau không phải tự đoán cách đánh giá.

M3 cần giải quyết bốn nhóm vấn đề chính.

Thứ nhất là **temporal evaluation**: xác định cách tổ chức train, validation và final test theo chiều quá khứ → tương lai.

Thứ hai là **evaluation strategy**: xác định vai trò của từng metric và cách đánh giá phù hợp với target mất cân bằng mạnh.

Thứ ba là **experiment comparability**: xác định điều kiện để hai thí nghiệm được coi là so sánh công bằng, đặc biệt đối với training window, preprocessing, feature set, model configuration và sau này là class-imbalance strategy.

Thứ tư là **data-boundary protection**: xác định partition nào được phép học preprocessing statistics, feature transformation, resampling, model parameter, hyperparameter và threshold.

Đầu ra cuối của M3 phải đủ để M4 và các giai đoạn modeling triển khai mà không phải tự phát minh lại quy tắc đánh giá.

---

## 4. Ranh giới của Milestone 3

M3 được phép:

* phân tích các temporal partition candidate;
* tính transaction count, fraud count, fraud rate và các thống kê cần thiết để audit split;
* thiết kế train / validation / test boundaries;
* thiết kế metric strategy;
* thiết kế training-window comparison;
* định nghĩa quyền sử dụng của từng partition;
* định nghĩa protocol cho class imbalance, validation và tuning sau này;
* xác định các leakage guardrail;
* ghi Decision Log và Experiment Specification.

M3 **chưa có nhiệm vụ**:

* quyết định final feature set;
* quyết định final encoding của MCC/location;
* quyết định transformation cuối của Amount;
* triển khai production behavioral-feature pipeline;
* chọn model cuối;
* tuning hyperparameter đầy đủ;
* chọn classification threshold cuối;
* chạy resampling như một cách tối ưu model;
* tuyên bố một training window tốt nhất nếu chưa có experiment hợp lệ để chứng minh;
* thực hiện final test của model.

M2 đã cố ý để các quyết định về feature set, encoding, model family và threshold cho những giai đoạn thích hợp phía sau.

`Quyết định M3.1-D02: M3 thiết kế experiment; không biến M3 thành giai đoạn model selection hoặc preprocessing implementation.`

---

# 5. Guardrail bắt buộc của Milestone 3

## 5.1. Guardrail về raw identifier

Không đưa trực tiếp:

`User`

`Card`

`Merchant Name`

vào classifier feature.

Các trường này chỉ được dùng làm identifier, grouping key hoặc history key để xây feature hợp lệ.

Đây là guardrail đã được khóa từ Risk Audit và tiếp tục có hiệu lực sau khi dataset được lựa chọn.

`M3.1-G01: LOCKED`

---

## 5.2. Guardrail đối với `Errors?`

`Errors?` không được sử dụng trong Model V1 vì prediction-time availability chưa được chứng minh chắc chắn.

Nếu sau này xuất hiện bằng chứng mới đủ mạnh về availability tại prediction point thì đây phải là một quyết định mới có audit riêng; không được tự động đưa cột này trở lại.

`M3.1-G02: LOCKED`

---

## 5.3. Evaluation chính phải theo thời gian

Không sử dụng random split đơn giản làm thiết kế evaluation chính.

Chiều đánh giá phải tuân thủ:

`quá khứ → train / validation → tương lai → test`

Đây không chỉ là một guardrail lý thuyết về leakage. M2 đã xác nhận target prevalence và nhiều feature-target relationship thực sự thay đổi theo thời gian.

`M3.1-G03: LOCKED`

---

## 5.4. Final test không được dùng như validation

Final test phải đóng vai trò dữ liệu chưa tham gia vào quá trình lựa chọn.

Không được xem final-test score sau mỗi lần:

* đổi feature;
* đổi preprocessing;
* đổi training window;
* đổi model;
* đổi hyperparameter;
* đổi class-imbalance strategy;
* đổi threshold.

Nguyên tắc học đã xác định test set là “bài thi cuối”; các lựa chọn lặp lại phải diễn ra trong training/validation thay vì tiêu hao test set.

`M3.1-G04: LOCKED`

---

## 5.5. Historical feature phải strict causal

Đối với transaction `T`:

`historical_feature(T)`

chỉ được sử dụng transaction có:

`timestamp(history) < timestamp(T)`

Không được dùng:

* transaction hiện tại;
* transaction tương lai;
* lifetime aggregate chứa tương lai;
* full-dataset statistic gắn ngược trở lại transaction quá khứ.

Đây là quy tắc causal đã được prototype và kiểm chứng trong M2.8.

`M3.1-G05: LOCKED`

---

## 5.6. Transaction cùng Timestamp không được dùng làm history cho nhau

Timestamp của artifact chỉ có độ phân giải tới phút và tồn tại timestamp tie.

Do đó hai transaction có cùng Timestamp không được sử dụng làm historical context cho nhau.

`M3.1-G06: LOCKED`

---

## 5.7. Historical context và classifier-training rows là hai khái niệm khác nhau

Older transaction có thể được sử dụng để tạo causal history cho transaction trong một recent modeling window mà không nhất thiết trở thành row dùng để fit classifier.

Do đó M3 phải luôn phân biệt:

`rows dùng để xây historical context`

và:

`rows dùng để fit classifier`

M2.8 đã xác nhận history warm-up là thiết kế causal hợp lệ và có thể giảm card-level cold-start trong recent modeling window.

`M3.1-G07: LOCKED`

---

## 5.8. Preprocessing chỉ được học từ TRAIN

Mọi bước cần học thông tin từ dữ liệu, ví dụ:

* category vocabulary;
* mean / median / mode;
* min / max;
* mean / standard deviation;
* imputation statistics;
* encoder;
* scaler;
* các thống kê dữ liệu khác;

chỉ được fit bằng training data rồi mới áp dụng sang validation/test.

Tài liệu preprocessing của project xác định bất kỳ transformer nào học thông tin từ dữ liệu đều phải fit trên TRAIN.

`M3.1-G08: LOCKED`

---

## 5.9. Resampling chỉ được thực hiện sau split và chỉ trên training data

Nếu sau này sử dụng:

`oversampling`

`undersampling`

`SMOTE`

hoặc kỹ thuật tương đương, việc đó chỉ được thực hiện trên training partition.

Không resample validation.

Không resample final test.

Không resample toàn dataset rồi mới chia.

Guardrail này đã được khóa từ M1.6.

`M3.1-G09: LOCKED`

---

## 5.10. Không dùng Accuracy đơn độc làm cơ sở đánh giá

Dataset có class imbalance rất mạnh.

M2 xác nhận fraud chỉ khoảng `0.122%`; classifier luôn dự đoán non-fraud vẫn có Accuracy gần `99.88%`. Vì vậy Accuracy đơn độc không phản ánh khả năng phát hiện positive class.

Accuracy vẫn có thể được báo cáo như một metric tham khảo, nhưng:

> **không được sử dụng một mình để lựa chọn hoặc tuyên bố model tốt.**

Primary metric và secondary metric cụ thể chưa được khóa tại M3.1; việc này thuộc M3.4.

`M3.1-G10: LOCKED`

---

## 5.11. Không dùng năm 2020 làm final fraud-performance test

Artifact năm 2020 chỉ chứa tháng 1–2 và không có fraud.

Một final test không có positive class không thể đánh giá Recall, Precision và F1 của fraud class một cách có ý nghĩa.

Do đó:

`2020 ≠ final fraud test`

M3.1 chưa khóa 2019 hoặc giai đoạn cụ thể nào là final test; quyết định boundary được để lại cho bước split design.

`M3.1-G11: LOCKED`

---

## 5.12. Không tự động xử lý structural missingness hoặc negative Amount

Các guardrail dữ liệu từ M1/M2 tiếp tục có hiệu lực khi thiết kế experiment.

Không tự động drop row thiếu `Merchant State` / `Zip`, vì missing location có tính cấu trúc.

Không tự động `abs()`, drop hoặc clip negative Amount khi exact semantic chưa được khóa.

M3 không phải nơi quyết định chiến lược preprocessing cuối cho các vấn đề này.

`M3.1-G12: LOCKED`

---

## 5.13. Không biến full-history association thành rule

M2 đã phát hiện rằng nhiều feature-target relationship không ổn định theo thời gian.

Vì vậy fraud rate cao của một MCC, location, transaction mode hoặc interaction trên full history không được xem như bằng chứng đủ rằng feature đó sẽ generalize sang tương lai.

M3 phải ưu tiên temporal validation thay vì full-history association.

`M3.1-G13: LOCKED`

---

## 5.14. Evaluation hiện tại chủ yếu phản ánh existing User/Card

M2 xác nhận strict card-level cold-start rất nhỏ và phần lớn future transaction có historical context.

Do đó kết quả evaluation chính sau này chủ yếu phản ánh khả năng screening transaction của existing User/Card.

Khả năng generalize sang hoàn toàn new User/Card là một limitation riêng và nếu cần đánh giá phải được xem như evaluation segment riêng.

`M3.1-G14: LOCKED`

---

## 5.15. Không overclaim kết quả

Dataset là fully synthetic.

Kết quả experiment và model của project không được trình bày như bằng chứng về hiệu quả của một hệ thống fraud detection trong ngân hàng production.

Project chỉ đánh giá model trong phạm vi artifact synthetic đã audit. Guardrail này đã được khóa từ M1.6.

`M3.1-G15: LOCKED`

---

# 6. Experiment Protocol chung của M3

## 6.1. Nguyên tắc “question before experiment”

Không chạy thí nghiệm chỉ vì một API hoặc thuật toán tồn tại.

Mỗi experiment phải xác định trước:

`câu hỏi đang cần trả lời`

`biến nào được thay đổi`

`những điều kiện nào phải giữ cố định`

`partition nào được sử dụng`

`metric nào sẽ được quan sát`

`kết quả nào có thể hỗ trợ hoặc bác bỏ giả thuyết`

Cách làm này kế thừa nguyên tắc của M2:

`Câu hỏi → Kiểm tra → Bằng chứng → Nhận xét → Chuyển giao`

M3 chuyển nó thành:

`Câu hỏi → Thiết kế experiment → Kiểm tra leakage → Thực thi → Bằng chứng → So sánh → Quyết định`

---

## 6.2. Mỗi experiment phải thay đổi có kiểm soát

Khi muốn đánh giá tác động của một yếu tố, cần cố gắng giữ các yếu tố còn lại nhất quán.

Ví dụ, khi so sánh training window:

`thay đổi = training window`

còn những thành phần như:

`validation period`

`feature version`

`preprocessing version`

`model/configuration`

`metric`

`threshold policy`

phải được giữ giống nhau ở mức phù hợp.

Nếu đồng thời đổi training window, feature, preprocessing và model thì không thể xác định yếu tố nào gây ra chênh lệch kết quả.

`M3.1-P01: mọi comparison phải ghi rõ biến thay đổi và biến được kiểm soát.`

---

## 6.3. Temporal partition được ưu tiên hơn random partition

M3 không mặc định sử dụng các kỹ thuật validation phổ biến nếu chúng phá temporal ordering.

Ví dụ, `StratifiedKFold` có giá trị trong classification thông thường, nhưng không được áp dụng máy móc nếu nó trộn tương lai vào quá khứ.

Việc lựa chọn validation/CV strategy phải dựa trên temporal structure thực tế của project.

`M3.1-P02: temporal validity > giữ tỷ lệ class đẹp bằng cách trộn thời gian.`

---

## 6.4. Final test phải được cách ly về mặt quy trình

Final test không chỉ phải “không đi vào model.fit()`”.

Con người cũng không được liên tục quan sát test result rồi điều chỉnh hệ thống.

Nếu quyết định của chúng ta thay đổi dựa trên final-test performance thì test set đã gián tiếp tham gia model selection.

`M3.1-P03: final test isolation áp dụng cho cả code lẫn quyết định của người làm project.`

---

## 6.5. Experiment phải có thể tái hiện

Mỗi experiment sau này tối thiểu phải ghi nhận:

* thời gian/boundary của train;
* thời gian/boundary của validation;
* thời gian/boundary của test nếu được phép dùng;
* số transaction;
* số fraud;
* fraud rate;
* feature-set version;
* preprocessing version;
* model/configuration;
* random seed nếu có stochastic operation;
* metric;
* threshold nếu có;
* kết quả;
* limitation hoặc anomaly quan sát được.

`RANDOM_STATE = 42` tiếp tục được sử dụng như quy ước project cho các thao tác ngẫu nhiên cần khả năng tái hiện; nó không dùng để thay thế temporal split.

`M3.1-P04: experiment không đủ metadata thì chưa được coi là bằng chứng chính thức.`

---

# 7. Quy tắc thay đổi kế hoạch M3

Kế hoạch M3.1 → M3.8 là khung ban đầu, không phải cấu trúc bất biến.

Trong quá trình thực hiện, có thể:

* thêm bước;
* bỏ bước;
* gộp bước;
* tách bước;
* đổi thứ tự;
* thay đổi candidate;
* điều chỉnh experiment protocol;

nếu output thực tế chứng minh rằng thay đổi đó cần thiết.

Tuy nhiên không được thay đổi âm thầm.

Mọi thay đổi có ảnh hưởng đến thiết kế experiment phải được ghi vào `M3 Decision Log` theo tối thiểu:

```text
Decision ID
Quyết định cũ
Quan sát / bằng chứng mới
Vấn đề phát hiện
Quyết định mới
Lý do
Ảnh hưởng đến các bước M3/M4/modeling phía sau
Trạng thái
```

Nguyên tắc ưu tiên:

`Problem statement + guardrail đã khóa`

`→ bằng chứng dữ liệu thực tế`

`→ output của experiment hiện tại`

`→ lập luận ML`

`→ kế hoạch M3 ban đầu`

Kế hoạch không được đứng trên bằng chứng thực tế.

`M3.1-P05: kế hoạch M3 là adaptive plan có kiểm soát, không phải template bắt buộc.`

---

# 8. Những quyết định M3.1 cố ý chưa khóa

M3.1 chưa có bằng chứng đủ để quyết định các mục sau.

### O01 — Final train / validation / test boundaries

`Trạng thái: OPEN`

Được chuyển sang temporal-map và split-design steps.

Mọi partition phải tôn trọng past → future ordering và có đủ positive sample để evaluation có ý nghĩa.

---

### O02 — Final training window

`Trạng thái: OPEN`

M2 đã chứng minh nhiều recent temporal subset khả thi, nhưng chưa có experiment chứng minh window nào phù hợp hơn.

---

### O03 — Primary metric và secondary metrics

`Trạng thái: OPEN`

Đã khóa:

`Accuracy không được dùng đơn độc.`

Chưa khóa metric nào sẽ là primary objective.

---

### O04 — Validation / temporal-CV strategy cụ thể

`Trạng thái: OPEN`

Chưa kết luận dùng một validation period, rolling/forward validation hay cấu trúc khác.

Quyết định phải dựa trên temporal map và quy mô positive sample thực tế.

---

### O05 — Training-window comparison winner

`Trạng thái: OPEN`

M3 trước hết phải thiết kế comparison protocol.

Chỉ được khóa winner khi có preprocessing/baseline/experiment đủ hợp lệ.

---

### O06 — Class-imbalance technique

`Trạng thái: OPEN`

Chưa quyết định:

`class_weight`

`oversampling`

`undersampling`

`SMOTE`

hay không xử lý.

M3.1 chỉ khóa vị trí hợp lệ của chúng: training data sau split.

---

### O07 — Final feature set

`Trạng thái: OUT OF M3.1 / chuyển M4 + modeling`

M2.8 chỉ chứng minh behavioral feature có thể tính causal, chưa chứng minh model gain.

---

### O08 — Final model family

`Trạng thái: OUT OF M3.1`

EDA không phải bằng chứng để chọn thuật toán cuối.

---

### O09 — Final classification threshold

`Trạng thái: OUT OF M3.1`

Threshold phải được lựa chọn bằng validation objective ở giai đoạn thích hợp, không được chốt từ EDA hoặc M3.1.

---

# 9. Decision Log M3.1

```markdown
| Decision ID | Nội dung | Trạng thái |
|---|---|---|
| M3.1-D01 | Giữ nguyên problem statement và prediction point đã khóa | LOCKED |
| M3.1-D02 | M3 là experiment-design milestone, không phải final modeling/preprocessing milestone | LOCKED |
| M3.1-G01 | Không dùng raw User/Card/Merchant Name trực tiếp | LOCKED |
| M3.1-G02 | Không dùng Errors? trong Model V1 | LOCKED |
| M3.1-G03 | Evaluation chính phải theo thời gian | LOCKED |
| M3.1-G04 | Final test không được dùng lặp lại để lựa chọn | LOCKED |
| M3.1-G05 | Historical feature phải strict causal | LOCKED |
| M3.1-G06 | Same-Timestamp transaction không làm history cho nhau | LOCKED |
| M3.1-G07 | Historical-context rows khác khái niệm classifier-training rows | LOCKED |
| M3.1-G08 | Preprocessing fit chỉ trên TRAIN | LOCKED |
| M3.1-G09 | Resampling chỉ trên training data sau split | LOCKED |
| M3.1-G10 | Accuracy không được dùng đơn độc | LOCKED |
| M3.1-G11 | 2020 không dùng làm final fraud-performance test | LOCKED |
| M3.1-G12 | Không xử lý missing location / negative Amount máy móc | LOCKED |
| M3.1-G13 | Không dùng full-history association như bằng chứng generalization | LOCKED |
| M3.1-G14 | Main evaluation chủ yếu phản ánh existing User/Card | LOCKED |
| M3.1-G15 | Không overclaim production-banking performance | LOCKED |
| M3.1-P01 | Comparison phải xác định biến thay đổi và điều kiện kiểm soát | LOCKED |
| M3.1-P02 | Temporal validity được ưu tiên hơn random/stratified convenience | LOCKED |
| M3.1-P03 | Final-test isolation áp dụng cho cả code và human decision | LOCKED |
| M3.1-P04 | Experiment phải có metadata để tái hiện | LOCKED |
| M3.1-P05 | Kế hoạch M3 được phép điều chỉnh có Decision Log | LOCKED |
```

---

# 10. M3.1 Gate

M3.1 chỉ PASS nếu các câu sau đều trả lời được.

### GATE-01 — Mục tiêu của M3 đã rõ chưa?

`PASS`

M3 thiết kế experiment và evaluation protocol; không phải giai đoạn tìm final model.

### GATE-02 — Prediction point đã rõ chưa?

`PASS`

Prediction tại thời điểm transaction cần được screening.

### GATE-03 — Temporal direction đã khóa chưa?

`PASS`

Evaluation chính phải theo hướng quá khứ → tương lai.

### GATE-04 — Test-isolation rule đã khóa chưa?

`PASS`

Final test không tham gia các vòng lựa chọn lặp lại.

### GATE-05 — Leakage rule cho historical feature đã rõ chưa?

`PASS`

Chỉ dùng `timestamp(history) < timestamp(current)`.

### GATE-06 — Data-learning boundary của preprocessing đã rõ chưa?

`PASS`

Mọi statistic/transformer có bước học chỉ fit từ training data.

### GATE-07 — Resampling boundary đã rõ chưa?

`PASS`

Chỉ training data sau split.

### GATE-08 — Feature exclusion quan trọng đã khóa chưa?

`PASS`

Không raw User/Card/Merchant Name; không `Errors?` cho Model V1.

### GATE-09 — Final-test temporal guardrail đã rõ chưa?

`PASS`

Không dùng 2020 làm final fraud test.

### GATE-10 — Metric limitation đã rõ chưa?

`PASS`

Accuracy không được dùng đơn độc; primary metric vẫn để OPEN cho M3.4.

### GATE-11 — Những quyết định chưa đủ bằng chứng đã được giữ OPEN chưa?

`PASS`

Split boundary, training window, metric chính, CV strategy và các lựa chọn modeling chưa bị khóa sớm.

### GATE-12 — Kế hoạch M3 có cơ chế điều chỉnh theo thực tế chưa?

`PASS`

Mọi điều chỉnh được phép khi có bằng chứng, nhưng phải ghi Decision Log.

---

# 11. Kết luận M3.1

`Decision ID: M3.1-FINAL-D01`

`Experiment objective: ĐÃ KHÓA`

`Prediction point: ĐÃ KHÓA`

`Temporal-evaluation principle: ĐÃ KHÓA`

`Leakage guardrails: ĐÃ KHÓA`

`Test-isolation principle: ĐÃ KHÓA`

`Preprocessing/resampling boundary: ĐÃ KHÓA`

`Metric limitation: ĐÃ KHÓA`

`Final split: CHƯA KHÓA — ĐÚNG PHẠM VI`

`Final training window: CHƯA KHÓA — ĐÚNG PHẠM VI`

`Primary metric: CHƯA KHÓA — ĐÚNG PHẠM VI`

`Final model / threshold / feature set: CHƯA KHÓA — ĐÚNG PHẠM VI`

`Blocking issue: NONE`

`M3.1 Gate: PASS`

`M3.1 Status: PASS — PROTOCOL LOCKED`

Milestone 3 từ thời điểm này phải tuân thủ các guardrail và experiment principles đã nêu ở trên.

Nếu các bước tiếp theo phát hiện bằng chứng mới yêu cầu điều chỉnh, thay đổi phải được ghi lại bằng Decision Log thay vì sửa âm thầm.

**Bước tiếp theo: M3.2 — xây temporal map và xác định các modeling / validation / future-evaluation region candidate dựa trên dữ liệu thực tế.**
