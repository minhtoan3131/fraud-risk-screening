# MILESTONE 0 — PROJECT CHARTER v0.1

**Project:** AI Transaction Fraud Risk Screening
**Học phần:** Trí tuệ nhân tạo — IT3234
**Loại đề:** Đề số 50 — Đề mở
**Trạng thái:** Approved

Đề thi xác nhận Đề số 50 là đề mở; bài tập lớn còn có yêu cầu báo cáo, xây dựng chương trình, kiểm thử và GitHub, nên project ngay từ đầu phải được định nghĩa sao cho vừa đúng bài toán ML, vừa có sản phẩm có thể trình bày và bảo vệ. 

Mục đích của giai đoạn này là trả lời thật rõ: **project giải bài toán gì, giải đến đâu, output là gì, cái gì không làm, và đến cuối dự án dựa vào đâu để nói rằng project đã hoàn thành**. Trong quy trình đã thống nhất, M0 chính là bước “khóa bài toán và phạm vi”; sau khi qua gate này mới chuyển sang M1 — chọn và hiểu nguồn dữ liệu. 

Tôi sẽ lập cho bạn một **Project Charter v0.1**. Những gì đã đủ căn cứ sẽ được khóa ngay. Những gì phụ thuộc dataset sẽ cố ý để trạng thái `TBD` thay vì tự đoán.

---

## 1. Tên đề tài

Tên đề tài đề xuất chính thức:

> **Xây dựng chương trình đánh giá dấu hiệu gian lận của giao dịch tài chính**

Tên này không phải đặt ngẫu nhiên. Nó cố ý dùng cụm **“đánh giá dấu hiệu gian lận”** thay vì “hệ thống phát hiện gian lận” hay “hệ thống chống gian lận”, vì model của chúng ta chỉ cung cấp một tín hiệu đánh giá ban đầu chứ không có quyền kết luận cuối cùng một giao dịch chắc chắn gian lận. Đây cũng là tên đã được định hướng xuyên suốt các tài liệu trước. 

**Quyết định M0-D01: KHÓA TẠM THỜI.**

“Tạm thời” ở đây chỉ vì tên đề cuối cùng vẫn có thể phải tuân theo cách đăng ký/chấp thuận của giảng viên. Về mặt thiết kế project, chúng ta sử dụng tên này từ bây giờ.

---

# 2. Bối cảnh và vấn đề cần giải quyết

Trong dữ liệu giao dịch tài chính, một số giao dịch có thể mang những pattern tương đồng với các giao dịch đã được gắn nhãn gian lận. Nếu có dữ liệu lịch sử có nhãn, Machine Learning có thể học mối quan hệ giữa các đặc trưng của giao dịch và nhãn đó để tạo dự đoán cho giao dịch chưa được model nhìn thấy.

Tuy nhiên, chúng ta không định nghĩa project là:

> “AI xác định giao dịch có thực sự gian lận hay không.”

Mà là:

> “AI sử dụng dữ liệu có nhãn để tạo một tín hiệu đánh giá mức độ nghiêng về lớp có dấu hiệu gian lận.”

Cách định vị này nhất quán với tài liệu đề xuất: model đóng vai trò **thành phần sàng lọc rủi ro ban đầu**, còn rule, xác minh bổ sung, review hay quyết định nghiệp vụ nằm ở các tầng khác. 

Có thể hình dung:

```text
Giao dịch
    │
    ▼
Thông tin/đặc trưng của giao dịch
    │
    ▼
Machine Learning model
    │
    ▼
Probability / Risk score
    │
    ▼
Tín hiệu sàng lọc
    │
    ▼
Các bước xử lý khác
    (ngoài phạm vi project)
```

Đây là điểm bạn cần nắm rất chắc ngay từ M0:

**Model không đồng nghĩa với quyết định nghiệp vụ.**

---

# 3. Problem Statement

Đây là câu quan trọng nhất của M0.

Tôi đề xuất chốt phiên bản 0.1 như sau:

> **Sử dụng các thông tin có sẵn về một giao dịch tại thời điểm đánh giá để xây dựng mô hình học máy phân loại nhị phân, nhằm ước lượng khả năng giao dịch thuộc lớp có dấu hiệu gian lận theo nhãn của bộ dữ liệu; kết quả được sử dụng như một tín hiệu hỗ trợ sàng lọc ban đầu, không phải quyết định cuối cùng về gian lận.**

Từng cụm trong câu này đều có chủ ý.

| Thành phần                        | Ý nghĩa                                                                                     |
| --------------------------------- | ------------------------------------------------------------------------------------------- |
| “thông tin có sẵn”                | Không được dùng feature biết tương lai hoặc thông tin chỉ xuất hiện sau khi kết quả đã biết |
| “một giao dịch”                   | Đối tượng mà chúng ta muốn đánh giá ở tầng khái niệm                                        |
| “tại thời điểm đánh giá”          | Đặt ranh giới thời gian để phòng data leakage                                               |
| “phân loại nhị phân”              | Bài toán ML có 2 lớp                                                                        |
| “ước lượng khả năng”              | Không khẳng định tuyệt đối                                                                  |
| “theo nhãn của bộ dữ liệu”        | Ground truth phải tuân theo nguồn dataset                                                   |
| “sàng lọc ban đầu”                | Xác định đúng vai trò ứng dụng                                                              |
| “không phải quyết định cuối cùng” | Tránh overclaim                                                                             |

Việc đưa yếu tố thời điểm vào problem statement đặc biệt quan trọng. Tài liệu Supervised Learning cũng yêu cầu trước khi `model.fit()` phải xác định sample là gì, target là gì và feature nào thực sự tồn tại tại thời điểm cần dự đoán. 

**Quyết định M0-D02: KHÓA.**

---

# 4. Bản chất bài toán Machine Learning

Project này là:

**Supervised Learning → Classification → Binary Classification.**

Không phải regression.

Không phải clustering.

Không phải anomaly detection thuần túy.

Không phải 3-class classification.

Không phải Deep Learning theo thiết kế hiện tại.

Luồng ML cốt lõi là:

```text
Training data có nhãn
        │
        ├── X = transaction features
        │
        └── y = target label
                 │
                 ▼
          Binary classifier
                 │
                 ▼
            Probability
                 │
                 ▼
          Predicted class
```

Tài liệu định hướng cũng chốt kiến trúc này: binary classification ở tầng ML, còn probability/risk score được sử dụng ở tầng ứng dụng. Ý nghĩa chính xác của nhãn phải tuân theo dataset được chọn, không tự thay đổi ground truth của nguồn dữ liệu. 

### Positive class

Về mặt khái niệm:

```text
Negative class → giao dịch bình thường / non-fraud
Positive class → giao dịch thuộc lớp fraud / suspicious theo dataset
```

Nhưng chúng ta **chưa khóa rằng `0 = normal`, `1 = fraud`**.

Tại sao?

Vì chưa có dataset.

Rất nhiều dataset dùng quy ước 0/1 như vậy, nhưng M0 không được phép lấy thói quen phổ biến thay cho tài liệu nguồn. M1 sẽ kiểm tra chính xác nhãn của dataset.

Đây là một ví dụ quan trọng của cách làm project bài bản:

> Biết điều gì chưa biết cũng là một phần của quy trình.

**Quyết định M0-D03: khóa binary classification; mapping nhãn cụ thể = TBD tại M1.**

---

# 5. Đơn vị dự đoán — một sample là gì?

Ở tầng bài toán, một sample được định nghĩa là:

> **Một giao dịch tài chính cần được đánh giá.**

Hay nói cách khác:

```text
1 sample ≈ 1 transaction
```

Ví dụ khái niệm:

```text
Transaction A
amount = ...
time = ...
channel = ...
...
label = ?
```

Model nhận các feature hợp lệ của transaction A và đưa ra đánh giá.

Tuy nhiên có một điểm chưa được phép khẳng định cho đến M1:

> Dataset thực tế có đúng grain “1 row = 1 transaction” hay không?

Đây phải được kiểm tra trực tiếp từ tài liệu dataset. Tài liệu EDA cũng nhấn mạnh câu hỏi đầu tiên khi đọc dữ liệu phải là “một hàng đại diện cho cái gì?”, vì identifier lặp hay cấu trúc dữ liệu chỉ có thể hiểu đúng khi biết grain. 

Vì vậy:

```text
Conceptual prediction unit:
1 transaction             → LOCKED

Dataset grain:
1 row = ?                 → TBD tại M1
```

**Quyết định M0-D04: khóa đối tượng nghiệp vụ là giao dịch; grain dữ liệu chờ xác minh.**

---

# 6. Input và output của hệ thống

Ở M0 chúng ta chỉ định nghĩa **interface về mặt khái niệm**, chưa định nghĩa field cụ thể.

## Input

Input là các feature của một giao dịch **được xác nhận là hợp lệ để sử dụng tại thời điểm screening**.

Ví dụ minh họa có thể là:

```text
amount
transaction_time
transaction_type
channel
...
```

Nhưng đây chưa phải feature list chính thức.

M0 tuyệt đối không được tự quyết định:

```text
amount
location
device
beneficiary
...
```

chỉ vì chúng “nghe hợp lý”.

Feature nào thật sự tồn tại phụ thuộc dataset.

Do đó:

> **Exact input schema = TBD tại M1.**

---

## Output

Ở đây chúng ta phân tách rõ ba khái niệm.

### Tầng 1 — Probability / score

Ví dụ:

```text
P(positive class) = 0.87
```

Đây là output quan trọng nếu classifier hỗ trợ xác suất.

### Tầng 2 — Predicted class

Ví dụ:

```text
prediction = SUSPICIOUS
```

Class được xác định từ score và threshold.

### Tầng 3 — Application interpretation

Có thể biểu diễn:

```text
riskScore = 0.87
riskLevel = HIGH
screeningResult = REVIEW_REQUIRED
```

Nhưng phải nhớ:

```text
LOW
MEDIUM
HIGH
```

**không phải ground truth mới**.

Tài liệu định hướng đã chỉ rõ không được tự biến dataset hai lớp thành bài toán ba lớp chỉ để project có vẻ phức tạp hơn. Cách sạch là giữ binary classification, sau đó dùng probability → threshold → risk band ở tầng ứng dụng. 

Do đó:

**Quyết định M0-D05:**

```text
ML output:
probability / score + binary prediction

Application output:
có thể thêm risk level / screening result

Exact threshold:
TBD, không khóa trong M0
```

---

# 7. Mục tiêu của project

Chúng ta cần phân biệt **mục tiêu học thuật**, **mục tiêu kỹ thuật** và **mục tiêu ứng dụng**.

| Nhóm            | Mục tiêu                                                                               |
| --------------- | -------------------------------------------------------------------------------------- |
| Học thuật       | Hiểu và thực hiện đúng quy trình một bài toán supervised binary classification         |
| Dữ liệu         | Hiểu dataset, target, feature, chất lượng dữ liệu và class distribution                |
| ML              | Xây dựng và so sánh một số classifier truyền thống phù hợp                             |
| Đánh giá        | Đánh giá bằng Confusion Matrix, Precision, Recall, F1 và Accuracy thay vì chỉ Accuracy |
| Thí nghiệm      | Giữ ranh giới train/validation/test đúng, hạn chế leakage                              |
| Ứng dụng        | Chuyển prediction/probability thành tín hiệu screening dễ quan sát                     |
| Phần mềm        | Có chương trình Python chạy inference cho mẫu giao dịch                                |
| Học tập cá nhân | Có thể tự giải thích toàn bộ pipeline và từng quyết định khi vấn đáp                   |

Một mục tiêu **không có** ở đây là:

> “Đạt Accuracy 99%.”

Ngay lúc này chưa có dataset, chưa biết imbalance, chưa biết baseline, nên việc tự đặt một con số metric đích sẽ không có cơ sở.

Metric và tiêu chí chọn model cụ thể sẽ được thiết kế ở **M3 — Experiment Design**.

---

# 8. Phạm vi project

Đây là phần phải khóa khá mạnh để tránh project phình ra.

| IN SCOPE — project môn AI                       | OUT OF SCOPE — không làm trong môn      |
| ----------------------------------------------- | --------------------------------------- |
| Dataset giao dịch có nhãn                       | Hệ thống ngân hàng thật                 |
| Dataset understanding / EDA                     | Core banking                            |
| Data cleaning / preprocessing                   | Payment processing                      |
| Feature engineering vừa phải nếu dữ liệu hỗ trợ | Fraud investigation workflow            |
| Binary classification                           | Quyết định chặn/cho phép giao dịch thật |
| 2–3 classifier truyền thống                     | Hệ thống fraud prevention hoàn chỉnh    |
| Train/validation/test đúng quy trình            | Streaming real-time                     |
| Cross-validation nếu phù hợp                    | Kafka/message broker                    |
| Confusion Matrix                                | Microservices                           |
| Accuracy / Precision / Recall / F1              | Kubernetes                              |
| Class imbalance nếu thực sự tồn tại             | Distributed infrastructure              |
| Probability / risk score                        | MLOps platform hoàn chỉnh               |
| Threshold analysis nếu có ý nghĩa               | Production banking integration          |
| Python inference/demo                           | Backend project quy mô lớn              |
| Lưu/load model nếu cần                          | Deep Learning chỉ để tăng độ phức tạp   |

Việc tách scope này phù hợp với định hướng đã thống nhất: sản phẩm AI có thể giữ khả năng tái sử dụng về sau, nhưng **không xây Project Backend số 2 trong môn AI**. 

Điểm cần nhớ:

> **Dữ liệu và bài toán nghiêm túc; AI vừa sức; backend để sau.**

Đây cũng chính là nguyên tắc kết luận trong tài liệu định hướng. 

**Quyết định M0-D06: KHÓA.**

---

# 9. Thuật toán — hiện tại khóa đến mức nào?

Chúng ta chưa chọn model cuối.

M0 chỉ khóa **candidate family** ở mức vừa sức:

```text
Logistic Regression
Decision Tree
Random Forest
```

Điều này không có nghĩa chắc chắn cả ba sẽ xuất hiện trong model cuối.

Mục tiêu sau này sẽ là:

```text
baseline
    ↓
các model ứng viên
    ↓
thí nghiệm
    ↓
đánh giá
    ↓
model phù hợp nhất
```

Chúng ta không làm:

```text
Random Forest nghe mạnh
→ chọn luôn
```

Model cuối phải là kết quả thực nghiệm.

**Quyết định M0-D07: candidate models được định hướng; final model = chưa chọn.**

---

# 10. Metric — hiện tại khóa đến mức nào?

Chúng ta có thể khóa **bộ metric phải quan tâm**:

```text
Confusion Matrix
Accuracy
Precision
Recall
F1-score
```

Nhưng chưa khóa:

```text
primary metric = ?
acceptable Recall = ?
acceptable F1 = ?
```

Lý do là chưa nhìn thấy class distribution và chưa hiểu đặc điểm dataset.

Tài liệu đánh giá Classification cũng chỉ ra rằng evaluation phải cho biết không chỉ model đúng tổng cộng bao nhiêu, mà còn **model đang sai theo kiểu nào**; hai model có cùng Accuracy vẫn có thể có hành vi rất khác nhau. 

Đối với fraud screening, nhiều khả năng Recall của positive class sẽ quan trọng, nhưng ta **chưa được chốt “Recall là metric số 1”** khi chưa thực hiện M1–M3.

Đây là sự khác nhau giữa:

```text
Có định hướng
```

và

```text
Đã có bằng chứng để ra quyết định
```

**Quyết định M0-D08: metric set khóa; primary metric và target value = TBD tại M3.**

---

# 11. Tiêu chí hoàn thành toàn project — Definition of Done

Project chỉ được xem là hoàn thành khi có đủ bằng chứng về **dữ liệu, ML, phần mềm và khả năng giải thích**.

| Điều kiện                                  | Ý nghĩa                                             |
| ------------------------------------------ | --------------------------------------------------- |
| Có dataset có nguồn rõ ràng                | Không dùng file không xác định xuất xứ              |
| Hiểu grain, target và feature              | Không train model trên dữ liệu chưa hiểu            |
| Có EDA có kết luận                         | Không chỉ chụp `describe()`                         |
| Preprocessing có lý do                     | Không thao tác máy móc                              |
| Không có leakage rõ ràng                   | Train/test và preprocessing đúng ranh giới          |
| Có baseline                                | Biết model cải thiện so với mốc nào                 |
| Có so sánh model hợp lý                    | Không chọn model theo cảm tính                      |
| Có Confusion Matrix + nhiều metric         | Không chọn dựa trên Accuracy đơn độc                |
| Có phân tích FP/FN                         | Hiểu model sai ở đâu                                |
| Có xử lý imbalance nếu cần                 | Chỉ dùng khi dữ liệu chứng minh cần                 |
| Có probability/risk score nếu model hỗ trợ | Phục vụ screening                                   |
| Threshold có lý do nếu thay đổi            | Không chọn ngưỡng tùy tiện                          |
| Có final test                              | Có đánh giá cuối trên dữ liệu chưa dùng để lựa chọn |
| Có chương trình Python demo                | Model thực sự sử dụng được                          |
| Có kiểm thử đầu vào                        | Đáp ứng yêu cầu chương trình                        |
| Có GitHub                                  | Đáp ứng yêu cầu học phần                            |
| Có báo cáo                                 | Trình bày lại bằng chứng của project                |
| Bạn tự giải thích được                     | Project thực sự thuộc quyền kiểm soát của bạn       |

Đề thi chính thức yêu cầu phần chương trình phải được kiểm thử với **tối thiểu hai mẫu dữ liệu đầu vào**, kèm kết quả; đồng thời báo cáo cần có đánh giá, hướng phát triển, tài liệu tham khảo và link GitHub. 

Tài liệu định hướng cũng có một bộ tiêu chí “đủ tốt để nộp môn” rất gần với Definition of Done trên: dataset có nguồn, EDA/preprocessing hợp lý, baseline/model, hiểu metric và imbalance, probability/risk score, chương trình Python demo, test mẫu, giải thích pipeline và hạn chế trung thực. 

---

# 12. Những thứ cố ý CHƯA quyết định trong M0

Đây là phần rất quan trọng.

| Chưa quyết định          | Tại sao chưa quyết định      | Sẽ quyết định ở đâu |
| ------------------------ | ---------------------------- | ------------------- |
| Dataset cụ thể           | Chưa audit các candidate     | M1                  |
| Nguồn dataset            | Chưa lựa chọn                | M1                  |
| Một row chính xác là gì  | Phải đọc tài liệu dataset    | M1                  |
| Target column            | Phụ thuộc dataset            | M1                  |
| Ý nghĩa 0/1              | Phụ thuộc ground truth       | M1                  |
| Feature list             | Phụ thuộc dataset            | M1–M2               |
| Feature nào bỏ           | Phải EDA/leakage check       | M1–M4               |
| Missing strategy         | Phải nhìn dữ liệu            | M2–M4               |
| Scaling                  | Phụ thuộc feature/model      | M4                  |
| Train/test ratio         | Phụ thuộc dataset            | M3                  |
| Primary metric           | Phụ thuộc imbalance/mục tiêu | M3                  |
| Model tốt nhất           | Phải thực nghiệm             | M5–M7               |
| Class weight / SMOTE     | Chỉ dùng nếu cần             | M7                  |
| Threshold                | Phải dựa trên validation     | M7                  |
| LOW/MEDIUM/HIGH boundary | Phải có cơ sở từ experiment  | M7                  |
| Final performance        | Chưa train model             | M7                  |

Nếu ai hỏi bạn ngay lúc này:

> “Dùng SMOTE chưa?”

Câu trả lời đúng không phải “có” hay “không”.

Mà là:

> “Chưa quyết định vì chưa đánh giá class distribution của dataset. Nếu minority class mất cân bằng đủ đáng kể, tôi sẽ thiết kế thí nghiệm so sánh các phương án trên training data.”

Đó là tư duy chúng ta cần giữ xuyên suốt.

---

# 13. Decision Log sau M0 v0.1

| ID     | Quyết định                                       | Trạng thái                            |
| ------ | ------------------------------------------------ | ------------------------------------- |
| M0-D01 | Tên đề tài                                       | 🟡 Chờ chốt hành chính với GV nếu cần |
| M0-D02 | Problem Statement                                | 🟢 Khóa                               |
| M0-D03 | Binary Classification                            | 🟢 Khóa                               |
| M0-D04 | Đối tượng cần đánh giá = transaction             | 🟢 Khóa về khái niệm                  |
| M0-D05 | Probability/risk score + binary prediction       | 🟢 Khóa                               |
| M0-D06 | Risk screening, không phải fraud decision system | 🟢 Khóa                               |
| M0-D07 | Logistic/Tree/Forest là candidate family         | 🟢 Khóa ở mức định hướng              |
| M0-D08 | Confusion Matrix + Accuracy/Precision/Recall/F1  | 🟢 Khóa                               |
| M0-D09 | Dataset cụ thể                                   | 🔵 Chờ M1                             |
| M0-D10 | Mapping target                                   | 🔵 Chờ M1                             |
| M0-D11 | Primary metric                                   | 🔵 Chờ M3                             |
| M0-D12 | Final model                                      | 🔵 Chờ M7                             |
| M0-D13 | Threshold/risk bands                             | 🔵 Chờ M7                             |

Nhìn bảng này bạn có thể thấy M0 **không phải “chưa làm gì”**. Chúng ta đã khóa rất nhiều quyết định kiến trúc, đồng thời cố ý giữ mở những quyết định chỉ có thể trả lời bằng dữ liệu.

---

# 14. Gate của Milestone 0

Tôi chưa muốn đánh dấu M0 là `DONE` ngay lập tức.

Vì mục tiêu của bạn là tự kiểm soát project, nên gate không phải là “tôi đã viết Project Charter cho bạn”. Gate là **bạn hiểu Project Charter đó**.

Trước khi chúng ta chuyển sang M1, hãy trả lời bằng lời của bạn, không cần văn phong báo cáo, 5 câu sau:

1. **Project này thực chất đang giải bài toán Machine Learning gì?**

2. **Tại sao tên đề dùng “đánh giá dấu hiệu gian lận” thay vì khẳng định “phát hiện giao dịch gian lận”?**

3. **Nếu model trả `P(suspicious) = 0.87`, con số 0.87 có phải bằng chứng rằng giao dịch chắc chắn gian lận 87% không? Theo bạn nên hiểu nó thế nào?**

4. **Tại sao hiện tại chúng ta chưa chọn Random Forest, SMOTE, threshold hay primary metric mặc dù đều đã học?**

5. **Hãy nói ngắn gọn những gì project này sẽ làm và những gì project này cố ý không làm.**

Bạn cứ trả lời theo hiểu biết hiện tại của mình, kể cả chỗ nào chưa chắc. Tôi sẽ review từng câu như một buổi vấn đáp nhỏ, sửa đúng những điểm còn lệch. **Khi 5 câu này đạt, chúng ta đóng M0 và chuyển chính thức sang M1 — Dataset Audit: tìm 2–3 dataset candidate rồi đánh giá nguồn, grain, feature, target, class imbalance và khả năng giải thích.**
