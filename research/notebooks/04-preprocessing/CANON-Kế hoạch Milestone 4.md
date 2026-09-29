# Kế hoạch M4
## 1. M4 thực chất là gì?

Sau M3, chúng ta đã đi qua một ranh giới rất quan trọng.

M3 không tạo model. M3 đã khóa “luật chơi” cho các thí nghiệm phía sau: chia dữ liệu theo thời gian, vai trò TRAIN/VALIDATION/FINAL TEST, metric, nguyên tắc lịch sử nhân quả, quyền sử dụng từng partition, class imbalance, temporal CV, test isolation… M3 đã PASS và không còn vấn đề phương pháp nào chặn M4. Những thứ vẫn OPEN chủ yếu là feature, preprocessing và những quyết định cần experiment thật. :chatgpt-content-reference{index="0"}

Vì vậy có thể hiểu ngắn gọn:

> **M3 trả lời: “thí nghiệm ML phải tuân theo luật nào?”**  
> **M4 trả lời: “từ raw transaction, chính xác chúng ta tạo ra X như thế nào để model có thể học đúng và không leakage?”**

CANON M3.8 xác định trọng tâm M4 là:

```text
feature engineering
+
preprocessing representation
+
baseline-ready modeling matrix
+
leakage-safe implementation
```

và đặc biệt M4 **không được mở FINAL TEST để hỗ trợ feature engineering**. :chatgpt-content-reference{index="1"}

Đây là điểm cần nắm chắc: mục tiêu M4 **không phải làm F1 càng cao càng tốt**, chưa phải chọn Random Forest hay Logistic Regression thắng, cũng chưa phải SMOTE/tuning/threshold. M4 phải tạo được một pipeline dữ liệu đủ đúng, đủ rõ và đủ tái hiện để sang M5 chúng ta có thể huấn luyện baseline một cách đáng tin.

---

# 2. Đầu vào của M4 đã có những gì?

M4 không bắt đầu lại từ raw CSV rồi tự suy nghĩ tất cả từ đầu.

Từ M2, chúng ta đã biết khá nhiều vấn đề preprocessing thực tế.

Negative Amount chiếm tỷ lệ đáng kể và chưa có căn cứ để coi là lỗi, vì vậy M4 không được tự động `abs()`, xóa hoặc clip. Location missing cũng mang tính cấu trúc: giao dịch online có State/Zip missing theo bản chất dữ liệu, nên không được coi toàn bộ missing như missing ngẫu nhiên rồi median/mode hoặc drop máy móc. MCC/location có nguy cơ trở thành shortcut hoặc proxy cho merchant identity. Đồng thời User/Card history đủ sâu và các behavioral feature causal đã được chứng minh là tính được. :chatgpt-content-reference{index="2"}

M3 sau đó để lại một số câu hỏi trực tiếp cho M4:

```text
Final/candidate feature set
Behavioral feature set
Exact historical lookback
MCC/location encoding
Amount transformation
```

Trong đó các behavioral candidate đã có ít nhất:

```text
time_since_previous_transaction
transactions_last_1h
amount_minus_previous_mean
is_new_merchant
```

Nhưng M2 mới chứng minh rằng chúng **tính được đúng causal**, chưa chứng minh chúng giúp classification tốt hơn. :chatgpt-content-reference{index="3"}

Điều đó có nghĩa M4 phải phân biệt hai câu rất khác nhau:

> “Feature này có thể xây đúng không?”

và

> “Feature này có thực sự giúp model không?”

Câu thứ nhất chủ yếu thuộc M4.

Câu thứ hai cần baseline/model experiment ở các milestone modeling sau. M3 còn ghi rõ rằng chưa được mang forward kết luận “behavioral feature cải thiện F1” vì chưa có experiment chứng minh. :chatgpt-content-reference{index="4"}

---

# 3. Tôi đề xuất chia M4 thành 8 bước

Hiện chúng ta **chưa có CANON Kế hoạch M4 riêng**. Vì vậy 8 bước dưới đây là cấu trúc tôi đề xuất dựa trên toàn bộ handoff M2–M3; khi bắt đầu M4 chúng ta nên khóa nó thành kế hoạch chính thức rồi mới làm notebook.

## M4.1 — Khóa phạm vi, input contract và guardrail

Bước đầu không cần vội code feature.

Chúng ta sẽ tạo một `M4 Preprocessing & Feature Engineering Protocol`, ghi rõ M4 được làm gì và không được làm gì.

Những thứ M4 phải kế thừa nguyên vẹn từ M3 gồm temporal split, metric strategy, partition permissions, causal-history rules, feature guardrails, hai training-window candidate, temporal-CV policy, imbalance boundaries và test-isolation policy. :chatgpt-content-reference{index="5"}

Ví dụ những invariant phải khóa ngay:

```text
Raw User/Card/Merchant Name
→ không đưa trực tiếp vào classifier.

Errors?
→ không dùng cho Model V1.

Preprocessing learned state
→ TRAIN only.

Historical feature
→ Timestamp(history) < Timestamp(current).

Transaction cùng Timestamp
→ không làm history cho nhau.

FINAL TEST
→ tuyệt đối chưa dùng để quyết định feature/preprocessing.

Raw artifact
→ không chỉnh sửa.
```

Kết quả M4.1 là chúng ta biết chính xác pipeline được phép làm gì trước khi viết nó.

---

## M4.2 — Xây representation cơ sở

Đây là lớp từ raw CSV sang một representation kỹ thuật nhất quán.

Ví dụ:

```text
Year + Month + Day + Time
→ Timestamp

Amount string
→ numeric Amount

Is Fraud?
→ target representation rõ ràng

Use Chip
→ categorical

MCC
→ categorical code, không phải số liên tục

User + Card
→ history key

Merchant Name
→ history key, không phải raw classifier feature
```

Sau đó gắn transaction vào đúng temporal partition theo specification M3.

Quan trọng là representation này chưa phải “feature engineering thông minh”. Nó chỉ bảo đảm mọi bước phía sau đang nói về cùng một transaction, cùng timestamp, cùng target và cùng boundary.

Ở đây chúng ta cũng đặt assertion mạnh, ví dụ parse timestamp phải thành công, không overlap partition, target mapping đúng, không có FINAL TEST lọt vào development data.

---

# 4. M4.3 — Khóa cách xử lý các vấn đề chất lượng dữ liệu

Đây có lẽ là phần quan trọng nhất về preprocessing.

Chúng ta sẽ không áp dụng checklist kiểu:

> missing → fill  
> duplicate → drop  
> negative → abs  
> category → one-hot

Thay vào đó, từng vấn đề phải có semantic decision.

Ví dụ với Amount:

M2 đã xác nhận Amount âm không phải vài dòng lỗi lẻ tẻ; vì vậy M4 phải xem các phương án representation hợp lệ, giữ dấu hay tách thêm indicator, có cần transformation cho độ lệch phân bố hay không… nhưng không được sửa âm thành dương chỉ vì model “dễ học hơn”. :chatgpt-content-reference{index="6"}

Với location:

```text
Merchant City
Merchant State
Zip
```

chúng ta phải phân biệt ít nhất:

```text
online transaction
→ location missing có cấu trúc

merchant quốc tế / các trường hợp khác
→ cơ chế missing có thể khác
```

Do đó một giá trị kiểu `"MISSING"` hoặc `"NOT_APPLICABLE"` đôi khi có ý nghĩa hơn việc mode-imputation, nhưng quyết định cụ thể phải được kiểm chứng trên representation thực tế.

Exact duplicate cũng cần được xử lý bằng reasoning. Dataset không có transaction ID duy nhất nên hai dòng giống hệt nhau không tự động chứng minh đó là duplicate lỗi cần xóa.

---

# 5. M4.4 — Xây nhóm feature giao dịch cơ bản

Sau representation và data-quality policy, chúng ta mới xây feature tĩnh của transaction.

Ví dụ có thể gồm các nhóm như:

```text
Amount representation

Use Chip / transaction mode

MCC representation

location representation

thời gian trong ngày

ngày trong tuần / tháng
```

Nhưng có một nguyên tắc rất quan trọng: **không phải feature nào EDA thấy fraud rate khác nhau mạnh cũng được tự động đưa vào model**.

M2 đã cảnh báo MCC/location có thể chứa association mạnh nhưng thiếu ổn định theo thời gian hoặc phản ánh logic của synthetic generator. :chatgpt-content-reference{index="7"}

Do đó M4 sẽ thiên về:

> tạo representation hợp lý → kiểm tra cardinality/support → kiểm tra leakage/memorization risk → đưa vào candidate set

chứ không phải:

> “fraud rate của category này rất cao → feature này chắc chắn tốt”.

---

# 6. M4.5 — Xây behavioral feature đúng causal

Đây là phần đặc trưng nhất của project.

M2 đã chứng minh User/Card history đủ sâu và bốn prototype behavioral feature có thể tính đúng causal; M4 bây giờ phải biến prototype thành **pipeline ổn định, tái hiện được**. :chatgpt-content-reference{index="8"}

Ví dụ với một transaction T:

```text
T = giao dịch lúc 10:30

history hợp lệ
= chỉ transaction có timestamp < 10:30

10:30 cùng timestamp
= KHÔNG được tính

10:31
= future → KHÔNG được dùng
```

Đây là nơi User/Card/Merchant Name có vai trò quan trọng.

Chúng không được đưa raw vào classifier, nhưng có thể dùng làm khóa để hỏi:

```text
Card này giao dịch gần nhất cách đây bao lâu?

Trong 1 giờ trước đã có bao nhiêu transaction?

Amount hiện tại lệch bao nhiêu so với lịch sử trước đó?

Merchant hiện tại đã từng xuất hiện trong history của Card chưa?
```

M2 đã xác nhận older transaction có thể dùng làm causal history cho recent modeling window. Nghĩa là:

```text
history rows
≠
classifier training rows
```

Đây là một điểm khá tinh tế của project.

Ví dụ classifier có thể chỉ train trên 2018, nhưng một transaction đầu 2018 vẫn có thể sử dụng transaction trước 2018 để tạo historical context, miễn chúng xảy ra trước prediction point và không sử dụng target label làm state. M2/M3 đã cố ý giữ distinction này. :chatgpt-content-reference{index="9"}

M4 cũng phải giải quyết câu hỏi `exact historical lookback`: dùng toàn bộ history trước đó, vài năm, hay rolling horizon. Hiện câu hỏi này vẫn OPEN. :chatgpt-content-reference{index="10"}

---

# 7. M4.6 — Xây preprocessing pipeline

Sau khi có feature logic, chúng ta mới xây phần mà model thực sự nhận.

Có thể hình dung:

```text
Raw transaction
      ↓
Representation
      ↓
Static feature engineering
      ↓
Causal behavioral feature engineering
      ↓
Feature selection / column roles
      ↓
Numeric preprocessing
+
Categorical preprocessing
      ↓
Modeling matrix X
```

Tại đây cần phân biệt hai loại transformation.

Một số transformation là deterministic, ví dụ parse timestamp hay tạo hour-of-day. Nó không “học” từ dataset.

Một số transformation có learned state, ví dụ scaler, imputer statistic, category vocabulary, encoder mapping. Những thứ đó phải fit từ TRAIN mà thôi; VALIDATION và FINAL TEST chỉ được transform bằng state đã học. Đây là guardrail đã khóa. M1/M3 đã quy định preprocessing không được học encoder/scaler/statistics từ test. :chatgpt-content-reference{index="11"}

Đặc biệt, chúng ta nên triển khai bằng pipeline rõ ràng thay vì preprocessing dataframe một lần trên full dataset rồi mới split.

Sai:

```text
full data
→ fit encoder/scaler
→ split
```

Đúng về nguyên tắc:

```text
TRAIN
→ fit preprocessing

VALIDATION
→ transform only

FINAL TEST
→ chưa đụng tới trong M4
```

---

# 8. M4.7 — Tạo “baseline-ready modeling matrix”

Đây là Gate kỹ thuật lớn của M4.

Sau M4, chúng ta cần có một representation mà M5 chỉ việc gắn baseline model vào chứ không phải quay lại tự chế preprocessing.

Tối thiểu phải chứng minh được:

```text
X_train
y_train

X_validation
y_validation
```

được tạo bằng cùng một feature contract và preprocessing contract.

Nếu cần phục vụ W_LONG/W_SHORT thì pipeline cũng phải có khả năng tạo hai training population mà không vô tình làm feature logic khác nhau giữa hai bên.

Chúng ta sẽ kiểm tra những thứ như:

```text
số row đúng

boundary đúng

schema train/validation nhất quán

không có target trong X

không có raw User/Card/Merchant Name trong X

không có Errors?

không có future history

không có same-timestamp history

preprocessing fit source = TRAIN

unknown category ở validation xử lý được

NaN/inf sau preprocessing nằm trong mức cho phép

feature order ổn định

pipeline chạy lại cho cùng kết quả
```

Các công việc kiểu này phải có assertion/gate; CANON quy định nếu invariant fail thì dừng, không được cứ chạy tiếp để lấy metric. :chatgpt-content-reference{index="12"}

---

# 9. M4.8 — Tổng hợp specification, Decision Log và M4 Gate

Cuối M4, tôi đề xuất chúng ta tạo một tài liệu tương tự M2.9/M3.8 nhưng nhỏ gọn hơn:

```text
Feature Specification v1.0

Preprocessing Specification v1.0

Behavioral Feature Contract

Feature Dictionary

Pipeline / Representation version

Integrity & Leakage Audit

Decision Log

Open Questions

M4 Gate
```

M4 PASS khi chúng ta có thể trả lời chắc chắn:

> Với một transaction bất kỳ thuộc development period, pipeline tạo feature như thế nào?

> Có sử dụng bất kỳ thông tin tương lai nào không?

> Transformer nào học state và state đó được fit từ đâu?

> Missing/Amount/category được xử lý thế nào và tại sao?

> Những cột nào cuối cùng đi vào baseline candidate?

> Nếu chạy lại, representation có tái hiện được không?

> M5 có thể lấy pipeline này và huấn luyện baseline ngay không?

Khi câu trả lời là “có”, M4 đã hoàn thành nhiệm vụ.

---

# 10. M4 chưa làm những gì?

Đây là phần rất dễ nhầm.

M4 **chưa cần** trả lời:

```text
W_LONG hay W_SHORT thắng?

Logistic Regression hay Random Forest tốt hơn?

class_weight có tốt hơn không?

SMOTE có cần không?

hyperparameter nào tối ưu?

threshold bao nhiêu?

final F1 bao nhiêu?
```

M3 đã ghi rõ những quyết định này cần model/validation experiment thật. :chatgpt-content-reference{index="13"}

Thứ tự CANON hiện tại thiên về:

```text
baseline feature/preprocessing implementation
        ↓
baseline model, chưa xử lý imbalance
        ↓
training-window comparison
        ↓
feature/preprocessing experiments
        ↓
model/config experiments
        ↓
imbalance experiments nếu cần
        ↓
temporal CV / tuning
        ↓
threshold
        ↓
freeze pipeline
        ↓
FINAL TEST
```

:chatgpt-content-reference{index="14"}

Vì vậy M4 tạo **baseline feature/preprocessing implementation**. Việc chứng minh một feature giúp F1 hay không sẽ bắt đầu có ý nghĩa khi đã có baseline model.

---

# 11. Tôi và bạn sẽ phối hợp như thế nào?

Phần này chúng ta đã có CANON riêng, và M4 sẽ đặc biệt cần nó vì phần lớn công việc M4 là loại C — cần chạy trên project thật.

Vai trò của bạn và tôi không giống nhau.

**Bạn là người giữ project thật và thực thi.** Bạn chạy notebook/script trên máy thật, đúng artifact và environment; giữ nguyên output, warning và error; không sửa tay số liệu để khớp kỳ vọng. Bạn không cần tự hiểu hết output trước khi gửi cho tôi. Output thô vẫn là evidence hữu ích. :chatgpt-content-reference{index="15"}

**Tôi phụ trách thiết kế và kiểm tra logic thực nghiệm.** Tôi đọc CANON, xác định câu hỏi, thiết kế notebook/code, assertions và gate; cố gắng tránh scan thừa; sau khi bạn chạy thì kiểm tra execution, consistency, temporal boundary và leakage trước khi diễn giải. Tôi không được tự tạo ra kết quả mà bạn chưa chạy. :chatgpt-content-reference{index="16"}

Nhịp làm việc của một bước M4 sẽ như sau:

```text
1. Tôi và bạn xác định đúng một câu hỏi.

2. Tôi đọc lại evidence/CANON liên quan.

3. Tôi thiết kế notebook hoặc cell code.
   - input
   - phép kiểm tra
   - output
   - assertion
   - gate
   - Decision = OPEN

4. Bạn đưa code vào project và Run All.

5. Bạn gửi lại:
   - notebook đã chạy
   hoặc
   - output đầy đủ
   hoặc
   - error/warning nếu thất bại.

6. Tôi kiểm tra:
   execution
   → integrity
   → consistency
   → temporal/leakage logic
   → output.

7. Nếu thiếu bằng chứng:
   tôi sửa/bổ sung check
   → bạn chạy lại.

8. Khi output đủ:
   chúng ta tách rõ
   fact
   → interpretation
   → decision.

9. Quyết định đủ bằng chứng mới LOCKED.

10. Cập nhật Decision Log rồi sang bước kế tiếp.
```

Đây chính là quy trình CANON đã khóa: trước khi chạy phải xác định câu hỏi và evidence cần có; AI thiết kế phép kiểm tra, người thực hiện chạy trên project thật, rồi AI kiểm tra output trước khi diễn giải. :chatgpt-content-reference{index="17"} :chatgpt-content-reference{index="18"}

---

# 12. Một ví dụ cụ thể để bạn hình dung cách chúng ta làm M4

Giả sử đến câu hỏi:

> “M4 nên xử lý `Merchant State` missing thế nào?”

Tôi sẽ **không** trả lời ngay:

> “Dùng SimpleImputer(strategy='most_frequent')”.

Thay vào đó:

```text
Evidence M2:
Online Transaction → State/Zip missing có cấu trúc.

Câu hỏi M4:
Representation nào bảo toàn semantic đó
mà vẫn tạo được feature cho model?

Candidate:
A. drop feature
B. special category
C. derived online/location-available indicators
D. cách representation khác

Cần kiểm tra:
category counts
missing mechanism
cardinality
train → validation unseen categories
temporal stability
memory cost
pipeline compatibility

FINAL TEST:
không sử dụng.
```

Tôi viết notebook kiểm tra.

Bạn chạy.

Bạn gửi output.

Sau đó chúng ta mới quyết định.

Tương tự với Amount, MCC, behavioral history, encoding hay historical lookback.

Đó là khác biệt giữa “làm preprocessing theo công thức học thuộc” và cách project hiện tại đang làm: **quyết định preprocessing từ evidence**.

---

# 13. Khối lượng công việc thực tế của hai bên

Phần nặng của bạn chủ yếu là **chạy**, không phải tự nghĩ toàn bộ ML methodology.

Bạn sẽ cần:

```text
giữ repository
+
đặt notebook/code đúng vị trí
+
Run All
+
quan sát lỗi
+
gửi output cho tôi
+
cùng tôi duyệt decision quan trọng
```

Phần nặng của tôi sẽ là:

```text
đọc toàn bộ context đã khóa
+
thiết kế M4 plan
+
viết notebook/cell
+
giải thích từng bước cho bạn
+
thiết kế assertion/gate
+
đọc output
+
phát hiện leakage/contradiction
+
viết nhận xét/kết luận sau output
+
duy trì Decision Log
+
chuẩn bị handoff M5
```

Điều này phù hợp với CANON phối hợp hiện tại: các quyết định phụ thuộc data phải có output thật, còn model-dependent decision thì phải chờ model experiment thật. :chatgpt-content-reference{index="19"}

---

# 14. Toàn cảnh từ chỗ chúng ta đang đứng

Tôi muốn bạn hình dung chuỗi sắp tới như sau:

```text
M1
Chọn + audit dataset
        ↓
DONE

M2
Hiểu dataset bằng EDA
        ↓
DONE

M3
Khóa luật thí nghiệm
        ↓
DONE

────────────────────────────────

M4  ← CHÚNG TA SẮP LÀM
Raw transaction
→ representation
→ preprocessing
→ causal feature engineering
→ baseline-ready matrix
→ leakage audit

        ↓

M5
Baseline model
        ↓

Training-window experiment
        ↓

Feature/preprocessing experiments
        ↓

Model experiments
        ↓

Imbalance
        ↓

Temporal CV / tuning
        ↓

Threshold
        ↓

Freeze
        ↓

FINAL TEST
```

Vì vậy, **đích đến gần nhất của chúng ta không phải một con số F1**.

Đích đến gần nhất là:

> **Có một pipeline dữ liệu mà cả bạn và tôi đều giải thích được: transaction đi vào thế nào, feature được tạo từ đâu, state nào học từ TRAIN, tại sao không leakage, và làm sao tạo được X/y sẵn sàng cho baseline.**

