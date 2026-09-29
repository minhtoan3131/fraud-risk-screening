# Candidate A — IBM Synthetic Credit Card Transactions
## M1.6 — Risk Audit

**Candidate ID:** `A-IBM-TABFORMER`  
**Dataset:** IBM Synthetic Credit Card Transactions / TabFormer  
**Status:** `PASS WITH CONDITIONS`

---

## 1. Mục tiêu của M1.6

M1.6 — Risk Audit không còn hỏi:

```text
Dataset có đọc được không?
Schema có đúng không?
```

những câu hỏi này đã được xử lý ở M1.5.

Mục tiêu của M1.6 là tìm những rủi ro có thể làm cho quá trình Machine Learning cho kết quả sai lệch, quá lạc quan hoặc khó bảo vệ về mặt học thuật.

Các nhóm rủi ro được kiểm tra:

```text
- target leakage;
- temporal leakage;
- class imbalance;
- identifier leakage / memorization;
- duplicate;
- synthetic bias;
- anonymization;
- label provenance;
- structural missingness;
- negative Amount;
- temporal distribution shift;
- dataset size;
- documentation gaps;
- khả năng đánh giá cold-start.
```

Trong đó:

```text
Leakage
→ model vô tình sử dụng thông tin mà tại thời điểm dự đoán
  thực tế nó chưa được phép biết.

Memorization
→ model ghi nhớ một User/Card/Merchant cụ thể
  thay vì học quy luật tổng quát.

Synthetic bias
→ sai lệch xuất phát từ việc dữ liệu được sinh bằng mô phỏng
  thay vì quan sát trực tiếp từ hệ thống giao dịch thật.

Cold-start
→ trường hợp User/Card mới hoàn toàn,
  chưa có lịch sử giao dịch trước đó.
```

---

## 2. Risk Matrix

| Risk | Exists? | Severity | Blocking? | Mitigation / Decision |
|---|---|---|---|---|
| Direct target leakage | Không phát hiện trực tiếp | LOW | NO | `Is Fraud?` chỉ được dùng làm target |
| `Errors?` temporal leakage | Có khả năng | HIGH nếu dùng | NO | Loại `Errors?` khỏi Model V1 |
| Extreme class imbalance | YES | HIGH | NO | Không dùng Accuracy đơn độc; dùng Confusion Matrix, Precision, Recall, F1 |
| Duplicate transaction | YES | LOW | NO | Chỉ có 66 exact duplicate extra rows |
| Excessive anonymization | NO | LOW | NO | Phần lớn feature vẫn có ngữ nghĩa nghiệp vụ |
| Label provenance limitation | PARTIAL | MEDIUM | NO | Synthetic label có nguồn, nhưng exact fraud composition chưa đầy đủ |
| Synthetic bias | YES | HIGH | NO trong phạm vi học tập | Không tuyên bố model đại diện cho production banking |
| Raw User identifier risk | YES | MEDIUM-HIGH | NO | Không dùng raw `User` trực tiếp làm model feature |
| Raw Card identifier risk | YES | HIGH | NO | Không dùng raw `Card` trực tiếp làm model feature |
| Merchant identifier risk | YES | HIGH | NO | Không dùng raw `Merchant Name` trực tiếp làm model feature |
| Temporal leakage | POTENTIAL | HIGH | NO | Split theo thời gian; behavioral feature chỉ dùng quá khứ |
| Temporal distribution shift | YES | MEDIUM-HIGH | NO | Không dùng 2020 làm final fraud test; ưu tiên kiểm tra 2019 |
| Structural location missingness | YES | MEDIUM | NO | Không drop Online Transaction chỉ vì State/Zip missing |
| Negative Amount semantic risk | YES | MEDIUM-HIGH | NO | Không abs/drop trước khi hiểu semantic |
| Dataset size | YES | MEDIUM | NO | Có thể dùng temporal subset vẫn giữ hàng nghìn fraud |
| Documentation gaps | YES | MEDIUM | NO | Ghi rõ limitation |
| Cold-start evaluation weakness | YES | MEDIUM | NO | Evaluation chủ yếu phản ánh existing User/Card |


---

# 3. Extreme Class Imbalance

Toàn bộ dataset:

```text
Total transactions:
24,386,900

Fraud:
29,757

Fraud rate:
0.122020%

Non-fraud:
24,357,143

Non-fraud rate:
99.877980%
```

Nếu sử dụng một classifier cực kỳ đơn giản:

```text
Luôn dự đoán:
NON-FRAUD
```

thì Accuracy đã đạt:

```text
99.877980%
```

### Risk

Điều này chứng minh Accuracy không thể được dùng một mình để đánh giá model.

Một model có Accuracy khoảng 99% vẫn có thể:

```text
không phát hiện được gần như bất kỳ fraud nào.
```

### Severity

```text
HIGH
```

### Blocking?

```text
NO
```

Lý do:

```text
Mặc dù fraud rate rất thấp,
dataset vẫn có 29,757 fraud samples.

Số positive tuyệt đối đủ lớn
để training và evaluation có ý nghĩa.
```

### Required mitigation

Sau này phải ưu tiên:

```text
Confusion Matrix
Precision
Recall
F1-score
```

và có thể xem thêm các metric phù hợp khác nếu cần.

Không được trình bày Accuracy như bằng chứng chính cho chất lượng model.

---

# 4. Fraud Distribution theo thời gian

Fraud rate không ổn định giữa các năm.

Một số ví dụ:

```text
2008:
0.303238%

2010:
0.257171%

2011:
0.003502%

2015:
0.192844%

2016:
0.209430%

2017:
0.014797%

2018:
0.144690%

2019:
0.121060%

2020:
0.000000%
```

Một số năm có biến động rất lớn.

Ví dụ:

```text
2011:
1,570,551 transaction
55 fraud

Fraud rate:
0.003502%
```

trong khi:

```text
2010:
1,491,225 transaction
3,835 fraud

Fraud rate:
0.257171%
```

Tương tự:

```text
2017:
255 fraud

2018:
2,491 fraud
```

mặc dù tổng transaction của hai năm gần tương đương.

### Risk

Điều này cho thấy tồn tại:

```text
temporal distribution shift
```

tức phân bố fraud thay đổi theo thời gian.

Một phần hiện tượng này có thể liên quan đến cách synthetic generator sinh dữ liệu.

### Severity

```text
MEDIUM-HIGH
```

### Blocking?

```text
NO
```

### Decision

Không nên dùng random split đơn giản trên toàn bộ 1991–2020.

Evaluation phải tôn trọng chiều thời gian.

---

# 5. Năm 2020 không phù hợp làm Final Fraud Test

Năm 2020 có:

```text
336,500 transactions
0 fraud
```

Trong khi các năm liền trước thường có khoảng:

```text
~1.7 triệu transactions / năm
```

và hàng trăm đến hàng nghìn fraud.

Do đó 2020 có hai vấn đề:

```text
1. Row count thấp bất thường so với các năm gần trước.

2. Không có positive fraud sample.
```

Nếu dùng 2020 làm test set:

```text
Recall fraud
Precision fraud
F1 fraud
```

không thể được đánh giá một cách có ý nghĩa.

### Decision

```text
2020:
KHÔNG dùng làm final fraud test set.
```

Đây là một guardrail bắt buộc.

---

# 6. 2019 là ứng viên tốt cho Future Holdout

Stress test sử dụng:

```text
PAST:
<= 2018

FUTURE:
2019
```

cho kết quả:

```text
PAST <= 2018

Rows:
22,326,462

Fraud:
27,670

Fraud rate:
0.123934%
```

và:

```text
2019

Rows:
1,723,938

Fraud:
2,087

Fraud rate:
0.121060%
```

Fraud rate giữa hai phía khá gần:

```text
Past:
0.123934%

2019:
0.121060%

Overall:
0.122020%
```

### Nhận xét

2019 có:

```text
- số lượng transaction lớn;
- 2,087 positive fraud;
- fraud rate gần overall dataset;
- nằm sau toàn bộ training history trước đó.
```

Do đó 2019 là một ứng viên tốt cho:

```text
future holdout / final test period
```

Tuy nhiên đây mới là recommendation của M1.6.

Final split sẽ được khóa ở milestone chuẩn bị training/evaluation.

---

# 7. `Errors?` và nguy cơ Temporal Leakage

So sánh transaction không có và có error:

```text
NO_ERROR

Rows:
23,998,469

Fraud:
28,471

Fraud rate:
0.118637%
```

```text
HAS_ERROR

Rows:
388,431

Fraud:
1,286

Fraud rate:
0.331076%
```

Transaction có error có fraud rate cao hơn đáng kể.

Một số error cụ thể:

```text
Bad PIN:
0.512577%

Bad Card Number:
0.788229%

Bad Expiration:
1.119821%

Bad CVV:
2.607076%
```

Một số tổ hợp hiếm còn có rate cao hơn:

```text
Bad CVV + Insufficient Balance:
4.494382%

Bad Expiration + Bad CVV:
4.255319%

Bad CVV + Technical Glitch:
4.761905%
```

### Risk

M1.4 đã xác định ý nghĩa của `Errors?`, nhưng chưa chứng minh được:

```text
error được biết trước hay sau thời điểm fraud screening?
```

Nếu workflow thực tế là:

```text
transaction request
        ↓
fraud screening
        ↓
transaction processing
        ↓
Bad PIN / Bad CVV / Insufficient Balance
```

thì việc sử dụng `Errors?` sẽ cho model nhìn thấy thông tin tương lai.

Đó là:

```text
temporal leakage
```

### Severity

```text
HIGH nếu sử dụng.
```

### Decision

Đối với Model V1:

```text
Errors?
→ EXCLUDE
```

Không sử dụng `Errors?` làm model feature.

Chỉ xem xét lại quyết định nếu sau này có bằng chứng chắc chắn từ nguồn cho thấy trường này tồn tại tại prediction time.

---

# 8. Identifier Leakage / Memorization

## 8.1. User

Kết quả:

```text
Total Users:
2,000

Users with >=1 fraud:
1,343

Users with no fraud:
657

Users where every transaction is fraud:
0

Maximum transaction count:
82,355

Maximum fraud count:
113

Maximum fraud rate
among users with >=100 transactions:
1.766980%
```

Có sự khác biệt giữa các User, nhưng không có User nào mà toàn bộ transaction đều là fraud.

Risk tồn tại nhưng chưa phải nghiêm trọng nhất.

---

## 8.2. User + Card

Kết quả:

```text
Total User+Card:
6,139

Cards with >=1 fraud:
2,752

Cards with no fraud:
3,387

Cards where every transaction is fraud:
0

Maximum transaction count:
70,008

Maximum fraud count:
57

Maximum fraud rate
among cards with >=100 transactions:
15.267176%
```

Một số Card có fraud rate rất cao so với overall baseline 0.122%.

Nếu raw Card identity được đưa trực tiếp vào model, model có thể học:

```text
Card X
→ risk cao
```

thay vì học transaction pattern.

---

## 8.3. Merchant

Đây là red flag mạnh nhất.

Kết quả:

```text
Total merchants:
100,343

Merchants with >=1 fraud:
2,831

Merchants with no fraud:
97,512

Merchants where EVERY transaction is fraud:
731
```

Ngoài ra:

```text
Maximum transaction count:
1,130,230

Maximum fraud count:
1,607

Maximum fraud rate among merchants
with >=100 transactions:
100%
```

### Risk

Nếu đưa raw `Merchant Name` vào model, classifier có thể chỉ học:

```text
Merchant ID X
→ fraud

Merchant ID Y
→ non-fraud
```

Điều này tạo nguy cơ:

```text
memorization
```

rất mạnh.

Model có thể đạt metric tốt nhưng chủ yếu vì ghi nhớ merchant của synthetic generator.

### Severity

```text
HIGH
```

### Decision

Các trường sau:

```text
User
Card
Merchant Name
```

không được dùng trực tiếp như model features trong Model V1.

Chúng chỉ được sử dụng như:

```text
group key
history key
feature-engineering key
```

Ví dụ hợp lệ:

```text
User + Card
→ tìm lịch sử giao dịch trước đó

Merchant Name
→ kiểm tra merchant đã từng xuất hiện trước đây chưa
```

để tạo:

```text
is_new_merchant
transactions_with_merchant_before
time_since_previous_transaction
transaction_count_last_10m
amount_vs_history
...
```

Nguyên tắc:

```text
RAW ID
→ không đưa trực tiếp vào classifier

RAW ID
→ được phép dùng để tạo historical feature hợp lệ
```

---

# 9. Temporal Leakage và Behavioral Feature

Dataset có lịch sử dài 1991–2020 và nhiều giao dịch lặp lại của cùng User/Card.

Điều này là một điểm mạnh để tạo behavioral feature nhưng đồng thời tạo nguy cơ temporal leakage.

### Ví dụ sai

```text
Card A có 100 transaction trong toàn dataset.

Tính:
average_amount = mean của cả 100 transaction.

Sau đó gắn average_amount này
cho transaction đầu tiên của Card A.
```

Khi đó transaction đầu tiên đã được sử dụng thông tin từ tương lai.

### Quy tắc bắt buộc

```text
RULE-M1.6-TEMPORAL-01

Đối với transaction T:

mọi historical / behavioral feature
chỉ được tính bằng transaction xảy ra TRƯỚC T.
```

Ví dụ:

```text
time_since_previous_transaction

transactions_last_10m

transactions_last_1h

total_amount_last_1h

average_amount_previous_30d

amount_vs_previous_history

is_new_merchant

is_new_mcc

is_new_location
```

đều phải tuân thủ:

```text
timestamp(history) < timestamp(current transaction)
```

Không được sử dụng transaction hiện tại hoặc tương lai để tạo feature cho transaction hiện tại.

---

# 10. Entity Overlap giữa Past và Future

Stress test:

```text
Past:
<=2018

Future:
2019
```

cho thấy:

```text
2019 users:
1,579

Đã từng xuất hiện trong past:
1,528
```

```text
2019 cards:
4,029

Đã từng xuất hiện trong past:
3,883
```

```text
2019 merchants:
34,936

Đã từng xuất hiện trong past:
32,216
```

Tỷ lệ transaction thuộc entity mới:

```text
NEW USER:
3,255 transactions
0.188812%

NEW CARD:
18,544 transactions
1.075677%

NEW MERCHANT:
4,905 transactions
0.284523%
```

### Nhận xét

Entity overlap cao không tự động là leakage.

Trong hệ thống fraud thực tế, việc:

```text
đánh giá transaction mới
của card đã có lịch sử
```

là hoàn toàn hợp lý.

Thậm chí đó là điều cần thiết để xây behavioral features.

Leakage chỉ xảy ra nếu:

```text
future information
→ được sử dụng để tính feature cho past/current transaction.
```

---

# 11. Cold-start Evaluation Limitation

Do transaction năm 2019 chủ yếu thuộc User/Card đã từng xuất hiện:

```text
NEW USER transaction:
0.188812%

NEW CARD transaction:
1.075677%
```

nên final evaluation sẽ chủ yếu đo:

> khả năng đánh giá giao dịch mới của những khách hàng/thẻ đã có lịch sử.

Nó không kiểm tra mạnh trường hợp:

```text
User hoàn toàn mới
Card hoàn toàn mới
không có historical context
```

### Risk

```text
Cold-start evaluation weakness:
YES
```

### Severity

```text
MEDIUM
```

### Blocking?

```text
NO
```

### Limitation

Báo cáo sau này phải nói rõ:

> Model V1 được đánh giá chủ yếu trong bối cảnh existing User/Card; khả năng tổng quát hóa cho hoàn toàn new User/Card chưa được kiểm chứng mạnh.

---

# 12. Structural Missingness của Merchant Location

M1.5 phát hiện:

```text
Merchant State missing:
~11.16%

Zip missing:
~11.80%
```

M1.6 cho thấy missing không phải ngẫu nhiên.

## Theo transaction mode

```text
Chip Transaction

State missing:
0.120889%

Zip missing:
0.940295%
```

```text
Swipe Transaction

State missing:
0%

Zip missing:
0.687589%
```

```text
Online Transaction

State missing:
100%

Zip missing:
100%
```

Điều này cho thấy phần lớn missing location có nguyên nhân nghiệp vụ:

```text
Online Transaction
→ không có physical merchant location
→ State / Zip missing
```

Đây là:

```text
structural missingness
```

---

## 12.1. Liên hệ với Fraud

Fraud rate:

```text
STATE_PRESENT:
0.052654%

STATE_MISSING:
0.674392%
```

```text
ZIP_PRESENT:
0.022805%

ZIP_MISSING:
0.863476%
```

Không nên kết luận:

```text
Missing location
→ gây fraud.
```

Vì missing location có quan hệ rất mạnh với:

```text
Online Transaction.
```

Do đó missing đang chứa một phần thông tin về transaction mode.

### Severity

```text
MEDIUM
```

### Decision

Không được:

```text
drop tất cả row thiếu Merchant State / Zip
```

vì điều đó gần như sẽ loại toàn bộ Online Transactions.

Cũng không được xem:

```text
location missing
```

là một fraud signal độc lập nếu chưa kiểm soát ảnh hưởng của transaction mode.

---

# 13. Negative Amount Risk

M1.5 phát hiện:

```text
Negative Amount:
1,244,683 rows
≈ 5.10%
```

M1.6 kiểm tra target cho thấy:

```text
NOT_NEGATIVE

Rows:
23,142,217

Fraud:
28,632

Fraud rate:
0.123722%
```

```text
NEGATIVE

Rows:
1,244,683

Fraud:
1,125

Fraud rate:
0.090384%
```

Nhìn toàn bộ dataset, negative Amount không phải fraud signal mạnh.

Zero Amount:

```text
Rows:
20,213

Fraud:
13

Fraud rate:
0.064315%
```

---

## 13.1. Negative Amount theo transaction mode

Kết quả:

```text
Chip Transaction

Negative rows:
342,120

Fraud:
83

Fraud rate:
0.024260%
```

```text
Swipe Transaction

Negative rows:
889,064

Fraud:
146

Fraud rate:
0.016422%
```

```text
Online Transaction

Negative rows:
13,499

Fraud:
896

Fraud rate:
6.637529%
```

`Negative + Online` có fraud rate rất cao so với baseline:

```text
Overall:
0.122020%

Negative Online:
6.637529%
```

Đây là một pattern rất mạnh và có thể phản ánh logic của synthetic generator.

### Semantic problem

Hiện chưa có đủ bằng chứng để khóa negative Amount là:

```text
refund
reversal
credit
hay convention khác của generator.
```

### Severity

```text
MEDIUM-HIGH
```

### Decision

Không được tự động:

```text
abs(Amount)

drop Amount < 0

clip Amount về 0
```

cho tới khi semantic được giải thích hoặc có quyết định preprocessing có lý do rõ ràng.

Nếu không tìm được documentation bổ sung, negative Amount phải được ghi thành limitation của dataset.

---

# 14. Duplicate Risk

Kết quả:

```text
Exact duplicate extra rows:
66

Rows belonging to duplicate groups:
132
```

Risk Audit kiểm tra thêm:

```text
Duplicate member rows:
132

Fraud:
0

Non-fraud:
132

Largest identical-row group:
2
```

Tất cả exact duplicate member đều là non-fraud.

Không có nhóm nào xuất hiện hơn hai lần.

### Severity

```text
LOW
```

### Blocking?

```text
NO
```

### Decision

Duplicate không phải vấn đề đáng kể đối với Candidate A.

Do dataset không có transaction ID duy nhất, exact duplicate cũng chưa đủ để khẳng định chắc chắn rằng đây là cùng một transaction được lưu hai lần.

Chưa cần ưu tiên xử lý vấn đề này.

---

# 15. Dataset Size / Feasibility

Full dataset:

```text
24,386,900 rows
```

là khá lớn đối với project sinh viên.

Tuy nhiên temporal-window audit cho thấy có thể giảm phạm vi mà vẫn giữ đủ fraud samples.

## 2015–2019

```text
Rows:
8,579,208

Fraud:
11,693

Fraud rate:
0.136295%

Users:
1,621

User+Card:
4,435
```

## 2017–2019

```text
Rows:
5,168,913

Fraud:
4,833

Fraud rate:
0.093501%

Users:
1,598

User+Card:
4,236
```

## 2018–2019

```text
Rows:
3,445,553

Fraud:
4,578

Fraud rate:
0.132867%

Users:
1,587

User+Card:
4,152
```

## 2019

```text
Rows:
1,723,938

Fraud:
2,087

Fraud rate:
0.121060%

Users:
1,579

User+Card:
4,029
```

### Nhận xét

Không bắt buộc sử dụng toàn bộ 24.4 triệu row để có đủ positive samples.

Ngay cả:

```text
2018–2019
```

vẫn có:

```text
3.45 triệu transactions
4,578 fraud
```

là quy mô lớn đối với bài tập môn học.

### Severity

```text
MEDIUM
```

### Blocking?

```text
NO
```

### Decision

Việc sử dụng:

```text
full dataset
hay
temporal subset
```

chưa được khóa tại M1.6.

Tuy nhiên M1.6 xác nhận rằng dataset size là vấn đề quản lý được.

---

# 16. Excessive Anonymization Risk

Candidate A không sử dụng schema dạng:

```text
V1
V2
V3
...
V28
```

Phần lớn các feature vẫn có ý nghĩa nghiệp vụ:

```text
Amount
Time
Use Chip
Merchant City
Merchant State
Zip
MCC
```

`User`, `Card` và `Merchant Name` là identifier nhưng vẫn có vai trò rõ trong historical processing.

### Risk

```text
Excessive anonymization:
NO
```

### Severity

```text
LOW
```

Đây vẫn là một điểm mạnh đáng kể của Candidate A.

---

# 17. Label Provenance Risk

Target:

```text
Is Fraud?
```

là synthetic label do generator tạo.

Source Audit đã xác định được ở mức tổng quát cách fraud được tạo trong synthetic environment.

Do đó đây không phải trường hợp:

```text
không biết label đến từ đâu
hoặc
không biết fraud có nghĩa gì.
```

Tuy nhiên documentation chưa cung cấp đầy đủ:

```text
tỷ lệ chính xác của từng fraud-generation mechanism
trong release hiện tại.
```

### Risk

```text
Label provenance limitation:
YES, nhưng chỉ ở mức chi tiết.
```

### Severity

```text
MEDIUM
```

### Blocking?

```text
NO
```

---

# 18. Synthetic Bias

Candidate A là:

```text
fully synthetic dataset
```

chứ không phải transaction log của một ngân hàng thật.

M1.6 phát hiện một số pattern đáng chú ý:

```text
- fraud rate biến động mạnh theo năm;

- một số Card có fraud rate rất cao;

- 731 Merchant có toàn bộ transaction là fraud;

- có Merchant >=100 transactions nhưng fraud rate 100%;

- Negative Online Transaction có fraud rate ~6.64%;

- 2020 có 0 fraud.
```

Các pattern này không chứng minh dataset sai.

Nhưng chúng củng cố rủi ro rằng model có thể học:

```text
logic của synthetic world
```

thay vì:

```text
fraud distribution của ngân hàng thật.
```

### Severity

```text
HIGH
```

### Blocking?

Trong phạm vi bài tập môn học:

```text
NO
```

Nếu định tuyên bố đây là production banking model:

```text
YES
```

nhưng project hiện tại không có mục tiêu đó.

### Required limitation

Không được tuyên bố:

```text
Model đã học fraud behavior của ngân hàng thật.

Model có thể triển khai trực tiếp vào production banking.

Metric trên synthetic dataset đại diện cho hiệu quả ngoài thực tế.
```

Cách diễn đạt đúng:

> Model được phát triển và đánh giá trên synthetic credit-card transaction data; kết quả thể hiện khả năng học tín hiệu rủi ro trong phạm vi dataset này, không chứng minh hiệu quả trên giao dịch ngân hàng thực tế.

---

# 19. Documentation Risk

Documentation tổng thể của Candidate A khá tốt:

```text
- có nguồn IBM;
- có paper;
- có repository;
- có source code;
- có generator methodology;
- raw artifact đã được xác minh.
```

Tuy nhiên vẫn còn một số semantic gap:

```text
1. Exact composition của các fraud-generation mechanisms
   chưa được tài liệu hóa đầy đủ.

2. Prediction-time availability của Errors?
   chưa được chứng minh chắc chắn.

3. Semantic của negative Amount
   chưa được khóa.

4. Một số metadata giữa các tài liệu từng bất nhất
   về 2,000 vs 20,000 users,
   dù local artifact đã giải quyết là 2,000.
```

### Severity

```text
MEDIUM
```

### Blocking?

```text
NO
```

---

# 20. Mandatory Guardrails

Nếu Candidate A được chọn làm dataset chính thức, các guardrail sau là bắt buộc.

## GUARDRAIL-01 — Không dùng raw identifiers trực tiếp

```text
Không dùng trực tiếp:

User
Card
Merchant Name

làm classifier features.
```

Được phép sử dụng chúng làm key để tạo historical features.

---

## GUARDRAIL-02 — Loại `Errors?` khỏi Model V1

```text
Errors?
→ không sử dụng làm ML feature
```

trừ khi sau này tìm được bằng chứng chắc chắn rằng error tồn tại tại prediction time.

---

## GUARDRAIL-03 — Không dùng random split đơn giản

Evaluation phải tôn trọng chiều thời gian.

Ưu tiên:

```text
past
→ train / validation

future
→ test
```

---

## GUARDRAIL-04 — Historical feature phải causal

Đối với transaction T:

```text
feature(T)
```

chỉ được sử dụng dữ liệu xảy ra:

```text
TRƯỚC T
```

Không được dùng transaction hiện tại hoặc tương lai.

---

## GUARDRAIL-05 — Preprocessing chỉ học trên TRAIN

Ví dụ:

```text
encoder
scaler
imputer
statistics
threshold tuning
```

không được fit bằng final test set.

---

## GUARDRAIL-06 — Không resample trước split

Nếu sau này sử dụng:

```text
undersampling
oversampling
SMOTE
```

thì chỉ thực hiện trên training data.

Không resample final test.

---

## GUARDRAIL-07 — Không dùng 2020 làm final fraud test

```text
2020:
0 fraud
```

nên không phù hợp để đánh giá detection performance.

---

## GUARDRAIL-08 — Không drop missing location một cách máy móc

Online Transaction có:

```text
Merchant State missing = 100%
Zip missing = 100%
```

Do đó drop missing sẽ làm mất gần như toàn bộ online transaction.

---

## GUARDRAIL-09 — Không tự động sửa Negative Amount

Không được:

```text
abs()
drop
clip
```

negative Amount trước khi có quyết định semantic rõ ràng.

---

## GUARDRAIL-10 — Không overclaim tính thực tế

Dataset là synthetic.

Kết quả model không được trình bày như bằng chứng hiệu quả trên production banking data.

---

# 21. Các Risk được xem là Manageable

Sau audit, các risk sau được đánh giá là có thể kiểm soát:

```text
Extreme class imbalance
→ xử lý bằng metric, training strategy và threshold phù hợp.

Identifier leakage
→ loại raw IDs khỏi classifier.

Errors? leakage
→ loại khỏi Model V1.

Temporal leakage
→ temporal split + causal feature engineering.

Dataset size
→ temporal subset nếu cần.

Structural missingness
→ xử lý theo semantic transaction mode.

Duplicate
→ tỷ lệ cực nhỏ.

Cold-start limitation
→ ghi rõ phạm vi evaluation.

Synthetic bias
→ giới hạn claim và ghi limitation.
```

---

# 22. Blocking Risk Assessment

Không phát hiện risk nào bắt buộc phải loại Candidate A khỏi project.

```text
Blocking risk:
NONE IDENTIFIED
```

Một số risk có severity HIGH, nhưng đều có mitigation rõ trong phạm vi project:

```text
Errors?
→ exclude.

Identifiers
→ không dùng trực tiếp.

Temporal leakage
→ temporal split + causal features.

Class imbalance
→ metric và training procedure phù hợp.

Synthetic bias
→ không thể loại bỏ,
  nhưng có thể giới hạn claim của project.
```

---

# 23. M1.6 Decision

```text
Decision ID:
M1.6-A01

Candidate:
A-IBM-TABFORMER

Dataset:
IBM Synthetic Credit Card Transactions

Status:
PASS WITH CONDITIONS
```

### Lý do

```text
Candidate A tồn tại một số rủi ro Machine Learning đáng kể:

- extreme class imbalance;
- synthetic bias;
- identifier memorization;
- prediction-time ambiguity của Errors?;
- temporal distribution shift;
- structural location missingness;
- negative Amount semantic gap;
- cold-start evaluation limitation.

Tuy nhiên không có rủi ro nào bắt buộc phải loại dataset.

Các risk nghiêm trọng nhất có mitigation rõ:

- raw User / Card / Merchant Name không dùng trực tiếp;
- Errors? bị loại khỏi Model V1;
- sử dụng temporal split;
- behavioral feature chỉ sử dụng lịch sử quá khứ;
- không dùng Accuracy đơn độc;
- không sử dụng 2020 làm final fraud test;
- không xóa location missing một cách máy móc;
- không tự động sửa negative Amount;
- giới hạn claim vì dataset synthetic.

Dataset vẫn có:

- transaction-level grain rõ;
- feature có ngữ nghĩa nghiệp vụ;
- timestamp chi tiết;
- lịch sử User/Card tốt;
- merchant và MCC;
- hàng chục nghìn fraud samples trên full dataset;
- hàng nghìn fraud samples ngay cả trong temporal subset nhỏ hơn;
- khả năng xây live demo và behavioral features.

Không phát hiện blocking risk.
```

---

# 24. Kết luận M1.6

Candidate A vượt qua Risk Audit với điều kiện bắt buộc.

```text
M1.3 — Source Audit
PASS WITH OPEN QUESTIONS

        ↓

M1.4 — Semantic Audit
PASS

        ↓

M1.5 — Technical Audit
PASS WITH FINDINGS

        ↓

M1.6 — Risk Audit
PASS WITH CONDITIONS
```

Sau M1.6, Candidate A được xem là phù hợp để tiếp tục project nếu toàn bộ guardrail của M1.6 được tuân thủ.

Những phát hiện của M1.6 không làm Candidate A trở thành dataset “không tốt”.

Ngược lại, chúng xác định rõ:

```text
feature nào không nên dùng;

split nào không nên dùng;

những leakage nào phải tránh;

metric nào không nên tin một mình;

những limitation nào phải ghi trong báo cáo;

và phạm vi claim hợp lý của model.
```
