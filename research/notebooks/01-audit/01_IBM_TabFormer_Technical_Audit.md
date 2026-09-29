# Candidate A — IBM Synthetic Credit Card Transactions
## M1.5 — Technical Audit

**Candidate ID:** `A-IBM-TABFORMER`  
**Dataset:** IBM Synthetic Credit Card Transactions / TabFormer  
**Status:** `PASS WITH FINDINGS`

---

## 1. Mục tiêu của M1.5

Mục tiêu của bước Technical Audit là kiểm tra xem file dữ liệu thực tế được sử dụng trong project có đúng với những gì Source Audit và Semantic Audit đã xác định hay không.

Cụ thể, bước này xác minh:

```text
- artifact thực tế được sử dụng;
- kích thước và SHA-256 của file;
- số dòng và số cột;
- raw schema;
- kiểu dữ liệu;
- missing values;
- cardinality;
- target values và class distribution;
- phạm vi thời gian;
- các giá trị kỹ thuật bất thường;
- exact duplicate rows.
```

M1.5 không thực hiện:

```text
- làm sạch dữ liệu;
- xóa duplicate;
- fill missing;
- chọn feature cuối cùng;
- encode / scale;
- tạo behavioral features;
- chia train/test;
- xử lý class imbalance;
- train model;
- phân tích feature nào dự đoán fraud mạnh.
```

---

## 2. Artifact Record

```text
Primary source:
IBM TabFormer repository / IBM Box

Archive:
transactions.tgz

Archive size:
278,576,638 bytes
265.67 MiB

Archive SHA-256:
e9f589a0958f40d60f81b1a2e8428db86e00c05755caf44fb055827976c0efa2


Extracted CSV:
card_transaction.v1.csv

CSV size:
2,354,626,737 bytes
2,245.55 MiB

CSV SHA-256:
68c438319cf27614d5564b7b520814036f0d7e087b3a17d8c15081848e0f02de


Audit execution date:
2026-09-14

Python:
3.14.6

Pandas:
3.0.5

NumPy:
2.5.3
```

SHA-256 được sử dụng như fingerprint, tức dấu vân tay của artifact mà project thực tế sử dụng.

Không sử dụng hash từ nguồn bên ngoài thay cho hash tính trực tiếp trên máy local.

---

## 3. Raw Schema Verification

Raw CSV thực tế có:

```text
Rows:
24,386,900

Columns:
15
```

Danh sách cột:

```text
User
Card
Year
Month
Day
Time
Amount
Use Chip
Merchant Name
Merchant City
Merchant State
Zip
MCC
Errors?
Is Fraud?
```

Kết quả kiểm tra schema:

```text
Expected column count:
15

Actual column count:
15

Missing columns:
[]

Unexpected columns:
[]

Column order matches:
True

Duplicate column names:
0
```

### Kết luận

```text
Raw schema:
PASS
```

File local khớp hoàn toàn với raw schema 15 cột đã xác định ở M1.4.

Con số 12 fields được repository TabFormer đề cập thuộc representation sau preprocessing, trong đó Year / Month / Day / Time được hợp nhất thành Timestamp.

---

## 4. Technical Verification Summary

```markdown
| Check | Expected / Published | Actual local artifact | Status |
|---|---|---|---|
| Row grain | 1 row ≈ 1 credit-card transaction | Phù hợp với raw schema | PASS |
| Raw row count | 24,386,900 theo tài liệu IBM đã audit | 24,386,900 | PASS |
| Raw column count | 15 | 15 | PASS |
| Required columns | 15 raw columns đã xác định | Đầy đủ | PASS |
| Unexpected columns | Không kỳ vọng | 0 | PASS |
| Duplicate column names | 0 | 0 | PASS |
| Target | Is Fraud? | Is Fraud? | PASS |
| Target values | Yes / No | Yes / No | PASS |
| Missing target | Không kỳ vọng | 0 | PASS |
| Unique Users | Documentation có bất nhất | 2,000 | VERIFIED |
| Unique User+Card pairs | Chưa khóa trước audit | 6,139 | VERIFIED |
| Use Chip categories | Swipe / Chip / Online | Đúng 3 loại | PASS |
| Year range | Multi-year / multi-decade | 1991–2020 | VERIFIED |
| Amount parse failures | Kỳ vọng 0 | 0 | PASS |
| Invalid Month | Kỳ vọng 0 | 0 | PASS |
| Invalid Day cơ bản | Kỳ vọng 0 | 0 | PASS |
| Invalid Time | Kỳ vọng 0 | 0 | PASS |
| Exact duplicate extra rows | Chưa khóa trước audit | 66 | REVIEW |
```

---

## 5. Kiểu dữ liệu đọc bởi Pandas

Kết quả trên chunk 500.000 dòng:

```text
User                int64
Card                int64
Year                int64
Month               int64
Day                 int64
Time                  str
Amount                str
Use Chip              str
Merchant Name       int64
Merchant City         str
Merchant State        str
Zip               float64
MCC                 int64
Errors?               str
Is Fraud?             str
```

Nhận xét:

```text
Amount
→ được đọc dạng string vì có ký hiệu "$".

Time
→ được đọc dạng string với định dạng HH:MM.

Zip
→ được đọc dạng float64 vì tồn tại missing values.

User / Card / Merchant Name / MCC
→ đọc được dưới dạng số,
  nhưng điều đó không có nghĩa chúng là numerical features
  có ý nghĩa về độ lớn.
```

Không phát hiện lỗi parser hoặc schema corruption.

---

## 6. Missing Values

Kết quả toàn bộ dataset:

```text
User                     0
Card                     0
Year                     0
Month                    0
Day                      0
Time                     0
Amount                   0
Use Chip                 0
Merchant Name            0
Merchant City            0
Merchant State     2,720,821
Zip                2,878,135
MCC                      0
Errors?          23,998,469
Is Fraud?                0
```

Tổng missing cells:

```text
29,597,425
```

Tỷ lệ xấp xỉ:

```text
Merchant State:
2,720,821
≈ 11.16%

Zip:
2,878,135
≈ 11.80%

Errors?:
23,998,469
≈ 98.41%
```

### Nhận xét

Missing không phân bố trên toàn dataset mà tập trung ở ba cột:

```text
Merchant State
Zip
Errors?
```

Các trường cốt lõi:

```text
User
Card
Year
Month
Day
Time
Amount
Use Chip
Merchant Name
MCC
Is Fraud?
```

không có missing.

`Errors?` có tỷ lệ missing rất cao nhưng chưa được coi là lỗi dữ liệu. Trong code TabFormer, missing của trường này được xử lý theo hướng có thể biểu diễn trường hợp không có error.

M1.5 chưa fill missing và chưa quyết định cách preprocessing.

---

## 7. Cardinality

Kết quả:

```text
Unique Users:
2,000

Unique User+Card pairs:
6,139

Unique Merchant Name:
100,343

Unique Merchant City:
13,429

Unique Merchant State:
223

Unique Zip:
27,321

Unique MCC:
109

Unique Time values:
1,440
```

### Nhận xét

Kết quả `Unique Users = 2,000` giải quyết câu hỏi mở từ M1.3 về bất nhất giữa 2.000 và 20.000 users trong các nguồn tài liệu.

Đối với artifact mà project sử dụng:

```text
2,000 User
6,139 User+Card pairs
```

được xem là số liệu chính thức đã kiểm chứng.

`1,440` giá trị Time tương ứng với:

```text
24 giờ × 60 phút = 1,440 phút
```

cho thấy dữ liệu có độ phân giải thời gian tới từng phút.

Cấu trúc User + Card + Time hỗ trợ tốt cho việc xây behavioral features ở các milestone sau.

---

## 8. Temporal Range

Phạm vi thời gian:

```text
Minimum Year:
1991

Maximum Year:
2020

Month range:
1–12

Day range:
1–31

Invalid Month:
0

Invalid Day theo kiểm tra cơ bản 1..31:
0

Invalid Time:
0
```

Số dòng theo năm:

```text
1991      1,585
1992      5,134
1993      8,378
1994     14,316
1995     20,928
1996     29,945
1997     49,753
1998     78,345
1999    118,250
2000    177,729
2001    257,998
2002    350,732
2003    466,408
2004    597,003
2005    746,653
2006    908,793
2007  1,064,483
2008  1,223,460
2009  1,355,434
2010  1,491,225
2011  1,570,551
2012  1,610,829
2013  1,650,917
2014  1,672,343
2015  1,701,371
2016  1,708,924
2017  1,723,360
2018  1,721,615
2019  1,723,938
2020    336,500
```

### Nhận xét

Dataset có cấu trúc thời gian dài và số transaction tăng mạnh qua các năm.

Đây chưa được xem là vấn đề ở M1.5.

Tuy nhiên cấu trúc temporal này sẽ rất quan trọng trong M1.6 khi đánh giá:

```text
- temporal leakage;
- cách chia train/test;
- khả năng chọn một temporal window nếu cần giảm quy mô dataset.
```

M1.5 chưa chọn subset.

---

## 9. Target Verification

Target:

```text
Is Fraud?
```

Giá trị thực tế:

```text
No
Yes
```

Không có:

```text
missing
class thứ ba
giá trị lạ
```

Phân bố:

```text
No:
24,357,143
99.877980%

Yes:
29,757
0.122020%
```

Fraud ratio:

```text
29,757 / 24,386,900
≈ 0.122020%
```

Tương đương gần:

```text
1 fraud
trên khoảng
820 transaction
```

### Nhận xét

Target schema:

```text
PASS
```

Tuy nhiên class imbalance rất mạnh.

Đây không phải lỗi kỹ thuật của dataset nên Candidate A không FAIL M1.5.

Rủi ro do imbalance sẽ được đánh giá chính thức tại M1.6.

Điểm đáng chú ý là mặc dù tỷ lệ fraud rất thấp, dataset vẫn có:

```text
29,757 positive samples
```

nên positive class không nhỏ về số lượng tuyệt đối.

---

## 10. `Use Chip` Verification

Có đúng ba giá trị:

```text
Swipe Transaction:
15,386,082

Chip Transaction:
6,287,598

Online Transaction:
2,713,220
```

Số giá trị duy nhất:

```text
3
```

Không có missing.

### Kết luận

```text
Use Chip semantics:
PASS
```

Kết quả khớp với Semantic Audit M1.4.

Tên cột `Use Chip` không nên hiểu đơn giản là Yes / No mà thực tế biểu diễn hình thức giao dịch:

```text
Swipe
Chip
Online
```

---

## 11. `Errors?` Verification

Các giá trị chính:

```text
<MISSING>                              23,998,469
Insufficient Balance,                    242,783
Bad PIN,                                  58,918
Technical Glitch,                         48,157
Bad Card Number,                          13,321
Bad CVV,                                  10,740
Bad Expiration,                           10,716
Bad Zipcode,                               2,079
```

Ngoài ra tồn tại một số tổ hợp nhiều error, ví dụ:

```text
Bad PIN,Insufficient Balance,
Insufficient Balance,Technical Glitch,
Bad PIN,Technical Glitch,
Bad Card Number,Insufficient Balance,
Bad CVV,Insufficient Balance,
...
```

Tổng số representation khác nhau, tính cả missing:

```text
24
```

### Nhận xét

Semantic của `Errors?` được xác nhận phù hợp với M1.4.

Tuy nhiên:

```text
Technical validity:
PASS

ML feature validity:
CHƯA KẾT LUẬN
```

Câu hỏi quan trọng vẫn còn:

```text
Errors? có tồn tại tại đúng thời điểm model cần screening không?
```

Nếu error chỉ được biết sau bước mà model lẽ ra phải dự đoán, sử dụng nó có thể tạo temporal leakage.

Vấn đề này được chuyển sang M1.6.

---

## 12. Amount Technical Check

Raw `Amount` có dạng string, ví dụ:

```text
$134.09
$38.48
$120.34
```

Sau khi chỉ loại ký hiệu `$` trong biến tạm để kiểm tra:

```text
Amount parse failures:
0

Negative amounts:
1,244,683

Zero amounts:
20,213

Minimum amount:
-500.0

Maximum amount:
12,390.5
```

Tỷ lệ xấp xỉ:

```text
Negative amounts:
≈ 5.10%

Zero amounts:
≈ 0.083%
```

### Nhận xét

Không có lỗi parse Amount.

Tuy nhiên số lượng negative amount lớn, không thể xem là vài lỗi dữ liệu ngẫu nhiên.

M1.5 chưa có đủ bằng chứng để kết luận negative amount có ý nghĩa nghiệp vụ gì.

Do đó:

```text
Không:
- xóa các row này;
- dùng abs();
- clip về 0;
- coi là corrupted data.
```

Trạng thái:

```text
Negative Amount semantic:
OPEN QUESTION
```

Cần được điều tra tiếp trước preprocessing.

---

## 13. Duplicate Verification

Full dataframe:

```text
Shape:
(24,386,900, 15)
```

Kết quả:

```text
Exact duplicate extra rows:
66

Rows belonging to duplicate groups:
132
```

Tỷ lệ extra duplicate rows:

```text
66 / 24,386,900
≈ 0.00027%
```

### Nhận xét

Exact duplicate rất nhỏ về tỷ lệ và không tạo technical blocking issue.

Tuy nhiên dataset không có transaction ID duy nhất.

Vì vậy:

```text
hai row giống hoàn toàn
```

không đủ để chứng minh chắc chắn:

```text
đó là cùng một transaction bị ghi lặp
```

M1.5 chỉ ghi nhận exact duplicate theo 15 raw columns.

Chưa thực hiện:

```python
df.drop_duplicates()
```

Quyết định giữ hoặc loại duplicate sẽ được thực hiện sau khi xem xét rủi ro và ý nghĩa dữ liệu.

---

## 14. Technical Findings cần chuyển sang M1.6

Sau Technical Audit, các vấn đề đáng quan tâm gồm:

### FINDING-01 — Extreme class imbalance

```text
Fraud:
29,757

Fraud ratio:
0.122020%
```

Rủi ro cần đánh giá:

```text
- Accuracy có thể gây hiểu nhầm;
- cách chia train/test;
- số positive trong từng tập;
- Precision / Recall / F1;
- có cần xử lý imbalance hay không.
```

---

### FINDING-02 — `Errors?` có nguy cơ temporal leakage

```text
Semantic:
đã hiểu.

Prediction-time availability:
chưa rõ.
```

Cần xác định liệu thông tin error có tồn tại trước hoặc tại thời điểm risk screening hay chỉ xuất hiện sau khi giao dịch được xử lý.

---

### FINDING-03 — Identifier leakage / memorization risk

Các trường:

```text
User
Card
Merchant Name
```

có cardinality và tính nhận diện cao.

Chúng rất hữu ích để:

```text
group history
tạo behavioral features
```

nhưng không nên mặc nhiên đưa trực tiếp vào model dưới dạng số.

Cần đánh giá nguy cơ model ghi nhớ entity thay vì học pattern có khả năng generalize.

---

### FINDING-04 — Temporal leakage / split risk

Dataset trải dài:

```text
1991–2020
```

và có nhiều transaction của cùng User/Card theo thời gian.

Random split đơn giản có thể làm:

```text
transaction tương lai
và
transaction quá khứ
```

của cùng entity xuất hiện ở các tập khác nhau theo cách không phản ánh quy trình prediction thực tế.

Cần đánh giá tại M1.6.

---

### FINDING-05 — Negative Amount

```text
1,244,683 negative amounts
≈ 5.10%
```

Không có lỗi parse nhưng semantic chưa được khóa.

Chưa được tự động xử lý như outlier hoặc dữ liệu lỗi.

---

### FINDING-06 — Missing location fields

```text
Merchant State:
≈ 11.16% missing

Zip:
≈ 11.80% missing
```

Không phải blocking issue nhưng phải được xử lý có lý do ở preprocessing.

Chưa kết luận quan hệ giữa missing và loại transaction.

---

### FINDING-07 — Exact duplicates

```text
66 extra rows
≈ 0.00027%
```

Mức rất nhỏ.

Do không có transaction ID duy nhất, chưa đủ bằng chứng để khẳng định đó là duplicated transaction thực tế.

---

### FINDING-08 — Dataset size

```text
24,386,900 rows

CSV:
≈ 2.19 GiB / 2,245.55 MiB
```

Máy local MacBook M1 Pro 16 GB có thể:

```text
- scan toàn bộ bằng chunk;
- load full dataframe;
- chạy exact duplicate check.
```

Như vậy dataset về mặt kỹ thuật có thể xử lý được trên môi trường hiện tại.

Tuy nhiên việc dùng toàn bộ 24 triệu row cho mọi thử nghiệm ML có thể không cần thiết.

Quyết định sử dụng full data hay temporal subset chưa được thực hiện ở M1.5.

---

## 15. Các câu hỏi mở sau M1.5

```text
1. `Errors?` có tồn tại tại prediction time không?

2. Có nên đưa User / Card / Merchant Name trực tiếp vào model
   hay chỉ dùng chúng làm group keys để tạo feature lịch sử?

3. Train/test split nên là random split,
   temporal split,
   hay cần thêm group constraint?

4. Class imbalance 0.122% có cần resampling hay
   class-weight handling không?

5. Negative Amount có ý nghĩa nghiệp vụ gì?

6. Missing Merchant State / Zip có liên quan tới
   Online Transaction hoặc loại merchant/location nào không?

7. Exact duplicate rows có nên giữ hay loại?

8. Có cần dùng toàn bộ 24,386,900 rows,
   hay chọn một temporal window liên tục phù hợp hơn
   với phạm vi bài tập lớn?
```

Các câu hỏi trên chưa được giải quyết ở Technical Audit.

Chúng được chuyển sang Risk Audit hoặc các milestone tiếp theo tùy loại vấn đề.

---

## 16. M1.5 Decision

```text
Decision ID:
M1.5-A01

Candidate:
A-IBM-TABFORMER

Dataset:
IBM Synthetic Credit Card Transactions

Local artifact:
card_transaction.v1.csv

Rows:
24,386,900

Columns:
15

Status:
PASS WITH FINDINGS
```

### Lý do

```text
Artifact local đọc được đầy đủ và khớp với raw schema
đã xác định trong Source Audit và Semantic Audit.

File có đúng 24,386,900 giao dịch và 15 raw columns,
không có cột thiếu hoặc cột thừa.

Target `Is Fraud?` chỉ gồm Yes / No và không có missing.

Dataset thực tế xác nhận:
2,000 User
6,139 User+Card pairs

qua đó giải quyết bất nhất về số lượng User
trong documentation.

Phạm vi thời gian được xác minh là 1991–2020.

Month, Day và Time không có lỗi định dạng cơ bản.

Amount có thể parse hoàn toàn nhưng tồn tại
1,244,683 giá trị âm (~5.10%);
ý nghĩa nghiệp vụ chưa được khóa.

Missing tập trung ở Merchant State, Zip và Errors?,
không ảnh hưởng đến target hoặc nhiều trường cốt lõi.

Có 66 exact duplicate extra rows trên hơn 24 triệu row,
mức rất nhỏ và chưa đủ bằng chứng để coi là
duplicated transaction cần xóa.

Fraud chiếm:
29,757 / 24,386,900
= 0.122020%

cho thấy class imbalance mạnh và phải được
đánh giá chính thức ở M1.6.

Không phát hiện technical blocking issue.
```

---

## 17. Kết luận

Candidate A vượt qua M1.5 — Technical Audit.

```text
M1.3 — Source Audit
PASS WITH OPEN QUESTIONS

        ↓

M1.4 — Semantic Audit
PASS

        ↓

M1.5 — Technical Audit
PASS WITH FINDINGS
```

Không phát hiện dấu hiệu:

```text
- tải nhầm dataset;
- artifact bị thiếu;
- schema không khớp;
- target bị lỗi;
- CSV hỏng;
- cấu trúc kỹ thuật không sử dụng được.
```

Ngược lại, file thực tế nhìn chung nhất quán với những gì Source Audit và Semantic Audit đã xác định.

Các vấn đề còn lại chủ yếu không phải lỗi kỹ thuật mà là rủi ro Machine Learning hoặc vấn đề thiết kế pipeline:

```text
- extreme class imbalance;
- temporal leakage;
- identifier leakage;
- prediction-time validity của Errors?;
- synthetic bias;
- cách chia train/test;
- quy mô dataset;
- cách hiểu negative Amount.
```

