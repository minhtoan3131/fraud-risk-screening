# M3.2 — Xây temporal map và xác định modeling/history window candidate

## 1. Mục tiêu

M3.1 đã khóa nguyên tắc:

`quá khứ → train / validation → tương lai → test`

đồng thời giữ `final train / validation / test boundaries` và `final training window` ở trạng thái OPEN.

M3.2 chưa thực hiện train model và chưa lựa chọn final split.

Mục tiêu của M3.2 là chuyển các finding temporal đã được xác nhận trong Milestone 2 thành một bản đồ thời gian phục vụ thiết kế experiment, bao gồm:

* nhận diện các temporal region có đặc điểm khác nhau;
* xác định vùng nào có thể đóng vai trò historical warm-up;
* xác định vùng nào có thể cung cấp modeling rows;
* xác định vùng có thể dùng làm validation/future-evaluation candidate;
* tách riêng zero-fraud regime;
* chuyển các temporal window của M2.8 thành các training-window family phù hợp với nguyên tắc chống leakage.

Đầu ra M3.2 gồm:

`Temporal regime map`

`Candidate evaluation regions`

`Candidate modeling-window families`

`History warm-up policy v0.1`

`Open questions chuyển sang M3.3`

Kế hoạch M3 ban đầu cũng xác định M3.2 theo đúng phạm vi này và yêu cầu chưa khóa final split nếu bằng chứng chưa đủ.

---

# 2. Bằng chứng temporal được kế thừa từ M2

## 2.1. Không có cơ sở xem toàn bộ lịch sử là một distribution ổn định

M2.3 xác nhận fraud rate thay đổi rất mạnh giữa các năm.

Một số ví dụ:

`2008: 0.303238%`

`2010: 0.257171%`

`2011: 0.003502%`

`2016: 0.209430%`

`2017: 0.014797%`

Riêng `2016 → 2017`, fraud rate giảm khoảng 14 lần.

Do đó không có bằng chứng để xem fraud prevalence là ổn định theo thời gian; đây cũng là lý do random split không phù hợp làm evaluation chính.

`M3.2-F01: toàn bộ lịch sử không phải một temporal regime đồng nhất.`

---

## 2.2. Giai đoạn 2015–2019 cũng không phải một khối đồng nhất

Các thống kê cấp năm gần cuối dataset:

```text
2015
Transaction : 1,701,371
Fraud       : 3,281
Fraud rate  : 0.192844%

2016
Transaction : 1,708,924
Fraud       : 3,579
Fraud rate  : 0.209430%

2017
Transaction : 1,723,360
Fraud       : 255
Fraud rate  : 0.014797%

2018
Transaction : 1,721,615
Fraud       : 2,491
Fraud rate  : 0.144690%

2019
Transaction : 1,723,938
Fraud       : 2,087
Fraud rate  : 0.121060%
```

Điểm quan trọng không phải chỉ là các con số riêng lẻ.

Ta thấy:

`2015–2016`
→ fraud prevalence tương đối cao.

`2017`
→ giảm rất mạnh.

`2018`
→ tăng trở lại.

`2019`
→ nếu nhìn cả năm thì có vẻ gần mức toàn dataset, nhưng cấp năm đang che giấu một điểm gãy bên trong năm.

Do đó:

> `RECENT_2015_2019` có quy mô lớn và đủ positive sample, nhưng không được diễn giải như một regime thống nhất.

`M3.2-F02: temporal heterogeneity tồn tại ngay cả bên trong recent window 2015–2019.`

---

# 3. Điểm gãy quan trọng nhất: 11/2019

M2.3 đã phân tích 2018–2020 ở cấp tháng.

Năm 2018:

* cả 12 tháng đều có fraud;
* monthly fraud rate dao động khoảng `0.099% → 0.212%`.

Năm 2019:

* tháng 1 → tháng 10 đều có fraud;
* tháng 11 và tháng 12 có `0 fraud`.

2020:

* chỉ có tháng 1–2;
* cả hai tháng đều có `0 fraud`.

Cụ thể:

```text
2019-11 : 141,946 transaction | 0 fraud
2019-12 : 146,579 transaction | 0 fraud
2020-01 : 170,731 transaction | 0 fraud
2020-02 : 165,769 transaction | 0 fraud
```

Tổng:

`625,025 transaction`

nhưng:

`0 fraud`

trong bốn tháng liên tiếp.

Toàn bộ `2,087 fraud` của năm 2019 đều nằm trong `01/2019 → 10/2019`.

Giai đoạn này có:

```text
2019-01 → 2019-10

Transaction : 1,435,413
Fraud       : 2,087
Fraud rate  : 0.145394%
```

Trong khi năm 2018:

```text
2018

Transaction : 1,721,615
Fraud       : 2,491
Fraud rate  : 0.144690%
```

Hai fraud rate rất gần nhau, trong khi từ `11/2019` trở đi fraud rate đột ngột bằng 0.

Không có bằng chứng để kết luận nguyên nhân của điểm gãy là lỗi dữ liệu hay cơ chế của synthetic generator.

Vì vậy M3 chỉ được kết luận:

> **Có một temporal regime shift rất mạnh bắt đầu khoảng 11/2019.**

Không được suy diễn nguyên nhân.

`M3.2-F03: 11/2019 được xem là temporal break quan trọng của experiment design.`

---

# 4. Temporal regime map của M3

Từ các bằng chứng trên, M3.2 xây bản đồ làm việc sau:

```text
1991 ───────────── 2014 │ 2015 ─ 2016 │ 2017 │ 2018 │ 2019-01 ───── 2019-10 │ 2019-11 ─ 2019-12 │ 2020-01 ─ 2020-02
                         │             │      │      │                       │                     │
                         │             │      │      │                       │                     │
 legacy history          │ high-rate   │ very │ recent positive-bearing    │ zero-fraud          │ partial-year
 reservoir               │ recent      │ low  │ pre-break region           │ post-break          │ zero-fraud
                         │ prevalence  │ rate │                            │ regime              │ regime
```

Đây là **temporal map phục vụ experiment**, không phải khẳng định rằng mỗi region tương ứng một cơ chế sinh dữ liệu thực tế khác nhau.

Các region được định nghĩa chi tiết như sau.

---

## 4.1. Region H0 — Legacy historical reservoir: 1991–2014

Vai trò ưu tiên:

`historical context / warm-up reservoir`

Không ưu tiên mặc định:

`main modeling rows`

Lý do không phải vì dữ liệu cũ “sai”, mà vì:

* toàn lịch sử có temporal instability mạnh;
* recent subsets đã đủ lớn để modeling;
* M2 đã chứng minh older transaction có thể được giữ làm causal history mà không cần trở thành classifier-training rows.

M2 kết luận sample size không buộc project phải dùng toàn bộ lịch sử làm training rows.

Do đó H0 được giữ lại thay vì drop.

`M3.2-R01: 1991–2014 là historical reservoir mặc định; không bị xóa khỏi pipeline chỉ vì không nằm trong modeling window.`

---

# 5. Region R1 — 2015–2016

Đây là giai đoạn recent hơn nhưng có fraud prevalence tương đối cao:

```text
2015 : 0.192844%
2016 : 0.209430%
```

Quy mô mỗi năm khoảng `1.7 triệu transaction`.

R1 có thể cung cấp modeling rows cho một **longer recent training-window strategy**.

Tuy nhiên không được mặc định rằng R1 đại diện cho distribution của future period.

`M3.2-R02: 2015–2016 là modeling candidate, không phải reference regime cố định.`

---

# 6. Region R2 — 2017

2017 có:

```text
Transaction : 1,723,360
Fraud       : 255
Fraud rate  : 0.014797%
```

Trong khi 2016 có fraud rate `0.209430%`.

Đây là thay đổi khoảng 14 lần.

Vì vậy 2017 không được gộp vào `2015–2019` rồi mặc định rằng toàn window có behavior giống nhau.

Tuy nhiên M3.2 cũng **không drop 2017**.

Nếu training strategy bắt đầu từ 2015 thì 2017 vẫn là dữ liệu quá khứ hợp lệ. Việc window chứa temporal diversity có thể là lợi ích hoặc bất lợi cho generalization; vấn đề đó phải được trả lời bằng experiment.

`M3.2-R03: giữ 2017 trong candidate longer-window strategy nhưng đánh dấu temporal heterogeneity rõ ràng.`

---

# 7. Region R3 — 2018

2018 có:

```text
Transaction : 1,721,615
Fraud       : 2,491
Fraud rate  : 0.144690%
```

Cả 12 tháng đều có positive samples.

2018 có hai vai trò tiềm năng:

1. recent training data;
2. temporal predecessor trực tiếp của 2019 pre-break evaluation region.

Do đó 2018 là một anchor quan trọng cho **short-recent training-window family**.

`M3.2-R04: 2018 là recent modeling candidate quan trọng.`

Điều này chưa có nghĩa final train sẽ bằng đúng năm 2018.

---

# 8. Region R4 — 2019-01 → 2019-10

Đây là region đặc biệt quan trọng cho future evaluation design.

R4 có:

```text
Transaction : 1,435,413
Fraud       : 2,087
Fraud rate  : 0.145394%
```

Mỗi tháng đều có positive class và fraud rate tổng thể khá gần 2018.

R4 vì vậy là **primary candidate reservoir cho validation/final-test design**.

Cụm từ “candidate reservoir” rất quan trọng.

M3.2 **không khóa toàn bộ 2019-01 → 2019-10 làm test**.

M3.3 có thể chia R4 thành các temporal đoạn nhỏ hơn, ví dụ:

```text
earlier part
→ validation candidate

later part
→ final-test candidate
```

nhưng boundary tháng chính xác phải được audit trước.

`M3.2-R05: 2019-01 → 2019-10 được ưu tiên làm positive-bearing future-evaluation reservoir.`

`Final validation/test boundary: OPEN.`

---

# 9. Region R5 — 2019-11 → 2019-12

Region này có:

```text
Transaction : 288,525
Fraud       : 0
Fraud rate  : 0%
```

R5 không phù hợp làm fraud-performance final test vì không có positive class.

Tuy nhiên cũng không nên tự động drop khỏi project.

Nó có thể được giữ như một:

`special temporal regime`

hoặc:

`supplementary stress / diagnostic period`

nếu sau này có câu hỏi rõ ràng cần sử dụng nó.

Ví dụ có thể nghiên cứu model behavior trong một zero-positive period, nhưng không được diễn giải Recall/F1 fraud từ region này.

`M3.2-R06: R5 bị loại khỏi candidate final fraud-performance test nhưng chưa bị loại khỏi mọi phân tích bổ trợ.`

Việc có thực sự sử dụng R5 như stress segment hay không vẫn OPEN.

---

# 10. Region R6 — 2020-01 → 2020-02

2020 chỉ có hai tháng:

```text
Transaction : 336,500
Fraud       : 0
```

M2.8 còn phát hiện khi thêm 2020, unique entity tăng đáng kể trong khi giai đoạn này không bổ sung fraud nào.

Do đó 2020 đồng thời có ba đặc điểm:

* partial year;
* zero fraud;
* chứa thêm entity không xuất hiện trong historical period trước đó.

M3.1 đã khóa:

`2020 ≠ final fraud-performance test`

M3.2 giữ nguyên quyết định này.

`M3.2-R07: 2020 nằm ngoài final fraud-performance candidate space.`

Có thể giữ cho một diagnostic/cold-start analysis sau này nếu có mục tiêu rõ ràng, nhưng không được sử dụng thay thế final fraud test.

---

# 11. Một sửa đổi quan trọng đối với cách hiểu các window của M2.8

M2.8 đã xác nhận feasibility của:

```text
RECENT_2015_2019
8,579,208 transaction
11,693 fraud

RECENT_2018_2019
3,445,553 transaction
4,578 fraud

PRE_BREAK_2018_TO_2019_10
3,157,028 transaction
4,578 fraud
```

Các con số này chứng minh:

> recent subset vẫn đủ lớn để experiment.

Nhưng chúng **không phải final training windows**.

Nếu validation và test nằm trong 2019 thì training set không thể đồng thời chứa toàn bộ 2019.

Ví dụ thiết kế sau là không hợp lệ:

```text
TRAIN
2018 → 2019-10

TEST
2019-07 → 2019-10
```

vì cùng các transaction tương lai đã có mặt trong training window.

Do đó từ M3.2 trở đi, các window M2.8 phải được diễn giải thành **window family**.

---

# 12. Candidate modeling-window families

## W-LONG — Longer recent window

Khái niệm:

```text
start = 2015-01
end   = thời điểm ngay trước validation start
```

Không còn định nghĩa cứng là:

`2015–2019`.

Ưu điểm cần kiểm nghiệm sau này:

* nhiều transaction;
* nhiều positive sample;
* nhiều temporal regimes hơn.

Rủi ro cần kiểm nghiệm:

* dữ liệu cũ hơn có thể ít đại diện hơn cho future distribution;
* chứa biến động rất mạnh như năm 2017.

`Trạng thái: CANDIDATE`

---

## W-SHORT — Short recent window

Khái niệm:

```text
start = 2018-01
end   = thời điểm ngay trước validation start
```

Ưu điểm tiềm năng:

* gần evaluation period hơn về thời gian;
* 2018 và 2019-01→10 có aggregate fraud prevalence tương đối gần nhau.

Rủi ro:

* ít training samples và positive samples hơn W-LONG.

M2 đã chứng minh 2018–2019 có hơn 3.44 triệu transaction và 4,578 fraud, nên quy mô tổng thể không phải blocker.

`Trạng thái: CANDIDATE`

---

## W-RECENT-ALT — Recent/pre-break alternative

M3.2 không khóa một start date thứ ba cụ thể.

Có thể tồn tại một window recent hơn, chẳng hạn bắt đầu sau 2018-01, nếu M3.3 cho thấy:

* training positives vẫn đủ;
* validation/test boundary hợp lý;
* sample size không trở thành blocker.

Window này chỉ được thêm khi split audit đưa ra lý do.

`Trạng thái: OPTIONAL CANDIDATE`

Điều này giữ đúng nguyên tắc adaptive plan của M3.1.

---

# 13. Không xem full-history window là default modeling strategy

`1991 → train cutoff`

không bị tuyên bố là “sai”.

Nhưng hiện không có lý do để xem nó là default.

Bằng chứng hiện tại cho thấy:

* distribution thay đổi mạnh theo thời gian;
* recent subsets đủ lớn;
* older rows vẫn có thể cung cấp historical context mà không cần fit classifier.

Vì vậy M3.2 ưu tiên so sánh **recent modeling windows**, thay vì mặc định càng nhiều năm càng tốt.

`M3.2-D01: full-history classifier training không phải default strategy của M3.`

Nếu sau này cần một full-history baseline để trả lời một câu hỏi cụ thể thì có thể bổ sung bằng Decision Log.

---

# 14. History warm-up policy v0.1

Đây là một output quan trọng của M3.2.

M2.8 đã so sánh giữ history và reset history tại đầu modeling window.

Với `2018–2019`:

```text
Có historical warm-up:
247 cold-start transaction
≈ 0.00717%

Reset tại 2018:
4,159 cold-start transaction
≈ 0.12071%
```

Historical warm-up làm giảm khoảng 94% số cold-start transaction theo nghĩa tương đối.

Với pre-break `2018 → 2019-10`, mức giảm khoảng 95.8%.

Do đó M3.2 khóa policy mặc định:

> **Khi sử dụng recent modeling window, không reset historical context một cách máy móc tại modeling-window start.**

Một transaction T có thể sử dụng transaction cũ hơn modeling window nếu:

`timestamp(history) < timestamp(T)`

và feature đó tuân thủ prediction-point guardrail.

Older transaction dùng làm history không tự động trở thành classifier-training rows.

---

## 14.1. History warm-up không được dùng target

History warm-up ở đây là historical transaction context phục vụ các feature causal hợp lệ.

Không được sử dụng:

`Is Fraud?`

hoặc investigation outcome của historical transaction như feature chỉ vì label của dataset có sẵn.

Target vẫn chỉ là target của supervised-learning task.

Việc có sử dụng một loại historical outcome nào đó trong một project khác phải phụ thuộc prediction-time availability; project hiện tại chưa có cơ sở để làm vậy.

---

## 14.2. Chưa khóa lookback cuối cùng

M3.2 chỉ khóa rằng pre-window historical context **được phép** và mặc định không bị reset.

M3.2 chưa khóa:

* history phải kéo dài toàn bộ lifetime;
* chỉ dùng 1 giờ;
* 1 ngày;
* 30 ngày;
* previous N transactions;
* hay các aggregation khác.

Đây là vấn đề của causal feature engineering ở M4.

`M3.2-D02: historical warm-up được giữ; exact feature lookback vẫn OPEN cho M4.`

---

# 15. Candidate evaluation regions chuyển sang M3.3

Sau M3.2, candidate space được thu hẹp như sau:

```text
PRIMARY POSITIVE-BEARING FUTURE REGION

2019-01 → 2019-10

Vai trò tiềm năng:
validation + final fraud test

Boundary cụ thể:
CHƯA KHÓA
```

```text
RECENT TRAINING REGION

2015/2018 → trước validation start

Vai trò:
classifier-training rows

Start date:
phụ thuộc window strategy

End date:
bắt buộc trước validation
```

```text
HISTORY RESERVOIR

1991 → trước current transaction

Vai trò:
causal historical context

Không đồng nghĩa:
classifier-training rows
```

```text
ZERO-FRAUD POST-BREAK REGION

2019-11 → 2020-02

Vai trò tiềm năng:
supplementary temporal diagnostic / stress segment

Không được:
làm final fraud-performance test
```

Đây mới là candidate map.

Chưa có tháng nào được tuyên bố chính thức là TRAIN, VALIDATION hoặc FINAL TEST.

---

# 16. Temporal map tổng hợp

```text
TIME
──────────────────────────────────────────────────────────────────────────────────────────────→

1991                    2015       2017       2018             2019-01             2019-10   11/2019        2020-02
│                        │          │          │                   │                    │        │               │
│                        │          │          │                   │                    │        │               │
└──── LEGACY HISTORY ────┴─ RECENT MODELING CANDIDATE ────────────┴─ FUTURE EVALUATION ┴────────┴ ZERO-FRAUD ───┘
       RESERVOIR                                                        RESERVOIR                 REGIME

1991–2014
→ historical warm-up ưu tiên

2015–2016
→ longer-window modeling candidate
→ fraud prevalence cao hơn các năm lân cận

2017
→ temporal heterogeneity rất mạnh
→ không drop nhưng phải được ghi nhận

2018
→ short-recent modeling anchor
→ 12/12 tháng có fraud

2019-01 → 2019-10
→ positive-bearing future-evaluation reservoir
→ 1,435,413 transaction
→ 2,087 fraud
→ chưa chia validation/test

2019-11 → 2019-12
→ zero-fraud post-break region
→ không final fraud test

2020-01 → 2020-02
→ partial-year + zero-fraud region
→ không final fraud test
```

---

# 17. Các quyết định được khóa tại M3.2

| Decision ID | Nội dung | Trạng thái |
|---|---|---|
| M3.2-D01 | Full-history classifier training không phải default strategy | LOCKED |
| M3.2-D02 | Giữ historical warm-up cho recent modeling windows; exact lookback để M4 quyết định | LOCKED |
| M3.2-D03 | 11/2019 là temporal break quan trọng phải được tôn trọng khi thiết kế split | LOCKED |
| M3.2-D04 | 2019-01→2019-10 là primary positive-bearing future-evaluation reservoir | LOCKED AS CANDIDATE REGION |
| M3.2-D05 | 2019-11→2020-02 không dùng làm final fraud-performance test | LOCKED |
| M3.2-D06 | M2.8 windows là feasibility envelopes, không phải final training boundaries | LOCKED |
| M3.2-D07 | Candidate training window phải kết thúc trước validation start | LOCKED |
| M3.2-D08 | W-LONG bắt đầu khoảng 2015 và W-SHORT bắt đầu khoảng 2018 là hai window family chính cần giữ để so sánh | CANDIDATE LOCKED |
| M3.2-D09 | 2017 không bị drop chỉ vì fraud prevalence khác biệt | LOCKED |
| M3.2-D10 | Zero-fraud post-break region chỉ có thể là supplementary diagnostic/stress region nếu có câu hỏi rõ | LOCKED PRINCIPLE |

---

# 18. Những quyết định cố ý chưa khóa

## M3.2-O01 — Validation boundary

`OPEN`

Chưa quyết định validation bắt đầu tháng nào.

M3.3 phải audit transaction count và fraud count cho từng candidate.

---

## M3.2-O02 — Final-test boundary

`OPEN`

Chưa quyết định các tháng cụ thể của 2019-01→10 thuộc final test.

Final test phải nằm sau validation và chứa đủ positive samples.

---

## M3.2-O03 — Số tháng validation/test

`OPEN`

Không dùng một tỷ lệ 80/20 hoặc một số tháng tùy ý trước khi kiểm tra support thực tế.

---

## M3.2-O04 — Training-window winner

`OPEN`

W-LONG và W-SHORT chỉ là candidate family.

Không có model experiment ở M3.2 để chứng minh window nào tốt hơn.

---

## M3.2-O05 — Có dùng zero-fraud regime làm stress segment không?

`OPEN`

2019-11→2020-02 không phải final fraud test.

Việc sử dụng như một supplementary diagnostic chỉ được thực hiện nếu sau này có câu hỏi evaluation cụ thể.

---

## M3.2-O06 — Exact history lookback

`OPEN / M4`

M3.2 cho phép pre-window history nhưng không quyết định feature engineering cuối.

---

## M3.2-O07 — Temporal-CV strategy

`OPEN`

Rolling/forward validation hay một fixed validation block vẫn cần M3.3/M3.7 xem xét.

---

# 19. Một cảnh báo cho M3.3

M3.3 không được xây candidate split theo cách:

`train = 2018`

`validation = 2019`

`test = 2020`

chỉ vì cấu trúc theo năm trông đẹp.

Thiết kế đó vi phạm bằng chứng EDA vì 2020 có `0 fraud`.

Tương tự, cũng không nên:

`train = 2015–2018`

`test = toàn bộ 2019`

mà không xem xét điểm gãy tháng 11/2019.

M2.3 đã chứng minh:

> 2019 không phải một temporal block đồng nhất.

Do đó M3.3 phải làm việc ở độ phân giải **tháng**, ít nhất đối với giai đoạn cuối dataset.

---

# 20. M3.2 Gate

### GATE-01 — Temporal instability đã được đưa vào experiment map chưa?

`PASS`

Không giả định full history hay 2015–2019 là một distribution ổn định.

### GATE-02 — Temporal break quan trọng đã xác định chưa?

`PASS`

Điểm gãy chính phục vụ design nằm khoảng `11/2019`.

### GATE-03 — Positive-bearing future region đã xác định chưa?

`PASS`

`2019-01 → 2019-10`.

### GATE-04 — Zero-fraud region đã được tách riêng chưa?

`PASS`

`2019-11 → 2020-02`.

### GATE-05 — 2020 có bị dùng làm final fraud test không?

`PASS`

Không.

### GATE-06 — Historical context và modeling rows đã được phân biệt chưa?

`PASS`

Older history có thể warm-up nhưng không tự động trở thành training rows.

### GATE-07 — Candidate training windows có còn vi phạm future-evaluation boundary không?

`PASS`

Window được định nghĩa lại dưới dạng:

`start → trước validation start`

thay vì bê nguyên `2015–2019` hoặc `2018–2019`.

### GATE-08 — Có premature lock final split không?

`PASS`

Validation/test month boundaries vẫn OPEN.

### GATE-09 — Candidate data có đủ quy mô để tiếp tục không?

`PASS WITH AUDIT REQUIRED`

M2.8 đã chứng minh recent windows giữ hàng triệu transaction và hàng nghìn fraud. Tuy nhiên số lượng trong **từng partition cụ thể** chỉ được xác nhận ở M3.3.

### GATE-10 — Có blocking issue cho M3.3 không?

`NO`

---

# 21. Kết luận M3.2

`Decision ID: M3.2-FINAL-D01`

`Temporal map: ĐÃ XÂY`

`11/2019 temporal break: ĐÃ KHÓA CHO EXPERIMENT DESIGN`

`Primary future-evaluation reservoir: 2019-01 → 2019-10`

`Zero-fraud regime: 2019-11 → 2020-02`

`2020 final fraud test: LOẠI`

`Historical warm-up policy v0.1: ĐÃ KHÓA`

`Long recent training family: CANDIDATE`

`Short recent training family: CANDIDATE`

`Final train boundary: CHƯA KHÓA — ĐÚNG PHẠM VI`

`Final validation boundary: CHƯA KHÓA — ĐÚNG PHẠM VI`

`Final test boundary: CHƯA KHÓA — ĐÚNG PHẠM VI`

`Training-window winner: CHƯA KHÓA — ĐÚNG PHẠM VI`

`Blocking issue: NONE`

`M3.2 Gate: PASS`

`M3.2 Status: PASS — TEMPORAL MAP ESTABLISHED`

M3.2 chính thức thu hẹp không gian thiết kế cho bước tiếp theo.

**M3.3 phải chuyển từ “region” sang “partition cụ thể”: xây một số candidate train/validation/test split ở cấp tháng, tính chính xác transaction count, fraud count, fraud rate và các đặc điểm entity/cold-start cần thiết cho từng partition, sau đó mới quyết định candidate nào đủ điều kiện để khóa.**
