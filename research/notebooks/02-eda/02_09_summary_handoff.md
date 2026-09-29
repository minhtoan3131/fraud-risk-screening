# M2.9 — Tổng hợp EDA, nhật ký quyết định và chuyển giao

## 1. Vai trò của M2.9

M2.1–M2.8 là các bước phân tích chi tiết, trong đó mỗi notebook thực hiện theo hướng: <br>
`Câu hỏi → Phương pháp kiểm tra → Bằng chứng → Nhận xét → Kết luận`. <br>

M2.9 không tiếp tục tạo thêm phép EDA mới. <br>

Vai trò của M2.9 là: <br>
- tổng hợp những phát hiện quan trọng nhất của toàn Milestone 2; <br>
- phân biệt điều đã biết với điều vẫn còn mở; <br>
- ghi lại các quyết định đã khóa; <br>
- chuyển giao các vấn đề phù hợp sang M3 và M4; <br>
- kiểm tra M2 Gate; <br>
- xác nhận Milestone 2 đã đủ điều kiện kết thúc hay chưa. <br>

Mỗi phát hiện quan trọng được trình bày theo cấu trúc: <br>
`Quan sát → Bằng chứng → Ý nghĩa → Chuyển tiếp`. <br>

M2.9 chỉ tổng hợp kết quả đã được xác nhận từ M2.1–M2.8. <br>
Không đọc lại raw dataset và không tạo thêm kết luận mới nếu chưa có bằng chứng từ các bước trước. <br>

## 2. Phạm vi dữ liệu và bài toán

Dataset: <br>
`IBM Synthetic Credit Card Transactions / TabFormer` <br>

Bài toán của project: <br>
`supervised binary classification` ở mức transaction để đánh giá dấu hiệu rủi ro fraud. <br>

Raw artifact có: <br>
`24,386,900 transaction` <br>
`15 cột` <br>

Khoảng thời gian: <br>
`1991-01-02 07:10 → 2020-02-28 23:58` <br>

Target: <br>
`Is Fraud?` <br>

Positive class: <br>
`Yes` <br>

Negative class: <br>
`No` <br>

Dataset là `fully synthetic`. <br>

Do đó các kết luận của Milestone 2 chỉ được hiểu trong phạm vi artifact synthetic này. <br>
Không được trình bày kết quả như bằng chứng về hiệu quả hoặc hành vi fraud trong hệ thống ngân hàng thực tế. <br>

---

# 3. Tóm tắt kết quả M2.1–M2.8

## M2.1 — Khóa quy trình EDA và môi trường

M2.1 đã khóa quy trình làm việc theo hướng `câu hỏi trước → code sau → kết luận từ output thực tế`. <br>

Artifact được xác minh đúng file đã audit ở M1. <br>

Môi trường chính: <br>
`Python 3.14.6` <br>
`Pandas 3.0.5` <br>
`NumPy 2.5.3` <br>

Các guardrail từ M1 tiếp tục có hiệu lực trong toàn bộ M2. <br>

`Trạng thái M2.1: PASS` <br>

## M2.2 — Cấu trúc và representation

Raw schema có đúng `15 cột`, không thiếu, không thừa và không trùng tên. <br>

`Amount` được chuyển sang representation số thành công trên `100%` dataset. <br>

`Timestamp` được tạo thành công trên `100%` dataset. <br>

Raw CSV không được sắp toàn cục theo Timestamp; phát hiện `5,789` lần Timestamp giảm so với dòng liền trước. <br>

`MCC` và `Zip` được hiểu là `categorical code`, không phải đại lượng số có thứ tự. <br>

`User`, `Card`, `Merchant Name` chỉ được dùng làm khóa grouping / history, không dùng raw trực tiếp làm classifier feature. <br>

`Errors?` tiếp tục bị loại khỏi Model V1. <br>

`Trạng thái M2.2: PASS WITH FINDING` <br>

## M2.3 — Target và chiều thời gian

Target sạch: chỉ có `Yes / No`, không missing. <br>

Toàn dataset: <br>
`Fraud = 29,757 — 0.122020%` <br>
`Non-fraud = 24,357,143 — 99.877980%` <br>

Tỷ lệ xấp xỉ: <br>
`818.5 non-fraud / 1 fraud` <br>

Một classifier luôn dự đoán non-fraud vẫn có thể đạt Accuracy khoảng `99.877980%`. <br>

Fraud rate thay đổi mạnh theo thời gian. <br>

Một điểm gãy quan trọng được phát hiện: <br>
`từ 11/2019 đến 02/2020 có 625,025 transaction nhưng 0 fraud`. <br>

2020 chỉ chứa tháng 1–2 và có `0 fraud`, nên không phù hợp làm final fraud test. <br>

`Trạng thái M2.3: PASS WITH FINDINGS` <br>

## M2.4 — Phân bố từng feature giao dịch

`Amount` có distribution lệch phải, tail dài và chứa nhóm Amount âm đáng kể. <br>

`Negative Amount = 1,244,683 — khoảng 5.10%`. <br>

`Zero Amount = 20,213 — khoảng 0.083%`. <br>

`Use Chip` có ba nhóm với support lớn: Swipe, Chip và Online Transaction. <br>

`MCC` có `109 category`, với cấu trúc tập trung kết hợp long-tail. <br>

Các feature thời gian cho thấy pattern theo giờ trong ngày rõ hơn theo thứ hoặc tháng. <br>

Location có cardinality rất khác nhau: <br>
`Merchant City = 13,429` <br>
`Merchant State = 223` <br>
`Zip = 27,321` <br>

`Merchant City = ONLINE` là một finding semantic cần được giải thích bằng transaction mode và missingness. <br>

`Trạng thái M2.4: PASS WITH FINDINGS` <br>

## M2.5 — Chất lượng dữ liệu và missingness

Chỉ ba cột có missing: <br>
`Errors?` <br>
`Merchant State` <br>
`Zip` <br>

Missing location không phải missing ngẫu nhiên thông thường. <br>

Toàn bộ `Online Transaction` có: <br>
`Merchant City = ONLINE` <br>
`Merchant State missing` <br>
`Zip missing` <br>

Do đó location missing của Online Transaction là `structural missingness`. <br>

Ngoài Online Transaction, một phần Zip missing còn liên quan tới merchant quốc tế, cho thấy Zip missing có nhiều cơ chế semantic khác nhau. <br>

Amount âm / zero có pattern theo transaction mode, không phù hợp với giả thuyết corruption ngẫu nhiên đơn giản. <br>

Exact duplicate tồn tại nhưng cực nhỏ: <br>
`66 duplicate groups` <br>
`132 duplicate-member rows` <br>
`66 extra rows` <br>
`0 fraud trong duplicate-member rows` <br>

Không phát hiện vấn đề chất lượng dữ liệu nào buộc phải sửa raw dataset ngay trong M2. <br>

`Trạng thái M2.5: PASS WITH FINDINGS` <br>

## M2.6 — Quan hệ giữa feature và target

Nhiều feature có association rõ với target trên full dataset. <br>

Association mạnh xuất hiện ở: <br>
`Use Chip` <br>
`upper tail của Amount` <br>
`một số MCC` <br>
`hour_of_day` <br>
`semantic location status` <br>
`một số raw location category` <br>

Tuy nhiên finding quan trọng hơn là: <br>
`feature-target association không ổn định theo thời gian`. <br>

Ví dụ, Online Transaction có fraud rate cao trên full history nhưng association thay đổi mạnh ở các temporal regime gần cuối dataset. <br>

`Negative Amount + Online Transaction` cũng có association mạnh trên full history nhưng suy yếu hoặc biến mất ở các giai đoạn gần 2019. <br>

Do đó không được sử dụng full-history fraud rate làm bằng chứng duy nhất để chọn feature cho mô hình dự đoán tương lai. <br>

MCC và location có một số association cực mạnh, nhưng đồng thời có rủi ro `synthetic shortcut`, memorization và high-cardinality overfitting. <br>

`Trạng thái M2.6: PASS WITH FINDINGS` <br>

## M2.7 — Entity và lịch sử User / Card

Dataset có: <br>
`2,000 User` <br>
`6,139 User+Card` <br>

Median User có hơn `10,860 transaction`. <br>

Median Card có `2,602 transaction`. <br>

Median active span của Card khoảng `3,189 ngày`, tương đương gần `8.7 năm`. <br>

Strict card-level cold-start chỉ khoảng `0.0253%` transaction. <br>

Trong stress test 2019: <br>
`new-user transaction ≈ 0.1888%` <br>
`new-card transaction ≈ 1.0757%` <br>

Phần lớn transaction vì vậy có historical context. <br>

Evaluation 2019 chủ yếu phản ánh transaction của existing User/Card hơn là hoàn toàn new entity. <br>

`Trạng thái M2.7: PASS WITH FINDINGS` <br>

## M2.8 — Khả năng xây behavioral feature và temporal window

Raw artifact có cấu trúc thuận lợi cho card-level causal streaming: <br>
- mỗi User+Card nằm trong một contiguous block; <br>
- không có Card block reappearance; <br>
- Timestamp không giảm bên trong Card. <br>

Tuy nhiên có nhiều transaction cùng Timestamp, nên transaction cùng thời điểm không được dùng làm history cho nhau. <br>

Bốn behavioral feature prototype đã được tính thành công theo strict causal rule: <br>
`time_since_previous_transaction` <br>
`transactions_last_1h` <br>
`amount_minus_previous_mean` <br>
`is_new_merchant` <br>

Prior Card history coverage khoảng `99.97%+`. <br>

Chỉ khoảng `19–21%` transaction có ít nhất một prior transaction trong 1 giờ, nên short-window velocity có cấu trúc zero-heavy. <br>

Causal feature pass xử lý toàn bộ `24,386,900 transaction / 6,139 Card` thành công và không gặp computational blocker. <br>

Các temporal window candidate vẫn đủ lớn: <br>
`2015–2019 = 8,579,208 transaction / 11,693 fraud` <br>
`2018–2019 = 3,445,553 transaction / 4,578 fraud` <br>
`2018 → 2019-10 = 3,157,028 transaction / 4,578 fraud` <br>

Older transaction có thể được dùng làm causal history warm-up cho recent modeling window mà không nhất thiết trở thành training rows. <br>

`Trạng thái M2.8: PASS WITH FINDINGS` <br>

---

# 4. Các phát hiện quan trọng nhất của Milestone 2

## M2-F01 — Target mất cân bằng rất mạnh

`Quan sát` <br>
Fraud chỉ chiếm khoảng `0.122%` toàn dataset. <br>

`Bằng chứng` <br>
`29,757 fraud / 24,386,900 transaction`. <br>

`Ý nghĩa` <br>
Accuracy đơn độc không phản ánh khả năng phát hiện fraud. <br>

`Chuyển tiếp` <br>
M3 phải chọn metric phù hợp với class imbalance và positive class. <br>

## M2-F02 — Dữ liệu thay đổi mạnh theo thời gian

`Quan sát` <br>
Fraud prevalence và nhiều feature-target relationship không ổn định theo thời gian. <br>

`Bằng chứng` <br>
Từ `11/2019 → 02/2020` có `625,025 transaction` nhưng `0 fraud`; một số association mạnh trên full history suy yếu hoặc biến mất ở 2018–2019. <br>

`Ý nghĩa` <br>
Random split có nguy cơ tạo đánh giá quá lạc quan và không đại diện cho dự đoán tương lai. <br>

`Chuyển tiếp` <br>
M3 bắt buộc dùng temporal evaluation làm thiết kế chính. <br>

## M2-F03 — 2020 không phù hợp làm final fraud test

`Quan sát` <br>
2020 chỉ có tháng 1–2 và không có fraud. <br>

`Ý nghĩa` <br>
Không thể đánh giá Recall / Precision / F1 cho positive class một cách có ý nghĩa nếu final test không có positive sample. <br>

`Chuyển tiếp` <br>
M3 không dùng 2020 làm final fraud-performance test. <br>

## M2-F04 — Negative Amount không được tự động coi là lỗi

`Quan sát` <br>
Negative Amount chiếm khoảng `5.10%` dataset và có pattern theo transaction mode. <br>

`Ý nghĩa` <br>
Không phù hợp với giả thuyết một vài lỗi nhập liệu ngẫu nhiên. Exact semantic vẫn chưa được khóa. <br>

`Chuyển tiếp` <br>
M4 không tự động `abs()`, drop hoặc clip Amount âm. <br>

## M2-F05 — Location missing mang tính cấu trúc

`Quan sát` <br>
Online Transaction có State / Zip missing `100%`; residual Zip missing còn liên quan merchant quốc tế. <br>

`Ý nghĩa` <br>
Missing location không phải một cơ chế duy nhất và không phải missing ngẫu nhiên thông thường. <br>

`Chuyển tiếp` <br>
M4 không drop hoặc impute location missing máy móc. <br>

## M2-F06 — MCC và location có rủi ro shortcut

`Quan sát` <br>
Một số MCC / location category có fraud association rất mạnh. <br>

`Ý nghĩa` <br>
Trong synthetic dataset, association cực mạnh có thể phản ánh logic generator hoặc proxy cho merchant identity. <br>

`Chuyển tiếp` <br>
M3/M4 phải xem support, temporal stability và generalization; không chọn feature chỉ dựa trên full-history fraud rate. <br>

## M2-F07 — Entity history đủ sâu

`Quan sát` <br>
Median Card có `2,602 transaction` và active span gần `8.7 năm`. <br>

`Ý nghĩa` <br>
Dataset có nền tảng tốt để xây behavioral feature. <br>

`Chuyển tiếp` <br>
M4 có thể triển khai card-level và user-level historical feature theo strict causal rule. <br>

## M2-F08 — Cold-start là phần nhỏ

`Quan sát` <br>
Strict card-level cold-start khoảng `0.0253%`; new-card transaction 2019 khoảng `1.0757%`. <br>

`Ý nghĩa` <br>
Evaluation chủ yếu đo khả năng screening transaction của existing entity. <br>

`Chuyển tiếp` <br>
M3 phải ghi rõ cold-start limitation và có thể báo cáo riêng segment nếu cần. <br>

## M2-F09 — Behavioral feature có thể tính đúng causal

`Quan sát` <br>
Bốn prototype đã được tính thành công trên toàn artifact với nguyên tắc `timestamp(history) < timestamp(current)`. <br>

`Ý nghĩa` <br>
Behavioral feature engineering khả thi cả về dữ liệu và chi phí tính toán. <br>

`Chuyển tiếp` <br>
M4 triển khai feature pipeline với order assertion hoặc explicit sort. <br>

## M2-F10 — History rows và training rows không cần giống nhau

`Quan sát` <br>
Older transaction có thể cung cấp causal history cho recent modeling window. <br>

`Ý nghĩa` <br>
Không cần train trên toàn bộ gần 30 năm chỉ để giữ historical context. <br>

`Chuyển tiếp` <br>
M3 có thể thử recent training window kết hợp older-history warm-up. <br>

## M2-F11 — Recent temporal subset vẫn đủ lớn

`Quan sát` <br>
2015–2019 vẫn có hơn `8.57 triệu transaction / 11,693 fraud`; 2018–2019 có hơn `3.44 triệu / 4,578 fraud`. <br>

`Ý nghĩa` <br>
Sample size không buộc project phải sử dụng toàn bộ lịch sử làm training rows. <br>

`Chuyển tiếp` <br>
M3 nên so sánh các temporal training-window strategy. <br>

---

# 5. Nhật ký các quyết định đã khóa

## M2-D01 — Evaluation chính phải tôn trọng chiều thời gian

Lý do: target và feature-target association thay đổi mạnh theo temporal regime. <br>

`Trạng thái: ĐÃ KHÓA` <br>

## M2-D02 — Không dùng 2020 làm final fraud-performance test

Lý do: 2020 là partial year và có `0 fraud`. <br>

`Trạng thái: ĐÃ KHÓA` <br>

## M2-D03 — Không dùng Accuracy đơn độc

Lý do: fraud chỉ khoảng `0.122%`. <br>

`Trạng thái: ĐÃ KHÓA` <br>

## M2-D04 — Không dùng raw User / Card / Merchant Name trực tiếp làm classifier feature

Lý do: đây là identifier / history key và có rủi ro memorization. <br>

`Trạng thái: ĐÃ KHÓA` <br>

## M2-D05 — Không dùng Errors? trong Model V1

Lý do: prediction-time availability chưa được chứng minh. <br>

`Trạng thái: ĐÃ KHÓA` <br>

## M2-D06 — Không drop State / Zip missing một cách máy móc

Lý do: location missing mang tính structural. <br>

`Trạng thái: ĐÃ KHÓA` <br>

## M2-D07 — Không tự động abs / drop / clip negative Amount

Lý do: negative Amount có quy mô đáng kể và semantic chưa được khóa. <br>

`Trạng thái: ĐÃ KHÓA` <br>

## M2-D08 — Mọi historical feature phải strict causal

Quy tắc: <br>
`timestamp(history) < timestamp(current transaction)` <br>

Không sử dụng transaction hiện tại hoặc tương lai để tạo historical feature. <br>

`Trạng thái: ĐÃ KHÓA` <br>

## M2-D09 — Transaction cùng Timestamp không được dùng làm history cho nhau

Lý do: Timestamp chỉ có độ phân giải tới phút và tồn tại timestamp tie. <br>

`Trạng thái: ĐÃ KHÓA` <br>

## M2-D10 — Entity overlap giữa train và future không tự động là leakage

Existing User/Card history là context hợp lệ nếu chỉ sử dụng dữ liệu quá khứ. <br>

`Trạng thái: ĐÃ KHÓA` <br>

## M2-D11 — Không biến full-history fraud rate thành rule cố định

Áp dụng đặc biệt cho MCC, location, Use Chip và các interaction có fraud rate cực đoan. <br>

Lý do: temporal instability, support không đồng đều và synthetic shortcut risk. <br>

`Trạng thái: ĐÃ KHÓA` <br>

## M2-D12 — History warm-up rows có thể khác training rows

Older past data có thể dùng làm causal history cho recent transaction mà không cần được dùng để fit classifier. <br>

`Trạng thái: ĐÃ KHÓA` <br>

## M2-D13 — Không overclaim tính thực tế

Dataset là fully synthetic. <br>

Kết quả sau này không được trình bày như bằng chứng hiệu quả trên production banking data. <br>

`Trạng thái: ĐÃ KHÓA` <br>

---

# 6. Những quyết định cố ý chưa khóa

Các mục dưới đây chưa được quyết định tại M2 vì cần experiment hoặc thuộc đúng phạm vi của milestone sau. <br>

## M2-O01 — Final train / validation / test boundaries

`Chuyển sang: M3` <br>

Cần thiết kế temporal split có đủ positive samples và tôn trọng past → future ordering. <br>

## M2-O02 — Final training window

`Chuyển sang: M3` <br>

2015–2019, 2018–2019 và pre-break recent window đều khả thi về sample size. <br>

Cần so sánh bằng experiment thay vì khóa bằng EDA. <br>

## M2-O03 — Metric chính và metric phụ

`Chuyển sang: M3` <br>

M2 chỉ khóa rằng Accuracy đơn độc không đủ. <br>

Metric cuối phải phù hợp với mục tiêu classification và class imbalance. <br>

## M2-O04 — Chiến lược xử lý class imbalance

`Chuyển sang: M3/M4` <br>

Nếu dùng class weighting, undersampling, oversampling hoặc kỹ thuật khác thì chỉ được thực hiện sau split và chỉ trên training data. <br>

## M2-O05 — Final feature set

`Chuyển sang: M4` <br>

EDA association không đủ để quyết định feature cuối. <br>

## M2-O06 — Encoding MCC và location

`Chuyển sang: M4` <br>

Cần cân nhắc cardinality, long-tail, structural missingness và shortcut risk. <br>

## M2-O07 — Transformation của Amount

`Chuyển sang: M4` <br>

Amount lệch phải và có extreme values nhưng chưa đủ cơ sở để khóa log / scaling / binning. <br>

## M2-O08 — Final behavioral feature set

`Chuyển sang: M4` <br>

M2.8 mới chứng minh computability, coverage và distribution; chưa chứng minh model gain. <br>

## M2-O09 — Model family

`Chuyển sang: giai đoạn modeling` <br>

EDA không phải bằng chứng đủ để chọn thuật toán cuối. <br>

## M2-O10 — Classification threshold

`Chuyển sang: giai đoạn đánh giá model` <br>

Threshold phải được chọn bằng validation objective, không được khóa từ EDA. <br>

---

# 7. Chuyển giao sang M3 — Thiết kế experiment và evaluation

## 7.1. Evaluation phải theo thời gian

Không dùng random split đơn giản làm evaluation chính. <br>

Thiết kế phải có hướng: <br>
`quá khứ → train / validation → tương lai → test` <br>

Lý do không chỉ là nguyên tắc chống leakage. <br>
M2 đã chứng minh distribution và feature-target relationship thực sự thay đổi theo thời gian. <br>

## 7.2. Không dùng 2020 làm final fraud test

2020 chỉ chứa tháng 1–2 và có `0 fraud`. <br>

Final test phải chứa đủ positive samples để đánh giá khả năng phát hiện fraud. <br>

## 7.3. Không dùng Accuracy làm metric chính

Fraud rate chỉ khoảng `0.122%`. <br>

Một classifier luôn dự đoán non-fraud vẫn đạt Accuracy gần `99.88%` nhưng không có giá trị phát hiện fraud. <br>

## 7.4. So sánh các temporal training window

Các candidate đáng thử gồm ít nhất: <br>
`2015–2019` <br>
`2018–2019` <br>
`2018 → 2019-10` hoặc một thiết kế tương đương có kiểm soát zero-fraud regime. <br>

M2 không khóa window nào là tốt nhất. <br>

## 7.5. Có thể sử dụng historical warm-up

Nếu chỉ dùng recent rows để train model, transaction cũ hơn vẫn có thể được dùng làm historical context nếu chúng xảy ra trước prediction point. <br>

Cần phân biệt rõ: <br>
`rows dùng để xây history` <br>
và <br>
`rows dùng để fit classifier`. <br>

## 7.6. Ghi rõ cold-start limitation

Evaluation tương lai chủ yếu phản ánh existing User/Card. <br>

Nếu cần đánh giá cold-start, nên xem đây là một evaluation segment riêng. <br>

---

# 8. Chuyển giao sang M4 — Preprocessing và feature engineering

## 8.1. Amount

Giữ nguyên dấu của Amount. <br>

Không tự động: <br>
`abs()` <br>
`drop` <br>
`clip` <br>

Có thể thử scaling / transformation / binning sau này, nhưng mọi parameter học từ dữ liệu phải fit chỉ trên training partition. <br>

## 8.2. Use Chip

Use Chip là categorical feature hợp lệ. <br>

Tuy nhiên association của từng mode với fraud không ổn định theo thời gian. <br>

Không biến fraud rate lịch sử của category thành rule cố định. <br>

## 8.3. MCC

MCC phải được xử lý như `categorical code`. <br>

Encoding cần cân nhắc support, long-tail và temporal/generalization risk. <br>

## 8.4. Location

Merchant City / State / Zip chỉ là conditional candidate. <br>

Không drop missing location máy móc. <br>

Không mặc định dùng một imputation strategy cho mọi Zip missing. <br>

Cần cân nhắc redundancy giữa: <br>
`Use Chip` <br>
`Merchant City = ONLINE` <br>
`State missing` <br>
`Zip missing`. <br>

## 8.5. Identifier

Không đưa raw: <br>
`User` <br>
`Card` <br>
`Merchant Name` <br>

trực tiếp vào classifier. <br>

Các field này chỉ dùng làm key cho historical / behavioral feature. <br>

## 8.6. Behavioral feature

Ít nhất bốn candidate đã được xác nhận có thể tính causal: <br>
`time_since_previous_transaction` <br>
`transactions_last_1h` <br>
`amount_minus_previous_mean` <br>
`is_new_merchant` <br>

Đây là candidate cần experiment, chưa phải final feature set. <br>

## 8.7. Causal ordering

Mọi historical feature của transaction T chỉ được dùng dữ liệu có: <br>
`timestamp(history) < timestamp(T)` <br>

Transaction cùng Timestamp không được dùng làm history cho nhau. <br>

Pipeline phải explicit sort hoặc kiểm tra invariant của physical order trước khi tính history. <br>

## 8.8. Cold-start

Không drop transaction chưa có history. <br>

Cần representation hợp lý cho trường hợp history unavailable. <br>

Có thể cân nhắc user-level history khi Card mới nhưng User đã có lịch sử. <br>

---

# 9. Các limitation còn lại sau Milestone 2

## 9.1. Synthetic data

Dataset là fully synthetic. <br>

Các association mạnh giữa merchant / MCC / location và fraud có thể phản ánh logic của simulator. <br>

Không được suy rộng trực tiếp sang hệ thống ngân hàng thật. <br>

## 9.2. Temporal instability

Target prevalence và nhiều feature-target relationship thay đổi mạnh theo thời gian. <br>

Performance ở một temporal regime không đảm bảo ổn định ở regime khác. <br>

## 9.3. Negative Amount semantic

Negative Amount được xác nhận có pattern nhưng exact semantic vẫn chưa được khóa. <br>

## 9.4. High-cardinality shortcut

Merchant/location/MCC có thể tạo shortcut trong synthetic artifact. <br>

Raw Merchant Name đặc biệt không được sử dụng trực tiếp. <br>

## 9.5. Cold-start

Future evaluation chủ yếu bao phủ existing User/Card. <br>

Khả năng tổng quát hóa cho hoàn toàn new User/Card được kiểm chứng yếu hơn. <br>

## 9.6. Behavioral feature

M2.8 mới chứng minh khả năng tính causal, coverage, distribution và chi phí. <br>

Chưa chứng minh behavioral feature cải thiện classification performance. <br>

## 9.7. Physical order

Causal streaming được xác minh trên artifact hiện tại. <br>

Không được mặc định file khác hoặc artifact tương lai có cùng physical ordering. <br>

---

# 10. M2 Gate

## G01 — Đã hiểu mức mất cân bằng target chưa?

`PASS` <br>

Fraud khoảng `0.122%`, tương đương khoảng `818.5 non-fraud / 1 fraud`. <br>

## G02 — Đã hiểu temporal behavior và các giai đoạn bất thường chưa?

`PASS` <br>

Fraud thay đổi mạnh theo thời gian; từ `11/2019 → 02/2020` có `0 fraud`. <br>

## G03 — Đã hiểu Amount, negative và zero Amount chưa?

`PASS` <br>

Amount lệch phải, tail dài; negative khoảng `5.10%`; zero khoảng `0.083%`. <br>

## G04 — Đã hiểu Use Chip và quan hệ với fraud chưa?

`PASS` <br>

Ba transaction mode đều có support lớn; association với fraud tồn tại nhưng không ổn định theo thời gian. <br>

## G05 — Đã hiểu MCC và long-tail chưa?

`PASS` <br>

Có `109 MCC`, với concentration + long-tail và fraud association không đồng đều. <br>

## G06 — Đã hiểu missing State / Zip có ngẫu nhiên hay không?

`PASS` <br>

Không. Missing chủ yếu mang tính structural; Online Transaction thiếu State / Zip 100%. <br>

## G07 — Đã hiểu các vấn đề của location field chưa?

`PASS` <br>

Location có cardinality cao, structural missingness, nhiều semantic mechanism và shortcut risk. <br>

## G08 — Đã hiểu rủi ro của raw identifier chưa?

`PASS` <br>

User / Card / Merchant Name không được dùng raw trực tiếp làm classifier feature. <br>

## G09 — User/Card history có đủ sâu không?

`PASS` <br>

Có. Median Card có `2,602 transaction`, active span gần `8.7 năm`. <br>

## G10 — Cold-start có chiếm tỷ lệ lớn không?

`PASS` <br>

Không. Strict card-level cold-start khoảng `0.0253%`; new-card transaction 2019 khoảng `1.0757%`. <br>

## G11 — Historical feature có thể tính đúng causal không?

`PASS` <br>

Có. Bốn prototype behavioral feature đã được tính strict-causal trên toàn artifact. <br>

## G12 — Có bằng chứng về full dataset và temporal subset không?

`PASS` <br>

Có. Recent subsets vẫn giữ hàng triệu transaction và hàng nghìn fraud. <br>

## G13 — Đã xác định rõ việc chuyển sang M3 và M4 chưa?

`PASS` <br>

M3 nhận temporal split / evaluation / training-window experiment. <br>
M4 nhận preprocessing / representation / causal feature engineering. <br>

## G14 — Có bước EDA nào phá guardrail temporal/leakage không?

`PASS` <br>

Không phát hiện. Historical prototype tuân thủ `timestamp(history) < timestamp(current transaction)`. <br>

`M2 Gate: PASS` <br>

---

# 11. Quyết định cuối Milestone 2

`Decision ID: M2-FINAL-D01` <br>

`Hiểu dataset: ĐỦ` <br>
`Hiểu target và temporal behavior: ĐỦ` <br>
`Hiểu chất lượng dữ liệu: ĐỦ` <br>
`Hiểu phân bố feature: ĐỦ` <br>
`Hiểu feature-target relationship: ĐỦ, CẦN CẢNH GIÁC TEMPORAL INSTABILITY` <br>
`Hiểu entity history: ĐỦ` <br>
`Khả năng xây behavioral feature: ĐÃ XÁC NHẬN` <br>
`Khả năng dùng recent temporal subset: ĐÃ XÁC NHẬN` <br>
`Blocking issue: NONE` <br>
`M2 Gate: PASS` <br>
`Milestone 2 Status: PASS WITH FINDINGS` <br>

---

# 12. Kết luận Milestone 2

Milestone 2 đã hoàn thành mục tiêu `hiểu dataset trước khi xây model`. <br>

EDA không phát hiện blocking issue buộc project phải thay dataset hoặc dừng đề tài. <br>

Điểm quan trọng nhất không phải là tìm feature có fraud rate cao nhất, mà là hiểu rằng: <br>

`association mạnh trên toàn lịch sử không đồng nghĩa association ổn định trong tương lai`. <br>

Vì vậy M3 phải đặt `temporal evaluation` ở trung tâm của thiết kế thí nghiệm. <br>

M2 cũng xác nhận User/Card history đủ sâu và behavioral feature có thể được tính theo strict causal rule với chi phí chấp nhận được. <br>

Do đó M4 có cơ sở để triển khai feature engineering lịch sử thay vì chỉ sử dụng raw transaction features. <br>

Recent temporal subsets vẫn giữ đủ dữ liệu và fraud cho experiment, trong khi older transaction vẫn có thể được sử dụng làm history warm-up. <br>

Điều này cho phép project tách hai vai trò: <br>
`dữ liệu dùng để tạo historical context` <br>
và <br>
`dữ liệu dùng để fit classifier`. <br>

Các quyết định về final split, final training window, final feature set, encoding, class-imbalance strategy, model family và threshold được cố ý để lại cho các milestone thích hợp thay vì khóa bằng EDA. <br>

`Milestone 2 chính thức đủ điều kiện chuyển sang giai đoạn thiết kế experiment và preprocessing.` <br>

---

# 13. Bước tiếp theo

`M3` <br>
→ thiết kế experiment, temporal split, evaluation protocol và training-window comparison. <br>

`M4` <br>
→ preprocessing, representation và causal feature engineering dựa trên các guardrail đã khóa ở M1–M2. <br>
