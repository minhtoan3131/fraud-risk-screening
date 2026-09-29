# Kế hoạch thực hiện M2 - EDA

# 1. Mục tiêu chính thức của M2

Tôi đề xuất khóa mục tiêu M2 như sau:

> **Khám phá có hệ thống dataset IBM đã được chọn để hiểu phân bố dữ liệu, sự thay đổi theo thời gian, cấu trúc của target, đặc điểm của các feature candidate, các vấn đề chất lượng dữ liệu và khả năng xây dựng feature hành vi; từ đó tạo bằng chứng đầu vào cho M3 — Experiment Design và M4 — Preprocessing.**

Điểm rất quan trọng là M2 chưa có nhiệm vụ tìm “feature tốt nhất” hay “model tốt nhất”.

Trong quy trình tổng thể của project, sản phẩm chính của M2 được xác định là **Notebook EDA + kết luận**, còn train/test strategy, metric strategy và baseline thuộc M3. 

Vì vậy M2 sẽ được phép trả lời những câu như:

“Amount phân bố thế nào?”, “fraud thay đổi ra sao theo năm?”, “Online Transaction có đặc điểm missing location gì?”, “MCC nào phổ biến?”, “các Card có đủ lịch sử để tạo behavioral feature không?”

Nhưng chưa được chốt kiểu:

“Train = 2015–2018, test = 2019”, “dùng median imputation”, “dùng OneHotEncoder”, “dùng SMOTE”, “Logistic Regression là model cuối”.

Những quyết định đó phải dựa trên bằng chứng M2 rồi mới khóa ở milestone thích hợp.

# 2. Các ranh giới M2 phải giữ nguyên

Trước khi bắt đầu notebook, tôi đề nghị đặt một cell Markdown đầu notebook tên là `M2 Guardrails`, vì đây là những thứ M2 không được vô tình phá.

Các quyết định M1.8 đã khóa gồm: `User`, `Card`, `Merchant Name` là identifier/grouping key và không được dùng trực tiếp làm input classifier; `Errors?` bị loại khỏi Model V1 do chưa chứng minh prediction-time availability; `Is Fraud?` chỉ là target.  

Ngoài ra phải giữ nguyên các guardrail sau: không random split đơn giản làm evaluation chính; historical feature chỉ dùng transaction quá khứ; preprocessing học tham số chỉ được fit trên train; không dùng 2020 làm final fraud test; không coi missing location là lỗi ngẫu nhiên; không tự `abs()`, drop hay clip negative Amount; và không tuyên bố kết quả đại diện cho fraud detection của ngân hàng thật. 

Nói cách khác, M2 có thể **phân tích** `User`, `Card`, `Merchant Name` để hiểu cấu trúc lịch sử hoặc rủi ro memorization, nhưng không được vì thấy chúng “dự đoán fraud tốt” mà đổi quyết định và đưa raw ID vào model.

# 3. Cấu trúc M2 tôi đề xuất

Tôi đề nghị chia M2 thành **9 bước**, từ `M2.1` đến `M2.9`.

| Bước | Nội dung | Câu hỏi trung tâm | Đầu ra chính |
|---|---|---|---|
| M2.1 | Khóa EDA protocol và môi trường | Chúng ta sẽ khám phá dữ liệu như thế nào để kết quả tái hiện được? | EDA protocol + notebook skeleton |
| M2.2 | EDA cấu trúc và tính toàn vẹn | File dùng cho M2 có đúng artifact đã audit không? Representation phân tích có đúng không? | Structural EDA record |
| M2.3 | Phân tích target và chiều thời gian | Fraud phân bố thế nào và thay đổi ra sao theo thời gian? | Target/temporal findings |
| M2.4 | EDA từng feature giao dịch | Amount, Use Chip, MCC, thời gian, location đang phân bố thế nào? | Univariate findings |
| M2.5 | Phân tích chất lượng và missingness | Negative Amount, zero Amount, missing location, duplicate có pattern gì? | Data-quality findings |
| M2.6 | Quan hệ feature ↔ target | Các nhóm giao dịch nào có fraud rate khác nhau và mức bằng chứng có đủ không? | Bivariate/target findings |
| M2.7 | Phân tích entity và lịch sử | User/Card có đủ lịch sử để tạo behavioral feature? Cold-start lớn đến đâu? | History feasibility report |
| M2.8 | Khảo sát candidate derived features + temporal windows | Feature hành vi nào khả thi? Full dataset hay temporal subset hợp lý hơn cho experiment? | Evidence cho M3/M4 |
| M2.9 | Tổng hợp kết luận + M2 Gate | EDA đã cho ta biết gì và milestone sau phải quyết định gì? | M2 EDA Report + Decision/Handoff Log |

# 4. M2.1 — Khóa EDA protocol trước khi phân tích

Đây là bước ngắn nhưng rất quan trọng.

Dataset có hơn 24 triệu dòng, nên nếu cứ thử lệnh trên toàn bộ dữ liệu một cách tùy ý, notebook sẽ nhanh chóng trở nên nặng, khó tái hiện và dễ có tình trạng biểu đồ này chạy trên full dataset, biểu đồ kia chạy trên sample mà không ghi chú.

Full dataset hiện được xác minh có 24.386.900 transaction. M1.8 cũng đã ghi nhận các cửa sổ nhỏ hơn vẫn giữ đủ fraud, chẳng hạn 2015–2019 có khoảng 8,58 triệu transaction và 11.693 fraud; 2018–2019 có khoảng 3,45 triệu transaction và 4.578 fraud. Việc dùng full hay temporal subset cho training chưa được khóa và phải dựa vào EDA cùng thiết kế evaluation. 

Ở M2.1, ta chưa chọn training subset. Ta chỉ thiết lập ba mức phân tích:

**Full scan** dùng cho các phép tổng hợp quan trọng mà cần con số chính xác: số row, class count, fraud theo năm, missing count, category frequency...

**Temporal/grouped aggregation** dùng khi không cần giữ toàn bộ row trong RAM, ví dụ fraud rate theo năm/MCC/Use Chip.

**Sample dùng để trực quan hóa** khi hàng triệu điểm không mang thêm giá trị. Sample phải được ghi rõ cách lấy; với fraud cực hiếm, không được lấy một sample nhỏ rồi vô tình gần như mất hết fraud.

Notebook cũng nên ghi version Python/Pandas, đường dẫn artifact, ngày chạy và SHA-256 đã khóa ở M1 để đảm bảo đang phân tích đúng file.

# 5. M2.2 — Kiểm tra representation phục vụ EDA

M1.5 đã audit schema rồi, nên M2 không cần làm lại Technical Audit. Nhưng chúng ta vẫn cần tạo representation phù hợp để phân tích.

Việc quan trọng nhất là ghép:

`Year + Month + Day + Time → Timestamp`

M1.8 đã xác định Timestamp là nền tảng cho sắp xếp lịch sử, temporal split, causal feature engineering và rolling windows. 

Tương tự, `Amount` cần được parse về số để phân tích, nhưng **giữ nguyên dấu âm**.

Ở bước này chúng ta nên xác nhận lại:

* Timestamp parse thành công bao nhiêu phần trăm.
* Amount parse thành công bao nhiêu phần trăm.
* Timestamp nhỏ nhất/lớn nhất.
* dữ liệu có được sắp theo thời gian hay không, nhưng không được giả định thứ tự raw CSV chính là causal order.
* số lượng transaction theo năm/tháng.
* category của `Use Chip`.
* cardinality thực tế của `MCC`, City, State, Zip.

Mục đích là tạo một `df_analysis` hoặc quy trình đọc dữ liệu phục vụ EDA, **không phải tạo preprocessing pipeline cho model**.

# 6. M2.3 — Target và thời gian: phần quan trọng nhất của EDA này

Tôi muốn M2 ưu tiên phần này trước Amount hay MCC, bởi project này có hai đặc điểm cực mạnh: **class imbalance** và **temporal distribution shift**.

Full dataset chỉ có 29.757 fraud trên 24.386.900 transaction, tức khoảng 0,12202%. 

Nhưng con số tổng không đủ. M2 phải dựng được bức tranh:

`year → transactions → fraud count → fraud rate`

sau đó nếu có giá trị thì đi tiếp xuống tháng ở các năm gần thời gian evaluation.

M1 đã biết 2020 có 336.500 transaction nhưng 0 fraud; 2019 có 1.723.938 transaction, 2.087 fraud và fraud rate khoảng 0,12106%. Vì thế 2020 đã bị cấm dùng làm final fraud test, còn 2019 mới chỉ là **candidate** future holdout, chưa phải split cuối cùng. 

M2 cần làm sâu hơn để trả lời:

Fraud rate có xu hướng ổn định hay nhảy theo từng regime?

Các năm bất thường là những năm nào?

Số fraud tuyệt đối của từng năm có đủ để evaluation không?

Các pattern năm 2011, 2017, 2020 có thể ảnh hưởng thế nào đến lựa chọn training window?

Có lý do gì khiến dùng quá nhiều năm cũ làm training không còn hợp lý?

Kết luận M2 ở đây chưa phải “chọn split”, mà phải chuyển cho M3 một tập bằng chứng đủ mạnh để M3 khóa split.

# 7. M2.4 — EDA từng feature giao dịch

Sau khi hiểu target và time, chúng ta mới đi từng nhóm feature.

### Amount

Đây sẽ là feature số quan trọng nhất.

Không chỉ xem `mean`, `median`, `min`, `max`. Tôi muốn xem:

phân vị; phân bố Amount dương; Amount âm; Amount bằng 0; log-scale hoặc các amount band nếu cần quan sát; Amount theo fraud/non-fraud; Amount theo transaction mode.

M1 đã xác nhận 1.244.683 row có Amount âm, khoảng 5,10%, cùng 20.213 row bằng 0. Ý nghĩa chính xác của Amount âm vẫn chưa được nguồn giải thích đủ rõ, nên dấu âm phải được giữ nguyên và M2 phải khảo sát sâu hơn thay vì “sửa sạch”. 

### Use Chip

Phân bố ba nhóm:

`Swipe Transaction`, `Chip Transaction`, `Online Transaction`.

Sau đó phân tích số lượng và fraud rate từng nhóm.

Đặc biệt cần dùng nó để giải thích missing location, chứ không nhìn missing State/Zip riêng lẻ.

### MCC

MCC phải được xem như category chứ không phải số có độ lớn. 

Ta cần biết:

bao nhiêu MCC, distribution long-tail ra sao, MCC nào nhiều transaction nhất, fraud distribution theo MCC, category có fraud cao nhưng chỉ có vài transaction hay thực sự có support đủ lớn.

Điểm “support đủ lớn” rất quan trọng. Không được nhìn một MCC có fraud rate 10% từ 10 transaction rồi tuyên bố đó là nhóm cực kỳ nguy hiểm.

### Time-derived variables

Từ Timestamp ta có thể phân tích ở mức EDA:

`hour_of_day`, `day_of_week`, `month`, `year`.

Đây mới là candidate phân tích; M2 chưa cam kết tất cả sẽ vào model.

# 8. M2.5 — Missingness và các vấn đề chất lượng dữ liệu

Đây là chỗ cần phân biệt EDA với cleaning.

Tài liệu học của chúng ta nêu khá rõ: EDA tập trung vào **hiểu và phát hiện**, còn cleaning là sửa hoặc chuẩn hóa; thứ tự phù hợp là quan sát → ghi nhận → xác minh → quyết định xử lý. 

Vì vậy ở M2 ta cần phân tích, không “chữa” ngay.

Đối với `Merchant State` và `Zip`, M1 đã biết Online Transaction thiếu 100% hai trường này, nghĩa là đây chủ yếu là **structural missingness**, không phải missing ngẫu nhiên. 

M2 nên tạo một phân tích kiểu:

`Use Chip × State missing`, `Use Chip × Zip missing`, rồi thêm `fraud/non-fraud`.

Mục tiêu là xác định rõ:

missingness phản ánh kênh giao dịch đến đâu;

nếu tạo missing indicator sau này có bị trùng thông tin quá mạnh với `Use Chip` không;

nhóm Online có fraud rate khác nhóm physical đến đâu;

ta có nên giữ City/State/Zip cho Model V1 hay cardinality/missingness khiến một số field quá phức tạp.

Cũng ở M2.5, ta nên kiểm tra 66 duplicate extra rows đã phát hiện ở M1, nhưng chưa xóa ngay. Ta cần xem duplicate nằm ở thời gian nào, có fraud hay không, có thể là duplicate thật của generator hay chỉ exact identical transaction hợp lệ.

# 9. M2.6 — Quan hệ giữa feature và target

Đây là phần rất dễ biến thành “săn feature có fraud rate cao”, nên cần kiểm soát.

Với mỗi phân tích target, tôi đề nghị luôn hiển thị **cả số mẫu và fraud rate**.

Ví dụ không chỉ:

`MCC X → fraud rate 3%`

mà phải có:

`transaction count`, `fraud count`, `fraud rate`.

Các phân tích chính nên gồm:

Amount band ↔ fraud;

Use Chip ↔ fraud;

MCC ↔ fraud;

hour/day-of-week/month ↔ fraud;

location availability ↔ fraud;

các trường location ↔ fraud ở mức tổng hợp hợp lý.

`Merchant Name` có thể được dùng để **audit pattern**, vì M1 đã phát hiện nhiều merchant có fraud concentration rất mạnh, nhưng không được đưa raw Merchant Name vào danh sách feature model. Mục đích của phân tích merchant là củng cố hiểu biết về synthetic bias/memorization, không phải tìm cách khai thác shortcut đó.

# 10. M2.7 — Phân tích User/Card history

Đây là phần tôi xem là đặc trưng nhất của dataset IBM và là lý do chúng ta chọn nó.

`User`, `Card`, `Merchant Name` không được đưa trực tiếp vào classifier, nhưng M1.8 cho phép dùng chúng làm key để group lịch sử và tạo behavioral features. 

M2 cần trả lời:

Một User có bao nhiêu transaction?

Một User+Card có bao nhiêu transaction?

Độ dài lịch sử theo card là bao nhiêu?

Khoảng thời gian hoạt động của card?

Có bao nhiêu transaction xảy ra khi card chưa có lịch sử trước đó?

2019 có bao nhiêu new user/new card?

Có đủ history để rolling-window feature có ý nghĩa không?

M1 đã chỉ ra ở stress test 2019, new user chỉ khoảng 0,1888%, new card khoảng 1,0757%, nên evaluation về sau chủ yếu phản ánh giao dịch mới của entity đã có lịch sử. 

M2 cần biến nhận xét này thành một phần EDA rõ ràng để sau này khi vấn đáp bạn có thể nói chính xác model của mình mạnh/yếu ở bối cảnh nào.

# 11. M2.8 — Khảo sát behavioral feature và temporal window

Tôi không đề nghị tạo toàn bộ feature engineering ngay trong M2. Nhưng M2 nên có một **feasibility study nhỏ**.

M1.8 đã cho phép nghiên cứu các candidate như:

`time_since_previous_transaction`;

`transactions_last_10m`;

`transactions_last_1h`;

`amount_sum_last_1h`;

`previous_amount_mean/median`;

`amount_vs_previous_history`;

`is_new_merchant`;

`is_new_mcc`;

`is_new_city/state`. 

Điều bắt buộc là mọi feature cho transaction T chỉ được dùng dữ liệu có `timestamp < timestamp(T)`. Không được lấy mean toàn dataset rồi gắn ngược cho quá khứ. 

Ở M2, tôi đề xuất chỉ prototype 3–4 feature đại diện, chẳng hạn:

`time_since_previous_transaction`;

`transactions_last_1h`;

`amount_vs_previous_history`;

`is_new_merchant`.

Mục đích không phải chứng minh chúng cải thiện model — vì chưa train model — mà để biết:

có tính được đúng causal không;

chi phí tính toán có chấp nhận được không;

bao nhiêu transaction không có history;

phân bố feature có quá cực đoan không;

candidate này có đáng mang sang M4 hay không.

Đồng thời M2.8 phải so sánh quy mô các cửa sổ thời gian khả thi, ví dụ full data, 2015–2019 và 2018–2019. Nhưng chỉ đưa ra **khuyến nghị cho M3**, chưa khóa split.

# 12. M2.9 — Kết luận phải biến EDA thành quyết định

Notebook không nên kết thúc ở biểu đồ cuối cùng.

Mỗi nhóm EDA phải sinh ra một bản ghi dạng:

> Quan sát → bằng chứng → ý nghĩa → việc chuyển tiếp.

Ví dụ:

> Fraud chỉ chiếm khoảng 0,122% toàn dữ liệu và thay đổi mạnh theo năm. Vì vậy Accuracy không thể dùng làm thước đo chính, evaluation phải theo thời gian và M3 cần chọn một temporal holdout có đủ positive samples.

Hoặc:

> Missing State/Zip gần như gắn với Online Transaction. Vì vậy không được drop missing row như dữ liệu hỏng; M4 phải thiết kế representation có ý thức về structural missingness.

Hoặc:

> Negative Amount chiếm khoảng 5,1% dataset và ý nghĩa nghiệp vụ chưa được xác minh đầy đủ. Vì vậy M2 giữ nguyên giá trị; M4 chỉ được biến đổi nếu có lý do rõ ràng.

Đây chính là thứ làm cho notebook của chúng ta trở thành EDA thực sự thay vì tập hợp biểu đồ.

# 13. Đầu ra chính thức của M2

Tôi đề nghị cuối M2 phải có 4 artifact chính.

**`02_eda.ipynb`** là notebook chính, chứa toàn bộ phép kiểm tra, biểu đồ, bảng và diễn giải.

**`M2_EDA_Findings.md`** là bản kết luận cô đọng hơn notebook, dùng để sau này chuyển vào báo cáo.

**`M2_Decision_Log.md`** ghi rõ những thứ đã biết sau EDA và những quyết định vẫn cố ý để M3/M4 xử lý.

**`reports/figures/m2/`** chỉ chứa các biểu đồ thực sự có giá trị cho báo cáo, không lưu hàng chục ảnh gần giống nhau.

Cấu trúc project trước đây cũng đã dự kiến riêng `02_eda.ipynb` và thư mục `reports/figures`, nên cách tổ chức này phù hợp với kiến trúc project đã định hướng. 

# 14. Gate để M2 được coi là DONE

Tôi đề xuất M2 chỉ được PASS khi bạn có thể tự trả lời được các câu sau mà không cần nhìn code:

1. Target mất cân bằng đến mức nào, cả toàn dataset và theo thời gian?
2. Những năm nào có distribution bất thường, và vì sao 2020 không phù hợp làm final fraud test?
3. Amount phân bố ra sao? Negative và zero Amount xuất hiện thế nào?
4. `Use Chip` có những nhóm nào và mỗi nhóm phân bố/fraud ra sao?
5. MCC có cardinality và long-tail như thế nào?
6. Missing State/Zip có thực sự là missing ngẫu nhiên không?
7. Các location field nào có khả năng còn hữu ích và field nào có thể quá phức tạp?
8. Raw identifier có pattern gì khiến memorization là rủi ro?
9. User/Card history có đủ sâu để tạo behavioral feature không?
10. Cold-start chiếm tỷ lệ đáng kể hay chỉ là phần nhỏ?
11. Ít nhất một vài historical feature candidate có thể tính causal đúng cách không?
12. Có bằng chứng nào ủng hộ dùng full dataset hay temporal subset cho các experiment sau?
13. Những vấn đề nào phải chuyển cho M3 và những vấn đề nào phải chuyển cho M4?
14. Có bước EDA nào vô tình sử dụng future data hoặc phá guardrail của M1 không?

Nếu còn không trả lời được các câu về target distribution, temporal behavior, Amount, missingness, entity history hoặc computational feasibility thì tôi chưa muốn đánh dấu M2 hoàn thành.

# 15. Trình tự thực hiện thực tế

Tôi đề nghị chúng ta không viết toàn bộ notebook một lượt.

Nhịp làm hợp lý nhất là:

**Buổi 1:** M2.1–M2.2, dựng notebook và representation phân tích.

**Buổi 2:** M2.3, target + temporal EDA. Đây là buổi quan trọng nhất.

**Buổi 3:** M2.4, Amount + Use Chip + MCC + temporal variables.

**Buổi 4:** M2.5, missingness + duplicate + negative/zero Amount.

**Buổi 5:** M2.6, quan hệ feature–target.

**Buổi 6:** M2.7, User/Card/history/cold-start.

**Buổi 7:** M2.8, behavioral-feature feasibility + temporal-window comparison.

**Buổi 8:** M2.9, tổng hợp findings, decision log và M2 Gate.

Tôi muốn giữ đúng mô hình đã thành công ở M1: **mỗi bước có mục tiêu → chạy notebook → xem output thật → chúng ta phân tích → chỉ sau đó mới chốt kết luận của bước đó**. Như vậy khi kết thúc M2, M3 sẽ nhận được bằng chứng đủ rõ để thiết kế thí nghiệm, chứ không phải tự đoán train/test split, metric hay baseline.



