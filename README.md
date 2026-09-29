# AI Transaction Fraud Risk Screening

[![Python Version](https://img.shields.io/badge/Python-%3E%3D3.11-blue.svg)](https://www.python.org/)
[![Tested Environment](https://img.shields.io/badge/Tested%20on-Python%203.14.6-informational.svg)](runtime_environment.json)
[![Project Version](https://img.shields.io/badge/Version-v0.1.0-blueviolet.svg)](release_manifest.json)
[![Framework](https://img.shields.io/badge/ML-Scikit--Learn%20%7C%20NumPy-orange.svg)](https://scikit-learn.org/)
[![Tests Status](https://img.shields.io/badge/Tests-192%20Passed-brightgreen.svg)](#7-kiểm-thử--đảm-bảo-chất-lượng-qa)
[![Artifact](https://img.shields.io/badge/Artifact-SHA256%20Locked-success.svg)](#8-quản-lý-artifact--tính-bất-biến)

> **Dự án Machine Learning hỗ trợ sàng lọc rủi ro gian lận giao dịch tài chính**, áp dụng kiến trúc phân tách trách nhiệm (Separation of Concerns), kiểm soát rò rỉ dữ liệu theo trục thời gian (*Strict-Prior Temporal Integrity*) và đóng gói hoàn chỉnh thành ứng dụng dòng lệnh (CLI).

---

## Mục lục

1. [Tổng quan bài toán & Bộ dữ liệu thực nghiệm](#1-tổng-quan-bài-toán--bộ-dữ-liệu-thực-nghiệm)
2. [Kiến trúc phân tách ba vùng](#2-kiến-trúc-phân-tách-ba-vùng)
3. [Luồng suy luận, Đặc trưng & Lựa chọn mô hình](#3-luồng-suy-luận-đặc-trưng--lựa-chọn-mô-hình)
4. [Kết quả thực nghiệm trên Final Test](#4-kết-quả-thực-nghiệm-trên-final-test)
5. [Các giới hạn kỹ thuật đã biết](#5-các-giới-hạn-kỹ-thuật-đã-biết)
6. [Hướng dẫn cài đặt & Khởi chạy nhanh](#6-hướng-dẫn-cài-đặt--khởi-chạy-nhanh)
7. [Kiểm thử & Đảm bảo chất lượng (QA)](#7-kiểm-thử--đảm-bảo-chất-lượng-qa)
8. [Quản lý Artifact & Tính bất biến](#8-quản-lý-artifact--tính-bất-biến)
9. [Diễn giải kết quả & Đạo đức AI](#9-diễn-giải-kết-quả--đạo-đức-ai)
10. [Cấu trúc mã nguồn](#10-cấu-trúc-mã-nguồn)

---

## 1. Tổng quan bài toán & Bộ dữ liệu thực nghiệm

Trong lĩnh vực thanh toán điện tử và ngân hàng số, sàng lọc giao dịch nghi ngờ gian lận là bài toán mang tính cấp thiết nhưng đi kèm các thách thức đặc thù:
- **Mất cân bằng lớp cực đoan (Extreme Class Imbalance)**: Tỷ lệ giao dịch gian lận thực tế chỉ chiếm một phần rất nhỏ trên tổng số giao dịch phát sinh.
- **Rò rỉ thông tin theo thời gian (Lookahead Bias / Data Leakage)**: Rất dễ xảy ra nếu sử dụng thông tin tương lai để dự đoán quá khứ hoặc chuẩn hóa dữ liệu xuyên suốt toàn bộ tập dữ liệu.
- **Cold-Start**: Khi thẻ mới phát sinh giao dịch đầu tiên hoặc chưa có lịch sử, hệ thống vẫn phải đưa ra đánh giá an toàn dựa trên thông tin giao dịch tức thời.

### Thông tin bộ dữ liệu thực nghiệm:
- **Tên bộ dữ liệu**: IBM TabFormer Credit Card Transaction Dataset.
- **Quy mô**: 24.386.900 giao dịch (1991-2020) với 15 thuộc tính ban đầu.
- **Bản chất dữ liệu**: Dữ liệu mô phỏng tổng hợp (synthetic dataset) mô phỏng hành vi chi tiêu thẻ tín dụng thực tế.
- **Tỷ lệ gian lận**: Khoảng **~0.12%** (29.757 giao dịch gian lận trên toàn bộ 24,38 triệu dòng).
- **Phân chia dữ liệu theo thời gian (Temporal Split)**:
  - *Tập huấn luyện (Train)*: Giai đoạn 2017 – Quý 1/2018.
  - *Tập thẩm định (Validation)*: Quý tiếp theo trong năm 2018 (dùng để chọn mô hình và ngưỡng phân định).
  - *Tập kiểm thử bảo vệ (Protected Final Test)*: Quý độc lập trong tương lai (năm 2019), hoàn toàn được cô lập và chỉ dùng cho đánh giá thực nghiệm cuối cùng.

---

## 2. Kiến trúc phân tách ba vùng

Dự án được cấu trúc theo nguyên tắc phân tách trách nhiệm (Separation of Concerns) nhằm bảo đảm tính độc lập giữa không gian nghiên cứu và mã nguồn ứng dụng:

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        AI FRAUD RISK SCREENING                         │
└────────────────────────────────────────────────────────────────────────┘
          │                                           │
          ▼                                           ▼
┌──────────────────┐                       ┌──────────────────┐
│    research/     │                       │ final_pipeline/  │
│                  │                       │                  │
│ • EDA & Audit    │                       │ • ML Package     │
│ • Thí nghiệm M0-8│                       │ • Frozen Artifact│
│ • Temporal CV    │                       │ • Verification   │
│ • Evidence logs  │                       │ • Rebuild Gate   │
└──────────────────┘                       └──────────────────┘
         ▲                                            │
         │ (Tách rời thực nghiệm, không tải runtime) │ (Export API)
         └────────────────────────────────────┐       ▼
                                       ┌──────────────────┐
                                       │   application/   │
                                       │                  │
                                       │ • CLI Interface  │
                                       │ • Demo Scenarios │
                                       │ • Pretty Report  │
                                       └──────────────────┘
```

1. **`research/` (Khu vực nghiên cứu & kiểm toán thực nghiệm)**: Lưu trữ toàn bộ notebook thí nghiệm (M0–M8), phân tích dữ liệu (EDA), kiểm toán dữ liệu và các biên bản đối chứng. Khu vực này không có phụ thuộc runtime vào ứng dụng.
2. **`final_pipeline/` (Lõi Machine Learning sản phẩm)**: Chứa thư viện Python chính thức `fraud_screening`, bộ tiền xử lý và mô hình đã được cố định (*frozen artifacts*), mã kiểm thử và cơ chế rebuild có kiểm soát.
3. **`application/` (Ứng dụng phía người dùng)**: Cung cấp giao diện dòng lệnh `fraud-screening-app` và các kịch bản demo mẫu, phụ thuộc một chiều vào `final_pipeline` và không gọi mã từ notebook.

---

## 3. Luồng suy luận, Đặc trưng & Lựa chọn mô hình

```text
[Giao dịch đầu vào (JSON)] ───┐
                              ▼
[Lịch sử thẻ (Tùy chọn)]   ───► [Bộ tiền kiểm & Lọc Strict-Prior]
                                              │
                                              ▼
                              [Trích xuất 10 Đặc trưng Ngữ nghĩa]
                              • 5 đặc trưng hiện thời + 5 đặc trưng hành vi
                                              │
                                              ▼
                              [Bộ tiền xử lý chuẩn hóa (Frozen)]
                              • One-Hot Encoding + Điền khuyết cấu trúc
                              • Mã hóa thành vector 47 chiều
                                              │
                                              ▼
                              [Mô hình Random Forest chính thức]
                              (RF-REF-100-GINI-SQRT-UNPRUNED-CW)
                                              │
                                              ▼
                              [Điểm số rủi ro (Risk Score: 0.0 – 1.0)]
                                              │
                                              ▼
                              [Ngưỡng sàng lọc cố định (Score > 0.50)]
                                              │
                                              ▼
                              [Báo cáo kết quả sàng lọc rủi ro]
```

### Danh mục 10 đặc trưng ngữ nghĩa:
Mọi đặc trưng hành vi tích lũy đều tuân thủ nguyên tắc **Strict-Prior**: chỉ tính toán dựa trên các giao dịch xảy ra **hoàn toàn trước** thời điểm giao dịch hiện tại (`Timestamp_prior < Timestamp_current`).

| STT | Tên đặc trưng | Nhóm | Kiểu dữ liệu | Ý nghĩa nghiệp vụ |
| :---: | :--- | :---: | :---: | :--- |
| 1 | `amount_numeric` | Hiện thời | Số thực | Số tiền phát sinh của giao dịch hiện tại (USD) |
| 2 | `transaction_mode` | Hiện thời | Phân loại | Phương thức thực hiện (Chip Transaction, Swipe, Online...) |
| 3 | `location_state` | Hiện thời | Phân loại | Bang / địa phương nơi đặt điểm chấp nhận thanh toán |
| 4 | `hour_of_day` | Hiện thời | Số nguyên | Khung giờ thực hiện giao dịch trong ngày (0 – 23) |
| 5 | `day_of_week` | Hiện thời | Số nguyên | Ngày trong tuần (0: Thứ 2 – 6: Chủ nhật) |
| 6 | `time_since_previous_transaction_min` | Hành vi | Số thực | Khoảng cách thời gian (phút) so với giao dịch trước gần nhất |
| 7 | `transactions_last_1h` | Hành vi | Số nguyên | Tần suất giao dịch của thẻ trong vòng 1 giờ gần nhất $[T-1h, T)$ |
| 8 | `amount_minus_previous_mean` | Hành vi | Số thực | Độ lệch giữa số tiền hiện tại so với mức chi tiêu trung bình quá khứ |
| 9 | `is_new_merchant` | Hành vi | Nhị phân | Cửa hàng/đơn vị này thẻ đã từng thực hiện giao dịch trước đây chưa |
| 10 | `has_prior_card_history` | Hành vi | Nhị phân | Trạng thái có lịch sử hay là giao dịch đầu tiên (phục vụ Cold-Start) |

### Lựa chọn mô hình Machine Learning chính thức:
Trong quá trình thực nghiệm, dự án đã tiến hành thử nghiệm và so sánh giữa 3 họ thuật toán phân lớp phổ biến trên cùng tập dữ liệu thẩm định độc lập (*External Validation*):

| Họ thuật toán | F1-Score | Recall | Precision | Số cảnh báo sai (FP) | Đánh giá & Quyết định |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Logistic Regression** | 0,3414 | 0,2490 | **0,5424** | **221** | Độ chính xác tốt nhưng bỏ sót quá nhiều gian lận (Recall chỉ ~25%) |
| **Decision Tree** | 0,3450 | 0,4401 | 0,2837 | 1.169 | Bắt được nhiều nhưng báo động giả quá mức (1.169 ca cảnh báo sai) |
| **Random Forest** *(Được chọn)* | **0,4091** | **0,4439** | 0,3794 | 764 | **Chiến thắng áp đảo**: F1 cao nhất, cân bằng tối ưu giữa phát hiện và cảnh báo sai |

**Căn cứ lựa chọn Random Forest:**
- **Xử lý phi tuyến tính**: Khả năng nắm bắt các mối quan hệ phi tuyến phức tạp giữa số tiền chi tiêu và chuỗi hành vi thời gian mà Logistic Regression không mô hình hóa được.
- **Ổn định và giảm phương sai**: Nhờ cơ chế kết hợp (Ensemble Bagging) của 100 cây quyết định, Random Forest kháng nhiễu vượt trội, giảm hẳn tình trạng báo động giả nghiêm trọng của cây quyết định đơn lẻ (Decision Tree).
- **Cấu hình mô hình chính thức**: `RF-REF-100-GINI-SQRT-UNPRUNED-CW` (`n_estimators=100`, tiêu chuẩn phân tách `criterion='gini'`, số thuộc tính ngẫu nhiên `max_features='sqrt'`, xử lý mất cân bằng `class_weight='balanced'`).

---

## 4. Kết quả thực nghiệm trên Final Test

Mô hình được đánh giá trên tập kiểm thử độc lập trong tương lai (**Protected Final Test** — dữ liệu năm 2019) với ngưỡng sàng lọc cố định `risk_score > 0.50`:

### Bảng chỉ số phân loại:
| Chỉ số thực nghiệm | Giá trị | Nhận định chuyên môn |
| :--- | :---: | :--- |
| **F1-Score (fraud)** | **0,3557** | Trung bình điều hòa giữa Precision và Recall trên lớp thiểu số |
| **Recall (fraud)** | **0,3797** | Phát hiện được ~38% số vụ gian lận thực tế |
| **Precision (fraud)** | **0,3345** | Cứ ~3 cảnh báo đưa ra thì có 1 giao dịch gian lận thực sự |
| **Accuracy** *(tham khảo)* | *0,9980* | **Không dùng để đo lường năng lực** (do lớp âm chiếm 99.88%) |

### Ma trận nhầm lẫn (Confusion Matrix):
| Thực tế \ Dự đoán | Dự đoán: An toàn (0) | Dự đoán: Gian lận (1) |
| :--- | :---: | :---: |
| **Thực tế: An toàn (0)** | 721.138 (TN) | 782 (FP) |
| **Thực tế: Gian lận (1)** | 642 (FN) | 393 (TP) |

### Cách đọc kết quả và giá trị của thực nghiệm

Kết quả F1 = **0,3557** cho thấy mô hình vẫn còn hạn chế, nhưng cần được xem xét trong bối cảnh bài toán có mức mất cân bằng lớp rất lớn và được đánh giá theo trục thời gian nghiêm ngặt.

1. **Mô hình đã học được tín hiệu gian lận có ý nghĩa**

   Trong FINAL TEST, tỷ lệ gian lận cơ sở chỉ khoảng **0,143%**. Mô hình cảnh báo 1.175 giao dịch, trong đó có 393 giao dịch gian lận, tương ứng **Precision = 33,45%**. Như vậy, mật độ gian lận trong nhóm cảnh báo cao hơn khoảng **233,6 lần** so với tỷ lệ cơ sở của cùng tập kiểm thử.

   Kết quả này cho thấy mô hình không lựa chọn giao dịch một cách ngẫu nhiên mà đã học được các tín hiệu giúp tập trung trường hợp rủi ro vào một nhóm nhỏ hơn đáng kể.

2. **Có giá trị như một tầng ưu tiên rà soát**

   Mô hình chỉ cảnh báo **0,1625%** trong tổng số 722.955 giao dịch và phát hiện 393 trong 1.035 trường hợp gian lận, tương ứng **Recall = 37,97%**. Nếu dùng làm tầng ưu tiên ban đầu, hệ thống có thể giúp chuyên viên tập trung trước vào một tập ứng viên nhỏ hơn khoảng **99,84%** so với việc kiểm tra toàn bộ giao dịch.

   Tuy nhiên, mô hình vẫn bỏ sót 642 trường hợp gian lận. Vì vậy, đầu ra chỉ nên được dùng để hỗ trợ sàng lọc và xếp mức ưu tiên, không được hiểu rằng các giao dịch không bị cảnh báo đều an toàn. Giá trị kinh tế thực tế cũng cần được đánh giá thêm dựa trên chi phí rà soát, cảnh báo sai và thiệt hại do bỏ sót.

3. **Kết quả được đánh giá theo quy trình có kiểm soát**

   Mô hình sử dụng dữ liệu năm 2018 để huấn luyện, validation từ tháng 01 đến tháng 05/2019 và protected FINAL TEST từ tháng 06 đến tháng 10/2019. Mô hình, tiền xử lý và ngưỡng dự đoán đã được cố định trước khi mở FINAL TEST; kết quả kiểm thử không được dùng để điều chỉnh lại mô hình.

   F1 giảm từ **0,4091** trên validation xuống **0,3557** trên FINAL TEST, tương ứng mức giảm tương đối khoảng **13,07%**. Điều này cho thấy mô hình tổng quát hóa kém hơn trên giai đoạn thời gian muộn hơn, nhưng chưa đủ bằng chứng để khẳng định nguyên nhân cụ thể là concept drift hoặc covariate drift.

4. **F1 không phải tiêu chí duy nhất đánh giá toàn bộ dự án**

   Chất lượng dự án còn thể hiện ở việc kiểm soát rò rỉ dữ liệu theo thời gian, xây dựng đặc trưng chỉ từ lịch sử hợp lệ, bảo vệ FINAL TEST, kiểm tra tính toàn vẹn của artifact và đóng gói pipeline thành ứng dụng có thể chạy lại.
   
   Vì vậy, F1 = 0,3557 nên được hiểu là kết quả thực nghiệm trung thực của phiên bản mô hình hiện tại, không phải bằng chứng duy nhất để kết luận toàn bộ dự án có chất lượng cao hay thấp. Kết quả đồng thời xác nhận mô hình đã học được tín hiệu hữu ích và chỉ ra rõ những phần cần tiếp tục cải thiện.
---

## 5. Các giới hạn kỹ thuật đã biết

Nhằm phản ánh đúng bản chất của một dự án trong phạm vi bài tập lớn, dự án ghi nhận các giới hạn kỹ thuật sau:

1. **Tỷ lệ bỏ sót còn cao (False Negative ~62%)**: Với Recall đạt khoảng 38%, mô hình bỏ sót phần lớn các hành vi gian lận mới mẻ do sự biến đổi của phương thức giao dịch theo thời gian.
2. **Tính chất dữ liệu tổng hợp (Synthetic Data)**: Bộ dữ liệu IBM TabFormer được sinh từ mô hình thuật toán, chưa phản ánh toàn bộ sự phức tạp và các thủ đoạn tinh vi trong thực tế ngành tài chính.
3. **Độ nhạy khi Cold-Start**: Đối với thẻ mới hoàn toàn chưa có lịch sử, các đặc trưng hành vi nhận giá trị khuyết cấu trúc, khiến hệ thống chỉ dựa vào đặc trưng tức thời và giảm khả năng phát hiện sớm.
4. **Điểm số chưa qua cân chỉnh xác suất (Uncalibrated Score)**: Giá trị `risk_score` là điểm số phân lớp từ Random Forest, dùng để so sánh với ngưỡng cắt, không đồng nhất với xác suất rủi ro tuyệt đối ngoài đời thực.
5. **Phạm vi đồ án môn học**: Dự án được xây dựng dưới dạng ứng dụng dòng lệnh (CLI) phục vụ nghiên cứu và báo cáo học phần, chưa có thử nghiệm tải nặng (stress test) hay độ trễ cấp độ mili-giây của hệ thống ngân hàng thương mại.

---

## 6. Hướng dẫn cài đặt & Khởi chạy nhanh

### Yêu cầu môi trường
- **Python**: Phiên bản tối thiểu `>= 3.11` (bản phát hành hiện tại đã được xác minh thực tế trên Python 3.14.6).
- Hệ điều hành: macOS, Linux hoặc Windows.

### 6.1. Cài đặt các gói phần mềm

Từ thư mục gốc của dự án, cài đặt các thư viện phụ thuộc và hai gói nội bộ ở chế độ editable:

```bash
# 1. Cài đặt các thư viện phụ thuộc (scikit-learn, numpy, pandas, joblib, ...)
python -m pip install -r requirements.txt

# 2. Cài đặt gói lõi final_pipeline
python -m pip install --no-deps -e final_pipeline

# 3. Cài đặt gói ứng dụng application
python -m pip install --no-deps -e application
```

Xác nhận cài đặt thành công:
```bash
fraud-screening-app --version-info
```

---

### 6.2. Khởi chạy thử nghiệm các kịch bản mẫu

#### Kịch bản 1: Chạy demo kiểm tra tự động nhanh (Smoke Demo)
```bash
python -m fraud_screening_app.demo
```
*Kết quả mẫu:*
```text
========================================================================
FRAUD-RISK SCREENING DEMO
========================================================================
cold-start: risk_score=0.00000000, prediction=0, cold_start=True, warnings=['COLD_START_NO_STRICT_PRIOR_CARD_HISTORY']
explicit-history: risk_score=0.00000000, prediction=0, cold_start=False, warnings=[]
Demo result: PASS
```

#### Kịch bản 2: Sàng lọc ca giao dịch thông thường an toàn (True Negative Case)
Giao dịch mua hàng thông thường với 4,861 bản ghi lịch sử tham chiếu hợp lệ:
```bash
fraud-screening-app screen \
  --transaction-json application/examples/verified_cases/non_fraud_tn_transaction.json \
  --history-json application/examples/verified_cases/non_fraud_tn_history.json
```
*Kết quả mẫu:*
```text
========================================================================
TRANSACTION FRAUD-RISK SCREENING RESULT
========================================================================

[Transaction]
User/Card: 0/0
Date/Time: 2019-6-2 06:18
Amount: $141.72
Mode: Chip Transaction

[Screening]
Risk score: 0.00000000
Threshold: 0.5
Screening prediction: 0
Cold start: False
History records supplied: 4861
Warnings: NONE
Model id: RF-REF-100-GINI-SQRT-UNPRUNED-CW

Interpretation: this is a screening support result, not a final fraud accusation.
```

#### Kịch bản 3: Sàng lọc ca giao dịch có dấu hiệu gian lận (True Positive Case)
Giao dịch có dấu hiệu bất thường về số tiền, địa điểm và tần suất thanh toán:
```bash
fraud-screening-app screen \
  --transaction-json application/examples/verified_cases/fraud_tp_transaction.json \
  --history-json application/examples/verified_cases/fraud_tp_history.json
```
*Kết quả mẫu:*
```text
========================================================================
TRANSACTION FRAUD-RISK SCREENING RESULT
========================================================================

[Transaction]
User/Card: 41/1
Date/Time: 2019-8-22 12:49
Amount: $4.30
Mode: Chip Transaction

[Screening]
Risk score: 0.69000000
Threshold: 0.5
Screening prediction: 1
Cold start: False
History records supplied: 1378
Warnings: NONE
Model id: RF-REF-100-GINI-SQRT-UNPRUNED-CW

Interpretation: this is a screening support result, not a final fraud accusation.
```

#### Kịch bản 4: Sàng lọc giao dịch khi chưa có lịch sử thẻ (Cold-Start)
```bash
fraud-screening-app screen \
  --transaction-json application/examples/current_transaction.json
```
*Kết quả:* Hệ thống ghi nhận cảnh báo `COLD_START_NO_STRICT_PRIOR_CARD_HISTORY` và áp dụng tiền xử lý khuyết cấu trúc để đưa ra đánh giá.

---

## 7. Kiểm thử & Đảm bảo chất lượng (QA)

Hệ thống được đóng gói cùng bộ kiểm thử tự động gồm **192 bài kiểm thử (100% Pass)** bao quát các tầng chức năng:

### 1. Kiểm thử gói lõi Machine Learning (175 tests):
Kiểm tra tính đúng đắn của logic tiền xử lý, trích xuất đặc trưng hành vi, tính bất biến của model artifact và cơ chế chống rò rỉ dữ liệu:
```bash
PYTHONPATH=final_pipeline/src python -m unittest discover -s final_pipeline/tests -v
```

### 2. Kiểm thử gói ứng dụng người dùng (17 tests):
Kiểm tra tính toàn vẹn của giao diện dòng lệnh, bộ kiểm tra tính hợp lệ của dữ liệu đầu vào (Input Validation) và các ca kiểm thử biên:
```bash
PYTHONPATH=application/src:final_pipeline/src python -m unittest discover -s application/tests -v
```

---

## 8. Quản lý Artifact & Tính bất biến

Nhằm bảo đảm tính toàn vẹn của mô hình và tính tái lập thực nghiệm, dự án áp dụng cơ chế xác thực mã băm SHA-256 đối với các artifact chính thức:

- **Mô hình chính thức**: `final_pipeline/artifacts/official/model/model.joblib`
  - SHA-256: `61bdeeba5fd163cce8a5a9efa028a9cdc2f95c0796822a5222c0604c0302115f`
- **Bộ tiền xử lý chính thức**: `final_pipeline/artifacts/official/preprocessing/preprocessing_state.json`
  - SHA-256: `c1dc5486acdcd43f5f75fbd0a11bd7ad3a3b00761d50ae96b2b1fbe5b5d9ef98`
- **Manifest kiểm định**: `final_pipeline/artifacts/official/manifest/artifact_manifest.json`

> **Cơ chế xác thực**: Trình nạp mô hình (`InferenceService`) sẽ tự động tính toán mã băm SHA-256 của các file artifact và đối soát với `artifact_manifest.json` trước khi thực hiện suy luận. Nếu phát hiện tệp bị thay đổi hoặc không khớp mã băm, hệ thống sẽ từ chối khởi động.

Chi tiết về nguyên tắc dữ liệu và quy trình huấn luyện lại (rebuild) được trình bày tại [DATA_AND_ARTIFACTS.md](DATA_AND_ARTIFACTS.md).

---

## 9. Diễn giải kết quả

1. **Bản chất của `risk_score`**: Giá trị đầu ra là điểm ước lượng rủi ro tương đối của mô hình đối với lớp gian lận, dùng làm căn cứ sàng lọc và cảnh báo sớm, không đại diện cho xác suất gian lận tuyệt đối ngoài đời thực.
2. **Hỗ trợ quyết định (Decision Support System)**: Kết quả sàng lọc đóng vai trò hỗ trợ chuyên viên phân tích rủi ro ra quyết định (*Human-in-the-loop*). Quyết định xử lý giao dịch hoặc khóa thẻ cần được kết hợp với các biện pháp xác thực bổ sung (OTP, sinh trắc học) hoặc nghiệp vụ tra soát ngân hàng.
3. **Minh bạch hóa lý do cảnh báo**: Hệ thống cung cấp đầy đủ thông tin về số lượng bản ghi lịch sử tham chiếu, trạng thái cold-start và các cảnh báo bất thường để chuyên viên dễ dàng thẩm định.

---

## 10. Cấu trúc mã nguồn

```text
fraud-risk-screening/
│
├── final_pipeline/              # Lõi Machine Learning sản phẩm
│   ├── artifacts/official/      # Model và Preprocessing state cố định (khóa SHA-256)
│   ├── src/fraud_screening/     # Package Python xử lý đặc trưng & suy luận
│   └── tests/                   # 175 bài kiểm thử đơn vị & tích hợp
│
├── application/                 # Ứng dụng người dùng (CLI Demo)
│   ├── src/fraud_screening_app/ # Giao diện dòng lệnh & định dạng báo cáo
│   ├── examples/                # Dữ liệu giao dịch mẫu (Cold-start, TP, TN)
│   └── tests/                   # 17 bài kiểm thử chấp nhận (Acceptance Tests)
│
├── research/                    # Lịch sử nghiên cứu thực nghiệm
│   ├── notebooks/               # Sổ tay thí nghiệm từ M0 đến M8
│   └── finalization_evidence/   # Biên bản kiểm toán và đối chứng
│
├── requirements.txt             # Thư viện runtime tối thiểu
├── runtime_environment.json     # Thông tin môi trường thực thi đã ghi nhận
├── release_manifest.json        # Manifest định danh phiên bản v0.1.0
├── DATA_AND_ARTIFACTS.md        # Hướng dẫn chi tiết về dữ liệu & artifact
├── INSTALL.md                   # Hướng dẫn chi tiết cài đặt và kiểm thử
└── README.md                    # Tài liệu tổng quan dự án
```

---

*Bài tập lớn học phần Trí tuệ nhân tạo (IT3234) — Trường Đại học Công nghệ Đông Á (EAUT).*
