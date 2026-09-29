# Milestone 2 — Phân tích khám phá dữ liệu

`Project:` AI Transaction Fraud Risk Screening <br>
`Dataset:` IBM Synthetic Credit Card Transactions / TabFormer <br>
`File dữ liệu gốc:` card_transaction.v1.csv <br>

## Mục tiêu của Milestone 2

Milestone 2 thực hiện `EDA — Phân tích khám phá dữ liệu` trên dataset đã được lựa chọn và audit ở Milestone 1. <br>

Các nội dung chính cần làm rõ gồm: <br>
- phân bố của `target`; <br>
- cấu trúc và sự thay đổi của dữ liệu theo thời gian; <br>
- phân bố các feature giao dịch; <br>
- missing value và các vấn đề chất lượng dữ liệu; <br>
- mối quan hệ giữa feature và target; <br>
- cấu trúc lịch sử `User / Card`; <br>
- khả năng xây dựng behavioral feature; <br>
- các bằng chứng cần chuyển sang `M3 — Thiết kế thí nghiệm` và `M4 — Tiền xử lý dữ liệu`. <br>

## Ranh giới của M2

M2 tập trung vào `khám phá, kiểm tra và ghi nhận bằng chứng`. <br>

M2 chưa thực hiện: <br>
- huấn luyện model; <br>
- tuning hyperparameter; <br>
- chọn threshold cuối cùng; <br>
- resampling; <br>
- khóa final train/test split; <br>
- tự động cleaning dữ liệu; <br>
- quyết định final feature set. <br>

Quy trình phân tích xuyên suốt notebook: <br>
`Câu hỏi → Kiểm tra → Bằng chứng → Nhận xét → Chuyển giao`

# M2.1 — Khóa quy trình EDA và môi trường làm việc

## Mục tiêu

M2.1 thiết lập nền móng để toàn bộ kết quả EDA phía sau có thể `kiểm chứng` và `tái hiện`. <br>

Các nhiệm vụ của bước này gồm: <br>
- khóa các `guardrail` được kế thừa từ Milestone 1; <br>
- thống nhất quy trình thực hiện EDA; <br>
- thiết lập đường dẫn và các cấu hình dùng chung; <br>
- xác nhận file dữ liệu đang sử dụng đúng artifact đã audit; <br>
- ghi lại môi trường thực thi của notebook. <br>

## Điều kiện hoàn thành

M2.1 chỉ được coi là hoàn thành khi xác nhận được: <br>
`File tồn tại → Kích thước đúng → SHA-256 khớp → Môi trường được ghi nhận`

## M2.1.1 — Khóa guardrail và quy trình thực hiện EDA

### Nhiệm vụ

Trước khi phân tích dữ liệu, cần khóa những nguyên tắc đã được xác định ở Milestone 1 để tránh việc EDA vô tình tạo `data leakage`, thay đổi dữ liệu hoặc đi trước sang giai đoạn modeling. <br>

### Guardrail kế thừa từ M1

`User`, `Card`, `Merchant Name` <br>
→ chỉ sử dụng làm identifier, khóa grouping hoặc xây dựng lịch sử. <br>
→ không đưa giá trị raw trực tiếp vào classifier. <br>

`Errors?` <br>
→ có thể khảo sát để hiểu dataset. <br>
→ không dùng làm feature cho Model V1 vì prediction-time availability chưa được chứng minh. <br>

`Is Fraud?` <br>
→ chỉ đóng vai trò target. <br>

`Temporal evaluation` <br>
→ không sử dụng random split đơn giản làm cách đánh giá chính. <br>

`Historical feature` <br>
→ transaction hiện tại chỉ được sử dụng thông tin xảy ra trước nó. <br>
→ không sử dụng thông tin tương lai. <br>

`Merchant State / Zip missing` <br>
→ không tự động xóa các dòng bị missing. <br>

`Negative Amount` <br>
→ không tự động `abs()`, `drop()` hoặc `clip()`. <br>

`Năm 2020` <br>
→ không dùng làm final fraud test. <br>

`Phạm vi tuyên bố` <br>
→ dataset là dữ liệu synthetic. <br>
→ không diễn giải kết quả như hiệu năng của một hệ thống fraud detection ngân hàng thực tế. <br>

### Quy trình EDA

Mọi phân tích quan trọng sẽ cố gắng tuân theo: <br>
`Câu hỏi → Phương pháp kiểm tra → Bằng chứng → Nhận xét → Chuyển giao` <br>

Các nguyên tắc bổ sung: <br>
- không tạo biểu đồ hoặc thống kê nếu chưa biết nó trả lời câu hỏi gì; <br>
- tách `quan sát từ dữ liệu` khỏi `diễn giải`; <br>
- không âm thầm cleaning dữ liệu; <br>
- luôn ghi rõ kết quả đến từ `toàn bộ dữ liệu`, `dữ liệu tổng hợp` hay `mẫu dữ liệu`; <br>
- sample chỉ dùng thay cho toàn bộ dataset khi mục đích và cách lấy mẫu đã được ghi rõ; <br>
- mọi phân tích lịch sử phải tôn trọng thứ tự thời gian. <br>

### Kết luận M2.1.1

Các `guardrail từ M1` tiếp tục có hiệu lực trong toàn bộ M2. <br>

Quy trình EDA đã được khóa theo hướng `câu hỏi trước → code sau → kết luận từ output thực tế`. <br>

M2 sẽ không tự thực hiện cleaning, preprocessing hoặc modeling khi chưa đến milestone tương ứng. <br>

`Trạng thái: ĐÃ KHÓA`

## M2.1.2 — Thiết lập môi trường và cấu hình dùng chung

### Nhiệm vụ

Thiết lập các thư viện, đường dẫn và hằng số sẽ được sử dụng xuyên suốt notebook. <br>

Các cấu hình chính gồm: <br>
`PROJECT_ROOT` — thư mục gốc của project. <br>
`DATA_PATH` — đường dẫn đến raw dataset. <br>
`FIGURE_DIR` — nơi lưu hình ảnh của M2. <br>
`RANDOM_STATE` — seed cố định để các thao tác lấy mẫu có thể tái hiện. <br>
`EXPECTED_SHA256` — fingerprint của artifact đã khóa ở M1. <br>
`TARGET_COLUMN` — tên cột target. <br>


```python
from pathlib import Path
import hashlib
import platform
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path("../..")

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ibm_tabformer"
    / "card_transaction.v1.csv"
)

FIGURE_DIR = (
    PROJECT_ROOT
    / "reports"
    / "figures"
    / "m2"
)

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

EXPECTED_SHA256 = (
    "68c438319cf27614d5564b7b520814036f0d7e087b3a17d8c15081848e0f02de"
)

TARGET_COLUMN = "Is Fraud?"
```

### Nhận xét M2.1.2

Notebook đã thiết lập các cấu hình dùng chung cho M2. <br>

`DATA_PATH` trỏ tới artifact IBM TabFormer nằm trong `data/raw/ibm_tabformer/`. <br>

`RANDOM_STATE = 42` được sử dụng như quy ước để các thao tác ngẫu nhiên về sau có khả năng tái hiện. <br>

`EXPECTED_SHA256` sẽ được sử dụng ở bước tiếp theo để xác minh artifact. <br>

Ở bước này chưa đọc toàn bộ dataset vào bộ nhớ. <br>

## M2.1.3 — Xác minh đường dẫn và kích thước artifact

### Câu hỏi

Notebook có đang trỏ tới một file dữ liệu thực sự tồn tại không? <br>
Kích thước file hiện tại có khớp với artifact đã được audit ở M1 không? <br>

### Phương pháp

Kiểm tra `DATA_PATH.exists()` và đọc kích thước file trực tiếp từ filesystem. <br>

Kích thước kỳ vọng từ M1 là `2,354,626,737 bytes`. <br>


```python
print("Đường dẫn dữ liệu:", DATA_PATH.resolve())
print("File tồn tại:", DATA_PATH.exists())

if DATA_PATH.exists():
    size_bytes = DATA_PATH.stat().st_size
    print("Kích thước (bytes):", size_bytes)
    print("Kích thước (GiB):", round(size_bytes / (1024 ** 3), 3))
```

    Đường dẫn dữ liệu: /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/raw/ibm_tabformer/card_transaction.v1.csv
    File tồn tại: True
    Kích thước (bytes): 2354626737
    Kích thước (GiB): 2.193


### Nhận xét M2.1.3

Kết quả cho thấy `DATA_PATH` trỏ tới file tồn tại trên máy. <br>

Kích thước thực tế là `2,354,626,737 bytes`, khớp với kích thước artifact đã được ghi nhận ở M1. <br>

Việc kích thước khớp là bằng chứng ban đầu rằng notebook đang sử dụng đúng file. <br>
Tuy nhiên, kích thước giống nhau chưa đủ để chứng minh nội dung file hoàn toàn giống nhau. <br>

Vì vậy cần tiếp tục xác minh bằng `SHA-256`. <br>

`Kết luận: PASS`

## M2.1.4 — Xác minh fingerprint SHA-256

### Câu hỏi

File hiện tại có đúng chính xác artifact đã được audit ở M1 hay không? <br>

### Phương pháp

Tính `SHA-256` trực tiếp từ file local theo từng chunk nhỏ và so sánh với fingerprint đã khóa trong M1. <br>

`SHA-256 kỳ vọng:` <br>
`68c438319cf27614d5564b7b520814036f0d7e087b3a17d8c15081848e0f02de` <br>


```python
def calculate_sha256(file_path, chunk_size=1024 * 1024):
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)

            if not chunk:
                break

            sha256.update(chunk)

    return sha256.hexdigest()
actual_sha256 = calculate_sha256(DATA_PATH)

print("SHA-256 kỳ vọng:")
print(EXPECTED_SHA256)

print("\nSHA-256 thực tế:")
print(actual_sha256)

print("\nKhớp:")
print(actual_sha256 == EXPECTED_SHA256)
```

    SHA-256 kỳ vọng:
    68c438319cf27614d5564b7b520814036f0d7e087b3a17d8c15081848e0f02de
    
    SHA-256 thực tế:
    68c438319cf27614d5564b7b520814036f0d7e087b3a17d8c15081848e0f02de
    
    Khớp:
    True


### Nhận xét M2.1.4

`SHA-256 thực tế` khớp hoàn toàn với `SHA-256 kỳ vọng`. <br>

Kết quả so sánh trả về `True`. <br>

Điều này xác nhận file được sử dụng trong M2 chính là artifact đã được khóa và audit ở M1 ở mức nội dung byte. <br>

Từ bước này trở đi, các kết quả EDA có thể được truy ngược về đúng artifact của project. <br>

`Kết luận: PASS`

## M2.1.5 — Ghi lại môi trường thực thi

### Câu hỏi

Notebook M2 đang được chạy bằng phiên bản Python và các thư viện chính nào? <br>

### Mục đích

Ghi lại môi trường giúp tăng khả năng `tái hiện kết quả` và hỗ trợ điều tra nếu cùng notebook cho kết quả khác trên một máy hoặc môi trường khác. <br>

Các thông tin cần ghi nhận gồm: <br>
`Python` <br>
`Pandas` <br>
`NumPy` <br>
`Platform` <br>


```python
environment = {
    "python": sys.version.split()[0],
    "platform": platform.platform(),
    "pandas": pd.__version__,
    "numpy": np.__version__,
}

environment
```




    {'python': '3.14.6',
     'platform': 'macOS-26.6.2-arm64-arm-64bit-Mach-O',
     'pandas': '3.0.5',
     'numpy': '2.5.3'}



### Nhận xét M2.1.5

Môi trường thực tế của M2: <br>
`Python: 3.14.6` <br>
`Pandas: 3.0.5` <br>
`NumPy: 2.5.3` <br>
`Platform: macOS-26.6.2-arm64-arm-64bit-Mach-O` <br>

Ba phiên bản chính `Python`, `Pandas` và `NumPy` trùng với môi trường đã được sử dụng trong Technical Audit của M1. <br>

Không phát hiện khác biệt môi trường cần điều tra trước khi tiếp tục EDA. <br>

`Kết luận: PASS`

# Tổng kết M2.1 — Khóa quy trình EDA và môi trường làm việc

## Kết quả

M2.1 đã hoàn thành các nhiệm vụ nền tảng trước khi bắt đầu phân tích dữ liệu. <br>

`Guardrail và quy trình EDA` <br>
→ đã được khóa và tiếp tục có hiệu lực trong toàn bộ M2. <br>

`Artifact path` <br>
→ file tồn tại và notebook đang trỏ đúng vào `data/raw/ibm_tabformer/card_transaction.v1.csv`. <br>

`File size` <br>
→ `2,354,626,737 bytes`, khớp với artifact đã audit ở M1. <br>

`SHA-256` <br>
→ khớp hoàn toàn với fingerprint đã khóa ở M1. <br>

`Environment` <br>
→ đã ghi nhận đầy đủ Python, Pandas, NumPy và platform thực tế. <br>

## Quyết định

`Decision ID: M2.1-D01` <br>
`Artifact verification: PASS` <br>
`EDA protocol: LOCKED` <br>
`M1 guardrails: ACTIVE` <br>
`Cleaning: CHƯA THỰC HIỆN` <br>
`Preprocessing: CHƯA QUYẾT ĐỊNH` <br>
`Train/test split: CHƯA QUYẾT ĐỊNH` <br>
`Modeling: CHƯA THỰC HIỆN` <br>
`Final feature set: CHƯA QUYẾT ĐỊNH` <br>

## Trạng thái

`M2.1: PASS` <br>

Không còn vấn đề nào ở M2.1 ngăn cản việc chuyển sang bước tiếp theo. <br>

`Next: M2.2 — EDA cấu trúc và biểu diễn phục vụ phân tích`
