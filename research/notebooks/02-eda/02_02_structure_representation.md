# M2.2 — Kiểm tra cấu trúc và xây biểu diễn dữ liệu phục vụ EDA

## Mục tiêu

Sau khi `M2.1` đã xác minh đúng artifact và khóa quy trình EDA, `M2.2` kiểm tra cách dataset được đọc trong notebook và xây các biểu diễn dữ liệu cần thiết cho những bước phân tích tiếp theo. <br>

M2.2 không làm lại toàn bộ `Technical Audit` của M1.5. <br>
Mục tiêu là xác nhận rằng dữ liệu hiện tại có thể được chuyển thành representation phù hợp cho EDA mà không thay đổi ý nghĩa của dữ liệu gốc. <br>

Các nhiệm vụ chính gồm: <br>
- xác minh lại `schema dữ liệu gốc`; <br>
- quan sát kiểu dữ liệu Pandas đọc được; <br>
- chuyển `Amount` sang dạng số phục vụ phân tích; <br>
- ghép `Year + Month + Day + Time` thành `Timestamp`; <br>
- xác minh các phép chuyển đổi trên toàn bộ dataset; <br>
- kiểm tra thứ tự thời gian của raw CSV; <br>
- xác nhận category của `Use Chip`; <br>
- xác nhận số lượng giá trị khác nhau của `MCC` và các trường location; <br>
- tạo thống kê số giao dịch theo năm và năm-tháng. <br>

## Ranh giới

Trong M2.2: <br>
- chưa làm cleaning; <br>
- chưa xử lý missing value; <br>
- chưa sửa Amount âm; <br>
- chưa encoding category; <br>
- chưa chia train/test; <br>
- chưa huấn luyện model. <br>

`Mục tiêu cuối cùng: khóa representation dùng cho các bước EDA phía sau.`

## Thiết lập môi trường thực thi


```python
from pathlib import Path
from collections import Counter

import pandas as pd

from IPython.display import display


# ============================================================
# Xác định thư mục gốc của project
# ============================================================

DATA_RELATIVE_PATH = (
    Path("data")
    / "raw"
    / "ibm_tabformer"
    / "card_transaction.v1.csv"
)

candidate_roots = [
    Path.cwd(),
    Path.cwd().parent,
    Path.cwd().parent.parent,
]

PROJECT_ROOT = None

for candidate in candidate_roots:
    candidate = candidate.resolve()

    if (candidate / DATA_RELATIVE_PATH).exists():
        PROJECT_ROOT = candidate
        break

if PROJECT_ROOT is None:
    raise FileNotFoundError(
        "Không xác định được PROJECT_ROOT. "
        "Không tìm thấy file dữ liệu tại đường dẫn kỳ vọng."
    )


# ============================================================
# Các biến dùng chung trong M2.2
# ============================================================

DATA_PATH = (
    PROJECT_ROOT
    / DATA_RELATIVE_PATH
)

RANDOM_STATE = 42


# ============================================================
# Kiểm tra nhanh
# ============================================================

print("PROJECT_ROOT:")
print(PROJECT_ROOT)

print("\nDATA_PATH:")
print(DATA_PATH)

print("\nFile tồn tại:")
print(DATA_PATH.exists())
```

    PROJECT_ROOT:
    /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening
    
    DATA_PATH:
    /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/raw/ibm_tabformer/card_transaction.v1.csv
    
    File tồn tại:
    True


## M2.2.1 — Xác minh schema dữ liệu gốc

### Câu hỏi

Dataset được đọc trong M2 có còn đúng `15 cột` đã được audit ở M1 hay không? <br>

Có xuất hiện tình trạng: <br>
- thiếu cột; <br>
- thừa cột; <br>
- sai thứ tự cột; <br>
- trùng tên cột hay không? <br>

### Phương pháp

Chỉ đọc phần `header` của CSV bằng `pd.read_csv(..., nrows=0)` rồi so sánh với danh sách cột đã được khóa từ M1. <br>

Phép kiểm tra này không cần đọc toàn bộ dataset. <br>


```python
EXPECTED_COLUMNS = [
    "User",
    "Card",
    "Year",
    "Month",
    "Day",
    "Time",
    "Amount",
    "Use Chip",
    "Merchant Name",
    "Merchant City",
    "Merchant State",
    "Zip",
    "MCC",
    "Errors?",
    "Is Fraud?",
]

raw_header = pd.read_csv(DATA_PATH, nrows=0)

actual_columns = raw_header.columns.tolist()

print("Số cột kỳ vọng:", len(EXPECTED_COLUMNS))
print("Số cột thực tế:", len(actual_columns))

print("\nThiếu cột:")
print(sorted(set(EXPECTED_COLUMNS) - set(actual_columns)))

print("\nCột không kỳ vọng:")
print(sorted(set(actual_columns) - set(EXPECTED_COLUMNS)))

print("\nThứ tự cột khớp:")
print(actual_columns == EXPECTED_COLUMNS)

print("\nTên cột bị trùng:")
print(raw_header.columns.duplicated().sum())
```

    Số cột kỳ vọng: 15
    Số cột thực tế: 15
    
    Thiếu cột:
    []
    
    Cột không kỳ vọng:
    []
    
    Thứ tự cột khớp:
    True
    
    Tên cột bị trùng:
    0


### Nhận xét M2.2.1

Kết quả cho thấy: <br>
`Số cột kỳ vọng: 15` <br>
`Số cột thực tế: 15` <br>
`Thiếu cột: []` <br>
`Cột không kỳ vọng: []` <br>
`Thứ tự cột khớp: True` <br>
`Tên cột bị trùng: 0` <br>

Schema dữ liệu được đọc trong M2 `khớp hoàn toàn` với schema đã được audit ở M1. <br>

Không phát hiện dấu hiệu thay đổi cấu trúc file trước khi bắt đầu EDA. <br>

`Kết luận: PASS`

## M2.2.2 — Quan sát representation dữ liệu gốc

### Câu hỏi

Pandas đang đọc các cột của raw CSV dưới kiểu dữ liệu nào? <br>

Representation hiện tại có nhất quán với những gì đã ghi nhận ở M1 hay không? <br>

### Phương pháp

Đọc `100,000 dòng đầu` làm mẫu xem trước. <br>

Mẫu này chỉ dùng để quan sát `representation`, không được sử dụng để suy ra các thống kê của toàn bộ dataset. <br>

Các nội dung cần quan sát gồm: <br>
- kích thước mẫu; <br>
- kiểu dữ liệu của từng cột; <br>
- một số dòng đầu; <br>
- một số dòng được chọn ngẫu nhiên với `RANDOM_STATE = 42`. <br>


```python
PREVIEW_ROWS = 100_000

df_preview = pd.read_csv(
    DATA_PATH,
    nrows=PREVIEW_ROWS
)

print("Kích thước mẫu xem trước:")
print(df_preview.shape)

print("\nKiểu dữ liệu:")
print(df_preview.dtypes)

print("\n5 dòng đầu:")
display(df_preview.head(5))

print("\n5 dòng lấy mẫu:")
display(
    df_preview.sample(
        n=5,
        random_state=RANDOM_STATE
    )
)
```

    Kích thước mẫu xem trước:
    (100000, 15)
    
    Kiểu dữ liệu:
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
    dtype: object
    
    5 dòng đầu:



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>User</th>
      <th>Card</th>
      <th>Year</th>
      <th>Month</th>
      <th>Day</th>
      <th>Time</th>
      <th>Amount</th>
      <th>Use Chip</th>
      <th>Merchant Name</th>
      <th>Merchant City</th>
      <th>Merchant State</th>
      <th>Zip</th>
      <th>MCC</th>
      <th>Errors?</th>
      <th>Is Fraud?</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>2002</td>
      <td>9</td>
      <td>1</td>
      <td>06:21</td>
      <td>$134.09</td>
      <td>Swipe Transaction</td>
      <td>3527213246127876953</td>
      <td>La Verne</td>
      <td>CA</td>
      <td>91750.0</td>
      <td>5300</td>
      <td>NaN</td>
      <td>No</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>2002</td>
      <td>9</td>
      <td>1</td>
      <td>06:42</td>
      <td>$38.48</td>
      <td>Swipe Transaction</td>
      <td>-727612092139916043</td>
      <td>Monterey Park</td>
      <td>CA</td>
      <td>91754.0</td>
      <td>5411</td>
      <td>NaN</td>
      <td>No</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>2002</td>
      <td>9</td>
      <td>2</td>
      <td>06:22</td>
      <td>$120.34</td>
      <td>Swipe Transaction</td>
      <td>-727612092139916043</td>
      <td>Monterey Park</td>
      <td>CA</td>
      <td>91754.0</td>
      <td>5411</td>
      <td>NaN</td>
      <td>No</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>2002</td>
      <td>9</td>
      <td>2</td>
      <td>17:45</td>
      <td>$128.95</td>
      <td>Swipe Transaction</td>
      <td>3414527459579106770</td>
      <td>Monterey Park</td>
      <td>CA</td>
      <td>91754.0</td>
      <td>5651</td>
      <td>NaN</td>
      <td>No</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>2002</td>
      <td>9</td>
      <td>3</td>
      <td>06:23</td>
      <td>$104.71</td>
      <td>Swipe Transaction</td>
      <td>5817218446178736267</td>
      <td>La Verne</td>
      <td>CA</td>
      <td>91750.0</td>
      <td>5912</td>
      <td>NaN</td>
      <td>No</td>
    </tr>
  </tbody>
</table>
</div>


    
    5 dòng lấy mẫu:



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>User</th>
      <th>Card</th>
      <th>Year</th>
      <th>Month</th>
      <th>Day</th>
      <th>Time</th>
      <th>Amount</th>
      <th>Use Chip</th>
      <th>Merchant Name</th>
      <th>Merchant City</th>
      <th>Merchant State</th>
      <th>Zip</th>
      <th>MCC</th>
      <th>Errors?</th>
      <th>Is Fraud?</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>75721</th>
      <td>3</td>
      <td>2</td>
      <td>2013</td>
      <td>1</td>
      <td>26</td>
      <td>13:24</td>
      <td>$17.37</td>
      <td>Swipe Transaction</td>
      <td>2027553650310142703</td>
      <td>New York</td>
      <td>NY</td>
      <td>10075.0</td>
      <td>5541</td>
      <td>NaN</td>
      <td>No</td>
    </tr>
    <tr>
      <th>80184</th>
      <td>3</td>
      <td>3</td>
      <td>2017</td>
      <td>12</td>
      <td>14</td>
      <td>13:21</td>
      <td>$34.16</td>
      <td>Chip Transaction</td>
      <td>-5475680618560174533</td>
      <td>Willsboro</td>
      <td>NY</td>
      <td>12996.0</td>
      <td>5942</td>
      <td>NaN</td>
      <td>No</td>
    </tr>
    <tr>
      <th>19864</th>
      <td>0</td>
      <td>3</td>
      <td>2020</td>
      <td>1</td>
      <td>8</td>
      <td>06:55</td>
      <td>$40.55</td>
      <td>Swipe Transaction</td>
      <td>-727612092139916043</td>
      <td>Monterey Park</td>
      <td>CA</td>
      <td>91754.0</td>
      <td>5411</td>
      <td>NaN</td>
      <td>No</td>
    </tr>
    <tr>
      <th>76699</th>
      <td>3</td>
      <td>2</td>
      <td>2016</td>
      <td>12</td>
      <td>24</td>
      <td>21:40</td>
      <td>$153.97</td>
      <td>Chip Transaction</td>
      <td>-2287552604203635169</td>
      <td>Shanghai</td>
      <td>China</td>
      <td>NaN</td>
      <td>5812</td>
      <td>NaN</td>
      <td>No</td>
    </tr>
    <tr>
      <th>92991</th>
      <td>4</td>
      <td>0</td>
      <td>2013</td>
      <td>3</td>
      <td>26</td>
      <td>06:45</td>
      <td>$105.09</td>
      <td>Swipe Transaction</td>
      <td>112925206871091074</td>
      <td>San Francisco</td>
      <td>CA</td>
      <td>94117.0</td>
      <td>5541</td>
      <td>NaN</td>
      <td>No</td>
    </tr>
  </tbody>
</table>
</div>


### Nhận xét M2.2.2

Mẫu xem trước có `100,000 dòng × 15 cột`. <br>

Các kiểu dữ liệu quan trọng gồm: <br>
`Time → str` <br>
`Amount → str` <br>
`Zip → float64` <br>
`User / Card / Merchant Name / MCC → int64` <br>

`Amount` được đọc dưới dạng chuỗi vì giá trị có ký hiệu tiền tệ như `$134.09`. <br>

`Time` được đọc dưới dạng chuỗi với dạng `HH:MM`. <br>

`Zip` được đọc thành `float64` do cột này có giá trị thiếu. <br>

Việc `User`, `Card`, `Merchant Name` hoặc `MCC` được Pandas đọc dạng số không có nghĩa chúng là biến số có ý nghĩa về độ lớn. <br>

Representation hiện tại nhất quán với kết quả Technical Audit của M1. <br>

`Kết luận: PASS`

## M2.2.3 — Chuyển Amount sang dạng số phục vụ EDA

### Câu hỏi

Cột raw `Amount` có thể chuyển sang dạng số để thực hiện các phân tích định lượng hay không? <br>

### Yêu cầu

Representation mới phải: <br>
- loại bỏ ký hiệu `$`; <br>
- hỗ trợ dấu phân cách `,` nếu có; <br>
- giữ nguyên giá trị âm; <br>
- không sửa hoặc ghi đè raw `Amount`; <br>
- biến giá trị không chuyển được thành missing để có thể đếm lỗi parse. <br>

### Phương pháp

Tạo hàm `parse_amount()` và thử trước trên mẫu 100,000 dòng. <br>


```python
def parse_amount(amount_series):
    cleaned = (
        amount_series
        .astype("string")
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
    )

    return pd.to_numeric(
        cleaned,
        errors="coerce"
    )

amount_preview = parse_amount(df_preview["Amount"])

print("Số dòng mẫu:", len(amount_preview))
print("Parse Amount thất bại:", amount_preview.isna().sum())

print("\nMột số giá trị raw:")
print(df_preview["Amount"].head())

print("\nSau khi chuyển thành số:")
print(amount_preview.head())
```

    Số dòng mẫu: 100000
    Parse Amount thất bại: 0
    
    Một số giá trị raw:
    0    $134.09
    1     $38.48
    2    $120.34
    3    $128.95
    4    $104.71
    Name: Amount, dtype: str
    
    Sau khi chuyển thành số:
    0    134.09
    1     38.48
    2    120.34
    3    128.95
    4    104.71
    Name: Amount, dtype: Float64


### Nhận xét M2.2.3

Trong mẫu `100,000 dòng`, số trường hợp không chuyển được `Amount` sang dạng số là `0`. <br>

Ví dụ representation: <br>
`"$134.09" → 134.09` <br>

Hàm chuyển đổi không sử dụng `abs()`, không loại bỏ Amount âm và không thay đổi raw column. <br>

Kết quả trên mẫu cho thấy representation hoạt động đúng về mặt kỹ thuật. <br>

Tuy nhiên, kết quả của mẫu chưa đủ để kết luận cho toàn bộ `24,386,900` giao dịch. <br>

Vì vậy khả năng chuyển đổi phải được xác minh lại trên toàn dataset. <br>

`Kết luận tạm thời: PASS trên mẫu`

## M2.2.4 — Ghép các thành phần thời gian thành Timestamp

### Câu hỏi

Bốn trường raw: <br>
`Year` <br>
`Month` <br>
`Day` <br>
`Time` <br>

có thể được hợp nhất thành một `Timestamp` hợp lệ hay không? <br>

### Mục đích

`Timestamp` sẽ là representation thời gian chính dùng cho: <br>
- phân tích theo thời gian; <br>
- sắp xếp lịch sử giao dịch; <br>
- thiết kế temporal split; <br>
- xây behavioral feature theo quan hệ nhân quả; <br>
- rolling window. <br>

### Phương pháp

Tạo phần ngày từ `Year / Month / Day`, chuyển `Time` thành khoảng thời gian và ghép hai phần thành một Timestamp. <br>

Trước tiên kiểm tra trên mẫu 100,000 dòng. <br>


```python
def build_timestamp(df):
    date_part = pd.to_datetime(
        {
            "year": df["Year"],
            "month": df["Month"],
            "day": df["Day"],
        },
        errors="coerce"
    )

    time_part = pd.to_timedelta(
        df["Time"].astype("string") + ":00",
        errors="coerce"
    )

    return date_part + time_part

timestamp_preview = build_timestamp(df_preview)

print("Số dòng mẫu:", len(timestamp_preview))
print("Parse Timestamp thất bại:", timestamp_preview.isna().sum())

print("\nTimestamp nhỏ nhất trong mẫu:")
print(timestamp_preview.min())

print("\nTimestamp lớn nhất trong mẫu:")
print(timestamp_preview.max())

df_preview_analysis = df_preview.copy()

df_preview_analysis["Timestamp"] = timestamp_preview
df_preview_analysis["Amount_numeric"] = amount_preview

df_preview_analysis[
    [
        "User",
        "Card",
        "Timestamp",
        "Amount",
        "Amount_numeric",
        "Use Chip",
        "MCC",
        "Merchant City",
        "Merchant State",
        "Zip",
        "Is Fraud?",
    ]
].head()
```

    Số dòng mẫu: 100000
    Parse Timestamp thất bại: 0
    
    Timestamp nhỏ nhất trong mẫu:
    1999-11-26 15:03:00
    
    Timestamp lớn nhất trong mẫu:
    2020-02-28 23:49:00





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>User</th>
      <th>Card</th>
      <th>Timestamp</th>
      <th>Amount</th>
      <th>Amount_numeric</th>
      <th>Use Chip</th>
      <th>MCC</th>
      <th>Merchant City</th>
      <th>Merchant State</th>
      <th>Zip</th>
      <th>Is Fraud?</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>2002-09-01 06:21:00</td>
      <td>$134.09</td>
      <td>134.09</td>
      <td>Swipe Transaction</td>
      <td>5300</td>
      <td>La Verne</td>
      <td>CA</td>
      <td>91750.0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>2002-09-01 06:42:00</td>
      <td>$38.48</td>
      <td>38.48</td>
      <td>Swipe Transaction</td>
      <td>5411</td>
      <td>Monterey Park</td>
      <td>CA</td>
      <td>91754.0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>2002-09-02 06:22:00</td>
      <td>$120.34</td>
      <td>120.34</td>
      <td>Swipe Transaction</td>
      <td>5411</td>
      <td>Monterey Park</td>
      <td>CA</td>
      <td>91754.0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>2002-09-02 17:45:00</td>
      <td>$128.95</td>
      <td>128.95</td>
      <td>Swipe Transaction</td>
      <td>5651</td>
      <td>Monterey Park</td>
      <td>CA</td>
      <td>91754.0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>2002-09-03 06:23:00</td>
      <td>$104.71</td>
      <td>104.71</td>
      <td>Swipe Transaction</td>
      <td>5912</td>
      <td>La Verne</td>
      <td>CA</td>
      <td>91750.0</td>
      <td>No</td>
    </tr>
  </tbody>
</table>
</div>



### Nhận xét M2.2.4

Trong mẫu `100,000 dòng`, số Timestamp không tạo được là `0`. <br>

Representation mới có dạng: <br>
`Year + Month + Day + Time → Timestamp` <br>

Ví dụ: <br>
`2002 / 09 / 01 / 06:21 → 2002-09-01 06:21:00` <br>

Ngoài `Timestamp`, notebook vẫn giữ nguyên các cột thời gian raw để bảo toàn dữ liệu gốc. <br>

Kết quả trên mẫu cho thấy phép chuyển đổi hoạt động đúng, nhưng vẫn cần kiểm tra trên toàn dataset. <br>

`Kết luận tạm thời: PASS trên mẫu`

## M2.2.5 — Quét toàn bộ dataset để xác minh representation

### Nhiệm vụ

Các kiểm tra trên `100,000 dòng` chỉ chứng minh code hoạt động trên mẫu. <br>

Để đưa ra kết luận cho toàn bộ dataset, cần quét đủ `24,386,900 dòng`. <br>

### Chiến lược

Không cần giữ toàn bộ dữ liệu trong RAM. <br>

CSV được đọc theo từng `chunk = 500,000 dòng`. <br>

Mỗi chunk chỉ đọc các cột cần cho M2.2: <br>
`Year`, `Month`, `Day`, `Time`, `Amount`, `Use Chip`, `Merchant City`, `Merchant State`, `Zip`, `MCC`. <br>

Trong một lần quét, notebook đồng thời thu thập: <br>
- số lỗi chuyển đổi `Amount`; <br>
- số lỗi tạo `Timestamp`; <br>
- Timestamp nhỏ nhất / lớn nhất; <br>
- thứ tự thời gian của raw CSV; <br>
- các giá trị của `Use Chip`; <br>
- số lượng giá trị khác nhau của các trường quan trọng; <br>
- số giao dịch theo năm; <br>
- số giao dịch theo năm-tháng. <br>

Cách làm này giúp tránh phải đọc file hơn 2 GB nhiều lần. <br>


```python
CHUNK_SIZE = 500_000
M22_USECOLS = [
    "Year",
    "Month",
    "Day",
    "Time",
    "Amount",
    "Use Chip",
    "Merchant City",
    "Merchant State",
    "Zip",
    "MCC",
]

from collections import Counter

total_rows = 0

amount_parse_failures = 0
timestamp_parse_failures = 0

min_timestamp = None
max_timestamp = None

use_chip_values = set()
time_values = set()

mcc_values = set()
merchant_city_values = set()
merchant_state_values = set()
zip_values = set()

year_counts = Counter()
year_month_counts = Counter()

adjacent_timestamp_decreases = 0
previous_last_timestamp = None




for chunk_number, chunk in enumerate(
    pd.read_csv(
        DATA_PATH,
        usecols=M22_USECOLS,
        chunksize=CHUNK_SIZE
    ),
    start=1
):
    total_rows += len(chunk)

    # -----------------------------------------
    # 1. Parse Amount
    # -----------------------------------------
    amount_numeric = parse_amount(chunk["Amount"])

    amount_failures_in_chunk = (
        chunk["Amount"].notna()
        & amount_numeric.isna()
    ).sum()

    amount_parse_failures += int(
        amount_failures_in_chunk
    )

    # -----------------------------------------
    # 2. Parse Timestamp
    # -----------------------------------------
    timestamp = build_timestamp(chunk)

    timestamp_parse_failures += int(
        timestamp.isna().sum()
    )

    valid_timestamp = timestamp.dropna()

    if not valid_timestamp.empty:
        chunk_min = valid_timestamp.min()
        chunk_max = valid_timestamp.max()

        if min_timestamp is None or chunk_min < min_timestamp:
            min_timestamp = chunk_min

        if max_timestamp is None or chunk_max > max_timestamp:
            max_timestamp = chunk_max

    # -----------------------------------------
    # 3. Kiểm tra thứ tự thời gian bên trong chunk
    # -----------------------------------------
    previous_timestamp = timestamp.shift(1)

    out_of_order_inside = (
        timestamp.notna()
        & previous_timestamp.notna()
        & (timestamp < previous_timestamp)
    )

    adjacent_timestamp_decreases += int(
        out_of_order_inside.sum()
    )

    # -----------------------------------------
    # 4. Kiểm tra thứ tự giữa hai chunk
    # -----------------------------------------
    if not valid_timestamp.empty:
        first_valid_timestamp = valid_timestamp.iloc[0]
        last_valid_timestamp = valid_timestamp.iloc[-1]

        if previous_last_timestamp is not None:
            if first_valid_timestamp < previous_last_timestamp:
                adjacent_timestamp_decreases += 1

        previous_last_timestamp = last_valid_timestamp

    # -----------------------------------------
    # 5. Category của Use Chip
    # -----------------------------------------
    use_chip_values.update(
        chunk["Use Chip"]
        .dropna()
        .unique()
        .tolist()
    )

    # -----------------------------------------
    # 6. Cardinality
    # -----------------------------------------
    time_values.update(
        chunk["Time"]
        .dropna()
        .unique()
        .tolist()
    )

    mcc_values.update(
        chunk["MCC"]
        .dropna()
        .unique()
        .tolist()
    )

    merchant_city_values.update(
        chunk["Merchant City"]
        .dropna()
        .unique()
        .tolist()
    )

    merchant_state_values.update(
        chunk["Merchant State"]
        .dropna()
        .unique()
        .tolist()
    )

    zip_values.update(
        chunk["Zip"]
        .dropna()
        .unique()
        .tolist()
    )

    # -----------------------------------------
    # 7. Transaction count theo năm
    # -----------------------------------------
    year_counts.update(
        chunk["Year"]
        .value_counts()
        .to_dict()
    )

    # -----------------------------------------
    # 8. Transaction count theo năm-tháng
    # -----------------------------------------
    year_month_chunk = (
        chunk
        .groupby(["Year", "Month"])
        .size()
    )

    year_month_counts.update(
        {
            tuple(key): int(value)
            for key, value in year_month_chunk.items()
        }
    )

    print(
        f"Đã xử lý chunk {chunk_number} "
        f"- tổng số dòng: {total_rows:,}"
    )
```

    Đã xử lý chunk 1 - tổng số dòng: 500,000
    Đã xử lý chunk 2 - tổng số dòng: 1,000,000
    Đã xử lý chunk 3 - tổng số dòng: 1,500,000
    Đã xử lý chunk 4 - tổng số dòng: 2,000,000
    Đã xử lý chunk 5 - tổng số dòng: 2,500,000
    Đã xử lý chunk 6 - tổng số dòng: 3,000,000
    Đã xử lý chunk 7 - tổng số dòng: 3,500,000
    Đã xử lý chunk 8 - tổng số dòng: 4,000,000
    Đã xử lý chunk 9 - tổng số dòng: 4,500,000
    Đã xử lý chunk 10 - tổng số dòng: 5,000,000
    Đã xử lý chunk 11 - tổng số dòng: 5,500,000
    Đã xử lý chunk 12 - tổng số dòng: 6,000,000
    Đã xử lý chunk 13 - tổng số dòng: 6,500,000
    Đã xử lý chunk 14 - tổng số dòng: 7,000,000
    Đã xử lý chunk 15 - tổng số dòng: 7,500,000
    Đã xử lý chunk 16 - tổng số dòng: 8,000,000
    Đã xử lý chunk 17 - tổng số dòng: 8,500,000
    Đã xử lý chunk 18 - tổng số dòng: 9,000,000
    Đã xử lý chunk 19 - tổng số dòng: 9,500,000
    Đã xử lý chunk 20 - tổng số dòng: 10,000,000
    Đã xử lý chunk 21 - tổng số dòng: 10,500,000
    Đã xử lý chunk 22 - tổng số dòng: 11,000,000
    Đã xử lý chunk 23 - tổng số dòng: 11,500,000
    Đã xử lý chunk 24 - tổng số dòng: 12,000,000
    Đã xử lý chunk 25 - tổng số dòng: 12,500,000
    Đã xử lý chunk 26 - tổng số dòng: 13,000,000
    Đã xử lý chunk 27 - tổng số dòng: 13,500,000
    Đã xử lý chunk 28 - tổng số dòng: 14,000,000
    Đã xử lý chunk 29 - tổng số dòng: 14,500,000
    Đã xử lý chunk 30 - tổng số dòng: 15,000,000
    Đã xử lý chunk 31 - tổng số dòng: 15,500,000
    Đã xử lý chunk 32 - tổng số dòng: 16,000,000
    Đã xử lý chunk 33 - tổng số dòng: 16,500,000
    Đã xử lý chunk 34 - tổng số dòng: 17,000,000
    Đã xử lý chunk 35 - tổng số dòng: 17,500,000
    Đã xử lý chunk 36 - tổng số dòng: 18,000,000
    Đã xử lý chunk 37 - tổng số dòng: 18,500,000
    Đã xử lý chunk 38 - tổng số dòng: 19,000,000
    Đã xử lý chunk 39 - tổng số dòng: 19,500,000
    Đã xử lý chunk 40 - tổng số dòng: 20,000,000
    Đã xử lý chunk 41 - tổng số dòng: 20,500,000
    Đã xử lý chunk 42 - tổng số dòng: 21,000,000
    Đã xử lý chunk 43 - tổng số dòng: 21,500,000
    Đã xử lý chunk 44 - tổng số dòng: 22,000,000
    Đã xử lý chunk 45 - tổng số dòng: 22,500,000
    Đã xử lý chunk 46 - tổng số dòng: 23,000,000
    Đã xử lý chunk 47 - tổng số dòng: 23,500,000
    Đã xử lý chunk 48 - tổng số dòng: 24,000,000
    Đã xử lý chunk 49 - tổng số dòng: 24,386,900


### Nhận xét M2.2.5

Toàn bộ `24,386,900 dòng` đã được xử lý thành công qua `49 chunk`. <br>

Không xảy ra lỗi đọc file hoặc gián đoạn trong quá trình quét toàn bộ dataset. <br>

Các biến tổng hợp thu được từ bước này sẽ được kiểm tra ở các phần tiếp theo. <br>

`Trạng thái quét toàn bộ dữ liệu: HOÀN THÀNH`

## M2.2.6 — Đánh giá khả năng chuyển đổi và thứ tự thời gian

### Câu hỏi

Sau khi quét toàn dataset: <br>
- có bao nhiêu `Amount` không chuyển được sang số? <br>
- có bao nhiêu Timestamp không tạo được? <br>
- khoảng thời gian thực tế của dataset là gì? <br>
- raw CSV có được sắp toàn cục theo Timestamp hay không? <br>


```python
timestamp_success_rate = (
    1 - timestamp_parse_failures / total_rows
) * 100

amount_success_rate = (
    1 - amount_parse_failures / total_rows
) * 100

m22_parse_summary = pd.Series(
    {
        "Tổng số dòng": total_rows,
        "Amount chuyển đổi thất bại": amount_parse_failures,
        "Tỷ lệ chuyển đổi Amount thành công (%)": amount_success_rate,
        "Timestamp tạo thất bại": timestamp_parse_failures,
        "Tỷ lệ tạo Timestamp thành công (%)": timestamp_success_rate,
        "Timestamp nhỏ nhất": min_timestamp,
        "Timestamp lớn nhất": max_timestamp,
        "Số lần Timestamp giảm so với dòng liền trước":
            adjacent_timestamp_decreases,
        "Raw CSV được sắp không giảm theo Timestamp":
            adjacent_timestamp_decreases == 0,
    }
)

m22_parse_summary
```




    Tổng số dòng                                               24386900
    Amount chuyển đổi thất bại                                        0
    Tỷ lệ chuyển đổi Amount thành công (%)                        100.0
    Timestamp tạo thất bại                                            0
    Tỷ lệ tạo Timestamp thành công (%)                            100.0
    Timestamp nhỏ nhất                              1991-01-02 07:10:00
    Timestamp lớn nhất                              2020-02-28 23:58:00
    Số lần Timestamp giảm so với dòng liền trước                   5789
    Raw CSV được sắp không giảm theo Timestamp                    False
    dtype: object



### Nhận xét M2.2.6

Toàn bộ dataset có `24,386,900 dòng`. <br>

`Amount chuyển đổi thất bại: 0` <br>
`Tỷ lệ chuyển đổi Amount thành công: 100%` <br>

Điều này xác nhận toàn bộ raw `Amount` có thể tạo `Amount_numeric` phục vụ EDA. <br>

`Timestamp tạo thất bại: 0` <br>
`Tỷ lệ tạo Timestamp thành công: 100%` <br>

Khoảng thời gian thực tế: <br>
`1991-01-02 07:10:00 → 2020-02-28 23:58:00` <br>

Đáng chú ý, dữ liệu năm `2020` trong artifact chỉ kéo dài đến cuối tháng 2. <br>
Hiện tại M2.2 chỉ ghi nhận hiện tượng này; nguyên nhân và ảnh hưởng đến phân bố target sẽ được phân tích ở M2.3. <br>

Có `5,789` lần Timestamp của một dòng nhỏ hơn Timestamp của dòng liền trước. <br>

Do đó: <br>
`Raw CSV không được sắp toàn cục theo Timestamp.` <br>

Con số `5,789` chỉ là số lần giảm giữa hai dòng liền kề, không phải tổng số tất cả các cặp transaction bị đảo thứ tự trong dataset. <br>

### Định hướng

Mọi phép tính history sau này không được dựa vào raw row order. <br>

Trước khi sử dụng `shift()`, `rolling()` hoặc cumulative statistics theo entity, dữ liệu phải được tổ chức theo `User + Card + Timestamp`. <br>

`Kết luận: PASS WITH FINDING`

## M2.2.7 — Kiểm tra category và số lượng giá trị khác nhau

### Câu hỏi

Các trường categorical quan trọng có cấu trúc như thế nào trên toàn dataset? <br>

Cần xác nhận: <br>
- các giá trị thực tế của `Use Chip`; <br>
- số giá trị khác nhau của `Time`; <br>
- số giá trị khác nhau của `MCC`; <br>
- số giá trị khác nhau của `Merchant City`; <br>
- số giá trị khác nhau của `Merchant State`; <br>
- số giá trị khác nhau của `Zip`. <br>

Ở bước này chỉ xác nhận cấu trúc, chưa quyết định encoding hay giữ/bỏ feature. <br>


```python
print("Các giá trị Use Chip:")

for value in sorted(use_chip_values):
    print("-", value)

EXPECTED_USE_CHIP = {
    "Swipe Transaction",
    "Chip Transaction",
    "Online Transaction",
}

print(
    "\nKhớp với M1:",
    use_chip_values == EXPECTED_USE_CHIP
)

m22_cardinality = pd.Series(
    {
        "Time": len(time_values),
        "MCC": len(mcc_values),
        "Merchant City": len(merchant_city_values),
        "Merchant State": len(merchant_state_values),
        "Zip": len(zip_values),
    },
    name="Số giá trị khác nhau"
)

m22_cardinality
```

    Các giá trị Use Chip:
    - Chip Transaction
    - Online Transaction
    - Swipe Transaction
    
    Khớp với M1: True





    Time               1440
    MCC                 109
    Merchant City     13429
    Merchant State      223
    Zip               27321
    Name: Số giá trị khác nhau, dtype: int64



### Nhận xét M2.2.7

`Use Chip` có đúng ba category: <br>
`Chip Transaction` <br>
`Online Transaction` <br>
`Swipe Transaction` <br>

Kết quả so sánh với M1 trả về `True`. <br>

Số giá trị khác nhau: <br>
`Time: 1,440` <br>
`MCC: 109` <br>
`Merchant City: 13,429` <br>
`Merchant State: 223` <br>
`Zip: 27,321` <br>

`MCC` có cardinality tương đối thấp hơn đáng kể so với `Merchant City` và `Zip`. <br>

Tuy nhiên, M2.2 chưa đưa ra quyết định encoding hoặc loại bỏ bất kỳ trường location nào. <br>

`Zip` và `MCC` tiếp tục được hiểu là `mã phân loại`, không phải biến số có ý nghĩa lớn/nhỏ. <br>

`Kết luận: PASS`

## M2.2.8 — Đối chiếu kết quả cấu trúc với Milestone 1

### Câu hỏi

Các thống kê cấu trúc vừa tính lại trong M2 có nhất quán với kết quả Technical Audit của M1 hay không? <br>

### Phương pháp

So sánh tự động các giá trị quan trọng với những con số đã được khóa ở M1. <br>


```python
m22_consistency_checks = pd.Series(
    {
        "Row count = 24,386,900":
            total_rows == 24_386_900,

        "Amount parse failures = 0":
            amount_parse_failures == 0,

        "Use Chip đúng 3 category":
            use_chip_values == EXPECTED_USE_CHIP,

        "Time cardinality = 1,440":
            len(time_values) == 1_440,

        "MCC cardinality = 109":
            len(mcc_values) == 109,

        "Merchant City cardinality = 13,429":
            len(merchant_city_values) == 13_429,

        "Merchant State cardinality = 223":
            len(merchant_state_values) == 223,

        "Zip cardinality = 27,321":
            len(zip_values) == 27_321,
    }
)

m22_consistency_checks
```




    Row count = 24,386,900                True
    Amount parse failures = 0             True
    Use Chip đúng 3 category              True
    Time cardinality = 1,440              True
    MCC cardinality = 109                 True
    Merchant City cardinality = 13,429    True
    Merchant State cardinality = 223      True
    Zip cardinality = 27,321              True
    dtype: bool



### Nhận xét M2.2.8

Tất cả các phép đối chiếu đều trả về `True`. <br>

Các nội dung khớp gồm: <br>
- tổng số dòng `24,386,900`; <br>
- `Amount parse failures = 0`; <br>
- đúng ba category của `Use Chip`; <br>
- `Time cardinality = 1,440`; <br>
- `MCC cardinality = 109`; <br>
- `Merchant City cardinality = 13,429`; <br>
- `Merchant State cardinality = 223`; <br>
- `Zip cardinality = 27,321`. <br>

Không phát hiện bất nhất giữa representation đang sử dụng trong M2 và Technical Audit của M1. <br>

`Kết luận: PASS`

## M2.2.9 — Tạo các bảng tổng hợp theo thời gian

### Câu hỏi

Dataset có bao nhiêu giao dịch theo từng `năm` và từng `năm-tháng`? <br>

Các bảng tổng hợp này có bảo toàn đầy đủ `24,386,900` giao dịch hay không? <br>

### Mục đích

Ở M2.2, các bảng này chỉ dùng để xác nhận và chuẩn bị cấu trúc thời gian. <br>

Phân tích fraud thay đổi theo thời gian sẽ được thực hiện ở `M2.3`. <br>


```python
year_counts_df = pd.DataFrame(
    sorted(year_counts.items()),
    columns=[
        "Year",
        "transaction_count",
    ]
)

display(year_counts_df)

print(
    "Tổng transaction từ bảng theo năm:",
    year_counts_df["transaction_count"].sum()
)

print(
    "Khớp tổng số dòng:",
    year_counts_df["transaction_count"].sum()
    == total_rows
)

year_month_counts_df = pd.DataFrame(
    [
        {
            "Year": year,
            "Month": month,
            "transaction_count": count,
        }
        for (year, month), count
        in sorted(year_month_counts.items())
    ]
)

print("24 tháng đầu:")
display(year_month_counts_df.head(24))

print("\n12 tháng cuối:")
display(year_month_counts_df.tail(12))

print(
    "\nTổng transaction từ bảng năm-tháng:",
    year_month_counts_df["transaction_count"].sum()
)

print(
    "Khớp tổng số dòng:",
    year_month_counts_df["transaction_count"].sum()
    == total_rows
)

print("Timestamp nhỏ nhất:")
print(min_timestamp)

print("\nTimestamp lớn nhất:")
print(max_timestamp)

print("\nNăm nhỏ nhất:")
print(min_timestamp.year if min_timestamp is not None else None)

print("\nNăm lớn nhất:")
print(max_timestamp.year if max_timestamp is not None else None)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>transaction_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1991</td>
      <td>1585</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1992</td>
      <td>5134</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1993</td>
      <td>8378</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1994</td>
      <td>14316</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1995</td>
      <td>20928</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1996</td>
      <td>29945</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1997</td>
      <td>49753</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1998</td>
      <td>78345</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1999</td>
      <td>118250</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2000</td>
      <td>177729</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2001</td>
      <td>257998</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2002</td>
      <td>350732</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2003</td>
      <td>466408</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2004</td>
      <td>597003</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2005</td>
      <td>746653</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2006</td>
      <td>908793</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2007</td>
      <td>1064483</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2008</td>
      <td>1223460</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2009</td>
      <td>1355434</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2010</td>
      <td>1491225</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2011</td>
      <td>1570551</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2012</td>
      <td>1610829</td>
    </tr>
    <tr>
      <th>22</th>
      <td>2013</td>
      <td>1650917</td>
    </tr>
    <tr>
      <th>23</th>
      <td>2014</td>
      <td>1672343</td>
    </tr>
    <tr>
      <th>24</th>
      <td>2015</td>
      <td>1701371</td>
    </tr>
    <tr>
      <th>25</th>
      <td>2016</td>
      <td>1708924</td>
    </tr>
    <tr>
      <th>26</th>
      <td>2017</td>
      <td>1723360</td>
    </tr>
    <tr>
      <th>27</th>
      <td>2018</td>
      <td>1721615</td>
    </tr>
    <tr>
      <th>28</th>
      <td>2019</td>
      <td>1723938</td>
    </tr>
    <tr>
      <th>29</th>
      <td>2020</td>
      <td>336500</td>
    </tr>
  </tbody>
</table>
</div>


    Tổng transaction từ bảng theo năm: 24386900
    Khớp tổng số dòng: True
    24 tháng đầu:



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>Month</th>
      <th>transaction_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1991</td>
      <td>1</td>
      <td>55</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1991</td>
      <td>2</td>
      <td>48</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1991</td>
      <td>3</td>
      <td>45</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1991</td>
      <td>4</td>
      <td>48</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1991</td>
      <td>5</td>
      <td>45</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1991</td>
      <td>6</td>
      <td>52</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1991</td>
      <td>7</td>
      <td>178</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1991</td>
      <td>8</td>
      <td>164</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1991</td>
      <td>9</td>
      <td>190</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1991</td>
      <td>10</td>
      <td>155</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1991</td>
      <td>11</td>
      <td>179</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1991</td>
      <td>12</td>
      <td>426</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1992</td>
      <td>1</td>
      <td>418</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1992</td>
      <td>2</td>
      <td>367</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1992</td>
      <td>3</td>
      <td>426</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1992</td>
      <td>4</td>
      <td>427</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1992</td>
      <td>5</td>
      <td>441</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1992</td>
      <td>6</td>
      <td>366</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1992</td>
      <td>7</td>
      <td>406</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1992</td>
      <td>8</td>
      <td>415</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1992</td>
      <td>9</td>
      <td>376</td>
    </tr>
    <tr>
      <th>21</th>
      <td>1992</td>
      <td>10</td>
      <td>375</td>
    </tr>
    <tr>
      <th>22</th>
      <td>1992</td>
      <td>11</td>
      <td>519</td>
    </tr>
    <tr>
      <th>23</th>
      <td>1992</td>
      <td>12</td>
      <td>598</td>
    </tr>
  </tbody>
</table>
</div>


    
    12 tháng cuối:



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>Month</th>
      <th>transaction_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>338</th>
      <td>2019</td>
      <td>3</td>
      <td>147202</td>
    </tr>
    <tr>
      <th>339</th>
      <td>2019</td>
      <td>4</td>
      <td>141409</td>
    </tr>
    <tr>
      <th>340</th>
      <td>2019</td>
      <td>5</td>
      <td>145648</td>
    </tr>
    <tr>
      <th>341</th>
      <td>2019</td>
      <td>6</td>
      <td>142399</td>
    </tr>
    <tr>
      <th>342</th>
      <td>2019</td>
      <td>7</td>
      <td>146599</td>
    </tr>
    <tr>
      <th>343</th>
      <td>2019</td>
      <td>8</td>
      <td>147139</td>
    </tr>
    <tr>
      <th>344</th>
      <td>2019</td>
      <td>9</td>
      <td>141744</td>
    </tr>
    <tr>
      <th>345</th>
      <td>2019</td>
      <td>10</td>
      <td>145074</td>
    </tr>
    <tr>
      <th>346</th>
      <td>2019</td>
      <td>11</td>
      <td>141946</td>
    </tr>
    <tr>
      <th>347</th>
      <td>2019</td>
      <td>12</td>
      <td>146579</td>
    </tr>
    <tr>
      <th>348</th>
      <td>2020</td>
      <td>1</td>
      <td>170731</td>
    </tr>
    <tr>
      <th>349</th>
      <td>2020</td>
      <td>2</td>
      <td>165769</td>
    </tr>
  </tbody>
</table>
</div>


    
    Tổng transaction từ bảng năm-tháng: 24386900
    Khớp tổng số dòng: True
    Timestamp nhỏ nhất:
    1991-01-02 07:10:00
    
    Timestamp lớn nhất:
    2020-02-28 23:58:00
    
    Năm nhỏ nhất:
    1991
    
    Năm lớn nhất:
    2020


### Nhận xét M2.2.9

Tổng số giao dịch trong bảng theo năm là `24,386,900`, khớp với tổng số dòng của dataset. <br>

Tổng số giao dịch trong bảng theo năm-tháng cũng là `24,386,900`. <br>

Do đó hai phép tổng hợp theo thời gian không làm thất thoát giao dịch. <br>

Khoảng thời gian của dữ liệu tiếp tục được xác nhận: <br>
`1991-01-02 07:10:00 → 2020-02-28 23:58:00` <br>

Các bảng `year_counts_df` và `year_month_counts_df` sẽ được tái sử dụng ở M2.3 để phân tích sự thay đổi của target theo thời gian. <br>

`Kết luận: PASS`

## M2.2.10 — Khóa representation dùng cho EDA

Sau các phép kiểm tra trên mẫu và toàn bộ dataset, M2.2 khóa representation phục vụ các bước EDA tiếp theo như sau. <br>

### Dữ liệu thời gian

Các trường raw: <br>
`Year` <br>
`Month` <br>
`Day` <br>
`Time` <br>

được giữ nguyên và đồng thời dùng để tạo: <br>
`Timestamp` <br>

`Timestamp` sẽ là representation thời gian chính phục vụ: <br>
- phân tích theo thời gian; <br>
- sắp xếp lịch sử giao dịch; <br>
- thiết kế temporal split; <br>
- historical feature; <br>
- rolling window. <br>

### Amount

Raw `Amount` được giữ nguyên. <br>

Đồng thời tạo representation số: <br>
`Amount_numeric` <br>

Không áp dụng `abs()`, `drop()` hoặc `clip()` đối với Amount âm ở M2.2. <br>

### Các mã phân loại

`MCC` và `Zip` dù được Pandas đọc dưới dạng số vẫn được hiểu là `categorical code`. <br>

Không diễn giải giá trị lớn/nhỏ của các mã này như đại lượng số. <br>

### Identifier và khóa lịch sử

`User` <br>
`Card` <br>
`Merchant Name` <br>

tiếp tục chỉ được sử dụng cho grouping, entity history và behavioral feature. <br>

Không đưa giá trị raw của chúng trực tiếp vào classifier. <br>

### Các trường đặc biệt

`Errors?` <br>
→ tiếp tục không sử dụng trong Model V1. <br>

`Is Fraud?` <br>
→ chỉ đóng vai trò target. <br>

### Quy tắc về thứ tự thời gian

Raw CSV `không được sắp toàn cục theo Timestamp`. <br>

Do đó mọi thao tác xây lịch sử sau này phải sắp xếp theo entity và Timestamp một cách tường minh. <br>

`Representation policy: LOCKED FOR EDA`

# Tổng kết M2.2 — Kiểm tra cấu trúc và biểu diễn phục vụ EDA

## Kết quả

`Raw schema` <br>
→ đúng `15 cột`, không thiếu, không thừa và không trùng tên. <br>

`Amount` <br>
→ chuyển sang dạng số thành công trên `100%` của `24,386,900` dòng. <br>

`Timestamp` <br>
→ tạo thành công trên `100%` dataset. <br>
→ khoảng thời gian: `1991-01-02 07:10:00 → 2020-02-28 23:58:00`. <br>

`Raw CSV temporal order` <br>
→ không được sắp toàn cục theo Timestamp. <br>
→ phát hiện `5,789` lần Timestamp giảm so với dòng liền trước. <br>

`Use Chip` <br>
→ có đúng ba category: `Chip Transaction`, `Online Transaction`, `Swipe Transaction`. <br>

`Cardinality` <br>
→ `Time = 1,440` <br>
→ `MCC = 109` <br>
→ `Merchant City = 13,429` <br>
→ `Merchant State = 223` <br>
→ `Zip = 27,321` <br>

`Consistency with M1` <br>
→ tất cả các phép đối chiếu đều đạt. <br>

`Temporal aggregation` <br>
→ tổng theo năm và tổng theo năm-tháng đều bảo toàn đủ `24,386,900` giao dịch. <br>

## Phát hiện quan trọng

Raw row order `không thể` được sử dụng thay cho thứ tự thời gian. <br>

Mọi historical feature sau này phải dựa trên `Timestamp` và được sắp xếp theo entity một cách tường minh. <br>

Dữ liệu năm `2020` kết thúc vào `2020-02-28`; hiện tượng này cần được phân tích tiếp ở M2.3. <br>

## Quyết định

`Decision ID: M2.2-D01` <br>
`Raw schema: PASS` <br>
`Amount representation: PASS` <br>
`Timestamp representation: PASS` <br>
`Category/cardinality checks: PASS` <br>
`Consistency with M1: PASS` <br>
`Temporal order: PASS WITH FINDING` <br>
`Representation policy: LOCKED FOR EDA` <br>
`Blocking issue: NONE` <br>

## Trạng thái

`M2.2: PASS` <br>

Không có vấn đề nào ngăn cản việc chuyển sang bước tiếp theo. <br>

`Next: M2.3 — Phân tích target và chiều thời gian`
