# M4.2 — Xây representation cơ sở và audit input

## Vai trò

M4.2 là bước đầu tiên của Milestone 4 làm việc trở lại với raw artifact thực tế.

Mục tiêu của M4.2 là xây một representation kỹ thuật nhất quán để các bước feature engineering và preprocessing phía sau cùng làm việc trên:

- cùng một transaction;
- cùng một Timestamp;
- cùng một Amount representation;
- cùng semantic role;
- cùng temporal boundary;
- cùng quy tắc target;
- cùng identifier/history-key contract.

M4.2 chưa thực hiện:

- xử lý missing value;
- quyết định policy cho negative Amount;
- encoding categorical feature;
- scaling;
- behavioral feature engineering;
- feature selection;
- model training;
- resampling;
- threshold selection.

M4.2 cũng không sử dụng FINAL TEST performance để hỗ trợ development.

## Câu hỏi trung tâm

> Raw transaction phải được biểu diễn thế nào để những bước feature/preprocessing phía sau có thể hoạt động trên một representation nhất quán, tái hiện được và không phá vỡ các guardrail đã khóa?

## Trạng thái khi bắt đầu notebook

Representation contract:
DEFINED

Runtime verification:
NOT YET EXECUTED

M4.2 Gate:
OPEN

# 1. Căn cứ kế thừa và phạm vi M4.2

M4.2 không khám phá dataset lại từ đầu.

Các evidence từ M1–M3 được tái sử dụng làm expected contract.

## Evidence đã có

Raw artifact đã được audit có:

- 24,386,900 transaction;
- 15 raw columns;
- target `Is Fraud?`;
- target values `Yes / No`;
- không missing target;
- Amount raw lưu dưới dạng chuỗi có ký hiệu tiền tệ;
- Time có dạng HH:MM;
- MCC và Zip là categorical code về mặt semantics;
- User / Card / Merchant Name là identifier/history key;
- Errors? bị loại khỏi Model V1.

M2.2 đã xác nhận:

- Amount có thể parse thành số trên toàn artifact;
- Timestamp có thể dựng trên toàn artifact;
- raw row order không phải global chronological order;
- Use Chip có ba category đã biết.

M3 đã khóa temporal boundary:

TRAIN:
Timestamp < 2019-01-01

VALIDATION:
2019-01-01 <= Timestamp < 2019-06-01

FINAL TEST:
2019-06-01 <= Timestamp < 2019-11-01

Training-window candidates:

W_LONG:
2015-01-01 <= Timestamp < 2019-01-01

W_SHORT:
2018-01-01 <= Timestamp < 2019-01-01

M4.2 chỉ kiểm chứng rằng current artifact và implementation hiện tại vẫn tuân thủ các contract trên.

## Nguyên tắc evidence

Không ghi PASS chỉ vì kết quả trước đây đã PASS.

Notebook hiện tại phải chạy code trên artifact thực tế.

Các kết luận phụ thuộc runtime sẽ được để OPEN cho tới khi có output.

# 2. Thiết lập môi trường

## Câu hỏi

Notebook có đang làm việc với đúng raw artifact của project hay không?

## Mục đích

Tránh trường hợp các phép kiểm tra phía sau đúng về code nhưng lại chạy trên:

- sai file;
- sai project root;
- artifact đã bị thay đổi;
- phiên bản dữ liệu khác với artifact đã audit ở M1.

## Expected artifact

Tên file:

card_transaction.v1.csv

Expected size:

2,354,626,737 bytes

Raw artifact không được chỉnh sửa trong M4.


```python
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd


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
        "Không tìm thấy raw artifact tại đường dẫn kỳ vọng."
    )

DATA_PATH = PROJECT_ROOT / DATA_RELATIVE_PATH

EXPECTED_FILE_SIZE = 2_354_626_737
CHUNK_SIZE = 500_000

print("PROJECT_ROOT:")
print(PROJECT_ROOT)

print("\nDATA_PATH:")
print(DATA_PATH)

print("\nFile tồn tại:")
print(DATA_PATH.exists())

print("\nFile size:")
print(DATA_PATH.stat().st_size)
```

    PROJECT_ROOT:
    /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening
    
    DATA_PATH:
    /Users/minhtoan/SE/HocTrenLop/Trí tuệ nhân tạo/fraud-risk-screening/data/raw/ibm_tabformer/card_transaction.v1.csv
    
    File tồn tại:
    True
    
    File size:
    2354626737



```python
assert DATA_PATH.exists(), (
    "STOP: Không tìm thấy raw artifact."
)

assert DATA_PATH.stat().st_size == EXPECTED_FILE_SIZE, (
    "STOP: File size khác artifact đã audit ở M1. "
    "Cần xác minh artifact identity trước khi tiếp tục."
)

print("M4.2 ENVIRONMENT / ARTIFACT GATE: PASS")
```

    M4.2 ENVIRONMENT / ARTIFACT GATE: PASS


### Nhận xét

Notebook đã xác định được đúng cấu trúc thư mục project và tìm thấy raw artifact tại:

`data/raw/ibm_tabformer/card_transaction.v1.csv`

File tồn tại tại thời điểm thực thi và có kích thước:

`2,354,626,737 bytes`

Kích thước này khớp chính xác với artifact đã được audit trước đó.

Cell kiểm tra environment/artifact đã chạy thành công và trả về:

`M4.2 ENVIRONMENT / ARTIFACT GATE: PASS`

Trong M4.2 hiện tại không tính lại SHA-256, vì vậy bằng chứng của lần chạy này xác nhận sự nhất quán dựa trên:

- project path;
- raw-data path;
- sự tồn tại của file;
- file size đã khóa.

Không có dấu hiệu cho thấy notebook đang đọc nhầm file hoặc artifact có kích thước khác với artifact đã được audit.

### Kết luận M4.2.0

Current raw artifact nhất quán với artifact mà project đã sử dụng trong các milestone trước theo các kiểm tra identity được thực hiện trong notebook M4.2.

Decision:

`M4.2-E00 — Current artifact identity`

Status:

`PASS`

Blocking issue:

`NONE`


# 3. M4.2.1 — Xác minh raw schema

## Câu hỏi

Current raw artifact có còn đúng schema 15 cột đã khóa hay không?

Cần kiểm tra:

- số cột;
- cột bị thiếu;
- cột ngoài dự kiến;
- tên cột trùng;
- thứ tự cột.

## Expected schema

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

Schema mismatch là blocking issue.

Nếu schema khác contract:

STOP

Không tiếp tục feature/preprocessing pipeline.


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

raw_header = pd.read_csv(
    DATA_PATH,
    nrows=0,
)

actual_columns = raw_header.columns.tolist()

missing_columns = sorted(
    set(EXPECTED_COLUMNS)
    - set(actual_columns)
)

unexpected_columns = sorted(
    set(actual_columns)
    - set(EXPECTED_COLUMNS)
)

duplicate_column_count = int(
    raw_header.columns
    .duplicated()
    .sum()
)

column_order_matches = (
    actual_columns == EXPECTED_COLUMNS
)

print("Expected column count:")
print(len(EXPECTED_COLUMNS))

print("\nActual column count:")
print(len(actual_columns))

print("\nMissing columns:")
print(missing_columns)

print("\nUnexpected columns:")
print(unexpected_columns)

print("\nDuplicate column names:")
print(duplicate_column_count)

print("\nColumn order matches:")
print(column_order_matches)
```

    Expected column count:
    15
    
    Actual column count:
    15
    
    Missing columns:
    []
    
    Unexpected columns:
    []
    
    Duplicate column names:
    0
    
    Column order matches:
    True



```python
assert len(actual_columns) == 15
assert missing_columns == []
assert unexpected_columns == []
assert duplicate_column_count == 0
assert column_order_matches

print("M4.2-G01 RAW SCHEMA: PASS")
```

    M4.2-G01 RAW SCHEMA: PASS


### Nhận xét

Raw header hiện tại có đúng:

`15 columns`

bằng số cột kỳ vọng.

Kết quả đối chiếu cho thấy:

- missing columns: `[]`;
- unexpected columns: `[]`;
- duplicate column names: `0`;
- column order matches: `True`.

Như vậy current raw artifact không xuất hiện schema drift so với canonical raw schema đã khóa.

Cell assertion hoàn thành với:

`M4.2-G01 RAW SCHEMA: PASS`

Điều này đặc biệt quan trọng vì các transformation phía sau sử dụng tên và vai trò của các raw field theo canonical contract. Nếu schema thay đổi mà không được phát hiện, Timestamp, target, Amount hoặc history key có thể bị xử lý sai.

### Kết luận M4.2.1

Current raw schema khớp hoàn toàn với canonical 15-column schema.

Không phát hiện:

- missing field;
- unexpected field;
- duplicated field name;
- sai thứ tự cột.

Decision:

`M4.2-E01 — Current raw schema consistency`

Status:

`PASS`

Raw schema contract:

`VERIFIED`


# 4. M4.2.2 — Quan sát raw representation

## Câu hỏi

Pandas hiện đang đọc các raw field dưới kiểu dữ liệu nào?

Representation hiện tại có còn nhất quán với M1/M2 hay không?

## Mục đích

Phân biệt:

physical dtype
≠
semantic role

Ví dụ:

MCC đọc thành số
không đồng nghĩa MCC là continuous numerical feature.

User/Card/Merchant Name đọc thành số
không đồng nghĩa ID magnitude có ý nghĩa với classifier.


```python
PREVIEW_ROWS = 100_000

df_preview = pd.read_csv(
    DATA_PATH,
    nrows=PREVIEW_ROWS,
)

print("Preview shape:")
print(df_preview.shape)

print("\nRaw dtypes:")
print(df_preview.dtypes)

print("\nFirst 5 rows:")
display(df_preview.head())
```

    Preview shape:
    (100000, 15)
    
    Raw dtypes:
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
    
    First 5 rows:



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


### Nhận xét

Preview gồm:

`100,000 rows × 15 columns`

và cho thấy Pandas đang đọc các field theo representation phù hợp với những gì đã audit trước đó.

Các dtype đáng chú ý:

`Time → str`

`Amount → str`

`Zip → float64`

`User → int64`

`Card → int64`

`Merchant Name → int64`

`MCC → int64`

`Errors? → str`

`Is Fraud? → str`

Raw examples cũng xác nhận:

- `Amount` vẫn chứa ký hiệu `$`;
- `Time` vẫn ở dạng `HH:MM`;
- `Is Fraud?` vẫn ở dạng nhãn `No/Yes`;
- `Errors?` có thể xuất hiện dưới dạng missing;
- `Zip` được đọc dạng số thực do khả năng có missing value.

Không có dấu hiệu cho thấy raw representation đã bị preprocessing trước khi vào M4.2.

Tuy nhiên, dtype vật lý không được dùng để suy ra semantic role.

Cụ thể:

- `User`, `Card`, `Merchant Name` đọc dưới dạng integer nhưng vẫn là identifier/history key;
- `MCC` đọc dưới dạng integer nhưng vẫn là categorical code;
- `Zip` đọc dưới dạng float nhưng không phải continuous numerical quantity.

### Kết luận M4.2.2

Raw representation trong current runtime nhất quán với schema và semantic assumptions đã khóa từ M1/M2.

Không phát hiện representation anomaly mới ở preview làm thay đổi contract M4.2.

Status:

`PASS`

Semantic type vẫn phải được xác định bằng data dictionary / project contract, không bằng Pandas dtype.



# 5. M4.2.3 — Xây Amount_numeric

## Câu hỏi

Raw Amount có thể được chuyển thành representation số một cách deterministic mà không làm thay đổi semantics đã biết hay không?

## Contract

Raw:

Amount

Canonical technical representation:

Amount_numeric

Transformation chỉ:

- bỏ ký hiệu `$`;
- bỏ dấu phân cách `,` nếu có;
- strip khoảng trắng;
- parse numeric.

Không:

- abs();
- clip();
- drop negative Amount;
- overwrite raw Amount.

M4.2 chỉ làm representation conversion.

Policy cuối cho negative Amount thuộc M4.3/M4.4.


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
        errors="coerce",
    )


amount_preview = parse_amount(
    df_preview["Amount"]
)

print("Preview rows:")
print(len(amount_preview))

print("\nAmount parse failures:")
print(amount_preview.isna().sum())

print("\nRaw → numeric examples:")
display(
    pd.DataFrame(
        {
            "Amount_raw": (
                df_preview["Amount"].head(10)
            ),
            "Amount_numeric": (
                amount_preview.head(10)
            ),
        }
    )
)
```

    Preview rows:
    100000
    
    Amount parse failures:
    0
    
    Raw → numeric examples:



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
      <th>Amount_raw</th>
      <th>Amount_numeric</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>$134.09</td>
      <td>134.09</td>
    </tr>
    <tr>
      <th>1</th>
      <td>$38.48</td>
      <td>38.48</td>
    </tr>
    <tr>
      <th>2</th>
      <td>$120.34</td>
      <td>120.34</td>
    </tr>
    <tr>
      <th>3</th>
      <td>$128.95</td>
      <td>128.95</td>
    </tr>
    <tr>
      <th>4</th>
      <td>$104.71</td>
      <td>104.71</td>
    </tr>
    <tr>
      <th>5</th>
      <td>$86.19</td>
      <td>86.19</td>
    </tr>
    <tr>
      <th>6</th>
      <td>$93.84</td>
      <td>93.84</td>
    </tr>
    <tr>
      <th>7</th>
      <td>$123.50</td>
      <td>123.5</td>
    </tr>
    <tr>
      <th>8</th>
      <td>$61.72</td>
      <td>61.72</td>
    </tr>
    <tr>
      <th>9</th>
      <td>$57.10</td>
      <td>57.1</td>
    </tr>
  </tbody>
</table>
</div>



```python
assert amount_preview.isna().sum() == 0

print("M4.2 AMOUNT PREVIEW GATE: PASS")
```

    M4.2 AMOUNT PREVIEW GATE: PASS


### Nhận xét

Trên preview `100,000` dòng:

`Amount parse failures = 0`

Các ví dụ cho thấy transformation hoạt động đúng theo contract:

`"$134.09" → 134.09`

`"$38.48" → 38.48`

`"$123.50" → 123.5`

Hàm `parse_amount()` chỉ:

- loại bỏ `$`;
- loại bỏ dấu `,` nếu có;
- loại bỏ khoảng trắng thừa;
- chuyển phần còn lại sang numeric.

Implementation không sử dụng:

- `abs()`;
- `clip()`;
- loại bỏ Amount âm;
- ghi đè raw Amount.

Full-artifact audit sau đó tiếp tục xác nhận:

`Amount parse failures = 0`

trên toàn bộ:

`24,386,900 transactions`

Như vậy parser hiện tại có thể chuyển toàn bộ raw Amount sang numeric representation mà không phát sinh parse failure.

M4.2 chỉ xác nhận tính đúng đắn về representation. Kết quả này không trả lời các câu hỏi semantic/preprocessing như:

- negative Amount nên được biểu diễn thế nào trong final feature set;
- có cần log transformation;
- có cần scaling;
- có cần additional indicator.

Các câu hỏi đó tiếp tục thuộc phạm vi M4.3/M4.4.

### Kết luận M4.2.3

`Amount_numeric` được xác nhận là deterministic base representation hợp lệ cho current artifact.

Full-artifact result:

`24,386,900 / 24,386,900` dòng parse thành công.

Parse failures:

`0`

Raw Amount không bị thay đổi bởi representation function.

Decision:

`Raw Amount → Amount_numeric`

Status:

`LOCKED FOR BASE REPRESENTATION`

Negative-Amount preprocessing policy:

`OPEN — HANDOFF TO M4.3/M4.4`

# 6. M4.2.4 — Xây Timestamp

## Câu hỏi

Year + Month + Day + Time có thể được hợp nhất thành một canonical Timestamp hợp lệ hay không?

## Vai trò của Timestamp

Timestamp không chỉ là một feature candidate.

Nó là temporal index dùng cho:

- temporal partition;
- history ordering;
- causal feature engineering;
- rolling windows;
- future/past boundary.

Raw physical row order không được sử dụng thay cho Timestamp.


```python
def build_timestamp(df):
    date_part = pd.to_datetime(
        {
            "year": df["Year"],
            "month": df["Month"],
            "day": df["Day"],
        },
        errors="coerce",
    )

    time_part = pd.to_timedelta(
        df["Time"].astype("string") + ":00",
        errors="coerce",
    )

    return date_part + time_part


timestamp_preview = build_timestamp(
    df_preview
)

print("Preview rows:")
print(len(timestamp_preview))

print("\nTimestamp parse failures:")
print(timestamp_preview.isna().sum())

print("\nTimestamp min:")
print(timestamp_preview.min())

print("\nTimestamp max:")
print(timestamp_preview.max())

display(
    pd.DataFrame(
        {
            "Year": df_preview["Year"].head(),
            "Month": df_preview["Month"].head(),
            "Day": df_preview["Day"].head(),
            "Time": df_preview["Time"].head(),
            "Timestamp": timestamp_preview.head(),
        }
    )
)
```

    Preview rows:
    100000
    
    Timestamp parse failures:
    0
    
    Timestamp min:
    1999-11-26 15:03:00
    
    Timestamp max:
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
      <th>Year</th>
      <th>Month</th>
      <th>Day</th>
      <th>Time</th>
      <th>Timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2002</td>
      <td>9</td>
      <td>1</td>
      <td>06:21</td>
      <td>2002-09-01 06:21:00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2002</td>
      <td>9</td>
      <td>1</td>
      <td>06:42</td>
      <td>2002-09-01 06:42:00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2002</td>
      <td>9</td>
      <td>2</td>
      <td>06:22</td>
      <td>2002-09-02 06:22:00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2002</td>
      <td>9</td>
      <td>2</td>
      <td>17:45</td>
      <td>2002-09-02 17:45:00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2002</td>
      <td>9</td>
      <td>3</td>
      <td>06:23</td>
      <td>2002-09-03 06:23:00</td>
    </tr>
  </tbody>
</table>
</div>



```python
assert timestamp_preview.isna().sum() == 0

print("M4.2 TIMESTAMP PREVIEW GATE: PASS")
```

    M4.2 TIMESTAMP PREVIEW GATE: PASS


### Nhận xét

Preview xác nhận implementation của `build_timestamp()` hoạt động đúng về mặt kỹ thuật trên mẫu kiểm tra.

Không có parse failure trong `100,000` dòng preview.

Min/max của preview chỉ mô tả mẫu đầu vào đang được xem, vì vậy không được dùng làm global temporal range.

Full-artifact audit mới là bằng chứng đủ để kiểm tra canonical Timestamp representation trên toàn dataset.

Kết quả full scan cho thấy:

- toàn bộ transaction đều tạo được Timestamp;
- không có Timestamp bị missing do parse;
- global min/max khớp với temporal extent đã audit trước đó.

Như vậy `Year`, `Month`, `Day`, `Time` có thể được hợp nhất thành một temporal index duy nhất mà không mất transaction do lỗi chuyển đổi.

### Kết luận M4.2.4

Canonical Timestamp representation được xác minh thành công trên toàn artifact.

Timestamp parse failures:

`0`

Global range:

`1991-01-02 07:10:00`
→
`2020-02-28 23:58:00`

Decision:

`Year + Month + Day + Time → Timestamp`

Timestamp role:

`CANONICAL TEMPORAL INDEX`

Status:

`LOCKED`

# 7. M4.2.5 — Khóa semantic role của các trường

Physical dtype và modeling semantic là hai khái niệm khác nhau.

M4.2 định nghĩa semantic-role registry sau.

raw_row_id
→ TECHNICAL_LINEAGE_ONLY

User
→ HISTORY_KEY_ONLY

Card
→ HISTORY_KEY_ONLY

Timestamp
→ TEMPORAL_INDEX

Amount_numeric
→ NUMERIC_TRANSACTION_CANDIDATE

Use Chip
→ CATEGORICAL_TRANSACTION_CANDIDATE

Merchant Name
→ HISTORY_KEY_ONLY

Merchant City
→ CONDITIONAL_CATEGORICAL_CANDIDATE

Merchant State
→ CONDITIONAL_CATEGORICAL_CANDIDATE

Zip
→ CONDITIONAL_CATEGORICAL_CODE

MCC
→ CATEGORICAL_CODE_CANDIDATE

temporal_region
→ PARTITION_METADATA_ONLY

is_w_long_train
→ PARTITION_METADATA_ONLY

is_w_short_train
→ PARTITION_METADATA_ONLY

is_validation
→ PARTITION_METADATA_ONLY

is_final_test_protected
→ PARTITION_METADATA_ONLY

target_binary
→ TARGET_ONLY

## Guardrails

Không dùng trực tiếp làm classifier feature:

- raw_row_id;
- User;
- Card;
- Merchant Name;
- partition metadata;
- target_binary.

Errors?:
EXCLUDED FROM MODEL V1.

MCC và Zip:
không được diễn giải như continuous numerical magnitude.

### Nhận xét

Semantic-role registry của M4.2 kế thừa trực tiếp các guardrail đã khóa ở M1–M3.

Không có evidence mới trong current run yêu cầu thay đổi các vai trò này.

Đặc biệt:

`User`

`Card`

`Merchant Name`

vẫn cần tồn tại trong base representation để phục vụ grouping và historical/behavioral feature engineering, nhưng không được sử dụng trực tiếp làm classifier feature.

Tương tự:

`MCC`

và:

`Zip`

không được coi là continuous numerical magnitude chỉ vì dtype vật lý của chúng là numeric.

`Errors?` vẫn nằm ngoài Model V1.

### Kết luận M4.2.5

Semantic-role registry được giữ nguyên và áp dụng cho M4 implementation.

Các vai trò sau được khóa:

`User → HISTORY_KEY_ONLY`

`Card → HISTORY_KEY_ONLY`

`Merchant Name → HISTORY_KEY_ONLY`

`Timestamp → TEMPORAL_INDEX`

`MCC → CATEGORICAL_CODE_CANDIDATE`

`Zip → CONDITIONAL_CATEGORICAL_CODE`

`target_binary → TARGET_ONLY`

`Errors? → EXCLUDED FROM MODEL V1`

Status:

`INHERITED — LOCKED`

Không cần model-performance experiment để thay đổi các semantic guardrail này.


# 8. M4.2.6 — Gắn temporal region và training-window eligibility

## Câu hỏi

Mỗi transaction có thể được gắn đúng vào temporal role đã khóa ở M3 hay không?

## Canonical regions

PRE_W_LONG_HISTORY:
Timestamp < 2015-01-01

TRAIN_2015_2017:
2015-01-01 <= Timestamp < 2018-01-01

TRAIN_2018:
2018-01-01 <= Timestamp < 2019-01-01

VALIDATION:
2019-01-01 <= Timestamp < 2019-06-01

FINAL_TEST_PROTECTED:
2019-06-01 <= Timestamp < 2019-11-01

POST_BREAK_DIAGNOSTIC:
Timestamp >= 2019-11-01

## Training-window eligibility

W_LONG:
2015-01-01 <= Timestamp < 2019-01-01

W_SHORT:
2018-01-01 <= Timestamp < 2019-01-01

W_SHORT phải là subset của W_LONG.

M4.2 không chọn winner.


```python
W_LONG_START = pd.Timestamp("2015-01-01")
W_SHORT_START = pd.Timestamp("2018-01-01")

TRAIN_END = pd.Timestamp("2019-01-01")
VALIDATION_END = pd.Timestamp("2019-06-01")
FINAL_TEST_END = pd.Timestamp("2019-11-01")


def assign_temporal_region(timestamp):
    conditions = [
        timestamp < W_LONG_START,

        (
            (timestamp >= W_LONG_START)
            & (timestamp < W_SHORT_START)
        ),

        (
            (timestamp >= W_SHORT_START)
            & (timestamp < TRAIN_END)
        ),

        (
            (timestamp >= TRAIN_END)
            & (timestamp < VALIDATION_END)
        ),

        (
            (timestamp >= VALIDATION_END)
            & (timestamp < FINAL_TEST_END)
        ),

        timestamp >= FINAL_TEST_END,
    ]

    choices = [
        "PRE_W_LONG_HISTORY",
        "TRAIN_2015_2017",
        "TRAIN_2018",
        "VALIDATION",
        "FINAL_TEST_PROTECTED",
        "POST_BREAK_DIAGNOSTIC",
    ]

    result = np.select(
        conditions,
        choices,
        default="UNASSIGNED",
    )

    return pd.Series(
        result,
        index=timestamp.index,
        dtype="string",
    )
```


```python
temporal_region_preview = (
    assign_temporal_region(
        timestamp_preview
    )
)

is_w_long_train_preview = (
    (timestamp_preview >= W_LONG_START)
    & (timestamp_preview < TRAIN_END)
)

is_w_short_train_preview = (
    (timestamp_preview >= W_SHORT_START)
    & (timestamp_preview < TRAIN_END)
)

is_validation_preview = (
    (timestamp_preview >= TRAIN_END)
    & (timestamp_preview < VALIDATION_END)
)

is_final_test_protected_preview = (
    (timestamp_preview >= VALIDATION_END)
    & (timestamp_preview < FINAL_TEST_END)
)

print("Temporal region counts:")
print(
    temporal_region_preview
    .value_counts(dropna=False)
)

print("\nUNASSIGNED:")
print(
    temporal_region_preview
    .eq("UNASSIGNED")
    .sum()
)

print("\nW_SHORT outside W_LONG:")
print(
    (
        is_w_short_train_preview
        & ~is_w_long_train_preview
    ).sum()
)
```

    Temporal region counts:
    PRE_W_LONG_HISTORY       67584
    TRAIN_2015_2017          18985
    TRAIN_2018                6346
    VALIDATION                2574
    FINAL_TEST_PROTECTED      2558
    POST_BREAK_DIAGNOSTIC     1953
    Name: count, dtype: Int64
    
    UNASSIGNED:
    0
    
    W_SHORT outside W_LONG:
    0


### Nhận xét

Ở preview `100,000` dòng, hàm phân vùng đã gắn mọi transaction vào một temporal region hợp lệ.

Kết quả:

`UNASSIGNED = 0`

Đồng thời:

`W_SHORT rows outside W_LONG = 0`

Điều này xác nhận implementation ban đầu tuân thủ quan hệ:

`W_SHORT ⊂ W_LONG`

Full-artifact audit sau đó tiếp tục xác nhận:

`PRE_W_LONG_HISTORY = 15,471,192`

`TRAIN_2015_2017 = 5,133,655`

`TRAIN_2018 = 1,721,615`

`VALIDATION = 712,458`

`FINAL_TEST_PROTECTED = 722,955`

`POST_BREAK_DIAGNOSTIC = 625,025`

và:

`UNASSIGNED = 0`

Các region trên bao phủ toàn bộ `24,386,900` transaction.

Training-window eligibility cũng khớp canonical counts:

`W_LONG = 6,855,270`

`W_SHORT = 1,721,615`

và không có transaction nào thuộc W_SHORT nhưng nằm ngoài W_LONG.

Kết quả này kiểm tra việc implementation hiện tại tái tạo đúng temporal boundary đã khóa ở M3. Nó không cung cấp evidence để chọn W_LONG hay W_SHORT làm winner.

### Kết luận M4.2.6

Temporal-region assignment và training-window eligibility hoạt động đúng trên toàn artifact.

Coverage:

`100%`

Unassigned rows:

`0`

W_SHORT outside W_LONG:

`0`

Decision:

`Temporal metadata implementation = VERIFIED`

W_LONG/W_SHORT winner:

`OPEN — REQUIRES MODEL EXPERIMENT`

Status:

`PASS / LOCKED FOR REPRESENTATION`


# 9. M4.2.7 — Target representation và target protection

## Raw target

Is Fraud?

Yes:
fraud / positive class

No:
non-fraud / negative class

## Canonical development representation

No → 0
Yes → 1

Field:

target_binary

## Quy tắc

target_binary chỉ được materialize cho development regions được phép:

- W_LONG-eligible TRAIN rows;
- VALIDATION.

target_binary không materialize cho:

- PRE_W_LONG_HISTORY;
- FINAL_TEST_PROTECTED;
- POST_BREAK_DIAGNOSTIC.

Lý do:

- historical warm-up không cần target;
- FINAL TEST phải được bảo vệ khỏi development decisions;
- post-break diagnostic không thuộc core development target population.

Raw artifact không bị chỉnh sửa.


```python
def prepare_base_representation(
    raw_df,
    raw_row_offset,
):
    df = raw_df.copy()

    timestamp = build_timestamp(df)

    amount_numeric = parse_amount(
        df["Amount"]
    )

    temporal_region = (
        assign_temporal_region(
            timestamp
        )
    )

    is_w_long_train = (
        (timestamp >= W_LONG_START)
        & (timestamp < TRAIN_END)
    )

    is_w_short_train = (
        (timestamp >= W_SHORT_START)
        & (timestamp < TRAIN_END)
    )

    is_validation = (
        (timestamp >= TRAIN_END)
        & (timestamp < VALIDATION_END)
    )

    is_final_test_protected = (
        (timestamp >= VALIDATION_END)
        & (timestamp < FINAL_TEST_END)
    )

    target_allowed = (
        is_w_long_train
        | is_validation
    )

    target_binary = pd.Series(
        pd.NA,
        index=df.index,
        dtype="Int8",
    )

    mapped_target = (
        df.loc[
            target_allowed,
            "Is Fraud?",
        ]
        .astype("string")
        .map(
            {
                "No": 0,
                "Yes": 1,
            }
        )
    )

    target_binary.loc[
        target_allowed
    ] = mapped_target.astype("Int8")

    raw_row_id = np.arange(
        raw_row_offset,
        raw_row_offset + len(df),
        dtype=np.int64,
    )

    base = pd.DataFrame(
        {
            "raw_row_id": raw_row_id,

            "User": df["User"].to_numpy(),
            "Card": df["Card"].to_numpy(),

            "Timestamp": (
                timestamp.to_numpy()
            ),

            "Amount_numeric": (
                amount_numeric.to_numpy()
            ),

            "Use Chip": (
                df["Use Chip"]
                .astype("string")
                .to_numpy()
            ),

            "Merchant Name": (
                df["Merchant Name"]
                .to_numpy()
            ),

            "Merchant City": (
                df["Merchant City"]
                .astype("string")
                .to_numpy()
            ),

            "Merchant State": (
                df["Merchant State"]
                .astype("string")
                .to_numpy()
            ),

            "Zip": (
                df["Zip"].to_numpy()
            ),

            "MCC": (
                df["MCC"].to_numpy()
            ),

            "temporal_region": (
                temporal_region.to_numpy()
            ),

            "is_w_long_train": (
                is_w_long_train.to_numpy()
            ),

            "is_w_short_train": (
                is_w_short_train.to_numpy()
            ),

            "is_validation": (
                is_validation.to_numpy()
            ),

            "is_final_test_protected": (
                is_final_test_protected
                .to_numpy()
            ),

            "target_binary": (
                target_binary.to_numpy()
            ),
        }
    )

    return base
```


```python
M42_USECOLS = [
    column
    for column in EXPECTED_COLUMNS
    if column != "Errors?"
]

preview_for_base = pd.read_csv(
    DATA_PATH,
    nrows=PREVIEW_ROWS,
    usecols=M42_USECOLS,
)

base_preview = (
    prepare_base_representation(
        preview_for_base,
        raw_row_offset=0,
    )
)

print("Raw preview shape:")
print(preview_for_base.shape)

print("\nBase preview shape:")
print(base_preview.shape)

print("\nBase columns:")
print(base_preview.columns.tolist())

print("\nBase dtypes:")
print(base_preview.dtypes)
```

    Raw preview shape:
    (100000, 14)
    
    Base preview shape:
    (100000, 17)
    
    Base columns:
    ['raw_row_id', 'User', 'Card', 'Timestamp', 'Amount_numeric', 'Use Chip', 'Merchant Name', 'Merchant City', 'Merchant State', 'Zip', 'MCC', 'temporal_region', 'is_w_long_train', 'is_w_short_train', 'is_validation', 'is_final_test_protected', 'target_binary']
    
    Base dtypes:
    raw_row_id                          int64
    User                                int64
    Card                                int64
    Timestamp                  datetime64[us]
    Amount_numeric                    float64
    Use Chip                              str
    Merchant Name                       int64
    Merchant City                         str
    Merchant State                        str
    Zip                               float64
    MCC                                 int64
    temporal_region                       str
    is_w_long_train                      bool
    is_w_short_train                     bool
    is_validation                        bool
    is_final_test_protected              bool
    target_binary                     float64
    dtype: object



```python
protected_mask = (
    base_preview[
        "is_final_test_protected"
    ]
)

history_only_mask = (
    base_preview["temporal_region"]
    == "PRE_W_LONG_HISTORY"
)

post_break_mask = (
    base_preview["temporal_region"]
    == "POST_BREAK_DIAGNOSTIC"
)

print(
    "FINAL TEST target exposure:",
    base_preview.loc[
        protected_mask,
        "target_binary",
    ]
    .notna()
    .sum(),
)

print(
    "History-only target exposure:",
    base_preview.loc[
        history_only_mask,
        "target_binary",
    ]
    .notna()
    .sum(),
)

print(
    "Post-break target exposure:",
    base_preview.loc[
        post_break_mask,
        "target_binary",
    ]
    .notna()
    .sum(),
)
```

    FINAL TEST target exposure: 0
    History-only target exposure: 0
    Post-break target exposure: 0



```python
assert (
    base_preview["Timestamp"]
    .isna()
    .sum()
    == 0
)

assert (
    base_preview["Amount_numeric"]
    .isna()
    .sum()
    == 0
)

assert (
    base_preview["temporal_region"]
    .eq("UNASSIGNED")
    .sum()
    == 0
)

assert "Errors?" not in base_preview.columns
assert "Is Fraud?" not in base_preview.columns

assert (
    base_preview.loc[
        protected_mask,
        "target_binary",
    ]
    .notna()
    .sum()
    == 0
)

assert (
    base_preview.loc[
        history_only_mask,
        "target_binary",
    ]
    .notna()
    .sum()
    == 0
)

assert (
    base_preview.loc[
        post_break_mask,
        "target_binary",
    ]
    .notna()
    .sum()
    == 0
)

print("M4.2 BASE REPRESENTATION PREVIEW GATE: PASS")
```

    M4.2 BASE REPRESENTATION PREVIEW GATE: PASS


### Nhận xét

Base representation preview có:

`100,000 rows × 17 columns`

và chứa đúng các technical/modeling fields được thiết kế:

`raw_row_id`

`User`

`Card`

`Timestamp`

`Amount_numeric`

`Use Chip`

`Merchant Name`

`Merchant City`

`Merchant State`

`Zip`

`MCC`

`temporal_region`

`is_w_long_train`

`is_w_short_train`

`is_validation`

`is_final_test_protected`

`target_binary`

Hai raw field cần được bảo vệ không xuất hiện trong base representation:

`Errors?`

`Is Fraud?`

Điều này xác nhận raw processing signal và raw target không bị propagate trực tiếp vào downstream base representation.

Target-protection checks trả về:

`FINAL TEST target exposure = 0`

`History-only target exposure = 0`

`Post-break target exposure = 0`

Preview gate hoàn thành với:

`M4.2 BASE REPRESENTATION PREVIEW GATE: PASS`

Một ghi chú kỹ thuật là `target_binary` được DataFrame hiển thị dưới dtype:

`float64`

thay vì nullable `Int8`.

Nguyên nhân là representation hiện tại chứa đồng thời giá trị `0/1` và missing target ở các protected regions, nên khi đưa qua cấu trúc NumPy/DataFrame, dtype được nâng sang floating-point để biểu diễn `NaN`.

Điểm này không làm thay đổi:

- domain `0/1` ở development rows;
- target semantics;
- target-isolation rule;
- các gate của M4.2.

Do M4.2 chưa yêu cầu `Int8` là một invariant bắt buộc, đây là implementation note chứ không phải blocking issue.

Nếu pipeline modeling phía sau cần integer target, việc cast có thể thực hiện rõ ràng trên `y_train` / `y_validation` sau khi loại các row có target unavailable.

### Kết luận M4.2.7

Canonical base representation được tạo đúng về:

- row preservation;
- field composition;
- raw-field protection;
- temporal metadata;
- target isolation.

Protected target exposure:

`0`

ở tất cả các vùng không được phép.

Decision:

`Base Representation Contract = VERIFIED`

Status:

`PASS`

Implementation note:

`target_binary dtype = float64 due to protected missing values; non-blocking.`

# 10. M4.2.8 — Full-artifact base-representation audit

## Câu hỏi

Base Representation Contract có hoạt động đúng trên toàn bộ current raw artifact hay không?

Preview chỉ dùng để phát hiện lỗi implementation sớm.

Kết luận M4.2 phải dựa trên full-artifact audit.

## Chiến lược thực thi

Dataset có hơn 24 triệu transaction và raw CSV hơn 2 GB.

Do đó notebook xử lý theo chunk để:

- tránh giữ toàn bộ raw artifact trong RAM;
- chỉ quét dataset một lần;
- đồng thời thu thập tất cả invariant cần cho M4.2.


```python
total_rows = 0

amount_parse_failures = 0
timestamp_parse_failures = 0
unassigned_region_rows = 0

unexpected_target_rows = 0

protected_target_exposure = 0
history_target_exposure = 0
post_break_target_exposure = 0

w_short_not_in_w_long = 0

region_counts = Counter()
use_chip_values = set()

w_long_rows = 0
w_short_rows = 0
validation_rows = 0
final_test_protected_rows = 0

w_long_fraud = 0
w_short_fraud = 0
validation_fraud = 0

min_timestamp = None
max_timestamp = None

raw_row_offset = 0


for chunk_number, chunk in enumerate(
    pd.read_csv(
        DATA_PATH,
        usecols=M42_USECOLS,
        chunksize=CHUNK_SIZE,
    ),
    start=1,
):
    base = prepare_base_representation(
        chunk,
        raw_row_offset=raw_row_offset,
    )

    timestamp = base["Timestamp"]
    amount = base["Amount_numeric"]

    timestamp_parse_failures += int(
        timestamp.isna().sum()
    )

    amount_parse_failures += int(
        amount.isna().sum()
    )

    unassigned_region_rows += int(
        base["temporal_region"]
        .eq("UNASSIGNED")
        .sum()
    )

    chunk_min = timestamp.min()
    chunk_max = timestamp.max()

    if pd.notna(chunk_min):
        if (
            min_timestamp is None
            or chunk_min < min_timestamp
        ):
            min_timestamp = chunk_min

    if pd.notna(chunk_max):
        if (
            max_timestamp is None
            or chunk_max > max_timestamp
        ):
            max_timestamp = chunk_max

    region_counts.update(
        base["temporal_region"]
        .value_counts()
        .to_dict()
    )

    w_long_mask = (
        base["is_w_long_train"]
    )

    w_short_mask = (
        base["is_w_short_train"]
    )

    validation_mask = (
        base["is_validation"]
    )

    final_mask = (
        base["is_final_test_protected"]
    )

    w_long_rows += int(
        w_long_mask.sum()
    )

    w_short_rows += int(
        w_short_mask.sum()
    )

    validation_rows += int(
        validation_mask.sum()
    )

    final_test_protected_rows += int(
        final_mask.sum()
    )

    w_short_not_in_w_long += int(
        (
            w_short_mask
            & ~w_long_mask
        ).sum()
    )

    target_allowed = (
        w_long_mask
        | validation_mask
    )

    raw_target_allowed = (
        chunk.loc[
            target_allowed.to_numpy(),
            "Is Fraud?",
        ]
        .astype("string")
    )

    unexpected_target_rows += int(
        (
            ~raw_target_allowed
            .isin(["No", "Yes"])
        ).sum()
    )

    w_long_fraud += int(
        base.loc[
            w_long_mask,
            "target_binary",
        ]
        .eq(1)
        .sum()
    )

    w_short_fraud += int(
        base.loc[
            w_short_mask,
            "target_binary",
        ]
        .eq(1)
        .sum()
    )

    validation_fraud += int(
        base.loc[
            validation_mask,
            "target_binary",
        ]
        .eq(1)
        .sum()
    )

    protected_target_exposure += int(
        base.loc[
            final_mask,
            "target_binary",
        ]
        .notna()
        .sum()
    )

    history_mask = (
        base["temporal_region"]
        == "PRE_W_LONG_HISTORY"
    )

    history_target_exposure += int(
        base.loc[
            history_mask,
            "target_binary",
        ]
        .notna()
        .sum()
    )

    post_break_mask = (
        base["temporal_region"]
        == "POST_BREAK_DIAGNOSTIC"
    )

    post_break_target_exposure += int(
        base.loc[
            post_break_mask,
            "target_binary",
        ]
        .notna()
        .sum()
    )

    use_chip_values.update(
        chunk["Use Chip"]
        .dropna()
        .astype("string")
        .unique()
        .tolist()
    )

    total_rows += len(chunk)
    raw_row_offset += len(chunk)

    print(
        f"Chunk {chunk_number:02d} | "
        f"rows processed = "
        f"{total_rows:,}"
    )
```

    Chunk 01 | rows processed = 500,000
    Chunk 02 | rows processed = 1,000,000
    Chunk 03 | rows processed = 1,500,000
    Chunk 04 | rows processed = 2,000,000
    Chunk 05 | rows processed = 2,500,000
    Chunk 06 | rows processed = 3,000,000
    Chunk 07 | rows processed = 3,500,000
    Chunk 08 | rows processed = 4,000,000
    Chunk 09 | rows processed = 4,500,000
    Chunk 10 | rows processed = 5,000,000
    Chunk 11 | rows processed = 5,500,000
    Chunk 12 | rows processed = 6,000,000
    Chunk 13 | rows processed = 6,500,000
    Chunk 14 | rows processed = 7,000,000
    Chunk 15 | rows processed = 7,500,000
    Chunk 16 | rows processed = 8,000,000
    Chunk 17 | rows processed = 8,500,000
    Chunk 18 | rows processed = 9,000,000
    Chunk 19 | rows processed = 9,500,000
    Chunk 20 | rows processed = 10,000,000
    Chunk 21 | rows processed = 10,500,000
    Chunk 22 | rows processed = 11,000,000
    Chunk 23 | rows processed = 11,500,000
    Chunk 24 | rows processed = 12,000,000
    Chunk 25 | rows processed = 12,500,000
    Chunk 26 | rows processed = 13,000,000
    Chunk 27 | rows processed = 13,500,000
    Chunk 28 | rows processed = 14,000,000
    Chunk 29 | rows processed = 14,500,000
    Chunk 30 | rows processed = 15,000,000
    Chunk 31 | rows processed = 15,500,000
    Chunk 32 | rows processed = 16,000,000
    Chunk 33 | rows processed = 16,500,000
    Chunk 34 | rows processed = 17,000,000
    Chunk 35 | rows processed = 17,500,000
    Chunk 36 | rows processed = 18,000,000
    Chunk 37 | rows processed = 18,500,000
    Chunk 38 | rows processed = 19,000,000
    Chunk 39 | rows processed = 19,500,000
    Chunk 40 | rows processed = 20,000,000
    Chunk 41 | rows processed = 20,500,000
    Chunk 42 | rows processed = 21,000,000
    Chunk 43 | rows processed = 21,500,000
    Chunk 44 | rows processed = 22,000,000
    Chunk 45 | rows processed = 22,500,000
    Chunk 46 | rows processed = 23,000,000
    Chunk 47 | rows processed = 23,500,000
    Chunk 48 | rows processed = 24,000,000
    Chunk 49 | rows processed = 24,386,900


# 11. M4.2.9 — Full-audit summary

Cell sau chỉ tổng hợp evidence từ full-artifact audit.

Code cell không tự diễn giải kết quả.

Phần phân tích và diễn giải evidence được trình bày tại mục 11.2 ngay sau output thực tế.


```python
print("===== M4.2 FULL AUDIT =====")

print("\nTotal rows:")
print(total_rows)

print("\nAmount parse failures:")
print(amount_parse_failures)

print("\nTimestamp parse failures:")
print(timestamp_parse_failures)

print("\nTimestamp min:")
print(min_timestamp)

print("\nTimestamp max:")
print(max_timestamp)

print("\nUnassigned temporal rows:")
print(unassigned_region_rows)

print("\nTemporal regions:")
for key in sorted(region_counts):
    print(
        key,
        f"{region_counts[key]:,}",
    )

print("\nW_LONG rows:")
print(f"{w_long_rows:,}")

print("\nW_SHORT rows:")
print(f"{w_short_rows:,}")

print("\nVALIDATION rows:")
print(f"{validation_rows:,}")

print("\nFINAL TEST protected rows:")
print(
    f"{final_test_protected_rows:,}"
)

print("\nW_SHORT rows outside W_LONG:")
print(w_short_not_in_w_long)

print("\nUnexpected development target rows:")
print(unexpected_target_rows)

print("\nW_LONG fraud:")
print(f"{w_long_fraud:,}")

print("\nW_SHORT fraud:")
print(f"{w_short_fraud:,}")

print("\nVALIDATION fraud:")
print(f"{validation_fraud:,}")

print("\nProtected target exposure:")
print(protected_target_exposure)

print("\nHistory-only target exposure:")
print(history_target_exposure)

print("\nPost-break target exposure:")
print(post_break_target_exposure)

print("\nUse Chip values:")
print(sorted(use_chip_values))
```

    ===== M4.2 FULL AUDIT =====
    
    Total rows:
    24386900
    
    Amount parse failures:
    0
    
    Timestamp parse failures:
    0
    
    Timestamp min:
    1991-01-02 07:10:00
    
    Timestamp max:
    2020-02-28 23:58:00
    
    Unassigned temporal rows:
    0
    
    Temporal regions:
    FINAL_TEST_PROTECTED 722,955
    POST_BREAK_DIAGNOSTIC 625,025
    PRE_W_LONG_HISTORY 15,471,192
    TRAIN_2015_2017 5,133,655
    TRAIN_2018 1,721,615
    VALIDATION 712,458
    
    W_LONG rows:
    6,855,270
    
    W_SHORT rows:
    1,721,615
    
    VALIDATION rows:
    712,458
    
    FINAL TEST protected rows:
    722,955
    
    W_SHORT rows outside W_LONG:
    0
    
    Unexpected development target rows:
    0
    
    W_LONG fraud:
    9,606
    
    W_SHORT fraud:
    2,491
    
    VALIDATION fraud:
    1,052
    
    Protected target exposure:
    0
    
    History-only target exposure:
    0
    
    Post-break target exposure:
    0
    
    Use Chip values:
    ['Chip Transaction', 'Online Transaction', 'Swipe Transaction']


## 11.2. Phân tích output

### Artifact size / total rows

Observed:

`24,386,900`

Expected:

`24,386,900`

Interpretation:

Full scan đã xử lý đủ toàn bộ số transaction của canonical raw artifact. Không có dấu hiệu scan kết thúc sớm hoặc bỏ sót chunk.

`PASS`

---

### Amount parse integrity

Observed failures:

`0`

Expected:

`0`

Interpretation:

Toàn bộ raw Amount có thể được chuyển thành `Amount_numeric` bằng transformation đã định nghĩa. Không có transaction nào bị mất numeric representation do parse failure.

Kết quả này xác nhận tính khả thi kỹ thuật của Amount representation, nhưng không quyết định policy xử lý negative Amount hoặc transformation phục vụ model.

`PASS`

---

### Timestamp parse integrity

Observed failures:

`0`

Expected:

`0`

Interpretation:

Toàn bộ transaction có thể được dựng canonical Timestamp từ `Year + Month + Day + Time`.

Không phát sinh missing Timestamp do lỗi parse.

`PASS`

---

### Temporal extent

Observed min:

`1991-01-02 07:10:00`

Observed max:

`2020-02-28 23:58:00`

Expected:

`1991-01-02 07:10:00`
→
`2020-02-28 23:58:00`

Interpretation:

Temporal extent của current artifact khớp hoàn toàn với evidence đã audit ở M2.

Không phát hiện boundary drift trong current runtime.

`PASS`

---

### Temporal-region coverage

UNASSIGNED:

`0`

Observed region counts:

`PRE_W_LONG_HISTORY = 15,471,192`

`TRAIN_2015_2017 = 5,133,655`

`TRAIN_2018 = 1,721,615`

`VALIDATION = 712,458`

`FINAL_TEST_PROTECTED = 722,955`

`POST_BREAK_DIAGNOSTIC = 625,025`

Interpretation:

Mọi transaction đều được ánh xạ vào đúng một temporal region đã định nghĩa.

Không có transaction nằm ngoài temporal contract của M4.2.

Tổng các region bảo toàn toàn bộ `24,386,900` transaction.

`PASS`

---

### W_LONG consistency

Observed rows:

`6,855,270`

Expected:

`6,855,270`

Observed fraud:

`9,606`

Expected:

`9,606`

Interpretation:

W_LONG eligibility được tái tạo chính xác theo M3 canonical boundary.

Cả row count và development target count đều khớp evidence đã khóa.

Kết quả này xác nhận implementation boundary; không chứng minh W_LONG tốt hơn W_SHORT.

`PASS`

---

### W_SHORT consistency

Observed rows:

`1,721,615`

Expected:

`1,721,615`

Observed fraud:

`2,491`

Expected:

`2,491`

Interpretation:

W_SHORT eligibility được tái tạo chính xác theo M3 canonical boundary.

Cả row count và fraud count đều khớp expected evidence.

Không có model-performance evidence mới được tạo ở M4.2.

`PASS`

---

### VALIDATION consistency

Observed rows:

`712,458`

Expected:

`712,458`

Observed fraud:

`1,052`

Expected:

`1,052`

Interpretation:

VALIDATION boundary và target mapping trong current implementation khớp hoàn toàn với M3 specification.

Điều này xác nhận development representation đang sử dụng đúng validation population.

`PASS`

---

### FINAL TEST boundary consistency

Observed protected rows:

`722,955`

Expected:

`722,955`

Interpretation:

FINAL TEST population được nhận diện đúng theo canonical temporal boundary:

`2019-06-01 <= Timestamp < 2019-11-01`

M4.2 chỉ kiểm tra row-boundary consistency và protection metadata.

Không sử dụng fraud count, model metric hoặc performance của FINAL TEST để hỗ trợ development.

`PASS`

---

### W_SHORT subset invariant

Observed violations:

`0`

Expected:

`0`

Interpretation:

Tất cả W_SHORT rows đều đồng thời là W_LONG-eligible rows.

Implementation đúng với quan hệ tập hợp đã định nghĩa:

`W_SHORT ⊂ W_LONG`

`PASS`

---

### Target-domain integrity

Unexpected development target rows:

`0`

Expected:

`0`

Interpretation:

Các row được phép materialize development target chỉ chứa raw target thuộc canonical domain:

`No`

`Yes`

Không phát hiện target value ngoài contract trong TRAIN/VALIDATION development population.

`PASS`

---

### Target isolation

FINAL TEST target exposure:

`0`

History-only target exposure:

`0`

Post-break target exposure:

`0`

Expected:

tất cả bằng `0`

Interpretation:

Target-protection implementation hoạt động đúng.

`target_binary` không được materialize cho:

- PRE_W_LONG_HISTORY;
- FINAL_TEST_PROTECTED;
- POST_BREAK_DIAGNOSTIC.

Do đó history context không vô tình mang target label và protected future regions không expose development target thông qua canonical base representation.

`PASS`

---

### Use Chip domain

Observed:

`Chip Transaction`

`Online Transaction`

`Swipe Transaction`

Expected:

`Chip Transaction`

`Online Transaction`

`Swipe Transaction`

Interpretation:

Current artifact không xuất hiện category `Use Chip` mới hoặc ngoài canonical domain đã audit.

Base representation có thể tiếp tục coi `Use Chip` là categorical transaction-mode candidate.

Final encoding vẫn chưa được quyết định ở M4.2.

`PASS`

---

## 11.3. Tổng hợp Full Audit

Tất cả invariant được kiểm tra trong full-artifact audit đều khớp expected contract.

Không phát hiện:

- row-count mismatch;
- Amount parse failure;
- Timestamp parse failure;
- temporal region chưa được gán;
- split-boundary mismatch;
- W_SHORT/W_LONG relation violation;
- development target-domain anomaly;
- protected target exposure;
- Use Chip category drift.

Full-artifact evidence đủ để chuyển sang Integrity Gate của M4.2.

# 12. M4.2.10 — M4.2 Integrity Gate

Gate này không dùng để tìm insight mới.

Nó chỉ kiểm tra implementation hiện tại có khớp các invariant đã được audit/khóa trước đó hay không.

Nếu bất kỳ assertion nào fail:

- không sửa expected value để notebook PASS;
- không bỏ assertion;
- không tiếp tục sang M4.3;
- giữ nguyên error/output;
- điều tra nguyên nhân.


```python
EXPECTED_USE_CHIP = {
    "Chip Transaction",
    "Online Transaction",
    "Swipe Transaction",
}


# ------------------------------------------------------------
# Artifact / conversion integrity
# ------------------------------------------------------------

assert total_rows == 24_386_900

assert amount_parse_failures == 0
assert timestamp_parse_failures == 0

assert unassigned_region_rows == 0


# ------------------------------------------------------------
# Temporal extent
# ------------------------------------------------------------

assert min_timestamp == pd.Timestamp(
    "1991-01-02 07:10:00"
)

assert max_timestamp == pd.Timestamp(
    "2020-02-28 23:58:00"
)


# ------------------------------------------------------------
# Category-domain integrity
# ------------------------------------------------------------

assert use_chip_values == EXPECTED_USE_CHIP


# ------------------------------------------------------------
# Canonical partition consistency
# ------------------------------------------------------------

assert w_long_rows == 6_855_270
assert w_short_rows == 1_721_615

assert validation_rows == 712_458

assert (
    final_test_protected_rows
    == 722_955
)


# ------------------------------------------------------------
# Development target consistency
# ------------------------------------------------------------

assert w_long_fraud == 9_606
assert w_short_fraud == 2_491
assert validation_fraud == 1_052

assert unexpected_target_rows == 0


# ------------------------------------------------------------
# Eligibility consistency
# ------------------------------------------------------------

assert w_short_not_in_w_long == 0


# ------------------------------------------------------------
# Target protection
# ------------------------------------------------------------

assert protected_target_exposure == 0
assert history_target_exposure == 0
assert post_break_target_exposure == 0


print("===================================")
print("M4.2 FULL GATE: PASS")
print("===================================")
```

    ===================================
    M4.2 FULL GATE: PASS
    ===================================


## Gate interpretation

Cell Integrity Gate đã hoàn thành với:

`M4.2 FULL GATE: PASS`

Không assertion nào fail.

Gate đã kiểm tra thành công các nhóm invariant:

- artifact row count;
- Amount parse integrity;
- Timestamp parse integrity;
- canonical temporal extent;
- Use Chip domain;
- W_LONG row count;
- W_SHORT row count;
- VALIDATION row count;
- FINAL TEST protected row count;
- W_LONG fraud count;
- W_SHORT fraud count;
- VALIDATION fraud count;
- W_SHORT subset invariant;
- development target domain;
- FINAL TEST target isolation;
- history-only target isolation;
- post-break target isolation.

Execution output và full-audit summary nhất quán với nhau.

Không phát hiện trường hợp assertion PASS nhưng summary cho thấy contradiction.

Do đó Integrity Gate có thể được chấp nhận là bằng chứng runtime hợp lệ cho M4.2.

Gate result:

`PASS`

Blocking issue:

`NONE`


# 13. M4.2 Findings

## M4.2-F01 — Artifact / schema

Observed fact:

Current raw artifact tồn tại tại canonical project path, có kích thước `2,354,626,737 bytes`, chứa đủ `24,386,900` transaction và đúng canonical `15-column schema`.

Evidence:

- file existence: `True`;
- file size matches expected;
- actual columns: `15`;
- missing columns: `[]`;
- unexpected columns: `[]`;
- duplicate column names: `0`;
- column order matches: `True`;
- full scan rows: `24,386,900`.

Interpretation:

Current notebook đang làm việc với raw artifact nhất quán với artifact đã audit và không phát hiện schema drift.

Implication:

Các transformation M4.2 có thể tiếp tục dựa trên canonical field contract mà không cần redesign schema handling.

Status:

`CONFIRMED`

---

## M4.2-F02 — Amount representation

Observed fact:

`Amount_numeric` được tạo thành công cho toàn bộ `24,386,900` transaction.

Evidence:

`Amount parse failures = 0`

Interpretation:

Raw Amount có thể được chuyển deterministic sang numeric representation bằng việc loại bỏ ký hiệu tiền tệ / separator mà không cần sửa giá trị theo semantics.

Implementation không áp dụng `abs`, `clip` hoặc loại bỏ Amount âm.

Implication:

M4.2 có thể khóa `Amount_numeric` làm base technical representation.

Cách xử lý negative Amount, transformation hoặc scaling tiếp tục là câu hỏi của M4.3/M4.4.

Status:

`CONFIRMED`

---

## M4.2-F03 — Timestamp representation

Observed fact:

Canonical Timestamp được tạo thành công cho toàn bộ dataset.

Evidence:

`Timestamp parse failures = 0`

Global range:

`1991-01-02 07:10:00`
→
`2020-02-28 23:58:00`

Interpretation:

`Year + Month + Day + Time` có thể được hợp nhất thành một temporal index nhất quán trên toàn artifact.

Implication:

Timestamp có thể được sử dụng làm nguồn chuẩn cho:

- temporal partition;
- causal ordering;
- historical feature;
- rolling-window logic.

Raw physical row order không cần và không được dùng thay cho Timestamp.

Status:

`CONFIRMED`

---

## M4.2-F04 — Temporal partition consistency

Observed fact:

Toàn bộ transaction được gán vào canonical temporal regions và các row counts khớp M3 specification.

Evidence:

`UNASSIGNED = 0`

`W_LONG = 6,855,270`

`W_SHORT = 1,721,615`

`VALIDATION = 712,458`

`FINAL_TEST_PROTECTED = 722,955`

`POST_BREAK_DIAGNOSTIC = 625,025`

`W_SHORT outside W_LONG = 0`

Interpretation:

Temporal boundary implementation trong M4.2 tái tạo chính xác experiment boundaries đã khóa ở M3.

Implication:

Các bước feature/preprocessing sau có thể sử dụng temporal metadata này mà không cần tự định nghĩa split mới.

Kết quả không chọn winner giữa W_LONG và W_SHORT.

Status:

`CONFIRMED`

---

## M4.2-F05 — Target representation / isolation

Observed fact:

Development target mapping phù hợp với canonical domain và target không bị expose ở các vùng được bảo vệ.

Evidence:

`Unexpected development target rows = 0`

`W_LONG fraud = 9,606`

`W_SHORT fraud = 2,491`

`VALIDATION fraud = 1,052`

`FINAL TEST target exposure = 0`

`History-only target exposure = 0`

`Post-break target exposure = 0`

Interpretation:

Mapping:

`No → 0`

`Yes → 1`

hoạt động nhất quán trên development regions.

History context và protected future regions không mang `target_binary` vào canonical base representation.

`target_binary` hiện được materialize dưới dtype `float64` vì representation cần chứa missing value tại protected rows. Đây là implementation detail, không làm thay đổi binary semantics hoặc target-isolation guarantee.

Implication:

Base representation cung cấp target cho TRAIN/VALIDATION mà vẫn duy trì target protection cho history context và protected regions.

Status:

`CONFIRMED`

---

## M4.2-F06 — Category-domain consistency

Observed fact:

`Use Chip` chỉ chứa đúng ba canonical category:

`Chip Transaction`

`Online Transaction`

`Swipe Transaction`

Evidence:

Full-artifact unique values không xuất hiện category ngoài ba giá trị trên.

Interpretation:

Không phát hiện category drift đối với transaction-mode field trong current artifact.

Implication:

`Use Chip` tiếp tục là categorical transaction-level candidate.

Encoding strategy vẫn thuộc M4.4/M4.6 và chưa được quyết định trong M4.2.

Status:

`CONFIRMED`



# 14. Decision Log M4.2

## M4.2-D01 — Raw schema contract

Decision:

Raw input phải khớp canonical 15-column schema.

Current-runtime verification:

Current artifact có đúng 15 cột, không thiếu, không thừa, không trùng tên và đúng canonical order.

Status:

`LOCKED`

---

## M4.2-D02 — Canonical temporal representation

Decision:

`Year + Month + Day + Time`
→
`Timestamp`

Timestamp là temporal index chính của pipeline.

Current-runtime verification:

`Timestamp parse failures = 0` trên toàn bộ `24,386,900` transaction.

Global temporal range khớp canonical evidence.

Status:

`LOCKED`

---

## M4.2-D03 — Amount representation

Decision:

Raw `Amount`
→
`Amount_numeric`

Negative sign được giữ nguyên bởi transformation logic.

Không `abs/drop/clip` trong M4.2.

Current-runtime verification:

`Amount parse failures = 0` trên toàn artifact.

Status:

`LOCKED FOR BASE REPRESENTATION`

Negative-Amount preprocessing policy:

`OPEN — M4.3/M4.4`

---

## M4.2-D04 — Target representation

Decision:

`No → 0`

`Yes → 1`

Canonical field:

`target_binary`

Protected target không materialize trong development representation.

Current-runtime verification:

Unexpected development target rows:

`0`

FINAL TEST target exposure:

`0`

History-only target exposure:

`0`

Post-break target exposure:

`0`

Status:

`LOCKED`

Implementation note:

`target_binary` hiện có dtype `float64` trong base DataFrame do các protected rows được biểu diễn bằng missing value. Binary semantics vẫn được giữ nguyên.

---

## M4.2-D05 — Temporal metadata

Decision:

Gắn:

`temporal_region`

`is_w_long_train`

`is_w_short_train`

`is_validation`

`is_final_test_protected`

theo canonical M3 boundaries.

Current-runtime verification:

Tất cả canonical row counts khớp M3 evidence.

`UNASSIGNED = 0`

`W_SHORT outside W_LONG = 0`

Status:

`LOCKED`

Training-window winner:

`OPEN — REQUIRES EXPERIMENT`

---

## M4.2-D06 — Identifier/history keys

Decision:

`User`

`Card`

`Merchant Name`

→
`HISTORY_KEY_ONLY`

Direct classifier use:

`PROHIBITED`

Status:

`INHERITED — LOCKED`

---

## M4.2-D07 — Categorical codes

Decision:

`MCC`

`Zip`

không được coi là continuous numerical magnitude.

Status:

`INHERITED — LOCKED`

Final encoding:

`OPEN`

---

## M4.2-D08 — Errors?

Decision:

`Errors?`

không propagate vào Model V1 base representation.

Current implementation:

Canonical base representation không chứa `Errors?`.

Status:

`INHERITED — LOCKED`

---

## M4.2-D09 — raw_row_id

Decision:

`raw_row_id` chỉ dùng cho lineage/debug.

Không dùng làm classifier feature hoặc temporal-order signal.

Current verification:

Implementation tạo `raw_row_id` deterministic từ raw row offset và field xuất hiện trong canonical base representation.

Không có model step nào sử dụng `raw_row_id` trong M4.2.

Status:

`LOCKED`

# 15. M4.2 Gate

## G01 — Artifact identity

Result:

`PASS`

Raw artifact tồn tại tại canonical project path và file size khớp artifact đã audit.

---

## G02 — Raw schema

Result:

`PASS`

15 expected columns, không missing, không unexpected, không duplicated name và đúng column order.

---

## G03 — Amount representation

Result:

`PASS`

Full-artifact Amount parse failures:

`0`

---

## G04 — Timestamp representation

Result:

`PASS`

Full-artifact Timestamp parse failures:

`0`

Global min/max khớp canonical temporal extent.

---

## G05 — Temporal-region coverage

Result:

`PASS`

`UNASSIGNED = 0`

Toàn bộ `24,386,900` transaction được gắn vào một canonical temporal region.

---

## G06 — M3 split consistency

Result:

`PASS`

Observed:

`W_LONG = 6,855,270`

`W_SHORT = 1,721,615`

`VALIDATION = 712,458`

`FINAL_TEST_PROTECTED = 722,955`

đều khớp M3 specification.

---

## G07 — W_SHORT subset invariant

Result:

`PASS`

`W_SHORT outside W_LONG = 0`

---

## G08 — Development target mapping

Result:

`PASS`

Unexpected development target rows:

`0`

W_LONG/W_SHORT/VALIDATION fraud counts đều khớp canonical evidence.

---

## G09 — Protected target isolation

Result:

`PASS`

FINAL TEST target exposure:

`0`

History-only target exposure:

`0`

Post-break target exposure:

`0`

---

## G10 — Use Chip domain

Result:

`PASS`

Observed domain:

`Chip Transaction`

`Online Transaction`

`Swipe Transaction`

Không phát hiện category ngoài contract.

---

## G11 — Identifier / field protection

Result:

`PASS`

`User`, `Card`, `Merchant Name` được giữ làm history key và không được định nghĩa là direct classifier feature.

`Errors?` không xuất hiện trong canonical base representation.

Raw `Is Fraud?` không xuất hiện trong canonical base representation.

---

## G12 — No premature preprocessing

Result:

`PASS`

M4.2 chỉ thực hiện deterministic technical representation và temporal metadata.

Không thực hiện:

- imputation;
- categorical encoding;
- scaling;
- Amount semantic correction;
- feature selection;
- behavioral feature construction;
- resampling;
- model training;
- hyperparameter tuning;
- threshold selection.


# M4.2 Gate

Overall:

`PASS`

Blocking issue:

`NONE`

M4.2 Status:

`PASS — READY FOR M4.3`



# 16. Kết luận M4.2

## Mục tiêu đã kiểm tra

M4.2 kiểm tra liệu raw transaction artifact hiện tại có thể được chuyển thành một canonical base representation nhất quán, tái hiện được và tuân thủ các temporal/leakage guardrail đã khóa hay không.

Phạm vi kiểm tra bao gồm:

- artifact/schema consistency;
- raw representation;
- Amount numeric representation;
- Timestamp reconstruction;
- semantic-role contract;
- temporal-region assignment;
- W_LONG/W_SHORT eligibility;
- target mapping;
- protected-target isolation;
- canonical base-representation structure;
- full-artifact integrity.

---

## Những gì được xác minh

Current raw artifact chứa đầy đủ:

`24,386,900 transactions`

và đúng canonical:

`15 raw columns`.

Không phát hiện schema drift.

`Amount_numeric` được tạo thành công trên toàn bộ dataset:

`Amount parse failures = 0`

`Timestamp` được tạo thành công trên toàn bộ dataset:

`Timestamp parse failures = 0`

với global temporal range:

`1991-01-02 07:10:00`
→
`2020-02-28 23:58:00`

Temporal-region assignment bao phủ toàn bộ artifact:

`UNASSIGNED = 0`

Canonical modeling populations được tái tạo đúng:

`W_LONG = 6,855,270`

`W_SHORT = 1,721,615`

`VALIDATION = 712,458`

`FINAL_TEST_PROTECTED = 722,955`

Quan hệ:

`W_SHORT ⊂ W_LONG`

được duy trì với:

`0 violations`

Development target domain không có giá trị ngoài `Yes/No`.

Protected target isolation đạt yêu cầu:

`FINAL TEST exposure = 0`

`history-only exposure = 0`

`post-break exposure = 0`

`Use Chip` không có category drift.

Full Integrity Gate trả về:

`M4.2 FULL GATE: PASS`

---

## Những finding quan trọng

Thứ nhất, representation layer hiện tại tái tạo đúng raw artifact và không làm thay đổi temporal population đã khóa ở M3.

Thứ hai, hai transformation nền tảng:

`Amount → Amount_numeric`

và:

`Year + Month + Day + Time → Timestamp`

hoạt động thành công trên toàn artifact.

Thứ ba, physical dtype và semantic role tiếp tục được tách biệt. Raw identifier và categorical code không bị hiểu sai thành continuous model feature.

Thứ tư, base representation duy trì explicit temporal metadata, nhờ đó các bước phía sau không cần tự định nghĩa lại split.

Thứ năm, target-protection logic hoạt động đúng: target chỉ được materialize cho development regions được phép và không xuất hiện trong history-only, FINAL TEST hoặc post-break representation.

Một implementation note không blocking là `target_binary` hiện được DataFrame biểu diễn dưới dtype `float64` vì protected rows cần missing value. Binary semantics và target isolation không bị ảnh hưởng.

---

## Những decision được khóa

M4.2 khóa các contract sau:

`Raw 15-column schema`

`Timestamp reconstruction`

`Amount_numeric base representation`

`Target No=0 / Yes=1`

`Temporal-region metadata`

`W_LONG/W_SHORT eligibility logic`

`User/Card/Merchant Name = HISTORY_KEY_ONLY`

`MCC/Zip = categorical code semantics`

`Errors? = excluded from Model V1`

`raw_row_id = lineage/debug only`

Các contract trên trở thành đầu vào cố định cho các bước M4 tiếp theo.

---

## Những vấn đề còn OPEN

M4.2 không quyết định:

- negative Amount preprocessing policy;
- zero Amount policy;
- missing location representation;
- Online Transaction location semantics;
- duplicate handling;
- categorical encoding;
- MCC encoding;
- Zip/location encoding;
- feature selection;
- behavioral feature lookback;
- W_LONG hay W_SHORT tốt hơn;
- model family;
- imbalance strategy;
- hyperparameters;
- threshold.

Các câu hỏi này tiếp tục được xử lý ở đúng milestone/substep tương ứng.

---

## Blocking issue

`NONE`

Không phát hiện lỗi representation, temporal-boundary violation hoặc target-isolation violation chặn việc chuyển sang M4.3.

---

## M4.2 Gate

`PASS`

Tất cả invariant trong M4.2 Integrity Gate đều đạt.

---

## Trạng thái cuối

`M4.2 — PASS`

`Base Representation Contract — LOCKED`

`READY FOR M4.3`

---

## Handoff

Next:

`M4.3 — Khóa policy xử lý các vấn đề data-quality trong preprocessing`

M4.3 có thể bắt đầu từ canonical base representation đã khóa tại M4.2 và tập trung vào các câu hỏi:

- negative Amount;
- zero Amount;
- structural location missingness;
- Online Transaction semantics;
- exact duplicate policy;
- category consistency;
- các representation anomaly khác nếu được phát hiện.

M4.3 không cần thiết kế lại:

- raw schema;
- Timestamp construction;
- Amount parser;
- target mapping;
- temporal split;
- W_LONG/W_SHORT boundaries;
- identifier/history-key contract.
