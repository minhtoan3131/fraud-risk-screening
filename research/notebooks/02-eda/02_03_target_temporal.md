# M2.3 — Phân tích target và chiều thời gian

## Mục tiêu

M2.3 tập trung phân tích `target Is Fraud?` và sự thay đổi của target theo thời gian. <br>

Hai vấn đề trung tâm cần làm rõ là: <br>
- mức độ `mất cân bằng lớp` của target; <br>
- mức độ `thay đổi phân bố theo thời gian` của fraud. <br>

Các nhiệm vụ chính gồm: <br>
- xác minh các giá trị thực tế của target trên toàn dataset; <br>
- tính số lượng và tỷ lệ `fraud / non-fraud`; <br>
- định lượng mức mất cân bằng giữa hai class; <br>
- phân tích số transaction, số fraud và fraud rate theo từng năm; <br>
- kiểm tra mức độ bao phủ theo tháng của từng năm; <br>
- phân tích chi tiết theo tháng ở giai đoạn cuối dataset; <br>
- xác định các năm có fraud count hoặc fraud rate cực trị; <br>
- đối chiếu các kết quả quan trọng với những mốc đã xác minh ở M1. <br>

## Ranh giới

Trong M2.3: <br>
- chưa khóa train/test split cuối cùng; <br>
- chưa quyết định metric đánh giá model cuối cùng; <br>
- chưa resampling; <br>
- chưa huấn luyện model; <br>
- chưa lựa chọn model; <br>
- chưa kết luận năm 2019 là test set; <br>
- tiếp tục tuân thủ guardrail `không sử dụng năm 2020 làm final fraud test`. <br>

`Mục tiêu của M2.3 là tạo bằng chứng cho M3 — Thiết kế thí nghiệm, không thay M3 đưa ra quyết định.`

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
# Các biến dùng chung trong M2.3
# ============================================================

DATA_PATH = (
    PROJECT_ROOT
    / DATA_RELATIVE_PATH
)

TARGET_COLUMN = "Is Fraud?"

CHUNK_SIZE = 500_000


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


### Nhận xét thiết lập môi trường

Notebook đã xác định được đúng `PROJECT_ROOT` và đường dẫn tới raw artifact. <br>

Kết quả `File tồn tại: True` xác nhận M2.3 có thể truy cập trực tiếp dataset mà không phụ thuộc vào kernel hoặc biến đã tạo ở M2.1 / M2.2. <br>

Điều này xác nhận `02_03_target_temporal.ipynb` có thể chạy độc lập theo thiết kế mới của project. <br>

`Kết luận: PASS`

## M2.3.1 — Quét toàn bộ target và thông tin thời gian

### Câu hỏi

Toàn bộ dataset thực tế có những giá trị target nào? <br>

Có target bị missing hoặc xuất hiện giá trị ngoài `Yes / No` hay không? <br>

Có thể đồng thời tổng hợp target theo `năm` và `năm-tháng` chỉ trong một lần đọc toàn dataset hay không? <br>

### Phương pháp

Đọc CSV theo từng chunk và chỉ sử dụng ba cột: <br>
`Year` <br>
`Month` <br>
`Is Fraud?` <br>

Trong cùng một lần quét, thu thập: <br>
- tổng số dòng; <br>
- số target missing; <br>
- các giá trị target khác nhau; <br>
- số lượng từng target; <br>
- số transaction theo năm; <br>
- số fraud / non-fraud theo năm; <br>
- số transaction và fraud / non-fraud theo năm-tháng. <br>

Raw target trong file không bị thay đổi. <br>

Một Series tạm thời được loại bỏ khoảng trắng đầu/cuối chỉ để kiểm tra representation. <br>


```python
M23_USECOLS = [
    "Year",
    "Month",
    TARGET_COLUMN,
]

total_rows = 0
chunk_count = 0
missing_target_count = 0

target_values = set()
target_counts = Counter()

year_transaction_counts = Counter()
year_fraud_counts = Counter()
year_nonfraud_counts = Counter()

year_month_transaction_counts = Counter()
year_month_fraud_counts = Counter()
year_month_nonfraud_counts = Counter()


for chunk_number, chunk in enumerate(
    pd.read_csv(
        DATA_PATH,
        usecols=M23_USECOLS,
        chunksize=CHUNK_SIZE,
        dtype={
            TARGET_COLUMN: "string",
        },
    ),
    start=1,
):
    chunk_count = chunk_number
    total_rows += len(chunk)

    # -----------------------------------------
    # Target representation dùng cho kiểm tra
    # -----------------------------------------

    target = (
        chunk[TARGET_COLUMN]
        .astype("string")
        .str.strip()
    )

    # Chuỗi rỗng cũng được xem như missing
    target = target.mask(
        target.eq("")
    )

    missing_target_count += int(
        target.isna().sum()
    )

    valid_target = target.dropna()

    target_values.update(
        valid_target.unique().tolist()
    )

    target_counts.update(
        valid_target
        .value_counts()
        .to_dict()
    )

    fraud_mask = (
        target.eq("Yes")
        .fillna(False)
    )

    nonfraud_mask = (
        target.eq("No")
        .fillna(False)
    )

    # -----------------------------------------
    # Tổng hợp theo năm
    # -----------------------------------------

    year_transaction_counts.update(
        chunk["Year"]
        .value_counts()
        .to_dict()
    )

    year_fraud_counts.update(
        chunk.loc[
            fraud_mask,
            "Year",
        ]
        .value_counts()
        .to_dict()
    )

    year_nonfraud_counts.update(
        chunk.loc[
            nonfraud_mask,
            "Year",
        ]
        .value_counts()
        .to_dict()
    )

    # -----------------------------------------
    # Tổng hợp theo năm-tháng
    # -----------------------------------------

    year_month_transaction = (
        chunk
        .groupby(
            ["Year", "Month"]
        )
        .size()
    )

    year_month_transaction_counts.update(
        {
            tuple(key): int(value)
            for key, value
            in year_month_transaction.items()
        }
    )

    year_month_fraud = (
        chunk.loc[fraud_mask]
        .groupby(
            ["Year", "Month"]
        )
        .size()
    )

    year_month_fraud_counts.update(
        {
            tuple(key): int(value)
            for key, value
            in year_month_fraud.items()
        }
    )

    year_month_nonfraud = (
        chunk.loc[nonfraud_mask]
        .groupby(
            ["Year", "Month"]
        )
        .size()
    )

    year_month_nonfraud_counts.update(
        {
            tuple(key): int(value)
            for key, value
            in year_month_nonfraud.items()
        }
    )

    if (
        chunk_number % 10 == 0
        or len(chunk) < CHUNK_SIZE
    ):
        print(
            f"Đã xử lý {chunk_number} chunk "
            f"- tổng số dòng: {total_rows:,}"
        )


print("\nHoàn tất quét M2.3.")
print("Số chunk:", chunk_count)
print("Tổng số dòng:", f"{total_rows:,}")
```

    Đã xử lý 10 chunk - tổng số dòng: 5,000,000
    Đã xử lý 20 chunk - tổng số dòng: 10,000,000
    Đã xử lý 30 chunk - tổng số dòng: 15,000,000
    Đã xử lý 40 chunk - tổng số dòng: 20,000,000
    Đã xử lý 49 chunk - tổng số dòng: 24,386,900
    
    Hoàn tất quét M2.3.
    Số chunk: 49
    Tổng số dòng: 24,386,900


### Nhận xét M2.3.1

Toàn bộ dataset đã được quét thành công theo `49 chunk`. <br>

Tổng số dòng được xử lý là `24,386,900`, đúng với kích thước artifact đã được xác minh ở Milestone 1. <br>

Quá trình quét hoàn tất mà không phát sinh lỗi đọc dữ liệu. <br>

Trong cùng một lần đọc, notebook đã thu thập được các thống kê cần thiết cho phân tích target theo toàn dataset, theo năm và theo năm-tháng. <br>

Cách xử lý theo chunk cho phép thực hiện full scan mà không cần giữ toàn bộ hơn 24 triệu transaction trong RAM. <br>

`Kết luận: PASS`

## M2.3.2 — Phân bố target trên toàn dataset

### Câu hỏi

Target `Is Fraud?` có đúng hai class `Yes / No` hay không? <br>

Fraud chiếm bao nhiêu transaction và bao nhiêu phần trăm toàn dataset? <br>

Mức chênh lệch giữa `non-fraud` và `fraud` lớn đến đâu? <br>

Nếu luôn dự đoán class chiếm đa số thì accuracy tham chiếu sẽ là bao nhiêu? <br>

### Phương pháp

Sử dụng số đếm target thu được từ lần quét toàn dataset ở M2.3.1. <br>

Ngoài `count` và `rate`, tính thêm: <br>
- tỷ lệ `non-fraud / fraud`; <br>
- accuracy của chiến lược luôn dự đoán class chiếm đa số. <br>

Các giá trị này chỉ dùng để mô tả mức mất cân bằng. <br>

M2.3 chưa quyết định metric đánh giá model cuối cùng. <br>


```python
EXPECTED_TARGET_VALUES = {
    "Yes",
    "No",
}

unexpected_target_values = (
    target_values
    - EXPECTED_TARGET_VALUES
)

fraud_count = int(
    target_counts.get(
        "Yes",
        0,
    )
)

nonfraud_count = int(
    target_counts.get(
        "No",
        0,
    )
)

other_or_missing_target_count = (
    total_rows
    - fraud_count
    - nonfraud_count
)


target_distribution_df = pd.DataFrame(
    [
        {
            "target": "No",
            "count": nonfraud_count,
        },
        {
            "target": "Yes",
            "count": fraud_count,
        },
    ]
)

target_distribution_df["rate_pct"] = (
    target_distribution_df["count"]
    / total_rows
    * 100
)


fraud_rate_pct = (
    fraud_count
    / total_rows
    * 100
)

nonfraud_to_fraud_ratio = (
    nonfraud_count / fraud_count
    if fraud_count > 0
    else float("inf")
)

majority_baseline_accuracy_pct = (
    max(
        fraud_count,
        nonfraud_count,
    )
    / total_rows
    * 100
)


print("Các giá trị target:")
print(sorted(target_values))

print("\nTarget ngoài Yes / No:")
print(sorted(unexpected_target_values))

print("\nTarget missing:")
print(missing_target_count)

print(
    "\nTarget ngoài Yes / No hoặc missing:"
)
print(other_or_missing_target_count)

print("\nPhân bố target:")
display(target_distribution_df)


target_summary = pd.Series(
    {
        "Tổng transaction":
            total_rows,

        "Fraud":
            fraud_count,

        "Non-fraud":
            nonfraud_count,

        "Fraud rate (%)":
            fraud_rate_pct,

        "Non-fraud / Fraud ratio":
            nonfraud_to_fraud_ratio,

        "Majority-class baseline accuracy (%)":
            majority_baseline_accuracy_pct,
    }
)

display(target_summary)
```

    Các giá trị target:
    ['No', 'Yes']
    
    Target ngoài Yes / No:
    []
    
    Target missing:
    0
    
    Target ngoài Yes / No hoặc missing:
    0
    
    Phân bố target:



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
      <th>target</th>
      <th>count</th>
      <th>rate_pct</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>No</td>
      <td>24357143</td>
      <td>99.87798</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Yes</td>
      <td>29757</td>
      <td>0.12202</td>
    </tr>
  </tbody>
</table>
</div>



    Tổng transaction                        2.438690e+07
    Fraud                                   2.975700e+04
    Non-fraud                               2.435714e+07
    Fraud rate (%)                          1.220204e-01
    Non-fraud / Fraud ratio                 8.185349e+02
    Majority-class baseline accuracy (%)    9.987798e+01
    dtype: float64


### Nhận xét M2.3.2

Target thực tế chỉ có đúng hai giá trị: <br>
`No` <br>
`Yes` <br>

Không phát hiện: <br>
- target ngoài `Yes / No`; <br>
- target missing; <br>
- target rỗng sau khi loại khoảng trắng đầu/cuối. <br>

Phân bố toàn dataset: <br>
`Non-fraud: 24,357,143 transaction — 99.877980%` <br>
`Fraud: 29,757 transaction — 0.122020%` <br>

Trung bình có khoảng `818.5 non-fraud` cho mỗi `1 fraud`. <br>

Nếu một classifier luôn dự đoán `non-fraud`, accuracy vẫn đạt khoảng `99.877980%` dù hoàn toàn không phát hiện được fraud. <br>

Điều này xác nhận dataset có `mất cân bằng lớp rất mạnh`. <br>

Vì vậy `Accuracy` không thể được sử dụng đơn độc để đánh giá khả năng phát hiện fraud. <br>

M2.3 chưa khóa metric cuối cùng; quyết định metric cụ thể được chuyển sang M3. <br>

`Kết luận: target hợp lệ về mặt cấu trúc nhưng mất cân bằng lớp ở mức rất mạnh.`

## M2.3.3 — Phân bố target theo từng năm

### Câu hỏi

Số lượng transaction và fraud thay đổi như thế nào qua từng năm? <br>

Fraud rate có tương đối ổn định hay thay đổi đáng kể theo thời gian? <br>

Có năm nào không có fraud hoặc có fraud rate khác rõ so với các năm còn lại hay không? <br>

### Phương pháp

Từ kết quả quét toàn dataset, tạo một bảng với mỗi dòng tương ứng một năm. <br>

Các trường được tính gồm: <br>
`transaction_count` <br>
`fraud_count` <br>
`nonfraud_count` <br>
`other_or_missing_target_count` <br>
`fraud_rate_pct` <br>

Ở bước này chưa đặt một threshold chủ quan để định nghĩa năm nào là bất thường. <br>
Trước tiên chỉ tạo bằng chứng từ dữ liệu thực tế. <br>


```python
years = sorted(
    year_transaction_counts.keys()
)


year_target_df = pd.DataFrame(
    [
        {
            "Year":
                year,

            "transaction_count":
                int(
                    year_transaction_counts.get(
                        year,
                        0,
                    )
                ),

            "fraud_count":
                int(
                    year_fraud_counts.get(
                        year,
                        0,
                    )
                ),

            "nonfraud_count":
                int(
                    year_nonfraud_counts.get(
                        year,
                        0,
                    )
                ),
        }
        for year in years
    ]
)


year_target_df[
    "other_or_missing_target_count"
] = (
    year_target_df[
        "transaction_count"
    ]
    - year_target_df[
        "fraud_count"
    ]
    - year_target_df[
        "nonfraud_count"
    ]
)


year_target_df[
    "fraud_rate_pct"
] = (
    year_target_df[
        "fraud_count"
    ]
    / year_target_df[
        "transaction_count"
    ]
    * 100
)


display(year_target_df)


print(
    "Tổng transaction theo năm:",
    year_target_df[
        "transaction_count"
    ].sum(),
)

print(
    "Tổng fraud theo năm:",
    year_target_df[
        "fraud_count"
    ].sum(),
)

print(
    "Tổng non-fraud theo năm:",
    year_target_df[
        "nonfraud_count"
    ].sum(),
)
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
      <th>fraud_count</th>
      <th>nonfraud_count</th>
      <th>other_or_missing_target_count</th>
      <th>fraud_rate_pct</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1991</td>
      <td>1585</td>
      <td>0</td>
      <td>1585</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1992</td>
      <td>5134</td>
      <td>0</td>
      <td>5134</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1993</td>
      <td>8378</td>
      <td>0</td>
      <td>8378</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1994</td>
      <td>14316</td>
      <td>0</td>
      <td>14316</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1995</td>
      <td>20928</td>
      <td>0</td>
      <td>20928</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1996</td>
      <td>29945</td>
      <td>10</td>
      <td>29935</td>
      <td>0</td>
      <td>0.033395</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1997</td>
      <td>49753</td>
      <td>32</td>
      <td>49721</td>
      <td>0</td>
      <td>0.064318</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1998</td>
      <td>78345</td>
      <td>32</td>
      <td>78313</td>
      <td>0</td>
      <td>0.040845</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1999</td>
      <td>118250</td>
      <td>24</td>
      <td>118226</td>
      <td>0</td>
      <td>0.020296</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2000</td>
      <td>177729</td>
      <td>171</td>
      <td>177558</td>
      <td>0</td>
      <td>0.096214</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2001</td>
      <td>257998</td>
      <td>354</td>
      <td>257644</td>
      <td>0</td>
      <td>0.137210</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2002</td>
      <td>350732</td>
      <td>139</td>
      <td>350593</td>
      <td>0</td>
      <td>0.039631</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2003</td>
      <td>466408</td>
      <td>311</td>
      <td>466097</td>
      <td>0</td>
      <td>0.066680</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2004</td>
      <td>597003</td>
      <td>620</td>
      <td>596383</td>
      <td>0</td>
      <td>0.103852</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2005</td>
      <td>746653</td>
      <td>229</td>
      <td>746424</td>
      <td>0</td>
      <td>0.030670</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2006</td>
      <td>908793</td>
      <td>1118</td>
      <td>907675</td>
      <td>0</td>
      <td>0.123020</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2007</td>
      <td>1064483</td>
      <td>1881</td>
      <td>1062602</td>
      <td>0</td>
      <td>0.176705</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2008</td>
      <td>1223460</td>
      <td>3710</td>
      <td>1219750</td>
      <td>0</td>
      <td>0.303238</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2009</td>
      <td>1355434</td>
      <td>1140</td>
      <td>1354294</td>
      <td>0</td>
      <td>0.084106</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2010</td>
      <td>1491225</td>
      <td>3835</td>
      <td>1487390</td>
      <td>0</td>
      <td>0.257171</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2011</td>
      <td>1570551</td>
      <td>55</td>
      <td>1570496</td>
      <td>0</td>
      <td>0.003502</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2012</td>
      <td>1610829</td>
      <td>1333</td>
      <td>1609496</td>
      <td>0</td>
      <td>0.082752</td>
    </tr>
    <tr>
      <th>22</th>
      <td>2013</td>
      <td>1650917</td>
      <td>2018</td>
      <td>1648899</td>
      <td>0</td>
      <td>0.122235</td>
    </tr>
    <tr>
      <th>23</th>
      <td>2014</td>
      <td>1672343</td>
      <td>1052</td>
      <td>1671291</td>
      <td>0</td>
      <td>0.062906</td>
    </tr>
    <tr>
      <th>24</th>
      <td>2015</td>
      <td>1701371</td>
      <td>3281</td>
      <td>1698090</td>
      <td>0</td>
      <td>0.192844</td>
    </tr>
    <tr>
      <th>25</th>
      <td>2016</td>
      <td>1708924</td>
      <td>3579</td>
      <td>1705345</td>
      <td>0</td>
      <td>0.209430</td>
    </tr>
    <tr>
      <th>26</th>
      <td>2017</td>
      <td>1723360</td>
      <td>255</td>
      <td>1723105</td>
      <td>0</td>
      <td>0.014797</td>
    </tr>
    <tr>
      <th>27</th>
      <td>2018</td>
      <td>1721615</td>
      <td>2491</td>
      <td>1719124</td>
      <td>0</td>
      <td>0.144690</td>
    </tr>
    <tr>
      <th>28</th>
      <td>2019</td>
      <td>1723938</td>
      <td>2087</td>
      <td>1721851</td>
      <td>0</td>
      <td>0.121060</td>
    </tr>
    <tr>
      <th>29</th>
      <td>2020</td>
      <td>336500</td>
      <td>0</td>
      <td>336500</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>


    Tổng transaction theo năm: 24386900
    Tổng fraud theo năm: 29757
    Tổng non-fraud theo năm: 24357143


### Nhận xét M2.3.3

Số transaction tăng mạnh theo thời gian, từ quy mô rất nhỏ ở các năm đầu lên khoảng `1.7 triệu transaction/năm` ở giai đoạn 2015–2019. <br>

Tuy nhiên số fraud và fraud rate không tăng ổn định theo số transaction. <br>

Fraud rate thay đổi rất mạnh giữa các năm. <br>

Một số ví dụ đáng chú ý: <br>
`2008: 0.303238%` <br>
`2010: 0.257171%` <br>
`2011: 0.003502%` <br>
`2016: 0.209430%` <br>
`2017: 0.014797%` <br>

Đặc biệt: <br>
`2010 → 2011` fraud rate giảm hơn `70 lần`. <br>
`2016 → 2017` fraud rate giảm khoảng `14 lần`. <br>

Fraud rate cao nhất quan sát được trong bảng lớn hơn rất nhiều so với những năm có fraud rate thấp. <br>

Do đó không có bằng chứng để xem fraud rate là một phân bố ổn định theo thời gian. <br>

Dataset thể hiện `dịch chuyển phân bố theo thời gian` rất rõ. <br>

Điều này củng cố guardrail rằng evaluation chính không nên dựa trên random split đơn giản, vì random split có thể trộn các giai đoạn có phân bố khác nhau vào cùng train và test. <br>

Năm `2019` có fraud rate `0.121060%`, nhìn ở cấp năm khá gần fraud rate toàn dataset `0.122020%`. <br>

Tuy nhiên cần kiểm tra ở cấp tháng trước khi diễn giải 2019 là một giai đoạn đồng nhất. <br>

`Kết luận: fraud distribution thay đổi mạnh theo thời gian; phân tích cấp năm chưa đủ để lựa chọn evaluation window.`

## M2.3.4 — Kiểm tra mức độ bao phủ theo tháng của từng năm

### Câu hỏi

Mỗi năm trong dataset có đủ dữ liệu của `12 tháng` hay không? <br>

Có năm nào chỉ được quan sát một phần, khiến việc so sánh số lượng transaction tuyệt đối với những năm đầy đủ cần thận trọng hay không? <br>

### Phương pháp

Tạo bảng tổng hợp `năm-tháng` từ kết quả quét toàn dataset. <br>

Sau đó với mỗi năm xác định: <br>
- số tháng có dữ liệu; <br>
- tháng đầu tiên; <br>
- tháng cuối cùng; <br>
- tổng số transaction; <br>
- tổng số fraud; <br>
- có đủ 12 tháng hay không. <br>


```python
year_month_keys = sorted(
    year_month_transaction_counts.keys()
)


year_month_target_df = pd.DataFrame(
    [
        {
            "Year":
                year,

            "Month":
                month,

            "transaction_count":
                int(
                    year_month_transaction_counts.get(
                        (year, month),
                        0,
                    )
                ),

            "fraud_count":
                int(
                    year_month_fraud_counts.get(
                        (year, month),
                        0,
                    )
                ),

            "nonfraud_count":
                int(
                    year_month_nonfraud_counts.get(
                        (year, month),
                        0,
                    )
                ),
        }
        for year, month
        in year_month_keys
    ]
)


year_month_target_df[
    "other_or_missing_target_count"
] = (
    year_month_target_df[
        "transaction_count"
    ]
    - year_month_target_df[
        "fraud_count"
    ]
    - year_month_target_df[
        "nonfraud_count"
    ]
)


year_month_target_df[
    "fraud_rate_pct"
] = (
    year_month_target_df[
        "fraud_count"
    ]
    / year_month_target_df[
        "transaction_count"
    ]
    * 100
)


year_coverage_df = (
    year_month_target_df
    .groupby(
        "Year",
        as_index=False,
    )
    .agg(
        observed_months=(
            "Month",
            "nunique",
        ),

        first_month=(
            "Month",
            "min",
        ),

        last_month=(
            "Month",
            "max",
        ),

        transaction_count=(
            "transaction_count",
            "sum",
        ),

        fraud_count=(
            "fraud_count",
            "sum",
        ),
    )
)


year_coverage_df[
    "full_12_months"
] = (
    year_coverage_df[
        "observed_months"
    ]
    == 12
)


display(year_coverage_df)
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
      <th>observed_months</th>
      <th>first_month</th>
      <th>last_month</th>
      <th>transaction_count</th>
      <th>fraud_count</th>
      <th>full_12_months</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1991</td>
      <td>12</td>
      <td>1</td>
      <td>12</td>
      <td>1585</td>
      <td>0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1992</td>
      <td>12</td>
      <td>1</td>
      <td>12</td>
      <td>5134</td>
      <td>0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1993</td>
      <td>12</td>
      <td>1</td>
      <td>12</td>
      <td>8378</td>
      <td>0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1994</td>
      <td>12</td>
      <td>1</td>
      <td>12</td>
      <td>14316</td>
      <td>0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1995</td>
      <td>12</td>
      <td>1</td>
      <td>12</td>
      <td>20928</td>
      <td>0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1996</td>
      <td>12</td>
      <td>1</td>
      <td>12</td>
      <td>29945</td>
      <td>10</td>
      <td>True</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1997</td>
      <td>12</td>
      <td>1</td>
      <td>12</td>
      <td>49753</td>
      <td>32</td>
      <td>True</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1998</td>
      <td>12</td>
      <td>1</td>
      <td>12</td>
      <td>78345</td>
      <td>32</td>
      <td>True</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1999</td>
      <td>12</td>
      <td>1</td>
      <td>12</td>
      <td>118250</td>
      <td>24</td>
      <td>True</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2000</td>
      <td>12</td>
      <td>1</td>
      <td>12</td>
      <td>177729</td>
      <td>171</td>
      <td>True</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2001</td>
      <td>12</td>
      <td>1</td>
      <td>12</td>
      <td>257998</td>
      <td>354</td>
      <td>True</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2002</td>
      <td>12</td>
      <td>1</td>
      <td>12</td>
      <td>350732</td>
      <td>139</td>
      <td>True</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2003</td>
      <td>12</td>
      <td>1</td>
      <td>12</td>
      <td>466408</td>
      <td>311</td>
      <td>True</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2004</td>
      <td>12</td>
      <td>1</td>
      <td>12</td>
      <td>597003</td>
      <td>620</td>
      <td>True</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2005</td>
      <td>12</td>
      <td>1</td>
      <td>12</td>
      <td>746653</td>
      <td>229</td>
      <td>True</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2006</td>
      <td>12</td>
      <td>1</td>
      <td>12</td>
      <td>908793</td>
      <td>1118</td>
      <td>True</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2007</td>
      <td>12</td>
      <td>1</td>
      <td>12</td>
      <td>1064483</td>
      <td>1881</td>
      <td>True</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2008</td>
      <td>12</td>
      <td>1</td>
      <td>12</td>
      <td>1223460</td>
      <td>3710</td>
      <td>True</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2009</td>
      <td>12</td>
      <td>1</td>
      <td>12</td>
      <td>1355434</td>
      <td>1140</td>
      <td>True</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2010</td>
      <td>12</td>
      <td>1</td>
      <td>12</td>
      <td>1491225</td>
      <td>3835</td>
      <td>True</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2011</td>
      <td>12</td>
      <td>1</td>
      <td>12</td>
      <td>1570551</td>
      <td>55</td>
      <td>True</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2012</td>
      <td>12</td>
      <td>1</td>
      <td>12</td>
      <td>1610829</td>
      <td>1333</td>
      <td>True</td>
    </tr>
    <tr>
      <th>22</th>
      <td>2013</td>
      <td>12</td>
      <td>1</td>
      <td>12</td>
      <td>1650917</td>
      <td>2018</td>
      <td>True</td>
    </tr>
    <tr>
      <th>23</th>
      <td>2014</td>
      <td>12</td>
      <td>1</td>
      <td>12</td>
      <td>1672343</td>
      <td>1052</td>
      <td>True</td>
    </tr>
    <tr>
      <th>24</th>
      <td>2015</td>
      <td>12</td>
      <td>1</td>
      <td>12</td>
      <td>1701371</td>
      <td>3281</td>
      <td>True</td>
    </tr>
    <tr>
      <th>25</th>
      <td>2016</td>
      <td>12</td>
      <td>1</td>
      <td>12</td>
      <td>1708924</td>
      <td>3579</td>
      <td>True</td>
    </tr>
    <tr>
      <th>26</th>
      <td>2017</td>
      <td>12</td>
      <td>1</td>
      <td>12</td>
      <td>1723360</td>
      <td>255</td>
      <td>True</td>
    </tr>
    <tr>
      <th>27</th>
      <td>2018</td>
      <td>12</td>
      <td>1</td>
      <td>12</td>
      <td>1721615</td>
      <td>2491</td>
      <td>True</td>
    </tr>
    <tr>
      <th>28</th>
      <td>2019</td>
      <td>12</td>
      <td>1</td>
      <td>12</td>
      <td>1723938</td>
      <td>2087</td>
      <td>True</td>
    </tr>
    <tr>
      <th>29</th>
      <td>2020</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>336500</td>
      <td>0</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>


### Nhận xét M2.3.4

Các năm từ `1991` đến `2019` đều có dữ liệu của đủ `12 tháng`. <br>

Riêng năm `2020` chỉ có dữ liệu của `2 tháng`: <br>
`tháng 1` <br>
`tháng 2` <br>

Do đó số `336,500 transaction` của năm 2020 không đại diện cho một năm hoàn chỉnh và không nên được so sánh trực tiếp với số transaction tuyệt đối của các năm có đủ 12 tháng. <br>

2020 đồng thời có `0 fraud`. <br>

Như vậy năm 2020 có hai hạn chế cùng lúc: <br>
- chỉ bao phủ một phần năm; <br>
- không chứa positive class. <br>

Điều này củng cố quyết định đã khóa từ M1 rằng `2020 không phù hợp làm final fraud test set`. <br>

Năm `2019` có đủ 12 tháng, vì vậy về mức độ bao phủ thời gian nó đầy đủ hơn 2020. <br>

Tuy nhiên việc 2019 có đủ 12 tháng chưa đủ để kết luận toàn bộ năm có cùng một fraud regime. <br>

`Kết luận: 2020 là partial-year period và không phù hợp làm final fraud evaluation window.`

## M2.3.5 — Phân bố target theo tháng ở giai đoạn 2018–2020

### Câu hỏi

Ở giai đoạn cuối dataset, fraud có phân bố tương đối đều giữa các tháng hay xuất hiện sự thay đổi đáng chú ý? <br>

Năm 2019 và phần dữ liệu năm 2020 có đặc điểm gì khi quan sát ở cấp tháng? <br>

### Phương pháp

Lọc bảng tổng hợp năm-tháng cho giai đoạn `2018–2020`. <br>

Mỗi dòng thể hiện: <br>
`Year` <br>
`Month` <br>
`transaction_count` <br>
`fraud_count` <br>
`nonfraud_count` <br>
`fraud_rate_pct` <br>

Giai đoạn `2018–2020` chỉ được chọn để quan sát phần cuối của chuỗi thời gian. <br>

Việc này không đồng nghĩa với việc khóa training window hoặc test set. <br>


```python
recent_monthly_target_df = (
    year_month_target_df[
        year_month_target_df["Year"]
        .between(
            2018,
            2020,
        )
    ]
    .copy()
    .reset_index(drop=True)
)


display(
    recent_monthly_target_df
)


print(
    "Số tháng được quan sát:",
    len(
        recent_monthly_target_df
    ),
)
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
      <th>Month</th>
      <th>transaction_count</th>
      <th>fraud_count</th>
      <th>nonfraud_count</th>
      <th>other_or_missing_target_count</th>
      <th>fraud_rate_pct</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2018</td>
      <td>1</td>
      <td>146357</td>
      <td>145</td>
      <td>146212</td>
      <td>0</td>
      <td>0.099073</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2018</td>
      <td>2</td>
      <td>131276</td>
      <td>183</td>
      <td>131093</td>
      <td>0</td>
      <td>0.139401</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2018</td>
      <td>3</td>
      <td>146272</td>
      <td>229</td>
      <td>146043</td>
      <td>0</td>
      <td>0.156558</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2018</td>
      <td>4</td>
      <td>141688</td>
      <td>269</td>
      <td>141419</td>
      <td>0</td>
      <td>0.189854</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2018</td>
      <td>5</td>
      <td>145438</td>
      <td>167</td>
      <td>145271</td>
      <td>0</td>
      <td>0.114826</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2018</td>
      <td>6</td>
      <td>141827</td>
      <td>154</td>
      <td>141673</td>
      <td>0</td>
      <td>0.108583</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2018</td>
      <td>7</td>
      <td>146638</td>
      <td>156</td>
      <td>146482</td>
      <td>0</td>
      <td>0.106384</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2018</td>
      <td>8</td>
      <td>146983</td>
      <td>216</td>
      <td>146767</td>
      <td>0</td>
      <td>0.146956</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2018</td>
      <td>9</td>
      <td>141557</td>
      <td>262</td>
      <td>141295</td>
      <td>0</td>
      <td>0.185084</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2018</td>
      <td>10</td>
      <td>145473</td>
      <td>181</td>
      <td>145292</td>
      <td>0</td>
      <td>0.124422</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2018</td>
      <td>11</td>
      <td>141843</td>
      <td>300</td>
      <td>141543</td>
      <td>0</td>
      <td>0.211501</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2018</td>
      <td>12</td>
      <td>146263</td>
      <td>229</td>
      <td>146034</td>
      <td>0</td>
      <td>0.156567</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2019</td>
      <td>1</td>
      <td>146097</td>
      <td>203</td>
      <td>145894</td>
      <td>0</td>
      <td>0.138949</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2019</td>
      <td>2</td>
      <td>132102</td>
      <td>156</td>
      <td>131946</td>
      <td>0</td>
      <td>0.118091</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2019</td>
      <td>3</td>
      <td>147202</td>
      <td>215</td>
      <td>146987</td>
      <td>0</td>
      <td>0.146058</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2019</td>
      <td>4</td>
      <td>141409</td>
      <td>218</td>
      <td>141191</td>
      <td>0</td>
      <td>0.154163</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2019</td>
      <td>5</td>
      <td>145648</td>
      <td>260</td>
      <td>145388</td>
      <td>0</td>
      <td>0.178513</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2019</td>
      <td>6</td>
      <td>142399</td>
      <td>215</td>
      <td>142184</td>
      <td>0</td>
      <td>0.150984</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2019</td>
      <td>7</td>
      <td>146599</td>
      <td>156</td>
      <td>146443</td>
      <td>0</td>
      <td>0.106413</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2019</td>
      <td>8</td>
      <td>147139</td>
      <td>247</td>
      <td>146892</td>
      <td>0</td>
      <td>0.167868</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2019</td>
      <td>9</td>
      <td>141744</td>
      <td>142</td>
      <td>141602</td>
      <td>0</td>
      <td>0.100181</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2019</td>
      <td>10</td>
      <td>145074</td>
      <td>275</td>
      <td>144799</td>
      <td>0</td>
      <td>0.189558</td>
    </tr>
    <tr>
      <th>22</th>
      <td>2019</td>
      <td>11</td>
      <td>141946</td>
      <td>0</td>
      <td>141946</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>23</th>
      <td>2019</td>
      <td>12</td>
      <td>146579</td>
      <td>0</td>
      <td>146579</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>24</th>
      <td>2020</td>
      <td>1</td>
      <td>170731</td>
      <td>0</td>
      <td>170731</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25</th>
      <td>2020</td>
      <td>2</td>
      <td>165769</td>
      <td>0</td>
      <td>165769</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>


    Số tháng được quan sát: 26


### Nhận xét M2.3.5

Ở năm `2018`, cả 12 tháng đều có fraud. <br>

Fraud rate theo tháng năm 2018 dao động khoảng từ `0.099%` đến `0.212%`. <br>

Ở năm `2019`, từ tháng 1 đến tháng 10 đều xuất hiện fraud. <br>

Tuy nhiên: <br>
`2019-11: 141,946 transaction — 0 fraud` <br>
`2019-12: 146,579 transaction — 0 fraud` <br>
`2020-01: 170,731 transaction — 0 fraud` <br>
`2020-02: 165,769 transaction — 0 fraud` <br>

Như vậy có `4 tháng liên tiếp`, từ `11/2019` đến `02/2020`, với tổng cộng `625,025 transaction` nhưng không xuất hiện một fraud nào. <br>

Toàn bộ `2,087 fraud` của năm 2019 đều nằm trong giai đoạn `tháng 1 → tháng 10`. <br>

Fraud rate của `01/2019 → 10/2019` xấp xỉ `0.1454%`, rất gần fraud rate toàn năm 2018 là `0.144690%`. <br>

Trong khi đó, fraud rate từ `11/2019` trở đi giảm đột ngột về `0%`. <br>

Điều này cho thấy một `điểm gãy thời gian` xuất hiện vào khoảng `11/2019`. <br>

Vì vậy fraud rate toàn năm 2019 là `0.121060%` tuy gần mức toàn dataset nhưng đã che khuất hai giai đoạn rất khác nhau bên trong cùng một năm. <br>

M2.3 hiện chưa có bằng chứng để kết luận nguyên nhân của điểm gãy này. <br>
Không nên tự động quy nó cho lỗi dữ liệu hoặc cơ chế tạo dữ liệu synthetic nếu chưa có bằng chứng bổ sung. <br>

Tuy nhiên về mặt evaluation, đây là một finding quan trọng: `2019 không nên được xem như một khối thời gian đồng nhất chỉ dựa trên thống kê cấp năm`. <br>

`Kết luận: tồn tại một temporal regime shift rất mạnh bắt đầu khoảng 11/2019.`


```python
period_specs = [
    ("2018", 2018, 1, 2018, 12),
    ("2019-01 đến 2019-10", 2019, 1, 2019, 10),
    ("2019-11 đến 2019-12", 2019, 11, 2019, 12),
    ("2020-01 đến 2020-02", 2020, 1, 2020, 2),
]

period_rows = []

for (
    label,
    start_year,
    start_month,
    end_year,
    end_month,
) in period_specs:

    start_key = (
        start_year * 100
        + start_month
    )

    end_key = (
        end_year * 100
        + end_month
    )

    month_key = (
        year_month_target_df["Year"] * 100
        + year_month_target_df["Month"]
    )

    period_df = (
        year_month_target_df[
            month_key.between(
                start_key,
                end_key,
            )
        ]
    )

    transaction_count = int(
        period_df[
            "transaction_count"
        ].sum()
    )

    fraud_count_period = int(
        period_df[
            "fraud_count"
        ].sum()
    )

    fraud_rate_period = (
        fraud_count_period
        / transaction_count
        * 100
    )

    period_rows.append(
        {
            "period": label,
            "transaction_count":
                transaction_count,
            "fraud_count":
                fraud_count_period,
            "fraud_rate_pct":
                fraud_rate_period,
        }
    )


period_comparison_df = pd.DataFrame(
    period_rows
)

display(
    period_comparison_df
)
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
      <th>period</th>
      <th>transaction_count</th>
      <th>fraud_count</th>
      <th>fraud_rate_pct</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2018</td>
      <td>1721615</td>
      <td>2491</td>
      <td>0.144690</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2019-01 đến 2019-10</td>
      <td>1435413</td>
      <td>2087</td>
      <td>0.145394</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2019-11 đến 2019-12</td>
      <td>288525</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2020-01 đến 2020-02</td>
      <td>336500</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>


## M2.3.6 — Xác định các năm có giá trị cực trị

### Câu hỏi

Năm nào có `fraud rate` cao nhất? <br>

Năm nào có fraud rate thấp nhất trong các năm vẫn xuất hiện fraud? <br>

Có năm nào hoàn toàn không có fraud hay không? <br>

### Phương pháp

Không tự đặt một threshold để định nghĩa `năm bất thường`. <br>

Thay vào đó: <br>
- liệt kê các năm có `fraud_count = 0`; <br>
- lấy 5 năm có `fraud_rate_pct` cao nhất; <br>
- lấy 5 năm có fraud rate thấp nhất trong các năm vẫn có ít nhất một fraud. <br>

Các bảng này chỉ được dùng để xác định những giai đoạn cần xem kỹ hơn. <br>

Nguyên nhân của các cực trị chưa được giải thích ở bước code. <br>


```python
zero_fraud_years_df = (
    year_target_df[
        year_target_df[
            "fraud_count"
        ]
        == 0
    ]
    .copy()
)


highest_fraud_rate_years_df = (
    year_target_df
    .nlargest(
        5,
        "fraud_rate_pct",
    )
    .copy()
)


lowest_positive_fraud_rate_years_df = (
    year_target_df[
        year_target_df[
            "fraud_count"
        ]
        > 0
    ]
    .nsmallest(
        5,
        "fraud_rate_pct",
    )
    .copy()
)


print("Các năm không có fraud:")
display(
    zero_fraud_years_df
)


print(
    "\n5 năm có fraud rate cao nhất:"
)
display(
    highest_fraud_rate_years_df
)


print(
    "\n5 năm có fraud rate thấp nhất "
    "trong các năm vẫn có fraud:"
)
display(
    lowest_positive_fraud_rate_years_df
)
```

    Các năm không có fraud:



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
      <th>fraud_count</th>
      <th>nonfraud_count</th>
      <th>other_or_missing_target_count</th>
      <th>fraud_rate_pct</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1991</td>
      <td>1585</td>
      <td>0</td>
      <td>1585</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1992</td>
      <td>5134</td>
      <td>0</td>
      <td>5134</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1993</td>
      <td>8378</td>
      <td>0</td>
      <td>8378</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1994</td>
      <td>14316</td>
      <td>0</td>
      <td>14316</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1995</td>
      <td>20928</td>
      <td>0</td>
      <td>20928</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>2020</td>
      <td>336500</td>
      <td>0</td>
      <td>336500</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>


    
    5 năm có fraud rate cao nhất:



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
      <th>fraud_count</th>
      <th>nonfraud_count</th>
      <th>other_or_missing_target_count</th>
      <th>fraud_rate_pct</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>17</th>
      <td>2008</td>
      <td>1223460</td>
      <td>3710</td>
      <td>1219750</td>
      <td>0</td>
      <td>0.303238</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2010</td>
      <td>1491225</td>
      <td>3835</td>
      <td>1487390</td>
      <td>0</td>
      <td>0.257171</td>
    </tr>
    <tr>
      <th>25</th>
      <td>2016</td>
      <td>1708924</td>
      <td>3579</td>
      <td>1705345</td>
      <td>0</td>
      <td>0.209430</td>
    </tr>
    <tr>
      <th>24</th>
      <td>2015</td>
      <td>1701371</td>
      <td>3281</td>
      <td>1698090</td>
      <td>0</td>
      <td>0.192844</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2007</td>
      <td>1064483</td>
      <td>1881</td>
      <td>1062602</td>
      <td>0</td>
      <td>0.176705</td>
    </tr>
  </tbody>
</table>
</div>


    
    5 năm có fraud rate thấp nhất trong các năm vẫn có fraud:



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
      <th>fraud_count</th>
      <th>nonfraud_count</th>
      <th>other_or_missing_target_count</th>
      <th>fraud_rate_pct</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>20</th>
      <td>2011</td>
      <td>1570551</td>
      <td>55</td>
      <td>1570496</td>
      <td>0</td>
      <td>0.003502</td>
    </tr>
    <tr>
      <th>26</th>
      <td>2017</td>
      <td>1723360</td>
      <td>255</td>
      <td>1723105</td>
      <td>0</td>
      <td>0.014797</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1999</td>
      <td>118250</td>
      <td>24</td>
      <td>118226</td>
      <td>0</td>
      <td>0.020296</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2005</td>
      <td>746653</td>
      <td>229</td>
      <td>746424</td>
      <td>0</td>
      <td>0.030670</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1996</td>
      <td>29945</td>
      <td>10</td>
      <td>29935</td>
      <td>0</td>
      <td>0.033395</td>
    </tr>
  </tbody>
</table>
</div>


### Nhận xét M2.3.6

Các năm không có fraud gồm: <br>
`1991` <br>
`1992` <br>
`1993` <br>
`1994` <br>
`1995` <br>
`2020` <br>

Tuy nhiên bối cảnh của hai nhóm này khác nhau. <br>

Các năm 1991–1995 có số transaction tương đối nhỏ, trong khi riêng phần dữ liệu năm 2020 đã có `336,500 transaction` nhưng vẫn không có fraud. <br>

Các năm có fraud rate cao nhất gồm: <br>
`2008: 0.303238%` <br>
`2010: 0.257171%` <br>
`2016: 0.209430%` <br>
`2015: 0.192844%` <br>
`2007: 0.176705%` <br>

Các năm có fraud rate thấp nhất nhưng vẫn có fraud gồm: <br>
`2011: 0.003502%` <br>
`2017: 0.014797%` <br>
`1999: 0.020296%` <br>
`2005: 0.030670%` <br>
`1996: 0.033395%` <br>

Fraud rate năm 2008 cao hơn năm 2011 khoảng `86 lần`. <br>

Khoảng biến động này quá lớn để coi fraud prevalence là ổn định theo năm. <br>

Kết quả củng cố bằng chứng về `temporal distribution shift` và cho thấy các giai đoạn đánh giá sau này cần được chọn có ý thức về thời gian. <br>

Các cực trị này hiện chỉ được ghi nhận như bằng chứng EDA. <br>
M2.3 chưa kết luận nguyên nhân tạo ra chúng. <br>

`Kết luận: fraud rate có các regime theo thời gian rất khác nhau, không chỉ dao động nhỏ quanh một mức cố định.`

## M2.3.7 — Đối chiếu kết quả với các mốc đã xác minh ở M1

### Câu hỏi

Các thống kê target và thời gian được tính độc lập trong M2.3 có nhất quán với những mốc đã được xác minh ở Milestone 1 hay không? <br>

### Phương pháp

Sau khi M2.3 đã tự tính kết quả từ raw artifact, mới đối chiếu với một số mốc tham chiếu từ M1: <br>
- tổng số transaction; <br>
- tổng số fraud; <br>
- tổng số non-fraud; <br>
- số transaction và fraud của năm 2019; <br>
- số transaction và fraud của năm 2020. <br>

Các giá trị của M1 chỉ dùng để `cross-check`, không dùng thay cho kết quả được tính trong M2.3. <br>


```python
year_2019 = (
    year_target_df
    .loc[
        year_target_df[
            "Year"
        ]
        == 2019
    ]
    .iloc[0]
)


year_2020 = (
    year_target_df
    .loc[
        year_target_df[
            "Year"
        ]
        == 2020
    ]
    .iloc[0]
)


m23_consistency_checks = pd.Series(
    {
        "Tổng transaction = 24,386,900":
            total_rows
            == 24_386_900,

        "Fraud = 29,757":
            fraud_count
            == 29_757,

        "Non-fraud = 24,357,143":
            nonfraud_count
            == 24_357_143,

        "Target missing = 0":
            missing_target_count
            == 0,

        "Chỉ có target Yes / No":
            target_values
            == {
                "Yes",
                "No",
            },

        "2019 transaction = 1,723,938":
            int(
                year_2019[
                    "transaction_count"
                ]
            )
            == 1_723_938,

        "2019 fraud = 2,087":
            int(
                year_2019[
                    "fraud_count"
                ]
            )
            == 2_087,

        "2020 transaction = 336,500":
            int(
                year_2020[
                    "transaction_count"
                ]
            )
            == 336_500,

        "2020 fraud = 0":
            int(
                year_2020[
                    "fraud_count"
                ]
            )
            == 0,
    }
)


display(
    m23_consistency_checks
)


print(
    "Tất cả phép đối chiếu đều khớp:",
    m23_consistency_checks.all(),
)
```


    Tổng transaction = 24,386,900    True
    Fraud = 29,757                   True
    Non-fraud = 24,357,143           True
    Target missing = 0               True
    Chỉ có target Yes / No           True
    2019 transaction = 1,723,938     True
    2019 fraud = 2,087               True
    2020 transaction = 336,500       True
    2020 fraud = 0                   True
    dtype: bool


    Tất cả phép đối chiếu đều khớp: True


### Nhận xét M2.3.7

Tất cả các phép đối chiếu với Milestone 1 đều trả về `True`. <br>

Các kết quả khớp gồm: <br>
`Tổng transaction = 24,386,900` <br>
`Fraud = 29,757` <br>
`Non-fraud = 24,357,143` <br>
`Target missing = 0` <br>
`Target chỉ có Yes / No` <br>
`2019 transaction = 1,723,938` <br>
`2019 fraud = 2,087` <br>
`2020 transaction = 336,500` <br>
`2020 fraud = 0` <br>

Điều này xác nhận các thống kê M2.3 được tính độc lập từ raw artifact nhưng vẫn nhất quán với kết quả đã được audit ở M1. <br>

Không phát hiện bất nhất cần điều tra giữa hai milestone. <br>

`Kết luận: PASS`

# Tổng kết M2.3 — Phân tích target và chiều thời gian

## Các phát hiện chính

Target `Is Fraud?` có cấu trúc sạch: chỉ gồm `Yes / No`, không có missing và không có giá trị ngoài miền kỳ vọng. <br>

Dataset có `24,386,900 transaction`, trong đó: <br>
`Fraud: 29,757 — 0.122020%` <br>
`Non-fraud: 24,357,143 — 99.877980%` <br>

Target có `mất cân bằng lớp rất mạnh`, với khoảng `818.5 non-fraud / 1 fraud`. <br>

Fraud rate thay đổi mạnh theo thời gian và không thể xem là ổn định giữa các năm. <br>

Các năm có fraud rate rất cao và rất thấp xen kẽ nhau; ví dụ `2008 = 0.303238%` trong khi `2011 = 0.003502%`. <br>

Phân tích theo tháng phát hiện một điểm gãy đặc biệt quan trọng: <br>
`từ 11/2019 đến 02/2020 có 625,025 transaction liên tiếp theo bốn tháng nhưng 0 fraud`. <br>

Điểm gãy này không thể nhìn thấy rõ nếu chỉ quan sát fraud rate toàn năm 2019. <br>

## Kết luận về mức mất cân bằng target

Dataset có class imbalance ở mức rất mạnh. <br>

Một classifier luôn dự đoán `non-fraud` vẫn có thể đạt accuracy khoảng `99.877980%`. <br>

Do đó `Accuracy` không đủ để đánh giá khả năng fraud screening nếu sử dụng độc lập. <br>

M2.3 chưa khóa bộ metric cuối cùng. <br>
M3 phải lựa chọn metric phù hợp với positive class hiếm và mục tiêu phát hiện fraud. <br>

## Kết luận về sự thay đổi theo thời gian

Fraud prevalence thể hiện `temporal distribution shift` rõ rệt. <br>

Fraud rate không thay đổi từ từ theo một xu hướng đơn giản mà có những bước nhảy rất lớn giữa các năm và giữa các giai đoạn. <br>

Vì vậy random split đơn giản có nguy cơ trộn nhiều temporal regime vào cả train và test, tạo evaluation không phản ánh đúng bài toán dự đoán tương lai. <br>

M3 phải tiếp tục sử dụng nguyên tắc `temporal evaluation`. <br>

## Kết luận về năm 2020

Năm 2020 chỉ bao phủ `tháng 1 và tháng 2`, với `336,500 transaction` và `0 fraud`. <br>

2020 vừa là `partial-year period`, vừa không chứa positive class. <br>

Do đó 2020 không phù hợp làm `final fraud test set`. <br>

Quyết định này giữ nguyên guardrail đã khóa từ M1. <br>

Nếu M3 muốn sử dụng 2020, nó chỉ nên được xem xét như một giai đoạn chẩn đoán / stress period riêng với mục đích được ghi rõ, không thay thế final evaluation có cả positive và negative class. <br>

## Kết luận về giai đoạn 2019

Ở cấp năm, 2019 có: <br>
`1,723,938 transaction` <br>
`2,087 fraud` <br>
`fraud rate = 0.121060%` <br>

Fraud rate này gần fraud rate toàn dataset. <br>

Tuy nhiên phân tích theo tháng cho thấy 2019 không phải một giai đoạn đồng nhất. <br>

Từ `01/2019 → 10/2019`, toàn bộ 2,087 fraud của năm 2019 đều xuất hiện và fraud rate xấp xỉ `0.1454%`. <br>

Từ `11/2019 → 12/2019`, fraud count giảm về `0`, và trạng thái này tiếp tục sang `01/2020 → 02/2020`. <br>

Do đó không nên chỉ dựa trên fraud rate toàn năm để quyết định `2019 = test set`. <br>

2019 vẫn có thể là một candidate evaluation period, nhưng ranh giới thời gian cụ thể phải được M3 xem xét dựa trên điểm gãy từ `11/2019`. <br>

## Các vấn đề cần chuyển sang M3

M3 cần quyết định: <br>
- temporal train / validation / test boundaries; <br>
- cách xử lý điểm gãy bắt đầu khoảng `11/2019` trong thiết kế evaluation; <br>
- evaluation window nào chứa đủ positive class để metric có ý nghĩa; <br>
- bộ metric phù hợp với class imbalance; <br>
- cách sử dụng hoặc không sử dụng giai đoạn 2020; <br>
- cách đảm bảo final test giữ nguyên distribution thực tế của cửa sổ thời gian được chọn. <br>

M2.3 không khóa các quyết định trên. <br>

## Các vấn đề cần tiếp tục theo dõi ở M2

M2.4–M2.6 cần kiểm tra liệu sự thay đổi của các feature giao dịch có đi kèm với các temporal regime đã phát hiện hay không. <br>

Đặc biệt cần lưu ý giai đoạn cuối 2019 khi phân tích `Amount`, `Use Chip`, `MCC`, location và mối quan hệ feature-target. <br>

M2.8 cần sử dụng bằng chứng temporal này khi so sánh các candidate training window và khả năng xây behavioral feature. <br>

Nguyên nhân chính xác của zero-fraud regime từ `11/2019` hiện vẫn là một câu hỏi mở. <br>

## Quyết định M2.3

`Decision ID: M2.3-D01` <br>
`Target integrity: PASS` <br>
`Class-imbalance analysis: PASS` <br>
`Year-level temporal analysis: PASS` <br>
`Month-level temporal analysis: PASS` <br>
`Consistency with M1: PASS` <br>
`Temporal distribution shift: CONFIRMED` <br>
`Late-2019 zero-fraud regime: DETECTED` <br>
`Blocking issue: NONE` <br>
`Status: PASS WITH FINDINGS` <br>

## Bước tiếp theo

`Next: M2.4 — Phân tích từng feature giao dịch`
