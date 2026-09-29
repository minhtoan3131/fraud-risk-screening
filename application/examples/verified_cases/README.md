# Verified Cases

Thư mục này chứa các trường hợp giao dịch được lấy từ tập FINAL TEST đã khóa của dự án và được chuẩn bị lại dưới dạng đầu vào cho Application.

Mục đích của các case này là kiểm tra và minh họa rằng chương trình cuối có thể tái hiện đúng hành vi của mô hình đã được đánh giá trước đó.

## Các trường hợp

### `fraud_tp/`

- Ground truth: Fraud (`Is Fraud? = Yes`).
- Kết quả screening: positive (`prediction = 1`).
- Loại kết quả: True Positive (TP).

Case này minh họa một giao dịch Fraud trong FINAL TEST được mô hình gắn cờ đúng.

### `non_fraud_tn/`

- Ground truth: Non-fraud (`Is Fraud? = No`).
- Kết quả screening: negative (`prediction = 0`).
- Loại kết quả: True Negative (TN).

Case này minh họa một giao dịch Non-fraud trong FINAL TEST không bị mô hình gắn cờ.

## Cấu trúc mỗi case

Mỗi thư mục có thể bao gồm:

- giao dịch hiện tại dùng cho inference;
- strict-prior history của đúng User/Card;
- metadata hoặc tài liệu mô tả ground truth và kết quả dự kiến.

## Lưu ý

Ground truth chỉ được sử dụng để đối chiếu kết quả.

Trường `Is Fraud?` không phải feature đầu vào của mô hình và không được truyền vào Application khi thực hiện inference.

Các case trong thư mục này khác với `demo_scenarios/` ở chỗ chúng có ground truth từ FINAL TEST và được dùng làm bằng chứng kiểm tra hành vi của chương trình.