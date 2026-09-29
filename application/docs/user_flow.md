# User Flow

## Vai trò của chương trình

Chương trình hỗ trợ sàng lọc dấu hiệu gian lận giao dịch tài chính.

`risk_score` là positive-class model score dùng cho screening. Không mô tả
giá trị này là xác suất gian lận thực tế đã hiệu chỉnh và không coi output
là kết luận gian lận cuối cùng.

## Chế độ tương tác

### Cold-start

Người dùng cung cấp giao dịch hiện tại nhưng không có lịch sử trước đó.

Ứng dụng phải hiển thị rõ:

- `cold_start=True`;
- warning tương ứng;
- risk score và screening result vẫn được tính bằng final pipeline.

### Có lịch sử rõ ràng

Người dùng cung cấp lịch sử strict-prior của cùng User+Card.

Yêu cầu:

- cùng User;
- cùng Card;
- mọi history timestamp nhỏ hơn timestamp của giao dịch hiện tại;
- không tự tra cứu hoặc tự tạo history.

### Dataset replay/demo

Dataset path phải được cung cấp rõ ràng từ bên ngoài.

Không có silent fallback sang `research/`.

## Kết quả hiển thị

Tối thiểu gồm:

- transaction summary;
- risk score;
- threshold;
- screening prediction;
- cold-start/history status;
- warnings;
- model identity.

## Xử lý lỗi

Ứng dụng phải trình bày lỗi rõ ràng khi gặp:

- Amount không hợp lệ;
- Timestamp không hợp lệ;
- thiếu trường bắt buộc;
- history không hợp lệ;
- lỗi load/compatibility của official artifacts.

Input không hợp lệ không được chuyển thành prediction mặc định.

## Ranh giới kiến trúc

Application:

- không preprocess riêng;
- không tính behavioral features riêng;
- không load estimator trực tiếp;
- không tự áp threshold;
- không import `research`;
- chỉ sử dụng API của `final_pipeline`.
