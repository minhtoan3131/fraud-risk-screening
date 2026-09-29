# Final Pipeline

Thư mục này chứa implementation Python chuẩn dùng cho hệ thống sàng lọc rủi ro gian lận giao dịch.

## Trách nhiệm

- kiểm tra và chuẩn hóa dữ liệu đầu vào;
- xây đặc trưng giao dịch và đặc trưng lịch sử theo quan hệ thời gian hợp lệ;
- áp dụng preprocessing đã được đóng băng;
- nạp model và trạng thái preprocessing đã được xác minh;
- tính risk score;
- áp dụng quy tắc screening;
- cung cấp interface ổn định cho application.

## Ranh giới

- không phụ thuộc runtime vào `research/`;
- không chứa notebook nghiên cứu;
- không huấn luyện model trong luồng inference;
- không tự thay đổi threshold;
- không chứa logic giao diện của application;
- không ghi đè artifact chính thức trong hoạt động thông thường.

## Cấu trúc mục tiêu

Các thành phần kỹ thuật sẽ được tổ chức theo source package, cấu hình, kiểm thử,
artifact chính thức và script vận hành có tên trung tính, dễ đọc.

