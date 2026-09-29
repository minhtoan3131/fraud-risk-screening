# Demo Scenarios

Thư mục này chứa các tình huống giao dịch được xây dựng riêng để minh họa tính thực tế và hợp lý của bài toán, tức là đưa ra các giao dịch mà con người cũng có thể nhận diện được các dấu hiệu khả nghi theo các đặc trưng mà đã nhận ra theo 3.4.5 trong báo cáo


## Các tình huống

### `normal_transaction/`

Tình huống giao dịch có lịch sử tương đối ổn định và không xuất hiện thay đổi đáng chú ý.

Mục đích là minh họa trường hợp chương trình không gắn cờ giao dịch:

prediction = 0

### `suspicious_transaction/`

Tình huống được xây dựng với một số dấu hiệu đáng chú ý so với lịch sử, chẳng hạn:

- Amount hiện tại thay đổi rõ so với mức giao dịch trước đó;
- merchant hiện tại là merchant mới;
- lịch sử giao dịch được cung cấp để chương trình có thể tạo các Behavioral Features.

Mục đích là minh họa trường hợp chương trình tạo screening positive:

prediction = 1