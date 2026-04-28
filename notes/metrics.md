Trong bài báo, **SNR** và **PSNR** là hai chỉ số quan trọng nhưng đo lường các khía cạnh khác nhau của hệ thống truyền thông ngữ nghĩa:

### 1. SNR (Signal-to-Noise Ratio - Tỉ số tín hiệu trên nhiễu)
*   **Đo lường gì:** SNR đo **chất lượng của kênh truyền dẫn**.
*   **Định nghĩa:** Nó được xác định bằng tỉ số giữa công suất phát trung bình ($P$) và phương sai của nhiễu kênh ($\sigma^2$), cụ thể là $P/\sigma^2$.
*   **Vai trò trong bài:** SNR được sử dụng như một **biến số môi trường** để đánh giá hiệu suất của hệ thống trong các điều kiện kênh khác nhau (từ -18 dB đến 18 dB). SNR càng cao nghĩa là điều kiện truyền dẫn càng ít nhiễu.

### 2. PSNR (Peak Signal-to-Noise Ratio - Tỉ số tín hiệu cực đại trên nhiễu)
*   **Đo lường gì:** PSNR đo **chất lượng phục hồi hình ảnh** tại máy thu.
*   **Định nghĩa:** Đây là tiêu chuẩn để đánh giá mức độ trung thực của hình ảnh nguồn được tái tạo so với hình ảnh gốc.
*   **Vai trò trong bài:** PSNR là một **chỉ số hiệu suất (metric)** để so sánh các phương pháp mã hóa và điều chế khác nhau. Nó liên quan trực tiếp đến sai số bình phương trung bình (MSE) giữa ảnh gốc và ảnh khôi phục. PSNR càng cao thì hình ảnh khôi phục càng sắc nét và gần với bản gốc.

### Sự khác biệt chính:
| Đặc điểm | SNR | PSNR |
| :--- | :--- | :--- |
| **Vị trí đo** | Đo tại **kênh truyền** (giữa máy phát và máy thu). | Đo tại **đầu ra của máy thu** (sau khi đã khôi phục ảnh). |
| **Mục đích** | Mô tả độ khắc nghiệt của môi trường truyền thông. | Mô tả hiệu quả của thuật toán nén và truyền hình ảnh. |
| **Đối tượng** | Tín hiệu điện và nhiễu trắng Gaussian (AWGN). | Pixel hình ảnh và lỗi tái tạo. |

**Tóm lại:** SNR cho biết "đường truyền tốt hay xấu", còn PSNR cho biết "hình ảnh nhận được rõ hay mờ". Trong các thí nghiệm của bài báo, khi SNR tăng lên (kênh truyền tốt hơn), PSNR của hệ thống JCM cũng tăng theo, cho thấy khả năng khôi phục hình ảnh tốt hơn ở các mức tín hiệu mạnh.