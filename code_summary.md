Dưới đây là nội dung bạn cung cấp đã được chuyển sang **tiếng Việt có dấu đầy đủ**, giữ nguyên cấu trúc Markdown:

---

# Code Summary: Joint Coding-Modulation for Digital Semantic Communications via VAE

Tài liệu này giải thích chi tiết cách repo hiện thực ý tưởng trong paper:
Y. Bo, Y. Duan, S. Shao and M. Tao, *"Joint Coding-Modulation for Digital Semantic Communications via Variational Autoencoder,"* IEEE Transactions on Communications, 2024.

## 1) Bài toán và ý tưởng chính

Mục tiêu của hệ thống là truyền "ngữ nghĩa" (semantic information) của ảnh qua kênh AWGN thay vì truyền bit-level theo pipeline truyền thống.

Model giải đồng thời 2 nhiệm vụ:

1. Nhiệm vụ semantic classification (dự đoán nhãn 10 lớp CIFAR-10).
2. Nhiệm vụ semantic reconstruction (tái tạo lại ảnh đầu vào).

Điểm đặc biệt của paper/repo:

1. Kết hợp coding + modulation thành một khối học được (joint coding-modulation).
2. Dùng latent xác suất và rời rạc hóa bằng Gumbel-Softmax để map trực tiếp vào chòm điểm điều chế (BPSK/4QAM/16QAM/64QAM).
3. Hướng tới tối ưu end-to-end dưới ảnh hưởng nhiễu kênh.

## 2) Tổng quan cấu trúc file

* main.py: điểm vào, parse argument, tạo dataloader, route train/test.
* modules.py: các module nền (Encoder, Decoder_Recon, Decoder_Class, AWGN, normalize, ResidualBlock).
* network.py: mô hình JCM (digital semantic communication, có modulation rời rạc).
* analog_network.py: baseline analog network (không qua lớp xác suất rời rạc).
* train.py: train cho JCM.
* train_analog.py: train cho analog net.
* evaluation.py: tính metric và đánh giá.
* utils.py: metric (MSE/PSNR/SSIM), save checkpoint, vẽ constellation distribution.

## 3) Data và tiền xử lý

Dataset: CIFAR-10 trong main.py.

Tiền xử lý:

1. Train: RandomHorizontalFlip + ToTensor + Normalize(mean=0.5, std=0.5).
2. Test: ToTensor + Normalize(mean=0.5, std=0.5).

Ảnh về miền [-1, 1] sau normalize. Một số metric (PSNR/MSE) map lại về [0, 1] trước khi tính.

## 4) Kiến trúc mô hình trong repo

### 4.1 Encoder (modules.py)

Encoder là CNN residual:

1. Conv 3→64 + BN + PReLU.
2. Nhiều ResidualBlock, downsample dần (stride=2).
3. Đầu ra kích thước 4×4 với số kênh phụ thuộc modulation:

   * BPSK: channel_use.
   * M-QAM (4/16/64QAM): channel_use * 2.

Ý nghĩa: M-QAM cần 2 trục I/Q, nên latent dimension lớn hơn BPSK.

### 4.2 Decoder cho reconstruction (Decoder_Recon)

Input là latent sau kênh (z_hat), reshape thành map 4×4, qua:

1. Conv + residual blocks.
2. DepthToSpace để upscale.
3. Conv cuối về 3 kênh RGB.

Nhiệm vụ: tái tạo ảnh đầu vào.

### 4.3 Decoder cho classification (Decoder_Class)

Sử dụng kiến trúc spinal fully-connected (4 tầng).
Đầu ra cuối là 10 lớp CIFAR-10.

Nhiệm vụ: giữ thông tin ngữ nghĩa phục vụ phân loại.

## 5) JCM (network.py): mapping từ latent xác suất sang chòm điều chế

Pipeline forward của JCM:

1. x → encoder → feature map → flatten.
2. Qua prob_convs tạo logits có dạng [B, L, K], với:

   * L: số symbol latent,
   * K: số category (2 cho BPSK/4QAM, 4 cho 16QAM, 8 cho 64QAM).
3. discrete_code = gumbel_softmax(logits, hard=True, tau=1.5).
4. Map discrete_code sang giá trị chòm (constellation levels) trong modulation().
5. normalize() để chuẩn hóa công suất trung bình.
6. Truyền qua kênh AWGN:

   * mode train dùng snr_train,
   * mode test dùng snr_test.
7. z_hat → decoder_class và decoder_recon.

### 5.1 Hàm modulation()

Với mỗi symbol, model chọn 1 category rồi map sang mức biên độ chòm:

* BPSK: {-1, +1}
* 4QAM: {-1, +1}
* 16QAM: {-3, -1, +1, +3}
* 64QAM: {-7, -5, -3, -1, +1, +3, +5, +7}

Với 16/64QAM, code được sắp xếp thành cặp (I, Q).

### 5.2 Kênh AWGN (modules.py)

Kênh nhiễu được tạo bởi:

n = 1 / (10^(SNR/10))

noise ~ N(0, n)

z_hat = z + noise

Model học trực tiếp trong điều kiện kênh này.

## 6) Analog network (analog_network.py)

AnalogNet bỏ qua bước logits → gumbel-softmax → modulation rời rạc.
Nó lấy latent liên tục từ encoder (qua AvgPool2d), normalize, AWGN, rồi decode cho class + recon.

Vai trò trong quy trình:

1. Pretrain analog để khởi tạo tham số tốt.
2. JCM train tiếp từ checkpoint analog (đặc biệt hữu ích cho hội tụ).

## 7) Hàm loss và objective đa nhiệm vụ

Trong train.py và train_analog.py:

Loss tổng:

L = L_cls + λ * L_rec

với:

1. L_cls: CrossEntropyLoss(y_class, y_true).
2. L_rec: MSELoss(y_recon, x).
3. λ = tradeoff_lambda (mặc định 200).

Ý nghĩa:

1. Tăng λ → ưu tiên reconstruction chất lượng ảnh.
2. Giảm λ → ưu tiên semantic classification.

## 8) Tối ưu và lịch học

* Optimizer: Adam.
* JCM có 2 nhóm tham số:

  1. Base params lr = lr.
  2. prob_convs lr = lr/2 (ổn định hơn cho lớp xác suất/modulation).
* Scheduler: CosineAnnealingWarmRestarts.

Checkpoint logic:

1. Theo dõi best test acc trong 10 epoch cuối.
2. Lưu file vào models/<mod_method>/...

## 9) Đánh giá và metric (evaluation.py + utils.py)

Mỗi batch test tính:

1. Accuracy: độ đúng phân loại.
2. MSE: sai số tái tạo ảnh.
3. PSNR: chất lượng ảnh tái tạo theo dB.
4. SSIM: độ tương đồng cấu trúc.

Tổng kết trên tập test bằng trung bình theo số mẫu.

### 9.1 Công thức metric trong code

* MSE/PSNR đều đưa ảnh từ [-1, 1] về [0, 1] trước khi tính.
* SSIM tính trên tensor hiện tại với data_range=1.0.

Ghi chú:

1. Log in ra chữ "ssmi" là typo, giá trị thực là SSIM.
2. Warning meshgrid khi test (PyTorch warning) không làm đổi metric.

## 10) Mapping kết quả với bảng README/paper

README đưa bảng theo SNR và modulation.
Nếu bạn chạy:

```bash
python main.py --mode test --mod_method 64qam --pretrain_analog 0 --load_checkpoint 1
```

thì nếu không truyền thêm SNR, mặc định:

1. snr_train = 12
2. snr_test = 12

Trong mode test, kênh sử dụng snr_test.
Vì vậy khi đối chiếu bảng 64QAM, cần ưu tiên map theo PSNR vì nó phân biệt rõ theo SNR hơn Accuracy.

Ví dụ kết quả:

* acc ~ 0.875
* psnr ~ 24.253

map hợp lý nhất vào hàng SNR = 12, 64QAM vì PSNR 24.2522 rất sát.

## 11) Quy trình chạy khuyến nghị (thực tế)

### 11.1 Pretrain analog

```bash
python main.py --mode train --pretrain_analog 1 --mod_method 64qam --snr_train 12
```

### 11.2 Train JCM với khởi tạo analog

```bash
python main.py --mode train --pretrain_analog 0 --mod_method 64qam --snr_train 12
```

### 11.3 Test JCM

```bash
python main.py --mode test --pretrain_analog 0 --load_checkpoint 1 --mod_method 64qam --snr_train 12 --snr_test 12
```

Nếu muốn tạo đường cong theo SNR, lặp test với:

snr_test ∈ {18, 12, 6, 0, -6, -12, -18}.

## 12) Những điểm cần lưu ý khi đọc/thử nghiệm

1. Repo đặt tên checkpoint theo snr_train trong tên file load test; thông thường bạn sẽ để snr_train trùng với model đã train.
2. Kết quả có dao động nhỏ do ngẫu nhiên và implementation details, nên cần so sánh theo xu hướng/chênh lệch nhỏ thay vì đòi đúng tuyệt đối từng số thập phân.
3. Đối với 16/64QAM, repo có thêm thống kê và vẽ phân bố constellation (thư mục cons_fig) để quan sát sự sử dụng chòm điểm.

## 13) Tóm tắt 1 câu

Code hiện thực paper theo hướng end-to-end semantic communication, trong đó latent được học, rời rạc hóa và map trực tiếp vào modulation constellation dưới AWGN, đồng tối ưu cho cả classification và reconstruction bằng loss đa nhiệm vụ có hệ số trade-off.


## 14) Phiên bản "học nhanh trong 15 phút" (tiếng Việt có dấu)

Mục tiêu của phần này là giúp bạn nắm được 80% giá trị của repo chỉ trong khoảng 15 phút, trước khi đi sâu vào chi tiết.

### Phút 0-2: Nắm bài toán trong 3 ý

1. Đây là semantic communication: truyền thông tin ngữ nghĩa của ảnh, không truyền bit theo pipeline truyền thống.
2. Mô hình học end-to-end qua kênh AWGN, nên trong quá trình train đã "thấy" nhiễu kênh.
3. Một latent chung phục vụ 2 đầu ra:
   - Phân loại ảnh (classification).
   - Tái tạo ảnh (reconstruction).

### Phút 2-5: Đọc đúng luồng chạy trong mã

Bắt đầu từ main.py:
1. Parse tham số và tạo DataLoader CIFAR-10.
2. Nếu pretrain_analog=1 thì train AnalogNet.
3. Nếu pretrain_analog=0:
   - mode=train: nạp analog pretrained rồi train JCM.
   - mode=test: nạp checkpoint JCM và gọi EVAL.

Bạn chỉ cần nhớ: main.py là file điều phối toàn bộ thực nghiệm.

### Phút 5-8: Hiểu lõi kỹ thuật của JCM

Trong network.py, forward của JCM đi theo chuỗi:

1. Encoder trích xuất đặc trưng ảnh.
2. prob_convs sinh logits xác suất cho từng symbol.
3. Gumbel-Softmax (hard=True) rời rạc hóa symbol.
4. modulation() ánh xạ symbol sang mức chòm sao (BPSK/4QAM/16QAM/64QAM).
5. normalize() chuẩn hóa công suất.
6. awgn() thêm nhiễu theo SNR.
7. Decoder_Class và Decoder_Recon tạo hai đầu ra nhiệm vụ.

Ý nghĩa quan trọng nhất: phần coding và modulation được học chung, không tách rời.

### Phút 8-10: Hiểu hàm loss để điều khiển hành vi mô hình

Trong train.py, loss tổng là:

L = L_cls + lambda * L_rec

Trong đó:
1. L_cls: CrossEntropy cho phân loại.
2. L_rec: MSE cho tái tạo ảnh.
3. lambda = tradeoff_lambda.

Diễn giải nhanh:
1. Tăng lambda: ảnh tái tạo đẹp hơn, có thể ảnh hưởng phân loại.
2. Giảm lambda: ưu tiên phân loại hơn.

### Phút 10-12: Biết đọc kết quả đúng cách

EVAL trả về 4 metric:
1. acc: độ chính xác phân loại.
2. mse: sai số tái tạo.
3. psnr: chất lượng tái tạo theo dB.
4. ssim: độ tương đồng cấu trúc.

Khi đối chiếu bảng README theo SNR:
1. Ưu tiên map theo PSNR trước vì thường phân tách theo SNR rõ hơn Accuracy.
2. Accuracy có thể gần nhau giữa nhiều mức SNR, nên dễ gây nhầm nếu dùng một mình.

### Phút 12-14: Chạy 3 lệnh tối thiểu để tự kiểm chứng

1. Pretrain analog:
   python main.py --mode train --pretrain_analog 1 --mod_method 64qam --snr_train 12
2. Train JCM:
   python main.py --mode train --pretrain_analog 0 --mod_method 64qam --snr_train 12
3. Test JCM:
   python main.py --mode test --pretrain_analog 0 --load_checkpoint 1 --mod_method 64qam --snr_train 12 --snr_test 12

Nếu muốn vẽ đường cong theo SNR, lặp lệnh test với snr_test trong tập {18, 12, 6, 0, -6, -12, -18}.

### Phút 14-15: Checklist tự tin trước khi đi sâu

Nếu bạn trả lời được 5 câu sau thì đã nắm phần cốt lõi:
1. Vì sao phải có pretrain_analog?
2. Gumbel-Softmax đang giải quyết bước nào trong pipeline?
3. Vì sao M-QAM cần chiều latent khác BPSK?
4. tradeoff_lambda thay đổi cân bằng nhiệm vụ như thế nào?
5. Vì sao nên map kết quả theo PSNR trước khi kết luận SNR?

Nếu còn thiếu 1-2 câu, hãy đọc lại 3 file theo thứ tự: main.py -> network.py -> train.py.
