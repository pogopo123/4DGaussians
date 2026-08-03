# F3GS: Fourier-Enhanced Sparse 4D Gaussian Splatting for Fast and High-Frequency Dynamic Scene Rendering

> **Abstract:** 
>
> 1. **Fourier-Enhanced HexPlane:** Mã hóa vị trí đa tần số (Fourier Positional Encoding) trên trục thời gian $t$ trước khi query, giúp bắt trọn các chi tiết chuyển động nhanh và phức tạp.

---

## 1. Problem Formulation & Motivation

Trong 4DGS gốc (Wu et al., arXiv:2310.10642), trạng thái động của Gaussian thứ $i$ tại thời điểm $t$ được dự đoán thông qua trường biến dạng $\mathcal{D}$ và HexPlane 4D $\Phi$:

$$\Delta \mu_i, \Delta s_i, \Delta q_i = \mathcal{D}(\Phi(\mu_i, t))$$

$$\mu_i(t) = \mu_i + \Delta \mu_i, \quad s_i(t) = s_i + \Delta s_i, \quad q_i(t) = \frac{q_i + \Delta q_i}{\Vert{}q_i + \Delta q_i\Vert{}}$$

### Hạn chế cốt lõi:

1. **Computational Waste:** Phép toán Deform được áp dụng bắt buộc lên toàn bộ $N$ Gaussians ($\forall i \in \{1, \dots, N\}$). Trong thực tế, các vùng tĩnh (sàn, tường, background) chiếm từ $60\% - 85\%$ số lượng điểm. Việc query HexPlane và forward MLP cho các điểm này gây lãng phí bộ nhớ và VRAM đệm.
2. **Static Artifacts:** Nhiễu đầu ra nhỏ từ MLP làm cho các điểm nền tĩnh bị "rung" hoặc "nhòe" nhẹ theo thời gian ($t$).
3. **Temporal Low-Pass Bottleneck:** Phép $F.grid\_sample$ trên 6 mặt phẳng 2D giới hạn khả năng biểu diễn của trục thời gian, làm mượt (smooth out) các sự kiện diễn ra nhanh giữa các frame.

---

## 2. Proposed Method

### 2.1. Fourier Feature HexPlane (High-Frequency Temporal Representation)

Nhằm giúp mạng nhận thức được các tần số chuyển động khác nhau (Frequency-Aware), tham số thời gian $t \in [-1, 1]$ được đưa qua hàm mã hóa Fourier Positional Encoding trước khi tiến hành query vào các mặt phẳng chứa trục thời gian ($XT, YT, ZT$):

$$e(t) = \left[ \sin(2^0 \pi t), \cos(2^0 \pi t), \dots, \sin(2^{L-1} \pi t), \cos(2^{L-1} \pi t) \right]^T \in \mathbb{R}^{2L}$$

Feature thu được từ bước mã hóa Fourier kết hợp cùng phép truy vấn không gian ($XY, XZ, YZ$) tạo ra vector đặc trưng phong phú $h_i \in \mathbb{R}^{128}$, giúp mô hình không bị phụ thuộc vào độ phân giải cố định của grid thời gian.

---

### 2.2. Motion-Mask Aware Deformation Network

Chúng tôi mở rộng MLP Decoder $\mathcal{D}^*$ bằng cách bổ sung một **Motion Mask Head** song song với các Deform Head ($\Delta \mu, \Delta s, \Delta q$):

$$m_i = \sigma(W_m \cdot h_i + b_m) \in [0, 1]$$

Trong đó $\sigma(\cdot)$ là hàm Sigmoid, $m_i$ đại diện cho xác suất/mức độ chuyển động của Gaussian thứ $i$. Hệ phương trình biến dạng theo $t$ được định nghĩa lại thành:

$$\mu_i(t) = \mu_i + m_i \cdot \Delta \mu_i$$

$$s_i(t) = s_i + m_i \cdot \Delta s_i$$

$$q_i(t) = \frac{q_i + m_i \cdot \Delta q_i}{\Vert{}q_i + m_i \cdot \Delta q_i\Vert{}}$$

---

### 2.3. System Architecture & Conditional Workflow