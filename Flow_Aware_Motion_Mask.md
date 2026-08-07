# Flow-Aware Auto-decoder cho Motion Mask trong 4DGS

Thay cơ chế "trigger theo biên độ" (kiểu EX4DGS, và `L_bind` ở bản Motion-Mask
trước) bằng một mask **feature-driven**: $m_i$ được giải mã từ một trường đặc
trưng chuyển động riêng (Flow-HexPlane), và trường đó chỉ có ý nghĩa nhờ được
giám sát bởi optical flow thật của video.

---

## 1. Kiến trúc

```
                    ┌─ HexPlane (appearance) ──► MLP decoder ─► Δμ, Δs, Δq, Δo, Δsh
 (x, y, z, t) ──────┤
                    └─ Flow-HexPlane ──┐
                                       ├─► [f_flow ‖ γ(t)] ─► MLP_flow ─┬─► m = σ(·)
                       γ(t) Fourier ───┘                                └─► v ∈ ℝ³
```

* **Flow-HexPlane** — bộ 6 mặt phẳng 2D thứ hai (XY, XZ, YZ, XT, YT, ZT),

  `scene/flow_field.py`. Cấu hình riêng (`flow_kplanes_config`, `flow_multires`)
  nên để capacity thấp hơn appearance grid: trường vận tốc trơn hơn radiance
  nhiều, và đây là chỗ tốn thêm VRAM.
* **Fourier trên t** — nội suy bilinear của grid là một bộ lọc thông thấp, làm

  mất các chuyển động nhanh. $\gamma(t) = [\sin(2^k\pi t), \cos(2^k\pi t)]_{k<L}$
  được nối thẳng vào đầu vào MLP để bù lại (`flow_time_pe`).
* **Motion Mask Head** — $m_i = \sigma(\text{MLP}_\text{flow}(f_\text{flow}))$.

  Gate nhân vào các delta: $\mu_i(t) = \mu_i + m_i \Delta\mu_i$ (tương tự cho
  $s, q, o, sh$).
* **Velocity Head** (tùy chọn) — đọc ra $v_i \in \mathbb{R}^3$ tường minh, biến

  Flow-HexPlane thành một auto-decoder vận tốc thật sự thay vì chỉ là trunk cho
  mask. Dùng để visualize/phân tích, và regularize latent về hướng "ngữ nghĩa
  chuyển động".

---

## 2. Flow-Consistency Loss

### 2.1. Điểm cần sửa so với đề xuất ban đầu

Đề xuất gốc chiếu $m_i \cdot \delta x_i$ lên màn hình để lấy `Flow_render`.
**Cái đó không so sánh được với optical flow.** $\delta x_i$ là dịch chuyển so
với *canonical space* — một mốc tùy ý, phi thời gian; còn optical flow đo dịch
chuyển giữa hai frame kề nhau. Hai đại lượng khác đơn vị vật lý, tối thiểu hóa
sai lệch giữa chúng không hội tụ về nghiệm đúng.

Bản triển khai dùng **sai phân hữu hạn theo thời gian**:

$$p_i(t) = \mu_i + m_i(t)\,\delta x_i(t), \qquad
  \text{Flow}^{\text{render}}_i = \pi\big(p_i(t{+}\Delta t)\big) - \pi\big(p_i(t)\big)$$

Gradient vẫn chảy vào **cả** $m_i$ lẫn $\delta x_i$ ở cả hai mốc thời gian —
đúng cơ chế "hạt đi sai hướng thì $m_i$ hoặc $\delta x_i$ phải chỉnh".

### 2.2. Rasterize flow

Flow 2D per-Gaussian được đưa vào rasterizer qua `colors_precomp = [f_x, f_y, 1]`
với background = 0, tức alpha-blend chính flow đó:

$$F(u) = \sum_i \alpha_i T_i \, \text{Flow}^{\text{render}}_i, \qquad
  A(u) = \sum_i \alpha_i T_i$$

Kênh thứ 3 trả về $A(u)$ (alpha tích lũy), dùng để (a) chuẩn hóa
$F/A$ và (b) loại các pixel không có gì được splat vào.

> Đã kiểm chứng: rasterizer **không** clamp `colors_precomp` (chỉ clamp màu suy
>

### 2.3. Loss

$$\mathcal{L}_\text{flow} = \frac{1}{|\Omega|}\sum_{u \in \Omega}
   \big\| F(u)/A(u) - F^{\text{prior}}(u) \big\|_1,
   \quad \Omega = \{u : \text{valid}(u) \wedge A(u) > \tau\}$$

Mọi thứ tính trong **đơn vị chuẩn hóa** (1.0 = trọn chiều rộng/cao ảnh), nên loss
độc lập với độ phân giải render lẫn độ phân giải của prior.

Kèm theo:

$$\mathcal{L}_\text{vel} = \big\| v_i - \mathrm{sg}[\,(p_i(t{+}\Delta t) - p_i(t))/\Delta t\,] \big\|_1$$

Bảng loss cuối: $\mathcal{L} = \mathcal{L}_1 + \mathcal{L}_\text{TV} +
\lambda_f \mathcal{L}_\text{flow} + \lambda_v \mathcal{L}_\text{vel} +
\lambda_s \mathcal{L}_\text{sparse} + \lambda_m \mathcal{L}_\text{smooth}$.
$\mathcal{L}_\text{bind}$ đặt về 0 (giữ code lại để ablation).

---

## 3. Lịch huấn luyện

| Giai đoạn  | Iteration (fine)           | Trạng thái                                                         |
| ---------- | -------------------------- | ------------------------------------------------------------------ |
| Coarse     | —                          | 3DGS tĩnh, học canonical                                           |
| Warm-up    | 1 – 3,000                  | Deform bật, $m_i \equiv 1$, chưa có `L_flow`                       |
| Fine-tune  | 3,001 – 15,000             | Mask head + `L_flow` + `L_sparse`/`L_smooth`                       |
| (tùy chọn) | từ `freeze_grid_from_iter` | Đóng băng appearance HexPlane, chỉ Flow-HexPlane + MLP decoder học |

Head mask khởi tạo bias = 2.0 ($m \approx 0.88$) để lúc bật mask không sốc. Trọng
số khởi tạo là nhiễu nhỏ chứ **không** phải 0 — với weight đúng bằng 0 thì
$\partial m/\partial h = 0$ và cả trunk lẫn Flow-HexPlane không nhận được gradient
nào ở những bước đầu.

---

## 4. Chuẩn bị optical-flow prior

```bash
python scripts/precompute_flow.py \
    --datadir data/multipleview/backpack_frame0_v2 \
    --gap 15 --max_side 512 --batch 6 --workers 8
```

Ghi ra `<datadir>/flow/<camXX>/flow_<frame:05d>.npz` (`flow` [2,h,w] float16 tính
bằng pixel ở **độ phân giải gốc**, `valid` [h,w] bool từ kiểm tra forward-backward)
cùng `meta.json`. Training tự phát hiện thư mục này; không có nó thì mọi thứ chạy
y như cũ.

### Chọn `--gap` — quan trọng

Đo trên `backpack_frame0_v2/cam01` (1408 frame, 2048×1536), biên độ flow tính
bằng pixel gốc:

| gap | mean        | p99       |
| --- | ----------- | --------- |
| 1   | 0.04 – 0.15 | 0.1 – 3   |
| 5   | 0.3 – 0.6   | 5 – 22    |
| 15  | 0.8 – 1.9   | 34 – 74   |
| 30  | 0.4 – 5.0   | 0.7 – 179 |

**gap = 1 gần như không mang tín hiệu** (dịch chuyển dưới mức nhiễu của RAFT).
`gap = 15` là điểm cân bằng: vùng động đạt vài chục pixel trong khi nền tĩnh vẫn
≈ 0, tức là loss có sức phân biệt động/tĩnh.

Còn một lý do thứ hai: với `resolution[3] = 150` trên 1408 frame, một ô lưới thời
gian trải 9.4 frame. Sai phân `gap = 1` nằm gọn *bên trong* một ô nên chỉ đọc được
nhiễu nội suy; `gap = 15` trải ~1.6 ô nên thực sự đo được độ dốc thời gian của
trường biến dạng.

Dung lượng: ~1 MB/cặp ở 512×384, ~5.5 GB cho 4 camera × 1393 cặp.

### Chất lượng prior trên scene này

Kiểm tra trực quan trên `cam01` (gap = 15): flow tách sạch người đi bộ + balo,
nền đúng bằng 0. Tỉ lệ pixel có `|flow| > 5px`:

| frame | p99 (px) | tỉ lệ pixel động | valid (fwd-bwd) |
| ----- | -------- | ---------------- | --------------- |
| 200   | 142.7    | 4.8 %            | 95.6 %          |
| 700   | 44.3     | 3.2 %            | 95.4 %          |
| 1100  | 28.7     | 2.6 %            | 95.8 %          |

Chỉ ~3–5 % pixel thực sự động. Đó là lý do trung bình của $\mathcal{L}_\text{flow}$
trên toàn ảnh luôn nhỏ, và là lý do phải theo dõi `epe_px` thay vì giá trị loss.

---

## 5. Kiểm chứng

`scripts/precompute_flow.py` và các thành phần loss đã được test:

1. `forward_position` (lượt $t{+}\Delta t$ rút gọn) khớp bit-wise với

   `forward_dynamic`, và gradient đến được Flow-HexPlane.
2. Param routing: mặt phẳng flow vào nhóm LR `grid`, MLP flow vào nhóm `deformation`.
3. Quy ước chiếu của `project_to_ndc` khớp CUDA rasterizer trong 0.3 pixel.
4. Flow render tại tâm blob trùng khớp giá trị giải tích, kể cả thành phần âm →

   xác nhận `colors_precomp` không bị clamp.
5. `flow_consistency_loss`: prior đúng → EPE = 0; prior đảo dấu → EPE = 2×|flow|.
6. Chạy train thật 120 iter trên `backpack_frame0_v2`: prior load đúng, `L_flow`
   vào loss, `epe_px ≈ 1.2`, `coverage ≈ 0.95`, freeze grid kích hoạt đúng iter.
7. Hồi quy: config `backpack_mask.py` cũ (`motion_mask` không flow) train y như
   trước, `bind_loss` vẫn hoạt động, không phát sinh chi phí flow pass.

---

## 6. Chạy

```bash
# 1. prior
python scripts/precompute_flow.py --datadir data/multipleview/backpack_frame0_v2 --gap 15

# 2. train
python train.py -s data/multipleview/backpack_frame0_v2 \
    --port 6017 --expname "multipleview/backpack_flow" \
    --configs arguments/multipleview/backpack_flow.py
```

Theo dõi trên tensorboard:

* `fine/flow/epe_px` — sai số flow tính bằng pixel gốc. Đây là con số để chỉnh `lambda_flow`; nó phải giảm sau khi flow bật ở iter 3000.
* `fine/flow/coverage` — tỉ lệ pixel hợp lệ. Quá thấp thì hạ `flow_alpha_threshold`.
* `fine/motion/dynamic_ratio` — tỉ lệ Gaussian có $m_i > \epsilon$. Kỳ vọng tụt từ 1.0 xuống khoảng 0.15 – 0.40.
* `fine/flow/rendered` vs `fine/flow/prior` — hai bản đồ flow tô màu cùng thang màu, so trực tiếp bằng mắt.

Về `lambda_flow`: cảnh chủ yếu là tĩnh nên trung bình $\mathcal{L}_\text{flow}$
trên toàn ảnh rất nhỏ — **đừng chỉnh theo giá trị loss, chỉnh theo `epe_px`**.
Mặc định 10.0 ở đơn vị chuẩn hóa tương đương ~0.005 loss cho mỗi pixel sai số ở
chiều rộng 2048.

Ablation bật/tắt bằng đúng một cờ trong config: `flow_mask=False` +
`motion_mask=True` + `lambda_motion_bind=0.1` là quay lại bản trigger-based cũ.

---

## 7. Chi phí

Mỗi iteration có flow tốn thêm: 1 lượt query deformation **chỉ vị trí** ở
$t{+}\Delta t$ (bỏ các head scale/rot/opacity/SH) + 1 lượt rasterize. Đo thực tế
xem `fine/iter_time`. Nếu chậm quá, tăng `flow_interval` lên 2–4 — mask vẫn học
được vì `L_sparse`/`L_smooth` chạy mọi iteration.

VRAM thêm vào chủ yếu là Flow-HexPlane; với `output_coordinate_dim=8` và
`multires=[1,2,4]` nó chỉ bằng khoảng 1/10 appearance grid
(`output_coordinate_dim=16`, `multires=[1,2,4,8,16,32]`).
