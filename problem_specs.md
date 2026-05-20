# Problem Specification: UAV-Assisted Ambient Backscatter Anti-Jamming Network with Multi-Agent DRL

> **Context:** Extension of *"Turning Interference into Power: Intelligent Anti-Jamming for Ambient Backscatter via Imitation-Augmented MADDPG"*  
> **Hướng mở rộng:** Thay thế/bổ sung relay tĩnh (RBS) bằng UAV relay động, tích hợp Ambient Backscatter Communication (AmB)

---

## 1. Tổng Quan Bài Toán

Bài toán nghiên cứu mạng truyền thông D2D (Device-to-Device) trong môi trường bị tấn công gây nhiễu (jamming), trong đó:

- **Nguồn phát (SU — Source User)** và **đích nhận (DU — Destination User)** là các thiết bị IoT/người dùng mặt đất, bị giới hạn năng lượng.
- Thay vì né tránh tín hiệu jamming, hệ thống **khai thác tín hiệu jamming công suất cao như nguồn RF kích hoạt** cho truyền thông Ambient Backscatter.
- Việc điều phối đa tác tử (UAV + SU) được giải quyết bằng **IA-MADDPG mở rộng** (Imitation-Augmented Multi-Agent Deep Deterministic Policy Gradient).

---

## 2. Mô Hình Hệ Thống

### 2.1 Các Thành Phần Mạng

| Thành phần | Ký hiệu | Mô tả |
|---|---|---|
| Source User | $S_i$, $i \in \mathcal{N}$ | Thiết bị nguồn mặt đất, truyền bằng AmB |
| Destination User | $D_i$ | Thiết bị đích mặt đất, nhận tín hiệu |
| Relay Base Station | RBS | Trạm relay **cố định** (giữ từ bài gốc) |
| UAV Relay | $U_k$, $k \in \mathcal{K}$ | Relay **di động**, bay ở độ cao $H_k$, thay đổi vị trí theo thời gian |
| Jammer | $J$ | Thiết bị gây nhiễu mặt đất, phát công suất $P_J$ liên tục |

### 2.2 Tại Sao Cần UAV Relay?

Truyền thông D2D gặp hai vấn đề chính:

1. **Khoảng cách quá xa:** Tín hiệu backscatter từ $S_i$ không đủ mạnh để đến thẳng $D_i$ (đường D2D trực tiếp suy hao cao).
2. **Bóng che (shadowing):** Địa hình, vật cản chặn đường truyền mặt đất.

**RBS cố định** chỉ giải quyết được một phần — nếu RBS nằm xa $S_i$ hoặc $D_i$, hiệu quả thấp. **UAV relay động** có thể:
- Bay đến vị trí tối ưu để tăng SINR trên cả hai chặng.
- Thích nghi với sự thay đổi vị trí jammer hoặc thiết bị người dùng.
- Cung cấp đường truyền LoS (Line-of-Sight) xác suất cao hơn.

---

## 3. Các Chế Độ Truyền Thông (Transmission Modes)

Mỗi cặp SU-DU có thể chọn một trong **ba chế độ**:

### Mode 0 — Direct D2D (Truyền thẳng)

```
S_i  ---[AmB, sóng jamming làm RF source]---> D_i
```

- $S_i$ backscatter trực tiếp tín hiệu jammer đến $D_i$.
- Phù hợp khi $S_i$ và $D_i$ gần nhau, tín hiệu $J \to S_i \to D_i$ đủ mạnh.
- **Hạn chế:** Fading mạnh khi khoảng cách lớn; jammer cũng gây nhiễu trực tiếp tại $D_i$.

### Mode 1 — RBS Relay (Relay qua trạm cố định)

```
S_i  --[AmB]--> RBS --[forward]--> D_i
```

- $S_i$ backscatter đến RBS; RBS thu năng lượng từ jammer, forward đến $D_i$.
- SINR đầu cuối bị giới hạn bởi nút thắt cổ chai (bottleneck) giữa hai chặng.
- **Hạn chế:** RBS vị trí cố định, không thích ứng được với thay đổi topo mạng.

### Mode 2 — UAV Relay (Relay qua UAV di động) *(mới)*

```
S_i  --[AmB]--> U_k --[forward]--> D_i
```

- $S_i$ backscatter đến UAV $U_k$ đang bay gần; UAV forward đến $D_i$.
- UAV thu năng lượng từ tín hiệu jammer (hoặc mang pin) để forward.
- **Ưu điểm:** UAV điều chỉnh vị trí 3D để tối ưu cả hai chặng đồng thời.

> **Hành động của mỗi tác tử SU:** chọn $m_i \in \{0, 1, 2\}$ và điều chỉnh hệ số phản xạ $\alpha_i \in [0, 1]$.  
> **Hành động của mỗi UAV:** chọn hướng di chuyển $\Delta x_k, \Delta y_k, \Delta z_k$ trong không gian 3D (hoặc tốc độ bay).

---

## 4. Mô Hình Kênh Truyền

### 4.1 Kênh Mặt Đất (Ground-to-Ground)

Mô hình path-loss tiêu chuẩn với fading Rayleigh:

$$g_{X,Y} = |h_{X,Y}|^2, \quad h_{X,Y} \sim \mathcal{CN}(0, d_{X,Y}^{-\eta})$$

với $\eta$ là hệ số suy hao đường truyền.

### 4.2 Kênh Không-Đất (Air-to-Ground) — UAV

Mô hình kênh LoS xác suất (Probabilistic LoS):

$$P_{\text{LoS}}(d, H) = \frac{1}{1 + a \cdot \exp(-b(\frac{180}{\pi}\arctan(\frac{H}{d}) - a))}$$

Tổn hao đường truyền trung bình:

$$\bar{L}_{U_k, X} = P_{\text{LoS}} \cdot L_{\text{LoS}} + (1 - P_{\text{LoS}}) \cdot L_{\text{NLoS}}$$

với $L_{\text{LoS}}$ và $L_{\text{NLoS}}$ là tổn hao trong điều kiện LoS và NLoS tương ứng.

### 4.3 SINR Theo Từng Mode

**Mode 0 (Direct D2D):**
$$\gamma_i^{\text{D2D}} = \frac{G P_J g_{J,S_i} g_{S_i,D_i} \alpha_i^2}{P_J g_{J,D_i} + N_0}$$

**Mode 1 (RBS Relay):**
$$\gamma_i^{\text{RBS}} = \min\!\left(\frac{G P_J g_{J,S_i} g_{S_i,R} \alpha_i^2}{P_J g_{J,R} + N_0},\; \frac{G P_J g_{J,R} g_{R,D_i} \alpha_i^2}{P_J g_{J,D_i} + N_0}\right)$$

**Mode 2 (UAV Relay):** *(mới)*
$$\gamma_i^{\text{UAV}} = \min\!\left(\frac{G P_J g_{J,S_i} \bar{g}_{S_i,U_k} \alpha_i^2}{P_J \bar{g}_{J,U_k} + N_0},\; \frac{G P_J \bar{g}_{J,U_k} \bar{g}_{U_k,D_i} \alpha_i^2}{P_J g_{J,D_i} + N_0}\right)$$

trong đó $\bar{g}_{X,U_k}$ là channel gain trung bình theo mô hình LoS xác suất.

---

## 5. Định Nghĩa Bài Toán Tối Ưu

### 5.1 Hàm Mục Tiêu

Tối đa hóa tổng phần thưởng mạng (network reward) trong T bước thời gian:

$$\max_{\{\alpha_i, m_i, \Delta \mathbf{p}_{U_k}\}} \sum_{t=1}^{T} \sum_{i=1}^{N} r_i(t)$$

với hàm phần thưởng composite:

$$r_i(t) = w_1 R_i(t) + w_2 \tanh\!\left(\frac{\gamma_i(t)}{\Gamma_{\text{th}}} - 1\right) - w_3 \alpha_i(t)^2 - w_4 \cdot \mathbf{1}[\text{mode}=2] \cdot E_{U_k}^{\text{move}}$$

| Số hạng | Ý nghĩa |
|---|---|
| $w_1 R_i(t) = w_1 \log_2(1+\gamma_i(t))$ | Thông lượng tức thời (throughput) |
| $w_2 \tanh(\cdot)$ | Ràng buộc QoS mềm — phạt khi SINR dưới ngưỡng $\Gamma_{\text{th}}$ |
| $-w_3 \alpha_i^2$ | Phạt năng lượng — khuyến khích chọn $\alpha_i$ nhỏ nhất đủ dùng |
| $-w_4 E_{U_k}^{\text{move}}$ | Phạt năng lượng di chuyển UAV *(mới)* |

### 5.2 Ràng Buộc

- $\alpha_i \in [0, 1]$: Hệ số phản xạ hợp lệ.
- $m_i \in \{0, 1, 2\}$: Chế độ truyền rời rạc.
- $\mathbf{p}_{U_k}(t+1) = \mathbf{p}_{U_k}(t) + \Delta\mathbf{p}_{U_k}(t)$: Cập nhật vị trí UAV.
- $H_{\min} \leq H_{U_k} \leq H_{\max}$: Giới hạn độ cao bay.
- $\|\Delta\mathbf{p}_{U_k}\| \leq v_{\max} \cdot \delta_t$: Tốc độ bay tối đa.
- $E_{U_k}^{\text{total}} \leq E_{\max}$: Giới hạn năng lượng tổng của UAV.

---

## 6. Phát Biểu POMDP Đa Tác Tử

Bài toán được mô hình hóa là **Decentralized POMDP** với tập tác tử gồm:
- $N$ SU agents: điều khiển $(\alpha_i, m_i)$
- $K$ UAV agents: điều khiển $(\Delta x_k, \Delta y_k, \Delta z_k)$

### 6.1 Không Gian Quan Sát

**SU agent $i$:**
$$o_i = \left\{\gamma_i^{(t-1)},\; E_i^{(t-1)},\; m_i^{(t-1)},\; \hat{P}_J^{(t-1)}\right\}$$

**UAV agent $k$:**
$$o_{U_k} = \left\{\mathbf{p}_{U_k}^{(t)},\; \{\bar{g}_{S_i, U_k}\}_{i},\; \{\bar{g}_{U_k, D_i}\}_{i},\; \hat{P}_J^{(t-1)},\; E_{U_k}^{\text{remain}}\right\}$$

### 6.2 Không Gian Hành Động

| Tác tử | Hành động | Loại |
|---|---|---|
| SU $i$ | $a_i = (\alpha_i, m_i)$ | Hỗn hợp (continuous + discrete) |
| UAV $k$ | $a_{U_k} = (\Delta x_k, \Delta y_k, \Delta z_k)$ | Continuous |

### 6.3 Hàm Phần Thưởng Toàn Cục

Phần thưởng toàn cục (dùng cho critic tập trung):

$$r^{\text{global}}(t) = \sum_{i=1}^{N} r_i(t) + \sum_{k=1}^{K} r_{U_k}^{\text{pos}}(t)$$

trong đó $r_{U_k}^{\text{pos}}(t)$ thưởng UAV vì đã di chuyển đến vị trí cải thiện SINR của các SU mà nó phục vụ.

---

## 7. Framework IA-MADDPG Mở Rộng

### 7.1 Kiến Trúc Đa Tác Tử

```
┌─────────────────────────────────────────────────────────┐
│                  CENTRALIZED CRITIC                      │
│  Input: global state (o1,...,oN, oU1,...,oUK)           │
│         all actions (a1,...,aN, aU1,...,aUK)            │
│  Output: Q-value                                         │
└─────────────────────────────────────────────────────────┘
         ▲                         ▲
         │ training only           │ training only
┌────────┴──────────┐    ┌─────────┴──────────┐
│  SU Actor μ_i     │    │  UAV Actor μ_{Uk}  │
│  Input:  o_i      │    │  Input:  o_{Uk}    │
│  Output: α_i, m_i │    │  Output: Δx,Δy,Δz  │
└───────────────────┘    └────────────────────┘
```

### 7.2 Expert Policy Mở Rộng

**Expert cho SU:** Greedy chọn mode và $\alpha_i$ tối đa SINR tức thời (như bài gốc).

**Expert cho UAV:** Thuật toán heuristic tối ưu vị trí UAV dựa trên thông tin kênh truyền đầy đủ:

$$\mathbf{p}_{U_k}^* = \arg\max_{\mathbf{p}} \sum_{i \in \mathcal{C}_k} \gamma_i^{\text{UAV}}(\mathbf{p})$$

với $\mathcal{C}_k$ là tập SU được phục vụ bởi UAV $k$.

### 7.3 Hàm Loss Actor (Behavior Cloning + RL)

$$\mathcal{L}(\theta^{\mu_i}) = \underbrace{-\mathbb{E}[Q_i(o, a)]}_{\mathcal{L}_{\text{MADDPG}}} + \lambda_{\text{IL}} \underbrace{\mathbb{E}\left[\|\mu_i(o_i) - \mu^E(o_i)\|^2\right]}_{\mathcal{L}_{\text{BC}}}$$

$\lambda_{\text{IL}}$ giảm dần theo thời gian (annealing) để tác tử vượt qua giới hạn của expert.

---

## 8. Baseline So Sánh

| Baseline | Mô tả |
|---|---|
| Direct Transmission (DT) | Truyền thẳng D2D, $\alpha = 0.2$ cố định, không relay |
| Greedy Strategy | Chọn mode (D2D/RBS) tối đa SINR tức thời, không UAV |
| Frequency Hopping (FH) | Né tránh ngẫu nhiên, không khai thác jamming |
| MADDPG (không IL) | MADDPG tiêu chuẩn, có UAV nhưng không imitation learning |
| IA-MADDPG gốc (RBS only) | Từ bài báo, không có UAV relay |
| **IA-MADDPG + UAV** *(đề xuất)* | **Framework đầy đủ với UAV relay động** |

---

## 9. Thông Số Mô Phỏng (Dự Kiến)

| Tham số | Giá trị |
|---|---|
| Diện tích mô phỏng | $200 \times 200$ m² |
| Số cặp SU-DU ($N$) | 5 |
| Số UAV ($K$) | 2–3 |
| Công suất jammer ($P_J$) | 1 W |
| Noise floor ($N_0$) | $10^{-4}$ W |
| Độ cao UAV ($H$) | 20–100 m |
| Tốc độ bay tối đa ($v_{\max}$) | 10 m/s |
| Hệ số suy hao mặt đất ($\eta$) | 2.7 |
| Backscatter gain ($G$) | $10^4$ |
| Ngưỡng SINR ($\Gamma_{\text{th}}$) | 5 dB |
| Số episode huấn luyện | 600 |
| Batch size | 256 |
| Kiến trúc Actor (SU) | [256, 256, 128] |
| Kiến trúc Actor (UAV) | [256, 256, 128] |
| Kiến trúc Critic | [512, 256, 128, 64] |

---

## 10. Chỉ Số Đánh Giá

| Chỉ số | Ký hiệu | Mô tả |
|---|---|---|
| Tổng phần thưởng mạng | $\bar{r}$ | Trung bình phần thưởng mỗi bước |
| Tỷ lệ truyền thành công | TSR | $\Pr[\gamma_i \geq \Gamma_{\text{th}}]$ |
| Thông lượng trung bình | $\bar{R}$ | bits/s/Hz |
| Quỹ đạo UAV | — | Visualize vị trí UAV theo thời gian |
| Phân phối mode chọn | — | Tỷ lệ D2D / RBS / UAV được chọn |
| Tốc độ hội tụ | — | Episode đầu tiên vượt qua baseline tốt nhất |

---

## 11. Điểm Khác Biệt So Với Bài Báo Gốc

| Khía cạnh | Bài báo gốc | Phiên bản mở rộng |
|---|---|---|
| Relay | RBS cố định | RBS cố định + $K$ UAV di động |
| Chế độ truyền | 2 modes (D2D, RBS) | 3 modes (D2D, RBS, UAV) |
| Không gian hành động | $(\alpha_i, m_i)$ cho SU | $(\alpha_i, m_i)$ cho SU + $(\Delta x,\Delta y,\Delta z)$ cho UAV |
| Số tác tử | $N$ SU | $N$ SU + $K$ UAV |
| Mô hình kênh | Path-loss mặt đất | Path-loss mặt đất + LoS xác suất A2G |
| Expert policy | Greedy SINR (SU) | Greedy SINR (SU) + Heuristic vị trí (UAV) |
| Diện tích mô phỏng | $120 \times 120$ m² | $200 \times 200$ m² |

---

## 12. Hướng Nghiên Cứu Tiếp Theo

- **Imperfect CSI:** Đánh giá ảnh hưởng của thông tin kênh truyền không hoàn hảo.
- **Mobile jammer:** Jammer di chuyển thay vì đứng yên.
- **Multi-jammer scenario:** Nhiều jammer tấn công đồng thời.
- **Heterogeneous network:** Kết hợp backscatter và truyền thông chủ động.
- **Large-scale deployment:** Mở rộng lên $N > 20$ cặp SU-DU, $K > 5$ UAV.
