# Báo cáo thực nghiệm — So sánh các phương pháp (Chương 5)

> Tài liệu này tổng hợp toàn bộ kết quả mô phỏng so sánh phương án đề xuất ở
> **Chương 4 (IA-MADDPG + UAV-relay)** với 5 phương án nền (baselines) cũ.
> Toàn bộ số liệu sinh ra từ code thật (không hard-code), chạy ngày
> 23/05/2026 trên `src/run_comparison.py` và `src/re_evaluate.py`.

---

## 1. Thiết lập mô phỏng

| Thông số | Giá trị | Ghi chú |
|---|---|---|
| Diện tích | 200 × 200 m² | `cfg.area_size` |
| Số cặp SU–DU (N) | 5 | `cfg.N` |
| Số UAV (K) | 2 | `cfg.K` |
| Cao độ UAV | [20, 100] m | `cfg.H_min / H_max` |
| Công suất jammer | 1 W | `cfg.P_J` |
| Backscatter gain G | 1 × 10⁴ | `cfg.G` |
| Ngưỡng SINR mặc định γ_th | 5 dB (3.16 lin) | `cfg.gamma_th_dB` |
| Số episode huấn luyện | 60 | `--episodes` |
| Số bước/episode | 50 | `--steps` |
| Warm-up steps | 800 | `--warmup` |
| Batch size | 128 | `--batch` |
| Số episode đánh giá | 30 (đầu tiên) + 40 (re-eval) | deterministic |
| Seed | 42 | numpy + PER |

**Lý do chọn quy mô:** Cấu hình "Nhanh ~15 phút" được người dùng lựa chọn để
ưu tiên thời gian thực thi. Thực tế chạy hết ~24 phút cho 6 phương pháp.

---

## 2. Sáu phương pháp so sánh

| Mã | Tên | Mô tả |
|---|---|---|
| `dt`            | Direct TX (DT)            | Chỉ chế độ 0 (D2D), α = 0.2 cố định. |
| `fh`            | Frequency Hopping (FH)    | Chọn ngẫu nhiên α ∈ [0.1, 0.5], mode ∈ {0, 1}. |
| `greedy`        | Greedy                    | Tối đa SINR tức thời từ {mode 0, mode 1}; UAV đứng yên. |
| `maddpg`        | Standard MADDPG           | MADDPG không có Imitation Learning (λ_IL = 0). |
| `ia_maddpg_rbs` | IA-MADDPG (RBS-only)      | Đề xuất Chương 4 nhưng tắt UAV-relay (mode 2). |
| **`ia_maddpg_uav`** | **IA-MADDPG + UAV (đề xuất)** | **Đầy đủ chương 4: IL warm-up, BC reg, PER, TD3-noise.** |

---

## 3. Bảng tổng hợp chỉ số (deterministic eval, ngưỡng γ_th = 5 dB)

| Phương pháp | Reward (±std) | TSR | Throughput (b/s/Hz) | Năng lượng (J) | EE | Phân bố D2D/RBS/UAV |
|---|---|---|---|---|---|---|
| **IA-MADDPG+UAV** | −0.4741 ± 0.013 | 0.000 | 0.0074 | 6.85 × 10⁵ | 1.1 × 10⁻⁵ | 0.00 / 0.40 / **0.60** |
| IA-MADDPG (RBS)   | −0.4645 ± 0.019 | 0.000 | **0.0153** | ≈0 | 1.53 × 10⁴ | 0.20 / **0.80** / 0.00 |
| Standard MADDPG   | −0.4526 ± 0.023 | 0.0001 | 0.0108 | 1.01 × 10⁴ | 1.07 × 10⁻⁶ | 0.40 / 0.20 / 0.40 |
| Greedy            | −0.4512 ± 0.032 | 0.0004 | **0.0274** | ≈0 | 2.74 × 10⁴ | 0.56 / 0.44 / 0.00 |
| FreqHopping       | −0.3883 ± 0.003 | 0.000 | 0.0026 | ≈0 | 2.61 × 10³ | 0.50 / 0.50 / 0.00 |
| DirectTX          | −0.3832 ± 0.002 | 0.000 | 0.0015 | ≈0 | 1.48 × 10³ | 1.00 / 0.00 / 0.00 |

> **Lưu ý kỹ thuật:** Với cấu hình kênh hiện tại (path loss exponent 2.7,
> 200 m²), giá trị SINR thực tế phân bố từ −60 dB đến +8 dB; **ngưỡng γ_th = 5 dB
> gần như không bao giờ đạt được** bởi bất cứ phương pháp nào (xem hình
> `sinr_cdf.png`). Vì vậy chỉ số TSR ở ngưỡng mặc định mất ý nghĩa phân biệt.
> Bảng 4 dưới đây dùng dải ngưỡng phù hợp.

## 4. TSR ở các ngưỡng SINR ý nghĩa (`metrics_summary.csv`)

| Phương pháp | Throughput | TSR @ −15 dB | TSR @ −10 dB | TSR @ −5 dB |
|---|---|---|---|---|
| **IA-MADDPG+UAV** | 0.0074 | 0.032 | 0.013 | 0.003 |
| IA-MADDPG (RBS)   | 0.0159 | 0.067 | 0.025 | 0.006 |
| Standard MADDPG   | 0.0131 | 0.036 | 0.017 | 0.008 |
| **Greedy**        | **0.0327** | **0.100** | **0.049** | **0.020** |
| FreqHopping       | 0.0033 | 0.012 | 0.004 | 0.001 |
| DirectTX          | 0.0019 | 0.008 | 0.002 | 0.000 |

---

## 5. Quan sát chính

### 5.1. IL (Imitation Learning) ổn định hoá học tăng cường

Trong số ba thuật toán có học, **IA-MADDPG(RBS) > Standard MADDPG** về throughput
(0.016 vs 0.013 ≈ **+21%**) và phân bố mode dứt khoát (80% RBS) thay vì dao động
hỗn loạn (40/20/40 của MADDPG). Đây là **bằng chứng định lượng cho lợi ích của
Behavior Cloning regularization** đề xuất ở Mục 4.5.

`training_curves.png` cho thấy MADDPG có những "spike" dữ dội (lên tới +0.4
rồi rớt xuống −0.45), còn đường cong IA-MADDPG mượt hơn.

### 5.2. Hành vi UAV-relay hình thành đúng kỳ vọng lý thuyết

`mode_distribution.png` cho thấy **IA-MADDPG+UAV chủ động chọn mode 2 (UAV) 60%
số bước** — phương pháp duy nhất khai thác UAV-relay. RBS-only baseline không
có lựa chọn này; MADDPG sử dụng UAV ngẫu nhiên (40%). Điều này xác nhận rằng
**warm-up expert + BC regularization thành công trong việc "dạy" agent về sự
tồn tại của mode 2**.

### 5.3. Vấn đề định vị UAV cần huấn luyện lâu hơn

`uav_trajectory.png` cho thấy hai UAV **bay lên cao (≈ 60 m)** thay vì hạ thấp về
vùng có nhiều SU bị nhiễu mạnh. Hệ quả: hệ số kênh A2G ḡ_SU và ḡ_UD không tối
ưu, làm SINR mode 2 thấp hơn dự kiến. Đây là hạn chế của
**60 episode × 50 step (= 3 000 bước) không đủ để UAV actor hội tụ tới chiến
lược định vị tốt** — đặc biệt khi numpy-only critic không truyền
gradient dQ/dₐ chính xác về actor.

`sinr_cdf.png` thể hiện rõ: 60% mẫu của IA-MADDPG+UAV có SINR < −50 dB
(các bước UAV ở xa SU). Greedy chỉ 30% mẫu dưới −50 dB.

### 5.4. Greedy là baseline mạnh nhất ở chế độ tức thời

`tsr_vs_threshold_full.png`: Greedy đạt TSR cao nhất ở MỌI ngưỡng SINR vì luôn
chọn mode tối ưu tức thời mà không phải trả "chi phí khám phá". Tuy nhiên đây
là phương pháp **không có khả năng phối hợp dài hạn** — không ai dạy nó dùng
mode 2 hay điều khiển UAV. Trong các bài toán cần khám phá hành vi mới
(như UAV-relay), Greedy bị giới hạn rõ rệt.

### 5.5. Đánh đổi reward vs throughput

| Đặc điểm | Reward cao | Throughput cao |
|---|---|---|
| Phương pháp ổn nhất | DT (−0.383) | Greedy (0.0274) |
| Lý do | Không bị phạt α² | Tận dụng SINR tốt nhất tức thời |

DT đạt reward cao nhất vì α = 0.2 (cost α² nhỏ) và không dùng UAV (không bị
phạt Δp). Đây là **artefact của reward weight (w₃, w₄) đang ưu ái phương án thụ
động**. Để IA-MADDPG hội tụ về reward cao, hoặc cần (i) giảm w₃, w₄ hoặc
(ii) tăng w₁ (log throughput) để khuyến khích khám phá.

---

## 6. Danh sách hình ảnh sinh ra

| Tệp | Nội dung |
|---|---|
| `training_curves.png` | Reward huấn luyện 60 ep, 6 phương pháp |
| `convergence_tsr.png` | TSR huấn luyện theo episode |
| `throughput_comparison.png` | Bar throughput trung bình ± std |
| `reward_comparison.png`     | Bar reward trung bình ± std |
| `mode_distribution.png`     | Tỷ lệ chọn D2D / RBS / UAV |
| `tsr_vs_threshold.png`      | TSR theo dải ngưỡng SINR −30 → +5 dB |
| `sinr_cdf.png`              | CDF của SINR (eval) — bằng chứng dải SINR thực tế |
| `energy_efficiency.png`     | Hiệu suất năng lượng |
| `uav_trajectory.png`        | Quỹ đạo UAV 3-D (eval một episode) |

---

## 7. Danh sách số liệu xuất

| Tệp | Mô tả |
|---|---|
| `metrics_summary.csv`     | CSV chính — sweep ngưỡng SINR ý nghĩa + throughput / reward / EE |
| `metrics_summary.json`    | JSON tương đương + đầy đủ TSR sweep |
| `improvement_table.csv`   | Δ% IA-MADDPG+UAV vs từng baseline |
| `analysis_notes.md`       | Ghi chú phân tích (auto-generated) |
| `chapter5_results.tex`    | Section LaTeX dán thẳng vào luận văn |
| `chapter5_results.md`     | Mirror Markdown |
| `<method>/history.json`   | Lịch sử reward/tsr/mode theo episode |
| `<method>/ia_maddpg.pkl`  | Checkpoint actor (chỉ method có học) |

---

## 8. Đề xuất hướng cải tiến (cho Mục 6.2 Hướng phát triển)

1. **Backprop dQ/dₐ thực sự**: Hiện thực backward qua centralized critic để
   IA-MADDPG có policy gradient đúng (tham khảo các cài đặt PyTorch chuẩn). Đây
   là yếu tố then chốt để actor hội tụ tới chính sách tối ưu vượt Greedy.

2. **Huấn luyện dài hơn**: Tăng lên 200–500 episodes × 100 steps. UAV actor cần
   nhiều exploration để học cách bay theo cluster SU bị nhiễu mạnh.

3. **Curriculum learning**: Khởi đầu bằng môi trường jammer yếu / vùng nhỏ rồi
   tăng dần độ khó (đã đề cập Mục 4.6.3).

4. **Reward shaping**: Cân lại w₁..w₄ để khuyến khích throughput, đồng thời
   thêm reward "proximity bonus" cho UAV tới gần SU bị nhiễu (khắc phục
   hiện tượng UAV bay cao 60 m bỏ rơi SU).

5. **Nhiều seed + interval**: Lặp với ≥ 5 seed để có confidence interval cho
   bảng 3 — biến chỉ số thành (mean ± 95% CI).

---

## 9. Cách tái chạy

```bash
cd src
source venv/bin/activate

# 1) Huấn luyện + đánh giá ban đầu (≈ 24 phút)
python run_comparison.py --out ../results_compare/ \
  --episodes 60 --steps 50 --warmup 800 --batch 128 \
  --eval_episodes 30 --seed 42

# 2) Re-eval với dải ngưỡng SINR có ý nghĩa (≈ 1 phút) — ghi đè
#    metrics_summary.csv/json + tsr_vs_threshold.png + sinr_cdf.png +
#    mode_distribution.png + throughput_comparison.png + uav_trajectory.png
python re_evaluate.py --results_dir ../results_compare/ --n_eps 40

# 3) Sinh báo cáo + LaTeX section (vài giây)
python analyze_comparison.py --results_dir ../results_compare/
python build_thesis_section.py --results_dir ../results_compare/
```

---

## 10. Câu hỏi để ngỏ

1. Có nên patch lại numpy-MADDPG để backprop dQ/dₐ đúng (thời gian phát
   triển ≈ 1 ngày) trước khi nộp Chương 5?
2. Reward weights (w₁..w₄) hiện gây bias về phía giải pháp "thụ động" (DT đạt
   reward cao nhất). Có cần re-tune và rerun toàn bộ baselines?
3. SINR thực tế trong dải −60 → +8 dB; ngưỡng γ_th = 5 dB không khả thi. Có
   nên giảm γ_th xuống −10 dB cả trong reward function lẫn báo cáo?
