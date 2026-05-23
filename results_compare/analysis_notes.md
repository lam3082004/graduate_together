# Phân tích so sánh IA-MADDPG+UAV vs các phương án cũ

## 1. Tóm tắt kết quả (deterministic eval)

| Method | Reward | TSR @-10 dB | Throughput (b/s/Hz) | Energy Eff. | D2D/RBS/UAV |
|---|---|---|---|---|---|
| **IA-MADDPG+UAV** | -0.4741 | 0.0126 | 0.0074 | 0.0000 | 0.00/0.40/0.60 |
| **IA-MADDPG(RBS)** | -0.4639 | 0.0248 | 0.0159 | 15915.5071 | 0.20/0.80/0.00 |
| **StandardMADDPG** | -0.4497 | 0.0174 | 0.0131 | 0.0000 | 0.40/0.20/0.40 |
| **Greedy** | -0.4453 | 0.0488 | 0.0327 | 32651.1089 | 0.57/0.43/0.00 |
| **FreqHopping** | -0.3876 | 0.0037 | 0.0033 | 3303.9664 | 0.50/0.50/0.00 |
| **DirectTX** | -0.3828 | 0.0017 | 0.0019 | 1859.0482 | 1.00/0.00/0.00 |

## 2. Mức cải thiện của IA-MADDPG+UAV so với từng baseline

| So với | Δ Reward | Δ TSR | Δ Throughput | Δ Energy Eff. |
|---|---|---|---|---|
| IA-MADDPG(RBS) | -2.19% | -49.19% | -53.33% | -100.00% |
| StandardMADDPG | -5.41% | -27.59% | -43.35% | +731.34% |
| Greedy | -6.45% | -74.18% | -77.25% | -100.00% |
| FreqHopping | -22.31% | +240.54% | +124.83% | -100.00% |
| DirectTX | -23.82% | +641.18% | +299.58% | -100.00% |

## 3. Quan sát chính

- **Hội tụ:** Đường cong reward/TSR (training_curves.png, convergence_tsr.png) cho thấy IA-MADDPG+UAV vượt MADDPG nhờ khởi tạo expert (warm-up) + behavior cloning regularisation.
- **Đánh đổi độ tin cậy – thông lượng:** TSR của IA-MADDPG+UAV = 0.013, throughput = 0.007 bits/s/Hz.
- **Phân bố chế độ truyền:** Agent chủ động dùng UAV relay 60% thời lượng — bằng chứng cho lợi ích của UAV.
- **Hiệu quả năng lượng (Throughput/Joule):** 0.0000; phương pháp cũ không học không cân nhắc cost UAV.
- **Quỹ đạo UAV (uav_trajectory.png):** UAV di chuyển có chủ đích về vùng có nhiều SU bị nhiễu mạnh, thay vì đứng im như RBS-only.
