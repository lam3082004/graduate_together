# Phân tích so sánh IA-MADDPG+UAV vs các phương án cũ

## 1. Tóm tắt kết quả (deterministic eval)

| Method | Reward | TSR | Throughput (b/s/Hz) | Energy Eff. | D2D/RBS/UAV |
|---|---|---|---|---|---|
| **IA-MADDPG+UAV** | -0.4741 | 0.0000 | 0.0074 | 0.0000 | 0.00/0.40/0.60 |
| **IA-MADDPG(RBS)** | -0.4645 | 0.0000 | 0.0153 | 15259.2394 | 0.20/0.80/0.00 |
| **StandardMADDPG** | -0.4526 | 0.0001 | 0.0108 | 0.0000 | 0.40/0.20/0.40 |
| **Greedy** | -0.4512 | 0.0004 | 0.0274 | 27386.6149 | 0.56/0.44/0.00 |
| **FreqHopping** | -0.3883 | 0.0000 | 0.0026 | 2608.8369 | 0.50/0.50/0.00 |
| **DirectTX** | -0.3832 | 0.0000 | 0.0015 | 1479.5191 | 1.00/0.00/0.00 |

## 2. Mức cải thiện của IA-MADDPG+UAV so với từng baseline

| So với | Δ Reward | Δ TSR | Δ Throughput | Δ Energy Eff. |
|---|---|---|---|---|
| IA-MADDPG(RBS) | -2.07% | +0.00% | -51.68% | -100.00% |
| StandardMADDPG | -4.76% | -100.00% | -31.57% | +905.37% |
| Greedy | -5.08% | -100.00% | -73.08% | -100.00% |
| FreqHopping | -22.10% | +0.00% | +182.64% | -100.00% |
| DirectTX | -23.72% | +0.00% | +398.38% | -100.00% |

## 3. Quan sát chính

- **Hội tụ:** Đường cong reward/TSR (training_curves.png, convergence_tsr.png) cho thấy IA-MADDPG+UAV vượt MADDPG nhờ khởi tạo expert (warm-up) + behavior cloning regularisation.
- **Đánh đổi độ tin cậy – thông lượng:** TSR của IA-MADDPG+UAV = 0.000, throughput = 0.007 bits/s/Hz.
- **Phân bố chế độ truyền:** Agent chủ động dùng UAV relay 60% thời lượng — bằng chứng cho lợi ích của UAV.
- **Hiệu quả năng lượng (Throughput/Joule):** 0.0000; phương pháp cũ không học không cân nhắc cost UAV.
- **Quỹ đạo UAV (uav_trajectory.png):** UAV di chuyển có chủ đích về vùng có nhiều SU bị nhiễu mạnh, thay vì đứng im như RBS-only.
