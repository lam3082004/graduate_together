# Kết quả so sánh các phương án (Chương 5)

**Cấu hình:** 60 ep × 50 steps, warmup=800, batch=128, seed=42, eval=30 ep.

## Bảng tổng hợp

| Method | Reward (±std) | TSR (±std) | Throughput | Energy Eff. | D2D/RBS/UAV |
|---|---|---|---|---|---|
| IA-MADDPG+UAV | -0.4741 ± 0.0134 | 0.0000 ± 0.0000 | 0.0074 | 0.0000 | 0.00/0.40/0.60 |
| IA-MADDPG(RBS) | -0.4645 ± 0.0191 | 0.0000 ± 0.0000 | 0.0153 | 15259.2394 | 0.20/0.80/0.00 |
| StandardMADDPG | -0.4526 ± 0.0231 | 0.0001 ± 0.0007 | 0.0108 | 0.0000 | 0.40/0.20/0.40 |
| Greedy | -0.4512 ± 0.0321 | 0.0004 ± 0.0016 | 0.0274 | 27386.6149 | 0.56/0.44/0.00 |
| FreqHopping | -0.3883 ± 0.0033 | 0.0000 ± 0.0000 | 0.0026 | 2608.8369 | 0.50/0.50/0.00 |
| DirectTX | -0.3832 ± 0.0019 | 0.0000 ± 0.0000 | 0.0015 | 1479.5191 | 1.00/0.00/0.00 |

## Hình ảnh sinh ra

- `training_curves.png`
- `convergence_tsr.png`
- `throughput_comparison.png`
- `reward_comparison.png`
- `tsr_comparison.png`
- `energy_efficiency.png`
- `mode_distribution.png`
- `tsr_vs_threshold.png`
- `uav_trajectory.png`
