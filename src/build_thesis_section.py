"""
build_thesis_section.py — Generate a Vietnamese LaTeX-ready section for
Chapter 5 of the thesis from the saved metrics in results_compare/.

Outputs:
  • chapter5_results.tex  — LaTeX figure + table macros
  • chapter5_results.md   — Markdown mirror for preview
"""

import argparse
import json
import os


METHOD_VI = {
    "IA-MADDPG+UAV":   "IA-MADDPG (UAV-relay)",
    "IA-MADDPG(RBS)":  "IA-MADDPG (chỉ RBS)",
    "StandardMADDPG":  "MADDPG tiêu chuẩn",
    "Greedy":          "Tham lam (Greedy)",
    "FreqHopping":     "Nhảy tần ngẫu nhiên (FH)",
    "DirectTX":        "Truyền trực tiếp (DT)",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results_compare/")
    args = parser.parse_args()

    with open(os.path.join(args.results_dir, "metrics_summary.json")) as f:
        raw = json.load(f)

    # New schema produced by re_evaluate.py
    metrics = {r["method"]: {
        "avg_reward":     r["reward"],
        "std_reward":     r.get("reward_std", 0.0),
        "avg_tsr":        r.get("tsr@-10dB", 0.0),  # ngưỡng ý nghĩa
        "std_tsr":        0.0,
        "avg_throughput": r["throughput"],
        "energy_efficiency": r["energy_efficiency"],
        "mode_counts":    [r["mode_d2d"], r["mode_rbs"], r["mode_uav"]],
    } for r in raw.get("summary", [])}

    # ── LaTeX section ────────────────────────────────────────────────────────
    tex = []
    tex.append(r"% Chapter 5 — Auto-generated results section")
    tex.append(r"\section{So sánh các phương pháp}")
    tex.append(r"\begin{table}[!htbp]")
    tex.append(r"\centering")
    tex.append(r"\caption{So sánh hiệu năng IA-MADDPG+UAV với các baselines.}")
    tex.append(r"\label{tab:methods-compare}")
    tex.append(r"\begin{tabular}{l c c c c}")
    tex.append(r"\hline")
    tex.append(r"\textbf{Phương pháp} & \textbf{Phần thưởng} "
               r"& \textbf{TSR @-10 dB} & \textbf{Thông lượng} "
               r"& \textbf{Hiệu quả NL} \\")
    tex.append(r"\hline")
    for name, v in metrics.items():
        tex.append(
            f"{METHOD_VI.get(name, name)} & "
            f"{v['avg_reward']:+.3f} $\\pm$ {v['std_reward']:.3f} & "
            f"{v['avg_tsr']:.4f} & "
            f"{v['avg_throughput']:.4f} & "
            f"{v['energy_efficiency']:.3f} \\\\"
        )
    tex.append(r"\hline")
    tex.append(r"\end{tabular}")
    tex.append(r"\end{table}")
    tex.append("")
    for fig, cap in [
        ("training_curves.png", "Đường cong phần thưởng huấn luyện."),
        ("convergence_tsr.png", "Tốc độ hội tụ của tỷ lệ truyền thành công (TSR)."),
        ("throughput_comparison.png", "So sánh thông lượng trung bình."),
        ("mode_distribution.png",     "Phân bố chế độ truyền (D2D/RBS/UAV)."),
        ("tsr_vs_threshold.png",      "TSR theo ngưỡng SINR (toàn dải)."),
        ("sinr_cdf.png",              "CDF của SINR đầu ra eval policy."),
        ("uav_trajectory.png",        "Quỹ đạo UAV của IA-MADDPG+UAV."),
        ("energy_efficiency.png",     "Hiệu quả năng lượng các phương pháp."),
    ]:
        tex.append(r"\begin{figure}[!htbp]")
        tex.append(r"\centering")
        tex.append(rf"\includegraphics[width=0.78\linewidth]{{figures/{fig}}}")
        tex.append(rf"\caption{{{cap}}}")
        tex.append(rf"\label{{fig:{os.path.splitext(fig)[0]}}}")
        tex.append(r"\end{figure}")
        tex.append("")
    tex_path = os.path.join(args.results_dir, "chapter5_results.tex")
    with open(tex_path, "w") as f:
        f.write("\n".join(tex))
    print(f"[done] LaTeX → {tex_path}")

    # ── Markdown mirror ──────────────────────────────────────────────────────
    md = []
    md.append("# Chương 5 — Kết quả mô phỏng và đánh giá\n")
    md.append("## 5.1 Bảng so sánh các phương pháp\n")
    md.append("| Phương pháp | Phần thưởng | TSR @-10 dB | "
              "Thông lượng (b/s/Hz) | Hiệu quả NL | D2D / RBS / UAV |")
    md.append("|---|---|---|---|---|---|")
    for name, v in metrics.items():
        mc = v.get("mode_counts", [0, 0, 0])
        md.append(
            f"| **{METHOD_VI.get(name, name)}** | "
            f"{v['avg_reward']:+.3f} ± {v['std_reward']:.3f} | "
            f"{v['avg_tsr']:.4f} | "
            f"{v['avg_throughput']:.4f} | "
            f"{v['energy_efficiency']:.3f} | "
            f"{mc[0]:.2f} / {mc[1]:.2f} / {mc[2]:.2f} |")
    md.append("\n## 5.2 Hình ảnh kết quả\n")
    for fig in ["training_curves.png", "convergence_tsr.png",
                "throughput_comparison.png", "mode_distribution.png",
                "tsr_vs_threshold.png", "sinr_cdf.png", "uav_trajectory.png",
                "energy_efficiency.png", "reward_comparison.png"]:
        md.append(f"![{fig}]({fig})\n")
    md_path = os.path.join(args.results_dir, "chapter5_results.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md))
    print(f"[done] Markdown → {md_path}")


if __name__ == "__main__":
    main()
