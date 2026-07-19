#!/usr/bin/env python3
"""Plots for expanded paper benchmarks: length–RMSD, class boxplots, long-chain time."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

BENCH = Path(__file__).resolve().parent
RES = BENCH / "results"
DOM = RES / "benchmark_domains100.csv"
LONG = RES / "benchmark_long100.csv"
ABL = RES / "benchmark_ablation_summary.json"
OUT1 = RES / "benchmark_plots_expanded.png"
OUT2 = RES / "benchmark_ablation_plot.png"
PAPER = BENCH.parent / "paper"


def main() -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)

    if DOM.exists():
        df = pd.read_csv(DOM)
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)

        ax = axes[0]
        for cls, marker in (
            ("all_alpha", "o"),
            ("all_beta", "s"),
            ("alpha_beta", "D"),
        ):
            sub = df[df["class"] == cls]
            if sub.empty:
                continue
            ax.scatter(
                sub["length"],
                sub["rmsd"],
                label=cls.replace("_", "/"),
                marker=marker,
                s=36,
                alpha=0.85,
            )
        if len(df) > 2:
            coef = np.polyfit(df["length"], df["rmsd"], 1)
            xs = np.linspace(df["length"].min(), df["length"].max(), 50)
            ax.plot(xs, np.poly1d(coef)(xs), "--", color="#444", linewidth=1.2)
            corr = float(np.corrcoef(df["length"], df["rmsd"])[0, 1])
            ax.set_title(f"Length vs Cα RMSD (r={corr:.2f}, n={len(df)})")
        else:
            ax.set_title("Length vs Cα RMSD")
        ax.set_xlabel("Sequence length (residues)")
        ax.set_ylabel("Kabsch Cα RMSD (Å)")
        ax.legend(fontsize=8)

        ax = axes[1]
        order = ["all_alpha", "all_beta", "alpha_beta"]
        present = [c for c in order if (df["class"] == c).any()]
        sns.boxplot(
            data=df,
            x="class",
            y="rmsd",
            order=present,
            ax=ax,
            color="#c4894a",
        )
        sns.stripplot(
            data=df,
            x="class",
            y="rmsd",
            order=present,
            ax=ax,
            color="#333",
            size=3,
            alpha=0.55,
        )
        ax.set_xlabel("Protein class")
        ax.set_ylabel("Kabsch Cα RMSD (Å)")
        ax.set_title("RMSD by structural class")
        ax.set_xticklabels([c.replace("_", "/") for c in present])

        fig.savefig(OUT1, dpi=200)
        fig.savefig(PAPER / "benchmark_plots_expanded.png", dpi=200)
        print(f"Wrote {OUT1}", flush=True)
        plt.close(fig)

    if LONG.exists():
        ldf = pd.read_csv(LONG)
        fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
        ax.scatter(ldf["length"], ldf["time_s"], s=28, alpha=0.8, color="#4b6bfb")
        ax.set_xlabel("Tiled sequence length")
        ax.set_ylabel("Wall-clock time (s)")
        ax.set_title(f"Long-chain scaling (n={len(ldf)}, 10k–50k)")
        fig.savefig(RES / "benchmark_long100_time.png", dpi=200)
        fig.savefig(PAPER / "benchmark_long100_time.png", dpi=200)
        print("Wrote long-chain time plot", flush=True)
        plt.close(fig)

    if ABL.exists():
        import json

        summary = json.loads(ABL.read_text(encoding="utf-8"))
        labels = [
            ("base_consensus", "Base"),
            ("plus_lookahead", "+Look-ahead"),
            ("plus_lever", "+Lever polish"),
            ("plus_ss", "+SS freeze"),
            ("full_pipeline", "Full"),
        ]
        xs, ys = [], []
        for key, lab in labels:
            if key in summary:
                xs.append(lab)
                ys.append(summary[key]["mean_rmsd"])
        if xs:
            fig, ax = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)
            ax.bar(xs, ys, color="#6b8f71")
            ax.set_ylabel("Mean Cα RMSD (Å)")
            ax.set_title("Ablation (mean over subset)")
            for i, v in enumerate(ys):
                ax.text(i, v + 0.15, f"{v:.2f}", ha="center", fontsize=8)
            fig.savefig(OUT2, dpi=200)
            fig.savefig(PAPER / "benchmark_ablation_plot.png", dpi=200)
            print(f"Wrote {OUT2}", flush=True)
            plt.close(fig)


if __name__ == "__main__":
    main()
