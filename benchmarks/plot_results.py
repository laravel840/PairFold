#!/usr/bin/env python3
"""Generate paper-ready benchmark plots from benchmark_results.csv."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

BENCH_DIR = Path(__file__).resolve().parent
CSV_PATH = BENCH_DIR / "results" / "benchmark_results.csv"
OUT_PATH = BENCH_DIR / "results" / "benchmark_plots.png"


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Missing {CSV_PATH}. Run benchmark.py first.")

    df = pd.read_csv(CSV_PATH)
    required = {"pdb_id", "length", "time_s", "rmsd"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")

    df = df.sort_values("length").reset_index(drop=True)

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
    palette = sns.color_palette("deep")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)

    # Left: Inference Speed
    ax = axes[0]
    sns.scatterplot(
        data=df,
        x="length",
        y="time_s",
        hue="pdb_id",
        s=90,
        palette=palette[: len(df)],
        ax=ax,
        legend=False,
        zorder=3,
    )
    sns.lineplot(
        data=df,
        x="length",
        y="time_s",
        color=palette[0],
        linewidth=1.8,
        marker="o",
        markersize=7,
        ax=ax,
        legend=False,
        zorder=2,
    )
    for _, row in df.iterrows():
        ax.annotate(
            str(row["pdb_id"]),
            (row["length"], row["time_s"]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
            color="#333333",
        )
    ax.set_xlabel("Sequence length (residues)")
    ax.set_ylabel("Execution time (s)")
    ax.set_title("Inference Speed")

    # Right: Prediction Error
    ax = axes[1]
    sns.scatterplot(
        data=df,
        x="length",
        y="rmsd",
        hue="pdb_id",
        s=90,
        palette=palette[: len(df)],
        ax=ax,
        legend=False,
        zorder=3,
    )
    sns.lineplot(
        data=df,
        x="length",
        y="rmsd",
        color=palette[3],
        linewidth=1.8,
        marker="o",
        markersize=7,
        ax=ax,
        legend=False,
        zorder=2,
    )
    for _, row in df.iterrows():
        ax.annotate(
            str(row["pdb_id"]),
            (row["length"], row["rmsd"]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
            color="#333333",
        )
    ax.set_xlabel("Sequence length (residues)")
    ax.set_ylabel(r"C$\alpha$ RMSD ($\mathrm{\AA}$)")
    ax.set_title("Prediction Error")

    fig.suptitle("PairFold Benchmark Results", fontsize=14, fontweight="bold", y=1.02)

    fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
