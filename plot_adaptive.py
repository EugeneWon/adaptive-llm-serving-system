import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "matplotlib", "pandas"])

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

OPT_CSV  = "/workspace/results/optimization_results.csv"
ADT_CSV  = "/workspace/results/adaptive_results.csv"
OUT_DIR  = "/workspace/results/figures"
os.makedirs(OUT_DIR, exist_ok=True)

opt = pd.read_csv(OPT_CSV)
adt = pd.read_csv(ADT_CSV)

BATCH_SIZES   = sorted(adt["batch_size"].unique())
INPUT_LENGTHS = sorted(adt["input_length"].unique())

STATIC_CONFIGS = {
    "baseline": "#555555",
    "fp16":     "#1f77b4",
    "compile":  "#2ca02c",
}
ADAPTIVE_COLOR = "#d62728"

REGIME_COLORS = {
    "low-utilization":      "#aec6e8",
    "kernel-overhead-bound":"#98df8a",
    "memory-bound":         "#ffbb78",
}


def get_static(config_name, batch_size, input_length):
    row = opt[(opt["config"] == config_name) &
              (opt["batch_size"] == batch_size) &
              (opt["input_length"] == input_length)]
    return row["throughput_tok_s"].values[0] if len(row) > 0 else None


# ── Figure 1: Throughput 비교 (seq별 패널, adaptive vs static) ─────────────────
fig, axes = plt.subplots(1, len(INPUT_LENGTHS), figsize=(5.5 * len(INPUT_LENGTHS), 5), sharey=False)

for ax, seq in zip(axes, INPUT_LENGTHS):
    # static configs
    for cfg, color in STATIC_CONFIGS.items():
        ys = [get_static(cfg, bs, seq) for bs in BATCH_SIZES]
        valid = [(bs, y) for bs, y in zip(BATCH_SIZES, ys) if y is not None]
        if valid:
            xs, ys_v = zip(*valid)
            ax.plot(xs, ys_v, marker="o", color=color, linestyle="--",
                    linewidth=1.4, alpha=0.7, label=cfg)

    # adaptive
    sub = adt[adt["input_length"] == seq].sort_values("batch_size")
    ax.plot(sub["batch_size"], sub["throughput_tok_s"],
            marker="*", markersize=10, color=ADAPTIVE_COLOR,
            linewidth=2, label="adaptive")

    # regime 배경
    for _, row in sub.iterrows():
        ax.axvspan(row["batch_size"] - 0.4, row["batch_size"] + 0.4,
                   alpha=0.15, color=REGIME_COLORS[row["regime"]], zorder=0)

    ax.set_title(f"seq={seq}", fontsize=12)
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Throughput (tok/s)")
    ax.set_xticks(BATCH_SIZES)
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.35)

regime_patches = [mpatches.Patch(color=v, alpha=0.4, label=k)
                  for k, v in REGIME_COLORS.items()]
fig.legend(handles=regime_patches, loc="lower center", ncol=3,
           title="Regime", fontsize=8, bbox_to_anchor=(0.5, -0.04))
fig.suptitle("Throughput: Adaptive Policy vs Static Configs", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "adaptive_throughput.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved: adaptive_throughput.png")


# ── Figure 2: Speedup of adaptive over each static (seq별 패널) ───────────────
fig, axes = plt.subplots(1, len(INPUT_LENGTHS), figsize=(5.5 * len(INPUT_LENGTHS), 5), sharey=True)

for ax, seq in zip(axes, INPUT_LENGTHS):
    sub_adt = adt[adt["input_length"] == seq].sort_values("batch_size")

    for cfg, color in STATIC_CONFIGS.items():
        speedups = []
        xs = []
        for _, row in sub_adt.iterrows():
            s = get_static(cfg, row["batch_size"], seq)
            if s is not None and s > 0:
                speedups.append(row["throughput_tok_s"] / s)
                xs.append(row["batch_size"])
        ax.plot(xs, speedups, marker="o", color=color,
                label=f"vs {cfg}", linewidth=1.5)

    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1.2, alpha=0.7)
    ax.set_title(f"seq={seq}", fontsize=12)
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Speedup (adaptive / static)")
    ax.set_xticks(BATCH_SIZES)
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.35)

fig.suptitle("Adaptive Policy Speedup over Static Configs", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "adaptive_speedup.png"), dpi=150)
plt.close()
print("Saved: adaptive_speedup.png")


# ── Figure 3: Regime 분포 히트맵 ──────────────────────────────────────────────
REGIME_INT = {"low-utilization": 0, "kernel-overhead-bound": 1, "memory-bound": 2}
REGIME_LABEL = {0: "low-util", 1: "kernel-oh", 2: "memory-bound"}

matrix   = np.full((len(INPUT_LENGTHS), len(BATCH_SIZES)), np.nan)
text_mat = [[""] * len(BATCH_SIZES) for _ in range(len(INPUT_LENGTHS))]

for _, row in adt.iterrows():
    r = INPUT_LENGTHS.index(row["input_length"])
    c = BATCH_SIZES.index(row["batch_size"])
    matrix[r, c]   = REGIME_INT[row["regime"]]
    text_mat[r][c] = f"{row['selected_config']}\n{row['throughput_tok_s']:.0f}"

fig, ax = plt.subplots(figsize=(9, 4))
im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=2)

ax.set_xticks(range(len(BATCH_SIZES)))
ax.set_xticklabels([f"bs={b}" for b in BATCH_SIZES])
ax.set_yticks(range(len(INPUT_LENGTHS)))
ax.set_yticklabels([f"seq={s}" for s in INPUT_LENGTHS])
ax.set_title("Adaptive Policy: Regime + Selected Config + Throughput (tok/s)")

for r in range(len(INPUT_LENGTHS)):
    for c in range(len(BATCH_SIZES)):
        if text_mat[r][c]:
            ax.text(c, r, text_mat[r][c], ha="center", va="center",
                    fontsize=8, color="black")

cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2])
cbar.ax.set_yticklabels(["low-util", "kernel-oh", "memory-bound"])
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "adaptive_regime_heatmap.png"), dpi=150)
plt.close()
print("Saved: adaptive_regime_heatmap.png")


# ── 요약 테이블 ───────────────────────────────────────────────────────────────
print("\n=== Adaptive vs Static — Mean Throughput (tok/s) ===")
summary = {"adaptive": adt["throughput_tok_s"].mean()}
for cfg in STATIC_CONFIGS:
    vals = opt[opt["config"] == cfg]["throughput_tok_s"]
    summary[cfg] = vals.mean()
for k, v in summary.items():
    print(f"  {k:12s}: {v:.1f} tok/s")

print("\n=== Adaptive Speedup over Baseline (by regime) ===")
for regime in ["low-utilization", "kernel-overhead-bound", "memory-bound"]:
    sub = adt[adt["regime"] == regime]
    if sub.empty:
        continue
    gains = []
    for _, row in sub.iterrows():
        b = get_static("baseline", row["batch_size"], row["input_length"])
        if b:
            gains.append(row["throughput_tok_s"] / b)
    if gains:
        print(f"  {regime:25s}: {np.mean(gains):.3f}x avg speedup")

print(f"\nAll figures → {OUT_DIR}")
