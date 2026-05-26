import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "matplotlib", "pandas"])

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

OPT_CSV = "/workspace/results/optimization_results.csv"
ADT_CSV = "/workspace/results/adaptive_results.csv"
OUT_DIR = "/workspace/results/figures"
os.makedirs(OUT_DIR, exist_ok=True)

opt = pd.read_csv(OPT_CSV)
adt = pd.read_csv(ADT_CSV)

# backward-compat
if "model_name" not in opt.columns:
    opt["model_name"] = "gpt2"
if "model_name" not in adt.columns:
    adt["model_name"] = "gpt2"

MODELS        = sorted(adt["model_name"].unique())
BATCH_SIZES   = sorted(adt["batch_size"].unique())
INPUT_LENGTHS = sorted(adt["input_length"].unique())

MODEL_DISPLAY = {
    "gpt2":         "GPT-2 (124M)",
    "gpt2-large":   "GPT-2 Large (762M)",
    "gpt-neo-125m": "GPT-Neo (125M)",
}

STATIC_CONFIGS = {"baseline": "#555555", "fp16": "#1f77b4", "compile": "#2ca02c"}
ADAPTIVE_COLOR = "#d62728"

REGIME_COLORS = {
    "low-utilization":       "#aec6e8",
    "kernel-overhead-bound": "#98df8a",
    "memory-bound":          "#ffbb78",
}


def get_static(model_name, config_name, batch_size, input_length):
    row = opt[(opt["model_name"] == model_name) &
              (opt["config"] == config_name) &
              (opt["batch_size"] == batch_size) &
              (opt["input_length"] == input_length)]
    return row["throughput_tok_s"].values[0] if len(row) > 0 else None


# ── Figure 1: Throughput — rows=models, cols=seq ─────────────────────────────
fig, axes = plt.subplots(len(MODELS), len(INPUT_LENGTHS),
                         figsize=(5.5 * len(INPUT_LENGTHS), 5 * len(MODELS)),
                         sharey=False, squeeze=False)

for row_i, model_name in enumerate(MODELS):
    madt = adt[adt["model_name"] == model_name]
    for col_i, seq in enumerate(INPUT_LENGTHS):
        ax = axes[row_i][col_i]
        for cfg, color in STATIC_CONFIGS.items():
            ys = [get_static(model_name, cfg, bs, seq) for bs in BATCH_SIZES]
            valid = [(bs, y) for bs, y in zip(BATCH_SIZES, ys) if y is not None]
            if valid:
                xs, ys_v = zip(*valid)
                ax.plot(xs, ys_v, marker="o", color=color, linestyle="--",
                        linewidth=1.4, alpha=0.7, label=cfg)
        sub = madt[madt["input_length"] == seq].sort_values("batch_size")
        ax.plot(sub["batch_size"], sub["throughput_tok_s"],
                marker="*", markersize=10, color=ADAPTIVE_COLOR,
                linewidth=2, label="adaptive")
        for _, row in sub.iterrows():
            ax.axvspan(row["batch_size"] - 0.4, row["batch_size"] + 0.4,
                       alpha=0.15, color=REGIME_COLORS[row["regime"]], zorder=0)
        ax.set_title(f"{MODEL_DISPLAY.get(model_name, model_name)}\nseq={seq}", fontsize=9)
        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Throughput (tok/s)")
        ax.set_xticks(BATCH_SIZES)
        ax.legend(fontsize=7)
        ax.grid(True, linestyle="--", alpha=0.35)

regime_patches = [mpatches.Patch(color=v, alpha=0.4, label=k)
                  for k, v in REGIME_COLORS.items()]
fig.legend(handles=regime_patches, loc="lower center", ncol=3,
           title="Regime", fontsize=8, bbox_to_anchor=(0.5, -0.02))
fig.suptitle("Throughput: Adaptive Policy vs Static Configs", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "adaptive_throughput.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved: adaptive_throughput.png")


# ── Figure 2: Speedup of adaptive over each static ───────────────────────────
fig, axes = plt.subplots(len(MODELS), len(INPUT_LENGTHS),
                         figsize=(5.5 * len(INPUT_LENGTHS), 5 * len(MODELS)),
                         sharey=True, squeeze=False)

has_ci = "speedup_ci95" in adt.columns

for row_i, model_name in enumerate(MODELS):
    madt = adt[adt["model_name"] == model_name]
    for col_i, seq in enumerate(INPUT_LENGTHS):
        ax = axes[row_i][col_i]
        sub_adt = madt[madt["input_length"] == seq].sort_values("batch_size")
        for cfg, color in STATIC_CONFIGS.items():
            speedups, xs, errs = [], [], []
            for _, row in sub_adt.iterrows():
                if cfg == "baseline" and pd.notna(row["speedup"]) and row["speedup"] > 0:
                    # Use in-experiment speedup to eliminate cross-experiment GPU noise
                    speedups.append(float(row["speedup"]))
                    xs.append(row["batch_size"])
                    if has_ci and pd.notna(row.get("speedup_ci95", float("nan"))):
                        errs.append(float(row["speedup_ci95"]))
                    else:
                        errs.append(0)
                else:
                    s = get_static(model_name, cfg, row["batch_size"], seq)
                    if s is not None and s > 0:
                        speedups.append(row["throughput_tok_s"] / s)
                        xs.append(row["batch_size"])
                        errs.append(0)
            if has_ci and any(e > 0 for e in errs) and cfg == "baseline":
                ax.errorbar(xs, speedups, yerr=errs, fmt="o-", color=color,
                            capsize=3, linewidth=1.5, label=f"vs {cfg}")
            else:
                ax.plot(xs, speedups, marker="o", color=color,
                        label=f"vs {cfg}", linewidth=1.5)
        ax.axhline(1.0, color="gray", linestyle=":", linewidth=1.2, alpha=0.7)
        ax.set_title(f"{MODEL_DISPLAY.get(model_name, model_name)}\nseq={seq}", fontsize=9)
        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Speedup (adaptive / static)")
        ax.set_xticks(BATCH_SIZES)
        ax.legend(fontsize=7)
        ax.grid(True, linestyle="--", alpha=0.35)

fig.suptitle("Adaptive Policy Speedup over Static Configs", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "adaptive_speedup.png"), dpi=150)
plt.close()
print("Saved: adaptive_speedup.png")


# ── Figure 3: Regime heatmap — one per model ─────────────────────────────────
REGIME_INT   = {"low-utilization": 0, "kernel-overhead-bound": 1, "memory-bound": 2}
REGIME_LABEL = {0: "low-util", 1: "kernel-oh", 2: "memory-bound"}

fig, axes = plt.subplots(1, len(MODELS), figsize=(9 * len(MODELS), 4), squeeze=False)

for col_i, model_name in enumerate(MODELS):
    ax   = axes[0][col_i]
    madt = adt[adt["model_name"] == model_name]
    matrix   = np.full((len(INPUT_LENGTHS), len(BATCH_SIZES)), np.nan)
    text_mat = [[""] * len(BATCH_SIZES) for _ in range(len(INPUT_LENGTHS))]

    for _, row in madt.iterrows():
        if row["input_length"] not in INPUT_LENGTHS or row["batch_size"] not in BATCH_SIZES:
            continue
        r = INPUT_LENGTHS.index(row["input_length"])
        c = BATCH_SIZES.index(row["batch_size"])
        matrix[r, c]   = REGIME_INT[row["regime"]]
        text_mat[r][c] = f"{row['selected_config']}\n{row['throughput_tok_s']:.0f}"

    im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=2)
    ax.set_xticks(range(len(BATCH_SIZES)))
    ax.set_xticklabels([f"bs={b}" for b in BATCH_SIZES])
    ax.set_yticks(range(len(INPUT_LENGTHS)))
    ax.set_yticklabels([f"seq={s}" for s in INPUT_LENGTHS])
    ax.set_title(MODEL_DISPLAY.get(model_name, model_name))
    for r in range(len(INPUT_LENGTHS)):
        for c in range(len(BATCH_SIZES)):
            if text_mat[r][c]:
                ax.text(c, r, text_mat[r][c], ha="center", va="center",
                        fontsize=8, color="black")
    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(["low-util", "kernel-oh", "memory-bound"])

fig.suptitle("Adaptive Policy: Regime + Selected Config + Throughput (tok/s)", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "adaptive_regime_heatmap.png"), dpi=150)
plt.close()
print("Saved: adaptive_regime_heatmap.png")


# ── Summary ──────────────────────────────────────────────────────────────────
print("\n=== Adaptive vs Static — Mean Throughput (tok/s) ===")
for model_name in MODELS:
    madt = adt[adt["model_name"] == model_name]
    mopt = opt[opt["model_name"] == model_name]
    print(f"\n  [{MODEL_DISPLAY.get(model_name, model_name)}]")
    summary = {"adaptive": madt["throughput_tok_s"].mean()}
    for cfg in STATIC_CONFIGS:
        vals = mopt[mopt["config"] == cfg]["throughput_tok_s"]
        summary[cfg] = vals.mean()
    for k, v in summary.items():
        print(f"    {k:12s}: {v:.1f} tok/s")

    # Use in-experiment speedup column if available, else fall back to cross-experiment ratio
    print(f"  Speedup by regime (in-experiment baseline):")
    for regime in ["low-utilization", "kernel-overhead-bound", "memory-bound"]:
        sub = madt[madt["regime"] == regime]
        if sub.empty:
            continue
        if "speedup" in sub.columns and sub["speedup"].notna().any():
            gains = sub["speedup"].dropna().tolist()
        else:
            gains = []
            for _, row in sub.iterrows():
                b = get_static(model_name, "baseline", row["batch_size"], row["input_length"])
                if b:
                    gains.append(row["throughput_tok_s"] / b)
        if gains:
            print(f"    {regime:28s}: {np.mean(gains):.3f}x avg  "
                  f"(min={min(gains):.3f}x  max={max(gains):.3f}x)")

print(f"\nAll figures → {OUT_DIR}")
