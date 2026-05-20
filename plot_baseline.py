import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "matplotlib", "pandas"])

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

CSV_PATH = "/workspace/results/baseline_results.csv"
OUT_DIR  = "/workspace/results/figures"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)

# backward-compat: if no model_name column, assume gpt2
if "model_name" not in df.columns:
    df["model_name"] = "gpt2"

MODELS        = sorted(df["model_name"].unique())
INPUT_LENGTHS = sorted(df["input_length"].unique())
BATCH_SIZES   = sorted(df["batch_size"].unique())
COLORS        = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

REGIME_COLORS = {
    "low-utilization": "#aec6e8",
    "compute-scaling": "#98df8a",
    "memory-bound":    "#ffbb78",
}

MODEL_DISPLAY = {
    "gpt2":         "GPT-2 (124M)",
    "gpt-neo-125m": "GPT-Neo (125M)",
}


def add_regime(sub_df):
    ref = sub_df[sub_df["batch_size"] == 1].set_index("input_length")["throughput_tok_s"]
    sub_df = sub_df.copy()
    sub_df["scaling_eff"] = sub_df.apply(
        lambda r: r["throughput_tok_s"] / (ref[r["input_length"]] * r["batch_size"]), axis=1
    )
    def classify(row):
        if row["gpu_util_pct"] < 40:
            return "low-utilization"
        elif row["scaling_eff"] < 0.75:
            return "memory-bound"
        else:
            return "compute-scaling"
    sub_df["regime"] = sub_df.apply(classify, axis=1)
    return sub_df


# add regime per model
df = pd.concat([add_regime(g) for _, g in df.groupby("model_name")], ignore_index=True)

print("\n=== Regime Classification ===")
print(df[["model_name", "batch_size", "input_length",
          "gpu_util_pct", "scaling_eff", "regime"]].to_string(index=False))


# ── Figure 1: Throughput vs Batch Size (side-by-side per model) ───────────────
fig, axes = plt.subplots(1, len(MODELS), figsize=(7 * len(MODELS), 4.5), sharey=False)
if len(MODELS) == 1:
    axes = [axes]

for ax, model_name in zip(axes, MODELS):
    sub = df[df["model_name"] == model_name]
    for i, seq_len in enumerate(INPUT_LENGTHS):
        s = sub[sub["input_length"] == seq_len].sort_values("batch_size")
        ax.plot(s["batch_size"], s["throughput_tok_s"],
                marker="o", label=f"seq={seq_len}", color=COLORS[i])

    ref_val = sub[(sub["batch_size"] == 1) & (sub["input_length"] == INPUT_LENGTHS[0])]["throughput_tok_s"].values[0]
    ax.plot(BATCH_SIZES, [ref_val * b for b in BATCH_SIZES],
            "k--", linewidth=1, alpha=0.4, label="ideal (linear)")
    ax.set_title(MODEL_DISPLAY.get(model_name, model_name))
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Throughput (tok/s)")
    ax.set_xticks(BATCH_SIZES)
    ax.legend(title="Input Length", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)

fig.suptitle("Throughput vs Batch Size", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "baseline_throughput_vs_batch.png"), dpi=150)
plt.close()
print("Saved: baseline_throughput_vs_batch.png")


# ── Figure 2: Latency vs Batch Size ──────────────────────────────────────────
fig, axes = plt.subplots(1, len(MODELS), figsize=(7 * len(MODELS), 4.5), sharey=False)
if len(MODELS) == 1:
    axes = [axes]

for ax, model_name in zip(axes, MODELS):
    sub = df[df["model_name"] == model_name]
    for i, seq_len in enumerate(INPUT_LENGTHS):
        s = sub[sub["input_length"] == seq_len].sort_values("batch_size")
        ax.plot(s["batch_size"], s["avg_latency_s"],
                marker="o", label=f"seq={seq_len}", color=COLORS[i])
        ax.fill_between(s["batch_size"], s["p50_latency_s"], s["p95_latency_s"],
                        alpha=0.12, color=COLORS[i])
    ax.set_title(MODEL_DISPLAY.get(model_name, model_name))
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Latency (s)")
    ax.set_xticks(BATCH_SIZES)
    ax.legend(title="Input Length", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)

fig.suptitle("Latency vs Batch Size (50 new tokens)", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "baseline_latency_vs_batch.png"), dpi=150)
plt.close()
print("Saved: baseline_latency_vs_batch.png")


# ── Figure 3: GPU Util + Regime Map ──────────────────────────────────────────
fig, axes = plt.subplots(1, len(MODELS), figsize=(7 * len(MODELS), 4.5), sharey=True)
if len(MODELS) == 1:
    axes = [axes]

for ax, model_name in zip(axes, MODELS):
    sub = df[df["model_name"] == model_name]
    for i, seq_len in enumerate(INPUT_LENGTHS):
        s = sub[sub["input_length"] == seq_len].sort_values("batch_size")
        for _, row in s.iterrows():
            ax.axvspan(row["batch_size"] - 0.35, row["batch_size"] + 0.35,
                       alpha=0.25, color=REGIME_COLORS[row["regime"]], zorder=0)
        ax.plot(s["batch_size"], s["gpu_util_pct"],
                marker="o", label=f"seq={seq_len}", color=COLORS[i])

    ax.axhline(40, color="gray", linestyle=":", linewidth=1, label="util=40% threshold")
    ax.set_title(MODEL_DISPLAY.get(model_name, model_name))
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("GPU Utilization (%)")
    ax.set_xticks(BATCH_SIZES)
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(title="Input Length", loc="lower right", fontsize=8)

patches = [mpatches.Patch(color=v, alpha=0.5, label=k) for k, v in REGIME_COLORS.items()]
fig.legend(handles=patches, loc="upper left", title="Regime", fontsize=8,
           bbox_to_anchor=(0.01, 0.99))
fig.suptitle("GPU Utilization — Regime Map", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "baseline_util_regime_map.png"), dpi=150)
plt.close()
print("Saved: baseline_util_regime_map.png")


# ── Figure 4: Scaling Efficiency ─────────────────────────────────────────────
fig, axes = plt.subplots(1, len(MODELS), figsize=(7 * len(MODELS), 4.5), sharey=True)
if len(MODELS) == 1:
    axes = [axes]

for ax, model_name in zip(axes, MODELS):
    sub = df[df["model_name"] == model_name]
    for i, seq_len in enumerate(INPUT_LENGTHS):
        s = sub[sub["input_length"] == seq_len].sort_values("batch_size")
        ax.plot(s["batch_size"], s["scaling_eff"] * 100,
                marker="o", label=f"seq={seq_len}", color=COLORS[i])
    ax.axhline(100, color="k",      linestyle="--", linewidth=1, alpha=0.4, label="ideal")
    ax.axhline(75,  color="orange", linestyle=":",  linewidth=1, alpha=0.7, label="memory-bound (75%)")
    ax.set_title(MODEL_DISPLAY.get(model_name, model_name))
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Scaling Efficiency (%)")
    ax.set_xticks(BATCH_SIZES)
    ax.set_ylim(0, 120)
    ax.legend(title="Input Length", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)

fig.suptitle("Throughput Scaling Efficiency vs Batch Size", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "baseline_scaling_efficiency.png"), dpi=150)
plt.close()
print("Saved: baseline_scaling_efficiency.png")


# ── Figure 5: Cross-model throughput comparison (new) ────────────────────────
if len(MODELS) > 1:
    model_linestyles = ["-", "--"]
    fig, axes = plt.subplots(1, len(INPUT_LENGTHS), figsize=(5 * len(INPUT_LENGTHS), 4.5), sharey=False)
    if len(INPUT_LENGTHS) == 1:
        axes = [axes]

    for ax, seq_len in zip(axes, INPUT_LENGTHS):
        for model_name, ls in zip(MODELS, model_linestyles):
            sub = df[(df["model_name"] == model_name) &
                     (df["input_length"] == seq_len)].sort_values("batch_size")
            label = MODEL_DISPLAY.get(model_name, model_name)
            ax.plot(sub["batch_size"], sub["throughput_tok_s"],
                    marker="o", linestyle=ls, label=label)
        ax.set_title(f"seq={seq_len}")
        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Throughput (tok/s)")
        ax.set_xticks(BATCH_SIZES)
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle("Throughput: GPT-2 vs GPT-Neo 125M", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "baseline_model_comparison.png"), dpi=150)
    plt.close()
    print("Saved: baseline_model_comparison.png")

print(f"\nAll figures → {OUT_DIR}")
