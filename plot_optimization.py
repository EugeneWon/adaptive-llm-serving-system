import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "matplotlib", "pandas"])

import pandas as pd
import matplotlib.pyplot as plt
import os

_BASE   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
OPT_CSV = os.path.join(_BASE, "optimization_results.csv")
OUT_DIR = os.path.join(_BASE, "figures")
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(OPT_CSV)

# backward-compat
if "model_name" not in df.columns:
    df["model_name"] = "gpt2"

MODELS        = sorted(df["model_name"].unique())
CONFIG_ORDER  = ["baseline", "fp16", "compile", "fp16+compile"]
CONFIG_COLORS = {
    "baseline":     "#555555",
    "fp16":         "#1f77b4",
    "compile":      "#2ca02c",
    "fp16+compile": "#d62728",
}
INPUT_LENGTHS = sorted(df["input_length"].unique())
BATCH_SIZES   = sorted(df["batch_size"].unique())

MODEL_DISPLAY = {
    "gpt2":         "GPT-2 (124M)",
    "gpt2-large":   "GPT-2 Large (762M)",
    "gpt-neo-125m": "GPT-Neo (125M)",
}

# speedup relative to per-model baseline
base = df[df["config"] == "baseline"][["model_name", "batch_size", "input_length", "throughput_tok_s"]]
base = base.rename(columns={"throughput_tok_s": "base_throughput"})
df   = df.merge(base, on=["model_name", "batch_size", "input_length"], how="left")
df["speedup"] = df["throughput_tok_s"] / df["base_throughput"]


# ── Figure 1: Throughput comparison — rows=models, cols=seq ──────────────────
fig, axes = plt.subplots(len(MODELS), len(INPUT_LENGTHS),
                         figsize=(5 * len(INPUT_LENGTHS), 4 * len(MODELS)),
                         sharey=False, squeeze=False)

for row_i, model_name in enumerate(MODELS):
    mdf = df[df["model_name"] == model_name]
    for col_i, seq_len in enumerate(INPUT_LENGTHS):
        ax = axes[row_i][col_i]
        for cfg in CONFIG_ORDER:
            sub = mdf[(mdf["config"] == cfg) &
                      (mdf["input_length"] == seq_len)].sort_values("batch_size")
            if sub.empty:
                continue
            ax.plot(sub["batch_size"], sub["throughput_tok_s"],
                    marker="o", label=cfg, color=CONFIG_COLORS[cfg])
        ax.set_title(f"{MODEL_DISPLAY.get(model_name, model_name)}\nseq={seq_len}", fontsize=9)
        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Throughput (tok/s)")
        ax.set_xticks(BATCH_SIZES)
        ax.legend(fontsize=7)
        ax.grid(True, linestyle="--", alpha=0.4)

fig.suptitle("Throughput: Baseline vs Optimizations", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "opt_throughput_comparison.png"), dpi=150)
plt.close()
print("Saved: opt_throughput_comparison.png")


# ── Figure 2: Speedup — rows=models, cols=seq ─────────────────────────────────
fig, axes = plt.subplots(len(MODELS), len(INPUT_LENGTHS),
                         figsize=(5 * len(INPUT_LENGTHS), 4 * len(MODELS)),
                         sharey=True, squeeze=False)

for row_i, model_name in enumerate(MODELS):
    mdf = df[df["model_name"] == model_name]
    for col_i, seq_len in enumerate(INPUT_LENGTHS):
        ax = axes[row_i][col_i]
        for cfg in CONFIG_ORDER:
            if cfg == "baseline":
                continue
            sub = mdf[(mdf["config"] == cfg) &
                      (mdf["input_length"] == seq_len)].sort_values("batch_size")
            if sub.empty:
                continue
            ax.plot(sub["batch_size"], sub["speedup"],
                    marker="o", label=cfg, color=CONFIG_COLORS[cfg])
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.6, label="baseline")
        ax.set_title(f"{MODEL_DISPLAY.get(model_name, model_name)}\nseq={seq_len}", fontsize=9)
        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Speedup (×)")
        ax.set_xticks(BATCH_SIZES)
        ax.legend(fontsize=7)
        ax.grid(True, linestyle="--", alpha=0.4)

fig.suptitle("Speedup over Baseline", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "opt_speedup.png"), dpi=150)
plt.close()
print("Saved: opt_speedup.png")


# ── Figure 3: Latency bar — batch=8, per model ───────────────────────────────
batch_fixed = 8
fig, axes = plt.subplots(1, len(MODELS), figsize=(8 * len(MODELS), 4.5),
                         sharey=False, squeeze=False)
axes = axes[0]

for ax, model_name in zip(axes, MODELS):
    mdf = df[(df["model_name"] == model_name) & (df["batch_size"] == batch_fixed)]
    x   = range(len(INPUT_LENGTHS))
    width = 0.2
    for i, cfg in enumerate(CONFIG_ORDER):
        vals = [mdf[(mdf["config"] == cfg) &
                    (mdf["input_length"] == s)]["avg_latency_s"].values for s in INPUT_LENGTHS]
        vals = [v[0] if len(v) > 0 else 0 for v in vals]
        ax.bar([xi + i * width for xi in x], vals, width,
               label=cfg, color=CONFIG_COLORS[cfg], alpha=0.85)
    ax.set_title(f"{MODEL_DISPLAY.get(model_name, model_name)}\nBatch={batch_fixed}")
    ax.set_xlabel("Input Length")
    ax.set_ylabel("Latency (s)")
    ax.set_xticks([xi + width * 1.5 for xi in x])
    ax.set_xticklabels([f"seq={s}" for s in INPUT_LENGTHS])
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

fig.suptitle(f"Latency Comparison at Batch={batch_fixed}", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "opt_latency_bar.png"), dpi=150)
plt.close()
print("Saved: opt_latency_bar.png")


# ── Figure 4: Memory Bandwidth vs Batch Size (FP32 vs FP16) ─────────────────
# Shows how FP16 cuts weight bytes by half, raising effective bandwidth utilization.
# Only plotted if the new data-movement columns are present.
if "estimated_bandwidth_GBs" in df.columns and "weight_bytes_mb" in df.columns:
    configs_bw = ["baseline", "fp16"]
    fig, axes = plt.subplots(len(MODELS), len(INPUT_LENGTHS),
                             figsize=(5 * len(INPUT_LENGTHS), 4 * len(MODELS)),
                             sharey=False, squeeze=False)
    for row_i, model_name in enumerate(MODELS):
        mdf = df[df["model_name"] == model_name]
        for col_i, seq_len in enumerate(INPUT_LENGTHS):
            ax = axes[row_i][col_i]
            for cfg in configs_bw:
                sub = mdf[(mdf["config"] == cfg) &
                          (mdf["input_length"] == seq_len)].sort_values("batch_size")
                if sub.empty:
                    continue
                label = f"{cfg} ({sub['weight_bytes_mb'].iloc[0]:.0f} MB)"
                ax.plot(sub["batch_size"], sub["estimated_bandwidth_GBs"],
                        marker="o", label=label, color=CONFIG_COLORS[cfg])
            ax.set_title(f"{MODEL_DISPLAY.get(model_name, model_name)}\nseq={seq_len}", fontsize=9)
            ax.set_xlabel("Batch Size")
            ax.set_ylabel("Est. Bandwidth (GB/s)")
            ax.set_xticks(BATCH_SIZES)
            ax.legend(fontsize=7)
            ax.grid(True, linestyle="--", alpha=0.4)
    # Figure 4 axis labels and title: clarify what this metric actually measures.
    # estimated_bandwidth_GBs = weight_bytes / latency_per_token.
    # It is a per-token weight-loading rate, NOT total HBM throughput.
    # It intentionally drops at large batch/seq because attention FLOPs grow
    # (O(seq²×batch)), increasing latency_per_token even as GPU util rises.
    # That drop signals attention-compute growth — a useful bottleneck indicator.
    for ax_row in axes:
        for ax in ax_row:
            ax.set_ylabel("Weight-Loading Rate (GB/s per token)")
    fig.suptitle(
        "Decode Weight-Loading Rate: FP32 vs FP16\n"
        "Drops at large batch/seq as attention compute grows (see Arithmetic Intensity plot)",
        fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "opt_bandwidth.png"), dpi=150)
    plt.close()
    print("Saved: opt_bandwidth.png")

    # Figure 5: Arithmetic Intensity vs Batch Size (decode phase)
    # All values << GPU ridge point, confirming decode is memory-bound throughout.
    # Compare with prefill_results.csv where AI >> ridge point (compute-bound).
    fig, axes = plt.subplots(len(MODELS), 1,
                             figsize=(7, 4 * len(MODELS)),
                             sharey=False, squeeze=False)
    for row_i, model_name in enumerate(MODELS):
        ax  = axes[row_i][0]
        mdf = df[(df["model_name"] == model_name) & (df["config"] == "baseline")]
        for seq_len in INPUT_LENGTHS:
            sub = mdf[mdf["input_length"] == seq_len].sort_values("batch_size")
            if sub.empty:
                continue
            ax.plot(sub["batch_size"], sub["arithmetic_intensity"],
                    marker="o", label=f"seq={seq_len}")
        ax.axhline(1.0, color="red", linestyle=":", linewidth=1.2,
                   label="AI=1.0 FLOPs/byte (reference)")
        ax.set_title(MODEL_DISPLAY.get(model_name, model_name))
        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Arithmetic Intensity (FLOPs/byte)")
        ax.set_xticks(BATCH_SIZES)
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)
    fig.suptitle(
        "Decode Arithmetic Intensity vs Batch Size\n"
        "All values << GPU ridge point (35–300 FLOPs/byte) → decode is memory-bound",
        fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "opt_arithmetic_intensity.png"), dpi=150)
    plt.close()
    print("Saved: opt_arithmetic_intensity.png")


# ── Summary ──────────────────────────────────────────────────────────────────
print("\n=== Speedup Summary (throughput vs baseline) ===")
for model_name in MODELS:
    print(f"\n  [{MODEL_DISPLAY.get(model_name, model_name)}]")
    mdf = df[df["model_name"] == model_name]
    pivot = mdf[mdf["config"] != "baseline"].pivot_table(
        index="config", columns=["input_length", "batch_size"],
        values="speedup", aggfunc="mean"
    )
    print(pivot.round(3).to_string())

print(f"\nAll figures → {OUT_DIR}")
