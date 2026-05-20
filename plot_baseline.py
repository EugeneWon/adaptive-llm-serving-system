import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "matplotlib", "pandas"])

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

CSV_PATH    = "/workspace/results/baseline_results.csv"
OUT_DIR     = "/workspace/results/figures"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)

INPUT_LENGTHS = sorted(df["input_length"].unique())
BATCH_SIZES   = sorted(df["batch_size"].unique())
COLORS        = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

# ── regime 분류 ────────────────────────────────────────────────────────────────
# scaling_eff: 실제 throughput / (batch=1 throughput * batch_size)
# 1에 가까울수록 linear scaling (low-util or overhead dominant)
# 낮아질수록 memory-bound
ref = df[df["batch_size"] == 1].set_index("input_length")["throughput_tok_s"]
df["scaling_eff"] = df.apply(
    lambda r: r["throughput_tok_s"] / (ref[r["input_length"]] * r["batch_size"]), axis=1
)

def classify_regime(row):
    if row["gpu_util_pct"] < 40:
        return "low-utilization"
    elif row["scaling_eff"] < 0.75:
        return "memory-bound"
    else:
        return "compute-scaling"

df["regime"] = df.apply(classify_regime, axis=1)

REGIME_COLORS = {
    "low-utilization":  "#aec6e8",
    "compute-scaling":  "#98df8a",
    "memory-bound":     "#ffbb78",
}

print("\n=== Regime Classification ===")
print(df[["batch_size", "input_length", "gpu_util_pct", "scaling_eff", "regime"]].to_string(index=False))


# ── Figure 1: Latency vs Batch Size ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))
for i, seq_len in enumerate(INPUT_LENGTHS):
    sub = df[df["input_length"] == seq_len].sort_values("batch_size")
    ax.plot(sub["batch_size"], sub["avg_latency_s"],
            marker="o", label=f"seq={seq_len}", color=COLORS[i])
    ax.fill_between(sub["batch_size"],
                    sub["p50_latency_s"], sub["p95_latency_s"],
                    alpha=0.12, color=COLORS[i])

ax.set_xlabel("Batch Size")
ax.set_ylabel("Latency (s)")
ax.set_title("Latency vs Batch Size (GPT-2, 50 new tokens)")
ax.set_xticks(BATCH_SIZES)
ax.legend(title="Input Length")
ax.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "baseline_latency_vs_batch.png"), dpi=150)
plt.close()
print("Saved: latency_vs_batch.png")


# ── Figure 2: Throughput vs Batch Size ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))
for i, seq_len in enumerate(INPUT_LENGTHS):
    sub = df[df["input_length"] == seq_len].sort_values("batch_size")
    ax.plot(sub["batch_size"], sub["throughput_tok_s"],
            marker="o", label=f"seq={seq_len}", color=COLORS[i])

# ideal linear reference (from batch=1, seq=32)
ref_val = df[(df["batch_size"] == 1) & (df["input_length"] == 32)]["throughput_tok_s"].values[0]
ideal = [ref_val * b for b in BATCH_SIZES]
ax.plot(BATCH_SIZES, ideal, "k--", linewidth=1, alpha=0.5, label="ideal (linear)")

ax.set_xlabel("Batch Size")
ax.set_ylabel("Throughput (tok/s)")
ax.set_title("Throughput vs Batch Size (GPT-2)")
ax.set_xticks(BATCH_SIZES)
ax.legend(title="Input Length")
ax.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "baseline_throughput_vs_batch.png"), dpi=150)
plt.close()
print("Saved: throughput_vs_batch.png")


# ── Figure 3: GPU Util + Regime Map ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))
for i, seq_len in enumerate(INPUT_LENGTHS):
    sub = df[df["input_length"] == seq_len].sort_values("batch_size")

    # regime background per segment
    for _, row in sub.iterrows():
        ax.axvspan(row["batch_size"] - 0.35, row["batch_size"] + 0.35,
                   alpha=0.25, color=REGIME_COLORS[row["regime"]], zorder=0)

    ax.plot(sub["batch_size"], sub["gpu_util_pct"],
            marker="o", label=f"seq={seq_len}", color=COLORS[i])

ax.axhline(40, color="gray", linestyle=":", linewidth=1, label="util=40% threshold")
ax.set_xlabel("Batch Size")
ax.set_ylabel("GPU Utilization (%)")
ax.set_title("GPU Util vs Batch Size — Regime Map")
ax.set_xticks(BATCH_SIZES)
ax.set_ylim(0, 100)

patches = [mpatches.Patch(color=v, alpha=0.5, label=k) for k, v in REGIME_COLORS.items()]
legend1 = ax.legend(handles=patches, loc="upper left", title="Regime", fontsize=8)
ax.legend(title="Input Length", loc="lower right")
ax.add_artist(legend1)
ax.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "baseline_util_regime_map.png"), dpi=150)
plt.close()
print("Saved: util_regime_map.png")


# ── Figure 4: Scaling Efficiency ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))
for i, seq_len in enumerate(INPUT_LENGTHS):
    sub = df[df["input_length"] == seq_len].sort_values("batch_size")
    ax.plot(sub["batch_size"], sub["scaling_eff"] * 100,
            marker="o", label=f"seq={seq_len}", color=COLORS[i])

ax.axhline(100, color="k", linestyle="--", linewidth=1, alpha=0.4, label="ideal")
ax.axhline(75,  color="orange", linestyle=":", linewidth=1, alpha=0.7, label="memory-bound boundary (75%)")
ax.set_xlabel("Batch Size")
ax.set_ylabel("Scaling Efficiency (%)")
ax.set_title("Throughput Scaling Efficiency vs Batch Size")
ax.set_xticks(BATCH_SIZES)
ax.set_ylim(0, 120)
ax.legend(title="Input Length")
ax.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "baseline_scaling_efficiency.png"), dpi=150)
plt.close()
print("Saved: scaling_efficiency.png")

print(f"\nAll figures → {OUT_DIR}")
print("\n=== Regime Summary ===")
print(df.groupby("regime")[["batch_size", "input_length"]].apply(
    lambda x: x.values.tolist()
).to_string())
