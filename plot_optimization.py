import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "matplotlib", "pandas"])

import pandas as pd
import matplotlib.pyplot as plt
import os

OPT_CSV  = "/workspace/results/optimization_results.csv"
BASE_CSV = "/workspace/results/baseline_results.csv"
OUT_DIR  = "/workspace/results/figures"
os.makedirs(OUT_DIR, exist_ok=True)

df     = pd.read_csv(OPT_CSV)
CONFIG_ORDER  = ["baseline", "fp16", "compile", "fp16+compile"]
CONFIG_COLORS = {
    "baseline":     "#555555",
    "fp16":         "#1f77b4",
    "compile":      "#2ca02c",
    "fp16+compile": "#d62728",
}
INPUT_LENGTHS = sorted(df["input_length"].unique())
BATCH_SIZES   = sorted(df["batch_size"].unique())

# speedup 계산 (baseline 대비)
base = df[df["config"] == "baseline"][["batch_size", "input_length", "throughput_tok_s"]]
base = base.rename(columns={"throughput_tok_s": "base_throughput"})
df   = df.merge(base, on=["batch_size", "input_length"], how="left")
df["speedup"] = df["throughput_tok_s"] / df["base_throughput"]


# ── Figure 1: Throughput vs Batch — config별 비교 (seq=128 고정) ───────────────
fig, axes = plt.subplots(1, len(INPUT_LENGTHS), figsize=(5 * len(INPUT_LENGTHS), 4.5), sharey=False)
if len(INPUT_LENGTHS) == 1:
    axes = [axes]

for ax, seq_len in zip(axes, INPUT_LENGTHS):
    for cfg in CONFIG_ORDER:
        sub = df[(df["config"] == cfg) & (df["input_length"] == seq_len)].sort_values("batch_size")
        if sub.empty:
            continue
        ax.plot(sub["batch_size"], sub["throughput_tok_s"],
                marker="o", label=cfg, color=CONFIG_COLORS[cfg])
    ax.set_title(f"seq={seq_len}")
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Throughput (tok/s)")
    ax.set_xticks(BATCH_SIZES)
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)

fig.suptitle("Throughput: Baseline vs Optimizations", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "opt_throughput_comparison.png"), dpi=150)
plt.close()
print("Saved: opt_throughput_comparison.png")


# ── Figure 2: Speedup vs Batch — config별 (각 seq) ────────────────────────────
fig, axes = plt.subplots(1, len(INPUT_LENGTHS), figsize=(5 * len(INPUT_LENGTHS), 4.5), sharey=True)
if len(INPUT_LENGTHS) == 1:
    axes = [axes]

for ax, seq_len in zip(axes, INPUT_LENGTHS):
    for cfg in CONFIG_ORDER:
        if cfg == "baseline":
            continue
        sub = df[(df["config"] == cfg) & (df["input_length"] == seq_len)].sort_values("batch_size")
        if sub.empty:
            continue
        ax.plot(sub["batch_size"], sub["speedup"],
                marker="o", label=cfg, color=CONFIG_COLORS[cfg])
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.6, label="baseline")
    ax.set_title(f"seq={seq_len}")
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Speedup (×)")
    ax.set_xticks(BATCH_SIZES)
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)

fig.suptitle("Speedup over Baseline", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "opt_speedup.png"), dpi=150)
plt.close()
print("Saved: opt_speedup.png")


# ── Figure 3: Latency 비교 바 차트 (batch=8, 각 seq) ──────────────────────────
batch_fixed = 8
sub = df[df["batch_size"] == batch_fixed]
x   = range(len(INPUT_LENGTHS))
width = 0.2

fig, ax = plt.subplots(figsize=(8, 4.5))
for i, cfg in enumerate(CONFIG_ORDER):
    vals = [sub[(sub["config"] == cfg) & (sub["input_length"] == s)]["avg_latency_s"].values
            for s in INPUT_LENGTHS]
    vals = [v[0] if len(v) > 0 else 0 for v in vals]
    ax.bar([xi + i * width for xi in x], vals, width, label=cfg, color=CONFIG_COLORS[cfg], alpha=0.85)

ax.set_xlabel("Input Length")
ax.set_ylabel("Latency (s)")
ax.set_title(f"Latency Comparison at Batch={batch_fixed}")
ax.set_xticks([xi + width * 1.5 for xi in x])
ax.set_xticklabels([f"seq={s}" for s in INPUT_LENGTHS])
ax.legend()
ax.grid(True, axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "opt_latency_bar.png"), dpi=150)
plt.close()
print("Saved: opt_latency_bar.png")


# ── 요약 테이블 출력 ───────────────────────────────────────────────────────────
print("\n=== Speedup Summary (throughput vs baseline) ===")
pivot = df[df["config"] != "baseline"].pivot_table(
    index=["config"], columns=["input_length", "batch_size"],
    values="speedup", aggfunc="mean"
)
print(pivot.round(2).to_string())

print(f"\nAll figures → {OUT_DIR}")
