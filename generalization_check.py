"""
Generalization Check — regime classification across diverse architectures.

Verifies that the three-regime bottleneck characterization and the adaptive
policy speedups are *not* specific to the GPT-2 architecture family.

Models tested:
  GPT-2 family   : gpt2 (124M), gpt2-large (762M)          [baseline, already done]
  GPT-Neo family : EleutherAI/gpt-neo-125m                  [baseline, already done]
  OPT family     : facebook/opt-125m, facebook/opt-350m     [new: different arch]
  BLOOM family   : bigscience/bloom-560m                    [new: different arch + tokenizer]

For each model × representative (batch, seq) point, we measure:
  - Baseline latency, throughput, GPU utilization
  - Regime classification (from profiling)
  - Speedup from adaptive policy config (compile or fp16)

This confirms:
  1. Regime boundaries (GPU util thresholds) are consistent across architectures.
  2. The adaptive policy delivers similar speedup magnitude across model families.
  3. The findings are not artefacts of GPT-2's specific kernel implementation.

Output: results/generalization_results.csv
"""

import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet",
                       "transformers==4.44.2", "nvidia-ml-py", "accelerate", "pandas", "tqdm"])

import time, csv, math, os, warnings, statistics
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
import pynvml
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore")

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Model matrix (architecture diversity) ─────────────────────────────────────
MODELS = [
    # GPT-2 family (results already exist; re-run for fair comparison)
    {"name": "gpt2",          "model_id": "gpt2",                       "family": "GPT-2"},
    {"name": "gpt2-large",    "model_id": "gpt2-large",                 "family": "GPT-2"},
    # GPT-Neo (rotary PE, local attention in some layers)
    {"name": "gpt-neo-125m",  "model_id": "EleutherAI/gpt-neo-125m",    "family": "GPT-Neo"},
    # OPT (learned PE, different LN placement, SwiGLU-free)
    {"name": "opt-125m",      "model_id": "facebook/opt-125m",           "family": "OPT"},
    {"name": "opt-350m",      "model_id": "facebook/opt-350m",           "family": "OPT"},
    # BLOOM (ALiBi PE, multi-lingual, different attention implementation)
    {"name": "bloom-560m",    "model_id": "bigscience/bloom-560m",       "family": "BLOOM"},
]

# Representative points: one per regime (low-util, kernel-overhead, memory-bound)
REGIME_POINTS = [
    (1,  128),   # low-utilization
    (8,  128),   # kernel-overhead-bound
    (16, 512),   # memory-bound
]

ADAPTIVE_CONFIG = {
    "low-utilization":       {"fp16": False, "compile": True},
    "kernel-overhead-bound": {"fp16": False, "compile": True},
    "memory-bound":          {"fp16": True,  "compile": False},
}

_GPU_UTIL_LOW          = 35
_GPU_UTIL_MEMORY_BOUND = 90

def classify_regime(batch_size, seq_length, gpu_util_pct):
    if gpu_util_pct < _GPU_UTIL_LOW:
        return "low-utilization"
    elif gpu_util_pct >= _GPU_UTIL_MEMORY_BOUND and seq_length >= 256:
        return "memory-bound"
    return "kernel-overhead-bound"

MAX_NEW_TOKENS = 50
WARMUP_RUNS    = 10
COMPILE_WARMUP = 30
MEASURE_RUNS   = 20
_T95           = 2.093

pynvml.nvmlInit()
_visible  = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
_gpu_idx  = int(_visible.split(",")[0]) if _visible.strip() else 0
gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(_gpu_idx)


def get_gpu_stats():
    mem  = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
    util = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle)
    return {"gpu_util_pct": util.gpu, "gpu_mem_used_mb": mem.used // (1024**2)}


def make_input(tokenizer, batch_size, input_length, device):
    prompt = "The quick brown fox jumps over the lazy dog"
    ids    = tokenizer.encode(prompt, return_tensors="pt")[0]
    ids    = ids.repeat((input_length // len(ids)) + 1)[:input_length]
    input_ids      = ids.unsqueeze(0).repeat(batch_size, 1).to(device)
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask


def run_timed(model, tokenizer, batch_size, input_length, device, compiled):
    input_ids, attention_mask = make_input(tokenizer, batch_size, input_length, device)
    gen_kwargs = dict(
        max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
        attention_mask=attention_mask, pad_token_id=tokenizer.eos_token_id,
    )
    n_warmup = COMPILE_WARMUP if compiled else WARMUP_RUNS
    try:
        for _ in range(n_warmup):
            with torch.no_grad():
                model.generate(input_ids, **gen_kwargs)
        torch.cuda.synchronize()

        latencies = []
        for _ in range(MEASURE_RUNS):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                model.generate(input_ids, **gen_kwargs)
            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - t0)

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None

    avg = sum(latencies) / len(latencies)
    std = statistics.stdev(latencies)
    thr = (MAX_NEW_TOKENS * batch_size) / avg
    return {
        "avg_latency_s":    round(avg, 4),
        "std_latency_s":    round(std, 5),
        "ci95_half_ms":     round(_T95 * std * 1000 / math.sqrt(MEASURE_RUNS), 3),
        "ms_per_token":     round(avg * 1000 / MAX_NEW_TOKENS, 3),
        "throughput_tok_s": round(thr, 2),
        **get_gpu_stats(),
    }


def load_model(model_cfg, opt_cfg, device):
    model = AutoModelForCausalLM.from_pretrained(model_cfg["model_id"])
    if opt_cfg.get("fp16"):
        model = model.half()
    model = model.to(device).eval()
    if opt_cfg.get("compile"):
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "N/A"
    print(f"Device: {device} | GPU: {gpu_name}\n")

    out_path   = os.path.join(RESULTS_DIR, "generalization_results.csv")
    fieldnames = [
        "model_name", "model_family", "param_count_m", "weight_mb",
        "batch_size", "input_length",
        "baseline_throughput_tok_s", "baseline_gpu_util_pct",
        "regime", "adaptive_config",
        "adaptive_throughput_tok_s", "speedup", "speedup_ci95",
        "avg_latency_s", "std_latency_s", "ci95_half_ms",
    ]

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for model_cfg in MODELS:
            print(f"\n{'='*60}")
            print(f"Model: {model_cfg['name']}  [{model_cfg['family']}]")

            tokenizer = AutoTokenizer.from_pretrained(model_cfg["model_id"])
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Load baseline to get model info and in-session measurements
            # BLOOM uses ALiBi (no fixed positional embedding limit) → fallback to 2048
            base_model   = load_model(model_cfg, {}, device)
            max_pos      = getattr(base_model.config, "max_position_embeddings", 2048)
            param_count  = sum(p.numel() for p in base_model.parameters())
            weight_mb    = sum(p.numel() * p.element_size() for p in base_model.parameters()) / (1024**2)
            print(f"  Params: {param_count/1e6:.0f}M  Weights: {weight_mb:.0f} MB  "
                  f"Max context: {max_pos}")

            for bs, seq in REGIME_POINTS:
                if seq + MAX_NEW_TOKENS > max_pos:
                    print(f"  batch={bs} seq={seq}: exceeds max context — skipped")
                    continue

                # Baseline measurement
                print(f"\n  batch={bs:2d}  seq={seq:3d} — baseline ...", end=" ", flush=True)
                base_r = run_timed(base_model, tokenizer, bs, seq, device, compiled=False)
                if base_r is None:
                    print("OOM"); continue
                regime = classify_regime(bs, seq, base_r["gpu_util_pct"])
                print(f"thr={base_r['throughput_tok_s']:.1f}  GPU={base_r['gpu_util_pct']}%  "
                      f"→ [{regime}]")

                # Adaptive-policy measurement
                opt_cfg    = ADAPTIVE_CONFIG[regime]
                cfg_label  = ("compile" if opt_cfg["compile"] else
                              "fp16"    if opt_cfg["fp16"]    else "baseline")
                del base_model; torch.cuda.empty_cache()

                opt_model = load_model(model_cfg, opt_cfg, device)
                print(f"  batch={bs:2d}  seq={seq:3d} — {cfg_label:8s} ...", end=" ", flush=True)
                opt_r = run_timed(opt_model, tokenizer, bs, seq, device,
                                  compiled=opt_cfg.get("compile", False))
                del opt_model; torch.cuda.empty_cache()

                if opt_r is None:
                    print("OOM")
                    # Reload baseline for next iteration
                    base_model = load_model(model_cfg, {}, device)
                    continue

                sp   = round(opt_r["throughput_tok_s"] / base_r["throughput_tok_s"], 4)
                cv_o = opt_r["std_latency_s"] / opt_r["avg_latency_s"]
                cv_b = base_r["std_latency_s"] / base_r["avg_latency_s"]
                ci   = round(sp * _T95 * math.sqrt(cv_o**2 + cv_b**2) / math.sqrt(MEASURE_RUNS), 4)
                print(f"thr={opt_r['throughput_tok_s']:.1f}  speedup={sp:.3f}±{ci:.4f}")

                writer.writerow({
                    "model_name":               model_cfg["name"],
                    "model_family":             model_cfg["family"],
                    "param_count_m":            round(param_count / 1e6, 1),
                    "weight_mb":                round(weight_mb, 1),
                    "batch_size":               bs,
                    "input_length":             seq,
                    "baseline_throughput_tok_s": base_r["throughput_tok_s"],
                    "baseline_gpu_util_pct":    base_r["gpu_util_pct"],
                    "regime":                   regime,
                    "adaptive_config":          cfg_label,
                    "adaptive_throughput_tok_s": opt_r["throughput_tok_s"],
                    "speedup":                  sp,
                    "speedup_ci95":             ci,
                    "avg_latency_s":            opt_r["avg_latency_s"],
                    "std_latency_s":            opt_r["std_latency_s"],
                    "ci95_half_ms":             opt_r["ci95_half_ms"],
                })
                f.flush()

                # Reload baseline for next point
                base_model = load_model(model_cfg, {}, device)

            del base_model; torch.cuda.empty_cache()

    print(f"\nGeneralization results saved → {out_path}")
    pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()
