import subprocess
import sys

def _ensure_packages():
    pkgs = [
        "transformers==4.44.2",
        "nvidia-ml-py",
        "accelerate",
        "pandas",
        "tqdm",
    ]
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet", *pkgs]
    )

_ensure_packages()

import time
import csv
import math
import os
import warnings
import statistics
import torch
import pynvml
from torch.profiler import profile, ProfilerActivity
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore")

RESULTS_DIR = "/workspace/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

MODELS = [
    {"name": "gpt2",         "model_id": "gpt2"},
    {"name": "gpt2-large",   "model_id": "gpt2-large"},
    {"name": "gpt-neo-125m", "model_id": "EleutherAI/gpt-neo-125m"},
]

BATCH_SIZES    = [1, 2, 4, 8, 16, 32]
INPUT_LENGTHS  = [32, 64, 128, 256, 512]
MAX_NEW_TOKENS = 50
WARMUP_RUNS    = 10
MEASURE_RUNS   = 20
# t(0.975, df=MEASURE_RUNS-1=19) for 95% CI
_T95 = 2.093

pynvml.nvmlInit()
gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)


def get_gpu_stats():
    mem  = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
    util = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle)
    return {
        "gpu_util_pct":     util.gpu,
        "gpu_mem_used_mb":  mem.used  // (1024 ** 2),
        "gpu_mem_total_mb": mem.total // (1024 ** 2),
    }


def get_model_weight_bytes(model):
    return sum(p.numel() * p.element_size() for p in model.parameters())


def compute_data_movement_metrics(weight_bytes, param_count, avg_latency, batch_size):
    """
    Compute decode-phase data movement metrics.

    estimated_bandwidth_GBs  (weight-loading rate per decode token)
        = weight_bytes / latency_per_token
        Measures how fast model weights are consumed per generated token.
        This is a DECODE bottleneck indicator:
          - High value  → weights load fast → memory-bound decode (HBM near saturation)
          - Low value   → latency_per_token is long → attention compute or kernel
                          overhead is growing relative to pure weight-streaming
        NOTE: This metric intentionally DECREASES at large batch × long seq because
        attention FLOPs grow as O(seq²×batch), increasing latency_per_token even
        though GPU utilisation rises.  That drop signals the onset of compute-bound
        behaviour in the attention layers and is not a measurement artefact.
        Use profiler_bw_proxy_GBs (from torch.profiler) for total HBM throughput.

    arithmetic_intensity  (FLOPs / byte, decode-phase roofline proxy)
        = (2 × params × batch) / weight_bytes
        All values well below any GPU ridge point (typically 35–300 FLOPs/byte),
        confirming decode is memory-bound for the model sizes used here.
    """
    latency_per_token = avg_latency / MAX_NEW_TOKENS
    estimated_bw_GBs = weight_bytes / latency_per_token / 1e9
    flops_per_token  = 2 * param_count * batch_size
    arithmetic_intensity = flops_per_token / weight_bytes
    return {
        "estimated_bandwidth_GBs":   round(estimated_bw_GBs, 2),
        "arithmetic_intensity":      round(arithmetic_intensity, 4),
    }


def profile_data_movement(model, tokenizer, batch_size, input_length, device):
    """
    One-shot torch.profiler run. Returns FLOPs, CUDA time, and memory allocation
    as reported by the PyTorch profiler (kernel-level breakdown).
    """
    input_ids, attention_mask = make_input(tokenizer, batch_size, input_length, device)
    gen_kwargs = dict(
        max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
        attention_mask=attention_mask, pad_token_id=tokenizer.eos_token_id,
    )
    with torch.no_grad():
        model.generate(input_ids, **gen_kwargs)
    torch.cuda.synchronize()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        profile_memory=True,
        with_flops=True,
        record_shapes=False,
    ) as prof:
        with torch.no_grad():
            model.generate(input_ids, **gen_kwargs)

    avgs = prof.key_averages()
    total_flops      = sum(e.flops for e in avgs if e.flops > 0)
    cuda_time_ms     = sum(e.cuda_time_total for e in avgs) / 1000
    mem_alloc_mb     = sum(max(e.self_cuda_memory_usage, 0) for e in avgs) / (1024 ** 2)
    # profiler-derived bandwidth: bytes allocated / CUDA time (proxy for memory traffic)
    profiler_bw_GBs  = (mem_alloc_mb * 1024 ** 2) / (cuda_time_ms * 1e-3) / 1e9 if cuda_time_ms > 0 else 0

    return {
        "profiler_flops":        total_flops,
        "profiler_cuda_time_ms": round(cuda_time_ms, 2),
        "profiler_mem_alloc_mb": round(mem_alloc_mb, 2),
        "profiler_bw_proxy_GBs": round(profiler_bw_GBs, 2),
    }


def compute_prefill_arithmetic_intensity(model, weight_bytes, param_count,
                                         batch_size, input_length):
    """
    Compute arithmetic intensity for the prefill (prompt processing) phase.

    Unlike decode (memory-bound, AI ≈ 0.5–32 FLOPs/byte), prefill is
    compute-bound because:
      1. All input tokens are processed in a single parallel forward pass.
      2. Linear-layer FLOPs scale as O(seq × params × batch).
      3. Attention FLOPs scale as O(seq² × d_model × layers × batch),
         dominant at long sequences.
      4. Together they push AI >> GPU ridge point (typically 35–300 FLOPs/byte).
    """
    num_layers  = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    # Linear layers: ~2 × seq × params × batch (matmul FLOPs, rough but standard)
    linear_flops = 2 * input_length * param_count * batch_size
    # Attention QK + AV: 4 × batch × seq² × hidden_size per layer (multi-head attention)
    attn_flops = 4 * batch_size * (input_length ** 2) * hidden_size * num_layers
    total_flops = linear_flops + attn_flops
    # Bytes: weight reads + KV cache writes (FP32 assumed)
    kv_bytes    = 2 * batch_size * num_layers * input_length * hidden_size * 4
    total_bytes = weight_bytes + kv_bytes
    return round(total_flops / total_bytes, 1)


def run_prefill(model, tokenizer, batch_size, input_length, device,
                weight_bytes, param_count):
    """
    Measure prefill (prompt processing) latency: single forward pass over
    input_length tokens, no autoregressive generation.

    Prefill is compute-bound (AI >> GPU ridge point), in contrast to decode
    which is memory-bound (AI << ridge point).  Including both phases allows
    the full three-regime roofline narrative to be demonstrated without
    requiring a larger model.
    """
    if input_length > model.config.max_position_embeddings:
        return None
    input_ids, attention_mask = make_input(tokenizer, batch_size, input_length, device)
    try:
        for _ in range(WARMUP_RUNS):
            with torch.no_grad():
                model(input_ids, attention_mask=attention_mask)
        torch.cuda.synchronize()

        latencies = []
        for _ in range(MEASURE_RUNS):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                model(input_ids, attention_mask=attention_mask)
            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - t0)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None

    avg_lat     = sum(latencies) / len(latencies)
    sorted_lat  = sorted(latencies)
    prefill_ai  = compute_prefill_arithmetic_intensity(
        model, weight_bytes, param_count, batch_size, input_length)
    return {
        "batch_size":          batch_size,
        "input_length":        input_length,
        "avg_latency_ms":      round(avg_lat * 1000, 3),
        "p50_latency_ms":      round(sorted_lat[(len(sorted_lat) - 1) // 2] * 1000, 3),
        "p95_latency_ms":      round(sorted_lat[int(len(sorted_lat) * 0.95) - 1] * 1000, 3),
        "std_latency_ms":      round(statistics.stdev(latencies) * 1000, 4),
        "tokens_per_s":        round(input_length * batch_size / avg_lat, 1),
        "prefill_ai_flops_byte": prefill_ai,
        **get_gpu_stats(),
    }


def make_input(tokenizer, batch_size, input_length, device):
    prompt = "The quick brown fox jumps over the lazy dog"
    ids    = tokenizer.encode(prompt, return_tensors="pt")[0]
    ids    = ids.repeat((input_length // len(ids)) + 1)[:input_length]
    input_ids      = ids.unsqueeze(0).repeat(batch_size, 1).to(device)
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask


def run_experiment(model, tokenizer, batch_size, input_length, device, weight_bytes, param_count):
    input_ids, attention_mask = make_input(tokenizer, batch_size, input_length, device)
    gen_kwargs = dict(
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        attention_mask=attention_mask,
        pad_token_id=tokenizer.eos_token_id,
    )

    for _ in range(WARMUP_RUNS):
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

    avg_latency = sum(latencies) / len(latencies)
    throughput  = (MAX_NEW_TOKENS * batch_size) / avg_latency

    return {
        "batch_size":       batch_size,
        "input_length":     input_length,
        "max_new_tokens":   MAX_NEW_TOKENS,
        "avg_latency_s":    round(avg_latency, 4),
        "p50_latency_s":    round(sorted(latencies)[(len(latencies) - 1) // 2], 4),
        "p95_latency_s":    round(sorted(latencies)[int(len(latencies) * 0.95) - 1], 4),
        "std_latency_s":    round(statistics.stdev(latencies), 5),
        "ci95_half_ms":     round(_T95 * statistics.stdev(latencies) * 1000 / math.sqrt(MEASURE_RUNS), 3),
        "ms_per_token":     round(avg_latency * 1000 / MAX_NEW_TOKENS, 3),
        "throughput_tok_s": round(throughput, 2),
        **get_gpu_stats(),
        **compute_data_movement_metrics(weight_bytes, param_count, avg_latency, batch_size),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    out_path      = os.path.join(RESULTS_DIR, "baseline_results.csv")
    profiler_path = os.path.join(RESULTS_DIR, "data_movement_profile.csv")
    prefill_path  = os.path.join(RESULTS_DIR, "prefill_results.csv")

    fieldnames = [
        "model_name",
        "batch_size", "input_length", "max_new_tokens",
        "avg_latency_s", "p50_latency_s", "p95_latency_s", "std_latency_s", "ci95_half_ms", "ms_per_token",
        "throughput_tok_s",
        "gpu_util_pct", "gpu_mem_used_mb", "gpu_mem_total_mb",
        "estimated_bandwidth_GBs", "arithmetic_intensity",
    ]
    profiler_fieldnames = [
        "model_name", "batch_size", "input_length",
        "weight_bytes_mb", "param_count_m",
        "profiler_flops", "profiler_cuda_time_ms",
        "profiler_mem_alloc_mb", "profiler_bw_proxy_GBs",
    ]
    prefill_fieldnames = [
        "model_name", "batch_size", "input_length",
        "avg_latency_ms", "p50_latency_ms", "p95_latency_ms", "std_latency_ms",
        "tokens_per_s", "prefill_ai_flops_byte",
        "gpu_util_pct", "gpu_mem_used_mb", "gpu_mem_total_mb",
        "weight_bytes_mb", "param_count_m",
    ]

    # Representative (batch, seq) points for profiler pass — one per regime
    PROFILER_POINTS = [(1, 256), (8, 256), (32, 256)]
    # Prefill points: chosen to show how compute-bound AI scales with seq and batch
    PREFILL_BATCH_SIZES  = [1, 4, 8, 16, 32]
    PREFILL_INPUT_LENGTHS = [128, 256, 512]

    with open(out_path, "w", newline="") as f, \
         open(profiler_path, "w", newline="") as pf, \
         open(prefill_path, "w", newline="") as pfil:

        writer  = csv.DictWriter(f,    fieldnames=fieldnames)
        pwriter = csv.DictWriter(pf,   fieldnames=profiler_fieldnames)
        prwriter = csv.DictWriter(pfil, fieldnames=prefill_fieldnames)
        writer.writeheader()
        pwriter.writeheader()
        prwriter.writeheader()

        for model_cfg in MODELS:
            print(f"\n{'='*55}")
            print(f"Loading {model_cfg['name']} ({model_cfg['model_id']})...")
            tokenizer = AutoTokenizer.from_pretrained(model_cfg["model_id"])
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model   = AutoModelForCausalLM.from_pretrained(model_cfg["model_id"]).to(device)
            model.eval()
            max_pos     = model.config.max_position_embeddings
            param_count = sum(p.numel() for p in model.parameters())
            weight_bytes = get_model_weight_bytes(model)
            print(f"Parameters: {param_count / 1e6:.0f}M  "
                  f"Weights: {weight_bytes / (1024**2):.0f} MB  "
                  f"Max context: {max_pos}")

            # ── torch.profiler pass (decode, representative regime points) ────
            print("  [profiler] running data movement analysis...")
            for bs, seq_len in PROFILER_POINTS:
                if seq_len + MAX_NEW_TOKENS > max_pos:
                    continue
                try:
                    p_stats = profile_data_movement(model, tokenizer, bs, seq_len, device)
                    pwriter.writerow({
                        "model_name":       model_cfg["name"],
                        "batch_size":       bs,
                        "input_length":     seq_len,
                        "weight_bytes_mb":  round(weight_bytes / (1024**2), 1),
                        "param_count_m":    round(param_count / 1e6, 1),
                        **p_stats,
                    })
                    pf.flush()
                    print(f"    profiler batch={bs:2d} seq={seq_len}: "
                          f"flops={p_stats['profiler_flops']/1e9:.2f}G  "
                          f"cuda={p_stats['profiler_cuda_time_ms']:.1f}ms  "
                          f"mem={p_stats['profiler_mem_alloc_mb']:.1f}MB  "
                          f"bw_proxy={p_stats['profiler_bw_proxy_GBs']:.1f}GB/s")
                except torch.cuda.OutOfMemoryError:
                    print(f"    profiler batch={bs} seq={seq_len}: OOM — skipped")

            # ── Prefill measurements (shows compute-bound regime) ─────────────
            # Prefill AI >> GPU ridge point (≈35–300 FLOPs/byte), confirming
            # compute-bound behaviour — in contrast to decode which is memory-bound.
            print("  [prefill] measuring prompt-processing latency and AI...")
            for bs in PREFILL_BATCH_SIZES:
                for seq_len in PREFILL_INPUT_LENGTHS:
                    print(f"    prefill batch={bs:2d} seq={seq_len:4d} ...", end=" ", flush=True)
                    pr = run_prefill(model, tokenizer, bs, seq_len, device,
                                     weight_bytes, param_count)
                    if pr is None:
                        print("OOM — skipped")
                        continue
                    prwriter.writerow({
                        "model_name":    model_cfg["name"],
                        "weight_bytes_mb": round(weight_bytes / (1024**2), 1),
                        "param_count_m":   round(param_count / 1e6, 1),
                        **pr,
                    })
                    pfil.flush()
                    print(f"lat={pr['avg_latency_ms']:.1f}ms  "
                          f"AI={pr['prefill_ai_flops_byte']:.0f} FLOPs/byte  "
                          f"GPU={pr['gpu_util_pct']}%")

            # ── Decode baseline sweep ─────────────────────────────────────────
            for bs in BATCH_SIZES:
                for seq_len in INPUT_LENGTHS:
                    tag = f"[{model_cfg['name']}] batch={bs:2d}  seq={seq_len:4d}"
                    print(f"  {tag} ...", end=" ", flush=True)
                    if seq_len + MAX_NEW_TOKENS > max_pos:
                        print(f"exceeds max context ({max_pos}) — skipped")
                        continue
                    try:
                        row = run_experiment(model, tokenizer, bs, seq_len, device,
                                             weight_bytes, param_count)
                        row["model_name"] = model_cfg["name"]
                        writer.writerow(row)
                        f.flush()
                        print(f"latency={row['avg_latency_s']:.3f}s  "
                              f"thr={row['throughput_tok_s']:.1f} tok/s  "
                              f"GPU={row['gpu_util_pct']}%  "
                              f"bw={row['estimated_bandwidth_GBs']:.1f}GB/s  "
                              f"AI={row['arithmetic_intensity']:.3f}")
                    except torch.cuda.OutOfMemoryError:
                        print("OOM — skipped")

            del model
            torch.cuda.empty_cache()

    print(f"\nResults saved   → {out_path}")
    print(f"Profiler data   → {profiler_path}")
    print(f"Prefill results → {prefill_path}")
    pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()
