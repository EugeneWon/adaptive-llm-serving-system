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

MODELS = [
    {"name": "gpt2",         "model_id": "gpt2"},
    {"name": "gpt2-large",   "model_id": "gpt2-large"},
    {"name": "gpt-neo-125m", "model_id": "EleutherAI/gpt-neo-125m"},
]

BATCH_SIZES    = [1, 2, 4, 8, 16, 32]
INPUT_LENGTHS  = [32, 128, 512]
MAX_NEW_TOKENS  = 50
WARMUP_RUNS     = 10
COMPILE_WARMUP  = 30  # compiled graphs need extra runs to JIT-stabilize all decode shapes
MEASURE_RUNS    = 20
_T95            = 2.093  # t(0.975, df=19) for 95% CI

CONFIGS = [
    {"name": "baseline",     "fp16": False, "compile": False},
    {"name": "fp16",         "fp16": True,  "compile": False},
    {"name": "compile",      "fp16": False, "compile": True},
    {"name": "fp16+compile", "fp16": True,  "compile": True},
]

pynvml.nvmlInit()
_visible  = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
_gpu_idx  = int(_visible.split(",")[0]) if _visible.strip() else 0
gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(_gpu_idx)


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
    latency_per_token = avg_latency / MAX_NEW_TOKENS
    estimated_bw_GBs = weight_bytes / latency_per_token / 1e9
    flops_per_token = 2 * param_count * batch_size
    arithmetic_intensity = flops_per_token / weight_bytes
    return {
        "weight_bytes_mb":           round(weight_bytes / (1024 ** 2), 1),
        "estimated_bandwidth_GBs":   round(estimated_bw_GBs, 2),
        "arithmetic_intensity":      round(arithmetic_intensity, 4),
    }


def make_input(tokenizer, batch_size, input_length, device):
    prompt = "The quick brown fox jumps over the lazy dog"
    ids    = tokenizer.encode(prompt, return_tensors="pt")[0]
    ids    = ids.repeat((input_length // len(ids)) + 1)[:input_length]
    input_ids      = ids.unsqueeze(0).repeat(batch_size, 1).to(device)
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask


def run_experiment(model, tokenizer, batch_size, input_length, device, max_pos, model_cfg, opt_cfg,
                   weight_bytes, param_count):
    if input_length + MAX_NEW_TOKENS > max_pos:
        return None

    input_ids, attention_mask = make_input(tokenizer, batch_size, input_length, device)
    gen_kwargs = dict(
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        attention_mask=attention_mask,
        pad_token_id=tokenizer.eos_token_id,
    )

    n_warmup = COMPILE_WARMUP if opt_cfg["compile"] else WARMUP_RUNS
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

    avg_latency = sum(latencies) / len(latencies)
    throughput  = (MAX_NEW_TOKENS * batch_size) / avg_latency

    sorted_lat = sorted(latencies)
    return {
        "model_name":       model_cfg["name"],
        "config":           opt_cfg["name"],
        "fp16":             opt_cfg["fp16"],
        "compiled":         opt_cfg["compile"],
        "batch_size":       batch_size,
        "input_length":     input_length,
        "max_new_tokens":   MAX_NEW_TOKENS,
        "avg_latency_s":    round(avg_latency, 4),
        "p50_latency_s":    round(sorted_lat[(len(sorted_lat) - 1) // 2], 4),
        "p95_latency_s":    round(sorted_lat[int(len(sorted_lat) * 0.95) - 1], 4),
        "std_latency_s":    round(statistics.stdev(latencies), 5),
        "ci95_half_ms":     round(_T95 * statistics.stdev(latencies) * 1000 / math.sqrt(MEASURE_RUNS), 3),
        "ms_per_token":     round(avg_latency * 1000 / MAX_NEW_TOKENS, 3),
        "throughput_tok_s": round(throughput, 2),
        **get_gpu_stats(),
        **compute_data_movement_metrics(weight_bytes, param_count, avg_latency, batch_size),
    }


def load_model(model_cfg, opt_cfg, device):
    model = AutoModelForCausalLM.from_pretrained(model_cfg["model_id"])
    if opt_cfg["fp16"]:
        model = model.half()
    model = model.to(device).eval()
    if opt_cfg["compile"]:
        # reduce-overhead mode uses CUDA graphs / kernel fusion to minimise kernel launch overhead.
        # Expected benefit scales with model depth: GPT-2 Large (36 layers, 762M) shows more
        # pronounced gain than GPT-2 (12 layers, 124M) because more layer-level kernels can be fused.
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "N/A"
    print(f"Device: {device} | GPU: {gpu_name}\n")

    out_path   = os.path.join(RESULTS_DIR, "optimization_results.csv")
    fieldnames = [
        "model_name", "config", "fp16", "compiled",
        "batch_size", "input_length", "max_new_tokens",
        "avg_latency_s", "p50_latency_s", "p95_latency_s", "std_latency_s", "ci95_half_ms", "ms_per_token",
        "throughput_tok_s",
        "gpu_util_pct", "gpu_mem_used_mb", "gpu_mem_total_mb",
        "weight_bytes_mb", "estimated_bandwidth_GBs", "arithmetic_intensity",
    ]

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for model_cfg in MODELS:
            print(f"\n{'='*55}")
            print(f"Model: {model_cfg['name']} ({model_cfg['model_id']})")
            tokenizer = AutoTokenizer.from_pretrained(model_cfg["model_id"])
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            for opt_cfg in CONFIGS:
                print(f"\n  --- Config: {opt_cfg['name']} ---")
                model        = load_model(model_cfg, opt_cfg, device)
                max_pos      = model.config.max_position_embeddings
                weight_bytes = get_model_weight_bytes(model)
                param_count  = sum(p.numel() for p in model.parameters())
                print(f"    weights: {weight_bytes/(1024**2):.0f} MB  "
                      f"(dtype={'fp16' if opt_cfg['fp16'] else 'fp32'})")

                for bs in BATCH_SIZES:
                    for seq_len in INPUT_LENGTHS:
                        print(f"    batch={bs:2d}  seq={seq_len:4d} ...", end=" ", flush=True)
                        row = run_experiment(
                            model, tokenizer, bs, seq_len, device,
                            max_pos, model_cfg, opt_cfg,
                            weight_bytes, param_count,
                        )
                        if row is None:
                            print("skipped")
                            continue
                        writer.writerow(row)
                        f.flush()
                        print(f"latency={row['avg_latency_s']:.3f}s  "
                              f"thr={row['throughput_tok_s']:.1f} tok/s  "
                              f"bw={row['estimated_bandwidth_GBs']:.1f}GB/s  "
                              f"AI={row['arithmetic_intensity']:.3f}")

                del model
                torch.cuda.empty_cache()

    print(f"\nResults saved → {out_path}")
    pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()
