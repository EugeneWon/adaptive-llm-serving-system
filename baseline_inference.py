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
import os
import torch
import pynvml
from transformers import AutoModelForCausalLM, AutoTokenizer

RESULTS_DIR = "/workspace/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

MODELS = [
    {"name": "gpt2",         "model_id": "gpt2"},
    {"name": "gpt-neo-125m", "model_id": "EleutherAI/gpt-neo-125m"},
]

BATCH_SIZES    = [1, 2, 4, 8, 16, 32]
INPUT_LENGTHS  = [32, 64, 128, 256, 512]
MAX_NEW_TOKENS = 50
WARMUP_RUNS    = 10
MEASURE_RUNS   = 20

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


def make_input(tokenizer, batch_size, input_length, device):
    prompt = "The quick brown fox jumps over the lazy dog"
    ids    = tokenizer.encode(prompt, return_tensors="pt")[0]
    ids    = ids.repeat((input_length // len(ids)) + 1)[:input_length]
    input_ids      = ids.unsqueeze(0).repeat(batch_size, 1).to(device)
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask


def run_experiment(model, tokenizer, batch_size, input_length, device):
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
        "p50_latency_s":    round(sorted(latencies)[len(latencies) // 2], 4),
        "p95_latency_s":    round(sorted(latencies)[int(len(latencies) * 0.95)], 4),
        "throughput_tok_s": round(throughput, 2),
        **get_gpu_stats(),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    out_path   = os.path.join(RESULTS_DIR, "baseline_results.csv")
    fieldnames = [
        "model_name",
        "batch_size", "input_length", "max_new_tokens",
        "avg_latency_s", "p50_latency_s", "p95_latency_s",
        "throughput_tok_s",
        "gpu_util_pct", "gpu_mem_used_mb", "gpu_mem_total_mb",
    ]

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for model_cfg in MODELS:
            print(f"\n{'='*55}")
            print(f"Loading {model_cfg['name']} ({model_cfg['model_id']})...")
            tokenizer = AutoTokenizer.from_pretrained(model_cfg["model_id"])
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model   = AutoModelForCausalLM.from_pretrained(model_cfg["model_id"]).to(device)
            model.eval()
            max_pos = model.config.max_position_embeddings
            print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M  "
                  f"Max context: {max_pos}")

            for bs in BATCH_SIZES:
                for seq_len in INPUT_LENGTHS:
                    tag = f"[{model_cfg['name']}] batch={bs:2d}  seq={seq_len:4d}"
                    print(f"  {tag} ...", end=" ", flush=True)
                    if seq_len + MAX_NEW_TOKENS > max_pos:
                        print(f"exceeds max context ({max_pos}) — skipped")
                        continue
                    try:
                        row = run_experiment(model, tokenizer, bs, seq_len, device)
                        row["model_name"] = model_cfg["name"]
                        writer.writerow(row)
                        f.flush()
                        print(f"latency={row['avg_latency_s']:.3f}s  "
                              f"thr={row['throughput_tok_s']:.1f} tok/s  "
                              f"GPU={row['gpu_util_pct']}%  "
                              f"mem={row['gpu_mem_used_mb']}MB")
                    except torch.cuda.OutOfMemoryError:
                        print("OOM — skipped")

            del model
            torch.cuda.empty_cache()

    print(f"\nResults saved → {out_path}")
    pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()
